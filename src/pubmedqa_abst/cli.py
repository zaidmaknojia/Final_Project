"""Command-line entry point: `python -m pubmedqa_abst.cli <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from .client import ANSWER_SCHEMA, OllamaClient
from .data import build_prompt, load_config, load_items, load_split
from .evaluate import REPO_ROOT, evaluate, fit_from_dev
from .freeze import verify_locks, write_evaluation_lock, write_protocol_lock
from .runner import compute_run_id, generation_source, read_lines, run
from .scores import incomplete_keys

log = logging.getLogger("pubmedqa_abst")
CONDITIONS = ("E+", "E-")
PILOT_SEED = 20261121
DETERMINISM_ITEMS = 5


class Context:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = Path(args.root)
        self.freeze = Path(args.freeze_dir) if args.freeze_dir else self.root / "freeze"
        self.runs = Path(args.runs_dir) if args.runs_dir else self.root / "runs"
        self.results = Path(args.results_dir) if args.results_dir else self.root / "results"
        self.force = args.force
        self.config = load_config(self.root / "configs" / "run.yaml")
        self.template = (self.root / "configs" / "prompts" / "v1.txt").read_text()

    def data(self):
        raw, manifest = self.root / "data" / "raw", self.root / "data" / "manifest.json"
        return load_items(raw, manifest), load_split(raw, manifest)

    def client(self):
        c = self.config
        return OllamaClient(c["ollama_url"], c["model"], c["timeout_s"], c["max_retries"])

    def run_path(self, split: str, condition: str, config: dict | None = None, runs: Path | None = None) -> Path:
        cfg = config or self.config
        run_id = compute_run_id(cfg, self.template, generation_source(), cfg["model_digest"], split, condition)
        return (runs or self.runs) / run_id / "raw.jsonl"


def _fail(message: str) -> None:
    log.error(message)
    sys.exit(2)


def _preflight(ctx: Context, client) -> dict:
    try:
        return client.preflight(ctx.config["model_digest"])
    except RuntimeError as exc:
        _fail(str(exc))


def cmd_smoke(ctx: Context, args) -> None:
    items, split = ctx.data()
    client = ctx.client()
    env = _preflight(ctx, client)
    cfg = {**ctx.config, "K": 2}
    raw = run([items[i] for i in split["dev"][:2]], "E+", "dev", client, cfg, ctx.template,
              ctx.runs / "_smoke", env)
    for row in read_lines(raw):
        log.info("%s status=%s answer=%s confidence=%s", row["key"], row["status"], row["answer"], row["confidence"])


def stratified_ids(dev_ids: list[str], items: dict, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    groups = {label: sorted(i for i in dev_ids if items[i].gold == label) for label in ("yes", "no", "maybe")}
    for g in groups.values():
        rng.shuffle(g)
    picked, labels = [], list(groups)
    while len(picked) < n and any(groups.values()):
        for label in labels:
            if groups[label] and len(picked) < n:
                picked.append(groups[label].pop())
    return picked


def pilot_report(rows_by_cond: dict[str, list[dict]], k: int, determinism: list[bool]) -> dict:
    report = {"K": k, "conditions": {}}
    latencies = []
    for cond, rows in rows_by_cond.items():
        ok = [r for r in rows if r["status"] != "transport_error"]
        latencies += [r["latency_s"] for r in ok]
        greedy = [r for r in ok if r["call_idx"] == 0]
        samples = [r for r in ok if r["call_idx"] > 0]
        report["conditions"][cond] = {
            "calls": len(rows),
            "transport_errors": len(rows) - len(ok),
            "greedy_invalid_rate": sum(r["status"] != "ok" for r in greedy) / max(len(greedy), 1),
            "sample_invalid_rate": sum(r["status"] not in ("ok", "invalid_confidence") for r in samples)
            / max(len(samples), 1),
            "thinking_rate": sum(r.get("thinking_len", 0) > 0 for r in ok) / max(len(ok), 1),
        }
    mean_latency = statistics.mean(latencies) if latencies else float("nan")
    report["mean_seconds_per_call"] = mean_latency
    report["greedy_deterministic"] = f"{sum(determinism)}/{len(determinism)}"
    report["eta_hours_full_run"] = mean_latency * 1000 * len(CONDITIONS) * (k + 1) / 3600
    return report


def cmd_pilot(ctx: Context, args) -> None:
    items, split = ctx.data()
    client = ctx.client()
    env = _preflight(ctx, client)
    ids = stratified_ids(split["dev"], items, args.n, PILOT_SEED)
    pilot_runs = ctx.runs / "_pilot"
    rows_by_cond = {}
    for cond in CONDITIONS:
        raw = run([items[i] for i in ids], cond, "dev", client, ctx.config, ctx.template, pilot_runs, env)
        rows_by_cond[cond] = read_lines(raw)
    greedy_text = {r["item_id"]: r["text"] for r in rows_by_cond["E+"] if r["call_idx"] == 0}
    determinism = []
    for iid in ids[:DETERMINISM_ITEMS]:
        again = client.generate(build_prompt(items[iid], "E+", ctx.template), 0, ctx.config["greedy_seed"],
                                ANSWER_SCHEMA)
        determinism.append(again.text == greedy_text.get(iid))
    report = pilot_report(rows_by_cond, ctx.config["K"], determinism)
    report["env"] = env
    (pilot_runs / "pilot_report.json").write_text(json.dumps(report, indent=2))
    log.info("pilot report:\n%s", json.dumps(report, indent=2))


def test_runs_exist(runs_dir: Path) -> bool:
    for manifest in Path(runs_dir).glob("*/manifest.json"):
        try:
            if json.loads(manifest.read_text()).get("split") == "test":
                return True
        except json.JSONDecodeError:
            continue
    return False


def guard_freeze(ctx: Context, command: str, target: str) -> None:
    """Locks and dev fits are write-once: refuse to overwrite them, and refuse
    to touch them at all once a test run exists, unless --force (logged)."""
    problems = []
    if test_runs_exist(ctx.runs):
        problems.append("a test run already exists")
    if (ctx.freeze / target).exists():
        problems.append(f"{target} already exists")
    if not problems:
        return
    if not ctx.force:
        _fail(f"refusing `{command}`: {'; '.join(problems)}. Use --force only with a reason you will disclose.")
    ctx.freeze.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "command": command, "problems": problems}
    with open(ctx.freeze / "overrides.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    log.warning("--force: overriding (%s); logged to overrides.jsonl", "; ".join(problems))


def cmd_lock_protocol(ctx: Context, args) -> None:
    guard_freeze(ctx, "lock-protocol", "protocol.lock.json")
    log.info("wrote %s", write_protocol_lock(ctx.root, ctx.freeze, ctx.config))


def cmd_run(ctx: Context, args) -> None:
    if args.split == "test":
        try:
            verify_locks(ctx.root, ctx.freeze)
        except RuntimeError as exc:
            _fail(f"refusing to run test split: {exc}")
    items, split = ctx.data()
    client = ctx.client()
    env = _preflight(ctx, client)
    ids = split[args.split][: args.limit] if args.limit else split[args.split]
    try:
        raw = run([items[i] for i in ids], args.condition, args.split, client, ctx.config,
                  ctx.template, ctx.runs, env)
    except RuntimeError as exc:
        _fail(str(exc))
    missing = incomplete_keys(raw, ids, ctx.config["K"])
    if missing:
        _fail(f"{len(missing)} calls still missing or failed in {raw} (first: {missing[:3]}); "
              "rerun the same command to resume")
    log.info("complete: %s (%d items x %d calls)", raw, len(ids), ctx.config["K"] + 1)


def _runs_for(ctx: Context, split: str) -> dict[str, Path]:
    paths = {c: ctx.run_path(split, c) for c in CONDITIONS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        _fail(f"missing run logs: {missing}")
    return paths


def cmd_fit(ctx: Context, args) -> None:
    guard_freeze(ctx, "fit", "thresholds.json")
    items, split = ctx.data()
    try:
        payload = fit_from_dev(_runs_for(ctx, "dev"), items, ctx.config, ctx.freeze, split["dev"])
    except RuntimeError as exc:
        _fail(str(exc))
    log.info("dev confident errors for scheme review: %s", ctx.freeze / "dev_confident_errors.csv")
    log.info("thresholds (primary):\n%s", json.dumps(payload["primary"], indent=2))


def cmd_lock_evaluation(ctx: Context, args) -> None:
    guard_freeze(ctx, "lock-evaluation", "evaluation.lock.json")
    log.info("wrote %s", write_evaluation_lock(ctx.root, ctx.freeze))


def cmd_evaluate(ctx: Context, args) -> None:
    items, split = ctx.data()
    try:
        evaluate(_runs_for(ctx, "dev"), _runs_for(ctx, "test"), items, ctx.config, ctx.freeze,
                 ctx.results, {"dev": split["dev"], "test": split["test"]}, root=ctx.root)
    except RuntimeError as exc:
        _fail(str(exc))
    log.info("results written to %s", ctx.results)


def cmd_figures(ctx: Context, args) -> None:
    from .figures import make_figures

    for path in make_figures(ctx.results):
        log.info("wrote %s", path)


COMMANDS = {
    "smoke": cmd_smoke, "pilot": cmd_pilot, "lock-protocol": cmd_lock_protocol, "run": cmd_run,
    "fit": cmd_fit, "lock-evaluation": cmd_lock_evaluation, "evaluate": cmd_evaluate, "figures": cmd_figures,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pubmedqa_abst")
    p.add_argument("--root", default=str(REPO_ROOT))
    p.add_argument("--freeze-dir")
    p.add_argument("--runs-dir")
    p.add_argument("--results-dir")
    p.add_argument("--force", action="store_true", help="override a write-once guard (logged)")
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        if name == "pilot":
            sp.add_argument("--n", type=int, default=20)
        if name == "run":
            sp.add_argument("--split", choices=("dev", "test"), required=True)
            sp.add_argument("--condition", choices=CONDITIONS, required=True)
            sp.add_argument("--limit", type=int)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    COMMANDS[args.command](Context(args), args)


if __name__ == "__main__":
    main()
