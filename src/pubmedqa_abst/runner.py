"""Resumable, append-only inference runner.

One raw line per model call. A key (`<item_id>|<call_idx>`) is complete when
its latest line has a status other than `transport_error`. `raw.jsonl` is
never rewritten; everything downstream is re-derived from it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from . import client as client_mod
from . import data as data_mod
from . import parse as parse_mod
from .client import ANSWER_SCHEMA
from .data import Item, build_prompt
from .parse import parse_output

log = logging.getLogger(__name__)

TRANSPORT_ERROR = "transport_error"
# Only these config keys change what the model is asked; analysis settings
# (bootstrap, thresholds, bins) must not force a re-run.
GENERATION_KEYS = ("model", "model_digest", "K", "T", "greedy_seed")
PROGRESS_EVERY = 25
MAX_CONSECUTIVE_ERRORS = 5


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generation_source() -> str:
    """Source of every module that shapes a request: prompt building, request
    options and schema, sampling seeds, and parsing. Hashed into run_id."""
    modules = (data_mod, client_mod, parse_mod)
    parts = [Path(m.__file__).read_text() for m in modules]
    parts.append(Path(__file__).read_text())
    return "\n".join(parts)


def compute_run_id(
    config: dict, prompt_text: str, parser_src: str, model_digest: str, split: str, condition: str
) -> str:
    identity = {
        "generation": {k: config.get(k) for k in GENERATION_KEYS},
        "prompt": _sha256_text(prompt_text),
        "parser": _sha256_text(parser_src),
        "model_digest": model_digest,
        "split": split,
        "condition": condition,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical)[:12]


def sample_seed(run_id: str, item_id: str, condition: str, k: int) -> int:
    return int(_sha256_text(f"{run_id}|{item_id}|{condition}|{k}"), 16) % 2**31


def read_lines(raw_path: Path) -> list[dict]:
    """Parse raw.jsonl, skipping unparseable lines with a warning.

    The only way such a line appears is a write cut off by power loss; the
    file is append-only, so the fragment stays in place (followed by the
    newline added on resume) and its key is simply re-run.
    """
    if not Path(raw_path).exists():
        return []
    rows, skipped = [], 0
    with open(raw_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        log.warning("skipped %d partial line(s) in %s; affected keys will be re-run", skipped, raw_path)
    return rows


def _ensure_trailing_newline(raw_path: Path) -> None:
    if raw_path.exists() and raw_path.stat().st_size:
        with open(raw_path, "rb") as fh:
            fh.seek(-1, 2)
            last = fh.read(1)
        if last != b"\n":
            with open(raw_path, "a", encoding="utf-8") as fh:
                fh.write("\n")


def latest_by_key(raw_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in read_lines(raw_path):
        latest[row["key"]] = row
    return latest


def completed_keys(raw_path: Path) -> set[str]:
    return {k for k, row in latest_by_key(raw_path).items() if row["status"] != TRANSPORT_ERROR}


def _write_manifest(run_dir: Path, run_id: str, config: dict, split: str, condition: str, env: dict) -> None:
    path = run_dir / "manifest.json"
    if path.exists():
        return
    manifest = {
        "run_id": run_id,
        "split": split,
        "condition": condition,
        "env": env,
        "model": config["model"],
        "K": config["K"],
        "T": config["T"],
        "greedy_seed": config["greedy_seed"],
        "options": {"stream": False, "think": False, "format": ANSWER_SCHEMA},
        "started": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(manifest, indent=2))


def _call_row(item: Item, condition: str, call_idx: int, temperature: float, seed: int,
              prompt: str, result) -> dict:
    parsed = parse_output(result.text) if result.error is None else None
    return {
        "key": f"{item.id}|{call_idx}",
        "item_id": item.id,
        "condition": condition,
        "call_idx": call_idx,
        "temperature": temperature,
        "seed": seed,
        "prompt_sha256": _sha256_text(prompt),
        "text": result.text,
        "thinking_len": result.thinking_len,
        "latency_s": round(result.latency_s, 4),
        "attempts": result.attempts,
        "error": result.error,
        "status": TRANSPORT_ERROR if parsed is None else parsed.status,
        "answer": None if parsed is None else parsed.answer,
        "confidence": None if parsed is None else parsed.confidence,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def run(items: list[Item], condition: str, split: str, client, config: dict, template: str,
        runs_dir: Path, env: dict, limit: int | None = None) -> Path:
    """Run greedy + K sampled calls per item, skipping keys already complete."""
    run_id = compute_run_id(config, template, generation_source(), config["model_digest"], split, condition)
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(run_dir, run_id, config, split, condition, env)
    raw_path = run_dir / "raw.jsonl"
    _ensure_trailing_newline(raw_path)
    done = completed_keys(raw_path)
    todo_items = items[:limit] if limit is not None else items
    started, n_calls, consecutive_errors = time.monotonic(), 0, 0
    with open(raw_path, "a", encoding="utf-8") as fh:
        for i, item in enumerate(todo_items, 1):
            prompt = build_prompt(item, condition, template)
            for call_idx in range(config["K"] + 1):
                if f"{item.id}|{call_idx}" in done:
                    continue
                if call_idx == 0:
                    temperature, seed = 0, config["greedy_seed"]
                else:
                    temperature, seed = config["T"], sample_seed(run_id, item.id, condition, call_idx)
                result = client.generate(prompt, temperature, seed, ANSWER_SCHEMA)
                row = _call_row(item, condition, call_idx, temperature, seed, prompt, result)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_calls += 1
                consecutive_errors = consecutive_errors + 1 if row["status"] == TRANSPORT_ERROR else 0
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"aborting after {consecutive_errors} consecutive transport errors; "
                        "check Ollama, then rerun the same command to resume")
            if i % PROGRESS_EVERY == 0 and n_calls:
                _log_progress(i, len(todo_items), n_calls, started, config["K"])
    return raw_path


def _log_progress(done_items: int, total_items: int, n_calls: int, started: float, k: int) -> None:
    elapsed = time.monotonic() - started
    per_min = n_calls / elapsed * 60 if elapsed else 0.0
    remaining_calls = (total_items - done_items) * (k + 1)
    eta_h = remaining_calls / per_min / 60 if per_min else float("nan")
    log.info("%d/%d items, %.1f calls/min, ETA %.1f h", done_items, total_items, per_min, eta_h)
