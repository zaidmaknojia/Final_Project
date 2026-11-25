"""Post-hoc, exploratory analyses added after the locked evaluation (Nov 23 review).

None of this was in the evaluation lock, so the reports label it exploratory.
It reads the same raw runs and frozen thresholds and never writes to a locked
file. It answers one question the locked analysis left open: how much of stated
confidence's ranking power comes from the answer label alone (committing to
yes/no versus hedging with "maybe")? When the token-probability runs exist
(tokenprob.py, new calls made after the test run), it also sets that baseline
beside the locked scores under the "token" key.

Run: python -m pubmedqa_abst.posthoc [results_dir]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .bootstrap import paired_bootstrap
from .metrics import aurc, auroc, brier, complete_curve, ece, risk_coverage
from .scores import ItemScore
from .thresholds import fit_threshold

COMMITTED = ("yes", "no")
CONDITIONS = ("E+", "E-")
_EPS = 1e-9


def label_score(s: ItemScore) -> float:
    """1.0 when the model committed to yes/no, else 0.0 (maybe or invalid)."""
    return 1.0 if s.a0 in COMMITTED else 0.0


def _group(scores: list[ItemScore], n_total: int) -> dict:
    n = len(scores)
    confs = [s.s_verb for s in scores if s.s_verb is not None]
    return {
        "n": n,
        "share": n / n_total if n_total else 0.0,
        "accuracy": float(np.mean([s.correct for s in scores])) if n else None,
        "mean_conf": float(np.mean(confs)) if confs else None,
    }


def answer_mix(scores: list[ItemScore]) -> dict:
    """Accuracy and mean stated confidence for committed vs "maybe" answers."""
    n = len(scores)
    return {
        "committed": _group([s for s in scores if s.a0 in COMMITTED], n),
        "maybe": _group([s for s in scores if s.a0 == "maybe"], n),
    }


def gold_maybe_outcome(scores: list[ItemScore]) -> dict:
    """What the model said on questions whose expert answer is maybe."""
    gm = [s for s in scores if s.gold == "maybe"]
    yes_no = [s for s in gm if s.a0 in COMMITTED]
    return {"n": len(gm), "said_maybe": sum(s.a0 == "maybe" for s in gm),
            "said_yes_no": len(yes_no), "correct_yes_no": sum(s.correct for s in yes_no)}


def rate_by_gold(scores: list[ItemScore], score_name: str, threshold: float) -> dict[str, float]:
    """Confident errors (accepted and wrong) per question of each gold label."""
    out = {}
    for label in ("yes", "no", "maybe"):
        group = [s for s in scores if s.gold == label]
        errors = sum(1 for s in group if (v := getattr(s, score_name)) is not None
                     and v >= threshold - _EPS and not s.correct)
        out[label] = errors / len(group) if group else 0.0
    return out


def reference_aurc(correct: np.ndarray) -> dict[str, float]:
    """AURC of a random ranking (the error rate) and of a perfect one (errors last)."""
    correct = np.asarray(correct, int)
    n, n_c = len(correct), int(correct.sum())
    oracle = sum(max(0, k - n_c) / k for k in range(1, n + 1)) / n
    return {"random": float(1 - correct.mean()), "oracle": float(oracle)}


def label_entropy_auroc(samples: dict[str, list[str]], correct: dict[str, bool], ids: list[str]) -> float | None:
    """AUROC of negative label entropy over the sampled answers (Kuhn et al.'s
    semantic entropy when each label is its own meaning cluster)."""
    def neg_entropy(answers: list[str]) -> float:
        p = np.array(list(Counter(answers).values()), float) / len(answers)
        return float((p * np.log(p)).sum())

    return auroc(np.array([neg_entropy(samples[i]) for i in ids]), np.array([correct[i] for i in ids]))


def error_profile(rows: list[dict], cond: str) -> dict:
    """Confident errors under stated confidence for one condition, by family."""
    mine = [r for r in rows if r["condition"] == cond and r["score"] == "s_verb"]
    over = [r for r in mine if r["confusion"].startswith("maybe->") and not r["confusion"].endswith("->maybe")]
    hedge = [r for r in mine if r["confusion"].endswith("->maybe")]
    return {
        "total": len(mine),
        "on_gold_maybe": sum(r["confusion"].startswith("maybe->") for r in mine),
        "overcommit": len(over),
        "overcommit_at_095": sum(float(r["s_verb"]) >= 0.95 - _EPS for r in over),
        "overcommit_unanimous": sum(float(r["s_cons"]) >= 1.0 - _EPS for r in over),
        "hedge": len(hedge),
        "hedge_at_085": sum(abs(float(r["s_verb"]) - 0.85) < _EPS for r in hedge),
    }


def _arr(scores: list[ItemScore], name: str) -> np.ndarray:
    return np.array([getattr(s, name) if getattr(s, name) is not None else -1.0 for s in scores])


def _bootstrap(scores: list[ItemScore], B: int, seed: int) -> dict:
    correct = np.array([s.correct for s in scores], int)
    verb, cons = _arr(scores, "s_verb"), _arr(scores, "s_cons")
    label = np.array([label_score(s) for s in scores])
    committed = label == 1.0

    def within(conf: np.ndarray):
        return lambda idx: auroc(conf[idx][committed[idx]], correct[idx][committed[idx]])

    def whole(conf: np.ndarray):
        return lambda idx: auroc(conf[idx], correct[idx])

    def minus(a, b):
        return lambda idx: None if (x := a(idx)) is None or (y := b(idx)) is None else x - y

    fns = {
        "label_minus_verb_auroc": minus(whole(label), whole(verb)),
        "within_verb_auroc": within(verb),
        "within_cons_auroc": within(cons),
        "within_cons_minus_verb_auroc": minus(within(cons), within(verb)),
    }
    return paired_bootstrap([s.item_id for s in scores], fns, B, seed)


def _condition(scores: list[ItemScore], threshold: float, error_rows: list[dict], cond: str,
               samples: dict[str, list[str]] | None) -> dict:
    correct = np.array([s.correct for s in scores], int)
    committed = [s for s in scores if s.a0 in COMMITTED]
    c_correct = np.array([s.correct for s in committed], int)
    gold_counts = Counter(s.gold for s in scores)
    majority = gold_counts.most_common(1)[0][0]
    out = {
        "n": len(scores),
        "auroc": {"s_verb": auroc(_arr(scores, "s_verb"), correct), "s_cons": auroc(_arr(scores, "s_cons"), correct),
                  "label": auroc(np.array([label_score(s) for s in scores]), correct)},
        "within_committed": {"n": len(committed),
                             "auroc_s_verb": auroc(_arr(committed, "s_verb"), c_correct),
                             "auroc_s_cons": auroc(_arr(committed, "s_cons"), c_correct)},
        "answer_mix": answer_mix(scores),
        "gold_maybe": gold_maybe_outcome(scores),
        "rate_by_gold_s_verb_0.7": rate_by_gold(scores, "s_verb", threshold),
        "reference_aurc": reference_aurc(correct),
        "majority": {"label": majority, "accuracy": gold_counts[majority] / len(scores)},
        "errors": error_profile(error_rows, cond),
    }
    if samples is not None:
        ids = [s.item_id for s in scores]
        out["auroc"]["label_entropy"] = label_entropy_auroc(samples, {s.item_id: s.correct for s in scores}, ids)
    return out


def posthoc(test_scores: dict[str, list[ItemScore]], thresholds: dict, error_rows: list[dict],
            samples: dict[str, dict[str, list[str]]] | None, B: int, seed: int) -> dict:
    threshold = thresholds["primary"]["s_verb"]["0.7"]["threshold"]
    out: dict = {"note": "Post-hoc and exploratory: not part of the evaluation lock.", "bootstrap": {}}
    for cond in CONDITIONS:
        scores = test_scores[cond]
        out[cond] = _condition(scores, threshold, error_rows, cond, samples[cond] if samples else None)
        out["bootstrap"][cond] = _bootstrap(scores, B, seed)
    return out


def fast_aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Step AURC over every item with ties kept together; equals metrics.aurc of the
    full risk-coverage curve, in O(n log n) so the bootstrap stays fast."""
    conf, wrong = np.asarray(conf, float), 1.0 - np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    c, cum_wrong = conf[order], np.cumsum(wrong[order])
    ends = np.r_[np.nonzero(np.diff(c))[0], len(c) - 1]  # last index of each tie group
    counts = ends + 1
    coverage = counts / len(c)
    return float(np.sum(np.diff(np.r_[0.0, coverage]) * cum_wrong[ends] / counts))


def score_summary(scores: list[ItemScore], conf: np.ndarray, threshold: float | None, n_bins: int = 10) -> dict:
    """Ranking, calibration and the accept-above-threshold policy for one score.
    NaN marks an unscored item: left out of AUROC/ECE/Brier, answered last for AURC."""
    correct = np.array([s.correct for s in scores], int)
    conf = np.asarray(conf, float)
    mask = ~np.isnan(conf)
    c, y, n = conf[mask], correct[mask], len(scores)
    filled = np.where(mask, conf, -1.0)
    committed = np.array([s.a0 in COMMITTED for s in scores])
    curve = complete_curve(risk_coverage(c, y, n), full_risk=1.0 - float(correct.mean()))
    out = {"n_scored": int(mask.sum()), "mean": float(c.mean()) if c.size else None,
           "auroc": auroc(c, y), "within_committed_auroc": auroc(filled[committed], correct[committed]),
           "ece": ece(c, y, n_bins)[0], "brier": brier(c, y), "aurc": aurc(curve)}
    if threshold is not None:
        accepted = filled >= threshold - _EPS
        out["policy"] = {"threshold": threshold, "coverage": float(accepted.mean()),
                         "accuracy": float(correct[accepted].mean()) if accepted.any() else None,
                         "confident_error_rate": float((accepted & (correct == 0)).sum() / n)}
    return out


def label_split(scores: list[ItemScore], conf: np.ndarray, threshold: float) -> dict:
    """Mean score and share accepted at `threshold`, for committed (yes/no) and "maybe" answers."""
    conf = np.asarray(conf, float)
    out = {}
    for kind, keep in (("committed", lambda a: a in COMMITTED), ("maybe", lambda a: a == "maybe")):
        c = conf[np.array([keep(s.a0) for s in scores], bool)]
        out[kind] = {"n": int(c.size), "mean": float(np.nanmean(c)) if c.size else None,
                     "accepted_share": float((c >= threshold - _EPS).mean()) if c.size else None}
    return out


def _token_bootstrap(scores: list[ItemScore], tok: np.ndarray, B: int, seed: int) -> dict:
    correct = np.array([s.correct for s in scores], int)
    t, verb = np.where(np.isnan(tok), -1.0, tok), _arr(scores, "s_verb")
    committed = np.array([s.a0 in COMMITTED for s in scores])

    def whole(conf):
        return lambda idx: auroc(conf[idx], correct[idx])

    def within(conf):
        return lambda idx: auroc(conf[idx][committed[idx]], correct[idx][committed[idx]])

    def minus(a, b):
        return lambda idx: None if (x := a(idx)) is None or (y := b(idx)) is None else x - y

    fns = {
        "tok_minus_verb_auroc": minus(whole(t), whole(verb)),
        "within_tok_auroc": within(t),
        "within_tok_minus_verb_auroc": minus(within(t), within(verb)),
        "tok_minus_verb_aurc": lambda idx: fast_aurc(t[idx], correct[idx]) - fast_aurc(verb[idx], correct[idx]),
    }
    return paired_bootstrap([s.item_id for s in scores], fns, B, seed)


def token_posthoc(test_scores: dict[str, list[ItemScore]], test_tok: dict[str, np.ndarray], dev_tok: np.ndarray,
                  n_dev: int, thresholds: dict, B: int, seed: int, target: str = "0.7") -> dict:
    """Token-probability baseline beside the locked scores. Its threshold is fitted on
    E+ dev at `target` coverage and applied to both test conditions (the primary policy)."""
    fit = fit_threshold(dev_tok[~np.isnan(dev_tok)], n_dev, float(target))
    locked = {name: thresholds["primary"][name][target]["threshold"] for name in ("s_verb", "s_cons")}
    out: dict = {"note": "Post-hoc and exploratory: new calls made after the test run.",
                 "target": target, "threshold": fit, "bootstrap": {}}
    for cond in CONDITIONS:
        scores = test_scores[cond]
        out[cond] = {"s_tok": score_summary(scores, test_tok[cond], fit["threshold"]),
                     **{name: score_summary(scores, _nan_arr(scores, name), t) for name, t in locked.items()},
                     "s_tok_by_label": label_split(scores, test_tok[cond], fit["threshold"])}
        out["bootstrap"][cond] = _token_bootstrap(scores, test_tok[cond], B, seed)
    return out


def _nan_arr(scores: list[ItemScore], name: str) -> np.ndarray:
    return np.array([np.nan if getattr(s, name) is None else getattr(s, name) for s in scores], float)


def write_posthoc(payload: dict, results_dir: Path) -> Path:
    path = Path(results_dir) / "posthoc.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _sampled_answers(raw_path: Path, K: int) -> dict[str, list[str]]:
    from .runner import latest_by_key

    out: dict[str, list[str]] = {}
    for row in latest_by_key(raw_path).values():
        if 1 <= row["call_idx"] <= K and row.get("answer") is not None:
            out.setdefault(row["item_id"], []).append(row["answer"])
    return out


def main(results_dir: Path) -> Path:  # pragma: no cover - thin wiring over tested functions
    import csv

    from .cli import CONDITIONS as CLI_CONDITIONS, Context, build_parser
    from .evaluate import load_scores

    ctx = Context(build_parser().parse_args(["--results-dir", str(results_dir), "evaluate"]))
    items, split = ctx.data()
    runs = {c: ctx.run_path("test", c) for c in CLI_CONDITIONS}
    K = ctx.config["K"]
    test = load_scores(runs, items, split["test"], K)
    thresholds = json.loads((ctx.freeze / "thresholds.json").read_text())
    with open(Path(results_dir) / "confident_errors.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    samples = {c: _sampled_answers(runs[c], K) for c in CLI_CONDITIONS}
    B, seed = ctx.config["bootstrap_B"], ctx.config["bootstrap_seed"]
    payload = posthoc(test, thresholds, rows, samples, B, seed)
    token = _token_payload(ctx, items, split, test, thresholds, K, B, seed)
    if token is not None:
        payload["token"] = token
    return write_posthoc(payload, results_dir)


def _token_payload(ctx, items, split, test, thresholds, K, B, seed) -> dict | None:  # pragma: no cover
    from .cli import CONDITIONS as CLI_CONDITIONS
    from .evaluate import load_scores
    from .tokenprob import agreement, load_rows, out_path, token_scores

    paths = {(s, c): out_path(ctx.runs, s, c) for s in ("dev", "test") for c in CLI_CONDITIONS}
    if not all(p.exists() for p in paths.values()):
        return None
    dev = load_scores({c: ctx.run_path("dev", c) for c in CLI_CONDITIONS}, items, split["dev"], K)
    scores = {"dev": dev, "test": test}
    rows = {key: load_rows(p) for key, p in paths.items()}
    tok = {key: token_scores(scores[key[0]][key[1]], r) for key, r in rows.items()}
    out = token_posthoc(test, {c: tok[("test", c)] for c in CLI_CONDITIONS}, tok[("dev", "E+")],
                        len(dev["E+"]), thresholds, B, seed)
    out["agreement"] = {f"{s}|{c}": agreement(scores[s][c], rows[(s, c)]) for s, c in paths}
    return out


if __name__ == "__main__":  # pragma: no cover
    root = Path(__file__).resolve().parents[2]
    print(main(Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results"))
