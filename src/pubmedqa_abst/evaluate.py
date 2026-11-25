"""Dev fitting and the single locked test evaluation (spec §7–10)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from .bootstrap import paired_bootstrap
from .data import Item
from .error_analysis import confident_errors, sample_examples
from .freeze import verify_locks
from .metrics import (
    accuracy, aurc, auroc, brier, complete_curve, confusion, ece, macro_f1, per_class, risk_coverage,
    selective,
)
from .recalibrate import IsotonicMap, apply_isotonic, fit_isotonic, load_isotonic, save_isotonic
from .scores import ItemScore, require_complete, score_run
from .stats import build_comparisons
from .thresholds import fit_all, fit_threshold

REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("E+", "E-")
SCORE_NAMES = ("s_verb", "s_cons", "s_verb_iso")
ERROR_SCORES = ("s_verb", "s_cons")
EXAMPLES_PER_CATEGORY = 3


def with_iso(scores: list[ItemScore], iso: IsotonicMap) -> list[ItemScore]:
    """Copies whose s_verb is replaced by the isotonic-recalibrated value."""
    out = []
    for s in scores:
        value = None if s.s_verb is None else float(apply_isotonic(iso, np.array([s.s_verb]))[0])
        out.append(replace(s, s_verb=value))
    return out


def score_arrays(scores: list[ItemScore], name: str, iso: IsotonicMap | None) -> tuple[np.ndarray, np.ndarray]:
    """(conf with NaN where the item abstains, correct as bool), aligned with `scores`."""
    source = with_iso(scores, iso) if name == "s_verb_iso" else scores
    attr = "s_verb" if name == "s_verb_iso" else name
    conf = np.array([getattr(s, attr) if s.valid and getattr(s, attr) is not None else np.nan
                     for s in source], float)
    return conf, np.array([s.correct for s in scores], bool)


def _fit_block(scores: list[ItemScore], iso: IsotonicMap, targets: list[float]) -> dict:
    block = fit_all(scores, targets)
    conf, _ = score_arrays(scores, "s_verb_iso", iso)
    valid = conf[~np.isnan(conf)]
    block["s_verb_iso"] = {str(t): fit_threshold(valid, len(scores), t) for t in targets}
    return block


def load_scores(runs: dict[str, Path], items: dict[str, Item], expected_ids: list[str], K: int) -> dict:
    """Score each condition's run after verifying every expected call completed."""
    wanted = set(expected_ids)
    out = {}
    for cond in CONDITIONS:
        require_complete(runs[cond], expected_ids, K)
        out[cond] = [s for s in score_run(runs[cond], items, K) if s.item_id in wanted]
    return out


def fit_from_dev(dev_runs: dict[str, Path], items: dict[str, Item], config: dict, freeze_dir: Path,
                 expected_ids: list[str]) -> dict:
    """Fit isotonic on E+ dev, thresholds on E+ dev (primary) and per condition (secondary).

    Also writes dev confident errors so the error scheme can be reviewed on dev
    before the evaluation lock (spec §10).
    """
    freeze_dir = Path(freeze_dir)
    freeze_dir.mkdir(parents=True, exist_ok=True)
    dev = load_scores(dev_runs, items, expected_ids, config["K"])
    conf, correct = score_arrays(dev["E+"], "s_verb", None)
    mask = ~np.isnan(conf)
    iso = fit_isotonic(conf[mask], correct[mask])
    save_isotonic(iso, freeze_dir / "isotonic.json")
    targets = config["coverage_targets"]
    payload = {
        "primary": _fit_block(dev["E+"], iso, targets),
        "secondary": {c: _fit_block(dev[c], iso, targets) for c in CONDITIONS},
    }
    (freeze_dir / "thresholds.json").write_text(json.dumps(payload, indent=2))
    target = str(config["error_threshold_target"])
    _write_csv(freeze_dir / "dev_confident_errors.csv", _error_rows(dev, items, payload, target))
    return payload


def _policy(conf: np.ndarray, correct: np.ndarray, threshold: float | None, n_total: int) -> dict:
    if threshold is None:
        return {"coverage": 0.0, "accuracy": None, "n_accepted": 0, "confident_error_rate": 0.0}
    sel = selective(conf, correct, threshold, n_total)
    wrong_accepted = int(((conf >= threshold - 1e-9) & ~correct.astype(bool)).sum())
    return {**sel, "confident_error_rate": wrong_accepted / n_total}


def _accept_all(correct_all: np.ndarray) -> dict:
    """Answer every item (coverage 1.0); invalid answers count as wrong (spec §6)."""
    acc = float(correct_all.mean())
    return {"coverage": 1.0, "accuracy": acc, "n_accepted": int(correct_all.size),
            "confident_error_rate": 1.0 - acc}


def _score_block(scores, name, iso, primary, secondary, n_bins) -> dict:
    conf_all, correct_all = score_arrays(scores, name, iso)
    mask = ~np.isnan(conf_all)
    c, y, n_total = conf_all[mask], correct_all[mask], len(scores)
    e, bins = ece(c, y, n_bins) if c.size else (None, [])
    curve = complete_curve(risk_coverage(c, y, n_total), full_risk=1.0 - float(correct_all.mean()))
    class_scores = with_iso(scores, iso) if name == "s_verb_iso" else scores
    class_name = "s_verb" if name == "s_verb_iso" else name
    block = {
        "n_scored": int(mask.sum()), "missing_rate": float(1 - mask.mean()),
        "brier": brier(c, y) if c.size else None, "ece": e, "ece_bins": bins,
        "auroc": auroc(c, y) if c.size else None, "curve": curve, "aurc": aurc(curve),
        "accept_all": _accept_all(correct_all),
        "accept_all_valid": _policy(c, y, -np.inf, n_total),  # secondary: scored items only
        "primary": {}, "secondary": {},
    }
    for target, fit in primary[name].items():
        t = fit["threshold"]
        entry = {**_policy(c, y, t, n_total), "threshold": t, "dev_coverage": fit["dev_coverage"]}
        entry["per_class_gold"] = per_class(class_scores, class_name, t) if t is not None else {}
        entry["per_class_pred"] = per_class(class_scores, class_name, t, by="pred") if t is not None else {}
        block["primary"][target] = entry
    for target, fit in secondary[name].items():
        block["secondary"][target] = {**_policy(c, y, fit["threshold"], n_total),
                                      "threshold": fit["threshold"], "dev_coverage": fit["dev_coverage"]}
    return block


def _full_block(scores: list[ItemScore]) -> dict:
    pred, gold = [s.a0 for s in scores], [s.gold for s in scores]
    return {
        "n": len(scores), "accuracy": accuracy(pred, gold), "macro_f1": macro_f1(pred, gold),
        "confusion": confusion(pred, gold),
        "pred_dist": dict(Counter(p if p else "invalid" for p in pred)), "gold_dist": dict(Counter(gold)),
        "invalid_rate": float(np.mean([not s.valid for s in scores])),
        "mean_invalid_samples": float(np.mean([s.invalid_samples for s in scores])),
    }


def split_metrics(scores_by_cond: dict[str, list[ItemScore]], iso, thresholds, n_bins) -> dict:
    out = {}
    for cond, scores in scores_by_cond.items():
        out[cond] = {
            "full": _full_block(scores),
            "scores": {name: _score_block(scores, name, iso, thresholds["primary"],
                                          thresholds["secondary"][cond], n_bins) for name in SCORE_NAMES},
        }
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table_rows(test: dict) -> list[dict]:
    rows = []
    for cond, block in test.items():
        for name, sb in block["scores"].items():
            common = {"condition": cond, "score": name, "brier": sb["brier"], "ece": sb["ece"],
                      "auroc": sb["auroc"], "aurc": sb["aurc"]}
            policies = [("accept_all", None, sb["accept_all"])]
            policies += [(f"primary@{t}", e["threshold"], e) for t, e in sb["primary"].items()]
            policies += [(f"secondary@{t}", e["threshold"], e) for t, e in sb["secondary"].items()]
            for policy, thr, e in policies:
                rows.append({**common, "policy": policy, "threshold": thr, "coverage": e["coverage"],
                             "accuracy": e["accuracy"], "confident_error_rate": e["confident_error_rate"]})
    return rows


def _per_class_rows(test: dict) -> list[dict]:
    rows = []
    for cond, block in test.items():
        for name, sb in block["scores"].items():
            for target, e in sb["primary"].items():
                for by in ("gold", "pred"):
                    for label, r in e[f"per_class_{by}"].items():
                        rows.append({"condition": cond, "score": name, "target": target, "by": by,
                                     "label": label, **r})
    return rows


def _error_rows(test_scores, items, thresholds, target: str) -> list[dict]:
    rows = []
    for cond, scores in test_scores.items():
        thr = {n: thresholds["primary"][n][target]["threshold"] for n in ERROR_SCORES}
        for name in ERROR_SCORES:
            rows.extend(confident_errors(scores, items, thr, name))
    return rows


def evaluate(dev_runs: dict[str, Path], test_runs: dict[str, Path], items: dict[str, Item], config: dict,
             freeze_dir: Path, out_dir: Path, expected: dict[str, list[str]], root: Path = REPO_ROOT) -> dict:
    verify_locks(root, freeze_dir)
    freeze_dir, out_dir = Path(freeze_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads((freeze_dir / "thresholds.json").read_text())
    iso = load_isotonic(freeze_dir / "isotonic.json")
    K, n_bins = config["K"], config["ece_bins"]
    dev = load_scores(dev_runs, items, expected["dev"], K)
    test = load_scores(test_runs, items, expected["test"], K)
    target = str(config["error_threshold_target"])
    errors = _error_rows(test, items, thresholds, target)
    metrics = {
        "dev": split_metrics(dev, iso, thresholds, n_bins),
        "test": split_metrics(test, iso, thresholds, n_bins),
        "thresholds": thresholds,
        "confident_error_counts": dict(Counter(f"{r['condition']}|{r['score']}|{r['confusion']}" for r in errors)),
    }
    comparisons = build_comparisons(test, iso, thresholds, n_bins)
    ids = sorted(set(s.item_id for s in test["E+"]) & set(s.item_id for s in test["E-"]))
    boot = {group: paired_bootstrap(ids, fns, config["bootstrap_B"], config["bootstrap_seed"])
            for group, fns in comparisons.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "bootstrap.json").write_text(json.dumps(boot, indent=2))
    _write_csv(out_dir / "tables.csv", _table_rows(metrics["test"]))
    _write_csv(out_dir / "per_class.csv", _per_class_rows(metrics["test"]))
    _write_csv(out_dir / "confident_errors.csv", errors)
    _write_csv(out_dir / "confident_error_examples.csv",
               sample_examples(errors, EXAMPLES_PER_CATEGORY, config.get("example_seed", 0)))
    return metrics
