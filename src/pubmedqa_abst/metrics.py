"""Calibration, discrimination and selective-prediction metrics (spec §8).

Undefined values are returned as None, never 0: AUROC with a single class,
selective accuracy at zero coverage, AURC with no valid items.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import LABELS
from .scores import ItemScore

_EPS = 1e-9


def accuracy(pred: Sequence[str | None], gold: Sequence[str]) -> float:
    return float(np.mean([p is not None and p == g for p, g in zip(pred, gold)]))


def macro_f1(pred: Sequence[str | None], gold: Sequence[str]) -> float:
    f1s = []
    for label in LABELS:
        tp = sum(p == label and g == label for p, g in zip(pred, gold))
        fp = sum(p == label and g != label for p, g in zip(pred, gold))
        fn = sum(p != label and g == label for p, g in zip(pred, gold))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def confusion(pred: Sequence[str | None], gold: Sequence[str]) -> dict[str, dict[str, int]]:
    rows = {p: {g: 0 for g in LABELS} for p in (*LABELS, "invalid")}
    for p, g in zip(pred, gold):
        rows[p if p is not None else "invalid"][g] += 1
    return rows


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum(np.floor(conf * n_bins + _EPS).astype(int), n_bins - 1)
    bins, total, n = [], 0.0, len(conf)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        mean_conf = float(conf[mask].mean()) if count else None
        acc = float(correct[mask].mean()) if count else None
        if count:
            total += count / n * abs(mean_conf - acc)
        bins.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "count": count,
                     "mean_conf": mean_conf, "acc": acc})
    return float(total), bins


def auroc(conf: np.ndarray, correct: np.ndarray) -> float | None:
    correct = np.asarray(correct, int)
    if len(np.unique(correct)) < 2:
        return None
    return float(roc_auc_score(correct, np.asarray(conf, float)))


def risk_coverage(conf: np.ndarray, correct: np.ndarray, n_total: int) -> list[tuple[float, float, float]]:
    """(threshold, coverage, risk) per unique score, descending; ties never split."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    curve = []
    for t in np.unique(conf)[::-1]:
        accepted = conf >= t
        n_acc = int(accepted.sum())
        curve.append((float(t), n_acc / n_total, float(1 - correct[accepted].mean())))
    return curve


def complete_curve(curve: list[tuple[float, float, float]], full_risk: float) -> list[tuple[float, float, float]]:
    """Extend a risk-coverage curve to coverage 1.0, where items without a score
    are answered too (invalid ones counted wrong). Makes AURC comparable across
    scores with different missing rates (spec §6: invalid = incorrect at full coverage)."""
    if curve and curve[-1][1] >= 1.0 - _EPS:
        return list(curve)
    return [*curve, (float("-inf"), 1.0, float(full_risk))]


def aurc(curve: list[tuple[float, float, float]]) -> float | None:
    if not curve:
        return None
    total, prev_c = 0.0, 0.0
    for _, c, r in curve:
        total += (c - prev_c) * r
        prev_c = c
    return float(total)


def selective(conf: np.ndarray, correct: np.ndarray, threshold: float, n_total: int) -> dict:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    accepted = conf >= threshold - _EPS
    n_acc = int(accepted.sum())
    return {
        "coverage": n_acc / n_total if n_total else 0.0,
        "accuracy": float(correct[accepted].mean()) if n_acc else None,
        "n_accepted": n_acc,
    }


def _accepts(score: ItemScore, score_name: str, threshold: float) -> bool:
    value = getattr(score, score_name)
    return score.valid and value is not None and value >= threshold - _EPS


def per_class(item_scores: list[ItemScore], score_name: str, threshold: float,
              by: str = "gold") -> dict[str, dict]:
    table = {}
    for label in LABELS:
        group = [s for s in item_scores if (s.gold if by == "gold" else s.a0) == label]
        accepted = [s for s in group if _accepts(s, score_name, threshold)]
        n = len(group)
        table[label] = {
            "n": n,
            "accepted_share": len(accepted) / n if n else None,
            "acc_accepted": sum(s.correct for s in accepted) / len(accepted) if accepted else None,
            "confident_error_share": sum(not s.correct for s in accepted) / n if n else None,
        }
    return table
