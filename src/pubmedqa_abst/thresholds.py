"""Tie-aware coverage-target thresholds, fitted on dev only (spec §7)."""

from __future__ import annotations

import numpy as np

from .scores import ItemScore

SCORE_NAMES = ("s_verb", "s_cons")


def fit_threshold(conf: np.ndarray, n_total: int, target: float) -> dict:
    """Largest unique score t with coverage(conf >= t) >= target; ties stay together.

    If no t reaches the target (e.g. too many invalid items), returns the minimum
    score and the coverage it achieves. With no valid items, threshold is None.
    """
    conf = np.asarray(conf, float)
    if conf.size == 0:
        return {"threshold": None, "dev_coverage": 0.0, "target": target}
    for t in np.unique(conf)[::-1]:
        coverage = float((conf >= t).sum()) / n_total
        if coverage >= target:
            return {"threshold": float(t), "dev_coverage": coverage, "target": target}
    t = float(conf.min())
    return {"threshold": t, "dev_coverage": float((conf >= t).sum()) / n_total, "target": target}


def score_values(scores: list[ItemScore], score_name: str) -> np.ndarray:
    return np.array([getattr(s, score_name) for s in scores
                     if s.valid and getattr(s, score_name) is not None], float)


def fit_all(dev_scores: list[ItemScore], targets: list[float],
            score_names: tuple[str, ...] = SCORE_NAMES) -> dict:
    n_total = len(dev_scores)
    return {
        name: {str(t): fit_threshold(score_values(dev_scores, name), n_total, t) for t in targets}
        for name in score_names
    }
