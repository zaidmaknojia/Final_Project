"""Index-based metric functions for the paired bootstrap (spec §9 planned comparisons).

Every function takes an index array into one shared item order (test items
present in both conditions), so E+ and E- resamples are paired item-for-item.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .bootstrap import diff
from .metrics import aurc, auroc, brier, complete_curve, ece, risk_coverage

Fn = Callable[[np.ndarray], float | None]
CONDITIONS = ("E+", "E-")
SCORE_NAMES = ("s_verb", "s_cons", "s_verb_iso")


class Arrays:
    """conf (NaN = abstain) and correct for one condition and score, in shared order."""

    def __init__(self, conf: np.ndarray, correct: np.ndarray) -> None:
        self.conf, self.correct = conf, correct

    def valid(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, y = self.conf[idx], self.correct[idx]
        mask = ~np.isnan(c)
        return c[mask], y[mask]

    def summary(self, kind: str, n_bins: int) -> Fn:
        def fn(idx: np.ndarray) -> float | None:
            c, y = self.valid(idx)
            if c.size == 0:
                if kind == "aurc":  # nothing scored: answering everything is the whole curve
                    return 1.0 - float(self.correct[idx].mean())
                return None
            if kind == "brier":
                return brier(c, y)
            if kind == "ece":
                return ece(c, y, n_bins)[0]
            if kind == "auroc":
                return auroc(c, y)
            full_risk = 1.0 - float(self.correct[idx].mean())
            return aurc(complete_curve(risk_coverage(c, y, len(idx)), full_risk))
        return fn

    def at(self, kind: str, threshold: float | None) -> Fn:
        def fn(idx: np.ndarray) -> float | None:
            c, y = self.valid(idx)
            accepted = (c >= threshold - 1e-9) if threshold is not None else np.zeros(c.size, bool)
            n_acc = int(accepted.sum())
            if kind == "coverage":
                return n_acc / len(idx)
            if kind == "cer":
                return int((accepted & ~y).sum()) / len(idx)
            return float(y[accepted].mean()) if n_acc else None
        return fn

    def accuracy_all(self) -> Fn:
        return lambda idx: float(self.correct[idx].mean())


def _arrays(test_scores: dict, iso, ids: list[str]) -> dict[tuple[str, str], Arrays]:
    from .evaluate import score_arrays  # local import avoids a cycle

    out = {}
    for cond in CONDITIONS:
        by_id = {s.item_id: s for s in test_scores[cond]}
        ordered = [by_id[i] for i in ids]
        for name in SCORE_NAMES:
            out[(cond, name)] = Arrays(*score_arrays(ordered, name, iso))
    return out


SUMMARIES = ("auroc", "aurc", "brier", "ece")


def build_comparisons(test_scores: dict, iso, thresholds: dict, n_bins: int) -> dict[str, dict[str, Fn]]:
    ids = sorted(set(s.item_id for s in test_scores["E+"]) & set(s.item_id for s in test_scores["E-"]))
    arr = _arrays(test_scores, iso, ids)
    thr = {n: {t: f["threshold"] for t, f in thresholds["primary"][n].items()} for n in SCORE_NAMES}
    targets = list(thr["s_verb"])
    groups: dict[str, dict[str, Fn]] = {k: {} for k in
                                        ("absolute", "cons_vs_verb", "threshold_vs_accept_all",
                                         "eplus_vs_eminus", "iso_vs_raw_verb")}
    for cond in CONDITIONS:
        for name in SCORE_NAMES:
            a = arr[(cond, name)]
            for kind in SUMMARIES:
                groups["absolute"][f"{cond}|{name}|{kind}"] = a.summary(kind, n_bins)
            for t in targets:
                for kind in ("coverage", "accuracy", "cer"):
                    groups["absolute"][f"{cond}|{name}|{kind}@{t}"] = a.at(kind, thr[name][t])
                groups["threshold_vs_accept_all"][f"{cond}|{name}|accuracy@{t}-accept_all"] = diff(
                    a.at("accuracy", thr[name][t]), a.accuracy_all())
        cons, verb = arr[(cond, "s_cons")], arr[(cond, "s_verb")]
        for kind in SUMMARIES:
            groups["cons_vs_verb"][f"{cond}|{kind}"] = diff(cons.summary(kind, n_bins), verb.summary(kind, n_bins))
        for t in targets:
            for kind in ("coverage", "accuracy"):
                groups["cons_vs_verb"][f"{cond}|{kind}@{t}"] = diff(
                    cons.at(kind, thr["s_cons"][t]), verb.at(kind, thr["s_verb"][t]))
        iso_a = arr[(cond, "s_verb_iso")]
        for kind in ("brier", "ece", "aurc"):
            groups["iso_vs_raw_verb"][f"{cond}|{kind}"] = diff(iso_a.summary(kind, n_bins), verb.summary(kind, n_bins))
    groups["eplus_vs_eminus"]["accuracy_full"] = diff(arr[("E+", "s_verb")].accuracy_all(),
                                                      arr[("E-", "s_verb")].accuracy_all())
    for name in SCORE_NAMES:
        ep, em = arr[("E+", name)], arr[("E-", name)]
        for kind in SUMMARIES:
            groups["eplus_vs_eminus"][f"{name}|{kind}"] = diff(ep.summary(kind, n_bins), em.summary(kind, n_bins))
        for t in targets:
            for kind in ("coverage", "accuracy", "cer"):
                groups["eplus_vs_eminus"][f"{name}|{kind}@{t}"] = diff(
                    ep.at(kind, thr[name][t]), em.at(kind, thr[name][t]))
    return groups
