"""Paired percentile bootstrap over test items (spec §9).

Every metric sees the same resampled index arrays, which is what makes
comparisons paired. Replicates where a metric is undefined (None) are dropped
for that metric and counted in `n_valid`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray], float | None]


def diff(metric_a: MetricFn, metric_b: MetricFn) -> MetricFn:
    def _diff(idx: np.ndarray) -> float | None:
        a, b = metric_a(idx), metric_b(idx)
        return None if a is None or b is None else a - b

    return _diff


def paired_bootstrap(item_ids: list[str], metric_fns: dict[str, MetricFn], B: int, seed: int) -> dict[str, dict]:
    n = len(item_ids)
    full = np.arange(n)
    draws = np.random.default_rng(seed).integers(0, n, size=(B, n))
    out = {}
    replicates: dict[str, list[float]] = {name: [] for name in metric_fns}
    points = {name: fn(full) for name, fn in metric_fns.items()}
    for idx in draws:
        for name, fn in metric_fns.items():
            value = fn(idx)
            if value is not None:
                replicates[name].append(value)
    for name, values in replicates.items():
        if values:
            lo, hi = np.percentile(values, [2.5, 97.5])
            out[name] = {"point": points[name], "lo": float(lo), "hi": float(hi), "n_valid": len(values)}
        else:
            out[name] = {"point": points[name], "lo": None, "hi": None, "n_valid": 0}
    return out
