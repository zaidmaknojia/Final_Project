"""Isotonic recalibration of stated confidence, fitted on dev (spec RQ4).

Saved as JSON breakpoints (no pickle); applied by linear interpolation with
clipping, which is exactly how a fitted IsotonicRegression predicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class IsotonicMap:
    x: tuple[float, ...]
    y: tuple[float, ...]


def fit_isotonic(conf: np.ndarray, correct: np.ndarray) -> IsotonicMap:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(conf, float), np.asarray(correct, float))
    return IsotonicMap(tuple(map(float, model.X_thresholds_)), tuple(map(float, model.y_thresholds_)))


def apply_isotonic(model: IsotonicMap, conf: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(np.asarray(conf, float), model.x, model.y), 0.0, 1.0)


def save_isotonic(model: IsotonicMap, path: Path) -> None:
    Path(path).write_text(json.dumps({"x": list(model.x), "y": list(model.y)}, indent=2))


def load_isotonic(path: Path) -> IsotonicMap:
    data = json.loads(Path(path).read_text())
    return IsotonicMap(tuple(data["x"]), tuple(data["y"]))
