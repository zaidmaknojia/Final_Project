import numpy as np
import pytest

from pubmedqa_abst.recalibrate import apply_isotonic, fit_isotonic, load_isotonic, save_isotonic
from pubmedqa_abst.scores import ItemScore
from pubmedqa_abst.thresholds import fit_all, fit_threshold


def test_fit_threshold_basic():
    r = fit_threshold(np.array([.9, .8, .7, .6]), n_total=4, target=.5)
    assert r == {"threshold": .8, "dev_coverage": .5, "target": .5}


def test_fit_threshold_all_tied():
    r = fit_threshold(np.array([1.0] * 8 + [.3] * 2), n_total=10, target=.7)
    assert r["threshold"] == 1.0 and r["dev_coverage"] == pytest.approx(.8)


def test_fit_threshold_invalid_items_reduce_coverage():
    r = fit_threshold(np.array([.9, .8, .8, .7, .6, .5, .5, .4]), n_total=10, target=.9)
    assert r["threshold"] == .4 and r["dev_coverage"] == pytest.approx(.8)


def test_fit_threshold_no_valid_items():
    r = fit_threshold(np.array([]), n_total=5, target=.5)
    assert r["threshold"] is None and r["dev_coverage"] == 0.0


def test_fit_all_covers_both_scores_and_targets():
    scores = [ItemScore(str(i), "E+", "yes", "yes", True, i / 10, i / 10, True, 0) for i in range(10)]
    scores.append(ItemScore("x", "E+", "yes", None, False, None, None, False, 0))
    out = fit_all(scores, [0.5, 0.9])
    assert set(out) == {"s_verb", "s_cons"} and set(out["s_verb"]) == {"0.5", "0.9"}
    assert out["s_verb"]["0.5"]["threshold"] == pytest.approx(0.4)  # 6 of 11 accepted
    assert out["s_verb"]["0.5"]["dev_coverage"] >= 0.5


def test_isotonic_roundtrip_json(tmp_path):
    conf = np.array([.1, .2, .5, .6, .9, .95])
    correct = np.array([0, 0, 1, 0, 1, 1])
    model = fit_isotonic(conf, correct)
    path = tmp_path / "iso.json"
    save_isotonic(model, path)
    loaded = load_isotonic(path)
    probe = np.array([0.0, .15, .55, .99, 1.0])
    out = apply_isotonic(loaded, probe)
    assert np.allclose(out, apply_isotonic(model, probe))
    assert ((out >= 0) & (out <= 1)).all()
