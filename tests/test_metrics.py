import numpy as np
import pytest

from pubmedqa_abst.metrics import (
    accuracy, aurc, auroc, brier, confusion, ece, macro_f1, per_class, risk_coverage, selective,
)
from pubmedqa_abst.scores import ItemScore


def mk(iid, gold, a0, s_verb):
    return ItemScore(iid, "E+", gold, a0, a0 == gold if a0 else False, s_verb, s_verb,
                     a0 is not None, 0)


@pytest.fixture
def item_scores_fixture():
    return [mk("1", "yes", "yes", .9), mk("2", "yes", "no", .85), mk("3", "no", "no", .5),
            mk("4", "maybe", "yes", .95), mk("5", "maybe", None, None)]


def test_ece_last_bin_closed_and_empty_bins_zero_weight():
    e, bins = ece(np.array([1.0, 1.0, 0.05]), np.array([1, 0, 0]), 10)
    assert bins[9]["count"] == 2 and bins[0]["count"] == 1
    assert sum(b["count"] for b in bins) == 3 and bins[5]["count"] == 0
    assert e == pytest.approx((2 / 3) * abs(1.0 - 0.5) + (1 / 3) * abs(0.05 - 0))


def test_ece_boundary_goes_to_upper_bin():
    _, bins = ece(np.array([0.1, 0.9]), np.array([1, 1]), 10)
    assert bins[1]["count"] == 1 and bins[9]["count"] == 1


def test_brier():
    assert brier(np.array([1.0, 0.0]), np.array([1, 1])) == pytest.approx(0.5)


def test_auroc_single_class_is_none():
    assert auroc(np.array([.2, .9]), np.array([1, 1])) is None


def test_auroc_ties_averaged():
    assert auroc(np.array([.5, .5]), np.array([1, 0])) == pytest.approx(0.5)


def test_risk_coverage_keeps_ties_together():
    curve = risk_coverage(np.array([.9, .9, .5]), np.array([1, 0, 1]), n_total=4)
    assert [round(c, 3) for _, c, _ in curve] == [0.5, 0.75]
    assert curve[0][2] == pytest.approx(0.5) and curve[1][2] == pytest.approx(1 / 3)


def test_risk_coverage_empty():
    assert risk_coverage(np.array([]), np.array([]), n_total=3) == []


def test_aurc_step_formula():
    curve = [(0.9, 0.5, 0.5), (0.5, 0.75, 1 / 3)]
    assert aurc(curve) == pytest.approx(0.5 * 0.5 + 0.25 * (1 / 3))


def test_aurc_empty_is_none():
    assert aurc([]) is None


def test_selective_zero_coverage_accuracy_none():
    assert selective(np.array([.2]), np.array([1]), 0.9, 1)["accuracy"] is None


def test_selective_counts_ties_at_threshold():
    r = selective(np.array([.9, .7, .7, .1]), np.array([1, 1, 0, 0]), 0.7, 5)
    assert r == {"coverage": pytest.approx(0.6), "accuracy": pytest.approx(2 / 3), "n_accepted": 3}


def test_accuracy_counts_invalid_as_wrong():
    assert accuracy(["yes", None], ["yes", "no"]) == 0.5


def test_macro_f1_perfect_and_invalid():
    assert macro_f1(["yes", "no", "maybe"], ["yes", "no", "maybe"]) == pytest.approx(1.0)
    assert macro_f1([None, "no", "maybe"], ["yes", "no", "maybe"]) == pytest.approx(2 / 3)


def test_confusion_has_invalid_row():
    c = confusion(["yes", None], ["yes", "no"])
    assert c["yes"]["yes"] == 1 and c["invalid"]["no"] == 1


def test_per_class_maybe_bucket(item_scores_fixture):
    t = per_class(item_scores_fixture, "s_verb", 0.8)
    assert set(t) == {"yes", "no", "maybe"}
    assert t["maybe"] == {"n": 2, "accepted_share": 0.5, "acc_accepted": 0.0, "confident_error_share": 0.5}
    assert t["no"]["accepted_share"] == 0.0 and t["no"]["acc_accepted"] is None
    assert t["yes"]["acc_accepted"] == pytest.approx(0.5)


def test_per_class_by_predicted_label(item_scores_fixture):
    t = per_class(item_scores_fixture, "s_verb", 0.8, by="pred")
    assert t["yes"]["n"] == 2 and t["maybe"]["n"] == 0


def test_complete_curve_adds_full_coverage_point():
    from pubmedqa_abst.metrics import complete_curve

    out = complete_curve([(0.9, 0.5, 0.5)], full_risk=0.6)
    assert out[-1][1:] == (1.0, 0.6) and out[0] == (0.9, 0.5, 0.5)
    assert complete_curve([(0.1, 1.0, 0.3)], full_risk=0.3) == [(0.1, 1.0, 0.3)]
    assert complete_curve([], full_risk=0.4)[0][1:] == (1.0, 0.4)


def test_aurc_penalizes_missing_scores_via_complete_curve():
    from pubmedqa_abst.metrics import complete_curve

    conf, correct = np.array([.9, .8]), np.array([1, 1])
    partial = aurc(risk_coverage(conf, correct, n_total=4))
    full = aurc(complete_curve(risk_coverage(conf, correct, n_total=4), full_risk=0.5))
    assert full > partial
