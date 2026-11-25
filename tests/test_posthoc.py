import json

import numpy as np
import pytest

from pubmedqa_abst.posthoc import (
    answer_mix, error_profile, gold_maybe_outcome, label_entropy_auroc, label_score,
    posthoc, rate_by_gold, reference_aurc, write_posthoc,
)
from pubmedqa_abst.scores import ItemScore


def _s(iid, gold, a0, s_verb, s_cons=1.0, cond="E+"):
    return ItemScore(item_id=iid, condition=cond, gold=gold, a0=a0, correct=a0 == gold,
                     s_verb=s_verb, s_cons=s_cons, valid=a0 is not None, invalid_samples=0)


SCORES = [
    _s("1", "yes", "yes", 0.95), _s("2", "no", "no", 0.95), _s("3", "yes", "no", 0.95),
    _s("4", "yes", "maybe", 0.85), _s("5", "maybe", "maybe", 0.85), _s("6", "maybe", "yes", 0.95),
    _s("7", "no", None, None, None),
]


def test_label_score_is_one_only_for_committed_answers():
    assert [label_score(s) for s in SCORES] == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_answer_mix_reports_accuracy_and_mean_confidence_by_predicted_label():
    mix = answer_mix(SCORES)
    assert mix["committed"] == {"n": 4, "share": pytest.approx(4 / 7), "accuracy": 0.5, "mean_conf": 0.95}
    assert mix["maybe"] == {"n": 2, "share": pytest.approx(2 / 7), "accuracy": 0.5, "mean_conf": 0.85}


def test_gold_maybe_outcome_counts_what_the_model_said_on_equivocal_questions():
    assert gold_maybe_outcome(SCORES) == {"n": 2, "said_maybe": 1, "said_yes_no": 1, "correct_yes_no": 0}


def test_rate_by_gold_counts_accepted_errors_per_gold_question():
    rates = rate_by_gold(SCORES, "s_verb", 0.9)
    assert rates["yes"] == pytest.approx(1 / 3)
    assert rates["no"] == 0.0
    assert rates["maybe"] == pytest.approx(1 / 2)


def test_reference_aurc_random_is_error_rate_and_oracle_puts_errors_last():
    correct = np.array([1, 1, 0, 0])
    ref = reference_aurc(correct)
    assert ref["random"] == 0.5
    assert ref["oracle"] == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)


def test_label_entropy_auroc_ranks_unanimous_samples_highest():
    samples = {"a": ["yes"] * 4, "b": ["yes", "no", "yes", "no"], "c": ["no"] * 4}
    correct = {"a": True, "b": False, "c": True}
    assert label_entropy_auroc(samples, correct, ["a", "b", "c"]) == 1.0


def test_error_profile_summarises_overcommits_and_hedges_under_stated_confidence():
    rows = [
        {"condition": "E+", "score": "s_verb", "confusion": "maybe->yes", "s_verb": "0.95", "s_cons": "1.0"},
        {"condition": "E+", "score": "s_verb", "confusion": "maybe->no", "s_verb": "0.85", "s_cons": "0.9"},
        {"condition": "E+", "score": "s_verb", "confusion": "yes->maybe", "s_verb": "0.85", "s_cons": "1.0"},
        {"condition": "E+", "score": "s_cons", "confusion": "maybe->yes", "s_verb": "0.95", "s_cons": "1.0"},
        {"condition": "E-", "score": "s_verb", "confusion": "no->maybe", "s_verb": "0.8", "s_cons": "1.0"},
    ]
    prof = error_profile(rows, "E+")
    assert prof == {"total": 3, "on_gold_maybe": 2, "overcommit": 2, "overcommit_at_095": 1,
                    "overcommit_unanimous": 1, "hedge": 1, "hedge_at_085": 1}


def test_posthoc_reports_label_only_baseline_and_within_label_aurocs():
    scores = {"E+": SCORES[:6], "E-": [_s(s.item_id, s.gold, s.a0, s.s_verb, s.s_cons, "E-") for s in SCORES[:6]]}
    thresholds = {"primary": {"s_verb": {"0.7": {"threshold": 0.9}}}}
    out = posthoc(scores, thresholds, error_rows=[], samples=None, B=50, seed=0)
    ep = out["E+"]
    assert ep["auroc"]["label"] == pytest.approx(0.5)
    assert ep["within_committed"]["n"] == 4
    assert ep["within_committed"]["auroc_s_verb"] == 0.5  # all committed answers share 0.95
    assert ep["majority"] == {"label": "yes", "accuracy": 0.5}
    assert ep["n"] == 6
    assert set(out["bootstrap"]["E+"]) >= {"label_minus_verb_auroc", "within_verb_auroc", "within_cons_auroc"}


def test_write_posthoc_round_trips_json(tmp_path):
    path = write_posthoc({"E+": {"x": 1}}, tmp_path)
    assert json.loads(path.read_text()) == {"E+": {"x": 1}}
    assert path.name == "posthoc.json"


# --- token-probability baseline (post hoc) ---

from pubmedqa_abst.metrics import aurc, complete_curve, risk_coverage  # noqa: E402
from pubmedqa_abst.posthoc import fast_aurc, label_split, score_summary, token_posthoc  # noqa: E402
from pubmedqa_abst.thresholds import fit_threshold  # noqa: E402


def test_fast_aurc_matches_the_locked_step_aurc_with_and_without_ties():
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, size=200)
    coarse = rng.choice([0.5, 0.75, 0.85, 0.95, 1.0], size=200)
    assert fast_aurc(coarse, correct) == pytest.approx(aurc(complete_curve(risk_coverage(coarse, correct, 200), 0.0)))
    fine = rng.random(200)
    assert fast_aurc(fine, correct) == pytest.approx(aurc(risk_coverage(fine, correct, 200)))


def test_score_summary_reports_ranking_calibration_and_the_policy_at_a_threshold():
    scores = [_s("1", "yes", "yes", 0.9), _s("2", "no", "no", 0.8), _s("3", "yes", "no", 0.7), _s("4", "yes", "maybe", 0.6)]
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    out = score_summary(scores, conf, threshold=0.75)
    assert out["n_scored"] == 4
    assert out["auroc"] == 1.0 and out["within_committed_auroc"] == 1.0
    assert out["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)
    assert out["ece"] == pytest.approx(0.4)
    assert out["brier"] == pytest.approx(np.mean((conf - [1, 1, 0, 0]) ** 2))
    assert out["policy"] == {"threshold": 0.75, "coverage": 0.5, "accuracy": 1.0, "confident_error_rate": 0.0}


def test_score_summary_answers_unscored_items_last_at_full_coverage():
    scores = [_s("1", "yes", "yes", 0.9), _s("2", "no", "no", 0.8), _s("3", "yes", "no", 0.7), _s("4", "yes", "maybe", 0.6)]
    out = score_summary(scores, np.array([0.9, np.nan, 0.7, 0.6]), threshold=0.55)
    assert out["n_scored"] == 3
    assert out["aurc"] == pytest.approx((0 + 0.5 + 2 / 3 + 0.5) / 4)
    assert out["policy"]["coverage"] == 0.75 and out["policy"]["confident_error_rate"] == 0.5


def test_token_posthoc_fits_its_threshold_on_dev_and_sets_it_beside_the_locked_scores():
    test = {"E+": SCORES[:6], "E-": [_s(s.item_id, s.gold, s.a0, s.s_verb, s.s_cons, "E-") for s in SCORES[:6]]}
    tok = {c: np.array([0.99, 0.98, 0.6, 0.7, 0.9, 0.5]) for c in test}
    dev_tok = np.array([0.99, 0.9, 0.8, 0.7, 0.6, np.nan])
    thresholds = {"primary": {"s_verb": {"0.7": {"threshold": 0.9}}, "s_cons": {"0.7": {"threshold": 1.0}}}}
    out = token_posthoc(test, tok, dev_tok, 6, thresholds, B=50, seed=0)
    assert out["threshold"] == fit_threshold(dev_tok[:5], 6, 0.7)
    assert out["threshold"]["threshold"] == 0.6
    assert set(out["E+"]) == {"s_tok", "s_verb", "s_cons", "s_tok_by_label"}
    assert out["E+"]["s_verb"]["policy"]["threshold"] == 0.9
    assert out["E-"]["s_tok"]["policy"]["threshold"] == 0.6
    assert set(out["bootstrap"]["E+"]) == {"tok_minus_verb_auroc", "within_tok_auroc",
                                          "within_tok_minus_verb_auroc", "tok_minus_verb_aurc"}
    point = out["bootstrap"]["E+"]["tok_minus_verb_auroc"]["point"]
    assert point == pytest.approx(out["E+"]["s_tok"]["auroc"] - out["E+"]["s_verb"]["auroc"])


def test_label_split_gives_mean_score_and_accepted_share_for_committed_and_maybe_answers():
    scores = [_s("1", "yes", "yes", 0.9), _s("2", "no", "no", 0.8), _s("3", "yes", "maybe", 0.7), _s("4", "no", "maybe", 0.6)]
    out = label_split(scores, np.array([0.99, 0.90, 0.98, 0.50]), threshold=0.95)
    assert out["committed"] == {"n": 2, "mean": pytest.approx(0.945), "accepted_share": 0.5}
    assert out["maybe"] == {"n": 2, "mean": pytest.approx(0.74), "accepted_share": 0.5}
