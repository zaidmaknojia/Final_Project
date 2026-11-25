import json
import re

import pytest

from pubmedqa_abst.posthoc import posthoc
from pubmedqa_abst.report_numbers import (
    build_macros, ci, coverage_shift_pp, error_families, examples_table, fmt, latex_escape, macro_name,
    posthoc_macros, posthoc_table, sample_stated_examples, write_report_inputs,
)
from pubmedqa_abst.scores import ItemScore


def test_fmt_rounds_and_marks_undefined():
    assert fmt(0.12345) == "0.123"
    assert fmt(0.5, digits=1) == "0.5"
    assert fmt(None) == "--"


def test_fmt_uses_a_typographic_minus_for_negatives():
    assert fmt(-0.2174) == r"\textminus{}0.217"
    assert fmt(-0.0001) == "0.000"


def test_ci_formats_point_and_interval():
    assert ci({"point": 0.464, "lo": 0.41, "hi": 0.518}) == "0.464 [0.410, 0.518]"
    assert ci({"point": None, "lo": None, "hi": None}) == "--"


@pytest.mark.parametrize("group,key,expected", [
    ("absolute", "E+|s_verb|auroc", "EpVerbAuroc"),
    ("absolute", "E-|s_cons|accuracy@0.7", "EmConsAccSeventy"),
    ("cons_vs_verb", "E+|coverage@0.5", "ConsVsVerbEpCovFifty"),
    ("threshold_vs_accept_all", "E-|s_verb_iso|accuracy@0.9-accept_all", "ThrVsAllEmIsoAccNinetyAcceptAll"),
    ("eplus_vs_eminus", "accuracy_full", "EvidDeltaAccFull"),
    ("iso_vs_raw_verb", "E+|ece", "IsoVsRawEpEce"),
])
def test_macro_name_maps_keys_to_letters_only(group, key, expected):
    assert macro_name(group, key) == expected


def test_error_families_groups_confusion_pairs():
    rows = [{"condition": "E+", "score": "s_verb", "confusion": c}
            for c in ("yes->maybe", "no->maybe", "maybe->yes", "yes->no")]
    fam = error_families(rows)
    assert fam[("E+", "s_verb")] == {"hedge": 2, "overcommit": 1, "flip": 1, "total": 4}


def test_build_macros_matches_metrics(results_dir):
    macros = build_macros(results_dir)
    metrics = json.loads((results_dir / "metrics.json").read_text())
    assert macros["EpAccAll"] == fmt(metrics["test"]["E+"]["full"]["accuracy"])
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in macros)
    boot = json.loads((results_dir / "bootstrap.json").read_text())
    entry = boot["eplus_vs_eminus"]["accuracy_full"]
    assert macros["EvidDeltaAccFull"] == fmt(entry["point"])
    assert macros["EvidDeltaAccFullCI"] == f"[{fmt(entry['lo'])}, {fmt(entry['hi'])}]"


def test_maybe_as_abstention_uses_confusion_matrix(results_dir):
    macros = build_macros(results_dir)
    cm = json.loads((results_dir / "metrics.json").read_text())["test"]["E+"]["full"]["confusion"]
    answered = sum(cm[p][g] for p in ("yes", "no") for g in cm[p])
    n = sum(cm[p][g] for p in cm for g in cm[p])
    assert macros["EpMaybeAbstainCov"] == fmt(answered / n)


def test_write_report_inputs_writes_macros_and_tables(results_dir, tmp_path):
    (results_dir / "posthoc.json").write_text(json.dumps(_posthoc_payload()))
    paths = write_report_inputs(results_dir, tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["numbers.tex", "table_calibration.tex", "table_deltas.tex",
                     "table_examples.tex", "table_perclass.tex", "table_posthoc.tex", "table_selective.tex"]
    numbers = (tmp_path / "numbers.tex").read_text()
    assert "\\newcommand{\\EpAccAll}" in numbers
    assert "\\newcommand{\\PhEpLabelAuroc}" in numbers
    for name in names[1:]:
        assert "\\\\" in (tmp_path / name).read_text()


def test_latex_escape_protects_special_characters():
    assert latex_escape("50% of A&B_c #1 ~x") == r"50\% of A\&B\_c \#1 \textasciitilde{}x"


def _example(cond, score, confusion, question="Q?"):
    return {"condition": cond, "score": score, "confusion": confusion, "gold": confusion.split("->")[0],
            "a0": confusion.split("->")[1], "s_verb": "0.95", "s_cons": "1.0", "question": question}


def test_examples_table_keeps_stated_rows_sorted_by_condition_then_family():
    rows = [_example("E-", "s_verb", "no->yes", "Third?"), _example("E+", "s_cons", "yes->no", "Skip?"),
            _example("E+", "s_verb", "yes->maybe", "First?"), _example("E+", "s_verb", "maybe->no", "Second?")]
    lines = examples_table(rows).strip().splitlines()
    assert len(lines) == 3
    assert [line.split(" & ")[-1] for line in lines] == [r"First? \\", r"Second? \\", r"Third? \\"]
    assert lines[0].split(" & ")[:5] == ["E+", "hedge", r"yes$\to$maybe", "0.95", "1.00"]
    assert lines[2].startswith("E$-$ & flip")


def _score(iid, gold, a0, s_verb, cond):
    return ItemScore(item_id=iid, condition=cond, gold=gold, a0=a0, correct=a0 == gold,
                     s_verb=s_verb, s_cons=1.0, valid=True, invalid_samples=0)


def _posthoc_payload():
    rows = [("1", "yes", "yes", 0.95), ("2", "no", "no", 0.9), ("3", "yes", "no", 0.95),
            ("4", "yes", "maybe", 0.85), ("5", "maybe", "maybe", 0.8), ("6", "maybe", "yes", 0.95)]
    scores = {c: [_score(*r, c) for r in rows] for c in ("E+", "E-")}
    thresholds = {"primary": {"s_verb": {"0.7": {"threshold": 0.9}}}}
    return posthoc(scores, thresholds, error_rows=[], samples=None, B=20, seed=0)


def test_posthoc_macros_are_letters_only_and_match_payload():
    payload = _posthoc_payload()
    m = posthoc_macros(payload)
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in m)
    assert m["PhEpLabelAuroc"] == fmt(payload["E+"]["auroc"]["label"])
    assert m["PhEpWithinN"] == "4"
    assert m["PhEpGoldMaybeCorrectYesNo"] == "0"
    assert m["PhMajorityAcc"] == fmt(0.5)
    entry = payload["bootstrap"]["E+"]["within_verb_auroc"]
    assert m["PhEpWithinVerbAurocCI"] == f"[{fmt(entry['lo'])}, {fmt(entry['hi'])}]"


def test_posthoc_table_has_three_signals_per_condition():
    lines = posthoc_table(_posthoc_payload()).strip().splitlines()
    rows = [line for line in lines if line.endswith(r"\\")]
    assert len(rows) == 6
    assert rows[2].startswith("E+ & Label only")
    assert rows[2].split(" & ")[-1] == r"-- \\"


def test_sample_stated_examples_fills_each_family_quota_from_stated_rows_only():
    rows = ([_example("E+", "s_verb", "yes->maybe", f"H{i}?") for i in range(5)]
            + [_example("E+", "s_verb", "maybe->no", f"O{i}?") for i in range(5)]
            + [_example("E+", "s_cons", "yes->maybe", "C?")]
            + [_example("E-", "s_verb", "no->yes", "F?")])
    for i, r in enumerate(rows):
        r["item_id"] = str(i)
    quota = {("E+", "hedge"): 2, ("E+", "overcommit"): 1, ("E-", "flip"): 3}
    chosen = sample_stated_examples(rows, quota, seed=1)
    assert sorted((r["condition"], r["confusion"]) for r in chosen) == [
        ("E+", "maybe->no"), ("E+", "yes->maybe"), ("E+", "yes->maybe"), ("E-", "no->yes")]
    assert all(r["score"] == "s_verb" for r in chosen)
    assert sample_stated_examples(rows, quota, seed=1) == chosen


def _token_payload():
    import numpy as np

    from pubmedqa_abst.posthoc import token_posthoc

    payload = _posthoc_payload()
    rows = [("1", "yes", "yes", 0.95), ("2", "no", "no", 0.9), ("3", "yes", "no", 0.95),
            ("4", "yes", "maybe", 0.85), ("5", "maybe", "maybe", 0.8), ("6", "maybe", "yes", 0.95)]
    scores = {c: [_score(*r, c) for r in rows] for c in ("E+", "E-")}
    tok = {c: np.array([0.99, 0.9, 0.6, 0.7, 0.8, 0.5]) for c in scores}
    thresholds = {"primary": {"s_verb": {"0.7": {"threshold": 0.9}}, "s_cons": {"0.7": {"threshold": 1.0}}}}
    payload["token"] = token_posthoc(scores, tok, np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5]), 6, thresholds, B=20, seed=0)
    one = {"n": 6, "ok": 6, "missing": 0, "no_answer_token": 0, "transport_error": 0, "answer_matches_a0": 6}
    payload["token"]["agreement"] = {f"{s}|{c}": dict(one) for s in ("dev", "test") for c in ("E+", "E-")}
    return payload


def test_token_macros_cover_the_baseline_and_its_comparison_with_stated_confidence():
    payload = _token_payload()
    m, t = posthoc_macros(payload), payload["token"]
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in m)
    assert m["TokEpAuroc"] == fmt(t["E+"]["s_tok"]["auroc"])
    assert m["TokEpWithinAuroc"] == fmt(t["E+"]["s_tok"]["within_committed_auroc"])
    assert m["TokEmCer"] == fmt(t["E-"]["s_tok"]["policy"]["confident_error_rate"])
    assert m["TokEpVerbCer"] == fmt(t["E+"]["s_verb"]["policy"]["confident_error_rate"])
    assert m["TokEpConsAurc"] == fmt(t["E+"]["s_cons"]["aurc"])
    entry = t["bootstrap"]["E+"]["tok_minus_verb_auroc"]
    assert m["TokEpMinusVerb"] == fmt(entry["point"])
    assert m["TokEpMinusVerbCI"] == f"[{fmt(entry['lo'])}, {fmt(entry['hi'])}]"
    assert m["TokEpWithinCI"].startswith("[")
    assert m["TokThreshold"] == fmt(t["threshold"]["threshold"])
    assert (m["TokCalls"], m["TokMatches"], m["TokUnscored"]) == ("24", "24", "0")
    assert m["TokEpMaybeMean"] == fmt(t["E+"]["s_tok_by_label"]["maybe"]["mean"])
    assert m["TokEmMaybeAccepted"] == fmt(t["E-"]["s_tok_by_label"]["maybe"]["accepted_share"])


def test_posthoc_macros_leave_out_token_macros_without_token_data():
    assert not any(name.startswith("Tok") for name in posthoc_macros(_posthoc_payload()))


def test_posthoc_table_adds_a_token_row_per_condition_when_token_data_exist():
    rows = [line for line in posthoc_table(_token_payload()).splitlines() if line.endswith(r"\\")]
    assert len(rows) == 8
    assert rows[3].startswith("E+ & Token prob.") and rows[7].startswith("E$-$ & Token prob.")
    assert "[" in rows[3].split(" & ")[-1]


def test_counts_use_a_latex_thousands_separator():
    from pubmedqa_abst.report_numbers import _count

    assert (_count(24), _count(2000), _count(22000)) == ("24", "2{,}000", "22{,}000")


def test_definite_error_rate_excludes_hedges_from_confident_errors():
    payload = _posthoc_payload()
    payload["E+"]["errors"] = {**payload["E+"]["errors"], "total": 5, "hedge": 2}
    m = posthoc_macros(payload)
    assert m["PhEpErrDefiniteRate"] == fmt(3 / 6)


def _entry(coverage, dev_coverage):
    return {"coverage": coverage, "dev_coverage": dev_coverage}


def test_coverage_shift_is_the_largest_dev_to_test_gap_in_percentage_points():
    scores = {score: {"primary": {t: _entry(0.80, 0.80) for t in ("0.5", "0.7", "0.9")}}
              for score in ("s_verb", "s_cons", "s_verb_iso")}
    scores["s_cons"]["primary"]["0.7"] = _entry(0.846, 0.870)
    scores["s_verb"]["primary"]["0.9"] = _entry(0.990, 0.975)
    metrics = {"test": {"E+": {"scores": scores}}}
    assert coverage_shift_pp(metrics, "E+") == pytest.approx(2.4)


def test_build_macros_reports_the_e_plus_coverage_shift(results_dir):
    macros = build_macros(results_dir)
    metrics = json.loads((results_dir / "metrics.json").read_text())
    assert macros["EpThrShiftMaxPP"] == f"{coverage_shift_pp(metrics, 'E+'):.1f}"
