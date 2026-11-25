import pytest

from helpers import raw_row, write_raw
from pubmedqa_abst.data import Item
from pubmedqa_abst.scores import score_run

ITEMS = {"a": Item("a", "q?", ("c",), "yes"), "b": Item("b", "q?", ("c",), "no")}


def test_s_cons_counts_invalid_as_nonmatch(tmp_path):
    rows = [raw_row("a", 0, answer="yes", confidence=80),
            raw_row("a", 1, answer="yes"), raw_row("a", 2, answer="yes"),
            raw_row("a", 3, status="invalid_format"), raw_row("a", 4, answer="no")]
    (s,) = score_run(write_raw(tmp_path / "raw.jsonl", rows), ITEMS, K=4)
    assert s.s_cons == pytest.approx(0.5) and s.invalid_samples == 1
    assert s.s_verb == pytest.approx(0.8) and s.correct and s.valid and s.a0 == "yes"


def test_missing_samples_count_as_nonmatch(tmp_path):
    rows = [raw_row("a", 0), raw_row("a", 1)]
    (s,) = score_run(write_raw(tmp_path / "raw.jsonl", rows), ITEMS, K=4)
    assert s.s_cons == pytest.approx(0.25)


def test_invalid_greedy_never_gets_zero_confidence(tmp_path):
    rows = [raw_row("b", 0, status="invalid_format"), raw_row("b", 1, answer="no")]
    (s,) = score_run(write_raw(tmp_path / "raw.jsonl", rows), ITEMS, K=1)
    assert s.a0 is None and s.s_verb is None and s.s_cons is None
    assert s.valid is False and s.correct is False


def test_invalid_confidence_keeps_answer(tmp_path):
    rows = [raw_row("b", 0, status="invalid_confidence", answer="no"), raw_row("b", 1, answer="no")]
    (s,) = score_run(write_raw(tmp_path / "raw.jsonl", rows), ITEMS, K=1)
    assert s.a0 == "no" and s.s_verb is None and s.s_cons == 1.0 and s.correct


def test_latest_line_wins(tmp_path):
    rows = [raw_row("a", 0, status="transport_error"), raw_row("a", 0, answer="no", confidence=60),
            raw_row("a", 1, answer="no")]
    (s,) = score_run(write_raw(tmp_path / "raw.jsonl", rows), ITEMS, K=1)
    assert s.a0 == "no" and s.s_verb == pytest.approx(0.6) and s.correct is False


def test_one_score_per_item_sorted(tmp_path):
    rows = [raw_row("b", 0, answer="no"), raw_row("a", 0)]
    scores = score_run(write_raw(tmp_path / "raw.jsonl", rows), ITEMS, K=0)
    assert [s.item_id for s in scores] == ["a", "b"] and all(s.condition == "E+" for s in scores)


def test_incomplete_keys_reports_missing_and_transport(tmp_path):
    from pubmedqa_abst.scores import incomplete_keys

    rows = [raw_row("a", 0), raw_row("a", 1), raw_row("b", 0), raw_row("b", 1, status="transport_error")]
    missing = incomplete_keys(write_raw(tmp_path / "raw.jsonl", rows), ["a", "b", "c"], K=1)
    assert sorted(missing) == ["b|1", "c|0", "c|1"]


def test_require_complete_raises_with_counts(tmp_path):
    from pubmedqa_abst.scores import require_complete

    raw = write_raw(tmp_path / "raw.jsonl", [raw_row("a", 0)])
    with pytest.raises(RuntimeError, match="incomplete.*3 of 4"):
        require_complete(raw, ["a", "b"], K=1)
    require_complete(write_raw(tmp_path / "ok.jsonl", [raw_row("a", 0), raw_row("a", 1)]), ["a"], K=1)
