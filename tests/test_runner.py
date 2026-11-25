import json
from collections import Counter

import pytest

from pubmedqa_abst.client import CallResult
from pubmedqa_abst.data import Item
from pubmedqa_abst.runner import completed_keys, compute_run_id, run, sample_seed

TEMPLATE = "{evidence_block}Q: {question}"
CONFIG = {"model": "gemma4:e4b", "model_digest": "c6eb396dbd59", "K": 2, "T": 0.7, "greedy_seed": 20261121}
ENV = {"ollama_version": "0.20.2", "model_digest": "c6eb396dbd59ff"}
ITEMS = [Item(f"id{i}", f"question {i}?", ("ctx",), "yes") for i in range(3)]
OK_TEXT = '{"answer":"yes","confidence":90}'


class FakeClient:
    def __init__(self, fail_at=None, error_calls=()):
        self.calls = []
        self.fail_at = fail_at
        self.error_calls = set(error_calls)

    def generate(self, prompt, temperature, seed, schema):
        self.calls.append((prompt, temperature, seed))
        n = len(self.calls)
        if self.fail_at is not None and n == self.fail_at:
            raise KeyboardInterrupt
        if n in self.error_calls:
            return CallResult(None, 0, 0.0, 4, "ConnectionError: x")
        return CallResult(OK_TEXT, 0, 0.01, 1, None)


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def go(tmp_path, client, **kw):
    return run(ITEMS, "E+", "dev", client, CONFIG, TEMPLATE, tmp_path, ENV, **kw)


def test_run_writes_1_plus_K_lines_per_item(tmp_path):
    raw = go(tmp_path, FakeClient())
    rows = lines(raw)
    assert len(rows) == 9
    assert {r["key"] for r in rows} == {f"id{i}|{k}" for i in range(3) for k in range(3)}
    assert all(r["status"] == "ok" and r["answer"] == "yes" and r["confidence"] == 90 for r in rows)


def test_greedy_uses_temp0_and_fixed_seed(tmp_path):
    rows = lines(go(tmp_path, FakeClient()))
    greedy = [r for r in rows if r["call_idx"] == 0]
    assert all(r["temperature"] == 0 and r["seed"] == 20261121 for r in greedy)
    samples = [r for r in rows if r["call_idx"] > 0]
    assert all(r["temperature"] == 0.7 for r in samples)
    assert len({r["seed"] for r in samples}) == len(samples)


def test_resume_after_interrupt_no_duplicates(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        go(tmp_path, FakeClient(fail_at=5))
    raw = go(tmp_path, FakeClient())
    terminal = Counter(r["key"] for r in lines(raw) if r["status"] != "transport_error")
    assert len(terminal) == 9 and set(terminal.values()) == {1}


def test_transport_error_is_retried_on_resume(tmp_path):
    raw = go(tmp_path, FakeClient(error_calls={1}))
    first = lines(raw)
    assert first[0]["key"] == "id0|0" and first[0]["status"] == "transport_error"
    assert "id0|0" not in completed_keys(raw)
    second_client = FakeClient()
    go(tmp_path, second_client)
    assert len(second_client.calls) == 1
    latest = [r for r in lines(raw) if r["key"] == "id0|0"][-1]
    assert latest["status"] == "ok"


def test_run_id_changes_with_prompt_or_condition():
    base = compute_run_id(CONFIG, "p", "parser", "dig", "dev", "E+")
    assert base != compute_run_id(CONFIG, "p2", "parser", "dig", "dev", "E+")
    assert base != compute_run_id(CONFIG, "p", "parser", "dig", "dev", "E-")
    assert len(base) == 12


def test_run_id_ignores_analysis_only_config():
    a = compute_run_id(CONFIG, "p", "parser", "dig", "dev", "E+")
    b = compute_run_id({**CONFIG, "bootstrap_B": 5}, "p", "parser", "dig", "dev", "E+")
    assert a == b


def test_sample_seed_deterministic_and_in_range():
    s = sample_seed("r", "id", "E+", 3)
    assert s == sample_seed("r", "id", "E+", 3) and 0 <= s < 2**31
    assert s != sample_seed("r", "id", "E+", 4)


def test_manifest_written_with_env(tmp_path):
    raw = go(tmp_path, FakeClient())
    manifest = json.loads((raw.parent / "manifest.json").read_text())
    assert manifest["env"] == ENV and manifest["split"] == "dev" and manifest["condition"] == "E+"
    assert manifest["K"] == 2


def test_limit_restricts_items(tmp_path):
    rows = lines(go(tmp_path, FakeClient(), limit=1))
    assert {r["item_id"] for r in rows} == {"id0"}


def test_truncated_last_line_is_rerun(tmp_path):
    raw = go(tmp_path, FakeClient())
    text = raw.read_text()
    raw.write_text(text[:-12])  # power loss mid-write: partial last line, no newline
    assert "id2|2" not in completed_keys(raw)
    second = FakeClient()
    go(tmp_path, second)
    assert len(second.calls) == 1
    from pubmedqa_abst.runner import read_lines

    rows = read_lines(raw)
    assert len(rows) == 9 and sum(r["key"] == "id2|2" for r in rows) == 1
    go(tmp_path, FakeClient())  # a later resume must not crash on the fragment


def test_consecutive_transport_errors_abort(tmp_path):
    from pubmedqa_abst.runner import MAX_CONSECUTIVE_ERRORS

    client = FakeClient(error_calls=set(range(1, MAX_CONSECUTIVE_ERRORS + 1)))
    with pytest.raises(RuntimeError, match="consecutive"):
        go(tmp_path, client)
    assert len(client.calls) == MAX_CONSECUTIVE_ERRORS


def test_generation_source_covers_prompt_and_request_code():
    from pubmedqa_abst.runner import generation_source

    src = generation_source()
    assert "def build_prompt" in src and "ANSWER_SCHEMA" in src and "def parse_output" in src
