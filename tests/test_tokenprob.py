import json
import math

import numpy as np
import pytest
import requests

from pubmedqa_abst.client import ANSWER_SCHEMA, NUM_PREDICT, OllamaClient
from pubmedqa_abst.data import Item
from pubmedqa_abst.scores import ItemScore
from pubmedqa_abst.tokenprob import (
    LABELS, NO_ANSWER_TOKEN, TOP_LOGPROBS, TRANSPORT_ERROR, LogprobClient, agreement,
    find_answer_token, label_probs, load_rows, make_row, run_tokenprob, token_scores,
)


def _tok(token, logprob=0.0, top=()):
    return {"token": token, "logprob": logprob, "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top]}


# Shape of Ollama 0.20.2's reply in the Nov 23 probe (answer token at index 4).
PROBE = [
    _tok('{"', -0.9, [("```", -0.5), ('{"', -0.9)]),
    _tok("answer"), _tok('":'), _tok(' "'),
    _tok("no", -0.193, [("no", -0.193), ("maybe", -1.744), ("yes", -7.699), ("No", -9.0)]),
    _tok('",'), _tok(' "'), _tok("confidence"), _tok('":'), _tok(" "), _tok("9"), _tok("0"), _tok("}"),
]
PROBE_TEXT = '{"answer": "no", "confidence": 90}'


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def post(self, url, json, timeout):
        self.bodies.append(json)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


def test_find_answer_token_reads_label_alternatives_at_the_answer_position():
    token = find_answer_token(PROBE)
    assert token.chosen == "no"
    assert token.label_logprobs == {"no": -0.193, "maybe": -1.744, "yes": -7.699}


def test_find_answer_token_ignores_label_tokens_before_the_answer_key():
    early = [_tok("yes", -0.1, [("yes", -0.1)]), *PROBE]
    assert find_answer_token(early).chosen == "no"


def test_find_answer_token_returns_none_without_a_label_after_the_key():
    assert find_answer_token([_tok('{"'), _tok("answer"), _tok('":'), _tok(' "'), _tok("may")]) is None
    assert find_answer_token([]) is None


def test_find_answer_token_keeps_the_chosen_label_when_it_is_missing_from_the_top_list():
    token = find_answer_token([_tok("answer"), _tok("maybe", -2.0, [("yes", -0.5)])])
    assert token.label_logprobs == {"yes": -0.5, "maybe": -2.0}


def test_label_probs_renormalises_over_the_three_labels_and_zeroes_missing_ones():
    probs = label_probs({"no": math.log(0.6), "maybe": math.log(0.2)})
    assert probs == pytest.approx({"yes": 0.0, "no": 0.75, "maybe": 0.25})
    assert set(probs) == set(LABELS)


def test_label_probs_rejects_an_empty_set():
    with pytest.raises(ValueError):
        label_probs({})


def test_make_row_records_parsed_answer_and_label_probabilities():
    row = make_row("123", "E+", "dev", {"response": PROBE_TEXT, "logprobs": PROBE})
    assert row["status"] == "ok"
    assert (row["item_id"], row["condition"], row["split"]) == ("123", "E+", "dev")
    assert (row["answer"], row["confidence"], row["answer_token"]) == ("no", 90, "no")
    assert sum(row["probs"].values()) == pytest.approx(1.0)
    assert row["probs"]["no"] == pytest.approx(math.exp(-0.193) / row["label_mass"])
    assert row["label_mass"] == pytest.approx(sum(math.exp(v) for v in (-0.193, -1.744, -7.699)))


def test_make_row_flags_missing_answer_token_and_transport_errors():
    assert make_row("1", "E+", "dev", {"response": PROBE_TEXT, "logprobs": []})["status"] == NO_ANSWER_TOKEN
    err = make_row("1", "E+", "dev", {"error": "ConnectionError: refused"})
    assert err["status"] == TRANSPORT_ERROR and err["error"].startswith("ConnectionError")


def test_logprob_call_is_the_locked_greedy_call_plus_logprob_fields():
    reply = {"response": PROBE_TEXT, "logprobs": PROBE}
    locked, mine = FakeSession([reply]), FakeSession([reply])
    OllamaClient("http://x", "gemma4:e4b", 5, 0, session=locked).generate("P", 0, 20261121, ANSWER_SCHEMA)
    out = LogprobClient("http://x", "gemma4:e4b", 20261121, 5, 0, session=mine).generate("P")
    body = dict(mine.bodies[0])
    assert body.pop("logprobs") is True and body.pop("top_logprobs") == TOP_LOGPROBS
    assert body == locked.bodies[0]
    assert body["options"] == {"temperature": 0, "seed": 20261121, "num_predict": NUM_PREDICT}
    assert out == {"response": PROBE_TEXT, "logprobs": PROBE}


def test_logprob_client_retries_then_reports_the_error():
    session = FakeSession([requests.ConnectionError("down"), requests.ConnectionError("down")])
    out = LogprobClient("http://x", "m", 1, 5, 1, session=session, sleep=lambda s: None).generate("P")
    assert out["error"].startswith("ConnectionError") and len(session.bodies) == 2


ITEMS = {i: Item(id=i, question=f"Q{i}?", contexts=(f"C{i}",), gold="yes")
         for i in ("a", "b", "c")}
TEMPLATE = "{evidence_block}Question: {question}\n"


def test_run_tokenprob_appends_rows_and_resumes_skipping_finished_items(tmp_path):
    out = tmp_path / "tok" / "dev_E+.jsonl"
    out.parent.mkdir()
    out.write_text(json.dumps({"item_id": "a", "status": "ok"}) + "\n"
                   + json.dumps({"item_id": "b", "status": TRANSPORT_ERROR}) + "\n")
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return {"response": PROBE_TEXT, "logprobs": PROBE}

    n = run_tokenprob(ITEMS, ["a", "b", "c"], "E+", "dev", TEMPLATE, generate, out)
    assert n == 2 and prompts == ["Abstract excerpts:\nCb\nQuestion: Qb?\n", "Abstract excerpts:\nCc\nQuestion: Qc?\n"]
    rows = load_rows(out)
    assert {k: r["status"] for k, r in rows.items()} == {"a": "ok", "b": "ok", "c": "ok"}
    assert run_tokenprob(ITEMS, ["a", "b", "c"], "E+", "dev", TEMPLATE, generate, out) == 0


def _s(iid, a0, gold="yes"):
    return ItemScore(item_id=iid, condition="E+", gold=gold, a0=a0, correct=a0 == gold,
                     s_verb=0.9, s_cons=1.0, valid=a0 is not None, invalid_samples=0)


def test_token_scores_take_the_probability_of_the_locked_greedy_label():
    rows = {"a": {"status": "ok", "answer": "yes", "probs": {"yes": 0.7, "no": 0.2, "maybe": 0.1}},
            "b": {"status": "ok", "answer": "no", "probs": {"yes": 0.3, "no": 0.6, "maybe": 0.1}},
            "c": {"status": NO_ANSWER_TOKEN}}
    out = token_scores([_s("a", "yes"), _s("b", "no"), _s("c", "yes"), _s("d", "yes"), _s("e", None)], rows)
    assert out[:2] == pytest.approx([0.7, 0.6])
    assert np.isnan(out[2:]).all()


def test_agreement_counts_statuses_and_matches_with_the_locked_answer():
    rows = {"a": {"status": "ok", "answer": "yes"}, "b": {"status": "ok", "answer": "maybe"},
            "c": {"status": NO_ANSWER_TOKEN, "answer": "yes"}}
    out = agreement([_s("a", "yes"), _s("b", "no"), _s("c", "yes"), _s("d", "yes")], rows)
    assert out == {"n": 4, "ok": 2, "missing": 1, "no_answer_token": 1, "transport_error": 0,
                   "answer_matches_a0": 1}
