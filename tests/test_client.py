import pytest
import requests

from pubmedqa_abst import client as client_mod
from pubmedqa_abst.client import ANSWER_SCHEMA, OllamaClient


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    """Scripted session: each entry is a payload dict or an exception to raise."""

    def __init__(self, post_script=(), get_map=None):
        self.post_script = list(post_script)
        self.get_map = get_map or {}
        self.posted = []

    def post(self, url, json, timeout):
        self.posted.append(json)
        step = self.post_script.pop(0)
        if isinstance(step, Exception):
            raise step
        return FakeResponse(step)

    def get(self, url, timeout):
        for suffix, step in self.get_map.items():
            if url.endswith(suffix):
                if isinstance(step, Exception):
                    raise step
                return FakeResponse(step)
        raise AssertionError(f"unexpected GET {url}")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)


def make(session):
    return OllamaClient("http://x", "gemma4:e4b", timeout_s=5, max_retries=3, session=session)


def test_generate_returns_response_field_and_thinking_len():
    s = FakeSession([{"response": '{"answer":"yes","confidence":80}', "thinking": "abc"}])
    r = make(s).generate("p", 0.0, 1, ANSWER_SCHEMA)
    assert r.text == '{"answer":"yes","confidence":80}'
    assert r.thinking_len == 3 and r.error is None and r.attempts == 1


def test_generate_retries_then_succeeds():
    s = FakeSession([requests.ConnectionError("x"), requests.ConnectionError("x"), {"response": "{}"}])
    r = make(s).generate("p", 0.0, 1, ANSWER_SCHEMA)
    assert r.attempts == 3 and r.error is None and r.text == "{}"


def test_generate_final_transport_error():
    s = FakeSession([requests.ConnectionError("x")] * 4)
    r = make(s).generate("p", 0.0, 1, ANSWER_SCHEMA)
    assert r.text is None and r.error and r.attempts == 4


def test_preflight_fails_when_server_down():
    s = FakeSession(get_map={"/api/version": requests.ConnectionError("down")})
    with pytest.raises(RuntimeError, match="Ollama"):
        make(s).preflight("c6eb396dbd59")


def test_preflight_rejects_wrong_digest():
    s = FakeSession(get_map={
        "/api/version": {"version": "0.20.2"},
        "/api/tags": {"models": [{"name": "gemma4:e4b", "digest": "deadbeef0000"}]},
    })
    with pytest.raises(RuntimeError, match="digest"):
        make(s).preflight("c6eb396dbd59")


def test_preflight_rejects_missing_model():
    s = FakeSession(get_map={
        "/api/version": {"version": "0.20.2"},
        "/api/tags": {"models": [{"name": "other:1b", "digest": "c6eb396dbd59aa"}]},
    })
    with pytest.raises(RuntimeError, match="gemma4:e4b"):
        make(s).preflight("c6eb396dbd59")


def test_preflight_ok_returns_env():
    s = FakeSession([{"response": '{"answer":"yes","confidence":50}'}], get_map={
        "/api/version": {"version": "0.20.2"},
        "/api/tags": {"models": [{"name": "gemma4:e4b", "digest": "c6eb396dbd59ffff"}]},
    })
    env = make(s).preflight("c6eb396dbd59")
    assert env == {"ollama_version": "0.20.2", "model_digest": "c6eb396dbd59ffff"}


def test_request_payload_shape():
    s = FakeSession([{"response": "{}"}])
    make(s).generate("hello", 0.7, 42, ANSWER_SCHEMA)
    body = s.posted[0]
    assert body["options"] == {"temperature": 0.7, "seed": 42, "num_predict": 64}
    assert body["format"] == ANSWER_SCHEMA and body["stream"] is False
    assert body["think"] is False and body["prompt"] == "hello" and body["model"] == "gemma4:e4b"


def test_answer_schema_constrains_fields():
    props = ANSWER_SCHEMA["properties"]
    assert props["answer"]["enum"] == ["yes", "no", "maybe"]
    assert props["confidence"] == {"type": "integer", "minimum": 0, "maximum": 100}
    assert set(ANSWER_SCHEMA["required"]) == {"answer", "confidence"}


def test_preflight_warmup_generate_failure_raises():
    s = FakeSession([requests.ConnectionError("x")] * 4, get_map={
        "/api/version": {"version": "0.20.2"},
        "/api/tags": {"models": [{"name": "gemma4:e4b", "digest": "c6eb396dbd59ffff"}]},
    })
    with pytest.raises(RuntimeError, match="warm-up"):
        make(s).preflight("c6eb396dbd59")
