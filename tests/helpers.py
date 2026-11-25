"""Shared test builders for raw.jsonl fixtures."""

import json
from pathlib import Path


def raw_row(item_id, call_idx, status="ok", answer="yes", confidence=90, condition="E+"):
    ok_like = status in ("ok", "invalid_confidence")
    return {
        "key": f"{item_id}|{call_idx}",
        "item_id": item_id,
        "condition": condition,
        "call_idx": call_idx,
        "status": status,
        "answer": answer if ok_like else None,
        "confidence": confidence if status == "ok" else None,
        "error": "ConnectionError" if status == "transport_error" else None,
    }


def write_raw(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


ANSWERS = ("yes", "no", "maybe")


class FakeOllama:
    """Deterministic stand-in for OllamaClient; never touches the network."""

    def __init__(self, *args, **kwargs):
        pass

    def preflight(self, expected_digest):
        return {"ollama_version": "fake", "model_digest": expected_digest + "ff"}

    def generate(self, prompt, temperature, seed, schema):
        import hashlib

        from pubmedqa_abst.client import CallResult

        h = int(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest(), 16)
        answer = ANSWERS[h % 3] if temperature else ANSWERS[len(prompt) % 3]
        return CallResult(json.dumps({"answer": answer, "confidence": 40 + h % 60}), 0, 0.01, 1, None)
