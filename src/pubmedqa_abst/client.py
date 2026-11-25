"""Ollama HTTP client: preflight checks, structured generation, transport retries."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no", "maybe"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["answer", "confidence"],
}
# The JSON answer is ~15 tokens; a cap turns a runaway generation into a
# recorded invalid_format instead of a timeout that retries forever.
NUM_PREDICT = 64
WARMUP_PROMPT = 'Reply with JSON only: {"answer": "yes", "confidence": 50}'


@dataclass(frozen=True)
class CallResult:
    text: str | None
    thinking_len: int
    latency_s: float
    attempts: int
    error: str | None


class OllamaClient:
    def __init__(
        self,
        url: str,
        model: str,
        timeout_s: float,
        max_retries: int,
        session: requests.Session | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = session or requests.Session()

    def _get_json(self, path: str) -> dict:
        resp = self.session.get(f"{self.url}{path}", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def preflight(self, expected_digest: str) -> dict:
        """Fail fast if the server is down or the installed model digest differs."""
        try:
            version = self._get_json("/api/version").get("version")
            models = self._get_json("/api/tags").get("models", [])
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama server not reachable at {self.url}: {exc}") from exc
        match = next((m for m in models if m.get("name") == self.model), None)
        if match is None:
            raise RuntimeError(f"model {self.model} not installed in Ollama")
        digest = match.get("digest", "")
        if not digest.startswith(expected_digest):
            raise RuntimeError(f"model digest {digest[:12]} does not match expected {expected_digest}")
        warmup = self.generate(WARMUP_PROMPT, 0, 0, ANSWER_SCHEMA)
        if warmup.error is not None:
            raise RuntimeError(f"warm-up generate failed: {warmup.error}")
        return {"ollama_version": version, "model_digest": digest}

    def generate(self, prompt: str, temperature: float, seed: int, schema: dict) -> CallResult:
        """One structured generation. Never raises on transport failure; returns `error`."""
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": temperature, "seed": seed, "num_predict": NUM_PREDICT},
        }
        last_error = None
        for attempt in range(1, self.max_retries + 2):
            start = time.monotonic()
            try:
                resp = self.session.post(f"{self.url}/api/generate", json=body, timeout=self.timeout_s)
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= self.max_retries:
                    time.sleep(2**attempt)
                continue
            return CallResult(
                text=payload.get("response"),
                thinking_len=len(payload.get("thinking") or ""),
                latency_s=time.monotonic() - start,
                attempts=attempt,
                error=None,
            )
        return CallResult(None, 0, 0.0, self.max_retries + 1, last_error)
