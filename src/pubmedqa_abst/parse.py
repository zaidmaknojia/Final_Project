"""Strict parser for the model's JSON answer. No regex fallback (executor review #1)."""

from __future__ import annotations

import json
from dataclasses import dataclass

VALID_ANSWERS = frozenset({"yes", "no", "maybe"})

OK = "ok"
INVALID_FORMAT = "invalid_format"
INVALID_CONFIDENCE = "invalid_confidence"


@dataclass(frozen=True)
class Parsed:
    answer: str | None
    confidence: int | None
    status: str


def _parse_confidence(value: object) -> int | None:
    # bool is a subclass of int in Python, so reject it first.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    else:
        return None
    return number if 0 <= number <= 100 else None


def parse_output(text: str | None) -> Parsed:
    """Parse `{"answer": yes|no|maybe, "confidence": int 0-100}` strictly."""
    if not text:
        return Parsed(None, None, INVALID_FORMAT)
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return Parsed(None, None, INVALID_FORMAT)
    if not isinstance(obj, dict):
        return Parsed(None, None, INVALID_FORMAT)
    answer = obj.get("answer")
    if not isinstance(answer, str) or answer.strip().lower() not in VALID_ANSWERS:
        return Parsed(None, None, INVALID_FORMAT)
    answer = answer.strip().lower()
    confidence = _parse_confidence(obj.get("confidence"))
    if confidence is None:
        return Parsed(answer, None, INVALID_CONFIDENCE)
    return Parsed(answer, confidence, OK)
