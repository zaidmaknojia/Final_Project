"""Rule-based confident-error categories (spec §10). Frozen by the evaluation lock."""

from __future__ import annotations

import random
import re
from collections import defaultdict

from .data import Item
from .scores import ItemScore

NEGATION = re.compile(r"\b(not|no|without|never|non)\b", re.IGNORECASE)
COMPARATIVE = re.compile(r"\b(more|less|better|worse|than|versus|vs\.?|compared|differ)\b", re.IGNORECASE)
NUMERIC = re.compile(r"\d")
_EPS = 1e-9


def features(item: Item) -> dict:
    q = item.question
    return {
        "negation": bool(NEGATION.search(q)),
        "comparative": bool(COMPARATIVE.search(q)),
        "numeric": bool(NUMERIC.search(q)),
    }


def _accepts(value: float | None, threshold: float | None) -> bool:
    return value is not None and threshold is not None and value >= threshold - _EPS


def categorize(item_score: ItemScore, item: Item, other_score_accepts: bool) -> dict:
    return {
        "item_id": item_score.item_id,
        "condition": item_score.condition,
        "gold": item_score.gold,
        "a0": item_score.a0,
        "s_verb": item_score.s_verb,
        "s_cons": item_score.s_cons,
        "confusion": f"{item_score.gold}->{item_score.a0}",
        "scores_agree": other_score_accepts,
        **features(item),
        "question": item.question,
    }


def confident_errors(scores: list[ItemScore], items: dict[str, Item], thresholds: dict,
                     score_name: str) -> list[dict]:
    """Every wrong, valid item accepted by `score_name` at its frozen threshold."""
    other = "s_cons" if score_name == "s_verb" else "s_verb"
    rows = []
    for s in scores:
        if not s.valid or s.correct:
            continue
        if not _accepts(getattr(s, score_name), thresholds.get(score_name)):
            continue
        other_ok = _accepts(getattr(s, other), thresholds.get(other))
        rows.append({"score": score_name, **categorize(s, items[s.item_id], other_ok)})
    return rows


def sample_examples(rows: list[dict], per_category: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["confusion"]].append(r)
    out = []
    for key in sorted(groups):
        group = groups[key]
        out.extend(rng.sample(group, min(per_category, len(group))))
    return out
