"""Per-item confidence scores on the fixed greedy answer A0 (spec §6)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .data import Item
from .runner import latest_by_key

VALID_STATUSES = frozenset({"ok", "invalid_confidence"})


@dataclass(frozen=True)
class ItemScore:
    item_id: str
    condition: str
    gold: str
    a0: str | None
    correct: bool
    s_verb: float | None
    s_cons: float | None
    valid: bool
    invalid_samples: int


def _score_item(item: Item, rows: dict[int, dict], k: int) -> ItemScore:
    greedy = rows.get(0)
    has_answer = greedy is not None and greedy["status"] in VALID_STATUSES
    a0 = greedy["answer"] if has_answer else None
    confidence = greedy.get("confidence") if has_answer else None
    samples = [rows.get(i) for i in range(1, k + 1)]
    invalid = sum(1 for r in samples if r is not None and r["status"] not in VALID_STATUSES)
    matches = sum(1 for r in samples if r is not None and r["status"] in VALID_STATUSES and r["answer"] == a0)
    s_cons = matches / k if (a0 is not None and k > 0) else None
    condition = next(iter(rows.values()))["condition"]
    return ItemScore(
        item_id=item.id,
        condition=condition,
        gold=item.gold,
        a0=a0,
        correct=a0 is not None and a0 == item.gold,
        s_verb=confidence / 100 if confidence is not None else None,
        s_cons=s_cons,
        valid=a0 is not None,
        invalid_samples=invalid,
    )


def score_run(raw_path: Path, items: dict[str, Item], K: int) -> list[ItemScore]:
    """Score every item that appears in the run, using the latest line per key."""
    by_item: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in latest_by_key(raw_path).values():
        by_item[row["item_id"]][row["call_idx"]] = row
    return [_score_item(items[iid], by_item[iid], K) for iid in sorted(by_item)]


def incomplete_keys(raw_path: Path, expected_ids: list[str], K: int) -> list[str]:
    """Expected `item|call_idx` keys whose latest record is absent or a transport error."""
    latest = latest_by_key(raw_path)
    missing = []
    for iid in expected_ids:
        for call_idx in range(K + 1):
            row = latest.get(f"{iid}|{call_idx}")
            if row is None or row["status"] == "transport_error":
                missing.append(f"{iid}|{call_idx}")
    return missing


def require_complete(raw_path: Path, expected_ids: list[str], K: int) -> None:
    """Refuse to score a run with missing calls or unresolved transport errors (spec §6)."""
    missing = incomplete_keys(raw_path, expected_ids, K)
    if missing:
        total = len(expected_ids) * (K + 1)
        raise RuntimeError(
            f"incomplete run {raw_path}: {len(missing)} of {total} calls missing or failed "
            f"(first: {missing[:3]}); rerun the same `run` command to resume")
