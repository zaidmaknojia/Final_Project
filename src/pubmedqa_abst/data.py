"""Load PubMedQA PQA-L, verify checksums, build the official split and prompts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

LABELS = ("yes", "no", "maybe")
CONDITIONS = ("E+", "E-")
TOTAL_ITEMS = 1000


@dataclass(frozen=True)
class Item:
    id: str
    question: str
    contexts: tuple[str, ...]
    gold: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(manifest_path: Path) -> dict:
    return json.loads(Path(manifest_path).read_text())


def verify_checksums(raw_dir: Path, manifest_path: Path) -> None:
    """Raise ValueError if any raw file differs from the manifest's SHA-256."""
    expected = _read_manifest(manifest_path)["files"]
    for name, digest in expected.items():
        path = Path(raw_dir) / name
        if not path.exists():
            raise ValueError(f"checksum: missing data file {path}")
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {name}: {actual} != {digest}")


def load_items(raw_dir: Path, manifest_path: Path) -> dict[str, Item]:
    """Return all 1,000 PQA-L items keyed by PubMed ID, after checksum verification."""
    verify_checksums(raw_dir, manifest_path)
    raw = json.loads((Path(raw_dir) / "ori_pqal.json").read_text())
    items = {
        iid: Item(
            id=iid,
            question=rec["QUESTION"].strip(),
            contexts=tuple(c.strip() for c in rec["CONTEXTS"]),
            gold=rec["final_decision"].strip().lower(),
        )
        for iid, rec in raw.items()
    }
    bad = [i.id for i in items.values() if i.gold not in LABELS]
    if bad:
        raise ValueError(f"unexpected gold labels for ids {bad[:5]}")
    return items


def load_split(raw_dir: Path, manifest_path: Path) -> dict[str, list[str]]:
    """Official test IDs vs. the manifest's development IDs; checks disjoint and complete."""
    verify_checksums(raw_dir, manifest_path)
    test_ids = sorted(json.loads((Path(raw_dir) / "test_ground_truth.json").read_text()))
    dev_ids = sorted(_read_manifest(manifest_path)["development_ids"])
    overlap = set(dev_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"dev/test overlap: {sorted(overlap)[:5]}")
    if len(set(dev_ids) | set(test_ids)) != TOTAL_ITEMS:
        raise ValueError("dev and test do not cover 1,000 items")
    return {"dev": dev_ids, "test": test_ids}


def build_prompt(item: Item, condition: Literal["E+", "E-"], template: str) -> str:
    """E+ adds the abstract's context passages; E- gives the question alone.

    LONG_ANSWER is never stored on Item, so it cannot leak into a prompt.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if condition == "E+":
        evidence = "Abstract excerpts:\n" + "\n".join(item.contexts) + "\n"
    else:
        evidence = ""
    return template.format(question=item.question, evidence_block=evidence)


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
