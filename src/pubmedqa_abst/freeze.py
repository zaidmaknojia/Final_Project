"""Protocol and evaluation locks that gate the single test run (spec §11).

protocol.lock.json   written after the timing pilot: prompt, parser, config.
evaluation.lock.json written after the dev fit: thresholds, isotonic map,
                     error-category scheme, plus the protocol lock itself.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_PKG = "src/pubmedqa_abst/"
# Everything that shapes a model request: config, prompt, prompt building,
# request options/schema, sampling seeds, parsing.
PROTOCOL_FILES = ("configs/run.yaml", "configs/prompts/v1.txt",
                  *(_PKG + m for m in ("parse.py", "data.py", "client.py", "runner.py")))
# Everything that turns raw logs into reported numbers.
EVAL_ROOT_FILES = tuple(_PKG + m for m in (
    "error_analysis.py", "scores.py", "metrics.py", "thresholds.py", "recalibrate.py",
    "bootstrap.py", "stats.py", "evaluate.py"))
EVAL_FREEZE_FILES = ("thresholds.json", "isotonic.json", "protocol.lock.json")
PROTOCOL_LOCK = "protocol.lock.json"
EVALUATION_LOCK = "evaluation.lock.json"

PLANNED_COMPARISONS = [
    "cons_vs_verb",
    "threshold_vs_accept_all",
    "eplus_vs_eminus",
    "iso_vs_raw_verb",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_protocol_lock(root: Path, freeze_dir: Path, config: dict) -> Path:
    freeze_dir = Path(freeze_dir)
    freeze_dir.mkdir(parents=True, exist_ok=True)
    lock = {
        "created": _now(),
        "files": {f"root:{rel}": _sha(Path(root) / rel) for rel in PROTOCOL_FILES},
        "generation": {k: config[k] for k in ("model", "model_digest", "K", "T", "greedy_seed")},
        "analysis": {k: config[k] for k in ("ece_bins", "bootstrap_B", "bootstrap_seed", "coverage_targets")},
        "planned_comparisons": PLANNED_COMPARISONS,
    }
    path = freeze_dir / PROTOCOL_LOCK
    path.write_text(json.dumps(lock, indent=2))
    return path


def write_evaluation_lock(root: Path, freeze_dir: Path) -> Path:
    freeze_dir = Path(freeze_dir)
    files = {f"root:{rel}": _sha(Path(root) / rel) for rel in EVAL_ROOT_FILES}
    files.update({f"freeze:{rel}": _sha(freeze_dir / rel) for rel in EVAL_FREEZE_FILES})
    path = freeze_dir / EVALUATION_LOCK
    path.write_text(json.dumps({"created": _now(), "files": files}, indent=2))
    return path


def _resolve(entry: str, root: Path, freeze_dir: Path) -> Path:
    scope, rel = entry.split(":", 1)
    return (Path(root) if scope == "root" else Path(freeze_dir)) / rel


def verify_locks(root: Path, freeze_dir: Path) -> None:
    """Raise RuntimeError naming every missing lock or changed file."""
    problems = []
    for name in (PROTOCOL_LOCK, EVALUATION_LOCK, "thresholds.json"):
        if not (Path(freeze_dir) / name).exists():
            problems.append(f"missing {name}")
    if problems:
        raise RuntimeError("lock check failed: " + "; ".join(problems))
    for name in (PROTOCOL_LOCK, EVALUATION_LOCK):
        lock = json.loads((Path(freeze_dir) / name).read_text())
        for entry, digest in lock["files"].items():
            path = _resolve(entry, root, freeze_dir)
            if not path.exists():
                problems.append(f"{entry} missing")
            elif _sha(path) != digest:
                problems.append(f"{entry} changed since {name}")
    if problems:
        raise RuntimeError("lock check failed: " + "; ".join(problems))
