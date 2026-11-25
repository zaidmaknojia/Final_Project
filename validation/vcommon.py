"""Shared helpers for the independent validation scripts (read-only).

Everything here is written from scratch with the standard library, numpy and
scipy. Nothing is imported from src/pubmedqa_abst, so each number is
recomputed from the raw files rather than re-read from the pipeline. Scripts
compare what they compute with the report macros in reports/shared/numbers.tex
and print PASS or FAIL per number.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9
LABELS = ("yes", "no", "maybe")
CONDS = ("E+", "E-")
TALLY = Counter()


# ---------- files ----------

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text())


def items() -> dict[str, dict]:
    """PQA-L items: question/contexts stripped, gold lowercased (as the pipeline reads them)."""
    raw = load_json("data/raw/ori_pqal.json")
    return {pid: {"question": r["QUESTION"].strip(),
                  "contexts": [c.strip() for c in r["CONTEXTS"]],
                  "long_answer": r["LONG_ANSWER"],
                  "gold": r["final_decision"].strip().lower()} for pid, r in raw.items()}


def split_ids() -> dict[str, list[str]]:
    m = load_json("data/manifest.json")
    return {"dev": list(m["development_ids"]), "test": list(m["test_ids"])}


def run_dirs() -> dict[tuple[str, str], Path]:
    """(split, condition) -> run directory, read from each run's manifest."""
    out = {}
    for mf in sorted(ROOT.glob("runs/*/manifest.json")):
        m = json.loads(mf.read_text())
        if m.get("split") in ("dev", "test"):
            out[(m["split"], m["condition"])] = mf.parent
    return out


def raw_rows(split: str, cond: str) -> list[dict]:
    path = run_dirs()[(split, cond)] / "raw.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(rows: list[dict]) -> dict[str, dict]:
    """Last line per key wins (the run file is append-only)."""
    out = {}
    for r in rows:
        out[r["key"]] = r
    return out


@dataclass(frozen=True)
class Scores:
    ids: list
    gold: np.ndarray
    a0: np.ndarray
    correct: np.ndarray
    s_verb: np.ndarray
    s_cons: np.ndarray
    samples: dict


def scores(split: str, cond: str, K: int = 10) -> Scores:
    """Per-item greedy answer, stated confidence and agreement, sorted by item id."""
    gold_by_id = {k: v["gold"] for k, v in items().items()}
    rows = latest(raw_rows(split, cond))
    wanted = set(split_ids()[split])
    by_item: dict[str, dict[int, dict]] = {}
    for r in rows.values():
        if r["item_id"] in wanted:
            by_item.setdefault(r["item_id"], {})[r["call_idx"]] = r
    ids = sorted(by_item)
    a0, verb, cons, samples = [], [], [], {}
    for i in ids:
        calls = by_item[i]
        g = calls[0]
        assert g["status"] == "ok", (i, g["status"])
        sampled = [calls[k]["answer"] for k in range(1, K + 1)]
        samples[i] = sampled
        a0.append(g["answer"])
        verb.append(g["confidence"] / 100)
        cons.append(sum(a == g["answer"] for a in sampled) / K)
    gold = np.array([gold_by_id[i] for i in ids])
    a0 = np.array(a0)
    return Scores(ids, gold, a0, (a0 == gold).astype(int), np.array(verb), np.array(cons), samples)


def isotonic_map() -> tuple[np.ndarray, np.ndarray]:
    iso = load_json("freeze/isotonic.json")
    return np.array(iso["x"]), np.array(iso["y"])


def apply_iso(conf: np.ndarray) -> np.ndarray:
    x, y = isotonic_map()
    return np.interp(conf, x, y)  # np.interp clamps outside [x0, xn], like out_of_bounds="clip"


# ---------- metrics (own implementations) ----------

def auroc(conf: np.ndarray, correct: np.ndarray) -> float | None:
    """Mann-Whitney AUROC with average ranks for ties; None if one class is absent."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, int)
    n1 = int(correct.sum())
    n0 = correct.size - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(conf)
    return float((ranks[correct == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum(np.floor(conf * bins + EPS).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / conf.size * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under risk-coverage: step at each unique threshold, high to low, ties together."""
    conf, wrong = np.asarray(conf, float), 1 - np.asarray(correct, float)
    n, total, prev = conf.size, 0.0, 0.0
    for t in np.unique(conf)[::-1]:
        acc = conf >= t
        cov = acc.sum() / n
        total += (cov - prev) * wrong[acc].mean()
        prev = cov
    return float(total)


def policy(conf: np.ndarray, correct: np.ndarray, t: float) -> dict:
    """Answer when conf >= t; coverage, accuracy among answered, confident-error rate."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, int)
    acc = conf >= t - EPS
    return {"coverage": float(acc.mean()),
            "accuracy": float(correct[acc].mean()) if acc.any() else None,
            "cer": float((acc & (correct == 0)).sum() / conf.size)}


def fit_threshold(conf: np.ndarray, n_total: int, target: float) -> tuple[float, float]:
    """Largest unique t whose coverage (conf >= t) / n_total reaches the target."""
    conf = np.asarray(conf, float)
    for t in np.unique(conf)[::-1]:
        cov = (conf >= t).sum() / n_total
        if cov >= target:
            return float(t), float(cov)
    t = float(conf.min())
    return t, float((conf >= t).sum() / n_total)


def macro_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    f1 = []
    for lab in LABELS:
        tp = int(((pred == lab) & (gold == lab)).sum())
        fp = int(((pred == lab) & (gold != lab)).sum())
        fn = int(((pred != lab) & (gold == lab)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1))


# ---------- comparison with the report macros ----------

def macros() -> dict[str, str]:
    text = (ROOT / "reports/shared/numbers.tex").read_text()
    out = {}
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text):
        out[name] = val.replace(r"\textminus{}", "-").replace("{,}", ",")
    return out


MACROS = macros()


def _fmt(x: float, digits: int) -> str:
    text = f"{abs(x):.{digits}f}"
    return ("-" if x < 0 and float(text) != 0 else "") + text


def _digits(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def check(name: str, value, note: str = "") -> bool:
    """Compare a recomputed value with macro `name`, at the macro's own precision."""
    shown = MACROS.get(name)
    if shown is None:
        ok, got = False, "(macro missing)"
    elif isinstance(value, str):
        ok, got = value == shown, value
    elif re.fullmatch(r"-?[\d,]+", shown):
        ok, got = int(shown.replace(",", "")) == int(value), str(value)
    else:
        got = _fmt(float(value), _digits(shown))
        ok = got == shown
    TALLY["pass" if ok else "fail"] += 1
    detail = f"{value:.6f}" if isinstance(value, float) else str(value)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<38} report={shown!s:<10} recomputed={detail:<12} {note}")
    return ok


def check_ci(name: str, lo: float, hi: float) -> bool:
    shown = MACROS.get(name, "")
    parts = re.findall(r"-?\d+\.\d+", shown)
    d = _digits(parts[0]) if parts else 3
    got = f"[{_fmt(lo, d)}, {_fmt(hi, d)}]"
    ok = got == shown
    TALLY["pass" if ok else "fail"] += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name:<38} report={shown:<18} recomputed={got}")
    return ok


def fact(label: str, ok: bool, detail: str = "") -> bool:
    """A yes/no check that has no macro (counts, hashes, orderings)."""
    TALLY["pass" if ok else "fail"] += 1
    print(f"{'PASS' if ok else 'FAIL'}  {label}  {detail}")
    return ok


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def finish() -> None:
    print(f"\n{TALLY['pass']} passed, {TALLY['fail']} failed")
    sys.exit(1 if TALLY["fail"] else 0)
