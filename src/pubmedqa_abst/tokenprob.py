"""Token-probability baseline: post hoc and exploratory (added Nov 23, after the test run).

Neither lock covers this module or its data. One extra call per item repeats
the locked greedy call exactly (same prompt, JSON schema, temperature 0, seed
and token cap) with Ollama's log-probabilities switched on. The score was fixed
before any of these calls were made: the probability of the model's locked
greedy label at the answer token, renormalised over the three labels,

    s_tok = p(A0) / (p(yes) + p(no) + p(maybe)).

Run:    python -m pubmedqa_abst.tokenprob dev test
Output: runs/tokenprob/<split>_<condition>.jsonl (append-only, resumable)
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import requests

from .client import ANSWER_SCHEMA, NUM_PREDICT
from .data import Item, build_prompt
from .parse import parse_output
from .scores import ItemScore

log = logging.getLogger("pubmedqa_abst.tokenprob")

LABELS = ("yes", "no", "maybe")
TOP_LOGPROBS = 20  # Ollama's maximum; the three labels always fit
OK = "ok"
NO_ANSWER_TOKEN = "no_answer_token"
TRANSPORT_ERROR = "transport_error"
ANSWER_KEY = "answer"


@dataclass(frozen=True)
class AnswerToken:
    chosen: str
    label_logprobs: dict[str, float]


def find_answer_token(logprobs: list[dict]) -> AnswerToken | None:
    """The first token after the "answer" key that is exactly a label, with the
    log-probabilities of the labels among that position's alternatives."""
    after_key = False
    for entry in logprobs:
        token = entry.get("token", "")
        if not after_key:
            after_key = ANSWER_KEY in token
            continue
        if token in LABELS:
            found: dict[str, float] = {}
            for alt in entry.get("top_logprobs") or []:
                if alt.get("token") in LABELS and alt["token"] not in found:
                    found[alt["token"]] = float(alt["logprob"])
            return AnswerToken(token, {token: float(entry["logprob"]), **found})
    return None


def label_probs(label_logprobs: dict[str, float]) -> dict[str, float]:
    """Probabilities over yes/no/maybe, renormalised; a label outside the top list gets 0."""
    raw = {lbl: math.exp(label_logprobs[lbl]) if lbl in label_logprobs else 0.0 for lbl in LABELS}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("no label log-probabilities to normalise")
    return {lbl: p / total for lbl, p in raw.items()}


def make_row(item_id: str, condition: str, split: str, result: dict) -> dict:
    base = {"item_id": item_id, "condition": condition, "split": split}
    if result.get("error"):
        return {**base, "status": TRANSPORT_ERROR, "error": result["error"]}
    parsed = parse_output(result.get("response"))
    row = {**base, "text": result.get("response"), "answer": parsed.answer,
           "confidence": parsed.confidence, "parse_status": parsed.status}
    token = find_answer_token(result.get("logprobs") or [])
    if token is None:
        return {**row, "status": NO_ANSWER_TOKEN}
    return {**row, "status": OK, "answer_token": token.chosen, "label_logprobs": token.label_logprobs,
            "label_mass": sum(math.exp(v) for v in token.label_logprobs.values()),
            "probs": label_probs(token.label_logprobs)}


class LogprobClient:
    """The locked greedy request (client.OllamaClient.generate) plus log-probabilities."""

    def __init__(self, url: str, model: str, seed: int, timeout_s: float, max_retries: int,
                 session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.seed = seed
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.sleep = sleep

    def body(self, prompt: str) -> dict:
        return {
            "model": self.model, "prompt": prompt, "stream": False, "think": False, "format": ANSWER_SCHEMA,
            "options": {"temperature": 0, "seed": self.seed, "num_predict": NUM_PREDICT},
            "logprobs": True, "top_logprobs": TOP_LOGPROBS,
        }

    def generate(self, prompt: str) -> dict:
        """Never raises on transport failure; returns {"error": ...} after the retries."""
        last_error = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = self.session.post(f"{self.url}/api/generate", json=self.body(prompt), timeout=self.timeout_s)
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= self.max_retries:
                    self.sleep(2**attempt)
                continue
            return {"response": payload.get("response"), "logprobs": payload.get("logprobs") or []}
        return {"error": last_error}


def load_rows(path: Path) -> dict[str, dict]:
    """Latest row per item (later lines win, so a retried item replaces its error)."""
    path = Path(path)
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["item_id"]] = row
    return rows


def run_tokenprob(items: dict[str, Item], ids: list[str], condition: str, split: str, template: str,
                  generate: Callable[[str], dict], out_path: Path) -> int:
    """Call once per unfinished item and append its row; returns the number of calls made."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {i for i, r in load_rows(out_path).items() if r["status"] != TRANSPORT_ERROR}
    todo = [i for i in ids if i not in done]
    with out_path.open("a") as fh:
        for n, item_id in enumerate(todo, 1):
            row = make_row(item_id, condition, split, generate(build_prompt(items[item_id], condition, template)))
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 50 == 0:
                log.info("%s %s: %d/%d", split, condition, n, len(todo))
    return len(todo)


def token_scores(scores: list[ItemScore], rows: dict[str, dict]) -> np.ndarray:
    """s_tok per item, aligned with `scores`; NaN where there is no usable row or no A0."""
    def one(s: ItemScore) -> float:
        row = rows.get(s.item_id)
        if row is None or row.get("status") != OK or s.a0 not in LABELS:
            return math.nan
        return float(row["probs"][s.a0])

    return np.array([one(s) for s in scores], float)


def agreement(scores: list[ItemScore], rows: dict[str, dict]) -> dict:
    """How the new call's rows line up with the locked greedy answers."""
    got = [rows.get(s.item_id) for s in scores]
    status = [r["status"] if r else None for r in got]
    return {
        "n": len(scores),
        "ok": status.count(OK),
        "missing": status.count(None),
        "no_answer_token": status.count(NO_ANSWER_TOKEN),
        "transport_error": status.count(TRANSPORT_ERROR),
        "answer_matches_a0": sum(1 for s, r in zip(scores, got) if r and r["status"] == OK and r["answer"] == s.a0),
    }


def out_path(runs_dir: Path, split: str, condition: str) -> Path:
    return Path(runs_dir) / "tokenprob" / f"{split}_{condition}.jsonl"


def main(splits: list[str]) -> None:  # pragma: no cover - thin wiring over tested functions
    from .cli import CONDITIONS, Context, build_parser

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ctx = Context(build_parser().parse_args(["evaluate"]))
    cfg = ctx.config
    ctx.client().preflight(cfg["model_digest"])
    items, split_ids = ctx.data()
    client = LogprobClient(cfg["ollama_url"], cfg["model"], cfg["greedy_seed"], cfg["timeout_s"], cfg["max_retries"])
    for split in splits:
        for cond in CONDITIONS:
            path = out_path(ctx.runs, split, cond)
            n = run_tokenprob(items, split_ids[split], cond, split, ctx.template, client.generate, path)
            log.info("%s %s: %d new calls -> %s", split, cond, n, path)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:] or ["dev", "test"])
