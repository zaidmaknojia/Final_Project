"""Step 3: the four model runs. Completeness, decoding settings, seeds and the exact prompts."""

import hashlib
import json
from collections import Counter

import numpy as np

from vcommon import ROOT, fact, finish, header, items, latest, raw_rows, run_dirs, split_ids

TEMPLATE = (ROOT / "configs/prompts/v1.txt").read_text()
IT = items()


def prompt(item: dict, cond: str) -> str:
    evidence = "Abstract excerpts:\n" + "\n".join(item["contexts"]) + "\n" if cond == "E+" else ""
    return TEMPLATE.format(question=item["question"], evidence_block=evidence)


def sample_seed(run_id: str, item_id: str, cond: str, k: int) -> int:
    return int(hashlib.sha256(f"{run_id}|{item_id}|{cond}|{k}".encode()).hexdigest(), 16) % 2**31


latencies = []
for (split, cond), d in sorted(run_dirs().items()):
    m = json.loads((d / "manifest.json").read_text())
    header(f"Step 3: {split} {cond}  run {d.name}")
    print(f"model {m['model']}  digest {m['env']['model_digest'][:12]}  ollama {m['env']['ollama_version']}  "
          f"K={m['K']} T={m['T']} greedy_seed={m['greedy_seed']} think={m['options']['think']}")
    rows = raw_rows(split, cond)
    last = latest(rows)
    want = {f"{i}|{k}" for i in split_ids()[split] for k in range(11)}
    fact("5,500 lines, one per item x call (500 x 11)", len(rows) == 5500, str(len(rows)))
    fact("keys are exactly the expected item|call pairs", set(last) == want)
    fact("every call has status ok", Counter(r["status"] for r in rows) == {"ok": 5500})
    fact("every answer is yes/no/maybe with an integer 0-100 confidence",
         all(r["answer"] in ("yes", "no", "maybe") and isinstance(r["confidence"], int)
             and 0 <= r["confidence"] <= 100 for r in rows))
    greedy = [r for r in rows if r["call_idx"] == 0]
    sampled = [r for r in rows if r["call_idx"] > 0]
    fact("greedy calls: temperature 0, seed 20261121",
         all(r["temperature"] == 0 and r["seed"] == 20261121 for r in greedy), str(len(greedy)))
    fact("sampled calls: temperature 0.7, seed = sha256(run|item|cond|k) mod 2^31",
         all(r["temperature"] == 0.7 and r["seed"] == sample_seed(d.name, r["item_id"], cond, r["call_idx"])
             for r in sampled), str(len(sampled)))
    fact("no reasoning tokens (thinking off)", all(r["thinking_len"] == 0 for r in rows))
    fact("prompt_sha256 matches the rebuilt prompt on every call",
         all(r["prompt_sha256"] == hashlib.sha256(prompt(IT[r["item_id"]], cond).encode()).hexdigest()
             for r in rows))
    if cond == "E+":
        fact("E+ prompts never contain the abstract conclusion (LONG_ANSWER)",
             all(IT[i]["long_answer"].strip() not in prompt(IT[i], cond) for i in split_ids()[split]))
    else:
        fact("E- prompts contain no abstract text", all("Abstract excerpts" not in prompt(IT[i], cond)
                                                         for i in split_ids()[split]))
    lat = np.array([r["latency_s"] for r in rows])
    latencies.extend(lat)
    print(f"latency per call: mean {lat.mean():.3f} s, median {np.median(lat):.3f} s; "
          f"retried calls: {sum(r['attempts'] > 1 for r in rows)}")

header("Step 3: totals")
print(f"{len(latencies)} calls, mean latency {np.mean(latencies):.3f} s "
      f"(the reports quote the pilot's 0.75 s, from planning/pilot.md)")
finish()
