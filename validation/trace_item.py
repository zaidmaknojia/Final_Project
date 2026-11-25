"""Follow one test item from the data file to its row in the results, by hand.

Usage:  .venv/bin/python validation/trace_item.py [item_id] [E+|E-]
Without arguments it traces the first E+ overcommit (gold maybe, answered yes/no at high confidence).
"""

import hashlib
import json
import sys

from vcommon import ROOT, apply_iso, items, latest, load_json, raw_rows, run_dirs, scores

it = items()
frozen = load_json("freeze/thresholds.json")["primary"]
cond = sys.argv[2] if len(sys.argv) > 2 else "E+"
s = scores("test", cond)
if len(sys.argv) > 1:
    iid = sys.argv[1]
else:
    t = frozen["s_verb"]["0.7"]["threshold"]
    iid = next(i for i, g, a, v in zip(s.ids, s.gold, s.a0, s.s_verb) if g == "maybe" and a != "maybe" and v >= t)
k = s.ids.index(iid)
item = it[iid]
template = (ROOT / "configs/prompts/v1.txt").read_text()
evidence = "Abstract excerpts:\n" + "\n".join(item["contexts"]) + "\n" if cond == "E+" else ""
prompt = template.format(question=item["question"], evidence_block=evidence)
rows = latest(raw_rows("test", cond))

print(f"item {iid}  condition {cond}  run {run_dirs()[('test', cond)].name}")
print(f"question: {item['question']}")
print(f"gold (final_decision): {item['gold']}")
print(f"prompt: {len(prompt)} characters, sha256 {hashlib.sha256(prompt.encode()).hexdigest()[:16]}; "
      f"row says {rows[f'{iid}|0']['prompt_sha256'][:16]}")
print("\ncalls (call_idx, temperature, seed, answer, confidence):")
for c in range(11):
    r = rows[f"{iid}|{c}"]
    print(f"  {c:2d}  T={r['temperature']:<3}  seed={r['seed']:<10}  {r['answer']:<5}  {r['confidence']}")
verb, cons = s.s_verb[k], s.s_cons[k]
print(f"\nA0 = {s.a0[k]}  correct = {bool(s.correct[k])}")
print(f"s_verb = {verb:.2f}   s_cons = {cons:.1f} ({int(cons * 10)} of 10 samples say {s.a0[k]})   "
      f"s_iso = {apply_iso(verb):.4f}")
tok = {r["item_id"]: r for r in map(json.loads, (ROOT / f"runs/tokenprob/test_{cond}.jsonl").read_text().splitlines())}
print(f"s_tok = {tok[iid]['probs'][s.a0[k]]:.4f}  (label log-probs {tok[iid]['label_logprobs']})")
print("\npolicy (answer if score >= threshold):")
for name, val in (("s_verb", verb), ("s_cons", cons)):
    for target in ("0.5", "0.7", "0.9"):
        t = frozen[name][target]["threshold"]
        verdict = "answer" if val >= t - 1e-9 else "abstain"
        error = "  <- confident error" if verdict == "answer" and not s.correct[k] else ""
        print(f"  {name} @ {target}: threshold {t:.2f} -> {verdict}{error}")
if not s.correct[k]:
    fam = "hedge" if s.a0[k] == "maybe" else "overcommit" if s.gold[k] == "maybe" else "flip"
    print(f"\nerror family: {fam}  (gold {s.gold[k]} -> said {s.a0[k]})")
