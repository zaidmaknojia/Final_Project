"""Step 6: the post hoc token-probability baseline (2,000 extra greedy calls with log-probabilities)."""

import json

import numpy as np

from vcommon import (MACROS, ROOT, auroc, aurc, check, ece, fact, finish, fit_threshold, header, policy,
                     scores)

C = {"E+": "Ep", "E-": "Em"}
LABELS = ("yes", "no", "maybe")


def rows(split: str, cond: str) -> dict[str, dict]:
    path = ROOT / f"runs/tokenprob/{split}_{cond}.jsonl"
    return {r["item_id"]: r for r in map(json.loads, path.read_text().splitlines())}


header("Step 6a: the extra calls line up with the locked greedy answers")
tok, total, matches = {}, 0, 0
for split in ("dev", "test"):
    for c in C:
        s, r = scores(split, c), rows(split, c)
        ok = all(r[i]["status"] == "ok" for i in s.ids)
        same = sum(r[i]["answer"] == a for i, a in zip(s.ids, s.a0))
        total += len(s.ids)
        matches += same
        # Renormalise the three label-token probabilities ourselves, from the raw log-probabilities.
        renorm = []
        for i in s.ids:
            lp = r[i]["label_logprobs"]
            p = {k: np.exp(lp[k]) for k in LABELS if k in lp}
            renorm.append(p[r[i]["answer"]] / sum(p.values()))
        stored = np.array([r[i]["probs"][a] for i, a in zip(s.ids, s.a0)])
        fact(f"{split} {c}: {len(s.ids)} calls, all ok, {same} answers equal the locked A0", ok and same == 500)
        fact(f"{split} {c}: stored probabilities equal our renormalisation", np.allclose(renorm, stored))
        tok[(split, c)] = (s, stored)
check("TokCalls", total)
check("TokMatches", matches)

header("Step 6b: threshold fitted on E+ dev at 70% coverage")
dev_s, dev_t = tok[("dev", "E+")]
thr, cov = fit_threshold(dev_t, 500, 0.7)
print(f"     threshold {thr:.6f}, dev coverage {cov:.3f}")
check("TokThreshold", thr)
if "TokDevCov" in MACROS:
    check("TokDevCov", cov)

header("Step 6c: s_tok on test")
for c, p in C.items():
    s, t = tok[("test", c)]
    committed = s.a0 != "maybe"
    r = policy(t, s.correct, thr)
    values = {"Auroc": auroc(t, s.correct), "Ece": ece(t, s.correct), "Aurc": aurc(t, s.correct),
              "WithinAuroc": auroc(t[committed], s.correct[committed]),
              "Cov": r["coverage"], "Acc": r["accuracy"], "Cer": r["cer"],
              "CommittedMean": t[committed].mean(), "MaybeMean": t[~committed].mean(),
              "MinusVerb": auroc(t, s.correct) - auroc(s.s_verb, s.correct),
              "WithinMinusVerb": auroc(t[committed], s.correct[committed])
              - auroc(s.s_verb[committed], s.correct[committed]),
              "AurcMinusVerb": aurc(t, s.correct) - aurc(s.s_verb, s.correct)}
    shown = [k for k in values if f"Tok{p}{k}" in MACROS]
    for k in shown:
        check(f"Tok{p}{k}", values[k])
    print(f"     {c}: AUROC {values['Auroc']:.3f}, within yes/no {values['WithinAuroc']:.3f}, "
          f"coverage {r['coverage']:.3f}, accuracy {r['accuracy']:.3f}")
finish()
