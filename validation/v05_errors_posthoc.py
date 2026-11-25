"""Step 5: confident errors (locked) and the post hoc, exploratory analyses."""

from collections import Counter

import numpy as np

from vcommon import MACROS, auroc, check, finish, header, load_json, scores

C = {"E+": "Ep", "E-": "Em"}
frozen = load_json("freeze/thresholds.json")["primary"]
T70 = {"s_verb": frozen["s_verb"]["0.7"]["threshold"], "s_cons": frozen["s_cons"]["0.7"]["threshold"]}
test = {c: scores("test", c) for c in C}


def family(gold: str, pred: str) -> str:
    if pred == "maybe":
        return "hedge"        # gold yes/no, answered maybe
    if gold == "maybe":
        return "overcommit"   # gold maybe, answered yes/no
    return "flip"             # yes <-> no


header("Step 5a: confident errors at the 70% threshold (accepted and wrong), by family")
for c, p in C.items():
    s = test[c]
    for name, tag in (("s_verb", "Verb"), ("s_cons", "Cons")):
        conf = getattr(s, name)
        err = (conf >= T70[name] - 1e-9) & (s.correct == 0)
        fam = Counter(family(g, a) for g, a in zip(s.gold[err], s.a0[err]))
        print(f"     {c} {name}: {int(err.sum())} errors {dict(fam)}")
        for key, word in (("hedge", "Hedge"), ("overcommit", "Overcommit"), ("flip", "Flip")):
            if f"{p}{tag}Err{word}" in MACROS:
                check(f"{p}{tag}Err{word}", fam[key])
        check(f"{p}{tag}ErrTotal", int(err.sum()))
        if name == "s_verb":
            hedge = err & (s.a0 == "maybe")
            over = err & (s.gold == "maybe") & (s.a0 != "maybe")
            check(f"Ph{p}ErrDefiniteRate", (err.sum() - hedge.sum()) / 500, "(overcommits + flips) / 500")
            check(f"Ph{p}ErrHedgeAtEightyFive", int((hedge & np.isclose(s.s_verb, 0.85)).sum()))
            if p == "Ep":
                check("PhEpErrOverHigh", int((over & (s.s_verb >= 0.95 - 1e-9)).sum()), "(overcommits at >= 0.95)")
                check("PhEpErrOverUnanimous", int((over & (s.s_cons >= 1 - 1e-9)).sum()), "(all 10 samples agree)")
                for lab in ("yes", "no", "maybe"):
                    g = s.gold == lab
                    check(f"PhEpCerGold{lab.capitalize()}", (err & g).sum() / g.sum())

header("Step 5b: committed (yes/no) versus 'maybe' answers (exploratory)")
for c, p in C.items():
    s = test[c]
    for kind, mask in (("Committed", s.a0 != "maybe"), ("Maybe", s.a0 == "maybe")):
        check(f"Ph{p}{kind}N", int(mask.sum()))
        check(f"Ph{p}{kind}Acc", s.correct[mask].mean())
        check(f"Ph{p}{kind}Conf", s.s_verb[mask].mean(), "(mean stated confidence)")

header("Step 5c: how much of the ranking is the label alone? (exploratory)")
for c, p in C.items():
    s = test[c]
    committed = s.a0 != "maybe"
    label = committed.astype(float)
    if p == "Ep":
        check("PhEpLabelAuroc", auroc(label, s.correct), "(score = 1 if yes/no, 0 if maybe)")
        check("PhEpLabelMinusVerb", auroc(label, s.correct) - auroc(s.s_verb, s.correct))
        check("PhEpWithinVerbAuroc", auroc(s.s_verb[committed], s.correct[committed]))
        check("PhEpWithinConsAuroc", auroc(s.s_cons[committed], s.correct[committed]))
        check("PhEpWithinConsMinusVerb", auroc(s.s_cons[committed], s.correct[committed])
              - auroc(s.s_verb[committed], s.correct[committed]))
    check(f"Ph{p}WithinN", int(committed.sum()))

header("Step 5d: reference AURCs and the label-entropy signal (exploratory)")
for c, p in C.items():
    s = test[c]
    n, n_c = len(s.ids), int(s.correct.sum())
    oracle = sum(max(0, k - n_c) / k for k in range(1, n + 1)) / n
    check(f"Ph{p}OracleAurc", oracle, "(perfect ranking: every error answered last)")
    check(f"Ph{p}RandomAurc", 1 - s.correct.mean(), "(random ranking = error rate)")

    def neg_entropy(answers):
        q = np.array(list(Counter(answers).values()), float) / len(answers)
        return float((q * np.log(q)).sum())

    ent = np.array([neg_entropy(s.samples[i]) for i in s.ids])
    check(f"Ph{p}EntropyAuroc", auroc(ent, s.correct), "(negative entropy of the 10 sampled labels)")
finish()
