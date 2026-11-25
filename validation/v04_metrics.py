"""Step 4: accuracy, the dev fits (thresholds, isotonic map) and the locked test metrics."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from vcommon import (MACROS, apply_iso, auroc, aurc, brier, check, ece, fact, finish, fit_threshold, header,
                     isotonic_map, load_json, macro_f1, policy, scores)

C = {"E+": "Ep", "E-": "Em"}
T = {"0.5": "Fifty", "0.7": "Seventy", "0.9": "Ninety"}
S = {"s_verb": "Verb", "s_cons": "Cons", "s_verb_iso": "Iso"}
dev = {c: scores("dev", c) for c in C}
test = {c: scores("test", c) for c in C}
frozen = load_json("freeze/thresholds.json")["primary"]


def conf_of(s, name):
    return apply_iso(s.s_verb) if name == "s_verb_iso" else getattr(s, name)


header("Step 4a: accuracy and answer mix")
for c, p in C.items():
    s = test[c]
    check(f"{p}N", len(s.ids))
    check(f"{p}AccAll", s.correct.mean())
    check(f"{p}MacroF", macro_f1(s.a0, s.gold))
    check(f"{p}PredMaybe", int((s.a0 == "maybe").sum()))
    check(f"{p}DevAcc", dev[c].correct.mean()) if p == "Ep" else None
    committed = s.a0 != "maybe"
    check(f"{p}MaybeAbstainCov", committed.mean(), "(read 'maybe' as declining)")
    check(f"{p}MaybeAbstainAcc", s.correct[committed].mean())
    print(f"     {c} predictions: yes {int((s.a0 == 'yes').sum())}, no {int((s.a0 == 'no').sum())}, "
          f"maybe {int((s.a0 == 'maybe').sum())}")

header("Step 4b: thresholds refitted on E+ dev (must equal freeze/thresholds.json)")
d = dev["E+"]
for name in ("s_verb", "s_cons", "s_verb_iso"):
    for t in T:
        thr, cov = fit_threshold(conf_of(d, name), 500, float(t))
        want = frozen[name][t]
        fact(f"{name} @ {t}: t={thr:.4f}, dev coverage {cov:.3f}",
             abs(thr - want["threshold"]) < 1e-12 and abs(cov - want["dev_coverage"]) < 1e-12)
check("EpVerbThrFifty", frozen["s_verb"]["0.5"]["threshold"])
check("EpVerbThrSeventy", frozen["s_verb"]["0.7"]["threshold"])
check("EmVerbThrSeventy", frozen["s_verb"]["0.7"]["threshold"], "(the E+ threshold, applied to E-)")
check("EpConsThrFifty", frozen["s_cons"]["0.5"]["threshold"])

header("Step 4c: isotonic map refitted on E+ dev (must equal freeze/isotonic.json)")
print("dev E+ stated confidence -> n, accuracy (isotonic pools adjacent values that break monotonicity)")
for v in np.unique(d.s_verb):
    m = d.s_verb == v
    print(f"  {v:.2f}: n={m.sum():3d}  accuracy={d.correct[m].mean():.4f}")
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(d.s_verb, d.correct)
x, y = isotonic_map()
fact("breakpoints x match", np.allclose(iso.X_thresholds_, x), str(np.round(x, 3).tolist()))
fact("fitted values y match", np.allclose(iso.y_thresholds_, y), str(np.round(y, 4).tolist()))

header("Step 4d: test ranking and calibration")
for c, p in C.items():
    s = test[c]
    for name, tag in S.items():
        conf = conf_of(s, name)
        values = {"Auroc": auroc(conf, s.correct), "Ece": ece(conf, s.correct),
                  "Brier": brier(conf, s.correct), "Aurc": aurc(conf, s.correct)}
        for k, v in values.items():
            macro = f"{p}{tag}{k}"
            if macro in MACROS:
                check(macro, v)
        print(f"     {c} {name:<10} AUROC {values['Auroc']:.3f}  ECE {values['Ece']:.3f}  "
              f"Brier {values['Brier']:.3f}  AURC {values['Aurc']:.3f}")

header("Step 4e: abstention policy on test (E+ dev thresholds applied to both conditions)")
shift = 0.0
for c, p in C.items():
    s = test[c]
    for name, tag in S.items():
        conf = conf_of(s, name)
        for t, word in T.items():
            r = policy(conf, s.correct, frozen[name][t]["threshold"])
            if c == "E+":
                shift = max(shift, abs(r["coverage"] - frozen[name][t]["dev_coverage"])) if name != "s_verb_iso" else shift
            for k, key in (("Cov", "coverage"), ("Acc", "accuracy"), ("Cer", "cer")):
                macro = f"{p}{tag}{k}{word}"
                if macro in MACROS:
                    check(macro, r[key])
            acc = "--" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
            print(f"     {c} {name:<10} @{t}: t={frozen[name][t]['threshold']:.3f} coverage {r['coverage']:.3f} "
                  f"accuracy {acc} confident-error rate {r['cer']:.3f}")
check("EpThrShiftMaxPP", 100 * shift, "(largest E+ test-vs-dev coverage gap, s_verb and s_cons)")

header("Step 4f: E+ stated confidence at the 50% threshold, by gold label")
s = test["E+"]
acc = s.s_verb >= frozen["s_verb"]["0.5"]["threshold"] - 1e-9
for lab in ("yes", "no", "maybe"):
    g = s.gold == lab
    check(f"EpVerbFiftyGold{lab.capitalize()}Share", (acc & g).sum() / g.sum(), "(share of that label answered)")
    check(f"EpVerbFiftyGold{lab.capitalize()}Acc", s.correct[acc & g].mean())
finish()
