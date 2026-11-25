"""Step 7: paired bootstrap intervals for every comparison the reports quote.

Draws mirror the locked protocol: numpy default_rng(20261130), B = 10,000 resamples of
the 500 test items in sorted-id order, the same index rows for every statistic (that is
what makes the comparisons paired), 2.5th/97.5th percentiles, undefined replicates
dropped. Each statistic is recomputed here with our own vectorised code.
"""

import json

import numpy as np

from vcommon import (MACROS, ROOT, apply_iso, auroc, aurc, check, check_ci, ece, fact, finish, header,
                     load_json, scores)

B, SEED, N = 10_000, 20261130, 500
frozen = load_json("freeze/thresholds.json")["primary"]
ep, em = scores("test", "E+"), scores("test", "E-")
fact("both conditions cover the same 500 sorted test ids", ep.ids == em.ids and len(ep.ids) == N)
tok_rows = {r["item_id"]: r for r in map(json.loads, (ROOT / "runs/tokenprob/test_E+.jsonl").read_text().splitlines())}
tok = np.array([tok_rows[i]["probs"][a] for i, a in zip(ep.ids, ep.a0)])


def fast_aurc(conf, correct):
    order = np.argsort(-conf, kind="stable")
    c, wrong = conf[order], np.cumsum(1 - correct[order])
    ends = np.r_[np.nonzero(np.diff(c))[0], c.size - 1]
    cov = (ends + 1) / c.size
    return float(np.sum(np.diff(np.r_[0.0, cov]) * wrong[ends] / (ends + 1)))


def fast_ece(conf, correct, bins=10):
    idx = np.minimum(np.floor(conf * bins + 1e-9).astype(int), bins - 1)
    n = np.bincount(idx, minlength=bins)
    sc, sy = np.bincount(idx, conf, bins), np.bincount(idx, correct, bins)
    m = n > 0
    return float(np.sum(np.abs(sc[m] - sy[m])) / conf.size)


def acc_at(conf, correct, t):
    a = conf >= t - 1e-9
    return correct[a].mean() if a.any() else None


def cov_at(conf, t):
    return (conf >= t - 1e-9).mean()


def cer_at(conf, correct, t):
    return ((conf >= t - 1e-9) & (correct == 0)).mean()


def sub(a, b):
    return None if a is None or b is None else a - b


def within(conf, correct, committed):
    return auroc(conf[committed], correct[committed])


t70 = {k: frozen[k]["0.7"]["threshold"] for k in ("s_verb", "s_cons")}
ep_iso, em_iso = apply_iso(ep.s_verb), apply_iso(em.s_verb)
ep_label = (ep.a0 != "maybe").astype(float)

fact("vectorised AURC equals the step-by-step AURC", np.isclose(fast_aurc(tok, ep.correct), aurc(tok, ep.correct)))
fact("vectorised ECE equals the bin-by-bin ECE", np.isclose(fast_ece(ep.s_verb, ep.correct), ece(ep.s_verb, ep.correct)))


def stats(i):
    """Every quoted comparison on one resample (i = index array into the sorted test ids)."""
    y, z = ep.correct[i], em.correct[i]
    cm = ep.a0[i] != "maybe"
    return {
        "ConsVsVerbEpAuroc": sub(auroc(ep.s_cons[i], y), auroc(ep.s_verb[i], y)),
        "ConsVsVerbEmAuroc": sub(auroc(em.s_cons[i], z), auroc(em.s_verb[i], z)),
        "ConsVsVerbEpAurc": fast_aurc(ep.s_cons[i], y) - fast_aurc(ep.s_verb[i], y),
        "EvidDeltaAccFull": y.mean() - z.mean(),
        "EvidDeltaVerbAuroc": sub(auroc(ep.s_verb[i], y), auroc(em.s_verb[i], z)),
        "EvidDeltaVerbCovSeventy": cov_at(ep.s_verb[i], t70["s_verb"]) - cov_at(em.s_verb[i], t70["s_verb"]),
        "EvidDeltaVerbCerSeventy": cer_at(ep.s_verb[i], y, t70["s_verb"]) - cer_at(em.s_verb[i], z, t70["s_verb"]),
        "EvidDeltaConsCovSeventy": cov_at(ep.s_cons[i], t70["s_cons"]) - cov_at(em.s_cons[i], t70["s_cons"]),
        "IsoVsRawEpEce": fast_ece(ep_iso[i], y) - fast_ece(ep.s_verb[i], y),
        "IsoVsRawEmEce": fast_ece(em_iso[i], z) - fast_ece(em.s_verb[i], z),
        "IsoVsRawEpAurc": fast_aurc(ep_iso[i], y) - fast_aurc(ep.s_verb[i], y),
        "ThrVsAllEpVerbAccSeventyAcceptAll": sub(acc_at(ep.s_verb[i], y, t70["s_verb"]), y.mean()),
        "ThrVsAllEpConsAccSeventyAcceptAll": sub(acc_at(ep.s_cons[i], y, t70["s_cons"]), y.mean()),
        "PhEpLabelMinusVerb": sub(auroc(ep_label[i], y), auroc(ep.s_verb[i], y)),
        "PhEpWithinVerbAuroc": within(ep.s_verb[i], y, cm),
        "PhEpWithinConsAuroc": within(ep.s_cons[i], y, cm),
        "PhEpWithinConsMinusVerb": sub(within(ep.s_cons[i], y, cm), within(ep.s_verb[i], y, cm)),
        "TokEpMinusVerb": sub(auroc(tok[i], y), auroc(ep.s_verb[i], y)),
        "TokEpWithin": within(tok[i], y, cm),
        "TokEpWithinMinusVerb": sub(within(tok[i], y, cm), within(ep.s_verb[i], y, cm)),
        "TokEpAurcMinusVerb": fast_aurc(tok[i], y) - fast_aurc(ep.s_verb[i], y),
    }


header("Step 7: point estimates (full test set)")
point = stats(np.arange(N))
for name, value in point.items():
    if name in MACROS:
        check(name, value)

header(f"Step 7: 95% intervals from {B:,} paired resamples")
draws = np.random.default_rng(SEED).integers(0, N, size=(B, N))
reps = {name: [] for name in point}
for row in draws:
    for name, value in stats(row).items():
        if value is not None:
            reps[name].append(value)
for name, values in reps.items():
    lo, hi = np.percentile(values, [2.5, 97.5])
    macro = "TokEpWithinCI" if name == "TokEpWithin" else f"{name}CI"
    check_ci(macro, lo, hi)
    if len(values) < B:
        print(f"      ({B - len(values)} undefined replicates dropped)")
finish()
