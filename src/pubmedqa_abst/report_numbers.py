"""LaTeX macros and tables for the reports, generated only from results/ (spec §13).

Every number in a report comes from here, so no value is ever typed by hand.
Run: python -m pubmedqa_abst.report_numbers [results_dir] [out_dir]
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("E+", "E-")
SCORES = ("s_verb", "s_cons", "s_verb_iso")
TARGETS = ("0.5", "0.7", "0.9")
TOKENS = {
    "E+": "Ep", "E-": "Em", "s_verb": "Verb", "s_cons": "Cons", "s_verb_iso": "Iso",
    "0.5": "Fifty", "0.7": "Seventy", "0.9": "Ninety", "accept_all": "AcceptAll",
    "accuracy": "Acc", "coverage": "Cov", "cer": "Cer", "auroc": "Auroc", "aurc": "Aurc",
    "brier": "Brier", "ece": "Ece", "accuracy_full": "AccFull",
}
GROUP_PREFIX = {
    "absolute": "", "cons_vs_verb": "ConsVsVerb", "threshold_vs_accept_all": "ThrVsAll",
    "eplus_vs_eminus": "EvidDelta", "iso_vs_raw_verb": "IsoVsRaw",
}
SCORE_LABEL = {"s_verb": "Stated", "s_cons": "Consistency", "s_verb_iso": "Stated (iso.)"}
COND_LABEL = {"E+": "E+", "E-": "E$-$"}


def fmt(x: float | None, digits: int = 3) -> str:
    """Fixed decimals; negatives get a text-mode minus; values that round to zero lose the sign."""
    if x is None:
        return "--"
    text = f"{abs(x):.{digits}f}"
    negative = x < 0 and float(text) != 0
    return ("\\textminus{}" if negative else "") + text


def ci(entry: dict) -> str:
    if entry["point"] is None:
        return "--"
    return f"{fmt(entry['point'])} [{fmt(entry['lo'])}, {fmt(entry['hi'])}]"


def _key_tokens(key: str) -> list[str]:
    tokens = []
    for part in key.split("|"):
        head, _, tail = part.partition("@")
        tokens.append(head)
        if tail:
            target, _, rest = tail.partition("-")
            tokens += [target] + ([rest] if rest else [])
    return tokens


def macro_name(group: str, key: str) -> str:
    return GROUP_PREFIX[group] + "".join(TOKENS[t] for t in _key_tokens(key))


def _family(confusion: str) -> str:
    if confusion.endswith("->maybe"):
        return "hedge"
    return "overcommit" if confusion.startswith("maybe->") else "flip"


def error_families(rows: list[dict]) -> dict[tuple[str, str], dict[str, int]]:
    """Confident errors per (condition, score), grouped into hedge / overcommit / flip (spec §14)."""
    out: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"hedge": 0, "overcommit": 0, "flip": 0, "total": 0})
    for r in rows:
        fam = out[(r["condition"], r["score"])]
        fam[_family(r["confusion"])] += 1
        fam["total"] += 1
    return dict(out)


def _maybe_as_abstention(confusion: dict) -> tuple[float, float | None]:
    """Coverage and accuracy when a "maybe" answer is read as declining (spec §14, exploratory)."""
    n = sum(v for row in confusion.values() for v in row.values())
    answered = sum(confusion[p][g] for p in ("yes", "no") for g in confusion[p])
    correct = confusion["yes"]["yes"] + confusion["no"]["no"]
    return answered / n, (correct / answered if answered else None)


def _load(results_dir: Path) -> tuple[dict, dict, list[dict]]:
    metrics = json.loads((results_dir / "metrics.json").read_text())
    boot = json.loads((results_dir / "bootstrap.json").read_text())
    with open(results_dir / "confident_errors.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if "confusion" in r]
    return metrics, boot, rows


def coverage_shift_pp(metrics: dict, cond: str) -> float:
    """Largest gap between test and dev coverage over all primary thresholds, in percentage points."""
    scores = metrics["test"][cond]["scores"]
    return 100 * max(abs(entry["coverage"] - entry["dev_coverage"])
                     for score in SCORES for entry in scores[score]["primary"].values())


def _metric_macros(metrics: dict) -> dict[str, str]:
    m: dict[str, str] = {"EpThrShiftMaxPP": f"{coverage_shift_pp(metrics, 'E+'):.1f}"}
    for cond in CONDITIONS:
        c = TOKENS[cond]
        full = metrics["test"][cond]["full"]
        m[f"{c}AccAll"] = fmt(full["accuracy"])
        m[f"{c}MacroF"] = fmt(full["macro_f1"])
        m[f"{c}PredMaybe"] = str(full["pred_dist"].get("maybe", 0))
        m[f"{c}N"] = str(full["n"])
        m[f"{c}DevAcc"] = fmt(metrics["dev"][cond]["full"]["accuracy"])
        cov, acc = _maybe_as_abstention(full["confusion"])
        m[f"{c}MaybeAbstainCov"], m[f"{c}MaybeAbstainAcc"] = fmt(cov), fmt(acc)
        m[f"{c}GoldMaybeRecall"] = str(full["confusion"]["maybe"]["maybe"])
        for score in SCORES:
            s = TOKENS[score]
            for t in TARGETS:
                entry = metrics["test"][cond]["scores"][score]["primary"][t]
                m[f"{c}{s}Thr{TOKENS[t]}"] = fmt(entry["threshold"], 2)
                for label, pc in entry["per_class_gold"].items():
                    name = f"{c}{s}{TOKENS[t]}Gold{label.capitalize()}"
                    m[f"{name}Share"] = fmt(pc["accepted_share"])
                    m[f"{name}Acc"] = fmt(pc["acc_accepted"])
    gold = metrics["test"]["E+"]["full"]["gold_dist"]
    for label, count in gold.items():
        m[f"Gold{label.capitalize()}"] = str(count)
    return m


def build_macros(results_dir: Path) -> dict[str, str]:
    metrics, boot, rows = _load(Path(results_dir))
    macros = _metric_macros(metrics)
    for group, entries in boot.items():
        for key, entry in entries.items():
            name = macro_name(group, key)
            macros[name] = fmt(entry["point"])
            macros[f"{name}CI"] = f"[{fmt(entry['lo'])}, {fmt(entry['hi'])}]"
    for (cond, score), fam in error_families(rows).items():
        for kind, count in fam.items():
            macros[f"{TOKENS[cond]}{TOKENS[score]}Err{kind.capitalize()}"] = str(count)
    return macros


def _calibration_table(boot: dict) -> str:
    lines = []
    for cond in CONDITIONS:
        for score in SCORES:
            a = boot["absolute"]
            cells = [ci(a[f"{cond}|{score}|{k}"]) for k in ("auroc", "aurc", "ece", "brier")]
            lines.append(" & ".join([COND_LABEL[cond], SCORE_LABEL[score], *cells]) + r" \\")
    return "\n".join(lines) + "\n"


def _selective_table(metrics: dict) -> str:
    lines = []
    for cond in CONDITIONS:
        sb = metrics["test"][cond]["scores"]
        acc_all = sb["s_verb"]["accept_all"]
        lines.append(f"{COND_LABEL[cond]} & Accept all & -- & -- & 1.000 & {fmt(acc_all['accuracy'])} "
                     f"& {fmt(acc_all['confident_error_rate'])} \\\\")
        for score in ("s_verb", "s_cons"):
            for t in TARGETS:
                e = sb[score]["primary"][t]
                lines.append(" & ".join([COND_LABEL[cond], SCORE_LABEL[score], f"{int(float(t) * 100)}\\%",
                                         fmt(e["threshold"], 2), fmt(e["coverage"]), fmt(e["accuracy"]),
                                         fmt(e["confident_error_rate"])]) + r" \\")
        lines.append(r"\midrule" if cond == "E+" else "")
    return "\n".join(line for line in lines if line) + "\n"


DELTA_ROWS = (
    ("cons_vs_verb", "E+|auroc", "Consistency $-$ stated, AUROC (E+)"),
    ("cons_vs_verb", "E-|auroc", "Consistency $-$ stated, AUROC (E$-$)"),
    ("cons_vs_verb", "E+|aurc", "Consistency $-$ stated, AURC (E+)"),
    ("threshold_vs_accept_all", "E+|s_verb|accuracy@0.7-accept_all", "Stated@70\\% $-$ accept all, accuracy (E+)"),
    ("threshold_vs_accept_all", "E+|s_cons|accuracy@0.7-accept_all", "Consistency@70\\% $-$ accept all, accuracy (E+)"),
    ("eplus_vs_eminus", "accuracy_full", "E+ $-$ E$-$, accuracy (all answered)"),
    ("eplus_vs_eminus", "s_verb|coverage@0.7", "E+ $-$ E$-$, coverage at stated@70\\%"),
    ("eplus_vs_eminus", "s_verb|auroc", "E+ $-$ E$-$, stated AUROC"),
    ("iso_vs_raw_verb", "E+|ece", "Isotonic $-$ raw stated, ECE (E+)"),
    ("iso_vs_raw_verb", "E-|ece", "Isotonic $-$ raw stated, ECE (E$-$)"),
)


def _deltas_table(boot: dict) -> str:
    return "\n".join(f"{label} & {ci(boot[g][k])} \\\\" for g, k, label in DELTA_ROWS) + "\n"


def _perclass_table(metrics: dict) -> str:
    lines = []
    for cond in CONDITIONS:
        for t in ("0.5", "0.7"):
            pc = metrics["test"][cond]["scores"]["s_verb"]["primary"][t]["per_class_gold"]
            cells = [f"{fmt(pc[g]['accepted_share'])} / {fmt(pc[g]['acc_accepted'])}" for g in ("yes", "no", "maybe")]
            lines.append(" & ".join([COND_LABEL[cond], f"{int(float(t) * 100)}\\%", *cells]) + r" \\")
    return "\n".join(lines) + "\n"


LATEX_SPECIALS = {"\\": r"\textbackslash{}", "%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#",
                  "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
FAMILY_ORDER = ("hedge", "overcommit", "flip")


def latex_escape(text: str) -> str:
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in text)


EXAMPLE_SEED = 20261124  # the protocol's example_seed (configs/run.yaml)
EXAMPLE_QUOTA = {("E+", "hedge"): 2, ("E+", "overcommit"): 2, ("E+", "flip"): 2,
                 ("E-", "hedge"): 1, ("E-", "overcommit"): 1, ("E-", "flip"): 1}


def sample_stated_examples(rows: list[dict], quota: dict[tuple[str, str], int], seed: int) -> list[dict]:
    """A fixed-seed draw of confident errors under stated confidence, `quota`
    rows per (condition, error family), so every family is represented."""
    rng = random.Random(seed)
    out = []
    for (cond, family), k in sorted(quota.items()):
        group = sorted((r for r in rows if r["score"] == "s_verb" and r["condition"] == cond
                        and _family(r["confusion"]) == family), key=lambda r: r["item_id"])
        out.extend(rng.sample(group, min(k, len(group))))
    return out


def examples_table(rows: list[dict]) -> str:
    """Rows of sampled confident errors under stated confidence, by condition then error family."""
    chosen = sorted((r for r in rows if r["score"] == "s_verb"),
                    key=lambda r: (CONDITIONS.index(r["condition"]), FAMILY_ORDER.index(_family(r["confusion"]))))
    lines = []
    for r in chosen:
        gold, pred = r["confusion"].split("->")
        cells = [COND_LABEL[r["condition"]], _family(r["confusion"]), f"{gold}$\\to${pred}",
                 fmt(float(r["s_verb"]), 2), fmt(float(r["s_cons"]), 2), latex_escape(r["question"])]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def _ci_text(entry: dict) -> str:
    return f"[{fmt(entry['lo'])}, {fmt(entry['hi'])}]"


def posthoc_macros(payload: dict) -> dict[str, str]:
    """Macros for the exploratory analyses in results/posthoc.json (prefix Ph)."""
    m: dict[str, str] = {}
    for cond in CONDITIONS:
        c, x = "Ph" + TOKENS[cond], payload[cond]
        m[f"{c}VerbAuroc"], m[f"{c}ConsAuroc"] = fmt(x["auroc"]["s_verb"]), fmt(x["auroc"]["s_cons"])
        m[f"{c}LabelAuroc"] = fmt(x["auroc"]["label"])
        if "label_entropy" in x["auroc"]:
            m[f"{c}EntropyAuroc"] = fmt(x["auroc"]["label_entropy"])
        w = x["within_committed"]
        m[f"{c}WithinN"] = str(w["n"])
        m[f"{c}WithinVerbAuroc"], m[f"{c}WithinConsAuroc"] = fmt(w["auroc_s_verb"]), fmt(w["auroc_s_cons"])
        for kind, label in (("committed", "Committed"), ("maybe", "Maybe")):
            g = x["answer_mix"][kind]
            m[f"{c}{label}N"], m[f"{c}{label}Share"] = str(g["n"]), fmt(g["share"])
            m[f"{c}{label}Acc"], m[f"{c}{label}Conf"] = fmt(g["accuracy"]), fmt(g["mean_conf"])
        gm = x["gold_maybe"]
        m[f"{c}GoldMaybeSaidMaybe"], m[f"{c}GoldMaybeSaidYesNo"] = str(gm["said_maybe"]), str(gm["said_yes_no"])
        m[f"{c}GoldMaybeCorrectYesNo"] = str(gm["correct_yes_no"])
        for label, rate in x["rate_by_gold_s_verb_0.7"].items():
            m[f"{c}CerGold{label.capitalize()}"] = fmt(rate)
        m[f"{c}RandomAurc"], m[f"{c}OracleAurc"] = fmt(x["reference_aurc"]["random"]), fmt(x["reference_aurc"]["oracle"])
        e = x["errors"]
        for key, name in (("total", "ErrTotal"), ("on_gold_maybe", "ErrOnGoldMaybe"), ("overcommit", "ErrOver"),
                          ("overcommit_at_095", "ErrOverHigh"), ("overcommit_unanimous", "ErrOverUnanimous"),
                          ("hedge", "ErrHedge"), ("hedge_at_085", "ErrHedgeAtEightyFive")):
            m[f"{c}{name}"] = str(e[key])
        # Accepted wrong answers that are definite (flips and overcommits), as a share of all questions.
        m[f"{c}ErrDefiniteRate"] = fmt((e["total"] - e["hedge"]) / x["n"])
        for key, name in (("label_minus_verb_auroc", "LabelMinusVerb"), ("within_verb_auroc", "WithinVerbAuroc"),
                          ("within_cons_auroc", "WithinConsAuroc"), ("within_cons_minus_verb_auroc", "WithinConsMinusVerb")):
            entry = payload["bootstrap"][cond][key]
            if name.startswith("Within") and name != "WithinConsMinusVerb":
                m[f"{c}{name}CI"] = _ci_text(entry)
            else:
                m[f"{c}{name}"], m[f"{c}{name}CI"] = fmt(entry["point"]), _ci_text(entry)
    m["PhMajorityLabel"] = payload["E+"]["majority"]["label"]
    m["PhMajorityAcc"] = fmt(payload["E+"]["majority"]["accuracy"])
    if "token" in payload:
        m.update(token_macros(payload["token"]))
    return m


TOKEN_SCORES = {"s_tok": "", "s_verb": "Verb", "s_cons": "Cons"}
TOKEN_FIELDS = {"auroc": "Auroc", "within_committed_auroc": "WithinAuroc", "ece": "Ece", "brier": "Brier",
                "aurc": "Aurc", "mean": "Mean"}
TOKEN_POLICY = {"coverage": "Cov", "accuracy": "Acc", "confident_error_rate": "Cer"}
TOKEN_BOOT = {"tok_minus_verb_auroc": "MinusVerb", "within_tok_minus_verb_auroc": "WithinMinusVerb",
              "tok_minus_verb_aurc": "AurcMinusVerb"}


def _count(n: int) -> str:
    """Integer with a LaTeX thousands separator: 2000 -> 2{,}000."""
    return f"{n:,}".replace(",", "{,}")


def token_macros(token: dict) -> dict[str, str]:
    """Macros for the post-hoc token-probability baseline (prefix Tok)."""
    m: dict[str, str] = {}
    for cond in CONDITIONS:
        c = "Tok" + TOKENS[cond]
        for score, label in TOKEN_SCORES.items():
            block = token[cond][score]
            m.update({f"{c}{label}{name}": fmt(block[key]) for key, name in TOKEN_FIELDS.items()})
            m.update({f"{c}{label}{name}": fmt(block["policy"][key]) for key, name in TOKEN_POLICY.items()})
        boot = token["bootstrap"][cond]
        for key, name in TOKEN_BOOT.items():
            m[f"{c}{name}"], m[f"{c}{name}CI"] = fmt(boot[key]["point"]), _ci_text(boot[key])
        m[f"{c}WithinCI"] = _ci_text(boot["within_tok_auroc"])
        for kind, label in (("committed", "Committed"), ("maybe", "Maybe")):
            g = token[cond]["s_tok_by_label"][kind]
            m[f"{c}{label}Mean"], m[f"{c}{label}Accepted"] = fmt(g["mean"]), fmt(g["accepted_share"])
    m["TokThreshold"] = fmt(token["threshold"]["threshold"])
    m["TokDevCov"] = fmt(token["threshold"]["dev_coverage"])
    rows = token["agreement"].values()
    m["TokCalls"] = _count(sum(r["n"] for r in rows))
    m["TokMatches"] = _count(sum(r["answer_matches_a0"] for r in rows))
    m["TokUnscored"] = str(sum(r["n"] - r["ok"] for r in rows))
    return m


def posthoc_table(payload: dict) -> str:
    """AUROC over all answers and within committed (yes/no) answers, per signal."""
    lines = []
    for cond in CONDITIONS:
        x, b = payload[cond], payload["bootstrap"][cond]
        n = x["within_committed"]["n"]
        rows = [("Stated", x["auroc"]["s_verb"], f"{fmt(x['within_committed']['auroc_s_verb'])} {_ci_text(b['within_verb_auroc'])}"),
                ("Consistency", x["auroc"]["s_cons"], f"{fmt(x['within_committed']['auroc_s_cons'])} {_ci_text(b['within_cons_auroc'])}"),
                ("Label only", x["auroc"]["label"], "--")]
        if "token" in payload:
            t, tb = payload["token"][cond]["s_tok"], payload["token"]["bootstrap"][cond]
            rows.append(("Token prob.", t["auroc"], f"{fmt(t['within_committed_auroc'])} {_ci_text(tb['within_tok_auroc'])}"))
        for name, whole, within in rows:
            lines.append(f"{COND_LABEL[cond]} & {name} & {fmt(whole)} & {within} \\\\")
        lines.append(f"% {cond}: {n} committed answers")
        lines.append(r"\midrule" if cond == "E+" else "")
    return "\n".join(line for line in lines if line) + "\n"


def write_report_inputs(results_dir: Path, out_dir: Path) -> list[Path]:
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, boot, rows = _load(results_dir)
    examples = sample_stated_examples(rows, EXAMPLE_QUOTA, EXAMPLE_SEED)
    macros = build_macros(results_dir)
    posthoc_path = results_dir / "posthoc.json"
    payload = json.loads(posthoc_path.read_text()) if posthoc_path.exists() else None
    if payload is not None:
        macros.update(posthoc_macros(payload))
    header = "% Generated by pubmedqa_abst.report_numbers from results/. Do not edit.\n"
    files = {
        "numbers.tex": "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())),
        "table_calibration.tex": _calibration_table(boot),
        "table_selective.tex": _selective_table(metrics),
        "table_deltas.tex": _deltas_table(boot),
        "table_perclass.tex": _perclass_table(metrics),
        "table_examples.tex": examples_table(examples),
    }
    if payload is not None:
        files["table_posthoc.tex"] = posthoc_table(payload)
    paths = []
    for name, body in files.items():
        path = out_dir / name
        path.write_text(header + body)
        paths.append(path)
    return paths


if __name__ == "__main__":  # pragma: no cover
    root = Path(__file__).resolve().parents[2]
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "reports" / "shared"
    for p in write_report_inputs(res, out):
        print(p)
