"""Write validation/numbers.md: every number the two reports use, its value, what it
means, which report uses it and which validation step recomputes it.

Run:  .venv/bin/python validation/make_numbers_md.py
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEPS = ["v01_data.py", "v02_locks.py", "v03_runs.py", "v04_metrics.py", "v05_errors_posthoc.py",
         "v06_token.py", "v07_bootstrap.py"]
REPORTS = {"AIHC": ROOT / "reports/aihc/main.tex", "CS388": ROOT / "reports/nlp-neurips/main.tex"}

COND = {"Ep": "E+", "Em": "E−"}
SCORE = {"Verb": "stated confidence (s_verb)", "Cons": "consistency (s_cons)", "Iso": "recalibrated (s_iso)"}
METRIC = {"Auroc": "AUROC", "Aurc": "AURC (lower is better)", "Ece": "ECE (lower is better)",
          "Brier": "Brier score"}
TARGET = {"Fifty": "50%", "Seventy": "70%", "Ninety": "90%"}
POLICY = {"Cov": "coverage (share answered)", "Acc": "accuracy of answered items",
          "Cer": "confident-error rate (answered and wrong, over all 500)"}
FAMILY = {"Hedge": "hedges (gold yes/no, said maybe)", "Overcommit": "overcommits (gold maybe, said yes/no)",
          "Flip": "flips (yes↔no)", "Total": "all families"}

FIXED = {
    "EpDevAcc": "E+ accuracy on the dev split",
    "EpThrShiftMaxPP": "largest gap between E+ test and dev coverage over all locked thresholds, percentage points",
    "PhMajorityAcc": "accuracy of always answering the majority label (yes)",
    "PhEpLabelAuroc": "E+ AUROC of the label alone (1 if yes/no, 0 if maybe)",
    "PhEpLabelMinusVerb": "E+ label-only AUROC minus s_verb AUROC",
    "PhEpWithinN": "E+ number of yes/no answers",
    "PhEmWithinN": "E− number of yes/no answers",
    "PhEpWithinVerbAuroc": "E+ s_verb AUROC within yes/no answers only",
    "PhEpWithinConsAuroc": "E+ s_cons AUROC within yes/no answers only",
    "PhEpWithinConsMinusVerb": "E+ within-yes/no AUROC, s_cons minus s_verb",
    "PhEpErrOverHigh": "E+ overcommits (s_verb, 70% threshold) stated at confidence ≥ 0.95",
    "PhEpErrOverUnanimous": "E+ overcommits (s_verb, 70% threshold) where all 10 samples agree",
    "TokCalls": "number of extra greedy calls with log-probabilities (dev+test, E+ and E−)",
    "TokMatches": "extra calls whose answer equals the locked greedy answer",
    "TokThreshold": "s_tok threshold fitted on E+ dev for 70% coverage",
    "TokDevCov": "dev coverage of that s_tok threshold",
}


def describe(name: str) -> str:
    ci = name.endswith("CI")
    base = name[:-2] if ci else name
    suffix = " — 95% paired bootstrap interval" if ci else ""
    if base in FIXED:
        return FIXED[base] + suffix
    patterns = [
        (r"(Ep|Em)N", lambda m: f"{COND[m[1]]}: number of test items"),
        (r"(Ep|Em)AccAll", lambda m: f"{COND[m[1]]}: accuracy over all test answers"),
        (r"(Ep|Em)MacroF", lambda m: f"{COND[m[1]]}: macro-F1 over yes/no/maybe"),
        (r"(Ep|Em)PredMaybe", lambda m: f"{COND[m[1]]}: number of 'maybe' answers"),
        (r"(Ep|Em)MaybeAbstainCov", lambda m: f"{COND[m[1]]}: share of yes/no answers (reading maybe as declining)"),
        (r"(Ep|Em)MaybeAbstainAcc", lambda m: f"{COND[m[1]]}: accuracy of the yes/no answers"),
        (r"(Ep|Em)(Verb|Cons|Iso)(Auroc|Aurc|Ece|Brier)", lambda m: f"{COND[m[1]]}, {SCORE[m[2]]}: {METRIC[m[3]]}"),
        (r"(Ep|Em)(Verb|Cons|Iso)(Cov|Acc|Cer)(Fifty|Seventy|Ninety)",
         lambda m: f"{COND[m[1]]}, {SCORE[m[2]]}, {TARGET[m[4]]} threshold: {POLICY[m[3]]}"),
        (r"(Ep|Em)(Verb|Cons)Thr(Fifty|Seventy|Ninety)",
         lambda m: f"{SCORE[m[2]]} threshold for {TARGET[m[3]]} dev coverage (fitted on E+ dev, used for both)"),
        (r"(Ep|Em)(Verb|Cons)Err(Hedge|Overcommit|Flip|Total)",
         lambda m: f"{COND[m[1]]}, {SCORE[m[2]]} at 70% threshold: confident errors, {FAMILY[m[3]]}"),
        (r"EpVerbFiftyGold(Yes|No|Maybe)Share",
         lambda m: f"E+, s_verb at 50% threshold: share of gold-{m[1].lower()} items answered"),
        (r"EpVerbFiftyGold(Yes|No|Maybe)Acc",
         lambda m: f"E+, s_verb at 50% threshold: accuracy on answered gold-{m[1].lower()} items"),
        (r"Gold(Yes|No|Maybe)", lambda m: f"test items whose expert answer is {m[1].lower()}"),
        (r"ConsVsVerb(Ep|Em)(Auroc|Aurc)", lambda m: f"{COND[m[1]]}: s_cons minus s_verb, {METRIC[m[2]]}"),
        (r"EvidDeltaAccFull", lambda m: "accuracy, E+ minus E−"),
        (r"EvidDeltaVerbAuroc", lambda m: "s_verb AUROC, E+ minus E−"),
        (r"EvidDelta(Verb|Cons)(Cov|Cer)Seventy",
         lambda m: f"{SCORE[m[1]]} at 70% threshold, {POLICY[m[2]].split(' (')[0]}, E+ minus E−"),
        (r"IsoVsRaw(Ep|Em)(Ece|Aurc)", lambda m: f"{COND[m[1]]}: s_iso minus s_verb, {METRIC[m[2]]}"),
        (r"ThrVsAll(Ep|Em)(Verb|Cons)AccSeventyAcceptAll",
         lambda m: f"{COND[m[1]]}, {SCORE[m[2]]}: accuracy at 70% threshold minus accuracy answering all"),
        (r"Ph(Ep|Em)(Committed|Maybe)N", lambda m: f"{COND[m[1]]}: number of {m[2].lower()} answers"),
        (r"Ph(Ep|Em)(Committed|Maybe)Acc", lambda m: f"{COND[m[1]]}: accuracy of {m[2].lower()} answers"),
        (r"Ph(Ep|Em)(Committed|Maybe)Conf", lambda m: f"{COND[m[1]]}: mean stated confidence of {m[2].lower()} answers"),
        (r"Ph(Ep|Em)EntropyAuroc", lambda m: f"{COND[m[1]]}: AUROC of negative entropy over the 10 sampled labels"),
        (r"Ph(Ep|Em)ErrDefiniteRate", lambda m: f"{COND[m[1]]}, s_verb at 70%: (overcommits + flips) / 500"),
        (r"Ph(Ep|Em)ErrHedgeAtEightyFive", lambda m: f"{COND[m[1]]}, s_verb at 70%: hedges stated at exactly 0.85"),
        (r"Ph(Ep|Em)OracleAurc", lambda m: f"{COND[m[1]]}: AURC of a perfect ranking"),
        (r"Ph(Ep|Em)RandomAurc", lambda m: f"{COND[m[1]]}: AURC of a random ranking (the error rate)"),
        (r"PhEpCerGold(Yes|No|Maybe)",
         lambda m: f"E+, s_verb at 70%: confident errors per gold-{m[1].lower()} item"),
        (r"Tok(Ep|Em)(Auroc|Aurc|Ece)", lambda m: f"{COND[m[1]]}, token probability (s_tok): {METRIC[m[2]]}"),
        (r"Tok(Ep|Em)WithinAuroc", lambda m: f"{COND[m[1]]}, s_tok AUROC within yes/no answers"),
        (r"TokEpWithin", lambda m: "E+, s_tok AUROC within yes/no answers"),
        (r"Tok(Ep|Em)(Cov|Acc|Cer)", lambda m: f"{COND[m[1]]}, s_tok at its threshold: {POLICY[m[2]]}"),
        (r"Tok(Ep|Em)(Committed|Maybe)Mean", lambda m: f"{COND[m[1]]}: mean s_tok of {m[2].lower()} answers"),
        (r"Tok(Ep|Em)MinusVerb", lambda m: f"{COND[m[1]]}: s_tok minus s_verb, AUROC"),
        (r"Tok(Ep|Em)WithinMinusVerb", lambda m: f"{COND[m[1]]}: within-yes/no AUROC, s_tok minus s_verb"),
        (r"Tok(Ep|Em)AurcMinusVerb", lambda m: f"{COND[m[1]]}: s_tok minus s_verb, AURC"),
    ]
    for pattern, text in patterns:
        if m := re.fullmatch(pattern, base):
            return text(m) + suffix
    return "(no description)" + suffix


def group(name: str) -> str:
    if name.startswith("Tok"):
        return "6. Token-probability baseline (post hoc, exploratory)"
    if name.startswith("Ph") and "Majority" not in name:
        return "5. Post hoc analyses (exploratory)"
    if any(name.startswith(p) for p in ("ConsVsVerb", "EvidDelta", "IsoVsRaw", "ThrVsAll")):
        return "4. Planned comparisons (locked)"
    if "Err" in name:
        return "3. Confident errors (locked)"
    return "1. Data, accuracy and thresholds" if re.search(r"Gold|N$|AccAll|MacroF|PredMaybe|DevAcc|Maybe"
                                                         r"Abstain|Thr|Majority", name) else \
        "2. Ranking, calibration and abstention (locked)"


macros = {k: v.replace(r"\textminus{}", "−").replace("{,}", ",")
          for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (ROOT / "reports/shared/numbers.tex").read_text())}
used: dict[str, list[str]] = {}
for label, path in REPORTS.items():
    for name in re.findall(r"\\([A-Za-z]+)", path.read_text()):
        if name in macros and label not in used.setdefault(name, []):
            used[name].append(label)

step_of: dict[str, str] = {}
for step in STEPS:
    out = subprocess.run([sys.executable, str(HERE / step)], cwd=HERE, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if m := re.match(r"PASS\s+(\w+)\s+report=", line):
            step_of.setdefault(m[1], step.split("_")[0])

rows: dict[str, list[str]] = {}
for name in sorted(used):
    rows.setdefault(group(name), []).append(
        f"| `\\{name}` | {macros[name]} | {describe(name)} | {' + '.join(used[name])} | "
        f"{step_of.get(name, '**not checked**')} |")

lines = ["# Every number in the two reports",
         "",
         "Generated by `validation/make_numbers_md.py` from `reports/shared/numbers.tex`. "
         f"{len(used)} of the {len(macros)} generated macros appear in at least one report "
         f"({sum('AIHC' in v for v in used.values())} in AIHC, {sum('CS388' in v for v in used.values())} in CS388). "
         "The last column names the validation script that recomputes each value from the raw files "
         "(see `README.md` in this folder).",
         "",
         "Naming: `Ep` = E+ (question + abstract contexts), `Em` = E− (question only); "
         "`Verb`/`Cons`/`Iso` = s_verb/s_cons/s_iso; `Fifty`/`Seventy`/`Ninety` = the thresholds fitted for "
         "50/70/90% dev coverage; `Ph` = post hoc; `Tok` = token-probability baseline; a `CI` suffix is the "
         "95% paired bootstrap interval of the macro without it.",
         ""]
for title in sorted(rows):
    lines += [f"## {title}", "", "| Macro | Value | Meaning | Used in | Checked by |", "|---|---|---|---|---|",
              *rows[title], ""]
(HERE / "numbers.md").write_text("\n".join(lines))
print(f"wrote {HERE / 'numbers.md'}: {len(used)} macros, "
      f"{sum(1 for n in used if n not in step_of)} unchecked")
