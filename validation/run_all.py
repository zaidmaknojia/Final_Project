"""Run steps 1-7, then confirm that every number the two reports use was recomputed.

Also lists every numeral typed directly into either report (outside the generated
macros), so you can judge by eye whether each is a design constant (K = 10, 500 items,
a year) or a result that should have been a macro.

Run from anywhere:  .venv/bin/python validation/run_all.py
Step 8 (full pipeline rerun and PDF rebuild) is separate:  validation/v08_reproduce.py
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

checked, failed = set(), 0
for step in STEPS:
    out = subprocess.run([sys.executable, str(HERE / step)], cwd=HERE, capture_output=True, text=True)
    lines = out.stdout.splitlines()
    print(f"{step:<24} {lines[-1] if lines else out.stderr.strip()[-300:]}")
    failed += out.returncode != 0
    checked |= {m.group(1) for line in lines if (m := re.match(r"PASS\s+(\w+)\s+report=", line))}

macros = set(re.findall(r"\\newcommand\{\\(\w+)\}", (ROOT / "reports/shared/numbers.tex").read_text()))
used = {name: set() for name in macros}
for label, path in REPORTS.items():
    for name in re.findall(r"\\([A-Za-z]+)", path.read_text()):
        if name in used:
            used[name].add(label)
used = {k: v for k, v in used.items() if v}
missing = sorted(set(used) - checked)
print(f"\n{len(used)} macros appear in the reports; {len(used) - len(missing)} were recomputed by steps 1-7.")
for name in missing:
    print(f"  NOT RECOMPUTED: {name}")

print("\nNumerals typed directly in the report bodies (review by eye):")
STRIP = [r"%.*", r"\\(cite[pt]?|ref|label|includegraphics|tablerows|input|bibliography\w*|url|href)(\[[^\]]*\])?\{[^}]*\}",
         r"\\[A-Za-z]+\{\}", r"\{?[\d.]+\\(textwidth|linewidth|columnwidth)\}?", r"\\(v|h)space\*?\{[^}]*\}",
         r"\[[\d.]+(pt|em|ex|cm|mm|in)\]", r"\\[A-Za-z]+"]
for label, path in REPORTS.items():
    text = path.read_text()
    start = text[:text.index(r"\begin{document}")].count("\n") + 1
    for n, line in enumerate(text.splitlines()[start:], start + 1):
        bare = line
        for pattern in STRIP:
            bare = re.sub(pattern, " ", bare)
        nums = re.findall(r"(?<![A-Za-z\d])\d[\d{},.]*", bare)
        if nums:
            print(f"  {label} main.tex:{n}: {' '.join(dict.fromkeys(nums))}")

print("\nOVERALL:", "PASS" if not failed and not missing else "CHECK THE LINES ABOVE")
sys.exit(1 if failed or missing else 0)
