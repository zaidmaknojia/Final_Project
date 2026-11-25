"""Step 8: reproducibility. Rerun the project's own pipeline into a temporary folder and
compare every output with the committed one; then rebuild both PDFs from the regenerated
numbers. Unlike steps 1-7 this step runs the project code (that is the point of it).
Nothing in the repository is written. Takes a few minutes (the locked bootstrap).
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from vcommon import ROOT, fact, finish, header

PY = str(ROOT / ".venv/bin/python")
TMP = Path(tempfile.mkdtemp(prefix="pubmedqa-validate-"))
RES = TMP / "results"


def run(args: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def same_json(a, b, tol: float = 1e-12) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol or a == b
    return a == b


print(f"working in {TMP}")

header("Step 8a: locks and tests")
r = run([PY, "-c", "from pathlib import Path; from pubmedqa_abst.freeze import verify_locks; "
                   "verify_locks(Path('.'), Path('freeze'))"])
fact("project lock check (verify_locks) raises nothing", r.returncode == 0, r.stderr.strip()[-200:])
r = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
summary = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
fact("test suite passes", r.returncode == 0, summary)

header("Step 8b: rerun the locked evaluation into the temporary folder")
r = run([PY, "-m", "pubmedqa_abst.cli", "--results-dir", str(RES), "evaluate"])
fact("evaluate ran (it refuses if any locked file changed)", r.returncode == 0, r.stderr.strip()[-200:])
for name in ("metrics.json", "bootstrap.json"):
    fact(f"results/{name} reproduced", same_json(json.loads((RES / name).read_text()),
                                                  json.loads((ROOT / "results" / name).read_text())))
for name in ("tables.csv", "per_class.csv", "confident_errors.csv", "confident_error_examples.csv"):
    fact(f"results/{name} reproduced", (RES / name).read_text() == (ROOT / "results" / name).read_text())

header("Step 8c: rerun the post hoc analyses and the token baseline summary (no model calls)")
r = run([PY, "-m", "pubmedqa_abst.posthoc", str(RES)])
fact("posthoc ran", r.returncode == 0, r.stderr.strip()[-200:])
fact("results/posthoc.json reproduced", same_json(json.loads((RES / "posthoc.json").read_text()),
                                                  json.loads((ROOT / "results/posthoc.json").read_text())))

header("Step 8d: regenerate the report macros and tables")
shared = TMP / "reports/shared"
r = run([PY, "-m", "pubmedqa_abst.report_numbers", str(RES), str(shared)])
fact("report_numbers ran", r.returncode == 0, r.stderr.strip()[-200:])
for f in sorted((ROOT / "reports/shared").glob("*.tex")):
    fact(f"reports/shared/{f.name} identical", (shared / f.name).read_text() == f.read_text())

header("Step 8e: rebuild both PDFs from the regenerated numbers")
shutil.copytree(ROOT / "results/figs", RES / "figs", dirs_exist_ok=True)
shutil.copy(ROOT / "reports/shared/refs.bib", shared / "refs.bib")
for report in ("aihc", "nlp-neurips"):
    dst = TMP / "reports" / report
    dst.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "reports" / report).glob("*"):
        if f.suffix in (".tex", ".sty"):
            shutil.copy(f, dst / f.name)
    r = run(["tectonic", "main.tex"], cwd=dst)
    built = dst / "main.pdf"
    fact(f"{report}: tectonic build", r.returncode == 0 and built.exists(), r.stderr.strip()[-200:])
    if not built.exists():
        continue
    pages = {p: re.search(r"Pages:\s+(\d+)", run(["pdfinfo", str(p)]).stdout).group(1)
             for p in (built, ROOT / "reports" / report / "main.pdf")}
    text_new = run(["pdftotext", "-layout", str(built), "-"]).stdout
    text_old = run(["pdftotext", "-layout", str(ROOT / "reports" / report / "main.pdf"), "-"]).stdout
    print(f"     {report}: rebuilt {pages[built]} pages, committed {list(pages.values())[1]} pages")
    fact(f"{report}: committed PDF text equals the rebuilt PDF text", text_new == text_old)
    if report == "nlp-neurips":
        refs = text_new.find("References")
        body_end = text_new[:refs].count("\f") + 1 if refs >= 0 else None
        print(f"     nlp-neurips: 'References' starts on page {body_end} (limit: body within 8 pages)")

print(f"\nTemporary outputs kept in {TMP} for inspection; delete it when done.")
finish()
