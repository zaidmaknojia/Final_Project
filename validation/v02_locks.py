"""Step 2: the pre-registration locks. Re-hash every locked file and check the timeline."""

import json
import subprocess

from vcommon import ROOT, fact, finish, header, load_json, run_dirs, sha256

LOCKS = ("freeze/protocol.lock.json", "freeze/evaluation.lock.json")

for lock in LOCKS:
    data = load_json(lock)
    header(f"Step 2: {lock} (created {data['created']})")
    for key, want in sorted(data["files"].items()):
        base, rel = key.split(":", 1)
        path = ROOT / rel if base == "root" else ROOT / "freeze" / rel
        fact(f"{key}", path.exists() and sha256(path) == want, want[:12])

proto = load_json("freeze/protocol.lock.json")
print("\nprotocol lock generation:", json.dumps(proto["generation"]))
print("protocol lock analysis:  ", json.dumps(proto["analysis"]))
print("planned comparisons:     ", proto["planned_comparisons"])

header("Step 2: timeline (UTC)")
events = [("protocol lock", proto["created"])]
for (split, cond), d in sorted(run_dirs().items(), key=lambda kv: json.loads((kv[1] / "manifest.json").read_text())["started"]):
    rows = (d / "raw.jsonl").read_text().splitlines()
    first, last = json.loads(rows[0])["ts"], json.loads(rows[-1])["ts"]
    events.append((f"{split} {cond} run {d.name} first call", first))
    events.append((f"{split} {cond} run {d.name} last call", last))
events.append(("evaluation lock", load_json("freeze/evaluation.lock.json")["created"]))
events.sort(key=lambda e: e[1])
for name, ts in events:
    print(f"  {ts[:19]}  {name}")
order = [name.split(" run")[0] if "run" in name else name for name, _ in events]
fact("protocol lock precedes every model call", order[0] == "protocol lock")
eval_at = [i for i, (n, _) in enumerate(events) if n == "evaluation lock"][0]
fact("both dev runs finish before the evaluation lock",
     all("dev" in n for n, _ in events[1:eval_at]) and sum("dev" in n for n, _ in events[1:eval_at]) == 4)
fact("both test runs start after the evaluation lock", all("test" in n for n, _ in events[eval_at + 1:]))

header("Step 2: commits that recorded the locks (local time)")
log = subprocess.run(["git", "log", "--reverse", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M %z",
                      "--", "freeze/protocol.lock.json", "freeze/evaluation.lock.json"],
                     cwd=ROOT, capture_output=True, text=True).stdout
print(log.strip())
finish()
