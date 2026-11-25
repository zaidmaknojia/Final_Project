"""Step 1: the data. Checksums, item counts, the dev/test split and the gold labels."""

from collections import Counter

from vcommon import ROOT, check, fact, finish, header, items, load_json, sha256, split_ids

header("Step 1: data files match the pinned checksums")
manifest = load_json("data/manifest.json")
print(f"source {manifest['source']} @ {manifest['revision']}")
for name, want in manifest["files"].items():
    got = sha256(ROOT / "data/raw" / name)
    fact(f"sha256 data/raw/{name}", got == want, got[:12])

header("Step 1: items and split")
it = items()
ids = split_ids()
dev, test = set(ids["dev"]), set(ids["test"])
fact("1,000 PQA-L items", len(it) == 1000, str(len(it)))
fact("500 dev and 500 test ids", len(dev) == 500 and len(test) == 500, f"{len(dev)} / {len(test)}")
fact("dev and test are disjoint", not dev & test)
fact("dev + test cover every item", dev | test == set(it))
truth = load_json("data/raw/test_ground_truth.json")
fact("test ids equal the official test_ground_truth ids", set(truth) == test)
fact("official test labels equal final_decision", all(truth[i].lower() == it[i]["gold"] for i in test))

header("Step 1: gold label counts")
gold_test = Counter(it[i]["gold"] for i in test)
gold_dev = Counter(it[i]["gold"] for i in dev)
print(f"test: {dict(gold_test)}   dev: {dict(gold_dev)}")
check("GoldYes", gold_test["yes"])
check("GoldNo", gold_test["no"])
check("GoldMaybe", gold_test["maybe"])
check("PhMajorityAcc", gold_test["yes"] / 500, "(always answer yes)")

header("Step 1: conclusion sentences repeated inside the contexts (dataset quirk)")
found = {}
for split, pool in (("dev", dev), ("test", test)):
    hits = []
    for i in sorted(pool):
        ctx = " ".join(it[i]["contexts"])
        sentences = [s.strip().rstrip(".") for s in it[i]["long_answer"].split(". ")]
        if any(len(s) > 40 and s in ctx for s in sentences):  # a whole conclusion sentence, verbatim
            hits.append(i)
    found[split] = hits
    print(f"{split}: {hits}")
fact("2 dev items repeat a whole conclusion sentence", len(found["dev"]) == 2, str(found["dev"]))
fact("no test item does", not found["test"])
finish()
