import csv
import json

import pytest

from conftest import ROOT
from helpers import raw_row, write_raw
from pubmedqa_abst.data import load_config
from pubmedqa_abst.evaluate import evaluate, fit_from_dev
from pubmedqa_abst.freeze import PLANNED_COMPARISONS, write_evaluation_lock, write_protocol_lock

K = 2
# (answer offset, confidence) patterns give a mix of right/wrong and score levels.
PATTERNS = [(True, 95), (True, 80), (False, 90), (True, 60), (False, 40), (True, 70)]


def make_rows(item_list, condition, invalid_last=False):
    rows = []
    for n, (it, (right, conf)) in enumerate(zip(item_list, PATTERNS)):
        wrong = "no" if it.gold != "no" else "yes"
        a0 = it.gold if right else wrong
        if invalid_last and n == len(item_list) - 1:
            rows.append(raw_row(it.id, 0, status="invalid_format", condition=condition))
        else:
            rows.append(raw_row(it.id, 0, answer=a0, confidence=conf, condition=condition))
        rows.append(raw_row(it.id, 1, answer=a0, condition=condition))
        rows.append(raw_row(it.id, 2, answer=a0 if n % 2 else wrong, condition=condition))
    return rows


@pytest.fixture
def setup(tmp_path, items):
    ids = sorted(items)
    dev_items = [items[i] for i in ids[:6]]
    test_items = [items[i] for i in ids[6:12]]
    runs = {}
    for split, its in (("dev", dev_items), ("test", test_items)):
        for cond in ("E+", "E-"):
            path = tmp_path / "runs" / f"{split}{cond}" / "raw.jsonl"
            runs[(split, cond)] = write_raw(path, make_rows(its, cond, invalid_last=(cond == "E-")))
    config = {**load_config(ROOT / "configs/run.yaml"), "K": K, "bootstrap_B": 200}
    freeze = tmp_path / "freeze"
    dev_runs = {c: runs[("dev", c)] for c in ("E+", "E-")}
    test_runs = {c: runs[("test", c)] for c in ("E+", "E-")}
    expected = {"dev": [i.id for i in dev_items], "test": [i.id for i in test_items]}
    return dict(items=items, config=config, freeze=freeze, dev=dev_runs, test=test_runs,
                out=tmp_path / "results", expected=expected)


def test_evaluate_refuses_without_locks(setup):
    s = setup
    fit_from_dev(s["dev"], s["items"], s["config"], s["freeze"], s["expected"]["dev"])
    with pytest.raises(RuntimeError):
        evaluate(s["dev"], s["test"], s["items"], s["config"], s["freeze"], s["out"], s["expected"], root=ROOT)


def test_evaluate_end_to_end_on_fixture(setup):
    s = setup
    write_protocol_lock(ROOT, s["freeze"], s["config"])
    fit_from_dev(s["dev"], s["items"], s["config"], s["freeze"], s["expected"]["dev"])
    thresholds = json.loads((s["freeze"] / "thresholds.json").read_text())
    assert set(thresholds) == {"primary", "secondary"}
    assert set(thresholds["primary"]) == {"s_verb", "s_cons", "s_verb_iso"}
    write_evaluation_lock(ROOT, s["freeze"])
    evaluate(s["dev"], s["test"], s["items"], s["config"], s["freeze"], s["out"], s["expected"], root=ROOT)

    metrics = json.loads((s["out"] / "metrics.json").read_text())
    for cond in ("E+", "E-"):
        assert "accuracy" in metrics["test"][cond]["full"]
        for score in ("s_verb", "s_cons", "s_verb_iso"):
            block = metrics["test"][cond]["scores"][score]
            assert "accept_all" in block and "0.7" in block["primary"] and "0.7" in block["secondary"]
            assert "curve" in block and "per_class_gold" in block["primary"]["0.7"]
    assert metrics["test"]["E-"]["full"]["invalid_rate"] == pytest.approx(1 / 6)
    for score in ("s_verb", "s_cons", "s_verb_iso"):
        block = metrics["test"]["E-"]["scores"][score]
        assert block["accept_all"]["coverage"] == 1.0
        assert block["accept_all"]["accuracy"] == pytest.approx(metrics["test"]["E-"]["full"]["accuracy"])
        assert block["curve"][-1][1] == pytest.approx(1.0)
        assert "accept_all_valid" in block
    assert (s["freeze"] / "dev_confident_errors.csv").exists()

    boot = json.loads((s["out"] / "bootstrap.json").read_text())
    assert set(PLANNED_COMPARISONS) <= set(boot)
    assert all(boot[c] for c in PLANNED_COMPARISONS)

    for name in ("per_class.csv", "confident_errors.csv", "tables.csv"):
        assert (s["out"] / name).exists()
    with open(s["out"] / "tables.csv") as fh:
        assert len(list(csv.DictReader(fh))) > 0


def test_fit_refuses_incomplete_run(setup):
    s = setup
    raw = s["dev"]["E-"]
    first_id = s["expected"]["dev"][0]
    kept = [l for l in raw.read_text().splitlines() if json.loads(l)["item_id"] != first_id]
    raw.write_text("\n".join(kept) + "\n")
    with pytest.raises(RuntimeError, match="incomplete"):
        fit_from_dev(s["dev"], s["items"], s["config"], s["freeze"], s["expected"]["dev"])
