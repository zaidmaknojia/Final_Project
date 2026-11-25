import json

import pytest

from pubmedqa_abst import cli


def test_smoke_runs_two_items(fake_env):
    tmp, dirs = fake_env
    cli.main(dirs + ["smoke"])
    raws = list((tmp / "runs" / "_smoke").glob("*/raw.jsonl"))
    assert len(raws) == 1 and len(raws[0].read_text().splitlines()) == 2 * 3


def test_pilot_writes_report(fake_env):
    tmp, dirs = fake_env
    cli.main(dirs + ["pilot", "--n", "6"])
    report = json.loads((tmp / "runs" / "_pilot" / "pilot_report.json").read_text())
    assert report["greedy_deterministic"] == "5/5" and report["K"] == 2
    assert set(report["conditions"]) == {"E+", "E-"} and report["eta_hours_full_run"] > 0


def test_stratified_ids_balances_labels(items):
    from pubmedqa_abst.data import load_split
    from conftest import ROOT

    split = load_split(ROOT / "data" / "raw", ROOT / "data" / "manifest.json")
    ids = cli.stratified_ids(split["dev"], items, 9, 1)
    assert sorted(items[i].gold for i in ids) == ["maybe"] * 3 + ["no"] * 3 + ["yes"] * 3


def test_full_locked_flow(fake_env):
    tmp, dirs = fake_env
    cli.main(dirs + ["lock-protocol"])
    for cond in ("E+", "E-"):
        cli.main(dirs + ["run", "--split", "dev", "--condition", cond])
    cli.main(dirs + ["fit"])
    cli.main(dirs + ["lock-evaluation"])
    for cond in ("E+", "E-"):
        cli.main(dirs + ["run", "--split", "test", "--condition", cond])
    cli.main(dirs + ["evaluate"])
    assert (tmp / "results" / "metrics.json").exists() and (tmp / "results" / "bootstrap.json").exists()


def test_fit_without_runs_exits(fake_env):
    _, dirs = fake_env
    with pytest.raises(SystemExit):
        cli.main(dirs + ["fit"])


def test_lock_protocol_refuses_overwrite_without_force(fake_env):
    tmp, dirs = fake_env
    cli.main(dirs + ["lock-protocol"])
    with pytest.raises(SystemExit):
        cli.main(dirs + ["lock-protocol"])
    cli.main(dirs + ["--force", "lock-protocol"])
    log = (tmp / "freeze" / "overrides.jsonl").read_text().splitlines()
    assert len(log) == 1 and json.loads(log[0])["command"] == "lock-protocol"


def test_fit_refused_once_test_run_exists(fake_env):
    tmp, dirs = fake_env
    d = tmp / "runs" / "abc123"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"split": "test"}))
    for cmd in (["fit"], ["lock-protocol"], ["lock-evaluation"]):
        with pytest.raises(SystemExit):
            cli.main(dirs + cmd)


def test_run_exits_nonzero_when_transport_errors_remain(fake_env, monkeypatch):
    from helpers import FakeOllama
    from pubmedqa_abst.client import CallResult

    class FlakyOllama(FakeOllama):
        def generate(self, prompt, temperature, seed, schema):
            if "question" in prompt.lower() and temperature == 0 and "Abstract" in prompt and len(prompt) % 7 == 0:
                return CallResult(None, 0, 0.0, 4, "ConnectionError: x")
            return super().generate(prompt, temperature, seed, schema)

    monkeypatch.setattr(cli, "OllamaClient", FlakyOllama)
    _, dirs = fake_env
    with pytest.raises(SystemExit) as exc:
        cli.main(dirs + ["run", "--split", "dev", "--condition", "E+"])
    assert exc.value.code != 0
