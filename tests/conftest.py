import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def manifest() -> Path:
    return ROOT / "data" / "manifest.json"


@pytest.fixture(scope="session")
def template() -> str:
    return (ROOT / "configs" / "prompts" / "v1.txt").read_text()


@pytest.fixture(scope="session")
def raw_json(raw_dir):
    return json.loads((raw_dir / "ori_pqal.json").read_text())


@pytest.fixture(scope="session")
def items(raw_dir, manifest):
    from pubmedqa_abst.data import load_items

    return load_items(raw_dir, manifest)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """CLI wired to FakeOllama, K=2, B=100, and tmp freeze/runs/results dirs."""
    from helpers import FakeOllama
    from pubmedqa_abst import cli
    from pubmedqa_abst.data import load_config

    monkeypatch.setattr(cli, "OllamaClient", FakeOllama)
    monkeypatch.setattr(cli, "load_config", lambda path: {**load_config(path), "K": 2, "bootstrap_B": 100})
    dirs = ["--freeze-dir", str(tmp_path / "freeze"), "--runs-dir", str(tmp_path / "runs"),
            "--results-dir", str(tmp_path / "results")]
    return tmp_path, dirs


@pytest.fixture
def results_dir(fake_env):
    """A complete locked fake run; returns the results directory."""
    from pubmedqa_abst import cli

    tmp, dirs = fake_env
    cli.main(dirs + ["lock-protocol"])
    for cond in ("E+", "E-"):
        cli.main(dirs + ["run", "--split", "dev", "--condition", cond])
    cli.main(dirs + ["fit", ])
    cli.main(dirs + ["lock-evaluation"])
    for cond in ("E+", "E-"):
        cli.main(dirs + ["run", "--split", "test", "--condition", cond])
    cli.main(dirs + ["evaluate"])
    return tmp / "results"
