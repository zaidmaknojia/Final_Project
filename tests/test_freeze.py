import json
import shutil

import pytest

from conftest import ROOT
from pubmedqa_abst.data import load_config
from pubmedqa_abst.freeze import (
    EVAL_ROOT_FILES, PLANNED_COMPARISONS, PROTOCOL_FILES, verify_locks, write_evaluation_lock, write_protocol_lock,
)


@pytest.fixture
def fake_root(tmp_path):
    for rel in (*PROTOCOL_FILES, *EVAL_ROOT_FILES):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dst)
    return tmp_path


def lock_all(root):
    freeze = root / "freeze"
    write_protocol_lock(root, freeze, load_config(root / "configs/run.yaml"))
    (freeze / "thresholds.json").write_text(json.dumps({"primary": {}}))
    (freeze / "isotonic.json").write_text(json.dumps({"x": [0, 1], "y": [0, 1]}))
    write_evaluation_lock(root, freeze)
    return freeze


def test_verify_locks_passes_when_unchanged(fake_root):
    verify_locks(fake_root, lock_all(fake_root))


def test_verify_locks_detects_prompt_change(fake_root):
    freeze = lock_all(fake_root)
    p = fake_root / "configs/prompts/v1.txt"
    p.write_text(p.read_text() + "\nchanged")
    with pytest.raises(RuntimeError, match="v1.txt"):
        verify_locks(fake_root, freeze)


def test_verify_locks_detects_threshold_change(fake_root):
    freeze = lock_all(fake_root)
    (freeze / "thresholds.json").write_text("{}")
    with pytest.raises(RuntimeError, match="thresholds.json"):
        verify_locks(fake_root, freeze)


def test_verify_locks_missing_evaluation_lock(fake_root):
    freeze = fake_root / "freeze"
    write_protocol_lock(fake_root, freeze, load_config(fake_root / "configs/run.yaml"))
    with pytest.raises(RuntimeError, match="evaluation.lock.json"):
        verify_locks(fake_root, freeze)


def test_protocol_lock_records_planned_comparisons(fake_root):
    freeze = lock_all(fake_root)
    lock = json.loads((freeze / "protocol.lock.json").read_text())
    assert lock["planned_comparisons"] == PLANNED_COMPARISONS and lock["generation"]["K"] == 10


def test_cli_run_test_refuses_without_lock(tmp_path, monkeypatch):
    from pubmedqa_abst import cli

    def boom(*a, **k):
        raise AssertionError("client must not be built")

    monkeypatch.setattr(cli, "OllamaClient", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--freeze-dir", str(tmp_path / "freeze"), "run", "--split", "test", "--condition", "E+"])
    assert exc.value.code != 0


@pytest.mark.parametrize("rel", ["src/pubmedqa_abst/data.py", "src/pubmedqa_abst/client.py",
                                 "src/pubmedqa_abst/runner.py", "src/pubmedqa_abst/metrics.py",
                                 "src/pubmedqa_abst/stats.py", "src/pubmedqa_abst/evaluate.py"])
def test_verify_locks_covers_generation_and_analysis_code(fake_root, rel):
    freeze = lock_all(fake_root)
    f = fake_root / rel
    f.write_text(f.read_text() + "\n# edit")
    with pytest.raises(RuntimeError, match=rel.split("/")[-1]):
        verify_locks(fake_root, freeze)
