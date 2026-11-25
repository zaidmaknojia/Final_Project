import shutil

import pytest

from pubmedqa_abst.data import build_prompt, load_config, load_items, load_split


def test_split_sizes_and_disjoint(raw_dir, manifest):
    s = load_split(raw_dir, manifest)
    assert len(s["dev"]) == 500 and len(s["test"]) == 500
    assert not set(s["dev"]) & set(s["test"])


def test_checksum_mismatch_raises(tmp_path, raw_dir, manifest):
    for name in ("ori_pqal.json", "test_ground_truth.json", "LICENSE"):
        shutil.copy(raw_dir / name, tmp_path / name)
    p = tmp_path / "ori_pqal.json"
    data = bytearray(p.read_bytes())
    data[10] = ord("X") if data[10] != ord("X") else ord("Y")
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        load_items(tmp_path, manifest)


def test_e_minus_prompt_has_no_context_and_no_long_answer(items, template):
    it = next(iter(items.values()))
    p = build_prompt(it, "E-", template)
    assert it.question in p and all(c not in p for c in it.contexts)


def test_e_plus_prompt_has_contexts_never_long_answer(items, template, raw_json):
    for iid, it in list(items.items())[:20]:
        p = build_prompt(it, "E+", template)
        assert all(c in p for c in it.contexts)
        assert raw_json[iid]["LONG_ANSWER"] not in p


def test_gold_labels_valid(items):
    assert {i.gold for i in items.values()} <= {"yes", "no", "maybe"}
    assert len(items) == 1000


def test_unknown_condition_rejected(items, template):
    with pytest.raises(ValueError):
        build_prompt(next(iter(items.values())), "E?", template)


def test_load_config_has_spec_values():
    from conftest import ROOT

    cfg = load_config(ROOT / "configs" / "run.yaml")
    assert cfg["coverage_targets"] == [0.5, 0.7, 0.9]
    assert cfg["model"] == "gemma4:e4b"
