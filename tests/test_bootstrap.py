import numpy as np

from pubmedqa_abst.bootstrap import diff, paired_bootstrap

IDS = [str(i) for i in range(20)]
VALUES = np.arange(20, dtype=float)


def mean_metric(idx):
    return float(VALUES[idx].mean())


def test_bootstrap_deterministic_with_seed():
    a = paired_bootstrap(IDS, {"m": mean_metric}, B=200, seed=7)
    b = paired_bootstrap(IDS, {"m": mean_metric}, B=200, seed=7)
    assert a == b
    assert a["m"]["point"] == VALUES.mean() and a["m"]["lo"] < a["m"]["point"] < a["m"]["hi"]
    assert a["m"]["n_valid"] == 200


def test_bootstrap_null_replicates_counted():
    def sometimes_none(idx):
        return None if idx[0] % 2 == 0 else float(VALUES[idx].mean())

    out = paired_bootstrap(IDS, {"m": sometimes_none}, B=400, seed=1)["m"]
    assert 0 < out["n_valid"] < 400
    assert np.isfinite(out["lo"]) and np.isfinite(out["hi"])


def test_bootstrap_all_null_gives_none_ci():
    out = paired_bootstrap(IDS, {"m": lambda idx: None}, B=50, seed=1)["m"]
    assert out == {"point": None, "lo": None, "hi": None, "n_valid": 0}


def test_bootstrap_paired_same_indices():
    seen_a, seen_b = [], []

    def a(idx):
        seen_a.append(idx.copy())
        return 0.0

    def b(idx):
        seen_b.append(idx.copy())
        return 0.0

    paired_bootstrap(IDS, {"a": a, "b": b}, B=30, seed=3)
    assert len(seen_a) == len(seen_b) == 31
    assert all((x == y).all() for x, y in zip(seen_a, seen_b))


def test_diff_propagates_none():
    f = diff(lambda idx: 1.0, lambda idx: None)
    assert f(np.arange(3)) is None
    assert diff(lambda i: 3.0, lambda i: 1.0)(np.arange(3)) == 2.0
