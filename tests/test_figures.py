import pytest

from pubmedqa_abst.figures import make_figures, step_points

EXPECTED = (
    ["reliability_E+_s_verb", "reliability_E+_s_cons", "reliability_E+_s_verb_iso",
     "reliability_E-_s_verb", "reliability_E-_s_cons", "reliability_E-_s_verb_iso",
     "risk_coverage_E+", "risk_coverage_E-", "per_class_coverage", "evidence_deltas", "workflow"]
)


def test_make_figures_writes_all_files(results_dir):
    paths = make_figures(results_dir)
    figs = results_dir / "figs"
    for stem in EXPECTED:
        for ext in ("pdf", "png"):
            f = figs / f"{stem}.{ext}"
            assert f.exists() and f.stat().st_size > 0, f
    assert len(paths) == 2 * len(EXPECTED)


def test_make_figures_idempotent(results_dir):
    first = make_figures(results_dir)
    assert make_figures(results_dir) == first


def test_step_points_match_aurc_convention():
    xs, ys = step_points([(0.9, 0.5, 0.5), (0.5, 0.75, 1 / 3)])
    assert xs == [0.0, 0.5, 0.75] and ys == pytest.approx([0.5, 0.5, 1 / 3])
