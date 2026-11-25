"""Report figures from results/metrics.json and bootstrap.json (static PDF + PNG).

Palette: dataviz reference categorical slots 1-3, validated light mode
(CVD worst ΔE 9.2; aqua < 3:1 contrast, so every figure carries a legend and
distinct line styles/markers as secondary encoding). Color follows the score.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# TrueType (Type 42) fonts in PDFs: embeddable and searchable, unlike Type 3.
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

CONDITIONS = ("E+", "E-")
SCORES = ("s_verb", "s_cons", "s_verb_iso")
LABELS = {"s_verb": "Stated confidence", "s_cons": "Sample consistency", "s_verb_iso": "Stated (isotonic)"}
COLORS = {"s_verb": "#2a78d6", "s_cons": "#eb6834", "s_verb_iso": "#1baf7a"}
STYLES = {"s_verb": ("-", "o"), "s_cons": ("--", "s"), "s_verb_iso": (":", "^")}
COND_LABELS = {"E+": "with abstract (E+)", "E-": "question only (E−)"}
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"
FOCUS_TARGET = "0.7"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, out: Path, stem: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "png"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def reliability(block: dict, cond: str, score: str, out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(2.7, 2.6))
    _style(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Perfect")
    for b in block["ece_bins"]:
        if not b["count"]:
            continue
        width = (b["hi"] - b["lo"]) * 0.9
        ax.bar(b["lo"] + 0.05 * (b["hi"] - b["lo"]), b["acc"], width=width, align="edge",
               color=COLORS[score], edgecolor="white", linewidth=1)
        ax.text((b["lo"] + b["hi"]) / 2, b["acc"] + 0.02, str(b["count"]), ha="center", fontsize=8, color=INK)
    ece_txt = "n/a" if block["ece"] is None else f"{block['ece']:.3f}"
    ax.set(xlim=(0, 1), ylim=(0, 1.12))
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_ylabel("Accuracy in bin", fontsize=10)
    ax.set_title(f"{LABELS[score]}, {cond}\nECE {ece_txt}", fontsize=10, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    return _save(fig, out, f"reliability_{cond}_{score}")


def step_points(curve: list) -> tuple[list[float], list[float]]:
    """Points for ax.step(where="pre"): risk r_i covers coverage (c_{i-1}, c_i],
    exactly the area AURC = sum (c_i - c_{i-1}) r_i integrates, starting at 0."""
    xs = [0.0] + [c for _, c, _ in curve]
    ys = [curve[0][2]] + [r for _, _, r in curve] if curve else [0.0]
    return xs, ys


def risk_coverage_plot(test: dict, cond: str, out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    _style(ax)
    for score in SCORES:
        block = test[cond]["scores"][score]
        if not block["curve"]:
            continue
        cov, risk = step_points(block["curve"])
        ls, mk = STYLES[score]
        aurc = "n/a" if block["aurc"] is None else f"{block['aurc']:.3f}"
        ax.step(cov, risk, where="pre", color=COLORS[score], linestyle=ls, linewidth=1.6,
                label=f"{LABELS[score]} (AURC {aurc})")
        for target, entry in block["primary"].items():
            if entry["accuracy"] is not None:
                ax.plot(entry["coverage"], 1 - entry["accuracy"], marker=mk, markersize=6,
                        color=COLORS[score], markeredgecolor="white", linestyle="none")
    ax.set(xlim=(0, 1), ylim=(0, None), xlabel="Coverage (share of all items answered)",
           ylabel="Risk (error among answered)")
    ax.set_title(f"Risk–coverage, {COND_LABELS[cond]}\nmarkers: frozen dev thresholds (50/70/90%)",
                 fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, out, f"risk_coverage_{cond}")


def per_class_plot(test: dict, out: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    classes, width = ("yes", "no", "maybe"), 0.26
    for ax, cond in zip(axes, CONDITIONS):
        _style(ax)
        for j, score in enumerate(SCORES):
            table = test[cond]["scores"][score]["primary"][FOCUS_TARGET]["per_class_gold"]
            vals = [(table.get(c) or {}).get("accepted_share") or 0 for c in classes]
            xs = [i + (j - 1) * width for i in range(len(classes))]
            ax.bar(xs, vals, width=width * 0.92, color=COLORS[score], label=LABELS[score],
                   hatch=["", "///", "..."][j], edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(classes)), [f"gold {c}" for c in classes])
        ax.set_title(COND_LABELS[cond], fontsize=9, color=INK)
    axes[0].set_ylabel("Share accepted at 70% threshold")
    axes[0].set_ylim(0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    return _save(fig, out, "per_class_coverage")


def evidence_deltas_plot(boot: dict, out: Path) -> list[Path]:
    group = boot.get("eplus_vs_eminus", {})
    metrics = [("coverage", "Coverage"), ("accuracy", "Accuracy (answered)"), ("cer", "Confident-error rate")]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    _style(ax)
    for j, score in enumerate(SCORES):
        for i, (key, _) in enumerate(metrics):
            entry = group.get(f"{score}|{key}@{FOCUS_TARGET}")
            if not entry or entry["point"] is None or entry["lo"] is None:
                continue
            y = i + (j - 1) * 0.22
            ax.errorbar(entry["point"], y, xerr=[[entry["point"] - entry["lo"]], [entry["hi"] - entry["point"]]],
                        fmt=STYLES[score][1], color=COLORS[score], markersize=6, capsize=2, linewidth=1.4,
                        label=LABELS[score] if i == 0 else None)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(metrics)), [m[1] for m in metrics])
    ax.set_xlabel("E+ minus E− (95% paired bootstrap CI)")
    ax.set_title("Effect of removing the abstract, 70% threshold", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    return _save(fig, out, "evidence_deltas")


WORKFLOW_STEPS = [
    "PubMedQA PQA-L\n1,000 expert items\n500 dev / 500 test",
    "Two prompts\nE+: question + abstract\nE\u2212: question only",
    "gemma4:e4b via Ollama\n1 greedy + K samples\nJSON answer + confidence",
    "Scores on greedy answer\nstated confidence\nsample consistency",
    "Dev only: fit thresholds\n(50/70/90% coverage)\n+ isotonic map, then lock",
    "Test once: calibration,\nrisk\u2013coverage, bootstrap\nCIs, error analysis",
]
# Snake layout: row 1 left->right, row 2 right->left, so every arrow is short.
WORKFLOW_POS = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]


def _box_xy(col: int, row: int, w: float, h: float) -> tuple[float, float]:
    return 0.03 + col * 0.33, 0.08 + row * 0.47


def workflow_plot(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.set_axis_off()
    w, h = 0.27, 0.34
    for i, (text, (col, row)) in enumerate(zip(WORKFLOW_STEPS, WORKFLOW_POS)):
        x, y = _box_xy(col, row, w, h)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    facecolor="#cde2fb" if i < 4 else "#f9d9cb", edgecolor=MUTED, linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, f"{i + 1}. {text}", ha="center", va="center", fontsize=7, color=INK)
        if i + 1 < len(WORKFLOW_STEPS):
            ncol, nrow = WORKFLOW_POS[i + 1]
            nx, ny = _box_xy(ncol, nrow, w, h)
            if nrow == row:
                start = (x + w, y + h / 2) if ncol > col else (x, y + h / 2)
                end = (nx, ny + h / 2) if ncol > col else (nx + w, ny + h / 2)
            else:
                start, end = (x + w / 2, y), (nx + w / 2, ny + h)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=MUTED, linewidth=1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Study workflow (blue: inference and scoring; orange: locked evaluation)", fontsize=9, color=INK)
    return _save(fig, out, "workflow")


def make_figures(results_dir: Path) -> list[Path]:
    results_dir = Path(results_dir)
    out = results_dir / "figs"
    out.mkdir(parents=True, exist_ok=True)
    test = json.loads((results_dir / "metrics.json").read_text())["test"]
    boot = json.loads((results_dir / "bootstrap.json").read_text())
    paths: list[Path] = []
    for cond in CONDITIONS:
        for score in SCORES:
            paths += reliability(test[cond]["scores"][score], cond, score, out)
    for cond in CONDITIONS:
        paths += risk_coverage_plot(test, cond, out)
    paths += per_class_plot(test, out)
    paths += evidence_deltas_plot(boot, out)
    paths += workflow_plot(out)
    return paths
