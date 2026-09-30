#!/usr/bin/env python3
"""Render terminology-normalized AI-residual figures from accepted audit tables.

This renderer does not fit, tune, or score a model. It reuses the accepted
prediction and audit tables and writes a versioned submission-audit export so
the previously reviewed figures remain unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "script"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

import build_ai_residual_story_figures as source


OUT_DIR = ROOT / "figure" / "paper_draft_v2" / "submission_v1_20260720"
REVISION_SUFFIX = "_submission_v1"
FINAL_WIDTH_IN = 183.0 / 25.4
MIN_FONT_PT = 5.5
AI3_VALIDATION_SELECTION_X: float | None = None

# Exact replacements avoid changing scientific statements or internal variable
# names. Long stack names are shortened to "final selected model" where space
# is constrained, while component names remain explicit elsewhere in the panel.
TEXT_REPLACEMENTS = {
    "Robust L1": "Robust median-offset\nregression (L1 loss)",
    "Robust\nL1": "Robust median-offset\nregression (L1 loss)",
    "Robust L1 regression": "Robust median-offset\nregression (L1 loss)",
    "HGB": "HGB regression",
    "Physics": "Physics-based\nbaseline",
    "Physics only": "Physics-based baseline",
    "Physics baseline": "Physics-based baseline",
    "Physics + robust L1": "Physics + robust median-offset\nregression (L1 loss)",
    "Physics + robust L1 + HGB": "Physics + AI residual model",
    "learned offset $\\beta_s$": "absolute-error calibration $\\beta_s$",
    "Before HGB": "Before HGB regression",
    "After HGB": "Final selected model",
    "Observed - model": "Observed - final selected model",
    "Observed - model\nerror (dB)": "Trend error\n(dB)",
    "+ robust L1": "+ robust median-offset\nregression (L1 loss)",
    "+ HGB": "+ HGB\nregression",
    "Robust L1  98.9%": "Median offset (L1 loss)  98.9%",
    "HGB 1.1%": "HGB regression 1.1%",
    "Post-L1 MSE\nremoved by HGB": "Post-offset MSE\nremoved by HGB regression",
    "Physical structure used by HGB": "Physical structure used by HGB regression",
    "Robust regression algorithm": "Median-offset choice (validation)",
    "HGB selection among trend regressors": "HGB learner choice (validation)",
    "Phase-resolved validation RMSE": "Validation RMSE by phase",
    "Validation-only HGB search (n=24)": "HGB tuning on validation (n=24)",
    "Final selected model across data partitions": "Post-selection evaluation",
    "Final selected model specification": "Selected model specification",
    "Full-model RMSE (dB-Hz)": "Final-model RMSE (dB-Hz)",
    "RMSE after robust-L1 correction (dB-Hz)": "Post-offset RMSE (dB-Hz)",
    "Learned robust-L1 correction $\\beta_s$ (dB)": "Robust median-offset coefficient $\\beta_s$ (dB; L1 loss)",
    "Validation": "Validation (selection)",
    "Complete-OP": "Complete-OP\n(post-selection)",
    "Test": "Internal test\n(post-selection)",
    "4 OP\nholdout": "4 OP holdout\n(post-selection)",
    "Algorithmic gains persist to complete-OP extrapolation": "Post-selection gains on complete operations",
    "Complete-OP RMSE (dB-Hz)": "Post-selection RMSE (dB-Hz)",
    "Physical structure used by HGB regression": "HGB feature sensitivity",
    "physics-only gap": "baseline discrepancy",
}


def normalize_visible_text(fig: plt.Figure) -> None:
    """Apply terminology-only edits to visible Matplotlib text artists."""

    for artist in fig.findobj(match=mpl.text.Text):
        original = artist.get_text()
        replacement = TEXT_REPLACEMENTS.get(original)
        if replacement is not None:
            artist.set_text(replacement)
        elif original.startswith("Complete-OP holdouts; mean"):
            artist.set_text(
                "Complete operations; grouped mean +/- s.d. after selection."
            )
        elif "Complete-OP holdout" in original:
            artist.set_text(original.replace("Complete-OP holdout", "Complete OP post-selection"))
        if artist.get_fontsize() < MIN_FONT_PT:
            artist.set_fontsize(MIN_FONT_PT)


def postprocess_ai1(fig: plt.Figure) -> None:
    """Recompose the narrow responsibility panel at final submission width."""

    if len(fig.axes) < 4:
        raise RuntimeError("AI1 does not contain the expected four axes")
    architecture_ax = fig.axes[0]
    for artist in architecture_ax.texts:
        if artist.get_text().startswith("Robust median-offset"):
            artist.set_text("Robust median-offset\nregression (L1 loss)")
            artist.set_position((0.314, 0.705))
            artist.set_va("center")
            artist.set_fontsize(5.5)
        elif artist.get_text() == "absolute-error calibration $\\beta_s$":
            artist.set_text(r"four band offsets $\beta_s$")
            artist.set_position((0.314, 0.595))
            artist.set_va("center")
            artist.set_fontsize(5.5)

    matrix_ax = fig.axes[-1]
    matrix_ax.clear()
    matrix_ax.axis("off")
    matrix_ax.set_title("Implemented responsibility", loc="left", pad=3, fontsize=7.3)

    mechanisms = [
        "Range / FSPL / occultation",
        "Nominal antenna / atmosphere",
        "EIRP / noise / hardware scale",
        "Pattern / attitude / limb mismatch",
        "Receiver / environment drift",
        "Fast fading / lock events",
    ]
    layers = [
        "Physics-based\nbaseline",
        "Physics-based\nbaseline",
        "Median-offset\nregression\n(L1 loss)",
        "HGB regression",
        "HGB regression",
        "Outside trend\nmodel",
    ]
    colors = [
        source.COL["physics"],
        source.COL["physics"],
        source.COL["beta"],
        source.COL["ai"],
        source.COL["ai"],
        source.COL["muted"],
    ]
    matrix_ax.text(0.00, 0.84, "Mechanism or uncertainty", transform=matrix_ax.transAxes, fontsize=5.5, color=source.COL["muted"], fontweight="bold")
    matrix_ax.text(0.72, 0.84, "Assigned layer", transform=matrix_ax.transAxes, fontsize=5.5, color=source.COL["muted"], fontweight="bold")
    matrix_ax.plot([0.00, 0.995], [0.80, 0.80], transform=matrix_ax.transAxes, color=source.COL["ink"], lw=0.65)
    row_height = 0.108
    for index, (mechanism, layer, color) in enumerate(zip(mechanisms, layers, colors)):
        ypos = 0.745 - index * row_height
        if index % 2:
            matrix_ax.add_patch(
                Rectangle((0.0, ypos - 0.048), 0.995, 0.096, transform=matrix_ax.transAxes, facecolor="#F7F8F9", edgecolor="none")
            )
        matrix_ax.text(0.00, ypos, mechanism, transform=matrix_ax.transAxes, ha="left", va="center", fontsize=5.5, color=source.COL["ink"])
        matrix_ax.scatter([0.705], [ypos], transform=matrix_ax.transAxes, s=23, marker="s", color=color, edgecolor="white", linewidth=0.4, zorder=3)
        matrix_ax.text(0.745, ypos, layer, transform=matrix_ax.transAxes, ha="left", va="center", fontsize=5.5, linespacing=0.86, color=source.COL["ink"])
        matrix_ax.plot([0.00, 0.995], [ypos - 0.054, ypos - 0.054], transform=matrix_ax.transAxes, color=source.COL["grid"], lw=0.38)
    matrix_ax.text(0.00, 0.055, "Responsibility assignment; sensitivity is non-causal.", transform=matrix_ax.transAxes, ha="left", va="bottom", fontsize=5.5, color=source.COL["muted"], style="italic")
    matrix_ax.text(-0.18, 1.08, "c", transform=matrix_ax.transAxes, fontweight="bold", fontsize=8.0, va="top", ha="left")


def postprocess_ai2(fig: plt.Figure) -> None:
    """Keep the right-hand sensitivity panel inside its final-width column."""

    if len(fig.axes) < 5:
        raise RuntimeError("AI2 does not contain the expected five axes")
    compact_stage_labels = [
        "Physics-based\nbaseline",
        "Robust median-offset\nregression\n(L1 loss)",
        "HGB\nregression",
    ]
    full_stage_labels = [
        "Physics-based\nbaseline",
        "+ robust median-offset\nregression (L1 loss)",
        "+ HGB\nregression",
    ]
    fig.axes[0].set_xticks([0, 1, 2], compact_stage_labels)
    fig.axes[0].tick_params(axis="x", labelsize=5.5, pad=2.0)
    fig.axes[3].set_xticks([0, 1, 2], full_stage_labels)
    fig.axes[0].set_ylabel("Post-selection RMSE\n(dB-Hz)")

    recovery_ax = fig.axes[1]
    recovery_ax.set_yticks(
        [1, 0],
        ["Recovered MSE", "Remaining MSE\nremoved by HGB regression"],
    )

    coefficient_ax = fig.axes[2]
    coefficient_ax.set_title("Band-specific median offsets", loc="left", pad=3.5)
    coefficient_ax.set_xlabel(r"L1-loss offset $\beta_s$ (dB)")

    sensitivity_ax = fig.axes[4]
    sensitivity_ax.set_title("HGB feature sensitivity", loc="left", pad=3.5)
    for artist in sensitivity_ax.texts:
        if artist.get_text().startswith("Complete operations;"):
            artist.set_text("Complete operations; grouped mean +/- s.d.\nafter selection.")
            artist.set_position((0.98, 0.03))
            artist.set_ha("right")
            artist.set_va("bottom")

def postprocess_ai3(fig: plt.Figure) -> None:
    """Enforce the locked validation-only selection semantics in AI3."""

    if AI3_VALIDATION_SELECTION_X is None or len(fig.axes) < 5:
        raise RuntimeError("AI3 validation selection coordinate was not initialized")

    selection_ax = fig.axes[0]
    neutral = source.COL["physics"]
    for collection in selection_ax.collections:
        if collection.get_label() == "Complete-OP":
            collection.set_edgecolor(neutral)
            collection.set_linewidth(0.9)

    for artist in selection_ax.texts:
        if artist.get_text() == "selected" and isinstance(artist, mpl.text.Annotation):
            artist.xy = (AI3_VALIDATION_SELECTION_X, 0)
            artist.set_position((5, -12))
            artist.set_color(source.COL["beta"])

    legend = selection_ax.get_legend()
    if legend is not None:
        handles = getattr(legend, "legend_handles", [])
        for label, handle in zip(legend.get_texts(), handles):
            if label.get_text().startswith("Complete-OP"):
                if hasattr(handle, "set_edgecolor"):
                    handle.set_edgecolor(neutral)
                if hasattr(handle, "set_facecolor"):
                    handle.set_facecolor("white")

    evaluation_ax = fig.axes[4]
    evaluation_ax.set_xticks(
        [0, 1, 2, 3],
        [
            "Train",
            "Validation\n(selection)",
            "Internal test*",
            "4 OPs*",
        ],
    )
    evaluation_ax.text(
        0.98,
        0.035,
        "* post-selection evaluation",
        transform=evaluation_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=source.COL["muted"],
    )
    for collection in evaluation_ax.collections:
        offsets = getattr(collection, "get_offsets", lambda: [])()
        if len(offsets) == 4:
            facecolors = collection.get_facecolors()
            if len(facecolors) == 4:
                facecolors[-1] = mpl.colors.to_rgba(neutral)
                collection.set_facecolors(facecolors)

    for axis_index in (1, 2):
        axis = fig.axes[axis_index]
        labels = [label.get_text() for label in axis.get_yticklabels()]
        if labels and labels[0] == "HGB":
            labels[0] = "HGB regression"
            axis.set_yticks(axis.get_yticks(), labels)

    selection_ax.set_yticks(
        selection_ax.get_yticks(),
        ["Median offset\n(L1 loss)", "Huber", "L2 mean", "Global\nmedian"],
    )
    selection_ax.tick_params(axis="y", pad=1.0)
    position = selection_ax.get_position()
    selection_ax.set_position(
        [position.x0 + 0.018, position.y0, position.width - 0.018, position.height]
    )


def save_revision(fig: plt.Figure, name: str) -> None:
    """Write a versioned four-format bundle without touching accepted files."""

    fig.canvas.draw()
    normalize_visible_text(fig)
    if name == "FigAI1_physics_guided_residual_learning_framework":
        postprocess_ai1(fig)
    elif name == "FigAI2_residual_contribution_and_interpretation":
        postprocess_ai2(fig)
    elif name == "FigAI3_algorithm_and_hyperparameter_selection":
        postprocess_ai3(fig)
    fig.set_size_inches(FINAL_WIDTH_IN, fig.get_figheight(), forward=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{name}{REVISION_SUFFIX}"
    for suffix, kwargs in [
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600}),
    ]:
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            bbox_inches=None,
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    global AI3_VALIDATION_SELECTION_X

    source.set_style()
    source.save_figure = save_revision

    pred = source.load_predictions()
    card = json.loads(source.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    analysis = source.ANALYSIS_DIR

    contribution = pd.read_csv(analysis / "physics_beta_hgb_contribution_metrics.csv")
    shares = pd.read_csv(analysis / "three_layer_sequential_contribution_shares.csv")
    importance = pd.read_csv(analysis / "external_holdout_grouped_permutation.csv")
    beta_values = pd.read_csv(analysis / "learned_signal_fixed_residual_beta.csv")
    fixed_benchmark = pd.read_csv(analysis / "fixed_residual_learner_benchmark.csv")
    benchmark = pd.read_csv(analysis / "final_split_algorithm_benchmark_with_block_bootstrap.csv")
    candidates = pd.read_csv(source.CANDIDATE_PATH)
    AI3_VALIDATION_SELECTION_X = float(
        fixed_benchmark.loc[
            fixed_benchmark["evaluation_split"].eq("validation")
            & fixed_benchmark["learner"].eq("Signal L1 fixed effect"),
            "rmse_dbhz",
        ].iloc[0]
    )

    source.figure_ai1_three_layer(pred, card)
    source.figure_ai2_three_layer(contribution, shares, importance, beta_values)
    source.figure_ai3_two_learners(fixed_benchmark, benchmark, candidates, contribution)

    print(f"Submission-width terminology figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
