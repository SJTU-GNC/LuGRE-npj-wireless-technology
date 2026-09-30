#!/usr/bin/env python3
"""Render the compact, validation-only Fig. AI3 revision without retraining."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure" / "paper_draft_v2" / "ai_residual_story"
STEM = "FigAI3_algorithm_and_hyperparameter_selection_compact_v2"

FIXED_BENCHMARK_PATH = (
    ROOT / "table" / "algorithm" / "cn0_ai_residual_story" / "fixed_residual_learner_benchmark.csv"
)
ALGORITHM_BENCHMARK_PATH = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_ai_residual_story"
    / "final_split_algorithm_benchmark_with_block_bootstrap.csv"
)
CANDIDATE_PATH = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_trend_residual_tuning_no_leakage"
    / "validation_only_candidate_scores.csv"
)
SELECTION_PATH = (
    ROOT / "table" / "algorithm" / "cn0_trend_residual_tuning_no_leakage" / "selected_model.json"
)
MANIFEST_PATH = ROOT / "table" / "paper_integration" / "algorithm_manifest" / "final_algorithm_manifest.md"
ORIGINAL_FIGURE_PATH = FIG_DIR / "FigAI3_algorithm_and_hyperparameter_selection.png"

PANEL_E_SOURCE_PATH = FIG_DIR / "FigAI3_panel_e_hyperparameter_robustness_source.csv"
PANEL_E_SUMMARY_PATH = FIG_DIR / "FigAI3_panel_e_hyperparameter_robustness_summary.csv"
CAPTION_QA_PATH = FIG_DIR / f"{STEM}_caption_QA.md"
PROVENANCE_PATH = FIG_DIR / f"{STEM}_provenance.json"

FIG_WIDTH_MM = 183.0
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
FIG_HEIGHT_IN = 4.62
EXPORT_DPI = 600

COL = {
    "ink": "#25282D",
    "muted": "#68737D",
    "grid": "#DCE2E7",
    "physics": "#566572",
    "physics_light": "#C9D1D7",
    "beta": "#C9912B",
    "beta_light": "#F1DFB7",
    "ai": "#B83A58",
    "ai_light": "#EBC8D1",
    "teal": "#267F78",
    "blue": "#4575A8",
}

PARAMETER_ORDER = [f"p{i}" for i in range(6)]
FEATURE_ORDER = ["stable_compact", "current", "no_grap_uncertainty", "direct_budget_only"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.titlesize": 7.3,
            "axes.titleweight": "normal",
            "axes.labelsize": 6.7,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.45,
            "axes.linewidth": 0.70,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float, y: float = 1.075) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=COL["ink"],
        clip_on=False,
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fixed = pd.read_csv(FIXED_BENCHMARK_PATH)
    benchmark = pd.read_csv(ALGORITHM_BENCHMARK_PATH)
    candidates = pd.read_csv(CANDIDATE_PATH)
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))

    required_candidate_columns = {
        "feature_set",
        "parameter_set",
        "phase_balanced_rmse_dbhz",
        "worst_phase_rmse_dbhz",
        "selected",
    }
    missing = required_candidate_columns.difference(candidates.columns)
    if missing:
        raise RuntimeError(f"Candidate table is missing columns: {sorted(missing)}")
    if len(candidates) != 24:
        raise RuntimeError(f"Expected 24 validation-only candidates, found {len(candidates)}")
    if set(candidates["feature_set"]) != set(FEATURE_ORDER):
        raise RuntimeError("The four feature-set identities changed")
    if set(candidates["parameter_set"]) != set(PARAMETER_ORDER):
        raise RuntimeError("The p0-p5 parameter-set identities changed")
    if not all(candidates.groupby("feature_set").size().eq(6)):
        raise RuntimeError("Each feature set must contain all six parameter sets")

    expected_parameters = {
        "max_iter": 400,
        "learning_rate": 0.04,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 32,
        "l2_regularization": 1.5,
    }
    if selection.get("selected_parameters") != expected_parameters:
        raise RuntimeError(
            "Selected HGB parameters differ from the signed model selection: "
            f"{selection.get('selected_parameters')}"
        )
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")
    if "robust median-offset regression (L1 loss)" not in manifest:
        raise RuntimeError("Authoritative manifest no longer confirms the robust L1 objective")
    if "four band-specific robust median offsets beta_s" not in manifest:
        raise RuntimeError("Authoritative manifest no longer confirms four beta_s offsets")
    return fixed, benchmark, candidates, selection


def build_panel_e_data(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = candidates[
        [
            "feature_set",
            "parameter_set",
            "phase_balanced_rmse_dbhz",
            "worst_phase_rmse_dbhz",
            "selected",
        ]
    ].copy()
    source["feature_set_best_phase_balanced_rmse_dbhz"] = source.groupby("feature_set")[
        "phase_balanced_rmse_dbhz"
    ].transform("min")
    source["delta_rmse_dbhz"] = (
        source["phase_balanced_rmse_dbhz"]
        - source["feature_set_best_phase_balanced_rmse_dbhz"]
    )
    source["is_feature_set_best"] = np.isclose(
        source["delta_rmse_dbhz"].to_numpy(float),
        0.0,
        atol=1e-12,
        rtol=0.0,
    )
    source["feature_order"] = source["feature_set"].map(
        {name: idx for idx, name in enumerate(FEATURE_ORDER)}
    )
    source["parameter_order"] = source["parameter_set"].map(
        {name: idx for idx, name in enumerate(PARAMETER_ORDER)}
    )
    source = source.sort_values(["parameter_order", "feature_order"]).drop(
        columns=["feature_order", "parameter_order"]
    )

    summary = (
        source.groupby("parameter_set", sort=False)["delta_rmse_dbhz"]
        .agg(
            feature_set_count="size",
            mean_delta_rmse_dbhz="mean",
            median_delta_rmse_dbhz="median",
            min_delta_rmse_dbhz="min",
            max_delta_rmse_dbhz="max",
        )
        .reindex(PARAMETER_ORDER)
        .reset_index()
    )
    best_counts = (
        source.groupby("parameter_set", sort=False)["is_feature_set_best"]
        .sum()
        .reindex(PARAMETER_ORDER)
        .astype(int)
    )
    summary["feature_sets_at_optimum"] = summary["parameter_set"].map(best_counts)

    p5 = summary.set_index("parameter_set").loc["p5"]
    p5_values = p5[
        [
            "mean_delta_rmse_dbhz",
            "min_delta_rmse_dbhz",
            "max_delta_rmse_dbhz",
        ]
    ].to_numpy(float)
    if not np.isfinite(p5_values).all():
        raise RuntimeError("The p5 robustness summary contains non-finite values")
    return source, summary


def draw_panel_a(ax: plt.Axes, fixed: pd.DataFrame) -> None:
    learner_order = [
        "Signal L1 fixed effect",
        "Signal Huber location",
        "Signal L2 mean",
        "Global L1 offset",
    ]
    validation = (
        fixed[fixed["evaluation_split"].eq("validation")]
        .set_index("learner")
        .reindex(learner_order)
    )
    complete_ops = (
        fixed[fixed["evaluation_split"].eq("external_holdout")]
        .set_index("learner")
        .reindex(learner_order)
    )
    if validation["rmse_dbhz"].isna().any() or complete_ops["rmse_dbhz"].isna().any():
        raise RuntimeError("Panel a source rows changed")

    y = np.arange(len(learner_order))
    for yi, learner in enumerate(learner_order):
        ax.plot(
            [validation.loc[learner, "rmse_dbhz"], complete_ops.loc[learner, "rmse_dbhz"]],
            [yi, yi],
            color=COL["grid"],
            lw=1.0,
            zorder=1,
        )
    validation_colors = [COL["beta"]] + [COL["physics_light"]] * 3
    validation_edges = [COL["beta"]] + [COL["physics"]] * 3
    ax.scatter(
        validation["rmse_dbhz"],
        y,
        s=32,
        color=validation_colors,
        edgecolor=validation_edges,
        linewidth=0.75,
        label="Validation",
        zorder=3,
    )
    ax.scatter(
        complete_ops["rmse_dbhz"],
        y,
        s=29,
        marker="D",
        facecolor="white",
        edgecolor=validation_edges,
        linewidth=0.95,
        label="Complete OPs",
        zorder=3,
    )
    ax.set_yticks(y, ["Robust L1", "Huber", "L2 mean", "Global L1"])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE after robust-L1 correction (dB-Hz)")
    ax.set_title("Robust regression algorithm", loc="left", pad=4.0)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.94),
        frameon=False,
        fontsize=5.25,
        handletextpad=0.28,
        labelspacing=0.24,
        borderaxespad=0.1,
    )
    ax.annotate(
        "selected",
        (complete_ops.loc["Signal L1 fixed effect", "rmse_dbhz"], 0),
        xytext=(5, -11),
        textcoords="offset points",
        color=COL["ink"],
        fontsize=5.35,
    )
    ax.margins(x=0.05, y=0.16)
    panel_label(ax, "a", x=-0.25)


def draw_panel_b(ax: plt.Axes, benchmark: pd.DataFrame) -> pd.DataFrame:
    ordered = benchmark.sort_values("validation_rmse_dbhz", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    low = ordered["validation_rmse_dbhz"] - ordered["validation_rmse_ci95_low_dbhz"]
    high = ordered["validation_rmse_ci95_high_dbhz"] - ordered["validation_rmse_dbhz"]
    ax.errorbar(
        ordered["validation_rmse_dbhz"],
        y,
        xerr=np.vstack([low, high]),
        fmt="none",
        ecolor=COL["physics_light"],
        elinewidth=1.55,
        capsize=1.9,
        zorder=1,
    )
    colors = [COL["ai"] if name == "HGB" else COL["physics_light"] for name in ordered["algorithm"]]
    edges = [COL["ai"] if name == "HGB" else COL["physics"] for name in ordered["algorithm"]]
    ax.scatter(
        ordered["validation_rmse_dbhz"],
        y,
        s=31,
        color=colors,
        edgecolor=edges,
        linewidth=0.85,
        zorder=3,
    )
    ax.scatter(
        ordered["worst_phase_rmse_dbhz"],
        y,
        s=21,
        marker="|",
        color=edges,
        linewidth=1.35,
        zorder=3,
    )
    ax.set_yticks(y, ordered["algorithm"])
    ax.invert_yaxis()
    ax.set_xlabel("Validation trend RMSE (dB-Hz)")
    ax.set_title("HGB selection among trend regressors", loc="left", pad=4.0)
    uncertainty_handles = [
        Line2D(
            [0],
            [0],
            color=COL["physics_light"],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=COL["physics"],
            lw=1.3,
            ms=3.5,
            label=r"Mean $\pm$ 95% CI",
        ),
        Line2D(
            [0],
            [0],
            color=COL["physics"],
            marker="|",
            linestyle="none",
            ms=5.3,
            markeredgewidth=1.1,
            label="Worst phase",
        ),
    ]
    ax.legend(
        handles=uncertainty_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),
        frameon=False,
        fontsize=4.9,
        handlelength=1.4,
        labelspacing=0.20,
        borderaxespad=0.0,
    )
    ax.margins(x=0.06, y=0.08)
    panel_label(ax, "b", x=-0.23)
    return ordered


def draw_panel_c(ax: plt.Axes, ordered: pd.DataFrame) -> None:
    phases = ["C", "T", "L", "S"]
    heat = ordered.set_index("algorithm")[
        [f"rmse_phase_{phase}_dbhz" for phase in phases]
    ].to_numpy(float)
    heat_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rmse_seq_compact_v2",
        ["#F7F6F3", "#E8D6B2", COL["beta"], "#C66D3D", COL["ai"], "#722A3D"],
    )
    upper = float(np.nanpercentile(heat, 95))
    ax.imshow(heat, aspect="auto", cmap=heat_cmap, vmin=float(np.nanmin(heat)), vmax=upper)
    ax.set_xticks(np.arange(4), phases)
    ax.set_yticks(np.arange(len(ordered)), ordered["algorithm"])
    threshold = float(np.nanpercentile(heat, 70))
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            ax.text(
                col,
                row,
                f"{heat[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=5.35,
                color="white" if heat[row, col] > threshold else COL["ink"],
            )
    ax.set_title("Phase-resolved validation RMSE", loc="left", pad=4.0)
    ax.tick_params(length=0)
    panel_label(ax, "c", x=-0.30)


def draw_panel_d(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    ax.scatter(
        candidates["phase_balanced_rmse_dbhz"],
        candidates["worst_phase_rmse_dbhz"],
        s=22,
        color=COL["physics_light"],
        edgecolor="white",
        linewidth=0.40,
        alpha=0.94,
        zorder=2,
    )
    selected = candidates[candidates["selected"].astype(str).str.lower().eq("true")]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one selected validation candidate, found {len(selected)}")
    selected_x = float(selected["phase_balanced_rmse_dbhz"].iloc[0])
    selected_y = float(selected["worst_phase_rmse_dbhz"].iloc[0])
    ax.scatter(
        [selected_x],
        [selected_y],
        s=55,
        facecolor=COL["ai"],
        edgecolor="white",
        linewidth=0.70,
        zorder=4,
    )
    ax.axvline(selected_x, color=COL["ai_light"], lw=0.60, linestyle=(0, (2, 2)), zorder=0)
    ax.axhline(selected_y, color=COL["ai_light"], lw=0.60, linestyle=(0, (2, 2)), zorder=0)
    ax.annotate(
        "selected",
        (selected_x, selected_y),
        xytext=(6, -11),
        textcoords="offset points",
        color=COL["ink"],
        fontsize=5.35,
    )
    ax.set_xlabel("Phase-balanced RMSE (dB-Hz)")
    ax.set_ylabel("Worst-phase RMSE (dB-Hz)")
    ax.set_title("Validation-only HGB search (n = 24)", loc="left", pad=4.0)
    x = candidates["phase_balanced_rmse_dbhz"].to_numpy(float)
    y = candidates["worst_phase_rmse_dbhz"].to_numpy(float)
    ax.set_xlim(float(np.nanmin(x)) - 0.006, float(np.nanmax(x)) + 0.006)
    ax.set_ylim(float(np.nanmin(y)) - 0.022, float(np.nanmax(y)) + 0.025)
    panel_label(ax, "d", x=-0.25)


def draw_panel_e(ax: plt.Axes, source: pd.DataFrame, summary: pd.DataFrame) -> None:
    x_positions = np.arange(len(PARAMETER_ORDER), dtype=float)
    jitter = {
        feature: offset
        for feature, offset in zip(FEATURE_ORDER, [-0.12, -0.04, 0.04, 0.12])
    }
    for idx, parameter_set in enumerate(PARAMETER_ORDER):
        rows = source[source["parameter_set"].eq(parameter_set)].set_index("feature_set").reindex(
            FEATURE_ORDER
        )
        deltas = rows["delta_rmse_dbhz"].to_numpy(float)
        is_selected = parameter_set == "p5"
        range_color = COL["ai"] if is_selected else COL["physics"]
        ax.vlines(
            idx,
            float(np.nanmin(deltas)),
            float(np.nanmax(deltas)),
            color=range_color,
            lw=1.15 if is_selected else 0.85,
            alpha=0.88,
            zorder=1,
        )
        ax.scatter(
            [idx + jitter[name] for name in FEATURE_ORDER],
            deltas,
            s=17,
            facecolor=COL["blue"],
            edgecolor="white",
            linewidth=0.38,
            alpha=0.88,
            zorder=3,
        )
        mean_value = float(
            summary.loc[
                summary["parameter_set"].eq(parameter_set),
                "mean_delta_rmse_dbhz",
            ].iloc[0]
        )
        ax.scatter(
            [idx],
            [mean_value],
            s=29 if is_selected else 24,
            marker="D",
            facecolor=COL["ai"] if is_selected else COL["ink"],
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )

    ax.axhline(0.0, color=COL["muted"], lw=0.60, linestyle=(0, (2, 2)), zorder=0)
    ax.set_xticks(x_positions, PARAMETER_ORDER)
    ax.get_xticklabels()[-1].set_color(COL["ai"])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylabel(r"Within-set $\Delta$RMSE (dB-Hz)")
    ax.set_xlabel("Parameter set")
    ax.set_title("Hyperparameter robustness\nacross feature sets", loc="left", pad=3.0)
    ymax = float(source["delta_rmse_dbhz"].max())
    ax.set_ylim(-0.005, ymax + 0.012)
    ax.set_xlim(-0.38, len(PARAMETER_ORDER) - 0.62)
    summary_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COL["blue"],
            markeredgecolor="white",
            markersize=3.8,
            label="Feature-set score",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=COL["physics"],
            markerfacecolor=COL["ink"],
            markeredgecolor="white",
            linewidth=0.9,
            markersize=3.6,
            label="Mean [min, max]",
        ),
    ]
    ax.legend(
        handles=summary_handles,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.99),
        frameon=False,
        fontsize=4.7,
        handlelength=1.25,
        handletextpad=0.35,
        labelspacing=0.22,
        borderaxespad=0.0,
    )
    panel_label(ax, "e", x=-0.23)


def draw_panel_f(ax: plt.Axes, selection: dict[str, Any]) -> None:
    parameters = selection["selected_parameters"]
    ax.axis("off")
    ax.set_title("Selected model specification", loc="left", pad=4.0)

    ax.plot([0.00, 1.00], [0.88, 0.88], transform=ax.transAxes, color=COL["ink"], lw=0.75)
    ax.plot([0.00, 1.00], [0.11, 0.11], transform=ax.transAxes, color=COL["ink"], lw=0.75)
    ax.plot([0.49, 0.49], [0.16, 0.82], transform=ax.transAxes, color=COL["grid"], lw=0.65)

    ax.scatter(
        [0.025],
        [0.79],
        transform=ax.transAxes,
        s=19,
        marker="s",
        color=COL["beta"],
        edgecolor="white",
        linewidth=0.35,
    )
    ax.text(
        0.065,
        0.79,
        "Robust L1",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.15,
        fontweight="bold",
        color=COL["ink"],
    )
    ax.text(
        0.025,
        0.62,
        "Objective",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.25,
        color=COL["muted"],
    )
    ax.text(
        0.025,
        0.53,
        "absolute-error (L1)",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.55,
        color=COL["ink"],
    )
    ax.text(
        0.025,
        0.38,
        "Offsets",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.25,
        color=COL["muted"],
    )
    ax.text(
        0.025,
        0.29,
        r"4 constellation-band $\beta_s$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.45,
        color=COL["ink"],
    )

    ax.scatter(
        [0.535],
        [0.79],
        transform=ax.transAxes,
        s=19,
        marker="s",
        color=COL["ai"],
        edgecolor="white",
        linewidth=0.35,
    )
    ax.text(
        0.575,
        0.79,
        "HGB",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.15,
        fontweight="bold",
        color=COL["ink"],
    )
    hgb_rows = [
        (
            "Estimator / iterations",
            f"HistGradientBoosting / {parameters['max_iter']}",
            0.62,
            0.53,
        ),
        (
            "Learning rate / leaves",
            f"{parameters['learning_rate']} / {parameters['max_leaf_nodes']}",
            0.43,
            0.34,
        ),
        (
            "Minimum leaf / L2 reg.",
            f"{parameters['min_samples_leaf']} / {parameters['l2_regularization']}",
            0.24,
            0.15,
        ),
    ]
    for key, value, key_y, value_y in hgb_rows:
        ax.text(
            0.535,
            key_y,
            key,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.05,
            color=COL["muted"],
        )
        ax.text(
            0.535,
            value_y,
            value,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.30,
            color=COL["ink"],
        )
    panel_label(ax, "f", x=-0.12)


def render_figure(
    fixed: pd.DataFrame,
    benchmark: pd.DataFrame,
    candidates: pd.DataFrame,
    panel_e_source: pd.DataFrame,
    panel_e_summary: pd.DataFrame,
    selection: dict[str, Any],
) -> tuple[plt.Figure, dict[str, Any]]:
    set_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=EXPORT_DPI)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.00, 1.11, 1.09],
        height_ratios=[1.00, 1.00],
        left=0.079,
        right=0.991,
        top=0.953,
        bottom=0.118,
        wspace=0.285,
        hspace=0.345,
    )

    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    draw_panel_a(axes[0], fixed)
    ordered = draw_panel_b(axes[1], benchmark)
    draw_panel_c(axes[2], ordered)
    draw_panel_d(axes[3], candidates)
    draw_panel_e(axes[4], panel_e_source, panel_e_summary)
    draw_panel_f(axes[5], selection)

    for ax in axes:
        ax.tick_params(direction="out")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    text_bboxes = []
    out_of_canvas = []
    for text_artist in fig.findobj(match=mpl.text.Text):
        if not text_artist.get_visible() or not text_artist.get_text().strip():
            continue
        bbox = text_artist.get_window_extent(renderer=renderer)
        text_bboxes.append((text_artist.get_text(), bbox))
        if (
            bbox.x0 < figure_bbox.x0 - 0.5
            or bbox.y0 < figure_bbox.y0 - 0.5
            or bbox.x1 > figure_bbox.x1 + 0.5
            or bbox.y1 > figure_bbox.y1 + 0.5
        ):
            out_of_canvas.append(
                {
                    "text": text_artist.get_text(),
                    "bbox_px": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                }
            )

    panel_label_title_overlaps = []
    for ax, label in zip(axes, "abcdef"):
        title_bbox = ax.title.get_window_extent(renderer=renderer)
        label_artist = next(
            artist
            for artist in ax.texts
            if artist.get_text() == label and artist.get_fontweight() == "bold"
        )
        label_bbox = label_artist.get_window_extent(renderer=renderer)
        if title_bbox.overlaps(label_bbox):
            panel_label_title_overlaps.append(label)

    row_gaps_px = []
    for top_ax, bottom_ax in zip(axes[:3], axes[3:]):
        lower_candidates = [
            label.get_window_extent(renderer=renderer)
            for label in top_ax.get_xticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        if top_ax.xaxis.label.get_text().strip():
            lower_candidates.append(top_ax.xaxis.label.get_window_extent(renderer=renderer))
        if lower_candidates:
            upper_row_lower_y = min(bbox.y0 for bbox in lower_candidates)
            lower_title_upper_y = bottom_ax.title.get_window_extent(renderer=renderer).y1
            row_gaps_px.append(float(upper_row_lower_y - lower_title_upper_y))

    panel_f_texts = [
        artist
        for artist in axes[5].texts
        if artist.get_visible()
        and artist.get_text().strip()
        and artist.get_text() != "f"
    ]
    panel_f_overlaps = []
    for index, first in enumerate(panel_f_texts):
        first_bbox = first.get_window_extent(renderer=renderer)
        for second in panel_f_texts[index + 1 :]:
            second_bbox = second.get_window_extent(renderer=renderer)
            if first_bbox.overlaps(second_bbox):
                panel_f_overlaps.append([first.get_text(), second.get_text()])

    if out_of_canvas:
        raise RuntimeError(f"Text outside canvas: {out_of_canvas}")
    if panel_label_title_overlaps:
        raise RuntimeError(f"Panel labels overlap titles: {panel_label_title_overlaps}")
    if row_gaps_px and min(row_gaps_px) < 0:
        raise RuntimeError(f"Top- and bottom-row text overlap: {row_gaps_px}")
    if panel_f_overlaps:
        raise RuntimeError(f"Panel f text overlaps: {panel_f_overlaps}")

    qa = {
        "figure_size_inches": [FIG_WIDTH_IN, FIG_HEIGHT_IN],
        "target_width_mm": FIG_WIDTH_MM,
        "dpi": EXPORT_DPI,
        "gridspec": {
            "left": 0.079,
            "right": 0.991,
            "top": 0.953,
            "bottom": 0.118,
            "wspace": 0.285,
            "hspace": 0.345,
        },
        "axes_count": len(axes),
        "panel_labels": list("abcdef"),
        "text_outside_canvas_count": len(out_of_canvas),
        "panel_label_title_overlap_count": len(panel_label_title_overlaps),
        "minimum_interrow_text_gap_px": min(row_gaps_px) if row_gaps_px else None,
        "panel_f_text_overlap_count": len(panel_f_overlaps),
    }
    return fig, qa


def save_outputs(fig: plt.Figure) -> dict[str, dict[str, Any]]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "png": FIG_DIR / f"{STEM}.png",
        "pdf": FIG_DIR / f"{STEM}.pdf",
        "svg": FIG_DIR / f"{STEM}.svg",
        "tiff": FIG_DIR / f"{STEM}.tiff",
    }
    fig.savefig(output_paths["png"], dpi=EXPORT_DPI, facecolor="white", edgecolor="white")
    fig.savefig(output_paths["pdf"], facecolor="white", edgecolor="white")
    fig.savefig(output_paths["svg"], facecolor="white", edgecolor="white")
    fig.savefig(
        output_paths["tiff"],
        dpi=EXPORT_DPI,
        facecolor="white",
        edgecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    output_audit: dict[str, dict[str, Any]] = {}
    for suffix, path in output_paths.items():
        output_audit[suffix] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    with Image.open(output_paths["png"]) as image:
        rgb = np.asarray(image.convert("RGB"))
        ink = np.any(rgb < 248, axis=2)
        rows, cols = np.where(ink)
        if len(rows) == 0:
            raise RuntimeError("PNG contains no visible content")
        width, height = image.size
        margins = {
            "left_px": int(cols.min()),
            "right_px": int(width - 1 - cols.max()),
            "top_px": int(rows.min()),
            "bottom_px": int(height - 1 - rows.max()),
        }
        output_audit["png"].update(
            {
                "pixel_dimensions": [width, height],
                "dpi_metadata": list(image.info.get("dpi", ())),
                "visible_ink_margins_px_at_rgb_lt_248": margins,
            }
        )
        if width not in {4322, 4323}:
            raise RuntimeError(f"Unexpected 183-mm PNG width: {width}px")
    svg_text = output_paths["svg"].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG text was not retained as editable text")
    output_audit["svg"]["editable_text_elements_present"] = True
    return output_audit


def write_caption_and_qa(
    panel_e_summary: pd.DataFrame,
    qa: dict[str, Any],
    output_audit: dict[str, dict[str, Any]],
) -> None:
    p5 = panel_e_summary.set_index("parameter_set").loc["p5"]
    caption = (
        "**Caption draft.** Algorithm and hyperparameter selection for the two learned "
        "residual corrections. **a,** Validation and complete-operation comparison of "
        "persistent-correction estimators. **b,c,** Validation-only comparison of trend "
        "regressors and their phase-resolved RMSE. **d,** Validation-only search across "
        "24 feature-set and parameter-set candidates; the highlighted point is the selected "
        "HGB candidate. **e,** Hyperparameter robustness across four feature sets. For each "
        "feature set, $\\Delta$RMSE is the phase-balanced validation RMSE minus the minimum "
        "within that feature set; small points show the four feature-set values, diamonds "
        "show their means and vertical lines span their minima and maxima. The selected p5 "
        f"setting was optimal for {int(p5['feature_sets_at_optimum'])}/4 feature sets, with "
        f"mean $\\Delta$RMSE {float(p5['mean_delta_rmse_dbhz']):.6f} dB-Hz and maximum "
        f"$\\Delta$RMSE {float(p5['max_delta_rmse_dbhz']):.6f} dB-Hz. **f,** Final robust-L1 "
        "and HGB settings. Selection used validation data only; test data and the four "
        "complete operations were not used for training or tuning.\n\n"
        "**QA note.** Panels a-d retain their original statistical definitions and read the "
        "same persisted source tables; no model was trained or refitted. Panel e is computed "
        "only from `validation_only_candidate_scores.csv`. The figure uses a 183-mm canvas, "
        "editable SVG/PDF text, and 600-dpi raster exports.\n\n"
        f"- Canvas: {qa['figure_size_inches'][0]:.6f} x {qa['figure_size_inches'][1]:.2f} in\n"
        f"- PNG: {output_audit['png']['pixel_dimensions'][0]} x "
        f"{output_audit['png']['pixel_dimensions'][1]} px\n"
        f"- Text outside canvas: {qa['text_outside_canvas_count']}\n"
        f"- Panel-label/title overlaps: {qa['panel_label_title_overlap_count']}\n"
        f"- Visible-ink margins: "
        f"{json.dumps(output_audit['png']['visible_ink_margins_px_at_rgb_lt_248'])}\n"
    )
    CAPTION_QA_PATH.write_text(caption, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fixed, benchmark, candidates, selection = load_inputs()
    panel_e_source, panel_e_summary = build_panel_e_data(candidates)
    panel_e_source.to_csv(PANEL_E_SOURCE_PATH, index=False, encoding="utf-8-sig")
    panel_e_summary.to_csv(PANEL_E_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    fig, qa = render_figure(
        fixed,
        benchmark,
        candidates,
        panel_e_source,
        panel_e_summary,
        selection,
    )
    output_audit = save_outputs(fig)
    plt.close(fig)
    write_caption_and_qa(panel_e_summary, qa, output_audit)

    p5 = panel_e_summary.set_index("parameter_set").loc["p5"]
    provenance = {
        "figure": STEM,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "no_retraining": True,
        "figure_contract": {
            "core_conclusion": (
                "Validation-only selection supports the robust-L1 and HGB choices, and p5 "
                "remains near-optimal across all four feature sets rather than winning only "
                "for one feature set."
            ),
            "archetype": "compact quantitative model-selection grid",
            "panels_a_to_d": "Definitions, persisted data and conclusions unchanged.",
            "panel_e_delta_definition": (
                "phase_balanced_rmse_dbhz minus the minimum phase_balanced_rmse_dbhz "
                "within the same feature_set"
            ),
            "selection_exclusions": selection.get("forbidden_during_selection"),
        },
        "sources": {
            str(FIXED_BENCHMARK_PATH): sha256(FIXED_BENCHMARK_PATH),
            str(ALGORITHM_BENCHMARK_PATH): sha256(ALGORITHM_BENCHMARK_PATH),
            str(CANDIDATE_PATH): sha256(CANDIDATE_PATH),
            str(SELECTION_PATH): sha256(SELECTION_PATH),
            str(MANIFEST_PATH): sha256(MANIFEST_PATH),
            str(ORIGINAL_FIGURE_PATH): sha256(ORIGINAL_FIGURE_PATH),
        },
        "panel_e": {
            "candidate_rows": int(len(panel_e_source)),
            "feature_sets": FEATURE_ORDER,
            "parameter_sets": PARAMETER_ORDER,
            "p5_mean_delta_rmse_dbhz": float(p5["mean_delta_rmse_dbhz"]),
            "p5_max_delta_rmse_dbhz": float(p5["max_delta_rmse_dbhz"]),
            "p5_feature_sets_at_optimum": int(p5["feature_sets_at_optimum"]),
            "source_csv": str(PANEL_E_SOURCE_PATH),
            "summary_csv": str(PANEL_E_SUMMARY_PATH),
        },
        "selected_model_parameters": selection["selected_parameters"],
        "qa": qa,
        "outputs": output_audit,
        "caption_qa_note": str(CAPTION_QA_PATH),
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(provenance["panel_e"], indent=2))
    print(json.dumps(output_audit["png"], indent=2))


if __name__ == "__main__":
    main()
