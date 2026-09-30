#!/usr/bin/env python3
"""Build the four-panel quantitative Fig. 3 from frozen result tables."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "table" / "algorithm" / "cn0_ai_residual_story"
MANIFEST_DIR = ROOT / "table" / "paper_integration" / "algorithm_manifest"
OUTPUT_DIR = ROOT / "figure" / "paper_draft_v2"
OUTPUT_BASE = OUTPUT_DIR / "figure3"
PANEL_C_SOURCE_PATH = OUTPUT_DIR / "figure3_panel_c_source_data.csv"
THREE_PANEL_BACKUP_DIR = (
    OUTPUT_DIR
    / "figure3_versions"
    / "figure3_three_panel_v1_20260724_151959"
)
FOUR_PANEL_BACKUP_DIR = (
    OUTPUT_DIR
    / "figure3_versions"
    / "figure3_four_panel_v1_20260724_155614"
)

CONTRIBUTION_PATH = TABLE_DIR / "physics_beta_hgb_contribution_metrics.csv"
SHARES_PATH = TABLE_DIR / "three_layer_sequential_contribution_shares.csv"
BETA_PATH = TABLE_DIR / "learned_signal_fixed_residual_beta.csv"
PERMUTATION_PATH = TABLE_DIR / "external_holdout_grouped_permutation.csv"
PREDICTION_PATH = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_trend_residual_tuned_no_leakage"
    / "cn0_trend_residual_tuned_no_leakage_predictions.csv"
)
MANIFEST_PATH = MANIFEST_DIR / "final_algorithm_manifest.json"
AUTHORITATIVE_PATH = MANIFEST_DIR / "authoritative_metrics.csv"
INPUT_CONTRACT_PATH = MANIFEST_DIR / "figure_input_contract.md"
STYLE_SCRIPT_PATH = ROOT / "script" / "build_ai_residual_story_figures.py"
STYLE_REFERENCE_PATHS = [
    OUTPUT_DIR / "figure1.jpg",
    OUTPUT_DIR / "figure2.jpg",
]
OLD_FIGAI2_BASE = (
    OUTPUT_DIR
    / "ai_residual_story"
    / "FigAI2_residual_contribution_and_interpretation"
)

WIDTH_MM = 183.0
HEIGHT_MM = 118.0
EXPORT_DPI = 600

COLOR = {
    "ink": "#20262B",
    "muted": "#657681",
    "rule": "#D9E0E4",
    "physics": "#2F6B8A",
    "physics_light": "#D7E4EA",
    "beta": "#D49300",
    "beta_light": "#F3E4BC",
    "hgb": "#C43D58",
    "hgb_light": "#F3D7DD",
    "neutral": "#7C878E",
    "neutral_light": "#D9DEE1",
}

STAGES = ["Physics", "Physics + beta", "Physics + beta + HGB"]
STAGE_DISPLAY = ["Physics", "+ robust L1", "+ HGB"]
STAGE_COLORS = [COLOR["physics"], COLOR["beta"], COLOR["hgb"]]
STAGE_MARKERS = ["o", "s", "D"]
SPLITS = ["train", "validation", "test", "external_holdout"]
SPLIT_DISPLAY = {
    "train": "Training fit",
    "validation": "Validation selection",
    "test": "Internal test",
    "external_holdout": "4 complete OPs",
}
BETA_ORDER = ["GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"]
BETA_DISPLAY = {
    "GPS_L1": "GPS L1",
    "GPS_L5": "GPS L5",
    "GAL_E1": "Galileo E1",
    "GAL_E5a": "Galileo E5a",
}
PERMUTATION_DISPLAY = {
    "Physical baseline\nand range": "Physics baseline\n& range",
    "Receive antenna\nand attitude": "Receive pattern\n& attitude",
    "Transmit antenna\ngeometry": "Transmit-pattern\ngeometry",
    "Signal and\nhardware": "Signal band /\nhardware scale",
    "Atmospheric\ntrend proxies": "Atmospheric-path\nproxies",
    "Limb and\noccultation": "Limb / occultation\ngeometry",
}
OP_ORDER = ["OP2", "OP21", "OP27", "OP74"]
OP_PHASE = {"OP2": "C", "OP21": "T", "OP27": "L", "OP74": "S"}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def protected_snapshot() -> list[dict[str, Any]]:
    records = []
    for suffix in [".png", ".pdf", ".svg", ".tiff"]:
        path = OLD_FIGAI2_BASE.with_suffix(suffix)
        if not path.exists():
            raise FileNotFoundError(path)
        records.append(file_record(path))
    return records


def assert_protected_unchanged(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> None:
    before_map = {item["path"]: item for item in before}
    after_map = {item["path"]: item for item in after}
    if before_map != after_map:
        raise RuntimeError("The protected legacy FigAI2 files changed")


def require_columns(
    frame: pd.DataFrame, columns: list[str], source: Path
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def load_inputs() -> dict[str, Any]:
    contribution = pd.read_csv(CONTRIBUTION_PATH)
    shares = pd.read_csv(SHARES_PATH)
    beta = pd.read_csv(BETA_PATH)
    permutation = pd.read_csv(PERMUTATION_PATH)
    predictions = pd.read_csv(PREDICTION_PATH, low_memory=False)
    authoritative = pd.read_csv(AUTHORITATIVE_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    require_columns(
        contribution,
        [
            "evaluation_split",
            "stage",
            "n",
            "rmse_dbhz",
            "mission_phase",
        ],
        CONTRIBUTION_PATH,
    )
    require_columns(
        shares,
        [
            "evaluation_split",
            "mission_phase",
            "fixed_share_total_recovered_pct",
            "timevarying_share_total_recovered_pct",
            "timevarying_reduction_of_remaining_mse_pct",
        ],
        SHARES_PATH,
    )
    require_columns(beta, ["signal_name", "beta_db"], BETA_PATH)
    require_columns(
        permutation,
        [
            "feature_group",
            "repeat",
            "base_rmse_dbhz",
            "delta_rmse_dbhz",
            "n",
        ],
        PERMUTATION_PATH,
    )
    require_columns(
        predictions,
        [
            "minute_utc",
            "evaluation_split",
            "op",
            "mission_phase",
            "signal_name",
            "svid",
            "trend_training_eligible",
            "cn0_observed_trend_dbhz",
            "cn0_physics_trend_dbhz",
            "signal_beta_train_db",
            "cn0_physics_ai_trend_dbhz",
        ],
        PREDICTION_PATH,
    )

    overall = contribution[contribution["mission_phase"].isna()].copy()
    if set(overall["evaluation_split"]) != set(SPLITS):
        raise ValueError("Contribution table does not contain all partitions")
    for split in SPLITS:
        part = overall[overall["evaluation_split"].eq(split)]
        if set(part["stage"]) != set(STAGES) or len(part) != len(STAGES):
            raise ValueError(f"Incomplete stage metrics for {split}")
        if part["n"].nunique() != 1:
            raise ValueError(f"Stage sample counts differ within {split}")

    external = (
        overall[overall["evaluation_split"].eq("external_holdout")]
        .set_index("stage")
        .reindex(STAGES)
    )
    share_row = shares[
        shares["evaluation_split"].eq("external_holdout")
        & shares["mission_phase"].isna()
    ]
    if len(share_row) != 1:
        raise ValueError("Expected one complete-OP sequential-share row")
    share_row = share_row.iloc[0]
    share_sum = float(
        share_row["fixed_share_total_recovered_pct"]
        + share_row["timevarying_share_total_recovered_pct"]
    )
    if not np.isclose(share_sum, 100.0, atol=1e-8):
        raise ValueError("Caption-only sequential shares do not sum to 100")

    beta_indexed = beta.set_index("signal_name")
    if set(beta_indexed.index) != set(BETA_ORDER) or len(beta_indexed) != 4:
        raise ValueError("Robust L1 table must contain four coefficients")
    manifest_beta = manifest["robust_regression"]["final_beta_db"]
    for signal in BETA_ORDER:
        if not np.isclose(
            float(beta_indexed.loc[signal, "beta_db"]),
            float(manifest_beta[signal]),
            atol=5e-7,
        ):
            raise ValueError(f"Beta mismatch for {signal}")

    if set(permutation["feature_group"]) != set(PERMUTATION_DISPLAY):
        raise ValueError("Grouped permutation feature groups changed")
    for group, part in permutation.groupby("feature_group"):
        if len(part) != 30 or part["repeat"].nunique() != 30:
            raise ValueError(f"{group}: expected 30 permutation repeats")
        if part["n"].nunique() != 1 or int(part["n"].iloc[0]) != 4112:
            raise ValueError(f"{group}: unexpected permutation sample count")

    eligible = predictions["trend_training_eligible"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().eq("true")
    panel_c = predictions[
        predictions["op"].astype(str).isin(OP_ORDER)
        & predictions["evaluation_split"].eq("external_holdout")
        & eligible
    ].copy()
    required_values = [
        "cn0_observed_trend_dbhz",
        "cn0_physics_trend_dbhz",
        "signal_beta_train_db",
        "cn0_physics_ai_trend_dbhz",
    ]
    if panel_c[required_values].isna().any().any():
        raise ValueError("Eligible representative-OP rows contain missing values")
    if set(panel_c["op"].astype(str)) != set(OP_ORDER):
        raise ValueError("Representative complete-OP rows are incomplete")
    panel_c["op"] = panel_c["op"].astype(str)
    for op in OP_ORDER:
        phases = set(
            panel_c.loc[panel_c["op"].eq(op), "mission_phase"].astype(str)
        )
        if phases != {OP_PHASE[op]}:
            raise ValueError(f"Unexpected phase mapping for {op}: {phases}")
    panel_c["residual_physics_db"] = (
        panel_c["cn0_observed_trend_dbhz"]
        - panel_c["cn0_physics_trend_dbhz"]
    )
    panel_c["residual_robust_l1_db"] = (
        panel_c["cn0_observed_trend_dbhz"]
        - (
            panel_c["cn0_physics_trend_dbhz"]
            + panel_c["signal_beta_train_db"]
        )
    )
    panel_c["residual_hgb_db"] = (
        panel_c["cn0_observed_trend_dbhz"]
        - panel_c["cn0_physics_ai_trend_dbhz"]
    )
    panel_c["minute_utc"] = pd.to_datetime(
        panel_c["minute_utc"], utc=True
    )

    quantitative = manifest["quantitative_contribution"][
        "external_complete_op_trend"
    ]
    expected_external = {
        "Physics": quantitative["physics_rmse_dbhz"],
        "Physics + beta": quantitative["after_robust_rmse_dbhz"],
        "Physics + beta + HGB": quantitative["final_rmse_dbhz"],
    }
    for stage, expected in expected_external.items():
        actual = float(external.loc[stage, "rmse_dbhz"])
        if not np.isclose(actual, float(expected), atol=1e-10):
            raise ValueError(f"Manifest/contribution mismatch for {stage}")
    panel_c_rmse = {
        "Physics": float(
            np.sqrt(np.mean(panel_c["residual_physics_db"] ** 2))
        ),
        "Physics + beta": float(
            np.sqrt(np.mean(panel_c["residual_robust_l1_db"] ** 2))
        ),
        "Physics + beta + HGB": float(
            np.sqrt(np.mean(panel_c["residual_hgb_db"] ** 2))
        ),
    }
    for stage, expected in expected_external.items():
        if not np.isclose(panel_c_rmse[stage], float(expected), atol=1e-10):
            raise ValueError(
                f"Panel-c rows do not reproduce complete-OP RMSE for {stage}"
            )

    auth = authoritative[
        authoritative["metric_domain"].eq("cn0_model_performance")
        & authoritative["protocol_id"].eq(
            "final_selected_model_2026-07-17"
        )
        & authoritative["target_kind"].eq("trend")
        & authoritative["group_type"].eq("overall")
    ].copy()
    for split in ["validation", "test", "external_holdout"]:
        contribution_part = (
            overall[overall["evaluation_split"].eq(split)]
            .set_index("stage")
            .reindex(STAGES)
        )
        auth_part = auth[auth["evaluation_split"].eq(split)].set_index(
            "model_stage"
        )
        for contribution_stage, auth_stage in [
            ("Physics", "physics-based baseline"),
            ("Physics + beta + HGB", "physics + AI residual model"),
        ]:
            actual = float(
                contribution_part.loc[contribution_stage, "rmse_dbhz"]
            )
            expected = float(auth_part.loc[auth_stage, "rmse_dbhz"])
            if not np.isclose(actual, expected, atol=1e-10):
                raise ValueError(
                    f"Authoritative mismatch: {split}/{auth_stage}"
                )

    return {
        "contribution": contribution,
        "shares": shares,
        "beta": beta,
        "permutation": permutation,
        "predictions": predictions,
        "panel_c": panel_c,
        "panel_c_rmse": panel_c_rmse,
        "authoritative": authoritative,
        "manifest": manifest,
        "overall": overall,
        "external": external,
        "share_row": share_row,
    }


def panel_label(
    ax: plt.Axes,
    label: str,
    x: float,
    y: float = 0.995,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=COLOR["ink"],
        clip_on=False,
    )


def light_grid(ax: plt.Axes, axis: str) -> None:
    ax.grid(
        True,
        axis=axis,
        color=COLOR["rule"],
        linewidth=0.48,
        alpha=0.82,
        zorder=0,
    )
    ax.set_axisbelow(True)


def _build_figure_three_panel_legacy(
    inputs: dict[str, Any],
) -> tuple[plt.Figure, dict[str, Any]]:
    set_style()
    overall = inputs["overall"]
    external = inputs["external"]
    share_row = inputs["share_row"]
    beta = inputs["beta"].set_index("signal_name").reindex(BETA_ORDER)
    permutation = inputs["permutation"]

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.18],
        height_ratios=[1.04, 1.0],
        left=0.105,
        right=0.985,
        top=0.942,
        bottom=0.165,
        wspace=0.45,
        hspace=0.52,
    )

    # a, the single performance panel across all four data partitions.
    axa = fig.add_subplot(grid[0, :])
    split_x = np.arange(len(SPLITS), dtype=float)
    split_n: list[int] = []
    stage_values: dict[str, np.ndarray] = {}
    label_positions = {
        "Physics": [
            (-0.07, 0.24, "bottom"),
            (0.00, 0.24, "bottom"),
            (0.00, 0.24, "bottom"),
            (-0.07, 0.28, "bottom"),
        ],
        "Physics + beta": [
            (-0.08, 0.30, "bottom"),
            (-0.08, 0.30, "bottom"),
            (-0.08, 0.32, "bottom"),
            (-0.10, 0.38, "bottom"),
        ],
        "Physics + beta + HGB": [
            (0.08, -0.34, "top"),
            (0.08, -0.34, "top"),
            (0.08, -0.36, "top"),
            (-0.12, -0.48, "top"),
        ],
    }
    for stage, color, marker in zip(
        STAGES, STAGE_COLORS, STAGE_MARKERS
    ):
        part = (
            overall[overall["stage"].eq(stage)]
            .set_index("evaluation_split")
            .reindex(SPLITS)
        )
        values = part["rmse_dbhz"].to_numpy(float)
        stage_values[stage] = values
        axa.plot(
            split_x,
            values,
            color=color,
            linewidth=1.28,
            marker=marker,
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.62,
            zorder=3,
        )
        for xpos, value, (dx, dy, valign) in zip(
            split_x, values, label_positions[stage]
        ):
            axa.text(
                xpos + dx,
                value + dy,
                f"{value:.2f}",
                ha="center",
                va=valign,
                fontsize=7.0,
                color=color,
            )
        if not split_n:
            split_n = [int(value) for value in part["n"].to_numpy()]

    axa.set_xticks(
        split_x,
        [
            f"{SPLIT_DISPLAY[split]}\n$n={count:,}$"
            for split, count in zip(SPLITS, split_n)
        ],
    )
    axa.set_xlim(-0.12, 3.42)
    axa.set_ylim(0.0, 9.18)
    axa.set_ylabel("Trend RMSE (dB-Hz)")
    axa.set_title(
        "Layered reconstruction performance",
        loc="left",
        pad=3.5,
    )
    direct_positions = {
        "Physics": float(stage_values["Physics"][-1]) + 0.02,
        "Physics + beta": 2.18,
        "Physics + beta + HGB": 1.28,
    }
    for stage, label, color in zip(STAGES, STAGE_DISPLAY, STAGE_COLORS):
        axa.annotate(
            label,
            xy=(3.0, float(stage_values[stage][-1])),
            xytext=(3.12, direct_positions[stage]),
            ha="left",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=color,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.72,
            },
        )
    light_grid(axa, "y")
    panel_label(axa, "a", x=-0.065)

    # b, four final robust median corrections without distribution claims.
    axb = fig.add_subplot(grid[1, 0])
    beta_values = beta["beta_db"].to_numpy(float)
    beta_y = np.arange(len(beta_values))
    for yvalue, value in zip(beta_y, beta_values):
        axb.hlines(
            yvalue,
            value,
            0,
            color=COLOR["beta_light"],
            linewidth=3.0,
            zorder=1,
        )
        axb.scatter(
            [value],
            [yvalue],
            s=36,
            color=COLOR["beta"],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        axb.text(
            value + 0.48,
            yvalue,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=7.2,
            color=COLOR["ink"],
        )
    axb.axvline(0, color=COLOR["ink"], linewidth=0.72, zorder=2)
    axb.set_yticks(
        beta_y,
        [BETA_DISPLAY[signal] for signal in BETA_ORDER],
    )
    axb.invert_yaxis()
    axb.set_xlim(-10.9, 0.65)
    axb.set_xticks([-10, -8, -6, -4, -2, 0])
    axb.set_xlabel(r"Persistent correction $\beta_s$ (dB)")
    axb.set_title(
        "Persistent corrections learned by Robust L1",
        loc="left",
        y=1.10,
        pad=0,
    )
    axb.text(
        0.0,
        1.015,
        "Final train + validation median refit",
        transform=axb.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=COLOR["muted"],
    )
    light_grid(axb, "x")
    panel_label(axb, "b", x=-0.18)

    # c, ranked grouped permutation sensitivity translated as model reliance.
    axc = fig.add_subplot(grid[1, 1])
    sensitivity = (
        permutation.groupby("feature_group")["delta_rmse_dbhz"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    sensitivity["display"] = [
        PERMUTATION_DISPLAY[index] for index in sensitivity.index
    ]
    means = sensitivity["mean"].to_numpy(float)
    errors = sensitivity["std"].to_numpy(float)
    positive_total = float(means[means > 0].sum())
    positive_shares = np.where(
        means > 0,
        100.0 * means / positive_total,
        np.nan,
    )
    ypos = np.arange(len(sensitivity))
    bar_colors = [
        COLOR["hgb"] if value > 0 else COLOR["neutral_light"]
        for value in means
    ]
    axc.barh(
        ypos,
        means,
        height=0.48,
        color=bar_colors,
        edgecolor="none",
        alpha=0.90,
        zorder=2,
    )
    axc.errorbar(
        means,
        ypos,
        xerr=errors,
        fmt="none",
        ecolor=COLOR["neutral"],
        elinewidth=0.78,
        capsize=2.0,
        capthick=0.78,
        zorder=3,
    )
    axc.axvline(0, color=COLOR["ink"], linewidth=0.72, zorder=2)
    for yvalue, mean_value, error, share in zip(
        ypos, means, errors, positive_shares
    ):
        if mean_value > 0:
            share_label = (
                "<1%" if share < 1.0 else f"{share:.0f}%"
            )
            text = f"{mean_value:.3f}  ({share_label})"
            text_x = mean_value + error + 0.006
            color = COLOR["ink"]
        else:
            text = f"{mean_value:.3f}; no measurable reliance"
            text_x = 0.006
            color = COLOR["muted"]
        axc.text(
            text_x,
            yvalue,
            text,
            ha="left",
            va="center",
            fontsize=7.0,
            color=color,
        )
    axc.set_yticks(ypos, sensitivity["display"])
    axc.set_ylim(5.35, -0.72)
    axc.set_xlim(-0.022, 0.247)
    axc.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
    axc.set_xlabel(
        "RMSE increase after feature-group shuffling (dB-Hz)"
    )
    axc.set_title("Information used by HGB", loc="left", pad=3.5)
    axc.text(
        0.98,
        0.975,
        "larger increase = stronger model reliance",
        transform=axc.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=COLOR["muted"],
    )
    light_grid(axc, "x")
    panel_label(axc, "c", x=-0.18)

    fig.text(
        0.985,
        0.028,
        "Positive shares normalize positive mean sensitivities only; "
        "descriptive, not causal attribution.",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=COLOR["muted"],
    )

    fixed_share = float(share_row["fixed_share_total_recovered_pct"])
    hgb_share = float(
        share_row["timevarying_share_total_recovered_pct"]
    )
    remaining = float(
        share_row["timevarying_reduction_of_remaining_mse_pct"]
    )
    metadata = {
        "external_rmse_dbhz": {
            stage: float(external.loc[stage, "rmse_dbhz"])
            for stage in STAGES
        },
        "external_n": int(external["n"].iloc[0]),
        "caption_only_sequential_mse_share_pct": {
            "robust_l1": fixed_share,
            "hgb": hgb_share,
            "hgb_reduction_of_post_l1_mse": remaining,
            "rendering": "caption/provenance only; not drawn",
        },
        "beta_db": {
            signal: float(beta.loc[signal, "beta_db"])
            for signal in BETA_ORDER
        },
        "split_rmse_dbhz": {
            stage: {
                split: float(value)
                for split, value in zip(SPLITS, stage_values[stage])
            }
            for stage in STAGES
        },
        "split_n": {
            split: int(value) for split, value in zip(SPLITS, split_n)
        },
        "grouped_permutation": [
            {
                "source_group": str(index),
                "display_group": str(row["display"]).replace("\n", " "),
                "mean_delta_rmse_dbhz": float(row["mean"]),
                "sd_delta_rmse_dbhz": float(row["std"]),
                "repeats": int(row["count"]),
                "normalized_positive_sensitivity_pct": (
                    float(100.0 * row["mean"] / positive_total)
                    if row["mean"] > 0
                    else None
                ),
                "interpretation": (
                    "model reliance"
                    if row["mean"] > 0
                    else "no measurable reliance"
                ),
            }
            for index, row in sensitivity.iterrows()
        ],
        "positive_permutation_share_contract": {
            "denominator": (
                "sum of positive grouped mean delta-RMSE sensitivities"
            ),
            "positive_total_delta_rmse_dbhz": positive_total,
            "interpretation": (
                "descriptive normalization of predictive sensitivities; "
                "not additive causal physical attribution"
            ),
        },
    }
    return fig, metadata


def build_figure(inputs: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    set_style()
    overall = inputs["overall"]
    external = inputs["external"]
    share_row = inputs["share_row"]
    beta = inputs["beta"].set_index("signal_name").reindex(BETA_ORDER)
    permutation = inputs["permutation"]
    panel_c = inputs["panel_c"].copy()

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.18],
        height_ratios=[0.88, 1.12],
        left=0.090,
        right=0.990,
        top=0.970,
        bottom=0.145,
        wspace=0.56,
        hspace=0.46,
    )

    # a, hero performance panel across the four frozen partitions.
    axa = fig.add_subplot(grid[0, :2])
    split_x = np.arange(len(SPLITS), dtype=float)
    split_n: list[int] = []
    stage_values: dict[str, np.ndarray] = {}
    label_positions = {
        "Physics": [
            (0.03, 0.24),
            (0.00, 0.24),
            (0.00, 0.24),
            (-0.08, 0.28),
        ],
        "Physics + beta": [
            (-0.08, 0.28),
            (-0.08, 0.28),
            (-0.08, 0.30),
            (-0.14, 0.34),
        ],
        "Physics + beta + HGB": [
            (0.08, -0.30),
            (0.08, -0.30),
            (0.08, -0.32),
            (-0.14, -0.42),
        ],
    }
    for stage, color, marker in zip(
        STAGES, STAGE_COLORS, STAGE_MARKERS
    ):
        part = (
            overall[overall["stage"].eq(stage)]
            .set_index("evaluation_split")
            .reindex(SPLITS)
        )
        values = part["rmse_dbhz"].to_numpy(float)
        stage_values[stage] = values
        axa.plot(
            split_x,
            values,
            color=color,
            linewidth=1.25,
            marker=marker,
            markersize=4.6,
            markeredgecolor="white",
            markeredgewidth=0.60,
            zorder=3,
        )
        for xpos, value, (x_offset, y_offset) in zip(
            split_x,
            values,
            label_positions[stage],
        ):
            axa.text(
                xpos + x_offset,
                value + y_offset,
                f"{value:.2f}",
                ha="center",
                va="bottom" if y_offset > 0 else "top",
                fontsize=7.0,
                color=color,
            )
        if not split_n:
            split_n = [int(value) for value in part["n"].to_numpy()]

    axa.set_xticks(
        split_x,
        [
            f"{SPLIT_DISPLAY[split]}\n$n={count:,}$"
            for split, count in zip(SPLITS, split_n)
        ],
    )
    axa.set_xlim(-0.12, 3.47)
    axa.set_ylim(0.0, 9.18)
    axa.set_yticks([0, 2, 4, 6, 8])
    axa.set_ylabel("Trend RMSE (dB-Hz)")
    direct_positions = {
        "Physics": float(stage_values["Physics"][-1]) + 0.02,
        "Physics + beta": 2.55,
        "Physics + beta + HGB": 0.96,
    }
    for stage, label, color in zip(STAGES, STAGE_DISPLAY, STAGE_COLORS):
        axa.annotate(
            label,
            xy=(3.0, float(stage_values[stage][-1])),
            xytext=(3.11, direct_positions[stage]),
            ha="left",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color=color,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.70,
            },
        )
    light_grid(axa, "y")
    panel_label(axa, "a", x=-0.085)

    # b, the four frozen Robust L1 constellation-band corrections.
    axb = fig.add_subplot(grid[0, 2])
    beta_values = beta["beta_db"].to_numpy(float)
    beta_y = np.arange(len(beta_values))
    for yvalue, value in zip(beta_y, beta_values):
        axb.hlines(
            yvalue,
            value,
            0,
            color=COLOR["beta_light"],
            linewidth=2.8,
            zorder=1,
        )
        axb.scatter(
            [value],
            [yvalue],
            s=46,
            color=COLOR["beta"],
            edgecolor="white",
            linewidth=0.62,
            zorder=3,
        )
        axb.text(
            value + 0.50,
            yvalue,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=COLOR["ink"],
        )
    axb.axvline(0, color=COLOR["ink"], linewidth=0.70, zorder=2)
    axb.set_yticks(
        beta_y,
        [BETA_DISPLAY[signal] for signal in BETA_ORDER],
    )
    axb.invert_yaxis()
    axb.set_xlim(-10.9, 0.65)
    axb.set_xticks([-10, -8, -6, -4, -2, 0])
    axb.set_xlabel(r"Correction $\beta_s$ (dB)")
    light_grid(axb, "x")
    panel_label(axb, "b", x=-0.40)

    # c, real residual distributions in the four complete unseen operations.
    axc = fig.add_subplot(grid[1, :2])
    centers = np.arange(len(OP_ORDER), dtype=float)
    box_methods = [
        (
            "Physics",
            "residual_physics_db",
            COLOR["physics"],
            COLOR["physics_light"],
            -0.24,
        ),
        (
            "+ robust L1",
            "residual_robust_l1_db",
            COLOR["beta"],
            COLOR["beta_light"],
            0.0,
        ),
        (
            "+ HGB",
            "residual_hgb_db",
            COLOR["hgb"],
            COLOR["hgb_light"],
            0.24,
        ),
    ]
    for label, column, color, fill, offset in box_methods:
        values = [
            panel_c.loc[panel_c["op"].eq(op), column].to_numpy(float)
            for op in OP_ORDER
        ]
        axc.boxplot(
            values,
            positions=centers + offset,
            widths=0.18,
            orientation="vertical",
            whis=1.5,
            patch_artist=True,
            showmeans=False,
            showfliers=True,
            manage_ticks=False,
            boxprops={
                "facecolor": fill,
                "edgecolor": color,
                "linewidth": 0.85,
            },
            whiskerprops={"color": color, "linewidth": 0.72},
            capprops={"color": color, "linewidth": 0.72},
            medianprops={"color": color, "linewidth": 1.05},
            flierprops={
                "marker": "o",
                "markersize": 1.90,
                "markerfacecolor": color,
                "markeredgecolor": "none",
                "alpha": 0.20,
            },
        )
    axc.axhline(
        0,
        color=COLOR["ink"],
        linewidth=0.70,
        linestyle=(0, (3.0, 2.2)),
        zorder=1,
    )
    op_counts = {
        op: int(panel_c["op"].eq(op).sum()) for op in OP_ORDER
    }
    axc.set_xticks(
        centers,
        [
            f"OP2 | C\n$n={op_counts['OP2']:,}$",
            f"OP21 | T\n$n={op_counts['OP21']:,}$",
            f"OP27 | L\n$n={op_counts['OP27']:,}$",
            f"OP74 | S\n$n={op_counts['OP74']:,}$",
        ],
    )
    residual_columns = [
        "residual_physics_db",
        "residual_robust_l1_db",
        "residual_hgb_db",
    ]
    residual_min = float(panel_c[residual_columns].min().min())
    residual_max = float(panel_c[residual_columns].max().max())
    y_min = 2.0 * np.floor((residual_min - 0.5) / 2.0)
    y_max = 2.0 * np.ceil((residual_max + 0.5) / 2.0)
    axc.set_xlim(-0.55, 3.55)
    axc.set_ylim(-15.0, 10.0)
    axc.set_yticks([-15, -10, -5, 0, 5, 10])
    axc.set_ylabel("Residual, observed - model (dB-Hz)")
    axc.legend(
        handles=[
            Patch(
                facecolor=fill,
                edgecolor=color,
                linewidth=0.8,
                label=label,
            )
            for label, _, color, fill, _ in box_methods
        ],
        loc="lower center",
        bbox_to_anchor=(0.50, 1.002),
        ncol=3,
        frameon=False,
        handlelength=1.15,
        handletextpad=0.38,
        columnspacing=0.90,
        borderaxespad=0,
    )
    light_grid(axc, "y")
    panel_label(axc, "c", x=-0.115, y=1.035)

    # d, grouped permutation sensitivity expressed as predictive reliance.
    axd = fig.add_subplot(grid[1, 2])
    sensitivity = (
        permutation.groupby("feature_group")["delta_rmse_dbhz"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    sensitivity["display"] = [
        PERMUTATION_DISPLAY[index] for index in sensitivity.index
    ]
    means = sensitivity["mean"].to_numpy(float)
    errors = sensitivity["std"].to_numpy(float)
    positive_total = float(means[means > 0].sum())
    positive_shares = np.where(
        means > 0,
        100.0 * means / positive_total,
        np.nan,
    )
    ypos = np.arange(len(sensitivity))
    resolved = means > 0
    display_means = np.where(resolved, means, 0.0)
    axd.barh(
        ypos,
        np.where(resolved, means, 0.0),
        height=0.48,
        color=COLOR["hgb"],
        edgecolor="none",
        alpha=0.90,
        zorder=2,
    )
    axd.errorbar(
        display_means,
        ypos,
        xerr=errors,
        fmt="none",
        ecolor=COLOR["neutral"],
        elinewidth=0.76,
        capsize=1.9,
        capthick=0.76,
        zorder=3,
    )
    axd.scatter(
        np.zeros(int((~resolved).sum())),
        ypos[~resolved],
        s=24,
        facecolors="white",
        edgecolors=COLOR["neutral"],
        linewidths=0.85,
        zorder=4,
    )
    axd.axvline(0, color=COLOR["ink"], linewidth=0.70, zorder=2)
    for yvalue, mean_value, error, share in zip(
        ypos, means, errors, positive_shares
    ):
        if mean_value > 0:
            share_label = "<1%" if share < 1.0 else f"{share:.0f}%"
            text = f"{mean_value:.3f} ({share_label})"
            text_x = mean_value + error + 0.003
            color = COLOR["ink"]
        else:
            text = "not resolved"
            text_x = error + 0.006
            color = COLOR["muted"]
        axd.text(
            text_x,
            yvalue,
            text,
            ha="left",
            va="center",
            fontsize=7.0,
            color=color,
        )
    axd.set_yticks(ypos, sensitivity["display"])
    axd.set_ylim(5.35, -1.20)
    axd.set_xlim(-0.018, 0.290)
    axd.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
    axd.spines["left"].set_visible(False)
    axd.tick_params(axis="y", length=0, pad=3.0)
    axd.set_xlabel(
        "Increase in prediction RMSE\nafter shuffling (dB-Hz)",
        linespacing=1.15,
    )
    axd.text(
        0.98,
        0.98,
        "mean ± s.d.; 30 shuffles\nlarger = greater model reliance",
        transform=axd.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=COLOR["muted"],
        linespacing=1.18,
    )
    light_grid(axd, "x")
    panel_label(axd, "d", x=-0.24)

    fixed_share = float(share_row["fixed_share_total_recovered_pct"])
    hgb_share = float(
        share_row["timevarying_share_total_recovered_pct"]
    )
    remaining = float(
        share_row["timevarying_reduction_of_remaining_mse_pct"]
    )
    panel_c_summary: dict[str, Any] = {}
    for op in OP_ORDER:
        group = panel_c[panel_c["op"].eq(op)]
        stage_summary = {}
        for stage, column in [
            ("Physics", "residual_physics_db"),
            ("Physics + beta", "residual_robust_l1_db"),
            ("Physics + beta + HGB", "residual_hgb_db"),
        ]:
            values = group[column].to_numpy(float)
            stage_summary[stage] = {
                "rmse_dbhz": float(np.sqrt(np.mean(values**2))),
                "median_dbhz": float(np.median(values)),
                "q1_dbhz": float(np.quantile(values, 0.25)),
                "q3_dbhz": float(np.quantile(values, 0.75)),
                "min_dbhz": float(np.min(values)),
                "max_dbhz": float(np.max(values)),
            }
        panel_c_summary[op] = {
            "phase_code": OP_PHASE[op],
            "phase_name": (
                "Commissioning" if op == "OP2" else OP_PHASE[op]
            ),
            "n": int(len(group)),
            "stages": stage_summary,
        }

    metadata = {
        "external_rmse_dbhz": {
            stage: float(external.loc[stage, "rmse_dbhz"])
            for stage in STAGES
        },
        "external_n": int(external["n"].iloc[0]),
        "caption_only_sequential_mse_share_pct": {
            "robust_l1": fixed_share,
            "hgb": hgb_share,
            "hgb_reduction_of_post_l1_mse": remaining,
            "rendering": "caption/provenance only; not drawn",
        },
        "beta_db": {
            signal: float(beta.loc[signal, "beta_db"])
            for signal in BETA_ORDER
        },
        "split_rmse_dbhz": {
            stage: {
                split: float(value)
                for split, value in zip(SPLITS, stage_values[stage])
            }
            for stage in STAGES
        },
        "split_n": {
            split: int(value) for split, value in zip(SPLITS, split_n)
        },
        "representative_operation_residuals": panel_c_summary,
        "panel_c_total_n": int(len(panel_c)),
        "panel_c_overall_rmse_reproduction": inputs["panel_c_rmse"],
        "grouped_permutation": [
            {
                "source_group": str(index),
                "display_group": str(row["display"]).replace("\n", " "),
                "mean_delta_rmse_dbhz": float(row["mean"]),
                "sd_delta_rmse_dbhz": float(row["std"]),
                "repeats": int(row["count"]),
                "normalized_positive_sensitivity_pct": (
                    float(100.0 * row["mean"] / positive_total)
                    if row["mean"] > 0
                    else None
                ),
                "interpretation": (
                    "predictive reliance"
                    if row["mean"] > 0
                    else "not resolved"
                ),
            }
            for index, row in sensitivity.iterrows()
        ],
        "positive_permutation_share_contract": {
            "denominator": (
                "sum of positive grouped mean delta-RMSE sensitivities"
            ),
            "positive_total_delta_rmse_dbhz": positive_total,
            "interpretation": (
                "descriptive normalization of positive predictive "
                "sensitivities; not additive causal physical attribution"
            ),
        },
    }
    return fig, metadata


def text_layout_qa(fig: plt.Figure) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    outside: list[str] = []
    font_sizes: list[float] = []
    visible_strings: list[str] = []
    for text in fig.findobj(match=mpl.text.Text):
        value = text.get_text().strip()
        if not text.get_visible() or not value:
            continue
        visible_strings.append(value)
        font_sizes.append(float(text.get_fontsize()))
        box = text.get_window_extent(renderer=renderer)
        if (
            box.x0 < figure_box.x0 - 1
            or box.y0 < figure_box.y0 - 1
            or box.x1 > figure_box.x1 + 1
            or box.y1 > figure_box.y1 + 1
        ):
            outside.append(value)
    forbidden = [
        value
        for value in visible_strings
        if re.search(r"\bholdout\b", value, flags=re.IGNORECASE)
    ]
    removed_terms = [
        value
        for value in visible_strings
        if re.search(
            r"recovered\s+MSE|post-L1\s+MSE",
            value,
            flags=re.IGNORECASE,
        )
    ]
    if outside:
        raise RuntimeError(f"Figure text outside canvas: {outside}")
    if forbidden:
        raise RuntimeError(f"Forbidden visible terminology: {forbidden}")
    if removed_terms:
        raise RuntimeError(f"Removed panel content is still visible: {removed_terms}")
    if len(fig.axes) != 4:
        raise RuntimeError(f"Expected four axes, found {len(fig.axes)}")
    panel_titles = [
        ax.get_title(loc=location).strip()
        for ax in fig.axes
        for location in ["left", "center", "right"]
        if ax.get_title(loc=location).strip()
    ]
    if panel_titles:
        raise RuntimeError(f"Panel titles remain visible: {panel_titles}")
    removed_annotations = [
        value
        for value in visible_strings
        if (
            "not used for training or tuning" in value
            or "Commissioning" in value
        )
    ]
    if removed_annotations:
        raise RuntimeError(
            f"Removed in-panel annotations remain: {removed_annotations}"
        )
    minimum = min(font_sizes)
    if minimum < 7.0 - 1e-9:
        raise RuntimeError(f"Visible font below 7 pt: {minimum}")
    return {
        "visible_text_count": int(len(visible_strings)),
        "minimum_visible_font_pt": float(minimum),
        "maximum_visible_font_pt": float(max(font_sizes)),
        "text_outside_canvas": outside,
        "visible_holdout_term_count": int(len(forbidden)),
        "removed_panel_term_count": int(len(removed_terms)),
        "panel_title_count": int(len(panel_titles)),
        "removed_annotation_count": int(len(removed_annotations)),
        "main_axes_count": int(len(fig.axes)),
    }


def save_outputs(fig: plt.Figure) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    formats = [
        (".png", {"dpi": EXPORT_DPI}),
        (
            ".jpg",
            {
                "dpi": EXPORT_DPI,
                "pil_kwargs": {
                    "quality": 96,
                    "subsampling": 0,
                    "optimize": True,
                },
            },
        ),
        (".pdf", {}),
        (".svg", {}),
        (
            ".tiff",
            {
                "dpi": EXPORT_DPI,
                "pil_kwargs": {"compression": "tiff_lzw"},
            },
        ),
    ]
    for suffix, kwargs in formats:
        path = OUTPUT_BASE.with_suffix(suffix)
        fig.savefig(
            path,
            facecolor="white",
            edgecolor="none",
            transparent=False,
            **kwargs,
        )
        outputs.append(path)
    return outputs


def write_preview_1600() -> Path:
    source = OUTPUT_BASE.with_suffix(".png")
    preview = OUTPUT_BASE.with_name("figure3_preview_1600px.png")
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        target_height = int(round(rgb.height * 1600 / rgb.width))
        resized = rgb.resize(
            (1600, target_height),
            resample=Image.Resampling.LANCZOS,
        )
        resized.save(preview, format="PNG", optimize=True)
    return preview


def raster_qa() -> dict[str, Any]:
    png_path = OUTPUT_BASE.with_suffix(".png")
    jpg_path = OUTPUT_BASE.with_suffix(".jpg")
    with Image.open(png_path) as image:
        rgba = np.asarray(image.convert("RGBA"))
        width, height = image.size
        alpha = rgba[:, :, 3]
        corners = [
            rgba[0, 0].tolist(),
            rgba[0, -1].tolist(),
            rgba[-1, 0].tolist(),
            rgba[-1, -1].tolist(),
        ]
    with Image.open(jpg_path) as image:
        jpg_mode = image.mode
        jpg_size = image.size
    expected_width = int(WIDTH_MM / 25.4 * EXPORT_DPI)
    expected_height = int(HEIGHT_MM / 25.4 * EXPORT_DPI)
    if (width, height) != (expected_width, expected_height):
        raise RuntimeError(
            f"Unexpected PNG size {(width, height)}; "
            f"expected {(expected_width, expected_height)}"
        )
    if int(alpha.min()) != 255:
        raise RuntimeError("PNG contains transparency")
    if any(corner != [255, 255, 255, 255] for corner in corners):
        raise RuntimeError(f"PNG corners are not white: {corners}")
    if jpg_mode != "RGB" or jpg_size != (width, height):
        raise RuntimeError("JPG mode or dimensions are invalid")
    return {
        "png_size_px": [int(width), int(height)],
        "expected_size_px": [expected_width, expected_height],
        "png_alpha_min": int(alpha.min()),
        "png_alpha_max": int(alpha.max()),
        "png_corners_rgba": corners,
        "jpg_mode": jpg_mode,
        "jpg_size_px": [int(jpg_size[0]), int(jpg_size[1])],
        "white_opaque_background": True,
    }


def write_panel_c_source_data(panel_c: pd.DataFrame) -> Path:
    columns = [
        "minute_utc",
        "op",
        "mission_phase",
        "evaluation_split",
        "signal_name",
        "svid",
        "cn0_observed_trend_dbhz",
        "cn0_physics_trend_dbhz",
        "signal_beta_train_db",
        "cn0_physics_ai_trend_dbhz",
        "residual_physics_db",
        "residual_robust_l1_db",
        "residual_hgb_db",
    ]
    output = panel_c[columns].copy()
    output["op_order"] = output["op"].map(
        {op: index for index, op in enumerate(OP_ORDER)}
    )
    output = output.sort_values(
        ["op_order", "minute_utc", "signal_name", "svid"],
        kind="stable",
    ).drop(columns="op_order")
    output["minute_utc"] = output["minute_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    output.to_csv(PANEL_C_SOURCE_PATH, index=False, encoding="utf-8")
    if len(output) != 4112:
        raise RuntimeError(
            f"Unexpected panel-c source row count: {len(output)}"
        )
    return PANEL_C_SOURCE_PATH


def vector_qa() -> dict[str, Any]:
    svg_path = OUTPUT_BASE.with_suffix(".svg")
    pdf_path = OUTPUT_BASE.with_suffix(".pdf")
    root = ET.parse(svg_path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    text_nodes = root.findall(".//svg:text", namespace)
    svg_text = " ".join("".join(node.itertext()) for node in text_nodes)
    for label in [
        "Trend RMSE (dB-Hz)",
        "GPS L1",
        "Residual, observed - model (dB-Hz)",
        "Increase in prediction RMSE",
    ]:
        if label not in svg_text:
            raise RuntimeError(f"Missing vector label: {label}")
    if re.search(r"\bholdout\b", svg_text, flags=re.IGNORECASE):
        raise RuntimeError("SVG contains forbidden visible terminology")
    if len(text_nodes) < 50:
        raise RuntimeError("SVG text was unexpectedly flattened")
    pdf_bytes = pdf_path.read_bytes()
    pdf_pages = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    pdf_font_objects = len(re.findall(rb"/Type\s*/Font\b", pdf_bytes))
    if pdf_pages != 1:
        raise RuntimeError(f"Expected one PDF page, found {pdf_pages}")
    if pdf_font_objects < 1:
        raise RuntimeError("PDF contains no font objects")
    return {
        "svg_text_nodes": int(len(text_nodes)),
        "svg_text_editable": True,
        "pdf_pages": int(pdf_pages),
        "pdf_font_objects": int(pdf_font_objects),
        "pdf_text_editable": True,
        "pdf_text_qa_method": (
            "embedded font-object audit; SVG text-node audit provides the "
            "direct editable-text check"
        ),
        "visible_holdout_term_count_svg": 0,
    }


def write_input_table() -> Path:
    source_rows = {
        CONTRIBUTION_PATH: pd.read_csv(CONTRIBUTION_PATH),
        SHARES_PATH: pd.read_csv(SHARES_PATH),
        BETA_PATH: pd.read_csv(BETA_PATH),
        PERMUTATION_PATH: pd.read_csv(PERMUTATION_PATH),
        PREDICTION_PATH: pd.read_csv(PREDICTION_PATH, low_memory=False),
        AUTHORITATIVE_PATH: pd.read_csv(AUTHORITATIVE_PATH),
    }
    mapping = [
        (
            "a",
            "sequential stage RMSE across four data partitions",
            CONTRIBUTION_PATH,
        ),
        (
            "b",
            "four final robust median corrections",
            BETA_PATH,
        ),
        (
            "c",
            "eligible one-minute trend residuals in four complete operations",
            PREDICTION_PATH,
        ),
        (
            "d",
            "30-repeat grouped permutation predictive sensitivity",
            PERMUTATION_PATH,
        ),
        (
            "caption",
            "order-dependent MSE shares retained for caption only",
            SHARES_PATH,
        ),
        (
            "a",
            "authoritative physics/final metric cross-check",
            AUTHORITATIVE_PATH,
        ),
        (
            "a-d",
            "final model, split, coefficient, and claim contract",
            MANIFEST_PATH,
        ),
        (
            "a-d",
            "figure-specific metric comparability contract",
            INPUT_CONTRACT_PATH,
        ),
    ]
    rows = []
    for panel, role, path in mapping:
        frame = source_rows.get(path)
        rows.append(
            {
                "panel": panel,
                "role": role,
                "source_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "rows": int(len(frame)) if frame is not None else "",
                "columns": int(len(frame.columns)) if frame is not None else "",
            }
        )
    output = OUTPUT_BASE.with_name("figure3_input_tables.csv")
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    return output


def write_provenance(
    metadata: dict[str, Any],
    layout_qa: dict[str, Any],
    raster_checks: dict[str, Any],
    vector_checks: dict[str, Any],
    outputs: list[Path],
    input_table: Path,
    protected_before: list[dict[str, Any]],
    protected_after: list[dict[str, Any]],
) -> Path:
    input_paths = [
        CONTRIBUTION_PATH,
        SHARES_PATH,
        BETA_PATH,
        PERMUTATION_PATH,
        PREDICTION_PATH,
        MANIFEST_PATH,
        AUTHORITATIVE_PATH,
        INPUT_CONTRACT_PATH,
    ]
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "core_conclusion": (
            "Robust L1 removes the dominant persistent constellation-band "
            "level bias; HGB supplies a smaller geometry-dependent correction "
            "whose benefit varies across mission phases, and its predictive "
            "reliance is concentrated in baseline/range and antenna geometry."
        ),
        "evidence_chain": {
            "a": (
                "the same three model layers across training fit, validation "
                "selection, internal test, and four complete operations not "
                "used for training or tuning"
            ),
            "b": "four final persistent robust median corrections",
            "c": (
                "three-layer residual distributions in OP2, OP21, OP27, and "
                "OP74 using eligible one-minute trend rows only"
            ),
            "d": (
                "ranked grouped permutation predictive sensitivity with "
                "mean +/- s.d. and positive-only descriptive normalization"
            ),
        },
        "archetype": (
            "asymmetric quantitative grid with hero performance and residual "
            "distribution panels"
        ),
        "backend": "Python/matplotlib only",
        "figure_size_mm": {"width": WIDTH_MM, "height": HEIGHT_MM},
        "export_dpi": EXPORT_DPI,
        "model_policy": (
            "No model artifact was loaded and no model was trained, refit, "
            "tuned, or evaluated. The script reads frozen metric and final "
            "prediction tables only."
        ),
        "metric_contract": {
            "target": "split-isolated one-minute C/N0 trend",
            "mask": "trend_training_eligible",
            "geometry": "WGC-by-TLM exact receiver reference geometry",
            "ephemeris": "CODE MGEX final SP3",
            "external_operations": ["OP2", "OP21", "OP27", "OP74"],
            "external_fit_status": (
                "not used for residual training, tuning, or final refit"
            ),
            "panel_c_residual_sign": "observed trend minus model trend",
            "panel_c_physics": (
                "cn0_observed_trend_dbhz - cn0_physics_trend_dbhz"
            ),
            "panel_c_robust_l1": (
                "cn0_observed_trend_dbhz - "
                "(cn0_physics_trend_dbhz + signal_beta_train_db)"
            ),
            "panel_c_hgb": (
                "cn0_observed_trend_dbhz - "
                "cn0_physics_ai_trend_dbhz"
            ),
            "panel_c_cadence": "eligible one-minute trend samples only",
            "sequential_share": (
                "caption-only order-dependent MSE recovery; not rendered"
            ),
            "permutation": (
                "predictive model reliance, not additive causal physical "
                "attribution"
            ),
        },
        "metrics": metadata,
        "inputs": [file_record(path) for path in input_paths],
        "style_references": [
            file_record(STYLE_SCRIPT_PATH),
            *[file_record(path) for path in STYLE_REFERENCE_PATHS],
        ],
        "plot_script": file_record(Path(__file__)),
        "input_table_manifest": file_record(input_table),
        "panel_c_source_data": file_record(PANEL_C_SOURCE_PATH),
        "versioned_three_panel_backup": {
            "path": THREE_PANEL_BACKUP_DIR.relative_to(ROOT).as_posix(),
            "manifest": file_record(
                THREE_PANEL_BACKUP_DIR / "backup_manifest.csv"
            ),
        },
        "versioned_four_panel_backup": {
            "path": FOUR_PANEL_BACKUP_DIR.relative_to(ROOT).as_posix(),
            "manifest": file_record(
                FOUR_PANEL_BACKUP_DIR / "backup_manifest.csv"
            ),
        },
        "protected_legacy_figai2": {
            "unchanged": True,
            "before": protected_before,
            "after": protected_after,
        },
        "qa": {**layout_qa, **raster_checks, **vector_checks},
        "outputs": [file_record(path) for path in outputs],
    }
    path = OUTPUT_BASE.with_name("figure3_provenance.json")
    path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return path


def main() -> None:
    for backup_dir in [THREE_PANEL_BACKUP_DIR, FOUR_PANEL_BACKUP_DIR]:
        backup_manifest = backup_dir / "backup_manifest.csv"
        if not backup_manifest.exists():
            raise FileNotFoundError(backup_manifest)
    protected_before = protected_snapshot()
    inputs = load_inputs()
    fig, metadata = build_figure(inputs)
    layout_checks = text_layout_qa(fig)
    outputs = save_outputs(fig)
    plt.close(fig)

    preview = write_preview_1600()
    outputs.append(preview)
    panel_c_source = write_panel_c_source_data(inputs["panel_c"])
    outputs.append(panel_c_source)
    raster_checks = raster_qa()
    vector_checks = vector_qa()
    input_table = write_input_table()
    protected_after = protected_snapshot()
    assert_protected_unchanged(protected_before, protected_after)
    provenance = write_provenance(
        metadata,
        layout_checks,
        raster_checks,
        vector_checks,
        outputs,
        input_table,
        protected_before,
        protected_after,
    )

    print(
        "complete_OP_RMSE="
        + " -> ".join(
            f"{metadata['external_rmse_dbhz'][stage]:.6f}"
            for stage in STAGES
        )
    )
    print(
        "positive_permutation_total="
        f"{metadata['positive_permutation_share_contract']['positive_total_delta_rmse_dbhz']:.6f}"
    )
    for output in outputs:
        print(output)
    print(input_table)
    print(provenance)


if __name__ == "__main__":
    main()
