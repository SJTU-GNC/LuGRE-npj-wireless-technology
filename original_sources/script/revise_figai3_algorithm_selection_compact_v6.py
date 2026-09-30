#!/usr/bin/env python3
"""Render Fig. AI3 compact v6 from persisted validation evidence only."""

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
from matplotlib.patches import Rectangle
from PIL import Image, ImageChops

import revise_figai3_algorithm_selection_compact_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure" / "paper_draft_v2" / "ai_residual_story"
STEM = "FigAI3_algorithm_and_hyperparameter_selection_compact_v6"

PREDICTION_PATH = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_ai_residual_story"
    / "validation_algorithm_predictions_for_block_bootstrap.csv"
)
PANEL_A_SOURCE_PATH = FIG_DIR / "FigAI3_compact_v6_panel_a_robust_regression.csv"
PANEL_B_SOURCE_PATH = FIG_DIR / "FigAI3_compact_v6_panel_b_algorithm_screen.csv"
PANEL_C_SOURCE_PATH = FIG_DIR / "FigAI3_compact_v6_panel_c_phase_rmse_matrix.csv"
PANEL_D_SOURCE_PATH = FIG_DIR / "FigAI3_compact_v6_panel_d_candidate_parameter_matrix.csv"
PANEL_E_LONG_SOURCE_PATH = FIG_DIR / "FigAI3_compact_v6_panel_e_candidate_robustness_long.csv"
PANEL_E_SOURCE_PATH = FIG_DIR / "FigAI3_compact_v6_panel_e_candidate_robustness_summary.csv"
PANEL_F_SOURCE_PATH = FIG_DIR / "FigAI3_compact_v6_panel_f_parameter_feature_combinations.csv"
CAPTION_QA_PATH = FIG_DIR / f"{STEM}_caption_QA.md"
PROVENANCE_PATH = FIG_DIR / f"{STEM}_provenance.json"

FIG_WIDTH_MM = 183.0
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
FIG_HEIGHT_IN = 4.67
EXPORT_DPI = 600
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 240719

COL = v2.COL
PARAMETER_ORDER = [f"p{i}" for i in range(6)]
ALGORITHM_ORDER = [
    "HGB",
    "LightGBM",
    "Extra trees",
    "CatBoost",
    "MLP",
    "Random forest",
    "Huber linear",
]
FEATURE_ORDER = ["current", "stable_compact", "no_grap_uncertainty", "direct_budget_only"]
FEATURE_LABELS = {
    "current": "Full\ndescriptors",
    "stable_compact": "Stable\ncompact",
    "no_grap_uncertainty": "Without GRAP\nuncertainty\nterms",
    "direct_budget_only": "Direct link-\nbudget only",
}
FEATURE_DEFINITIONS = {
    "current": "All 61 persisted smoothed model descriptors.",
    "stable_compact": (
        "The 44-descriptor direct-budget subset after removal of the six "
        "persisted unstable raw descriptors."
    ),
    "no_grap_uncertainty": (
        "The 53-descriptor full set after removal of GRAP uncertainty-summary terms."
    ),
    "direct_budget_only": (
        "The 50-descriptor full set after removal of all GRAP-derived descriptors."
    ),
}
PANEL_F_FEATURE_ORDER = [
    "stable_compact",
    "current",
    "no_grap_uncertainty",
    "direct_budget_only",
]
PANEL_F_FEATURE_LABELS = {
    "stable_compact": "Compact\nphysical\n(44)",
    "current": "Full\nphysical\n(61)",
    "no_grap_uncertainty": "Reduced\nuncertainty\n(53)",
    "direct_budget_only": "Direct\nbudget\n(50)",
}
PARAMETER_COLUMNS = [
    "max_iter",
    "learning_rate",
    "max_leaf_nodes",
    "min_samples_leaf",
    "l2_regularization",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(observed - predicted))))


def set_style() -> None:
    v2.set_style()
    mpl.rcParams.update(
        {
            "font.size": 6.35,
            "axes.titlesize": 7.15,
            "axes.labelsize": 6.45,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.25,
        }
    )


def prepare_algorithm_comparison(
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pair = (
        benchmark[benchmark["algorithm"].isin(["HGB", "LightGBM"])]
        .set_index("algorithm")
        .reindex(["HGB", "LightGBM"])
    )
    if len(pair) != 2 or pair["validation_rmse_dbhz"].isna().any():
        raise RuntimeError("Persisted HGB/LightGBM algorithm benchmark is incomplete")
    if pair.loc["HGB", "validation_rmse_dbhz"] >= pair.loc["LightGBM", "validation_rmse_dbhz"]:
        raise RuntimeError(
            "Contradiction: HGB is not lower under the persisted pooled validation criterion"
        )

    predictions = pd.read_csv(PREDICTION_PATH)
    required = {
        "observed_trend_dbhz",
        "pred_hgb_dbhz",
        "pred_lightgbm_dbhz",
        "bootstrap_block",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise RuntimeError(f"Persisted algorithm predictions are missing {sorted(missing)}")
    observed = predictions["observed_trend_dbhz"].to_numpy(float)
    hgb = predictions["pred_hgb_dbhz"].to_numpy(float)
    lightgbm = predictions["pred_lightgbm_dbhz"].to_numpy(float)
    hgb_rmse = rmse(observed, hgb)
    lightgbm_rmse = rmse(observed, lightgbm)
    if not np.isclose(hgb_rmse, pair.loc["HGB", "validation_rmse_dbhz"], atol=1e-12):
        raise RuntimeError("Persisted HGB predictions do not reproduce the benchmark table")
    if not np.isclose(
        lightgbm_rmse,
        pair.loc["LightGBM", "validation_rmse_dbhz"],
        atol=1e-12,
    ):
        raise RuntimeError("Persisted LightGBM predictions do not reproduce the benchmark table")

    block_indices = {
        block: group.index.to_numpy()
        for block, group in predictions.groupby("bootstrap_block", sort=False)
    }
    blocks = np.array(list(block_indices), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_delta = np.empty(BOOTSTRAP_REPEATS, dtype=float)
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        indices = np.concatenate([block_indices[block] for block in sampled_blocks])
        bootstrap_delta[repeat] = rmse(observed[indices], hgb[indices]) - rmse(
            observed[indices],
            lightgbm[indices],
        )

    delta = hgb_rmse - lightgbm_rmse
    ci_low, median, ci_high = np.percentile(bootstrap_delta, [2.5, 50.0, 97.5])
    comparison = pd.DataFrame(
        [
            {
                "predefined_family_selection_metric": "pooled validation RMSE",
                "validation_n": len(predictions),
                "bootstrap_block_minutes": 15,
                "bootstrap_block_count": len(blocks),
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "hgb_rmse_dbhz": hgb_rmse,
                "lightgbm_rmse_dbhz": lightgbm_rmse,
                "delta_hgb_minus_lightgbm_dbhz": delta,
                "paired_delta_ci95_low_dbhz": ci_low,
                "paired_delta_bootstrap_median_dbhz": median,
                "paired_delta_ci95_high_dbhz": ci_high,
                "bootstrap_probability_delta_below_zero": float(
                    np.mean(bootstrap_delta < 0)
                ),
                "selection_from_point_estimate": "HGB",
                "paired_ci_includes_zero": bool(ci_low <= 0 <= ci_high),
            }
        ]
    )

    metric_matrix = pd.DataFrame(
        {
            "algorithm": ["HGB", "LightGBM"],
            "overall_validation_rmse_dbhz": pair["validation_rmse_dbhz"].to_numpy(float),
            "rmse_phase_C_dbhz": pair["rmse_phase_C_dbhz"].to_numpy(float),
            "rmse_phase_T_dbhz": pair["rmse_phase_T_dbhz"].to_numpy(float),
            "rmse_phase_L_dbhz": pair["rmse_phase_L_dbhz"].to_numpy(float),
            "rmse_phase_S_dbhz": pair["rmse_phase_S_dbhz"].to_numpy(float),
            "phase_balanced_rmse_dbhz": pair["phase_balanced_rmse_dbhz"].to_numpy(float),
            "worst_phase_rmse_dbhz": pair["worst_phase_rmse_dbhz"].to_numpy(float),
        }
    )
    metric_columns = [
        "overall_validation_rmse_dbhz",
        "rmse_phase_C_dbhz",
        "rmse_phase_T_dbhz",
        "rmse_phase_L_dbhz",
        "rmse_phase_S_dbhz",
        "phase_balanced_rmse_dbhz",
    ]
    for column in metric_columns:
        best = float(metric_matrix[column].min())
        metric_matrix[f"{column}_is_lower"] = np.isclose(
            metric_matrix[column],
            best,
            atol=1e-12,
            rtol=0.0,
        )

    audit = {
        "criterion": "pooled validation RMSE point estimate",
        "hgb_rmse_dbhz": hgb_rmse,
        "lightgbm_rmse_dbhz": lightgbm_rmse,
        "delta_hgb_minus_lightgbm_dbhz": delta,
        "paired_delta_ci95_dbhz": [float(ci_low), float(ci_high)],
        "paired_ci_includes_zero": bool(ci_low <= 0 <= ci_high),
        "phase_balanced_hgb_dbhz": float(pair.loc["HGB", "phase_balanced_rmse_dbhz"]),
        "phase_balanced_lightgbm_dbhz": float(
            pair.loc["LightGBM", "phase_balanced_rmse_dbhz"]
        ),
        "worst_phase_hgb_dbhz": float(pair.loc["HGB", "worst_phase_rmse_dbhz"]),
        "worst_phase_lightgbm_dbhz": float(
            pair.loc["LightGBM", "worst_phase_rmse_dbhz"]
        ),
    }
    return comparison, metric_matrix, audit


def format_parameter_value(parameter: str, value: float) -> str:
    if parameter in {"max_iter", "max_leaf_nodes", "min_samples_leaf"}:
        return str(int(round(value)))
    if parameter == "learning_rate":
        return f"{value:.3f}".lstrip("0")
    if parameter == "l2_regularization":
        return f"{value:.1f}"
    raise KeyError(parameter)


def prepare_hyperparameter_evidence(
    candidates: pd.DataFrame,
    selection: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    combinations = candidates[
        ["parameter_set", *PARAMETER_COLUMNS]
    ].drop_duplicates()
    counts = combinations.groupby("parameter_set").size()
    if len(combinations) != 6 or not counts.eq(1).all():
        raise RuntimeError("Expected exactly six persisted joint HGB parameter combinations")
    combinations = combinations.set_index("parameter_set").reindex(PARAMETER_ORDER).reset_index()

    selected_parameters = selection["selected_parameters"]
    selected_mask = np.ones(len(combinations), dtype=bool)
    for column in PARAMETER_COLUMNS:
        selected_mask &= np.isclose(
            combinations[column].to_numpy(float),
            float(selected_parameters[column]),
            atol=1e-12,
            rtol=0.0,
        )
    if int(selected_mask.sum()) != 1:
        raise RuntimeError("Selected parameter combination is not unique in the candidate table")
    selected_parameter_set = str(combinations.loc[selected_mask, "parameter_set"].iloc[0])

    source = candidates.copy()
    source["feature_set_best_phase_balanced_rmse_dbhz"] = source.groupby("feature_set")[
        "phase_balanced_rmse_dbhz"
    ].transform("min")
    source["delta_rmse_dbhz"] = (
        source["phase_balanced_rmse_dbhz"]
        - source["feature_set_best_phase_balanced_rmse_dbhz"]
    )
    source["is_feature_set_best"] = np.isclose(
        source["delta_rmse_dbhz"],
        0.0,
        atol=1e-12,
        rtol=0.0,
    )
    source["is_selected_parameter_combination"] = source["parameter_set"].eq(
        selected_parameter_set
    )
    parameter_rank = {name: index for index, name in enumerate(PARAMETER_ORDER)}
    feature_rank = {name: index for index, name in enumerate(FEATURE_ORDER)}
    source["_parameter_rank"] = source["parameter_set"].map(parameter_rank)
    source["_feature_rank"] = source["feature_set"].map(feature_rank)
    source = source.sort_values(["_parameter_rank", "_feature_rank"]).drop(
        columns=["_parameter_rank", "_feature_rank"]
    )

    combinations["parameter_label"] = combinations.apply(
        lambda row: " / ".join(
            [
                format_parameter_value("max_iter", float(row["max_iter"])),
                format_parameter_value(
                    "learning_rate", float(row["learning_rate"])
                ),
                format_parameter_value(
                    "max_leaf_nodes", float(row["max_leaf_nodes"])
                ),
                format_parameter_value(
                    "min_samples_leaf", float(row["min_samples_leaf"])
                ),
                format_parameter_value(
                    "l2_regularization", float(row["l2_regularization"])
                ),
            ]
        ),
        axis=1,
    )
    combinations["is_selected"] = combinations["parameter_set"].eq(selected_parameter_set)
    combinations["candidate_label"] = [
        (
            f"Candidate {index + 1} (selected)"
            if parameter_set == selected_parameter_set
            else f"Candidate {index + 1}"
        )
        for index, parameter_set in enumerate(combinations["parameter_set"])
    ]

    matrix = (
        source.pivot(
            index="parameter_set",
            columns="feature_set",
            values="delta_rmse_dbhz",
        )
        .reindex(index=PARAMETER_ORDER, columns=FEATURE_ORDER)
        .reset_index()
        .merge(
            combinations[
                [
                    "parameter_set",
                    "parameter_label",
                    "candidate_label",
                    "is_selected",
                ]
            ],
            on="parameter_set",
            how="left",
        )
    )
    matrix = matrix[
        [
            "parameter_set",
            "parameter_label",
            "candidate_label",
            "is_selected",
            *FEATURE_ORDER,
        ]
    ]

    table_rows: list[dict[str, Any]] = []
    display_names = {
        "max_iter": "Iterations",
        "learning_rate": "Learning rate",
        "max_leaf_nodes": "Max leaves",
        "min_samples_leaf": "Minimum leaf",
        "l2_regularization": "L2 regularization",
    }
    for column in PARAMETER_COLUMNS:
        values = sorted({float(value) for value in combinations[column]})
        table_rows.append(
            {
                "component": "HGB",
                "parameter": display_names[column],
                "candidate_values_across_six_joint_combinations": ", ".join(
                    format_parameter_value(column, value) for value in values
                ),
                "selected_value": format_parameter_value(
                    column, float(selected_parameters[column])
                ),
                "hyperparameter_search_performed": True,
            }
        )
    table_rows.extend(
        [
            {
                "component": "Robust L1",
                "parameter": "Objective",
                "candidate_values_across_six_joint_combinations": "not searched",
                "selected_value": "absolute-error (L1)",
                "hyperparameter_search_performed": False,
            },
            {
                "component": "Robust L1",
                "parameter": "Offsets",
                "candidate_values_across_six_joint_combinations": "not searched",
                "selected_value": "4 constellation-band beta_s",
                "hyperparameter_search_performed": False,
            },
        ]
    )
    parameter_table = pd.DataFrame(table_rows)
    return source, matrix, parameter_table, selected_parameter_set


def prepare_v4_candidate_evidence(
    candidates: pd.DataFrame,
    selection: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Build the v4 robustness summary and the six-row parameter matrix."""
    combinations = (
        candidates[["parameter_set", *PARAMETER_COLUMNS]]
        .drop_duplicates()
        .set_index("parameter_set")
        .reindex(PARAMETER_ORDER)
        .reset_index()
    )
    if len(combinations) != 6 or combinations[PARAMETER_COLUMNS].isna().any().any():
        raise RuntimeError("Expected one complete persisted parameter row for p0-p5")

    selected_parameters = selection["selected_parameters"]
    selected_mask = np.ones(len(combinations), dtype=bool)
    for column in PARAMETER_COLUMNS:
        selected_mask &= np.isclose(
            combinations[column].to_numpy(float),
            float(selected_parameters[column]),
            atol=1e-12,
            rtol=0.0,
        )
    if int(selected_mask.sum()) != 1:
        raise RuntimeError("Selected parameter combination is not unique")
    selected_parameter_set = str(
        combinations.loc[selected_mask, "parameter_set"].iloc[0]
    )

    long_source = candidates.copy()
    long_source["feature_set_best_phase_balanced_rmse_dbhz"] = (
        long_source.groupby("feature_set")["phase_balanced_rmse_dbhz"].transform("min")
    )
    long_source["excess_phase_balanced_rmse_dbhz"] = (
        long_source["phase_balanced_rmse_dbhz"]
        - long_source["feature_set_best_phase_balanced_rmse_dbhz"]
    )
    long_source["is_feature_set_best"] = np.isclose(
        long_source["excess_phase_balanced_rmse_dbhz"],
        0.0,
        atol=1e-12,
        rtol=0.0,
    )
    parameter_rank = {name: index for index, name in enumerate(PARAMETER_ORDER)}
    feature_rank = {name: index for index, name in enumerate(FEATURE_ORDER)}
    long_source["_parameter_rank"] = long_source["parameter_set"].map(parameter_rank)
    long_source["_feature_rank"] = long_source["feature_set"].map(feature_rank)
    long_source = long_source.sort_values(
        ["_parameter_rank", "_feature_rank"]
    ).drop(columns=["_parameter_rank", "_feature_rank"])

    robustness_summary = (
        long_source.groupby("parameter_set", sort=False)[
            "excess_phase_balanced_rmse_dbhz"
        ]
        .agg(
            feature_definition_count="size",
            mean_excess_rmse_dbhz="mean",
            min_excess_rmse_dbhz="min",
            max_excess_rmse_dbhz="max",
        )
        .reindex(PARAMETER_ORDER)
        .reset_index()
    )
    best_counts = (
        long_source.groupby("parameter_set", sort=False)["is_feature_set_best"]
        .sum()
        .reindex(PARAMETER_ORDER)
        .astype(int)
    )
    robustness_summary["feature_definitions_at_optimum"] = (
        robustness_summary["parameter_set"].map(best_counts)
    )
    robustness_summary["candidate"] = [
        f"Candidate {index + 1}" for index in range(len(robustness_summary))
    ]
    robustness_summary["selected"] = robustness_summary["parameter_set"].eq(
        selected_parameter_set
    )
    robustness_summary = robustness_summary[
        [
            "candidate",
            "parameter_set",
            "selected",
            "feature_definition_count",
            "feature_definitions_at_optimum",
            "mean_excess_rmse_dbhz",
            "min_excess_rmse_dbhz",
            "max_excess_rmse_dbhz",
        ]
    ]

    parameter_matrix = combinations.copy()
    parameter_matrix.insert(
        0,
        "candidate",
        [f"Candidate {index + 1}" for index in range(len(parameter_matrix))],
    )
    parameter_matrix["selected"] = parameter_matrix["parameter_set"].eq(
        selected_parameter_set
    )
    parameter_matrix = parameter_matrix[
        ["candidate", "parameter_set", "selected", *PARAMETER_COLUMNS]
    ]

    selected_summary = robustness_summary[
        robustness_summary["selected"]
    ].iloc[0]
    if (
        int(selected_summary["feature_definitions_at_optimum"]) != 3
        or not np.isclose(
            float(selected_summary["mean_excess_rmse_dbhz"]),
            0.0017110349,
            atol=5e-10,
        )
        or not np.isclose(
            float(selected_summary["max_excess_rmse_dbhz"]),
            0.0068441397,
            atol=5e-10,
        )
    ):
        raise RuntimeError("Persisted Candidate 6 robustness values changed")
    return long_source, robustness_summary, parameter_matrix, selected_parameter_set


def prepare_panel_f_grouped_data(candidates: pd.DataFrame) -> pd.DataFrame:
    """Map the persisted 6 x 4 search table to grouped plotting coordinates."""
    required = {
        "feature_set",
        "parameter_set",
        "phase_balanced_rmse_dbhz",
        "selected",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise RuntimeError(f"Panel f candidate data are missing {sorted(missing)}")

    source = candidates.copy()
    feature_rank = {
        feature_set: index
        for index, feature_set in enumerate(PANEL_F_FEATURE_ORDER)
    }
    parameter_rank = {
        parameter_set: index
        for index, parameter_set in enumerate(PARAMETER_ORDER)
    }
    source["feature_group_index"] = source["feature_set"].map(feature_rank)
    source["candidate_index"] = source["parameter_set"].map(parameter_rank)
    if source[["feature_group_index", "candidate_index"]].isna().any().any():
        raise RuntimeError("Panel f contains an unknown feature or parameter-set identifier")

    offsets = np.linspace(-0.10, 0.10, len(PARAMETER_ORDER))
    source["candidate"] = source["candidate_index"].astype(int).map(
        lambda index: f"Candidate {index + 1}"
    )
    source["feature_display_label"] = source["feature_set"].map(
        {
            key: value.replace("\n", " ")
            for key, value in PANEL_F_FEATURE_LABELS.items()
        }
    )
    source["candidate_x_offset"] = source["candidate_index"].astype(int).map(
        dict(enumerate(offsets))
    )
    source["plot_x"] = (
        source["feature_group_index"].astype(float)
        + source["candidate_x_offset"].astype(float)
    )
    source["is_candidate_6"] = source["parameter_set"].eq("p5")
    source["is_selected_combination"] = (
        source["selected"].astype(str).str.lower().eq("true")
    )
    source = source.sort_values(
        ["feature_group_index", "candidate_index"]
    ).reset_index(drop=True)

    counts = source.groupby("feature_set").size().reindex(PANEL_F_FEATURE_ORDER)
    selected = source[source["is_selected_combination"]]
    if len(source) != 24 or not counts.eq(6).all():
        raise RuntimeError("Panel f must contain six candidates in each of four feature sets")
    if (
        len(selected) != 1
        or selected["feature_set"].iloc[0] != "stable_compact"
        or selected["parameter_set"].iloc[0] != "p5"
    ):
        raise RuntimeError(
            "The persisted selected combination is not stable_compact + Candidate 6"
        )
    if not np.isfinite(source["phase_balanced_rmse_dbhz"]).all():
        raise RuntimeError("Panel f contains a non-finite phase-balanced RMSE")
    return source


def draw_panel_b(
    ax: plt.Axes,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    ordered = benchmark.set_index("algorithm").reindex(ALGORITHM_ORDER).reset_index()
    required = [
        "validation_rmse_dbhz",
        "validation_rmse_ci95_low_dbhz",
        "validation_rmse_ci95_high_dbhz",
        "worst_phase_rmse_dbhz",
    ]
    if ordered[required].isna().any().any():
        raise RuntimeError("The complete seven-algorithm validation screen is incomplete")

    y = np.arange(len(ordered))
    low = (
        ordered["validation_rmse_dbhz"]
        - ordered["validation_rmse_ci95_low_dbhz"]
    )
    high = (
        ordered["validation_rmse_ci95_high_dbhz"]
        - ordered["validation_rmse_dbhz"]
    )
    ax.errorbar(
        ordered["validation_rmse_dbhz"],
        y,
        xerr=np.vstack([low, high]),
        fmt="none",
        ecolor=COL["physics_light"],
        elinewidth=1.45,
        capsize=1.8,
        zorder=1,
    )
    fill = [
        COL["ai"] if algorithm == "HGB" else COL["physics_light"]
        for algorithm in ordered["algorithm"]
    ]
    edge = [
        COL["ai"] if algorithm == "HGB" else COL["physics"]
        for algorithm in ordered["algorithm"]
    ]
    ax.scatter(
        ordered["validation_rmse_dbhz"],
        y,
        s=30,
        color=fill,
        edgecolor=edge,
        linewidth=0.80,
        zorder=3,
    )
    ax.scatter(
        ordered["worst_phase_rmse_dbhz"],
        y,
        s=22,
        marker="|",
        color=edge,
        linewidth=1.25,
        zorder=3,
    )
    ax.set_yticks(y, ordered["algorithm"])
    ax.invert_yaxis()
    ax.set_xlabel("Validation trend RMSE (dB-Hz)")
    ax.set_title("Trend-regressor validation screen", loc="left", pad=4.0)
    ax.annotate(
        "selected",
        (
            float(ordered.loc[ordered["algorithm"].eq("HGB"), "validation_rmse_dbhz"].iloc[0]),
            0,
        ),
        xytext=(5, -10),
        textcoords="offset points",
        fontsize=5.1,
        color=COL["ai"],
    )
    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=COL["physics_light"],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=COL["physics"],
            lw=1.2,
            ms=3.4,
            label="Pooled RMSE (95% CI)",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=COL["physics"],
            marker="|",
            linestyle="none",
            ms=5.2,
            markeredgewidth=1.1,
            label="Worst phase",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.985),
        frameon=False,
        fontsize=4.65,
        handlelength=1.35,
        handletextpad=0.35,
        labelspacing=0.18,
        borderaxespad=0.0,
    )
    ax.margins(x=0.055, y=0.08)
    v2.panel_label(ax, "b", x=-0.23)
    return ordered


def draw_panel_c(ax: plt.Axes, ordered: pd.DataFrame) -> None:
    phases = ["C", "T", "L", "S"]
    heat = ordered.set_index("algorithm").reindex(ALGORITHM_ORDER)[
        [f"rmse_phase_{phase}_dbhz" for phase in phases]
    ].to_numpy(float)
    if not np.isfinite(heat).all():
        raise RuntimeError("The seven-algorithm phase RMSE matrix is incomplete")
    heat_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rmse_seq_compact_v4",
        ["#F7F6F3", "#E8D6B2", COL["beta"], "#C66D3D", COL["ai"], "#722A3D"],
    )
    upper = float(np.nanpercentile(heat, 95))
    ax.imshow(
        heat,
        aspect="auto",
        cmap=heat_cmap,
        vmin=float(np.nanmin(heat)),
        vmax=upper,
    )
    ax.set_xticks(np.arange(4), phases)
    ax.set_yticks(np.arange(len(ALGORITHM_ORDER)), ALGORITHM_ORDER)
    threshold = float(np.nanpercentile(heat, 70))
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            value = heat[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.15,
                color="white" if value > threshold else COL["ink"],
            )
    hgb_row = ALGORITHM_ORDER.index("HGB")
    ax.add_patch(
        Rectangle(
            (-0.5, hgb_row - 0.5),
            4.0,
            1.0,
            fill=False,
            edgecolor=COL["ai"],
            linewidth=1.05,
            clip_on=False,
        )
    )
    ax.get_yticklabels()[hgb_row].set_color(COL["ai"])
    ax.get_yticklabels()[hgb_row].set_fontweight("bold")
    ax.set_title("Phase-resolved validation RMSE", loc="left", pad=4.0)
    ax.tick_params(length=0)
    v2.panel_label(ax, "c", x=-0.30)


def draw_panel_e(
    ax: plt.Axes,
    robustness_summary: pd.DataFrame,
) -> None:
    ordered = robustness_summary.set_index("parameter_set").reindex(
        PARAMETER_ORDER
    ).reset_index()
    x = np.arange(1, len(ordered) + 1)
    mean = ordered["mean_excess_rmse_dbhz"].to_numpy(float)
    lower = ordered["min_excess_rmse_dbhz"].to_numpy(float)
    upper = ordered["max_excess_rmse_dbhz"].to_numpy(float)
    selected = ordered["selected"].to_numpy(bool)

    for xi, mean_value, low_value, high_value, is_selected in zip(
        x, mean, lower, upper, selected
    ):
        color = COL["ai"] if is_selected else COL["physics"]
        ax.vlines(
            xi,
            low_value,
            high_value,
            color=color,
            lw=1.35 if is_selected else 1.0,
            zorder=2,
        )
        ax.hlines(
            [low_value, high_value],
            xi - 0.10,
            xi + 0.10,
            color=color,
            lw=0.9,
            zorder=2,
        )
        ax.scatter(
            [xi],
            [mean_value],
            s=31 if is_selected else 24,
            color=color if is_selected else COL["physics_light"],
            edgecolor=color,
            linewidth=0.75,
            zorder=3,
        )

    selected_index = int(np.flatnonzero(selected)[0])
    ax.annotate(
        "selected",
        (x[selected_index], mean[selected_index]),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=5.1,
        color=COL["ai"],
    )
    ax.axhline(0, color=COL["grid"], lw=0.65, zorder=0)
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Excess phase-balanced\nRMSE (dB-Hz)")
    ax.set_xlim(0.55, 6.45)
    y_max = float(np.nanmax(upper))
    ax.set_ylim(-0.0045, y_max * 1.12)
    tick_step = 0.03
    ax.set_yticks(
        np.arange(0.0, y_max * 1.12 + tick_step * 0.25, tick_step)
    )
    ax.set_title("Hyperparameter robustness", loc="left", pad=4.0)
    ax.text(
        0.03,
        0.96,
        "mean and min-max across four feature definitions; lower is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.25,
        color=COL["muted"],
    )
    v2.panel_label(ax, "e", x=-0.27)


def draw_panel_d_table(ax: plt.Axes, parameter_matrix: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_title("Candidate parameter combinations", loc="left", pad=4.0)
    v2.panel_label(ax, "d", x=-0.22, y=1.075)

    ordered = parameter_matrix.set_index("parameter_set").reindex(
        PARAMETER_ORDER
    ).reset_index()
    columns = [
        ("Candidate", "candidate"),
        ("Iteration\ncount", "max_iter"),
        ("Learning\nrate", "learning_rate"),
        ("Max\nleaves", "max_leaf_nodes"),
        ("Min.\nleaf", "min_samples_leaf"),
        ("L2", "l2_regularization"),
    ]
    x_positions = [0.01, 0.345, 0.525, 0.680, 0.815, 0.945]
    alignments = ["left", "center", "center", "center", "center", "center"]
    header_y = 0.865
    row_y = np.linspace(0.735, 0.285, len(ordered))

    ax.plot([0.00, 1.00], [0.945, 0.945], transform=ax.transAxes, color=COL["ink"], lw=0.72)
    for (label, _), xpos, alignment in zip(columns, x_positions, alignments):
        ax.text(
            xpos,
            header_y,
            label,
            transform=ax.transAxes,
            ha=alignment,
            va="center",
            fontsize=4.25,
            color=COL["muted"],
            fontweight="bold",
            linespacing=0.92,
        )
    ax.plot([0.00, 1.00], [0.795, 0.795], transform=ax.transAxes, color=COL["grid"], lw=0.58)

    selected_index = int(np.flatnonzero(ordered["selected"].to_numpy(bool))[0])
    selected_y = float(row_y[selected_index])
    row_half_height = 0.041
    ax.add_patch(
        Rectangle(
            (0.0, selected_y - row_half_height),
            1.0,
            row_half_height * 2,
            transform=ax.transAxes,
            facecolor=COL["ai_light"],
            edgecolor=COL["ai"],
            linewidth=0.72,
            alpha=0.58,
            zorder=0,
        )
    )

    for ypos, row in zip(row_y, ordered.itertuples(index=False)):
        values = [
            row.candidate,
            format_parameter_value("max_iter", float(row.max_iter)),
            format_parameter_value("learning_rate", float(row.learning_rate)),
            format_parameter_value("max_leaf_nodes", float(row.max_leaf_nodes)),
            format_parameter_value("min_samples_leaf", float(row.min_samples_leaf)),
            format_parameter_value("l2_regularization", float(row.l2_regularization)),
        ]
        for value, xpos, alignment in zip(values, x_positions, alignments):
            ax.text(
                xpos,
                ypos,
                value,
                transform=ax.transAxes,
                ha=alignment,
                va="center",
                fontsize=4.50,
                color=COL["ai"] if row.selected else COL["ink"],
                fontweight="bold" if row.selected else "normal",
                zorder=2,
            )
    ax.plot([0.00, 1.00], [0.215, 0.215], transform=ax.transAxes, color=COL["ink"], lw=0.72)
    ax.text(
        0.00,
        0.105,
        "Robust L1: absolute-error objective; four constellation-band offsets.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.20,
        color=COL["muted"],
    )


def draw_panel_f_grouped(
    ax: plt.Axes,
    panel_f_source: pd.DataFrame,
) -> None:
    """Show all persisted parameter-feature combinations without a second metric."""
    ordinary = panel_f_source[~panel_f_source["is_candidate_6"]]
    candidate_6 = panel_f_source[panel_f_source["is_candidate_6"]]
    selected = panel_f_source[panel_f_source["is_selected_combination"]].iloc[0]

    ax.scatter(
        ordinary["plot_x"],
        ordinary["phase_balanced_rmse_dbhz"],
        s=22,
        color=COL["physics_light"],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
        zorder=2,
    )
    ax.scatter(
        candidate_6["plot_x"],
        candidate_6["phase_balanced_rmse_dbhz"],
        s=34,
        color=COL["ai"],
        edgecolor="white",
        linewidth=0.50,
        zorder=3,
    )
    ax.scatter(
        [float(selected["plot_x"])],
        [float(selected["phase_balanced_rmse_dbhz"])],
        s=68,
        facecolor="none",
        edgecolor=COL["ink"],
        linewidth=1.15,
        zorder=4,
    )
    ax.annotate(
        "selected",
        (
            float(selected["plot_x"]),
            float(selected["phase_balanced_rmse_dbhz"]),
        ),
        xytext=(6, 7),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=COL["ink"],
    )

    x = np.arange(len(PANEL_F_FEATURE_ORDER))
    ax.set_xticks(
        x,
        [PANEL_F_FEATURE_LABELS[name] for name in PANEL_F_FEATURE_ORDER],
    )
    ax.set_xlim(-0.34, 3.34)
    values = panel_f_source["phase_balanced_rmse_dbhz"].to_numpy(float)
    ax.set_ylim(float(values.min()) - 0.009, float(values.max()) + 0.035)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.set_ylabel("Phase-balanced validation\nRMSE (dB-Hz)")
    ax.set_title(
        "Parameter-feature\nselection",
        loc="left",
        pad=3.0,
        linespacing=0.95,
    )
    ax.tick_params(axis="x", labelsize=4.8, pad=2.5)
    ax.text(
        0.02,
        0.97,
        "lower is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.45,
        color=COL["muted"],
    )
    ax.grid(axis="y", color=COL["grid"], lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COL["physics_light"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=4.0,
            label="Candidates 1-5",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COL["ai"],
            markeredgecolor="white",
            markeredgewidth=0.40,
            markersize=4.4,
            label="Candidate 6",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COL["ai"],
            markeredgecolor=COL["ink"],
            markeredgewidth=1.0,
            markersize=5.0,
            label="Selected combination",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.985),
        frameon=False,
        fontsize=4.45,
        handletextpad=0.30,
        labelspacing=0.18,
        borderaxespad=0.0,
    )
    v2.panel_label(ax, "f", x=-0.25)


def render_figure(
    fixed: pd.DataFrame,
    benchmark: pd.DataFrame,
    candidates: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    parameter_matrix: pd.DataFrame,
    panel_f_source: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, Any]]:
    set_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=EXPORT_DPI)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.00, 1.00, 1.00],
        height_ratios=[1.00, 1.00],
        left=0.077,
        right=0.994,
        top=0.955,
        bottom=0.112,
        wspace=0.360,
        hspace=0.315,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])
    panel_axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]

    v2.draw_panel_a(ax_a, fixed)
    ordered_algorithms = draw_panel_b(ax_b, benchmark)
    draw_panel_c(ax_c, ordered_algorithms)
    draw_panel_d_table(ax_d, parameter_matrix)
    draw_panel_e(ax_e, robustness_summary)
    draw_panel_f_grouped(ax_f, panel_f_source)
    for ax in [ax_a, ax_b, ax_c, ax_e, ax_f]:
        ax.tick_params(direction="out")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    out_of_canvas = []
    for artist in fig.findobj(match=mpl.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if (
            bbox.x0 < figure_bbox.x0 - 0.5
            or bbox.y0 < figure_bbox.y0 - 0.5
            or bbox.x1 > figure_bbox.x1 + 0.5
            or bbox.y1 > figure_bbox.y1 + 0.5
        ):
            out_of_canvas.append(
                {
                    "text": artist.get_text(),
                    "bbox_px": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                }
            )

    label_title_overlaps = []
    for ax, label in zip(panel_axes, "abcdef"):
        title_bbox = ax.title.get_window_extent(renderer=renderer)
        label_artist = next(
            artist
            for artist in ax.texts
            if artist.get_text() == label and artist.get_fontweight() == "bold"
        )
        if title_bbox.overlaps(label_artist.get_window_extent(renderer=renderer)):
            label_title_overlaps.append(label)

    row_gaps_px = []
    for top_ax, bottom_ax in zip(panel_axes[:3], panel_axes[3:]):
        top_lower_bboxes = [
            label.get_window_extent(renderer=renderer)
            for label in top_ax.get_xticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        if top_ax.xaxis.label.get_text().strip():
            top_lower_bboxes.append(top_ax.xaxis.label.get_window_extent(renderer=renderer))
        if top_lower_bboxes:
            top_row_lower_y = min(bbox.y0 for bbox in top_lower_bboxes)
            bottom_title_upper_y = bottom_ax.title.get_window_extent(renderer=renderer).y1
            row_gaps_px.append(float(top_row_lower_y - bottom_title_upper_y))

    panel_d_texts = [
        artist
        for artist in ax_d.texts
        if artist.get_visible() and artist.get_text().strip() and artist.get_text() != "d"
    ]
    panel_d_overlaps = []
    for index, first in enumerate(panel_d_texts):
        first_bbox = first.get_window_extent(renderer=renderer)
        for second in panel_d_texts[index + 1 :]:
            if first_bbox.overlaps(second.get_window_extent(renderer=renderer)):
                panel_d_overlaps.append([first.get_text(), second.get_text()])

    panel_f_xtick_bboxes = [
        label.get_window_extent(renderer=renderer)
        for label in ax_f.get_xticklabels()
        if label.get_visible() and label.get_text().strip()
    ]
    panel_f_xtick_overlaps = 0
    for index, first_bbox in enumerate(panel_f_xtick_bboxes):
        for second_bbox in panel_f_xtick_bboxes[index + 1 :]:
            panel_f_xtick_overlaps += int(first_bbox.overlaps(second_bbox))
    panel_f_legend = ax_f.get_legend()
    panel_f_legend_title_overlap = bool(
        panel_f_legend
        and panel_f_legend.get_window_extent(renderer=renderer).overlaps(
            ax_f.title.get_window_extent(renderer=renderer)
        )
    )
    panel_f_internal_label_literals = [
        artist.get_text()
        for artist in ax_f.findobj(match=mpl.text.Text)
        if "GRAP" in artist.get_text().upper()
    ]

    panel_bounds_px = {
        label: [float(value) for value in ax.get_window_extent(renderer=renderer).bounds]
        for label, ax in zip("abcdef", panel_axes)
    }
    panel_widths_px = [bounds[2] for bounds in panel_bounds_px.values()]
    panel_heights_px = [bounds[3] for bounds in panel_bounds_px.values()]
    max_panel_width_delta_px = float(max(panel_widths_px) - min(panel_widths_px))
    max_panel_height_delta_px = float(max(panel_heights_px) - min(panel_heights_px))

    if out_of_canvas:
        raise RuntimeError(f"Text outside canvas: {out_of_canvas}")
    if label_title_overlaps:
        raise RuntimeError(f"Panel labels overlap titles: {label_title_overlaps}")
    if row_gaps_px and min(row_gaps_px) < 0:
        raise RuntimeError(f"Inter-row text overlap: {row_gaps_px}")
    if panel_d_overlaps:
        raise RuntimeError(f"Panel d table text overlap: {panel_d_overlaps}")
    if panel_f_xtick_overlaps:
        raise RuntimeError("Panel f feature labels overlap")
    if panel_f_legend_title_overlap:
        raise RuntimeError("Panel f legend overlaps its title")
    if panel_f_internal_label_literals:
        raise RuntimeError(
            f"Panel f exposes internal feature labels: {panel_f_internal_label_literals}"
        )
    if max_panel_width_delta_px > 0.5 or max_panel_height_delta_px > 0.5:
        raise RuntimeError(
            "Panel plotting bboxes are not equal: "
            f"width delta={max_panel_width_delta_px}, "
            f"height delta={max_panel_height_delta_px}"
        )

    qa = {
        "figure_size_inches": [FIG_WIDTH_IN, FIG_HEIGHT_IN],
        "target_width_mm": FIG_WIDTH_MM,
        "dpi": EXPORT_DPI,
        "gridspec": {
            "left": 0.077,
            "right": 0.994,
            "top": 0.955,
            "bottom": 0.112,
            "wspace": 0.360,
            "hspace": 0.315,
        },
        "panel_count": 6,
        "matplotlib_axes_count": len(fig.axes),
        "panel_b_algorithm_count": int(len(ordered_algorithms)),
        "panel_c_heatmap_shape": [int(len(ordered_algorithms)), 4],
        "panel_e_candidate_count": int(len(robustness_summary)),
        "panel_d_candidate_row_count": int(len(parameter_matrix)),
        "panel_d_parameter_column_count": int(len(PARAMETER_COLUMNS)),
        "panel_f_combination_count": int(len(panel_f_source)),
        "panel_f_feature_group_count": int(
            panel_f_source["feature_set"].nunique()
        ),
        "panel_f_candidates_per_group": [
            int(value)
            for value in panel_f_source.groupby("feature_set")
            .size()
            .reindex(PANEL_F_FEATURE_ORDER)
            .tolist()
        ],
        "panel_f_selected_combination_count": int(
            panel_f_source["is_selected_combination"].sum()
        ),
        "panel_f_xtick_label_overlap_count": panel_f_xtick_overlaps,
        "panel_f_legend_title_overlap": panel_f_legend_title_overlap,
        "panel_f_internal_label_literal_count": len(
            panel_f_internal_label_literals
        ),
        "panel_plotting_bboxes_px": panel_bounds_px,
        "max_panel_width_delta_px": max_panel_width_delta_px,
        "max_panel_height_delta_px": max_panel_height_delta_px,
        "text_outside_canvas_count": len(out_of_canvas),
        "panel_label_title_overlap_count": len(label_title_overlaps),
        "minimum_interrow_text_gap_px": min(row_gaps_px) if row_gaps_px else None,
        "panel_d_table_text_overlap_count": len(panel_d_overlaps),
    }
    return fig, qa


def save_outputs(fig: plt.Figure) -> dict[str, dict[str, Any]]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": FIG_DIR / f"{STEM}.png",
        "pdf": FIG_DIR / f"{STEM}.pdf",
        "svg": FIG_DIR / f"{STEM}.svg",
        "tiff": FIG_DIR / f"{STEM}.tiff",
    }
    fig.savefig(paths["png"], dpi=EXPORT_DPI, facecolor="white", edgecolor="white")
    fig.savefig(paths["pdf"], facecolor="white", edgecolor="white")
    fig.savefig(paths["svg"], facecolor="white", edgecolor="white")
    fig.savefig(
        paths["tiff"],
        dpi=EXPORT_DPI,
        facecolor="white",
        edgecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    audit: dict[str, dict[str, Any]] = {}
    for suffix, path in paths.items():
        audit[suffix] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    with Image.open(paths["png"]) as image:
        rgb = np.asarray(image.convert("RGB"))
        ink = np.any(rgb < 248, axis=2)
        rows, columns = np.where(ink)
        width, height = image.size
        audit["png"].update(
            {
                "pixel_dimensions": [width, height],
                "dpi_metadata": list(image.info.get("dpi", ())),
                "visible_ink_margins_px_at_rgb_lt_248": {
                    "left_px": int(columns.min()),
                    "right_px": int(width - 1 - columns.max()),
                    "top_px": int(rows.min()),
                    "bottom_px": int(height - 1 - rows.max()),
                },
            }
        )
        if width not in {4322, 4323}:
            raise RuntimeError(f"Unexpected 183-mm PNG width: {width}px")
    with Image.open(paths["tiff"]) as image:
        if image.info.get("compression") != "tiff_lzw":
            raise RuntimeError("TIFF is not LZW-compressed")
        audit["tiff"]["compression"] = image.info.get("compression")
        audit["tiff"]["dpi_metadata"] = [
            float(value) for value in image.info.get("dpi", ())
        ]
    svg_text = paths["svg"].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG text is not editable")
    audit["svg"]["editable_text_elements_present"] = True
    return audit


def audit_panels_a_to_e_unchanged(
    reference_png: Path,
    output_png: Path,
    qa: dict[str, Any],
) -> dict[str, Any]:
    if not reference_png.exists():
        raise RuntimeError(f"Missing compact-v5 reference image: {reference_png}")
    reference = Image.open(reference_png).convert("RGB")
    output = Image.open(output_png).convert("RGB")
    if reference.size != output.size:
        raise RuntimeError("compact-v5 and compact-v6 PNG dimensions differ")

    height = output.height
    audit: dict[str, Any] = {}
    for label in "abcde":
        x, y, width, panel_height = qa["panel_plotting_bboxes_px"][label]
        crop_box = (
            round(x),
            round(height - (y + panel_height)),
            round(x + width),
            round(height - y),
        )
        difference = ImageChops.difference(
            reference.crop(crop_box),
            output.crop(crop_box),
        )
        difference_bbox = difference.getbbox()
        audit[label] = {
            "crop_box_px": crop_box,
            "pixel_identical": difference_bbox is None,
            "difference_bbox": difference_bbox,
        }
    if not all(item["pixel_identical"] for item in audit.values()):
        raise RuntimeError(f"Panels a-e changed relative to compact-v5: {audit}")
    return audit


def write_caption_qa(
    comparison: pd.DataFrame,
    benchmark: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    qa: dict[str, Any],
    output_audit: dict[str, dict[str, Any]],
) -> None:
    row = comparison.iloc[0]
    selected_row = robustness_summary[robustness_summary["selected"]].iloc[0]
    selected_best_count = int(selected_row["feature_definitions_at_optimum"])
    selected_mean_delta = float(selected_row["mean_excess_rmse_dbhz"])
    selected_max_delta = float(selected_row["max_excess_rmse_dbhz"])
    pair = benchmark.set_index("algorithm")
    caption = (
        "**Caption draft.** Validation-only algorithm and hyperparameter selection for the "
        "two learned residual corrections. "
        "**a,** Comparison of persistent-correction estimators on validation data and four "
        "complete operations not used for training or tuning. **b,** Validation trend RMSE "
        "for all seven persisted regression algorithms; horizontal intervals are the saved "
        "95% confidence intervals and vertical ticks mark the worst-phase RMSE. HGB is "
        "highlighted because it was selected by the predefined pooled validation criterion "
        f"({row['hgb_rmse_dbhz']:.6f} dB-Hz versus "
        f"{row['lightgbm_rmse_dbhz']:.6f} dB-Hz for LightGBM). The paired HGB-minus-"
        f"LightGBM difference is {row['delta_hgb_minus_lightgbm_dbhz']:.6f} dB-Hz, with a "
        f"15-min block-bootstrap 95% CI of [{row['paired_delta_ci95_low_dbhz']:.6f}, "
        f"{row['paired_delta_ci95_high_dbhz']:.6f}] dB-Hz; this interval includes zero. "
        "**c,** Phase-resolved validation RMSE for the same seven algorithms. The outlined "
        "HGB row denotes the selected family, not phase-wise dominance: LightGBM is lower "
        f"in T ({pair.loc['LightGBM', 'rmse_phase_T_dbhz']:.3f} versus "
        f"{pair.loc['HGB', 'rmse_phase_T_dbhz']:.3f} dB-Hz) and L "
        f"({pair.loc['LightGBM', 'rmse_phase_L_dbhz']:.3f} versus "
        f"{pair.loc['HGB', 'rmse_phase_L_dbhz']:.3f} dB-Hz). **d,** The six persisted HGB "
        "parameter combinations, with Candidate 6 highlighted as selected. Robust L1 uses "
        "the absolute-error objective and four constellation-band offsets. **e,** Robustness of six joint "
        "parameter candidates across four feature definitions. Points are mean excess "
        "phase-balanced RMSE and ranges span the minimum to maximum, where excess is "
        "relative to the best setting within each feature definition. Candidate 6 is "
        f"optimal in {selected_best_count}/4 definitions, with mean and maximum excess of "
        f"{selected_mean_delta:.6f} and {selected_max_delta:.6f} dB-Hz. **f,** All 24 "
        "parameter-feature combinations shown as six candidates within each of four feature "
        "sets. Candidate 6 is magenta in every group, and the black-outlined point marks the "
        "selected Compact physical + Candidate 6 combination. The display labels map as "
        "follows: Compact physical (44) to `stable_compact`, Full physical (61) to `current`, "
        "Reduced uncertainty (53) to `no_grap_uncertainty`, and Direct budget (50) to "
        "`direct_budget_only`. "
        "Internal test data and the four complete operations were not used for selection.\n\n"
        "**QA note.** Panels a-e retain their compact-v5 content and layout. Panel f is a "
        "direct regrouping of the same persisted 24-row validation table; no model was fitted, "
        "updated or rescored.\n\n"
        f"- Canvas: {qa['figure_size_inches'][0]:.6f} x "
        f"{qa['figure_size_inches'][1]:.2f} in at {EXPORT_DPI} dpi\n"
        f"- PNG: {output_audit['png']['pixel_dimensions'][0]} x "
        f"{output_audit['png']['pixel_dimensions'][1]} px\n"
        f"- Text outside canvas: {qa['text_outside_canvas_count']}\n"
        f"- Panel-label/title overlaps: {qa['panel_label_title_overlap_count']}\n"
        f"- Panel-d table text overlaps: {qa['panel_d_table_text_overlap_count']}\n"
        f"- Panel b algorithms / panel c matrix: {qa['panel_b_algorithm_count']} / "
        f"{qa['panel_c_heatmap_shape']}\n"
        f"- Panel d rows x parameters / panel e candidates / panel f combinations: "
        f"{qa['panel_d_candidate_row_count']} x {qa['panel_d_parameter_column_count']} / "
        f"{qa['panel_e_candidate_count']} / {qa['panel_f_combination_count']}\n"
        f"- Panel f groups / candidates per group / selected combinations: "
        f"{qa['panel_f_feature_group_count']} / {qa['panel_f_candidates_per_group']} / "
        f"{qa['panel_f_selected_combination_count']}\n"
        f"- Panel f feature-label overlaps / legend-title overlap / internal-label literals: "
        f"{qa['panel_f_xtick_label_overlap_count']} / "
        f"{qa['panel_f_legend_title_overlap']} / "
        f"{qa['panel_f_internal_label_literal_count']}\n"
        f"- Panels a-e pixel-identical to compact-v5 plotting regions: "
        f"{all(item['pixel_identical'] for item in qa['panels_a_to_e_pixel_identity_vs_v5'].values())}\n"
        f"- Maximum panel width/height differences: "
        f"{qa['max_panel_width_delta_px']:.3f} / "
        f"{qa['max_panel_height_delta_px']:.3f} px\n"
        f"- Minimum inter-row text gap: {qa['minimum_interrow_text_gap_px']:.1f} px\n"
        f"- Visible-ink margins: "
        f"{json.dumps(output_audit['png']['visible_ink_margins_px_at_rgb_lt_248'])}\n"
    )
    CAPTION_QA_PATH.write_text(caption, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fixed, benchmark, candidates, selection = v2.load_inputs()
    comparison, _, algorithm_audit = prepare_algorithm_comparison(benchmark)
    (
        hyperparameter_long,
        robustness_summary,
        parameter_matrix,
        selected_parameter_set,
    ) = prepare_v4_candidate_evidence(candidates, selection)
    panel_f_source = prepare_panel_f_grouped_data(candidates)

    ordered_benchmark = benchmark.set_index("algorithm").reindex(
        ALGORITHM_ORDER
    ).reset_index()
    panel_c_columns = [
        "algorithm",
        "rmse_phase_C_dbhz",
        "rmse_phase_T_dbhz",
        "rmse_phase_L_dbhz",
        "rmse_phase_S_dbhz",
    ]
    fixed.to_csv(PANEL_A_SOURCE_PATH, index=False, encoding="utf-8-sig")
    ordered_benchmark.to_csv(PANEL_B_SOURCE_PATH, index=False, encoding="utf-8-sig")
    ordered_benchmark[panel_c_columns].to_csv(
        PANEL_C_SOURCE_PATH, index=False, encoding="utf-8-sig"
    )
    parameter_matrix.to_csv(PANEL_D_SOURCE_PATH, index=False, encoding="utf-8-sig")
    hyperparameter_long.to_csv(
        PANEL_E_LONG_SOURCE_PATH, index=False, encoding="utf-8-sig"
    )
    robustness_summary.to_csv(PANEL_E_SOURCE_PATH, index=False, encoding="utf-8-sig")
    panel_f_source.to_csv(PANEL_F_SOURCE_PATH, index=False, encoding="utf-8-sig")

    fig, qa = render_figure(
        fixed,
        benchmark,
        candidates,
        robustness_summary,
        parameter_matrix,
        panel_f_source,
    )
    output_audit = save_outputs(fig)
    plt.close(fig)
    v5_reference_png = (
        FIG_DIR / "FigAI3_algorithm_and_hyperparameter_selection_compact_v5.png"
    )
    qa["panels_a_to_e_pixel_identity_vs_v5"] = audit_panels_a_to_e_unchanged(
        v5_reference_png,
        Path(output_audit["png"]["path"]),
        qa,
    )
    write_caption_qa(
        comparison,
        benchmark,
        robustness_summary,
        qa,
        output_audit,
    )

    selected_row = robustness_summary[robustness_summary["selected"]].iloc[0]
    v3_path = FIG_DIR / "FigAI3_algorithm_and_hyperparameter_selection_compact_v3.png"
    v4_path = FIG_DIR / "FigAI3_algorithm_and_hyperparameter_selection_compact_v4.png"
    v5_path = v5_reference_png
    provenance = {
        "figure": STEM,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "no_retraining": True,
        "figure_contract": {
            "core_conclusion": (
                "HGB was selected by the lower pooled validation RMSE point estimate under "
                "the persisted family criterion, while paired uncertainty crosses zero and "
                "phase-resolved comparisons show mixed HGB-LightGBM performance."
            ),
            "algorithm_family_selection_metric": "pooled validation RMSE",
            "hgb_internal_tuning_metric": "phase-balanced validation RMSE",
            "claim_boundary": (
                "HGB is the selected persisted algorithm under the predefined pooled "
                "validation criterion, not a universally superior algorithm across phases."
            ),
            "archetype": "quantitative grid",
            "bottom_row_panel_order": [
                "d: candidate parameter combinations",
                "e: hyperparameter robustness",
                "f: grouped parameter-feature combinations",
            ],
            "panel_e_role": "hyperparameter robustness across feature definitions",
            "panel_f_role": (
                "show all six persisted parameter candidates within each of four "
                "feature definitions using phase-balanced validation RMSE only"
            ),
        },
        "algorithm_comparison": algorithm_audit,
        "algorithm_order": ALGORITHM_ORDER,
        "paired_bootstrap": {
            "prediction_source": str(PREDICTION_PATH),
            "block_minutes": 15,
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "model_refitting_per_repeat": False,
        },
        "hyperparameter_robustness": {
            "selected_parameter_set_internal_id": selected_parameter_set,
            "feature_definitions": FEATURE_ORDER,
            "feature_display_labels": {
                name: FEATURE_LABELS[name].replace("-\n", "-").replace("\n", " ")
                for name in FEATURE_ORDER
            },
            "feature_definition_audit": FEATURE_DEFINITIONS,
            "candidate_display_labels": {
                row.parameter_set: row.candidate
                for row in parameter_matrix.itertuples(index=False)
            },
            "selected_feature_sets_at_optimum": int(
                selected_row["feature_definitions_at_optimum"]
            ),
            "selected_mean_delta_rmse_dbhz": float(
                selected_row["mean_excess_rmse_dbhz"]
            ),
            "selected_max_delta_rmse_dbhz": float(
                selected_row["max_excess_rmse_dbhz"]
            ),
            "delta_definition": (
                "phase_balanced_rmse_dbhz minus the minimum within the same feature_set"
            ),
        },
        "panel_f_grouped_search": {
            "metric": "phase_balanced_rmse_dbhz",
            "direction": "lower is better",
            "feature_order": PANEL_F_FEATURE_ORDER,
            "display_to_source_mapping": {
                PANEL_F_FEATURE_LABELS[name].replace("\n", " "): name
                for name in PANEL_F_FEATURE_ORDER
            },
            "parameter_set_to_candidate_mapping": {
                parameter_set: f"Candidate {index + 1}"
                for index, parameter_set in enumerate(PARAMETER_ORDER)
            },
            "combination_count": int(len(panel_f_source)),
            "selected_feature_set": "stable_compact",
            "selected_parameter_set": "p5",
            "selected_display": "Compact physical (44) + Candidate 6",
            "worst_phase_metric_displayed": False,
        },
        "sources": {
            str(v2.FIXED_BENCHMARK_PATH): sha256(v2.FIXED_BENCHMARK_PATH),
            str(v2.ALGORITHM_BENCHMARK_PATH): sha256(v2.ALGORITHM_BENCHMARK_PATH),
            str(v2.CANDIDATE_PATH): sha256(v2.CANDIDATE_PATH),
            str(v2.SELECTION_PATH): sha256(v2.SELECTION_PATH),
            str(v2.MANIFEST_PATH): sha256(v2.MANIFEST_PATH),
            str(PREDICTION_PATH): sha256(PREDICTION_PATH),
            str(FIG_DIR / "FigAI3_algorithm_and_hyperparameter_selection_compact_v2.png"): sha256(
                FIG_DIR / "FigAI3_algorithm_and_hyperparameter_selection_compact_v2.png"
            ),
            **({str(v3_path): sha256(v3_path)} if v3_path.exists() else {}),
            **({str(v4_path): sha256(v4_path)} if v4_path.exists() else {}),
            **({str(v5_path): sha256(v5_path)} if v5_path.exists() else {}),
        },
        "panel_source_data": {
            "a": str(PANEL_A_SOURCE_PATH),
            "b": str(PANEL_B_SOURCE_PATH),
            "c": str(PANEL_C_SOURCE_PATH),
            "d": str(PANEL_D_SOURCE_PATH),
            "e_long": str(PANEL_E_LONG_SOURCE_PATH),
            "e_summary": str(PANEL_E_SOURCE_PATH),
            "f": str(PANEL_F_SOURCE_PATH),
        },
        "unchanged_panel_drawers": {
            "a": "revise_figai3_algorithm_selection_compact_v2.draw_panel_a",
        },
        "unchanged_from_compact_v5": ["a", "b", "c", "d", "e"],
        "qa": qa,
        "outputs": output_audit,
        "caption_qa_note": str(CAPTION_QA_PATH),
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(algorithm_audit, indent=2))
    print(
        json.dumps(
            {
                "selected_parameter_set_internal_id": selected_parameter_set,
                "selected_feature_sets_at_optimum": int(
                    selected_row["feature_definitions_at_optimum"]
                ),
                "selected_mean_delta_rmse_dbhz": float(
                    selected_row["mean_excess_rmse_dbhz"]
                ),
                "selected_max_delta_rmse_dbhz": float(
                    selected_row["max_excess_rmse_dbhz"]
                ),
            },
            indent=2,
        )
    )
    print(json.dumps(output_audit["png"], indent=2))


if __name__ == "__main__":
    main()
