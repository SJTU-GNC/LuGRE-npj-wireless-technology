#!/usr/bin/env python3
"""Plot fixed-final-model HGB on/off ablation for all eligible operations.

This script does not fit or update beta values, HGB parameters, training data,
or hyperparameters. It compares two outputs on the same cached prediction rows:

    A = physics-based baseline + saved robust median offset
    B = the stored final prediction from the same selected HGB model

Delta RMSE is RMSE(B) - RMSE(A), so negative values favour HGB.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


MODEL_DIR = ROOT / "table" / "algorithm" / "cn0_trend_residual_tuned_no_leakage"
MODEL_PATH = MODEL_DIR / "cn0_trend_residual_tuned_no_leakage.joblib"
PREDICTION_PATH = MODEL_DIR / "cn0_trend_residual_tuned_no_leakage_predictions.csv"
SPLIT_MANIFEST_PATH = (
    ROOT / "table" / "paper_integration" / "algorithm_manifest" / "dataset_split_manifest.csv"
)

TABLE_DIR = ROOT / "table" / "algorithm" / "fixed_model_ablation_all_operations"
METRICS_CSV = TABLE_DIR / "fixed_model_hgb_on_off_all_operations_metrics.csv"
AUDIT_CSV = TABLE_DIR / "fixed_model_hgb_on_off_all_operations_inclusion_audit.csv"
WORKBOOK_PATH = TABLE_DIR / "fixed_model_hgb_on_off_all_operations.xlsx"

FIGURE_DIR = ROOT / "figure" / "paper_draft_v2" / "extended"
STEM = "Fig_extended_fixed_model_hgb_on_off_all_operations_delta_rmse"
OUTPUT_BASE = FIGURE_DIR / STEM
PROVENANCE_PATH = FIGURE_DIR / f"{STEM}_provenance.json"

EXPECTED_MODEL_SHA256 = (
    "4ce4d4fd0344d1b500e182cb1e21852335e1a68cd856e20a6a86d3d23e8bceb8"
)
EXPECTED_PREDICTION_SHA256 = (
    "9471629de90e7860679a456af2c974cade4b8e0f9ad6700ff29b0f244b45f54b"
)
EXPECTED_MANIFEST_SHA256 = (
    "a4341b473fa6314c53e3222882cedd4be0114a16467d0a93321c19bc10985808"
)

MIN_COMMON_ROWS = 20
MIN_BOOTSTRAP_BLOCKS = 2
BOOTSTRAP_REPEATS = 20_000
BOOTSTRAP_BASE_SEED = 20260724

PHASE_ORDER = {"C": 0, "T": 1, "L": 2, "S": 3}
PHASE_NAMES = {
    "C": "Commissioning",
    "T": "Transfer",
    "L": "Lunar orbit",
    "S": "Surface",
}
PHASE_COLORS = {
    "C": "#5A8984",
    "T": "#BF9453",
    "L": "#9A7488",
    "S": "#71879C",
}
SPLIT_COLUMNS = ["train", "validation", "test", "external_holdout"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def input_snapshot(paths: list[Path]) -> dict[str, tuple[int, str]]:
    return {
        str(path.resolve()): (int(path.stat().st_size), sha256_file(path))
        for path in paths
    }


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def op_number(operation: str) -> int:
    return int(str(operation).replace("OP", ""))


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def verify_authoritative_inputs() -> None:
    expected = {
        MODEL_PATH: EXPECTED_MODEL_SHA256,
        PREDICTION_PATH: EXPECTED_PREDICTION_SHA256,
        SPLIT_MANIFEST_PATH: EXPECTED_MANIFEST_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Authoritative input hash mismatch for {path}: {actual_hash}"
            )


def load_model_artifact() -> dict[str, object]:
    artifact = joblib.load(MODEL_PATH)
    required = {"model", "model_family", "model_features", "signal_beta_db"}
    missing = required.difference(artifact)
    if missing:
        raise RuntimeError(f"Final model artifact keys missing: {sorted(missing)}")
    if len(artifact["model_features"]) != 44:
        raise RuntimeError("Expected 44 final selected HGB features")
    expected_signals = {"GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"}
    if set(artifact["signal_beta_db"]) != expected_signals:
        raise RuntimeError("Saved robust offsets do not cover the four signals")
    return artifact


def load_operation_universe() -> pd.DataFrame:
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH, low_memory=False)
    components = manifest[
        manifest["record_type"].eq("operation_split_component")
        & manifest["operation"].astype(str).str.fullmatch(r"OP\d+", na=False)
    ].copy()
    components["complete_operation_holdout"] = parse_bool(
        components["complete_operation_holdout"]
    )

    records: list[dict[str, object]] = []
    for operation, group in components.groupby("operation", sort=False):
        phases = group["mission_phase"].dropna().astype(str).unique()
        holdout_flags = group["complete_operation_holdout"].unique()
        if len(phases) != 1 or len(holdout_flags) != 1:
            raise RuntimeError(f"Inconsistent split metadata for {operation}")
        phase = phases[0]
        if phase not in PHASE_NAMES:
            raise RuntimeError(f"Unknown phase code for {operation}: {phase}")
        records.append(
            {
                "op": operation,
                "mission_phase_code": phase,
                "mission_phase_name": PHASE_NAMES[phase],
                "complete_operation_external": bool(holdout_flags[0]),
                "phase_order": PHASE_ORDER[phase],
                "op_number": op_number(operation),
            }
        )

    universe = pd.DataFrame(records).sort_values(
        ["phase_order", "op_number"], kind="stable"
    )
    universe = universe.reset_index(drop=True)
    if len(universe) != 20:
        raise RuntimeError(f"Expected 20 operations in the manifest, found {len(universe)}")
    return universe


def load_cached_predictions(
    artifact: dict[str, object],
) -> pd.DataFrame:
    columns = [
        "minute_utc",
        "evaluation_split",
        "time_block_id",
        "op",
        "mission_phase",
        "system",
        "signal_name",
        "svid",
        "cn0_physics_trend_dbhz",
        "cn0_trend_target_dbhz",
        "signal_beta_train_db",
        "ai_trend_residual_pred_db",
        "cn0_physics_ai_trend_dbhz",
        "trend_training_eligible",
    ]
    data = pd.read_csv(PREDICTION_PATH, usecols=columns, low_memory=False)
    if len(data) != 17_439:
        raise RuntimeError(f"Expected 17,439 cached rows, found {len(data)}")

    data["minute_utc"] = pd.to_datetime(data["minute_utc"], utc=True, errors="coerce")
    numeric = [
        "svid",
        "cn0_physics_trend_dbhz",
        "cn0_trend_target_dbhz",
        "signal_beta_train_db",
        "ai_trend_residual_pred_db",
        "cn0_physics_ai_trend_dbhz",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["trend_training_eligible"] = parse_bool(data["trend_training_eligible"])

    beta_map = artifact["signal_beta_db"]
    data["saved_robust_offset_db"] = data["signal_name"].map(beta_map)
    if data["saved_robust_offset_db"].isna().any():
        raise RuntimeError("A cached signal has no saved robust offset")
    if not np.allclose(
        data["signal_beta_train_db"].to_numpy(float),
        data["saved_robust_offset_db"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Cached robust offsets differ from the final artifact")

    data["prediction_hgb_off_dbhz"] = (
        data["cn0_physics_trend_dbhz"] + data["saved_robust_offset_db"]
    )
    data["prediction_hgb_on_dbhz"] = data["cn0_physics_ai_trend_dbhz"]

    finite_residual = data[
        [
            "cn0_physics_trend_dbhz",
            "ai_trend_residual_pred_db",
            "prediction_hgb_on_dbhz",
        ]
    ].notna().all(axis=1)
    reconstructed_on = (
        data["cn0_physics_trend_dbhz"] + data["ai_trend_residual_pred_db"]
    )
    if not np.allclose(
        data.loc[finite_residual, "prediction_hgb_on_dbhz"],
        reconstructed_on.loc[finite_residual],
        rtol=0.0,
        atol=5e-12,
    ):
        raise RuntimeError("Stored HGB-on output does not match the cached final residual")
    return data


def select_common_rows(data: pd.DataFrame, operation: str) -> pd.DataFrame:
    selected = data[data["op"].eq(operation)].copy()
    finite = selected[
        [
            "cn0_trend_target_dbhz",
            "prediction_hgb_off_dbhz",
            "prediction_hgb_on_dbhz",
        ]
    ].notna().all(axis=1)
    return selected[
        selected["trend_training_eligible"]
        & finite
        & selected["time_block_id"].notna()
        & selected["minute_utc"].notna()
    ].copy()


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["evaluation_split"].value_counts()
    return {f"n_{split}": int(counts.get(split, 0)) for split in SPLIT_COLUMNS}


def build_inclusion_audit(
    universe: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records: list[dict[str, object]] = []
    common_rows: dict[str, pd.DataFrame] = {}

    for meta in universe.itertuples(index=False):
        cached = predictions[predictions["op"].eq(meta.op)]
        common = select_common_rows(predictions, meta.op)
        common_rows[meta.op] = common
        row_count = int(len(common))
        block_count = int(common["time_block_id"].nunique())
        reasons: list[str] = []
        if cached.empty:
            reasons.append("no_rows_in_final_prediction_cache")
        if common.empty:
            reasons.append("no_quality_eligible_common_rows")
        if row_count < MIN_COMMON_ROWS:
            reasons.append(f"common_rows<{MIN_COMMON_ROWS}")
        if block_count < MIN_BOOTSTRAP_BLOCKS:
            reasons.append(f"15min_blocks<{MIN_BOOTSTRAP_BLOCKS}")

        splits = sorted(common["evaluation_split"].dropna().astype(str).unique())
        record = {
            "op": meta.op,
            "mission_phase_code": meta.mission_phase_code,
            "mission_phase_name": meta.mission_phase_name,
            "complete_operation_external": meta.complete_operation_external,
            "rows_in_final_prediction_cache": int(len(cached)),
            "common_quality_eligible_rows": row_count,
            "bootstrap_15min_blocks": block_count,
            "evaluation_splits_used": ",".join(splits),
            **split_counts(common),
            "included_in_figure": not reasons,
            "exclusion_reason": ";".join(reasons),
            "phase_order": meta.phase_order,
            "op_number": meta.op_number,
        }
        records.append(record)

    audit = pd.DataFrame(records).sort_values(
        ["phase_order", "op_number"], kind="stable"
    )
    audit = audit.reset_index(drop=True)

    included = set(audit.loc[audit["included_in_figure"], "op"])
    excluded = set(audit.loc[~audit["included_in_figure"], "op"])
    if len(included) != 18 or excluded != {"OP3", "OP37"}:
        raise RuntimeError(
            f"Unexpected all-operation eligibility: included={len(included)}, "
            f"excluded={sorted(excluded)}"
        )
    return audit, common_rows


def paired_block_bootstrap(frame: pd.DataFrame, operation: str) -> tuple[float, float]:
    block_data = frame.assign(
        squared_error_off=np.square(
            frame["prediction_hgb_off_dbhz"] - frame["cn0_trend_target_dbhz"]
        ),
        squared_error_on=np.square(
            frame["prediction_hgb_on_dbhz"] - frame["cn0_trend_target_dbhz"]
        ),
    )
    blocks = (
        block_data.groupby("time_block_id", sort=True)
        .agg(
            n=("time_block_id", "size"),
            sse_off=("squared_error_off", "sum"),
            sse_on=("squared_error_on", "sum"),
        )
        .reset_index(drop=True)
    )
    if len(blocks) < MIN_BOOTSTRAP_BLOCKS:
        raise RuntimeError(f"Insufficient 15-min blocks for {operation}")

    rng = np.random.default_rng(BOOTSTRAP_BASE_SEED + op_number(operation))
    sampled_indices = rng.integers(
        0, len(blocks), size=(BOOTSTRAP_REPEATS, len(blocks))
    )
    n = blocks["n"].to_numpy(float)[sampled_indices].sum(axis=1)
    sse_off = blocks["sse_off"].to_numpy(float)[sampled_indices].sum(axis=1)
    sse_on = blocks["sse_on"].to_numpy(float)[sampled_indices].sum(axis=1)
    deltas = np.sqrt(sse_on / n) - np.sqrt(sse_off / n)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def build_metrics(
    audit: pd.DataFrame, common_rows: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for meta in audit[audit["included_in_figure"]].itertuples(index=False):
        frame = common_rows[meta.op]
        target = frame["cn0_trend_target_dbhz"].to_numpy(float)
        prediction_off = frame["prediction_hgb_off_dbhz"].to_numpy(float)
        prediction_on = frame["prediction_hgb_on_dbhz"].to_numpy(float)
        rmse_off = rmse(target, prediction_off)
        rmse_on = rmse(target, prediction_on)
        delta = rmse_on - rmse_off
        low, high = paired_block_bootstrap(frame, meta.op)

        if high < 0.0:
            interpretation = "resolved_improvement"
        elif low > 0.0:
            interpretation = "resolved_degradation"
        else:
            interpretation = "no_resolved_change"

        splits = sorted(frame["evaluation_split"].dropna().astype(str).unique())
        records.append(
            {
                "op": meta.op,
                "mission_phase_code": meta.mission_phase_code,
                "mission_phase_name": meta.mission_phase_name,
                "complete_operation_external": meta.complete_operation_external,
                "evaluation_splits_used": ",".join(splits),
                **split_counts(frame),
                "n_common_rows": int(len(frame)),
                "bootstrap_15min_blocks": int(frame["time_block_id"].nunique()),
                "unique_epochs": int(frame["minute_utc"].nunique()),
                "unique_links": int(
                    frame[["signal_name", "svid"]].drop_duplicates().shape[0]
                ),
                "first_utc": frame["minute_utc"].min().isoformat(),
                "last_utc": frame["minute_utc"].max().isoformat(),
                "rmse_hgb_off_dbhz": rmse_off,
                "rmse_hgb_on_dbhz": rmse_on,
                "delta_rmse_on_minus_off_dbhz": delta,
                "delta_rmse_ci95_low_dbhz": low,
                "delta_rmse_ci95_high_dbhz": high,
                "ci95_crosses_zero": bool(low <= 0.0 <= high),
                "point_estimate_direction": (
                    "improvement" if delta < 0.0 else "degradation"
                ),
                "interval_interpretation": interpretation,
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "bootstrap_seed": BOOTSTRAP_BASE_SEED + op_number(meta.op),
                "phase_order": meta.phase_order,
                "op_number": meta.op_number,
            }
        )

    metrics = pd.DataFrame(records).sort_values(
        ["phase_order", "op_number"], kind="stable"
    )
    metrics = metrics.reset_index(drop=True)
    if len(metrics) != 18:
        raise RuntimeError(f"Expected 18 plotted operations, found {len(metrics)}")
    if not np.allclose(
        metrics["delta_rmse_on_minus_off_dbhz"],
        metrics["rmse_hgb_on_dbhz"] - metrics["rmse_hgb_off_dbhz"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Delta RMSE identity check failed")
    return metrics


def write_workbook(metrics: pd.DataFrame, audit: pd.DataFrame) -> None:
    definitions = pd.DataFrame(
        [
            {
                "field": "comparison",
                "definition": (
                    "Same cached rows and same selected model. HGB off = "
                    "physics-based baseline + saved robust median offset; "
                    "HGB on = stored final selected-model prediction."
                ),
            },
            {
                "field": "delta_rmse_on_minus_off_dbhz",
                "definition": (
                    "RMSE(HGB on) - RMSE(HGB off); negative values favour HGB."
                ),
            },
            {
                "field": "95% CI",
                "definition": (
                    "Paired 15-min block bootstrap, 20,000 repeats, within each OP; "
                    "the model is never refitted."
                ),
            },
            {
                "field": "inclusion threshold",
                "definition": (
                    f"At least {MIN_COMMON_ROWS} common quality-eligible rows and "
                    f"at least {MIN_BOOTSTRAP_BLOCKS} distinct 15-min blocks."
                ),
            },
            {
                "field": "interpretation boundary",
                "definition": (
                    "Operations containing train or validation rows are descriptive "
                    "fixed-model ablations, not independent external evaluations."
                ),
            },
        ]
    )

    with pd.ExcelWriter(WORKBOOK_PATH, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="per_operation", index=False)
        audit.to_excel(writer, sheet_name="inclusion_audit", index=False)
        definitions.to_excel(writer, sheet_name="definitions", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.sheet_view.showGridLines = False
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                width = min(max(max(map(len, values), default=0) + 2, 10), 42)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def x_positions_with_phase_gaps(metrics: pd.DataFrame) -> tuple[np.ndarray, list[float]]:
    positions: list[float] = []
    separators: list[float] = []
    cursor = 0.0
    previous_phase: str | None = None
    for row in metrics.itertuples(index=False):
        if previous_phase is not None and row.mission_phase_code != previous_phase:
            cursor += 0.65
            separators.append(cursor - 0.825)
        positions.append(cursor)
        cursor += 1.0
        previous_phase = row.mission_phase_code
    return np.asarray(positions, dtype=float), separators


def plot_figure(metrics: pd.DataFrame) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    x, separators = x_positions_with_phase_gaps(metrics)
    delta = metrics["delta_rmse_on_minus_off_dbhz"].to_numpy(float)
    low = metrics["delta_rmse_ci95_low_dbhz"].to_numpy(float)
    high = metrics["delta_rmse_ci95_high_dbhz"].to_numpy(float)
    colors = [PHASE_COLORS[phase] for phase in metrics["mission_phase_code"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.35), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.25)

    bars = ax.bar(
        x,
        delta,
        width=0.70,
        color=colors,
        edgecolor="#3F4448",
        linewidth=0.45,
        alpha=0.86,
        zorder=2,
    )
    for patch, crosses_zero in zip(bars, metrics["ci95_crosses_zero"], strict=True):
        if bool(crosses_zero):
            patch.set_alpha(0.42)
            patch.set_hatch("//")

    ax.vlines(x, low, high, color="#34383C", linewidth=0.75, zorder=4)
    cap_half_width = 0.11
    ax.hlines(
        low,
        x - cap_half_width,
        x + cap_half_width,
        color="#34383C",
        linewidth=0.75,
        zorder=4,
    )
    ax.hlines(
        high,
        x - cap_half_width,
        x + cap_half_width,
        color="#34383C",
        linewidth=0.75,
        zorder=4,
    )
    ax.scatter(x, delta, s=6.5, c="#292D30", linewidths=0, zorder=5)

    ax.axhline(0.0, color="#25282B", linewidth=0.85, zorder=1)
    ax.grid(axis="y", color="#D8DADD", linewidth=0.45, alpha=0.72, zorder=0)
    for separator in separators:
        ax.axvline(separator, color="#D2D5D8", linewidth=0.55, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics["op"], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(
        r"$\Delta$RMSE = RMSE(HGB on) $-$ RMSE(HGB off) (dB-Hz)",
        labelpad=5,
    )

    lower = min(float(low.min()), 0.0)
    upper = max(float(high.max()), 0.0)
    span = max(upper - lower, 1.0)
    ax.set_ylim(lower - 0.08 * span, upper + 0.12 * span)
    ax.set_xlim(x.min() - 0.7, x.max() + 0.7)

    ax.text(
        0.008,
        0.035,
        "negative: HGB improves",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#555B60",
        fontsize=6.4,
    )
    ax.text(
        0.008,
        0.965,
        "positive: HGB worsens",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#555B60",
        fontsize=6.4,
    )

    legend_handles = [
        Patch(
            facecolor=PHASE_COLORS[phase],
            edgecolor="#3F4448",
            linewidth=0.45,
            label=PHASE_NAMES[phase],
        )
        for phase in ["C", "T", "L", "S"]
    ]
    legend_handles.append(
        Patch(
            facecolor="#FFFFFF",
            edgecolor="#3F4448",
            linewidth=0.55,
            hatch="//",
            label="95% CI includes zero",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
        handlelength=1.25,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    for suffix in [".png", ".pdf", ".svg"]:
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(OUTPUT_BASE.with_suffix(suffix), **kwargs)
    plt.close(fig)


def write_provenance(
    artifact: dict[str, object], metrics: pd.DataFrame, audit: pd.DataFrame
) -> None:
    included = audit[audit["included_in_figure"]]
    excluded = audit[~audit["included_in_figure"]]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_stem": STEM,
        "scientific_contract": {
            "core_conclusion": (
                "The contribution of the already selected HGB correction varies by "
                "operation when evaluated against the same HGB-off prediction rows."
            ),
            "comparison_A_hgb_off": (
                "physics-based baseline plus saved robust median offset"
            ),
            "comparison_B_hgb_on": "stored final selected-model prediction",
            "delta_definition": "RMSE(B) - RMSE(A); negative values favour HGB",
            "same_row_pairing": True,
            "model_refit_or_parameter_update": False,
            "prediction_reinference": False,
            "interpretation_boundary": (
                "This is a fixed-model ablation diagnostic. Operations that include "
                "train or validation rows are not independent external evaluations."
            ),
        },
        "model_artifact": {
            **file_record(MODEL_PATH),
            "model_family_as_stored": str(artifact["model_family"]),
            "model_feature_count": len(artifact["model_features"]),
            "saved_robust_offsets_db": artifact["signal_beta_db"],
        },
        "prediction_cache": file_record(PREDICTION_PATH),
        "operation_manifest": file_record(SPLIT_MANIFEST_PATH),
        "eligibility": {
            "operation_universe_count": int(len(audit)),
            "included_count": int(len(included)),
            "excluded_count": int(len(excluded)),
            "minimum_common_quality_eligible_rows": MIN_COMMON_ROWS,
            "minimum_distinct_15min_blocks": MIN_BOOTSTRAP_BLOCKS,
            "included_operations": included["op"].tolist(),
            "excluded_operations": excluded[
                [
                    "op",
                    "mission_phase_name",
                    "common_quality_eligible_rows",
                    "bootstrap_15min_blocks",
                    "exclusion_reason",
                ]
            ].to_dict(orient="records"),
        },
        "uncertainty": {
            "method": "paired within-operation 15-min block bootstrap",
            "repeats": BOOTSTRAP_REPEATS,
            "confidence_interval": "2.5th and 97.5th percentiles",
            "model_refit_within_bootstrap": False,
            "base_seed": BOOTSTRAP_BASE_SEED,
        },
        "per_operation_metrics": metrics.drop(
            columns=["phase_order", "op_number"]
        ).round(10).to_dict(orient="records"),
        "outputs": {
            "metrics_csv": file_record(METRICS_CSV),
            "inclusion_audit_csv": file_record(AUDIT_CSV),
            "workbook": file_record(WORKBOOK_PATH),
            "figure_files": [
                file_record(OUTPUT_BASE.with_suffix(suffix))
                for suffix in [".png", ".pdf", ".svg"]
            ],
            "plot_script": file_record(Path(__file__)),
        },
        "figure_contract": {
            "archetype": "quantitative comparison",
            "whole_figure_title": False,
            "phase_encoding": "restrained bar fill and legend",
            "zero_reference_line": True,
            "ci_crosses_zero_encoding": "reduced opacity and hatch",
            "white_background": True,
            "png_dpi": 600,
            "svg_text_editable": True,
            "pdf_fonttype": 42,
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_outputs(metrics: pd.DataFrame, audit: pd.DataFrame) -> dict[str, object]:
    for path in [
        METRICS_CSV,
        AUDIT_CSV,
        WORKBOOK_PATH,
        OUTPUT_BASE.with_suffix(".png"),
        OUTPUT_BASE.with_suffix(".pdf"),
        OUTPUT_BASE.with_suffix(".svg"),
        PROVENANCE_PATH,
    ]:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")

    svg_text = OUTPUT_BASE.with_suffix(".svg").read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG does not retain editable text")
    if any(
        re.search(rf">\s*{operation}\s*<", svg_text)
        for operation in ["OP3", "OP37"]
    ):
        raise RuntimeError("Excluded operations appeared in the figure")
    if len(metrics) != 18 or int(audit["included_in_figure"].sum()) != 18:
        raise RuntimeError("Displayed-operation count changed")

    return {
        "displayed_operation_count": 18,
        "excluded_operations": audit.loc[
            ~audit["included_in_figure"], "op"
        ].tolist(),
        "delta_min_dbhz": float(metrics["delta_rmse_on_minus_off_dbhz"].min()),
        "delta_max_dbhz": float(metrics["delta_rmse_on_minus_off_dbhz"].max()),
        "ci_crosses_zero_count": int(metrics["ci95_crosses_zero"].sum()),
        "svg_editable_text": True,
        "png_dpi": 600,
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    verify_authoritative_inputs()
    protected_inputs = [MODEL_PATH, PREDICTION_PATH, SPLIT_MANIFEST_PATH]
    before = input_snapshot(protected_inputs)

    artifact = load_model_artifact()
    universe = load_operation_universe()
    predictions = load_cached_predictions(artifact)
    audit, common_rows = build_inclusion_audit(universe, predictions)
    metrics = build_metrics(audit, common_rows)

    metrics.drop(columns=["phase_order", "op_number"]).to_csv(
        METRICS_CSV, index=False, float_format="%.10f"
    )
    audit.drop(columns=["phase_order", "op_number"]).to_csv(
        AUDIT_CSV, index=False
    )
    write_workbook(
        metrics.drop(columns=["phase_order", "op_number"]),
        audit.drop(columns=["phase_order", "op_number"]),
    )
    plot_figure(metrics)
    write_provenance(artifact, metrics, audit)
    qa = verify_outputs(metrics, audit)

    after = input_snapshot(protected_inputs)
    if after != before:
        raise RuntimeError("An authoritative input changed during execution")

    print(json.dumps(qa, indent=2))
    print(metrics[[
        "op",
        "mission_phase_name",
        "n_common_rows",
        "rmse_hgb_off_dbhz",
        "rmse_hgb_on_dbhz",
        "delta_rmse_on_minus_off_dbhz",
        "delta_rmse_ci95_low_dbhz",
        "delta_rmse_ci95_high_dbhz",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
