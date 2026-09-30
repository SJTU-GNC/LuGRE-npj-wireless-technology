#!/usr/bin/env python3
"""Fixed-architecture leave-one-operation-out evaluation for the final C/N0 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "script"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import train_cn0_trend_residual_tuned_no_leakage as formal


ENGINE = formal.engine
OUT_DIR = ROOT / "table" / "algorithm" / "leave_one_operation_out"
FIGURE_DIR = ROOT / "figure" / "paper_draft_v2" / "extended"
PHYSICAL_INPUT = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_constellation_physics_baseline_sp3_tlm_exact"
    / "cn0_constellation_physics_predictions.csv"
)
SELECTION_PATH = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_trend_residual_tuning_no_leakage"
    / "selected_model.json"
)
FINAL_MANIFEST = (
    ROOT
    / "table"
    / "paper_integration"
    / "algorithm_manifest"
    / "final_algorithm_manifest.json"
)
RANDOM_SEED = 42
BOOTSTRAP_BLOCK_MINUTES = 15
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_MIN_ELIGIBLE_ROWS = 20
DEFAULT_MIN_BLOCKS = 2

PHASE_NAMES = {
    "C": "Commissioning",
    "T": "Transfer",
    "L": "Lunar orbit",
    "S": "Surface",
}
PHASE_COLORS = {
    "C": "#2A9D8F",
    "T": "#E9A23B",
    "L": "#D65A4A",
    "S": "#457B9D",
}
FORBIDDEN_MODEL_INPUTS = {
    "cn0_dbhz_mean",
    "residual_trend_target_db",
    "cn0_observed_trend_dbhz",
    "cn0_trend_target_dbhz",
    "residual_fast_db",
    "samples",
    "op",
    "mission_phase",
    "system",
    "signal_name",
    "signal_id",
    "svid",
    "frequency_band",
    "rx_utc",
    "rx_utc_dt",
    "minute_utc",
    "time_bin_gps_seconds",
    "source_bin_gps_seconds",
    "evaluation_split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--min-eligible-rows",
        type=int,
        default=DEFAULT_MIN_ELIGIBLE_ROWS,
    )
    parser.add_argument("--min-blocks", type=int, default=DEFAULT_MIN_BLOCKS)
    parser.add_argument(
        "--ops",
        nargs="*",
        help="Optional OP subset for a diagnostic run. The default evaluates all eligible OPs.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def natural_op_key(value: str) -> tuple[int, str]:
    digits = "".join(character for character in str(value) if character.isdigit())
    return (int(digits) if digits else 10**9, str(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path) if include_sha256 else None,
    }


def model_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "n": int(len(observed)),
        "rmse_dbhz": float(math.sqrt(mean_squared_error(observed, predicted))),
        "mae_dbhz": float(mean_absolute_error(observed, predicted)),
        "bias_model_minus_observed_dbhz": float(np.mean(error)),
        "pearson_r": (
            float(np.corrcoef(observed, predicted)[0, 1])
            if len(observed) > 1
            and float(np.std(observed)) > 0
            and float(np.std(predicted)) > 0
            else np.nan
        ),
    }


def add_bootstrap_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    starts = output.groupby("op")["minute_utc"].transform("min")
    block_index = (
        (output["minute_utc"] - starts).dt.total_seconds()
        // (BOOTSTRAP_BLOCK_MINUTES * 60)
    ).astype(int)
    output["bootstrap_block_index"] = block_index
    output["bootstrap_block_id"] = (
        output["op"].astype(str) + "|B" + block_index.astype(str).str.zfill(3)
    )
    return output


def bootstrap_delta_rmse(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float, int]:
    grouped = (
        frame.assign(
            l1_sq=np.square(frame["error_l1_model_minus_observed_dbhz"]),
            hgb_sq=np.square(frame["error_hgb_model_minus_observed_dbhz"]),
        )
        .groupby("bootstrap_block_id", as_index=False)
        .agg(n=("op", "size"), l1_sse=("l1_sq", "sum"), hgb_sse=("hgb_sq", "sum"))
    )
    block_count = len(grouped)
    if block_count < 2 or replicates <= 0:
        return (np.nan, np.nan, block_count)
    n = grouped["n"].to_numpy(float)
    l1_sse = grouped["l1_sse"].to_numpy(float)
    hgb_sse = grouped["hgb_sse"].to_numpy(float)
    draws = rng.integers(0, block_count, size=(replicates, block_count))
    sampled_n = n[draws].sum(axis=1)
    delta = np.sqrt(hgb_sse[draws].sum(axis=1) / sampled_n) - np.sqrt(
        l1_sse[draws].sum(axis=1) / sampled_n
    )
    return (
        float(np.quantile(delta, 0.025)),
        float(np.quantile(delta, 0.975)),
        block_count,
    )


def operation_inventory(
    eligible: pd.DataFrame,
    all_bins: pd.DataFrame,
    minimum_rows: int,
    minimum_blocks: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_ops = sorted(all_bins["op"].astype(str).unique(), key=natural_op_key)
    for op in all_ops:
        all_part = all_bins[all_bins["op"].astype(str).eq(op)]
        part = eligible[eligible["op"].astype(str).eq(op)]
        phases = all_part["mission_phase"].dropna().astype(str).unique().tolist()
        phase = phases[0] if len(phases) == 1 else "mixed"
        block_count = int(part["bootstrap_block_id"].nunique()) if not part.empty else 0
        eligible_for_lopo = len(part) >= minimum_rows and block_count >= minimum_blocks
        reasons = []
        if len(part) < minimum_rows:
            reasons.append(f"eligible_rows<{minimum_rows}")
        if block_count < minimum_blocks:
            reasons.append(f"bootstrap_blocks<{minimum_blocks}")
        if len(phases) != 1:
            reasons.append("mission_phase_not_unique")
            eligible_for_lopo = False
        rows.append(
            {
                "op": op,
                "mission_phase_code": phase,
                "mission_phase_name": PHASE_NAMES.get(phase, phase),
                "rows_1min_total": int(len(all_part)),
                "rows_1min_quality_eligible": int(len(part)),
                "bootstrap_15min_blocks": block_count,
                "eligible_for_lopo": bool(eligible_for_lopo),
                "exclusion_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def summary_rows(
    predictions: pd.DataFrame,
    per_op: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pooled_groups: list[tuple[str, str, pd.DataFrame]] = [
        ("pooled", "all_operations", predictions)
    ]
    pooled_groups.extend(
        ("phase_pooled", PHASE_NAMES.get(str(phase), str(phase)), group)
        for phase, group in predictions.groupby("mission_phase_code", sort=False)
    )
    for scope, group_name, frame in pooled_groups:
        observed = frame["cn0_trend_target_dbhz"].to_numpy(float)
        l1 = model_metrics(observed, frame["cn0_robust_l1_only_dbhz"].to_numpy(float))
        hgb = model_metrics(
            observed, frame["cn0_robust_l1_hgb_dbhz"].to_numpy(float)
        )
        ci_low, ci_high, blocks = bootstrap_delta_rmse(frame, rng, replicates)
        rows.append(
            {
                "summary_scope": scope,
                "group": group_name,
                "n": int(len(frame)),
                "operation_count": int(frame["op"].nunique()),
                "bootstrap_15min_blocks": blocks,
                "rmse_l1_dbhz": l1["rmse_dbhz"],
                "rmse_l1_hgb_dbhz": hgb["rmse_dbhz"],
                "delta_rmse_hgb_minus_l1_dbhz": hgb["rmse_dbhz"]
                - l1["rmse_dbhz"],
                "delta_rmse_ci95_low_dbhz": ci_low,
                "delta_rmse_ci95_high_dbhz": ci_high,
                "mae_l1_dbhz": l1["mae_dbhz"],
                "mae_l1_hgb_dbhz": hgb["mae_dbhz"],
                "bias_l1_model_minus_observed_dbhz": l1[
                    "bias_model_minus_observed_dbhz"
                ],
                "bias_l1_hgb_model_minus_observed_dbhz": hgb[
                    "bias_model_minus_observed_dbhz"
                ],
                "pearson_r_l1": l1["pearson_r"],
                "pearson_r_l1_hgb": hgb["pearson_r"],
            }
        )

    macro_groups: list[tuple[str, str, pd.DataFrame]] = [
        ("macro_operation", "all_operations", per_op)
    ]
    macro_groups.extend(
        ("phase_macro_operation", PHASE_NAMES.get(str(phase), str(phase)), group)
        for phase, group in per_op.groupby("mission_phase_code", sort=False)
    )
    for scope, group_name, frame in macro_groups:
        rows.append(
            {
                "summary_scope": scope,
                "group": group_name,
                "n": int(frame["n"].sum()),
                "operation_count": int(len(frame)),
                "bootstrap_15min_blocks": int(frame["bootstrap_15min_blocks"].sum()),
                "rmse_l1_dbhz": float(frame["rmse_l1_dbhz"].mean()),
                "rmse_l1_hgb_dbhz": float(frame["rmse_l1_hgb_dbhz"].mean()),
                "delta_rmse_hgb_minus_l1_dbhz": float(
                    frame["delta_rmse_hgb_minus_l1_dbhz"].mean()
                ),
                "delta_rmse_ci95_low_dbhz": np.nan,
                "delta_rmse_ci95_high_dbhz": np.nan,
                "mae_l1_dbhz": float(frame["mae_l1_dbhz"].mean()),
                "mae_l1_hgb_dbhz": float(frame["mae_l1_hgb_dbhz"].mean()),
                "bias_l1_model_minus_observed_dbhz": float(
                    frame["bias_l1_model_minus_observed_dbhz"].mean()
                ),
                "bias_l1_hgb_model_minus_observed_dbhz": float(
                    frame["bias_l1_hgb_model_minus_observed_dbhz"].mean()
                ),
                "pearson_r_l1": float(frame["pearson_r_l1"].mean()),
                "pearson_r_l1_hgb": float(frame["pearson_r_l1_hgb"].mean()),
            }
        )
    return pd.DataFrame(rows)


def draw_delta_figure(per_op: pd.DataFrame, output_stem: Path) -> None:
    frame = per_op.sort_values(
        ["mission_phase_order", "op_number"], kind="stable"
    ).reset_index(drop=True)
    x = np.arange(len(frame))
    values = frame["delta_rmse_hgb_minus_l1_dbhz"].to_numpy(float)
    lower = values - frame["delta_rmse_ci95_low_dbhz"].to_numpy(float)
    upper = frame["delta_rmse_ci95_high_dbhz"].to_numpy(float) - values
    colors = [PHASE_COLORS.get(code, "#6C757D") for code in frame["mission_phase_code"]]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(12.6, 5.5), constrained_layout=True)
    ax.axhline(0.0, color="#222222", linewidth=1.0, zorder=1)
    ax.bar(
        x,
        values,
        color=colors,
        edgecolor="#333333",
        linewidth=0.45,
        width=0.72,
        zorder=2,
    )
    finite_ci = np.isfinite(lower) & np.isfinite(upper)
    ax.errorbar(
        x[finite_ci],
        values[finite_ci],
        yerr=np.vstack([lower[finite_ci], upper[finite_ci]]),
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(frame["op"], rotation=45, ha="right")
    ax.set_ylabel(r"$\Delta$RMSE (HGB - robust median offset), dB-Hz")
    ax.set_title("Leave-one-operation-out effect of the HGB residual stage")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[code], label=PHASE_NAMES[code])
        for code in ["C", "T", "L", "S"]
        if code in set(frame["mission_phase_code"])
    ]
    ax.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper right")
    ax.text(
        0.01,
        0.02,
        "Negative values indicate lower RMSE after adding HGB; bars show 15-min block-bootstrap 95% CIs.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    for suffix in [".svg", ".pdf", ".png"]:
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, dpi=300 if suffix == ".png" else None)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected_features = list(selection["selected_features"])
    selected_parameters = dict(selection["selected_parameters"])
    if len(selected_features) != 44:
        raise RuntimeError(f"Expected 44 selected features, found {len(selected_features)}")
    forbidden = sorted(FORBIDDEN_MODEL_INPUTS.intersection(selected_features))
    derived = sorted(
        feature
        for feature in selected_features
        if feature.startswith("error_")
        or ("residual" in feature.lower() and feature != "cn0_physics_trend_dbhz")
    )
    if forbidden or derived:
        raise RuntimeError(
            f"Forbidden or target-derived model inputs found: {forbidden + derived}"
        )
    if not ENGINE.SMOOTH_TREND_MODEL or ENGINE.TREND_WINDOW_MINUTES != 9:
        raise RuntimeError("Formal 9-minute trend configuration was not imported")

    raw = ENGINE.base_mod.load_data()
    source_features = ENGINE.base_mod.feature_columns(raw)
    bins = ENGINE.aggregate_one_minute(raw, source_features)
    bins["evaluation_split"] = "operation_isolated_target_construction"
    bins = ENGINE.add_trend_target(bins)
    eligible_mask = (
        bins["trend_training_eligible"].astype(bool)
        & bins["residual_trend_target_db"].notna()
        & bins["cn0_trend_target_dbhz"].notna()
        & bins[ENGINE.PHYSICS_TREND_COLUMN].notna()
    )
    eligible = add_bootstrap_blocks(bins.loc[eligible_mask].copy())
    bins_with_blocks = bins.merge(
        eligible[
            [
                "op",
                "mission_phase",
                "system",
                "signal_name",
                "svid",
                "source_bin_gps_seconds",
                "bootstrap_block_index",
                "bootstrap_block_id",
            ]
        ],
        on=[
            "op",
            "mission_phase",
            "system",
            "signal_name",
            "svid",
            "source_bin_gps_seconds",
        ],
        how="left",
        validate="one_to_one",
    )
    inventory = operation_inventory(
        eligible,
        bins_with_blocks,
        args.min_eligible_rows,
        args.min_blocks,
    )
    operations = inventory.loc[inventory["eligible_for_lopo"], "op"].tolist()
    if args.ops:
        requested = set(args.ops)
        missing = sorted(requested.difference(operations), key=natural_op_key)
        if missing:
            raise RuntimeError(f"Requested OPs are not eligible for LOPO: {missing}")
        operations = [op for op in operations if op in requested]
    if not operations:
        raise RuntimeError("No operations satisfy the LOPO eligibility contract")

    rng = np.random.default_rng(RANDOM_SEED)
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_provenance: list[dict[str, Any]] = []
    all_eligible_ops = sorted(
        eligible["op"].astype(str).unique(), key=natural_op_key
    )

    for fold_index, heldout_op in enumerate(operations, start=1):
        train = eligible[~eligible["op"].astype(str).eq(heldout_op)].copy()
        test = eligible[eligible["op"].astype(str).eq(heldout_op)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"Empty train or test partition for {heldout_op}")
        if train["op"].astype(str).eq(heldout_op).any():
            raise RuntimeError(f"Outer held operation leaked into training: {heldout_op}")

        beta = ENGINE.estimate_signal_beta(train)
        train_beta = ENGINE.signal_beta_values(train, beta)
        target = train["residual_trend_target_db"].to_numpy(float) - train_beta
        model = formal.selected_model_function(selected_features)
        model.fit(train[selected_features], target)

        test["signal_beta_train_only_db"] = ENGINE.signal_beta_values(test, beta)
        test["hgb_residual_raw_pred_db"] = (
            np.asarray(model.predict(test[selected_features]), dtype=float)
            + test["signal_beta_train_only_db"].to_numpy(float)
        )
        test["evaluation_split"] = "outer_lopo_test"
        test["hgb_residual_smoothed_pred_db"] = ENGINE.smooth_predicted_residual(
            test,
            "hgb_residual_raw_pred_db",
            isolate_evaluation_splits=True,
        )
        test["cn0_robust_l1_only_dbhz"] = (
            test[ENGINE.PHYSICS_TREND_COLUMN]
            + test["signal_beta_train_only_db"]
        )
        test["cn0_robust_l1_hgb_dbhz"] = (
            test[ENGINE.PHYSICS_TREND_COLUMN]
            + test["hgb_residual_smoothed_pred_db"]
        )
        test["error_l1_model_minus_observed_dbhz"] = (
            test["cn0_robust_l1_only_dbhz"] - test["cn0_trend_target_dbhz"]
        )
        test["error_hgb_model_minus_observed_dbhz"] = (
            test["cn0_robust_l1_hgb_dbhz"] - test["cn0_trend_target_dbhz"]
        )
        phase_codes = test["mission_phase"].astype(str).unique().tolist()
        if len(phase_codes) != 1:
            raise RuntimeError(f"Mission phase is not unique for {heldout_op}")
        phase_code = phase_codes[0]
        observed = test["cn0_trend_target_dbhz"].to_numpy(float)
        l1_metrics = model_metrics(
            observed, test["cn0_robust_l1_only_dbhz"].to_numpy(float)
        )
        hgb_metrics = model_metrics(
            observed, test["cn0_robust_l1_hgb_dbhz"].to_numpy(float)
        )
        ci_low, ci_high, block_count = bootstrap_delta_rmse(
            test, rng, args.bootstrap_replicates
        )
        op_number = natural_op_key(heldout_op)[0]
        metric_rows.append(
            {
                "op": heldout_op,
                "op_number": op_number,
                "mission_phase_code": phase_code,
                "mission_phase_name": PHASE_NAMES.get(phase_code, phase_code),
                "mission_phase_order": ["C", "T", "L", "S"].index(phase_code),
                "n": int(len(test)),
                "bootstrap_15min_blocks": block_count,
                "rmse_l1_dbhz": l1_metrics["rmse_dbhz"],
                "rmse_l1_hgb_dbhz": hgb_metrics["rmse_dbhz"],
                "delta_rmse_hgb_minus_l1_dbhz": hgb_metrics["rmse_dbhz"]
                - l1_metrics["rmse_dbhz"],
                "delta_rmse_ci95_low_dbhz": ci_low,
                "delta_rmse_ci95_high_dbhz": ci_high,
                "mae_l1_dbhz": l1_metrics["mae_dbhz"],
                "mae_l1_hgb_dbhz": hgb_metrics["mae_dbhz"],
                "bias_l1_model_minus_observed_dbhz": l1_metrics[
                    "bias_model_minus_observed_dbhz"
                ],
                "bias_l1_hgb_model_minus_observed_dbhz": hgb_metrics[
                    "bias_model_minus_observed_dbhz"
                ],
                "pearson_r_l1": l1_metrics["pearson_r"],
                "pearson_r_l1_hgb": hgb_metrics["pearson_r"],
            }
        )
        prediction_parts.append(
            test[
                [
                    "op",
                    "mission_phase",
                    "minute_utc",
                    "system",
                    "signal_name",
                    "svid",
                    "source_bin_gps_seconds",
                    "bootstrap_block_index",
                    "bootstrap_block_id",
                    "cn0_trend_target_dbhz",
                    ENGINE.PHYSICS_TREND_COLUMN,
                    "signal_beta_train_only_db",
                    "cn0_robust_l1_only_dbhz",
                    "hgb_residual_raw_pred_db",
                    "hgb_residual_smoothed_pred_db",
                    "cn0_robust_l1_hgb_dbhz",
                    "error_l1_model_minus_observed_dbhz",
                    "error_hgb_model_minus_observed_dbhz",
                ]
            ].copy()
        )
        imputer = model.named_steps["preprocess"].named_transformers_["physics"]
        fold_provenance.append(
            {
                "fold_index": fold_index,
                "outer_heldout_op": heldout_op,
                "outer_heldout_phase_code": phase_code,
                "outer_heldout_phase_name": PHASE_NAMES.get(phase_code, phase_code),
                "train_operations": [
                    op for op in all_eligible_ops if op != heldout_op
                ],
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "heldout_rows_in_beta_fit": 0,
                "heldout_rows_in_imputer_fit": 0,
                "heldout_rows_in_hgb_fit": 0,
                "signal_beta_train_only_db": beta,
                "imputer_median_by_feature": {
                    feature: (
                        float(value) if np.isfinite(value) else None
                    )
                    for feature, value in zip(
                        selected_features, imputer.statistics_, strict=True
                    )
                },
            }
        )
        print(
            f"[{fold_index:02d}/{len(operations):02d}] {heldout_op}: "
            f"n={len(test)}, RMSE L1={l1_metrics['rmse_dbhz']:.4f}, "
            f"L1+HGB={hgb_metrics['rmse_dbhz']:.4f}, "
            f"delta={hgb_metrics['rmse_dbhz'] - l1_metrics['rmse_dbhz']:+.4f}",
            flush=True,
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["mission_phase_code"] = predictions["mission_phase"].astype(str)
    predictions["mission_phase_name"] = predictions["mission_phase_code"].map(
        PHASE_NAMES
    )
    per_op = pd.DataFrame(metric_rows).sort_values("op_number").reset_index(drop=True)
    summaries = summary_rows(predictions, per_op, rng, args.bootstrap_replicates)

    predictions_path = output_dir / "cn0_lopo_predictions.csv"
    metrics_path = output_dir / "cn0_lopo_per_operation_metrics.csv"
    summary_path = output_dir / "cn0_lopo_summary_metrics.csv"
    inventory_path = output_dir / "cn0_lopo_operation_eligibility.csv"
    predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
    per_op.drop(columns=["mission_phase_order", "op_number"]).to_csv(
        metrics_path, index=False
    )
    summaries.to_csv(summary_path, index=False)
    inventory.to_csv(inventory_path, index=False)

    figure_stem = figure_dir / "Fig_extended_leave_one_operation_out_delta_rmse"
    draw_delta_figure(per_op, figure_stem)

    raw_quality_files = []
    raw_ops = set(raw["op"].astype(str))
    for path in sorted((ROOT / "data" / "receiver_observation" / "TLM").glob("TLM_RAW_*.txt")):
        if any(f"_{op}_" in path.name for op in raw_ops):
            raw_quality_files.append(artifact(path))

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "fixed-architecture outer leave-one-operation-out",
            "outer_unit": "complete operation",
            "outer_training_rule": (
                "The held operation contributes zero rows to the signal-wise robust "
                "median offset, SimpleImputer medians, and HGB fit."
            ),
            "target": (
                "split-isolated 9-minute observed C/N0 trend; median one-minute "
                "aggregation, centered 9-point rolling median and Savitzky-Golay order 2"
            ),
            "physics_prediction": "9-minute physics-based baseline trend",
            "robust_stage": (
                "one training-only median residual_trend_target_db per signal_name; "
                "absolute-error/L1-loss minimizer, not L1 coefficient regularization"
            ),
            "hgb_stage": "training-only HGB fit to residual after the robust signal offset",
            "prediction_smoothing": (
                "9-point rolling median plus Savitzky-Golay order 2, within held "
                "OP x signal x SVID continuous segments; no observed value is an input"
            ),
            "bias_sign": "model minus observed",
            "delta_rmse_definition": "RMSE(L1+HGB) - RMSE(L1); negative is improvement",
            "bootstrap": {
                "unit": "synchronized 15-minute blocks within the held operation",
                "replicates": args.bootstrap_replicates,
                "master_seed": RANDOM_SEED,
                "interval": "percentile 2.5% to 97.5%",
            },
            "eligibility": {
                "minimum_quality_eligible_rows": args.min_eligible_rows,
                "minimum_15min_blocks": args.min_blocks,
                "quality_mask": (
                    "trend_training_eligible AND finite trend target AND finite "
                    "physics trend"
                ),
            },
            "selection_status": (
                "No outer OP is used for selection in this run. The HGB architecture "
                "is frozen from the existing validation-selected formal model. Because "
                "that historical selection was not nested separately inside every "
                "outer fold, this is fixed-architecture LOPO rather than a fully nested "
                "unbiased hyperparameter-selection estimate."
            ),
        },
        "inputs": {
            "physical_input": artifact(PHYSICAL_INPUT),
            "selected_model": artifact(SELECTION_PATH),
            "final_algorithm_manifest": artifact(FINAL_MANIFEST),
            "formal_wrapper_script": artifact(
                ROOT / "script" / "train_cn0_trend_residual_tuned_no_leakage.py"
            ),
            "formal_engine_script": artifact(
                ROOT
                / "script"
                / "train_cn0_single_global_residual_1min_op74_holdout.py"
            ),
            "this_script": artifact(Path(__file__)),
            "raw_quality_tlm_files": raw_quality_files,
        },
        "model": {
            "feature_count": len(selected_features),
            "features": selected_features,
            "forbidden_features_present": forbidden,
            "target_derived_features_present": derived,
            "hgb_parameters": {**selected_parameters, "random_state": RANDOM_SEED},
        },
        "folds": fold_provenance,
        "excluded_operations": inventory.loc[
            ~inventory["eligible_for_lopo"],
            ["op", "exclusion_reason"],
        ].to_dict(orient="records"),
        "outputs": {
            "predictions": artifact(predictions_path),
            "per_operation_metrics": artifact(metrics_path),
            "summary_metrics": artifact(summary_path),
            "operation_eligibility": artifact(inventory_path),
            "figure_svg": artifact(figure_stem.with_suffix(".svg")),
            "figure_pdf": artifact(figure_stem.with_suffix(".pdf")),
            "figure_png": artifact(figure_stem.with_suffix(".png")),
        },
    }
    provenance_path = output_dir / "cn0_lopo_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pooled = summaries[
        summaries["summary_scope"].eq("pooled")
        & summaries["group"].eq("all_operations")
    ].iloc[0]
    readme = f"""# Leave-one-operation-out C/N0 trend evaluation

This directory contains a fixed-architecture outer LOPO evaluation of the
formal 1-minute trend model.

## Leakage contract

- The complete held operation contributes no row to the signal-wise robust
  median offset, imputer medians, or HGB fit.
- The four signal offsets are training-only medians of
  `residual_trend_target_db`; they minimize absolute error and are **not** L1
  coefficient regularization.
- The HGB feature set contains {len(selected_features)} physical/numeric
  features. OP, phase, constellation, signal name, SVID, raw sample count,
  observed C/N0, residuals, and absolute time are absent.
- Hyperparameters are frozen from the existing formal validation-selected
  architecture. This is fixed-architecture LOPO, not fully nested
  hyperparameter selection; see the provenance limitation.

## Evaluation

- Target: split-isolated 9-minute observed C/N0 trend at native 1-minute bins.
- Error/bias sign: model minus observed.
- Delta RMSE: `RMSE(robust median offset + HGB) - RMSE(robust median offset)`;
  negative values mean HGB improves the held operation.
- CI: {args.bootstrap_replicates} percentile-bootstrap replicates using
  synchronized 15-minute blocks, master seed {RANDOM_SEED}.
- Included operations: {len(operations)}. Exclusions and exact reasons are in
  `cn0_lopo_operation_eligibility.csv`.

## Pooled result

- Robust median-offset RMSE: {pooled['rmse_l1_dbhz']:.6f} dB-Hz
- Robust median-offset + HGB RMSE: {pooled['rmse_l1_hgb_dbhz']:.6f} dB-Hz
- Delta RMSE: {pooled['delta_rmse_hgb_minus_l1_dbhz']:+.6f} dB-Hz
- 95% block-bootstrap CI: [{pooled['delta_rmse_ci95_low_dbhz']:+.6f},
  {pooled['delta_rmse_ci95_high_dbhz']:+.6f}] dB-Hz

The PNG/PDF/SVG chart is diagnostic. Final visual styling remains outside this
algorithm deliverable.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {len(predictions)} held-out prediction rows to {output_dir}")


if __name__ == "__main__":
    main()
