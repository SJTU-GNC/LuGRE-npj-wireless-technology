#!/usr/bin/env python3
"""Build leakage-safe C/N0 physics baselines from the WGC feature table.

The absolute link-budget output is intentionally left unavailable until the
transmit/receive gain and receiver noise terms are supplied. Calibrated model
parameters are estimated on training rows only; held-out observations are
never used to fit or adjust predictions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("table/cn0_physics_features_available_wgc_new.csv")
DEFAULT_OUTPUT_DIR = Path("table/algorithm/cn0_physics_baseline")

MODEL_FSPL = "fspl_only_global_train_median"
MODEL_PHYSICS = "physics_only_signal_fspl_train_median"
MODEL_PROXY = "proxy_geometry_phase_robust_train_only"
MODEL_ABSOLUTE = "physics_absolute_link_budget_unavailable"

REQUIRED_COLUMNS = {
    "rx_utc",
    "rx_gps_seconds",
    "mission_phase",
    "system",
    "svid",
    "signal_name",
    "frequency_mhz",
    "cn0_dbhz_mean",
    "geometric_range_km",
    "fspl_db",
    "tx_offboresight_deg",
    "earth_limb_margin_deg",
    "moon_limb_margin_deg",
    "earth_grazing_altitude_km",
    "moon_grazing_altitude_km",
    "earth_blocked",
    "moon_blocked",
}

PROXY_CONTINUOUS = [
    "tx_offboresight_deg",
    "earth_limb_proximity_proxy",
    "moon_limb_proximity_proxy",
    "earth_blocked",
    "moon_blocked",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--holdout-policy",
        choices=["last_day_per_phase"],
        default="last_day_per_phase",
        help="Strict chronological holdout policy.",
    )
    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input table has no rows")
    if df["cn0_dbhz_mean"].isna().any() or df["fspl_db"].isna().any():
        raise ValueError("Observed C/N0 and FSPL must be complete")


def assign_strict_holdout(df: pd.DataFrame) -> tuple[pd.Series, dict[str, str]]:
    """Hold out the latest complete UTC date independently in each phase."""
    utc_day = df["rx_utc"].dt.strftime("%Y-%m-%d")
    holdout_days = (
        pd.DataFrame({"phase": df["mission_phase"].astype(str), "utc_day": utc_day})
        .groupby("phase", sort=True)["utc_day"]
        .max()
        .to_dict()
    )
    is_test = pd.Series(
        [day == holdout_days[str(phase)] for day, phase in zip(utc_day, df["mission_phase"])],
        index=df.index,
    )
    if not is_test.any() or is_test.all():
        raise ValueError("Holdout policy did not produce nonempty train and test sets")
    return is_test, holdout_days


def median_intercept(y_plus_fspl: pd.Series) -> float:
    values = y_plus_fspl.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate an intercept from an empty training set")
    return float(np.median(values))


def add_proxy_features(df: pd.DataFrame) -> None:
    # These are smooth geometric indicators, not measured propagation losses.
    df["earth_limb_proximity_proxy"] = np.exp(
        -np.minimum(np.abs(df["earth_limb_margin_deg"].to_numpy(float)), 90.0) / 2.0
    )
    df["moon_limb_proximity_proxy"] = np.exp(
        -np.minimum(np.abs(df["moon_limb_margin_deg"].to_numpy(float)), 90.0) / 2.0
    )


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scale = np.where((~np.isfinite(scale)) | (scale < 1e-12), fallback, scale)
    scale = np.where((~np.isfinite(scale)) | (scale < 1e-12), 1.0, scale)
    standardized = (np.where(np.isfinite(values), values, center) - center) / scale
    # Bound proxy extrapolation to the range represented by training data only.
    clip_low = np.nanpercentile(standardized, 0.5, axis=0)
    clip_high = np.nanpercentile(standardized, 99.5, axis=0)
    clip_low = np.minimum(clip_low, 0.0)
    clip_high = np.maximum(clip_high, 0.0)
    return center, scale, clip_low, clip_high


def build_proxy_design(
    df: pd.DataFrame,
    signal_levels: list[str],
    phase_levels: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    clip_low: np.ndarray,
    clip_high: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    continuous = df[PROXY_CONTINUOUS].to_numpy(dtype=float)
    continuous = np.where(np.isfinite(continuous), continuous, center)
    z = (continuous - center) / scale
    z = np.clip(z, clip_low, clip_high)

    columns = [np.ones(len(df), dtype=float)]
    names = ["intercept_effective_link_margin_db"]
    for level in signal_levels[1:]:
        columns.append((df["signal_name"].astype(str).to_numpy() == level).astype(float))
        names.append(f"signal[{level}]_offset_db")
    for level in phase_levels[1:]:
        columns.append((df["mission_phase"].astype(str).to_numpy() == level).astype(float))
        names.append(f"phase[{level}]_offset_db")
    columns.extend(z[:, i] for i in range(z.shape[1]))
    names.extend(f"{name}_standardized_coefficient_db" for name in PROXY_CONTINUOUS)
    return np.column_stack(columns), names


def huber_irls(
    x: np.ndarray,
    y: np.ndarray,
    delta: float = 1.345,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, int, float]:
    """Deterministic Huber iteratively reweighted least squares."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    ridge = np.eye(x.shape[1]) * 1e-10
    ridge[0, 0] = 0.0
    scale = 1.0
    for iteration in range(1, max_iter + 1):
        residual = y - x @ beta
        median = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - median))
        if not np.isfinite(scale) or scale < 1e-9:
            scale = max(float(np.std(residual)), 1e-9)
        normalized = np.abs(residual) / scale
        weights = np.ones_like(normalized)
        mask = normalized > delta
        weights[mask] = delta / normalized[mask]
        root_w = np.sqrt(weights)
        xw = x * root_w[:, None]
        yw = y * root_w
        new_beta = np.linalg.solve(xw.T @ xw + ridge, xw.T @ yw)
        if np.max(np.abs(new_beta - beta)) <= tolerance * (1.0 + np.max(np.abs(beta))):
            beta = new_beta
            break
        beta = new_beta
    return beta, iteration, float(scale)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if observed.size == 0:
        return {
            "n": 0,
            "mae_dbhz": math.nan,
            "rmse_dbhz": math.nan,
            "median_ae_dbhz": math.nan,
            "p95_ae_dbhz": math.nan,
            "bias_dbhz": math.nan,
            "r2": math.nan,
            "pearson_r": math.nan,
        }
    residual = predicted - observed
    abs_error = np.abs(residual)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    pearson = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if observed.size > 1 and np.std(observed) > 0 and np.std(predicted) > 0
        else math.nan
    )
    return {
        "n": int(observed.size),
        "mae_dbhz": float(np.mean(abs_error)),
        "rmse_dbhz": float(np.sqrt(np.mean(residual**2))),
        "median_ae_dbhz": float(np.median(abs_error)),
        "p95_ae_dbhz": float(np.percentile(abs_error, 95)),
        "bias_dbhz": float(np.mean(residual)),
        "r2": r2,
        "pearson_r": pearson,
    }


def metric_rows(df: pd.DataFrame, prediction_columns: dict[str, tuple[str, str]]) -> list[dict]:
    rows: list[dict] = []
    group_specs = [("overall", pd.Series("all", index=df.index)), ("phase", df["mission_phase"]), ("signal", df["signal_name"])]
    for model_name, (model_class, prediction_column) in prediction_columns.items():
        for split in ["train", "test"]:
            split_mask = df["split"].eq(split)
            for group_type, groups in group_specs:
                for group_value in sorted(groups[split_mask].astype(str).unique()):
                    mask = split_mask & groups.astype(str).eq(group_value)
                    metrics = regression_metrics(
                        df.loc[mask, "cn0_dbhz_mean"].to_numpy(float),
                        df.loc[mask, prediction_column].to_numpy(float),
                    )
                    rows.append(
                        {
                            "model_name": model_name,
                            "model_class": model_class,
                            "evaluation_status": "evaluated_train_only_calibration",
                            "split": split,
                            "group_type": group_type,
                            "group_value": group_value,
                            **metrics,
                        }
                    )
    rows.append(
        {
            "model_name": MODEL_ABSOLUTE,
            "model_class": "absolute_physics_link_budget",
            "evaluation_status": "not_evaluated_missing_required_link_budget_parameters",
            "split": "test",
            "group_type": "overall",
            "group_value": "all",
            **regression_metrics(np.array([]), np.array([])),
        }
    )
    return rows


def parameter_row(model_name: str, category: str, parameter: str, value: object, unit: str, source: str, status: str, notes: str) -> dict:
    return {
        "model_name": model_name,
        "parameter_category": category,
        "parameter_name": parameter,
        "value": value,
        "unit": unit,
        "source": source,
        "status": status,
        "notes": notes,
    }


def build_readme(
    input_path: Path,
    output_dir: Path,
    df: pd.DataFrame,
    holdout_days: dict[str, str],
    metrics: pd.DataFrame,
) -> str:
    test_overall = metrics[(metrics["split"] == "test") & (metrics["group_type"] == "overall")]
    metric_lines = []
    for row in test_overall.itertuples(index=False):
        if int(row.n) > 0:
            metric_lines.append(
                f"| `{row.model_name}` | {int(row.n):,} | {row.mae_dbhz:.3f} | {row.rmse_dbhz:.3f} | {row.p95_ae_dbhz:.3f} | {row.bias_dbhz:.3f} |"
            )
    holdout_lines = [f"| `{phase}` | `{day}` |" for phase, day in sorted(holdout_days.items())]
    return f"""# LuGRE C/N0 Physics Baseline - WGC New

## Scope

This directory contains a leakage-safe first-version C/N0 baseline built from
`{input_path.as_posix()}`. The source feature table is read only and is not modified.

## Reproduction

```text
python script/build_cn0_physics_baseline.py --input {input_path.as_posix()} --output-dir {output_dir.as_posix()}
```

Dependencies: Python 3, NumPy, and pandas. No random seed is needed; the split and
Huber IRLS fit are deterministic.

## A. Absolute Link Budget Status

The current table does **not** support an absolute direct link budget. The output
column `cn0_physics_absolute_link_budget_dbhz` is therefore blank by design and
`absolute_link_budget_status` is set to
`unavailable_missing_eirp_gtx_grx_tsys_lrx`.

An absolute prediction would require either

```text
C/N0 = EIRP_tx + G_rx - FSPL - L_other + 228.6 - 10 log10(T_sys)
```

or the equivalent decomposition using `P_tx + G_tx`. Reliable numerical values
are currently missing for:

- per-signal `P_tx` or EIRP;
- direction-, frequency-, satellite/block-dependent `G_tx` (main/side lobes);
- LuGRE direction-, frequency-, attitude-dependent `G_rx`;
- receiver/system noise temperature `T_sys` or calibrated `G/T`;
- receiver implementation/cable loss `L_rx` and other propagation losses.

`tx_offboresight_deg` is not gain in dB. Limb margins, grazing altitudes, and
blocked flags are geometry, not calibrated attenuation. They are not silently
substituted for the missing absolute terms.

## B. Training-Calibrated Baselines

All calibrations use training rows only. No row's observed C/N0 is used in its
prediction except when that row belongs to the explicitly labelled training set;
held-out C/N0 is used only after prediction for metrics.

| Model | Class | Definition |
|---|---|---|
| `{MODEL_FSPL}` | FSPL-only | One global robust training median `alpha`; prediction is `alpha - FSPL`. |
| `{MODEL_PHYSICS}` | Physics-only, train-calibrated | One training median `alpha_signal` per signal; prediction is `alpha_signal - FSPL`. Frequency is already in FSPL. |
| `{MODEL_PROXY}` | Geometry proxy | Huber fit of effective link margin using signal, phase, off-boresight, limb-proximity, and blocked indicators. This is explicitly not a direct physical gain/loss model. |

The Earth/Moon blocked flags are not assigned an arbitrary deterministic dB loss
in the physics-only models because diffraction, terrain/frame accuracy, and link
detection thresholds are not yet characterized. They enter only the model named
`proxy`.

## Strict Holdout

The latest UTC observation date in each mission phase is held out in full:

| Mission phase | Test UTC date |
|---|---|
{chr(10).join(holdout_lines)}

Rows: {len(df):,} total, {(df['split'] == 'train').sum():,} train, and
{(df['split'] == 'test').sum():,} test. The parameter table records every split
date and fitting parameter.

## Held-out Overall Metrics

| Model | N | MAE (dB-Hz) | RMSE (dB-Hz) | P95 absolute error (dB-Hz) | Bias (dB-Hz) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

Metrics must be interpreted as temporal/phase generalization, not random-row
interpolation. The proxy model can improve numerical accuracy while remaining
less physically identifiable than a true antenna/noise-aware link budget.

## Outputs

| File | Contents |
|---|---|
| `cn0_physics_baseline_predictions.csv` | Source rows plus split labels, prediction columns, and row-wise errors. |
| `cn0_physics_baseline_metrics.csv` | Train/test metrics overall and by mission phase/signal. |
| `cn0_physics_baseline_parameters.csv` | Missing absolute terms, split dates, robust intercepts, proxy scaling, and Huber coefficients. |

## Leakage Audit

- Split assignment uses only UTC date and mission phase.
- Global/signal intercepts use only rows labelled `train`.
- Proxy scaling, categories, Huber coefficients, and fallback values use only training rows.
- Test C/N0 is read only by the final metric calculation.
- No SPICE/WGC truth, test residual, per-row observed C/N0, or test-set mean is added to a prediction.
"""


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    df["rx_utc"] = pd.to_datetime(df["rx_utc"], utc=True)
    validate_input(df)
    add_proxy_features(df)

    is_test, holdout_days = assign_strict_holdout(df)
    df["utc_day"] = df["rx_utc"].dt.strftime("%Y-%m-%d")
    df["split"] = np.where(is_test, "test", "train")
    df["split_policy"] = "last_complete_utc_day_per_mission_phase"
    df["calibration_target_used"] = np.where(is_test, False, True)
    df["heldout_for_validation"] = is_test
    df["absolute_link_budget_status"] = "unavailable_missing_eirp_gtx_grx_tsys_lrx"
    df["cn0_physics_absolute_link_budget_dbhz"] = np.nan

    train_mask = ~is_test
    train = df.loc[train_mask]
    effective_margin_train = train["cn0_dbhz_mean"] + train["fspl_db"]

    global_alpha = median_intercept(effective_margin_train)
    df["cn0_fspl_only_global_dbhz"] = global_alpha - df["fspl_db"]

    signal_alphas = (
        train.assign(effective_margin=effective_margin_train)
        .groupby("signal_name", sort=True)["effective_margin"]
        .median()
        .to_dict()
    )
    df["physics_signal_alpha_db"] = df["signal_name"].map(signal_alphas).fillna(global_alpha)
    df["cn0_physics_only_signal_fspl_dbhz"] = df["physics_signal_alpha_db"] - df["fspl_db"]

    signal_levels = sorted(train["signal_name"].astype(str).unique())
    phase_levels = sorted(train["mission_phase"].astype(str).unique())
    center, scale, clip_low, clip_high = robust_scale(
        train[PROXY_CONTINUOUS].to_numpy(dtype=float)
    )
    x_train, proxy_parameter_names = build_proxy_design(
        train, signal_levels, phase_levels, center, scale, clip_low, clip_high
    )
    x_all, _ = build_proxy_design(
        df, signal_levels, phase_levels, center, scale, clip_low, clip_high
    )
    proxy_beta, proxy_iterations, proxy_residual_scale = huber_irls(
        x_train, effective_margin_train.to_numpy(dtype=float)
    )
    df["cn0_proxy_geometry_phase_dbhz"] = x_all @ proxy_beta - df["fspl_db"].to_numpy(float)

    prediction_specs = {
        MODEL_FSPL: ("physics_fspl_only_train_calibrated", "cn0_fspl_only_global_dbhz"),
        MODEL_PHYSICS: ("physics_only_signal_train_calibrated", "cn0_physics_only_signal_fspl_dbhz"),
        MODEL_PROXY: ("geometry_proxy_not_direct_physics", "cn0_proxy_geometry_phase_dbhz"),
    }
    for _, (_, column) in prediction_specs.items():
        df[f"error_{column}"] = df[column] - df["cn0_dbhz_mean"]

    metrics = pd.DataFrame(metric_rows(df, prediction_specs))

    parameters: list[dict] = []
    missing_terms = [
        ("P_tx_or_EIRP", "dBW or dBW-equivalent", "Missing per signal/satellite; EIRP would already include G_tx."),
        ("G_tx", "dBi", "Missing direction/frequency/SVN-dependent main- and side-lobe gain map."),
        ("G_rx", "dBi", "Missing LuGRE antenna pattern, attitude, and mounting transform."),
        ("T_sys_or_G_over_T", "K or dB/K", "Missing calibrated receiver/system noise temperature or G/T."),
        ("L_rx", "dB", "Missing cable, implementation, and receiver loss calibration."),
        ("L_atmos_iono_obstruction", "dB", "Geometry proxies exist, but calibrated propagation/obstruction losses do not."),
    ]
    for name, unit, notes in missing_terms:
        parameters.append(parameter_row(MODEL_ABSOLUTE, "required_absolute_link_budget_term", name, "", unit, "not_present_in_input_or_reliable_project_parameter", "missing_blocks_absolute_prediction", notes))
    parameters.append(parameter_row(MODEL_ABSOLUTE, "known_absolute_link_budget_term", "FSPL", "row_wise_fspl_db", "dB", input_path.as_posix(), "available", "Computed from geometric range and signal frequency."))
    parameters.append(parameter_row(MODEL_FSPL, "training_calibration", "alpha_global", global_alpha, "dB-Hz+dB", "training_median(cn0_dbhz_mean + fspl_db)", "estimated_train_only", "Single robust effective link-margin intercept."))
    for signal, alpha in sorted(signal_alphas.items()):
        parameters.append(parameter_row(MODEL_PHYSICS, "training_calibration", f"alpha_signal[{signal}]", alpha, "dB-Hz+dB", "training_median(cn0_dbhz_mean + fspl_db) within signal", "estimated_train_only", "Applied unchanged to held-out rows."))
    for phase, day in sorted(holdout_days.items()):
        parameters.append(parameter_row("all_calibrated_models", "data_split", f"holdout_day[{phase}]", day, "UTC date", "latest observed UTC day within mission phase", "fixed_before_fitting", "Every row on this phase-specific date is test-only."))
    for name, feature_center, feature_scale, feature_clip_low, feature_clip_high in zip(
        PROXY_CONTINUOUS, center, scale, clip_low, clip_high
    ):
        parameters.append(parameter_row(MODEL_PROXY, "proxy_feature_scaling", f"center[{name}]", feature_center, "native feature unit", "training median", "estimated_train_only", "No test rows used."))
        parameters.append(parameter_row(MODEL_PROXY, "proxy_feature_scaling", f"scale[{name}]", feature_scale, "native feature unit", "training IQR with standard-deviation fallback", "estimated_train_only", "No test rows used."))
        parameters.append(parameter_row(MODEL_PROXY, "proxy_feature_scaling", f"standardized_clip_low[{name}]", feature_clip_low, "standardized unit", "training 0.5 percentile bounded to include zero", "estimated_train_only", "Prevents unsupported test-range extrapolation."))
        parameters.append(parameter_row(MODEL_PROXY, "proxy_feature_scaling", f"standardized_clip_high[{name}]", feature_clip_high, "standardized unit", "training 99.5 percentile bounded to include zero", "estimated_train_only", "Prevents unsupported test-range extrapolation."))
    for name, beta in zip(proxy_parameter_names, proxy_beta):
        parameters.append(parameter_row(MODEL_PROXY, "huber_coefficient", name, beta, "dB per design unit", "Huber IRLS on training effective link margin", "estimated_train_only_proxy", "Proxy coefficient; do not interpret as measured antenna gain or physical loss."))
    parameters.append(parameter_row(MODEL_PROXY, "fit_diagnostic", "huber_iterations", proxy_iterations, "iterations", "deterministic IRLS", "diagnostic", "Maximum 100 iterations."))
    parameters.append(parameter_row(MODEL_PROXY, "fit_diagnostic", "robust_residual_scale", proxy_residual_scale, "dB-Hz", "1.4826 * training residual MAD", "diagnostic", "Training residual scale only."))
    parameter_df = pd.DataFrame(parameters)

    predictions_path = output_dir / "cn0_physics_baseline_predictions.csv"
    metrics_path = output_dir / "cn0_physics_baseline_metrics.csv"
    parameters_path = output_dir / "cn0_physics_baseline_parameters.csv"
    readme_path = output_dir / "README.md"

    output_df = df.copy()
    output_df["rx_utc"] = output_df["rx_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    output_df.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    parameter_df.to_csv(parameters_path, index=False)
    readme_path.write_text(
        build_readme(args.input, args.output_dir, df, holdout_days, metrics),
        encoding="utf-8",
    )

    print(f"input_rows={len(df)}")
    print(f"train_rows={int(train_mask.sum())}")
    print(f"test_rows={int(is_test.sum())}")
    print(f"holdout_days={holdout_days}")
    for row in metrics[(metrics['split'] == 'test') & (metrics['group_type'] == 'overall')].itertuples(index=False):
        if int(row.n) > 0:
            print(f"{row.model_name}: n={int(row.n)} mae={row.mae_dbhz:.3f} rmse={row.rmse_dbhz:.3f} p95={row.p95_ae_dbhz:.3f} bias={row.bias_dbhz:.3f}")
    print(f"wrote={predictions_path}")
    print(f"wrote={metrics_path}")
    print(f"wrote={parameters_path}")
    print(f"wrote={readme_path}")


if __name__ == "__main__":
    main()
