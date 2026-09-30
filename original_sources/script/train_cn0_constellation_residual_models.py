from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None

INPUT = ROOT / "table" / "algorithm" / "cn0_constellation_physics_baseline" / "cn0_constellation_physics_predictions.csv"
OUT_DIR = ROOT / "table" / "algorithm" / "cn0_constellation_ai_residual"

TARGET = "cn0_dbhz_mean"
BASELINE_COL = "cn0_constellation_physics_proxy_dbhz"
RANDOM_STATE = 42
OLD_B1_RMSE_DBHZ = 1.615532


STRICT_FORBIDDEN = {"system", "signal_name", "signal_id", "svid", "frequency_mhz"}
FORBIDDEN_EXACT = {
    TARGET,
    "pseudorange_raw_m_mean",
    "doppler_raw_hz_mean",
    "split",
    "split_policy",
    "calibration_target_used",
    "heldout_for_validation",
    "rx_utc",
    "rx_gps_seconds",
    "time_bin_gps_seconds",
    "utc_day",
}


BASE_NUMERIC = [
    BASELINE_COL,
    "cn0_constellation_direct_available_dbhz",
    "cn0_galileo_grap_peakrx_dbhz",
    "cn0_galileo_grap_upper95_peakrx_dbhz",
    "cn0_galileo_grap_lower95_peakrx_dbhz",
    "cn0_reference_trajectory_2d_dbhz",
    "fspl_db",
    "geometric_range_km",
    "range_rate_rx_only_km_s",
    "tx_offboresight_deg",
    "earth_observer_altitude_km",
    "earth_limb_margin_deg",
    "earth_grazing_altitude_km",
    "earth_blocked",
    "moon_observer_altitude_km",
    "moon_limb_margin_deg",
    "moon_grazing_altitude_km",
    "moon_blocked",
    "nav_age_seconds",
    "samples",
    "earth_limb_proximity_proxy",
    "moon_limb_proximity_proxy",
    "tx_eirp_grap_lookup_coelevation_deg",
    "tx_eirp_grap_azimuth_median_dbw",
    "tx_eirp_grap_azimuth_mean_dbw",
    "tx_eirp_grap_azimuth_min_dbw",
    "tx_eirp_grap_azimuth_max_dbw",
    "tx_eirp_grap_upper95_median_dbw",
    "tx_eirp_grap_lower95_median_dbw",
    "tx_eirp_grap_azimuth_samples",
    "rx_peak_gain_dbic",
    "system_noise_temperature_k",
    "implementation_loss_assumed_db",
    "ionosphere_shell_lower_alt_km",
    "ionosphere_shell_upper_alt_km",
    "ionosphere_l30_vertical_assumed_db",
    "ionosphere_path_length_shell_km",
    "m_ion_shell_chord_raw",
    "m_ion_proxy",
    "l_ion_abs_proxy_db",
    "l_ion_abs_budget_db",
    "troposphere_shell_upper_alt_km",
    "gas_scale_height_assumed_km",
    "gas_vertical_equivalent_airmass_km",
    "gas_equivalent_airmass_km_raw",
    "gas_equivalent_airmass_km",
    "m_gas_proxy",
    "gas_gamma_surface_db_per_km",
    "l_gas_abs_proxy_db",
    "l_gas_abs_budget_db",
    "edge_near_limb_alt_threshold_km",
    "neutral_refraction_upper_alt_km",
    "earth_edge_near_limb_0_20km",
    "moon_edge_near_limb_0_20km",
    "earth_neutral_refraction_20_60km",
    "moon_neutral_refraction_20_60km",
    "rx_offboresight_spice_deg",
    "rx_azimuth_spice_deg",
    "rx_gain_envelope_dbic",
    "tx_theta_body_deg",
    "tx_phi_body_deg",
    "tx_gain_2d_db",
    "tx_power_dbw",
    "frequency_mhz",
    "tx_ssv_main_lobe_boundary_deg",
    "tx_ssv_signed_lower_boundary_deg",
    "tx_ssv_signed_upper_boundary_deg",
    "tx_ssv_double_sided_full_width_deg",
]

BASE_CATEGORICAL = [
    "mission_phase",
    "op",
    "orbit_geometry_status",
    "fspl_status",
    "occultation_status",
    "limb_margin_status",
    "tx_gain_status",
    "rx_gain_status",
    "atmos_iono_status",
    "frequency_band",
    "galileo_grap_lookup_status",
    "galileo_grap_coordinate_mapping_status",
    "gps_l1_direct_merge_status",
    "gps_tx_physics_status",
    "receiver_gain_model_status",
    "constellation_direct_status",
    "constellation_physics_proxy_status",
    "tx_yaw_quality",
    "tx_pattern_source",
    "tx_phi_alignment_quality",
    "tx_gnss_ssv_signal_band",
    "tx_ssv_main_lobe_classification",
    "tx_ssv_main_lobe_boundary_basis",
    "tx_ssv_classification_semantics",
    "tx_central_half_power_classification",
    "ionosphere_absorption_proxy_status",
    "gas_absorption_proxy_status",
    "edge_ai_feature_status",
    "cn0_2d_model_status",
    "system",
    "signal_name",
    "signal_id",
    "svid",
]


@dataclass(frozen=True)
class FeatureMode:
    name: str
    description: str
    allow_constellation_labels: bool


FEATURE_MODES = [
    FeatureMode(
        "physics_strict",
        "No direct system/signal/svid/frequency shortcuts; uses physical lookup fields and status flags.",
        False,
    ),
    FeatureMode(
        "category_assisted",
        "Diagnostic upper bound with system/signal/svid/frequency retained.",
        True,
    ),
]


def onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT, low_memory=False)
    df["rx_utc_dt"] = pd.to_datetime(df["rx_utc"], errors="coerce", utc=True)
    df["utc_day"] = df["rx_utc_dt"].dt.strftime("%Y-%m-%d")
    for col in set(BASE_NUMERIC + [TARGET, BASELINE_COL]):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BASE_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("missing")
    df = df.dropna(subset=[TARGET, BASELINE_COL, "rx_utc_dt"]).copy()
    df["residual_target_db"] = df[TARGET] - df[BASELINE_COL]
    return df


def validation_mask(train_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=train_df.index)
    for _, g in train_df.groupby("mission_phase", dropna=False):
        days = sorted(g["utc_day"].dropna().unique())
        if len(days) >= 2:
            mask.loc[g[g["utc_day"].isin(days[-1:])].index] = True
        else:
            n = max(1, int(math.ceil(0.2 * len(g))))
            mask.loc[g.sort_values("rx_utc_dt").tail(n).index] = True
    return mask


def feature_columns(df: pd.DataFrame, mode: FeatureMode) -> tuple[list[str], list[str]]:
    numeric = [c for c in BASE_NUMERIC if c in df.columns and c not in FORBIDDEN_EXACT]
    categorical = [c for c in BASE_CATEGORICAL if c in df.columns and c not in FORBIDDEN_EXACT]
    if not mode.allow_constellation_labels:
        numeric = [c for c in numeric if c not in STRICT_FORBIDDEN]
        categorical = [c for c in categorical if c not in STRICT_FORBIDDEN]
    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))


def preprocessors(numeric: list[str], categorical: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    tree_pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", onehot())]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    scaled_pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", onehot())]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return tree_pre, scaled_pre


def model_specs() -> list[tuple[str, str, Any]]:
    specs: list[tuple[str, str, Any]] = [
        ("ridge_residual", "linear", Ridge(alpha=5.0)),
        (
            "hist_gradient_boosting_residual",
            "gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=280,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    if LGBMRegressor is not None:
        specs.append(
            (
                "lightgbm_residual",
                "gradient_boosting",
                LGBMRegressor(
                    n_estimators=460,
                    learning_rate=0.035,
                    num_leaves=31,
                    min_child_samples=25,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="huber",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    if CatBoostRegressor is not None:
        specs.append(
            (
                "catboost_residual",
                "gradient_boosting",
                CatBoostRegressor(
                    iterations=360,
                    depth=6,
                    learning_rate=0.04,
                    loss_function="Huber:delta=1.5",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                ),
            )
        )
    return specs


def make_pipeline(model_name: str, estimator: Any, tree_pre: ColumnTransformer, scaled_pre: ColumnTransformer) -> Pipeline:
    pre = scaled_pre if model_name.startswith("ridge") else tree_pre
    return Pipeline([("preprocess", pre), ("model", estimator)])


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ae = np.abs(err)
    r2 = np.nan if len(y_true) < 2 or np.std(y_true) == 0 else r2_score(y_true, y_pred)
    corr = np.nan if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0 else float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "n": int(len(y_true)),
        "mae_dbhz": float(mean_absolute_error(y_true, y_pred)),
        "rmse_dbhz": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae_dbhz": float(np.median(ae)),
        "p95_ae_dbhz": float(np.percentile(ae, 95)),
        "bias_dbhz": float(np.mean(err)),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "pearson_r": corr,
    }


def availability(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    truth = y_true >= threshold
    pred = y_pred >= threshold
    logit = np.clip((y_pred - threshold) / 2.0, -50, 50)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return {
        "availability_threshold_dbhz": threshold,
        "availability_accuracy": float(np.mean(truth == pred)),
        "availability_f1": float(f1_score(truth, pred, zero_division=0)),
        "availability_brier": float(np.mean((prob - truth.astype(float)) ** 2)),
    }


def append_metrics(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
    model_class: str,
    feature_mode: str,
    eval_split: str,
) -> None:
    pred_s = pd.Series(y_pred, index=df.index)
    groups = [("overall", "all", df.index)]
    for col in ["system", "signal_name", "mission_phase", "svid", "constellation_physics_proxy_status"]:
        if col in df.columns:
            for val, g in df.groupby(col, dropna=False):
                groups.append((col, val, g.index))
    for group_type, group_value, idx in groups:
        if len(idx) == 0:
            continue
        yt = df.loc[idx, TARGET].to_numpy(float)
        yp = pred_s.loc[idx].to_numpy(float)
        base = {
            "feature_mode": feature_mode,
            "model_name": model_name,
            "model_class": model_class,
            "evaluation_split": eval_split,
            "group_type": group_type,
            "group_value": group_value,
            **metric_dict(yt, yp),
        }
        for th in [20.0, 23.0, 30.0]:
            rows.append({**base, **availability(yt, yp, th)})


def prediction_frame(df: pd.DataFrame, residual_pred: np.ndarray, model_name: str, model_class: str, feature_mode: str, eval_split: str) -> pd.DataFrame:
    cn0_pred = df[BASELINE_COL].to_numpy(float) + residual_pred
    return pd.DataFrame(
        {
            "row_id_constellation": df["row_id_constellation"].to_numpy() if "row_id_constellation" in df else df.index.to_numpy(),
            "rx_utc": df["rx_utc"].to_numpy(),
            "rx_gps_seconds": df["rx_gps_seconds"].to_numpy() if "rx_gps_seconds" in df else np.nan,
            "split": df["split"].to_numpy(),
            "evaluation_split": eval_split,
            "feature_mode": feature_mode,
            "model_name": model_name,
            "model_class": model_class,
            "system": df["system"].to_numpy() if "system" in df else "",
            "svid": df["svid"].to_numpy() if "svid" in df else "",
            "signal_name": df["signal_name"].to_numpy() if "signal_name" in df else "",
            "mission_phase": df["mission_phase"].to_numpy() if "mission_phase" in df else "",
            "constellation_physics_proxy_status": df["constellation_physics_proxy_status"].to_numpy(),
            "cn0_obs_dbhz": df[TARGET].to_numpy(float),
            "cn0_constellation_baseline_dbhz": df[BASELINE_COL].to_numpy(float),
            "residual_true_db": df["residual_target_db"].to_numpy(float),
            "residual_pred_db": residual_pred,
            "cn0_ai_pred_dbhz": cn0_pred,
            "signed_error_dbhz": cn0_pred - df[TARGET].to_numpy(float),
            "abs_error_dbhz": np.abs(cn0_pred - df[TARGET].to_numpy(float)),
            "leakage_guard_status": "no_cn0_or_error_columns;blocked_time_split;physics_strict_removes_constellation_shortcuts"
            if feature_mode == "physics_strict"
            else "no_cn0_or_error_columns;blocked_time_split;category_assisted_diagnostic",
        }
    )


def feature_importance(pipe: Pipeline, model_name: str, feature_mode: str) -> list[dict[str, Any]]:
    model = pipe.named_steps["model"]
    try:
        names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        names = []
    vals = None
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.ravel(model.coef_))
    if vals is None or not names:
        return []
    n = min(len(names), len(vals))
    order = np.argsort(vals[:n])[::-1]
    return [
        {
            "feature_mode": feature_mode,
            "model_name": model_name,
            "feature_name": names[i],
            "importance_value": float(vals[i]),
            "importance_rank": rank,
            "importance_type": "model_native_feature_importance_or_abs_coef",
        }
        for rank, i in enumerate(order[:100], start=1)
    ]


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    df = load_data()
    train_full = df[df["split"].eq("train")].copy()
    test_df = df[df["split"].eq("test")].copy()
    val_mask = validation_mask(train_full)
    train_core = train_full.loc[~val_mask].copy()
    val_df = train_full.loc[val_mask].copy()

    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    feature_inventory: dict[str, list[str]] = {}

    split_summary = pd.DataFrame(
        [
            {
                "split_name": name,
                "n": len(part),
                "start_utc": part["rx_utc_dt"].min().isoformat() if len(part) else "",
                "end_utc": part["rx_utc_dt"].max().isoformat() if len(part) else "",
                "utc_days": int(part["utc_day"].nunique()) if len(part) else 0,
                "signals": "|".join(sorted(part["signal_name"].astype(str).unique())) if len(part) else "",
                "mission_phases": "|".join(sorted(part["mission_phase"].astype(str).unique())) if len(part) else "",
                "split_policy": "existing train/test plus last train UTC day per mission phase as blocked validation",
            }
            for name, part in [
                ("train_core", train_core),
                ("validation_blocked_within_train", val_df),
                ("train_full", train_full),
                ("test_existing_holdout", test_df),
            ]
        ]
    )

    for mode in FEATURE_MODES:
        numeric, categorical = feature_columns(df, mode)
        features = numeric + categorical
        feature_inventory[mode.name] = features
        tree_pre, scaled_pre = preprocessors(numeric, categorical)

        for eval_name, eval_df in [("validation", val_df), ("test", test_df)]:
            baseline_pred = eval_df[BASELINE_COL].to_numpy(float)
            pred_rows.append(
                prediction_frame(eval_df, np.zeros(len(eval_df)), "no_ai_constellation_physics_baseline", "physics_baseline", mode.name, eval_name)
            )
            append_metrics(metric_rows, eval_df, baseline_pred, "no_ai_constellation_physics_baseline", "physics_baseline", mode.name, eval_name)
            comparison_rows.append(
                {
                    "feature_mode": mode.name,
                    "model_name": "no_ai_constellation_physics_baseline",
                    "model_class": "physics_baseline",
                    "selection_split": "test_after_full_train_not_for_selection" if eval_name == "test" else "validation",
                    "status": "evaluated_no_ai_reference",
                    **metric_dict(eval_df[TARGET].to_numpy(float), baseline_pred),
                }
            )

        best_val: dict[str, Any] | None = None
        for model_name, model_class, estimator in model_specs():
            pipe = make_pipeline(model_name, estimator, tree_pre, scaled_pre)
            try:
                pipe.fit(train_core[features], train_core["residual_target_db"].to_numpy(float))
                val_resid = np.asarray(pipe.predict(val_df[features]), dtype=float)
                val_pred = val_df[BASELINE_COL].to_numpy(float) + val_resid
                val_m = metric_dict(val_df[TARGET].to_numpy(float), val_pred)
                pred_rows.append(prediction_frame(val_df, val_resid, model_name, model_class, mode.name, "validation"))
                append_metrics(metric_rows, val_df, val_pred, model_name, model_class, mode.name, "validation")
                comparison_rows.append(
                    {
                        "feature_mode": mode.name,
                        "model_name": model_name,
                        "model_class": model_class,
                        "selection_split": "validation",
                        "status": "evaluated",
                        **val_m,
                    }
                )
                if best_val is None or val_m["rmse_dbhz"] < best_val["rmse_dbhz"]:
                    best_val = {"model_name": model_name, "model_class": model_class, **val_m}
            except Exception as exc:
                comparison_rows.append(
                    {
                        "feature_mode": mode.name,
                        "model_name": model_name,
                        "model_class": model_class,
                        "selection_split": "validation",
                        "status": f"failed_validation_fit:{type(exc).__name__}:{exc}",
                    }
                )
                continue

            test_pipe = make_pipeline(model_name, estimator, tree_pre, scaled_pre)
            try:
                test_pipe.fit(train_full[features], train_full["residual_target_db"].to_numpy(float))
                test_resid = np.asarray(test_pipe.predict(test_df[features]), dtype=float)
                test_pred = test_df[BASELINE_COL].to_numpy(float) + test_resid
                test_m = metric_dict(test_df[TARGET].to_numpy(float), test_pred)
                pred_rows.append(prediction_frame(test_df, test_resid, model_name, model_class, mode.name, "test"))
                append_metrics(metric_rows, test_df, test_pred, model_name, model_class, mode.name, "test")
                comparison_rows.append(
                    {
                        "feature_mode": mode.name,
                        "model_name": model_name,
                        "model_class": model_class,
                        "selection_split": "test_after_full_train_not_for_selection",
                        "status": "evaluated",
                        **test_m,
                    }
                )
                importance_rows.extend(feature_importance(test_pipe, model_name, mode.name))
            except Exception as exc:
                comparison_rows.append(
                    {
                        "feature_mode": mode.name,
                        "model_name": model_name,
                        "model_class": model_class,
                        "selection_split": "test_after_full_train_not_for_selection",
                        "status": f"failed_test_fit:{type(exc).__name__}:{exc}",
                    }
                )

    return (
        pd.concat(pred_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(comparison_rows),
        pd.DataFrame(importance_rows),
        split_summary,
        feature_inventory,
    )


def write_model_card(comparison: pd.DataFrame, split_summary: pd.DataFrame, feature_inventory: dict[str, list[str]]) -> None:
    test = comparison[
        (comparison["selection_split"].eq("test_after_full_train_not_for_selection"))
        & (comparison["status"].astype(str).str.startswith("evaluated"))
    ].copy()
    best_resid = test[test["status"].eq("evaluated")].sort_values("rmse_dbhz").head(1)
    strict = test[(test["feature_mode"].eq("physics_strict")) & (test["status"].eq("evaluated"))].sort_values("rmse_dbhz").head(1)
    assisted = test[(test["feature_mode"].eq("category_assisted")) & (test["status"].eq("evaluated"))].sort_values("rmse_dbhz").head(1)
    no_ai = test[test["status"].eq("evaluated_no_ai_reference")].head(1)
    lines = [
        "# Constellation AI Residual Model Card",
        "",
        "Objective: train residual models after moving available GPS/Galileo transmit-side physics into a constellation-aware baseline.",
        "",
        "## Split",
        "",
        split_summary.to_csv(index=False),
        "",
        "## Best Rows",
        "",
    ]
    for label, frame in [("best_residual", best_resid), ("physics_strict", strict), ("category_assisted", assisted), ("no_ai", no_ai)]:
        if not frame.empty:
            r = frame.iloc[0]
            lines.append(
                f"- {label}: `{r['feature_mode']}` / `{r['model_name']}` RMSE={float(r['rmse_dbhz']):.3f}, MAE={float(r['mae_dbhz']):.3f}, P95={float(r['p95_ae_dbhz']):.3f}, bias={float(r['bias_dbhz']):.3f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Previous B1 CatBoost residual RMSE reference: {OLD_B1_RMSE_DBHZ:.3f} dB-Hz.",
            "The constellation baseline includes Galileo GRAP only as azimuth-aggregated diagnostic EIRP until body/yaw-frame azimuth mapping is validated.",
            "GPS direct transmit physics is row-wise only for the existing GPS L1 2-D direct subset; unsupported GPS rows keep explicit proxy status.",
            "Physics-strict mode removes direct `system`, `signal_name`, `signal_id`, `svid`, and `frequency_mhz` shortcuts.",
            "",
            "## Feature Inventory",
            "",
            "```json",
            json.dumps(feature_inventory, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    (OUT_DIR / "ConstellationAIResidualModelCard.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred, metrics_df, comparison_df, importance_df, split_summary, feature_inventory = run()
    pred.to_csv(OUT_DIR / "cn0_constellation_ai_residual_predictions.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "cn0_constellation_ai_residual_metrics.csv", index=False)
    comparison_df.to_csv(OUT_DIR / "cn0_constellation_ai_residual_model_comparison.csv", index=False)
    importance_df.to_csv(OUT_DIR / "cn0_constellation_ai_residual_feature_importance.csv", index=False)
    split_summary.to_csv(OUT_DIR / "cn0_constellation_ai_residual_split_summary.csv", index=False)
    write_model_card(comparison_df, split_summary, feature_inventory)
    test = comparison_df[
        (comparison_df.selection_split == "test_after_full_train_not_for_selection")
        & (comparison_df.status.astype(str).str.startswith("evaluated"))
    ].sort_values(["rmse_dbhz", "mae_dbhz"])
    print(test.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
