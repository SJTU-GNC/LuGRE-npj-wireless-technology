from __future__ import annotations

import json
import math
import sys
import warnings
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
OUT_DIR = ROOT / "table" / "algorithm" / "cn0_general_residual"
TARGET = "cn0_dbhz_mean"
BASELINE_COL = "cn0_constellation_physics_proxy_dbhz"
RANDOM_STATE = 42

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
    "row_id_constellation",
}

# Excluded to avoid a residual model that memorizes individual satellites or
# observing campaigns rather than learning constellation/frequency-transferable
# corrections.
NON_GENERALIZING_LABELS = {"svid", "op"}

NUMERIC_CANDIDATES = [
    BASELINE_COL,
    "fspl_db",
    "geometric_range_km",
    "tx_offboresight_deg",
    "earth_observer_altitude_km",
    "earth_limb_margin_deg",
    "earth_grazing_altitude_km",
    "earth_blocked",
    "moon_observer_altitude_km",
    "moon_limb_margin_deg",
    "moon_grazing_altitude_km",
    "moon_blocked",
    "earth_limb_proximity_proxy",
    "moon_limb_proximity_proxy",
    "system_noise_temperature_k",
    "ionosphere_path_length_shell_km",
    "m_ion_shell_chord_raw",
    "m_ion_proxy",
    "l_ion_abs_proxy_db",
    "l_ion_abs_budget_db",
    "gas_equivalent_airmass_km_raw",
    "gas_equivalent_airmass_km",
    "m_gas_proxy",
    "gas_gamma_surface_db_per_km",
    "l_gas_abs_proxy_db",
    "l_gas_abs_budget_db",
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
    "tx_ssv_double_sided_full_width_deg",
]

CATEGORICAL_CANDIDATES = [
    "system",
    "signal_name",
    "frequency_band",
    "mission_phase",
    "tx_yaw_quality",
    "tx_pattern_family",
    "tx_pattern_coverage",
    "tx_gnss_ssv_signal_band",
    "tx_ssv_main_lobe_classification",
]


def onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=8)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT, low_memory=False)
    df["rx_utc_dt"] = pd.to_datetime(df["rx_utc"], errors="coerce", utc=True)
    df["utc_day"] = df["rx_utc_dt"].dt.strftime("%Y-%m-%d")
    for col in set(NUMERIC_CANDIDATES + [TARGET, BASELINE_COL]):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("missing")
    df = df.dropna(subset=[TARGET, BASELINE_COL, "rx_utc_dt"]).copy()
    # Never learn from a physical baseline that contains an observation-fitted
    # legacy proxy. Direct link-budget rows are independent of the C/N0 target.
    if "constellation_physics_proxy_status" in df.columns:
        df = df[
            ~df["constellation_physics_proxy_status"].astype(str).str.contains(
                "fallback_legacy_proxy", case=False, na=False
            )
        ].copy()
    df["residual_target_db"] = df[TARGET] - df[BASELINE_COL]
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns and c not in FORBIDDEN_EXACT and c not in NON_GENERALIZING_LABELS]
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in df.columns and c not in FORBIDDEN_EXACT and c not in NON_GENERALIZING_LABELS]
    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))


def validation_mask(train_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=train_df.index)
    for _, g in train_df.groupby(["mission_phase", "signal_name"], dropna=False):
        days = sorted(g["utc_day"].dropna().unique())
        if len(days) >= 2:
            mask.loc[g[g["utc_day"].isin(days[-1:])].index] = True
        else:
            n = max(1, int(math.ceil(0.2 * len(g))))
            mask.loc[g.sort_values("rx_utc_dt").tail(n).index] = True
    return mask


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
        ("ridge_residual", "linear", Ridge(alpha=8.0)),
        (
            "random_forest_residual",
            "bagging",
            RandomForestRegressor(
                n_estimators=260,
                min_samples_leaf=8,
                max_features=0.75,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gradient_boosting_residual",
            "gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=360,
                learning_rate=0.035,
                max_leaf_nodes=31,
                l2_regularization=0.08,
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
                    n_estimators=520,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=35,
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
                    iterations=420,
                    depth=6,
                    learning_rate=0.035,
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
    return {
        "n": int(len(y_true)),
        "mae_dbhz": float(mean_absolute_error(y_true, y_pred)),
        "rmse_dbhz": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae_dbhz": float(np.median(ae)),
        "p95_ae_dbhz": float(np.percentile(ae, 95)),
        "bias_dbhz": float(np.mean(err)),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
    }


def append_metrics(rows: list[dict[str, Any]], df: pd.DataFrame, y_pred: np.ndarray, model_name: str, model_class: str, eval_split: str) -> None:
    pred_s = pd.Series(y_pred, index=df.index)
    groups = [("overall", "all", df.index)]
    for col in ["system", "signal_name", "mission_phase", "constellation_physics_proxy_status"]:
        if col in df.columns:
            for val, g in df.groupby(col, dropna=False):
                groups.append((col, val, g.index))
    for group_type, group_value, idx in groups:
        yt = df.loc[idx, TARGET].to_numpy(float)
        yp = pred_s.loc[idx].to_numpy(float)
        rows.append(
            {
                "model_name": model_name,
                "model_class": model_class,
                "feature_mode": "global_signal_aware_no_svid_no_op",
                "evaluation_split": eval_split,
                "group_type": group_type,
                "group_value": str(group_value),
                **metric_dict(yt, yp),
            }
        )


def prediction_frame(df: pd.DataFrame, residual_pred: np.ndarray, model_name: str, model_class: str, eval_split: str) -> pd.DataFrame:
    pred = df[BASELINE_COL].to_numpy(float) + residual_pred
    return pd.DataFrame(
        {
            "row_id_constellation": df["row_id_constellation"].to_numpy() if "row_id_constellation" in df else df.index.to_numpy(),
            "rx_utc": df["rx_utc"].to_numpy(),
            "split": df["split"].to_numpy(),
            "evaluation_split": eval_split,
            "feature_mode": "global_signal_aware_no_svid_no_op",
            "model_name": model_name,
            "model_class": model_class,
            "system": df["system"].to_numpy(),
            "svid": df["svid"].to_numpy() if "svid" in df else "",
            "signal_name": df["signal_name"].to_numpy(),
            "mission_phase": df["mission_phase"].to_numpy(),
            "constellation_physics_proxy_status": df["constellation_physics_proxy_status"].to_numpy(),
            "cn0_obs_dbhz": df[TARGET].to_numpy(float),
            "cn0_baseline_dbhz": df[BASELINE_COL].to_numpy(float),
            "residual_true_db": df["residual_target_db"].to_numpy(float),
            "residual_pred_db": residual_pred,
            "cn0_ai_pred_dbhz": pred,
            "signed_error_dbhz": pred - df[TARGET].to_numpy(float),
            "abs_error_dbhz": np.abs(pred - df[TARGET].to_numpy(float)),
            "generalization_guard_status": "single_global_model_all_systems_signals;system_signal_frequency_allowed;svid_op_excluded",
        }
    )


def feature_importance(pipe: Pipeline, model_name: str) -> pd.DataFrame:
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
        return pd.DataFrame()
    n = min(len(names), len(vals))
    order = np.argsort(vals[:n])[::-1]
    return pd.DataFrame(
        [
            {
                "feature_mode": "global_signal_aware_no_svid_no_op",
                "model_name": model_name,
                "feature_name": names[i],
                "importance_value": float(vals[i]),
                "importance_rank": rank,
            }
            for rank, i in enumerate(order[:120], start=1)
        ]
    )


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_data()
    numeric, categorical = feature_columns(df)
    tree_pre, scaled_pre = preprocessors(numeric, categorical)

    train_full = df[df["split"].eq("train")].copy()
    test_df = df[df["split"].eq("test")].copy()
    val_mask = validation_mask(train_full)
    train_core = train_full.loc[~val_mask].copy()
    val_df = train_full.loc[val_mask].copy()

    metrics_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []

    for split_name, eval_df in [("validation", val_df), ("test", test_df)]:
        zero = np.zeros(len(eval_df), dtype=float)
        append_metrics(metrics_rows, eval_df, eval_df[BASELINE_COL].to_numpy(float), "no_ai_physics_baseline", "physics", split_name)
        pred_frames.append(prediction_frame(eval_df, zero, "no_ai_physics_baseline", "physics", split_name))

    best_val: tuple[float, str, str, Any] | None = None
    for model_name, model_class, estimator in model_specs():
        pipe = make_pipeline(model_name, estimator, tree_pre, scaled_pre)
        pipe.fit(train_core[numeric + categorical], train_core["residual_target_db"].to_numpy(float))
        val_resid = np.asarray(pipe.predict(val_df[numeric + categorical]), dtype=float)
        val_pred = val_df[BASELINE_COL].to_numpy(float) + val_resid
        val_metric = metric_dict(val_df[TARGET].to_numpy(float), val_pred)
        comparison_rows.append({"evaluation_split": "validation", "model_name": model_name, "model_class": model_class, **val_metric})
        append_metrics(metrics_rows, val_df, val_pred, model_name, model_class, "validation")
        pred_frames.append(prediction_frame(val_df, val_resid, model_name, model_class, "validation"))
        if best_val is None or val_metric["rmse_dbhz"] < best_val[0]:
            best_val = (val_metric["rmse_dbhz"], model_name, model_class, estimator)

    if best_val is None:
        raise RuntimeError("No residual model was trained.")

    _, best_name, best_class, best_estimator = best_val
    final_pipe = make_pipeline(best_name, best_estimator, tree_pre, scaled_pre)
    final_pipe.fit(train_full[numeric + categorical], train_full["residual_target_db"].to_numpy(float))
    test_resid = np.asarray(final_pipe.predict(test_df[numeric + categorical]), dtype=float)
    test_pred = test_df[BASELINE_COL].to_numpy(float) + test_resid
    test_metric = metric_dict(test_df[TARGET].to_numpy(float), test_pred)
    comparison_rows.append({"evaluation_split": "test_best_from_validation", "model_name": best_name, "model_class": best_class, **test_metric})
    append_metrics(metrics_rows, test_df, test_pred, best_name, best_class, "test_best_from_validation")
    pred_frames.append(prediction_frame(test_df, test_resid, best_name, best_class, "test_best_from_validation"))
    importances.append(feature_importance(final_pipe, best_name))

    split_summary = pd.DataFrame(
        [
            {
                "split_name": name,
                "n": len(part),
                "start_utc": part["rx_utc_dt"].min().isoformat() if len(part) else "",
                "end_utc": part["rx_utc_dt"].max().isoformat() if len(part) else "",
                "signals": "|".join(sorted(part["signal_name"].astype(str).unique())) if len(part) else "",
                "systems": "|".join(sorted(part["system"].astype(str).unique())) if len(part) else "",
            }
            for name, part in [("train_core", train_core), ("validation", val_df), ("train_full", train_full), ("test", test_df)]
        ]
    )
    inventory = pd.DataFrame(
        [
            {"kind": "numeric", "feature_name": c}
            for c in numeric
        ]
        + [{"kind": "categorical", "feature_name": c} for c in categorical]
    )
    pred = pd.concat(pred_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    importance_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return pred, metrics_df, comparison_df, importance_df, pd.concat([split_summary, inventory], ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred, metrics_df, comparison_df, importance_df, summary_df = run()
    pred.to_csv(OUT_DIR / "cn0_general_residual_predictions.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "cn0_general_residual_metrics.csv", index=False)
    comparison_df.to_csv(OUT_DIR / "cn0_general_residual_model_comparison.csv", index=False)
    importance_df.to_csv(OUT_DIR / "cn0_general_residual_feature_importance.csv", index=False)
    summary_df.to_csv(OUT_DIR / "cn0_general_residual_split_and_feature_summary.csv", index=False)
    best = comparison_df[comparison_df["evaluation_split"].eq("test_best_from_validation")].sort_values("rmse_dbhz").head(1)
    model_card = {
        "objective": "Single global residual model for all available GNSS systems/signals while excluding SVID and OP memorization.",
        "baseline": BASELINE_COL,
        "target": TARGET,
        "feature_mode": "global_signal_aware_no_svid_no_op",
        "best_test": best.to_dict("records"),
    }
    (OUT_DIR / "GeneralResidualModelCard.json").write_text(json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
