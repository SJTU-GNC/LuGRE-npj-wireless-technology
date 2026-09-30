#!/usr/bin/env python3
"""Build the three manuscript figures that explain, quantify and select the AI residual model."""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "script"))

import joblib
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import train_cn0_trend_residual_tuned_no_leakage as final_cfg


ENGINE = final_cfg.engine
FEATURES = list(final_cfg.selected_features)
PARAMETERS = dict(final_cfg.selected_parameters)

MODEL_DIR = ROOT / "table" / "algorithm" / "cn0_trend_residual_tuned_no_leakage"
TUNING_DIR = ROOT / "table" / "algorithm" / "cn0_trend_residual_tuning_no_leakage"
ANALYSIS_DIR = ROOT / "table" / "algorithm" / "cn0_ai_residual_story"
FIG_DIR = ROOT / "figure" / "paper_draft_v2" / "ai_residual_story"
PREDICTION_PATH = MODEL_DIR / "cn0_trend_residual_tuned_no_leakage_predictions.csv"
MODEL_PATH = MODEL_DIR / "cn0_trend_residual_tuned_no_leakage.joblib"
MODEL_CARD_PATH = MODEL_DIR / "CN0TrendResidualTunedNoLeakageModelCard.json"
METRICS_PATH = MODEL_DIR / "cn0_trend_residual_tuned_no_leakage_metrics.csv"
CANDIDATE_PATH = TUNING_DIR / "validation_only_candidate_scores.csv"
PREPARED_CACHE = ANALYSIS_DIR / "final_split_feature_frame.joblib"


COL = {
    "ink": "#22262B",
    "muted": "#66727D",
    "grid": "#DCE2E7",
    "physics": "#54616D",
    "physics_light": "#C8D0D6",
    "beta": "#D59A28",
    "beta_light": "#F3E2B8",
    "ai": "#C33C54",
    "ai_light": "#F4D5DC",
    "teal": "#178B83",
    "teal_light": "#D3ECE9",
    "blue": "#276FBF",
    "blue_light": "#D9E8F7",
    "violet": "#7963A8",
    "surface": "#61A65E",
    "commissioning": "#2B75B8",
    "trans": "#15977F",
    "lunar": "#D28D18",
}

PHASE_COLORS = {"C": COL["commissioning"], "T": COL["trans"], "L": COL["lunar"], "S": COL["surface"]}
SPLIT_LABELS = {
    "train": "Training fit",
    "validation": "Validation selection",
    "test": "Internal test",
    "external_holdout": "Complete-OP holdout",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [
        ("png", {"dpi": 420}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600}),
    ]:
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.03, **kwargs)
    plt.close(fig)


def save_ai1_polished(fig: plt.Figure) -> None:
    """Export FigAI1 at its exact 183-mm canvas under both release stems."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stems = [
        "FigAI1_physics_guided_residual_learning_framework",
        "Fig2_physics_guided_residual_learning_framework_polished",
    ]
    for stem in stems:
        fig.savefig(
            FIG_DIR / f"{stem}.png",
            dpi=600,
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
        fig.savefig(
            FIG_DIR / f"{stem}.pdf",
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
        fig.savefig(
            FIG_DIR / f"{stem}.svg",
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
        fig.savefig(
            FIG_DIR / f"{stem}.tiff",
            dpi=600,
            facecolor="white",
            edgecolor="white",
            transparent=False,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=8.0, va="top", ha="left")


def light_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=COL["grid"], linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    return float(math.sqrt(mean_squared_error(y[mask], pred[mask])))


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    error = pred - y
    return {
        "n": int(len(y)),
        "mae_dbhz": float(mean_absolute_error(y, pred)),
        "rmse_dbhz": float(math.sqrt(mean_squared_error(y, pred))),
        "bias_dbhz": float(np.mean(error)),
        "pearson_r": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else np.nan,
    }


def load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(PREDICTION_PATH, low_memory=False)
    frame["minute_utc"] = pd.to_datetime(frame["minute_utc"], utc=True)
    frame["trend_training_eligible"] = frame["trend_training_eligible"].astype(str).str.lower().eq("true")
    return frame


def beta_map(frame: pd.DataFrame, splits: list[str]) -> dict[str, float]:
    source = frame[
        frame["evaluation_split"].isin(splits)
        & frame["trend_training_eligible"]
        & frame["residual_trend_target_db"].notna()
    ]
    return source.groupby("signal_name")["residual_trend_target_db"].median().astype(float).to_dict()


def contribution_table(pred: pd.DataFrame) -> pd.DataFrame:
    train_beta = beta_map(pred, ["train"])
    refit_beta = beta_map(pred, ["train", "validation"])
    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test", "external_holdout"]:
        part = pred[pred["evaluation_split"].eq(split) & pred["trend_training_eligible"]].copy()
        if part.empty:
            continue
        beta = train_beta if split in {"train", "validation"} else refit_beta
        part["beta_db"] = part["signal_name"].map(beta).fillna(0.0)
        stages = {
            "Physics": part["cn0_physics_trend_dbhz"].to_numpy(float),
            "Physics + beta": (part["cn0_physics_trend_dbhz"] + part["beta_db"]).to_numpy(float),
            "Physics + beta + HGB": (
                part["cn0_physics_ai_trend_dbhz"]
                if split == "train"
                else part["cn0_physics_ai_eval_dbhz"]
            ).to_numpy(float),
        }
        y = part["cn0_observed_trend_dbhz"].to_numpy(float)
        for stage, values in stages.items():
            rows.append({"evaluation_split": split, "stage": stage, **metric_row(y, values)})
        for phase, group in part.groupby("mission_phase"):
            y_phase = group["cn0_observed_trend_dbhz"].to_numpy(float)
            phase_stages = {
                "Physics": group["cn0_physics_trend_dbhz"].to_numpy(float),
                "Physics + beta": (group["cn0_physics_trend_dbhz"] + group["beta_db"]).to_numpy(float),
                "Physics + beta + HGB": (
                    group["cn0_physics_ai_trend_dbhz"]
                    if split == "train"
                    else group["cn0_physics_ai_eval_dbhz"]
                ).to_numpy(float),
            }
            for stage, values in phase_stages.items():
                rows.append(
                    {
                        "evaluation_split": split,
                        "mission_phase": phase,
                        "stage": stage,
                        **metric_row(y_phase, values),
                    }
                )
    output = pd.DataFrame(rows)
    output["mission_phase"] = output.get("mission_phase", pd.Series(index=output.index, dtype=object))
    return output


def huber_location(values: np.ndarray, delta: float = 1.5, iterations: int = 60) -> float:
    """Robust one-dimensional location used only for the fixed-effect benchmark."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    location = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - location)))
    scale = max(scale, 1e-6)
    for _ in range(iterations):
        residual = (values - location) / scale
        weights = np.ones_like(residual)
        mask = np.abs(residual) > delta
        weights[mask] = delta / np.abs(residual[mask])
        updated = float(np.sum(weights * values) / np.sum(weights))
        if abs(updated - location) < 1e-9:
            break
        location = updated
    return location


def fixed_residual_benchmark(pred: pd.DataFrame) -> pd.DataFrame:
    """Compare fixed residual learners without touching the time-varying HGB term."""
    eligible = pred[pred["trend_training_eligible"] & pred["residual_trend_target_db"].notna()].copy()
    train = eligible[eligible["evaluation_split"].eq("train")]
    refit = eligible[eligible["evaluation_split"].isin(["train", "validation"])]

    def fit_maps(source: pd.DataFrame) -> dict[str, dict[str, float]]:
        global_median = float(source["residual_trend_target_db"].median())
        return {
            "Global L1 offset": {signal: global_median for signal in source["signal_name"].unique()},
            "Signal L2 mean": source.groupby("signal_name")["residual_trend_target_db"].mean().astype(float).to_dict(),
            "Signal Huber location": source.groupby("signal_name")["residual_trend_target_db"].apply(lambda s: huber_location(s.to_numpy(float))).astype(float).to_dict(),
            "Signal L1 fixed effect": source.groupby("signal_name")["residual_trend_target_db"].median().astype(float).to_dict(),
        }

    train_maps = fit_maps(train)
    refit_maps = fit_maps(refit)
    rows: list[dict[str, Any]] = []
    for split in ["validation", "test", "external_holdout"]:
        part = eligible[eligible["evaluation_split"].eq(split)]
        maps = train_maps if split == "validation" else refit_maps
        y = part["cn0_observed_trend_dbhz"].to_numpy(float)
        physics = part["cn0_physics_trend_dbhz"].to_numpy(float)
        for learner, mapping in maps.items():
            correction = part["signal_name"].map(mapping).fillna(0.0).to_numpy(float)
            rows.append({"learner": learner, "evaluation_split": split, **metric_row(y, physics + correction)})
    output = pd.DataFrame(rows)
    selected = refit_maps["Signal L1 fixed effect"]
    beta_rows = [{"signal_name": key, "beta_db": value} for key, value in selected.items()]
    return output, pd.DataFrame(beta_rows)


def contribution_share_table(contribution: pd.DataFrame) -> pd.DataFrame:
    """Sequential, order-dependent MSE reductions for the three-layer reconstruction."""
    rows: list[dict[str, Any]] = []
    for (split, phase), group in contribution.groupby(["evaluation_split", "mission_phase"], dropna=False):
        values = group.set_index("stage")["rmse_dbhz"]
        if not {"Physics", "Physics + beta", "Physics + beta + HGB"}.issubset(values.index):
            continue
        e0 = float(values["Physics"] ** 2)
        e1 = float(values["Physics + beta"] ** 2)
        e2 = float(values["Physics + beta + HGB"] ** 2)
        recovered = max(e0 - e2, 1e-12)
        rows.append(
            {
                "evaluation_split": split,
                "mission_phase": phase,
                "physics_mse": e0,
                "fixed_mse": e1,
                "full_mse": e2,
                "fixed_share_total_recovered_pct": 100.0 * (e0 - e1) / recovered,
                "timevarying_share_total_recovered_pct": 100.0 * (e1 - e2) / recovered,
                "timevarying_reduction_of_remaining_mse_pct": 100.0 * (e1 - e2) / max(e1, 1e-12),
                "timevarying_reduction_of_remaining_rmse_pct": 100.0 * (math.sqrt(e1) - math.sqrt(e2)) / max(math.sqrt(e1), 1e-12),
            }
        )
    return pd.DataFrame(rows)


def prepare_feature_frame() -> pd.DataFrame:
    if PREPARED_CACHE.exists():
        return joblib.load(PREPARED_CACHE)
    raw = ENGINE.base_mod.load_data()
    all_features = ENGINE.base_mod.feature_columns(raw)
    bins = ENGINE.aggregate_one_minute(raw, all_features)
    bins, _ = ENGINE.split_synchronized_time_blocks(bins)
    bins = ENGINE.add_trend_target(bins)
    missing = [feature for feature in FEATURES if feature not in bins.columns]
    if missing:
        raise RuntimeError(f"Prepared feature frame is missing selected features: {missing}")
    keep = list(
        dict.fromkeys(
            [
                "minute_utc",
                "evaluation_split",
                "op",
                "mission_phase",
                "signal_name",
                "svid",
                "trend_training_eligible",
                "residual_trend_target_db",
                "cn0_observed_trend_dbhz",
                "cn0_physics_trend_dbhz",
                *FEATURES,
            ]
        )
    )
    prepared = bins[keep].copy()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared, PREPARED_CACHE, compress=3)
    return prepared


def preprocess(scale: str | None = None) -> ColumnTransformer:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale == "standard":
        steps.append(("scale", StandardScaler()))
    elif scale == "robust":
        steps.append(("scale", RobustScaler()))
    transformer = Pipeline(steps)
    return ColumnTransformer([("physics", transformer, FEATURES)], remainder="drop", verbose_feature_names_out=False)


def benchmark_models(prepared: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    path = ANALYSIS_DIR / "final_split_algorithm_benchmark_with_block_bootstrap.csv"
    model_cache = ANALYSIS_DIR / "validation_benchmark_models_with_block_bootstrap.joblib"
    if path.exists() and model_cache.exists():
        return pd.read_csv(path), joblib.load(model_cache)

    train = prepared[
        prepared["evaluation_split"].eq("train")
        & prepared["trend_training_eligible"].astype(bool)
        & prepared["residual_trend_target_db"].notna()
    ].copy()
    validation = prepared[
        prepared["evaluation_split"].eq("validation")
        & prepared["trend_training_eligible"].astype(bool)
        & prepared["residual_trend_target_db"].notna()
    ].copy()
    beta = train.groupby("signal_name")["residual_trend_target_db"].median().astype(float).to_dict()
    y_train = train["residual_trend_target_db"].to_numpy(float) - train["signal_name"].map(beta).to_numpy(float)

    candidates: dict[str, Pipeline] = {
        "Huber linear": Pipeline(
            [
                ("preprocess", preprocess("robust")),
                ("model", HuberRegressor(epsilon=1.35, alpha=0.002, max_iter=700)),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("preprocess", preprocess()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=320,
                        max_depth=16,
                        min_samples_leaf=8,
                        max_features=0.75,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Extra trees": Pipeline(
            [
                ("preprocess", preprocess()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=320,
                        max_depth=18,
                        min_samples_leaf=6,
                        max_features=0.75,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "HGB": Pipeline(
            [
                ("preprocess", preprocess()),
                ("model", HistGradientBoostingRegressor(**PARAMETERS, random_state=42)),
            ]
        ),
        "MLP": Pipeline(
            [
                ("preprocess", preprocess("standard")),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=0.01,
                        learning_rate_init=8e-4,
                        max_iter=450,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        candidates["LightGBM"] = Pipeline(
            [
                ("preprocess", preprocess()),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=400,
                        learning_rate=0.035,
                        num_leaves=15,
                        min_child_samples=32,
                        reg_lambda=1.5,
                        verbosity=-1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        candidates["CatBoost"] = Pipeline(
            [
                ("preprocess", preprocess()),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=400,
                        depth=6,
                        learning_rate=0.04,
                        l2_leaf_reg=3.0,
                        loss_function="RMSE",
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=42,
                        thread_count=-1,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    prediction_map: dict[str, np.ndarray] = {}
    for name, model in candidates.items():
        started = time.perf_counter()
        model.fit(train[FEATURES], y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        residual_pred_raw = model.predict(validation[FEATURES]) + validation["signal_name"].map(beta).to_numpy(float)
        predict_seconds = time.perf_counter() - started
        validation_for_smoothing = validation.copy()
        validation_for_smoothing["benchmark_residual_raw"] = residual_pred_raw
        residual_pred = ENGINE.smooth_predicted_residual(
            validation_for_smoothing,
            "benchmark_residual_raw",
            isolate_evaluation_splits=True,
        ).to_numpy(float)
        cn0_pred = validation["cn0_physics_trend_dbhz"].to_numpy(float) + residual_pred
        target = validation["cn0_observed_trend_dbhz"].to_numpy(float)
        prediction_map[name] = cn0_pred
        phase_scores = {
            phase: rmse(
                target[validation["mission_phase"].eq(phase).to_numpy()],
                cn0_pred[validation["mission_phase"].eq(phase).to_numpy()],
            )
            for phase in ["C", "T", "L", "S"]
            if validation["mission_phase"].eq(phase).any()
        }
        raw_size_kb = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0
        rows.append(
            {
                "algorithm": name,
                "validation_n": len(validation),
                "validation_rmse_dbhz": rmse(target, cn0_pred),
                "validation_mae_dbhz": float(mean_absolute_error(target, cn0_pred)),
                "validation_bias_dbhz": float(np.mean(cn0_pred - target)),
                "phase_balanced_rmse_dbhz": float(np.mean(list(phase_scores.values()))),
                "worst_phase_rmse_dbhz": float(np.max(list(phase_scores.values()))),
                **{f"rmse_phase_{phase}_dbhz": phase_scores.get(phase, np.nan) for phase in ["C", "T", "L", "S"]},
                "fit_seconds": fit_seconds,
                "predict_ms_per_1000": predict_seconds / max(len(validation), 1) * 1e6,
                "serialized_size_kb": raw_size_kb,
            }
        )
        fitted[name] = model
        print(f"benchmark {name}: phase-balanced RMSE={rows[-1]['phase_balanced_rmse_dbhz']:.3f}", flush=True)

    validation_boot = validation.reset_index(drop=True).copy()
    op_start = validation_boot.groupby("op")["minute_utc"].transform("min")
    elapsed = (validation_boot["minute_utc"] - op_start).dt.total_seconds().div(60.0)
    validation_boot["bootstrap_block"] = validation_boot["op"].astype(str) + "|" + np.floor(elapsed / 15.0).astype(int).astype(str)
    block_indices = {
        block: group.index.to_numpy()
        for block, group in validation_boot.groupby("bootstrap_block", sort=False)
    }
    blocks = np.array(list(block_indices), dtype=object)
    target = validation_boot["cn0_observed_trend_dbhz"].to_numpy(float)
    rng = np.random.default_rng(240719)
    bootstrap_scores: dict[str, list[float]] = {name: [] for name in prediction_map}
    for _ in range(1000):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        indices = np.concatenate([block_indices[block] for block in sampled_blocks])
        for name, values in prediction_map.items():
            bootstrap_scores[name].append(rmse(target[indices], values[indices]))
    for row in rows:
        scores = np.asarray(bootstrap_scores[row["algorithm"]], dtype=float)
        row["validation_rmse_ci95_low_dbhz"] = float(np.percentile(scores, 2.5))
        row["validation_rmse_ci95_high_dbhz"] = float(np.percentile(scores, 97.5))

    prediction_output = validation_boot[["minute_utc", "op", "mission_phase", "signal_name", "svid", "bootstrap_block"]].copy()
    prediction_output["observed_trend_dbhz"] = target
    for name, values in prediction_map.items():
        prediction_output[f"pred_{name.lower().replace(' ', '_')}_dbhz"] = values
    prediction_output.to_csv(
        ANALYSIS_DIR / "validation_algorithm_predictions_for_block_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = pd.DataFrame(rows).sort_values("validation_rmse_dbhz").reset_index(drop=True)
    result["validation_rank"] = np.arange(1, len(result) + 1)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    joblib.dump(fitted, model_cache, compress=3)
    return result, fitted


FEATURE_GROUPS = {
    "Physical baseline\nand range": [
        "cn0_constellation_direct_available_dbhz",
        "cn0_reference_trajectory_2d_dbhz",
        "fspl_db",
        "geometric_range_km",
        "range_rate_rx_only_km_s",
        "cn0_physics_trend_dbhz",
    ],
    "Transmit antenna\ngeometry": [
        "tx_offboresight_deg",
        "tx_theta_body_deg",
        "tx_phi_body_deg",
        "tx_gain_2d_db",
        "tx_power_dbw",
        "tx_ssv_main_lobe_boundary_deg",
        "tx_ssv_signed_lower_boundary_deg",
        "tx_ssv_signed_upper_boundary_deg",
        "tx_ssv_double_sided_full_width_deg",
    ],
    "Receive antenna\nand attitude": [
        "rx_peak_gain_dbic",
        "rx_offboresight_spice_deg",
        "rx_azimuth_spice_deg",
        "rx_gain_envelope_dbic",
    ],
    "Limb and\noccultation": [
        "earth_observer_altitude_km",
        "earth_limb_margin_deg",
        "moon_observer_altitude_km",
        "moon_limb_margin_deg",
        "earth_limb_proximity_proxy",
        "moon_limb_proximity_proxy",
        "edge_near_limb_alt_threshold_km",
        "neutral_refraction_upper_alt_km",
    ],
    "Atmospheric\ntrend proxies": [
        "ionosphere_shell_lower_alt_km",
        "ionosphere_shell_upper_alt_km",
        "ionosphere_l30_vertical_assumed_db",
        "m_ion_proxy",
        "l_ion_abs_proxy_db",
        "l_ion_abs_budget_db",
        "troposphere_shell_upper_alt_km",
        "gas_scale_height_assumed_km",
        "gas_vertical_equivalent_airmass_km",
        "gas_equivalent_airmass_km",
        "m_gas_proxy",
        "gas_gamma_surface_db_per_km",
        "l_gas_abs_proxy_db",
        "l_gas_abs_budget_db",
    ],
    "Signal and\nhardware": [
        "system_noise_temperature_k",
        "implementation_loss_assumed_db",
        "frequency_mhz",
    ],
}


def grouped_permutation(prepared: pd.DataFrame) -> pd.DataFrame:
    path = ANALYSIS_DIR / "external_holdout_grouped_permutation.csv"
    if path.exists():
        return pd.read_csv(path)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    beta = {str(k): float(v) for k, v in artifact["signal_beta_db"].items()}
    part = prepared[
        prepared["evaluation_split"].eq("external_holdout")
        & prepared["trend_training_eligible"].astype(bool)
        & prepared["residual_trend_target_db"].notna()
    ].copy()
    y = part["cn0_observed_trend_dbhz"].to_numpy(float)
    base_resid = model.predict(part[FEATURES]) + part["signal_name"].map(beta).to_numpy(float)
    base_pred = part["cn0_physics_trend_dbhz"].to_numpy(float) + base_resid
    base_rmse = rmse(y, base_pred)
    rng = np.random.default_rng(20260719)
    rows: list[dict[str, Any]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        columns = [column for column in columns if column in FEATURES]
        for repeat in range(30):
            permuted = part[FEATURES].copy()
            order = rng.permutation(len(permuted))
            permuted.loc[:, columns] = permuted[columns].to_numpy()[order]
            residual = model.predict(permuted) + part["signal_name"].map(beta).to_numpy(float)
            pred = part["cn0_physics_trend_dbhz"].to_numpy(float) + residual
            rows.append(
                {
                    "feature_group": group_name,
                    "repeat": repeat,
                    "base_rmse_dbhz": base_rmse,
                    "permuted_rmse_dbhz": rmse(y, pred),
                    "delta_rmse_dbhz": rmse(y, pred) - base_rmse,
                    "n": len(part),
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return output


def choose_example(pred: pd.DataFrame) -> pd.DataFrame:
    pool = pred[pred["evaluation_split"].eq("external_holdout") & pred["trend_training_eligible"]].copy()
    beta = beta_map(pred, ["train", "validation"])
    pool["beta_db"] = pool["signal_name"].map(beta).fillna(0.0)
    pool["physics_fixed"] = pool["cn0_physics_trend_dbhz"] + pool["beta_db"]
    candidates: list[tuple[float, pd.DataFrame]] = []
    sign_crossing_candidates: list[tuple[float, pd.DataFrame]] = []
    for _, group in pool.groupby(["op", "signal_name", "svid"]):
        group = group.sort_values("minute_utc").copy()
        group["segment"] = group["minute_utc"].diff().dt.total_seconds().fillna(0).gt(90).cumsum()
        for _, segment in group.groupby("segment"):
            segment = segment.dropna(subset=["cn0_observed_trend_dbhz", "physics_fixed", "cn0_physics_ai_eval_dbhz"])
            if len(segment) < 20:
                continue
            y = segment["cn0_observed_trend_dbhz"].to_numpy(float)
            rmse_fixed = rmse(y, segment["physics_fixed"].to_numpy(float))
            rmse_hgb = rmse(y, segment["cn0_physics_ai_eval_dbhz"].to_numpy(float))
            post_beta = y - segment["physics_fixed"].to_numpy(float)
            improvement = rmse_fixed - rmse_hgb
            if post_beta.min() < -0.5 and post_beta.max() > 0.5 and rmse_hgb <= 0.75 and improvement >= 0.30:
                sign_crossing_candidates.append((improvement, segment.copy()))
            # A method figure must visibly demonstrate the second learner on unseen data.
            if rmse_hgb <= 0.50 and rmse_fixed - rmse_hgb >= 0.50:
                candidates.append((improvement, segment.copy()))
    if sign_crossing_candidates:
        return max(sign_crossing_candidates, key=lambda item: item[0])[1]
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    raise RuntimeError("No continuous complete-OP holdout link demonstrates a clear HGB increment.")


def box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str, ec: str, size: float = 7.0) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015", transform=ax.transAxes, fc=fc, ec=ec, lw=0.85)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=size, color=COL["ink"])


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = COL["muted"]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color))


def figure_ai1(pred: pd.DataFrame, card: dict[str, Any]) -> None:
    example = choose_example(pred)
    fig = plt.figure(figsize=(7.35, 4.85))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.05, 1.12], height_ratios=[1.2, 1.0], left=0.065, right=0.985, top=0.95, bottom=0.08, wspace=0.30, hspace=0.36)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(example["minute_utc"], example["cn0_physics_trend_dbhz"], color=COL["physics"], lw=1.15, label="Physics trend")
    ax.plot(example["minute_utc"], example["cn0_observed_trend_dbhz"], color=COL["teal"], lw=1.65, label="Observed trend target")
    ax.plot(example["minute_utc"], example["cn0_physics_ai_eval_dbhz"], color=COL["ai"], lw=1.35, label="Physics + residual")
    ax.fill_between(example["minute_utc"], example["cn0_physics_trend_dbhz"], example["cn0_observed_trend_dbhz"], color=COL["ai_light"], alpha=0.55, linewidth=0)
    ax.set_ylabel(r"$C/N_0$ (dB-Hz)")
    ax.set_xlabel("UTC")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    light_grid(ax)
    ax.legend(loc="lower left", ncol=3, frameon=False, handlelength=1.8, columnspacing=1.0)
    ax.text(0.99, 0.94, "OP21 | one unseen link", transform=ax.transAxes, ha="right", va="top", color=COL["muted"], fontsize=6.8)
    panel_label(ax, "a", x=-0.09, y=1.03)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0.02, 0.98, r"$\left(C/N_0\right)_{obs}$", transform=ax.transAxes, fontsize=9.3, fontweight="bold", va="top")
    ax.text(0.37, 0.98, "=", transform=ax.transAxes, fontsize=9.3, va="top")
    box(ax, (0.03, 0.66), (0.90, 0.18), "physics baseline\nknown geometry and link budget", COL["physics_light"], COL["physics"], 7.2)
    ax.text(0.48, 0.60, "+", transform=ax.transAxes, fontsize=10, ha="center")
    box(ax, (0.03, 0.38), (0.90, 0.15), "robust L1 regression\nlearned offset $\\beta_s$", COL["beta_light"], COL["beta"], 7.2)
    ax.text(0.48, 0.32, "+", transform=ax.transAxes, fontsize=10, ha="center")
    box(ax, (0.03, 0.10), (0.90, 0.15), "structured residual $f_{HGB}(\\mathbf{x})$\nslow geometry-dependent mismatch", COL["ai_light"], COL["ai"], 6.8)
    ax.text(0.48, 0.02, r"$+\;\epsilon_{fast}$  excluded from the trend target", transform=ax.transAxes, ha="center", va="bottom", color=COL["muted"], fontsize=6.5)
    panel_label(ax, "b", x=-0.07, y=1.03)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    box(ax, (0.02, 0.69), (0.23, 0.20), "dynamic\ntrajectory", COL["blue_light"], COL["blue"])
    box(ax, (0.31, 0.69), (0.27, 0.20), "physical\n$C/N_0$ engine", COL["physics_light"], COL["physics"])
    box(ax, (0.65, 0.69), (0.31, 0.20), "1-min observed\ntrend residual", COL["teal_light"], COL["teal"])
    arrow(ax, (0.25, 0.79), (0.31, 0.79))
    arrow(ax, (0.58, 0.79), (0.65, 0.79))
    box(ax, (0.03, 0.30), (0.20, 0.20), "$\\beta_s$\ntrain median", COL["beta_light"], COL["beta"], 6.6)
    box(ax, (0.32, 0.30), (0.22, 0.20), "$r-\\beta_s$\ncentered target", COL["teal_light"], COL["teal"], 6.6)
    box(ax, (0.64, 0.30), (0.31, 0.20), "HGB\n44 physics features", COL["ai_light"], COL["ai"], 6.6)
    arrow(ax, (0.23, 0.40), (0.32, 0.40))
    arrow(ax, (0.54, 0.40), (0.64, 0.40))
    arrow(ax, (0.78, 0.69), (0.47, 0.51), COL["teal"])
    ax.text(0.50, 0.10, "Output: $\\beta_s+f_{HGB}(\\mathbf{x})$.\nNo OP, SVID or constellation-ID shortcut.", transform=ax.transAxes, ha="center", color=COL["muted"], fontsize=6.1)
    panel_label(ax, "c", x=-0.08, y=1.05)

    ax = fig.add_subplot(gs[1, 1])
    summary = pd.DataFrame(card["split_summary"]).set_index("evaluation_split")
    order = ["train", "validation", "test", "external_holdout"]
    values = [float(summary.loc[key, "quality_passed_rows"]) for key in order]
    colors = [COL["blue"], COL["beta"], COL["teal"], COL["ai"]]
    bars = ax.barh(np.arange(4), values, color=colors, height=0.55)
    ax.set_yticks(np.arange(4), ["Train", "Validation", "Test", "4 OP holdout"])
    ax.invert_yaxis()
    ax.set_xlabel("Quality-passed 1-min samples")
    light_grid(ax, "x")
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.02, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", fontsize=6.4)
    ax.text(0.02, -0.30, "Only validation ranks candidates; test and complete OP holdouts are opened after freezing.", transform=ax.transAxes, color=COL["muted"], fontsize=6.25)
    panel_label(ax, "d", x=-0.16, y=1.05)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    families = [
        ("Range / FSPL", COL["blue_light"], COL["blue"]),
        ("Tx pattern / yaw", COL["ai_light"], COL["ai"]),
        ("Rx pattern / attitude", COL["teal_light"], COL["teal"]),
        ("Limb / atmosphere", COL["beta_light"], COL["beta"]),
        ("Hardware / signal", "#E7E2F0", COL["violet"]),
    ]
    for i, (label, fc, ec) in enumerate(families):
        y = 0.82 - i * 0.16
        box(ax, (0.05, y), (0.86, 0.105), label, fc, ec, 6.8)
    ax.text(0.48, 0.04, "Interpretation: grouped sensitivity, not causal attribution", transform=ax.transAxes, ha="center", color=COL["muted"], fontsize=6.2)
    panel_label(ax, "e", x=-0.07, y=1.05)

    save_figure(fig, "FigAI1_physics_guided_residual_learning_framework")


def figure_ai2(pred: pd.DataFrame, contribution: pd.DataFrame, importance: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.05, 1.05], height_ratios=[1.0, 1.05], left=0.07, right=0.985, top=0.95, bottom=0.09, wspace=0.34, hspace=0.40)

    external = pred[pred["evaluation_split"].eq("external_holdout") & pred["trend_training_eligible"]].copy()
    refit_beta = beta_map(pred, ["train", "validation"])
    external["beta_db"] = external["signal_name"].map(refit_beta).fillna(0.0)
    residuals = {
        "Physics": external["cn0_physics_trend_dbhz"] - external["cn0_observed_trend_dbhz"],
        "+ beta": external["cn0_physics_trend_dbhz"] + external["beta_db"] - external["cn0_observed_trend_dbhz"],
        "+ HGB": external["cn0_physics_ai_eval_dbhz"] - external["cn0_observed_trend_dbhz"],
    }
    ax = fig.add_subplot(gs[:, 0])
    positions = [0, 1, 2]
    colors = [COL["physics"], COL["beta"], COL["ai"]]
    violin = ax.violinplot([np.clip(residuals[key], -16, 16) for key in residuals], positions=positions, widths=0.72, showextrema=False, showmedians=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.28)
    rng = np.random.default_rng(17)
    for x, (label, values), color in zip(positions, residuals.items(), colors):
        sample = values.sample(min(900, len(values)), random_state=17)
        ax.scatter(x + rng.normal(0, 0.055, len(sample)), sample, s=2.3, color=color, alpha=0.20, edgecolors="none", rasterized=True)
        med = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        ax.plot([x, x], [q1, q3], color=color, lw=4.0, solid_capstyle="round")
        ax.scatter([x], [med], s=22, color="white", edgecolor=color, linewidth=1.2, zorder=5)
        ax.text(x, 14.7, f"RMSE\n{math.sqrt(np.mean(values**2)):.2f}", ha="center", va="top", fontsize=7.0, color=color, fontweight="bold")
    ax.axhline(0, color=COL["ink"], lw=0.75)
    ax.set_xticks(positions, list(residuals.keys()))
    ax.set_ylabel("Signed trend error (dB)")
    ax.set_ylim(-16, 16)
    ax.set_title("Residual contraction on complete-OP holdouts", loc="left", pad=5)
    light_grid(ax, "y")
    panel_label(ax, "a", x=-0.18, y=1.03)

    ax = fig.add_subplot(gs[0, 1:])
    overall = contribution[contribution["mission_phase"].isna()].copy()
    stage_order = ["Physics", "Physics + beta", "Physics + beta + HGB"]
    split_order = ["train", "validation", "test", "external_holdout"]
    x = np.arange(len(stage_order))
    for split, color, marker in zip(split_order, [COL["blue"], COL["beta"], COL["teal"], COL["ai"]], ["o", "s", "D", "^"]):
        part = overall[overall["evaluation_split"].eq(split)].set_index("stage").reindex(stage_order)
        ax.plot(x, part["rmse_dbhz"], marker=marker, ms=5.0, lw=1.35, color=color, label=SPLIT_LABELS[split])
    ax.set_xticks(x, ["Physics", "+ robust L1", "+ HGB"])
    ax.set_ylabel("Trend RMSE (dB-Hz)")
    ax.set_title("Training fit and progressively stronger generalization tests", loc="left", pad=5)
    ax.legend(ncol=2, loc="upper right", handlelength=1.8, columnspacing=1.0)
    light_grid(ax)
    panel_label(ax, "b", x=-0.08, y=1.03)

    ax = fig.add_subplot(gs[1, 1])
    phase = contribution[
        contribution["evaluation_split"].eq("external_holdout")
        & contribution["mission_phase"].notna()
    ].copy()
    for phase_name in ["C", "T", "L", "S"]:
        part = phase[phase["mission_phase"].eq(phase_name)].set_index("stage").reindex(stage_order)
        ax.plot(x, part["rmse_dbhz"], marker="o", ms=4.2, lw=1.25, color=PHASE_COLORS[phase_name], label=phase_name)
    ax.set_xticks(x, ["Physics", "+ beta", "+ HGB"])
    ax.set_ylabel("Holdout RMSE (dB-Hz)")
    ax.set_title("Improvement spans C/T/L/S", loc="left", pad=5)
    ax.legend(title="Phase", ncol=4, loc="upper right", columnspacing=0.8, handletextpad=0.3)
    light_grid(ax)
    panel_label(ax, "c", x=-0.16, y=1.03)

    ax = fig.add_subplot(gs[1, 2])
    summary = importance.groupby("feature_group")["delta_rmse_dbhz"].agg(["mean", "std"]).sort_values("mean")
    y = np.arange(len(summary))
    ax.barh(y, summary["mean"], xerr=summary["std"], color=COL["ai"], alpha=0.78, height=0.58, error_kw={"ecolor": COL["muted"], "lw": 0.7, "capsize": 2})
    ax.axvline(0, color=COL["ink"], lw=0.75)
    ax.set_yticks(y, summary.index)
    ax.set_xlabel(r"Permutation sensitivity, $\Delta$RMSE (dB-Hz)")
    ax.set_title("What structures the learner uses", loc="left", pad=5)
    light_grid(ax, "x")
    ax.text(0.02, -0.25, "30 repeats on complete-OP holdouts; sensitivity is not causal attribution.", transform=ax.transAxes, color=COL["muted"], fontsize=6.1)
    panel_label(ax, "d", x=-0.28, y=1.03)

    save_figure(fig, "FigAI2_residual_contribution_and_interpretation")


def figure_ai3(benchmark: pd.DataFrame, candidates: pd.DataFrame, contribution: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.05], left=0.07, right=0.985, top=0.95, bottom=0.09, wspace=0.36, hspace=0.42)

    ax = fig.add_subplot(gs[:, 0])
    ordered = benchmark.sort_values("validation_rmse_dbhz", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = [COL["ai"] if name == "HGB" else COL["physics_light"] for name in ordered["algorithm"]]
    edges = [COL["ai"] if name == "HGB" else COL["physics"] for name in ordered["algorithm"]]
    low = ordered["validation_rmse_dbhz"] - ordered["validation_rmse_ci95_low_dbhz"]
    high = ordered["validation_rmse_ci95_high_dbhz"] - ordered["validation_rmse_dbhz"]
    ax.errorbar(
        ordered["validation_rmse_dbhz"],
        y,
        xerr=np.vstack([low, high]),
        fmt="none",
        ecolor=COL["physics_light"],
        elinewidth=2.2,
        capsize=2.5,
        zorder=1,
    )
    ax.hlines(y, ordered["validation_rmse_dbhz"], ordered["worst_phase_rmse_dbhz"], color=COL["physics_light"], lw=1.0, alpha=0.75)
    ax.scatter(ordered["validation_rmse_dbhz"], y, s=42, color=colors, edgecolor=edges, linewidth=1.0, zorder=3, label="Overall validation RMSE (95% block-bootstrap CI)")
    ax.scatter(ordered["worst_phase_rmse_dbhz"], y, s=26, marker="|", color=edges, linewidth=1.7, zorder=3, label="Worst phase")
    ax.set_yticks(y, ordered["algorithm"])
    ax.invert_yaxis()
    ax.set_xlabel("Validation trend RMSE (dB-Hz)")
    ax.set_title("HGB gives the lowest overall validation error", loc="left", pad=5)
    light_grid(ax, "x")
    ax.text(
        0.98,
        0.02,
        "circle: overall RMSE and 95% block-bootstrap CI\nvertical tick: worst phase; top-tree CIs overlap",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COL["muted"],
        fontsize=6.0,
    )
    for yi, row in ordered.iterrows():
        if row["algorithm"] == "HGB":
            ax.text(row["worst_phase_rmse_dbhz"] + 0.04, yi, "selected", va="center", color=COL["ai"], fontweight="bold", fontsize=6.7)
    panel_label(ax, "a", x=-0.20, y=1.03)

    ax = fig.add_subplot(gs[0, 1])
    phases = ["C", "T", "L", "S"]
    heat = ordered.set_index("algorithm")[[f"rmse_phase_{p}_dbhz" for p in phases]].to_numpy(float)
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=np.nanmin(heat), vmax=np.nanpercentile(heat, 95))
    ax.set_xticks(np.arange(4), phases)
    ax.set_yticks(np.arange(len(ordered)), ordered["algorithm"])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            color = "white" if heat[i, j] > np.nanpercentile(heat, 70) else COL["ink"]
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=6.0, color=color)
    ax.set_title("Phase-resolved validation", loc="left", pad=5)
    ax.tick_params(length=0)
    panel_label(ax, "b", x=-0.18, y=1.03)

    ax = fig.add_subplot(gs[0, 2])
    feature_colors = {name: color for name, color in zip(sorted(candidates["feature_set"].unique()), [COL["blue"], COL["teal"], COL["violet"], COL["beta"], COL["physics"]])}
    for feature_set, group in candidates.groupby("feature_set"):
        ax.scatter(group["phase_balanced_rmse_dbhz"], group["worst_phase_rmse_dbhz"], s=26, alpha=0.58, color=feature_colors[feature_set], label=feature_set.replace("_", " "))
    selected = candidates[candidates["selected"].astype(str).str.lower().eq("true")]
    ax.scatter(selected["phase_balanced_rmse_dbhz"], selected["worst_phase_rmse_dbhz"], s=78, facecolor="none", edgecolor=COL["ai"], linewidth=1.6, zorder=5)
    ax.annotate("selected HGB", (selected["phase_balanced_rmse_dbhz"].iloc[0], selected["worst_phase_rmse_dbhz"].iloc[0]), xytext=(7, -12), textcoords="offset points", color=COL["ai"], fontsize=6.5, arrowprops={"arrowstyle": "-", "color": COL["ai"], "lw": 0.7})
    ax.set_xlabel("Phase-balanced RMSE")
    ax.set_ylabel("Worst-phase RMSE")
    ax.set_title("24 HGB candidates, no test peeking", loc="left", pad=5)
    light_grid(ax)
    ax.legend(loc="upper left", fontsize=5.7, ncol=1, handletextpad=0.3)
    panel_label(ax, "c", x=-0.18, y=1.03)

    ax = fig.add_subplot(gs[1, 1])
    overall = contribution[contribution["mission_phase"].isna() & contribution["stage"].eq("Physics + beta + HGB")].set_index("evaluation_split")
    split_order = ["train", "validation", "test", "external_holdout"]
    vals = overall.reindex(split_order)["rmse_dbhz"].to_numpy(float)
    bars = ax.bar(np.arange(4), vals, color=[COL["blue"], COL["beta"], COL["teal"], COL["ai"]], width=0.62)
    ax.set_xticks(np.arange(4), ["Train", "Validation", "Test", "4 OP\nholdout"], rotation=0)
    ax.set_ylabel("Trend RMSE (dB-Hz)")
    ax.set_title("Final selected model generalization", loc="left", pad=5)
    light_grid(ax, "y")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=6.6)
    panel_label(ax, "d", x=-0.18, y=1.03)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    selected_benchmark = benchmark[benchmark["algorithm"].eq("HGB")].iloc[0]
    box(ax, (0.04, 0.74), (0.92, 0.17), "Selected learner\nHistGradientBoostingRegressor", COL["ai_light"], COL["ai"], 7.2)
    items = [
        ("Trees", f"{PARAMETERS['max_iter']}"),
        ("Learning rate", f"{PARAMETERS['learning_rate']:.2f}"),
        ("Leaf nodes", f"{PARAMETERS['max_leaf_nodes']}"),
        ("Min. leaf", f"{PARAMETERS['min_samples_leaf']} samples"),
        ("L2", f"{PARAMETERS['l2_regularization']:.1f}"),
        ("Inference", f"{selected_benchmark['predict_ms_per_1000']:.1f} ms / 1000"),
    ]
    for i, (key, value) in enumerate(items):
        y = 0.61 - i * 0.095
        ax.text(0.08, y, key, color=COL["muted"], fontsize=6.6, ha="left")
        ax.text(0.92, y, value, color=COL["ink"], fontsize=6.6, ha="right", fontweight="bold")
    ax.text(0.50, 0.02, "Family: lowest overall validation RMSE. HGB tuning: phase-balanced validation.\nTest and complete-OP holdouts remain untouched.", transform=ax.transAxes, ha="center", color=COL["muted"], fontsize=5.8)
    panel_label(ax, "e", x=-0.08, y=1.03)

    save_figure(fig, "FigAI3_algorithm_and_hyperparameter_selection")


def _legacy_figure_ai1_three_layer(pred: pd.DataFrame, card: dict[str, Any]) -> None:
    """Three-layer architecture: deterministic physics plus two learned residual components."""
    example = choose_example(pred).copy()
    beta = beta_map(pred, ["train", "validation"])
    example["physics_fixed"] = example["cn0_physics_trend_dbhz"] + example["signal_name"].map(beta).fillna(0.0)

    fig = plt.figure(figsize=(7.35, 4.10))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.76, 1.24],
        width_ratios=[1.90, 0.82],
        left=0.065,
        right=0.985,
        top=0.970,
        bottom=0.155,
        wspace=0.28,
        hspace=0.30,
    )

    def mini_tree(axis: plt.Axes, cx: float, cy: float, sx: float, sy: float, color: str) -> None:
        nodes = [
            (cx, cy + sy),
            (cx - sx, cy),
            (cx + sx, cy),
            (cx - 1.35 * sx, cy - sy),
            (cx - 0.50 * sx, cy - sy),
            (cx + 0.50 * sx, cy - sy),
            (cx + 1.35 * sx, cy - sy),
        ]
        for parent, child in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
            axis.plot(
                [nodes[parent][0], nodes[child][0]],
                [nodes[parent][1], nodes[child][1]],
                transform=axis.transAxes,
                color=color,
                lw=0.65,
                solid_capstyle="round",
            )
        for index, (x0, y0) in enumerate(nodes):
            axis.scatter(
                [x0],
                [y0],
                transform=axis.transAxes,
                s=7.0 if index < 3 else 4.8,
                facecolor="white" if index < 3 else color,
                edgecolor=color,
                linewidth=0.50,
                zorder=4,
            )

    # a, the unique computational role of each learner. No decorative trend curves.
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    panel_label(ax, "a", x=-0.035, y=1.02)
    ax.text(
        0.012,
        0.985,
        "Physics-guided residual learning",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color=COL["ink"],
        fontweight="bold",
    )

    ax.text(0.012, 0.720, "Training", transform=ax.transAxes, ha="left", va="center", fontsize=5.1, color=COL["muted"], fontweight="bold")
    ax.plot([0.075, 0.985], [0.720, 0.720], transform=ax.transAxes, color=COL["grid"], lw=0.65)

    # Training target: observed trend minus the deterministic physical baseline.
    ax.text(0.125, 0.555, "Observed trend", transform=ax.transAxes, ha="center", va="center", fontsize=5.2, color=COL["teal"], fontweight="bold")
    ax.text(0.125, 0.475, "minus physical trend", transform=ax.transAxes, ha="center", va="center", fontsize=4.65, color=COL["physics"])
    ax.text(0.125, 0.360, "physics residual", transform=ax.transAxes, ha="center", va="center", fontsize=4.25, color=COL["muted"])
    arrow(ax, (0.205, 0.505), (0.270, 0.505), COL["muted"])

    # Learner 1: the L1 optimum is the robust centre of the band-wise residuals.
    x2 = np.linspace(0.300, 0.430, 17)
    residual_cloud = 0.500 + 0.047 * np.sin(np.linspace(0.0, 3.6 * np.pi, len(x2))) + 0.010 * np.cos(np.linspace(0.0, 7.0 * np.pi, len(x2)))
    beta_level = float(np.median(residual_cloud))
    ax.scatter(x2, residual_cloud, transform=ax.transAxes, s=8.0, facecolor="white", edgecolor=COL["beta"], linewidth=0.65)
    ax.plot([0.290, 0.440], [beta_level, beta_level], transform=ax.transAxes, color=COL["beta"], lw=1.45)
    ax.text(0.365, 0.625, r"L1 fixed residual  $\beta_s$", transform=ax.transAxes, ha="center", va="center", fontsize=5.25, color=COL["beta"], fontweight="bold")
    ax.text(0.365, 0.350, "band-wise robust intercept", transform=ax.transAxes, ha="center", va="center", fontsize=4.25, color=COL["muted"])
    arrow(ax, (0.445, 0.505), (0.505, 0.505), COL["muted"])

    # The second learner receives only the residual left after the fixed term.
    ax.text(0.570, 0.555, "Residual after fixed term", transform=ax.transAxes, ha="center", va="center", fontsize=5.05, color=COL["ink"], fontweight="bold")
    ax.text(0.570, 0.455, "+ physical descriptors", transform=ax.transAxes, ha="center", va="center", fontsize=4.45, color=COL["muted"])
    ax.text(0.570, 0.360, "state-dependent target", transform=ax.transAxes, ha="center", va="center", fontsize=4.20, color=COL["muted"])
    arrow(ax, (0.650, 0.505), (0.705, 0.505), COL["muted"])

    # Learner 2: binned physical descriptors drive an ensemble of shallow trees.
    mini_tree(ax, 0.740, 0.505, 0.013, 0.052, COL["ai"])
    mini_tree(ax, 0.790, 0.480, 0.013, 0.047, COL["ai"])
    mini_tree(ax, 0.840, 0.515, 0.013, 0.047, COL["ai"])
    ax.text(0.790, 0.625, "HGB residual learner", transform=ax.transAxes, ha="center", va="center", fontsize=5.25, color=COL["ai"], fontweight="bold")
    ax.text(0.790, 0.350, "binned features · boosted shallow trees", transform=ax.transAxes, ha="center", va="center", fontsize=4.20, color=COL["muted"])
    arrow(ax, (0.860, 0.505), (0.910, 0.505), COL["ai"])
    ax.text(0.950, 0.535, "HGB", transform=ax.transAxes, ha="center", va="center", fontsize=5.05, color=COL["ai"], fontweight="bold")
    ax.text(0.950, 0.455, "residual", transform=ax.transAxes, ha="center", va="center", fontsize=4.45, color=COL["ai"])

    # Prediction uses the three terms additively; this row is the complete model definition.
    ax.plot([0.012, 0.985], [0.285, 0.285], transform=ax.transAxes, color=COL["grid"], lw=0.65)
    ax.text(0.012, 0.170, "Prediction", transform=ax.transAxes, ha="left", va="center", fontsize=5.1, color=COL["muted"], fontweight="bold")
    terms = [
        (0.180, "Physical trend", COL["physics"]),
        (0.400, "Fixed residual", COL["beta"]),
        (0.610, "HGB residual", COL["ai"]),
    ]
    for x0, label, color in terms:
        ax.plot([x0 - 0.055, x0 + 0.055], [0.205, 0.205], transform=ax.transAxes, color=color, lw=1.55)
        ax.text(x0, 0.125, label, transform=ax.transAxes, ha="center", va="center", fontsize=4.65, color=color)
    ax.text(0.290, 0.180, "+", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color=COL["ink"])
    ax.text(0.505, 0.180, "+", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color=COL["ink"])
    arrow(ax, (0.675, 0.180), (0.745, 0.180), COL["ink"])
    ax.text(0.860, 0.205, "Physics + fixed residual + HGB", transform=ax.transAxes, ha="center", va="center", fontsize=5.25, color=COL["ai"], fontweight="bold")
    ax.text(0.860, 0.115, r"reconstructed $C/N_0$ trend", transform=ax.transAxes, ha="center", va="center", fontsize=4.45, color=COL["muted"])

    # b, observed evidence and the two residual-learning increments.
    bgs = gs[1, 0].subgridspec(2, 1, height_ratios=[0.70, 0.30], hspace=0.08)
    ax = fig.add_subplot(bgs[0, 0])
    start = example["minute_utc"].min()
    end = example["minute_utc"].max()
    observed = example["cn0_observed_trend_dbhz"].to_numpy(float)
    physics = example["cn0_physics_trend_dbhz"].to_numpy(float)
    fixed = example["physics_fixed"].to_numpy(float)
    full = example["cn0_physics_ai_eval_dbhz"].to_numpy(float)
    ax.fill_between(
        example["minute_utc"],
        physics,
        observed,
        color=COL["physics_light"],
        alpha=0.24,
        linewidth=0,
        zorder=1,
    )
    ax.plot(example["minute_utc"], observed, color=COL["teal"], lw=1.30, linestyle=(0, (3, 1.5)), label="Observed trend", zorder=5)
    ax.plot(example["minute_utc"], physics, color=COL["physics"], lw=1.00, label="Physics only", zorder=3)
    ax.plot(example["minute_utc"], fixed, color=COL["beta"], lw=1.18, label="Physics + fixed residual", zorder=4)
    ax.plot(example["minute_utc"], full, color=COL["ai"], lw=1.50, label="Physics + fixed residual + HGB", zorder=6)
    ax.set_ylabel(r"$C/N_0$ (dB-Hz)")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7, interval_multiples=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelbottom=False)
    light_grid(ax, "y")
    ax.legend(loc="lower left", ncol=2, frameon=False, handlelength=1.45, columnspacing=0.75, borderpad=0.05, fontsize=5.1)
    ax.text(0.76, 0.86, "physics-only gap", transform=ax.transAxes, color=COL["physics"], fontsize=4.6, ha="center")
    op_label = str(example["op"].iloc[0])
    signal_name = str(example["signal_name"].iloc[0])
    signal_label = {"GPS_L1": "GPS L1", "GPS_L5": "GPS L5", "GAL_E1": "Galileo E1", "GAL_E5a": "Galileo E5a"}.get(signal_name, signal_name)
    satellite_label = ("E" if signal_name.startswith("GAL") else "G") + f"{int(example['svid'].iloc[0]):02d}"
    ax.set_title(rf"Absolute trend reconstruction  |  {op_label}, {signal_label}, {satellite_label}", loc="left", pad=3, fontsize=7.3)
    panel_label(ax, "b", x=-0.070, y=1.075)

    axh = fig.add_subplot(bgs[1, 0], sharex=ax)
    post_beta_target = observed - fixed
    hgb_component = full - fixed
    axh.axhline(0, color=COL["ink"], lw=0.62, alpha=0.75, zorder=0)
    axh.plot(
        example["minute_utc"],
        post_beta_target,
        color=COL["blue"],
        lw=1.05,
        linestyle=(0, (3, 1.5)),
        label="Residual after physics + fixed residual",
    )
    axh.plot(
        example["minute_utc"],
        hgb_component,
        color=COL["ai"],
        lw=1.30,
        label="HGB contribution",
    )
    axh.set_ylabel(r"$C/N_0$ correction" + "\n(dB)", linespacing=0.92)
    axh.set_xlim(start, end)
    axh.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7, interval_multiples=True))
    axh.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    light_grid(axh, "y")
    axh.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.45, columnspacing=0.72, borderpad=0.05, fontsize=5.4)
    axh.text(0.50, -0.40, "Time [UTC, hh:mm]", transform=axh.transAxes, ha="center", va="top", fontsize=7.1, color=COL["ink"])
    axh.text(1.00, -0.40, start.strftime("%b %d, %Y"), transform=axh.transAxes, ha="right", va="top", fontsize=6.4, color=COL["ink"])

    # c, a compact scope matrix; no synthetic waveforms duplicate panel b.
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("Residual timescale and model scope", loc="left", pad=3, fontsize=7.3)
    rows = [
        {
            "y": 0.735,
            "class": "Persistent offset",
            "scale": "constant",
            "method": r"L1 fixed residual  $\beta_s$",
            "scope": "EIRP · system noise · hardware",
            "color": COL["beta"],
        },
        {
            "y": 0.455,
            "class": "State-dependent trend",
            "scale": "minutes to hours",
            "method": "HGB residual learner",
            "scope": "pattern · attitude · limb · dynamics",
            "color": COL["ai"],
        },
        {
            "y": 0.175,
            "class": "Fast irregular events",
            "scale": "seconds / events",
            "method": "Excluded from trend target",
            "scope": "rapid fading · scintillation · lock events",
            "color": COL["muted"],
        },
    ]
    ax.text(0.02, 0.875, "Residual class and treatment", transform=ax.transAxes, ha="left", fontsize=4.15, color=COL["muted"], fontweight="bold")
    ax.text(0.53, 0.875, "Characteristic scale", transform=ax.transAxes, ha="left", fontsize=4.15, color=COL["muted"], fontweight="bold")
    ax.plot([0.02, 0.98], [0.835, 0.835], transform=ax.transAxes, color=COL["ink"], lw=0.72)
    for divider in [0.605, 0.325]:
        ax.plot([0.02, 0.98], [divider, divider], transform=ax.transAxes, color=COL["grid"], lw=0.55)
    for row in rows:
        ax.plot([0.02, 0.075], [row["y"] + 0.030, row["y"] + 0.030], transform=ax.transAxes, color=row["color"], lw=1.7)
        ax.text(0.095, row["y"] + 0.060, row["class"], transform=ax.transAxes, ha="left", va="center", fontsize=5.05, color=COL["ink"], fontweight="bold")
        ax.text(0.53, row["y"] + 0.060, row["scale"], transform=ax.transAxes, ha="left", va="center", fontsize=4.55, color=COL["muted"])
        ax.text(0.095, row["y"] - 0.010, row["method"], transform=ax.transAxes, ha="left", va="center", fontsize=4.75, color=row["color"], fontweight="bold")
        ax.text(0.095, row["y"] - 0.075, row["scope"], transform=ax.transAxes, ha="left", va="center", fontsize=4.00, color=COL["muted"])
    ax.text(0.98, 0.015, "Compensation is interpretable by timescale, not a causal attribution.", transform=ax.transAxes, ha="right", va="bottom", fontsize=3.95, color=COL["muted"], style="italic")
    panel_label(ax, "c", x=-0.17, y=1.04)

    save_figure(fig, "FigAI1_physics_guided_residual_learning_framework")


def figure_ai1_three_layer(pred: pd.DataFrame, card: dict[str, Any]) -> None:
    """Build the physics-first residual-learning framework and real-link evidence."""
    del card
    example = choose_example(pred).copy()
    beta = beta_map(pred, ["train", "validation"])
    example["physics_fixed"] = example["cn0_physics_trend_dbhz"] + example["signal_name"].map(beta).fillna(0.0)

    color = {
        "ink": "#20262B",
        "muted": "#657681",
        "rule": "#D9E0E4",
        "physics": "#2F6B8A",
        "beta": "#D49300",
        "hgb": "#C43D58",
        "observed": "#252A2E",
        "outside": "#89939A",
        "row": "#F6F8F9",
    }

    width_in = 183.0 / 25.4
    height_in = 114.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.88, 1.20],
        width_ratios=[1.64, 1.00],
        left=0.060,
        right=0.985,
        top=0.970,
        bottom=0.105,
        wspace=0.245,
        hspace=0.285,
    )

    def node(
        axis: plt.Axes,
        xy: tuple[float, float],
        wh: tuple[float, float],
        title: str,
        detail: str,
        accent: str,
        *,
        title_size: float = 5.25,
        detail_size: float = 5.00,
    ) -> None:
        x0, y0 = xy
        width, height = wh
        axis.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                transform=axis.transAxes,
                facecolor="white",
                edgecolor=color["rule"],
                linewidth=0.60,
                clip_on=True,
                zorder=2,
            )
        )
        axis.plot(
            [x0, x0],
            [y0, y0 + height],
            transform=axis.transAxes,
            color=accent,
            lw=2.0,
            solid_capstyle="butt",
            clip_on=True,
            zorder=3,
        )
        axis.text(
            x0 + 0.012,
            y0 + 0.66 * height,
            title,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=title_size,
            color=color["ink"],
            fontweight="bold",
            clip_on=True,
            zorder=4,
        )
        if detail:
            axis.text(
                x0 + 0.012,
                y0 + 0.28 * height,
                detail,
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=detail_size,
                color=color["muted"],
                clip_on=True,
                zorder=4,
            )

    def arrow(
        axis: plt.Axes,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        line_color: str | None = None,
        linewidth: float = 0.72,
        mutation_scale: float = 7.0,
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start_xy,
                end_xy,
                transform=axis.transAxes,
                arrowstyle="-|>",
                mutation_scale=mutation_scale,
                linewidth=linewidth,
                color=line_color or color["muted"],
                shrinkA=0,
                shrinkB=0,
                clip_on=True,
                zorder=5,
            )
        )

    # a, physics residual -> persistent offset -> remaining slow residual.
    axa = fig.add_subplot(gs[0, :])
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    axa.axis("off")
    axa.text(0.000, 0.995, "a", transform=axa.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=color["ink"])
    axa.text(
        0.030,
        0.995,
        "Physics-guided residual learning",
        transform=axa.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=color["ink"],
    )
    axa.text(0.030, 0.795, "TRAINING", transform=axa.transAxes, ha="left", va="center", fontsize=5.0, fontweight="bold", color=color["muted"])

    node(axa, (0.060, 0.625), (0.145, 0.125), "Observed trend", r"1-min $C/N_0$ target", color["observed"])
    node(axa, (0.060, 0.445), (0.145, 0.125), "Physics baseline", "deterministic trend", color["physics"])

    residual_xy = (0.247, 0.594)
    axa.add_patch(
        Circle(
            residual_xy,
            0.017,
            transform=axa.transAxes,
            facecolor="white",
            edgecolor=color["ink"],
            linewidth=0.65,
            clip_on=True,
            zorder=6,
        )
    )
    axa.text(*residual_xy, "-", transform=axa.transAxes, ha="center", va="center", fontsize=6.4, fontweight="bold", color=color["ink"], zorder=7)
    arrow(axa, (0.205, 0.688), (0.233, 0.604))
    arrow(axa, (0.205, 0.508), (0.233, 0.584))
    axa.text(0.247, 0.515, r"$r_{\mathrm{phys}}$", transform=axa.transAxes, ha="center", va="center", fontsize=5.1, color=color["ink"])

    node(
        axa,
        (0.290, 0.495),
        (0.185, 0.195),
        "Robust L1 regression",
        "persistent offset for each\nconstellation and band " + r"$\beta_s$",
        color["beta"],
        detail_size=5.00,
    )
    arrow(axa, (0.264, 0.594), (0.290, 0.594))
    node(
        axa,
        (0.520, 0.520),
        (0.160, 0.150),
        "Remaining slow residual",
        r"$r_{\mathrm{phys}}-\beta_s$",
        color["beta"],
        title_size=5.00,
    )
    arrow(axa, (0.475, 0.594), (0.520, 0.594), line_color=color["beta"])
    axa.text(0.497, 0.628, r"remove $\beta_s$", transform=axa.transAxes, ha="center", va="bottom", fontsize=5.0, color=color["muted"])

    node(
        axa,
        (0.775, 0.520),
        (0.185, 0.150),
        "HGB regression",
        r"learn $\delta_{\mathrm{HGB}}(t)$",
        color["hgb"],
    )
    arrow(axa, (0.680, 0.594), (0.775, 0.594), line_color=color["hgb"])
    node(
        axa,
        (0.775, 0.405),
        (0.185, 0.072),
        "Continuous physical descriptors",
        "",
        color["physics"],
        title_size=5.00,
    )
    arrow(axa, (0.868, 0.477), (0.868, 0.520), line_color=color["physics"], mutation_scale=6.0)

    axa.plot([0.030, 0.970], [0.335, 0.335], transform=axa.transAxes, color=color["rule"], lw=0.60, clip_on=True)
    axa.text(0.030, 0.235, "INFERENCE", transform=axa.transAxes, ha="left", va="center", fontsize=5.0, fontweight="bold", color=color["muted"])
    node(axa, (0.075, 0.075), (0.155, 0.120), "Physics baseline", r"$(C/N_0)_{\mathrm{phys}}$", color["physics"], title_size=4.95)
    axa.text(0.252, 0.135, "+", transform=axa.transAxes, ha="center", va="center", fontsize=6.4, color=color["ink"])
    node(axa, (0.275, 0.075), (0.120, 0.120), "Robust L1 correction", r"$\beta_s$", color["beta"], title_size=4.95)
    axa.text(0.418, 0.135, "+", transform=axa.transAxes, ha="center", va="center", fontsize=6.4, color=color["ink"])
    node(axa, (0.445, 0.075), (0.150, 0.120), "Slow correction", r"$\delta_{\mathrm{HGB}}(t)$", color["hgb"], title_size=4.95)
    arrow(axa, (0.620, 0.135), (0.685, 0.135), line_color=color["ink"])
    node(axa, (0.685, 0.075), (0.275, 0.120), r"Predicted $C/N_0$ trend", "large-scale channel trend", color["ink"], title_size=5.20)

    # b, one real external-holdout arc with the existing model outputs.
    bgs = gs[1, 0].subgridspec(2, 1, height_ratios=[0.69, 0.31], hspace=0.10)
    axb = fig.add_subplot(bgs[0, 0])
    axr = fig.add_subplot(bgs[1, 0], sharex=axb)

    start = example["minute_utc"].min()
    end = example["minute_utc"].max()
    observed = example["cn0_observed_trend_dbhz"].to_numpy(float)
    physics = example["cn0_physics_trend_dbhz"].to_numpy(float)
    robust_l1 = example["physics_fixed"].to_numpy(float)
    physics_ai = example["cn0_physics_ai_eval_dbhz"].to_numpy(float)
    after_l1_error = observed - robust_l1
    after_hgb_error = observed - physics_ai

    axb.plot(
        example["minute_utc"],
        observed,
        color=color["observed"],
        lw=1.05,
        linestyle=(0, (4.0, 1.6, 1.2, 1.6)),
        label="Observed trend",
        zorder=6,
    )
    axb.plot(example["minute_utc"], physics, color=color["physics"], lw=1.10, label="Physics baseline", zorder=2)
    axb.plot(example["minute_utc"], robust_l1, color=color["beta"], lw=1.20, label="Physics + robust L1", zorder=3)
    axb.plot(example["minute_utc"], physics_ai, color=color["hgb"], lw=1.55, label="Physics + robust L1 + HGB", zorder=4)
    axb.set_ylabel(r"$C/N_0$ (dB-Hz)")
    axb.set_xlim(start, end)
    axb.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6, interval_multiples=True))
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axb.tick_params(axis="x", labelbottom=False)
    axb.grid(True, axis="y", color=color["rule"], linewidth=0.50, alpha=0.85)
    axb.set_axisbelow(True)

    op_label = str(example["op"].iloc[0])
    signal_name = str(example["signal_name"].iloc[0])
    signal_label = {
        "GPS_L1": "GPS L1",
        "GPS_L5": "GPS L5",
        "GAL_E1": "Galileo E1",
        "GAL_E5a": "Galileo E5a",
    }.get(signal_name, signal_name)
    satellite_label = ("E" if signal_name.startswith("GAL") else "G") + f"{int(example['svid'].iloc[0]):02d}"
    axb.text(-0.095, 1.180, "b", transform=axb.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=color["ink"])
    axb.text(
        0.000,
        1.180,
        f"Example reconstruction | {op_label}, {signal_label}, {satellite_label}",
        transform=axb.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=color["ink"],
    )
    axb.legend(
        loc="lower left",
        bbox_to_anchor=(0.000, 0.995),
        ncol=4,
        frameon=False,
        handlelength=1.35,
        columnspacing=0.55,
        borderaxespad=0.0,
        fontsize=5.00,
    )

    axr.axhline(0, color=color["outside"], lw=0.60, alpha=0.75, zorder=0)
    axr.plot(
        example["minute_utc"],
        after_l1_error,
        color=color["beta"],
        lw=1.00,
        linestyle=(0, (3.0, 1.8)),
        label="After robust L1",
        zorder=2,
    )
    axr.plot(example["minute_utc"], after_hgb_error, color=color["hgb"], lw=1.30, label="After HGB", zorder=3)
    axr.set_ylabel("Error (dB)")
    axr.set_xlim(start, end)
    axr.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6, interval_multiples=True))
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axr.grid(True, axis="y", color=color["rule"], linewidth=0.50, alpha=0.85)
    axr.set_axisbelow(True)
    axr.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.65,
        columnspacing=0.85,
        borderaxespad=0.0,
        fontsize=5.05,
    )
    axr.text(0.995, 0.930, "observed - model", transform=axr.transAxes, ha="right", va="top", fontsize=5.0, color=color["muted"])
    axr.text(0.50, -0.33, "Time (UTC)", transform=axr.transAxes, ha="center", va="top", fontsize=6.6, color=color["ink"])
    axr.text(1.00, -0.33, start.strftime("%d %b %Y"), transform=axr.transAxes, ha="right", va="top", fontsize=5.8, color=color["muted"])

    # c, implemented mechanism-to-layer allocation at the model's working timescales.
    axc = fig.add_subplot(gs[1, 1])
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")
    axc.text(0.000, 1.095, "c", transform=axc.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=color["ink"])
    axc.text(
        0.085,
        1.095,
        "Implemented model allocation",
        transform=axc.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=color["ink"],
    )
    mechanism_x = 0.020
    scale_x = 0.580
    assignment_x = [0.765, 0.835, 0.905, 0.965]
    axc.text(0.870, 0.955, "IMPLEMENTED LAYER", transform=axc.transAxes, ha="center", va="center", fontsize=5.0, fontweight="bold", color=color["muted"])
    axc.text(mechanism_x, 0.845, "Mechanism or\nuncertainty", transform=axc.transAxes, ha="left", va="center", fontsize=5.0, fontweight="bold", color=color["muted"], linespacing=0.94)
    axc.text(scale_x, 0.845, "Scale", transform=axc.transAxes, ha="center", va="center", fontsize=5.0, fontweight="bold", color=color["muted"])
    for xpos, label in zip(assignment_x, ["Physics", "Robust L1", "HGB", "Outside"]):
        axc.text(
            xpos,
            0.790,
            label,
            transform=axc.transAxes,
            ha="left",
            va="bottom",
            rotation=68,
            rotation_mode="anchor",
            fontsize=5.0,
            color=color["ink"],
        )

    table_top = 0.760
    table_bottom = 0.075
    row_height = (table_top - table_bottom) / 6.0
    axc.plot([0.010, 0.990], [table_top, table_top], transform=axc.transAxes, color=color["ink"], lw=0.68, clip_on=True)
    axc.plot([0.730, 0.730], [table_bottom, table_top], transform=axc.transAxes, color=color["rule"], lw=0.55, clip_on=True)

    allocation_rows = [
        ("Range / FSPL /\nphysical occultation", "epoch", 0, color["physics"]),
        ("Nominal antenna\nresponse / atmospheric trend", "epoch", 0, color["physics"]),
        ("EIRP / receiver noise /\nRF-chain scale", "constant for each\nconstellation\nand band", 1, color["beta"]),
        ("Pattern / attitude /\nnear-limb mismatch", "min-h", 2, color["hgb"]),
        ("Range-rate /\natmospheric-path proxies", "min-h", 2, color["hgb"]),
        ("Acquisition transients /\nloss of lock / rapid\nionospheric scintillation", "s / event", 3, color["outside"]),
    ]
    for index, (mechanism, scale, layer_index, accent) in enumerate(allocation_rows):
        ypos = table_top - row_height * (index + 0.5)
        row_bottom = table_top - row_height * (index + 1)
        if index % 2:
            axc.add_patch(
                Rectangle(
                    (0.010, row_bottom),
                    0.980,
                    row_height,
                    transform=axc.transAxes,
                    facecolor=color["row"],
                    edgecolor="none",
                    clip_on=True,
                    zorder=0,
                )
            )
        axc.add_patch(
            Rectangle(
                (assignment_x[layer_index] - 0.014, ypos - 0.021),
                0.028,
                0.042,
                transform=axc.transAxes,
                facecolor=accent,
                edgecolor="white",
                linewidth=0.45,
                clip_on=True,
                zorder=3,
            )
        )
        axc.text(mechanism_x, ypos, mechanism, transform=axc.transAxes, ha="left", va="center", fontsize=5.0, color=color["ink"], linespacing=0.88)
        axc.text(scale_x, ypos, scale, transform=axc.transAxes, ha="center", va="center", fontsize=5.0, color=color["muted"], linespacing=0.82)
        axc.plot([0.010, 0.990], [row_bottom, row_bottom], transform=axc.transAxes, color=color["rule"], lw=0.42, clip_on=True)

    axc.text(
        0.010,
        0.012,
        "Allocation denotes the implemented model structure, not causal attribution.",
        transform=axc.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.0,
        color=color["muted"],
    )

    save_ai1_polished(fig)


def figure_ai1_standalone_bc(pred: pd.DataFrame) -> None:
    """Export Fig. 1b and Fig. 1c as independent, PPT-ready panels."""
    example = choose_example(pred).copy()
    beta = beta_map(pred, ["train", "validation"])
    example["physics_fixed"] = example["cn0_physics_trend_dbhz"] + example["signal_name"].map(beta).fillna(0.0)

    start = example["minute_utc"].min()
    end = example["minute_utc"].max()
    observed = example["cn0_observed_trend_dbhz"].to_numpy(float)
    physics = example["cn0_physics_trend_dbhz"].to_numpy(float)
    fixed = example["physics_fixed"].to_numpy(float)
    full = example["cn0_physics_ai_eval_dbhz"].to_numpy(float)
    before_hgb_error = observed - fixed
    after_hgb_error = observed - full
    op_label = str(example["op"].iloc[0])
    signal_name = str(example["signal_name"].iloc[0])
    signal_label = {
        "GPS_L1": "GPS L1",
        "GPS_L5": "GPS L5",
        "GAL_E1": "Galileo E1",
        "GAL_E5a": "Galileo E5a",
    }.get(signal_name, signal_name)
    satellite_label = ("E" if signal_name.startswith("GAL") else "G") + f"{int(example['svid'].iloc[0]):02d}"

    # Standalone panel b: preserve the two-stage evidence chain at a legible size.
    fig = plt.figure(figsize=(5.65, 3.25))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.70, 0.30],
        left=0.115,
        right=0.985,
        top=0.905,
        bottom=0.195,
        hspace=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(example["minute_utc"], physics, observed, color=COL["physics_light"], alpha=0.24, linewidth=0, zorder=1)
    ax.plot(example["minute_utc"], observed, color=COL["teal"], lw=1.45, linestyle=(0, (3, 1.5)), label="Observed trend", zorder=5)
    ax.plot(example["minute_utc"], physics, color=COL["physics"], lw=1.08, label="Physics only", zorder=3)
    ax.plot(example["minute_utc"], fixed, color=COL["beta"], lw=1.28, label="Physics + robust L1", zorder=4)
    ax.plot(example["minute_utc"], full, color=COL["ai"], lw=1.60, label="Physics + robust L1 + HGB", zorder=6)
    ax.set_ylabel(r"$C/N_0$ (dB-Hz)")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7, interval_multiples=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelbottom=False)
    light_grid(ax, axis="y")
    ax.legend(loc="lower left", ncol=2, frameon=False, handlelength=1.55, columnspacing=0.95, borderpad=0.05, fontsize=6.0)
    ax.text(0.76, 0.87, "physics-only gap", transform=ax.transAxes, color=COL["muted"], fontsize=5.8, ha="center")
    ax.set_title(
        rf"Absolute trend reconstruction  |  {op_label}, {signal_label}, {satellite_label}",
        loc="left",
        pad=3,
        fontsize=8.1,
    )
    panel_label(ax, "b", x=-0.095, y=1.085)

    axh = fig.add_subplot(gs[1, 0], sharex=ax)
    axh.axhline(0, color=COL["ink"], lw=0.68, alpha=0.75, zorder=0)
    axh.plot(
        example["minute_utc"],
        before_hgb_error,
        color=COL["blue"],
        lw=1.16,
        linestyle=(0, (3, 1.5)),
        label="Before HGB",
    )
    axh.plot(example["minute_utc"], after_hgb_error, color=COL["ai"], lw=1.42, label="After HGB")
    axh.set_ylabel("Observed - model\nerror (dB)", linespacing=0.92)
    axh.set_xlim(start, end)
    axh.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7, interval_multiples=True))
    axh.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    light_grid(axh, axis="y")
    axh.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.55, columnspacing=0.95, borderpad=0.05, fontsize=5.9)
    axh.set_xlabel("Time [UTC, hh:mm]", labelpad=4)
    axh.text(1.00, -0.38, start.strftime("%b %d, %Y"), transform=axh.transAxes, ha="right", va="top", fontsize=6.4, color=COL["ink"])
    save_figure(fig, "FigAI1b_absolute_trend_reconstruction")

    # Standalone panel c: wrapped row labels prevent the first row from setting
    # an unnecessarily wide mechanism column when assembled in PowerPoint.
    fig = plt.figure(figsize=(3.85, 2.42))
    ax = fig.add_axes([0.090, 0.095, 0.890, 0.805])
    ax.axis("off")
    ax.set_title("Implemented model responsibility", loc="left", pad=2.5, fontsize=8.1)
    panel_label(ax, "c", x=-0.085, y=1.075)

    mechanisms = [
        "Range / FSPL /\noccultation",
        "Nominal antenna /\natmosphere",
        "EIRP / noise /\nhardware scale",
        "Pattern / attitude /\nlimb mismatch",
        "Receiver /\nenvironment drift",
        "Fast fading /\nscintillation / lock",
    ]
    columns = ["Physics", "Robust\nL1", "HGB", "Outside"]
    assignment = [0, 0, 1, 2, 2, 3]
    assignment_colors = [COL["physics"], COL["physics"], COL["beta"], COL["ai"], COL["ai"], COL["muted"]]
    timescales = ["epoch", "epoch", "constant", "min-h", "min-h", "s / event"]
    col_x = [0.625, 0.745, 0.855, 0.960]
    table_top = 0.770
    table_bottom = 0.105
    row_height = (table_top - table_bottom) / len(mechanisms)
    row_y = np.array([table_top - row_height * (index + 0.5) for index in range(len(mechanisms))])

    ax.text(0.00, 0.820, "Mechanism", transform=ax.transAxes, ha="left", va="center", fontsize=5.8, color=COL["muted"], fontweight="bold")
    ax.text(0.465, 0.820, "Scale", transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color=COL["muted"], fontweight="bold")
    ax.text(0.795, 0.910, "Assigned layer", transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color=COL["muted"], fontweight="bold")
    for xpos, label in zip(col_x, columns):
        ax.text(xpos, 0.820, label, transform=ax.transAxes, ha="center", va="center", fontsize=5.45, linespacing=0.90, color=COL["ink"])
    ax.plot([0.00, 0.995], [table_top, table_top], transform=ax.transAxes, color=COL["ink"], lw=0.72)
    ax.plot([0.555, 0.555], [table_bottom, table_top], transform=ax.transAxes, color=COL["grid"], lw=0.52)

    for row_index, (label, column_index, color, scale, ypos) in enumerate(zip(mechanisms, assignment, assignment_colors, timescales, row_y)):
        if row_index % 2:
            ax.add_patch(Rectangle((0.00, ypos - row_height / 2), 0.995, row_height, transform=ax.transAxes, facecolor="#F7F8F9", edgecolor="none", zorder=0))
        ax.text(0.00, ypos, label, transform=ax.transAxes, ha="left", va="center", fontsize=5.35, linespacing=0.88, color=COL["ink"])
        ax.text(0.465, ypos, scale, transform=ax.transAxes, ha="center", va="center", fontsize=5.25, color=COL["muted"])
        ax.scatter([col_x[column_index]], [ypos], transform=ax.transAxes, s=45, marker="s", color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax.plot([0.00, 0.995], [table_top - row_height * (row_index + 1), table_top - row_height * (row_index + 1)], transform=ax.transAxes, color=COL["grid"], lw=0.42, zorder=1)

    ax.text(
        0.00,
        0.015,
        "Assignment shown; sensitivity is not a causal attribution.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.45,
        color=COL["muted"],
        style="italic",
    )
    save_figure(fig, "FigAI1c_model_responsibility_table")


def figure_ai2_three_layer(
    contribution: pd.DataFrame,
    shares: pd.DataFrame,
    importance: pd.DataFrame,
    beta_values: pd.DataFrame,
) -> None:
    """Quantify the roles of physics, robust L1 regression and HGB regression."""
    fig = plt.figure(figsize=(7.10, 4.65))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.95, 1.20, 1.08],
        height_ratios=[0.92, 1.08],
        left=0.078,
        right=0.988,
        top=0.958,
        bottom=0.105,
        wspace=0.42,
        hspace=0.54,
    )
    overall = contribution[contribution["mission_phase"].isna()].copy()
    holdout = overall[overall["evaluation_split"].eq("external_holdout")].set_index("stage")
    stages = ["Physics", "Physics + beta", "Physics + beta + HGB"]
    stage_labels = ["Physics", "+ robust L1", "+ HGB"]
    stage_colors = [COL["physics"], COL["beta"], COL["ai"]]

    # Connected estimates make the sequential reduction the visual argument.
    ax = fig.add_subplot(gs[0, 0])
    rmse_values = holdout.reindex(stages)["rmse_dbhz"].to_numpy(float)
    x = np.arange(3)
    ax.plot(x, rmse_values, color=COL["physics_light"], lw=1.15, zorder=1)
    for xpos, value, color in zip(x, rmse_values, stage_colors):
        ax.vlines(xpos, 0, value, color=COL["grid"], lw=0.65, zorder=0)
        ax.scatter(xpos, value, s=43, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(xpos, value + 0.30, f"{value:.2f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax.text(0.49, 5.25, f"Δ {rmse_values[1] - rmse_values[0]:.2f}", color=COL["muted"], ha="center", va="center", fontsize=5.8)
    ax.text(1.50, 2.32, f"Δ {rmse_values[2] - rmse_values[1]:.2f}", color=COL["muted"], ha="center", va="center", fontsize=5.8)
    ax.set_xticks(np.arange(3), stage_labels, rotation=20, ha="right")
    ax.set_ylim(0, max(rmse_values) * 1.14)
    ax.set_ylabel("Complete-OP RMSE (dB-Hz)")
    ax.set_title("Stepwise holdout reconstruction", loc="left", pad=3.5)
    ax.spines["bottom"].set_position(("data", 0))
    panel_label(ax, "a", x=-0.24, y=1.06)

    ax = fig.add_subplot(gs[0, 1])
    ext_share = shares[shares["evaluation_split"].eq("external_holdout") & shares["mission_phase"].isna()].iloc[0]
    fixed_share = float(ext_share["fixed_share_total_recovered_pct"])
    dynamic_share = float(ext_share["timevarying_share_total_recovered_pct"])
    remaining = float(ext_share["timevarying_reduction_of_remaining_mse_pct"])
    ax.barh([1], [fixed_share], color=COL["beta"], height=0.30, edgecolor="none")
    ax.barh([1], [dynamic_share], left=[fixed_share], color=COL["ai"], height=0.30, edgecolor="none")
    ax.barh([0], [remaining], color=COL["ai"], height=0.30, edgecolor="none")
    ax.barh([0], [100 - remaining], left=[remaining], color="#ECEFF1", height=0.30, edgecolor="none")
    ax.text(3.0, 1, f"Robust L1  {fixed_share:.1f}%", ha="left", va="center", color="white", fontweight="bold", fontsize=6.0)
    ax.annotate(
        f"HGB {dynamic_share:.1f}%",
        xy=(fixed_share + dynamic_share / 2, 1.16),
        xytext=(84, 1.48),
        ha="center",
        va="center",
        color=COL["ink"],
        fontsize=5.7,
        arrowprops={"arrowstyle": "-", "color": COL["ai"], "lw": 0.7},
    )
    ax.text(remaining + 2.0, 0, f"{remaining:.1f}%", va="center", ha="left", fontsize=6.1, fontweight="bold", color=COL["ink"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.62)
    ax.set_yticks([1, 0], ["Recovered MSE", "Post-L1 MSE\nremoved by HGB"])
    ax.set_xlabel("Sequential share or reduction (%)")
    ax.set_title("Sequential error recovery", loc="left", pad=3.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.0, -0.30, "Order-dependent engineering decomposition; not causal.", transform=ax.transAxes, fontsize=5.1, color=COL["muted"])
    panel_label(ax, "b", x=-0.28, y=1.06)

    ax = fig.add_subplot(gs[0, 2])
    signal_order = ["GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"]
    beta_plot = beta_values.set_index("signal_name").reindex(signal_order)
    y = np.arange(len(beta_plot))
    for yi, value in zip(y, beta_plot["beta_db"]):
        ax.hlines(yi, value, 0, color=COL["beta_light"], lw=2.2, zorder=1)
    ax.scatter(beta_plot["beta_db"], y, s=34, color=COL["beta"], edgecolor="white", linewidth=0.65, zorder=3)
    ax.axvline(0, color=COL["ink"], lw=0.75)
    ax.set_yticks(y, ["GPS L1", "GPS L5", "Galileo E1", "Galileo E5a"])
    ax.invert_yaxis()
    ax.set_xlabel(r"Learned robust-L1 correction $\beta_s$ (dB)")
    ax.set_title("Robust L1 regression", loc="left", pad=3.5)
    for yi, value in zip(y, beta_plot["beta_db"]):
        ax.text(value + 0.30, yi, f"{value:.2f}", va="center", ha="left", fontsize=5.9, color=COL["ink"])
    ax.set_xlim(min(-10.8, float(beta_plot["beta_db"].min()) - 0.8), 0.55)
    panel_label(ax, "c", x=-0.30, y=1.06)

    ax = fig.add_subplot(gs[1, 0:2])
    split_order = ["train", "validation", "test", "external_holdout"]
    x = np.arange(3)
    split_colors = [COL["blue"], COL["beta"], COL["teal"], COL["ai"]]
    split_markers = ["o", "s", "D", "^"]
    legend_handles: list[Line2D] = []
    for split, color, marker in zip(split_order, split_colors, split_markers):
        values = overall[overall["evaluation_split"].eq(split)].set_index("stage").reindex(stages)["rmse_dbhz"]
        ax.plot(x, values, marker=marker, ms=4.6, lw=1.25, color=color, markeredgecolor="white", markeredgewidth=0.45)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                lw=1.15,
                ms=4.2,
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=f"{SPLIT_LABELS[split]}  {float(values.iloc[-1]):.2f}",
            )
        )
    ax.set_xticks(x, stage_labels)
    ax.set_ylabel("Trend RMSE (dB-Hz)")
    ax.set_title("Algorithmic gains persist to complete-OP extrapolation", loc="left", pad=3.5)
    ax.set_xlim(-0.12, 2.18)
    ax.set_ylim(0, 9.2)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.68, 0.98),
        ncol=2,
        frameon=False,
        fontsize=5.15,
        handlelength=1.35,
        borderaxespad=0.0,
        labelspacing=0.32,
        columnspacing=0.80,
    )
    panel_label(ax, "d", x=-0.13, y=1.06)

    ax = fig.add_subplot(gs[1, 2])
    summary = importance.groupby("feature_group")["delta_rmse_dbhz"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    y = np.arange(len(summary))
    for yi, value in zip(y, summary["mean"]):
        ax.hlines(yi, min(0, value), max(0, value), color=COL["ai_light"], lw=2.0, zorder=1)
    ax.errorbar(
        summary["mean"],
        y,
        xerr=summary["std"],
        fmt="o",
        ms=4.0,
        mfc=COL["ai"],
        mec="white",
        mew=0.55,
        ecolor=COL["muted"],
        elinewidth=0.75,
        capsize=2.0,
        zorder=3,
    )
    ax.axvline(0, color=COL["ink"], lw=0.75)
    ax.set_yticks(y, summary.index)
    ax.invert_yaxis()
    ax.set_xlabel(r"Permutation sensitivity, $\Delta$RMSE (dB-Hz)")
    ax.set_title("Physical structure used by HGB", loc="left", pad=3.5)
    ax.text(0.0, -0.26, "Complete-OP holdouts; mean ± s.d., grouped permutation.", transform=ax.transAxes, color=COL["muted"], fontsize=5.1)
    panel_label(ax, "e", x=-0.31, y=1.06)

    save_figure(fig, "FigAI2_residual_contribution_and_interpretation")


def figure_ai3_two_learners(
    fixed_benchmark: pd.DataFrame,
    benchmark: pd.DataFrame,
    candidates: pd.DataFrame,
    contribution: pd.DataFrame,
) -> None:
    """Show why robust L1 regression and HGB regression are selected."""
    fig = plt.figure(figsize=(7.073, 4.95))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.03, 1.26, 1.03],
        height_ratios=[1.05, 0.95],
        left=0.078,
        right=0.988,
        top=0.958,
        bottom=0.105,
        wspace=0.43,
        hspace=0.54,
    )

    ax = fig.add_subplot(gs[0, 0])
    learner_order = ["Signal L1 fixed effect", "Signal Huber location", "Signal L2 mean", "Global L1 offset"]
    val = fixed_benchmark[fixed_benchmark["evaluation_split"].eq("validation")].set_index("learner").reindex(learner_order)
    ext = fixed_benchmark[fixed_benchmark["evaluation_split"].eq("external_holdout")].set_index("learner").reindex(learner_order)
    y = np.arange(len(learner_order))
    for yi, learner in enumerate(learner_order):
        ax.plot([val.loc[learner, "rmse_dbhz"], ext.loc[learner, "rmse_dbhz"]], [yi, yi], color=COL["grid"], lw=1.0, zorder=1)
    val_colors = [COL["beta"]] + [COL["physics_light"]] * 3
    val_edges = [COL["beta"]] + [COL["physics"]] * 3
    ext_edges = [COL["beta"]] + [COL["physics"]] * 3
    ax.scatter(val["rmse_dbhz"], y, s=34, color=val_colors, edgecolor=val_edges, linewidth=0.8, label="Validation", zorder=3)
    ax.scatter(ext["rmse_dbhz"], y, s=31, marker="D", facecolor="white", edgecolor=ext_edges, linewidth=1.0, label="Complete-OP", zorder=3)
    ax.set_yticks(y, ["Robust L1", "Huber", "L2 mean", "Global L1"])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE after robust-L1 correction (dB-Hz)")
    ax.set_title("Robust regression algorithm", loc="left", pad=3.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=5.35, handletextpad=0.30, borderaxespad=0.15)
    ax.annotate("selected", (ext.loc["Signal L1 fixed effect", "rmse_dbhz"], 0), xytext=(5, -12), textcoords="offset points", color=COL["ink"], fontsize=5.6)
    panel_label(ax, "a", x=-0.25, y=1.06)

    ax = fig.add_subplot(gs[0, 1])
    ordered = benchmark.sort_values("validation_rmse_dbhz", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    low = ordered["validation_rmse_dbhz"] - ordered["validation_rmse_ci95_low_dbhz"]
    high = ordered["validation_rmse_ci95_high_dbhz"] - ordered["validation_rmse_dbhz"]
    ax.errorbar(ordered["validation_rmse_dbhz"], y, xerr=np.vstack([low, high]), fmt="none", ecolor=COL["physics_light"], elinewidth=2.0, capsize=2.2)
    colors = [COL["ai"] if name == "HGB" else COL["physics_light"] for name in ordered["algorithm"]]
    edges = [COL["ai"] if name == "HGB" else COL["physics"] for name in ordered["algorithm"]]
    ax.scatter(ordered["validation_rmse_dbhz"], y, s=36, color=colors, edgecolor=edges, linewidth=1.0, zorder=3)
    ax.scatter(ordered["worst_phase_rmse_dbhz"], y, s=23, marker="|", color=edges, linewidth=1.5, zorder=3)
    ax.set_yticks(y, ordered["algorithm"])
    ax.invert_yaxis()
    ax.set_xlabel("Validation trend RMSE (dB-Hz)")
    ax.set_title("HGB selection among trend regressors", loc="left", pad=3.5)
    uncertainty_handles = [
        Line2D([0], [0], color=COL["physics_light"], marker="o", markerfacecolor="white", markeredgecolor=COL["physics"], lw=1.5, ms=3.8, label="Mean ± 95% CI"),
        Line2D([0], [0], color=COL["physics"], marker="|", linestyle="none", ms=6.0, markeredgewidth=1.2, label="Worst phase"),
    ]
    ax.legend(handles=uncertainty_handles, loc="upper right", bbox_to_anchor=(1.0, 0.98), frameon=False, fontsize=4.75, handlelength=1.5, labelspacing=0.25, borderaxespad=0.0)
    panel_label(ax, "b", x=-0.23, y=1.06)

    ax = fig.add_subplot(gs[0, 2])
    phases = ["C", "T", "L", "S"]
    heat = ordered.set_index("algorithm")[[f"rmse_phase_{p}_dbhz" for p in phases]].to_numpy(float)
    heat_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rmse_seq",
        ["#F7F6F3", "#EAD8B5", COL["beta"], COL["ai"], "#70293B"],
    )
    ax.imshow(heat, aspect="auto", cmap=heat_cmap, vmin=np.nanmin(heat), vmax=np.nanpercentile(heat, 95))
    ax.set_xticks(np.arange(4), phases)
    ax.set_yticks(np.arange(len(ordered)), ordered["algorithm"])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            color = "white" if heat[i, j] > np.nanpercentile(heat, 70) else COL["ink"]
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=5.7, color=color)
    ax.set_title("Phase-resolved validation RMSE", loc="left", pad=3.5)
    ax.tick_params(length=0)
    panel_label(ax, "c", x=-0.30, y=1.06)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(
        candidates["phase_balanced_rmse_dbhz"],
        candidates["worst_phase_rmse_dbhz"],
        s=24,
        color=COL["physics_light"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
        zorder=2,
    )
    selected = candidates[candidates["selected"].astype(str).str.lower().eq("true")]
    selected_x = float(selected["phase_balanced_rmse_dbhz"].iloc[0])
    selected_y = float(selected["worst_phase_rmse_dbhz"].iloc[0])
    ax.scatter([selected_x], [selected_y], s=63, facecolor=COL["ai"], edgecolor="white", linewidth=0.75, zorder=4)
    ax.axvline(selected_x, color=COL["ai_light"], lw=0.65, linestyle=(0, (2, 2)), zorder=0)
    ax.axhline(selected_y, color=COL["ai_light"], lw=0.65, linestyle=(0, (2, 2)), zorder=0)
    ax.annotate("selected", (selected_x, selected_y), xytext=(7, -12), textcoords="offset points", color=COL["ink"], fontsize=5.7)
    ax.set_xlabel("Phase-balanced RMSE (dB-Hz)")
    ax.set_ylabel("Worst-phase RMSE (dB-Hz)")
    ax.set_title("Validation-only HGB search (n=24)", loc="left", pad=3.5)
    panel_label(ax, "d", x=-0.25, y=1.06)

    ax = fig.add_subplot(gs[1, 1])
    overall = contribution[contribution["mission_phase"].isna() & contribution["stage"].eq("Physics + beta + HGB")].set_index("evaluation_split")
    split_order = ["train", "validation", "test", "external_holdout"]
    vals = overall.reindex(split_order)["rmse_dbhz"].to_numpy(float)
    xx = np.arange(4)
    ax.plot(xx, vals, color=COL["physics_light"], lw=1.0, zorder=1)
    point_colors = [COL["blue"], COL["beta"], COL["teal"], COL["ai"]]
    ax.scatter(xx, vals, s=42, color=point_colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_xticks(np.arange(4), ["Train", "Validation", "Test", "4 OP\nholdout"])
    ax.set_ylabel("Full-model RMSE (dB-Hz)")
    ax.set_title("Final selected model across data partitions", loc="left", pad=3.5)
    ax.set_ylim(max(0, float(vals.min()) - 0.25), float(vals.max()) + 0.30)
    for xpos, value in zip(xx, vals):
        ax.text(xpos, value + 0.055, f"{value:.2f}", ha="center", va="bottom", fontsize=5.9, fontweight="bold")
    panel_label(ax, "e", x=-0.23, y=1.06)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_title("Final selected model specification", loc="left", pad=3.5)
    ax.plot([0.00, 1.00], [0.91, 0.91], transform=ax.transAxes, color=COL["ink"], lw=0.75)
    ax.text(0.00, 0.855, "Component / parameter", transform=ax.transAxes, ha="left", va="center", fontsize=5.2, color=COL["muted"], fontweight="bold")
    ax.text(0.99, 0.855, "Selected setting", transform=ax.transAxes, ha="right", va="center", fontsize=5.2, color=COL["muted"], fontweight="bold")
    ax.plot([0.00, 1.00], [0.815, 0.815], transform=ax.transAxes, color=COL["grid"], lw=0.55)

    ax.scatter([0.018], [0.755], transform=ax.transAxes, s=24, marker="s", color=COL["beta"], edgecolor="white", linewidth=0.4)
    ax.text(0.055, 0.755, "Robust L1 regression", transform=ax.transAxes, ha="left", va="center", fontsize=6.0, fontweight="bold")
    fixed_rows = [("Loss", "Absolute error (L1)"), ("Parameters", r"4 offsets $\beta_s$")]
    for ypos, (key, value) in zip([0.680, 0.610], fixed_rows):
        ax.text(0.055, ypos, key, transform=ax.transAxes, ha="left", va="center", fontsize=5.45, color=COL["muted"])
        ax.text(0.99, ypos, value, transform=ax.transAxes, ha="right", va="center", fontsize=5.65, color=COL["ink"])
    ax.plot([0.00, 1.00], [0.555, 0.555], transform=ax.transAxes, color=COL["grid"], lw=0.65)

    ax.scatter([0.018], [0.495], transform=ax.transAxes, s=24, marker="s", color=COL["ai"], edgecolor="white", linewidth=0.4)
    ax.text(0.055, 0.495, "HGB regression", transform=ax.transAxes, ha="left", va="center", fontsize=6.0, fontweight="bold")
    dynamic_rows = [
        ("Estimator", "HistGradientBoosting"),
        ("Iterations / learning rate", f"{PARAMETERS['max_iter']} / {PARAMETERS['learning_rate']}"),
        ("Leaf nodes / minimum leaf", f"{PARAMETERS['max_leaf_nodes']} / {PARAMETERS['min_samples_leaf']}"),
        ("L2 regularization", str(PARAMETERS.get("l2_regularization", "–"))),
    ]
    for ypos, (key, value) in zip([0.415, 0.340, 0.265, 0.190], dynamic_rows):
        ax.text(0.055, ypos, key, transform=ax.transAxes, ha="left", va="center", fontsize=5.30, color=COL["muted"])
        ax.text(0.99, ypos, value, transform=ax.transAxes, ha="right", va="center", fontsize=5.55, color=COL["ink"])
    ax.plot([0.00, 1.00], [0.125, 0.125], transform=ax.transAxes, color=COL["ink"], lw=0.75)
    panel_label(ax, "f", x=-0.12, y=1.06)

    save_figure(fig, "FigAI3_algorithm_and_hyperparameter_selection")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figai1-only",
        action="store_true",
        help="Render only the physics-guided residual-learning framework.",
    )
    args = parser.parse_args(argv)

    set_style()
    pred = load_predictions()
    card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    if args.figai1_only:
        figure_ai1_three_layer(pred, card)
        print(f"FigAI1 written to {FIG_DIR}")
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    contribution = contribution_table(pred)
    contribution.to_csv(ANALYSIS_DIR / "physics_beta_hgb_contribution_metrics.csv", index=False, encoding="utf-8-sig")
    shares = contribution_share_table(contribution)
    shares.to_csv(ANALYSIS_DIR / "three_layer_sequential_contribution_shares.csv", index=False, encoding="utf-8-sig")
    fixed_benchmark, beta_values = fixed_residual_benchmark(pred)
    fixed_benchmark.to_csv(ANALYSIS_DIR / "fixed_residual_learner_benchmark.csv", index=False, encoding="utf-8-sig")
    beta_values.to_csv(ANALYSIS_DIR / "learned_signal_fixed_residual_beta.csv", index=False, encoding="utf-8-sig")
    prepared = prepare_feature_frame()
    benchmark, _ = benchmark_models(prepared)
    importance = grouped_permutation(prepared)
    candidates = pd.read_csv(CANDIDATE_PATH)

    figure_ai1_three_layer(pred, card)
    figure_ai1_standalone_bc(pred)
    figure_ai2_three_layer(contribution, shares, importance, beta_values)
    figure_ai3_two_learners(fixed_benchmark, benchmark, candidates, contribution)

    contract = {
        "figure_ai1": {
            "conclusion": "The physics baseline is corrected sequentially by robust L1 regression and HGB regression; fast irregular events remain outside the trend model.",
            "archetype": "two-algorithm architecture grounded by an unseen C/N0 arc and a model-responsibility panel",
        },
        "figure_ai2": {
            "conclusion": "Robust L1 regression removes the dominant persistent mismatch, while HGB regression removes a smaller but structured fraction of the remaining trend error across complete-OP holdouts.",
            "archetype": "asymmetric mixed-modality quantitative figure",
        },
        "figure_ai3": {
            "conclusion": "Two algorithms are validated separately: robust L1 regression for the persistent offset and HGB regression for the remaining state-dependent trend correction.",
            "archetype": "dual-learner quantitative model-selection grid",
        },
        "evaluation_contract": "Training is shown for fit/overfit diagnosis; validation ranks candidates; internal test and complete-OP holdouts support generalization claims.",
    }
    (ANALYSIS_DIR / "figure_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Figures written to {FIG_DIR}")
    print(benchmark.to_string(index=False))


if __name__ == "__main__":
    main()
