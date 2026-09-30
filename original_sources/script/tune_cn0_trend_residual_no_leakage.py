#!/usr/bin/env python3
"""Tune the precision-geometry trend model without reading test/holdout scores."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import train_cn0_constellation_residual_models as source_model


SOURCE = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_constellation_physics_baseline_sp3_tlm_exact"
    / "cn0_constellation_physics_predictions.csv"
)
OUT_DIR = ROOT / "table" / "algorithm" / "cn0_trend_residual_tuning_no_leakage"
source_model.INPUT = SOURCE

import train_cn0_single_global_residual_1min_op74_holdout as engine  # noqa: E402


engine.SMOOTH_TREND_MODEL = True
engine.TREND_WINDOW_MINUTES = 9
engine.HOLDOUT_OPS = ("OP2", "OP21", "OP27", "OP74")
engine.HOLDOUT_SPLIT_NAME = "external_holdout"
engine.REPRESENTATIVE_TEST_OPS = ()

PARAMETER_CANDIDATES = [
    {"name": "p0", "max_iter": 420, "learning_rate": 0.035, "max_leaf_nodes": 15, "min_samples_leaf": 24, "l2_regularization": 0.8},
    {"name": "p1", "max_iter": 450, "learning_rate": 0.030, "max_leaf_nodes": 11, "min_samples_leaf": 40, "l2_regularization": 2.0},
    {"name": "p2", "max_iter": 500, "learning_rate": 0.025, "max_leaf_nodes": 15, "min_samples_leaf": 48, "l2_regularization": 2.0},
    {"name": "p3", "max_iter": 450, "learning_rate": 0.035, "max_leaf_nodes": 9, "min_samples_leaf": 64, "l2_regularization": 3.0},
    {"name": "p4", "max_iter": 550, "learning_rate": 0.020, "max_leaf_nodes": 23, "min_samples_leaf": 48, "l2_regularization": 3.0},
    {"name": "p5", "max_iter": 400, "learning_rate": 0.040, "max_leaf_nodes": 7, "min_samples_leaf": 32, "l2_regularization": 1.5},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_sets(all_features: list[str]) -> dict[str, list[str]]:
    current = engine.smooth_model_features(all_features)

    grap_uncertainty_tokens = (
        "upper95",
        "lower95",
        "azimuth_mean",
        "azimuth_min",
        "azimuth_max",
        "azimuth_samples",
    )
    no_grap_uncertainty = [
        feature
        for feature in current
        if not (
            ("grap" in feature or "tx_eirp_grap" in feature)
            and any(token in feature for token in grap_uncertainty_tokens)
        )
    ]
    direct_budget_only = [
        feature
        for feature in current
        if "grap" not in feature and not feature.startswith("tx_eirp_grap_")
    ]

    unstable_raw = {
        "nav_age_seconds",
        "ionosphere_path_length_shell_km",
        "m_ion_shell_chord_raw",
        "gas_equivalent_airmass_km_raw",
        "earth_grazing_altitude_km",
        "moon_grazing_altitude_km",
    }
    stable_compact = [feature for feature in direct_budget_only if feature not in unstable_raw]
    return {
        "current": current,
        "no_grap_uncertainty": no_grap_uncertainty,
        "direct_budget_only": direct_budget_only,
        "stable_compact": stable_compact,
    }


def make_model(features: list[str], params: dict) -> Pipeline:
    preprocess = ColumnTransformer(
        [("physics", SimpleImputer(strategy="median"), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        random_state=engine.RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def rmse(observed: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(observed.to_numpy(float) - predicted.to_numpy(float)))))


def evaluate_validation(
    bins: pd.DataFrame,
    train: pd.DataFrame,
    features: list[str],
    params: dict,
) -> dict[str, float]:
    beta = engine.estimate_signal_beta(train)
    target = train["residual_trend_target_db"].to_numpy(float) - engine.signal_beta_values(train, beta)
    model = make_model(features, params)
    model.fit(train[features], target)

    # Candidate models never predict test or external-holdout rows.  This is
    # stricter than merely excluding their metrics from the ranking table.
    scored = bins[bins["evaluation_split"].eq("validation")].copy()
    scored["candidate_raw_residual_db"] = (
        np.asarray(model.predict(scored[features]), dtype=float)
        + engine.signal_beta_values(scored, beta)
    )
    scored["candidate_residual_db"] = engine.smooth_predicted_residual(scored, "candidate_raw_residual_db")
    scored["candidate_cn0_dbhz"] = scored[engine.PHYSICS_TREND_COLUMN] + scored["candidate_residual_db"]
    validation = scored[
        scored["trend_training_eligible"].astype(bool)
        & scored[engine.OBSERVED_TREND_COLUMN].notna()
    ].copy()
    if validation.empty:
        raise RuntimeError("Validation set is empty")

    phase_rmse = {
        str(phase): rmse(group[engine.OBSERVED_TREND_COLUMN], group["candidate_cn0_dbhz"])
        for phase, group in validation.groupby("mission_phase")
    }
    error = validation["candidate_cn0_dbhz"] - validation[engine.OBSERVED_TREND_COLUMN]
    result = {
        "validation_n": int(len(validation)),
        "validation_rmse_dbhz": rmse(validation[engine.OBSERVED_TREND_COLUMN], validation["candidate_cn0_dbhz"]),
        "validation_mae_dbhz": float(np.mean(np.abs(error))),
        "validation_bias_dbhz": float(np.mean(error)),
        "phase_balanced_rmse_dbhz": float(np.mean(list(phase_rmse.values()))),
        "worst_phase_rmse_dbhz": float(np.max(list(phase_rmse.values()))),
    }
    for phase in ["C", "T", "L", "S"]:
        result[f"rmse_phase_{phase}_dbhz"] = phase_rmse.get(phase, np.nan)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = engine.base_mod.load_data()
    all_features = engine.base_mod.feature_columns(raw)
    bins = engine.aggregate_one_minute(raw, all_features)
    bins, blocks = engine.split_synchronized_time_blocks(bins)
    bins = engine.add_trend_target(bins)

    holdout_mask = bins["op"].isin(engine.HOLDOUT_OPS)
    if not bins.loc[holdout_mask, "evaluation_split"].eq(engine.HOLDOUT_SPLIT_NAME).all():
        raise RuntimeError("External holdout isolation failed")
    if bins.loc[bins["evaluation_split"].isin(["train", "validation"]), "op"].isin(engine.HOLDOUT_OPS).any():
        raise RuntimeError("External holdout leakage into model selection")

    train = bins[
        bins["evaluation_split"].eq("train")
        & bins["trend_training_eligible"].astype(bool)
        & bins["residual_trend_target_db"].notna()
    ].copy()
    sets = feature_sets(all_features)
    rows = []
    for set_name, features in sets.items():
        for params in PARAMETER_CANDIDATES:
            score = evaluate_validation(bins, train, features, params)
            rows.append(
                {
                    "feature_set": set_name,
                    "feature_count": len(features),
                    "parameter_set": params["name"],
                    **{key: value for key, value in params.items() if key != "name"},
                    **score,
                }
            )
            print(set_name, params["name"], score["phase_balanced_rmse_dbhz"], score["validation_rmse_dbhz"])

    candidates = pd.DataFrame(rows).sort_values(
        ["phase_balanced_rmse_dbhz", "worst_phase_rmse_dbhz", "validation_rmse_dbhz"],
        kind="stable",
    ).reset_index(drop=True)
    candidates["selected"] = False
    candidates.loc[0, "selected"] = True
    candidates.to_csv(OUT_DIR / "validation_only_candidate_scores.csv", index=False)
    blocks.to_csv(OUT_DIR / "frozen_split_assignments.csv", index=False)

    winner = candidates.iloc[0]
    selected_params = next(item for item in PARAMETER_CANDIDATES if item["name"] == winner["parameter_set"])
    selection = {
        "selection_rule": "minimum phase-balanced validation trend RMSE; test and external holdout excluded",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "holdout_ops": list(engine.HOLDOUT_OPS),
        "selected_feature_set": str(winner["feature_set"]),
        "selected_features": sets[str(winner["feature_set"])],
        "selected_parameters": {key: value for key, value in selected_params.items() if key != "name"},
        "selected_validation_scores": {
            key: float(winner[key])
            for key in [
                "phase_balanced_rmse_dbhz",
                "worst_phase_rmse_dbhz",
                "validation_rmse_dbhz",
                "validation_mae_dbhz",
                "validation_bias_dbhz",
            ]
        },
        "candidate_count": int(len(candidates)),
        "forbidden_during_selection": ["test", engine.HOLDOUT_SPLIT_NAME],
    }
    (OUT_DIR / "selected_model.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
