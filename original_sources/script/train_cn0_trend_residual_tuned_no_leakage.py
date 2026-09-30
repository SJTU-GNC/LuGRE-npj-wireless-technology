#!/usr/bin/env python3
"""Fit the validation-selected model, then evaluate untouched test/OP holdouts once."""

from __future__ import annotations

import json
import sys
from pathlib import Path


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
SELECTION_PATH = ROOT / "table" / "algorithm" / "cn0_trend_residual_tuning_no_leakage" / "selected_model.json"
source_model.INPUT = SOURCE

import train_cn0_single_global_residual_1min_op74_holdout as engine  # noqa: E402


selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
selected_features = list(selection["selected_features"])
selected_parameters = dict(selection["selected_parameters"])


def selected_feature_function(_: list[str]) -> list[str]:
    return selected_features


def selected_model_function(features: list[str]) -> Pipeline:
    if features != selected_features:
        raise RuntimeError("Selected feature contract changed")
    preprocess = ColumnTransformer(
        [("physics", SimpleImputer(strategy="median"), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = HistGradientBoostingRegressor(
        **selected_parameters,
        random_state=engine.RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


OUT_DIR = ROOT / "table" / "algorithm" / "cn0_trend_residual_tuned_no_leakage"
engine.OUT_DIR = OUT_DIR
engine.MODEL_PATH = OUT_DIR / "cn0_trend_residual_tuned_no_leakage.joblib"
engine.SMOOTH_TREND_MODEL = True
engine.TREND_WINDOW_MINUTES = 9
engine.HOLDOUT_OPS = ("OP2", "OP21", "OP27", "OP74")
engine.HOLDOUT_SPLIT_NAME = "external_holdout"
engine.REPRESENTATIVE_TEST_OPS = ()
engine.OUTPUT_PREDICTIONS_FILENAME = "cn0_trend_residual_tuned_no_leakage_predictions.csv"
engine.OUTPUT_METRICS_FILENAME = "cn0_trend_residual_tuned_no_leakage_metrics.csv"
engine.OUTPUT_LEAKAGE_FILENAME = "cn0_trend_residual_tuned_no_leakage_audit.csv"
engine.MODEL_CARD_FILENAME = "CN0TrendResidualTunedNoLeakageModelCard.json"
engine.RUN_OBJECTIVE = (
    "phase-balanced validation-selected trend residual model; internal test and complete OP2/OP21/OP27/OP74 "
    "external holdouts excluded from candidate ranking"
)
engine.smooth_model_features = selected_feature_function
engine.make_smooth_model = selected_model_function


if __name__ == "__main__":
    engine.main()
