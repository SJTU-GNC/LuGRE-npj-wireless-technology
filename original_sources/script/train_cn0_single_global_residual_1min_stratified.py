#!/usr/bin/env python3
"""Train one global 1-minute C/N0 residual model with stratified arc splits.

Every OP may contribute samples to training, validation, and test. One-minute
samples are grouped into continuous link arcs before a seeded 70/15/15 split,
so adjacent samples from the same arc never cross subsets. The model is fitted
exactly once on training arcs and evaluated at original observation timestamps.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "script"))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import train_cn0_constellation_residual_models as source_mod


OUT_DIR = ROOT / "table" / "algorithm" / "cn0_single_global_residual_1min"
MODEL_PATH = OUT_DIR / "cn0_single_global_hgb_residual_1min.joblib"
CADENCE = "1min"
RANDOM_STATE = 42
ARC_GAP_MINUTES = 15.0
SPLIT_PROBABILITIES = {"train": 0.70, "validation": 0.15, "test": 0.15}

TARGET = source_mod.TARGET
BASELINE = source_mod.BASELINE_COL

IDENTITY_SHORTCUTS = {
    "system",
    "signal_name",
    "signal_id",
    "svid",
    "frequency_band",
    "mission_phase",
    "op",
    "tx_pattern_family",
    "tx_pattern_source",
    "tx_gnss_ssv_signal_band",
    "galileo_grap_lookup_status",
    "galileo_grap_coordinate_mapping_status",
    "gps_l1_direct_merge_status",
    "gps_tx_physics_status",
}

OBSERVATION_DERIVED_FORBIDDEN = {
    TARGET,
    "pseudorange_raw_m_mean",
    "doppler_raw_hz_mean",
    "samples",
    "residual_target_db",
    "split",
    "split_policy",
    "calibration_target_used",
    "heldout_for_validation",
    "rx_utc",
    "rx_utc_dt",
    "rx_gps_seconds",
    "time_bin_gps_seconds",
    "utc_day",
    "row_id_constellation",
}

QUALITY_FLAG_SPECS = {
    "quality_has_constellation_direct_budget": "cn0_constellation_direct_available_dbhz",
    "quality_has_grap_eirp": "tx_eirp_grap_azimuth_median_dbw",
    "quality_has_tx_gain_2d": "tx_gain_2d_db",
    "quality_has_rx_spice_attitude": "rx_offboresight_spice_deg",
    "quality_has_rx_gain_envelope": "rx_gain_envelope_dbic",
    "quality_has_reference_2d_budget": "cn0_reference_trajectory_2d_dbhz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> pd.DataFrame:
    data = source_mod.load_data().copy()
    data["op"] = data["op"].astype(str)
    for flag, source_column in QUALITY_FLAG_SPECS.items():
        data[flag] = data[source_column].notna().astype(float)
    data["quality_tx_yaw_nominal"] = (
        data.get("tx_yaw_quality", pd.Series("", index=data.index))
        .astype(str)
        .str.contains("nominal|reliable|good", case=False, regex=True, na=False)
        .astype(float)
    )
    data["quality_tx_phi_alignment_available"] = (
        ~data.get("tx_phi_alignment_quality", pd.Series("missing", index=data.index))
        .astype(str)
        .str.contains("missing|unavailable|unknown", case=False, regex=True, na=False)
    ).astype(float)

    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    numeric = [
        column
        for column in source_mod.BASE_NUMERIC
        if column in data.columns
        and column not in source_mod.FORBIDDEN_EXACT
        and column not in IDENTITY_SHORTCUTS
        and column not in OBSERVATION_DERIVED_FORBIDDEN
    ]
    numeric.extend(QUALITY_FLAG_SPECS)
    numeric.extend(["quality_tx_yaw_nominal", "quality_tx_phi_alignment_available"])
    features = list(dict.fromkeys(numeric))
    if "frequency_mhz" not in features:
        raise RuntimeError("frequency_mhz must remain as a physical input")
    forbidden = (IDENTITY_SHORTCUTS | OBSERVATION_DERIVED_FORBIDDEN).intersection(features)
    if forbidden:
        raise RuntimeError(f"Forbidden model inputs remain: {sorted(forbidden)}")
    derived = [name for name in features if name.startswith("error_") or "residual" in name.lower()]
    if derived:
        raise RuntimeError(f"Target-derived inputs remain: {derived}")
    return features


def aggregate_one_minute(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = data.copy()
    frame["minute_utc"] = frame["rx_utc_dt"].dt.floor(CADENCE)
    keys = [
        "op",
        "mission_phase",
        "system",
        "signal_name",
        "svid",
        "minute_utc",
    ]
    numeric = list(dict.fromkeys(features + [TARGET, BASELINE]))
    aggregations: dict[str, str] = {
        column: "median" for column in numeric if column in frame.columns and column not in keys
    }
    aggregations["raw_rows_in_bin"] = "sum"
    frame["raw_rows_in_bin"] = 1
    bins = frame.groupby(keys, dropna=False, as_index=False).agg(aggregations)
    bins["residual_target_db"] = bins[TARGET] - bins[BASELINE]
    return bins


def assign_arc_splits(bins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer continuous arcs and assign complete arcs with a seeded split."""
    output = bins.sort_values(["op", "signal_name", "svid", "minute_utc"]).copy()
    link_keys = ["op", "signal_name", "svid"]
    gap = output.groupby(link_keys)["minute_utc"].diff().dt.total_seconds().div(60.0)
    output["new_arc"] = gap.isna() | gap.gt(ARC_GAP_MINUTES)
    output["arc_number"] = output.groupby(link_keys)["new_arc"].cumsum().astype(int)
    output["arc_id_model"] = (
        output["op"].astype(str)
        + "|"
        + output["signal_name"].astype(str)
        + "|"
        + output["svid"].astype(str)
        + "|"
        + output["arc_number"].astype(str)
    )

    arcs = (
        output.groupby(
            ["arc_id_model", "mission_phase", "system", "signal_name"],
            as_index=False,
            dropna=False,
        )
        .agg(rows_1min=("minute_utc", "size"), start_utc=("minute_utc", "min"))
    )
    rng = np.random.default_rng(RANDOM_STATE)
    arcs["evaluation_split"] = ""
    strata = ["mission_phase", "signal_name"]
    for _, group in arcs.groupby(strata, dropna=False, sort=True):
        order = group.index.to_numpy(copy=True)
        rng.shuffle(order)
        draws = rng.random(len(order))
        assigned = np.where(
            draws < SPLIT_PROBABILITIES["train"],
            "train",
            np.where(
                draws < SPLIT_PROBABILITIES["train"] + SPLIT_PROBABILITIES["validation"],
                "validation",
                "test",
            ),
        )
        arcs.loc[order, "evaluation_split"] = assigned

        # Ensure each sufficiently populated phase-signal stratum is represented
        # in every subset without splitting an arc.
        if len(order) >= 3:
            present = set(arcs.loc[order, "evaluation_split"])
            for missing in ["train", "validation", "test"]:
                if missing in present:
                    continue
                donor_counts = arcs.loc[order, "evaluation_split"].value_counts()
                donor = str(donor_counts.index[0])
                candidates = arcs.loc[
                    order[arcs.loc[order, "evaluation_split"].eq(donor).to_numpy()]
                ].sort_values(["rows_1min", "arc_id_model"])
                arcs.loc[candidates.index[0], "evaluation_split"] = missing
                present.add(missing)

    split_map = arcs.set_index("arc_id_model")["evaluation_split"]
    output["evaluation_split"] = output["arc_id_model"].map(split_map)
    if output["evaluation_split"].eq("").any() or output["evaluation_split"].isna().any():
        raise RuntimeError("At least one one-minute sample has no split assignment")
    return output.drop(columns=["new_arc"]), arcs


def attach_splits_to_raw(raw: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["minute_utc"] = frame["rx_utc_dt"].dt.floor(CADENCE)
    keys = ["op", "mission_phase", "system", "signal_name", "svid", "minute_utc"]
    mapping = bins[keys + ["arc_id_model", "evaluation_split"]].drop_duplicates(keys)
    if mapping.duplicated(keys).any():
        raise RuntimeError("One-minute key maps to more than one split")
    frame = frame.merge(mapping, on=keys, how="left", validate="many_to_one")
    if frame["evaluation_split"].isna().any():
        raise RuntimeError("At least one raw observation has no split assignment")
    return frame


def make_model(features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [("physics", SimpleImputer(strategy="median"), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = HistGradientBoostingRegressor(
        max_iter=360,
        learning_rate=0.035,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def metric_row(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    split: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    observed = frame[TARGET].to_numpy(float)
    error = prediction - observed
    return {
        "evaluation_split": split,
        "group_type": group_type,
        "group_value": group_value,
        "n": int(len(frame)),
        "mae_dbhz": float(mean_absolute_error(observed, prediction)),
        "rmse_dbhz": float(math.sqrt(mean_squared_error(observed, prediction))),
        "bias_dbhz": float(np.mean(error)),
        "median_abs_error_dbhz": float(np.median(np.abs(error))),
        "p95_abs_error_dbhz": float(np.percentile(np.abs(error), 95)),
        "pearson_r": (
            float(np.corrcoef(observed, prediction)[0, 1])
            if len(frame) > 1 and np.std(observed) > 0 and np.std(prediction) > 0
            else np.nan
        ),
        "r2": float(r2_score(observed, prediction)) if len(frame) > 1 else np.nan,
    }


def evaluate(
    raw: pd.DataFrame,
    model: Pipeline,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for split in ["validation", "test"]:
        part = raw[raw["evaluation_split"].eq(split)].copy()
        residual_pred = np.asarray(model.predict(part[features]), dtype=float)
        prediction = part[BASELINE].to_numpy(float) + residual_pred
        output = pd.DataFrame(
            {
                "row_id_constellation": part.get(
                    "row_id_constellation", pd.Series(part.index, index=part.index)
                ).to_numpy(),
                "rx_utc": part["rx_utc"].to_numpy(),
                "evaluation_split": split,
                "op": part["op"].to_numpy(),
                "mission_phase": part["mission_phase"].to_numpy(),
                "system": part["system"].to_numpy(),
                "signal_name": part["signal_name"].to_numpy(),
                "svid": part["svid"].to_numpy(),
                "cn0_observed_dbhz": part[TARGET].to_numpy(float),
                "cn0_physics_dbhz": part[BASELINE].to_numpy(float),
                "ai_residual_pred_db": residual_pred,
                "cn0_physics_ai_dbhz": prediction,
                "signed_error_dbhz": prediction - part[TARGET].to_numpy(float),
                "abs_error_dbhz": np.abs(prediction - part[TARGET].to_numpy(float)),
                "model_id": "single_global_hgb_residual_1min_stratified_arc",
            }
        )
        prediction_frames.append(output)
        metric_rows.append(metric_row(part, prediction, split, "overall", "all"))
        for column in ["op", "mission_phase", "system", "signal_name"]:
            for value, group in part.groupby(column, dropna=False):
                indices = part.index.get_indexer(group.index)
                metric_rows.append(
                    metric_row(group, prediction[indices], split, column, str(value))
                )
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(metric_rows)


def split_inventory(raw: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    bin_counts = bins.groupby(["evaluation_split", "op"]).size().rename("rows_1min")
    rows = []
    for (split, op, phase), group in raw.groupby(
        ["evaluation_split", "op", "mission_phase"], dropna=False
    ):
        rows.append(
            {
                "evaluation_split": split,
                "op": op,
                "mission_phase": phase,
                "raw_rows": int(len(group)),
                "rows_1min": int(bin_counts.get((split, op), 0)),
                "start_utc": group["rx_utc_dt"].min().isoformat(),
                "end_utc": group["rx_utc_dt"].max().isoformat(),
                "signals": "|".join(sorted(group["signal_name"].astype(str).unique())),
                "satellites": int(group["svid"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["evaluation_split", "start_utc"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_raw = load_data()
    features = feature_columns(source_raw)
    bins = aggregate_one_minute(source_raw, features)
    bins, arcs = assign_arc_splits(bins)
    raw = attach_splits_to_raw(source_raw, bins)
    train_bins = bins[bins["evaluation_split"].eq("train")].copy()
    if train_bins.empty:
        raise RuntimeError("Training split contains no one-minute bins")

    model = make_model(features)
    model.fit(train_bins[features], train_bins["residual_target_db"])
    joblib.dump(model, MODEL_PATH)

    predictions, metrics = evaluate(raw, model, features)
    inventory = split_inventory(raw, bins)
    feature_inventory = pd.DataFrame(
        {
            "feature": features,
            "feature_type": "numeric_physics_or_generic_availability",
            "allowed_in_global_model": True,
        }
    )

    arc_split_counts = arcs.groupby("arc_id_model")["evaluation_split"].nunique()
    minute_split_counts = bins.groupby(
        ["op", "signal_name", "svid", "minute_utc"]
    )["evaluation_split"].nunique()
    phase_signal_coverage = (
        arcs.groupby(["mission_phase", "signal_name"])["evaluation_split"]
        .nunique()
        .reset_index(name="subsets_present")
    )
    leakage_checks = [
        {
            "check": "complete_arcs_are_disjoint",
            "status": "pass" if arc_split_counts.max() == 1 else "fail",
            "evidence": f"arcs={len(arc_split_counts)}; max_subsets_per_arc={arc_split_counts.max()}",
        },
        {
            "check": "one_minute_keys_are_disjoint",
            "status": "pass" if minute_split_counts.max() == 1 else "fail",
            "evidence": f"minute_keys={len(minute_split_counts)}; max_subsets_per_key={minute_split_counts.max()}",
        },
        {
            "check": "model_fitted_once_on_training_arcs_only",
            "status": "pass",
            "evidence": "one Pipeline.fit call; validation/test are inference only",
        },
        {
            "check": "one_minute_training_scale",
            "status": "pass",
            "evidence": f"CADENCE={CADENCE}; train_bins={len(train_bins)}",
        },
        {
            "check": "seeded_stratified_arc_split",
            "status": "pass",
            "evidence": (
                f"seed={RANDOM_STATE}; probabilities={SPLIT_PROBABILITIES}; "
                f"phase_signal_strata={len(phase_signal_coverage)}"
            ),
        },
        {
            "check": "identity_shortcuts_excluded",
            "status": "pass" if not IDENTITY_SHORTCUTS.intersection(features) else "fail",
            "evidence": "no OP/SVID/system/signal/phase categorical identity inputs",
        },
        {
            "check": "observed_cn0_and_residual_excluded",
            "status": "pass"
            if not OBSERVATION_DERIVED_FORBIDDEN.intersection(features)
            else "fail",
            "evidence": "target and residual are labels only",
        },
    ]
    leakage = pd.DataFrame(leakage_checks)
    if not leakage["status"].eq("pass").all():
        raise RuntimeError("Leakage audit failed")

    predictions.to_csv(
        OUT_DIR / "cn0_single_global_1min_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        OUT_DIR / "cn0_single_global_1min_metrics.csv", index=False, encoding="utf-8-sig"
    )
    inventory.to_csv(
        OUT_DIR / "cn0_single_global_1min_op_split.csv", index=False, encoding="utf-8-sig"
    )
    arcs.to_csv(
        OUT_DIR / "cn0_single_global_1min_arc_split.csv", index=False, encoding="utf-8-sig"
    )
    feature_inventory.to_csv(
        OUT_DIR / "cn0_single_global_1min_features.csv", index=False, encoding="utf-8-sig"
    )
    leakage.to_csv(
        OUT_DIR / "cn0_single_global_1min_leakage_audit.csv", index=False, encoding="utf-8-sig"
    )

    summary = raw.groupby("evaluation_split").agg(
        raw_rows=("op", "size"), ops=("op", "nunique")
    )
    summary["raw_fraction"] = summary["raw_rows"] / len(raw)
    bin_summary = bins.groupby("evaluation_split").size().rename("rows_1min")
    summary = summary.join(bin_summary)
    model_card = {
        "objective": "one global all-band physics residual model",
        "cadence": CADENCE,
        "algorithm": "HistGradientBoostingRegressor",
        "fit_policy": (
            "one fit on seeded stratified training arcs only; frozen for validation and test"
        ),
        "split_policy": {
            "probabilities": SPLIT_PROBABILITIES,
            "seed": RANDOM_STATE,
            "strata": ["mission_phase", "signal_name"],
            "group": "continuous OP-signal-SVID arc",
            "arc_gap_minutes": ARC_GAP_MINUTES,
            "note": "every OP may contribute arcs to multiple subsets; no arc is split",
        },
        "features": features,
        "source": str(source_mod.INPUT),
        "source_sha256": sha256(source_mod.INPUT),
        "model_sha256": sha256(MODEL_PATH),
        "split_summary": summary.reset_index().to_dict("records"),
    }
    (OUT_DIR / "SingleGlobalResidualModelCard.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Single global 1-minute stratified-arc residual model complete.")
    print(summary.to_string())
    print("\nOverall validation/test metrics:")
    print(metrics[metrics["group_type"].eq("overall")].to_string(index=False))
    print("\nPer-OP test metrics:")
    print(
        metrics[
            metrics["evaluation_split"].eq("test") & metrics["group_type"].eq("op")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
