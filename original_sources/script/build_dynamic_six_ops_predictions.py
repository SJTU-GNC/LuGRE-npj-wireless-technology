#!/usr/bin/env python3
"""Recompute the six-operation C/N0 trends with reconstructed receiver dynamics."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "runtime_cache" / "python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "script"))

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import build_lugre_dynamic_receiver_trajectory_1min as dynamic_receiver
import build_plot_wgc_multiband_cn0 as builder
import plot_full_mission_four_band_ai_cn0_sidelobe as frozen_model


OPS = ["OP2", "OP21", "OP27", "OP40", "OP74", "OP76"]
TEMPLATE = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_trend_residual_tuned_no_leakage"
    / "cn0_trend_residual_tuned_no_leakage_predictions.csv"
)
OUTPUT_DIR = ROOT / "table" / "algorithm" / "cn0_dynamic_six_ops_frozen_ai"
PHYSICS_PATH = OUTPUT_DIR / "cn0_dynamic_six_ops_physics.csv"
PREDICTION_PATH = OUTPUT_DIR / "cn0_dynamic_six_ops_predictions.csv"
AUDIT_PATH = OUTPUT_DIR / "cn0_dynamic_six_ops_geometry_audit.csv"


def interpolate_sp3_queries(query: pd.DataFrame, sp3: pd.DataFrame) -> pd.DataFrame:
    output = query.copy()
    output["sat_id"] = (
        output["system"].astype(str)
        + output["svid"].astype(int).astype(str).str.zfill(2)
    )
    output[["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]] = np.nan
    for sat_id, rows in output.groupby("sat_id", sort=False):
        source = (
            sp3[sp3["sat_id"].eq(sat_id)]
            .groupby("utc", as_index=False)[
                ["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]
            ]
            .mean()
            .sort_values("utc")
        )
        if len(source) < 4:
            continue
        source_seconds = source["utc"].map(lambda t: t.timestamp()).to_numpy(float)
        target_seconds = rows["utc"].map(lambda t: t.timestamp()).to_numpy(float)
        inside = (target_seconds >= source_seconds.min()) & (
            target_seconds <= source_seconds.max()
        )
        indices = rows.index.to_numpy()[inside]
        for column in ["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]:
            spline = CubicSpline(source_seconds, source[column].to_numpy(float))
            output.loc[indices, column] = spline(target_seconds[inside])
    output = output.dropna(
        subset=["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]
    )
    output[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]] = builder.ecef_to_eci(
        output
    )
    return output


def build_physics() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not dynamic_receiver.OUTPUT.exists():
        dynamic_receiver.build()

    template = pd.read_csv(
        TEMPLATE,
        usecols=[
            "minute_utc",
            "evaluation_split",
            "op",
            "mission_phase",
            "system",
            "signal_name",
            "svid",
        ],
        low_memory=False,
    )
    template = template[template["op"].isin(OPS)].copy()
    template["utc"] = pd.to_datetime(
        template["minute_utc"], utc=True, errors="coerce"
    ).dt.floor("min")
    template["svid"] = pd.to_numeric(template["svid"], errors="coerce").astype(
        "Int64"
    )
    template = template.dropna(subset=["utc", "svid"]).drop_duplicates(
        ["op", "utc", "signal_name", "svid"]
    )

    receiver = pd.read_csv(dynamic_receiver.OUTPUT, low_memory=False)
    receiver["utc"] = pd.to_datetime(receiver["utc"], utc=True, errors="coerce")
    receiver = (
        receiver.dropna(subset=["utc", "rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"])
        .drop_duplicates("utc")
        .sort_values("utc")
    )

    # Build every observed link on every natural one-minute dynamics epoch in
    # its operation window. Observation metadata are attached separately and
    # never become frozen-model inputs.
    windows = template.groupby("op", as_index=False).agg(
        start_utc=("utc", "min"),
        end_utc=("utc", "max"),
        mission_phase=("mission_phase", "first"),
    )
    link_catalog = template[
        ["op", "system", "signal_name", "svid"]
    ].drop_duplicates()
    query_parts: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        epochs = receiver[
            receiver["utc"].between(window.start_utc, window.end_utc)
        ].copy()
        links = link_catalog[link_catalog["op"].eq(window.op)].copy()
        if epochs.empty or links.empty:
            continue
        epochs["_cross"] = 1
        links["_cross"] = 1
        part = links.merge(epochs, on="_cross", how="inner").drop(columns="_cross")
        part["mission_phase"] = window.mission_phase
        query_parts.append(part)
    if not query_parts:
        raise RuntimeError("No continuous one-minute dynamic queries were built")
    query = pd.concat(query_parts, ignore_index=True)
    exact_split = template[
        ["op", "utc", "signal_name", "svid", "evaluation_split"]
    ].rename(columns={"evaluation_split": "template_evaluation_split"})
    query = query.merge(
        exact_split,
        on=["op", "utc", "signal_name", "svid"],
        how="left",
        validate="one_to_one",
    )
    query["observation_template_available"] = query[
        "template_evaluation_split"
    ].notna()
    query["evaluation_split"] = query["template_evaluation_split"].fillna(
        "geometry_only"
    )

    satellite_ids = set(
        query["system"].astype(str)
        + query["svid"].astype(int).astype(str).str.zfill(2)
    )
    sp3 = builder.parse_mgex_sp3(satellite_ids)
    query = interpolate_sp3_queries(query, sp3)
    spice_cache = builder.spice_geometry(query["utc"])
    observations = pd.read_csv(builder.OBS, low_memory=False)
    _, mapping, patterns = builder.available_satellites(observations)

    query["operation_id"] = query["op"]
    physics = builder.build_physics(query, spice_cache, mapping, patterns)
    physics["op"] = physics["operation_id"]
    physics["evaluation_split"] = physics["template_evaluation_split"]
    physics["minute_utc"] = physics["utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    physics["orbit_geometry_status"] = (
        "LuGRE_dynamics_reconstruction_plus_CODE_MGEX_final_SP3_cubic"
    )
    physics["gnss_orbit_source"] = "CODE_MGEX_final_5min_SP3_cubic_interpolation"
    physics["fspl_status"] = "computed_signal_frequency_dynamic_receiver_SP3_geometry"
    physics.to_csv(PHYSICS_PATH, index=False, encoding="utf-8-sig")

    audit = (
        physics.groupby(
            ["op", "dynamic_stage", "dynamic_quality_status"],
            as_index=False,
            dropna=False,
        )
        .agg(
            rows=("utc", "size"),
            epochs=("utc", "nunique"),
            start_utc=("utc", "min"),
            end_utc=("utc", "max"),
            satellites=("sat_id", "nunique"),
            median_reference_position_error_km=(
                "reference_position_error_km",
                "median",
            ),
        )
    )
    audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")
    return physics


def predict() -> pd.DataFrame:
    build_physics()
    original = frozen_model.CONTINUOUS_PATH
    frozen_model.CONTINUOUS_PATH = PHYSICS_PATH
    try:
        prediction = frozen_model.load_and_predict()
    finally:
        frozen_model.CONTINUOUS_PATH = original
    prediction["minute_utc"] = prediction["utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    # Attach observation-side bookkeeping only after inference. These columns
    # support plotting and metrics and are never passed to the frozen model.
    observation_columns = [
        "observation_time_median_utc",
        "source_bin_gps_seconds",
        "time_block_id",
        "cn0_dbhz_mean",
        "valid_seconds_in_bin",
        "coverage_fraction",
        "longest_gap_s",
        "reacquisition_flag",
        "trend_training_eligible",
    ]
    template = pd.read_csv(TEMPLATE, low_memory=False)
    template = template[template["op"].isin(OPS)].copy()
    template["minute_key"] = pd.to_datetime(
        template["minute_utc"], utc=True, errors="coerce"
    ).dt.floor("min")
    template["svid"] = pd.to_numeric(template["svid"], errors="coerce").astype(
        "Int64"
    )
    template = template.drop_duplicates(
        ["op", "minute_key", "signal_name", "svid"]
    )
    prediction["minute_key"] = prediction["utc"].dt.floor("min")
    prediction = prediction.merge(
        template[["op", "minute_key", "signal_name", "svid", *observation_columns]],
        on=["op", "minute_key", "signal_name", "svid"],
        how="left",
        validate="one_to_one",
    ).drop(columns="minute_key")
    prediction["cn0_ai_trend_dbhz"] = prediction[
        "cn0_physics_ai_trend_dbhz"
    ]
    prediction["ai_trend_residual_pred_db"] = prediction["ai_residual_trend_db"]
    observed = prediction["cn0_dbhz_mean"].notna()
    prediction["observation_template_available"] = observed
    prediction["geometry_role"] = np.where(
        observed,
        "dynamics_reconstruction_observation_aligned_1min",
        "dynamics_reconstruction_continuous_1min",
    )
    prediction["reference_role"] = np.where(
        observed,
        "LuGRE_C/N0_used_only_for_plot_validation",
        "no_reference_geometry_only_prediction",
    )
    prediction.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
    return prediction


def main() -> None:
    result = predict()
    print(f"prediction_rows={len(result):,}")
    print(result.groupby("op").size().to_string())
    print(PREDICTION_PATH)


if __name__ == "__main__":
    main()
