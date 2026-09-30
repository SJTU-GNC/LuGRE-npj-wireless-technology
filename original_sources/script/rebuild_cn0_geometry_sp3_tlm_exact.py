from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy as sp
from scipy.interpolate import CubicSpline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from build_cn0_attitude_2d_gain import load_spice  # noqa: E402
from build_link_geometry_features import (  # noqa: E402
    EARTH_RADIUS_KM,
    MOON_RADIUS_KM,
    angle_deg,
    closest_approach_altitude_km,
    limb_margin_deg,
    line_intersects_sphere,
)
from build_plot_wgc_multiband_cn0 import parse_mgex_sp3  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "table" / "cn0_physics_features_available_wgc_new.csv"
DEFAULT_RECEIVER = (
    PROJECT_ROOT
    / "data"
    / "trajectory"
    / "wgc_lugre_target_earth_observer_by_tlm"
    / "wgc_state_vectors_lugre_target_earth_observer_by_tlm_all.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "table" / "cn0_physics_features_sp3_tlm_exact.csv"
SPICE_DIR = PROJECT_ROOT / "data" / "external_reference" / "spice"

RX_POS_COLS = [
    "lugre_rel_earth_x_j2000_km",
    "lugre_rel_earth_y_j2000_km",
    "lugre_rel_earth_z_j2000_km",
]
RX_VEL_COLS = [
    "lugre_rel_earth_vx_j2000_km_s",
    "lugre_rel_earth_vy_j2000_km_s",
    "lugre_rel_earth_vz_j2000_km_s",
]


def timestamp_seconds(values: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    return (pd.DatetimeIndex(values).asi8 - origin.value) / 1.0e9


def interpolate_receiver(receiver_path: Path, query_times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    usecols = ["utc", *RX_POS_COLS, *RX_VEL_COLS]
    receiver = pd.read_csv(receiver_path, usecols=usecols, low_memory=False)
    receiver["utc"] = pd.to_datetime(receiver["utc"], utc=True, errors="coerce")
    receiver = receiver.dropna(subset=usecols).groupby("utc", as_index=False)[RX_POS_COLS + RX_VEL_COLS].mean()
    receiver = receiver.sort_values("utc")

    origin = pd.Timestamp(query_times.min())
    source_seconds = timestamp_seconds(receiver["utc"], origin)
    query_seconds = timestamp_seconds(query_times, origin)
    insertion = np.searchsorted(source_seconds, query_seconds)
    bracketed = (insertion > 0) & (insertion < len(source_seconds))
    gap_seconds = np.full(len(query_seconds), np.inf)
    gap_seconds[bracketed] = source_seconds[insertion[bracketed]] - source_seconds[insertion[bracketed] - 1]
    if not np.all(bracketed & (gap_seconds <= 2.0)):
        bad = int(np.sum(~(bracketed & (gap_seconds <= 2.0))))
        raise RuntimeError(f"TLM-aligned receiver states do not tightly bracket {bad} query epochs")

    values = receiver[RX_POS_COLS + RX_VEL_COLS].to_numpy(float)
    interpolated = np.column_stack(
        [np.interp(query_seconds, source_seconds, values[:, column]) for column in range(values.shape[1])]
    )
    return interpolated[:, :3], interpolated[:, 3:]


def interpolate_sp3(
    features: pd.DataFrame, query_times: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[pd.Timestamp, pd.Timestamp]]]:
    sat_ids = (features["system"] + features["svid"].astype(int).astype(str).str.zfill(2)).to_numpy()
    sp3 = parse_mgex_sp3(set(sat_ids))
    if sp3.empty:
        raise RuntimeError("No CODE MGEX final SP3 records were parsed")

    origin = pd.Timestamp(query_times.min())
    query_seconds = timestamp_seconds(query_times, origin)
    sat_ecef = np.full((len(features), 3), np.nan)
    coverage: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    for sat_id in sorted(set(sat_ids)):
        source = sp3.loc[sp3["sat_id"].eq(sat_id)].copy()
        source = source.groupby("utc", as_index=False)[
            ["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]
        ].mean().sort_values("utc")
        if len(source) < 4:
            raise RuntimeError(f"Insufficient SP3 epochs for {sat_id}: {len(source)}")
        coverage[sat_id] = (source["utc"].iloc[0], source["utc"].iloc[-1])
        source_seconds = timestamp_seconds(source["utc"], origin)
        row_index = np.flatnonzero(sat_ids == sat_id)
        sat_query_seconds = query_seconds[row_index]
        if sat_query_seconds.min() < source_seconds.min() or sat_query_seconds.max() > source_seconds.max():
            raise RuntimeError(f"SP3 does not cover all query epochs for {sat_id}")
        for axis, column in enumerate(["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]):
            spline = CubicSpline(source_seconds, source[column].to_numpy(float))
            sat_ecef[row_index, axis] = spline(sat_query_seconds)
    return sat_ecef, sat_ids, coverage


def spice_geometry(query_times: pd.DatetimeIndex, sat_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sat_eci = np.empty_like(sat_ecef)
    moon_eci = np.empty_like(sat_ecef)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row, timestamp in enumerate(query_times):
        key = int(timestamp.value)
        if key not in cache:
            et = sp.utc2et(timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f"))
            rotation = np.asarray(sp.pxform("ITRF93", "J2000", et), dtype=float)
            moon = np.asarray(sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")[0], dtype=float)
            cache[key] = (rotation, moon)
        rotation, moon = cache[key]
        sat_eci[row] = rotation @ sat_ecef[row]
        moon_eci[row] = moon
    return sat_eci, moon_eci


def rebuild(input_path: Path, receiver_path: Path, output_path: Path) -> pd.DataFrame:
    features = pd.read_csv(input_path, low_memory=False)
    features["rx_utc"] = pd.to_datetime(features["rx_utc"], utc=True, errors="raise")
    query_times = pd.DatetimeIndex(features["rx_utc"])

    rx_eci, rx_velocity = interpolate_receiver(receiver_path, query_times)
    sat_ecef, _, _ = interpolate_sp3(features, query_times)

    load_spice(SPICE_DIR.relative_to(PROJECT_ROOT))
    try:
        sat_eci, moon_eci = spice_geometry(query_times, sat_ecef)
    finally:
        sp.kclear()

    los = sat_eci - rx_eci
    ranges = np.linalg.norm(los, axis=1)
    los_unit = los / ranges[:, None]
    frequencies = features["frequency_mhz"].to_numpy(float)

    features[["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]] = rx_eci
    features[["rx_vx_eci_km_s", "rx_vy_eci_km_s", "rx_vz_eci_km_s"]] = rx_velocity
    features[["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]] = sat_ecef
    features[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]] = sat_eci
    features["geometric_range_km"] = ranges
    features["range_rate_rx_only_km_s"] = np.einsum("ij,ij->i", los_unit, -rx_velocity)
    features["fspl_db"] = 32.44 + 20.0 * np.log10(ranges) + 20.0 * np.log10(frequencies)
    features["tx_offboresight_deg"] = [angle_deg(-sat, rx - sat) for rx, sat in zip(rx_eci, sat_eci)]

    earth_center = np.zeros(3)
    features["rx_to_sat_earth_center_sep_deg"] = [angle_deg(sat - rx, -rx) for rx, sat in zip(rx_eci, sat_eci)]
    features["rx_to_sat_moon_center_sep_deg"] = [
        angle_deg(sat - rx, moon - rx) for rx, sat, moon in zip(rx_eci, sat_eci, moon_eci)
    ]
    features["earth_observer_altitude_km"] = np.linalg.norm(rx_eci, axis=1) - EARTH_RADIUS_KM
    features["moon_observer_altitude_km"] = np.linalg.norm(rx_eci - moon_eci, axis=1) - MOON_RADIUS_KM
    features["earth_limb_margin_deg"] = [
        limb_margin_deg(rx, sat, earth_center, EARTH_RADIUS_KM) for rx, sat in zip(rx_eci, sat_eci)
    ]
    features["moon_limb_margin_deg"] = [
        limb_margin_deg(rx, sat, moon, MOON_RADIUS_KM) for rx, sat, moon in zip(rx_eci, sat_eci, moon_eci)
    ]
    features["earth_grazing_altitude_km"] = [
        closest_approach_altitude_km(rx, sat, earth_center, EARTH_RADIUS_KM) for rx, sat in zip(rx_eci, sat_eci)
    ]
    features["moon_grazing_altitude_km"] = [
        closest_approach_altitude_km(rx, sat, moon, MOON_RADIUS_KM)
        for rx, sat, moon in zip(rx_eci, sat_eci, moon_eci)
    ]
    features["earth_blocked"] = np.asarray([
        line_intersects_sphere(rx, sat, earth_center, EARTH_RADIUS_KM) for rx, sat in zip(rx_eci, sat_eci)
    ], dtype=int)
    features["moon_blocked"] = np.asarray([
        line_intersects_sphere(rx, sat, moon, MOON_RADIUS_KM) for rx, sat, moon in zip(rx_eci, sat_eci, moon_eci)
    ], dtype=int)

    features["nav_toe_gps_seconds"] = np.nan
    features["nav_age_seconds"] = np.nan
    features["feature_set"] = "cn0_physics_sp3_tlm_exact"
    features["orbit_geometry_status"] = "CODE_MGEX_final_SP3_cubic_TLM_exact_WGC"
    features["geometry_frame_note"] = (
        "rx=TLM-aligned WGC geocentric J2000; sat=CODE MGEX final SP3 ITRF93 transformed "
        "to J2000 by SPICE"
    )
    features["receiver_trajectory_source"] = "WGC_by_TLM_exact_approximately_1s"
    features["gnss_orbit_source"] = "CODE_MGEX_final_5min_SP3_cubic_interpolation"
    features["earth_orientation_source"] = "SPICE_ITRF93_to_J2000"
    features["moon_ephemeris_source"] = "JPL_DE440_SPICE"

    features["rx_utc"] = features["rx_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade LuGRE C/N0 geometry to exact WGC-by-TLM and final SP3")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--receiver", type=Path, default=DEFAULT_RECEIVER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = rebuild(args.input, args.receiver, args.output)
    print(f"Wrote {len(result):,} rows to {args.output}")
    print(result[["geometric_range_km", "fspl_db", "tx_offboresight_deg"]].describe().to_string())


if __name__ == "__main__":
    main()
