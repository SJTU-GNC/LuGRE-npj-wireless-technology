#!/usr/bin/env python3
"""Simulate continuous GPS L1 C/N0 on the official reference trajectory.

The time grid comes from daily 15-minute IGS final SP3 epochs, not LuGRE
observation times. Blue Ghost position comes from the official CLPS SPICE
trajectory chain. SPICE receiver attitude is used where available; documented
Earth-pointing is the trend-only fallback across CK gaps.
"""

from __future__ import annotations

import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy as sp

from build_cn0_attitude_2d_gain import (
    EARTH_RADIUS_KM,
    K_DB,
    RX_TSYS_K,
    TX_POWER_DBW,
    bilinear_periodic,
    load_spice,
    normalize,
    parse_iiia_2d,
    parse_iirm_2d,
)


SP3_DIR = Path("data/external_reference/precise_orbit/igs_bkg_opsfin")
SPICE_DIR = Path("data/external_reference/spice")
GPS_DIR = Path("data/external_reference/gnss_antenna/gps")
MAPPING = Path("data/external_reference/mapping/active_prn_svn_block_20250115_20250316.csv")
RX_PATTERN = Path("data/external_reference/lugre_antenna/LuGRE_Fig3_gain_outer_envelope_digitized.csv")
OUTPUT_DIR = Path("table/algorithm/cn0_continuous_reference")
MISSION_START = pd.Timestamp("2025-01-15T07:32:00Z")
MISSION_STOP = pd.Timestamp("2025-03-16T23:15:00Z")
GPS_UTC_OFFSET_S = 18.0
MOON_RADIUS_KM = 1737.4


def parse_sp3(files: list[Path], svids: set[int]) -> pd.DataFrame:
    rows = []
    for path in files:
        epoch = None
        with gzip.open(path, "rt", errors="replace") as stream:
            for line in stream:
                if line.startswith("*"):
                    parts = line[1:].split()
                    gps_time = datetime(
                        int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]),
                        int(parts[4]), tzinfo=timezone.utc,
                    ) + timedelta(seconds=float(parts[5]))
                    epoch = pd.Timestamp(gps_time - timedelta(seconds=GPS_UTC_OFFSET_S))
                elif epoch is not None and line.startswith("PG"):
                    svid = int(line[2:4])
                    if svid not in svids:
                        continue
                    xyz = [float(line[4:18]), float(line[18:32]), float(line[32:46])]
                    if max(abs(value) for value in xyz) >= 999999.0:
                        continue
                    rows.append({"utc": epoch, "svid": svid, "sat_x_ecef_km": xyz[0], "sat_y_ecef_km": xyz[1], "sat_z_ecef_km": xyz[2]})
    result = pd.DataFrame(rows)
    return result[(result["utc"] >= MISSION_START) & (result["utc"] <= MISSION_STOP)].reset_index(drop=True)


def ecef_to_eci(df: pd.DataFrame) -> np.ndarray:
    # Pandas datetime integer units are version-dependent in this environment
    # (microseconds rather than nanoseconds in newer builds). Use Timestamp
    # conversion so the sidereal rotation is always driven by true UTC seconds.
    unix = pd.to_datetime(df["utc"], utc=True).map(lambda t: t.timestamp()).to_numpy(dtype=float)
    jd = unix / 86400.0 + 2440587.5
    centuries = (jd - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * centuries**2 - centuries**3 / 38710000.0
    theta = np.radians(np.mod(gmst_deg, 360.0))
    c, s = np.cos(theta), np.sin(theta)
    ecef = df[["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]].to_numpy(float)
    return np.column_stack([c * ecef[:, 0] - s * ecef[:, 1], s * ecef[:, 0] + c * ecef[:, 1], ecef[:, 2]])


def spice_reference(times: pd.Series) -> dict[pd.Timestamp, dict]:
    cache = {}
    for value in sorted(times.unique()):
        timestamp = pd.Timestamp(value)
        et = sp.utc2et(timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        rx = np.asarray(sp.spkpos("BGM1", et, "J2000", "NONE", "EARTH")[0], dtype=float)
        sun = np.asarray(sp.spkpos("SUN", et, "J2000", "NONE", "EARTH")[0], dtype=float)
        moon = np.asarray(sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")[0], dtype=float)
        try:
            rotation = np.asarray(sp.pxform("BGM1_LUGRE", "J2000", et), dtype=float).T
            attitude = "spice_attitude"
        except Exception:
            rotation = None
            attitude = "nominal_earth_pointing_fallback"
        cache[timestamp] = {"rx": rx, "sun": sun, "moon": moon, "rotation": rotation, "attitude": attitude}
    return cache


def sphere_blocked(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    start = p0 - center
    direction = p1 - p0
    denom = np.sum(direction * direction, axis=1)
    t = -np.sum(start * direction, axis=1) / denom
    closest = start + np.clip(t, 0.0, 1.0)[:, None] * direction
    outside = np.linalg.norm(start, axis=1) > radius + 0.5
    blocked_outside = outside & (t > 0.0) & (t < 1.0) & (np.linalg.norm(closest, axis=1) < radius)
    blocked_surface = (~outside) & (np.sum(start * direction, axis=1) < 0.0)
    return blocked_outside | blocked_surface


def rx_envelope(theta: np.ndarray) -> np.ndarray:
    pattern = pd.read_csv(RX_PATTERN)
    pattern = pattern[pattern["band"].eq("L1/E1")].sort_values("signed_elevation_deg")
    axis = pattern["signed_elevation_deg"].to_numpy(float)
    gain = pattern["outer_envelope_gain_dB_axis"].to_numpy(float)
    return np.maximum(np.interp(theta, axis, gain), np.interp(-theta, axis, gain))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPPING)
    mapping = mapping[mapping["system"].eq("G")].copy()
    mapping["svid"] = mapping["prn"].str.extract(r"G(\d+)").astype(int)
    mapping["svn"] = mapping["sv_identifier"].str.extract(r"G(\d+)").astype(int)
    supported = mapping[mapping["antenna_type"].isin(TX_POWER_DBW)].copy()

    patterns = parse_iirm_2d(GPS_DIR / "AppBAntennaPanelPatterns.pptx")
    patterns.update(parse_iiia_2d(GPS_DIR))
    supported = supported[supported["svn"].isin(patterns)].copy()
    svids = set(supported["svid"].astype(int))
    df = parse_sp3(sorted(SP3_DIR.glob("*ORB.SP3.gz")), svids)
    df = df.merge(supported[["svid", "svn", "antenna_type"]], on="svid", how="left")
    sat = ecef_to_eci(df)
    df[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]] = sat

    load_spice(SPICE_DIR)
    try:
        reference = spice_reference(df["utc"])
    finally:
        sp.kclear()
    rx = np.vstack([reference[pd.Timestamp(t)]["rx"] for t in df["utc"]])
    sun = np.vstack([reference[pd.Timestamp(t)]["sun"] for t in df["utc"]])
    moon = np.vstack([reference[pd.Timestamp(t)]["moon"] for t in df["utc"]])
    rotations = [reference[pd.Timestamp(t)]["rotation"] for t in df["utc"]]
    df["receiver_attitude_source"] = [reference[pd.Timestamp(t)]["attitude"] for t in df["utc"]]
    df[["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]] = rx

    los_rx, ranges = normalize(sat - rx)
    los_tx = -los_rx
    rx_theta = np.full(len(df), np.nan)
    for index, rotation in enumerate(rotations):
        if rotation is not None:
            local = rotation @ los_rx[index]
            rx_theta[index] = np.degrees(np.arccos(np.clip(local[2], -1.0, 1.0)))
        else:
            earth_direction = -rx[index] / np.linalg.norm(rx[index])
            rx_theta[index] = np.degrees(np.arccos(np.clip(np.dot(los_rx[index], earth_direction), -1.0, 1.0)))
    df["rx_offboresight_deg"] = rx_theta
    df["rx_gain_envelope_dbic"] = rx_envelope(rx_theta)

    z_body, _ = normalize(-sat)
    sun_direction, _ = normalize(sun - sat)
    projection = sun_direction - np.sum(sun_direction * z_body, axis=1)[:, None] * z_body
    x_body, projection_norm = normalize(projection)
    y_body, _ = normalize(np.cross(z_body, x_body))
    tx_x = np.sum(los_tx * x_body, axis=1)
    tx_y = np.sum(los_tx * y_body, axis=1)
    tx_z = np.sum(los_tx * z_body, axis=1)
    df["tx_theta_deg"] = np.degrees(np.arccos(np.clip(tx_z, -1.0, 1.0)))
    df["tx_phi_deg"] = np.mod(np.degrees(np.arctan2(tx_y, tx_x)), 360.0)
    df["yaw_singularity_angle_deg"] = np.degrees(np.arcsin(np.clip(projection_norm, 0.0, 1.0)))
    df["yaw_quality"] = np.where(df["yaw_singularity_angle_deg"] < 5.0, "turn_risk", "nominal")

    df["tx_gain_2d_db"] = np.nan
    for svn, pattern in patterns.items():
        mask = df["svn"].eq(svn)
        df.loc[mask, "tx_gain_2d_db"] = bilinear_periodic(
            pattern, df.loc[mask, "tx_theta_deg"].to_numpy(float), df.loc[mask, "tx_phi_deg"].to_numpy(float)
        )
    df["tx_power_dbw"] = df["antenna_type"].map(TX_POWER_DBW)
    df["range_km"] = ranges
    df["fspl_db"] = 32.44 + 20.0 * np.log10(ranges) + 20.0 * np.log10(1575.42)
    df["cn0_continuous_potential_dbhz"] = (
        df["tx_power_dbw"] + df["tx_gain_2d_db"] + df["rx_gain_envelope_dbic"]
        - df["fspl_db"] + K_DB - 10.0 * np.log10(RX_TSYS_K["GPS_L1"])
    )
    earth_center = np.zeros_like(rx)
    df["earth_blocked"] = sphere_blocked(rx, sat, earth_center, EARTH_RADIUS_KM)
    df["moon_blocked"] = sphere_blocked(rx, sat, moon, MOON_RADIUS_KM)
    df["line_of_sight_visible"] = ~(df["earth_blocked"] | df["moon_blocked"])
    df["cn0_visible_dbhz"] = df["cn0_continuous_potential_dbhz"].where(df["line_of_sight_visible"])
    df["trajectory_source"] = "official_CLPS_SPICE_reference_chain"
    df["satellite_orbit_source"] = "IGS_final_SP3_15min"
    df["simulation_use"] = "continuous_trend_only_no_observation_gate"
    df.to_csv(OUTPUT_DIR / "cn0_continuous_reference_15min.csv", index=False)
    summary = df.groupby(["svid", "svn", "antenna_type"], as_index=False).agg(
        rows=("utc", "size"), visible_rows=("line_of_sight_visible", "sum"),
        cn0_median_dbhz=("cn0_continuous_potential_dbhz", "median"),
        cn0_min_dbhz=("cn0_continuous_potential_dbhz", "min"),
        cn0_max_dbhz=("cn0_continuous_potential_dbhz", "max"),
    )
    summary.to_csv(OUTPUT_DIR / "cn0_continuous_reference_satellite_summary.csv", index=False)
    print(f"rows={len(df)} epochs={df['utc'].nunique()} satellites={df['svid'].nunique()}")
    print(df["receiver_attitude_source"].value_counts().to_dict())
    print(f"visible_fraction={df['line_of_sight_visible'].mean():.4f}")


if __name__ == "__main__":
    main()
