#!/usr/bin/env python3
"""Build four-band C/N0 on WGC_new epochs and plot paired LuGRE validation.

Receiver states come from the standardized 1-min WGC_new reference trajectory.
GPS and Galileo transmitter states come from CODE MGEX final 5-min SP3
products. LuGRE C/N0 observations are used only after prediction, for paired
validation; their exact receiver states are maintained separately in the
TLM-aligned WGC state-vector archive.
"""

from __future__ import annotations

import gzip
import io
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "runtime_cache" / "python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "script"))

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as sp
from pypdf import PdfReader

from build_cn0_attitude_2d_gain import (
    EARTH_RADIUS_KM,
    GNSS_SSV_BOUNDARY_BASIS,
    GNSS_SSV_MAIN_LOBE_BOUNDARY_DEG,
    K_DB,
    RX_TSYS_K,
    TX_POWER_DBW,
    bilinear_periodic,
    load_spice,
    normalize,
    parse_iiia_2d,
    parse_iirm_2d,
)
from build_cn0_constellation_physics_baseline import (
    add_edge_risk_features,
    add_ionosphere_absorption_proxy,
    add_tropospheric_gas_absorption_proxy,
)
from plot_wgc_reference_time_cn0_best_model import (
    limb_margin_deg,
    mission_phase,
    sphere_grazing_altitude,
)
from simulate_continuous_cn0_reference import ecef_to_eci, sphere_blocked
from orbex_attitude import DEFAULT_ORBEX_DIR, orbex_body_angles
from gps_iif_antenna_patterns import parse_iif_patterns
import train_cn0_general_residual_models as train_mod


WGC = ROOT / "table/receiver_state_vectors_wgc_new.csv"
MGEX = ROOT / "data/external_reference/precise_orbit/code_mgex_fin"
MAPPING = ROOT / "data/external_reference/mapping/active_prn_svn_block_20250115_20250316.csv"
GPS_DIR = ROOT / "data/external_reference/gnss_antenna/gps"
GRAP = ROOT / "table/external_reference/galileo_grap_eirp_grid.csv"
RX_PATTERN = ROOT / "data/external_reference/lugre_antenna/LuGRE_Fig3_gain_outer_envelope_digitized.csv"
SPICE_DIR = Path("data/external_reference/spice")
OBS = ROOT / "table/algorithm/cn0_constellation_physics_baseline/cn0_constellation_physics_predictions.csv"
OUT_TABLE = ROOT / "table/algorithm/cn0_wgc_multiband_complete"
OUT_FIG = ROOT / "figure/algorithm/cn0_wgc_multiband_complete"

MISSION_START = pd.Timestamp("2025-01-15T07:32:00Z")
MISSION_STOP = pd.Timestamp("2025-03-16T23:15:00Z")
GPS_UTC_OFFSET_S = 18.0
MOON_RADIUS_KM = 1737.4
CADENCE = "5min"
PAIR_TOLERANCE = pd.Timedelta("2min30s")
SIGNALS = ["GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"]
SIGNAL_SYSTEM = {"GPS_L1": "G", "GPS_L5": "G", "GAL_E1": "E", "GAL_E5a": "E"}
SIGNAL_ID = {"GPS_L1": 0, "GPS_L5": 1, "GAL_E1": 2, "GAL_E5a": 3}
FREQUENCY_MHZ = {"GPS_L1": 1575.42, "GPS_L5": 1176.45, "GAL_E1": 1575.42, "GAL_E5a": 1176.45}
FREQUENCY_BAND = {"GPS_L1": "L1_E1", "GAL_E1": "L1_E1", "GPS_L5": "L5_E5a", "GAL_E5a": "L5_E5a"}
COLORS = {"GPS_L1": "#356D9A", "GPS_L5": "#79A6C7", "GAL_E1": "#2F8F83", "GAL_E5a": "#87BFB6"}
DISPLAY_SIGNAL = {"GPS_L1": "GPS L1", "GPS_L5": "GPS L5", "GAL_E1": "Galileo E1", "GAL_E5a": "Galileo E5a"}


def parse_wgc_time(value: object) -> pd.Timestamp:
    return pd.to_datetime(str(value).replace(" UTC", "Z"), utc=True, errors="coerce")


def load_wgc() -> pd.DataFrame:
    wgc = pd.read_csv(
        WGC,
        usecols=["utc", "rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"],
        low_memory=False,
    )
    wgc["utc"] = pd.to_datetime(wgc["utc"], utc=True, errors="coerce")
    wgc = wgc.dropna(subset=["utc", "rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"])
    # nearest(limit=2) preserves the genuine WGC_new gaps rather than bridging
    # them with a synthetic trajectory.
    wgc = wgc.set_index("utc").sort_index().resample(CADENCE).nearest(limit=2)
    return wgc.dropna(subset=["rx_x_eci_km"]).reset_index()[["utc", "rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]]


def parse_mgex_sp3(satellites: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(MGEX.glob("*ORB.SP3.gz")):
        epoch = None
        with gzip.open(path, "rt", errors="replace") as stream:
            for line in stream:
                if line.startswith("*"):
                    p = line[1:].split()
                    gps_time = datetime(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), tzinfo=timezone.utc) + timedelta(seconds=float(p[5]))
                    epoch = pd.Timestamp(gps_time - timedelta(seconds=GPS_UTC_OFFSET_S))
                elif epoch is not None and line.startswith("P") and line[1:4] in satellites:
                    xyz = [float(line[4:18]), float(line[18:32]), float(line[32:46])]
                    if max(abs(v) for v in xyz) < 999999.0:
                        rows.append({"utc": epoch, "sat_id": line[1:4], "sat_x_ecef_km": xyz[0], "sat_y_ecef_km": xyz[1], "sat_z_ecef_km": xyz[2]})
    out = pd.DataFrame(rows)
    return out[(out["utc"] >= MISSION_START) & (out["utc"] <= MISSION_STOP)].reset_index(drop=True)


def parse_gps_iiia_band(band: str) -> dict[int, dict]:
    if band == "L1":
        return parse_iiia_2d(GPS_DIR)
    patterns: dict[int, dict] = {}
    for svn in range(74, 79):
        candidates = sorted(GPS_DIR.glob(f"GPS_III_SVN{svn}*Directivity.zip"))
        if not candidates:
            continue
        with zipfile.ZipFile(candidates[0]) as archive:
            pdf_name = next(name for name in archive.namelist() if re.search(fr"_{band}_.*Directivity\.pdf$", name))
            reader = PdfReader(io.BytesIO(archive.read(pdf_name)))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        theta_rows, gains = [], []
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) == 37 and re.fullmatch(r"-?\d+", tokens[0]) and 0 <= int(tokens[0]) <= 90:
                theta_rows.append(float(tokens[0]))
                gains.append([float(x) for x in tokens[1:]])
        if len(theta_rows) == 46:
            patterns[svn] = {"theta": np.asarray(theta_rows), "phi": np.arange(0.0, 360.0, 10.0), "gain": np.asarray(gains), "source": f"{candidates[0].as_posix()}#{pdf_name}", "block_pattern_family": "GPS_IIIA"}
    return patterns


def available_satellites(obs: pd.DataFrame) -> tuple[dict[str, list[int]], pd.DataFrame, dict[str, dict[int, dict]]]:
    mapping = pd.read_csv(MAPPING)
    mapping["svid"] = mapping["prn"].str.extract(r"[GE](\d+)").astype(int)
    mapping["svn"] = pd.to_numeric(mapping["sv_identifier"].str.extract(r"G(\d+)")[0], errors="coerce")
    gps_patterns = {"GPS_L1": parse_iirm_2d(GPS_DIR / "AppBAntennaPanelPatterns.pptx"), "GPS_L5": {}}
    gps_patterns["GPS_L1"].update(parse_gps_iiia_band("L1"))
    gps_patterns["GPS_L5"].update(parse_gps_iiia_band("L5"))
    iif_patterns = parse_iif_patterns(GPS_DIR)
    for signal in ("GPS_L1", "GPS_L5"):
        gps_patterns[signal].update(iif_patterns[signal])

    # Public satellite-specific GPS III patterns currently stop at SVN78.
    # Keep later Block IIIA spacecraft calculable with a clearly labelled
    # same-block median pattern, matching the validated baseline pipeline.
    for signal in ("GPS_L1", "GPS_L5"):
        exact_iii = {
            svn: pattern
            for svn, pattern in gps_patterns[signal].items()
            if 74 <= int(svn) <= 78
        }
        if not exact_iii:
            continue
        theta = next(iter(exact_iii.values()))["theta"]
        phi = next(iter(exact_iii.values()))["phi"]
        median_gain = np.nanmedian(
            np.stack([pattern["gain"] for pattern in exact_iii.values()]), axis=0
        )
        block_iii_svns = (
            mapping.loc[mapping["antenna_type"].eq("BLOCK IIIA"), "svn"]
            .dropna()
            .astype(int)
            .unique()
        )
        for svn in block_iii_svns:
            if svn not in gps_patterns[signal]:
                gps_patterns[signal][svn] = {
                    "theta": theta,
                    "phi": phi,
                    "gain": median_gain,
                    "source": f"GPS III SVN74-78 median {signal} 2D proxy",
                    "block_pattern_family": "GPS_IIIA_same_block_median_proxy",
                    "pattern_coverage": "same_block_median_proxy_full_2D",
                }
    selected: dict[str, list[int]] = {}
    for signal in SIGNALS:
        counts = obs[obs["signal_name"].eq(signal)].groupby("svid").size().sort_values(ascending=False)
        if signal.startswith("GPS"):
            gps_map = mapping[mapping["system"].eq("G")].dropna(subset=["svn"])
            eligible_rows = gps_map[
                gps_map["svn"].astype(int).isin(gps_patterns[signal])
                & gps_map["antenna_type"].isin(TX_POWER_DBW)
            ]
            eligible = set(eligible_rows["svid"].astype(int))
            counts = counts[counts.index.astype(int).isin(eligible)]
        selected[signal] = [int(x) for x in counts.head(3).index]
    return selected, mapping, gps_patterns


def interpolate_sp3(wgc: pd.DataFrame, sp3: pd.DataFrame, sat_ids: list[str]) -> pd.DataFrame:
    target = wgc["utc"].map(lambda t: t.timestamp()).to_numpy(float)
    rows = []
    for sat_id in sat_ids:
        g = sp3[sp3["sat_id"].eq(sat_id)].sort_values("utc")
        if g.empty:
            continue
        source = g["utc"].map(lambda t: t.timestamp()).to_numpy(float)
        inside = (target >= source.min()) & (target <= source.max())
        out = wgc.loc[inside].copy()
        out["sat_id"] = sat_id
        out["system"] = sat_id[0]
        out["svid"] = int(sat_id[1:])
        for col in ["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]:
            out[col] = np.interp(target[inside], source, g[col].to_numpy(float))
        rows.append(out)
    out = pd.concat(rows, ignore_index=True)
    out[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]] = ecef_to_eci(out)
    return out


def spice_geometry(times: pd.Series) -> dict[pd.Timestamp, dict]:
    cache: dict[pd.Timestamp, dict] = {}
    load_spice(SPICE_DIR)
    try:
        for value in sorted(times.unique()):
            ts = pd.Timestamp(value)
            et = sp.utc2et(ts.strftime("%Y-%m-%dT%H:%M:%S.%f"))
            sun = np.asarray(sp.spkpos("SUN", et, "J2000", "NONE", "EARTH")[0], float)
            moon = np.asarray(sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")[0], float)
            try:
                rotation = np.asarray(sp.pxform("BGM1_LUGRE", "J2000", et), float).T
                status = "spice_attitude_available"
            except Exception:
                rotation, status = None, "nominal_earth_pointing_fallback"
            cache[ts] = {"sun": sun, "moon": moon, "rotation": rotation, "status": status}
    finally:
        sp.kclear()
    return cache


def rx_gain(theta: np.ndarray, signal: str) -> np.ndarray:
    pattern = pd.read_csv(RX_PATTERN)
    band = "L1/E1" if signal in {"GPS_L1", "GAL_E1"} else "L5/E5a"
    p = pattern[pattern["band"].eq(band)].sort_values("signed_elevation_deg")
    axis, gain = p["signed_elevation_deg"].to_numpy(float), p["outer_envelope_gain_dB_axis"].to_numpy(float)
    return np.maximum(np.interp(theta, axis, gain), np.interp(-theta, axis, gain))


def grap_lookup(signal: str, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    grid = pd.read_csv(GRAP, usecols=["signal_name", "azimuth_deg", "coelevation_deg", "eirp_dbw"])
    grid = grid[grid["signal_name"].eq(signal)]
    pivot = grid.pivot_table(index="coelevation_deg", columns="azimuth_deg", values="eirp_dbw", aggfunc="mean").sort_index().sort_index(axis=1)
    pattern = {"theta": pivot.index.to_numpy(float), "phi": pivot.columns.to_numpy(float)[:-1], "gain": pivot.to_numpy(float)[:, :-1]}
    return bilinear_periodic(pattern, theta, phi)


def expand_signals(geometry: pd.DataFrame, selected: dict[str, list[int]]) -> pd.DataFrame:
    parts = []
    for signal in SIGNALS:
        system = SIGNAL_SYSTEM[signal]
        d = geometry[geometry["system"].eq(system) & geometry["svid"].isin(selected[signal])].copy()
        d["signal_name"] = signal
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def build_physics(df: pd.DataFrame, cache: dict[pd.Timestamp, dict], mapping: pd.DataFrame, patterns: dict[str, dict[int, dict]]) -> pd.DataFrame:
    gps_map = mapping[mapping["system"].eq("G")][
        ["system", "svid", "svn", "antenna_type"]
    ].drop_duplicates(["system", "svid"])
    df = df.merge(gps_map, on=["system", "svid"], how="left")
    rx = df[["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]].to_numpy(float)
    sat = df[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]].to_numpy(float)
    sun = np.vstack([cache[pd.Timestamp(t)]["sun"] for t in df["utc"]])
    moon = np.vstack([cache[pd.Timestamp(t)]["moon"] for t in df["utc"]])
    los_rx, ranges = normalize(sat - rx)
    los_tx = -los_rx
    rx_theta = np.empty(len(df))
    rx_phi = np.zeros(len(df))
    status = []
    for i, t in enumerate(df["utc"]):
        item = cache[pd.Timestamp(t)]
        status.append(item["status"])
        if item["rotation"] is not None:
            local = item["rotation"] @ los_rx[i]
            rx_theta[i] = np.degrees(np.arccos(np.clip(local[2], -1, 1)))
            rx_phi[i] = np.mod(np.degrees(np.arctan2(local[1], local[0])), 360.0)
        else:
            earth_dir = -rx[i] / np.linalg.norm(rx[i])
            rx_theta[i] = np.degrees(np.arccos(np.clip(np.dot(los_rx[i], earth_dir), -1, 1)))
    df["rx_attitude_status"] = status
    df["rx_offboresight_spice_deg"] = rx_theta
    df["rx_azimuth_spice_deg"] = rx_phi
    df["rx_gain_envelope_dbic"] = np.nan
    for signal in SIGNALS:
        mask = df["signal_name"].eq(signal)
        df.loc[mask, "rx_gain_envelope_dbic"] = rx_gain(df.loc[mask, "rx_offboresight_spice_deg"].to_numpy(float), signal)

    z_body, _ = normalize(-sat)
    sat_to_sun, _ = normalize(sun - sat)
    projection = sat_to_sun - np.sum(sat_to_sun * z_body, axis=1)[:, None] * z_body
    x_body, projection_norm = normalize(projection)
    y_body, _ = normalize(np.cross(z_body, x_body))
    df["tx_theta_body_deg"] = np.degrees(np.arccos(np.clip(np.sum(los_tx * z_body, axis=1), -1, 1)))
    df["tx_phi_body_deg"] = np.mod(np.degrees(np.arctan2(np.sum(los_tx * y_body, axis=1), np.sum(los_tx * x_body, axis=1))), 360.0)
    df["tx_offboresight_deg"] = df["tx_theta_body_deg"]
    df["tx_yaw_singularity_angle_deg"] = np.degrees(np.arcsin(np.clip(projection_norm, 0, 1)))
    df["tx_yaw_quality"] = np.where(df["tx_yaw_singularity_angle_deg"] < 5, "noon_midnight_turn_risk", "nominal_yaw_steering")

    sat_ids = df["system"].astype(str) + df["svid"].astype(int).astype(str).str.zfill(2)
    orbex_theta, orbex_phi, orbex_gap_s, orbex_status = orbex_body_angles(
        df["utc"], sat_ids, los_tx, orbex_dir=DEFAULT_ORBEX_DIR
    )
    orbex_valid = np.isfinite(orbex_theta) & np.isfinite(orbex_phi)
    df.loc[orbex_valid, "tx_theta_body_deg"] = orbex_theta[orbex_valid]
    df.loc[orbex_valid, "tx_phi_body_deg"] = orbex_phi[orbex_valid]
    df["tx_orbex_nearest_gap_s"] = orbex_gap_s
    df["tx_attitude_source"] = np.where(
        orbex_valid, orbex_status, "nominal_yaw_steering_fallback"
    )
    df.loc[orbex_valid, "tx_yaw_quality"] = "CODE_MGEX_final_ORBEX_30s"

    df["tx_gain_2d_db"] = np.nan
    df["tx_eirp_2d_dbw"] = np.nan
    df["tx_pattern_source"] = ""
    df["tx_pattern_family"] = "unavailable"
    df["tx_pattern_coverage"] = "unavailable"
    for signal in ["GPS_L1", "GPS_L5"]:
        for svn, pattern in patterns[signal].items():
            mask = df["signal_name"].eq(signal) & df["svn"].eq(svn)
            df.loc[mask, "tx_gain_2d_db"] = bilinear_periodic(pattern, df.loc[mask, "tx_theta_body_deg"].to_numpy(float), df.loc[mask, "tx_phi_body_deg"].to_numpy(float))
            df.loc[mask, "tx_pattern_source"] = pattern["source"]
            df.loc[mask, "tx_pattern_family"] = pattern.get("block_pattern_family", "GPS_pattern_family_unspecified")
            df.loc[mask, "tx_pattern_coverage"] = pattern.get("pattern_coverage", "satellite_specific_full_2D")
    df["tx_power_dbw"] = df["antenna_type"].map(TX_POWER_DBW)
    gps = df["system"].eq("G")
    df.loc[gps, "tx_eirp_2d_dbw"] = df.loc[gps, "tx_power_dbw"] + df.loc[gps, "tx_gain_2d_db"]
    for signal in ["GAL_E1", "GAL_E5a"]:
        mask = df["signal_name"].eq(signal)
        df.loc[mask, "tx_eirp_2d_dbw"] = grap_lookup(signal, df.loc[mask, "tx_theta_body_deg"].to_numpy(float), df.loc[mask, "tx_phi_body_deg"].to_numpy(float))
        df.loc[mask, "tx_pattern_source"] = "Galileo_GRAP_v1_0_2D_reference_EIRP_ORBEX_body_mapping"
        df.loc[mask, "tx_pattern_family"] = "Galileo_GRAP_v1_0_2D_EIRP"
        df.loc[mask, "tx_pattern_coverage"] = "official_reference_2D_0_90deg"

    df["frequency_mhz"] = df["signal_name"].map(FREQUENCY_MHZ)
    df["frequency_band"] = df["signal_name"].map(FREQUENCY_BAND)
    df["signal_id"] = df["signal_name"].map(SIGNAL_ID)
    df["geometric_range_km"] = ranges
    df["fspl_db"] = 32.44 + 20 * np.log10(ranges) + 20 * np.log10(df["frequency_mhz"])
    df["earth_blocked"] = sphere_blocked(rx, sat, np.zeros_like(rx), EARTH_RADIUS_KM).astype(int)
    df["moon_blocked"] = sphere_blocked(rx, sat, moon, MOON_RADIUS_KM).astype(int)
    df["earth_observer_altitude_km"] = np.linalg.norm(rx, axis=1) - EARTH_RADIUS_KM
    df["moon_observer_altitude_km"] = np.linalg.norm(rx - moon, axis=1) - MOON_RADIUS_KM
    df["earth_grazing_altitude_km"] = sphere_grazing_altitude(rx, sat, np.zeros_like(rx), EARTH_RADIUS_KM)
    df["moon_grazing_altitude_km"] = sphere_grazing_altitude(rx, sat, moon, MOON_RADIUS_KM)
    df["earth_limb_margin_deg"] = limb_margin_deg(rx, sat, np.zeros_like(rx), EARTH_RADIUS_KM)
    df["moon_limb_margin_deg"] = limb_margin_deg(rx, sat, moon, MOON_RADIUS_KM)
    df["earth_limb_proximity_proxy"] = np.exp(-np.minimum(np.abs(df["earth_limb_margin_deg"]), 90) / 2)
    df["moon_limb_proximity_proxy"] = np.exp(-np.minimum(np.abs(df["moon_limb_margin_deg"]), 90) / 2)
    df["system_noise_temperature_k"] = df["signal_name"].map(RX_TSYS_K)
    df["rx_peak_gain_dbic"] = df["signal_name"].map({"GPS_L1": 15.35, "GAL_E1": 15.35, "GPS_L5": 14.56, "GAL_E5a": 14.56})
    df["implementation_loss_assumed_db"] = 0.0
    df["mission_phase"] = df["utc"].map(mission_phase)
    df["op"] = "WGC_new_MGEX_continuous"
    df["samples"] = 1
    df["range_rate_rx_only_km_s"] = np.nan
    df["nav_age_seconds"] = np.nan
    df["rx_utc"] = df["utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    gps_epoch = pd.Timestamp("1980-01-06T00:00:00Z")
    df["rx_gps_seconds"] = (df["utc"] - gps_epoch).dt.total_seconds() + GPS_UTC_OFFSET_S
    df["time_bin_gps_seconds"] = np.floor(df["rx_gps_seconds"] / 60) * 60

    df["orbit_geometry_status"] = "WGC_new_reference_receiver_plus_CODE_MGEX_final_5min_GPS_Galileo"
    df["fspl_status"] = "computed_signal_frequency_WGC_MGEX_geometry"
    df["occultation_status"] = "computed_spherical_earth_moon"
    df["limb_margin_status"] = "computed_spherical_limb_margin"
    df["tx_gain_status"] = np.where(df["system"].eq("G"), "GPS_band_specific_2D_directivity", "Galileo_band_specific_GRAP_2D_reference_EIRP")
    df["rx_gain_status"] = "LuGRE_Fig3_band_envelope_with_SPICE_attitude"
    df["atmos_iono_status"] = "engineering_shell_proxy"
    df["receiver_gain_model_status"] = "LuGRE_Fig3_band_envelope_not_full_azimuth_pattern"
    df["galileo_grap_lookup_status"] = np.where(df["system"].eq("E"), "full_2D_GRAP_lookup", "not_applicable")
    df["galileo_grap_coordinate_mapping_status"] = np.where(
        df["system"].eq("E") & orbex_valid,
        "CODE_MGEX_ORBEX_ECEF_to_body_mapping",
        np.where(df["system"].eq("E"), "nominal_yaw_body_mapping_quality_flagged", "not_applicable"),
    )
    df["gps_l1_direct_merge_status"] = np.where(df["signal_name"].eq("GPS_L1"), "direct_2D_available", "not_applicable")
    df["gps_tx_physics_status"] = np.where(df["system"].eq("G"), "band_specific_2D_direct_budget", "not_gps")
    df["tx_phi_alignment_quality"] = np.where(
        orbex_valid,
        "CODE_MGEX_ORBEX_body_frame",
        np.where(df["system"].eq("G"), "nominal_yaw_pattern_phi_quality_flagged", "Galileo_GRAP_nominal_yaw_mapping"),
    )
    df["tx_gnss_ssv_signal_band"] = df["signal_name"].str.replace("GPS_", "", regex=False).str.replace("GAL_", "", regex=False)
    df["tx_ssv_main_lobe_boundary_deg"] = df["signal_name"].map(GNSS_SSV_MAIN_LOBE_BOUNDARY_DEG)
    df["tx_ssv_double_sided_full_width_deg"] = 2 * df["tx_ssv_main_lobe_boundary_deg"]
    df["tx_ssv_main_lobe_classification"] = np.where(df["tx_theta_body_deg"] <= df["tx_ssv_main_lobe_boundary_deg"], "ssv_main_lobe_service", "outside_ssv_main_lobe")
    df["tx_ssv_main_lobe_boundary_basis"] = GNSS_SSV_BOUNDARY_BASIS
    df["tx_ssv_classification_semantics"] = "service_angle_classification"
    df["tx_central_half_power_classification"] = "not_evaluated"
    df["cn0_2d_model_status"] = "WGC_MGEX_band_specific_transmit_and_receive_envelope"

    add_ionosphere_absorption_proxy(df)
    add_tropospheric_gas_absorption_proxy(df)
    add_edge_risk_features(df)
    df["cn0_constellation_physics_proxy_dbhz"] = df["tx_eirp_2d_dbw"] + df["rx_gain_envelope_dbic"] - df["fspl_db"] + K_DB - 10 * np.log10(df["system_noise_temperature_k"]) - df["l_ion_abs_budget_db"] - df["l_gas_abs_budget_db"]
    visible = df["earth_blocked"].eq(0) & df["moon_blocked"].eq(0)
    df.loc[~visible, "cn0_constellation_physics_proxy_dbhz"] = np.nan
    df["cn0_constellation_direct_available_dbhz"] = df["cn0_constellation_physics_proxy_dbhz"]
    df["cn0_reference_trajectory_2d_dbhz"] = df["cn0_constellation_physics_proxy_dbhz"] + df["l_ion_abs_budget_db"] + df["l_gas_abs_budget_db"]
    df["cn0_reference_trajectory_2d_atm_proxy_dbhz"] = df["cn0_constellation_physics_proxy_dbhz"]
    df["constellation_direct_status"] = np.where(visible, "WGC_MGEX_band_specific_direct_budget", "occulted")
    df["constellation_physics_proxy_status"] = df["constellation_direct_status"]
    return df


def apply_general_ai(df: pd.DataFrame) -> pd.DataFrame:
    train = train_mod.load_data()
    numeric, categorical = train_mod.feature_columns(train)
    tree_pre, scaled_pre = train_mod.preprocessors(numeric, categorical)
    estimator = next(spec[2] for spec in train_mod.model_specs() if spec[0] == "hist_gradient_boosting_residual")
    pipe = train_mod.make_pipeline("hist_gradient_boosting_residual", estimator, tree_pre, scaled_pre)
    train_full = train[train["split"].eq("train")]
    features = numeric + categorical
    pipe.fit(train_full[features], train_full["residual_target_db"].to_numpy(float))
    out = df.copy()
    for col in features:
        if col not in out:
            out[col] = np.nan
    valid = out["cn0_constellation_physics_proxy_dbhz"].notna()
    out["residual_pred_db"] = np.nan
    out.loc[valid, "residual_pred_db"] = pipe.predict(out.loc[valid, features])
    out["cn0_model_dbhz"] = out["cn0_constellation_physics_proxy_dbhz"] + out["residual_pred_db"]
    out["model_name"] = "global_all_band_hist_gradient_boosting_residual"
    return out


def pair_reference(model: pd.DataFrame, obs: pd.DataFrame, selected: dict[str, list[int]]) -> pd.DataFrame:
    refs = obs[obs["signal_name"].isin(SIGNALS)].copy()
    refs["utc_ref"] = pd.to_datetime(refs["rx_utc"], utc=True, errors="coerce").dt.floor("min")
    refs = refs.groupby(["signal_name", "svid", "utc_ref"], as_index=False).agg(cn0_reference_dbhz=("cn0_dbhz_mean", "median"), reference_rows=("cn0_dbhz_mean", "size"))
    parts = []
    for signal, svids in selected.items():
        for svid in svids:
            r = refs[refs["signal_name"].eq(signal) & refs["svid"].eq(svid)].sort_values("utc_ref")
            m = model[model["signal_name"].eq(signal) & model["svid"].eq(svid) & model["cn0_model_dbhz"].notna()].sort_values("utc")
            if r.empty or m.empty:
                continue
            paired = pd.merge_asof(r, m[["utc", "cn0_model_dbhz", "cn0_constellation_physics_proxy_dbhz", "residual_pred_db", "tx_theta_body_deg", "geometric_range_km"]], left_on="utc_ref", right_on="utc", direction="nearest", tolerance=PAIR_TOLERANCE)
            paired = paired.dropna(subset=["utc", "cn0_model_dbhz"])
            paired = paired.groupby("utc", as_index=False).agg(signal_name=("signal_name", "first"), svid=("svid", "first"), cn0_reference_dbhz=("cn0_reference_dbhz", "median"), cn0_model_dbhz=("cn0_model_dbhz", "first"), cn0_physics_dbhz=("cn0_constellation_physics_proxy_dbhz", "first"), residual_pred_db=("residual_pred_db", "first"), reference_rows=("reference_rows", "sum"), tx_theta_body_deg=("tx_theta_body_deg", "first"), geometric_range_km=("geometric_range_km", "first"))
            paired["signed_error_dbhz"] = paired["cn0_model_dbhz"] - paired["cn0_reference_dbhz"]
            paired["abs_error_dbhz"] = paired["signed_error_dbhz"].abs()
            parts.append(paired)
    return pd.concat(parts, ignore_index=True)


def configure_style() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.75, "legend.frameon": False})


def plot_paired(paired: pd.DataFrame, selected: dict[str, list[int]]) -> tuple[Path, Path]:
    configure_style()
    y_min = min(paired["cn0_reference_dbhz"].min(), paired["cn0_model_dbhz"].min()) - 1
    y_max = max(paired["cn0_reference_dbhz"].max(), paired["cn0_model_dbhz"].max()) + 1
    err_lim = max(abs(paired["signed_error_dbhz"].min()), abs(paired["signed_error_dbhz"].max())) * 1.05

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.8), sharey=True)
    for row, signal in enumerate(SIGNALS):
        for col, svid in enumerate(selected[signal]):
            ax = axes[row, col]
            d = paired[paired["signal_name"].eq(signal) & paired["svid"].eq(svid)].sort_values("utc")
            ax.vlines(d["utc"], d["cn0_reference_dbhz"], d["cn0_model_dbhz"], color="#939BA2", lw=0.7, alpha=0.78, zorder=1)
            ax.scatter(d["utc"], d["cn0_reference_dbhz"], s=18, facecolors="white", edgecolors="#20262B", linewidths=0.75, marker="o", label="LuGRE reference", zorder=3)
            ax.scatter(d["utc"], d["cn0_model_dbhz"], s=18, color=COLORS[signal], linewidths=0.45, marker="x", label="Physics + AI", zorder=4)
            mae, bias = d["abs_error_dbhz"].mean(), d["signed_error_dbhz"].mean()
            ax.set_title(f"{DISPLAY_SIGNAL[signal]}  SVID {svid} | n={len(d)} | MAE={mae:.2f}, bias={bias:+.2f} dB", fontsize=6.4, pad=3)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, color="#E7EAED", lw=0.45)
            loc = mdates.AutoDateLocator(minticks=2, maxticks=4)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=False))
    for ax in axes[:, 0]:
        ax.set_ylabel("C/N0 (dB-Hz)")
    for ax in axes[-1, :]:
        ax.set_xlabel("WGC epoch (UTC)")
    handles = [plt.Line2D([], [], marker="o", mfc="white", mec="#20262B", ls="", ms=4.5, label="LuGRE reference"), plt.Line2D([], [], marker="x", color="#356D9A", ls="", ms=4.5, label="Physics + AI at WGC epoch"), plt.Line2D([], [], color="#B7BDC3", lw=0.7, label="Same-epoch difference")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.998))
    fig.suptitle("Paired C/N0 validation on WGC epochs (2025)", y=1.02, fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    base = OUT_FIG / "cn0_wgc_multiband_3sat_paired_comparison"
    for ext, kw in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {})]:
        fig.savefig(base.with_suffix(f".{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.6), sharey=True)
    for row, signal in enumerate(SIGNALS):
        for col, svid in enumerate(selected[signal]):
            ax = axes[row, col]
            d = paired[paired["signal_name"].eq(signal) & paired["svid"].eq(svid)].sort_values("utc")
            ax.axhline(0, color="#20262B", lw=0.75)
            ax.vlines(d["utc"], 0, d["signed_error_dbhz"], color=COLORS[signal], lw=0.55, alpha=0.55)
            ax.scatter(d["utc"], d["signed_error_dbhz"], s=13, color=COLORS[signal], marker="x", linewidths=0.55)
            ax.set_title(f"{DISPLAY_SIGNAL[signal]}  SVID {svid} | model - reference", fontsize=6.4, pad=3)
            ax.set_ylim(-err_lim, err_lim)
            ax.grid(True, color="#E7EAED", lw=0.45)
            loc = mdates.AutoDateLocator(minticks=2, maxticks=4)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=False))
    for ax in axes[:, 0]:
        ax.set_ylabel("Error (dB)")
    for ax in axes[-1, :]:
        ax.set_xlabel("WGC epoch (UTC)")
    fig.suptitle("Paired WGC-epoch C/N0 residuals (2025)", y=1.01, fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    err_base = OUT_FIG / "cn0_wgc_multiband_3sat_same_epoch_residuals"
    for ext, kw in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {})]:
        fig.savefig(err_base.with_suffix(f".{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    return base.with_suffix(".png"), err_base.with_suffix(".png")


def main() -> None:
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(OBS, low_memory=False)
    obs["svid"] = pd.to_numeric(obs["svid"], errors="coerce").astype("Int64")
    selected, mapping, patterns = available_satellites(obs)
    sat_ids = sorted({f"{SIGNAL_SYSTEM[sig]}{svid:02d}" for sig, svids in selected.items() for svid in svids})
    wgc = load_wgc()
    sp3 = parse_mgex_sp3(set(sat_ids))
    geometry = interpolate_sp3(wgc, sp3, sat_ids)
    cache = spice_geometry(geometry["utc"])
    links = expand_signals(geometry, selected)
    physics = build_physics(links, cache, mapping, patterns)
    model = apply_general_ai(physics)
    paired = pair_reference(model, obs, selected)
    metrics = paired.groupby(["signal_name", "svid"], as_index=False).agg(n=("signed_error_dbhz", "size"), mae_db=("abs_error_dbhz", "mean"), rmse_db=("signed_error_dbhz", lambda x: float(np.sqrt(np.mean(np.square(x))))), bias_db=("signed_error_dbhz", "mean"), median_error_db=("signed_error_dbhz", "median"))
    model.to_csv(OUT_TABLE / "cn0_wgc_multiband_complete_predictions.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT_TABLE / "cn0_wgc_multiband_paired_reference.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_TABLE / "cn0_wgc_multiband_paired_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"signal_name": k, "selected_svids": ",".join(map(str, v))} for k, v in selected.items()]).to_csv(OUT_TABLE / "cn0_wgc_multiband_selection.csv", index=False, encoding="utf-8-sig")
    figs = plot_paired(paired, selected)
    print(f"selected={selected}")
    print(f"wgc_epochs={len(wgc)} geometry_rows={len(geometry)} link_rows={len(model)} paired_rows={len(paired)}")
    print(metrics.to_string(index=False))
    print("\n".join(map(str, figs)))


if __name__ == "__main__":
    main()
