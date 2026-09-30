#!/usr/bin/env python3
"""Compute C/N0 on WGC reference-trajectory times and overlay LuGRE observations.

This figure is intentionally different from the observation-row evaluation
figures: the time base is the WGC reference trajectory.  GNSS satellite states
come from IGS final SP3 orbits and are interpolated onto the WGC cadence used
here.  LuGRE observed C/N0 is only overlaid where it exists.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import spiceypy as sp

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
from simulate_continuous_cn0_reference import (
    GPS_UTC_OFFSET_S,
    MOON_RADIUS_KM,
    SP3_DIR,
    ecef_to_eci,
    parse_sp3,
    rx_envelope,
    sphere_blocked,
)
import train_cn0_constellation_residual_models as train_mod


WGC_XLSX = ROOT / "data" / "trajectory" / "wgc_mission_minute_lugre_earth_omit_errors" / (
    "WGC_StateVector_LuGRE_target_Earth_observer_1min_no_aberration_omit_errors_repaired.xlsx"
)
MAPPING = ROOT / "data" / "external_reference" / "mapping" / "active_prn_svn_block_20250115_20250316.csv"
GPS_DIR = ROOT / "data" / "external_reference" / "gnss_antenna" / "gps"
# Keep SPICE kernels relative to the workspace. CSPICE on Windows can fail on
# non-ASCII absolute paths.
SPICE_DIR = Path("data") / "external_reference" / "spice"
OBS_PATH = ROOT / "table" / "algorithm" / "cn0_constellation_physics_baseline" / "cn0_constellation_physics_predictions.csv"
OUT_DIR = ROOT / "figure" / "algorithm" / "cn0_constellation_ai_residual"
TABLE_DIR = ROOT / "table" / "algorithm" / "cn0_constellation_ai_residual"

FEATURE_MODE = "category_assisted"
MODEL_NAME = "lightgbm_residual"
WGC_MODEL_CADENCE = "15min"
FREQUENCY_MHZ = 1575.42
SIGNAL_NAME = "GPS_L1"
SYSTEM = "G"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def parse_wgc_time(value: object) -> pd.Timestamp:
    text = str(value).replace(" UTC", "Z")
    return pd.to_datetime(text, utc=True, errors="coerce")


def load_wgc_reference() -> pd.DataFrame:
    wgc = pd.read_excel(WGC_XLSX, sheet_name="Results")
    wgc["utc"] = wgc["UTC calendar date"].map(parse_wgc_time)
    wgc = wgc.dropna(subset=["utc", "X (km)", "Y (km)", "Z (km)"]).copy()
    wgc = wgc.rename(
        columns={
            "X (km)": "rx_x_eci_km",
            "Y (km)": "rx_y_eci_km",
            "Z (km)": "rx_z_eci_km",
            "dX/dt (km/s)": "rx_vx_eci_km_s",
            "dY/dt (km/s)": "rx_vy_eci_km_s",
            "dZ/dt (km/s)": "rx_vz_eci_km_s",
        }
    )
    # Keep the WGC trajectory as the master clock, but use the 15-minute SP3
    # physical support rather than inventing sub-SP3 precision.
    wgc = wgc.set_index("utc").sort_index()
    wgc = wgc.resample(WGC_MODEL_CADENCE).nearest(limit=1).dropna(subset=["rx_x_eci_km"])
    return wgc.reset_index()


def active_supported_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING)
    mapping = mapping[mapping["system"].eq(SYSTEM)].copy()
    mapping["svid"] = mapping["prn"].str.extract(r"G(\d+)").astype(int)
    mapping["svn"] = mapping["sv_identifier"].str.extract(r"G(\d+)").astype(int)
    patterns = parse_iirm_2d(GPS_DIR / "AppBAntennaPanelPatterns.pptx")
    patterns.update(parse_iiia_2d(GPS_DIR))
    supported = mapping[mapping["antenna_type"].isin(TX_POWER_DBW)].copy()
    supported = supported[supported["svn"].isin(patterns)].copy()
    return supported[["svid", "svn", "antenna_type"]].drop_duplicates("svid"), patterns


def interpolate_sp3_to_wgc(wgc: pd.DataFrame, supported: pd.DataFrame) -> pd.DataFrame:
    svids = set(supported["svid"].astype(int))
    sp3 = parse_sp3(sorted(SP3_DIR.glob("*ORB.SP3.gz")), svids)
    if sp3.empty:
        raise RuntimeError("No SP3 rows available for supported GPS satellites.")

    target_ns = wgc["utc"].astype("int64").to_numpy(dtype=float)
    rows = []
    for svid, group in sp3.groupby("svid"):
        group = group.sort_values("utc")
        source_ns = group["utc"].astype("int64").to_numpy(dtype=float)
        inside = (target_ns >= source_ns.min()) & (target_ns <= source_ns.max())
        if not inside.any():
            continue
        out = wgc.loc[inside, ["utc", "rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km", "rx_vx_eci_km_s", "rx_vy_eci_km_s", "rx_vz_eci_km_s"]].copy()
        out["svid"] = int(svid)
        for col in ["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]:
            out[col] = np.interp(target_ns[inside], source_ns, group[col].to_numpy(float))
        rows.append(out)
    merged = pd.concat(rows, ignore_index=True)
    merged = merged.merge(supported, on="svid", how="left")
    sat_eci = ecef_to_eci(merged.rename(columns={"utc": "utc"}))
    merged[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]] = sat_eci
    return merged


def mission_phase(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2025-01-17T00:00:00Z"):
        return "C"
    if timestamp < pd.Timestamp("2025-02-14T01:51:00Z"):
        return "T"
    if timestamp < pd.Timestamp("2025-03-02T08:34:00Z"):
        return "L"
    return "S"


def gps_seconds(utc: pd.Series) -> np.ndarray:
    gps_epoch = pd.Timestamp("1980-01-06T00:00:00Z")
    return (utc - gps_epoch).dt.total_seconds().to_numpy(float) + GPS_UTC_OFFSET_S


def sphere_grazing_altitude(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    start = p0 - center
    direction = p1 - p0
    denom = np.sum(direction * direction, axis=1)
    t = -np.sum(start * direction, axis=1) / denom
    t_clip = np.clip(t, 0.0, 1.0)
    closest = start + t_clip[:, None] * direction
    return np.linalg.norm(closest, axis=1) - radius


def limb_margin_deg(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    to_target, _ = normalize(p1 - p0)
    to_center, distance = normalize(center - p0)
    sep = np.degrees(np.arccos(np.clip(np.sum(to_target * to_center, axis=1), -1.0, 1.0)))
    horizon = np.degrees(np.arcsin(np.clip(radius / distance, 0.0, 1.0)))
    return sep - horizon


def attitude_sun_moon(times: pd.Series) -> dict[pd.Timestamp, dict]:
    cache = {}
    load_spice(SPICE_DIR)
    try:
        for timestamp in sorted(pd.to_datetime(times, utc=True).unique()):
            ts = pd.Timestamp(timestamp)
            et = sp.utc2et(ts.strftime("%Y-%m-%dT%H:%M:%S.%f"))
            sun = np.asarray(sp.spkpos("SUN", et, "J2000", "NONE", "EARTH")[0], dtype=float)
            moon = np.asarray(sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")[0], dtype=float)
            try:
                rotation = np.asarray(sp.pxform("BGM1_LUGRE", "J2000", et), dtype=float).T
                attitude = "spice_attitude_available"
            except Exception:
                rotation = None
                attitude = "nominal_earth_pointing_fallback"
            cache[ts] = {"sun": sun, "moon": moon, "rotation": rotation, "attitude": attitude}
    finally:
        sp.kclear()
    return cache


def build_wgc_link_features() -> pd.DataFrame:
    wgc = load_wgc_reference()
    supported, patterns = active_supported_mapping()
    df = interpolate_sp3_to_wgc(wgc, supported)
    df["utc"] = pd.to_datetime(df["utc"], utc=True)
    cache = attitude_sun_moon(df["utc"])

    rx = df[["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]].to_numpy(float)
    sat = df[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]].to_numpy(float)
    sun = np.vstack([cache[pd.Timestamp(t)]["sun"] for t in df["utc"]])
    moon = np.vstack([cache[pd.Timestamp(t)]["moon"] for t in df["utc"]])
    rotations = [cache[pd.Timestamp(t)]["rotation"] for t in df["utc"]]
    df["rx_attitude_status"] = [cache[pd.Timestamp(t)]["attitude"] for t in df["utc"]]

    los_rx, ranges = normalize(sat - rx)
    los_tx = -los_rx
    los_lugre = np.full_like(los_rx, np.nan)
    for index, rotation in enumerate(rotations):
        if rotation is not None:
            los_lugre[index] = rotation @ los_rx[index]
        else:
            earth_pointing = -rx[index] / np.linalg.norm(rx[index])
            los_lugre[index, 2] = float(np.dot(los_rx[index], earth_pointing))
            los_lugre[index, 0:2] = 0.0

    df["rx_offboresight_spice_deg"] = np.degrees(np.arccos(np.clip(los_lugre[:, 2], -1.0, 1.0)))
    df["rx_azimuth_spice_deg"] = np.mod(np.degrees(np.arctan2(los_lugre[:, 1], los_lugre[:, 0])), 360.0)
    df["rx_gain_envelope_dbic"] = rx_envelope(df["rx_offboresight_spice_deg"].to_numpy(float))

    z_body, _ = normalize(-sat)
    sat_to_sun, _ = normalize(sun - sat)
    x_projection = sat_to_sun - np.sum(sat_to_sun * z_body, axis=1)[:, None] * z_body
    x_body, projection_norm = normalize(x_projection)
    y_body, _ = normalize(np.cross(z_body, x_body))
    tx_x = np.sum(los_tx * x_body, axis=1)
    tx_y = np.sum(los_tx * y_body, axis=1)
    tx_z = np.sum(los_tx * z_body, axis=1)
    df["tx_theta_body_deg"] = np.degrees(np.arccos(np.clip(tx_z, -1.0, 1.0)))
    df["tx_phi_body_deg"] = np.mod(np.degrees(np.arctan2(tx_y, tx_x)), 360.0)
    df["tx_offboresight_deg"] = df["tx_theta_body_deg"]
    df["tx_yaw_singularity_angle_deg"] = np.degrees(np.arcsin(np.clip(projection_norm, 0.0, 1.0)))
    df["tx_yaw_quality"] = np.where(df["tx_yaw_singularity_angle_deg"] < 5.0, "noon_midnight_turn_risk", "nominal_yaw_steering")

    df["tx_gain_2d_db"] = np.nan
    df["tx_pattern_source"] = ""
    for svn, pattern in patterns.items():
        mask = df["svn"].eq(svn)
        df.loc[mask, "tx_gain_2d_db"] = bilinear_periodic(
            pattern,
            df.loc[mask, "tx_theta_body_deg"].to_numpy(float),
            df.loc[mask, "tx_phi_body_deg"].to_numpy(float),
        )
        df.loc[mask, "tx_pattern_source"] = pattern["source"]

    df["rx_utc"] = df["utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    df["rx_gps_seconds"] = gps_seconds(df["utc"])
    df["time_bin_gps_seconds"] = np.floor(df["rx_gps_seconds"] / 60.0) * 60.0
    df["system"] = SYSTEM
    df["signal_name"] = SIGNAL_NAME
    df["signal_id"] = 0
    df["frequency_mhz"] = FREQUENCY_MHZ
    df["frequency_band"] = "L1_E1"
    df["mission_phase"] = df["utc"].map(mission_phase)
    df["op"] = "WGC_continuous"
    df["samples"] = 1
    df["geometric_range_km"] = ranges
    df["range_rate_rx_only_km_s"] = np.nan
    df["fspl_db"] = 32.44 + 20.0 * np.log10(ranges) + 20.0 * np.log10(FREQUENCY_MHZ)

    earth_center = np.zeros_like(rx)
    df["earth_blocked"] = sphere_blocked(rx, sat, earth_center, EARTH_RADIUS_KM).astype(int)
    df["moon_blocked"] = sphere_blocked(rx, sat, moon, MOON_RADIUS_KM).astype(int)
    df["earth_observer_altitude_km"] = np.linalg.norm(rx, axis=1) - EARTH_RADIUS_KM
    df["moon_observer_altitude_km"] = np.linalg.norm(rx - moon, axis=1) - MOON_RADIUS_KM
    df["earth_grazing_altitude_km"] = sphere_grazing_altitude(rx, sat, earth_center, EARTH_RADIUS_KM)
    df["moon_grazing_altitude_km"] = sphere_grazing_altitude(rx, sat, moon, MOON_RADIUS_KM)
    df["earth_limb_margin_deg"] = limb_margin_deg(rx, sat, earth_center, EARTH_RADIUS_KM)
    df["moon_limb_margin_deg"] = limb_margin_deg(rx, sat, moon, MOON_RADIUS_KM)
    df["earth_limb_proximity_proxy"] = np.exp(-np.minimum(np.abs(df["earth_limb_margin_deg"].to_numpy(float)), 90.0) / 2.0)
    df["moon_limb_proximity_proxy"] = np.exp(-np.minimum(np.abs(df["moon_limb_margin_deg"].to_numpy(float)), 90.0) / 2.0)

    df["rx_peak_gain_dbic"] = 15.35
    df["system_noise_temperature_k"] = RX_TSYS_K[SIGNAL_NAME]
    df["implementation_loss_assumed_db"] = 0.0
    df["receiver_gain_model_status"] = "lugre_fig3_envelope_with_spice_attitude_wgc_time"
    df["nav_age_seconds"] = np.nan
    df["orbit_geometry_status"] = "wgc_reference_trajectory_plus_igs_sp3_gps"
    df["fspl_status"] = "computed_from_wgc_reference_geometry"
    df["occultation_status"] = "computed_spherical_earth_moon"
    df["limb_margin_status"] = "computed_spherical_limb_margin"
    df["tx_gain_status"] = "gps_l1_2d_transmit_pattern"
    df["rx_gain_status"] = "lugre_fig3_envelope_spice_attitude"
    df["atmos_iono_status"] = "engineering_proxy"

    add_ionosphere_absorption_proxy(df)
    add_tropospheric_gas_absorption_proxy(df)
    add_edge_risk_features(df)

    df["tx_power_dbw"] = df["antenna_type"].map(TX_POWER_DBW)
    df["tx_phi_alignment_quality"] = np.where(
        df["antenna_type"].eq("BLOCK IIR-M"),
        "A_confirmed_phi0_plusX_positive_toward_plusY",
        "B_spherical_phi_defined_but_pattern_to_body_installation_not_explicit",
    )
    df["tx_gnss_ssv_signal_band"] = "L1"
    df["tx_ssv_main_lobe_boundary_deg"] = GNSS_SSV_MAIN_LOBE_BOUNDARY_DEG[SIGNAL_NAME]
    df["tx_ssv_signed_lower_boundary_deg"] = -GNSS_SSV_MAIN_LOBE_BOUNDARY_DEG[SIGNAL_NAME]
    df["tx_ssv_signed_upper_boundary_deg"] = GNSS_SSV_MAIN_LOBE_BOUNDARY_DEG[SIGNAL_NAME]
    df["tx_ssv_double_sided_full_width_deg"] = 2.0 * GNSS_SSV_MAIN_LOBE_BOUNDARY_DEG[SIGNAL_NAME]
    df["tx_ssv_main_lobe_classification"] = np.where(
        df["tx_theta_body_deg"] <= df["tx_ssv_main_lobe_boundary_deg"],
        "ssv_main_lobe_service",
        "outside_ssv_main_lobe",
    )
    df["tx_ssv_main_lobe_boundary_basis"] = GNSS_SSV_BOUNDARY_BASIS
    df["tx_ssv_classification_semantics"] = "service_angle_classification;outside_does_not_identify_specific_electromagnetic_sidelobe"
    df["tx_central_half_power_classification"] = "not_evaluated_pattern_gain_unavailable"

    df["cn0_reference_trajectory_2d_dbhz"] = (
        df["tx_power_dbw"]
        + df["tx_gain_2d_db"]
        + df["rx_gain_envelope_dbic"]
        - df["fspl_db"]
        + K_DB
        - 10.0 * np.log10(df["system_noise_temperature_k"])
        - df["implementation_loss_assumed_db"]
    )
    df["cn0_reference_trajectory_2d_atm_proxy_dbhz"] = (
        df["cn0_reference_trajectory_2d_dbhz"] - df["l_ion_abs_budget_db"] - df["l_gas_abs_budget_db"]
    )
    df["cn0_reference_trajectory_2d_iono_proxy_dbhz"] = df["cn0_reference_trajectory_2d_atm_proxy_dbhz"]
    df["cn0_constellation_direct_available_dbhz"] = df["cn0_reference_trajectory_2d_atm_proxy_dbhz"].where(df["earth_blocked"].eq(0) & df["moon_blocked"].eq(0))
    df["constellation_direct_status"] = np.where(
        df["cn0_constellation_direct_available_dbhz"].notna(),
        "gps_l1_wgc_time_2d_direct_budget",
        "occulted_or_missing_direct_budget",
    )
    df["cn0_constellation_physics_proxy_dbhz"] = df["cn0_constellation_direct_available_dbhz"]
    df["constellation_physics_proxy_status"] = df["constellation_direct_status"]
    df["cn0_2d_model_status"] = "wgc_time_quality_A_or_B_nominal_yaw_spice_attitude_2d_gtx_rx_envelope"
    return df


def fit_best_residual_model():
    train_df = train_mod.load_data()
    mode = next(m for m in train_mod.FEATURE_MODES if m.name == FEATURE_MODE)
    numeric, categorical = train_mod.feature_columns(train_df, mode)
    features = numeric + categorical
    tree_pre, scaled_pre = train_mod.preprocessors(numeric, categorical)
    spec = next((s for s in train_mod.model_specs() if s[0] == MODEL_NAME), None)
    if spec is None:
        raise RuntimeError(f"Missing model spec: {MODEL_NAME}")
    model_name, _model_class, estimator = spec
    pipe = train_mod.make_pipeline(model_name, estimator, tree_pre, scaled_pre)
    train_core = train_df[train_df["split"].eq("train")].copy()
    pipe.fit(train_core[features], train_core["residual_target_db"].to_numpy(float))
    return pipe, features


def apply_ai_model(df: pd.DataFrame) -> pd.DataFrame:
    pipe, features = fit_best_residual_model()
    out = df.copy()
    for col in features:
        if col not in out.columns:
            out[col] = np.nan
    valid = out["cn0_constellation_physics_proxy_dbhz"].notna()
    out["residual_pred_db"] = np.nan
    out.loc[valid, "residual_pred_db"] = pipe.predict(out.loc[valid, features])
    out["cn0_wgc_physics_ai_dbhz"] = out["cn0_constellation_physics_proxy_dbhz"] + out["residual_pred_db"]
    out["model_feature_mode"] = FEATURE_MODE
    out["model_name"] = MODEL_NAME
    out["model_timebase"] = "WGC_reference_trajectory_time"
    out["gnss_orbit_timebase"] = "IGS_final_SP3_15min_interpolated_to_WGC_cadence"
    return out


def load_reference_observations() -> pd.DataFrame:
    usecols = ["rx_utc", "system", "svid", "signal_name", "mission_phase", "cn0_dbhz_mean"]
    obs = pd.read_csv(OBS_PATH, usecols=usecols)
    obs = obs[(obs["system"].eq(SYSTEM)) & (obs["signal_name"].eq(SIGNAL_NAME))].copy()
    obs["rx_utc"] = pd.to_datetime(obs["rx_utc"], utc=True, errors="coerce")
    obs = obs.dropna(subset=["rx_utc", "cn0_dbhz_mean"])
    obs["svid"] = pd.to_numeric(obs["svid"], errors="coerce").astype("Int64")
    return obs


def representative_svids(model: pd.DataFrame, obs: pd.DataFrame, n: int = 8) -> list[int]:
    counts = (
        model.groupby("svid").agg(model_rows=("cn0_wgc_physics_ai_dbhz", "count")).join(
            obs.groupby("svid").agg(obs_rows=("cn0_dbhz_mean", "size")), how="left"
        )
    ).fillna(0)
    counts["score"] = counts["model_rows"] + 5.0 * counts["obs_rows"]
    return [int(x) for x in counts.sort_values("score", ascending=False).head(n).index]


def style_time_axis(ax: plt.Axes, y_min: float = 5.0, y_max: float = 55.0) -> None:
    ax.grid(True, color="#E5E7EB", linewidth=0.55, alpha=0.88)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylim(y_min, y_max)


def plot_median_timeline(model: pd.DataFrame, obs: pd.DataFrame) -> Path:
    model_med = (
        model[model["cn0_wgc_physics_ai_dbhz"].notna()]
        .groupby("utc", as_index=False)
        .agg(
            cn0_model_median_dbhz=("cn0_wgc_physics_ai_dbhz", "median"),
            cn0_model_q90_dbhz=("cn0_wgc_physics_ai_dbhz", lambda x: float(np.nanpercentile(x, 90))),
            cn0_model_max_dbhz=("cn0_wgc_physics_ai_dbhz", "max"),
            visible_links=("cn0_wgc_physics_ai_dbhz", "size"),
        )
    )
    obs_med = (
        obs.assign(minute_utc=obs["rx_utc"].dt.floor("min"))
        .groupby("minute_utc", as_index=False)
        .agg(cn0_reference_median_dbhz=("cn0_dbhz_mean", "median"), reference_links=("cn0_dbhz_mean", "size"))
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.scatter(
        model_med["utc"],
        model_med["cn0_model_q90_dbhz"],
        marker="X",
        s=24,
        color="#D94841",
        edgecolors="#9F1D17",
        linewidths=0.25,
        alpha=0.72,
        label="Model C/N0 on WGC time (90th percentile)",
        zorder=2,
    )
    ax.scatter(
        obs_med["minute_utc"],
        obs_med["cn0_reference_median_dbhz"],
        s=30,
        facecolors="none",
        edgecolors="#1F2933",
        linewidths=0.85,
        alpha=0.72,
        label="LuGRE reference C/N0",
        zorder=4,
    )
    style_time_axis(ax)
    ax.set_ylabel("GPS L1 C/N0\n(dB-Hz)")
    ax.set_xlabel("UTC time")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, handletextpad=0.5)
    ax.set_title("WGC-reference-time GPS L1 C/N0: best physics-informed AI model with observation overlay", fontsize=9.5, fontweight="bold")
    fig.text(
        0.01,
        0.012,
        "Model markers summarize calculated visible GPS L1 links at WGC reference-trajectory times. Reference circles: LuGRE observations only where available.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    out = OUT_DIR / "cn0_wgc_reference_time_best_model_gps_l1_median_overlay"
    for suffix, kwargs in [(".png", {"dpi": 600}), (".pdf", {}), (".svg", {})]:
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    obs_med.to_csv(TABLE_DIR / "cn0_wgc_reference_overlay_gps_l1_observed_median.csv", index=False, encoding="utf-8-sig")
    model_med.to_csv(TABLE_DIR / "cn0_wgc_reference_time_best_model_gps_l1_epoch_envelope.csv", index=False, encoding="utf-8-sig")
    return out.with_suffix(".png")


def plot_representative_svids(model: pd.DataFrame, obs: pd.DataFrame) -> Path:
    svids = representative_svids(model, obs, 8)
    fig, axes = plt.subplots(4, 2, figsize=(7.3, 8.2), sharey=True)
    axes = axes.ravel()
    for ax, svid in zip(axes, svids):
        m = model[(model["svid"].eq(svid)) & (model["cn0_wgc_physics_ai_dbhz"].notna())].sort_values("utc")
        o = obs[obs["svid"].eq(svid)].sort_values("rx_utc")
        ax.scatter(
            m["utc"],
            m["cn0_wgc_physics_ai_dbhz"],
            marker="X",
            s=12.0,
            color="#D94841",
            edgecolors="#9F1D17",
            alpha=0.30,
            linewidths=0.18,
            zorder=2,
        )
        ax.scatter(
            o["rx_utc"],
            o["cn0_dbhz_mean"],
            s=28,
            facecolors="none",
            edgecolors="#1F2933",
            linewidths=0.82,
            alpha=0.78,
            zorder=4,
        )
        label = f"G{svid:02d}  model {len(m):,} pts"
        if len(o):
            label += f"\nref {len(o):,} pts"
        else:
            label += "\nref none"
        ax.text(
            0.02,
            0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
        )
        style_time_axis(ax, y_min=-5, y_max=55)
    for ax in axes[::2]:
        ax.set_ylabel("GPS L1 C/N0\n(dB-Hz)")
    for ax in axes[-2:]:
        ax.set_xlabel("UTC time")
    handles = [
        plt.Line2D([], [], marker="X", linestyle="", color="#D94841", markeredgecolor="#9F1D17", markersize=5.0, label="Model on WGC time"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#1F2933", alpha=0.80, markersize=5.2, label="LuGRE reference"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2)
    fig.suptitle("Representative GPS L1 satellites: WGC-time model C/N0 and non-overlapping reference markers", y=1.02, fontsize=10.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "cn0_wgc_reference_time_best_model_gps_l1_representative_svids"
    for suffix, kwargs in [(".png", {"dpi": 600}), (".pdf", {}), (".svg", {})]:
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    return out.with_suffix(".png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    features = build_wgc_link_features()
    model = apply_ai_model(features)
    obs = load_reference_observations()

    model.to_csv(TABLE_DIR / "cn0_wgc_reference_time_best_model_gps_l1_predictions.csv", index=False, encoding="utf-8-sig")
    summary = model.groupby(["svid", "svn", "antenna_type"], as_index=False).agg(
        rows=("utc", "size"),
        modeled_rows=("cn0_wgc_physics_ai_dbhz", "count"),
        model_median_dbhz=("cn0_wgc_physics_ai_dbhz", "median"),
        model_min_dbhz=("cn0_wgc_physics_ai_dbhz", "min"),
        model_max_dbhz=("cn0_wgc_physics_ai_dbhz", "max"),
    )
    summary.to_csv(TABLE_DIR / "cn0_wgc_reference_time_best_model_gps_l1_satellite_summary.csv", index=False, encoding="utf-8-sig")
    median_fig = plot_median_timeline(model, obs)
    svid_fig = plot_representative_svids(model, obs)
    print(median_fig)
    print(svid_fig)
    print(f"model rows={len(model)} modeled={model['cn0_wgc_physics_ai_dbhz'].notna().sum()} WGC epochs={model['utc'].nunique()} satellites={model['svid'].nunique()}")
    print(f"reference observation rows={len(obs)}")


if __name__ == "__main__":
    main()
