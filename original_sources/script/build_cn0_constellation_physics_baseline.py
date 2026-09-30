from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FEATURES = ROOT / "table" / "cn0_physics_features_available_wgc_new.csv"
INPUT_OLD_BASELINE = ROOT / "table" / "algorithm" / "cn0_physics_baseline" / "cn0_physics_baseline_predictions.csv"
INPUT_GPS_DIRECT = ROOT / "table" / "algorithm" / "cn0_attitude_2d_gain" / "cn0_reference_trajectory_attitude_2d_predictions.csv"
INPUT_GRAP = ROOT / "table" / "external_reference" / "galileo_grap_eirp_grid.csv"
INPUT_INVENTORY = ROOT / "table" / "external_reference" / "gnss_transmit_physics_inventory.csv"
INPUT_INTERFACE = ROOT / "table" / "external_reference" / "gnss_transmit_physics_interface.md"

OUT_DIR = ROOT / "table" / "algorithm" / "cn0_constellation_physics_baseline"

LIGHT_SPEED_LINK_CONSTANT_DB = 228.6
IMPLEMENTATION_LOSS_ASSUMED_DB = 0.0
EARTH_RADIUS_KM = 6378.137
IONOSPHERE_LOWER_ALT_KM = 60.0
IONOSPHERE_UPPER_ALT_KM = 1000.0
IONOSPHERE_VERTICAL_THICKNESS_KM = IONOSPHERE_UPPER_ALT_KM - IONOSPHERE_LOWER_ALT_KM
IONOSPHERE_L30_VERTICAL_DB = 0.5
IONOSPHERE_MAPPING_FACTOR_CAP = 5.0
TROPOSPHERE_UPPER_ALT_KM = 20.0
GAS_SCALE_HEIGHT_KM = 7.5
GAS_MAPPING_FACTOR_CAP = 10.0
GAS_GAMMA_SURFACE_DB_PER_KM = {
    "GPS_L1": 0.006514909135228911,
    "GAL_E1": 0.006514909135228911,
    "GPS_L5": 0.005879918537161931,
    "GAL_E5a": 0.005879918537161931,
}
EDGE_NEAR_LIMB_ALT_KM = 20.0
NEUTRAL_REFRACTION_UPPER_ALT_KM = 60.0


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rx_peak_gain_dbic(signal: str) -> float:
    if signal in {"GPS_L1", "GAL_E1"}:
        return 15.35
    if signal in {"GPS_L5", "GAL_E5a"}:
        return 14.56
    return np.nan


def system_noise_temperature_k(signal: str) -> float:
    if signal in {"GPS_L1", "GAL_E1"}:
        return 182.0
    if signal in {"GPS_L5", "GAL_E5a"}:
        return 231.0
    return np.nan


def band_label(signal: str) -> str:
    if signal in {"GPS_L1", "GAL_E1"}:
        return "L1_E1"
    if signal in {"GPS_L5", "GAL_E5a"}:
        return "L5_E5a"
    return "unknown"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {
            "n": 0,
            "mae_dbhz": np.nan,
            "rmse_dbhz": np.nan,
            "median_ae_dbhz": np.nan,
            "p95_ae_dbhz": np.nan,
            "bias_dbhz": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
        }
    err = y_pred - y_true
    ae = np.abs(err)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    corr = np.nan if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0 else float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "n": int(len(y_true)),
        "mae_dbhz": float(np.mean(ae)),
        "rmse_dbhz": float(math.sqrt(np.mean(err**2))),
        "median_ae_dbhz": float(np.median(ae)),
        "p95_ae_dbhz": float(np.percentile(ae, 95)),
        "bias_dbhz": float(np.mean(err)),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "pearson_r": corr,
    }


def segment_length_inside_sphere(start: np.ndarray, end: np.ndarray, radius_km: float) -> np.ndarray:
    direction = end - start
    a = np.sum(direction * direction, axis=1)
    b = 2.0 * np.sum(start * direction, axis=1)
    c = np.sum(start * start, axis=1) - radius_km**2
    discriminant = b * b - 4.0 * a * c
    length = np.zeros(len(start), dtype=float)
    valid = (a > 0.0) & (discriminant > 0.0)
    if not valid.any():
        return length
    sqrt_disc = np.sqrt(discriminant[valid])
    denom = 2.0 * a[valid]
    t1 = (-b[valid] - sqrt_disc) / denom
    t2 = (-b[valid] + sqrt_disc) / denom
    lo = np.maximum(np.minimum(t1, t2), 0.0)
    hi = np.minimum(np.maximum(t1, t2), 1.0)
    seg = np.maximum(hi - lo, 0.0)
    length[valid] = seg * np.sqrt(a[valid])
    return length


def interval_inside_sphere(
    start: np.ndarray, end: np.ndarray, radius_km: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = end - start
    a = np.sum(direction * direction, axis=1)
    b = 2.0 * np.sum(start * direction, axis=1)
    c = np.sum(start * start, axis=1) - radius_km**2
    discriminant = b * b - 4.0 * a * c
    lo = np.full(len(start), np.nan)
    hi = np.full(len(start), np.nan)
    valid = (a > 0.0) & (discriminant > 0.0)
    if not valid.any():
        return lo, hi, np.zeros(len(start), dtype=bool)
    sqrt_disc = np.sqrt(discriminant[valid])
    denom = 2.0 * a[valid]
    t1 = (-b[valid] - sqrt_disc) / denom
    t2 = (-b[valid] + sqrt_disc) / denom
    lo_v = np.maximum(np.minimum(t1, t2), 0.0)
    hi_v = np.minimum(np.maximum(t1, t2), 1.0)
    active = hi_v > lo_v
    valid_indices = np.flatnonzero(valid)
    lo[valid_indices[active]] = lo_v[active]
    hi[valid_indices[active]] = hi_v[active]
    out_valid = np.zeros(len(start), dtype=bool)
    out_valid[valid_indices[active]] = True
    return lo, hi, out_valid


def integrate_standard_atmosphere_shell(
    start: np.ndarray,
    end: np.ndarray,
    lower_radius_km: float,
    upper_radius_km: float,
    scale_height_km: float,
    samples: int = 16,
) -> np.ndarray:
    outer_lo, outer_hi, outer_valid = interval_inside_sphere(start, end, upper_radius_km)
    inner_lo, inner_hi, inner_valid = interval_inside_sphere(start, end, lower_radius_km)
    direction = end - start
    path_len = np.linalg.norm(direction, axis=1)
    equivalent_km = np.zeros(len(start), dtype=float)
    sample_offsets = (np.arange(samples, dtype=float) + 0.5) / samples

    for i in np.flatnonzero(outer_valid):
        intervals: list[tuple[float, float]] = []
        lo_o, hi_o = float(outer_lo[i]), float(outer_hi[i])
        if inner_valid[i]:
            lo_i, hi_i = float(inner_lo[i]), float(inner_hi[i])
            if lo_o < min(lo_i, hi_o):
                intervals.append((lo_o, min(lo_i, hi_o)))
            if max(hi_i, lo_o) < hi_o:
                intervals.append((max(hi_i, lo_o), hi_o))
        else:
            intervals.append((lo_o, hi_o))

        for lo_t, hi_t in intervals:
            if hi_t <= lo_t:
                continue
            ts = lo_t + (hi_t - lo_t) * sample_offsets
            points = start[i] + ts[:, None] * direction[i]
            height = np.maximum(np.linalg.norm(points, axis=1) - EARTH_RADIUS_KM, 0.0)
            density_weight = np.exp(-height / scale_height_km)
            equivalent_km[i] += path_len[i] * (hi_t - lo_t) * float(np.mean(density_weight))
    return equivalent_km


def add_ionosphere_absorption_proxy(pred: pd.DataFrame) -> None:
    rx = pred[["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]].to_numpy(float)
    sat = pred[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]].to_numpy(float)
    upper_radius = EARTH_RADIUS_KM + IONOSPHERE_UPPER_ALT_KM
    lower_radius = EARTH_RADIUS_KM + IONOSPHERE_LOWER_ALT_KM
    upper_len = segment_length_inside_sphere(rx, sat, upper_radius)
    lower_len = segment_length_inside_sphere(rx, sat, lower_radius)
    shell_len = np.maximum(upper_len - lower_len, 0.0)
    mion_raw = shell_len / IONOSPHERE_VERTICAL_THICKNESS_KM
    mion = np.minimum(mion_raw, IONOSPHERE_MAPPING_FACTOR_CAP)
    frequency_mhz = pred["frequency_mhz"].to_numpy(float)
    loss_db = IONOSPHERE_L30_VERTICAL_DB * mion * (30.0 / frequency_mhz) ** 2
    loss_db = np.where(np.isfinite(loss_db), loss_db, np.nan)

    pred["ionosphere_shell_lower_alt_km"] = IONOSPHERE_LOWER_ALT_KM
    pred["ionosphere_shell_upper_alt_km"] = IONOSPHERE_UPPER_ALT_KM
    pred["ionosphere_l30_vertical_assumed_db"] = IONOSPHERE_L30_VERTICAL_DB
    pred["ionosphere_path_length_shell_km"] = shell_len
    pred["m_ion_shell_chord_raw"] = mion_raw
    pred["m_ion_proxy"] = mion
    pred["l_ion_abs_proxy_db"] = loss_db
    pred["l_ion_abs_budget_db"] = np.where(pred["earth_blocked"].astype(bool), 0.0, loss_db)
    pred["ionosphere_absorption_proxy_status"] = np.select(
        [
            pred["earth_blocked"].astype(bool) & (shell_len > 0.0),
            shell_len > 0.0,
        ],
        [
            "earth_blocked_but_shell_path_computed_for_flagging_not_direct_link",
            "engineering_proxy_spherical_shell_60_1000km_L30_0p5dB_capped_M5",
        ],
        default="no_intersection_with_reference_ionosphere_shell",
    )


def add_tropospheric_gas_absorption_proxy(pred: pd.DataFrame) -> None:
    rx = pred[["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]].to_numpy(float)
    sat = pred[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]].to_numpy(float)
    vertical_equiv_km = GAS_SCALE_HEIGHT_KM * (
        1.0 - math.exp(-TROPOSPHERE_UPPER_ALT_KM / GAS_SCALE_HEIGHT_KM)
    )
    equivalent_airmass_km_raw = integrate_standard_atmosphere_shell(
        rx,
        sat,
        EARTH_RADIUS_KM,
        EARTH_RADIUS_KM + TROPOSPHERE_UPPER_ALT_KM,
        GAS_SCALE_HEIGHT_KM,
    )
    equivalent_airmass_km = np.minimum(
        equivalent_airmass_km_raw,
        GAS_MAPPING_FACTOR_CAP * vertical_equiv_km,
    )
    m_gas = np.divide(
        equivalent_airmass_km,
        vertical_equiv_km,
        out=np.zeros_like(equivalent_airmass_km),
        where=vertical_equiv_km > 0.0,
    )
    gamma_surface = pred["signal_name"].astype(str).map(GAS_GAMMA_SURFACE_DB_PER_KM).astype(float)
    loss_db = gamma_surface.to_numpy(float) * equivalent_airmass_km

    pred["troposphere_shell_upper_alt_km"] = TROPOSPHERE_UPPER_ALT_KM
    pred["gas_scale_height_assumed_km"] = GAS_SCALE_HEIGHT_KM
    pred["gas_vertical_equivalent_airmass_km"] = vertical_equiv_km
    pred["gas_equivalent_airmass_km_raw"] = equivalent_airmass_km_raw
    pred["gas_equivalent_airmass_km"] = equivalent_airmass_km
    pred["m_gas_proxy"] = m_gas
    pred["gas_gamma_surface_db_per_km"] = gamma_surface
    pred["l_gas_abs_proxy_db"] = np.where(np.isfinite(loss_db), loss_db, np.nan)
    pred["l_gas_abs_budget_db"] = np.where(pred["earth_blocked"].astype(bool), 0.0, pred["l_gas_abs_proxy_db"])
    pred["gas_absorption_proxy_status"] = np.select(
        [
            pred["earth_blocked"].astype(bool) & (equivalent_airmass_km_raw > 0.0),
            equivalent_airmass_km_raw > equivalent_airmass_km,
            equivalent_airmass_km_raw > 0.0,
        ],
        [
            "earth_blocked_but_troposphere_path_computed_for_flagging_not_direct_link",
            "engineering_proxy_standard_atmosphere_0_20km_P676_surface_gamma_capped_M10",
            "engineering_proxy_standard_atmosphere_0_20km_P676_surface_gamma",
        ],
        default="no_intersection_with_reference_troposphere_shell",
    )


def add_edge_risk_features(pred: pd.DataFrame) -> None:
    earth_h = pred["earth_grazing_altitude_km"].astype(float)
    moon_h = pred["moon_grazing_altitude_km"].astype(float)
    pred["edge_near_limb_alt_threshold_km"] = EDGE_NEAR_LIMB_ALT_KM
    pred["neutral_refraction_upper_alt_km"] = NEUTRAL_REFRACTION_UPPER_ALT_KM
    pred["earth_edge_near_limb_0_20km"] = (
        (~pred["earth_blocked"].astype(bool))
        & earth_h.ge(0.0)
        & earth_h.le(EDGE_NEAR_LIMB_ALT_KM)
    ).astype(int)
    pred["moon_edge_near_limb_0_20km"] = (
        (~pred["moon_blocked"].astype(bool))
        & moon_h.ge(0.0)
        & moon_h.le(EDGE_NEAR_LIMB_ALT_KM)
    ).astype(int)
    pred["earth_neutral_refraction_20_60km"] = (
        (~pred["earth_blocked"].astype(bool))
        & earth_h.gt(EDGE_NEAR_LIMB_ALT_KM)
        & earth_h.le(NEUTRAL_REFRACTION_UPPER_ALT_KM)
    ).astype(int)
    pred["moon_neutral_refraction_20_60km"] = (
        (~pred["moon_blocked"].astype(bool))
        & moon_h.gt(EDGE_NEAR_LIMB_ALT_KM)
        & moon_h.le(NEUTRAL_REFRACTION_UPPER_ALT_KM)
    ).astype(int)
    pred["edge_ai_feature_status"] = np.select(
        [
            pred["earth_edge_near_limb_0_20km"].eq(1) | pred["moon_edge_near_limb_0_20km"].eq(1),
            pred["earth_neutral_refraction_20_60km"].eq(1) | pred["moon_neutral_refraction_20_60km"].eq(1),
        ],
        [
            "near_limb_soft_edge_high_risk_0_20km_ai_residual",
            "neutral_refraction_defocus_transition_20_60km_ai_residual",
        ],
        default="not_in_near_limb_or_neutral_transition_zone",
    )


def build_grap_azimuth_aggregate() -> pd.DataFrame:
    grap = pd.read_csv(INPUT_GRAP)
    grap = numeric(
        grap,
        ["azimuth_deg", "coelevation_deg", "eirp_dbw", "eirp_upper95_dbw", "eirp_lower95_dbw"],
    )
    # Avoid choosing an arbitrary phi before Galileo body/yaw mapping is validated.
    agg = (
        grap.groupby(["signal_name", "coelevation_deg"], as_index=False)
        .agg(
            tx_eirp_grap_azimuth_median_dbw=("eirp_dbw", "median"),
            tx_eirp_grap_azimuth_mean_dbw=("eirp_dbw", "mean"),
            tx_eirp_grap_azimuth_min_dbw=("eirp_dbw", "min"),
            tx_eirp_grap_azimuth_max_dbw=("eirp_dbw", "max"),
            tx_eirp_grap_upper95_median_dbw=("eirp_upper95_dbw", "median"),
            tx_eirp_grap_lower95_median_dbw=("eirp_lower95_dbw", "median"),
            tx_eirp_grap_azimuth_samples=("azimuth_deg", "count"),
        )
        .rename(columns={"coelevation_deg": "tx_eirp_grap_lookup_coelevation_deg"})
    )
    return agg


def merge_gps_direct(df: pd.DataFrame) -> pd.DataFrame:
    if not INPUT_GPS_DIRECT.exists():
        df["gps_l1_direct_merge_status"] = "gps_direct_file_missing"
        return df
    direct = pd.read_csv(INPUT_GPS_DIRECT, low_memory=False)
    keep_cols = [
        "rx_utc",
        "svid",
        "signal_name",
        "rx_offboresight_spice_deg",
        "rx_azimuth_spice_deg",
        "rx_gain_envelope_dbic",
        "tx_theta_body_deg",
        "tx_phi_body_deg",
        "tx_gain_2d_db",
        "tx_eirp_2d_dbw",
        "tx_power_dbw",
        "tx_yaw_quality",
        "tx_pattern_source",
        "tx_pattern_family",
        "tx_pattern_coverage",
        "tx_phi_alignment_quality",
        "tx_gnss_ssv_signal_band",
        "tx_ssv_main_lobe_boundary_deg",
        "tx_ssv_signed_lower_boundary_deg",
        "tx_ssv_signed_upper_boundary_deg",
        "tx_ssv_double_sided_full_width_deg",
        "tx_ssv_main_lobe_classification",
        "tx_ssv_main_lobe_boundary_basis",
        "tx_ssv_classification_semantics",
        "tx_central_half_power_classification",
        "cn0_reference_trajectory_2d_dbhz",
        "cn0_2d_model_status",
    ]
    direct = direct[[c for c in keep_cols if c in direct.columns]].copy()
    for col in ["rx_utc", "svid", "signal_name"]:
        direct[col] = direct[col].astype(str)
        df[col] = df[col].astype(str)
    direct = direct.drop_duplicates(["rx_utc", "svid", "signal_name"])
    merged = df.merge(direct, on=["rx_utc", "svid", "signal_name"], how="left", suffixes=("", "_gps_direct"))
    merged["gps_l1_direct_merge_status"] = np.where(
        merged["cn0_reference_trajectory_2d_dbhz"].notna(),
        "gps_band_specific_2d_direct_available",
        "gps_direct_not_available_for_row",
    )
    return merged


def build_predictions() -> pd.DataFrame:
    old = pd.read_csv(INPUT_OLD_BASELINE, low_memory=False)
    old = numeric(
        old,
        [
            "cn0_dbhz_mean",
            "fspl_db",
            "frequency_mhz",
            "tx_offboresight_deg",
            "cn0_fspl_only_global_dbhz",
            "cn0_physics_only_signal_fspl_dbhz",
            "cn0_proxy_geometry_phase_dbhz",
        ],
    )
    old["row_id_constellation"] = np.arange(len(old), dtype=int)
    old["frequency_band"] = old["signal_name"].astype(str).map(band_label)
    old["rx_peak_gain_dbic"] = old["signal_name"].astype(str).map(rx_peak_gain_dbic).astype(float)
    old["system_noise_temperature_k"] = old["signal_name"].astype(str).map(system_noise_temperature_k).astype(float)
    old["implementation_loss_assumed_db"] = IMPLEMENTATION_LOSS_ASSUMED_DB
    old["receiver_gain_model_status"] = "lugre_fig3_peak_gain_or_gps_l1_envelope_proxy;not_full_theta_phi"
    add_ionosphere_absorption_proxy(old)
    add_tropospheric_gas_absorption_proxy(old)
    add_edge_risk_features(old)

    grap_agg = build_grap_azimuth_aggregate()
    old["tx_eirp_grap_lookup_coelevation_deg"] = (
        old["tx_offboresight_deg"].clip(lower=0, upper=90).round().astype("Int64")
    )
    pred = old.merge(
        grap_agg,
        on=["signal_name", "tx_eirp_grap_lookup_coelevation_deg"],
        how="left",
    )
    pred["galileo_grap_lookup_status"] = np.select(
        [
            pred["signal_name"].isin(["GAL_E1", "GAL_E5a"]) & pred["tx_eirp_grap_azimuth_median_dbw"].notna(),
            pred["signal_name"].isin(["GAL_E1", "GAL_E5a"]) & pred["tx_eirp_grap_azimuth_median_dbw"].isna(),
        ],
        [
            "diagnostic_azimuth_aggregated_grap_lookup_offboresight_as_coelevation",
            "galileo_grap_lookup_missing",
        ],
        default="not_galileo_grap_applicable",
    )
    pred["galileo_grap_coordinate_mapping_status"] = np.where(
        pred["signal_name"].isin(["GAL_E1", "GAL_E5a"]),
        "diagnostic_not_full_body_yaw_mapping;azimuth_aggregated_not_phi_specific",
        "not_applicable",
    )

    pred = merge_gps_direct(pred)
    pred = numeric(
        pred,
        [
            "rx_gain_envelope_dbic",
            "tx_gain_2d_db",
            "tx_power_dbw",
            "cn0_reference_trajectory_2d_dbhz",
        ],
    )
    pred["gps_tx_physics_status"] = np.select(
        [
            pred["cn0_reference_trajectory_2d_dbhz"].notna(),
            pred["system"].astype(str).eq("G"),
        ],
        [
            "gps_band_specific_2d_direct_budget_available_from_attitude_pipeline",
            "gps_transmit_directivity_not_rowwise_available_keep_proxy_status",
        ],
        default="not_gps",
    )

    pred["cn0_galileo_grap_peakrx_dbhz"] = np.nan
    gal = pred["signal_name"].isin(["GAL_E1", "GAL_E5a"]) & pred["tx_eirp_grap_azimuth_median_dbw"].notna()
    pred.loc[gal, "cn0_galileo_grap_peakrx_dbhz"] = (
        pred.loc[gal, "tx_eirp_grap_azimuth_median_dbw"]
        - pred.loc[gal, "fspl_db"]
        + pred.loc[gal, "rx_peak_gain_dbic"]
        + LIGHT_SPEED_LINK_CONSTANT_DB
        - 10.0 * np.log10(pred.loc[gal, "system_noise_temperature_k"])
        - IMPLEMENTATION_LOSS_ASSUMED_DB
        - pred.loc[gal, "l_ion_abs_budget_db"]
        - pred.loc[gal, "l_gas_abs_budget_db"]
    )
    for src, out_col in [
        ("tx_eirp_grap_upper95_median_dbw", "cn0_galileo_grap_upper95_peakrx_dbhz"),
        ("tx_eirp_grap_lower95_median_dbw", "cn0_galileo_grap_lower95_peakrx_dbhz"),
    ]:
        pred[out_col] = np.nan
        pred.loc[gal, out_col] = (
            pred.loc[gal, src]
            - pred.loc[gal, "fspl_db"]
            + pred.loc[gal, "rx_peak_gain_dbic"]
            + LIGHT_SPEED_LINK_CONSTANT_DB
            - 10.0 * np.log10(pred.loc[gal, "system_noise_temperature_k"])
            - IMPLEMENTATION_LOSS_ASSUMED_DB
            - pred.loc[gal, "l_ion_abs_budget_db"]
            - pred.loc[gal, "l_gas_abs_budget_db"]
        )

    pred["cn0_constellation_direct_available_dbhz"] = np.nan
    pred["constellation_direct_status"] = "unsupported_no_direct_transmit_or_receive_model"
    rowwise_direct = pred["cn0_reference_trajectory_2d_dbhz"].notna()
    pred["cn0_reference_trajectory_2d_atm_proxy_dbhz"] = (
        pred["cn0_reference_trajectory_2d_dbhz"]
        - pred["l_ion_abs_budget_db"]
        - pred["l_gas_abs_budget_db"]
    )
    pred["cn0_reference_trajectory_2d_iono_proxy_dbhz"] = pred["cn0_reference_trajectory_2d_atm_proxy_dbhz"]
    pred.loc[rowwise_direct, "cn0_constellation_direct_available_dbhz"] = pred.loc[
        rowwise_direct, "cn0_reference_trajectory_2d_atm_proxy_dbhz"
    ]
    pred.loc[rowwise_direct, "constellation_direct_status"] = "band_specific_rowwise_2d_direct_budget"
    gal_fallback = gal & ~rowwise_direct
    pred.loc[gal_fallback, "cn0_constellation_direct_available_dbhz"] = pred.loc[
        gal_fallback, "cn0_galileo_grap_peakrx_dbhz"
    ]
    pred.loc[gal_fallback, "constellation_direct_status"] = "galileo_grap_azimuth_aggregated_fallback_budget"

    # Full-row baseline for residual learning. Unsupported rows keep the old proxy baseline
    # and are visibly flagged; direct rows use the newly available physical/diagnostic term.
    pred["cn0_constellation_physics_proxy_dbhz"] = (
        pred["cn0_proxy_geometry_phase_dbhz"]
        - pred["l_ion_abs_budget_db"]
        - pred["l_gas_abs_budget_db"]
    )
    pred["constellation_physics_proxy_status"] = "fallback_legacy_proxy_geometry_phase_train_calibrated"
    supported = pred["cn0_constellation_direct_available_dbhz"].notna()
    pred.loc[supported, "cn0_constellation_physics_proxy_dbhz"] = pred.loc[
        supported, "cn0_constellation_direct_available_dbhz"
    ]
    pred.loc[supported, "constellation_physics_proxy_status"] = pred.loc[supported, "constellation_direct_status"]
    pred["constellation_physics_no_observation_tx_fit"] = True
    pred["constellation_physics_note"] = (
        "Galileo GRAP is azimuth-aggregated diagnostic EIRP; GPS direct only where existing 2D pipeline supports row; "
        "unsupported rows explicitly fall back to legacy proxy baseline for full-row residual training; "
        "L-band ionospheric absorption and tropospheric gaseous absorption use small engineering spherical-shell proxies; "
        "only non-Earth-blocked direct-LOS rows use them as C/N0 budget losses."
    )
    pred["error_cn0_constellation_physics_proxy_dbhz"] = (
        pred["cn0_constellation_physics_proxy_dbhz"] - pred["cn0_dbhz_mean"]
    )
    pred["error_cn0_constellation_direct_available_dbhz"] = (
        pred["cn0_constellation_direct_available_dbhz"] - pred["cn0_dbhz_mean"]
    )
    return pred


def build_metric_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_cols = [
        ("cn0_constellation_physics_proxy_dbhz", "constellation_physics_proxy_full_rows"),
        ("cn0_constellation_direct_available_dbhz", "constellation_direct_available_subset"),
        ("cn0_proxy_geometry_phase_dbhz", "legacy_proxy_geometry_phase_reference"),
        ("cn0_physics_only_signal_fspl_dbhz", "legacy_signal_fspl_reference"),
    ]
    groups = [("overall", "all", pred.index)]
    for col in ["split", "signal_name", "mission_phase", "system", "constellation_physics_proxy_status"]:
        for value, g in pred.groupby(col, dropna=False):
            groups.append((col, value, g.index))
    for col, model_name in model_cols:
        for group_type, group_value, idx in groups:
            d = pred.loc[idx]
            m = metrics(d["cn0_dbhz_mean"].to_numpy(float), d[col].to_numpy(float))
            rows.append(
                {
                    "model_name": model_name,
                    "prediction_column": col,
                    "group_type": group_type,
                    "group_value": group_value,
                    **m,
                }
            )
    return pd.DataFrame(rows)


def write_readme(pred: pd.DataFrame, metric_df: pd.DataFrame) -> None:
    inventory_text = INPUT_INVENTORY.read_text(encoding="utf-8", errors="replace") if INPUT_INVENTORY.exists() else ""
    interface_note = INPUT_INTERFACE.read_text(encoding="utf-8", errors="replace") if INPUT_INTERFACE.exists() else ""
    status_counts = pred["constellation_physics_proxy_status"].value_counts(dropna=False).to_dict()
    test_metrics = metric_df[
        (metric_df["group_type"].eq("split"))
        & (metric_df["group_value"].eq("test"))
        & (metric_df["model_name"].isin(["constellation_physics_proxy_full_rows", "legacy_proxy_geometry_phase_reference", "legacy_signal_fspl_reference"]))
    ][["model_name", "n", "mae_dbhz", "rmse_dbhz", "bias_dbhz", "r2"]]
    metric_lines = ["| model_name | n | MAE | RMSE | bias | R2 |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in test_metrics.iterrows():
        metric_lines.append(
            f"| {row['model_name']} | {int(row['n'])} | {row['mae_dbhz']:.3f} | {row['rmse_dbhz']:.3f} | {row['bias_dbhz']:.3f} | {row['r2']:.3f} |"
        )
    lines = [
        "# C/N0 Constellation-Aware Physics Baseline",
        "",
        "This table front-loads known GPS/Galileo transmitter-side differences where public data permit it.",
        "",
        "## Key Interpretation",
        "",
        "- Galileo E1/E5a uses the official GRAP v1.0 EIRP grid only as an azimuth-aggregated diagnostic lookup because full Galileo yaw/body-frame azimuth mapping is not yet validated.",
        "- GPS uses the existing GPS L1 2-D attitude/directivity direct-budget subset where available; other GPS rows remain proxy/status only.",
        "- LuGRE receive-side gain remains peak/Fig.3-envelope proxy, not a full `G_rx(theta,phi,band)` model.",
        "- No observation C/N0 is used to fit GPS/Galileo transmit-side differences in the direct/diagnostic terms.",
        "- `cn0_constellation_physics_proxy_dbhz` is full-row for residual training; unsupported rows are explicitly marked as legacy proxy fallback.",
        "",
        "## Status Counts",
        "",
        "```json",
        pd.Series(status_counts).to_json(force_ascii=False, indent=2),
        "```",
        "",
        "## Test Metrics Snapshot",
        "",
        "\n".join(metric_lines),
        "",
        "## Source Interface Summary",
        "",
        interface_note[:3000],
        "",
        "## Inventory Snapshot",
        "",
        inventory_text[:3000],
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = build_predictions()
    features_cols = [
        c
        for c in pred.columns
        if c
        in {
            "row_id_constellation",
            "rx_utc",
            "rx_gps_seconds",
            "split",
            "system",
            "svid",
            "signal_name",
            "mission_phase",
            "frequency_mhz",
            "fspl_db",
            "tx_offboresight_deg",
            "tx_eirp_grap_lookup_coelevation_deg",
            "tx_eirp_grap_azimuth_median_dbw",
            "tx_eirp_grap_azimuth_mean_dbw",
            "tx_eirp_grap_azimuth_min_dbw",
            "tx_eirp_grap_azimuth_max_dbw",
            "tx_eirp_grap_upper95_median_dbw",
            "tx_eirp_grap_lower95_median_dbw",
            "galileo_grap_lookup_status",
            "galileo_grap_coordinate_mapping_status",
            "gps_l1_direct_merge_status",
            "gps_tx_physics_status",
            "rx_peak_gain_dbic",
            "system_noise_temperature_k",
            "ionosphere_shell_lower_alt_km",
            "ionosphere_shell_upper_alt_km",
            "ionosphere_l30_vertical_assumed_db",
            "ionosphere_path_length_shell_km",
            "m_ion_shell_chord_raw",
            "m_ion_proxy",
            "l_ion_abs_proxy_db",
            "l_ion_abs_budget_db",
            "ionosphere_absorption_proxy_status",
            "troposphere_shell_upper_alt_km",
            "gas_scale_height_assumed_km",
            "gas_vertical_equivalent_airmass_km",
            "gas_equivalent_airmass_km_raw",
            "gas_equivalent_airmass_km",
            "m_gas_proxy",
            "gas_gamma_surface_db_per_km",
            "l_gas_abs_proxy_db",
            "l_gas_abs_budget_db",
            "gas_absorption_proxy_status",
            "edge_near_limb_alt_threshold_km",
            "neutral_refraction_upper_alt_km",
            "earth_edge_near_limb_0_20km",
            "moon_edge_near_limb_0_20km",
            "earth_neutral_refraction_20_60km",
            "moon_neutral_refraction_20_60km",
            "edge_ai_feature_status",
            "receiver_gain_model_status",
            "cn0_galileo_grap_peakrx_dbhz",
            "cn0_reference_trajectory_2d_dbhz",
            "cn0_reference_trajectory_2d_iono_proxy_dbhz",
            "cn0_reference_trajectory_2d_atm_proxy_dbhz",
            "cn0_constellation_direct_available_dbhz",
            "cn0_constellation_physics_proxy_dbhz",
            "constellation_direct_status",
            "constellation_physics_proxy_status",
        }
    ]
    metric_df = build_metric_table(pred)
    pred[features_cols].to_csv(OUT_DIR / "cn0_constellation_physics_features.csv", index=False)
    pred.to_csv(OUT_DIR / "cn0_constellation_physics_predictions.csv", index=False)
    metric_df.to_csv(OUT_DIR / "cn0_constellation_physics_metrics.csv", index=False)
    write_readme(pred, metric_df)
    print(f"rows={len(pred)}")
    print(pred["constellation_physics_proxy_status"].value_counts(dropna=False).to_string())
    print(metric_df[(metric_df.group_type == "split") & (metric_df.group_value == "test")][["model_name", "n", "mae_dbhz", "rmse_dbhz", "bias_dbhz"]].to_string(index=False))


if __name__ == "__main__":
    main()
