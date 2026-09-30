#!/usr/bin/env python3
"""Add LuGRE SPICE attitude and full 2-D GPS transmit gain to C/N0.

The receiver geometry uses the public Blue Ghost CK/FK chain. GPS satellite
attitude uses a nominal yaw-steering frame (+Z nadir, +X toward the projected
Sun, +Y completing a right-handed frame). Eclipse and noon/midnight singular
regions are explicitly flagged because actual yaw maneuvers are not modeled.
No observed C/N0 value is used to fit or correct the direct prediction.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import openpyxl
import pandas as pd
from pypdf import PdfReader
import spiceypy as sp

from gps_iif_antenna_patterns import parse_iif_patterns
from orbex_attitude import DEFAULT_ORBEX_DIR, orbex_body_angles


DEFAULT_INPUT = Path("table/cn0_physics_features_available_wgc_new.csv")
DEFAULT_MAPPING = Path("data/external_reference/mapping/active_prn_svn_block_20250115_20250316.csv")
DEFAULT_GPS_DIR = Path("data/external_reference/gnss_antenna/gps")
DEFAULT_RX_PATTERN = Path("data/external_reference/lugre_antenna/LuGRE_Fig3_gain_outer_envelope_digitized.csv")
DEFAULT_GRAP = Path("table/external_reference/galileo_grap_eirp_grid.csv")
DEFAULT_SPICE = Path("data/external_reference/spice")
DEFAULT_OUTPUT = Path("table/algorithm/cn0_attitude_2d_gain")

RX_TSYS_K = {"GPS_L1": 182.0, "GAL_E1": 182.0, "GPS_L5": 231.0, "GAL_E5a": 231.0}
TX_POWER_DBW = {
    # IIR/IIF retain the LuGRE pre-flight values; the 2026 flight analysis
    # provides CYGNSS-calibrated averages only for IIR-M and IIIA.
    "BLOCK IIR-A": 17.3,
    "BLOCK IIR-B": 17.3,
    "BLOCK IIR-M": 16.5,
    "BLOCK IIF": 16.2,
    "BLOCK IIIA": 15.6,
}
EARTH_RADIUS_KM = 6378.137
K_DB = 228.6
GPS_SIGNAL_TO_BAND = {"GPS_L1": "L1", "GPS_L2": "L2", "GPS_L5": "L5"}
GNSS_SIGNAL_TO_SSV_BAND = {
    "GPS_L1": "L1",
    "GPS_L2": "L2",
    "GPS_L5": "L5",
    "GAL_E1": "E1",
    "GAL_E6": "E6",
    "GAL_E5b": "E5b",
    "GAL_E5": "E5",
    "GAL_E5a": "E5a",
}
GNSS_SSV_MAIN_LOBE_BOUNDARY_DEG = {
    "GPS_L1": 23.5,
    "GPS_L2": 26.0,
    "GPS_L5": 26.0,
    "GAL_E1": 20.5,
    "GAL_E6": 21.5,
    "GAL_E5b": 22.5,
    "GAL_E5": 23.5,
    "GAL_E5a": 23.5,
}
GNSS_SSV_BOUNDARY_BASIS = (
    "UNOOSA_ICG_ST_SPACE_75_Rev1_Table_4_2_reference_off_boresight_angle;"
    "GPS_L1_23.5deg_GPS_L2_L5_26.0deg_GAL_E1_20.5deg_GAL_E5a_23.5deg;"
    "service_boundary_not_electromagnetic_sidelobe_boundary"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--gps-antenna-dir", type=Path, default=DEFAULT_GPS_DIR)
    parser.add_argument("--rx-pattern", type=Path, default=DEFAULT_RX_PATTERN)
    parser.add_argument("--grap", type=Path, default=DEFAULT_GRAP)
    parser.add_argument("--spice-dir", type=Path, default=DEFAULT_SPICE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    output = np.full_like(vectors, np.nan, dtype=float)
    valid = np.isfinite(norms) & (norms > 0.0)
    output[valid] = vectors[valid] / norms[valid, None]
    return output, norms


def parse_iirm_2d(pptx_path: Path) -> dict[int, dict]:
    patterns: dict[int, dict] = {}
    with zipfile.ZipFile(pptx_path) as archive:
        for slide_index in range(3, 43):
            slide_xml = archive.read(f"ppt/slides/slide{slide_index}.xml").decode("utf-8", errors="ignore")
            match = re.search(r"SVN(\d+) Antenna Pattern .* L1", slide_xml)
            if not match:
                continue
            svn = int(match.group(1))
            rel_xml = archive.read(f"ppt/slides/_rels/slide{slide_index}.xml.rels").decode("utf-8", errors="ignore")
            target = re.search(r'Target="\.\./embeddings/([^\"]+\.xlsx)"', rel_xml)
            if not target:
                continue
            workbook = openpyxl.load_workbook(
                io.BytesIO(archive.read(f"ppt/embeddings/{target.group(1)}")),
                data_only=True,
                read_only=True,
            )
            sheet = next(ws for ws in workbook.worksheets if "data" in ws.title.lower())
            header = list(next(sheet.iter_rows(min_row=2, max_row=2, values_only=True)))
            phi = np.asarray([x for x in header[1:] if isinstance(x, (int, float))], dtype=float)
            theta_rows, gains = [], []
            for row in sheet.iter_rows(min_row=4, values_only=True):
                if not isinstance(row[0], (int, float)):
                    continue
                values = row[1 : 1 + len(phi)]
                if len(values) == len(phi) and all(isinstance(x, (int, float)) for x in values):
                    theta_rows.append(float(row[0]))
                    gains.append([float(x) for x in values])
            theta = np.asarray(theta_rows)
            gain = np.asarray(gains)
            keep = theta >= 0.0
            patterns[svn] = {
                "theta": theta[keep],
                "phi": phi,
                "gain": gain[keep],
                "source": pptx_path.as_posix() + f"#SVN{svn}_L1_embedded_xlsx",
                "block_pattern_family": "GPS_IIR_IIR-M",
            }
    return patterns


def parse_iiia_2d(gps_dir: Path, band: str = "L1") -> dict[int, dict]:
    patterns: dict[int, dict] = {}
    for svn in range(74, 79):
        candidates = sorted(gps_dir.glob(f"GPS_III_SVN{svn}*Directivity.zip"))
        if not candidates:
            continue
        archive_path = candidates[0]
        with zipfile.ZipFile(archive_path) as archive:
            pdf_name = next(
                name
                for name in archive.namelist()
                if re.search(fr"_{band}_.*Directivity\.pdf$", name)
            )
            reader = PdfReader(io.BytesIO(archive.read(pdf_name)))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        theta_rows, gains = [], []
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) == 37 and re.fullmatch(r"-?\d+", tokens[0]):
                theta = int(tokens[0])
                if 0 <= theta <= 90:
                    theta_rows.append(float(theta))
                    gains.append([float(x) for x in tokens[1:]])
        if len(theta_rows) != 46:
            raise ValueError(
                f"Unexpected GPS III {band} grid for SVN{svn}: {len(theta_rows)} rows"
            )
        patterns[svn] = {
            "theta": np.asarray(theta_rows),
            "phi": np.arange(0.0, 360.0, 10.0),
            "gain": np.asarray(gains),
            "source": archive_path.as_posix() + f"#{pdf_name}",
            "block_pattern_family": "GPS_IIIA",
        }
    return patterns


def bilinear_periodic(pattern: dict, theta_q: np.ndarray, phi_q: np.ndarray) -> np.ndarray:
    theta = np.asarray(pattern["theta"], dtype=float)
    phi = np.asarray(pattern["phi"], dtype=float)
    gain = np.asarray(pattern["gain"], dtype=float)
    phi_ext = np.r_[phi, 360.0]
    gain_ext = np.column_stack([gain, gain[:, 0]])
    output = np.full(len(theta_q), np.nan)
    valid = np.isfinite(theta_q) & np.isfinite(phi_q) & (theta_q >= theta.min()) & (theta_q <= theta.max())
    if not valid.any():
        return output
    tq = theta_q[valid]
    pq = np.mod(phi_q[valid], 360.0)
    ti = np.clip(np.searchsorted(theta, tq, side="right") - 1, 0, len(theta) - 2)
    pi = np.clip(np.searchsorted(phi_ext, pq, side="right") - 1, 0, len(phi_ext) - 2)
    t0, t1 = theta[ti], theta[ti + 1]
    p0, p1 = phi_ext[pi], phi_ext[pi + 1]
    wt = np.divide(tq - t0, t1 - t0, out=np.zeros_like(tq), where=(t1 != t0))
    wp = np.divide(pq - p0, p1 - p0, out=np.zeros_like(pq), where=(p1 != p0))
    g00 = gain_ext[ti, pi]
    g01 = gain_ext[ti, pi + 1]
    g10 = gain_ext[ti + 1, pi]
    g11 = gain_ext[ti + 1, pi + 1]
    output[valid] = (1 - wt) * ((1 - wp) * g00 + wp * g01) + wt * ((1 - wp) * g10 + wp * g11)
    return output


def load_spice(spice_dir: Path) -> None:
    files = [
        "lsk/naif0012.tls", "pck/pck00011.tpc", "pck/earth_2025_250826_2125_predict.bpc",
        "pck/earth_000101_260716_260419.bpc", "pck/moon_pa_de440_200625.bpc",
        "fk/moon_de440_250416.tf", "fk/earth_topo_201023.tf", "fk/earth_assoc_itrf93.tf",
        "fk/moon_assoc_me.tf", "spk/de440s.bsp", "spk/earthstns_itrf93_201023.bsp",
        "fk/clps_to19d_bgm1_v01.tf", "sclk/clps_to19d_bgm1_v01.tsc",
        "spk/clps_to19d_bgm1_struct_v01.bsp", "spk/clps_to19d_bgm1_ls_250302_v01.bsp",
        "spk/clps_to19d_bgm1_atls_250302_v01.bsp", "spk/clps_to19d_bgm1_cru_rec_250115_250302_v01.bsp",
        "spk/clps_to19d_bgm1_cru_pre_250302_v01.bsp", "spk/clps_to19d_bgm1_edl_rec_250302_v01.bsp",
        "ck/clps_to19d_bgm1_cru_rec_250115_250310_v01.bc", "ck/clps_to19d_bgm1_edl_rec_250302_v01.bc",
        "ck/clps_to19d_bgm1_surf_ops_v01.bc", "ck/clps_to19d_bgm1_hga_stowed_v01.bc",
        "ck/clps_to19d_bgm1_hga_mea_v01.bc",
    ]
    missing = [name for name in files if not (spice_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing SPICE kernels: {missing}")
    for name in files:
        # CSPICE on Windows can fail on non-ASCII absolute paths. The pipeline
        # is run from the workspace root, so keep kernel paths relative.
        sp.furnsh(str(spice_dir / name))


def time_geometry(times: pd.Series) -> dict[pd.Timestamp, tuple[np.ndarray | None, np.ndarray, str]]:
    result = {}
    for timestamp in sorted(times.unique()):
        ts = pd.Timestamp(timestamp)
        et = sp.utc2et(ts.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        sun = np.asarray(sp.spkpos("SUN", et, "J2000", "NONE", "EARTH")[0], dtype=float)
        try:
            lugre_to_j2000 = np.asarray(sp.pxform("BGM1_LUGRE", "J2000", et), dtype=float)
            result[ts] = (lugre_to_j2000.T, sun, "spice_attitude_available")
        except Exception:
            result[ts] = (None, sun, "spice_attitude_gap")
    return result


def earth_eclipse(sat: np.ndarray, sun: np.ndarray) -> np.ndarray:
    direction = sun - sat
    denominator = np.sum(direction * direction, axis=1)
    t = -np.sum(sat * direction, axis=1) / denominator
    closest = sat + t[:, None] * direction
    return (t > 0.0) & (t < 1.0) & (np.linalg.norm(closest, axis=1) < EARTH_RADIUS_KM)


def receiver_gain(df: pd.DataFrame, pattern_path: Path) -> None:
    pattern = pd.read_csv(pattern_path)
    df["rx_gain_envelope_dbic"] = np.nan
    df["rx_gain_digitization_uncertainty_db"] = np.nan
    band_map = {"GPS_L1": "L1/E1", "GAL_E1": "L1/E1", "GPS_L5": "L5/E5a", "GAL_E5a": "L5/E5a"}
    for signal, band in band_map.items():
        source = pattern[pattern["band"].eq(band)].sort_values("signed_elevation_deg")
        mask = df["signal_name"].eq(signal) & df["rx_offboresight_spice_deg"].notna()
        angles = df.loc[mask, "rx_offboresight_spice_deg"].to_numpy(float)
        axis = source["signed_elevation_deg"].to_numpy(float)
        gain = source["outer_envelope_gain_dB_axis"].to_numpy(float)
        uncertainty = source["digitization_uncertainty_bound_dB"].to_numpy(float)
        gain_plus = np.interp(angles, axis, gain)
        gain_minus = np.interp(-angles, axis, gain)
        choose_plus = gain_plus >= gain_minus
        df.loc[mask, "rx_gain_envelope_dbic"] = np.maximum(gain_plus, gain_minus)
        df.loc[mask, "rx_gain_digitization_uncertainty_db"] = np.where(
            choose_plus, np.interp(angles, axis, uncertainty), np.interp(-angles, axis, uncertainty)
        )


def galileo_grap_eirp(df: pd.DataFrame, grap_path: Path) -> None:
    """Evaluate the public Galileo GRAP EIRP grid in the same yaw frame as GPS."""
    grap = pd.read_csv(grap_path)
    for signal in ("GAL_E1", "GAL_E5a"):
        source = grap[grap["signal_name"].eq(signal)]
        pivot = (
            source.pivot_table(
                index="coelevation_deg",
                columns="azimuth_deg",
                values="eirp_dbw",
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        columns = pivot.columns.to_numpy(float)
        keep = columns < 360.0
        pattern = {
            "theta": pivot.index.to_numpy(float),
            "phi": columns[keep],
            "gain": pivot.to_numpy(float)[:, keep],
        }
        mask = df["signal_name"].eq(signal)
        df.loc[mask, "tx_eirp_2d_dbw"] = bilinear_periodic(
            pattern,
            df.loc[mask, "tx_theta_body_deg"].to_numpy(float),
            df.loc[mask, "tx_phi_body_deg"].to_numpy(float),
        )
        df.loc[mask, "tx_pattern_source"] = (
            f"{grap_path.as_posix()}#{signal}_full_2D_EIRP"
        )
        df.loc[mask, "tx_pattern_family"] = "Galileo_GRAP_v1_0_2D_EIRP"
        df.loc[mask, "tx_pattern_coverage"] = "official_reference_2D_0_90deg"


def add_gnss_ssv_lobe_labels(
    df: pd.DataFrame, patterns_by_signal: dict[str, dict[int, dict]]
) -> pd.DataFrame:
    """Add service-volume and independent half-power classifications.

    The SSV boundary is a service-angle definition. Exceeding it does not, by
    itself, identify a particular electromagnetic sidelobe. The half-power
    label is separately derived from the satellite-specific 2-D pattern.
    """
    df["tx_gps_frequency_band"] = df["signal_name"].map(GPS_SIGNAL_TO_BAND)
    df["tx_gnss_ssv_signal_band"] = df["signal_name"].map(GNSS_SIGNAL_TO_SSV_BAND)
    df["tx_ssv_main_lobe_boundary_deg"] = df["signal_name"].map(
        GNSS_SSV_MAIN_LOBE_BOUNDARY_DEG
    )
    df["tx_ssv_signed_lower_boundary_deg"] = -df["tx_ssv_main_lobe_boundary_deg"]
    df["tx_ssv_signed_upper_boundary_deg"] = df["tx_ssv_main_lobe_boundary_deg"]
    df["tx_ssv_double_sided_full_width_deg"] = 2.0 * df["tx_ssv_main_lobe_boundary_deg"]
    df["tx_ssv_angle_representation"] = np.where(
        df["tx_gnss_ssv_signal_band"].notna(),
        "3d_radial_half_cone_theta;equivalent_signed_2d_cut_runs_from_negative_to_positive_boundary",
        "not_applicable",
    )
    df["tx_ssv_main_lobe_classification"] = "not_applicable_unknown_signal_or_band"
    evaluable_ssv = (
        df["tx_gnss_ssv_signal_band"].notna()
        & df["tx_theta_body_deg"].notna()
        & df["tx_ssv_main_lobe_boundary_deg"].notna()
    )
    inside_ssv = evaluable_ssv & (
        df["tx_theta_body_deg"] <= df["tx_ssv_main_lobe_boundary_deg"]
    )
    df.loc[inside_ssv, "tx_ssv_main_lobe_classification"] = "ssv_main_lobe_service"
    df.loc[evaluable_ssv & ~inside_ssv, "tx_ssv_main_lobe_classification"] = (
        "outside_ssv_main_lobe"
    )
    df["tx_ssv_main_lobe_boundary_basis"] = np.where(
        df["tx_gnss_ssv_signal_band"].notna(), GNSS_SSV_BOUNDARY_BASIS, "not_applicable"
    )
    df["tx_ssv_classification_semantics"] = np.where(
        df["tx_gnss_ssv_signal_band"].notna(),
        "service_angle_classification;outside_does_not_identify_specific_electromagnetic_sidelobe",
        "not_applicable",
    )

    df["tx_pattern_boresight_gain_db"] = np.nan
    for signal, patterns in patterns_by_signal.items():
        for svn, pattern in patterns.items():
            theta = np.asarray(pattern["theta"], dtype=float)
            gain = np.asarray(pattern["gain"], dtype=float)
            boresight_rows = np.isclose(theta, 0.0)
            if boresight_rows.any():
                mask = df["signal_name"].eq(signal) & df["svn"].eq(svn)
                df.loc[mask, "tx_pattern_boresight_gain_db"] = float(
                    np.nanmedian(gain[boresight_rows])
                )
    df["tx_gain_relative_to_boresight_db"] = (
        df["tx_gain_2d_db"] - df["tx_pattern_boresight_gain_db"]
    )
    df["tx_central_half_power_threshold_relative_db"] = np.where(
        df["tx_pattern_boresight_gain_db"].notna(), -3.0, np.nan
    )
    df["tx_central_half_power_classification"] = "not_evaluated_pattern_gain_unavailable"
    df.loc[df["tx_gnss_ssv_signal_band"].isna(), "tx_central_half_power_classification"] = (
        "not_applicable_unknown_signal_or_band"
    )
    evaluable_half_power = (
        df["tx_gain_relative_to_boresight_db"].notna()
        & df["tx_ssv_main_lobe_boundary_deg"].notna()
    )
    central_half_power = (
        evaluable_half_power
        & (df["tx_gain_relative_to_boresight_db"] >= -3.0)
        & (df["tx_theta_body_deg"] <= df["tx_ssv_main_lobe_boundary_deg"])
    )
    df.loc[central_half_power, "tx_central_half_power_classification"] = (
        "central_half_power_region"
    )
    df.loc[
        evaluable_half_power & ~central_half_power,
        "tx_central_half_power_classification",
    ] = "outside_central_half_power_region"
    df["tx_central_half_power_basis"] = np.where(
        evaluable_half_power,
        "satellite_specific_2d_gain_at_theta_phi_at_least_boresight_minus_3dB;independent_of_ssv_boundary",
        "not_evaluated_pattern_gain_unavailable",
    )

    ssv_counts = (
        df[df["tx_gnss_ssv_signal_band"].notna()]
        .groupby(
            [
                "signal_name",
                "tx_gnss_ssv_signal_band",
                "tx_gps_frequency_band",
                "tx_ssv_main_lobe_boundary_deg",
                "tx_ssv_signed_lower_boundary_deg",
                "tx_ssv_signed_upper_boundary_deg",
                "tx_ssv_double_sided_full_width_deg",
                "tx_ssv_main_lobe_classification",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .rename(columns={"tx_ssv_main_lobe_classification": "classification"})
    )
    ssv_counts["classification_family"] = "gnss_ssv_service_main_lobe"
    ssv_counts["basis"] = GNSS_SSV_BOUNDARY_BASIS

    half_counts = (
        df[df["tx_gnss_ssv_signal_band"].notna()]
        .groupby(
            [
                "signal_name",
                "tx_gnss_ssv_signal_band",
                "tx_gps_frequency_band",
                "tx_ssv_main_lobe_boundary_deg",
                "tx_ssv_signed_lower_boundary_deg",
                "tx_ssv_signed_upper_boundary_deg",
                "tx_ssv_double_sided_full_width_deg",
                "tx_central_half_power_classification",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .rename(columns={"tx_central_half_power_classification": "classification"})
    )
    half_counts["classification_family"] = "central_half_power_optional_label"
    half_counts["basis"] = (
        "satellite_specific_2d_gain_at_theta_phi_at_least_boresight_minus_3dB"
    )
    columns = [
        "classification_family",
        "signal_name",
        "tx_gnss_ssv_signal_band",
        "tx_gps_frequency_band",
        "tx_ssv_main_lobe_boundary_deg",
        "tx_ssv_signed_lower_boundary_deg",
        "tx_ssv_signed_upper_boundary_deg",
        "tx_ssv_double_sided_full_width_deg",
        "classification",
        "rows",
        "basis",
    ]
    return pd.concat([ssv_counts[columns], half_counts[columns]], ignore_index=True)


def summary_metrics(df: pd.DataFrame, prediction: str) -> pd.DataFrame:
    rows = []
    valid = df[prediction].notna()
    groups = [("overall", pd.Series("all", index=df.index)), ("phase", df["mission_phase"]), ("yaw_quality", df["tx_yaw_quality"]), ("block", df["antenna_type"])]
    for group_type, values in groups:
        labels = values.fillna("missing").astype(str)
        for group in sorted(labels[valid].unique()):
            mask = valid & labels.eq(group)
            residual = df.loc[mask, "cn0_dbhz_mean"].to_numpy(float) - df.loc[mask, prediction].to_numpy(float)
            rows.append({
                "group_type": group_type, "group_value": group, "n": int(mask.sum()),
                "median_flight_minus_model_db": float(np.median(residual)),
                "mae_dbhz": float(np.mean(np.abs(residual))),
                "rmse_dbhz": float(np.sqrt(np.mean(residual**2))),
                "p05_flight_minus_model_db": float(np.percentile(residual, 5)),
                "p95_flight_minus_model_db": float(np.percentile(residual, 95)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)
    df["rx_utc"] = pd.to_datetime(df["rx_utc"], utc=True)
    if "receiver_trajectory_source" not in df.columns:
        df["receiver_trajectory_source"] = "WGC_new_reference_state_vectors"
    df["reference_trajectory_use"] = "trend_only_not_dynamic_prediction"

    mapping = pd.read_csv(args.mapping)
    mapping = mapping[mapping["system"].eq("G")].copy()
    mapping["svid"] = mapping["prn"].str.extract(r"G(\d+)").astype(int)
    mapping["svn"] = mapping["sv_identifier"].str.extract(r"G(\d+)").astype(int)
    mapping = mapping[["system", "svid", "svn", "antenna_type"]].drop_duplicates(
        ["system", "svid"]
    )
    df = df.merge(mapping, on=["system", "svid"], how="left")

    load_spice(args.spice_dir)
    try:
        geometry = time_geometry(df["rx_utc"])
    finally:
        sp.kclear()

    rx = df[["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]].to_numpy(float)
    sat = df[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]].to_numpy(float)
    los_rx, _ = normalize(sat - rx)
    los_tx = -los_rx
    sun = np.vstack([geometry[pd.Timestamp(t)][1] for t in df["rx_utc"]])
    rotations = [geometry[pd.Timestamp(t)][0] for t in df["rx_utc"]]
    df["rx_attitude_status"] = [geometry[pd.Timestamp(t)][2] for t in df["rx_utc"]]
    los_lugre = np.full_like(los_rx, np.nan)
    for index, rotation in enumerate(rotations):
        if rotation is not None:
            los_lugre[index] = rotation @ los_rx[index]
    df["rx_offboresight_spice_deg"] = np.degrees(np.arccos(np.clip(los_lugre[:, 2], -1.0, 1.0)))
    df["rx_azimuth_spice_deg"] = np.mod(np.degrees(np.arctan2(los_lugre[:, 1], los_lugre[:, 0])), 360.0)
    receiver_gain(df, args.rx_pattern)

    z_body, _ = normalize(-sat)
    sat_to_sun, _ = normalize(sun - sat)
    x_projection = sat_to_sun - np.sum(sat_to_sun * z_body, axis=1)[:, None] * z_body
    x_body, projection_norm = normalize(x_projection)
    y_body, _ = normalize(np.cross(z_body, x_body))
    df["tx_yaw_singularity_angle_deg"] = np.degrees(np.arcsin(np.clip(projection_norm, 0.0, 1.0)))
    df["tx_in_earth_eclipse"] = earth_eclipse(sat, sun)
    df["tx_yaw_quality"] = "nominal_yaw_steering"
    df.loc[df["tx_yaw_singularity_angle_deg"] < 5.0, "tx_yaw_quality"] = "noon_midnight_turn_risk"
    df.loc[df["tx_in_earth_eclipse"], "tx_yaw_quality"] = "eclipse_actual_yaw_unmodeled"

    tx_x = np.sum(los_tx * x_body, axis=1)
    tx_y = np.sum(los_tx * y_body, axis=1)
    tx_z = np.sum(los_tx * z_body, axis=1)
    df["tx_theta_body_deg"] = np.degrees(np.arccos(np.clip(tx_z, -1.0, 1.0)))
    df["tx_phi_body_deg"] = np.mod(np.degrees(np.arctan2(tx_y, tx_x)), 360.0)
    sat_ids = df["system"].astype(str) + df["svid"].astype(int).astype(str).str.zfill(2)
    orbex_theta, orbex_phi, orbex_gap_s, orbex_status = orbex_body_angles(
        df["rx_utc"], sat_ids, los_tx, orbex_dir=DEFAULT_ORBEX_DIR
    )
    orbex_valid = np.isfinite(orbex_theta) & np.isfinite(orbex_phi)
    df.loc[orbex_valid, "tx_theta_body_deg"] = orbex_theta[orbex_valid]
    df.loc[orbex_valid, "tx_phi_body_deg"] = orbex_phi[orbex_valid]
    df["tx_orbex_nearest_gap_s"] = orbex_gap_s
    df["tx_attitude_source"] = np.where(
        orbex_valid, orbex_status, "nominal_yaw_steering_fallback"
    )
    df.loc[orbex_valid, "tx_yaw_quality"] = "CODE_MGEX_final_ORBEX_30s"
    df["tx_yaw_frame_convention"] = "z=nadir;x=projected_sun;y=z_cross_x;right_handed"
    df["tx_theta_vs_previous_offboresight_diff_deg"] = df["tx_theta_body_deg"] - df["tx_offboresight_deg"]

    patterns_by_signal: dict[str, dict[int, dict]] = {
        "GPS_L1": parse_iirm_2d(
            args.gps_antenna_dir / "AppBAntennaPanelPatterns.pptx"
        ),
        "GPS_L5": {},
    }
    patterns_by_signal["GPS_L1"].update(parse_iiia_2d(args.gps_antenna_dir, "L1"))
    patterns_by_signal["GPS_L5"].update(parse_iiia_2d(args.gps_antenna_dir, "L5"))
    iif_patterns = parse_iif_patterns(args.gps_antenna_dir)
    for signal in ("GPS_L1", "GPS_L5"):
        patterns_by_signal[signal].update(iif_patterns[signal])

    # Public GPS III directivity is available through SVN78.  Later vehicles
    # remain calculable with an explicitly labelled same-block median template.
    for signal in ("GPS_L1", "GPS_L5"):
        iii = {
            svn: pattern
            for svn, pattern in patterns_by_signal[signal].items()
            if 74 <= svn <= 78
        }
        theta = next(iter(iii.values()))["theta"]
        phi = next(iter(iii.values()))["phi"]
        median_gain = np.nanmedian(
            np.stack([pattern["gain"] for pattern in iii.values()]), axis=0
        )
        for svn in mapping.loc[
            mapping["antenna_type"].eq("BLOCK IIIA"), "svn"
        ].unique():
            if int(svn) not in patterns_by_signal[signal]:
                patterns_by_signal[signal][int(svn)] = {
                    "theta": theta,
                    "phi": phi,
                    "gain": median_gain,
                    "source": f"GPS III SVN74-78 median {signal} 2D proxy",
                    "block_pattern_family": "GPS_IIIA_same_block_median_proxy",
                }
    df["tx_gain_2d_db"] = np.nan
    df["tx_eirp_2d_dbw"] = np.nan
    df["tx_pattern_source"] = ""
    df["tx_pattern_family"] = "unavailable"
    df["tx_pattern_coverage"] = "unavailable"
    for signal, patterns in patterns_by_signal.items():
        for svn, pattern in patterns.items():
            mask = df["signal_name"].eq(signal) & df["svn"].eq(svn)
            df.loc[mask, "tx_gain_2d_db"] = bilinear_periodic(
                pattern,
                df.loc[mask, "tx_theta_body_deg"].to_numpy(float),
                df.loc[mask, "tx_phi_body_deg"].to_numpy(float),
            )
            df.loc[mask, "tx_pattern_source"] = pattern["source"]
            df.loc[mask, "tx_pattern_family"] = pattern.get(
                "block_pattern_family", "GPS_pattern_family_unspecified"
            )
            df.loc[mask, "tx_pattern_coverage"] = pattern.get(
                "pattern_coverage", "satellite_specific_full_2D"
            )

    df["tx_power_dbw"] = df["antenna_type"].map(TX_POWER_DBW)
    gps_supported = df["signal_name"].isin(["GPS_L1", "GPS_L5"])
    df.loc[gps_supported, "tx_eirp_2d_dbw"] = (
        df.loc[gps_supported, "tx_power_dbw"]
        + df.loc[gps_supported, "tx_gain_2d_db"]
    )
    galileo_grap_eirp(df, args.grap)
    df["tx_phi_alignment_quality"] = "unsupported"
    df.loc[df["antenna_type"].eq("BLOCK IIR-M"), "tx_phi_alignment_quality"] = (
        "A_confirmed_phi0_plusX_positive_toward_plusY"
    )
    df.loc[df["antenna_type"].eq("BLOCK IIIA"), "tx_phi_alignment_quality"] = (
        "B_spherical_phi_defined_but_pattern_to_body_installation_not_explicit"
    )
    df.loc[df["antenna_type"].eq("BLOCK IIF"), "tx_phi_alignment_quality"] = (
        "B_IIF_ANGLE_CUT_coordinates_used_nominal_yaw_installation_not_explicit"
    )
    df.loc[df["signal_name"].isin(["GAL_E1", "GAL_E5a"]), "tx_phi_alignment_quality"] = (
        "B_Galileo_GRAP_spherical_phi_nominal_yaw_installation_not_explicit"
    )
    lobe_counts = add_gnss_ssv_lobe_labels(df, patterns_by_signal)
    df["rx_pattern_status"] = "figure3_outer_envelope_not_full_azimuth_pattern"
    df["system_noise_temperature_k"] = df["signal_name"].map(RX_TSYS_K)
    direct_supported = df["tx_eirp_2d_dbw"].notna()
    df["implementation_loss_assumed_db"] = np.where(direct_supported, 0.0, np.nan)
    df["other_unmodeled_loss_assumed_db"] = np.where(direct_supported, 0.0, np.nan)
    df["cn0_reference_trajectory_2d_dbhz"] = (
        df["tx_eirp_2d_dbw"] + df["rx_gain_envelope_dbic"]
        - df["fspl_db"] + K_DB - 10.0 * np.log10(df["system_noise_temperature_k"])
        - df["implementation_loss_assumed_db"] - df["other_unmodeled_loss_assumed_db"]
    )
    nominal = (
        direct_supported
        & df["rx_gain_envelope_dbic"].notna() & df["rx_attitude_status"].eq("spice_attitude_available")
        & df["tx_yaw_quality"].eq("nominal_yaw_steering")
    )
    quality_a = nominal & df["tx_phi_alignment_quality"].str.startswith("A_")
    quality_b = nominal & df["tx_phi_alignment_quality"].str.startswith("B_")
    computed = df["cn0_reference_trajectory_2d_dbhz"].notna()
    df["cn0_2d_model_status"] = "unsupported_missing_required_term"
    df.loc[computed, "cn0_2d_model_status"] = "computed_with_flagged_nominal_yaw_limit"
    df.loc[quality_a, "cn0_2d_model_status"] = "quality_A_nominal_yaw_spice_attitude_2d_gtx_rx_envelope"
    df.loc[quality_b, "cn0_2d_model_status"] = "quality_B_nominal_yaw_spice_attitude_2d_gtx_rx_envelope"

    metrics = summary_metrics(df, "cn0_reference_trajectory_2d_dbhz")
    status = df.groupby(["cn0_2d_model_status", "rx_attitude_status", "tx_yaw_quality"], dropna=False).size().reset_index(name="rows")
    output = df.copy()
    output["rx_utc"] = output["rx_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    output.to_csv(args.output_dir / "cn0_reference_trajectory_attitude_2d_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "cn0_reference_trajectory_attitude_2d_metrics.csv", index=False)
    status.to_csv(args.output_dir / "cn0_reference_trajectory_attitude_2d_status.csv", index=False)
    lobe_counts.to_csv(args.output_dir / "gnss_ssv_lobe_classification_counts.csv", index=False)
    lobe_counts.to_csv(args.output_dir / "gps_ssv_lobe_classification_counts.csv", index=False)
    print(
        f"rows={len(df)} computed={int(computed.sum())} "
        f"quality_A={int(quality_a.sum())} quality_B={int(quality_b.sum())}"
    )
    print(f"rx_attitude_available={int(df['rx_attitude_status'].eq('spice_attitude_available').sum())}")
    print(f"theta_consistency_p95_deg={np.nanpercentile(np.abs(df['tx_theta_vs_previous_offboresight_diff_deg']),95):.6g}")
    overall = metrics[(metrics["group_type"] == "overall")].iloc[0]
    print(f"median_flight_minus_model_db={overall['median_flight_minus_model_db']:.3f}")


if __name__ == "__main__":
    main()
