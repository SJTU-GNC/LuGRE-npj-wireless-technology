#!/usr/bin/env python3
"""Recreate saved model's physical minute anchors without fitting or prediction.

The authoritative training module implements interpolation and common geometry.
This script calls that module unchanged, replacing only its raw-TLM quality scan
with already-persisted quality fields. It preserves the prediction table intact
and appends physical inputs for post-hoc diagnostics, not new model training.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "runtime_cache" / "python_deps"))
sys.path.insert(0, str(ROOT / "script"))

import numpy as np
import pandas as pd
import train_cn0_trend_residual_tuned_no_leakage as wrapper

ENGINE = wrapper.engine
OUT = Path(__file__).resolve().parent / "prepared"
PRIMARY = wrapper.OUT_DIR / ENGINE.OUTPUT_PREDICTIONS_FILENAME
PHYSICAL = wrapper.SOURCE
KEYS = ["op", "system", "signal_name", "svid", "source_bin_gps_seconds"]
QUALITY = [
    "valid_seconds_in_bin", "coverage_fraction", "longest_gap_s",
    "first_valid_offset_s", "last_valid_offset_s", "preceding_gap_s",
    "reacquisition_flag",
]
STATUS = [
    "constellation_physics_proxy_status", "constellation_direct_status",
    "constellation_physics_no_observation_tx_fit", "tx_pattern_family",
    "tx_pattern_source", "tx_pattern_coverage", "tx_yaw_quality",
    "tx_phi_alignment_quality", "cn0_2d_model_status",
    "receiver_gain_model_status", "ionosphere_absorption_proxy_status",
    "gas_absorption_proxy_status", "galileo_grap_lookup_status",
    "galileo_grap_coordinate_mapping_status",
]
TOLERANCE = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name in KEYS[:-1]:
        frame[name] = frame[name].astype(str)
    frame[KEYS[-1]] = pd.to_numeric(frame[KEYS[-1]], errors="raise").astype("int64")
    return frame


def numeric_comparison(left: pd.Series, right: pd.Series) -> dict:
    a = pd.to_numeric(left, errors="coerce").to_numpy(float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(float)
    valid = np.isfinite(a) & np.isfinite(b)
    delta = a[valid] - b[valid]
    missing_mismatch = np.isfinite(a) != np.isfinite(b)
    mismatch = valid & ~np.isclose(a, b, rtol=0, atol=TOLERANCE, equal_nan=True)
    return {
        "n": len(a), "finite_pairs": int(valid.sum()),
        "missingness_mismatches": int(missing_mismatch.sum()),
        "value_mismatches": int(mismatch.sum()),
        "max_absolute_difference": float(np.max(np.abs(delta))) if len(delta) else None,
        "mean_difference": float(np.mean(delta)) if len(delta) else None,
        "pass": bool(not missing_mismatch.any() and not mismatch.any()),
    }


def budget_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Select equation by builder's documented branch, not by constellation alone."""
    out = frame.copy()
    direct = out["constellation_physics_proxy_status"].eq("band_specific_rowwise_2d_direct_budget")
    gal = out["constellation_physics_proxy_status"].eq("galileo_grap_azimuth_aggregated_fallback_budget")
    # For this source every row is direct. Galileo direct uses phi-specific EIRP,
    # whereas the alternate GRAP fallback uses azimuth-median EIRP and peak G_rx.
    out["budget_selected_tx_eirp_dbw"] = np.where(
        direct, out["tx_eirp_2d_dbw"],
        np.where(gal, out["tx_eirp_grap_azimuth_median_dbw"], np.nan),
    )
    out["budget_selected_rx_gain_dbic"] = np.where(
        direct, out["rx_gain_envelope_dbic"],
        np.where(gal, out["rx_peak_gain_dbic"], np.nan),
    )
    out["budget_noise_psd_dbw_hz"] = -228.6 + 10.0 * np.log10(out["system_noise_temperature_k"])
    out["budget_reconstructed_cn0_dbhz"] = (
        out["budget_selected_tx_eirp_dbw"] + out["budget_selected_rx_gain_dbic"]
        - out["fspl_db"] - out["budget_noise_psd_dbw_hz"]
        - out["implementation_loss_assumed_db"]
        - out["l_ion_abs_budget_db"] - out["l_gas_abs_budget_db"]
    )
    out["budget_closure_error_db"] = out["budget_reconstructed_cn0_dbhz"] - out[ENGINE.BASELINE]
    return out


def closure_summary(frame: pd.DataFrame) -> list[dict]:
    records = []
    for keys, group in frame.groupby(["signal_name", "constellation_physics_proxy_status"], dropna=False):
        stats = numeric_comparison(group["budget_reconstructed_cn0_dbhz"], group[ENGINE.BASELINE])
        records.append({"signal_name": keys[0], "baseline_branch": keys[1], **stats})
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [
        PRIMARY, PHYSICAL, wrapper.SELECTION_PATH, Path(wrapper.__file__),
        Path(ENGINE.__file__), Path(ENGINE.base_mod.__file__),
        Path(ENGINE.base_mod.source_mod.__file__),
        ROOT / "script" / "build_cn0_constellation_physics_baseline.py",
        ROOT / "script" / "build_cn0_attitude_2d_gain.py",
    ]
    before = {str(path): sha256(path) for path in source_paths}
    primary = canonical_keys(pd.read_csv(PRIMARY, dtype={"svid": str}, low_memory=False))
    raw = ENGINE.base_mod.load_data()
    raw["source_bin_gps_seconds"] = pd.to_numeric(raw["time_bin_gps_seconds"]).round().astype("int64")
    raw = canonical_keys(raw)
    if primary.duplicated(KEYS).any() or raw.duplicated(KEYS).any():
        raise RuntimeError("Duplicate native keys: investigate before aligning categorical provenance")
    original_features = ENGINE.base_mod.feature_columns(raw)
    extras = [
        "tx_eirp_2d_dbw", "noise_psd_dbw_hz", "other_unmodeled_loss_assumed_db",
        "cn0_reference_trajectory_2d_atm_proxy_dbhz",
        "cn0_reference_trajectory_2d_iono_proxy_dbhz",
        "rx_to_sat_earth_center_sep_deg", "rx_to_sat_moon_center_sep_deg",
    ]
    extras.extend(column for column in raw.columns if column.startswith(("tx_eirp_grap_", "baseline_2d_")))
    physical_features = list(dict.fromkeys(original_features + [name for name in extras if name in raw]))
    for name in physical_features:
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    quality = primary[KEYS + QUALITY].copy()
    original_quality_function = ENGINE.build_raw_quality
    try:
        ENGINE.build_raw_quality = lambda operations: quality.loc[quality.op.isin(operations)].copy()
        bins = canonical_keys(ENGINE.aggregate_one_minute(raw, physical_features))
    finally:
        ENGINE.build_raw_quality = original_quality_function
    key_check = primary[KEYS].merge(bins[KEYS], on=KEYS, how="outer", indicator=True, validate="one_to_one")
    if not key_check["_merge"].eq("both").all():
        raise RuntimeError(f"Unmatched alignment keys: {key_check['_merge'].value_counts().to_dict()}")

    # Check every overlap. Existing saved values win; none are silently replaced.
    shared = [name for name in bins if name in primary and name not in KEYS]
    check = primary.merge(bins[KEYS + shared], on=KEYS, how="left", suffixes=("", "__recreated"), validate="one_to_one")
    overlap_checks = {}
    for name in shared:
        if name.endswith("utc"):
            a = pd.to_datetime(check[name], utc=True)
            b = pd.to_datetime(check[name + "__recreated"], utc=True)
            # Pandas CSV datetime reparsing can differ sub-microsecond for the
            # float Unix median conversion. Retain and report nanosecond error.
            delta = (a - b).dt.total_seconds().abs()
            overlap_checks[name] = {
                "max_absolute_difference_seconds": float(delta.max()),
                "pass": bool(delta.le(1e-6).all()),
            }
        elif name == "mission_phase":
            overlap_checks[name] = {"pass": bool(check[name].astype(str).eq(check[name + "__recreated"].astype(str)).all())}
        else:
            overlap_checks[name] = numeric_comparison(check[name], check[name + "__recreated"])
    # Verify the saved smoothed physical baseline independently; no fitting or
    # model loading occurs. Physical input features themselves are NOT smoothed.
    smooth = ENGINE.smooth_predicted_residual(bins, ENGINE.BASELINE)
    smooth_frame = bins[KEYS].assign(_recreated_physics_trend=smooth)
    smooth_check = primary.merge(smooth_frame, on=KEYS, validate="one_to_one")
    overlap_checks[ENGINE.PHYSICS_TREND_COLUMN] = numeric_comparison(
        smooth_check[ENGINE.PHYSICS_TREND_COLUMN], smooth_check["_recreated_physics_trend"]
    )
    new_columns = [name for name in bins if name not in primary]
    merged = primary.merge(bins[KEYS + new_columns], on=KEYS, how="left", validate="one_to_one", sort=False)
    available_status = [name for name in STATUS if name in raw and name not in merged]
    merged = merged.merge(raw[KEYS + available_status], on=KEYS, how="left", validate="one_to_one", sort=False)
    merged = budget_components(merged)
    raw_budget = budget_components(raw)
    missing_selected = sorted(set(wrapper.selected_features) - set(merged.columns))
    if missing_selected:
        raise RuntimeError(f"Missing selected features: {missing_selected}")
    preserved = primary.equals(merged[primary.columns])
    closure = closure_summary(merged)
    after = {str(path): sha256(path) for path in source_paths}
    unchanged = before == after
    common_keys = ["op", "system", "svid", "source_bin_gps_seconds"]
    common_checks = {}
    for name in ENGINE.COMMON_GEOMETRY_COLUMNS:
        if name in merged:
            counts = merged.groupby(common_keys, dropna=False)[name].nunique(dropna=False)
            common_checks[name] = {"conflicting_satellite_bins": int(counts.gt(1).sum()), "pass": bool(counts.le(1).all())}
    passed = (
        len(merged) == 17439 and preserved and unchanged
        and all(value["pass"] for value in overlap_checks.values())
        and all(value["pass"] for value in closure)
        and all(value["pass"] for value in common_checks.values())
    )
    qa = {
        "status": "pass" if passed else "mismatch_identified",
        "rows": len(merged), "columns": len(merged.columns),
        "raw_physical_rows": len(raw), "primary_prediction_rows": len(primary),
        "primary_columns_preserved_exactly_in_memory": preserved,
        "primary_duplicate_keys": int(primary.duplicated(KEYS).sum()),
        "aligned_duplicate_keys": int(merged.duplicated(KEYS).sum()),
        "keys": KEYS,
        "alignment_method": {
            "implementation": str(Path(ENGINE.__file__)),
            "function": "aggregate_one_minute",
            "quality_source": "saved primary prediction quality fields; raw TLM not reparsed",
            "anchor": "source_bin_gps_seconds + 30 s - 18 s leap offset, from GPS epoch",
            "physical_values": "original engine band-specific linear interpolation to anchor; nearest boundary extrapolation via numpy.interp",
            "common_geometry": "original engine high-samples-coverage then nearest-anchor ray, broadcast across frequencies",
            "frequency_terms": "original engine recomputes FSPL and atmospheric proxies from common geometry, adjusts baseline by old-new losses",
            "target": "native bin median, never interpolated",
            "primary_physics_trend": "saved 9-point rolling median plus Savitzky-Golay polynomial order 2 within continuous links, independently verified",
            "status_fields": "unaltered native-row provenance, unique original row per native key; not inferred from interpolated geometry",
        },
        "value_tolerance_absolute_db": TOLERANCE,
        "overlap_checks": overlap_checks,
        "common_geometry_checks": common_checks,
        "physical_budget": {
            "equation": "C/N0 = selected_EIRP + selected_G_rx - FSPL - noise_PSD - implementation_loss - ion_budget_loss - gas_budget_loss",
            "noise_psd_definition": "-228.6 + 10*log10(system_noise_temperature_k), using source-builder rounded Boltzmann constant",
            "direct_branch": {"tx": "tx_eirp_2d_dbw", "rx": "rx_gain_envelope_dbic", "gps_eirp": "tx_power_dbw + tx_gain_2d_db", "galileo_eirp": "phi-specific two-dimensional GRAP EIRP, NOT azimuth median"},
            "galileo_fallback_branch": {"tx": "tx_eirp_grap_azimuth_median_dbw", "rx": "rx_peak_gain_dbic"},
            "legacy_fallback_branch": "proxy-calibrated baseline; no forced component reconstruction",
            "raw_source_closure": closure_summary(raw_budget),
            "aligned_minute_closure": closure,
            "branch_counts": raw["constellation_physics_proxy_status"].value_counts().to_dict(),
            "caution": "Additive closure is for the raw minute baseline, not smoothed cn0_physics_trend_dbhz. Rolling median is nonlinear; independently smoothed components need not sum.",
        },
        "column_groups": {
            "primary": list(primary.columns), "original_engine_features": original_features,
            "additional_numeric_features": [c for c in physical_features if c not in original_features],
            "selected_model_features": wrapper.selected_features,
            "native_status_provenance": available_status,
            "derived_budget_audit": [c for c in merged if c.startswith("budget_")],
        },
        "output_columns": list(merged.columns),
        "missingness_by_numeric_feature": {c: int(merged[c].isna().sum()) for c in physical_features},
        "source_hashes_before": before,
        "source_hashes_after": after,
        "all_original_sources_unchanged": unchanged,
        "no_model_training_or_prediction_executed": True,
    }
    output_path = OUT / "physical_minutes.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    qa["output_sha256"] = sha256(output_path)
    (OUT / "alignment_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": qa["status"], "rows": len(merged), "columns": len(merged.columns), "preserved": preserved, "unchanged": unchanged, "overlap_checks": overlap_checks, "budget_closure": closure}, indent=2))
    if not passed:
        raise RuntimeError("Alignment QA mismatch identified; inspect alignment_qa.json, no values were forced")


if __name__ == "__main__":
    main()
