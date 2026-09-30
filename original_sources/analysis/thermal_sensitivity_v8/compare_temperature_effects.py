"""Temperature sensitivity against saved C/N0 predictions; no model retraining.

Figure-derived operation statistics are not synchronized thermal telemetry.
The HGA/LNA high-gain approximation is used as a relative sensitivity proxy, not a
replacement absolute calibration. Original inputs and manuscript are read-only.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"D:\月球导航\table\algorithm\cn0_trend_residual_tuned_no_leakage\cn0_trend_residual_tuned_no_leakage_predictions.csv")
TEMP = ROOT / "alex_temperature_inputs/fig11_operation_temperature_long.csv"
RELATIVE = ROOT / "alex_temperature_inputs/alex_relative_thermal_sensitivity.csv"
FINAL = ROOT.parent / "phase_compensation_attribution_v4/attrib_rows.csv"
BANDS = {"GPS_L1": "L1_E1", "GAL_E1": "L1_E1", "GPS_L5": "L5_E5a", "GAL_E5a": "L5_E5a"}
PARAMS = {"L1_E1": (182., .8, 41.5), "L5_E5a": (231., 1.3, 44.2)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, name):
    frame.to_csv(ROOT / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def rmse(x):
    a = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def surface_analysis(d, means, h_ref_c, l_ref_c):
    path = ROOT / "alex_temperature_inputs/surface_series.csv"
    if not path.exists():
        return {"status": "surface input not yet available"}
    raw = pd.read_csv(path)
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True)
    hga = raw[raw.sensor.eq("HGA")][["timestamp_utc", "temp_k", "source_segment"]].rename(columns={"temp_k": "hga_k"}).sort_values("timestamp_utc")
    lna = raw[raw.sensor.eq("LNA")][["timestamp_utc", "temp_k"]].rename(columns={"timestamp_utc": "lna_utc", "temp_k": "lna_k"}).sort_values("lna_utc")
    pair = pd.merge_asof(hga, lna, left_on="timestamp_utc", right_on="lna_utc", direction="nearest", tolerance=pd.Timedelta(seconds=60)).dropna(subset=["lna_k"])
    pair["thermal_id"] = np.arange(len(pair))
    pair["sensor_pair_separation_s"] = (pair.timestamp_utc - pair.lna_utc).dt.total_seconds().abs()
    for band, (t0, nf, gain) in PARAMS.items():
        proxy = pair.hga_k + (10**(nf/10)-1)*pair.lna_k
        tref = means[means.band.eq(band)].alex_reference_mean_across_16_operations_k.iloc[0]
        pair["cn0_shift_" + band + "_db"] = -10*np.log10(proxy/tref)
        for alpha in [0., .5, 1.]:
            scenario_t = t0 + alpha*(pair.hga_k-273.15-h_ref_c) + (10**(nf/10)-1)*(pair.lna_k-273.15-l_ref_c)
            assert (scenario_t > 0).all()
            pair[f"anchored_{band}_alpha_{alpha:g}_db"] = -10*np.log10(scenario_t/t0)
    save(pair, "surface_paired_temperature_sensitivity.csv")
    surface = d[d.mission_phase.eq("S")].copy()
    surface["minute_utc"] = pd.to_datetime(surface.minute_utc, utc=True)
    surface = surface.sort_values("minute_utc")
    # Nearest published point within five minutes; no interpolation and no
    # propagation over the 24-hour LNA gap or across the broken-axis panels.
    joined = pd.merge_asof(surface, pair.sort_values("timestamp_utc"), left_on="minute_utc", right_on="timestamp_utc", direction="nearest", tolerance=pd.Timedelta(minutes=5))
    joined["thermal_match"] = joined.thermal_id.notna()
    coverage = joined.groupby("op").agg(total_link_minutes=("op", "size"), matched_link_minutes=("thermal_match", "sum"), matched_thermal_points=("thermal_id", "nunique")).reset_index()
    coverage["coverage_fraction"] = coverage.matched_link_minutes/coverage.total_link_minutes
    save(coverage, "surface_temperature_match_coverage.csv")
    g = joined[joined.thermal_match].copy()
    g["thermal_shift_db"] = np.where(g.band.eq("L1_E1"), g.cn0_shift_L1_E1_db, g.cn0_shift_L5_E5a_db)
    g["time_match_abs_seconds"] = (g.minute_utc-g.timestamp_utc).dt.total_seconds().abs()
    cols = ["op", "signal_name", "svid", "minute_utc", "evaluation_split", "thermal_id", "timestamp_utc", "hga_k", "lna_k", "thermal_shift_db", "residual_baseline_db", "residual_after_beta_db", "hgb_correction_db", "residual_final_db", "time_match_abs_seconds"]
    save(g[cols], "surface_temperature_matched_residuals.csv")
    coupling_records = []
    for (op, band), sub in g.groupby(["op", "band"]):
        for alpha in [0., .5, 1.]:
            v = sub[f"anchored_{band}_alpha_{alpha:g}_db"]
            coupling_records.append(dict(op=op, band=band, hga_coupling_k_per_k=alpha,
                cn0_shift_min_db=v.min(), cn0_shift_max_db=v.max(), within_operation_span_db=v.max()-v.min(),
                interpretation="assumed incremental coupling; not measured antenna efficiency or calibrated system noise"))
    save(pd.DataFrame(coupling_records), "surface_calibration_preserving_scenario_spans.csv")
    binned = g.groupby(["op", "signal_name", "thermal_id"], as_index=False).agg(
        timestamp_utc=("timestamp_utc", "first"), n_link_minutes=("op", "size"),
        thermal_shift_db=("thermal_shift_db", "first"),
        residual_after_beta_median_db=("residual_after_beta_db", "median"),
        hgb_correction_median_db=("hgb_correction_db", "median"),
        residual_final_median_db=("residual_final_db", "median"))
    save(binned, "surface_temperature_binned_residuals.csv")
    records = []
    for (op, sig), sub in binned.groupby(["op", "signal_name"]):
        sub = sub.sort_values("timestamp_utc")
        time = (pd.to_datetime(sub.timestamp_utc, utc=True)-pd.to_datetime(sub.timestamp_utc, utc=True).min()).dt.total_seconds().to_numpy()/3600.
        design = np.column_stack([np.ones(len(time)), time])
        thermal_detrended = sub.thermal_shift_db.to_numpy()-design@np.linalg.lstsq(design, sub.thermal_shift_db.to_numpy(), rcond=None)[0]
        hgb_detrended = sub.hgb_correction_median_db.to_numpy()-design@np.linalg.lstsq(design, sub.hgb_correction_median_db.to_numpy(), rcond=None)[0]
        residual_detrended = sub.residual_after_beta_median_db.to_numpy()-design@np.linalg.lstsq(design, sub.residual_after_beta_median_db.to_numpy(), rcond=None)[0]
        span_db = sub.thermal_shift_db.max()-sub.thermal_shift_db.min()
        records.append(dict(op=op, signal_name=sig, n_thermal_points=len(sub),
            thermal_shift_min_db=sub.thermal_shift_db.min(), thermal_shift_max_db=sub.thermal_shift_db.max(),
            thermal_within_operation_span_db=span_db,
            pearson_r_with_remaining_residual=correlation(sub.thermal_shift_db, sub.residual_after_beta_median_db),
            pearson_r_with_hgb_correction=correlation(sub.thermal_shift_db, sub.hgb_correction_median_db),
            pearson_r_with_final_residual=correlation(sub.thermal_shift_db, sub.residual_final_median_db),
            pearson_r_after_linear_time_detrending_with_hgb=correlation(thermal_detrended, hgb_detrended) if len(sub)>=8 else np.nan,
            pearson_r_after_linear_time_detrending_with_remaining_residual=correlation(thermal_detrended, residual_detrended) if len(sub)>=8 else np.nan,
            interpretation_status="too_few_temperature_points" if len(sub)<8 else ("small_thermal_span_sensitive_to_digitization" if span_db<.05 else "exploratory_temporal_association"),
            note=("descriptive approximately 10-minute figure-point bins; changing satellite mixture and serial correlation; "
                  + ("combined OP77_0 and OP77_1 supported windows, not a continuous operation-wide trace" if op == "OP77"
                     else "partial temperature coverage" if op in ["OP74", "OP76"]
                     else "coverage defined by the saved eligible model samples and available signal-specific temperature matches"))))
    save(pd.DataFrame(records), "surface_temperature_operation_summary.csv")
    # Registration sensitivity: shift the figure time by +/- one minute and
    # assess changed match coverage and thermal shifts for common model rows.
    tests = []
    for seconds in [-60, 0, 60]:
        shifted = pair.copy()
        shifted["timestamp_utc"] += pd.Timedelta(seconds=seconds)
        j = pd.merge_asof(surface, shifted.sort_values("timestamp_utc"), left_on="minute_utc", right_on="timestamp_utc", direction="nearest", tolerance=pd.Timedelta(minutes=5))
        j = j[j.thermal_id.notna()]
        vals = np.where(j.band.eq("L1_E1"), j.cn0_shift_L1_E1_db, j.cn0_shift_L5_E5a_db)
        tests.append(dict(time_shift_seconds=seconds, matched_link_minutes=len(j), mean_thermal_shift_db=float(np.mean(vals))))
    save(pd.DataFrame(tests), "surface_time_registration_sensitivity.csv")
    return dict(status="completed", figure_points=len(raw), synchronized_hga_lna_pairs=len(pair),
                max_sensor_pair_separation_s=float(pair.sensor_pair_separation_s.max()),
                matched_link_minutes=len(g), max_model_temperature_time_separation_s=float(g.time_match_abs_seconds.max()),
                operations=coverage.to_dict("records"), interpolation="none", source_sha256=sha(path))


def main():
    hashes = {str(p): sha(p) for p in [SOURCE, TEMP, RELATIVE, FINAL]}
    raw = pd.read_csv(SOURCE)
    mask = raw.trend_training_eligible.astype(str).str.lower().eq("true")
    mask &= np.isfinite(raw[["cn0_trend_target_dbhz", "cn0_physics_trend_dbhz", "cn0_physics_ai_eval_dbhz"]]).all(axis=1)
    d = raw.loc[mask].copy()
    d["band"] = d.signal_name.map(BANDS)
    d["residual_baseline_db"] = d.cn0_trend_target_dbhz - d.cn0_physics_trend_dbhz
    d["residual_after_beta_db"] = d.residual_baseline_db - d.signal_beta_train_db
    # Saved validation eval belongs to model selection. Use the existing exact
    # final-refit split-isolated reconstruction for every operation instead.
    keys = ["op", "signal_name", "svid", "source_bin_gps_seconds"]
    exact = pd.read_csv(FINAL, usecols=keys+["reference", "beta_db", "delta_eval_db"])
    exact = exact[exact.reference.eq("empirical_fit32")].drop(columns="reference")
    assert not exact.duplicated(keys).any()
    d = d.merge(exact, on=keys, how="left", validate="one_to_one")
    assert d.delta_eval_db.notna().all()
    assert np.allclose(d.beta_db, d.signal_beta_train_db, rtol=0, atol=1e-12)
    d["hgb_correction_db"] = d.delta_eval_db
    d["residual_final_db"] = d.residual_after_beta_db-d.hgb_correction_db

    temps = pd.read_csv(TEMP)
    rel = pd.read_csv(RELATIVE)
    rel["op"] = rel.operation.str.replace(r"_\d+$", "", regex=True)
    means = rel[rel.scenario.eq("mean")].copy()
    span = rel.groupby(["op", "band"]).delta_cn0_alex_relative_reference_db.agg(["min", "max"])
    span["thermal_envelope_width_db"] = span["max"] - span["min"]
    op_records = []
    for (op, sig, band, phase), g in d.groupby(["op", "signal_name", "band", "mission_phase"]):
        v = means[(means.op == op) & (means.band == band)]
        rec = dict(op=op, signal_name=sig, band=band, phase=phase, n_link_minutes=len(g),
                   evaluation_splits=";".join(sorted(g.evaluation_split.unique())),
                   residual_baseline_median_db=g.residual_baseline_db.median(),
                   residual_after_beta_median_db=g.residual_after_beta_db.median(),
                   hgb_correction_median_db=g.hgb_correction_db.median(),
                   hgb_correction_p95_p05_db=g.hgb_correction_db.quantile(.95)-g.hgb_correction_db.quantile(.05),
                   residual_final_median_db=g.residual_final_db.median(),
                   baseline_rmse_dbhz=rmse(g.residual_baseline_db),
                   temperature_coverage="fig11_operation_statistics" if len(v) else "not_in_fig11")
        if len(v):
            shift = float(v.delta_cn0_alex_relative_reference_db.iloc[0])
            rec.update(thermal_mean_shift_db=shift,
                       thermal_min_shift_db=span.loc[(op, band), "min"],
                       thermal_max_shift_db=span.loc[(op, band), "max"],
                       thermal_envelope_width_db=span.loc[(op, band), "thermal_envelope_width_db"],
                       baseline_with_relative_thermal_shift_rmse_dbhz=rmse(g.residual_baseline_db-shift))
        op_records.append(rec)
    op_summary = pd.DataFrame(op_records)
    save(op_summary, "operation_signal_temperature_residual_comparison.csv")

    joined = d.merge(means[["op", "band", "delta_cn0_alex_relative_reference_db"]], on=["op", "band"], validate="many_to_one", how="inner")
    checks = []
    for label, g in [("all_matched_descriptive", joined), ("matched_heldout_operations", joined[joined.evaluation_split.eq("external_holdout")])]:
        if g.empty:
            continue
        shift = g.delta_cn0_alex_relative_reference_db
        checks.append(dict(scope=label, n=len(g), operations=";".join(sorted(g.op.unique())),
                           baseline_rmse_dbhz=rmse(g.residual_baseline_db),
                           baseline_with_relative_thermal_mean_rmse_dbhz=rmse(g.residual_baseline_db-shift),
                           note="noise-only sensitivity; original 16-operation mean temperature reference; not a refitted-model evaluation"))
    save(pd.DataFrame(checks), "flight_noise_only_sensitivity_metrics.csv")

    # Equal weight per operation, separately per signal. These are descriptive
    # correlations, not estimates based on thousands of independent temperatures.
    corrs = []
    for sig, g in op_summary.dropna(subset=["thermal_mean_shift_db"]).groupby("signal_name"):
        for target in ["residual_baseline_median_db", "residual_after_beta_median_db", "hgb_correction_median_db", "residual_final_median_db"]:
            corrs.append(dict(signal_name=sig, target=target, n_operations=len(g),
                              pearson_r=correlation(g.thermal_mean_shift_db, g[target]),
                              spearman_r=correlation(g.thermal_mean_shift_db.rank(), g[target].rank()),
                              status="descriptive; not causal; phase and geometry confounding not removed"))
    save(pd.DataFrame(corrs), "flight_operation_level_correlations.csv")

    # Alternative, calibration-preserving incremental scenarios. alpha is an
    # illustrative HGA physical-temperature coupling in K/K, NOT measured efficiency.
    wide = temps.pivot(index="operation_model_label", columns="component", values=["minimum_c", "mean_c", "maximum_c"])
    h_ref = float(wide[("mean_c", "HGA")].mean())
    l_ref = float(wide[("mean_c", "LNA")].mean())
    anchored = []
    for op, row in wide.iterrows():
        for band, (t0, nf, gain) in PARAMS.items():
            a = 10 ** (nf / 10) - 1
            for alpha in [0., .5, 1.]:
                for scenario in ["minimum", "mean", "maximum"]:
                    dh = row[(scenario + "_c", "HGA")] - h_ref
                    dl = row[(scenario + "_c", "LNA")] - l_ref
                    dt = alpha * dh + a * dl
                    t = t0 + dt
                    assert t > 0
                    anchored.append(dict(op=op, band=band, scenario=scenario,
                        hga_coupling_k_per_k=alpha, reference_hga_c=h_ref, reference_lna_c=l_ref,
                        reference_system_noise_k=t0, scenario_system_noise_k=t,
                        cn0_change_db=-10*np.log10(t/t0),
                        assumption="reference-anchored incremental sensitivity; coupling and LNA temperature slope not flight-calibrated"))
    anchored = pd.DataFrame(anchored)
    save(anchored, "calibration_preserving_thermal_scenarios.csv")
    surface_summary = surface_analysis(d, means, h_ref, l_ref)

    summary = {
        "primary_input_rows": len(raw), "eligible_rows": len(d),
        "fig11_operations": temps.operation.nunique(),
        "matched_operations": joined.op.nunique(), "matched_link_minutes": len(joined),
        "missing_operations": sorted(set(d.op)-set(joined.op)),
        "thermal_proxy_means_db_by_band": means.groupby("band").delta_cn0_alex_relative_reference_db.agg(["min", "max"]).to_dict("index"),
        "thermal_proxy_scenario_db_by_band": rel.groupby("band").delta_cn0_alex_relative_reference_db.agg(["min", "max"]).to_dict("index"),
        "absolute_proxy_replacement_mean_db_by_band_NOT_temperature_variation": means.groupby("band").absolute_proxy_replacement_delta_db_not_recommended.agg(["min", "max"]).to_dict("index"),
        "noise_only_metrics": checks,
        "source_hashes": hashes,
        "sources_unchanged": all(sha(Path(p)) == digest for p, digest in hashes.items()),
        "baseline_prediction_and_residual_model_unchanged": True,
        "hgb_series": "exact existing final-refit split-isolated delta_eval_db; not the mixed saved selection/evaluation output",
        "correlations_include_refit_data_and_are_descriptive_not_independent_prediction_tests": True,
        "no_target_operation_CNO_used_to_compute_temperature_proxy": True,
        "reference_uses_all_16_thermal_operation_means_for_descriptive_sensitivity_only": True,
        "no_raw_thermal_time_series_or_within_operation_correlation_from_fig11": True,
        "surface_analysis": surface_summary,
    }
    assert summary["sources_unchanged"]
    (ROOT / "temperature_comparison_qa.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(op_summary[op_summary.op.isin(["OP2", "OP21", "OP27", "OP37"])][["op", "signal_name", "thermal_mean_shift_db", "thermal_envelope_width_db", "residual_after_beta_median_db", "hgb_correction_median_db"]].to_string(index=False))


if __name__ == "__main__":
    main()
