"""Paired retrospective temperature refits; original models and inputs read-only."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(r"D:\月球导航")
OUT = Path(__file__).resolve().parent
THERMAL = OUT.parent / "thermal_sensitivity_v8"
sys.path.insert(0, str(ROOT / "runtime_cache/python_deps"))
sys.path.insert(0, str(ROOT / "script"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import train_cn0_single_global_residual_1min_op74_holdout as engine

engine.TREND_WINDOW_MINUTES = 9
MODEL = ROOT / "table/algorithm/cn0_trend_residual_tuned_no_leakage/cn0_trend_residual_tuned_no_leakage.joblib"
PRED = MODEL.with_name("cn0_trend_residual_tuned_no_leakage_predictions.csv")
CACHE = ROOT / "table/algorithm/cn0_ai_residual_story/final_split_feature_frame.joblib"
FIG11 = THERMAL / "alex_temperature_inputs/fig11_operation_temperature_long.csv"
FIG12 = THERMAL / "alex_temperature_inputs/surface_series.csv"
KEYS = ["minute_utc", "op", "signal_name", "svid"]
SCENARIOS = ["fixed", "alex_relative", "alpha_0", "alpha_0.5", "alpha_1", "descriptor_only"]
BASE = "cn0_physics_trend_dbhz"
RAW = "cn0_constellation_physics_proxy_dbhz"
DERIVED = ["cn0_constellation_direct_available_dbhz", "cn0_reference_trajectory_2d_dbhz"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, filename):
    frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig", float_format="%.15g")


def normalize(frame):
    frame = frame.copy()
    frame.minute_utc = pd.to_datetime(frame.minute_utc, utc=True)
    frame.svid = frame.svid.astype(int)
    assert not frame.duplicated(KEYS).any()
    return frame


def smooth(frame, values, split=False):
    temp = frame.copy()
    temp["_value"] = np.asarray(values, float)
    keys = ["op", "signal_name", "svid"]
    if "thermal_segment" in temp:
        keys.append("thermal_segment")
    if split:
        keys.insert(0, "evaluation_split")
    result = pd.Series(np.nan, index=temp.index)
    for _, group in temp.groupby(keys, dropna=False, sort=False):
        result.loc[group.index] = engine._smooth_continuous_segments(group, "_value")
    assert result.notna().all()
    return result.to_numpy(float)


def attach_temperatures(frame):
    long = pd.read_csv(FIG11)
    op = long.pivot(index="operation_model_label", columns="component", values="mean_k")
    f = frame.copy()
    f["hga_k"] = f.op.map(op.HGA)
    f["lna_k"] = f.op.map(op.LNA)
    f["thermal_id"] = "flight_" + f.op
    f["thermal_segment"] = f.thermal_id
    f["thermal_source"] = "figure11_operation_mean"
    points = pd.read_csv(FIG12)
    points.timestamp_utc = pd.to_datetime(points.timestamp_utc, utc=True)
    h = points[points.sensor.eq("HGA")][["timestamp_utc", "temp_k"]].rename(columns={"temp_k": "hga_k"}).sort_values("timestamp_utc")
    l = points[points.sensor.eq("LNA")][["timestamp_utc", "temp_k"]].rename(columns={"timestamp_utc": "lna_utc", "temp_k": "lna_k"}).sort_values("lna_utc")
    pairs = pd.merge_asof(h, l, left_on="timestamp_utc", right_on="lna_utc", direction="nearest", tolerance=pd.Timedelta(seconds=60)).dropna().reset_index(drop=True)
    pairs["thermal_id"] = "surface_" + pairs.index.astype(str)
    pairs["thermal_segment"] = "surface_segment_" + pairs.timestamp_utc.diff().gt(pd.Timedelta(minutes=15)).cumsum().astype(str)
    surface = f[f.mission_phase.eq("S")].copy()
    surface["_original_index"] = surface.index
    match = pd.merge_asof(surface.drop(columns=["hga_k", "lna_k", "thermal_id", "thermal_segment"]).sort_values("minute_utc"), pairs, left_on="minute_utc", right_on="timestamp_utc", direction="nearest", tolerance=pd.Timedelta(minutes=5))
    match = match.set_index("_original_index")
    for c in ["hga_k", "lna_k", "thermal_id", "thermal_segment"]:
        f.loc[match.index, c] = match[c]
    f.loc[match.index, "thermal_source"] = "figure12_nearest_paired_point"
    f["thermal_supported"] = f.hga_k.notna() & f.lna_k.notna()
    f["band"] = np.where(f.signal_name.isin(["GPS_L1", "GAL_E1"]), "L1_E1", "L5_E5a")
    f["t0_k"] = np.where(f.band.eq("L1_E1"), 182., 231.)
    f["lna_slope"] = 10**(np.where(f.band.eq("L1_E1"), .8, 1.3)/10)-1
    return f


def prepare_scenario(frame, scenario, fit_mask):
    f = frame.copy()
    unique = f.loc[fit_mask].drop_duplicates("thermal_id")
    h_ref, l_ref = float(unique.hga_k.mean()), float(unique.lna_k.mean())
    if scenario == "fixed":
        t = f.t0_k.to_numpy()
    elif scenario in ["alex_relative", "descriptor_only"]:
        t = f.t0_k * (f.hga_k+f.lna_slope*f.lna_k)/(h_ref+f.lna_slope*l_ref)
    else:
        alpha = float(scenario.split("_")[1])
        t = f.t0_k + alpha*(f.hga_k-h_ref) + f.lna_slope*(f.lna_k-l_ref)
    assert np.isfinite(t).all() and np.all(np.asarray(t)>0)
    f["thermal_shift_db"] = -10*np.log10(t/f.t0_k)
    f["system_noise_temperature_k"] = np.asarray(t)
    if scenario == "descriptor_only":
        f["thermal_shift_db"] = 0.
    for col in DERIVED:
        f[col] += f.thermal_shift_db
    f[RAW] += f.thermal_shift_db
    f[BASE] = smooth(f, f[RAW])
    f["residual_trend_target_db"] = f.cn0_observed_trend_dbhz-f[BASE]
    artifact_features = joblib.load(MODEL)["model_features"]
    unchanged = [c for c in artifact_features if c not in DERIVED+[BASE, "system_noise_temperature_k"]]
    assert f[unchanged].equals(frame[unchanged])
    ref = dict(hga_reference_k=h_ref, lna_reference_k=l_ref, n_fitting_temperature_states=len(unique),
               reference_scope="unique temperature states from fitting rows only", scenario=scenario)
    return f, ref


def fit_model(frame, fit_mask, template, features):
    assert fit_mask.any()
    train = frame.loc[fit_mask]
    beta = train.groupby("signal_name").residual_trend_target_db.median().to_dict()
    assert len(beta) == 4
    model = clone(template)
    model.fit(train[features], train.residual_trend_target_db-train.signal_name.map(beta))
    pred = frame[KEYS+["evaluation_split", "mission_phase", "eligible", "thermal_segment"]].copy() if "thermal_segment" in frame else frame[KEYS+["evaluation_split", "mission_phase", "eligible"]].copy()
    pred["observed_dbhz"] = frame.cn0_observed_trend_dbhz
    pred["baseline_dbhz"] = frame[BASE]
    pred["beta_db"] = frame.signal_name.map(beta)
    pred["delta_raw_db"] = model.predict(frame[features])
    pred["correction_filtered_db"] = smooth(frame, pred.beta_db+pred.delta_raw_db, split=True)
    pred["delta_filtered_db"] = pred.correction_filtered_db-pred.beta_db
    pred["persistent_only_dbhz"] = pred.baseline_dbhz+pred.beta_db
    pred["final_dbhz"] = pred.baseline_dbhz+pred.correction_filtered_db
    return model, beta, pred


def metric_rows(pred, mask, scope, protocol, scenario):
    d = pred.loc[mask]
    if not len(d):
        return []
    records = []
    subsets = [("pooled", d)] + [(op, g) for op, g in d.groupby("op")]
    for op, g in subsets:
        rec = dict(protocol=protocol, scenario=scenario, scope=scope, operation=op, n=len(g), phases=";".join(sorted(g.mission_phase.unique())))
        for name, col in [("baseline", "baseline_dbhz"), ("persistent", "persistent_only_dbhz"), ("final", "final_dbhz")]:
            error = g[col]-g.observed_dbhz
            rec[name+"_rmse_dbhz"] = float(np.sqrt(np.mean(error**2)))
            rec[name+"_bias_db"] = float(error.mean())
            rec[name+"_phase_weighted_rmse_dbhz"] = float(np.sqrt((error**2).groupby(g.mission_phase).mean().mean()))
        records.append(rec)
    return records


def main():
    inputs = [MODEL, PRED, CACHE, FIG11, FIG12, OUT / "PLAN.md"]
    hashes = {str(p): sha(p) for p in inputs}
    artifact = joblib.load(MODEL)
    features = artifact["model_features"]
    bins = normalize(joblib.load(CACHE))
    original = normalize(pd.read_csv(PRED))
    extra = [RAW, "ai_trend_residual_raw_pred_db", "cn0_physics_ai_eval_dbhz", "signal_beta_train_db", "source_bin_gps_seconds"]
    bins = bins.merge(original[KEYS+extra], on=KEYS, how="left", validate="one_to_one")
    assert len(bins) == 17439 and len(features) == 44
    bins["eligible"] = bins.trend_training_eligible.astype(str).str.lower().eq("true") & np.isfinite(bins.residual_trend_target_db)
    fit_full = bins.eligible & bins.evaluation_split.isin(["train", "validation"])
    assert fit_full.sum() == 9894 and bins.eligible.sum() == 15806
    print("Checking full-data model reproduction", flush=True)
    with threadpool_limits(limits=4):
        full_model, full_beta, full_pred = fit_model(bins, fit_full, artifact["model"], features)
    max_beta = max(abs(full_beta[k]-artifact["signal_beta_db"][k]) for k in full_beta)
    max_raw = float(np.max(np.abs(full_pred.beta_db+full_pred.delta_raw_db-bins.ai_trend_residual_raw_pred_db)))
    external = bins.evaluation_split.eq("external_holdout") & bins.eligible
    test = bins.evaluation_split.eq("test") & bins.eligible
    max_eval = float(np.max(np.abs(full_pred.loc[external|test, "final_dbhz"]-bins.loc[external|test, "cn0_physics_ai_eval_dbhz"])))
    print(json.dumps(dict(full_beta_max_error=max_beta, full_raw_max_error=max_raw, full_test_max_error=max_eval)), flush=True)
    assert max_beta < 1e-9 and max_raw < 1e-7 and max_eval < 1e-7, "Reproduction failed; stop before comparison"
    full_metrics = metric_rows(full_pred, external, "external_holdout", "full_reproduction", "fixed") + metric_rows(full_pred, test, "test", "full_reproduction", "fixed")
    save(pd.DataFrame(full_metrics), "original_model_reproduction.csv")
    all_rows = attach_temperatures(bins)
    save(all_rows.groupby(["op", "evaluation_split", "eligible"]).agg(n=("op", "size"), supported=("thermal_supported", "sum")).reset_index(), "temperature_coverage.csv")
    matched = all_rows.loc[all_rows.thermal_supported].copy().reset_index(drop=True)
    matched_fit = matched.eligible & matched.evaluation_split.isin(["train", "validation"])
    assert not matched.loc[matched_fit, "op"].isin(["OP2", "OP21", "OP27", "OP74"]).any()
    save(matched[KEYS+["evaluation_split", "eligible", "hga_k", "lna_k", "thermal_id", "thermal_segment", "thermal_source"]], "matched_temperature_inputs.csv")
    refs, betas, metrics, predictions = [], [], [], []
    for protocol in ["matched_main", "matched_op77_excluded"]:
        fit_mask = matched_fit & (matched.op.ne("OP77") if protocol.endswith("excluded") else True)
        for scenario in SCENARIOS:
            print(f"Fitting {protocol} / {scenario}: {fit_mask.sum()} samples", flush=True)
            scenario_frame, reference = prepare_scenario(matched, scenario, fit_mask)
            with threadpool_limits(limits=4):
                model, beta, pred = fit_model(scenario_frame, fit_mask, artifact["model"], features)
            reference.update(protocol=protocol, n_fitting_samples=int(fit_mask.sum()), min_system_noise_k=float(scenario_frame.system_noise_temperature_k.min()))
            refs.append(reference)
            betas.extend(dict(protocol=protocol, scenario=scenario, signal_name=k, beta_db=v) for k,v in beta.items())
            pred["protocol"], pred["scenario"] = protocol, scenario
            pred["thermal_shift_db"] = scenario_frame.thermal_shift_db
            predictions.append(pred)
            joblib.dump(dict(model=model, signal_beta_db=beta, model_features=features, reference=reference), OUT / f"model_{protocol}_{scenario}.joblib")
            if protocol == "matched_main":
                for split in ["test", "external_holdout"]:
                    metrics += metric_rows(pred, pred.eligible & pred.evaluation_split.eq(split), split, protocol, scenario)
            else:
                metrics += metric_rows(pred, pred.eligible & pred.op.eq("OP77"), "OP77_all_original_partitions", protocol, scenario)
                times = matched.loc[matched.op.eq("OP77"), "minute_utc"].drop_duplicates().sort_values()
                windows = times.diff().gt(pd.Timedelta(minutes=30)).cumsum()
                time_to_window = dict(zip(times, windows))
                for win in windows.unique():
                    win_mask = pred.op.eq("OP77") & pred.minute_utc.map(time_to_window).eq(win) & pred.eligible
                    metrics += metric_rows(pred, win_mask, f"OP77_window_{int(win)+1}", protocol, scenario)
    metric = pd.DataFrame(metrics)
    control = metric[metric.scenario.eq("fixed")][["protocol", "scope", "operation", "final_rmse_dbhz"]].rename(columns={"final_rmse_dbhz": "fixed_control_final_rmse_dbhz"})
    metric = metric.merge(control, on=["protocol", "scope", "operation"], validate="many_to_one")
    metric["change_from_fixed_control_dbhz"] = metric.final_rmse_dbhz-metric.fixed_control_final_rmse_dbhz
    save(metric, "paired_refit_metrics.csv")
    save(pd.DataFrame(refs), "fitting_temperature_references.csv")
    save(pd.DataFrame(betas), "refitted_beta.csv")
    save(pd.concat(predictions, ignore_index=True), "paired_refit_predictions.csv")
    qa = dict(input_hashes=hashes, sources_unchanged=all(sha(Path(p)) == h for p,h in hashes.items()), full_reproduction_max_beta_error=max_beta, full_reproduction_max_raw_error=max_raw, full_reproduction_max_eval_error=max_eval,
              matched_all_bins=len(matched), matched_eligible_bins=int(matched.eligible.sum()), matched_fitting_bins=int(matched_fit.sum()),
              matched_split_counts=matched.loc[matched.eligible].evaluation_split.value_counts().to_dict(),
              fixed_parameters=artifact["model"].named_steps["model"].get_params(),
              source_models_unmodified=True, scenario_selection="none; all five initial scenarios retained; descriptor_only added as an explanatory follow-up", main_temperature_reference="fitting-only unique thermal states", future_temperature_forecast_test=False)
    assert qa["sources_unchanged"]
    (OUT / "refit_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(metric[metric.operation.eq("pooled")][["protocol", "scenario", "scope", "n", "final_rmse_dbhz", "change_from_fixed_control_dbhz"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
