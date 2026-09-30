"""Prespecified past-only residual refits; original sources remain read-only."""
from pathlib import Path
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ROOT = Path(r"D:\月球导航")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime_cache/python_deps"))
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

CACHE = ROOT / "table/algorithm/cn0_ai_residual_story/final_split_feature_frame.joblib"
PRED = ROOT / "table/algorithm/cn0_trend_residual_tuned_no_leakage/cn0_trend_residual_tuned_no_leakage_predictions.csv"
MODEL = PRED.with_name("cn0_trend_residual_tuned_no_leakage.joblib")
SELECT = ROOT / "table/algorithm/cn0_trend_residual_tuning_no_leakage/selected_model.json"
KEYS = ["minute_utc", "op", "signal_name", "svid"]
TARGET = "cn0_dbhz_mean"
RAW = "cn0_constellation_physics_proxy_dbhz"
BASE = "cn0_physics_trend_dbhz"
SIGNALS = {"GAL_E1", "GAL_E5a", "GPS_L1", "GPS_L5"}
CUTOFFS = ["OP2", "OP22", "OP37", "OP38"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def flags(series):
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def normalize(frame):
    frame = frame.copy()
    frame.minute_utc = pd.to_datetime(frame.minute_utc, utc=True)
    frame.svid = frame.svid.astype(int)
    assert not frame.duplicated(KEYS).any()
    return frame


def smooth(frame, values, observed=False):
    work = frame[KEYS + ["role", "quality"]].copy()
    work["value"] = np.asarray(values, float)
    output = pd.Series(np.nan, index=work.index, dtype=float)
    for _, group in work.groupby(["role", "op", "signal_name", "svid"], sort=False):
        ordered = group.sort_values("minute_utc")
        segments = ordered.minute_utc.diff().gt(pd.Timedelta(seconds=90)).cumsum()
        for _, segment in ordered.groupby(segments, sort=False):
            value = segment.value.where(segment.quality) if observed else segment.value
            value = value.interpolate(limit_direction="both")
            if not value.notna().any():
                continue
            filtered = value.rolling(9, center=True, min_periods=1).median()
            if len(filtered) >= 9:
                filtered = savgol_filter(filtered.to_numpy(float), 9, 2, mode="interp")
            output.loc[segment.index] = np.asarray(filtered, float)
    return output.to_numpy(float)


def metrics(group, milestone, scope, name):
    result = dict(milestone=milestone, scope=scope, group=name, n=len(group),
                  n_operations=group.op.nunique(), n_unique_epochs=group.minute_utc.nunique())
    for stage, column in [("physical", BASE), ("beta", "beta_only_dbhz"), ("hgb", "final_dbhz")]:
        error = group[column] - group.observed_dbhz
        result[stage + "_rmse_dbhz"] = float(np.sqrt(np.mean(error**2)))
        result[stage + "_mae_db"] = float(np.mean(np.abs(error)))
        result[stage + "_mean_bias_db"] = float(np.mean(error))
        result[stage + "_median_bias_db"] = float(np.median(error))
        result[stage + "_equal_phase_rmse_dbhz"] = float(np.sqrt((error**2).groupby(group.mission_phase).mean().mean()))
        result[stage + "_equal_operation_rmse_dbhz"] = float(np.sqrt((error**2).groupby(group.op).mean().mean()))
        result[stage + "_mean_operation_rmse_dbhz"] = float(np.sqrt((error**2).groupby(group.op).mean()).mean())
    result["delta_hgb_minus_beta_rmse_dbhz"] = result["hgb_rmse_dbhz"] - result["beta_rmse_dbhz"]
    result["small_sample_descriptive"] = len(group) < 20
    return result


def main():
    inputs = [CACHE, PRED, MODEL, SELECT, OUT / "PLAN.md", Path(__file__)]
    initial_hashes = {str(path): sha(path) for path in inputs}
    selection = json.loads(SELECT.read_text(encoding="utf-8"))
    features = selection["selected_features"]
    parameters = dict(selection["selected_parameters"], random_state=42)
    assert len(features) == 44 and BASE in features
    forbidden = set(KEYS + ["mission_phase", TARGET, "cn0_observed_trend_dbhz", "residual_trend_target_db", "evaluation_split", "trend_training_eligible"])
    assert not set(features) & forbidden
    cached = normalize(joblib.load(CACHE))
    raw = normalize(pd.read_csv(PRED, usecols=KEYS + [TARGET, RAW, "trend_training_eligible", "source_bin_gps_seconds"], float_precision="round_trip"))
    frame = cached.merge(raw, on=KEYS, validate="one_to_one", suffixes=("", "_source"))
    assert len(frame) == len(cached) == len(raw) == 17439
    frame["quality"] = flags(frame.trend_training_eligible)
    assert np.array_equal(frame.quality, flags(frame.trend_training_eligible_source))
    inventory = frame.groupby("op").agg(start=("minute_utc", "min"), end=("minute_utc", "max"),
                                          n_bins=("op", "size"), n_quality=("quality", "sum"), phase=("mission_phase", "first"))
    inventory = inventory.sort_values("start")
    save(inventory.reset_index(), "operation_inventory.csv")
    all_predictions, all_metrics, coverage, betas, support, assignments, audits = [], [], [], [], [], [], []
    for milestone in CUTOFFS:
        cutoff = inventory.loc[milestone, "end"]
        past_ops = inventory.index[inventory.end.le(cutoff)].tolist()
        future_ops = inventory.index[inventory.start.gt(cutoff)].tolist()
        crossing = inventory.index[~inventory.index.isin(past_ops + future_ops)].tolist()
        assert milestone in past_ops and not set(past_ops) & set(future_ops)
        f = frame.copy()
        f["role"] = np.where(f.op.isin(past_ops), "past", np.where(f.op.isin(future_ops), "future", "crossing_excluded"))
        for op, row in inventory.iterrows():
            assignments.append(dict(milestone=milestone, cutoff_utc=cutoff, op=op, start_utc=row.start, end_utc=row.end,
                                    role="past" if op in past_ops else "future" if op in future_ops else "crossing_excluded"))
        f[BASE] = smooth(f, f[RAW])
        f["observed_dbhz"] = smooth(f, f[TARGET], observed=True)
        f["residual_target_db"] = f.observed_dbhz - f[BASE]
        eligible = f.quality & np.isfinite(f.observed_dbhz) & np.isfinite(f[BASE])
        assert int(eligible.sum()) == 15806
        fit_mask = eligible & f.role.eq("past")
        score_mask = eligible & f.role.eq("future")
        train = f.loc[fit_mask]
        assert set(train.signal_name) == SIGNALS
        assert train.minute_utc.max() < f.loc[score_mask, "minute_utc"].min()
        beta = train.groupby("signal_name").residual_target_db.median().to_dict()
        preprocess = ColumnTransformer([("physics", SimpleImputer(strategy="median"), features)], remainder="drop", verbose_feature_names_out=False)
        model = Pipeline([("preprocess", preprocess), ("model", HistGradientBoostingRegressor(**parameters))])
        with threadpool_limits(limits=4):
            model.fit(train[features], train.residual_target_db - train.signal_name.map(beta))
            raw_delta = model.predict(f[features])
        f["beta_db"] = f.signal_name.map(beta)
        assert f.beta_db.notna().all()
        f["hgb_raw_db"] = raw_delta
        f["filtered_total_correction_db"] = smooth(f, f.beta_db + f.hgb_raw_db)
        f["hgb_filtered_db"] = f.filtered_total_correction_db - f.beta_db
        f["beta_only_dbhz"] = f[BASE] + f.beta_db
        f["final_dbhz"] = f[BASE] + f.filtered_total_correction_db
        assert np.isfinite(f.loc[score_mask, [BASE, "beta_only_dbhz", "final_dbhz"]].to_numpy(float)).all()
        later = f.loc[score_mask].copy()
        later["milestone"] = milestone
        prediction_columns = ["milestone"] + KEYS + ["mission_phase", "source_bin_gps_seconds", "observed_dbhz", BASE, "beta_db", "hgb_raw_db", "hgb_filtered_db", "beta_only_dbhz", "final_dbhz"]
        all_predictions.append(later[prediction_columns])
        all_metrics.append(metrics(later, milestone, "pooled", "all_later_operations"))
        for op, group in later.groupby("op"):
            all_metrics.append(metrics(group, milestone, "operation", op))
        for phase, group in later.groupby("mission_phase"):
            all_metrics.append(metrics(group, milestone, "phase", phase))
        for signal, group in train.groupby("signal_name"):
            betas.append(dict(milestone=milestone, signal_name=signal, beta_db=beta[signal], n_fit=len(group)))
        for name, group in [("ALL", train)] + list(train.groupby("signal_name")):
            block_start = group.groupby("op").minute_utc.transform("min")
            # Use a shared operation start for simultaneous links, not a signal-specific start.
            block_start = group.op.map(inventory.start)
            blocks = (group.minute_utc - block_start).dt.total_seconds().floordiv(900).astype(int)
            coverage.append(dict(milestone=milestone, signal_name=name, n_fit=len(group), n_operations=group.op.nunique(),
                                 n_unique_epochs=group.minute_utc.nunique(), n_links=len(group[["op", "signal_name", "svid"]].drop_duplicates()),
                                 n_15min_blocks=(group.op + "|" + blocks.astype(str)).nunique(), first_utc=group.minute_utc.min(), last_utc=group.minute_utc.max()))
        imputer = model.named_steps["preprocess"].named_transformers_["physics"]
        for feature, median in zip(features, imputer.statistics_, strict=True):
            values = pd.to_numeric(train[feature], errors="coerce")
            tests = pd.to_numeric(later[feature], errors="coerce")
            valid = tests.notna()
            support.append(dict(milestone=milestone, feature=feature, train_min=values.min(), train_max=values.max(), imputer_median=median,
                                train_unique=values.nunique(), train_missing_fraction=values.isna().mean(), test_missing_fraction=tests.isna().mean(),
                                test_outside_train_range_fraction=float(((tests[valid] < values.min()) | (tests[valid] > values.max())).mean()) if valid.any() and values.notna().any() else np.nan))
        perturbed = f[TARGET].copy()
        perturbed.loc[f.role.ne("past")] += 1000.
        modified_observed = smooth(f, perturbed, observed=True)
        assert np.allclose(modified_observed[fit_mask], f.loc[fit_mask, "observed_dbhz"], rtol=0, atol=0)
        # Target perturbation changes no features: target-derived columns are not selected.
        test_features_before = f.loc[f.role.eq("future"), features].copy()
        mutated = f.copy()
        mutated[TARGET] = perturbed
        mutated["observed_dbhz"] = modified_observed
        mutated["residual_target_db"] = mutated.observed_dbhz - mutated[BASE]
        assert test_features_before.equals(mutated.loc[mutated.role.eq("future"), features])
        audit = dict(milestone=milestone, cutoff_utc=str(cutoff), past_operations=past_ops, future_operations=future_ops, crossing_excluded=crossing,
                     n_fit=int(fit_mask.sum()), n_score=int(score_mask.sum()), latest_fit_utc=str(train.minute_utc.max()), earliest_score_utc=str(later.minute_utc.min()),
                     later_rows_in_beta_fit=0, later_rows_in_imputer_fit=0, later_rows_in_hgb_fit=0,
                     future_target_perturbation_training_targets_unchanged=True, future_target_perturbation_features_unchanged=True,
                     all_missing_training_features=[c for c in features if train[c].isna().all()], actual_hgb_iterations=int(model.named_steps["model"].n_iter_),
                     effective_transformed_feature_count=int(model.named_steps["model"].n_features_in_))
        audits.append(audit)
        joblib.dump(dict(model=model, beta=beta, features=features, protocol=audit), OUT / f"model_after_{milestone}.joblib", compress=3)
        print(json.dumps(all_metrics[-(1 + later.op.nunique() + later.mission_phase.nunique())], ensure_ascii=False), flush=True)
    metric_frame = pd.DataFrame(all_metrics)
    save(pd.concat(all_predictions, ignore_index=True), "later_operation_predictions.csv")
    save(metric_frame, "metrics.csv")
    save(metric_frame[metric_frame.scope.eq("pooled")], "pooled_metrics.csv")
    save(pd.DataFrame(coverage), "training_coverage.csv")
    save(pd.DataFrame(betas), "training_beta.csv")
    save(pd.DataFrame(support), "feature_support.csv")
    save(pd.DataFrame(assignments), "operation_assignments.csv")
    unchanged = {path: sha(Path(path)) == value for path, value in initial_hashes.items()}
    assert all(unchanged.values())
    qa = dict(all_checks_passed=True, sources_unchanged=unchanged, source_sha256=initial_hashes, feature_count=len(features), features=features,
              selected_parameters=parameters, n_all_bins=len(frame), n_eligible=15806, protocols=audits,
              labels="retrospective temporal-transfer stress test; historically full-record-selected architecture; postprocessed geometry; centered offline trends; no prospective validation claim",
              compared_stages_share_identical_rows=True, overlapping_cutoffs_not_pooled=True, no_bootstrap=True)
    (OUT / "QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metric_frame.loc[metric_frame.scope.eq("pooled"), ["milestone", "n", "physical_rmse_dbhz", "beta_rmse_dbhz", "hgb_rmse_dbhz", "delta_hgb_minus_beta_rmse_dbhz"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
