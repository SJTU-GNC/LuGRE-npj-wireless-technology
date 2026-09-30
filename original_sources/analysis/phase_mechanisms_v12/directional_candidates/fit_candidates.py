"""Predeclared directional-response candidates; original data/models read-only."""
from pathlib import Path
import os
import sys
import json
import hashlib

ROOT = Path(r"D:\月球导航")
OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parent.parent
sys.path.insert(0, str(ROOT / "runtime_cache/python_deps"))
sys.path.insert(0, str(ANALYSIS / "thermal_refit_v9"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import refit_temperature_models as helper

KEYS = helper.KEYS
SIGNALS = ["GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"]
RESPONSE = OUT.parent / "response_design/minute_reference_responses.joblib"
MODELS = ["constant", "tx_response", "rx_response", "tx_rx_response"]
SLOPES = ["d_tx_theta_db_per_deg", "d_tx_phi_db_per_deg", "d_rx_theta_db_per_deg"]


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flag(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def design(f, name):
    columns, arrays, bounds = [], [], []
    for sig in SIGNALS:
        columns.append("beta_" + sig)
        arrays.append(f.signal_name.eq(sig).to_numpy(float))
        bounds.append((None, None))
    if name in ["tx_response", "tx_rx_response"]:
        for sig in SIGNALS:
            mask = f.signal_name.eq(sig).to_numpy(float)
            for axis, slope, bound in [("theta", SLOPES[0], 1.), ("phi", SLOPES[1], 5.)]:
                columns.append("tx_" + axis + "_" + sig)
                arrays.append(mask * f[slope].to_numpy(float))
                bounds.append((-bound, bound))
    if name in ["rx_response", "tx_rx_response"]:
        for band, signals in [("L1_E1", ["GPS_L1", "GAL_E1"]), ("L5_E5a", ["GPS_L5", "GAL_E5a"])]:
            columns.append("rx_theta_" + band)
            arrays.append(f.signal_name.isin(signals).to_numpy(float) * f[SLOPES[2]].to_numpy(float))
            bounds.append((-1., 1.))
    x = np.column_stack(arrays)
    assert np.isfinite(x).all()
    return x, columns, bounds


def fit_l1(x, y, bounds):
    n, p = x.shape
    mat, ident = csr_matrix(x), eye(n, format="csr")
    constraints = vstack([hstack([mat, -ident]), hstack([-mat, -ident])], format="csr")
    result = linprog(np.r_[np.zeros(p), np.ones(n)/n], A_ub=constraints,
                     b_ub=np.r_[y, -y], bounds=bounds + [(0., None)] * n,
                     method="highs", options={"time_limit": 180.})
    assert result.success, result.message
    coef = result.x[:p]
    loss = float(np.mean(np.abs(y - x @ coef)))
    assert abs(loss - result.fun) < 1e-7
    return coef, dict(mean_absolute_training_residual_db=loss, iterations=int(result.nit),
                      rank=int(np.linalg.matrix_rank(x)), n_columns=p,
                      condition_number=float(np.linalg.cond(x)))


def metric_groups(d):
    yield "pooled", "all", d
    for field in ["mission_phase", "op", "signal_name"]:
        for value, g in d.groupby(field, sort=True):
            yield field, value, g
    for fields, level in [(["mission_phase", "signal_name"], "phase_signal"),
                          (["op", "signal_name", "svid"], "link")]:
        for value, g in d.groupby(fields, sort=True):
            yield level, "|".join(map(str, value)), g


def metrics(pred, name):
    rows = []
    masks = {"fitting_descriptive": pred.eligible & pred.evaluation_split.isin(["train", "validation"]),
             "internal_test": pred.eligible & pred.evaluation_split.eq("test"),
             "external_holdout": pred.eligible & pred.evaluation_split.eq("external_holdout"),
             "all_supported_descriptive": pred.eligible}
    for scope, mask in masks.items():
        for level, key, g in metric_groups(pred.loc[mask]):
            if not len(g):
                continue
            row = dict(model=name, scope=scope, level=level, group=key, n=len(g),
                       n_operations=g.op.nunique(), n_satellites=g[["signal_name", "svid"]].drop_duplicates().shape[0])
            for label, c in [("baseline", "baseline_dbhz"), ("final", "final_dbhz")]:
                error = g[c] - g.observed_dbhz
                row[label + "_rmse_dbhz"] = float(np.sqrt(np.mean(error**2)))
                row[label + "_mean_error_db"] = float(error.mean())
                row[label + "_median_error_db"] = float(error.median())
                row[label + "_equal_phase_rmse_dbhz"] = float(np.sqrt((error**2).groupby(g.mission_phase).mean().mean()))
            for c in ["beta_db", "tx_raw_db", "rx_raw_db", "directional_filtered_db", "correction_filtered_db"]:
                row[c + "_mean"] = float(g[c].mean())
                row[c + "_rms"] = float(np.sqrt(np.mean(g[c]**2)))
            rows.append(row)
    return rows


def main():
    inputs = [helper.MODEL, helper.CACHE, helper.PRED, RESPONSE, OUT.parent / "PLAN.md"]
    hashes = {str(p): sha(p) for p in inputs}
    artifact = joblib.load(helper.MODEL)
    f = helper.normalize(joblib.load(helper.CACHE))
    original = helper.normalize(pd.read_csv(helper.PRED))
    extra = [helper.RAW, "cn0_physics_ai_eval_dbhz", "source_bin_gps_seconds"]
    f = f.merge(original[KEYS + extra], on=KEYS, validate="one_to_one")
    response = helper.normalize(joblib.load(RESPONSE))
    response_cols = [c for c in response if c not in f or c in KEYS]
    f = f.merge(response[response_cols], on=KEYS, validate="one_to_one", how="left")
    f["eligible"] = flag(f.trend_training_eligible) & np.isfinite(f.residual_trend_target_db)
    f["candidate_support"] = (flag(f.response_supported) & ~flag(f.is_iov)
                                & ~flag(f.earth_blocked) & ~flag(f.moon_blocked)
                                & np.isfinite(f[SLOPES]).all(axis=1))
    # Preserve gaps caused by unsupported records even when adjacent retained times are 60 s apart.
    f["thermal_segment"] = ""
    for _, g in f.groupby(["evaluation_split", "op", "signal_name", "svid"], sort=False):
        g = g.sort_values("minute_utc")
        segments = (~g.candidate_support).cumsum().astype(str)
        f.loc[g.index, "thermal_segment"] = "response_" + segments
    save(f.groupby(["op", "mission_phase", "evaluation_split", "eligible"], dropna=False)
         .agg(n_original=("op", "size"), n_supported=("candidate_support", "sum")).reset_index(), "sample_counts.csv")
    full_count = len(f)
    f = f[f.candidate_support].copy().reset_index(drop=True)
    fit = f.eligible & f.evaluation_split.isin(["train", "validation"])
    assert not f.loc[fit, "op"].isin(["OP2", "OP21", "OP27", "OP74"]).any()
    assert f.loc[fit, "signal_name"].nunique() == 4
    save(f[KEYS + ["evaluation_split", "mission_phase", "eligible", "thermal_segment"] + SLOPES], "sample_manifest.csv")
    print(f"Common supported rows {len(f)}, eligible {f.eligible.sum()}, fitting {fit.sum()}", flush=True)
    coeffs, stats, all_metrics, all_preds = [], [], [], []
    designs = {}
    for name in MODELS:
        print("L1 fit: " + name, flush=True)
        x, columns, bounds = design(f, name)
        if name == "constant":
            beta = f.loc[fit].groupby("signal_name").residual_trend_target_db.median()
            coef = np.array([beta[s] for s in SIGNALS])
            stat = dict(mean_absolute_training_residual_db=float(np.mean(np.abs(f.loc[fit].residual_trend_target_db - x[fit] @ coef))),
                        iterations=0, rank=4, n_columns=4, condition_number=float(np.linalg.cond(x[fit])))
        else:
            coef, stat = fit_l1(x[fit], f.loc[fit].residual_trend_target_db.to_numpy(), bounds)
        stat.update(model=name, n_fitting=int(fit.sum()))
        stats.append(stat)
        for c, value, bound in zip(columns, coef, bounds):
            coeffs.append(dict(model=name, coefficient=c, value=value, lower=bound[0], upper=bound[1],
                               unit="dB" if c.startswith("beta") else "equivalent_degrees",
                               at_bound=bool(bound[0] is not None and min(abs(value-bound[0]), abs(value-bound[1])) < 1e-6)))
        p = f[KEYS + ["evaluation_split", "mission_phase", "eligible", "thermal_segment"]].copy()
        p["observed_dbhz"], p["baseline_dbhz"] = f.cn0_observed_trend_dbhz, f[helper.BASE]
        p["beta_db"] = x[:, :4] @ coef[:4]
        for part in ["tx", "rx"]:
            indices = [i for i, c in enumerate(columns) if c.startswith(part + "_")]
            p[part + "_raw_db"] = x[:, indices] @ coef[indices] if indices else 0.
        p["correction_raw_db"] = x @ coef
        p["correction_filtered_db"] = helper.smooth(f, p.correction_raw_db, split=True)
        p["directional_filtered_db"] = p.correction_filtered_db - p.beta_db
        p["filter_adjustment_db"] = p.correction_filtered_db - p.correction_raw_db
        p["final_dbhz"] = p.baseline_dbhz + p.correction_filtered_db
        assert np.max(np.abs(p.beta_db+p.tx_raw_db+p.rx_raw_db+p.filter_adjustment_db-p.correction_filtered_db)) < 1e-10
        p["model"] = name
        all_preds.append(p)
        all_metrics += metrics(p, name)
        joblib.dump(dict(columns=columns, bounds=bounds, coefficients=coef, stats=stat), OUT / (name + ".joblib"))
        designs[name] = (x, columns, bounds)
    print("Fitting original 44-feature HGB on the identical cohort", flush=True)
    with threadpool_limits(limits=4):
        hgb, beta, p = helper.fit_model(f, fit, artifact["model"], artifact["model_features"])
    p["tx_raw_db"], p["rx_raw_db"] = np.nan, np.nan
    p["directional_filtered_db"] = p.delta_filtered_db
    p["correction_raw_db"] = p.beta_db + p.delta_raw_db
    p["filter_adjustment_db"] = p.correction_filtered_db - p.correction_raw_db
    p["model"] = "hgb_same_cohort"
    all_preds.append(p)
    all_metrics += metrics(p, "hgb_same_cohort")
    joblib.dump(dict(model=hgb, beta=beta, features=artifact["model_features"]), OUT / "hgb_same_cohort.joblib")
    predictions = pd.concat(all_preds, ignore_index=True)
    save(predictions, "candidate_predictions.csv")
    joblib.dump(predictions, OUT / "candidate_predictions.joblib")
    metric = pd.DataFrame(all_metrics)
    control = metric[metric.model.eq("constant")][["scope", "level", "group", "final_rmse_dbhz"]].rename(columns={"final_rmse_dbhz": "constant_rmse_dbhz"})
    metric = metric.merge(control, on=["scope", "level", "group"], validate="many_to_one")
    metric["rmse_change_from_constant_dbhz"] = metric.final_rmse_dbhz - metric.constant_rmse_dbhz
    save(metric, "candidate_metrics.csv")
    save(pd.DataFrame(coeffs), "coefficients.csv")
    save(pd.DataFrame(stats), "fit_diagnostics.csv")
    # Descriptive correspondence with HGB, after removing per-operation/link medians.
    correspondence = []
    hgb_p = predictions[predictions.model.eq("hgb_same_cohort")][KEYS + ["directional_filtered_db"]].rename(columns={"directional_filtered_db": "hgb_delta_db"})
    for name in MODELS[1:]:
        q = predictions[predictions.model.eq(name)].merge(hgb_p, on=KEYS, validate="one_to_one")
        for scope, m in [("external_holdout", q.eligible & q.evaluation_split.eq("external_holdout")),
                         ("internal_test", q.eligible & q.evaluation_split.eq("test")),
                         ("all_supported_descriptive", q.eligible)]:
            for phase, g in q.loc[m].groupby("mission_phase"):
                v = g[["directional_filtered_db", "hgb_delta_db"]]
                centred = v - g.groupby(["op", "signal_name", "svid"])[v.columns].transform("median")
                rho = float(spearmanr(centred.iloc[:, 0], centred.iloc[:, 1]).statistic) if len(g)>10 and (centred.std()>1e-9).all() else np.nan
                correspondence.append(dict(model=name, scope=scope, phase=phase, n=len(g), within_link_spearman=rho))
    save(pd.DataFrame(correspondence), "within_link_hgb_correspondence.csv")
    # Stability diagnostic: no held-out observations enter these fits.
    x, columns, bounds = designs["tx_rx_response"]
    stability = []
    for omitted in sorted(f.loc[fit, "op"].unique(), key=lambda s: int(s[2:])):
        keep = fit & f.op.ne(omitted)
        print("Coefficient stability, omit fitting " + omitted, flush=True)
        coef, stat = fit_l1(x[keep], f.loc[keep].residual_trend_target_db.to_numpy(), bounds)
        for c, val in zip(columns, coef):
            stability.append(dict(omitted_fitting_operation=omitted, coefficient=c, value=val, n_fitting=int(keep.sum())))
    save(pd.DataFrame(stability), "coefficient_stability.csv")
    assert all(sha(Path(p)) == h for p,h in hashes.items())
    qa = dict(input_sha256=hashes, original_rows=full_count, supported_rows=len(f),
              eligible_rows=int(f.eligible.sum()), fitting_rows=int(fit.sum()),
              internal_test_rows=int((f.eligible & f.evaluation_split.eq("test")).sum()),
              external_holdout_rows=int((f.eligible & f.evaluation_split.eq("external_holdout")).sum()),
              fit_operations=sorted(f.loc[fit, "op"].unique().tolist()),
              source_files_unchanged=True, observed_target_unchanged=True,
              same_masks_all_models=True, physical_geometry_unchanged=True,
              unit_note="Bounded coefficients are equivalent reference-angle offsets, not measured pointing errors")
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metric[(metric.scope.eq("external_holdout")) & metric.level.isin(["pooled", "op"])][["model", "group", "n", "final_rmse_dbhz", "rmse_change_from_constant_dbhz"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
