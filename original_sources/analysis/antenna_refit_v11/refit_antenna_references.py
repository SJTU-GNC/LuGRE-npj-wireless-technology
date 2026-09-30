"""Prespecified antenna-reference tests; read-only original model/data."""
from pathlib import Path
import os
import sys
import json
import hashlib

ROOT = Path(r"D:\月球导航")
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "runtime_cache/python_deps"))
sys.path.insert(0, str(ROOT / "script"))
sys.path.insert(0, str(OUT.parent / "thermal_refit_v9"))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from build_cn0_attitude_2d_gain import bilinear_periodic
import refit_temperature_models as helper

MODEL, PRED, CACHE = helper.MODEL, helper.PRED, helper.CACHE
SOURCE = ROOT / "table/algorithm/cn0_constellation_physics_baseline_sp3_tlm_exact/cn0_constellation_physics_predictions.csv"
GRID = ROOT / "table/external_reference/galileo_grap_eirp_grid.csv"
RX = ROOT / "data/external_reference/lugre_antenna/LuGRE_Fig3_gain_outer_envelope_digitized.csv"
MAPPING = ROOT / "data/external_reference/mapping/active_prn_svn_block_20250115_20250316.csv"
KEYS, RAW, BASE, DERIVED = helper.KEYS, helper.RAW, helper.BASE, helper.DERIVED
SCENARIOS = ["control", "foc_lower", "foc_upper", "foc_radial_minus1", "foc_radial_plus1", "rx_two_side_min", "rx_two_side_power_mean"]
PROTOCOLS = ["original", "exclude_iov"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def maxerr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    m = np.isfinite(a)
    return float(np.max(np.abs(a[m]-b[m]))) if m.any() else 0.


def identify(frame, mapping):
    frame = frame.copy()
    frame["system"] = np.where(frame.signal_name.str.startswith("GAL"), "E", "G")
    prn = frame.system + frame.svid.astype(int).astype(str).str.zfill(2)
    frame["antenna_family"] = prn.map(mapping.antenna_type)
    assert frame.antenna_family.notna().all()
    frame["foc"] = frame.antenna_family.eq("GALILEO-2")
    frame["iov"] = frame.antenna_family.eq("GALILEO-1")
    return frame


def prepare_shifts(bins, mapping):
    geometry_budget = ["fspl_db", "l_ion_abs_budget_db", "l_gas_abs_budget_db"]
    cols = ["rx_utc", "op", "system", "signal_name", "svid", "tx_theta_body_deg", "tx_phi_body_deg", "tx_eirp_2d_dbw", "rx_offboresight_spice_deg", "rx_gain_envelope_dbic", RAW, *DERIVED, *geometry_budget]
    raw = pd.read_csv(SOURCE, usecols=cols, float_precision="round_trip")
    raw = identify(raw, mapping)
    raw["rx_utc_dt"] = pd.to_datetime(raw.rx_utc, utc=True)
    grid = pd.read_csv(GRID, float_precision="round_trip")
    curve = pd.read_csv(RX, float_precision="round_trip")
    native = pd.DataFrame(0., index=raw.index, columns=["shift_"+s for s in SCENARIOS])
    native["check_tx_mean"] = np.nan
    native["check_rx_max"] = np.nan
    angular_ranges = []
    for signal in ["GAL_E1", "GAL_E5a"]:
        m = raw.signal_name.eq(signal)
        foc = m & raw.foc
        table = grid.loc[grid.signal_name.eq(signal)]
        patterns = {}
        for name, column in [("mean", "eirp_dbw"), ("lower", "eirp_lower95_dbw"), ("upper", "eirp_upper95_dbw")]:
            p = table.pivot(index="coelevation_deg", columns="azimuth_deg", values=column).sort_index().sort_index(axis=1)
            keep = p.columns.to_numpy(float)<360
            patterns[name] = {"theta":p.index.to_numpy(float), "phi":p.columns.to_numpy(float)[keep], "gain":p.to_numpy(float)[:,keep]}
        theta, phi = raw.loc[m,"tx_theta_body_deg"].to_numpy(), raw.loc[m,"tx_phi_body_deg"].to_numpy()
        native.loc[m,"check_tx_mean"] = bilinear_periodic(patterns["mean"], theta, phi)
        theta, phi = raw.loc[foc,"tx_theta_body_deg"].to_numpy(), raw.loc[foc,"tx_phi_body_deg"].to_numpy()
        assert np.isfinite(theta).all() and (theta-1>=0).all() and (theta+1<=90).all()
        angular_ranges.append(dict(signal=signal,n=int(foc.sum()),theta_min=float(theta.min()),theta_max=float(theta.max()),shifted_max=float((theta+1).max()),n_shifted_outside_53=int(((theta+1)>53).sum())))
        for name in ["lower","upper"]:
            native.loc[foc,"shift_foc_"+name] = bilinear_periodic(patterns[name],theta,phi)-raw.loc[foc,"tx_eirp_2d_dbw"]
        for name, offset in [("minus1",-1.),("plus1",1.)]:
            native.loc[foc,"shift_foc_radial_"+name] = bilinear_periodic(patterns["mean"],theta+offset,phi)-raw.loc[foc,"tx_eirp_2d_dbw"]
    for signal,band in [("GPS_L1","L1/E1"),("GAL_E1","L1/E1"),("GPS_L5","L5/E5a"),("GAL_E5a","L5/E5a")]:
        m = raw.signal_name.eq(signal)
        c = curve.loc[curve.band.eq(band)].sort_values("signed_elevation_deg")
        theta = raw.loc[m,"rx_offboresight_spice_deg"].to_numpy()
        xp,yp = c.signed_elevation_deg.to_numpy(),c.outer_envelope_gain_dB_axis.to_numpy()
        assert np.isfinite(theta).all() and (theta>=0).all() and (theta<=90).all()
        plus,minus=np.interp(theta,xp,yp),np.interp(-theta,xp,yp)
        native.loc[m,"check_rx_max"]=np.maximum(plus,minus)
        native.loc[m,"shift_rx_two_side_min"]=np.minimum(plus,minus)-raw.loc[m,"rx_gain_envelope_dbic"]
        native.loc[m,"shift_rx_two_side_power_mean"]=10*np.log10((10**(plus/10)+10**(minus/10))/2)-raw.loc[m,"rx_gain_envelope_dbic"]
    gal = raw.system.eq("E")
    native_tx_error=maxerr(native.loc[gal,"check_tx_mean"],raw.loc[gal,"tx_eirp_2d_dbw"])
    native_rx_error=maxerr(native.check_rx_max,raw.rx_gain_envelope_dbic)
    assert native_tx_error<1e-9 and native_rx_error<1e-9
    aligned_names=list(native)+["check_"+c for c in [RAW,*DERIVED,*geometry_budget]]
    raw=pd.concat([raw,native],axis=1)
    for c in [RAW,*DERIVED,*geometry_budget]:
        raw["check_"+c]=raw[c]
    aligned=bins.copy()
    for c in aligned_names:
        aligned[c]=np.nan
    aligned=helper.engine._interpolate_link_features_to_anchor(raw,aligned,aligned_names)
    # The original minute pipeline also replaces the per-band path geometry by
    # a shared ray and corrects these three budget terms. Retain that exact
    # correction when checking the original composite budget reproduction.
    geometry_correction=sum((aligned["check_"+c].fillna(0)-bins[c].fillna(0) for c in geometry_budget))
    reproduction={c:maxerr(aligned["check_"+c]+geometry_correction,bins[c]) for c in [RAW,*DERIVED]}
    reproduction["rx_gain_envelope_dbic"]=maxerr(aligned.check_rx_max,bins.rx_gain_envelope_dbic)
    assert max(reproduction.values())<1e-8,reproduction
    for s in SCENARIOS:
        assert aligned["shift_"+s].notna().all()
        if s.startswith("foc"):
            assert aligned.loc[~bins.foc,"shift_"+s].eq(0).all()
    assert aligned.shift_foc_lower.max()<1e-8 and aligned.shift_foc_upper.min()>-1e-8
    shift=aligned[KEYS+["antenna_family","foc","iov","check_tx_mean","check_rx_max"]+["shift_"+s for s in SCENARIOS]].copy()
    save(shift,"aligned_reference_changes.csv")
    return aligned,dict(native_tx_reproduction_error=native_tx_error,native_rx_reproduction_error=native_rx_error,anchor_reproduction_errors=reproduction,maximum_original_common_geometry_budget_adjustment_db=float(geometry_correction.abs().max()),angular_support=angular_ranges)


def scenario_frame(bins, aligned, scenario, features):
    f=bins.copy()
    shift=aligned["shift_"+scenario].to_numpy()
    for c in [RAW,*DERIVED]:
        f[c]=bins[c]+shift
    changed=[*DERIVED,BASE]
    if scenario.startswith("rx_"):
        f["rx_gain_envelope_dbic"]=bins.rx_gain_envelope_dbic+shift
        changed.append("rx_gain_envelope_dbic")
    f[BASE]=helper.smooth(f,f[RAW])
    f["residual_trend_target_db"]=f.cn0_observed_trend_dbhz-f[BASE]
    unmodified=[c for c in features if c not in changed]
    assert f[unmodified].equals(bins[unmodified])
    assert f.cn0_observed_trend_dbhz.equals(bins.cn0_observed_trend_dbhz)
    assert f.trend_training_eligible.equals(bins.trend_training_eligible)
    assert f.evaluation_split.equals(bins.evaluation_split)
    return f,shift,changed


def summaries(pred, mask, scope, protocol, scenario, control):
    records=[]
    partitions=[("all_links",mask),("foc",mask & pred.foc),("gps",mask & pred.signal_name.str.startswith("GPS")),("iov",mask & pred.iov)]
    for population,selected in partitions:
        d=pred.loc[selected]
        if not len(d):continue
        groups=[("pooled","all",d)]
        for column,level in [("mission_phase","phase"),("op","operation"),("signal_name","signal")]:
            groups.extend((level,str(name),g) for name,g in d.groupby(column,sort=True))
        groups.extend(("phase_signal",str(phase)+"|"+str(signal),g) for (phase,signal),g in d.groupby(["mission_phase","signal_name"],sort=True))
        for level,group,g in groups:
            original=control.loc[g.index]
            rec=dict(protocol=protocol,scenario=scenario,scope=scope,population=population,level=level,group=group,n=len(g),n_links=len(g[["op","signal_name","svid"]].drop_duplicates()),n_operations=g.op.nunique())
            for label,col in [("baseline","baseline_dbhz"),("persistent","persistent_only_dbhz"),("final","final_dbhz")]:
                e=g[col]-g.observed_dbhz
                e0=original[col]-original.observed_dbhz
                rec[label+"_rmse_dbhz"]=float(np.sqrt(np.mean(e**2)))
                rec[label+"_rmse_change_dbhz"]=float(np.sqrt(np.mean(e**2))-np.sqrt(np.mean(e0**2)))
                rec[label+"_mean_error_db"]=float(e.mean())
                rec[label+"_median_error_db"]=float(e.median())
                rec[label+"_phase_weighted_rmse_dbhz"]=float(np.sqrt((e**2).groupby(g.mission_phase).mean().mean()))
            diff=g.baseline_dbhz-original.baseline_dbhz
            rec.update(baseline_shift_mean_db=float(diff.mean()),baseline_shift_p05_db=float(diff.quantile(.05)),baseline_shift_median_db=float(diff.median()),baseline_shift_p95_db=float(diff.quantile(.95)),baseline_shift_abs_p95_db=float(diff.abs().quantile(.95)),beta_change_mean_db=float((g.beta_db-original.beta_db).mean()),hgb_mean_db=float(g.delta_filtered_db.mean()),hgb_rms_db=float(np.sqrt(np.mean(g.delta_filtered_db**2))),hgb_change_mean_db=float((g.delta_filtered_db-original.delta_filtered_db).mean()),hgb_change_rms_db=float(np.sqrt(np.mean((g.delta_filtered_db-original.delta_filtered_db)**2))),final_change_rms_db=float(np.sqrt(np.mean((g.final_dbhz-original.final_dbhz)**2))))
            rec["mean_change_closure_error_db"]=float((g.final_dbhz-original.final_dbhz).mean()-rec["baseline_shift_mean_db"]-rec["beta_change_mean_db"]-rec["hgb_change_mean_db"])
            records.append(rec)
    return records


def main():
    inputs=[MODEL,PRED,CACHE,SOURCE,GRID,RX,MAPPING,OUT/"PLAN.md",Path(helper.__file__),ROOT/"script/build_cn0_attitude_2d_gain.py",Path(helper.engine.__file__)]
    before={str(p):sha(p) for p in inputs}
    artifact=joblib.load(MODEL)
    features=artifact["model_features"]
    bins=helper.normalize(joblib.load(CACHE))
    original=helper.normalize(pd.read_csv(PRED,float_precision="round_trip"))
    bins=bins.merge(original[KEYS+[RAW,"ai_trend_residual_raw_pred_db","cn0_physics_ai_eval_dbhz"]],on=KEYS,how="left",validate="one_to_one")
    bins=identify(bins,pd.read_csv(MAPPING).set_index("prn"))
    bins["eligible"]=bins.trend_training_eligible.astype(str).str.lower().eq("true") & np.isfinite(bins.residual_trend_target_db)
    fit=bins.eligible & bins.evaluation_split.isin(["train","validation"])
    assert len(bins)==17439 and bins.eligible.sum()==15806 and fit.sum()==9894 and len(features)==44
    print("Preparing native pattern references and minute alignment",flush=True)
    aligned,qa=prepare_shifts(bins,pd.read_csv(MAPPING).set_index("prn"))
    records,betas,predictions,contract=[],[],[],[]
    counts=[]
    for protocol in PROTOCOLS:
        valid=bins.eligible & (~bins.iov if protocol=="exclude_iov" else True)
        fitmask=valid & bins.evaluation_split.isin(["train","validation"])
        counts.append(dict(protocol=protocol,fit=int(fitmask.sum()),test=int((valid & bins.evaluation_split.eq("test")).sum()),external=int((valid & bins.evaluation_split.eq("external_holdout")).sum()),eligible=int(valid.sum())))
        control=None
        beta0=None
        for scenario in SCENARIOS:
            print(f"Fitting {protocol} / {scenario}, n={fitmask.sum()}",flush=True)
            f,shift,changed=scenario_frame(bins,aligned,scenario,features)
            if scenario=="control":
                assert maxerr(f[BASE],bins[BASE])<1e-9
            with threadpool_limits(limits=4):
                model,beta,pred=helper.fit_model(f,fitmask,artifact["model"],features)
            pred["foc"],pred["iov"]=bins.foc,bins.iov
            pred["raw_baseline_shift_db"]=shift
            pred["scenario"],pred["protocol"]=scenario,protocol
            pred["used_for_fit"]=fitmask
            pred["used_for_metrics"]=valid
            if scenario=="control":
                control,beta0=pred.copy(),beta.copy()
                if protocol=="original":
                    qe=dict(beta=max(abs(beta[k]-artifact["signal_beta_db"][k]) for k in beta),raw=maxerr(pred.beta_db+pred.delta_raw_db,bins.ai_trend_residual_raw_pred_db),test=maxerr(pred.loc[valid & bins.evaluation_split.isin(["test","external_holdout"]),"final_dbhz"],bins.loc[valid & bins.evaluation_split.isin(["test","external_holdout"]),"cn0_physics_ai_eval_dbhz"]))
                    assert max(qe.values())<1e-7,qe
                    qa["original_model_reproduction"]=qe
            assert maxerr(pred.observed_dbhz,control.observed_dbhz)==0
            betas.extend(dict(protocol=protocol,scenario=scenario,signal_name=k,beta_db=v,control_beta_db=beta0[k],beta_change_db=v-beta0[k],absolute_beta_change_db=abs(v)-abs(beta0[k]),n_fit=int((fitmask & bins.signal_name.eq(k)).sum())) for k,v in beta.items())
            for scope,mask in [("all_eligible",valid),("fit",fitmask),("test",valid & bins.evaluation_split.eq("test")),("external_holdout",valid & bins.evaluation_split.eq("external_holdout"))]:
                records+=summaries(pred,mask,scope,protocol,scenario,control)
            contract.append(dict(protocol=protocol,scenario=scenario,changed_feature_columns=changed,unchanged_feature_count=len(features)-len(changed),n_minute_effect_rows=int(np.count_nonzero(np.abs(shift)>1e-12)),target_and_split_unchanged=True))
            predictions.append(pred)
            joblib.dump(dict(model=model,signal_beta_db=beta,model_features=features,scenario=scenario,protocol=protocol,scenario_contract=contract[-1]),OUT/f"model_{protocol}_{scenario}.joblib")
    metrics=pd.DataFrame(records)
    predall=pd.concat(predictions,ignore_index=True)
    assert metrics.mean_change_closure_error_db.abs().max()<1e-10
    save(metrics,"paired_metrics.csv")
    save(pd.DataFrame(betas),"refitted_beta.csv")
    save(predall,"paired_predictions.csv")
    save(pd.DataFrame(counts),"sample_counts.csv")
    save(bins[KEYS+["evaluation_split","eligible","foc","iov","antenna_family","tx_theta_body_deg"]],"sample_manifest.csv")
    qa.update(original_inputs_unchanged=all(sha(Path(p))==h for p,h in before.items()),input_hashes=before,feature_contracts=contract,sample_counts=counts,scenario_selection="none; all seven prespecified scenarios reported",original_model_modified=False,manuscript_modified=False,maximum_mean_allocation_closure_error=float(metrics.mean_change_closure_error_db.abs().max()),model_parameters=artifact["model"].named_steps["model"].get_params())
    assert qa["original_inputs_unchanged"]
    (OUT/"qa.json").write_text(json.dumps(qa,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print(metrics.loc[metrics.scope.isin(["test","external_holdout"]) & metrics.population.eq("all_links") & metrics.level.eq("pooled"),["protocol","scenario","scope","n","baseline_rmse_dbhz","persistent_rmse_dbhz","final_rmse_dbhz","final_rmse_change_dbhz"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
