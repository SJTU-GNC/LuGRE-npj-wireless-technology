"""Reference-surface local responses. No geometry or model fitting is performed."""
from pathlib import Path
import sys
import json
import hashlib

ROOT = Path(r"D:\月球导航")
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "runtime_cache/python_deps"))
sys.path.insert(0, str(ROOT / "script"))
import numpy as np
import pandas as pd
import joblib
from build_cn0_attitude_2d_gain import (
    parse_iirm_2d, parse_iiia_2d, parse_iif_patterns,
    bilinear_periodic, TX_POWER_DBW,
)
from train_cn0_single_global_residual_1min_op74_holdout import _interpolate_link_features_to_anchor

SOURCE = ROOT / "table/algorithm/cn0_constellation_physics_baseline_sp3_tlm_exact/cn0_constellation_physics_predictions.csv"
CACHE = ROOT / "table/algorithm/cn0_ai_residual_story/final_split_feature_frame.joblib"
PHYSICAL = OUT.parents[1] / "phase_residual_analysis_v1/prepared/physical_minutes.csv"
MAPPING = ROOT / "data/external_reference/mapping/active_prn_svn_block_20250115_20250316.csv"
GRID = ROOT / "table/external_reference/galileo_grap_eirp_grid.csv"
RX = ROOT / "data/external_reference/lugre_antenna/LuGRE_Fig3_gain_outer_envelope_digitized.csv"
GPS_DIR = ROOT / "data/external_reference/gnss_antenna/gps"
KEYS = ["minute_utc", "op", "signal_name", "svid"]
LINK = ["op", "system", "signal_name", "svid"]
DELTAS = ["theta_minus1_delta_db", "theta_plus1_delta_db", "phi_minus5_delta_db", "phi_plus5_delta_db", "rx_minus1_delta_db", "rx_plus1_delta_db"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.17g")


def compare(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    joint = np.isfinite(a) & np.isfinite(b)
    return dict(n_compared=int(joint.sum()), finite_mask_mismatches=int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))), max_abs_error=float(np.max(np.abs(a[joint]-b[joint]))) if joint.any() else None)


def identify(frame, mapping):
    frame = frame.copy()
    frame["svid"] = frame.svid.astype(int)
    frame["system"] = np.where(frame.signal_name.str.startswith("GAL"), "E", "G")
    frame["prn"] = frame.system + frame.svid.astype(int).astype(str).str.zfill(2)
    frame["sv_identifier"] = frame.prn.map(mapping.sv_identifier)
    frame["antenna_family"] = frame.prn.map(mapping.antenna_type)
    assert frame.antenna_family.notna().all()
    frame["is_foc"] = frame.antenna_family.eq("GALILEO-2")
    frame["is_iov"] = frame.antenna_family.eq("GALILEO-1")
    frame["grap_applicable"] = frame.is_foc
    return frame


def load_patterns(mapping):
    print("Parsing existing GPS IIR, IIIA and IIF source products", flush=True)
    patterns = {"GPS_L1":parse_iirm_2d(GPS_DIR / "AppBAntennaPanelPatterns.pptx"), "GPS_L5":{}}
    for signal, band in [("GPS_L1","L1"),("GPS_L5","L5")]:
        patterns[signal].update(parse_iiia_2d(GPS_DIR, band))
    iif = parse_iif_patterns(GPS_DIR)
    for signal in patterns:
        patterns[signal].update(iif[signal])
        iii = {svn:p for svn,p in patterns[signal].items() if 74<=svn<=78}
        p0 = next(iter(iii.values()))
        gain = np.nanmedian(np.stack([p["gain"] for p in iii.values()]), axis=0)
        for identifier in mapping.loc[mapping.antenna_type.eq("BLOCK IIIA"), "sv_identifier"]:
            svn = int(identifier[1:])
            if svn not in patterns[signal]:
                patterns[signal][svn] = dict(theta=p0["theta"], phi=p0["phi"], gain=gain,
                    source=f"GPS III SVN74-78 median {signal} 2D proxy",
                    block_pattern_family="GPS_IIIA_same_block_median_proxy")
    grid = pd.read_csv(GRID, float_precision="round_trip")
    for signal in ["GAL_E1","GAL_E5a"]:
        p = grid.loc[grid.signal_name.eq(signal)].pivot(index="coelevation_deg", columns="azimuth_deg", values="eirp_dbw").sort_index().sort_index(axis=1)
        keep = p.columns.to_numpy(float)<360
        patterns[signal] = dict(theta=p.index.to_numpy(float), phi=p.columns.to_numpy(float)[keep], gain=p.to_numpy(float)[:,keep],
            source=str(GRID), block_pattern_family="Galileo_FOC_GRAP_reference")
    return patterns


def native_responses(raw, patterns):
    result = raw.copy()
    for c in ["nominal_tx_gain_db", "nominal_tx_eirp_dbw", "nominal_rx_gain_dbic", "tx_theta_min_deg", "tx_theta_max_deg", "rx_theta_max_deg", *DELTAS]:
        result[c] = np.nan
    result["reference_pattern_source"] = "unavailable"
    result["reference_pattern_family"] = "unavailable"
    result["reference_fallback"] = "unavailable"
    result["tx_block_proxy_used"] = False
    inventory = []
    for (signal, identifier), g in result.groupby(["signal_name","sv_identifier"], sort=True):
        gps = signal.startswith("GPS")
        svn = int(identifier[1:])
        pattern = patterns[signal].get(svn) if gps else patterns[signal]
        if pattern is None:
            continue
        idx = g.index
        theta, phi = g.tx_theta_body_deg.to_numpy(float), g.tx_phi_body_deg.to_numpy(float)
        nominal = bilinear_periodic(pattern, theta, phi)
        power = g.antenna_family.map(TX_POWER_DBW).to_numpy(float) if gps else np.zeros(len(g))
        result.loc[idx,"nominal_tx_eirp_dbw"] = nominal + power
        if gps:
            result.loc[idx,"nominal_tx_gain_db"] = nominal
        result.loc[idx,"tx_theta_min_deg"] = float(np.min(pattern["theta"]))
        result.loc[idx,"tx_theta_max_deg"] = float(np.max(pattern["theta"]))
        result.loc[idx,"reference_pattern_source"] = pattern["source"]
        result.loc[idx,"reference_pattern_family"] = pattern["block_pattern_family"]
        coverage = pattern.get("pattern_coverage", "satellite_specific_full_2D") if gps else "FOC_GRAP_constellation_reference"
        fallback = "none"
        if "same_block_median_proxy" in pattern["block_pattern_family"]:
            fallback = "GPS_IIIA_SVN74_78_median_for_later_vehicle"
            result.loc[idx,"tx_block_proxy_used"] = True
        elif "block_median_proxy" in coverage:
            fallback = "GPS_IIF_satellite_near_field_plus_boresight_scaled_block_median_far_field"
            result.loc[idx,"tx_block_proxy_used"] = theta>23.
        elif not gps and g.is_iov.all():
            fallback = "FOC_GRAP_applied_to_IOV_diagnostic_only_not_IOV_template"
        result.loc[idx,"reference_fallback"] = fallback
        for name,dt,dp in [(DELTAS[0],-1.,0.),(DELTAS[1],1.,0.),(DELTAS[2],0.,-5.),(DELTAS[3],0.,5.)]:
            shifted = bilinear_periodic(pattern, theta+dt, phi+dp)
            result.loc[idx,name] = shifted-nominal
        inventory.append(dict(signal_name=signal,sv_identifier=identifier,antenna_family=g.antenna_family.iloc[0],n_native=len(g),
            reference_pattern_source=pattern["source"],reference_pattern_family=pattern["block_pattern_family"],reference_fallback=fallback,
            coverage=coverage,theta_min=float(np.min(pattern["theta"])),theta_max=float(np.max(pattern["theta"])),
            n_theta=len(pattern["theta"]),n_phi=len(pattern["phi"]),phi_periodic=True))
    curve = pd.read_csv(RX, float_precision="round_trip")
    for signal,band in [("GPS_L1","L1/E1"),("GAL_E1","L1/E1"),("GPS_L5","L5/E5a"),("GAL_E5a","L5/E5a")]:
        m = result.signal_name.eq(signal)
        c = curve.loc[curve.band.eq(band)].sort_values("signed_elevation_deg")
        theta = result.loc[m,"rx_offboresight_spice_deg"].to_numpy(float)
        xp, yp = c.signed_elevation_deg.to_numpy(float), c.outer_envelope_gain_dB_axis.to_numpy(float)
        maxangle = min(float(xp.max()),float(-xp.min()),90.)
        def eval_rx(angle):
            value = np.maximum(np.interp(angle,xp,yp),np.interp(-angle,xp,yp))
            value[~np.isfinite(angle) | (angle<0.) | (angle>maxangle)] = np.nan
            return value
        nominal = eval_rx(theta)
        result.loc[m,"nominal_rx_gain_dbic"] = nominal
        result.loc[m,"rx_theta_max_deg"] = maxangle
        result.loc[m,DELTAS[4]] = eval_rx(theta-1.)-nominal
        result.loc[m,DELTAS[5]] = eval_rx(theta+1.)-nominal
    for c in DELTAS:
        result[c.replace("_delta_db","_supported")] = np.isfinite(result[c])
    result["response_supported"] = np.isfinite(result[DELTAS]).all(axis=1)
    return result, pd.DataFrame(inventory)


def align_conservative(native, bins, columns):
    """Original anchor interpolation, requiring finite actual bracketing points.

    Original endpoint holding is retained; no invalid shifted point is skipped.
    Exact source-time matches require only that source value, not its neighbour.
    """
    output = bins.copy()
    for c in columns:
        output[c] = np.nan
    output["anchor_uses_endpoint_hold"] = False
    output["anchor_bracket_span_s"] = np.nan
    output["anchor_tx_block_proxy_used"] = False
    for c in ["reference_pattern_source","reference_pattern_family","reference_fallback"]:
        output[c] = "unavailable"
    grouped = {key:g.sort_values("rx_utc_dt") for key,g in native.groupby(LINK,sort=False)}
    for key, target in output.groupby(LINK,sort=False):
        source = grouped.get(key)
        if source is None:
            continue
        times = source.rx_utc_dt.map(pd.Timestamp.timestamp).to_numpy(float)
        times, ui = np.unique(times,return_index=True)
        source = source.iloc[ui]
        tq = target.minute_utc.map(pd.Timestamp.timestamp).to_numpy(float)
        left = np.clip(np.searchsorted(times,tq,side="right")-1,0,len(times)-1)
        right = np.clip(np.searchsorted(times,tq,side="left"),0,len(times)-1)
        output.loc[target.index,"anchor_uses_endpoint_hold"] = (tq<times[0]) | (tq>times[-1])
        output.loc[target.index,"anchor_bracket_span_s"] = times[right]-times[left]
        b = source.tx_block_proxy_used.to_numpy(bool)
        output.loc[target.index,"anchor_tx_block_proxy_used"] = b[left] | b[right]
        for c in ["reference_pattern_source","reference_pattern_family","reference_fallback"]:
            assert source[c].nunique()==1, (key,c)
            output.loc[target.index,c] = source[c].iloc[0]
        for c in columns:
            values = source[c].to_numpy(float)
            valid = np.isfinite(values[left]) & np.isfinite(values[right])
            z = np.interp(tq,times,values)
            z[~valid] = np.nan
            output.loc[target.index,c] = z
    for c in DELTAS:
        output[c.replace("_delta_db","_supported")] = np.isfinite(output[c])
    output["d_tx_theta_db_per_deg"] = (output.theta_plus1_delta_db-output.theta_minus1_delta_db)/2.
    output["d_tx_phi_db_per_deg"] = (output.phi_plus5_delta_db-output.phi_minus5_delta_db)/10.
    output["d_rx_theta_db_per_deg"] = (output.rx_plus1_delta_db-output.rx_minus1_delta_db)/2.
    output["response_supported"] = np.isfinite(output[DELTAS]).all(axis=1)
    return output


def main():
    inputs=[SOURCE,CACHE,PHYSICAL,MAPPING,GRID,RX,ROOT/"script/build_cn0_attitude_2d_gain.py",ROOT/"script/gps_iif_antenna_patterns.py"]
    products=[GPS_DIR/"AppBAntennaPanelPatterns.pptx",*sorted(GPS_DIR.glob("GPS_III*Directivity.zip")),*sorted(GPS_DIR.glob("SSC PA Final Assess*.xlsx"))]
    before={str(p):sha(p) for p in inputs+products}
    mapping=pd.read_csv(MAPPING).set_index("prn")
    cols=["rx_utc","time_bin_gps_seconds",*LINK,"mission_phase","tx_theta_body_deg","tx_phi_body_deg","tx_gain_2d_db","tx_eirp_2d_dbw","tx_power_dbw","rx_offboresight_spice_deg","rx_gain_envelope_dbic","tx_pattern_source","tx_pattern_family","tx_pattern_coverage","tx_yaw_quality","tx_phi_alignment_quality"]
    raw=identify(pd.read_csv(SOURCE,usecols=cols,float_precision="round_trip"),mapping)
    raw["rx_utc_dt"]=pd.to_datetime(raw.rx_utc,utc=True)
    bins=joblib.load(CACHE).copy()
    bins.minute_utc=pd.to_datetime(bins.minute_utc,utc=True)
    bins=identify(bins,mapping)
    physical=pd.read_csv(PHYSICAL,usecols=KEYS+["tx_eirp_2d_dbw","tx_gain_2d_db","rx_gain_envelope_dbic","earth_blocked","moon_blocked"],float_precision="round_trip")
    physical.minute_utc=pd.to_datetime(physical.minute_utc,utc=True)
    bins=bins.merge(physical[KEYS+["earth_blocked","moon_blocked"]],on=KEYS,how="left",validate="one_to_one")
    assert len(bins)==17439 and not bins.duplicated(KEYS).any()
    patterns=load_patterns(mapping)
    native,inventory=native_responses(raw,patterns)
    qa=dict(native_rows=len(native),minute_rows=len(bins),nominal_native={
        "gps_tx_gain":compare(native.loc[native.system.eq("G"),"nominal_tx_gain_db"],native.loc[native.system.eq("G"),"tx_gain_2d_db"]),
        "all_tx_eirp":compare(native.nominal_tx_eirp_dbw,native.tx_eirp_2d_dbw),
        "rx_gain":compare(native.nominal_rx_gain_dbic,native.rx_gain_envelope_dbic)})
    print(json.dumps(qa,indent=2),flush=True)
    for v in qa["nominal_native"].values():
        assert v["finite_mask_mismatches"]==0 and v["max_abs_error"]<1e-8, v
    meta=[*KEYS,"system","mission_phase","evaluation_split","trend_training_eligible","antenna_family","prn","sv_identifier","is_foc","is_iov","grap_applicable","earth_blocked","moon_blocked"]
    numeric=["nominal_tx_gain_db","nominal_tx_eirp_dbw","nominal_rx_gain_dbic","tx_theta_min_deg","tx_theta_max_deg","rx_theta_max_deg",*DELTAS]
    aligned=align_conservative(native,bins[meta],numeric)
    # Independently verify nominal columns against original anchor function and cache.
    reference=bins[meta].copy()
    for c in numeric:
        reference[c]=np.nan
    reference=_interpolate_link_features_to_anchor(native,reference,numeric)
    qa["anchor_function_nominal"]={c:compare(aligned[c],reference[c]) for c in numeric[:3]}
    aligned=aligned.merge(physical.drop(columns=["earth_blocked","moon_blocked"]),on=KEYS,how="left",validate="one_to_one")
    qa["nominal_minute"]={c:compare(aligned[c],aligned[r]) for c,r in [("nominal_tx_gain_db","tx_gain_2d_db"),("nominal_tx_eirp_dbw","tx_eirp_2d_dbw"),("nominal_rx_gain_dbic","rx_gain_envelope_dbic")]}
    for v in [*qa["anchor_function_nominal"].values(),*qa["nominal_minute"].values()]:
        assert v["finite_mask_mismatches"]==0 and v["max_abs_error"]<1e-8,v
    # Input-only derivative construction: targets/residuals are never read here.
    aligned["nominal_tx_eirp_reproduction_error_db"]=aligned.nominal_tx_eirp_dbw-aligned.tx_eirp_2d_dbw
    aligned["nominal_rx_gain_reproduction_error_db"]=aligned.nominal_rx_gain_dbic-aligned.rx_gain_envelope_dbic
    aligned["reference_applicable"]=~aligned.is_iov
    aligned["unblocked"]=aligned.earth_blocked.eq(0)&aligned.moon_blocked.eq(0)
    aligned["primary_response_supported"]=aligned.response_supported&aligned.reference_applicable&aligned.unblocked
    domains=[]
    for (phase,signal,split),g in aligned.groupby(["mission_phase","signal_name","evaluation_split"]):
        domains.append(dict(mission_phase=phase,signal_name=signal,evaluation_split=split,n=len(g),n_response_supported=int(g.response_supported.sum()),n_primary_response_supported=int(g.primary_response_supported.sum()),n_iov=int(g.is_iov.sum()),n_block_proxy=int(g.anchor_tx_block_proxy_used.sum()),n_endpoint_hold=int(g.anchor_uses_endpoint_hold.sum()),**{c+"_invalid":int(g[c].isna().sum()) for c in DELTAS}))
    qa.update(n_response_supported=int(aligned.response_supported.sum()),n_primary_response_supported=int(aligned.primary_response_supported.sum()),n_iov=int(aligned.is_iov.sum()),
        unsupported_by_response={c:int(aligned[c].isna().sum()) for c in DELTAS},n_endpoint_hold=int(aligned.anchor_uses_endpoint_hold.sum()),
        input_hashes=before,inputs_unchanged=all(sha(Path(p))==h for p,h in before.items()),models_fitted=0,observed_targets_used=False,
        theta_shift_deg=1,phi_shift_deg=5,rx_shift_deg=1,nominal_geometry_unchanged=True,
        interpolation="original per-operation/signal/satellite anchor interpolation with endpoint holding; any nonfinite actual source bracket rejects a perturbed output",angles_clipped=False,
        derivative_units="dB/deg; theta and rx denominators 2, phi denominator 10",iov_policy="FOC reference retained only as flagged reproduction diagnostic; primary_response_supported excludes IOV")
    assert qa["inputs_unchanged"]
    save(native.drop(columns="rx_utc_dt"),"native_reference_responses.csv")
    save(aligned,"minute_reference_responses.csv")
    joblib.dump(aligned,OUT/"minute_reference_responses.joblib")
    save(inventory,"pattern_inventory.csv")
    save(pd.DataFrame(domains),"response_domain_summary.csv")
    (OUT/"reference_response_qa.json").write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in qa.items() if k!="input_hashes"},indent=2),flush=True)


if __name__=="__main__":
    main()
