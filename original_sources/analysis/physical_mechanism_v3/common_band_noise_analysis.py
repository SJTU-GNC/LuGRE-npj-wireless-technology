"""Same-band constellation contrasts and noise-equivalent diagnostics.

Exploratory analysis of saved outputs, not a new fit or causal attribution.
Pair satellites simultaneously and by receiver arrival direction, never by error.
"""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\月球导航\runtime_cache\python_deps")
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'phase_residual_analysis_v1/prepared/physical_minutes.csv'
PHASES = ['C', 'T', 'L', 'S']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unit(g):
    theta = np.deg2rad(g.rx_offboresight_spice_deg.to_numpy())
    phi = np.deg2rad(g.rx_azimuth_spice_deg.to_numpy())
    return np.column_stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def summary(g):
    v = g.delta_bias_raw_db
    byop = g.groupby('op').delta_bias_raw_db.median()
    return dict(n_pairs=len(g), n_ops=g.op.nunique(), median_db=v.median(), p05_db=v.quantile(.05),
                p95_db=v.quantile(.95), equal_op_median_db=byop.median(),
                positive_op_fraction=float((byop>0).mean()),
                rx_separation_median_deg=g.rx_separation_deg.median(),
                rx_gain_difference_median_db=g.delta_rx_gain_db.median(),
                noise_psd_max_abs_difference_db=g.delta_noise_psd_db.abs().max())

def main():
    original_hash = sha(SOURCE)
    d = pd.read_csv(SOURCE, dtype={'svid': str}, low_memory=False)
    eligible = d.trend_training_eligible.astype(str).str.lower().isin(['true','1','1.0'])
    d = d.loc[eligible & d.earth_blocked.eq(0) & d.moon_blocked.eq(0)
              & np.isfinite(d.cn0_trend_target_dbhz) & np.isfinite(d.cn0_physics_trend_dbhz)].copy()
    assert len(d) == 15734
    d['bias_raw_db'] = d.cn0_constellation_physics_proxy_dbhz-d.cn0_dbhz_mean
    d['bias_trend_db'] = d.cn0_physics_trend_dbhz-d.cn0_trend_target_dbhz
    records=[]
    # Constants are diagnostic cutoffs fixed before examining pair outcomes.
    for caliper in [.5, 1., 2.]:
        for band, signals in [('L1_E1', ('GPS_L1','GAL_E1')), ('L5_E5a', ('GPS_L5','GAL_E5a'))]:
            for (op, epoch), g in d[d.signal_name.isin(signals)].groupby(['op','source_bin_gps_seconds']):
                a = g[g.signal_name.eq(signals[0])].sort_values('svid').reset_index(drop=True)
                b = g[g.signal_name.eq(signals[1])].sort_values('svid').reset_index(drop=True)
                if a.empty or b.empty: continue
                dist = np.rad2deg(np.arccos(np.clip(unit(a) @ unit(b).T, -1, 1)))
                # Large penalty makes matching maximize admissible pair count first.
                cost = np.where(dist<=caliper, dist, 10000.)
                ia, ib = linear_sum_assignment(cost)
                for i,j in zip(ia,ib):
                    if dist[i,j]>caliper: continue
                    x,y = a.iloc[i],b.iloc[j]
                    records.append(dict(caliper_deg=caliper,band=band,op=op,phase=x.mission_phase,
                        epoch=epoch,gps_svid=x.svid,gal_svid=y.svid,rx_separation_deg=dist[i,j],
                        delta_bias_raw_db=x.bias_raw_db-y.bias_raw_db,
                        delta_bias_trend_db=x.bias_trend_db-y.bias_trend_db,
                        gps_bias_raw_db=x.bias_raw_db,gal_bias_raw_db=y.bias_raw_db,
                        delta_rx_gain_db=x.rx_gain_envelope_dbic-y.rx_gain_envelope_dbic,
                        delta_noise_psd_db=x.budget_noise_psd_dbw_hz-y.budget_noise_psd_dbw_hz,
                        delta_range_km=x.geometric_range_km-y.geometric_range_km,
                        gps_rx_angle_deg=x.rx_offboresight_spice_deg,gal_rx_angle_deg=y.rx_offboresight_spice_deg))
    pairs = pd.DataFrame(records)
    pairs.to_csv(OUT/'common_band_pairs.csv', index=False, encoding='utf-8-sig')
    summaries=[]
    for (caliper, band), group in pairs.groupby(['caliper_deg','band']):
        for phase in ['ALL']+PHASES:
            g=group if phase=='ALL' else group[group.phase.eq(phase)]
            if not g.empty: summaries.append(dict(caliper_deg=caliper,band=band,phase=phase,**summary(g)))
    s=pd.DataFrame(summaries)
    s.to_csv(OUT/'common_band_summary.csv',index=False,encoding='utf-8-sig')
    op=[]
    for (c,b,o,p),g in pairs.groupby(['caliper_deg','band','op','phase']):
        op.append(dict(caliper_deg=c,band=b,op=o,phase=p,**summary(g)))
    pd.DataFrame(op).to_csv(OUT/'common_band_op.csv',index=False,encoding='utf-8-sig')
    # Require both frequencies on both matched satellites. The four-way contrast
    # cancels any band-common receive/noise offset at this epoch; arbitrary
    # satellite-specific propagation and signal-estimator biases can remain.
    lookup=d.set_index(['op','source_bin_gps_seconds','signal_name','svid'])
    quad=[]
    for row in pairs[pairs.band.eq('L1_E1')].itertuples(index=False):
        keys=[(row.op,row.epoch,'GPS_L1',row.gps_svid),
              (row.op,row.epoch,'GPS_L5',row.gps_svid),
              (row.op,row.epoch,'GAL_E1',row.gal_svid),
              (row.op,row.epoch,'GAL_E5a',row.gal_svid)]
        if not all(k in lookup.index for k in keys): continue
        gh,gl,eh,el=[lookup.loc[k] for k in keys]
        dg=gh.bias_raw_db-gl.bias_raw_db; de=eh.bias_raw_db-el.bias_raw_db
        quad.append(dict(caliper_deg=row.caliper_deg,op=row.op,phase=row.phase,epoch=row.epoch,
                         gps_svid=row.gps_svid,gal_svid=row.gal_svid,rx_separation_deg=row.rx_separation_deg,
                         gps_high_minus_low_db=dg,gal_high_minus_low_db=de,double_difference_db=dg-de))
    quad=pd.DataFrame(quad)
    quad.to_csv(OUT/'common_band_fourway_pairs.csv',index=False,encoding='utf-8-sig')
    qsum=[]
    for c,g in quad.groupby('caliper_deg'):
        for phase in ['ALL']+PHASES:
            z=g if phase=='ALL' else g[g.phase.eq(phase)]
            if z.empty: continue
            qsum.append(dict(caliper_deg=c,phase=phase,n=len(z),n_ops=z.op.nunique(),
                  gps_high_minus_low_median_db=z.gps_high_minus_low_db.median(),
                  gal_high_minus_low_median_db=z.gal_high_minus_low_db.median(),
                  double_difference_median_db=z.double_difference_db.median(),
                  double_difference_p05_db=z.double_difference_db.quantile(.05),
                  double_difference_p95_db=z.double_difference_db.quantile(.95),
                  equal_op_median_db=z.groupby('op').double_difference_db.median().median()))
    pd.DataFrame(qsum).to_csv(OUT/'common_band_fourway_summary.csv',index=False,encoding='utf-8-sig')
    # If the entire observed discrepancy were a noise error, this is the required
    # equivalent Tsys. It is not an estimate of the flight hardware temperature.
    eq=[]
    for signal,g in d.groupby('signal_name'):
        for phase in ['ALL']+PHASES:
            z=g if phase=='ALL' else g[g.mission_phase.eq(phase)]
            if z.empty: continue
            bias=float(z.bias_raw_db.median()); nominal=float(z.system_noise_temperature_k.median())
            eq.append(dict(signal=signal,phase=phase,n=len(z),median_bias_db=bias,nominal_tsys_k=nominal,
                           required_noise_ratio=10**(bias/10),equivalent_tsys_k=nominal*10**(bias/10)))
    pd.DataFrame(eq).to_csv(OUT/'noise_equivalent_temperatures.csv',index=False,encoding='utf-8-sig')
    scale=[]
    for tnom in [182.,231.]:
        for extra in [30.,50.,100.,300.]:
            scale.append(dict(nominal_tsys_k=tnom,assumed_added_system_noise_k=extra,
                              delta_noise_db=10*np.log10((tnom+extra)/tnom)))
    pd.DataFrame(scale).to_csv(OUT/'noise_added_temperature_sensitivity.csv',index=False,encoding='utf-8-sig')
    atmosphere=[]
    for signal,g in d.groupby('signal_name'):
        for case,mask in [('above_1000km',g.earth_grazing_altitude_km.gt(1000)),
                          ('50_to_1000km',g.earth_grazing_altitude_km.between(50,1000)),
                          ('0_to_50km',g.earth_grazing_altitude_km.between(0,50,inclusive='left'))]:
            z=g[mask]
            if z.empty: continue
            atmosphere.append(dict(signal=signal,ray_case=case,n=len(z),n_ops=z.op.nunique(),
                    median_bias_raw_db=z.bias_raw_db.median(),p05_bias_raw_db=z.bias_raw_db.quantile(.05),
                    p95_bias_raw_db=z.bias_raw_db.quantile(.95),
                    ion_budget_max_db=z.l_ion_abs_budget_db.max(),gas_budget_max_db=z.l_gas_abs_budget_db.max(),
                    min_grazing_altitude_km=z.earth_grazing_altitude_km.min()))
    pd.DataFrame(atmosphere).to_csv(OUT/'atmosphere_path_controls.csv',index=False,encoding='utf-8-sig')
    # Test sign predicted by a non-negative missing absorption amplitude scaling
    # as f^-2 on a common ray. Does not cover scintillation/multipath/refraction.
    pp=pd.read_csv(OUT.parent/'distance_mechanism_v2/pair_epoch_differences.csv',low_memory=False)
    pp=pp[pp.scope.eq('primary_nonocculted')]
    ion=[]
    ratio=(1575.42/1176.45)**2
    for pair,g in pp.groupby('pair'):
        for phase in ['ALL']+PHASES:
            z=g if phase=='ALL' else g[g.mission_phase.eq(phase)]
            if z.empty: continue
            difference=float(z.delta_bias_raw_db.median())
            ion.append(dict(pair=pair,phase=phase,n=len(z),observed_high_minus_low_median_db=difference,
                    positive_difference_fraction=float(z.delta_bias_raw_db.gt(0).mean()),
                    low_to_high_absorption_ratio=ratio,
                    equivalent_missing_high_absorption_db=difference/(1-ratio),
                    sign_compatible_with_nonnegative_missing_f_inverse_square_absorption=bool(difference<=0)))
    pd.DataFrame(ion).to_csv(OUT/'ionosphere_absorption_sign_test.csv',index=False,encoding='utf-8-sig')
    qa=dict(source=str(SOURCE),hash_before=original_hash,hash_after=sha(SOURCE),primary_n=len(d),
            sign='baseline minus observation; cross-constellation differences GPS minus Galileo',
            pairing='within OP and native GPS minute; maximum cardinality then minimal angular separation; no outcome matching',
            calipers_deg=[.5,1.,2.],raw_target='unfiltered one-minute mean',
            distinct_rays=True,interpretation='Shared band noise cancels. Transmitter, modulation/CN0 estimator, differential pattern and propagation terms do not.',
            no_manuscript_or_original_model_changes=True)
    assert qa['hash_before']==qa['hash_after']
    assert pairs.delta_noise_psd_db.abs().max()<1e-10
    assert not pairs.duplicated(['caliper_deg','band','op','epoch','gps_svid']).any()
    assert not pairs.duplicated(['caliper_deg','band','op','epoch','gal_svid']).any()
    (OUT/'common_band_qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    print(s[s.phase.eq('ALL')].to_string(index=False))
    print(pd.DataFrame(eq).query("phase == 'ALL'").to_string(index=False))
    print(pd.DataFrame(qsum).query("phase == 'ALL'").to_string(index=False))

if __name__=='__main__': main()
