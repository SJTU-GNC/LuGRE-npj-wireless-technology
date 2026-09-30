"""Prespecified geometry-only cross-phase matching of identical satellite signals.

Only this analysis directory receives outputs. No model fits or input edits.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parent.parent
PHYSICAL = ANALYSIS / 'phase_residual_analysis_v1/prepared/physical_minutes.csv'
ATTRIB = ANALYSIS / 'phase_compensation_attribution_v4/attrib_rows.csv'
KEYS = ['op', 'system', 'signal_name', 'svid', 'source_bin_gps_seconds']
PHASES = ['C', 'T', 'L', 'S']
PHASE_PAIRS = list(combinations(PHASES, 2))
SCENARIOS = {'strict': (1., 10., .5), 'primary': (2., 15., 1.), 'loose': (3., 30., 2.)}
METRICS = ['baseline_raw', 'after_beta_raw', 'hgb_raw', 'final_raw',
           'baseline_trend', 'after_beta_trend', 'hgb_trend', 'final_trend']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def opnum(op):
    return int(str(op).replace('OP', ''))


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.15g')


def summary(g):
    """Return paired statistics and OP-pair balanced counterparts."""
    ans = {'n_pairs': len(g), 'n_satellites': g.svid.nunique() if len(g) else 0,
           'n_op_pairs': g.op_pair.nunique() if len(g) else 0,
           'n_ops_a': g.op_a.nunique() if len(g) else 0,
           'n_ops_b': g.op_b.nunique() if len(g) else 0,
           'operations_a': ';'.join(sorted(g.op_a.unique(), key=opnum)) if len(g) else '',
           'operations_b': ';'.join(sorted(g.op_b.unique(), key=opnum)) if len(g) else ''}
    if len(g):
        ans['largest_op_pair_fraction'] = g.op_pair.value_counts().max()/len(g)
        for f in ['theta_difference_deg', 'phi_difference_deg', 'rx_difference_deg',
                  'rx_azimuth_difference_deg', 'relative_range_difference']:
            ans[f + '_median'] = g[f].median()
            ans[f + '_max'] = g[f].max()
        op_balanced = g.groupby('op_pair')[[m + '_difference_db' for m in METRICS]].median()
        sat_balanced = g.groupby('svid')[[m + '_difference_db' for m in METRICS]].median()
        for m in METRICS:
            v = g[m+'_difference_db']
            ans[m+'_pooled_median_db'] = v.median()
            ans[m+'_pooled_p05_db'] = v.quantile(.05)
            ans[m+'_pooled_p95_db'] = v.quantile(.95)
            ans[m+'_op_pair_equal_median_db'] = op_balanced[m+'_difference_db'].median()
            ans[m+'_op_pair_equal_mean_db'] = op_balanced[m+'_difference_db'].mean()
            ans[m+'_satellite_equal_median_db'] = sat_balanced[m+'_difference_db'].median()
        ans['diagnostic_support'] = ('single_op_pair' if ans['n_op_pairs'] == 1 else
                                     'few_pairs' if len(g) < 10 else 'multiple_op_pairs')
    else:
        ans['diagnostic_support'] = 'no_matched_support'
    return ans


def match(a, b, tolerances, range_limit):
    """Geometry-only round-robin greedy matching across candidate OP pairs."""
    ta, pa, ra = tolerances
    if a.empty or b.empty:
        return [], 0, 0
    dt = np.abs(a.tx_theta_body_deg.to_numpy()[:, None] - b.tx_theta_body_deg.to_numpy()[None, :])
    dp = np.abs((a.tx_phi_body_deg.to_numpy()[:, None] - b.tx_phi_body_deg.to_numpy()[None, :] + 180.) % 360. - 180.)
    dr = np.abs(a.rx_offboresight_spice_deg.to_numpy()[:, None] - b.rx_offboresight_spice_deg.to_numpy()[None, :])
    rg_a, rg_b = a.geometric_range_km.to_numpy()[:, None], b.geometric_range_km.to_numpy()[None, :]
    relrange = 2*np.abs(rg_a-rg_b)/(rg_a+rg_b)
    angular = (dt <= ta+1e-10) & (dp <= pa+1e-10) & (dr <= ra+1e-10)
    valid = angular & ((relrange <= .1+1e-12) if range_limit else True)
    ii, jj = np.nonzero(valid)
    if not len(ii):
        return [], int(angular.sum()), 0
    score = (dt[ii, jj]/ta)**2 + (dp[ii, jj]/pa)**2 + (dr[ii, jj]/ra)**2
    ai, bj = a.index.to_numpy(), b.index.to_numpy()
    opa, opb = a.op.to_numpy(), b.op.to_numpy()
    queues = defaultdict(list)
    for i, j, cost in zip(ii, jj, score):
        queues[(opa[i], opb[j])].append((float(cost), int(ai[i]), int(bj[j]),
                                       float(dt[i,j]), float(dp[i,j]), float(dr[i,j]), float(relrange[i,j])))
    for key in queues:
        queues[key].sort(key=lambda z: (z[0], z[1], z[2]))
    order = sorted(queues, key=lambda x: (opnum(x[0]), opnum(x[1])))
    pointers = {k: 0 for k in order}
    used_a, used_b, accepted = set(), set(), []
    while True:
        added = False
        for key in order:
            q, k = queues[key], pointers[key]
            while k < len(q) and (q[k][1] in used_a or q[k][2] in used_b):
                k += 1
            pointers[key] = k
            if k == len(q):
                continue
            z = q[k]
            pointers[key] += 1
            accepted.append(z)
            used_a.add(z[1]); used_b.add(z[2]); added = True
        if not added:
            break
    return accepted, int(angular.sum()), int(valid.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [PHYSICAL, ATTRIB, OUT/'PLAN.md']
    hashes = {str(p): sha(p) for p in source_paths}
    pcols = KEYS + ['minute_utc', 'mission_phase', 'evaluation_split', 'trend_training_eligible',
        'earth_blocked', 'moon_blocked', 'tx_theta_body_deg', 'tx_phi_body_deg',
        'rx_offboresight_spice_deg', 'rx_azimuth_spice_deg', 'geometric_range_km',
        'earth_limb_margin_deg', 'moon_limb_margin_deg', 'earth_grazing_altitude_km',
        'cn0_dbhz_mean', 'cn0_constellation_physics_proxy_dbhz',
        'cn0_physics_trend_dbhz', 'cn0_observed_trend_dbhz']
    p = pd.read_csv(PHYSICAL, usecols=pcols, float_precision='round_trip')
    acols = KEYS + ['reference', 'beta_db', 'delta_raw_db', 'delta_eval_db', 'residual_after_beta_db']
    a = pd.read_csv(ATTRIB, usecols=acols, float_precision='round_trip')
    a = a.loc[a.reference.eq('empirical_fit32')].copy()
    assert not p.duplicated(KEYS).any() and not a.duplicated(KEYS).any()
    assert len(p)==17439 and len(a)==15806
    d = p.merge(a, on=KEYS, how='inner', validate='one_to_one')
    assert len(d)==15806 and d.trend_training_eligible.eq(True).all()
    before_occultation = len(d)
    d = d.loc[~d.earth_blocked.astype(bool) & ~d.moon_blocked.astype(bool)].copy()
    after_occultation = len(d)
    required = ['tx_theta_body_deg','tx_phi_body_deg','rx_offboresight_spice_deg','geometric_range_km',
                'cn0_dbhz_mean','cn0_constellation_physics_proxy_dbhz','cn0_physics_trend_dbhz',
                'cn0_observed_trend_dbhz','beta_db','delta_raw_db','delta_eval_db']
    d = d.loc[np.isfinite(d[required]).all(axis=1) & d.geometric_range_km.gt(0)].sort_values(KEYS).reset_index(drop=True)
    d['row_id'] = d.index
    d['baseline_raw_db'] = d.cn0_constellation_physics_proxy_dbhz - d.cn0_dbhz_mean
    d['baseline_trend_db'] = d.cn0_physics_trend_dbhz - d.cn0_observed_trend_dbhz
    d['hgb_raw_db'], d['hgb_trend_db'] = d.delta_raw_db, d.delta_eval_db
    for layer in ['raw', 'trend']:
        d['after_beta_'+layer+'_db'] = d['baseline_'+layer+'_db'] + d.beta_db
        d['final_'+layer+'_db'] = d['after_beta_'+layer+'_db'] + d['hgb_'+layer+'_db']
    assert np.max(np.abs(d.after_beta_trend_db + d.residual_after_beta_db)) < 1e-9
    assert d.groupby('signal_name').beta_db.nunique().max() == 1
    save(d, 'eligible_endpoint_data.csv')
    cover = d.groupby(['signal_name', 'svid', 'mission_phase']).agg(n=('row_id','size'),n_operations=('op','nunique'),
        operations=('op', lambda x: ';'.join(sorted(set(x),key=opnum)))).reset_index()
    save(cover, 'source_coverage.csv')
    pairrows, coverage = [], []
    for scenario, tol in SCENARIOS.items():
        for range_limit in [False, True]:
            range_control = 'none' if not range_limit else 'relative_10pct'
            for (signal, sv), g in d.groupby(['signal_name', 'svid'], sort=True):
                for phase_a, phase_b in PHASE_PAIRS:
                    aa, bb = g.loc[g.mission_phase.eq(phase_a)], g.loc[g.mission_phase.eq(phase_b)]
                    selected, ncand_angles, ncand = match(aa, bb, tol, range_limit)
                    context = dict(scenario=scenario, range_control=range_control, phase_pair=phase_a+'_'+phase_b,
                                   phase_a=phase_a, phase_b=phase_b, signal_name=signal, svid=int(sv))
                    coverage.append(context | dict(n_available_a=len(aa),n_available_b=len(bb),
                        n_ops_available_a=aa.op.nunique(),n_ops_available_b=bb.op.nunique(),
                        n_angular_candidates=ncand_angles,n_candidates_after_range=ncand,n_matched=len(selected)))
                    for cost, ia, ib, dt, dp, dr, rg in selected:
                        x, y = d.loc[ia], d.loc[ib]
                        out = context | dict(row_id_a=ia,row_id_b=ib,op_a=x.op,op_b=y.op,op_pair=x.op+'__'+y.op,
                            minute_utc_a=x.minute_utc,minute_utc_b=y.minute_utc,
                            source_bin_gps_seconds_a=x.source_bin_gps_seconds,source_bin_gps_seconds_b=y.source_bin_gps_seconds,
                            evaluation_split_a=x.evaluation_split,evaluation_split_b=y.evaluation_split,
                            geometry_cost=cost,theta_difference_deg=dt,phi_difference_deg=dp,rx_difference_deg=dr,
                            rx_azimuth_difference_deg=abs((x.rx_azimuth_spice_deg-y.rx_azimuth_spice_deg+180)%360-180),
                            relative_range_difference=rg,range_a_km=x.geometric_range_km,range_b_km=y.geometric_range_km,
                            beta_a_db=x.beta_db,beta_b_db=y.beta_db,beta_difference_db=y.beta_db-x.beta_db)
                        for field in ['tx_theta_body_deg','tx_phi_body_deg','rx_offboresight_spice_deg',
                                      'rx_azimuth_spice_deg','earth_grazing_altitude_km','earth_limb_margin_deg','moon_limb_margin_deg']:
                            out[field+'_a'], out[field+'_b'] = x[field], y[field]
                        for m in METRICS:
                            out[m+'_a_db'], out[m+'_b_db'] = x[m+'_db'],y[m+'_db']
                            out[m+'_difference_db'] = y[m+'_db'] - x[m+'_db']
                        pairrows.append(out)
            print(scenario,range_control,'done',flush=True)
    pairs = pd.DataFrame(pairrows)
    save(pairs,'matched_pairs.csv')
    save(pd.DataFrame(coverage),'matching_coverage.csv')
    rows, satrows, oprows, deletions = [], [], [], []
    for scenario in SCENARIOS:
        for rc in ['none','relative_10pct']:
            for pa,pb in PHASE_PAIRS:
                for signal in sorted(d.signal_name.unique()):
                    ctx=dict(scenario=scenario,range_control=rc,phase_pair=pa+'_'+pb,signal_name=signal)
                    g=pairs.loc[pairs.scenario.eq(scenario)&pairs.range_control.eq(rc)&pairs.phase_pair.eq(pa+'_'+pb)&pairs.signal_name.eq(signal)]
                    rows.append(ctx | summary(g))
                    for sv,h in g.groupby('svid'):
                        satrows.append(ctx|dict(svid=int(sv))|summary(h))
                    for opp,h in g.groupby('op_pair'):
                        oprows.append(ctx|dict(op_pair=opp)|summary(h))
                    if g.op_pair.nunique()>1:
                        for op in sorted(set(g.op_a)|set(g.op_b),key=opnum):
                            h=g.loc[~g.op_a.eq(op)&~g.op_b.eq(op)]
                            deletions.append(ctx|dict(omitted_operation=op)|summary(h))
    save(pd.DataFrame(rows),'phase_signal_summary.csv')
    save(pd.DataFrame(satrows),'satellite_summary.csv')
    save(pd.DataFrame(oprows),'operation_pair_summary.csv')
    save(pd.DataFrame(deletions),'delete_one_operation_summary.csv')
    # Independent checks against endpoints and geometric definitions.
    grouped=pairs.groupby(['scenario','range_control','phase_pair','signal_name','svid'])
    assert all(not g.row_id_a.duplicated().any() and not g.row_id_b.duplicated().any() for _,g in grouped)
    for sc,tol in SCENARIOS.items():
        q=pairs.loc[pairs.scenario.eq(sc)]
        for field,limit in zip(['theta_difference_deg','phi_difference_deg','rx_difference_deg'],tol):
            assert q[field].max()<=limit+1e-9
    assert pairs.loc[pairs.range_control.eq('relative_10pct'),'relative_range_difference'].max()<=.1+1e-9
    assert np.max(np.abs(pairs.beta_difference_db))==0
    closure={}
    for layer in ['raw','trend']:
        err=pairs['final_'+layer+'_difference_db']-pairs['baseline_'+layer+'_difference_db']-pairs.beta_difference_db-pairs['hgb_'+layer+'_difference_db']
        closure[layer]=float(np.abs(err).max())
        assert closure[layer]<1e-9
        assert np.max(np.abs(pairs['after_beta_'+layer+'_difference_db']-pairs['baseline_'+layer+'_difference_db']))<1e-9
    for _,r in pairs.sample(min(100,len(pairs)),random_state=73).iterrows():
        aa,bb=d.loc[int(r.row_id_a)],d.loc[int(r.row_id_b)]
        assert aa.signal_name==bb.signal_name==r.signal_name and aa.svid==bb.svid==r.svid
        assert aa.mission_phase==r.phase_a and bb.mission_phase==r.phase_b
        phidiff=min(abs(aa.tx_phi_body_deg-bb.tx_phi_body_deg)%360,360-abs(aa.tx_phi_body_deg-bb.tx_phi_body_deg)%360)
        assert abs(phidiff-r.phi_difference_deg)<1e-9
    qa=dict(source_rows=len(p),eligible_rows=before_occultation,nonocculted_rows=after_occultation,
            complete_geometry_rows=len(d),n_saved_matched_pairs=len(pairs),maximum_closure_error_db=closure,
            beta_difference_is_zero=True,one_to_one_within_scenario_phasepair_satellite=True,
            periodic_phi_wrap_example_degrees=float(abs((359-1+180)%360-180)),
            residual_sign='baseline minus observed; positive prediction bias; pair differences later minus earlier phase',
            source_hashes_before=hashes,source_hashes_after={str(p):sha(p) for p in source_paths})
    qa['sources_unchanged']=qa['source_hashes_before']==qa['source_hashes_after']
    assert qa['sources_unchanged']
    (OUT/'QA.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    r=pd.DataFrame(rows)
    print(r.loc[r.scenario.eq('primary'),['range_control','phase_pair','signal_name','n_pairs','n_op_pairs',
       'baseline_trend_op_pair_equal_median_db','hgb_trend_op_pair_equal_median_db','final_trend_op_pair_equal_median_db']].to_string(index=False))
    print(json.dumps({k:v for k,v in qa.items() if 'hash' not in k},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
