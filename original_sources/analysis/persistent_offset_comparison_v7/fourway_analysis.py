"""Matched four-signal contrasts, with no assumed signal-power adjustment.

Uses existing geometry-only matches and the original saved baseline/targets.
Positive bias means physical baseline exceeds the observed value, in dB.
"""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SOURCE = BASE / 'phase_residual_analysis_v1/prepared/physical_minutes.csv'
MATCHES = BASE / 'physical_mechanism_v3/common_band_fourway_pairs.csv'
BETA_SOURCE = BASE / 'persistent_offset_attribution_v6/joint_fit_signal_summary.csv'
SIGNALS = ['GPS_L1', 'GPS_L5', 'GAL_E1', 'GAL_E5a']
PHASES = ['C', 'T', 'L', 'S']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(g):
    result = {'n_pairs': len(g), 'n_operations': g.op.nunique(),
              'n_epochs': g[['op', 'epoch']].drop_duplicates().shape[0]}
    columns = [f'{kind}_{metric}_db' for kind in ['trend', 'raw', 'trend_after_beta']
               for metric in ['gps_high_low', 'gal_high_low', 'high_gps_gal',
                              'low_gps_gal', 'double_difference']]
    for col in columns:
        vals = g[col]
        result[col + '_median'] = vals.median()
        result[col + '_p05'] = vals.quantile(.05)
        result[col + '_p95'] = vals.quantile(.95)
        op_medians = g.groupby('op')[col].median()
        result[col + '_median_of_operation_medians'] = op_medians.median()
    for signal in SIGNALS:
        for kind in ['raw', 'trend']:
            col = f'{kind}_{signal}_bias_db'
            result[col + '_median'] = g[col].median()
    result['median_time_spread_s_max'] = g.median_time_spread_s.max()
    result['direction_separation_deg_max'] = g.rx_separation_deg.max()
    return result


def main():
    hashes = {str(p): sha(p) for p in [SOURCE, MATCHES, BETA_SOURCE]}
    beta_table = pd.read_csv(BETA_SOURCE)
    beta = beta_table[beta_table.rx_scenario.eq('two_side_min')].set_index('signal_name').beta_original_db.to_dict()
    assert len(beta) == 4
    full = pd.read_csv(SOURCE, dtype={'svid': str}, low_memory=False)
    eligible = full.trend_training_eligible.astype(str).str.lower().isin(['true', '1', '1.0'])
    data = full[eligible & full.earth_blocked.eq(0) & full.moon_blocked.eq(0)
                & np.isfinite(full.cn0_physics_trend_dbhz)
                & np.isfinite(full.cn0_trend_target_dbhz)].copy()
    assert len(data) == 15734
    data['raw_bias_db'] = data.cn0_constellation_physics_proxy_dbhz - data.cn0_dbhz_mean
    data['trend_bias_db'] = data.cn0_physics_trend_dbhz - data.cn0_trend_target_dbhz
    keys = ['op', 'source_bin_gps_seconds', 'signal_name', 'svid']
    assert not data.duplicated(keys).any()
    lookup = data.set_index(keys)
    matches = pd.read_csv(MATCHES, dtype={'gps_svid': str, 'gal_svid': str})
    rows = []
    max_old_difference = 0.0
    for m in matches.itertuples(index=False):
        values = [lookup.loc[(m.op, m.epoch, s, m.gps_svid if s.startswith('GPS') else m.gal_svid)]
                  for s in SIGNALS]
        assert all(x.mission_phase == m.phase for x in values)
        r = dict(caliper_deg=m.caliper_deg, op=m.op, phase=m.phase, epoch=m.epoch,
                 gps_svid=m.gps_svid, gal_svid=m.gal_svid, rx_separation_deg=m.rx_separation_deg)
        times = pd.to_datetime([x.observation_time_median_utc for x in values], utc=True)
        r['median_time_spread_s'] = (times.max()-times.min()).total_seconds()
        for kind in ['raw', 'trend', 'trend_after_beta']:
            base_kind = 'trend' if kind == 'trend_after_beta' else kind
            b = [float(x[base_kind + '_bias_db']) + (beta[s] if kind == 'trend_after_beta' else 0.0)
                 for s, x in zip(SIGNALS, values)]
            for signal, bias in zip(SIGNALS, b):
                r[f'{kind}_{signal}_bias_db'] = bias
            r[f'{kind}_gps_high_low_db'] = b[0]-b[1]
            r[f'{kind}_gal_high_low_db'] = b[2]-b[3]
            r[f'{kind}_high_gps_gal_db'] = b[0]-b[2]
            r[f'{kind}_low_gps_gal_db'] = b[1]-b[3]
            r[f'{kind}_double_difference_db'] = (b[0]-b[1])-(b[2]-b[3])
            assert abs(r[f'{kind}_double_difference_db'] -
                       (r[f'{kind}_high_gps_gal_db']-r[f'{kind}_low_gps_gal_db'])) < 1e-12
        max_old_difference = max(max_old_difference, abs(r['raw_double_difference_db']-m.double_difference_db))
        # Same-band nominal noise is common. The physical RX response need not be.
        r['nominal_noise_double_difference_db'] = (
            values[0].budget_noise_psd_dbw_hz-values[1].budget_noise_psd_dbw_hz
            -values[2].budget_noise_psd_dbw_hz+values[3].budget_noise_psd_dbw_hz)
        rows.append(r)
    pairs = pd.DataFrame(rows)
    ops = full[['op', 'mission_phase']].drop_duplicates()
    assert ops.op.nunique() == 20
    summaries, robust = [], []
    for c, g in pairs.groupby('caliper_deg'):
        for phase in ['ALL'] + PHASES:
            h = g if phase == 'ALL' else g[g.phase.eq(phase)]
            summaries.append(dict(caliper_deg=c, level='phase', stratum=phase, **describe(h)))
        for o in ops.itertuples(index=False):
            h = g[g.op.eq(o.op)]
            summaries.append(dict(caliper_deg=c, level='operation', stratum=o.op,
                                  mission_phase=o.mission_phase, **describe(h)))
        for omitted in g.op.unique():
            h = g[~g.op.eq(omitted)]
            robust.append(dict(caliper_deg=c, omitted_operation=omitted, **describe(h)))
    assert max_old_difference < 1e-12
    assert pairs.nominal_noise_double_difference_db.abs().max() < 1e-12
    summary = pd.DataFrame(summaries)
    pairs.to_csv(OUT/'fourway_pairs.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUT/'fourway_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(robust).to_csv(OUT/'fourway_leave_operation_out.csv', index=False, encoding='utf-8-sig')
    qa = dict(original_nonocculted_n=len(data), n_pairs=len(pairs),
              matched_counts=pairs.groupby('caliper_deg').size().to_dict(),
              original_raw_double_difference_max_error_db=max_old_difference,
              noise_double_difference_max_abs_db=float(pairs.nominal_noise_double_difference_db.abs().max()),
              hashes_before=hashes, hashes_after={str(p): sha(p) for p in [SOURCE, MATCHES, BETA_SOURCE]},
              original_final_fit_beta_db=beta,
              sample_note='Original high-band geometry-only matches requiring both lower-band signals; not a new optimal four-way matching.',
              interpretation='Raw double difference tests sufficiency of a band-common additive error, not causal partition of absolute bias. Trend contrasts describe independently smoothed outputs; raw budget supports physical cancellation. Original final-fit beta correction is descriptive on mixed fitting/held-out samples.',
              original_models_data_manuscript_unchanged=True)
    assert qa['hashes_before'] == qa['hashes_after']
    (OUT/'fourway_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.loc[summary.stratum.eq('ALL'), ['caliper_deg','n_pairs',
           'trend_high_gps_gal_db_median','trend_low_gps_gal_db_median',
           'trend_double_difference_db_median','raw_double_difference_db_median']].to_string(index=False))


if __name__ == '__main__':
    main()
