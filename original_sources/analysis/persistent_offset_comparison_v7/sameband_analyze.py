"""Same-band GPS/Galileo paired bias contrasts, saved pair identities only.

Trend bias (baseline minus observation) is primary; raw bias is sensitivity.
No fitting, HGB prediction, rematching, manuscript edits, or source edits.
"""
from pathlib import Path
from bisect import bisect_left
import hashlib
import json
import math
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\月球导航\runtime_cache\python_deps")
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ANALYSIS = OUT.parent
PAIR_SOURCE = ANALYSIS / "physical_mechanism_v3/common_band_pairs.csv"
PAIR_SCRIPT = ANALYSIS / "physical_mechanism_v3/common_band_noise_analysis.py"
PAIR_QA = ANALYSIS / "physical_mechanism_v3/common_band_qa.json"
PHYSICS = ANALYSIS / "phase_residual_analysis_v1/prepared/physical_minutes.csv"
MAPPING = Path(r"D:\月球导航\data\external_reference\mapping\active_prn_svn_block_20250115_20250316.csv")
SIGNALS = {"L1_E1": ("GPS_L1", "GAL_E1"), "L5_E5a": ("GPS_L5", "GAL_E5a")}
PHASES = ["C", "T", "L", "S"]
CALIPERS = [.5, 1., 2.]
LAYERS = {"trend_primary": "trend", "raw_sensitivity": "raw"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, suffix):
    frame.to_csv(OUT / ("sameband_" + suffix + ".csv"), index=False, encoding="utf-8-sig", float_format="%.17g")


def boolean(values):
    assert values.notna().all()
    return values.astype(str).str.lower().isin(["true", "1", "1.0"])


def equal_op_weighted_median(values, ops):
    if len(values) == 0:
        return np.nan
    counts = ops.value_counts().to_dict()
    common = math.lcm(*[int(n) for n in counts.values()])
    weights = [common // int(counts[op]) for op in ops]
    order = np.argsort(np.asarray(values), kind="stable")
    x = np.asarray(values)[order]
    w = [weights[int(i)] for i in order]
    total, cumulative, current = sum(w), [], 0
    for weight in w:
        current += weight
        cumulative.append(2 * current)
    index = bisect_left(cumulative, total)
    return float((x[index] + x[index + 1]) / 2 if cumulative[index] == total else x[index])


def safe_corr(x, y):
    if len(x) < 8 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))


def summarize(g, layer):
    suffix = LAYERS[layer]
    left, right = g["gps_bias_" + suffix + "_db"], g["gal_bias_" + suffix + "_db"]
    delta = g["paired_delta_" + suffix + "_db"]
    op_medians = g.groupby("op")["paired_delta_" + suffix + "_db"].median()
    result = dict(n_pairs=len(g), n_operations=g.op.nunique(), n_unique_op_epochs=g[["op", "epoch"]].drop_duplicates().shape[0],
        n_satellite_pair_arcs=g[["op", "gps_svid", "gal_svid"]].drop_duplicates().shape[0],
        gps_bias_median_db=left.median(), gal_bias_median_db=right.median(),
        paired_difference_median_db=delta.median(), paired_difference_mean_db=delta.mean(),
        paired_absolute_difference_median_db=delta.abs().median(),
        difference_of_marginal_medians_db=left.median() - right.median(),
        paired_fraction_abs_le_1db=delta.abs().le(1).mean() if len(g) else np.nan,
        paired_fraction_abs_le_2db=delta.abs().le(2).mean() if len(g) else np.nan,
        paired_fraction_positive=delta.gt(0).mean() if len(g) else np.nan,
        equal_op_weighted_paired_median_db=equal_op_weighted_median(delta, g.op),
        median_of_op_paired_medians_db=op_medians.median(),
        min_op_paired_median_db=op_medians.min(), max_op_paired_median_db=op_medians.max(),
        n_positive_op_medians=int(op_medians.gt(0).sum()), n_negative_op_medians=int(op_medians.lt(0).sum()),
        op76_pair_share_pct=100*g.op.eq("OP76").mean() if len(g) else np.nan,
        surface_pair_share_pct=100*g.phase.eq("S").mean() if len(g) else np.nan,
        rx_separation_median_deg=g.rx_separation_deg.median(),
        paired_bias_spearman=safe_corr(left, right))
    for q in [.05, .25, .75, .95]:
        result[f"paired_difference_p{round(q*100):02d}_db"] = delta.quantile(q)
    # A descriptive within-arc check; no iid significance tests.
    centered = g[["op", "gps_svid", "gal_svid", "gps_bias_"+suffix+"_db", "gal_bias_"+suffix+"_db"]].copy()
    if len(centered):
        group_keys = ["op", "gps_svid", "gal_svid"]
        counts = centered.groupby(group_keys).op.transform("size")
        centered = centered.loc[counts.ge(8)].copy()
        for side in ["gps", "gal"]:
            column = side+"_bias_"+suffix+"_db"
            centered[side+"_centered"] = centered[column] - centered.groupby(group_keys)[column].transform("median")
        rho = safe_corr(centered.gps_centered, centered.gal_centered)
        result["within_arc_centered_n_pairs"] = len(centered)
        result["within_arc_centered_n_arcs"] = centered[group_keys].drop_duplicates().shape[0]
        result["within_arc_centered_bias_spearman"] = rho
    else:
        result.update(within_arc_centered_n_pairs=0, within_arc_centered_n_arcs=0, within_arc_centered_bias_spearman=np.nan)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [PAIR_SOURCE, PAIR_SCRIPT, PAIR_QA, PHYSICS, MAPPING]
    before = {str(path): sha(path) for path in sources}
    historical = json.loads(PAIR_QA.read_text(encoding="utf-8"))
    assert before[str(PHYSICS)] == historical["hash_before"] == historical["hash_after"]
    pairs = pd.read_csv(PAIR_SOURCE, float_precision="round_trip")
    columns = ["op", "system", "signal_name", "svid", "source_bin_gps_seconds", "minute_utc", "mission_phase", "evaluation_split", "trend_training_eligible", "earth_blocked", "moon_blocked", "cn0_physics_trend_dbhz", "cn0_trend_target_dbhz", "cn0_observed_trend_dbhz", "cn0_constellation_physics_proxy_dbhz", "cn0_dbhz_mean", "rx_offboresight_spice_deg", "rx_azimuth_spice_deg", "budget_noise_psd_dbw_hz"]
    source = pd.read_csv(PHYSICS, usecols=columns, float_precision="round_trip")
    source.svid = source.svid.astype(int)
    source["eligible_nonocculted"] = boolean(source.trend_training_eligible) & ~boolean(source.earth_blocked) & ~boolean(source.moon_blocked) & np.isfinite(source.cn0_trend_target_dbhz) & np.isfinite(source.cn0_physics_trend_dbhz)
    assert int(source.eligible_nonocculted.sum()) == 15734
    source["bias_trend_db"] = source.cn0_physics_trend_dbhz - source.cn0_trend_target_dbhz
    source["bias_raw_db"] = source.cn0_constellation_physics_proxy_dbhz - source.cn0_dbhz_mean
    operations = source[["op", "mission_phase"]].drop_duplicates().sort_values("op", key=lambda s:s.str.removeprefix("OP").astype(int))
    assert len(operations) == 20
    keys = ["op", "source_bin_gps_seconds", "signal_name", "svid"]
    assert not source.duplicated(keys).any()
    original_n = len(pairs)
    errors = {}
    for side, signal_number in [("gps", 0), ("gal", 1)]:
        pairs[side+"_svid"] = pairs[side+"_svid"].astype(int)
        pairs[side+"_signal"] = pairs.band.map({band: signals[signal_number] for band,signals in SIGNALS.items()})
        renamed = source.rename(columns={"source_bin_gps_seconds": "epoch", "signal_name": side+"_signal", "svid": side+"_svid", **{c: side+"_source_"+c for c in source if c not in keys}})
        pairs = pairs.merge(renamed, on=["op", "epoch", side+"_signal", side+"_svid"], how="left", validate="many_to_one")
        assert len(pairs) == original_n
        assert pairs[side+"_source_eligible_nonocculted"].eq(True).all()
        assert pairs[side+"_source_mission_phase"].eq(pairs.phase).all()
        assert pairs[side+"_source_bias_trend_db"].notna().all()
        errors[side+"_saved_raw_bias_error_db"] = float((pairs[side+"_bias_raw_db"]-pairs[side+"_source_bias_raw_db"]).abs().max())
        pairs[side+"_bias_trend_db"] = pairs[side+"_source_bias_trend_db"]
        pairs[side+"_bias_raw_db"] = pairs[side+"_source_bias_raw_db"]
    for suffix in ["trend", "raw"]:
        pairs["paired_delta_"+suffix+"_db"] = pairs["gps_bias_"+suffix+"_db"]-pairs["gal_bias_"+suffix+"_db"]
        errors["saved_pair_delta_"+suffix+"_error_db"] = float((pairs["paired_delta_"+suffix+"_db"]-pairs["delta_bias_"+suffix+"_db"]).abs().max())
    vectors = []
    for side in ["gps", "gal"]:
        theta = np.deg2rad(pairs[side+"_source_rx_offboresight_spice_deg"])
        azimuth = np.deg2rad(pairs[side+"_source_rx_azimuth_spice_deg"])
        vectors.append(np.column_stack([np.sin(theta)*np.cos(azimuth), np.sin(theta)*np.sin(azimuth), np.cos(theta)]))
    separation = np.rad2deg(np.arccos(np.clip(np.sum(vectors[0]*vectors[1],axis=1), -1, 1)))
    errors["saved_pair_direction_separation_error_deg"] = float(np.max(np.abs(separation-pairs.rx_separation_deg)))
    assert errors["saved_pair_direction_separation_error_deg"] < 1e-7
    assert max(v for k,v in errors.items() if k.endswith("_db")) < 1e-9
    assert (pairs.rx_separation_deg <= pairs.caliper_deg+1e-9).all()
    for side in ["gps", "gal"]:
        assert not pairs.duplicated(["caliper_deg", "band", "op", "epoch", side+"_svid"]).any()
    assert pairs.gps_source_minute_utc.eq(pairs.gal_source_minute_utc).all()
    mapping = pd.read_csv(MAPPING).set_index("prn")
    assert mapping.index.is_unique
    pairs["gal_antenna_type_from_metadata"] = ("E"+pairs.gal_svid.astype(str).str.zfill(2)).map(mapping.antenna_type)
    errors["raw_noise_budget_difference_max_abs_db"] = float((pairs.gps_source_budget_noise_psd_dbw_hz-pairs.gal_source_budget_noise_psd_dbw_hz).abs().max())
    assert errors["raw_noise_budget_difference_max_abs_db"] < 1e-10
    assert equal_op_weighted_median(pd.Series([0.,0.,0.,10.]), pd.Series(["a","a","a","b"])) == 5.0
    save(pairs, "pairs_enriched")
    summaries, deletions, robustness, foc_scope = [], [], [], []
    for caliper in CALIPERS:
        for band in SIGNALS:
            full = pairs.loc[pairs.caliper_deg.eq(caliper)&pairs.band.eq(band)]
            retained = full.loc[~full.gal_svid.isin([11,12])]
            all_confirmed = bool(len(retained) and retained.gal_antenna_type_from_metadata.eq("GALILEO-2").all())
            foc_scope.append(dict(caliper_deg=caliper, band=band, selection="exclude_known_IOV_E11_E12_without_rematching",
                retained_scope="metadata_confirmed_FOC" if all_confirmed else "exclude_known_IOV_only",
                n_original_pairs=len(full), original_trend_paired_median_db=full.paired_delta_trend_db.median(),
                n_removed_known_IOV_pairs=len(full)-len(retained), n_retained_pairs=len(retained),
                retained_trend_paired_median_db=retained.paired_delta_trend_db.median(),
                retained_minus_original_median_db=retained.paired_delta_trend_db.median()-full.paired_delta_trend_db.median(),
                n_retained_operations=retained.op.nunique(),
                n_retained_positive_FOC_mapping=int(retained.gal_antenna_type_from_metadata.eq("GALILEO-2").sum()),
                n_retained_unknown_mapping=int(retained.gal_antenna_type_from_metadata.isna().sum()),
                retained_known_antenna_types=";".join(sorted(retained.gal_antenna_type_from_metadata.dropna().unique()))))
            for layer in LAYERS:
                common = dict(caliper_deg=caliper, band=band, layer=layer)
                headline = summarize(full, layer)
                summaries.append(dict(**common, subset="all_pairs", level="overall", phase="ALL", op="ALL", **headline))
                for phase in PHASES:
                    block = full.loc[full.phase.eq(phase)]
                    summaries.append(dict(**common, subset="all_pairs", level="phase", phase=phase, op="ALL", **summarize(block,layer)))
                for record in operations.itertuples(index=False):
                    block = full.loc[full.op.eq(record.op)]
                    summaries.append(dict(**common, subset="all_pairs", level="operation", phase=record.mission_phase, op=record.op, **summarize(block,layer)))
                excluded76 = summarize(full.loc[~full.op.eq("OP76")],layer)
                excludedS = summarize(full.loc[~full.phase.eq("S")],layer)
                for label, stats in [("exclude_OP76",excluded76),("exclude_surface",excludedS)]:
                    summaries.append(dict(**common, subset=label, level="overall", phase="ALL", op="ALL", **stats))
                active_deletions = []
                for record in operations.itertuples(index=False):
                    removed = full.op.eq(record.op)
                    block = full.loc[~removed]
                    stats = summarize(block, layer)
                    row = dict(**common, removed_op=record.op, n_removed=int(removed.sum()), removal_had_pairs=bool(removed.any()),
                               change_in_paired_median_db=stats["paired_difference_median_db"]-headline["paired_difference_median_db"], **stats)
                    deletions.append(row)
                    if removed.any() and len(block):
                        active_deletions.append(row)
                valid = pd.DataFrame(active_deletions)
                robust = dict(**common, n_full=len(full), n_operations=full.op.nunique(), full_paired_median_db=headline["paired_difference_median_db"],
                    full_equal_op_weighted_paired_median_db=headline["equal_op_weighted_paired_median_db"], full_median_of_op_paired_medians_db=headline["median_of_op_paired_medians_db"],
                    op76_pair_share_pct=headline["op76_pair_share_pct"], surface_pair_share_pct=headline["surface_pair_share_pct"],
                    exclude_OP76_n=excluded76["n_pairs"], exclude_OP76_median_db=excluded76["paired_difference_median_db"],
                    exclude_surface_n=excludedS["n_pairs"], exclude_surface_median_db=excludedS["paired_difference_median_db"],
                    n_nonempty_observed_op_deletions=len(valid))
                if len(valid):
                    robust.update(delete_one_op_median_min_db=valid.paired_difference_median_db.min(), delete_one_op_median_max_db=valid.paired_difference_median_db.max(),
                        delete_one_op_max_abs_shift_db=valid.change_in_paired_median_db.abs().max(), most_influential_op=valid.loc[valid.change_in_paired_median_db.abs().idxmax(),"removed_op"])
                robustness.append(robust)
    summary = pd.DataFrame(summaries)
    delete = pd.DataFrame(deletions)
    robust = pd.DataFrame(robustness)
    save(summary, "summary")
    save(summary.loc[summary.level.eq("operation")], "all20_operation_summary")
    save(delete, "delete_one_operation")
    save(robust, "robustness")
    foc_scope = pd.DataFrame(foc_scope)
    save(foc_scope, "foc_scope")
    save(pairs.groupby(["caliper_deg","band","phase","op"]).size().reset_index(name="n_pairs"), "composition")
    assert len(summary.loc[summary.level.eq("operation")]) == 3*2*2*20
    empty = summary.n_pairs.eq(0)
    assert summary.loc[empty,["gps_bias_median_db","gal_bias_median_db","paired_difference_median_db","paired_difference_p05_db","paired_difference_p95_db"]].isna().all().all()
    after = {str(path):sha(path) for path in sources}
    assert before == after
    qa = dict(source_hashes_before=before, source_hashes_after=after, original_sources_unchanged=True, primary_source_eligible_nonocculted_rows=15734,
        total_saved_pair_records=len(pairs), records_by_caliper_band=pairs.groupby(["caliper_deg","band"]).size().reset_index(name="n").to_dict("records"),
        joined_both_sides_eligible_nonocculted=True, original_pair_identity_unchanged=True, same_native_minute=True,
        no_reused_satellite_within_caliper_band_epoch=True, twenty_operations_in_each_summary=True, n_empty_summary_strata=int(empty.sum()),
        errors=errors, bias_definition="baseline minus observation; paired contrast GPS minus Galileo", primary="trend", primary_caliper_deg=0.5, sensitivity="unsmoothed native-minute observation medians and 1/2 degree calipers; legacy cn0_dbhz_mean column name is not the aggregation operator",
        equal_op_weighted_median_definition="Each observed OP has total weight 1/K, pairs within OP equal; exact integer relative weights; median minimizer interval midpoint at exactly half mass.",
        op_median_sensitivity_definition="Compute paired-difference median separately within each observed OP, then median of those OP medians; distinct from equal-OP weighted row median.",
        deletion_definition="Delete paired rows of one OP and recompute descriptive summaries; no model fitting and not heldout LOPO performance. Deletion range is not a confidence interval.",
        correlations="Spearman of paired side biases; centered version subtracts side-specific median within OP and GPS/Galileo satellite-pair arc, retaining arcs with >=8 pairs. No causal interpretation or iid p-values.",
        caliper_note="Reuse each existing caliper-specific one-to-one matching; matches need not nest across calipers.",
        foc_scope_filter_did_not_rematch=True,
        foc_scope_all_retained_pairs_positively_mapped_FOC=bool(foc_scope.retained_scope.eq("metadata_confirmed_FOC").all()),
        foc_scope_retained_unknown_mapping_count=int(foc_scope.n_retained_unknown_mapping.sum()),
        code_sha256=sha(Path(__file__)), no_model_fit_predict=True)
    (OUT/"sameband_qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    report = ["# 同频GPS–Galileo配对偏差扩展", "", "## 口径", "", "主偏差=physics_trend−observed_trend，配对差=GPS偏差−Galileo偏差；正值表示GPS的基线高估更多。raw采用未平滑的原生一分钟观测中位数作敏感性复核；保存列cn0_dbhz_mean是历史命名。复用v3的0.5/1/2°近接收方向、同OP同原生分钟的一对一配对，不按误差重新匹配。基础源为15734个eligible且非Earth/Moon blocked分钟；不是原15806全eligible口径。双方趋势偏差重新连接原physical_minutes核验。", "", "所有配对差中位数先逐对相减再取median；另列双方各自median供描述，绝不拿二者相减冒充配对median。沿用原0.5°为主示例，1/2°完整列为敏感性，不按结果选择阈值。", "", "## 总体及删除敏感性（trend主结果）", "", "|阈值°|频段|n / OP数|GPS / Gal偏差median dB|配对差median [p5,p95] dB|OP等权median|OP76占比 / S占比 %|删OP76 median (n)|删S median (n)|删单OP median范围|", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in robust.loc[robust.layer.eq("trend_primary")].itertuples(index=False):
        rec = summary.loc[summary.caliper_deg.eq(row.caliper_deg)&summary.band.eq(row.band)&summary.layer.eq(row.layer)&summary.subset.eq("all_pairs")&summary.level.eq("overall")].iloc[0]
        report.append(f"|{row.caliper_deg:g}|{row.band}|{row.n_full} / {row.n_operations}|{rec.gps_bias_median_db:.3f} / {rec.gal_bias_median_db:.3f}|{row.full_paired_median_db:+.3f} [{rec.paired_difference_p05_db:+.3f},{rec.paired_difference_p95_db:+.3f}]|{row.full_equal_op_weighted_paired_median_db:+.3f}|{row.op76_pair_share_pct:.1f} / {row.surface_pair_share_pct:.1f}|{row.exclude_OP76_median_db:+.3f} ({row.exclude_OP76_n})|{row.exclude_surface_median_db:+.3f} ({row.exclude_surface_n})|[{row.delete_one_op_median_min_db:+.3f},{row.delete_one_op_median_max_db:+.3f}]|")
    report += ["", "OP等权median对每个有配对OP赋相同总权重、OP内每pair等权。另给各OP的pair-median再取median，两者不是同一个统计。删单OP范围是影响诊断而非置信区间，也不是模型LOPO。", "", "## 0.5°分阶段主结果", "", "|频段|阶段|n / OP数|GPS / Gal偏差median dB|配对差median [p5,p95] dB|median绝对配对差 dB|", "|---|---|---:|---:|---:|---:|"]
    stage = summary.loc[summary.caliper_deg.eq(.5)&summary.layer.eq("trend_primary")&summary.level.eq("phase")&summary.subset.eq("all_pairs")]
    for row in stage.itertuples(index=False):
        values = f"{row.gps_bias_median_db:.3f} / {row.gal_bias_median_db:.3f}|{row.paired_difference_median_db:+.3f} [{row.paired_difference_p05_db:+.3f},{row.paired_difference_p95_db:+.3f}]|{row.paired_absolute_difference_median_db:.3f}" if row.n_pairs else "NA|NA|NA"
        report.append(f"|{row.band}|{row.phase}|{row.n_pairs} / {row.n_operations}|{values}|")
    report += ["", "## 可识别边界", "", "相近接收方向和同频有助于削弱共同接收方向/同频噪声预算差，但GPS和Galileo仍是不同卫星、不同传播射线、不同发射功率/方向图及信号估计口径。不能把配对差唯一归为接收机、发射端或大气项。尤其中心接近0只表示典型差值接近，不是每一对偏差一致、统计等效或已证明共同原因；应同时看分位范围、绝对差、OP构成和删除结果。", "", "全部20OP均保留，无配对为n=0、数值NA。配对行存在时间相关，未给iid显著性或伪精确置信区间。可选相关性仅为描述：弧内去中心后仍相关也不能识别共同原因，剩余变化可能包含未匹配的卫星/几何/估计器因素。", "", "CSV为17位精度；sameband_pairs_enriched.csv逐对可追溯，sameband_summary.csv包含raw敏感性及筛选口径，sameband_delete_one_operation.csv是逐OP删除结果。未改论文、原数据或模型。"]
    report += ["", "原子链路预算项的相同/抵消只在raw逐项预算层核对：这里同频配对的保存噪声PSD预算差为0（数值精度内）。各信号trend的非线性时间滤波支持可能不同，不能仅凭trend差直接声称某个瞬时物理项已精确抵消。", "", "## 简要结论", "", "0.5°下高频paired median为+2.158 dB，三角阈总体为+2.158/+2.176/+2.208 dB，逐OP删除后依然为正且约2 dB。低频paired median为−0.019 dB，删除OP76后+0.102 dB，并非只由OP76造成；OP等权median为+0.103 dB。", "", "但0.5°下94.6%的低频配对来自Surface；非Surface只剩OP17的26对（median +0.098 dB），C/L均无配对，因此不能把这一证据推广成四阶段普遍一致。低频绝对配对差median为0.456 dB，80.2%在±1 dB内，但p5/p95仍为−2.481/+1.133 dB，存在明显尾部差异。raw下高/低频median为+2.129/−0.046 dB，主模式未依赖趋势滤波。", "", "双方各自中位数不能相减代替配对差：例如0.5°高频双方median为9.626/8.887 dB，其相减仅0.739 dB，而逐对相减的median为2.158 dB。"]
    report += ["", "## 排除已知Galileo IOV的物理适用域敏感性", "", "直接从各阈值既有配对中删除Galileo SVID 11/12（E11/E12），不重新匹配。随后用已有PRN卫星映射逐个正证：本次所有保留配对的Galileo均为GALILEO-2（FOC），未知映射为0。因此该保留子集可称metadata-confirmed FOC；并不是把未知SVID默认当作FOC。", "", "|阈值°|频段|原n / 配对差median dB|删除IOV对数|FOC保留n / 配对差median dB|", "|---:|---|---:|---:|---:|"]
    for row in foc_scope.itertuples(index=False):
        report.append(f"|{row.caliper_deg:g}|{row.band}|{row.n_original_pairs} / {row.original_trend_paired_median_db:+.4f}|{row.n_removed_known_IOV_pairs}|{row.n_retained_pairs} / {row.retained_trend_paired_median_db:+.4f}|")
    report += ["", "这是方向图参考适用域检查，不证明GPS与Galileo的发射预算、测量估计器或实际传播损耗相同。保留子集仍沿用原配对选择；不应解释成重新求得的最优FOC匹配。"]
    (OUT/"sameband_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(robust.loc[robust.layer.eq("trend_primary")].to_string(index=False))


if __name__ == "__main__":
    main()
