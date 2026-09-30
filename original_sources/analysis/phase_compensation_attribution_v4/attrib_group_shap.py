"""Exact grouped interventional Shapley of the existing final HGB; never fits.

All source files are read only. The finite, signal-specific background is fixed
across explained phases/operations. Six groups imply 64 coalitions; coalitions
that differ only in exactly constant groups are evaluated once. Attribution is
for raw HGB output, with final split-isolated smoothing discrepancy kept separate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(r"D:\月球导航")
sys.path.insert(0, str(ROOT / "runtime_cache/python_deps"))
sys.path.insert(0, str(ROOT / "script"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

MANUSCRIPT = ROOT / "fabio_xz/AI-driven cislunar GNSS channel modelling from lunar GNSS observations"
OUT = MANUSCRIPT / "analysis/phase_compensation_attribution_v4"
MODEL = ROOT / "table/algorithm/cn0_trend_residual_tuned_no_leakage/cn0_trend_residual_tuned_no_leakage.joblib"
PRED = MODEL.with_name("cn0_trend_residual_tuned_no_leakage_predictions.csv")
CACHE = ROOT / "table/algorithm/cn0_ai_residual_story/final_split_feature_frame.joblib"
PHYSICS = MANUSCRIPT / "analysis/phase_residual_analysis_v1/prepared/physical_minutes.csv"
KEYS = ["minute_utc", "op", "signal_name", "svid"]
PHASES = ["C", "T", "L", "S"]
PHASE_NAMES = dict(zip(PHASES, ["Commissioning", "Trans-lunar", "Lunar orbit", "Surface"]))
GROUPS = {
    "TX_direction": ["tx_offboresight_deg", "tx_theta_body_deg", "tx_phi_body_deg", "tx_gain_2d_db", "tx_power_dbw", "tx_ssv_main_lobe_boundary_deg", "tx_ssv_signed_lower_boundary_deg", "tx_ssv_signed_upper_boundary_deg", "tx_ssv_double_sided_full_width_deg"],
    "RX_direction": ["rx_peak_gain_dbic", "rx_offboresight_spice_deg", "rx_azimuth_spice_deg", "rx_gain_envelope_dbic"],
    "Atmospheric_proxies": ["ionosphere_shell_lower_alt_km", "ionosphere_shell_upper_alt_km", "ionosphere_l30_vertical_assumed_db", "m_ion_proxy", "l_ion_abs_proxy_db", "l_ion_abs_budget_db", "troposphere_shell_upper_alt_km", "gas_scale_height_assumed_km", "gas_vertical_equivalent_airmass_km", "gas_equivalent_airmass_km", "m_gas_proxy", "gas_gamma_surface_db_per_km", "l_gas_abs_proxy_db", "l_gas_abs_budget_db"],
    "Limb_proximity": ["earth_limb_margin_deg", "moon_limb_margin_deg", "earth_limb_proximity_proxy", "moon_limb_proximity_proxy", "edge_near_limb_alt_threshold_km", "neutral_refraction_upper_alt_km"],
    "Composite_baseline_range": ["cn0_constellation_direct_available_dbhz", "cn0_reference_trajectory_2d_dbhz", "cn0_physics_trend_dbhz", "fspl_db", "geometric_range_km", "range_rate_rx_only_km_s", "earth_observer_altitude_km", "moon_observer_altitude_km"],
    "Instrument_band_constants": ["system_noise_temperature_k", "implementation_loss_assumed_db", "frequency_mhz"],
}
GROUP_NAMES = list(GROUPS)
N_BACKGROUND = 32
SEED = 421729


def say(message):
    print(message, flush=True)


def boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    assert series.notna().all()
    assert set(series.astype(str).str.lower().unique()) <= {"true", "false", "1", "0", "1.0", "0.0"}
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def normalize_keys(frame):
    frame["minute_utc"] = pd.to_datetime(frame["minute_utc"], utc=True)
    frame["svid"] = frame["svid"].astype(int)
    assert not frame.duplicated(KEYS).any()
    return frame


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.17g")


def exact_group_shap(model, x, background, column_indices, label):
    """All 64 coalition expectations, averaged over the given finite background."""
    n, g = len(x), len(column_indices)
    union = np.vstack([x, background])
    active = [j for j, columns in enumerate(column_indices) if np.any(np.ptp(union[:, columns], axis=0) != 0)]
    active_mask = sum(1 << j for j in active)
    canonical_masks = sorted(set(mask & active_mask for mask in range(1 << g)))
    baseline = float(np.mean(model.predict(background)))
    values = {0: np.full(n, baseline), active_mask: model.predict(x)}
    started = time.monotonic()
    for count, mask in enumerate(canonical_masks):
        if mask in values:
            continue
        columns = [col for j, indices in enumerate(column_indices) if mask & (1 << j) for col in indices]
        result = np.empty(n)
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            mixed = np.tile(background, (stop - start, 1))
            mixed[:, columns] = np.repeat(x[start:stop, columns], len(background), axis=0)
            result[start:stop] = model.predict(mixed).reshape(stop - start, len(background)).mean(axis=1)
        values[mask] = result
        if count % 8 == 0:
            say(f"{label}: coalition {count + 1}/{len(canonical_masks)} elapsed={time.monotonic()-started:.1f}s")
    phi = np.zeros((n, g))
    for j in range(g):
        for mask in range(1 << g):
            if mask & (1 << j):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(g - k - 1) / math.factorial(g)
            phi[:, j] += weight * (values[(mask | (1 << j)) & active_mask] - values[mask & active_mask])
    closure = np.max(np.abs(baseline + phi.sum(axis=1) - values[active_mask]))
    assert closure < 1e-10, closure
    return phi, baseline, {"inactive_groups": [GROUP_NAMES[j] for j in range(g) if j not in active], "unique_coalitions": len(canonical_masks), "max_local_accuracy_error_db": float(closure)}


def summarize(block, metadata):
    result = dict(metadata)
    result.update(n=len(block), n_signals=block.signal_name.nunique(), n_operations=block.op.nunique(), n_links=block[["op", "signal_name", "svid"]].drop_duplicates().shape[0], n_earth_blocked=int(block.earth_blocked.sum()), n_moon_blocked=int(block.moon_blocked.sum()))
    quantities = ["beta_db", "delta_raw_db", "delta_eval_db", "phi0_db", "smooth_discrepancy_db", "total_compensation_db", "residual_after_beta_db", "saved_eval_minus_final_refit_db"]
    for column in quantities:
        values = block[column].to_numpy()
        stem = column.removesuffix("_db")
        result[stem + "_mean_db"] = np.mean(values) if len(values) else np.nan
        result[stem + "_rms_db"] = np.sqrt(np.mean(values ** 2)) if len(values) else np.nan
    mags = np.array([block["phi_" + name + "_db"].abs().mean() for name in GROUP_NAMES])
    denominator = mags.sum()
    for name, mag in zip(GROUP_NAMES, mags):
        phi = block["phi_" + name + "_db"]
        result[name + "_signed_mean_db"] = phi.mean()
        result[name + "_mean_abs_db"] = mag
        result[name + "_share_pct"] = 100 * mag / denominator if denominator > 0 else np.nan
        result[name + "_mean_product_with_residual_after_beta_db2"] = (phi * block.residual_after_beta_db).mean()
    result["sum_group_mean_abs_db"] = denominator
    order = np.argsort(-mags, kind="stable")
    result["top_group"] = GROUP_NAMES[order[0]] if len(block) and denominator > 0 else ""
    result["top_group_share_pct"] = 100 * mags[order[0]] / denominator if len(block) and denominator > 0 else np.nan
    result["ranking"] = ";".join(GROUP_NAMES[j] for j in order) if len(block) else ""
    return result


def build_summaries(rows, operation_metadata):
    summaries = []
    for reference, reference_rows in rows.groupby("reference", sort=False):
        for mask_name in ["primary_continuity", "nonocculted_sensitivity"]:
            data = reference_rows if mask_name == "primary_continuity" else reference_rows.loc[~(reference_rows.earth_blocked | reference_rows.moon_blocked)]
            for scope in ["all_eligible", "external_holdout", "internal_test"]:
                subset = data if scope == "all_eligible" else data.loc[data.evaluation_split.eq("test" if scope == "internal_test" else scope)]
                meta = dict(reference=reference, mask=mask_name, scope=scope, weighting="sample_weighted")
                summaries.append(summarize(subset, dict(meta, level="overall", mission_phase="ALL", op="ALL", signal_name="ALL")))
                for phase in PHASES:
                    phase_rows = subset.loc[subset.mission_phase.eq(phase)]
                    summaries.append(summarize(phase_rows, dict(meta, level="phase", mission_phase=phase, op="ALL", signal_name="ALL")))
                    for signal, signal_rows in phase_rows.groupby("signal_name", sort=True):
                        summaries.append(summarize(signal_rows, dict(meta, level="phase_signal", mission_phase=phase, op="ALL", signal_name=signal)))
                ops = operation_metadata if scope == "all_eligible" else operation_metadata.loc[operation_metadata.op.isin(subset.op)]
                for op_record in ops.itertuples(index=False):
                    op_rows = subset.loc[subset.op.eq(op_record.op)]
                    summaries.append(summarize(op_rows, dict(meta, level="operation", mission_phase=op_record.mission_phase, op=op_record.op, signal_name="ALL")))
                    for signal, signal_rows in op_rows.groupby("signal_name", sort=True):
                        summaries.append(summarize(signal_rows, dict(meta, level="operation_signal", mission_phase=op_record.mission_phase, op=op_record.op, signal_name=signal)))
    table = pd.DataFrame(summaries)
    table["phase_name"] = table.mission_phase.map(PHASE_NAMES).fillna("All phases")
    # Equal signal weights apply to mean contributions first, before shares.
    balanced = []
    for keys, block in table.loc[table.level.eq("phase_signal")].groupby(["reference", "mask", "scope", "mission_phase"], sort=False):
        rec = dict(zip(["reference", "mask", "scope", "mission_phase"], keys))
        rec.update(level="phase", weighting="equal_signal", op="ALL", signal_name="ALL", phase_name=PHASE_NAMES[rec["mission_phase"]], n=int(block.n.sum()), n_signals=len(block), n_operations=np.nan, n_links=int(block.n_links.sum()), n_earth_blocked=int(block.n_earth_blocked.sum()), n_moon_blocked=int(block.n_moon_blocked.sum()))
        for column in table:
            if column.endswith("_mean_db") or column.endswith("_mean_abs_db") or column.endswith("_db2"):
                rec[column] = block[column].mean()
            elif column.endswith("_rms_db"):
                rec[column] = np.sqrt(np.mean(block[column] ** 2))
        denominator = sum(rec[name + "_mean_abs_db"] for name in GROUP_NAMES)
        rec["sum_group_mean_abs_db"] = denominator
        order = sorted(GROUP_NAMES, key=lambda name: -rec[name + "_mean_abs_db"])
        for name in GROUP_NAMES:
            rec[name + "_share_pct"] = 100 * rec[name + "_mean_abs_db"] / denominator if denominator > 0 else np.nan
        rec["ranking"] = ";".join(order)
        rec["top_group"] = order[0]
        rec["top_group_share_pct"] = rec[order[0] + "_share_pct"]
        balanced.append(rec)
    return pd.concat([table, pd.DataFrame(balanced)], ignore_index=True)


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(MODEL)
    frame = normalize_keys(pd.read_csv(PRED, float_precision="round_trip"))
    cache = normalize_keys(joblib.load(CACHE))
    cache["cache_row"] = np.arange(len(cache))
    match = frame[KEYS].merge(cache[KEYS + ["cache_row"]], on=KEYS, how="left", validate="one_to_one")
    assert match.cache_row.notna().all() and len(frame) == 17439
    cache_ordered = cache.iloc[match.cache_row.to_numpy()].reset_index(drop=True)
    features = artifact["model_features"]
    mapped = [feature for columns in GROUPS.values() for feature in columns]
    assert len(mapped) == len(set(mapped)) == 44 and set(mapped) == set(features)
    x_all = np.asarray(artifact["model"].named_steps["preprocess"].transform(cache_ordered[features]), dtype=float)
    model = artifact["model"].named_steps["model"]
    with threadpool_limits(limits=4):
        raw = model.predict(x_all)
    beta = frame.signal_name.map(artifact["signal_beta_db"]).to_numpy()
    raw_saved_error = float(np.max(np.abs(raw + beta - frame.ai_trend_residual_raw_pred_db)))
    assert raw_saved_error < 1e-10
    assert np.max(np.abs(beta - frame.signal_beta_train_db)) < 1e-10
    assert np.allclose(cache_ordered.cn0_physics_trend_dbhz, frame.cn0_physics_trend_dbhz, atol=1e-12, rtol=0)
    import train_cn0_trend_residual_tuned_no_leakage as tuned
    final_correction = np.asarray(tuned.engine.smooth_predicted_residual(frame, "ai_trend_residual_raw_pred_db", isolate_evaluation_splits=True))
    final_eval = frame.cn0_physics_trend_dbhz.to_numpy() + final_correction
    saved_difference = frame.cn0_physics_ai_eval_dbhz.to_numpy() - final_eval
    validation = frame.evaluation_split.eq("validation").to_numpy()
    assert np.max(np.abs(saved_difference[~validation])) < 1e-10
    flags = normalize_keys(pd.read_csv(PHYSICS, usecols=KEYS + ["earth_blocked", "moon_blocked"]))
    frame = frame.merge(flags, on=KEYS, how="left", validate="one_to_one")
    for column in ["earth_blocked", "moon_blocked"]:
        assert frame[column].notna().all()
        frame[column] = boolean(frame[column])
    eligible = boolean(frame.trend_training_eligible).to_numpy() & np.isfinite(frame.cn0_observed_trend_dbhz) & np.isfinite(frame.cn0_physics_trend_dbhz)
    assert int(np.sum(eligible)) == 15806
    frame["source_row"] = np.arange(len(frame))
    frame["beta_db"] = beta
    frame["delta_raw_db"] = raw
    frame["delta_eval_db"] = final_correction - beta
    frame["smooth_discrepancy_db"] = frame.delta_eval_db - raw
    frame["total_compensation_db"] = final_correction
    frame["residual_after_beta_db"] = frame.cn0_observed_trend_dbhz - frame.cn0_physics_trend_dbhz - beta
    frame["saved_eval_minus_final_refit_db"] = saved_difference
    operation_metadata = frame[["op", "mission_phase"]].drop_duplicates().sort_values("op", key=lambda s: s.str.removeprefix("OP").astype(int))
    assert len(operation_metadata) == 20
    target = frame.loc[eligible].copy().reset_index(drop=True)
    fit = frame.loc[eligible & frame.evaluation_split.isin(["train", "validation"])].copy()
    assert len(fit) == 9894
    columns = [[features.index(feature) for feature in group] for group in GROUPS.values()]
    write_csv(pd.DataFrame([dict(feature=feature, group=name, model_column=features.index(feature)) for name, group in GROUPS.items() for feature in group]), "attrib_feature_groups.csv")
    references, row_outputs, coalition_qa = [], [], []
    with threadpool_limits(limits=4):
        for reference_index, reference in enumerate(["empirical_fit32", "phase_balanced_fit32"]):
            output = target.copy()
            output["reference"] = reference
            output["phi0_db"] = np.nan
            for group in GROUP_NAMES:
                output["phi_" + group + "_db"] = np.nan
            for signal_index, signal in enumerate(sorted(frame.signal_name.unique())):
                pool = fit.loc[fit.signal_name.eq(signal)]
                rng = np.random.default_rng(SEED + 1000 * reference_index + signal_index)
                if reference_index == 0:
                    background_indices = np.sort(rng.choice(pool.source_row.to_numpy(), N_BACKGROUND, replace=False))
                else:
                    background_indices = np.sort(np.concatenate([rng.choice(pool.loc[pool.mission_phase.eq(phase), "source_row"].to_numpy(), N_BACKGROUND // 4, replace=False) for phase in PHASES]))
                background = x_all[background_indices]
                reference_rows = frame.iloc[background_indices][["source_row"] + KEYS + ["mission_phase", "evaluation_split", "earth_blocked", "moon_blocked"]].copy()
                reference_rows["reference"] = reference
                reference_rows["reference_weight"] = 1 / N_BACKGROUND
                references.append(reference_rows)
                idx = output.index[output.signal_name.eq(signal)].to_numpy()
                x = x_all[output.loc[idx, "source_row"].to_numpy()]
                phi, phi0, checks = exact_group_shap(model, x, background, columns, reference + "/" + signal)
                output.loc[idx, "phi0_db"] = phi0
                output.loc[idx, ["phi_" + name + "_db" for name in GROUP_NAMES]] = phi
                checks.update(reference=reference, signal_name=signal, n_explained=len(idx), n_background=len(background), phi0_db=phi0)
                coalition_qa.append(checks)
                say(f"Finished {reference}/{signal}; phi0={phi0:.8f}; max closure={checks['max_local_accuracy_error_db']:.3g}")
            phi_sum = output[["phi_" + name + "_db" for name in GROUP_NAMES]].sum(axis=1)
            output["raw_local_accuracy_error_db"] = output.phi0_db + phi_sum - output.delta_raw_db
            output["total_closure_error_db"] = output.beta_db + output.phi0_db + phi_sum + output.smooth_discrepancy_db - output.total_compensation_db
            assert output.total_closure_error_db.abs().max() < 1e-10
            row_outputs.append(output)
            write_csv(output, "attrib_rows_" + reference + ".csv")
            interim_summary = build_summaries(output, operation_metadata)
            write_csv(interim_summary.loc[interim_summary.level.isin(["overall", "phase", "phase_signal"])], "attrib_phase_summary_" + reference + ".csv")
            write_csv(interim_summary.loc[interim_summary.level.isin(["overall", "operation", "operation_signal"])], "attrib_operation_summary_" + reference + ".csv")
            say("Reference summaries ready: " + reference)
    rows = pd.concat(row_outputs, ignore_index=True)
    write_csv(rows, "attrib_rows.csv")
    write_csv(pd.concat(references, ignore_index=True), "attrib_reference_rows.csv")
    summary = build_summaries(rows, operation_metadata)
    write_csv(summary, "attrib_summary_all.csv")
    write_csv(summary.loc[summary.level.isin(["overall", "phase", "phase_signal"])], "attrib_phase_summary.csv")
    write_csv(summary.loc[summary.level.isin(["overall", "operation", "operation_signal"])], "attrib_operation_summary.csv")
    id_columns = ["reference", "mask", "scope", "weighting", "level", "mission_phase", "phase_name", "op", "signal_name", "n", "n_signals", "n_operations", "n_links"]
    long_records = []
    for record in summary.to_dict("records"):
        for group in GROUP_NAMES:
            long_records.append({**{key: record[key] for key in id_columns}, "group": group, "signed_mean_db": record[group + "_signed_mean_db"], "mean_abs_db": record[group + "_mean_abs_db"], "share_pct": record[group + "_share_pct"], "mean_product_with_residual_after_beta_db2": record[group + "_mean_product_with_residual_after_beta_db2"]})
    write_csv(pd.DataFrame(long_records), "attrib_group_summary.csv")
    sensitivity = summary.loc[summary.level.eq("phase")].pivot(index=["mask", "scope", "mission_phase"], columns=["reference", "weighting"], values=["top_group", "ranking"])
    sensitivity.columns = ["__".join(col) for col in sensitivity.columns]
    write_csv(sensitivity.reset_index(), "attrib_phase_rank_sensitivity.csv")
    qa = {
        "model_path": str(MODEL), "feature_cache_path": str(CACHE), "prediction_path": str(PRED),
        "source_sha256": {str(path): sha256(path) for path in [MODEL, CACHE, PRED, PHYSICS]},
        "python": sys.version, "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "n_input": len(frame), "n_eligible": len(target), "n_fit_reference_pool": len(fit), "n_eligible_nonocculted": int((~(target.earth_blocked | target.moon_blocked)).sum()),
        "eligible_by_split": target.evaluation_split.value_counts().to_dict(), "n_operation_rows": len(operation_metadata), "eligible_by_op": {op: int((target.op == op).sum()) for op in operation_metadata.op},
        "n_external_nonocculted": int((target.evaluation_split.eq("external_holdout") & ~(target.earth_blocked | target.moon_blocked)).sum()),
        "beta_db": artifact["signal_beta_db"], "raw_plus_beta_max_abs_saved_error_db": raw_saved_error,
        "final_refit_vs_saved_eval_nonvalidation_max_abs_error_db": float(np.max(np.abs(saved_difference[~validation]))),
        "validation_saved_eval_origin": "train-only selection model; final artifact is refitted on eligible train+validation; difference kept separate, NOT absorbed into smoothing discrepancy",
        "n_validation_all": int(validation.sum()), "n_validation_eligible": int((validation & eligible).sum()),
        "validation_saved_minus_final_refit_rms_db": float(np.sqrt(np.mean(saved_difference[validation] ** 2))),
        "validation_saved_minus_final_refit_min_db": float(np.min(saved_difference[validation])), "validation_saved_minus_final_refit_max_db": float(np.max(saved_difference[validation])),
        "max_raw_local_accuracy_error_db": float(rows.raw_local_accuracy_error_db.abs().max()), "max_total_closure_error_db": float(rows.total_closure_error_db.abs().max()),
        "reference_seed": SEED, "reference_rows_per_signal": N_BACKGROUND, "coalition_qa": coalition_qa,
        "interpretation": "Exact six-group interventional Shapley for finite signal-stratified background; not physical loss attribution, not conditional Shapley, not LOPO-heldout inference for all operations.",
        "runtime_seconds": time.monotonic() - started,
    }
    (OUT / "attrib_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# 最终 HGB 六组模型归因（非物理损失占比）", "", "## 方法和口径", "", f"主样本延续原质量规则：{len(target):,} 条，20 个 OP 中 OP3 无保留样本；OP37 为13条最终模型描述，不是 LOPO 测试。另给排除 Earth/Moon blocked 的 {qa['n_eligible_nonocculted']:,} 条敏感性。外部留出原4112条，非遮挡4099条。", "", "不重新训练。使用最终 joblib 及未降精度 feature cache，raw HGB + β 全17439行重现保存 raw 预测。对原始完整分钟表使用正式 split-isolated 平滑，归因数据随后按质量筛选。validation 原 eval 列由 train-only selection 模型覆盖，因此本分析全OP统一使用 final-refit模型；这1897条 eligible validation 不是原selection性能结果。", "", "每个信号使用固定32条 eligible train+validation 背景，跨phase/OP不变。主参考从该信号经验拟合池无放回抽样；第二参考从 C/T/L/S 各取8条，仍跨解释phase固定。全部64个组联盟的背景平均预测精确计算，组中全部变量同时替换；常数组重复联盟只算一次。所得是有限参考下精确的 grouped interventional Shapley。参考池的32点取样仍有近似误差，未给出无限总体精确性或因果性保证。", "", "闭合式：总补偿 = β + φ0(signal, reference) + Σφ_group(raw HGB) + [δ_eval(final refit) − δ_raw]。φ0和最后的非线性时间平滑差单列；不对各组SHAP独立平滑后假装仍然相加。validation selection-vs-final差另列。", "", "份额 = mean(|φ_group|) / Σ_group mean(|φ_group|)。这里先组内合并再取绝对值，百分比仅在六组raw HGB参考差异归因之间归一化，不包含β、φ0或平滑差，不能扩大成总补偿或真实物理损失占比。各组signed mean与mean absolute同时提供，避免忽视相互抵消。", "", "Composite_baseline_range包含复合C/N0基线、FSPL/距离及接收机高度等代理，不等于FSPL误差。联合替换保留组内相关性，但可能破坏组间相关性并形成训练分布外组合。常数/零归因不等于物理作用为零。TX组包含发射功率/方向图边界等相关预算信息。Instrument_band_constants含噪温、假设实现损耗、频率，分信号参考下若常数，其贡献为零。", "", "阶段另先在每个phase×signal汇总signed和meanabs，再四信号等权（不是等权平均百分比）重算组份额；用以辨别样本信号构成影响。没有把不均衡行数对应的阶段变化直接解释成物理机制变化。", "", "## 主参考、全合格样本阶段结果", "", "|Phase|n|β mean dB|δ eval mean / RMS dB|φ0 mean dB|smooth discrepancy mean dB|最大组|份额|", "|---|---:|---:|---:|---:|---:|---|---:|"]
    headline = summary.loc[(summary.reference == "empirical_fit32") & (summary["mask"] == "primary_continuity") & (summary.scope == "all_eligible") & (summary.weighting == "sample_weighted") & (summary.level == "phase")]
    for rec in headline.to_dict("records"):
        report.append(f"|{rec['mission_phase']}|{rec['n']}|{rec['beta_mean_db']:.4f}|{rec['delta_eval_mean_db']:.4f} / {rec['delta_eval_rms_db']:.4f}|{rec['phi0_mean_db']:.4f}|{rec['smooth_discrepancy_mean_db']:.4f}|{rec['top_group']}|{rec['top_group_share_pct']:.2f}%|")
    report += ["", "## 参考与信号等权敏感性", ""]
    for phase in PHASES:
        choices = summary.loc[(summary["mask"] == "primary_continuity") & (summary.scope == "all_eligible") & (summary.level == "phase") & (summary.mission_phase == phase)]
        report.append(f"- {phase}: " + "; ".join(f"{r.reference}/{r.weighting}: {r.top_group} {r.top_group_share_pct:.2f}%" for r in choices.itertuples(index=False)))
    report += ["", "如果某阶段主组或份额随共同参考明显改变，应报告为参考敏感性，不能从其中挑选更像物理故事的结果。逐组与β后真实残差的乘积只作为方向性描述量，不是独立因果验证；留出误差是否降低仍以外部/LOPO原性能表为准。", "", "## QA", "", f"raw重现最大误差={raw_saved_error:.3g} dB；非validation final eval重现最大误差={qa['final_refit_vs_saved_eval_nonvalidation_max_abs_error_db']:.3g} dB；SHAP local accuracy最大误差={qa['max_raw_local_accuracy_error_db']:.3g} dB；总补偿闭合最大误差={qa['max_total_closure_error_db']:.3g} dB。", "", f"原validation selection eval与final-refit eval的差异RMS={qa['validation_saved_minus_final_refit_rms_db']:.6f} dB（全部2043条，非质量筛后1897条的RMS）；没有用其差异解释平滑。", "", "CSV保留17位有效数字。attrib_rows.csv有两个reference，各15806行；summary含reference/mask/scope/weighting/level维度，筛选这些列后方可使用。全部OP表须选level=operation、scope=all_eligible、reference=empirical_fit32、mask=primary_continuity、weighting=sample_weighted。未改源文件、未拟合模型、未改论文。"]
    report += ["", "等信号权重在全样本四阶段均为四信号各1/4；若细分到internal_test等scope缺少某信号，则只对实际可用信号等权，并在n_signals列明确（最少2个），不将未观测信号填零。"]
    (OUT / "attrib_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    say(json.dumps({"completed": str(OUT), "runtime_seconds": qa["runtime_seconds"], "closure": qa["max_total_closure_error_db"], "phase_headlines": headline[["mission_phase", "n", "beta_mean_db", "delta_eval_mean_db", "delta_eval_rms_db", "top_group", "top_group_share_pct"]].to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
