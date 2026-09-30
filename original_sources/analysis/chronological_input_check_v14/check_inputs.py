"""Read-only replay of saved chronological fits with one missing query input."""
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
V13 = OUT.parent / "chronological_prediction_v13"
sys.path.insert(0, str(ROOT / "runtime_cache/python_deps"))
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from threadpoolctl import threadpool_limits

CACHE = ROOT / "table/algorithm/cn0_ai_residual_story/final_split_feature_frame.joblib"
SOURCE = ROOT / "table/algorithm/cn0_trend_residual_tuned_no_leakage/cn0_trend_residual_tuned_no_leakage_predictions.csv"
SELECT = ROOT / "table/algorithm/cn0_trend_residual_tuning_no_leakage/selected_model.json"
ORIGINAL_MODEL = SOURCE.with_name("cn0_trend_residual_tuned_no_leakage.joblib")
KEYS = ["minute_utc", "op", "signal_name", "svid"]
TARGET = "cn0_dbhz_mean"
RAW = "cn0_constellation_physics_proxy_dbhz"
BASE = "cn0_physics_trend_dbhz"
RATE = "range_rate_rx_only_km_s"
CUTOFFS = ["OP2", "OP22", "OP37", "OP38"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            values = segment.value.where(segment.quality) if observed else segment.value
            values = values.interpolate(limit_direction="both")
            if not values.notna().any():
                continue
            filtered = values.rolling(9, center=True, min_periods=1).median()
            if len(filtered) >= 9:
                filtered = savgol_filter(filtered.to_numpy(float), 9, 2, mode="interp")
            output.loc[segment.index] = np.asarray(filtered, float)
    return output.to_numpy(float)


def metric(group, milestone, scope, name):
    result = dict(milestone=milestone, scope=scope, group=name, n=len(group),
                  n_operations=group.op.nunique(), n_unique_epochs=group.minute_utc.nunique())
    for stage, column in [("physical", BASE), ("beta", "beta_only_dbhz"),
                          ("original_hgb", "original_final_dbhz"),
                          ("imputed_hgb", "imputed_final_dbhz")]:
        error = group[column] - group.observed_dbhz
        result[stage + "_rmse_dbhz"] = float(np.sqrt(np.mean(error**2)))
        result[stage + "_mae_db"] = float(np.mean(np.abs(error)))
        result[stage + "_mean_bias_db"] = float(np.mean(error))
        result[stage + "_median_bias_db"] = float(np.median(error))
        result[stage + "_equal_phase_rmse_dbhz"] = float(np.sqrt((error**2).groupby(group.mission_phase).mean().mean()))
        result[stage + "_equal_operation_rmse_dbhz"] = float(np.sqrt((error**2).groupby(group.op).mean().mean()))
        result[stage + "_mean_operation_rmse_dbhz"] = float(np.sqrt((error**2).groupby(group.op).mean()).mean())
    difference = group.imputed_final_dbhz - group.original_final_dbhz
    result["delta_imputed_minus_original_rmse_dbhz"] = result["imputed_hgb_rmse_dbhz"] - result["original_hgb_rmse_dbhz"]
    result["delta_imputed_minus_beta_rmse_dbhz"] = result["imputed_hgb_rmse_dbhz"] - result["beta_rmse_dbhz"]
    result["prediction_difference_mean_db"] = float(difference.mean())
    result["prediction_difference_rms_db"] = float(np.sqrt(np.mean(difference**2)))
    result["prediction_difference_max_abs_db"] = float(difference.abs().max())
    result["small_sample_descriptive"] = len(group) < 20
    return result


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def main():
    existing_inputs = [CACHE, SOURCE, SELECT, ORIGINAL_MODEL] + sorted(p for p in V13.rglob("*") if p.is_file())
    initial_hashes = {str(path): sha(path) for path in existing_inputs}
    provenance = {str(path): sha(path) for path in [OUT / "PLAN.md", Path(__file__)]}
    cached = normalize(joblib.load(CACHE))
    source = normalize(pd.read_csv(SOURCE, usecols=KEYS + [TARGET, RAW, "trend_training_eligible", "source_bin_gps_seconds"], float_precision="round_trip"))
    frame = cached.merge(source, on=KEYS, validate="one_to_one", suffixes=("", "_source"))
    assert len(frame) == len(cached) == len(source) == 17439
    frame["quality"] = frame.trend_training_eligible.astype(str).str.lower().isin(["true", "1", "1.0"])
    source_quality = frame.trend_training_eligible_source.astype(str).str.lower().isin(["true", "1", "1.0"])
    assert np.array_equal(frame.quality, source_quality)
    saved = pd.read_csv(V13 / "later_operation_predictions.csv", float_precision="round_trip")
    saved.minute_utc = pd.to_datetime(saved.minute_utc, utc=True)
    assert not saved.duplicated(["milestone"] + KEYS).any()
    inventory = frame.groupby("op").agg(start=("minute_utc", "min"), end=("minute_utc", "max"))
    all_rows, all_metrics, audits, rate_medians = [], [], [], []
    for milestone in CUTOFFS:
        package = joblib.load(V13 / f"model_after_{milestone}.joblib")
        model, beta, features, protocol = (package[key] for key in ["model", "beta", "features", "protocol"])
        assert RATE in features and len(features) == 44
        cutoff = inventory.loc[milestone, "end"]
        past = inventory.index[inventory.end.le(cutoff)].tolist()
        future = inventory.index[inventory.start.gt(cutoff)].tolist()
        assert set(past) == set(protocol["past_operations"])
        assert set(future) == set(protocol["future_operations"])
        f = frame.copy()
        f["role"] = np.where(f.op.isin(past), "past", np.where(f.op.isin(future), "future", "crossing_excluded"))
        assert not f.role.eq("crossing_excluded").any()
        f[BASE] = smooth(f, f[RAW])
        f["observed_dbhz"] = smooth(f, f[TARGET], observed=True)
        eligible = f.quality & np.isfinite(f.observed_dbhz) & np.isfinite(f[BASE])
        fit = eligible & f.role.eq("past")
        score = eligible & f.role.eq("future")
        assert int(eligible.sum()) == 15806
        assert int(fit.sum()) == protocol["n_fit"] and int(score.sum()) == protocol["n_score"]
        recomputed_beta = (f.loc[fit, "observed_dbhz"] - f.loc[fit, BASE]).groupby(f.loc[fit, "signal_name"]).median().to_dict()
        beta_difference = max(abs(beta[key] - recomputed_beta[key]) for key in beta)
        assert beta_difference < 1e-12
        imputer = model.named_steps["preprocess"].named_transformers_["physics"]
        imputer_difference = float(np.max(np.abs(f.loc[fit, features].median().to_numpy() - imputer.statistics_)))
        assert imputer_difference < 1e-12
        with threadpool_limits(limits=4):
            original_raw = model.predict(f[features])
        f["beta_db"] = f.signal_name.map(beta)
        f["beta_only_dbhz"] = f[BASE] + f.beta_db
        f["original_hgb_raw_db"] = original_raw
        f["original_hgb_filtered_db"] = smooth(f, f.beta_db + original_raw) - f.beta_db
        f["original_final_dbhz"] = f[BASE] + f.beta_db + f.original_hgb_filtered_db
        original_later = f.loc[score]
        comparison = original_later.merge(saved.loc[saved.milestone.eq(milestone)], on=KEYS, validate="one_to_one", suffixes=("", "_saved"))
        assert len(comparison) == int(score.sum())
        pairs = [("observed_dbhz", "observed_dbhz_saved"), (BASE, BASE + "_saved"), ("beta_db", "beta_db_saved"),
                 ("beta_only_dbhz", "beta_only_dbhz_saved"), ("original_hgb_raw_db", "hgb_raw_db"),
                 ("original_hgb_filtered_db", "hgb_filtered_db"), ("original_final_dbhz", "final_dbhz")]
        reproduction = {left: float(np.max(np.abs(comparison[left] - comparison[right]))) for left, right in pairs}
        assert max(reproduction.values()) < 1e-10
        # Alter precisely one input at later query epochs; keep every fitting row unchanged.
        changed_features = f[features].copy()
        future_mask = f.role.eq("future")
        assert f.loc[future_mask, RATE].notna().all()
        changed_features.loc[future_mask, RATE] = np.nan
        assert changed_features.drop(columns=RATE).equals(f[features].drop(columns=RATE))
        assert changed_features.loc[~future_mask].equals(f.loc[~future_mask, features])
        with threadpool_limits(limits=4):
            imputed_raw = model.predict(changed_features)
            transformed = model.named_steps["preprocess"].transform(changed_features.loc[future_mask])
        rate_index = features.index(RATE)
        rate_median = float(imputer.statistics_[rate_index])
        assert np.all(transformed[:, rate_index] == rate_median)
        assert np.array_equal(imputed_raw[~future_mask], original_raw[~future_mask])
        f["imputed_hgb_raw_db"] = imputed_raw
        f["imputed_hgb_filtered_db"] = smooth(f, f.beta_db + imputed_raw) - f.beta_db
        f["imputed_final_dbhz"] = f[BASE] + f.beta_db + f.imputed_hgb_filtered_db
        assert np.isfinite(f.loc[score, [BASE, "beta_only_dbhz", "original_final_dbhz", "imputed_final_dbhz"]].to_numpy()).all()
        later = f.loc[score].copy()
        later["milestone"] = milestone
        later["query_range_rate_imputer_median_km_s"] = rate_median
        columns = ["milestone"] + KEYS + ["mission_phase", "source_bin_gps_seconds", RATE, "query_range_rate_imputer_median_km_s",
                  "observed_dbhz", BASE, "beta_db", "beta_only_dbhz", "original_hgb_raw_db", "original_hgb_filtered_db",
                  "original_final_dbhz", "imputed_hgb_raw_db", "imputed_hgb_filtered_db", "imputed_final_dbhz"]
        all_rows.append(later[columns])
        all_metrics.append(metric(later, milestone, "pooled", "all_later_operations"))
        for op, group in later.groupby("op"):
            all_metrics.append(metric(group, milestone, "operation", op))
        for phase, group in later.groupby("mission_phase"):
            all_metrics.append(metric(group, milestone, "phase", phase))
        rate_medians.append(dict(milestone=milestone, n_fit=int(fit.sum()), n_score=int(score.sum()),
                                 n_later_context_bins=int(future_mask.sum()), fitting_range_rate_median_km_s=rate_median,
                                 fitting_range_rate_min_km_s=float(f.loc[fit, RATE].min()),
                                 fitting_range_rate_max_km_s=float(f.loc[fit, RATE].max()),
                                 query_original_range_rate_min_km_s=float(later[RATE].min()),
                                 query_original_range_rate_max_km_s=float(later[RATE].max())))
        audits.append(dict(milestone=milestone, cutoff_utc=str(cutoff), n_fit=int(fit.sum()), n_score=int(score.sum()),
                           reproduction_max_abs_differences=reproduction, beta_reconstruction_max_abs_difference=beta_difference,
                           imputer_reconstruction_max_abs_difference=imputer_difference, modified_input_columns=[RATE],
                           all_past_features_and_raw_predictions_unchanged=True,
                           all_query_imputed_values_equal_saved_past_median=True, saved_rate_median_km_s=rate_median))
        print(json.dumps(all_metrics[-(1 + later.op.nunique() + later.mission_phase.nunique())], ensure_ascii=False), flush=True)
    metrics = pd.DataFrame(all_metrics)
    pooled = metrics.loc[metrics.scope.eq("pooled")]
    save(pd.concat(all_rows, ignore_index=True), "later_query_predictions.csv")
    save(metrics, "metrics.csv")
    save(pooled, "pooled_metrics.csv")
    save(pd.DataFrame(rate_medians), "rate_input_summary.csv")
    unchanged = {path: sha(Path(path)) == digest for path, digest in initial_hashes.items()}
    assert all(unchanged.values())
    assert all(sha(Path(path)) == digest for path, digest in provenance.items())
    qa = dict(all_checks_passed=True, sources_unchanged=unchanged, source_sha256=initial_hashes,
              script_and_plan_sha256=provenance, no_model_fitting_or_selection=True,
              identical_scoring_rows_and_targets=True, unchanged_baseline_and_beta=True,
              changed_feature=RATE, chronological_models_use_own_past_only_imputers=True,
              n_prediction_rows=sum(len(frame) for frame in all_rows), n_metric_rows=len(metrics),
              protocols=audits, no_bootstrap=True,
              scope="saved-model single-query-input sensitivity; not end-to-end prospective validation")
    (OUT / "QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 时间顺序模型：缺失 range-rate 查询策略敏感性", "", "## 固定比较与结果", "",
             "未训练或重新选择任何模型。先复现四个 v13 保存模型的后续预测，再仅将后续查询中的 receiver-motion range-rate 设为缺失；每个模型使用自身早期拟合的中位数。基线、持续修正、目标、评分样本和输出滤波保持不变。", "",
             "| 训练至 | 后续样本 | 物理基线 RMSE | + beta RMSE | 原 HGB RMSE | 缺失策略 HGB RMSE | 相对原 HGB 变化 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in pooled.itertuples():
        lines.append(f"| {row.milestone} | {row.n:,} | {row.physical_rmse_dbhz:.3f} | {row.beta_rmse_dbhz:.3f} | {row.original_hgb_rmse_dbhz:.3f} | {row.imputed_hgb_rmse_dbhz:.3f} | {row.delta_imputed_minus_original_rmse_dbhz:+.3f} |")
    lines.extend(["", "单位采用原稿约定：RMSE 为 dB-Hz；差异均是配对点估计，无显著性判断。四个截止点评分集不同且重叠，不能合并或解释为同一测试集上的学习曲线。", "", "## 各阶段", "", "| 训练至 | 后续阶段 | 样本 | 原 HGB RMSE | 缺失策略 HGB RMSE | 变化 |", "|---|---|---:|---:|---:|---:|"])
    for row in metrics.loc[metrics.scope.eq("phase")].itertuples():
        lines.append(f"| {row.milestone} | {row.group} | {row.n:,} | {row.original_hgb_rmse_dbhz:.3f} | {row.imputed_hgb_rmse_dbhz:.3f} | {row.delta_imputed_minus_original_rmse_dbhz:+.3f} |")
    lines.extend(["", "## 边界与文件", "", "该检查只衡量已记录的缺失 range-rate 输入策略对现有后续操作预测的影响，不验证其他缺失输入、真实预报轨道/姿态误差、前瞻性模型选择或未跟踪信号的可用性。观察支持、完整操作离线趋势与历史固定架构限制仍然存在。", "", "- `later_query_predictions.csv`：逐行原策略与缺失策略预测。", "- `metrics.csv`：全部 pooled、phase、operation 配对指标，含等阶段/等操作权重及偏差。", "- `pooled_metrics.csv`：四个截止点总表。", "- `rate_input_summary.csv`：四个各自过去拟合的中位数和输入范围。", "- `QA.json`：复现误差、单特征修改检查、数据不变哈希。", "", "所有新输出仅保存于本目录；原文稿、v13 输出、源数据和原模型均未修改。", ""])
    (OUT / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(pooled[["milestone", "n", "original_hgb_rmse_dbhz", "imputed_hgb_rmse_dbhz", "delta_imputed_minus_original_rmse_dbhz"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
