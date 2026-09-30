"""Prepare an all-operation Table 1 candidate without editing the manuscript.

The 18 published LOPO rows and their historical confidence intervals are copied
from saved metrics. Baseline/geometry columns are descriptive on the previous
quality mask. No fitting, new predictions, or occultation exclusions are made.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\月球导航\runtime_cache\python_deps")
import numpy as np
import pandas as pd

WORKSPACE = Path(r"D:\月球导航")
MANUSCRIPT = WORKSPACE / "fabio_xz/AI-driven cislunar GNSS channel modelling from lunar GNSS observations"
OUT = Path(__file__).resolve().parent
PHYSICAL = MANUSCRIPT / "analysis/phase_residual_analysis_v1/prepared/physical_minutes.csv"
MAIN = WORKSPACE / "table/algorithm/cn0_trend_residual_tuned_no_leakage/cn0_trend_residual_tuned_no_leakage_predictions.csv"
LOPO_DIR = WORKSPACE / "table/algorithm/leave_one_operation_out"
RESULTS = MANUSCRIPT / "sections/results.tex"
KEY = ["op", "system", "signal_name", "svid", "source_bin_gps_seconds"]
PHASE_NAMES = {"C": "Commissioning", "T": "Trans-lunar", "L": "Lunar orbit", "S": "Surface"}
PUBLISHED_FIELDS = ["rmse_l1_dbhz", "rmse_l1_hgb_dbhz", "delta_rmse_hgb_minus_l1_dbhz",
                    "delta_rmse_ci95_low_dbhz", "delta_rmse_ci95_high_dbhz"]


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def csv(frame, filename):
    frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig", float_format="%.17g")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def numerical_summary(g):
    """Main-source target and baseline; old eligibility applied by caller."""
    result = {"n_quality_eligible": len(g), "n_unique_minute_epochs": g.source_bin_gps_seconds.nunique(),
              "n_links": len(g[["system", "signal_name", "svid"]].drop_duplicates()),
              "n_earth_blocked_retained": int(g.earth_blocked.fillna(0).gt(0).sum()),
              "n_moon_blocked_retained": int(g.moon_blocked.fillna(0).gt(0).sum())}
    a = g.cn0_physics_trend_dbhz.to_numpy() - g.cn0_trend_target_dbhz.to_numpy()
    result.update({"baseline_rmse_dbhz": float(np.sqrt(np.mean(a * a))) if len(a) else np.nan,
                   "baseline_mae_dbhz": float(np.mean(abs(a))) if len(a) else np.nan,
                   "baseline_minus_observed_median_db": float(np.median(a)) if len(a) else np.nan,
                   "baseline_minus_observed_mean_db": float(np.mean(a)) if len(a) else np.nan})
    for field, prefix, scale in [("geometric_range_km", "tx_rx_range", 1.0),
                                 ("cn0_dbhz_mean", "observed_1min", 1.0),
                                 ("cn0_trend_target_dbhz", "observed_trend", 1.0)]:
        values = g[field].to_numpy(dtype=float) / scale
        values = values[np.isfinite(values)]
        unit = "km" if field == "geometric_range_km" else "dbhz"
        result[f"n_finite_{prefix}"] = len(values)
        for name, quantile in [("min", 0), ("p05", .05), ("p25", .25), ("median", .5),
                               ("p75", .75), ("p95", .95), ("max", 1)]:
            result[f"{prefix}_{name}_{unit}"] = float(np.quantile(values, quantile)) if len(values) else np.nan
    result["tx_rx_range_median_1e3_km"] = result["tx_rx_range_median_km"] / 1000
    result["tx_rx_range_min_1e3_km"] = result["tx_rx_range_min_km"] / 1000
    result["tx_rx_range_max_1e3_km"] = result["tx_rx_range_max_km"] / 1000
    return result


def lopo_summary(g):
    result = {"n_lopo": len(g)}
    if not len(g):
        for name in ["lopo_beta_only_rmse_dbhz", "lopo_final_rmse_dbhz", "lopo_hgb_delta_rmse_dbhz",
                     "lopo_hgb_correction_rms_db", "lopo_hgb_correction_median_db",
                     "lopo_beta_correction_median_db"]:
            result[name] = np.nan
        return result
    beta = g.signal_beta_train_only_db.to_numpy()
    # Stored smoothed field is beta+HGB; this subtraction isolates HGB-only.
    hgb = g.hgb_residual_smoothed_pred_db.to_numpy() - beta
    target = g.cn0_trend_target_dbhz.to_numpy()
    l1 = np.sqrt(np.mean((g.cn0_robust_l1_only_dbhz.to_numpy() - target) ** 2))
    final = np.sqrt(np.mean((g.cn0_robust_l1_hgb_dbhz.to_numpy() - target) ** 2))
    return {"n_lopo": len(g), "lopo_beta_only_rmse_dbhz": l1, "lopo_final_rmse_dbhz": final,
            "lopo_hgb_delta_rmse_dbhz": final - l1,
            "lopo_hgb_correction_rms_db": np.sqrt(np.mean(hgb * hgb)),
            "lopo_hgb_correction_median_db": np.median(hgb),
            "lopo_beta_correction_median_db": np.median(beta)}


def use_matching_baseline_target(record, outer):
    """Do not mix main split-smoothed targets with outer-LOPO targets."""
    fields = ["baseline_rmse_dbhz", "baseline_mae_dbhz", "baseline_minus_observed_median_db", "baseline_minus_observed_mean_db"]
    for field in fields:
        record["main_" + field] = record[field]
    record["baseline_target_source"] = "main_saved_target_descriptive_only"
    if len(outer):
        error = outer.cn0_physics_trend_dbhz.to_numpy() - outer.cn0_trend_target_dbhz.to_numpy()
        values = [np.sqrt(np.mean(error * error)), np.mean(abs(error)), np.median(error), np.mean(error)]
        record.update(dict(zip(fields, values)))
        record["baseline_target_source"] = "LOPO_saved_target_matches_heldout_RMSE_columns"
    return record


def original_tex_cells():
    snapshot = OUT / "table_historical_lopo_display.json"
    if snapshot.exists():
        saved = json.loads(snapshot.read_text(encoding="utf-8"))
        cells = saved["operation_cells"]
        assert len(cells) == 18 and all(len(v) == 7 for v in cells.values())
        return cells
    text = RESULTS.read_text(encoding="utf-8")
    after_label = text.split(r"\label{tab:lopo_sensitivity}", 1)[1].split(r"\end{table}", 1)[0]
    cells = {}
    for line in after_label.splitlines():
        if re.match(r"\s*OP\d+\s*&", line):
            parts = [v.strip() for v in line.strip().removesuffix(r"\\").split("&")]
            assert len(parts) == 7, parts
            cells[parts[0]] = parts
    assert len(cells) == 18
    snapshot.write_text(json.dumps({"source": str(RESULTS), "source_sha256": digest(RESULTS),
                                    "purpose": "Freeze original seven-column LOPO display for reruns after manuscript revision",
                                    "operation_cells": cells}, ensure_ascii=False, indent=2), encoding="utf-8")
    return cells


def num(v, decimals=1):
    return "---" if not np.isfinite(v) else f"{v:.{decimals}f}"


def marked(value, new=False):
    return (r"\priorchange{" if new else r"\revised{") + str(value) + "}"


def make_tex(table, published, lopo, good):
    # Kept as a candidate only; root reviews and integrates the final layout.
    lines = [r"% Candidate replacement only; sections/results.tex is not modified.",
             r"% Existing 18 LOPO RMSE, delta and interval display cells are preserved verbatim.",
             r"\begin{table*}[t]", r"  \centering", r"  \scriptsize",
             r"  \setlength{\tabcolsep}{2.2pt}",
             r"  \caption{\revised{Leave-one-operation-out sensitivity of the selected model configuration. Each operation was excluded from estimation of its fold's persistent corrections and feature medians and from fitting the HGB estimator. RMSE values and changes are in dB-Hz; intervals are the existing 95\% block-bootstrap intervals for HGB-on minus HGB-off RMSE. Changes were calculated from unrounded values.} \priorchange{Range is the median transmitter--receiver distance in $10^3$ km; $m_B$ is the median baseline-minus-observed discrepancy in dB. For the 18 LOPO operations, baseline and learned-model errors use the same saved outer-fold targets. OP3 has 3 original rows but no eligible samples. OP37 has 23 original rows and 13 eligible samples, below the LOPO minimum of 20; its distance and baseline summaries are descriptive only. Dashes indicate unavailable results. Samples follow the previous quality mask, with no new occultation exclusion.} \revised{The pooled row weights all one-minute link samples equally. Mean operation RMSE is the arithmetic mean of the 18 operation-specific values. C, T, L and S denote commissioning, trans-lunar flight, lunar orbit and surface operation.}}",
             r"  \label{tab:lopo_sensitivity}", r"  \begin{tabular}{llrrrrrrrl}",
             r"    \toprule",
             r"    \revised{Operation} & \revised{Phase} & \revised{Samples} & \priorchange{Range} & \priorchange{$m_B$} & \priorchange{Baseline} & \revised{Robust $\mathcal{L}_1$} & \revised{Final} & \revised{Change} & \revised{95\% interval} \\",
             r"    \midrule"]
    last_phase = None
    for _, r in table.iterrows():
        if last_phase is not None and r.mission_phase != last_phase:
            lines.append(r"    \addlinespace[2pt]")
        last_phase = r.mission_phase
        if r.op in published:
            orig = published[r.op]
            l1, final, delta, ci = orig[3:7]
            n = orig[2]
        else:
            l1 = final = delta = ci = "---"
            n = str(int(r.n_quality_eligible))
        new_op = r.op not in published
        cells = [marked(r.op, new_op), marked(r.mission_phase, new_op), marked(n, new_op),
                 marked(num(r.tx_rx_range_median_1e3_km), True),
                 marked(num(r.baseline_minus_observed_median_db), True), marked(num(r.baseline_rmse_dbhz), True),
                 marked(l1, new_op), marked(final, new_op), marked(delta, new_op), marked(ci, new_op)]
        lines.append("    " + " & ".join(cells) + r" \\")
    summary = read(LOPO_DIR / "cn0_lopo_summary_metrics.csv")
    pooled = summary.loc[summary.summary_scope.eq("pooled")].iloc[0]
    pool_error = lopo.cn0_physics_trend_dbhz - lopo.cn0_trend_target_dbhz
    pool_geometry = lopo[KEY].merge(good[KEY + ["geometric_range_km"]], on=KEY, validate="one_to_one")
    pool_cells = [marked("Pooled"), marked("All"), marked("15,793"),
                  marked(num(pool_geometry.geometric_range_km.median() / 1000), True),
                  marked(num(pool_error.median()), True), marked(num(np.sqrt(np.mean(pool_error ** 2))), True),
                  marked("2.1"), marked("1.9"), marked(r"\(-0.2\)"), marked(r"[\(-0.3\), \(-0.08\)]")]
    macro_baseline = table.loc[table.n_lopo.gt(0), "baseline_rmse_dbhz"].mean()
    lines += [r"    \midrule", "    " + " & ".join(pool_cells) + r" \\",
              r"    \multicolumn{3}{l}{\revised{Mean operation RMSE}} & \priorchange{---} & \priorchange{---} & " + marked(num(macro_baseline), True) + r" & \revised{2.1} & \revised{1.6} & \revised{\(-0.5\)} & \revised{---} \\",
              r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    assert int(pooled.n) == 15793
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [MAIN, PHYSICAL, LOPO_DIR / "cn0_lopo_predictions.csv",
             LOPO_DIR / "cn0_lopo_per_operation_metrics.csv", LOPO_DIR / "cn0_lopo_operation_eligibility.csv", RESULTS]
    before = {str(p): digest(p) for p in paths}
    main_data = read(MAIN)
    physical = read(PHYSICAL)
    lopo = read(LOPO_DIR / "cn0_lopo_predictions.csv")
    historical = read(LOPO_DIR / "cn0_lopo_per_operation_metrics.csv").set_index("op")
    eligibility = read(LOPO_DIR / "cn0_lopo_operation_eligibility.csv").set_index("op")
    assert not main_data.duplicated(KEY).any() and not physical.duplicated(KEY).any() and not lopo.duplicated(KEY).any()
    selected_physical = ["geometric_range_km", "earth_blocked", "moon_blocked"]
    d = main_data.merge(physical[KEY + selected_physical], on=KEY, how="left", validate="one_to_one", indicator=True)
    assert d._merge.eq("both").all()
    d.drop(columns="_merge", inplace=True)
    d["quality_eligible_analysis"] = (d.trend_training_eligible.astype(str).str.lower().eq("true") &
                                      np.isfinite(d[["cn0_trend_target_dbhz", "cn0_physics_trend_dbhz"]]).all(axis=1))
    good = d.loc[d.quality_eligible_analysis].copy()
    # Keys and baselines agree; targets can differ with split-isolated smoothing.
    matches = lopo[KEY + ["cn0_trend_target_dbhz", "cn0_physics_trend_dbhz"]].merge(
        good[KEY + ["cn0_trend_target_dbhz", "cn0_physics_trend_dbhz"]], on=KEY, suffixes=("_lopo", "_main"),
        how="left", validate="one_to_one", indicator=True)
    assert matches._merge.eq("both").all()
    alignment = {}
    for field in ["cn0_trend_target_dbhz", "cn0_physics_trend_dbhz"]:
        maximum = float(abs(matches[field + "_lopo"] - matches[field + "_main"]).max())
        alignment[field] = {"maximum_absolute_error": maximum,
                            "n_differences_above_1e_minus10": int((abs(matches[field + "_lopo"] - matches[field + "_main"]) > 1e-10).sum())}
        if field == "cn0_physics_trend_dbhz":
            assert maximum < 1e-10
    operations, signals = [], []
    for op in sorted(d.op.unique(), key=lambda x: int(x.removeprefix("OP"))):
        raw = d.loc[d.op.eq(op)]
        g = good.loc[good.op.eq(op)]
        outer = lopo.loc[lopo.op.eq(op)]
        source_elig = eligibility.loc[op]
        assert len(raw) == source_elig.rows_1min_total
        assert len(g) == source_elig.rows_1min_quality_eligible
        assert len(outer) == (len(g) if bool(source_elig.eligible_for_lopo) else 0)
        phase = raw.mission_phase.iloc[0]
        header = {"op": op, "mission_phase": phase, "phase_name": PHASE_NAMES[phase],
                  "n_original_1min_rows": len(raw), "eligible_for_lopo": bool(source_elig.eligible_for_lopo),
                  "lopo_exclusion_reason": source_elig.exclusion_reason if isinstance(source_elig.exclusion_reason, str) else "",
                  "lopo_bootstrap_15min_blocks": int(source_elig.bootstrap_15min_blocks),
                  "descriptive_source": "saved main prediction table; previous quality mask",
                  "lopo_source": "saved fixed-architecture outer LOPO" if len(outer) else "not evaluated"}
        record = {**header, **numerical_summary(g), **lopo_summary(outer)}
        record = use_matching_baseline_target(record, outer)
        for name in PUBLISHED_FIELDS:
            record["published_" + name] = historical.loc[op, name] if op in historical.index else np.nan
        if len(outer):
            assert np.isclose(record["lopo_beta_only_rmse_dbhz"], historical.loc[op, "rmse_l1_dbhz"], rtol=0, atol=1e-12)
            assert np.isclose(record["lopo_final_rmse_dbhz"], historical.loc[op, "rmse_l1_hgb_dbhz"], rtol=0, atol=1e-12)
            # Authoritative table fields are copied, not recomputed/rebootstrapped.
            record["lopo_beta_only_rmse_dbhz"] = historical.loc[op, "rmse_l1_dbhz"]
            record["lopo_final_rmse_dbhz"] = historical.loc[op, "rmse_l1_hgb_dbhz"]
            record["lopo_hgb_delta_rmse_dbhz"] = historical.loc[op, "delta_rmse_hgb_minus_l1_dbhz"]
        operations.append(record)
        for signal, raw_signal in raw.groupby("signal_name", sort=True):
            s = g.loc[g.signal_name.eq(signal)]
            fold = outer.loc[outer.signal_name.eq(signal)]
            signal_record = {**header, "signal_name": signal, "n_original_1min_rows": len(raw_signal),
                            **numerical_summary(s), **lopo_summary(fold),
                            "confidence_interval_status": "not estimated for OP-by-signal groups"}
            signals.append(use_matching_baseline_target(signal_record, fold))
    table = pd.DataFrame(operations)
    signal_table = pd.DataFrame(signals)
    for field in ["n_original_1min_rows", "n_quality_eligible", "n_lopo"]:
        assert signal_table.groupby("op")[field].sum().to_dict() == table.set_index("op")[field].to_dict()
    assert len(table) == 20 and table.n_original_1min_rows.sum() == 17439
    assert table.n_quality_eligible.sum() == 15806 and table.n_lopo.sum() == 15793
    assert int(table.n_quality_eligible.gt(0).sum()) == 19
    assert int(table.n_earth_blocked_retained.sum()) == 72
    published = original_tex_cells()
    tex = make_tex(table, published, lopo, good)
    csv(table, "all_operation_table.csv")
    saved_table = read(OUT / "all_operation_table.csv").set_index("op")
    for op in historical.index:
        for field in PUBLISHED_FIELDS:
            assert saved_table.loc[op, "published_" + field] == historical.loc[op, field]
    csv(signal_table, "all_operation_signal_detail.csv")
    range_columns = ["op", "mission_phase", "phase_name", "n_original_1min_rows", "n_quality_eligible",
                     "n_finite_tx_rx_range", "tx_rx_range_min_km", "tx_rx_range_p05_km", "tx_rx_range_median_km",
                     "tx_rx_range_p95_km", "tx_rx_range_max_km", "tx_rx_range_median_1e3_km",
                     "n_earth_blocked_retained", "n_moon_blocked_retained"]
    csv(table.loc[table.n_quality_eligible.gt(0), range_columns], "table_observed_operation_physical_range.csv")
    op37 = good.loc[good.op.eq("OP37"), KEY + ["mission_phase", "minute_utc", "evaluation_split",
                  "cn0_dbhz_mean", "cn0_trend_target_dbhz", "cn0_physics_trend_dbhz", "geometric_range_km",
                  "earth_blocked", "moon_blocked", "trend_training_eligible"]].copy()
    op37["baseline_minus_observed_db"] = op37.cn0_physics_trend_dbhz - op37.cn0_trend_target_dbhz
    assert len(op37) == 13
    csv(op37, "table_op37_eligible_rows.csv")
    (OUT / "all_operation_table.tex").write_text(tex, encoding="utf-8")
    unchanged = all(digest(p) == before[str(p)] for p in paths)
    qa = {"checks_passed": True, "source_files_unchanged": unchanged, "source_sha256": before,
          "n_original_rows": len(d), "n_all_operations": len(table), "n_eligible_rows": len(good),
          "n_eligible_operations": 19, "n_lopo_rows": len(lopo), "n_lopo_operations": len(historical),
          "n_earth_blocked_retained": int(table.n_earth_blocked_retained.sum()),
          "main_lopo_alignment_max_abs_error": alignment,
          "historical_lopo_fields_copied_exactly": PUBLISHED_FIELDS,
          "historical_tex_operation_cells_preserved": len(published),
          "historical_tex_display_snapshot": "table_historical_lopo_display.json",
          "csv_historical_numeric_roundtrip_exact": True,
          "op3": table.loc[table.op.eq("OP3")].replace({np.nan: None}).to_dict(orient="records")[0],
          "op37": table.loc[table.op.eq("OP37")].replace({np.nan: None}).to_dict(orient="records")[0],
          "new_training": False, "new_predictions": False, "new_bootstrap": False,
          "primary_source_of_hgb_correction": "LOPO smoothed total residual minus fold-specific signal beta"}
    assert unchanged
    (OUT / "table_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    r37 = table.loc[table.op.eq("OP37")].iloc[0]
    notes = ["# 全 OP 表候选与数据口径", "",
             "## 排版建议", "",
             "按主任务确认，候选 LaTeX 是紧凑 10 列 table*：OP、phase、Samples、Range median、Baseline−observed median、Baseline RMSE、L1 RMSE、Final RMSE、ΔRMSE、CI。HGB RMS 仅保留在 CSV。新增三列及 OP3/OP37 用 priorchange 标蓝，既有 LOPO 列用 revised 标红。尚未修改正文。", "",
             "## 数据与保留规则", "",
             "- 所有 20 OP 和 17,439 原始一分钟行均被计数。按上一轮 eligibility，19 个 OP 共 15,806 行可作描述统计；18 OP 共 15,793 行有既有 LOPO 结果。",
             "- 几何距离来自上一轮与主表逐键一致的 prepared/physical_minutes.csv，指 GNSS 发射机到接收机的距离，不是接收机地心距。CSV 同时保留 km 及 10^3 km。",
             "- 沿用 trend_training_eligible 且目标/基线有限的筛选；保留原 pipeline 中 72 条 earth_blocked 行，不进行新的筛除。",
             "- 主表与 LOPO 键和物理基线一致，但目标在部分行因平滑隔离范围不同而不相同（最大差 7.321999 dB）。因此 18 个 LOPO OP 的新增 baseline RMSE/median 使用同一 LOPO target，避免表内跨目标比较。OP37 只能使用主表 target 作描述。main_baseline_* 另保留主表原目标统计，baseline_target_source 明确来源。observed_trend_* 描述列保持主表目标。",
             "- 新增的 HGB RMS 只来自已保存 LOPO 外层预测：hgb_residual_smoothed_pred_db 减 signal_beta_train_only_db。不得用主模型 refit 的预测补齐缺失 LOPO 值。",
             "- 18 OP 原 RMSE、delta 与 95% CI 从 cn0_lopo_per_operation_metrics.csv 原值复制；候选 tex 中这几列逐格保留当前 results.tex 的显示文本，不重算区间。",
             "- OP×signal 表只重新汇总已保存的 LOPO 预测，不增加训练；未估计细分组 CI，不将 OP 级 CI 冒充信号级区间。", "",
             "## OP3 与 OP37", "",
             "- OP3：原 3 行、合格 0 行、合格 bootstrap block 0；原排除原因 eligible_rows<20;bootstrap_blocks<2。表中物理描述与 LOPO 量均留空，不能据原始不合格行填造合格结果。",
             f"- OP37：原 23 行、合格 13 行、合格块 2；仅因 eligible_rows<20 排除。13 行描述基线 RMSE={r37.baseline_rmse_dbhz:.9f} dB，基线减观测中位数={r37.baseline_minus_observed_median_db:.9f} dB，距离中位数={r37.tx_rx_range_median_1e3_km:.9f}×10^3 km。LOPO列留空，13条源行另存 table_op37_eligible_rows.csv。", "",
             "全部输入文件与论文源文件 SHA-256 保持不变。未拟合、未预测、未重新 bootstrap。", ""]
    (OUT / "table_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(table[["op", "mission_phase", "n_original_1min_rows", "n_quality_eligible", "n_lopo",
                 "tx_rx_range_median_1e3_km", "baseline_rmse_dbhz", "baseline_minus_observed_median_db",
                 "lopo_hgb_correction_rms_db"]].to_string(index=False))
    print(json.dumps({"checks_passed": True, "source_files_unchanged": unchanged}, indent=2))


if __name__ == "__main__":
    main()
