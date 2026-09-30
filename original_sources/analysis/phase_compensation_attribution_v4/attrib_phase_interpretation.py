"""Small reproducible report from existing attribution CSVs; no predictions/fit."""
from pathlib import Path
import sys
sys.path.insert(0, r"D:\月球导航\runtime_cache\python_deps")
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
GROUPS = ["TX_direction", "RX_direction", "Atmospheric_proxies", "Limb_proximity", "Composite_baseline_range", "Instrument_band_constants"]
SHORT = dict(zip(GROUPS, ["TX", "RX", "Atmos", "Limb", "Composite", "Constants"]))
PHASES = ["C", "T", "L", "S"]


def main():
    table = pd.read_csv(OUT / "attrib_phase_summary.csv", float_precision="round_trip")
    summaries = table.loc[table["mask"].eq("primary_continuity") & table.scope.eq("all_eligible") & table.level.eq("phase")]
    rows = pd.read_csv(OUT / "attrib_rows_empirical_fit32.csv", float_precision="round_trip")
    ranges = []
    for phase in PHASES:
        stage = summaries.loc[summaries.mission_phase.eq(phase) & summaries.weighting.eq("sample_weighted")].set_index("reference")
        assert len(stage) == 2
        for group in GROUPS:
            signed = stage[group + "_signed_mean_db"]
            ranges.append(dict(mission_phase=phase, group=group,
                mean_abs_min_db=stage[group + "_mean_abs_db"].min(), mean_abs_max_db=stage[group + "_mean_abs_db"].max(),
                share_min_pct=stage[group + "_share_pct"].min(), share_max_pct=stage[group + "_share_pct"].max(),
                signed_mean_empirical_db=signed.loc["empirical_fit32"], signed_mean_phase_balanced_db=signed.loc["phase_balanced_fit32"],
                sign_stable=bool((signed.eq(0).all()) or (signed.gt(0).all()) or (signed.lt(0).all()))))
    ranges = pd.DataFrame(ranges)
    ranges.to_csv(OUT / "attrib_phase_interpretation_ranges.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    magnitudes = []
    for phase in PHASES + ["ALL"]:
        stage = rows if phase == "ALL" else rows.loc[rows.mission_phase.eq(phase)]
        beta = float(stage.beta_db.abs().mean())
        delta = float(stage.delta_eval_db.abs().mean())
        magnitudes.append(dict(mission_phase=phase, n=len(stage), mean_abs_beta_db=beta, mean_abs_delta_eval_db=delta,
            sum_component_mean_abs_db=beta + delta, beta_magnitude_share_pct=100 * beta / (beta + delta),
            delta_magnitude_share_pct=100 * delta / (beta + delta), mean_abs_total_compensation_db=float(stage.total_compensation_db.abs().mean()),
            delta_eval_mean_db=float(stage.delta_eval_db.mean()), delta_eval_rms_db=float(np.sqrt(np.mean(stage.delta_eval_db ** 2)))))
    magnitudes = pd.DataFrame(magnitudes)
    magnitudes.to_csv(OUT / "attrib_compensation_magnitude.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    report = ["# 阶段补偿解释：可报告的模式与限制", "", "样本为原质量规则15806条；同一final-refit模型，含非留出描述，不能当作各OP的LOPO归因。范围来自两个跨phase固定、分signal参考（empirical_fit32与phase_balanced_fit32），不是置信区间。下表为样本加权；signed列依次为经验/阶段平衡参考。", "", "## 归因幅度与方向", "", "|Phase|组|mean absolute归因范围 dB|份额范围 %|signed mean dB：经验 / 平衡|", "|---|---|---:|---:|---:|"]
    for rec in ranges.itertuples(index=False):
        report.append(f"|{rec.mission_phase}|{SHORT[rec.group]}|{rec.mean_abs_min_db:.4f}–{rec.mean_abs_max_db:.4f}|{rec.share_min_pct:.2f}–{rec.share_max_pct:.2f}|{rec.signed_mean_empirical_db:+.4f} / {rec.signed_mean_phase_balanced_db:+.4f}|")
    report += ["", "Composite为复合基线/距离状态组，不是自由空间损耗误差；TX包含方向、方向图及发射功率信息；Atmos是原pipeline可用的吸收/气体代理，不覆盖所有大气/电离层效应。Constants在同signal背景下不变而归因零，不能推断噪温/实现损耗等实际为零。", "", "## 第一、第二组稳健性", ""]
    for phase in PHASES:
        choices = summaries.loc[summaries.mission_phase.eq(phase)]
        first = {value.split(";")[0] for value in choices.ranking}
        second = {value.split(";")[1] for value in choices.ranking}
        assert len(first) == len(second) == 1
        report.append(f"- {phase}：第一组{SHORT[next(iter(first))]}，第二组{SHORT[next(iter(second))]}；在两种共同参考×样本/四signal等权的4种设置中都保持。")
    flips = ranges.loc[~ranges.sign_stable]
    report += ["", "方向发生参考依赖翻转的组：" + ("；".join(f"{r.mission_phase}-{SHORT[r.group]}" for r in flips.itertuples(index=False)) if len(flips) else "无") + "。即使符号一致，其含义也只是相对各参考的预测变化方向，不是同名物理损耗绝对正负。C/S复合组份额随参考有约9/7个百分点变化，第三名以下顺序也有变化，故不应宣称精确损失分摊。", "", "可用的最小结论：β为所有阶段共有的大幅负校准项；HGB的状态相关部分主要由复合预算/距离信息表达。其之外，C/S的TX信息、T/L的limb/proximity信息相对更突出。这是模型使用的信息结构，与候选物理机制相容，但不能据此唯一拆成天线、大气或其他真实损耗的百分比。", "", "## β与最终δ的幅度比较", "", "定义 Aβ=mean(|β|)，Aδ=mean(|δ_eval|)，幅度比例为 Aβ/(Aβ+Aδ) 与 Aδ/(Aβ+Aδ)。分母是两个分量绝对幅度之和，不是mean(|β+δ|)、不是RMSE下降，也不是物理损耗之和；相反方向的补偿不先相消。", "", "|Phase|n|Aβ dB|Aδ dB|β / δ幅度比例 %|", "|---|---:|---:|---:|---:|"]
    for rec in magnitudes.itertuples(index=False):
        report.append(f"|{rec.mission_phase}|{rec.n}|{rec.mean_abs_beta_db:.4f}|{rec.mean_abs_delta_eval_db:.4f}|{rec.beta_magnitude_share_pct:.2f} / {rec.delta_magnitude_share_pct:.2f}|")
    report += ["", "建议不在已很宽的表1B增加这一百分比列：表中β mean与δ mean(RMS)已能显示两层补偿量级，而再放第二类不同分母百分比容易被误读。若正文确需一句量化比较，写“按mean-absolute-component之和归一化，β幅度约为×%，这不构成损耗归因”，并引用独立CSV；不要与五组SHAP份额相乘来生成总物理损失比例。", "", "完整证据：attrib_phase_interpretation_ranges.csv、attrib_compensation_magnitude.csv。没有新增模型训练或预测，没有修改正文。"]
    (OUT / "phase_interpretation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(magnitudes.to_string(index=False))


if __name__ == "__main__":
    main()
