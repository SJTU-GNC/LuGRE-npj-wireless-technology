"""Reproducible, descriptive final-correction and thermal diagnostic grids."""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = PROJECT.parents[1]
sys.path.insert(0, str(WORKSPACE / "runtime_cache/python_deps"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

SIGNALS = ["GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"]
LABELS = ["GPS L1", "GPS L5", "Galileo E1", "Galileo E5a"]
COLORS = dict(zip(SIGNALS, ["#286895", "#D58B35", "#539282", "#956CA6"]))
STYLES = dict(zip(SIGNALS, ["-", "--", "-.", ":"]))
OPS = [f"OP{x}" for x in [1, 2, 3, 5, 9, 12, 14, 17, 18, 21, 22, 23, 27, 37, 38, 40, 74, 76, 77, 78]]
SURFACE_OPS = ["OP38", "OP40", "OP74", "OP76", "OP77", "OP78"]
INPUTS = {
    "final_corrections": PROJECT / "analysis/phase_compensation_attribution_v4/attrib_rows.csv",
    "thermal_bins": PROJECT / "analysis/thermal_sensitivity_v8/surface_temperature_binned_residuals.csv",
    "thermal_pairs": PROJECT / "analysis/thermal_sensitivity_v8/surface_paired_temperature_sensitivity.csv",
    "thermal_summary": PROJECT / "analysis/thermal_sensitivity_v8/surface_temperature_operation_summary.csv",
    "thermal_matched": PROJECT / "analysis/thermal_sensitivity_v8/surface_temperature_matched_residuals.csv",
    "sensor_series": PROJECT / "analysis/thermal_sensitivity_v8/alex_temperature_inputs/surface_series.csv",
    "coverage": PROJECT / "analysis/thermal_sensitivity_v8/surface_temperature_match_coverage.csv",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(df, name):
    df.to_csv(HERE / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def utc(s):
    return pd.to_datetime(s, utc=True, format="mixed")


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def detrended(g, col):
    y = g[col].to_numpy(float)
    t = (g.timestamp_utc - g.timestamp_utc.min()).dt.total_seconds().to_numpy() / 3600
    ok = np.isfinite(y)
    z = np.full(len(y), np.nan)
    design = np.column_stack([np.ones(ok.sum()), t[ok]])
    if ok.sum() >= 3:
        z[ok] = y[ok] - design @ np.linalg.lstsq(design, y[ok], rcond=None)[0]
    return z


def corr_record(g, scope, segment):
    g = g.sort_values("timestamp_utc")
    span = float(g.thermal_shift_db.max() - g.thermal_shift_db.min())
    r = {
        "op": g.op.iloc[0], "signal_name": g.signal_name.iloc[0],
        "scope": scope, "segment": segment, "n_graph_points": len(g),
        "n_link_minutes": int(g.n_link_minutes.sum()),
        "start_utc": g.timestamp_utc.min(), "end_utc": g.timestamp_utc.max(),
        "proxy_span_db": span,
        "r_proxy_delta": corr(g.thermal_shift_db, g.hgb_correction_median_db),
        "r_proxy_delta_detrended": corr(detrended(g, "thermal_shift_db"), detrended(g, "hgb_correction_median_db")) if len(g) >= 8 else np.nan,
        "status": "fewer_than_8_points" if len(g) < 8 else "proxy_span_below_0.05_dB" if span < .05 else "descriptive_only",
    }
    for component in ["HGA", "LNA", "Receiver"]:
        c = f"{component}_temp_c"
        paired = g[np.isfinite(g[c]) & np.isfinite(g.hgb_correction_median_db)]
        r[f"n_{component}_points"] = len(paired)
        r[f"r_{component}_temperature_delta"] = corr(paired[c], paired.hgb_correction_median_db)
        r[f"r_{component}_temperature_delta_detrended"] = corr(detrended(paired, c), detrended(paired, "hgb_correction_median_db")) if len(paired) >= 8 else np.nan
    return r


def axis_base(ax):
    ax.axhline(0, color="0.83", linewidth=.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(.5)
    ax.tick_params(width=.5, length=2, pad=1.5, labelsize=6.3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, min_n_ticks=2))


def time_axis(ax, g, col, compact=False):
    lo, hi = g[col].min(), g[col].max()
    if lo == hi:
        lo -= pd.Timedelta(minutes=1)
        hi += pd.Timedelta(minutes=1)
    pad = (hi - lo) * .04
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xticks([lo, hi])
    same_day = lo.date() == hi.date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M" if same_day else "%d %H:%M", tz=lo.tzinfo))
    labs = ax.get_xticklabels()
    labs[0].set_ha("left")
    labs[-1].set_ha("right")


def export(fig, stem):
    for ext in ["svg", "pdf"]:
        fig.savefig(HERE / f"{stem}.{ext}", facecolor="white")
    fig.savefig(HERE / f"{stem}.png", dpi=220, facecolor="white")
    plt.close(fig)


def operation_figure(df, inventory):
    fig, axes = plt.subplots(5, 4, figsize=(183 / 25.4, 225 / 25.4))
    fig.subplots_adjust(left=.075, right=.987, bottom=.064, top=.925, wspace=.34, hspace=.72)
    for i, (op, ax) in enumerate(zip(OPS, axes.flat)):
        g = df[df.op.eq(op)]
        axis_base(ax)
        inv = inventory[inventory.op.eq(op)].iloc[0]
        if g.empty:
            ax.text(.5, .5, "No eligible samples", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{chr(97+i)}  {op}", loc="left", fontsize=7, pad=4, weight="bold")
            continue
        for sig in SIGNALS:
            z = g[g.signal_name.eq(sig)].sort_values("minute_utc")
            cut = z.minute_utc.diff().dt.total_seconds().gt(90) | z.partition_set.ne(z.partition_set.shift())
            for _, seg in z.groupby(cut.cumsum()):
                ax.plot(seg.minute_utc, seg.delta_median_db, color=COLORS[sig], ls=STYLES[sig], lw=.9, marker="." if len(seg) <= 2 else None, ms=2)
        ymin, ymax = min(0, g.delta_median_db.min()), max(0, g.delta_median_db.max())
        pad = max(.2, .12*(ymax-ymin))
        ax.set_ylim(ymin-pad, ymax+pad)
        time_axis(ax, g, "minute_utc")
        start = g.minute_utc.min()
        date_label = start.strftime("%d %b")
        end = g.minute_utc.max()
        if start.date() != end.date():
            date_label += end.strftime("–%d %b")
        ax.set_title(f"{chr(97+i)}  {op} · {inv.phase}\n{date_label}; n={int(inv.n_link_minutes):,}", loc="left", fontsize=6.8, pad=4)
    handles = [Line2D([0], [0], color=COLORS[s], ls=STYLES[s], lw=1.2, label=l) for s, l in zip(SIGNALS, LABELS)]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(.54, .989), handlelength=2.8, columnspacing=1.4, fontsize=7)
    fig.text(.018, .515, r"Final state-dependent correction, $\delta$ (dB)", rotation=90, ha="center", va="center", fontsize=8)
    fig.text(.54, .025, "UTC time (2025); tick labels include day where the interval crosses midnight", ha="center", fontsize=7)
    export(fig, "supp_correction_all_operations")


def surface_figure(df, coverage):
    fig, axes = plt.subplots(6, 4, figsize=(183 / 25.4, 235 / 25.4))
    fig.subplots_adjust(left=.085, right=.987, bottom=.075, top=.924, wspace=.34, hspace=.78)
    handles = [Line2D([0], [0], color="0.25", lw=1.1, label=r"Matched $\delta$"), Line2D([0], [0], color="0.25", ls="--", lw=1.1, label=r"Temperature-derived $\Delta C/N_0$")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(.54, .998), handlelength=2.7, columnspacing=1.6, fontsize=7)
    for j, (sig, label) in enumerate(zip(SIGNALS, LABELS)):
        fig.text(.184 + j * .236, .953, label, ha="center", fontsize=8, color=COLORS[sig])
    for i, op in enumerate(SURFACE_OPS):
        row = df[df.op.eq(op)]
        ymin = min(0, row.delta_centered_db.min(), row.proxy_centered_db.min())
        ymax = max(0, row.delta_centered_db.max(), row.proxy_centered_db.max())
        pad = max(.08, .1*(ymax-ymin))
        for j, sig in enumerate(SIGNALS):
            ax = axes[i, j]
            axis_base(ax)
            g = df[df.op.eq(op) & df.signal_name.eq(sig)].sort_values("timestamp_utc")
            # Centres are computed over all matched points of one operation/signal,
            # never separately over windows, retaining between-window contrasts.
            for _, seg in g.groupby("plot_segment", sort=False):
                ax.plot(seg.timestamp_utc, seg.delta_centered_db, color=COLORS[sig], lw=1, marker="." if len(seg) <= 2 else None, ms=2)
                ax.plot(seg.timestamp_utc, seg.proxy_centered_db, color=COLORS[sig], ls="--", lw=1, marker="." if len(seg) <= 2 else None, ms=2)
            ax.set_ylim(ymin-pad, ymax+pad)
            time_axis(ax, g, "timestamp_utc")
            pc = float(coverage.loc[coverage.op.eq(op), "coverage_fraction"].iloc[0])
            ax.set_title(f"{chr(97+i*4+j)}  {op} · n={len(g)}\n{g.timestamp_utc.min():%d %b}; coverage {pc:.0%}", loc="left", fontsize=6.5, pad=3)
    fig.text(.018, .515, "Change from each matched-series median (dB)", rotation=90, ha="center", va="center", fontsize=8)
    fig.text(.54, .036, "UTC time (March 2025); n = matched graph points per signal", ha="center", fontsize=7)
    fig.text(.54, .014, "Surface graph-derived temperatures; missing intervals and OP77 windows remain disconnected", ha="center", fontsize=6.7)
    export(fig, "supp_surface_thermal_correspondence")


def main():
    hashes = {key: sha(path) for key, path in INPUTS.items()}
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False, "axes.linewidth": .5})
    cols = ["op", "mission_phase", "minute_utc", "signal_name", "svid", "evaluation_split", "delta_eval_db", "reference"]
    exact = pd.read_csv(INPUTS["final_corrections"], usecols=cols)
    exact = exact[exact.reference.eq("empirical_fit32")].copy()
    exact["minute_utc"] = utc(exact.minute_utc)
    assert len(exact) == 15806
    assert not exact.duplicated(["op", "signal_name", "svid", "minute_utc"]).any()
    ts = exact.groupby(["op", "mission_phase", "signal_name", "minute_utc"], as_index=False).agg(
        delta_median_db=("delta_eval_db", "median"), n_links=("svid", "nunique"),
        partition_set=("evaluation_split", lambda x: ";".join(sorted(set(x)))),
        delta_min_db=("delta_eval_db", "min"), delta_max_db=("delta_eval_db", "max"))
    inventory = []
    for op in OPS:
        g = exact[exact.op.eq(op)]
        inventory.append(dict(op=op, phase=g.mission_phase.iloc[0] if len(g) else {"OP3": "T"}[op], n_link_minutes=len(g), n_minute_epochs=g.minute_utc.nunique(),
            start_utc=g.minute_utc.min(), end_utc=g.minute_utc.max(), evaluation_partitions=";".join(sorted(g.evaluation_split.unique())),
            status="no_eligible_samples" if not len(g) else "small_sample_descriptive" if len(g)<20 else "descriptive_final_model_including_fitting_rows"))
    inventory = pd.DataFrame(inventory)
    write_csv(ts, "operation_signal_minute_corrections.csv")
    write_csv(inventory, "operation_inventory.csv")

    surface = pd.read_csv(INPUTS["thermal_bins"])
    surface["timestamp_utc"] = utc(surface.timestamp_utc)
    pairs = pd.read_csv(INPUTS["thermal_pairs"])
    pairs["timestamp_utc"] = utc(pairs.timestamp_utc)
    pairs = pairs.sort_values("timestamp_utc")
    pairs["source_thermal_segment"] = pairs.timestamp_utc.diff().dt.total_seconds().gt(15*60).cumsum()+1
    sensors = pd.read_csv(INPUTS["sensor_series"])
    sensors["timestamp_utc"] = utc(sensors.timestamp_utc)
    surface = surface.merge(pairs[["thermal_id", "source_thermal_segment"]], on="thermal_id", validate="many_to_one")
    for component in ["HGA", "LNA", "Receiver"]:
        sensor = sensors[sensors.sensor.eq(component)][["timestamp_utc", "temp_c"]].rename(columns={"timestamp_utc": f"{component}_timestamp_utc", "temp_c": f"{component}_temp_c"})
        surface = pd.merge_asof(surface.sort_values("timestamp_utc"), sensor.sort_values(f"{component}_timestamp_utc"), left_on="timestamp_utc", right_on=f"{component}_timestamp_utc", direction="nearest", tolerance=pd.Timedelta(seconds=60))
        surface[f"{component}_match_seconds"] = (surface.timestamp_utc-surface[f"{component}_timestamp_utc"]).dt.total_seconds().abs()
    assert surface[["HGA_temp_c", "LNA_temp_c"]].notna().all().all(), surface[surface.HGA_temp_c.isna()|surface.LNA_temp_c.isna()][["op","timestamp_utc"]].to_string()
    # The independent exact final-model minute rows reproduce the archived bins.
    matched = pd.read_csv(INPUTS["thermal_matched"])
    matched["minute_utc"] = utc(matched.minute_utc)
    joined = matched.merge(exact[["op", "signal_name", "svid", "minute_utc", "delta_eval_db"]], on=["op", "signal_name", "svid", "minute_utc"], validate="one_to_one")
    delta_error = float(np.max(np.abs(joined.hgb_correction_db-joined.delta_eval_db)))
    rebin = joined.groupby(["op", "signal_name", "thermal_id"], as_index=False).delta_eval_db.median()
    check = surface.merge(rebin, on=["op", "signal_name", "thermal_id"], validate="one_to_one")
    bin_error = float(np.max(np.abs(check.hgb_correction_median_db-check.delta_eval_db)))
    assert delta_error < 1e-10 and bin_error < 1e-10
    blocks = []
    records = []
    for _, g in surface.groupby(["op", "signal_name"], sort=False):
        g = g.sort_values("timestamp_utc").copy()
        # At least a 1-hour observation gap defines separate supported windows;
        # source thermal gaps are split even if shorter than that threshold.
        cut = g.source_thermal_segment.ne(g.source_thermal_segment.shift()) | g.timestamp_utc.diff().dt.total_seconds().gt(3600)
        g["analysis_segment"] = cut.cumsum()
        g["plot_segment"] = (cut | g.timestamp_utc.diff().dt.total_seconds().gt(15*60+5)).cumsum()
        g["delta_centered_db"] = g.hgb_correction_median_db-g.hgb_correction_median_db.median()
        g["proxy_centered_db"] = g.thermal_shift_db-g.thermal_shift_db.median()
        records.append(corr_record(g, "pooled_operation_descriptive", "all"))
        for seg, z in g.groupby("analysis_segment"):
            records.append(corr_record(z, "source_gap_and_window_separated", int(seg)))
        blocks.append(g)
    surface = pd.concat(blocks, ignore_index=True)
    correlations = pd.DataFrame(records)
    archived = pd.read_csv(INPUTS["thermal_summary"])
    comparison = correlations[correlations.scope.eq("pooled_operation_descriptive")].merge(archived, on=["op", "signal_name"], validate="one_to_one")
    raw_error = float(np.nanmax(np.abs(comparison.r_proxy_delta-comparison.pearson_r_with_hgb_correction)))
    detrend_error = float(np.nanmax(np.abs(comparison.r_proxy_delta_detrended-comparison.pearson_r_after_linear_time_detrending_with_hgb)))
    assert raw_error < 1e-8 and detrend_error < 1e-8
    write_csv(surface, "surface_plot_source_data.csv")
    write_csv(correlations, "surface_all_correlations.csv")
    coverage = pd.read_csv(INPUTS["coverage"])
    write_csv(coverage, "surface_coverage.csv")
    operation_figure(ts, inventory)
    surface_figure(surface, coverage)
    qa = {"input_sha256": hashes, "sources_unchanged": all(sha(INPUTS[k]) == h for k,h in hashes.items()),
          "final_model_link_minutes": len(exact), "operation_labels": len(OPS), "operations_with_eligible_samples": exact.op.nunique(),
          "minute_signal_medians": len(ts), "surface_graph_signal_bins": len(surface),
          "matched_surface_link_minutes": int(surface.n_link_minutes.sum()), "surface_sensor_missing_values": int(surface[["HGA_temp_c","LNA_temp_c","Receiver_temp_c"]].isna().sum().sum()),
          "max_exact_delta_error_db": delta_error, "max_reaggregated_delta_error_db": bin_error,
          "max_archived_raw_correlation_error": raw_error, "max_archived_detrended_correlation_error": detrend_error,
          "y_limits": "Full data range with padding; operation-specific in figure 1 and common within operation rows in figure 2; no clipping",
          "max_sensor_match_seconds": {component: float(surface[f"{component}_match_seconds"].max()) for component in ["HGA","LNA","Receiver"]},
          "no_thermal_interpolation": True, "correlations_are_descriptive_no_independent_sample_pvalues": True,
          "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__, "pandas_version": pd.__version__,
          "figures_mm": {"supp_correction_all_operations": [183,225], "supp_surface_thermal_correspondence": [183,235]}}
    assert qa["sources_unchanged"]
    (HERE / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
