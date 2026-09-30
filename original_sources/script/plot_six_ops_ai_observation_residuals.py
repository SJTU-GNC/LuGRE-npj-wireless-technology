#!/usr/bin/env python3
"""Plot observed-minus-model C/N0 residuals for the six-operation figure.

The script reads the frozen-model and raw-observation source data exported by
the accepted six-OP reconstruction figure. It does not retrain or refit the AI
model. The 1-min model trend is interpolated only within contiguous model arcs
and evaluated at the original approximately 1-s observation epochs.
"""

from __future__ import annotations

import math
import sys
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "script"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import plot_four_phase_holdout_six_ops_1s_reference as base


SOURCE_DIR = ROOT / "table" / "figure_source_data"
FIGURE_DIR = ROOT / "figure" / "paper_draft_v2"
INPUT_STEM = "Fig6_all_satellites_smoothed_AI_trend_vs_1s_reference"
STEM = "Fig_six_OP_AI_model_residuals_vs_observed_CN0"
MODEL_SOURCE = SOURCE_DIR / f"{INPUT_STEM}_model_1min_source_data.csv"
OBSERVATION_SOURCE = SOURCE_DIR / f"{INPUT_STEM}_observed_1s_source_data.csv"

MODEL_MAX_GAP = pd.Timedelta(minutes=1.5)
TREND_MAX_GAP = pd.Timedelta(minutes=2.0)
TREND_WINDOW_MINUTES = 5


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_columns = [
        "rx_utc",
        "evaluation_split",
        "op",
        "mission_phase",
        "system",
        "signal_name",
        "svid",
        "satellite_id",
        "link_id",
        "cn0_ai_trend_dbhz",
    ]
    observation_columns = [
        "op",
        "rx_utc",
        "system",
        "signal_name",
        "svid",
        "satellite_id",
        "cn0_reference_dbhz",
        "source_file",
        "source_cadence",
        "link_id",
    ]
    model = pd.read_csv(MODEL_SOURCE, usecols=model_columns, low_memory=False)
    observations = pd.read_csv(
        OBSERVATION_SOURCE, usecols=observation_columns, low_memory=False
    )
    model["rx_utc"] = pd.to_datetime(model["rx_utc"], utc=True, errors="coerce")
    observations["rx_utc"] = pd.to_datetime(
        observations["rx_utc"], utc=True, errors="coerce"
    )
    model["cn0_ai_trend_dbhz"] = pd.to_numeric(
        model["cn0_ai_trend_dbhz"], errors="coerce"
    )
    observations["cn0_reference_dbhz"] = pd.to_numeric(
        observations["cn0_reference_dbhz"], errors="coerce"
    )
    model["svid"] = pd.to_numeric(model["svid"], errors="coerce").astype("Int64")
    observations["svid"] = pd.to_numeric(
        observations["svid"], errors="coerce"
    ).astype("Int64")
    selected_ops = {op for op, _, _ in base.OP_SPECS}
    model = model[
        model["op"].isin(selected_ops)
        & model["rx_utc"].notna()
        & model["cn0_ai_trend_dbhz"].notna()
        & model["svid"].notna()
    ].copy()
    observations = observations[
        observations["op"].isin(selected_ops)
        & observations["rx_utc"].notna()
        & observations["cn0_reference_dbhz"].notna()
        & observations["svid"].notna()
    ].copy()
    return model, observations


def interpolate_model_at_observation_epochs(
    model: pd.DataFrame, observations: pd.DataFrame
) -> pd.DataFrame:
    keys = ["op", "signal_name", "svid"]
    output_parts: list[pd.DataFrame] = []
    model_groups = {
        key: group.sort_values("rx_utc")
        for key, group in model.groupby(keys, sort=False, dropna=False)
    }
    for key, observed_link in observations.groupby(
        keys, sort=False, dropna=False
    ):
        observed_link = observed_link.sort_values("rx_utc").copy()
        observed_link["cn0_model_at_observation_dbhz"] = np.nan
        model_link = model_groups.get(key)
        if model_link is None or model_link.empty:
            output_parts.append(observed_link)
            continue
        model_link = (
            model_link[["rx_utc", "cn0_ai_trend_dbhz"]]
            .dropna()
            .drop_duplicates("rx_utc", keep="last")
            .sort_values("rx_utc")
        )
        segment_id = model_link["rx_utc"].diff().gt(MODEL_MAX_GAP).cumsum()
        predicted = np.full(len(observed_link), np.nan, dtype=float)
        observed_ns = (
            observed_link["rx_utc"]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        for _, segment in model_link.groupby(segment_id, sort=False):
            if len(segment) < 2:
                continue
            segment_ns = (
                segment["rx_utc"]
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            in_segment = (observed_ns >= segment_ns[0]) & (
                observed_ns <= segment_ns[-1]
            )
            if not np.any(in_segment):
                continue
            x = (segment_ns - segment_ns[0]) / 1e9
            x_observed = (observed_ns[in_segment] - segment_ns[0]) / 1e9
            y = segment["cn0_ai_trend_dbhz"].to_numpy(float)
            if len(segment) >= 3:
                values = PchipInterpolator(x, y, extrapolate=False)(x_observed)
            else:
                values = np.interp(x_observed, x, y)
            predicted[in_segment] = values
        observed_link["cn0_model_at_observation_dbhz"] = predicted
        output_parts.append(observed_link)
    aligned = pd.concat(output_parts, ignore_index=True)
    aligned["residual_observed_minus_model_db"] = (
        aligned["cn0_reference_dbhz"]
        - aligned["cn0_model_at_observation_dbhz"]
    )
    aligned["model_interpolation"] = (
        "PCHIP within contiguous 1-min frozen-model arcs; no gap extrapolation"
    )
    return aligned


def build_residual_trend(matched: pd.DataFrame) -> pd.DataFrame:
    working = matched.copy()
    working["minute_utc"] = working["rx_utc"].dt.floor("min")
    group_columns = [
        "op",
        "mission_phase",
        "system",
        "signal_name",
        "svid",
        "satellite_id",
        "link_id",
    ]
    phase_map = (
        {op: phase for op, phase, _ in base.OP_SPECS}
    )
    working["mission_phase"] = working["op"].map(phase_map)
    minute = (
        working.groupby(group_columns + ["minute_utc"], as_index=False)
        .agg(
            residual_1min_median_db=(
                "residual_observed_minus_model_db",
                "median",
            ),
            residual_1min_mean_db=(
                "residual_observed_minus_model_db",
                "mean",
            ),
            observation_rows=("residual_observed_minus_model_db", "size"),
        )
        .sort_values(group_columns + ["minute_utc"])
    )
    trend_parts: list[pd.DataFrame] = []
    for _, link in minute.groupby(group_columns, sort=False, dropna=False):
        link = link.sort_values("minute_utc").copy()
        segment_id = link["minute_utc"].diff().gt(TREND_MAX_GAP).cumsum()
        for _, segment in link.groupby(segment_id, sort=False):
            segment = segment.copy()
            segment["residual_5min_mean_db"] = (
                segment["residual_1min_mean_db"]
                .rolling(
                    TREND_WINDOW_MINUTES,
                    center=True,
                    min_periods=1,
                )
                .mean()
            )
            segment["trend_definition"] = (
                "centered 5-min rolling mean of per-minute residual means"
            )
            trend_parts.append(segment)
    return pd.concat(trend_parts, ignore_index=True)


def residual_limit(residual: pd.Series) -> float:
    values = pd.to_numeric(residual, errors="coerce").dropna().abs()
    if values.empty:
        return 10.0
    robust = float(values.quantile(0.995))
    return float(np.clip(math.ceil(robust / 2.5) * 2.5, 7.5, 20.0))


def add_satellite_legend(
    legend_ax: plt.Axes,
    panel: pd.DataFrame,
    colors: dict[str, object],
    wide: bool,
) -> None:
    link_table = panel[["satellite_id", "link_id"]].drop_duplicates().sort_values(
        "satellite_id"
    )
    link_ids = link_table["link_id"].tolist()
    labels = {row.link_id: row.satellite_id for row in link_table.itertuples()}
    legend_rows = 3 if wide else 2
    entries: list[str | None] = []
    for constellation in ["E", "G"]:
        group_ids = [
            link_id
            for link_id in link_ids
            if labels[link_id].startswith(constellation)
        ]
        if not group_ids:
            continue
        entries.extend(group_ids)
        entries.extend([None] * ((-len(group_ids)) % legend_rows))
    columns = max(1, math.ceil(len(entries) / legend_rows))
    fontsize = 6.35 if wide else 6.45
    handles = [
        (
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=3.5,
                markerfacecolor=colors[link_id],
                markeredgewidth=0,
            )
            if link_id is not None
            else Line2D([0], [0], linestyle="none", alpha=0.0)
        )
        for link_id in entries
    ]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        labels=[labels[link_id] if link_id is not None else "" for link_id in entries],
        title="Satellite",
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        ncol=columns,
        fontsize=fontsize,
        title_fontsize=fontsize + 0.25,
        frameon=False,
        borderpad=0.0,
        handletextpad=0.28,
        columnspacing=0.68,
        labelspacing=0.28,
    )


def plot_panel(
    ax: plt.Axes,
    header_ax: plt.Axes,
    model: pd.DataFrame,
    residuals: pd.DataFrame,
    trend: pd.DataFrame,
    op: str,
    phase_code: str,
    phase_name: str,
    letter: str,
    colors: dict[str, object],
    wide: bool,
    y_limit: float,
    raw_observation_count: int,
) -> dict[str, object]:
    model_panel = model[model["op"].eq(op)].copy()
    residual_panel = residuals[residuals["op"].eq(op)].copy()
    trend_panel = trend[trend["op"].eq(op)].copy()
    if model_panel.empty or residual_panel.empty:
        raise RuntimeError(f"Missing model or residual rows for {op}")

    base.shade_training_segments(ax, model_panel)
    point_size = 2.5 if wide else 3.4
    for link_id, points in residual_panel.groupby("link_id", sort=False):
        ax.scatter(
            points["rx_utc"],
            points["residual_observed_minus_model_db"],
            s=point_size,
            c=[colors[link_id]],
            alpha=0.23,
            linewidths=0,
            marker="o",
            rasterized=True,
            zorder=2,
        )
    for _, link in trend_panel.groupby(
        ["signal_name", "svid"], sort=False, dropna=False
    ):
        base.plot_gap_aware(
            ax,
            link.rename(columns={"minute_utc": "rx_utc"}),
            "residual_5min_mean_db",
            max_gap_minutes=2.0,
            color="#111111",
            lw=0.95,
            alpha=0.85,
            linestyle="-",
            zorder=4,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    ax.axhline(
        0.0,
        color="#6F777D",
        lw=0.8,
        linestyle=(0, (4, 2)),
        alpha=0.9,
        zorder=1,
    )
    ax.set_ylim(-y_limit, y_limit)
    tick_step = 5.0 if y_limit >= 10 else 2.5
    ax.set_yticks(np.arange(-y_limit, y_limit + 0.1, tick_step))
    ax.set_ylabel(r"Residual, observed $-$ model (dB)")
    ax.grid(True, color=base.COLORS["grid"], linewidth=0.65, zorder=0)
    base.format_utc_axis(ax, residual_panel["rx_utc"], wide)

    add_satellite_legend(header_ax, residual_panel, colors, wide)
    is_external = op in base.EXTERNAL_HOLDOUT_OPS
    header_ax.text(
        0.0,
        0.72,
        f"{op} | {phase_name} ({phase_code})",
        ha="left",
        va="center",
        fontsize=11.0,
        color=base.COLORS["text"],
    )
    header_ax.text(
        0.0,
        0.30,
        "not used for training" if is_external else "used for training",
        ha="left",
        va="center",
        fontsize=8.2,
        color=base.COLORS["soft"],
    )
    header_ax.text(
        -0.020 if wide else -0.055,
        0.72,
        letter,
        ha="left",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color=base.COLORS["text"],
        clip_on=False,
    )

    values = residual_panel["residual_observed_minus_model_db"].dropna().to_numpy(float)
    return {
        "op": op,
        "phase": phase_code,
        "raw_observation_rows": int(raw_observation_count),
        "matched_residual_rows": int(len(values)),
        "mean_residual_db": float(np.mean(values)),
        "median_residual_db": float(np.median(values)),
        "mae_db": float(np.mean(np.abs(values))),
        "rmse_db": float(np.sqrt(np.mean(values**2))),
        "p05_residual_db": float(np.quantile(values, 0.05)),
        "p95_residual_db": float(np.quantile(values, 0.95)),
        "display_y_limit_db": y_limit,
    }


def save_figure(fig: plt.Figure) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension, kwargs in [
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600}),
    ]:
        path = FIGURE_DIR / f"{STEM}.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, **kwargs)
        outputs.append(path)
    return outputs


def main() -> None:
    base.configure()
    model, observations = load_sources()
    aligned = interpolate_model_at_observation_epochs(model, observations)
    matched = aligned.dropna(
        subset=["cn0_model_at_observation_dbhz", "residual_observed_minus_model_db"]
    ).copy()
    if matched.empty:
        raise RuntimeError("No one-second observations could be paired to model arcs")
    trend = build_residual_trend(matched)
    y_limit = residual_limit(matched["residual_observed_minus_model_db"])
    colors = base.all_satellite_palette(
        sorted(set(matched["link_id"].dropna().astype(str)))
    )

    fig = plt.figure(figsize=(18.845, 10.2))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.06, 1.0, 1.0, 1.0])
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.058,
        top=0.955,
        wspace=0.12,
        hspace=0.28,
    )

    panel_rows: list[dict[str, object]] = []
    raw_observation_counts = aligned.groupby("op").size().to_dict()
    for index, (op, phase_code, phase_name) in enumerate(base.OP_SPECS):
        wide = index >= 3
        cell = grid[0, index] if index < 3 else grid[index - 2, :]
        inner = cell.subgridspec(
            2,
            1,
            height_ratios=[0.27 if wide else 0.22, 1.0],
            hspace=0.0,
        )
        header_ax = fig.add_subplot(inner[0])
        panel_rows.append(
            plot_panel(
                fig.add_subplot(inner[1]),
                header_ax,
                model,
                matched,
                trend,
                op,
                phase_code,
                phase_name,
                chr(ord("a") + index),
                colors,
                wide,
                y_limit,
                raw_observation_counts.get(op, 0),
            )
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="#0072B2",
            markeredgewidth=0,
            markersize=4.4,
            linestyle="none",
            label=r"1-s residual, observed $-$ model (satellite colors)",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            lw=1.9,
            linestyle="-",
            label="5-min mean residual",
        ),
        Line2D(
            [0],
            [0],
            color="#6F777D",
            lw=1.0,
            linestyle=(0, (4, 2)),
            label="zero residual",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        fontsize=9.0,
        columnspacing=1.6,
        handlelength=2.2,
    )
    outputs = save_figure(fig)
    plt.close(fig)

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    residual_source = SOURCE_DIR / f"{STEM}_residual_1s_source_data.csv"
    trend_source = SOURCE_DIR / f"{STEM}_residual_1min_trend_source_data.csv"
    summary_source = SOURCE_DIR / f"{STEM}_panel_summary.csv"
    alignment_source = SOURCE_DIR / f"{STEM}_alignment_audit.csv"
    matched.to_csv(residual_source, index=False, encoding="utf-8-sig")
    trend.to_csv(trend_source, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(panel_rows)
    summary.to_csv(summary_source, index=False, encoding="utf-8-sig")
    alignment_rows = []
    for op, observed_op in aligned.groupby("op", sort=False):
        matched_rows = int(observed_op["cn0_model_at_observation_dbhz"].notna().sum())
        alignment_rows.append(
            {
                "op": op,
                "observation_rows": int(len(observed_op)),
                "matched_rows": matched_rows,
                "matched_fraction": matched_rows / len(observed_op),
                "unmatched_rows": int(len(observed_op) - matched_rows),
                "alignment_rule": (
                    "PCHIP only inside contiguous 1-min model arcs; no gap extrapolation"
                ),
            }
        )
    pd.DataFrame(alignment_rows).to_csv(
        alignment_source, index=False, encoding="utf-8-sig"
    )

    print(summary.to_string(index=False))
    for path in outputs:
        print(path)
    print(residual_source)
    print(trend_source)
    print(summary_source)
    print(alignment_source)


if __name__ == "__main__":
    main()
