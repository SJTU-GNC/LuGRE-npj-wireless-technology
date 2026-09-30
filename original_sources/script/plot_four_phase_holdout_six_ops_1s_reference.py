#!/usr/bin/env python3
"""Plot 1-s observed C/N0 points against the frozen 1-min AI trend model."""

from __future__ import annotations

import math
import re
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
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

import train_cn0_single_global_residual_1min_op74_holdout as model_mod


MODEL_DIR = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_trend_residual_tuned_no_leakage"
)
PREDICTIONS = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_dynamic_six_ops_frozen_ai"
    / "cn0_dynamic_six_ops_predictions.csv"
)
TLM_DIR = ROOT / "data" / "receiver_observation" / "TLM"
OUT_DIR = ROOT / "figure" / "paper_draft_v2"
SOURCE_DIR = ROOT / "table" / "figure_source_data"
STEM = "Fig6_all_satellites_smoothed_AI_trend_vs_1s_reference"

EXTERNAL_HOLDOUT_OPS = ("OP2", "OP21", "OP27", "OP74")
EXTERNAL_HOLDOUT_SPLIT = "external_holdout"
OP_SPECS = [
    ("OP2", "C", "Commissioning"),
    ("OP21", "T", "Translunar"),
    ("OP27", "L", "Lunar orbit"),
    ("OP40", "S", "Surface"),
    ("OP74", "S", "Surface"),
    ("OP76", "S", "Surface"),
]

GPS_EPOCH = pd.Timestamp("1980-01-06T00:00:00Z")
SIGNAL_META = {
    0: ("G", "GPS_L1"),
    1: ("G", "GPS_L5"),
    2: ("E", "GAL_E1"),
    3: ("E", "GAL_E5a"),
}
SIGNAL_ORDER = ["GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"]
SIGNAL_SHORT = {
    "GPS_L1": "L1",
    "GPS_L5": "L5",
    "GAL_E1": "E1",
    "GAL_E5a": "E5a",
}
SIGNAL_COLORS = {
    "GPS_L1": ("#0072B2", "#56B4E9"),
    "GPS_L5": ("#009E73", "#00A6A6"),
    "GAL_E1": ("#D55E00", "#E69F00"),
    "GAL_E5a": ("#CC79A7", "#7B2CBF"),
}
MAX_LINKS_PER_SIGNAL = 2
RX_TIME_RE = re.compile(r"rxTime:\s*([0-9.]+)")
MEASURE_RE = re.compile(
    r"svid:\s*(?P<svid>\d+).*?cn0:\s*(?P<cn0>[-+0-9.Ee]+)\s+"
    r"signalId:\s*(?P<signal_id>\d+)"
)

COLORS = {
    "model": "#111111",
    "grid": "#E3E7EB",
    "text": "#202124",
    "soft": "#5F6B76",
    "training_background": "#E8F3FA",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9.8,
            "axes.titlesize": 10.8,
            "axes.labelsize": 10.3,
            "xtick.labelsize": 9.1,
            "ytick.labelsize": 9.1,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#000000",
            "axes.spines.right": (
                STEM == "Fig6_all_satellites_smoothed_AI_trend_vs_1s_reference"
            ),
            "axes.spines.top": (
                STEM == "Fig6_all_satellites_smoothed_AI_trend_vs_1s_reference"
            ),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def load_model_predictions() -> pd.DataFrame:
    data = pd.read_csv(PREDICTIONS, low_memory=False)
    data["rx_utc"] = pd.to_datetime(data["minute_utc"], utc=True, errors="coerce")
    data["cn0_ai_trend_dbhz"] = pd.to_numeric(
        data["cn0_physics_ai_trend_dbhz"], errors="coerce"
    )
    data["satellite_id"] = (
        data["system"].astype(str)
        + pd.to_numeric(data["svid"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )
    return data


def load_one_second_reference(operations: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for op in sorted(operations):
        paths = sorted(TLM_DIR.glob(f"TLM_RAW_*_{op}_*.txt"))
        if not paths:
            raise FileNotFoundError(f"No TLM RAW text file found for {op}")
        for path in paths:
            with path.open("r", encoding="utf-8", errors="ignore") as stream:
                for line in stream:
                    time_match = RX_TIME_RE.search(line)
                    if not time_match:
                        continue
                    gps_seconds = float(time_match.group(1))
                    utc = GPS_EPOCH + pd.to_timedelta(gps_seconds - 18.0, unit="s")
                    for measure in MEASURE_RE.finditer(line):
                        signal_id = int(measure.group("signal_id"))
                        if signal_id not in SIGNAL_META:
                            continue
                        system, signal_name = SIGNAL_META[signal_id]
                        svid = int(measure.group("svid"))
                        rows.append(
                            {
                                "op": op,
                                "rx_utc": utc,
                                "system": system,
                                "signal_name": signal_name,
                                "svid": svid,
                                "satellite_id": f"{system}{svid:02d}",
                                "cn0_reference_dbhz": float(measure.group("cn0")),
                                "source_file": path.name,
                                "source_cadence": "LuGRE TLM RAW approximately 1 s",
                            }
                        )
    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("No one-second LuGRE C/N0 observations were parsed")
    return data.sort_values(["op", "signal_name", "svid", "rx_utc"]).reset_index(
        drop=True
    )


def select_representative_links(
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = reference.copy()
    working["minute_bin"] = working["rx_utc"].dt.floor("min")
    audit_rows: list[dict[str, object]] = []
    for (op, signal_name), group in working.groupby(["op", "signal_name"]):
        summaries = (
            group.groupby(["system", "svid", "satellite_id"])
            .agg(
                reference_1s_rows=("cn0_reference_dbhz", "size"),
                start_utc=("rx_utc", "min"),
                end_utc=("rx_utc", "max"),
                unique_minute_bins=("minute_bin", "nunique"),
            )
            .reset_index()
        )
        minute_sets = {
            int(svid): set(link["minute_bin"].dropna())
            for svid, link in group.groupby("svid")
        }
        row_by_svid = {
            int(row.svid): row for row in summaries.itertuples(index=False)
        }
        covered: set[pd.Timestamp] = set()
        remaining = set(row_by_svid)
        for rank in range(1, min(MAX_LINKS_PER_SIGNAL, len(remaining)) + 1):
            best_svid = max(
                remaining,
                key=lambda svid: (
                    len(minute_sets[svid] - covered),
                    len(minute_sets[svid]),
                    int(row_by_svid[svid].reference_1s_rows),
                    -svid,
                ),
            )
            row = row_by_svid[best_svid]
            incremental_bins = len(minute_sets[best_svid] - covered)
            audit_rows.append(
                {
                    "op": op,
                    "signal_name": signal_name,
                    "system": row.system,
                    "svid": best_svid,
                    "satellite_id": row.satellite_id,
                    "selection_rank": rank,
                    "reference_1s_rows": int(row.reference_1s_rows),
                    "start_utc": row.start_utc,
                    "end_utc": row.end_utc,
                    "duration_hours": (row.end_utc - row.start_utc).total_seconds()
                    / 3600.0,
                    "unique_minute_bins": int(row.unique_minute_bins),
                    "incremental_minute_bins": incremental_bins,
                }
            )
            covered |= minute_sets[best_svid]
            remaining.remove(best_svid)
    audit = pd.DataFrame(audit_rows)
    audit["selection_rule"] = (
        "rank 1 maximizes covered minutes; rank 2 maximizes additional uncovered minutes"
    )
    audit["link_id"] = audit["signal_name"] + ":" + audit["satellite_id"]
    keys = audit[["op", "signal_name", "svid", "link_id"]]
    selected = reference.merge(keys, on=["op", "signal_name", "svid"], how="inner")
    return selected, audit


def all_satellite_palette(satellite_ids: list[str]) -> dict[str, object]:
    palette: dict[str, object] = {}
    golden_ratio = 0.61803398875
    for index, satellite_id in enumerate(sorted(satellite_ids)):
        hue = (index * golden_ratio) % 1.0
        saturation = (0.78, 0.86, 0.94)[index % 3]
        value = (0.76, 0.88)[(index // 3) % 2]
        palette[satellite_id] = mpl.colors.hsv_to_rgb((hue, saturation, value))
    return palette


def all_satellite_markers(satellite_ids: list[str]) -> dict[str, object]:
    marker_pool: list[object] = [
        "o",
        "s",
        "^",
        "v",
        "D",
        "P",
        "X",
        "<",
        ">",
        "h",
        "H",
        "p",
        "*",
        "d",
        "8",
    ]
    for sides in range(3, 9):
        for style in (0, 1):
            for angle in (0, 22.5):
                marker_pool.append((sides, style, angle))
    return {
        satellite_id: marker_pool[index % len(marker_pool)]
        for index, satellite_id in enumerate(sorted(satellite_ids))
    }


def add_display_smoothing(model: pd.DataFrame) -> pd.DataFrame:
    smoothed = model.copy()
    smoothed["cn0_ai_trend_display_dbhz"] = pd.to_numeric(
        smoothed["cn0_ai_trend_dbhz"], errors="coerce"
    )
    smoothed["ai_trend_residual_display_db"] = pd.to_numeric(
        smoothed["ai_trend_residual_pred_db"], errors="coerce"
    )
    return smoothed


def plot_gap_aware(
    ax: plt.Axes,
    frame: pd.DataFrame,
    y: str,
    max_gap_minutes: float = 1.5,
    **kwargs: object,
) -> None:
    points = frame[["rx_utc", y]].dropna().sort_values("rx_utc")
    if points.empty:
        return
    segments = points["rx_utc"].diff().gt(
        pd.Timedelta(minutes=max_gap_minutes)
    ).cumsum()
    for _, segment in points.groupby(segments):
        if len(segment) >= 3:
            elapsed = (
                segment["rx_utc"] - segment["rx_utc"].iloc[0]
            ).dt.total_seconds().to_numpy(float)
            values = segment[y].to_numpy(float)
            unique_elapsed, unique_index = np.unique(elapsed, return_index=True)
            unique_values = values[unique_index]
            if len(unique_elapsed) >= 3 and unique_elapsed[-1] > 0:
                dense_elapsed = np.arange(0.0, unique_elapsed[-1] + 0.1, 15.0)
                dense_values = PchipInterpolator(unique_elapsed, unique_values)(
                    dense_elapsed
                )
                dense_time = segment["rx_utc"].iloc[0] + pd.to_timedelta(
                    dense_elapsed, unit="s"
                )
                ax.plot(dense_time, dense_values, **kwargs)
                continue
        ax.plot(segment["rx_utc"], segment[y], **kwargs)


def format_utc_axis(ax: plt.Axes, times: pd.Series, wide: bool) -> None:
    parsed = pd.to_datetime(times, utc=True, errors="coerce").dropna()
    if parsed.empty:
        ax.set_xlabel("UTC")
        return
    start, end = parsed.min(), parsed.max()
    ticks = pd.date_range(start=start, end=end, periods=5 if wide else 4)
    padding = max((end - start) * 0.015, pd.Timedelta(seconds=30))
    ax.set_xlim(start - padding, end + padding)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    ax.set_xlabel("UTC", labelpad=2.0)


def rmse(frame: pd.DataFrame, prediction: str) -> float:
    paired = frame[[prediction, model_mod.TARGET]].dropna()
    if paired.empty:
        return float("nan")
    error = paired[prediction].to_numpy(float) - paired[model_mod.TARGET].to_numpy(float)
    return float(math.sqrt(np.mean(error**2)))


def set_panel_ylim(
    ax: plt.Axes, op: str
) -> int:
    upper = 45 if op in {"OP2", "OP21"} else 40
    ax.set_ylim(20, upper)
    ax.set_yticks(np.arange(20, upper + 0.1, 5))
    return upper


def shade_training_segments(ax: plt.Axes, model: pd.DataFrame) -> None:
    """Shade only time intervals used by the final train+validation refit."""
    used = (
        model.loc[
            model["evaluation_split"].isin(["train", "validation"]),
            ["rx_utc"],
        ]
        .drop_duplicates()
        .sort_values("rx_utc")
    )
    if used.empty:
        return
    segment_id = used["rx_utc"].diff().gt(pd.Timedelta(minutes=1.5)).cumsum()
    half_bin = pd.Timedelta(seconds=30)
    for _, segment in used.groupby(segment_id):
        ax.axvspan(
            segment["rx_utc"].min() - half_bin,
            segment["rx_utc"].max() + half_bin,
            color=COLORS["training_background"],
            alpha=0.45,
            linewidth=0,
            zorder=0,
        )


def plot_panel(
    ax: plt.Axes,
    header_ax: plt.Axes,
    legend_ax: plt.Axes,
    model_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    op: str,
    phase_code: str,
    phase_name: str,
    letter: str,
    colors: dict[str, object],
    markers: dict[str, object],
    wide: bool,
) -> dict[str, object]:
    model = model_data[model_data["op"].eq(op)].copy()
    reference = reference_data[reference_data["op"].eq(op)].copy()
    if model.empty or reference.empty:
        raise RuntimeError(f"Missing model or one-second reference data for {op}")

    start, end = model["rx_utc"].min(), model["rx_utc"].max()
    reference = reference[reference["rx_utc"].between(start, end)].copy()
    is_external = op in EXTERNAL_HOLDOUT_OPS
    evaluation_split = EXTERNAL_HOLDOUT_SPLIT if is_external else "test"
    evaluation = model[model["evaluation_split"].eq(evaluation_split)]

    shade_training_segments(ax, model)

    for (_, _), link in model.groupby(["signal_name", "svid"], dropna=False):
        plot_gap_aware(
            ax,
            link,
            "cn0_ai_trend_display_dbhz",
            max_gap_minutes=1.5,
            color="#111111",
            lw=0.95,
            alpha=0.82,
            linestyle="-",
            zorder=4,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    point_size = 2.5 if wide else 3.4
    point_alpha = 0.25
    for link_id, points in reference.groupby("link_id"):
        ax.scatter(
            points["rx_utc"],
            points["cn0_reference_dbhz"],
            s=point_size,
            c=[colors[link_id]],
            alpha=point_alpha,
            linewidths=0,
            marker="o",
            rasterized=True,
            zorder=2,
        )

    link_table = reference[["satellite_id", "link_id"]].drop_duplicates().sort_values(
        "satellite_id"
    )
    link_ids = link_table["link_id"].tolist()
    link_labels = {row.link_id: row.satellite_id for row in link_table.itertuples()}
    legend_rows = 3 if wide else 2
    legend_entries: list[str | None] = []
    for constellation in ["E", "G"]:
        group_ids = [
            link_id
            for link_id in link_ids
            if link_labels[link_id].startswith(constellation)
        ]
        if not group_ids:
            continue
        legend_entries.extend(group_ids)
        legend_entries.extend([None] * ((-len(group_ids)) % legend_rows))
    legend_columns = max(1, math.ceil(len(legend_entries) / legend_rows))
    legend_fontsize = 6.35 if wide else 6.45
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
        for link_id in legend_entries
    ]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        labels=[link_labels[link_id] if link_id is not None else "" for link_id in legend_entries],
        title="Satellite",
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        ncol=legend_columns,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize + 0.25,
        frameon=False,
        borderpad=0.0,
        handletextpad=0.28,
        columnspacing=0.68,
        labelspacing=0.28,
    )

    ai_rmse = rmse(evaluation, "cn0_ai_trend_dbhz")
    header_ax.axis("off")
    header_ax.text(
        0.0,
        0.72,
        f"{op} | {phase_name} ({phase_code})",
        ha="left",
        va="center",
        fontsize=11.0,
        color=COLORS["text"],
    )
    header_ax.text(
        0.0,
        0.30,
        "not used for training" if is_external else "used for training",
        ha="left",
        va="center",
        fontsize=8.2,
        color=COLORS["soft"],
    )
    header_ax.text(
        -0.020 if wide else -0.055,
        0.72,
        letter,
        ha="left",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )

    ax.set_ylabel(r"$C/N_0$ (dB-Hz)")
    y_upper = set_panel_ylim(ax, op)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65, zorder=0)
    format_utc_axis(ax, reference["rx_utc"], wide)
    return {
        "op": op,
        "reference_1s_rows": len(reference),
        "evaluation_1min_rows": len(evaluation),
        "ai_rmse_dbhz": ai_rmse,
        "y_min_dbhz": 20,
        "y_max_dbhz": y_upper,
        "reference_max_dbhz": float(reference["cn0_reference_dbhz"].max()),
        "model_max_dbhz": float(model["cn0_ai_trend_dbhz"].max()),
    }


def save_figure(fig: plt.Figure) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension, kwargs in [
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600}),
    ]:
        path = OUT_DIR / f"{STEM}.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, **kwargs)
        outputs.append(path)
    return outputs


def main() -> None:
    configure()
    selected_ops = {spec[0] for spec in OP_SPECS}
    model = load_model_predictions()
    if "observation_template_available" in model.columns:
        model = model[model["observation_template_available"].fillna(False)].copy()
    model = model[model["op"].isin(selected_ops)].dropna(
        subset=["cn0_ai_trend_dbhz", "rx_utc"]
    )
    reference = load_one_second_reference(selected_ops)
    reference["link_id"] = reference["satellite_id"]
    model["link_id"] = model["satellite_id"]
    model = add_display_smoothing(model)
    colors = all_satellite_palette(
        sorted(
            set(reference["satellite_id"].astype(str))
            | set(model["satellite_id"].astype(str))
        )
    )
    markers = all_satellite_markers(sorted(colors))

    # Compensate for this layout's tighter exported bounding box so the final
    # visible width matches the 18.5-in full-mission four-band figure.
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

    panel_rows = []
    for index, (op, phase_code, phase_name) in enumerate(OP_SPECS):
        wide = index >= 3
        cell = grid[0, index] if index < 3 else grid[index - 2, :]
        inner = cell.subgridspec(
            2,
            1,
            height_ratios=[0.27 if wide else 0.22, 1.0],
            hspace=0.0,
        )
        meta_ax = fig.add_subplot(inner[0])
        panel_rows.append(
            plot_panel(
                fig.add_subplot(inner[1]),
                meta_ax,
                meta_ax,
                model,
                reference,
                op,
                phase_code,
                phase_name,
                chr(ord("a") + index),
                colors,
                markers,
                wide,
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
            label=r"observed 1-s $C/N_0$ (satellite colors)",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            lw=1.9,
            linestyle="-",
            label="physics + AI residual model output",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=2,
        fontsize=9.0,
        columnspacing=1.6,
        handlelength=2.2,
    )
    outputs = save_figure(fig)
    plt.close(fig)

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    model_columns = [
        "rx_utc",
        "observation_time_median_utc",
        "source_bin_gps_seconds",
        "evaluation_split",
        "time_block_id",
        "op",
        "mission_phase",
        "system",
        "signal_name",
        "svid",
        "satellite_id",
        "link_id",
        "cn0_ai_trend_dbhz",
        "ai_trend_residual_display_db",
        "cn0_ai_trend_display_dbhz",
        "valid_seconds_in_bin",
        "coverage_fraction",
        "longest_gap_s",
        "reacquisition_flag",
        "trend_training_eligible",
        "receiver_trajectory_source",
        "orbit_geometry_status",
        "gnss_orbit_source",
        "dynamic_stage",
        "dynamic_quality_status",
        "dynamic_status_priority",
        "dynamic_segment_id",
        "reference_position_error_km",
        "geometry_role",
        "reference_role",
    ]
    model_source = SOURCE_DIR / f"{STEM}_model_1min_source_data.csv"
    reference_source = SOURCE_DIR / f"{STEM}_observed_1s_source_data.csv"
    panel_source = SOURCE_DIR / f"{STEM}_panel_summary.csv"
    available_model_columns = [column for column in model_columns if column in model.columns]
    model[available_model_columns].to_csv(
        model_source, index=False, encoding="utf-8-sig"
    )
    reference.to_csv(reference_source, index=False, encoding="utf-8-sig")
    pd.DataFrame(panel_rows).to_csv(panel_source, index=False, encoding="utf-8-sig")

    print(pd.DataFrame(panel_rows).to_string(index=False))
    for path in outputs:
        print(path)
    print(model_source)
    print(reference_source)
    print(panel_source)


if __name__ == "__main__":
    main()
