#!/usr/bin/env python3
"""Left-margin-trimmed revision of the frozen wide compact timeline.

The script reads the exact source tables already exported by the accepted
figure. It does not run inference, choose satellites, select windows, smooth,
clip, or otherwise recompute scientific content. Relative to the wide compact
baseline, it changes only the GridSpec left boundary, assigning the released
width to all main and detail axes while preserving the canvas, right boundary,
vertical geometry, typography, legend, and scientific content.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str((ROOT / "script").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from PIL import Image

import build_dynamic_multiband_one_common_1min as one_minute_builder
import plot_full_mission_four_band_ai_cn0_sidelobe_1min_three_sat_24h as layout_base


BASE_STEM = "Fig_full_mission_four_band_AI_CN0_1min_one_sat_24h"
ORIGINAL_STEM = f"{BASE_STEM}_reselected_24h"
INTERLEAVED_STEM = f"{ORIGINAL_STEM}_interleaved_details"
READABLE_STEM = f"{INTERLEAVED_STEM}_readable_timeline"
REFERENCE_STEM = (
    f"{READABLE_STEM}_legend_up_large_detail_numbers_wide_compact"
)
STEM = f"{ORIGINAL_STEM}_interleaved_details_wide_compact_left_trimmed"
FIGURE_DIR = ROOT / "figure" / "paper_draft_v2"
OUTPUT_BASE = FIGURE_DIR / STEM
PROVENANCE_PATH = FIGURE_DIR / f"{STEM}_provenance.json"
SOURCE_DIR = ROOT / "table" / "figure_source_data"

ORIGINAL_FIGURE = FIGURE_DIR / f"{ORIGINAL_STEM}.png"
REFERENCE_FIGURE = FIGURE_DIR / f"{REFERENCE_STEM}.png"
REFERENCE_PROVENANCE = FIGURE_DIR / f"{REFERENCE_STEM}_provenance.json"
REFERENCE_SCRIPT = (
    ROOT
    / "script"
    / (
        "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_"
        "interleaved_details_readable_timeline_"
        "legend_up_large_detail_numbers_wide_compact.py"
    )
)
ORIGINAL_ENTRY = (
    ROOT / "script" / "plot_full_mission_four_band_ai_cn0_sidelobe_1min_one_sat_24h.py"
)
ORIGINAL_LAYOUT_MODULE = (
    ROOT / "script" / "plot_full_mission_four_band_ai_cn0_sidelobe_1min_three_sat_24h.py"
)
SOURCE_DATA = SOURCE_DIR / f"{ORIGINAL_STEM}_source_data.csv"
SELECTION_DATA = SOURCE_DIR / f"{ORIGINAL_STEM}_satellite_selection.csv"
DETAIL_STATUS_DATA = SOURCE_DIR / f"{ORIGINAL_STEM}_detail_anomaly_timeline.csv"
OBSERVATION_DATA = SOURCE_DIR / f"{ORIGINAL_STEM}_observed_1s_source_data.csv"
WINDOW_SCORE_DATA = SOURCE_DIR / f"{ORIGINAL_STEM}_detail_window_scores.csv"
WINDOW_AUDIT_DATA = SOURCE_DIR / f"{ORIGINAL_STEM}_window_selection_audit.csv"

DETAIL_DAYS = layout_base.load_detail_days_from_audit(WINDOW_AUDIT_DATA)
DETAIL_WINDOWS = [
    (mode, start, start + pd.Timedelta(days=1))
    for mode, start in DETAIL_DAYS.items()
]

REFERENCE_WIDTH_PX = 12_403
REFERENCE_HEIGHT_PX = 5_880
ORIGINAL_WIDTH_PX = 10_785
ORIGINAL_HEIGHT_PX = 4_052
WIDTH_SCALE = 1.0
TARGET_WIDTH_PX = REFERENCE_WIDTH_PX
RASTER_DPI = 600
FIGURE_WIDTH_IN = TARGET_WIDTH_PX / RASTER_DPI
FIGURE_HEIGHT_IN = REFERENCE_HEIGHT_PX / RASTER_DPI
TARGET_HEIGHT_PX = round(FIGURE_HEIGHT_IN * RASTER_DPI)
FONT_SCALE = 1.18
DETAIL_MARKER_FONT_PT = 14.0
REFERENCE_LEGEND_UP_SHIFT_PX = 22.68
LEGEND_UP_SHIFT_FIGURE_FRACTION = (
    REFERENCE_LEGEND_UP_SHIFT_PX / TARGET_HEIGHT_PX
)

GRID_HEIGHT_RATIOS = [
    1.05,
    0.335,
    0.82,
    0.457,
    1.05,
    0.457,
    1.05,
    0.335,
    0.82,
    0.457,
    1.05,
]
DETAIL_WSPACE = 0.056
REFERENCE_GRID_LEFT = 0.047
GRID_LEFT = 0.036
GRID_RIGHT = 0.994
MINIMUM_LEFT_SAFETY_PX = 48

MAIN_Y_LIMITS = (10.0, 40.0)
ASSUMED_LOSS_OF_LOCK_DBHZ = 10.0
PHASE_COLORS = {
    "C": "#1875B7",
    "T": "#14A77B",
    "L": "#F0A202",
    "S": "#D474A7",
}
ZOOM_COLOR = "#DDEAF3"


# Reproduce the accepted one-satellite figure's plotting configuration.
layout_base.common_builder = one_minute_builder
layout_base.base.CONTINUOUS_PATH = one_minute_builder.OUTPUT_PATH
layout_base.N_COMMON_SATELLITES = 1
layout_base.LINE_COLORS = ["#2F6DAE"]
layout_base.base.SIDELOBE_COLOR = "#D3D6DA"
layout_base.DETAIL_PANEL_MODES = ("model_count", "coverage")
layout_base.OMIT_BELOW_LOCK_THRESHOLD = False
layout_base.COLOR_BY_PHASE = True
layout_base.MAIN_Y_LIMITS = MAIN_Y_LIMITS
layout_base.PHASE_COLORS = PHASE_COLORS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def snapshot(paths: list[Path]) -> dict[str, tuple[int, str]]:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required frozen source is missing: {missing}")
    return {
        str(path.resolve()): (int(path.stat().st_size), sha256_file(path))
        for path in paths
    }


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_frozen_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_columns = [
        "signal_name",
        "svid",
        "utc",
        "mission_phase",
        "cn0_physics_ai_trend_dbhz",
        "display_eligible",
        "pattern_first_null_resolved",
        "pattern_side_lobe",
    ]
    data = pd.read_csv(SOURCE_DATA, usecols=source_columns, low_memory=False)
    data["utc"] = pd.to_datetime(data["utc"], utc=True, errors="coerce")
    data["svid"] = pd.to_numeric(data["svid"], errors="coerce").astype("Int64")
    data["cn0_physics_ai_trend_dbhz"] = pd.to_numeric(
        data["cn0_physics_ai_trend_dbhz"], errors="coerce"
    )
    for column in [
        "display_eligible",
        "pattern_first_null_resolved",
        "pattern_side_lobe",
    ]:
        data[column] = parse_bool(data[column])
    if data["utc"].isna().any() or data["svid"].isna().any():
        raise RuntimeError("Frozen figure source contains invalid UTC or SVID")
    data["svid"] = data["svid"].astype(int)
    data = data.sort_values(["signal_name", "svid", "utc"], kind="stable")

    selected = pd.read_csv(SELECTION_DATA, low_memory=False)
    selected["svid"] = pd.to_numeric(selected["svid"], errors="raise").astype(int)
    selected = selected.sort_values(
        ["signal_name", "rank_within_constellation"], kind="stable"
    )

    status_columns = [
        "utc",
        "signal_name",
        "svid",
        "detail_background_state",
    ]
    detail_status = pd.read_csv(
        DETAIL_STATUS_DATA, usecols=status_columns, low_memory=False
    )
    detail_status["utc"] = pd.to_datetime(
        detail_status["utc"], utc=True, errors="coerce"
    )
    detail_status["svid"] = pd.to_numeric(
        detail_status["svid"], errors="raise"
    ).astype(int)

    observation_columns = [
        "op",
        "rx_utc",
        "system",
        "signal_name",
        "svid",
        "cn0_observed_dbhz",
        "source_file",
        "source_cadence",
    ]
    observations = pd.read_csv(
        OBSERVATION_DATA, usecols=observation_columns, low_memory=False
    )
    observations["rx_utc"] = pd.to_datetime(
        observations["rx_utc"], utc=True, errors="coerce"
    )
    observations["svid"] = pd.to_numeric(
        observations["svid"], errors="raise"
    ).astype(int)
    observations["cn0_observed_dbhz"] = pd.to_numeric(
        observations["cn0_observed_dbhz"], errors="coerce"
    )

    expected_pairs = {
        ("GPS_L1", 11),
        ("GPS_L5", 11),
        ("GAL_E1", 34),
        ("GAL_E5a", 34),
    }
    selected_pairs = set(zip(selected["signal_name"], selected["svid"]))
    if selected_pairs != expected_pairs:
        raise RuntimeError(f"Frozen satellite selection changed: {selected_pairs}")
    if set(data["signal_name"].dropna().unique()) != set(layout_base.base.SIGNALS):
        raise RuntimeError("Frozen source does not contain the four expected signals")
    return data, selected, detail_status, observations


def configure_style() -> None:
    layout_base.base.configure_style()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2 * FONT_SCALE,
            "axes.labelsize": 8.8 * FONT_SCALE,
            "axes.titlesize": 9.0 * FONT_SCALE,
            "xtick.labelsize": 7.4 * FONT_SCALE,
            "ytick.labelsize": 7.4 * FONT_SCALE,
            "legend.fontsize": 7.1 * FONT_SCALE,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def draw_main_window_marker(
    ax: plt.Axes,
    detail_start: pd.Timestamp,
    detail_stop: pd.Timestamp,
    marker: str,
) -> dict[str, object]:
    ax.axvspan(
        detail_start,
        detail_stop,
        color=ZOOM_COLOR,
        alpha=0.16,
        lw=0,
        zorder=4,
    )
    start_num = mdates.date2num(detail_start.to_pydatetime())
    stop_num = mdates.date2num(detail_stop.to_pydatetime())
    ax.add_patch(
        Rectangle(
            (start_num, 0.0),
            stop_num - start_num,
            1.0,
            transform=ax.get_xaxis_transform(),
            fill=False,
            edgecolor="#111111",
            linewidth=1.1,
            linestyle=(0, (4, 2)),
            alpha=0.98,
            zorder=8,
            clip_on=False,
        )
    )
    midpoint = detail_start + (detail_stop - detail_start) / 2
    text = ax.text(
        midpoint,
        0.965,
        marker,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11.0 * FONT_SCALE,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        color="#111111",
        clip_on=False,
        zorder=9,
    )
    text.set_gid(f"main-window-marker-{marker}")
    return {
        "axis": ax,
        "artist": text,
        "marker": marker,
        "start": detail_start,
        "stop": detail_stop,
        "midpoint": midpoint,
    }


def detail_limits(
    detail: pd.DataFrame, observed_detail: pd.DataFrame
) -> tuple[float, float]:
    values = pd.to_numeric(
        detail["cn0_physics_ai_trend_dbhz"], errors="coerce"
    )
    actual = values.loc[detail["display_eligible"]].dropna()
    observed_values = pd.to_numeric(
        observed_detail["cn0_observed_dbhz"], errors="coerce"
    ).dropna()
    if not observed_values.empty:
        actual = pd.concat([actual, observed_values], ignore_index=True)
    if actual.empty:
        return layout_base.base.DISPLAY_MIN_DBHZ, layout_base.base.DISPLAY_MAX_DBHZ
    detail_min = float(actual.min())
    detail_max = float(actual.max())
    span = max(detail_max - detail_min, 2.0)
    padding = max(0.6, 0.06 * span)
    return detail_min - padding, detail_max + padding


def build_figure(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    detail_status: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[
    plt.Figure,
    list[plt.Axes],
    list[plt.Axes],
    dict[str, object],
]:
    configure_style()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        dpi=RASTER_DPI,
    )
    grid = fig.add_gridspec(
        11,
        4,
        height_ratios=GRID_HEIGHT_RATIOS,
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=0.055,
        top=0.934,
        hspace=0.0,
        wspace=DETAIL_WSPACE,
    )
    main_rows = [0, 4, 6, 10]
    main_axes: list[plt.Axes] = []
    for index, row in enumerate(main_rows):
        share = main_axes[0] if main_axes else None
        main_axes.append(fig.add_subplot(grid[row, :], sharex=share))

    detail_axes: list[plt.Axes] = []
    detail_axis_map: dict[tuple[int, int], plt.Axes] = {}
    for panel_index in range(4):
        detail_row = 2 if panel_index < 2 else 8
        column_start = 0 if panel_index % 2 == 0 else 2
        for window_index in range(2):
            detail_ax = fig.add_subplot(grid[detail_row, column_start + window_index])
            detail_axis_map[(panel_index, window_index)] = detail_ax
            detail_axes.append(detail_ax)

    phases = layout_base.base.phase_intervals(data)
    mission_start = data["utc"].min()
    mission_stop = data["utc"].max()
    detail_markers = layout_base.DETAIL_PANEL_MARKERS
    window_records: list[dict[str, object]] = []
    detail_marker_records: list[dict[str, object]] = []
    title_artists: list[plt.Text] = []

    for panel_index, signal in enumerate(layout_base.base.SIGNALS):
        main_ax = main_axes[panel_index]
        panel = data[data["signal_name"].eq(signal)].copy()
        chosen = selected[selected["signal_name"].eq(signal)].sort_values(
            "rank_within_constellation"
        )

        layout_base.draw_background(main_ax, panel)
        for window_index, (_, detail_start, detail_stop) in enumerate(DETAIL_WINDOWS):
            marker_index = panel_index * 2 + window_index
            window_records.append(
                draw_main_window_marker(
                    main_ax,
                    detail_start,
                    detail_stop,
                    detail_markers[marker_index],
                )
            )
        layout_base.draw_tracks(main_ax, panel, chosen, with_labels=False)

        for window_index, (_, detail_start, detail_stop) in enumerate(DETAIL_WINDOWS):
            detail_ax = detail_axis_map[(panel_index, window_index)]
            detail = panel[
                panel["utc"].between(detail_start, detail_stop, inclusive="left")
            ].copy()
            status_detail = detail_status[
                detail_status["signal_name"].eq(signal)
                & detail_status["utc"].between(
                    detail_start, detail_stop, inclusive="left"
                )
            ].copy()
            observed_detail = observations[
                observations["signal_name"].eq(signal)
                & observations["rx_utc"].between(
                    detail_start, detail_stop, inclusive="left"
                )
            ].copy()

            layout_base.draw_detail_background(detail_ax, status_detail)
            layout_base.draw_observed_points(detail_ax, observed_detail, chosen)
            layout_base.draw_tracks(
                detail_ax,
                detail,
                chosen,
                with_labels=False,
                detail_style=True,
            )
            detail_min, detail_max = detail_limits(detail, observed_detail)
            detail_ax.set_ylim(detail_min, detail_max)
            if detail_min <= ASSUMED_LOSS_OF_LOCK_DBHZ <= detail_max:
                detail_ax.axhline(
                    ASSUMED_LOSS_OF_LOCK_DBHZ,
                    color="#6F777D",
                    lw=0.55,
                    ls=(0, (3, 2)),
                    alpha=0.8,
                    zorder=2,
                )
            detail_ax.grid(True, color="#D8DDE2", lw=0.45, alpha=0.82, zorder=0)
            detail_ax.set_xlim(detail_start, detail_stop)
            detail_ax.tick_params(labelleft=True, labelbottom=True, pad=1.7)
            detail_ax.xaxis.set_major_locator(
                mdates.HourLocator(byhour=[0, 6, 12, 18])
            )
            detail_ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%d\n%H:%M", tz=detail_start.tz)
            )
            detail_ax.set_xlabel("")
            detail_marker_artist = detail_ax.text(
                0.022,
                0.95,
                detail_markers[panel_index * 2 + window_index],
                transform=detail_ax.transAxes,
                ha="left",
                va="top",
                fontsize=DETAIL_MARKER_FONT_PT,
                fontweight="bold",
                fontfamily="DejaVu Sans",
                color="#111111",
                zorder=8,
            )
            detail_marker_records.append(
                {
                    "axis": detail_ax,
                    "artist": detail_marker_artist,
                    "marker": detail_markers[panel_index * 2 + window_index],
                }
            )

        for _, start, _ in phases[1:]:
            main_ax.axvline(
                start,
                color="#8B9298",
                lw=0.42,
                ls=(0, (2, 2)),
                alpha=0.62,
                zorder=1,
            )
        main_ax.set_ylim(*MAIN_Y_LIMITS)
        main_ax.set_yticks([10, 20, 30, 40])
        main_ax.axhline(
            ASSUMED_LOSS_OF_LOCK_DBHZ,
            color="#6F777D",
            lw=0.55,
            ls=(0, (3, 2)),
            alpha=0.8,
            zorder=2,
        )
        main_ax.grid(True, color="#D8DDE2", lw=0.45, alpha=0.82, zorder=0)
        main_ax.set_ylabel(r"$C/N_0$ (dB-Hz)", labelpad=3.0)
        title = layout_base.base.DISPLAY_NAMES[signal]
        if len(chosen) == 1:
            title += (
                f" | {layout_base.satellite_label(signal, int(chosen.iloc[0]['svid']))}"
            )
        title_artist = main_ax.set_title(
            title,
            loc="left",
            fontsize=9.4 * FONT_SCALE,
            fontweight="bold",
            pad=3.0,
        )
        title_artist.set_gid(f"main-title-{chr(ord('a') + panel_index)}")
        title_artists.append(title_artist)
        main_ax.text(
            -0.022,
            1.006,
            chr(ord("a") + panel_index),
            transform=main_ax.transAxes,
            fontsize=10.0 * FONT_SCALE,
            fontweight="bold",
            va="bottom",
        )
        main_ax.set_xlim(mission_start, mission_stop)

    for panel_index, main_ax in enumerate(main_axes):
        locator = mdates.AutoDateLocator(minticks=8, maxticks=13)
        main_ax.xaxis.set_major_locator(locator)
        main_ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M", tz=mission_start.tz)
        )
        main_ax.tick_params(labelbottom=True, pad=2.0)
        if panel_index == len(main_axes) - 1:
            main_ax.set_xlabel(
                "UTC",
                labelpad=1.4,
                fontsize=8.8 * FONT_SCALE,
            )
        else:
            main_ax.set_xlabel("")

    phase_handles = [
        plt.Line2D(
            [],
            [],
            color=PHASE_COLORS[phase],
            lw=1.5,
            label=f"{phase} phase",
        )
        for phase in ["C", "T", "L", "S"]
    ]
    detail_handles = [layout_base.observed_legend_handle()]
    detail_handles.extend(
        Patch(
            facecolor=layout_base.DETAIL_STATE_STYLES[state_name][0],
            edgecolor="none",
            alpha=layout_base.DETAIL_STATE_STYLES[state_name][1],
            label=layout_base.DETAIL_STATE_STYLES[state_name][2],
        )
        for state_name in layout_base.VISIBLE_DETAIL_LEGEND_STATES
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    first_title_bbox = title_artists[0].get_window_extent(renderer)
    first_title_bottom = first_title_bbox.y0 / fig.bbox.height
    target_legend_bottom = (
        first_title_bottom + LEGEND_UP_SHIFT_FIGURE_FRACTION
    )
    merged_legend = fig.legend(
        handles=[*phase_handles, *detail_handles],
        loc="lower right",
        bbox_to_anchor=(0.994, target_legend_bottom),
        bbox_transform=fig.transFigure,
        ncol=8,
        frameon=True,
        fancybox=False,
        edgecolor="#697077",
        facecolor="white",
        framealpha=1.0,
        handlelength=1.5,
        columnspacing=0.82,
        handletextpad=0.42,
        borderpad=0.35,
        fontsize=7.1 * FONT_SCALE,
    )
    merged_legend.set_gid("merged-phase-observation-legend")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = merged_legend.get_window_extent(renderer)
    target_legend_bottom_px = (
        first_title_bbox.y0
        + LEGEND_UP_SHIFT_FIGURE_FRACTION * fig.bbox.height
    )
    alignment_shift = (
        target_legend_bottom_px - legend_bbox.y0
    ) / fig.bbox.height
    if abs(alignment_shift) > 1e-12:
        merged_legend.set_bbox_to_anchor(
            (0.994, target_legend_bottom + alignment_shift),
            transform=fig.transFigure,
        )
        fig.canvas.draw()

    layout_meta = {
        "window_records": window_records,
        "detail_marker_records": detail_marker_records,
        "title_artists": title_artists,
        "merged_legend": merged_legend,
    }
    return fig, main_axes, detail_axes, layout_meta


def save_outputs(fig: plt.Figure) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in [
        ("png", {"dpi": RASTER_DPI}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": RASTER_DPI, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        path = OUTPUT_BASE.with_suffix(f".{suffix}")
        fig.savefig(
            path,
            bbox_inches=None,
            facecolor="white",
            edgecolor="white",
            transparent=False,
            **kwargs,
        )
        outputs.append(path)
    return outputs


def raster_qa(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
        dpi = image.info.get("dpi")
        array = np.asarray(image.convert("RGB"))
    with Image.open(REFERENCE_FIGURE) as image:
        reference_array = np.asarray(image.convert("RGB"))
    if width != TARGET_WIDTH_PX:
        raise RuntimeError(f"PNG width changed: expected {TARGET_WIDTH_PX}, found {width}")
    if height != TARGET_HEIGHT_PX:
        raise RuntimeError(
            f"PNG height changed: expected {TARGET_HEIGHT_PX}, found {height}"
        )
    nonwhite_fraction = float((array.min(axis=2) < 250).mean())
    if nonwhite_fraction < 0.02:
        raise RuntimeError("PNG appears unexpectedly blank")
    if reference_array.shape != array.shape:
        raise RuntimeError("Reference and revised PNG canvas shapes differ")
    nonwhite_mask = array.min(axis=2) < 250
    reference_nonwhite_mask = reference_array.min(axis=2) < 250
    y_indices, x_indices = np.nonzero(nonwhite_mask)
    reference_y, reference_x = np.nonzero(reference_nonwhite_mask)
    if not len(x_indices) or not len(reference_x):
        raise RuntimeError("Could not locate visible raster content")
    reference_left_px = int(reference_x.min())
    revised_left_px = int(x_indices.min())
    reference_right_blank_px = int(
        width - 1 - reference_x.max()
    )
    revised_right_blank_px = int(width - 1 - x_indices.max())
    left_blank_reduction_px = reference_left_px - revised_left_px
    left_blank_reduction_fraction = (
        left_blank_reduction_px / reference_left_px
    )
    if not 0.30 <= left_blank_reduction_fraction <= 0.45:
        raise RuntimeError(
            "Left blank-space reduction is outside the requested 30-45%"
        )
    if revised_left_px < MINIMUM_LEFT_SAFETY_PX:
        raise RuntimeError("Revised left visible content lacks safety margin")
    if abs(revised_right_blank_px - reference_right_blank_px) > 1:
        raise RuntimeError("Rendered right boundary changed from the baseline")
    return {
        "png_width_px": int(width),
        "png_height_px": int(height),
        "png_dpi": [float(value) for value in dpi] if dpi else None,
        "nonwhite_fraction": nonwhite_fraction,
        "raster_margin_comparison": {
            "visibility_threshold": "minimum RGB channel < 250",
            "reference_leftmost_visible_ink_px": reference_left_px,
            "revised_leftmost_visible_ink_px": revised_left_px,
            "left_blank_reduction_px": left_blank_reduction_px,
            "left_blank_reduction_fraction": (
                left_blank_reduction_fraction
            ),
            "reference_right_blank_px": reference_right_blank_px,
            "revised_right_blank_px": revised_right_blank_px,
            "right_blank_delta_px": (
                revised_right_blank_px - reference_right_blank_px
            ),
            "minimum_left_safety_px": MINIMUM_LEFT_SAFETY_PX,
            "left_safety_pass": True,
            "right_boundary_preserved_within_px": 1,
        },
        "corner_pixels_rgb": [
            array[0, 0].astype(int).tolist(),
            array[0, -1].astype(int).tolist(),
            array[-1, 0].astype(int).tolist(),
            array[-1, -1].astype(int).tolist(),
        ],
    }


def rendered_layout_qa(
    fig: plt.Figure,
    main_axes: list[plt.Axes],
    detail_axes: list[plt.Axes],
    layout_meta: dict[str, object],
) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_artists = layout_meta["title_artists"]
    legend = layout_meta["merged_legend"]
    window_records = layout_meta["window_records"]
    detail_marker_records = layout_meta["detail_marker_records"]

    title_bboxes = [
        artist.get_window_extent(renderer) for artist in title_artists
    ]
    legend_bbox = legend.get_window_extent(renderer)
    first_title_bbox = title_bboxes[0]
    legend_bottom_offset_px = float(legend_bbox.y0 - first_title_bbox.y0)
    expected_legend_offset_px = float(
        LEGEND_UP_SHIFT_FIGURE_FRACTION * fig.bbox.height
    )
    legend_shift_error_px = float(
        legend_bottom_offset_px - expected_legend_offset_px
    )
    legend_title_gap_px = float(legend_bbox.x0 - first_title_bbox.x1)
    legend_plot_gap_px = float(legend_bbox.y0 - main_axes[0].bbox.y1)
    legend_top_margin_px = float(fig.bbox.y1 - legend_bbox.y1)

    if abs(legend_shift_error_px) > 0.5:
        raise RuntimeError(
            "Merged legend upward shift differs from its target by >0.5px"
        )
    if legend_title_gap_px <= 0 or legend_plot_gap_px <= 0:
        raise RuntimeError("Merged legend overlaps the first title or main axis")
    if legend_top_margin_px <= 0:
        raise RuntimeError("Merged legend extends beyond the figure canvas")

    window_checks: list[dict[str, object]] = []
    for record in window_records:
        axis = record["axis"]
        artist = record["artist"]
        start = record["start"]
        stop = record["stop"]
        midpoint = record["midpoint"]
        text_bbox = artist.get_window_extent(renderer)
        midpoint_num = mdates.date2num(midpoint.to_pydatetime())
        start_num = mdates.date2num(start.to_pydatetime())
        stop_num = mdates.date2num(stop.to_pydatetime())
        midpoint_x_px = float(axis.transData.transform((midpoint_num, 0.0))[0])
        start_x_px = float(axis.transData.transform((start_num, 0.0))[0])
        stop_x_px = float(axis.transData.transform((stop_num, 0.0))[0])
        text_center_x_px = float((text_bbox.x0 + text_bbox.x1) / 2.0)
        centre_error_px = text_center_x_px - midpoint_x_px
        clearance_px = min(
            float(text_bbox.x0 - start_x_px),
            float(stop_x_px - text_bbox.x1),
        )
        within_axis_top = bool(text_bbox.y1 <= axis.bbox.y1 + 0.5)
        if abs(centre_error_px) > 0.5:
            raise RuntimeError(
                f"Window marker {record['marker']} is not centred"
            )
        if clearance_px <= 0:
            raise RuntimeError(
                f"Window marker {record['marker']} touches a window boundary"
            )
        if not within_axis_top:
            raise RuntimeError(
                f"Window marker {record['marker']} extends above its main axis"
            )
        window_checks.append(
            {
                "marker": str(record["marker"]),
                "start_utc": start.isoformat(),
                "stop_utc": stop.isoformat(),
                "midpoint_utc": midpoint.isoformat(),
                "rendered_centre_error_px": float(centre_error_px),
                "time_anchor_error_seconds": 0.0,
                "minimum_boundary_clearance_px": float(clearance_px),
                "transparent_text_background": artist.get_bbox_patch() is None,
                "inside_main_axis_top": within_axis_top,
            }
        )

    detail_marker_checks: list[dict[str, object]] = []
    for record in detail_marker_records:
        axis = record["axis"]
        artist = record["artist"]
        marker_bbox = artist.get_window_extent(renderer)
        left_clearance_px = float(marker_bbox.x0 - axis.bbox.x0)
        top_clearance_px = float(axis.bbox.y1 - marker_bbox.y1)
        if left_clearance_px < 0 or top_clearance_px < 0:
            raise RuntimeError(
                f"Detail marker {record['marker']} extends outside its axis"
            )
        detail_marker_checks.append(
            {
                "marker": str(record["marker"]),
                "font_size_pt": float(artist.get_fontsize()),
                "left_axis_clearance_px": left_clearance_px,
                "top_axis_clearance_px": top_clearance_px,
                "transparent_text_background": artist.get_bbox_patch() is None,
            }
        )

    main_time_axes: list[dict[str, object]] = []
    for panel_index, axis in enumerate(main_axes):
        visible_labels = [
            label.get_text()
            for label in axis.get_xticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        expected_xlabel = (
            "UTC" if panel_index == len(main_axes) - 1 else ""
        )
        if not visible_labels or axis.get_xlabel() != expected_xlabel:
            raise RuntimeError(
                f"Main panel {chr(ord('a') + panel_index)} has an invalid x label"
            )
        main_time_axes.append(
            {
                "panel": chr(ord("a") + panel_index),
                "visible_tick_label_count": len(visible_labels),
                "xlabel": axis.get_xlabel(),
                "two_line_date_time_labels": all(
                    "\n" in label for label in visible_labels
                ),
            }
        )

    detail_time_axes: list[dict[str, object]] = []
    for panel_index, axis in enumerate(detail_axes):
        visible_labels = [
            label.get_text()
            for label in axis.get_xticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        if not visible_labels or axis.get_xlabel():
            raise RuntimeError(
                f"Detail panel {panel_index + 1} has an invalid x label"
            )
        detail_time_axes.append(
            {
                "panel": int(panel_index + 1),
                "visible_tick_label_count": len(visible_labels),
                "xlabel": axis.get_xlabel(),
                "two_line_date_time_labels": all(
                    "\n" in label for label in visible_labels
                ),
            }
        )

    def lowest_x_decoration_y0(axis: plt.Axes) -> float:
        bboxes = [
            label.get_window_extent(renderer)
            for label in axis.get_xticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        if axis.get_xlabel():
            bboxes.append(axis.xaxis.label.get_window_extent(renderer))
        if not bboxes:
            raise RuntimeError("Axis has no visible x-axis decoration")
        return float(min(bbox.y0 for bbox in bboxes))

    first_detail_row = detail_axes[:4]
    second_detail_row = detail_axes[4:]
    row_clearances_px = {
        "a_to_details_1_4": float(
            lowest_x_decoration_y0(main_axes[0])
            - max(axis.bbox.y1 for axis in first_detail_row)
        ),
        "details_1_4_to_b": float(
            min(lowest_x_decoration_y0(axis) for axis in first_detail_row)
            - title_bboxes[1].y1
        ),
        "b_to_c": float(
            lowest_x_decoration_y0(main_axes[1])
            - title_bboxes[2].y1
        ),
        "c_to_details_5_8": float(
            lowest_x_decoration_y0(main_axes[2])
            - max(axis.bbox.y1 for axis in second_detail_row)
        ),
        "details_5_8_to_d": float(
            min(lowest_x_decoration_y0(axis) for axis in second_detail_row)
            - title_bboxes[3].y1
        ),
    }
    minimum_row_clearance_px = min(row_clearances_px.values())
    if minimum_row_clearance_px <= 0:
        raise RuntimeError(
            "Adjacent rows overlap: "
            + json.dumps(row_clearances_px, sort_keys=True)
        )

    def visible_tick_bboxes(
        axis: plt.Axes,
        orientation: str,
    ) -> list[object]:
        labels = (
            axis.get_xticklabels()
            if orientation == "x"
            else axis.get_yticklabels()
        )
        return [
            label.get_window_extent(renderer)
            for label in labels
            if label.get_visible() and label.get_text().strip()
        ]

    horizontal_clearances: list[dict[str, object]] = []
    for row_name, row_axes in [
        ("details_1_4", first_detail_row),
        ("details_5_8", second_detail_row),
    ]:
        for pair_index, (left_axis, right_axis) in enumerate(
            zip(row_axes[:-1], row_axes[1:]),
            start=1,
        ):
            left_x_bboxes = visible_tick_bboxes(left_axis, "x")
            right_x_bboxes = visible_tick_bboxes(right_axis, "x")
            right_y_bboxes = visible_tick_bboxes(right_axis, "y")
            axis_gap_px = float(right_axis.bbox.x0 - left_axis.bbox.x1)
            x_tick_gap_px = float(
                min(bbox.x0 for bbox in right_x_bboxes)
                - max(bbox.x1 for bbox in left_x_bboxes)
            )
            right_y_to_left_axis_px = float(
                min(bbox.x0 for bbox in right_y_bboxes)
                - left_axis.bbox.x1
            )
            minimum_text_clearance_px = min(
                x_tick_gap_px,
                right_y_to_left_axis_px,
            )
            if (
                axis_gap_px <= 0
                or minimum_text_clearance_px <= 0
            ):
                raise RuntimeError(
                    "Horizontal overlap in "
                    f"{row_name}, pair {pair_index}: "
                    + json.dumps(
                        {
                            "axis_gap_px": axis_gap_px,
                            "x_tick_label_clearance_px": x_tick_gap_px,
                            "right_y_labels_to_left_axis_px": (
                                right_y_to_left_axis_px
                            ),
                        },
                        sort_keys=True,
                    )
                )
            horizontal_clearances.append(
                {
                    "row": row_name,
                    "pair": f"{pair_index}-{pair_index + 1}",
                    "axis_gap_px": axis_gap_px,
                    "x_tick_label_clearance_px": x_tick_gap_px,
                    "right_y_labels_to_left_axis_px": (
                        right_y_to_left_axis_px
                    ),
                    "minimum_text_clearance_px": (
                        minimum_text_clearance_px
                    ),
                }
            )

    main_axis_sizes_px = [
        {
            "panel": chr(ord("a") + index),
            "width_px": float(axis.bbox.width),
            "height_px": float(axis.bbox.height),
        }
        for index, axis in enumerate(main_axes)
    ]
    detail_axis_sizes_px = [
        {
            "panel": int(index + 1),
            "width_px": float(axis.bbox.width),
            "height_px": float(axis.bbox.height),
        }
        for index, axis in enumerate(detail_axes)
    ]

    return {
        "font_scale_relative_to_reference": FONT_SCALE,
        "font_sizes_pt": {
            "base": 8.2 * FONT_SCALE,
            "axes_labels": 8.8 * FONT_SCALE,
            "axes_titles": 9.4 * FONT_SCALE,
            "tick_labels": 7.4 * FONT_SCALE,
            "panel_letters": 10.0 * FONT_SCALE,
            "main_window_markers": 11.0 * FONT_SCALE,
            "detail_window_markers": DETAIL_MARKER_FONT_PT,
            "merged_legend": 7.1 * FONT_SCALE,
        },
        "main_time_axes": main_time_axes,
        "detail_time_axes": detail_time_axes,
        "window_marker_checks": window_checks,
        "detail_marker_checks": detail_marker_checks,
        "maximum_window_centre_error_px": float(
            max(abs(item["rendered_centre_error_px"]) for item in window_checks)
        ),
        "minimum_window_boundary_clearance_px": float(
            min(item["minimum_boundary_clearance_px"] for item in window_checks)
        ),
        "merged_legend": {
            "labels": [text.get_text() for text in legend.get_texts()],
            "columns": 8,
            "upward_shift_figure_fraction": (
                LEGEND_UP_SHIFT_FIGURE_FRACTION
            ),
            "title_bottom_offset_px": legend_bottom_offset_px,
            "target_offset_px": expected_legend_offset_px,
            "shift_error_px": legend_shift_error_px,
            "title_horizontal_clearance_px": legend_title_gap_px,
            "main_axis_clearance_px": legend_plot_gap_px,
            "top_canvas_margin_px": legend_top_margin_px,
        },
        "row_clearances_px": row_clearances_px,
        "minimum_row_clearance_px": float(minimum_row_clearance_px),
        "horizontal_detail_clearances_px": horizontal_clearances,
        "minimum_horizontal_text_clearance_px": float(
            min(
                item["minimum_text_clearance_px"]
                for item in horizontal_clearances
            )
        ),
        "main_axis_sizes_px": main_axis_sizes_px,
        "detail_axis_sizes_px": detail_axis_sizes_px,
        "main_axis_bounds_px": [
            {
                "panel": chr(ord("a") + index),
                "x0": float(axis.bbox.x0),
                "y0": float(axis.bbox.y0),
                "x1": float(axis.bbox.x1),
                "y1": float(axis.bbox.y1),
            }
            for index, axis in enumerate(main_axes)
        ],
        "detail_axis_bounds_px": [
            {
                "panel": int(index + 1),
                "x0": float(axis.bbox.x0),
                "y0": float(axis.bbox.y0),
                "x1": float(axis.bbox.x1),
                "y1": float(axis.bbox.y1),
            }
            for index, axis in enumerate(detail_axes)
        ],
    }


def baseline_layout_lock_qa(
    fig: plt.Figure,
    main_axes: list[plt.Axes],
    detail_axes: list[plt.Axes],
    layout_meta: dict[str, object],
    current_qa: dict[str, object],
) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reference = json.loads(
        REFERENCE_PROVENANCE.read_text(encoding="utf-8")
    )
    reference_qa = reference["visual_qa"]

    reference_main_sizes = reference_qa["main_axis_sizes_px"]
    reference_detail_sizes = reference_qa["detail_axis_sizes_px"]
    current_main_sizes = current_qa["main_axis_sizes_px"]
    current_detail_sizes = current_qa["detail_axis_sizes_px"]
    main_height_deltas = [
        float(current["height_px"] - prior["height_px"])
        for current, prior in zip(
            current_main_sizes,
            reference_main_sizes,
            strict=True,
        )
    ]
    detail_height_deltas = [
        float(current["height_px"] - prior["height_px"])
        for current, prior in zip(
            current_detail_sizes,
            reference_detail_sizes,
            strict=True,
        )
    ]
    main_width_gains = [
        float(current["width_px"] - prior["width_px"])
        for current, prior in zip(
            current_main_sizes,
            reference_main_sizes,
            strict=True,
        )
    ]
    detail_width_gains = [
        float(current["width_px"] - prior["width_px"])
        for current, prior in zip(
            current_detail_sizes,
            reference_detail_sizes,
            strict=True,
        )
    ]
    expected_main_width_gain_px = (
        (REFERENCE_GRID_LEFT - GRID_LEFT) * fig.bbox.width
    )
    if max(abs(value) for value in main_height_deltas) > 1e-6:
        raise RuntimeError("A main-axis height changed from the baseline")
    if max(abs(value) for value in detail_height_deltas) > 1e-6:
        raise RuntimeError("A detail-axis height changed from the baseline")
    if max(
        abs(value - expected_main_width_gain_px)
        for value in main_width_gains
    ) > 1e-6:
        raise RuntimeError("Main-axis width gain does not match left release")
    if min(detail_width_gains) <= 0:
        raise RuntimeError("Released width was not assigned to detail axes")

    reference_rows = reference_qa["row_clearances_px"]
    current_rows = current_qa["row_clearances_px"]
    row_clearance_deltas = {
        key: float(current_rows[key] - reference_rows[key])
        for key in reference_rows
    }
    if max(abs(value) for value in row_clearance_deltas.values()) > 1e-6:
        raise RuntimeError("Vertical row clearances changed from the baseline")
    if current_qa["font_sizes_pt"] != reference_qa["font_sizes_pt"]:
        raise RuntimeError("Figure typography changed from the baseline")

    right_target_px = GRID_RIGHT * fig.bbox.width
    main_right_errors = [
        float(axis.bbox.x1 - right_target_px)
        for axis in main_axes
    ]
    detail_right_error = float(
        max(axis.bbox.x1 for axis in detail_axes) - right_target_px
    )
    if max(abs(value) for value in main_right_errors) > 1e-6:
        raise RuntimeError("A main-axis right boundary changed")
    if abs(detail_right_error) > 1e-6:
        raise RuntimeError("The detail-grid right boundary changed")

    visible_texts = [
        artist
        for artist in fig.findobj(match=lambda item: isinstance(item, Text))
        if (
            artist.get_visible()
            and artist.get_text()
            and artist.get_text().strip()
        )
    ]
    text_boxes = [
        artist.get_window_extent(renderer)
        for artist in visible_texts
    ]
    text_outside = [
        artist.get_text()
        for artist, box in zip(
            visible_texts,
            text_boxes,
            strict=True,
        )
        if (
            box.x0 < -0.5
            or box.y0 < -0.5
            or box.x1 > fig.bbox.x1 + 0.5
            or box.y1 > fig.bbox.y1 + 0.5
        )
    ]
    if text_outside:
        raise RuntimeError(
            "Rendered text extends outside the canvas: "
            + json.dumps(text_outside, ensure_ascii=False)
        )

    panel_letter_boxes = []
    for index, axis in enumerate(main_axes):
        letter = chr(ord("a") + index)
        matches = [
            text.get_window_extent(renderer)
            for text in axis.texts
            if text.get_text() == letter
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Could not isolate panel letter {letter}")
        panel_letter_boxes.extend(matches)
    y_label_boxes = [
        axis.yaxis.label.get_window_extent(renderer)
        for axis in main_axes
    ]
    protected_left_x = float(
        min(
            box.x0
            for box in [*panel_letter_boxes, *y_label_boxes]
        )
    )
    if protected_left_x < MINIMUM_LEFT_SAFETY_PX:
        raise RuntimeError(
            "Panel letters or y-axis labels lack left safety clearance"
        )

    legend = layout_meta["merged_legend"]
    legend_box = legend.get_window_extent(renderer)
    legend_right_margin_px = float(fig.bbox.x1 - legend_box.x1)
    if legend_right_margin_px <= 0:
        raise RuntimeError("Merged legend crosses the right canvas edge")

    return {
        "baseline_layout_lock": {
            "reference_stem": REFERENCE_STEM,
            "canvas_fraction": {
                "reference_grid_left": REFERENCE_GRID_LEFT,
                "revised_grid_left": GRID_LEFT,
                "grid_left_delta": GRID_LEFT - REFERENCE_GRID_LEFT,
                "grid_right_unchanged": GRID_RIGHT,
            },
            "main_axis_width_gain_px": main_width_gains,
            "expected_main_axis_width_gain_px": (
                expected_main_width_gain_px
            ),
            "detail_axis_width_gain_px": detail_width_gains,
            "main_axis_height_delta_px": main_height_deltas,
            "detail_axis_height_delta_px": detail_height_deltas,
            "vertical_row_clearance_delta_px": row_clearance_deltas,
            "font_sizes_unchanged": True,
            "main_axis_right_edge_error_px": main_right_errors,
            "detail_grid_right_edge_error_px": detail_right_error,
            "minimum_panel_letter_or_y_label_left_x_px": (
                protected_left_x
            ),
            "minimum_required_left_safety_px": (
                MINIMUM_LEFT_SAFETY_PX
            ),
            "text_outside_canvas": text_outside,
            "legend_right_canvas_margin_px": legend_right_margin_px,
            "vertical_layout_unchanged": True,
            "only_intended_layout_change": (
                "GridSpec left boundary moved left; released width "
                "distributed across all main and detail axes"
            ),
        }
    }


def vector_qa() -> dict[str, object]:
    svg_path = OUTPUT_BASE.with_suffix(".svg")
    svg = svg_path.read_text(encoding="utf-8")
    if "<text" not in svg:
        raise RuntimeError("SVG text is not editable")
    missing_markers = [
        marker
        for marker in layout_base.DETAIL_PANEL_MARKERS
        if marker not in svg
    ]
    if missing_markers:
        raise RuntimeError(f"Missing detail markers in SVG: {missing_markers}")
    if "merged-phase-observation-legend" not in svg:
        raise RuntimeError("Merged legend group is missing from SVG")

    def svg_text_counter(path: Path) -> Counter[str]:
        root = ElementTree.parse(path).getroot()
        texts = []
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] != "text":
                continue
            text = "".join(element.itertext()).strip()
            if text:
                texts.append(text)
        return Counter(texts)

    current_text = svg_text_counter(svg_path)
    reference_text = svg_text_counter(
        FIGURE_DIR / f"{REFERENCE_STEM}.svg"
    )
    if current_text != reference_text:
        raise RuntimeError("SVG visible text content changed from the baseline")
    return {
        "svg_editable_text": True,
        "all_eight_detail_markers_present": True,
        "merged_legend_present": True,
        "svg_text_content_matches_reference": True,
        "svg_text_element_count": int(sum(current_text.values())),
    }


def write_provenance(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    detail_status: pd.DataFrame,
    observations: pd.DataFrame,
    outputs: list[Path],
    qa: dict[str, object],
    protected_before: dict[str, tuple[int, str]],
) -> None:
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_stem": STEM,
        "backend": "Python/matplotlib",
        "figure_archetype": "quantitative grid with interleaved detail rows",
        "core_conclusion": (
            "Full-mission four-band C/N0 trends remain primary, while the two "
            "unchanged 24-hour evidence windows are placed immediately after "
            "the corresponding pair of full-mission panels."
        ),
        "scientific_content_policy": {
            "model_inference_run": False,
            "satellite_reselection": False,
            "window_reselection": False,
            "smoothing_or_clipping_added": False,
            "data_values_changed": False,
            "background_states_changed": False,
            "source": "accepted figure source tables only",
            "canvas_dimensions_changed": False,
            "right_boundary_changed": False,
            "vertical_layout_changed": False,
            "typography_changed": False,
            "only_layout_change": (
                "GridSpec left boundary moved from 0.047 to 0.036"
            ),
        },
        "immediate_reference_wide_compact_figure": {
            **file_record(REFERENCE_FIGURE),
            "width_px": REFERENCE_WIDTH_PX,
            "height_px": REFERENCE_HEIGHT_PX,
        },
        "original_figure": {
            **file_record(ORIGINAL_FIGURE),
            "width_px": ORIGINAL_WIDTH_PX,
            "height_px": ORIGINAL_HEIGHT_PX,
        },
        "direct_inputs": [
            file_record(path)
            for path in [
                SOURCE_DATA,
                SELECTION_DATA,
                DETAIL_STATUS_DATA,
                OBSERVATION_DATA,
                WINDOW_SCORE_DATA,
                REFERENCE_SCRIPT,
                ORIGINAL_ENTRY,
                ORIGINAL_LAYOUT_MODULE,
            ]
        ],
        "frozen_data_summary": {
            "model_rows": int(len(data)),
            "model_rows_by_signal": {
                str(key): int(value)
                for key, value in data.groupby("signal_name").size().items()
            },
            "selection_rows": int(len(selected)),
            "selected_links": [
                {
                    "signal_name": str(row.signal_name),
                    "svid": int(row.svid),
                }
                for row in selected.itertuples()
            ],
            "detail_status_rows": int(len(detail_status)),
            "observed_1s_rows": int(len(observations)),
            "mission_start_utc": data["utc"].min().isoformat(),
            "mission_stop_utc": data["utc"].max().isoformat(),
            "detail_windows": [
                {
                    "mode": mode,
                    "start_utc": start.isoformat(),
                    "stop_utc": stop.isoformat(),
                }
                for mode, start, stop in DETAIL_WINDOWS
            ],
        },
        "layout": {
            "row_order": [
                "a full-width",
                "details 1-4",
                "b full-width",
                "c full-width",
                "details 5-8",
                "d full-width",
            ],
            "detail_mapping": {
                "1": "GPS L1 coverage window",
                "2": "GPS L1 sidelobe-count window",
                "3": "GPS L5 coverage window",
                "4": "GPS L5 sidelobe-count window",
                "5": "Galileo E1 coverage window",
                "6": "Galileo E1 sidelobe-count window",
                "7": "Galileo E5a coverage window",
                "8": "Galileo E5a sidelobe-count window",
            },
            "target_png_width_px": TARGET_WIDTH_PX,
            "target_png_height_px": TARGET_HEIGHT_PX,
            "figure_width_in": FIGURE_WIDTH_IN,
            "figure_height_in": FIGURE_HEIGHT_IN,
            "canvas_width_scale_vs_reference": WIDTH_SCALE,
            "canvas_height_scale_vs_reference": (
                TARGET_HEIGHT_PX / REFERENCE_HEIGHT_PX
            ),
            "grid_height_ratios": GRID_HEIGHT_RATIOS,
            "grid_hspace": 0.0,
            "detail_wspace": DETAIL_WSPACE,
            "spacing_strategy": (
                "explicit spacer rows separate data axes; smaller spacers "
                "couple each main panel to its adjacent detail row"
            ),
            "all_main_panels_show_date_time_ticks": True,
            "utc_axis_title_policy": (
                "detail panels and main panels a-c omit UTC; "
                "only main panel d retains UTC"
            ),
            "window_numbers_use_exact_time_midpoints": True,
            "legend_structure": "one merged eight-entry legend",
            "legend_upward_shift_figure_fraction": (
                LEGEND_UP_SHIFT_FIGURE_FRACTION
            ),
            "detail_marker_font_size_pt": DETAIL_MARKER_FONT_PT,
            "reference_grid_left": REFERENCE_GRID_LEFT,
            "revised_grid_left": GRID_LEFT,
            "grid_right": GRID_RIGHT,
            "released_width_fraction": (
                REFERENCE_GRID_LEFT - GRID_LEFT
            ),
            "released_width_px": (
                (REFERENCE_GRID_LEFT - GRID_LEFT)
                * TARGET_WIDTH_PX
            ),
            "left_margin_strategy": (
                "move GridSpec left edge and distribute released width "
                "across all main and detail axes; no raster cropping"
            ),
        },
        "visual_qa": qa,
        "outputs": [file_record(path) for path in outputs],
        "plot_script": file_record(Path(__file__)),
        "protected_prior_files": [
            {
                "path": str(
                    Path(path)
                    .resolve()
                    .relative_to(ROOT.resolve())
                ).replace("\\", "/"),
                "bytes_before": int(size_hash[0]),
                "sha256_before": size_hash[1],
                "sha256_after": sha256_file(Path(path)),
                "unchanged": sha256_file(Path(path)) == size_hash[1],
            }
            for path, size_hash in protected_before.items()
        ],
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    reference_outputs = [
        FIGURE_DIR / f"{REFERENCE_STEM}.{suffix}"
        for suffix in ["png", "pdf", "svg", "tiff"]
    ]
    protected = [
        ORIGINAL_FIGURE,
        REFERENCE_FIGURE,
        REFERENCE_PROVENANCE,
        REFERENCE_SCRIPT,
        *reference_outputs,
        ORIGINAL_ENTRY,
        ORIGINAL_LAYOUT_MODULE,
        SOURCE_DATA,
        SELECTION_DATA,
        DETAIL_STATUS_DATA,
        OBSERVATION_DATA,
        WINDOW_SCORE_DATA,
    ]
    before = snapshot(protected)
    data, selected, detail_status, observations = load_frozen_sources()
    fig, main_axes, detail_axes, layout_meta = build_figure(
        data, selected, detail_status, observations
    )
    if len(main_axes) != 4 or len(detail_axes) != 8:
        raise RuntimeError("Unexpected main/detail axes count")
    qa = rendered_layout_qa(
        fig,
        main_axes,
        detail_axes,
        layout_meta,
    )
    qa.update(
        baseline_layout_lock_qa(
            fig,
            main_axes,
            detail_axes,
            layout_meta,
            qa,
        )
    )
    outputs = save_outputs(fig)
    plt.close(fig)

    qa.update(raster_qa(OUTPUT_BASE.with_suffix(".png")))
    qa.update(vector_qa())
    qa.update(
        {
            "main_axes": len(main_axes),
            "detail_axes": len(detail_axes),
            "white_background": True,
            "merged_phase_and_anomaly_legend": True,
            "all_reference_files_unchanged": True,
        }
    )
    after = snapshot(protected)
    if before != after:
        raise RuntimeError("A frozen source or original figure changed")
    write_provenance(
        data,
        selected,
        detail_status,
        observations,
        outputs,
        qa,
        before,
    )

    print(json.dumps(qa, indent=2))
    for output in outputs:
        print(output)
    print(PROVENANCE_PATH)


if __name__ == "__main__":
    main()
