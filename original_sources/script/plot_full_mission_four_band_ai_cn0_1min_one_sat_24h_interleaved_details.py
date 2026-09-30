#!/usr/bin/env python3
"""Re-layout the frozen full-mission C/N0 figure with interleaved details.

The script reads the exact source tables already exported by the accepted
figure. It does not run inference, choose satellites, select windows, smooth,
clip, or otherwise recompute scientific content.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


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
from PIL import Image

import build_dynamic_multiband_one_common_1min as one_minute_builder
import plot_full_mission_four_band_ai_cn0_sidelobe_1min_three_sat_24h as layout_base


BASE_STEM = "Fig_full_mission_four_band_AI_CN0_1min_one_sat_24h"
ORIGINAL_STEM = f"{BASE_STEM}_reselected_24h"
STEM = f"{ORIGINAL_STEM}_interleaved_details"
FIGURE_DIR = ROOT / "figure" / "paper_draft_v2"
OUTPUT_BASE = FIGURE_DIR / STEM
PROVENANCE_PATH = FIGURE_DIR / f"{STEM}_provenance.json"
SOURCE_DIR = ROOT / "table" / "figure_source_data"

ORIGINAL_FIGURE = FIGURE_DIR / f"{ORIGINAL_STEM}.png"
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

TARGET_WIDTH_PX = 10_785
RASTER_DPI = 600
FIGURE_WIDTH_IN = TARGET_WIDTH_PX / RASTER_DPI
FIGURE_HEIGHT_IN = 10.0

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
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.1,
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
) -> None:
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
    ax.text(
        detail_stop - pd.Timedelta(hours=1.2),
        0.95,
        marker,
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        color="#111111",
        bbox={
            "boxstyle": "square,pad=0.07",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
        clip_on=False,
        zorder=9,
    )


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
) -> tuple[plt.Figure, list[plt.Axes], list[plt.Axes]]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    grid = fig.add_gridspec(
        6,
        4,
        height_ratios=[1.05, 0.82, 1.05, 1.05, 0.82, 1.05],
        left=0.047,
        right=0.994,
        bottom=0.058,
        top=0.947,
        hspace=0.36,
        wspace=0.09,
    )
    main_rows = [0, 2, 3, 5]
    main_axes: list[plt.Axes] = []
    for index, row in enumerate(main_rows):
        share = main_axes[0] if main_axes else None
        main_axes.append(fig.add_subplot(grid[row, :], sharex=share))

    detail_axes: list[plt.Axes] = []
    detail_axis_map: dict[tuple[int, int], plt.Axes] = {}
    for panel_index in range(4):
        detail_row = 1 if panel_index < 2 else 4
        column_start = 0 if panel_index % 2 == 0 else 2
        for window_index in range(2):
            detail_ax = fig.add_subplot(grid[detail_row, column_start + window_index])
            detail_axis_map[(panel_index, window_index)] = detail_ax
            detail_axes.append(detail_ax)

    phases = layout_base.base.phase_intervals(data)
    mission_start = data["utc"].min()
    mission_stop = data["utc"].max()
    detail_markers = layout_base.DETAIL_PANEL_MARKERS

    for panel_index, signal in enumerate(layout_base.base.SIGNALS):
        main_ax = main_axes[panel_index]
        panel = data[data["signal_name"].eq(signal)].copy()
        chosen = selected[selected["signal_name"].eq(signal)].sort_values(
            "rank_within_constellation"
        )

        layout_base.draw_background(main_ax, panel)
        for window_index, (_, detail_start, detail_stop) in enumerate(DETAIL_WINDOWS):
            marker_index = panel_index * 2 + window_index
            draw_main_window_marker(
                main_ax,
                detail_start,
                detail_stop,
                detail_markers[marker_index],
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
            detail_ax.set_xlabel("UTC", labelpad=1.0, fontsize=7.4)
            detail_ax.text(
                0.022,
                0.95,
                detail_markers[panel_index * 2 + window_index],
                transform=detail_ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.4,
                fontweight="bold",
                fontfamily="DejaVu Sans",
                color="#111111",
                zorder=8,
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
        main_ax.set_title(title, loc="left", fontsize=9.4, fontweight="bold", pad=2)
        main_ax.text(
            -0.022,
            1.006,
            chr(ord("a") + panel_index),
            transform=main_ax.transAxes,
            fontsize=10.0,
            fontweight="bold",
            va="bottom",
        )
        main_ax.set_xlim(mission_start, mission_stop)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=13)
    for main_ax in main_axes:
        main_ax.xaxis.set_major_locator(locator)
        main_ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M", tz=mission_start.tz)
        )
    for main_ax in main_axes[:-1]:
        main_ax.tick_params(labelbottom=False)
    main_axes[-1].set_xlabel("UTC", labelpad=2.0)

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
    fig.legend(
        handles=phase_handles,
        loc="upper center",
        bbox_to_anchor=(0.265, 0.991),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#697077",
        facecolor="white",
        framealpha=1.0,
        handlelength=1.6,
        columnspacing=1.1,
        fontsize=7.1,
    )
    fig.legend(
        handles=detail_handles,
        loc="upper center",
        bbox_to_anchor=(0.755, 0.991),
        ncol=len(detail_handles),
        frameon=True,
        fancybox=False,
        edgecolor="#697077",
        facecolor="white",
        framealpha=1.0,
        handlelength=1.5,
        columnspacing=0.9,
        fontsize=7.1,
    )
    return fig, main_axes, detail_axes


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
    if width != TARGET_WIDTH_PX:
        raise RuntimeError(f"PNG width changed: expected {TARGET_WIDTH_PX}, found {width}")
    if height < 5_500:
        raise RuntimeError(f"Interleaved layout is unexpectedly short: {height}px")
    nonwhite_fraction = float((array.min(axis=2) < 250).mean())
    if nonwhite_fraction < 0.02:
        raise RuntimeError("PNG appears unexpectedly blank")
    return {
        "png_width_px": int(width),
        "png_height_px": int(height),
        "png_dpi": [float(value) for value in dpi] if dpi else None,
        "nonwhite_fraction": nonwhite_fraction,
        "corner_pixels_rgb": [
            array[0, 0].astype(int).tolist(),
            array[0, -1].astype(int).tolist(),
            array[-1, 0].astype(int).tolist(),
            array[-1, -1].astype(int).tolist(),
        ],
    }


def vector_qa() -> dict[str, object]:
    svg = OUTPUT_BASE.with_suffix(".svg").read_text(encoding="utf-8")
    if "<text" not in svg:
        raise RuntimeError("SVG text is not editable")
    missing_markers = [
        marker
        for marker in layout_base.DETAIL_PANEL_MARKERS
        if marker not in svg
    ]
    if missing_markers:
        raise RuntimeError(f"Missing detail markers in SVG: {missing_markers}")
    return {
        "svg_editable_text": True,
        "all_eight_detail_markers_present": True,
    }


def write_provenance(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    detail_status: pd.DataFrame,
    observations: pd.DataFrame,
    outputs: list[Path],
    qa: dict[str, object],
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
            "source": "accepted figure source tables only",
        },
        "original_figure": {
            **file_record(ORIGINAL_FIGURE),
            "width_px": TARGET_WIDTH_PX,
        },
        "direct_inputs": [
            file_record(path)
            for path in [
                SOURCE_DATA,
                SELECTION_DATA,
                DETAIL_STATUS_DATA,
                OBSERVATION_DATA,
                WINDOW_SCORE_DATA,
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
            "figure_width_in": FIGURE_WIDTH_IN,
            "figure_height_in": FIGURE_HEIGHT_IN,
        },
        "visual_qa": qa,
        "outputs": [file_record(path) for path in outputs],
        "plot_script": file_record(Path(__file__)),
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    protected = [
        ORIGINAL_FIGURE,
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
    fig, main_axes, detail_axes = build_figure(
        data, selected, detail_status, observations
    )
    if len(main_axes) != 4 or len(detail_axes) != 8:
        raise RuntimeError("Unexpected main/detail axes count")
    outputs = save_outputs(fig)
    plt.close(fig)

    qa = raster_qa(OUTPUT_BASE.with_suffix(".png"))
    qa.update(vector_qa())
    qa.update(
        {
            "main_axes": len(main_axes),
            "detail_axes": len(detail_axes),
            "white_background": True,
            "shared_phase_legend": True,
            "shared_anomaly_legend": True,
        }
    )
    write_provenance(
        data, selected, detail_status, observations, outputs, qa
    )
    after = snapshot(protected)
    if before != after:
        raise RuntimeError("A frozen source or original figure changed")

    print(json.dumps(qa, indent=2))
    for output in outputs:
        print(output)
    print(PROVENANCE_PATH)


if __name__ == "__main__":
    main()
