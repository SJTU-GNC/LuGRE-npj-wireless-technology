"""Paired full boxplots for published LuGRE and this-work C/N0 discrepancies.

Published boxplot elements are digitized from the archived Parker et al.
Figure 19 raster. This-work elements are computed from the matched approximately
1-Hz residual samples with standard Tukey 1.5-IQR whiskers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figure" / "paper_draft_v2" / "extended"
TABLE_DIR = ROOT / "table" / "algorithm" / "published_lugre_vs_ai_comparison"
PAIRED_SOURCE = TABLE_DIR / "published_lugre_vs_ai_paired_comparison.csv"
RAW_SAMPLE_SOURCE = (
    ROOT
    / "table"
    / "algorithm"
    / "figure19_segments_full_model_residual"
    / "figure19_segments_full_model_residual_samples.csv"
)
ARCHIVED_RASTER = TABLE_DIR / "official_figure19_digitization_source.png"
STEM = "Fig_extended_LuGRE_published_vs_AI_CN0_discrepancy_full_boxplot"

BOX_STATS_OUT = TABLE_DIR / f"{STEM}_box_statistics.csv"
OUTLIERS_OUT = TABLE_DIR / f"{STEM}_outliers.csv"
DIGITIZATION_OUT = TABLE_DIR / f"{STEM}_official_digitization_pixels.csv"
CAPTION_OUT = TABLE_DIR / f"{STEM}_caption.txt"
PROVENANCE_OUT = FIGURE_DIR / f"{STEM}_provenance.json"
QA_OUT = FIGURE_DIR / f"{STEM}_qa.json"

WIDTH_IN = 183.0 / 25.4
HEIGHT_IN = 4.65
DPI = 600
GROUP_OFFSET = 0.18
BOX_WIDTH = 0.27
CAP_WIDTH = 0.15

PUBLISHED_EDGE = "#607392"
PUBLISHED_WHISKER = "#313638"
PUBLISHED_CENTRE = "#D13A2F"
PUBLISHED_OUTLIER = "#B9BCBD"
WORK_EDGE = "#236E72"
WORK_FILL = "#9EC5C0"
WORK_OUTLIER = "#588C8A"
GRID = "#E1E5E6"
INK = "#292D2F"


SEGMENT_LABELS = [
    "OP1_0",
    "OP2_0",
    "OP21_0",
    "OP9_0",
    "OP12_0",
    "OP5_0",
    "OP22_0",
    "OP38_0",
    "OP40_0",
    "OP23_0",
    "OP74_0",
    "OP27_0",
    "OP76_0",
    "OP77_0",
    "OP78_1",
]

# Archived raster geometry. y increases downwards.
X_CENTRE_PX = np.array(
    [251, 358, 466, 573, 681, 788, 896, 1003, 1111, 1218, 1326, 1433, 1541, 1648, 1756],
    dtype=float,
)
MEAN_Y_PX = np.array(
    [123.0, 167.0, 168.0, 263.5, 293.0, 202.0, 259.5, 274.0, 249.0, 290.0, 261.0, 190.0, 172.5, 151.0, 151.0]
)
MEDIAN_Y_PX = np.array(
    [122.5, 166.5, 167.5, 263.5, 292.5, 202.5, 259.5, 274.5, 248.5, 290.5, 260.5, 189.5, 172.5, 151.5, 150.5]
)
Q3_Y_PX = np.array(
    [114.5, 154.5, 140.5, 193.5, 197.5, 187.0, 252.5, 237.5, 235.5, 282.5, 247.5, 178.5, 155.5, 132.5, 145.5]
)
Q1_Y_PX = np.array(
    [131.5, 181.5, 192.5, 307.5, 309.5, 210.5, 266.5, 308.5, 263.5, 297.5, 274.5, 200.5, 185.0, 181.0, 156.5]
)
UPPER_WHISKER_Y_PX = np.array(
    [90.5, 118.0, 98.5, 156.5, 149.0, 156.0, 235.5, 131.5, 194.5, 261.0, 207.5, 151.0, 110.5, 104.0, 133.0]
)
LOWER_WHISKER_Y_PX = np.array(
    [152.0, 218.0, 235.5, 371.5, 437.5, 227.5, 284.0, 402.5, 304.5, 316.5, 314.5, 227.0, 229.5, 235.5, 168.0]
)

# Only separable grey marker centres are retained. Dense overlapping grey
# columns that cannot be decomposed unambiguously are not expanded or guessed.
OUTLIER_Y_PX = {
    "OP1_0": [159],
    "OP2_0": [226, 236, 246, 254, 509],
    "OP21_0": [277],
    "OP9_0": [],
    "OP12_0": [507],
    "OP5_0": [121, 132, 148],
    "OP22_0": [291, 303],
    "OP38_0": [111, 124, 425, 461, 479, 488, 525],
    "OP40_0": [74, 98, 148, 307, 325, 347, 480, 542],
    "OP23_0": [258, 322, 340],
    "OP74_0": [110, 173, 201, 327, 379, 390, 411, 451, 471, 548],
    "OP27_0": [144, 234, 243, 411],
    "OP76_0": [66, 73, 83, 101, 232, 243, 260, 268, 281, 297, 338, 348, 372, 477],
    "OP77_0": [273],
    "OP78_1": [46, 58, 82, 100, 108, 116, 123, 182, 190, 197, 219, 228, 273],
}

GRID_Y_PX = np.array([76.5, 152.5, 227.5, 302.5, 378.5, 453.5, 529.5])
GRID_DBHZ = np.array([-6.0, -8.0, -10.0, -12.0, -14.0, -16.0, -18.0])
DIGITIZATION_UNCERTAINTY_DB = 0.08


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def raster_mapping() -> tuple[float, float, float]:
    slope, intercept = np.polyfit(GRID_Y_PX, GRID_DBHZ, 1)
    residual = float(np.max(np.abs(intercept + slope * GRID_Y_PX - GRID_DBHZ)))
    return float(slope), float(intercept), residual


def build_published_digitization(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not ARCHIVED_RASTER.exists():
        raise FileNotFoundError(ARCHIVED_RASTER)
    with Image.open(ARCHIVED_RASTER) as image:
        if image.size != (1878, 732):
            raise ValueError(f"Unexpected archived raster size: {image.size}")

    slope, intercept, grid_residual = raster_mapping()

    def y_to_db(values) -> np.ndarray:
        return intercept + slope * np.asarray(values, dtype=float)

    pixel_stats = pd.DataFrame(
        {
            "segment_order": np.arange(1, 16),
            "segment_label": SEGMENT_LABELS,
            "x_centre_px": X_CENTRE_PX,
            "upper_whisker_y_px": UPPER_WHISKER_Y_PX,
            "q3_y_px": Q3_Y_PX,
            "median_y_px": MEDIAN_Y_PX,
            "mean_y_px": MEAN_Y_PX,
            "q1_y_px": Q1_Y_PX,
            "lower_whisker_y_px": LOWER_WHISKER_Y_PX,
            "digitization_uncertainty_db": DIGITIZATION_UNCERTAINTY_DB,
        }
    )
    stats = pd.DataFrame(
        {
            "segment_order": np.arange(1, 16),
            "segment_label": SEGMENT_LABELS,
            "operation": paired["operation"],
            "radius_re": paired["radius_re"],
            "mission_phase_code": paired["mission_phase_code"],
            "model": "Published LuGRE simulation (digitized)",
            "n": np.nan,
            "whisker_low_db": y_to_db(LOWER_WHISKER_Y_PX),
            "q1_db": y_to_db(Q1_Y_PX),
            "median_db": y_to_db(MEDIAN_Y_PX),
            "mean_db": y_to_db(MEAN_Y_PX),
            "q3_db": y_to_db(Q3_Y_PX),
            "whisker_high_db": y_to_db(UPPER_WHISKER_Y_PX),
            "source_kind": "digitized published figure elements",
            "whisker_definition": "visible published whisker/cap endpoint",
        }
    )
    outlier_rows = []
    for order, label in enumerate(SEGMENT_LABELS, start=1):
        for index, y_px in enumerate(OUTLIER_Y_PX[label], start=1):
            outlier_rows.append(
                {
                    "segment_order": order,
                    "segment_label": label,
                    "model": "Published LuGRE simulation (digitized)",
                    "outlier_index": index,
                    "outlier_db": float(y_to_db([y_px])[0]),
                    "outlier_y_px": float(y_px),
                    "source_kind": "digitized separable grey outlier marker",
                }
            )
    outliers = pd.DataFrame(outlier_rows)
    metadata = {
        "source_raster_size_px": [1878, 732],
        "gridline_y_px": GRID_Y_PX.tolist(),
        "gridline_values_dbhz": GRID_DBHZ.tolist(),
        "mapping_formula": "discrepancy_dbhz = intercept + slope * raster_y_px",
        "slope_db_per_px": slope,
        "intercept_db": intercept,
        "max_gridline_fit_residual_db": grid_residual,
        "pixel_selection_uncertainty_db": DIGITIZATION_UNCERTAINTY_DB,
        "digitized_separable_outlier_count": int(len(outliers)),
        "outlier_policy": (
            "Only visually separable grey marker centres were retained. "
            "Dense overlapping grey columns were not decomposed or supplemented."
        ),
    }
    return stats, outliers, pixel_stats, metadata


def build_this_work_statistics(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(
        RAW_SAMPLE_SOURCE,
        usecols=["segment_label", "residual_observed_minus_full_model_db"],
    )
    stats_rows = []
    outlier_rows = []
    for order, label in enumerate(SEGMENT_LABELS, start=1):
        values = (
            raw.loc[
                raw["segment_label"].eq(label),
                "residual_observed_minus_full_model_db",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        if values.size == 0:
            raise ValueError(f"No matched residual samples for {label}")
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        inside = values[(values >= lower_fence) & (values <= upper_fence)]
        outliers = values[(values < lower_fence) | (values > upper_fence)]
        pair_row = paired.loc[paired["segment_label"].eq(label)].iloc[0]
        stats_rows.append(
            {
                "segment_order": order,
                "segment_label": label,
                "operation": pair_row["operation"],
                "radius_re": pair_row["radius_re"],
                "mission_phase_code": pair_row["mission_phase_code"],
                "model": "Physics + AI residual model (this work)",
                "n": int(values.size),
                "whisker_low_db": float(inside.min()),
                "q1_db": float(q1),
                "median_db": float(median),
                "mean_db": float(values.mean()),
                "q3_db": float(q3),
                "whisker_high_db": float(inside.max()),
                "source_kind": "computed from matched approximately 1-Hz residual samples",
                "whisker_definition": "Tukey 1.5 x IQR",
            }
        )
        for index, value in enumerate(outliers, start=1):
            outlier_rows.append(
                {
                    "segment_order": order,
                    "segment_label": label,
                    "model": "Physics + AI residual model (this work)",
                    "outlier_index": index,
                    "outlier_db": float(value),
                    "outlier_y_px": np.nan,
                    "source_kind": "matched raw residual sample outside Tukey whisker",
                }
            )
    stats = pd.DataFrame(stats_rows)
    outlier_df = pd.DataFrame(outlier_rows)
    metadata = {
        "sample_count": int(len(raw)),
        "outlier_count": int(len(outlier_df)),
        "whisker_definition": "most extreme sample within Q1/Q3 +/- 1.5 IQR",
        "retraining": False,
    }
    return stats, outlier_df, metadata


def draw_full_box(
    ax: plt.Axes,
    x: float,
    row: pd.Series,
    outliers: np.ndarray,
    published: bool,
) -> dict:
    if published:
        edge = PUBLISHED_EDGE
        face = "white"
        whisker = PUBLISHED_WHISKER
        median_color = PUBLISHED_CENTRE
        outlier_color = PUBLISHED_OUTLIER
        outlier_size = 6.0
        outlier_alpha = 0.58
    else:
        edge = WORK_EDGE
        face = WORK_FILL
        whisker = WORK_EDGE
        median_color = WORK_EDGE
        outlier_color = WORK_OUTLIER
        outlier_size = 2.4
        outlier_alpha = 0.14

    if outliers.size:
        ax.scatter(
            np.full(outliers.size, x),
            outliers,
            s=outlier_size,
            marker="o",
            facecolor=outlier_color,
            edgecolor="none",
            alpha=outlier_alpha,
            linewidths=0,
            clip_on=True,
            zorder=1,
        )
    ax.plot(
        [x, x],
        [row["q3_db"], row["whisker_high_db"]],
        color=whisker,
        lw=0.8,
        zorder=2,
    )
    ax.plot(
        [x, x],
        [row["q1_db"], row["whisker_low_db"]],
        color=whisker,
        lw=0.8,
        zorder=2,
    )
    ax.plot(
        [x - CAP_WIDTH / 2, x + CAP_WIDTH / 2],
        [row["whisker_high_db"], row["whisker_high_db"]],
        color=whisker,
        lw=0.8,
        zorder=2,
    )
    ax.plot(
        [x - CAP_WIDTH / 2, x + CAP_WIDTH / 2],
        [row["whisker_low_db"], row["whisker_low_db"]],
        color=whisker,
        lw=0.8,
        zorder=2,
    )
    ax.add_patch(
        Rectangle(
            (x - BOX_WIDTH / 2, row["q1_db"]),
            BOX_WIDTH,
            row["q3_db"] - row["q1_db"],
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=3,
        )
    )
    ax.plot(
        [x - BOX_WIDTH / 2, x + BOX_WIDTH / 2],
        [row["median_db"], row["median_db"]],
        color=median_color,
        lw=1.0,
        zorder=4,
    )
    if published:
        for marker in ["+", "x"]:
            ax.plot(
                x,
                row["mean_db"],
                marker=marker,
                markersize=5.3,
                markeredgewidth=0.85,
                color=PUBLISHED_CENTRE,
                linestyle="none",
                zorder=5,
            )
    else:
        ax.plot(
            x,
            row["mean_db"],
            marker="o",
            markersize=3.8,
            markerfacecolor=WORK_EDGE,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle="none",
            zorder=5,
        )
    return {
        "box": 1,
        "median": 1,
        "mean": 1,
        "whisker_segments": 2,
        "caps": 2,
        "outliers": int(outliers.size),
    }


def make_figure(
    published_stats: pd.DataFrame,
    published_outliers: pd.DataFrame,
    work_stats: pd.DataFrame,
    work_outliers: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes, dict]:
    configure_style()
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
    fig.subplots_adjust(left=0.105, right=0.992, bottom=0.285, top=0.865)

    centres = np.arange(15, dtype=float)
    published_x = centres - GROUP_OFFSET
    work_x = centres + GROUP_OFFSET
    element_counts = {
        "box": 0,
        "median": 0,
        "mean": 0,
        "whisker_segments": 0,
        "caps": 0,
        "published_outliers": 0,
        "this_work_outliers": 0,
    }
    for index, label in enumerate(SEGMENT_LABELS):
        pub_row = published_stats.loc[published_stats["segment_label"].eq(label)].iloc[0]
        work_row = work_stats.loc[work_stats["segment_label"].eq(label)].iloc[0]
        pub_values = published_outliers.loc[
            published_outliers["segment_label"].eq(label), "outlier_db"
        ].to_numpy(dtype=float)
        work_values = work_outliers.loc[
            work_outliers["segment_label"].eq(label), "outlier_db"
        ].to_numpy(dtype=float)
        pub_counts = draw_full_box(ax, published_x[index], pub_row, pub_values, True)
        work_counts = draw_full_box(ax, work_x[index], work_row, work_values, False)
        for key in ["box", "median", "mean", "whisker_segments", "caps"]:
            element_counts[key] += pub_counts[key] + work_counts[key]
        element_counts["published_outliers"] += pub_counts["outliers"]
        element_counts["this_work_outliers"] += work_counts["outliers"]

    ax.axhline(0, color="#747A7D", lw=0.7, linestyle=(0, (2.4, 2.2)), zorder=0)
    ax.set_xlim(-0.65, 14.65)
    ax.set_ylim(-19, 5)
    ax.set_yticks(np.arange(-18, 4, 3))
    ax.set_ylabel("$C/N_0$ discrepancy\n(observed $-$ model, dB-Hz)", labelpad=8)
    ax.set_xlabel("LuGRE operations", labelpad=7)
    radii = published_stats["radius_re"].to_numpy(dtype=float)
    ax.set_xticks(
        centres,
        [
            f"{label}\n{radius:.2f} $R_E$"
            for label, radius in zip(SEGMENT_LABELS, radii)
        ],
        rotation=90,
        ha="center",
        va="top",
        rotation_mode="default",
    )
    ax.tick_params(
        axis="x",
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        direction="out",
        pad=9,
    )
    ax.tick_params(axis="y", right=False, left=True, direction="out", pad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)

    handles = [
        Patch(
            facecolor="white",
            edgecolor=PUBLISHED_EDGE,
            linewidth=1.0,
            label="Published LuGRE simulation (digitized)",
        ),
        Patch(
            facecolor=WORK_FILL,
            edgecolor=WORK_EDGE,
            linewidth=1.0,
            label="Physics + AI residual model (this work)",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=2,
        handlelength=1.5,
        columnspacing=1.8,
        borderaxespad=0,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    axes_bbox = ax.get_window_extent(renderer)
    legend_bbox = legend.get_window_extent(renderer)
    tick_bboxes = [
        label.get_window_extent(renderer)
        for label in ax.get_xticklabels()
        if label.get_visible() and label.get_text()
    ]
    all_text = [
        ax.xaxis.label,
        ax.yaxis.label,
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
        *legend.get_texts(),
    ]
    text_bboxes = [
        artist.get_window_extent(renderer)
        for artist in all_text
        if artist.get_visible() and artist.get_text()
    ]
    adjacent_overlap = []
    for left, right in zip(tick_bboxes[:-1], tick_bboxes[1:]):
        dx = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
        dy = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
        adjacent_overlap.append(dx * dy)
    all_text_inside = all(
        bbox.x0 >= figure_bbox.x0
        and bbox.y0 >= figure_bbox.y0
        and bbox.x1 <= figure_bbox.x1
        and bbox.y1 <= figure_bbox.y1
        for bbox in text_bboxes
    )
    tick_centres_display = ax.transData.transform(
        np.column_stack([centres, np.zeros_like(centres)])
    )[:, 0]
    published_offsets = published_x - centres
    work_offsets = work_x - centres
    x_tick_axis_gap = axes_bbox.y0 - max(bbox.y1 for bbox in tick_bboxes)

    qa = {
        "backend": "python/matplotlib only",
        "archetype": "single-panel paired quantitative comparison",
        "target_width_mm": 183.0,
        "figure_size_inches": [WIDTH_IN, HEIGHT_IN],
        "group_count": 15,
        "group_centres_data": centres.tolist(),
        "tick_positions_data": ax.get_xticks().tolist(),
        "max_tick_to_group_centre_error_data_units": float(
            np.max(np.abs(ax.get_xticks() - centres))
        ),
        "max_tick_to_group_centre_error_px": float(
            np.max(
                np.abs(
                    tick_centres_display
                    - ax.transData.transform(
                        np.column_stack([ax.get_xticks(), np.zeros_like(centres)])
                    )[:, 0]
                )
            )
        ),
        "published_offset_data_units": np.unique(np.round(published_offsets, 12)).tolist(),
        "this_work_offset_data_units": np.unique(np.round(work_offsets, 12)).tolist(),
        "offset_magnitude_error": float(
            max(
                np.max(np.abs(published_offsets + GROUP_OFFSET)),
                np.max(np.abs(work_offsets - GROUP_OFFSET)),
            )
        ),
        "minimum_between_model_box_gap_px": float(
            ax.transData.transform((GROUP_OFFSET - BOX_WIDTH / 2, 0))[0]
            - ax.transData.transform((-GROUP_OFFSET + BOX_WIDTH / 2, 0))[0]
        ),
        "max_adjacent_x_tick_overlap_px2": float(max(adjacent_overlap, default=0.0)),
        "minimum_x_tick_label_to_axis_gap_px": float(x_tick_axis_gap),
        "all_axis_tick_and_legend_text_inside_canvas": bool(all_text_inside),
        "legend_to_axes_gap_px": float(legend_bbox.y0 - axes_bbox.y1),
        "top_ticks_visible": False,
        "right_ticks_visible": False,
        "element_counts": element_counts,
        "complete_boxplot_elements_per_model_per_group": {
            "box": True,
            "median": True,
            "mean": True,
            "upper_whisker": True,
            "lower_whisker": True,
            "upper_cap": True,
            "lower_cap": True,
        },
    }
    return fig, ax, qa


def save_outputs(fig: plt.Figure) -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": FIGURE_DIR / f"{STEM}.png",
        "pdf": FIGURE_DIR / f"{STEM}.pdf",
        "svg": FIGURE_DIR / f"{STEM}.svg",
        "tiff": FIGURE_DIR / f"{STEM}.tiff",
    }
    fig.savefig(paths["png"], dpi=DPI, facecolor="white")
    fig.savefig(paths["pdf"], facecolor="white")
    fig.savefig(paths["svg"], facecolor="white")
    fig.savefig(
        paths["tiff"],
        dpi=DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return paths


def raster_qa(path: Path) -> dict:
    with Image.open(path).convert("RGB") as image:
        arr = np.asarray(image)
        ink = np.any(arr < 248, axis=2)
        ys, xs = np.where(ink)
        return {
            "png_size_px": list(image.size),
            "png_dpi": [float(v) for v in image.info.get("dpi", (0, 0))],
            "outer_white_margins_px": {
                "left": int(xs.min()),
                "top": int(ys.min()),
                "right": int(image.width - 1 - xs.max()),
                "bottom": int(image.height - 1 - ys.max()),
            },
        }


def write_deliverables(
    published_stats: pd.DataFrame,
    published_outliers: pd.DataFrame,
    pixel_stats: pd.DataFrame,
    work_stats: pd.DataFrame,
    work_outliers: pd.DataFrame,
    digitization_metadata: dict,
    work_metadata: dict,
    paths: dict[str, Path],
    qa: dict,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    combined_stats = pd.concat([published_stats, work_stats], ignore_index=True)
    combined_stats.to_csv(BOX_STATS_OUT, index=False, float_format="%.9f")
    combined_outliers = pd.concat(
        [published_outliers, work_outliers], ignore_index=True
    )
    combined_outliers.to_csv(OUTLIERS_OUT, index=False, float_format="%.9f")
    pixel_stats.to_csv(DIGITIZATION_OUT, index=False, float_format="%.3f")

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "figure_stem": STEM,
        "core_conclusion": (
            "For the same 15 GPS L1 IIR-M/IIIA operation segments, the "
            "this-work final-model residual distributions are generally closer "
            "to zero than the published link-budget simulation discrepancies."
        ),
        "scientific_contract": {
            "residual_sign": "observed C/N0 minus model C/N0",
            "grouping": "same 15 operations and geocentric-radius order",
            "published_model": "Published LuGRE link-budget simulation; not an AI model",
            "this_work_model": (
                "existing physics + robust median-offset regression (L1 loss) "
                "+ HGB final model; no retraining"
            ),
        },
        "published_digitization": digitization_metadata,
        "this_work_computation": work_metadata,
        "qa": qa,
        "source_hashes_sha256": {
            str(path): sha256(path)
            for path in [PAIRED_SOURCE, RAW_SAMPLE_SOURCE, ARCHIVED_RASTER]
        },
        "output_hashes_sha256": {
            str(path): sha256(path)
            for path in [
                *paths.values(),
                BOX_STATS_OUT,
                OUTLIERS_OUT,
                DIGITIZATION_OUT,
            ]
            if path.exists()
        },
    }
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    QA_OUT.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    caption = """Extended Data Fig. X | Published LuGRE and this-work C/N0
discrepancy distributions for matched operation segments. Paired boxplots show
observed minus model C/N0 for the same 15 GPS L1 IIR-M/IIIA operation segments.
For each box, the centre line is the median, the box spans Q1-Q3, whiskers
extend to the most extreme non-outlying value and symbols show the mean.
Published LuGRE simulation elements (open blue-grey boxes, red median lines and
red mean asterisks) were digitized from Parker et al. Figure 19, including
visible whisker/cap endpoints and only visually separable grey outlier markers;
they are not official row-level data. This-work elements (teal boxes and mean
circles) were calculated directly from the matched approximately 1-Hz
residuals using Tukey 1.5 x IQR whiskers; all corresponding raw-sample outliers
are displayed. The physics + AI residual model was not retrained for this
comparison.
"""
    CAPTION_OUT.write_text(caption, encoding="utf-8")


def main() -> None:
    paired = pd.read_csv(PAIRED_SOURCE).sort_values("segment_order").reset_index(drop=True)
    if paired["segment_label"].tolist() != SEGMENT_LABELS:
        raise ValueError("Paired source does not match the locked segment order.")
    published_stats, published_outliers, pixel_stats, digitization_metadata = (
        build_published_digitization(paired)
    )
    work_stats, work_outliers, work_metadata = build_this_work_statistics(paired)

    fig, _, qa = make_figure(
        published_stats,
        published_outliers,
        work_stats,
        work_outliers,
    )
    paths = save_outputs(fig)
    plt.close(fig)
    qa.update(raster_qa(paths["png"]))
    qa["all_formats_exist"] = all(
        path.exists() and path.stat().st_size > 0 for path in paths.values()
    )
    qa["svg_editable_text_elements"] = paths["svg"].read_text(
        encoding="utf-8"
    ).count("<text")
    qa["official_digitized_outlier_count"] = int(len(published_outliers))
    qa["this_work_raw_outlier_count"] = int(len(work_outliers))
    qa["this_work_sample_count"] = int(work_metadata["sample_count"])

    write_deliverables(
        published_stats,
        published_outliers,
        pixel_stats,
        work_stats,
        work_outliers,
        digitization_metadata,
        work_metadata,
        paths,
        qa,
    )
    print(f"PNG={paths['png']}")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
