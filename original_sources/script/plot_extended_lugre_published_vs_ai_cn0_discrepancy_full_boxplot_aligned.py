"""Overlay published and this-work full boxplots at identical OP centres.

Statistics are reused unchanged from the preceding complete paired-boxplot
version. Only geometric encoding, legend wording, and page margins change.
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
SOURCE_STEM = "Fig_extended_LuGRE_published_vs_AI_CN0_discrepancy_full_boxplot"
SOURCE_STATS = TABLE_DIR / f"{SOURCE_STEM}_box_statistics.csv"
SOURCE_OUTLIERS = TABLE_DIR / f"{SOURCE_STEM}_outliers.csv"
SOURCE_DIGITIZATION = TABLE_DIR / f"{SOURCE_STEM}_official_digitization_pixels.csv"
SOURCE_PROVENANCE = FIGURE_DIR / f"{SOURCE_STEM}_provenance.json"

STEM = "Fig_extended_LuGRE_published_vs_AI_CN0_discrepancy_full_boxplot_aligned"
CAPTION_OUT = TABLE_DIR / f"{STEM}_caption.txt"
SOURCE_MANIFEST_OUT = TABLE_DIR / f"{STEM}_source_data_manifest.json"
PROVENANCE_OUT = FIGURE_DIR / f"{STEM}_provenance.json"
QA_OUT = FIGURE_DIR / f"{STEM}_qa.json"

PUBLISHED_MODEL_SOURCE = "Published LuGRE simulation (digitized)"
WORK_MODEL_SOURCE = "Physics + AI residual model (this work)"
PUBLISHED_LEGEND = "Published LuGRE simulation"
WORK_LEGEND = "Physics + AI residual model"

WIDTH_IN = 183.0 / 25.4
HEIGHT_IN = 4.30
DPI = 600
GROUP_OFFSET = 0.0
PUBLISHED_BOX_WIDTH = 0.48
WORK_BOX_WIDTH = 0.25
PUBLISHED_CAP_WIDTH = 0.29
WORK_CAP_WIDTH = 0.15
PAD_IN = 0.02

PUBLISHED_EDGE = "#607392"
PUBLISHED_WHISKER = "#33383A"
PUBLISHED_CENTRE = "#D13A2F"
PUBLISHED_OUTLIER = "#B9BCBD"
WORK_EDGE = "#236E72"
WORK_FILL = "#9EC5C0"
WORK_OUTLIER = "#588C8A"
GRID = "#E1E5E6"
INK = "#292D2F"


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


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = pd.read_csv(SOURCE_STATS)
    outliers = pd.read_csv(SOURCE_OUTLIERS)
    digitization = pd.read_csv(SOURCE_DIGITIZATION)
    counts = stats.groupby("model").size().to_dict()
    expected = {PUBLISHED_MODEL_SOURCE: 15, WORK_MODEL_SOURCE: 15}
    if counts != expected:
        raise ValueError(f"Unexpected model/group counts: {counts}")
    if sorted(stats["segment_order"].unique().tolist()) != list(range(1, 16)):
        raise ValueError("Expected 15 equally ordered OP groups.")
    return stats, outliers, digitization


def draw_outliers(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    published: bool,
) -> int:
    if not values.size:
        return 0
    ax.scatter(
        np.full(values.size, x),
        values,
        s=6.0 if published else 2.4,
        marker="o",
        facecolor=PUBLISHED_OUTLIER if published else WORK_OUTLIER,
        edgecolor="none",
        alpha=0.55 if published else 0.13,
        linewidths=0,
        clip_on=True,
        zorder=1.0 if published else 1.2,
    )
    return int(values.size)


def draw_whiskers_and_box(
    ax: plt.Axes,
    x: float,
    row: pd.Series,
    published: bool,
) -> None:
    if published:
        edge = PUBLISHED_EDGE
        whisker = PUBLISHED_WHISKER
        face = "none"
        width = PUBLISHED_BOX_WIDTH
        cap_width = PUBLISHED_CAP_WIDTH
        box_zorder = 2.0
        whisker_zorder = 1.8
    else:
        edge = WORK_EDGE
        whisker = WORK_EDGE
        face = mpl.colors.to_rgba(WORK_FILL, 0.76)
        width = WORK_BOX_WIDTH
        cap_width = WORK_CAP_WIDTH
        box_zorder = 3.0
        whisker_zorder = 2.8

    ax.plot(
        [x, x],
        [row["q3_db"], row["whisker_high_db"]],
        color=whisker,
        lw=0.8,
        zorder=whisker_zorder,
    )
    ax.plot(
        [x, x],
        [row["q1_db"], row["whisker_low_db"]],
        color=whisker,
        lw=0.8,
        zorder=whisker_zorder,
    )
    ax.plot(
        [x - cap_width / 2, x + cap_width / 2],
        [row["whisker_high_db"], row["whisker_high_db"]],
        color=whisker,
        lw=0.8,
        zorder=whisker_zorder,
    )
    ax.plot(
        [x - cap_width / 2, x + cap_width / 2],
        [row["whisker_low_db"], row["whisker_low_db"]],
        color=whisker,
        lw=0.8,
        zorder=whisker_zorder,
    )
    ax.add_patch(
        Rectangle(
            (x - width / 2, row["q1_db"]),
            width,
            row["q3_db"] - row["q1_db"],
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=box_zorder,
        )
    )


def draw_centres(
    ax: plt.Axes,
    x: float,
    published_row: pd.Series,
    work_row: pd.Series,
) -> None:
    # The wide published median remains visible at both ends of the narrow box.
    ax.plot(
        [x - PUBLISHED_BOX_WIDTH / 2, x + PUBLISHED_BOX_WIDTH / 2],
        [published_row["median_db"], published_row["median_db"]],
        color=PUBLISHED_CENTRE,
        lw=1.05,
        zorder=4.0,
    )
    ax.plot(
        [x - WORK_BOX_WIDTH / 2, x + WORK_BOX_WIDTH / 2],
        [work_row["median_db"], work_row["median_db"]],
        color=WORK_EDGE,
        lw=1.15,
        zorder=4.2,
    )
    for marker in ["+", "x"]:
        ax.plot(
            x,
            published_row["mean_db"],
            marker=marker,
            markersize=6.2,
            markeredgewidth=0.9,
            color=PUBLISHED_CENTRE,
            linestyle="none",
            zorder=5.0,
        )
    ax.plot(
        x,
        work_row["mean_db"],
        marker="o",
        markersize=4.0,
        markerfacecolor=WORK_EDGE,
        markeredgecolor="white",
        markeredgewidth=0.55,
        linestyle="none",
        zorder=5.2,
    )


def make_figure(
    stats: pd.DataFrame,
    outliers: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes, dict]:
    configure_style()
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
    fig.subplots_adjust(left=0.100, right=0.995, bottom=0.255, top=0.915)

    centres = np.arange(15, dtype=float)
    published_centres = centres.copy()
    work_centres = centres.copy()
    element_counts = {
        "box": 0,
        "median": 0,
        "mean": 0,
        "whisker_segments": 0,
        "caps": 0,
        "published_outliers": 0,
        "this_work_outliers": 0,
    }

    for index, order in enumerate(range(1, 16)):
        pub_row = stats.loc[
            stats["segment_order"].eq(order) & stats["model"].eq(PUBLISHED_MODEL_SOURCE)
        ].iloc[0]
        work_row = stats.loc[
            stats["segment_order"].eq(order) & stats["model"].eq(WORK_MODEL_SOURCE)
        ].iloc[0]
        pub_outliers = outliers.loc[
            outliers["segment_order"].eq(order)
            & outliers["model"].eq(PUBLISHED_MODEL_SOURCE),
            "outlier_db",
        ].to_numpy(dtype=float)
        work_outliers = outliers.loc[
            outliers["segment_order"].eq(order)
            & outliers["model"].eq(WORK_MODEL_SOURCE),
            "outlier_db",
        ].to_numpy(dtype=float)

        element_counts["published_outliers"] += draw_outliers(
            ax, centres[index], pub_outliers, True
        )
        element_counts["this_work_outliers"] += draw_outliers(
            ax, centres[index], work_outliers, False
        )
        draw_whiskers_and_box(ax, centres[index], pub_row, True)
        draw_whiskers_and_box(ax, centres[index], work_row, False)
        draw_centres(ax, centres[index], pub_row, work_row)
        element_counts["box"] += 2
        element_counts["median"] += 2
        element_counts["mean"] += 2
        element_counts["whisker_segments"] += 4
        element_counts["caps"] += 4

    ax.axhline(0, color="#747A7D", lw=0.7, linestyle=(0, (2.4, 2.2)), zorder=0)
    ax.set_xlim(-0.65, 14.65)
    ax.set_ylim(-19, 5)
    ax.set_yticks(np.arange(-18, 4, 3))
    ax.set_ylabel("$C/N_0$ discrepancy\n(observed $-$ model, dB-Hz)", labelpad=6)
    ax.set_xlabel("LuGRE operations", labelpad=4)

    published_stats = stats.loc[stats["model"].eq(PUBLISHED_MODEL_SOURCE)].sort_values(
        "segment_order"
    )
    ax.set_xticks(
        centres,
        [
            f"{label}\n{radius:.2f} $R_E$"
            for label, radius in zip(
                published_stats["segment_label"],
                published_stats["radius_re"],
            )
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
        pad=5,
    )
    ax.tick_params(axis="y", right=False, left=True, direction="out", pad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)

    legend = ax.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor=PUBLISHED_EDGE,
                linewidth=1.0,
                label=PUBLISHED_LEGEND,
            ),
            Patch(
                facecolor=WORK_FILL,
                edgecolor=WORK_EDGE,
                linewidth=1.0,
                label=WORK_LEGEND,
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.008),
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
    transformed_centres = ax.transData.transform(
        np.column_stack([centres, np.zeros_like(centres)])
    )[:, 0]
    tick_display = ax.transData.transform(
        np.column_stack([ax.get_xticks(), np.zeros_like(centres)])
    )[:, 0]

    qa = {
        "backend": "python/matplotlib only",
        "archetype": "single-panel aligned overlay comparison",
        "group_count": 15,
        "group_centres_data": centres.tolist(),
        "published_box_centres_data": published_centres.tolist(),
        "this_work_box_centres_data": work_centres.tolist(),
        "published_offset_unique": np.unique(published_centres - centres).tolist(),
        "this_work_offset_unique": np.unique(work_centres - centres).tolist(),
        "max_tick_to_group_centre_error_px": float(
            np.max(np.abs(tick_display - transformed_centres))
        ),
        "published_box_width_data_units": PUBLISHED_BOX_WIDTH,
        "this_work_box_width_data_units": WORK_BOX_WIDTH,
        "max_adjacent_x_tick_overlap_px2": float(max(adjacent_overlap, default=0.0)),
        "minimum_x_tick_label_to_axis_gap_px": float(
            axes_bbox.y0 - max(bbox.y1 for bbox in tick_bboxes)
        ),
        "all_axis_tick_and_legend_text_inside_original_canvas": bool(all_text_inside),
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
            "outliers_retained": True,
        },
        "source_statistics_changed": False,
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
    save_args = {"bbox_inches": "tight", "pad_inches": PAD_IN, "facecolor": "white"}
    fig.savefig(paths["png"], dpi=DPI, **save_args)
    fig.savefig(paths["pdf"], **save_args)
    fig.savefig(paths["svg"], **save_args)
    fig.savefig(
        paths["tiff"],
        dpi=DPI,
        pil_kwargs={"compression": "tiff_lzw"},
        **save_args,
    )
    return paths


def raster_qa(path: Path) -> dict:
    with Image.open(path).convert("RGB") as image:
        arr = np.asarray(image)
        nonwhite = np.any(arr < 248, axis=2)
        ys, xs = np.where(nonwhite)
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
    stats: pd.DataFrame,
    outliers: pd.DataFrame,
    digitization: pd.DataFrame,
    paths: dict[str, Path],
    qa: dict,
) -> None:
    source_manifest = {
        "statistics_reused_unchanged": True,
        "source_files": {
            "box_statistics": str(SOURCE_STATS),
            "outliers": str(SOURCE_OUTLIERS),
            "official_digitization_pixels": str(SOURCE_DIGITIZATION),
            "source_provenance": str(SOURCE_PROVENANCE),
        },
        "source_hashes_sha256": {
            str(path): sha256(path)
            for path in [
                SOURCE_STATS,
                SOURCE_OUTLIERS,
                SOURCE_DIGITIZATION,
                SOURCE_PROVENANCE,
            ]
        },
        "row_counts": {
            "box_statistics": int(len(stats)),
            "outliers": int(len(outliers)),
            "official_digitization_pixels": int(len(digitization)),
        },
        "change_scope": (
            "Identical x centres, nested box widths, legend wording, z-order, "
            "and tight page bounds only; no statistic or sample changed."
        ),
    }
    SOURCE_MANIFEST_OUT.write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "figure_stem": STEM,
        "scientific_contract": {
            "residual_sign": "observed C/N0 minus model C/N0",
            "grouping": "same 15 GPS L1 IIR-M/IIIA operations",
            "statistics": "reused unchanged from the complete paired-boxplot source data",
            "published_elements": "digitized Parker et al. Figure 19 elements",
            "this_work_elements": "matched approximately 1-Hz final-model residual samples",
            "retraining": False,
        },
        "layout_contract": {
            "published_box_offset": 0.0,
            "this_work_box_offset": 0.0,
            "published_box_width": PUBLISHED_BOX_WIDTH,
            "this_work_box_width": WORK_BOX_WIDTH,
            "published_drawn_as": "wide hollow blue-grey box",
            "this_work_drawn_as": "narrow low-saturation teal filled box",
            "legend_labels": [PUBLISHED_LEGEND, WORK_LEGEND],
            "tight_export_pad_inches": PAD_IN,
        },
        "source_data_manifest": str(SOURCE_MANIFEST_OUT),
        "qa": qa,
        "source_hashes_sha256": source_manifest["source_hashes_sha256"],
        "output_hashes_sha256": {
            str(path): sha256(path)
            for path in [*paths.values(), SOURCE_MANIFEST_OUT]
            if path.exists()
        },
    }
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    QA_OUT.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    caption = """Extended Data Fig. X | Aligned published LuGRE and physics +
AI residual-model C/N0 discrepancies. For each of the same 15 GPS L1
IIR-M/IIIA operation segments, the two complete boxplots share an identical x
centre. Wide open blue-grey boxes show the digitized published LuGRE
link-budget simulation; narrower teal boxes show the physics + AI residual
model. Centre lines denote medians, boxes span Q1-Q3, whiskers and caps retain
their previously documented definitions, and symbols denote means. Published
whiskers, caps and visually separable outliers were digitized from Parker et
al. Figure 19 and are not row-level source data. This-work Tukey boxplots and
outliers were calculated from the matched approximately 1-Hz residual samples.
All statistics and samples are unchanged from the preceding complete-boxplot
version; only the aligned visual encoding and page layout differ.
"""
    CAPTION_OUT.write_text(caption, encoding="utf-8")


def main() -> None:
    stats, outliers, digitization = load_sources()
    fig, _, qa = make_figure(stats, outliers)
    paths = save_outputs(fig)
    plt.close(fig)
    qa.update(raster_qa(paths["png"]))
    qa["all_formats_exist"] = all(
        path.exists() and path.stat().st_size > 0 for path in paths.values()
    )
    qa["svg_editable_text_elements"] = paths["svg"].read_text(
        encoding="utf-8"
    ).count("<text")
    qa["source_box_statistics_rows"] = int(len(stats))
    qa["source_outlier_rows"] = int(len(outliers))

    write_deliverables(stats, outliers, digitization, paths, qa)
    print(f"PNG={paths['png']}")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
