#!/usr/bin/env python3
"""Directly re-export the locked interleaved figure with compact page edges.

The left-top-trimmed source builds every artist unchanged. This wrapper changes
only the Matplotlib export page: the existing top and right edges are retained,
while the left and bottom visible-ink margins are reduced to 16 px at 600 dpi.
No saved raster is cropped or used to reconstruct the figure.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str((ROOT / "script").resolve()))

import matplotlib as mpl
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_interleaved_details_wide_compact_left_top_trimmed as source


FIGURE_DIR = ROOT / "figure" / "paper_draft_v2"
REFERENCE_STEM = source.STEM
STEM = f"{REFERENCE_STEM}_all_sides_trimmed"
OUTPUT_BASE = FIGURE_DIR / STEM
PROVENANCE_PATH = FIGURE_DIR / f"{STEM}_provenance.json"
REFERENCE_SCRIPT = (
    ROOT
    / "script"
    / (
        "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_"
        "interleaved_details_wide_compact_left_top_trimmed.py"
    )
)
REFERENCE_PNG = FIGURE_DIR / f"{REFERENCE_STEM}.png"
REFERENCE_SVG = FIGURE_DIR / f"{REFERENCE_STEM}.svg"
REFERENCE_PROVENANCE = FIGURE_DIR / f"{REFERENCE_STEM}_provenance.json"
REFERENCE_OUTPUTS = [
    FIGURE_DIR / f"{REFERENCE_STEM}.{suffix}"
    for suffix in ("png", "pdf", "svg", "tiff")
]
REFERENCE_PROTECTED = [
    REFERENCE_SCRIPT,
    *REFERENCE_OUTPUTS,
    REFERENCE_PROVENANCE,
]

RASTER_DPI = 600
VISIBLE_INK_THRESHOLD = 250
TARGET_LEFT_MARGIN_PX = 16
TARGET_BOTTOM_MARGIN_PX = 18
ALLOWED_TRIMMED_MARGIN_RANGE_PX = (14, 18)


def _read_rgb(path: Path) -> tuple[np.ndarray, tuple[float, float] | None]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
        dpi = image.info.get("dpi")
    return array, dpi


def _visible_ink_bbox(array: np.ndarray) -> dict[str, int]:
    mask = array.min(axis=2) < VISIBLE_INK_THRESHOLD
    y_indices, x_indices = np.nonzero(mask)
    if not len(x_indices):
        raise RuntimeError("No visible ink found in the immediate reference")
    return {
        "x0": int(x_indices.min()),
        "y0": int(y_indices.min()),
        "x1": int(x_indices.max()),
        "y1": int(y_indices.max()),
    }


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, str]]:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Immediate-reference file is missing: {missing}")
    return {
        str(path.resolve()): (int(path.stat().st_size), source.sha256_file(path))
        for path in paths
    }


REFERENCE_ARRAY, REFERENCE_DPI = _read_rgb(REFERENCE_PNG)
REFERENCE_HEIGHT_PX, REFERENCE_WIDTH_PX = REFERENCE_ARRAY.shape[:2]
REFERENCE_INK_BBOX = _visible_ink_bbox(REFERENCE_ARRAY)
REFERENCE_MARGINS_PX = {
    "left": REFERENCE_INK_BBOX["x0"],
    "right": REFERENCE_WIDTH_PX - 1 - REFERENCE_INK_BBOX["x1"],
    "top": REFERENCE_INK_BBOX["y0"],
    "bottom": REFERENCE_HEIGHT_PX - 1 - REFERENCE_INK_BBOX["y1"],
}

# Preserve the complete source top and right boundaries. Reduce only the left
# and bottom export boundaries around the unchanged visible content.
EXPORT_LEFT_PX = REFERENCE_INK_BBOX["x0"] - TARGET_LEFT_MARGIN_PX
EXPORT_RIGHT_EXCLUSIVE_PX = REFERENCE_WIDTH_PX
EXPORT_TOP_PX = 0
EXPORT_BOTTOM_EXCLUSIVE_PX = (
    REFERENCE_INK_BBOX["y1"] + 1 + TARGET_BOTTOM_MARGIN_PX
)
if min(EXPORT_LEFT_PX, EXPORT_TOP_PX) < 0:
    raise RuntimeError("Requested compact export extends beyond the source canvas")
if EXPORT_BOTTOM_EXCLUSIVE_PX > REFERENCE_HEIGHT_PX:
    raise RuntimeError("Requested compact export extends beyond the source canvas")

TARGET_WIDTH_PX = EXPORT_RIGHT_EXCLUSIVE_PX - EXPORT_LEFT_PX
TARGET_HEIGHT_PX = EXPORT_BOTTOM_EXCLUSIVE_PX - EXPORT_TOP_PX

# savefig's explicit Bbox is expressed in inches with a bottom-left origin.
EXPORT_BOTTOM_DISPLAY_PX = (
    REFERENCE_HEIGHT_PX - EXPORT_BOTTOM_EXCLUSIVE_PX
)
EXPORT_TOP_DISPLAY_PX = REFERENCE_HEIGHT_PX - EXPORT_TOP_PX
EXPORT_BBOX_INCHES = Bbox.from_extents(
    EXPORT_LEFT_PX / RASTER_DPI,
    EXPORT_BOTTOM_DISPLAY_PX / RASTER_DPI,
    EXPORT_RIGHT_EXCLUSIVE_PX / RASTER_DPI,
    EXPORT_TOP_DISPLAY_PX / RASTER_DPI,
)

_ORIGINAL_WRITE_PROVENANCE = source.write_provenance
_REFERENCE_BEFORE: dict[str, tuple[int, str]] = {}
_EXPORT_TEXT_QA: dict[str, object] = {}


def _svg_text_counter(path: Path) -> Counter[str]:
    root = ElementTree.parse(path).getroot()
    return Counter(
        "".join(element.itertext()).strip()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "text"
        and "".join(element.itertext()).strip()
    )


def _rendered_text_boundary_qa(fig: mpl.figure.Figure) -> dict[str, object]:
    """Verify all visible text against the direct export page at 600 dpi."""
    fig.set_dpi(RASTER_DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    export_edges = {
        "left": float(EXPORT_LEFT_PX),
        "right": float(EXPORT_RIGHT_EXCLUSIVE_PX),
        "bottom": float(EXPORT_BOTTOM_DISPLAY_PX),
        "top": float(EXPORT_TOP_DISPLAY_PX),
    }
    minimum = {
        "left": float("inf"),
        "right": float("inf"),
        "bottom": float("inf"),
        "top": float("inf"),
    }
    outside: list[dict[str, object]] = []
    count = 0
    for artist in fig.findobj(match=lambda item: isinstance(item, Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        count += 1
        bbox = artist.get_window_extent(renderer)
        clearances = {
            "left": float(bbox.x0 - export_edges["left"]),
            "right": float(export_edges["right"] - bbox.x1),
            "bottom": float(bbox.y0 - export_edges["bottom"]),
            "top": float(export_edges["top"] - bbox.y1),
        }
        for side, value in clearances.items():
            minimum[side] = min(minimum[side], value)
        if min(clearances.values()) < -0.25:
            outside.append(
                {
                    "text": artist.get_text(),
                    "gid": artist.get_gid(),
                    "clearance_to_export_edge_px": clearances,
                }
            )
    if outside:
        raise RuntimeError(
            "Visible text extends outside the compact export page: "
            + json.dumps(outside, ensure_ascii=False)
        )
    return {
        "visible_text_artist_count": count,
        "visible_text_outside_export_bbox": [],
        "minimum_text_bbox_clearance_to_export_edge_px": minimum,
        "all_visible_text_bboxes_inside_export_bbox": True,
    }


def save_outputs(fig: mpl.figure.Figure) -> list[Path]:
    """Render each format directly through the same explicit page Bbox."""
    global _EXPORT_TEXT_QA
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _EXPORT_TEXT_QA = _rendered_text_boundary_qa(fig)
    outputs: list[Path] = []
    formats = [
        ("png", {"dpi": RASTER_DPI}),
        ("pdf", {}),
        ("svg", {}),
        (
            "tiff",
            {"dpi": RASTER_DPI, "pil_kwargs": {"compression": "tiff_lzw"}},
        ),
    ]
    for suffix, kwargs in formats:
        path = OUTPUT_BASE.with_suffix(f".{suffix}")
        fig.savefig(
            path,
            bbox_inches=EXPORT_BBOX_INCHES,
            pad_inches=0,
            facecolor="white",
            edgecolor="white",
            transparent=False,
            **kwargs,
        )
        outputs.append(path)
    return outputs


def _pixel_equivalence(
    current: np.ndarray,
    expected: np.ndarray,
) -> dict[str, object]:
    if current.shape != expected.shape:
        raise RuntimeError(
            f"Export/reference-window shape mismatch: "
            f"{current.shape} vs {expected.shape}"
        )
    absolute = np.abs(current.astype(np.int16) - expected.astype(np.int16))
    changed = np.any(absolute != 0, axis=2)
    changed_fraction = float(changed.mean())
    mean_absolute_channel_delta = float(absolute.mean())
    maximum_absolute_channel_delta = int(absolute.max())
    if (
        changed_fraction > 3.0e-5
        or mean_absolute_channel_delta > 1.0e-4
        or maximum_absolute_channel_delta > 64
    ):
        raise RuntimeError(
            "Compact export differs materially from the corresponding "
            "immediate-reference pixel window"
        )
    changed_y, changed_x = np.nonzero(changed)
    return {
        "reference_window_top_left_exclusive_px": {
            "left": EXPORT_LEFT_PX,
            "right_exclusive": EXPORT_RIGHT_EXCLUSIVE_PX,
            "top": EXPORT_TOP_PX,
            "bottom_exclusive": EXPORT_BOTTOM_EXCLUSIVE_PX,
        },
        "changed_pixel_count": int(changed.sum()),
        "changed_pixel_fraction": changed_fraction,
        "mean_absolute_channel_delta": mean_absolute_channel_delta,
        "maximum_absolute_channel_delta": maximum_absolute_channel_delta,
        "difference_bbox_px": (
            [
                int(changed_x.min()),
                int(changed_y.min()),
                int(changed_x.max()),
                int(changed_y.max()),
            ]
            if changed.any()
            else None
        ),
        "strict_tolerance": {
            "maximum_changed_pixel_fraction": 3.0e-5,
            "maximum_mean_absolute_channel_delta": 1.0e-4,
            "maximum_absolute_channel_delta": 64,
        },
        "difference_interpretation": (
            "sub-threshold raster antialiasing from direct Matplotlib "
            "page-bbox re-rendering; vector text and artist geometry unchanged"
        ),
        "content_equivalent_within_strict_tolerance": True,
    }


def raster_qa(path: Path) -> dict[str, object]:
    current, dpi = _read_rgb(path)
    height, width = current.shape[:2]
    if (width, height) != (TARGET_WIDTH_PX, TARGET_HEIGHT_PX):
        raise RuntimeError(
            f"Compact PNG canvas mismatch: {(width, height)} vs "
            f"{(TARGET_WIDTH_PX, TARGET_HEIGHT_PX)}"
        )
    current_bbox = _visible_ink_bbox(current)
    new_margins = {
        "left": current_bbox["x0"],
        "right": width - 1 - current_bbox["x1"],
        "top": current_bbox["y0"],
        "bottom": height - 1 - current_bbox["y1"],
    }
    for side in ("left", "bottom"):
        value = new_margins[side]
        if not (
            ALLOWED_TRIMMED_MARGIN_RANGE_PX[0]
            <= value
            <= ALLOWED_TRIMMED_MARGIN_RANGE_PX[1]
        ):
            raise RuntimeError(
                f"{side} visible-ink margin is outside the 14-18 px contract"
            )
    if new_margins["top"] != REFERENCE_MARGINS_PX["top"]:
        raise RuntimeError("The top visible-ink boundary changed")
    if new_margins["right"] != REFERENCE_MARGINS_PX["right"]:
        raise RuntimeError("The right visible-ink boundary changed")
    expected = REFERENCE_ARRAY[
        EXPORT_TOP_PX:EXPORT_BOTTOM_EXCLUSIVE_PX,
        EXPORT_LEFT_PX:EXPORT_RIGHT_EXCLUSIVE_PX,
    ]
    equivalence = _pixel_equivalence(current, expected)
    nonwhite = current.min(axis=2) < VISIBLE_INK_THRESHOLD
    return {
        "png_width_px": int(width),
        "png_height_px": int(height),
        "png_dpi": [float(value) for value in dpi] if dpi else None,
        "width_in_at_600dpi": width / RASTER_DPI,
        "height_in_at_600dpi": height / RASTER_DPI,
        "width_mm_at_600dpi": width / RASTER_DPI * 25.4,
        "height_mm_at_600dpi": height / RASTER_DPI * 25.4,
        "nonwhite_fraction": float(nonwhite.mean()),
        "rendered_content_bbox_px": current_bbox,
        "raster_margin_comparison": {
            "visibility_threshold": "minimum RGB channel < 250",
            "reference_margins_px": REFERENCE_MARGINS_PX,
            "new_margins_px": new_margins,
            "left_blank_reduction_px": (
                REFERENCE_MARGINS_PX["left"] - new_margins["left"]
            ),
            "bottom_blank_reduction_px": (
                REFERENCE_MARGINS_PX["bottom"] - new_margins["bottom"]
            ),
            "top_boundary_preserved_exactly": True,
            "right_boundary_preserved_exactly": True,
            "trimmed_margin_target_range_px": list(
                ALLOWED_TRIMMED_MARGIN_RANGE_PX
            ),
        },
        "direct_matplotlib_export_equivalence": equivalence,
        "corner_pixels_rgb": [
            current[0, 0].astype(int).tolist(),
            current[0, -1].astype(int).tolist(),
            current[-1, 0].astype(int).tolist(),
            current[-1, -1].astype(int).tolist(),
        ],
        **_EXPORT_TEXT_QA,
    }


def vector_qa() -> dict[str, object]:
    svg_path = OUTPUT_BASE.with_suffix(".svg")
    svg = svg_path.read_text(encoding="utf-8")
    if "<text" not in svg:
        raise RuntimeError("SVG text is not editable")
    missing_markers = [
        marker
        for marker in source.layout_base.DETAIL_PANEL_MARKERS
        if marker not in svg
    ]
    if missing_markers:
        raise RuntimeError(f"Detail markers missing from SVG: {missing_markers}")
    if "merged-phase-observation-legend" not in svg:
        raise RuntimeError("Merged legend group is missing from SVG")
    current_text = _svg_text_counter(svg_path)
    reference_text = _svg_text_counter(REFERENCE_SVG)
    if current_text != reference_text:
        raise RuntimeError("SVG visible-text inventory changed")
    return {
        "svg_editable_text": True,
        "all_eight_detail_markers_present": True,
        "merged_legend_present": True,
        "svg_text_content_matches_immediate_reference": True,
        "svg_text_element_count": int(sum(current_text.values())),
        "vector_page_reframed_only": True,
    }


def write_provenance(*args, **kwargs) -> None:
    _ORIGINAL_WRITE_PROVENANCE(*args, **kwargs)
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    qa = provenance["visual_qa"]
    provenance["generated_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["output_stem"] = STEM
    provenance["backend"] = "Python/matplotlib only"
    provenance["figure_archetype"] = (
        "quantitative grid with interleaved detail rows"
    )
    provenance["core_conclusion"] = (
        "Each full-mission four-band C/N0 trend is paired with the common "
        "maximum-model-output and maximum-observed-C/N0 24-hour windows."
    )
    provenance["scientific_content_policy"] = {
        "model_inference_run": False,
        "satellite_reselection": False,
        "window_reselection_in_current_wrapper": False,
        "upstream_window_reselection": True,
        "upstream_window_selection_changes_values": False,
        "smoothing_or_clipping_added": False,
        "data_values_changed": False,
        "background_states_changed_in_current_wrapper": False,
        "upstream_background_change": (
            "antenna back-hemisphere shading uses transmitter-pattern "
            "unsupported geometry only"
        ),
        "typography_changed": False,
        "axes_or_artist_geometry_changed": False,
        "line_or_marker_style_changed_in_current_wrapper": False,
        "detail_line_or_marker_style_inherited_from_reference": True,
        "top_boundary_changed": False,
        "right_boundary_changed": False,
        "left_export_boundary_changed": True,
        "bottom_export_boundary_changed": True,
        "post_render_raster_crop_used": False,
        "current_wrapper_change": (
            "direct Matplotlib export-page reframing at the left and bottom"
        ),
    }
    provenance.pop("immediate_reference_left_trimmed_figure", None)
    provenance["immediate_reference_left_top_trimmed_figure"] = {
        **source.file_record(REFERENCE_PNG),
        "width_px": REFERENCE_WIDTH_PX,
        "height_px": REFERENCE_HEIGHT_PX,
        "visible_ink_bbox_px": REFERENCE_INK_BBOX,
        "visible_ink_margins_px": REFERENCE_MARGINS_PX,
    }
    provenance["figure_contract"] = {
        "core_conclusion": provenance["core_conclusion"],
        "evidence_chain": [
            "panels a-d preserve the four full-mission signal trends",
            "odd detail panels use the common maximum finite-model-output window",
            "even detail panels use the common maximum observed-C/N0 window",
            "main-panel boxes and detail axes use identical UTC limits",
            "the compact export wrapper changes no data or artist geometry",
        ],
        "archetype": provenance["figure_archetype"],
        "reviewer_risk_controls": [
            "the left-top-trimmed figure is the sole immediate reference",
            "all artists are rebuilt by the locked Python source",
            "only an explicit Matplotlib export bbox changes",
            "the top and right page boundaries are preserved exactly",
            "all visible text bboxes remain inside the compact page",
            "the SVG visible-text inventory is unchanged",
            "the PNG is pixel-equivalent to the corresponding reference window",
            "all immediate-reference files remain hash-identical",
        ],
        "export": "PNG/PDF/SVG/TIFF; 600 dpi raster; editable vector text",
    }
    provenance["export_reframing"] = {
        "method": (
            "direct Matplotlib savefig with explicit bbox_inches; "
            "no post-render raster crop"
        ),
        "source_canvas_px": [REFERENCE_WIDTH_PX, REFERENCE_HEIGHT_PX],
        "source_visible_ink_bbox_px": REFERENCE_INK_BBOX,
        "source_visible_ink_margins_px": REFERENCE_MARGINS_PX,
        "export_window_source_top_left_px": {
            "left": EXPORT_LEFT_PX,
            "right_exclusive": EXPORT_RIGHT_EXCLUSIVE_PX,
            "top": EXPORT_TOP_PX,
            "bottom_exclusive": EXPORT_BOTTOM_EXCLUSIVE_PX,
        },
        "matplotlib_bbox_inches_bottom_left": [
            float(EXPORT_BBOX_INCHES.x0),
            float(EXPORT_BBOX_INCHES.y0),
            float(EXPORT_BBOX_INCHES.x1),
            float(EXPORT_BBOX_INCHES.y1),
        ],
        "output_canvas_px": [TARGET_WIDTH_PX, TARGET_HEIGHT_PX],
        "output_visible_ink_margins_px": qa[
            "raster_margin_comparison"
        ]["new_margins_px"],
        "top_and_right_source_boundaries_preserved": True,
    }
    layout = provenance["layout"]
    layout.update(
        {
            "source_construction_canvas_width_px": REFERENCE_WIDTH_PX,
            "source_construction_canvas_height_px": REFERENCE_HEIGHT_PX,
            "target_png_width_px": TARGET_WIDTH_PX,
            "target_png_height_px": TARGET_HEIGHT_PX,
            "export_width_in": TARGET_WIDTH_PX / RASTER_DPI,
            "export_height_in": TARGET_HEIGHT_PX / RASTER_DPI,
            "export_width_mm": TARGET_WIDTH_PX / RASTER_DPI * 25.4,
            "export_height_mm": TARGET_HEIGHT_PX / RASTER_DPI * 25.4,
            "internal_layout_changed": False,
            "top_page_boundary_changed": False,
            "right_page_boundary_changed": False,
            "left_page_trim_px": EXPORT_LEFT_PX,
            "bottom_page_trim_px": (
                REFERENCE_HEIGHT_PX - EXPORT_BOTTOM_EXCLUSIVE_PX
            ),
        }
    )
    provenance["plot_script"] = source.file_record(Path(__file__))
    provenance["outputs"] = [
        source.file_record(OUTPUT_BASE.with_suffix(f".{suffix}"))
        for suffix in ("png", "pdf", "svg", "tiff")
    ]
    provenance["immediate_reference_protection"] = [
        {
            "path": str(Path(path).resolve().relative_to(ROOT.resolve())).replace(
                "\\", "/"
            ),
            "bytes_before": int(size_hash[0]),
            "sha256_before": size_hash[1],
            "sha256_after": source.sha256_file(Path(path)),
            "unchanged": source.sha256_file(Path(path)) == size_hash[1],
        }
        for path, size_hash in _REFERENCE_BEFORE.items()
    ]
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def configure_source_module() -> None:
    source.STEM = STEM
    source.OUTPUT_BASE = OUTPUT_BASE
    source.PROVENANCE_PATH = PROVENANCE_PATH
    source.save_outputs = save_outputs
    source.raster_qa = raster_qa
    source.vector_qa = vector_qa
    source.write_provenance = write_provenance


def main() -> None:
    global _REFERENCE_BEFORE
    _REFERENCE_BEFORE = _snapshot(REFERENCE_PROTECTED)
    configure_source_module()
    source.main()
    reference_after = _snapshot(REFERENCE_PROTECTED)
    if _REFERENCE_BEFORE != reference_after:
        raise RuntimeError("An immediate-reference file changed")


if __name__ == "__main__":
    main()
