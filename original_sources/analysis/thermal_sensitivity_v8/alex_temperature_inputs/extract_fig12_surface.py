"""Extract public surface-temperature point centres from published Fig. 12.

No interpolation or extrapolation is performed, especially across the broken
March 5--14 axis or the separate LNA gap in the second panel.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import csv
import json

import pdfplumber

SOURCE = Path(r"D:\月球导航\data\external_reference\lugre_antenna\LuGRE_first_results_NAVIGATION_2026_official.pdf")
OUT = Path(__file__).resolve().parent
COLORS = {(0.467, 0.671, 0.259): "Receiver", (0.780, 0.090, 0.502): "HGA", (0.298, 0.580, 0.816): "LNA"}
PANELS = {
    "early_surface": (127.5874, 281.9974, datetime(2025, 3, 2, tzinfo=timezone.utc)),
    "late_surface": (322.1045, 476.5145, datetime(2025, 3, 14, tzinfo=timezone.utc)),
}
YTOP, YBOTTOM = 50.489, 191.329  # 80 and -80 degrees Celsius, visually verified.


def main():
    rows, duplicates, seen, legend = [], 0, set(), 0
    with pdfplumber.open(SOURCE) as doc:
        page = doc.pages[17]
        for object_index, shape in enumerate(page.curves):
            color = tuple(shape["non_stroking_color"] or ())
            if (color not in COLORS or len(shape["pts"]) != 5
                    or not (1.77 < shape["width"] < 1.80 and 1.77 < shape["height"] < 1.80)
                    or not (40 < shape["top"] < shape["bottom"] < 195)):
                continue
            x = (shape["x0"] + shape["x1"]) / 2
            y = (shape["top"] + shape["bottom"]) / 2
            # The three colored legend circles have this exact x coordinate.
            if abs(x - 343.7765) < 0.005 and 160 < y < 181:
                legend += 1
                continue
            panel = next((name for name, (xmin, xmax, _) in PANELS.items() if xmin <= x <= xmax), None)
            assert panel is not None, (x, y)
            key = (COLORS[color], panel, round(x, 6), round(y, 6))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            xmin, xmax, start = PANELS[panel]
            instant = start + timedelta(seconds=(x - xmin) / (xmax - xmin) * 3 * 86400)
            temp = 80 - (y - YTOP) / (YBOTTOM - YTOP) * 160
            rows.append({"source_segment": panel, "sensor": COLORS[color],
                         "timestamp_utc": instant.isoformat(timespec="microseconds"),
                         "temp_c": round(temp, 6), "temp_k": round(temp + 273.15, 6),
                         "source_pdf_x_pt": round(x, 6), "source_pdf_y_pt": round(y, 6),
                         "source_curve_object_index": object_index,
                         "source_pdf_page": 18, "source_figure": "12",
                         "source_kind": "publication_vector_figure_digitization",
                         "suggested_time_shift_sensitivity_seconds": 60,
                         "suggested_temperature_sensitivity_c": 0.5,
                         "trace_segment_id": "", "gap_before_seconds": ""})
    rows.sort(key=lambda r: (r["source_segment"], r["sensor"], r["timestamp_utc"]))
    segments = []
    for panel in PANELS:
        for sensor in COLORS.values():
            points = [r for r in rows if r["source_segment"] == panel and r["sensor"] == sensor]
            prev, group = None, 0
            for row in points:
                now = datetime.fromisoformat(row["timestamp_utc"])
                gap = None if prev is None else (now - prev).total_seconds()
                # Allow five seconds for inverse-coordinate quantization near
                # an exactly 30-minute gap; no values are interpolated.
                if gap is None or gap > 1805:
                    group += 1
                row["trace_segment_id"] = f"{panel}_{sensor}_{group}"
                row["gap_before_seconds"] = "" if gap is None else round(gap, 3)
                prev = now
            for group_id in dict.fromkeys(r["trace_segment_id"] for r in points):
                pp = [r for r in points if r["trace_segment_id"] == group_id]
                segments.append({"trace_segment_id": group_id, "n": len(pp),
                                 "first_utc": pp[0]["timestamp_utc"], "last_utc": pp[-1]["timestamp_utc"],
                                 "minimum_c": min(r["temp_c"] for r in pp),
                                 "maximum_c": max(r["temp_c"] for r in pp)})
    with (OUT / "surface_series.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source_pdf": str(SOURCE), "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "page": 18, "figure": 12, "n_extracted_unique_points": len(rows),
        "n_exact_duplicate_plot_points_removed": duplicates, "n_legend_markers_excluded": legend,
        "panel_calibration": {name: {"xleft_pdf_pt": v[0], "xright_pdf_pt": v[1],
                                       "left_utc": v[2].isoformat(), "duration_days": 3} for name, v in PANELS.items()},
        "y_calibration": {"top_pdf_pt": YTOP, "bottom_pdf_pt": YBOTTOM, "top_c": 80, "bottom_c": -80},
        "method": "Extract circle centres by sensor fill color; independently calibrated broken-axis panels; no resampling/interpolation.",
        "trace_segmentation_rule": "Split within each panel/sensor at gaps greater than 30 min plus 5 s graphics tolerance; analyst flag, not instrument metadata.",
        "segments": segments,
        "uncertainty_note": "Displayed digits preserve inverse coordinate calculation only. +/-0.5 C and +/-60 s are analyst-selected sensitivity perturbations, not measurement uncertainty or confidence intervals.",
        "limits": ["These are published plotted points, not raw telemetry or necessarily every recorded sample.",
                   "No interpolation across March 5--14 or across missing sensor segments is justified.",
                   "The LNA trace has an additional long gap in the second panel; other sensors must not substitute for it without a new explicit assumption.",
                   "UTC mapping comes from date ticks with 0.001-point graphics quantization, not a machine-readable timestamp series.",
                   "Sensor temperature is only a proxy for physical hardware temperature and is not effective antenna/system noise temperature."],
    }
    (OUT / "surface_series_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(rows), "duplicates_removed": duplicates, "legend_excluded": legend,
                      "segments": segments}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
