#!/usr/bin/env python3
"""Read the official GPS IIF antenna-pattern workbooks published by NAVCEN.

The workbooks store the useful arrays behind Excel chart sheets rather than in
the short ``Summary Data`` table.  LBSCC provides the measured 360-degree
azimuth grid through 23 degrees off boresight.  LBS GC provides four signed
far-field cuts; their negative halves supply the four opposite azimuth planes.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

import numpy as np


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CHART_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
CELL_RE = re.compile(r"([A-Z]+)(\d+)")
FORMULA_RE = re.compile(
    r"(?:'((?:[^']|'')+)'|([^!]+))!\$([A-Z]+)\$(\d+)"
    r"(?::\$([A-Z]+)\$(\d+))?"
)


def _norm_zip_path(base: str, target: str) -> str:
    if target.startswith("/xl/"):
        return target.lstrip("/")
    parts: list[str] = []
    for part in (PurePosixPath(base) / PurePosixPath(target)).parts:
        if part == "..":
            parts.pop()
        elif part not in ("", ".", "/"):
            parts.append(part)
    return "/".join(parts)


def _col_index(label: str) -> int:
    value = 0
    for char in label:
        value = value * 26 + ord(char) - ord("A") + 1
    return value


def _workbook_sheets(archive: zipfile.ZipFile) -> dict[str, str]:
    ns = {"m": MAIN_NS, "r": REL_NS}
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
    sheets: dict[str, str] = {}
    for sheet in workbook.find("m:sheets", ns):
        rid = sheet.attrib[f"{{{REL_NS}}}id"]
        sheets[sheet.attrib["name"]] = _norm_zip_path("xl", targets[rid])
    return sheets


def _relationship_targets(archive: zipfile.ZipFile, owner_path: str) -> dict[str, str]:
    owner = PurePosixPath(owner_path)
    rel_path = owner.parent / "_rels" / f"{owner.name}.rels"
    root = ET.fromstring(archive.read(rel_path.as_posix()))
    return {
        item.attrib["Id"]: _norm_zip_path(owner.parent.as_posix(), item.attrib["Target"])
        for item in root
    }


def _chart_formulas(
    archive: zipfile.ZipFile, sheets: dict[str, str], chart_sheet_name: str
) -> list[str]:
    chart_sheet_path = sheets[chart_sheet_name]
    chart_sheet = ET.fromstring(archive.read(chart_sheet_path))
    drawing = chart_sheet.find(f".//{{{MAIN_NS}}}drawing")
    if drawing is None:
        raise ValueError(f"Chart sheet {chart_sheet_name!r} has no drawing")
    chart_sheet_rels = _relationship_targets(archive, chart_sheet_path)
    drawing_path = chart_sheet_rels[drawing.attrib[f"{{{REL_NS}}}id"]]
    drawing_xml = ET.fromstring(archive.read(drawing_path))
    drawing_rels = _relationship_targets(archive, drawing_path)
    formulas: list[str] = []
    for chart in drawing_xml.findall(f".//{{{CHART_NS}}}chart"):
        chart_path = drawing_rels[chart.attrib[f"{{{REL_NS}}}id"]]
        chart_xml = ET.fromstring(archive.read(chart_path))
        formulas.extend(
            item.text
            for item in chart_xml.findall(f".//{{{CHART_NS}}}f")
            if item.text
        )
    return formulas


def _read_numeric_rectangle(
    archive: zipfile.ZipFile,
    sheet_path: str,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    with archive.open(sheet_path) as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != f"{{{MAIN_NS}}}c":
                continue
            match = CELL_RE.fullmatch(element.attrib.get("r", ""))
            if not match:
                element.clear()
                continue
            col = _col_index(match.group(1))
            row = int(match.group(2))
            if row_min <= row <= row_max and col_min <= col <= col_max:
                node = element.find(f"{{{MAIN_NS}}}v")
                if node is not None and node.text not in (None, ""):
                    try:
                        values[(row, col)] = float(node.text)
                    except ValueError:
                        pass
            element.clear()
    return values


def _extract_chart_matrix(
    archive: zipfile.ZipFile, sheets: dict[str, str], chart_sheet_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    references = []
    for formula in _chart_formulas(archive, sheets, chart_sheet_name):
        match = FORMULA_RE.fullmatch(formula)
        if not match or match.group(5) is None:
            continue
        sheet_name = (match.group(1) or match.group(2)).replace("''", "'")
        references.append(
            (
                sheet_name,
                _col_index(match.group(3)),
                int(match.group(4)),
                _col_index(match.group(5)),
                int(match.group(6)),
            )
        )
    y_references = []
    seen = set()
    for reference in references:
        sheet_name, col0, row0, col1, row1 = reference
        key = (sheet_name, col0, row0, col1, row1)
        if col0 == col1 and col0 > 1 and row1 > row0 and key not in seen:
            y_references.append(reference)
            seen.add(key)
    if not y_references:
        raise ValueError(f"No numeric series found in chart sheet {chart_sheet_name!r}")
    sheet_name = y_references[0][0]
    row0 = y_references[0][2]
    row1 = y_references[0][4]
    columns = [reference[1] for reference in y_references]
    cells = _read_numeric_rectangle(
        archive,
        sheets[sheet_name],
        row0 - 1,
        row1,
        1,
        max(columns),
    )
    x = np.asarray([cells.get((row, 1), np.nan) for row in range(row0, row1 + 1)])
    labels = np.asarray([cells.get((row0 - 1, col), np.nan) for col in columns])
    data = np.asarray(
        [[cells.get((row, col), np.nan) for col in columns] for row in range(row0, row1 + 1)]
    )
    valid_rows = np.isfinite(x) & np.isfinite(data).all(axis=1)
    valid_columns = np.isfinite(labels) & np.isfinite(data[valid_rows]).all(axis=0)
    return x[valid_rows], labels[valid_columns], data[valid_rows][:, valid_columns]


def _periodic_resample(phi: np.ndarray, values: np.ndarray, target_phi: np.ndarray) -> np.ndarray:
    order = np.argsort(np.mod(phi, 360.0))
    source_phi = np.mod(phi[order], 360.0)
    source_values = values[order]
    unique_phi, unique_index = np.unique(np.round(source_phi, 6), return_index=True)
    source_values = source_values[unique_index]
    return np.interp(
        np.mod(target_phi, 360.0),
        np.r_[unique_phi, unique_phi[0] + 360.0],
        np.r_[source_values, source_values[0]],
    )


def _build_band_pattern(path: Path, band: str, far_template: dict | None = None) -> dict:
    with zipfile.ZipFile(path) as archive:
        sheets = _workbook_sheets(archive)
        main_phi, main_theta, main_values = _extract_chart_matrix(archive, sheets, band)
        has_satellite_far_field = f"{band} GC" in sheets
        if has_satellite_far_field:
            far_signed_theta, far_phi, far_values = _extract_chart_matrix(
                archive, sheets, f"{band} GC"
            )

    target_phi = np.arange(0.0, 360.0, 1.0)
    main_gain = np.vstack(
        [_periodic_resample(main_phi, main_values[:, index], target_phi) for index in range(len(main_theta))]
    )

    far_theta = np.arange(24.0, 91.0, 1.0)
    if has_satellite_far_field:
        far_samples: dict[int, dict[float, list[float]]] = {}
        for row, signed_theta in enumerate(far_signed_theta):
            theta = int(round(abs(float(signed_theta))))
            if not 0 <= theta <= 90:
                continue
            for col, base_phi in enumerate(far_phi):
                phi = float(base_phi) if signed_theta >= 0 else float(base_phi + 180.0)
                far_samples.setdefault(theta, {}).setdefault(phi % 360.0, []).append(
                    float(far_values[row, col])
                )
        far_gain_rows = []
        available_theta = np.asarray(sorted(far_samples), dtype=float)
        for theta in far_theta:
            nearest = int(available_theta[np.argmin(np.abs(available_theta - theta))])
            plane_samples = far_samples[nearest]
            plane_phi = np.asarray(sorted(plane_samples), dtype=float)
            plane_gain = np.asarray(
                [np.mean(plane_samples[value]) for value in plane_phi], dtype=float
            )
            far_gain_rows.append(_periodic_resample(plane_phi, plane_gain, target_phi))
        far_gain = np.asarray(far_gain_rows)
        far_source = "official_LBS_GC_8plane_periodic_reconstruction_theta_24_90deg"
        far_coverage = "official_satellite_specific_8plane_interpolation_24_90deg"
    else:
        if far_template is None:
            raise ValueError(f"{path.name} has no {band} GC chart and no IIF far-field template")
        boresight = float(np.nanmedian(main_gain[np.isclose(main_theta, 0.0)]))
        delta = boresight - float(far_template["boresight_reference_db"])
        far_gain = np.asarray(far_template["gain"], dtype=float) + delta
        far_source = "official_IIF_block_median_far_field_scaled_to_satellite_boresight"
        far_coverage = "official_satellite_specific_2D_0_23deg;official_IIF_block_median_proxy_24_90deg"

    theta = np.r_[main_theta.astype(float), far_theta]
    gain = np.vstack([main_gain, far_gain])
    order = np.argsort(theta)
    theta = theta[order]
    gain = gain[order]
    unique_theta, unique_index = np.unique(np.round(theta, 6), return_index=True)
    return {
        "theta": unique_theta,
        "phi": target_phi,
        "gain": gain[unique_index],
        "source": (
            f"{path.as_posix()}#{band}:official_LBSCC_2D_theta_0_23deg+"
            f"{far_source}"
        ),
        "block_pattern_family": "GPS_IIF_official_hybrid_2D",
        "pattern_coverage": far_coverage,
    }


def parse_iif_patterns(gps_dir: Path) -> dict[str, dict[int, dict]]:
    """Return band- and SVN-specific callable 2-D GPS IIF gain grids."""
    patterns: dict[str, dict[int, dict]] = {"GPS_L1": {}, "GPS_L5": {}}
    files: dict[int, Path] = {}
    has_far_field: dict[tuple[int, str], bool] = {}
    for path in sorted(gps_dir.glob("SSC PA Final Assess*.xlsx")):
        match = re.search(r"SVN(\d+)", path.name)
        if not match:
            continue
        svn = int(match.group(1))
        files[svn] = path
        with zipfile.ZipFile(path) as archive:
            sheet_names = _workbook_sheets(archive)
        for chart_name in ("L1", "L5"):
            has_far_field[(svn, chart_name)] = f"{chart_name} GC" in sheet_names

    for signal, chart_name in (("GPS_L1", "L1"), ("GPS_L5", "L5")):
        direct: dict[int, dict] = {}
        for svn, path in files.items():
            if has_far_field[(svn, chart_name)]:
                direct[svn] = _build_band_pattern(path, chart_name)
        if not direct:
            raise ValueError(f"No GPS IIF workbooks contain {chart_name} far-field cuts")
        far_gain = np.nanmedian(
            np.stack(
                [
                    pattern["gain"][pattern["theta"] >= 24.0]
                    for pattern in direct.values()
                ]
            ),
            axis=0,
        )
        boresight_reference = float(
            np.nanmedian(
                [
                    np.nanmedian(pattern["gain"][np.isclose(pattern["theta"], 0.0)])
                    for pattern in direct.values()
                ]
            )
        )
        far_template = {
            "gain": far_gain,
            "boresight_reference_db": boresight_reference,
        }
        patterns[signal].update(direct)
        for svn, path in files.items():
            if svn not in patterns[signal]:
                patterns[signal][svn] = _build_band_pattern(path, chart_name, far_template)
    return patterns


if __name__ == "__main__":
    import sys

    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/external_reference/gnss_antenna/gps"
    )
    output = parse_iif_patterns(directory)
    for signal, items in output.items():
        for svn, pattern in sorted(items.items()):
            print(signal, svn, pattern["gain"].shape, pattern["source"])
