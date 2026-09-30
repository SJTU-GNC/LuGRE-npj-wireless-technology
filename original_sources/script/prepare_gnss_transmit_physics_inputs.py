from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GALILEO_ZIP = ROOT / "data" / "external_reference" / "gnss_antenna" / "galileo" / "GRAP_metadata.zip"
GPS_DIR = ROOT / "data" / "external_reference" / "gnss_antenna" / "gps"
GALILEO_DIR = ROOT / "data" / "external_reference" / "gnss_antenna" / "galileo"
OUT_DIR = ROOT / "table" / "external_reference"


GRAP_SIGNALS = {
    "GAL_E1": {
        "file": "GRAP_File_E1_.xlsx",
        "eirp_sheet": "GRAP_EIRP_dBW_E1__",
        "upper_sheet": "GRAP_UB_dBW_E1__",
        "lower_sheet": "GRAP_LB_dBW_E1__    ",
    },
    "GAL_E5a": {
        "file": "GRAP_File_E5a_.xlsx",
        "eirp_sheet": "GRAP_EIRP_dBW_E5a_",
        "upper_sheet": "GRAP_UB_dBW_E5a_",
        "lower_sheet": "GRAP_LB_dBW_E5a_    ",
    },
}


def parse_grap_sheet(xlsx_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(xlsx_bytes), sheet_name=sheet_name, header=None, engine="openpyxl")
    coelev = pd.to_numeric(raw.iloc[1, 2:], errors="coerce")
    azimuth = pd.to_numeric(raw.iloc[2:, 1], errors="coerce")
    values = raw.iloc[2:, 2:].apply(pd.to_numeric, errors="coerce")
    values.index = azimuth
    values.columns = coelev
    values.index.name = "azimuth_deg"
    values.columns.name = "coelevation_deg"
    long = values.stack().rename("value_dbw").reset_index()
    long["azimuth_deg"] = long["azimuth_deg"].astype(int)
    long["coelevation_deg"] = long["coelevation_deg"].astype(int)
    return long


def build_grap_grid() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(GALILEO_ZIP) as zf:
        for signal, spec in GRAP_SIGNALS.items():
            xlsx_bytes = zf.read(spec["file"])
            eirp = parse_grap_sheet(xlsx_bytes, spec["eirp_sheet"]).rename(columns={"value_dbw": "eirp_dbw"})
            upper = parse_grap_sheet(xlsx_bytes, spec["upper_sheet"]).rename(columns={"value_dbw": "eirp_upper95_dbw"})
            lower = parse_grap_sheet(xlsx_bytes, spec["lower_sheet"]).rename(columns={"value_dbw": "eirp_lower95_dbw"})
            merged = eirp.merge(upper, on=["azimuth_deg", "coelevation_deg"], how="inner")
            merged = merged.merge(lower, on=["azimuth_deg", "coelevation_deg"], how="inner")
            merged.insert(0, "signal_name", signal)
            merged["source_file"] = spec["file"]
            merged["source_sheet"] = spec["eirp_sheet"]
            merged["coordinate_note"] = (
                "Galileo GRAP grid: azimuth 0..360 deg and co-elevation 0..90 deg; "
                "EIRP in dBW; coordinate mapping to observation body/yaw frame must be validated before row-wise use."
            )
            merged["reliability_tier"] = "official_galileo_grap_v1_0_reference_eirp"
            rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def existing(path: Path) -> str:
    return str(path.as_posix()) if path.exists() else ""


def build_inventory() -> pd.DataFrame:
    gps_iif = sorted(GPS_DIR.glob("SSC PA Final Assess*.pdf"))
    gps_iii = sorted(GPS_DIR.glob("GPS_III_SVN*_Directivity.zip"))
    rows = [
        {
            "system": "Galileo",
            "signal_name": "GAL_E1",
            "data_product": "Galileo Reference Antenna Pattern v1.0",
            "local_path": existing(GALILEO_ZIP),
            "quantity_type": "EIRP_tx",
            "unit": "dBW",
            "satellite_coverage": "Galileo FOC reference / constellation pattern, not satellite-specific",
            "angular_grid": "azimuth 0..360 deg, co-elevation 0..90 deg, 1 deg resolution",
            "coordinate_frame": "GRAP azimuth/co-elevation; needs mapping to Galileo body/yaw frame",
            "physics_status": "direct_physics_after_coordinate_ingest",
            "limitations": "Reference EIRP grid, not individual satellite pattern; coordinate mapping must be validated.",
            "recommended_use": "Use as Galileo E1 transmit EIRP lookup after yaw/body-frame convention is implemented.",
        },
        {
            "system": "Galileo",
            "signal_name": "GAL_E5a",
            "data_product": "Galileo Reference Antenna Pattern v1.0",
            "local_path": existing(GALILEO_ZIP),
            "quantity_type": "EIRP_tx",
            "unit": "dBW",
            "satellite_coverage": "Galileo FOC reference / constellation pattern, not satellite-specific",
            "angular_grid": "azimuth 0..360 deg, co-elevation 0..90 deg, 1 deg resolution",
            "coordinate_frame": "GRAP azimuth/co-elevation; needs mapping to Galileo body/yaw frame",
            "physics_status": "direct_physics_after_coordinate_ingest",
            "limitations": "Reference EIRP grid, not individual satellite pattern; coordinate mapping must be validated.",
            "recommended_use": "Use as Galileo E5a transmit EIRP lookup after yaw/body-frame convention is implemented.",
        },
        {
            "system": "GPS",
            "signal_name": "GPS_L1/GPS_L5",
            "data_product": "NAVCEN GPS IIR/IIR-M transmit antenna patterns",
            "local_path": existing(GPS_DIR / "GPS_IIR_IIR-M_LM.zip"),
            "quantity_type": "directivity_or_gain_pattern",
            "unit": "dB or dBi according to source table",
            "satellite_coverage": "GPS IIR/IIR-M where active SVN/block mapping is available",
            "angular_grid": "source-dependent theta/phi pattern",
            "coordinate_frame": "GPS body-frame convention reviewed in gnss_yaw_pattern_convention_report.md",
            "physics_status": "direct_physics_after_parser_and_svn_join",
            "limitations": "Parser normalization and active PRN/SVN/block join still required.",
            "recommended_use": "Use for GPS transmitter gain when row satellite/block/frequency is covered.",
        },
        {
            "system": "GPS",
            "signal_name": "GPS_L1/GPS_L5",
            "data_product": "NAVCEN GPS IIF SVN62-73 antenna pattern PDFs",
            "local_path": "|".join(p.as_posix() for p in gps_iif),
            "quantity_type": "directivity_or_gain_pattern",
            "unit": "dB or dBi according to source table",
            "satellite_coverage": "GPS IIF SVN62-73",
            "angular_grid": "source-dependent theta/phi pattern",
            "coordinate_frame": "GPS body-frame convention; parser normalization needed",
            "physics_status": "direct_physics_after_parser_and_svn_join",
            "limitations": "PDF extraction/parser required before row-wise lookup.",
            "recommended_use": "Use where LuGRE-observed GPS satellite maps to covered IIF SVN.",
        },
        {
            "system": "GPS",
            "signal_name": "GPS_L1/GPS_L5",
            "data_product": "NAVCEN GPS III SVN74-78 EC antenna pattern directivity",
            "local_path": "|".join(p.as_posix() for p in gps_iii),
            "quantity_type": "directivity_or_gain_pattern",
            "unit": "dB or dBi according to source table",
            "satellite_coverage": "GPS III SVN74-78",
            "angular_grid": "source-dependent theta/phi pattern",
            "coordinate_frame": "GPS III measurement frame; phi-to-body alignment quality flag required",
            "physics_status": "direct_physics_available_for_svn74_78_after_parser",
            "limitations": "SVN79/SVN80 directivity not public; GPS III absolute phi mapping remains quality-flagged.",
            "recommended_use": "Use for GPS III SVN74-78 direct link-budget subset; flag phi alignment quality.",
        },
        {
            "system": "GPS",
            "signal_name": "GPS_L1/GPS_L5",
            "data_product": "GPS III SVN79/SVN80 public directivity",
            "local_path": existing(GPS_DIR / "NAVCEN_GPS_Technical_References_downloaded_20260709.html"),
            "quantity_type": "missing_directivity",
            "unit": "not_available",
            "satellite_coverage": "SVN79/SVN80 directivity not found in public NAVCEN recheck",
            "angular_grid": "not_available",
            "coordinate_frame": "not_available",
            "physics_status": "proxy_only_until_public_directivity_available",
            "limitations": "APC/ISC release is not an antenna gain/directivity product.",
            "recommended_use": "Use block-level proxy/status flag only; do not treat APC/ISC as G_tx.",
        },
        {
            "system": "GPS/Galileo",
            "signal_name": "all",
            "data_product": "SSV service-angle and received-power definitions",
            "local_path": "data/external_reference/gnss_ssv",
            "quantity_type": "service_definition_not_gain",
            "unit": "degree / dBW as applicable",
            "satellite_coverage": "GPS/Galileo SSV interpretation",
            "angular_grid": "not a radiation pattern",
            "coordinate_frame": "service-volume geometry",
            "physics_status": "interpretation_status_only",
            "limitations": "SSV boundary is not electromagnetic half-power beamwidth or lobe boundary.",
            "recommended_use": "Use for classification/status annotation, not as G_tx.",
        },
    ]
    return pd.DataFrame(rows)


def write_interface() -> None:
    path = OUT_DIR / "gnss_transmit_physics_interface.md"
    text = """# GNSS Transmit Physics Interface

## Galileo GRAP grid

Machine-readable file:

```text
table/external_reference/galileo_grap_eirp_grid.csv
```

Fields:

- `signal_name`: `GAL_E1` or `GAL_E5a`.
- `azimuth_deg`: GRAP azimuth grid, integer degrees from 0 to 360.
- `coelevation_deg`: GRAP co-elevation grid, integer degrees from 0 to 90.
- `eirp_dbw`: reference EIRP in dBW.
- `eirp_upper95_dbw`, `eirp_lower95_dbw`: 95% upper/lower bounds in dBW.
- `coordinate_note`: reminder that row-wise use requires mapping the observation line-of-sight into the GRAP azimuth/co-elevation convention.

Recommended interpolation:

1. Convert the satellite-to-receiver line-of-sight into the Galileo transmitter antenna/body frame.
2. Convert the direction to GRAP azimuth and co-elevation.
3. Clamp or flag directions outside the 0..90 deg co-elevation grid.
4. Use bilinear interpolation for production; nearest-neighbour lookup is acceptable only for a first diagnostic.
5. Preserve uncertainty by optionally evaluating lower/upper 95% grids.

Important interpretation:

- GRAP provides Galileo FOC reference EIRP, not a LuGRE receive pattern.
- GRAP is not satellite-specific; per-satellite deviations remain residual/uncertainty unless a satellite-specific product is found.
- Do not use GPS SSV service-angle boundaries as Galileo radiation-pattern boundaries.

## GPS transmit products

GPS NAVCEN pattern files are physically usable only after:

1. the LuGRE-observed PRN is mapped to SVN/block at the observation epoch;
2. the source pattern is parsed into a consistent angular grid;
3. the nominal or actual yaw/body-frame convention is applied;
4. quality flags are set for noon/midnight turns, eclipse recovery, missing directivity, or ambiguous phi alignment.

SVN79/SVN80 directivity remains unavailable in the public recheck and must stay proxy/status-only.

## Residual-model rule

After `EIRP_tx` or `G_tx` has been physically encoded, the main AI residual model should not use unrestricted `system`, `signal_name`, `signal_id`, or `svid` categorical shortcuts. Such labels may be retained only in a diagnostic `category_assisted` model.
"""
    path.write_text(text, encoding="utf-8")


def write_update_report() -> None:
    path = ROOT / "table" / "missing_data_constellation_physics_update.md"
    text = """# Constellation Physics Update

Date: 2026-07-10

## Newly physicalized GPS/Galileo differences

- Galileo E1/E5a can now use the official Galileo Reference Antenna Pattern (GRAP) v1.0 as a transmitter-side EIRP grid after coordinate mapping is implemented.
- GPS transmit directivity remains available for several NAVCEN products, including IIR/IIR-M, IIF SVN62-73, and GPS III SVN74-78, after parser normalization and PRN/SVN/block joins.
- GPS/Galileo carrier-frequency differences are already represented through `FSPL(R,f)`.

## Still proxy or residual

- Galileo GRAP is a FOC reference/constellation EIRP model, not a satellite-specific flight pattern.
- GPS III SVN79/SVN80 public directivity was not found; use block-level proxy or row-status flags only.
- Actual yaw turns and eclipse recovery remain quality-flagged unless precise laws are obtained.
- LuGRE receive Fig. 3 remains a receive-gain outer-envelope proxy, not a full `G_rx(theta,phi,f)` table.
- RF PCO/PCV, Blue Ghost obstruction/CAD, local blockage, and flight hardware calibration remain AI-residual or uncertainty terms.

## Modeling implication

The next residual-learning run should use a constellation-aware physical baseline and include a `physics_strict` mode in which `system`, `signal_name`, `signal_id`, and `svid` are not direct AI categorical shortcuts. A `category_assisted` mode can be reported as a diagnostic upper bound.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = build_grap_grid()
    grid.to_csv(OUT_DIR / "galileo_grap_eirp_grid.csv", index=False)
    inv = build_inventory()
    inv.to_csv(OUT_DIR / "gnss_transmit_physics_inventory.csv", index=False)
    write_interface()
    write_update_report()
    print(f"GRAP rows: {len(grid)}")
    print(f"Inventory rows: {len(inv)}")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
