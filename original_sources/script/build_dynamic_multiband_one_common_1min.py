#!/usr/bin/env python3
"""Build full-mission C/N0 physics features on reconstructed LuGRE dynamics."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "runtime_cache" / "python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "script"))

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import build_plot_wgc_multiband_cn0 as builder
import build_lugre_dynamic_receiver_trajectory_1min as dynamic_receiver


CADENCE = "1min"
OUTPUT_DIR = ROOT / "table" / "algorithm" / "cn0_dynamic_multiband_one_common_1min"
OUTPUT_PATH = OUTPUT_DIR / "cn0_dynamic_multiband_one_common_1min_predictions.csv"
SELECTION_PATH = OUTPUT_DIR / "cn0_dynamic_multiband_one_common_1min_selection.csv"
SELECTED = {
    "GPS_L1": [11],
    "GPS_L5": [11],
    "GAL_E1": [34],
    "GAL_E5a": [34],
}


def interpolate_sp3_cubic(
    receiver: pd.DataFrame, sp3: pd.DataFrame, sat_ids: list[str]
) -> pd.DataFrame:
    target = receiver["utc"].map(lambda t: t.timestamp()).to_numpy(float)
    rows: list[pd.DataFrame] = []
    for sat_id in sat_ids:
        source = (
            sp3[sp3["sat_id"].eq(sat_id)]
            .groupby("utc", as_index=False)[
                ["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]
            ]
            .mean()
            .sort_values("utc")
        )
        if len(source) < 4:
            continue
        source_seconds = source["utc"].map(lambda t: t.timestamp()).to_numpy(float)
        inside = (target >= source_seconds.min()) & (target <= source_seconds.max())
        out = receiver.loc[inside].copy()
        out["sat_id"] = sat_id
        out["system"] = sat_id[0]
        out["svid"] = int(sat_id[1:])
        for column in ["sat_x_ecef_km", "sat_y_ecef_km", "sat_z_ecef_km"]:
            spline = CubicSpline(source_seconds, source[column].to_numpy(float))
            out[column] = spline(target[inside])
        rows.append(out)
    geometry = pd.concat(rows, ignore_index=True)
    geometry[["sat_x_eci_km", "sat_y_eci_km", "sat_z_eci_km"]] = builder.ecef_to_eci(
        geometry
    )
    return geometry


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not dynamic_receiver.OUTPUT.exists():
        dynamic_receiver.build()

    observations = pd.read_csv(builder.OBS, low_memory=False)
    _, mapping, patterns = builder.available_satellites(observations)
    satellite_ids = sorted(
        {
            f"{builder.SIGNAL_SYSTEM[signal]}{svid:02d}"
            for signal, svids in SELECTED.items()
            for svid in svids
        }
    )
    receiver = pd.read_csv(dynamic_receiver.OUTPUT, low_memory=False)
    receiver["utc"] = pd.to_datetime(receiver["utc"], utc=True, errors="coerce")
    receiver = receiver.dropna(
        subset=["utc", "rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]
    )
    sp3 = builder.parse_mgex_sp3(set(satellite_ids))
    geometry = interpolate_sp3_cubic(receiver, sp3, satellite_ids)
    spice_cache = builder.spice_geometry(geometry["utc"])
    links = builder.expand_signals(geometry, SELECTED)
    physics = builder.build_physics(links, spice_cache, mapping, patterns)
    physics["op"] = "dynamic_reconstruction_continuous"
    physics["orbit_geometry_status"] = (
        "LuGRE_dynamics_reconstruction_plus_CODE_MGEX_final_SP3_cubic"
    )
    physics["receiver_trajectory_source"] = physics[
        "receiver_trajectory_source"
    ].fillna("stitched_orbit_dynamics_reconstruction")
    physics["gnss_orbit_source"] = "CODE_MGEX_final_5min_SP3_cubic_interpolation"
    physics["fspl_status"] = "computed_signal_frequency_dynamic_receiver_SP3_geometry"
    physics["cn0_2d_model_status"] = (
        "dynamic_receiver_SP3_band_specific_transmit_and_receive_envelope"
    )
    physics["constellation_direct_status"] = np.where(
        physics["earth_blocked"].eq(0) & physics["moon_blocked"].eq(0),
        "dynamic_receiver_SP3_band_specific_direct_budget",
        "occulted",
    )
    physics["constellation_physics_proxy_status"] = physics[
        "constellation_direct_status"
    ]
    physics["full_mission_geometry_cadence"] = CADENCE
    physics["full_mission_selected_satellites"] = physics["system"].map(
        {"G": "G11", "E": "E34"}
    )
    physics.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "signal_name": signal,
                "selected_svids": ",".join(map(str, svids)),
                "geometry_cadence": CADENCE,
                "receiver_geometry": "LuGRE orbit-dynamics reconstruction",
                "gnss_geometry": "CODE MGEX final SP3 cubic interpolation",
            }
            for signal, svids in SELECTED.items()
        ]
    ).to_csv(SELECTION_PATH, index=False, encoding="utf-8-sig")
    print(f"dynamic_receiver_epochs={len(receiver):,}")
    print(f"geometry_rows={len(geometry):,}")
    print(f"physics_rows={len(physics):,}")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
