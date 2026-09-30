#!/usr/bin/env python3
"""Build full-mission physics features for three common dual-band satellites."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "runtime_cache" / "python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "script"))

import pandas as pd

import build_plot_wgc_multiband_cn0 as builder


OUTPUT_DIR = ROOT / "table" / "algorithm" / "cn0_wgc_multiband_common_three"
OUTPUT_PATH = OUTPUT_DIR / "cn0_wgc_multiband_common_three_predictions.csv"
SELECTION_PATH = OUTPUT_DIR / "cn0_wgc_multiband_common_three_selection.csv"
SELECTED = {
    "GPS_L1": [11, 18, 14],
    "GPS_L5": [11, 18, 14],
    "GAL_E1": [34, 15, 26],
    "GAL_E5a": [34, 15, 26],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(builder.OBS, low_memory=False)
    _, mapping, patterns = builder.available_satellites(observations)
    satellite_ids = sorted(
        {
            f"{builder.SIGNAL_SYSTEM[signal]}{svid:02d}"
            for signal, svids in SELECTED.items()
            for svid in svids
        }
    )
    wgc = builder.load_wgc()
    sp3 = builder.parse_mgex_sp3(set(satellite_ids))
    geometry = builder.interpolate_sp3(wgc, sp3, satellite_ids)
    spice_cache = builder.spice_geometry(geometry["utc"])
    links = builder.expand_signals(geometry, SELECTED)
    physics = builder.build_physics(links, spice_cache, mapping, patterns)
    physics["common_three_satellite_set"] = physics["system"].map(
        {"G": "G11,G18,G14", "E": "E34,E15,E26"}
    )
    physics.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"signal_name": signal, "selected_svids": ",".join(map(str, svids))}
            for signal, svids in SELECTED.items()
        ]
    ).to_csv(SELECTION_PATH, index=False, encoding="utf-8-sig")
    print(f"wgc_epochs={len(wgc):,}")
    print(f"geometry_rows={len(geometry):,}")
    print(f"physics_rows={len(physics):,}")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
