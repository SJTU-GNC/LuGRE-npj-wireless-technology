#!/usr/bin/env python3
"""Run the frozen constellation physics model with upgraded orbit geometry."""

from pathlib import Path

import build_cn0_constellation_physics_baseline as model


ROOT = Path(__file__).resolve().parents[1]
model.INPUT_FEATURES = ROOT / "table" / "cn0_physics_features_sp3_tlm_exact.csv"
model.INPUT_OLD_BASELINE = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_physics_baseline_sp3_tlm_exact"
    / "cn0_physics_baseline_predictions.csv"
)
model.INPUT_GPS_DIRECT = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_attitude_2d_gain_sp3_tlm_exact"
    / "cn0_reference_trajectory_attitude_2d_predictions.csv"
)
model.OUT_DIR = ROOT / "table" / "algorithm" / "cn0_constellation_physics_baseline_sp3_tlm_exact"


if __name__ == "__main__":
    model.main()
