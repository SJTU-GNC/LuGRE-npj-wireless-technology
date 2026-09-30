#!/usr/bin/env python3
"""Render the accepted Figure 5 with audited model/observation 24-h windows."""

import plot_full_mission_four_band_ai_cn0_sidelobe_1min_three_sat_24h as figure
import build_dynamic_multiband_one_common_1min as one_minute_builder


BASE_STEM = "Fig_full_mission_four_band_AI_CN0_1min_one_sat_24h"

figure.common_builder = one_minute_builder
figure.base.CONTINUOUS_PATH = one_minute_builder.OUTPUT_PATH
figure.STEM = f"{BASE_STEM}_reselected_24h"
figure.N_COMMON_SATELLITES = 1
figure.LINE_COLORS = ["#2F6DAE"]
figure.base.SIDELOBE_COLOR = "#D3D6DA"
figure.DETAIL_PANEL_MODES = ("model_count", "coverage")
figure.DETAIL_WINDOW_OVERRIDES = {}
figure.OMIT_BELOW_LOCK_THRESHOLD = False
figure.COLOR_BY_PHASE = True
figure.MAIN_Y_LIMITS = (10.0, 40.0)


if __name__ == "__main__":
    figure.main()
