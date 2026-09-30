#!/usr/bin/env python3
"""Full-mission 1-min C/N0 trends with representative 24-h detail windows.

The frozen residual model is evaluated on the configured receiver-trajectory
and precise-SP3 geometry. No additional fitting is performed for this figure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "script"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

import plot_full_mission_four_band_ai_cn0_sidelobe as base
import build_wgc_multiband_common_three as common_builder


STEM = (
    "Fig_full_mission_four_band_AI_CN0_1min_three_sat_24h_"
    "chronological_observation_supported_v2"
)
N_COMMON_SATELLITES = 3
RAW_SEGMENT_GAP = pd.Timedelta("12min30s")
MINUTE_SEGMENT_GAP = pd.Timedelta("2min")
SIDELOBE_MAJORITY_THRESHOLD = 0.5
ZOOM_COLOR = "#DDEAF3"
SIDELOBE_ZOOM_COLOR = "#E9E2F0"
LINE_COLORS = base.COLORS
DETAIL_PANEL_MODES = ("model_count", "coverage")
DETAIL_WINDOW_OVERRIDES: dict[str, str] = {}
ASSUMED_LOSS_OF_LOCK_DBHZ = 10.0
OMIT_BELOW_LOCK_THRESHOLD = True
COLOR_BY_PHASE = False
MAIN_Y_LIMITS = (base.DISPLAY_MIN_DBHZ, base.DISPLAY_MAX_DBHZ)
TLM_DIR = ROOT / "data" / "receiver_observation" / "TLM"
GPS_EPOCH = pd.Timestamp("1980-01-06T00:00:00Z")
RX_TIME_RE = re.compile(r"rxTime:\s*([0-9.]+)")
MEASURE_RE = re.compile(
    r"svid:\s*(?P<svid>\d+).*?cn0:\s*(?P<cn0>[-+0-9.Ee]+)\s+"
    r"signalId:\s*(?P<signal_id>\d+)"
)
TLM_FILE_RE = re.compile(
    r"TLM_RAW_(?P<date>\d{8})_(?P<time>\d{6})_(?P<hours>\d+)H_"
    r"(?P<phase>[CTLS])_(?P<op>OP\d+)_"
)
SIGNAL_META = {
    0: ("G", "GPS_L1"),
    1: ("G", "GPS_L5"),
    2: ("E", "GAL_E1"),
    3: ("E", "GAL_E5a"),
}
OBSERVATION_COLOR = "#20262C"
DETAIL_OBSERVATION_SIZE_PT2 = 3.2
DETAIL_OBSERVATION_MARKER_SIZE_PT = float(np.sqrt(DETAIL_OBSERVATION_SIZE_PT2))
DETAIL_OBSERVATION_ALPHA = 0.16
DETAIL_OBSERVATION_ZORDER = 2.35
DETAIL_MODEL_LINEWIDTH_PT = 1.45
DETAIL_MODEL_ALPHA = 1.0
DETAIL_MODEL_ZORDER = 5.0
DETAIL_STATE_STYLES = {
    "side_lobe": ("#D3D6DA", 0.46, "Side lobe"),
    "earth_occultation": ("#ECA8A4", 0.52, "Earth occultation"),
    "moon_occultation": ("#D5C5E4", 0.58, "Moon occultation"),
    "transmit_pattern_unsupported": (
        "#EED3A4",
        0.54,
        "Antenna back-hemisphere region",
    ),
}
VISIBLE_DETAIL_LEGEND_STATES = (
    "side_lobe",
    "earth_occultation",
    "transmit_pattern_unsupported",
)
PHASE_COLORS = {
    "C": "#0072B2",
    "T": "#009E73",
    "L": "#E69F00",
    "S": "#CC79A7",
}
DETAIL_PANEL_MARKERS = tuple(chr(0x2460 + index) for index in range(8))

base.CONTINUOUS_PATH = common_builder.OUTPUT_PATH


def detail_panel_marker(
    signal_index: int, window_index: int
) -> tuple[int, str]:
    """Number each band's earlier/later 24-h pair consecutively."""
    marker_index = signal_index * len(DETAIL_PANEL_MODES) + window_index
    return marker_index + 1, DETAIL_PANEL_MARKERS[marker_index]


def select_three_common_satellites(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select three satellites shared by both bands of each constellation."""
    counts = (
        data[data["display_eligible"]]
        .groupby(["signal_name", "svid"], as_index=False)
        .agg(
            valid_model_epochs=("utc", "size"),
            first_utc=("utc", "min"),
            last_utc=("utc", "max"),
            sidelobe_epochs=("outside_ssv_main_lobe", "sum"),
        )
    )
    observations = pd.read_csv(
        base.OBSERVATION_PATH,
        usecols=["signal_name", "svid"],
        low_memory=False,
    )
    observations["svid"] = pd.to_numeric(
        observations["svid"], errors="coerce"
    ).astype("Int64")
    observation_counts = (
        observations.groupby(["signal_name", "svid"], as_index=False)
        .size()
        .rename(columns={"size": "tlm_observation_rows"})
    )
    counts = counts.merge(
        observation_counts, on=["signal_name", "svid"], how="left"
    )
    counts["tlm_observation_rows"] = counts["tlm_observation_rows"].fillna(0)

    chosen_rows: list[dict] = []
    for signal_pair in [("GPS_L1", "GPS_L5"), ("GAL_E1", "GAL_E5a")]:
        pair = counts[counts["signal_name"].isin(signal_pair)].copy()
        model_pivot = pair.pivot(
            index="svid", columns="signal_name", values="valid_model_epochs"
        )
        tlm_pivot = pair.pivot(
            index="svid", columns="signal_name", values="tlm_observation_rows"
        )
        common = model_pivot.dropna(subset=list(signal_pair)).copy()
        if len(common) < N_COMMON_SATELLITES:
            raise RuntimeError(
                f"Only {len(common)} common satellites are available for {signal_pair}"
            )
        common["minimum_band_epochs"] = common[list(signal_pair)].min(axis=1)
        common["total_band_epochs"] = common[list(signal_pair)].sum(axis=1)
        common["minimum_tlm_rows"] = (
            tlm_pivot.reindex(common.index)[list(signal_pair)].fillna(0).min(axis=1)
        )
        common["total_tlm_rows"] = (
            tlm_pivot.reindex(common.index)[list(signal_pair)].fillna(0).sum(axis=1)
        )
        common = common.sort_values(
            [
                "minimum_tlm_rows",
                "total_tlm_rows",
                "minimum_band_epochs",
                "total_band_epochs",
            ],
            ascending=False,
        )
        selected_svids = [int(value) for value in common.head(N_COMMON_SATELLITES).index]
        for rank, svid in enumerate(selected_svids, start=1):
            for signal_name in signal_pair:
                row = pair[
                    pair["signal_name"].eq(signal_name) & pair["svid"].eq(svid)
                ].iloc[0]
                chosen_rows.append(
                    {
                        "signal_name": signal_name,
                        "svid": svid,
                        "rank_within_constellation": rank,
                        "valid_model_epochs": int(row["valid_model_epochs"]),
                        "tlm_observation_rows": int(row["tlm_observation_rows"]),
                        "first_utc": row["first_utc"],
                        "last_utc": row["last_utc"],
                        "ssv_outside_fraction_diagnostic": float(
                            row["sidelobe_epochs"] / row["valid_model_epochs"]
                        ),
                        "selection_basis": (
                            f"top {N_COMMON_SATELLITES} satellite(s) common to both "
                            "constellation bands, "
                            "ranked by telemetry coverage then full-mission model coverage"
                        ),
                    }
                )

    selected = pd.DataFrame(chosen_rows)
    keep = pd.Series(False, index=data.index)
    for row in selected.itertuples():
        keep |= data["signal_name"].eq(row.signal_name) & data["svid"].eq(row.svid)
    return data[keep].copy(), selected


def load_minute_observation_coverage(selected: pd.DataFrame) -> pd.DataFrame:
    """Load observed C/N0 coverage for choosing a detail window with telemetry."""
    observations = pd.read_csv(
        base.OBSERVATION_PATH,
        usecols=["rx_utc", "signal_name", "svid", "cn0_dbhz_mean"],
        low_memory=False,
    )
    observations["utc"] = pd.to_datetime(
        observations["rx_utc"], utc=True, errors="coerce"
    ).dt.floor("min")
    observations["svid"] = pd.to_numeric(
        observations["svid"], errors="coerce"
    ).astype("Int64")
    observations["cn0_dbhz_mean"] = pd.to_numeric(
        observations["cn0_dbhz_mean"], errors="coerce"
    )
    keys = selected[["signal_name", "svid"]].drop_duplicates()
    observations = observations.merge(keys, on=["signal_name", "svid"], how="inner")
    return observations[
        observations["utc"].notna() & observations["cn0_dbhz_mean"].notna()
    ].copy()


def _tlm_file_overlaps_detail_windows(
    path: Path, windows: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> bool:
    match = TLM_FILE_RE.search(path.name)
    if match is None:
        return True
    start = pd.Timestamp(
        f"{match.group('date')} {match.group('time')}", tz="UTC"
    )
    stop = start + pd.Timedelta(hours=int(match.group("hours"))) + pd.Timedelta(hours=1)
    return any(start < window_stop and stop > window_start for window_start, window_stop in windows)


def load_one_second_observations(
    selected: pd.DataFrame,
    detail_days: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Parse raw LuGRE observations only for the displayed 24-h detail windows."""
    windows = [
        (pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=1))
        for start in detail_days.values()
    ]
    selected_keys = {
        (str(row.signal_name), int(row.svid)) for row in selected.itertuples()
    }
    rows: list[dict[str, object]] = []
    for path in sorted(TLM_DIR.glob("TLM_RAW_*.txt")):
        if not _tlm_file_overlaps_detail_windows(path, windows):
            continue
        file_match = TLM_FILE_RE.search(path.name)
        operation = file_match.group("op") if file_match is not None else "unknown"
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                time_match = RX_TIME_RE.search(line)
                if time_match is None:
                    continue
                gps_seconds = float(time_match.group(1))
                utc = GPS_EPOCH + pd.to_timedelta(gps_seconds - 18.0, unit="s")
                if not any(start <= utc < stop for start, stop in windows):
                    continue
                for measure in MEASURE_RE.finditer(line):
                    signal_id = int(measure.group("signal_id"))
                    if signal_id not in SIGNAL_META:
                        continue
                    system, signal_name = SIGNAL_META[signal_id]
                    svid = int(measure.group("svid"))
                    if (signal_name, svid) not in selected_keys:
                        continue
                    rows.append(
                        {
                            "op": operation,
                            "rx_utc": utc,
                            "system": system,
                            "signal_name": signal_name,
                            "svid": svid,
                            "cn0_observed_dbhz": float(measure.group("cn0")),
                            "source_file": path.name,
                            "source_cadence": "LuGRE TLM RAW approximately 1 s",
                        }
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "op",
                "rx_utc",
                "system",
                "signal_name",
                "svid",
                "cn0_observed_dbhz",
                "source_file",
                "source_cadence",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["signal_name", "svid", "rx_utc"]
    ).reset_index(drop=True)


def build_detail_status(data: pd.DataFrame) -> pd.DataFrame:
    """Build mutually exclusive background states for the 24-h detail panels."""
    status = data.copy()
    earth = pd.to_numeric(status["earth_blocked"], errors="coerce").eq(1)
    moon = pd.to_numeric(status["moon_blocked"], errors="coerce").eq(1)
    tx_offboresight = pd.to_numeric(
        status.get("tx_theta_body_deg"), errors="coerce"
    )

    pattern_query_finite = pd.Series(False, index=status.index, dtype=bool)
    for column in ("tx_eirp_2d_dbw", "tx_gain_2d_db"):
        if column in status.columns:
            pattern_query_finite |= pd.to_numeric(
                status[column], errors="coerce"
            ).notna()
    model_output_finite = pd.Series(False, index=status.index, dtype=bool)
    for column in (
        "cn0_reference_trajectory_2d_dbhz",
        "cn0_physics_ai_trend_dbhz",
    ):
        if column in status.columns:
            model_output_finite |= pd.to_numeric(
                status[column], errors="coerce"
            ).notna()
    # The displayed back-hemisphere mask is transmitter-side only. The
    # implemented two-dimensional transmit patterns support the 0-90 degree
    # front hemisphere; receiver off-boresight is never used for this mask.
    transmit_pattern_unsupported = tx_offboresight.gt(90.0)
    if (transmit_pattern_unsupported & model_output_finite).any():
        raise RuntimeError(
            "Transmit-pattern unsupported state overlaps finite model output"
        )

    side_lobe = status["pattern_first_null_resolved"].fillna(False).astype(bool) & status[
        "pattern_side_lobe"
    ].fillna(False).astype(bool)
    status["detail_background_state"] = np.select(
        [
            earth,
            moon,
            transmit_pattern_unsupported,
            side_lobe,
        ],
        [
            "earth_occultation",
            "moon_occultation",
            "transmit_pattern_unsupported",
            "side_lobe",
        ],
        default="nominal",
    )
    status["transmit_pattern_unsupported"] = transmit_pattern_unsupported
    status["tx_pattern_query_finite"] = pattern_query_finite
    status["model_output_finite"] = model_output_finite
    status["detail_background_priority"] = status["detail_background_state"].map(
        {
            "nominal": 0,
            "side_lobe": 1,
            "transmit_pattern_unsupported": 2,
            "moon_occultation": 3,
            "earth_occultation": 4,
        }
    )
    status["detail_background_basis"] = np.select(
        [
            earth,
            moon,
            transmit_pattern_unsupported,
            side_lobe,
        ],
        [
            "earth_occultation_mask",
            "moon_occultation_mask",
            "transmit_pattern_query_outside_supported_front_hemisphere",
            "resolved_first_null_2D_pattern_side_lobe",
        ],
        default="nominal_supported_geometry",
    )
    keep = [
        "utc",
        "signal_name",
        "svid",
        "earth_blocked",
        "moon_blocked",
        "rx_offboresight_spice_deg",
        "tx_theta_body_deg",
        "pattern_first_null_boundary_deg",
        "pattern_side_lobe",
        "tx_pattern_query_finite",
        "model_output_finite",
        "transmit_pattern_unsupported",
        "detail_background_state",
        "detail_background_priority",
        "detail_background_basis",
    ]
    return status[[column for column in keep if column in status.columns]].copy()


def first_null_angle(theta_deg: np.ndarray, gain_db: np.ndarray) -> float:
    """Return the first resolved local minimum after the central pattern peak."""
    theta = np.asarray(theta_deg, dtype=float)
    gain = np.asarray(gain_db, dtype=float)
    finite = np.isfinite(theta) & np.isfinite(gain)
    theta = theta[finite]
    gain = gain[finite]
    if len(theta) < 7:
        return np.nan
    order = np.argsort(theta)
    theta = theta[order]
    gain = gain[order]
    smooth = (
        pd.Series(gain)
        .rolling(window=3, center=True, min_periods=1)
        .mean()
        .to_numpy(float)
    )
    peak_candidates = np.flatnonzero(theta <= 18.0)
    if not len(peak_candidates):
        return np.nan
    peak_index = int(peak_candidates[np.nanargmax(smooth[peak_candidates])])
    for index in range(peak_index + 2, len(theta) - 1):
        if theta[index] > 45.0:
            break
        local_minimum = smooth[index] <= smooth[index - 1] and smooth[index] < smooth[index + 1]
        significant_drop = smooth[index] <= smooth[peak_index] - 6.0
        if local_minimum and significant_drop:
            return float(theta[index])
    return np.nan


def nearest_azimuth_boundary(
    azimuth_deg: pd.Series,
    azimuth_grid: np.ndarray,
    boundary_grid: np.ndarray,
) -> np.ndarray:
    query = np.mod(pd.to_numeric(azimuth_deg, errors="coerce").to_numpy(float), 360.0)
    grid = np.mod(np.asarray(azimuth_grid, dtype=float), 360.0)
    boundaries = np.asarray(boundary_grid, dtype=float)
    result = np.full(len(query), np.nan, dtype=float)
    valid_query = np.isfinite(query)
    if not valid_query.any() or not len(grid):
        return result
    distance = np.abs(query[valid_query, None] - grid[None, :])
    distance = np.minimum(distance, 360.0 - distance)
    nearest = np.argmin(distance, axis=1)
    result[valid_query] = boundaries[nearest]
    return result


def circular_interpolated_boundary(
    azimuth_deg: pd.Series,
    azimuth_grid: np.ndarray,
    boundary_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate a continuous first-null contour across pattern azimuth."""
    query = np.mod(pd.to_numeric(azimuth_deg, errors="coerce").to_numpy(float), 360.0)
    grid = np.mod(np.asarray(azimuth_grid, dtype=float), 360.0)
    boundaries = np.asarray(boundary_grid, dtype=float)
    finite = np.isfinite(grid) & np.isfinite(boundaries)
    result = np.full(len(query), np.nan, dtype=float)
    if finite.sum() < 2:
        return result

    contour = (
        pd.DataFrame({"azimuth": grid[finite], "boundary": boundaries[finite]})
        .groupby("azimuth", as_index=False)["boundary"]
        .median()
        .sort_values("azimuth")
    )
    azimuth = contour["azimuth"].to_numpy(float)
    boundary = contour["boundary"].to_numpy(float)
    extended_azimuth = np.concatenate(
        ([azimuth[-1] - 360.0], azimuth, [azimuth[0] + 360.0])
    )
    extended_boundary = np.concatenate(([boundary[-1]], boundary, [boundary[0]]))
    valid_query = np.isfinite(query)
    result[valid_query] = np.interp(
        query[valid_query], extended_azimuth, extended_boundary
    )
    return result


def dense_pattern_first_null_contour(
    pattern: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve first nulls after applying the same 2-D interpolation as physics."""
    azimuth_grid = np.arange(0.0, 360.0, 1.0)
    theta_grid = np.asarray(pattern["theta"], dtype=float)
    boundaries = np.full(len(azimuth_grid), np.nan, dtype=float)
    for index, azimuth in enumerate(azimuth_grid):
        gain_cut = common_builder.builder.bilinear_periodic(
            pattern,
            theta_grid,
            np.full(len(theta_grid), azimuth, dtype=float),
        )
        boundaries[index] = first_null_angle(theta_grid, gain_cut)
    return azimuth_grid, boundaries


def same_cut_main_peak(
    pattern: dict[str, np.ndarray], azimuth_deg: pd.Series
) -> np.ndarray:
    """Return the central-lobe peak on each row's interpolated azimuth cut."""
    azimuth = pd.to_numeric(azimuth_deg, errors="coerce").to_numpy(float)
    peak = np.full(len(azimuth), np.nan, dtype=float)
    valid = np.isfinite(azimuth)
    if not valid.any():
        return peak
    central_theta = np.arange(0.0, 18.01, 0.5)
    values = np.full(valid.sum(), -np.inf, dtype=float)
    for theta in central_theta:
        evaluated = common_builder.builder.bilinear_periodic(
            pattern,
            np.full(valid.sum(), theta, dtype=float),
            azimuth[valid],
        )
        values = np.maximum(values, evaluated)
    peak[valid] = values
    return peak


def add_pattern_lobe_classification(data: pd.DataFrame) -> pd.DataFrame:
    """Classify main/side lobe from the same 2-D gain maps used by physics."""
    output = data.copy()
    output["pattern_first_null_boundary_deg"] = np.nan
    output["pattern_first_null_resolved"] = False
    output["pattern_first_null_boundary_source"] = "unavailable"
    output["pattern_side_lobe"] = False
    output["pattern_lobe_classification"] = "unavailable"
    output["pattern_lobe_basis"] = (
        "dense_first-null_contour_after_same_2D_interpolation_used_by_link_budget"
    )
    output["tx_pattern_value_used_db"] = np.nan
    output["tx_pattern_main_peak_same_cut_db"] = np.nan
    output["tx_pattern_relative_to_main_peak_db"] = np.nan

    observations = pd.read_csv(
        common_builder.builder.OBS,
        usecols=["signal_name", "svid"],
        low_memory=False,
    )
    _, mapping, gps_patterns = common_builder.builder.available_satellites(observations)
    mapping = mapping.copy()
    mapping["svid"] = mapping["prn"].str.extract(r"[GE](\d+)").astype(int)
    mapping["svn_numeric"] = pd.to_numeric(
        mapping["sv_identifier"].str.extract(r"G(\d+)")[0], errors="coerce"
    )
    gps_svn_by_svid = (
        mapping[mapping["system"].eq("G")]
        .dropna(subset=["svn_numeric"])
        .drop_duplicates("svid")
        .set_index("svid")["svn_numeric"]
        .astype(int)
        .to_dict()
    )

    for signal_name in ["GPS_L1", "GPS_L5"]:
        for svid in output.loc[output["signal_name"].eq(signal_name), "svid"].dropna().unique():
            svn = gps_svn_by_svid.get(int(svid))
            pattern = gps_patterns.get(signal_name, {}).get(svn)
            if pattern is None:
                continue
            boundaries = np.asarray(
                dense_pattern_first_null_contour(pattern)[1], dtype=float
            )
            dense_azimuths = np.arange(0.0, 360.0, 1.0)
            mask = output["signal_name"].eq(signal_name) & output["svid"].eq(svid)
            output.loc[mask, "pattern_first_null_boundary_deg"] = circular_interpolated_boundary(
                output.loc[mask, "tx_phi_body_deg"], dense_azimuths, boundaries
            )
            output.loc[mask, "pattern_first_null_boundary_source"] = (
                "dense_1deg_phi_contour_from_same_bilinear_2D_pattern_used_in_physics"
            )
            output.loc[mask, "tx_pattern_value_used_db"] = pd.to_numeric(
                output.loc[mask, "tx_gain_2d_db"], errors="coerce"
            )
            output.loc[mask, "tx_pattern_main_peak_same_cut_db"] = same_cut_main_peak(
                pattern, output.loc[mask, "tx_phi_body_deg"]
            )

    grap = pd.read_csv(base.GRAP_PATH, low_memory=False)
    for signal_name in ["GAL_E1", "GAL_E5a"]:
        signal_pattern = grap[grap["signal_name"].eq(signal_name)]
        pivot = (
            signal_pattern.pivot_table(
                index="coelevation_deg",
                columns="azimuth_deg",
                values="eirp_dbw",
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        pattern = {
            "theta": pivot.index.to_numpy(float),
            "phi": pivot.columns.to_numpy(float)[:-1],
            "gain": pivot.to_numpy(float)[:, :-1],
        }
        azimuths, boundaries = dense_pattern_first_null_contour(pattern)
        mask = output["signal_name"].eq(signal_name)
        output.loc[mask, "pattern_first_null_boundary_deg"] = circular_interpolated_boundary(
            output.loc[mask, "tx_phi_body_deg"],
            azimuths,
            boundaries,
        )
        output.loc[mask, "pattern_first_null_boundary_source"] = (
            "dense_1deg_phi_contour_from_same_bilinear_2D_pattern_used_in_physics"
        )
        output.loc[mask, "tx_pattern_value_used_db"] = pd.to_numeric(
            output.loc[mask, "tx_eirp_2d_dbw"], errors="coerce"
        )
        output.loc[mask, "tx_pattern_main_peak_same_cut_db"] = same_cut_main_peak(
            pattern, output.loc[mask, "tx_phi_body_deg"]
        )

    boundary = pd.to_numeric(output["pattern_first_null_boundary_deg"], errors="coerce")
    theta = pd.to_numeric(output["tx_theta_body_deg"], errors="coerce")
    resolved = boundary.notna() & theta.notna()
    output["pattern_first_null_resolved"] = resolved
    output["pattern_side_lobe"] = resolved & theta.gt(boundary)
    output.loc[resolved & theta.le(boundary), "pattern_lobe_classification"] = (
        "main_lobe_inside_first_null"
    )
    output.loc[resolved & theta.gt(boundary), "pattern_lobe_classification"] = (
        "side_lobe_beyond_first_null"
    )
    output.loc[~resolved, "pattern_lobe_classification"] = "first_null_unresolved"
    output["pattern_angular_margin_deg"] = theta - boundary
    output["tx_pattern_relative_to_main_peak_db"] = (
        pd.to_numeric(output["tx_pattern_value_used_db"], errors="coerce")
        - pd.to_numeric(output["tx_pattern_main_peak_same_cut_db"], errors="coerce")
    )
    relative_gain = pd.to_numeric(
        output["tx_pattern_relative_to_main_peak_db"], errors="coerce"
    )
    output["pattern_label_gain_consistency"] = np.select(
        [
            ~resolved | relative_gain.isna(),
            output["pattern_side_lobe"] & relative_gain.gt(-6.0),
        ],
        [
            "not_auditable_missing_pattern_value_or_boundary",
            "inconsistent_side_label_less_than_6dB_below_main_peak",
        ],
        default="consistent_with_same_pattern_value_used_by_model",
    )
    return output


def raw_contiguous_segments(track: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = track.sort_values("utc")
    usable = ordered[ordered["display_eligible"]].copy()
    if usable.empty:
        return []
    breaks = usable["utc"].diff().gt(RAW_SEGMENT_GAP).fillna(False)
    blocked = ordered["earth_blocked"].eq(1) | ordered["moon_blocked"].eq(1)
    times = usable["utc"].to_list()
    for index in range(1, len(times)):
        if breaks.iloc[index]:
            continue
        between = ordered["utc"].gt(times[index - 1]) & ordered["utc"].lt(times[index])
        if bool(blocked.loc[between].any()):
            breaks.iloc[index] = True
    return [segment for _, segment in usable.groupby(breaks.cumsum()) if len(segment) >= 2]


def interpolate_to_one_minute(data: pd.DataFrame) -> pd.DataFrame:
    """Interpolate only the already-smoothed model trend onto a 1-min grid."""
    pieces: list[pd.DataFrame] = []
    for _, track in data.groupby(["signal_name", "svid"], sort=False, dropna=False):
        for segment in raw_contiguous_segments(track):
            source = (
                segment.sort_values("utc")
                .drop_duplicates("utc", keep="first")
                .set_index("utc")
            )
            minute_index = pd.date_range(
                source.index.min().ceil("min"),
                source.index.max().floor("min"),
                freq="1min",
                tz="UTC",
            )
            if minute_index.empty:
                continue
            minute = source.reindex(
                minute_index, method="nearest", tolerance=pd.Timedelta("3min")
            )
            raw_trend = pd.to_numeric(
                source["cn0_physics_ai_trend_dbhz"], errors="coerce"
            )
            expanded_index = raw_trend.index.union(minute_index)
            interpolated = (
                raw_trend.reindex(expanded_index)
                .sort_index()
                .interpolate(method="time")
                .reindex(minute_index)
            )
            minute["cn0_physics_ai_trend_dbhz"] = interpolated.to_numpy(float)
            minute["cn0_display_dbhz"] = interpolated.clip(
                lower=base.DISPLAY_MIN_DBHZ, upper=base.DISPLAY_MAX_DBHZ
            ).to_numpy(float)
            minute["display_eligible"] = (
                interpolated.notna().to_numpy() & minute["signal_name"].notna().to_numpy()
            )
            minute["display_value_clipped"] = ~interpolated.between(
                base.DISPLAY_MIN_DBHZ, base.DISPLAY_MAX_DBHZ
            ).to_numpy()
            minute["sidelobe_background_eligible"] = interpolated.between(
                base.DISPLAY_MIN_DBHZ, base.DISPLAY_MAX_DBHZ
            ).to_numpy()
            minute["sidelobe_fraction_in_display_bin"] = minute[
                "outside_ssv_main_lobe"
            ].astype(float)
            minute["model_output_cadence"] = (
                "1-min frozen-model output from reconstructed-trajectory/SP3 physics epochs; "
                "nine-minute robust trend head; no post-model cadence upsampling"
            )
            minute["utc"] = minute_index
            pieces.append(minute.reset_index(drop=True))
    if not pieces:
        raise RuntimeError("No contiguous model segments were available for 1-min display")
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["signal_name", "svid", "utc"]
    )


def majority_sidelobe_intervals(panel: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    eligible = panel[panel["display_eligible"]].copy()
    if eligible.empty:
        return []
    resolved = eligible[eligible["pattern_first_null_resolved"]].copy()
    if resolved.empty:
        return []
    state = resolved.groupby("utc")["pattern_side_lobe"].agg(
        active_links="size", outside_fraction="mean"
    )
    side_times = pd.Series(
        state.index[state["outside_fraction"].gt(SIDELOBE_MAJORITY_THRESHOLD)]
    )
    if side_times.empty:
        return []
    groups = side_times.diff().gt(MINUTE_SEGMENT_GAP).cumsum()
    return [
        (
            pd.Timestamp(group.iloc[0]) - pd.Timedelta(seconds=30),
            pd.Timestamp(group.iloc[-1]) + pd.Timedelta(seconds=30),
        )
        for _, group in side_times.groupby(groups)
    ]


def minute_segments(
    track: pd.DataFrame,
    value_column: str = "cn0_physics_ai_trend_dbhz",
) -> list[pd.DataFrame]:
    cn0 = pd.to_numeric(track[value_column], errors="coerce")
    usable = track["display_eligible"]
    if OMIT_BELOW_LOCK_THRESHOLD:
        usable &= cn0.ge(ASSUMED_LOSS_OF_LOCK_DBHZ)
    ordered = track[usable].sort_values("utc")
    if ordered.empty:
        return []
    groups = ordered["utc"].diff().gt(MINUTE_SEGMENT_GAP).cumsum()
    return [segment for _, segment in ordered.groupby(groups) if len(segment) >= 2]


def choose_detail_days(
    data: pd.DataFrame,
    observation_coverage: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.Timestamp], pd.DataFrame]:
    """Select common model-count and observation-count sliding 24-h windows."""
    working = data.copy()
    working["utc"] = pd.to_datetime(working["utc"], utc=True, errors="coerce")
    working = working[working["utc"].notna()].copy()
    working["display_eligible"] = working["display_eligible"].fillna(False).astype(bool)
    working["pattern_first_null_resolved"] = (
        working["pattern_first_null_resolved"].fillna(False).astype(bool)
    )
    working["pattern_side_lobe"] = (
        working["pattern_side_lobe"].fillna(False).astype(bool)
    )
    working["_model_valid"] = (
        working["display_eligible"]
        & pd.to_numeric(
            working["cn0_physics_ai_trend_dbhz"], errors="coerce"
        ).notna()
    )
    working["_resolved_valid"] = (
        working["_model_valid"] & working["pattern_first_null_resolved"]
    )
    working["_sidelobe_valid"] = (
        working["_resolved_valid"] & working["pattern_side_lobe"]
    )

    minute_start = working["utc"].min().ceil("min")
    minute_stop = working["utc"].max().floor("min")
    minute_index = pd.date_range(minute_start, minute_stop, freq="1min", tz="UTC")
    window_minutes = 24 * 60
    if len(minute_index) < window_minutes:
        raise RuntimeError("The mission span is shorter than 24 hours")

    def sliding_sum(values: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(values, dtype=np.int64)]
        )
        return cumulative[window_minutes:] - cumulative[:-window_minutes]

    def minute_presence(frame: pd.DataFrame, mask: pd.Series) -> np.ndarray:
        times = (
            frame.loc[mask, "utc"]
            .dt.floor("min")
            .dropna()
            .drop_duplicates()
        )
        if times.empty:
            return np.zeros(len(minute_index), dtype=np.int64)
        return (
            pd.Series(1, index=pd.DatetimeIndex(times), dtype=np.int64)
            .groupby(level=0)
            .max()
            .reindex(minute_index, fill_value=0)
            .to_numpy(np.int64)
        )

    window_count = len(minute_index) - window_minutes + 1
    windows = pd.DataFrame(
        {
            "window_start": minute_index[:window_count],
        }
    )
    windows["window_stop"] = windows["window_start"] + pd.Timedelta(hours=24)

    observation = (
        observation_coverage.copy()
        if observation_coverage is not None
        else pd.DataFrame(columns=["utc", "signal_name"])
    )
    if not observation.empty:
        observation["utc"] = pd.to_datetime(
            observation["utc"], utc=True, errors="coerce"
        )
        observation = observation[observation["utc"].notna()].copy()

    model_columns: list[str] = []
    resolved_columns: list[str] = []
    sidelobe_columns: list[str] = []
    observation_columns: list[str] = []
    for signal_name in base.SIGNALS:
        signal_model = working["signal_name"].eq(signal_name)
        model_column = f"{signal_name}_model_minutes"
        resolved_column = f"{signal_name}_pattern_resolved_minutes"
        sidelobe_column = f"{signal_name}_sidelobe_minutes"
        observation_column = f"{signal_name}_observation_minutes"
        windows[model_column] = sliding_sum(
            minute_presence(
                working,
                signal_model & working["_model_valid"],
            )
        )
        windows[resolved_column] = sliding_sum(
            minute_presence(
                working,
                signal_model & working["_resolved_valid"],
            )
        )
        windows[sidelobe_column] = sliding_sum(
            minute_presence(
                working,
                signal_model & working["_sidelobe_valid"],
            )
        )
        if observation.empty:
            observation_presence = np.zeros(len(minute_index), dtype=np.int64)
        else:
            observation_presence = minute_presence(
                observation,
                observation["signal_name"].eq(signal_name),
            )
        windows[observation_column] = sliding_sum(observation_presence)
        model_columns.append(model_column)
        resolved_columns.append(resolved_column)
        sidelobe_columns.append(sidelobe_column)
        observation_columns.append(observation_column)

    windows["observation_min_across_bands"] = windows[
        observation_columns
    ].min(axis=1)
    windows["observation_total_minutes"] = windows[
        observation_columns
    ].sum(axis=1)
    windows["model_min_across_bands"] = windows[model_columns].min(axis=1)
    windows["model_total_minutes"] = windows[model_columns].sum(axis=1)
    windows["pattern_resolved_total_minutes"] = windows[
        resolved_columns
    ].sum(axis=1)
    windows["sidelobe_min_across_bands"] = windows[
        sidelobe_columns
    ].min(axis=1)
    windows["sidelobe_total_minutes"] = windows[
        sidelobe_columns
    ].sum(axis=1)
    windows["all_four_bands_have_observation"] = (
        windows[observation_columns].gt(0).all(axis=1)
    )
    windows["all_four_bands_present"] = windows[model_columns].gt(0).all(axis=1)
    windows["all_four_bands_have_sidelobe"] = (
        windows[sidelobe_columns].gt(0).all(axis=1)
    )

    # Backward-compatible aggregate names retained for downstream provenance.
    windows["eligible_rows"] = windows["model_total_minutes"]
    windows["sidelobe_rows"] = windows["sidelobe_total_minutes"]
    windows["pattern_resolved_rows"] = windows[
        "pattern_resolved_total_minutes"
    ]
    windows["observation_rows"] = windows["observation_total_minutes"]
    windows["sidelobe_fraction"] = (
        windows["sidelobe_rows"]
        / windows["pattern_resolved_rows"].replace(0, np.nan)
    )

    model_ranked = windows.sort_values(
        ["model_total_minutes", "window_start"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    model_start = pd.Timestamp(model_ranked.iloc[0]["window_start"])
    coverage_ranked = windows.sort_values(
        [
            "observation_min_across_bands",
            "observation_total_minutes",
            "model_total_minutes",
            "window_start",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    chosen = {
        "model_count": model_start,
        "coverage": pd.Timestamp(coverage_ranked.iloc[0]["window_start"]),
    }
    for mode, timestamp in DETAIL_WINDOW_OVERRIDES.items():
        if mode in chosen:
            chosen[mode] = pd.Timestamp(timestamp, tz="UTC")
    windows["selected_as_coverage_window"] = windows["window_start"].eq(
        chosen["coverage"]
    )
    windows["selected_as_model_count_window"] = windows["window_start"].eq(
        chosen["model_count"]
    )
    return chosen, windows.sort_values("window_start").reset_index(drop=True)


def build_detail_window_audit(
    windows: pd.DataFrame,
    chosen: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Return the two selected windows with all per-band counts and criteria."""
    criteria = {
        "model_count": (
            "maximum total finite physics-plus-AI model-output minutes across "
            "the four selected links; earliest UTC start resolves exact ties"
        ),
        "coverage": (
            "maximum minimum observed-C/N0 minute count across the four "
            "selected links, then maximum total observed minutes, maximum "
            "model-output minutes, and earliest UTC start"
        ),
    }
    definitions = {
        "model_count": (
            "model-output minute = display_eligible=true and finite frozen "
            "physics-plus-AI C/N0 for the selected link"
        ),
        "coverage": (
            "observation minute = at least one finite selected-link C/N0 sample "
            "in that UTC minute"
        ),
    }
    rows: list[dict[str, object]] = []
    for mode in ("model_count", "coverage"):
        start = pd.Timestamp(chosen[mode])
        selected = windows[windows["window_start"].eq(start)]
        if len(selected) != 1:
            raise RuntimeError(
                f"Selected {mode} window is absent or duplicated: {start}"
            )
        record = selected.iloc[0].to_dict()
        record.update(
            {
                "window_mode": mode,
                "selection_criterion": criteria[mode],
                "count_definition": definitions[mode],
                "window_duration_minutes": 24 * 60,
                "candidate_start_cadence": "1 minute",
                "window_interval_convention": "[start, stop)",
                "candidate_window_count": int(len(windows)),
                "natural_day_boundary_required": False,
            }
        )
        rows.append(record)
    leading = [
        "window_mode",
        "window_start",
        "window_stop",
        "window_duration_minutes",
        "candidate_start_cadence",
        "window_interval_convention",
        "selection_criterion",
        "count_definition",
        "candidate_window_count",
        "natural_day_boundary_required",
    ]
    audit = pd.DataFrame(rows)
    return audit[leading + [column for column in audit.columns if column not in leading]]


def build_chronological_panel_audit(
    window_audit: pd.DataFrame,
    chosen: dict[str, pd.Timestamp],
    detail_status: pd.DataFrame,
    observations_1s: pd.DataFrame,
) -> pd.DataFrame:
    """Return one auditable row for each numbered 24-h detail panel."""
    ordered_modes = sorted(chosen, key=lambda mode: pd.Timestamp(chosen[mode]))
    rows: list[dict[str, object]] = []
    for window_index, mode in enumerate(ordered_modes):
        for signal_index, signal_name in enumerate(base.SIGNALS):
            start = pd.Timestamp(chosen[mode])
            stop = start + pd.Timedelta(days=1)
            selected_window = window_audit[
                window_audit["window_mode"].eq(mode)
            ]
            if len(selected_window) != 1:
                raise RuntimeError(f"Missing audit row for detail mode: {mode}")
            record = selected_window.iloc[0]
            panel = detail_status[
                detail_status["signal_name"].eq(signal_name)
                & detail_status["utc"].between(start, stop, inclusive="left")
            ]
            counts = panel["detail_background_state"].value_counts()
            observed = observations_1s[
                observations_1s["signal_name"].eq(signal_name)
                & observations_1s["rx_utc"].between(start, stop, inclusive="left")
            ].copy()
            observed["cn0_observed_dbhz"] = pd.to_numeric(
                observed["cn0_observed_dbhz"], errors="coerce"
            )
            observed = observed[observed["cn0_observed_dbhz"].notna()]
            panel_number, panel_marker = detail_panel_marker(
                signal_index, window_index
            )
            rows.append(
                {
                    "panel_number": panel_number,
                    "panel_marker": panel_marker,
                    "signal_name": signal_name,
                    "window_mode": mode,
                    "window_start_utc": start,
                    "window_stop_utc": stop,
                    "model_minutes": int(record[f"{signal_name}_model_minutes"]),
                    "sidelobe_minutes": int(
                        record[f"{signal_name}_sidelobe_minutes"]
                    ),
                    "observation_minutes": int(
                        record[f"{signal_name}_observation_minutes"]
                    ),
                    "observed_1s_points": int(len(observed)),
                    "observed_1s_unique_minutes": int(
                        observed["rx_utc"].dt.floor("min").nunique()
                    ),
                    "nominal_epochs": int(counts.get("nominal", 0)),
                    "side_lobe_epochs": int(counts.get("side_lobe", 0)),
                    "transmit_pattern_unsupported_epochs": int(
                        counts.get("transmit_pattern_unsupported", 0)
                    ),
                    "earth_occultation_epochs": int(
                        counts.get("earth_occultation", 0)
                    ),
                    "moon_occultation_epochs": int(
                        counts.get("moon_occultation", 0)
                    ),
                    "finite_model_epochs": int(
                        panel["model_output_finite"].fillna(False).astype(bool).sum()
                    ),
                    "transmit_unsupported_with_finite_model_epochs": int(
                        (
                            panel["transmit_pattern_unsupported"]
                            .fillna(False)
                            .astype(bool)
                            & panel["model_output_finite"]
                            .fillna(False)
                            .astype(bool)
                        ).sum()
                    ),
                }
            )
    audit = pd.DataFrame(rows).sort_values("panel_number").reset_index(drop=True)
    coverage_missing = audit["window_mode"].eq("coverage") & audit[
        "observed_1s_points"
    ].le(0)
    if coverage_missing.any():
        missing = audit.loc[
            coverage_missing,
            ["panel_marker", "signal_name", "window_start_utc"],
        ]
        raise RuntimeError(
            "One or more observation-count panels contain no observed 1-s C/N0 "
            "points:\n"
            f"{missing.to_string(index=False)}"
        )
    if audit["transmit_unsupported_with_finite_model_epochs"].gt(0).any():
        raise RuntimeError(
            "Transmit-pattern unsupported audit includes finite model output"
        )
    return audit


def load_detail_days_from_audit(path: Path) -> dict[str, pd.Timestamp]:
    """Load the two audited window starts for downstream rendering stages."""
    audit = pd.read_csv(path, low_memory=False)
    required = {"window_mode", "window_start", "window_stop"}
    missing = required.difference(audit.columns)
    if missing:
        raise RuntimeError(f"Window audit is missing columns: {sorted(missing)}")
    selected = audit[audit["window_mode"].isin(["model_count", "coverage"])].copy()
    if selected["window_mode"].nunique() != 2 or len(selected) != 2:
        raise RuntimeError("Window audit must contain one row for each detail mode")
    selected["window_start"] = pd.to_datetime(
        selected["window_start"], utc=True, errors="raise"
    )
    selected["window_stop"] = pd.to_datetime(
        selected["window_stop"], utc=True, errors="raise"
    )
    duration = selected["window_stop"] - selected["window_start"]
    if not duration.eq(pd.Timedelta(hours=24)).all():
        raise RuntimeError("Audited detail windows are not exactly 24 hours")
    selected = selected.sort_values("window_start", kind="stable")
    return {
        str(row.window_mode): pd.Timestamp(row.window_start)
        for row in selected.itertuples()
    }


def draw_background(ax: plt.Axes, panel: pd.DataFrame) -> None:
    for start, stop in majority_sidelobe_intervals(panel):
        ax.axvspan(start, stop, color=base.SIDELOBE_COLOR, alpha=0.38, lw=0, zorder=0)


def detail_state_intervals(
    panel: pd.DataFrame, state_name: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if panel.empty:
        return []
    state = (
        panel.sort_values("utc")
        .groupby("utc", as_index=True)["detail_background_state"]
        .agg(lambda values: values.iloc[np.argmax(
            pd.Series(values).map(
                {
                    "nominal": 0,
                    "side_lobe": 1,
                    "transmit_pattern_unsupported": 2,
                    "moon_occultation": 3,
                    "earth_occultation": 4,
                }
            ).to_numpy()
        )])
    )
    times = pd.Series(state.index[state.eq(state_name)])
    if times.empty:
        return []
    groups = times.diff().gt(MINUTE_SEGMENT_GAP).cumsum()
    return [
        (
            pd.Timestamp(group.iloc[0]) - pd.Timedelta(seconds=30),
            pd.Timestamp(group.iloc[-1]) + pd.Timedelta(seconds=30),
        )
        for _, group in times.groupby(groups)
    ]


def draw_detail_background(ax: plt.Axes, panel: pd.DataFrame) -> None:
    for state_name, (color, alpha, _) in DETAIL_STATE_STYLES.items():
        for start, stop in detail_state_intervals(panel, state_name):
            ax.axvspan(start, stop, color=color, alpha=alpha, lw=0, zorder=0)


def draw_observed_points(
    ax: plt.Axes,
    observations: pd.DataFrame,
    chosen: pd.DataFrame,
) -> None:
    if observations.empty:
        return
    for row in chosen.itertuples():
        points = observations[observations["svid"].eq(row.svid)].sort_values("rx_utc")
        if points.empty:
            continue
        ax.scatter(
            points["rx_utc"],
            points["cn0_observed_dbhz"],
            s=DETAIL_OBSERVATION_SIZE_PT2,
            marker="o",
            color=OBSERVATION_COLOR,
            edgecolors="none",
            alpha=DETAIL_OBSERVATION_ALPHA,
            rasterized=True,
            zorder=DETAIL_OBSERVATION_ZORDER,
        )


def observed_legend_handle() -> plt.Line2D:
    """Return a marker-only legend handle matching the detail observations."""
    return plt.Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=DETAIL_OBSERVATION_MARKER_SIZE_PT,
        markerfacecolor=OBSERVATION_COLOR,
        markeredgecolor="none",
        color=OBSERVATION_COLOR,
        alpha=DETAIL_OBSERVATION_ALPHA,
        label=r"Observed $C/N_0$",
    )


def build_lobe_audit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit lobe labels against the exact pattern value used by the model."""
    audited = data[data["display_eligible"]].copy()
    audited["pattern_side_lobe"] = audited["pattern_side_lobe"].fillna(False).astype(bool)
    audited["pattern_first_null_resolved"] = (
        audited["pattern_first_null_resolved"].fillna(False).astype(bool)
    )
    cn0 = pd.to_numeric(audited["cn0_physics_ai_trend_dbhz"], errors="coerce")
    margin = pd.to_numeric(audited["pattern_angular_margin_deg"], errors="coerce")
    relative_gain = pd.to_numeric(
        audited["tx_pattern_relative_to_main_peak_db"], errors="coerce"
    )
    audited["pattern_boundary_transition_2deg"] = margin.abs().le(2.0)
    audited["pattern_gain_depth_class"] = pd.cut(
        relative_gain,
        bins=[-np.inf, -15.0, -6.0, -3.0, np.inf],
        labels=[
            "deep_attenuation_gt15dB",
            "attenuated_6_to_15dB",
            "moderate_3_to_6dB",
            "near_main_peak_lt3dB",
        ],
    ).astype("string")
    audited["lobe_audit_case"] = np.select(
        [
            audited["pattern_label_gain_consistency"].ne(
                "consistent_with_same_pattern_value_used_by_model"
            ),
            audited["pattern_side_lobe"] & cn0.gt(40.0),
            ~audited["pattern_side_lobe"]
            & cn0.lt(15.0)
            & audited["pattern_boundary_transition_2deg"],
            ~audited["pattern_side_lobe"] & cn0.lt(15.0),
        ],
        [
            "label_gain_inconsistency_requires_review",
            "valid_sidelobe_high_cn0_check_short_range_link_budget",
            "valid_main_lobe_edge_low_cn0_near_first_null",
            "valid_main_lobe_low_cn0_due_pattern_and_link_budget",
        ],
        default="no_threshold_anomaly",
    )

    summary_rows: list[dict[str, object]] = []
    for signal_name, group in audited.groupby("signal_name", sort=False):
        group_cn0 = pd.to_numeric(group["cn0_physics_ai_trend_dbhz"], errors="coerce")
        summary_rows.append(
            {
                "signal_name": signal_name,
                "audited_minute_rows": len(group),
                "pattern_side_lobe_rows": int(group["pattern_side_lobe"].sum()),
                "pattern_main_lobe_rows": int((~group["pattern_side_lobe"]).sum()),
                "boundary_transition_rows_abs_margin_le_2deg": int(
                    group["pattern_boundary_transition_2deg"].sum()
                ),
                "side_lobe_cn0_gt_40_rows": int(
                    (group["pattern_side_lobe"] & group_cn0.gt(40.0)).sum()
                ),
                "main_lobe_cn0_lt_15_rows": int(
                    (~group["pattern_side_lobe"] & group_cn0.lt(15.0)).sum()
                ),
                "label_gain_inconsistency_rows": int(
                    group["lobe_audit_case"]
                    .eq("label_gain_inconsistency_requires_review")
                    .sum()
                ),
                "minimum_relative_pattern_gain_db": float(
                    pd.to_numeric(
                        group["tx_pattern_relative_to_main_peak_db"], errors="coerce"
                    ).min()
                ),
                "maximum_side_lobe_relative_pattern_gain_db": float(
                    pd.to_numeric(
                        group.loc[
                            group["pattern_side_lobe"],
                            "tx_pattern_relative_to_main_peak_db",
                        ],
                        errors="coerce",
                    ).max()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    details = audited[
        audited["lobe_audit_case"].ne("no_threshold_anomaly")
    ].copy()
    return summary, details


def draw_tracks(
    ax: plt.Axes,
    panel: pd.DataFrame,
    chosen: pd.DataFrame,
    with_labels: bool,
    value_column: str = "cn0_physics_ai_trend_dbhz",
    detail_style: bool = False,
) -> None:
    line_width = DETAIL_MODEL_LINEWIDTH_PT if detail_style else 1.15
    line_alpha = DETAIL_MODEL_ALPHA if detail_style else 0.96
    line_zorder = DETAIL_MODEL_ZORDER if detail_style else 3
    colors = [LINE_COLORS[index % len(LINE_COLORS)] for index in range(len(chosen))]
    for color, row in zip(colors, chosen.itertuples()):
        track = panel[panel["svid"].eq(row.svid)]
        first = True
        for segment in minute_segments(track, value_column=value_column):
            phase_groups = (
                segment.groupby("mission_phase", sort=False)
                if COLOR_BY_PHASE
                else [(None, segment)]
            )
            for phase, phase_segment in phase_groups:
                ax.plot(
                    phase_segment["utc"],
                    phase_segment[value_column],
                    color=PHASE_COLORS.get(str(phase), color),
                    lw=line_width,
                    alpha=line_alpha,
                    solid_capstyle="round",
                    label=(
                        f"SVID {int(row.svid)}"
                        if with_labels and first and not COLOR_BY_PHASE
                        else None
                    ),
                    zorder=line_zorder,
                )
            first = False


def satellite_label(signal: str, svid: int) -> str:
    """Return the compact constellation identifier used in the paper figure."""
    prefix = "G" if str(signal).startswith("GPS") else "E"
    return f"{prefix}{int(svid):02d}"


def plot(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    detail_days: dict[str, pd.Timestamp],
    detail_status: pd.DataFrame,
    observations_1s: pd.DataFrame,
) -> Path:
    base.configure_style()
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.4,
        }
    )
    detail_modes = sorted(
        [mode for mode in DETAIL_PANEL_MODES if mode in detail_days],
        key=lambda mode: pd.Timestamp(detail_days[mode]),
    )
    detail_windows = [
        (mode, detail_days[mode], detail_days[mode] + pd.Timedelta(days=1))
        for mode in detail_modes
    ]
    figure_width = 15.8 + max(0, len(detail_windows) - 1) * 2.7
    fig = plt.figure(figsize=(figure_width, 6.75))
    grid = fig.add_gridspec(
        4,
        1 + len(detail_windows),
        width_ratios=[4.8] + [1.35] * len(detail_windows),
        left=0.055,
        right=0.992,
        bottom=0.075,
        top=0.925,
        hspace=0.16,
        wspace=0.035,
    )
    main_axes = [fig.add_subplot(grid[row, 0]) for row in range(4)]
    detail_axes = [
        [fig.add_subplot(grid[row, column + 1]) for row in range(4)]
        for column in range(len(detail_windows))
    ]
    phases = base.phase_intervals(data)
    mission_start = data["utc"].min()
    mission_stop = data["utc"].max()

    for panel_index, signal in enumerate(base.SIGNALS):
        main_ax = main_axes[panel_index]
        panel = data[data["signal_name"].eq(signal)].copy()
        chosen = selected[selected["signal_name"].eq(signal)].sort_values(
            "rank_within_constellation"
        )

        draw_background(main_ax, panel)
        window_colors = [ZOOM_COLOR, ZOOM_COLOR]
        window_styles = [(0, (4, 2)), (0, (4, 2))]
        for window_index, (_, detail_start, detail_stop) in enumerate(detail_windows):
            window_style = window_styles[window_index % len(window_styles)]
            _, window_label = detail_panel_marker(panel_index, window_index)
            main_ax.axvspan(
                detail_start,
                detail_stop,
                color=window_colors[window_index % len(window_colors)],
                alpha=0.16,
                lw=0,
                zorder=4,
            )
            detail_start_num = mdates.date2num(detail_start.to_pydatetime())
            detail_stop_num = mdates.date2num(detail_stop.to_pydatetime())
            main_ax.add_patch(
                Rectangle(
                    (detail_start_num, 0.0),
                    detail_stop_num - detail_start_num,
                    1.0,
                    transform=main_ax.get_xaxis_transform(),
                    fill=False,
                    edgecolor="#111111",
                    linewidth=1.25,
                    linestyle=window_style,
                    alpha=0.98,
                    zorder=8,
                    clip_on=False,
                )
            )
            main_ax.text(
                detail_stop - pd.Timedelta(hours=1.2),
                0.955,
                window_label,
                transform=main_ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=12.0,
                fontweight="bold",
                fontfamily="DejaVu Sans",
                color="#111111",
                bbox={
                    "boxstyle": "square,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
                clip_on=False,
                zorder=9,
            )
        draw_tracks(main_ax, panel, chosen, with_labels=True)

        for window_index, (_, detail_start, detail_stop) in enumerate(detail_windows):
            detail_ax = detail_axes[window_index][panel_index]
            detail = panel[
                panel["utc"].between(detail_start, detail_stop, inclusive="left")
            ].copy()
            status_detail = detail_status[
                detail_status["signal_name"].eq(signal)
                & detail_status["utc"].between(
                    detail_start, detail_stop, inclusive="left"
                )
            ].copy()
            observed_detail = observations_1s[
                observations_1s["signal_name"].eq(signal)
                & observations_1s["rx_utc"].between(
                    detail_start, detail_stop, inclusive="left"
                )
            ].copy()
            draw_detail_background(detail_ax, status_detail)
            draw_observed_points(detail_ax, observed_detail, chosen)
            draw_tracks(
                detail_ax,
                detail,
                chosen,
                with_labels=False,
                detail_style=True,
            )

            detail_values = pd.to_numeric(
                detail["cn0_physics_ai_trend_dbhz"], errors="coerce"
            )
            detail_usable = detail["display_eligible"]
            if OMIT_BELOW_LOCK_THRESHOLD:
                detail_usable &= detail_values.ge(ASSUMED_LOSS_OF_LOCK_DBHZ)
            actual = detail_values.loc[detail_usable].dropna()
            observed_values = pd.to_numeric(
                observed_detail["cn0_observed_dbhz"], errors="coerce"
            ).dropna()
            if not observed_values.empty:
                actual = pd.concat([actual, observed_values], ignore_index=True)
            if actual.empty:
                detail_min, detail_max = base.DISPLAY_MIN_DBHZ, base.DISPLAY_MAX_DBHZ
            else:
                detail_min = float(actual.min())
                detail_max = float(actual.max())
                span = max(detail_max - detail_min, 2.0)
                padding = max(0.6, 0.06 * span)
                detail_min -= padding
                detail_max += padding
            detail_ax.set_ylim(detail_min, detail_max)
            if detail_min <= ASSUMED_LOSS_OF_LOCK_DBHZ <= detail_max:
                detail_ax.axhline(
                    ASSUMED_LOSS_OF_LOCK_DBHZ,
                    color="#6F777D",
                    lw=0.55,
                    ls=(0, (3, 2)),
                    alpha=0.8,
                    zorder=2,
                )
            detail_ax.grid(True, color="#D8DDE2", lw=0.45, alpha=0.82, zorder=0)
            detail_ax.set_xlim(detail_start, detail_stop)
            detail_ax.tick_params(labelleft=True, labelsize=8.0)
            detail_ax.xaxis.set_major_locator(
                mdates.HourLocator(byhour=[0, 6, 12, 18])
            )
            detail_ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%d\n%H:%M", tz=detail_start.tz)
            )
            detail_ax.text(
                0.025,
                0.955,
                detail_panel_marker(panel_index, window_index)[1],
                transform=detail_ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                fontfamily="DejaVu Sans",
                color="#111111",
                zorder=8,
            )
        for phase, start, _ in phases[1:]:
            main_ax.axvline(
                start, color="#8B9298", lw=0.42, ls=(0, (2, 2)), alpha=0.62, zorder=1
            )

        main_ax.set_ylim(*MAIN_Y_LIMITS)
        if MAIN_Y_LIMITS[0] >= 10:
            main_ax.set_yticks(
                [tick for tick in [10, 20, 30, 40] if MAIN_Y_LIMITS[0] <= tick <= MAIN_Y_LIMITS[1]]
            )
        else:
            main_ax.set_yticks([-20, 0, 10, 20, 40])
        if MAIN_Y_LIMITS[0] <= ASSUMED_LOSS_OF_LOCK_DBHZ <= MAIN_Y_LIMITS[1]:
            main_ax.axhline(
                ASSUMED_LOSS_OF_LOCK_DBHZ,
                color="#6F777D",
                lw=0.55,
                ls=(0, (3, 2)),
                alpha=0.8,
                zorder=2,
            )
        main_ax.grid(True, color="#D8DDE2", lw=0.45, alpha=0.82, zorder=0)
        main_ax.set_ylabel(r"$C/N_0$ (dB-Hz)")
        title = base.DISPLAY_NAMES[signal]
        if COLOR_BY_PHASE and len(chosen) == 1:
            title += f" | {satellite_label(signal, int(chosen.iloc[0]['svid']))}"
        main_ax.set_title(title, loc="left", fontsize=10.0, fontweight="bold", pad=2)
        if not COLOR_BY_PHASE:
            main_ax.legend(
                loc="upper right", ncol=3, handlelength=2.1, columnspacing=1.05,
                borderaxespad=0.35,
            )
        main_ax.text(
            -0.025, 1.008, chr(ord("a") + panel_index), transform=main_ax.transAxes,
            fontsize=10.5, fontweight="bold", va="bottom",
        )

        main_ax.set_xlim(mission_start, mission_stop)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
    for ax in main_axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M", tz=mission_start.tz)
        )
    non_bottom_detail_axes = [ax for column in detail_axes for ax in column[:-1]]
    for ax in main_axes[:-1] + non_bottom_detail_axes:
        ax.tick_params(labelbottom=False)
    main_axes[-1].set_xlabel("UTC")
    for window_index, _ in enumerate(detail_windows):
        detail_axes[window_index][-1].set_xlabel("UTC")

    main_legend_handles = []
    if COLOR_BY_PHASE:
        main_legend_handles.extend(
            plt.Line2D([], [], color=PHASE_COLORS[phase], lw=1.5, label=f"{phase} phase")
            for phase in ["C", "T", "L", "S"]
        )
    else:
        main_legend_handles.append(
            plt.Line2D(
                [], [], color=LINE_COLORS[0], lw=1.1,
                label="Physics + AI residual model output",
            )
        )
    detail_legend_handles = [observed_legend_handle()]
    detail_legend_handles.extend(
        Patch(
            facecolor=DETAIL_STATE_STYLES[state_name][0],
            edgecolor="none",
            alpha=DETAIL_STATE_STYLES[state_name][1],
            label=DETAIL_STATE_STYLES[state_name][2],
        )
        for state_name in VISIBLE_DETAIL_LEGEND_STATES
    )
    fig.legend(
        handles=main_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.355, 0.982),
        ncol=len(main_legend_handles),
        frameon=True,
        fancybox=False,
        edgecolor="#697077",
        facecolor="white",
        framealpha=1.0,
        handlelength=1.7,
        handleheight=0.75,
        columnspacing=1.15,
        fontsize=7.5,
    )
    fig.legend(
        handles=detail_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.823, 0.982),
        ncol=len(detail_legend_handles),
        frameon=True,
        fancybox=False,
        edgecolor="#697077",
        facecolor="white",
        framealpha=1.0,
        handlelength=1.7,
        handleheight=0.75,
        columnspacing=1.05,
        fontsize=7.5,
    )

    base.FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_base = base.FIG_DIR / STEM
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_base.with_suffix(".png")


def main() -> None:
    base.SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if not common_builder.OUTPUT_PATH.exists():
        common_builder.main()
    predicted = base.load_and_predict()
    selected_raw, selected = select_three_common_satellites(predicted)
    selected_raw = add_pattern_lobe_classification(selected_raw)
    detail_status = build_detail_status(selected_raw)
    minute_data = interpolate_to_one_minute(selected_raw)
    minute_data["mission_phase"] = minute_data["utc"].map(
        common_builder.builder.mission_phase
    )
    minute_data["assumed_tracking_lock"] = pd.to_numeric(
        minute_data["cn0_physics_ai_trend_dbhz"], errors="coerce"
    ).ge(ASSUMED_LOSS_OF_LOCK_DBHZ)
    minute_data["assumed_loss_of_lock"] = (
        minute_data["display_eligible"] & ~minute_data["assumed_tracking_lock"]
    )
    audit_summary, audit_details = build_lobe_audit(minute_data)
    observation_coverage = load_minute_observation_coverage(selected)
    detail_days, daily_scores = choose_detail_days(
        minute_data, observation_coverage=observation_coverage
    )
    window_audit = build_detail_window_audit(daily_scores, detail_days)
    observations_1s = load_one_second_observations(selected, detail_days)
    chronological_panel_audit = build_chronological_panel_audit(
        window_audit,
        detail_days,
        detail_status,
        observations_1s,
    )
    minute_data["in_most_data_24h_window"] = minute_data["utc"].between(
        detail_days["coverage"],
        detail_days["coverage"] + pd.Timedelta(days=1),
        inclusive="left",
    )
    minute_data["in_most_model_output_24h_window"] = minute_data["utc"].between(
        detail_days["model_count"],
        detail_days["model_count"] + pd.Timedelta(days=1),
        inclusive="left",
    )
    minute_data.to_csv(
        base.SOURCE_DIR / f"{STEM}_source_data.csv", index=False, encoding="utf-8-sig"
    )
    selected.to_csv(
        base.SOURCE_DIR / f"{STEM}_satellite_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily_scores.to_csv(
        base.SOURCE_DIR / f"{STEM}_detail_window_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    window_audit.to_csv(
        base.SOURCE_DIR / f"{STEM}_window_selection_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    chronological_panel_audit.to_csv(
        base.SOURCE_DIR / f"{STEM}_chronological_panel_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit_summary.to_csv(
        base.SOURCE_DIR / f"{STEM}_lobe_audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit_details.to_csv(
        base.SOURCE_DIR / f"{STEM}_lobe_audit_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail_status.to_csv(
        base.SOURCE_DIR / f"{STEM}_detail_anomaly_timeline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    observations_1s.to_csv(
        base.SOURCE_DIR / f"{STEM}_observed_1s_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    output = plot(
        minute_data,
        selected,
        detail_days,
        detail_status,
        observations_1s,
    )
    print(selected.to_string(index=False))
    print(f"model_count_detail_start={detail_days['model_count']}")
    print(f"coverage_detail_start={detail_days['coverage']}")
    print(f"detail_observation_rows={len(observations_1s):,}")
    print(f"source_rows={len(minute_data):,}")
    print(output)


if __name__ == "__main__":
    main()
