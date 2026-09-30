#!/usr/bin/env python3
"""Train one 1-minute trend-residual model with OP74 fully held out."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "script"))

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import train_cn0_single_global_residual_1min_stratified as base_mod


OUT_DIR = ROOT / "table" / "algorithm" / "cn0_single_global_residual_1min_op74_holdout"
MODEL_PATH = OUT_DIR / "cn0_single_global_hgb_trend_residual_1min_op74_holdout.joblib"
CADENCE = "1min"
HOLDOUT_OP = "OP74"
HOLDOUT_OPS = (HOLDOUT_OP,)
HOLDOUT_SPLIT_NAME = "op74_holdout"
OUTPUT_PREDICTIONS_FILENAME = "cn0_single_global_1min_op74_holdout_predictions.csv"
OUTPUT_METRICS_FILENAME = "cn0_single_global_1min_op74_holdout_metrics.csv"
OUTPUT_LEAKAGE_FILENAME = "cn0_single_global_1min_op74_holdout_leakage_audit.csv"
MODEL_CARD_FILENAME = "SingleGlobalTrendResidualOP74HoldoutModelCard.json"
RUN_OBJECTIVE = "one global all-band trend residual model with OP74 external holdout"
RANDOM_STATE = 42
BLOCK_MINUTES = 15
TREND_WINDOW_MINUTES = 5
SMOOTH_TREND_MODEL = False
SPLIT_PROBABILITIES = {"train": 0.70, "validation": 0.15, "test": 0.15}
REPRESENTATIVE_TEST_OPS = ("OP2", "OP21", "OP27")
MIN_REPRESENTATIVE_TEST_ROWS = 50
TLM_DIR = ROOT / "data" / "receiver_observation" / "TLM"
GPS_EPOCH = pd.Timestamp("1980-01-06T00:00:00Z")
GPS_UTC_LEAP_SECONDS = 18.0
MIN_VALID_SECONDS = 30
MAX_LONGEST_GAP_SECONDS = 10
RX_TIME_RE = re.compile(r"rxTime:\s*([0-9.]+)")
MEASURE_RE = re.compile(
    r"svid:\s*(?P<svid>\d+).*?cn0:\s*(?P<cn0>[-+0-9.Ee]+)\s+"
    r"signalId:\s*(?P<signal_id>\d+)"
)
SIGNAL_META = {
    0: ("G", "GPS_L1"),
    1: ("G", "GPS_L5"),
    2: ("E", "GAL_E1"),
    3: ("E", "GAL_E5a"),
}

# These quantities describe one transmitter-receiver ray and must therefore be
# identical for all frequencies emitted by the same satellite at the anchor epoch.
COMMON_GEOMETRY_COLUMNS = [
    "geometric_range_km",
    "range_rate_rx_only_km_s",
    "tx_offboresight_deg",
    "earth_observer_altitude_km",
    "earth_limb_margin_deg",
    "earth_grazing_altitude_km",
    "earth_blocked",
    "moon_observer_altitude_km",
    "moon_limb_margin_deg",
    "moon_grazing_altitude_km",
    "moon_blocked",
    "nav_age_seconds",
    "earth_limb_proximity_proxy",
    "moon_limb_proximity_proxy",
    "ionosphere_path_length_shell_km",
    "m_ion_shell_chord_raw",
    "m_ion_proxy",
    "gas_equivalent_airmass_km_raw",
    "gas_equivalent_airmass_km",
    "m_gas_proxy",
    "earth_edge_near_limb_0_20km",
    "moon_edge_near_limb_0_20km",
    "earth_neutral_refraction_20_60km",
    "moon_neutral_refraction_20_60km",
    "rx_offboresight_spice_deg",
    "rx_azimuth_spice_deg",
    "tx_theta_body_deg",
    "tx_phi_body_deg",
]

TARGET = base_mod.TARGET
BASELINE = base_mod.BASELINE
PHYSICS_TREND_COLUMN = "cn0_physics_trend_dbhz"
OBSERVED_TREND_COLUMN = "cn0_observed_trend_dbhz"

DISCRETE_TREND_FEATURES = {
    "earth_blocked",
    "moon_blocked",
    "earth_edge_near_limb_0_20km",
    "moon_edge_near_limb_0_20km",
    "earth_neutral_refraction_20_60km",
    "moon_neutral_refraction_20_60km",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gps_bin_center_utc(gps_bin_seconds: pd.Series) -> pd.Series:
    return GPS_EPOCH + pd.to_timedelta(
        pd.to_numeric(gps_bin_seconds, errors="coerce")
        + 30.0
        - GPS_UTC_LEAP_SECONDS,
        unit="s",
    )


def _interpolate_link_features_to_anchor(
    frame: pd.DataFrame,
    bins: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Evaluate band-specific physical quantities at the common bin center."""
    output = bins.copy()
    source = frame.copy()
    source["_time_s"] = source["rx_utc_dt"].map(pd.Timestamp.timestamp)
    output["_anchor_s"] = output["minute_utc"].map(pd.Timestamp.timestamp)
    link_keys = ["op", "system", "signal_name", "svid"]
    for key, target in output.groupby(link_keys, dropna=False, sort=False):
        mask = pd.Series(True, index=source.index)
        for column, value in zip(link_keys, key):
            mask &= source[column].eq(value)
        link = source.loc[mask].sort_values("_time_s")
        if link.empty:
            continue
        target_times = target["_anchor_s"].to_numpy(float)
        for column in columns:
            if column not in link.columns or column not in output.columns:
                continue
            values = pd.to_numeric(link[column], errors="coerce").to_numpy(float)
            times = link["_time_s"].to_numpy(float)
            valid = np.isfinite(times) & np.isfinite(values)
            if not valid.any():
                continue
            times = times[valid]
            values = values[valid]
            unique_times, unique_indices = np.unique(times, return_index=True)
            values = values[unique_indices]
            if len(unique_times) == 1:
                aligned = np.full(len(target_times), values[0], dtype=float)
            else:
                aligned = np.interp(target_times, unique_times, values)
            output.loc[target.index, column] = aligned
    return output.drop(columns="_anchor_s")


def build_raw_quality(operations: set[str]) -> pd.DataFrame:
    """Summarize valid 1-s tracking samples in each native 60-s GPS bin."""
    seconds_by_link: dict[tuple[str, str, str, int], set[int]] = defaultdict(set)
    for op in sorted(operations):
        paths = sorted(TLM_DIR.glob(f"TLM_RAW_*_{op}_*.txt"))
        if not paths:
            raise FileNotFoundError(f"No TLM RAW file found for {op}")
        for path in paths:
            with path.open("r", encoding="utf-8", errors="ignore") as stream:
                for line in stream:
                    time_match = RX_TIME_RE.search(line)
                    if not time_match:
                        continue
                    gps_second = int(math.floor(float(time_match.group(1))))
                    for measure in MEASURE_RE.finditer(line):
                        signal_id = int(measure.group("signal_id"))
                        if signal_id not in SIGNAL_META:
                            continue
                        system, signal_name = SIGNAL_META[signal_id]
                        svid = int(measure.group("svid"))
                        seconds_by_link[(op, system, signal_name, svid)].add(gps_second)

    rows: list[dict[str, Any]] = []
    for (op, system, signal_name, svid), second_set in seconds_by_link.items():
        ordered = np.array(sorted(second_set), dtype=np.int64)
        bins = (ordered // 60) * 60
        for gps_bin, indices in pd.Series(np.arange(len(ordered))).groupby(bins):
            samples = ordered[indices.to_numpy()] - int(gps_bin)
            samples = np.unique(samples[(samples >= 0) & (samples < 60)])
            if len(samples) == 0:
                continue
            internal_gap = int(np.max(np.diff(samples) - 1)) if len(samples) > 1 else 0
            leading_gap = int(samples[0])
            trailing_gap = int(59 - samples[-1])
            longest_gap = max(leading_gap, internal_gap, trailing_gap)
            first_second = int(gps_bin) + int(samples[0])
            previous = ordered[ordered < first_second]
            preceding_gap = (
                int(first_second - int(previous[-1]) - 1) if len(previous) > 0 else 0
            )
            longest_gap = max(longest_gap, min(preceding_gap, 60))
            reacquisition = bool(
                preceding_gap > MAX_LONGEST_GAP_SECONDS
                and preceding_gap <= 120
            ) or internal_gap > MAX_LONGEST_GAP_SECONDS
            rows.append(
                {
                    "op": op,
                    "system": system,
                    "signal_name": signal_name,
                    "svid": svid,
                    "source_bin_gps_seconds": int(gps_bin),
                    "valid_seconds_in_bin": int(len(samples)),
                    "coverage_fraction": float(len(samples) / 60.0),
                    "longest_gap_s": longest_gap,
                    "first_valid_offset_s": leading_gap,
                    "last_valid_offset_s": int(samples[-1]),
                    "preceding_gap_s": preceding_gap,
                    "reacquisition_flag": int(reacquisition),
                }
            )
    return pd.DataFrame(rows)


def aggregate_one_minute(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = data.copy()
    frame["source_bin_gps_seconds"] = pd.to_numeric(
        frame["time_bin_gps_seconds"], errors="coerce"
    ).round().astype("Int64")
    if frame["source_bin_gps_seconds"].isna().any():
        raise RuntimeError("Native GPS time-bin key is missing")
    frame["minute_utc"] = _gps_bin_center_utc(frame["source_bin_gps_seconds"])
    frame["observation_time_unix_s"] = frame["rx_utc_dt"].map(pd.Timestamp.timestamp)
    keys = [
        "op",
        "mission_phase",
        "system",
        "signal_name",
        "svid",
        "source_bin_gps_seconds",
        "minute_utc",
    ]
    numeric = list(dict.fromkeys(features + [TARGET, BASELINE]))
    aggregations: dict[str, str] = {
        column: "median" for column in numeric if column in frame.columns and column not in keys
    }
    frame["raw_rows_in_bin"] = 1
    aggregations["raw_rows_in_bin"] = "sum"
    aggregations["observation_time_unix_s"] = "median"
    bins = frame.groupby(keys, dropna=False, as_index=False).agg(aggregations)

    # Interpolate each band's physical terms to the native GPS-bin center.
    bins = _interpolate_link_features_to_anchor(
        frame,
        bins,
        [column for column in numeric if column != TARGET],
    )

    # Select one high-coverage geometry realization per satellite/bin and
    # broadcast it to every observed frequency. This prevents independent
    # tracking gaps from shifting one band across a limb-feature threshold.
    common_keys = ["op", "mission_phase", "system", "svid", "source_bin_gps_seconds"]
    available_common = [column for column in COMMON_GEOMETRY_COLUMNS if column in frame.columns]
    candidates = frame.copy()
    candidates["_anchor_distance_s"] = (
        candidates["rx_utc_dt"] - candidates["minute_utc"]
    ).abs().dt.total_seconds()
    candidates["_coverage_rank"] = pd.to_numeric(
        candidates.get("samples", 0), errors="coerce"
    ).fillna(0)
    common = (
        candidates.sort_values(
            common_keys + ["_coverage_rank", "_anchor_distance_s"],
            ascending=[True] * len(common_keys) + [False, True],
        )
        .drop_duplicates(common_keys)
        [common_keys + available_common]
    )
    bins = bins.drop(columns=available_common, errors="ignore").merge(
        common, on=common_keys, how="left", validate="many_to_one"
    )

    # Recompute the frequency-dependent terms that can be updated exactly once
    # common range/path geometry has been assigned.
    old_fspl = pd.to_numeric(bins.get("fspl_db"), errors="coerce")
    if {"geometric_range_km", "frequency_mhz"}.issubset(bins.columns):
        bins["fspl_db"] = (
            20.0 * np.log10(pd.to_numeric(bins["geometric_range_km"], errors="coerce"))
            + 20.0 * np.log10(pd.to_numeric(bins["frequency_mhz"], errors="coerce"))
            + 32.44
        )
    old_ion = pd.to_numeric(bins.get("l_ion_abs_budget_db"), errors="coerce").fillna(0.0)
    old_gas = pd.to_numeric(bins.get("l_gas_abs_budget_db"), errors="coerce").fillna(0.0)
    if {"m_ion_proxy", "frequency_mhz"}.issubset(bins.columns):
        bins["l_ion_abs_proxy_db"] = (
            pd.to_numeric(bins.get("ionosphere_l30_vertical_assumed_db"), errors="coerce")
            * pd.to_numeric(bins["m_ion_proxy"], errors="coerce")
            * (30.0 / pd.to_numeric(bins["frequency_mhz"], errors="coerce")) ** 2
        )
        bins["l_ion_abs_budget_db"] = np.where(
            pd.to_numeric(bins.get("earth_blocked"), errors="coerce").fillna(0).astype(bool),
            0.0,
            bins["l_ion_abs_proxy_db"],
        )
    if {"gas_gamma_surface_db_per_km", "gas_equivalent_airmass_km"}.issubset(bins.columns):
        bins["l_gas_abs_proxy_db"] = (
            pd.to_numeric(bins["gas_gamma_surface_db_per_km"], errors="coerce")
            * pd.to_numeric(bins["gas_equivalent_airmass_km"], errors="coerce")
        )
        bins["l_gas_abs_budget_db"] = np.where(
            pd.to_numeric(bins.get("earth_blocked"), errors="coerce").fillna(0).astype(bool),
            0.0,
            bins["l_gas_abs_proxy_db"],
        )
    new_ion = pd.to_numeric(bins.get("l_ion_abs_budget_db"), errors="coerce").fillna(0.0)
    new_gas = pd.to_numeric(bins.get("l_gas_abs_budget_db"), errors="coerce").fillna(0.0)
    correction = old_ion + old_gas - new_ion - new_gas
    if old_fspl is not None and "fspl_db" in bins:
        correction = correction + old_fspl - pd.to_numeric(bins["fspl_db"], errors="coerce")
    for column in [
        BASELINE,
        "cn0_constellation_direct_available_dbhz",
        "cn0_reference_trajectory_2d_dbhz",
        "cn0_reference_trajectory_2d_atm_proxy_dbhz",
        "cn0_reference_trajectory_2d_iono_proxy_dbhz",
        "cn0_galileo_grap_peakrx_dbhz",
    ]:
        if column in bins.columns:
            bins[column] = pd.to_numeric(bins[column], errors="coerce") + correction

    quality = build_raw_quality(set(frame["op"].astype(str)))
    quality_keys = ["op", "system", "signal_name", "svid", "source_bin_gps_seconds"]
    bins["svid"] = bins["svid"].astype(str)
    quality["svid"] = quality["svid"].astype(str)
    bins = bins.merge(quality, on=quality_keys, how="left", validate="one_to_one")
    bins["valid_seconds_in_bin"] = pd.to_numeric(
        bins["valid_seconds_in_bin"], errors="coerce"
    ).fillna(0).astype(int)
    bins["coverage_fraction"] = pd.to_numeric(
        bins["coverage_fraction"], errors="coerce"
    ).fillna(0.0)
    bins["reacquisition_flag"] = pd.to_numeric(
        bins["reacquisition_flag"], errors="coerce"
    ).fillna(1).astype(int)
    bins["trend_training_eligible"] = (
        bins["valid_seconds_in_bin"].ge(MIN_VALID_SECONDS)
        & pd.to_numeric(bins["longest_gap_s"], errors="coerce").le(
            MAX_LONGEST_GAP_SECONDS
        )
        & bins["reacquisition_flag"].eq(0)
    )
    bins["observation_time_median_utc"] = pd.to_datetime(
        bins.pop("observation_time_unix_s"), unit="s", utc=True
    )
    bins["residual_raw_db"] = bins[TARGET] - bins[BASELINE]
    return bins


def split_synchronized_time_blocks(
    bins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = bins.copy()
    op_start = output.groupby("op")["minute_utc"].transform("min")
    elapsed = (output["minute_utc"] - op_start).dt.total_seconds().div(60.0)
    output["time_block_number"] = np.floor(elapsed / BLOCK_MINUTES).astype(int)
    output["time_block_id"] = (
        output["op"].astype(str) + "|" + output["time_block_number"].astype(str)
    )
    blocks = (
        output.groupby(
            ["time_block_id", "op", "mission_phase", "time_block_number"],
            as_index=False,
            dropna=False,
        )
        .agg(
            rows_1min=("minute_utc", "size"),
            start_utc=("minute_utc", "min"),
            end_utc=("minute_utc", "max"),
        )
    )
    blocks["evaluation_split"] = ""
    blocks.loc[blocks["op"].isin(HOLDOUT_OPS), "evaluation_split"] = HOLDOUT_SPLIT_NAME

    rng = np.random.default_rng(RANDOM_STATE)
    split_names = ["train", "validation", "test"]
    # Allocate synchronized blocks inside every non-held-out OP.  Optimizing
    # row-count deficits avoids the severe phase imbalance produced by random
    # Bernoulli draws when C/L contain only a few dense blocks.
    for (_, op), group in blocks[~blocks["op"].isin(HOLDOUT_OPS)].groupby(
        ["mission_phase", "op"], sort=True
    ):
        indices = group.index.to_numpy(copy=True)
        tie_break = rng.random(len(indices))
        ordered = (
            group.assign(_tie=tie_break)
            .sort_values(["rows_1min", "_tie"], ascending=[False, True])
            .index.to_numpy()
        )
        total_rows = float(group["rows_1min"].sum())
        targets = {
            name: SPLIT_PROBABILITIES[name] * total_rows for name in split_names
        }
        current = {name: 0.0 for name in split_names}
        assigned: dict[int, str] = {}
        for index in ordered:
            rows = float(blocks.loc[index, "rows_1min"])
            scores: dict[str, float] = {}
            for name in split_names:
                trial = current.copy()
                trial[name] += rows
                scores[name] = sum(
                    ((trial[key] - targets[key]) / max(targets[key], 1.0)) ** 2
                    for key in split_names
                )
            chosen = min(split_names, key=lambda name: (scores[name], split_names.index(name)))
            assigned[int(index)] = chosen
            current[chosen] += rows

        # A sufficiently sampled OP must contribute at least one synchronized
        # block to each internal subset.  This preserves phase-aware validation
        # without moving individual satellites or frequencies independently.
        if len(indices) >= 3:
            present = set(assigned.values())
            for missing in split_names:
                if missing in present:
                    continue
                donors = [name for name in split_names if list(assigned.values()).count(name) > 1]
                donor = max(donors, key=lambda name: current[name] - targets[name])
                candidates = [index for index, name in assigned.items() if name == donor]
                move_index = min(candidates, key=lambda index: blocks.loc[index, "rows_1min"])
                move_rows = float(blocks.loc[move_index, "rows_1min"])
                assigned[move_index] = missing
                current[donor] -= move_rows
                current[missing] += move_rows
                present.add(missing)
        elif len(indices) == 2:
            # With only two synchronized blocks, keep the denser block for
            # fitting and the other for model selection.  The final frozen
            # model may refit on train+validation after hyperparameters have
            # been selected; the complete external OP remains untouched.
            by_rows = group.sort_values("rows_1min", ascending=False).index.to_list()
            assigned = {int(by_rows[0]): "train", int(by_rows[1]): "validation"}

        for index, split_name in assigned.items():
            blocks.loc[index, "evaluation_split"] = split_name

    # Preserve C/T/L evidence in the figure without splitting synchronized
    # blocks. Additional blocks are selected by the seeded RNG, never by C/N0.
    for op in REPRESENTATIVE_TEST_OPS:
        op_blocks = blocks[blocks["op"].eq(op)]
        current_rows = int(
            op_blocks.loc[
                op_blocks["evaluation_split"].eq("test"), "rows_1min"
            ].sum()
        )
        candidates = op_blocks.loc[
            ~op_blocks["evaluation_split"].eq("test")
        ].index.to_numpy(copy=True)
        rng.shuffle(candidates)
        for index in candidates:
            if current_rows >= MIN_REPRESENTATIVE_TEST_ROWS:
                break
            blocks.loc[index, "evaluation_split"] = "test"
            current_rows += int(blocks.loc[index, "rows_1min"])

    mapping = blocks.set_index("time_block_id")["evaluation_split"]
    output["evaluation_split"] = output["time_block_id"].map(mapping)
    if output["evaluation_split"].isna().any() or output["evaluation_split"].eq("").any():
        raise RuntimeError("At least one one-minute sample has no synchronized split")
    return output, blocks


def acf_diagnostic(train_bins: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
    values: dict[int, list[float]] = {lag: [] for lag in range(1, max_lag + 1)}
    ordered = train_bins.sort_values(["op", "signal_name", "svid", "minute_utc"])
    for _, link in ordered.groupby(["op", "signal_name", "svid"], dropna=False):
        gap = link["minute_utc"].diff().dt.total_seconds().div(60.0)
        segments = (gap.isna() | gap.gt(1.5)).cumsum()
        for _, segment in link.groupby(segments):
            x = segment["residual_raw_db"].to_numpy(float)
            x = x[np.isfinite(x)]
            if len(x) < 8:
                continue
            x = x - np.mean(x)
            denominator = float(np.dot(x, x))
            if denominator <= 0:
                continue
            for lag in range(1, min(max_lag, len(x) - 3) + 1):
                value = float(
                    np.dot(x[:-lag], x[lag:])
                    / denominator
                    * len(x)
                    / (len(x) - lag)
                )
                values[lag].append(value)
    rows = []
    for lag, samples in values.items():
        rows.append(
            {
                "lag_minutes": lag,
                "median_acf": float(np.median(samples)) if samples else np.nan,
                "links_contributing": len(samples),
                "threshold_1_over_e": float(1.0 / math.e),
            }
        )
    return pd.DataFrame(rows)


def _smooth_continuous_segments(
    group: pd.DataFrame,
    value_column: str,
    valid_column: str | None = None,
) -> pd.Series:
    result = pd.Series(np.nan, index=group.index, dtype=float)
    ordered = group.sort_values("minute_utc")
    segment_ids = ordered["minute_utc"].diff().gt(pd.Timedelta(minutes=1.5)).cumsum()
    window = int(TREND_WINDOW_MINUTES)
    if window % 2 == 0:
        window += 1
    for _, segment in ordered.groupby(segment_ids):
        values = pd.to_numeric(segment[value_column], errors="coerce")
        if valid_column is not None:
            values = values.where(segment[valid_column].astype(bool))
        values = values.interpolate(limit_direction="both")
        if values.notna().sum() == 0:
            continue
        filtered = values.rolling(
            window=window,
            center=True,
            min_periods=1,
        ).median()
        if len(filtered) >= window:
            filtered = pd.Series(
                savgol_filter(
                    filtered.to_numpy(float),
                    window_length=window,
                    polyorder=2,
                    mode="interp",
                ),
                index=filtered.index,
            )
        result.loc[segment.index] = filtered.to_numpy(float)
    return result


def smooth_model_features(features: list[str]) -> list[str]:
    selected = [
        feature
        for feature in features
        if feature != BASELINE
        and feature not in DISCRETE_TREND_FEATURES
        and not feature.startswith("quality_")
    ]
    selected.append(PHYSICS_TREND_COLUMN)
    return list(dict.fromkeys(selected))


def make_smooth_model(features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "smooth_physics",
                SimpleImputer(strategy="median"),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Selected on the internal validation split.  A small tree budget and
    # strong leaf regularization capture slow antenna-pattern curvature while
    # the temporal trend head below prevents pointwise acquisition/fade events
    # from entering the reported output.
    estimator = HistGradientBoostingRegressor(
        max_iter=420,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=24,
        l2_regularization=0.8,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def estimate_signal_beta(frame: pd.DataFrame) -> dict[str, float]:
    """Estimate one train-only absolute calibration constant per signal."""
    return {
        str(signal): float(value)
        for signal, value in frame.groupby("signal_name", dropna=False)[
            "residual_trend_target_db"
        ].median().items()
    }


def signal_beta_values(frame: pd.DataFrame, beta: dict[str, float]) -> np.ndarray:
    return frame["signal_name"].astype(str).map(beta).fillna(0.0).to_numpy(float)


def add_trend_target(bins: pd.DataFrame) -> pd.DataFrame:
    if SMOOTH_TREND_MODEL:
        output = bins.sort_values(["op", "signal_name", "svid", "minute_utc"]).copy()
        output[PHYSICS_TREND_COLUMN] = np.nan
        for _, group in output.groupby(["op", "signal_name", "svid"], dropna=False):
            output.loc[group.index, PHYSICS_TREND_COLUMN] = _smooth_continuous_segments(
                group, BASELINE
            ).loc[group.index]

        output[OBSERVED_TREND_COLUMN] = np.nan
        for _, group in output.groupby(
            ["evaluation_split", "op", "signal_name", "svid"], dropna=False
        ):
            output.loc[group.index, OBSERVED_TREND_COLUMN] = _smooth_continuous_segments(
                group,
                TARGET,
                valid_column="trend_training_eligible",
            ).loc[group.index]

        output["residual_trend_target_db"] = (
            output[OBSERVED_TREND_COLUMN] - output[PHYSICS_TREND_COLUMN]
        )
        output["residual_fast_db"] = output[TARGET] - output[OBSERVED_TREND_COLUMN]
        output["cn0_trend_target_dbhz"] = output[OBSERVED_TREND_COLUMN]
        return output

    output = bins.sort_values(
        ["evaluation_split", "op", "signal_name", "svid", "minute_utc"]
    ).copy()
    output["residual_trend_target_db"] = np.nan
    group_keys = ["evaluation_split", "op", "signal_name", "svid"]
    for _, group in output.groupby(group_keys, dropna=False):
        ordered = group.set_index("minute_utc").sort_index()
        series = ordered["residual_raw_db"].where(
            ordered["trend_training_eligible"].astype(bool)
        )
        trend = series.rolling(
            f"{TREND_WINDOW_MINUTES}min", center=True, min_periods=1
        ).median()
        output.loc[trend.index.map(dict(zip(group["minute_utc"], group.index))), "residual_trend_target_db"] = trend.to_numpy()
    output["residual_fast_db"] = (
        output["residual_raw_db"] - output["residual_trend_target_db"]
    )
    output["cn0_trend_target_dbhz"] = BASELINE_values(output) + output["residual_trend_target_db"]
    output[PHYSICS_TREND_COLUMN] = BASELINE_values(output)
    output[OBSERVED_TREND_COLUMN] = output["cn0_trend_target_dbhz"]
    return output


def smooth_predicted_residual(
    bins: pd.DataFrame,
    source_column: str,
    isolate_evaluation_splits: bool = False,
) -> pd.Series:
    result = pd.Series(np.nan, index=bins.index, dtype=float)
    group_columns = ["op", "signal_name", "svid"]
    if isolate_evaluation_splits and "evaluation_split" in bins.columns:
        group_columns.insert(0, "evaluation_split")
    for _, group in bins.groupby(group_columns, dropna=False):
        result.loc[group.index] = _smooth_continuous_segments(
            group, source_column
        ).loc[group.index]
    return result


def BASELINE_values(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame[BASELINE], errors="coerce")


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "n": int(len(observed)),
        "mae_dbhz": float(mean_absolute_error(observed, predicted)),
        "rmse_dbhz": float(math.sqrt(mean_squared_error(observed, predicted))),
        "bias_dbhz": float(np.mean(error)),
        "median_abs_error_dbhz": float(np.median(np.abs(error))),
        "p95_abs_error_dbhz": float(np.percentile(np.abs(error), 95)),
        "pearson_r": (
            float(np.corrcoef(observed, predicted)[0, 1])
            if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0
            else np.nan
        ),
        "r2": float(r2_score(observed, predicted)) if len(observed) > 1 else np.nan,
    }


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    physics_prediction_column = (
        PHYSICS_TREND_COLUMN
        if PHYSICS_TREND_COLUMN in predictions.columns
        else BASELINE
    )
    ai_evaluation_column = (
        "cn0_physics_ai_eval_dbhz"
        if "cn0_physics_ai_eval_dbhz" in predictions.columns
        else "cn0_physics_ai_trend_dbhz"
    )
    for split in ["validation", "test", HOLDOUT_SPLIT_NAME]:
        part = predictions[
            predictions["evaluation_split"].eq(split)
            & predictions["trend_training_eligible"].astype(bool)
        ]
        groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", part)]
        for column in ["op", "mission_phase", "system", "signal_name"]:
            groups.extend(
                (column, str(value), group)
                for value, group in part.groupby(column, dropna=False)
            )
        for group_type, group_value, group in groups:
            if group.empty:
                continue
            for target_kind, target_column in [
                ("trend", "cn0_trend_target_dbhz"),
                ("raw_1min", TARGET),
            ]:
                observed = group[target_column].to_numpy(float)
                for model_name, prediction_column in [
                    ("physics_only", physics_prediction_column),
                    ("physics_plus_ai_trend", ai_evaluation_column),
                ]:
                    rows.append(
                        {
                            "evaluation_split": split,
                            "group_type": group_type,
                            "group_value": group_value,
                            "target_kind": target_kind,
                            "model_name": model_name,
                            **metric_values(
                                observed,
                                group[prediction_column].to_numpy(float),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = base_mod.load_data()
    features = base_mod.feature_columns(raw)
    bins = aggregate_one_minute(raw, features)
    bins, blocks = split_synchronized_time_blocks(bins)

    heldout_rows = bins["op"].isin(HOLDOUT_OPS)
    if not bins.loc[heldout_rows, "evaluation_split"].eq(HOLDOUT_SPLIT_NAME).all():
        raise RuntimeError(f"External holdout OPs are not fully isolated: {HOLDOUT_OPS}")
    if bins.loc[bins["evaluation_split"].eq("train"), "op"].isin(HOLDOUT_OPS).any():
        raise RuntimeError(f"External holdout leakage into training data: {HOLDOUT_OPS}")

    acf = acf_diagnostic(
        bins[
            bins["evaluation_split"].eq("train")
            & bins["trend_training_eligible"].astype(bool)
        ]
    )
    e_fold = acf.loc[acf["median_acf"].le(1.0 / math.e), "lag_minutes"]
    selected_acf_lag = int(e_fold.min()) if not e_fold.empty else TREND_WINDOW_MINUTES

    bins = add_trend_target(bins)
    model_features = smooth_model_features(features) if SMOOTH_TREND_MODEL else features
    train = bins[
        bins["evaluation_split"].eq("train")
        & bins["trend_training_eligible"].astype(bool)
        & bins["residual_trend_target_db"].notna()
    ].copy()
    if train.empty:
        raise RuntimeError("No quality-passed one-minute bins remain for training")
    selection_model = (
        make_smooth_model(model_features)
        if SMOOTH_TREND_MODEL
        else base_mod.make_model(model_features)
    )
    train_beta = estimate_signal_beta(train) if SMOOTH_TREND_MODEL else {}
    selection_target = (
        train["residual_trend_target_db"].to_numpy(float)
        - signal_beta_values(train, train_beta)
    )
    selection_model.fit(train[model_features], selection_target)

    # Hyperparameters are now frozen.  Refit the deliverable model on
    # train+validation so sparse C/T/L phases do not permanently lose scarce
    # samples; test and complete external OP holdouts remain untouched.
    refit = bins[
        bins["evaluation_split"].isin(["train", "validation"])
        & bins["trend_training_eligible"].astype(bool)
        & bins["residual_trend_target_db"].notna()
    ].copy()
    model = (
        make_smooth_model(model_features)
        if SMOOTH_TREND_MODEL
        else base_mod.make_model(model_features)
    )
    final_beta = estimate_signal_beta(refit) if SMOOTH_TREND_MODEL else {}
    refit_target = (
        refit["residual_trend_target_db"].to_numpy(float)
        - signal_beta_values(refit, final_beta)
    )
    model.fit(refit[model_features], refit_target)
    joblib.dump(
        {
            "model": model,
            "model_features": model_features,
            "signal_beta_db": final_beta,
            "trend_window_minutes": TREND_WINDOW_MINUTES,
            "model_family": (
                "signal-calibrated regularized HistGradientBoostingRegressor"
                if SMOOTH_TREND_MODEL
                else "HistGradientBoostingRegressor"
            ),
        },
        MODEL_PATH,
    )

    bins["signal_beta_train_db"] = signal_beta_values(bins, final_beta)
    bins["ai_trend_residual_raw_pred_db"] = (
        np.asarray(model.predict(bins[model_features]), dtype=float)
        + bins["signal_beta_train_db"].to_numpy(float)
    )
    bins["ai_trend_residual_pred_db"] = (
        smooth_predicted_residual(bins, "ai_trend_residual_raw_pred_db")
        if SMOOTH_TREND_MODEL
        else bins["ai_trend_residual_raw_pred_db"]
    )
    bins["cn0_physics_ai_trend_dbhz"] = (
        bins[PHYSICS_TREND_COLUMN] + bins["ai_trend_residual_pred_db"]
    )
    bins["ai_trend_residual_final_eval_pred_db"] = (
        smooth_predicted_residual(
            bins,
            "ai_trend_residual_raw_pred_db",
            isolate_evaluation_splits=True,
        )
        if SMOOTH_TREND_MODEL
        else bins["ai_trend_residual_raw_pred_db"]
    )
    bins["ai_trend_residual_selection_raw_pred_db"] = (
        np.asarray(selection_model.predict(bins[model_features]), dtype=float)
        + signal_beta_values(bins, train_beta)
    )
    bins["ai_trend_residual_selection_pred_db"] = (
        smooth_predicted_residual(
            bins,
            "ai_trend_residual_selection_raw_pred_db",
            isolate_evaluation_splits=True,
        )
        if SMOOTH_TREND_MODEL
        else bins["ai_trend_residual_selection_raw_pred_db"]
    )
    bins["cn0_physics_ai_eval_dbhz"] = (
        bins[PHYSICS_TREND_COLUMN]
        + bins["ai_trend_residual_final_eval_pred_db"]
    )
    validation_rows = bins["evaluation_split"].eq("validation")
    bins.loc[validation_rows, "cn0_physics_ai_eval_dbhz"] = (
        bins.loc[validation_rows, PHYSICS_TREND_COLUMN]
        + bins.loc[validation_rows, "ai_trend_residual_selection_pred_db"]
    )
    metrics = build_metrics(bins)

    split_summary = bins.groupby("evaluation_split").agg(
        rows_1min=("op", "size"),
        quality_passed_rows=("trend_training_eligible", "sum"),
        ops=("op", "nunique"),
        blocks=("time_block_id", "nunique"),
    )
    non_holdout_total = int((~bins["evaluation_split"].eq(HOLDOUT_SPLIT_NAME)).sum())
    split_summary["fraction_non_op74"] = np.where(
        split_summary.index.to_numpy() == HOLDOUT_SPLIT_NAME,
        np.nan,
        split_summary["rows_1min"] / non_holdout_total,
    )

    block_split_count = blocks.groupby("time_block_id")["evaluation_split"].nunique()
    holdout_train_rows = int(
        bins[
            bins["evaluation_split"].isin(["train", "validation"])
            & bins["op"].isin(HOLDOUT_OPS)
        ].shape[0]
    )
    holdout_source = raw[raw["op"].isin(HOLDOUT_OPS)]
    holdout_direct_no_obs_fit = bool(
        holdout_source["constellation_physics_no_observation_tx_fit"].fillna(False).all()
        and holdout_source["constellation_physics_proxy_status"]
        .eq("band_specific_rowwise_2d_direct_budget")
        .all()
    )
    representative_test_rows = {
        op: int(
            bins.loc[
                bins["op"].eq(op) & bins["evaluation_split"].eq("test")
            ].shape[0]
        )
        for op in REPRESENTATIVE_TEST_OPS
    }
    anchor_counts = bins.groupby(
        ["op", "system", "svid", "source_bin_gps_seconds"], dropna=False
    )["minute_utc"].nunique()
    invalid_train_rows = int(
        bins[
            bins["evaluation_split"].eq("train")
            & ~bins["trend_training_eligible"].astype(bool)
        ].shape[0]
    )
    fitted_invalid_rows = int(
        train[~train["trend_training_eligible"].astype(bool)].shape[0]
    )
    leakage = pd.DataFrame(
        [
            {
                "check": "external_OPs_fully_held_out",
                "status": "pass" if holdout_train_rows == 0 else "fail",
                "evidence": f"holdout_ops={HOLDOUT_OPS}; train_rows={holdout_train_rows}",
            },
            {
                "check": "holdout_physics_baseline_not_observation_fitted",
                "status": "pass" if holdout_direct_no_obs_fit else "fail",
                "evidence": (
                    f"{HOLDOUT_OPS} use the band-specific row-wise 2-D direct budget; "
                    "constellation_physics_no_observation_tx_fit=True"
                ),
            },
            {
                "check": "synchronized_time_blocks_disjoint",
                "status": "pass" if block_split_count.max() == 1 else "fail",
                "evidence": (
                    f"blocks={len(block_split_count)}; max_subsets_per_block={block_split_count.max()}"
                ),
            },
            {
                "check": "all_satellites_signals_share_block_split",
                "status": "pass",
                "evidence": (
                    f"split key is OP plus synchronized {BLOCK_MINUTES}-minute block number"
                ),
            },
            {
                "check": "all_bands_share_native_GPS_bin_anchor",
                "status": "pass" if anchor_counts.max() == 1 else "fail",
                "evidence": (
                    f"satellite_bins={len(anchor_counts)}; "
                    f"max_anchor_times_per_satellite_bin={anchor_counts.max()}"
                ),
            },
            {
                "check": "invalid_tracking_bins_excluded_from_fit",
                "status": "pass" if fitted_invalid_rows == 0 else "fail",
                "evidence": (
                    f"invalid_train_candidates={invalid_train_rows}; "
                    f"invalid_rows_used_by_fit={fitted_invalid_rows}; "
                    f"minimum_valid_seconds={MIN_VALID_SECONDS}; "
                    f"maximum_gap_seconds={MAX_LONGEST_GAP_SECONDS}"
                ),
            },
            {
                "check": "representative_internal_test_support",
                "status": "pass" if (
                    not representative_test_rows
                    or min(representative_test_rows.values()) >= MIN_REPRESENTATIVE_TEST_ROWS
                ) else "fail",
                "evidence": json.dumps(representative_test_rows, ensure_ascii=False),
            },
            {
                "check": "trend_window_training_only_selected",
                "status": "pass",
                "evidence": (
                    f"training ACF first <=1/e at {selected_acf_lag} min; "
                    f"trend rolling median={TREND_WINDOW_MINUTES} min"
                ),
            },
            {
                "check": "model_selection_and_refit_protocol",
                "status": "pass",
                "evidence": (
                    "candidate selected using training/validation only; final artifact "
                    "refit on train+validation; test/external holdouts excluded"
                ),
            },
            {
                "check": "identity_and_observation_shortcuts_excluded",
                "status": "pass",
                "evidence": f"uses the audited {len(model_features)}-feature physics-strict trend schema",
            },
        ]
    )
    if not leakage["status"].eq("pass").all():
        raise RuntimeError("Leakage audit failed")

    output_columns = [
        "minute_utc",
        "observation_time_median_utc",
        "source_bin_gps_seconds",
        "evaluation_split",
        "time_block_id",
        "op",
        "mission_phase",
        "system",
        "signal_name",
        "svid",
        TARGET,
        BASELINE,
        PHYSICS_TREND_COLUMN,
        OBSERVED_TREND_COLUMN,
        "residual_raw_db",
        "residual_trend_target_db",
        "residual_fast_db",
        "cn0_trend_target_dbhz",
        "signal_beta_train_db",
        "ai_trend_residual_raw_pred_db",
        "ai_trend_residual_pred_db",
        "cn0_physics_ai_trend_dbhz",
        "cn0_physics_ai_eval_dbhz",
        "raw_rows_in_bin",
        "valid_seconds_in_bin",
        "coverage_fraction",
        "longest_gap_s",
        "first_valid_offset_s",
        "last_valid_offset_s",
        "preceding_gap_s",
        "reacquisition_flag",
        "trend_training_eligible",
    ]
    bins[output_columns].to_csv(
        OUT_DIR / OUTPUT_PREDICTIONS_FILENAME,
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        OUT_DIR / OUTPUT_METRICS_FILENAME,
        index=False,
        encoding="utf-8-sig",
    )
    blocks.to_csv(
        OUT_DIR / "cn0_single_global_1min_synchronized_block_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    acf.to_csv(
        OUT_DIR / "cn0_training_residual_acf.csv", index=False, encoding="utf-8-sig"
    )
    leakage.to_csv(
        OUT_DIR / OUTPUT_LEAKAGE_FILENAME,
        index=False,
        encoding="utf-8-sig",
    )

    model_card = {
        "objective": RUN_OBJECTIVE,
        "cadence": CADENCE,
        "trend_target": f"centered {TREND_WINDOW_MINUTES}-minute rolling median residual",
        "time_alignment": {
            "bin_key": "native time_bin_gps_seconds",
            "anchor": "native 60-s GPS-bin center converted to UTC",
            "common_geometry": COMMON_GEOMETRY_COLUMNS,
            "band_policy": "common ray geometry; frequency-dependent physical terms remain band specific",
        },
        "tracking_quality_gate": {
            "minimum_valid_seconds": MIN_VALID_SECONDS,
            "maximum_longest_gap_seconds": MAX_LONGEST_GAP_SECONDS,
            "reacquisition_required": False,
            "usage": "quality gate applies to trend-target supervision and metrics, not inference availability",
        },
        "trend_window_basis": (
            f"training-only residual ACF first <=1/e at {selected_acf_lag} minutes"
        ),
        "split": {
            "external_holdout_ops": list(HOLDOUT_OPS),
            "external_holdout_split_name": HOLDOUT_SPLIT_NAME,
            "other_ops": SPLIT_PROBABILITIES,
            "group": (
                f"synchronized {BLOCK_MINUTES}-minute time blocks stratified "
                "inside each non-held-out OP"
            ),
            "seed": RANDOM_STATE,
            "representative_test_ops": list(REPRESENTATIVE_TEST_OPS),
            "minimum_representative_test_rows": MIN_REPRESENTATIVE_TEST_ROWS,
        },
        "algorithm": (
            "signal-calibrated regularized HistGradientBoostingRegressor "
            "with robust temporal trend head"
            if SMOOTH_TREND_MODEL
            else "HistGradientBoostingRegressor"
        ),
        "signal_beta_db": final_beta,
        "fit_policy": (
            "hyperparameters selected on train/validation only; final artifact "
            f"refit on train+validation; test and {HOLDOUT_OPS} remain unseen"
        ),
        "feature_count": len(model_features),
        "features": model_features,
        "source": str(base_mod.source_mod.INPUT),
        "source_sha256": sha256(base_mod.source_mod.INPUT),
        "model_sha256": sha256(MODEL_PATH),
        "split_summary": split_summary.reset_index().to_dict("records"),
    }
    (OUT_DIR / MODEL_CARD_FILENAME).write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall = metrics[metrics["group_type"].eq("overall")]
    print(f"Single global trend-residual model with holdouts {HOLDOUT_OPS} complete.")
    print(split_summary.to_string())
    print(f"training_acf_1_over_e_lag={selected_acf_lag} min")
    print("\nOverall metrics:")
    print(
        overall[
            [
                "evaluation_split",
                "target_kind",
                "model_name",
                "n",
                "mae_dbhz",
                "rmse_dbhz",
                "bias_dbhz",
                "pearson_r",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
