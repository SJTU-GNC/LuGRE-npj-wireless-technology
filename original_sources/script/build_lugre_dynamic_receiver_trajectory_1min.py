#!/usr/bin/env python3
"""Build a one-minute LuGRE receiver trajectory from dynamics reconstruction.

The stitch never uses the WGC truth error to choose an arc. At duplicate epochs,
it selects the best dynamics-status class and takes the median reconstructed
state across arcs in that class. Interpolation is restricted to continuous
segments whose source-point gaps do not exceed 15 minutes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "runtime_cache" / "python_deps"
ORBIT_MODULES = ROOT / "script" / "algorithm" / "orbit_dynamics"
for path in (DEPS, ORBIT_MODULES):
    if path.exists():
        sys.path.insert(0, str(path))

from orbit_models import (  # noqa: E402
    ForceModelConfig,
    StateVector,
    propagate_arc,
    propagate_to_epoch,
    select_force_model,
)
from run_lugre_lunar_orbit_inversion import (  # noqa: E402
    DEFAULT_CR_AREA_OVER_MASS_M2_KG,
    DEFAULT_GRAVITY_DEGREE,
    GRAIL_FILE,
    GrailGravityModel,
    LunarDynamics,
    SpiceLunarEnvironment,
)
from run_lugre_surface_fixed_site_geometry import (  # noqa: E402
    SpiceMoonFixedTransform,
    site_vector_moon_fixed,
    spice as surface_spice,
)
INPUT = (
    ROOT
    / "table"
    / "algorithm"
    / "orbit_dynamics"
    / "all_stage_orbit_geometry_trajectory_points.csv"
)
OUTPUT = (
    ROOT
    / "table"
    / "algorithm"
    / "orbit_dynamics"
    / "lugre_dynamic_receiver_trajectory_1min.csv"
)
AUDIT = OUTPUT.with_name("lugre_dynamic_receiver_trajectory_1min_audit.csv")
GAP_AUDIT = OUTPUT.with_name("lugre_dynamic_receiver_trajectory_1min_gap_fill_audit.csv")
EARTH_TRANSFER_STATES = OUTPUT.with_name("lugre_dynamic_reconstruction_points.csv")
LUNAR_STATES = OUTPUT.with_name("lugre_lunar_orbit_dynamic_reconstruction_points.csv")

MISSION_START = pd.Timestamp("2025-01-15T07:32:00Z")
MISSION_STOP = pd.Timestamp("2025-03-16T23:15:00Z")

MAX_SOURCE_GAP = pd.Timedelta("15min")
POSITION_COLUMNS = [
    "reconstructed_x_eci_km",
    "reconstructed_y_eci_km",
    "reconstructed_z_eci_km",
]


def gps_seconds(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    values = pd.DatetimeIndex(times)
    return (
        (values - pd.Timestamp("1980-01-06T00:00:00Z")).total_seconds().to_numpy()
        + 18.0
    )


def smoothstep5(value: np.ndarray) -> np.ndarray:
    return 6.0 * value**5 - 15.0 * value**4 + 10.0 * value**3


def propagated_rows(states: list[StateVector]) -> np.ndarray:
    return np.asarray([state.as_array() for state in states], dtype=float)


def bidirectional_earth_bridge(
    target_times: pd.DatetimeIndex,
    left: pd.Series | None,
    right: pd.Series,
    stage: str,
) -> tuple[np.ndarray, float]:
    target_gps = gps_seconds(target_times)

    def state(row: pd.Series) -> StateVector:
        return StateVector(
            epoch_gps_seconds=float(row["gps_seconds"]),
            r_km=tuple(float(row[f"reconstructed_{axis}_eci_km"]) for axis in "xyz"),
            v_km_s=tuple(float(row[f"reconstructed_v{axis}_eci_km_s"]) for axis in "xyz"),
            frame="J2000",
            metadata={"mission_phase": stage},
        )

    right_state = state(right)
    phase = "EARTH_ORBIT" if stage == "earth_orbit" else "CISLUNAR_TRANSFER"
    config = select_force_model(right_state, phase)
    config = ForceModelConfig(**{**config.__dict__, "max_step_s": min(config.max_step_s, 60.0)})
    if left is None:
        return np.vstack(
            [propagate_to_epoch(right_state, epoch, config).as_array() for epoch in target_gps]
        ), float("nan")

    left_state = state(left)
    forward = propagated_rows(propagate_arc(left_state, target_gps, config))
    current = right_state
    backward_reversed: list[StateVector] = []
    for epoch in target_gps[::-1]:
        current = propagate_to_epoch(current, float(epoch), config)
        backward_reversed.append(current)
    backward = propagated_rows(list(reversed(backward_reversed)))
    span = right_state.epoch_gps_seconds - left_state.epoch_gps_seconds
    alpha = np.clip((target_gps - left_state.epoch_gps_seconds) / span, 0.0, 1.0)
    weight = smoothstep5(alpha)[:, None]
    bridged = (1.0 - weight) * forward + weight * backward
    closure = propagate_to_epoch(left_state, right_state.epoch_gps_seconds, config)
    closure_error = float(
        np.linalg.norm(np.asarray(closure.r_km) - np.asarray(right_state.r_km))
    )
    return bridged, closure_error


def estimate_lunar_velocity(
    source: pd.DataFrame, boundary_index: int, side: str
) -> np.ndarray:
    row = source.loc[boundary_index]
    same_arc = source[source["arc_id"].eq(row["arc_id"])].sort_values("gps_seconds")
    if side == "left":
        local = same_arc[same_arc["gps_seconds"].le(row["gps_seconds"])].tail(5)
    else:
        local = same_arc[same_arc["gps_seconds"].ge(row["gps_seconds"])].head(5)
    if len(local) < 2:
        local = same_arc.iloc[
            max(0, same_arc.index.get_loc(boundary_index) - 2) :
            same_arc.index.get_loc(boundary_index) + 3
        ]
    t = local["gps_seconds"].to_numpy(float) - float(row["gps_seconds"])
    position = local[
        [
            "reconstructed_x_moon_j2000_km",
            "reconstructed_y_moon_j2000_km",
            "reconstructed_z_moon_j2000_km",
        ]
    ].to_numpy(float)
    return np.asarray([np.polyfit(t, position[:, axis], 1)[0] for axis in range(3)])


def bidirectional_lunar_bridge(
    target_times: pd.DatetimeIndex,
    left: pd.Series,
    right: pd.Series,
    left_velocity: np.ndarray,
    right_velocity: np.ndarray,
    dynamics: LunarDynamics,
) -> tuple[np.ndarray, float]:
    target_gps = gps_seconds(target_times)
    left_state = np.hstack(
        ([left[f"reconstructed_{axis}_moon_j2000_km"] for axis in "xyz"], left_velocity)
    ).astype(float)
    right_state = np.hstack(
        ([right[f"reconstructed_{axis}_moon_j2000_km"] for axis in "xyz"], right_velocity)
    ).astype(float)
    left_epoch = float(left["gps_seconds"])
    right_epoch = float(right["gps_seconds"])
    forward = dynamics.propagate(left_state, np.r_[left_epoch, target_gps])[1:]
    backward_desc = dynamics.propagate(right_state, np.r_[right_epoch, target_gps[::-1]])[1:]
    backward = backward_desc[::-1]
    alpha = np.clip((target_gps - left_epoch) / (right_epoch - left_epoch), 0.0, 1.0)
    weight = smoothstep5(alpha)[:, None]
    moon_relative = (1.0 - weight) * forward + weight * backward
    moon_geocentric = np.vstack(
        [dynamics.env.moon_geocentric_state(float(epoch)) for epoch in target_gps]
    )
    eci = moon_relative + moon_geocentric
    closure = dynamics.propagate(left_state, [left_epoch, right_epoch])[-1]
    closure_error = float(np.linalg.norm(closure[:3] - right_state[:3]))
    return eci, closure_error


def complete_with_stage_dynamics(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.DataFrame({"utc": pd.date_range(MISSION_START, MISSION_STOP, freq="1min")})
    working = full.merge(result, on="utc", how="left", validate="one_to_one")
    missing = working["rx_x_eci_km"].isna()
    if not missing.any():
        return result, pd.DataFrame()

    earth_transfer = pd.read_csv(EARTH_TRANSFER_STATES, low_memory=False)
    earth_transfer["utc"] = pd.to_datetime(earth_transfer["utc"], utc=True, errors="coerce")
    earth_transfer = earth_transfer.dropna(
        subset=[
            "utc",
            "gps_seconds",
            "reconstructed_x_eci_km",
            "reconstructed_y_eci_km",
            "reconstructed_z_eci_km",
            "reconstructed_vx_eci_km_s",
            "reconstructed_vy_eci_km_s",
            "reconstructed_vz_eci_km_s",
        ]
    ).sort_values("utc")
    lunar = pd.read_csv(LUNAR_STATES, low_memory=False)
    lunar["utc"] = pd.to_datetime(lunar["utc"], utc=True, errors="coerce")
    lunar = lunar.dropna(
        subset=[
            "utc",
            "gps_seconds",
            "reconstructed_x_moon_j2000_km",
            "reconstructed_y_moon_j2000_km",
            "reconstructed_z_moon_j2000_km",
        ]
    ).sort_values("utc").reset_index(drop=True)

    lunar_dynamics: LunarDynamics | None = None
    surface_transform: SpiceMoonFixedTransform | None = None
    site_fixed: np.ndarray | None = None
    surface_sign: float | None = None
    gap_rows: list[dict[str, object]] = []
    groups = missing.ne(missing.shift()).cumsum()
    for gap_id, gap in working[missing].groupby(groups[missing], sort=True):
        target_times = pd.DatetimeIndex(gap["utc"])
        first_index, last_index = int(gap.index.min()), int(gap.index.max())
        left_row = working.iloc[first_index - 1] if first_index > 0 else None
        right_row = working.iloc[last_index + 1] if last_index + 1 < len(working) else None
        if right_row is None:
            raise RuntimeError("Mission-ending dynamics gap cannot be right-anchored")
        stage = str(
            left_row["dynamic_stage"]
            if left_row is not None and pd.notna(left_row["dynamic_stage"])
            else right_row["dynamic_stage"]
        )
        closure_error = float("nan")
        if stage in {"earth_orbit", "cislunar_transfer"}:
            left_source = None
            if left_row is not None:
                candidates = earth_transfer[
                    earth_transfer["phase"].eq(stage)
                    & earth_transfer["utc"].le(target_times.min())
                ]
                if not candidates.empty:
                    left_source = candidates.iloc[-1]
            candidates = earth_transfer[
                earth_transfer["phase"].eq(stage)
                & earth_transfer["utc"].ge(target_times.max())
            ]
            if candidates.empty:
                raise RuntimeError(f"No right dynamics state for {stage} gap {gap_id}")
            right_source = candidates.iloc[0]
            states, closure_error = bidirectional_earth_bridge(
                target_times, left_source, right_source, stage
            )
            positions = states[:, :3]
            method = f"{stage}_bidirectional_force_model_endpoint_constrained"
            source_arcs = "|".join(
                str(value)
                for value in [
                    None if left_source is None else left_source.get("arc_id"),
                    right_source.get("arc_id"),
                ]
                if value is not None
            )
        elif stage == "lunar_orbit":
            if lunar_dynamics is None:
                environment = SpiceLunarEnvironment()
                gravity = GrailGravityModel.load(GRAIL_FILE, DEFAULT_GRAVITY_DEGREE)
                lunar_dynamics = LunarDynamics(
                    environment, gravity, DEFAULT_CR_AREA_OVER_MASS_M2_KG
                )
            left_candidates = lunar[lunar["utc"].le(target_times.min())]
            right_candidates = lunar[lunar["utc"].ge(target_times.max())]
            if left_candidates.empty or right_candidates.empty:
                raise RuntimeError(f"Missing lunar boundary state for gap {gap_id}")
            left_index = int(left_candidates.index[-1])
            right_index = int(right_candidates.index[0])
            left_source, right_source = lunar.loc[left_index], lunar.loc[right_index]
            positions, closure_error = bidirectional_lunar_bridge(
                target_times,
                left_source,
                right_source,
                estimate_lunar_velocity(lunar, left_index, "left"),
                estimate_lunar_velocity(lunar, right_index, "right"),
                lunar_dynamics,
            )
            positions = positions[:, :3]
            method = "lunar_DE440_GRGM8_SRP_bidirectional_endpoint_constrained"
            source_arcs = f"{left_source['arc_id']}|{right_source['arc_id']}"
        elif stage == "surface_fixed_site":
            if surface_transform is None:
                # Lunar-orbit propagation and surface geometry use overlapping
                # SPICE kernels. Reset the process-wide pool so the fixed-site
                # transform is reproducible and not affected by prior load order.
                if surface_spice is not None:
                    surface_spice.kclear()
                surface_transform = SpiceMoonFixedTransform()
                site_fixed = site_vector_moon_fixed()
                # Keep the same Earth-centred vector convention as the already
                # reconstructed surface arc.  Determine this once from a valid
                # model epoch; never use reference samples inside a gap.
                surface_rows = working.loc[
                    (working["dynamic_stage"] == "surface_fixed_site")
                    & working["rx_x_eci_km"].notna()
                ]
                if surface_rows.empty:
                    raise RuntimeError("No valid surface-model epoch is available to set the SPICE sign")
                sign_row = surface_rows.iloc[0]
                sign_utc = pd.Timestamp(sign_row["utc"]).strftime("%Y-%m-%dT%H:%M:%SZ")
                raw_sign_position, sign_frame_source = surface_transform.inertial_site(
                    sign_utc, site_fixed
                )
                if not sign_frame_source.startswith("SPICE_DE440_MOON_ME_to_J2000"):
                    raise RuntimeError(
                        "Surface gap fill requires the controlled SPICE transform; "
                        f"received {sign_frame_source}"
                    )
                existing_sign_position = sign_row[
                    ["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]
                ].to_numpy(dtype=float)
                surface_sign = (
                    1.0
                    if np.linalg.norm(raw_sign_position - existing_sign_position)
                    <= np.linalg.norm(-raw_sign_position - existing_sign_position)
                    else -1.0
                )
            surface_positions = []
            for timestamp in target_times:
                spice_utc = pd.Timestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
                position, frame_source = surface_transform.inertial_site(
                    spice_utc, site_fixed
                )
                if not frame_source.startswith("SPICE_DE440_MOON_ME_to_J2000"):
                    raise RuntimeError(
                        "Surface gap fill requires the controlled SPICE transform; "
                        f"received {frame_source} at {timestamp.isoformat()}"
                    )
                surface_positions.append(surface_sign * position)
            positions = np.vstack(surface_positions)
            method = "surface_fixed_site_SPICE_DE440_MOON_ME_existing_arc_sign"
            source_arcs = "fixed_BGM1_TOPO_site"
        else:
            raise RuntimeError(f"Unsupported dynamics gap stage: {stage}")

        indices = gap.index
        working.loc[indices, ["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"]] = positions
        working.loc[indices, "dynamic_stage"] = stage
        working.loc[indices, "dynamic_quality_status"] = (
            "dynamics_propagated_gap_fill_no_interior_truth"
        )
        working.loc[indices, "dynamic_status_priority"] = 3
        working.loc[indices, "contributing_arcs"] = source_arcs
        working.loc[indices, "contributing_arc_count"] = len(source_arcs.split("|"))
        working.loc[indices, "reference_position_error_km"] = np.nan
        working.loc[indices, "dynamic_segment_id"] = -int(gap_id)
        left_offset = (
            (target_times - pd.Timestamp(left_row["utc"])).total_seconds()
            if left_row is not None
            else np.full(len(target_times), np.inf)
        )
        right_offset = (pd.Timestamp(right_row["utc"]) - target_times).total_seconds()
        working.loc[indices, "nearest_source_epoch_offset_s"] = np.minimum(
            left_offset, right_offset
        )
        working.loc[indices, "receiver_trajectory_source"] = method
        working.loc[indices, "interpolation_policy"] = (
            "stage_dynamics_propagation_with_endpoint_constraint_no_truth_inside_gap"
        )
        gap_rows.append(
            {
                "gap_id": int(gap_id),
                "stage": stage,
                "start_utc": target_times.min(),
                "end_utc": target_times.max(),
                "minutes_filled": len(target_times),
                "method": method,
                "source_arcs": source_arcs,
                "forward_closure_error_at_right_boundary_km": closure_error,
                "interior_truth_used": False,
            }
        )
    return working.sort_values("utc").reset_index(drop=True), pd.DataFrame(gap_rows)


def status_priority(value: object) -> int:
    text = str(value).lower()
    if "valid_dynamic" in text or "surface_fixed" in text:
        return 0
    if "insufficient" in text:
        return 1
    return 2


def build() -> pd.DataFrame:
    raw = pd.read_csv(INPUT, low_memory=False)
    raw["utc"] = pd.to_datetime(raw["utc"], utc=True, errors="coerce")
    for column in POSITION_COLUMNS + ["position_error_km"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["utc", *POSITION_COLUMNS]).copy()
    raw["status_priority"] = raw["validation_status"].map(status_priority)

    best_priority = raw.groupby("utc")["status_priority"].transform("min")
    chosen = raw[raw["status_priority"].eq(best_priority)].copy()
    stitched = (
        chosen.groupby("utc", as_index=False)
        .agg(
            reconstructed_x_eci_km=("reconstructed_x_eci_km", "median"),
            reconstructed_y_eci_km=("reconstructed_y_eci_km", "median"),
            reconstructed_z_eci_km=("reconstructed_z_eci_km", "median"),
            dynamic_stage=("stage", lambda x: "|".join(sorted(set(map(str, x))))),
            dynamic_quality_status=(
                "validation_status",
                lambda x: "|".join(sorted(set(map(str, x)))),
            ),
            dynamic_status_priority=("status_priority", "min"),
            contributing_arcs=("arc_id", lambda x: "|".join(sorted(set(map(str, x))))),
            contributing_arc_count=("arc_id", "nunique"),
            reference_position_error_km=("position_error_km", "median"),
        )
        .sort_values("utc")
        .reset_index(drop=True)
    )
    stitched["segment_id"] = stitched["utc"].diff().gt(MAX_SOURCE_GAP).cumsum()

    outputs: list[pd.DataFrame] = []
    for segment_id, segment in stitched.groupby("segment_id", sort=True):
        segment = segment.sort_values("utc").drop_duplicates("utc")
        if len(segment) == 1:
            minute_index = pd.DatetimeIndex([segment["utc"].iloc[0].floor("min")])
        else:
            minute_index = pd.date_range(
                segment["utc"].iloc[0].ceil("min"),
                segment["utc"].iloc[-1].floor("min"),
                freq="1min",
                tz="UTC",
            )
        if minute_index.empty:
            continue

        source_seconds = segment["utc"].astype("int64").to_numpy(float) / 1.0e9
        target_seconds = minute_index.astype("int64").to_numpy(float) / 1.0e9
        out = pd.DataFrame({"utc": minute_index})
        for source, target in zip(
            POSITION_COLUMNS,
            ["rx_x_eci_km", "rx_y_eci_km", "rx_z_eci_km"],
        ):
            out[target] = np.interp(
                target_seconds,
                source_seconds,
                segment[source].to_numpy(float),
            )

        nearest_index = np.abs(
            source_seconds[:, None] - target_seconds[None, :]
        ).argmin(axis=0)
        nearest = segment.iloc[nearest_index].reset_index(drop=True)
        out["dynamic_stage"] = nearest["dynamic_stage"]
        out["dynamic_quality_status"] = nearest["dynamic_quality_status"]
        out["dynamic_status_priority"] = nearest["dynamic_status_priority"].to_numpy()
        out["contributing_arcs"] = nearest["contributing_arcs"]
        out["contributing_arc_count"] = nearest["contributing_arc_count"].to_numpy()
        out["reference_position_error_km"] = nearest[
            "reference_position_error_km"
        ].to_numpy()
        out["dynamic_segment_id"] = int(segment_id)
        out["nearest_source_epoch_offset_s"] = np.min(
            np.abs(source_seconds[:, None] - target_seconds[None, :]), axis=0
        )
        outputs.append(out)

    result = pd.concat(outputs, ignore_index=True).sort_values("utc")
    result["receiver_trajectory_source"] = (
        "stitched_orbit_dynamics_reconstruction_no_truth_based_arc_selection"
    )
    result["interpolation_policy"] = (
        "linear_to_natural_1min_within_source_segments_gap_le_15min"
    )
    result, gap_audit = complete_with_stage_dynamics(result)
    result["gps_seconds"] = (
        result["utc"] - pd.Timestamp("1980-01-06T00:00:00Z")
    ).dt.total_seconds() + 18.0
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    gap_audit.to_csv(GAP_AUDIT, index=False, encoding="utf-8-sig")

    audit = (
        result.groupby(
            ["dynamic_stage", "dynamic_quality_status", "dynamic_status_priority"],
            dropna=False,
            as_index=False,
        )
        .agg(
            rows=("utc", "size"),
            start_utc=("utc", "min"),
            end_utc=("utc", "max"),
            median_reference_error_km=("reference_position_error_km", "median"),
            p95_reference_error_km=(
                "reference_position_error_km",
                lambda x: float(np.nanpercentile(x, 95)),
            ),
        )
    )
    audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    result = build()
    print(f"rows={len(result):,}")
    print(f"segments={result['dynamic_segment_id'].nunique():,}")
    print(f"span={result['utc'].min()} to {result['utc'].max()}")
    print(result.groupby("dynamic_stage").size().to_string())
    print(OUTPUT)


if __name__ == "__main__":
    main()
