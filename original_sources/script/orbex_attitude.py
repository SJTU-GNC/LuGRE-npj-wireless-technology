#!/usr/bin/env python3
"""Read CODE MGEX ORBEX attitudes and evaluate transmitter body angles.

ORBEX ATT quaternions are scalar-first and rotate terrestrial-frame vectors
into the satellite body frame. The CODE files use GPS time and IGS20 ECEF.
"""

from __future__ import annotations

import gzip
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


GPS_UTC_OFFSET_S = 18.0
DEFAULT_ORBEX_DIR = Path(
    "data/external_reference/gnss_attitude/code_mgex_fin_20250115_20250316"
)
DEFAULT_CACHE_DIR = Path("table/cache/orbex_attitude")


def _request_hash(utc: pd.Series, sat_ids: pd.Series) -> str:
    frame = pd.DataFrame(
        {
            "utc_ns": pd.to_datetime(utc, utc=True).map(lambda x: int(x.value)),
            "sat_id": sat_ids.astype(str).to_numpy(),
        }
    )
    return hashlib.sha256(frame.to_csv(index=False).encode("ascii")).hexdigest()[:20]


def _read_requested_file(path: Path, requested: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    times: dict[str, list[float]] = {sat_id: [] for sat_id in requested}
    quaternions: dict[str, list[list[float]]] = {sat_id: [] for sat_id in requested}
    epoch_utc_s: float | None = None
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as stream:
        for line in stream:
            if line.startswith("##"):
                fields = line.split()
                gps_dt = datetime(
                    int(fields[1]), int(fields[2]), int(fields[3]),
                    int(fields[4]), int(fields[5]), tzinfo=timezone.utc,
                ) + timedelta(seconds=float(fields[6]))
                epoch_utc_s = (gps_dt - timedelta(seconds=GPS_UTC_OFFSET_S)).timestamp()
            elif epoch_utc_s is not None and line.startswith(" ATT "):
                fields = line.split()
                sat_id = fields[1]
                if sat_id in requested:
                    times[sat_id].append(epoch_utc_s)
                    quaternions[sat_id].append([float(value) for value in fields[-4:]])
    return {
        sat_id: (np.asarray(times[sat_id], dtype=float), np.asarray(quaternions[sat_id], dtype=float))
        for sat_id in requested
        if times[sat_id]
    }


def _interpolate_quaternion(
    query_s: np.ndarray,
    source_s: np.ndarray,
    source_q: np.ndarray,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.full((len(query_s), 4), np.nan, dtype=float)
    gap = np.full(len(query_s), np.nan, dtype=float)
    if len(source_s) == 0:
        return result, gap
    right = np.searchsorted(source_s, query_s, side="left")
    lo = np.clip(right - 1, 0, len(source_s) - 1)
    hi = np.clip(right, 0, len(source_s) - 1)
    nearest_gap = np.minimum(np.abs(query_s - source_s[lo]), np.abs(source_s[hi] - query_s))
    valid = nearest_gap <= max_gap_s
    denom = source_s[hi] - source_s[lo]
    weight = np.divide(
        query_s - source_s[lo], denom,
        out=np.zeros_like(query_s, dtype=float), where=denom > 0,
    )
    q0 = source_q[lo].copy()
    q1 = source_q[hi].copy()
    q1[np.sum(q0 * q1, axis=1) < 0.0] *= -1.0
    q = (1.0 - weight[:, None]) * q0 + weight[:, None] * q1
    norm = np.linalg.norm(q, axis=1)
    valid &= np.isfinite(norm) & (norm > 0.0)
    result[valid] = q[valid] / norm[valid, None]
    gap[valid] = nearest_gap[valid]
    return result, gap


def lookup_orbex_quaternions(
    utc: pd.Series,
    sat_ids: pd.Series,
    orbex_dir: Path = DEFAULT_ORBEX_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_gap_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scalar-first ECEF-to-body quaternions at requested UTC epochs."""
    utc = pd.to_datetime(utc, utc=True).reset_index(drop=True)
    sat_ids = sat_ids.astype(str).reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"orbex_lookup_{_request_hash(utc, sat_ids)}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["q"], cached["gap_s"], cached["status"].astype(str)

    q_out = np.full((len(utc), 4), np.nan, dtype=float)
    gap_out = np.full(len(utc), np.nan, dtype=float)
    status = np.full(len(utc), "orbex_file_unavailable", dtype="U48")
    gps_time = utc + pd.to_timedelta(GPS_UTC_OFFSET_S, unit="s")
    day_key = gps_time.dt.strftime("%Y%j")
    query_s = utc.map(lambda x: x.timestamp()).to_numpy(float)

    for key in sorted(day_key.unique()):
        day_mask = day_key.eq(key).to_numpy()
        indices = np.flatnonzero(day_mask)
        requested = set(sat_ids.iloc[indices])
        matches = sorted(orbex_dir.glob(f"COD0MGXFIN_{key}0000_01D_30S_ATT.OBX.gz"))
        if not matches:
            continue
        source = _read_requested_file(matches[0], requested)
        for sat_id in requested:
            sat_indices = indices[sat_ids.iloc[indices].eq(sat_id).to_numpy()]
            if sat_id not in source:
                status[sat_indices] = "orbex_satellite_unavailable"
                continue
            q, gap = _interpolate_quaternion(
                query_s[sat_indices], source[sat_id][0], source[sat_id][1], max_gap_s
            )
            q_out[sat_indices] = q
            gap_out[sat_indices] = gap
            valid = np.isfinite(q).all(axis=1)
            status[sat_indices[valid]] = "CODE_MGEX_final_ORBEX_30s"
            status[sat_indices[~valid]] = "orbex_epoch_outside_tolerance"

    np.savez_compressed(cache_path, q=q_out, gap_s=gap_out, status=status)
    return q_out, gap_out, status


def eci_to_ecef_vectors(vectors: np.ndarray, utc: pd.Series) -> np.ndarray:
    unix = pd.to_datetime(utc, utc=True).map(lambda t: t.timestamp()).to_numpy(dtype=float)
    jd = unix / 86400.0 + 2440587.5
    centuries = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837 + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * centuries**2 - centuries**3 / 38710000.0
    )
    theta = np.radians(np.mod(gmst_deg, 360.0))
    c, s = np.cos(theta), np.sin(theta)
    return np.column_stack(
        [c * vectors[:, 0] + s * vectors[:, 1], -s * vectors[:, 0] + c * vectors[:, 1], vectors[:, 2]]
    )


def quaternion_rotate(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    scalar = q[:, :1]
    vector = q[:, 1:]
    return vectors + 2.0 * scalar * np.cross(vector, vectors) + 2.0 * np.cross(
        vector, np.cross(vector, vectors)
    )


def orbex_body_angles(
    utc: pd.Series,
    sat_ids: pd.Series,
    los_tx_eci: np.ndarray,
    orbex_dir: Path = DEFAULT_ORBEX_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q, gap_s, status = lookup_orbex_quaternions(utc, sat_ids, orbex_dir=orbex_dir)
    valid = np.isfinite(q).all(axis=1)
    theta = np.full(len(q), np.nan, dtype=float)
    phi = np.full(len(q), np.nan, dtype=float)
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        los_ecef = eci_to_ecef_vectors(los_tx_eci[valid], utc.iloc[valid_idx])
        local = quaternion_rotate(q[valid], los_ecef)
        local /= np.linalg.norm(local, axis=1)[:, None]
        theta[valid] = np.degrees(np.arccos(np.clip(local[:, 2], -1.0, 1.0)))
        phi[valid] = np.mod(np.degrees(np.arctan2(local[:, 1], local[:, 0])), 360.0)
    return theta, phi, gap_s, status
