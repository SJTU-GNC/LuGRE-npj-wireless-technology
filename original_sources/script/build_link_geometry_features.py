from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from inspect_statevectors import parse_workbook


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAJECTORY_DIR = DATA_DIR / "trajectory"
TLM_DIR = DATA_DIR / "receiver_observation" / "TLM"
OUT_DIR = PROJECT_ROOT / "table"
NAV_DIR = DATA_DIR / "external_reference" / "nav"

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
UNIX_EPOCH_JD = 2440587.5
GPS_UTC_LEAP_SECONDS = 18.0

MU_GPS = 3.986005e14
MU_GAL = 3.986004418e14
OMEGA_E_DOT = 7.2921151467e-5
EARTH_RADIUS_KM = 6378.137
MOON_RADIUS_KM = 1737.4
C_MPS = 299792458.0

SIGNAL_META = {
    0: ("G", "GPS_L1", 1575.42),
    1: ("G", "GPS_L5", 1176.45),
    2: ("E", "GAL_E1", 1575.42),
    3: ("E", "GAL_E5a", 1176.45),
}

WGC_START = {
    "WGC_StateVector_20260519234114.xls": datetime(2025, 1, 1, tzinfo=timezone.utc),
    "WGC_StateVector_20260519230403.xls": datetime(2025, 1, 16, tzinfo=timezone.utc),
    "WGC_StateVector_20260519231120.xls": datetime(2025, 2, 1, tzinfo=timezone.utc),
    "WGC_StateVector_20260519231421.xls": datetime(2025, 2, 15, tzinfo=timezone.utc),
    "WGC_StateVector_20260519232935.xls": datetime(2025, 3, 1, tzinfo=timezone.utc),
    "WGC_StateVector_20260519232744.xls": datetime(2025, 3, 16, tzinfo=timezone.utc),
}

RAW_FILE_RE = re.compile(
    r"^TLM_RAW_(?P<date>\d{8})_(?P<time>\d{6})_(?P<dur>\d+)H_(?P<phase>[A-Z])_(?P<op>OP\d+)_(?P<rep>\d+)\.txt$"
)
RX_TIME_RE = re.compile(r"rxTime:\s*([0-9.]+)")
MEASURE_RE = re.compile(
    r"svid:\s*(?P<svid>\d+)\s+"
    r"prRaw:\s*(?P<prRaw>[-+0-9.Ee]+)\s+"
    r"cn0:\s*(?P<cn0>[-+0-9.Ee]+)\s+"
    r"signalId:\s*(?P<signalId>\d+)\s+"
    r"fdRaw:\s*(?P<fdRaw>[-+0-9.Ee]+)"
)


@dataclass
class NavRecord:
    system: str
    prn: int
    epoch_gps: float
    af0: float
    af1: float
    af2: float
    iode: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    ecc: float
    cus: float
    sqrt_a: float
    toe: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    omega: float
    omega_dot: float
    idot: float
    week: float

    @property
    def toe_abs_gps(self) -> float:
        return self.week * 604800.0 + self.toe


def utc_to_gps_seconds(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - GPS_EPOCH).total_seconds() + GPS_UTC_LEAP_SECONDS


def gps_seconds_to_utc(gps_seconds: float) -> datetime:
    return GPS_EPOCH + timedelta(seconds=gps_seconds - GPS_UTC_LEAP_SECONDS)


def parse_wgc_time(value: str) -> datetime:
    text = value.replace(" UTC", "")
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


def load_receiver_state_vectors() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(TRAJECTORY_DIR.glob("WGC_StateVector_*.xls")):
        file_start = WGC_START.get(path.name)
        cells = parse_workbook(path).get("Results", {})
        for r in sorted({r for r, _ in cells}):
            row = [cells.get((r, c)) for c in range(10)]
            if r < 3 or not isinstance(row[0], str):
                continue
            if not all(isinstance(row[c], (int, float)) for c in (1, 2, 3, 4, 5, 6, 7, 8)):
                continue
            try:
                dt = parse_wgc_time(row[0])
            except ValueError:
                if file_start is None:
                    continue
                dt = file_start + timedelta(minutes=r - 3)
            rows.append(
                {
                    "utc": dt.isoformat().replace("+00:00", "Z"),
                    "gps_seconds": utc_to_gps_seconds(dt),
                    "distance_from_earth_center_km": float(row[1]),
                    "speed_km_s": float(row[2]),
                    "rx_x_eci_km": float(row[3]),
                    "rx_y_eci_km": float(row[4]),
                    "rx_z_eci_km": float(row[5]),
                    "rx_vx_eci_km_s": float(row[6]),
                    "rx_vy_eci_km_s": float(row[7]),
                    "rx_vz_eci_km_s": float(row[8]),
                    "source_wgc_file": path.name,
                }
            )
    rows.sort(key=lambda x: x["gps_seconds"])
    dedup: dict[float, dict] = {}
    for row in rows:
        dedup[row["gps_seconds"]] = row
    return [dedup[k] for k in sorted(dedup)]


def load_receiver_state_vectors_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "utc": row["utc"],
                        "gps_seconds": float(row["gps_seconds"]),
                        "distance_from_earth_center_km": float(row["distance_from_earth_center_km"]),
                        "speed_km_s": float(row["speed_km_s"]),
                        "rx_x_eci_km": float(row["rx_x_eci_km"]),
                        "rx_y_eci_km": float(row["rx_y_eci_km"]),
                        "rx_z_eci_km": float(row["rx_z_eci_km"]),
                        "rx_vx_eci_km_s": float(row["rx_vx_eci_km_s"]),
                        "rx_vy_eci_km_s": float(row["rx_vy_eci_km_s"]),
                        "rx_vz_eci_km_s": float(row["rx_vz_eci_km_s"]),
                        "source_wgc_file": row.get("source_wgc_file", path.name),
                    }
                )
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda x: x["gps_seconds"])
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_raw_observations(sample_step_seconds: int) -> list[dict]:
    buckets: dict[tuple[int, int, int, str, str], dict] = {}
    for path in sorted(TLM_DIR.glob("TLM_RAW_*.txt")):
        file_match = RAW_FILE_RE.match(path.name)
        phase = file_match.group("phase") if file_match else ""
        op = file_match.group("op") if file_match else ""
        with path.open("r", errors="ignore") as f:
            for line in f:
                rx_match = RX_TIME_RE.search(line)
                if not rx_match:
                    continue
                rx_gps = float(rx_match.group(1))
                bin_gps = int(math.floor(rx_gps / sample_step_seconds) * sample_step_seconds)
                for meas in MEASURE_RE.finditer(line):
                    signal_id = int(meas.group("signalId"))
                    svid = int(meas.group("svid"))
                    key = (bin_gps, signal_id, svid, phase, op)
                    bucket = buckets.setdefault(
                        key,
                        {
                            "rx_gps_sum": 0.0,
                            "cn0_sum": 0.0,
                            "pr_sum": 0.0,
                            "fd_sum": 0.0,
                            "samples": 0,
                        },
                    )
                    bucket["rx_gps_sum"] += rx_gps
                    bucket["cn0_sum"] += float(meas.group("cn0"))
                    bucket["pr_sum"] += float(meas.group("prRaw"))
                    bucket["fd_sum"] += float(meas.group("fdRaw"))
                    bucket["samples"] += 1

    rows: list[dict] = []
    for (bin_gps, signal_id, svid, phase, op), b in sorted(buckets.items()):
        n = b["samples"]
        system, signal_name, freq_mhz = SIGNAL_META.get(signal_id, ("", f"signal_{signal_id}", float("nan")))
        rx_gps = b["rx_gps_sum"] / n
        rows.append(
            {
                "rx_utc": gps_seconds_to_utc(rx_gps).isoformat().replace("+00:00", "Z"),
                "rx_gps_seconds": rx_gps,
                "time_bin_gps_seconds": bin_gps,
                "system": system,
                "svid": svid,
                "signal_id": signal_id,
                "signal_name": signal_name,
                "frequency_mhz": freq_mhz,
                "cn0_dbhz_mean": b["cn0_sum"] / n,
                "pseudorange_raw_m_mean": b["pr_sum"] / n,
                "doppler_raw_hz_mean": b["fd_sum"] / n,
                "samples": n,
                "mission_phase": phase,
                "op": op,
            }
        )
    return rows


def brdc_url(day: datetime) -> str:
    year = day.year
    doy = day.timetuple().tm_yday
    return (
        "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/"
        f"{year}/{doy:03d}/BRDC00WRD_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
    )


def ensure_nav_file(day: datetime) -> Path | None:
    NAV_DIR.mkdir(parents=True, exist_ok=True)
    year = day.year
    doy = day.timetuple().tm_yday
    path = NAV_DIR / f"BRDC00WRD_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = brdc_url(day)
    try:
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as exc:
        print(f"WARNING: failed to download {url}: {exc}")
        return None


def rinex_float(text: str) -> float:
    text = text.replace("D", "E").strip()
    return float(text) if text else float("nan")


def parse_nav_values(line: str) -> list[float]:
    return [rinex_float(line[i : i + 19]) for i in range(4, min(len(line), 80), 19)]


def parse_rinex_nav(path: Path) -> list[NavRecord]:
    records: list[NavRecord] = []
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if "END OF HEADER" in line:
                break
        while True:
            first = f.readline()
            if not first:
                break
            if len(first) < 23 or first[0] not in ("G", "E"):
                continue
            body = [f.readline() for _ in range(7)]
            try:
                system = first[0]
                prn = int(first[1:3])
                year = int(first[4:8])
                month = int(first[9:11])
                day = int(first[12:14])
                hour = int(first[15:17])
                minute = int(first[18:20])
                second = int(float(first[21:23]))
                epoch = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                clk = [rinex_float(first[i : i + 19]) for i in (23, 42, 61)]
                vals = []
                for nav_line in body:
                    vals.extend(parse_nav_values(nav_line))
                if len(vals) < 19:
                    continue
                records.append(
                    NavRecord(
                        system=system,
                        prn=prn,
                        epoch_gps=utc_to_gps_seconds(epoch),
                        af0=clk[0],
                        af1=clk[1],
                        af2=clk[2],
                        iode=vals[0],
                        crs=vals[1],
                        delta_n=vals[2],
                        m0=vals[3],
                        cuc=vals[4],
                        ecc=vals[5],
                        cus=vals[6],
                        sqrt_a=vals[7],
                        toe=vals[8],
                        cic=vals[9],
                        omega0=vals[10],
                        cis=vals[11],
                        i0=vals[12],
                        crc=vals[13],
                        omega=vals[14],
                        omega_dot=vals[15],
                        idot=vals[16],
                        week=vals[18],
                    )
                )
            except Exception:
                continue
    return records


def load_nav_records(obs_rows: list[dict]) -> dict[tuple[str, int], list[NavRecord]]:
    if not obs_rows:
        return {}
    utc_days = {
        gps_seconds_to_utc(float(r["rx_gps_seconds"])).replace(hour=0, minute=0, second=0, microsecond=0)
        for r in obs_rows
        if r["system"] in ("G", "E")
    }
    days = set()
    for day in utc_days:
        days.add(day - timedelta(days=1))
        days.add(day)
        days.add(day + timedelta(days=1))

    by_sat: dict[tuple[str, int], list[NavRecord]] = defaultdict(list)
    for day in sorted(days):
        nav_file = ensure_nav_file(day)
        if not nav_file:
            continue
        for rec in parse_rinex_nav(nav_file):
            by_sat[(rec.system, rec.prn)].append(rec)
    for records in by_sat.values():
        records.sort(key=lambda r: r.toe_abs_gps)
    return dict(by_sat)


def solve_kepler(mk: float, ecc: float) -> float:
    ek = mk
    for _ in range(12):
        ek -= (ek - ecc * math.sin(ek) - mk) / max(1e-12, 1.0 - ecc * math.cos(ek))
    return ek


def wrap_gnss_time(dt: float) -> float:
    if dt > 302400.0:
        return dt - 604800.0
    if dt < -302400.0:
        return dt + 604800.0
    return dt


def select_nav_record(records: list[NavRecord], gps_seconds: float, max_age_seconds: float = 21600.0) -> NavRecord | None:
    if not records:
        return None
    best = min(records, key=lambda rec: abs(gps_seconds - rec.toe_abs_gps))
    if abs(gps_seconds - best.toe_abs_gps) > max_age_seconds:
        return None
    return best


def broadcast_ecef_km(rec: NavRecord, gps_seconds: float) -> np.ndarray:
    # First-order clock/light-time correction is enough for feature generation.
    tk0 = wrap_gnss_time(gps_seconds - rec.toe_abs_gps)
    a = rec.sqrt_a * rec.sqrt_a
    mu = MU_GAL if rec.system == "E" else MU_GPS
    n0 = math.sqrt(mu / (a**3))
    n = n0 + rec.delta_n
    mk = rec.m0 + n * tk0
    ek = solve_kepler(mk, rec.ecc)
    vk = math.atan2(math.sqrt(1 - rec.ecc * rec.ecc) * math.sin(ek), math.cos(ek) - rec.ecc)
    phik = vk + rec.omega
    duk = rec.cus * math.sin(2 * phik) + rec.cuc * math.cos(2 * phik)
    drk = rec.crs * math.sin(2 * phik) + rec.crc * math.cos(2 * phik)
    dik = rec.cis * math.sin(2 * phik) + rec.cic * math.cos(2 * phik)
    uk = phik + duk
    rk = a * (1 - rec.ecc * math.cos(ek)) + drk
    ik = rec.i0 + dik + rec.idot * tk0
    x_orb = rk * math.cos(uk)
    y_orb = rk * math.sin(uk)
    omega_k = rec.omega0 + (rec.omega_dot - OMEGA_E_DOT) * tk0 - OMEGA_E_DOT * rec.toe
    cos_o = math.cos(omega_k)
    sin_o = math.sin(omega_k)
    cos_i = math.cos(ik)
    sin_i = math.sin(ik)
    x = x_orb * cos_o - y_orb * cos_i * sin_o
    y = x_orb * sin_o + y_orb * cos_i * cos_o
    z = y_orb * sin_i
    return np.array([x, y, z], dtype=float) / 1000.0


def gmst_rad(dt_utc: datetime) -> float:
    unix_seconds = (dt_utc - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
    jd = unix_seconds / 86400.0 + UNIX_EPOCH_JD
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t**3) / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def ecef_to_eci_km(ecef: np.ndarray, gps_seconds: float) -> np.ndarray:
    theta = gmst_rad(gps_seconds_to_utc(gps_seconds))
    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = ecef
    return np.array([c * x - s * y, s * x + c * y, z], dtype=float)


def interp_vector(times: np.ndarray, values: np.ndarray, t: float, max_gap_seconds: float) -> tuple[np.ndarray | None, bool]:
    idx = int(np.searchsorted(times, t))
    if idx == 0 or idx >= len(times):
        return None, False
    if times[idx] - times[idx - 1] > max_gap_seconds:
        return None, False
    w = (t - times[idx - 1]) / (times[idx] - times[idx - 1])
    return values[idx - 1] * (1 - w) + values[idx] * w, True


def jd_to_gps_seconds(jd: float) -> float:
    unix_seconds = (jd - UNIX_EPOCH_JD) * 86400.0
    gps_epoch_unix = (GPS_EPOCH - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
    return unix_seconds - gps_epoch_unix + GPS_UTC_LEAP_SECONDS


def load_moon_ephemeris() -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    path = TRAJECTORY_DIR / "moon_j2000_geocentric_20250101_20250401_10min.csv"
    if not path.exists():
        return None, None
    times = []
    pos = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(jd_to_gps_seconds(float(row["jd"])))
            pos.append([float(row["x_km"]), float(row["y_km"]), float(row["z_km"])])
    return np.array(times, dtype=float), np.array(pos, dtype=float)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def limb_margin_deg(observer: np.ndarray, target: np.ndarray, body_center: np.ndarray, radius_km: float) -> float:
    to_target = target - observer
    to_body = body_center - observer
    d_body = float(np.linalg.norm(to_body))
    sep = angle_deg(to_target, to_body)
    if d_body <= radius_km:
        # Surface or slightly-inside observers are common when WGC uses a
        # different lunar radius/reference point than the simple sphere here.
        # In that case the useful quantity is local horizon margin: positive
        # means the target is above the local tangent plane, negative below.
        return sep - 90.0
    angular_radius = math.degrees(math.asin(radius_km / d_body))
    return sep - angular_radius


def line_intersects_sphere(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius_km: float) -> bool:
    d = p1 - p0
    f = p0 - center
    p0_radius = float(np.linalg.norm(f))
    if p0_radius <= radius_km:
        # If the observer is on/inside the reference sphere, a ray to any
        # visible satellite exits the sphere. That exit is not an occultation.
        # Treat the body as blocking only when the line of sight points below
        # the local tangent plane.
        return float(np.dot(d, f)) < 0.0
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(f, d))
    c = float(np.dot(f, f)) - radius_km * radius_km
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0.0 < t1 < 1.0) or (0.0 < t2 < 1.0)


def closest_approach_altitude_km(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius_km: float) -> float:
    d = p1 - p0
    denom = float(np.dot(d, d))
    if denom <= 0.0:
        return float("nan")
    t = -float(np.dot(p0 - center, d)) / denom
    t = max(0.0, min(1.0, t))
    closest = p0 + t * d
    return float(np.linalg.norm(closest - center) - radius_km)


def make_geometry_features(obs_rows: list[dict], receiver_rows: list[dict], nav_by_sat: dict[tuple[str, int], list[NavRecord]]) -> list[dict]:
    rx_times = np.array([float(r["gps_seconds"]) for r in receiver_rows], dtype=float)
    rx_pos = np.array([[r["rx_x_eci_km"], r["rx_y_eci_km"], r["rx_z_eci_km"]] for r in receiver_rows], dtype=float)
    rx_vel = np.array([[r["rx_vx_eci_km_s"], r["rx_vy_eci_km_s"], r["rx_vz_eci_km_s"]] for r in receiver_rows], dtype=float)
    moon_times, moon_pos = load_moon_ephemeris()

    rows: list[dict] = []
    for obs in obs_rows:
        system = obs["system"]
        svid = int(obs["svid"])
        t = float(obs["rx_gps_seconds"])
        rx_vec, ok_pos = interp_vector(rx_times, rx_pos, t, 180.0)
        rx_v, ok_vel = interp_vector(rx_times, rx_vel, t, 180.0)
        nav = select_nav_record(nav_by_sat.get((system, svid), []), t)
        if not ok_pos or not ok_vel or nav is None:
            continue

        sat_ecef = broadcast_ecef_km(nav, t)
        sat_eci = ecef_to_eci_km(sat_ecef, t)
        los = sat_eci - rx_vec
        range_km = float(np.linalg.norm(los))
        freq_mhz = float(obs["frequency_mhz"])
        fspl_db = 32.44 + 20 * math.log10(range_km) + 20 * math.log10(freq_mhz)
        range_rate_km_s = float(np.dot(los / range_km, -rx_v))

        earth_margin = limb_margin_deg(rx_vec, sat_eci, np.zeros(3), EARTH_RADIUS_KM)
        earth_blocked = line_intersects_sphere(rx_vec, sat_eci, np.zeros(3), EARTH_RADIUS_KM)
        earth_grazing_altitude = closest_approach_altitude_km(rx_vec, sat_eci, np.zeros(3), EARTH_RADIUS_KM)
        earth_observer_altitude = float(np.linalg.norm(rx_vec) - EARTH_RADIUS_KM)
        moon_margin = float("nan")
        moon_grazing_altitude = float("nan")
        moon_observer_altitude = float("nan")
        moon_blocked = False
        moon_vec = None
        if moon_times is not None and moon_pos is not None:
            moon_vec, ok_moon = interp_vector(moon_times, moon_pos, t, 900.0)
            if ok_moon and moon_vec is not None:
                moon_observer_altitude = float(np.linalg.norm(rx_vec - moon_vec) - MOON_RADIUS_KM)
                moon_margin = limb_margin_deg(rx_vec, sat_eci, moon_vec, MOON_RADIUS_KM)
                moon_blocked = line_intersects_sphere(rx_vec, sat_eci, moon_vec, MOON_RADIUS_KM)
                moon_grazing_altitude = closest_approach_altitude_km(rx_vec, sat_eci, moon_vec, MOON_RADIUS_KM)

        tx_off = angle_deg(-sat_eci, rx_vec - sat_eci)
        rx_earth_sep = angle_deg(los, -rx_vec)
        rx_moon_sep = angle_deg(los, moon_vec - rx_vec) if moon_vec is not None else float("nan")

        row = dict(obs)
        row.update(
            {
                "feature_set": "cn0_physics_available_wgc_new",
                "cn0_status": "observed_receiver_cn0_raw_aggregated",
                "rx_x_eci_km": rx_vec[0],
                "rx_y_eci_km": rx_vec[1],
                "rx_z_eci_km": rx_vec[2],
                "rx_vx_eci_km_s": rx_v[0],
                "rx_vy_eci_km_s": rx_v[1],
                "rx_vz_eci_km_s": rx_v[2],
                "sat_x_eci_km": sat_eci[0],
                "sat_y_eci_km": sat_eci[1],
                "sat_z_eci_km": sat_eci[2],
                "sat_x_ecef_km": sat_ecef[0],
                "sat_y_ecef_km": sat_ecef[1],
                "sat_z_ecef_km": sat_ecef[2],
                "geometric_range_km": range_km,
                "range_rate_rx_only_km_s": range_rate_km_s,
                "fspl_db": fspl_db,
                "tx_offboresight_deg": tx_off,
                "rx_to_sat_earth_center_sep_deg": rx_earth_sep,
                "rx_to_sat_moon_center_sep_deg": rx_moon_sep,
                "earth_observer_altitude_km": earth_observer_altitude,
                "earth_limb_margin_deg": earth_margin,
                "earth_grazing_altitude_km": earth_grazing_altitude,
                "earth_blocked": int(earth_blocked),
                "moon_observer_altitude_km": moon_observer_altitude,
                "moon_limb_margin_deg": moon_margin,
                "moon_grazing_altitude_km": moon_grazing_altitude,
                "moon_blocked": int(moon_blocked),
                "nav_toe_gps_seconds": nav.toe_abs_gps,
                "nav_age_seconds": t - nav.toe_abs_gps,
                "orbit_geometry_status": "physics_available_wgc_new" if "20260622" in str(receiver_rows[0].get("source_wgc_file", "")) else "physics_available",
                "fspl_status": "physics_available",
                "occultation_status": "physics_available_spherical_body_with_surface_horizon_mode" if moon_observer_altitude <= 0.0 else "physics_available_spherical_body",
                "limb_margin_status": "physics_available_limb_margin_or_surface_horizon_margin",
                "tx_gain_status": "proxy_tx_offboresight_until_gain_map_ingested",
                "rx_gain_status": "ai_residual_until_lugre_pattern_attitude_mount_available",
                "atmos_iono_status": "proxy_grazing_altitude_limb_margin_until_tec_model_ingested",
                "geometry_frame_note": "rx=WGC geocentric J2000-like ECI; sat=broadcast ECEF rotated to ECI by GMST approximation",
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LuGRE link geometry features from WGC state vectors and TLM RAW observations.")
    parser.add_argument("--sample-step-seconds", type=int, default=60, help="RAW aggregation step; 60 seconds matches WGC cadence.")
    parser.add_argument("--receiver-state-csv", type=Path, default=None, help="Optional receiver state CSV to use instead of parsing WGC_StateVector_*.xls.")
    parser.add_argument("--receiver-output", type=Path, default=OUT_DIR / "receiver_state_vectors.csv", help="Receiver state-vector CSV output path.")
    parser.add_argument("--obs-output", type=Path, default=OUT_DIR / "raw_observations_aggregated.csv", help="Aggregated RAW observation CSV output path.")
    parser.add_argument("--geometry-output", type=Path, default=OUT_DIR / "link_geometry_features.csv", help="Link geometry feature CSV output path.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.receiver_state_csv is not None:
        receiver_path = args.receiver_state_csv
        if not receiver_path.is_absolute():
            receiver_path = PROJECT_ROOT / receiver_path
        receiver_rows = load_receiver_state_vectors_csv(receiver_path)
    else:
        receiver_rows = load_receiver_state_vectors()
    receiver_fields = [
        "utc",
        "gps_seconds",
        "distance_from_earth_center_km",
        "speed_km_s",
        "rx_x_eci_km",
        "rx_y_eci_km",
        "rx_z_eci_km",
        "rx_vx_eci_km_s",
        "rx_vy_eci_km_s",
        "rx_vz_eci_km_s",
        "source_wgc_file",
    ]
    write_csv(args.receiver_output if args.receiver_output.is_absolute() else PROJECT_ROOT / args.receiver_output, receiver_rows, receiver_fields)
    print(f"receiver_state_vectors: {len(receiver_rows)} rows")

    obs_rows = aggregate_raw_observations(args.sample_step_seconds)
    obs_fields = [
        "rx_utc",
        "rx_gps_seconds",
        "time_bin_gps_seconds",
        "system",
        "svid",
        "signal_id",
        "signal_name",
        "frequency_mhz",
        "cn0_dbhz_mean",
        "pseudorange_raw_m_mean",
        "doppler_raw_hz_mean",
        "samples",
        "mission_phase",
        "op",
    ]
    write_csv(args.obs_output if args.obs_output.is_absolute() else PROJECT_ROOT / args.obs_output, obs_rows, obs_fields)
    print(f"raw_observations_aggregated: {len(obs_rows)} rows")

    nav_by_sat = load_nav_records(obs_rows)
    print(f"nav satellites loaded: {len(nav_by_sat)}")

    geom_rows = make_geometry_features(obs_rows, receiver_rows, nav_by_sat)
    geom_fields = obs_fields + [
        "feature_set",
        "cn0_status",
        "rx_x_eci_km",
        "rx_y_eci_km",
        "rx_z_eci_km",
        "rx_vx_eci_km_s",
        "rx_vy_eci_km_s",
        "rx_vz_eci_km_s",
        "sat_x_eci_km",
        "sat_y_eci_km",
        "sat_z_eci_km",
        "sat_x_ecef_km",
        "sat_y_ecef_km",
        "sat_z_ecef_km",
        "geometric_range_km",
        "range_rate_rx_only_km_s",
        "fspl_db",
        "tx_offboresight_deg",
        "rx_to_sat_earth_center_sep_deg",
        "rx_to_sat_moon_center_sep_deg",
        "earth_observer_altitude_km",
        "earth_limb_margin_deg",
        "earth_grazing_altitude_km",
        "earth_blocked",
        "moon_observer_altitude_km",
        "moon_limb_margin_deg",
        "moon_grazing_altitude_km",
        "moon_blocked",
        "nav_toe_gps_seconds",
        "nav_age_seconds",
        "orbit_geometry_status",
        "fspl_status",
        "occultation_status",
        "limb_margin_status",
        "tx_gain_status",
        "rx_gain_status",
        "atmos_iono_status",
        "geometry_frame_note",
    ]
    write_csv(args.geometry_output if args.geometry_output.is_absolute() else PROJECT_ROOT / args.geometry_output, geom_rows, geom_fields)
    print(f"link_geometry_features: {len(geom_rows)} rows")


if __name__ == "__main__":
    main()
