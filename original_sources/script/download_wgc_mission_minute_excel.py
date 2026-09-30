from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from openpyxl import Workbook


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "trajectory" / "wgc_mission_minute_lugre_earth_omit_errors"
API_BASE = "https://wgc2.jpl.nasa.gov:8443/webgeocalc/api"

COLUMNS = [
    "UTC calendar date",
    "Distance (km)",
    "Speed (km/s)",
    "X (km)",
    "Y (km)",
    "Z (km)",
    "dX/dt (km/s)",
    "dY/dt (km/s)",
    "dZ/dt (km/s)",
    "Time at Target",
    "Light Time (s)",
]


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def to_wgc_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def to_wgc_calendar(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.000000 UTC")


def parse_wgc_calendar(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.startswith("2025-"):
        return None
    try:
        return datetime.strptime(value.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def api_request(path: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    url = f"{API_BASE}/{path.lstrip('/')}"
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"WGC API HTTP {exc.code} for {url}: {detail}") from exc


def submit_state_vector(
    start: datetime,
    stop: datetime,
    *,
    poll_interval_seconds: float,
    max_poll_seconds: int,
) -> dict[str, Any]:
    payload = {
        "kernels": [{"type": "KERNEL_SET", "id": 8}],
        "timeSystem": "UTC",
        "timeFormat": "CALENDAR",
        "intervals": [{"startTime": to_wgc_time(start), "endTime": to_wgc_time(stop)}],
        "timeStep": 1,
        "timeStepUnits": "MINUTES",
        "calculationType": "STATE_VECTOR",
        "targetType": "OBJECT",
        "target": "BGM1_LUGRE",
        "observerType": "OBJECT",
        "observer": "EARTH",
        "referenceFrame": "J2000",
        "aberrationCorrection": "NONE",
        "stateRepresentation": "RECTANGULAR",
        "errorHandling": "OMIT_ERRORS",
    }
    created = api_request("calculation/new", payload)
    if created.get("status") != "OK":
        raise RuntimeError(f"WGC calculation/new failed: {created}")
    calculation_id = created["calculationId"]
    phase = created.get("result", {}).get("phase")
    deadline = time.monotonic() + max_poll_seconds
    while phase != "COMPLETE":
        if phase in {"FAILED", "CANCELED", "EXPIRED"}:
            raise RuntimeError(f"WGC calculation {calculation_id} ended with phase {phase}: {created}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"WGC calculation {calculation_id} did not complete within {max_poll_seconds}s")
        time.sleep(poll_interval_seconds)
        created = api_request(f"calculation/{calculation_id}")
        phase = created.get("result", {}).get("phase")
    results = api_request(f"calculation/{calculation_id}/results")
    if results.get("status") != "OK":
        raise RuntimeError(f"WGC results failed for {calculation_id}: {results}")
    return results


def download_rows(
    start: datetime,
    stop: datetime,
    *,
    max_error_span_minutes: int,
    poll_interval_seconds: float,
    max_poll_seconds: int,
) -> list[list[Any]]:
    try:
        result = submit_state_vector(
            start,
            stop,
            poll_interval_seconds=poll_interval_seconds,
            max_poll_seconds=max_poll_seconds,
        )
        return result.get("rows", [])
    except Exception as exc:
        span_minutes = int((stop - start).total_seconds() // 60)
        if span_minutes <= max_error_span_minutes:
            return []
        mid = start + timedelta(minutes=span_minutes // 2)
        left = download_rows(
            start,
            mid,
            max_error_span_minutes=max_error_span_minutes,
            poll_interval_seconds=poll_interval_seconds,
            max_poll_seconds=max_poll_seconds,
        )
        right_start = mid + timedelta(minutes=1)
        right = (
            download_rows(
                right_start,
                stop,
                max_error_span_minutes=max_error_span_minutes,
                poll_interval_seconds=poll_interval_seconds,
                max_poll_seconds=max_poll_seconds,
            )
            if right_start <= stop
            else []
        )
        return left + right


def iter_chunks(start: datetime, stop: datetime, chunk_minutes: int) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    current = start
    while current <= stop:
        chunk_stop = min(current + timedelta(minutes=chunk_minutes - 1), stop)
        chunks.append((current, chunk_stop))
        current = chunk_stop + timedelta(minutes=1)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 1-minute WGC LuGRE/Earth mission state vectors to Excel.")
    parser.add_argument("--start", default="2025-01-15T00:00:00")
    parser.add_argument("--stop", default="2025-03-16T23:59:00")
    parser.add_argument("--chunk-minutes", type=int, default=1440)
    parser.add_argument("--max-error-span-minutes", type=int, default=30)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--max-poll-seconds", type=int, default=600)
    args = parser.parse_args()

    start = parse_utc(args.start)
    stop = parse_utc(args.stop)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_xlsx = OUT_DIR / "WGC_StateVector_LuGRE_target_Earth_observer_1min_no_aberration_omit_errors.xlsx"
    manifest_path = OUT_DIR / "download_manifest.json"

    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Results")
    ws.append(COLUMNS)

    total_rows = 0
    expected_rows = 0
    omitted_rows = 0
    chunks = iter_chunks(start, stop, args.chunk_minutes)
    manifest_chunks: list[dict[str, Any]] = []
    for idx, (chunk_start, chunk_stop) in enumerate(chunks, start=1):
        print(f"[{idx}/{len(chunks)}] {to_wgc_time(chunk_start)} .. {to_wgc_time(chunk_stop)}", flush=True)
        rows = download_rows(
            chunk_start,
            chunk_stop,
            max_error_span_minutes=args.max_error_span_minutes,
            poll_interval_seconds=args.poll_interval_seconds,
            max_poll_seconds=args.max_poll_seconds,
        )
        chunk_expected = int((chunk_stop - chunk_start).total_seconds() // 60) + 1
        valid_rows = 0
        for row in rows:
            padded = list(row[: len(COLUMNS)]) + [""] * (len(COLUMNS) - len(row))
            row_dt = parse_wgc_calendar(padded[0] if padded else None)
            if row_dt is None:
                continue
            padded[0] = to_wgc_calendar(row_dt)
            ws.append(padded)
            valid_rows += 1
        chunk_omitted = chunk_expected - valid_rows
        expected_rows += chunk_expected
        total_rows += valid_rows
        omitted_rows += chunk_omitted
        manifest_chunks.append(
            {
                "start_utc": to_wgc_time(chunk_start),
                "stop_utc": to_wgc_time(chunk_stop),
                "expected_rows": chunk_expected,
                "raw_rows": len(rows),
                "rows": valid_rows,
                "omitted_rows": chunk_omitted,
            }
        )

    wb.save(out_xlsx)
    manifest = {
        "wgc_api_base": API_BASE,
        "calculation_type": "STATE_VECTOR",
        "kernel_set": "CLPS",
        "kernel_set_id": 8,
        "target": "BGM1_LUGRE",
        "observer": "EARTH",
        "reference_frame": "J2000",
        "light_propagation": "None",
        "stellar_aberration": "None",
        "aberration_correction": "NONE",
        "state_representation": "RECTANGULAR",
        "time_system": "UTC",
        "time_format": "Calendar date and time",
        "time_step": 1,
        "time_step_units": "MINUTES",
        "error_handling": "OMIT_ERRORS",
        "error_handling_label": "Silently omit errors",
        "start_utc": to_wgc_time(start),
        "stop_utc": to_wgc_time(stop),
        "excel": str(out_xlsx.relative_to(PROJECT_ROOT)),
        "columns": COLUMNS,
        "expected_rows": expected_rows,
        "rows": total_rows,
        "omitted_rows": omitted_rows,
        "chunks": manifest_chunks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_xlsx}")
    print(f"Wrote {manifest_path}")
    print(f"Rows: {total_rows}, omitted_rows: {omitted_rows}, expected_rows: {expected_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
