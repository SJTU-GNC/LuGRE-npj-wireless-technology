from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TLM_DIR = PROJECT_ROOT / "data" / "receiver_observation" / "TLM"
OUT_DIR = PROJECT_ROOT / "data" / "trajectory" / "wgc_no_aberration_by_tlm"

API_BASE = "https://wgc2.jpl.nasa.gov:8443/webgeocalc/api"
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_UTC_LEAP_SECONDS = 18.0
RX_TIME_RE = re.compile(r"rxTime:\s*([0-9.]+)")


@dataclass(frozen=True)
class TlmWindow:
    path: Path
    start_gps: float
    stop_gps: float
    count: int


def gps_to_utc(gps_seconds: float) -> datetime:
    return GPS_EPOCH + timedelta(seconds=gps_seconds - GPS_UTC_LEAP_SECONDS)


def utc_to_wgc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def wgc_calendar_to_utc(text: str) -> datetime:
    clean = text.replace(" UTC", "")
    return datetime.strptime(clean, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


def utc_to_gps(dt: datetime) -> float:
    return (dt - GPS_EPOCH).total_seconds() + GPS_UTC_LEAP_SECONDS


def api_request(path: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    url = f"{API_BASE}/{path.lstrip('/')}"
    data = None
    headers = {"Accept": "application/json"}
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


def collect_tlm_windows(pattern: str) -> list[TlmWindow]:
    windows: list[TlmWindow] = []
    for path in sorted(TLM_DIR.glob(pattern)):
        start: float | None = None
        stop: float | None = None
        count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = RX_TIME_RE.search(line)
                if not match:
                    continue
                value = float(match.group(1))
                start = value if start is None else min(start, value)
                stop = value if stop is None else max(stop, value)
                count += 1
        if start is not None and stop is not None:
            windows.append(TlmWindow(path=path, start_gps=start, stop_gps=stop, count=count))
    return windows


def iter_chunks(start_gps: float, stop_gps: float, max_rows: int) -> list[tuple[float, float]]:
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2")
    chunks: list[tuple[float, float]] = []
    chunk_span = max_rows - 1
    current = start_gps
    while current <= stop_gps + 1e-6:
        stop = min(current + chunk_span, stop_gps)
        chunks.append((current, stop))
        current = stop + 1.0
    return chunks


def submit_state_vector(
    start_gps: float,
    stop_gps: float,
    *,
    kernel_set_id: int,
    max_poll_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    payload = {
        "kernels": [{"type": "KERNEL_SET", "id": kernel_set_id}],
        "timeSystem": "UTC",
        "timeFormat": "CALENDAR",
        "intervals": [
            {
                "startTime": utc_to_wgc(gps_to_utc(start_gps)),
                "endTime": utc_to_wgc(gps_to_utc(stop_gps)),
            }
        ],
        "timeStep": 1,
        "timeStepUnits": "SECONDS",
        "calculationType": "STATE_VECTOR",
        "targetType": "OBJECT",
        "target": "EARTH",
        "observerType": "OBJECT",
        "observer": "BGM1_LUGRE",
        "referenceFrame": "J2000",
        "aberrationCorrection": "NONE",
        "stateRepresentation": "RECTANGULAR",
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
    results = api_request(f"calculation/{calculation_id}/results", timeout=180)
    if results.get("status") != "OK":
        raise RuntimeError(f"WGC results failed for {calculation_id}: {results}")
    results["calculationId"] = calculation_id
    return results


def is_frame_coverage_error(message: str) -> bool:
    return "SPICE(NOFRAMECONNECT)" in message or "SPICE(SPKINSUFFDATA)" in message


def probe_single_state(
    source_file: str,
    gps_seconds: float,
    *,
    kernel_set_id: int,
    max_poll_seconds: int,
    poll_interval_seconds: float,
) -> tuple[bool, list[dict[str, Any]] | str]:
    try:
        result = submit_state_vector(
            gps_seconds,
            gps_seconds,
            kernel_set_id=kernel_set_id,
            max_poll_seconds=max_poll_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        return True, [row_to_record(source_file, row, result["calculationId"]) for row in result.get("rows", [])]
    except Exception as exc:
        short = str(exc)[:1000].replace("\r", " ").replace("\n", " ")
        return False, short


def error_record(source_file: str, gps_seconds: float, error_message: str) -> dict[str, Any]:
    utc_dt = gps_to_utc(gps_seconds)
    return {
        "source_tlm_file": source_file,
        "utc": utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
        "gps_seconds": f"{gps_seconds:.6f}".rstrip("0").rstrip("."),
        "distance_from_earth_center_km": "",
        "speed_km_s": "",
        "earth_rel_bgm1_lugre_x_j2000_km": "",
        "earth_rel_bgm1_lugre_y_j2000_km": "",
        "earth_rel_bgm1_lugre_z_j2000_km": "",
        "earth_rel_bgm1_lugre_vx_j2000_km_s": "",
        "earth_rel_bgm1_lugre_vy_j2000_km_s": "",
        "earth_rel_bgm1_lugre_vz_j2000_km_s": "",
        "time_at_target_utc": "",
        "light_time_s": "",
        "wgc_calculation_id": "",
        "wgc_error": error_message,
    }


def error_records_for_range(
    source_file: str, start_gps: float, stop_gps: float, error_message: str
) -> list[dict[str, Any]]:
    count = int(round(stop_gps - start_gps)) + 1
    return [error_record(source_file, start_gps + offset, error_message) for offset in range(max(0, count))]


def row_to_record(source_file: str, row: list[Any], calculation_id: str) -> dict[str, Any]:
    utc_text = row[0]
    try:
        utc_dt = wgc_calendar_to_utc(utc_text)
        gps_seconds: float | str = f"{utc_to_gps(utc_dt):.6f}".rstrip("0").rstrip(".")
    except Exception:
        gps_seconds = ""
    numeric = row[1:9] if len(row) >= 9 else []
    numeric += [""] * (8 - len(numeric))
    return {
        "source_tlm_file": source_file,
        "utc": utc_text,
        "gps_seconds": gps_seconds,
        "distance_from_earth_center_km": numeric[0],
        "speed_km_s": numeric[1],
        "earth_rel_bgm1_lugre_x_j2000_km": numeric[2],
        "earth_rel_bgm1_lugre_y_j2000_km": numeric[3],
        "earth_rel_bgm1_lugre_z_j2000_km": numeric[4],
        "earth_rel_bgm1_lugre_vx_j2000_km_s": numeric[5],
        "earth_rel_bgm1_lugre_vy_j2000_km_s": numeric[6],
        "earth_rel_bgm1_lugre_vz_j2000_km_s": numeric[7],
        "time_at_target_utc": row[9] if len(row) > 9 else "",
        "light_time_s": row[10] if len(row) > 10 else "",
        "wgc_calculation_id": calculation_id,
        "wgc_error": "",
    }


def download_chunk_records(
    source_file: str,
    start_gps: float,
    stop_gps: float,
    *,
    kernel_set_id: int,
    max_poll_seconds: int,
    poll_interval_seconds: float,
    max_error_span_seconds: int,
) -> list[dict[str, Any]]:
    try:
        result = submit_state_vector(
            start_gps,
            stop_gps,
            kernel_set_id=kernel_set_id,
            max_poll_seconds=max_poll_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        calculation_id = result["calculationId"]
        return [row_to_record(source_file, row, calculation_id) for row in result.get("rows", [])]
    except Exception as exc:
        message = str(exc)
        short = message[:1000].replace("\r", " ").replace("\n", " ")
        if stop_gps - start_gps < max_error_span_seconds:
            return error_records_for_range(source_file, start_gps, stop_gps, short)
        if is_frame_coverage_error(message):
            midpoint = start_gps + int((stop_gps - start_gps) // 2)
            midpoint_ok, midpoint_result = probe_single_state(
                source_file,
                midpoint,
                kernel_set_id=kernel_set_id,
                max_poll_seconds=max_poll_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            if midpoint_ok:
                left = download_chunk_records(
                    source_file,
                    start_gps,
                    midpoint,
                    kernel_set_id=kernel_set_id,
                    max_poll_seconds=max_poll_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    max_error_span_seconds=max_error_span_seconds,
                )
                right_start = midpoint + 1.0
                right = download_chunk_records(
                    source_file,
                    right_start,
                    stop_gps,
                    kernel_set_id=kernel_set_id,
                    max_poll_seconds=max_poll_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    max_error_span_seconds=max_error_span_seconds,
                )
                return left + right
            start_ok, start_result = probe_single_state(
                source_file,
                start_gps,
                kernel_set_id=kernel_set_id,
                max_poll_seconds=max_poll_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            stop_ok, stop_result = probe_single_state(
                source_file,
                stop_gps,
                kernel_set_id=kernel_set_id,
                max_poll_seconds=max_poll_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            if not start_ok and not stop_ok:
                return error_records_for_range(source_file, start_gps, stop_gps, short)
            if start_ok and not stop_ok:
                left = download_chunk_records(
                    source_file,
                    start_gps,
                    midpoint - 1.0,
                    kernel_set_id=kernel_set_id,
                    max_poll_seconds=max_poll_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    max_error_span_seconds=max_error_span_seconds,
                )
                return left + error_records_for_range(source_file, midpoint, stop_gps, str(midpoint_result))
            if not start_ok and stop_ok:
                return error_records_for_range(source_file, start_gps, midpoint, str(midpoint_result)) + download_chunk_records(
                    source_file,
                    midpoint + 1.0,
                    stop_gps,
                    kernel_set_id=kernel_set_id,
                    max_poll_seconds=max_poll_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    max_error_span_seconds=max_error_span_seconds,
                )
        if stop_gps - start_gps >= 1.0:
            midpoint = start_gps + int((stop_gps - start_gps) // 2)
            left = download_chunk_records(
                source_file,
                start_gps,
                midpoint,
                kernel_set_id=kernel_set_id,
                max_poll_seconds=max_poll_seconds,
                poll_interval_seconds=poll_interval_seconds,
                max_error_span_seconds=max_error_span_seconds,
            )
            right_start = midpoint + 1.0
            right = (
                download_chunk_records(
                    source_file,
                    right_start,
                    stop_gps,
                    kernel_set_id=kernel_set_id,
                    max_poll_seconds=max_poll_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    max_error_span_seconds=max_error_span_seconds,
                )
                if right_start <= stop_gps + 1e-6
                else []
            )
            return left + right
        return [error_record(source_file, start_gps, short)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_tlm_file",
        "utc",
        "gps_seconds",
        "distance_from_earth_center_km",
        "speed_km_s",
        "earth_rel_bgm1_lugre_x_j2000_km",
        "earth_rel_bgm1_lugre_y_j2000_km",
        "earth_rel_bgm1_lugre_z_j2000_km",
        "earth_rel_bgm1_lugre_vx_j2000_km_s",
        "earth_rel_bgm1_lugre_vy_j2000_km_s",
        "earth_rel_bgm1_lugre_vz_j2000_km_s",
        "time_at_target_utc",
        "light_time_s",
        "wgc_calculation_id",
        "wgc_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download WGC State Vector data aligned to LuGRE TLM rxTime windows."
    )
    parser.add_argument("--tlm-pattern", default="TLM_RAW_*.txt", help="TLM files to scan for rxTime.")
    parser.add_argument("--kernel-set-id", type=int, default=8, help="WGC kernel set ID; 8 is CLPS.")
    parser.add_argument("--max-rows-per-request", type=int, default=3601)
    parser.add_argument("--limit-files", type=int, default=0, help="Debug: only process the first N files.")
    parser.add_argument("--resume", action="store_true", help="Skip per-file CSVs that already exist.")
    parser.add_argument("--max-poll-seconds", type=int, default=600)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-error-span-seconds",
        type=int,
        default=30,
        help="When WGC fails for a short interval, mark all seconds in that interval as errors.",
    )
    args = parser.parse_args()

    windows = collect_tlm_windows(args.tlm_pattern)
    if args.limit_files:
        windows = windows[: args.limit_files]
    if not windows:
        print(f"No TLM windows found for pattern {args.tlm_pattern}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for file_index, window in enumerate(windows, start=1):
        out_path = OUT_DIR / f"{window.path.stem}_wgc_state_vectors.csv"
        if args.resume and out_path.exists():
            print(f"[{file_index}/{len(windows)}] skip existing {out_path.name}")
            with out_path.open("r", encoding="utf-8", newline="") as handle:
                all_rows.extend(csv.DictReader(handle))
            continue

        chunks = iter_chunks(window.start_gps, window.stop_gps, args.max_rows_per_request)
        rows: list[dict[str, Any]] = []
        print(
            f"[{file_index}/{len(windows)}] {window.path.name}: "
            f"{window.count} TLM records, {len(chunks)} WGC request(s)"
        )
        for chunk_index, (start_gps, stop_gps) in enumerate(chunks, start=1):
            print(
                f"  chunk {chunk_index}/{len(chunks)} "
                f"{utc_to_wgc(gps_to_utc(start_gps))} .. {utc_to_wgc(gps_to_utc(stop_gps))}",
                flush=True,
            )
            rows.extend(download_chunk_records(
                window.path.name,
                start_gps,
                stop_gps,
                kernel_set_id=args.kernel_set_id,
                max_poll_seconds=args.max_poll_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                max_error_span_seconds=args.max_error_span_seconds,
            ))
        write_csv(out_path, rows)
        all_rows.extend(rows)
        manifest.append(
            {
                "source_tlm_file": window.path.name,
                "source_tlm_records": window.count,
                "start_gps": window.start_gps,
                "stop_gps": window.stop_gps,
                "start_utc": gps_to_utc(window.start_gps).isoformat().replace("+00:00", "Z"),
                "stop_utc": gps_to_utc(window.stop_gps).isoformat().replace("+00:00", "Z"),
                "wgc_rows": len(rows),
                "output_csv": str(out_path.relative_to(PROJECT_ROOT)),
            }
        )

    combined = OUT_DIR / "wgc_state_vectors_from_observer_by_tlm_all.csv"
    write_csv(combined, all_rows)
    manifest_path = OUT_DIR / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "wgc_api_base": API_BASE,
                "calculation_type": "STATE_VECTOR",
                "kernel_set": "CLPS",
                "kernel_set_id": args.kernel_set_id,
                "target": "EARTH",
                "observer": "BGM1_LUGRE",
                "reference_frame": "J2000",
                "light_propagation": "None",
                "stellar_aberration": "None",
                "aberration_correction": "NONE",
                "state_representation": "RECTANGULAR",
                "time_system": "UTC",
                "time_format": "CALENDAR",
                "time_step_seconds": 1,
                "files": manifest,
                "combined_csv": str(combined.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {combined}")
    print(f"Wrote {manifest_path}")
    print(f"Total WGC rows: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
