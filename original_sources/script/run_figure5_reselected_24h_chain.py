#!/usr/bin/env python3
"""Run and publish the audited Figure 5 model/observation-window chain."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"
FIGURE_DIR = ROOT / "figure" / "paper_draft_v2"
SOURCE_DIR = ROOT / "table" / "figure_source_data"
FINAL_DIR = ROOT / "figure" / "定稿图"

BASE_STEM = "Fig_full_mission_four_band_AI_CN0_1min_one_sat_24h"
VARIANT_STEM = f"{BASE_STEM}_reselected_24h"
FINAL_STEM = (
    f"{VARIANT_STEM}_interleaved_details_wide_compact_"
    "left_top_trimmed_all_sides_trimmed"
)
LOG_PATH = SOURCE_DIR / "figure5_reselected_24h_chain_run.log"
MANIFEST_PATH = SOURCE_DIR / "figure5_reselected_24h_chain_manifest.json"

CHAIN = [
    "plot_full_mission_four_band_ai_cn0_sidelobe_1min_one_sat_24h.py",
    "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_interleaved_details.py",
    (
        "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_"
        "interleaved_details_readable_timeline.py"
    ),
    (
        "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_"
        "interleaved_details_readable_timeline_legend_up_large_detail_numbers.py"
    ),
    (
        "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_"
        "interleaved_details_readable_timeline_legend_up_large_detail_numbers_"
        "wide_compact.py"
    ),
    (
        "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_"
        "interleaved_details_readable_timeline_legend_up_large_detail_numbers_"
        "wide_compact_left_trimmed.py"
    ),
    (
        "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_"
        "interleaved_details_wide_compact_left_top_trimmed.py"
    ),
    (
        "plot_full_mission_four_band_ai_cn0_1min_one_sat_24h_"
        "interleaved_details_wide_compact_left_top_trimmed_"
        "all_sides_trimmed.py"
    ),
]

PROTECTED = [
    ROOT
    / "table"
    / "algorithm"
    / "cn0_dynamic_multiband_one_common_1min"
    / "cn0_dynamic_multiband_one_common_1min_predictions.csv",
    ROOT
    / "table"
    / "algorithm"
    / "cn0_dynamic_multiband_one_common_1min"
    / "cn0_dynamic_multiband_one_common_1min_selection.csv",
    ROOT
    / "table"
    / "algorithm"
    / "cn0_trend_residual_tuned_no_leakage"
    / "cn0_trend_residual_tuned_no_leakage.joblib",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def snapshot(paths: list[Path]) -> dict[str, dict[str, object]]:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Protected Figure 5 input is missing: {missing}")
    return {str(path.resolve()): record(path) for path in paths}


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    prior_final = FINAL_DIR / "figure5.png"
    prior_backup = (
        FINAL_DIR / "figure5_before_model_observation_window_revision_20260727.png"
    )
    if prior_final.is_file() and not prior_backup.exists():
        shutil.copy2(prior_final, prior_backup)
    protected_before = snapshot(PROTECTED)
    run_records: list[dict[str, object]] = []
    with LOG_PATH.open("w", encoding="utf-8") as log:
        for index, script_name in enumerate(CHAIN, start=1):
            script_path = SCRIPT_DIR / script_name
            started = datetime.now(timezone.utc)
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=SCRIPT_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            finished = datetime.now(timezone.utc)
            log.write(f"===== {index}/{len(CHAIN)} {script_name} =====\n")
            log.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                log.write("\n")
            log.flush()
            run_records.append(
                {
                    "order": index,
                    "script": script_name,
                    "returncode": int(completed.returncode),
                    "started_utc": started.isoformat(),
                    "finished_utc": finished.isoformat(),
                    "elapsed_seconds": (finished - started).total_seconds(),
                }
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Figure 5 chain failed at {script_name}; see {LOG_PATH}"
                )

    protected_after = snapshot(PROTECTED)
    if protected_before != protected_after:
        raise RuntimeError("A protected original Figure 5 file changed")

    generated = sorted(
        path
        for path in FIGURE_DIR.glob(f"{VARIANT_STEM}*")
        if path.is_file()
    )
    final_outputs = [
        FIGURE_DIR / f"{FINAL_STEM}.{suffix}"
        for suffix in ("png", "pdf", "svg", "tiff")
    ]
    missing_final = [path for path in final_outputs if not path.is_file()]
    if missing_final:
        raise RuntimeError(f"Final reselected outputs are missing: {missing_final}")
    published_outputs = []
    for source_path in final_outputs:
        target = FINAL_DIR / f"figure5{source_path.suffix}"
        shutil.copy2(source_path, target)
        published_outputs.append(target)
    final_provenance = FIGURE_DIR / f"{FINAL_STEM}_provenance.json"
    published_provenance = FINAL_DIR / "figure5_provenance.json"
    shutil.copy2(final_provenance, published_provenance)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Figure 5 only: common sliding 24-hour detail-window reselection",
        "chain": run_records,
        "protected_files_unchanged": True,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "window_audit": record(
            SOURCE_DIR / f"{VARIANT_STEM}_window_selection_audit.csv"
        ),
        "panel_audit": record(
            SOURCE_DIR / f"{VARIANT_STEM}_chronological_panel_audit.csv"
        ),
        "final_outputs": [record(path) for path in final_outputs],
        "published_outputs": [record(path) for path in published_outputs],
        "published_provenance": record(published_provenance),
        "prior_final_backup": (
            record(prior_backup) if prior_backup.is_file() else None
        ),
        "all_generated_variant_files": [record(path) for path in generated],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["chain"], indent=2))
    print(FIGURE_DIR / f"{FINAL_STEM}.png")
    print(LOG_PATH)
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
