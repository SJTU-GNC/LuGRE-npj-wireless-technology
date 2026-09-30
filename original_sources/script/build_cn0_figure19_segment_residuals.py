#!/usr/bin/env python3
"""Reproduce the official Figure 19 operation segments with the frozen full model."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_1S = (
    ROOT
    / "lizhihong"
    / "scripts"
    / "all_OP_CN0_residuals_WGC_excel_ingest.csv"
)
INPUT_1S_MANIFEST = ROOT / "lizhihong" / "scripts" / "input_model_manifest.csv"
INPUT_1S_RUN_LOG = ROOT / "lizhihong" / "scripts" / "run_log_1s.json"
MAPPING = (
    ROOT
    / "data"
    / "external_reference"
    / "mapping"
    / "active_prn_svn_block_20250115_20250316.csv"
)
TLM_DIR = ROOT / "data" / "receiver_observation" / "TLM"
OFFICIAL_HTML = (
    ROOT
    / "data"
    / "external_reference"
    / "lugre_antenna"
    / "ION_NAVIGATION_LuGRE_first_results_article_page_20260709.html"
)
OUT_DIR = (
    ROOT
    / "table"
    / "algorithm"
    / "figure19_segments_full_model_residual"
)
FIGURE_DIR = ROOT / "figure" / "paper_draft_v2" / "extended"

SEGMENTS = [
    ("OP1_0", "15.03 RE"),
    ("OP2_0", "25.25 RE"),
    ("OP21_0", "32.63 RE"),
    ("OP9_0", "34.88 RE"),
    ("OP12_0", "39.93 RE"),
    ("OP5_0", "51.69 RE"),
    ("OP22_0", "52.92 RE"),
    ("OP38_0", "55.85 RE"),
    ("OP40_0", "56.25 RE"),
    ("OP23_0", "61.19 RE"),
    ("OP74_0", "61.90 RE"),
    ("OP27_0", "62.18 RE"),
    ("OP76_0", "62.20 RE"),
    ("OP77_0", "62.42 RE"),
    ("OP78_1", "62.45 RE"),
]
TARGET_BLOCKS = {"BLOCK IIR-M", "BLOCK IIIA"}
CHUNK_SIZE = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def excel_serial_to_utc(series: "pd.Series") -> "pd.Series":
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(
        numeric,
        unit="D",
        origin="1899-12-30",
        utc=True,
    )


def metric_summary(values: "pd.Series") -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(array):
        return {
            key: np.nan
            for key in [
                "mean_db",
                "std_db",
                "rmse_db",
                "mae_db",
                "median_db",
                "q1_db",
                "q3_db",
                "p05_db",
                "p95_db",
                "whisker_low_db",
                "whisker_high_db",
            ]
        }
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    within = array[(array >= q1 - 1.5 * iqr) & (array <= q3 + 1.5 * iqr)]
    return {
        "mean_db": float(np.mean(array)),
        "std_db": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "rmse_db": float(math.sqrt(np.mean(np.square(array)))),
        "mae_db": float(np.mean(np.abs(array))),
        "median_db": float(median),
        "q1_db": float(q1),
        "q3_db": float(q3),
        "p05_db": float(np.quantile(array, 0.05)),
        "p95_db": float(np.quantile(array, 0.95)),
        "whisker_low_db": float(np.min(within)),
        "whisker_high_db": float(np.max(within)),
    }


def draw_boxplot(samples: "pd.DataFrame", output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = [
        samples.loc[
            samples["segment_label"].eq(segment),
            "residual_observed_minus_full_model_db",
        ].to_numpy(float)
        for segment, _ in SEGMENTS
    ]
    counts = np.array([len(values) for values in arrays], dtype=float)
    root_counts = np.sqrt(np.maximum(counts, 1))
    if float(root_counts.max() - root_counts.min()) > 0:
        widths = 0.42 + 0.42 * (
            root_counts - root_counts.min()
        ) / (root_counts.max() - root_counts.min())
    else:
        widths = np.full_like(root_counts, 0.62)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(14.0, 6.5), constrained_layout=True)
    box = ax.boxplot(
        arrays,
        positions=np.arange(1, len(SEGMENTS) + 1),
        widths=widths,
        whis=1.5,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        manage_ticks=False,
        boxprops={"facecolor": "#76A7D5", "edgecolor": "#24527A", "linewidth": 0.9},
        whiskerprops={"color": "#4D4D4D", "linewidth": 0.8},
        capprops={"color": "#4D4D4D", "linewidth": 0.8},
        medianprops={"color": "#C9332B", "linewidth": 1.4},
        meanprops={
            "marker": "+",
            "markeredgecolor": "#C9332B",
            "markerfacecolor": "#C9332B",
            "markersize": 6,
            "markeredgewidth": 1.2,
        },
        flierprops={
            "marker": ".",
            "markerfacecolor": "#8A8A8A",
            "markeredgecolor": "none",
            "markersize": 1.4,
            "alpha": 0.30,
        },
    )
    del box
    ax.axhline(0.0, color="#222222", linewidth=0.9, linestyle="--")
    ax.set_xticks(np.arange(1, len(SEGMENTS) + 1))
    ax.set_xticklabels(
        [f"{segment}\n{distance}" for segment, distance in SEGMENTS],
        rotation=42,
        ha="right",
    )
    ax.set_ylabel(r"$C/N_0$ residual (dB; observed - full model)")
    ax.set_xlabel("LuGRE operation segment and Earth-center distance")
    ax.set_title(
        "Official Figure 19 operation segments evaluated with the frozen full trend model"
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.02,
        "GPS L1, blocks IIR-M and IIIA. Box widths scale with sqrt(n); red + is the mean.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    for suffix in [".svg", ".pdf", ".png"]:
        fig.savefig(output_stem.with_suffix(suffix), dpi=300 if suffix == ".png" else None)
    plt.close(fig)


def main() -> None:
    import numpy as np
    import pandas as pd

    globals()["np"] = np
    globals()["pd"] = pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    segment_files: dict[str, str] = {}
    for segment, _ in SEGMENTS:
        matches = sorted(TLM_DIR.glob(f"TLM_RAW_*_{segment}.txt"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one raw TLM file for {segment}, found {[p.name for p in matches]}"
            )
        segment_files[segment] = matches[0].name
    file_to_segment = {filename: segment for segment, filename in segment_files.items()}

    mapping = pd.read_csv(MAPPING)
    target_mapping = mapping[
        mapping["system"].astype(str).eq("G")
        & mapping["antenna_type"].astype(str).isin(TARGET_BLOCKS)
    ][["prn", "sv_identifier", "antenna_type"]].copy()
    target_prns = set(target_mapping["prn"].astype(str))
    block_by_prn = dict(
        zip(
            target_mapping["prn"].astype(str),
            target_mapping["antenna_type"].astype(str),
            strict=True,
        )
    )
    svn_by_prn = dict(
        zip(
            target_mapping["prn"].astype(str),
            target_mapping["sv_identifier"].astype(str),
            strict=True,
        )
    )

    usecols = [
        "UTC_time",
        "OP",
        "phase",
        "constellation",
        "satellite",
        "signal",
        "band",
        "CN0_observed_raw_dBHz",
        "CN0_model_trend_dBHz",
        "residual_observed_minus_model_dB",
        "alignment_valid",
        "alignment_status",
        "model_nearest_anchor_delta_s",
        "model_bracket_interval_s",
        "link_arc_id",
        "source_file",
        "source_line_number",
        "source_row_id",
    ]
    selected_chunks: list[pd.DataFrame] = []
    target_files = set(file_to_segment)
    for chunk in pd.read_csv(
        INPUT_1S,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        mask = (
            chunk["source_file"].astype(str).isin(target_files)
            & chunk["constellation"].astype(str).eq("GPS")
            & chunk["signal"].astype(str).eq("GPS_L1")
            & chunk["band"].astype(str).eq("L1")
            & chunk["satellite"].astype(str).isin(target_prns)
        )
        if mask.any():
            selected_chunks.append(chunk.loc[mask].copy())
    if not selected_chunks:
        raise RuntimeError("No Figure 19 candidate rows matched the traceable filter")
    candidates = pd.concat(selected_chunks, ignore_index=True)
    candidates["segment_label"] = candidates["source_file"].map(file_to_segment)
    candidates["segment_order"] = candidates["segment_label"].map(
        {segment: index for index, (segment, _) in enumerate(SEGMENTS)}
    )
    candidates["distance_label"] = candidates["segment_label"].map(dict(SEGMENTS))
    candidates["gps_block"] = candidates["satellite"].map(block_by_prn)
    candidates["svn"] = candidates["satellite"].map(svn_by_prn)
    candidates["UTC_time"] = excel_serial_to_utc(candidates["UTC_time"])
    candidates["alignment_valid"] = (
        candidates["alignment_valid"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
        .astype(bool)
    )
    finite = (
        pd.to_numeric(candidates["CN0_observed_raw_dBHz"], errors="coerce").notna()
        & pd.to_numeric(candidates["CN0_model_trend_dBHz"], errors="coerce").notna()
    )
    candidates["figure19_current_model_sample_valid"] = (
        candidates["alignment_valid"] & finite
    )
    candidates["residual_observed_minus_full_model_db"] = (
        pd.to_numeric(candidates["CN0_observed_raw_dBHz"], errors="coerce")
        - pd.to_numeric(candidates["CN0_model_trend_dBHz"], errors="coerce")
    )
    candidates.loc[
        ~candidates["figure19_current_model_sample_valid"],
        "residual_observed_minus_full_model_db",
    ] = np.nan
    valid = candidates[candidates["figure19_current_model_sample_valid"]].copy()
    if valid.empty:
        raise RuntimeError("No aligned full-model rows remain after filtering")
    arithmetic_error = np.abs(
        valid["residual_observed_minus_full_model_db"]
        - pd.to_numeric(valid["residual_observed_minus_model_dB"], errors="coerce")
    )
    if float(arithmetic_error.max()) > 1e-10:
        raise RuntimeError(
            f"Stored residual sign/arithmetic mismatch: {arithmetic_error.max()}"
        )

    stats_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for order, (segment, distance) in enumerate(SEGMENTS):
        all_part = candidates[candidates["segment_label"].eq(segment)]
        part = valid[valid["segment_label"].eq(segment)]
        phase_codes = all_part["phase"].dropna().astype(str).unique().tolist()
        stats_rows.append(
            {
                "segment_order": order + 1,
                "segment_label": segment,
                "distance_label": distance,
                "operation": segment.split("_")[0],
                "mission_phase_code": phase_codes[0] if len(phase_codes) == 1 else "mixed",
                "n": int(len(part)),
                "satellite_count": int(part["satellite"].nunique()),
                "start_utc": (
                    part["UTC_time"].min().isoformat() if not part.empty else ""
                ),
                "end_utc": part["UTC_time"].max().isoformat() if not part.empty else "",
                **metric_summary(part["residual_observed_minus_full_model_db"]),
            }
        )
        coverage_rows.append(
            {
                "segment_order": order + 1,
                "segment_label": segment,
                "source_file": segment_files[segment],
                "candidate_rows_gps_l1_iirm_iiia": int(len(all_part)),
                "valid_current_full_model_rows": int(len(part)),
                "invalid_alignment_rows": int(len(all_part) - len(part)),
                "alignment_coverage": (
                    float(len(part) / len(all_part)) if len(all_part) else np.nan
                ),
                "invalid_reason_counts_json": json.dumps(
                    all_part.loc[
                        ~all_part["figure19_current_model_sample_valid"],
                        "alignment_status",
                    ]
                    .fillna("missing")
                    .astype(str)
                    .value_counts()
                    .to_dict(),
                    sort_keys=True,
                ),
            }
        )
    stats = pd.DataFrame(stats_rows)
    coverage = pd.DataFrame(coverage_rows)
    if (stats["n"] <= 0).any():
        raise RuntimeError(
            "At least one official segment has zero valid current-model residual rows"
        )

    detail_columns = [
        "segment_order",
        "segment_label",
        "distance_label",
        "UTC_time",
        "OP",
        "phase",
        "satellite",
        "svn",
        "gps_block",
        "signal",
        "band",
        "CN0_observed_raw_dBHz",
        "CN0_model_trend_dBHz",
        "residual_observed_minus_full_model_db",
        "model_nearest_anchor_delta_s",
        "model_bracket_interval_s",
        "link_arc_id",
        "source_file",
        "source_line_number",
        "source_row_id",
    ]
    detail = valid[detail_columns].sort_values(
        ["segment_order", "UTC_time", "satellite", "source_row_id"]
    )
    detail_path = OUT_DIR / "figure19_segments_full_model_residual_samples.csv"
    stats_path = OUT_DIR / "figure19_segments_full_model_residual_statistics.csv"
    coverage_path = OUT_DIR / "figure19_segments_alignment_coverage.csv"
    detail.to_csv(
        detail_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%fZ",
    )
    stats.to_csv(stats_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    figure_stem = FIGURE_DIR / "Fig_extended_F19_segments_full_model_residual_boxplot"
    draw_boxplot(detail, figure_stem)

    cadence = (
        detail.sort_values(["segment_label", "satellite", "UTC_time"])
        .groupby(["segment_label", "satellite"])["UTC_time"]
        .diff()
        .dt.total_seconds()
        .dropna()
    )
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "residual_definition": (
            "observed raw approximately-1-Hz TLM C/N0 minus same-timestamp frozen "
            "full trend-model C/N0"
        ),
        "sign": "observed minus model",
        "sample_scale": (
            "raw approximately-1-Hz observations; not the 1-minute trend target"
        ),
        "selection": {
            "operation_segments_in_exact_display_order": [
                {"segment": segment, "distance": distance}
                for segment, distance in SEGMENTS
            ],
            "exact_source_files": segment_files,
            "constellation": "GPS",
            "signal": "GPS_L1",
            "satellite_blocks": sorted(TARGET_BLOCKS),
            "eligible_prns": sorted(target_prns),
            "current_model_alignment_required": True,
        },
        "original_figure_evidence": {
            "article": "Parker et al., LuGRE: First GNSS Results from the Moon",
            "official_figure": "Figure 19",
            "official_caption_method": (
                "Flight minus simulation for GPS L1 blocks IIR-M and IIIA, "
                "summarized by operation; MATLAB boxplot; width proportional to sqrt(n)"
            ),
            "official_figure_url": "https://navi.ion.org/content/navi/73/1/navi.756/F19.large.jpg",
            "official_powerpoint_url": "https://navi.ion.org/highwire/powerpoint/153287",
            "evidence_gap": (
                "The official raster/PPT contains no embedded row-level simulation "
                "table or executable filter. Exact operation files, order, labels, "
                "GPS L1 and block filtering are reproduced; numerical residuals use "
                "the current frozen full model rather than the paper's GGMS simulation."
            ),
        },
        "current_full_model": {
            "definition": (
                "WGC-reference physics-based baseline + training-frozen per-signal "
                "robust median offset (L1 loss) + selected 44-feature HGB slow residual"
            ),
            "one_second_evaluation": (
                "Frozen 1-minute trend anchors linearly interpolated only inside the "
                "same OP x signal x satellite continuous raw link arc; raw gaps >10 s "
                "start a new arc; anchor interval <=90 s; no extrapolation."
            ),
            "observed_cn0_or_residual_used_as_model_input": False,
            "retraining_or_tuning_in_this_task": False,
        },
        "qa": {
            "candidate_rows": int(len(candidates)),
            "valid_residual_rows": int(len(detail)),
            "invalid_alignment_rows": int(len(candidates) - len(detail)),
            "coverage": float(len(detail) / len(candidates)),
            "segment_count": int(detail["segment_label"].nunique()),
            "satellite_count": int(detail["satellite"].nunique()),
            "cadence_positive_delta_mode_s": (
                float(cadence.mode().iloc[0]) if not cadence.mode().empty else None
            ),
            "cadence_positive_delta_median_s": (
                float(cadence[cadence > 0].median()) if (cadence > 0).any() else None
            ),
            "residual_recalculation_max_abs_error_db": float(arithmetic_error.max()),
            "duplicate_source_row_id_count": int(detail["source_row_id"].duplicated().sum()),
        },
        "inputs": {
            "one_second_full_model_delivery": artifact(INPUT_1S),
            "one_second_input_model_manifest": artifact(INPUT_1S_MANIFEST),
            "one_second_run_log": artifact(INPUT_1S_RUN_LOG),
            "prn_svn_block_mapping": artifact(MAPPING),
            "official_article_html": artifact(OFFICIAL_HTML),
            "this_script": artifact(Path(__file__)),
            "raw_segment_files": {
                segment: artifact(TLM_DIR / filename)
                for segment, filename in segment_files.items()
            },
        },
        "outputs": {
            "samples": artifact(detail_path),
            "statistics": artifact(stats_path),
            "coverage": artifact(coverage_path),
            "figure_svg": artifact(figure_stem.with_suffix(".svg")),
            "figure_pdf": artifact(figure_stem.with_suffix(".pdf")),
            "figure_png": artifact(figure_stem.with_suffix(".png")),
        },
    }
    provenance_path = OUT_DIR / "figure19_segments_full_model_residual_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    readme = f"""# Figure 19 operation segments with the current full model

## Exact residual definition

`residual_dB = observed raw approximately-1-Hz TLM C/N0 - same-timestamp
frozen full trend-model C/N0`.

This is a raw-observation-minus-slow-trend diagnostic. It is **not** the formal
1-minute, 9-minute-smoothed trend target used by the LOPO experiment.

## Reproduced evidence

The 15 operation segments, display order, distance labels, GPS L1 restriction,
and block IIR-M/IIIA restriction follow official NAVIGATION Figure 19. The
official downloadable raster and PPT contain no row-level data or embedded
workbook. Therefore this is the closest traceable current-model reproduction:
the sample selection is reproduced, while the original GGMS simulation is
replaced by the frozen LuGRE physics + robust median-offset + HGB trend model.

All {len(candidates):,} matching raw observations are represented in the
coverage table; {len(detail):,} have a valid same-link model interpolation and
enter the boxplot. Invalid rows are not silently discarded.

## Alignment

The frozen model uses WGC reference receiver geometry. One-minute model anchors
are linearly interpolated only within the same OP, satellite, signal, and
continuous raw link arc. A raw gap over 10 s starts a new arc, the maximum model
anchor interval is 90 s, and extrapolation is prohibited.

No model fitting, tuning, or observed-C/N0 feature is used in this task.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(
        f"Wrote {len(detail)} valid residual samples from {len(candidates)} "
        f"candidates across {detail['segment_label'].nunique()} segments"
    )


if __name__ == "__main__":
    main()
