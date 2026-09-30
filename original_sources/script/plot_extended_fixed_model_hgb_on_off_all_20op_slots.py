#!/usr/bin/env python3
"""Render the fixed-model HGB ablation with all 20 OP slots retained.

The 18 eligible operations reuse the same fixed-model metrics and paired
15-min block-bootstrap intervals as the all-eligible-operations figure.
OP3 and OP37 remain on the x axis as explicit insufficient-data placeholders.
No model fitting, parameter update, or prediction recomputation is performed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str((ROOT / "script").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_extended_fixed_model_hgb_on_off_all_operations as core


TABLE_DIR = ROOT / "table" / "algorithm" / "fixed_model_ablation_all_operations"
ALL_20_CSV = TABLE_DIR / "fixed_model_hgb_on_off_all_20op_slots_metrics.csv"
WORKBOOK_PATH = TABLE_DIR / "fixed_model_hgb_on_off_all_20op_slots.xlsx"

FIGURE_DIR = ROOT / "figure" / "paper_draft_v2" / "extended"
STEM = "Fig_extended_fixed_model_hgb_on_off_all_20op_slots_delta_rmse"
OUTPUT_BASE = FIGURE_DIR / STEM
PROVENANCE_PATH = FIGURE_DIR / f"{STEM}_provenance.json"

PREVIOUS_STEM = "Fig_extended_fixed_model_hgb_on_off_all_operations_delta_rmse"
PREVIOUS_OUTPUTS = [
    FIGURE_DIR / f"{PREVIOUS_STEM}{suffix}"
    for suffix in [".png", ".pdf", ".svg"]
] + [
    FIGURE_DIR / f"{PREVIOUS_STEM}_provenance.json",
    TABLE_DIR / "fixed_model_hgb_on_off_all_operations_metrics.csv",
    TABLE_DIR / "fixed_model_hgb_on_off_all_operations_inclusion_audit.csv",
    TABLE_DIR / "fixed_model_hgb_on_off_all_operations.xlsx",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def snapshot(paths: list[Path]) -> dict[str, tuple[int, str]]:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Protected current-version files are missing: {missing}")
    return {
        str(path.resolve()): (int(path.stat().st_size), sha256_file(path))
        for path in paths
    }


def combine_all_operations(
    audit: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    metric_columns = [
        "op",
        "unique_epochs",
        "unique_links",
        "first_utc",
        "last_utc",
        "rmse_hgb_off_dbhz",
        "rmse_hgb_on_dbhz",
        "delta_rmse_on_minus_off_dbhz",
        "delta_rmse_ci95_low_dbhz",
        "delta_rmse_ci95_high_dbhz",
        "ci95_crosses_zero",
        "point_estimate_direction",
        "interval_interpretation",
        "bootstrap_repeats",
        "bootstrap_seed",
    ]
    all_operations = audit.merge(
        metrics[metric_columns],
        on="op",
        how="left",
        validate="one_to_one",
    )
    all_operations = all_operations.sort_values(
        ["phase_order", "op_number"], kind="stable"
    ).reset_index(drop=True)

    if len(all_operations) != 20:
        raise RuntimeError(f"Expected 20 OP slots, found {len(all_operations)}")
    placeholders = all_operations.loc[
        ~all_operations["included_in_figure"], "op"
    ].tolist()
    if placeholders != ["OP3", "OP37"]:
        raise RuntimeError(f"Unexpected placeholder operations: {placeholders}")
    if all_operations.loc[
        ~all_operations["included_in_figure"],
        [
            "rmse_hgb_off_dbhz",
            "rmse_hgb_on_dbhz",
            "delta_rmse_on_minus_off_dbhz",
        ],
    ].notna().any(axis=None):
        raise RuntimeError("Insufficient-data OPs must not receive fabricated metrics")
    return all_operations


def write_workbook(all_operations: pd.DataFrame, metrics: pd.DataFrame) -> None:
    definitions = pd.DataFrame(
        [
            {
                "field": "comparison",
                "definition": (
                    "Same cached rows and same selected model. HGB off = "
                    "physics-based baseline + saved robust median offset; "
                    "HGB on = stored final selected-model prediction."
                ),
            },
            {
                "field": "delta_rmse_on_minus_off_dbhz",
                "definition": (
                    "RMSE(HGB on) - RMSE(HGB off); negative values favour HGB."
                ),
            },
            {
                "field": "95% CI",
                "definition": (
                    "Paired 15-min block bootstrap, 20,000 repeats, within each OP; "
                    "the model is never refitted."
                ),
            },
            {
                "field": "inclusion threshold",
                "definition": (
                    f"At least {core.MIN_COMMON_ROWS} common quality-eligible rows "
                    f"and at least {core.MIN_BOOTSTRAP_BLOCKS} distinct 15-min blocks."
                ),
            },
            {
                "field": "placeholder policy",
                "definition": (
                    "OP3 and OP37 retain x-axis slots but have no bar, CI, or "
                    "fabricated zero metric; the grey x is a visual status marker."
                ),
            },
            {
                "field": "interpretation boundary",
                "definition": (
                    "Operations containing train or validation rows are descriptive "
                    "fixed-model ablations, not independent external evaluations."
                ),
            },
        ]
    )

    with pd.ExcelWriter(WORKBOOK_PATH, engine="openpyxl") as writer:
        all_operations.drop(columns=["phase_order", "op_number"]).to_excel(
            writer, sheet_name="all_20_operations", index=False
        )
        metrics.drop(columns=["phase_order", "op_number"]).to_excel(
            writer, sheet_name="eligible_metrics", index=False
        )
        definitions.to_excel(writer, sheet_name="definitions", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.sheet_view.showGridLines = False
            for column_cells in worksheet.columns:
                values = [
                    str(cell.value) if cell.value is not None else ""
                    for cell in column_cells
                ]
                width = min(max(max(map(len, values), default=0) + 2, 10), 42)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def x_positions_with_phase_gaps(
    all_operations: pd.DataFrame,
) -> tuple[np.ndarray, list[float]]:
    positions: list[float] = []
    separators: list[float] = []
    cursor = 0.0
    previous_phase: str | None = None
    for row in all_operations.itertuples(index=False):
        if previous_phase is not None and row.mission_phase_code != previous_phase:
            cursor += 0.65
            separators.append(cursor - 0.825)
        positions.append(cursor)
        cursor += 1.0
        previous_phase = row.mission_phase_code
    return np.asarray(positions, dtype=float), separators


def plot_figure(all_operations: pd.DataFrame, metrics: pd.DataFrame) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    all_x, separators = x_positions_with_phase_gaps(all_operations)
    position_by_op = dict(zip(all_operations["op"], all_x, strict=True))
    bar_x = metrics["op"].map(position_by_op).to_numpy(float)
    delta = metrics["delta_rmse_on_minus_off_dbhz"].to_numpy(float)
    low = metrics["delta_rmse_ci95_low_dbhz"].to_numpy(float)
    high = metrics["delta_rmse_ci95_high_dbhz"].to_numpy(float)
    colors = [
        core.PHASE_COLORS[phase] for phase in metrics["mission_phase_code"]
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.35), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.25)

    bars = ax.bar(
        bar_x,
        delta,
        width=0.70,
        color=colors,
        edgecolor="#3F4448",
        linewidth=0.45,
        alpha=0.86,
        zorder=2,
    )
    for patch, crosses_zero in zip(bars, metrics["ci95_crosses_zero"], strict=True):
        if bool(crosses_zero):
            patch.set_alpha(0.42)
            patch.set_hatch("//")

    ax.vlines(bar_x, low, high, color="#34383C", linewidth=0.75, zorder=4)
    cap_half_width = 0.11
    ax.hlines(
        low,
        bar_x - cap_half_width,
        bar_x + cap_half_width,
        color="#34383C",
        linewidth=0.75,
        zorder=4,
    )
    ax.hlines(
        high,
        bar_x - cap_half_width,
        bar_x + cap_half_width,
        color="#34383C",
        linewidth=0.75,
        zorder=4,
    )
    ax.scatter(bar_x, delta, s=6.5, c="#292D30", linewidths=0, zorder=5)

    placeholder_rows = all_operations[~all_operations["included_in_figure"]]
    placeholder_x = placeholder_rows["op"].map(position_by_op).to_numpy(float)
    ax.scatter(
        placeholder_x,
        np.zeros(len(placeholder_x)),
        marker="x",
        s=25,
        linewidths=1.05,
        color="#777D82",
        zorder=7,
    )
    for x in placeholder_x:
        ax.annotate(
            "insufficient\ndata",
            xy=(x, 0.0),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#686E73",
            fontsize=5.8,
            linespacing=0.95,
            zorder=7,
        )

    ax.axhline(0.0, color="#25282B", linewidth=0.85, zorder=1)
    ax.grid(axis="y", color="#D8DADD", linewidth=0.45, alpha=0.72, zorder=0)
    for separator in separators:
        ax.axvline(separator, color="#D2D5D8", linewidth=0.55, zorder=0)

    ax.set_xticks(all_x)
    ax.set_xticklabels(
        all_operations["op"], rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel(
        r"$\Delta$RMSE = RMSE(HGB on) $-$ RMSE(HGB off) (dB-Hz)",
        labelpad=5,
    )

    lower = min(float(low.min()), 0.0)
    upper = max(float(high.max()), 0.0)
    span = max(upper - lower, 1.0)
    ax.set_ylim(lower - 0.08 * span, upper + 0.12 * span)
    ax.set_xlim(all_x.min() - 0.7, all_x.max() + 0.7)

    ax.text(
        0.008,
        0.035,
        "negative: HGB improves",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#555B60",
        fontsize=6.4,
    )
    ax.text(
        0.008,
        0.965,
        "positive: HGB worsens",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#555B60",
        fontsize=6.4,
    )

    legend_handles: list[Patch | Line2D] = [
        Patch(
            facecolor=core.PHASE_COLORS[phase],
            edgecolor="#3F4448",
            linewidth=0.45,
            label=core.PHASE_NAMES[phase],
        )
        for phase in ["C", "T", "L", "S"]
    ]
    legend_handles.append(
        Patch(
            facecolor="#FFFFFF",
            edgecolor="#3F4448",
            linewidth=0.55,
            hatch="//",
            label="95% CI includes zero",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
        handlelength=1.25,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    for suffix in [".png", ".pdf", ".svg"]:
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(OUTPUT_BASE.with_suffix(suffix), **kwargs)
    plt.close(fig)


def write_provenance(
    artifact: dict[str, object],
    all_operations: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    placeholders = all_operations[~all_operations["included_in_figure"]]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_stem": STEM,
        "scientific_contract": {
            "comparison_hgb_off": (
                "physics-based baseline plus saved robust median offset"
            ),
            "comparison_hgb_on": "stored final selected-model prediction",
            "delta_definition": "RMSE(HGB on) - RMSE(HGB off); negative values favour HGB",
            "same_row_pairing": True,
            "model_refit_or_parameter_update": False,
            "prediction_reinference": False,
            "placeholder_policy": (
                "All 20 OP x-axis slots are retained. OP3 and OP37 receive a grey "
                "status cross at y=0 and an insufficient-data annotation, but no "
                "bar, CI, or numerical zero metric."
            ),
            "interpretation_boundary": (
                "Operations that include train or validation rows are descriptive "
                "fixed-model ablations, not independent external evaluations."
            ),
        },
        "model_artifact": {
            **file_record(core.MODEL_PATH),
            "model_family_as_stored": str(artifact["model_family"]),
            "model_feature_count": len(artifact["model_features"]),
            "saved_robust_offsets_db": artifact["signal_beta_db"],
        },
        "prediction_cache": file_record(core.PREDICTION_PATH),
        "operation_manifest": file_record(core.SPLIT_MANIFEST_PATH),
        "eligibility": {
            "operation_slot_count": int(len(all_operations)),
            "bar_and_ci_count": int(len(metrics)),
            "placeholder_count": int(len(placeholders)),
            "minimum_common_quality_eligible_rows": core.MIN_COMMON_ROWS,
            "minimum_distinct_15min_blocks": core.MIN_BOOTSTRAP_BLOCKS,
            "placeholder_operations": placeholders[
                [
                    "op",
                    "mission_phase_name",
                    "common_quality_eligible_rows",
                    "bootstrap_15min_blocks",
                    "exclusion_reason",
                ]
            ].to_dict(orient="records"),
        },
        "uncertainty": {
            "method": "paired within-operation 15-min block bootstrap",
            "repeats": core.BOOTSTRAP_REPEATS,
            "confidence_interval": "2.5th and 97.5th percentiles",
            "model_refit_within_bootstrap": False,
            "base_seed": core.BOOTSTRAP_BASE_SEED,
        },
        "all_operation_rows": json.loads(
            all_operations.drop(columns=["phase_order", "op_number"])
            .round(10)
            .to_json(orient="records", date_format="iso")
        ),
        "outputs": {
            "all_20_operation_csv": file_record(ALL_20_CSV),
            "workbook": file_record(WORKBOOK_PATH),
            "figure_files": [
                file_record(OUTPUT_BASE.with_suffix(suffix))
                for suffix in [".png", ".pdf", ".svg"]
            ],
            "plot_script": file_record(Path(__file__)),
        },
        "figure_contract": {
            "archetype": "quantitative comparison",
            "whole_figure_title": False,
            "x_axis_operation_slots": 20,
            "bars_with_ci": 18,
            "insufficient_data_markers": 2,
            "insufficient_marker": "grey x at y=0 with two-line annotation",
            "phase_encoding": "restrained bar fill and legend",
            "zero_reference_line": True,
            "ci_crosses_zero_encoding": "reduced opacity and hatch",
            "white_background": True,
            "png_dpi": 600,
            "svg_text_editable": True,
            "pdf_fonttype": 42,
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_outputs(
    all_operations: pd.DataFrame, metrics: pd.DataFrame
) -> dict[str, object]:
    outputs = [
        ALL_20_CSV,
        WORKBOOK_PATH,
        OUTPUT_BASE.with_suffix(".png"),
        OUTPUT_BASE.with_suffix(".pdf"),
        OUTPUT_BASE.with_suffix(".svg"),
        PROVENANCE_PATH,
    ]
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")

    svg_text = OUTPUT_BASE.with_suffix(".svg").read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG does not retain editable text")
    for operation in all_operations["op"]:
        if operation not in svg_text:
            raise RuntimeError(f"Missing OP slot in SVG: {operation}")
    if svg_text.count("insufficient") != 2:
        raise RuntimeError("Expected exactly two insufficient-data annotations")
    if len(all_operations) != 20 or len(metrics) != 18:
        raise RuntimeError("Unexpected OP-slot or bar count")

    return {
        "operation_slots": 20,
        "bars_with_ci": 18,
        "insufficient_data_markers": ["OP3", "OP37"],
        "ci_crosses_zero_count": int(metrics["ci95_crosses_zero"].sum()),
        "svg_editable_text": True,
        "png_dpi": 600,
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    protected = [
        core.MODEL_PATH,
        core.PREDICTION_PATH,
        core.SPLIT_MANIFEST_PATH,
        *PREVIOUS_OUTPUTS,
    ]
    before = snapshot(protected)

    core.verify_authoritative_inputs()
    artifact = core.load_model_artifact()
    universe = core.load_operation_universe()
    predictions = core.load_cached_predictions(artifact)
    audit, common_rows = core.build_inclusion_audit(universe, predictions)
    metrics = core.build_metrics(audit, common_rows)
    all_operations = combine_all_operations(audit, metrics)

    all_operations.drop(columns=["phase_order", "op_number"]).to_csv(
        ALL_20_CSV, index=False, float_format="%.10f"
    )
    write_workbook(all_operations, metrics)
    plot_figure(all_operations, metrics)
    write_provenance(artifact, all_operations, metrics)
    qa = verify_outputs(all_operations, metrics)

    after = snapshot(protected)
    if after != before:
        raise RuntimeError("A protected input or current-version output changed")

    print(json.dumps(qa, indent=2))
    print(
        all_operations[
            [
                "op",
                "mission_phase_name",
                "common_quality_eligible_rows",
                "bootstrap_15min_blocks",
                "included_in_figure",
                "delta_rmse_on_minus_off_dbhz",
                "exclusion_reason",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
