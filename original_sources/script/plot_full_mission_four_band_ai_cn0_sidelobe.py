#!/usr/bin/env python3
"""Plot full-mission four-band C/N0 trends from the frozen unified AI model.

The light-red background denotes epochs where at least one displayed link is
outside the official GNSS SSV main-lobe service-angle boundary.  This is a
service-angle sidelobe proxy, not a measured electromagnetic lobe label.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "runtime_cache" / "python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "script"))

import joblib
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


CONTINUOUS_PATH = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_wgc_multiband_complete"
    / "cn0_wgc_multiband_complete_predictions.csv"
)
MODEL_PATH = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_trend_residual_tuned_no_leakage"
    / "cn0_trend_residual_tuned_no_leakage.joblib"
)
GRAP_PATH = ROOT / "table" / "external_reference" / "galileo_grap_eirp_grid.csv"
SELECTION_PATH = (
    ROOT / "table" / "algorithm" / "cn0_wgc_multiband_complete" / "cn0_wgc_multiband_selection.csv"
)
OBSERVATION_PATH = (
    ROOT
    / "table"
    / "algorithm"
    / "cn0_constellation_physics_baseline"
    / "cn0_constellation_physics_predictions.csv"
)
FIG_DIR = ROOT / "figure" / "paper_draft_v2"
SOURCE_DIR = ROOT / "table" / "figure_source_data"
STEM = "Fig_full_mission_four_band_AI_CN0_sidelobe"

SIGNALS = ["GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"]
DISPLAY_NAMES = {
    "GPS_L1": "GPS L1",
    "GPS_L5": "GPS L5",
    "GAL_E1": "Galileo E1",
    "GAL_E5a": "Galileo E5a",
}
PHASE_NAMES = {"C": "Commissioning", "T": "Trans-lunar", "L": "Lunar orbit", "S": "Surface"}
COLORS = ["#E69F00", "#365A8C", "#2A8C82"]
SIDELOBE_COLOR = "#F6C5C2"
DISPLAY_MIN_DBHZ = 10.0
DISPLAY_MAX_DBHZ = 45.0
MODEL_MAX_LINK_GAP = pd.Timedelta("7min30s")
PLOT_MAX_LINK_GAP = pd.Timedelta("12min30s")
DISPLAY_RESAMPLE: str | None = None
DISPLAY_EPOCH_HALF_WIDTH = pd.Timedelta(minutes=2.5)
DISPLAY_CADENCE_NOTE = "5-min geometry epochs"
SIDELOBE_WINDOW_FRACTION_THRESHOLD = 0.5
MODEL_TREND_MINUTES = 9


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def smooth_at_native_epochs(
    frame: pd.DataFrame, value_column: str, time_column: str = "utc"
) -> pd.Series:
    """Apply the model's nine-minute trend head on an interpolated 1-min grid."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    ordered = frame.sort_values(time_column)
    usable = ordered[pd.to_numeric(ordered[value_column], errors="coerce").notna()]
    if usable.empty:
        return result
    segment_ids = usable[time_column].diff().gt(MODEL_MAX_LINK_GAP).cumsum()
    window = MODEL_TREND_MINUTES if MODEL_TREND_MINUTES % 2 else MODEL_TREND_MINUTES + 1
    for _, segment in usable.groupby(segment_ids):
        values = pd.to_numeric(segment[value_column], errors="coerce")
        native = pd.Series(values.to_numpy(float), index=pd.DatetimeIndex(segment[time_column]))
        native = native[~native.index.duplicated(keep="first")].sort_index()
        if native.empty:
            continue
        minute_index = pd.date_range(native.index.min(), native.index.max(), freq="1min", tz="UTC")
        minute_values = native.reindex(native.index.union(minute_index)).sort_index().interpolate(method="time")
        minute_values = minute_values.reindex(minute_index)
        filtered = minute_values.rolling(window=window, center=True, min_periods=1).median()
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
        sampled = filtered.reindex(native.index, method="nearest")
        result.loc[segment.index] = sampled.to_numpy(float)
    return result


def add_grap_diagnostic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the Galileo azimuth-aggregate diagnostics used during training."""
    grap = pd.read_csv(GRAP_PATH, low_memory=False)
    numeric = [
        "azimuth_deg",
        "coelevation_deg",
        "eirp_dbw",
        "eirp_upper95_dbw",
        "eirp_lower95_dbw",
    ]
    for column in numeric:
        grap[column] = pd.to_numeric(grap[column], errors="coerce")
    aggregate = (
        grap.groupby(["signal_name", "coelevation_deg"], as_index=False)
        .agg(
            tx_eirp_grap_azimuth_median_dbw=("eirp_dbw", "median"),
            tx_eirp_grap_azimuth_mean_dbw=("eirp_dbw", "mean"),
            tx_eirp_grap_azimuth_min_dbw=("eirp_dbw", "min"),
            tx_eirp_grap_azimuth_max_dbw=("eirp_dbw", "max"),
            tx_eirp_grap_upper95_median_dbw=("eirp_upper95_dbw", "median"),
            tx_eirp_grap_lower95_median_dbw=("eirp_lower95_dbw", "median"),
            tx_eirp_grap_azimuth_samples=("azimuth_deg", "count"),
        )
        .rename(columns={"coelevation_deg": "tx_eirp_grap_lookup_coelevation_deg"})
    )
    output = data.copy()
    output["tx_eirp_grap_lookup_coelevation_deg"] = (
        pd.to_numeric(output["tx_theta_body_deg"], errors="coerce")
        .clip(lower=0, upper=90)
        .round()
        .astype("Int64")
    )
    output = output.merge(
        aggregate,
        on=["signal_name", "tx_eirp_grap_lookup_coelevation_deg"],
        how="left",
        sort=False,
    )
    galileo = output["signal_name"].isin(["GAL_E1", "GAL_E5a"])
    common = (
        -pd.to_numeric(output["fspl_db"], errors="coerce")
        + pd.to_numeric(output["rx_peak_gain_dbic"], errors="coerce")
        + 228.6
        - 10.0 * np.log10(pd.to_numeric(output["system_noise_temperature_k"], errors="coerce"))
        - pd.to_numeric(output["implementation_loss_assumed_db"], errors="coerce")
        - pd.to_numeric(output["l_ion_abs_budget_db"], errors="coerce")
        - pd.to_numeric(output["l_gas_abs_budget_db"], errors="coerce")
    )
    for source, target in [
        ("tx_eirp_grap_azimuth_median_dbw", "cn0_galileo_grap_peakrx_dbhz"),
        ("tx_eirp_grap_upper95_median_dbw", "cn0_galileo_grap_upper95_peakrx_dbhz"),
        ("tx_eirp_grap_lower95_median_dbw", "cn0_galileo_grap_lower95_peakrx_dbhz"),
    ]:
        output[target] = np.nan
        output.loc[galileo, target] = (
            pd.to_numeric(output.loc[galileo, source], errors="coerce")
            + common.loc[galileo]
        )
    return output


def load_and_predict() -> pd.DataFrame:
    artifact = joblib.load(MODEL_PATH)
    features = list(artifact["model_features"])
    derived = {
        "tx_ssv_signed_lower_boundary_deg",
        "tx_ssv_signed_upper_boundary_deg",
        "cn0_physics_trend_dbhz",
    }
    metadata = [
        "utc",
        "op",
        "evaluation_split",
        "mission_phase",
        "system",
        "signal_name",
        "svid",
        "earth_blocked",
        "moon_blocked",
        "tx_ssv_main_lobe_classification",
        "svn",
        "tx_theta_body_deg",
        "tx_phi_body_deg",
        "tx_gain_2d_db",
        "tx_eirp_2d_dbw",
        "tx_pattern_source",
        "tx_pattern_family",
        "tx_pattern_coverage",
        "tx_ssv_main_lobe_boundary_deg",
        "rx_attitude_status",
        "rx_offboresight_spice_deg",
        "rx_gain_envelope_dbic",
        "tx_attitude_source",
        "tx_yaw_quality",
        "receiver_trajectory_source",
        "dynamic_stage",
        "dynamic_quality_status",
        "dynamic_status_priority",
        "dynamic_segment_id",
        "reference_position_error_km",
        "gnss_orbit_source",
        "orbit_geometry_status",
    ]
    requested = list(dict.fromkeys(metadata + [column for column in features if column not in derived]))
    available = set(pd.read_csv(CONTINUOUS_PATH, nrows=0).columns)
    usecols = [column for column in requested if column in available]
    data = pd.read_csv(CONTINUOUS_PATH, usecols=usecols, low_memory=False)
    data["utc"] = pd.to_datetime(data["utc"], utc=True, errors="coerce")
    data["svid"] = pd.to_numeric(data["svid"], errors="coerce").astype("Int64")
    data = data[data["utc"].notna() & data["signal_name"].isin(SIGNALS)].copy()

    boundary = pd.to_numeric(data["tx_ssv_main_lobe_boundary_deg"], errors="coerce")
    data["tx_ssv_signed_lower_boundary_deg"] = -boundary
    data["tx_ssv_signed_upper_boundary_deg"] = boundary
    data = add_grap_diagnostic_features(data)
    data["cn0_physics_trend_dbhz"] = np.nan
    for _, group in data.groupby(["signal_name", "svid"], dropna=False, sort=False):
        data.loc[group.index, "cn0_physics_trend_dbhz"] = smooth_at_native_epochs(
            group, "cn0_constellation_direct_available_dbhz"
        ).loc[group.index]

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise RuntimeError(f"Frozen-model features remain unavailable: {missing}")

    valid = (
        pd.to_numeric(data["cn0_constellation_direct_available_dbhz"], errors="coerce").notna()
        & data["earth_blocked"].eq(0)
        & data["moon_blocked"].eq(0)
    )
    data["ai_residual_raw_db"] = np.nan
    beta = data.loc[valid, "signal_name"].map(artifact["signal_beta_db"]).fillna(0.0).to_numpy(float)
    data.loc[valid, "ai_residual_raw_db"] = (
        np.asarray(artifact["model"].predict(data.loc[valid, features]), dtype=float) + beta
    )
    data["ai_residual_trend_db"] = np.nan
    for _, group in data.groupby(["signal_name", "svid"], dropna=False, sort=False):
        data.loc[group.index, "ai_residual_trend_db"] = smooth_at_native_epochs(
            group, "ai_residual_raw_db"
        ).loc[group.index]
    data["cn0_physics_ai_trend_dbhz"] = (
        data["cn0_physics_trend_dbhz"] + data["ai_residual_trend_db"]
    )
    data["display_eligible"] = valid & data["cn0_physics_ai_trend_dbhz"].notna()
    data["cn0_display_dbhz"] = data["cn0_physics_ai_trend_dbhz"].clip(
        lower=DISPLAY_MIN_DBHZ,
        upper=DISPLAY_MAX_DBHZ,
    )
    data["display_value_clipped"] = (
        data["display_eligible"]
        & ~data["cn0_physics_ai_trend_dbhz"].between(DISPLAY_MIN_DBHZ, DISPLAY_MAX_DBHZ)
    )
    data["sidelobe_background_eligible"] = (
        valid
        & data["cn0_physics_ai_trend_dbhz"].between(
            DISPLAY_MIN_DBHZ,
            DISPLAY_MAX_DBHZ,
        )
    )
    data["outside_ssv_main_lobe"] = data["tx_ssv_main_lobe_classification"].eq(
        "outside_ssv_main_lobe"
    )
    data["sidelobe_fraction_in_display_bin"] = data["outside_ssv_main_lobe"].astype(float)
    data["model_artifact"] = MODEL_PATH.name
    cadence_seconds = (
        data[["utc", "signal_name", "svid"]]
        .sort_values(["signal_name", "svid", "utc"])
        .groupby(["signal_name", "svid"], dropna=False)["utc"]
        .diff()
        .dt.total_seconds()
    )
    cadence_seconds = cadence_seconds[cadence_seconds.gt(0)]
    native_cadence_minutes = (
        float(cadence_seconds.median() / 60.0) if not cadence_seconds.empty else np.nan
    )
    data["model_output_cadence"] = (
        f"1-min-trained residual model evaluated on native "
        f"{native_cadence_minutes:g}-min WGC/SP3 physics epochs; "
        "nine-minute robust trend head"
    )
    return data


def select_one_common_per_constellation(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_counts = (
        data[data["display_eligible"]]
        .groupby(["signal_name", "svid"], as_index=False)
        .agg(
            valid_model_epochs=("utc", "size"),
            first_utc=("utc", "min"),
            last_utc=("utc", "max"),
            sidelobe_epochs=("outside_ssv_main_lobe", "sum"),
        )
    )
    model_counts["sidelobe_fraction"] = (
        model_counts["sidelobe_epochs"] / model_counts["valid_model_epochs"]
    )

    observations = pd.read_csv(
        OBSERVATION_PATH, usecols=["signal_name", "svid"], low_memory=False
    )
    observations["svid"] = pd.to_numeric(observations["svid"], errors="coerce").astype("Int64")
    observation_counts = (
        observations.groupby(["signal_name", "svid"], as_index=False)
        .size()
        .rename(columns={"size": "tlm_observation_rows"})
    )
    selection = pd.read_csv(SELECTION_PATH)
    ranked = {
        str(row.signal_name): [int(value) for value in str(row.selected_svids).split(",")]
        for row in selection.itertuples()
    }
    chosen_rows: list[dict] = []
    for signal_pair in [("GPS_L1", "GPS_L5"), ("GAL_E1", "GAL_E5a")]:
        common = set(ranked[signal_pair[0]]) & set(ranked[signal_pair[1]])
        if not common:
            raise RuntimeError(f"No common satellite is available for {signal_pair}")
        selected_svid = min(
            common,
            key=lambda svid: (
                ranked[signal_pair[0]].index(svid) + ranked[signal_pair[1]].index(svid),
                svid,
            ),
        )
        for signal_name in signal_pair:
            chosen_rows.append(
                {"signal_name": signal_name, "svid": selected_svid, "rank_within_band": 1}
            )
    selected = pd.DataFrame(chosen_rows)
    selected = selected.merge(observation_counts, on=["signal_name", "svid"], how="left")
    selected = selected.merge(model_counts, on=["signal_name", "svid"], how="left")
    selected["selection_basis"] = (
        "highest combined rank shared by both frequency bands of the same constellation"
    )
    keep = pd.Series(False, index=data.index)
    for row in selected.itertuples():
        keep |= data["signal_name"].eq(row.signal_name) & data["svid"].eq(row.svid)
    return data[keep].copy(), selected


def contiguous_segments(track: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = track.sort_values("utc")
    usable = ordered[ordered["display_eligible"]].copy()
    if usable.empty:
        return []
    breaks = usable["utc"].diff().gt(PLOT_MAX_LINK_GAP).fillna(False)
    hard_blocked = ordered["earth_blocked"].eq(1) | ordered["moon_blocked"].eq(1)
    usable_times = usable["utc"].to_list()
    for index in range(1, len(usable_times)):
        if breaks.iloc[index]:
            continue
        between = ordered["utc"].gt(usable_times[index - 1]) & ordered["utc"].lt(usable_times[index])
        if bool(hard_blocked.loc[between].any()):
            breaks.iloc[index] = True
    segment_ids = breaks.cumsum()
    return [segment for _, segment in usable.groupby(segment_ids) if len(segment) >= 2]


def resample_for_display(data: pd.DataFrame) -> pd.DataFrame:
    if DISPLAY_RESAMPLE is None:
        return data
    rows: list[pd.Series] = []
    for _, track in data.groupby(["signal_name", "svid"], dropna=False, sort=False):
        ordered = track.sort_values("utc").copy()
        ordered["display_bin"] = ordered["utc"].dt.floor(DISPLAY_RESAMPLE)
        for bin_start, window in ordered.groupby("display_bin", sort=True):
            representative = window.iloc[len(window) // 2].copy()
            valid = window[window["display_eligible"]]
            background_valid = window[window["sidelobe_background_eligible"]]
            representative["utc"] = pd.Timestamp(bin_start) + pd.Timedelta(DISPLAY_RESAMPLE) / 2
            representative["model_output_cadence"] = (
                f"{DISPLAY_RESAMPLE} median display output from the 1-min-trained model "
                "evaluated at 5-min geometry epochs"
            )
            if valid.empty:
                representative["cn0_physics_ai_trend_dbhz"] = np.nan
                representative["cn0_display_dbhz"] = np.nan
                representative["display_eligible"] = False
                representative["display_value_clipped"] = False
                representative["outside_ssv_main_lobe"] = False
                representative["sidelobe_fraction_in_display_bin"] = np.nan
                representative["sidelobe_background_eligible"] = False
            else:
                value = float(pd.to_numeric(valid["cn0_physics_ai_trend_dbhz"], errors="coerce").median())
                representative["cn0_physics_ai_trend_dbhz"] = value
                representative["cn0_display_dbhz"] = float(
                    np.clip(value, DISPLAY_MIN_DBHZ, DISPLAY_MAX_DBHZ)
                )
                representative["display_eligible"] = True
                representative["display_value_clipped"] = not (
                    DISPLAY_MIN_DBHZ <= value <= DISPLAY_MAX_DBHZ
                )
                representative["sidelobe_background_eligible"] = not background_valid.empty
                if background_valid.empty:
                    representative["sidelobe_fraction_in_display_bin"] = np.nan
                    representative["outside_ssv_main_lobe"] = False
                else:
                    sidelobe_fraction = float(
                        background_valid["outside_ssv_main_lobe"].astype(bool).mean()
                    )
                    representative["sidelobe_fraction_in_display_bin"] = sidelobe_fraction
                    representative["outside_ssv_main_lobe"] = bool(
                        sidelobe_fraction >= SIDELOBE_WINDOW_FRACTION_THRESHOLD
                    )
            rows.append(representative.drop(labels=["display_bin"], errors="ignore"))
    return pd.DataFrame(rows).sort_values(["signal_name", "svid", "utc"]).reset_index(drop=True)


def sidelobe_intervals(panel: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    side = panel[
        panel["sidelobe_background_eligible"]
        & panel["outside_ssv_main_lobe"]
    ]
    if side.empty:
        return []
    epochs = pd.Series(sorted(side["utc"].dropna().unique()))
    groups = epochs.diff().gt(PLOT_MAX_LINK_GAP).cumsum()
    return [
        (
            pd.Timestamp(group.iloc[0]) - DISPLAY_EPOCH_HALF_WIDTH,
            pd.Timestamp(group.iloc[-1]) + DISPLAY_EPOCH_HALF_WIDTH,
        )
        for _, group in epochs.groupby(groups)
    ]


def phase_intervals(data: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    phases = (
        data.groupby("utc")["mission_phase"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else str(values.iloc[0]))
        .sort_index()
    )
    changed = phases.ne(phases.shift()).cumsum()
    return [
        (str(group.iloc[0]), pd.Timestamp(group.index.min()), pd.Timestamp(group.index.max()))
        for _, group in phases.groupby(changed)
    ]


def plot(data: pd.DataFrame, selected: pd.DataFrame) -> Path:
    configure_style()
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 7.0), sharex=True)
    phases = phase_intervals(data)

    for panel_index, (ax, signal) in enumerate(zip(axes, SIGNALS)):
        panel = data[data["signal_name"].eq(signal)].copy()
        for start, stop in sidelobe_intervals(panel):
            ax.axvspan(start, stop, color=SIDELOBE_COLOR, alpha=0.42, lw=0, zorder=0)

        chosen = selected[selected["signal_name"].eq(signal)].sort_values("rank_within_band")
        for color, row in zip(COLORS, chosen.itertuples()):
            track = panel[panel["svid"].eq(row.svid)]
            first = True
            for segment in contiguous_segments(track):
                ax.plot(
                    segment["utc"],
                    segment["cn0_display_dbhz"],
                    color=color,
                    lw=1.05,
                    alpha=0.98,
                    solid_capstyle="round",
                    label=f"SVID {int(row.svid)}" if first else None,
                    zorder=3,
                )
                first = False

        for phase, start, _ in phases[1:]:
            ax.axvline(start, color="#8B9298", lw=0.45, ls=(0, (2, 2)), alpha=0.65, zorder=1)
        ax.set_title(DISPLAY_NAMES[signal], loc="left", fontsize=8.3, fontweight="bold", pad=3)
        ax.set_ylabel(r"$C/N_0$ (dB-Hz)")
        ax.set_ylim(DISPLAY_MIN_DBHZ, DISPLAY_MAX_DBHZ)
        ax.set_yticks([10, 20, 30, 40])
        ax.grid(True, color="#D8DDE2", lw=0.5, alpha=0.88, zorder=0)
        ax.legend(loc="upper right", ncol=3, handlelength=2.2, columnspacing=1.25)
        ax.text(-0.042, 1.02, chr(ord("a") + panel_index), transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="bottom")

    start_all = data["utc"].min()
    stop_all = data["utc"].max()
    for phase, start, stop in phases:
        midpoint = start + (stop - start) / 2
        if start_all <= midpoint <= stop_all:
            axes[0].text(
                midpoint,
                1.035,
                phase,
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6.4,
                color="#5B6268",
                clip_on=False,
            )

    axes[-1].set_xlabel("Time (UTC, 2025)")
    locator = mdates.AutoDateLocator(minticks=10, maxticks=16)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlim(start_all, stop_all)

    semantics = [
        plt.Line2D([], [], color="#30363B", lw=1.2, label="Physics + AI residual model output"),
        Patch(
            facecolor=SIDELOBE_COLOR,
            edgecolor="none",
            alpha=0.55,
            label="Side-lobe-dominant interval (>50% outside SSV main-lobe boundary)",
        ),
    ]
    fig.legend(
        handles=semantics,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.964),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#697077",
        facecolor="white",
        framealpha=1.0,
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.suptitle("Full-mission AI-model $C/N_0$ trends", y=0.998, fontsize=10.2, fontweight="bold")
    fig.text(
        0.5,
        0.008,
        f"Curves: frozen unified 1-min-trained model displayed at {DISPLAY_CADENCE_NOTE}; values outside 10-45 dB-Hz are clipped at the display boundary. Light red: among epochs within the displayed C/N0 range, at least 50% are outside the official SSV main-lobe service-angle boundary; this is not a measured antenna-pattern lobe label.",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="#50585F",
    )
    fig.tight_layout(rect=[0.025, 0.035, 0.995, 0.925], h_pad=0.8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / STEM
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".png")


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_predict()
    selected_data, selected = select_one_common_per_constellation(data)
    selected_data = resample_for_display(selected_data)
    selected_data.to_csv(
        SOURCE_DIR / f"{STEM}_source_data.csv", index=False, encoding="utf-8-sig"
    )
    selected.to_csv(
        SOURCE_DIR / f"{STEM}_satellite_selection.csv", index=False, encoding="utf-8-sig"
    )
    output = plot(selected_data, selected)
    print(selected.to_string(index=False))
    print(f"source_rows={len(selected_data):,}")
    print(output)


if __name__ == "__main__":
    main()
