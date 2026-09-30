"""Summarize the saved final-model rows; no model fitting or prediction."""
from pathlib import Path
import hashlib
import json
import math

import pandas as pd


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
SOURCE = PROJECT / "analysis/phase_compensation_attribution_v4/attrib_rows.csv"
CHECK = SOURCE.with_name("attrib_summary_all.csv")
SIGNALS = ["GPS_L1", "GPS_L5", "GAL_E1", "GAL_E5a"]
OPS = [(1, "C"), (2, "C"), (3, "T"), (5, "T"), (9, "T"),
       (12, "T"), (14, "T"), (17, "T"), (18, "T"), (21, "T"),
       (22, "T"), (23, "L"), (27, "L"), (37, "L"), (38, "S"),
       (40, "S"), (74, "S"), (76, "S"), (77, "S"), (78, "S")]
BASE, BETA, DELTA = "cn0_physics_trend_dbhz", "beta_db", "delta_eval_db"
rows = pd.read_csv(SOURCE)
rows = rows.loc[rows.reference.eq("empirical_fit32")].copy()
rows["op"] = rows["op"].astype(str).str.removeprefix("OP").astype(int)
assert len(rows) == 15806
assert not rows.duplicated(["minute_utc", "op", "signal_name", "svid"]).any()
assert rows[[BASE, BETA, DELTA]].notna().all().all()
assert rows.trend_training_eligible.all()
assert rows.groupby("signal_name")[BETA].nunique().eq(1).all()
assert rows.groupby("mission_phase").size().to_dict() == {
    "C": 576, "T": 914, "L": 255, "S": 14061}


def summarize(part):
    if part.empty:
        return dict(n=0, baseline_mean_dbhz=None, beta_mean_db=None,
                    delta_mean_db=None, delta_rms_db=None)
    return dict(n=len(part), baseline_mean_dbhz=float(part[BASE].mean()),
                beta_mean_db=float(part[BETA].mean()),
                delta_mean_db=float(part[DELTA].mean()),
                delta_rms_db=float((part[DELTA] ** 2).mean() ** 0.5))


records = []
for level, op, phase, part in (
    [("operation", op, phase, rows.loc[rows.op.eq(op)]) for op, phase in OPS]
    + [("phase", "ALL", phase, rows.loc[rows.mission_phase.eq(phase)])
       for phase in "CTLS"]
    + [("overall", "ALL", "ALL", rows)]
):
    for signal in SIGNALS:
        records.append(dict(level=level, op=op, mission_phase=phase,
                            signal_name=signal,
                            **summarize(part.loc[part.signal_name.eq(signal)])))

# Independently reconcile with the original table's saved aggregate statistics.
check = pd.read_csv(CHECK)
check = check.loc[check.reference.eq("empirical_fit32")
                  & check["mask"].eq("primary_continuity")
                  & check.scope.eq("all_eligible")
                  & check.weighting.eq("sample_weighted")]
checked = []
for _, ref in check.iterrows():
    part = rows
    if ref.level in ("operation", "operation_signal"):
        part = part.loc[part.op.eq(int(str(ref.op).removeprefix("OP")))]
    elif ref.level in ("phase", "phase_signal"):
        part = part.loc[part.mission_phase.eq(ref.mission_phase)]
    elif ref.level != "overall":
        continue
    if ref.level.endswith("_signal"):
        part = part.loc[part.signal_name.eq(ref.signal_name)]
    stats = summarize(part)
    assert stats["n"] == int(ref.n), (ref.level, ref.op, ref.mission_phase, ref.signal_name, stats["n"], ref.n)
    if stats["n"] == 0:
        continue
    for ours, original in [("beta_mean_db", "beta_mean_db"),
                           ("delta_mean_db", "delta_eval_mean_db"),
                           ("delta_rms_db", "delta_eval_rms_db")]:
        assert math.isclose(stats[ours], float(ref[original]), abs_tol=1e-10)
    checked.append(str(ref.level))

result = pd.DataFrame(records)
result.to_csv(OUT / "signal_summary.csv", index=False, float_format="%.17g")
qa = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          filter="reference == empirical_fit32", n=len(rows),
          aggregation="Arithmetic means of stored dB/dB-Hz values; delta RMS = sqrt(mean(delta_eval_db**2))",
          baseline_column=BASE, beta_column=BETA, delta_column=DELTA,
          signal_order=SIGNALS, original_summary_groups_verified=len(checked),
          original_summary_levels=sorted(set(checked)),
          beta_by_signal=rows.groupby("signal_name")[BETA].first().to_dict(),
          n_by_signal=rows.groupby("signal_name").size().to_dict(),
          model_refitted=False)
(OUT / "aggregation_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")


def fmt(value, integer=False, signed=False):
    if value is None:
        return "---"
    if integer:
        return f"{value:.0f}"
    if round(value, 2) == 0:
        return "0.0"
    digits = 2 if 0 < abs(value) < 0.1 else 1
    return format(value, f"{'+' if signed else ''}.{digits}f")


for start in range(0, len(records), 4):
    group = records[start:start + 4]
    first = group[0]
    name = f"OP{first['op']}" if first["level"] == "operation" else first["level"]
    baseline = "/".join(fmt(r["baseline_mean_dbhz"], integer=True) for r in group)
    beta = "/".join(fmt(r["beta_mean_db"], integer=True) for r in group)
    delta = "/".join("---" if r["n"] == 0 else
                     f"{fmt(r['delta_mean_db'], signed=True)}({fmt(r['delta_rms_db'])})"
                     for r in group)
    print(f"{name} & {first['mission_phase']} & {baseline} & {beta} & {delta}")
print(json.dumps(qa, indent=2))
