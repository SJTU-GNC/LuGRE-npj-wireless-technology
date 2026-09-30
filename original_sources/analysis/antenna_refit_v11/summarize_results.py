"""Compact exports from the completed, prespecified reference experiments."""
from pathlib import Path
import sys
sys.path.insert(0,r"D:\月球导航\runtime_cache\python_deps")
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
m=pd.read_csv(OUT/"paired_metrics.csv")
p=pd.read_csv(OUT/"paired_predictions.csv")

def save(d,name):
    d.to_csv(OUT/name,index=False,encoding="utf-8-sig",float_format="%.15g")

q=m.loc[m.protocol.eq("original") & m.scope.eq("external_holdout") & m.population.eq("all_links") & m.level.eq("pooled")].copy()
save(q,"primary_holdout_summary.csv")
q=m.loc[m.protocol.eq("original") & m.scope.eq("all_eligible") & m.population.eq("foc") & m.level.eq("phase")].copy()
save(q,"foc_phase_allocation.csv")

# Retain all 20 original operations, including operations without eligible bins.
manifest=p.loc[p.protocol.eq("original") & p.scenario.eq("control")]
ops=manifest.groupby("op").agg(mission_phase=("mission_phase","first"),minute_bins=("op","size"),eligible_samples=("eligible","sum"),foc_minute_bins=("foc","sum")).reset_index()
ops=ops.assign(_order=ops.op.str.removeprefix("OP").astype(int)).sort_values("_order").drop(columns="_order")
rows=[]
for scenario in m.scenario.unique():
    q=m.loc[m.protocol.eq("original") & m.scenario.eq(scenario) & m.scope.eq("all_eligible") & m.population.eq("all_links") & m.level.eq("operation")].rename(columns={"group":"op"})
    out=ops.merge(q.drop(columns=["scenario"],errors="ignore"),on="op",how="left",validate="one_to_one")
    out["scenario"]=scenario
    out["protocol"],out["scope"],out["population"],out["level"]="original","all_eligible","all_links","operation"
    out["n"]=out.n.fillna(0).astype(int)
    out["result_status"]=np.where(out.n.gt(0),"descriptive_fixed_final_refit_including_fit_rows","no_eligible_samples")
    rows.append(out)
save(pd.concat(rows,ignore_index=True),"all20_operation_diagnostics.csv")

# Compare fitting protocols on exactly the same retained non-IOV evaluation rows.
rows=[]
for (protocol,scenario,split),g in p.loc[p.eligible & ~p.iov & p.evaluation_split.isin(["test","external_holdout"])].groupby(["protocol","scenario","evaluation_split"],sort=False):
    groups=[("all",g)]+[(op,h) for op,h in g.groupby("op")]
    for op,h in groups:
        rec=dict(protocol=protocol,scenario=scenario,evaluation_split=split,op=op,n=len(h))
        for label,col in [("baseline","baseline_dbhz"),("persistent","persistent_only_dbhz"),("final","final_dbhz")]:
            rec[label+"_rmse_dbhz"]=float(np.sqrt(np.mean((h[col]-h.observed_dbhz)**2)))
        rows.append(rec)
save(pd.DataFrame(rows),"same_non_iov_sample_protocol_comparison.csv")
print("Saved primary, phase, all-20-operation and same-sample IOV-control tables.")
