#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_shared_funding_consumer as consumer
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools.kraken_a1_full_streaming_reducer import _funding_context, _panel_for_boundaries

RESCORE=Path("results/rebaseline/phase_kraken_shared_funding_consumer_a1_tsmom_rescore_20260711_v1")
FUNDING=Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
PRIOR=Path("results/rebaseline/phase_kraken_tsmom_v6_survivor_forensic_decomposition_20260708_v1")
FULL=Path("results/rebaseline/phase_kraken_full_tsmom_v6_aggregate_20260707_v1")
OUTCOME=Path("results/rebaseline/phase_kraken_tsmom_outcome_grouped_aggregate_20260707_v1/cache/tsmom_interval_outcome.parquet")
ROOT=Path("results/rebaseline/phase_kraken_tsmom_funding_corrected_reopened_forensics_20260712_v1")
MODES=("exact_only_slice","central_imputed","conservative_imputed","severe_imputed")
BPS=(4,8,12)
PRESERVED={"tsmom_v6_079","tsmom_v6_059","tsmom_v6_041"}

def csv(path:Path,x:pd.DataFrame|list[dict[str,Any]]): runner.write_csv(path,x if isinstance(x,pd.DataFrame) else pd.DataFrame(x))

class UF:
 def __init__(self,x): self.p={i:i for i in x}
 def find(self,x):
  while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
  return x
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)

def pool()->pd.DataFrame:
 audit=pd.read_csv(RESCORE/"tsmom/reopened_candidate_audit.csv")
 x=audit[audit.newly_nonfutile_all_funding_and_slippage_gates.astype(bool)].copy()
 if len(x)!=28 or set(x.candidate_definition_id)&PRESERVED: raise RuntimeError("reopened pool contract failed")
 previous=pd.read_csv(FULL/"aggregate/tsmom_v6_definition_level_aggregate_summary.csv")
 cols=[c for c in ["candidate_definition_id","parameter_vector_hash","side","universe_policy","lookback_days","hold_interval","rebalance_interval","vol_target","parent_regime_gate","funding_gate","rank_top_n","rank_metric","continuation_filter"] if c in previous]
 return x.merge(previous[cols].drop_duplicates("candidate_definition_id"),on="candidate_definition_id",how="left",validate="one_to_one")

def rescore(events:pd.DataFrame)->pd.DataFrame:
 e=consumer.normalize_frozen_events(events,"tsmom"); b=consumer.build_event_boundary_rows(e)
 model=json.loads((FUNDING/"decision_summary.json").read_text()); ctx=_funding_context(FUNDING,sorted(e.symbol.astype(str).unique()),str(model["selected_model_hash"])); panel,_=_panel_for_boundaries(b,ctx)
 joined=consumer.join_boundaries_to_panel(b,panel); missing=int((joined._merge!="both").sum()); dup=int(joined.duplicated(["event_key","boundary_ts"]).sum())
 if missing or dup: raise RuntimeError(f"funding join failure {missing}/{dup}")
 return balanced.scenario_event_rows(consumer.aggregate_event_funding(e,joined),MODES,BPS)

def clusters(pooldf:pd.DataFrame,events:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
 ids=sorted(pooldf.candidate_definition_id.astype(str)); sets={i:set(events.loc[events.candidate_definition_id==i,"symbol"].astype(str)+"|"+pd.to_datetime(events.loc[events.candidate_definition_id==i,"decision_ts"],utc=True).astype(str)) for i in ids}
 streams={i:events[events.candidate_definition_id==i].assign(month=lambda z:pd.to_datetime(z.decision_ts,utc=True).dt.strftime("%Y-%m")).groupby(["symbol","month"]).scaled_gross_R.sum() for i in ids}
 fields=[c for c in ["side","universe_policy","lookback_days","hold_interval","rebalance_interval","vol_target","parent_regime_gate","funding_gate","rank_top_n","rank_metric","continuation_filter"] if c in pooldf]
 params=pooldf.set_index("candidate_definition_id"); uf=UF(ids); rows=[]
 for a,b in combinations(ids,2):
  inter=len(sets[a]&sets[b]); union=len(sets[a]|sets[b]); jac=inter/union if union else 1.; exact=sets[a]==sets[b]
  common=streams[a].index.intersection(streams[b].index); corr=float(streams[a].loc[common].corr(streams[b].loc[common])) if len(common)>=3 else np.nan
  sim=sum(str(params.loc[a,f])==str(params.loc[b,f]) for f in fields)/max(1,len(fields)); near=not exact and jac>=.8 and (not np.isfinite(corr) or corr>=.8)
  if exact or near:uf.union(a,b)
  if exact or near or jac>=.5 or sim>=.8: rows.append({"left_definition":a,"right_definition":b,"exact_selected_event_equality":exact,"event_set_jaccard":jac,"return_stream_correlation":corr,"parameter_similarity":sim,"near_duplicate":near})
 cluster={i:runner.stable_hash("tsmom_reopened_cluster",uf.find(i),n=16) for i in ids}; report=pd.DataFrame(rows)
 if not report.empty: report["left_cluster_id"]=report.left_definition.map(cluster);report["right_cluster_id"]=report.right_definition.map(cluster)
 reps=[]
 for cl,members in pd.Series(cluster).groupby(lambda i:cluster[i]):
  ordered=sorted(members.index); reps.extend({"cluster_id":cl,"candidate_definition_id":cid,"selection_role":"representative" if n==0 else "stability_neighbour","selected_for_forensics":n<2,"selection_uses_pnl":False} for n,cid in enumerate(ordered[:2]))
 return report,pd.DataFrame(reps)

def forensics(s:pd.DataFrame,reps:set[str])->dict[str,pd.DataFrame]:
 s=s[s.candidate_definition_id.isin(reps)].copy(); base=s[(s.funding_mode=="severe_imputed")&(s.slippage_round_trip_bps==12)].copy(); base["month"]=pd.to_datetime(base.entry_ts,utc=True).dt.strftime("%Y-%m");base["symbol_month"]=base.symbol.astype(str)+"|"+base.month
 funding=s.groupby(["candidate_definition_id","funding_mode","slippage_round_trip_bps"],dropna=False).agg(events=("event_key","nunique"),net_R=("scenario_scaled_net_R","sum"),mean_R=("scenario_scaled_net_R","mean"),median_R=("scenario_scaled_net_R","median"),exact_boundaries=("exact_boundary_rows","sum"),imputed_boundaries=("imputed_boundary_rows","sum")).reset_index()
 top=[]; los=[];lom=[];losm=[];period=[];exact=[]
 for cid,g in base.groupby("candidate_definition_id"):
  d=g.sort_values("scenario_scaled_net_R",ascending=False); total=d.scenario_scaled_net_R.sum(); n=max(1,math.ceil(len(d)*.01)); denom=max(float(d.scenario_scaled_net_R.abs().sum()),1e-12)
  top.append({"candidate_definition_id":cid,"events":len(d),"base_net_R":total,"net_without_top_1":d.iloc[1:].scenario_scaled_net_R.sum(),"net_without_top_3":d.iloc[3:].scenario_scaled_net_R.sum(),"net_after_top_1pct_trim":d.iloc[n:].scenario_scaled_net_R.sum(),"largest_event_R":d.iloc[0].scenario_scaled_net_R,"largest_event_abs_contribution_share":abs(d.iloc[0].scenario_scaled_net_R)/denom,"dominant_symbol_abs_share":g.groupby("symbol").scenario_scaled_net_R.sum().abs().max()/denom,"dominant_month_abs_share":g.groupby("month").scenario_scaled_net_R.sum().abs().max()/denom,"dominant_symbol_month_abs_share":g.groupby("symbol_month").scenario_scaled_net_R.sum().abs().max()/denom})
  for val,x in g.groupby("symbol"):los.append({"candidate_definition_id":cid,"excluded_symbol":val,"net_R_after_exclusion":total-x.scenario_scaled_net_R.sum()})
  for val,x in g.groupby("month"):lom.append({"candidate_definition_id":cid,"excluded_month":val,"net_R_after_exclusion":total-x.scenario_scaled_net_R.sum()})
  for val,x in g.groupby("symbol_month"):losm.append({"candidate_definition_id":cid,"excluded_symbol_month":val,"net_R_after_exclusion":total-x.scenario_scaled_net_R.sum()})
 for cid,g in s.groupby("candidate_definition_id"):
  entry=pd.to_datetime(g.entry_ts,utc=True); g=g.copy();g["period"]=np.select([entry.dt.year.eq(2024),entry<pd.Timestamp("2025-07-01",tz="UTC")],["2024","2025_h1"],default="2025_h2")
  for keys,x in g.groupby(["period","funding_mode","slippage_round_trip_bps"]):period.append({"candidate_definition_id":cid,"period":keys[0],"funding_mode":keys[1],"slippage_round_trip_bps":keys[2],"events":x.event_key.nunique(),"net_R":x.scenario_scaled_net_R.sum()})
  sev=g[(g.funding_mode=="severe_imputed")&(g.slippage_round_trip_bps==12)]
  for flag,x in sev.groupby("all_boundaries_exact"):exact.append({"candidate_definition_id":cid,"funding_coverage":"exact_or_zero_boundary" if flag else "imputed_boundary_present","events":x.event_key.nunique(),"net_R":x.scenario_scaled_net_R.sum(),"exact_boundaries":x.exact_boundary_rows.sum(),"imputed_boundaries":x.imputed_boundary_rows.sum()})
 return {"funding":funding,"top":pd.DataFrame(top),"los":pd.DataFrame(los),"lom":pd.DataFrame(lom),"losm":pd.DataFrame(losm),"period":pd.DataFrame(period),"exact":pd.DataFrame(exact)}

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--run-root",default=str(ROOT));args=ap.parse_args();root=Path(args.run_root)
 if root.exists() and any(root.iterdir()):raise RuntimeError("run root not fresh")
 p=pool();csv(root/"selection/reopened_definition_pool.csv",p)
 all_events=pd.read_parquet(OUTCOME); events=all_events[all_events.candidate_definition_id.isin(set(p.candidate_definition_id))].copy(); scen=rescore(events)
 cr,reps=clusters(p,events);csv(root/"selection/duplicate_cluster_report.csv",cr);csv(root/"selection/cluster_representatives.csv",reps)
 selected=set(reps.loc[reps.selected_for_forensics,"candidate_definition_id"]); f=forensics(scen,selected)
 for rel,key in [("forensics/funding_slippage_summary.csv","funding"),("forensics/top_event_dependency.csv","top"),("forensics/leave_one_symbol.csv","los"),("forensics/leave_one_month.csv","lom"),("forensics/leave_one_symbol_month.csv","losm"),("forensics/period_support.csv","period"),("forensics/exact_vs_imputed_support.csv","exact")]:csv(root/rel,f[key])
 top=f["top"].set_index("candidate_definition_id"); mins={"symbol":f["los"].groupby("candidate_definition_id").net_R_after_exclusion.min(),"month":f["lom"].groupby("candidate_definition_id").net_R_after_exclusion.min(),"symbol_month":f["losm"].groupby("candidate_definition_id").net_R_after_exclusion.min()}; period=f["period"][(f["period"].funding_mode=="severe_imputed")&(f["period"].slippage_round_trip_bps==12)].pivot_table(index="candidate_definition_id",columns="period",values="net_R",aggfunc="sum"); exact=f["exact"].pivot_table(index="candidate_definition_id",columns="funding_coverage",values="net_R",aggfunc="sum")
 decisions=[]
 for cid in sorted(p.candidate_definition_id):
  if cid not in selected:decision="preserve_for_later_redesign";reason="duplicate_cluster_nonrepresentative"
  else:
   t=top.loc[cid]; robust=t.net_without_top_3>0 and t.net_after_top_1pct_trim>0 and all(mins[k].get(cid,-np.inf)>0 for k in mins) and t.largest_event_abs_contribution_share<.2 and t.dominant_symbol_month_abs_share<.5
   periods=period.loc[cid] if cid in period.index else pd.Series(dtype=float); support=int((periods>0).sum())>=2 and periods.get("2025_h2",-np.inf)>0
   ex=exact.loc[cid] if cid in exact.index else pd.Series(dtype=float); funding_support=ex.get("exact_or_zero_boundary",-np.inf)>0 and ex.get("imputed_boundary_present",-np.inf)>0
   decision="advance_to_targeted_materialization_preflight" if robust and support and funding_support else ("preserve_for_later_redesign" if robust else "defer_current_translation");reason=f"winner_concentration={robust};period_support={support};exact_imputed_support={funding_support}"
  decisions.append({"candidate_definition_id":cid,"cluster_id":reps.loc[reps.candidate_definition_id==cid,"cluster_id"].iloc[0] if cid in set(reps.candidate_definition_id) else "","selected_representative_or_neighbour":cid in selected,"decision":decision,"reason":reason,"evidence_label":"train_only_funding_corrected_forensic_capped_not_validation"})
 prior=pd.read_csv(PRIOR/"decision/forensic_candidate_decision_table.csv")
 for cid in PRESERVED:
  row=prior[prior.candidate_definition_id==cid].iloc[0];decisions.append({"candidate_definition_id":cid,"cluster_id":"prior_forensic_preserved","selected_representative_or_neighbour":False,"decision":"diagnostic_only" if "reject" in row.forensic_decision else "preserve_for_later_redesign","reason":"prior_forensic_decision_preserved_no_reopening","evidence_label":row.evidence_level})
 d=pd.DataFrame(decisions);csv(root/"decision/candidate_decision_table.csv",d);csv(root/"candidate_library/tsmom_candidate_library_update.csv",d)
 exact_clusters=int(cr.exact_selected_event_equality.sum()) if len(cr) else 0;near=int(cr.near_duplicate.sum()) if len(cr) else 0;advanced=d[d.decision=="advance_to_targeted_materialization_preflight"].candidate_definition_id.tolist();survivors=[cid for cid in selected if cid in top.index and top.loc[cid].net_without_top_3>0 and top.loc[cid].net_after_top_1pct_trim>0 and all(mins[k].get(cid,-np.inf)>0 for k in mins)]
 summary={"run_root":str(root),"status":"complete","signals_regenerated":False,"reopened_definitions_reviewed":28,"exact_duplicate_pairs":exact_clusters,"near_duplicate_pairs":near,"cluster_count":int(reps.cluster_id.nunique()),"representatives_and_neighbours_evaluated":len(selected),"winner_and_concentration_survivors":sorted(survivors),"advanced_candidates":advanced,"preserved_candidates":d[d.decision=="preserve_for_later_redesign"].candidate_definition_id.tolist(),"deferred_candidates":d[d.decision=="defer_current_translation"].candidate_definition_id.tolist(),"diagnostic_candidates":d[d.decision=="diagnostic_only"].candidate_definition_id.tolist(),"targeted_materialization_allowed":bool(advanced),"validation_launched":False,"final_holdout_touched":False,"compact_bundle_path":str(root/"compact_review_bundle")};runner.write_json(root/"decision_summary.json",summary)
 req=["selection/reopened_definition_pool.csv","selection/duplicate_cluster_report.csv","selection/cluster_representatives.csv","forensics/funding_slippage_summary.csv","forensics/top_event_dependency.csv","forensics/leave_one_symbol.csv","forensics/leave_one_month.csv","forensics/leave_one_symbol_month.csv","forensics/period_support.csv","forensics/exact_vs_imputed_support.csv","decision/candidate_decision_table.csv","candidate_library/tsmom_candidate_library_update.csv","decision_summary.json"]
 rows=[]
 for rel in req:
  src=root/rel;dst=root/"compact_review_bundle"/rel.replace("/","__");dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);rows.append({"source":rel,"bundle_path":str(dst.relative_to(root)),"sha256":runner.sha256_file(dst)})
 csv(root/"compact_review_bundle/compact_bundle_manifest.csv",rows);return 0

if __name__=="__main__":raise SystemExit(main())
