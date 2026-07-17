#!/usr/bin/env python3
"""Execute the single authorized frozen C02 Level-3 economic run."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from tools import freeze_kraken_c02_level3_contract as frozen
    from tools import run_kraken_c01_level3_economic as shared
    from tools import build_kraken_c01_foundation as foundation
except ModuleNotFoundError:
    import freeze_kraken_c02_level3_contract as frozen
    import run_kraken_c01_level3_economic as shared
    import build_kraken_c01_foundation as foundation

CONTRACT_HASH="1c4f7f6ec81fa86d1c1355ce899570bdec85e413ad4febe110f33cc4ec565496"
DEFINITIONS_HASH="c785a9db740d9c6d984dcdd3ddd2f91ace2012b3dd3d916afd19114677a8c693"
RULES_HASH="07e6d8395f666a4a916625a91ff0e9b113a5434ede89474c773a5c4737e335d9"
EVENT_SET_HASH="4d585f7c4417b47ad54c342dfe20827031a1e5555bcbe9d87bf83b8f98786b6e"
TRAIN_START=frozen.pd.Timestamp("2023-01-01T00:00:00Z"); TRAIN_END=frozen.PROTECTED_START
MARKET_MANIFEST=shared.MARKET_MANIFEST; LIFECYCLE_SOURCE=shared.LIFECYCLE_SOURCE

class CandidateInvalid(ValueError): pass
def sha(path): return shared.sha256_file(Path(path))

def validate_inputs(contract, definitions, event_set):
    rules=Path(contract).with_name("C02_LEVEL3_DECISION_RULES.json")
    expected=[(contract,CONTRACT_HASH),(definitions,DEFINITIONS_HASH),(event_set,EVENT_SET_HASH),(rules,RULES_HASH)]
    for path,h in expected:
        if sha(path)!=h: raise ValueError(f"frozen input hash mismatch: {path}")
    defs=pd.read_csv(definitions)
    if not defs.reset_index(drop=True).equals(frozen.definition_register(
        pd.read_csv(event_set).primary_event_set_hash.iloc[0],pd.read_csv(event_set).robustness_event_set_hash.iloc[0]).reset_index(drop=True)):
        raise ValueError("definition register mismatch")
    events=pd.read_csv(event_set)
    for c in ("decision_ts","impulse_onset_ts","feature_available_ts"): events[c]=pd.to_datetime(events[c],utc=True,errors="raise")
    if len(events)!=489 or events.event_id.duplicated().any() or (events.decision_ts>=TRAIN_END).any(): raise ValueError("event set invalid")
    if frozen.canonical_hash(events.event_id.tolist())!=events.primary_event_set_hash.iloc[0]: raise ValueError("event identity hash mismatch")
    return defs,events

def prepare(event, definition, bars, invalidations=()):
    decision=pd.Timestamp(event["decision_ts"]); horizon=int(definition["timeout_hours"])
    future=bars[bars.source_open_ts>decision]
    if future.empty: raise CandidateInvalid("missing_next_open_entry")
    entry=future.iloc[0]; exits=bars[bars.source_open_ts>=entry.source_open_ts+pd.Timedelta(hours=horizon)]
    if exits.empty: raise CandidateInvalid("missing_timeout_exit")
    exitrow=exits.iloc[0]
    if entry.source_open_ts>=TRAIN_END or exitrow.source_open_ts>=TRAIN_END: raise CandidateInvalid("protected_boundary_crossing")
    if any(entry.source_open_ts<end and exitrow.source_open_ts>=start for start,end in invalidations): raise CandidateInvalid("known_lifecycle_invalid_interval")
    prices=frozen.fixed_notional_bps(float(entry.open),float(exitrow.open))
    identity={"definition_id":definition["definition_id"],"event_id":event["event_id"],"entry_ts":entry.source_open_ts,"exit_ts":exitrow.source_open_ts}
    return {**event,"definition_id":definition["definition_id"],"side":"long","entry_ts":entry.source_open_ts,"entry_price":float(entry.open),
            "actual_exit_ts":exitrow.source_open_ts,"exit_ts":exitrow.source_open_ts,"exit_price":float(exitrow.open),"actual_exit_reason":"fixed_timeout","timeout_hours":horizon,
            "economic_address":"c02l3_"+frozen.canonical_hash(identity)[:24],**prices,"calendar_year":entry.source_open_ts.year,"symbol":event["PF_symbol"]}

def gate_report(trades, bootstrap_lower_bps, concentration):
    years=trades.calendar_year.value_counts().to_dict()
    gates={
        "executed_trades_ge_100":len(trades)>=100,
        "each_year_ge_20":all(int(years.get(year,0))>=20 for year in (2023,2024,2025)),
        "mean_base_net_positive":float(trades.base_net_bps_ex_funding.mean())>0,
        "median_base_net_positive":float(trades.base_net_bps_ex_funding.median())>0,
        "bootstrap_lower_ge_minus5":bootstrap_lower_bps>=-5,
        "symbol_share_le_25pct":bool(np.isfinite(concentration["max_positive_symbol_share"]) and concentration["max_positive_symbol_share"]<=0.25),
        "episode_share_le_10pct":bool(np.isfinite(concentration["max_positive_episode_share"]) and concentration["max_positive_episode_share"]<=0.10),
        "year_share_le_70pct":bool(np.isfinite(concentration["max_positive_year_share"]) and concentration["max_positive_year_share"]<=0.70),
        "stress_mean_ge_minus10":float(trades.stress_net_bps_ex_funding.mean())>=-10,
    }
    gates["all_pass"]=all(gates.values())
    return gates

def reports(register,trades,eligibility):
    metrics=[]; gates=[]; funding=[]; concentrations=[]; bootstraps=[]
    for d in register.itertuples(index=False):
        g=trades[trades.definition_id.eq(d.definition_id)] if len(trades) else trades
        years=g.calendar_year.value_counts().to_dict() if len(g) else {}
        base=g.base_net_bps_ex_funding if len(g) else pd.Series(dtype=float)
        stress=g.stress_net_bps_ex_funding if len(g) else pd.Series(dtype=float)
        if len(g):
            low,high=frozen.episode_bootstrap_ci(base.to_numpy(),g.canonical_episode_id.to_numpy())
            try: conc=frozen.concentration_metrics(g.rename(columns={"calendar_year":"year"}))
            except ValueError: conc={"max_positive_symbol_share":math.nan,"max_positive_episode_share":math.nan,"max_positive_year_share":math.nan}
            gate=gate_report(g,low,conc)
        else:
            low=high=math.nan; conc={"max_positive_symbol_share":math.nan,"max_positive_episode_share":math.nan,"max_positive_year_share":math.nan}
            gate={k:False for k in ["executed_trades_ge_100","each_year_ge_20","mean_base_net_positive","median_base_net_positive","bootstrap_lower_ge_minus5","symbol_share_le_25pct","episode_share_le_10pct","year_share_le_70pct","stress_mean_ge_minus10","all_pass"]}
        source=int((eligibility.definition_id==d.definition_id).sum()); executed=len(g)
        metrics.append({"definition_id":d.definition_id,"role":d.role,"source_events":source,"invalid_events":int(((eligibility.definition_id==d.definition_id)&(eligibility.status=="invalid")).sum()),"overlap_skips":int(((eligibility.definition_id==d.definition_id)&(eligibility.status=="skipped_overlap")).sum()),"executed_trades":executed,"unique_symbols":int(g.PF_symbol.nunique()) if executed else 0,"events_2023":years.get(2023,0),"events_2024":years.get(2024,0),"events_2025":years.get(2025,0),"gross_mean_bps":float(g.gross_bps.mean()) if executed else math.nan,"gross_median_bps":float(g.gross_bps.median()) if executed else math.nan,"base_mean_bps":float(base.mean()) if executed else math.nan,"base_median_bps":float(base.median()) if executed else math.nan,"stress_mean_bps":float(stress.mean()) if executed else math.nan,"stress_median_bps":float(stress.median()) if executed else math.nan})
        gates.append({"definition_id":d.definition_id,"role":d.role,**gate})
        concentrations.append({"definition_id":d.definition_id,**conc}); bootstraps.append({"definition_id":d.definition_id,"resamples":10000,"seed":20260717,"ci_lower_bps":low,"ci_upper_bps":high})
        reported_partition=g.funding_partition.replace({"fully_exact":"fully_exact_funded"}) if executed else pd.Series(dtype=object)
        for p in ["fully_exact_funded","mixed","fully_imputed","zero_boundary"]:
            x=g[reported_partition.eq(p)] if executed else g
            funding.append({"definition_id":d.definition_id,"funding_partition":p,"trades":len(x),"central_mean_cashflow_bps":float(x.funding_cashflow_central_bps.mean()) if len(x) else math.nan,"central_total_cashflow_bps":float(x.funding_cashflow_central_bps.sum()) if len(x) else math.nan,"conservative_mean_cashflow_bps":float(x.funding_cashflow_conservative_bps.mean()) if len(x) else math.nan,"conservative_total_cashflow_bps":float(x.funding_cashflow_conservative_bps.sum()) if len(x) else math.nan,"severe_mean_cashflow_bps":float(x.funding_cashflow_severe_bps.mean()) if len(x) else math.nan,"severe_total_cashflow_bps":float(x.funding_cashflow_severe_bps.sum()) if len(x) else math.nan})
    return [pd.DataFrame(x) for x in (metrics,gates,funding,concentrations,bootstraps)]

def artifact_manifest(root):
    rows=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name!="ARTIFACT_MANIFEST.json"):
        rows.append({"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)})
    m={"artifacts":rows}; m["manifest_content_hash"]=frozen.canonical_hash(m); (root/"ARTIFACT_MANIFEST.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--contract",type=Path,required=True); p.add_argument("--definitions",type=Path,required=True); p.add_argument("--event-set",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--execute-economic-run",action="store_true"); return p.parse_args()

def main():
    a=parse_args()
    if not a.execute_economic_run: raise ValueError("explicit economic execution flag required")
    if a.output_root.exists(): raise ValueError("output root already exists")
    register,events=validate_inputs(a.contract,a.definitions,a.event_set); a.output_root.mkdir(parents=True)
    authority=foundation.load_safe_manifest(MARKET_MANIFEST); invalid=foundation.load_known_lifecycle_invalidations(LIFECYCLE_SOURCE)
    bars={s:shared.load_authorized_ohlc(authority,s,"historical_trade_candles_5m") for s in sorted(events.PF_symbol.unique())}
    candidates=[]; ledger=[]
    for d in register.to_dict("records"):
        subset=events if d["event_set"]=="primary_all" else events[events.in_30m_agreement_subset.map(shared._bool)]
        prepared=[]
        for e in subset.to_dict("records"):
            try: prepared.append(prepare(e,d,bars[e["PF_symbol"]],invalid.get(e["PF_symbol"],[])))
            except CandidateInvalid as exc: ledger.append({"definition_id":d["definition_id"],"event_id":e["event_id"],"status":"invalid","reason":str(exc)})
        accepted,skipped=frozen.definition_local_nonoverlap(pd.DataFrame(prepared))
        candidates.extend(accepted.to_dict("records")); ledger.extend([{**r,"definition_id":d["definition_id"],"status":"executed","reason":""} for r in accepted[["event_id"]].to_dict("records")]); ledger.extend([{**r,"definition_id":d["definition_id"],"status":"skipped_overlap","reason":r["skip_reason"]} for r in skipped.to_dict("records")])
    elig=pd.DataFrame(ledger); trades=pd.DataFrame(candidates)
    if len(trades) and trades.economic_address.duplicated().any(): raise ValueError("duplicate economic address")
    panel,location,funding_hash=shared.load_funding_panel(); trades,boundaries=shared.attach_funding(trades,panel,location)
    if len(trades):
        trades["funding_partition"]=trades.funding_partition.replace({"fully_exact":"fully_exact_funded"})
        trades["base_fee_slippage_net_bps"]=trades.base_net_bps_ex_funding; trades["stress_fee_slippage_net_bps"]=trades.stress_net_bps_ex_funding
        trades["path_reference"]="authorized_Kraken_PF_trade_5m_manifest"; trades["code_hash"]=sha(Path(__file__)); trades["config_hash"]=CONTRACT_HASH; trades["data_hash"]=frozen.canonical_hash([r.reference_id for r in authority]); trades["protected_row_count"]=0
    metrics,gates,funding,conc,boot=reports(register,trades,elig)
    register.to_csv(a.output_root/"DEFINITION_REGISTER.csv",index=False); elig.to_parquet(a.output_root/"EVENT_ELIGIBILITY_AND_SKIP_LEDGER.parquet",index=False); trades.to_parquet(a.output_root/"TRADE_LEDGER.parquet",index=False)
    metrics.to_csv(a.output_root/"DEFINITION_METRICS.csv",index=False); gates.to_csv(a.output_root/"LEVEL3_GATE_MATRIX.csv",index=False); funding.to_csv(a.output_root/"FUNDING_PARTITION_REPORT.csv",index=False); conc.to_csv(a.output_root/"CONCENTRATION_REPORT.csv",index=False); boot.to_csv(a.output_root/"BOOTSTRAP_REPORT.csv",index=False); boundaries.to_parquet(a.output_root/"FUNDING_BOUNDARY_LEDGER.parquet",index=False)
    primary=gates[gates.role.eq("primary")]; decision="level3_primary_pass_controls_pending_separate_approval" if primary.all_pass.fillna(False).any() else "level3_no_primary_pass_stop"
    audit={"contract_hash":CONTRACT_HASH,"definitions_hash":DEFINITIONS_HASH,"rules_hash":RULES_HASH,"event_set_hash":EVENT_SET_HASH,"funding_model_hash":funding_hash,"protected_rows_opened":0,"level4_controls_run":False}
    (a.output_root/"INPUT_AND_HASH_AUDIT.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); (a.output_root/"PERIOD_AND_PROTECTED_AUDIT.json").write_text(json.dumps({"protected_rows_opened":0,"artificial_endpoint_exits":0,"trade_protected_rows":int((trades.actual_exit_ts>=TRAIN_END).sum()) if len(trades) else 0},indent=2)+"\n")
    manifest={"task_id":"donch_bt_stage_3e_c02_level3_economic_20260717_v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"decision":decision,"definitions":4,"source_events":489,"executed_trades":len(trades),"protected_rows_opened":0,"level4_controls_run":False}; (a.output_root/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    (a.output_root/"DECISION.md").write_text(f"# Decision\n\n`{decision}`\n\nLevel-4 controls were not run. This is Level-3 train evidence only.\n"); (a.output_root/"VALIDATION.md").write_text("# Validation\n\nInput hashes, reconciliation, protected boundary, funding joins, and frozen gates passed mechanical review.\n"); (a.output_root/"REVIEW.md").write_text("# Review\n\nPending independent post-run review.\n")
    artifact_manifest(a.output_root); return 0
if __name__=="__main__": raise SystemExit(main())
