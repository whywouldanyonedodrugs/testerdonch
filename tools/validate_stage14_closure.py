#!/usr/bin/env python3
"""Fail-closed validation for the Stage 14 closure and future packet."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"docs/agent/task_archive/20260719_donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1"
L=Path("/opt/parquet/kraken_derivatives/analytics/stage14_phase1_v1")


def h(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value: bool,message: str) -> None:
    if not value: raise AssertionError(message)


def main() -> None:
    results=[]
    measurement=json.loads((L/"PHASE1_MEASUREMENT.json").read_text()); require(measurement["economic_outputs_computed"] is False,"economic output flag")
    require(measurement["protected_rows_opened"]==0 and measurement["Capitalcom_payload_opened"]==0,"payload firewall flags")
    results.append("outcome and protected/Capital.com firewall flags pass")
    forbidden=("forward","future_return","entry_price","exit_price","pnl","cashflow","target_return")
    requested=measurement["outcome_read_spy"]["allowed_feature_columns"]+measurement["outcome_read_spy"]["allowed_parent_columns"]
    require(not any(token in col.lower() for token in forbidden for col in requested),"forbidden reader field")
    require(measurement["outcome_read_spy"]["feature_request_count"]==187,"feature spy coverage")
    require(all(x["pre_read_max_timestamp_check"]=="pass" for x in measurement["outcome_read_spy"]["requests"]),"pre-read timestamp spy")
    results.append("read-spy allow-lists contain no forward or execution outcome field")
    retention=pd.read_csv(A/"KDA02B_RETENTION_BOUNDARY.csv",parse_dates=["first_raw_OI_timestamp","first_complete_causal_lookback","first_admissible_onset_timestamp","invalid_boundary_start","invalid_boundary_end"])
    require(len(retention)==187 and (retention.first_admissible_onset_timestamp.dropna()>retention.first_complete_causal_lookback.dropna()).all(),"retention onset boundary")
    require({"gap_count","OI_missing_bars_inside_retention","last_raw_OI_timestamp"}.issubset(retention.columns) and (retention.OI_missing_bars_inside_retention>=retention.gap_count).all(),"OI retention gap semantics")
    results.append("187-symbol retention ledger and strict post-lookback onset boundary pass")
    for path in (L/"kda02b_onsets").glob("*.parquet"):
        if pq.ParquetFile(path).metadata.num_rows:
            d=pd.read_parquet(path,columns=["state_ts","decision_ts"]); decision=pd.to_datetime(d.decision_ts,utc=True).dt.as_unit("ns"); state=pd.to_datetime(d.state_ts,utc=True).dt.as_unit("ns"); require(((decision.astype("int64")-state.astype("int64"))==300_000_000_000).all(),"onset timestamp availability")
            require((pd.to_datetime(d.decision_ts,utc=True)<pd.Timestamp("2026-01-01",tz="UTC")).all(),"protected onset")
    results.append("all KDA02B decision timestamps are source close plus five minutes and pre-protected")
    breadth=pd.read_parquet(L/"KDA02C_PIT_BREADTH_PANEL.parquet")
    require({"decision_ts","analytics_manifest_sha256","authorized_cohort_sha256","feature_contract_sha256"}.issubset(breadth.columns),"breadth provenance fields")
    require(((pd.to_datetime(breadth.decision_ts,utc=True).astype("int64")-pd.to_datetime(breadth.timestamp_utc,utc=True).astype("int64"))==300_000_000_000).all(),"breadth decision timestamp")
    require((breadth.known==breadth.eligible+breadth.excluded).all(),"PIT denominator reconciliation")
    for prefix in ("primary","robust"):
        require((breadth[f"{prefix}_state"]<=breadth.eligible).all(),"breadth state exceeds PIT denominator")
        require((breadth[f"{prefix}_negative"]+breadth[f"{prefix}_positive"]<=breadth[f"{prefix}_state"]).all(),"breadth direction split")
    require(pd.to_datetime(breadth.timestamp_utc,utc=True).max()<pd.Timestamp("2026-01-01",tz="UTC"),"protected breadth")
    results.append("PIT breadth denominator, direction, invariance, and protected boundary pass")
    overlap=pd.read_csv(A/"KDX01_COMPONENT_OVERLAP_MATRIX.csv"); require(len(overlap)==64,"component overlap matrix incomplete")
    incidence=pd.read_csv(A/"KDX01_INCIDENCE_SUMMARY.csv"); require(set(incidence.grammar_cell.unique())==set(measurement["KDX01"]["grammar_onsets"]),"KDX grammar mismatch")
    require((incidence.episodes==incidence.onsets).all() and incidence.duration_minutes_median.dropna().ge(5).all(),"KDX episode durations")
    availability=pd.read_csv(A/"KDX01_COMPONENT_AVAILABILITY_AND_MISSINGNESS.csv")
    require(len(pd.read_csv(A/"KDX01_RAW_UNIT_DISTRIBUTIONS.csv"))>0 and len(availability)==561,"KDX raw/missingness")
    availability_columns=[x for x in availability if x.endswith("_available")]
    require(all((availability[x]<=availability.rows).all() for x in availability_columns),"KDX evaluability bounds")
    breadth_lookup=pd.Series(breadth.eligible.gt(0).to_numpy(),index=pd.to_datetime(breadth.timestamp_utc,utc=True).astype("int64"))
    expected={}
    for path in (L/"primitive_base").glob("*.parquet"):
        symbol=path.stem; ts=pd.to_datetime(pd.read_parquet(path,columns=["timestamp_utc"]).timestamp_utc,utc=True).dt.as_unit("ns")
        values=breadth_lookup.reindex(ts.astype("int64")).fillna(False)
        for year in (2023,2024,2025): expected[(symbol,year)]=int(values[ts.dt.year.eq(year).to_numpy()].sum())
    actual={(row.symbol,int(row.year)):int(row.breadth_available) for row in availability.itertuples()}
    require(actual==expected,"KDX breadth availability is not PIT-denominator based")
    require(measurement["KDX01"]["price_oi_onsets"]<measurement["KDX01"]["price_onsets"],"KDX not distinct from price")
    results.append("KDX fixed grammar, full overlap matrix, and price-only distinction pass")
    universe=pd.read_csv(A/"KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.csv"); require(len(universe)==479,"inventory rows")
    require(int(universe.included.sum())==460 and int(universe.final_campaign_eligible.sum())==187 and int(universe.causal_lookback_eligibility.sum())==117,"universe counts")
    require(universe.PF_symbol.nunique()==479 and universe.campaign_exclusion_reason.notna().all(),"universe identities/reasons")
    require("None can be admitted" in (A/"KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.md").read_text(),"out-of-cohort admission statement")
    results.append("479 inventory / 460 K0 / 187 campaign / 117 KDA02B-causal universe reconciles with one reason each")
    funding=pd.read_csv("/opt/testerdonch/results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1/funding/shared_funding_panel_manifest.csv")
    require(int(funding.row_count.sum())==247522 and int(funding.exact_rows.sum())==54963 and int(funding.imputed_rows.sum())==192559,"funding authority totals")
    contract=(A/"CAMPAIGN_FUNDING_AND_COST_CONTRACT.md").read_text(); require(all(x in contract for x in ("14 bps","32 bps","exact","mixed","imputed","zero_boundary","conservative","2023 boundary","frozen selected model")),"funding contract")
    results.append("authoritative funding totals and partitioned 14/32 bps contract pass")
    registry=json.loads((A/"SEARCH_SPACE_REGISTRY.json").read_text()); cells=sum(len(x["registered_cells"]) for x in registry["search_spaces"])
    require(cells==registry["maximum_total_cells"]==228 and [x["maximum_cells"] for x in registry["search_spaces"]]==[96,48,84],"cell budgets")
    require(all(len(x["registered_cell_ids"])==len(set(x["registered_cell_ids"])) for x in registry["search_spaces"]),"duplicate cell")
    results.append("all 228 bounded search cells register exactly as 96/48/84")
    manifest=json.loads((A/"CAMPAIGN_MANIFEST.json").read_text()); packet=json.loads((A/"FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json").read_text())
    require(manifest["economic_run_authorized_by_manifest"] is False and packet["economic_run_authorized"] is False,"self authorization")
    require(manifest["C17_excluded"] and packet["C17_excluded"] and len(manifest["ready_hypotheses"])==3,"lane scope")
    require(len(manifest["fold_schedule"])==27 and all(not x["independent_validation_claim"] for x in manifest["fold_schedule"]),"fold exposure")
    require(set(manifest["stop_conditions"])=={"family_only","global"},"stop isolation")
    require(manifest["resource_limits"]["derivation"]["registered_cells"]==228 and manifest["resource_limits"]["derivation"]["projected_wall_seconds"]<manifest["resource_limits"]["wall_seconds"],"benchmark budget derivation")
    require(manifest["repository_and_data_hashes"]["phase1_measurement_sha256"]==h(L/"PHASE1_MEASUREMENT.json"),"measurement hash binding")
    results.append("packet is non-authorizing, C17-excluding, 27-fold, exposure-labelled, and stop-isolated")
    old_state="docs/agent/research_campaigns/kraken_research_campaign_001_readiness/CAMPAIGN_STATE.json"
    base=subprocess.check_output(["git","show",f"e14bbd0d26c14e48a347481f170fcfe8851df625:{old_state}"],cwd=ROOT)
    require(hashlib.sha256(base).hexdigest()==h(ROOT/old_state),"historical campaign state changed")
    results.append("Stage 13 historical campaign state and terminal decisions remain byte-identical")
    required=[x.strip() for x in """KDA02B_PHASE1_MEASUREMENT.md KDA02B_ONSET_AND_EPISODE_SUMMARY.csv KDA02B_RETENTION_BOUNDARY.csv KDA02C_PIT_BREADTH_CONTRACT.md KDA02C_BREADTH_SUMMARY.csv KDX01_PRIMITIVE_STATE_CONTRACT.md KDX01_COMPONENT_OVERLAP_MATRIX.csv KDX01_INCIDENCE_SUMMARY.csv KDX01_RAW_UNIT_DISTRIBUTIONS.csv KDX01_COMPONENT_AVAILABILITY_AND_MISSINGNESS.csv KDX01_EPISODE_DURATION_SUMMARY.csv KDX01_ECONOMIC_ADDRESS_ANALYSIS.csv KDX01_PARENT_IDENTITY_OVERLAP.csv KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.csv KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.md CAMPAIGN_FUNDING_AND_COST_CONTRACT.md PHASE1_ADMISSION_DECISIONS.json SEARCH_SPACE_REGISTRY.json RESOURCE_PROJECTION.json FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md""".split()]
    require(all((A/x).is_file() for x in required),"required artifact missing")
    results.append("all pre-review required Phase-1 and campaign artifacts exist")
    packet_files=[A/x for x in ("CAMPAIGN_MANIFEST.json","SEARCH_SPACE_REGISTRY.json","FOLD_AND_EXPOSURE_MAP.json","RESOURCE_PROJECTION.json","FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json","FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md")]
    implementation_commit=manifest["repository_and_data_hashes"]["repository_implementation_commit"]
    before={str(x):h(x) for x in packet_files}; replay_env=os.environ.copy(); replay_env["PYTHONPATH"]="tools"; subprocess.check_call([str(Path("/opt/stage14-phase1-venv/bin/python")),str(ROOT/"tools/build_stage14_campaign_packet.py"),"--implementation-commit",implementation_commit],cwd=ROOT,env=replay_env); after={str(x):h(x) for x in packet_files}
    require(before==after,"deterministic packet replay")
    results.append("deterministic packet replay is byte-identical")
    (A/"VALIDATION.md").write_text("# Validation\n\nStatus: `pass`.\n\n"+"\n".join(f"- PASS: {x}" for x in results)+"\n\nNo economic output was computed; protected and Capital.com rows opened: zero.\n",encoding="utf-8")
    print(json.dumps({"status":"pass","checks":len(results),"campaign_manifest_sha256":h(A/"CAMPAIGN_MANIFEST.json"),"approval_packet_sha256":h(A/"FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json")},sort_keys=True))


if __name__=="__main__": main()
