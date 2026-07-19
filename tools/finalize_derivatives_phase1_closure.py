#!/usr/bin/env python3
"""Resume Stage 14 after completed tapes; performs no source feature reread."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from build_derivatives_phase1_closure import (
    ARCHIVE, FEATURE_MANIFEST, FUNDING_MANIFEST, INVENTORY, FIXED_AT,
    load_parents, q, write_measurement_docs,
)
from qlmg_derivatives_phase1 import (
    FEATURE_COLUMNS, GRAMMAR_LADDERS, OutcomeReadSpy, PARENT_EVENT_COLUMNS,
    completed_purge_states, oi_retention_gap_counts, onset_mask, reconcile_universe, stable_hash,
    strict_base_valid, validate_grammar,
)


def finalize(output: Path) -> None:
    started=time.monotonic()
    manifest=json.loads(FEATURE_MANIFEST.read_text()); parts=sorted(manifest["partitions"],key=lambda x:x["symbol"])
    symbols={x["symbol"] for x in parts}
    for folder in ("kda02b_onsets","kda02b_episodes","primitive_base"):
        if len(list((output/folder).glob("*.parquet"))) != 187: raise ValueError(f"incomplete local tape: {folder}")
    breadth=pd.read_parquet(output/"KDA02C_PIT_BREADTH_PANEL.parquet"); grid=pd.DatetimeIndex(pd.to_datetime(breadth.timestamp_utc,utc=True))
    breadth["decision_ts"]=pd.to_datetime(breadth.timestamp_utc,utc=True).dt.as_unit("ns")+pd.Timedelta(minutes=5)
    breadth["analytics_manifest_sha256"]="f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d"
    breadth["authorized_cohort_sha256"]="5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636"
    breadth["feature_contract_sha256"]="4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4"
    origin=grid[0].value; step=300_000_000_000
    spy=OutcomeReadSpy(); parent_keys=load_parents(spy)
    comps=("trade_downside","mark_downside","structural_rejection","oi","liquidation","basis_level","basis_change","breadth")
    component_counts={(a,b):0 for a in comps for b in comps}; incidence=[]
    state_totals={"+".join(x):0 for x in GRAMMAR_LADDERS}; onset_totals={"+".join(x):0 for x in GRAMMAR_LADDERS}
    cluster_hours={"+".join(x):set() for x in GRAMMAR_LADDERS}; kdx_trade_samples=[]; kdx_mark_samples=[]
    purge_trade_samples=[]; purge_mark_samples=[]; purge_cluster_hours=set()
    raw_distribution_rows=[]; availability_rows=[]; coverage_by_symbol={}; oi_gap_records=[]
    address_rows=[]; parent_overlap_rows=[]
    breadth_flag=breadth.primary_cohort_share.fillna(0).gt(0).to_numpy()
    for part in parts:
        source=spy.read(Path(part["path"]),FEATURE_COLUMNS,kind="feature").sort_values("timestamp_utc",kind="mergesort").reset_index(drop=True)
        source_ts=pd.to_datetime(source.timestamp_utc,utc=True).dt.as_unit("ns"); source_valid=strict_base_valid(source)
        primary,_,_=completed_purge_states(source); purge_onset=onset_mask(primary,source_valid,source_ts)
        purge_trade_samples.append(source.loc[purge_onset,"trade_return_1h"].abs()*10000); purge_mark_samples.append(source.loc[purge_onset,"mark_return_1h"].abs()*10000)
        purge_cluster_hours.update(source_ts[purge_onset].dt.floor("h").astype(str))
        coverage_by_symbol[part["symbol"]]={"OI_coverage":bool(source.oi_close_base_units.notna().any()),"basis_coverage":bool(source.basis_decimal.notna().any()),"liquidation_coverage":bool(source.liquidation_base_units_1h.notna().any())}
        oi_ts=source_ts[source.oi_close_base_units.notna()]; gap_count,missing_bars=oi_retention_gap_counts(source_ts,source.oi_close_base_units)
        oi_gap_records.append({"symbol":part["symbol"],"gap_count":gap_count,"OI_missing_bars_inside_retention":missing_bars,"last_raw_OI_timestamp":oi_ts.max() if len(oi_ts) else pd.NaT})
        p=pd.read_parquet(output/"primitive_base"/f"{part['symbol']}.parquet"); ts=pd.to_datetime(p.timestamp_utc,utc=True)
        require_alignment=pd.to_datetime(source.timestamp_utc,utc=True).dt.as_unit("ns").astype("int64").equals(ts.dt.as_unit("ns").astype("int64"))
        if not require_alignment: raise ValueError(f"source/primitive timestamp mismatch: {part['symbol']}")
        p["trade_downside"]=source.trade_return_1h.lt(0); p["mark_downside"]=source.mark_return_1h.lt(0); p["structural_rejection"]=source.trade_log_return_5m.lt(0)
        p["basis_level"]=source.basis_level_normalization_valid.fillna(False)&source.basis_level_robust_z.abs().ge(2)
        pos=((ts.dt.as_unit("ns").astype("int64").to_numpy()-origin)//step).astype(int); p["breadth"]=breadth_flag[pos]; valid=p.eligible.astype(bool)
        address_memberships=pd.Series(0,index=p.index,dtype="int16")
        raw_fields=("trade_abs_bps","mark_abs_bps","oi_log_change_1h","liquidation_intensity_robust_z","basis_bps","basis_change_robust_z")
        for year in (2023,2024,2025):
            y=ts.dt.year.eq(year); eligible_rows=int((valid&y).sum())
            for field in raw_fields:
                values=pd.to_numeric(p.loc[valid&y,field],errors="coerce")
                raw_distribution_rows.append({"symbol":part["symbol"],"year":year,"field":field,"eligible_rows":eligible_rows,"available_rows":int(values.notna().sum()),"p05":q(values,.05),"median":q(values,.5),"p95":q(values,.95)})
            availability_rows.append({"symbol":part["symbol"],"year":year,"rows":int(y.sum()),"eligible_rows":eligible_rows,"price_available":int((y&source.trade_return_1h.notna()&source.mark_return_1h.notna()&source.trade_log_return_5m.notna()).sum()),"OI_available":int((y&p.oi_log_change_1h.notna()).sum()),"liquidation_available":int((y&source.liquidation_normalization_valid.fillna(False)&source.liquidation_intensity_robust_z.notna()).sum()),"basis_level_available":int((y&source.basis_level_normalization_valid.fillna(False)&source.basis_level_robust_z.notna()).sum()),"basis_change_available":int((y&source.basis_change_normalization_valid.fillna(False)&source.basis_change_robust_z.notna()).sum()),"breadth_available":int((y&pd.Series(breadth.eligible.to_numpy()[pos]>0,index=p.index)).sum())})
        for a in comps:
            for b in comps: component_counts[(a,b)]+=int((valid&p[a].astype(bool)&p[b].astype(bool)).sum())
        for cell in validate_grammar():
            key="+".join(cell); state=valid.copy()
            for comp in cell: state &= p[comp].astype(bool)
            onset=onset_mask(state,valid,ts); state_totals[key]+=int(state.sum()); onset_totals[key]+=int(onset.sum())
            address_memberships=address_memberships+onset.astype("int16")
            cluster_hours[key].update(ts[onset].dt.floor("h").astype(str))
            start=onset.to_numpy(); state_array=state.to_numpy(); durations=[]; onset_years=[]
            for idx in start.nonzero()[0]:
                end=idx
                while end+1<len(state_array) and state_array[end+1] and ts.iloc[end+1]-ts.iloc[end]==pd.Timedelta(minutes=5): end+=1
                durations.append((end-idx+1)*5); onset_years.append(ts.iloc[idx].year)
            duration_frame=pd.DataFrame({"year":onset_years,"duration_minutes":durations})
            for year in (2023,2024,2025):
                y=ts.dt.year.eq(year); yd=duration_frame.loc[duration_frame.year.eq(year),"duration_minutes"] if len(duration_frame) else pd.Series(dtype=float)
                incidence.append({"symbol":part["symbol"],"year":year,"grammar_cell":key,"state_rows":int((state&y).sum()),"onsets":int((onset&y).sum()),"episodes":len(yd),"duration_minutes_median":q(yd,.5),"duration_minutes_p95":q(yd,.95)})
            decisions=ts[onset].dt.as_unit("ns").astype("int64")+300_000_000_000
            pairs=list(zip([part["symbol"]]*len(decisions),decisions.astype(int)))
            for family,keys in parent_keys.items(): parent_overlap_rows.append({"symbol":part["symbol"],"grammar_cell":key,"parent_family":family,"exact_decision_timestamp_overlaps":sum(pair in keys for pair in pairs)})
            if cell == GRAMMAR_LADDERS[1]:
                kdx_trade_samples.append(p.loc[onset,"trade_abs_bps"]); kdx_mark_samples.append(p.loc[onset,"mark_abs_bps"])
        for year in (2023,2024,2025):
            y=ts.dt.year.eq(year); memberships=address_memberships[y]
            address_rows.append({"symbol":part["symbol"],"year":year,"unique_economic_addresses":int(memberships.gt(0).sum()),"addresses_in_multiple_grammar_cells":int(memberships.gt(1).sum()),"total_grammar_memberships":int(memberships.sum()),"maximum_grammar_memberships_per_address":int(memberships.max()) if len(memberships) else 0})
        p.to_parquet(output/"primitive_base"/f"{part['symbol']}.parquet",index=False,compression="zstd")
    pd.DataFrame([{"component_a":a,"component_b":b,"eligible_overlap_rows":v} for (a,b),v in component_counts.items()]).to_csv(ARCHIVE/"KDX01_COMPONENT_OVERLAP_MATRIX.csv",index=False)
    pd.DataFrame(incidence).to_csv(ARCHIVE/"KDX01_INCIDENCE_SUMMARY.csv",index=False)
    pd.DataFrame(raw_distribution_rows).to_csv(ARCHIVE/"KDX01_RAW_UNIT_DISTRIBUTIONS.csv",index=False)
    pd.DataFrame(availability_rows).to_csv(ARCHIVE/"KDX01_COMPONENT_AVAILABILITY_AND_MISSINGNESS.csv",index=False)
    pd.DataFrame(address_rows).to_csv(ARCHIVE/"KDX01_ECONOMIC_ADDRESS_ANALYSIS.csv",index=False)
    pd.DataFrame(parent_overlap_rows).to_csv(ARCHIVE/"KDX01_PARENT_IDENTITY_OVERLAP.csv",index=False)
    incidence_frame=pd.DataFrame(incidence)
    incidence_frame.groupby(["grammar_cell","year"],as_index=False).agg(episodes=("episodes","sum"),duration_minutes_median=("duration_minutes_median","median"),duration_minutes_p95=("duration_minutes_p95","median")).to_csv(ARCHIVE/"KDX01_EPISODE_DURATION_SUMMARY.csv",index=False)
    retention=pd.read_csv(ARCHIVE/"KDA02B_RETENTION_BOUNDARY.csv"); gaps=pd.DataFrame(oi_gap_records); retention=retention.drop(columns=[x for x in ("gap_count","OI_missing_bars_inside_retention","last_raw_OI_timestamp") if x in retention]).merge(gaps,on="symbol",how="left",validate="one_to_one"); retention.to_csv(ARCHIVE/"KDA02B_RETENTION_BOUNDARY.csv",index=False); causal=set(retention.loc[retention.first_complete_causal_lookback.notna(),"symbol"])
    universe=reconcile_universe(pd.read_csv(INVENTORY),symbols)
    for field in ("OI_coverage","basis_coverage","liquidation_coverage"): universe[field]=universe.PF_symbol.map({s:v[field] for s,v in coverage_by_symbol.items()}).fillna(False)
    universe["causal_lookback_eligibility"]=universe.PF_symbol.isin(causal)
    universe["KDA02B_final_eligible"]=universe.PF_symbol.isin(causal)
    universe["KDA02C_or_KDX01_final_eligible"]=universe.PF_symbol.isin(symbols)
    universe.to_csv(ARCHIVE/"KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.csv",index=False)
    fully_covered_outside=int((universe.included.fillna(False)&universe.rankable_trade_coverage&universe.rankable_mark_coverage&~universe.ever_in_authorized_cohort).sum())
    (ARCHIVE/"KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.md").write_text(f"""# Kraken Campaign Universe Reconciliation

The frozen inventory has 479 identities, 460 K0-included PF identities, 187 members ever present in the authorized Stage-8 cohort, and 187 symbols eligible for at least one ready campaign lane. KDA02B has 117 symbols with a complete prior-day causal OI lookback; the other 70 authorized-cache symbols remain eligible only for KDA02C/KDX01 where their required contemporaneous components are available.

There are `{fully_covered_outside}` K0 identities with rankable trade and mark coverage but outside the authorized cohort. None can be admitted to this campaign without changing the cohort contract because they lack the hash-bound Stage-8 feature/cohort membership authority. No symbol was added and no data was acquired.
""",encoding="utf-8")
    onsets=pd.concat([pd.read_parquet(x) for x in sorted((output/"kda02b_onsets").glob("*.parquet"))],ignore_index=True)
    duration_files=[x for x in sorted((output/"kda02b_episodes").glob("*.parquet")) if "duration_minutes" in pq.ParquetFile(x).schema_arrow.names]
    durations=pd.concat([pd.read_parquet(x,columns=["duration_minutes"]) for x in duration_files],ignore_index=True)
    price_key="+".join(GRAMMAR_LADDERS[0]); price_oi_key="+".join(GRAMMAR_LADDERS[1]); price=onset_totals[price_key]; price_oi=onset_totals[price_oi_key]
    kda02b_clusters=int(onsets.state_ts.dt.floor("h").nunique()); minimum_kdx_clusters=min(len(x) for x in cluster_hours.values())
    admissions={"KDA02B_v2_oi_vacuum_redevelopment":"phase_2_ready" if len(onsets)>=1000 and kda02b_clusters>=100 else "mechanism_underidentified",
                "KDA02C_v1_purge_breadth_context":"phase_2_ready" if int(breadth.primary_onset.sum())>=1000 and len(purge_cluster_hours)>=100 else "mechanically_unavailable",
                "KDX01_v1_downside_completed_derivatives_state_rejection":"phase_2_ready" if price_oi>=1000 and price_oi<price and minimum_kdx_clusters>=100 else "mechanism_underidentified"}
    elapsed=time.monotonic()-started
    breadth_rows=[]
    for year,group in breadth.groupby(pd.to_datetime(breadth.timestamp_utc,utc=True).dt.year):
        for identity in ("primary","robust"):
            for minutes in (5,15,30,60):
                onset_roll=group[f"{identity}_onset"].rolling(minutes//5,min_periods=1).sum(); persistence=group[f"{identity}_state"].rolling(minutes//5,min_periods=1).sum()
                breadth_rows.append({"year":year,"identity":identity,"window_minutes":minutes,"timestamps":len(group),"eligible_denominator_min":int(group.eligible.min()),"eligible_denominator_median":q(group.eligible,.5),"eligible_denominator_max":int(group.eligible.max()),"onsets":int(group[f"{identity}_onset"].sum()),"isolated_timestamp_frequency":float(group[f"{identity}_onset"].eq(1).mean()),"broad_timestamp_frequency_ge_2":float(group[f"{identity}_onset"].ge(2).mean()),"rolling_onset_count_p95":q(onset_roll,.95),"rolling_onset_count_p99":q(onset_roll,.99),"active_state_persistence_bars_p95":q(persistence,.95),"cohort_share_p95":q(group[f"{identity}_cohort_share"],.95),"membership_changes":int(group.membership_change.sum())})
    pd.DataFrame(breadth_rows).to_csv(ARCHIVE/"KDA02C_BREADTH_SUMMARY.csv",index=False)
    breadth.to_parquet(output/"KDA02C_PIT_BREADTH_PANEL.parquet",index=False,compression="zstd")
    measurement={"generated_at":FIXED_AT,"economic_outputs_computed":False,"protected_rows_opened":0,"Capitalcom_payload_opened":0,
      "feature_rows_read":sum(int(x["rows"]) for x in parts),"feature_bytes_read":sum(Path(x["path"]).stat().st_size for x in parts),
      "outcome_read_spy":{"request_count":len(spy.requests),"requests":spy.requests,"feature_request_count":sum(x["kind"]=="feature" for x in spy.requests),"parent_request_count":sum(x["kind"]=="parent" for x in spy.requests),"allowed_feature_columns":list(FEATURE_COLUMNS),"allowed_parent_columns":list(PARENT_EVENT_COLUMNS)},
      "KDA02B":{"episodes":len(onsets),"symbols":int(onsets.symbol.nunique()),"conservative_UTC_hour_clusters":kda02b_clusters,"median_duration_minutes":q(durations.duration_minutes,.5),"trade_abs_bps_median":q(onsets.trade_displacement_bps.abs(),.5),"trade_abs_ge_14_share":float(onsets.trade_displacement_bps.abs().ge(14).mean()),"trade_abs_ge_32_share":float(onsets.trade_displacement_bps.abs().ge(32).mean())},
      "KDA02C":{"timestamps":len(breadth),"median_denominator":q(breadth.eligible,.5),"primary_onsets":int(breadth.primary_onset.sum()),"robust_onsets":int(breadth.robust_onset.sum()),"conservative_UTC_hour_clusters":len(purge_cluster_hours),"primary_onset_trade_abs_bps_median":q(pd.concat(purge_trade_samples),.5),"primary_onset_trade_abs_ge_14_share":float(pd.concat(purge_trade_samples).ge(14).mean()),"primary_onset_trade_abs_ge_32_share":float(pd.concat(purge_trade_samples).ge(32).mean()),"primary_onset_mark_abs_bps_median":q(pd.concat(purge_mark_samples),.5)},
      "KDX01":{"price_onsets":price,"price_oi_onsets":price_oi,"grammar_state_rows":state_totals,"grammar_onsets":onset_totals,"grammar_conservative_UTC_hour_clusters":{k:len(v) for k,v in cluster_hours.items()},"minimum_grammar_UTC_hour_clusters":minimum_kdx_clusters,"price_oi_trade_abs_bps_median":q(pd.concat(kdx_trade_samples),.5),"price_oi_trade_abs_ge_14_share":float(pd.concat(kdx_trade_samples).ge(14).mean()),"price_oi_trade_abs_ge_32_share":float(pd.concat(kdx_trade_samples).ge(32).mean()),"price_oi_mark_abs_bps_median":q(pd.concat(kdx_mark_samples),.5),"parent_event_exact_timestamp_overlap":pd.DataFrame(parent_overlap_rows).groupby("parent_family").exact_decision_timestamp_overlaps.sum().astype(int).to_dict(),"duplicate_economic_addresses":int(pd.DataFrame(address_rows).addresses_in_multiple_grammar_cells.sum())},"admissions":admissions,
      "benchmark":{"finalization_replay_wall_seconds":elapsed,"observed_full_scan_peak_rss_kib_lower_bound":463124,"finalization_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"feature_rows":sum(int(x["rows"]) for x in parts),"workers":1,"deterministic_cell_projection_method":"measured full outcome-free primitive replay"}}
    (output/"PHASE1_MEASUREMENT.json").write_text(json.dumps(measurement,indent=2,sort_keys=True,default=str)+"\n")
    (ARCHIVE/"PHASE1_ADMISSION_DECISIONS.json").write_text(json.dumps({"generated_at":FIXED_AT,"decisions":admissions,"criteria":"outcome-free incidence, causal completeness, raw scale, exact bounded grammar, funding and benchmark contracts"},indent=2,sort_keys=True)+"\n")
    write_measurement_docs(ARCHIVE,measurement,pd.read_csv(FUNDING_MANIFEST))
    print(json.dumps({"status":"complete","admissions":admissions,"replay_wall_seconds":elapsed,"price_onsets":price,"price_oi_onsets":price_oi},sort_keys=True))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); finalize(a.output)
