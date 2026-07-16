#!/usr/bin/env python3
"""Ledger-only RFBS v1_010 train stability review."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tools import qlmg_signal_state_contract as state
from tools import run_kraken_rfbs_signal_state_repaired as repaired


RUN_ROOT = Path("results/rebaseline/phase_kraken_rfbs_010_train_only_stability_review_20260715_v1")
SCREEN_ROOT = Path("results/rebaseline/phase_kraken_rfbs_signal_state_repaired_screen_20260715_v1")
MATERIALIZATION_ROOT = Path("results/rebaseline/phase_kraken_rfbs_signal_state_repaired_materialization_20260715_v1")
CLOSURE_ROOT = Path("results/rebaseline/phase_kraken_cross_family_repair_campaign_closure_20260715_v1")
CAMPAIGN_ROOT = Path("results/rebaseline/phase_kraken_cross_family_signal_state_repair_campaign_20260715_v1")
FORMAL_ID = "rfbs_v1_010"
NEIGHBOR_ID = "rfbs_v1_007"
SEED = 20260715
START_LEVEL = "level_4_event_ledger_plus_real_controls"
PASS_LEVEL = "level_5_walkforward_cpcv_parameter_stability"
CONTRACT_VERSION = "rfbs_010_train_only_stability_review_v1_20260715"
MODES = {
    "central": "net_base_R", "conservative": "net_conservative_R",
    "severe": "net_severe_R", "zero_funding": "net_zero_funding_base_R",
}
ADEQUATE_CLASSES = (
    "same_symbol_same_regime_random_short", "countertrend_rally_without_completed_failure",
    "generic_20d_failed_breakout_short", "non_rally_red_candle_short",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    return state.stable_hash([(str(path.relative_to(root)), file_hash(path)) for path in sorted(root.rglob("*")) if path.is_file()])


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def pf(values: pd.Series) -> float:
    positive = values[values > 0].sum(); negative = -values[values < 0].sum()
    return float(positive / negative) if negative > 0 else np.inf if positive > 0 else np.nan


def metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = frame[column].dropna()
    return {
        "events": len(values), "mean_R": values.mean(), "median_R": values.median(), "total_R": values.sum(),
        "profit_factor": pf(values), "hit_rate": values.gt(0).mean(), "symbols": frame.symbol.nunique(),
        "months": pd.to_datetime(frame.entry_ts, utc=True).dt.strftime("%Y-%m").nunique(),
        "fully_exact_events": int(frame.funding_partition.eq("fully_exact").sum()),
        "mixed_events": int(frame.funding_partition.eq("mixed").sum()),
        "fully_imputed_events": int(frame.funding_partition.eq("fully_imputed").sum()),
    }


def bootstrap_cluster(frame: pd.DataFrame, cluster_col: str, iterations: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + sum(map(ord, cluster_col)))
    groups = {key: group for key, group in frame.groupby(cluster_col)}
    keys = np.array(list(groups), dtype=object)
    rows = []
    for mode, column in MODES.items():
        means, pfs = [], []
        for _ in range(iterations):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            values = pd.concat([groups[key][column] for key in sampled], ignore_index=True)
            means.append(values.mean()); pfs.append(pf(values))
        rows.append({
            "cluster_scheme": cluster_col, "mode": mode, "iterations": iterations,
            "mean_ci_025": np.quantile(means, .025), "mean_median": np.median(means), "mean_ci_975": np.quantile(means, .975),
            "pf_ci_025": np.nanquantile(pfs, .025), "pf_median": np.nanmedian(pfs), "pf_ci_975": np.nanquantile(pfs, .975),
            "positive_mean_probability": np.mean(np.asarray(means) > 0),
        })
    return pd.DataFrame(rows)


def compact_bundle(root: Path, files: tuple[str, ...]) -> Path:
    temp = root / ".compact_review_bundle.tmp"
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir()
    inventory = []
    for relative in files:
        source = root / relative; target = temp / relative.replace("/", "__")
        shutil.copy2(source, target)
        inventory.append({"source_path": relative, "bundle_path": target.name, "sha256": file_hash(source), "bytes": source.stat().st_size})
    write_csv(temp / "bundle_manifest.csv", inventory)
    final = root / "compact_review_bundle"
    if final.exists(): shutil.rmtree(final)
    os.replace(temp, final)
    return final


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True); started = time.monotonic()
    closure_repro = json.loads((CLOSURE_ROOT / "reproducibility/run_manifest.json").read_text())
    source_hashes = {"screen": tree_hash(SCREEN_ROOT), "materialization": tree_hash(MATERIALIZATION_ROOT)}
    expected_hashes = {"screen": closure_repro["screen_root_hash"], "materialization": closure_repro["materialization_root_hash"]}
    hash_audit = [{"source": key, "expected_hash": expected_hashes[key], "actual_hash": source_hashes[key], "pass": source_hashes[key] == expected_hashes[key]} for key in source_hashes]
    write_csv(root / "audit/source_hash_validation.csv", hash_audit)
    if not all(row["pass"] for row in hash_audit): raise RuntimeError("immutable source hash mismatch")

    all_events = pd.read_csv(SCREEN_ROOT / "materialized/event_ledger.csv")
    controls = pd.read_csv(SCREEN_ROOT / "controls/control_event_ledger.csv")
    candidate = all_events[all_events.definition_id.eq(FORMAL_ID)].copy()
    neighbor = all_events[all_events.definition_id.eq(NEIGHBOR_ID)].copy()
    for frame in (candidate, neighbor, controls):
        for column in [name for name in frame.columns if name.endswith("_ts")]: frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if len(candidate) != 148: raise RuntimeError(f"formal candidate row mismatch: {len(candidate)}")
    candidate["funding_partition"] = np.select(
        [candidate.exact_funding_boundaries.gt(0) & candidate.imputed_funding_boundaries.eq(0), candidate.exact_funding_boundaries.gt(0)],
        ["fully_exact", "mixed"], default="fully_imputed",
    )
    neighbor["funding_partition"] = np.select(
        [neighbor.exact_funding_boundaries.gt(0) & neighbor.imputed_funding_boundaries.eq(0), neighbor.exact_funding_boundaries.gt(0)],
        ["fully_exact", "mixed"], default="fully_imputed",
    )
    candidate["month"] = candidate.entry_ts.dt.strftime("%Y-%m")
    candidate["symbol_month"] = candidate.symbol + "|" + candidate.month

    # Freeze designs from timestamps and counts only.
    fold_rows = []
    train_start = pd.Timestamp("2023-01-01", tz="UTC")
    test_start = train_start + pd.DateOffset(months=18)
    while test_start < pd.Timestamp("2026-01-01", tz="UTC"):
        test_end = min(test_start + pd.DateOffset(months=3), pd.Timestamp("2026-01-01", tz="UTC"))
        train_end = test_start - pd.Timedelta(hours=72)
        test = candidate[(candidate.entry_ts >= test_start) & (candidate.entry_ts < test_end)]
        train = candidate[(candidate.entry_ts >= train_start) & (candidate.exit_ts < train_end)]
        fold_rows.append({"fold_id": f"WF_{len(fold_rows)+1:02d}", "train_start": train_start, "train_end_exclusive": train_end, "test_start": test_start, "test_end_exclusive": test_end, "train_events_after_purge": len(train), "test_events": len(test), "underpowered": len(test) < 15, "embargo_hours": 72})
        test_start += pd.DateOffset(months=3)
    fold_contract = pd.DataFrame(fold_rows)
    write_csv(root / "contract/walk_forward_fold_contract.csv", fold_contract)

    sorted_events = candidate.sort_values("entry_ts").reset_index(drop=True)
    designs = []
    selected_k = None
    for k in (6, 8, 10):
        blocks = np.array_split(np.arange(len(sorted_events)), k)
        counts = [len(block) for block in blocks]
        adequate = min(counts) >= 15
        designs.append({"K": k, "block_counts": "|".join(map(str, counts)), "minimum_block_events": min(counts), "all_blocks_at_least_15": adequate})
        if adequate: selected_k = k
    write_csv(root / "contract/cpcv_design_selection.csv", designs)
    if selected_k is None: raise RuntimeError("CPCV underpowered for all K candidates")
    cpcv_blocks = np.array_split(np.arange(len(sorted_events)), selected_k)
    block_map = pd.DataFrame([{"event_id": sorted_events.iloc[index].event_id, "block_id": block_id} for block_id, block in enumerate(cpcv_blocks) for index in block])
    write_csv(root / "contract/cpcv_block_manifest.csv", block_map)
    contract = {
        "candidate": FORMAL_ID, "neighbour": NEIGHBOR_ID, "candidate_fixed_no_fold_selection": True,
        "walk_forward": "18_month_train_3_month_test_3_month_step", "minimum_test_events": 15,
        "purge_hours": 72, "embargo_hours": 72, "cpcv_K_candidates": [6,8,10], "selected_K": selected_k,
        "cpcv_test_blocks_per_path": 2, "bootstrap_seed": SEED, "risk_match_tolerance_daily_atr": .25,
        "design_frozen_before_fold_outcome_aggregation": True,
    }
    write_json(root / "contract/stability_design_contract.json", contract)

    walk = []
    for fold in fold_rows:
        test = candidate[(candidate.entry_ts >= fold["test_start"]) & (candidate.entry_ts < fold["test_end_exclusive"])]
        for mode, column in MODES.items(): walk.append({**fold, "mode": mode, **metrics(test, column)})
    walk = pd.DataFrame(walk); write_csv(root / "walk_forward/rolling_walk_forward_results.csv", walk)

    cpcv = []
    sorted_events = sorted_events.merge(block_map, on="event_id", how="left")
    for path_number, test_blocks in enumerate(itertools.combinations(range(selected_k), 2), 1):
        test = sorted_events[sorted_events.block_id.isin(test_blocks)]
        test_start, test_end = test.entry_ts.min(), test.exit_ts.max()
        train = sorted_events[~sorted_events.block_id.isin(test_blocks) & ((sorted_events.exit_ts < test_start-pd.Timedelta(hours=72)) | (sorted_events.entry_ts >= test_end+pd.Timedelta(hours=72)))]
        for mode, column in MODES.items(): cpcv.append({"path_id": f"CPCV_{path_number:03d}", "test_blocks": "|".join(map(str,test_blocks)), "mode": mode, "train_events_after_purge_embargo": len(train), **metrics(test, column)})
    cpcv = pd.DataFrame(cpcv); write_csv(root / "cpcv/cpcv_path_results.csv", cpcv)
    cpcv_dist = cpcv.groupby("mode").agg(paths=("path_id","nunique"),median_mean_R=("mean_R","median"),worst_mean_R=("mean_R","min"),positive_path_fraction=("mean_R",lambda x:x.gt(0).mean()),median_profit_factor=("profit_factor","median")).reset_index()
    write_csv(root / "cpcv/cpcv_distribution_summary.csv", cpcv_dist)

    # Multiplicity: PSR plus Bonferroni-adjusted PSR, CSCV/PBO, and centered month-block max-t reality check.
    # Deflate the formal t-statistic by the expected maximum null statistic
    # across the frozen 24-definition search family.
    all_events["month"] = pd.to_datetime(all_events.entry_ts, utc=True).dt.strftime("%Y-%m")
    definitions = sorted(all_events.definition_id.unique())
    euler_gamma = 0.5772156649015329
    expected_max_null_t = (
        (1 - euler_gamma) * stats.norm.ppf(1 - 1 / len(definitions))
        + euler_gamma * stats.norm.ppf(1 - 1 / (len(definitions) * math.e))
    )
    multi_rows = []
    for mode, column in {"conservative":"net_conservative_R","severe":"net_severe_R"}.items():
        values = candidate[column].dropna(); n=len(values); mean=values.mean(); sd=values.std(ddof=1); t=mean/(sd/math.sqrt(n)); psr=stats.norm.cdf(t)
        dsr_equivalent = stats.norm.cdf(t - expected_max_null_t)
        multi_rows.append({"definition_id":FORMAL_ID,"mode":mode,"events":n,"mean_R":mean,"studentized_mean":t,"psr_probability_positive":psr,"expected_max_null_t_24_trials":expected_max_null_t,"dsr_equivalent_probability":dsr_equivalent,"dsr_equivalent_pass":dsr_equivalent>.95,"bonferroni_24_threshold":1-.05/24,"bonferroni_adjusted_psr_pass":psr>1-.05/24})
    multi = pd.DataFrame(multi_rows)
    block_labels = sorted(all_events.month.unique())
    block_groups = np.array_split(np.array(block_labels,dtype=object), 8)
    pbo_rows=[]
    for combo in itertools.combinations(range(8),4):
        train_months=set(np.concatenate([block_groups[i] for i in combo])); train=all_events[all_events.month.isin(train_months)]; test=all_events[~all_events.month.isin(train_months)]
        train_means=train.groupby("definition_id").net_conservative_R.mean().reindex(definitions); winner=train_means.idxmax(); test_means=test.groupby("definition_id").net_conservative_R.mean().reindex(definitions)
        rank=test_means.rank(pct=True).loc[winner]
        pbo_rows.append({"train_blocks":"|".join(map(str,combo)),"train_winner":winner,"test_percentile_rank":rank,"overfit":rank<.5,"formal_candidate_test_rank":test_means.rank(pct=True).loc[FORMAL_ID]})
    pbo=pd.DataFrame(pbo_rows); write_csv(root / "multiplicity/cscv_pbo_paths.csv",pbo)
    rng=np.random.default_rng(SEED); pivot=all_events.pivot_table(index="month",columns="definition_id",values="net_conservative_R",aggfunc="mean").fillna(0); centered=pivot-pivot.mean(); observed={d:all_events[all_events.definition_id.eq(d)].net_conservative_R.mean()/(all_events[all_events.definition_id.eq(d)].net_conservative_R.std(ddof=1)/math.sqrt(len(all_events[all_events.definition_id.eq(d)]))) for d in definitions}
    max_t=[]
    for _ in range(3000):
        sample=centered.iloc[rng.integers(0,len(centered),len(centered))]; means=sample.mean(); ses=sample.std(ddof=1)/math.sqrt(len(sample)); max_t.append((means/ses.replace(0,np.nan)).max())
    formal_t=observed[FORMAL_ID]; reality_p=float(np.mean(np.asarray(max_t)>=formal_t))
    multi["cscv_pbo_probability"] = pbo.overfit.mean(); multi["reality_check_max_t_pvalue"] = reality_p; multi["multiplicity_adjusted_pass"] = multi.dsr_equivalent_pass & multi.bonferroni_adjusted_psr_pass & (reality_p < .10) & (pbo.overfit.mean() < .5)
    write_csv(root / "multiplicity/psr_dsr_reality_check.csv",multi)
    write_json(root / "multiplicity/method_contract.json",{"method":"PSR; expected-maximum-null-t DSR equivalent plus Bonferroni-24 positive-mean threshold; 8-block CSCV/PBO; centered monthly block max-t White Reality Check equivalent","definitions":24,"bootstrap_iterations":3000,"no_candidate_reselection":True})

    boot = pd.concat([bootstrap_cluster(candidate,"month"),bootstrap_cluster(candidate,"symbol"),bootstrap_cluster(candidate,"symbol_month")],ignore_index=True)
    write_csv(root / "bootstrap/clustered_bootstrap_summary.csv",boot)
    removals=[]
    for mode,column in MODES.items():
        ordered=candidate.sort_values(column,ascending=False)
        for label,removed in (("none",0),("top_one",1),("top_three",3),("top_1pct",max(1,math.ceil(len(candidate)*.01)))):
            remaining=ordered.iloc[removed:]; removals.append({"mode":mode,"removal":label,"removed_events":removed,**metrics(remaining,column)})
    write_csv(root / "bootstrap/winner_removal_stability.csv",removals)
    fold_removals=[]
    for fold in fold_rows:
        test=candidate[(candidate.entry_ts>=fold["test_start"])&(candidate.entry_ts<fold["test_end_exclusive"])]
        for mode,column in MODES.items():
            ordered=test.sort_values(column,ascending=False)
            for label,removed in (("none",0),("top_one",1),("top_three",3),("top_1pct",max(1,math.ceil(len(test)*.01)))):
                fold_removals.append({"fold_id":fold["fold_id"],"underpowered":fold["underpowered"],"mode":mode,"removal":label,"removed_events":min(removed,len(ordered)),**metrics(ordered.iloc[removed:],column)})
    write_csv(root / "bootstrap/walk_forward_winner_removal.csv",fold_removals)

    # Neighbour by the same frozen folds/blocks; no replacement selection.
    neighbor_walk=[]
    for fold in fold_rows:
        for definition_id,frame in ((FORMAL_ID,candidate),(NEIGHBOR_ID,neighbor)):
            test=frame[(frame.entry_ts>=fold["test_start"])&(frame.entry_ts<fold["test_end_exclusive"])]
            neighbor_walk.append({"fold_id":fold["fold_id"],"definition_id":definition_id,"events":len(test),"conservative_mean_R":test.net_conservative_R.mean(),"severe_mean_R":test.net_severe_R.mean(),"underpowered":len(test)<15})
    write_csv(root / "neighborhood/fold_comparison_007_010.csv",neighbor_walk)
    shared=set(candidate.raw_signal_address_hash)&set(neighbor.raw_signal_address_hash); candidate["partition"]=np.where(candidate.raw_signal_address_hash.isin(shared),"shared_strict_parent","broader_only")
    partition=candidate.groupby(["partition","evaluation_period"]).agg(events=("event_id","size"),conservative_mean_R=("net_conservative_R","mean"),severe_mean_R=("net_severe_R","mean")).reset_index(); write_csv(root / "neighborhood/shared_broader_partition.csv",partition)

    # Frozen controls: official, candidate-band-only, and predeclared +/-0.25 ATR-distance match diagnostics.
    controls=controls[controls.definition_id.eq(FORMAL_ID)&controls.control_class.isin(ADEQUATE_CLASSES)].copy(); controls["risk_to_daily_atr"]=controls.risk_denominator/controls.daily_atr
    candidate_risk=candidate.set_index("candidate_key").risk_denominator/candidate.set_index("candidate_key").daily_atr
    controls["candidate_risk_to_daily_atr"]=controls.candidate_key.map(candidate_risk); controls["inside_candidate_band"]=controls.risk_to_daily_atr.between(.25,1.5); controls["risk_distance_match"]=((controls.risk_to_daily_atr-controls.candidate_risk_to_daily_atr).abs()<=.25)
    control_fold=[]
    for fold in fold_rows:
        test=candidate[(candidate.entry_ts>=fold["test_start"])&(candidate.entry_ts<fold["test_end_exclusive"])]
        for cls in ADEQUATE_CLASSES:
            group=controls[controls.control_class.eq(cls)&controls.candidate_key.isin(test.candidate_key)]
            for diagnostic,subset in (("official_raw",group),("candidate_band_only",group[group.inside_candidate_band]),("risk_distance_matched",group[group.risk_distance_match])):
                matched=test[test.candidate_key.isin(subset.candidate_key)]; unmatched=test[~test.candidate_key.isin(subset.candidate_key)]
                row={"fold_id":fold["fold_id"],"control_class":cls,"diagnostic":diagnostic,"candidate_test_events":len(test),"matched_events":len(matched),"unmatched_events":len(unmatched),"coverage":len(matched)/max(1,len(test)),"underpowered":len(matched)<15}
                for mode,column in MODES.items():
                    row[f"candidate_{mode}_mean_R"]=matched[column].mean(); row[f"control_{mode}_mean_R"]=subset[column].mean(); row[f"{mode}_uplift_R"]=matched[column].mean()-subset[column].mean()
                control_fold.append(row)
    control_fold=pd.DataFrame(control_fold); write_csv(root / "controls/fold_level_control_uplift.csv",control_fold)
    write_csv(root / "controls/frozen_control_diagnostic_manifest.csv",controls[["control_event_id","candidate_key","control_class","risk_to_daily_atr","candidate_risk_to_daily_atr","inside_candidate_band","risk_distance_match"]])
    complements=pd.read_csv(MATERIALIZATION_ROOT / "controls/matched_unmatched_bias_repaired.csv"); write_csv(root / "controls/matched_unmatched_complements.csv",complements[(complements.definition_id==FORMAL_ID)&complements.control_class.isin(ADEQUATE_CLASSES)])

    funding=[]
    for partition_name,frame in candidate.groupby("funding_partition"):
        for mode,column in MODES.items(): funding.append({"funding_partition":partition_name,"mode":mode,**metrics(frame,column)})
    write_csv(root / "funding/funding_partition_stability.csv",funding)
    period_funding=[]
    for (period,partition_name),frame in candidate.groupby(["evaluation_period","funding_partition"]):
        for mode,column in MODES.items(): period_funding.append({"evaluation_period":period,"funding_partition":partition_name,"mode":mode,**metrics(frame,column)})
    write_csv(root / "funding/period_funding_stability.csv",period_funding)
    integrity=pd.read_csv(CLOSURE_ROOT / "forensics/rfbs_010_event_path_integrity.csv"); pathological=set(integrity[integrity.pathological_wick_diagnostic].event_id); no_wick=candidate[~candidate.event_id.isin(pathological)]
    sensitivity=[]
    for label,frame in (("official_all",candidate),("remove_preexisting_pathological_wicks",no_wick)):
        for mode,column in MODES.items(): sensitivity.append({"diagnostic":label,"mode":mode,"removed_events":len(candidate)-len(frame),**metrics(frame,column)})
    for bps in (4,8,12):
        sensitivity.append({"diagnostic":f"additional_{bps}bps_roundtrip", "mode":"conservative", "removed_events":0, **metrics(candidate.assign(stressed=candidate.net_conservative_R-(bps/1e4)*candidate.entry_price/candidate.risk_denominator),"stressed")})
    write_csv(root / "execution/funding_execution_sensitivity.csv",sensitivity)
    write_csv(root / "execution/mark_gap_diagnostics.csv",integrity[["event_id","entry_trade_mark_gap_bps","exit_trade_mark_gap_bps","mark_entry_available","mark_exit_available","pathological_wick_diagnostic"]])

    powered_walk=walk[(walk["mode"].isin(["conservative","severe"]))&~walk.underpowered]; walk_pass=bool(len(powered_walk) and powered_walk.groupby("mode").mean_R.median().gt(0).all() and powered_walk.groupby("mode").mean_R.apply(lambda x:x.gt(0).mean()).gt(.5).all())
    cpcv_key=cpcv_dist[cpcv_dist["mode"].isin(["conservative","severe"])]
    cpcv_pass=bool(cpcv_key.median_mean_R.gt(0).all() and cpcv_key.positive_path_fraction.gt(.5).all())
    multiplicity_pass=bool(multi.multiplicity_adjusted_pass.all())
    clustered=boot[(boot["mode"].isin(["conservative","severe"]))]; cluster_pass=bool(clustered.mean_ci_025.gt(0).all())
    official=control_fold[(control_fold.diagnostic=="official_raw")&~control_fold.underpowered]; stable_classes=official.groupby("control_class").conservative_uplift_R.apply(lambda x: bool(len(x) and x.median()>0 and x.gt(0).mean()>.5)); control_pass=int(stable_classes.sum())>=2
    neighbor_powered=pd.DataFrame(neighbor_walk); neighbor_powered=neighbor_powered[~neighbor_powered.underpowered]; neighbor_pass=bool(neighbor_powered.groupby("definition_id").conservative_mean_R.median().gt(0).all())
    mechanics_pass=bool((json.loads((CLOSURE_ROOT/"decision_summary.json").read_text())["mechanical_failures"]==0) and not candidate.protected_violation.any())
    gates=pd.DataFrame([{"gate":"rolling_paths","pass":walk_pass},{"gate":"cpcv_paths","pass":cpcv_pass},{"gate":"multiplicity_adjustment","pass":multiplicity_pass},{"gate":"clustered_confidence","pass":cluster_pass},{"gate":"two_stable_controls","pass":control_pass},{"gate":"neighborhood_consistency","pass":neighbor_pass},{"gate":"mechanics","pass":mechanics_pass}]); write_csv(root / "decision/level5_gate_audit.csv",gates)
    passed=bool(gates["pass"].all()); final="level_5_train_only_stability_pass_with_caps" if passed else "fragile_context_sleeve"
    final_level=PASS_LEVEL if passed else START_LEVEL

    library=pd.read_csv(CLOSURE_ROOT/"candidate_library/central_full_schema_candidate_library.csv"); mask=library.definition_id.eq(FORMAL_ID); library.loc[mask,["candidate_library_state","candidate_decision"]]=final; library.loc[mask,"evidence_level"]=final_level; library.loc[mask,"evidence_cap_reason"]="funding_imputation|ohlcv_stop|no_depth|near_150_events|negative_exact_funding|negative_2023"; library["stability_review_root"]=str(root); write_csv(root/"candidate_library/central_full_schema_candidate_library.csv",library)
    next_target="human pre-holdout data-sufficiency review for RFBS 010 only" if passed else "Close-confirmed breakout retest long screen"
    launch_path=CAMPAIGN_ROOT/"campaign/new_family_launch_gate.json"; launch=json.loads(launch_path.read_text()); launch.update({"rfbs_v1_010_stability_decision":final,"rfbs_v1_010_evidence_level":final_level,"final_holdout_sealed":True,"next_authorized_target":next_target,"next_prompt_target":next_target,"new_family_launch_allowed":not passed}); write_json(launch_path,launch)
    repaired.refresh_campaign_bundle()
    continuity={"formal_candidate":FORMAL_ID,"decision":final,"evidence_level":final_level,"next_authorized_target":next_target,"final_holdout_sealed":True,"paused_family_authorized":not passed,"signals_regenerated":False,"controls_regenerated":False}; write_json(root/"continuity/continuity_state_snapshot.json",continuity)

    source_hashes_after={"screen":tree_hash(SCREEN_ROOT),"materialization":tree_hash(MATERIALIZATION_ROOT)}; mutations=sum(source_hashes[k]!=source_hashes_after[k] for k in source_hashes); write_csv(root/"audit/source_immutability_audit.csv",[{"source":k,"before":source_hashes[k],"after":source_hashes_after[k],"mutated":source_hashes[k]!=source_hashes_after[k]} for k in source_hashes]);
    if mutations: raise RuntimeError("source root mutation")
    report=f"""# RFBS 010 Train-Only Stability Review\n\nStarting evidence: `{START_LEVEL}`. Final decision: `{final}`; final evidence: `{final_level}`. This review used immutable repaired ledgers, fixed `rfbs_v1_010`, 72-hour purge/embargo, 18m/3m/3m walk-forward, K={selected_k} CPCV, 24-definition multiplicity burden, clustered bootstraps, and frozen controls. No final holdout was opened. Negative 2023 and exact-funded evidence, imputation, OHLCV-stop, no-depth and near-150-event caps remain active. Next authorized target: `{next_target}`.\n"""; (root/"STABILITY_REVIEW_REPORT.md").write_text(report)
    repro={"commit_hash":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"code_path":str(Path(__file__)),"code_hash":file_hash(Path(__file__)),"contract_version":CONTRACT_VERSION,"source_hashes":source_hashes,"seed":SEED,"protected_boundary":"2026-01-01T00:00:00Z","signals_regenerated":False,"controls_regenerated":False}; write_json(root/"reproducibility/run_manifest.json",repro)
    decision={"run_root":str(root),"status":"complete","formal_candidate":FORMAL_ID,"starting_evidence_level":START_LEVEL,"final_evidence_level":final_level,"final_decision":final,"selected_cpcv_K":selected_k,"walk_forward_pass":walk_pass,"cpcv_pass":cpcv_pass,"multiplicity_pass":multiplicity_pass,"clustered_confidence_pass":cluster_pass,"controls_pass":control_pass,"neighborhood_pass":neighbor_pass,"mechanics_pass":mechanics_pass,"event_count":len(candidate),"negative_exact_funding_cap":True,"negative_2023_cap":True,"source_mutations":mutations,"final_holdout_sealed":True,"next_authorized_target":next_target,"validation_launched":False,"holdout_launched":False,"runtime_seconds":time.monotonic()-started,"compact_bundle_path":str(root/"compact_review_bundle")}; write_json(root/"decision_summary.json",decision)
    compact_bundle(root,("STABILITY_REVIEW_REPORT.md","decision_summary.json","reproducibility/run_manifest.json","contract/stability_design_contract.json","contract/walk_forward_fold_contract.csv","contract/cpcv_design_selection.csv","walk_forward/rolling_walk_forward_results.csv","cpcv/cpcv_distribution_summary.csv","multiplicity/psr_dsr_reality_check.csv","multiplicity/cscv_pbo_paths.csv","bootstrap/clustered_bootstrap_summary.csv","bootstrap/winner_removal_stability.csv","bootstrap/walk_forward_winner_removal.csv","neighborhood/fold_comparison_007_010.csv","neighborhood/shared_broader_partition.csv","controls/fold_level_control_uplift.csv","controls/matched_unmatched_complements.csv","funding/funding_partition_stability.csv","funding/period_funding_stability.csv","execution/funding_execution_sensitivity.csv","execution/mark_gap_diagnostics.csv","decision/level5_gate_audit.csv","candidate_library/central_full_schema_candidate_library.csv","continuity/continuity_state_snapshot.json","audit/source_hash_validation.csv","audit/source_immutability_audit.csv"))
    return decision


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-root",type=Path,default=RUN_ROOT); args=parser.parse_args(); result=run(args.run_root); print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
