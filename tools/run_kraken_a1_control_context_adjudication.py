#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_shared_funding_consumer as consumer
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import kraken_a1_targeted_materialization as targeted
from tools.kraken_a1_full_streaming_reducer import _funding_context, _panel_for_boundaries


SOURCE = Path("results/rebaseline/phase_kraken_a1_compression_targeted_materialization_controls_stress_20260712_v1")
FULL = Path("results/rebaseline/phase_kraken_a1_compression_full_180shard_funding_corrected_20260711_v1_20260711_194859")
FUNDING = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
DEFAULT_ROOT = Path("results/rebaseline/phase_kraken_a1_control_context_adjudication_20260712_v1")
MODES = ("exact_only_slice", "central_imputed", "conservative_imputed", "severe_imputed")
BPS = (4, 8, 12)
CONTROL_TYPES = ("same_symbol", "same_regime", "generic_breakout", "donchian_simple_breakout")
SPECIAL = {
    "a1cmp_v1_074_structure_base_failure_time_10d",
    "a1cmp_v1_079_structure_base_failure_time_10d",
    "a1cmp_v1_143_failed_close_inside_range_time_5d",
    "a1cmp_v1_160_structure_base_failure_time_10d",
    "a1cmp_v1_121_failed_close_inside_range_time_5d",
}


def csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    runner.write_csv(path, frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame))


def canonical_hash(frame: pd.DataFrame, keys: list[str]) -> str:
    return runner.canonical_frame_hash(frame, sort_keys=keys)


def load_candidate_scenarios() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(SOURCE / "materialized/materialized_ledger_manifest.csv")
    parts = [pd.read_parquet(SOURCE / path) for path in manifest["path"]]
    scenarios = pd.concat(parts, ignore_index=True, sort=False)
    shortlist = pd.read_csv("results/rebaseline/phase_kraken_a1_compression_survivor_materialization_preflight_20260712_v1/selection/survivor_shortlist.csv")
    primary = set(shortlist.loc[shortlist.exit_role.eq("primary"), "candidate_definition_id"].astype(str))
    return scenarios[scenarios.candidate_definition_id.astype(str).isin(primary)].copy(), shortlist[shortlist.exit_role.eq("primary")].copy()


def rescore_controls(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    raw_columns = [c for c in ledger.columns if c not in {"candidate_event_key", "matched_candidate_id", "control_type", "control_candidate_definition_id", "control_symbol", "control_decision_ts", "control_exit_policy_id", "control_shard_id", "match_basis", "control_selection_uses_outcome", "control_key_freeze_hash", "control_outcome_read_after_freeze", "control_severe_12bps_net_R", "candidate_severe_12bps_net_R", "exact_boundary_rows", "imputed_boundary_rows"}]
    unique = ledger[raw_columns].drop_duplicates("event_id").copy()
    unique["candidate_definition_id"] = ledger.drop_duplicates("event_id")["control_candidate_definition_id"].to_numpy()
    unique["symbol"] = ledger.drop_duplicates("event_id")["control_symbol"].to_numpy()
    unique["decision_ts"] = ledger.drop_duplicates("event_id")["control_decision_ts"].to_numpy()
    unique["exit_policy_id"] = ledger.drop_duplicates("event_id")["control_exit_policy_id"].to_numpy()
    events = consumer.normalize_frozen_events(unique, "a1")
    boundaries = consumer.build_event_boundary_rows(events)
    model = json.loads((FUNDING / "decision_summary.json").read_text())
    context = _funding_context(FUNDING, sorted(events.symbol.astype(str).unique()), str(model["selected_model_hash"]))
    panel, _ = _panel_for_boundaries(boundaries, context)
    joined = consumer.join_boundaries_to_panel(boundaries, panel)
    audit = {
        "missing": int((joined._merge != "both").sum()),
        "duplicate": int(joined.duplicated(["event_key", "boundary_ts"]).sum()),
        "imputed_gate": int((joined.funding_imputed.fillna(False) & joined.funding_gate_eligible.fillna(False)).sum()),
    }
    if any(audit.values()):
        raise RuntimeError(f"control funding join failed: {audit}")
    rescored = consumer.aggregate_event_funding(events, joined)
    scenarios = balanced.scenario_event_rows(rescored, MODES, BPS)
    return rescored, scenarios, audit


def reconstruct_controls(keys: pd.DataFrame, ledger: pd.DataFrame, primary_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for (cid, ctype), group in keys[keys.matched_candidate_id.astype(str).isin(primary_ids)].groupby(["matched_candidate_id", "control_type"], sort=True):
        sample = group.sort_values(["candidate_event_key", "control_candidate_definition_id"], kind="mergesort").iloc[0]
        shard = FULL / "aggregate_shards" / str(sample.control_shard_id)
        selected = pd.read_csv(shard / "selected_keys.csv")
        selected["decision_ts"] = pd.to_datetime(selected.decision_ts, utc=True, errors="coerce")
        match = selected[(selected.candidate_definition_id.astype(str) == str(sample.control_candidate_definition_id)) & (selected.symbol.astype(str) == str(sample.control_symbol)) & (selected.decision_ts == pd.Timestamp(sample.control_decision_ts)) & (selected.exit_policy_id.astype(str) == str(sample.control_exit_policy_id))]
        direct = ledger[(ledger.matched_candidate_id.astype(str) == str(cid)) & (ledger.control_type.astype(str) == str(ctype)) & (ledger.candidate_event_key.astype(str) == str(sample.candidate_event_key))]
        row = direct.iloc[0] if len(direct) else pd.Series(dtype=object)
        feature_ts = pd.to_datetime(match.feature_available_ts.iloc[0], utc=True, errors="coerce") if len(match) else pd.NaT
        decision = pd.Timestamp(sample.control_decision_ts)
        entry = pd.to_datetime(row.get("entry_ts"), utc=True, errors="coerce")
        atr_ts = pd.to_datetime(row.get("atr_feature_source_ts"), utc=True, errors="coerce")
        reconstructed_exit_policy = str(row.get("exit_policy_id", row.get("control_exit_policy_id", "")))
        mismatch = int(len(match) != 1 or len(direct) != 1 or feature_ts > decision or atr_ts > decision or entry <= decision or reconstructed_exit_policy != str(sample.control_exit_policy_id))
        samples.append({"candidate_definition_id": cid, "control_type": ctype, "candidate_event_key": sample.candidate_event_key, "control_definition_id": sample.control_candidate_definition_id, "control_symbol": sample.control_symbol, "control_decision_ts": decision, "feature_available_ts": feature_ts, "control_entry_ts": entry, "control_exit_ts": row.get("exit_ts", ""), "control_exit_policy_id": sample.control_exit_policy_id, "risk_price": row.get("risk_price", np.nan), "raw_fee_R": row.get("raw_fee_R", np.nan), "match_basis": sample.match_basis, "control_selection_uses_outcome": sample.control_selection_uses_outcome, "reconstruction_mismatch": mismatch})
        source_exit = direct["exit_policy_id"] if "exit_policy_id" in direct else direct["control_exit_policy_id"]
        audits.append({"candidate_definition_id": cid, "control_type": ctype, "control_rows": len(group), "decision_before_entry": bool((pd.to_datetime(direct.entry_ts, utc=True) > pd.to_datetime(direct.control_decision_ts, utc=True)).all()), "feature_available_lte_decision": bool(feature_ts <= decision), "atr_source_lte_decision": bool((pd.to_datetime(direct.atr_feature_source_ts, utc=True) <= pd.to_datetime(direct.control_decision_ts, utc=True)).all()), "same_exit_policy": bool((source_exit.astype(str) == direct.control_exit_policy_id.astype(str)).all()), "positive_r_denominator": bool((pd.to_numeric(direct.risk_price, errors="coerce") > 0).all()), "fees_bound": bool(pd.to_numeric(direct.raw_fee_R, errors="coerce").notna().all()), "funding_and_slippage_rescored_symmetrically": True, "outcome_informed_matching": bool(direct.control_selection_uses_outcome.astype(bool).any()), "holding_period_comparability": "same_exit_policy_and_time_stop_contract_realized_exit_may_differ", "status": "pass" if mismatch == 0 and not direct.control_selection_uses_outcome.astype(bool).any() else "fail"})
    return pd.DataFrame(audits), pd.DataFrame(samples)


def build_pairs(candidate: pd.DataFrame, control_scenarios: pd.DataFrame, ledger: pd.DataFrame, primary_ids: set[str]) -> pd.DataFrame:
    base = ledger[ledger.matched_candidate_id.astype(str).isin(primary_ids)][["candidate_event_key", "matched_candidate_id", "control_type", "event_id", "control_symbol", "control_decision_ts"]].copy()
    controls = control_scenarios[["event_id", "funding_mode", "slippage_round_trip_bps", "scenario_raw_net_R", "all_boundaries_exact", "exact_boundary_rows", "imputed_boundary_rows"]].rename(columns={"scenario_raw_net_R": "control_net_R", "all_boundaries_exact": "control_all_boundaries_exact", "exact_boundary_rows": "control_exact_boundary_rows", "imputed_boundary_rows": "control_imputed_boundary_rows"})
    candidates = candidate[["event_key", "candidate_definition_id", "symbol", "decision_ts", "funding_mode", "slippage_round_trip_bps", "scenario_raw_net_R", "all_boundaries_exact", "exact_boundary_rows", "imputed_boundary_rows", "parent_regime_state", "breadth_state", "universe_state", "funding_gate_availability"]].rename(columns={"event_key": "candidate_event_key", "scenario_raw_net_R": "candidate_net_R", "symbol": "candidate_symbol", "all_boundaries_exact": "candidate_all_boundaries_exact", "exact_boundary_rows": "candidate_exact_boundary_rows", "imputed_boundary_rows": "candidate_imputed_boundary_rows"})
    paired = base.merge(candidates, left_on=["candidate_event_key", "matched_candidate_id"], right_on=["candidate_event_key", "candidate_definition_id"], how="inner", validate="many_to_many")
    paired = paired.merge(controls, on=["event_id", "funding_mode", "slippage_round_trip_bps"], how="inner", validate="many_to_one")
    paired["paired_uplift_R"] = paired.candidate_net_R - paired.control_net_R
    paired["year_month"] = pd.to_datetime(paired.decision_ts, utc=True).dt.strftime("%Y-%m")
    paired["symbol_month"] = paired.candidate_symbol.astype(str) + "|" + paired.year_month
    return paired


def bootstrap_ci(group: pd.DataFrame, seed: int) -> tuple[float, float]:
    blocks = [g.paired_uplift_R.to_numpy() for _, g in group.groupby("symbol_month", sort=True)]
    if not blocks:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(250)
    for i in range(250):
        sample = [blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]
        means[i] = np.concatenate(sample).mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def symmetric_forensics(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    leave: list[dict[str, Any]] = []
    keys = ["matched_candidate_id", "control_type", "funding_mode", "slippage_round_trip_bps"]
    for values, g in paired.groupby(keys, sort=True):
        cid, ctype, mode, bps = values
        d = g.paired_uplift_R.sort_values(ascending=False).reset_index(drop=True)
        lo, hi = d.quantile(.01), d.quantile(.99)
        trim_n = max(1, math.ceil(len(d) * .01))
        ci_lo, ci_hi = bootstrap_ci(g, int(hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()[:8], 16))
        summaries.append({"candidate_definition_id": cid, "control_type": ctype, "funding_mode": mode, "slippage_round_trip_bps": bps, "paired_rows": len(g), "paired_mean_uplift_R": d.mean(), "paired_median_uplift_R": d.median(), "candidate_win_fraction": float((d > 0).mean()), "winsorized_mean_uplift_R": d.clip(lo, hi).mean(), "mean_after_top_1_difference_removal": d.iloc[1:].mean(), "mean_after_top_3_difference_removal": d.iloc[3:].mean(), "mean_after_top_1pct_difference_trim": d.iloc[trim_n:].mean(), "block_bootstrap_mean_ci_low": ci_lo, "block_bootstrap_mean_ci_high": ci_hi, "candidate_exact_boundary_rows": int(g.candidate_exact_boundary_rows.sum()), "control_exact_boundary_rows": int(g.control_exact_boundary_rows.sum()), "candidate_imputed_boundary_rows": int(g.candidate_imputed_boundary_rows.sum()), "control_imputed_boundary_rows": int(g.control_imputed_boundary_rows.sum())})
        for scope in ("candidate_symbol", "year_month", "symbol_month"):
            total = d.mean()
            for excluded, kept in g.groupby(scope, dropna=False):
                remaining = g[g[scope] != excluded]
                leave.append({"candidate_definition_id": cid, "control_type": ctype, "funding_mode": mode, "slippage_round_trip_bps": bps, "scope": scope, "excluded_value": excluded, "base_mean_uplift_R": total, "mean_uplift_after_exclusion": remaining.paired_uplift_R.mean(), "paired_rows_after_exclusion": len(remaining)})
    return pd.DataFrame(summaries), pd.DataFrame(leave)


def concentration(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    severe = paired[(paired.funding_mode == "severe_imputed") & (paired.slippage_round_trip_bps == 12)]
    for (cid, ctype), g in severe.groupby(["matched_candidate_id", "control_type"], sort=True):
        d = g.paired_uplift_R.sort_values(ascending=False)
        rows.append({"candidate_definition_id": cid, "control_type": ctype, "paired_rows": len(g), "base_mean_uplift_R": d.mean(), "mean_without_top_1": d.iloc[1:].mean(), "mean_without_top_3": d.iloc[3:].mean(), "mean_after_top_1pct_trim": d.iloc[max(1, math.ceil(len(d)*.01)):].mean(), "largest_positive_difference_R": d.iloc[0], "largest_negative_difference_R": d.iloc[-1]})
    return pd.DataFrame(rows)


def same_symbol_audits(paired: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, regime_rows = [], []
    ss = paired[paired.control_type == "same_symbol"]
    for (cid, mode, bps), cand in candidate.groupby(["candidate_definition_id", "funding_mode", "slippage_round_trip_bps"], sort=True):
        match = ss[(ss.matched_candidate_id == cid) & (ss.funding_mode == mode) & (ss.slippage_round_trip_bps == bps)]
        matched = set(match.candidate_event_key)
        yes = cand[cand.event_key.isin(matched)]; no = cand[~cand.event_key.isin(matched)]
        rows.append({"candidate_definition_id": cid, "funding_mode": mode, "slippage_round_trip_bps": bps, "candidate_events": len(cand), "matched_events": len(yes), "unmatched_events": len(no), "coverage": len(yes)/max(len(cand),1), "matched_candidate_mean_R": yes.scenario_raw_net_R.mean(), "unmatched_candidate_mean_R": no.scenario_raw_net_R.mean(), "coverage_bias_matched_minus_unmatched_R": yes.scenario_raw_net_R.mean()-no.scenario_raw_net_R.mean() if len(no) else np.nan})
    ss = ss.copy()
    context = targeted._parent_context_map(_CTX, ss.control_decision_ts)
    ss = ss.merge(context[["decision_ts", "parent_regime_state"]].rename(columns={"decision_ts": "control_decision_ts", "parent_regime_state": "control_parent_regime_state"}), on="control_decision_ts", how="left", validate="many_to_one")
    ssr = ss[ss.parent_regime_state == ss.control_parent_regime_state]
    for values, g in ssr.groupby(["matched_candidate_id", "funding_mode", "slippage_round_trip_bps"], sort=True):
        regime_rows.append({"candidate_definition_id": values[0], "funding_mode": values[1], "slippage_round_trip_bps": values[2], "paired_rows": len(g), "mean_uplift_R": g.paired_uplift_R.mean(), "median_uplift_R": g.paired_uplift_R.median(), "win_fraction": float((g.paired_uplift_R>0).mean()), "same_symbol_same_regime": True})
    return pd.DataFrame(rows), pd.DataFrame(regime_rows)


def simple_scorecard(control_scenarios: pd.DataFrame, ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = ledger[ledger.control_type.isin(["generic_breakout", "donchian_simple_breakout"])][["control_type", "event_id"]].drop_duplicates()
    tape = mapping.merge(control_scenarios, on="event_id", how="inner", validate="many_to_many")
    rows, library = [], []
    for (ctype, mode, bps), g in tape.groupby(["control_type", "funding_mode", "slippage_round_trip_bps"], sort=True):
        d = g.scenario_raw_net_R.sort_values(ascending=False)
        total = d.sum(); trim = d.iloc[max(1, math.ceil(len(d)*.01)):].sum()
        g = g.copy(); g["month"] = pd.to_datetime(g.entry_ts, utc=True).dt.strftime("%Y-%m"); g["symbol_month"] = g.symbol.astype(str)+"|"+g.month
        min_symbol = min((total-v for v in g.groupby("symbol").scenario_raw_net_R.sum()), default=np.nan)
        min_month = min((total-v for v in g.groupby("month").scenario_raw_net_R.sum()), default=np.nan)
        ci_lo, ci_hi = bootstrap_ci(g.rename(columns={"scenario_raw_net_R":"paired_uplift_R"}), int(hashlib.sha256(f"{ctype}|{mode}|{bps}".encode()).hexdigest()[:8],16))
        rows.append({"control_strategy": ctype, "funding_mode": mode, "slippage_round_trip_bps": bps, "deduplicated_events": g.event_id.nunique(), "mean_net_R": d.mean(), "median_net_R": d.median(), "total_net_R": total, "net_without_top_1": d.iloc[1:].sum(), "net_without_top_3": d.iloc[3:].sum(), "net_after_top_1pct_trim": trim, "minimum_leave_one_symbol_net_R": min_symbol, "minimum_leave_one_month_net_R": min_month, "block_bootstrap_mean_ci_low": ci_lo, "block_bootstrap_mean_ci_high": ci_hi, "tape_scope": "deduplicated_matched_control_tape_not_full_strategy_scan"})
    score = pd.DataFrame(rows)
    for ctype in ("generic_breakout", "donchian_simple_breakout"):
        x = score[(score.control_strategy==ctype)&(score.funding_mode=="severe_imputed")&(score.slippage_round_trip_bps==12)]
        robust = bool(len(x) and x.iloc[0].net_without_top_3>0 and x.iloc[0].net_after_top_1pct_trim>0 and x.iloc[0].minimum_leave_one_symbol_net_R>0 and x.iloc[0].minimum_leave_one_month_net_R>0 and x.iloc[0].block_bootstrap_mean_ci_low>0)
        library.append({"candidate_id": f"a1_control_{ctype}_matched_tape_v1", "candidate_type": "simpler_control_strategy", "decision": "advance_simpler_control_to_train_stability_review" if robust else "defer_current_translation", "evidence_label": "train_only_deduplicated_matched_control_tape_capped_not_validation", "pure_donchian_implementation": False if ctype.startswith("donchian") else "not_applicable", "robust_severe_12bps": robust})
    return score, pd.DataFrame(library)


def canonical_caps(shortlist: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for r in shortlist.itertuples(index=False):
        caps={"train_only_evidence_cap","funding_imputed_train_screen_cap","parent_basket_breadth_proxy_cap","nearest_neighbor_control_unavailable_cap"}
        if r.parent_regime_gate=="no_parent_gate_diagnostic": caps.add("no_parent_gate_diagnostic")
        if r.funding_gate=="no_funding_gate_diagnostic_cap": caps.add("no_funding_gate_diagnostic_cap")
        caps.add("oi_omitted_or_sidecar_cap")
        cov=coverage[(coverage.candidate_definition_id.astype(str)==str(r.candidate_definition_id))&(coverage.control_type=="same_symbol")]
        if len(cov) and float(cov.iloc[0].candidate_event_coverage)<.7: caps.add("low_same_symbol_control_coverage_cap")
        rows.append({"candidate_definition_id":r.candidate_definition_id,"canonical_active_caps":";".join(sorted(caps)),"superseded_labels_removed":"train_only_caps_pending_binding_audit;funding_proxy_selection_cap;funding_missing_adverse_proxy;base_no_slippage_event_ledger_requires_stress","status":"pass"})
    return pd.DataFrame(rows)


def decisions(shortlist: pd.DataFrame, sym: pd.DataFrame, leave: pd.DataFrame, same_regime: pd.DataFrame, simple_lib: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    severe=sym[(sym.funding_mode=="severe_imputed")&(sym.slippage_round_trip_bps==12)]
    rows=[]
    for r in shortlist.itertuples(index=False):
        cid=str(r.candidate_definition_id); g=severe[severe.candidate_definition_id==cid]
        robust=bool(len(g)==4 and (g.paired_mean_uplift_R>0).all() and (g.paired_median_uplift_R>0).all() and (g.winsorized_mean_uplift_R>0).all() and (g.mean_after_top_3_difference_removal>0).all() and (g.mean_after_top_1pct_difference_trim>0).all() and (g.block_bootstrap_mean_ci_low>0).all())
        sr=same_regime[(same_regime.candidate_definition_id==cid)&(same_regime.funding_mode=="severe_imputed")&(same_regime.slippage_round_trip_bps==12)]
        robust=robust and bool(len(sr) and sr.mean_uplift_R.iloc[0]>0 and sr.median_uplift_R.iloc[0]>0)
        diagnostic=r.parent_regime_gate=="no_parent_gate_diagnostic" or r.funding_gate=="no_funding_gate_diagnostic_cap"
        if diagnostic: decision="diagnostic_only"
        elif robust: decision="advance_candidate_to_train_stability_review"
        elif r.definition_lane=="h06_vcp_like_contraction" and len(g) and (g.paired_mean_uplift_R>0).sum()>=3: decision="preserve_as_context_sleeve"
        else: decision="defer_current_translation"
        rows.append({"candidate_id":cid,"candidate_type":"a1_primary","definition_lane":r.definition_lane,"special_review":cid in SPECIAL,"decision":decision,"robust_control_classes_passed":int((g.paired_mean_uplift_R>0).sum()),"required_control_classes":4,"exact_only_rows":int(sym[(sym.candidate_definition_id==cid)&(sym.funding_mode=="exact_only_slice")].paired_rows.sum()),"canonical_active_caps":caps.loc[caps.candidate_definition_id==cid,"canonical_active_caps"].iloc[0],"evidence_label":"train_only_symmetric_control_adjudication_capped_not_validation"})
    return pd.concat([pd.DataFrame(rows),simple_lib.rename(columns={"candidate_id":"candidate_id"})],ignore_index=True,sort=False)


def compact(root: Path, required: list[str]) -> None:
    rows=[]
    for rel in required:
        src=root/rel; dst=root/"compact_review_bundle"/rel.replace("/","__"); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        rows.append({"source":rel,"bundle_path":str(dst.relative_to(root)),"sha256":runner.sha256_file(dst)})
    csv(root/"compact_review_bundle/compact_bundle_manifest.csv",rows)


_CTX: runner.Context


def main() -> int:
    global _CTX
    ap=argparse.ArgumentParser(); ap.add_argument("--run-root",default=str(DEFAULT_ROOT)); ap.add_argument("--disable-telegram",action="store_true"); args=ap.parse_args()
    root=Path(args.run_root)
    if root.exists() and any(root.iterdir()): raise RuntimeError(f"run root not fresh: {root}")
    rargs=runner.parse_args(["--phase-profile",runner.A1_COMPRESSION_TARGETED_MATERIALIZATION_CONTROLS_PHASE_PROFILE,"--run-root",str(root),"--start","2024-01-01","--end","2025-12-31",*( ["--disable-telegram"] if args.disable_telegram else [])]); _CTX=runner.init_context(rargs)
    candidate, shortlist=load_candidate_scenarios(); primary_ids=set(shortlist.candidate_definition_id.astype(str))
    keys=pd.read_parquet(SOURCE/"controls/control_ledgers/control_key_manifest.parquet"); ledger=pd.read_parquet(SOURCE/"controls/control_ledgers/control_ledger.parquet")
    sem,recon=reconstruct_controls(keys,ledger,primary_ids); csv(root/"controls/control_semantics_audit.csv",sem); csv(root/"controls/control_reconstruction_sample.csv",recon)
    if int(recon.reconstruction_mismatch.sum()) or not sem.status.eq("pass").all(): raise RuntimeError("control reconstruction mismatch")
    rescored_ctrl,ctrl_scen,fjoin=rescore_controls(ledger)
    pairs=build_pairs(candidate,ctrl_scen,ledger,primary_ids); sym,leave=symmetric_forensics(pairs); winner=concentration(pairs)
    csv(root/"controls/symmetric_paired_forensics.csv",sym); csv(root/"controls/control_winner_concentration.csv",winner); csv(root/"controls/control_leave_one_out.csv",leave); csv(root/"controls/funding_slippage_control_comparison.csv",sym)
    bias,ssr=same_symbol_audits(pairs,candidate); csv(root/"controls/same_symbol_coverage_bias.csv",bias); csv(root/"controls/same_symbol_same_regime_comparison.csv",ssr)
    (root/"controls/nearest_neighbor_status.md").parent.mkdir(parents=True,exist_ok=True); (root/"controls/nearest_neighbor_status.md").write_text("# Nearest-neighbor status\n\nBlocked. The frozen decision-time matrix lacks a complete comparable feature vector across candidate and control pools. No substitute or outcome-derived distance was used.\n")
    score,simple_lib=simple_scorecard(ctrl_scen,ledger); csv(root/"controls/simple_breakout_candidate_scorecard.csv",score)
    severe_pairs=pairs[(pairs.funding_mode=="severe_imputed")&(pairs.slippage_round_trip_bps==12)].copy(); severe_pairs["period_scope"]=np.select([pd.to_datetime(severe_pairs.decision_ts,utc=True).dt.year.eq(2024),pd.to_datetime(severe_pairs.decision_ts,utc=True)<pd.Timestamp("2025-07-01",tz="UTC")],["2024","2025_h1"],default="2025_h2")
    context=severe_pairs.groupby(["matched_candidate_id","control_type","period_scope","parent_regime_state","breadth_state","universe_state","funding_gate_availability"],dropna=False).agg(paired_rows=("paired_uplift_R","size"),mean_uplift_R=("paired_uplift_R","mean"),median_uplift_R=("paired_uplift_R","median"),win_fraction=("paired_uplift_R",lambda s:float((s>0).mean()))).reset_index().rename(columns={"matched_candidate_id":"candidate_definition_id"}); csv(root/"regime/candidate_control_context_uplift.csv",context)
    coverage=pd.read_csv(SOURCE/"controls/control_match_coverage.csv"); caps=canonical_caps(shortlist,coverage); csv(root/"caps/canonical_active_cap_table.csv",caps)
    decision=decisions(shortlist,sym,leave,ssr,simple_lib,caps); csv(root/"decision/candidate_and_control_decision_table.csv",decision); csv(root/"candidate_library/candidate_library_update.csv",decision)
    protected=int((pd.to_datetime(ledger.exit_interval_end_ts,utc=True)>=runner.PROTECTED_TS).sum()); leaks=int((pd.to_datetime(recon.feature_available_ts,utc=True)>pd.to_datetime(recon.control_decision_ts,utc=True)).sum())
    summary={"run_root":str(root),"status":"complete","control_semantics_pass":True,"control_reconstruction_mismatches":0,"controls_reconstructed":len(recon),"primary_candidates":13,"funding_missing_joins":fjoin["missing"],"funding_duplicate_joins":fjoin["duplicate"],"protected_period_violations":protected,"decision_input_leaks":leaks,"validation_launched":False,"final_holdout_touched":False,"candidates_beating_all_symmetric_robust_controls":decision[(decision.candidate_type=="a1_primary")&(decision.decision=="advance_candidate_to_train_stability_review")].candidate_id.astype(str).tolist(),"simpler_controls_advanced":decision[decision.decision=="advance_simpler_control_to_train_stability_review"].candidate_id.astype(str).tolist(),"context_sleeves_preserved":decision[decision.decision=="preserve_as_context_sleeve"].candidate_id.astype(str).tolist(),"deferred_candidates":decision[decision.decision=="defer_current_translation"].candidate_id.astype(str).tolist(),"diagnostic_candidates":decision[decision.decision=="diagnostic_only"].candidate_id.astype(str).tolist(),"compact_bundle_path":str(root/"compact_review_bundle"),"evidence_label":"train_only_control_context_adjudication_capped_not_validation"}
    runner.write_json(root/"decision_summary.json",summary)
    required=["controls/control_semantics_audit.csv","controls/control_reconstruction_sample.csv","controls/symmetric_paired_forensics.csv","controls/control_winner_concentration.csv","controls/control_leave_one_out.csv","controls/funding_slippage_control_comparison.csv","controls/same_symbol_coverage_bias.csv","controls/same_symbol_same_regime_comparison.csv","controls/nearest_neighbor_status.md","controls/simple_breakout_candidate_scorecard.csv","regime/candidate_control_context_uplift.csv","caps/canonical_active_cap_table.csv","decision/candidate_and_control_decision_table.csv","candidate_library/candidate_library_update.csv","decision_summary.json"]
    compact(root,required); return 0


if __name__=="__main__": raise SystemExit(main())
