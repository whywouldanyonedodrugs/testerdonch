#!/usr/bin/env python3
"""Frozen backside-blowoff replay under signal_state_contract_v1_20260715."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools import qlmg_evidence_contracts as evidence
from tools import qlmg_signal_state_contract as state
from tools import run_kraken_backside_blowoff_short_screen as base


RUN_ROOT = Path("results/rebaseline/phase_kraken_backside_blowoff_signal_state_repaired_screen_20260715_v1")
SOURCE_ROOT = Path("results/rebaseline/phase_kraken_backside_blowoff_short_screen_20260713_v1")
CAMPAIGN_ROOT = Path("results/rebaseline/phase_kraken_cross_family_signal_state_repair_campaign_20260715_v1")
CONTRACT_VERSION = state.SIGNAL_STATE_CONTRACT_VERSION
PRESERVED_IDS = ("bcbs_v1_002", "bcbs_v1_008")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_hash(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append((str(path.relative_to(root)), file_hash(path)))
    return state.stable_hash(rows)


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def raw_specs(definitions: pd.DataFrame) -> list[dict[str, Any]]:
    rows = definitions[["extension_profile", "confirmation_bars"]].drop_duplicates().sort_values(["extension_profile", "confirmation_bars"])
    return [{**row, "raw_policy_hash": base.raw_policy_hash(row)} for row in rows.to_dict("records")]


def economic_address(row: Mapping[str, Any]) -> str:
    return base.candidate_address(row)


def address_map(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    old_ids = set(zip(old.definition_id.astype(str), old.candidate_economic_address_hash.astype(str))) if len(old) else set()
    new_ids = set(zip(new.definition_id.astype(str), new.candidate_economic_address_hash.astype(str))) if len(new) else set()
    rows = []
    for definition_id, address in sorted(old_ids | new_ids):
        rows.append({
            "definition_id": definition_id,
            "candidate_economic_address_hash": address,
            "old_rows": int(((old.definition_id.astype(str) == definition_id) & (old.candidate_economic_address_hash.astype(str) == address)).sum()) if len(old) else 0,
            "new_rows": int(((new.definition_id.astype(str) == definition_id) & (new.candidate_economic_address_hash.astype(str) == address)).sum()) if len(new) else 0,
            "address_status": "unchanged_address" if (definition_id, address) in old_ids & new_ids else "removed_address" if (definition_id, address) in old_ids else "restored_or_new_address",
        })
    return pd.DataFrame(rows)


def matched_unmatched(events: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if controls.empty:
        return pd.DataFrame(rows)
    for (definition_id, control_class), group in controls.groupby(["definition_id", "control_class"]):
        unique = group.sort_values(["control_economic_address_hash", "candidate_key"]).drop_duplicates("control_economic_address_hash")
        full = events[events.definition_id.eq(definition_id)]
        matched_keys = set(unique.candidate_key)
        matched = full[full.candidate_key.isin(matched_keys)]
        unmatched = full[~full.candidate_key.isin(matched_keys)]
        for mode in ("base", "conservative", "severe"):
            rows.append({
                "definition_id": definition_id, "control_class": control_class, "cost_mode": mode,
                "full_count": len(full), "matched_count": len(matched), "unmatched_count": len(unmatched),
                "full_candidate_mean_R": full[f"net_{mode}_R"].mean(),
                "matched_candidate_mean_R": matched[f"net_{mode}_R"].mean(),
                "unmatched_candidate_mean_R": unmatched[f"net_{mode}_R"].mean(),
                "matched_minus_unmatched_R": matched[f"net_{mode}_R"].mean() - unmatched[f"net_{mode}_R"].mean(),
            })
    return pd.DataFrame(rows)


def leave_one_reports(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_rows, month_rows = [], []
    work = events.copy()
    work["year_month"] = pd.to_datetime(work.entry_ts, utc=True).dt.strftime("%Y-%m")
    for definition_id, definition_rows in work.groupby("definition_id"):
        for mode in ("base", "conservative", "severe"):
            column = f"net_{mode}_R"
            for symbol in sorted(definition_rows.symbol.unique()):
                remaining = definition_rows[definition_rows.symbol.ne(symbol)]
                symbol_rows.append({"definition_id": definition_id, "cost_mode": mode, "omitted_symbol": symbol, "events_remaining": len(remaining), "mean_R": remaining[column].mean(), "total_R": remaining[column].sum()})
            for month in sorted(definition_rows.year_month.unique()):
                remaining = definition_rows[definition_rows.year_month.ne(month)]
                month_rows.append({"definition_id": definition_id, "cost_mode": mode, "omitted_month": month, "events_remaining": len(remaining), "mean_R": remaining[column].mean(), "total_R": remaining[column].sum()})
    return pd.DataFrame(symbol_rows), pd.DataFrame(month_rows)


def compact_bundle(root: Path) -> Path:
    files = (
        "BACKSIDE_REPAIRED_RECONCILIATION_REPORT.md", "decision_summary.json", "contract/backside_signal_state_contract.md",
        "manifest/backside_blowoff_short_definitions.csv", "reproducibility/run_manifest.json",
        "audit/rankable_signal_state_contract.json", "audit/raw_parent_policy_nesting.csv", "audit/deterministic_replay_audit.csv",
        "audit/non_overlap_skip_ledger.csv", "audit/outcome_exclusion_ledger.csv", "audit/old_new_signal_trade_counts.csv",
        "audit/old_to_new_address_map.csv", "economics/definition_summary.csv", "economics/period_summary.csv",
        "forensics/concentration_and_removal.csv", "forensics/leave_one_symbol.csv", "forensics/leave_one_month.csv",
        "forensics/exact_vs_imputed_funding.csv", "forensics/exit_policy_comparison.csv", "forensics/parameter_neighborhood.csv",
        "controls/control_summary.csv", "controls/matched_unmatched_complements.csv", "controls/parabolic_extension_control_comparison.csv",
        "decision/candidate_decisions.csv", "decision/preserved_definition_replay.csv", "candidate_library/backside_candidate_library_update.csv",
    )
    temp = root / ".compact_review_bundle.tmp"
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True)
    inventory = []
    for relative in files:
        source = root / relative
        target = temp / relative.replace("/", "__")
        shutil.copy2(source, target)
        inventory.append({"source_path": relative, "bundle_path": target.name, "sha256": file_hash(source), "bytes": source.stat().st_size})
    write_csv(temp / "bundle_manifest.csv", inventory)
    final = root / "compact_review_bundle"
    if final.exists(): shutil.rmtree(final)
    os.replace(temp, final)
    return final


def refresh_campaign_bundle() -> None:
    bundle = CAMPAIGN_ROOT / "compact_review_bundle"
    manifest = pd.read_csv(bundle / "bundle_manifest.csv")
    files = manifest.source_path.astype(str).tolist()
    temp = CAMPAIGN_ROOT / ".compact_review_bundle.refresh.tmp"
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir()
    inventory = []
    for relative in files:
        source = CAMPAIGN_ROOT / relative
        target = temp / relative.replace("/", "__")
        shutil.copy2(source, target)
        inventory.append({"source_path": relative, "bundle_path": target.name, "sha256": file_hash(source), "bytes": source.stat().st_size})
    write_csv(temp / "bundle_manifest.csv", inventory)
    shutil.rmtree(bundle)
    os.replace(temp, bundle)


def update_campaign(root: Path, classification: str) -> None:
    registry_path = CAMPAIGN_ROOT / "campaign/affected_run_registry.csv"
    registry = pd.read_csv(registry_path)
    mask = registry.family.eq("Backside blowoff")
    registry.loc[mask, "required_rerun"] = False
    registry.loc[mask, "required_downstream_replay"] = False
    registry.loc[mask, "repaired_root_placeholder"] = str(root)
    registry.loc[mask, "candidate_library_status"] = f"repaired_evidence_{classification}"
    registry.loc[mask, "closure_state"] = "closed"
    write_csv(registry_path, registry)

    dependency_path = CAMPAIGN_ROOT / "campaign/downstream_dependency_map.csv"
    dependency = pd.read_csv(dependency_path)
    dependency = dependency.drop(columns=["status"], errors="ignore")
    affected = dependency.upstream_root.eq(str(SOURCE_ROOT))
    dependency.loc[affected, "replay_status"] = "upstream_repaired_no_identity_bearing_replay_required"
    write_csv(dependency_path, dependency)

    preservation_path = CAMPAIGN_ROOT / "campaign/hypothesis_preservation_ledger.csv"
    preservation = pd.read_csv(preservation_path)
    preservation = preservation.drop(columns=["next_allowed_action"], errors="ignore")
    row = preservation.family.eq("Backside blowoff")
    preservation.loc[row, "preservation_status"] = "repaired_lineage_closed_not_rejected"
    preservation.loc[row, "permitted_claim"] = "repaired train-only classification is decision-bearing subject to existing caps"
    preservation.loc[row, "forbidden_claim"] = "prior quarantined result is decision-bearing"
    write_csv(preservation_path, preservation)

    matrix_path = CAMPAIGN_ROOT / "campaign/repair_completion_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    matrix.loc[matrix.family.eq("Shared contract"), "next_action"] = "apply frozen contract to RFBS screen and downstream materialization only"
    row = matrix.family.eq("Backside blowoff")
    for column in ("runner_migrated", "regression_tests_passed", "repaired_replay_complete", "downstream_replay_complete", "reconciled_and_closed"):
        matrix.loc[row, column] = True
    matrix.loc[row, "next_action"] = "none; lineage closed"
    matrix.loc[matrix.family.eq("RFBS"), "priority"] = 1
    matrix.loc[matrix.family.eq("RFBS"), "next_action"] = "next repair target: screen and downstream materialization only"
    write_csv(matrix_path, matrix)

    gate_path = CAMPAIGN_ROOT / "campaign/new_family_launch_gate.json"
    gate = json.loads(gate_path.read_text())
    gate.update({"new_family_launch_allowed": False, "unresolved_registry_count": 2, "directly_affected_roots": 1, "downstream_affected_roots": 1, "next_prompt_target": "RFBS screen and downstream materialization only", "backside_lineage_closed": True})
    write_json(gate_path, gate)
    decision_path = CAMPAIGN_ROOT / "decision_summary.json"
    decision = json.loads(decision_path.read_text())
    decision.update({"directly_affected_roots": 1, "downstream_affected_roots": 1, "unresolved_registry_count": 2, "next_recommended_prompt_target": "RFBS screen and downstream materialization only", "backside_lineage_closed": True})
    write_json(decision_path, decision)
    refresh_campaign_bundle()


def run(root: Path, *, resume: bool = False) -> dict[str, Any]:
    if root.exists() and not resume:
        raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True, exist_ok=resume)
    started = time.monotonic()
    peak_rss = 0
    source_hash_before = directory_hash(SOURCE_ROOT)
    definitions = base.frozen_manifest()
    source_definitions = pd.read_csv(SOURCE_ROOT / "manifest/backside_blowoff_short_definitions.csv")
    pd.testing.assert_frame_equal(definitions.reset_index(drop=True), source_definitions.reset_index(drop=True), check_dtype=False)
    write_csv(root / "manifest/backside_blowoff_short_definitions.csv", definitions)
    contract = f"""# Backside Blowoff Signal-State Contract\n\nVersion: `{CONTRACT_VERSION}`. The frozen 24 definitions, extension profiles, confirmation windows, parent policies, entries, stops, exits, universe, costs, funding, and controls are unchanged. Four parent-neutral raw tapes preserve pending-sequence, higher-high reset, and confirmation-expiry semantics. Parent policies are PIT projections. Non-overlap is definition-local and uses actual executable exits.\n"""
    (root / "contract").mkdir(exist_ok=True)
    (root / "contract/backside_signal_state_contract.md").write_text(contract)
    ctx = base.context(root)
    panel = base.runner.full_panel_for_launch_gate(ctx)
    write_csv(root / "manifest/pit_panel.csv", panel)
    paths = base.runner.data_paths(ctx)

    raw_rows: list[dict[str, Any]] = []
    for number, symbol in enumerate(panel.symbol.astype(str), 1):
        shard = root / "signals/raw_symbol_shards" / f"{symbol}.parquet"
        if resume and shard.exists():
            frame = pd.read_parquet(shard)
            raw_rows.extend(frame.to_dict("records"))
        else:
            bars = base.runner.load_symbol_bars(paths, symbol, base.START - pd.Timedelta(days=100), base.END)
            symbol_rows: list[dict[str, Any]] = []
            if not bars.empty:
                feature, work = base.feature_frames(bars)
                for spec in raw_specs(definitions):
                    symbol_rows.extend(base.enumerate_raw_signals(ctx, panel, symbol, bars, spec, frame=feature, work=work))
            shard.parent.mkdir(parents=True, exist_ok=True)
            temporary = shard.with_suffix(".tmp.parquet")
            base.runner.parquet_safe_frame(pd.DataFrame(symbol_rows)).to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, shard)
            raw_rows.extend(symbol_rows)
        peak_rss = max(peak_rss, base.runner.current_rss_bytes())
        write_json(root / "watch_status.json", {"status": "running", "stage": "parent_neutral_raw_signal_build", "symbols_completed": number, "symbols_planned": len(panel), "raw_signals": len(raw_rows), "rss_bytes": peak_rss, "elapsed_seconds": time.monotonic()-started, "updated_ts": base.runner.utc_now()})
    raw = pd.DataFrame(raw_rows)
    if raw.empty: raise RuntimeError("no parent-neutral backside signals")
    raw = state.suppress_repeated_unresolved_setups(raw, setup_id_col="setup_sequence_id", order_fields=("raw_policy_hash", "symbol", "decision_ts", "raw_signal_address_hash"))
    raw, raw_hash = state.freeze_raw_signal_tape(raw)
    write_csv(root / "signals/raw_signal_manifest.csv", raw)

    policies = definitions.drop_duplicates("selected_key_policy_hash").to_dict("records")
    projected, projection_hash = state.project_parent_policies(raw, policies, feature_ts_col="parent_feature_ts", is_allowed=lambda source, policy: (
        str(source["extension_profile"]) == str(policy["extension_profile"])
        and int(source["confirmation_bars"]) == int(policy["confirmation_bars"])
        and (policy["parent_context"] == "all_regime_comparator" or source["parent_state"] == "both_down")
    ))
    projected["selected_key_frozen"] = True
    write_csv(root / "keys/candidate_key_manifest.csv", projected)
    write_csv(root / "signals/parent_policy_projection_manifest.csv", projected)
    nesting = []
    unique_policies = definitions.drop_duplicates("selected_key_policy_hash")
    for (extension, confirmation), group in unique_policies.groupby(["extension_profile", "confirmation_bars"]):
        strict_hash = group[group.parent_context.eq("fragile_countertrend_stress")].selected_key_policy_hash.iloc[0]
        broad_hash = group[group.parent_context.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
        strict = set(projected[projected.selected_key_policy_hash.eq(strict_hash)].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq(broad_hash)].raw_signal_address_hash)
        nesting.append({"extension_profile": extension, "confirmation_bars": confirmation, "strict_rows": len(strict), "broad_rows": len(broad), "strict_not_in_broad": len(strict-broad), "pass": not bool(strict-broad)})
    nesting = pd.DataFrame(nesting)
    write_csv(root / "audit/raw_parent_policy_nesting.csv", nesting)

    bars_cache, feature_cache = {}, {}
    for symbol in sorted(projected.symbol.unique()):
        bars = base.runner.load_symbol_bars(paths, symbol, base.START-pd.Timedelta(days=100), base.END)
        bars_cache[symbol] = bars
        feature_cache[symbol] = base.feature_frames(bars)[0]
    accepted_frames, skip_frames, exclusion_frames, parity_rows = [], [], [], []
    eligible_rows = 0
    for number, definition in enumerate(definitions.to_dict("records"), 1):
        selected = projected[projected.selected_key_policy_hash.eq(definition["selected_key_policy_hash"])]
        eligible_rows += len(selected)
        def execute(key: Mapping[str, Any], definition_row: Mapping[str, Any]):
            event, exclusion = base.execute_event(key, definition_row["exit_policy"], bars_cache[key["symbol"]], feature_cache[key["symbol"]])
            if event is not None:
                event["parameter_vector_hash"] = definition_row["parameter_vector_hash"]
            return event, exclusion
        first = state.simulate_definition_non_overlap(selected, definition, execute)
        second = state.simulate_definition_non_overlap(selected.sample(frac=1, random_state=19), definition, execute)
        first_hash = state.canonical_frame_hash(first[0], sort_fields=("definition_id", "symbol", "entry_ts", "candidate_key")) if len(first[0]) else state.stable_hash([])
        second_hash = state.canonical_frame_hash(second[0], sort_fields=("definition_id", "symbol", "entry_ts", "candidate_key")) if len(second[0]) else state.stable_hash([])
        parity_rows.append({"definition_id": definition["definition_id"], "selected_key_policy_hash": definition["selected_key_policy_hash"], "first_rows": len(first[0]), "second_rows": len(second[0]), "first_hash": first_hash, "second_hash": second_hash, "mismatch": first_hash != second_hash})
        if len(first[0]): accepted_frames.append(first[0])
        if len(first[1]): skip_frames.append(first[1])
        if len(first[2]): exclusion_frames.append(first[2])
        peak_rss = max(peak_rss, base.runner.current_rss_bytes())
        write_json(root / "watch_status.json", {"status": "running", "stage": "definition_local_actual_exit_non_overlap", "definitions_completed": number, "definitions_planned": 24, "eligible_definition_rows": eligible_rows, "accepted_rows": sum(len(x) for x in accepted_frames), "overlap_skips": sum(len(x) for x in skip_frames), "rss_bytes": peak_rss, "elapsed_seconds": time.monotonic()-started, "updated_ts": base.runner.utc_now()})
    accepted = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    skips = pd.concat(skip_frames, ignore_index=True) if skip_frames else pd.DataFrame()
    exclusions = pd.concat(exclusion_frames, ignore_index=True) if exclusion_frames else pd.DataFrame()
    parity = pd.DataFrame(parity_rows)
    if accepted.empty: raise RuntimeError("no accepted backside trades")
    accepted["candidate_economic_address_hash"] = [economic_address(row) for row in accepted.to_dict("records")]
    contract_manifest = state.build_rankable_contract_manifest(raw_signals=raw, projected=projected, accepted=accepted, skips=skips, exclusions=exclusions, eligible_definition_rows=eligible_rows)
    evidence.assert_rankable_signal_state_contract(contract_manifest)
    accepted["accepted_trade_hash"] = contract_manifest["accepted_trade_hash"]
    write_json(root / "audit/rankable_signal_state_contract.json", contract_manifest)
    write_csv(root / "audit/non_overlap_skip_ledger.csv", skips)
    write_csv(root / "audit/outcome_exclusion_ledger.csv", exclusions)
    write_csv(root / "audit/deterministic_replay_audit.csv", parity)
    write_csv(root / "materialized/accepted_trade_identity_manifest.csv", accepted)

    funding = base.lfbs.funding_panel()
    outcomes, boundaries = base.attach_costs(accepted, funding, "event_id")
    write_csv(root / "materialized/event_ledger.csv", outcomes)
    accepted_candidates = projected[projected.candidate_key.isin(set(outcomes.candidate_key))].drop_duplicates("candidate_key")
    control_keys, unavailable = base.build_control_keys(accepted_candidates, outcomes, feature_cache, bars_cache, panel, ctx)
    control_freeze = state.canonical_frame_hash(control_keys, sort_fields=("definition_id", "candidate_key", "control_key")) if len(control_keys) else state.stable_hash([])
    control_keys["control_key_freeze_hash"] = control_freeze
    write_csv(root / "controls/control_key_manifest.csv", control_keys)
    write_csv(root / "controls/control_unavailable_reasons.csv", unavailable)
    control_rows, control_exclusions = [], []
    for control in control_keys.to_dict("records"):
        event, exclusion = base.execute_event(control, control["exit_policy"], bars_cache[control["symbol"]], feature_cache[control["symbol"]])
        if exclusion is not None:
            control_exclusions.append({**exclusion, "control_key": control["control_key"]})
            continue
        event.update({"control_event_id": control["control_key"], "candidate_key": control["candidate_key"], "definition_id": control["definition_id"], "control_class": control["control_class"], "control_economic_address_hash": control["control_economic_address_hash"]})
        control_rows.append(event)
    controls = pd.DataFrame(control_rows)
    if len(controls): controls, control_boundaries = base.attach_costs(controls, funding, "control_event_id")
    else: control_boundaries = pd.DataFrame()
    write_csv(root / "controls/control_event_ledger.csv", controls)
    write_csv(root / "audit/control_outcome_exclusion_ledger.csv", control_exclusions)
    address_audit, control_summary = base.controls_report(outcomes, controls)
    complements = matched_unmatched(outcomes, controls)
    write_csv(root / "controls/control_address_audit.csv", address_audit)
    write_csv(root / "controls/control_summary.csv", control_summary)
    write_csv(root / "controls/matched_unmatched_complements.csv", complements)
    structural = control_summary[control_summary.control_class.eq("parabolic_extension_without_backside_confirmation")]
    write_csv(root / "controls/parabolic_extension_control_comparison.csv", structural)

    summary, attribution, period = base.summarize_economics(outcomes, definitions)
    concentration, neighborhood, exits = base.forensics(outcomes, definitions)
    funding_report = base.funding_partition_report(outcomes)
    leave_symbol, leave_month = leave_one_reports(outcomes)
    decisions = base.decision_table(summary, concentration, control_summary, period, definitions)
    write_csv(root / "economics/definition_summary.csv", summary)
    write_csv(root / "economics/cost_funding_attribution.csv", attribution)
    write_csv(root / "economics/period_summary.csv", period)
    write_csv(root / "forensics/concentration_and_removal.csv", concentration)
    write_csv(root / "forensics/parameter_neighborhood.csv", neighborhood)
    write_csv(root / "forensics/exit_policy_comparison.csv", exits)
    write_csv(root / "forensics/exact_vs_imputed_funding.csv", funding_report)
    write_csv(root / "forensics/leave_one_symbol.csv", leave_symbol)
    write_csv(root / "forensics/leave_one_month.csv", leave_month)
    write_csv(root / "decision/candidate_decisions.csv", decisions)

    old = pd.read_csv(SOURCE_ROOT / "materialized/event_ledger.csv")
    mapping = address_map(old, outcomes)
    write_csv(root / "audit/old_to_new_address_map.csv", mapping)
    counts = [
        {"stage": "raw_signals", "old_count": np.nan, "new_count": len(raw), "old_count_status": "not_recorded_in_legacy_root"},
        {"stage": "parent_filtered_selected_keys", "old_count": pd.read_csv(SOURCE_ROOT / "keys/candidate_key_manifest.csv").candidate_key.nunique(), "new_count": len(projected), "old_count_status": "recorded"},
        {"stage": "accepted_definition_trades", "old_count": len(old), "new_count": len(outcomes), "old_count_status": "recorded"},
        {"stage": "actual_overlap_skips", "old_count": np.nan, "new_count": len(skips), "old_count_status": "legacy_preblock_not_auditable_as_actual_overlap"},
    ]
    write_csv(root / "audit/old_new_signal_trade_counts.csv", counts)
    preserved = summary[summary.definition_id.isin(PRESERVED_IDS)].merge(pd.read_csv(SOURCE_ROOT / "economics/definition_summary.csv"), on=["definition_id", "cost_mode"], suffixes=("_repaired", "_old"))
    write_csv(root / "decision/preserved_definition_replay.csv", preserved)

    interval_violations = []
    for label, (window_start, window_end) in base.EVALUATION_WINDOWS.items():
        interval_violations.extend(evidence.validate_evaluation_window_intervals(outcomes[outcomes.evaluation_period.eq(label)], window_start=window_start, window_end=window_end).violations)
    manifest_mismatch = int(file_hash(root / "manifest/backside_blowoff_short_definitions.csv") != file_hash(SOURCE_ROOT / "manifest/backside_blowoff_short_definitions.csv"))
    source_mutation = int(directory_hash(SOURCE_ROOT) != source_hash_before)
    hard = {
        "definitions_evaluated": int(summary.definition_id.nunique()), "raw_policy_hashes": int(raw.raw_policy_hash.nunique()), "selected_key_policy_hashes": int(projected.selected_key_policy_hash.nunique()),
        "frozen_manifest_hash_mismatches": manifest_mismatch, "source_root_content_mutations": source_mutation,
        "raw_signal_duplicates": int(raw.duplicated("raw_signal_address_hash").sum()), "parent_nesting_failures": int((~nesting["pass"]).sum()),
        "deterministic_replay_mismatches": int(parity.mismatch.sum()), "candidate_duplicate_economic_addresses": int(outcomes.duplicated(["definition_id", "candidate_economic_address_hash"]).sum()),
        "unexplained_attrition": int(eligible_rows != len(outcomes)+len(skips)+len(exclusions)), "artificial_horizon_exits": int(outcomes.artificial_horizon_exit.sum()),
        "evaluation_interval_contract_violations": len(interval_violations), "funding_join_missing": int(boundaries.missing.sum()) if len(boundaries) else 0,
        "funding_join_duplicates": int(boundaries.duplicated(["event_id", "boundary_ts"]).sum()) if len(boundaries) else 0,
        "control_funding_join_missing": int(control_boundaries.missing.sum()) if len(control_boundaries) else 0,
        "control_funding_join_duplicates": int(control_boundaries.duplicated(["control_event_id", "boundary_ts"]).sum()) if len(control_boundaries) else 0,
        "decision_input_leaks": int((pd.to_datetime(projected.feature_available_ts, utc=True)>pd.to_datetime(projected.decision_ts, utc=True)).sum()) + (int((pd.to_datetime(control_keys.feature_available_ts, utc=True)>pd.to_datetime(control_keys.decision_ts, utc=True)).sum()) if len(control_keys) else 0),
        "protected_period_violations": int(outcomes.protected_violation.sum()), "control_outcomes_accessed_before_freeze": int(control_keys.outcome_accessed_before_freeze.sum()) if len(control_keys) else 0,
        "placeholder_controls": int(control_keys.placeholder_control.sum()) if len(control_keys) else 0,
        "duplicate_control_addresses_counted_independently": int(address_audit.duplicated_addresses_counted_independently.sum()) if len(address_audit) else 0,
        "matched_unmatched_complement_failures": int(((complements.matched_count+complements.unmatched_count)!=complements.full_count).sum()) if len(complements) else 0,
        "preserved_definition_replay_missing": len(set(PRESERVED_IDS)-set(summary.definition_id)),
    }
    expected = {"definitions_evaluated": 24, "raw_policy_hashes": 4, "selected_key_policy_hashes": 8}
    gate_pass = all(value == expected.get(key, 0) for key, value in hard.items())
    write_csv(root / "audit/hard_gate_audit.csv", [{"gate": key, "value": value, "pass": value == expected.get(key, 0)} for key, value in hard.items()])
    final = "focused_mechanical_repair_required" if not gate_pass else "materialization_candidate" if decisions.decision.eq("materialization_candidate").any() else "fragile_context_sleeve" if decisions.decision.eq("context_sleeve_candidate").any() else "current_translation_weak"

    library = []
    for definition in definitions.itertuples(index=False):
        row = decisions[decisions.definition_id.eq(definition.definition_id)].iloc[0]
        library.append({
            "candidate_id": definition.definition_id, "candidate_definition_id": definition.definition_id, "definition_id": definition.definition_id,
            "hypothesis_id": "backside_confirmed_blowoff_short", "family_engine_id": "kraken_bcbs_v1_signal_state_repaired",
            "parameter_vector_hash": definition.parameter_vector_hash, "selected_key_policy_hash": definition.selected_key_policy_hash,
            "candidate_library_state": row.decision if gate_pass else "focused_mechanical_repair_required", "candidate_decision": row.decision if gate_pass else "focused_mechanical_repair_required",
            "evidence_level": "level_2_train_only_bounded_screen_capped", "evidence_level_contract": "train_only_not_validation_not_holdout_not_live",
            "clean_evidence_allowed": False, "evidence_cap_reason": "shared_funding_imputation_ohlcv_stop_and_no_depth_execution_caps",
            "family_rejected": False, "train_only": True, "validation_run": False, "holdout_touched": False, "live_ready": False,
            "event_rows": row.events, "symbols": row.symbols, "base_mean_R": row.base_mean_R, "conservative_mean_R": row.conservative_mean_R, "severe_mean_R": row.severe_mean_R,
            "source_run_root": str(root), "contract_version": CONTRACT_VERSION,
        })
    write_csv(root / "candidate_library/backside_candidate_library_update.csv", library)
    data_manifest = Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv")
    funding_manifest = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1/funding/shared_funding_panel_manifest.csv")
    repro = {
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "code_path": str(Path(__file__)), "code_hash": file_hash(Path(__file__)),
        "source_runner_hash": file_hash(Path(base.__file__)), "config_hash": file_hash(root / "manifest/backside_blowoff_short_definitions.csv"), "contract_hash": file_hash(root / "contract/backside_signal_state_contract.md"),
        "data_snapshot_manifest_hash": file_hash(data_manifest), "pit_universe_manifest_hash": file_hash(root / "manifest/pit_panel.csv"), "funding_manifest_hash": file_hash(funding_manifest),
        "source_root_tree_hash": source_hash_before, "date_range": [base.START, base.END], "protected_boundary": base.PROTECTED,
        "contract_type": "Kraken PF perpetual instruments; R-normalized OHLCV train-only screen", "fee_model": "base 5bps/side; conservative 5bps/side; severe 10bps/side",
        "slippage_model": "base 4bps round trip; conservative 8bps; severe 12bps", "funding_model": "frozen shared exact/imputed direction-adverse panel", "seed_values": [19],
    }
    write_json(root / "reproducibility/run_manifest.json", repro)
    report = f"""# Backside Blowoff Repaired Reconciliation\n\nThe frozen 24-definition screen was replayed under `{CONTRACT_VERSION}` without seven-day selected-key preblocking. Old root: `{SOURCE_ROOT}`. Repaired root: `{root}`. Final classification: `{final}`. Old addresses unchanged/restored/removed: {int(mapping.address_status.eq('unchanged_address').sum())}/{int(mapping.address_status.eq('restored_or_new_address').sum())}/{int(mapping.address_status.eq('removed_address').sum())}. Preserved variants `bcbs_v1_002` and `bcbs_v1_008` were replayed explicitly. Prior economic decisions remain quarantined and are superseded by this repaired train-only evidence.\n"""
    (root / "BACKSIDE_REPAIRED_RECONCILIATION_REPORT.md").write_text(report)
    result = {
        "run_root": str(root), "status": "complete" if gate_pass else "blocked_by_protocol_issue", "final_classification": final, **hard,
        "raw_signals": len(raw), "parent_filtered_signals": len(projected), "accepted_trades": len(outcomes), "actual_overlap_skips": len(skips),
        "raw_signal_hash": raw_hash, "projection_hash": projection_hash, "accepted_trade_hash": contract_manifest["accepted_trade_hash"], "control_key_freeze_hash": control_freeze,
        "old_unchanged_addresses": int(mapping.address_status.eq("unchanged_address").sum()), "restored_addresses": int(mapping.address_status.eq("restored_or_new_address").sum()), "removed_addresses": int(mapping.address_status.eq("removed_address").sum()),
        "preserved_definitions": list(PRESERVED_IDS), "materialization_candidates": decisions[decisions.decision.eq("materialization_candidate")].definition_id.tolist(), "context_sleeves": decisions[decisions.decision.eq("context_sleeve_candidate")].definition_id.tolist(),
        "peak_rss_bytes": peak_rss, "runtime_seconds": time.monotonic()-started, "source_root_preserved_unchanged": source_mutation == 0,
        "validation_launched": False, "cpcv_launched": False, "holdout_launched": False, "portfolio_work_launched": False, "live_work_launched": False,
        "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    write_json(root / "decision_summary.json", result)
    compact_bundle(root)
    if gate_pass: update_campaign(root, final)
    write_json(root / "watch_status.json", {**result, "stage": "complete", "updated_ts": base.runner.utc_now()})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.run_root, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
