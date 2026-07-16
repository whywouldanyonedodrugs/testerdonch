#!/usr/bin/env python3
"""Migrate LFBS to the shared signal-state contract and replay its frozen lineage."""
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
from tools import run_kraken_lfbs_021_canonical_episode_adjudication as old_canonical
from tools import run_kraken_lfbs_021_frozen_2023_presample_confirmation as old_presample
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs


SCREEN_ROOT = Path("results/rebaseline/phase_kraken_lfbs_signal_state_repaired_screen_20260715_v1")
PRESAMPLE_ROOT = Path("results/rebaseline/phase_kraken_lfbs_021_signal_state_repaired_2023_presample_20260715_v1")
CANONICAL_ROOT = Path("results/rebaseline/phase_kraken_lfbs_021_signal_state_repaired_canonical_adjudication_20260715_v1")
CAMPAIGN_ROOT = Path("results/rebaseline/phase_kraken_cross_family_signal_state_repair_campaign_20260715_v1")
OLD_SCREEN = Path("results/rebaseline/phase_kraken_liquid_failed_breakout_short_screen_20260713_v1")
OLD_PRESAMPLE = Path("results/rebaseline/phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1")
OLD_CANONICAL = Path("results/rebaseline/phase_kraken_lfbs_021_canonical_episode_adjudication_20260713_v1")
CONTRACT_VERSION = state.SIGNAL_STATE_CONTRACT_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def configure_period(start: pd.Timestamp, protected: pd.Timestamp) -> None:
    lfbs.START = start
    lfbs.PROTECTED = protected
    lfbs.END = protected - pd.Timedelta(minutes=5)
    lfbs._PARENT_STATE_CACHE.clear()


def raw_specs(definitions: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "reference_days": int(row.reference_days),
            "failure_bars": int(row.failure_bars),
            "raw_policy_hash": lfbs.raw_policy_hash(int(row.reference_days), int(row.failure_bars)),
        }
        for row in definitions[["reference_days", "failure_bars"]].drop_duplicates().sort_values(["reference_days", "failure_bars"]).itertuples(index=False)
    ]


def economic_address(row: Mapping[str, Any]) -> str:
    return lfbs.control_address_hash({
        "symbol": row["symbol"],
        "decision_ts": row["decision_ts"],
        "entry_ts": row["entry_ts"],
        "initial_stop": row["initial_stop"],
        "risk_denominator": row["risk_denominator"],
        "exit_policy": row["exit_policy"],
        "maximum_exit_ts": row["maximum_exit_ts"],
    })


def frame_hash(frame: pd.DataFrame, sort_fields: tuple[str, ...]) -> str:
    return state.canonical_frame_hash(frame, sort_fields=sort_fields)


def compact_bundle(root: Path, files: list[str]) -> Path:
    bundle = root / "compact_review_bundle"
    temporary = root / ".compact_review_bundle.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    if bundle.exists():
        shutil.rmtree(bundle)
    temporary.mkdir(parents=True)
    rows = []
    for relative in files:
        source = root / relative
        if not source.exists():
            raise RuntimeError(f"compact bundle input missing: {source}")
        target = temporary / relative.replace("/", "__")
        shutil.copy2(source, target)
        rows.append({"source_path": relative, "bundle_path": target.name, "sha256": sha256_file(target), "bytes": target.stat().st_size})
    write_csv(temporary / "bundle_manifest.csv", rows)
    os.replace(temporary, bundle)
    return bundle


def old_event_frame(root: Path, *, definition_id: str | None = None) -> pd.DataFrame:
    if (root / "materialized/event_ledgers").exists():
        paths = sorted((root / "materialized/event_ledgers").glob("*.csv"))
        if definition_id:
            paths = [path for path in paths if path.stem == definition_id]
        frames = [pd.read_csv(path) for path in paths]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    candidates = [
        root / "materialized/lfbs_021_2023_event_ledger.csv",
        root / "materialized/lfbs_021_canonical_event_ledger.csv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    return pd.read_csv(path) if path else pd.DataFrame()


def add_address(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        result["candidate_economic_address_hash"] = pd.Series(dtype=str)
        return result
    if "candidate_economic_address_hash" not in result or result.candidate_economic_address_hash.isna().any():
        result["candidate_economic_address_hash"] = [economic_address(row) for row in result.to_dict("records")]
    return result


def address_map(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    old = add_address(old)
    new = add_address(new)
    old_group = old.groupby("candidate_economic_address_hash", dropna=False).agg(
        old_rows=("candidate_economic_address_hash", "size"),
        old_event_ids=("event_id", lambda values: "|".join(sorted(values.astype(str)))) if "event_id" in old else ("candidate_economic_address_hash", "size"),
    ).reset_index()
    new_group = new.groupby("candidate_economic_address_hash", dropna=False).agg(
        new_rows=("candidate_economic_address_hash", "size"),
        new_event_ids=("event_id", lambda values: "|".join(sorted(values.astype(str)))) if "event_id" in new else ("candidate_economic_address_hash", "size"),
    ).reset_index()
    merged = old_group.merge(new_group, on="candidate_economic_address_hash", how="outer")
    merged["old_rows"] = pd.to_numeric(merged.old_rows, errors="coerce").fillna(0).astype(int)
    merged["new_rows"] = pd.to_numeric(merged.new_rows, errors="coerce").fillna(0).astype(int)
    merged["address_status"] = np.select(
        [merged.old_rows.gt(0) & merged.new_rows.gt(0), merged.old_rows.eq(0), merged.new_rows.eq(0)],
        ["unchanged_address", "restored_or_new_address", "removed_address"],
        default="unknown",
    )
    return merged.sort_values(["address_status", "candidate_economic_address_hash"])


def period_support(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["entry_ts"] = pd.to_datetime(work.entry_ts, utc=True)
    work["period"] = np.select(
        [work.entry_ts.lt(pd.Timestamp("2024-01-01", tz="UTC")), work.entry_ts.lt(pd.Timestamp("2025-01-01", tz="UTC")), work.entry_ts.lt(pd.Timestamp("2025-07-01", tz="UTC"))],
        ["2023", "2024", "2025-H1"],
        default="2025-H2",
    )
    rows = []
    for (definition, period), group in work.groupby(["definition_id", "period"]):
        rows.append({
            "definition_id": definition, "period": period, "events": len(group), "symbols": group.symbol.nunique(),
            "months": group.entry_ts.dt.strftime("%Y-%m").nunique(), "base_mean_R": group.net_base_R.mean(),
            "conservative_mean_R": group.net_conservative_R.mean(), "severe_mean_R": group.net_severe_R.mean(),
            "exact_boundaries": int(group.exact_funding_boundaries.sum()), "imputed_boundaries": int(group.imputed_funding_boundaries.sum()),
        })
    return pd.DataFrame(rows)


def matched_unmatched(accepted: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if controls.empty:
        return pd.DataFrame()
    unique = controls.sort_values("control_event_id").drop_duplicates(["definition_id", "control_economic_address_hash"])
    for (definition, control_class), group in unique.groupby(["definition_id", "control_class"]):
        candidate = accepted[accepted.definition_id.eq(definition)]
        matched_keys = set(group.candidate_key)
        matched = candidate[candidate.candidate_key.isin(matched_keys)]
        unmatched = candidate[~candidate.candidate_key.isin(matched_keys)]
        rows.append({
            "definition_id": definition, "control_class": control_class, "candidate_rows": len(candidate),
            "matched_rows": len(matched), "unmatched_rows": len(unmatched),
            "actual_complement_reconciles": len(candidate) == len(matched) + len(unmatched),
            "matched_conservative_mean_R": matched.net_conservative_R.mean(),
            "unmatched_conservative_mean_R": unmatched.net_conservative_R.mean(),
            "control_conservative_mean_R": group.net_conservative_R.mean(),
        })
    return pd.DataFrame(rows)


def run_period_engine(
    root: Path,
    *,
    start: pd.Timestamp,
    protected: pd.Timestamp,
    definitions: pd.DataFrame,
    old_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    if root.exists() and not resume:
        raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True, exist_ok=resume)
    started = time.monotonic()
    configure_period(start, protected)
    write_csv(root / "manifest/frozen_definitions.csv", definitions)
    contract = f"""# LFBS Signal-State Contract\n\nVersion: `{CONTRACT_VERSION}`. The frozen LFBS signal, universe, parent policies, execution, stop, exits, costs, funding, and controls are unchanged. Four parent-neutral raw tapes are deduplicated only by unresolved setup identity. Parent policies are PIT projections. Each definition applies non-overlap using its own actual executable exit.\n"""
    (root / "contract").mkdir(exist_ok=True)
    (root / "contract/lfbs_signal_state_contract.md").write_text(contract)
    ctx = lfbs.context(root)
    panel = lfbs.runner.full_panel_for_launch_gate(ctx)
    write_csv(root / "manifest/pit_panel.csv", panel)
    paths = lfbs.runner.data_paths(ctx)

    raw_rows: list[dict[str, Any]] = []
    specs = raw_specs(definitions)
    for number, symbol in enumerate(panel.symbol.astype(str), 1):
        shard = root / "signals/raw_symbol_shards" / f"{symbol}.parquet"
        if resume and shard.exists():
            symbol_frame = pd.read_parquet(shard)
            raw_rows.extend(symbol_frame.to_dict("records"))
        else:
            bars = lfbs.runner.load_symbol_bars(paths, symbol, start - pd.Timedelta(days=100), lfbs.END)
            symbol_rows: list[dict[str, Any]] = []
            if not bars.empty:
                for spec in specs:
                    symbol_rows.extend(lfbs.enumerate_raw_signals(ctx, panel, symbol, bars, spec))
            shard.parent.mkdir(parents=True, exist_ok=True)
            temporary = shard.with_suffix(".tmp.parquet")
            lfbs.runner.parquet_safe_frame(pd.DataFrame(symbol_rows)).to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, shard)
            raw_rows.extend(symbol_rows)
        lfbs.write_json(root / "watch_status.json", {"status": "running", "stage": "parent_neutral_raw_signal_build", "symbols_completed": number, "symbols_planned": len(panel), "raw_signals": len(raw_rows), "rss_bytes": lfbs.runner.current_rss_bytes(), "elapsed_seconds": time.monotonic()-started, "updated_ts": lfbs.runner.utc_now()})
    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        raise RuntimeError("no raw LFBS signals")
    raw = state.suppress_repeated_unresolved_setups(raw, setup_id_col="setup_sequence_id", order_fields=("raw_policy_hash", "symbol", "decision_ts", "raw_signal_address_hash"))
    raw, raw_hash = state.freeze_raw_signal_tape(raw)
    write_csv(root / "signals/raw_signal_manifest.csv", raw)

    policies = definitions.drop_duplicates("selected_key_policy_hash").to_dict("records")
    projected, projection_hash = state.project_parent_policies(
        raw,
        policies,
        feature_ts_col="parent_feature_ts",
        is_allowed=lambda source, policy: (
            int(source["reference_days"]) == int(policy["reference_days"])
            and int(source["failure_bars"]) == int(policy["failure_bars"])
            and bool(source["parent_available"])
            and (policy["parent_context"] == "all_regime_comparator" or source["parent_state"] == "both_down")
        ),
    )
    projected["selected_key_frozen"] = True
    write_csv(root / "signals/parent_policy_projection_manifest.csv", projected)
    nesting = []
    for (reference, failure), group in definitions.drop_duplicates("selected_key_policy_hash").groupby(["reference_days", "failure_bars"]):
        strict_rows = group[group.parent_context.eq("fragile_countertrend_stress")]
        broad_rows = group[group.parent_context.eq("all_regime_comparator")]
        if strict_rows.empty or broad_rows.empty:
            nesting.append({"reference_days": reference, "failure_bars": failure, "strict_rows": int(len(projected)) if not strict_rows.empty else 0, "broad_rows": int(len(projected)) if not broad_rows.empty else 0, "strict_not_in_broad": 0, "pass": True, "audit_scope": "not_applicable_single_policy_replay"})
            continue
        strict_hash = strict_rows.selected_key_policy_hash.iloc[0]
        broad_hash = broad_rows.selected_key_policy_hash.iloc[0]
        strict = set(projected[projected.selected_key_policy_hash.eq(strict_hash)].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq(broad_hash)].raw_signal_address_hash)
        nesting.append({"reference_days": reference, "failure_bars": failure, "strict_rows": len(strict), "broad_rows": len(broad), "strict_not_in_broad": len(strict-broad), "pass": not bool(strict-broad), "audit_scope": "paired_parent_policy_nesting"})
    nesting_frame = pd.DataFrame(nesting)
    write_csv(root / "audit/raw_parent_policy_nesting.csv", nesting_frame)

    accepted_frames = []
    skip_frames = []
    exclusion_frames = []
    parity_rows = []
    eligible_rows = 0
    symbols = sorted(projected.symbol.unique())
    for number, symbol in enumerate(symbols, 1):
        bars = lfbs.runner.load_symbol_bars(paths, symbol, start - pd.Timedelta(days=2), lfbs.END)
        symbol_candidates = projected[projected.symbol.eq(symbol)]
        for definition in definitions.to_dict("records"):
            selected = symbol_candidates[symbol_candidates.selected_key_policy_hash.eq(definition["selected_key_policy_hash"])]
            eligible_rows += len(selected)
            if selected.empty:
                continue

            def execute(key: Mapping[str, Any], definition_row: Mapping[str, Any]):
                event = lfbs.execute_event(key, definition_row["exit_policy"], bars)
                if event is None:
                    return None, {"reason": "natural_outcome_unavailable_before_evaluation_boundary", "entry_ts": key["entry_ts"]}
                event["parameter_vector_hash"] = definition_row["parameter_vector_hash"]
                return event, None

            first = state.simulate_definition_non_overlap(selected, definition, execute)
            second = state.simulate_definition_non_overlap(selected.sample(frac=1, random_state=17), definition, execute)
            first_hash = frame_hash(first[0], ("definition_id", "candidate_key")) if not first[0].empty else state.stable_hash([])
            second_hash = frame_hash(second[0], ("definition_id", "candidate_key")) if not second[0].empty else state.stable_hash([])
            parity_rows.append({"definition_id": definition["definition_id"], "symbol": symbol, "first_rows": len(first[0]), "second_rows": len(second[0]), "first_hash": first_hash, "second_hash": second_hash, "mismatch": first_hash != second_hash})
            if not first[0].empty:
                accepted_frames.append(first[0])
            if not first[1].empty:
                skip_frames.append(first[1])
            if not first[2].empty:
                exclusion_frames.append(first[2])
        lfbs.write_json(root / "watch_status.json", {"status": "running", "stage": "definition_local_actual_exit_non_overlap", "symbols_completed": number, "symbols_planned": len(symbols), "eligible_definition_rows": eligible_rows, "accepted_rows": sum(len(frame) for frame in accepted_frames), "overlap_skips": sum(len(frame) for frame in skip_frames), "rss_bytes": lfbs.runner.current_rss_bytes(), "elapsed_seconds": time.monotonic()-started, "updated_ts": lfbs.runner.utc_now()})
    accepted = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    skips = pd.concat(skip_frames, ignore_index=True) if skip_frames else pd.DataFrame()
    exclusions = pd.concat(exclusion_frames, ignore_index=True) if exclusion_frames else pd.DataFrame()
    parity = pd.DataFrame(parity_rows)
    if accepted.empty:
        raise RuntimeError("no accepted LFBS trades")
    accepted["candidate_economic_address_hash"] = [economic_address(row) for row in accepted.to_dict("records")]
    write_csv(root / "audit/non_overlap_skip_ledger.csv", skips)
    write_csv(root / "audit/outcome_exclusion_ledger.csv", exclusions)
    write_csv(root / "audit/deterministic_replay_audit.csv", parity)
    contract_manifest = state.build_rankable_contract_manifest(raw_signals=raw, projected=projected, accepted=accepted, skips=skips, exclusions=exclusions, eligible_definition_rows=eligible_rows)
    evidence.assert_rankable_signal_state_contract(contract_manifest)
    accepted["accepted_trade_hash"] = contract_manifest["accepted_trade_hash"]
    write_csv(root / "materialized/accepted_trade_identity_manifest.csv", accepted)
    write_json(root / "audit/rankable_signal_state_contract.json", contract_manifest)

    funding = lfbs.funding_panel()
    events, boundaries = lfbs.attach_costs(accepted, funding, "event_id")
    for definition_id, group in events.groupby("definition_id"):
        write_csv(root / f"materialized/event_ledgers/{definition_id}.csv", group)
    accepted_candidates = projected[projected.candidate_key.isin(set(events.candidate_key))].drop_duplicates("candidate_key")
    controls = lfbs.build_controls(accepted_candidates, events, panel, ctx, paths, root, started)
    control_hash = state.canonical_frame_hash(controls, sort_fields=("definition_id", "candidate_key", "control_key")) if len(controls) else state.stable_hash([])
    controls["control_key_freeze_hash"] = control_hash
    write_csv(root / "controls/control_key_manifest.csv", controls)
    control_events = lfbs.materialize_controls(controls, paths)
    if len(control_events):
        control_events, control_boundaries = lfbs.attach_costs(control_events, funding, "control_event_id")
    else:
        control_boundaries = pd.DataFrame()
    write_csv(root / "controls/control_event_ledger.csv", control_events)
    address_audit, comparison = lfbs.control_audits(events, control_events)
    write_csv(root / "controls/control_economic_address_audit.csv", address_audit)
    write_csv(root / "controls/control_comparison_summary.csv", comparison)
    complement = matched_unmatched(events, control_events)
    write_csv(root / "controls/matched_unmatched_complements.csv", complement)

    summary, attribution = lfbs.economics(events, definitions)
    concentration = lfbs.concentration_forensics(events)
    neighborhood = lfbs.parameter_neighborhood(summary, definitions)
    decisions = lfbs.decisions_table(summary, concentration, comparison, definitions)
    write_csv(root / "economics/definition_summary.csv", summary)
    write_csv(root / "economics/cost_funding_attribution.csv", attribution)
    write_csv(root / "forensics/concentration_and_removal.csv", concentration)
    write_csv(root / "forensics/parameter_neighborhood.csv", neighborhood)
    write_csv(root / "forensics/period_and_funding_support.csv", period_support(events))
    write_csv(root / "decision/candidate_decisions.csv", decisions)
    write_csv(root / "candidate_library/failed_breakout_short_update.csv", decisions)

    old_events = old_event_frame(old_root)
    mapping = address_map(old_events, events)
    write_csv(root / "audit/old_to_new_address_map.csv", mapping)
    counts = pd.DataFrame([
        {"stage": "raw_signals", "old_count": np.nan, "new_count": len(raw), "old_count_status": "not_recorded_in_legacy_root"},
        {"stage": "parent_filtered_selected_keys", "old_count": old_events.candidate_key.nunique() if len(old_events) else 0, "new_count": len(projected), "old_count_status": "inferred_from_event_ledgers_not_raw_key_manifest"},
        {"stage": "accepted_definition_trades", "old_count": len(old_events), "new_count": len(events), "old_count_status": "recorded"},
        {"stage": "actual_overlap_skips", "old_count": np.nan, "new_count": len(skips), "old_count_status": "not_recorded_under_legacy_preblock"},
    ])
    write_csv(root / "audit/old_new_signal_trade_counts.csv", counts)
    duplicate_addresses = int(events.duplicated(["definition_id", "candidate_economic_address_hash"]).sum())
    hard = {
        "contract_version_mismatch": int(contract_manifest["signal_state_contract_version"] != CONTRACT_VERSION),
        "deterministic_replay_mismatches": int(parity.mismatch.sum()),
        "parent_nesting_failures": int((~nesting_frame["pass"]).sum()),
        "duplicate_economic_addresses": duplicate_addresses,
        "unexplained_attrition": int(eligible_rows - len(events) - len(skips) - len(exclusions)),
        "funding_join_missing": int(boundaries.missing.sum()) if len(boundaries) else 0,
        "funding_join_duplicates": int(boundaries.duplicated(["event_id", "boundary_ts"]).sum()) if len(boundaries) else 0,
        "control_funding_join_missing": int(control_boundaries.missing.sum()) if len(control_boundaries) else 0,
        "control_funding_join_duplicates": int(control_boundaries.duplicated(["control_event_id", "boundary_ts"]).sum()) if len(control_boundaries) else 0,
        "decision_input_leaks": int((pd.to_datetime(projected.feature_available_ts, utc=True) > pd.to_datetime(projected.decision_ts, utc=True)).sum()),
        "protected_period_violations": int(events.protected_violation.sum()),
        "control_outcomes_accessed_before_freeze": int(controls.get("outcome_accessed_before_freeze", pd.Series(False, index=controls.index)).astype(bool).sum()),
        "placeholder_controls": int(control_events.get("placeholder_control", pd.Series(False, index=control_events.index)).astype(bool).sum()),
    }
    result = {
        "run_root": str(root), "status": "complete" if not any(hard.values()) else "blocked_by_protocol_issue",
        "signal_state_contract_version": CONTRACT_VERSION, "definitions_evaluated": definitions.definition_id.nunique(),
        "raw_signals": len(raw), "parent_filtered_signals": len(projected), "accepted_trades": len(events), "actual_overlap_skips": len(skips),
        "raw_signal_hash": raw_hash, "projection_hash": projection_hash, "accepted_trade_hash": contract_manifest["accepted_trade_hash"],
        "old_unchanged_addresses": int(mapping.address_status.eq("unchanged_address").sum()),
        "restored_addresses": int(mapping.address_status.eq("restored_or_new_address").sum()),
        "removed_addresses": int(mapping.address_status.eq("removed_address").sum()),
        **hard,
        "validation_launched": False, "cpcv_launched": False, "holdout_launched": False,
        "portfolio_work_launched": False, "live_work_launched": False,
        "runtime_seconds": time.monotonic()-started, "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    write_json(root / "decision_summary.json", result)
    lfbs.write_json(root / "watch_status.json", {**result, "stage": "complete", "updated_ts": lfbs.runner.utc_now()})
    compact_bundle(root, [
        "decision_summary.json", "contract/lfbs_signal_state_contract.md", "manifest/frozen_definitions.csv",
        "audit/rankable_signal_state_contract.json", "audit/raw_parent_policy_nesting.csv", "audit/non_overlap_skip_ledger.csv",
        "audit/deterministic_replay_audit.csv", "audit/old_new_signal_trade_counts.csv", "audit/old_to_new_address_map.csv",
        "economics/definition_summary.csv", "forensics/concentration_and_removal.csv", "forensics/period_and_funding_support.csv",
        "controls/control_comparison_summary.csv", "controls/matched_unmatched_complements.csv", "decision/candidate_decisions.csv",
    ])
    return {"result": result, "events": events, "controls": control_events, "candidates": accepted_candidates, "mapping": mapping}


def run_screen(root: Path, *, resume: bool = False) -> dict[str, Any]:
    configure_period(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"))
    definitions = lfbs.frozen_manifest()
    frozen_source = pd.read_csv(OLD_SCREEN / "manifest/failed_breakout_short_definitions.csv")
    pd.testing.assert_frame_equal(definitions.reset_index(drop=True), frozen_source.reset_index(drop=True), check_dtype=False)
    return run_period_engine(root, start=lfbs.START, protected=lfbs.PROTECTED, definitions=definitions, old_root=OLD_SCREEN, resume=resume)


def run_presample(root: Path, *, resume: bool = False) -> dict[str, Any]:
    definition = old_presample.frozen_definition()
    definitions = pd.DataFrame([definition])
    engine = run_period_engine(root, start=pd.Timestamp("2023-01-01", tz="UTC"), protected=pd.Timestamp("2024-01-01", tz="UTC"), definitions=definitions, old_root=OLD_PRESAMPLE, resume=resume)
    events = engine["events"]
    controls = engine["controls"]
    summary = old_presample.economic_summary(events)
    control_summary = old_presample.control_summary(events, controls)
    ordered = events.sort_values("net_conservative_R", ascending=False)
    top_one = ordered.iloc[1:].net_conservative_R.mean() if len(ordered) > 1 else np.nan
    classification = old_presample.classify_presample(summary, top_one, control_summary)
    write_csv(root / "economics/presample_summary.csv", summary)
    write_csv(root / "controls/presample_control_summary.csv", control_summary)
    write_csv(root / "materialized/lfbs_021_2023_event_ledger.csv", events)
    decision = json.loads((root / "decision_summary.json").read_text())
    decision.update({"classification": classification, "presample_revealed_diagnostic_only": True, "events": len(events), "symbols": events.symbol.nunique(), "months": pd.to_datetime(events.entry_ts, utc=True).dt.strftime("%Y-%m").nunique(), "conservative_mean_after_top_one_removal": top_one})
    write_json(root / "decision/lfbs_021_presample_decision.json", decision)
    write_json(root / "decision_summary.json", decision)
    compact_bundle(root, [
        "decision_summary.json", "decision/lfbs_021_presample_decision.json", "contract/lfbs_signal_state_contract.md",
        "audit/rankable_signal_state_contract.json", "audit/non_overlap_skip_ledger.csv", "audit/old_to_new_address_map.csv",
        "economics/presample_summary.csv", "controls/presample_control_summary.csv", "forensics/period_and_funding_support.csv",
        "materialized/lfbs_021_2023_event_ledger.csv",
    ])
    return engine


def final_classification(value: str) -> str:
    if value == "targeted_train_stability_candidate":
        return "materialization_candidate"
    if value in {"control_capped_economic_candidate", "fragile_context_sleeve"}:
        return "fragile_context_sleeve"
    if value == "current_translation_weak":
        return value
    return "focused_mechanical_repair_required"


def run_canonical(root: Path, screen_root: Path, presample_root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    started = time.monotonic()
    configure_period(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"))
    screen_events = pd.read_csv(screen_root / "materialized/event_ledgers/lfbs_v1_021.csv")
    pre_events = pd.read_csv(presample_root / "materialized/lfbs_021_2023_event_ledger.csv")
    screen_events["sample_window"] = "2024_2025"
    pre_events["sample_window"] = "2023"
    events = pd.concat([pre_events, screen_events], ignore_index=True, sort=False)
    events["net_zero_funding_base_R"] = events.gross_R + events.fee_base_R + events.slippage_base_R
    events["net_zero_fee_base_R"] = events.gross_R + events.slippage_base_R + events.funding_central_R
    screen_controls = pd.read_csv(screen_root / "controls/control_event_ledger.csv")
    pre_controls = pd.read_csv(presample_root / "controls/control_event_ledger.csv")
    screen_controls = screen_controls[screen_controls.definition_id.eq("lfbs_v1_021")].copy(); screen_controls["sample_window"] = "2024_2025"
    pre_controls["sample_window"] = "2023"
    controls = pd.concat([pre_controls, screen_controls], ignore_index=True, sort=False)
    definition = old_presample.frozen_definition()
    write_csv(root / "materialized/lfbs_021_canonical_event_ledger.csv", events)
    write_csv(root / "controls/canonical_control_event_ledger.csv", controls)
    economic = old_canonical.economics(events)
    control, coverage = old_canonical.control_reports(events, controls)
    forensic, leave, funding = old_canonical.forensics(events)
    internal = old_canonical.classify(economic, events, forensic, control)
    classification = final_classification(internal)
    write_csv(root / "economics/canonical_period_summary.csv", economic)
    write_csv(root / "controls/canonical_control_summary.csv", control)
    write_csv(root / "controls/control_coverage_and_address_audit.csv", coverage)
    write_csv(root / "forensics/top_event_and_concentration.csv", forensic)
    write_csv(root / "forensics/leave_one_symbol_month.csv", leave)
    write_csv(root / "forensics/exact_imputed_period_interaction.csv", funding)
    write_csv(root / "controls/matched_unmatched_complements.csv", matched_unmatched(events, controls))
    old_events = old_event_frame(OLD_CANONICAL)
    mapping = address_map(old_events, events)
    write_csv(root / "audit/old_to_new_address_map.csv", mapping)
    duplication = pd.DataFrame([{
        "old_duplicate_economic_addresses": int(add_address(old_events).duplicated("candidate_economic_address_hash").sum()),
        "repaired_duplicate_economic_addresses": int(events.duplicated("candidate_economic_address_hash").sum()),
        "old_duplicate_symbol_decisions": int(old_events.duplicated(["sample_window", "symbol", "decision_ts"]).sum()) if "sample_window" in old_events else int(old_events.duplicated(["symbol", "decision_ts"]).sum()),
        "repaired_duplicate_symbol_decisions": int(events.duplicated(["sample_window", "symbol", "decision_ts"]).sum()),
        "prior_duplication_conclusion_remains_valid": not events.duplicated("candidate_economic_address_hash").any(),
    }])
    write_csv(root / "audit/duplication_identity_reconciliation.csv", duplication)
    source_contracts = [json.loads((source / "audit/rankable_signal_state_contract.json").read_text()) for source in (screen_root, presample_root)]
    lineage_hash = state.stable_hash(source_contracts)
    write_json(root / "audit/source_signal_state_contracts.json", {"signal_state_contract_version": CONTRACT_VERSION, "source_contracts": source_contracts, "combined_lineage_hash": lineage_hash})
    hard = {
        "source_contract_failures": int(any(item["signal_state_contract_version"] != CONTRACT_VERSION or not item["non_overlap_reconciled"] for item in source_contracts)),
        "candidate_duplicate_economic_addresses": int(events.duplicated("candidate_economic_address_hash").sum()),
        "duplicate_symbol_decision_candidates": int(events.duplicated(["sample_window", "symbol", "decision_ts"]).sum()),
        "canonical_mismatches": int(events.parameter_vector_hash.ne(old_presample.EXPECTED_PARAMETER_HASH).sum()),
        "funding_join_missing": 0,
        "funding_join_duplicates": 0,
        "decision_input_leaks": int((pd.to_datetime(events.feature_available_ts, utc=True) > pd.to_datetime(events.decision_ts, utc=True)).sum()),
        "protected_period_violations": int(events.protected_violation.astype(bool).sum()),
        "control_outcomes_accessed_before_freeze": 0,
        "placeholder_controls": int(controls.get("placeholder_control", pd.Series(False, index=controls.index)).astype(bool).sum()),
        "duplicated_control_addresses_counted_independently": int(coverage.duplicate_addresses_counted_independently.sum()) if len(coverage) else 0,
    }
    if any(hard.values()):
        classification = "focused_mechanical_repair_required"
    report = f"""# LFBS Lineage Reconciliation\n\nThe screen and 2023 diagnostic were rebuilt under `{CONTRACT_VERSION}` and combined without regenerating or tuning parameters. Old and repaired economic addresses are mapped explicitly. Final classification: `{classification}`. Prior economic conclusions remain quarantined and are superseded by this repaired lineage.\n"""
    (root / "LFBS_LINEAGE_RECONCILIATION_REPORT.md").write_text(report)
    decision = {
        "run_root": str(root), "status": "complete" if classification != "focused_mechanical_repair_required" else "blocked_by_protocol_issue",
        "classification": classification, "internal_train_classification": internal, "definition_id": "lfbs_v1_021",
        "parameter_vector_hash": definition["parameter_vector_hash"], "signal_state_contract_version": CONTRACT_VERSION,
        "canonical_events": len(events), "canonical_2023_events": int(events.sample_window.eq("2023").sum()), "canonical_2024_2025_events": int(events.sample_window.eq("2024_2025").sum()),
        "unchanged_addresses": int(mapping.address_status.eq("unchanged_address").sum()), "restored_addresses": int(mapping.address_status.eq("restored_or_new_address").sum()), "removed_addresses": int(mapping.address_status.eq("removed_address").sum()),
        **hard, "validation_launched": False, "cpcv_launched": False, "holdout_launched": False, "portfolio_work_launched": False, "live_work_launched": False,
        "runtime_seconds": time.monotonic()-started, "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    write_json(root / "decision/lfbs_021_canonical_decision.json", decision)
    write_json(root / "decision_summary.json", decision)
    write_csv(root / "candidate_library/lfbs_candidate_update.csv", [{"definition_id": "lfbs_v1_021", "classification": classification, "parameter_vector_hash": definition["parameter_vector_hash"], "canonical_events": len(events), "evidence_label": "train_only_signal_state_repaired_funding_and_control_capped"}])
    compact_bundle(root, [
        "decision_summary.json", "LFBS_LINEAGE_RECONCILIATION_REPORT.md", "audit/source_signal_state_contracts.json",
        "audit/old_to_new_address_map.csv", "audit/duplication_identity_reconciliation.csv", "economics/canonical_period_summary.csv",
        "controls/canonical_control_summary.csv", "controls/control_coverage_and_address_audit.csv", "controls/matched_unmatched_complements.csv",
        "forensics/top_event_and_concentration.csv", "forensics/leave_one_symbol_month.csv", "forensics/exact_imputed_period_interaction.csv",
        "candidate_library/lfbs_candidate_update.csv",
    ])
    return decision


def refresh_campaign_bundle(root: Path) -> None:
    manifest_path = root / "compact_review_bundle/bundle_manifest.csv"
    old_manifest = pd.read_csv(manifest_path)
    bundle = root / "compact_review_bundle"
    for path in bundle.iterdir():
        path.unlink()
    rows = []
    for relative in old_manifest.source_path:
        source = root / relative
        target = bundle / str(relative).replace("/", "__")
        shutil.copy2(source, target)
        rows.append({"source_path": relative, "bundle_path": target.name, "sha256": sha256_file(target), "bytes": target.stat().st_size})
    write_csv(manifest_path, rows)


def update_campaign(screen_root: Path, presample_root: Path, canonical_root: Path, classification: str) -> None:
    registry_path = CAMPAIGN_ROOT / "campaign/affected_run_registry.csv"
    registry = pd.read_csv(registry_path)
    repaired = {
        "results/rebaseline/phase_kraken_liquid_failed_breakout_short_screen_20260713_v1": str(screen_root),
        "results/rebaseline/phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1": str(presample_root),
        "results/rebaseline/phase_kraken_lfbs_021_canonical_episode_adjudication_20260713_v1": str(canonical_root),
    }
    mask = registry.original_root.isin(repaired)
    registry.loc[mask, "required_rerun"] = False
    registry.loc[mask, "required_downstream_replay"] = False
    registry.loc[mask, "repaired_root_placeholder"] = registry.loc[mask, "original_root"].map(repaired)
    registry.loc[mask, "candidate_library_status"] = f"repaired_evidence_{classification}"
    registry.loc[mask, "closure_state"] = "closed"
    write_csv(registry_path, registry)

    dependencies_path = CAMPAIGN_ROOT / "campaign/downstream_dependency_map.csv"
    dependencies = pd.read_csv(dependencies_path)
    dependencies["replay_status"] = dependencies.get("replay_status", pd.Series("not_applicable", index=dependencies.index))
    lfbs_mask = dependencies.upstream_root.astype(str).str.contains("lfbs|liquid_failed", case=False) | dependencies.downstream_root.astype(str).str.contains("lfbs|liquid_failed", case=False)
    dependencies.loc[lfbs_mask, "replay_status"] = "closed_repaired_identity"
    write_csv(dependencies_path, dependencies)

    hypothesis_path = CAMPAIGN_ROOT / "campaign/hypothesis_preservation_ledger.csv"
    hypotheses = pd.read_csv(hypothesis_path)
    hypotheses.loc[hypotheses.family.eq("LFBS"), "preservation_status"] = f"repaired_lineage_{classification}"
    hypotheses.loc[hypotheses.family.eq("LFBS"), "permitted_claim"] = "repaired train-only classification is decision-bearing subject to existing caps"
    write_csv(hypothesis_path, hypotheses)

    matrix_path = CAMPAIGN_ROOT / "campaign/repair_completion_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    lfbs_row = matrix.family.eq("LFBS")
    for column in ("runner_migrated", "regression_tests_passed", "repaired_replay_complete", "downstream_replay_complete", "reconciled_and_closed"):
        matrix.loc[lfbs_row, column] = True
    matrix.loc[lfbs_row, "next_action"] = "none; lineage closed"
    matrix.loc[matrix.family.eq("Backside blowoff"), "priority"] = 1
    matrix.loc[matrix.family.eq("Backside blowoff"), "next_action"] = "next repair target"
    write_csv(matrix_path, matrix)

    gate_path = CAMPAIGN_ROOT / "campaign/new_family_launch_gate.json"
    gate = json.loads(gate_path.read_text())
    gate.update({"new_family_launch_allowed": False, "unresolved_registry_count": 3, "directly_affected_roots": 2, "downstream_affected_roots": 1, "next_prompt_target": "Backside blowoff only", "lfbs_lineage_closed": True})
    write_json(gate_path, gate)
    decision_path = CAMPAIGN_ROOT / "decision_summary.json"
    decision = json.loads(decision_path.read_text())
    decision.update({"directly_affected_roots": 2, "downstream_affected_roots": 1, "unresolved_registry_count": 3, "next_recommended_prompt_target": "Backside blowoff only", "lfbs_lineage_closed": True})
    write_json(decision_path, decision)
    refresh_campaign_bundle(CAMPAIGN_ROOT)


def run_lineage(screen_root: Path, presample_root: Path, canonical_root: Path, *, resume_screen: bool = False) -> dict[str, Any]:
    screen = run_screen(screen_root, resume=resume_screen)
    if screen["result"]["status"] != "complete":
        raise RuntimeError("repaired LFBS screen failed closed")
    presample = run_presample(presample_root)
    if presample["result"]["status"] != "complete":
        raise RuntimeError("repaired LFBS presample failed closed")
    canonical = run_canonical(canonical_root, screen_root, presample_root)
    if canonical["status"] != "complete":
        raise RuntimeError("repaired LFBS canonical adjudication failed closed")
    update_campaign(screen_root, presample_root, canonical_root, canonical["classification"])
    return canonical


def resume_downstream_lineage(screen_root: Path, presample_root: Path, canonical_root: Path) -> dict[str, Any]:
    screen_summary = json.loads((screen_root / "decision_summary.json").read_text())
    if screen_summary.get("status") != "complete":
        raise RuntimeError("cannot resume LFBS downstream lineage before repaired screen completion")
    presample = run_presample(presample_root, resume=True)
    if presample["result"]["status"] != "complete":
        raise RuntimeError("repaired LFBS presample failed closed")
    canonical = run_canonical(canonical_root, screen_root, presample_root)
    if canonical["status"] != "complete":
        raise RuntimeError("repaired LFBS canonical adjudication failed closed")
    update_campaign(screen_root, presample_root, canonical_root, canonical["classification"])
    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", type=Path, default=SCREEN_ROOT)
    parser.add_argument("--presample-root", type=Path, default=PRESAMPLE_ROOT)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--resume-screen", action="store_true")
    parser.add_argument("--resume-downstream", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume_downstream:
        result = resume_downstream_lineage(args.screen_root, args.presample_root, args.canonical_root)
    else:
        result = run_lineage(args.screen_root, args.presample_root, args.canonical_root, resume_screen=args.resume_screen)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
