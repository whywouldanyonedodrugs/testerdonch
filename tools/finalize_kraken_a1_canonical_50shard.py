#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_shared_funding_consumer as consumer
from tools import run_kraken_a1_funding_corrected_balanced_50shard_canonical as run
from tools import run_kraken_family_engine_aggregate_first_sweep as runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    started = time.monotonic()
    selected = pd.read_csv(root / "shards/selected_50_shard_plan.csv")
    manifest = runner.load_a1_compression_manifest()
    runner_args = runner.parse_args(["--phase-profile", runner.A1_COMPRESSION_PRODUCTION_SHARDED_AGGREGATE_PHASE_PROFILE, "--run-root", str(root), "--start", "2024-01-01", "--end", "2025-12-31"])
    ctx = runner.Context(args=runner_args, run_root=root, start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2025-12-31", tz="UTC"), notifier=None)
    selected_defs = runner.a1_definitions_for_selected_key_specs(manifest, ctx, selected["selected_key_policy_hash"])
    if len(selected) != 50 or len(selected_defs) != 400:
        raise RuntimeError("reducer-only continuation requires exactly 50 specs and 400 definitions")
    validations = [runner.a1_validate_finalized_aggregate_shard(root, str(row.shard_id)) for row in selected.itertuples()]
    if len(validations) != 50 or any(row.get("status") != "pass" for row in validations):
        raise RuntimeError("reducer-only continuation found an invalid finalized shard")
    status_rows = [json.loads((root / "aggregate_shards" / str(row.shard_id) / "shard_manifest.json").read_text()) for row in selected.itertuples()]
    status = pd.DataFrame(status_rows)
    run.write_csv(root / "shards/shard_status_summary.csv", status)

    gate_frames = [pd.read_csv(path) for path in sorted((root / "gates").glob("a1shard_*_funding_gate_pre_post_event_counts.csv"))]
    gate_counts = pd.concat(gate_frames, ignore_index=True, sort=False) if gate_frames else pd.DataFrame()
    run.write_csv(root / "gates/funding_gate_pre_post_event_counts.csv", gate_counts)
    recognized = {"exclude_top_20pct_positive_funding", "exclude_top_decile_positive_funding", "funding_aware_cap", "no_funding_gate_diagnostic_cap"}
    binding_rows = []
    for gate in sorted(set(selected_defs["funding_gate"].astype(str))):
        subset = gate_counts[gate_counts["funding_gate"].astype(str).eq(gate)] if not gate_counts.empty else pd.DataFrame()
        binding_rows.append({
            "funding_gate": gate, "recognized": gate in recognized,
            "definition_count": int(selected_defs["funding_gate"].astype(str).eq(gate).sum()),
            "pre_gate_rows": int(pd.to_numeric(subset.get("pre_funding_gate_definition_event_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "post_gate_rows": int(pd.to_numeric(subset.get("post_funding_gate_definition_event_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "imputed_funding_used_for_gate": False, "status": "pass" if gate in recognized else "fail",
        })
    run.write_csv(root / "gates/funding_gate_binding_audit.csv", binding_rows)

    ledger_paths = [root / "aggregate_shards" / str(row.shard_id) / "outcome_events.parquet" for row in selected.itertuples()]
    events = consumer.normalize_frozen_events(pd.concat([pd.read_parquet(path) for path in ledger_paths], ignore_index=True), "a1")
    expected_hash = selected_defs.set_index("candidate_definition_id")["selected_key_policy_hash"]
    expected_event_hash = events["candidate_definition_id"].map(expected_hash)
    lineage_rows = []
    for _, shard in selected.iterrows():
        shard_id = str(shard["shard_id"])
        expected = str(shard["selected_key_policy_hash"])
        shard_dir = root / "aggregate_shards" / shard_id
        selected_rows = pd.read_csv(shard_dir / "selected_keys.csv")
        outcome_rows = pd.read_parquet(shard_dir / "outcome_events.parquet")
        shard_manifest = json.loads((shard_dir / "shard_manifest.json").read_text())
        observations = {
            "plan": {expected},
            "definition_rows": set(selected_defs.loc[selected_defs["candidate_definition_id"].isin(outcome_rows["candidate_definition_id"]), "selected_key_policy_hash"].astype(str)),
            "selected_event_rows": set(selected_rows["selected_key_policy_hash"].astype(str)),
            "outcome_rows": set(outcome_rows["selected_key_policy_hash"].astype(str)),
            "shard_manifest": {str(shard_manifest.get("selected_key_policy_hash", ""))},
        }
        for source, observed in observations.items():
            lineage_rows.append({"shard_id": shard_id, "artifact_source": source, "planned_canonical_hash": expected, "observed_canonical_hash": ";".join(sorted(observed)), "contract_version": runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION, "status": "pass" if observed == {expected} else "fail"})
    lineage = pd.DataFrame(lineage_rows)
    canonical_mismatches = int((lineage["status"] != "pass").sum())
    run.write_csv(root / "audit/canonical_hash_lineage_audit.csv", lineage)
    if canonical_mismatches or not events["selected_key_policy_hash"].astype(str).eq(expected_event_hash.astype(str)).all():
        raise RuntimeError("canonical lineage failed during reducer-only continuation")

    boundaries = consumer.build_event_boundary_rows(events)
    panel, extension = balanced.extend_frozen_panel_with_verified_model(
        run.load_frozen_panel(), boundaries,
        "results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1",
    )
    run.write_csv(root / "funding/panel_extension_audit.csv", extension)
    joined = consumer.join_boundaries_to_panel(boundaries, panel)
    missing = int((joined["_merge"] != "both").sum())
    duplicate = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    exact = joined["funding_exact"].fillna(False)
    exact_unchanged = bool(np.isclose(joined.loc[exact, "relativeFundingRate"], joined.loc[exact, "funding_rate_central"], rtol=0, atol=0).all())
    run.write_csv(root / "audit/funding_boundary_join_audit.csv", [{"required_boundary_rows": len(boundaries), "joined_boundary_rows": len(joined), "missing_boundary_joins": missing, "duplicate_boundary_joins": duplicate, "exact_rows_preserved": exact_unchanged, "status": "pass" if missing == 0 and duplicate == 0 and exact_unchanged else "fail"}])
    if missing or duplicate or not exact_unchanged:
        raise RuntimeError("funding boundary integration failed")
    rescored = consumer.aggregate_event_funding(events, joined)
    scenarios = balanced.scenario_event_rows(rescored, consumer.FUNDING_MODES, (4, 8, 12))
    scorecard = run.definition_scorecard(scenarios)
    scorecard["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    lane = consumer.grouped_rescore(rescored, ["definition_lane"], family="a1")
    lane = lane[lane["slippage_round_trip_bps"].isin([4, 8, 12])]
    lane["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    exits = consumer.grouped_rescore(rescored, ["exit_policy_id"], family="a1")
    exits = exits[exits["slippage_round_trip_bps"].isin([4, 8, 12])]
    exits["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    spec = scenarios.groupby(["selected_key_policy_hash", "definition_lane", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(definitions=("candidate_definition_id", "nunique"), events=("event_key", "nunique"), median_event_net_R=("scenario_scaled_net_R", "median"), total_net_R=("scenario_scaled_net_R", "sum")).reset_index()
    spec["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    entry = pd.to_datetime(scenarios["entry_ts"], utc=True)
    scenarios["period_scope"] = np.select([entry.dt.year.eq(2024), entry.between(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"), inclusive="left"), entry.ge(pd.Timestamp("2025-07-01", tz="UTC"))], ["2024", "2025_h1", "2025_h2"], default="other")
    period = scenarios.groupby(["definition_lane", "period_scope", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(events=("event_key", "nunique"), definitions=("candidate_definition_id", "nunique"), median_event_net_R=("scenario_scaled_net_R", "median"), total_net_R=("scenario_scaled_net_R", "sum")).reset_index()
    concentration = run.concentration_preview(scenarios)
    exact_rows = []
    for lane_name, group in rescored.groupby("definition_lane", sort=True):
        exact_rows.append({"definition_lane": lane_name, "events": len(group), "events_with_one_or_more_exact_boundaries": int(group["exact_boundary_rows"].gt(0).sum()), "events_with_zero_funding_boundaries": int(group["funding_boundary_rows"].eq(0).sum()), "events_fully_covered_by_exact_funding": int(group["all_boundaries_exact"].sum()), "events_with_imputed_boundaries": int(group["imputed_boundary_rows"].gt(0).sum()), "exact_boundary_rows": int(group["exact_boundary_rows"].sum()), "imputed_boundary_rows": int(group["imputed_boundary_rows"].sum())})
    for path, frame in [
        ("aggregate/definition_scorecard_50shard.csv", scorecard), ("aggregate/lane_summary_50shard.csv", lane),
        ("aggregate/exit_policy_summary_50shard.csv", exits), ("aggregate/spec_robustness_summary.csv", spec),
        ("aggregate/period_support_summary.csv", period), ("forensics/concentration_preview.csv", concentration),
        ("audit/exact_slice_composition_audit.csv", exact_rows),
    ]:
        run.write_csv(root / path, frame)

    decisions = []
    for lane_name in balanced.LONG_LANES:
        severe = scorecard[(scorecard["definition_lane"] == lane_name) & (scorecard["funding_mode"] == "severe_imputed") & (scorecard["slippage_round_trip_bps"] == 8)]
        central = scorecard[(scorecard["definition_lane"] == lane_name) & (scorecard["funding_mode"] == "central_imputed") & (scorecard["slippage_round_trip_bps"] == 4)]
        positive_fraction = float((severe["total_net_R"] > 0).mean()) if len(severe) else 0.0
        severe_median = float(severe["total_net_R"].median()) if len(severe) else np.nan
        h2 = period[(period["definition_lane"] == lane_name) & (period["period_scope"] == "2025_h2") & (period["funding_mode"] == "severe_imputed") & (period["slippage_round_trip_bps"] == 8)]
        h2_positive = bool(len(h2) and h2["total_net_R"].iloc[0] > 0)
        if positive_fraction >= 0.5 and severe_median > 0 and h2_positive:
            decision = "advance_to_full_scan"
        elif positive_fraction > 0 or bool(len(central) and (central["total_net_R"] > 0).any()):
            decision = "preserve_for_full_scan_exploration"
        else:
            decision = "defer_current_translation_after_50shard"
        decisions.append({"definition_lane": lane_name, "decision": decision, "positive_definition_fraction_severe_funding_plus_8bps": positive_fraction, "median_definition_net_R_severe_funding_plus_8bps": severe_median, "support_2025_h2_severe_plus_8bps": h2_positive, "family_rejected": False})
    decision_table = pd.DataFrame(decisions)
    run.write_csv(root / "decision/lane_decision_table.csv", decision_table)
    peak = int(status["peak_rss_bytes"].max())
    runtime = float(status["runtime_seconds"].sum())
    projection = runtime * 180 / 50
    mapping_failures = int(sum(row["status"] != "pass" for row in binding_rows))
    hard_pass = canonical_mismatches == 0 and mapping_failures == 0 and missing == 0 and duplicate == 0 and int(status["decision_input_leak_violations"].sum()) == 0 and int(status["protected_interval_violations"].sum()) == 0 and int(status["static_topn_failures"].sum()) == 0 and peak < 8 * 1024**3
    summary = {"run_root": str(root), "status": "complete" if hard_pass else "blocked", "shards_completed": 50, "definitions_scored": int(events["candidate_definition_id"].nunique()), "events_scored": len(events), "canonical_hash_mismatch_count": canonical_mismatches, "selected_key_policy_contract_version": runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION, "funding_panel_extended": bool(extension["panel_extended"].any()), "funding_panel_refit": False, "missing_funding_joins": missing, "duplicate_funding_joins": duplicate, "funding_gate_mapping_failures": mapping_failures, "decision_input_leak_violations": int(status["decision_input_leak_violations"].sum()), "protected_period_violations": int(status["protected_interval_violations"].sum()), "static_topn_failures": int(status["static_topn_failures"].sum()), "peak_rss_bytes": peak, "runtime_seconds": runtime, "projected_full_180shard_runtime_seconds": projection, "selected_key_or_outcome_cache_reuse_rate": 0.0, "full_180_shard_scan_allowed": bool(hard_pass and decision_table["decision"].isin(["advance_to_full_scan", "preserve_for_full_scan_exploration"]).any()), "evidence_label": "train_only_aggregate_screen_funding_imputed_capped_not_validation", "reducer_only_continuation": True, "compact_bundle_path": str(root / "compact_review_bundle")}
    run.write_json(root / "decision_summary.json", summary)
    (root / "performance/runtime_report.md").write_text(f"# Runtime Report\n\nSum of shard runtime seconds: `{runtime:.3f}`\nPeak RSS bytes: `{peak}`\nReducer-only continuation seconds: `{time.monotonic()-started:.3f}`\n")
    (root / "performance/full_180shard_runtime_projection.md").write_text(f"# Full 180-Shard Projection\n\nMeasured shard runtime seconds: `{runtime:.3f}`\nLinear projection seconds: `{projection:.3f}`\nNot a portfolio-return estimate.\n")
    required = ["shards/selected_50_shard_plan.csv", "shards/shard_status_summary.csv", "audit/canonical_hash_lineage_audit.csv", "audit/parameter_diversity_audit.csv", "audit/exact_slice_composition_audit.csv", "audit/funding_boundary_join_audit.csv", "gates/funding_gate_binding_audit.csv", "gates/funding_gate_pre_post_event_counts.csv", "funding/panel_extension_audit.csv", "aggregate/definition_scorecard_50shard.csv", "aggregate/lane_summary_50shard.csv", "aggregate/exit_policy_summary_50shard.csv", "aggregate/spec_robustness_summary.csv", "aggregate/period_support_summary.csv", "forensics/concentration_preview.csv", "performance/runtime_report.md", "performance/full_180shard_runtime_projection.md", "decision/lane_decision_table.csv", "decision_summary.json"]
    bundle_rows = []
    for rel in required:
        source = root / rel; target = root / "compact_review_bundle" / rel.replace("/", "__")
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        bundle_rows.append({"source": rel, "bundle_path": str(target.relative_to(root)), "sha256": run.file_hash(target)})
    run.write_csv(root / "compact_review_bundle/compact_bundle_manifest.csv", bundle_rows)
    runner.write_json(root / "watch_status.json", {"run_root": str(root.resolve()), "status": summary["status"], "stage": "reducer-only-continuation-complete", "ts_utc": runner.utc_now()})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
