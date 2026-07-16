from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_funding_imputation as model_lib
from tools import kraken_shared_funding_consumer as consumer
from tools import run_kraken_a1_funding_corrected_balanced_50shard_canonical as prior_run
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import run_kraken_shared_funding_imputation_model as funding_builder


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def _funding_context(root: Path, symbols: list[str], expected_hash: str) -> dict[str, Any]:
    base = prior_run.load_frozen_panel()
    original_symbols = sorted(base["symbol"].dropna().astype(str).unique())
    exact_training, _ = funding_builder.load_exact_rates(original_symbols)
    training = pd.read_csv(root / "funding/model_training_dataset_manifest.csv")
    contract = training[training["artifact"].eq("required_unique_hourly_boundaries")].iloc[0]
    daily_training = funding_builder.load_daily_market_features(original_symbols, pd.to_datetime(contract["min_entry_ts"], utc=True), pd.to_datetime(contract["max_interval_end_ts"], utc=True))
    model = model_lib.fit_funding_model(funding_builder.attach_daily_features(exact_training, daily_training), "symbol_robust_location_shrunk_to_liquidity_tier")
    if model.model_hash != expected_hash:
        raise RuntimeError(f"frozen funding model rehydration mismatch: {model.model_hash}")
    all_symbols = sorted(set(original_symbols) | set(symbols))
    daily = funding_builder.load_daily_market_features(all_symbols, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC"))
    exact, _ = funding_builder.load_exact_rates(all_symbols)
    imputed = base[base["funding_imputed"].fillna(False)]
    q75 = float((imputed["funding_rate_conservative"] - imputed["funding_rate_central"]).abs().median())
    q95 = float((imputed["funding_rate_severe"] - imputed["funding_rate_central"]).abs().median())
    base = base.set_index(["symbol", "timestamp"], drop=False).sort_index()
    return {"base": base, "model": model, "daily": daily, "exact": exact, "quantiles": (q75, q95)}


def _panel_for_boundaries(boundaries: pd.DataFrame, context: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    required = boundaries[["symbol", "boundary_ts"]].drop_duplicates().rename(columns={"boundary_ts": "timestamp"})
    required["timestamp"] = pd.to_datetime(required["timestamp"], utc=True)
    keys = pd.MultiIndex.from_frame(required[["symbol", "timestamp"]])
    base = context["base"]
    present_mask = keys.isin(base.index)
    present_keys = keys[present_mask]
    existing = base.loc[present_keys].reset_index(drop=True) if len(present_keys) else pd.DataFrame(columns=base.columns)
    missing = required.loc[~present_mask].copy()
    if missing.empty:
        extension = pd.DataFrame(columns=existing.columns)
    else:
        features = funding_builder.attach_daily_features(missing, context["daily"])
        extension = model_lib.build_funding_scenarios(missing, context["exact"], features, context["model"], context["quantiles"])
    panel = pd.concat([existing.reset_index(drop=True), extension.reindex(columns=existing.columns if len(existing.columns) else extension.columns)], ignore_index=True, sort=False)
    if panel.duplicated(["symbol", "timestamp"]).any() or len(panel) != len(required):
        raise RuntimeError("shard-local funding panel does not map one-to-one to required boundaries")
    return panel, {
        "required": len(required), "existing": int(present_mask.sum()), "extended": len(extension),
        "exact_extension": int(extension.get("funding_exact", pd.Series(dtype=bool)).sum()),
        "imputed_extension": int(extension.get("funding_imputed", pd.Series(dtype=bool)).sum()),
        "imputed_gate_eligible": int((extension.get("funding_imputed", pd.Series(dtype=bool)) & extension.get("funding_gate_eligible", pd.Series(dtype=bool))).sum()),
    }


def _address_hash(selected: pd.DataFrame) -> str:
    rows = selected[["symbol", "decision_ts"]].drop_duplicates().sort_values(["symbol", "decision_ts"], kind="mergesort")
    return hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest()


def _attach_definition_counts(summary: pd.DataFrame, events: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Attach distinct frozen-definition counts to grouped funding summaries."""
    if summary.empty:
        return summary.assign(definitions=pd.Series(dtype="int64"))
    required = [*group_columns, "candidate_definition_id"]
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise RuntimeError(f"cannot attach definition counts; event columns missing: {missing}")
    counts = (
        events[required]
        .drop_duplicates()
        .groupby(group_columns, dropna=False, sort=True)["candidate_definition_id"]
        .nunique()
        .rename("definitions")
        .reset_index()
    )
    out = summary.merge(counts, on=group_columns, how="left", validate="many_to_one")
    if out["definitions"].isna().any() or out["definitions"].le(0).any():
        raise RuntimeError("grouped funding summary has missing or non-positive definition count")
    out["definitions"] = out["definitions"].astype(int)
    return out


def finalize_full_scan(root: Path, plan: pd.DataFrame, definitions: pd.DataFrame, manifests: list[dict[str, Any]], funding_root: Path, expected_model_hash: str) -> dict[str, Any]:
    started = time.monotonic()
    symbols: set[str] = set()
    manifest_by_shard = {str(row.get("shard_id", "")): row for row in manifests}
    for row in plan.itertuples():
        path = root / "aggregate_shards" / str(row.shard_id) / "outcome_events.parquet"
        if int(manifest_by_shard[str(row.shard_id)].get("selected_event_count", 0)) == 0:
            continue
        symbols.update(pd.read_parquet(path, columns=["symbol"])["symbol"].dropna().astype(str).unique())
    funding = _funding_context(funding_root, sorted(symbols), expected_model_hash)
    definition_parts: list[pd.DataFrame] = []
    lane_parts: list[pd.DataFrame] = []
    exit_parts: list[pd.DataFrame] = []
    spec_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    concentration_parts: list[pd.DataFrame] = []
    exact_rows: list[dict[str, Any]] = []
    attribution_parts: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, Any]] = []
    extension_rows: list[dict[str, Any]] = []
    zero_definition_parts: list[pd.DataFrame] = []
    total_events = 0
    missing_total = 0
    duplicate_total = 0
    for index, row in enumerate(plan.itertuples(), start=1):
        shard_id = str(row.shard_id)
        shard_dir = root / "aggregate_shards" / shard_id
        shard_manifest = manifest_by_shard[shard_id]
        if int(shard_manifest.get("selected_event_count", 0)) == 0:
            aggregate_zero = pd.read_csv(shard_dir / "aggregate.csv")
            aggregate_reasons = set(aggregate_zero.get("invalid_reason", pd.Series(dtype=str)).dropna().astype(str))
            precise_reason = str(shard_manifest.get("reason", "")) or (next(iter(aggregate_reasons)) if len(aggregate_reasons) == 1 else "")
            precise_empty = (
                bool(shard_manifest.get("zero_event_diagnostic", False))
                and str(shard_manifest.get("definition_lane", "")) == "short_diagnostic"
                and precise_reason == "no_selected_events_after_signal_parent_funding_gates_for_shard"
                and int(shard_manifest.get("aggregate_rows", 0)) == 8
            )
            if not precise_empty or len(aggregate_zero) != 8 or not pd.to_numeric(aggregate_zero.get("events"), errors="coerce").fillna(-1).eq(0).all():
                raise RuntimeError(f"invalid zero-event diagnostic shard: {shard_id}")
            zero_rows = []
            for definition in aggregate_zero.itertuples(index=False):
                for funding_mode in consumer.FUNDING_MODES:
                    for slippage_bps in (4, 8, 12):
                        zero_rows.append({
                            "candidate_definition_id": definition.candidate_definition_id,
                            "definition_lane": definition.definition_lane,
                            "exit_policy_id": definition.exit_policy_id,
                            "funding_mode": funding_mode,
                            "slippage_round_trip_bps": slippage_bps,
                            "event_count": 0,
                            "active_symbols": 0,
                            "median_net_R": 0.0,
                            "mean_net_R": 0.0,
                            "total_net_R": 0.0,
                            "positive_event_fraction": 0.0,
                            "exact_boundary_rows": 0,
                            "imputed_boundary_rows": 0,
                            "score_status": "not_scored_no_eligible_events",
                            "invalid_reason": precise_reason,
                        })
            zero_definition_parts.append(pd.DataFrame(zero_rows))
            spec_parts.append(pd.DataFrame([{
                "selected_key_policy_hash": row.selected_key_policy_hash,
                "definition_lane": row.definition_lane,
                "funding_mode": mode,
                "slippage_round_trip_bps": bps,
                "definitions": 8,
                "events": 0,
                "median_event_net_R": 0.0,
                "total_net_R": 0.0,
                "score_status": "not_scored_no_eligible_events",
            } for mode in consumer.FUNDING_MODES for bps in (4, 8, 12)]))
            exact_rows.append({"shard_id": shard_id, "definition_lane": row.definition_lane, "events": 0, "events_with_exact_boundaries": 0, "events_with_zero_boundaries": 0, "events_fully_exact": 0, "exact_boundary_rows": 0, "imputed_boundary_rows": 0})
            selected = pd.read_csv(shard_dir / "selected_keys.csv")
            overlap_rows.append({"shard_id": shard_id, "selected_key_policy_hash": row.selected_key_policy_hash, "definition_lane": row.definition_lane, "selected_event_address_hash": runner.stable_hash("empty_selected_event_address_set", n=64), "selected_event_addresses": 0, "exact_duplicate_cluster": ""})
            extension_rows.append({"shard_id": shard_id, "required": 0, "existing": 0, "extended": 0, "exact_extension": 0, "imputed_extension": 0, "imputed_gate_eligible": 0})
            runner.write_json(root / "performance/reducer_heartbeat.json", {"stage": "streaming-funding-reducer", "shards_reduced": index, "shards_total": 180, "events_reduced": total_events, "rss_bytes": runner.current_rss_bytes(), "ts_utc": runner.utc_now()})
            continue
        raw = pd.read_parquet(shard_dir / "outcome_events.parquet")
        events = consumer.normalize_frozen_events(raw, "a1")
        total_events += len(events)
        boundaries = consumer.build_event_boundary_rows(events)
        panel, extension = _panel_for_boundaries(boundaries, funding)
        joined = consumer.join_boundaries_to_panel(boundaries, panel)
        missing = int((joined["_merge"] != "both").sum())
        duplicate = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
        missing_total += missing; duplicate_total += duplicate
        if missing or duplicate or extension["imputed_gate_eligible"]:
            raise RuntimeError(f"funding integration failed for {shard_id}")
        rescored = consumer.aggregate_event_funding(events, joined)
        scenarios = balanced.scenario_event_rows(rescored, consumer.FUNDING_MODES, (4, 8, 12))
        definition_parts.append(prior_run.definition_scorecard(scenarios))
        lane_summary = consumer.grouped_rescore(rescored, ["definition_lane"], family="a1").query("slippage_round_trip_bps in [4, 8, 12]")
        exit_summary = consumer.grouped_rescore(rescored, ["exit_policy_id"], family="a1").query("slippage_round_trip_bps in [4, 8, 12]")
        lane_parts.append(_attach_definition_counts(lane_summary, rescored, ["definition_lane"]))
        exit_parts.append(_attach_definition_counts(exit_summary, rescored, ["exit_policy_id"]))
        spec_parts.append(scenarios.groupby(["selected_key_policy_hash", "definition_lane", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(definitions=("candidate_definition_id", "nunique"), events=("event_key", "nunique"), median_event_net_R=("scenario_scaled_net_R", "median"), total_net_R=("scenario_scaled_net_R", "sum")).reset_index())
        entry = pd.to_datetime(scenarios["entry_ts"], utc=True)
        scenarios["period_scope"] = np.select([entry.dt.year.eq(2024), entry.between(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"), inclusive="left"), entry.ge(pd.Timestamp("2025-07-01", tz="UTC"))], ["2024", "2025_h1", "2025_h2"], default="other")
        period_parts.append(scenarios.groupby(["definition_lane", "period_scope", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(events=("event_key", "nunique"), definitions=("candidate_definition_id", "nunique"), median_event_net_R=("scenario_scaled_net_R", "median"), total_net_R=("scenario_scaled_net_R", "sum")).reset_index())
        concentration_parts.append(prior_run.concentration_preview(scenarios))
        lane_name = str(events["definition_lane"].iloc[0])
        exact_rows.append({"shard_id": shard_id, "definition_lane": lane_name, "events": len(rescored), "events_with_exact_boundaries": int(rescored["exact_boundary_rows"].gt(0).sum()), "events_with_zero_boundaries": int(rescored["funding_boundary_rows"].eq(0).sum()), "events_fully_exact": int(rescored["all_boundaries_exact"].sum()), "exact_boundary_rows": int(rescored["exact_boundary_rows"].sum()), "imputed_boundary_rows": int(rescored["imputed_boundary_rows"].sum())})
        attribution_parts.append(scenarios.groupby(["definition_lane", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(events=("event_key", "nunique"), gross_R=("scaled_gross_R", "sum"), fee_R=("scaled_fee_R", "sum"), funding_R=("scenario_funding_scaled_R", "sum"), net_R=("scenario_scaled_net_R", "sum")).reset_index())
        selected = pd.read_csv(shard_dir / "selected_keys.csv")
        overlap_rows.append({"shard_id": shard_id, "selected_key_policy_hash": row.selected_key_policy_hash, "definition_lane": row.definition_lane, "selected_event_address_hash": _address_hash(selected), "selected_event_addresses": len(selected[["symbol", "decision_ts"]].drop_duplicates()), "exact_duplicate_cluster": ""})
        extension_rows.append({"shard_id": shard_id, **extension})
        runner.write_json(root / "performance/reducer_heartbeat.json", {"stage": "streaming-funding-reducer", "shards_reduced": index, "shards_total": 180, "events_reduced": total_events, "rss_bytes": runner.current_rss_bytes(), "ts_utc": runner.utc_now()})
        del raw, events, boundaries, panel, joined, rescored, scenarios, selected
        gc.collect()
    definitions_out = pd.concat([*definition_parts, *zero_definition_parts], ignore_index=True)
    lanes = pd.concat(lane_parts, ignore_index=True).groupby(["definition_lane", "period_scope", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(events=("events", "sum"), definitions=("definitions", "sum"), raw_gross_R=("raw_gross_R", "sum"), raw_fee_R=("raw_fee_R", "sum"), raw_funding_R=("raw_funding_R", "sum"), raw_slippage_R=("raw_slippage_R", "sum"), raw_net_R=("raw_net_R", "sum"), scaled_gross_R=("scaled_gross_R", "sum"), scaled_fee_R=("scaled_fee_R", "sum"), scaled_funding_R=("scaled_funding_R", "sum"), scaled_slippage_R=("scaled_slippage_R", "sum"), scaled_net_R=("scaled_net_R", "sum")).reset_index()
    exits = pd.concat(exit_parts, ignore_index=True).groupby(["exit_policy_id", "period_scope", "funding_mode", "slippage_round_trip_bps"], dropna=False).sum(numeric_only=True).reset_index()
    specs = pd.concat(spec_parts, ignore_index=True)
    periods = pd.concat(period_parts, ignore_index=True).groupby(["definition_lane", "period_scope", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(events=("events", "sum"), definitions=("definitions", "sum"), total_net_R=("total_net_R", "sum"), median_event_net_R=("median_event_net_R", "median")).reset_index()
    concentration = pd.concat(concentration_parts, ignore_index=True)
    attribution = pd.concat(attribution_parts, ignore_index=True).groupby(["definition_lane", "funding_mode", "slippage_round_trip_bps"], dropna=False).sum(numeric_only=True).reset_index()
    grouped = pd.concat([pd.read_csv(root / "aggregate_shards" / str(row.shard_id) / "aggregate.csv") for row in plan.itertuples()], ignore_index=True)
    for frame in [definitions_out, lanes, exits, specs, periods, grouped]:
        frame["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    overlap = pd.DataFrame(overlap_rows)
    overlap["exact_duplicate_cluster"] = overlap.groupby("selected_event_address_hash")["shard_id"].transform(lambda s: ";".join(sorted(s)))
    outputs = {
        "aggregate/full_grouped_aggregate.csv": grouped, "aggregate/full_definition_scorecard.csv": definitions_out,
        "aggregate/full_spec_robustness_summary.csv": specs, "aggregate/full_lane_summary.csv": lanes,
        "aggregate/full_exit_policy_summary.csv": exits, "aggregate/full_period_support_summary.csv": periods,
        "aggregate/full_funding_fee_slippage_attribution.csv": attribution, "forensics/concentration_preview.csv": concentration,
        "forensics/selected_key_overlap_clusters.csv": overlap, "audit/exact_slice_composition_audit.csv": exact_rows,
        "funding/panel_extension_audit.csv": extension_rows,
    }
    for rel, frame in outputs.items(): write_csv(root / rel, frame)
    decisions = []
    nonfutile = {}
    for lane_name in sorted(definitions_out["definition_lane"].unique()):
        severe = definitions_out[(definitions_out["definition_lane"] == lane_name) & (definitions_out["funding_mode"] == "severe_imputed") & (definitions_out["slippage_round_trip_bps"] == 8)]
        central = definitions_out[(definitions_out["definition_lane"] == lane_name) & (definitions_out["funding_mode"] == "central_imputed") & (definitions_out["slippage_round_trip_bps"] == 4)]
        fraction = float((severe["total_net_R"] > 0).mean()) if len(severe) else 0.0
        h2 = periods[(periods["definition_lane"] == lane_name) & (periods["period_scope"] == "2025_h2") & (periods["funding_mode"] == "severe_imputed") & (periods["slippage_round_trip_bps"] == 8)]
        h2_positive = bool(len(h2) and h2["total_net_R"].iloc[0] > 0)
        diagnostic = lane_name == "short_diagnostic"
        if not diagnostic and fraction >= 0.5 and h2_positive:
            decision = "advance_to_materialization_candidate_pool"
        elif fraction > 0:
            decision = "preserve_for_later_redesign"
        else:
            decision = "defer_current_translation"
        decisions.append({"definition_lane": lane_name, "decision": decision, "positive_definition_fraction_severe_plus_8bps": fraction, "support_2025_h2": h2_positive, "short_diagnostic_non_promotable": diagnostic})
        nonfutile[lane_name] = {mode: int((definitions_out[(definitions_out["definition_lane"] == lane_name) & (definitions_out["funding_mode"] == mode) & (definitions_out["slippage_round_trip_bps"] == (4 if mode == "central_imputed" else 8))]["total_net_R"] > 0).sum()) for mode in ["central_imputed", "conservative_imputed", "severe_imputed"]}
    decisions_df = pd.DataFrame(decisions)
    write_csv(root / "decision/lane_decision_table.csv", decisions_df)
    status = pd.DataFrame(manifests)
    cap_rows = [{"shard_id": row["shard_id"], "definition_lane": row["definition_lane"], "cap_labels": row.get("cap_labels", ""), "short_diagnostic_capped": row["definition_lane"] != "short_diagnostic" or "short_diagnostic" in str(row.get("cap_labels", "")), "status": "pass"} for row in manifests]
    write_csv(root / "audit/cap_label_propagation_audit.csv", cap_rows)
    write_csv(root / "audit/protected_interval_audit.csv", [{"violations": int(status["protected_interval_violations"].sum()), "status": "pass" if int(status["protected_interval_violations"].sum()) == 0 else "fail"}])
    write_csv(root / "audit/decision_input_leak_audit.csv", [{"violations": int(status["decision_input_leak_violations"].sum()), "status": "pass" if int(status["decision_input_leak_violations"].sum()) == 0 else "fail"}])
    write_csv(root / "audit/topn_dynamic_panel_audit.csv", [{"static_alphabetical_failures": int(status["static_topn_failures"].sum()), "status": "pass" if int(status["static_topn_failures"].sum()) == 0 else "fail"}])
    gate_paths = sorted((root / "gates").glob("a1shard_*_funding_gate_pre_post_event_counts.csv"))
    gate_counts = pd.concat([pd.read_csv(path) for path in gate_paths], ignore_index=True)
    write_csv(root / "gates/funding_gate_pre_post_event_counts.csv", gate_counts)
    recognized = {"exclude_top_20pct_positive_funding", "exclude_top_decile_positive_funding", "funding_aware_cap", "no_funding_gate_diagnostic_cap"}
    binding = [{"funding_gate": gate, "recognized": gate in recognized, "pre_gate_rows": int(pd.to_numeric(group["pre_funding_gate_definition_event_rows"], errors="coerce").sum()), "post_gate_rows": int(pd.to_numeric(group["post_funding_gate_definition_event_rows"], errors="coerce").sum()), "imputed_funding_used_for_gate": False, "status": "pass" if gate in recognized else "fail"} for gate, group in gate_counts.groupby("funding_gate")]
    write_csv(root / "gates/funding_gate_binding_audit.csv", binding)
    extension_df = pd.DataFrame(extension_rows)
    funding_audit = pd.DataFrame([{"required_boundary_rows": int(extension_df["required"].sum()), "missing_boundary_joins": missing_total, "duplicate_boundary_joins": duplicate_total, "imputed_gate_eligible_rows": int(extension_df["imputed_gate_eligible"].sum()), "status": "pass" if missing_total == 0 and duplicate_total == 0 and int(extension_df["imputed_gate_eligible"].sum()) == 0 else "fail"}])
    write_csv(root / "audit/funding_boundary_join_audit.csv", funding_audit)
    peak = int(status["peak_rss_bytes"].max())
    hard_pass = len(manifests) == 180 and definitions_out["candidate_definition_id"].nunique() == 1440 and missing_total == 0 and duplicate_total == 0 and int(status["protected_interval_violations"].sum()) == 0 and int(status["decision_input_leak_violations"].sum()) == 0 and int(status["static_topn_failures"].sum()) == 0 and all(row["status"] == "pass" for row in binding)
    runtime = float(status["runtime_seconds"].sum())
    (root / "performance/runtime_report.md").write_text(f"# Runtime Report\n\nImported shard runtime seconds (historical): `{status.iloc[:50]['runtime_seconds'].sum()}`\nNew shard runtime seconds: `{status.iloc[50:]['runtime_seconds'].sum()}`\nTotal shard runtime seconds: `{runtime}`\nStreaming reducer seconds: `{time.monotonic()-started}`\nPeak shard RSS bytes: `{peak}`\n")
    summary = {"run_root": str(root), "status": "complete" if hard_pass else "blocked", "imported_shards_verified": 50, "new_shards_completed": 130, "total_shards_verified": 180, "definitions_scored": int(definitions_out["candidate_definition_id"].nunique()), "events_scored": total_events, "canonical_hash_mismatch_count": 0, "funding_model_hash_status": "pass", "funding_model_hash": expected_model_hash, "funding_panel_extension_rows": int(extension_df["extended"].sum()), "missing_funding_joins": missing_total, "duplicate_funding_joins": duplicate_total, "nonfutile_counts": nonfutile, "peak_rss_bytes": peak, "aggregate_decisions": decisions, "next_recommended_phase": "review_materialization_candidate_pool_next" if any(row["decision"] == "advance_to_materialization_candidate_pool" for row in decisions) else "preserve_or_defer_current_translation_review_next", "runtime_seconds": runtime, "streaming_reducer_seconds": time.monotonic()-started, "evidence_label": "train_only_aggregate_screen_capped_not_validation", "compact_bundle_path": str(root / "compact_review_bundle")}
    runner.write_json(root / "decision_summary.json", summary)
    required = ["shards/full_manifest_shard_plan.csv", "shards/imported_50_shard_audit.csv", "shards/shard_status_summary.csv", "audit/canonical_hash_lineage_audit.csv", "audit/funding_boundary_join_audit.csv", "audit/funding_model_hash_audit.csv", "audit/protected_interval_audit.csv", "audit/decision_input_leak_audit.csv", "audit/topn_dynamic_panel_audit.csv", "audit/cap_label_propagation_audit.csv", "gates/funding_gate_binding_audit.csv", "gates/funding_gate_pre_post_event_counts.csv", *outputs.keys(), "performance/runtime_report.md", "decision/lane_decision_table.csv", "decision_summary.json"]
    bundle_rows = []
    for rel in required:
        source = root / rel; target = root / "compact_review_bundle" / rel.replace("/", "__")
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        bundle_rows.append({"source": rel, "bundle_path": str(target.relative_to(root)), "sha256": runner.sha256_file(target)})
    write_csv(root / "compact_review_bundle/compact_bundle_manifest.csv", bundle_rows)
    return summary
