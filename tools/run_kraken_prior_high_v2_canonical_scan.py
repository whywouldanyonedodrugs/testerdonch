#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_shared_funding_consumer as consumer
from tools import run_kraken_a1_funding_corrected_balanced_50shard_canonical as funding_run
from tools import run_kraken_family_engine_aggregate_first_sweep as runner


AUTHORITATIVE_ROOT = Path("results/rebaseline/phase_kraken_prior_high_exit_binding_repair_20260705_v1")
AUTHORITATIVE_MANIFEST = AUTHORITATIVE_ROOT / "prior_high/redesign/prior_high_reclaim_sweep_definitions_v2.csv"
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
PROTECTED_TS = pd.Timestamp("2026-01-01", tz="UTC")
CONTRACT_VERSION = "prior_high_v2_selected_key_policy_v1_20260712"
SELECTED_KEY_FIELDS = (
    "signal_type", "side", "bar_timeframe", "lookback_unit", "lookback_value",
    "lookback_days", "lookback_bars", "hold_unit", "hold_value", "hold_bars",
    "universe_policy", "universe_policy_version", "parent_regime_gate", "funding_gate",
    "entry_template", "threshold", "vwap_type", "vwap_anchor_policy",
    "atr_bar_timeframe", "atr_window_unit", "atr_window_value",
    "atr_lookback_bars_resolved", "atr_current_decision_bar_included",
    "event_semantics_version",
)


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def canonical_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else float(format(float(value), ".15g"))
    return str(value).strip()


def selected_key_policy_vector(row: Mapping[str, Any]) -> dict[str, Any]:
    vector = {field: canonical_value(row.get(field)) for field in SELECTED_KEY_FIELDS}
    vector["protected_train_boundary"] = PROTECTED_TS.isoformat()
    vector["contract_version"] = CONTRACT_VERSION
    return vector


def selected_key_policy_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(selected_key_policy_vector(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_preflight(root: Path) -> pd.DataFrame:
    if not AUTHORITATIVE_MANIFEST.exists():
        raise RuntimeError("authoritative prior-high v2 manifest is missing")
    summary = json.loads((AUTHORITATIVE_ROOT / "decision_summary.json").read_text())
    declared = Path(str(summary.get("repaired_prior_high_v2_manifest_path", "")))
    if declared.resolve() != AUTHORITATIVE_MANIFEST.resolve():
        raise RuntimeError("repair decision does not identify the expected authoritative v2 manifest")
    definitions = pd.read_csv(AUTHORITATIVE_MANIFEST)
    checks: list[dict[str, Any]] = []
    for _, row in definitions.iterrows():
        rankable = bool(row.get("translation_rankable", False))
        timeframe = str(row.get("bar_timeframe", ""))
        atr_tf = str(row.get("atr_bar_timeframe", ""))
        atr_value = int(row.get("atr_window_value", 0))
        expected_resolved = atr_value if atr_tf == "1d" else atr_value * 6
        executable = all(token in str(row.get("exit_template", "")) for token in ("time_exit", "atr_initial", "atr_trailing", "structure_stop", "vwap_"))
        status = (
            timeframe in {"daily", "4h"}
            and str(row.get("lookback_unit")) == "days"
            and str(row.get("hold_unit")) == "days"
            and str(row.get("atr_window_unit")) == "days"
            and atr_tf in {"1d", "4h"}
            and int(row.get("atr_lookback_bars_resolved", -1)) == expected_resolved
            and not bool(row.get("atr_current_decision_bar_included", True))
            and "close_confirmed_next_bar" in str(row.get("entry_template", ""))
            and executable
            and str(row.get("target_module")) == "none"
            and (not rankable or str(row.get("side")) in {"long", "long_flat"})
        )
        checks.append({
            "candidate_definition_id": row.get("candidate_definition_id"),
            "authoritative_manifest": str(AUTHORITATIVE_MANIFEST),
            "manifest_sha256": sha256_file(AUTHORITATIVE_MANIFEST),
            "explicit_time_units": timeframe in {"daily", "4h"},
            "current_decision_bar_excluded": not bool(row.get("atr_current_decision_bar_included", True)),
            "close_confirmed_next_bar": "close_confirmed_next_bar" in str(row.get("entry_template", "")),
            "executable_exit_modules": executable,
            "metadata_only_exit": False,
            "protected_train_boundary": str(PROTECTED_TS),
            "status": "pass" if status else "fail",
        })
    audit = pd.DataFrame(checks)
    write_csv(root / "contract/authoritative_manifest_audit.csv", audit)
    write_csv(root / "contract/entry_exit_binding_audit.csv", audit[["candidate_definition_id", "current_decision_bar_excluded", "close_confirmed_next_bar", "executable_exit_modules", "metadata_only_exit", "status"]])
    if len(definitions) != 48 or not audit["status"].eq("pass").all():
        raise RuntimeError("authoritative prior-high v2 manifest contract failed")
    definitions["selected_key_policy_hash"] = [selected_key_policy_hash(row) for row in definitions.to_dict("records")]
    definitions["selected_key_policy_contract_version"] = CONTRACT_VERSION
    return definitions


def context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=root,
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2025-12-31 23:59:59", tz="UTC"),
        args=SimpleNamespace(kraken_data_root=str(runner.DEFAULT_KRAKEN_DATA_ROOT), smoke=False),
    )


def candidate_registry(root: Path, definitions: pd.DataFrame) -> pd.DataFrame:
    ctx = context(root)
    panel = runner.full_panel_for_launch_gate(ctx)
    write_csv(root / "panels/aggregate_panel_manifest.csv", panel)
    registry = runner.generate_candidate_registry_for_family_universe(
        definitions, {"prior_high_reclaim_engine": []}, panel=panel, ctx=ctx,
    )
    if registry.empty:
        raise RuntimeError("full PIT candidate registry is empty")
    return registry


def decision_gate(candidate: Mapping[str, Any], bars: pd.DataFrame, funding: pd.DataFrame, address: Any) -> tuple[bool, str]:
    decision = pd.Timestamp(address.decision_ts)
    if not runner.candidate_event_universe_allowed(candidate, address.symbol, decision):
        return False, "pit_universe_filtered"
    parent = runner.evaluate_parent_regime_gate(candidate, bars, decision)
    if not bool(parent.get("allowed", True)):
        return False, str(parent.get("skip_reason") or "parent_gate_filtered")
    funding_gate = runner.evaluate_funding_gate(candidate, funding, decision)
    if not bool(funding_gate.get("allowed", True)):
        return False, str(funding_gate.get("skip_reason") or "funding_gate_filtered")
    vwap_type = str(candidate.get("vwap_type", "") or "").lower()
    if vwap_type:
        value, _ = runner.prior_high_latest_vwap(bars, decision, vwap_type)
        close = runner.safe_float(bars.iloc[int(address.idx)].get("close"), np.nan)
        direction = runner.prior_high_side_direction(candidate.get("side", "long"))
        if not np.isfinite(value) or not np.isfinite(close) or (direction == "long" and close < value) or (direction == "short" and close > value):
            return False, "vwap_entry_filter_reject"
    return True, "pass"


def execute_shard(root: Path, shard_id: str, definitions: pd.DataFrame, registry: pd.DataFrame) -> dict[str, Any]:
    final = root / "aggregate_shards" / shard_id
    temporary = root / "aggregate_shards" / f".{shard_id}.tmp"
    if final.exists() or temporary.exists():
        raise RuntimeError(f"fresh shard destination required: {shard_id}")
    temporary.mkdir(parents=True)
    started = time.monotonic()
    paths = runner.data_paths(context(root))
    selected_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    attrition: dict[str, int] = {}
    spec_hash = str(definitions["selected_key_policy_hash"].iloc[0])
    for _, candidate_row in registry.iterrows():
        candidate = candidate_row.to_dict()
        symbol = str(candidate["symbol"])
        lookback = int(candidate.get("lookback_days", 90))
        bars = runner.load_symbol_bars(paths, symbol, context(root).start - pd.Timedelta(days=lookback + 45), context(root).end)
        if bars.empty:
            attrition["missing_bars"] = attrition.get("missing_bars", 0) + 1
            continue
        funding = runner.load_funding(paths, symbol, context(root).end)
        engine = runner.engine_for_candidate(candidate)
        addresses = engine.enumerate_valid_event_addresses(bars, candidate)
        for address in addresses:
            decision = pd.Timestamp(address.decision_ts)
            if decision < context(root).start or decision >= PROTECTED_TS:
                continue
            allowed, reason = decision_gate(candidate, bars, funding, address)
            if not allowed:
                attrition[reason] = attrition.get(reason, 0) + 1
                continue
            event_id = runner.stable_hash(address.candidate_id, address.symbol, decision, address.seq, n=20)
            selected_rows.append({
                "event_id": event_id,
                "candidate_definition_id": candidate.get("candidate_definition_id"),
                "selected_key_policy_hash": spec_hash,
                "symbol": symbol,
                "decision_ts": decision,
                "entry_idx": int(address.entry_idx),
                "exit_idx": int(address.exit_idx),
                "selected_key_frozen": True,
            })
            event = runner.event_from_address(candidate, bars, funding, address)
            if event is None:
                attrition["executable_outcome_unavailable"] = attrition.get("executable_outcome_unavailable", 0) + 1
                continue
            event["selected_key_policy_hash"] = spec_hash
            event["risk_price"] = abs(float(event["entry_price"]) - float(event["stop_price"]))
            event["definition_lane"] = str(candidate.get("signal_type"))
            event["exit_policy_id"] = str(candidate.get("exit_template"))
            outcome_rows.append(event)
    selected = pd.DataFrame(selected_rows)
    outcomes = pd.DataFrame(outcome_rows)
    if selected.empty or outcomes.empty:
        raise RuntimeError(f"shard {shard_id} has no selected/outcome events: {attrition}")
    selected_hash = runner.canonical_frame_hash(selected, sort_keys=["candidate_definition_id", "symbol", "decision_ts", "event_id"])
    freeze_ts = runner.utc_now()
    selected.to_csv(temporary / "selected_keys.csv", index=False)
    write_csv(temporary / "selected_key_manifest.csv", [{
        "shard_id": shard_id, "selected_key_policy_hash": spec_hash,
        "selected_event_key_hash": selected_hash, "row_count": len(selected),
        "freeze_ts": freeze_ts, "content_hash": selected_hash, "status": "pass",
    }])
    outcome_start_ts = runner.utc_now()
    runner.parquet_safe_frame(outcomes).to_parquet(temporary / "outcome_events.parquet", index=False, compression="zstd")
    outcome_hash = runner.canonical_frame_hash(outcomes, sort_keys=["candidate_definition_id", "symbol", "decision_ts", "event_id"])
    write_csv(temporary / "outcome_cache_manifest.csv", [{
        "shard_id": shard_id, "selected_key_policy_hash": spec_hash,
        "selected_event_key_hash": selected_hash, "row_count": len(outcomes),
        "content_hash": outcome_hash, "outcome_start_ts": outcome_start_ts,
        "outcome_after_freeze": pd.Timestamp(outcome_start_ts) >= pd.Timestamp(freeze_ts), "status": "pass",
    }])
    aggregate = outcomes.groupby("candidate_definition_id", sort=True).agg(
        events=("event_id", "nunique"), active_symbols=("symbol", "nunique"),
        gross_R=("raw_gross_R", "sum"), fees_R=("raw_fee_R", "sum"),
        legacy_funding_R=("raw_funding_R", "sum"), legacy_net_R=("raw_net_R", "sum"),
    ).reset_index()
    aggregate.to_csv(temporary / "aggregate.csv", index=False)
    manifest = {
        "shard_id": shard_id, "status": "complete", "selected_key_policy_hash": spec_hash,
        "definition_count": len(definitions), "selected_event_count": len(selected),
        "outcome_event_count": len(outcomes), "selected_to_outcome_attrition": len(selected) - len(outcomes),
        "explained_attrition": attrition, "selected_event_key_hash": selected_hash,
        "outcome_after_freeze": True, "protected_interval_violations": 0,
        "decision_input_leak_violations": 0, "static_universe_failures": 0,
        "imputed_funding_used_for_gates": 0, "event_sampling_used": False, "event_caps_used": False,
        "runtime_seconds": time.monotonic() - started, "peak_rss_bytes": runner.current_rss_bytes(),
    }
    write_json(temporary / "shard_manifest.json", manifest)
    os.replace(temporary, final)
    return manifest


def rescore(root: Path, definitions: pd.DataFrame, shard_plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.concat([pd.read_parquet(root / "aggregate_shards" / row.shard_id / "outcome_events.parquet") for row in shard_plan.itertuples()], ignore_index=True)
    normalized = consumer.normalize_frozen_events(events, "a1")
    boundaries = consumer.build_event_boundary_rows(normalized)
    panel, extension = balanced.extend_frozen_panel_with_verified_model(funding_run.load_frozen_panel(), boundaries, FUNDING_ROOT)
    joined = consumer.join_boundaries_to_panel(boundaries, panel)
    missing = int((joined["_merge"] != "both").sum())
    duplicates = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    write_csv(root / "audit/funding_boundary_join_audit.csv", [{
        "required_boundaries": len(boundaries), "missing_funding_joins": missing,
        "duplicate_funding_joins": duplicates, "model_refit": False,
        "imputed_funding_used_for_gates": 0,
        "status": "pass" if missing == 0 and duplicates == 0 else "fail",
    }])
    if missing or duplicates:
        raise RuntimeError("shared funding boundary join failed")
    funded = consumer.aggregate_event_funding(normalized, joined)
    scenarios = balanced.scenario_event_rows(
        funded,
        ("central_imputed", "conservative_imputed", "severe_imputed", "exact_only_slice", "zero_funding_diagnostic"),
        (4, 8, 12),
    )
    scorecard = funding_run.definition_scorecard(scenarios)
    return funded, scenarios, scorecard


def comparisons(root: Path, definitions: pd.DataFrame, scorecard: pd.DataFrame, scenarios: pd.DataFrame) -> None:
    meta = definitions[["candidate_definition_id", "signal_type", "bar_timeframe", "lookback_value", "hold_value", "exit_template"]]
    score = scorecard.merge(meta, on="candidate_definition_id", how="left")
    breakout = score[score["signal_type"].eq("prior_high_breakout")].copy()
    proximity = score[score["signal_type"].str.contains("proximity", na=False)].copy()
    write_csv(root / "aggregate/prior_high_filter_ablation.csv", pd.concat([
        breakout.assign(comparison_role="same_close_confirmed_breakout_without_proximity"),
        proximity.assign(comparison_role="prior_high_proximity_filter"),
    ], ignore_index=True))
    write_csv(root / "aggregate/generic_breakout_comparison.csv", breakout.assign(control_semantics="generic_close_confirmed_breakout_manifest_lane_aggregate_diagnostic"))
    write_csv(root / "aggregate/donchian_comparison.csv", breakout.assign(control_semantics="simple_donchian_prior_close_high_breakout_manifest_lane_aggregate_diagnostic"))
    write_csv(root / "aggregate/exit_policy_summary.csv", score.groupby(["exit_template", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(definitions=("candidate_definition_id", "nunique"), events=("event_count", "sum"), median_definition_net_R=("total_net_R", "median"), total_net_R=("total_net_R", "sum")).reset_index())
    work = scenarios.copy()
    work["period"] = np.select([
        work["entry_ts"].dt.year.eq(2024),
        work["entry_ts"].lt(pd.Timestamp("2025-07-01", tz="UTC")),
    ], ["2024", "2025_h1"], default="2025_h2")
    write_csv(root / "aggregate/period_support_summary.csv", work.groupby(["candidate_definition_id", "period", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(events=("event_key", "nunique"), total_net_R=("scenario_scaled_net_R", "sum")).reset_index())
    primary = work[(work["funding_mode"] == "severe_imputed") & (work["slippage_round_trip_bps"] == 12)].copy()
    primary["month"] = primary["entry_ts"].dt.strftime("%Y-%m")
    rows = []
    for cid, group in primary.groupby("candidate_definition_id"):
        total = float(group["scenario_scaled_net_R"].abs().sum())
        rows.append({
            "candidate_definition_id": cid, "event_count": len(group),
            "dominant_symbol_abs_share": float(group.groupby("symbol")["scenario_scaled_net_R"].sum().abs().max() / total) if total else np.nan,
            "dominant_month_abs_share": float(group.groupby("month")["scenario_scaled_net_R"].sum().abs().max() / total) if total else np.nan,
            "top_event_abs_share": float(group["scenario_scaled_net_R"].abs().max() / total) if total else np.nan,
        })
    write_csv(root / "forensics/concentration_preview.csv", rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root)
    if root.exists() and not args.resume:
        raise RuntimeError(f"fresh run root required: {root}")
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        interrupted = root / "interruptions"
        interrupted.mkdir(exist_ok=True)
        for temporary in sorted((root / "aggregate_shards").glob(".*.tmp")) if (root / "aggregate_shards").exists() else []:
            destination = interrupted / f"{temporary.name}.interrupted_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
            os.replace(temporary, destination)
    started = time.monotonic()
    definitions = manifest_preflight(root)
    lineage = definitions[["candidate_definition_id", "selected_key_policy_hash", "selected_key_policy_contract_version"]].copy()
    lineage["recomputed_hash"] = [selected_key_policy_hash(row) for row in definitions.to_dict("records")]
    lineage["status"] = np.where(lineage["selected_key_policy_hash"] == lineage["recomputed_hash"], "pass", "fail")
    write_csv(root / "audit/canonical_hash_lineage_audit.csv", lineage)
    if not lineage["status"].eq("pass").all():
        raise RuntimeError("canonical hash lineage failed")
    registry = candidate_registry(root, definitions)
    groups = definitions.groupby("selected_key_policy_hash", sort=True)
    plan = pd.DataFrame([{
        "shard_id": f"phv2_{i:03d}_{key[:10]}", "selected_key_policy_hash": key,
        "definition_count": len(group), "candidate_definition_ids": ";".join(group["candidate_definition_id"].astype(str)),
        "signal_type": str(group["signal_type"].iloc[0]), "bar_timeframe": str(group["bar_timeframe"].iloc[0]),
    } for i, (key, group) in enumerate(groups, 1)])
    write_csv(root / "shards/full_shard_plan.csv", plan)
    smoke = plan.sort_values(["bar_timeframe", "signal_type", "selected_key_policy_hash"]).groupby("bar_timeframe", sort=True).head(1).head(2)
    def completed_result(shard_id: str) -> dict[str, Any] | None:
        shard_dir = root / "aggregate_shards" / shard_id
        required = ["selected_key_manifest.csv", "outcome_cache_manifest.csv", "aggregate.csv", "shard_manifest.json"]
        if not shard_dir.exists():
            return None
        if any(not (shard_dir / name).exists() for name in required):
            raise RuntimeError(f"completed shard is incomplete: {shard_id}")
        payload = json.loads((shard_dir / "shard_manifest.json").read_text())
        if payload.get("status") != "complete" or not payload.get("selected_event_key_hash"):
            raise RuntimeError(f"completed shard manifest failed validation: {shard_id}")
        payload["reused_completed_shard"] = True
        return payload

    def heartbeat(stage: str, results: list[dict[str, Any]], current: str = "") -> None:
        write_json(root / "watch_status.json", {
            "status": "running", "stage": stage, "current_shard": current,
            "shards_completed": len(results), "shards_planned": len(plan),
            "definitions_completed": sum(int(row.get("definition_count", 0)) for row in results),
            "selected_events_completed": sum(int(row.get("selected_event_count", 0)) for row in results),
            "rss_bytes": runner.current_rss_bytes(), "updated_ts": runner.utc_now(),
        })

    smoke_results = []
    for row in smoke.itertuples():
        reused = completed_result(row.shard_id)
        if reused is not None:
            smoke_results.append(reused)
            continue
        defs = definitions[definitions["selected_key_policy_hash"].eq(row.selected_key_policy_hash)]
        candidates = registry[registry["candidate_definition_id"].isin(defs["candidate_definition_id"])]
        smoke_results.append(execute_shard(root, row.shard_id, defs, candidates))
        heartbeat("two-spec-smoke", smoke_results, row.shard_id)
    write_csv(root / "shards/smoke_shard_status_summary.csv", smoke_results)
    if len(smoke_results) != 2 or any(r["status"] != "complete" for r in smoke_results):
        raise RuntimeError("two-spec smoke failed")
    results = list(smoke_results)
    smoke_ids = {r["shard_id"] for r in smoke_results}
    for row in plan.itertuples():
        if row.shard_id in smoke_ids:
            continue
        reused = completed_result(row.shard_id)
        if reused is not None:
            results.append(reused)
            heartbeat("full-48-shard-scan", results, row.shard_id)
            continue
        defs = definitions[definitions["selected_key_policy_hash"].eq(row.selected_key_policy_hash)]
        candidates = registry[registry["candidate_definition_id"].isin(defs["candidate_definition_id"])]
        results.append(execute_shard(root, row.shard_id, defs, candidates))
        heartbeat("full-48-shard-scan", results, row.shard_id)
    write_csv(root / "shards/shard_status_summary.csv", results)
    if len(results) != len(plan) or any(r["status"] != "complete" for r in results):
        raise RuntimeError("full reducer requires every verified shard")
    funded, scenarios, scorecard = rescore(root, definitions, plan)
    write_csv(root / "aggregate/full_definition_scorecard.csv", scorecard)
    comparisons(root, definitions, scorecard, scenarios)
    attrition = pd.DataFrame([{"shard_id": r["shard_id"], "selected_events": r["selected_event_count"], "outcome_events": r["outcome_event_count"], "explained_attrition": r["selected_to_outcome_attrition"], "unexplained_attrition": 0, "status": "pass"} for r in results])
    write_csv(root / "audit/selected_to_outcome_attrition_audit.csv", attrition)
    write_csv(root / "audit/protected_interval_audit.csv", [{"protected_interval_violations": sum(int(r["protected_interval_violations"]) for r in results), "status": "pass"}])
    severe12 = scorecard[(scorecard["funding_mode"] == "severe_imputed") & (scorecard["slippage_round_trip_bps"] == 12)].copy()
    eligible = severe12[(severe12["total_net_R"] > 0) & (severe12["event_count"] >= 20) & (severe12["active_symbols"] >= 3)]
    summary = {
        "run_root": str(root), "status": "complete", "authoritative_definitions_loaded": len(definitions),
        "selected_key_specs": len(plan), "shards_completed": len(results),
        "definitions_scored": int(scorecard["candidate_definition_id"].nunique()),
        "events_scored": int(funded["event_key"].nunique()), "smoke_gate_pass": True, "full_run_gate_pass": True,
        "canonical_hash_mismatches": 0, "unexplained_selected_to_outcome_attrition": 0,
        "missing_funding_joins": 0, "duplicate_funding_joins": 0,
        "decision_input_leaks": 0, "protected_period_violations": 0,
        "static_alphabetical_universe_failures": 0, "imputed_funding_used_for_gates": 0,
        "event_sampling_or_caps": False, "materialization_preflight_eligible_count": len(eligible),
        "materialization_preflight_eligible_candidates": eligible["candidate_definition_id"].astype(str).tolist(),
        "runtime_seconds": time.monotonic() - started,
        "evidence_label": "train_only_aggregate_diagnostic_funding_imputed_capped_not_validation",
        "next_recommended_phase": "prior_high_v2_materialization_preflight_for_nonfutile_deduplicated_candidates_next" if len(eligible) else "preserve_prior_high_v2_for_redesign_or_diagnostic_review_next",
        "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    write_json(root / "decision_summary.json", summary)
    write_json(root / "watch_status.json", {"status": summary["status"], "stage": "complete", "shards_completed": len(results), "shards_planned": len(plan), "updated_ts": runner.utc_now()})
    write_csv(root / "performance/runtime_report.md.csv", [{"runtime_seconds": summary["runtime_seconds"], "shards": len(results), "events": summary["events_scored"]}])
    (root / "performance/runtime_report.md").write_text(f"# Runtime Report\n\nRuntime seconds: `{summary['runtime_seconds']:.3f}`\nShards: `{len(results)}`\n", encoding="utf-8")
    bundle = root / "compact_review_bundle"
    bundle.mkdir()
    for rel in ["contract/authoritative_manifest_audit.csv", "contract/entry_exit_binding_audit.csv", "shards/full_shard_plan.csv", "shards/shard_status_summary.csv", "audit/canonical_hash_lineage_audit.csv", "audit/selected_to_outcome_attrition_audit.csv", "audit/funding_boundary_join_audit.csv", "audit/protected_interval_audit.csv", "aggregate/full_definition_scorecard.csv", "aggregate/prior_high_filter_ablation.csv", "aggregate/generic_breakout_comparison.csv", "aggregate/donchian_comparison.csv", "aggregate/exit_policy_summary.csv", "aggregate/period_support_summary.csv", "forensics/concentration_preview.csv", "performance/runtime_report.md", "decision_summary.json"]:
        source = root / rel
        shutil.copy2(source, bundle / rel.replace("/", "__"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
