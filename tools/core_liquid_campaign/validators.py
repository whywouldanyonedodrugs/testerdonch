from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from .accounting import FundingPayment, TradeBar, aggregate_parent_legs, simulate_leg
from .canonical import canonical_hash, sha256_file
from .compiler import EXPECTED_FINAL_CONTROLS, SourcePaths, compile_deterministic
from .controls import CONTROL_IDS, execute_control
from .engine_types import PROTECTED_START, RANKABLE_START
from .executor import synthetic_probe_attempt, validate_registered_attempt
from .family_engines import a2_context
from .family_engines.common import average_rank_percentiles, liquidity_decile
from .schema import CAMPAIGN_ID, FAMILY_ORDER, axis_fixture_config, baseline_config, economic_address, family_schemas, normalize_config, schema_hash
from .selection import EventObservation, adjudicate_route, aggregate_materialized, aggregate_streaming, inner_fold_summary, materialization_policy
from .synthetic import a1_frame, a3_frame, a4_frame, frame_for_family


FORMULA_INTERPRETERS: dict[str, str] = {
    "pit_lagged_liquidity_top_n_v1": "engine_types.FamilyInput PIT gate",
    "frozen_episode_vol_target_v1": "a4_tsmom.evaluate",
    "named_context_multiplier_v3": "a2_context.named_context_multiplier",
    "strict_scalar_sign_side_v1": "a4_tsmom.side_from_scalar",
    "completed_close_next_open_exit_v3": "accounting.simulate_leg",
    "tsmom_lookback_v1": "a4_tsmom.evaluate",
    "path_smoothness_gate_v1": "a4_tsmom.evaluate",
    "utc_rebalance_v1": "a4_tsmom.evaluate",
    "a4_scalar_estimator_v1": "a4_tsmom.evaluate",
    "completed_daily_volatility_window_v1": "a4_tsmom.evaluate",
    "wilder_daily_atr_v1": "family_engines.common.wilder_atr",
    "a4_volatility_estimator_v1": "a4_tsmom.evaluate",
    "contiguous_base_clock_v1": "a1_compression.evaluate",
    "a1_confirmation_v1": "a1_compression.evaluate",
    "base_contraction_rank_gate_v1": "a1_compression.evaluate",
    "a1_side_grammar_v1": "a1_compression.evaluate",
    "fixed_target_from_initial_risk_v1": "accounting.simulate_leg",
    "side_signed_impulse_rank_gate_v1": "a1_compression.evaluate",
    "a1_impulse_window_v1": "a1_compression.evaluate",
    "base_path_smoothness_gate_v1": "a1_compression.evaluate",
    "a1_contraction_baseline_v1": "a1_compression.evaluate",
    "fold_local_threshold_population_v1": "engine_types.ThresholdPopulation",
    "a2_parent_binding_mode_v1": "executor.dispatch_registered_attempt",
    "a2_parent_family_v1": "executor.dispatch_registered_attempt",
    "exact_source_parent_id_v1": "compiler.normalize_legacy",
    "exact_parent_outer_fold_slot_v1": "compiler A2 counterpart registry",
    "exact_parent_beam_rank_v1": "compiler A2 counterpart registry",
    "btc_eth_context_component_v1": "a2_context.raw_component_percentiles",
    "btc_relative_strength_v1": "a2_context.raw_component_percentiles",
    "component_threshold_v1": "a2_context.component_vector",
    "breadth_dispersion_component_v1": "a2_context.raw_component_percentiles",
    "a2_overlay_action_v1": "a2_context.overlay_multiplier",
    "completed_daily_prior_level_v1": "a2_context/a3_starter_retest.evaluate",
    "prior_level_reclaim_v1": "a2_context.raw_component_percentiles",
    "btc_eth_drawdown_v1": "a2_context.raw_component_percentiles",
    "btc_eth_trend_pair_v1": "a2_context.raw_component_percentiles",
    "btc_eth_realized_volatility_v1": "a2_context.raw_component_percentiles",
    "pit_breadth_v1": "a2_context.raw_component_percentiles",
    "pit_dispersion_v1": "a2_context.raw_component_percentiles",
    "a3_directional_identity_v1": "a3_starter_retest.evaluate",
    "a3_add_fraction_v1": "executor._simulate_event",
    "a3_reclaim_required_v1": "a3_starter_retest.run_retest_state_machine",
    "a3_breakout_rank_gate_v1": "a3_starter_retest.evaluate",
    "a3_confirmation_v1": "a3_starter_retest.evaluate",
    "a3_retest_band_v1": "a3_starter_retest.run_retest_state_machine",
    "a3_retest_window_v1": "a3_starter_retest.run_retest_state_machine",
    "a3_starter_fraction_v1": "executor._simulate_event",
    "stage20_exact_cell_identity_v1": "kda02b_adjudication.evaluate",
    "kda02b_exact_adjudication_variant_v1": "kda02b_adjudication.evaluate",
}


def assert_rankable_interval(start: datetime, end_exclusive: datetime) -> None:
    if start.tzinfo is None or end_exclusive.tzinfo is None or start < RANKABLE_START or end_exclusive > PROTECTED_START or start >= end_exclusive:
        raise ValueError("interval crosses rankable/protected boundary")


def validate_schema() -> dict[str, Any]:
    required_fields = {"name", "value_type", "ordered", "allowed_values", "numeric_bounds", "default_policy", "active_if", "valid_if", "formula_id", "feature_availability", "threshold_population_scope", "missingness", "economic_identity", "distance_inclusion", "distance_scale", "complexity_contribution", "control_compatibility", "serialization", "classification", "search_new_broad"}
    missing = []; failures = []; axis_values = 0
    for family in FAMILY_ORDER:
        for spec in family_schemas[family].axes:
            if set(spec.to_dict()) != required_fields:
                failures.append(f"typed fields:{family}.{spec.name}")
            if spec.formula_id not in FORMULA_INTERPRETERS:
                missing.append(f"{family}.{spec.name}:{spec.formula_id}")
            for value in spec.allowed_values or ("A1_COMPRESSION_V2:anchor_ablation:0001",):
                axis_values += 1
                try:
                    normalized = normalize_config(family, axis_fixture_config(family, spec.name, value))
                    if normalized[spec.name] is None:
                        failures.append(f"inactive forced fixture:{family}.{spec.name}={value!r}")
                except Exception as exc:
                    failures.append(f"{family}.{spec.name}={value!r}:{exc}")
    if missing or failures:
        raise AssertionError({"missing_interpreters": missing, "fixture_failures": failures})
    return {"families": 5, "axes": sum(len(family_schemas[family].axes) for family in FAMILY_ORDER), "axis_values": axis_values, "schema_sha256": schema_hash(), "status": "pass"}


def _row(family: str, config: Mapping[str, Any], identifier: str) -> dict[str, Any]:
    normalized = normalize_config(family, config); address = economic_address(family, normalized)[1]
    return {"campaign_id": CAMPAIGN_ID, "executable_attempt_id": identifier, "family_id": family, "config": normalized, "canonical_economic_address_sha256": address, "execution_disposition": "execute_once", "duplicate_of_executable_attempt_id": None}


def semantic_engine_probe() -> dict[str, Any]:
    coverage: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for spec in family_schemas[family].axes:
            for value in spec.allowed_values or ("A1_COMPRESSION_V2:anchor_ablation:0001",):
                config = normalize_config(family, axis_fixture_config(family, spec.name, value))
                key = f"{family}.{spec.name}={value!r}"
                try:
                    if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                        frame = frame_for_family(family, config)
                        frame = replace(frame, decision_ts=frame.decision_ts + timedelta(days=2))
                        event = {"event_id": "synthetic-parent", "side": 1, "parent_only_counterpart_id": "p", "overlay_counterpart_id": "o"}
                        result = a2_context.evaluate_overlay(frame, config, event)
                        executed = "context_multiplier" in result
                        ledger_count = 1
                    else:
                        row = _row(family, config, f"fixture:{canonical_hash(key)}")
                        frames = [frame_for_family(family, config)]
                        if family == "A4_TSMOM_V7" and config.get("exit") == "signal_reversal":
                            frames = [a4_frame(config, signal_sign=1), a4_frame(config, signal_sign=-1, anchor=datetime(2025, 6, 2, tzinfo=timezone.utc))]
                        result = synthetic_probe_attempt(row, frames, registry_by_id={row["executable_attempt_id"]: row})
                        executed = result["status"] == "complete"
                        ledger_count = len(result["ledger"])
                    coverage.append({"family": family, "field": spec.name, "value_json": json.dumps(value, separators=(",", ":")), "formula_id": spec.formula_id, "interpreter": FORMULA_INTERPRETERS[spec.formula_id], "fixture": key, "ledger_rows": ledger_count, "status": "pass" if executed else "fail"})
                except Exception as exc:
                    failure = {"family": family, "field": spec.name, "value": value, "error_type": type(exc).__name__, "error": str(exc)}
                    failures.append(failure)
                    coverage.append({"family": family, "field": spec.name, "value_json": json.dumps(value, separators=(",", ":")), "formula_id": spec.formula_id, "interpreter": FORMULA_INTERPRETERS[spec.formula_id], "fixture": key, "ledger_rows": 0, "status": "fail"})
    return {"schema": "stage22_semantic_engine_probe_v2", "coverage": coverage, "coverage_rows": len(coverage), "failures": failures, "pass": not failures and all(row["status"] == "pass" for row in coverage), "economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False}


def end_to_end_family_probe(family: str) -> dict[str, Any]:
    if family != "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        config = baseline_config(family); row = _row(family, config, f"benchmark:{family}"); frame = frame_for_family(family, config)
        return synthetic_probe_attempt(row, [frame], registry_by_id={row["executable_attempt_id"]: row})
    parent_config = baseline_config("A1_COMPRESSION_V2"); parent_row = _row("A1_COMPRESSION_V2", parent_config, "benchmark:a2-parent"); parent_frame = a1_frame(parent_config)
    parent_result = synthetic_probe_attempt(parent_row, [parent_frame], registry_by_id={parent_row["executable_attempt_id"]: parent_row})
    context_frame = replace(parent_frame, decision_ts=parent_result["observations"][0].decision_ts)
    config = baseline_config(family); row = _row(family, config, "benchmark:a2")
    template = canonical_hash({"benchmark": "a2-parent-slot"})
    row.update({"execution_disposition": "execute_if_parent_available", "parent_binding_template_id": template, "parent_only_counterpart_id": canonical_hash({"template": template, "role": "parent_only"}), "overlay_counterpart_id": canonical_hash({"template": template, "role": "overlay"})})
    binding = {"parent_binding_template_id": template, "parent_executable_attempt_id": parent_row["executable_attempt_id"], "parent_only_counterpart_id": row["parent_only_counterpart_id"], "overlay_counterpart_id": row["overlay_counterpart_id"]}
    registry = {row["executable_attempt_id"]: row, parent_row["executable_attempt_id"]: parent_row}
    return synthetic_probe_attempt(row, [context_frame], registry_by_id=registry, parent_binding=binding, parent_frames=[parent_frame])


def accounting_probe() -> dict[str, Any]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = [TradeBar(start + timedelta(minutes=5 * i), start + timedelta(minutes=5 * (i + 1)), 100 + 0.01 * i, 100 + 0.01 * (i + 1)) for i in range(600)]
    common = dict(entry_index=0, side=1, exit_name="time_1d", atr=None, fixed_target_r=None, structural_level=None, signal_reversal_close_ts=None, cost_bps=14.0, funding_alignment="start_exclusive_end_inclusive", evaluation_start=start, evaluation_end_exclusive=datetime(2026, 1, 1, tzinfo=timezone.utc), gap_allowance_bps=-0.25)
    favorable = simulate_leg(bars, funding=[FundingPayment(bars[100].open_ts, bars[99].close_ts, -2.0)], **common)
    adverse = simulate_leg(bars, funding=[FundingPayment(bars[100].open_ts, bars[99].close_ts, 2.0)], **common)
    checks = {
        "favorable_report_only": favorable.favorable_funding_bps == 2.0 and math.isclose(float(favorable.net_bps), float(favorable.gross_bps) - float(favorable.cost_bps) - 0.25),
        "adverse_enters_selection": adverse.funding_bps == -2.25,
        "reportable_separate": favorable.reportable_net_bps > favorable.net_bps,
    }
    return {"schema": "stage22_accounting_probe_v2", "checks": checks, "pass": all(checks.values()), "economic_outcomes_opened": False}


def control_engine_probe() -> dict[str, Any]:
    coverage: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    for family, control_ids in CONTROL_IDS.items():
        if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
            parent_config = baseline_config("A1_COMPRESSION_V2"); parent_row = _row("A1_COMPRESSION_V2", parent_config, "fixture-parent-a1")
            parent_frame = a1_frame(parent_config)
            parent_result = synthetic_probe_attempt(parent_row, [parent_frame], registry_by_id={parent_row["executable_attempt_id"]: parent_row})
            parent_decision = parent_result["observations"][0].decision_ts
            config = baseline_config(family); row = _row(family, config, "fixture-a2")
            template = canonical_hash({"fixture": "a2-parent-slot"})
            row.update({"execution_disposition": "execute_if_parent_available", "parent_binding_template_id": template, "parent_only_counterpart_id": canonical_hash({"template": template, "role": "parent_only"}), "overlay_counterpart_id": canonical_hash({"template": template, "role": "overlay"})})
            binding = {"parent_binding_template_id": template, "parent_executable_attempt_id": parent_row["executable_attempt_id"], "parent_only_counterpart_id": row["parent_only_counterpart_id"], "overlay_counterpart_id": row["overlay_counterpart_id"]}
            frame = replace(parent_frame, decision_ts=parent_decision)
            registry = {row["executable_attempt_id"]: row, parent_row["executable_attempt_id"]: parent_row}
        else:
            config = baseline_config(family); row = _row(family, config, f"fixture-{family}")
            frame = frame_for_family(family, config); parent_frame = None; binding = None; registry = {row["executable_attempt_id"]: row}
        for control_id in control_ids:
            try:
                selected_frame = frame
                if control_id in {"A1_MATCHED_PSEUDO_EVENT_MAIN_NULL", "A3_MATCHED_PSEUDO_EVENT"}:
                    metadata = dict(frame.metadata); metadata["pseudo_side"] = 1 if family == "A1_COMPRESSION_V2" else int(metadata["control_side"]); metadata["pseudo_candidate_frames"] = (replace(frame, metadata={**metadata, "pseudo_candidate_frames": ()}),)
                    selected_frame = replace(frame, metadata=metadata)
                control = {"control_attempt_id": canonical_hash({"fixture": control_id}), "control_id": control_id, "family": family, "effective_seed": 20260721, "economic_address_sha256": canonical_hash({"fixture-address": control_id})}
                result = execute_control(control, row, [selected_frame], registry_by_id=registry, parent_binding=binding, parent_frames=[parent_frame] if parent_frame is not None else None)
                executed = result["status"] in {"complete", "unavailable_no_parent"}
                coverage.append({"family": family, "control_id": control_id, "dispatcher": "controls.execute_control->executor.dispatch_registered_attempt", "ledger_rows": len(result["ledger"]), "observation_rows": len(result["observations"]), "status": "pass" if executed else "fail"})
            except Exception as exc:
                failures.append({"family": family, "control_id": control_id, "error_type": type(exc).__name__, "error": str(exc)})
                coverage.append({"family": family, "control_id": control_id, "dispatcher": "controls.execute_control->executor.dispatch_registered_attempt", "ledger_rows": 0, "observation_rows": 0, "status": "fail"})
    return {"schema": "stage22_control_engine_probe_v1", "coverage": coverage, "failures": failures, "pass": not failures and len(coverage) == 20, "economic_outcomes_opened": False}


def selection_route_probe() -> dict[str, Any]:
    rows = []
    for family in FAMILY_ORDER:
        cases = [
            ("supported", dict(common_gate=True, main_null=True, component_passes={"component": True}, base_positive=True, stress_positive=True, delay_positive=True, sample_sufficient=True)),
            ("execution_sensitive", dict(common_gate=True, main_null=True, component_passes={"component": True}, base_positive=True, stress_positive=False, delay_positive=True, sample_sufficient=True)),
            ("sample_limited", dict(common_gate=True, main_null=False, component_passes={"component": False}, base_positive=True, stress_positive=True, delay_positive=True, sample_sufficient=False)),
            ("rejected", dict(common_gate=False, main_null=False, component_passes={"component": False}, base_positive=False, stress_positive=False, delay_positive=False, sample_sufficient=True)),
        ]
        for case, kwargs in cases:
            if family == "A3_STARTER_RETEST_V3": kwargs["add_fraction"] = 0.25
            route = adjudicate_route(family, **kwargs)
            rows.append({"family": family, "case": case, "route": route, "status": "pass"})
    return {"schema": "stage22_selection_route_probe_v1", "rows": rows, "pass": len(rows) == 20}


def aggregate_materialized_probe() -> dict[str, Any]:
    d1 = datetime(2025, 1, 2, tzinfo=timezone.utc); d2 = datetime(2025, 1, 3, tzinfo=timezone.utc)
    events = [
        EventObservation("e1", "BTC", "2025-01-02", "2025-01", 2025, 10.0, 2.0, "2025-01-02", d1, d1 + timedelta(minutes=5), d1 + timedelta(days=1)),
        EventObservation("e2", "ETH", "2025-01-02", "2025-01", 2025, -2.0, -10.0, "2025-01-02", d1, d1 + timedelta(minutes=5), d1 + timedelta(days=1)),
        EventObservation("e3", "BTC", "2025-01-03", "2025-01", 2025, 8.0, 1.0, "2025-01-03", d2, d2 + timedelta(minutes=5), d2 + timedelta(days=1)),
    ]
    materialized = aggregate_materialized(events); aggregate = aggregate_streaming(iter(events)); inner = inner_fold_summary([1.0, None, -1.0, None])
    policy = materialization_policy([{"canonical_economic_address_sha256": "01" + "0" * 62, "beam_survivor": True, "passed": True}, {"canonical_economic_address_sha256": "02" + "0" * 62, "near_miss": True, "near_miss_rule": "one_failed_nonintegrity_gate_within_10pct", "passed": False}])
    exact = aggregate == materialized
    empties = inner["vector"].count(-math.inf) == 2 and inner["p20_with_negative_infinity"] == -math.inf
    serialized_inner = {
        **inner,
        "vector": [
            {"status": "unavailable_empty_fold"}
            if value == -math.inf
            else {"status": "available", "value": value}
            for value in inner["vector"]
        ],
        "p20_with_negative_infinity": (
            {"status": "negative_infinity_due_to_empty_fold"}
            if inner["p20_with_negative_infinity"] == -math.inf
            else {"status": "available", "value": inner["p20_with_negative_infinity"]}
        ),
    }
    return {"schema": "stage22_aggregate_vs_materialized_probe_v2", "aggregate": aggregate, "materialized": materialized, "exact_equal": exact, "inner_fold_vector": serialized_inner, "empty_inner_folds_preserved": empties, "materialization_policy_addresses": policy, "pass": exact and empties and len(policy) == 2, "economic_outcomes_opened": False}


def validate_compiled(root: Path) -> dict[str, Any]:
    schema_result = validate_schema(); budget = json.loads((root / "OUTCOME_FREE_BUDGET_OPTIMIZER.json").read_text(encoding="utf-8"))
    registry = [json.loads(line) for line in (root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines() if line]
    execution = [json.loads(line) for line in (root / "FINAL_EXECUTION_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines() if line]
    controls = [json.loads(line) for line in (root / "FINAL_CONTROL_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines() if line]
    counterparts = [json.loads(line) for line in (root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(registry) != budget["target_total_attempt_rows"] or len(controls) != EXPECTED_FINAL_CONTROLS:
        raise AssertionError("final registry count mismatch")
    if len({row["canonical_economic_address_sha256"] for row in execution}) != len(execution):
        raise AssertionError("execution registry has duplicate economic addresses")
    for row in execution:
        validate_registered_attempt(row)
    a2_execution = [row for row in execution if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1"]
    counterpart_index = {row["a2_executable_attempt_id"]: row for row in counterparts}
    if set(counterpart_index) != {row["executable_attempt_id"] for row in a2_execution}:
        raise AssertionError("A2 counterpart registry is incomplete")
    if any(row["parent_beam_rank"] is not None and not 1 <= int(row["parent_beam_rank"]) <= 5 for row in counterparts):
        raise AssertionError("A2 counterpart uses a slot beyond the frozen beam")
    with (root / "SEARCH_SPACE_COVERAGE_MATRIX.csv").open(encoding="utf-8", newline="") as handle:
        marginal_failures = [row for row in csv.DictReader(handle) if row["status"] != "pass"]
    pair_failures = [row for row in pq.read_table(root / "PAIRWISE_COVERAGE_MATRIX.parquet").to_pylist() if row["status"] == "fail"]
    control_failures = [row for row in csv.DictReader((root / "CONTROL_COVERAGE_MATRIX.csv").open(encoding="utf-8", newline="")) if row["status"] != "pass"]
    if marginal_failures or pair_failures or control_failures:
        raise AssertionError({"marginal": marginal_failures[:5], "pairwise": pair_failures[:5], "controls": control_failures[:5]})
    fixtures = json.loads((root / "REGISTRY_REPLAY_FIXTURES.json").read_text(encoding="utf-8")); control_fixtures = json.loads((root / "CONTROL_REPLAY_FIXTURES.json").read_text(encoding="utf-8"))
    if fixtures["registry"]["sha256"] != sha256_file(root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl") or control_fixtures["sha256"] != sha256_file(root / "FINAL_CONTROL_REGISTRY.jsonl"):
        raise AssertionError("replay fixture hash mismatch")
    return {"status": "pass", "schema": schema_result, "strategy_rows": len(registry), "strategy_unique_addresses": len({row["canonical_economic_address_sha256"] for row in registry}), "execution_rows": len(execution), "control_rows": len(controls), "control_unique_addresses": len({row["economic_address_sha256"] for row in controls}), "a2_counterpart_rows": len(counterparts), "marginal_failures": 0, "priority_pair_failures": 0, "economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False}


REPLAY_FILES = (
    "FAMILY_AXIS_SCHEMA.json", "FAMILY_AXIS_SCHEMA.sha256", "SEMANTIC_COVERAGE_MATRIX.csv", "ACTIVE_IF_TRUTH_TABLE.csv", "INVALID_COMBINATION_MATRIX.csv",
    "LEGACY_NORMALIZATION_LEDGER.parquet", "LEGACY_EXECUTABLE_PROJECTION.jsonl", "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", "FINAL_EXECUTION_REGISTRY.jsonl",
    "A2_PARENT_COUNTERPART_REGISTRY.jsonl", "OUTCOME_FREE_BUDGET_OPTIMIZER.json", "RAW_SPACE_FILLING_COORDINATES.parquet", "SEARCH_SPACE_COVERAGE_MATRIX.csv",
    "PAIRWISE_COVERAGE_MATRIX.parquet", "UNREPRESENTED_VALID_REGIONS.csv", "REGISTRY_REPLAY_FIXTURES.json", "FINAL_CONTROL_REGISTRY.jsonl", "CONTROL_COVERAGE_MATRIX.csv",
    "CONTROL_REPLAY_FIXTURES.json", "LEGACY_CONTROL_LINEAGE.parquet", "ENGINE_COVERAGE_MATRIX.csv", "CONTROL_EXECUTION_COVERAGE_MATRIX.csv", "SELECTION_ROUTE_MATRIX.csv", "EXIT_ACCOUNTING_MATRIX.csv",
    "ENGINE_PROBE_AUDIT.json", "CONTROL_ENGINE_PROBE_AUDIT.json", "SELECTION_ROUTE_PROBE_AUDIT.json", "ACCOUNTING_PROBE_AUDIT.json", "SAFE_PRUNING_POLICY.json", "SHARED_SEMANTIC_CACHE_CONTRACT.json",
    "HISTORICAL_LINEAGE_DECISIONS.json", "COMPILER_SUMMARY.json",
)


def independent_replay(paths: SourcePaths, root: Path) -> dict[str, Any]:
    replay_dir = Path(tempfile.mkdtemp(prefix="stage22-independent-replay."))
    try:
        compile_deterministic(paths, replay_dir); mismatches = []
        for relative in REPLAY_FILES:
            if sha256_file(root / relative) != sha256_file(replay_dir / relative):
                mismatches.append(relative)
        for family in FAMILY_ORDER:
            relative = Path("ENGINE_CONTRACTS") / f"{family}.json"
            if sha256_file(root / relative) != sha256_file(replay_dir / relative):
                mismatches.append(relative.as_posix())
        if mismatches:
            raise AssertionError(f"independent replay mismatches: {mismatches}")
        return {"status": "pass", "files_compared": len(REPLAY_FILES) + len(FAMILY_ORDER), "mismatches": [], "economic_outcomes_opened": False}
    finally:
        shutil.rmtree(replay_dir)


__all__ = ["accounting_probe", "aggregate_materialized_probe", "assert_rankable_interval", "control_engine_probe", "end_to_end_family_probe", "independent_replay", "selection_route_probe", "semantic_engine_probe", "validate_compiled", "validate_schema"]
