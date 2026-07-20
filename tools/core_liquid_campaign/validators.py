from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow.parquet as pq

from .accounting import FundingPayment, TradeBar, aggregate_parent_legs, simulate_leg
from .canonical import canonical_hash, sha256_file
from .compiler import EXPECTED_FINAL_ATTEMPTS, EXPECTED_FINAL_CONTROLS, SourcePaths, compile_deterministic
from .family_engines import ENGINES
from .family_engines import a1_compression, a2_context, a3_starter_retest, a4_tsmom, kda02b_adjudication
from .family_engines.common import average_rank_percentiles, liquidity_decile
from .schema import FAMILY_ORDER, SCHEMA_VERSION, axis_fixture_config, family_schemas, normalize_config, schema_hash
from .selection import EventObservation, aggregate_materialized, aggregate_streaming, inner_fold_summary, materialization_policy


RANKABLE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
PROTECTED_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


FORMULA_INTERPRETERS: dict[str, str] = {
    "pit_lagged_liquidity_top_n_v1": "PIT universe adapter",
    "frozen_episode_vol_target_v1": "a4_tsmom.volatility",
    "named_context_multiplier_v3": "a2_context.overlay_multiplier",
    "strict_scalar_sign_side_v1": "a4_tsmom.side_from_scalar",
    "completed_close_next_open_exit_v3": "accounting.simulate_leg",
    "tsmom_lookback_v1": "a4_tsmom.signal_scalar",
    "path_smoothness_gate_v1": "family_engines.common.path_smoothness",
    "utc_rebalance_v1": "a4 engine contract",
    "a4_scalar_estimator_v1": "a4_tsmom.signal_scalar",
    "completed_daily_volatility_window_v1": "a4_tsmom.volatility",
    "wilder_daily_atr_v1": "accounting ATR input contract",
    "a4_volatility_estimator_v1": "a4_tsmom.volatility",
    "contiguous_base_clock_v1": "a1_compression.features",
    "a1_confirmation_v1": "a1_compression.confirmation_pass",
    "base_contraction_rank_gate_v1": "a1_compression.features",
    "a1_side_grammar_v1": "a1 engine contract",
    "fixed_target_from_initial_risk_v1": "accounting.simulate_leg",
    "side_signed_impulse_rank_gate_v1": "a1_compression.features",
    "a1_impulse_window_v1": "a1_compression.features",
    "base_path_smoothness_gate_v1": "a1_compression.features",
    "a1_contraction_baseline_v1": "a1_compression.features",
    "fold_local_threshold_population_v1": "family_engines.common ranking contract",
    "a2_parent_binding_mode_v1": "a2_context.parent_slot_id",
    "a2_parent_family_v1": "a2_context.parent_slot_id",
    "exact_source_parent_id_v1": "a2 exact source grammar",
    "exact_parent_outer_fold_slot_v1": "a2_context.parent_slot_id",
    "exact_parent_beam_rank_v1": "a2_context.parent_slot_id",
    "btc_eth_context_component_v1": "a2_context.component_vector",
    "btc_relative_strength_v1": "a2_context.component_vector",
    "component_threshold_v1": "family_engines.common.component_threshold",
    "breadth_dispersion_component_v1": "a2_context.component_vector",
    "a2_overlay_action_v1": "a2_context.overlay_multiplier",
    "completed_daily_prior_level_v1": "a2/a3 engine contract",
    "prior_level_reclaim_v1": "a2_context.component_vector",
    "btc_eth_drawdown_v1": "a2_context.component_vector",
    "btc_eth_trend_pair_v1": "a2_context.component_vector",
    "btc_eth_realized_volatility_v1": "a2_context.component_vector",
    "pit_breadth_v1": "a2_context.component_vector",
    "pit_dispersion_v1": "a2_context.component_vector",
    "a3_directional_identity_v1": "a3_starter_retest.event_id",
    "a3_add_fraction_v1": "a3_starter_retest.parent_weights",
    "a3_reclaim_required_v1": "a3_starter_retest.retest_state",
    "a3_breakout_rank_gate_v1": "a3_starter_retest.breakout_magnitude",
    "a3_confirmation_v1": "a3 engine contract",
    "a3_retest_band_v1": "a3_starter_retest.retest_state",
    "a3_retest_window_v1": "a3 engine contract",
    "a3_starter_fraction_v1": "a3_starter_retest.parent_weights",
    "stage20_exact_cell_identity_v1": "kda02b engine contract",
    "kda02b_exact_adjudication_variant_v1": "kda02b_adjudication.apply_variant",
}


def assert_rankable_interval(start: datetime, end_exclusive: datetime) -> None:
    if start.tzinfo is None or end_exclusive.tzinfo is None:
        raise ValueError("interval timestamps must be timezone-aware")
    if start < RANKABLE_START or end_exclusive > PROTECTED_START or start >= end_exclusive:
        raise ValueError("interval crosses rankable/protected boundary")


def validate_schema() -> dict[str, Any]:
    missing_interpreters = []
    axis_values = 0
    active_fixture_failures = []
    required_fields = {
        "name", "value_type", "ordered", "allowed_values", "numeric_bounds", "default_policy", "active_if", "valid_if",
        "formula_id", "feature_availability", "threshold_population_scope", "missingness", "economic_identity", "distance_inclusion",
        "distance_scale", "complexity_contribution", "control_compatibility", "serialization", "classification", "search_new_broad",
    }
    for family in FAMILY_ORDER:
        for spec in family_schemas[family].axes:
            if set(spec.to_dict()) != required_fields:
                raise AssertionError(f"typed axis fields differ for {family}.{spec.name}")
            if spec.formula_id not in FORMULA_INTERPRETERS:
                missing_interpreters.append(f"{family}.{spec.name}:{spec.formula_id}")
            levels = spec.allowed_values or ("A1_COMPRESSION_V2:anchor_ablation:0001",)
            for value in levels:
                axis_values += 1
                try:
                    normalized = normalize_config(family, axis_fixture_config(family, spec.name, value))
                    if normalized[spec.name] is None:
                        active_fixture_failures.append(f"{family}.{spec.name}={value!r}")
                except Exception as exc:
                    active_fixture_failures.append(f"{family}.{spec.name}={value!r}:{exc}")
    if missing_interpreters or active_fixture_failures:
        raise AssertionError({"missing_interpreters": missing_interpreters, "active_fixture_failures": active_fixture_failures})
    return {"families": len(FAMILY_ORDER), "axes": sum(len(family_schemas[family].axes) for family in FAMILY_ORDER), "axis_values": axis_values, "schema_sha256": schema_hash(), "status": "pass"}


def _synthetic_bars(count: int = 600) -> list[TradeBar]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = []
    price = 100.0
    for index in range(count):
        open_ts = start + timedelta(minutes=5 * index)
        close_ts = open_ts + timedelta(minutes=5)
        close = price + 0.02
        result.append(TradeBar(open_ts, close_ts, price, close))
        price = close
    return result


def engine_probe() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    closes = [100.0 + index * 0.1 for index in range(40)]
    highs = [value + 0.2 for value in closes]
    lows = [value - 0.2 for value in closes]
    a4_config = {"signal_estimator": "signed_return", "volatility_estimator": "close_to_close"}
    checks["a4_signed_return"] = a4_tsmom.signal_scalar(a4_config, closes, highs=highs, lows=lows) > 0
    checks["a4_parkinson"] = a4_tsmom.volatility({"volatility_estimator": "parkinson"}, closes, highs, lows) > 0
    checks["a4_sides"] = [a4_tsmom.side_from_scalar(value, "long_short") for value in (1.0, 0.0, -1.0)] == [1, 0, -1]
    a1 = a1_compression.features([100, 101, 102], [102, 102.1, 102.0, 102.1], [100, 101, 100, 101], 1)
    checks["a1_features"] = a1["side_signed_impulse"] > 0 and a1["contraction_ratio"] < 1
    checks["a1_confirmation"] = a1_compression.confirmation_pass([101, 102], 100, 1, "two_closes")
    a2_config = {"proximity_rank": "q40", "RS_rank": "continuous", "reclaim_state": "none", "BTC_ETH_context": "none", "breadth_dispersion": "none"}
    vector = a2_context.component_vector(a2_config, {"proximity": 0.8, "RS": 0.7})
    checks["a2_components"] = a2_context.overlay_multiplier("permission", vector) == 1.0
    parent_only, overlay = a2_context.counterpart_ids("a" * 64, "event")
    checks["a2_counterparts"] = parent_only != overlay
    checks["a3_breakout"] = a3_starter_retest.breakout_magnitude(105, 100, 2, "long") == 2.5
    checks["a3_parent"] = a3_starter_retest.parent_weights(0.5, 0.25, 10, 20) == 10
    checks["kda_variant"] = kda02b_adjudication.apply_variant({"price_x": 1, "open_interest_component": 2}, "OI_removed")["open_interest_component"] is None
    bars = _synthetic_bars()
    time_result = simulate_leg(
        bars, entry_index=0, side=1, exit_name="time_1d", atr=None, fixed_target_r=None, structural_level=None,
        signal_reversal_close_ts=None, funding=[FundingPayment(bars[100].open_ts, bars[99].close_ts, 1.0)], cost_bps=14,
        funding_alignment="start_exclusive_end_inclusive", evaluation_end_exclusive=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    checks["accounting_time_exit"] = time_result.status == "complete" and time_result.exit_reason == "time"
    stop_bars = _synthetic_bars(5)
    stop_bars[1] = TradeBar(stop_bars[1].open_ts, stop_bars[1].close_ts, 100.0, 95.0)
    stop_result = simulate_leg(
        stop_bars, entry_index=0, side=1, exit_name="ATR_stop_1.5", atr=2.0, fixed_target_r=None, structural_level=None,
        signal_reversal_close_ts=None, funding=[], cost_bps=14, funding_alignment="start_exclusive_end_inclusive",
        evaluation_end_exclusive=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    checks["accounting_stop"] = stop_result.status == "complete" and stop_result.exit_reason == "structural_or_ATR_stop"
    target_bars = _synthetic_bars(5)
    target_bars[1] = TradeBar(target_bars[1].open_ts, target_bars[1].close_ts, 100.0, 104.0)
    target_result = simulate_leg(
        target_bars, entry_index=0, side=1, exit_name="ATR_stop_1.5", atr=2.0, fixed_target_r=1.0, structural_level=None,
        signal_reversal_close_ts=None, funding=[], cost_bps=14, funding_alignment="start_exclusive_end_inclusive",
        evaluation_end_exclusive=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    checks["accounting_target"] = target_result.status == "complete" and target_result.exit_reason == "fixed_target"
    parent = aggregate_parent_legs(time_result, 0.5, time_result, 0.25)
    checks["a3_parent_accounting"] = parent["event_count"] == 1 and math.isclose(parent["net_bps"], 0.75 * float(time_result.net_bps))
    checks["rank_ties"] = average_rank_percentiles([1.0, 2.0, 2.0, 4.0]) == [0.0, 0.5, 0.5, 1.0]
    checks["deciles"] = [liquidity_decile(value) for value in (0.0, 0.1, 0.999, 1.0)] == [1, 2, 10, 10]
    assert_rankable_interval(RANKABLE_START, PROTECTED_START)
    try:
        assert_rankable_interval(RANKABLE_START, PROTECTED_START + timedelta(microseconds=1))
        checks["protected_firewall"] = False
    except ValueError:
        checks["protected_firewall"] = True
    return {"schema": "stage22_engine_probe_v1", "checks": checks, "pass": all(checks.values()), "economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False}


def aggregate_materialized_probe() -> dict[str, Any]:
    events = [
        EventObservation("e1", "BTC", "2025-01-02", "2025-01", 2025, 10.0, 2.0, "2025-01-02"),
        EventObservation("e2", "ETH", "2025-01-02", "2025-01", 2025, -2.0, -10.0, "2025-01-02"),
        EventObservation("e3", "BTC", "2025-01-03", "2025-01", 2025, 8.0, 1.0, "2025-01-03"),
    ]
    materialized = aggregate_materialized(events)
    aggregate = aggregate_streaming(iter(events))
    inner = inner_fold_summary([1.0, None, -1.0, None])
    policy_rows = [
        {"canonical_economic_address_sha256": "01" + "0" * 62, "beam_survivor": True, "passed": True},
        {"canonical_economic_address_sha256": "02" + "0" * 62, "near_miss": True, "near_miss_rule": "one_failed_nonintegrity_gate_within_10pct", "passed": False},
    ]
    selected = materialization_policy(policy_rows)
    return {
        "schema": "stage22_aggregate_vs_materialized_probe_v1",
        "aggregate": aggregate,
        "materialized": materialized,
        "exact_equal": aggregate == materialized,
        "inner_fold_vector": inner,
        "empty_inner_folds_preserved": inner["empty_count"] == 2 and inner["p20_with_empties_unavailable"] is None,
        "materialization_policy_addresses": selected,
        "pass": aggregate == materialized and inner["empty_count"] == 2 and len(selected) == 2,
        "economic_outcomes_opened": False,
    }


def validate_compiled(root: Path) -> dict[str, Any]:
    schema_result = validate_schema()
    registry_path = root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"
    control_path = root / "FINAL_CONTROL_REGISTRY.jsonl"
    registry = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line]
    controls = [json.loads(line) for line in control_path.read_text(encoding="utf-8").splitlines() if line]
    if len(registry) != EXPECTED_FINAL_ATTEMPTS or len(controls) != EXPECTED_FINAL_CONTROLS:
        raise AssertionError("final registry count mismatch")
    if len({row["executable_attempt_id"] for row in registry}) != len(registry):
        raise AssertionError("duplicate executable attempt ID")
    if len({row["economic_address_sha256"] for row in controls}) != len(controls):
        raise AssertionError("duplicate control address")
    for row in registry:
        normalized = normalize_config(row["family_id"], row["config"])
        if normalized != row["config"]:
            raise AssertionError(f"noncanonical config: {row['executable_attempt_id']}")
    with (root / "SEARCH_SPACE_COVERAGE_MATRIX.csv").open(encoding="utf-8", newline="") as handle:
        marginal_failures = [row for row in csv.DictReader(handle) if row["status"] != "pass"]
    pair_table = pq.read_table(root / "PAIRWISE_COVERAGE_MATRIX.parquet").to_pylist()
    pair_failures = [row for row in pair_table if row["status"] == "fail"]
    control_coverage = list(csv.DictReader((root / "CONTROL_COVERAGE_MATRIX.csv").open(encoding="utf-8", newline="")))
    if marginal_failures or pair_failures or any(row["status"] != "pass" for row in control_coverage):
        raise AssertionError({"marginal": marginal_failures[:5], "pairwise": pair_failures[:5], "controls": [row for row in control_coverage if row["status"] != "pass"]})
    fixtures = json.loads((root / "REGISTRY_REPLAY_FIXTURES.json").read_text(encoding="utf-8"))
    control_fixtures = json.loads((root / "CONTROL_REPLAY_FIXTURES.json").read_text(encoding="utf-8"))
    if fixtures["registry"]["sha256"] != sha256_file(registry_path) or control_fixtures["sha256"] != sha256_file(control_path):
        raise AssertionError("replay fixture hash mismatch")
    return {
        "status": "pass",
        "schema": schema_result,
        "strategy_rows": len(registry),
        "strategy_unique_addresses": len({row["canonical_economic_address_sha256"] for row in registry}),
        "control_rows": len(controls),
        "control_unique_addresses": len({row["economic_address_sha256"] for row in controls}),
        "marginal_failures": 0,
        "priority_pair_failures": 0,
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
    }


REPLAY_FILES = (
    "FAMILY_AXIS_SCHEMA.json",
    "FAMILY_AXIS_SCHEMA.sha256",
    "SEMANTIC_COVERAGE_MATRIX.csv",
    "ACTIVE_IF_TRUTH_TABLE.csv",
    "INVALID_COMBINATION_MATRIX.csv",
    "LEGACY_NORMALIZATION_LEDGER.parquet",
    "LEGACY_EXECUTABLE_PROJECTION.jsonl",
    "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl",
    "RAW_SPACE_FILLING_COORDINATES.parquet",
    "SEARCH_SPACE_COVERAGE_MATRIX.csv",
    "PAIRWISE_COVERAGE_MATRIX.parquet",
    "UNREPRESENTED_VALID_REGIONS.csv",
    "REGISTRY_REPLAY_FIXTURES.json",
    "FINAL_CONTROL_REGISTRY.jsonl",
    "CONTROL_COVERAGE_MATRIX.csv",
    "CONTROL_REPLAY_FIXTURES.json",
    "LEGACY_CONTROL_LINEAGE.parquet",
    "ENGINE_COVERAGE_MATRIX.csv",
    "SAFE_PRUNING_POLICY.json",
    "SHARED_SEMANTIC_CACHE_CONTRACT.json",
    "HISTORICAL_LINEAGE_DECISIONS.json",
    "COMPILER_SUMMARY.json",
)


def independent_replay(paths: SourcePaths, root: Path) -> dict[str, Any]:
    replay_dir = Path(tempfile.mkdtemp(prefix="stage22-independent-replay."))
    try:
        compile_deterministic(paths, replay_dir)
        mismatches = []
        for relative in REPLAY_FILES:
            source = root / relative
            replay = replay_dir / relative
            if sha256_file(source) != sha256_file(replay):
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
