from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.core_liquid_campaign.accounting import FundingPayment, TradeBar, aggregate_parent_legs, simulate_leg
from tools.core_liquid_campaign.controls import apply_control, compile_controls
from tools.core_liquid_campaign.executor import AuthorizationError, CacheAuthority, ExecutionAuthorization, execute_registered_attempt
from tools.core_liquid_campaign.family_engines import a1_compression, a2_context, a3_starter_retest, a4_tsmom, kda02b_adjudication
from tools.core_liquid_campaign.family_engines.common import EngineInputError, average_rank_percentiles, liquidity_decile, require_available, weak_percentile
from tools.core_liquid_campaign.generator import point, radical_inverse
from tools.core_liquid_campaign.runtime import LazySupervisor, ResourceGateError, ResourceLimits, process_tree_rss, synthetic_recovery_canary
from tools.core_liquid_campaign.schema import (
    FAMILY_ORDER,
    SchemaError,
    axis_fixture_config,
    baseline_config,
    economic_address,
    engine_kwargs,
    evaluate_expression,
    family_schemas,
    gower_distance,
    normalize_config,
    schema_document,
)
from tools.core_liquid_campaign.selection import EventObservation, aggregate_materialized, aggregate_streaming, deduplicate_event_overlap, family_outer_vector, inner_fold_summary, materialization_policy, select_beam
from tools.core_liquid_campaign.validators import aggregate_materialized_probe, assert_rankable_interval, engine_probe, validate_schema


UTC = timezone.utc


def bars(count: int = 600, start_price: float = 100.0) -> list[TradeBar]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    result = []
    price = start_price
    for index in range(count):
        open_ts = start + timedelta(minutes=5 * index)
        result.append(TradeBar(open_ts, open_ts + timedelta(minutes=5), price, price + 0.02))
        price += 0.02
    return result


def inactive_fixture(family: str, field: str) -> dict[str, object]:
    config = baseline_config(family)
    if family == "A4_TSMOM_V7" and field == "ATR_window_days_for_ATR_exits":
        config["exit"] = "time_1d"
    elif family == "A1_COMPRESSION_V2" and field in {"fixed_target_R", "ATR_window_days"}:
        config["exit"] = "time_1d"
    elif family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        if field == "parent_source_attempt_id":
            config["parent_binding_mode"] = "beam_slot"
        elif field in {"parent_fold_id", "parent_beam_rank"}:
            config["parent_binding_mode"] = "source_attempt"
            config["parent_source_attempt_id"] = "A1_COMPRESSION_V2:anchor_ablation:0001"
        elif field in {"RS_lookback_days", "RS_population_scope"}:
            config["RS_rank"] = "none"
            config["proximity_rank"] = "q40"
        elif field in {"prior_high_lookback_days", "ATR_window_days_for_proximity"}:
            config["proximity_rank"] = "none"
            config["reclaim_state"] = "none"
        elif field.startswith("BTC_ETH_drawdown"):
            config["BTC_ETH_context"] = "none"
        elif field.startswith("BTC_ETH_trend"):
            config["BTC_ETH_context"] = "none"
        elif field.startswith("BTC_ETH_volatility"):
            config["BTC_ETH_context"] = "none"
        elif field == "breadth_return_lookback_days":
            config["breadth_dispersion"] = "none"
        elif field == "dispersion_return_lookback_days":
            config["breadth_dispersion"] = "none"
    elif family == "A3_STARTER_RETEST_V3" and field in {"add_requires_reclaim", "retest_depth_ATR", "retest_window"}:
        config["add_fraction"] = 0.0
    elif family == "A3_STARTER_RETEST_V3" and field in {"fixed_target_R", "ATR_window_days"}:
        config["exit"] = "time_1d"
    return config


class SchemaTests(unittest.TestCase):
    def test_schema_has_every_required_field_and_interpreter(self) -> None:
        result = validate_schema()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["families"], 5)
        self.assertEqual(schema_document()["removed_axes"]["A4_TSMOM_V7.signal_rank_scope"]["classification"], "removed_as_unsupported")

    def test_every_axis_value_has_a_valid_active_fixture(self) -> None:
        for family in FAMILY_ORDER:
            for spec in family_schemas[family].axes:
                values = spec.allowed_values or ("A1_COMPRESSION_V2:anchor_ablation:0001",)
                for value in values:
                    with self.subTest(family=family, field=spec.name, value=value):
                        normalized = normalize_config(family, axis_fixture_config(family, spec.name, value))
                        self.assertEqual(normalized[spec.name], value)

    def test_every_conditional_axis_has_active_and_inactive_branch(self) -> None:
        for family in FAMILY_ORDER:
            for spec in family_schemas[family].axes:
                if spec.active_if["op"] == "always":
                    continue
                value = (spec.allowed_values or ("A1_COMPRESSION_V2:anchor_ablation:0001",))[0]
                active = axis_fixture_config(family, spec.name, value)
                with self.subTest(family=family, field=spec.name, branch="active"):
                    self.assertTrue(evaluate_expression(spec.active_if, active))
                    self.assertIsNotNone(normalize_config(family, active)[spec.name])
                inactive = inactive_fixture(family, spec.name)
                with self.subTest(family=family, field=spec.name, branch="inactive"):
                    self.assertFalse(evaluate_expression(spec.active_if, inactive))
                    self.assertIsNone(normalize_config(family, inactive)[spec.name])
                    self.assertNotIn(spec.name, engine_kwargs(family, inactive))

    def test_missing_active_unknown_and_wrong_type_fail_closed(self) -> None:
        config = baseline_config("A4_TSMOM_V7")
        config.pop("lookback_days")
        with self.assertRaises(SchemaError):
            normalize_config("A4_TSMOM_V7", config)
        config = baseline_config("A4_TSMOM_V7")
        config["future_guess"] = 1
        with self.assertRaises(SchemaError):
            normalize_config("A4_TSMOM_V7", config)
        config = baseline_config("A1_COMPRESSION_V2")
        config["PIT_liquidity_top_n"] = "20"
        with self.assertRaises(SchemaError):
            normalize_config("A1_COMPRESSION_V2", config)

    def test_identity_excludes_provenance_and_collapses_inactive_axes(self) -> None:
        left = baseline_config("A3_STARTER_RETEST_V3")
        left["add_fraction"] = 0.0
        left["retest_depth_ATR"] = 0.25
        right = dict(left)
        right["retest_depth_ATR"] = 1.5
        _, left_hash = economic_address("A3_STARTER_RETEST_V3", left)
        _, right_hash = economic_address("A3_STARTER_RETEST_V3", right)
        self.assertEqual(left_hash, right_hash)
        left["direction"] = "long"
        right["direction"] = "short"
        self.assertNotEqual(economic_address("A3_STARTER_RETEST_V3", left)[1], economic_address("A3_STARTER_RETEST_V3", right)[1])

    def test_gower_ignores_jointly_inactive_values(self) -> None:
        left = baseline_config("A1_COMPRESSION_V2")
        left["exit"] = "time_1d"
        left["ATR_window_days"] = 10
        right = dict(left)
        right["ATR_window_days"] = 60
        self.assertEqual(gower_distance("A1_COMPRESSION_V2", left, right), 0.0)

    def test_invalid_combinations(self) -> None:
        config = baseline_config("A3_STARTER_RETEST_V3")
        config.update({"starter_fraction": 1.0, "add_fraction": 0.75})
        with self.assertRaises(SchemaError):
            normalize_config("A3_STARTER_RETEST_V3", config)
        config = baseline_config("A2_PRIOR_HIGH_RS_CONTEXT_V1")
        config.update({"RS_rank": "none", "BTC_ETH_context": "none", "breadth_dispersion": "none", "proximity_rank": "none", "reclaim_state": "none"})
        with self.assertRaises(SchemaError):
            normalize_config("A2_PRIOR_HIGH_RS_CONTEXT_V1", config)
        config["overlay_action"] = "parent_only"
        self.assertEqual(normalize_config("A2_PRIOR_HIGH_RS_CONTEXT_V1", config)["overlay_action"], "parent_only")


class GeneratorAndControlTests(unittest.TestCase):
    def test_radical_inverse_fixture(self) -> None:
        self.assertEqual(radical_inverse(1, 2), 0.5)
        self.assertEqual(radical_inverse(2, 2), 0.25)
        self.assertAlmostEqual(radical_inverse(3, 3), 1 / 9)

    def test_hard_coded_generator_fixture_independent_of_registry(self) -> None:
        self.assertEqual(point("A4_TSMOM_V7", 0, 4), (0.2177356780122084, 0.5008439442248556, 0.5540227863773837, 0.5121447628219784))
        self.assertEqual(point("A1_COMPRESSION_V2", 2, 4), (0.8296487740977443, 0.13746536182537225, 0.8837268649078494, 0.7842963900187956))

    def test_controls_are_finite_unique_and_slot_bound(self) -> None:
        rows = compile_controls()
        self.assertEqual(len(rows), 1600)
        self.assertEqual(len({row["economic_address_sha256"] for row in rows}), 1600)
        self.assertEqual(len({row["parent_slot"] for row in rows}), 320)
        self.assertTrue(all(row["missing_parent_behavior"] == "unavailable_no_parent" for row in rows))
        self.assertTrue(all(row["replicate_index"] == 0 for row in rows))

    def test_control_transformations_replay_and_preserve_groups(self) -> None:
        events = [
            {"event_id": "e1", "symbol": "BTC", "year": 2025, "signal_scalar": 2.0},
            {"event_id": "e2", "symbol": "BTC", "year": 2025, "signal_scalar": -1.0},
            {"event_id": "e3", "symbol": "BTC", "year": 2025, "signal_scalar": 0.0},
        ]
        first = apply_control("sign_permutation", events, 123)
        second = apply_control("sign_permutation", events, 123)
        self.assertEqual(first, second)
        self.assertEqual(sorted(row["signal_magnitude"] for row in first), [0.0, 1.0, 2.0])
        self.assertEqual([row["event_id"] for row in first], ["e1", "e2", "e3"])


class FamilyEngineTests(unittest.TestCase):
    def test_a4_estimators_and_side_grammar(self) -> None:
        closes = [100 + index * 0.1 for index in range(40)]
        highs = [value + 0.2 for value in closes]
        lows = [value - 0.2 for value in closes]
        scalar = a4_tsmom.signal_scalar({"signal_estimator": "signed_return", "volatility_estimator": "close_to_close"}, closes, highs=highs, lows=lows)
        self.assertGreater(scalar, 0)
        self.assertGreater(a4_tsmom.volatility({"volatility_estimator": "parkinson"}, closes, highs, lows), 0)
        self.assertEqual([a4_tsmom.side_from_scalar(value, "long_short") for value in (1, 0, -1)], [1, 0, -1])

    def test_a1_features_confirmation_and_identity(self) -> None:
        result = a1_compression.features([100, 101, 102], [102, 102.1, 102, 102.1], [100, 101, 100, 101], 1)
        self.assertGreater(result["side_signed_impulse"], 0)
        self.assertLess(result["contraction_ratio"], 1)
        self.assertTrue(a1_compression.confirmation_pass([101, 102], 100, 1, "two_closes"))
        self.assertEqual(a1_compression.event_id("BTC", 1, "a", "b", "c", "d", "e"), a1_compression.event_id("BTC", 1, "a", "b", "c", "d", "e"))

    def test_a2_parent_counterpart_and_overlay(self) -> None:
        config = {"proximity_rank": "q40", "RS_rank": "continuous", "reclaim_state": "none", "BTC_ETH_context": "none", "breadth_dispersion": "none"}
        components = a2_context.component_vector(config, {"proximity": 0.8, "RS": 0.7})
        self.assertEqual(a2_context.overlay_multiplier("permission", components), 1.0)
        self.assertEqual(a2_context.overlay_multiplier("parent_only", []), 1.0)
        self.assertEqual(a2_context.parent_slot_id("A1_COMPRESSION_V2", "2024Q1", 1), "A1_COMPRESSION_V2:2024Q1:beam:01")
        parent, overlay = a2_context.counterpart_ids("a" * 64, "event")
        self.assertNotEqual(parent, overlay)
        with self.assertRaises(EngineInputError):
            a2_context.paired_uplift(1, 2, "parent", "overlay", True, True)

    def test_a3_directional_retest_and_parent(self) -> None:
        self.assertEqual(a3_starter_retest.breakout_magnitude(105, 100, 2, "long"), 2.5)
        self.assertEqual(a3_starter_retest.retest_state("long", 100, 2, 0.5, 102, 100.5), "activated")
        self.assertEqual(a3_starter_retest.retest_state("long", 100, 2, 0.5, 100, 101), "reclaimed")
        self.assertEqual(a3_starter_retest.parent_weights(0.5, 0.25, 10, 20), 10)

    def test_kda_variants_are_explicit(self) -> None:
        source = {"price_x": 1, "open_interest_component": 2, "liquidation_component": 3}
        self.assertIsNone(kda02b_adjudication.apply_variant(source, "OI_removed")["open_interest_component"])
        self.assertNotIn("open_interest_component", kda02b_adjudication.apply_variant(source, "price_only"))
        with self.assertRaises(EngineInputError):
            kda02b_adjudication.apply_variant(source, "invented")


class AccountingAndSelectionTests(unittest.TestCase):
    def leg(self, source: list[TradeBar], **overrides: object):
        kwargs = {
            "entry_index": 0, "side": 1, "exit_name": "time_1d", "atr": None, "fixed_target_r": None,
            "structural_level": None, "signal_reversal_close_ts": None, "funding": [], "cost_bps": 14.0,
            "funding_alignment": "start_exclusive_end_inclusive", "evaluation_end_exclusive": datetime(2026, 1, 1, tzinfo=UTC),
        }
        kwargs.update(overrides)
        return simulate_leg(source, **kwargs)

    def test_time_stop_target_trail_structure_reversal_and_boundary_paths(self) -> None:
        self.assertEqual(self.leg(bars()).exit_reason, "time")
        stop = bars(5); stop[1] = TradeBar(stop[1].open_ts, stop[1].close_ts, 100, 95)
        self.assertEqual(self.leg(stop, exit_name="ATR_stop_1.5", atr=2.0).exit_reason, "structural_or_ATR_stop")
        target = bars(5); target[1] = TradeBar(target[1].open_ts, target[1].close_ts, 100, 104)
        self.assertEqual(self.leg(target, exit_name="ATR_stop_1.5", atr=2.0, fixed_target_r=1.0).exit_reason, "fixed_target")
        trail = bars(6); trail[1] = TradeBar(trail[1].open_ts, trail[1].close_ts, 100, 105); trail[2] = TradeBar(trail[2].open_ts, trail[2].close_ts, 105, 101)
        self.assertEqual(self.leg(trail, exit_name="ATR_trail_2", atr=1.0).exit_reason, "trail")
        structure = bars(5); structure[1] = TradeBar(structure[1].open_ts, structure[1].close_ts, 100, 98)
        self.assertEqual(self.leg(structure, exit_name="base_failure", structural_level=99).exit_reason, "structural_or_ATR_stop")
        reversal = bars(5)
        self.assertEqual(self.leg(reversal, exit_name="signal_reversal", signal_reversal_close_ts={reversal[1].close_ts}).exit_reason, "signal_reversal")
        boundary = self.leg(bars(), evaluation_end_exclusive=datetime(2025, 1, 1, 12, tzinfo=UTC))
        self.assertEqual(boundary.status, "unavailable_boundary_crossing")

    def test_funding_cost_sign_and_parent_accounting(self) -> None:
        source = bars()
        payment = FundingPayment(source[100].open_ts, source[99].close_ts, 1.0)
        long = self.leg(source, funding=[payment])
        short = self.leg(source, side=-1, funding=[payment])
        self.assertEqual(long.funding_bps, -1.0)
        self.assertEqual(short.funding_bps, 1.0)
        self.assertEqual(long.cost_bps, 14.0)
        parent = aggregate_parent_legs(long, 0.5, long, 0.25)
        self.assertEqual(parent["event_count"], 1)
        self.assertAlmostEqual(parent["net_bps"], 0.75 * float(long.net_bps))

    def test_feature_availability_and_rank_boundaries(self) -> None:
        decision = datetime(2025, 1, 1, tzinfo=UTC)
        require_available(decision, decision)
        with self.assertRaises(EngineInputError):
            require_available(decision + timedelta(microseconds=1), decision)
        self.assertEqual(average_rank_percentiles([1, 2, 2, 4]), [0.0, 0.5, 0.5, 1.0])
        self.assertEqual([liquidity_decile(value) for value in (0, 0.1, 0.999, 1)], [1, 2, 10, 10])
        population = list(range(30))
        self.assertEqual(weak_percentile(29, population), 1.0)

    def test_aggregate_materialized_parity_and_empty_folds(self) -> None:
        probe = aggregate_materialized_probe()
        self.assertTrue(probe["pass"])
        events = [EventObservation("e1", "BTC", "2025-01-01", "2025-01", 2025, 1, -1, "2025-01-01")]
        self.assertEqual(aggregate_materialized(events), aggregate_streaming(iter(events)))
        folds = inner_fold_summary([1.0, None, -1.0])
        self.assertEqual(folds["empty_count"], 1)
        self.assertIsNone(folds["p20_with_empties_unavailable"])

    def test_selection_and_materialization_are_deterministic(self) -> None:
        rows = []
        for index in range(8):
            rows.append({"canonical_economic_address_sha256": f"{index:064x}", "stable_region": True, "accepted_trades": 30, "market_days": 20, "base_net_bps": 1, "threshold_coverage": 0.7, "plateau_support_count": 5 + index, "day_cluster_bootstrap_q05": 1, "p20_inner_fold": 1, "stress_net_bps": 0, "opportunity_frequency": 1, "complexity": 1})
        self.assertEqual(len(select_beam(rows)), 5)
        policy = [{"canonical_economic_address_sha256": "1" * 64, "beam_survivor": True, "passed": True}]
        self.assertEqual(materialization_policy(policy), ["1" * 64])
        overlap_rows = [
            {**rows[0], "event_ids": ["a", "b", "c"]},
            {**rows[1], "event_ids": ["a", "b", "c"]},
        ]
        retained, rejected = deduplicate_event_overlap(overlap_rows)
        self.assertEqual(len(retained), 1)
        self.assertEqual(rejected[0]["status"], "rejected_event_overlap")
        self.assertEqual(family_outer_vector({"f1": [1, 3]}, ["f1", "f2"]), [2, -float("inf")])

    def test_protected_firewall(self) -> None:
        assert_rankable_interval(datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
        with self.assertRaises(ValueError):
            assert_rankable_interval(datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC) + timedelta(microseconds=1))


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def status(pid: int, ppid: int, rss_kib: int) -> str:
        return f"Pid:\t{pid}\nPPid:\t{ppid}\nVmRSS:\t{rss_kib} kB\n"

    def test_child_disappearance_and_normal_aggregation(self) -> None:
        root = 100
        def reader(pid: int) -> str:
            if pid == root:
                return self.status(root, 1, 100)
            if pid == 101:
                raise ProcessLookupError()
            if pid == 102:
                return self.status(102, root, 200)
            raise AssertionError(pid)
        self.assertEqual(process_tree_rss(root, pid_list=lambda: [root, 101, 102], status_reader=reader), 300 * 1024)

    def test_esrch_is_vanished_but_unrelated_oserror_propagates(self) -> None:
        root = 100
        def vanished(pid: int) -> str:
            if pid == root:
                return self.status(root, 1, 100)
            raise OSError(errno.ESRCH, "gone")
        self.assertEqual(process_tree_rss(root, pid_list=lambda: [root, 101], status_reader=vanished), 100 * 1024)
        def denied(pid: int) -> str:
            if pid == root:
                return self.status(root, 1, 100)
            raise OSError(errno.EACCES, "denied")
        with self.assertRaises(PermissionError):
            process_tree_rss(root, pid_list=lambda: [root, 101], status_reader=denied)

    def test_root_sampling_fails_closed(self) -> None:
        with self.assertRaises(ResourceGateError):
            process_tree_rss(100, pid_list=lambda: [100], status_reader=lambda _: (_ for _ in ()).throw(ProcessLookupError()))

    def test_lazy_stop_and_idempotent_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = synthetic_recovery_canary(Path(raw))
            self.assertTrue(result["pass"])
            self.assertEqual(result["completed"], 4)

    def test_hard_wall_is_prohibited(self) -> None:
        with self.assertRaises(ResourceGateError):
            ResourceLimits(wall_time_seconds=1).validate()

    def test_worker_failure_retries_once_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            calls = {"count": 0}
            def flaky() -> int:
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError("transient worker death")
                return 7
            limits = ResourceLimits(max_workers=1, max_jobs_in_flight=1, max_output_bytes=1024**3, minimum_free_disk_bytes=1)
            state = LazySupervisor(Path(raw), limits).run([("flaky", flaky)])
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["completed"]["flaky"], 7)
            self.assertEqual(state["attempts"]["flaky"], 2)


class WholeProbeTests(unittest.TestCase):
    def test_complete_synthetic_engine_probe(self) -> None:
        result = engine_probe()
        self.assertTrue(result["pass"])
        self.assertFalse(result["economic_outcomes_opened"])
        self.assertEqual(result["protected_rows_opened"], 0)
        self.assertFalse(result["capitalcom_payload_opened"])

    def test_economic_executor_requires_exact_external_approval_and_closed_cache(self) -> None:
        authorization = ExecutionAuthorization("kraken_core_liquid_discovery_campaign_003_code_first_stage22", "a" * 64, "b" * 64, None, False)
        cache = CacheAuthority("kraken_native_linear_pf", datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), "c" * 64, "d" * 64, "e" * 64, "f" * 64, 0, 0)
        with self.assertRaises(AuthorizationError):
            execute_registered_attempt({}, [], cache_authority=cache, authorization=authorization, expected_manifest_sha256="a" * 64, expected_approval_request_sha256="b" * 64)
        protected = CacheAuthority("kraken_native_linear_pf", datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), "c" * 64, "d" * 64, "e" * 64, "f" * 64, 1, 0)
        with self.assertRaises(AuthorizationError):
            protected.validate()


if __name__ == "__main__":
    unittest.main()
