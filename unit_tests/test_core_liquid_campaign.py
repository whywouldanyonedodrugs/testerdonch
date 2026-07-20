from __future__ import annotations

import errno
import json
import math
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.core_liquid_campaign.accounting import FundingPayment, TradeBar, aggregate_parent_legs, simulate_leg
from tools.core_liquid_campaign.budget import optimize_budget
from tools.core_liquid_campaign.canonical import atomic_write_json, canonical_hash, sha256_file
from tools.core_liquid_campaign.controls import CONTROL_IDS, compile_controls, transformed_inputs
from tools.core_liquid_campaign.engine_types import FamilyInput
from tools.core_liquid_campaign.executor import AuthorizationError, CacheAuthority, ExecutionAuthorization, validate_registered_attempt
from tools.core_liquid_campaign.family_engines import a2_context, a3_starter_retest
from tools.core_liquid_campaign.family_engines.common import EngineInputError, average_rank_percentiles, liquidity_decile, require_available
from tools.core_liquid_campaign.generator import point, radical_inverse
from tools.core_liquid_campaign.packet import _require_bound_review
from tools.core_liquid_campaign.runtime import LazySupervisor, ResourceGateError, ResourceLimits, process_tree_rss, synthetic_recovery_canary
from tools.core_liquid_campaign.schema import (
    FAMILY_ORDER,
    OUTER_FOLDS,
    SchemaError,
    axis_fixture_config,
    baseline_config,
    economic_address,
    engine_kwargs,
    evaluate_expression,
    family_schemas,
    gower_distance,
    normalize_config,
)
from tools.core_liquid_campaign.selection import (
    EventObservation,
    adjudicate_route,
    aggregate_materialized,
    aggregate_streaming,
    day_cluster_bootstrap_q05,
    deduplicate_event_overlap,
    family_outer_vector,
    inner_fold_summary,
    materialization_policy,
    refinement_neighbors,
    select_beam,
    stable_neighborhoods,
)
from tools.core_liquid_campaign.synthetic import a1_frame, a3_frame
from tools.core_liquid_campaign.validators import (
    accounting_probe,
    aggregate_materialized_probe,
    assert_rankable_interval,
    control_engine_probe,
    end_to_end_family_probe,
    selection_route_probe,
    semantic_engine_probe,
    validate_schema,
)


UTC = timezone.utc


def trade_bars(count: int = 700) -> list[TradeBar]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        open_ts = start + timedelta(minutes=5 * index)
        price = 100.0 + index * 0.001
        rows.append(TradeBar(open_ts, open_ts + timedelta(minutes=5), price, price, high=price + 0.1, low=price - 0.1, source_close_ts=open_ts + timedelta(minutes=5), feature_available_ts=open_ts + timedelta(minutes=5)))
    return rows


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
            config["RS_rank"] = "none"; config["proximity_rank"] = "q40"
        elif field in {"prior_high_lookback_days", "ATR_window_days_for_proximity"}:
            config["proximity_rank"] = "none"; config["reclaim_state"] = "none"
        elif field.startswith("BTC_ETH_"):
            config["BTC_ETH_context"] = "none"
        elif field in {"breadth_return_lookback_days", "dispersion_return_lookback_days"}:
            config["breadth_dispersion"] = "none"
    elif family == "A3_STARTER_RETEST_V3" and field in {"add_requires_reclaim", "retest_depth_ATR", "retest_window"}:
        config["add_fraction"] = 0.0
    elif family == "A3_STARTER_RETEST_V3" and field in {"fixed_target_R"}:
        config["exit"] = "time_1d"
    return config


class SchemaAndGeneratorTests(unittest.TestCase):
    def test_schema_and_every_registered_axis_value_execute(self) -> None:
        schema = validate_schema()
        self.assertEqual(schema["status"], "pass")
        probe = semantic_engine_probe()
        self.assertTrue(probe["pass"], probe["failures"])
        self.assertGreater(probe["coverage_rows"], 250)

    def test_active_if_inactive_branch_and_unknowns_fail_closed(self) -> None:
        for family in FAMILY_ORDER:
            for spec in family_schemas[family].axes:
                value = (spec.allowed_values or ("A1_COMPRESSION_V2:anchor_ablation:0001",))[0]
                active = axis_fixture_config(family, spec.name, value)
                self.assertTrue(evaluate_expression(spec.active_if, active), f"{family}.{spec.name}")
                self.assertEqual(normalize_config(family, active)[spec.name], value)
                if spec.active_if["op"] != "always":
                    inactive = inactive_fixture(family, spec.name)
                    self.assertFalse(evaluate_expression(spec.active_if, inactive), f"{family}.{spec.name}")
                    self.assertIsNone(normalize_config(family, inactive)[spec.name])
                    self.assertNotIn(spec.name, engine_kwargs(family, inactive))
        broken = baseline_config("A4_TSMOM_V7"); broken["future_guess"] = 1
        with self.assertRaises(SchemaError):
            normalize_config("A4_TSMOM_V7", broken)

    def test_identity_serialization_and_inactive_axes(self) -> None:
        left = baseline_config("A3_STARTER_RETEST_V3"); left.update({"add_fraction": 0.0, "retest_depth_ATR": 0.25})
        right = dict(left); right["retest_depth_ATR"] = 1.5
        self.assertEqual(economic_address("A3_STARTER_RETEST_V3", left), economic_address("A3_STARTER_RETEST_V3", right))
        right["direction"] = "short"
        self.assertNotEqual(economic_address("A3_STARTER_RETEST_V3", left), economic_address("A3_STARTER_RETEST_V3", right))
        a1_left = baseline_config("A1_COMPRESSION_V2"); a1_left.update({"exit": "time_1d", "ATR_window_days": 10})
        a1_right = dict(a1_left); a1_right["ATR_window_days"] = 60
        self.assertEqual(gower_distance("A1_COMPRESSION_V2", a1_left, a1_right), 0.0)

    def test_invalid_economic_combinations(self) -> None:
        a3 = baseline_config("A3_STARTER_RETEST_V3"); a3.update({"starter_fraction": 1.0, "add_fraction": 0.75})
        with self.assertRaises(SchemaError): normalize_config("A3_STARTER_RETEST_V3", a3)
        a2 = baseline_config("A2_PRIOR_HIGH_RS_CONTEXT_V1"); a2.update({"RS_rank": "none", "BTC_ETH_context": "none", "breadth_dispersion": "none", "proximity_rank": "none", "reclaim_state": "none"})
        with self.assertRaises(SchemaError): normalize_config("A2_PRIOR_HIGH_RS_CONTEXT_V1", a2)
        a2["overlay_action"] = "parent_only"
        self.assertEqual(normalize_config("A2_PRIOR_HIGH_RS_CONTEXT_V1", a2)["overlay_action"], "parent_only")

    def test_generator_fixtures_and_outcome_free_budget(self) -> None:
        self.assertEqual(radical_inverse(1, 2), 0.5)
        self.assertEqual(radical_inverse(2, 2), 0.25)
        self.assertEqual(point("A4_TSMOM_V7", 0, 4), (0.2177356780122084, 0.5008439442248556, 0.5540227863773837, 0.5121447628219784))
        budget = optimize_budget({"A4_TSMOM_V7": 704, "A1_COMPRESSION_V2": 848, "A2_PRIOR_HIGH_RS_CONTEXT_V1": 863, "A3_STARTER_RETEST_V3": 1152, "KDA02B_SURVIVOR_ADJUDICATION_V1": 209})
        self.assertEqual(budget["target_total_attempt_rows"], 9088)
        self.assertNotIn("return", json.dumps(budget["inputs"]).lower())


class EngineAccountingControlTests(unittest.TestCase):
    def _leg(self, source: list[TradeBar], **changes: object):
        values = dict(entry_index=0, side=1, exit_name="time_1d", atr=None, fixed_target_r=None, structural_level=None, signal_reversal_close_ts=None, funding=(), cost_bps=14.0, funding_alignment="start_exclusive_end_inclusive", evaluation_start=datetime(2025, 1, 1, tzinfo=UTC), evaluation_end_exclusive=datetime(2026, 1, 1, tzinfo=UTC), gap_allowance_bps=-0.25)
        values.update(changes)
        return simulate_leg(source, **values)

    def test_every_family_real_dispatch_and_a2_exact_parent(self) -> None:
        for family in FAMILY_ORDER:
            result = end_to_end_family_probe(family)
            self.assertEqual(result["status"], "complete", family)
            self.assertGreater(len(result["observations"]), 0, family)
        parent, overlay = a2_context.counterpart_ids("a" * 64, "parent")
        self.assertNotEqual(parent, overlay)
        self.assertEqual(a2_context.parent_slot_id("A1_COMPRESSION_V2", "2024Q1", 5), "A1_COMPRESSION_V2:2024Q1:beam:05")
        with self.assertRaises(EngineInputError): a2_context.parent_slot_id("A1_COMPRESSION_V2", "2024Q1", 6)

    def test_a3_requires_activation_before_reclaim(self) -> None:
        frame = a3_frame()
        config = baseline_config("A3_STARTER_RETEST_V3")
        events = a3_starter_retest.evaluate(frame, config)
        self.assertTrue(events)
        event = events[0]
        result = a3_starter_retest.run_retest_state_machine(frame, direction=config["direction"], level=event["level"], atr=event["atr"], depth=float(event["retest_depth"]), starter_entry_index=event["entry_index"], starter_exit_ts=frame.five_minute_bars[-1].open_ts, window=str(event["retest_window"]))
        self.assertIn(result.status, {"complete", "unavailable_no_reclaim", "unavailable_invalidated"})
        if result.reclaim_index is not None:
            self.assertIsNotNone(result.activation_index)
            self.assertLess(result.activation_index, result.reclaim_index)

    def test_all_accounting_paths_and_favorable_funding_is_report_only(self) -> None:
        self.assertEqual(self._leg(trade_bars()).exit_reason, "time")
        stop = trade_bars(); stop[1] = replace(stop[1], close=95.0, low=94.9, high=100.2)
        self.assertEqual(self._leg(stop, exit_name="ATR_stop_1.5", atr=2.0).exit_reason, "structural_or_ATR_stop")
        target = trade_bars(); target[1] = replace(target[1], close=104.0, high=104.1)
        self.assertEqual(self._leg(target, exit_name="ATR_stop_1.5", atr=2.0, fixed_target_r=1.0).exit_reason, "fixed_target")
        trail = trade_bars(); trail[1] = replace(trail[1], close=105.0, high=105.1); trail[2] = replace(trail[2], open=105.0, close=101.0, high=105.1, low=100.9)
        self.assertEqual(self._leg(trail, exit_name="ATR_trail_2", atr=1.0).exit_reason, "trail")
        reversal = trade_bars()
        self.assertEqual(self._leg(reversal, exit_name="signal_reversal", signal_reversal_close_ts={reversal[1].close_ts}).exit_reason, "signal_reversal")
        structure = trade_bars(); structure[1] = replace(structure[1], close=98.0, low=97.9)
        self.assertEqual(self._leg(structure, exit_name="base_failure", structural_level=99.0).exit_reason, "structural_or_ATR_stop")
        payment = FundingPayment(trade_bars()[100].open_ts, trade_bars()[99].close_ts, -2.0)
        favorable = self._leg(trade_bars(), funding=(payment,))
        self.assertGreater(favorable.favorable_funding_bps, 0)
        self.assertGreater(favorable.reportable_net_bps, favorable.net_bps)
        self.assertAlmostEqual(favorable.net_bps, favorable.gross_bps - favorable.cost_bps - 0.25)
        parent = aggregate_parent_legs(favorable, 0.5, favorable, 0.25)
        self.assertAlmostEqual(parent["net_bps"], 0.75 * favorable.net_bps)
        self.assertTrue(accounting_probe()["pass"])

    def test_point_in_time_and_platform_firewalls(self) -> None:
        decision = datetime(2025, 1, 1, tzinfo=UTC)
        require_available(decision, decision)
        with self.assertRaises(EngineInputError): require_available(decision + timedelta(microseconds=1), decision)
        frame = a1_frame(); future = replace(frame.five_minute_bars[0], feature_available_ts=frame.five_minute_bars[0].close_ts + timedelta(seconds=1))
        with self.assertRaises(EngineInputError): replace(frame, five_minute_bars=(future, *frame.five_minute_bars[1:])).validate()
        with self.assertRaises(EngineInputError): replace(frame, platform="capitalcom_otc").validate()
        assert_rankable_interval(datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
        with self.assertRaises(ValueError): assert_rankable_interval(datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC) + timedelta(microseconds=1))
        self.assertEqual(average_rank_percentiles([1, 2, 2, 4]), [0.0, 0.5, 0.5, 1.0])
        self.assertEqual([liquidity_decile(value) for value in (0, 0.1, 0.999, 1)], [1, 2, 10, 10])

    def test_finite_controls_and_real_dispatch_coverage(self) -> None:
        source = []
        for family, ids in CONTROL_IDS.items():
            for fold in OUTER_FOLDS:
                for rank in range(1, 6):
                    for control_id in ids:
                        source.append({"family_id": family, "outer_fold_id": fold, "deterministic_beam_slot": rank, "control_id": control_id, "control_template_address_sha256": canonical_hash([family, fold, rank, control_id]), "prior_control_template_address_sha256": canonical_hash(["prior", family, fold, rank, control_id]), "seed": rank})
        rows = compile_controls(source)
        self.assertEqual(len(rows), 800)
        self.assertEqual(len({row["economic_address_sha256"] for row in rows}), 800)
        self.assertTrue(all(1 <= row["beam_rank"] <= 5 for row in rows))
        probe = control_engine_probe()
        self.assertTrue(probe["pass"], probe["failures"])
        frame = a1_frame(); metadata = dict(frame.metadata); metadata["control_signal_sign"] = -1
        changed, branch = transformed_inputs("A4_SIGN_PERMUTED_MAIN_NULL", [replace(frame, metadata=metadata)], 1)
        self.assertEqual(branch, "A4_SIGN_PERMUTED_MAIN_NULL")
        self.assertIsInstance(changed[0], FamilyInput)


class SelectionTests(unittest.TestCase):
    def observation(self, identifier: str, value: float, day: int) -> EventObservation:
        decision = datetime(2025, 1, day, tzinfo=UTC)
        return EventObservation(identifier, "BTC", f"2025-01-{day:02d}", "2025-01", 2025, value, value - 2, f"2025-01-{day:02d}", decision, decision + timedelta(minutes=5), decision + timedelta(days=1), eligible_days=365, holding_seconds_weighted=3600, eligible_symbol_seconds=365 * 86400)

    def test_streaming_materialized_and_empty_fold_arithmetic(self) -> None:
        events = [self.observation("e1", 3.0, 2), self.observation("e2", -1.0, 3)]
        self.assertEqual(aggregate_materialized(events), aggregate_streaming(iter(events)))
        inner = inner_fold_summary([1.0, None, -1.0, None])
        self.assertEqual(inner["vector"].count(-math.inf), 2)
        self.assertEqual(inner["p20_with_negative_infinity"], -math.inf)
        probe = aggregate_materialized_probe()
        self.assertTrue(probe["pass"])
        self.assertEqual(
            [item["status"] for item in probe["inner_fold_vector"]["vector"]],
            ["available", "unavailable_empty_fold", "available", "unavailable_empty_fold"],
        )
        self.assertEqual(
            probe["inner_fold_vector"]["p20_with_negative_infinity"],
            {"status": "negative_infinity_due_to_empty_fold"},
        )
        json.dumps(probe, allow_nan=False)

    def test_plateau_medoid_refinement_beam_dedup_and_routes(self) -> None:
        configs = []
        base = baseline_config("A4_TSMOM_V7")
        for index, lookback in enumerate((10, 20, 40, 60, 90)):
            config = dict(base); config["lookback_days"] = lookback; config["vol_window_days"] = (10, 20, 40, 60, 10)[index]
            address = economic_address("A4_TSMOM_V7", config)[1]
            configs.append({"canonical_economic_address_sha256": address, "config": normalize_config("A4_TSMOM_V7", config), "base_net_bps": 1 + index, "stress_net_bps": -1, "inner_fold_vector": [1, 1, 1, 1], "inner_nonempty_fraction": 1.0})
        regions = stable_neighborhoods("A4_TSMOM_V7", configs, radius=1.0)
        self.assertTrue(regions[0]["passed"])
        self.assertTrue(refinement_neighbors("A4_TSMOM_V7", configs[0]["config"], family_schemas["A4_TSMOM_V7"].priority_pairs))
        rows = [{**item, "stable_region": True, "accepted_trades": 30, "market_days": 20, "threshold_coverage": 0.7, "plateau_support_count": 5, "day_cluster_bootstrap_q05": 1, "p20_inner_fold": 1, "opportunity_frequency": 1, "complexity": 1, "event_ids": ["same"]} for item in configs]
        self.assertEqual(len(select_beam(rows)), 5)
        retained, rejected = deduplicate_event_overlap(rows)
        self.assertEqual((len(retained), len(rejected)), (1, 4))
        self.assertEqual(family_outer_vector({"f1": [1, 3]}, ["f1", "f2"]), [2, -math.inf])
        self.assertEqual(day_cluster_bootstrap_q05([1, 2, 3], 7, 50), day_cluster_bootstrap_q05([1, 2, 3], 7, 50))
        self.assertTrue(selection_route_probe()["pass"])
        self.assertEqual(adjudicate_route("A2_PRIOR_HIGH_RS_CONTEXT_V1", True, True, {"c": True}, base_positive=True, stress_positive=True, delay_positive=True, sample_sufficient=True), "context_uplift_candidate")

    def test_materialization_policy_is_frozen_and_deterministic(self) -> None:
        address = "1" * 64
        self.assertEqual(materialization_policy([{"canonical_economic_address_sha256": address, "beam_survivor": True, "passed": True}]), [address])


class RuntimeAuthorityAndReviewTests(unittest.TestCase):
    @staticmethod
    def status(pid: int, ppid: int, rss_kib: int) -> str:
        return f"Pid:\t{pid}\nPPid:\t{ppid}\nVmRSS:\t{rss_kib} kB\n"

    def test_process_tree_pid_churn_and_fail_closed_root(self) -> None:
        root = 100
        def reader(pid: int) -> str:
            if pid == root: return self.status(root, 1, 100)
            if pid == 101: raise ProcessLookupError()
            if pid == 102: return self.status(102, root, 200)
            raise AssertionError(pid)
        self.assertEqual(process_tree_rss(root, pid_list=lambda: [root, 101, 102], status_reader=reader), 300 * 1024)
        def esrch(pid: int) -> str:
            if pid == root: return self.status(root, 1, 100)
            raise OSError(errno.ESRCH, "gone")
        self.assertEqual(process_tree_rss(root, pid_list=lambda: [root, 101], status_reader=esrch), 100 * 1024)
        with self.assertRaises(ResourceGateError): process_tree_rss(root, pid_list=lambda: [root], status_reader=lambda _: (_ for _ in ()).throw(ProcessLookupError()))
        with self.assertRaises(PermissionError): process_tree_rss(root, pid_list=lambda: [root, 101], status_reader=lambda pid: self.status(root, 1, 100) if pid == root else (_ for _ in ()).throw(PermissionError()))

    def test_runtime_bound_stop_recovery_retry_and_no_hard_wall(self) -> None:
        with self.assertRaises(ResourceGateError): ResourceLimits(wall_time_seconds=1).validate()
        with tempfile.TemporaryDirectory() as raw:
            result = synthetic_recovery_canary(Path(raw))
            self.assertTrue(result["pass"], result)
            self.assertTrue(result["worker_death_retried"])
            self.assertTrue(result["continuous_resource_excursion_stopped"])
            self.assertTrue(result["idempotent"])

    def test_cache_authority_binds_physical_artifact_and_frame_content(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); artifact = root / "frames" / "a.json"; artifact.parent.mkdir(); artifact.write_bytes(b"frame-bytes")
            bindings = {"source_manifest_sha256": "a" * 64, "pit_universe_sha256": "b" * 64, "funding_manifest_sha256": "c" * 64, "cache_contract_sha256": "d" * 64}
            original = a1_frame(); metadata = {**original.metadata, "cache_bindings": bindings, "cache_artifact_path": "frames/a.json"}; frame = replace(original, metadata=metadata)
            records = [{"path": "frames/a.json", "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact), "frame_content_sha256": frame.content_sha256()}]
            cache = {"schema": "stage22_semantic_cache_manifest_v1", "platform": "kraken_native_linear_pf", "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)", **bindings, "artifacts": records, "artifact_inventory_sha256": canonical_hash(records)}
            manifest_path = root / "cache.json"; atomic_write_json(manifest_path, cache)
            campaign = {"execution_input_authority": {**bindings, "cache_manifest_contract": {"schema": "stage22_semantic_cache_manifest_v1"}}}
            CacheAuthority(manifest_path, root).validate(campaign, [frame])
            corrupted = replace(frame, symbol="TAMPER")
            with self.assertRaises(AuthorizationError): CacheAuthority(manifest_path, root).validate(campaign, [corrupted])
            artifact.write_bytes(b"tampered")
            with self.assertRaises(AuthorizationError): CacheAuthority(manifest_path, root).validate(campaign, [frame])

    def test_execution_authority_is_file_and_commit_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); subprocess.run(["git", "init", "-q", str(root)], check=True); subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True); subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            code = root / "code.py"; code.write_text("x=1\n", encoding="utf-8"); subprocess.run(["git", "-C", str(root), "add", "code.py"], check=True); subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            inventory = {"files": [{"path": "code.py", "sha256": sha256_file(code)}]}; inventory_path = root / "CODE_HASH_INVENTORY.json"; atomic_write_json(inventory_path, inventory)
            manifest = {"campaign_id": "kraken_core_liquid_discovery_campaign_003_code_first_stage22", "repository": {"implementation_commit": commit}, "primary_hashes": {"code_inventory": sha256_file(inventory_path)}}; manifest_path = root / "manifest.json"; atomic_write_json(manifest_path, manifest)
            request_path = root / "request.json"; atomic_write_json(request_path, {"campaign_id": manifest["campaign_id"], "final_campaign_manifest_sha256": sha256_file(manifest_path)})
            approval_path = root / "approval.json"; atomic_write_json(approval_path, {"campaign_id": manifest["campaign_id"], "final_campaign_manifest_sha256": sha256_file(manifest_path), "final_human_approval_request_sha256": sha256_file(request_path), "repository_implementation_commit": commit, "approved": True, "authorization": "launch_exact_frozen_stage22_campaign"})
            ExecutionAuthorization(manifest_path, request_path, approval_path, root).require()
            code.write_text("x=2\n", encoding="utf-8")
            with self.assertRaises(AuthorizationError): ExecutionAuthorization(manifest_path, request_path, approval_path, root).require()

    def test_registered_duplicate_and_review_binding_fail_closed(self) -> None:
        config = baseline_config("A4_TSMOM_V7"); address = economic_address("A4_TSMOM_V7", config)[1]
        row = {"campaign_id": "kraken_core_liquid_discovery_campaign_003_code_first_stage22", "family_id": "A4_TSMOM_V7", "config": config, "canonical_economic_address_sha256": address, "execution_disposition": "multiplicity_only_duplicate", "duplicate_of_executable_attempt_id": "x"}
        with self.assertRaises(ValueError): validate_registered_attempt(row)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = {"required_review_schema": "stage22_independent_preoutcome_review_v2", "review_bindings": {"implementation_commit": "a" * 40}}; atomic_write_json(root / "INDEPENDENT_REVIEW_TARGET.json", candidate)
            review = {"schema": "stage22_independent_preoutcome_review_v2", "verdict": "PASS", "blocking_findings": 0, "economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False, "bindings": {"implementation_commit": "a" * 40, "review_target_sha256": "wrong"}}
            path = root / "review.json"; atomic_write_json(path, review)
            with self.assertRaises(ValueError): _require_bound_review(root, path, "a" * 40)


if __name__ == "__main__":
    unittest.main()
