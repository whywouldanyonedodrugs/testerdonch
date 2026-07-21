from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from tools.core_liquid_campaign.accounting import simulate_leg
from tools.core_liquid_campaign.canonical import canonical_hash
from tools.core_liquid_campaign.controls import (
    _pcg_permute,
    compile_controls,
    control_semantic_signature,
    effective_seed,
    reconcile_control_duplicates,
)
from tools.core_liquid_campaign.engine_types import ThresholdPopulation
from tools.core_liquid_campaign.family_engines import a1_compression, a3_starter_retest, a4_tsmom
from tools.core_liquid_campaign.family_engines.common import EngineInputError
from tools.core_liquid_campaign.runtime import LazySupervisor, ResourceGateError, ResourceLimits
from tools.core_liquid_campaign.schema import OUTER_FOLDS, baseline_config, economic_address
from tools.core_liquid_campaign.selection import materialization_policy, select_beam
from tools.core_liquid_campaign.synthetic import a1_frame, a3_frame, a4_frame
from tools.core_liquid_campaign.family_engines import a2_context
from tools.core_liquid_campaign.terminal import TerminalContractError, reconcile_identities, terminal_package


class TemporalAndSelectionTests(unittest.TestCase):
    def test_nonfinite_threshold_population_fails_closed(self) -> None:
        frame = a4_frame()
        population = next(iter(frame.threshold_populations.values()))
        broken = replace(population, values=(*population.values[:-1], math.inf))
        with self.assertRaises(EngineInputError):
            broken.validate(decision_ts=frame.decision_ts)

    def test_a4_required_window_does_not_bridge_gap(self) -> None:
        frame = a4_frame()
        index = len(frame.five_minute_bars) // 2
        shifted = tuple(
            replace(bar, open_ts=bar.open_ts + timedelta(minutes=5), close_ts=bar.close_ts + timedelta(minutes=5), source_close_ts=bar.source_close_ts + timedelta(minutes=5), feature_available_ts=bar.feature_available_ts + timedelta(minutes=5)) if item >= index else bar
            for item, bar in enumerate(frame.five_minute_bars)
        )
        with self.assertRaises(EngineInputError):
            a4_tsmom.evaluate(replace(frame, five_minute_bars=shifted), baseline_config("A4_TSMOM_V7"))

    def test_active_exit_gap_is_local_unavailable(self) -> None:
        frame = a4_frame(); bars = [bar.trade_bar() for bar in frame.five_minute_bars[-700:]]
        bars = [*bars[:2], *bars[3:]]
        result = simulate_leg(bars, entry_index=0, side=1, exit_name="time_1d", atr=None, fixed_target_r=None, structural_level=None, signal_reversal_close_ts=None, funding=(), cost_bps=14, funding_alignment="start_exclusive_end_inclusive", evaluation_start=bars[0].open_ts, evaluation_end_exclusive=bars[-1].close_ts, gap_allowance_bps_per_hour=0.25)
        self.assertEqual((result.status, result.exit_reason), ("unavailable_temporal_gap", "temporal_gap"))

    def test_a1_owning_side_strict_q50_rearm(self) -> None:
        self.assertFalse(a1_compression.rearm_ready((1, -1), 1, {1: 0.50, -1: 0.1}))
        self.assertFalse(a1_compression.rearm_ready((1, -1), 1, {1: 0.7, -1: 0.1}))
        self.assertTrue(a1_compression.rearm_ready((1, -1), 1, {1: 0.49, -1: 0.9}))
        self.assertFalse(a1_compression.rearm_ready((1, -1), None, {1: 0.49, -1: 0.50}))
        self.assertTrue(a1_compression.rearm_ready((1, -1), None, {1: 0.49, -1: 0.49}))

    def test_a3_daily_window_does_not_bridge_gap(self) -> None:
        frame = a3_frame(); daily = list(frame.daily_bars)
        del daily[-10]
        with self.assertRaises(EngineInputError):
            a3_starter_retest.evaluate(replace(frame, daily_bars=tuple(daily)), baseline_config("A3_STARTER_RETEST_V3"))

    def test_a2_context_daily_window_does_not_bridge_gap(self) -> None:
        frame = a1_frame(); btc = list(frame.context.btc_daily); del btc[-10]
        broken = replace(frame, context=replace(frame.context, btc_daily=tuple(btc)))
        with self.assertRaises(EngineInputError):
            a2_context.named_context_multiplier(broken, "BTC_ETH", 1)

    def test_a1_gap_rebuild_requires_and_preserves_owner_state(self) -> None:
        frame = a1_frame(); bars = list(frame.five_minute_bars); del bars[200]
        broken = replace(frame, five_minute_bars=tuple(bars))
        with self.assertRaises(EngineInputError):
            a1_compression.evaluate(broken, baseline_config("A1_COMPRESSION_V2"))
        restored = replace(broken, metadata={**broken.metadata, "a1_pre_gap_owning_side": 1})
        a1_compression.evaluate(restored, baseline_config("A1_COMPRESSION_V2"))

    def test_pit_liquidity_uses_average_rank_ties(self) -> None:
        frame = a4_frame(); snapshot = dict(frame.metadata["pit_universe_snapshot"])
        eligible = list(snapshot["eligible_symbols"]); notionals = {symbol: float(1000 - index) for index, symbol in enumerate(eligible)}
        notionals[eligible[1]] = notionals[eligible[0]]
        groups = {}
        for symbol, value in notionals.items(): groups.setdefault(value, []).append(symbol)
        ranks = {}; position = 1
        for value in sorted(groups, reverse=True):
            members = groups[value]; rank = (position + position + len(members) - 1) / 2
            ranks.update({symbol: rank for symbol in members}); position += len(members)
        snapshot.update({"lagged_quote_notional": notionals, "lagged_liquidity_ranks": ranks, "top_n_symbols": {str(n): tuple(symbol for symbol, rank in sorted(ranks.items(), key=lambda item: (item[1], item[0])) if rank <= n) for n in (10, 20, 40)}})
        frame = replace(frame, metadata={**frame.metadata, "pit_universe_snapshot": snapshot})
        frame.require_pit_top_n(10)
        self.assertEqual(ranks[eligible[0]], 1.5)
        self.assertEqual(ranks[eligible[1]], 1.5)

    def test_selection_role_a2_pairing_and_event_dedup_are_gates(self) -> None:
        base = {"stable_region": True, "accepted_trades": 30, "market_days": 20, "base_net_bps": 1.0, "threshold_coverage": 0.8, "plateau_support_count": 5, "day_cluster_bootstrap_q05": 1.0, "p20_inner_fold": 1.0, "stress_net_bps": 1.0, "opportunity_frequency": 1.0, "complexity": 1.0}
        blocked = {**base, "canonical_economic_address_sha256": "1" * 64, "selection_role": "source_prior_anchor_not_main_beam", "family_id": "A1_COMPRESSION_V2"}
        self.assertEqual(select_beam([blocked]), [])
        corroborated = {**blocked, "context_corroborated": True}
        self.assertEqual(select_beam([corroborated]), [corroborated])
        a2 = {**base, "canonical_economic_address_sha256": "2" * 64, "selection_role": "conditional_parent_overlay_template", "family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1", "paired_uplift_mean": 1.0, "paired_uplift_bootstrap_q05": 0.1, "paired_coverage": 0.70, "parent_event_identity_match": True, "a2_main_null_pass": True}
        self.assertEqual(select_beam([a2]), [a2])
        self.assertEqual(select_beam([{**a2, "parent_event_identity_match": False}]), [])

    def test_materialization_two_near_misses_and_one_percent_bounds(self) -> None:
        rows = []
        for index in range(1000):
            rows.append({"family_id": "A4", "outer_fold_id": "2024Q1", "canonical_economic_address_sha256": f"{index:064x}", "passed": False, "integrity_valid": True, "failure_class": "base_positive", "near_miss": index < 3, "failed_eligibility_gate_count": 1, "plateau_support_count": index, "day_cluster_bootstrap_q05": 0, "p20_inner_fold": 0, "stress_net_bps": 0, "opportunity_frequency": 0, "complexity": 1})
        selected = materialization_policy(rows)
        self.assertGreaterEqual(len(selected), 10)
        self.assertLessEqual(len(selected), 102)  # two near misses plus bounded failed audit


class ControlAndRuntimeTests(unittest.TestCase):
    def _source_controls(self):
        rows = []
        control_ids = {
            "A4_TSMOM_V7": ("A4_SIGN_PERMUTED_MAIN_NULL", "A4_GENERIC_SIGNED_RETURN", "A4_VOL_SCALING_REMOVED", "A4_PATH_COMPONENT_REMOVED", "A4_CONTEXT_REMOVED"),
            "A1_COMPRESSION_V2": ("A1_MATCHED_PSEUDO_EVENT_MAIN_NULL", "A1_PRICE_ONLY_IMPULSE", "A1_CONTRACTION_REMOVED", "A1_SMOOTHNESS_REMOVED", "A1_CONTEXT_REMOVED"),
            "A2_PRIOR_HIGH_RS_CONTEXT_V1": ("A2_CONTEXT_PERMUTED_MAIN_NULL", "A2_PARENT_ONLY", "A2_PRIOR_HIGH_REMOVED", "A2_RS_REMOVED", "A2_EXTERNAL_CONTEXT_REMOVED"),
            "A3_STARTER_RETEST_V3": ("A3_RETEST_TIME_PERMUTED_MAIN_NULL", "A3_STARTER_ONLY", "A3_MATCHED_PSEUDO_EVENT", "A3_CONFIRMATION_REMOVED", "A3_CONTEXT_REMOVED"),
        }
        for family, ids in control_ids.items():
            for fold in OUTER_FOLDS:
                for rank in range(1, 6):
                    for control_id in ids:
                        rows.append({"family_id": family, "outer_fold_id": fold, "deterministic_beam_slot": rank, "control_id": control_id, "control_template_address_sha256": canonical_hash([family, fold, rank, control_id]), "prior_control_template_address_sha256": canonical_hash(["prior", family, fold, rank, control_id]), "seed": 1})
        return rows

    def test_fixed_control_seed_fixture(self) -> None:
        payload = {"campaign_id": "kraken_core_liquid_discovery_campaign_003_code_first_stage22", "family": "A4_TSMOM_V7", "fold": "2024Q1", "parent_slot": "A4_TSMOM_V7:2024Q1:beam:01", "control_id": "A4_SIGN_PERMUTED_MAIN_NULL", "replicate_index": 0, "transformation_allocator_version": "stage22_exact_control_dispatch_v2"}
        self.assertEqual(effective_seed(payload), 14734187594875452374)
        import numpy as np
        vector = [1, -1, 1, -1, 0, 1]
        self.assertEqual(_pcg_permute(vector, np.random.Generator(np.random.PCG64(14734187594875452374))), [1, -1, 1, 1, 0, -1])

    def test_parent_identical_and_mutual_control_duplicates_are_explicit(self) -> None:
        controls = compile_controls(self._source_controls())
        config = baseline_config("A4_TSMOM_V7"); config.update({"annualized_vol_target": "none", "context_overlay": "none", "path_smoothness_quantile_min": "none"})
        parent = {"executable_attempt_id": "parent", "family_id": "A4_TSMOM_V7", "config": config, "canonical_economic_address_sha256": economic_address("A4_TSMOM_V7", config)[1]}
        slot = "A4_TSMOM_V7:2024Q1:beam:01"
        selected = [row for row in controls if row["parent_slot"] == slot]
        resolved = reconcile_control_duplicates(selected, {slot: parent})
        by_id = {row["control_id"]: row for row in resolved}
        self.assertEqual(by_id["A4_VOL_SCALING_REMOVED"]["execution_status"], "unavailable_duplicate_address")
        self.assertEqual(by_id["A4_CONTEXT_REMOVED"]["execution_status"], "unavailable_duplicate_address")

    def test_missing_control_parent_is_not_reassigned(self) -> None:
        control = compile_controls(self._source_controls())[0]
        resolved = reconcile_control_duplicates([control], {})[0]
        self.assertEqual(resolved["execution_status"], "unavailable_no_parent")
        self.assertIsNone(resolved["resolved_parent_executable_attempt_id"])

    def test_large_worker_result_cannot_pipe_deadlock_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); heartbeats = []
            limits = ResourceLimits(max_workers=1, max_jobs_in_flight=1, max_output_bytes=32 * 1024**2, minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0, heartbeat_seconds=1, monitor_interval_seconds=0.001)
            payload = "x" * (2 * 1024 * 1024)
            jobs = [("large", lambda: {"registered_attempt_id": "large", "payload": payload})]
            validator = lambda job, result: result.get("registered_attempt_id") == job
            first = LazySupervisor(root, limits, heartbeat=lambda value: heartbeats.append(value) or True, real_unit_validator=validator).run(iter(jobs))
            replay = LazySupervisor(root, limits, heartbeat=lambda value: True, real_unit_validator=validator).run(iter(jobs))
            self.assertEqual(first["status"], "complete")
            self.assertEqual(first["completed"], replay["completed"])

    def test_engine_input_error_is_persisted_local_unavailable_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            limits = ResourceLimits(max_workers=1, max_jobs_in_flight=1, minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0)
            def unavailable(): raise EngineInputError("local missing bar")
            state = LazySupervisor(root, limits).run(iter([("local", unavailable)]))
            self.assertEqual(state["attempts"]["local"], 1)
            artifact = next((root / "artifacts").glob("*.json"))
            self.assertEqual(json.loads(artifact.read_text())["result"]["status"], "unavailable_data")

    def test_resume_fails_closed_on_identity_binding_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            limits = ResourceLimits(max_workers=1, max_jobs_in_flight=1, minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0)
            job = ("bound", lambda: {"registered_attempt_id": "bound"})
            LazySupervisor(root, limits, identity_bindings={"manifest_sha256": "a" * 64}).run(iter([job]))
            with self.assertRaises(ResourceGateError):
                LazySupervisor(root, limits, identity_bindings={"manifest_sha256": "b" * 64}).run(iter([job]))


class TerminalTests(unittest.TestCase):
    JOBS = {"schema": "fixture", "pass": True}
    def test_terminal_reconciliation_has_no_silent_omission(self) -> None:
        rows = [{"attempt_id": "a", "terminal_status": "completed"}, {"attempt_id": "b", "terminal_status": "unavailable_data"}]
        self.assertTrue(reconcile_identities(["a", "b"], rows, "attempt_id")["pass"])
        self.assertFalse(reconcile_identities(["a", "b", "c"], rows, "attempt_id")["pass"])

    def test_completion_requires_workers_stopped_and_full_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaises(TerminalContractError):
                terminal_package(root, attempt_ids=["a"], control_ids=[], attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}], control_rows=[], routes=[], forensics=[], all_workers_stopped=False)
            payload = terminal_package(root, attempt_ids=["a"], control_ids=["c"], attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}], control_rows=[{"control_attempt_id": "c", "terminal_status": "unavailable_no_parent"}], routes=[{"family": "fixture", "route": "translation_rejected"}], forensics=[{"family": "fixture", "status": "complete"}], all_workers_stopped=True, job_reconciliation=self.JOBS)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(json.loads((root / "TERMINAL_PACKAGE.json").read_text())["artifact_inventory_sha256"], payload["artifact_inventory_sha256"])
            with self.assertRaises(TerminalContractError):
                terminal_package(root / "failed", attempt_ids=["a"], control_ids=[], attempt_rows=[{"attempt_id": "a", "terminal_status": "mechanical_failure"}], control_rows=[], routes=[{"family": "fixture", "route": "translation_rejected"}], forensics=[{"family": "fixture"}], all_workers_stopped=True, job_reconciliation=self.JOBS)

    def test_bound_stop_package_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            payload = terminal_package(Path(raw), attempt_ids=["a"], control_ids=[], attempt_rows=[], control_rows=[], routes=[], forensics=[], all_workers_stopped=True, bound_stop=True)
            self.assertTrue(payload["resumable"])
            self.assertEqual(payload["status"], "global_bound_stop_incomplete")


if __name__ == "__main__":
    unittest.main()
