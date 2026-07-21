from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.core_liquid_campaign.cache import SemanticCacheWriter, _restore_metadata
from tools.core_liquid_campaign.canonical import atomic_write_json, canonical_hash, pretty_json_bytes, sha256_file
from tools.core_liquid_campaign.a1_state import initial_state, transition
from tools.core_liquid_campaign.family_engines.common import EngineInputError, weak_percentile, weak_percentile_prevalidated_sorted
from tools.core_liquid_campaign.campaign import CampaignOrchestrator
from tools.core_liquid_campaign.controls import CONTROL_IDS, derive_control_inputs, execute_control
from tools.core_liquid_campaign.executor import dispatch_registered_attempt
from tools.core_liquid_campaign.schema import CAMPAIGN_ID, baseline_config, economic_address, normalize_config
from tools.core_liquid_campaign.shadow_payoff import ShadowPayoffProvider
from tools.core_liquid_campaign.synthetic import a1_frame, a3_frame, a4_frame, frame_for_family
from tools.core_liquid_campaign.terminal import TerminalContractError, independent_terminal_recomputation, terminal_package, verify_terminal_inventory
from tools.core_liquid_campaign.runtime import LazySupervisor, ResourceLimits
from tools.core_liquid_campaign.production_readiness_gate import _a1_state_gate
from tools.core_liquid_campaign.production_inputs import _thresholds


class Stage24KnownDefectTests(unittest.TestCase):
    @staticmethod
    def _attempt(family: str, attempt_id: str) -> dict[str, object]:
        config = normalize_config(family, baseline_config(family))
        return {
            "campaign_id": CAMPAIGN_ID,
            "family_id": family,
            "config": config,
            "execution_disposition": "execute_once",
            "executable_attempt_id": attempt_id,
            "canonical_economic_address_sha256": economic_address(family, config)[1],
            "duplicate_of_executable_attempt_id": None,
        }

    @staticmethod
    def _control_result_signature(result: dict[str, object]) -> str:
        def serializable(value: object) -> object:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {str(key): serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            return value

        observations = result.get("observations", ())
        return canonical_hash(serializable({
            "status": result.get("status"),
            "event_ids": sorted(item.event_id for item in observations),
            "ledger": result.get("ledger", ()),
            "aggregate": result.get("aggregate", {}),
            "allocation_unavailable": result.get("allocation_unavailable", ()),
        }))

    def test_prevalidated_percentile_is_exactly_equivalent(self) -> None:
        population = tuple(float(index % 37) for index in range(100))
        ordered = tuple(sorted(population))
        for value in (-1.0, 0.0, 12.5, 36.0, 99.0):
            self.assertEqual(weak_percentile(value, population), weak_percentile_prevalidated_sorted(value, ordered))

    def test_a1_persistent_state_covers_owner_gap_cooldown_and_strict_rearm(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state = initial_state()
        state = transition(state, timestamp=start, action="history_complete")
        with self.assertRaises(EngineInputError):
            transition(state, timestamp=start, action="rearm", percentiles={1: 0.49, -1: 0.49})
        state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.49, -1: 0.49})
        state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=-1)
        state = transition(state, timestamp=start + timedelta(minutes=15), action="base")
        state = transition(state, timestamp=start + timedelta(minutes=20), action="confirmation")
        state = transition(state, timestamp=start + timedelta(minutes=25), action="gap")
        self.assertEqual((state.state, state.owner, state.terminal_episode_reason), ("history_rebuild", -1, "temporal_gap"))
        state = transition(state, timestamp=start + timedelta(days=1), action="history_complete")
        state = transition(state, timestamp=start + timedelta(days=1, minutes=5), action="rearm", percentiles={-1: 0.50, 1: 0.1})
        self.assertEqual("disarmed", state.state)
        state = transition(state, timestamp=start + timedelta(days=1, minutes=10), action="rearm", percentiles={-1: 0.49, 1: 0.9})
        self.assertEqual("armed", state.state)
        self.assertIsNone(state.owner)

    def test_a1_persistent_state_round_trips_cooldown(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state = transition(initial_state(), timestamp=start, action="history_complete")
        state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.1, -1: 0.1})
        state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=1)
        state = transition(
            state,
            timestamp=start + timedelta(minutes=15),
            action="episode_terminal",
            terminal_reason="actual_exit",
            cooldown_until=start + timedelta(hours=1),
        )
        state = transition(state, timestamp=start + timedelta(minutes=20), action="rearm", percentiles={1: 0.1})
        self.assertEqual("cooldown", state.state)
        state = transition(state, timestamp=start + timedelta(hours=1), action="cooldown_expired")
        self.assertEqual("disarmed", state.state)
        self.assertEqual(6, state.state_generation)

    def test_a1_gap_at_each_active_state_preserves_owner_and_reason(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for side in (-1, 1):
            state = transition(initial_state(), timestamp=start, action="history_complete")
            state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.1, -1: 0.1})
            state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=side)
            active = [state]
            active.append(transition(active[-1], timestamp=start + timedelta(minutes=15), action="base"))
            active.append(transition(active[-1], timestamp=start + timedelta(minutes=20), action="confirmation"))
            for offset, candidate in enumerate(active, start=3):
                gapped = transition(candidate, timestamp=start + timedelta(hours=1, minutes=offset * 5), action="gap")
                self.assertEqual(("history_rebuild", side, "temporal_gap"), (gapped.state, gapped.owner, gapped.terminal_episode_reason))

    def test_production_a1_engine_consumes_persisted_start_state_and_fails_closed_mid_episode(self) -> None:
        frame = a1_frame()
        metadata = {**frame.metadata, "production_input": True, "a1_persistent_state": initial_state().payload()}
        a1_frame_from_rebuild = replace(frame, metadata=metadata)
        dispatch_registered_attempt(
            self._attempt("A1_COMPRESSION_V2", "persisted-a1"),
            (a1_frame_from_rebuild,),
            registry_by_id={"persisted-a1": self._attempt("A1_COMPRESSION_V2", "persisted-a1")},
            payoff_provider=ShadowPayoffProvider("stage24-a1-persisted-state"),
        )
        start = frame.five_minute_bars[0].open_ts - timedelta(minutes=15)
        state = transition(initial_state(), timestamp=start, action="history_complete")
        state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.1, -1: 0.1})
        state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=1)
        broken = replace(frame, metadata={**metadata, "a1_persistent_state": state.payload()})
        with self.assertRaises(EngineInputError):
            dispatch_registered_attempt(
                self._attempt("A1_COMPRESSION_V2", "persisted-a1"),
                (broken,), registry_by_id={"persisted-a1": self._attempt("A1_COMPRESSION_V2", "persisted-a1")},
                payoff_provider=ShadowPayoffProvider("stage24-a1-persisted-state"),
            )

    def test_production_gate_a1_evidence_is_canonical_json_serializable(self) -> None:
        evidence = _a1_state_gate()
        self.assertEqual("pass", evidence["status"])
        payload = pretty_json_bytes(evidence)
        self.assertIn(b'"last_valid_ts": "2025-01-01T00:25:00+00:00"', payload)

    def test_cache_restores_a1_cooldown_deadline_as_utc_datetime(self) -> None:
        value = _restore_metadata({"cooldown_until": "2025-01-01T01:00:00+00:00"})
        self.assertEqual(datetime(2025, 1, 1, 1, tzinfo=timezone.utc), value["cooldown_until"])

    def test_typed_kda_cache_unavailability_becomes_terminal_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.json"
            atomic_write_json(source, {"fixture": True})
            source_record = {"role": "fixture", "path": "source.json", "bytes": source.stat().st_size, "sha256": sha256_file(source)}
            authority = {
                "platform": "kraken_native_linear_pf",
                "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
                "source_manifest_sha256": source_record["sha256"], "pit_universe_sha256": "b" * 64,
                "funding_manifest_sha256": "c" * 64, "cache_contract_sha256": "d" * 64,
                "fold_graph_sha256": "e" * 64, "rankable_funding_package_sha256": "f" * 64,
                "source_records": [source_record], "cache_manifest_contract": {"schema": "stage22_semantic_cache_manifest_v1"},
            }
            writer = SemanticCacheWriter(root / "cache", authority, authority_root=root, synthetic_only=False)
            partition = {
                "phase": "outer_evaluation", "outer_fold_id": "2024Q1", "inner_fold_id": None,
                "training_start": datetime(2023, 1, 1, tzinfo=timezone.utc),
                "training_end_exclusive": datetime(2023, 12, 22, tzinfo=timezone.utc),
                "evaluation_start": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "evaluation_end_exclusive": datetime(2024, 4, 1, tzinfo=timezone.utc),
            }
            record = writer.add_unavailable(
                family_id="KDA02B_SURVIVOR_ADJUDICATION_V1", partition=partition,
                reason="exact raw decision fields unavailable", authority_sha256="a" * 64,
            )
            cache = {"artifacts": [], "typed_unavailable": [record]}
            row = self._attempt("KDA02B_SURVIVOR_ADJUDICATION_V1", "kda-row")
            orchestrator = object.__new__(CampaignOrchestrator)
            jobs = list(orchestrator._kda_jobs([row], {"kda-row": row}, cache))
            self.assertEqual(1, len(jobs))
            result = jobs[0][1]()
            self.assertEqual(("unavailable_data", "explicit_empty_unavailable_observation"), (result["status"], result["materialization"]))
            self.assertEqual("a" * 64, result["authority_sha256"])

    def test_early_inner_fold_persists_long_a4_features_as_unavailable(self) -> None:
        frame = a4_frame()
        bars = frame.five_minute_bars
        daily = frame.daily_bars
        populations, unavailable = _thresholds(
            {"PF_XBTUSD": bars, "PF_ETHUSD": bars},
            {"PF_XBTUSD": daily, "PF_ETHUSD": daily},
            target="PF_XBTUSD",
            training_start=bars[0].open_ts,
            training_end=frame.decision_ts,
        )
        name = "A4_ensemble:ema_slope:lookback=180:volatility=close_to_close"
        self.assertNotIn(name, populations)
        self.assertIn(name, {row["feature_signature"] for row in unavailable})

    def test_shadow_provider_uses_actual_accounting_without_real_post_entry_data(self) -> None:
        config = baseline_config("A4_TSMOM_V7")
        _, address = economic_address("A4_TSMOM_V7", config)
        attempt_id = "stage24-shadow-a4"
        row = {
            "campaign_id": CAMPAIGN_ID,
            "family_id": "A4_TSMOM_V7",
            "config": config,
            "execution_disposition": "execute_once",
            "executable_attempt_id": attempt_id,
            "canonical_economic_address_sha256": address,
            "duplicate_of_executable_attempt_id": None,
        }
        provider = ShadowPayoffProvider("stage24-test")
        result = dispatch_registered_attempt(
            row,
            (a4_frame(config),),
            registry_by_id={attempt_id: row},
            payoff_provider=provider,
        )
        self.assertEqual("complete", result["status"])
        self.assertGreater(provider.calls, 0)
        self.assertFalse(provider.attestation()["economic_outcomes_opened"])
        self.assertTrue(all(item["shadow_only"] for item in result["ledger"]))
        self.assertTrue(all(item["actual_accounting_path_executed"] for item in result["ledger"]))
        self.assertTrue(all(item["real_post_entry_rows_opened"] == 0 for item in result["ledger"]))

    def test_every_deterministic_control_receives_parent_frames(self) -> None:
        for family, controls in CONTROL_IDS.items():
            engine_family = family
            config = baseline_config(engine_family)
            parent = {
                "family_id": engine_family,
                "config": config,
                "executable_attempt_id": f"parent-{family}",
                "canonical_economic_address_sha256": economic_address(engine_family, config)[1],
            }
            frames = [frame_for_family(engine_family, config)]
            for control_id in controls:
                if control_id in {
                    "A4_SIGN_PERMUTED_MAIN_NULL", "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL",
                    "A2_CONTEXT_PERMUTED_MAIN_NULL", "A3_RETEST_TIME_PERMUTED_MAIN_NULL",
                    "A3_MATCHED_PSEUDO_EVENT",
                }:
                    continue
                control = {"control_id": control_id, "effective_seed": 1, "economic_address_sha256": "a" * 64}
                transformed, directives, unavailable = derive_control_inputs(
                    control, parent, {"observations": [], "ledger": []}, frames,
                )
                self.assertEqual(transformed, frames, control_id)
                self.assertEqual(directives, {}, control_id)
                self.assertEqual(unavailable, [], control_id)

    def test_all_twenty_controls_execute_nonempty_and_replay_invariant(self) -> None:
        utc = timezone.utc
        anchors = (datetime(2025, 6, 1, tzinfo=utc), datetime(2025, 6, 15, tzinfo=utc))
        fixtures: dict[str, tuple[dict[str, object], list[object], dict[str, dict[str, object]], dict[str, object] | None]] = {}
        a4 = self._attempt("A4_TSMOM_V7", "parent-a4")
        fixtures["A4_TSMOM_V7"] = (
            a4,
            [a4_frame(a4["config"], signal_sign=1, anchor=anchors[0]), a4_frame(a4["config"], signal_sign=-1, anchor=anchors[1])],
            {"parent-a4": a4},
            None,
        )
        a1 = self._attempt("A1_COMPRESSION_V2", "parent-a1")
        fixtures["A1_COMPRESSION_V2"] = (
            a1,
            [a1_frame(a1["config"], anchor=anchors[0]), a1_frame(a1["config"], anchor=anchors[1])],
            {"parent-a1": a1},
            None,
        )
        a3 = self._attempt("A3_STARTER_RETEST_V3", "parent-a3")
        fixtures["A3_STARTER_RETEST_V3"] = (
            a3,
            [a3_frame(a3["config"], anchor=anchors[0]), a3_frame(a3["config"], anchor=anchors[1])],
            {"parent-a3": a3},
            None,
        )
        a2_parent = self._attempt("A1_COMPRESSION_V2", "a2-parent")
        a2_config = normalize_config("A2_PRIOR_HIGH_RS_CONTEXT_V1", baseline_config("A2_PRIOR_HIGH_RS_CONTEXT_V1"))
        template = canonical_hash({"mode": "beam_slot", "parent_slot": "A1_COMPRESSION_V2:2024Q1:beam:01"})
        a2 = {
            "campaign_id": CAMPAIGN_ID, "family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1",
            "config": a2_config, "execution_disposition": "execute_if_parent_available",
            "executable_attempt_id": "parent-a2", "canonical_economic_address_sha256": economic_address("A2_PRIOR_HIGH_RS_CONTEXT_V1", a2_config)[1],
            "duplicate_of_executable_attempt_id": None, "parent_binding_template_id": template,
            "parent_only_counterpart_id": "parent-only", "overlay_counterpart_id": "overlay",
        }
        first = a1_frame(a2_parent["config"], anchor=anchors[0])
        second = a1_frame(a2_parent["config"], anchor=anchors[1])
        by_lookback = {key: dict(value) for key, value in second.context.cross_section_returns_by_lookback.items()}
        by_lookback[20][second.symbol] = -0.08
        second_context = replace(
            second.context,
            cross_section_returns=dict(by_lookback[20]),
            cross_section_returns_by_lookback=by_lookback,
            source_sha256=canonical_hash({"stage24_control_fixture": "second_context"}),
        )
        second = replace(second, context=second_context)
        binding = {
            "parent_binding_template_id": template, "parent_executable_attempt_id": "a2-parent",
            "parent_only_counterpart_id": "parent-only", "overlay_counterpart_id": "overlay",
        }
        fixtures["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = (a2, [first, second], {"parent-a2": a2, "a2-parent": a2_parent}, binding)

        executed = []
        for family, control_ids in CONTROL_IDS.items():
            parent, frames, registry, binding = fixtures[family]
            for control_id in control_ids:
                seed = 3 if control_id in {"A4_SIGN_PERMUTED_MAIN_NULL", "A2_CONTEXT_PERMUTED_MAIN_NULL", "A3_RETEST_TIME_PERMUTED_MAIN_NULL"} else 1
                control = {
                    "family": family, "control_id": control_id, "effective_seed": seed,
                    "economic_address_sha256": canonical_hash({"control": control_id}),
                    "control_attempt_id": f"stage24-{control_id}", "execution_status": "execute_once",
                }
                kwargs = {
                    "registry_by_id": registry,
                    "payoff_provider": ShadowPayoffProvider("stage24-all-controls"),
                }
                if binding is not None:
                    kwargs.update({"parent_binding": binding, "parent_frames": frames})
                forward = execute_control(control, parent, frames, **kwargs)
                reverse = execute_control(control, parent, list(reversed(frames)), **kwargs)
                self.assertEqual("complete", forward["status"], control_id)
                self.assertGreater(len(forward["observations"]), 0, control_id)
                self.assertEqual(self._control_result_signature(forward), self._control_result_signature(reverse), control_id)
                executed.append(control_id)
        self.assertEqual(20, len(executed))
        self.assertEqual(20, len(set(executed)))

    def test_missing_a2_parent_is_an_explicit_empty_fold_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stage = root / "inner_development"
            artifact = stage / "artifacts/result.json"
            payload = {
                "status": "unavailable_no_parent",
                "registered_attempt_id": "a2",
                "registered_job_id": "inner:2024Q1:M_202307:a2",
                "aggregate": {}, "observation_count": 0,
                "day_base_net_bps": {}, "event_ids": [],
            }
            atomic_write_json(artifact, {"result": payload})
            atomic_write_json(stage / "markers/marker.json", {
                "artifact": "artifacts/result.json", "artifact_sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
            })
            config = baseline_config("A2_PRIOR_HIGH_RS_CONTEXT_V1")
            row = {
                "family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1", "config": config,
                "executable_attempt_id": "a2", "canonical_economic_address_sha256": economic_address("A2_PRIOR_HIGH_RS_CONTEXT_V1", config)[1],
                "selection_role": "conditional_parent_overlay_template",
            }
            orchestrator = object.__new__(CampaignOrchestrator)
            orchestrator.run_root = root
            self.assertEqual(orchestrator._freeze_beams([row]), [])

    def test_completed_terminal_requires_forensics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(TerminalContractError):
                terminal_package(
                    Path(raw), attempt_ids=["a"], control_ids=[],
                    attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}], control_rows=[],
                    routes=[{"family": "A4_TSMOM_V7", "route": "translation_rejected"}],
                    forensics=[], all_workers_stopped=True,
                    job_reconciliation={"pass": True},
                )

    def test_terminal_inventory_is_last_and_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            terminal_package(
                root,
                attempt_ids=["a"], control_ids=["c"],
                attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}],
                control_rows=[{"control_attempt_id": "c", "terminal_status": "completed"}],
                routes=[{"family": "A4_TSMOM_V7", "route": "translation_rejected"}],
                forensics=[{"family": "A4_TSMOM_V7", "event_count": 0}],
                    all_workers_stopped=True,
                    job_reconciliation={"pass": True},
                )
            self.assertEqual("pass", verify_terminal_inventory(root)["status"])
            (root / "FORENSIC_RECORDS.json").write_bytes(b"tampered")
            with self.assertRaises(TerminalContractError):
                verify_terminal_inventory(root)

    def test_terminal_independent_recomputation_round_trips_frozen_sources(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload = terminal_package(
                root, attempt_ids=["a"], control_ids=["c"],
                attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}],
                control_rows=[{"control_attempt_id": "c", "terminal_status": "unavailable_no_parent"}],
                routes=[{"family": "fixture", "route": "shadow_verified"}],
                forensics=[{"family": "fixture", "status": "shadow_verified"}],
                all_workers_stopped=True, job_reconciliation={"pass": True},
            )
            stored = json.loads((root / "INDEPENDENT_RECOMPUTATION.json").read_text())
            replay = independent_terminal_recomputation(root, attempt_ids=["a"], control_ids=["c"], require_complete=True)
            self.assertEqual(stored, replay)
            self.assertEqual(sha256_file(root / "INDEPENDENT_RECOMPUTATION.json"), payload["independent_recomputation_sha256"])
            self.assertEqual("pass", verify_terminal_inventory(root)["status"])

    def test_stale_scheduled_heartbeat_stops_workers_without_late_commit(self) -> None:
        import time

        class Clock:
            value = 0.0

            def __call__(self) -> float:
                self.value += 1000.0
                return self.value

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            late = root / "late-write"
            deliveries = {"count": 0}

            def heartbeat(_payload: object) -> bool:
                deliveries["count"] += 1
                return deliveries["count"] == 1

            def slow() -> dict[str, object]:
                time.sleep(1.0)
                late.write_text("unsafe\n", encoding="utf-8")
                return {"registered_attempt_id": "slow", "status": "complete", "aggregate": {}}

            limits = ResourceLimits(
                max_workers=1, max_jobs_in_flight=1, max_output_bytes=32 * 1024**2,
                minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0,
                heartbeat_seconds=1, monitor_interval_seconds=0.001,
            )
            state = LazySupervisor(root, limits, heartbeat=heartbeat, monotonic=Clock()).run(iter([("slow", slow)]))
            time.sleep(0.05)
            self.assertEqual("global_resumable_bound_stop_heartbeat_stale", state["status"])
            self.assertTrue(state["all_workers_stopped"])
            self.assertFalse(late.exists())

    def test_abrupt_worker_pipe_eof_requeues_and_recovers_once(self) -> None:
        import os

        class Clock:
            value = 0.0

            def __call__(self) -> float:
                self.value += 100.0
                return self.value

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first_attempt = root / "first-attempt"

            def abrupt_then_complete() -> dict[str, object]:
                if not first_attempt.exists():
                    first_attempt.write_text("abrupt worker exit\n", encoding="utf-8")
                    os._exit(17)
                return {"registered_attempt_id": "abrupt", "status": "complete", "aggregate": {}}

            limits = ResourceLimits(
                max_workers=1, max_jobs_in_flight=1, max_output_bytes=32 * 1024**2,
                minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0,
                heartbeat_seconds=1800, monitor_interval_seconds=0.001,
            )
            state = LazySupervisor(root, limits, heartbeat=lambda _payload: True, monotonic=Clock()).run(
                iter([("abrupt", abrupt_then_complete)])
            )
            self.assertEqual("complete", state["status"])
            self.assertEqual(2, state["attempts"]["abrupt"])
            self.assertEqual(1, state["completed_count"])
            self.assertTrue(state["all_workers_stopped"])
            self.assertEqual([], state["worker_pids"])


if __name__ == "__main__":
    unittest.main()
