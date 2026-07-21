from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.core_liquid_campaign.canonical import atomic_write_json, canonical_hash
from tools.core_liquid_campaign.a1_state import initial_state, transition
from tools.core_liquid_campaign.family_engines.common import EngineInputError, weak_percentile, weak_percentile_prevalidated_sorted
from tools.core_liquid_campaign.campaign import CampaignOrchestrator
from tools.core_liquid_campaign.controls import CONTROL_IDS, derive_control_inputs
from tools.core_liquid_campaign.executor import dispatch_registered_attempt
from tools.core_liquid_campaign.schema import CAMPAIGN_ID, baseline_config, economic_address
from tools.core_liquid_campaign.shadow_payoff import ShadowPayoffProvider
from tools.core_liquid_campaign.synthetic import a4_frame, frame_for_family
from tools.core_liquid_campaign.terminal import TerminalContractError, terminal_package, verify_terminal_inventory


class Stage24KnownDefectTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
