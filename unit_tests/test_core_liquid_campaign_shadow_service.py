from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.core_liquid_campaign.canonical import atomic_write_json, atomic_write_jsonl, sha256_file
from tools.core_liquid_campaign.campaign import CampaignContractError
from tools.core_liquid_campaign.schema import CAMPAIGN_ID
from tools.core_liquid_campaign.shadow_campaign import (
    BoundedShadowPopulationSchedule,
    BoundedShadowKDA02BAdapter,
    ShadowCampaignAuthorization,
    ShadowCampaignPacketError,
    build_bounded_shadow_packet,
)
from tools.core_liquid_campaign.lazy_production_inputs import FamilyDecisionLocator
from tools.core_liquid_campaign.shadow_service import run_shadow_service
from tools.core_liquid_campaign.shadow_service import ProductionShadowCampaignOrchestrator


class Stage24RealShadowServiceTests(unittest.TestCase):
    @staticmethod
    def _event_locator_policy(days: int = 1) -> dict[str, object]:
        return {
            "schema": "stage24_shadow_event_locator_policy_v2",
            "selection": "actual_production_enumerator_pre_entry_event_locators",
            "target_eligible_event_days_per_attempt_partition": days,
            "maximum_candidate_locators_per_attempt_partition": 100,
            "empty_attempt_partitions_preserved": True,
            "real_post_entry_values_used": False,
            "economic_values_used_for_selection": False,
            "synthetic_payoff_generated_after_locator_freeze": True,
            "full_launch_population_authority_preserved": True,
        }

    @patch("tools.core_liquid_campaign.executor._generate_events")
    def test_event_locator_sampler_yields_only_actual_enumerator_match(self, generate: Mock) -> None:
        decision = datetime(2024, 2, 1, tzinfo=timezone.utc)
        complete = Mock(); complete.partitions = {}
        adapter = Mock(); frame = Mock(); adapter.frame.return_value = frame
        bounded = BoundedShadowPopulationSchedule(
            complete, self._event_locator_policy(), population_adapter=adapter,
        )
        bounded._candidate_pairs = Mock(return_value=iter([
            ("PF_XBTUSD", decision - timedelta(days=1)), ("PF_XBTUSD", decision),
        ]))
        generate.side_effect = [[], [{"decision_ts": decision}]]
        row = {
            "family_id": "A4_TSMOM_V7", "executable_attempt_id": "a4",
            "canonical_economic_address_sha256": "a" * 64, "config": {},
        }
        selected = list(bounded.iter_batch_locators(
            (row,), phase="inner_validation", outer_fold_id="2024Q1", inner_fold_id="M_202402",
        ))
        self.assertEqual([decision], [item.decision_ts for item in selected])
        self.assertEqual(2, adapter.frame.call_count)
        self.assertIs(frame, bounded.frame(selected[0]))

    @patch("tools.core_liquid_campaign.executor._generate_events", return_value=[])
    def test_event_locator_sampler_preserves_explicit_empty_when_no_real_event_exists(self, generate: Mock) -> None:
        decision = datetime(2024, 2, 1, tzinfo=timezone.utc)
        complete = Mock(); complete.partitions = {}; adapter = Mock(); adapter.frame.return_value = Mock()
        bounded = BoundedShadowPopulationSchedule(
            complete, self._event_locator_policy(), population_adapter=adapter,
        )
        bounded._candidate_pairs = Mock(return_value=iter([("PF_XBTUSD", decision)]))
        row = {
            "family_id": "A4_TSMOM_V7", "executable_attempt_id": "a4",
            "canonical_economic_address_sha256": "a" * 64, "config": {},
        }
        self.assertEqual([], list(bounded.iter_batch_locators(
            (row,), phase="inner_validation", outer_fold_id="2024Q1", inner_fold_id="M_202402",
        )))
        self.assertEqual("fail_insufficient_event_days", bounded.last_reconciliation["status"])
        self.assertEqual({"a4": 0}, bounded.last_reconciliation["eligible_event_days_by_attempt"])

    def test_bounded_schedule_populates_exact_attempt_identity_for_real_adapter(self) -> None:
        locator = FamilyDecisionLocator(
            "A4_TSMOM_V7", "outer_evaluation", "2024Q1", None,
            "PF_XBTUSD", datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
        complete = Mock(); complete.partitions = {"fixture": True}; complete.iter_locators.return_value = iter([locator])
        policy = {
            "schema": "stage24_shadow_population_slice_policy_v1",
            "selection": "first_authority_order_locator_on_each_distinct_UTC_day",
            "maximum_distinct_days_per_attempt_partition": 1,
            "economic_values_used_for_selection": False,
            "benchmark_frame_values_used": False,
            "full_launch_population_authority_preserved": True,
        }
        bounded = BoundedShadowPopulationSchedule(complete, policy)
        selected = list(bounded.iter_locators({
            "executable_attempt_id": "attempt", "canonical_economic_address_sha256": "a" * 64,
        }))
        self.assertEqual(("attempt", "a" * 64), (
            selected[0].executable_attempt_id, selected[0].canonical_economic_address_sha256,
        ))

    def test_bounded_kda_slice_preserves_encountered_unavailable_and_all_nine_folds(self) -> None:
        folds = ("2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4")
        source = Mock()
        source.stream.return_value = iter([
            SimpleNamespace(status="typed_unavailable", outer_fold_id="2023Q4"),
            *(SimpleNamespace(status="eligible", outer_fold_id=fold) for fold in folds),
        ])
        policy = {
            "schema": "stage24_shadow_kda02b_slice_policy_v1",
            "selection": "first_authority_order_eligible_record_per_outer_fold",
            "maximum_eligible_records_per_cell_fold": 1,
            "typed_unavailable_rows": "preserve_every_row_encountered_before_slice_completion",
            "economic_values_used_for_selection": False,
            "full_kda02b_population_authority_preserved": True,
        }
        bounded = BoundedShadowKDA02BAdapter(source, policy)
        rows = list(bounded.stream(cell_id="KDA02B_001"))
        self.assertEqual(10, len(rows))
        self.assertEqual(1, bounded.last_reconciliation["typed_unavailable_rows_preserved"])
        self.assertEqual(set(folds), set(bounded.last_reconciliation["eligible_records_by_outer_fold"]))

    @staticmethod
    def _source_packet(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        rows: list[dict[str, object]] = [
            {"campaign_id": CAMPAIGN_ID, "family_id": "A4_TSMOM_V7", "executable_attempt_id": "a4", "config": {}},
            {"campaign_id": CAMPAIGN_ID, "family_id": "A1_COMPRESSION_V2", "executable_attempt_id": "a1", "config": {}},
            {"campaign_id": CAMPAIGN_ID, "family_id": "A3_STARTER_RETEST_V3", "executable_attempt_id": "a3", "config": {}},
            {
                "campaign_id": CAMPAIGN_ID,
                "family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1",
                "executable_attempt_id": "a2",
                "config": {"parent_binding_mode": "source_attempt"},
                "resolved_parent_executable_attempt_id": "a1",
            },
        ]
        variants = [
            "identity_replay", "price_only", "OI_removed", "liquidation_removed",
            "funding_zero", "generic_structure_control", "stress_cost_32bps",
            "entry_delay_15m", "funding_start_alignment", "funding_end_alignment",
            "entry_delay_60m",
        ]
        rows.extend({
            "campaign_id": CAMPAIGN_ID,
            "family_id": "KDA02B_SURVIVOR_ADJUDICATION_V1",
            "executable_attempt_id": f"kda-{index}",
            "config": {"stage20_cell_id": "KDA02B_001", "adjudication_variant": variant},
        } for index, variant in enumerate(variants))
        controls = [{
            "control_attempt_id": f"control-{index}",
            "control_id": f"CONTROL_CLASS_{index:02d}",
            "family": "A4_TSMOM_V7",
            "fold": "2024Q1",
            "parent_slot": "A4_TSMOM_V7:2024Q1:beam:01",
        } for index in range(20)]
        atomic_write_jsonl(root / "FINAL_EXECUTION_REGISTRY.jsonl", rows)
        atomic_write_jsonl(root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", rows)
        atomic_write_jsonl(root / "FINAL_CONTROL_REGISTRY.jsonl", controls)
        atomic_write_jsonl(root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl", [{"a2_executable_attempt_id": "a2"}])
        return rows, controls

    def test_bounded_packet_copies_only_exact_frozen_rows_and_labels_probe_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); source = root / "source"; source.mkdir()
            rows, controls = self._source_packet(source)
            authority = root / "EXECUTION_INPUT_AUTHORITY.json"; atomic_write_json(authority, {"source_records": []})
            cache = root / "SEMANTIC_CACHE_MANIFEST.json"; atomic_write_json(cache, {"artifacts": [{} for _ in range(567)]})
            launch = root / "LAUNCH_POPULATION_AUTHORITY.json"; atomic_write_json(launch, {"status": "fixture"})
            kda = root / "KDA02B_LAZY_POPULATION_MANIFEST.json"; atomic_write_json(kda, {"status": "fixture"})
            packet = build_bounded_shadow_packet(
                source_packet_root=source, output_root=root / "packet",
                execution_input_authority_path=authority, cache_manifest_path=cache,
                launch_population_authority_path=launch, kda02b_population_manifest_path=kda,
                executable_attempt_ids=[str(row["executable_attempt_id"]) for row in rows],
                control_attempt_ids=[str(row["control_attempt_id"]) for row in controls],
            )
            copied = [json.loads(line) for line in (root / "packet/FINAL_EXECUTION_REGISTRY.jsonl").read_text().splitlines()]
            self.assertEqual(rows, copied)
            manifest = json.loads((root / "packet/SHADOW_CAMPAIGN_MANIFEST.json").read_text())
            self.assertEqual("benchmark_probe_only", manifest["cache_role"])
            self.assertFalse(manifest["cache_is_launch_input_authority"])
            self.assertEqual(20, len(json.loads((root / "packet/SHADOW_SUBSET_AUTHORITY.json").read_text())["control_classes"]))
            self.assertEqual(sha256_file(root / "packet/SHADOW_CAMPAIGN_MANIFEST.json"), packet["manifest"]["sha256"])

    def test_shadow_authorization_fails_when_frozen_source_registry_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); source = root / "source"; source.mkdir()
            rows, controls = self._source_packet(source)
            authority = root / "authority.json"; atomic_write_json(authority, {})
            cache = root / "cache.json"; atomic_write_json(cache, {"artifacts": [{} for _ in range(567)]})
            launch = root / "launch.json"; atomic_write_json(launch, {})
            kda = root / "kda.json"; atomic_write_json(kda, {})
            packet = build_bounded_shadow_packet(
                source_packet_root=source, output_root=root / "packet",
                execution_input_authority_path=authority, cache_manifest_path=cache,
                launch_population_authority_path=launch, kda02b_population_manifest_path=kda,
                executable_attempt_ids=[str(row["executable_attempt_id"]) for row in rows],
                control_attempt_ids=[str(row["control_attempt_id"]) for row in controls],
            )
            source.joinpath("FINAL_EXECUTION_REGISTRY.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ShadowCampaignPacketError):
                ShadowCampaignAuthorization({"shadow_campaign_packet": packet}, Path.cwd()).require()

    @patch("tools.core_liquid_campaign.shadow_service.reconcile_control_duplicates")
    @patch("tools.core_liquid_campaign.shadow_service.execute_control")
    def test_all_control_work_uses_population_adapter_and_never_probe_frames(
        self, execute_control: Mock, reconcile: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = {"executable_attempt_id": "a4", "family_id": "A4_TSMOM_V7", "config": {}}
            control = {
                "control_attempt_id": "control", "control_id": "A4_CONTEXT_REMOVED",
                "family": "A4_TSMOM_V7", "fold": "2024Q1",
                "parent_slot": "A4_TSMOM_V7:2024Q1:beam:01", "effective_seed": 1,
                "execution_status": "execute_once",
            }
            reconcile.return_value = [control]
            execute_control.return_value = {"status": "complete", "observations": [], "ledger": [], "aggregate": {}}
            campaign = ProductionShadowCampaignOrchestrator.__new__(ProductionShadowCampaignOrchestrator)
            campaign.population_schedule = Mock(); campaign.population_schedule.iter_locators.return_value = iter(["locator"])
            campaign.population_adapter = Mock(); campaign.population_adapter.frame.return_value = "actual-production-frame"
            campaign.run_root = Path(raw)
            campaign.payoff_provider = Mock(); campaign.payoff_provider.attestation.return_value = {"economic_outcomes_opened": False}
            campaign.cache_authority = Mock()
            jobs = campaign._control_jobs(
                [parent], [control], {"a4": parent}, {},
                [{"parent_slot": control["parent_slot"], "executable_attempt_id": "a4"}], {},
            )
            job_id, task = next(iter(jobs)); result = task()
            self.assertEqual("control:control", job_id)
            campaign.population_adapter.frame.assert_called_once_with("locator")
            campaign.cache_authority.load_frames.assert_not_called()
            self.assertEqual("LaunchPopulationSchedule->LazyProductionFamilyInputAdapter", result["input_path"])
            self.assertEqual(0, result["benchmark_probe_frames_used"])
            self.assertEqual(("actual-production-frame",), execute_control.call_args.args[2])

    @patch("tools.core_liquid_campaign.shadow_service.BoundedShadowKDA02BAdapter")
    @patch("tools.core_liquid_campaign.shadow_service.KDA02BLazyFamilyInputAdapter")
    @patch("tools.core_liquid_campaign.shadow_service.LazyProductionFamilyInputAdapter")
    @patch("tools.core_liquid_campaign.shadow_service.BoundedShadowPopulationSchedule")
    @patch("tools.core_liquid_campaign.shadow_service.LaunchPopulationSchedule")
    @patch("tools.core_liquid_campaign.shadow_service._verify_shadow_worker_evidence")
    @patch("tools.core_liquid_campaign.shadow_service.verify_terminal_inventory")
    @patch("tools.core_liquid_campaign.shadow_service.ProductionShadowCampaignOrchestrator")
    @patch("tools.core_liquid_campaign.shadow_service.CacheAuthority")
    @patch("tools.core_liquid_campaign.shadow_service.ShadowCampaignAuthorization")
    @patch("tools.core_liquid_campaign.shadow_service.ShadowAuthorization")
    def test_service_invokes_actual_campaign_orchestrator_with_only_shadow_provider(
        self, shadow_authorization: Mock, campaign_authorization: Mock, cache_authority: Mock,
        orchestrator: Mock, verify_terminal: Mock, verify_worker_evidence: Mock,
        launch_schedule: Mock, bounded_schedule: Mock, population_adapter: Mock,
        kda_population_adapter: Mock, bounded_kda_adapter: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); packet_root = root / "packet"; packet_root.mkdir()
            atomic_write_jsonl(packet_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", [])
            atomic_write_jsonl(packet_root / "FINAL_CONTROL_REGISTRY.jsonl", [])
            cache_path = root / "cache.json"; atomic_write_json(cache_path, {})
            reused_path = root / "reused.json"
            atomic_write_json(reused_path, {"prior_campaign_run_root": str(root / "prior")})
            spec = {
                "repository_root": str(Path.cwd()), "run_root": str(root / "run"),
                "service_identity": "fixture.service", "workers": 1, "heartbeat_seconds": 1,
                "identity_bindings": {"fixture": True}, "synthetic_provider_version": "fixture-shadow",
                "population_slice_policy": {
                    "schema": "stage24_shadow_event_locator_policy_v2",
                    "selection": "actual_production_enumerator_pre_entry_event_locators",
                    "target_eligible_event_days_per_attempt_partition": 2,
                    "maximum_candidate_locators_per_attempt_partition": 100,
                    "empty_attempt_partitions_preserved": True,
                    "real_post_entry_values_used": False,
                    "economic_values_used_for_selection": False,
                    "synthetic_payoff_generated_after_locator_freeze": True,
                    "full_launch_population_authority_preserved": True,
                },
                "reused_evidence_authority": {"path": str(reused_path)},
                "kda02b_slice_policy": {
                    "schema": "stage24_shadow_kda02b_slice_policy_v1",
                    "selection": "first_authority_order_eligible_record_per_outer_fold",
                    "maximum_eligible_records_per_cell_fold": 1,
                    "typed_unavailable_rows": "preserve_every_row_encountered_before_slice_completion",
                    "economic_values_used_for_selection": False,
                    "full_kda02b_population_authority_preserved": True,
                },
                "shadow_campaign_packet": {
                    "packet_root": str(packet_root),
                    "cache_manifest": {"path": str(cache_path)},
                    "execution_input_authority": {"path": str(cache_path)},
                },
            }
            shadow_authorization.return_value.require.return_value = spec
            launch = root / "launch.json"; atomic_write_json(launch, {})
            kda = root / "kda.json"; atomic_write_json(kda, {})
            campaign_authorization.return_value.require.return_value = {
                "campaign_id": CAMPAIGN_ID,
                "launch_population_authority": {"path": str(launch), "sha256": sha256_file(launch)},
                "kda02b_lazy_population_authority": {"path": str(kda), "sha256": sha256_file(kda)},
            }
            campaign = orchestrator.return_value
            campaign.run.side_effect = lambda: self._complete_campaign_fixture(
                root / "run/campaign", stage="event_locator_inner_development",
            )
            verify_terminal.return_value = {"status": "pass", "inventory_sha256": "a" * 64}
            verify_worker_evidence.return_value = {"economic_outcomes_opened": False, "materialized_shadow_ledger_rows": 1}
            transport = Mock(); transport.preflight.return_value = True; transport.heartbeat.return_value = True
            with patch("tools.run_stage22_core_liquid_campaign.TelegramTransport", return_value=transport):
                result = run_shadow_service(root / "spec.json")
            self.assertEqual("complete", result["status"])
            self.assertEqual("actual_production_stage_graph", result["campaign_orchestrator"])
            campaign.run.assert_called_once_with()
            kwargs = orchestrator.call_args.kwargs
            self.assertEqual(packet_root, kwargs["packet_root"])
            self.assertIs(kwargs["population_schedule"], bounded_schedule.return_value)
            self.assertIs(kwargs["population_adapter"], population_adapter.from_launch_population_authority.return_value)
            self.assertIs(kwargs["kda02b_population_adapter"], bounded_kda_adapter.return_value)
            self.assertEqual("fixture-shadow", kwargs["payoff_provider"].campaign_identity)
            self.assertEqual(0, kwargs["payoff_provider"].real_post_entry_rows_opened)
            self.assertNotIn("dispatch_registered_attempt", run_shadow_service.__code__.co_names)
            self.assertEqual("event_locator_inner_development", campaign.inner_development_stage_name)
            transport.launch.assert_called_once()
            transport.complete.assert_called_once()

    @patch("tools.core_liquid_campaign.shadow_service._write_shadow_bound_stop")
    @patch("tools.core_liquid_campaign.shadow_service.ProductionShadowCampaignOrchestrator")
    @patch("tools.core_liquid_campaign.shadow_service.BoundedShadowKDA02BAdapter")
    @patch("tools.core_liquid_campaign.shadow_service.KDA02BLazyFamilyInputAdapter")
    @patch("tools.core_liquid_campaign.shadow_service.LazyProductionFamilyInputAdapter")
    @patch("tools.core_liquid_campaign.shadow_service.BoundedShadowPopulationSchedule")
    @patch("tools.core_liquid_campaign.shadow_service.LaunchPopulationSchedule")
    @patch("tools.core_liquid_campaign.shadow_service.CacheAuthority")
    @patch("tools.core_liquid_campaign.shadow_service.ShadowCampaignAuthorization")
    @patch("tools.core_liquid_campaign.shadow_service.ShadowAuthorization")
    def test_service_delivers_bound_stop_notification(
        self, shadow_authorization: Mock, campaign_authorization: Mock, cache_authority: Mock,
        launch_schedule: Mock, bounded_schedule: Mock, population_adapter: Mock,
        kda_adapter: Mock, bounded_kda_adapter: Mock, orchestrator: Mock,
        write_bound_stop: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); packet_root = root / "packet"; packet_root.mkdir()
            atomic_write_jsonl(packet_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", [])
            atomic_write_jsonl(packet_root / "FINAL_CONTROL_REGISTRY.jsonl", [])
            cache_path = root / "cache.json"; atomic_write_json(cache_path, {})
            launch = root / "launch.json"; atomic_write_json(launch, {})
            kda = root / "kda.json"; atomic_write_json(kda, {})
            spec = {
                "repository_root": str(Path.cwd()), "run_root": str(root / "run"),
                "service_identity": "fixture.service", "workers": 1, "heartbeat_seconds": 1,
                "identity_bindings": {"fixture": True}, "synthetic_provider_version": "fixture-shadow",
                "population_slice_policy": {}, "kda02b_slice_policy": {},
                "shadow_campaign_packet": {
                    "packet_root": str(packet_root), "cache_manifest": {"path": str(cache_path)},
                    "execution_input_authority": {"path": str(cache_path)},
                },
            }
            shadow_authorization.return_value.require.return_value = spec
            campaign_authorization.return_value.require.return_value = {
                "campaign_id": CAMPAIGN_ID,
                "launch_population_authority": {"path": str(launch), "sha256": sha256_file(launch)},
                "kda02b_lazy_population_authority": {"path": str(kda), "sha256": sha256_file(kda)},
            }
            orchestrator.return_value.run.side_effect = CampaignContractError("fixture stage failure")
            write_bound_stop.return_value = {"status": "pass"}
            transport = Mock(); transport.preflight.return_value = True; transport.bound_stop.return_value = True
            with patch("tools.run_stage22_core_liquid_campaign.TelegramTransport", return_value=transport):
                result = run_shadow_service(root / "spec.json")
            self.assertEqual("global_resumable_bound_stop", result["status"])
            transport.bound_stop.assert_called_once_with({
                "service_identity": "fixture.service", "status": "global_resumable_bound_stop",
                "reason": "fixture stage failure", "resumable": True,
            })

    @staticmethod
    def _complete_campaign_fixture(root: Path, *, stage: str = "inner_development") -> dict[str, object]:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(root / "STAGE_JOB_RECONCILIATION.json", {"pass": True})
        atomic_write_json(root / stage / "SUPERVISOR_STATE.json", {
            "first_real_unit_reconciled": True, "health_release": True, "heartbeat_success_count": 1,
        })
        return {"status": "complete", "completed_stages": ["terminal_reconciliation"]}


if __name__ == "__main__":
    unittest.main()
