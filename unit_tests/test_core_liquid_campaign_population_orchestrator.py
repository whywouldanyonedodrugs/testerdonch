from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from tools.core_liquid_campaign.campaign import CampaignContractError, CampaignOrchestrator
from tools.core_liquid_campaign.canonical import atomic_write_json, sha256_file
from tools.core_liquid_campaign.kda02b_lazy_family_input import KDA02BLazyFamilyInputRecord
from tools.core_liquid_campaign.kda02b_denominator import STAGE20_ELIGIBLE_SYMBOLS
from tools.core_liquid_campaign.schema import CAMPAIGN_ID, baseline_config, economic_address, normalize_config
from tools.core_liquid_campaign.shadow_payoff import ShadowPayoffProvider
from tools.core_liquid_campaign.synthetic import a4_frame, kda_frame
from tools.core_liquid_campaign.lazy_production_inputs import FamilyDecisionLocator


UTC = timezone.utc


class _KDAAdapter:
    def __init__(self, records):
        self.records = records
        self.requested = []

    def stream(self, *, cell_id=None, outer_fold_id=None):
        self.requested.append((cell_id, outer_fold_id))
        yield from self.records


class _Schedule:
    def __init__(self, locator):
        self.locator = locator

    def iter_locators(self, *_args, **_kwargs):
        yield self.locator


class _FrameAdapter:
    def __init__(self, frame):
        self.value = frame; self.calls = 0

    def frame(self, _locator):
        self.calls += 1
        return self.value


class PopulationOrchestratorTests(unittest.TestCase):
    def test_completed_stage_resume_reuses_only_exact_reconciled_markers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); stage = root / "inner_development"
            artifact = stage / "artifacts/result.json"
            atomic_write_json(artifact, {"job_id": "legacy-inner", "result": {
                "registered_job_id": "legacy-inner", "registered_attempt_id": "attempt",
                "status": "complete", "aggregate": {},
            }})
            marker = {
                "job_id": "legacy-inner", "artifact": "artifacts/result.json",
                "artifact_sha256": sha256_file(artifact), "status": "complete",
                "reconciled_real_registered_unit": True,
            }
            atomic_write_json(stage / "markers/marker.json", marker)
            atomic_write_json(stage / "SUPERVISOR_STATE.json", {
                "status": "complete", "all_workers_stopped": True, "worker_pids": [],
                "in_flight_count": 0, "failed": {}, "completed_count": 1,
                "completed": {"legacy-inner": marker},
            })
            orchestrator = CampaignOrchestrator.__new__(CampaignOrchestrator); orchestrator.run_root = root
            self.assertEqual({"legacy-inner"}, orchestrator._require_completed_stage("inner_development"))
            artifact.write_text(json.dumps({"tampered": True}), encoding="utf-8")
            with self.assertRaisesRegex(CampaignContractError, "artifact hash mismatch"):
                orchestrator._require_completed_stage("inner_development")

    def test_direct_population_batch_constructs_one_frame_for_same_route(self) -> None:
        first_config = normalize_config("A4_TSMOM_V7", baseline_config("A4_TSMOM_V7"))
        second_config = normalize_config("A4_TSMOM_V7", {**first_config, "exit": "time_3d"})
        rows = []
        for index, config in enumerate((first_config, second_config)):
            rows.append({
                "campaign_id": CAMPAIGN_ID, "family_id": "A4_TSMOM_V7", "config": config,
                "execution_disposition": "execute_once", "executable_attempt_id": f"a4-{index}",
                "canonical_economic_address_sha256": economic_address("A4_TSMOM_V7", config)[1],
                "duplicate_of_executable_attempt_id": None,
            })
        frame = a4_frame(first_config)
        locator = FamilyDecisionLocator(
            "A4_TSMOM_V7", "outer_evaluation", "2024Q1", None,
            "PF_XBTUSD", frame.decision_ts, rows[0]["executable_attempt_id"], rows[0]["canonical_economic_address_sha256"],
        )
        adapter = _FrameAdapter(frame)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); cache = root / "cache.json"; cache.write_text("{}\n"); approval = root / "approval.json"; approval.write_text("{}\n")
            orchestrator = CampaignOrchestrator.__new__(CampaignOrchestrator)
            orchestrator.population_schedule = _Schedule(locator); orchestrator.population_adapter = adapter
            orchestrator.cache_authority = SimpleNamespace(manifest_path=cache)
            orchestrator.authorization = SimpleNamespace(external_approval_path=approval)
            orchestrator.payoff_provider = ShadowPayoffProvider("direct-population-batch-test")
            task = orchestrator._direct_population_batch_job(
                tuple(rows), {row["executable_attempt_id"]: row for row in rows}, "batch-job",
                phase="outer_evaluation", outer_fold_id="2024Q1", inner_fold_id=None,
            )
            result = task()
        self.assertEqual(1, adapter.calls)
        self.assertEqual(2, result["batch_size"])
        self.assertEqual({"a4-0", "a4-1"}, {row["registered_attempt_id"] for row in result["batch_results"]})

    def test_kda_population_job_dispatches_eligible_and_preserves_local_unavailable(self) -> None:
        config = normalize_config("KDA02B_SURVIVOR_ADJUDICATION_V1", baseline_config("KDA02B_SURVIVOR_ADJUDICATION_V1"))
        address = economic_address("KDA02B_SURVIVOR_ADJUDICATION_V1", config)[1]
        row = {
            "campaign_id": CAMPAIGN_ID,
            "family_id": "KDA02B_SURVIVOR_ADJUDICATION_V1",
            "config": config,
            "execution_disposition": "execute_once",
            "executable_attempt_id": "kda-fixture",
            "canonical_economic_address_sha256": address,
            "duplicate_of_executable_attempt_id": None,
        }
        intervals = (
            ("2023Q4", "2023-10-01", "2024-01-01"),
            ("2024Q1", "2024-01-01", "2024-04-01"),
            ("2024Q2", "2024-04-01", "2024-07-01"),
            ("2024Q3", "2024-07-01", "2024-10-01"),
            ("2024Q4", "2024-10-01", "2025-01-01"),
            ("2025Q1", "2025-01-01", "2025-04-01"),
            ("2025Q2", "2025-04-01", "2025-07-01"),
            ("2025Q3", "2025-07-01", "2025-10-01"),
            ("2025Q4", "2025-10-01", "2026-01-01"),
        )
        eligible = []
        for index, (fold, raw_start, raw_end) in enumerate(intervals):
            start = datetime.fromisoformat(raw_start).replace(tzinfo=UTC)
            end = datetime.fromisoformat(raw_end).replace(tzinfo=UTC)
            frame = kda_frame(config, anchor=start + timedelta(days=10))
            seconds = (end - start).total_seconds()
            frame = replace(frame, fold_id=f"Q_{fold}", threshold_populations={}, metadata={
                **frame.metadata,
                "campaign_partition": {
                    "phase": "kda02b_adjudication", "outer_fold_id": fold,
                    "inner_fold_id": None, "evaluation_start": start,
                    "evaluation_end_exclusive": end,
                },
                "evaluation_start": start, "evaluation_end_exclusive": end,
                "eligible_days": int(seconds // 86400),
                "eligible_symbol_seconds": seconds * STAGE20_ELIGIBLE_SYMBOLS,
            })
            eligible.append(KDA02BLazyFamilyInputRecord(
                "eligible", f"event-{index}", str(config["stage20_cell_id"]), f"Q_{fold}",
                fold, frame.symbol, frame.decision_ts, f"{index + 1:x}" * 64, None, frame,
            ))
        unavailable = KDA02BLazyFamilyInputRecord(
            "typed_unavailable", "event-2", str(config["stage20_cell_id"]), "Q_2024Q1",
            "2024Q1", "PF_UNAVAILABLE", datetime(2024, 1, 2, tzinfo=UTC),
            "2" * 64, "stage14_kda02b_final_eligible_false", None,
        )
        adapter = _KDAAdapter((*eligible, unavailable))
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache_manifest = root / "cache.json"; cache_manifest.write_text("{}\n", encoding="utf-8")
            approval = root / "approval.json"; approval.write_text("{}\n", encoding="utf-8")
            orchestrator = CampaignOrchestrator.__new__(CampaignOrchestrator)
            orchestrator.kda02b_population_adapter = adapter
            orchestrator.cache_authority = SimpleNamespace(manifest_path=cache_manifest)
            orchestrator.authorization = SimpleNamespace(external_approval_path=approval)
            orchestrator.payoff_provider = ShadowPayoffProvider("population-orchestrator-test")
            jobs = list(orchestrator._kda_jobs((row,), {"kda-fixture": row}, {}))
            self.assertEqual(1, len(jobs))
            batch = jobs[0][1]()
            self.assertEqual("kda02b-batch:" + str(config["stage20_cell_id"]), batch["registered_job_id"])
            self.assertEqual(1, batch["batch_size"])
            result = batch["batch_results"][0]
        self.assertEqual("complete", result["status"])
        self.assertEqual(9, result["population_eligible_records"])
        self.assertEqual(1, result["typed_unavailable_observation_count"])
        self.assertEqual("stage14_kda02b_final_eligible_false", result["typed_unavailable_observations"][0]["reason"])
        self.assertEqual([(config["stage20_cell_id"], None)], adapter.requested)
        self.assertFalse(result["shadow_attestation"]["economic_outcomes_opened"])


if __name__ == "__main__":
    unittest.main()
