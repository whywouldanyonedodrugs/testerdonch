from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from tools.core_liquid_campaign.campaign import CampaignOrchestrator
from tools.core_liquid_campaign.kda02b_lazy_family_input import KDA02BLazyFamilyInputRecord
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
        frame = kda_frame(config)
        eligible = KDA02BLazyFamilyInputRecord(
            "eligible", "event-1", str(config["stage20_cell_id"]), "Q_2024Q1",
            "2024Q1", frame.symbol, frame.decision_ts, "1" * 64, None, frame,
        )
        unavailable = KDA02BLazyFamilyInputRecord(
            "typed_unavailable", "event-2", str(config["stage20_cell_id"]), "Q_2024Q1",
            "2024Q1", "PF_UNAVAILABLE", datetime(2024, 1, 2, tzinfo=UTC),
            "2" * 64, "stage14_kda02b_final_eligible_false", None,
        )
        adapter = _KDAAdapter((eligible, unavailable))
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
        self.assertEqual(1, result["population_eligible_records"])
        self.assertEqual(1, result["typed_unavailable_observation_count"])
        self.assertEqual("stage14_kda02b_final_eligible_false", result["typed_unavailable_observations"][0]["reason"])
        self.assertEqual([(config["stage20_cell_id"], None)], adapter.requested)
        self.assertFalse(result["shadow_attestation"]["economic_outcomes_opened"])


if __name__ == "__main__":
    unittest.main()
