from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.core_liquid_campaign.canonical import atomic_write_json, canonical_hash, sha256_file
from tools.core_liquid_campaign.population_execution import LaunchPopulationSchedule, PopulationExecutionError


class PopulationExecutionTests(unittest.TestCase):
    def _authority(self, root: Path) -> Path:
        pit = root / "pit.jsonl"
        rows = [
            {"day_open_ms": 1_704_067_200_000, "symbol": "PF_XBTUSD", "average_liquidity_rank": 1.0, "decision_count_5m": 288, "top_10": True, "top_20": True, "top_40": True},
            {"day_open_ms": 1_704_067_200_000, "symbol": "PF_ETHUSD", "average_liquidity_rank": 2.0, "decision_count_5m": 288, "top_10": True, "top_20": True, "top_40": True},
            {"day_open_ms": 1_704_153_600_000, "symbol": "PF_XBTUSD", "average_liquidity_rank": 1.0, "decision_count_5m": 1, "top_10": True, "top_20": True, "top_40": True},
        ]
        pit.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        partitions = [{"phase": "outer_evaluation", "outer_fold_id": "2024Q1", "inner_fold_id": None, "evaluation_start_ms": 1_704_067_200_000, "evaluation_end_exclusive_ms": 1_704_240_000_000}]
        partitions.extend({"phase": "inner_validation", "outer_fold_id": f"O{index:03d}", "inner_fold_id": f"I{index:03d}", "evaluation_start_ms": 1_704_067_200_000, "evaluation_end_exclusive_ms": 1_704_240_000_000} for index in range(131))
        payload = {
            "schema": "stage24_launch_population_authority_v1", "status": "bound_outcome_free",
            "population_census": {"partitions": partitions},
            "pit_membership": {"path": str(pit), "bytes": pit.stat().st_size, "sha256": sha256_file(pit)},
        }
        path = root / "authority.json"; atomic_write_json(path, payload); return path

    def test_schedule_counts_every_pit_decision_without_eager_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = self._authority(Path(raw))
            with patch("tools.core_liquid_campaign.population_execution.validate_launch_population_authority"):
                schedule = LaunchPopulationSchedule(path, sha256_file(path))
            a1 = {"family_id": "A1_COMPRESSION_V2", "config": {"PIT_liquidity_top_n": 10}, "executable_attempt_id": "a1", "canonical_economic_address_sha256": "1" * 64}
            a4 = {"family_id": "A4_TSMOM_V7", "config": {"PIT_liquidity_top_n": 10, "rebalance": "8h"}, "executable_attempt_id": "a4", "canonical_economic_address_sha256": "2" * 64}
            self.assertEqual(577, schedule.count(a1, phase="outer_evaluation", outer_fold_id="2024Q1", inner_fold_id=None))
            self.assertEqual(7, schedule.count(a4, phase="outer_evaluation", outer_fold_id="2024Q1", inner_fold_id=None))
            locators = list(schedule.iter_locators(a4, phase="outer_evaluation", outer_fold_id="2024Q1", inner_fold_id=None))
            self.assertEqual(7, len(locators))
            self.assertEqual(("PF_XBTUSD", 0), (locators[0].symbol, locators[0].decision_ts.hour))
            self.assertEqual("a4", locators[0].executable_attempt_id)
            self.assertEqual("2" * 64, locators[0].canonical_economic_address_sha256)

    def test_a2_requires_exact_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = self._authority(Path(raw))
            with patch("tools.core_liquid_campaign.population_execution.validate_launch_population_authority"):
                schedule = LaunchPopulationSchedule(path, sha256_file(path))
            a2 = {"family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1", "config": {"parent_family": "A4_TSMOM_V7"}}
            with self.assertRaisesRegex(PopulationExecutionError, "exact registered parent"):
                list(schedule.iter_locators(a2, phase="outer_evaluation", outer_fold_id="2024Q1", inner_fold_id=None))


if __name__ == "__main__":
    unittest.main()
