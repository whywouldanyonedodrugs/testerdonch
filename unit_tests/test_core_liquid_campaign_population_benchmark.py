from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.core_liquid_campaign.population_benchmark import benchmark_strata


class PopulationBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path("results/rebaseline/stage23_stage22_v04_remediation_20260721_v07/FINAL_EXECUTION_REGISTRY.jsonl")
        cls.execution = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def test_configuration_only_strata_are_deterministic_and_complete(self) -> None:
        first = benchmark_strata(self.execution)
        second = benchmark_strata(tuple(reversed(self.execution)))
        first_ids = {family: [row["executable_attempt_id"] for row in rows] for family, rows in first.items()}
        second_ids = {family: [row["executable_attempt_id"] for row in rows] for family, rows in second.items()}
        self.assertEqual(first_ids, second_ids)
        self.assertTrue(all(len(rows) == 10 for rows in first.values()))
        for family in ("A4_TSMOM_V7", "A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"):
            self.assertEqual(1, len({
                (row["config"]["PIT_liquidity_top_n"], row["config"].get("rebalance", "5m"))
                for row in first[family]
            }))
        self.assertEqual(1, len({row["resolved_parent_executable_attempt_id"] for row in first["A2_PRIOR_HIGH_RS_CONTEXT_V1"]}))

    def test_exit_and_overlay_classes_are_covered(self) -> None:
        rows = benchmark_strata(self.execution)
        self.assertGreaterEqual(len({row["config"]["exit"] for row in rows["A4_TSMOM_V7"]}), 8)
        self.assertGreaterEqual(len({row["config"]["exit"] for row in rows["A1_COMPRESSION_V2"]}), 9)
        self.assertGreaterEqual(len({row["config"]["exit"] for row in rows["A3_STARTER_RETEST_V3"]}), 9)
        self.assertEqual(
            {"permission", "linear_size_0_to_1", "tercile_size", "parent_only"},
            {row["config"]["overlay_action"] for row in rows["A2_PRIOR_HIGH_RS_CONTEXT_V1"]},
        )


if __name__ == "__main__":
    unittest.main()
