from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from tools.qlmg_kda01_timestamp_repair import (
    expected_entry_at_or_after,
    locate_repaired_execution,
    repaired_execution_records,
)
from tools.run_kda01_stage8c1_forensic_timing import frozen_repair_contract


class KDA01Stage8C1TimestampRepairTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2024-01-01T00:00:00Z", periods=100, freq="5min")

    def test_aligned_decision_enters_at_decision_not_five_minutes_later(self):
        result = locate_repaired_execution("2024-01-01T00:05:00Z", 1, self.times)
        self.assertEqual(result.expected_entry_ts, pd.Timestamp("2024-01-01T00:05:00Z"))
        self.assertEqual(result.entry_ts, pd.Timestamp("2024-01-01T00:05:00Z"))
        self.assertEqual(result.exit_ts, pd.Timestamp("2024-01-01T01:05:00Z"))

    def test_unaligned_decision_uses_first_grid_at_or_after(self):
        self.assertEqual(expected_entry_at_or_after("2024-01-01T00:06:00Z"), pd.Timestamp("2024-01-01T00:10:00Z"))
        self.assertEqual(locate_repaired_execution("2024-01-01T00:06:00Z", 1, self.times).entry_ts, pd.Timestamp("2024-01-01T00:10:00Z"))

    def test_missing_expected_bar_uses_later_bar_with_declared_delay(self):
        sparse = self.times.delete(1)
        result = locate_repaired_execution("2024-01-01T00:05:00Z", 1, sparse)
        self.assertEqual(result.entry_ts, pd.Timestamp("2024-01-01T00:10:00Z"))
        self.assertEqual(result.entry_delay_minutes, 5)

    def test_definition_local_actual_exit_overlap(self):
        events = pd.DataFrame([
            {"event_id": "a", "economic_address": "a", "branch_id": "b", "symbol": "PF_X", "decision_ts": "2024-01-01T00:05:00Z"},
            {"event_id": "b", "economic_address": "b", "branch_id": "b", "symbol": "PF_X", "decision_ts": "2024-01-01T00:30:00Z"},
            {"event_id": "c", "economic_address": "c", "branch_id": "b", "symbol": "PF_X", "decision_ts": "2024-01-01T01:05:00Z"},
        ])
        definitions = pd.DataFrame([{"definition_id": "d", "definition_contract_hash": "h", "branch_id": "b", "timeout_hours": 1}])
        out = repaired_execution_records(events, definitions, {"PF_X": self.times})
        self.assertEqual(out.accepted.tolist(), [True, False, True])
        self.assertEqual(out.status.tolist(), ["eligible", "actual_position_overlap", "eligible"])

    def test_contract_changes_only_execution_version(self):
        contract = frozen_repair_contract()
        self.assertEqual(contract["definitions"], 16)
        self.assertEqual(contract["costs_bps"], {"base": 14, "stress": 32})
        self.assertFalse(contract["controls_executed"])
        self.assertTrue(contract["repair_contract_hash"])

    def test_timestamp_decision_precedes_original_outcome_load_in_audit_phase(self):
        source = Path("tools/run_kda01_stage8c1_forensic_timing.py").read_text()
        body = source[source.index("def audit_phase"):source.index("def execute_phase")]
        self.assertLess(body.index("KDA01_TIMESTAMP_REPAIR_DECISION.md"), body.index("independently_recompute_original"))

    def test_execute_phase_requires_matching_review(self):
        source = Path("tools/run_kda01_stage8c1_forensic_timing.py").read_text()
        self.assertIn('review.get("repair_contract_hash") != contract["repair_contract_hash"]', source)
        self.assertIn("contract != frozen_repair_contract()", source)
        self.assertIn("frozen source mismatch before repaired outcomes", source)

    def test_direct_script_bootstraps_repository_import_path(self):
        source = Path("tools/run_kda01_stage8c1_forensic_timing.py").read_text()
        self.assertLess(source.index("sys.path.insert"), source.index("from tools import"))


if __name__ == "__main__":
    unittest.main()
