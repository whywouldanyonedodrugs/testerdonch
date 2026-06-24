from __future__ import annotations

import unittest

import pandas as pd

from tools.perp_state_v2_diagnostics import EventClusterSpec, assign_event_cluster_ids, concentration_summary, leave_one_group_out_summary


class PerpStateV2DiagnosticsTests(unittest.TestCase):
    def test_event_cluster_ids_are_stable_and_overlap_aware(self) -> None:
        entries = pd.DataFrame(
            {
                "entry_id": ["a", "b", "c"],
                "symbol": ["AAA", "AAA", "AAA"],
                "family": ["fam", "fam", "fam"],
                "decision_ts": pd.to_datetime(["2025-01-01 10:00Z", "2025-01-01 11:00Z", "2025-01-03 10:00Z"], utc=True),
            }
        )
        out1 = assign_event_cluster_ids(entries, EventClusterSpec(trigger_hours_by_family={"fam": 4}))
        out2 = assign_event_cluster_ids(entries, EventClusterSpec(trigger_hours_by_family={"fam": 4}))
        self.assertEqual(out1.loc[0, "event_cluster_id"], out1.loc[1, "event_cluster_id"])
        self.assertNotEqual(out1.loc[1, "event_cluster_id"], out1.loc[2, "event_cluster_id"])
        self.assertEqual(out1["event_cluster_id"].tolist(), out2["event_cluster_id"].tolist())

    def test_concentration_and_leave_one_group(self) -> None:
        ledger = pd.DataFrame(
            {
                "entry_id": ["a", "b", "c", "d"],
                "symbol": ["A", "A", "B", "C"],
                "family": ["fam", "fam", "fam", "fam"],
                "decision_ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-02-01", "2025-02-02"], utc=True),
                "event_cluster_id": ["x", "x", "y", "z"],
                "year_month": ["2025-01", "2025-01", "2025-02", "2025-02"],
            }
        )
        path = pd.DataFrame({"entry_id": ["a", "b", "c", "d"], "fwd_ret_close_72h": [0.1, -0.1, 0.2, 0.0]})
        c = concentration_summary(ledger, path)
        self.assertIn("event_cluster", set(c["dimension"]))
        loo = leave_one_group_out_summary(ledger, path, group_col="year_month")
        self.assertEqual(len(loo), 2)


if __name__ == "__main__":
    unittest.main()
