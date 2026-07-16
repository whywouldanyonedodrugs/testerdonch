import unittest

import pandas as pd

from tools import run_kraken_c2_shock_episode_budget_repair as c2


class C2ShockEpisodeBudgetRepairTests(unittest.TestCase):
    def records(self):
        return pd.DataFrame([
            {"production_event_id": "A", "Source publication timestamp": "2024-01-01", "Effective timestamp UTC": "2024-01-01", "Audited provisional event ID": "PA"},
            {"production_event_id": "B", "Source publication timestamp": "2024-01-10", "Effective timestamp UTC": "2024-01-10", "Audited provisional event ID": "PB"},
            {"production_event_id": "C", "Source publication timestamp": "2024-06-01", "Effective timestamp UTC": "2024-06-01", "Audited provisional event ID": "PC"},
        ])

    def exposures(self):
        return pd.DataFrame([
            {"parent_event_id": "A", "event_exposure_id": "A1", "catalyst_cluster_id": "P", "event_anchor_ts": "2024-01-01T00:00:00Z", "maximum_candidate_exit_ts": "2024-01-25T00:00:00Z", "audited_ticker": "BTC"},
            {"parent_event_id": "A", "event_exposure_id": "A2", "catalyst_cluster_id": "P", "event_anchor_ts": "2024-01-01T00:00:00Z", "maximum_candidate_exit_ts": "2024-01-25T00:00:00Z", "audited_ticker": "ETH"},
            {"parent_event_id": "B", "event_exposure_id": "B1", "catalyst_cluster_id": "P", "event_anchor_ts": "2024-01-10T00:00:00Z", "maximum_candidate_exit_ts": "2024-02-03T00:00:00Z", "audited_ticker": "BTC"},
            {"parent_event_id": "C", "event_exposure_id": "C1", "catalyst_cluster_id": "P", "event_anchor_ts": "2024-06-01T00:00:00Z", "maximum_candidate_exit_ts": "2024-06-25T00:00:00Z", "audited_ticker": "BTC"},
        ])

    def test_overlap_joins_and_nonoverlap_separates(self):
        assigned, _, _ = c2.episode_assignments(self.exposures(), self.records())
        ids = assigned.groupby("parent_event_id").shock_episode_id.first()
        self.assertEqual(ids.A, ids.B)
        self.assertNotEqual(ids.A, ids.C)

    def test_basket_exposures_share_one_episode(self):
        assigned, _, _ = c2.episode_assignments(self.exposures(), self.records())
        self.assertEqual(assigned[assigned.parent_event_id == "A"].shock_episode_id.nunique(), 1)

    def test_definition_budget_thresholds(self):
        self.assertEqual(len(c2.build_definitions(7)), 0)
        self.assertEqual(len(c2.build_definitions(8)), 8)
        self.assertEqual(len(c2.build_definitions(12)), 12)


if __name__ == "__main__":
    unittest.main()
