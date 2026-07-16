import unittest

import pandas as pd

from tools import run_kraken_lfbs_021_canonical_episode_adjudication as adjudication


class Lfbs021CanonicalEpisodeAdjudicationTests(unittest.TestCase):
    def test_candidate_economic_address_is_deterministic_and_exit_sensitive(self):
        row = {"symbol":"PF_XBTUSD","decision_ts":"2023-01-01T00:00:00Z","entry_ts":"2023-01-01T00:05:00Z","initial_stop":101.0,"risk_denominator":1.0,"exit_policy":"fixed_72h_comparator","maximum_exit_ts":"2023-01-04T00:05:00Z"}
        self.assertEqual(adjudication.candidate_address(row), adjudication.candidate_address(dict(reversed(list(row.items())))))
        changed = dict(row); changed["maximum_exit_ts"] = "2023-01-03T00:05:00Z"
        self.assertNotEqual(adjudication.candidate_address(row), adjudication.candidate_address(changed))

    def test_period_labels_use_explicit_utc_boundaries(self):
        self.assertEqual(adjudication.period_label(pd.Timestamp("2023-12-31T23:59:00Z")), "2023")
        self.assertEqual(adjudication.period_label(pd.Timestamp("2024-01-01T00:00:00Z")), "2024")
        self.assertEqual(adjudication.period_label(pd.Timestamp("2025-06-30T23:59:00Z")), "2025-H1")
        self.assertEqual(adjudication.period_label(pd.Timestamp("2025-07-01T00:00:00Z")), "2025-H2")

    def test_active_month_count_uses_timezone_aware_series_accessor(self):
        values = pd.Series(pd.to_datetime(["2023-01-01T00:00:00Z", "2023-01-31T12:00:00Z", "2023-02-01T00:00:00Z"], utc=True))
        self.assertEqual(adjudication.active_month_count(values), 2)


if __name__ == "__main__":
    unittest.main()
