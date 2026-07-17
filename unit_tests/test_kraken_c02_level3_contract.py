import unittest

import numpy as np
import pandas as pd

from tools import freeze_kraken_c02_level3_contract as c


class C02Level3ContractTests(unittest.TestCase):
    def synthetic_events(self):
        rows = []
        for i in range(489):
            rows.append({"event_id": f"e{i}", "economic_address": f"a{i}", "PF_symbol": f"PF_{i % 10}", "direction_label": "positive",
                         "leadership_state": "resolved_spot_led", "leadership_30m": "resolved_spot_led" if i < 425 else "coincident_or_unresolved",
                         "leadership_lookback": "15m_primary", "decision_ts": pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(hours=i)})
        return pd.DataFrame(rows)

    def test_exact_primary_and_robustness_membership(self):
        selected, _, _ = c.select_event_sets(self.synthetic_events())
        self.assertEqual(len(selected), 489)
        self.assertEqual(int(selected.in_30m_agreement_subset.sum()), 425)

    def test_exactly_four_definitions(self):
        register = c.definition_register("p", "r")
        self.assertEqual(len(register), 4)
        self.assertEqual(int(register.can_earn_level3_permission.sum()), 2)

    def test_robustness_cannot_rescue_primary(self):
        self.assertFalse(c.primary_permission({"p1": {"all_pass": False}, "p2": {"all_pass": False}}, {"r": {"all_pass": True}}))

    def test_next_open_entry_and_timeout_exit(self):
        bars = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
        entry, exit_ts = c.executable_interval(bars, bars[2], 1)
        self.assertEqual(entry, bars[3])
        self.assertEqual(exit_ts, bars[15])

    def test_missing_or_protected_exit_fails_closed(self):
        bars = pd.date_range("2025-12-31 23:50", periods=3, freq="5min", tz="UTC")
        with self.assertRaises(ValueError):
            c.executable_interval(bars, bars[0], 1)

    def test_actual_exit_nonoverlap(self):
        trades = pd.DataFrame([
            {"event_id": "a", "PF_symbol": "PF_X", "entry_ts": pd.Timestamp("2024-01-01 00:05Z"), "exit_ts": pd.Timestamp("2024-01-01 01:05Z")},
            {"event_id": "b", "PF_symbol": "PF_X", "entry_ts": pd.Timestamp("2024-01-01 00:30Z"), "exit_ts": pd.Timestamp("2024-01-01 01:30Z")},
            {"event_id": "c", "PF_symbol": "PF_X", "entry_ts": pd.Timestamp("2024-01-01 01:05Z"), "exit_ts": pd.Timestamp("2024-01-01 02:05Z")},
        ])
        accepted, skipped = c.definition_local_nonoverlap(trades)
        self.assertEqual(accepted.event_id.tolist(), ["a", "c"])
        self.assertEqual(skipped.event_id.tolist(), ["b"])

    def test_fixed_notional_cost_arithmetic(self):
        result = c.fixed_notional_bps(100, 101)
        self.assertAlmostEqual(result["gross_bps"], 100)
        self.assertAlmostEqual(result["base_net_bps_ex_funding"], 86)
        self.assertAlmostEqual(result["stress_net_bps_ex_funding"], 68)

    def test_funding_partitions(self):
        self.assertEqual(c.funding_partition(0, 0), "zero_boundary")
        self.assertEqual(c.funding_partition(2, 0), "fully_exact_funded")
        self.assertEqual(c.funding_partition(0, 2), "fully_imputed")
        self.assertEqual(c.funding_partition(1, 1), "mixed")

    def test_bootstrap_is_deterministic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0]); episodes = np.array(["a", "a", "b", "c"])
        self.assertEqual(c.episode_bootstrap_ci(values, episodes, resamples=100), c.episode_bootstrap_ci(values, episodes, resamples=100))

    def test_nonpositive_concentration_denominator_fails(self):
        trades = pd.DataFrame({"base_net_bps_ex_funding": [-1.0], "PF_symbol": ["x"], "canonical_episode_id": ["e"], "year": [2024]})
        with self.assertRaises(ValueError):
            c.concentration_metrics(trades)

    def test_control_calipers_never_widen(self):
        treated = pd.Series({"PF_symbol": "PF_X", "year": 2024, "spot_z_15m": 3.0, "perp_z_15m": 2.0,
                             "prior_day_pf_liquidity_rank": 10, "lagged_pf_vol_24h": .10, "canonical_episode_id": "t",
                             "decision_ts": pd.Timestamp("2024-06-01", tz="UTC")})
        pool = pd.DataFrame([{"PF_symbol": "PF_X", "year": 2024, "spot_z_15m": 3.51, "perp_z_15m": 2.0,
                              "prior_day_pf_liquidity_rank": 10, "lagged_pf_vol_24h": .10, "canonical_episode_id": "c",
                              "decision_ts": pd.Timestamp("2024-05-01", tz="UTC")}])
        self.assertIsNone(c.match_leadership_control(treated, pool))

    def test_outcome_columns_prohibited(self):
        with self.assertRaises(ValueError):
            c.assert_no_outcome_columns(["event_id", "net_bps"])


if __name__ == "__main__":
    unittest.main()
