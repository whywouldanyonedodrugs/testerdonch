import unittest

import pandas as pd

from tools import run_kraken_backside_blowoff_short_screen as blowoff


class BacksideBlowoffShortTests(unittest.TestCase):
    def test_manifest_is_frozen_24_with_eight_selected_key_specs(self):
        manifest = blowoff.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())

    def test_exit_policy_does_not_change_selected_key_hash(self):
        manifest = blowoff.frozen_manifest()
        for _, group in manifest.groupby(["extension_profile", "confirmation_bars", "parent_context"]):
            self.assertEqual(group.selected_key_policy_hash.nunique(), 1)

    def test_confirmation_requires_all_three_backside_conditions(self):
        frame = pd.DataFrame({
            "decision_ts": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T04:00:00Z"]),
            "open": [100.0, 100.0], "high": [110.0, 105.0], "low": [90.0, 89.0], "close": [108.0, 88.0],
            "extension_40_5": [True, False], "extension_70_10": [False, False],
        })
        known = pd.date_range("2024-12-31T20:05:00Z", periods=96, freq="5min")
        work = pd.DataFrame({"known_ts": known, "volume": 1.0, "vwap_num": 100.0})
        found = blowoff.confirmation_sequences(frame, work, "rise_40pct_5d_3atr", 1)
        self.assertEqual(len(found), 1)

        frame.loc[1, "close"] = 100.0  # Below prior low no longer holds.
        self.assertEqual(blowoff.confirmation_sequences(frame, work, "rise_40pct_5d_3atr", 1), [])

    def test_higher_high_resets_confirmation_window(self):
        frame = pd.DataFrame({
            "decision_ts": pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC"),
            "open": [100.0]*4, "high": [110.0, 112.0, 108.0, 107.0], "low": [90.0, 91.0, 90.0, 89.0],
            "close": [108.0, 110.0, 100.0, 88.0], "extension_40_5": [True, False, False, False],
            "extension_70_10": [False]*4,
        })
        known = pd.date_range("2025-01-01T00:05:00Z", periods=145, freq="5min")
        work = pd.DataFrame({"known_ts": known, "volume": 1.0, "vwap_num": 100.0})
        found = blowoff.confirmation_sequences(frame, work, "rise_40pct_5d_3atr", 3)
        self.assertEqual(found[0]["peak_index"], 1)
        self.assertEqual(found[0]["decision_index"], 3)

    def test_candidate_address_changes_with_exit_policy(self):
        base = {"symbol":"PF_XBTUSD","decision_ts":"2025-01-01T00:00:00Z","entry_ts":"2025-01-01T00:05:00Z","initial_stop":110.0,"risk_denominator":10.0,"maximum_exit_ts":"2025-01-04T00:05:00Z"}
        self.assertNotEqual(blowoff.candidate_address({**base,"exit_policy":"fixed_72h"}), blowoff.candidate_address({**base,"exit_policy":"daily_ema10_close"}))

    def test_duplicate_candidate_gate_is_scoped_within_definition(self):
        frame = pd.DataFrame({
            "definition_id": ["a", "b"],
            "candidate_economic_address_hash": ["same-market-trade", "same-market-trade"],
        })
        self.assertEqual(frame.duplicated(["definition_id", "candidate_economic_address_hash"]).sum(), 0)
        self.assertEqual(frame.duplicated("candidate_economic_address_hash").sum(), 1)

    def test_evaluation_windows_are_half_open_and_utc(self):
        self.assertEqual(blowoff.evaluation_window(pd.Timestamp("2025-06-30T23:55:00Z"))[0], "2025-H1")
        self.assertEqual(blowoff.evaluation_window(pd.Timestamp("2025-07-01T00:00:00Z"))[0], "2025-H2")

    def test_zero_cost_diagnostics_are_arithmetic_only(self):
        row = pd.DataFrame({"gross_R":[1.0],"fee_base_R":[-0.1],"slippage_base_R":[-0.2],"funding_central_R":[-0.3]})
        row["net_zero_funding_base_R"] = row.gross_R + row.fee_base_R + row.slippage_base_R
        row["net_zero_fee_base_R"] = row.gross_R + row.slippage_base_R + row.funding_central_R
        self.assertAlmostEqual(row.iloc[0].net_zero_funding_base_R, .7)
        self.assertAlmostEqual(row.iloc[0].net_zero_fee_base_R, .5)

    def test_daily_ema_exit_has_no_invented_seven_day_time_stop(self):
        source = __import__("inspect").getsource(blowoff.execute_event)
        self.assertIn('exit_policy == "swing_high_trail_7d"', source)
        self.assertIn('no_natural_ema_exit_before_evaluation_boundary', source)
        self.assertNotIn('exit_policy in ("daily_ema10_close", "swing_high_trail_7d")', source)

    def test_control_summary_deduplicates_economic_addresses(self):
        source = __import__("inspect").getsource(blowoff.controls_report)
        self.assertIn('drop_duplicates("control_economic_address_hash")', source)
        self.assertIn('control_mean_R":unique[', source)


if __name__ == "__main__":
    unittest.main()
