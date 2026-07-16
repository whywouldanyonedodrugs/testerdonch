import inspect
import unittest

import pandas as pd

from tools import run_kraken_riskoff_failed_bounce_short_screen as riskoff


class RiskOffFailedBounceShortTests(unittest.TestCase):
    def test_manifest_is_frozen_24_with_eight_selected_key_policies(self):
        manifest = riskoff.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())

    def test_exit_policy_does_not_change_selected_key_hash(self):
        manifest = riskoff.frozen_manifest()
        grouped = manifest.groupby(["rally_profile", "confirmation_bars", "parent_policy"])
        self.assertTrue(all(group.selected_key_policy_hash.nunique() == 1 for _, group in grouped))

    def test_parent_policies_are_distinct_and_fail_closed(self):
        self.assertTrue(riskoff.parent_allowed("strict_both_down_stress", "both_down"))
        self.assertFalse(riskoff.parent_allowed("strict_both_down_stress", "mixed_at_least_one_down"))
        self.assertTrue(riskoff.parent_allowed("broader_fragile_countertrend_stress", "mixed_at_least_one_down"))
        self.assertFalse(riskoff.parent_allowed("broader_fragile_countertrend_stress", "both_up"))
        self.assertFalse(riskoff.parent_allowed("broader_fragile_countertrend_stress", "unknown"))

    def test_confirmation_requires_prior_low_and_anchored_vwap(self):
        frame = pd.DataFrame({
            "decision_ts": pd.date_range("2025-01-04", periods=2, freq="4h", tz="UTC"),
            "daily_source_ts": pd.to_datetime(["2025-01-04T00:00:00Z", "2025-01-04T00:00:00Z"]),
            "open": [100.0, 101.0], "high": [110.0, 106.0], "low": [95.0, 94.0], "close": [108.0, 94.0],
            "moderate_rally": [True, True], "moderate_pre_source_ts": pd.to_datetime(["2025-01-01T00:00:00Z"]*2),
            "moderate_pre_atr": [5.0, 5.0], "moderate_pre_low": [90.0, 90.0],
            "strong_rally": [False, False], "strong_pre_source_ts": pd.to_datetime(["2024-12-30T00:00:00Z"]*2),
            "strong_pre_atr": [5.0, 5.0], "strong_pre_low": [90.0, 90.0],
        })
        known = pd.date_range("2025-01-01T00:05:00Z", "2025-01-04T04:00:00Z", freq="5min")
        work = pd.DataFrame({"known_ts": known, "volume": 1.0, "vwap_num": 100.0})
        confirmed, _ = riskoff.confirmation_sequences(frame, work, "moderate_12pct_3d_1.5atr", 1)
        self.assertEqual(len(confirmed), 1)
        frame.loc[1, "close"] = 100.0
        confirmed, _ = riskoff.confirmation_sequences(frame, work, "moderate_12pct_3d_1.5atr", 1)
        self.assertEqual(confirmed, [])

    def test_higher_high_resets_failure_window(self):
        frame = pd.DataFrame({
            "decision_ts": pd.date_range("2025-01-04", periods=4, freq="4h", tz="UTC"),
            "daily_source_ts": pd.to_datetime(["2025-01-04T00:00:00Z"]*4),
            "open": [100.0]*4, "high": [110.0, 112.0, 108.0, 107.0], "low": [95.0, 96.0, 95.0, 94.0],
            "close": [108.0, 110.0, 100.0, 94.0], "moderate_rally": [True]*4,
            "moderate_pre_source_ts": pd.to_datetime(["2025-01-01T00:00:00Z"]*4), "moderate_pre_atr": [5.0]*4,
            "moderate_pre_low": [90.0]*4, "strong_rally": [False]*4,
            "strong_pre_source_ts": pd.to_datetime(["2024-12-30T00:00:00Z"]*4), "strong_pre_atr": [5.0]*4,
            "strong_pre_low": [90.0]*4,
        })
        known = pd.date_range("2025-01-01T00:05:00Z", "2025-01-04T12:00:00Z", freq="5min")
        work = pd.DataFrame({"known_ts": known, "volume": 1.0, "vwap_num": 100.0})
        confirmed, _ = riskoff.confirmation_sequences(frame, work, "moderate_12pct_3d_1.5atr", 3)
        self.assertEqual(confirmed[0]["peak_index"], 1)
        self.assertEqual(confirmed[0]["decision_index"], 3)

    def test_daily_downtrend_and_rally_use_shifted_completed_daily_state(self):
        source = inspect.getsource(riskoff.feature_frames)
        self.assertIn("daily.close < daily.ema_20", source)
        self.assertIn("daily.ema20_change_5d < 0", source)
        self.assertIn("daily.return_20d < 0", source)
        self.assertIn("daily.daily_downtrend.shift(days)", source)
        self.assertIn("daily.atr_14d.shift(days)", source)

    def test_daily_ema_exit_has_no_artificial_time_exit(self):
        source = inspect.getsource(riskoff.execute_event)
        self.assertIn("no_natural_ema_exit_before_evaluation_boundary", source)
        self.assertIn("completed_daily_close_below_ema10", source)
        self.assertNotIn('exit_policy in ("daily_ema10_close", "swing_high_trail_7d")', source)

    def test_sentinel_selects_one_definition_per_selected_key_hash(self):
        source = inspect.getsource(riskoff.run)
        self.assertIn('groupby("selected_key_policy_hash", as_index=False).first()', source)
        self.assertIn("sentinel.selected_key_policy_hash.nunique() != 8", source)

    def test_controls_are_frozen_before_outcome_execution(self):
        source = inspect.getsource(riskoff.run)
        freeze_position = source.index('control_keys["control_key_freeze_hash"]')
        outcome_position = source.index("for control in control_keys.to_dict")
        self.assertLess(freeze_position, outcome_position)

    def test_control_summary_deduplicates_economic_addresses(self):
        source = inspect.getsource(riskoff.controls_report)
        self.assertIn('drop_duplicates("control_economic_address_hash")', source)
        self.assertIn("unique_address_coverage", source)

    def test_matched_unmatched_bias_uses_matching_cost_mode(self):
        source = inspect.getsource(riskoff.run)
        self.assertIn('outcomes.groupby("definition_id")[f"net_{mode}_R"]', source)
        self.assertIn('unmatched[mode].get(definition', source)

    def test_imputed_funding_is_not_a_signal_gate(self):
        source = inspect.getsource(riskoff.enumerate_candidates)
        self.assertIn('"imputed_funding_gate_activated": False', source)
        self.assertNotIn("funding_central", source)

    def test_candidate_duplicate_gate_is_definition_scoped(self):
        source = inspect.getsource(riskoff.run)
        self.assertIn('duplicated(["definition_id", "candidate_economic_address_hash"])', source)
        self.assertIn("cross_definition_overlap.csv", source)

    def test_full_5m_history_is_retained_only_for_candidate_symbols(self):
        source = inspect.getsource(riskoff.run)
        self.assertIn("if symbol_rows:", source)
        self.assertIn("bars_cache[symbol] = bars", source)
        self.assertLess(source.index("if symbol_rows:"), source.index("bars_cache[symbol] = bars"))

    def test_evaluation_windows_are_half_open(self):
        self.assertEqual(riskoff.evaluation_window(pd.Timestamp("2025-06-30T23:55:00Z"))[0], "2025-H1")
        self.assertEqual(riskoff.evaluation_window(pd.Timestamp("2025-07-01T00:00:00Z"))[0], "2025-H2")


if __name__ == "__main__":
    unittest.main()
