import inspect
import unittest

import pandas as pd

from tools import run_kraken_delayed_flush_reclaim_long_screen as delayed


class DelayedFlushReclaimLongTests(unittest.TestCase):
    def test_manifest_is_frozen_24_with_eight_selected_key_policies(self):
        manifest = delayed.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())

    def test_exit_policy_does_not_change_selected_key_hash(self):
        manifest = delayed.frozen_manifest()
        grouped = manifest.groupby(["flush_profile", "stabilization_bars", "parent_policy"])
        self.assertTrue(all(group.selected_key_policy_hash.nunique() == 1 for _, group in grouped))

    def test_parent_policies_fail_closed_on_unknown(self):
        self.assertTrue(delayed.parent_allowed("stress_both_down", "both_down"))
        self.assertFalse(delayed.parent_allowed("stress_both_down", "mixed_at_least_one_down"))
        self.assertTrue(delayed.parent_allowed("all_regime_comparator", "mixed_at_least_one_down"))
        self.assertFalse(delayed.parent_allowed("all_regime_comparator", "unknown"))

    @staticmethod
    def _sequence_frame():
        times = pd.date_range("2025-01-04", periods=4, freq="4h", tz="UTC")
        return pd.DataFrame({
            "decision_ts": times,
            "daily_source_ts": pd.to_datetime(["2025-01-04T00:00:00Z"] * 4),
            "open": [90.0, 88.0, 89.0, 90.0],
            "high": [92.0, 90.0, 91.0, 95.0],
            "low": [85.0, 84.0, 86.0, 87.0],
            "close": [87.0, 88.0, 90.5, 94.0],
            "moderate_flush": [True] * 4,
            "moderate_pre_source_ts": pd.to_datetime(["2025-01-01T00:00:00Z"] * 4),
            "moderate_pre_atr": [5.0] * 4,
            "moderate_pre_high": [100.0] * 4,
            "strong_flush": [False] * 4,
            "strong_pre_source_ts": pd.to_datetime(["2024-12-30T00:00:00Z"] * 4),
            "strong_pre_atr": [5.0] * 4,
            "strong_pre_high": [100.0] * 4,
        })

    @staticmethod
    def _vwap_work():
        known = pd.date_range("2025-01-01T00:05:00Z", "2025-01-04T12:00:00Z", freq="5min")
        return pd.DataFrame({"known_ts": known, "volume": 1.0, "vwap_num": 89.0})

    def test_lower_low_resets_stabilization_count(self):
        confirmed, _ = delayed.reclaim_sequences(
            self._sequence_frame(), self._vwap_work(), "moderate_12pct_3d_1.5atr", 1
        )
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["low_index"], 1)
        self.assertEqual(confirmed[0]["decision_index"], 2)

    def test_reclaim_requires_prior_high_and_anchored_vwap(self):
        frame = self._sequence_frame()
        frame.loc[2, "close"] = 89.5  # Not above the previous completed high of 90.
        confirmed, expired = delayed.reclaim_sequences(
            frame, self._vwap_work(), "moderate_12pct_3d_1.5atr", 1
        )
        self.assertEqual(confirmed, [])
        self.assertEqual(len(expired), 1)

    def test_flush_inputs_are_shifted_completed_daily_state(self):
        source = inspect.getsource(delayed.feature_frames)
        self.assertIn("daily.close.shift(days)", source)
        self.assertIn("daily.high.shift(days)", source)
        self.assertIn("daily.atr_14d.shift(days)", source)
        self.assertIn("daily.daily_source_ts.shift(days)", source)
        self.assertIn('frame[flag] = frame[flag].eq(True)', source)

    def test_daily_ema_exit_has_no_artificial_time_exit(self):
        source = inspect.getsource(delayed.execute_event)
        self.assertIn("no_natural_ema_exit_before_evaluation_boundary", source)
        self.assertIn("completed_daily_close_below_ema10", source)
        self.assertIn('"artificial_horizon_exit": False', source)

    def test_long_stop_gap_is_exchange_adverse(self):
        gap = pd.Series({"open": 90.0, "low": 88.0})
        touched = pd.Series({"open": 100.0, "low": 94.0})
        self.assertEqual(delayed.stop_fill_long(gap, 95.0), 90.0)
        self.assertEqual(delayed.stop_fill_long(touched, 95.0), 95.0)

    def test_long_funding_sign_uses_signed_panel_rate(self):
        events = pd.DataFrame([{
            "event_id": "e1", "symbol": "PF_XUSD",
            "entry_ts": pd.Timestamp("2025-01-01T00:05:00Z"),
            "exit_ts": pd.Timestamp("2025-01-01T01:05:00Z"),
            "entry_price": 100.0, "exit_price": 105.0,
            "risk_denominator": 10.0, "gross_R": .5,
        }])
        panel = pd.DataFrame({
            "symbol": ["PF_XUSD"], "timestamp": [pd.Timestamp("2025-01-01T01:00:00Z")],
            "funding_rate_central": [.001], "funding_rate_conservative": [.002],
            "funding_rate_severe": [.003], "funding_exact": [True], "funding_imputed": [False],
        })
        scored, boundaries = delayed.attach_long_costs(events, panel, "event_id")
        self.assertEqual(len(boundaries), 1)
        self.assertAlmostEqual(scored.iloc[0].funding_central_R, -.01)
        self.assertGreater(scored.iloc[0].funding_central_R, scored.iloc[0].funding_conservative_R)
        self.assertGreater(scored.iloc[0].funding_conservative_R, scored.iloc[0].funding_severe_R)

    def test_control_volatility_match_is_frozen_at_25_percent(self):
        eligible = pd.DataFrame({"atr_14d": [10.0, 20.0], "close": [100.0, 100.0]})
        result = delayed.volatility_matched(eligible, .10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.index.tolist(), [0])

    def test_controls_are_frozen_before_outcome_execution(self):
        source = inspect.getsource(delayed.run)
        freeze_position = source.index('control_keys["control_key_freeze_hash"]')
        outcome_position = source.index("for control in control_keys.to_dict")
        self.assertLess(freeze_position, outcome_position)

    def test_control_summary_uses_actual_complement(self):
        source = inspect.getsource(delayed.control_report)
        self.assertIn("unmatched = candidate[~candidate.candidate_key.isin", source)
        self.assertIn("unmatched_only_candidate_mean_R", source)

    def test_overlap_uses_policy_independent_signal_address(self):
        source = inspect.getsource(delayed.overlap_audits)
        self.assertIn("canonical_signal_address_hash", source)
        self.assertIn("trade_economic_address_hash", source)
        self.assertNotIn("set(left_rows.candidate_key)", source)

    def test_exactness_sentinel_covers_every_selected_key_hash(self):
        source = inspect.getsource(delayed.run)
        self.assertIn('groupby("selected_key_policy_hash", as_index=False).first()', source)
        self.assertIn("sentinel.selected_key_policy_hash.nunique() != 8", source)

    def test_imputed_funding_cannot_activate_signal(self):
        source = inspect.getsource(delayed.enumerate_candidates)
        self.assertIn('"imputed_funding_gate_activated": False', source)
        self.assertNotIn("funding_rate", source)

    def test_evaluation_windows_are_half_open(self):
        self.assertEqual(delayed.evaluation_window(pd.Timestamp("2025-06-30T23:55:00Z"))[0], "2025-H1")
        self.assertEqual(delayed.evaluation_window(pd.Timestamp("2025-07-01T00:00:00Z"))[0], "2025-H2")


if __name__ == "__main__":
    unittest.main()
