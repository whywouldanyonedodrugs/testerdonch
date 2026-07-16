import inspect
import unittest

import pandas as pd

from tools import qlmg_evidence_contracts as evidence
from tools import run_kraken_close_confirmed_breakout_retest_long_screen_v2 as screen


class CloseConfirmedBreakoutRetestV2Tests(unittest.TestCase):
    def test_frozen_manifest_has_24_definitions_and_eight_selected_policies(self):
        manifest = screen.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())

    def test_reference_is_completed_and_shifted(self):
        source = inspect.getsource(screen.feature_frames)
        self.assertIn(".max().shift(1)", source)
        self.assertIn("frame.close.shift(1).le(reference)", source)

    def test_raw_enumerator_has_no_parent_or_maximum_hold_preblock(self):
        source = inspect.getsource(screen.enumerate_raw_signals)
        self.assertNotIn("parent_allowed", source)
        self.assertNotIn("blocked_until", source)
        self.assertNotIn("Timedelta(days=7)", source)

    def test_retest_must_precede_later_reclaim(self):
        ts = pd.date_range("2025-01-01T00:00:00Z", periods=6, freq="4h")
        frame = pd.DataFrame({
            "decision_ts": ts, "high": [98, 99, 102, 101, 103, 104],
            "low": [96, 97, 100.5, 99.5, 100.5, 102], "close": [97, 98, 101, 100, 102, 103],
            "atr_14d": [10] * 6, "range_high_2": [pd.NA, pd.NA, 100, 100, 102, 103],
            "breakout_2": [False, False, True, False, False, False],
        })
        known = pd.date_range("2025-01-01T00:05:00Z", periods=200, freq="5min")
        work = pd.DataFrame({"known_ts": known, "volume": 1.0, "vwap_num": 100.0})
        confirmed, _ = screen.breakout_sequences(frame, work, 2, 3)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["retest_index"], 3)
        self.assertEqual(confirmed[0]["decision_index"], 4)

    @staticmethod
    def _candidate_rows():
        return pd.DataFrame([
            {"candidate_key":"a","symbol":"PF_XUSD","decision_ts":pd.Timestamp("2024-12-31T20:00:00Z"),"entry_ts":pd.Timestamp("2025-01-01T00:00:00Z")},
            {"candidate_key":"b","symbol":"PF_XUSD","decision_ts":pd.Timestamp("2025-01-04T20:00:00Z"),"entry_ts":pd.Timestamp("2025-01-05T00:00:00Z")},
        ])

    @staticmethod
    def _executor(key, exit_policy):
        hours = 24 if exit_policy == "daily_ema10_close" else 72 if exit_policy == "fixed_72h" else 168
        event = {**key,"entry_price":100.0,"initial_stop":90.0,"risk_denominator":10.0,"exit_policy":exit_policy,"exit_ts":key["entry_ts"]+pd.Timedelta(hours=hours),"exit_price":101.0,"exit_reason":"fixture","gross_R":.1,"maximum_exit_ts":key["entry_ts"]+pd.Timedelta(hours=hours)}
        return event, None

    def test_definition_local_actual_exit_allows_early_reentry(self):
        definition = {"definition_id":"x","exit_policy":"daily_ema10_close","parameter_vector_hash":"p"}
        accepted, skips, excluded = screen.simulate_definition(self._candidate_rows(), definition, self._executor)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(skips, [])
        self.assertEqual(excluded, [])

    def test_seven_day_trade_blocks_only_while_open(self):
        definition = {"definition_id":"x","exit_policy":"swing_low_trail_7d","parameter_vector_hash":"p"}
        accepted, skips, _ = screen.simulate_definition(self._candidate_rows(), definition, self._executor)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(skips), 1)
        self.assertEqual(skips[0]["skip_reason"], "same_symbol_same_definition_position_actually_open")

    def test_both_up_projection_is_subset_of_all_regime(self):
        definitions = screen.frozen_manifest()
        raw = pd.DataFrame([
            {"reference_bars":20,"retest_window_bars":3,"raw_signal_address_hash":"a","symbol":"PF_X","entry_ts":pd.Timestamp("2025-01-01",tz="UTC"),"parent_state":"both_up"},
            {"reference_bars":20,"retest_window_bars":3,"raw_signal_address_hash":"b","symbol":"PF_X","entry_ts":pd.Timestamp("2025-01-02",tz="UTC"),"parent_state":"both_down"},
        ])
        projected = screen.project_parent_policies(raw, definitions)
        group = definitions[(definitions.reference_bars == 20) & (definitions.retest_window_bars == 3)].drop_duplicates("selected_key_policy_hash")
        strict_hash = group[group.parent_policy.eq("both_up")].selected_key_policy_hash.iloc[0]
        broad_hash = group[group.parent_policy.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
        strict = set(projected[projected.selected_key_policy_hash.eq(strict_hash)].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq(broad_hash)].raw_signal_address_hash)
        self.assertEqual(strict, {"a"})
        self.assertEqual(broad, {"a", "b"})

    def test_control_keys_freeze_before_outcomes(self):
        source = inspect.getsource(screen.run)
        self.assertLess(source.index('control_keys["control_key_freeze_hash"]'), source.index("for key in control_keys.to_dict"))

    def test_production_executor_is_indexed_not_bar_iterrows(self):
        source = inspect.getsource(screen.execute_event_indexed)
        self.assertIn("searchsorted", source)
        self.assertNotIn("iterrows", source)

    def test_no_forbidden_signal_gates(self):
        source = inspect.getsource(screen.enumerate_raw_signals)
        for forbidden in ("open_interest", "relative_strength", "compression_rank", "prior_high_distance"):
            self.assertNotIn(forbidden, source)
        self.assertIn('"imputed_funding_gate_activated": False', source)

    def test_rolling_walk_forward_label_rejects_expanding_start(self):
        starts = [pd.Timestamp("2023-01-01", tz="UTC")] * 3
        result = evidence.validate_walk_forward_window_label("18_month_train_3_month_test_3_month_step", starts)
        self.assertEqual(result.status, "fail")
        expanding = evidence.validate_walk_forward_window_label("expanding_start_18_month_minimum_3_month_test_3_month_step", starts)
        self.assertEqual(expanding.status, "pass")

    def test_rolling_walk_forward_label_accepts_moving_start(self):
        starts = pd.date_range("2023-01-01", periods=3, freq="3MS", tz="UTC")
        result = evidence.validate_walk_forward_window_label("18_month_train_3_month_test_3_month_step", starts)
        self.assertEqual(result.status, "pass")


if __name__ == "__main__":
    unittest.main()
