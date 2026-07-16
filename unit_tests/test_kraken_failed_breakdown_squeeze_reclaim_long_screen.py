import inspect
import unittest

import pandas as pd

from tools import run_kraken_failed_breakdown_squeeze_reclaim_long_screen as screen


class FailedBreakdownSqueezeReclaimLongTests(unittest.TestCase):
    def test_frozen_manifest_has_24_definitions_and_eight_selected_policies(self):
        manifest = screen.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())
        self.assertEqual(set(manifest.reference_bars), {20, 60})
        self.assertEqual(set(manifest.reclaim_window_bars), {3, 9})

    def test_support_reference_is_completed_shifted_and_cross_confirmed(self):
        source = inspect.getsource(screen.feature_frames)
        self.assertIn(".min().shift(1)", source)
        self.assertIn("frame.close.shift(1).ge(reference)", source)
        self.assertIn("frame.close.lt(reference)", source)

    def test_raw_enumerator_has_no_parent_or_maximum_hold_preblock(self):
        source = inspect.getsource(screen.enumerate_raw_signals)
        self.assertNotIn("parent_allowed", source)
        self.assertNotIn("blocked_until", source)
        self.assertNotIn("next_allowed", source)
        self.assertNotIn("Timedelta(days=7)", source)

    @staticmethod
    def _sequence_fixture():
        ts = pd.date_range("2025-01-01T00:00:00Z", periods=7, freq="4h")
        frame = pd.DataFrame({
            "decision_ts": ts,
            "low": [101, 100, 97, 94, 96, 100, 101],
            "close": [102, 101, 98, 96, 101, 102, 103],
            "atr_14d": [10.0] * 7,
            "range_low_2": [pd.NA, pd.NA, 100.0, 100.0, 100.0, 94.0, 94.0],
            "breakdown_2": [False, False, True, False, False, False, False],
        })
        known = pd.date_range("2025-01-01T08:05:00Z", periods=100, freq="5min")
        work = pd.DataFrame({"known_ts": known, "volume": 1.0, "vwap_num": 98.0})
        return frame, work

    def test_lower_low_updates_sequence_without_duplicate_and_reclaim_is_later(self):
        frame, work = self._sequence_fixture()
        confirmed, expired = screen.breakdown_sequences(frame, work, 2, 3)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(expired, [])
        self.assertEqual(confirmed[0]["breakdown_index"], 2)
        self.assertEqual(confirmed[0]["decision_index"], 4)
        self.assertEqual(confirmed[0]["sequence_low"], 94.0)

    def test_expired_sequence_emits_no_candidate(self):
        frame, work = self._sequence_fixture()
        frame.loc[4:, "close"] = 99.0
        confirmed, expired = screen.breakdown_sequences(frame, work, 2, 3)
        self.assertEqual(confirmed, [])
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0]["reason"], "reclaim_window_expired")

    @staticmethod
    def _candidate_rows():
        return pd.DataFrame([
            {"candidate_key": "a", "symbol": "PF_XUSD", "decision_ts": pd.Timestamp("2025-01-01T00:00:00Z"), "entry_ts": pd.Timestamp("2025-01-01T00:05:00Z")},
            {"candidate_key": "b", "symbol": "PF_XUSD", "decision_ts": pd.Timestamp("2025-01-03T00:00:00Z"), "entry_ts": pd.Timestamp("2025-01-03T00:05:00Z")},
            {"candidate_key": "c", "symbol": "PF_XUSD", "decision_ts": pd.Timestamp("2025-01-05T00:00:00Z"), "entry_ts": pd.Timestamp("2025-01-05T00:05:00Z")},
        ])

    @staticmethod
    def _executor(key, exit_policy):
        hours = 24 if exit_policy == "daily_ema10_close" else 72 if exit_policy == "fixed_72h" else 168
        event = {
            **key, "entry_price": 100.0, "initial_stop": 90.0, "risk_denominator": 10.0,
            "exit_policy": exit_policy, "exit_ts": key["entry_ts"] + pd.Timedelta(hours=hours),
            "exit_price": 101.0, "exit_reason": "fixture", "gross_R": 0.1,
            "maximum_exit_ts": key["entry_ts"] + pd.Timedelta(hours=hours),
        }
        return event, None

    def test_early_structural_exit_allows_reentry_inside_seven_days(self):
        definition = {"definition_id": "x", "exit_policy": "daily_ema10_close", "parameter_vector_hash": "p"}
        accepted, skips, excluded = screen.simulate_definition(self._candidate_rows(), definition, self._executor)
        self.assertEqual(len(accepted), 3)
        self.assertEqual(skips, [])
        self.assertEqual(excluded, [])

    def test_fixed_hold_blocks_only_until_actual_72_hour_expiry(self):
        definition = {"definition_id": "x", "exit_policy": "fixed_72h", "parameter_vector_hash": "p"}
        accepted, skips, _ = screen.simulate_definition(self._candidate_rows(), definition, self._executor)
        self.assertEqual([row["candidate_key"] for row in accepted], ["a", "c"])
        self.assertEqual([row["candidate_key"] for row in skips], ["b"])

    def test_no_cross_definition_state_sharing(self):
        candidates = self._candidate_rows()
        short = {"definition_id": "short", "exit_policy": "daily_ema10_close", "parameter_vector_hash": "p1"}
        long = {"definition_id": "long", "exit_policy": "swing_low_trail_7d", "parameter_vector_hash": "p2"}
        short_events, _, _ = screen.simulate_definition(candidates, short, self._executor)
        long_events, long_skips, _ = screen.simulate_definition(candidates, long, self._executor)
        self.assertEqual(len(short_events), 3)
        self.assertEqual(len(long_events), 1)
        self.assertEqual(len(long_skips), 2)

    def test_both_up_projection_is_strict_subset_of_all_regime(self):
        definitions = screen.frozen_manifest()
        raw = pd.DataFrame([
            {"reference_bars": 20, "reclaim_window_bars": 3, "raw_signal_address_hash": "a", "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-01", tz="UTC"), "parent_state": "both_up"},
            {"reference_bars": 20, "reclaim_window_bars": 3, "raw_signal_address_hash": "b", "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-02", tz="UTC"), "parent_state": "both_down"},
        ])
        projected = screen.project_parent_policies(raw, definitions)
        group = definitions[(definitions.reference_bars == 20) & (definitions.reclaim_window_bars == 3)].drop_duplicates("selected_key_policy_hash")
        strict_hash = group[group.parent_policy.eq("both_up")].selected_key_policy_hash.iloc[0]
        broad_hash = group[group.parent_policy.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
        strict = set(projected[projected.selected_key_policy_hash.eq(strict_hash)].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq(broad_hash)].raw_signal_address_hash)
        self.assertEqual(strict, {"a"})
        self.assertEqual(broad, {"a", "b"})

    def test_cross_class_control_economic_addresses_are_not_double_counted(self):
        controls = pd.DataFrame([
            {"definition_id": "d", "candidate_key": "a", "control_class": "immediate_completed_breakdown", "control_economic_address_hash": "same"},
            {"definition_id": "d", "candidate_key": "b", "control_class": "generic_20bar_failed_breakdown_reclaim", "control_economic_address_hash": "same"},
            {"definition_id": "d", "candidate_key": "c", "control_class": "same_symbol_same_parent_random_long", "control_economic_address_hash": "other"},
        ])
        retained, rejected = screen.deduplicate_control_addresses(controls)
        self.assertEqual(len(retained), 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected.iloc[0].retained_control_class, "immediate_completed_breakdown")
        self.assertFalse(retained.duplicated(["definition_id", "control_economic_address_hash"]).any())

    def test_control_keys_freeze_before_outcomes(self):
        source = inspect.getsource(screen.run)
        self.assertLess(source.index('control_keys["control_key_freeze_hash"]'), source.index("for key in control_keys.to_dict"))

    def test_indexed_executor_avoids_bar_iterrows(self):
        source = inspect.getsource(screen.execute_event_indexed)
        self.assertIn("searchsorted", source)
        self.assertNotIn("iterrows", source)

    def test_forbidden_signal_gates_are_absent(self):
        source = inspect.getsource(screen.enumerate_raw_signals)
        for forbidden in ("open_interest", "relative_strength", "compression_rank", "prior_high_distance", "flush_threshold"):
            self.assertNotIn(forbidden, source)
        self.assertIn('"imputed_funding_gate_activated": False', source)


if __name__ == "__main__":
    unittest.main()
