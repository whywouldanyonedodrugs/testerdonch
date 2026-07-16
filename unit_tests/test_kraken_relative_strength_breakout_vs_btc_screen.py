import inspect
import unittest

import numpy as np
import pandas as pd

from tools import run_kraken_relative_strength_breakout_vs_btc_screen as screen


class RelativeStrengthBreakoutVersusBTCTests(unittest.TestCase):
    def test_manifest_is_frozen_24_with_four_raw_and_eight_selected_policies(self):
        manifest = screen.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())
        self.assertEqual(
            {screen.raw_policy_hash(u, r) for u in (20, 60) for r in (20, 60)},
            {screen.raw_policy_hash(u, r) for u, r in manifest[["usd_lookback_bars", "relative_lookback_bars"]].drop_duplicates().itertuples(index=False)},
        )

    def test_usd_and_relative_references_are_prior_completed(self):
        source = inspect.getsource(screen.feature_frames)
        self.assertIn("rolling(lookback, min_periods=lookback).max().shift(1)", source)
        self.assertIn("alt.merge(btc", source)
        self.assertIn('validate="one_to_one"', source)
        self.assertIn('how="inner"', source)

    def test_dual_breakout_is_same_bar_and_unresolved_state_resets_only_below_level(self):
        ts = pd.date_range("2025-01-01", periods=7, freq="4h", tz="UTC")
        frame = pd.DataFrame({
            "decision_ts": ts,
            "close": [9.0, 9.0, 11.0, 12.0, 9.0, 11.0, 12.0],
            "relative_close": [0.9, 0.9, 1.1, 1.2, 1.1, 1.1, 1.2],
            "usd_level_2": [np.nan, np.nan, 10.0, 10.0, 10.0, 10.0, 10.0],
            "relative_level_2": [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        signals = screen.dual_breakout_signals(frame, 2, 2)
        self.assertEqual([row["decision_index"] for row in signals], [2, 5])

    def test_previous_bar_above_both_prevents_a_repeated_candidate(self):
        frame = pd.DataFrame({
            "decision_ts": pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC"),
            "close": [9.0, 11.0, 12.0, 13.0],
            "relative_close": [0.9, 1.1, 1.2, 1.3],
            "usd_level_2": [np.nan, np.nan, 10.0, 10.0],
            "relative_level_2": [np.nan, np.nan, 1.0, 1.0],
        })
        self.assertEqual(screen.dual_breakout_signals(frame, 2, 2), [])

    def test_parent_projection_is_a_strict_pit_subset(self):
        manifest = screen.frozen_manifest()
        raw = pd.DataFrame([
            {"usd_lookback_bars": 20, "relative_lookback_bars": 20, "raw_signal_address_hash": "up", "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-01", tz="UTC"), "parent_state": "both_up"},
            {"usd_lookback_bars": 20, "relative_lookback_bars": 20, "raw_signal_address_hash": "down", "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-02", tz="UTC"), "parent_state": "both_down"},
        ])
        projected = screen.project_parent_policies(raw, manifest)
        policies = manifest[(manifest.usd_lookback_bars == 20) & (manifest.relative_lookback_bars == 20)].drop_duplicates("selected_key_policy_hash")
        strict_hash = policies[policies.parent_policy.eq("both_up")].selected_key_policy_hash.iloc[0]
        broad_hash = policies[policies.parent_policy.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
        strict = set(projected[projected.selected_key_policy_hash.eq(strict_hash)].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq(broad_hash)].raw_signal_address_hash)
        self.assertEqual(strict, {"up"})
        self.assertEqual(broad, {"up", "down"})

    @staticmethod
    def _candidate_rows():
        return pd.DataFrame([
            {"candidate_key": "a", "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-01", tz="UTC")},
            {"candidate_key": "b", "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-02", tz="UTC")},
            {"candidate_key": "c", "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-05", tz="UTC")},
        ])

    @staticmethod
    def _fixture_executor(key, exit_policy):
        hours = 12 if exit_policy == "daily_ema10_close" else 72
        return ({
            **key, "side": "long", "decision_ts": key["entry_ts"], "entry_price": 100.0,
            "initial_stop": 90.0, "risk_denominator": 10.0, "exit_policy": exit_policy,
            "exit_ts": key["entry_ts"] + pd.Timedelta(hours=hours), "maximum_exit_ts": key["entry_ts"] + pd.Timedelta(hours=hours),
            "exit_price": 101.0, "exit_reason": "fixture", "gross_R": 0.1,
        }, None)

    def test_actual_exit_controls_reentry_and_no_state_is_shared_between_definitions(self):
        early = {"definition_id": "early", "exit_policy": "daily_ema10_close", "parameter_vector_hash": "p"}
        fixed = {"definition_id": "fixed", "exit_policy": "fixed_72h", "parameter_vector_hash": "q"}
        early_events, early_skips, _ = screen.simulate_definition(self._candidate_rows(), early, self._fixture_executor)
        fixed_events, fixed_skips, _ = screen.simulate_definition(self._candidate_rows(), fixed, self._fixture_executor)
        self.assertEqual(len(early_events), 3)
        self.assertEqual(early_skips, [])
        self.assertEqual([row["candidate_key"] for row in fixed_events], ["a", "c"])
        self.assertEqual([row["candidate_key"] for row in fixed_skips], ["b"])

    @staticmethod
    def _execution_fixture():
        ts = pd.date_range("2025-01-01", periods=2020, freq="5min", tz="UTC")
        bars = pd.DataFrame({"ts": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
        bars.loc[bars.index[864], ["open", "high", "low", "close"]] = [102.0, 1000.0, 0.0, 500.0]
        frame_ts = pd.date_range("2025-01-01T04:00Z", periods=42, freq="4h")
        frame = pd.DataFrame({
            "decision_ts": frame_ts, "daily_source_ts": frame_ts,
            "daily_close": 110.0, "ema_10": 90.0, "relative_close": 1.1,
        })
        key = {
            "candidate_key": "k", "symbol": "PF_X", "side": "long", "decision_ts": ts[0],
            "entry_ts": ts[0], "entry_price": 100.0, "initial_stop": 90.0,
            "risk_denominator": 10.0, "relative_breakout_level": 1.0,
            "evaluation_window_end": pd.Timestamp("2025-07-01", tz="UTC"),
        }
        return bars, frame, key

    def test_fixed_exit_scalar_indexed_parity_and_horizon_bar_is_not_in_mae_mfe(self):
        bars, frame, key = self._execution_fixture()
        indexed, _ = screen.execute_event_indexed(key, "fixed_72h", screen.indexed_execution_data(bars, frame))
        scalar, _ = screen.execute_event_scalar(key, "fixed_72h", bars, frame)
        self.assertEqual(indexed["exit_ts"], scalar["exit_ts"])
        self.assertEqual(indexed["exit_reason"], scalar["exit_reason"])
        self.assertAlmostEqual(indexed["gross_R"], scalar["gross_R"])
        self.assertAlmostEqual(indexed["mae_R"], scalar["mae_R"])
        self.assertAlmostEqual(indexed["mfe_R"], scalar["mfe_R"])
        self.assertLess(indexed["mfe_R"], 1.0)

    def test_relative_exit_uses_completed_aligned_relative_close(self):
        bars, frame, key = self._execution_fixture()
        frame.loc[frame.index[2], "relative_close"] = 0.9
        indexed, _ = screen.execute_event_indexed(key, "relative_break_7d", screen.indexed_execution_data(bars, frame))
        scalar, _ = screen.execute_event_scalar(key, "relative_break_7d", bars, frame)
        self.assertEqual(indexed["exit_reason"], "completed_alt_btc_close_below_frozen_relative_level")
        self.assertEqual(indexed["exit_ts"], scalar["exit_ts"])

    def test_control_addresses_are_unique_across_evidence_classes(self):
        controls = pd.DataFrame([
            {"definition_id": "d", "candidate_key": "a", "control_class": screen.CONTROL_CLASSES[0], "control_economic_address_hash": "same"},
            {"definition_id": "d", "candidate_key": "b", "control_class": screen.CONTROL_CLASSES[1], "control_economic_address_hash": "same"},
        ])
        retained, rejected = screen.deduplicate_control_addresses(controls)
        self.assertEqual(len(retained), 1)
        self.assertEqual(len(rejected), 1)

    def test_product_exclusions_and_forbidden_signal_gates_are_explicit(self):
        universe_source = inspect.getsource(screen.instrument_universe_audit)
        self.assertIn('"btc_reference_not_alt"', universe_source)
        self.assertIn('"Stablecoin"', universe_source)
        self.assertIn('"Pre-IPO"', universe_source)
        source = inspect.getsource(screen.enumerate_raw_signals)
        for forbidden in ("funding_gate_policy", "open_interest", "prior_high", "compression", "retest", "session_handoff"):
            self.assertNotIn(forbidden, source)
        self.assertIn('"imputed_funding_gate_activated": False', source)

    def test_control_keys_are_frozen_before_outcomes(self):
        source = inspect.getsource(screen.run)
        self.assertLess(source.index('control_keys["control_key_freeze_hash"]'), source.index("for key in control_keys.to_dict"))


if __name__ == "__main__":
    unittest.main()
