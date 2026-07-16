import inspect
import unittest

import numpy as np
import pandas as pd

from tools import run_kraken_strong_close_session_handoff_continuation as screen


class StrongCloseSessionHandoffTests(unittest.TestCase):
    def test_manifest_is_frozen_24_by_eight_selected_policies(self):
        manifest = screen.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())
        self.assertEqual(set(manifest.handoff_hour_utc), {8, 16})
        self.assertEqual(set(manifest.direction), {"long", "short"})

    def test_session_references_are_prior_corresponding_sessions(self):
        source = inspect.getsource(screen._session_grid)
        self.assertIn('rolling(20, min_periods=20).median().shift(1)', source)
        self.assertIn('session_end_hour_utc.eq(hour)', source)
        self.assertIn('execution_bar_count.eq(96)', source)

    def test_quote_volume_is_explicitly_a_capped_proxy(self):
        contract = inspect.getsource(screen._session_grid) + inspect.getsource(screen.enumerate_raw_signals)
        self.assertIn('quote_notional_proxy', contract)
        self.assertIn('"ohlcv_quote_volume_proxy_cap": True', contract)

    def test_strong_close_masks_are_directional_and_close_confirmed(self):
        source = inspect.getsource(screen._session_grid)
        self.assertIn('sessions.body.gt(0)', source)
        self.assertIn('sessions.body.lt(0)', source)
        self.assertIn('close_location_value.ge(0.8)', source)
        self.assertIn('close_location_value.le(0.2)', source)

    def test_raw_enumerator_has_no_parent_or_hold_preblock(self):
        source = inspect.getsource(screen.enumerate_raw_signals)
        for forbidden in ("parent_allowed", "blocked_until", "next_allowed", "maximum_hold"):
            self.assertNotIn(forbidden, source)

    def test_directional_parent_projection_nests(self):
        manifest = screen.frozen_manifest()
        common = {
            "handoff_hour_utc": 8, "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-01", tz="UTC"),
            "decision_ts": pd.Timestamp("2025-01-01", tz="UTC"), "raw_policy_hash": "x",
        }
        raw = pd.DataFrame([
            {**common, "direction": "long", "raw_signal_address_hash": "lu", "parent_state": "both_up"},
            {**common, "direction": "long", "raw_signal_address_hash": "ld", "parent_state": "both_down"},
            {**common, "direction": "short", "raw_signal_address_hash": "sd", "parent_state": "both_down"},
            {**common, "direction": "short", "raw_signal_address_hash": "su", "parent_state": "both_up"},
        ])
        projected = screen.project_parent_policies(raw, manifest)
        for direction, expected in (("long", {"lu"}), ("short", {"sd"})):
            policies = manifest[(manifest.handoff_hour_utc == 8) & (manifest.direction == direction)].drop_duplicates("selected_key_policy_hash")
            aligned = policies[policies.parent_policy.eq("directionally_aligned")].selected_key_policy_hash.iloc[0]
            broad = policies[policies.parent_policy.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
            aligned_rows = set(projected[projected.selected_key_policy_hash.eq(aligned)].raw_signal_address_hash)
            broad_rows = set(projected[projected.selected_key_policy_hash.eq(broad)].raw_signal_address_hash)
            self.assertEqual(aligned_rows, expected)
            self.assertTrue(aligned_rows < broad_rows)

    @staticmethod
    def _candidate_rows():
        return pd.DataFrame([
            {"candidate_key": "a", "symbol": "PF_X", "side": "long", "decision_ts": pd.Timestamp("2025-01-01", tz="UTC"), "entry_ts": pd.Timestamp("2025-01-01", tz="UTC")},
            {"candidate_key": "b", "symbol": "PF_X", "side": "long", "decision_ts": pd.Timestamp("2025-01-01T12:00Z"), "entry_ts": pd.Timestamp("2025-01-01T12:00Z")},
            {"candidate_key": "c", "symbol": "PF_X", "side": "long", "decision_ts": pd.Timestamp("2025-01-02T12:00Z"), "entry_ts": pd.Timestamp("2025-01-02T12:00Z")},
        ])

    @staticmethod
    def _fixture_executor(key, exit_policy):
        hours = 4 if exit_policy == "fixed_8h" else 24
        event = {
            **key, "entry_price": 100.0, "initial_stop": 90.0, "risk_denominator": 10.0,
            "exit_policy": exit_policy, "exit_ts": key["entry_ts"] + pd.Timedelta(hours=hours),
            "exit_price": 101.0, "exit_reason": "fixture", "gross_R": 0.1,
            "maximum_exit_ts": key["entry_ts"] + pd.Timedelta(hours=hours),
        }
        return event, None

    def test_actual_early_exit_allows_reentry_and_state_is_definition_local(self):
        early = {"definition_id": "early", "exit_policy": "fixed_8h", "parameter_vector_hash": "p"}
        long = {"definition_id": "long", "exit_policy": "fixed_24h", "parameter_vector_hash": "q"}
        early_events, early_skips, _ = screen.simulate_definition(self._candidate_rows(), early, self._fixture_executor)
        long_events, long_skips, _ = screen.simulate_definition(self._candidate_rows(), long, self._fixture_executor)
        self.assertEqual(len(early_events), 3)
        self.assertEqual(early_skips, [])
        self.assertEqual([row["candidate_key"] for row in long_events], ["a", "c"])
        self.assertEqual([row["candidate_key"] for row in long_skips], ["b"])

    @staticmethod
    def _execution_fixture(side="long"):
        ts = pd.date_range("2025-01-01", periods=97, freq="5min", tz="UTC")
        bars = pd.DataFrame({"ts": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
        bars.loc[bars.index[-1], ["open", "high", "low", "close"]] = [100.0, 1000.0, 0.0, 500.0]
        four = pd.DataFrame({"decision_ts": pd.date_range("2025-01-01T04:00Z", periods=2, freq="4h"), "high": [101.0, 101.0], "low": [99.0, 99.0]})
        key = {
            "candidate_key": "k", "symbol": "PF_X", "side": side,
            "decision_ts": ts[0], "entry_ts": ts[0], "entry_price": 100.0,
            "initial_stop": 90.0 if side == "long" else 110.0, "risk_denominator": 10.0,
            "evaluation_window_end": pd.Timestamp("2025-07-01", tz="UTC"),
        }
        return bars, four, key

    def test_time_exit_does_not_read_horizon_bar_high_low(self):
        for side in ("long", "short"):
            bars, four, key = self._execution_fixture(side)
            indexed, _ = screen.execute_event_indexed(key, "fixed_8h", screen.indexed_execution_data(bars, four))
            scalar, _ = screen.execute_event_scalar(key, "fixed_8h", bars, four)
            self.assertTrue(indexed["exit_reason"].endswith("_time_exit"))
            self.assertAlmostEqual(indexed["mae_R"], scalar["mae_R"])
            self.assertAlmostEqual(indexed["mfe_R"], scalar["mfe_R"])
            self.assertLess(abs(indexed["mae_R"]), 1.0)
            self.assertLess(indexed["mfe_R"], 1.0)

    def test_control_addresses_are_unique_across_classes(self):
        controls = pd.DataFrame([
            {"definition_id": "d", "candidate_key": "a", "control_class": screen.CONTROL_CLASSES[0], "control_economic_address_hash": "same"},
            {"definition_id": "d", "candidate_key": "b", "control_class": screen.CONTROL_CLASSES[1], "control_economic_address_hash": "same"},
        ])
        retained, rejected = screen.deduplicate_control_addresses(controls)
        self.assertEqual(len(retained), 1)
        self.assertEqual(len(rejected), 1)

    def test_controls_freeze_before_outcome_access(self):
        source = inspect.getsource(screen.run)
        self.assertLess(source.index('control_keys["control_key_freeze_hash"]'), source.index("for key in control_keys.to_dict"))

    def test_forbidden_signal_families_are_absent(self):
        source = inspect.getsource(screen.enumerate_raw_signals)
        for forbidden in ("open_interest", "prior_high", "compression", "relative_strength", "reclaim", "breakdown"):
            self.assertNotIn(forbidden, source)
        self.assertIn('"imputed_funding_gate_activated": False', source)

    def test_fbsr_evidence_level_correction_is_owned_by_snapshot_writer(self):
        source = inspect.getsource(screen.update_central_artifacts)
        self.assertIn('"kraken_fbsr_v1"', source)
        self.assertIn('"level_4_event_ledger_plus_real_controls"', source)


if __name__ == "__main__":
    unittest.main()
