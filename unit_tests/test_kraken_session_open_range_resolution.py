import inspect
import unittest

import pandas as pd

from tools import run_kraken_session_open_range_resolution as screen


class SessionOpenRangeResolutionTests(unittest.TestCase):
    def test_manifest_is_frozen_24_by_eight_selected_policies(self):
        manifest = screen.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())
        self.assertEqual(set(manifest.session_open), {"asia_0000_utc", "us_cash_open"})
        self.assertEqual(set(manifest.range_minutes), {30, 60})
        self.assertEqual(set(manifest.exit_policy), {"fixed_2h", "fixed_4h", "fixed_8h"})

    def test_xnys_calendar_is_dst_aware_and_excludes_holidays(self):
        calendar = screen.xnys_session_calendar().set_index("session_date")
        self.assertEqual(calendar.loc["2025-03-07", "open_ts"], pd.Timestamp("2025-03-07T14:30Z"))
        self.assertEqual(calendar.loc["2025-03-10", "open_ts"], pd.Timestamp("2025-03-10T13:30Z"))
        for closed in ("2024-03-29", "2025-01-09", "2025-12-25"):
            self.assertNotIn(closed, calendar.index)

    @staticmethod
    def _bars_for_opening_range():
        ts = pd.date_range("2025-01-06T00:00Z", periods=49, freq="5min")
        bars = pd.DataFrame({"ts": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
        # The 00:30-00:45 completed 15m bar closes above the frozen 00:00-00:30 range.
        bars.loc[bars.ts.isin(pd.date_range("2025-01-06T00:30Z", periods=3, freq="5min")), ["high", "close"]] = [102.0, 102.0]
        return bars

    def test_opening_range_is_completed_and_break_is_close_confirmed(self):
        bars = self._bars_for_opening_range()
        _, work, _ = screen.execution.feature_frames(bars)
        daily = pd.DataFrame([{"daily_source_ts": pd.Timestamp("2025-01-06T00:00Z"), "atr_14d": 5.0, "ema_10": 100.0, "close": 100.0}])
        schedule = pd.DataFrame([{"session_date": "2025-01-06", "open_ts": pd.Timestamp("2025-01-06T00:00Z"), "session_open": "asia_0000_utc"}])
        catalog = screen.opening_range_catalog(work, daily, schedule)
        row = catalog[catalog.range_minutes.eq(30)].iloc[0]
        self.assertEqual(row.range_end_ts, pd.Timestamp("2025-01-06T00:30Z"))
        self.assertEqual(row.decision_ts, pd.Timestamp("2025-01-06T00:45Z"))
        self.assertEqual(row.direction, "long")
        self.assertLessEqual(row.previous_15m_close, row.opening_range_high)

    def test_raw_enumerator_has_no_parent_or_nominal_hold_preblock(self):
        source = inspect.getsource(screen.enumerate_raw_signals)
        for forbidden in ("parent_allowed", "blocked_until", "next_allowed", "maximum_hold"):
            self.assertNotIn(forbidden, source)

    def test_directional_parent_projection_nests_for_each_opening_range(self):
        manifest = screen.frozen_manifest()
        common = {"session_open": "asia_0000_utc", "range_minutes": 30, "symbol": "PF_X", "entry_ts": pd.Timestamp("2025-01-01", tz="UTC"), "decision_ts": pd.Timestamp("2025-01-01", tz="UTC"), "raw_policy_hash": "x"}
        raw = pd.DataFrame([
            {**common, "direction": "long", "raw_signal_address_hash": "lu", "parent_state": "both_up"},
            {**common, "direction": "long", "raw_signal_address_hash": "ld", "parent_state": "both_down"},
            {**common, "direction": "short", "raw_signal_address_hash": "sd", "parent_state": "both_down"},
            {**common, "direction": "short", "raw_signal_address_hash": "su", "parent_state": "both_up"},
        ])
        projected = screen.project_parent_policies(raw, manifest)
        policies = manifest[(manifest.session_open == "asia_0000_utc") & (manifest.range_minutes == 30)].drop_duplicates("selected_key_policy_hash")
        aligned = policies[policies.parent_policy.eq("directionally_aligned")].selected_key_policy_hash.iloc[0]
        broad = policies[policies.parent_policy.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
        self.assertEqual(set(projected[projected.selected_key_policy_hash.eq(aligned)].raw_signal_address_hash), {"lu", "sd"})
        self.assertEqual(set(projected[projected.selected_key_policy_hash.eq(broad)].raw_signal_address_hash), {"lu", "ld", "sd", "su"})

    @staticmethod
    def _candidate_rows():
        return pd.DataFrame([
            {"candidate_key": "a", "symbol": "PF_X", "side": "long", "decision_ts": pd.Timestamp("2025-01-01", tz="UTC"), "entry_ts": pd.Timestamp("2025-01-01", tz="UTC")},
            {"candidate_key": "b", "symbol": "PF_X", "side": "long", "decision_ts": pd.Timestamp("2025-01-01T03:00Z"), "entry_ts": pd.Timestamp("2025-01-01T03:00Z")},
            {"candidate_key": "c", "symbol": "PF_X", "side": "long", "decision_ts": pd.Timestamp("2025-01-01T09:00Z"), "entry_ts": pd.Timestamp("2025-01-01T09:00Z")},
        ])

    @staticmethod
    def _fixture_executor(key, exit_policy):
        hours = 1 if exit_policy == "fixed_2h" else 8
        return {**key, "entry_price": 100.0, "initial_stop": 90.0, "risk_denominator": 10.0, "exit_policy": exit_policy, "exit_ts": key["entry_ts"] + pd.Timedelta(hours=hours), "exit_price": 101.0, "exit_reason": "fixture", "gross_R": 0.1, "maximum_exit_ts": key["entry_ts"] + pd.Timedelta(hours=hours)}, None

    def test_actual_exit_allows_reentry_and_state_is_definition_local(self):
        early = {"definition_id": "early", "exit_policy": "fixed_2h", "parameter_vector_hash": "p"}
        long = {"definition_id": "long", "exit_policy": "fixed_8h", "parameter_vector_hash": "q"}
        early_events, early_skips, _ = screen.simulate_definition(self._candidate_rows(), early, self._fixture_executor)
        long_events, long_skips, _ = screen.simulate_definition(self._candidate_rows(), long, self._fixture_executor)
        self.assertEqual(len(early_events), 3); self.assertEqual(early_skips, [])
        self.assertEqual([row["candidate_key"] for row in long_events], ["a", "c"])
        self.assertEqual([row["candidate_key"] for row in long_skips], ["b"])

    def test_control_addresses_are_unique_across_classes(self):
        controls = pd.DataFrame([
            {"definition_id": "d", "candidate_key": "a", "control_class": screen.CONTROL_CLASSES[0], "control_economic_address_hash": "same"},
            {"definition_id": "d", "candidate_key": "b", "control_class": screen.CONTROL_CLASSES[1], "control_economic_address_hash": "same"},
        ])
        retained, rejected = screen.deduplicate_control_addresses(controls)
        self.assertEqual(len(retained), 1); self.assertEqual(len(rejected), 1)

    def test_controls_freeze_before_outcome_access(self):
        source = inspect.getsource(screen.run)
        self.assertLess(source.index('control_keys["control_key_freeze_hash"]'), source.index("for key in control_keys.to_dict"))

    def test_control_pools_are_streamed_per_symbol(self):
        source = inspect.getsource(screen.build_control_keys)
        self.assertIn('groupby("symbol", sort=True)', source)
        self.assertIn("del pools", source)
        self.assertNotIn("for symbol in feature_cache", source)

    def test_random_control_grid_is_frozen_four_hourly(self):
        source = inspect.getsource(screen.build_control_pool)
        self.assertIn("decision_ts.dt.hour.mod(4).eq(0)", source)

    def test_forbidden_signal_families_are_absent(self):
        source = inspect.getsource(screen.enumerate_raw_signals)
        for forbidden in ("open_interest", "prior_high", "compression", "reclaim"):
            self.assertNotIn(forbidden, source)
        self.assertIn('"imputed_funding_gate_activated": False', source)


if __name__ == "__main__":
    unittest.main()
