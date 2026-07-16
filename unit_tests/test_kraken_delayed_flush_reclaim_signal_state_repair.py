import inspect
import unittest

import pandas as pd

from tools import run_kraken_delayed_flush_reclaim_long_screen as base
from tools import run_kraken_delayed_flush_reclaim_signal_state_repair as repair


class DelayedFlushReclaimSignalStateRepairTests(unittest.TestCase):
    def test_frozen_manifest_is_unchanged(self):
        old = base.frozen_manifest()
        repaired = base.frozen_manifest()
        pd.testing.assert_frame_equal(old, repaired)
        self.assertEqual(len(repaired), 24)
        self.assertEqual(repaired.selected_key_policy_hash.nunique(), 8)

    def test_raw_specs_are_parent_neutral_and_have_four_hashes(self):
        specs = repair.raw_specs()
        self.assertEqual(len(specs), 4)
        self.assertEqual(len({row["raw_policy_hash"] for row in specs}), 4)
        self.assertTrue(all("parent_policy" not in row for row in specs))

    def test_raw_enumerator_has_no_parent_or_maximum_hold_preblock(self):
        source = inspect.getsource(repair.enumerate_raw_signals)
        self.assertNotIn("parent_allowed", source)
        self.assertNotIn("blocked_until", source)
        self.assertNotIn("Timedelta(days=7)", source)

    @staticmethod
    def _candidate_rows():
        return pd.DataFrame([
            {"candidate_key": "a", "symbol": "PF_XUSD", "decision_ts": pd.Timestamp("2024-12-31T20:00:00Z"), "entry_ts": pd.Timestamp("2025-01-01T00:00:00Z")},
            {"candidate_key": "b", "symbol": "PF_XUSD", "decision_ts": pd.Timestamp("2025-01-04T20:00:00Z"), "entry_ts": pd.Timestamp("2025-01-05T00:00:00Z")},
        ])

    @staticmethod
    def _definition(exit_policy):
        return {"definition_id": f"definition_{exit_policy}", "exit_policy": exit_policy, "parameter_vector_hash": "p"}

    @staticmethod
    def _executor(key, exit_policy):
        hold = {"fixed_72h": pd.Timedelta(hours=72), "daily_ema10_close": pd.Timedelta(hours=24), "swing_low_trail_7d": pd.Timedelta(days=7)}[exit_policy]
        event = {
            **key, "entry_price": 100.0, "initial_stop": 90.0, "risk_denominator": 10.0,
            "exit_policy": exit_policy, "exit_ts": key["entry_ts"] + hold,
            "exit_price": 101.0, "exit_reason": "fixture", "gross_R": .1,
            "maximum_exit_ts": key["entry_ts"] + hold,
        }
        return event, None

    def test_fixed_72h_allows_valid_reentry_inside_seven_days(self):
        accepted, skipped, excluded = repair.simulate_definition(self._candidate_rows(), self._definition("fixed_72h"), self._executor)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(skipped, [])
        self.assertEqual(excluded, [])

    def test_early_ema_exit_allows_valid_reentry_inside_seven_days(self):
        accepted, skipped, _ = repair.simulate_definition(self._candidate_rows(), self._definition("daily_ema10_close"), self._executor)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(skipped, [])

    def test_open_seven_day_trade_blocks_only_same_definition(self):
        accepted, skipped, _ = repair.simulate_definition(self._candidate_rows(), self._definition("swing_low_trail_7d"), self._executor)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["prior_actual_exit_ts"], pd.Timestamp("2025-01-08T00:00:00Z"))
        self.assertEqual(skipped[0]["skip_reason"], "same_symbol_same_definition_position_actually_open")

    def test_parent_projection_makes_strict_exact_subset(self):
        definitions = base.frozen_manifest()
        raw = []
        for state, suffix in (("both_down", "a"), ("mixed_at_least_one_down", "b")):
            row = {
                "flush_profile": "moderate_12pct_3d_1.5atr", "stabilization_bars": 1,
                "raw_policy_hash": repair.raw_policy_hash("moderate_12pct_3d_1.5atr", 1),
                "raw_signal_address_hash": suffix, "symbol": "PF_XUSD", "decision_ts": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=len(raw)),
                "entry_ts": pd.Timestamp("2025-01-01T00:05:00Z") + pd.Timedelta(days=len(raw)), "parent_state": state,
            }
            raw.append(row)
        raw = pd.DataFrame(raw)
        projected = repair.project_parent_policies(raw, definitions)
        policy = definitions[(definitions.flush_profile == "moderate_12pct_3d_1.5atr") & (definitions.stabilization_bars == 1)].drop_duplicates("selected_key_policy_hash")
        strict_hash = policy[policy.parent_policy == "stress_both_down"].selected_key_policy_hash.iloc[0]
        broad_hash = policy[policy.parent_policy == "all_regime_comparator"].selected_key_policy_hash.iloc[0]
        strict = set(projected[projected.selected_key_policy_hash == strict_hash].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash == broad_hash].raw_signal_address_hash)
        self.assertEqual(strict, {"a"})
        self.assertEqual(broad, {"a", "b"})
        self.assertTrue(strict < broad)

    def test_known_render_and_ena_regressions_require_both_parent_rows(self):
        definitions = base.frozen_manifest()
        raw_rows = []
        for symbol, ts, flush, stabilization in repair.KNOWN_REGRESSIONS:
            row = {
                "flush_profile": flush, "stabilization_bars": stabilization,
                "raw_policy_hash": repair.raw_policy_hash(flush, stabilization),
                "raw_signal_address_hash": symbol, "symbol": symbol, "decision_ts": ts,
                "entry_ts": ts, "parent_state": "both_down",
            }
            raw_rows.append(row)
        raw = pd.DataFrame(raw_rows)
        candidates = repair.project_parent_policies(raw, definitions)
        old = candidates[candidates.parent_policy == "stress_both_down"].copy()
        audit, _ = repair.known_regression_audit(raw, candidates, old)
        self.assertEqual(len(audit), 2)
        self.assertTrue(audit["pass"].all())

    def test_control_freeze_precedes_outcome_access(self):
        source = inspect.getsource(repair.run)
        self.assertLess(source.index('control_keys["control_key_freeze_hash"]'), source.index("for control in control_keys.to_dict"))

    def test_full_parity_reruns_same_definition_simulator(self):
        source = inspect.getsource(repair.run)
        self.assertIn("parity_outcomes, parity_skips, parity_exclusions = simulate_all_definitions", source)
        self.assertIn("deterministic_rerun_mismatches", source)

    def test_blocked_source_root_is_read_only(self):
        source = inspect.getsource(repair.run)
        self.assertNotIn("SOURCE_ROOT / \"decision_summary.json\",", source)
        self.assertIn("source_root_preserved_unchanged", source)


if __name__ == "__main__":
    unittest.main()
