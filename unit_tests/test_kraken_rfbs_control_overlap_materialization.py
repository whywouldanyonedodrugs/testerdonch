import inspect
import unittest

import pandas as pd

from tools import run_kraken_rfbs_control_overlap_materialization as closure


class RfbsControlOverlapMaterializationTests(unittest.TestCase):
    def fixture_row(self):
        return {
            "symbol": "PF_XBTUSD", "decision_ts": "2025-01-01T00:00:00Z", "entry_ts": "2025-01-01T00:00:00Z",
            "entry_price": 100.0, "initial_stop": 110.0, "risk_denominator": 10.0,
            "exit_policy": "fixed_72h", "maximum_exit_ts": "2025-01-04T00:00:00Z",
        }

    def test_signal_address_excludes_definition_and_policy_ids(self):
        row = self.fixture_row()
        first = closure.signal_address_hash({**row, "definition_id": "a", "candidate_key": "one", "selected_key_policy_hash": "x"})
        second = closure.signal_address_hash({**row, "definition_id": "b", "candidate_key": "two", "selected_key_policy_hash": "y"})
        self.assertEqual(first, second)

    def test_signal_address_changes_with_stop_or_risk(self):
        row = self.fixture_row()
        self.assertNotEqual(closure.signal_address_hash(row), closure.signal_address_hash({**row, "initial_stop": 111.0}))
        self.assertNotEqual(closure.signal_address_hash(row), closure.signal_address_hash({**row, "risk_denominator": 11.0}))

    def test_trade_address_includes_exit_and_maximum(self):
        row = self.fixture_row()
        self.assertNotEqual(closure.trade_address_hash(row), closure.trade_address_hash({**row, "exit_policy": "daily_ema10_close"}))
        self.assertNotEqual(closure.trade_address_hash(row), closure.trade_address_hash({**row, "maximum_exit_ts": "2025-01-05T00:00:00Z"}))

    def test_canonical_timestamp_normalizes_utc(self):
        self.assertEqual(closure.canonical_ts("2025-01-01T01:00:00+01:00"), "2025-01-01T00:00:00Z")

    def test_pairwise_overlap_uses_requested_address_column(self):
        frame = pd.DataFrame({"group": ["a", "a", "b", "b"], "address": ["1", "2", "2", "3"], "candidate_key": ["x", "y", "q", "r"]})
        result = closure.pairwise_overlap(frame, "group", "address").iloc[0]
        self.assertEqual(result.shared_count, 1)
        self.assertAlmostEqual(result.jaccard, 1/3)

    def test_strict_broader_nesting_requires_row_level_blocker_attribution(self):
        source = inspect.getsource(closure.nesting_audit)
        self.assertIn("broader_rows.entry_ts+pd.Timedelta(days=7) > entry_ts", source)
        self.assertIn("unexplained_count", source)
        self.assertIn("sequencing_blocker_details", source)

    def test_repaired_bias_uses_actual_unmatched_only_rows(self):
        events = pd.DataFrame({
            "definition_id": ["d"]*3, "candidate_key": ["a", "b", "c"],
            "net_base_R": [1.0, 2.0, -3.0], "net_conservative_R": [.5, 1.5, -3.5], "net_severe_R": [0.0, 1.0, -4.0],
        })
        controls = pd.DataFrame({
            "definition_id": ["d"], "control_class": ["same_symbol_same_regime_random_short"], "candidate_key": ["a"],
            "control_economic_address_hash": ["u"], "net_base_R": [0.0], "net_conservative_R": [0.0], "net_severe_R": [0.0],
        })
        result = closure.repaired_bias(events, controls)
        row = result[(result.control_class == "same_symbol_same_regime_random_short") & (result.cost_mode == "conservative")].iloc[0]
        self.assertEqual(row.matched_count, 1)
        self.assertEqual(row.unmatched_count, 2)
        self.assertEqual(row.matched_candidate_mean_R, .5)
        self.assertEqual(row.unmatched_only_candidate_mean_R, -1.0)

    def test_control_diagnostic_has_no_floor_or_winsorization(self):
        source = inspect.getsource(closure.add_symmetric_diagnostics)
        self.assertIn("raw_short_price_return", source)
        self.assertIn("atr_normalized_short_pnl", source)
        self.assertNotIn("clip(", source)
        self.assertNotIn("maximum(", source)

    def test_replay_checks_exit_and_r_components(self):
        source = inspect.getsource(closure.replay_and_path_audits)
        for field in ("exit_ts", "exit_reason", "exit_price", "gross_R", "mae_R", "mfe_R"):
            self.assertIn(field, source)

    def test_horizon_paths_use_next_executable_open(self):
        source = inspect.getsource(closure.replay_and_path_audits)
        self.assertIn("bars[bars.ts >= horizon].head(1)", source)
        self.assertIn("marked_at_horizon_next_5m_open", source)

    def test_comparators_are_frozen_and_not_promotable(self):
        source = inspect.getsource(closure.run)
        self.assertIn("frozen_neighborhood_comparator_not_promotable", source)
        self.assertEqual(closure.FORMAL_CANDIDATE, "rfbs_v1_004")

    def test_gate_comparison_does_not_relabel_extreme_valid_controls(self):
        source = inspect.getsource(closure.original_vs_repaired_gate)
        self.assertIn('"mechanically_defective_control_calculation_found": False', source)
        self.assertIn("raw control outcomes", source)

    def test_stability_requires_2023_not_negative(self):
        source = inspect.getsource(closure.run)
        self.assertIn("weakness_explained_without_filter", source)
        self.assertIn('period_2023.iloc[0].mean_R >= 0', source)

    def test_candidate_library_contains_manual_audit_fields(self):
        source = inspect.getsource(closure.run)
        required = (
            "candidate_definition_id", "parameter_vector_hash", "candidate_library_state", "evidence_level_contract",
            "evidence_cap_reason", "validation_run", "holdout_touched", "can_support_strategy_claim",
            "adequate_positive_control_classes", "identity_repair_pass", "bias_repair_pass", "parity_pass",
        )
        for field in required:
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
