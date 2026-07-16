import unittest

import pandas as pd

from tools import run_kraken_lfbs_021_frozen_2023_presample_confirmation as presample


class Lfbs021FrozenPresampleTests(unittest.TestCase):
    def test_empty_loader_frame_without_timestamp_is_not_usable(self):
        self.assertFalse(presample.has_timestamp_bars(pd.DataFrame()))
        self.assertFalse(presample.has_timestamp_bars(pd.DataFrame({"close": []})))
        self.assertTrue(presample.has_timestamp_bars(pd.DataFrame({"ts": [pd.Timestamp("2023-01-01", tz="UTC")]})))

    def test_frozen_definition_hashes_and_policy_are_exact(self):
        definition = presample.frozen_definition()
        self.assertEqual(definition["definition_id"], "lfbs_v1_021")
        self.assertEqual(definition["parameter_vector_hash"], presample.EXPECTED_PARAMETER_HASH)
        self.assertEqual(definition["selected_key_policy_hash"], presample.EXPECTED_SELECTED_HASH)
        self.assertEqual(definition["reference_days"], 60)
        self.assertEqual(definition["failure_bars"], 3)
        self.assertEqual(definition["parent_context"], "fragile_countertrend_stress")
        self.assertEqual(definition["exit_policy"], "fixed_72h_comparator")

    def test_classification_requires_two_unique_control_groups_and_both_semantics(self):
        summary = pd.DataFrame([
            {"cost_mode": "base", "events": 20, "symbols": 10, "months": 6, "mean_R": 0.2, "profit_factor": 1.2},
            {"cost_mode": "conservative", "events": 20, "symbols": 10, "months": 6, "mean_R": 0.1, "profit_factor": 1.1},
            {"cost_mode": "severe", "events": 20, "symbols": 10, "months": 6, "mean_R": -0.1, "profit_factor": 0.9},
        ])
        controls = pd.DataFrame([
            {"control_class": "same_symbol_same_regime_random_short", "cost_mode": "conservative", "adequate_control": True, "mean_uplift_R": 0.1},
            {"control_class": "generic_failed_breakout_5d_high", "cost_mode": "conservative", "adequate_control": True, "mean_uplift_R": 0.1},
        ])
        self.assertEqual(presample.classify_presample(summary, 0.05, controls), "independent_presample_support")
        same_address_labels = controls.iloc[[0]].copy()
        same_address_labels.loc[:, "control_class"] = "same_symbol_same_regime_random_short|generic_failed_breakout_5d_high"
        self.assertEqual(presample.classify_presample(summary, 0.05, same_address_labels), "fragile_presample_support")

    def test_nonpositive_economics_fail(self):
        summary = pd.DataFrame([
            {"cost_mode": "base", "events": 20, "symbols": 10, "months": 6, "mean_R": -0.1, "profit_factor": 0.9},
            {"cost_mode": "conservative", "events": 20, "symbols": 10, "months": 6, "mean_R": -0.2, "profit_factor": 0.8},
            {"cost_mode": "severe", "events": 20, "symbols": 10, "months": 6, "mean_R": -0.3, "profit_factor": 0.7},
        ])
        controls = pd.DataFrame(columns=["control_class", "cost_mode", "adequate_control", "mean_uplift_R"])
        self.assertEqual(presample.classify_presample(summary, -0.2, controls), "presample_failure")


if __name__ == "__main__":
    unittest.main()
