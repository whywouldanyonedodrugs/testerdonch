import unittest

import pandas as pd

from tools import run_kraken_c2_sample_limited_economic_tranche as c2


class C2SampleLimitedEconomicTrancheTests(unittest.TestCase):
    def test_frozen_definition_hash_reproduces_manifest(self):
        definitions = pd.read_csv(c2.BUDGET_ROOT / "redesign/c2_sample_limited_definition_manifest.csv")
        self.assertTrue(all(c2.canonical_definition_hash(row) == row.parameter_vector_hash for _, row in definitions.iterrows()))

    def test_daily_bar_source_close_is_after_underlying_bar_knowledge(self):
        bars = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01T23:55:00Z", "2024-01-02T00:00:00Z"]), "open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]})
        daily = c2.daily_bars(bars)
        self.assertTrue((daily.source_close_ts.dt.tz is not None))
        self.assertGreaterEqual(daily.source_close_ts.max(), bars.ts.min() + pd.Timedelta(minutes=5))

    def test_cost_modes_are_monotonic_in_fee_and_slippage(self):
        self.assertLessEqual(c2.COST_MODES["base"][0], c2.COST_MODES["conservative"][0])
        self.assertLess(c2.COST_MODES["base"][1], c2.COST_MODES["conservative"][1])
        self.assertLess(c2.COST_MODES["conservative"][0], c2.COST_MODES["severe"][0])
        self.assertLess(c2.COST_MODES["conservative"][1], c2.COST_MODES["severe"][1])

    def test_controls_without_paired_candidate_outcome_are_not_adjudicated(self):
        controls = pd.DataFrame({"candidate_key": ["kept", "attrited"]})
        candidates = pd.DataFrame({"candidate_key": ["kept"]})
        marked = c2.mark_paired_control_outcomes(controls, candidates)
        self.assertEqual(marked.paired_candidate_outcome_present.tolist(), [True, False])
        self.assertEqual(marked.excluded_from_paired_comparison.tolist(), [False, True])

    def test_classification_requires_control_uplift_for_fragile_positive(self):
        self.assertEqual(c2.classify_definition(False, True, 1), "fragile_positive_sample_limited")
        self.assertEqual(c2.classify_definition(False, True, 0), "current_translation_weak")
        self.assertEqual(c2.classify_definition(True, True, 2), "sample_limited_economic_lead")


if __name__ == "__main__":
    unittest.main()
