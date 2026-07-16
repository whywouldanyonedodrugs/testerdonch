import inspect
import unittest
from tools import run_kraken_rfbs_010_train_only_stability_review as review


class RFBS010StabilityReviewTests(unittest.TestCase):
    def test_candidate_and_neighbor_frozen(self):
        self.assertEqual(review.FORMAL_ID,"rfbs_v1_010"); self.assertEqual(review.NEIGHBOR_ID,"rfbs_v1_007")

    def test_no_signal_control_or_cost_regeneration(self):
        source=inspect.getsource(review.run)
        for forbidden in ("enumerate_raw_signals","build_control_keys","attach_costs","execute_event"):
            self.assertNotIn(forbidden,source)

    def test_design_is_purged_and_frozen(self):
        source=inspect.getsource(review.run)
        self.assertIn('"purge_hours": 72',source); self.assertIn('"embargo_hours": 72',source); self.assertIn('"design_frozen_before_fold_outcome_aggregation": True',source)

    def test_level5_requires_all_seven_gates(self):
        source=inspect.getsource(review.run)
        for gate in ("rolling_paths","cpcv_paths","multiplicity_adjustment","clustered_confidence","two_stable_controls","neighborhood_consistency","mechanics"):
            self.assertIn(gate,source)

    def test_holdout_sealed(self):
        self.assertIn('"final_holdout_sealed":True',inspect.getsource(review.run))

    def test_mode_column_access_is_not_dataframe_method(self):
        source=inspect.getsource(review.run)
        self.assertNotIn("walk.mode.isin",source)
        self.assertNotIn("cpcv_dist.mode.isin",source)
        self.assertNotIn("boot.mode.isin",source)


if __name__=="__main__": unittest.main()
