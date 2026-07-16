import inspect
import unittest

from tools import run_kraken_cross_family_repair_campaign_closure as closure


class CrossFamilyRepairCampaignClosureTests(unittest.TestCase):
    def test_formal_candidate_is_repaired_screen_winner(self):
        self.assertEqual(closure.FORMAL_ID, "rfbs_v1_010")
        self.assertEqual(closure.NEIGHBOR_ID, "rfbs_v1_007")

    def test_closure_is_ledger_only(self):
        source = inspect.getsource(closure.run)
        self.assertNotIn("enumerate_raw_signals", source)
        self.assertNotIn("build_control_keys", source)
        self.assertNotIn("attach_costs", source)
        self.assertIn("replay_and_path_audits", source)

    def test_raw_controls_are_preserved(self):
        source = inspect.getsource(closure.run)
        self.assertIn('"raw_results_preserved": True', source)
        self.assertNotIn("winsor", source.lower())
        self.assertNotIn("clip(", source)

    def test_stability_gate_requires_economics_forensics_controls_neighbor_and_mechanics(self):
        source = inspect.getsource(closure.run)
        for name in (
            "positive_severe_economics", "positive_after_top_three", "positive_worst_leave_one_symbol",
            "contextual_control_support", "structural_control_support", "frozen_neighbor_support", "event_path_mechanics",
        ):
            self.assertIn(name, source)

    def test_holdout_remains_sealed_and_next_target_is_conditional(self):
        source = inspect.getsource(closure.run)
        self.assertIn('"final_holdout_sealed": True', source)
        self.assertIn("RFBS 010 train-only stability review only", source)
        self.assertIn("Close-confirmed breakout retest long screen", source)


if __name__ == "__main__":
    unittest.main()
