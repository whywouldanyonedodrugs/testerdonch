import unittest
import pandas as pd

from tools.qlmg_real_controls import build_real_controls, normalize_control_net, standardize_event_ledger, apply_real_control_labels


class RealControlsTest(unittest.TestCase):
    def _fixture(self):
        rows = [
            {"event_id":"a1","candidate_id":"A3c","variant_id":"A3c","family":"A3","symbol":"BTCUSDT","decision_ts":"2025-01-01T00:00:00Z","entry_ts":"2025-01-01T00:05:00Z","exit_ts":"2025-01-01T01:00:00Z","net_R_variant":1.0,"risk_bps_used":100,"parent_regime":"up","mark_price_available":True,"funding_exact":True},
            {"event_id":"a2","candidate_id":"A3c","variant_id":"A3c","family":"A3","symbol":"ETHUSDT","decision_ts":"2025-01-02T00:00:00Z","entry_ts":"2025-01-02T00:05:00Z","exit_ts":"2025-01-02T01:00:00Z","net_R_variant":1.0,"risk_bps_used":120,"parent_regime":"up","mark_price_available":True,"funding_exact":True},
            {"event_id":"b1","candidate_id":"A2c","variant_id":"A2c","family":"A2_redesign_only","symbol":"BTCUSDT","decision_ts":"2025-01-03T00:00:00Z","entry_ts":"2025-01-03T00:05:00Z","exit_ts":"2025-01-03T01:00:00Z","net_R_variant":-1.0,"risk_bps_used":110,"parent_regime":"up","mark_price_available":True,"funding_exact":True},
            {"event_id":"b2","candidate_id":"A2c","variant_id":"A2c","family":"A2_redesign_only","symbol":"ETHUSDT","decision_ts":"2025-01-04T00:00:00Z","entry_ts":"2025-01-04T00:05:00Z","exit_ts":"2025-01-04T01:00:00Z","net_R_variant":-0.5,"risk_bps_used":115,"parent_regime":"up","mark_price_available":True,"funding_exact":True},
            {"event_id":"c1","candidate_id":"B1c","family":"B1","symbol":"BTCUSDT","decision_ts":"2025-01-05T00:00:00Z","entry_ts":"2025-01-05T00:05:00Z","exit_ts":"2025-01-05T01:00:00Z","net_R":0.2,"parent_regime":"unknown","mark_available":False,"funding_exact":False,"mark_proxy_used":True,"funding_proxy_used":True},
        ]
        return standardize_event_ledger(pd.DataFrame(rows), "fixture.parquet")

    def test_normalize_control_net(self):
        self.assertEqual(normalize_control_net(10, 9.0, 30), 3.0)

    def test_build_controls_have_source_ids_and_no_placeholder_formula(self):
        pool = self._fixture()
        cand, ledger, summary = build_real_controls(pool, candidate_keys=["A3c"], control_types=["same_symbol", "same_regime", "generic_momentum"], nulls_per_event=1, seed=1)
        self.assertFalse(ledger.empty)
        self.assertTrue(ledger["control_source_row_id"].notna().all())
        self.assertTrue(ledger["control_window_id"].notna().all())
        self.assertIn("same_symbol", set(summary["control_type"]))
        self.assertIn("same_regime", set(summary["control_type"]))
        # A hard-coded -0.03/event placeholder would be exactly -0.06 here; real controls are sourced from fixture R values.
        self.assertNotIn(-0.06, set(summary["raw_control_net_R"].round(8)))

    def test_b1_capped_even_with_controls(self):
        pool = self._fixture()
        cand, ledger, summary = build_real_controls(pool, candidate_keys=["B1c"], control_types=["generic_momentum", "A2_A3_overlap"], nulls_per_event=1, seed=2)
        labelled = apply_real_control_labels(cand, summary)
        self.assertTrue(labelled["real_control_label"].iloc[0].startswith("seed_limited_support_only"))

    def test_protected_slice_rejected(self):
        bad = pd.DataFrame([{"candidate_id":"x","family":"A3","symbol":"BTCUSDT","decision_ts":"2026-01-01T00:00:00Z","net_R":1.0}])
        with self.assertRaises(ValueError):
            standardize_event_ledger(bad, "bad")


if __name__ == "__main__":
    unittest.main()
