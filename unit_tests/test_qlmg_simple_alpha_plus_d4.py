from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from tools import run_qlmg_simple_alpha_plus_d4 as mod
from tools.qlmg_regime_stack import validate_no_protected


class SimpleAlphaPlusD4Tests(unittest.TestCase):
    def test_allocate_ready_family_minimum_budget(self):
        budgets = mod.allocate_family_budgets(2400)
        readiness = {r["family"]: r for r in mod.family_readiness_rows()}
        for fam in mod.SIMPLE_FAMILIES:
            if readiness[fam]["readiness"] == "ready_for_path_and_replay":
                self.assertGreaterEqual(budgets[fam], mod.min_budget_for_family(readiness[fam]))

    def test_small_budget_still_gives_ready_families_nonzero(self):
        budgets = mod.allocate_family_budgets(40)
        readiness = {r["family"]: r for r in mod.family_readiness_rows()}
        for fam in mod.SIMPLE_FAMILIES:
            if readiness[fam]["readiness"] == "ready_for_path_and_replay":
                self.assertGreater(budgets[fam], 0)

    def test_generate_candidates_has_subfamilies(self):
        budgets = {fam: 5 for fam in mod.SIMPLE_FAMILIES}
        cands = mod.generate_candidates(budgets, 123, mod.family_readiness_rows(), smoke=True)
        self.assertTrue(cands)
        self.assertTrue(all("subfamily" in c for c in cands))
        self.assertTrue(any(c.get("candidate_type") == "diagnostic_seed" for c in cands))

    def test_cost_x2_not_auto_reject_label(self):
        label, _, _ = mod.label_from_evidence({"events": 10, "net_R": 5, "PF": 1.2, "liquidation_count": 0, "proxy_mark_or_liquidation_evidence_share": 0, "beats_matched_null": True}, stress_x125=2)
        self.assertEqual(label, "promote_to_family_specific_validation")

    def test_validate_no_protected_rejects_timestamp(self):
        df = pd.DataFrame({"decision_ts": ["2026-01-01T00:00:00Z"]})
        with self.assertRaises(RuntimeError):
            validate_no_protected(df, ["decision_ts"])

    def test_wrapper_requires_launch_flag_for_full(self):
        text = Path("tools/run_qlmg_simple_alpha_plus_d4_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("Full tmux launch not started", text)

    def test_data_build_table_columns(self):
        rows = mod.family_readiness_rows()
        required = {"needs_targeted_1m", "needs_top_of_book", "needs_shallow_depth", "needs_public_trades", "needs_liquidation_feed", "needs_pit_sector_map", "needs_catalyst_database", "needs_listing_lifecycle_metadata"}
        for row in rows:
            self.assertTrue(required.issubset(row.keys()))

    def test_d4_next_contract_required_output(self):
        self.assertIn("D4", mod.FAMILY_SUBFAMILIES)
        self.assertIn("dynamic_1p25", mod.FAMILY_SUBFAMILIES["D4"])

    def test_portfolio_event_count_nan_safe_formula(self):
        event_count = int(pd.to_numeric(pd.Series([np.nan]), errors="coerce").fillna(100.0).clip(lower=1.0).iloc[0])
        self.assertEqual(event_count, 100)

    def test_zero_event_candidate_scores_below_losing_nonzero(self):
        zero = mod.score_candidate_row({"events": 0, "net_R": 0, "PF": 0})
        losing = mod.score_candidate_row({"events": 10, "net_R": -100, "PF": 0.5})
        self.assertIn("no_events", zero["hard_penalty_reasons"])
        self.assertLess(zero["robustness_score"], losing["robustness_score"])

    def test_event_count_for_candidate_filters_family(self):
        df = pd.DataFrame({
            "family": ["funding_window_orb_failure", "leader_breakout_long"],
            "simple_subfamily": ["funding_5m_failure", "daily_close_breakout"],
            "liquidity_tier": ["C", "A"],
        })
        cand = {"family": "funding_window_orb_failure", "subfamily": "funding_5m_failure", "tier_filter": "C"}
        self.assertEqual(mod.event_count_for_candidate(df, cand), 1)

    def test_event_thresholds_have_smoke_and_full_modes(self):
        smoke_ctx = SimpleNamespace(args=SimpleNamespace(smoke=True))
        full_ctx = SimpleNamespace(args=SimpleNamespace(smoke=False))
        self.assertLess(mod.min_evaluation_events(smoke_ctx), mod.min_evaluation_events(full_ctx))
        self.assertLess(mod.min_prelead_events(smoke_ctx), mod.min_prelead_events(full_ctx))
        self.assertLess(mod.min_evaluation_events_for_family(full_ctx, "new_perp_listing_event_study"), mod.min_evaluation_events_for_family(full_ctx, "leader_breakout_long"))
        self.assertLess(mod.min_prelead_events_for_family(full_ctx, "post_catalyst_continuation_base"), mod.min_prelead_events_for_family(full_ctx, "weak_asset_spike_fade"))

    def test_negative_positive_count_candidate_not_rankable(self):
        scored = mod.score_candidate_row({"events": 500, "net_R": -1.0, "PF": 0.9})
        self.assertFalse(scored["lead_rankable"])
        self.assertIn("net_R_nonpositive", scored["hard_penalty_reasons"])
        self.assertIn("PF_lte_1", scored["hard_penalty_reasons"])

    def test_path_edge_negative_replay_is_preserved_not_rejected(self):
        label, reason, nxt = mod.label_from_evidence({"events": 500, "net_R": -1.0, "PF": 0.9, "path_edge_flag": True, "beats_matched_null": False})
        self.assertEqual(label, "path_edge_exit_problem")
        self.assertIn("not rejected", reason)
        self.assertIn("rerun", nxt)

    def test_positive_without_null_support_is_not_serious_lead(self):
        label, _, _ = mod.label_from_evidence({"events": 500, "net_R": 10.0, "PF": 1.2, "path_edge_flag": True, "beats_matched_null": False})
        self.assertEqual(label, "path_edge_exit_problem")

    def test_surface_return_supports_percent_risk_override(self):
        df = pd.DataFrame({
            "30m_mfe_bps": [600.0],
            "30m_mae_bps": [25.0],
            "30m_close_return_bps": [200.0],
            "reference_risk_bps": [50.0],
            "liquidity_tier": ["A"],
        })
        ref_based = mod.surface_return_r(df, "30m", 5.0, stop_mult=1.0, cost_mult=0.0).iloc[0]
        pct_based = mod.surface_return_r(df, "30m", 5.0, stop_mult=1.0, cost_mult=0.0, risk_bps_override=300.0).iloc[0]
        self.assertEqual(ref_based, 5.0)
        self.assertEqual(pct_based, 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
