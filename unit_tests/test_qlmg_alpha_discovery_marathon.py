from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START
from tools.run_qlmg_alpha_discovery_marathon import (
    allocate_family_budgets,
    apply_candidate_filter,
    done_path,
    generate_candidates,
    matched_null_for_candidate,
    required_outputs_for_stage,
    score_candidate_row,
    stage_complete,
    surface_return_r,
)


class QlmgAlphaDiscoveryMarathonTests(unittest.TestCase):
    def test_budget_allocation_respects_total_and_caps(self) -> None:
        alloc = allocate_family_budgets(1800)
        self.assertEqual(sum(alloc.values()), 1800)
        self.assertLessEqual(alloc["D3_D4_E1"], 350)
        self.assertGreater(alloc["D3_D4_E1"], 0)
        small = allocate_family_budgets(40)
        self.assertEqual(sum(small.values()), 40)
        self.assertGreater(small["D3_D4_E1"], 0)

    def test_scoring_hard_penalties(self) -> None:
        bad = score_candidate_row({"events": 100, "net_R": -1, "PF": 2, "liquidation_count": 0})
        self.assertFalse(bad["lead_rankable"])
        self.assertIn("net_R_nonpositive", bad["hard_penalty_reasons"])
        liq = score_candidate_row({"events": 100, "net_R": 10, "PF": 2, "liquidation_count": 1})
        self.assertFalse(liq["lead_rankable"])
        good = score_candidate_row({"events": 100, "net_R": 10, "PF": 1.2, "liquidation_count": 0, "max_symbol_positive_share": 0.1, "max_month_positive_share": 0.1, "max_event_cluster_positive_share": 0.1})
        self.assertTrue(good["lead_rankable"])

    def test_candidate_generation_blocks_h1_when_not_ready(self) -> None:
        readiness = [
            {"family": "D3", "readiness": "ready_for_discovery", "max_verdict_cap": "x", "proxy_sector_only": False},
            {"family": "H1", "readiness": "needs_targeted_1m", "max_verdict_cap": "x", "proxy_sector_only": False},
            {"family": "B1", "readiness": "needs_sector_map", "max_verdict_cap": "x", "proxy_sector_only": True},
        ]
        cands = generate_candidates({"D3_D4_E1": 5, "H1": 5, "B1_C2": 5}, 123, readiness, smoke=False)
        fams = {c["family"] for c in cands}
        self.assertIn("D3", fams)
        self.assertNotIn("H1", fams)
        self.assertNotIn("B1", fams)

    def test_surface_returns_pessimistic_same_bar(self) -> None:
        df = pd.DataFrame({"reference_risk_bps": [100.0], "24h_mfe_bps": [600.0], "24h_mae_bps": [120.0], "24h_close_return_bps": [50.0], "liquidity_tier": ["A"]})
        ret = surface_return_r(df, "24h", 5.0, 1.0, 0.0, branch="pessimistic")
        self.assertEqual(float(ret.iloc[0]), -1.0)
        opt = surface_return_r(df, "24h", 5.0, 1.0, 0.0, branch="optimistic")
        self.assertEqual(float(opt.iloc[0]), 5.0)

    def test_apply_candidate_filter_uses_proxy_mappings(self) -> None:
        df = pd.DataFrame({
            "family": ["D3", "E1", "A2"],
            "liquidity_tier": ["C", "B", "A"],
            "btc_eth_non_deteriorating": [True, True, True],
            "parent_trend_label": ["neutral_up", "down", "strong_up"],
            "deleveraged_2of4": [True, True, False],
            "funding_percentile_bucket": ["low", "mid", "high"],
            "price_oi_matrix_24h": ["price_down_oi_down", "price_down_oi_down", "price_up_oi_up"],
            "liquidity_quality_label": ["normal_or_unknown", "normal_or_unknown", "normal_or_unknown"],
        })
        d4 = apply_candidate_filter(df, {"family": "D4", "deleveraged_gate": True, "tier_filter": "any", "funding_gate": "low_mid", "price_oi_gate": "price_down_oi_down", "parent_gate": "none", "liquidity_quality_gate": "none", "bad_wick_gate": "none"})
        self.assertEqual(len(d4), 2)
        a1 = apply_candidate_filter(df, {"family": "A1", "tier_filter": "A_B", "parent_gate": "up_or_neutral", "deleveraged_gate": False, "funding_gate": "none", "price_oi_gate": "none", "liquidity_quality_gate": "none", "bad_wick_gate": "none"})
        self.assertEqual(len(a1), 1)

    def test_matched_null_is_deterministic_and_counts_effective_support(self) -> None:
        ev = pd.DataFrame({
            "event_id": ["e1", "e2"], "family": ["D3", "D3"], "symbol": ["A", "B"], "liquidity_tier": ["C", "C"],
            "decision_ts": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"), "reference_risk_bps": [100, 100],
            "24h_mfe_bps": [300, 300], "24h_mae_bps": [50, 50], "24h_close_return_bps": [100, 100]
        })
        nu = pd.concat([ev.assign(event_id=["n1", "n2"]), ev.assign(event_id=["n3", "n4"]), ev.assign(event_id=["n5", "n6"])], ignore_index=True)
        cand = {"candidate_id": "c", "family": "D3", "tier_filter": "C", "parent_gate": "none", "deleveraged_gate": False, "funding_gate": "none", "price_oi_gate": "none", "liquidity_quality_gate": "none", "bad_wick_gate": "none", "horizon": "24h", "target_r": 2.0, "stop_mult": 1.0, "cost_mult": 0.0}
        a = matched_null_for_candidate(ev, nu, cand, 3, 42)
        b = matched_null_for_candidate(ev, nu, cand, 3, 42)
        self.assertEqual(a, b)
        self.assertEqual(a["effective_nulls_per_event"], 3.0)

    def test_stage_complete_requires_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage = "seal-guard"
            done_path(root, stage).parent.mkdir(parents=True)
            done_path(root, stage).write_text("done")
            self.assertFalse(stage_complete(root, stage))
            for p in required_outputs_for_stage(root, stage):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
            self.assertTrue(stage_complete(root, stage))

    def test_tmux_wrapper_exists(self) -> None:
        self.assertTrue(Path("tools/run_qlmg_alpha_discovery_marathon_tmux.sh").exists())
        self.assertEqual(str(FINAL_HOLDOUT_START), "2026-01-01 00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
