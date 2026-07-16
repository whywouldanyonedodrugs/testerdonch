from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.qlmg_regime_stack import (
    FINAL_HOLDOUT_START,
    build_regime_panel,
    label_overlap_matrix,
    trailing_percentile,
    validate_no_protected,
)
from tools.qlmg_strategy_contracts import allocate_sweep_budget, family_contract
from tools.run_qlmg_regime_stack_and_smart_sweep import (
    apply_candidate_filter,
    candidate_result,
    done_path,
    generate_candidates,
    required_outputs_for_stage,
    stage_complete,
    surface_return_r,
)


class QlmgRegimeStackSmartSweepTests(unittest.TestCase):
    def test_protected_slice_rejected(self) -> None:
        df = pd.DataFrame({"decision_ts": [pd.Timestamp("2026-01-01", tz="UTC")]})
        with self.assertRaises(RuntimeError):
            validate_no_protected(df, ["decision_ts"])
        self.assertEqual(str(FINAL_HOLDOUT_START), "2026-01-01 00:00:00+00:00")

    def test_trailing_percentile_uses_no_future_rows(self) -> None:
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        s2 = pd.Series([1.0, 2.0, 3.0, 4.0, -1000.0])
        p1 = trailing_percentile(s1, window=4, min_periods=2)
        p2 = trailing_percentile(s2, window=4, min_periods=2)
        pd.testing.assert_series_equal(p1.iloc[:4], p2.iloc[:4])

    def test_regime_panel_basic_labels_and_no_protected(self) -> None:
        df = pd.DataFrame({
            "event_id": ["e1", "e2", "e3"],
            "family": ["D3", "D3", "E1"],
            "variant_id": ["v", "v", "v"],
            "symbol": ["AAA", "AAA", "BBB"],
            "side": ["long", "long", "long"],
            "liquidity_tier": ["C", "C", "B"],
            "decision_ts": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            "btc_eth_regime": ["both_positive", "both_negative", "mixed"],
            "range_pct": [0.01, 0.02, 0.03],
            "atr_bps": [100, 120, 150],
            "turnover": [1000, 2000, 0],
            "ret_24h": [-0.10, 0.08, -0.20],
            "oi_chg_24h": [-0.05, 0.03, -0.1],
            "funding_rate": [0.0, 0.001, -0.001],
            "24h_mfe_bps": [500, 200, 900],
            "24h_mae_bps": [100, 400, 100],
        })
        reg = build_regime_panel(df, min_history=1)
        self.assertEqual(len(reg), 3)
        self.assertIn("parent_trend_label", reg.columns)
        self.assertTrue(reg["feature_ts"].le(reg["decision_ts"]).all())
        self.assertIn("price_down_oi_down", set(reg["price_oi_matrix_24h"]))

    def test_regime_panel_does_not_use_future_path_for_deleveraging(self) -> None:
        base = pd.DataFrame({
            "event_id": ["e1", "e2", "e3", "e4"],
            "family": ["D3"] * 4,
            "variant_id": ["v"] * 4,
            "symbol": ["AAA"] * 4,
            "side": ["long"] * 4,
            "liquidity_tier": ["C"] * 4,
            "decision_ts": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "ret_24h": [-0.10, -0.10, -0.10, -0.10],
            "oi_chg_24h": [-0.05, -0.05, -0.05, -0.05],
            "funding_rate": [0.0, 0.0, 0.0, 0.0],
            "24h_mfe_bps": [1000, 1000, -1000, -1000],
            "24h_mae_bps": [-1000, -1000, 1000, 1000],
        })
        changed_future_path = base.copy()
        changed_future_path["24h_mfe_bps"] = [-1000, -1000, 1000, 1000]
        changed_future_path["24h_mae_bps"] = [1000, 1000, -1000, -1000]
        a = build_regime_panel(base, min_history=1)
        b = build_regime_panel(changed_future_path, min_history=1)
        pd.testing.assert_series_equal(a["deleveraged_2of4"], b["deleveraged_2of4"], check_names=False)
        self.assertTrue(a["post_flush_reclaim_proxy"].eq(False).all())
        self.assertIn("disabled_future_path_label", a["post_flush_reclaim_proxy_status"].iloc[0])

    def test_label_overlap_matrix_shape(self) -> None:
        reg = pd.DataFrame({"a": ["x", "y"], "b": ["x", "z"]})
        out = label_overlap_matrix(reg, ["a", "b"])
        self.assertEqual(len(out), 4)

    def test_budget_allocation_deterministic_and_total(self) -> None:
        a = allocate_sweep_budget(360, include_shorts=True)
        b = allocate_sweep_budget(360, include_shorts=True)
        self.assertEqual(a, b)
        self.assertEqual(sum(a.values()), 360)
        self.assertEqual(a["D3_D4_E1"], 160)

    def test_family_contract_contains_governance(self) -> None:
        c = family_contract("D3")
        self.assertTrue(c["no_live_trading"])
        self.assertEqual(c["protected_holdout_start"], "2026-01-01T00:00:00Z")
        self.assertIn("active_regimes", c)

    def test_candidate_generation_deterministic(self) -> None:
        a = generate_candidates(20, 123, include_shorts=False)
        b = generate_candidates(20, 123, include_shorts=False)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 28)
        self.assertTrue(any(x["candidate_type"] == "no_regime_baseline" for x in a))

    def test_surface_returns_pessimistic_same_bar(self) -> None:
        df = pd.DataFrame({"reference_risk_bps": [100.0], "24h_mfe_bps": [600.0], "24h_mae_bps": [150.0], "24h_close_return_bps": [50.0], "liquidity_tier": ["A"]})
        ret = surface_return_r(df, "24h", 5.0, 1.0, 0.0, branch="pessimistic")
        self.assertEqual(float(ret.iloc[0]), -1.0)
        opt = surface_return_r(df, "24h", 5.0, 1.0, 0.0, branch="optimistic")
        self.assertEqual(float(opt.iloc[0]), 5.0)

    def test_candidate_filter_and_result(self) -> None:
        df = pd.DataFrame({
            "family": ["D3", "D3", "A2"],
            "symbol": ["AAA", "BBB", "CCC"],
            "liquidity_tier": ["C", "C", "A"],
            "decision_ts": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            "parent_trend_label": ["strong_up", "down", "strong_up"],
            "btc_eth_non_deteriorating": [True, False, True],
            "deleveraged_2of4": [True, False, False],
            "funding_percentile_bucket": ["low", "high", "mid"],
            "price_oi_matrix_24h": ["price_down_oi_down", "price_down_oi_up", "price_up_oi_up"],
            "reference_risk_bps": [100, 100, 100],
            "24h_mfe_bps": [500, 100, 300],
            "24h_mae_bps": [50, 200, 50],
            "24h_close_return_bps": [100, -50, 50],
            "24h_liquidation_10x": [False, False, False],
        })
        cand = {"family": "D3", "tier_filter": "C", "parent_gate": "non_deteriorating", "deleveraged_gate": True, "funding_gate": "low_mid", "price_oi_gate": "price_down_oi_down", "liquidity_quality_gate": "none", "horizon": "24h", "target_r": 3.0, "stop_mult": 1.0, "cost_mult": 0.0}
        sub = apply_candidate_filter(df, cand)
        self.assertEqual(len(sub), 1)
        res = candidate_result(df, {"candidate_id": "x", **cand})
        self.assertEqual(res["events"], 1)
        self.assertGreater(res["mean_R"], 0)

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
        self.assertTrue(Path("tools/run_qlmg_regime_stack_and_smart_sweep_tmux.sh").exists())


if __name__ == "__main__":
    unittest.main()
