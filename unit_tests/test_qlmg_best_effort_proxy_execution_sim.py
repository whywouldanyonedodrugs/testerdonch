from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import run_qlmg_best_effort_proxy_execution_sim as mod
from tools.qlmg_regime_stack import validate_no_protected


class BestEffortProxyExecutionSimTests(unittest.TestCase):
    def features(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"candidate_id": "c1", "event_id": "e1", "symbol": "AAAUSDT", "window_role": "candidate_event", "risk_bps_used": 100.0, "base_net_R_1m_mark_proxy": 2.0, "high_low_range_bps": 100.0, "notional_1m": 100000.0, "volume_percentile_symbol_month": 0.8, "wick_ratio_percentile_symbol_month": 0.2, "volatility_percentile_symbol_month": 0.3, "entry_minute_close_minus_open_bps": -10.0, "stop_hit_1m": False, "target_hit_1m": True, "target_r_used": 3.0, "entry_ts_effective": "2025-01-01T00:00:00Z"},
            {"candidate_id": "c1", "event_id": "e2", "symbol": "AAAUSDT", "window_role": "candidate_event", "risk_bps_used": 100.0, "base_net_R_1m_mark_proxy": -1.0, "high_low_range_bps": 400.0, "notional_1m": 1000.0, "volume_percentile_symbol_month": 0.1, "wick_ratio_percentile_symbol_month": 0.95, "volatility_percentile_symbol_month": 0.95, "entry_minute_close_minus_open_bps": 50.0, "stop_hit_1m": True, "target_hit_1m": True, "target_r_used": 3.0, "entry_ts_effective": "2025-01-02T00:00:00Z"},
            {"candidate_id": "c1", "event_id": "x1", "symbol": "AAAUSDT", "window_role": "control", "risk_bps_used": 100.0, "base_net_R_1m_mark_proxy": 0.5, "high_low_range_bps": 80.0, "notional_1m": 100000.0, "volume_percentile_symbol_month": 0.8, "wick_ratio_percentile_symbol_month": 0.2, "volatility_percentile_symbol_month": 0.2, "entry_minute_close_minus_open_bps": -5.0, "stop_hit_1m": False, "target_hit_1m": True, "target_r_used": 3.0, "entry_ts_effective": "2025-01-01T00:00:00Z"},
            {"candidate_id": "c1", "event_id": "x2", "symbol": "AAAUSDT", "window_role": "control", "risk_bps_used": 100.0, "base_net_R_1m_mark_proxy": -0.25, "high_low_range_bps": 80.0, "notional_1m": 100000.0, "volume_percentile_symbol_month": 0.8, "wick_ratio_percentile_symbol_month": 0.2, "volatility_percentile_symbol_month": 0.2, "entry_minute_close_minus_open_bps": -5.0, "stop_hit_1m": False, "target_hit_1m": False, "target_r_used": 3.0, "entry_ts_effective": "2025-01-02T00:00:00Z"},
        ])

    def test_protected_slice_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"ts": ["2026-01-01T00:00:00Z"]}), ["ts"])

    def test_parse_defaults(self):
        args = mod.parse_args([])
        self.assertEqual(args.seed, 20260628)
        self.assertTrue(args.include_listing)
        self.assertTrue(args.include_controls)
        self.assertFalse(args.include_d4)
        self.assertEqual(args.max_output_gb, 30.0)
        self.assertEqual(args.intended_order_notional_usdt, 100.0)

    def test_full_hold_baseline_excludes_core(self):
        df = pd.DataFrame([
            {"candidate_id": "c", "window_role": "candidate_event", "window_scope": "full_hold", "net_R_1m_mark_proxy": 1.0},
            {"candidate_id": "c", "window_role": "candidate_event", "window_scope": "core_24h", "net_R_1m_mark_proxy": 100.0},
            {"candidate_id": "c", "window_role": "control", "window_scope": "full_hold", "net_R_1m_mark_proxy": 0.25},
        ])
        base = mod.baseline_from_events(df[df["window_scope"].eq("full_hold")])
        self.assertEqual(int(base.iloc[0]["candidate_event_count"]), 1)
        self.assertAlmostEqual(float(base.iloc[0]["event_signal_R"]), 1.0)

    def test_spread_models(self):
        f = self.features()
        self.assertAlmostEqual(float(mod.spread_bps_for(f, "flat25").iloc[0]), 25.0)
        self.assertAlmostEqual(float(mod.spread_bps_for(f, "vol50_0p2").iloc[1]), 80.0)
        self.assertAlmostEqual(float(mod.spread_bps_for(f, "stress100_0p3").iloc[1]), 120.0)

    def test_proxy_replay_haircut_and_partial_fill(self):
        sc = {"scenario_id": "t", "scenario_band": "optimistic", "spread_model": "flat25", "cap": 0.01, "fill_policy": "partial", "adverse_bps": 0.0, "gap_bps": 0.0, "random_miss": 0.0, "ambiguity_policy": "pessimistic", "rankable": True}
        out = mod.replay_scenario(self.features(), sc, seed=1, intended_notional=100.0)
        e1 = out[out["event_id"].eq("e1")].iloc[0]
        self.assertAlmostEqual(float(e1["execution_haircut_R"]), 0.5)
        self.assertAlmostEqual(float(e1["signal_R"]), 1.5)
        self.assertAlmostEqual(float(e1["fill_ratio"]), 1.0)

    def test_skip_policy_uses_participation_and_no_fill(self):
        sc = {"scenario_id": "base", "scenario_band": "base", "spread_model": "vol50_0p2", "cap": 0.005, "fill_policy": "skip", "adverse_bps": 50.0, "gap_bps": 50.0, "random_miss": 0.0, "ambiguity_policy": "pessimistic", "rankable": True}
        out = mod.replay_scenario(self.features(), sc, seed=1, intended_notional=100.0)
        e2 = out[out["event_id"].eq("e2")].iloc[0]
        self.assertTrue(bool(e2["skipped_flag"]))
        self.assertEqual(float(e2["fill_ratio"]), 0.0)

    def test_control_normalized_aggregation(self):
        sc = {"scenario_id": "t", "scenario_band": "optimistic", "spread_model": "flat25", "cap": 0.01, "fill_policy": "partial", "adverse_bps": 0.0, "gap_bps": 0.0, "random_miss": 0.0, "ambiguity_policy": "pessimistic", "rankable": True}
        events = mod.replay_scenario(self.features(), sc, seed=1, intended_notional=100.0)
        agg = mod.aggregate_replay(events)
        row = agg.iloc[0]
        self.assertEqual(int(row["event_count_total"]), 2)
        self.assertEqual(int(row["control_count_total"]), 2)
        self.assertIn("normalized_control_uplift_R", row)

    def test_safe_leverage_caps_at_ten(self):
        lev = mod.safe_leverage(pd.Series([100.0, 5000.0]), 1.5)
        self.assertLessEqual(float(lev.iloc[0]), 10.0)
        self.assertLess(float(lev.iloc[1]), 2.0)

    def test_allowed_verdicts_are_conservative(self):
        self.assertIn("micro_canary_possible_execution_only", mod.FINAL_VERDICTS)
        self.assertIn("fails_proxy_execution_current_expression_only", mod.SURVIVAL_LABELS)
        for bad in ["validated", "sealed_ready", "live_ready"]:
            self.assertNotIn(bad, mod.FINAL_VERDICTS)

    def test_wrapper_launch_gate_text(self):
        text = Path("tools/run_qlmg_best_effort_proxy_execution_sim_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("Full tmux launch not started", text)

    def test_required_outputs_include_final_report(self):
        with tempfile.TemporaryDirectory() as td:
            outs = mod.required_outputs(Path(td), "decision-report")
            self.assertTrue(any("QLMG_BEST_EFFORT_PROXY_EXECUTION_SIM_REPORT.md" in str(p) for p in outs))

    def test_stage_list_all_excludes_all(self):
        self.assertNotIn("all", mod.stage_list("all"))


if __name__ == "__main__":
    unittest.main()
