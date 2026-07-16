from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import run_qlmg_brutal_no_depth_stress as mod
from tools.qlmg_regime_stack import validate_no_protected


class BrutalNoDepthStressTests(unittest.TestCase):
    def fixture_rows(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"candidate_id": "c1", "event_id": "e1", "window_role": "candidate_event", "risk_bps_used": 100.0, "net_R_1m_mark_proxy": 2.0, "stop_hit_1m": False, "target_hit_1m": True, "mfe_bps_1m": 300.0, "turnover": 1000.0, "decision_ts": "2025-01-01T00:00:00Z"},
            {"candidate_id": "c1", "event_id": "e2", "window_role": "candidate_event", "risk_bps_used": 100.0, "net_R_1m_mark_proxy": -1.0, "stop_hit_1m": True, "target_hit_1m": False, "mfe_bps_1m": 20.0, "turnover": 900.0, "decision_ts": "2025-01-02T00:00:00Z"},
            {"candidate_id": "c1", "event_id": "x1", "window_role": "control", "risk_bps_used": 100.0, "net_R_1m_mark_proxy": 0.5, "stop_hit_1m": False, "target_hit_1m": True, "mfe_bps_1m": 120.0, "turnover": 800.0, "decision_ts": "2025-01-01T00:00:00Z"},
            {"candidate_id": "c1", "event_id": "x2", "window_role": "control", "risk_bps_used": 100.0, "net_R_1m_mark_proxy": -0.25, "stop_hit_1m": True, "target_hit_1m": False, "mfe_bps_1m": 10.0, "turnover": 700.0, "decision_ts": "2025-01-02T00:00:00Z"},
        ])

    def test_protected_slice_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"ts": ["2026-01-01T00:00:00Z"]}), ["ts"])

    def test_parse_defaults(self):
        args = mod.parse_args([])
        self.assertEqual(args.seed, 20260628)
        self.assertTrue(args.include_listing)
        self.assertFalse(args.include_d4)
        self.assertEqual(args.max_output_gb, 30.0)
        self.assertEqual(args.slippage_bps_values, [25.0, 50.0, 100.0, 200.0, 300.0])
        self.assertEqual(args.participation_values, [0.001, 0.0025, 0.005, 0.01])

    def test_bps_to_r_haircut_math(self):
        df = self.fixture_rows()
        stressed, meta = mod.stress_dataframe(df, seed=1, scenario_id="s", entry_slip_bps=25, exit_slip_bps=25, gap_bps=0, participation_cap=None)
        e1 = stressed[stressed["event_id"].eq("e1")].iloc[0]
        self.assertAlmostEqual(float(e1["execution_haircut_R"]), 0.5)
        self.assertAlmostEqual(float(e1["stressed_R"]), 1.5)
        self.assertEqual(meta["invalid_risk_rows"], 0)

    def test_participation_missing_labels_not_fairly_tested(self):
        df = self.fixture_rows().drop(columns=["turnover"])
        stressed, meta = mod.stress_dataframe(df, seed=1, scenario_id="s", participation_cap=0.001)
        self.assertTrue(stressed.empty)
        self.assertEqual(meta["participation_status"], "not_fairly_tested_missing_execution_depth")

    def test_deterministic_missed_fill_sampling(self):
        df = self.fixture_rows()
        a = mod.apply_missed_fill_mask(df, 0.5, 42, "scenario")
        b = mod.apply_missed_fill_mask(df, 0.5, 42, "scenario")
        c = mod.apply_missed_fill_mask(df, 0.5, 43, "scenario")
        self.assertTrue(a.equals(b))
        self.assertFalse(a.equals(c))

    def test_control_normalization_uses_candidate_count(self):
        df = self.fixture_rows()
        stressed, _ = mod.stress_dataframe(df, seed=1, scenario_id="s", entry_slip_bps=0, exit_slip_bps=0)
        rows = mod.summarize_candidate_control(stressed, {"scenario_id": "s", "stress_band": "moderate", "rankable": True})
        row = rows[0]
        self.assertEqual(row["candidate_events"], 2)
        self.assertEqual(row["control_events"], 2)
        self.assertAlmostEqual(row["candidate_net_R"], 1.0)
        self.assertAlmostEqual(row["control_normalized_net_R"], 0.25)
        self.assertAlmostEqual(row["normalized_uplift_R"], 0.75)

    def test_full_hold_baseline_reproduction_no_core_count(self):
        df = pd.DataFrame([
            {"candidate_id": "c", "window_role": "candidate_event", "window_scope": "full_hold", "net_R_1m_mark_proxy": 1.0},
            {"candidate_id": "c", "window_role": "candidate_event", "window_scope": "core_24h", "net_R_1m_mark_proxy": 100.0},
            {"candidate_id": "c", "window_role": "control", "window_scope": "full_hold", "net_R_1m_mark_proxy": 0.25},
        ])
        full = df[df["window_scope"].eq("full_hold")]
        base = mod.baseline_from_events(full)
        self.assertEqual(int(base.iloc[0]["candidate_event_count"]), 1)
        self.assertAlmostEqual(float(base.iloc[0]["event_signal_R"]), 1.0)

    def test_allowed_labels_conservative(self):
        self.assertIn("not_fairly_tested_missing_execution_depth", mod.SURVIVAL_LABELS)
        self.assertIn("vendor_pilot_high_priority", mod.FINAL_VERDICTS)
        for bad in ["validated", "live_ready", "sealed_ready"]:
            self.assertNotIn(bad, mod.FINAL_VERDICTS)

    def test_wrapper_launch_gate_text(self):
        text = Path("tools/run_qlmg_brutal_no_depth_stress_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("Full tmux launch not started", text)

    def test_required_outputs_include_final_report(self):
        with tempfile.TemporaryDirectory() as td:
            outs = mod.required_outputs(Path(td), "decision-report")
            self.assertTrue(any("QLMG_BRUTAL_NO_DEPTH_STRESS_REPORT.md" in str(p) for p in outs))

    def test_stage_list_all_excludes_all(self):
        self.assertNotIn("all", mod.stage_list("all"))


if __name__ == "__main__":
    unittest.main()
