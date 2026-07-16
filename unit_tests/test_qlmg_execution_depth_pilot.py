from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import run_qlmg_execution_depth_pilot as mod
from tools.qlmg_regime_stack import validate_no_protected


class ExecutionDepthPilotTests(unittest.TestCase):
    def test_protected_slice_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"ts": ["2026-01-01T00:00:00Z"]}), ["ts"])

    def test_parse_defaults(self):
        args = mod.parse_args([])
        self.assertEqual(args.seed, 20260628)
        self.assertEqual(args.pilot_window_count, 250)
        self.assertEqual(args.depth_download_cap_gb, 15.0)
        self.assertTrue(args.include_d4)
        self.assertTrue(args.include_listing)
        self.assertFalse(args.download_free_bybit_data)
        self.assertFalse(args.download_vendor_data_if_configured)

    def test_allowed_decision_labels_are_conservative(self):
        self.assertIn("d4_carry_forward_execution_depth", mod.FINAL_STATUSES)
        self.assertIn("pilot_depth_data_unavailable_procure_vendor", mod.FINAL_STATUSES)
        self.assertNotIn("validated", mod.FINAL_STATUSES)
        self.assertNotIn("sealed_ready", mod.FINAL_STATUSES)

    def test_make_d4_window_rejects_protected(self):
        row = {"event_id": "e", "symbol": "BTCUSDT", "entry_ts": "2025-12-31T12:30:00Z"}
        self.assertIsNone(mod.make_d4_window(row, "x", 1))

    def test_pilot_selection_prioritizes_controls(self):
        rows = []
        for i in range(30):
            rows.append({"target_window_id": f"c{i}", "candidate_id": mod.LISTING_IDS[0], "symbol": "A", "window_start": pd.Timestamp("2025-01-01T00:00:00Z") + pd.Timedelta(hours=i), "window_end": pd.Timestamp("2025-01-01T01:00:00Z") + pd.Timedelta(hours=i), "window_role": "control", "selection_bucket": "control_same_time", "priority": 1})
        for i in range(100):
            rows.append({"target_window_id": f"e{i}", "candidate_id": mod.LISTING_IDS[0], "symbol": "A", "window_start": pd.Timestamp("2025-02-01T00:00:00Z") + pd.Timedelta(hours=i), "window_end": pd.Timestamp("2025-02-01T01:00:00Z") + pd.Timedelta(hours=i), "window_role": "candidate_event", "selection_bucket": "high_positive", "priority": 2})
        full = pd.DataFrame(rows)
        pilot, omitted = mod.select_pilot_windows(full, 40)
        self.assertEqual(len(pilot), 40)
        self.assertGreaterEqual((pilot["window_role"] == "control").sum(), 15)
        self.assertGreater(len(omitted), 0)

    def test_source_rows_separate_live_and_historical_types(self):
        with tempfile.TemporaryDirectory() as td:
            args = mod.parse_args(["--run-root", td, "--disable-telegram"])
            ctx = mod.RunContext(args=args, run_root=Path(td), notifier=mod.RunNotifier(Path(td), disabled=True), start=pd.Timestamp("2025-01-01T00:00:00Z"), end=pd.Timestamp("2025-02-01T00:00:00Z"))
            rows = mod.source_rows(ctx)
        types = {r["data_type"] for r in rows}
        self.assertIn("historical_public_trades", types)
        self.assertIn("historical_top_of_book", types)
        self.assertIn("historical_shallow_depth", types)
        self.assertIn("historical_liquidation_events", types)
        self.assertIn("live_liquidation_stream_only", types)
        live = [r for r in rows if r["data_type"] == "live_liquidation_stream_only"][0]
        self.assertTrue(live["live_stream_only"])
        self.assertFalse(live["historical_usable_for_2025_windows"])

    def test_risk_grid_includes_conservative_baselines(self):
        self.assertIn(0.0025, mod.RISK_PCTS)
        self.assertIn(0.005, mod.RISK_PCTS)
        self.assertIn(0.01, mod.RISK_PCTS)
        self.assertIn(0.20, mod.RISK_PCTS)

    def test_vendor_config_does_not_print_secret(self):
        configured, reason = mod.vendor_source_configured("tardis")
        self.assertIsInstance(configured, bool)
        self.assertNotIn("api", reason.lower().replace("api_key", ""))

    def test_required_outputs_include_final_report(self):
        with tempfile.TemporaryDirectory() as td:
            outs = mod.required_outputs(Path(td), "decision-report")
            self.assertTrue(any("QLMG_EXECUTION_DEPTH_PILOT_REPORT.md" in str(p) for p in outs))

    def test_wrapper_launch_gate_text(self):
        text = Path("tools/run_qlmg_execution_depth_pilot_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("Full tmux launch not started", text)

    def test_stage_list_all_excludes_all(self):
        self.assertNotIn("all", mod.stage_list("all"))


if __name__ == "__main__":
    unittest.main()
