from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import run_qlmg_bybit_historical_execution_data_route as mod
from tools.qlmg_regime_stack import validate_no_protected


class BybitHistoricalExecutionDataRouteTests(unittest.TestCase):
    def test_protected_slice_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"ts": ["2026-01-01T00:00:00Z"]}), ["ts"])

    def test_parse_defaults(self):
        args = mod.parse_args([])
        self.assertEqual(args.seed, 20260628)
        self.assertEqual(args.pilot_window_count, 250)
        self.assertEqual(args.download_cap_gb, 10.0)
        self.assertEqual(args.max_output_gb, 30.0)
        self.assertEqual(args.orderbook_depth_levels, "top5,top25")
        self.assertFalse(args.download_if_feasible)
        self.assertTrue(args.include_d4)
        self.assertTrue(args.include_listing)
        self.assertTrue(args.include_controls)

    def test_allowed_verdicts_conservative(self):
        self.assertIn("candidate_unresolved_missing_execution_data", mod.ALLOWED_VERDICTS)
        self.assertIn("d4_carry_forward_execution_depth", mod.ALLOWED_VERDICTS)
        self.assertNotIn("validated", mod.ALLOWED_VERDICTS)
        self.assertNotIn("live_ready", mod.ALLOWED_VERDICTS)

    def test_route_matrix_separates_historical_and_current_live(self):
        rows = mod.route_matrix_rows()
        types = {r["data_type"] for r in rows}
        self.assertIn("historical_public_trades", types)
        self.assertIn("historical_orderbook", types)
        self.assertIn("historical_liquidation_events", types)
        self.assertIn("v5_current_orderbook", types)
        current = [r for r in rows if r["data_type"] == "v5_current_orderbook"][0]
        self.assertEqual(current["historical_or_live_only"], "current_snapshot_only")
        self.assertFalse(current["contains_enough_information_for_replay"])

    def test_orderbook_classification_fields_present(self):
        row = [r for r in mod.route_matrix_rows() if r["data_type"] == "historical_orderbook"][0]
        for key in ["snapshot_only", "snapshot_plus_deltas", "top_n_levels", "bbo_reconstructable", "sequence_consistent", "timestamp_tight_enough_for_replay"]:
            self.assertIn(key, row)

    def test_probe_cases_three_when_windows_exist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            df = pd.DataFrame([
                {"candidate_id": mod.LISTING_IDS[0], "target_window_id": "w1", "symbol": "0GUSDT", "window_start": "2025-09-01T00:00:00Z", "window_end": "2025-09-01T01:00:00Z"},
                {"candidate_id": mod.D4_CANDIDATE_ID, "target_window_id": "d1", "symbol": "ENAUSDT", "window_start": "2025-01-01T00:00:00Z", "window_end": "2025-01-01T01:00:00Z"},
            ])
            (root / "windows").mkdir()
            df.to_csv(root / "windows/full_window_manifest.csv", index=False)
            cases = mod.probe_cases_from_windows(root)
        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0]["probe_case"], "current_liquid_symbol_metadata_only")
        self.assertEqual(cases[1]["probe_case"], "actual_listing_candidate_window")
        self.assertEqual(cases[2]["probe_case"], "lifecycle_sensitive_or_d4_target_window")

    def test_trades_only_is_not_full_execution_depth(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = mod.RunContext(args=mod.parse_args(["--run-root", td, "--disable-telegram"]), run_root=Path(td), notifier=mod.RunNotifier(Path(td), disabled=True), start=pd.Timestamp("2025-01-01T00:00:00Z"), end=pd.Timestamp("2025-02-01T00:00:00Z"))
            suff = mod.route_sufficiency(ctx)
        self.assertIn("trades_only_not_full_execution_depth", suff["trades_only_status"])
        self.assertEqual(suff["listing_sufficiency"], "insufficient_without_historical_orderbook_plus_public_trades")
        self.assertEqual(suff["d4_sufficiency"], "insufficient_without_liquidation_history_plus_depth_trades")

    def test_download_route_fails_closed_without_verified_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "probe").mkdir()
            pd.DataFrame([{"data_type": "official_history_page", "status": "ok", "http_status": 200}]).to_csv(root / "probe/bybit_probe_results.csv", index=False)
            ctx = mod.RunContext(args=mod.parse_args(["--run-root", td, "--download-if-feasible", "--disable-telegram"]), run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01T00:00:00Z"), end=pd.Timestamp("2025-02-01T00:00:00Z"))
            feasible, reason = mod.route_feasible_downloads(ctx)
        self.assertFalse(feasible)
        self.assertIn("no_machine_verified_target_file_url", reason)

    def test_required_outputs_include_final_report(self):
        with tempfile.TemporaryDirectory() as td:
            outs = mod.required_outputs(Path(td), "decision-report")
            self.assertTrue(any("QLMG_BYBIT_HISTORICAL_EXECUTION_DATA_ROUTE_REPORT.md" in str(p) for p in outs))

    def test_wrapper_launch_gate_text(self):
        text = Path("tools/run_qlmg_bybit_historical_execution_data_route_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("Full tmux launch not started", text)

    def test_stage_list_all_excludes_all(self):
        self.assertNotIn("all", mod.stage_list("all"))


if __name__ == "__main__":
    unittest.main()
