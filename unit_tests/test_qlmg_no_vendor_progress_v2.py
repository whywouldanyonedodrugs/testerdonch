import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools import qlmg_evidence_contracts as contracts
from tools import run_qlmg_no_vendor_progress_v2 as mod


class NoVendorProgressV2HelpersTest(unittest.TestCase):
    def test_protected_slice_rejected(self):
        args = mod.parse_args(["--start", "2026-01-01", "--disable-telegram"])
        with self.assertRaises(RuntimeError):
            mod.clamp_window(args)

    def test_manual_path_fallback_to_research_input(self):
        p = mod.resolve_manual_path()
        self.assertTrue(p.exists())
        self.assertIn(p.name, {"QLMG_BACKTESTING_MANUAL_20260630_FULL.md", "testmanual.txt"})

    def test_candidate_registry_budget_accounting_and_no_touch_fills(self):
        args = mod.parse_args(["--a1-budget", "7", "--a4-budget", "8", "--a3-budget", "9", "--disable-telegram"])
        ctx = mod.RunContext(args=args, run_root=Path("/tmp/not_used"), notifier=mod.RunNotifier(Path(tempfile.mkdtemp()), disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-15", tz="UTC"))
        reg = mod.make_candidate_registry(ctx)
        counts = reg.groupby("family").size().to_dict()
        self.assertEqual(counts["A1"], 7)
        self.assertEqual(counts["A4"], 8)
        self.assertEqual(counts["A3_overlay"], 9)
        self.assertFalse(bool(reg["touch_fills_allowed"].fillna(False).any()))
        self.assertFalse(bool(reg.loc[reg["family"].eq("A3_overlay"), "rankable"].any()))

    def test_prior_audit_detects_single_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram", "--smoke"])
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-15", tz="UTC"))
            mod.stage_previous_audit(ctx)
            text = (root / "audit/previous_no_vendor_run_audit.md").read_text()
            self.assertIn("previous_a1_a4_sweep_was_single_translation_not_full_sweep", text)

    def test_no_vendor_outcomes_exclude_waiting_for_vendor(self):
        self.assertNotIn("waiting_for_vendor_data", mod.NO_VENDOR_OUTCOMES)
        self.assertIn("discard_current_translation_no_vendor_path", mod.NO_VENDOR_OUTCOMES)

    def test_capture_calibration_not_validation_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram", "--smoke"])
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-15", tz="UTC"))
            mod.stage_branch_x(ctx)
            mat = pd.read_csv(root / "branch_x/micro_canary_readiness_matrix.csv")
            self.assertTrue(bool(mat["not_alpha_validation"].all()))

    def test_real_control_contract_rejects_placeholder(self):
        df = pd.DataFrame({
            "control_event_id": ["c1"],
            "control_symbol": ["BTCUSDT"],
            "control_decision_ts": ["2025-01-01T00:00:00Z"],
            "matched_candidate_id": ["cand"],
            "matching_basis": ["placeholder copied controls"],
            "source_window_id": ["w1"],
            "feature_source_ts": ["2024-12-31T00:00:00Z"],
        })
        self.assertFalse(contracts.validate_control_rows(df).passed)

    def test_tmux_wrapper_has_launch_gate_and_v2_runner(self):
        text = Path("tools/run_qlmg_no_vendor_progress_v2_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("run_qlmg_no_vendor_progress_v2.py", text)
        self.assertIn("remote Telegram required", text)


if __name__ == "__main__":
    unittest.main()
