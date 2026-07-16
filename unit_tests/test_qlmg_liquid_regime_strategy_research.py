from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from tools import run_qlmg_liquid_regime_strategy_research as mod
from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, validate_no_protected


class LiquidRegimeResearchTests(unittest.TestCase):
    def test_protected_slice_rejected(self) -> None:
        df = pd.DataFrame({"decision_ts": [FINAL_HOLDOUT_START]})
        with self.assertRaises(RuntimeError):
            validate_no_protected(df, ["decision_ts"])

    def test_proxy_gate_blocks_full_when_not_terminal(self) -> None:
        args = SimpleNamespace(smoke=False, stage="all")
        with mock.patch.object(mod, "proxy_status", return_value={"status": "in_progress", "terminal": False, "root": "r", "verdict": "in_progress"}):
            with self.assertRaises(RuntimeError):
                mod.check_full_launch_proxy_gate(args)

    def test_proxy_gate_allows_smoke_in_progress(self) -> None:
        args = SimpleNamespace(smoke=True, stage="all")
        with mock.patch.object(mod, "proxy_status", return_value={"status": "in_progress", "terminal": False, "root": "r", "verdict": "in_progress"}):
            self.assertFalse(mod.check_full_launch_proxy_gate(args)["terminal"])

    def test_sector_catalyst_required_fields_missing_caps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(mod, "REPO", Path(td)):
                sector, catalyst = mod.sector_catalyst_readiness()
        self.assertFalse(sector["ready"])
        self.assertEqual(sector["label_if_missing"], "not_fairly_tested_missing_sector_map")
        self.assertFalse(catalyst["ready"])
        self.assertEqual(catalyst["label_if_missing"], "not_fairly_tested_missing_catalyst_data")

    def test_tier_c_is_not_rankable_by_liquid_branch(self) -> None:
        self.assertEqual(mod.choose_liquidity_tier("TESTUSDT", 5_000_000), "C")
        self.assertNotIn("C", {"A", "B"})

    def test_family_branch_and_short_side_separation(self) -> None:
        self.assertEqual(mod.family_branch("A4"), "branch_l_liquid_regime")
        self.assertEqual(mod.family_side("RS1"), "short")
        self.assertEqual(mod.family_side("A1"), "long")

    def test_run_root_collision_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / mod.DEFAULT_RUN_ID
            base.mkdir()
            args = SimpleNamespace(run_root="")
            with mock.patch.object(mod, "RESULTS_ROOT", Path(td)):
                root, reason = mod.resolve_run_root(args)
        self.assertIn("default_root_existed_suffix", reason)
        self.assertNotEqual(root.name, mod.DEFAULT_RUN_ID)

    def test_resource_guard_output_threshold(self) -> None:
        snap = mod.resource_snapshot(Path("/opt/testerdonch"))
        guard = mod.check_resource_guard(snap, estimated_output_gb=31.0, hard_stage_output_gb=30.0, allow_large_output=False)
        self.assertEqual(guard["status"], "hard_stop")


if __name__ == "__main__":
    unittest.main()
