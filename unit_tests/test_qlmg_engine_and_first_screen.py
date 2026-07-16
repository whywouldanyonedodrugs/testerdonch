from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from tools.qlmg_screening_core import (
    FundingEvent,
    ReplayConfig,
    ResourceSnapshot,
    check_resource_guard,
    funding_cashflow,
    gross_pnl,
    replay_trade,
)
from tools.run_qlmg_engine_and_first_screen import FINAL_HOLDOUT_START, SCREENING_END, cap_signal_indices, done_path, mark_done


class ResourceGuardTests(unittest.TestCase):
    def test_disk_thresholds_match_phase_contract(self) -> None:
        snap = ResourceSnapshot("/tmp", total_bytes=100, used_bytes=96, free_bytes=4 * 1024**3, mem_total_bytes=None, mem_available_bytes=None)
        res = check_resource_guard(snap, estimated_output_gb=1, hard_free_gb=5, warn_free_gb=7, hard_stage_output_gb=20)
        self.assertEqual(res["status"], "hard_stop")
        snap2 = ResourceSnapshot("/tmp", total_bytes=100, used_bytes=94, free_bytes=6 * 1024**3, mem_total_bytes=None, mem_available_bytes=None)
        res2 = check_resource_guard(snap2, estimated_output_gb=1, hard_free_gb=5, warn_free_gb=7, hard_stage_output_gb=20)
        self.assertEqual(res2["status"], "pass")
        self.assertTrue(res2["warnings"])

    def test_output_threshold_requires_allow_large_output(self) -> None:
        snap = ResourceSnapshot("/tmp", total_bytes=100, used_bytes=0, free_bytes=100 * 1024**3, mem_total_bytes=None, mem_available_bytes=None)
        res = check_resource_guard(snap, estimated_output_gb=21, hard_stage_output_gb=20, allow_large_output=False)
        self.assertEqual(res["status"], "hard_stop")
        res2 = check_resource_guard(snap, estimated_output_gb=21, hard_stage_output_gb=20, allow_large_output=True)
        self.assertEqual(res2["status"], "pass")


class ReplayCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.idx = pd.date_range("2025-01-01T00:00:00Z", periods=8, freq="5min")
        self.bars = pd.DataFrame(
            {
                "open": [100] * 8,
                "high": [100, 103, 104, 105, 106, 107, 108, 109],
                "low": [100, 99, 98, 97, 96, 95, 94, 93],
                "close": [100, 102, 103, 104, 105, 106, 107, 108],
                "mark_high": [100, 103, 104, 105, 106, 107, 108, 109],
                "mark_low": [100, 99, 98, 97, 96, 95, 94, 93],
            },
            index=self.idx,
        )

    def test_long_and_short_gross_pnl(self) -> None:
        self.assertEqual(gross_pnl("long", 100, 110, 1), 10)
        self.assertEqual(gross_pnl("short", 100, 90, 1), 10)
        self.assertEqual(gross_pnl("long", 100, 90, 1), -10)
        self.assertEqual(gross_pnl("short", 100, 110, 1), -10)

    def test_long_stop_and_take_profit(self) -> None:
        tp = replay_trade(self.bars, ReplayConfig("long", self.idx[0], self.idx[0], 100, 95, 103, 1), [])
        self.assertEqual(tp.exit_reason, "target")
        stop = replay_trade(self.bars, ReplayConfig("long", self.idx[0], self.idx[0], 100, 99.5, 120, 1), [])
        self.assertEqual(stop.exit_reason, "stop")

    def test_short_stop_and_take_profit(self) -> None:
        tp = replay_trade(self.bars, ReplayConfig("short", self.idx[0], self.idx[0], 100, 110, 98, 1), [])
        self.assertEqual(tp.exit_reason, "target")
        stop = replay_trade(self.bars, ReplayConfig("short", self.idx[0], self.idx[0], 100, 102, 80, 1), [])
        self.assertEqual(stop.exit_reason, "stop")

    def test_same_bar_ambiguity_stop_first(self) -> None:
        res = replay_trade(self.bars, ReplayConfig("long", self.idx[0], self.idx[0], 100, 99, 103, 1), [])
        self.assertTrue(res.ambiguity_flag)
        self.assertEqual(res.exit_reason, "stop")

    def test_trailing_stop_long_and_short(self) -> None:
        long_res = replay_trade(self.bars, ReplayConfig("long", self.idx[0], self.idx[0], 100, 95, None, 1, trailing_stop_distance=2), [])
        self.assertIn(long_res.exit_reason, {"stop", "time_exit"})
        short_bars = self.bars.copy()
        short_bars["high"] = [100, 101, 102, 103, 104, 105, 106, 107]
        short_bars["low"] = [100, 97, 96, 95, 94, 93, 92, 91]
        short_bars["mark_high"] = short_bars["high"]
        short_bars["mark_low"] = short_bars["low"]
        short_res = replay_trade(short_bars, ReplayConfig("short", self.idx[0], self.idx[0], 100, 105, None, 1, trailing_stop_distance=2), [])
        self.assertIn(short_res.exit_reason, {"stop", "time_exit"})

    def test_funding_sign_by_side(self) -> None:
        self.assertLess(funding_cashflow("long", 1, 100, 0.01), 0)
        self.assertGreater(funding_cashflow("short", 1, 100, 0.01), 0)
        self.assertGreater(funding_cashflow("long", 1, 100, -0.01), 0)
        self.assertLess(funding_cashflow("short", 1, 100, -0.01), 0)

    def test_mark_liquidation_uses_mark_not_last(self) -> None:
        bars = self.bars.copy()
        bars["low"] = 1  # last-price wick only
        bars["mark_low"] = 99
        res = replay_trade(bars, ReplayConfig("long", self.idx[0], self.idx[0], 100, 50, None, 1, leverage=10), [])
        self.assertFalse(res.liquidation_flag)
        bars2 = self.bars.copy()
        bars2["mark_low"] = 80
        res2 = replay_trade(bars2, ReplayConfig("long", self.idx[0], self.idx[0], 100, 50, None, 1, leverage=10), [])
        self.assertTrue(res2.liquidation_flag)

    def test_delist_settlement_closes_long_and_short(self) -> None:
        dts = self.idx[2]
        long_res = replay_trade(self.bars, ReplayConfig("long", self.idx[0], self.idx[0], 100, 50, None, 1, delist_ts=dts), [])
        short_res = replay_trade(self.bars, ReplayConfig("short", self.idx[0], self.idx[0], 100, 150, None, 1, delist_ts=dts), [])
        self.assertEqual(long_res.exit_reason, "delist_settlement")
        self.assertEqual(short_res.exit_reason, "delist_settlement")


class RunnerContractTests(unittest.TestCase):
    def test_screening_cutoff_before_holdout(self) -> None:
        self.assertLess(SCREENING_END, FINAL_HOLDOUT_START)

    def test_stage_done_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mark_done(root, "example")
            self.assertTrue(done_path(root, "example").exists())

    def test_signal_cap_is_deterministic_and_spread(self) -> None:
        self.assertEqual(cap_signal_indices([1, 2, 3], 0), [1, 2, 3])
        self.assertEqual(cap_signal_indices([1, 2, 3], 1), [2])
        self.assertEqual(cap_signal_indices([0, 10, 20, 30, 40], 3), [0, 20, 40])


if __name__ == "__main__":
    unittest.main()
