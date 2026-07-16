from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.qlmg_regime_stack import validate_no_protected
from tools.run_qlmg_d4_survivability_redesign import (
    CANDIDATE_ID,
    FINAL_HOLDOUT_START,
    conservative_r,
    done_path,
    liquidation_adverse_bps_for_leverage,
    replay_long_path,
    required_outputs,
    stage_complete,
    summarize_returns,
)


class D4SurvivabilityRedesignTests(unittest.TestCase):
    def test_protected_slice_rejected(self) -> None:
        df = pd.DataFrame({"decision_ts": [pd.Timestamp("2026-01-01T00:00:00Z")]})
        with self.assertRaises(RuntimeError):
            validate_no_protected(df, ["decision_ts"])

    def test_liquidation_formula(self) -> None:
        self.assertAlmostEqual(liquidation_adverse_bps_for_leverage(10.0), 950.0)
        self.assertAlmostEqual(liquidation_adverse_bps_for_leverage(5.0), 1950.0)

    def test_conservative_liquidation_and_ambiguous_accounting(self) -> None:
        df = pd.DataFrame({
            "candidate_net_R": [0.5, -0.2, -0.8],
            "candidate_actual_liquidation": [False, True, False],
            "candidate_same_minute_ambiguous": [False, False, True],
        })
        liq = conservative_r(df, include_liq=True, include_ambiguous=False)
        self.assertEqual(float(liq.loc[1, "candidate_net_R"]), -1.0)
        amb = conservative_r(df, include_liq=True, include_ambiguous=True)
        self.assertEqual(float(amb.loc[2, "candidate_net_R"]), -1.25)

    def test_summarize_returns_pf_and_dd(self) -> None:
        df = pd.DataFrame({"candidate_net_R": [1.0, -0.5, 0.5], "symbol": ["A", "A", "B"], "decision_ts": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")})
        out = summarize_returns(df, candidate_id="x")
        self.assertEqual(out["events"], 3)
        self.assertAlmostEqual(out["net_R"], 1.0)
        self.assertAlmostEqual(out["PF"], 3.0)

    def test_replay_long_path_liquidation_before_stop(self) -> None:
        ts = pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="min")
        ohlcv = pd.DataFrame({"timestamp": ts, "open": [100, 100, 100], "high": [101, 101, 101], "low": [98, 89, 89], "close": [100, 99, 98]})
        mark = pd.DataFrame({"timestamp": ts, "open": [100, 100, 100], "high": [101, 101, 101], "low": [99, 90, 89], "close": [100, 99, 98]})
        row = {"entry_ts": ts[0], "stop_distance_bps_1m": 1500.0, "cost_bps_1m": 30.0}
        out = replay_long_path(row, ohlcv, mark, entry_ts=ts[0], entry_price_override=100.0, stop_bps=1500.0, target_r=1.0, leverage=10.0)
        self.assertEqual(out["exit_reason_model"], "liquidation")
        self.assertTrue(out["candidate_actual_liquidation"])

    def test_replay_long_path_stop_before_liquidation(self) -> None:
        ts = pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="min")
        ohlcv = pd.DataFrame({"timestamp": ts, "open": [100, 100, 100], "high": [101, 101, 101], "low": [98, 94, 94], "close": [100, 96, 95]})
        mark = pd.DataFrame({"timestamp": ts, "open": [100, 100, 100], "high": [101, 101, 101], "low": [99, 94, 94], "close": [100, 96, 95]})
        row = {"entry_ts": ts[0], "stop_distance_bps_1m": 500.0, "cost_bps_1m": 30.0}
        out = replay_long_path(row, ohlcv, mark, entry_ts=ts[0], entry_price_override=100.0, stop_bps=500.0, target_r=1.0, leverage=10.0)
        self.assertEqual(out["exit_reason_model"], "stop")
        self.assertFalse(out["candidate_actual_liquidation"])

    def test_stage_complete_requires_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage = "seal-guard"
            done_path(root, stage).parent.mkdir(parents=True)
            done_path(root, stage).write_text("done")
            self.assertFalse(stage_complete(root, stage))
            for p in required_outputs(root, stage):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
            self.assertTrue(stage_complete(root, stage))

    def test_constants_and_wrapper(self) -> None:
        self.assertEqual(CANDIDATE_ID, "D4__b4c9487fe82c")
        self.assertEqual(str(FINAL_HOLDOUT_START), "2026-01-01 00:00:00+00:00")
        self.assertTrue(Path("tools/run_qlmg_d4_survivability_redesign_tmux.sh").exists())


if __name__ == "__main__":
    unittest.main()
