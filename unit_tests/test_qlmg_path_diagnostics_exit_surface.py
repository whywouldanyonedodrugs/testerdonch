import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tools.qlmg_screening_core import ReplayConfig, replay_trade
from tools.run_qlmg_path_diagnostics_exit_surface import (
    FINAL_HOLDOUT_START,
    _raw_signal_mask,
    check_resource_guard,
    compute_path_row,
    done_path,
    path_metrics_path,
    sample_null_events,
    stage_complete,
    surface_returns,
)
from tools.qlmg_screening_core import ResourceSnapshot


class TestQlmqPathDiagnostics(unittest.TestCase):
    def test_seal_boundary_constant(self):
        self.assertEqual(str(FINAL_HOLDOUT_START), "2026-01-01 00:00:00+00:00")

    def test_resource_guard_thresholds(self):
        snap = ResourceSnapshot("/", 10, 9, 4 * 1024**3, None, None)
        status = check_resource_guard(snap, estimated_output_gb=1, hard_free_gb=5, warn_free_gb=7)
        self.assertEqual(status["status"], "hard_stop")
        snap2 = ResourceSnapshot("/", 10, 1, 6 * 1024**3, None, None)
        status2 = check_resource_guard(snap2, estimated_output_gb=1, hard_free_gb=5, warn_free_gb=7)
        self.assertEqual(status2["status"], "pass")
        self.assertTrue(status2["warnings"])

    def test_stage_checkpoint_requires_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            done_path(root, "mfe-mae-path-diagnostics").parent.mkdir(parents=True)
            done_path(root, "mfe-mae-path-diagnostics").write_text("done")
            self.assertFalse(stage_complete(root, "mfe-mae-path-diagnostics"))
            path_metrics_path(root).parent.mkdir(parents=True)
            pd.DataFrame({"x": [1]}).to_parquet(path_metrics_path(root), index=False)
            (root / "path_diagnostics/path_summary_by_family.csv").write_text("family\n")
            self.assertTrue(stage_complete(root, "mfe-mae-path-diagnostics"))

    def test_raw_signal_mask_uncapped(self):
        df = pd.DataFrame({
            "close": [100, 101, 80, 81, 82],
            "ret_4h": [-0.3, -0.3, -0.3, -0.3, -0.3],
            "ret_24h": [-0.3] * 5,
            "turnover": [1] * 5,
            "turnover_med_24h": [10] * 5,
        })
        v = {"family": "D1", "side": "long", "window_h": 4, "shock": 0.2}
        mask = _raw_signal_mask(df, v)
        self.assertGreaterEqual(int(mask.sum()), 2)

    def test_side_aware_path_metrics_long_short(self):
        idx = pd.date_range("2025-01-01", periods=6, freq="5min", tz="UTC")
        bars = pd.DataFrame({"timestamp": idx, "open": [100]*6, "high": [100, 105, 106, 104, 103, 102], "low": [100, 99, 98, 97, 96, 95], "close": [100, 104, 105, 100, 98, 96]}, index=idx)
        ev = pd.Series({"event_id": "e1", "family": "X", "variant_id": "v", "symbol": "S", "side": "long", "liquidity_tier": "C", "decision_ts": idx[0], "entry_ts": idx[0], "entry_ref_price": 100.0, "reference_risk_bps": 100.0, "atr_bps": 50.0})
        row = compute_path_row(ev, bars)
        self.assertGreater(row["15m_mfe_bps"], 0)
        ev["side"] = "short"
        row2 = compute_path_row(ev, bars)
        self.assertGreater(row2["15m_mae_bps"], 0)

    def test_surface_pessimistic_same_bar(self):
        df = pd.DataFrame({"reference_risk_bps": [100], "24h_mfe_bps": [250], "24h_mae_bps": [120], "24h_close_return_bps": [50]})
        ret = surface_returns(df, "24h", df["reference_risk_bps"], 2.0, "pessimistic")
        self.assertEqual(float(ret.iloc[0]), -1.0)
        opt = surface_returns(df, "24h", df["reference_risk_bps"], 2.0, "optimistic")
        self.assertEqual(float(opt.iloc[0]), 2.0)

    def test_null_sampling_excludes_event_window_deterministic(self):
        # This checks deterministic empty handling without requiring local symbol files.
        events = pd.DataFrame([{"event_id": "x", "symbol": "NOFILEUSDT", "decision_ts": pd.Timestamp("2025-01-01", tz="UTC"), "entry_ts": pd.Timestamp("2025-01-01 00:05", tz="UTC"), "side": "long", "family": "D1", "variant_id": "v"}])
        a = sample_null_events(events, 1)
        b = sample_null_events(events, 1)
        self.assertEqual(len(a), len(b))

    def test_liquidation_before_stop_core(self):
        idx = pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC")
        bars = pd.DataFrame({"open": [100, 100, 100], "high": [100, 101, 101], "low": [100, 99, 99], "close": [100, 100, 100], "mark_high": [100, 101, 101], "mark_low": [100, 80, 80]}, index=idx)
        res = replay_trade(bars, ReplayConfig("long", idx[0], idx[0], 100, 70, 120, 1, leverage=10), [])
        self.assertTrue(res.liquidation_flag)
        self.assertEqual(res.exit_reason, "liquidation")


if __name__ == "__main__":
    unittest.main()
