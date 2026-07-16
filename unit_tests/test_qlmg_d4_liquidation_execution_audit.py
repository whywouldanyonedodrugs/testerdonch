from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.run_qlmg_d4_liquidation_execution_audit import (
    CANDIDATE_ID,
    FINAL_HOLDOUT_START,
    classify_liquidation_row,
    compare_reconstruction_metrics,
    dedupe_windows,
    done_path,
    estimate_storage_gb,
    liquidation_adverse_bps_for_leverage,
    make_null_window,
    required_outputs_for_stage,
    sample_matched_nulls,
    stage_complete,
    stop_distance_bps,
)
from tools.qlmg_regime_stack import validate_no_protected


class D4LiquidationExecutionAuditTests(unittest.TestCase):
    def test_protected_slice_rejected(self) -> None:
        df = pd.DataFrame({"decision_ts": [pd.Timestamp("2026-01-01T00:00:00Z")]})
        with self.assertRaises(RuntimeError):
            validate_no_protected(df, ["decision_ts"])

    def test_reconstruction_mismatch_fails_closed(self) -> None:
        prior = {"events": 10, "net_R": 1.0, "PF": 1.2, "liquidation_count": 2}
        recon = {"events": 9, "net_R": 1.0, "PF": 1.2, "liquidation_count": 2}
        with self.assertRaises(RuntimeError):
            compare_reconstruction_metrics(prior, recon)

    def test_liquidation_taxonomy_proxy_not_actual_mark(self) -> None:
        row = {"liquidation_flag_proxy_10x": True, "mark_path_status": "last_price_proxy"}
        self.assertEqual(classify_liquidation_row(row), "last_price_proxy_liquidation")
        row2 = {"liquidation_flag_proxy_10x": True, "mark_path_status": "missing"}
        self.assertEqual(classify_liquidation_row(row2), "missing_mark_proxy_only")
        row3 = {"liquidation_flag_proxy_10x": True, "mark_path_status": "last_price_proxy"}
        self.assertEqual(classify_liquidation_row(row3, has_one_minute_mark=True, mark_liquidated=False), "leverage_assumption_only")
        row4 = {"liquidation_flag_proxy_10x": True, "mark_path_status": "mark_available_ok"}
        self.assertEqual(classify_liquidation_row(row4), "five_minute_mark_liquidation")

    def test_stop_before_liquidation_classification(self) -> None:
        row = {"liquidation_flag_proxy_10x": True, "mark_path_status": "mark_available_ok"}
        self.assertEqual(classify_liquidation_row(row, has_one_minute_mark=True, mark_liquidated=True, stop_before_liq=True), "stop_would_trigger_before_liquidation")
        self.assertEqual(classify_liquidation_row(row, has_one_minute_mark=True, mark_liquidated=True, stop_before_liq=False), "liquidation_before_stop")

    def test_targeted_window_dedup_and_storage(self) -> None:
        df = pd.DataFrame({
            "symbol": ["AAAUSDT", "AAAUSDT"],
            "window_start": [pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:30:00Z")],
            "window_end": [pd.Timestamp("2025-01-01T03:00:00Z"), pd.Timestamp("2025-01-01T04:00:00Z")],
            "window_type": ["accepted", "matched_null"],
            "event_id": ["e1", "e2"],
        })
        out = dedupe_windows(df)
        self.assertEqual(len(out), 1)
        rows = estimate_storage_gb(out)
        self.assertTrue(any(r["dataset"] == "total_core" for r in rows))
        self.assertGreater(next(r for r in rows if r["dataset"] == "total_core")["estimated_rows"], 0)

    def test_null_window_before_holdout(self) -> None:
        row = {"event_id": "e", "symbol": "AAAUSDT", "decision_ts": pd.Timestamp("2025-12-20T00:00:00Z")}
        w = make_null_window(row, 0, 20260625)
        self.assertLess(pd.Timestamp(w["window_end"]), FINAL_HOLDOUT_START)

    def test_matched_null_sampling_deterministic(self) -> None:
        events = pd.DataFrame({
            "event_id": ["e1", "e2"],
            "symbol": ["A", "B"],
            "liquidity_tier": ["A", "B"],
            "decision_ts": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
        })
        pool = pd.DataFrame({
            "event_id": ["n1", "n2", "n3", "n4"],
            "symbol": ["A", "A", "B", "B"],
            "liquidity_tier": ["A", "A", "B", "B"],
            "decision_ts": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
        })
        a = sample_matched_nulls(events, pool, 2, 7)
        b = sample_matched_nulls(events, pool, 2, 7)
        self.assertEqual(a["event_id"].tolist(), b["event_id"].tolist())

    def test_leverage_threshold_and_stop_distance(self) -> None:
        self.assertAlmostEqual(liquidation_adverse_bps_for_leverage(10.0), 950.0)
        df = pd.DataFrame({"reference_risk_bps": [100.0]})
        self.assertEqual(float(stop_distance_bps(df).iloc[0]), 200.0)

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

    def test_tmux_wrapper_exists_and_candidate_constant(self) -> None:
        self.assertTrue(Path("tools/run_qlmg_d4_liquidation_execution_audit_tmux.sh").exists())
        self.assertEqual(CANDIDATE_ID, "D4__b4c9487fe82c")


if __name__ == "__main__":
    unittest.main()
