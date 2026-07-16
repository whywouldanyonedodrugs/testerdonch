from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.run_qlmg_targeted_1m_data_pilot import (
    FINAL_HOLDOUT_START,
    build_windows_from_path_metrics,
    dedupe_windows,
    estimate_storage,
    normalize_kline_rows,
    normalize_funding_rows,
    normalize_oi_rows,
    stage_complete,
    done_path,
    validate_no_protected,
    window_hash,
)


class Targeted1mPilotTests(unittest.TestCase):
    def test_protected_timestamp_rejected(self) -> None:
        df = pd.DataFrame({"window_end": [pd.Timestamp("2026-01-01", tz="UTC")]})
        with self.assertRaises(RuntimeError):
            validate_no_protected(df, ["window_end"])

    def test_window_extraction_drops_protected(self) -> None:
        df = pd.DataFrame({
            "family": ["D3", "E1"],
            "event_id": ["a", "b"],
            "symbol": ["AAAUSDT", "BBBUSDT"],
            "decision_ts": [pd.Timestamp("2025-12-30 01:00", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")],
            "24h_mfe_bps": [1000.0, 1000.0],
            "24h_mae_bps": [100.0, 100.0],
            "24h_pos1R_before_neg1R": [True, True],
            "24h_liquidation_10x": [False, False],
            "oi_chg_24h": [-0.1, -0.1],
            "funding_rate": [0.0, 0.0],
        })
        out = build_windows_from_path_metrics(df)
        self.assertTrue((pd.to_datetime(out["window_end"], utc=True) < FINAL_HOLDOUT_START).all())

    def test_dedup_merges_overlap_same_symbol(self) -> None:
        df = pd.DataFrame({
            "family": ["D3", "E1"],
            "event_id": ["a", "b"],
            "symbol": ["AAAUSDT", "AAAUSDT"],
            "window_start": [pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-01 00:30", tz="UTC")],
            "window_end": [pd.Timestamp("2025-01-01 02:00", tz="UTC"), pd.Timestamp("2025-01-01 03:00", tz="UTC")],
        })
        out = dedupe_windows(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["source_event_count"]), 2)

    def test_storage_estimate_has_total_core(self) -> None:
        df = pd.DataFrame({"window_start": [pd.Timestamp("2025-01-01", tz="UTC")], "window_end": [pd.Timestamp("2025-01-01 01:00", tz="UTC")]})
        rows = estimate_storage(df)
        self.assertTrue(any(r["dataset"] == "total_core" for r in rows))
        self.assertGreater(next(r for r in rows if r["dataset"] == "total_core")["estimated_rows"], 0)

    def test_kline_normalization(self) -> None:
        rows = [["1735689600000", "1", "2", "0.5", "1.5", "10", "15"]]
        df = normalize_kline_rows(rows, ["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        self.assertEqual(len(df), 1)
        self.assertEqual(str(df.iloc[0]["timestamp"]), "2025-01-01 00:00:00+00:00")
        self.assertEqual(float(df.iloc[0]["close"]), 1.5)

    def test_oi_and_funding_normalization(self) -> None:
        oi = normalize_oi_rows([{"timestamp": "1735689600000", "openInterest": "123"}])
        fu = normalize_funding_rows([{"fundingRateTimestamp": "1735689600000", "fundingRate": "0.0001"}])
        self.assertEqual(float(oi.iloc[0]["open_interest"]), 123.0)
        self.assertAlmostEqual(float(fu.iloc[0]["funding_rate"]), 0.0001)

    def test_stage_complete_requires_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done_path(root, "pilot-window-selection").parent.mkdir(parents=True)
            done_path(root, "pilot-window-selection").write_text("done")
            self.assertFalse(stage_complete(root, "pilot-window-selection"))
            (root / "pilot").mkdir()
            (root / "pilot/pilot_windows.csv").write_text("x\n")
            (root / "pilot/pilot_selection_report.md").write_text("# x\n")
            self.assertTrue(stage_complete(root, "pilot-window-selection"))

    def test_window_hash_deterministic(self) -> None:
        row = {"window_type": "event", "family": "D3", "event_id": "a", "symbol": "AAA", "window_start": "s", "window_end": "e"}
        self.assertEqual(window_hash(row), window_hash(row))

    def test_tmux_wrapper_exists(self) -> None:
        self.assertTrue(Path("tools/run_qlmg_targeted_1m_data_pilot_tmux.sh").exists())


if __name__ == "__main__":
    unittest.main()
