import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.donch_autopar_compare import (  # noqa: E402
    _build_failure_bucket_audit,
    _build_regime_attribution,
)


class TestJT012AutoparCompare(unittest.TestCase):
    def test_failure_bucket_assignment(self):
        merged = pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "ts_bucket": "2026-02-10T00:00:00Z",
                    "enter_live": 0,
                    "reason_live": "schema_fail",
                    "reason_raw_live": "schema_fail",
                    "schema_ok_live": False,
                    "scope_ok_live": True,
                    "meta_ok_live": False,
                    "strat_ok_live": False,
                    "_merge": "both",
                },
                {
                    "symbol": "ETHUSDT",
                    "ts_bucket": "2026-02-10T00:05:00Z",
                    "enter_live": 0,
                    "reason_live": "below_pstar",
                    "reason_raw_live": "below_pstar",
                    "schema_ok_live": True,
                    "scope_ok_live": True,
                    "meta_ok_live": False,
                    "strat_ok_live": True,
                    "_merge": "both",
                },
                {
                    "symbol": "SOLUSDT",
                    "ts_bucket": "2026-02-10T00:10:00Z",
                    "enter_live": 0,
                    "reason_live": "",
                    "reason_raw_live": "",
                    "schema_ok_live": True,
                    "scope_ok_live": True,
                    "meta_ok_live": True,
                    "strat_ok_live": True,
                    "_merge": "left_only",
                },
                {
                    "symbol": "XRPUSDT",
                    "ts_bucket": "2026-02-10T00:15:00Z",
                    "enter_live": 1,
                    "reason_live": "ok",
                    "reason_raw_live": "ok",
                    "schema_ok_live": True,
                    "scope_ok_live": True,
                    "meta_ok_live": True,
                    "strat_ok_live": True,
                    "_merge": "both",
                },
            ]
        )
        diag, monthly, rows = _build_failure_bucket_audit(merged)
        self.assertEqual(diag.get("status"), "ok")
        self.assertEqual(diag.get("skip_rows_live"), 3)
        counts = diag.get("canonical_bucket_counts") or {}
        self.assertEqual(int(counts.get("schema_fail", 0)), 1)
        self.assertEqual(int(counts.get("below_pstar", 0)), 1)
        self.assertEqual(int(counts.get("no_signals", 0)), 1)
        self.assertAlmostEqual(float(diag.get("assigned_rate_live")), 1.0, places=9)
        self.assertFalse(monthly.empty)
        self.assertEqual(int(len(rows)), 3)

    def test_regime_attribution_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trades.csv"
            pd.DataFrame(
                [
                    {
                        "exit_ts": "2026-01-10T00:00:00Z",
                        "pnl_R": 1.0,
                        "risk_on": 1,
                        "eth_macd_hist_4h": 0.2,
                        "btc_vol_regime_level": 0.6,
                    },
                    {
                        "exit_ts": "2026-01-20T00:00:00Z",
                        "pnl_R": -0.5,
                        "risk_on": 0,
                        "eth_macd_hist_4h": -0.1,
                        "btc_vol_regime_level": 0.9,
                    },
                    {
                        "exit_ts": "2026-02-02T00:00:00Z",
                        "pnl_R": 0.3,
                        "risk_on": 0,
                        "eth_macd_hist_4h": -0.05,
                        "btc_vol_regime_level": 0.85,
                    },
                ]
            ).to_csv(p, index=False)

            diag, regime_rows, monthly = _build_regime_attribution(p)
            self.assertEqual(diag.get("status"), "ok")
            self.assertFalse(regime_rows.empty)
            dims = set(regime_rows["dimension"].astype(str).tolist())
            self.assertTrue({"risk_on", "eth_hist_sign", "btc_vol_regime"}.issubset(dims))
            self.assertEqual(int(diag.get("monthly_rows")), 2)
            self.assertFalse(monthly.empty)


if __name__ == "__main__":
    unittest.main()
