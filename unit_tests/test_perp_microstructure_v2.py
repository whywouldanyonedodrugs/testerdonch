from __future__ import annotations

import unittest

import pandas as pd

from tools.perp_microstructure_v2 import aggregate_orderbook_snapshot_features, aggregate_recent_trade_features, parse_orderbook_response, parse_recent_trade_response


class PerpMicrostructureV2Tests(unittest.TestCase):
    def test_parse_orderbook_and_aggregate(self) -> None:
        resp = {"result": {"s": "AAAUSDT", "b": [["99", "10"], ["98", "5"]], "a": [["101", "7"], ["102", "3"]], "ts": "1735689600000", "cts": "1735689600000", "u": 1, "seq": "2"}}
        raw = parse_orderbook_response(resp, observed_available_ts=pd.Timestamp("2025-01-01 00:04:30Z"))
        self.assertEqual(len(raw), 4)
        silver = aggregate_orderbook_snapshot_features(raw, max_age=pd.Timedelta(minutes=5))
        self.assertEqual(len(silver), 1)
        self.assertAlmostEqual(float(silver.loc[0, "spread_pct"]), 0.02)
        self.assertAlmostEqual(float(silver.loc[0, "top5_depth_imbalance"]), (15 - 10) / 25)

    def test_parse_trades_uses_observed_before_cutoff(self) -> None:
        resp = {"result": {"list": [
            {"execId": "1", "symbol": "AAAUSDT", "price": "100", "size": "2", "side": "Buy", "time": "1735689660000", "isBlockTrade": False, "isRPITrade": False, "seq": "1"},
            {"execId": "2", "symbol": "AAAUSDT", "price": "100", "size": "1", "side": "Sell", "time": "1735689720000", "isBlockTrade": False, "isRPITrade": False, "seq": "2"},
        ]}}
        raw = parse_recent_trade_response(resp, observed_available_ts=pd.Timestamp("2025-01-01 00:05:00Z"))
        silver = aggregate_recent_trade_features(raw)
        self.assertEqual(len(silver), 1)
        self.assertEqual(int(silver.loc[0, "trade_count"]), 2)
        self.assertAlmostEqual(float(silver.loc[0, "trade_size_imbalance"]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
