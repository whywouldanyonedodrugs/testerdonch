from __future__ import annotations

import unittest
from decimal import Decimal

import pandas as pd

from tools.kraken_candle_volume_authority import (
    PROXY_FIELD,
    aggregate_execution_interval,
    assert_proxy_claim_boundary,
    daily_close_based_proxy,
    lagged_top_n_membership,
    validate_candle_interval,
    validate_semantic_versions,
)


def execution(uid: str, ts: int, quantity: str, symbol: str = "PF_XBTUSD") -> dict:
    return {"event": {"Execution": {"execution": {
        "uid": uid, "timestamp": ts, "quantity": quantity,
        "makerOrder": {"tradeable": symbol},
    }}}}


class CandleVolumeAuthorityTests(unittest.TestCase):
    def test_complete_paginated_exact_decimal_aggregation(self) -> None:
        pages = [
            {"elements": [execution("a", 1_000, "0.0001")]},
            {"elements": [execution("b", 2_000, "0.0002")]},
        ]
        total, rows = aggregate_execution_interval(
            pages, symbol="PF_XBTUSD", start_ms=0, end_ms=300_000,
        )
        self.assertEqual((total, rows), (Decimal("0.0003"), 2))
        audit = validate_candle_interval(
            [{"time": 0, "volume": "0.00030000"}], symbol="PF_XBTUSD",
            start_ms=0, execution_volume=total,
        )
        self.assertTrue(audit["exact_match"])
        self.assertEqual(audit["exact_difference"], "0.00000000")

    def test_duplicate_missing_boundary_and_wrong_symbol_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_execution_interval(
                [{"elements": [execution("a", 1, "1"), execution("a", 2, "1")]}],
                symbol="PF_XBTUSD", start_ms=0, end_ms=300_000,
            )
        with self.assertRaisesRegex(ValueError, "half-open"):
            aggregate_execution_interval(
                [{"elements": [execution("a", 300_000, "1")]}],
                symbol="PF_XBTUSD", start_ms=0, end_ms=300_000,
            )
        with self.assertRaisesRegex(ValueError, "wrong"):
            aggregate_execution_interval(
                [{"elements": [execution("a", 1, "1", "PF_ETHUSD")]}],
                symbol="PF_XBTUSD", start_ms=0, end_ms=300_000,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_candle_interval([], symbol="PF_XBTUSD", start_ms=0, execution_volume=Decimal("1"))

    def test_fractional_scaled_lot_and_semantic_change(self) -> None:
        total, _ = aggregate_execution_interval(
            [{"elements": [execution("a", 1, "0.01", "PF_AAVEUSD"), execution("b", 2, "0.02", "PF_AAVEUSD")]}],
            symbol="PF_AAVEUSD", start_ms=0, end_ms=300_000,
        )
        self.assertEqual(total, Decimal("0.03"))
        rows = pd.DataFrame([
            {"symbol": "PF_AAVEUSD", "snapshot_ts": "2023-01-01T00:00:00Z", "base_currency": "AAVE", "min_lot": "0.01", "source_sha256": "a"},
            {"symbol": "PF_AAVEUSD", "snapshot_ts": "2024-01-01T00:00:00Z", "base_currency": "AAVE", "min_lot": "0.1", "source_sha256": "b"},
        ])
        self.assertFalse(validate_semantic_versions(rows)["semantic_consistent"].any())

    def test_future_day_cannot_change_prior_rank(self) -> None:
        rows = []
        for day in pd.date_range("2023-01-01", periods=32, tz="UTC"):
            rows.extend([
                {"symbol": "PF_AUSD", "utc_day": day, PROXY_FIELD: 20.0},
                {"symbol": "PF_BUSD", "utc_day": day, PROXY_FIELD: 10.0},
            ])
        daily = pd.DataFrame(rows)
        before = lagged_top_n_membership(daily, top_n=1)
        future = pd.concat([daily, pd.DataFrame([
            {"symbol": "PF_BUSD", "utc_day": pd.Timestamp("2023-03-01", tz="UTC"), PROXY_FIELD: 1e12},
        ])], ignore_index=True)
        after = lagged_top_n_membership(future, top_n=1)
        cutoff = pd.Timestamp("2023-02-01", tz="UTC")
        left = before[before["utc_day"] <= cutoff].reset_index(drop=True)
        right = after[after["utc_day"] <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_daily_proxy_name_and_claim_boundary(self) -> None:
        bars = pd.DataFrame([
            {"symbol": "PF_AUSD", "source_open_ts": "2023-01-01T00:00:00Z", "close": 2, "volume": 3},
            {"symbol": "PF_AUSD", "source_open_ts": "2023-01-01T00:05:00Z", "close": 4, "volume": 5},
        ])
        daily = daily_close_based_proxy(bars)
        self.assertEqual(daily.loc[0, PROXY_FIELD], 26)
        contract = (
            f"{PROXY_FIELD} uses data through the prior UTC day. It is not exact quote volume "
            "and not capacity evidence."
        )
        assert_proxy_claim_boundary(daily.columns, contract)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            assert_proxy_claim_boundary([PROXY_FIELD, "quote_volume"], contract)


if __name__ == "__main__":
    unittest.main()
