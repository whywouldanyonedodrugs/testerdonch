from __future__ import annotations

from collections import deque
import unittest

import numpy as np
import pandas as pd

from backtester import Backtester
from feature_registry import ACTIVE_FEATURE_REGISTRY, registry_feature_names, registry_rows_by_test_type
from indicators import donchian_upper_days_no_lookahead
from live.feature_builder import FeatureBuilder
from live.oi_funding import compute_oi_funding_feature_panel
from live.regime_features import drop_incomplete_last_bar


def _mk_df5(start: str = "2025-01-01 00:00:00+00:00", days: int = 80) -> pd.DataFrame:
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz="UTC")
    base = 100 + np.linspace(0, 25, len(idx)) + 2.5 * np.sin(np.arange(len(idx)) / 47.0)
    close = pd.Series(base, index=idx)
    out = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]).to_numpy(),
            "high": (close + 0.75).to_numpy(),
            "low": (close - 0.75).to_numpy(),
            "close": close.to_numpy(),
            "volume": (1000 + 100 * np.cos(np.arange(len(idx)) / 23.0)).astype(float),
        },
        index=idx,
    )
    out.index.name = "timestamp"
    return out


class TestFeatureLeakageContracts(unittest.TestCase):
    def test_feature_registry_has_unique_names_and_required_buckets(self) -> None:
        names = registry_feature_names()
        self.assertEqual(len(names), len(set(names)))

        buckets = registry_rows_by_test_type()
        for name in (
            "decision_bar_close",
            "htf_last_closed",
            "daily_snapshot",
            "truncation_equivalence",
            "exogenous_publish",
            "derived_from_causal",
            "trade_history",
            "same_timestamp_cross_sectional",
        ):
            self.assertIn(name, buckets)
            self.assertGreater(len(buckets[name]), 0)

        self.assertGreaterEqual(len(ACTIVE_FEATURE_REGISTRY), 80)

    def test_entry_quality_truncation_equivalence(self) -> None:
        df5 = _mk_df5(days=90)
        fb = FeatureBuilder(
            {
                "ATR_LEN": 14,
                "RSI_LEN": 14,
                "ADX_LEN": 14,
                "VOL_LOOKBACK_DAYS": 30,
                "DON_N_DAYS": 20,
                "PULLBACK_WINDOW_BARS": 12,
            }
        )
        fields = [
            "atr_1h",
            "rsi_1h",
            "adx_1h",
            "vol_mult",
            "atr_pct",
            "days_since_prev_break",
            "consolidation_range_atr",
            "prior_1d_ret",
            "rv_3d",
            "don_break_level",
            "don_dist_atr",
        ]

        for ts in df5.index[70 * 288 :: 173][:25]:
            full = fb.compute_entry_quality_features(df5, ts)
            trunc = fb.compute_entry_quality_features(df5.loc[:ts].copy(), ts)
            for field in fields:
                a = float(full.get(field, np.nan))
                b = float(trunc.get(field, np.nan))
                if np.isnan(a) and np.isnan(b):
                    continue
                self.assertAlmostEqual(a, b, places=10, msg=f"{field} mismatch at {ts}")

    def test_daily_completed_day_contract(self) -> None:
        idx = pd.date_range("2025-02-01 00:00:00+00:00", periods=4 * 288, freq="5min", tz="UTC")
        highs = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        don = pd.Series(donchian_upper_days_no_lookahead(highs, 1), index=idx)

        ts = pd.Timestamp("2025-02-03 12:00:00+00:00")
        day2 = highs.loc["2025-02-02 00:00:00+00:00":"2025-02-02 23:55:00+00:00"].max()
        day3_partial = highs.loc["2025-02-03 00:00:00+00:00":ts].max()

        self.assertEqual(float(don.loc[ts]), float(day2))
        self.assertNotEqual(float(don.loc[ts]), float(day3_partial))

    def test_live_daily_snapshot_drops_incomplete_same_day_bar(self) -> None:
        idx = pd.date_range("2025-02-01 00:00:00+00:00", periods=5, freq="1D", tz="UTC")
        df_daily = pd.DataFrame(
            {
                "open": [10, 11, 12, 13, 14],
                "high": [11, 12, 13, 14, 99],
                "low": [9, 10, 11, 12, 13],
                "close": [10.5, 11.5, 12.5, 13.5, 98.0],
            },
            index=idx,
        )
        trimmed = drop_incomplete_last_bar(df_daily, "1d", pd.Timestamp("2025-02-05 12:00:00+00:00"))
        self.assertEqual(trimmed.index.max(), pd.Timestamp("2025-02-04 00:00:00+00:00"))
        self.assertNotIn(pd.Timestamp("2025-02-05 00:00:00+00:00"), trimmed.index)

    def test_oi_funding_publish_lag_and_truncation_equivalence(self) -> None:
        df5 = _mk_df5(days=12)
        oi = pd.Series(10_000 + np.arange(len(df5), dtype=float), index=df5.index)
        fr_idx = pd.to_datetime(
            [
                "2025-01-02 00:00:00+00:00",
                "2025-01-02 08:00:00+00:00",
                "2025-01-02 16:00:00+00:00",
            ],
            utc=True,
        )
        fr = pd.Series([0.001, 0.002, -0.001], index=fr_idx, dtype=float)

        panel = compute_oi_funding_feature_panel(df5, oi, fr)
        self.assertTrue(np.isnan(float(panel.loc[pd.Timestamp("2025-01-02 00:00:00+00:00"), "funding_rate"])))
        self.assertAlmostEqual(float(panel.loc[pd.Timestamp("2025-01-02 00:05:00+00:00"), "funding_rate"]), 0.001, places=12)

        fields = [
            "oi_level",
            "oi_notional_est",
            "oi_pct_1h",
            "oi_pct_4h",
            "oi_pct_1d",
            "oi_z_7d",
            "oi_chg_norm_vol_1h",
            "oi_price_div_1h",
            "funding_rate",
            "funding_abs",
            "funding_z_7d",
            "funding_rollsum_3d",
            "funding_oi_div",
            "est_leverage",
        ]
        for ts in df5.index[3 * 288 :: 111][:20]:
            full_row = panel.loc[ts]
            trunc_panel = compute_oi_funding_feature_panel(df5.loc[:ts].copy(), oi.loc[:ts], fr.loc[:ts])
            trunc_row = trunc_panel.iloc[-1]
            for field in fields:
                a = float(full_row[field])
                b = float(trunc_row[field])
                if np.isnan(a) and np.isnan(b):
                    continue
                self.assertAlmostEqual(a, b, places=10, msg=f"{field} mismatch at {ts}")

    def test_recent_winrate_uses_prior_closed_trades_only(self) -> None:
        dummy = type("Dummy", (), {})()
        dummy._meta_win_hist = deque(maxlen=50)
        dummy._meta_ewm_win = None

        for pnl_r in [1.0, -1.0, 2.0]:
            snap = Backtester._meta_recent_winrate_features(dummy)
            hist = list(dummy._meta_win_hist)
            if hist:
                self.assertAlmostEqual(float(snap["recent_winrate_20"]), float(np.mean(hist[-20:])), places=12)
            else:
                self.assertTrue(np.isnan(float(snap["recent_winrate_20"])))
            Backtester._meta_update_winrate(dummy, pnl_r)

        snap_after = Backtester._meta_recent_winrate_features(dummy)
        self.assertAlmostEqual(float(snap_after["recent_winrate_20"]), 2.0 / 3.0, places=12)

    def test_same_timestamp_cross_sectional_aggregation_is_local(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-03-01 00:00:00+00:00",
                        "2025-03-01 00:00:00+00:00",
                        "2025-03-01 00:05:00+00:00",
                        "2025-03-01 00:05:00+00:00",
                    ],
                    utc=True,
                ),
                "asset_macd_hist_4h": [1.0, -1.0, 2.0, 3.0],
            }
        )
        df["_asset4h_pos"] = (pd.to_numeric(df["asset_macd_hist_4h"], errors="coerce") > 0).astype(float)
        df["asset4h_share_at_timestamp"] = df.groupby("timestamp")["_asset4h_pos"].transform("mean")
        self.assertEqual(float(df.loc[0, "asset4h_share_at_timestamp"]), 0.5)
        self.assertEqual(float(df.loc[2, "asset4h_share_at_timestamp"]), 1.0)


if __name__ == "__main__":
    unittest.main()
