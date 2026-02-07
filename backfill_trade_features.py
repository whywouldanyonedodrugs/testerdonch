from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import config as cfg
from shared_utils import load_parquet_data
from indicators import resample_ohlcv, atr


def _ensure_utc(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, utc=True, errors="coerce")
    return out


def _compute_est_leverage(df5: pd.DataFrame) -> pd.Series:
    """
    df5: 5m OHLCV(+open_interest) with DatetimeIndex
    Returns est_leverage indexed like df5
    """
    if "open_interest" not in df5.columns:
        return pd.Series(index=df5.index, data=np.nan, name="est_leverage", dtype="float64")

    oi = pd.to_numeric(df5["open_interest"], errors="coerce")
    close = pd.to_numeric(df5["close"], errors="coerce")
    oi_notional = oi * close

    WIN_1H = 12
    WIN_1D = 288
    notional_24h = oi_notional.rolling(WIN_1D, min_periods=WIN_1H).mean()

    est = oi_notional / (notional_24h + 1e-9)
    est.name = "est_leverage"
    return est


def _compute_daily_regime_and_trend(df5: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Returns a DAILY dataframe with columns:
      {prefix}_vol_regime_level, {prefix}_trend_slope
    indexed by day timestamp (UTC).
    """
    ohlc = df5[["open", "high", "low", "close", "volume"]].copy()
    daily = resample_ohlcv(ohlc, "1D")
    if daily.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))

    atr1d = atr(daily, 20)
    atr_pct = atr1d / daily["close"].replace(0, np.nan)

    # trailing median to avoid look-ahead
    base = atr_pct.rolling(365, min_periods=60).median()
    vol_regime = atr_pct / (base + 1e-12)

    ma20 = daily["close"].rolling(20).mean()
    ma50 = daily["close"].rolling(50).mean()
    trend_slope = (ma20 - ma50).diff()

    out = pd.DataFrame(
        {
            f"{prefix}_vol_regime_level": vol_regime,
            f"{prefix}_trend_slope": trend_slope,
        },
        index=daily.index,
    )
    out.index.name = "timestamp"
    return out


def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_index: bool = True) -> pd.DataFrame:
    """
    Merge right (time-indexed) onto left (has a timestamp column) using backward asof.
    """
    l = left.sort_values(left_on).copy()
    if right_index:
        r = right.sort_index().copy()
        r = r.reset_index().rename(columns={"timestamp": "_ts"})
        out = pd.merge_asof(l, r, left_on=left_on, right_on="_ts", direction="backward")
        out = out.drop(columns=["_ts"])
        return out
    else:
        raise ValueError("right_index must be True in this project")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=str, default="results/trades.csv")
    ap.add_argument("--out", type=str, default="results/trades_enriched.csv")
    args = ap.parse_args()

    trades_path = Path(args.trades)
    out_path = Path(args.out)

    trades = pd.read_csv(trades_path)
    if "entry_ts" not in trades.columns or "symbol" not in trades.columns:
        raise RuntimeError("Expected trades.csv to contain columns: entry_ts, symbol")

    trades["entry_ts"] = _ensure_utc(trades["entry_ts"])
    trades = trades.dropna(subset=["entry_ts"])

    # --- BTC/ETH regime + trend (OHLCV-only) ---
    def load_5m(sym: str) -> pd.DataFrame:
        df = load_parquet_data(
            sym,
            start_date=cfg.START_DATE,
            end_date=cfg.END_DATE,
            drop_last_partial=True,
            columns=["open", "high", "low", "close", "volume", "open_interest", "funding_rate"],
        )
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index.name = "timestamp"
        df = df.sort_index()
        return df

    btc5 = load_5m("BTCUSDT")
    eth5 = load_5m("ETHUSDT")

    btc_daily = _compute_daily_regime_and_trend(btc5, "btc")
    eth_daily = _compute_daily_regime_and_trend(eth5, "eth")

    trades2 = _merge_asof(trades, btc_daily, left_on="entry_ts")
    trades2 = _merge_asof(trades2, eth_daily, left_on="entry_ts")

    # --- Per-symbol est_leverage (OI-dependent) ---
    # Do it per symbol to keep memory bounded
    enriched_parts = []
    for sym, grp in trades2.groupby("symbol", sort=False):
        grp = grp.copy()

        try:
            df5 = load_5m(sym)
        except Exception as e:
            # no data -> keep NaNs
            grp["est_leverage"] = np.nan
            enriched_parts.append(grp)
            continue

        est = _compute_est_leverage(df5)
        feat = pd.DataFrame({"est_leverage": est}, index=df5.index)
        feat.index.name = "timestamp"

        grp = _merge_asof(grp, feat, left_on="entry_ts")

        enriched_parts.append(grp)

    out_df = pd.concat(enriched_parts, axis=0, ignore_index=True)

    # If original columns exist, only fill missing (don’t overwrite existing values)
    for col in ["est_leverage", "btc_vol_regime_level", "btc_trend_slope", "eth_vol_regime_level", "eth_trend_slope"]:
        if col in out_df.columns and col in trades.columns:
            out_df[col] = out_df[col].where(trades[col].isna(), trades[col])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[backfill] wrote: {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
