#!/usr/bin/env python3
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as cfg
import indicators as ta
from indicators import resample_ohlcv
from shared_utils import load_parquet_data


def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean().rolling(period).mean()


def export_daily_intermediate(symbol: str, out_path: Path) -> None:
    ma_period = int(getattr(cfg, "REGIME_MA_PERIOD", 200))
    atr_period = int(getattr(cfg, "REGIME_ATR_PERIOD", 20))
    atr_mult = float(getattr(cfg, "REGIME_ATR_MULT", 2.0))

    user_start = getattr(cfg, "START_DATE", None)
    user_end = getattr(cfg, "END_DATE", None)
    slice_start = pd.to_datetime(user_start, utc=True) if user_start else None
    slice_end = pd.to_datetime(user_end, utc=True) if user_end else None

    warmup_start = user_start
    if slice_start is not None:
        warmup_days = int(2 * ma_period + atr_period + 30)
        warmup_start = (slice_start - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")

    df5 = load_parquet_data(
        symbol,
        start_date=warmup_start,
        end_date=user_end,
        drop_last_partial=True,
        columns=["open", "high", "low", "close", "volume"],
    )
    if df5 is None or df5.empty:
        raise RuntimeError(f"No 5m data for {symbol}")

    # Daily OHLCV (pandas defaults: label/closed not specified)
    daily_ohlc_full = df5.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    daily_full = pd.DataFrame(index=daily_ohlc_full.index)
    daily_full["close"] = daily_ohlc_full["close"].astype(float)

    if slice_start is not None:
        w0 = slice_start.floor("D")
    else:
        w0 = daily_full.index.min()
    if slice_end is not None:
        w1 = slice_end.floor("D")
    else:
        w1 = daily_full.index.max()

    daily = daily_full.loc[(daily_full.index >= w0) & (daily_full.index <= w1)].copy()
    if daily.empty:
        raise RuntimeError("Daily slice empty after applying START_DATE/END_DATE")

    # Trend intermediates
    tma_full = triangular_moving_average(daily_full["close"], ma_period)
    atr_full = ta.atr(daily_ohlc_full, length=atr_period).reindex(daily_full.index, method="ffill")
    upper_full = tma_full + atr_mult * atr_full
    lower_full = tma_full - atr_mult * atr_full

    trend_full = pd.Series(index=daily_full.index, dtype="object")
    trend_full[daily_full["close"] > upper_full] = "BULL"
    trend_full[daily_full["close"] < lower_full] = "BEAR"
    trend_full = trend_full.ffill().bfill()

    daily["tma"] = tma_full.reindex(daily.index).astype(float)
    daily["atr"] = atr_full.reindex(daily.index).astype(float)
    daily["upper"] = upper_full.reindex(daily.index).astype(float)
    daily["lower"] = lower_full.reindex(daily.index).astype(float)
    daily["trend_regime"] = trend_full.reindex(daily.index).astype(str)

    # Volatility intermediates (global fit; smoothed probs)
    ret = daily["close"].pct_change()
    daily["ret_pct"] = ret.astype(float)

    vol_prob_low = pd.Series(np.nan, index=daily.index, dtype=float)
    vol_regime = pd.Series("UNKNOWN", index=daily.index, dtype=object)
    prob0 = pd.Series(np.nan, index=daily.index, dtype=float)
    prob1 = pd.Series(np.nan, index=daily.index, dtype=float)
    low_idx = None

    r = ret.dropna()
    if not r.empty:
        model = sm.tsa.MarkovRegression(r, k_regimes=2, switching_variance=True, trend="c")
        results = model.fit(disp=False, maxiter=200)
        p0 = results.smoothed_marginal_probabilities[0].clip(0, 1)
        p1 = results.smoothed_marginal_probabilities[1].clip(0, 1)

        rr = r.reindex(p0.index)

        def wvar(p):
            w = p.values
            mu = np.sum(w * rr.values) / np.sum(w)
            return np.sum(w * (rr.values - mu) ** 2) / np.sum(w)

        v0 = wvar(p0)
        v1 = wvar(p1)
        low_idx = int(np.argmin([v0, v1]))

        low_prob = (p0 if low_idx == 0 else p1)
        vol_prob_low.loc[low_prob.index] = low_prob.values
        vol_regime.loc[low_prob.index] = np.where(low_prob > 0.5, "LOW_VOL", "HIGH_VOL")

        prob0.loc[p0.index] = p0.values
        prob1.loc[p1.index] = p1.values

    daily["vol_prob_low"] = vol_prob_low
    daily["vol_regime"] = vol_regime
    daily["smoothed_prob_state0"] = prob0
    daily["smoothed_prob_state1"] = prob1
    daily["low_var_state_idx"] = low_idx if low_idx is not None else np.nan

    daily["regime"] = (daily["trend_regime"].fillna("NA") + "_" + daily["vol_regime"].fillna("UNKNOWN")).astype(str)
    code_map = {"BEAR_HIGH_VOL": 0, "BEAR_LOW_VOL": 1, "BULL_HIGH_VOL": 2, "BULL_LOW_VOL": 3}
    daily["regime_code"] = daily["regime"].map(code_map).astype("Int64")

    out = daily.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)


def export_markov4h_intermediate(symbol: str, out_path: Path) -> None:
    alpha = float(getattr(cfg, "MARKOV4H_PROB_EWMA_ALPHA", 0.2))

    df5 = load_parquet_data(
        symbol,
        start_date=getattr(cfg, "START_DATE", None),
        end_date=getattr(cfg, "END_DATE", None),
        drop_last_partial=True,
        columns=["open", "high", "low", "close", "volume"],
    )
    if df5 is None or df5.empty:
        raise RuntimeError(f"No 5m data for {symbol}")

    df4 = resample_ohlcv(df5, getattr(cfg, "REGIME_TIMEFRAME", "4h"))  # pandas defaults
    close = df4["close"].astype(float)
    ret = np.log(close).diff()

    r = ret.dropna()
    if r.empty:
        raise RuntimeError("4h log returns empty")

    mod = sm.tsa.MarkovRegression(r, k_regimes=2, switching_variance=True, trend="c")
    res = mod.fit(disp=False, maxiter=200)

    fp0 = res.filtered_marginal_probabilities[0].clip(0, 1)
    fp1 = res.filtered_marginal_probabilities[1].clip(0, 1)

    rr0 = r.reindex(fp0.index)

    def wmean(p):
        w = p.values
        rv = rr0.values
        return np.sum(w * rv) / max(np.sum(w), 1e-12)

    means = [wmean(fp0), wmean(fp1)]
    up_idx = int(np.argmax(means))

    prob_up_raw = (fp0 if up_idx == 0 else fp1).copy()
    prob_up = prob_up_raw.ewm(alpha=alpha, adjust=False).mean().clip(0, 1)
    state_up = (prob_up > 0.5).astype(int)

    out = pd.DataFrame(
        {
            "close": close,
            "logret": ret,
            "filtered_prob_state0": fp0,
            "filtered_prob_state1": fp1,
            "up_state_idx": up_idx,
            "prob_up_raw": prob_up_raw,
            "prob_up_ewm": prob_up,
            "state_up": state_up,
        }
    )
    out.index.name = "timestamp"
    out = out.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    export_daily_intermediate(args.symbol, out_dir / f"{args.symbol}_daily_regime_intermediate.parquet")
    export_markov4h_intermediate(args.symbol, out_dir / f"{args.symbol}_markov4h_intermediate.parquet")
    print("WROTE", out_dir)


if __name__ == "__main__":
    main()
