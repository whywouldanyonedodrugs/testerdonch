# regime_detector.py
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Optional

import config as cfg
import shared_utils
import indicators as ta

from typing import Optional, Tuple
from shared_utils import load_parquet_data
from indicators import resample_ohlcv


def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    # TMA = SMA(SMA(price, period), period)
    return series.rolling(period).mean().rolling(period).mean()

@dataclass
class DailyRegimeConfig:
    benchmark: str = getattr(cfg, "REGIME_BENCHMARK_SYMBOL", getattr(cfg, "REGIME_ASSET", "ETHUSDT"))
    ma_period: int = getattr(cfg, "REGIME_MA_PERIOD", 200)
    atr_period: int = getattr(cfg, "REGIME_ATR_PERIOD", 20)
    atr_mult: float = getattr(cfg, "REGIME_ATR_MULT", 2.0)
    save_path: Optional[str] = None  # e.g., str(cfg.PROJECT_ROOT / "regime_data.parquet")

def compute_markov_regime_4h(
    asset: Optional[str] = None,
    timeframe: str = "4h",
    prob_ewm_alpha: Optional[float] = None,
) -> pd.DataFrame:
    """
    Two-state Markov regime on 4h close-to-close returns of `asset` (default: cfg.REGIME_ASSET, e.g., ETHUSDT).
    - Uses statsmodels.MarkovRegression with switching mean & variance.
    - Returns *filtered* (past-only) probabilities to avoid look-ahead.
    Columns: ['ret','state_up','prob_up'], indexed by 4h timestamps.
    """
    sym = asset or getattr(cfg, "REGIME_ASSET", "ETHUSDT")
    alpha = float(prob_ewm_alpha if prob_ewm_alpha is not None else getattr(cfg, "MARKOV4H_PROB_EWMA_ALPHA", 0.2))

    # Load 5m, build 4h bars
    df5 = load_parquet_data(
        sym,
        start_date=getattr(cfg, "START_DATE", None),
        end_date=getattr(cfg, "END_DATE", None),
        drop_last_partial=True,
        columns=["open","high","low","close","volume"],
    )
    if df5.empty:
        return pd.DataFrame(columns=["ret","state_up","prob_up"])

    df4 = resample_ohlcv(df5, timeframe)
    close = df4["close"].astype(float)
    # use log returns; add a tiny epsilon to avoid log(0) (shouldn't happen on close)
    ret = np.log(close).diff().dropna()
    if ret.empty:
        return pd.DataFrame(columns=["ret","state_up","prob_up"])

    # Markov regression with switching mean & variance, constant trend
    try:
        mod = sm.tsa.MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
        res = mod.fit(disp=False, maxiter=200)
        # FILTERED probs (past-only)
        filt_probs = [res.filtered_marginal_probabilities[i] for i in range(2)]
        # Identify "UP" state as the one with higher filtered mean return
        means = []
        for p in filt_probs:
            w = p.values
            r = ret.reindex(p.index).values
            mu = np.sum(w * r) / max(np.sum(w), 1e-12)
            means.append(mu)
        up_idx = int(np.argmax(means))
        prob_up = filt_probs[up_idx].clip(0, 1)
        # Light smoothing (helps stability)
        prob_up = prob_up.ewm(alpha=alpha, adjust=False).mean().clip(0, 1)
        state_up = (prob_up > 0.5).astype(int)

        out = pd.DataFrame({"ret": ret, "state_up": state_up, "prob_up": prob_up})
        out.index.name = "timestamp"
        return out
    except Exception as e:
        print(f"[regime_detector] 4h Markov model failed: {e}")
        return pd.DataFrame(columns=["ret","state_up","prob_up"])

def compute_daily_combined_regime(conf: DailyRegimeConfig | None = None) -> pd.DataFrame:
    """
    Build daily combined regime (trend via TMA+Keltner; volatility via 2-state Markov switching on daily returns).
    Returns a DataFrame indexed by daily timestamps with:
      ['trend_regime','vol_regime','vol_prob_low','regime','regime_code'].
    """
    c = conf or DailyRegimeConfig(save_path=str((getattr(cfg, "PROJECT_ROOT", None) or ".") / "regime_data.parquet"))

    # 1) Load 5m (or finer) benchmark and resample to daily
    df5 = shared_utils.load_parquet_data(c.benchmark,
                                         start_date=getattr(cfg, "START_DATE", None),
                                         end_date=getattr(cfg, "END_DATE", None),
                                         drop_last_partial=True,
                                         columns=["open","high","low","close","volume"])
    if df5.empty:
        return pd.DataFrame(columns=["trend_regime","vol_regime","vol_prob_low","regime","regime_code"])

    daily_ohlc = df5.resample("1D").agg({"open":"first", "high":"max", "low":"min", "close":"last"}).dropna()
    daily = pd.DataFrame(index=daily_ohlc.index)
    daily["close"] = daily_ohlc["close"]

    # Daily returns for MSM
    ret = daily["close"].pct_change().dropna()
    daily["vol_regime"] = "UNKNOWN"
    vol_prob_low = pd.Series(np.nan, index=daily.index, dtype=float)

    # 2) Markov switching model on daily returns (2 regimes, switching variance)
    try:
        model = sm.tsa.MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
        results = model.fit(disp=False, maxiter=200)
        # Smoothed probabilities per regime (pandas Series indexed by ret.index)
        probs = [results.smoothed_marginal_probabilities[i] for i in range(2)]

        # Identify low-vol regime by weighted variance
        r = ret.reindex(probs[0].index)
        var_est = []
        for p in probs:
            w = p.values
            mu = np.sum(w * r.values) / np.sum(w)
            var = np.sum(w * (r.values - mu)**2) / np.sum(w)
            var_est.append(var)
        low_idx = int(np.argmin(var_est))

        low_prob = probs[low_idx].clip(0, 1)
        # Classify low/high by 0.5 threshold; keep soft prob
        vol_reg = np.where(low_prob > 0.5, "LOW_VOL", "HIGH_VOL")
        daily.loc[ret.index, "vol_regime"] = vol_reg
        vol_prob_low.loc[ret.index] = low_prob.values
    except Exception as e:
        print(f"[regime_detector] Markov model failed: {e}. Using UNKNOWN vol_regime.")

    daily["vol_prob_low"] = vol_prob_low

    # 3) Trend via TMA + Keltner Channel
    tma = triangular_moving_average(daily["close"], int(c.ma_period))
    atr_d = ta.atr(daily_ohlc, length=int(c.atr_period))
    atr_d = atr_d.reindex(daily.index, method="ffill")
    upper = tma + c.atr_mult * atr_d
    lower = tma - c.atr_mult * atr_d

    trend = pd.Series(index=daily.index, dtype="object")
    trend[daily["close"] > upper] = "BULL"
    trend[daily["close"] < lower] = "BEAR"
    trend = trend.ffill().bfill()
    daily["trend_regime"] = trend

    # 4) Combine + numeric code
    daily["regime"] = (daily["trend_regime"].fillna("NA") + "_" + daily["vol_regime"].fillna("UNKNOWN")).astype(str)
    code_map = {"BEAR_HIGH_VOL": 0, "BEAR_LOW_VOL": 1, "BULL_HIGH_VOL": 2, "BULL_LOW_VOL": 3}
    daily["regime_code"] = daily["regime"].map(code_map).astype("Int64")

    out = daily[["trend_regime","vol_regime","vol_prob_low","regime","regime_code"]].copy()
    out.index.name = "timestamp"

    # Optional save
    if c.save_path:
        try:
            out.to_parquet(c.save_path)
        except Exception as e:
            print(f"[regime_detector] save failed: {e}")

    return out

# CLI (optional)
if __name__ == "__main__":
    df = compute_daily_combined_regime()
    print(df.tail(10))
