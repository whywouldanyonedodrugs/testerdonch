# =============================================
# indicators.py – TA‑Lib / pandas_ta wrapper
# =============================================
"""Compute technical indicators.

• Uses TA‑Lib if installed (fast C bindings).
• Falls back to pandas_ta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated.*")

# ----------------------------------------------------------------------
# NumPy 2 compatibility patch for older pandas_ta versions
# ----------------------------------------------------------------------
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore

# Prefer TA‑Lib; else pandas_ta
try:
    import talib
    _HAS_TA = True
except ImportError:
    try:
        import pandas_ta as pta  # type: ignore
    except ImportError as exc:
        pta = None
    _HAS_TA = False

# --- MODIFICATION: Add 'adx' to the export list ---
__all__ = ["ema", "atr", "rsi", "macd", "bollinger", "lbr_310", "adx"]


def ema(series: pd.Series, span: int) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.EMA(series, timeperiod=span), index=series.index)
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.ATR(df["high"], df["low"], df["close"], timeperiod=period), index=df.index)
    if pta is None:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()
    atr_series = pta.atr(high=df["high"], low=df["low"], close=df["close"], length=period)
    if atr_series is None:
        return pd.Series(dtype='float64', index=df.index)
    return atr_series


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.RSI(series, timeperiod=period), index=series.index)
    if pta is None:
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        up_ema = up.ewm(alpha=1 / period, adjust=False).mean()
        down_ema = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = up_ema / down_ema.replace(0.0, np.nan)
        return 100.0 - 100.0 / (1.0 + rs)
    rsi_series = pta.rsi(series, length=period)
    if rsi_series is None:
        return pd.Series(dtype='float64', index=series.index)
    return rsi_series

def macd(series: pd.Series) -> pd.DataFrame:
    """
    Return DataFrame with columns 'macd', 'signal', 'hist', regardless of
    whether TA-Lib or pandas_ta is used behind the scenes.
    """
    if _HAS_TA:
        macd, sig, hist = talib.MACD(series)
        return pd.DataFrame(
            {"macd": macd, "signal": sig, "hist": hist}, index=series.index
        )

    if pta is None:
        macd_line = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
        signal = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal
        return pd.DataFrame({"macd": macd_line, "signal": signal, "hist": hist}, index=series.index)

    df_raw = pta.macd(series)
    mapping = {}
    for col in df_raw.columns:
        if "MACDh" in col or "hist" in col: mapping[col] = "hist"
        elif "MACDs" in col: mapping[col] = "signal"
        elif "MACD" in col: mapping[col] = "macd"
    df = df_raw.rename(columns=mapping)

    if "hist" not in df.columns: df["hist"] = df["macd"] - df["signal"]
    if "macd" not in df.columns: df["macd"] = np.nan
    if "signal" not in df.columns: df["signal"] = np.nan

    return df[["macd", "signal", "hist"]]


def bollinger(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    if _HAS_TA:
        upper, mid, lower = talib.BBANDS(series, timeperiod=length, nbdevup=std, nbdevdn=std)
        return pd.DataFrame({"upper": upper, "mid": mid, "lower": lower}, index=series.index)
    
    bbands = pta.bbands(series, length=length, std=std)
    
    if bbands is None or bbands.empty:
        return pd.DataFrame(columns=["upper", "mid", "lower"], index=series.index)

    upper_col = [col for col in bbands.columns if 'BBU' in col.upper()]
    mid_col = [col for col in bbands.columns if 'BBM' in col.upper()]
    lower_col = [col for col in bbands.columns if 'BBL' in col.upper()]

    if not (upper_col and mid_col and lower_col):
         return pd.DataFrame(columns=["upper", "mid", "lower"], index=series.index)

    return pd.DataFrame({
        "upper": bbands[upper_col[0]],
        "mid": bbands[mid_col[0]],
        "lower": bbands[lower_col[0]],
    }, index=series.index)


def lbr_310(series: pd.Series) -> pd.Series:
    """Linda Bradford Raschke 3‑10 oscillator = SMA3(close) – SMA10(close)."""
    return series.rolling(3).mean() - series.rolling(10).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average Directional Index (ADX).
    Returns a Series containing the ADX values, ensuring it's robust.
    """
    # Ensure there's enough data to calculate
    if len(df) < period:
        return pd.Series(dtype='float64', index=df.index)

    if _HAS_TA:
        adx_series = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
        return pd.Series(adx_series, index=df.index)

    if pta is None:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()
        up_move = high.diff().clip(lower=0.0)
        down_move = (low.shift(1) - low).clip(lower=0.0)
        plus_dm = ((up_move > down_move) * up_move).ewm(alpha=1 / period, adjust=False).mean()
        minus_dm = ((down_move > up_move) * down_move).ewm(alpha=1 / period, adjust=False).mean()
        pdi = 100 * plus_dm / atr_series.replace(0.0, np.nan)
        ndi = 100 * minus_dm / atr_series.replace(0.0, np.nan)
        dx = ((pdi - ndi).abs() / (pdi + ndi).replace(0.0, np.nan)) * 100
        return dx.ewm(alpha=1 / period, adjust=False).mean()
    
    # pandas_ta fallback
    adx_df = pta.adx(high=df["high"], low=df["low"], close=df["close"], length=period)
    if adx_df is None or adx_df.empty:
        return pd.Series(dtype='float64', index=df.index)
        
    # Find the column that contains 'ADX'
    adx_col = [col for col in adx_df.columns if 'ADX' in col.upper()]
    if not adx_col:
        return pd.Series(dtype='float64', index=df.index)
        
    return adx_df[adx_col[0]]

def vwap_stack_features(df: pd.DataFrame, lookback_bars: int = 12, band_pct: float = 0.004):
    """
    Compute VWAP-stack features from a 5m OHLCV DataFrame with columns:
    ['ts','open','high','low','close','volume'] in ascending time.

    Returns:
      {
        "vwap_frac_in_band": float in [0,1],
        "vwap_expansion_pct": abs(close/vwap - 1),
        "vwap_slope_pph": slope of rolling VWAP (% per hour) over ~lookback
      }
    """
    out0 = {"vwap_frac_in_band": 0.0, "vwap_expansion_pct": 0.0, "vwap_slope_pph": 0.0}

    if df is None or df.empty:
        return out0

    # Ensure numeric types
    df = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)
    # Need at least lookback+2 bars so we can exclude the current (last) bar for consolidation
    if n < lookback_bars + 2:
        return out0

    # Rolling VWAP over 'lookback_bars'
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0.0)

    # Sum over window; vv can be 0 if volume sums to zero
    tpv = (tp * vol).rolling(lookback_bars, min_periods=lookback_bars).sum()
    vv = vol.rolling(lookback_bars, min_periods=lookback_bars).sum()

    # If the most recent window has no volume, bail safely
    if not np.isfinite(vv.iloc[-1]) or vv.iloc[-1] == 0:
        return out0

    # Current rolling VWAP (unshifted) → for expansion on the last completed bar
    cur_vwap = (tpv / vv).iloc[-1]
    cur_close = float(df["close"].iloc[-1])
    expansion = abs(cur_close / cur_vwap - 1.0) if np.isfinite(cur_vwap) and cur_vwap != 0 else 0.0

    # Consolidation window: previous 'lookback_bars' bars ending at -1 (exclude current)
    # Use the SAME index range for closes and RVWAP so lengths match.
    prior_slice = slice(-lookback_bars - 1, -1)

    rvwap = (tpv / vv)  # already aligned to bar closes
    prior_vwap = rvwap.iloc[prior_slice].to_numpy()
    prior_close = df["close"].iloc[prior_slice].to_numpy()

    # Drop any pairs where VWAP is NaN/inf
    m = np.isfinite(prior_vwap) & np.isfinite(prior_close)
    prior_vwap = prior_vwap[m]
    prior_close = prior_close[m]

    if prior_vwap.size == 0 or prior_close.size == 0:
        return out0

    band_hi = prior_vwap * (1.0 + band_pct)
    band_lo = prior_vwap * (1.0 - band_pct)
    in_band = (prior_close >= band_lo) & (prior_close <= band_hi)
    frac = float(in_band.mean())

    # Slope of RVWAP over a short recent span (cap to 12 bars)
    k = int(min(lookback_bars, 12))
    vsub = rvwap.iloc[-k:].to_numpy()
    if k >= 2 and np.isfinite(vsub[0]) and vsub[0] != 0 and np.isfinite(vsub[-1]):
        slope = (vsub[-1] - vsub[0]) / vsub[0]           # fraction over k bars
        slope_pph = float(slope * (60 / 5) / k)          # percent-per-hour on 5m bars
    else:
        slope_pph = 0.0

    return {
        "vwap_frac_in_band": float(max(0.0, min(1.0, frac))),
        "vwap_expansion_pct": float(expansion),
        "vwap_slope_pph": float(slope_pph),
    }
