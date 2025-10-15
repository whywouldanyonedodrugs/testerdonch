# indicators.py  — MTF-safe indicators (ATR/MACD/Donch helpers)
from __future__ import annotations
import numpy as np
import pandas as pd

# If you have bottleneck installed, rolling ops get faster automatically
try:
    import bottleneck as bn
    _HAS_BN = True
except Exception:
    _HAS_BN = False

# --------------------- Utilities ---------------------

def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tz-aware UTC DatetimeIndex named 'timestamp'."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        if df.index is not idx:
            df = df.copy()
            df.index = idx
        df.index.name = "timestamp"
        return df
    if "timestamp" in df.columns:
        out = df.copy()
        out.index = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.drop(columns=["timestamp"])
        out.index.name = "timestamp"
        return out
    raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Simple OHLCV resampler; expects tz-aware 'timestamp' index."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    o = df["open"].resample(timeframe).first()
    h = df["high"].resample(timeframe).max()
    l = df["low"].resample(timeframe).min()
    c = df["close"].resample(timeframe).last()
    v = df["volume"].resample(timeframe).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    out.index = out.index.tz_convert("UTC")
    out.index.name = "timestamp"
    return out.dropna(how="any")

# --------------------- Core indicators ---------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([(high - low).abs(),
                      (high - prev_close).abs(),
                      (low - prev_close).abs()], axis=1).max(axis=1)

def atr_wilder_from_ohlc(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's ATR using TR and Wilder smoothing (EMA with alpha=1/length)."""
    df = _ensure_dtindex(df)
    tr = true_range(df["high"], df["low"], df["close"])
    # Wilder smoothing can be emulated with EMA(alpha=1/length)
    atr = tr.ewm(alpha=1.0/float(length), adjust=False, min_periods=length).mean()
    return atr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1/length, adjust=False).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def ema_np(series: pd.Series, length: int) -> np.ndarray:
    return ema(series, length).to_numpy(dtype=float)

def macd_histogram(close: pd.Series, fast:int=12, slow:int=26, signal:int=9) -> pd.Series:
    """Classic MACD histogram on the given series."""
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - sig

def macd_histogram_tf(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    timeframe: str | None = None
) -> pd.Series:
    """
    MACD histogram on base series. If `timeframe` is given (e.g., "4h"),
    compute MACD on the resampled close and forward-fill back to the base index.
    """
    # Ensure tz-aware UTC index
    close = close.copy()
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index, utc=True)
    else:
        close.index = close.index.tz_localize("UTC") if close.index.tz is None else close.index.tz_convert("UTC")
    close.index.name = "timestamp"

    if timeframe:
        close_tf = close.resample(timeframe).last().dropna()
        macd_line = close_tf.ewm(span=fast, adjust=False).mean() - close_tf.ewm(span=slow, adjust=False).mean()
        sig      = macd_line.ewm(span=signal, adjust=False).mean()
        hist_tf  = (macd_line - sig).astype(float)
        return hist_tf.reindex(close.index, method="ffill").astype(float)

    # base-TF MACD
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig       = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - sig).astype(float)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    up_ema = up.ewm(alpha=1/length, adjust=False).mean()
    dn_ema = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = up_ema / dn_ema.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.bfill()

def _dm_plus(high, low):  return (high.diff().clip(lower=0.0) > (low.shift(1) - low)).astype(float) * (high.diff().clip(lower=0.0))
def _dm_minus(high, low): return ((low.shift(1) - low).clip(lower=0.0) > high.diff()).astype(float) * ((low.shift(1) - low).clip(lower=0.0))


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/length, adjust=False).mean()
    dmp = _dm_plus(high, low).ewm(alpha=1/length, adjust=False).mean()
    dmn = _dm_minus(high, low).ewm(alpha=1/length, adjust=False).mean()
    pdi = 100 * (dmp / atr_).replace(0.0, np.nan)
    ndi = 100 * (dmn / atr_).replace(0.0, np.nan)
    dx = ( (pdi - ndi).abs() / (pdi + ndi).replace(0.0, np.nan) ) * 100
    return dx.ewm(alpha=1/length, adjust=False).mean()

# --------------------- Donchian helpers ---------------------

def donchian_upper_bars(high: pd.Series, length: int) -> pd.Series:
    """Upper Donchian channel on 'length' *bars* (shifted 1 to avoid same-bar look-ahead)."""
    if _HAS_BN:
        a = high.to_numpy(dtype=np.float64)
        # trailing rolling max excluding current bar => shift 1
        m = bn.move_max(a, window=length, min_count=length)
        return pd.Series(m, index=high.index).shift(1)
    else:
        return high.rolling(length, min_periods=length).max().shift(1)

def donchian_upper_days(high_5m: pd.Series, n_days: int) -> pd.Series:
    """
    Daily Donchian upper on n_days (shifted 1 day to avoid look-ahead),
    forward-filled to 5m index, keeping the index intact.
    """
    high_5m = high_5m.copy()
    if not isinstance(high_5m.index, pd.DatetimeIndex):
        high_5m.index = pd.to_datetime(high_5m.index, utc=True)
    else:
        high_5m.index = high_5m.index.tz_localize("UTC") if high_5m.index.tz is None else high_5m.index.tz_convert("UTC")

    daily_high = high_5m.resample("1D").max().dropna()
    don_daily  = daily_high.rolling(n_days, min_periods=n_days).max().shift(1)
    return don_daily.reindex(high_5m.index, method="ffill")

# --------------------- Volume spike (two flavors) ---------------------

_MAX_BN_WINDOW = 1000


def map_to_left_index(target_index: pd.DatetimeIndex, ts_to_value: pd.Series) -> pd.Series:
    """Align right (feature) series to left timestamps via last-known value forward-fill."""
    s = ts_to_value.copy()
    s.index = pd.to_datetime(s.index, utc=True)
    return s.reindex(target_index, method="ffill")

def rolling_median_multiple(vol: pd.Series, lookback_bars: int) -> pd.Series:
    med = vol.rolling(lookback_bars, min_periods=max(5, lookback_bars//10)).median()
    return vol / med.replace(0, np.nan)

def rolling_median(series: pd.Series, bars: int) -> pd.Series:
    """
    Rolling median that uses Bottleneck for small windows and
    falls back to pandas for large windows to avoid BN's window cap.
    """
    mp = max(1, bars // 4)  # tolerant warmup
    if _HAS_BN and bars <= _MAX_BN_WINDOW:
        a = series.to_numpy(dtype=np.float64)
        m = bn.move_median(a, window=bars, min_count=mp)
        return pd.Series(m, index=series.index)
    # pandas path (handles arbitrarily large windows)
    return series.rolling(bars, min_periods=mp).median()

def rolling_quantile(series: pd.Series, bars: int, q: float) -> pd.Series:
    """
    Rolling quantile with safe fallback for very large windows.
    pandas.rolling(...).quantile handles arbitrary windows.
    """
    mp = max(1, bars // 4)
    # Bottleneck has no move_quantile; always use pandas here
    return series.rolling(bars, min_periods=mp).quantile(q)

def volume_spike_multiple(vol: pd.Series, lookback_bars: int, multiple: float) -> pd.Series:
    """Spike if vol >= multiple * rolling median (use multiple=3.0 by default)."""
    base = rolling_median(vol, lookback_bars)
    return (vol >= multiple * base).astype(np.uint8)

def volume_spike_quantile(vol: pd.Series, lookback_bars: int, q: float=0.95) -> pd.Series:
    """Spike if vol >= rolling quantile q (use q=0.95 by default)."""
    thr = rolling_quantile(vol, lookback_bars, q)
    return (vol >= thr).astype(np.uint8)
