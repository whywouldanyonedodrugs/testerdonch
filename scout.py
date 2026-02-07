# scout.py
from __future__ import annotations
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

from uuid import uuid4
import gc

import config as cfg

# ✅ Bring in everything we actually use
from indicators import (
    resample_ohlcv,
    atr,
    ema,
    rsi,
    adx,
    macd_histogram as macd_hist,      # keep alias if you reference macd_hist elsewhere
    rolling_median_multiple,
    map_to_left_index,
    volume_spike_multiple,
    volume_spike_quantile,
)

from shared_utils import get_symbols_from_file, load_parquet_data

from regime_detector import compute_daily_combined_regime, DailyRegimeConfig, compute_markov_regime_4h

_SENTIMENT_INDEX_CACHE: Optional[pd.DataFrame] = None

# --- Local helpers -----------------------------------------------------------

def _macd_components(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD line / signal / hist for a price series.

    Returns: (macd_line, macd_signal, macd_hist) as pandas Series.
    """
    close = close.astype(float)
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist




def _load_sentiment_index() -> pd.DataFrame:
    """
    Load perps-wide sentiment index from results/sentiment_index.parquet (or cfg.SENTIMENT_INDEX_PATH if set),
    cache it in memory, and return a frame indexed by 'timestamp'.
    """
    global _SENTIMENT_INDEX_CACHE
    if _SENTIMENT_INDEX_CACHE is not None:
        return _SENTIMENT_INDEX_CACHE

    default_path = cfg.RESULTS_DIR / "sentiment_index.parquet"
    path = Path(getattr(cfg, "SENTIMENT_INDEX_PATH", default_path))

    if not path.exists():
        print(f"[scout] sentiment index not found at {path}; skipping sentiment features.")
        _SENTIMENT_INDEX_CACHE = pd.DataFrame()
        return _SENTIMENT_INDEX_CACHE

    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    df.index.name = "timestamp"
    df = df.sort_index()
    _SENTIMENT_INDEX_CACHE = df
    return _SENTIMENT_INDEX_CACHE

def _merge_cross_asset_context(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Attach BTCUSDT / ETHUSDT context to the per-symbol feature panel.

    Keeps existing behavior:
      - adds btcusdt_/ethusdt_ prefixed oi_* and funding_* columns (when available)

    Adds robust OHLCV-only context (always computable if OHLCV exists):
      - btcusdt_vol_regime_level, btcusdt_trend_slope
      - ethusdt_vol_regime_level, ethusdt_trend_slope

    Look-ahead protection:
      Daily features are shifted by 1 day before mapping to intraday timestamps.
    """
    base_dir = cfg.PARQUET_DIR
    ctx: dict[str, pd.Series] = {}

    # Normalize target timestamps once (avoid repeated parsing)
    ts5 = pd.to_datetime(feat["timestamp"], utc=True, errors="coerce")

    for asset in ("BTCUSDT", "ETHUSDT"):
        path = base_dir / f"{asset}.parquet"
        if not path.exists():
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"[scout] could not read {asset} parquet for cross-asset context: {e}")
            continue

        if df.empty:
            continue

        # Normalize to a DatetimeIndex named 'timestamp'
        df = _normalize_ts_index(df)
        asset_prefix = asset.lower()  # "btcusdt" / "ethusdt"

        # ------------------------------------------------------------------
        # (A) OHLCV-only DAILY regime/trend (do NOT depend on OI/funding)
        # ------------------------------------------------------------------
        needed_ohlc = ["open", "high", "low", "close", "volume"]
        if all(c in df.columns for c in needed_ohlc):
            ohlc = df[needed_ohlc].copy()
            ohlc = _normalize_ts_index(ohlc)

            daily = resample_ohlcv(ohlc, "1D")
            if not daily.empty:
                atr1d = atr(daily, 20)
                atr_pct_1d = atr1d / daily["close"].replace(0, np.nan)

                # Past-only baseline; shift(1) avoids using same-day close info intraday
                base = atr_pct_1d.expanding(min_periods=50).median().replace(0, np.nan)
                vol_regime_level = (atr_pct_1d / (base + 1e-12)).shift(1)

                ma20 = daily["close"].rolling(20, min_periods=20).mean()
                ma50 = daily["close"].rolling(50, min_periods=50).mean()
                trend_slope = (ma20 - ma50).diff().shift(1)

                ctx[f"{asset_prefix}_vol_regime_level"] = map_to_left_index(ts5, vol_regime_level)
                ctx[f"{asset_prefix}_trend_slope"] = map_to_left_index(ts5, trend_slope)

        # ------------------------------------------------------------------
        # (B) OI + funding features (only if columns exist)
        # ------------------------------------------------------------------
        needed_of = ["close", "volume", "open_interest", "funding_rate"]
        if not all(c in df.columns for c in needed_of):
            # still keep (A) even if (B) unavailable
            continue

        tmp = pd.DataFrame(
            {
                "timestamp": df.index,
                "close": df["close"].values,
                "volume": df["volume"].values,
                "open_interest": df["open_interest"].values,
                "funding_rate": df["funding_rate"].values,
            }
        )

        try:
            tmp = add_oi_funding_features(tmp)
        except Exception as e:
            print(f"[scout] add_oi_funding_features failed for {asset}: {e}")
            continue

        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp")
        if tmp.empty:
            continue

        cols = [c for c in tmp.columns if c.startswith("oi_") or c.startswith("funding_")]
        if not cols:
            continue

        tmp_indexed = tmp.set_index("timestamp")
        for c in cols:
            ctx[f"{asset_prefix}_{c}"] = map_to_left_index(ts5, tmp_indexed[c])

    # Attach aligned cross-asset columns to the per-symbol feature panel
    for k, v in ctx.items():
        feat[k] = v.values

    return feat


# --- Schema guards: keep 'timestamp'/'symbol' tidy & unambiguous ----------------
def _normalize_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the data has a DatetimeIndex named 'timestamp' and NO 'timestamp' column.
    This prevents frames from ever having 'timestamp' both as index and as a column.
    """
    df = df.copy()
    # If a 'timestamp' *column* exists, use it as the index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp", drop=True)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # Name the index and remove any duplicate 'timestamp' columns
    df.index.name = "timestamp"
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Remove duplicate columns if any (keep first)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df.sort_index()

def _ensure_ts_sym_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'timestamp' and 'symbol' are plain columns (not index levels) before merges.
    Prevents: ValueError: '<key>' is both an index level and a column label.
    """
    df = df.copy()

    # If either is on the index, reset them to columns
    if isinstance(df.index, pd.MultiIndex):
        need = [n for n in ("timestamp", "symbol") if n in (df.index.names or [])]
        if need:
            df = df.reset_index(level=need)
    else:
        if df.index.name in ("timestamp", "symbol"):
            df = df.reset_index()

    # De-name index to avoid future collisions
    if df.index.name in ("timestamp", "symbol"):
        df.index.name = None

    # Drop duplicate columns if any
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df



# --- Schema guards & feature builders (keep this section grouped) ----------------

def add_oi_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: timestamp (UTC), close, volume, open_interest, funding_rate.
    Uses only past data at each timestamp (no look-ahead).
    """
    df = df.sort_values("timestamp").copy()

    # Ensure numeric
    for col in ["open_interest", "funding_rate", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rolling window sizes for 5-minute bars
    WIN_1H = 12
    WIN_4H = 48
    WIN_1D = 288
    WIN_3D = 3 * WIN_1D
    WIN_7D = 7 * WIN_1D

    # ---- Open Interest (OI) features ----
    df["oi_level"] = df["open_interest"]
    df["oi_notional_est"] = df["open_interest"] * df["close"]

    # Crude leverage proxy: current OI notional vs 24h average notional
    # (using 288 × 5m bars ≈ 1 day)
    notional_24h = df["oi_notional_est"].rolling(WIN_1D, min_periods=WIN_1H).mean()
    df["est_leverage"] = df["oi_notional_est"] / (notional_24h + 1e-9)

    # Explicitly set fill_method=None to silence pandas deprecation warnings
    df["oi_pct_1h"] = df["open_interest"].pct_change(WIN_1H, fill_method=None)

    df["oi_pct_4h"] = df["open_interest"].pct_change(WIN_4H, fill_method=None)
    df["oi_pct_1d"] = df["open_interest"].pct_change(WIN_1D, fill_method=None)

    oi_mean_7d = df["open_interest"].rolling(WIN_7D, min_periods=WIN_1D).mean()
    oi_std_7d  = df["open_interest"].rolling(WIN_7D, min_periods=WIN_1D).std()
    df["oi_z_7d"] = (df["open_interest"] - oi_mean_7d) / (oi_std_7d + 1e-12)

    vol_1h = df["volume"].rolling(WIN_1H).sum()
    df["oi_chg_norm_vol_1h"] = df["open_interest"].diff(WIN_1H) / (vol_1h + 1e-9)

    # Simple OI–price interaction (keeps signs consistent and is cheap)
    ret_1h = df["close"].pct_change(WIN_1H, fill_method=None)
    df["oi_price_div_1h"] = np.sign(ret_1h) * df["oi_pct_1h"]

    # ---- Funding features ----
    # Forward-fill to 5m grid to avoid NaNs between funding prints
    df["funding_rate"] = df["funding_rate"].ffill()
    df["funding_abs"]  = df["funding_rate"].abs()

    fr_mean_7d = df["funding_rate"].rolling(WIN_7D, min_periods=WIN_1D).mean()
    fr_std_7d  = df["funding_rate"].rolling(WIN_7D, min_periods=WIN_1D).std()
    df["funding_z_7d"] = (df["funding_rate"] - fr_mean_7d) / (fr_std_7d + 1e-12)

    # Cumulative funding (3d)
    df["funding_rollsum_3d"] = df["funding_rate"].rolling(WIN_3D, min_periods=WIN_1D).sum()

    # Interaction: leverage + bias together
    df["funding_oi_div"] = df["funding_z_7d"] * df["oi_z_7d"]

    # --- Simple crowding flags from OI & funding z-scores --------------------
    # Uses thresholds from config.py; falls back to 1.0 / -1.0 if missing.
    high = float(getattr(cfg, "CROWD_Z_HIGH", 1.0))
    low  = float(getattr(cfg, "CROWD_Z_LOW", -1.0))

    # "Crowded long" → OI elevated and funding strongly positive
    crowded_long = (df["oi_z_7d"] >= high) & (df["funding_z_7d"] >= high)

    # "Crowded short" → OI elevated and funding strongly negative
    crowded_short = (df["oi_z_7d"] >= high) & (df["funding_z_7d"] <= low)

    # Store as both binary flags and a signed side indicator
    df["crowded_long"] = crowded_long.astype(int)
    df["crowded_short"] = crowded_short.astype(int)

    # -1 = crowded shorts, 0 = neutral/mixed, +1 = crowded longs
    df["crowd_side"] = 0
    df.loc[crowded_long, "crowd_side"] = 1
    df.loc[crowded_short, "crowd_side"] = -1

    # Cleanups: keep finite only
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df




def _eth_macd_full_4h_to_5m(
    signals_df: pd.DataFrame,
    eth_parquet_path: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Merge ETH MACD context onto 5m signals.

    Existing:
      - eth_macd_line_4h / eth_macd_signal_4h / eth_macd_hist_4h
      - eth_macd_both_pos_4h

    New:
      - eth_macd_hist_slope_4h: (hist_4h[t] - hist_4h[t-1]) on 4h bars, forward-filled to 5m
      - eth_macd_hist_slope_1h: (hist_1h[t] - hist_1h[t-1]) on 1h bars, forward-filled to 5m
    """

    if signals_df.empty:
        return signals_df

    eth_path = Path(eth_parquet_path)
    if not eth_path.exists():
        return signals_df

    eth = pd.read_parquet(eth_path)
    if eth.empty:
        return signals_df

    # Accept either:
    #  - a 'timestamp' column, OR
    #  - a DatetimeIndex named 'timestamp'
    if "timestamp" not in eth.columns:
        if isinstance(eth.index, pd.DatetimeIndex) and eth.index.name == "timestamp":
            eth = eth.reset_index()
        else:
            raise ValueError("ETH parquet must include a 'timestamp' column or index named 'timestamp'.")

    eth = eth.copy()
    eth["timestamp"] = pd.to_datetime(eth["timestamp"], utc=True, errors="coerce")
    eth = eth.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    # ---------- 4h MACD (line/signal/hist + slope) ----------
    close_4h = (
        eth["close"].astype(float)
        .resample("4h", label="right", closed="right")
        .last()
        .dropna()
    )
    if close_4h.empty:
        return signals_df

    ema_fast = close_4h.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_4h.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal

    # Δhist per 4h bar
    macd_hist_slope_4h = macd_hist.diff()

    macd_4h = pd.DataFrame(
        {
            "eth_macd_line_4h": macd_line,
            "eth_macd_signal_4h": macd_signal,
            "eth_macd_hist_4h": macd_hist,
            "eth_macd_hist_slope_4h": macd_hist_slope_4h,
        }
    )
    macd_5m = macd_4h.resample("5min").ffill()

    # ---------- 1h MACD histogram slope ----------
    close_1h = (
        eth["close"].astype(float)
        .resample("1h", label="right", closed="right")
        .last()
        .dropna()
    )
    if not close_1h.empty:
        ema_fast_1h = close_1h.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow_1h = close_1h.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line_1h = ema_fast_1h - ema_slow_1h
        macd_signal_1h = macd_line_1h.ewm(span=signal, adjust=False, min_periods=signal).mean()
        macd_hist_1h = macd_line_1h - macd_signal_1h
        macd_hist_slope_1h = macd_hist_1h.diff()

        macd_1h = pd.DataFrame({"eth_macd_hist_slope_1h": macd_hist_slope_1h})
        macd_5m = macd_5m.join(macd_1h.resample("5min").ffill(), how="left")

    out = signals_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.merge(macd_5m, left_on="timestamp", right_index=True, how="left")

    out["eth_macd_both_pos_4h"] = (
        (out["eth_macd_line_4h"] > 0).astype("int8")
        & (out["eth_macd_hist_4h"] > 0).astype("int8")
    ).astype("int8")

    return out




def _optimize_signal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    # timestamps
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    # IMPORTANT: keep label-like columns as pandas 'string' (NOT 'category')
    for col in ("symbol", "pullback_type", "entry_rule"):
        if col in out.columns:
            out[col] = out[col].astype("string")

    # floats → float32 where possible
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="float")
    # ints → int32/int16
    for c in out.select_dtypes(include=["int64"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="integer")
    # bools → uint8
    for c in out.select_dtypes(include=["bool"]).columns:
        out[c] = out[c].astype("uint8")

    return out

def _write_partitioned(df: pd.DataFrame, base_dir: Path) -> int:
    """Write a chunk to signals/ as symbol-partitioned Parquet; return rows written."""
    if df.empty:
        return 0

    df = _optimize_signal_dtypes(df)

    # Determine target partition from the first row
    sym = str(df["symbol"].iloc[0])

    # DROP the in-file 'symbol' column — rely on Hive partition ('symbol=...') only
    df = df.drop(columns=["symbol"], errors="ignore")

    # Partition by symbol manually → signals/symbol=XYZ/part-*.parquet
    part_dir = base_dir / f"symbol={sym}"
    part_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        part_dir / f"part-{uuid4().hex}.parquet",
        compression="snappy",
        row_group_size=int(getattr(cfg, "SCOUT_ROW_GROUP_SIZE", 100_000)),
    )
    return len(df)


def _process_one_symbol(sym: str, rs_table: Optional[pd.DataFrame]) -> int:
    """Detect + enrich for a single symbol, then write that symbol's rows; return count."""
    sig = detect_signals_for_symbol(sym, rs_table)
    if sig.empty:
        return 0

    # Order, dedup within busy window
    sig = (sig.sort_values(["timestamp"])
              .drop_duplicates(subset=["timestamp"], keep="first")
              .reset_index(drop=True))
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True)
    sig = _dedup_busy_window(sig, minutes=int(getattr(cfg, "DEDUP_BUSY_WINDOW_MIN", 0)))

    # Build per-symbol feature panel (now returns columns, not an index)
    feat = _build_feature_panel(sym)

    if not feat.empty:
        # Make sure both sides have unambiguous columns before merging
        sig  = _ensure_ts_sym_columns(sig)
        feat = _ensure_ts_sym_columns(feat)

        sig = sig.merge(
            feat,
            on=["timestamp", "symbol"],
            how="left",
            copy=False,
            validate="m:1",
        )

        # don_dist_atr if we have Donchian upper and ATR context (prefer atr_1h, fallback to 'atr')
        if "don_break_level" in sig.columns:
            sig["don_upper"] = sig.get("don_upper", sig["don_break_level"])
        if "don_upper" in sig.columns:
            scale = sig["atr_1h"].where(sig["atr_1h"].notna(), sig.get("atr", np.nan))
            sig["don_dist_atr"] = (sig["close"] - sig["don_upper"]) / scale.replace(0, np.nan)

    try:
        sig = _eth_macd_full_4h_to_5m(
            signals_df=sig,
            eth_parquet_path=str(Path("parquet") / f"{getattr(cfg, 'REGIME_ASSET', 'ETHUSDT')}.parquet"),
            fast=int(getattr(cfg, "REGIME_MACD_FAST", 12)),
            slow=int(getattr(cfg, "REGIME_MACD_SLOW", 26)),
            signal=int(getattr(cfg, "REGIME_MACD_SIGNAL", 9)),
        )
    except Exception:
        pass

    # Final write for this symbol
    return _write_partitioned(sig, cfg.SIGNALS_DIR)



# ---------------- Donchian helpers (days vs bars) ----------------

if "volume_spike_multiple" not in globals() or "volume_spike_quantile" not in globals():
    def volume_spike_multiple(vol: pd.Series, lookback_bars: int, multiple: float) -> pd.Series:
        mp = max(1, lookback_bars // 4)
        base = vol.rolling(lookback_bars, min_periods=mp).median()
        return (vol >= multiple * base).astype(np.uint8)

    def volume_spike_quantile(vol: pd.Series, lookback_bars: int, q: float = 0.95) -> pd.Series:
        mp = max(1, lookback_bars // 4)
        thr = vol.rolling(lookback_bars, min_periods=mp).quantile(q)
        return (vol >= thr).astype(np.uint8)

def _attach_entry_quality_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Adds to 'out' (signals):
      - atr_1h, rsi_1h, adx_1h, vol_mult, atr_pct           (if not already there)
      - days_since_prev_break (vs DON_N_DAYS Donch upper)
      - consolidation_range_atr (pullback-window hi-lo / ATR(1h))
      - prior_1d_ret (rolling 288-bar return)
      - rv_3d (rolling 3-day realized vol of 5m log returns)
    """
    if out.empty:
        return out

    out = out.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    syms = out["symbol"].unique().tolist()

    panels = []
    for sym in syms:
        df5 = load_parquet_data(
            sym, start_date=cfg.START_DATE, end_date=cfg.END_DATE,
            drop_last_partial=True, columns=["open","high","low","close","volume"]
        )
        if df5.empty:
            continue
        df5 = df5.sort_index()

        # 1h context
        df1h = resample_ohlcv(df5, "1h")
        atr1h = atr(df1h, int(getattr(cfg, "ATR_LEN", 14)))
        rsi1h = rsi(df1h["close"], int(getattr(cfg, "RSI_LEN", 14)))
        adx1h = adx(df1h, int(getattr(cfg, "ADX_LEN", 14)))

        atr1h_5m = atr1h.reindex(df5.index, method="ffill")
        rsi1h_5m = rsi1h.reindex(df5.index, method="ffill")
        adx1h_5m = adx1h.reindex(df5.index, method="ffill")

        # vol_mult over rolling 30 days on 5m (approx 288 bars/day)
        bars_per_day = 288
        lookback_days = int(getattr(cfg, "VOL_LOOKBACK_DAYS", 30))
        lb_bars = bars_per_day * lookback_days
        med = df5["volume"].rolling(lb_bars, min_periods=max(5, lb_bars//10)).median()
        vol_mult = (df5["volume"] / med.replace(0, np.nan)).astype(float)

        # ATR% of price
        atr_pct = (atr1h_5m / df5["close"]).astype(float)

        # days_since_prev_break using DON_N_DAYS (daily Donch upper)
        N = int(getattr(cfg, "DON_N_DAYS", 20))
        daily_high = df5["high"].resample("1D").max().dropna()
        don_daily = daily_high.rolling(N, min_periods=N).max().shift(1)  # no look-ahead
        don_5m = don_daily.reindex(df5.index, method="ffill")
        touch = (df5["high"] >= don_5m)
        # last touch timestamp (ffill)
        touch_time = pd.Series(df5.index.where(touch), index=df5.index)
        last_touch = touch_time.ffill()
        days_since_prev_break = (df5.index.to_series() - last_touch).dt.total_seconds() / 86400.0
        days_since_prev_break = days_since_prev_break.replace([np.inf, -np.inf], np.nan)

        # consolidation_range_atr over pullback window
        win_bars = int(getattr(cfg, "PULLBACK_WINDOW_BARS", 12))
        if win_bars <= 0:
            win_bars = int(round((getattr(cfg, "PULLBACK_WINDOW_HOURS", 24) or 24) * 60 / 5))
        cons_range = (df5["high"].rolling(win_bars).max() - df5["low"].rolling(win_bars).min())
        consolidation_range_atr = (cons_range / atr1h_5m.replace(0, np.nan)).astype(float)

        # prior_1d_ret (rolling 288-bar return; strictly past)
        prior_1d_ret = (df5["close"] / df5["close"].shift(bars_per_day) - 1.0).astype(float)

        # rv_3d: rolling std of 5m log returns over 3 days
        logret = np.log(df5["close"]).diff()
        rv_3d = logret.rolling(3 * bars_per_day).std().astype(float)

        p = pd.DataFrame({
            "timestamp": df5.index,
            "symbol": sym,
            "atr_1h": atr1h_5m.values,
            "rsi_1h": rsi1h_5m.values,
            "adx_1h": adx1h_5m.values,
            "vol_mult": vol_mult.values,
            "atr_pct": atr_pct.values,
            "days_since_prev_break": days_since_prev_break.values,
            "consolidation_range_atr": consolidation_range_atr.values,
            "prior_1d_ret": prior_1d_ret.values,
            "rv_3d": rv_3d.values,
        }).set_index(["timestamp","symbol"])
        panels.append(p)

    if not panels:
        return out

    feat_panel = pd.concat(panels).sort_index()
    # Drop exact duplicate index rows if any
    feat_panel = feat_panel[~feat_panel.index.duplicated(keep="last")]

    # Ensure same MultiIndex on both sides
    out_idx = out.set_index(["timestamp", "symbol"]).sort_index()
    # Align feature panel to out's index (ffill is OK because features are built on past-only info)
    feat_panel = feat_panel.reindex(out_idx.index, method="ffill")

    # Add only brand-new columns; for overlaps, coalesce (keep existing non-NA, otherwise take new)
    for col in feat_panel.columns:
        if col in out_idx.columns:
            out_idx[col] = out_idx[col].where(out_idx[col].notna(), feat_panel[col])
        else:
            out_idx[col] = feat_panel[col]

    out = out_idx.reset_index()
    return out

    # === Append 4h Markov (returns) regime features ===
    try:
        mk4 = compute_markov_regime_4h(
            asset=getattr(cfg, "REGIME_ASSET", "ETHUSDT"),
            timeframe=getattr(cfg, "REGIME_TIMEFRAME", "4h"),
        )
    except Exception as e:
        print(f"[scout] 4h markov regime failed: {e}")
        mk4 = pd.DataFrame()

    if not out.empty and not mk4.empty:
        # forward-fill from 4h to entry timestamps
        ts_sorted_unique = np.unique(np.sort(out["timestamp"].values))
        mk4_ff = mk4.reindex(ts_sorted_unique, method="ffill")
        out = out.set_index("timestamp")
        out["markov_state_up_4h"] = mk4_ff["state_up"].astype("Int8").reindex(out.index, method="ffill").values
        out["markov_prob_up_4h"]  = mk4_ff["prob_up"].astype(float).reindex(out.index, method="ffill").values
        out = out.reset_index()


    return out

def donchian_upper_days_no_lookahead(high_5m: pd.Series, n_days: int) -> pd.Series:
    """
    Daily Donchian upper on *completed* days only:
      1) resample 5m highs to daily highs
      2) rolling max over n_days
      3) shift(1) => prior days only (no look-ahead)
      4) map/ffill back to 5m index within each day
    """
    # daily highs by completed sessions
    daily_high = high_5m.resample("1D", label="right", closed="right").max()
    # N-day rolling max of completed days
    don_daily = daily_high.rolling(n_days, min_periods=n_days).max().shift(1)
    # map to 5m bars: align by day start (floor('D')) then forward-fill inside the day
    keyed = don_daily.reindex(high_5m.index.floor("D"))
    return keyed.ffill().to_numpy(dtype=float)

def donchian_upper_bars(high_5m: pd.Series, n_bars: int) -> pd.Series:
    return high_5m.rolling(n_bars, min_periods=n_bars).max().shift(1).to_numpy()

def _dedup_busy_window(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty or minutes <= 0: return df
    out = []
    win = pd.Timedelta(minutes=int(minutes))
    for sym, g in df.sort_values("timestamp").groupby("symbol"):
        last = None
        for _, row in g.iterrows():
            ts = row["timestamp"]
            if last is None or ts >= last + win:
                out.append(row); last = ts
    return pd.DataFrame(out)

# ---------------- Weekly RS (unchanged structure) ----------------

def _daily_row_for_symbol(sym: str) -> pd.DataFrame:
    """
    Build a DAILY RS / liquidity row for a single symbol.
    Matches Live Bot's rolling 24h/7d logic much closer than weekly snapshots.
    """
    df5 = load_parquet_data(
        sym,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        drop_last_partial=True,
        columns=["close", "volume"],
    )
    if df5.empty:
        return pd.DataFrame()

    df5 = _normalize_ts_index(df5)

    # 1. Resample to Daily
    daily = df5.resample("1D").agg({
        "close": "last",
        "volume": "sum"
    }).dropna()
    
    if daily.empty:
        return pd.DataFrame()

    # 2. Calculate USD Volume (Daily)
    daily["usd_vol_24h"] = daily["close"] * daily["volume"]

    # 3. Calculate Liquidity Proxy (Median of last 7 days)
    daily["usd_vol_med_24h"] = daily["usd_vol_24h"].rolling(7, min_periods=1).median()

    # 4. Calculate 7-Day Return (RS)
    daily["ret_7d"] = daily["close"].pct_change(7)

    daily["symbol"] = sym
    daily = daily.reset_index().rename(columns={"timestamp": "week_start"})
    
    return daily[["week_start", "symbol", "ret_7d", "usd_vol_med_24h"]]

def build_weekly_rs(symbols: List[str]) -> pd.DataFrame:
    """
    Builds a DAILY RS table (despite the function name) for higher fidelity parity.
    """
    from concurrent.futures import ThreadPoolExecutor as Exec

    rows: list[pd.DataFrame] = []
    workers = int(getattr(cfg, "RS_N_WORKERS", 4))

    print(f"[RS] Building DAILY RS/Liquidity metrics for {len(symbols)} symbols...")
    
    if workers <= 1:
        for s in tqdm(symbols, desc="RS Daily"):
            r = _daily_row_for_symbol(s)
            if not r.empty:
                rows.append(r)
    else:
        with Exec(max_workers=workers) as ex:
            futs = {ex.submit(_daily_row_for_symbol, s): s for s in symbols}
            for f in tqdm(as_completed(futs), total=len(futs), desc="RS Daily (parallel)"):
                try:
                    r = f.result()
                    if not r.empty:
                        rows.append(r)
                except Exception as e:
                    print(f"[RS] {futs[f]} failed: {e}")

    if not rows:
        return pd.DataFrame(columns=["week_start", "symbol", "rs_pct", "usd_vol_med_24h"])

    rs = pd.concat(rows, ignore_index=True)
    rs = rs.dropna(subset=["ret_7d"])

    # Rank by day
    rs["rs_rank"] = rs.groupby("week_start")["ret_7d"].rank(method="average", ascending=False)
    counts = rs.groupby("week_start")["ret_7d"].transform("count")
    denom = (counts - 1).clip(lower=1)
    rs["rs_pct"] = (100.0 * (counts - rs["rs_rank"]) / denom).clip(0.0, 100.0)

    # Save
    out_cols = ["week_start", "symbol", "rs_pct", "usd_vol_med_24h"]
    rs = rs[out_cols].sort_values(["week_start", "symbol"])
    
    pq.write_table(pa.Table.from_pandas(rs), cfg.RESULTS_DIR / "rs_weekly.parquet")
    print(f"[RS] Saved DAILY metrics to {cfg.RESULTS_DIR / 'rs_weekly.parquet'}")
    return rs


def rs_lookup(rs_table: pd.DataFrame, sym: str, ts: pd.Timestamp) -> float | None:
    """Return RS percentile for (symbol, timestamp) or None if unavailable."""
    rs_sym = rs_table[rs_table["symbol"] == sym]
    if rs_sym.empty:
        return None
    candidates = rs_sym[rs_sym["week_start"] <= ts.floor("D")]
    if candidates.empty:
        return None
    return float(candidates.iloc[-1]["rs_pct"])


def liq_lookup(rs_table: pd.DataFrame, sym: str, ts: pd.Timestamp) -> float | None:
    """
    Look up the weekly liquidity proxy (~median 24h USD turnover)
    for a given symbol at time ts.

    Uses the same weekly 'week_start' alignment as rs_lookup.
    """
    rs_sym = rs_table[rs_table["symbol"] == sym]
    if rs_sym.empty:
        return None
    candidates = rs_sym[rs_sym["week_start"] <= ts.floor("D")]
    if candidates.empty:
        return None
    return float(candidates.iloc[-1]["usd_vol_med_24h"])

# ---------------- Per-symbol signal detection ----------------

def _build_feature_panel(sym: str) -> pd.DataFrame:
    df5 = load_parquet_data(
        sym,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        drop_last_partial=True,
        columns=["open", "high", "low", "close", "volume", "open_interest", "funding_rate"],
    )
    if df5.empty:
        return pd.DataFrame(columns=["timestamp", "symbol"])

    # Normalize 5m frame: DatetimeIndex named 'timestamp', sorted, no 'timestamp' column
    df5 = _normalize_ts_index(df5)

    # --- 1h context features --------------------------------------------------
    df1h = resample_ohlcv(df5, "1h")
    atr1h = atr(df1h, cfg.ATR_LEN)
    rsi1h = rsi(df1h["close"], cfg.RSI_LEN)
    adx1h = adx(df1h, cfg.ADX_LEN)

    # Map 1h to 5m (left-aligned, ffill semantics inside map_to_left_index)
    atr1h_5m = map_to_left_index(df5.index, atr1h)
    rsi1h_5m = map_to_left_index(df5.index, rsi1h)
    adx1h_5m = map_to_left_index(df5.index, adx1h)

    # Volume structure (5m bars/day ≈ 288)
    bars_per_day = 288
    vol_mult = rolling_median_multiple(
        df5["volume"],
        lookback_bars=bars_per_day * cfg.VOL_LOOKBACK_DAYS,
    )

    # ATR % of price
    atr_pct = (atr1h_5m / df5["close"]).astype(float)

    # --- OI & Funding features for the traded symbol -------------------------
    tmp = pd.DataFrame(
        {
            "timestamp": df5.index,
            "close": df5["close"].values,
            "volume": df5["volume"].values,
            "open_interest": df5["open_interest"].values
            if "open_interest" in df5.columns
            else np.nan,
            "funding_rate": df5["funding_rate"].values
            if "funding_rate" in df5.columns
            else np.nan,
        }
    )

    if ("open_interest" in df5.columns) or ("funding_rate" in df5.columns):
        oi_feat = add_oi_funding_features(tmp)
    else:
        oi_feat = pd.DataFrame()

    # Base feature frame (plain columns; we'll merge by timestamp later)
    feat = pd.DataFrame(
        {
            "timestamp": df5.index,
            "symbol": sym,
            "atr_1h": atr1h_5m.values,
            "rsi_1h": rsi1h_5m.values,
            "adx_1h": adx1h_5m.values,
            "vol_mult": vol_mult.values,
            "atr_pct": atr_pct.values,
            "close": df5["close"].values,
            "don_upper": np.nan,  # filled later when we know Donchian level
        }
    )

    if not oi_feat.empty:
        # Ensure ordering by time (add_oi_funding_features already does this,
        # but re-sort here defensively)
        oi_feat = oi_feat.sort_values("timestamp").copy()

        # Forward-fill funding-related features between prints
        for col in [
            "funding_rate",
            "funding_abs",
            "funding_z_7d",
            "funding_rollsum_3d",
            "funding_oi_div",
        ]:
            if col in oi_feat.columns:
                oi_feat[col] = oi_feat[col].ffill()

    # Attach known engineered columns by position (same length as df5)
    for c in [
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
        "crowded_long",
        "crowded_short",
        "crowd_side",
        "est_leverage",
    ]:
        if c in oi_feat.columns:
            feat[c] = oi_feat[c].values



    # --- Additional RSI / volatility / MACD features (Chunk 2 base) ----------
    df15m = resample_ohlcv(df5, "15min")
    df4h = resample_ohlcv(df5, "4h")

    # RSI at 15m and 4h
    rsi15m = rsi(df15m["close"], 14)
    rsi4h = rsi(df4h["close"], 14)
    feat["asset_rsi_15m"] = map_to_left_index(df5.index, rsi15m).values
    feat["asset_rsi_4h"] = map_to_left_index(df5.index, rsi4h).values

    # MACD 1h & 4h (line / signal / hist + simple slope over 3 steps)
    macd1h_line, macd1h_signal, macd1h_hist = _macd_components(df1h["close"])
    macd4h_line, macd4h_signal, macd4h_hist = _macd_components(df4h["close"])

    macd1h_slope = macd1h_hist.diff(3)
    macd4h_slope = macd4h_hist.diff(3)

    feat["asset_macd_line_1h"] = map_to_left_index(df5.index, macd1h_line).values
    feat["asset_macd_signal_1h"] = map_to_left_index(df5.index, macd1h_signal).values
    feat["asset_macd_hist_1h"] = map_to_left_index(df5.index, macd1h_hist).values
    feat["asset_macd_slope_1h"] = map_to_left_index(df5.index, macd1h_slope).values

    feat["asset_macd_line_4h"] = map_to_left_index(df5.index, macd4h_line).values
    feat["asset_macd_signal_4h"] = map_to_left_index(df5.index, macd4h_signal).values
    feat["asset_macd_hist_4h"] = map_to_left_index(df5.index, macd4h_hist).values
    feat["asset_macd_slope_4h"] = map_to_left_index(df5.index, macd4h_slope).values

    # Realized volatility proxies (log-return stdevs)
    ret1h = np.log(df1h["close"]).diff()
    vol1h = ret1h.rolling(20).std() * np.sqrt(20.0)

    ret4h = np.log(df4h["close"]).diff()
    vol4h = ret4h.rolling(20).std() * np.sqrt(20.0)

    feat["asset_vol_1h"] = map_to_left_index(df5.index, vol1h).values
    feat["asset_vol_4h"] = map_to_left_index(df5.index, vol4h).values

    # MA gap and pre-breakout congestion
    ma1d = df5["close"].rolling(288).mean()
    gap_from_1d_ma = (df5["close"] - ma1d) / atr1h_5m.replace(0, np.nan)
    prebreak_congestion = df5["close"].pct_change().rolling(3 * 288).std()

    feat["gap_from_1d_ma"] = gap_from_1d_ma.values
    feat["prebreak_congestion"] = prebreak_congestion.values

    # Hook for future BTC/ETH cross-asset context
    feat = _merge_cross_asset_context(feat)

    # --- Cross-sectional sentiment features (from sentiment_index.py) ---------
    try:
        sent = _load_sentiment_index()
        if not sent.empty:
            sent_cols = [
                c
                for c in [
                    "sent_rets_1h_z",
                    "sent_rets_1d_z",
                    "sent_oi_chg_1h_z",
                    "sent_oi_chg_1d_z",
                    "sent_funding_mean_1d",
                    "sent_funding_z_1d",
                    "sent_beta_risk_on",
                ]
                if c in sent.columns
            ]
            if sent_cols:
                sent_reset = sent[sent_cols].reset_index()  # brings 'timestamp' column back
                feat = feat.merge(sent_reset, on="timestamp", how="left")
    except Exception as e:
        print(f"[scout] sentiment index merge failed for {sym}: {e}")

    return feat



def _eth_macd_hist_4h_to_5m() -> pd.Series:
    eth = load_parquet_data(cfg.REGIME_ASSET,
                            start_date=cfg.START_DATE,
                            end_date=cfg.END_DATE,
                            drop_last_partial=True,
                            columns=["open","high","low","close","volume"])
    if eth.empty:
        return pd.Series(dtype=float)
    eth4 = resample_ohlcv(eth, cfg.REGIME_TIMEFRAME)
    hist4 = macd_hist(eth4["close"], cfg.REGIME_MACD_FAST, cfg.REGIME_MACD_SLOW, cfg.REGIME_MACD_SIGNAL)
    # Map to 5m timestamps covering whole project window
    # We'll align later on the left index of signals
    return hist4

def detect_signals_for_symbol(sym: str, rs_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Donchian + pullback detector rewritten to mirror the live YAML logic more closely.

    Conditions (per-symbol, 5m bars):
      - Daily Donchian breakout confirm: daily close > prior N-day high
        using *daily* bars and using the last completed day as in live.
      - Retest: within RETEST_LOOKBACK_BARS, some bar's [low, high] crosses
        a band around the level [level*(1-eps), level*(1+eps)].
      - Current close > level (same as live pullback op).
      - Volume median multiple >= VOL_MULTIPLE (or quantile mode).
      - RS percentile >= RS_MIN_PERCENTILE (if enabled).
      - ATR is computed as in backtester cfg (for later use by meta model).
    """
    df = load_parquet_data(
        sym,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        drop_last_partial=True,
        columns=["open", "high", "low", "close", "volume"],
    )
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "entry",
                "atr",
                "don_break_len",
                "don_break_level",
                "pullback_type",
                "entry_rule",
                "vol_spike",
                "rs_pct",
            ]
        ).astype({"timestamp": "datetime64[ns, UTC]"})

    df = _normalize_ts_index(df)
    df = df.sort_index()

    # --- ATR (same as before: higher TF ATR forward-filled to 5m) ---
    tf = getattr(cfg, "ATR_TIMEFRAME", None)
    if tf:
        dft = resample_ohlcv(df, str(tf))
        atr_tf = atr(dft, int(getattr(cfg, "ATR_LEN", 14)))
        df["atr"] = atr_tf.reindex(df.index, method="ffill")
    else:
        df["atr"] = atr(df, int(getattr(cfg, "ATR_LEN", 14)))

    # --- Daily Donchian breakout level using *daily* bars ---
    if cfg.DONCH_BASIS.lower() != "days":
        raise ValueError("This detector expects DONCH_BASIS='days' to mirror live YAML logic.")

    don_n_days = int(cfg.DON_N_DAYS)

    # Group 5m bars into calendar days and compute daily highs and closes
    day_index = df.index.floor("D")
    daily = (
        df.assign(_day=day_index)
        .groupby("_day")
        .agg({"high": "max", "close": "last"})
        .sort_index()
    )

    # Prior N-day rolling high, shifted by 1 day to avoid look-ahead,
    # as in _op_donch_breakout_daily_confirm.
    daily["donch_upper"] = daily["high"].rolling(don_n_days, min_periods=don_n_days).max().shift(1)

    # Daily breakout flag: daily close > prior N-day high
    daily["donch_break_ok"] = (daily["close"] > daily["donch_upper"]) & daily["donch_upper"].notna()

    # In live, for any intraday 5m bar on day D, the breakout op sees the
    # last *completed* 1d bar, i.e. day D-1 (because they drop the last
    # partial 1d candle). We mirror this by shifting the daily breakout
    # information forward by 1 day and joining on the 5m day index.
    daily_effect = daily[["donch_upper", "donch_break_ok"]].copy()
    daily_effect.index = daily_effect.index + pd.Timedelta(days=1)
    daily_effect = daily_effect.rename(
        columns={
            "donch_upper": "donch_break_level",
            "donch_break_ok": "donch_break_ok",
        }
    )

    df = df.assign(day=day_index)
    df = df.join(daily_effect, on="day")

    # --- Volume spike: mirror volume_median_multiple op (using bars) ---
    vol_mode = getattr(cfg, "VOL_SPIKE_MODE", "multiple")
    vol_enabled = bool(getattr(cfg, "VOL_SPIKE_ENABLED", True))

    # Estimate bars per day from the index spacing
    if len(df) >= 2:
        minutes = max(1, int((df.index[1] - df.index[0]).total_seconds() / 60.0))
    else:
        minutes = 5
    bars_per_day = max(1, int(round(24 * 60 / minutes)))

    vol_days = int(getattr(cfg, "VOL_LOOKBACK_DAYS", 30))
    cap_bars = int(getattr(cfg, "VOL_CAP_BARS", 9000))

    lookback = min(cap_bars, vol_days * bars_per_day)
    if lookback <= 0:
        lookback = bars_per_day  # fallback to ~1 day

    if vol_enabled:
        vol = df["volume"].astype(float)
        vol_med = vol.rolling(
            window=lookback,
            min_periods=max(5, bars_per_day, lookback // 10),
        ).median()
        vol_med = vol_med.replace(0.0, np.nan)
        vol_mult = vol / vol_med
        if vol_mode == "multiple":
            vol_thr = float(getattr(cfg, "VOL_MULTIPLE", 1.0))
            vol_ok = vol_mult >= vol_thr
        else:
            q = float(getattr(cfg, "VOL_QUANTILE_Q", 0.90))
            vol_q = vol.rolling(
                window=lookback,
                min_periods=max(5, bars_per_day),
            ).quantile(q)
            vol_ok = vol >= vol_q
    else:
        vol_mult = pd.Series(1.0, index=df.index)
        vol_ok = pd.Series(True, index=df.index)

    # --- Micro vol filter (ATR/price >= MICRO_VOL_MIN) ---
    micro_min = float(getattr(cfg, "MICRO_VOL_MIN", 0.0))
    if micro_min > 0.0:
        micro_ratio = df["atr"].astype(float) / df["close"].astype(float)
        micro_ok = micro_ratio >= micro_min
    else:
        micro_ok = pd.Series(True, index=df.index)

    # --- Retest logic (pullback_retest_close_above_break) ---
    eps = float(getattr(cfg, "RETEST_EPS_PCT", 0.0))
    lb = int(getattr(cfg, "RETEST_LOOKBACK_BARS", 288))
    N = len(df)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    level = df["donch_break_level"].to_numpy(dtype=float)
    atrv = df["atr"].to_numpy(dtype=float)

    vol_ok_arr = vol_ok.to_numpy(dtype=bool)
    micro_ok_arr = micro_ok.to_numpy(dtype=bool)
    # daily breakout flag aligned to 5m bars
    _break = df["donch_break_ok"]
    _break_arr = _break.to_numpy()  # may contain True/False/NaN (object dtype)
    break_ok_arr = np.where(pd.isna(_break_arr), False, _break_arr).astype(bool)
    
    rows = []
    for i in range(N):
        lev = level[i]
        if not np.isfinite(lev):
            continue

        # Daily Donch breakout confirm: require breakout flag to be True,
        # mirroring _op_donch_breakout_daily_confirm.
        if not break_ok_arr[i]:
            continue

        # Current close above level (same as live retest op)
        if not (close[i] > lev):
            continue

        # Volume / micro filters
        if not vol_ok_arr[i]:
            continue
        if not micro_ok_arr[i]:
            continue

        # Retest: any of last lb bars touches [lev*(1-eps), lev*(1+eps)]
        start = max(0, i - lb + 1)
        band_hi = lev * (1.0 + eps)
        band_lo = lev * (1.0 - eps)

        sub_low = low[start : i + 1]
        sub_high = high[start : i + 1]
        touched = bool(((sub_low <= band_hi) & (sub_high >= band_lo)).any())
        if not touched:
            continue

        ts = df.index[i]

        # --- RS gating: mirror universe_rs_pct_gte ---
        rs_pct = None
        if cfg.RS_ENABLED and rs_table is not None:
            rs_pct = rs_lookup(rs_table, sym, ts)
        min_rs = int(getattr(cfg, "RS_MIN_PERCENTILE", 0))
        if min_rs > 0 and rs_pct is not None and rs_pct < min_rs:
            # Below required RS percentile → skip, as live YAML would.
            continue

        # --- Liquidity gating: mirror liquidity_median_24h_usd_gte ---
        liq_min = float(getattr(cfg, "RS_LIQ_MIN_USD_24H", 0.0))
        if rs_table is not None and liq_min > 0.0:
            liq_usd = liq_lookup(rs_table, sym, ts)
            if liq_usd is None or liq_usd < liq_min:
                # Not liquid enough at this time → skip.
                continue

        rows.append(
            {
                "timestamp": ts,
                "symbol": sym,
                "entry": float(close[i]),
                "atr": float(atrv[i]) if np.isfinite(atrv[i]) else np.nan,
                "don_break_len": don_n_days,
                "don_break_level": float(lev),
                "pullback_type": "retest",
                "entry_rule": "donch_yaml_v1",
                "vol_spike": bool(vol_ok_arr[i]),
                "rs_pct": rs_pct,
            }
        )


    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "entry",
                "atr",
                "don_break_len",
                "don_break_level",
                "pullback_type",
                "entry_rule",
                "vol_spike",
                "rs_pct",
            ]
        ).astype({"timestamp": "datetime64[ns, UTC]"})

    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out



# ---------------- Orchestrator ----------------

def run_scout() -> int:
    symbols = get_symbols_from_file()
    rs_table = build_weekly_rs(symbols) if cfg.RS_ENABLED else None

    # Liquidity filter from RS weekly table
    liq_min = float(getattr(cfg, "RS_LIQ_MIN_USD_24H", 0.0))
    if rs_table is not None and liq_min > 0:
        # Instead of using only the last weekly liquidity value, which can
        # incorrectly exclude symbols that were liquid earlier in the window
        # (but illiquid by END_DATE), we use the *max* over the backtest window.
        liq_by_symbol = rs_table.groupby("symbol")["usd_vol_med_24h"].max()
        keep = liq_by_symbol[liq_by_symbol >= liq_min].index.tolist()
        symbols_before = len(symbols)
        symbols = [s for s in symbols if s in keep]
        print(
            f"[RS] Liquidity filter kept {len(symbols)} / {symbols_before} "
            f"symbols (≥ {liq_min:,.0f} USD median 24h at some point in window)."
        )

    # Prepare output dir: partitioned dataset under signals/symbol=*
    if getattr(cfg, "SCOUT_CLEAN_OUTPUT_DIR", True) and cfg.SIGNALS_DIR.exists():
        import shutil
        shutil.rmtree(cfg.SIGNALS_DIR, ignore_errors=True)
        cfg.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


    total = 0
    workers = int(getattr(cfg, "N_WORKERS", 1))
    backend = str(getattr(cfg, "SCOUT_BACKEND", "process")).lower()
    Exec = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor

    if workers <= 1:
        for s in tqdm(symbols, desc="Scouting"):
            total += _process_one_symbol(s, rs_table)
            gc.collect()
    else:
        with Exec(max_workers=workers) as ex:
            futs = {ex.submit(_process_one_symbol, s, rs_table): s for s in symbols}
            for f in tqdm(as_completed(futs), total=len(futs), desc="Scouting (parallel)"):
                try:
                    total += int(f.result())
                except Exception as e:
                    print(f"[scout] {futs[f]} failed: {e}")

    print(f"[stream] Wrote ~{total:,} rows to {cfg.SIGNALS_DIR}/symbol=*/part-*.parquet")
    return total

if __name__ == "__main__":
    n = run_scout()
    print(f"Signals written: {n} → {cfg.SIGNALS_DIR}/symbol=*/")