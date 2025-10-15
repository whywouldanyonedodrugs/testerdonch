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

    # Cleanups: keep finite only
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


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

    # ETH 4h MACD histogram (align on columns only)
    try:
        if getattr(cfg, "REGIME_FILTER_ENABLED", True):
            eth = load_parquet_data(
                cfg.REGIME_ASSET,
                start_date=cfg.START_DATE,
                end_date=cfg.END_DATE,
                drop_last_partial=True,
                columns=["open","high","low","close","volume"],
            )
            if not eth.empty:
                eth = _normalize_ts_index(eth)
                eth4 = resample_ohlcv(eth, cfg.REGIME_TIMEFRAME)
                hist4 = macd_hist(eth4["close"], cfg.REGIME_MACD_FAST, cfg.REGIME_MACD_SLOW, cfg.REGIME_MACD_SIGNAL)
                tsu = np.unique(np.sort(sig["timestamp"].values))
                hmap = hist4.reindex(tsu, method="ffill").rename("eth_macd_hist_4h").to_frame()
                hmap = hmap.rename_axis("timestamp").reset_index()
                sig = sig.merge(hmap, on="timestamp", how="left", copy=False)
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

def _weekly_row_for_symbol(sym: str) -> pd.DataFrame:
    df5 = load_parquet_data(sym, start_date=cfg.START_DATE, end_date=cfg.END_DATE,
                            drop_last_partial=True, columns=["open","high","low","close","volume"])
    if df5.empty: return pd.DataFrame()
    daily_close = df5["close"].resample("1D").last().dropna()
    daily_ret = daily_close.pct_change()
    daily_vol = daily_ret.rolling(7, min_periods=5).std()
    usd = (df5["close"]*df5["volume"]).resample("1D").sum()
    usd_med = usd.rolling(30, min_periods=10).median()
    week_ix = daily_close.resample("W-MON", label="left", closed="left").last().index
    week = pd.DataFrame(index=week_ix)
    week["ret_1w"] = daily_close.resample("W-MON", label="left", closed="left").last().pct_change()
    week["vol_1w"] = daily_vol.resample("W-MON", label="left", closed="left").last()
    week["usd_vol_med_24h"] = usd_med.resample("W-MON", label="left", closed="left").last()
    week["symbol"] = sym
    return week.reset_index(names="week_start")

def build_weekly_rs(symbols: List[str]) -> pd.DataFrame:
    rows = []
    workers = int(getattr(cfg, "N_WORKERS", 1))
    backend = str(getattr(cfg, "SCOUT_BACKEND", "thread")).lower()
    Exec = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    if workers <= 1:
        for s in tqdm(symbols, desc="RS weekly"):
            r = _weekly_row_for_symbol(s); 
            if not r.empty: rows.append(r)
    else:
        with Exec(max_workers=workers) as ex:
            futs = {ex.submit(_weekly_row_for_symbol, s): s for s in symbols}
            for f in tqdm(as_completed(futs), total=len(futs), desc="RS weekly (parallel)"):
                try:
                    r = f.result()
                except Exception as e:
                    print(f"[RS] {futs[f]} failed: {e}"); continue
                if not r.empty: rows.append(r)
    if not rows:
        rs = pd.DataFrame(columns=["week_start","symbol","ret_1w","vol_1w","rs_raw","rs_pct","usd_vol_med_24h"])
    else:
        rs = pd.concat(rows, ignore_index=True).dropna(subset=["ret_1w"])
        rs["rs_raw"] = np.where(rs["vol_1w"].gt(0), rs["ret_1w"]/rs["vol_1w"], rs["ret_1w"])
        rs["rs_rank"] = rs.groupby("week_start")["rs_raw"].rank(method="average", ascending=False)
        counts = rs.groupby("week_start")["rs_raw"].transform("count")
        rs["rs_pct"] = (100.0 * (1.0 - (rs["rs_rank"] - 1) / counts.clip(lower=1))).clip(0,100)
    pq.write_table(pa.Table.from_pandas(rs), cfg.RESULTS_DIR / "rs_weekly.parquet")
    return rs

def rs_lookup(rs_table: pd.DataFrame, sym: str, ts: pd.Timestamp) -> float | None:
    rs_sym = rs_table[rs_table["symbol"] == sym]
    if rs_sym.empty: return None
    candidates = rs_sym[rs_sym["week_start"] <= ts.floor("D")]
    if candidates.empty: return None
    return float(candidates.iloc[-1]["rs_pct"])

# ---------------- Per-symbol signal detection ----------------

def _build_feature_panel(sym: str) -> pd.DataFrame:
    df5 = load_parquet_data(
        sym,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        drop_last_partial=True,
        columns=["open","high","low","close","volume","open_interest","funding_rate"]
    )
    if df5.empty:
        return pd.DataFrame(columns=["timestamp","symbol"])

    # NEW: normalize the 5m frame (DatetimeIndex named 'timestamp', no 'timestamp' column)
    df5 = _normalize_ts_index(df5)

    # 1h context features
    df1h = resample_ohlcv(df5, "1h")
    atr1h = atr(df1h, cfg.ATR_LEN)
    rsi1h = rsi(df1h["close"], cfg.RSI_LEN)
    adx1h = adx(df1h, cfg.ADX_LEN)

    # Map back to 5m timestamps (forward-fill)
    atr1h_5m = map_to_left_index(df5.index, atr1h)
    rsi1h_5m = map_to_left_index(df5.index, rsi1h)
    adx1h_5m = map_to_left_index(df5.index, adx1h)

    # Volume structure (5m bars/day ≈ 288)
    bars_per_day = 288
    vol_mult = rolling_median_multiple(df5["volume"], lookback_bars=bars_per_day * cfg.VOL_LOOKBACK_DAYS)

    # ATR % of price
    atr_pct = (atr1h_5m / df5["close"]).astype(float)

    # OI & Funding features (compute on a temp frame with 'timestamp' as a column)
    tmp = pd.DataFrame({
        "timestamp": df5.index,
        "close": df5["close"].values,
        "volume": df5["volume"].values,
        "open_interest": df5["open_interest"].values if "open_interest" in df5.columns else np.nan,
        "funding_rate": df5["funding_rate"].values if "funding_rate" in df5.columns else np.nan,
    })
    oi_feat = add_oi_funding_features(tmp) if ("open_interest" in df5.columns or "funding_rate" in df5.columns) else pd.DataFrame()

    # Return as plain columns (no set_index here — we’ll merge on columns later)
    feat = pd.DataFrame({
        "timestamp": df5.index,
        "symbol": sym,
        "atr_1h": atr1h_5m.values,
        "rsi_1h": rsi1h_5m.values,
        "adx_1h": adx1h_5m.values,
        "vol_mult": vol_mult.values,
        "atr_pct": atr_pct.values,
        "close": df5["close"].values,
        "don_upper": np.nan,
    })

    if not oi_feat.empty:
        # keep only the known engineered columns if present
        for c in [
            "oi_level","oi_notional_est","oi_pct_1h","oi_pct_4h","oi_pct_1d",
            "oi_z_7d","oi_chg_norm_vol_1h","oi_price_div_1h",
            "funding_rate","funding_abs","funding_z_7d","funding_rollsum_3d","funding_oi_div"
        ]:
            if c in oi_feat.columns:
                feat[c] = oi_feat[c].values

    return feat



def _eth_macd_hist_4h_to_5m() -> pd.Series:
    eth = load_parquet_data(cfg.REGIME_ASSET,
                            start_date=cfg.START_DATE, end_date=cfg.END_DATE,
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
    df = load_parquet_data(sym, start_date=cfg.START_DATE, end_date=cfg.END_DATE,
                           drop_last_partial=True, columns=["open","high","low","close","volume"])
    if df.empty: return pd.DataFrame()

    df = _normalize_ts_index(df)

    # Donchian upper (days vs bars) --- no look-ahead on days
    if cfg.DONCH_BASIS.lower() == "days":
        don_up = donchian_upper_days_no_lookahead(df["high"], int(cfg.DON_N_DAYS))
        # pullback window in HOURS for a days-based breakout (configurable)
        pb_hours = int(getattr(cfg, "PULLBACK_WINDOW_HOURS", 24))
        M = max(1, (pb_hours * 60) // 5)  # 5m bars per hour
    else:
        don_up = donchian_upper_bars(df["high"], int(cfg.DON_N_BARS))
        M = int(cfg.PULLBACK_WINDOW_BARS)

    df["don_up"] = don_up

    # Volume spike (multiple or quantile)
    bars_per_day = 288  # 5m bars/day
    lookback_bars = max(1, int(cfg.VOL_LOOKBACK_DAYS) * bars_per_day)
    if cfg.VOL_SPIKE_ENABLED:
        if cfg.VOL_SPIKE_MODE == "multiple":
            vspike = volume_spike_multiple(df["volume"], lookback_bars, float(cfg.VOL_MULTIPLE))
        else:
            vspike = volume_spike_quantile(df["volume"], lookback_bars, float(cfg.VOL_QUANTILE_Q))
    else:
        vspike = pd.Series(0, index=df.index, dtype=np.uint8)

    # ATR & EMA band for pullback logic (works whether 'atr' supports timeframe or not)
    tf = getattr(cfg, "ATR_TIMEFRAME", None)
    if tf:  # compute ATR on the higher TF and forward-fill to 5m
        dft = resample_ohlcv(df, str(tf))
        atr_tf = atr(dft, int(getattr(cfg, "ATR_LEN", 14)))  # no 'timeframe' kw here
        df["atr"] = atr_tf.reindex(df.index, method="ffill")
    else:
        df["atr"] = atr(df, int(getattr(cfg, "ATR_LEN", 14)))
    ema20 = ema(df["close"], cfg.MEAN_MA_LEN)
    band = cfg.MEAN_BAND_ATR_MULT * df["atr"]

    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    don   = df["don_up"].to_numpy(dtype=float)
    vsp   = vspike.to_numpy(dtype=np.uint8)
    atrv  = df["atr"].to_numpy(dtype=float)
    emaa  = ema20.to_numpy(dtype=float)
    bandv = band.to_numpy(dtype=float)

    entry_rule = 1 if cfg.ENTRY_RULE == "rebreak_high" else 2
    pb_model = 1 if cfg.PULLBACK_MODEL == "retest" else 2
    brk_close = 1 if cfg.DON_CONFIRM_CLOSE_ABOVE else 0
    eps = float(cfg.RETEST_EPS_PCT)

    rows = []
    N = len(df)
    for i in range(N):
        du = don[i]
        if not np.isfinite(du): continue

        # breakout: price pierced prior N-day high (optionally require close above)
        if high[i] <= du: 
            continue
        if brk_close and not (close[i] >= du):
            continue
        if cfg.VOL_SPIKE_ENABLED and vsp[i] == 0:
            continue

        # pullback window after breakout
        j_end = min(N-1, i + M)
        touched = False; touch_i = -1
        if pb_model == 1:  # retest near the breakout level
            thr = du * (1.0 + eps)
            for j in range(i+1, j_end+1):
                if low[j] <= thr:
                    touched=True; touch_i=j; break
        else:  # mean reversion to EMA±ATR band
            for j in range(i+1, j_end+1):
                if abs(close[j]-emaa[j]) <= bandv[j]:
                    touched=True; touch_i=j; break
        if not touched: 
            continue

        # entry trigger after pullback: re-break high or close-above
        ent = -1; end2 = min(N-1, touch_i + 3*M)  # give more time after touch
        if entry_rule == 1:  # rebreak of max since breakout
            m = np.nanmax(high[i:j_end+1])
            for k in range(touch_i, end2+1):
                if high[k] > m: ent = k; break
        else:                # close above donch again
            for k in range(touch_i, end2+1):
                if close[k] > du: ent = k; break
        if ent == -1: 
            continue

        ts = df.index[ent]
        rs_pct = rs_lookup(rs_table, sym, ts) if (cfg.RS_ENABLED and rs_table is not None) else None
        rows.append({
            "timestamp": ts,
            "symbol": sym,
            "entry": float(close[ent]),
            "atr": float(atrv[ent]) if np.isfinite(atrv[ent]) else np.nan,
            "don_break_len": int(cfg.DON_N_DAYS if cfg.DONCH_BASIS=='days' else cfg.DON_N_BARS),
            "don_break_level": float(du),
            "pullback_type": "retest" if pb_model==1 else "mean",
            "entry_rule": cfg.ENTRY_RULE,
            "vol_spike": bool(vsp[ent]),
            "rs_pct": rs_pct,
        })

    if not rows: 
        return pd.DataFrame(columns=["timestamp","symbol","entry","atr","don_break_len",
                                     "don_break_level","pullback_type","entry_rule",
                                     "vol_spike","rs_pct"]).astype({"timestamp":"datetime64[ns, UTC]"})
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
        last_liq = (rs_table.sort_values("week_start").groupby("symbol")["usd_vol_med_24h"].last())
        keep = last_liq[last_liq >= liq_min].index.tolist()
        symbols = [s for s in symbols if s in keep]
        print(f"[RS] Liquidity filter kept {len(symbols)} symbols (≥ {liq_min:,.0f} USD median 24h).")

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






    