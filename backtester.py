# backtester.py
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import time

from tqdm import tqdm

import json
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config as cfg
from indicators import resample_ohlcv, atr
from bt_intrabar import resolve_first_touch_1m
from shared_utils import load_parquet_data
from fill_entry_quality_features import compute_entry_quality_panel

from collections import OrderedDict, deque
import heapq
from itertools import count
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)

# Optional online meta scorer (LightGBM + calibration), used only if enabled in config
try:
    from bt_meta_online import score_signal_with_meta
except Exception:
    score_signal_with_meta = None  # type: ignore

pd.set_option("future.no_silent_downcasting", True)

from regime_detector import compute_daily_combined_regime, DailyRegimeConfig, compute_markov_regime_4h
from config import META_PROB_THRESHOLD, META_SIZING_ENABLED, RISK_PCT

class _SigFileStream:
    """
    Stream rows from ONE parquet file in small batches.
    Assumes rows are already in chronological order inside the file (scout writes that way).
    """
    def __init__(self, path: Path, symbol: str, batch_size: int = 5000):
        self.path = Path(path)
        self.symbol = symbol
        self.batch_size = int(batch_size)

        self._pf = pq.ParquetFile(str(self.path))
        self._batches = self._pf.iter_batches(batch_size=self.batch_size)

        self._df = None
        self._i = 0
        self._ts = None  # numpy/pandas vector of timestamps in current batch
        self._load_next_batch()

    def _load_next_batch(self) -> bool:
        try:
            batch = next(self._batches)
        except StopIteration:
            self._df = None
            self._ts = None
            self._i = 0
            return False

        df = batch.to_pandas()
        # parquet parts from scout may not contain "symbol" column (partitioned dir); restore it
        if "symbol" not in df.columns:
            df["symbol"] = self.symbol

        # normalize timestamp column to tz-aware UTC once per batch
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        self._df = df.reset_index(drop=True)
        self._i = 0
        self._ts = self._df["timestamp"].to_numpy() if "timestamp" in self._df.columns else None
        return True

    def peek_key(self):
        """Return a comparable heap key for the next row, or None if exhausted."""
        if self._df is None or self._ts is None or self._i >= len(self._df):
            return None
        ts = self._ts[self._i]
        if pd.isna(ts):
            # If timestamp is missing, treat as +inf so it sinks (or you can skip)
            return None
        # heap key must be comparable across streams
        return int(pd.Timestamp(ts).value)

    def pop_row(self):
        """Pop the next row (pandas Series) and advance."""
        if self._df is None or self._i >= len(self._df):
            return None
        row = self._df.iloc[self._i]
        self._i += 1
        if self._i >= len(self._df):
            self._load_next_batch()
        return row


def iter_partitioned_signals_sorted(signals_dir: Path, batch_size: int = 5000):
    """
    K-way merge across symbol-partitioned parquet files:
      signals_dir/symbol=XYZ/*.parquet
    Yields pandas Series rows in global timestamp order with low memory use.
    """
    signals_dir = Path(signals_dir)

    files = list(signals_dir.glob("symbol=*/*.parquet"))
    if not files:
        # fallback: maybe it's just a bunch of parquet files without partition dirs
        files = list(signals_dir.glob("*.parquet"))

    streams = []
    for f in sorted(files):
        # infer symbol from folder name "symbol=XYZ"
        sym = None
        parent = f.parent.name
        if parent.startswith("symbol="):
            sym = parent.split("=", 1)[1]
        if sym is None:
            sym = "UNKNOWN"

        st = _SigFileStream(f, sym, batch_size=batch_size)
        key = st.peek_key()
        if key is not None:
            streams.append(st)

    heap = []
    seq = count()
    for st in streams:
        key = st.peek_key()
        if key is not None:
            heapq.heappush(heap, (key, next(seq), st))

    while heap:
        _, _, st = heapq.heappop(heap)
        row = st.pop_row()
        if row is None:
            continue
        yield row

        key2 = st.peek_key()
        if key2 is not None:
            heapq.heappush(heap, (key2, next(seq), st))


def _to_utc_dt_index(idx) -> pd.DatetimeIndex:
    """Convert index to tz-aware UTC DatetimeIndex, handling PeriodIndex safely."""
    if isinstance(idx, pd.PeriodIndex):
        out = idx.to_timestamp(how="start")  # start-of-day timestamps
        if out.tz is None:
            out = out.tz_localize("UTC")
        else:
            out = out.tz_convert("UTC")
        return pd.DatetimeIndex(out)

    out = pd.to_datetime(idx, utc=True, errors="coerce")
    if not isinstance(out, pd.DatetimeIndex):
        out = pd.DatetimeIndex(out)
    return out


def _normalize_daily_regime_df(dr: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily regime DF so lookups are reliable."""
    if dr is None or not isinstance(dr, pd.DataFrame) or dr.empty:
        return pd.DataFrame()

    dr = dr.copy()

    # If the function returned a date-like column instead of a DatetimeIndex, use it.
    if not isinstance(dr.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        for c in ("date", "day", "dt", "timestamp", "ts"):
            if c in dr.columns:
                tmp = pd.to_datetime(dr[c], utc=True, errors="coerce")
                if tmp.notna().any():
                    dr = dr.set_index(tmp)
                # keep column if you want; not needed for lookup
                break

    # Flatten MultiIndex columns if present
    if isinstance(dr.columns, pd.MultiIndex):
        dr.columns = [
            "_".join([str(x) for x in tup if x is not None and str(x) != ""])
            for tup in dr.columns
        ]

    # Normalize column names to lowercase (tolerate different versions)
    dr.columns = [str(c).strip() for c in dr.columns]
    dr.rename(columns={c: c.lower() for c in dr.columns}, inplace=True)

    # Canonicalize common variants -> canonical lowercase names
    rename_map = {
        # trend
        "trend_regime_1d": "trend_regime",
        "trendregime": "trend_regime",
        "trendregime1d": "trend_regime",
        # vol
        "vol_regime_1d": "vol_regime",
        "volregime": "vol_regime",
        "volregime1d": "vol_regime",
        # vol prob low
        "vol_prob_low_1d": "vol_prob_low",
        "volproblow1d": "vol_prob_low",
        "volproblow": "vol_prob_low",
        # regime code
        "regime_code_1d": "regime_code",
        "regimecode1d": "regime_code",
        "regimecode": "regime_code",
        # regime string
        "regime_1d": "regime",
        "regimestr1d": "regime",
        "regime_str": "regime",
    }
    for src, dst in rename_map.items():
        if src in dr.columns and dst not in dr.columns:
            dr.rename(columns={src: dst}, inplace=True)

    # Normalize index to tz-aware UTC, drop NaT, sort
    dr.index = _to_utc_dt_index(dr.index)
    dr = dr[~dr.index.isna()].sort_index()

    return dr



def _attach_meta_if_needed(sig: pd.DataFrame) -> pd.DataFrame:
    # Only attach if user enabled meta gating/sizing and column not already present
    need_meta = (getattr(cfg, "META_PROB_THRESHOLD", None) is not None) or bool(getattr(cfg, "META_SIZING_ENABLED", False))
    if (not need_meta) or ("meta_p" in sig.columns):
        return sig

    pred_path = getattr(cfg, "META_PRED_PATH", None)
    if pred_path is None or not Path(pred_path).exists():
        print("[meta] predictions file not found; skipping meta gating/sizing merge")
        return sig

    pred = pd.read_parquet(pred_path)
    if pred.empty:
        print("[meta] predictions file empty; skipping merge")
        return sig

    # --- normalize predictions to (ts, sym, meta_p)
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        ts = next((cols[k] for k in ["entry_ts","timestamp","ts","time","dt"] if k in cols), None)
        sy = next((cols[k] for k in ["symbol","asset","pair","ticker"] if k in cols), None)
        pr = next((cols[k] for k in ["meta_p","y_proba","proba","prob","p","y_pred","pred_proba"] if k in cols), None)
        if ts is None or sy is None or pr is None:
            raise ValueError("[meta] Could not locate ts/symbol/prob columns in predictions")
        out = df[[ts, sy, pr]].rename(columns={ts:"ts", sy:"sym", pr:"meta_p"}).copy()
        out["ts"]  = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        out["sym"] = out["sym"].astype(str)
        out = out.dropna(subset=["ts","sym","meta_p"])
        return out

    pred = _norm(pred)

    # --- normalize signals
    s_ts = "timestamp" if "timestamp" in sig.columns else ("entry_ts" if "entry_ts" in sig.columns else None)
    if s_ts is None:
        raise ValueError("[meta] signals are missing a timestamp column ('timestamp' or 'entry_ts')")

    sig = sig.copy()
    sig["ts"]  = pd.to_datetime(sig[s_ts], utc=True, errors="coerce")
    sig["sym"] = sig["symbol"].astype(str)

    # optional rounding & tolerance
    round_to = getattr(cfg, "META_MERGE_ROUND", "5min") or None
    tol      = getattr(cfg, "META_MERGE_TOL",  "10min") or None
    if round_to:
        sig["ts"]  = sig["ts"].dt.floor(round_to)
        pred["ts"] = pred["ts"].dt.floor(round_to)

    # --- critical: ensure GLOBAL sort by 'ts' (then 'sym' for stability)
    sig  = sig.dropna(subset=["ts","sym"]).sort_values(["ts","sym"], kind="mergesort", ignore_index=True)

    # de-dup predictions and ensure proper ordering
    pred = (pred.dropna(subset=["ts","sym","meta_p"])
                 .groupby(["ts","sym"], as_index=False)["meta_p"].mean()
                 .sort_values(["ts","sym"], kind="mergesort", ignore_index=True))

    # asof-merge by symbol with tolerance
    merged = pd.merge_asof(
        sig, pred, on="ts", by="sym",
        tolerance=(pd.Timedelta(tol) if tol else None),
        direction="backward",
        allow_exact_matches=True,
    )

    # leave NaNs if no match; gating/sizing will handle them
    n = merged["meta_p"].notna().sum()
    print(f"[meta] attached probabilities for {n} / {len(merged)} signals")
    return merged


# ---------------- I/O helpers ----------------

def _size_from_risk_with_portfolio_caps(
    entry_price: float,
    sl_price: float,
    equity_at_entry: float,
    risk_mode: str,
    risk_pct: float,
    fixed_risk_cash: float,
    fee_rate: float,
    notional_cap_pct_of_equity: float,
    max_leverage: float,
    # portfolio state (provide at call site)
    current_open_positions: list[dict],
    portfolio_risk_cap_pct: float,
    gross_exposure_cap_mult: float,
    on_cap_breach: str = "scale",
) -> Tuple[float, float]:
    """
    Returns (qty, per_trade_risk_cash_after_caps)

    current_open_positions: list of dicts with at least:
      {"entry": float, "sl": float, "qty": float}
    """
    # --- per-trade base risk
    unit_risk = max(entry_price - sl_price, 0.0)
    if unit_risk <= 0:
        return 0.0, 0.0

    if risk_mode == "percent":
        base_risk_cash = equity_at_entry * float(risk_pct)
    else:
        base_risk_cash = float(fixed_risk_cash)

    qty = base_risk_cash / unit_risk

    # --- per-trade notional & leverage caps
    notional = qty * entry_price
    max_notional = equity_at_entry * notional_cap_pct_of_equity * max_leverage  # e.g., 2.5x equity with 0.25*10
    if max_notional > 0 and notional > max_notional:
        qty *= max_notional / max(1e-12, notional)
        notional = qty * entry_price  # refresh

    # --- portfolio-level caps (gross exposure & risk)
    # current totals
    gross_now = float(sum(abs(pos["qty"]) * pos["entry"] for pos in current_open_positions))
    risk_now  = float(sum(max(pos["entry"] - pos["sl"], 0.0) * pos["qty"] for pos in current_open_positions))

    # proposed totals with this new trade
    gross_cap = equity_at_entry * gross_exposure_cap_mult
    risk_cap  = equity_at_entry * portfolio_risk_cap_pct

    gross_next = gross_now + abs(qty) * entry_price
    risk_next  = risk_now  + qty * unit_risk

    scale = 1.0
    if gross_cap > 0 and gross_next > gross_cap:
        scale = min(scale, gross_cap / max(gross_next, 1e-12))
    if risk_cap > 0 and risk_next > risk_cap:
        scale = min(scale, risk_cap / max(risk_next, 1e-12))

    if scale < 1.0:
        if on_cap_breach == "skip":
            return 0.0, 0.0
        qty *= scale

    # fees don’t change qty here; they’re handled on fills/exits
    final_risk_cash = qty * unit_risk
    return float(qty), float(final_risk_cash)

def _infer_breakout_anchor_ts(
    df5: pd.DataFrame,
    entry_ts: pd.Timestamp,
    don_break_level: float | None,
    lookback_days: int = 40
) -> pd.Timestamp:
    """
    Infer the breakout bar timestamp (anchor) on the 5m stream when a Donchian
    breakout first occurred, using the breakout level carried in the signal.
    If not found, fall back to entry_ts.

    Assumes df5.index is tz-aware (UTC). Safe against empty segments or NaNs.
    """
    # If no usable level, fallback immediately
    try:
        if don_break_level is None or not np.isfinite(don_break_level):
            return entry_ts
    except Exception:
        return entry_ts

    start_ts = entry_ts - pd.Timedelta(days=lookback_days)
    seg = df5.loc[(df5.index >= start_ts) & (df5.index <= entry_ts)]
    if seg.empty:
        return entry_ts

    # Close cross above breakout (first bar where close >= level and previous < level)
    crossed = (seg["close"] >= don_break_level) & (seg["close"].shift(1) < don_break_level)
    if crossed.any():
        # idxmax returns the LABEL (timestamp) of the first True (the max), not a position
        return crossed.idxmax()

    # Fallback: first bar whose HIGH touched/exceeded the level
    touched_mask = seg["high"] >= don_break_level
    if touched_mask.any():
        return seg.index[touched_mask.argmax()]  # or: touched_mask[touched_mask].index[0]

    # Nothing found; anchor at entry
    return entry_ts



def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns=["timestamp"])
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()

def _prob_size_multiplier(p: float, p0: float = 0.50, lo: float = 0.50, hi: float = 2.00) -> float:
    """Monotone map p∈[0,1] -> [lo,hi]. Centered at p0."""
    try:
        p = float(p)
    except Exception:
        return 1.0
    if not (0.0 <= p <= 1.0):
        return 1.0
    if p <= p0:
        # scale from 0..p0 to lo..1
        return lo + (1.0 - lo) * (p / max(p0, 1e-9))
    # scale from p0..1 to 1..hi
    return 1.0 + (hi - 1.0) * ((p - p0) / max(1.0 - p0, 1e-9))


def _dyn_exit_multipliers(row: pd.Series, cfg) -> Tuple[float, float]:
    if not getattr(cfg, "DYN_EXITS_ENABLED", False):
        return 1.0, 1.0

    try:
        hist = float(row.get("eth_macd_hist_4h", np.nan))
    except Exception:
        hist = np.nan
    try:
        both_pos = int(row.get("eth_macd_both_pos_4h", 0))
    except Exception:
        both_pos = 0

    if not np.isfinite(hist):
        return 1.0, 1.0

    thresh = float(getattr(cfg, "DYN_MACD_HIST_THRESH", 0.0))
    if hist >= thresh and both_pos == 1:
        return float(getattr(cfg, "DYN_TP_MULT_POS", 1.0)), float(getattr(cfg, "DYN_SL_MULT_POS", 1.0))
    if hist < thresh:
        return float(getattr(cfg, "DYN_TP_MULT_NEG", 1.0)), float(getattr(cfg, "DYN_SL_MULT_NEG", 1.0))
    return 1.0, 1.0


def _dyn_size_multiplier(prob: float, hist_4h: float, cfg) -> float:
    base = 1.0
    if getattr(cfg, "META_SIZING_ENABLED", False):
        p0 = float(getattr(cfg, "META_SIZING_P0", 0.60))
        p1 = float(getattr(cfg, "META_SIZING_P1", 0.95))
        lo = float(getattr(cfg, "META_SIZING_MIN", 0.5))
        hi = float(getattr(cfg, "META_SIZING_MAX", 2.0))
        try:
            p = float(prob)
        except Exception:
            p = np.nan
        if np.isfinite(p):
            if p <= p0:
                base = lo
            elif p >= p1:
                base = hi
            else:
                base = lo + (hi - lo) * ((p - p0) / max(p1 - p0, 1e-9))

    try:
        hist = float(hist_4h)
    except Exception:
        hist = np.nan
    regime_mult = 1.0
    if np.isfinite(hist) and hist < float(getattr(cfg, "DYN_MACD_HIST_THRESH", 0.0)):
        regime_mult = float(getattr(cfg, "REGIME_DOWNSIZE_MULT", 1.0))

    out = base * regime_mult
    out = float(np.clip(out, float(getattr(cfg, "SIZE_MIN_CAP", 0.1)), float(getattr(cfg, "SIZE_MAX_CAP", 5.0))))
    return out


def _is_trade_week(ts: pd.Timestamp, pattern: str) -> bool:
    if not pattern or set(pattern) <= {"1"}:
        return True
    iso_week = int(ts.isocalendar().week)
    i = (iso_week - 1) % len(pattern)
    return pattern[i] == "1"


def _must_flatten_now(ts: pd.Timestamp, cfg) -> bool:
    if not getattr(cfg, "WEEK_PATTERN_ENABLED", False):
        return False
    return bool(ts.weekday() == 4 and ts.time() >= time(23, 55))

# ---------------- Regime gate (ETH MACD on configured TF) ----------------
class RegimeGate:
    """
    Computes MACD/Signal/Histogram for cfg.REGIME_ASSET on cfg.REGIME_TIMEFRAME,
    caches it, and provides an as-of boolean 'up' gate.
    """
    def __init__(self):
        self._cache_tf: Dict[str, pd.DataFrame] = {}


    def _ensure_eth_tf(self, timeframe: str) -> pd.DataFrame:
        if timeframe in self._cache_tf:
            return self._cache_tf[timeframe]

        # Load base (5m) ETH and resample to the configured TF (e.g. "4h")
        eth5 = load_parquet_data(
            cfg.REGIME_ASSET,
            start_date=cfg.START_DATE,
            end_date=cfg.END_DATE,
            drop_last_partial=True,
            columns=["open", "high", "low", "close", "volume"],
        )
        if eth5.empty:
            self._cache_tf[timeframe] = pd.DataFrame()
            return self._cache_tf[timeframe]

        eth_tf = resample_ohlcv(eth5, timeframe)

        # MACD on TF close
        fast, slow, sig = cfg.REGIME_MACD_FAST, cfg.REGIME_MACD_SLOW, cfg.REGIME_MACD_SIGNAL
        macd_line = eth_tf["close"].ewm(span=fast, adjust=False).mean() - \
                    eth_tf["close"].ewm(span=slow, adjust=False).mean()
        signal = macd_line.ewm(span=sig, adjust=False).mean()
        hist = macd_line - signal

        out = pd.DataFrame(
            {"close": eth_tf["close"].values, "macd": macd_line.values,
             "signal": signal.values, "hist": hist.values},
            index=eth_tf.index
        ).dropna()
        self._cache_tf[timeframe] = out
        return out

    def is_up(self, ts: pd.Timestamp) -> bool:
        if not getattr(cfg, "REGIME_FILTER_ENABLED", True):
            return True
        tf = str(getattr(cfg, "REGIME_TIMEFRAME", "4h"))
        df = self._ensure_eth_tf(tf)
        if df.empty:
            return True
        row = df.loc[:ts].iloc[-1:]  # as-of
        if row.empty:
            return True
        macd_up = bool(row["macd"].iloc[0] > row["signal"].iloc[0])
        hist_up = bool(row["hist"].iloc[0] > 0)
        both = bool(getattr(cfg, "REGIME_REQUIRE_BOTH_POSITIVE", True))
        return (macd_up and hist_up) if both else hist_up

    def get_hist_and_slope(self, ts: pd.Timestamp) -> tuple[float | None, float | None]:
        """
        Return (hist_now, hist_slope) for the MACD histogram at or before ts
        on cfg.REGIME_TIMEFRAME (e.g. 4h). hist_slope = hist_now - hist_prev.
        If not enough history, returns (None, None).
        """
        tf = str(getattr(cfg, "REGIME_TIMEFRAME", "4h"))
        df = self._ensure_eth_tf(tf)
        if df.empty:
            return None, None

        # as-of locate: last bar at or before ts
        row = df.loc[:ts].iloc[-1:]
        if row.empty:
            return None, None

        hist_now = float(row["hist"].iloc[0])

        # previous bar for slope
        # df.index is a DatetimeIndex with unique 4h timestamps
        idx_label = row.index[0]
        try:
            pos = df.index.get_loc(idx_label)
        except KeyError:
            return hist_now, None

        # get_loc can return int or slice; normalize to int
        if isinstance(pos, slice):
            # take the last index position in that slice
            idx_vals = df.index[pos]
            if len(idx_vals) == 0:
                return hist_now, None
            pos = df.index.get_loc(idx_vals[-1])

        if isinstance(pos, (list, np.ndarray)):
            if len(pos) == 0:
                return hist_now, None
            pos = int(pos[-1])

        if pos == 0:
            # no previous bar
            return hist_now, None

        hist_prev = float(df["hist"].iloc[pos - 1])
        slope = hist_now - hist_prev
        return hist_now, slope



# ---------------- Types ----------------
@dataclass
class Trade:
    trade_id: int
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: Optional[pd.Timestamp]
    entry: float
    exit: Optional[float]
    qty: float
    side: str
    sl: float
    tp: float
    exit_reason: Optional[str]
    atr_at_entry: float
    regime_up: bool
    rs_pct: Optional[float]
    pullback_type: Optional[str]
    entry_rule: Optional[str]
    don_break_len: Optional[int]
    fees: float
    pnl: float
    pnl_R: float

    # --- Diagnostics at trade level ---
    mae_over_atr: Optional[float] = None
    mfe_over_atr: Optional[float] = None

    # --- Regime diagnostics ---
    markov_state_4h: Optional[int] = None
    markov_prob_up_4h: Optional[float] = None
    trend_regime_1d: Optional[str] = None
    vol_regime_1d: Optional[str] = None
    vol_prob_low_1d: Optional[float] = None
    regime_code_1d: Optional[int] = None
    regime_1d: Optional[str] = None

    # --- Snapshot of per-signal features at entry (for offline meta/stat analysis) ---

    # 1h context & Donchian
    atr_1h: Optional[float] = None
    rsi_1h: Optional[float] = None
    adx_1h: Optional[float] = None
    vol_mult: Optional[float] = None
    atr_pct: Optional[float] = None
    days_since_prev_break: Optional[float] = None
    consolidation_range_atr: Optional[float] = None
    prior_1d_ret: Optional[float] = None
    rv_3d: Optional[float] = None
    don_break_level: Optional[float] = None
    don_dist_atr: Optional[float] = None  # (entry - donch_upper) / ATR at entry


    # Additional multi-timeframe asset indicators at entry (Chunk 2)
    asset_rsi_15m: Optional[float] = None
    asset_rsi_4h: Optional[float] = None

    asset_macd_line_1h: Optional[float] = None
    asset_macd_signal_1h: Optional[float] = None
    asset_macd_hist_1h: Optional[float] = None
    asset_macd_slope_1h: Optional[float] = None

    asset_macd_line_4h: Optional[float] = None
    asset_macd_signal_4h: Optional[float] = None
    asset_macd_hist_4h: Optional[float] = None
    asset_macd_slope_4h: Optional[float] = None

    asset_vol_1h: Optional[float] = None
    asset_vol_4h: Optional[float] = None

    gap_from_1d_ma: Optional[float] = None
    prebreak_congestion: Optional[float] = None


    # ETH MACD 4h context merged by _eth_macd_full_4h_to_5m in scout.py
    eth_macd_line_4h: Optional[float] = None
    eth_macd_signal_4h: Optional[float] = None
    eth_macd_hist_4h: Optional[float] = None
    # NEW: ETH MACD histogram slope variants
    eth_macd_hist_slope_4h: Optional[float] = None  # Δhist per 4h bar
    eth_macd_hist_slope_1h: Optional[float] = None  # Δhist per 1h bar
    eth_macd_both_pos_4h: Optional[int] = None


    # Open interest & funding features from add_oi_funding_features(...)
    oi_level: Optional[float] = None
    oi_notional_est: Optional[float] = None
    oi_pct_1h: Optional[float] = None
    oi_pct_4h: Optional[float] = None
    oi_pct_1d: Optional[float] = None
    oi_z_7d: Optional[float] = None
    oi_chg_norm_vol_1h: Optional[float] = None
    oi_price_div_1h: Optional[float] = None

    funding_rate: Optional[float] = None
    funding_abs: Optional[float] = None
    funding_z_7d: Optional[float] = None
    funding_rollsum_3d: Optional[float] = None
    funding_oi_div: Optional[float] = None

    crowded_long: Optional[int] = None
    crowded_short: Optional[int] = None
    crowd_side: Optional[int] = None
    est_leverage: Optional[float] = None

    btc_funding_rate: Optional[float] = None
    btc_oi_z_7d: Optional[float] = None
    btc_vol_regime_level: Optional[float] = None
    btc_trend_slope: Optional[float] = None

    eth_funding_rate: Optional[float] = None
    eth_oi_z_7d: Optional[float] = None
    eth_vol_regime_level: Optional[float] = None
    eth_trend_slope: Optional[float] = None

# ---------------- helpers: risk & intrabar ----------------
# --- sizing ---------------------------------------------------------------
def _size_from_risk(
    entry_price: float,
    sl_price: float,
    equity: float,
    *,
    risk_mode: str | None = None,
    risk_pct: float | None = None,
    fixed_cash: float | None = None,
    notional_cap_pct: float | None = None,
    max_leverage: float | None = None,
) -> float:
    """
    Return position size (qty) from either percent-of-equity or fixed cash risk,
    respecting notional/leverage caps. No gating logic here.
    """
    risk_per_unit = abs(float(entry_price) - float(sl_price))
    if risk_per_unit <= 0:
        return 0.0

    mode = (risk_mode or getattr(cfg, "RISK_MODE", "percent")).lower()
    if mode == "cash":
        cash_risk = float(fixed_cash if fixed_cash is not None else getattr(cfg, "FIXED_RISK_CASH", 10.0))
    else:
        rp = float(risk_pct if risk_pct is not None else getattr(cfg, "RISK_PCT", 0.01))
        cash_risk = rp * float(equity)

    qty = cash_risk / risk_per_unit

    # notional / leverage caps
    cap_pct = float(notional_cap_pct if notional_cap_pct is not None else getattr(cfg, "NOTIONAL_CAP_PCT_OF_EQUITY", 0.25))
    max_lev  = float(max_leverage if max_leverage is not None else getattr(cfg, "MAX_LEVERAGE", 10.0))
    max_notional = float(equity) * cap_pct * max_lev
    notional = qty * float(entry_price)
    if notional > max_notional:
        qty = max_notional / float(entry_price)

    return max(qty, 0.0)


def _resolve_intrabar(symbol: str, bar_ts, levels: dict, bar_ohlc: dict):
    """
    levels: {"sl": price, "tp1": price_or_None, "tp": price, "trail": price_or_None}
    bar_ohlc: {"open":o, "high":h, "low":l, "close":c}
    Returns: one of ("sl","tp1","tp","trail", None)
    """
    h, l = bar_ohlc["high"], bar_ohlc["low"]
    inside = {k: v for k, v in levels.items() if v is not None and (l <= v <= h)}
    if not inside:
        return None

    # Prefer 1m sequencing if configured
    tie = str(getattr(cfg, "TIE_BREAKER", "use_1m"))
    use_1m = bool(getattr(cfg, "USE_INTRABAR_1M", False))
    if tie == "use_1m" and use_1m:
        try:
            return resolve_first_touch_1m(symbol, bar_ts, inside)
        except Exception:
            pass

    # Fallback deterministic policy
    order = ["sl", "tp1", "tp", "trail"] if tie == "sl_wins" else ["tp1", "tp", "trail", "sl"]
    o = bar_ohlc["open"]
    if len(inside) > 1 and tie not in ("sl_wins", "tp_wins"):
        # nearest to open heuristic
        k = min(inside.keys(), key=lambda kk: abs(inside[kk] - o))
        return k
    for k in order:
        if k in inside:
            return k
    return None


def _simulate_long_with_partials_trail(
    symbol: str,
    entry_ts: pd.Timestamp,
    entry_price: float,
    atr_at_entry: float,
    equity_at_entry: float,
    df5: pd.DataFrame,
    *,
    risk_mode_override: str | None = None,
    risk_pct_override: float | None = None,
    fixed_cash_override: float | None = None,
    avwap_anchor_ts: pd.Timestamp | None = None,
    sl_mult_override: float | None = None,
    tp_mult_override: float | None = None,
    av_sl_mult_override: float | None = None,
    av_tp_mult_override: float | None = None,
) -> dict | None:
    
    # --- config
    fee_rate   = float(getattr(cfg, "FEE_RATE", 0.00055))
    time_hours = getattr(cfg, "TIME_EXIT_HOURS", None)
    
    # Spread config
    use_spread = getattr(cfg, "SIMULATE_SPREAD_ENABLED", False)
    spread_pct = float(getattr(cfg, "SPREAD_PCT", 0.001)) if use_spread else 0.0
    
    # legacy ATR-price basis (default)
    sl_mult_legacy = float(sl_mult_override if sl_mult_override is not None else getattr(cfg, "SL_ATR_MULT", 2.0))
    tp_mult_legacy = float(tp_mult_override if tp_mult_override is not None else getattr(cfg, "TP_ATR_MULT", 8.0))

    # AVWAP basis (new)
    exit_basis   = str(getattr(cfg, "EXIT_BASIS", "price_atr")).lower()
    av_mode      = str(getattr(cfg, "AVWAP_MODE", "static")).lower()
    av_anchor    = pd.to_datetime(avwap_anchor_ts or entry_ts, utc=True)
    av_sl_mult   = float(av_sl_mult_override if av_sl_mult_override is not None else getattr(cfg, "AVWAP_SL_MULT", 2.0))
    av_tp_mult   = float(av_tp_mult_override if av_tp_mult_override is not None else getattr(cfg, "AVWAP_TP_MULT", 8.0))
    av_use_entry_atr = bool(getattr(cfg, "AVWAP_USE_ENTRY_ATR", True))

    # partials / trail
    partial_on = bool(getattr(cfg, "PARTIAL_TP_ENABLED", False))
    tp1_mult   = float(getattr(cfg, "PARTIAL_TP1_ATR_MULT", 5.0))
    tp1_ratio  = float(getattr(cfg, "PARTIAL_TP_RATIO", 0.5))
    move_be    = bool(getattr(cfg, "MOVE_SL_TO_BE_ON_TP1", True))

    trail_on   = bool(getattr(cfg, "TRAIL_AFTER_TP1", False))
    trail_mult = float(getattr(cfg, "TRAIL_ATR_MULT", 1.0))
    use_hwm    = bool(getattr(cfg, "TRAIL_USE_HIGH_WATERMARK", True))

    # --- vectorized AVWAP
    hlc3  = (df5["high"].astype(float) + df5["low"].astype(float) + df5["close"].astype(float)) / 3.0
    vol   = df5["volume"].astype(float)
    pv_cum = (hlc3 * vol).cumsum()
    v_cum  = vol.cumsum()

    if av_anchor <= df5.index[0]:
        pv0 = 0.0; v0 = 0.0
    else:
        pre = df5.loc[df5.index < av_anchor]
        if pre.empty: pv0 = 0.0; v0 = 0.0
        else:
            pv0 = float(pv_cum.loc[pre.index[-1]])
            v0  = float(v_cum.loc[pre.index[-1]])

    def avwap_at(ts: pd.Timestamp) -> float:
        upto = df5.loc[:ts]
        if upto.empty: return float(hlc3.iloc[0])
        pv = float(pv_cum.loc[upto.index[-1]]) - pv0
        vv = float(v_cum.loc[upto.index[-1]]) - v0
        if vv <= 0: return float(hlc3.loc[upto.index[-1]])
        return pv / vv

    # --- initial SL/TP levels
    if exit_basis == "avwap_atr":
        av_at_entry = avwap_at(entry_ts)
        atr_ref = atr_at_entry if av_use_entry_atr else float(df5.loc[entry_ts, "atr_pre"])
        sl_initial = av_at_entry - av_sl_mult * atr_ref
        tp         = av_at_entry + av_tp_mult * atr_ref
        tp1        = (av_at_entry + tp1_mult * atr_ref) if partial_on else None
        sl_initial = min(sl_initial, entry_price - 1e-8)
        tp         = max(tp,        entry_price + 1e-8)
        if tp1 is not None: tp1 = max(tp1, entry_price + 1e-8)
        trail_gap  = (trail_mult * atr_ref) if trail_on else None
    else:
        sl_initial = entry_price - sl_mult_legacy * atr_at_entry
        tp         = entry_price + tp_mult_legacy * atr_at_entry
        tp1        = (entry_price + tp1_mult * atr_at_entry) if partial_on else None
        trail_gap  = (trail_mult * atr_at_entry) if trail_on else None

    sl_final = sl_initial

    # --- size from risk
    qty = _size_from_risk(
        entry_price=entry_price,
        sl_price=sl_initial,
        equity=equity_at_entry,
        risk_mode=risk_mode_override,
        risk_pct=risk_pct_override,
        fixed_cash=fixed_cash_override,
        notional_cap_pct=getattr(cfg, "NOTIONAL_CAP_PCT_OF_EQUITY", 0.25),
        max_leverage=getattr(cfg, "MAX_LEVERAGE", 10.0),
    )
    if qty <= 0:
        return None

    # --- Strict Entry Validation (Immediate Stop Check) ---
    entry_bar = df5.loc[entry_ts]
    check_low = float(entry_bar["low"]) * (1 - spread_pct / 2)
    
    if check_low <= sl_initial:
        fees = fee_rate * abs(qty * entry_price) * 2
        exit_px = sl_initial 
        pnl_cash = qty * (exit_px - entry_price) - fees
        return dict(
            exit_ts=entry_ts,
            exit_price=exit_px,
            exit_reason="immediate_sl",
            fees=fees,
            qty_filled=qty,
            pnl_cash=float(pnl_cash),
            sl_initial=sl_initial,
            sl_final=sl_final,
            tp=tp,
            mae_over_atr=0.0,
            mfe_over_atr=0.0,
        )

    fees = fee_rate * abs(qty * entry_price)

    # --- path after entry
    walk = df5.loc[df5.index > entry_ts].copy()
    if walk.empty:
        return None

    took_tp1 = False
    remaining_qty = qty
    highest_since_tp1 = entry_price
    curr_trail = None
    tp1_fill_px: float | None = None

    t_exit: pd.Timestamp | None = None
    px_exit: float | None = None
    reason: str | None = None
    mae = 0.0
    mfe = 0.0

    deadline = entry_ts + pd.Timedelta(hours=float(time_hours)) if time_hours is not None else None

    for ts, row in walk.iterrows():
        o, h, l, c = map(float, (row["open"], row["high"], row["low"], row["close"]))
        
        # Simulate Bid/Ask for checks
        bid_low = l * (1 - spread_pct / 2)
        bid_high = h * (1 - spread_pct / 2)

        # excursions (approx)
        mae = min(mae, bid_low - entry_price)
        mfe = max(mfe, bid_high - entry_price)

        if took_tp1 and trail_on:
            highest_since_tp1 = max(highest_since_tp1, bid_high) if use_hwm else c
            curr_trail = highest_since_tp1 - trail_gap

        # Update levels (Dynamic AVWAP)
        if exit_basis == "avwap_atr" and av_mode == "dynamic":
            atr_ref_bar = (atr_at_entry if av_use_entry_atr else float(row["atr_pre"]))
            av = avwap_at(ts)
            sl_dyn = min(av - av_sl_mult * atr_ref_bar, entry_price - 1e-8)
            tp_dyn = max(av + av_tp_mult * atr_ref_bar, entry_price + 1e-8)
            levels = {"sl": sl_dyn, "tp": tp_dyn}
            if (not took_tp1) and (tp1 is not None):
                tp1_dyn = max(av + tp1_mult * atr_ref_bar, entry_price + 1e-8)
                levels["tp1"] = tp1_dyn
            if took_tp1 and (curr_trail is not None):
                levels["trail"] = curr_trail
        else:
            levels = {"sl": sl_final, "tp": tp}
            if (not took_tp1) and (tp1 is not None):
                levels["tp1"] = tp1
            if took_tp1 and (curr_trail is not None):
                levels["trail"] = curr_trail

        # Check SL (Bid Low <= SL)
        if bid_low <= levels["sl"]:
            t_exit, px_exit, reason = ts, levels["sl"], "sl"
            break
            
        # Check TP (Bid High >= TP)
        if bid_high >= levels["tp"]:
            t_exit, px_exit, reason = ts, levels["tp"], "tp_final"
            break
            
        # Check TP1
        if "tp1" in levels and bid_high >= levels["tp1"]:
            fill_px = float(levels["tp1"])
            take_qty = remaining_qty * tp1_ratio
            remaining_qty -= take_qty
            fees += fee_rate * abs(take_qty * fill_px)
            took_tp1 = True
            tp1_fill_px = fill_px
            highest_since_tp1 = max(highest_since_tp1, bid_high)
            if move_be:
                sl_final = entry_price
            continue
            
        # Check Trail
        if "trail" in levels and bid_low <= levels["trail"]:
            t_exit, px_exit, reason = ts, levels["trail"], "trail"
            break

        # Time Exit
        if deadline is not None and ts >= deadline:
            t_exit, px_exit, reason = ts, c * (1 - spread_pct/2), "time"
            break

    if t_exit is None:
        last_ts = walk.index[-1]
        t_exit, px_exit, reason = last_ts, float(walk.iloc[-1]["close"]) * (1 - spread_pct/2), "time"

    fees += fee_rate * abs(remaining_qty * px_exit)
    pnl_cash = remaining_qty * (px_exit - entry_price)
    if took_tp1 and tp1_fill_px is not None:
        realized_qty = qty * tp1_ratio
        pnl_cash += realized_qty * (tp1_fill_px - entry_price)
    pnl_cash -= fees

    mae_over_atr = abs(mae) / atr_at_entry if atr_at_entry > 0 else np.nan
    mfe_over_atr = abs(mfe) / atr_at_entry if atr_at_entry > 0 else np.nan

    return dict(
        exit_ts=t_exit,
        exit_price=px_exit,
        exit_reason=reason,
        fees=fees,
        qty_filled=qty,
        pnl_cash=float(pnl_cash),
        sl_initial=sl_initial,
        sl_final=sl_final,
        tp=(tp if exit_basis != "avwap_atr" or av_mode == "static" else None),
        mae_over_atr=mae_over_atr,
        mfe_over_atr=mfe_over_atr,
    )


# ---------------- Engine ----------------
class Backtester:

    def _markov_at(self, ts: pd.Timestamp) -> Tuple[Optional[int], Optional[float]]:
        if getattr(self, "_markov_df", None) is None or self._markov_df.empty:
            return None, None
        row = self._markov_df.loc[:ts].iloc[-1:] if ts >= self._markov_df.index[0] else pd.DataFrame()
        if row.empty:
            return None, None
        return int(row["markov_state_up"].iloc[0]), float(row["markov_prob_up"].iloc[0])

    def _daily_regime_at(self, ts: pd.Timestamp):
        dr = getattr(self, "_daily_regime_day", None)
        if dr is None or dr.empty:
            return None, None, None, None, None

        ts = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(ts):
            return None, None, None, None, None

        day = ts.floor("D")
        if day.tz is None:
            day = day.tz_localize("UTC")
        else:
            day = day.tz_convert("UTC")

        if day < dr.index[0]:
            return None, None, None, None, None

        pos = dr.index.searchsorted(day, side="right") - 1
        if pos < 0:
            return None, None, None, None, None

        r = dr.iloc[pos]

        def _pick(keys):
            for k in keys:
                k = k.lower()
                if k in r.index:
                    v = r[k]
                    if v is None or pd.isna(v):
                        continue
                    return v
            return None

        trend = _pick(["trend_regime", "trend"])
        vol = _pick(["vol_regime", "vol"])
        vprob = _pick(["vol_prob_low", "vol_prob"])
        regime_str = _pick(["regime", "regime_str"])
        raw_code = _pick(["regime_code", "code"])

        regime_code = int(raw_code) if raw_code is not None and not pd.isna(raw_code) else None

        return (
            str(trend) if trend is not None else None,
            str(vol) if vol is not None else None,
            float(vprob) if vprob is not None and not pd.isna(vprob) else None,
            regime_code,
            str(regime_str) if regime_str is not None else None,
        )




    def __init__(self, initial_capital: float, risk_pct: float, max_leverage: float):
        self.initial_capital = float(initial_capital)
        self.equity = float(initial_capital)
        self.risk_pct = float(risk_pct)
        self.max_leverage = float(max_leverage)
        self.regime = RegimeGate()

        try:
            # IMPORTANT: do NOT force save_path=None (often results in empty DF)
            dr = compute_daily_combined_regime(DailyRegimeConfig())
            dr = _normalize_daily_regime_df(dr)

            # Build a day-aligned table for robust lookup
            if not dr.empty:
                dr_day = dr.copy()
                dr_day["_day"] = dr_day.index.floor("D")
                dr_day = dr_day.groupby("_day").last()
                dr_day.index = _to_utc_dt_index(dr_day.index)
            else:
                dr_day = pd.DataFrame()

            self._daily_regime = dr
            self._daily_regime_day = dr_day

            if dr.empty:
                print("[backtester] WARNING: daily regime table is EMPTY -> trend_regime_1d will be NaN")
            else:
                print(
                    f"[backtester] daily regime OK: rows={len(dr)} "
                    f"range={dr.index.min()}..{dr.index.max()} "
                    f"cols={list(dr.columns)}"
                )

        except Exception as e:
            print(f"[backtester] daily regime build failed: {e}")
            self._daily_regime = pd.DataFrame()
            self._daily_regime_day = pd.DataFrame()





        self._cache_5m: Dict[str, pd.DataFrame] = {}
        self._cache_1m: Dict[str, pd.DataFrame] = {}
        # --- meta parity caches/state ---
        self._cache_entry_quality = OrderedDict()  # sym -> entry-quality panel
        self._meta_win_hist = deque(maxlen=50)
        self._meta_ewm_win = None
        self._meta_regime_thresholds = None


        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

        # Per-signal decision log (for parity debugging)
        self._decision_log: List[Dict[str, object]] = []
        self._active_trades: List[Trade] = []
        
        # NEW: Lock timeline log (for cooldown debugging)
        self._lock_timeline: List[Dict[str, object]] = []

        # Throughput / de-dup / position-count guards (used by run())
        self._lock_until = {}        # sym -> timestamp until which trading is blocked
        self._open_until = {}        # sym -> timestamp until which position is considered open
        self._cooldown_until = {}    # sym -> timestamp until which cooldown applies
        self._last_entry = {}        # sym -> last entry timestamp (dedup window)
        self._trades_per_day = {}    # day(ts.floor("D")) -> count
        self._skipped = {
            "lock": 0,
            "open": 0,
            "cooldown": 0,
            "daycap": 0,
            "max_open": 0,
        }



        try:
            self._markov4h = compute_markov_regime_4h(
                asset=getattr(cfg, "REGIME_ASSET", "ETHUSDT"),
                timeframe=getattr(cfg, "REGIME_TIMEFRAME", "4h"),
            )
        except Exception as e:
            print(f"[backtester] 4h markov regime build failed: {e}")
            self._markov4h = pd.DataFrame()

    def _markov4h_at(self, ts: pd.Timestamp):
        if getattr(self, "_markov4h", None) is None or self._markov4h.empty:
            return None, None
        df = self._markov4h
        if ts < df.index[0]:
            return None, None
        row = df.loc[:ts].iloc[-1]
        return int(row["state_up"]), float(row["prob_up"])

    # ----- Data caches -----
    def _maybe_downcast_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast numeric columns to float32 to reduce memory.
        Safe for backtesting accuracy in this context.
        """
        if not bool(getattr(cfg, "BT_DOWNCAST_FLOAT32", True)):
            return df

        for c in ("open", "high", "low", "close", "volume", "atr_pre", "open_interest", "funding_rate"):

            if c in df.columns:
                # to_numeric guards against weird dtypes
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        return df

    def _lru_get(self, cache: "OrderedDict[str, pd.DataFrame]", key: str) -> Optional[pd.DataFrame]:
        if key in cache:
            cache.move_to_end(key)  # mark as recently used
            return cache[key]
        return None

    def _lru_put(self, cache: "OrderedDict[str, pd.DataFrame]", key: str, value: pd.DataFrame, max_items: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        # evict least recently used
        while max_items is not None and max_items > 0 and len(cache) > max_items:
            cache.popitem(last=False)

    def _get_5m(self, sym: str) -> pd.DataFrame:
        # Ensure caches exist as OrderedDict even if older code initialized dicts
        if not isinstance(getattr(self, "_cache_5m", None), OrderedDict):
            self._cache_5m = OrderedDict()

        max_items = int(getattr(cfg, "BT_CACHE_5M_MAX_SYMBOLS", 6))

        cached = self._lru_get(self._cache_5m, sym)
        if cached is not None:
            return cached

        # We must load timestamp because resample_ohlcv() uses df["timestamp"]
        # and compute_entry_quality_panel() requires a DatetimeIndex.
        base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        opt_cols = ["open_interest", "funding_rate"]
        want_cols = base_cols + opt_cols

        # Load with optional cols; if not present, load base and add NaNs
        try:
            df = load_parquet_data(
                sym,
                start_date=cfg.START_DATE,
                end_date=cfg.END_DATE,
                drop_last_partial=True,
                columns=want_cols,
            )
        except Exception:
            df = load_parquet_data(
                sym,
                start_date=cfg.START_DATE,
                end_date=cfg.END_DATE,
                drop_last_partial=True,
                columns=base_cols,
            )
            for c in opt_cols:
                if c not in df.columns:
                    df[c] = np.nan

        # FIX: Clear index name to prevent collision with 'timestamp' column during sort/manipulation
        if df.index.name == "timestamp":
            df.index.name = None

        # Guarantee timestamp column exists; if loader returned DatetimeIndex only, reconstruct it
        if "timestamp" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df["timestamp"] = df.index
            else:
                raise KeyError("load_parquet_data returned no 'timestamp' column and index is not DatetimeIndex")

        # Normalize UTC timestamp and set as index (keep the column for resample_ohlcv)
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        
        # FIX: drop=True to avoid ambiguity between index and column
        df = df.set_index("timestamp", drop=True)

        # PRECOMPUTE ATR ON CONFIGURED TF (e.g. "1h") AND FFILL TO 5m
        tf = getattr(cfg, "ATR_TIMEFRAME", None)
        if tf:
            dft = resample_ohlcv(df, str(tf))
            atr_tf = atr(dft, int(getattr(cfg, "ATR_LEN", 14)))
            df["atr_pre"] = atr_tf.reindex(df.index, method="ffill")
        else:
            df["atr_pre"] = atr(df, int(getattr(cfg, "ATR_LEN", 14)))

        df = self._maybe_downcast_ohlcv(df)
        self._lru_put(self._cache_5m, sym, df, max_items)
        return df


    # ============================
    # META: TRAINING FEATURE STORE (replay exact post-filled features)
    # ============================

    def _meta_store_candidates(self) -> list[Path]:
        """
        Ordered list of candidate paths for the filled trade feature file used in training.
        The first existing path will be used.
        """
        cands: list[Path] = []

        p_cfg = getattr(cfg, "META_TRADE_FEATURES_PATH", None)
        if p_cfg:
            cands.append(Path(str(p_cfg)).expanduser())

        # common defaults / outputs
        cands += [
            Path("results/trades.clean.csv"),
            Path("results/trades.clean.parquet"),
            Path("results/trades.enriched.filled.csv"),
            Path("results/trades.enriched.filled.parquet"),
            Path("results/trades.enriched.csv"),
            Path("results/trades.csv"),
        ]

        # also consider absolute within project root if RESULTS_DIR exists
        try:
            rd = Path(getattr(cfg, "RESULTS_DIR", Path("results")))
            cands += [
                rd / "trades.clean.csv",
                rd / "trades.clean.parquet",
                rd / "trades.enriched.filled.csv",
                rd / "trades.enriched.filled.parquet",
                rd / "trades.enriched.csv",
                rd / "trades.csv",
            ]
        except Exception:
            pass

        # de-dup while preserving order
        out: list[Path] = []
        seen = set()
        for p in cands:
            pr = p.resolve()
            if pr not in seen:
                seen.add(pr)
                out.append(pr)
        return out

    def _meta_store_path(self) -> Path | None:
        for p in self._meta_store_candidates():
            if p.exists():
                return p
        return None

    def _meta_store_load(self) -> None:
        """
        Load the filled trade feature table used for training and index it for fast lookup.

        Key: (SYMBOL_UPPER, entry_ts_floor_5m)
        Value: dict of manifest feature columns (the same names the model expects).
        """
        if getattr(self, "_meta_store_df", None) is not None:
            return

        path = self._meta_store_path()
        if path is None:
            self._meta_store_df = None
            self._meta_store_cols = []
            print("[meta_store] no filled trade feature file found; replay disabled", flush=True)
            return

        # Read manifest raw cols (authoritative list)
        model_dir = Path(getattr(cfg, "META_MODEL_DIR", "results/meta_export")).resolve()
        man_path = model_dir / "feature_manifest.json"

        raw_cols: list[str] = []
        try:
            with man_path.open("r", encoding="utf-8") as f:
                man = json.load(f)
            feats = man.get("features") or {}
            raw_cols = list(feats.get("numeric_cols") or []) + list(feats.get("cat_cols") or [])
        except Exception as e:
            print(f"[meta_store] WARNING: cannot read {man_path}: {e!r}", flush=True)

        # Get columns without reading full file (parquet) or with nrows=0 (csv)
        cols: list[str]
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(path))
            cols = list(pf.schema.names)
        else:
            df0 = pd.read_csv(path, nrows=0, low_memory=False)
            cols = list(df0.columns)

        if "symbol" not in cols:
            raise RuntimeError(f"[meta_store] {path} missing required column: symbol")

        entry_col = None
        for c in ("entry_ts", "entry_time", "entry_timestamp", "entry_datetime"):
            if c in cols:
                entry_col = c
                break
        if entry_col is None:
            raise RuntimeError(f"[meta_store] {path} missing required entry timestamp column (expected entry_ts)")

        keep = {"symbol", entry_col}
        if "entry" in cols:
            keep.add("entry")

        # Keep only manifest columns that exist in the file
        for c in raw_cols:
            if c in cols:
                keep.add(c)

        keep_cols = sorted(keep)

        # Load
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path, columns=keep_cols)
        else:
            df = pd.read_csv(path, usecols=keep_cols, low_memory=False)

        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df[entry_col] = pd.to_datetime(df[entry_col], utc=True, errors="coerce")
        df = df.dropna(subset=["symbol", entry_col])

        df["entry_ts_5m"] = df[entry_col].dt.floor("5min")
        df = df.drop(columns=[entry_col])

        # Ensure entry is numeric if present
        if "entry" in df.columns:
            df["entry"] = pd.to_numeric(df["entry"], errors="coerce")

        # Index for lookup
        df = df.set_index(["symbol", "entry_ts_5m"]).sort_index()

        # Store
        self._meta_store_df = df
        self._meta_store_cols = [c for c in df.columns if c != "entry"]

        print(f"[meta_store] using {path} rows={len(df):,} cols={len(df.columns)}", flush=True)

    def _meta_store_lookup(self, sym: str, ts: pd.Timestamp, entry_price: float | None) -> Dict[str, Any]:
        """
        Return a dict of replayed training features for this (sym, ts).
        If multiple trades share the same (sym, ts_5m), choose closest entry price when possible.
        """
        self._meta_store_load()
        df = getattr(self, "_meta_store_df", None)
        if df is None:
            return {}

        sym_u = str(sym).upper().strip()
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
        if ts is pd.NaT:
            return {}

        key = (sym_u, ts.floor("5min"))

        try:
            rows = df.loc[key]
        except KeyError:
            return {}

        # If unique match -> Series
        if isinstance(rows, pd.Series):
            row = rows
        else:
            # Multiple rows match same key -> disambiguate
            if (entry_price is not None) and ("entry" in rows.columns):
                ep = pd.to_numeric(rows["entry"], errors="coerce")
                diffs = (ep - float(entry_price)).abs()
                diffs_np = diffs.to_numpy()

                # If all NaN diffs, fallback first row
                if np.isfinite(diffs_np).any():
                    j = int(np.nanargmin(diffs_np))
                    row = rows.iloc[j]
                else:
                    row = rows.iloc[0]
            else:
                row = rows.iloc[0]

        out: Dict[str, Any] = {}
        for c in getattr(self, "_meta_store_cols", []):
            out[c] = row.get(c, np.nan)

        # Include entry if present (can be useful)
        if "entry" in getattr(row, "index", []):
            out["entry"] = row.get("entry", np.nan)

        return out



    def _meta_oi_funding_features_at(self, df5: pd.DataFrame, ts: pd.Timestamp) -> Dict[str, float]:
        """
        Compute the 14 OI/Funding features required by the meta-model manifest, using ONLY data <= ts.
        Mirrors scout.add_oi_funding_features() math (5m bars):
          - WIN_1H=12, WIN_4H=48, WIN_1D=288, WIN_3D=864, WIN_7D=2016
        Required raw cols in df5: open_interest, funding_rate, close, volume.
        """
        KEYS = [
            "oi_level",
            "oi_notional_est",
            "est_leverage",
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
        ]
        out: Dict[str, float] = {k: float("nan") for k in KEYS}

        if df5 is None or df5.empty:
            return out

        for req in ("open_interest", "funding_rate", "close", "volume"):
            if req not in df5.columns:
                return out

        ts = pd.to_datetime(ts, utc=True, errors="coerce")
        if ts is pd.NaT:
            return out

        idx = df5.index
        pos = int(idx.searchsorted(ts, side="right") - 1)
        if pos < 0:
            return out

        # Window sizes for 5-minute bars
        WIN_1H = 12
        WIN_4H = 48
        WIN_1D = 288
        WIN_3D = 3 * WIN_1D
        WIN_7D = 7 * WIN_1D

        oi = pd.to_numeric(df5["open_interest"], errors="coerce").to_numpy()
        fr = pd.to_numeric(df5["funding_rate"], errors="coerce").to_numpy()
        close = pd.to_numeric(df5["close"], errors="coerce").to_numpy()
        vol = pd.to_numeric(df5["volume"], errors="coerce").to_numpy()

        oi_now = oi[pos]
        close_now = close[pos]
        if not (np.isfinite(oi_now) and np.isfinite(close_now)):
            return out

        # funding is expected to be ffilled already; still guard
        fr_now = fr[pos]
        if not np.isfinite(fr_now):
            j = pos
            while j >= 0 and (not np.isfinite(fr[j])):
                j -= 1
            fr_now = fr[j] if j >= 0 else np.nan

        out["oi_level"] = float(oi_now)
        out["oi_notional_est"] = float(oi_now * close_now)
        out["funding_rate"] = float(fr_now) if np.isfinite(fr_now) else float("nan")
        out["funding_abs"] = float(abs(fr_now)) if np.isfinite(fr_now) else float("nan")

        def pct_change(arr: np.ndarray, lag: int) -> float:
            if pos - lag < 0:
                return float("nan")
            prev = arr[pos - lag]
            cur = arr[pos]
            if not (np.isfinite(prev) and np.isfinite(cur)) or prev == 0:
                return float("nan")
            return float(cur / prev - 1.0)

        out["oi_pct_1h"] = pct_change(oi, WIN_1H)
        out["oi_pct_4h"] = pct_change(oi, WIN_4H)
        out["oi_pct_1d"] = pct_change(oi, WIN_1D)

        # est_leverage = oi_notional_est / mean(oi_notional_est over last 1D), min_periods=1H
        notional = oi * close
        start_1d = max(0, pos - WIN_1D + 1)
        win_notional = notional[start_1d : pos + 1]
        if win_notional.size >= WIN_1H:
            notional_24h = float(np.nanmean(win_notional))
            cur_notional = float(out["oi_notional_est"])
            if np.isfinite(notional_24h):
                out["est_leverage"] = float(cur_notional / (notional_24h + 1e-9))

        # oi_z_7d (rolling mean/std, min_periods=1D, ddof=1 like pandas)
        start_7d = max(0, pos - WIN_7D + 1)
        win_oi = oi[start_7d : pos + 1]
        if win_oi.size >= WIN_1D:
            mu = float(np.nanmean(win_oi))
            sd = float(np.nanstd(win_oi, ddof=1)) if win_oi.size >= 2 else float("nan")
            if np.isfinite(mu) and np.isfinite(sd):
                out["oi_z_7d"] = float((oi_now - mu) / (sd + 1e-12))

        # oi_chg_norm_vol_1h = diff(1H) / sum(volume over last 1H), requires full 1H
        if pos - WIN_1H >= 0 and pos - WIN_1H + 1 >= 0:
            if pos >= WIN_1H - 1:
                vol_1h = vol[pos - WIN_1H + 1 : pos + 1]
                if vol_1h.size == WIN_1H:
                    denom = float(np.nansum(vol_1h))
                    num = oi[pos] - oi[pos - WIN_1H]
                    if np.isfinite(num) and np.isfinite(denom):
                        out["oi_chg_norm_vol_1h"] = float(num / (denom + 1e-9))

        # oi_price_div_1h = sign(ret_1h) * oi_pct_1h
        if pos - WIN_1H >= 0:
            prev_close = close[pos - WIN_1H]
            if np.isfinite(prev_close) and prev_close != 0 and np.isfinite(close_now):
                ret_1h = float(close_now / prev_close - 1.0)
                oi_pct_1h = out["oi_pct_1h"]
                if np.isfinite(oi_pct_1h):
                    out["oi_price_div_1h"] = float(np.sign(ret_1h) * oi_pct_1h)

        # funding_z_7d (same windowing rules)
        win_fr = fr[start_7d : pos + 1]
        if win_fr.size >= WIN_1D:
            mu = float(np.nanmean(win_fr))
            sd = float(np.nanstd(win_fr, ddof=1)) if win_fr.size >= 2 else float("nan")
            if np.isfinite(mu) and np.isfinite(sd) and np.isfinite(fr_now):
                out["funding_z_7d"] = float((fr_now - mu) / (sd + 1e-12))

        # funding_rollsum_3d (window=3D, min_periods=1D)
        start_3d = max(0, pos - WIN_3D + 1)
        win_fr3 = fr[start_3d : pos + 1]
        if win_fr3.size >= WIN_1D:
            out["funding_rollsum_3d"] = float(np.nansum(win_fr3))

        # funding_oi_div = funding_z_7d * oi_z_7d
        if np.isfinite(out["funding_z_7d"]) and np.isfinite(out["oi_z_7d"]):
            out["funding_oi_div"] = float(out["funding_z_7d"] * out["oi_z_7d"])

        return out



    def _get_1m(self, sym: str) -> Optional[pd.DataFrame]:
        if not getattr(cfg, "USE_INTRABAR_1M", False):
            return None

        if not isinstance(getattr(self, "_cache_1m", None), OrderedDict):
            self._cache_1m = OrderedDict()

        max_items = int(getattr(cfg, "BT_CACHE_1M_MAX_SYMBOLS", 2))

        cached = self._lru_get(self._cache_1m, sym)
        if cached is not None:
            return cached

        p = cfg.PARQUET_1M_DIR / f"{sym}.parquet"
        if not p.exists():
            return None

        df = pd.read_parquet(p)
        df = _ensure_dt(df)[["open", "high", "low", "close", "volume"]]
        df = self._maybe_downcast_ohlcv(df)

        self._lru_put(self._cache_1m, sym, df, max_items)
        return df

    # ----- entry-quality features (parity with training) -----
    def _get_entry_quality_panel(self, sym: str, df5: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(getattr(self, "_cache_entry_quality", None), OrderedDict):
            self._cache_entry_quality = OrderedDict()

        max_items = int(getattr(cfg, "BT_CACHE_EQ_MAX_SYMBOLS", 64))

        cached = self._lru_get(self._cache_entry_quality, sym)
        if cached is not None:
            return cached

        base = df5

        # Force DatetimeIndex UTC (compute_entry_quality_panel requires tz-aware index)
        if not isinstance(base.index, pd.DatetimeIndex):
            if "timestamp" in base.columns:
                base = base.copy()
                base.index = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
            else:
                raise ValueError("df5 must have DatetimeIndex or a 'timestamp' column for entry-quality features")

        if base.index.tz is None:
            base = base.copy()
            base.index = base.index.tz_localize("UTC")
        else:
            base = base.copy()
            base.index = base.index.tz_convert("UTC")

        base = base[~base.index.isna()].sort_index()

        # Only required OHLCV columns
        need = ["open", "high", "low", "close", "volume"]
        for c in need:
            if c not in base.columns:
                raise KeyError(f"df5 missing required column for entry-quality panel: {c}")

        panel = compute_entry_quality_panel(base[need].copy())

        self._lru_put(self._cache_entry_quality, sym, panel, max_items)
        return panel



    def _entry_quality_at(self, sym: str, ts: pd.Timestamp, df5: pd.DataFrame) -> Dict[str, float]:
        panel = self._get_entry_quality_panel(sym, df5)

        t = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(t) or panel.empty:
            return {
                "days_since_prev_break": np.nan,
                "consolidation_range_atr": np.nan,
                "prior_1d_ret": np.nan,
                "rv_3d": np.nan,
            }

        if t not in panel.index:
            idx = panel.index[panel.index >= t]
            if len(idx) == 0:
                return {
                    "days_since_prev_break": np.nan,
                    "consolidation_range_atr": np.nan,
                    "prior_1d_ret": np.nan,
                    "rv_3d": np.nan,
                }
            t = idx[0]

        row = panel.loc[t]
        return {
            "days_since_prev_break": float(row.get("days_since_prev_break", np.nan)),
            "consolidation_range_atr": float(row.get("consolidation_range_atr", np.nan)),
            "prior_1d_ret": float(row.get("prior_1d_ret", np.nan)),
            "rv_3d": float(row.get("rv_3d", np.nan)),
        }


    # ============================
    # META PARITY HELPERS (CLASS METHODS)
    # ============================

    def _meta_require_regimes_report(self) -> dict:
        cached = getattr(self, "_meta_regime_thresholds", None)
        if isinstance(cached, dict) and cached:
            return cached

        model_dir = Path(getattr(cfg, "META_MODEL_DIR", Path("results/meta_export"))).resolve()
        p = model_dir / "regimes_report.json"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing regimes_report.json at {p}. "
                f"Copy it from the 02_make_regimes.py output directory into META_MODEL_DIR "
                f"(must match the training run used to build this meta model)."
            )
        with p.open("r", encoding="utf-8") as f:
            rep = json.load(f)
        thr = rep.get("thresholds") or {}
        if not isinstance(thr, dict) or not thr:
            raise RuntimeError(f"regimes_report.json has no thresholds: {p}")
        self._meta_regime_thresholds = thr
        return thr

    def _meta_get_threshold(self, name: str, default: float) -> float:
        try:
            thr = getattr(self, "_meta_regime_thresholds", None)
            if isinstance(thr, dict) and (name in thr) and (thr[name] is not None):
                return float(thr[name])
        except Exception:
            pass
        return float(default)

    def _meta_recent_winrate_features(self) -> dict:
        hist = getattr(self, "_meta_win_hist", None)
        ewm = getattr(self, "_meta_ewm_win", None)

        if hist is None or len(hist) == 0:
            return {
                "recent_winrate_20": np.nan,
                "recent_winrate_50": np.nan,
                "recent_winrate_ewm_20": np.nan,
            }

        h = list(hist)
        w20 = float(np.mean(h[-20:])) if len(h) >= 1 else np.nan
        w50 = float(np.mean(h[-50:])) if len(h) >= 1 else np.nan
        w_ewm = float(ewm) if ewm is not None else np.nan

        return {
            "recent_winrate_20": w20,
            "recent_winrate_50": w50,
            "recent_winrate_ewm_20": w_ewm,
        }

    def _meta_update_winrate(self, pnl_R: float) -> None:
        if pnl_R is None:
            return
        try:
            pr = float(pnl_R)
        except Exception:
            return
        if not np.isfinite(pr):
            return

        if getattr(self, "_meta_win_hist", None) is None:
            self._meta_win_hist = deque(maxlen=50)

        win = 1.0 if pr > 0.0 else 0.0
        self._meta_win_hist.append(win)

        # pandas ewm(halflife=20, adjust=False) equivalent update
        halflife = 20.0
        alpha = 1.0 - math.exp(math.log(0.5) / halflife)
        prev = getattr(self, "_meta_ewm_win", None)
        if prev is None:
            self._meta_ewm_win = float(win)
        else:
            self._meta_ewm_win = float(alpha * win + (1.0 - alpha) * float(prev))

    def _meta_trend_vol_codes(
        self,
        trend_regime_1d: Optional[str],
        vol_regime_1d: Optional[str],
        vol_prob_low_1d: Optional[float],
        regime_code_1d: Optional[int],
    ) -> tuple[float, float]:
        # trend
        tcode = np.nan
        if trend_regime_1d is not None:
            s = str(trend_regime_1d).upper()
            if "BULL" in s:
                tcode = 1.0
            elif "BEAR" in s:
                tcode = 0.0
        if not np.isfinite(tcode) and regime_code_1d is not None:
            try:
                rc = int(regime_code_1d)
                tcode = 1.0 if rc in (2, 3) else 0.0
            except Exception:
                pass

        # vol
        vcode = np.nan
        if vol_regime_1d is not None:
            s = str(vol_regime_1d).upper()
            if "LOW" in s:
                vcode = 0.0
            elif "HIGH" in s:
                vcode = 1.0
        if not np.isfinite(vcode) and (vol_prob_low_1d is not None):
            try:
                vpl = float(vol_prob_low_1d)
                if np.isfinite(vpl):
                    vcode = 1.0 if vpl < 0.5 else 0.0
            except Exception:
                pass
        if not np.isfinite(vcode) and regime_code_1d is not None:
            try:
                rc = int(regime_code_1d)
                vcode = 1.0 if rc in (0, 2) else 0.0
            except Exception:
                pass

        return tcode, vcode

    def _meta_bucket_terciles(self, x: float, q33: float, q66: float) -> float:
        try:
            xv = float(x)
            q33v = float(q33)
            q66v = float(q66)
        except Exception:
            return np.nan
        if not (np.isfinite(xv) and np.isfinite(q33v) and np.isfinite(q66v)):
            return np.nan
        if xv <= q33v:
            return 0.0
        if xv >= q66v:
            return 2.0
        return 1.0

    def _meta_build_regime_sets(
        self,
        *,
        ts: pd.Timestamp,
        s: pd.Series | dict,
        regime_up: bool,
        markov_state_4h: Optional[int],
        trend_regime_1d: Optional[str],
        vol_regime_1d: Optional[str],
        vol_prob_low_1d: Optional[float],
        regime_code_1d: Optional[int],
        days_since_prev_break: float,
        consolidation_range_atr: float,
    ) -> dict:
        thr = self._meta_require_regimes_report()

        eps = float(thr.get("funding_neutral_eps"))
        oi_source = thr.get("oi_source") or "oi_z_7d"
        oi_q33 = thr.get("oi_q33")
        oi_q66 = thr.get("oi_q66")
        btc_vol_hi = float(thr.get("btc_vol_hi"))

        fresh_q33 = thr.get("fresh_q33")
        fresh_q66 = thr.get("fresh_q66")
        comp_q33 = thr.get("compression_q33")
        comp_q66 = thr.get("compression_q66")

        def _gf(k, default=np.nan):
            try:
                return float(s.get(k, default))  # type: ignore[attr-defined]
            except Exception:
                return default

        funding_rate = _gf("funding_rate", np.nan)
        oi_z_7d = _gf("oi_z_7d", np.nan)
        oi_pct_1d = _gf("oi_pct_1d", np.nan)

        btc_trend_slope = _gf("btc_trend_slope", _gf("btcusdt_trend_slope", np.nan))
        btc_vol_level = _gf("btc_vol_regime_level", _gf("btcusdt_vol_regime_level", np.nan))

        crowd_side = s.get("crowd_side", np.nan)  # type: ignore[attr-defined]
        try:
            crowd_side = float(crowd_side)
        except Exception:
            crowd_side = np.nan

        funding_regime_code = np.nan
        if np.isfinite(funding_rate) and np.isfinite(eps) and eps > 0:
            if funding_rate <= -eps:
                funding_regime_code = -1.0
            elif funding_rate >= eps:
                funding_regime_code = 1.0
            else:
                funding_regime_code = 0.0

        oi_val = oi_z_7d if str(oi_source) == "oi_z_7d" else oi_pct_1d
        oi_regime_code = np.nan
        if np.isfinite(oi_val) and (oi_q33 is not None) and (oi_q66 is not None):
            try:
                q33v = float(oi_q33)
                q66v = float(oi_q66)
                if oi_val <= q33v:
                    oi_regime_code = -1.0
                elif oi_val >= q66v:
                    oi_regime_code = 1.0
                else:
                    oi_regime_code = 0.0
            except Exception:
                oi_regime_code = np.nan

        btc_trend_up = np.nan
        btc_vol_high = np.nan
        btc_risk_regime_code = np.nan
        if np.isfinite(btc_trend_slope) and np.isfinite(btc_vol_level) and np.isfinite(btc_vol_hi):
            btc_trend_up = 1.0 if btc_trend_slope > 0.0 else 0.0
            btc_vol_high = 1.0 if btc_vol_level >= btc_vol_hi else 0.0
            btc_risk_regime_code = btc_trend_up * 2.0 + btc_vol_high

        ru = 1.0 if bool(regime_up) else 0.0
        risk_on = np.nan
        if np.isfinite(btc_trend_up) and np.isfinite(btc_vol_high):
            risk_on = 1.0 if (ru == 1.0 and btc_trend_up == 1.0 and btc_vol_high == 0.0) else 0.0

        freshness_code = (
            self._meta_bucket_terciles(days_since_prev_break, float(fresh_q33), float(fresh_q66))
            if (fresh_q33 is not None and fresh_q66 is not None)
            else np.nan
        )
        compression_code = (
            self._meta_bucket_terciles(consolidation_range_atr, float(comp_q33), float(comp_q66))
            if (comp_q33 is not None and comp_q66 is not None)
            else np.nan
        )

        trend_code, vol_code = self._meta_trend_vol_codes(trend_regime_1d, vol_regime_1d, vol_prob_low_1d, regime_code_1d)

        S1 = float(regime_code_1d) if regime_code_1d is not None else np.nan

        S2 = np.nan
        if (markov_state_4h is not None) and np.isfinite(vol_code):
            try:
                ms = int(markov_state_4h)
                S2 = float(ms * 2 + int(vol_code))
            except Exception:
                S2 = np.nan

        S3 = np.nan
        if np.isfinite(funding_regime_code) and np.isfinite(oi_regime_code):
            S3 = float((int(funding_regime_code) + 1) * 3 + (int(oi_regime_code) + 1))

        S4 = np.nan
        if np.isfinite(crowd_side) and np.isfinite(trend_code):
            S4 = float((int(crowd_side) + 1) * 2 + int(trend_code))

        S5 = np.nan
        if np.isfinite(btc_risk_regime_code):
            S5 = float(int(btc_risk_regime_code) * 2 + int(ru))

        S6 = np.nan
        if np.isfinite(freshness_code) and np.isfinite(compression_code):
            S6 = float(int(freshness_code) * 3 + int(compression_code))

        return {
            "funding_regime_code": funding_regime_code,
            "oi_regime_code": oi_regime_code,
            "btc_risk_regime_code": btc_risk_regime_code,
            "risk_on": risk_on,
            "risk_on_1": risk_on,
            "S1_regime_code_1d": S1,
            "S2_markov_x_vol1d": S2,
            "S3_funding_x_oi": S3,
            "S4_crowd_x_trend1d": S4,
            "S5_btcRisk_x_regimeUp": S5,
            "S6_fresh_x_compress": S6,
        }

    def _meta_extra_features(
        self,
        *,
        sym: str,
        s: pd.Series | dict,
        ts: pd.Timestamp,
        entry_price: float,
        atr_now: float,
        df5: pd.DataFrame,
        regime_up: bool,
        replay_values: Optional[Dict[str, float]] = None,
    ) -> dict:
        # entry-quality (4 cols) from your existing parity method
        eq = self._entry_quality_at(sym, ts, df5)

        # daily regimes (strings + codes + vol_prob)
        trend1d, vol1d, vprob1d, rcode1d, rstr1d = self._daily_regime_at(ts)

        # markov 4h
        m4_state, m4_prob = self._markov4h_at(ts)

        # recent winrates (shifted(1) by online state design)
        rw = self._meta_recent_winrate_features()

        def _gf(k, default=np.nan):
            try:
                return float(s.get(k, default))  # type: ignore[attr-defined]
            except Exception:
                return default

        extra = {
            "atr_at_entry": float(atr_now),
            "markov_prob_up_4h": float(m4_prob) if m4_prob is not None else np.nan,
            "markov_state_4h": float(m4_state) if m4_state is not None else np.nan,
            "vol_prob_low_1d": float(vprob1d) if vprob1d is not None else np.nan,

            "days_since_prev_break": float(eq.get("days_since_prev_break", np.nan)),
            "consolidation_range_atr": float(eq.get("consolidation_range_atr", np.nan)),
            "prior_1d_ret": float(eq.get("prior_1d_ret", np.nan)),
            "rv_3d": float(eq.get("rv_3d", np.nan)),

            "regime_code_1d": float(rcode1d) if rcode1d is not None else np.nan,
            "regime_up": float(1 if bool(regime_up) else 0),

            "recent_winrate_20": rw.get("recent_winrate_20", np.nan),
            "recent_winrate_50": rw.get("recent_winrate_50", np.nan),
            "recent_winrate_ewm_20": rw.get("recent_winrate_ewm_20", np.nan),

            "trend_regime_1d": trend1d,
            "vol_regime_1d": vol1d,
            "regime_1d": rstr1d,

            "markov_state_up_4h": float(m4_state) if m4_state is not None else np.nan,
        }

        # Ensure per-symbol OI/Funding keys required by the meta manifest exist at scoring time.
        # This computes using ONLY df5 bars <= ts and returns NaNs if underlying data is unavailable,
        # but critically it always returns the required KEYS.
        extra.update(self._meta_oi_funding_features_at(df5, ts))



        # FIX: Merge replay values into extra so they are available for scoring
        if replay_values:
            extra.update(replay_values)

        extra.update(
            {
                "btc_funding_rate": _gf("btcusdt_funding_rate", np.nan),
                "btc_oi_z_7d": _gf("btcusdt_oi_z_7d", np.nan),
                "btc_vol_regime_level": _gf("btcusdt_vol_regime_level", np.nan),
                "btc_trend_slope": _gf("btcusdt_trend_slope", np.nan),
            }
        )
        extra.update(
            {
                "eth_funding_rate": _gf("ethusdt_funding_rate", np.nan),
                "eth_oi_z_7d": _gf("ethusdt_oi_z_7d", np.nan),
                "eth_vol_regime_level": _gf("ethusdt_vol_regime_level", np.nan),
                "eth_trend_slope": _gf("ethusdt_trend_slope", np.nan),
            }
        )

        # FIX: Pass combined dict (s + extra) to regime builder so it sees replay values
        sets = self._meta_build_regime_sets(
            ts=ts,
            s={**dict(s), **extra},
            regime_up=regime_up,
            markov_state_4h=(int(m4_state) if m4_state is not None else None),
            trend_regime_1d=trend1d,
            vol_regime_1d=vol1d,
            vol_prob_low_1d=(float(vprob1d) if vprob1d is not None else None),
            regime_code_1d=(int(rcode1d) if rcode1d is not None else None),
            days_since_prev_break=float(eq.get("days_since_prev_break", np.nan)),
            consolidation_range_atr=float(eq.get("consolidation_range_atr", np.nan)),
        )
        extra.update(sets)
        return extra




    # ----- decision logging (parity debugging) -----
    def _log_decision(

        self,
        symbol: str,
        signal_ts: pd.Timestamp,
        ts_effective: pd.Timestamp,
        decision: str,
        reason: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Append one row to the internal decision log.

        - symbol:        instrument symbol, e.g. "ETHUSDT"
        - signal_ts:     original signal timestamp (from scout)
        - ts_effective:  timestamp actually used for entry in backtest
        - decision:      "taken" or "skipped"
        - reason:        short string like "regime_down", "atr_too_small", ...
        - extra:         any extra debug fields (pnl, exit_reason, ...)
        """

        if not bool(getattr(cfg, "BT_DECISION_LOG_ENABLED", True)):
            return

        if extra is None:
            extra = {}
        row = {
            "symbol": symbol,
            "signal_ts": signal_ts,
            "ts_effective": ts_effective,
            "decision": decision,
            "reason": reason,
        }
        row.update(extra)
        self._decision_log.append(row)

    # ----- main loop -----
    def run(self, signals):
        if signals is None:
            print("No signals to backtest.")
            return

        # FIX: Initialize meta store for replay
        self._meta_store_load()

        rs_enabled = bool(getattr(cfg, "RS_ENABLED", False))
        rs_min = float(getattr(cfg, "RS_MIN_PERCENTILE", 0.0))

        # Accept either:
        # - a DataFrame (single-file signals.parquet)
        # - an iterator/generator yielding row-like objects (partitioned signals dir)
        if isinstance(signals, pd.DataFrame):
            sig = signals
            if sig.empty:
                print("No signals to backtest.")
                return

            # If offline meta attach is used elsewhere, this is safe (no-op if already present)
            try:
                sig = _attach_meta_if_needed(sig)
            except Exception:
                pass

            if rs_enabled and "rs_pct" in sig.columns:
                sig = sig[sig["rs_pct"] >= rs_min]

            if sig.empty:
                print("No signals after RS filter.")
                return

            sig = sig.copy()
            sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
            sig = sig.dropna(subset=["timestamp"])

            if "symbol" in sig.columns:
                sig = sig.sort_values(["timestamp", "symbol"], kind="mergesort")
            else:
                sig = sig.sort_values(["timestamp"], kind="mergesort")

            # --- MODIFIED: Wrap with tqdm for progress bar ---
            total_rows = len(sig)
            print(f"[bt] Processing {total_rows:,} signals (DataFrame mode)...")
            sig_iter = tqdm((row for _, row in sig.iterrows()), total=total_rows, desc="Simulating", unit="sig", mininterval=1.0)
            stream_mode = False
        else:
            # Streaming iterator path (already sorted by iter_partitioned_signals_sorted)
            # --- MODIFIED: Wrap with tqdm for progress bar ---
            print(f"[bt] Processing signals (Streaming mode)...")
            sig_iter = tqdm(signals, desc="Simulating", unit="sig", mininterval=1.0)
            stream_mode = True


        for s in sig_iter:
            # normalize ts/symbol safely for both dict and pandas Series rows
            ts = pd.to_datetime(s.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue

            sym = str(s.get("symbol", "")).strip()
            if not sym:
                continue

            # Apply RS filter in streaming mode (DataFrame mode already filtered)
            if stream_mode and rs_enabled:
                try:
                    rs_val = float(s.get("rs_pct", float("nan")))
                except Exception:
                    rs_val = float("nan")
                if np.isfinite(rs_val) and rs_val < rs_min:
                    continue

            entry_price = float(s.get("entry"))


            # remember original signal timestamp before any alignment
            sig_ts = ts

            # --- INSERT START: Manage Active Trades & Check Limit ---
            
            # 1. Clean up closed trades (remove trades that exited before this signal)
            # We keep trades where exit_ts is None (open) or exit_ts > current ts
            self._active_trades = [
                t for t in self._active_trades 
                if t.exit_ts is None or t.exit_ts > ts
            ]

            # 2. Check Max Open Positions
            max_open = int(getattr(cfg, "MAX_OPEN_POSITIONS", 20))
            if len(self._active_trades) >= max_open:
                self._skipped["max_open"] += 1
                self._log_decision(
                    symbol=sym, 
                    signal_ts=sig_ts, 
                    ts_effective=ts, 
                    decision="skipped", 
                    reason="max_open_positions", 
                    extra={"active_count": len(self._active_trades)}
                )
                continue

            if getattr(cfg, "WEEK_PATTERN_ENABLED", False):
                pattern = str(getattr(cfg, "WEEK_PATTERN", ""))
                if not _is_trade_week(ts, pattern):
                    self._log_decision(
                        symbol=sym,
                        signal_ts=sig_ts,
                        ts_effective=ts,
                        decision="skipped",
                        reason="not_trade_week",
                    )
                    continue
                if _must_flatten_now(ts, cfg):
                    self._log_decision(
                        symbol=sym,
                        signal_ts=sig_ts,
                        ts_effective=ts,
                        decision="skipped",
                        reason="flatten_now",
                    )
                    continue


            # --- regime gate
            regime_up = self.regime.is_up(ts)
            if (
                getattr(cfg, "REGIME_FILTER_ENABLED", True)
                and getattr(cfg, "REGIME_BLOCK_WHEN_DOWN", True)
                and not regime_up
            ):
                self._log_decision(
                    symbol=sym,
                    signal_ts=sig_ts,
                    ts_effective=ts,
                    decision="skipped",
                    reason="regime_down",
                )
                continue

            # --- additional ETH MACD slope filter (optional) ---
            if getattr(cfg, "REGIME_SLOPE_FILTER_ENABLED", False):
                hist_now, hist_slope = self.regime.get_hist_and_slope(ts)
                slope_min = float(getattr(cfg, "REGIME_SLOPE_MIN", 0.0))

                # If we don't have slope yet, just allow (no early skip)
                if hist_slope is not None:
                    if hist_slope < slope_min:
                        self._log_decision(
                            symbol=sym,
                            signal_ts=sig_ts,
                            ts_effective=ts,
                            decision="skipped",
                            reason="regime_slope_down",
                            extra={"hist_now": hist_now, "hist_slope": hist_slope},
                        )
                        continue


    
            # --- data align
            df5 = self._get_5m(sym)
            if ts not in df5.index:
                idx = df5.index[df5.index >= ts]
                if len(idx) == 0:
                    self._log_decision(
                        symbol=sym,
                        signal_ts=sig_ts,
                        ts_effective=ts,
                        decision="skipped",
                        reason="no_5m_data_after_signal",
                    )
                    continue
                ts = idx[0]
                entry_price = float(df5.loc[ts, "close"])

            # Apply 0.2% slippage to simulate real execution/spread.
            # This raises the effective entry and Stop Loss, helping clear "Zombie" trades.
            entry_price = entry_price * 1.002

  

            # --- throughput guards (computed after aligning 'ts')
            day = ts.floor("D")
            if (
                getattr(cfg, "MAX_TRADES_PER_DAY", None) is not None
                and not getattr(cfg, "LABELING_MODE", False)
            ):
                if self._trades_per_day.get(day, 0) >= int(cfg.MAX_TRADES_PER_DAY):
                    self._skipped["daycap"] += 1
                    self._log_decision(
                        symbol=sym,
                        signal_ts=sig_ts,
                        ts_effective=ts,
                        decision="skipped",
                        reason="daycap",
                        extra={"trades_today": self._trades_per_day.get(day, 0)},
                    )
                    continue


            if not getattr(cfg, "LABELING_MODE", False):
                dedup_hours = int(getattr(cfg, "DEDUP_WINDOW_HOURS", 8))
                last_entry = self._last_entry.get(sym)
                if last_entry is not None:
                    # Calculate hours since last entry
                    hours_since = (ts - last_entry).total_seconds() / 3600.0
                    if hours_since < dedup_hours:
                        self._skipped["lock"] += 1 # Count as lock/cooldown skip
                        self._log_decision(
                            symbol=sym, 
                            signal_ts=sig_ts, 
                            ts_effective=ts, 
                            decision="skipped", 
                            reason="dedup_entry", 
                            extra={"last_entry": last_entry, "hours_since": hours_since}
                        )
                        continue
                # Per-symbol `lock` semantics = (in_position) OR (cooldown after exit).
                # We keep `lock_until` for backward compatibility in CSV outputs.
                open_until = self._open_until.get(sym)
                if open_until is not None and ts < open_until:
                    self._skipped["lock"] += 1
                    self._skipped["open"] += 1
                    cooldown_until = self._cooldown_until.get(sym) or self._lock_until.get(sym)
                    self._log_decision(
                        symbol=sym,
                        signal_ts=sig_ts,
                        ts_effective=ts,
                        decision="skipped",
                        reason="in_position",
                        extra={
                            "open_until": open_until,
                            "cooldown_until": cooldown_until,
                            "lock_until": cooldown_until,
                            "open_remaining_hr": float((open_until - ts).total_seconds() / 3600.0),
                        },
                    )
                    # NEW: Log lock skip event
                    self._lock_timeline.append({
                        "symbol": sym,
                        "event": "skip",
                        "ts": ts,
                        "open_until": open_until,
                        "cooldown_until": cooldown_until,
                        "lock_until": cooldown_until,
                        "reason": "open",
                    })
                    continue
                
                cooldown_until = self._cooldown_until.get(sym) or self._lock_until.get(sym)
                if cooldown_until is not None and ts < cooldown_until:
                    self._skipped["lock"] += 1
                    self._skipped["cooldown"] += 1
                    self._log_decision(
                        symbol=sym,
                        signal_ts=sig_ts,
                        ts_effective=ts,
                        decision="skipped",
                        reason="cooldown",
                        extra={
                            "cooldown_until": cooldown_until,
                            "lock_until": cooldown_until,
                            "cooldown_remaining_hr": float((cooldown_until - ts).total_seconds() / 3600.0),
                        },
                    )
                    # NEW: Log lock skip event
                    self._lock_timeline.append({
                        "symbol": sym,
                        "event": "skip",
                        "ts": ts,
                        "cooldown_until": cooldown_until,
                        "lock_until": cooldown_until,
                        "reason": "cooldown",
                    })
                    continue


            atr_now = float(df5.loc[ts, "atr_pre"])
            if not np.isfinite(atr_now) or atr_now <= 0:
                self._log_decision(
                    symbol=sym,
                    signal_ts=sig_ts,
                    ts_effective=ts,
                    decision="skipped",
                    reason="atr_invalid",
                    extra={"atr_now": atr_now, "entry_price": entry_price},
                )
                continue

            min_atr_pct = float(getattr(cfg, "MIN_ATR_PCT_OF_PRICE", 0.0001))
            if (atr_now / entry_price) < min_atr_pct:
                self._log_decision(
                    symbol=sym,
                    signal_ts=sig_ts,
                    ts_effective=ts,
                    decision="skipped",
                    reason="atr_too_small",
                    extra={
                        "atr_now": atr_now,
                        "entry_price": entry_price,
                        "min_atr_pct": min_atr_pct,
                        "atr_pct": atr_now / entry_price,
                    },
                )
                continue







            # --- regime size-down (if not blocked)
            equity_for_sizing = self.equity
            if (not regime_up) and (not getattr(cfg, "REGIME_BLOCK_WHEN_DOWN", True)):
                equity_for_sizing *= float(getattr(cfg, "REGIME_SIZE_WHEN_DOWN", 0.5))

            # === meta: gate + dynamic sizing ===
            prob_val = float("nan")

            if getattr(cfg, "BT_META_ONLINE_ENABLED", False) and (score_signal_with_meta is not None):
                # FIX: Lookup replay values
                replay = self._meta_store_lookup(sym, ts, entry_price)
                
                # IMPORTANT:
                # - ts         = effective aligned timestamp used for entry execution
                # - entry_price = effective (post-slippage) entry used by the simulator
                meta_extra = self._meta_extra_features(
                    sym=sym,
                    s=s,
                    ts=ts,
                    entry_price=float(entry_price),
                    atr_now=float(atr_now),
                    df5=df5,
                    regime_up=bool(regime_up),
                    replay_values=replay # FIX: Pass replay values
                )

                prob_val = float(
                    score_signal_with_meta(
                        s,
                        ts,
                        extra=meta_extra,
                        entry_override=float(entry_price),
                    )
                )
            else:
                # Offline meta: rely on pre-merged predictions (if any)
                prob = s.get("meta_p", np.nan)
                try:
                    prob_val = float(prob)
                except Exception:
                    prob_val = float("nan")


            # --- risk_on state used for scope-gating and risk-off probe cap ---
            btc_vol_hi = float(self._meta_get_threshold("btc_vol_hi", float(getattr(cfg, "BTC_VOL_HI", 1.0))))

            # IMPORTANT: do not reassign `regime_up`
            ru = 1 if bool(regime_up) else 0

            # BTC trend up
            try:
                btc_trend_slope = float(s.get("btcusdt_trend_slope", np.nan))
            except Exception:
                btc_trend_slope = np.nan
            btc_trend_up = 1 if (np.isfinite(btc_trend_slope) and btc_trend_slope > 0.0) else 0

            # BTC vol high
            try:
                btc_vol_level = float(s.get("btcusdt_vol_regime_level", np.nan))
            except Exception:
                btc_vol_level = np.nan
            btc_vol_high = 1 if (np.isfinite(btc_vol_level) and btc_vol_level >= btc_vol_hi) else 0

            # risk_on = 1[(regime_up==1) & (btc_trend_up==1) & (btc_vol_high==0)]
            risk_on = 1 if (ru == 1 and btc_trend_up == 1 and btc_vol_high == 0) else 0


            # 2) Apply gating ONLY when we have a valid probability in [0,1]
            thr = getattr(cfg, "META_PROB_THRESHOLD", None)
            if thr is not None:
                thr_f = float(thr)
                valid = np.isfinite(prob_val) and (0.0 <= prob_val <= 1.0)

                gate_scope = str(getattr(cfg, "META_GATE_SCOPE", "all"))
                fail_closed = bool(getattr(cfg, "META_GATE_FAIL_CLOSED", False))

                in_scope = True
                if gate_scope.upper() in ("RISK_ON_1", "RISK_ON", "RISKON"):
                    in_scope = (risk_on == 1)
                    if (not in_scope) and fail_closed:
                        self._log_decision(
                            symbol=sym,
                            signal_ts=sig_ts,
                            ts_effective=ts,
                            decision="skipped",
                            reason="meta_scope",
                            extra={"risk_on": risk_on, "scope": gate_scope, "fail_closed": True},
                        )
                        continue

                if in_scope and valid and (prob_val < thr_f):
                    self._log_decision(
                        symbol=sym,
                        signal_ts=sig_ts,
                        ts_effective=ts,
                        decision="skipped",
                        reason="meta_prob",
                        extra={"prob_val": prob_val, "thr": thr_f, "scope": gate_scope},
                    )
                    continue
                # if not valid → treat as "no meta available" => do NOT gate here



            risk_mode_cfg = str(getattr(cfg, "RISK_MODE", "percent"))


            try:
                hist4h_val = float(s.get("eth_macd_hist_4h", np.nan))
            except Exception:
                hist4h_val = np.nan

            scale_in = s.get("risk_scale", None)
            if scale_in is not None:
                try:
                    size_mult = float(scale_in)
                except Exception:
                    size_mult = 1.0
            else:
                size_mult = _dyn_size_multiplier(prob_val, hist4h_val, cfg)

            # --- risk-off probe sizing (absolute cap; do NOT overwrite regime_up) ---
            probe = float(getattr(cfg, "RISK_OFF_PROBE_MULT", 1.0))

            risk_mode_cfg = str(getattr(cfg, "RISK_MODE", "cash"))
            base_risk_pct = float(getattr(cfg, "RISK_PCT", 0.01))
            base_fixed_cash = float(getattr(cfg, "FIXED_RISK_CASH", 10.0))
            if risk_mode_cfg.lower() == "percent":
                risk_pct_override = max(base_risk_pct * size_mult, 0.0)
                fixed_cash_override = base_fixed_cash
            else:
                risk_pct_override = base_risk_pct
                fixed_cash_override = max(base_fixed_cash * size_mult, 0.0)

            tp_mult_scale, sl_mult_scale = _dyn_exit_multipliers(s, cfg)
            sl_override = float(getattr(cfg, "SL_ATR_MULT", 2.0)) * sl_mult_scale
            tp_override = float(getattr(cfg, "TP_ATR_MULT", 8.0)) * tp_mult_scale
            av_sl_override = float(getattr(cfg, "AVWAP_SL_MULT", 2.0)) * sl_mult_scale
            av_tp_override = float(getattr(cfg, "AVWAP_TP_MULT", 8.0)) * tp_mult_scale

            # --- simulate exit path with partials / trailing
            equity_at_entry = float(equity_for_sizing)

            # choose AVWAP anchor (only used if EXIT_BASIS="avwap_atr")
            # Choose AVWAP anchor
            anchor_ts = ts
            if str(getattr(cfg, "EXIT_BASIS", "price_atr")).lower() == "avwap_atr":
                anchor_mode = str(getattr(cfg, "AVWAP_ANCHOR", "breakout")).lower()
                if anchor_mode == "breakout":
                    # 1) prefer explicit timestamp if scout provided it
                    explicit = s.get("don_break_ts", None)
                    if pd.notna(explicit):
                        anchor_ts = pd.to_datetime(explicit, utc=True, errors="coerce") or ts
                    else:
                        # 2) infer from breakout level carried by the signal (expected to exist)
                        anchor_ts = _infer_breakout_anchor_ts(
                            df5=df5,
                            entry_ts=ts,
                            don_break_level=float(s.get("don_break_level", float("nan"))),
                            lookback_days=40
                        )
                else:
                    # entry anchor
                    anchor_ts = ts

            sim = _simulate_long_with_partials_trail(
                symbol=sym,
                entry_ts=ts,
                entry_price=entry_price,
                atr_at_entry=atr_now,
                equity_at_entry=equity_at_entry,
                df5=df5,
                risk_mode_override=risk_mode_cfg,
                risk_pct_override=risk_pct_override,
                fixed_cash_override=fixed_cash_override,
                avwap_anchor_ts=anchor_ts,
                sl_mult_override=sl_override,
                tp_mult_override=tp_override,
                av_sl_mult_override=av_sl_override,
                av_tp_mult_override=av_tp_override,
            )
            if sim is None:
                self._log_decision(
                    symbol=sym,
                    signal_ts=sig_ts,
                    ts_effective=ts,
                    decision="skipped",
                    reason="simulation_none",
                )
                continue

            exit_ts     = sim["exit_ts"]
            exit_price  = sim["exit_price"]

            exit_reason = sim["exit_reason"]
            qty         = sim["qty_filled"]
            pnl         = sim["pnl_cash"]
            fees        = sim["fees"]
            sl_init     = sim["sl_initial"]
            sl_final    = sim["sl_final"]
            tp          = sim["tp"]

            sl = sl_final  # what ends up on the book when we exit

            # --- R based on *initial* stop distance (stable even if SL moved to BE)
            risk_per_unit = max(entry_price - sl_init, 1e-12)
            risk_notional = risk_per_unit * qty
            pnl_R = (pnl / risk_notional) if risk_notional > 0 else np.nan

            # Optional: R normalized by risk budget (useful for meta/reporting)
            if str(getattr(cfg, "RISK_MODE", "percent")).lower() == "cash":
                denom_cash = float(getattr(cfg, "FIXED_RISK_CASH", 10.0))
            else:
                denom_cash = float(getattr(cfg, "RISK_PCT", 0.01)) * float(equity_at_entry)
            pnl_Rcash = pnl / denom_cash if denom_cash > 0 else np.nan  # (kept if you want to store it)

            # --- update equity & bookkeeping
            self.equity += pnl
            self.equity_curve.append((exit_ts or ts, self.equity))




            # optional regimes for diagnostics
            mkv_state, mkv_prob = self._markov_at(ts)
            trend1d, vol1d, vprob1d, rcode1d, rstr1d = self._daily_regime_at(ts)
            m4_state, m4_prob = self._markov4h_at(ts)

            # Helper to safely pull numeric from signal row
            def _get_float(col: str) -> Optional[float]:
                if col in s and pd.notna(s[col]):
                    try:
                        return float(s[col])
                    except Exception:
                        return None
                return None

            def _get_int(col: str) -> Optional[int]:
                if col in s and pd.notna(s[col]):
                    try:
                        return int(s[col])
                    except Exception:
                        try:
                            return int(float(s[col]))
                        except Exception:
                            return None
                return None

            # Breakout strength at entry in ATR units:
            # don_dist_atr = (entry - don_break_level) / atr_at_entry
            don_level = _get_float("don_break_level")
            entry_don_dist = None
            if (
                don_level is not None
                and atr_now is not None
                and atr_now > 0
                and entry_price is not None
            ):
                entry_don_dist = (entry_price - don_level) / atr_now

            new_trade = Trade(
                trade_id=len(self.trades) + 1,
                symbol=sym,
                entry_ts=ts,
                exit_ts=exit_ts,
                entry=entry_price,
                exit=exit_price,
                qty=qty,
                side="long",
                sl=sl,
                tp=tp,
                exit_reason=exit_reason,
                atr_at_entry=atr_now,
                regime_up=bool(regime_up),

                # original meta-ish fields
                rs_pct=_get_float("rs_pct"),
                pullback_type=s.get("pullback_type"),
                entry_rule=s.get("entry_rule"),
                don_break_len=_get_int("don_break_len"),
                fees=float(fees),
                pnl=float(pnl),
                pnl_R=float(pnl_R),
                mae_over_atr=float(sim["mae_over_atr"]),
                mfe_over_atr=float(sim["mfe_over_atr"]),

                # regimes
                markov_state_4h=m4_state,
                trend_regime_1d=trend1d,
                vol_regime_1d=vol1d,
                vol_prob_low_1d=vprob1d,
                regime_code_1d=rcode1d,
                regime_1d=rstr1d,
                markov_prob_up_4h=m4_prob,

                # --- 1h context & Donchian snapshot (from scout feature panel) ---
                atr_1h=_get_float("atr_1h"),
                rsi_1h=_get_float("rsi_1h"),
                adx_1h=_get_float("adx_1h"),
                vol_mult=_get_float("vol_mult"),
                atr_pct=_get_float("atr_pct"),

                days_since_prev_break=_get_float("days_since_prev_break"),
                consolidation_range_atr=_get_float("consolidation_range_atr"),
                prior_1d_ret=_get_float("prior_1d_ret"),
                rv_3d=_get_float("rv_3d"),

                don_break_level=don_level,
                don_dist_atr=entry_don_dist,

                # Additional multi-timeframe asset indicators (Chunk 2)
                asset_rsi_15m=_get_float("asset_rsi_15m"),
                asset_rsi_4h=_get_float("asset_rsi_4h"),

                asset_macd_line_1h=_get_float("asset_macd_line_1h"),
                asset_macd_signal_1h=_get_float("asset_macd_signal_1h"),
                asset_macd_hist_1h=_get_float("asset_macd_hist_1h"),
                asset_macd_slope_1h=_get_float("asset_macd_slope_1h"),

                asset_macd_line_4h=_get_float("asset_macd_line_4h"),
                asset_macd_signal_4h=_get_float("asset_macd_signal_4h"),
                asset_macd_hist_4h=_get_float("asset_macd_hist_4h"),
                asset_macd_slope_4h=_get_float("asset_macd_slope_4h"),

                asset_vol_1h=_get_float("asset_vol_1h"),
                asset_vol_4h=_get_float("asset_vol_4h"),

                gap_from_1d_ma=_get_float("gap_from_1d_ma"),
                prebreak_congestion=_get_float("prebreak_congestion"),


                # --- ETH 4h MACD context (added in _eth_macd_full_4h_to_5m) ---
                eth_macd_line_4h=_get_float("eth_macd_line_4h"),
                eth_macd_signal_4h=_get_float("eth_macd_signal_4h"),
                eth_macd_hist_4h=_get_float("eth_macd_hist_4h"),
                eth_macd_both_pos_4h=_get_int("eth_macd_both_pos_4h"),
                eth_macd_hist_slope_4h=_get_float("eth_macd_hist_slope_4h"),
                eth_macd_hist_slope_1h=_get_float("eth_macd_hist_slope_1h"),

                # --- OI + funding snapshot (from add_oi_funding_features via _build_feature_panel) ---
                oi_level=_get_float("oi_level"),
                oi_notional_est=_get_float("oi_notional_est"),
                oi_pct_1h=_get_float("oi_pct_1h"),
                oi_pct_4h=_get_float("oi_pct_4h"),
                oi_pct_1d=_get_float("oi_pct_1d"),
                oi_z_7d=_get_float("oi_z_7d"),
                oi_chg_norm_vol_1h=_get_float("oi_chg_norm_vol_1h"),
                oi_price_div_1h=_get_float("oi_price_div_1h"),
                funding_rate=_get_float("funding_rate"),
                funding_abs=_get_float("funding_abs"),
                funding_z_7d=_get_float("funding_z_7d"),
                funding_rollsum_3d=_get_float("funding_rollsum_3d"),
                funding_oi_div=_get_float("funding_oi_div"),

                # Cross-asset BTC/ETH OI + funding snapshot (from _merge_cross_asset_context)
                btc_funding_rate=_get_float("btcusdt_funding_rate"),
                btc_oi_z_7d=_get_float("btcusdt_oi_z_7d"),
                btc_vol_regime_level=_get_float("btcusdt_vol_regime_level"),
                btc_trend_slope=_get_float("btcusdt_trend_slope"),

                eth_funding_rate=_get_float("ethusdt_funding_rate"),
                eth_oi_z_7d=_get_float("ethusdt_oi_z_7d"),
                eth_vol_regime_level=_get_float("ethusdt_vol_regime_level"),
                eth_trend_slope=_get_float("ethusdt_trend_slope"),


                crowded_long=_get_int("crowded_long"),
                crowded_short=_get_int("crowded_short"),
                crowd_side=_get_int("crowd_side"),
                est_leverage=_get_float("est_leverage"),
        

            )

            self.trades.append(new_trade)
            # meta parity: update recent winrate state
            try:
                self._meta_update_winrate(pnl_R)
            except Exception:
                pass
            self._active_trades.append(new_trade)
            self._last_entry[sym] = ts



            # record that we actually took this signal (include lock timestamps for analyzers)
            cd_min = int(getattr(cfg, "SYMBOL_COOLDOWN_MINUTES", 120))
            cd = pd.Timedelta(minutes=cd_min) if cd_min else pd.Timedelta(0)

            # exit_ts is the simulated exit timestamp for this trade
            open_until = exit_ts or ts
            cooldown_until = open_until + cd

            self._log_decision(
                symbol=sym,
                signal_ts=sig_ts,
                ts_effective=ts,
                decision="taken",
                reason="ok",
                extra={
                    # existing fields
                    "exit_ts": exit_ts,
                    "exit_reason": exit_reason,
                    "pnl": float(pnl),
                    "pnl_R": float(pnl_R),

                    # new fields (what the analyzers need)
                    "open_until": open_until,
                    "cooldown_until": cooldown_until,
                    "lock_until": cooldown_until,
                    "cooldown_minutes": int(cd_min),
                },
            )




            # --- guardrails
            max_tr = int(getattr(cfg, "MAX_TRADES_PER_VARIANT", 5000))
            min_eq = float(getattr(cfg, "MIN_EQUITY_FRACTION_BEFORE_ABORT", 0.20))
            if len(self.trades) >= max_tr:
                print(f"[guard] aborting variant: trades>{max_tr}")
                break
            if self.equity <= (self.initial_capital * min_eq):
                print(f"[guard] aborting variant: equity<{min_eq*100:.0f}% of initial")
                break

            # --- update locks / per-day counters
            if not getattr(cfg, "LABELING_MODE", False):
                cd_min = int(getattr(cfg, "SYMBOL_COOLDOWN_MINUTES", 120))
                cd = pd.Timedelta(minutes=cd_min) if cd_min else pd.Timedelta(0)
                
                # Distinguish open-position window vs post-exit cooldown window
                open_until = exit_ts or ts
                cooldown_until = open_until + cd
                self._open_until[sym] = open_until
                self._cooldown_until[sym] = cooldown_until
                
                # Backward-compat alias used by existing analysis: lock_until == cooldown_until
                new_lock = cooldown_until
                self._lock_until[sym] = new_lock
                self._trades_per_day[day] = self._trades_per_day.get(day, 0) + 1

                # NEW: Log lock update event
                self._lock_timeline.append({
                    "symbol": sym,
                    "event": "update",
                    "ts": (exit_ts or ts),
                    "open_until": open_until,
                    "cooldown_until": cooldown_until,
                    "lock_until": new_lock,
                    "reason": "trade_exit"
                })

        # finalize
        self._save_outputs()
        if any(self._skipped.values()):
            print(f"[throughput] skipped due to lock={self._skipped['lock']} (open={self._skipped['open']}, cooldown={self._skipped['cooldown']}), daycap={self._skipped['daycap']}, max_open={self._skipped['max_open']}")

    def _save_outputs(self):

        # Build trades DataFrame robustly (even if there are zero fills)
        if not self.trades:
            cols = [f.name for f in fields(Trade)]
            trades_df = pd.DataFrame(columns=cols)
        else:
            trades_df = pd.DataFrame([asdict(t) for t in self.trades])
            if "entry_ts" in trades_df.columns:
                trades_df = trades_df.sort_values("entry_ts")

            # --- NEW: recent winrate features (for offline analysis) ---
            if "pnl_R" in trades_df.columns:
                is_win = (trades_df["pnl_R"] > 0).astype(float)

                # rolling simple winrate over last N trades, lagged by 1 to avoid look-ahead
                trades_df["recent_winrate_20"] = (
                    is_win.rolling(window=20, min_periods=1).mean().shift(1)
                )
                trades_df["recent_winrate_50"] = (
                    is_win.rolling(window=50, min_periods=1).mean().shift(1)
                )

                # exponentially weighted winrate with halflife in "trades"
                trades_df["recent_winrate_ewm_20"] = (
                    is_win.ewm(halflife=20, min_periods=1, adjust=False).mean().shift(1)
                )

        # write trades
        try:
            pq.write_table(
                pa.Table.from_pandas(trades_df, preserve_index=False),
                cfg.RESULTS_DIR / "trades.parquet",
            )
        except Exception:
            pass
        trades_df.to_csv(cfg.RESULTS_DIR / "trades.csv", index=False)

        # write equity if available
        if self.equity_curve:
            eq = (
                pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
                .sort_values("timestamp")
            )
            try:
                pq.write_table(
                    pa.Table.from_pandas(eq, preserve_index=False),
                    cfg.RESULTS_DIR / "equity.parquet",
                )
            except Exception:
                pass
            eq.to_csv(cfg.RESULTS_DIR / "equity.csv", index=False)

        # write decision log for parity debugging (if any decisions collected)
        if getattr(self, "_decision_log", None):
            dec_df = pd.DataFrame(self._decision_log)
            dec_df.to_csv(cfg.RESULTS_DIR / "signal_decisions.csv", index=False)

        # NEW: Save lock timeline
        if getattr(self, "_lock_timeline", None):
            lock_df = pd.DataFrame(self._lock_timeline)
            lock_df.to_csv(cfg.RESULTS_DIR / "lock_timeline.csv", index=False)


def run_backtest(signals_path: Path | None = None):
    """
    Read signals either from a partitioned dataset directory (signals/symbol=*)
    or from a single file signals.parquet, then run the simulation.
    """
    # Resolve path: prefer explicit argument; else choose intelligently
    if signals_path is None:
        part_dirs = list(cfg.SIGNALS_DIR.glob("symbol=*"))
        file_path = cfg.SIGNALS_DIR / "signals.parquet"

        if part_dirs:
            # Partitioned dataset exists: use the directory
            signals_path = cfg.SIGNALS_DIR
        elif file_path.exists():
            # Fallback: single consolidated file
            signals_path = file_path
        else:
            # Nothing at all → graceful exit instead of crashing
            print(
                f"[bt] No signals found in {cfg.SIGNALS_DIR} "
                "(no symbol=* partitions and no signals.parquet). "
                "Nothing to backtest."
            )
            return

    signals_path = Path(signals_path)

    print(f"[bt] reading signals from: {signals_path} (is_dir={signals_path.is_dir()})")


    tester = Backtester(
        initial_capital=float(getattr(cfg, "INITIAL_CAPITAL", 1000.0)),
        risk_pct=float(getattr(cfg, "RISK_PCT", 0.01)),
        max_leverage=float(getattr(cfg, "MAX_LEVERAGE", 10.0)),
    )

    if signals_path.is_dir():
        bs = int(getattr(cfg, "BT_SIGNAL_BATCH_SIZE", 5000))
        sig_iter = iter_partitioned_signals_sorted(signals_path, batch_size=bs)
        tester.run(sig_iter)
    else:
        sig = pd.read_parquet(signals_path, engine="pyarrow")
        if sig.empty:
            print("[bt] Signals file/dataset is empty. Nothing to backtest.")
            return
        sig = _attach_meta_if_needed(sig)
        tester.run(sig)



if __name__ == "__main__":
    run_backtest()