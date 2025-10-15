# backtester.py
from __future__ import annotations




from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional, Tuple

import config as cfg
from indicators import resample_ohlcv, atr
from bt_intrabar import resolve_first_touch_1m
from shared_utils import load_parquet_data

pd.set_option("future.no_silent_downcasting", True)

from regime_detector import compute_daily_combined_regime, DailyRegimeConfig, compute_markov_regime_4h
from config import META_PROB_THRESHOLD, META_SIZING_ENABLED, RISK_PCT

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
# ---- backtester.py (imports near top) ----
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# ---- inside backtester.py ----
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

        # Load base (5m) ETH and resample to the configured TF (e.g., "4h")
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
    # NEW diagnostics
    mae_over_atr: Optional[float] = None
    mfe_over_atr: Optional[float] = None
    markov_state_4h: Optional[int] = None
    markov_prob_up_4h: Optional[float] = None
    trend_regime_1d: Optional[str] = None
    vol_regime_1d: Optional[str] = None
    vol_prob_low_1d: Optional[float] = None
    regime_code_1d: Optional[int] = None
    regime_1d: Optional[str] = None
    markov_state_up_4h: Optional[int] = None
    markov_prob_up_4h: Optional[float] = None

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
    # optional overrides so the caller can inject meta-weighted sizing etc.
    risk_mode_override: str | None = None,
    risk_pct_override: float | None = None,
    fixed_cash_override: float | None = None,
    # NEW: AVWAP anchor override (None → defaults to entry_ts)
    avwap_anchor_ts: pd.Timestamp | None = None,
) -> dict | None:
    """
    Simulate a long trade with:
      - single final TP
      - optional TP1 partial + move-SL-to-break-even
      - optional trailing stop after TP1
      - optional time stop
      - deterministic intrabar resolution via _resolve_intrabar(..)

    Returns a dict with exit info, pnl_cash, fees, qty_filled, and diagnostics.
    """

    # --- config
    fee_rate   = float(getattr(cfg, "FEE_RATE", 0.00055))
    time_hours = getattr(cfg, "TIME_EXIT_HOURS", None)

    # legacy ATR-price basis (default)
    sl_mult_legacy = float(getattr(cfg, "SL_ATR_MULT", 2.0))
    tp_mult_legacy = float(getattr(cfg, "TP_ATR_MULT", 8.0))

    # AVWAP basis (new)
    exit_basis   = str(getattr(cfg, "EXIT_BASIS", "price_atr")).lower()
    av_mode      = str(getattr(cfg, "AVWAP_MODE", "static")).lower()      # "static" | "dynamic"
    av_anchor    = pd.to_datetime(avwap_anchor_ts or entry_ts, utc=True)
    av_sl_mult   = float(getattr(cfg, "AVWAP_SL_MULT", 2.0))
    av_tp_mult   = float(getattr(cfg, "AVWAP_TP_MULT", 8.0))
    av_use_entry_atr = bool(getattr(cfg, "AVWAP_USE_ENTRY_ATR", True))

    # partials / trail
    partial_on = bool(getattr(cfg, "PARTIAL_TP_ENABLED", False))
    tp1_mult   = float(getattr(cfg, "PARTIAL_TP1_ATR_MULT", 5.0))
    tp1_ratio  = float(getattr(cfg, "PARTIAL_TP_RATIO", 0.5))
    move_be    = bool(getattr(cfg, "MOVE_SL_TO_BE_ON_TP1", True))

    trail_on   = bool(getattr(cfg, "TRAIL_AFTER_TP1", False))
    trail_mult = float(getattr(cfg, "TRAIL_ATR_MULT", 1.0))
    use_hwm    = bool(getattr(cfg, "TRAIL_USE_HIGH_WATERMARK", True))

    # --- vectorized AVWAP (for avwap_atr modes)
    hlc3  = (df5["high"].astype(float) + df5["low"].astype(float) + df5["close"].astype(float)) / 3.0
    vol   = df5["volume"].astype(float)
    pv_cum = (hlc3 * vol).cumsum()
    v_cum  = vol.cumsum()

    # index position just before/at the anchor
    if av_anchor <= df5.index[0]:
        pv0 = 0.0; v0 = 0.0
    else:
        pre = df5.loc[df5.index < av_anchor]
        if pre.empty:
            pv0 = 0.0; v0 = 0.0
        else:
            pv0 = float(pv_cum.loc[pre.index[-1]])
            v0  = float(v_cum.loc[pre.index[-1]])

    def avwap_at(ts: pd.Timestamp) -> float:
        """AVWAP from anchor up to as-of ts (inclusive if ts in index)."""
        upto = df5.loc[:ts]
        if upto.empty:
            return float(hlc3.iloc[0])
        pv = float(pv_cum.loc[upto.index[-1]]) - pv0
        vv = float(v_cum.loc[upto.index[-1]]) - v0
        if vv <= 0:
            return float(hlc3.loc[upto.index[-1]])
        return pv / vv

    # --- initial SL/TP levels
    if exit_basis == "avwap_atr":
        av_at_entry = avwap_at(entry_ts)
        atr_ref = atr_at_entry if av_use_entry_atr else float(df5.loc[entry_ts, "atr_pre"])
        sl_initial = av_at_entry - av_sl_mult * atr_ref
        tp         = av_at_entry + av_tp_mult * atr_ref
        tp1        = (av_at_entry + tp1_mult * atr_ref) if partial_on else None
        # guards: keep long's SL below entry, TP above entry
        sl_initial = min(sl_initial, entry_price - 1e-8)
        tp         = max(tp,        entry_price + 1e-8)
        if tp1 is not None:
            tp1 = max(tp1, entry_price + 1e-8)
        trail_gap  = (trail_mult * atr_ref) if trail_on else None
    else:
        # legacy price±ATR_at_entry
        sl_initial = entry_price - sl_mult_legacy * atr_at_entry
        tp         = entry_price + tp_mult_legacy * atr_at_entry
        tp1        = (entry_price + tp1_mult * atr_at_entry) if partial_on else None
        trail_gap  = (trail_mult * atr_at_entry) if trail_on else None

    sl_final = sl_initial  # mutable (can move to BE on TP1)

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

    fees = fee_rate * abs(qty * entry_price)

    # --- path after entry
    walk = df5.loc[df5.index > entry_ts].copy()
    if walk.empty:
        return None

    took_tp1 = False
    remaining_qty = qty
    highest_since_tp1 = entry_price
    curr_trail = None
    tp1_fill_px: float | None = None  # capture realized partial price

    t_exit: pd.Timestamp | None = None
    px_exit: float | None = None
    reason: str | None = None

    # MAE/MFE vs entry (price units; converted to ATR later)
    mae = 0.0
    mfe = 0.0

    deadline = entry_ts + pd.Timedelta(hours=float(time_hours)) if time_hours is not None else None

    for ts, row in walk.iterrows():
        o, h, l, c = map(float, (row["open"], row["high"], row["low"], row["close"]))

        # excursions
        mae = min(mae, l - entry_price)
        mfe = max(mfe, h - entry_price)

        # trailing after TP1
        if took_tp1 and trail_on:
            highest_since_tp1 = max(highest_since_tp1, h) if use_hwm else c
            curr_trail = highest_since_tp1 - trail_gap  # price level

        # per-bar levels
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

        touch = _resolve_intrabar(symbol, ts, levels, {"open": o, "high": h, "low": l, "close": c})

        if touch is None:
            if deadline is not None and ts >= deadline:
                t_exit, px_exit, reason = ts, c, "time"
                break
            continue

        if touch == "sl":
            t_exit, px_exit, reason = ts, levels["sl"], "sl"
            break

        if touch == "tp":
            t_exit, px_exit, reason = ts, levels["tp"], "tp_final"
            break

        if touch == "tp1":
            fill_px = float(levels["tp1"])
            take_qty = remaining_qty * tp1_ratio
            remaining_qty -= take_qty
            fees += fee_rate * abs(take_qty * fill_px)
            took_tp1 = True
            tp1_fill_px = fill_px
            highest_since_tp1 = max(highest_since_tp1, h)
            if move_be:
                sl_final = entry_price  # move *final* SL to BE
            continue

        if touch == "trail":
            t_exit, px_exit, reason = ts, levels["trail"], "trail"
            break

    # if still open at the end, close at last bar close (or time stop handled above)
    if t_exit is None:
        last_ts = walk.index[-1]
        t_exit, px_exit, reason = last_ts, float(walk.iloc[-1]["close"]), "time"

    # exit fee on remaining position
    fees += fee_rate * abs(remaining_qty * px_exit)

    # --- PnL in cash (partials realized at TP1)
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
        pnl_cash=float(pnl_cash),      # <-- required by caller
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
        if getattr(self, "_daily_regime", None) is None or self._daily_regime.empty:
            return None, None, None, None, None
        row = self._daily_regime.loc[:ts].iloc[-1:] if ts >= self._daily_regime.index[0] else pd.DataFrame()
        if row.empty:
            return None, None, None, None, None
        r = row.iloc[0]
        return (
            str(r.get("trend_regime")),
            str(r.get("vol_regime")),
            float(r.get("vol_prob_low")) if pd.notna(r.get("vol_prob_low")) else None,
            int(r.get("regime_code")) if pd.notna(r.get("regime_code")) else None,
            str(r.get("regime")),
        )

    def __init__(self, initial_capital: float, risk_pct: float, max_leverage: float):
        self.initial_capital = float(initial_capital)
        self.equity = float(initial_capital)
        self.risk_pct = float(risk_pct)
        self.max_leverage = float(max_leverage)
        self.regime = RegimeGate()

        try:
            self._daily_regime = compute_daily_combined_regime(DailyRegimeConfig(save_path=None))
        except Exception as e:
            print(f"[backtester] daily regime build failed: {e}")
            self._daily_regime = pd.DataFrame()

        self._cache_5m: Dict[str, pd.DataFrame] = {}
        self._cache_1m: Dict[str, pd.DataFrame] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

        # Throughput guards
        self._lock_until: Dict[str, pd.Timestamp] = {}
        self._trades_per_day: Dict[pd.Timestamp, int] = {}
        self._skipped = {"lock": 0, "daycap": 0}

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
    def _get_5m(self, sym: str) -> pd.DataFrame:
        if sym not in self._cache_5m:
            df = load_parquet_data(
                sym,
                start_date=cfg.START_DATE,
                end_date=cfg.END_DATE,
                drop_last_partial=True,
                columns=["open", "high", "low", "close", "volume"],
            ).copy()
            # PRECOMPUTE ATR ON CONFIGURED TF (e.g., "1h") AND FFILL TO 5m
            tf = getattr(cfg, "ATR_TIMEFRAME", None)
            if tf:
                dft = resample_ohlcv(df, str(tf))
                atr_tf = atr(dft, int(getattr(cfg, "ATR_LEN", 14)))
                df["atr_pre"] = atr_tf.reindex(df.index, method="ffill")
            else:
                df["atr_pre"] = atr(df, int(getattr(cfg, "ATR_LEN", 14)))
            self._cache_5m[sym] = df
        return self._cache_5m[sym]

    def _get_1m(self, sym: str) -> Optional[pd.DataFrame]:
        if not getattr(cfg, "USE_INTRABAR_1M", False):
            return None
        if sym not in self._cache_1m:
            p = cfg.PARQUET_1M_DIR / f"{sym}.parquet"
            if not p.exists():
                return None
            df = pd.read_parquet(p)
            df = _ensure_dt(df)[["open", "high", "low", "close", "volume"]]
            self._cache_1m[sym] = df
        return self._cache_1m[sym]

    # ----- main loop -----
    def run(self, signals: pd.DataFrame):
        if signals.empty:
            print("No signals to backtest.")
            return

        sig = signals.copy()
        if getattr(cfg, "RS_ENABLED", True) and "rs_pct" in sig.columns and getattr(cfg, "RS_MIN_PERCENTILE", None) is not None:
            sig = sig[(sig["rs_pct"].fillna(-1) >= cfg.RS_MIN_PERCENTILE)]

        sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True)
        sig = sig.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        for _, s in sig.iterrows():
            ts = s["timestamp"]
            sym = str(s["symbol"])
            entry_price = float(s["entry"])

            # --- regime gate
            regime_up = self.regime.is_up(ts)
            if getattr(cfg, "REGIME_FILTER_ENABLED", True) and getattr(cfg, "REGIME_BLOCK_WHEN_DOWN", True) and not regime_up:
                continue

            # --- data align
            df5 = self._get_5m(sym)
            if ts not in df5.index:
                idx = df5.index[df5.index >= ts]
                if len(idx) == 0:
                    continue
                ts = idx[0]
                entry_price = float(df5.loc[ts, "close"])

            # --- throughput guards (computed after aligning 'ts')
            day = ts.floor("D")
            if (getattr(cfg, "MAX_TRADES_PER_DAY", None) is not None) and (not getattr(cfg, "LABELING_MODE", False)):
                if self._trades_per_day.get(day, 0) >= int(cfg.MAX_TRADES_PER_DAY):
                    self._skipped["daycap"] += 1
                    continue

            if not getattr(cfg, "LABELING_MODE", False):
                lock_until = self._lock_until.get(sym)
                if lock_until is not None and ts <= lock_until:
                    self._skipped["lock"] += 1
                    continue

            atr_now = float(df5.loc[ts, "atr_pre"])
            if not np.isfinite(atr_now) or atr_now <= 0:
                continue

            min_atr_pct = float(getattr(cfg, "MIN_ATR_PCT_OF_PRICE", 0.0001))
            if (atr_now / entry_price) < min_atr_pct:
                continue

            # --- regime size-down (if not blocked)
            equity_for_sizing = self.equity
            if (not regime_up) and (not getattr(cfg, "REGIME_BLOCK_WHEN_DOWN", True)):
                equity_for_sizing *= float(getattr(cfg, "REGIME_SIZE_WHEN_DOWN", 0.5))

            # === meta: gate + probability-weighted sizing ===
            prob = s.get("meta_p", np.nan)  # may not exist in plain signals; OK

            # gate (only if user set META_PROB_THRESHOLD)
            thr = getattr(cfg, "META_PROB_THRESHOLD", None)
            if thr is not None:
                try:
                    p = float(prob)
                    if not (0.0 <= p <= 1.0):
                        # if prob missing, *do not* auto-skip; just don’t pass the gate
                        continue
                    if p < float(thr):
                        continue
                except Exception:
                    continue

            # base risk pct
            risk_pct_eff = float(getattr(cfg, "RISK_PCT", 0.01))

            # user-provided scale wins; else scale by prob if enabled
            scale_in = s.get("risk_scale", None)
            if scale_in is not None:
                try:
                    risk_pct_eff *= float(scale_in)
                except Exception:
                    pass
            elif bool(getattr(cfg, "META_SIZING_ENABLED", False)):
                risk_pct_eff *= _prob_size_multiplier(prob)

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
                risk_mode_override=getattr(cfg, "RISK_MODE", None),
                risk_pct_override=getattr(cfg, "RISK_PCT", None),
                fixed_cash_override=getattr(cfg, "FIXED_RISK_CASH", None),
                avwap_anchor_ts=anchor_ts,
            )
            if sim is None:
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

            self.trades.append(Trade(
                trade_id=len(self.trades)+1, symbol=sym, entry_ts=ts, exit_ts=exit_ts,
                entry=entry_price, exit=exit_price, qty=qty, side="long",
                sl=sl, tp=tp, exit_reason=exit_reason, atr_at_entry=atr_now,
                regime_up=bool(regime_up),
                rs_pct=float(s.get("rs_pct")) if "rs_pct" in s and pd.notna(s["rs_pct"]) else None,
                pullback_type=s.get("pullback_type"),
                entry_rule=s.get("entry_rule"),
                don_break_len=int(s.get("don_break_len")) if pd.notna(s.get("don_break_len")) else None,
                fees=float(fees), pnl=float(pnl), pnl_R=float(pnl_R),
                mae_over_atr=float(sim["mae_over_atr"]), mfe_over_atr=float(sim["mfe_over_atr"]),
                trend_regime_1d=trend1d,
                vol_regime_1d=vol1d,
                vol_prob_low_1d=vprob1d,
                regime_code_1d=rcode1d,
                regime_1d=rstr1d,
                markov_state_up_4h=m4_state,
                markov_prob_up_4h=m4_prob
            ))

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
                self._lock_until[sym] = (exit_ts or ts) + cd
                self._trades_per_day[day] = self._trades_per_day.get(day, 0) + 1

        # finalize
        self._save_outputs()
        if any(self._skipped.values()):
            print(f"[throughput] skipped due to lock={self._skipped['lock']} daycap={self._skipped['daycap']}")

    def _save_outputs(self):
        # Build trades DataFrame robustly (even if there are zero fills)
        if not self.trades:
            cols = [f.name for f in fields(Trade)]
            trades_df = pd.DataFrame(columns=cols)
        else:
            trades_df = pd.DataFrame([asdict(t) for t in self.trades])
            if "entry_ts" in trades_df.columns:
                trades_df = trades_df.sort_values("entry_ts")

        # write trades
        try:
            pq.write_table(pa.Table.from_pandas(trades_df, preserve_index=False),
                           cfg.RESULTS_DIR / "trades.parquet")
        except Exception:
            pass
        trades_df.to_csv(cfg.RESULTS_DIR / "trades.csv", index=False)

        # write equity if available
        if self.equity_curve:
            eq = (pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
                  .sort_values("timestamp"))
            try:
                pq.write_table(pa.Table.from_pandas(eq, preserve_index=False),
                               cfg.RESULTS_DIR / "equity.parquet")
            except Exception:
                pass
            eq.to_csv(cfg.RESULTS_DIR / "equity.csv", index=False)


def run_backtest(signals_path: Path | None = None):
    """
    Read signals either from a partitioned dataset directory (signals/symbol=*)
    or from a single file signals.parquet, then run the simulation.
    """
    # Resolve path: prefer explicit argument; else choose intelligently
    if signals_path is None:
        # If partitioned dataset exists, use the directory; else fall back to file
        if any(cfg.SIGNALS_DIR.glob("symbol=*")):
            signals_path = cfg.SIGNALS_DIR
        else:
            signals_path = cfg.SIGNALS_DIR / "signals.parquet"

    signals_path = Path(signals_path)
    if not signals_path.exists():
        raise FileNotFoundError(f"signals path not found: {signals_path}")

    print(f"[bt] reading signals from: {signals_path} (is_dir={signals_path.is_dir()})")

    # Force PyArrow — reliable for reading a directory of partitioned Parquet files.
    # Pandas supports passing a directory that contains partitioned Parquet. :contentReference[oaicite:2]{index=2}
    sig = pd.read_parquet(signals_path, engine="pyarrow")
    if sig.empty:
        print("No signals to backtest.")
        return

    # Attach meta only if enabled (sorting handled inside)
    sig = _attach_meta_if_needed(sig)

    tester = Backtester(
        initial_capital=float(getattr(cfg, "INITIAL_CAPITAL", 1000.0)),
        risk_pct=float(getattr(cfg, "RISK_PCT", 0.01)),
        max_leverage=float(getattr(cfg, "MAX_LEVERAGE", 10.0)),
    )
    tester.run(sig)


if __name__ == "__main__":
    run_backtest()
