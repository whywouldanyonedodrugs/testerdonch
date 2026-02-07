#!/usr/bin/env python
"""
Diagnose why "core unmatched live" trades in the `signal_no_bt_trade`
bucket were skipped by the backtester.

Outputs:
    results/core_unmatched_skip_reasons.csv
"""

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

import config as cfg
from indicators import resample_ohlcv, atr
from shared_utils import load_parquet_data


def _results_path(name: str) -> Path:
    p = Path(getattr(cfg, "RESULTS_DIR", "results")) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_core_unmatched() -> pd.DataFrame:
    """
    Load the core unmatched live analysis file and filter to the
    'signal_no_bt_trade' bucket.
    """
    core_path = _results_path("parity_unmatched_live_core.csv")
    if not core_path.exists():
        raise FileNotFoundError(
            f"{core_path} not found. "
            "Make sure you've run analyze_core_unmatched_live.py first."
        )

    core = pd.read_csv(core_path)
    # Expect at least: symbol, signal_ts, classification, entry_ts_live, exit_ts_live, ...
    if "classification" not in core.columns:
        raise ValueError("Expected 'classification' column in core unmatched CSV.")

    # Normalise symbol casing
    if "symbol" not in core.columns:
        raise ValueError("Expected 'symbol' column in core unmatched CSV.")

    core["symbol"] = core["symbol"].astype(str).str.upper()

    # Parse timestamps we know we need; missing columns are handled with errors
    for col in ["signal_ts", "entry_ts_live", "exit_ts_live"]:
        if col in core.columns:
            core[col] = pd.to_datetime(core[col], utc=True, errors="coerce")

    core = core[core["classification"] == "signal_no_bt_trade"].copy()
    core = core.dropna(subset=["signal_ts"])
    core = core.reset_index(drop=True)

    print(f"[debug] Loaded {len(core)} core unmatched 'signal_no_bt_trade' rows.")
    return core


def _load_backtest_trades() -> pd.DataFrame:
    """
    Load backtester trades.csv to reconstruct cooldown and daycap effects.
    """
    trades_path = _results_path("trades.csv")
    if not trades_path.exists():
        raise FileNotFoundError(
            f"{trades_path} not found. Run the backtester first."
        )

    trades = pd.read_csv(trades_path, parse_dates=["entry_ts", "exit_ts"])
    trades["symbol"] = trades["symbol"].astype(str).str.upper()
    trades = trades.dropna(subset=["entry_ts", "exit_ts"])
    trades = trades.sort_values(["symbol", "exit_ts"]).reset_index(drop=True)
    print(f"[debug] Loaded {len(trades)} backtest trades.")
    return trades


def _build_exit_index_by_symbol(trades: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    For cooldown reconstruction: map symbol -> DataFrame with sorted exit_ts.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sym, g in trades.groupby("symbol"):
        out[sym] = g[["exit_ts"]].sort_values("exit_ts").reset_index(drop=True)
    return out


def _load_5m_with_atr(sym: str) -> pd.DataFrame:
    """
    Reproduce Backtester._get_5m for a single symbol:

      - load 5m OHLCV via load_parquet_data
      - compute ATR on cfg.ATR_TIMEFRAME (e.g. '1h') and ffill to 5m
        or directly on 5m if ATR_TIMEFRAME is None
      - store as 'atr_pre'
    """
    start_date = getattr(cfg, "START_DATE", None)
    end_date = getattr(cfg, "END_DATE", None)

    df = load_parquet_data(
        sym,
        start_date=start_date,
        end_date=end_date,
        drop_last_partial=True,
        columns=["open", "high", "low", "close", "volume"],
    ).copy()

    if df.empty:
        raise ValueError(f"No 5m data for {sym} in [{start_date}, {end_date}]")

    tf = getattr(cfg, "ATR_TIMEFRAME", None)
    atr_len = int(getattr(cfg, "ATR_LEN", 14))

    if tf:
        dft = resample_ohlcv(df, str(tf))
        atr_tf = atr(dft, atr_len)
        df["atr_pre"] = atr_tf.reindex(df.index, method="ffill")
    else:
        df["atr_pre"] = atr(df, atr_len)

    return df


def _align_ts_and_get_atr_price(
    df5: pd.DataFrame, ts: pd.Timestamp
) -> Tuple[pd.Timestamp, float, float]:
    """
    Match Backtester.run ts alignment logic:

      - if ts not in df5 index: move to the first index >= ts
      - if there is no such index: return (NaT, nan, nan)
    """
    if ts not in df5.index:
        later_idx = df5.index[df5.index >= ts]
        if len(later_idx) == 0:
            return pd.NaT, np.nan, np.nan
        ts = later_idx[0]

    atr_now = float(df5.loc[ts, "atr_pre"])
    entry_price = float(df5.loc[ts, "close"])
    return ts, atr_now, entry_price


def _is_locked_by_cooldown(
    sym: str,
    ts: pd.Timestamp,
    exits_by_sym: Dict[str, pd.DataFrame],
    cooldown_min: float,
) -> Tuple[bool, Optional[pd.Timestamp]]:
    """
    Reconstruct symbol-level cooldown from trades.csv:

      - find last exit_ts <= ts for that sym
      - compute lock_until = exit_ts + cooldown_min
      - locked if ts <= lock_until
    """
    df = exits_by_sym.get(sym)
    if df is None or df.empty or ts is pd.NaT:
        return False, None

    before = df[df["exit_ts"] <= ts]
    if before.empty:
        return False, None

    last_exit = before["exit_ts"].iloc[-1]
    lock_until = last_exit + pd.Timedelta(minutes=cooldown_min)
    return bool(ts <= lock_until), last_exit


def _maybe_daycap_blocked(
    ts: pd.Timestamp,
    trades: pd.DataFrame,
    max_trades_per_day: Optional[int],
) -> bool:
    """
    Approximate whether MAX_TRADES_PER_DAY could have blocked a trade
    on this signal's day.

    This is conservative: if no day ever hits the cap, this returns False
    everywhere. If some days do hit the cap, we just mark any signal on
    that day as 'daycap_possible'.
    """
    if max_trades_per_day is None or max_trades_per_day <= 0 or ts is pd.NaT:
        return False

    day = ts.floor("D")
    day_mask = trades["entry_ts"].dt.floor("D") == day
    trades_this_day = int(day_mask.sum())
    return trades_this_day >= int(max_trades_per_day)


def main():
    core = _load_core_unmatched()
    trades = _load_backtest_trades()

    exits_by_sym = _build_exit_index_by_symbol(trades)

    cooldown_min = float(getattr(cfg, "SYMBOL_COOLDOWN_MINUTES", 0.0))
    max_trades_per_day = getattr(cfg, "MAX_TRADES_PER_DAY", None)
    min_atr_pct = float(getattr(cfg, "MIN_ATR_PCT_OF_PRICE", 0.0001))

    # Preload 5m+ATR for symbols we actually need
    symbols = sorted(core["symbol"].unique())
    df5_cache: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            df5_cache[sym] = _load_5m_with_atr(sym)
            print(f"[debug] Loaded 5m+ATR for {sym} ({len(df5_cache[sym])} rows).")
        except Exception as e:
            print(f"[warn] Failed to load 5m+ATR for {sym}: {e}")
            df5_cache[sym] = pd.DataFrame()

    rows = []

    for _, row in core.iterrows():
        sym = row["symbol"]
        sig_ts = row["signal_ts"]

        df5 = df5_cache.get(sym, pd.DataFrame())
        ts_aligned = pd.NaT
        atr_now = np.nan
        entry_price = np.nan
        atr_pct = np.nan
        atr_blocked = False
        reasons = []

        # --- ATR / price guard (if data is available) ---
        if not df5.empty:
            ts_aligned, atr_now, entry_price = _align_ts_and_get_atr_price(df5, sig_ts)
            if not np.isfinite(atr_now) or atr_now <= 0:
                atr_blocked = True
                reasons.append("atr_nan_or_nonpositive")
            else:
                if entry_price > 0:
                    atr_pct = atr_now / entry_price
                    if atr_pct < min_atr_pct:
                        atr_blocked = True
                        reasons.append("atr_too_small")
                else:
                    atr_blocked = True
                    reasons.append("entry_price_nonpositive")
        else:
            # No data for symbol despite being 'core'; treat as data issue
            reasons.append("no_5m_data")

        # --- Cooldown reconstruction ---
        cooldown_blocked, last_exit = _is_locked_by_cooldown(
            sym=sym,
            ts=ts_aligned if ts_aligned is not pd.NaT else sig_ts,
            exits_by_sym=exits_by_sym,
            cooldown_min=cooldown_min,
        )
        if cooldown_blocked:
            reasons.append("cooldown")

        # --- Daycap approximation ---
        daycap_blocked = _maybe_daycap_blocked(
            ts=ts_aligned if ts_aligned is not pd.NaT else sig_ts,
            trades=trades,
            max_trades_per_day=max_trades_per_day,
        )
        if daycap_blocked:
            reasons.append("daycap")

        # --- Decide primary_reason by a simple priority order ---
        primary_reason = "unknown"
        priority = ["no_5m_data", "atr_nan_or_nonpositive", "atr_too_small",
                    "entry_price_nonpositive", "cooldown", "daycap"]

        for r in priority:
            if r in reasons:
                primary_reason = r
                break

        out_row = row.to_dict()
        out_row.update(
            {
                "ts_aligned": ts_aligned,
                "atr_now": atr_now,
                "atr_pct_price": atr_pct,
                "cooldown_blocked": cooldown_blocked,
                "cooldown_last_exit": last_exit,
                "daycap_blocked": daycap_blocked,
                "primary_reason": primary_reason,
            }
        )
        rows.append(out_row)

    out = pd.DataFrame(rows)
    out_path = _results_path("core_unmatched_skip_reasons.csv")
    out.to_csv(out_path, index=False)
    print(f"[done] Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
