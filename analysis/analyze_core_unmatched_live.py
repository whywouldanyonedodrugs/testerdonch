#!/usr/bin/env python
"""
Analyze "core" unmatched LIVE trades (all gates OK, no data holes) and
classify why the backtester did not produce a matching trade.

Classification per live trade:

  - no_signals_for_symbol : no signals parquet at all for this symbol, or
                            no usable time column could be inferred
  - no_signal_in_window   : signals exist for symbol, but none in lookback window
  - signal_and_bt_trade   : there is at least one signal AND at least one
                            backtest trade near that signal (entry time)
  - signal_no_bt_trade    : there is at least one signal in window, but
                            zero backtest trades near any such signal

Results are written to:
  results/core_unmatched_live_analysis.csv

and a summary of classification counts is printed.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as cfg  # backtester config


# --- Paths / constants -------------------------------------------------------

RESULTS_DIR: Path = Path(getattr(cfg, "RESULTS_DIR", "results"))
SIGNALS_DIR: Path = Path(getattr(cfg, "SIGNALS_DIR", "signals"))

PARITY_UNMATCHED_LIVE_CSV = RESULTS_DIR / "parity_unmatched_live.csv"
BACKTEST_TRADES_CSV = RESULTS_DIR / "trades.csv"

# How far back from live exit we search for signals (in hours)
PARITY_LOOKBACK_HOURS: int = int(getattr(cfg, "PARITY_LOOKBACK_HOURS", 72))

# How close (in minutes) a backtest trade entry must be to a signal to count
PARITY_SIGNAL_MATCH_MINUTES: int = int(
    getattr(cfg, "PARITY_SIGNAL_MATCH_MINUTES", 30)
)


# --- Helpers -----------------------------------------------------------------


def _infer_time_column(df: pd.DataFrame, sym: str) -> Optional[pd.Series]:
    """
    Try to infer the correct time column from a signals dataframe.

    Priority:
      1) If cfg.SIGNAL_TIME_COL exists and is in df, use it.
      2) Common candidate names ['ts', 'signal_ts', 'entry_ts', 'bar_ts',
         'time', 'timestamp', 'open_time', 'close_time'].
      3) Any column that already has datetime dtype.
      4) Any column that can be parsed by to_datetime with at least one
         non-NaT value.

    Returns a pandas Series (datetime64[ns, UTC]) or None if inference fails.
    """
    # 1) Explicit config override
    time_col_cfg = getattr(cfg, "SIGNAL_TIME_COL", None)
    if time_col_cfg and time_col_cfg in df.columns:
        s = pd.to_datetime(df[time_col_cfg], utc=True, errors="coerce")
        if not s.isna().all():
            return s

    # 2) Try common names
    candidate_names = [
        "ts",
        "signal_ts",
        "entry_ts",
        "bar_ts",
        "time",
        "timestamp",
        "open_time",
        "close_time",
        "dt",
    ]

    for name in candidate_names:
        if name in df.columns:
            s = pd.to_datetime(df[name], utc=True, errors="coerce")
            if not s.isna().all():
                return s

    # 3) Already datetime-typed columns
    #    (datetime64[ns] or datetime64[ns, tz])
    datetime_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            datetime_cols.append(c)

    if len(datetime_cols) == 1:
        s = df[datetime_cols[0]]
        # ensure UTC
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
        return s
    elif len(datetime_cols) > 1:
        # If multiple, just pick the first non-empty one
        for c in datetime_cols:
            s = df[c]
            if not s.isna().all():
                if s.dt.tz is None:
                    s = s.dt.tz_localize("UTC")
                else:
                    s = s.dt.tz_convert("UTC")
                return s

    # 4) Last resort: brute-force over all columns and see if any can be parsed
    for c in df.columns:
        s = pd.to_datetime(df[c], utc=True, errors="coerce")
        if not s.isna().all():
            return s

    print(
        f"[signals] WARNING: could not infer any datetime column for symbol={sym}; "
        f"columns={list(df.columns)}"
    )
    return None


def load_signals_for_symbol(symbol: str) -> pd.DataFrame:
    """
    Load all scout signals for a given symbol from the partitioned signals dir.

    Expects files like:  signals/symbol=XYZUSDT/part-*.parquet

    Attempts to infer the time column robustly using _infer_time_column.
    """
    sym = symbol.upper()
    pattern = str(SIGNALS_DIR / f"symbol={sym}" / "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[signals] no files for {sym} under {SIGNALS_DIR}")
        return pd.DataFrame()

    dfs = []
    for fn in files:
        table = pq.read_table(fn)
        df = table.to_pandas()
        dfs.append(df)

    sig = pd.concat(dfs, ignore_index=True)

    # Normalise symbol
    if "symbol" in sig.columns:
        sig["symbol"] = sig["symbol"].astype(str).str.upper()
    else:
        sig["symbol"] = sym

    # Infer time column
    ts = _infer_time_column(sig, sym)
    if ts is None:
        # Treat as having no signals for this symbol
        # to avoid crashing the analysis.
        print(
            f"[signals] WARNING: treating {sym} as having no usable signals due to missing/inferable time column."
        )
        return pd.DataFrame()

    sig["ts"] = ts
    sig = sig.dropna(subset=["ts"])
    sig = sig.sort_values("ts").reset_index(drop=True)
    return sig


def load_unmatched_live_core() -> pd.DataFrame:
    """
    Load parity_unmatched_live.csv and return the "core" subset:

      - all_gates_ok == True (computed from rs_ok, liq_ok, vol_ok, micro_ok
        if not already present)
      - no data holes in rs_pct, liq_usd, vol_mult, micro_vol_ratio
    """
    if not PARITY_UNMATCHED_LIVE_CSV.exists():
        raise FileNotFoundError(f"{PARITY_UNMATCHED_LIVE_CSV} not found.")

    df = pd.read_csv(PARITY_UNMATCHED_LIVE_CSV)

    if "exit_ts" not in df.columns:
        raise KeyError("parity_unmatched_live.csv must contain an 'exit_ts' column.")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)

    if "symbol" not in df.columns:
        raise KeyError("parity_unmatched_live.csv must contain a 'symbol' column.")
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # Compute data holes: any of these features missing
    feature_cols = [c for c in ["rs_pct", "liq_usd", "vol_mult", "micro_vol_ratio"] if c in df.columns]
    if feature_cols:
        df["data_hole"] = df[feature_cols].isna().any(axis=1)
    else:
        # If we don't have those columns at all, treat everything as data hole
        df["data_hole"] = True

    # Compute all_gates_ok from gate booleans if needed
    if "all_gates_ok" not in df.columns:
        gate_cols = ["rs_ok", "liq_ok", "vol_ok", "micro_ok"]
        missing_gates = [c for c in gate_cols if c not in df.columns]
        if missing_gates:
            raise KeyError(
                "parity_unmatched_live.csv must contain either 'all_gates_ok' or all of "
                f"{gate_cols}. Missing: {missing_gates}"
            )
        # Ensure boolean
        for c in gate_cols:
            if df[c].dtype != bool:
                df[c] = df[c].astype(bool)
        df["all_gates_ok"] = df[gate_cols].all(axis=1)

    # Core = all gates OK AND no data holes
    core = df[(df["all_gates_ok"]) & (~df["data_hole"])].copy()
    core = core.reset_index(drop=True)
    print(
        f"[core] unmatched LIVE trades: {len(df)} → core (all_gates_ok & no data holes): {len(core)}"
    )
    return core


def load_backtest_trades() -> pd.DataFrame:
    """
    Load backtester trades from results/trades.csv and normalise timestamps.
    """
    if not BACKTEST_TRADES_CSV.exists():
        raise FileNotFoundError(f"{BACKTEST_TRADES_CSV} not found.")

    df = pd.read_csv(BACKTEST_TRADES_CSV)
    for col in ["entry_ts", "exit_ts"]:
        if col not in df.columns:
            raise KeyError(f"trades.csv must contain column '{col}'")
        df[col] = pd.to_datetime(df[col], utc=True)

    if "symbol" not in df.columns:
        raise KeyError("trades.csv must contain a 'symbol' column.")

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df.sort_values(["symbol", "entry_ts"]).reset_index(drop=True)
    return df


# --- Core analysis -----------------------------------------------------------


def explain_core_unmatched_live() -> pd.DataFrame:
    """
    For each core unmatched LIVE trade, classify why the backtester did not
    produce a matching trade, based on signals and trades.

    Returns a dataframe with one row per core unmatched LIVE trade
    and writes it to results/core_unmatched_live_analysis.csv
    """
    core = load_unmatched_live_core()
    bt_trades = load_backtest_trades()

    # Cache signals per symbol to avoid re-reading
    signals_cache: Dict[str, pd.DataFrame] = {}

    lookback = pd.Timedelta(hours=PARITY_LOOKBACK_HOURS)
    dt_signal_match = pd.Timedelta(minutes=PARITY_SIGNAL_MATCH_MINUTES)

    rows = []

    for idx, row in core.iterrows():
        sym = row["symbol"]
        t_exit = row["exit_ts"]

        # Load / reuse signals for this symbol
        sig = signals_cache.get(sym)
        if sig is None:
            sig = load_signals_for_symbol(sym)
            signals_cache[sym] = sig

        classification: str
        signal_ts: Optional[pd.Timestamp] = None
        n_signals_in_window: int = 0
        n_bt_trades_near_signal: int = 0

        if sig.empty:
            classification = "no_signals_for_symbol"
        else:
            t_min = t_exit - lookback
            mask = (sig["ts"] >= t_min) & (sig["ts"] <= t_exit)
            cand = sig.loc[mask]

            if cand.empty:
                classification = "no_signal_in_window"
            else:
                # For now, take the last signal before exit as the most likely one
                j = cand["ts"].idxmax()
                signal_ts = cand.loc[j, "ts"]
                n_signals_in_window = len(cand)

                bt_sym = bt_trades[bt_trades["symbol"] == sym]
                if bt_sym.empty:
                    classification = "signal_no_bt_trade"
                else:
                    # Any bt trade whose entry is close to this signal_ts?
                    t0 = signal_ts - dt_signal_match
                    t1 = signal_ts + dt_signal_match
                    bt_near = bt_sym[
                        (bt_sym["entry_ts"] >= t0) & (bt_sym["entry_ts"] <= t1)
                    ]
                    n_bt_trades_near_signal = len(bt_near)

                    if n_bt_trades_near_signal > 0:
                        classification = "signal_and_bt_trade"
                    else:
                        classification = "signal_no_bt_trade"

        out_row = {
            "symbol": sym,
            "exit_ts_live": t_exit,
            "classification": classification,
            "n_signals_in_window": n_signals_in_window,
            "n_bt_trades_near_signal": n_bt_trades_near_signal,
            "signal_ts": signal_ts,
        }

        # carry over some useful features if present
        for col in ["rs_pct", "liq_usd", "vol_mult", "micro_vol_ratio", "pnl_live"]:
            if col in core.columns:
                out_row[col] = row[col]

        rows.append(out_row)

    result = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "core_unmatched_live_analysis.csv"
    result.to_csv(out_path, index=False)
    print(f"[info] wrote core unmatched LIVE analysis to: {out_path}")

    print("\n=== Classification counts (core unmatched LIVE) ===")
    print(result["classification"].value_counts())

    return result


def main() -> None:
    explain_core_unmatched_live()


if __name__ == "__main__":
    main()
