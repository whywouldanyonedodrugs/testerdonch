#!/usr/bin/env python
"""
analyze_parity_deep.py

Deep diagnostic of live vs backtest parity:

- Loads live trades (results/livetrading.csv) and backtester trades (results/trades.csv).
- Matches trades by symbol + exit timestamp (within a tolerance).
- For every trade (live + bt), attaches:
    * RS percentile from rs_weekly.parquet
    * Liquidity proxy (~median 24h USD turnover) from rs_weekly.parquet
    * Volume spike features (rolling median multiple)
    * Micro-vol features (1h ATR / close)
- Computes per-trade gating flags:
    * rs_ok, liq_ok, vol_ok, micro_ok
- Summarises:
    * how many unmatched trades fail each gate
    * basic parity stats (counts, match rate, exit-time and PnL differences)
- Saves:
    * results/parity_matched_trades.csv
    * results/parity_unmatched_live.csv
    * results/parity_unmatched_backtest.csv
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as cfg
from shared_utils import load_parquet_data  # same helper used by scout/backtester


# ------------ Tunable diagnostics parameters -----------------

EXIT_MATCH_TOL_MIN: int = 15     # live vs bt exit time match window
VOL_MARGIN_DAYS: int = 2         # extra days before earliest trade for volume lookback
RS_TABLE_FILENAME: str = "rs_weekly.parquet"


# ------------ Helpers: RS & liquidity lookup -----------------


def load_rs_table() -> Optional[pd.DataFrame]:
    """
    Load rs_weekly.parquet written by scout.build_weekly_rs().
    If not present, RS / liquidity features are disabled.
    """
    path = cfg.RESULTS_DIR / RS_TABLE_FILENAME
    if not path.exists():
        print(f"[RS] WARNING: {path} not found. RS/liquidity features will be disabled.")
        return None

    table = pq.read_table(path)
    rs = table.to_pandas()

    # Ensure expected columns exist
    required = {"week_start", "symbol", "rs_pct", "usd_vol_med_24h"}
    missing = required - set(rs.columns)
    if missing:
        print(f"[RS] WARNING: missing columns in RS table: {missing}. RS/liquidity features partial.")

    # Normalise symbol
    rs["symbol"] = rs["symbol"].astype(str).str.upper()

    # Normalise week_start to UTC, regardless of whether it is naive or tz-aware
    ws = pd.to_datetime(rs["week_start"])
    # If naive → localize; if tz-aware → convert
    tz = getattr(ws.dt, "tz", None)
    if tz is None:
        ws = ws.dt.tz_localize("UTC")
    else:
        ws = ws.dt.tz_convert("UTC")
    rs["week_start"] = ws

    return rs



def rs_lookup(rs_table: pd.DataFrame, sym: str, ts: pd.Timestamp) -> Optional[float]:
    """
    Lookup RS percentile for symbol at time ts.
    Semantics: last week_start <= floor(ts to day).
    """
    if rs_table is None:
        return None
    rs_sym = rs_table[rs_table["symbol"] == sym]
    if rs_sym.empty:
        return None
    day = ts.floor("D")
    candidates = rs_sym[rs_sym["week_start"] <= day]
    if candidates.empty:
        return None
    return float(candidates.iloc[-1]["rs_pct"])


def liq_lookup(rs_table: pd.DataFrame, sym: str, ts: pd.Timestamp) -> Optional[float]:
    """
    Lookup liquidity proxy (~median 24h USD turnover) for symbol at time ts.
    Uses same weekly alignment as rs_lookup().
    """
    if rs_table is None:
        return None
    rs_sym = rs_table[rs_table["symbol"] == sym]
    if rs_sym.empty:
        return None
    day = ts.floor("D")
    candidates = rs_sym[rs_sym["week_start"] <= day]
    if candidates.empty:
        return None
    if "usd_vol_med_24h" not in candidates.columns:
        return None
    val = candidates.iloc[-1]["usd_vol_med_24h"]
    if pd.isna(val):
        return None
    return float(val)


# ------------ OHLCV feature helpers --------------------------


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Simple ATR implementation for OHLC data (index must be time).
    df: columns ['high', 'low', 'close']
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(length, min_periods=length).mean()
    return atr


def compute_symbol_features(
    sym: str,
    exits: pd.Series,
) -> Optional[pd.DataFrame]:
    """
    For a given symbol and a series of exit_ts (UTC), load 5m OHLCV
    from Parquet and compute:

      - vol: volume at bar
      - vol_mult: volume / rolling_median(volume, VOL_LOOKBACK_DAYS)
      - vol_ok: volume spike flag (vs VOL_MULTIPLE or VOL_QUANTILE_Q)
      - micro_vol_ratio: 1h ATR / close
      - micro_ok: micro-vol filter vs MICRO_VOL_MIN

    Returns a 5m frame with features (indexed by timestamp), or None if no data.
    """
    exits = pd.to_datetime(exits).dt.tz_convert("UTC")
    if exits.empty:
        return None

    # Determine time window [start, end] for OHLC load
    earliest_exit = exits.min()
    latest_exit = exits.max()

    start_ts = earliest_exit - pd.Timedelta(days=int(cfg.VOL_LOOKBACK_DAYS) + VOL_MARGIN_DAYS)
    end_ts = latest_exit + pd.Timedelta(days=1)

    start_date = start_ts.date().isoformat()
    end_date = end_ts.date().isoformat()

    df = load_parquet_data(
        sym,
        start_date=start_date,
        end_date=end_date,
        drop_last_partial=False,
        columns=["open", "high", "low", "close", "volume"],
    )
    if df.empty:
        print(f"[features] WARNING: no OHLC data for {sym} in [{start_date}, {end_date}]")
        return None

    # Ensure a clean DatetimeIndex in UTC
    df.index = pd.to_datetime(df.index)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    # --- Volume spike features ---
    vol = df["volume"].astype(float)

    # Estimate bars-per-day from spacing
    if len(df.index) >= 2:
        bar_minutes = max(
            1,
            int(round((df.index[1] - df.index[0]).total_seconds() / 60.0)),
        )
    else:
        bar_minutes = 5  # fall back

    bars_per_day = max(1, int(round(24 * 60 / bar_minutes)))
    lookback_bars = int(cfg.VOL_LOOKBACK_DAYS) * bars_per_day

    if lookback_bars < 1:
        lookback_bars = bars_per_day

    vol_med = vol.rolling(
        window=lookback_bars,
        min_periods=max(5, bars_per_day, lookback_bars // 10),
    ).median()
    vol_mult = vol / vol_med

    mode = str(getattr(cfg, "VOL_SPIKE_MODE", "multiple")).lower()
    if mode == "multiple":
        thr = float(getattr(cfg, "VOL_MULTIPLE", 2.0))
        vol_ok = vol_mult >= thr
    else:
        q = float(getattr(cfg, "VOL_QUANTILE_Q", 0.95))
        vol_q = vol.rolling(
            window=lookback_bars,
            min_periods=max(5, bars_per_day),
        ).quantile(q)
        vol_ok = vol >= vol_q

    # --- Micro-vol features: 1h ATR / close ---
    df_1h = df.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    ).dropna()


    atr_len = int(getattr(cfg, "ATR_LEN", 14))
    atr_1h = compute_atr(df_1h, atr_len)
    atr_1h_5m = atr_1h.reindex(df.index, method="ffill")

    close = df["close"].astype(float)
    micro_vol_ratio = atr_1h_5m / close

    df_feat = pd.DataFrame(
        {
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": close,
            "volume": vol,
            "vol_mult": vol_mult,
            "vol_ok": vol_ok.astype(bool),
            "micro_vol_ratio": micro_vol_ratio,
        },
        index=df.index,
    )

    micro_min = float(getattr(cfg, "MICRO_VOL_MIN", 0.0))
    df_feat["micro_ok"] = df_feat["micro_vol_ratio"] >= micro_min

    return df_feat


def attach_features_to_trades(
    trades: pd.DataFrame,
    rs_table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    For each trade (live or bt), attach:

      - bar_ts: timestamp of nearest 5m bar <= exit_ts
      - vol, vol_mult, vol_ok
      - micro_vol_ratio, micro_ok
      - rs_pct, liq_usd, rs_ok, liq_ok

    Assumes trades has columns:
      - symbol (upper case)
      - exit_ts (tz-aware UTC)
    """
    trades = trades.copy()

    # Numeric feature columns – initialise as NaN
    for col in ["vol", "vol_mult", "micro_vol_ratio", "rs_pct", "liq_usd"]:
        trades[col] = np.nan

    # Boolean feature columns – initialise as False (simple bool dtype)
    for col in ["vol_ok", "micro_ok", "rs_ok", "liq_ok"]:
        trades[col] = False

    # NOTE: we do NOT pre-create 'bar_ts' – we let assignment below create it
    # with the correct datetime64[ns, UTC] dtype.

    # Group by symbol to load OHLC once per symbol
    groups = trades.groupby("symbol")
    for sym, idxs in groups.groups.items():
        t_sym = trades.loc[idxs]
        exits = t_sym["exit_ts"]
        feat = compute_symbol_features(sym, exits)
        if feat is None or feat.empty:
            continue

        # Align each trade's exit_ts to nearest bar <= exit_ts.
        # We do this in integer nanoseconds to avoid tz-naive vs tz-aware issues.
        bar_index = feat.index  # DatetimeIndex, tz-aware UTC
        bar_ns = bar_index.asi8  # int64 nanoseconds since epoch

        exits_ts = pd.to_datetime(exits)
        if exits_ts.dt.tz is None:
            exits_ts = exits_ts.dt.tz_localize("UTC")
        else:
            exits_ts = exits_ts.dt.tz_convert("UTC")

        # int64 ns array for exits
        exit_ns = exits_ts.astype("int64").to_numpy()

        # For each exit, find the last bar <= exit_ts
        pos = np.searchsorted(bar_ns, exit_ns, side="right") - 1
        pos = np.clip(pos, 0, len(bar_index) - 1)

        aligned_ts = bar_index[pos]
        # New column 'bar_ts' will be created with datetime64[ns, UTC] dtype
        trades.loc[idxs, "bar_ts"] = aligned_ts

        trades.loc[idxs, "vol"] = feat["volume"].iloc[pos].values
        trades.loc[idxs, "vol_mult"] = feat["vol_mult"].iloc[pos].values
        trades.loc[idxs, "vol_ok"] = feat["vol_ok"].iloc[pos].values
        trades.loc[idxs, "micro_vol_ratio"] = feat["micro_vol_ratio"].iloc[pos].values
        trades.loc[idxs, "micro_ok"] = feat["micro_ok"].iloc[pos].values

        # RS & liquidity lookups
        if rs_table is not None:
            rs_vals: List[Optional[float]] = []
            liq_vals: List[Optional[float]] = []
            for ts in exits_ts:
                rs_vals.append(rs_lookup(rs_table, sym, ts))
                liq_vals.append(liq_lookup(rs_table, sym, ts))

            # Clean to pure floats + NaN (no None) to avoid dtype warnings
            rs_vals_clean = [
                np.nan if (v is None or pd.isna(v)) else float(v) for v in rs_vals
            ]
            liq_vals_clean = [
                np.nan if (v is None or pd.isna(v)) else float(v) for v in liq_vals
            ]
            trades.loc[idxs, "rs_pct"] = rs_vals_clean
            trades.loc[idxs, "liq_usd"] = liq_vals_clean

            min_rs = int(getattr(cfg, "RS_MIN_PERCENTILE", 0))
            liq_min = float(getattr(cfg, "RS_LIQ_MIN_USD_24H", 0.0))

            rs_ok = [
                (v is not None) and (not pd.isna(v)) and (float(v) >= min_rs)
                for v in rs_vals
            ]
            liq_ok = [
                (v is not None) and (not pd.isna(v)) and (float(v) >= liq_min)
                for v in liq_vals
            ]
            trades.loc[idxs, "rs_ok"] = rs_ok
            trades.loc[idxs, "liq_ok"] = liq_ok

    return trades


# ------------ Loaders for live & bt trades -------------------


def load_live_trades(path: Path | None = None) -> pd.DataFrame:
    """
    Load results/livetrading.csv and normalise columns.

    Expected columns (Bybit export):
      - Market
      - Order Quantity
      - Entry Price
      - Exit Price
      - Realized P&L
      - Trade time

    Trade time is treated as exit time and parsed as UTC.
    The file may use formats like:
      - '02:10 2025-11-14'
      - '14/11/2025 02:10'
    so we first try generic parsing, then fall back to a dayfirst pattern
    for any rows that fail.
    """
    if path is None:
        path = cfg.RESULTS_DIR / "livetrading.csv"

    df = pd.read_csv(path)

    df["symbol"] = df["Market"].astype(str).str.upper()

    # First attempt: let pandas infer the format (covers '02:10 2025-11-14')
    ts = pd.to_datetime(df["Trade time"], utc=True, errors="coerce")

    # If there are any NaT values, try a day-first fallback like '14/11/2025 02:10'
    if ts.isna().any():
        mask = ts.isna()
        ts_fallback = pd.to_datetime(
            df.loc[mask, "Trade time"],
            format="%d/%m/%Y %H:%M",
            utc=True,
            errors="coerce",
        )
        ts.loc[mask] = ts_fallback

    df["exit_ts"] = ts

    df["entry_price"] = pd.to_numeric(df["Entry Price"], errors="coerce")
    df["exit_price"] = pd.to_numeric(df["Exit Price"], errors="coerce")
    df["pnl_live"] = pd.to_numeric(df["Realized P&L"], errors="coerce")

    df = df.sort_values(["symbol", "exit_ts"]).reset_index(drop=True)
    return df



def load_bt_trades(path: Path | None = None) -> pd.DataFrame:
    """
    Load backtester trades from results/trades.csv.

    Expected columns:
      - trade_id
      - symbol
      - entry_ts
      - exit_ts
      - entry
      - exit
      - pnl
      - plus strategy/regime fields
    """
    if path is None:
        path = cfg.RESULTS_DIR / "trades.csv"

    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)

    if "pnl" in df.columns:
        df["pnl_bt"] = pd.to_numeric(df["pnl"], errors="coerce")
    else:
        df["pnl_bt"] = np.nan

    df = df.sort_values(["symbol", "exit_ts"]).reset_index(drop=True)
    return df


# ------------ Matching logic (live ↔ backtest) ---------------


@dataclass
class MatchResult:
    pairs: List[Tuple[int, int]]           # list of (live_idx, bt_idx)
    unmatched_live: List[int]
    unmatched_bt: List[int]


def match_live_to_bt(
    live: pd.DataFrame,
    bt: pd.DataFrame,
    tol_minutes: int = EXIT_MATCH_TOL_MIN,
) -> MatchResult:
    """
    Greedy 1-1 matching of live and bt trades by symbol + exit_ts within a tolerance.

    For each live trade, we look for an unmatched bt trade with same symbol and
    abs(exit_ts_bt - exit_ts_live) <= tol, and choose the closest in time.

    Returns:
      - list of (live_idx, bt_idx) pairs
      - indices of unmatched live trades
      - indices of unmatched bt trades
    """
    tol = pd.Timedelta(minutes=tol_minutes)

    used_bt: set[int] = set()
    pairs: List[Tuple[int, int]] = []

    # Work on original index to keep mapping stable
    live_idx = live.index
    bt_idx = bt.index

    # Group by symbol to reduce search space
    live_by_sym = live.groupby("symbol")
    bt_by_sym = bt.groupby("symbol")

    for sym, live_group in live_by_sym:
        if sym not in bt_by_sym.groups:
            continue
        bt_group = bt.loc[bt_by_sym.groups[sym]]

        # For speed, pre-sort by exit_ts
        live_group = live_group.sort_values("exit_ts")
        bt_group = bt_group.sort_values("exit_ts")

        for li, row in live_group.iterrows():
            lt = row["exit_ts"]
            # Candidate bt trades for this symbol within tolerance and not yet used
            mask_time = (bt_group["exit_ts"] >= lt - tol) & (bt_group["exit_ts"] <= lt + tol)
            candidates = bt_group[mask_time & (~bt_group.index.isin(used_bt))]
            if candidates.empty:
                continue
            # Choose candidate with minimal absolute time difference
            diffs = (candidates["exit_ts"] - lt).abs()
            best_bt_idx = diffs.idxmin()
            pairs.append((li, best_bt_idx))
            used_bt.add(best_bt_idx)

    matched_live = {li for (li, _) in pairs}
    matched_bt = {bi for (_, bi) in pairs}

    unmatched_live = [i for i in live_idx if i not in matched_live]
    unmatched_bt = [i for i in bt_idx if i not in matched_bt]

    return MatchResult(pairs=pairs, unmatched_live=unmatched_live, unmatched_bt=unmatched_bt)


# ------------ Summaries & output -----------------------------


def summarise_gates(df: pd.DataFrame, name: str) -> None:
    n = len(df)
    print(f"\n=== Gating summary for {name} (n={n}) ===")
    if n == 0:
        return

    for col in ["rs_ok", "liq_ok", "vol_ok", "micro_ok"]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        print(f"\n{col}:")
        print(vc)

    # Combined gate: all four must pass
    for col in ["rs_ok", "liq_ok", "vol_ok", "micro_ok"]:
        if col not in df.columns:
            return
    gate_all_ok = df["rs_ok"].fillna(False) & df["liq_ok"].fillna(False) \
        & df["vol_ok"].fillna(False) & df["micro_ok"].fillna(False)

    print("\nall_gates_ok value counts:")
    print(gate_all_ok.value_counts(dropna=False))


def main():
    # --- Load inputs ---
    live = load_live_trades()
    bt = load_bt_trades()
    rs_table = load_rs_table()

    print(f"[info] Live trades: {len(live)}")
    print(f"[info] Backtest trades: {len(bt)}")

    # --- Match live to bt ---
    match = match_live_to_bt(live, bt, tol_minutes=EXIT_MATCH_TOL_MIN)
    print(f"[info] Matched trades: {len(match.pairs)} "
          f"({len(match.pairs) / max(1, len(live)) * 100:.3f} % of live)")

    # Attach match indices to live and bt
    live["bt_match_idx"] = pd.NA
    bt["live_match_idx"] = pd.NA

    for li, bi in match.pairs:
        live.loc[li, "bt_match_idx"] = bi
        bt.loc[bi, "live_match_idx"] = li

    # --- Attach features to ALL trades (live and bt) ---
    live_feat = attach_features_to_trades(
        live.rename(columns={"exit_ts": "exit_ts"}), rs_table
    )
    bt_feat = attach_features_to_trades(
        bt.rename(columns={"exit_ts": "exit_ts"}), rs_table
    )

    # --- Build matched/unmatched sets ---
    matched_rows: List[pd.DataFrame] = []
    for li, bi in match.pairs:
        lrow = live_feat.loc[li].add_prefix("live_")
        brow = bt_feat.loc[bi].add_prefix("bt_")
        row = pd.concat([lrow, brow])
        # Exit time diff and PnL diff as diagnostics
        exit_dt_diff_min = (
            brow["bt_exit_ts"] - lrow["live_exit_ts"]
        ).total_seconds() / 60.0
        row["exit_dt_diff_min"] = exit_dt_diff_min
        if pd.notna(brow.get("bt_pnl_bt", np.nan)) and pd.notna(lrow.get("live_pnl_live", np.nan)):
            row["pnl_diff"] = float(brow["bt_pnl_bt"]) - float(lrow["live_pnl_live"])
        else:
            row["pnl_diff"] = np.nan
        matched_rows.append(row)

    matched_df = pd.DataFrame(matched_rows)

    unmatched_live = live_feat.loc[match.unmatched_live].copy()
    unmatched_bt = bt_feat.loc[match.unmatched_bt].copy()

    # --- High-level parity summary (similar to previous script) ---
    print("\n=== Live vs Backtest Parity Summary (deep) ===")
    print(f"Live trades:      {len(live)}")
    print(f"Backtest trades:  {len(bt)}")
    print(f"Matched trades:   {len(match.pairs)}")
    print(f"Match rate:       {len(match.pairs) / max(1, len(live)) * 100:.3f} %")

    if not matched_df.empty:
        print("\nExit time difference (minutes):")
        print(matched_df["exit_dt_diff_min"].describe())

        if "pnl_diff" in matched_df.columns and matched_df["pnl_diff"].notna().any():
            print("\nPnL difference (bt - live, USDT):")
            print(matched_df["pnl_diff"].describe())

    # --- Gate summaries for unmatched sets ---
    summarise_gates(unmatched_live, "UNMATCHED LIVE trades")
    summarise_gates(unmatched_bt, "UNMATCHED BACKTEST trades")

    # --- Persist to CSV for manual inspection ---
    out_dir = cfg.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    matched_path = out_dir / "parity_matched_trades.csv"
    unmatched_live_path = out_dir / "parity_unmatched_live.csv"
    unmatched_bt_path = out_dir / "parity_unmatched_backtest.csv"

    matched_df.to_csv(matched_path, index=False)
    unmatched_live.to_csv(unmatched_live_path, index=True)
    unmatched_bt.to_csv(unmatched_bt_path, index=True)

    print(f"\n[info] Saved matched trades with features to: {matched_path}")
    print(f"[info] Saved unmatched LIVE trades with features to: {unmatched_live_path}")
    print(f"[info] Saved unmatched BACKTEST trades with features to: {unmatched_bt_path}")


if __name__ == "__main__":
    main()
