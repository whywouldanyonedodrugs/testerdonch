#!/usr/bin/env python
import pandas as pd
from pathlib import Path

# --- Paths ---
LIVE_CSV_PATH = Path("results/livetrading.csv")   # <- adjust if needed
BT_TRADES_PATH = Path("results/trades.csv")        # backtester output

# --- Load live trades ---
live = pd.read_csv(LIVE_CSV_PATH)

# Expect columns: Market, Order Quantity, Entry Price, Exit Price,
# cumClosedPzOpenFeeInfo, cumClosedPzTradeFeeInfo, Trade Type,
# Realized P&L, Trade time

live["symbol"] = live["Market"].astype(str).str.upper()

# Parse "HH:MM YYYY-MM-DD" as UTC tz-aware timestamps; coerce bad rows to NaT
live["exit_ts_live"] = pd.to_datetime(
    live["Trade time"], format="%H:%M %Y-%m-%d", utc=True, errors="coerce"
)

# Keep only real trades with valid timestamps
live = live[live["Trade Type"] == "Trade"].copy()
live = live.dropna(subset=["exit_ts_live"])

# Ensure tz-aware UTC (defensive)
if live["exit_ts_live"].dt.tz is None:
    live["exit_ts_live"] = live["exit_ts_live"].dt.tz_localize("UTC")
else:
    live["exit_ts_live"] = live["exit_ts_live"].dt.tz_convert("UTC")

# --- Load backtester trades ---
bt = pd.read_csv(BT_TRADES_PATH, parse_dates=["entry_ts", "exit_ts"])

bt["symbol"] = bt["symbol"].astype(str).str.upper()
bt["exit_ts_bt"] = bt["exit_ts"]

# Drop trades with missing exit_ts
bt = bt.dropna(subset=["exit_ts_bt"])

# Normalise bt timestamps to UTC tz-aware
if bt["exit_ts_bt"].dt.tz is None:
    bt["exit_ts_bt"] = bt["exit_ts_bt"].dt.tz_localize("UTC")
else:
    bt["exit_ts_bt"] = bt["exit_ts_bt"].dt.tz_convert("UTC")

# --- Filter bt window to live window with a 1-day cushion ---
start = live["exit_ts_live"].min()
end   = live["exit_ts_live"].max()

bt = bt[(bt["exit_ts_bt"] >= start - pd.Timedelta("1D")) &
        (bt["exit_ts_bt"] <= end   + pd.Timedelta("1D"))].copy()

# --- Sort keys as merge_asof actually requires: by time ("on") only ---
# "by" is handled separately; the 'on' column itself must be globally sorted
live_sorted = live.sort_values("exit_ts_live").reset_index(drop=True)
bt_sorted   = bt.sort_values("exit_ts_bt").reset_index(drop=True)

# --- As-of merge by symbol + exit time ---
merged = pd.merge_asof(
    left=live_sorted,
    right=bt_sorted,
    by="symbol",
    left_on="exit_ts_live",
    right_on="exit_ts_bt",
    direction="nearest",
    tolerance=pd.Timedelta("15min"),
    suffixes=("_live", "_bt"),
)


merged["matched"] = ~merged["exit_ts_bt"].isna()

matched = merged[merged["matched"]].copy()
unmatched_live = merged[~merged["matched"]].copy()

# Compute time difference in minutes
if len(matched):
    matched["exit_dt_diff_min"] = (
        matched["exit_ts_bt"] - matched["exit_ts_live"]
    ).dt.total_seconds() / 60.0

# Compute PnL differences if bt has pnl column
if len(matched) and "pnl" in matched.columns:
    matched["pnl_diff"] = matched["pnl"] - matched["Realized P&L"]
else:
    matched["pnl_diff"] = float("nan")

# --- Summary stats ---
total_live = len(live_sorted)
total_bt   = len(bt_sorted)
total_matched = len(matched)
match_rate = total_matched / total_live if total_live else float("nan")

print("=== Live vs Backtest Parity Summary ===")
print(f"Live trades:      {total_live}")
print(f"Backtest trades:  {total_bt}")
print(f"Matched trades:   {total_matched}")
print(f"Match rate:       {match_rate:.3%}")

if total_matched:
    print("\nExit time difference (minutes):")
    print(matched["exit_dt_diff_min"].describe())

    if "pnl_diff" in matched.columns and matched["pnl_diff"].notna().any():
        print("\nPnL difference (bt - live, USDT):")
        print(matched["pnl_diff"].describe())

print("\n=== Unmatched live trades (head) ===")
print(unmatched_live[["symbol", "exit_ts_live", "Realized P&L"]].head(20))

OUT_PATH = Path("results/live_vs_bt_merged.csv")
merged.to_csv(OUT_PATH, index=False)
print(f"\nFull merged file saved to: {OUT_PATH}")
