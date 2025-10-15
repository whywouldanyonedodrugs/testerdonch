# manager.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from scout import run_scout
from backtester import run_backtest


def _results_path(name: str) -> Path:
    p = cfg.RESULTS_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_returns_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-trade returns columns if missing."""
    if "pnl" not in df.columns:
        df["pnl"] = (df["exit"] - df["entry"]) * df["qty"]
    if "pnl_R" not in df.columns:
        risk_per_unit = (df["entry"] - df["sl"]).replace(0, np.nan)
        df["pnl_R"] = (df["exit"] - df["entry"]) / risk_per_unit
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple aggregated CSV similar to the old reporter."""
    group_cols = list(cfg.AGGREGATE_BY) if getattr(cfg, "AGGREGATE_BY", None) else ["symbol"]

    def _row(g: pd.DataFrame) -> pd.Series:
        g = g.copy()
        g["win"] = (g["pnl"] > 0).astype(int)
        gp = g.loc[g["pnl"] > 0, "pnl"].sum()
        gl = -g.loc[g["pnl"] <= 0, "pnl"].sum()
        pf = (gp / gl) if gl > 0 else np.nan
        return pd.Series({
            "trades": len(g),
            "win_rate_pct": 100.0 * g["win"].mean() if len(g) else np.nan,
            "profit_factor": pf,
            "avg_pnl": g["pnl"].mean() if len(g) else np.nan,
            "median_pnl": g["pnl"].median() if len(g) else np.nan,
            "avg_pnl_R": g["pnl_R"].mean() if len(g) else np.nan,
            "median_pnl_R": g["pnl_R"].median() if len(g) else np.nan,
        })

    agg = (df.groupby(group_cols, dropna=False)
         .apply(_row, include_groups=False)  # future-proof pandas groupby.apply
         .reset_index())
    return agg


def _postprocess_and_aggregate():
    """Add returns columns, rewrite trades.csv, and write trades_aggregated.csv."""
    trades_csv = _results_path("trades.csv")
    if not trades_csv.exists():
        print(f"[warn] {trades_csv} not found; skipping aggregation.")
        return
    df = pd.read_csv(trades_csv, parse_dates=["entry_ts", "exit_ts"])
    if df.empty:
        print("[warn] trades.csv is empty; skipping aggregation.")
        return

    df = _ensure_returns_cols(df)
    df.to_csv(trades_csv, index=False)

    agg = _aggregate(df)
    out = _results_path("trades_aggregated.csv")
    agg.to_csv(out, index=False)
    print(f"Aggregates saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Long-only breakout → pullback → continuation tester")
    parser.add_argument("--start", type=str, default=cfg.START_DATE)
    parser.add_argument("--end", type=str, default=cfg.END_DATE)
    parser.add_argument("--no-rs", action="store_true", help="Disable weekly RS filter")
    parser.add_argument("--no-regime", action="store_true", help="Disable ETH 4h MACD regime gate")
    args = parser.parse_args()

    # apply overrides
    if args.start is not None:
        cfg.START_DATE = args.start
    if args.end is not None:
        cfg.END_DATE = args.end
    if args.no_rs:
        cfg.RS_ENABLED = False
    if args.no_regime:
        cfg.REGIME_FILTER_ENABLED = False

    print("Scouting signals…")
    nrows = run_scout()
    print(f"Signals: {nrows:,}")

    print("Running backtest…")
    # pandas + pyarrow can read a partitioned dataset by passing the directory
    run_backtest(cfg.SIGNALS_DIR)

    print("Post-processing & aggregation…")
    _postprocess_and_aggregate()

    print("\nTip: for robustness stats (CPCV/PBO + PSR/DSR), run:")
    print("  python reporting.py --run-all --returns-col pnl_R "
          "--variant-cols pullback_type entry_rule don_break_len regime_up")


if __name__ == "__main__":
    main()
