#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg


# === Basic loaders ============================================================

def load_trades(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"trades.csv not found in {results_dir}")
    tr = pd.read_csv(path, parse_dates=["entry_ts", "exit_ts"])
    return tr


def load_equity(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "equity.csv"
    if not path.exists():
        raise FileNotFoundError(f"equity.csv not found in {results_dir}")
    eq = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    return eq


# === Core backtest stats ======================================================

def compute_core_stats(tr: pd.DataFrame, eq: pd.DataFrame) -> dict:
    stats: dict[str, float | int | str | None] = {}

    n_trades = len(tr)
    stats["n_trades"] = int(n_trades)

    if n_trades == 0:
        return stats

    # Profit factor (cash PnL)
    gross_profit = tr.loc[tr["pnl"] > 0, "pnl"].sum()
    gross_loss = -tr.loc[tr["pnl"] < 0, "pnl"].sum()
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
    stats["profit_factor"] = pf

    # Win-rate & average R
    stats["win_rate"] = float((tr["pnl"] > 0).mean())
    stats["avg_R"] = float(tr["pnl_R"].mean())

    # Equity-based stats
    equity = eq["equity"].astype(float)
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan
    stats["max_drawdown"] = max_dd  # negative number

    # CAGR
    if len(eq) >= 2:
        dt_years = (eq["timestamp"].iloc[-1] - eq["timestamp"].iloc[0]).total_seconds() / (
            365.25 * 24 * 3600
        )
        if dt_years > 0:
            stats["cagr"] = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / dt_years) - 1.0)
        else:
            stats["cagr"] = np.nan
    else:
        stats["cagr"] = np.nan

    # Daily returns & Sharpe
    daily_eq = eq.set_index("timestamp")["equity"].resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    stats["daily_ret_mean"] = float(daily_ret.mean()) if len(daily_ret) else np.nan
    stats["daily_ret_std"] = float(daily_ret.std(ddof=1)) if len(daily_ret) else np.nan

    if len(daily_ret) > 1 and daily_ret.std(ddof=1) > 0:
        stats["sharpe"] = float(np.sqrt(252.0) * daily_ret.mean() / daily_ret.std(ddof=1))
    else:
        stats["sharpe"] = np.nan

    # Best / worst day by PnL
    if len(daily_ret):
        daily_pnl = daily_eq.diff().dropna()
        stats["best_day_pnl"] = float(daily_pnl.max())
        stats["worst_day_pnl"] = float(daily_pnl.min())
        stats["best_day_date"] = str(daily_pnl.idxmax().date())
        stats["worst_day_date"] = str(daily_pnl.idxmin().date())
    else:
        stats["best_day_pnl"] = stats["worst_day_pnl"] = np.nan
        stats["best_day_date"] = stats["worst_day_date"] = None

    return stats


def print_stats(stats: dict) -> None:
    print("\n=== Backtest Summary ===")
    if stats.get("n_trades", 0) == 0:
        print("No trades.")
        return

    print(f"Trades           : {stats['n_trades']}")
    print(f"Profit factor    : {stats['profit_factor']:.2f}")
    print(f"Win rate         : {stats['win_rate']:.1%}")
    print(f"Avg R / trade    : {stats['avg_R']:.3f}")
    print(f"Max drawdown     : {stats['max_drawdown']*100:.1f}%")
    print(f"CAGR             : {stats['cagr']*100:.1f}%")
    print(f"Sharpe (daily)   : {stats['sharpe']:.2f}")

    if stats.get("best_day_date"):
        print(
            f"Best day         : {stats['best_day_date']}  PnL={stats['best_day_pnl']:.2f}"
        )
        print(
            f"Worst day        : {stats['worst_day_date']} PnL={stats['worst_day_pnl']:.2f}"
        )
    print("==============")


def print_profitable_periods(eq: pd.DataFrame) -> None:
    """Monthly returns sign."""
    daily_eq = eq.set_index("timestamp")["equity"].resample("1D").last().dropna()
    if len(daily_eq) < 2:
        print("Not enough equity data to compute monthly returns.")
        return

    # Use 'ME' (month-end) to avoid pandas FutureWarning about 'M'
    monthly = daily_eq.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        print("No monthly periods found.")
        return

    print("\nMonthly returns (equity-based):")
    for ts, r in monthly.items():
        mark = "+" if r > 0 else "-"
        print(f"  {ts.strftime('%Y-%m')} : {r*100:6.2f}%  [{mark}]")
    print("--------------")


def plot_equity(eq: pd.DataFrame, results_dir: Optional[Path] = None, title: str = "Equity (indexed: start=1)") -> None:
    if eq.empty:
        print("No equity data to plot.")
        return
    eq_sorted = eq.sort_values("timestamp")
    equity = eq_sorted["equity"].astype(float)
    idx_equity = equity / equity.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(eq_sorted["timestamp"], idx_equity, label="backtest", linewidth=1.5)
    plt.xlabel("time")
    plt.ylabel("equity index (start=1)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if results_dir is not None:
        out_path = results_dir / "equity_index.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved equity curve to: {out_path}")

    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close()


# === ETH 4h MACD histogram export ============================================

def compute_eth_macd_4h_df() -> pd.DataFrame:
    """
    Recompute ETH 4h MACD (line, signal, hist) from the same parquet the
    scout/backtester use. Returns a DataFrame indexed by 4h timestamps.
    """
    eth_path = cfg.PARQUET_DIR / f"{cfg.REGIME_ASSET}.parquet"
    if not eth_path.exists():
        raise FileNotFoundError(f"ETH parquet not found at {eth_path}")

    eth = pd.read_parquet(eth_path)

    # Accept either a 'timestamp' column or a DatetimeIndex
    if "timestamp" in eth.columns:
        eth = eth.copy()
        eth["timestamp"] = pd.to_datetime(eth["timestamp"], utc=True, errors="coerce")
        eth = eth.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    else:
        if not isinstance(eth.index, pd.DatetimeIndex):
            raise ValueError("ETH parquet must have a datetime index or a 'timestamp' column.")
        eth = eth.copy()
        eth.index = pd.to_datetime(eth.index, utc=True, errors="coerce")
        eth = eth[~eth.index.isna()]
        eth.index.name = "timestamp"

    close_4h = (
        eth["close"].astype(float)
        .resample("4h", label="right", closed="right")
        .last()
        .dropna()
    )
    if close_4h.empty:
        raise ValueError("No 4h close data after resampling ETH.")

    fast = int(getattr(cfg, "REGIME_MACD_FAST", 12))
    slow = int(getattr(cfg, "REGIME_MACD_SLOW", 26))
    signal = int(getattr(cfg, "REGIME_MACD_SIGNAL", 9))

    ema_fast = close_4h.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_4h.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal

    df = pd.DataFrame(
        {
            "close_4h": close_4h,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        }
    )

    # Optionally restrict to backtest window from config
    start_str = getattr(cfg, "START_DATE", None)
    end_str = getattr(cfg, "END_DATE", None)
    if start_str is not None:
        start_ts = pd.to_datetime(start_str, utc=True, errors="coerce")
        df = df[df.index >= start_ts]
    if end_str is not None:
        # include up to end of that day
        end_ts = pd.to_datetime(end_str, utc=True, errors="coerce") + pd.Timedelta(days=1)
        df = df[df.index < end_ts]

    return df


def export_eth_macd_4h_md(results_dir: Path) -> None:
    """
    Export ETH 4h MACD series as a markdown table for manual inspection
    vs. real data.
    """
    try:
        macd_df = compute_eth_macd_4h_df()
    except Exception as e:
        print(f"\n[warn] Could not compute ETH 4h MACD for export: {e}")
        return

    out_path = results_dir / "eth_macd_4h.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make index a column for nicer markdown
    df_out = macd_df.reset_index().rename(columns={"timestamp": "ts_4h"})
    # Limit columns to what you care about
    df_out = df_out[["ts_4h", "close_4h", "macd_line", "macd_signal", "macd_hist"]]

    md = []
    md.append("# ETHUSDT 4h MACD histogram\n")
    md.append("")
    md.append("| ts_4h | close_4h | macd_line | macd_signal | macd_hist |")
    md.append("|------|----------|-----------|-------------|-----------|")
    for _, row in df_out.iterrows():
        ts = row["ts_4h"]
        close = float(row["close_4h"])
        line = float(row["macd_line"])
        sig = float(row["macd_signal"])
        hist = float(row["macd_hist"])
        md.append(
            f"| {ts} | {close:.4f} | {line:.6f} | {sig:.6f} | {hist:.6f} |"
        )

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"ETH 4h MACD markdown exported to: {out_path}")


# === Main ====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Analyze backtest results in results/trades.csv and results/equity.csv"
    )
    ap.add_argument(
        "--results-dir",
        default="results",
        help="Directory with trades.csv and equity.csv (default: results)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    tr = load_trades(results_dir)
    eq = load_equity(results_dir)

    stats = compute_core_stats(tr, eq)
    print_stats(stats)
    print_profitable_periods(eq)

    # Export ETH 4h MACD series as markdown for manual comparison
    export_eth_macd_4h_md(results_dir)

    # Plot equity curve last (so prints appear first)
    plot_equity(eq, results_dir)


if __name__ == "__main__":
    main()
