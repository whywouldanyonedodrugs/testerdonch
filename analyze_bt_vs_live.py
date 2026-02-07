#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Small helpers ----------


def _norm_col(name: str) -> str:
    """
    Normalize a column name to increase robustness across slightly different exports.

    - lowercases
    - strips spaces
    - replaces spaces and dashes with underscore
    - removes certain punctuation
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    for ch in [" ", "-", "/"]:
        s = s.replace(ch, "_")
    for ch in [".", ",", "(", ")", "[", "]"]:
        s = s.replace(ch, "")
    return s


def _infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Try to infer a column from a list of candidates using normalized names.

    Returns the *actual* column name from df.columns, or None if none match.
    """
    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _require_cols(df: pd.DataFrame, cols: List[str], label: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}; have {list(df.columns)}")


# ---------- Equity reconstruction ----------


@dataclass
class EquityCurve:
    ts: pd.Series
    equity: pd.Series


def build_equity_from_trades(
    trades: pd.DataFrame,
    initial_equity: float,
    label: str,
) -> EquityCurve:
    """
    Build a simple equity curve from a trades DataFrame.

    Requirements:
    - Must have some timestamp column (we'll try several candidates).
    - Must have a realized PnL / pnl column (we'll try several candidates).

    We:
    - sort by timestamp
    - compute cumulative pnl
    - add to initial_equity
    """

    # Attempt to infer timestamp and pnl columns from common exports
    ts_col_candidates = ["exit_ts", "exit_ts_live", "trade_time", "timestamp", "Trade time"]
    ts_col = _infer_col(trades, ts_col_candidates)
    pnl_col_candidates = ["pnl", "Realized P&L", "realized_pnl", "Realized PnL", "profit", "pnl_usdt"]
    pnl_col = _infer_col(trades, pnl_col_candidates)

    if ts_col is None or pnl_col is None:
        raise ValueError(
            f"could not infer timestamp col from {ts_col_candidates} "
            f"and pnl col from {pnl_col_candidates}; got {list(trades.columns)}"
        )

    df = trades.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col).reset_index(drop=True)

    # Drop rows with NaN pnl to avoid contaminating the curve
    df = df[pd.notna(df[pnl_col])].copy()

    cum_pnl = df[pnl_col].cumsum()
    equity = initial_equity + cum_pnl

    return EquityCurve(ts=df[ts_col], equity=equity)


# ---------- Metrics on backtester trades ----------


@dataclass
class BtMetrics:
    n_trades: int
    pnl_sum: float
    pnl_mean: float
    pnl_std: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_dd: float
    final_equity: float


def compute_bt_metrics(trades: pd.DataFrame, initial_equity: float) -> BtMetrics:
    """
    Compute core analytical metrics from backtester trades.csv.
    Assumes:
    - 'pnl' column in USDT
    - one row per closed trade
    """
    if trades.empty:
        raise ValueError("No trades in backtester results")

    if "pnl" not in trades.columns:
        # try to infer
        pnl_col = _infer_col(trades, ["pnl", "pnl_usdt", "Realized P&L", "profit"])
        if pnl_col is None:
            raise ValueError(f"Backtester trades has no 'pnl' and could not infer; have {list(trades.columns)}")
        trades = trades.rename(columns={pnl_col: "pnl"})

    pnl = trades["pnl"].astype(float)

    n = len(pnl)
    pnl_sum = float(pnl.sum())
    pnl_mean = float(pnl.mean())
    pnl_std = float(pnl.std(ddof=1)) if n > 1 else 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float(len(wins) / n) if n > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    # Build equity for drawdown calculation
    eq = initial_equity + pnl.cumsum()
    running_max = eq.cummax()
    dd = (eq - running_max)
    max_dd = float(dd.min())

    final_equity = float(eq.iloc[-1])

    return BtMetrics(
        n_trades=n,
        pnl_sum=pnl_sum,
        pnl_mean=pnl_mean,
        pnl_std=pnl_std,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_dd=max_dd,
        final_equity=final_equity,
    )


# ---------- Reading backtester outputs ----------


@dataclass
class BtResults:
    trades: pd.DataFrame
    equity_curve: EquityCurve
    metrics: BtMetrics


def load_backtester_results(results_dir: Path, initial_equity: float) -> BtResults:
    """
    Load `trades.csv` and optionally `equity_curve.csv` from backtester outputs.

    If `equity_curve.csv` not present, reconstruct from trades.
    """
    trades_path = results_dir / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"Backtester trades file not found: {trades_path}")

    trades = pd.read_csv(trades_path)

    # Try to reuse saved equity curve if it exists
    eq_path = results_dir / "equity_curve.csv"
    if eq_path.exists():
        eq_df = pd.read_csv(eq_path)
        ts_col = _infer_col(eq_df, ["timestamp", "ts", "time"])
        eq_col = _infer_col(eq_df, ["equity", "equity_usdt"])
        if ts_col is None or eq_col is None:
            raise ValueError(f"Could not infer columns in saved equity_curve.csv; have {list(eq_df.columns)}")
        eq_df[ts_col] = pd.to_datetime(eq_df[ts_col], utc=True)
        equity_curve = EquityCurve(ts=eq_df[ts_col], equity=eq_df[eq_col])
    else:
        equity_curve = build_equity_from_trades(trades, initial_equity, label="backtest")

    metrics = compute_bt_metrics(trades, initial_equity=initial_equity)
    return BtResults(trades=trades, equity_curve=equity_curve, metrics=metrics)


# ---------- Live vs BT comparison helpers ----------


def align_live_and_bt_on_exit(
    bt_trades: pd.DataFrame,
    live_trades: pd.DataFrame,
    ts_tolerance_min: int = 2,
) -> pd.DataFrame:
    """
    Align live and backtest trades by:
    - symbol
    - nearest exit timestamp (within tolerance, in minutes)

    Returns merged DataFrame with:
    - bt_* columns
    - live_* columns
    """
    bt = bt_trades.copy()
    lv = live_trades.copy()

    # Infer timestamp columns
    bt_ts_col = _infer_col(bt, ["exit_ts", "exit_time", "timestamp"])
    lv_ts_col = _infer_col(lv, ["exit_ts_live", "exit_ts", "Trade time", "trade_time", "timestamp"])

    if bt_ts_col is None:
        raise ValueError(f"Could not infer bt exit timestamp column; have {list(bt.columns)}")
    if lv_ts_col is None:
        raise ValueError(f"Could not infer live exit timestamp column; have {list(lv.columns)}")

    bt[bt_ts_col] = pd.to_datetime(bt[bt_ts_col], utc=True)
    lv[lv_ts_col] = pd.to_datetime(lv[lv_ts_col], utc=True)

    # Normalize symbol column names
    sym_bt = _infer_col(bt, ["symbol", "Market", "market"])
    sym_lv = _infer_col(lv, ["symbol", "Market", "market"])

    if sym_bt is None or sym_lv is None:
        raise ValueError(f"Could not infer symbol columns for alignment; bt={list(bt.columns)}, lv={list(lv.columns)}")

    bt = bt.rename(columns={sym_bt: "symbol_bt", bt_ts_col: "exit_ts_bt"})
    lv = lv.rename(columns={sym_lv: "symbol_lv", lv_ts_col: "exit_ts_live"})

    # We'll merge by symbol and nearest timestamp
    bt = bt.sort_values(["symbol_bt", "exit_ts_bt"]).reset_index(drop=True)
    lv = lv.sort_values(["symbol_lv", "exit_ts_live"]).reset_index(drop=True)

    merged_rows = []
    tol = pd.Timedelta(minutes=ts_tolerance_min)

    # For simplicity, do per-symbol merge using nearest join
    for sym in sorted(set(bt["symbol_bt"]).intersection(set(lv["symbol_lv"]))):
        bt_sym = bt[bt["symbol_bt"] == sym].copy()
        lv_sym = lv[lv["symbol_lv"] == sym].copy()
        if bt_sym.empty or lv_sym.empty:
            continue

        # For each live trade, find nearest bt trade in time (within tolerance)
        j = 0
        for _, row_lv in lv_sym.iterrows():
            ts_lv = row_lv["exit_ts_live"]
            # Move j so that bt_sym.exit_ts_bt[j] is close to ts_lv
            # We'll search around current j
            best_k = None
            best_diff = None
            for k in range(max(0, j - 5), min(len(bt_sym), j + 6)):
                ts_bt = bt_sym.iloc[k]["exit_ts_bt"]
                diff = abs(ts_bt - ts_lv)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_k = k
            if best_k is not None and best_diff <= tol:
                merged_rows.append(
                    {
                        **{f"bt_{col}": bt_sym.iloc[best_k][col] for col in bt_sym.columns},
                        **{f"live_{col}": row_lv[col] for col in lv_sym.columns},
                    }
                )
                j = best_k  # move pointer

    merged = pd.DataFrame(merged_rows)
    return merged


# ---------- Plotting: equity & PnL overlays ----------


def plot_equity_overlay(bt_eq: EquityCurve, live_eq: EquityCurve, out_path: Path) -> None:
    """
    Plot both backtester and live equity curves on the same axis (normalized to initial equity).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(bt_eq.ts, bt_eq.equity, label="Backtest equity")
    plt.plot(live_eq.ts, live_eq.equity, label="Live equity", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Equity (USDT)")
    plt.title("Backtest vs Live Equity Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scaled_cum_pnl(
    bt_trades: pd.DataFrame,
    live_trades: pd.DataFrame,
    out_path: Path,
    label_bt: str = "backtest",
    label_live: str = "live",
) -> None:
    """
    Plot cumulative PnL of backtest vs live, scaled so that both start at 0
    and end at their respective totals (i.e., compare *shape* of the PnL path).

    This is useful when position sizing / risk differs between backtest and live.
    """
    def _series(tr: pd.DataFrame, label: str, ts_candidates: List[str]) -> Tuple[pd.Series, pd.Series]:
        if tr.empty:
            raise ValueError(f"[{label}] no trades")

        # Use robust column inference (handles "Trade time" vs "trade_time", etc.)
        ts_col = _infer_col(tr, ts_candidates)
        pnl_col = _infer_col(tr, ["pnl", "Realized P&L", "realized_pnl", "Realized PnL", "profit", "pnl_usdt"])

        if ts_col is None or pnl_col is None:
            raise ValueError(f"[{label}] cannot infer ts/pnl cols; have {list(tr.columns)}")

        df = tr.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.sort_values(ts_col).reset_index(drop=True)

        pnl = df[pnl_col].astype(float)
        cum = pnl.cumsum()

        # Scale so that start=0, end=cum[-1] (i.e., keep total, but can rescale if needed)
        # In practice, this is just the raw cum PnL, but we could normalize by |total| if desired.
        return df[ts_col], cum

    x_bt, y_bt = _series(bt_trades, label_bt, ["exit_ts", "exit_time", "timestamp"])
    x_lv, y_lv = _series(live_trades, label_live, ["exit_ts_live", "exit_ts", "trade_time", "Trade time", "timestamp"])

    plt.figure(figsize=(12, 6))
    plt.plot(x_bt, y_bt, label=f"{label_bt} cum PnL")
    plt.plot(x_lv, y_lv, label=f"{label_live} cum PnL", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (USDT, raw)")
    plt.title("Cumulative PnL: Backtest vs Live (shape comparison)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- CLI / main ----------


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze backtester vs live PnL / equity.")
    ap.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory with backtester outputs (trades.csv, equity_curve.csv).",
    )
    ap.add_argument(
        "--initial-equity-bt",
        type=float,
        default=1000.0,
        help="Initial equity used in backtest (for metrics).",
    )
    ap.add_argument(
        "--live-trades",
        type=str,
        required=True,
        help="CSV file with live trades exported from Bybit (closed PnL).",
    )
    ap.add_argument(
        "--initial-equity-live",
        type=float,
        default=1000.0,
        help="Initial equity used for synthetic live equity curve.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Directory where plots / merged CSV will be saved.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load backtester
    bt = load_backtester_results(results_dir, initial_equity=args.initial_equity_bt)

    print("=== Backtester metrics ===")
    print(f"n_trades:      {bt.metrics.n_trades}")
    print(f"pnl_sum:       {bt.metrics.pnl_sum:.4f}")
    print(f"pnl_mean:      {bt.metrics.pnl_mean:.4f}")
    print(f"pnl_std:       {bt.metrics.pnl_std:.4f}")
    print(f"win_rate:      {bt.metrics.win_rate*100:.2f}%")
    print(f"avg_win:       {bt.metrics.avg_win:.4f}")
    print(f"avg_loss:      {bt.metrics.avg_loss:.4f}")
    print(f"max_drawdown:  {bt.metrics.max_dd:.4f}")
    print(f"final_equity:  {bt.metrics.final_equity:.4f}")

    # 2) Load live trades
    live_path = Path(args.live_trades)
    if not live_path.exists():
        raise FileNotFoundError(f"Live trades file not found: {live_path}")

    # Try common delimiters; Bybit often uses comma
    live = pd.read_csv(live_path)

    # 3) Build equity from live trades
    live_eq = build_equity_from_trades(live, initial_equity=args.initial_equity_live, label="live")

    # 4) Plot equity overlay
    eq_plot_path = out_dir / "equity_bt_vs_live.png"
    plot_equity_overlay(bt.equity_curve, live_eq, eq_plot_path)
    print(f"Equity overlay plot written to: {eq_plot_path}")

    # 5) Plot scaled cumulative PnL overlay
    pnl_plot_path = out_dir / "cum_pnl_bt_vs_live.png"
    plot_scaled_cum_pnl(
        bt_trades=bt.trades,
        live_trades=live,
        out_path=pnl_plot_path,
        label_bt="backtest",
        label_live="live",
    )
    print(f"Cumulative PnL overlay plot written to: {pnl_plot_path}")

    # 6) Align trades by exit timestamps and write merged CSV for manual inspection
    merged = align_live_and_bt_on_exit(bt_trades=bt.trades, live_trades=live, ts_tolerance_min=2)
    merged_path = out_dir / "bt_vs_live_merged_trades.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Merged bt/live trades written to: {merged_path}")
    print(f"Merged rows: {len(merged)}")


if __name__ == "__main__":
    main()
