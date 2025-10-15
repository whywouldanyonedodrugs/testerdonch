import argparse
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import sys

@dataclass
class VariantSpec:
    name: str
    trades_path: Path
    equity_path: Path

@dataclass
class VariantMetrics:
    name: str
    n_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    net_pnl: float
    avg_win: float
    avg_loss: float
    avg_hold_hours: float
    cagr: float
    max_dd: float
    mar: float
    sharpe_ann: float
    start: pd.Timestamp
    end: pd.Timestamp

def _read_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize timestamps (UTC)
    for col in ("entry_ts", "exit_ts", "timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

def _read_equity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expect columns: timestamp, equity (fallback to first/second col)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    eq_col = "equity" if "equity" in df.columns else df.columns[1]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    return df.rename(columns={ts_col: "timestamp", eq_col: "equity"})[["timestamp","equity"]]

def _compute_trade_metrics(tr: pd.DataFrame) -> Dict[str, float]:
    # choose pnl col or compute from (exit-entry)*qty
    pnl_col = None
    for c in ["pnl", "pnl_cash", "pnl_usd"]:
        if c in tr.columns:
            pnl_col = c
            break
    if pnl_col is None:
        if all(c in tr.columns for c in ("entry","exit","qty")):
            tr["_pnl_calc"] = (tr["exit"] - tr["entry"]) * tr["qty"]
            pnl_col = "_pnl_calc"
        else:
            raise ValueError("Trades file must have pnl/pnl_cash or entry,exit,qty.")

    holds = None
    if all(c in tr.columns for c in ("entry_ts","exit_ts")):
        holds = (pd.to_datetime(tr["exit_ts"]) - pd.to_datetime(tr["entry_ts"])).dt.total_seconds()/3600.0

    winners = tr[pnl_col] > 0
    gross_profit = float(tr.loc[winners, pnl_col].sum())
    gross_loss   = float(-tr.loc[~winners, pnl_col].clip(upper=0).sum())
    profit_factor = float(gross_profit/gross_loss) if gross_loss>0 else (float("inf") if gross_profit>0 else np.nan)

    return dict(
        n_trades=int(len(tr)),
        win_rate=float(winners.mean()) if len(tr) else 0.0,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        net_pnl=float(tr[pnl_col].sum()),
        avg_win=float(tr.loc[winners, pnl_col].mean()) if winners.any() else 0.0,
        avg_loss=float(tr.loc[~winners, pnl_col].mean()) if (~winners).any() else 0.0,
        avg_hold_hours=float(holds.mean()) if holds is not None and len(holds)>0 else np.nan,
    )

def _compute_curve_metrics(eq: pd.DataFrame) -> Dict[str, float]:
    eq = eq.dropna(subset=["timestamp","equity"]).sort_values("timestamp")
    if eq.empty:
        return dict(cagr=np.nan, max_dd=np.nan, mar=np.nan, sharpe_ann=np.nan, start=pd.NaT, end=pd.NaT)

    start_ts, end_ts = eq["timestamp"].iloc[0], eq["timestamp"].iloc[-1]
    start_val, end_val = float(eq["equity"].iloc[0]), float(eq["equity"].iloc[-1])

    # CAGR
    years = (end_ts - start_ts).total_seconds()/(365.25*24*3600)
    cagr = (end_val/start_val)**(1/years)-1 if (years>0 and start_val>0) else np.nan

    # Max DD
    run_max = eq["equity"].cummax()
    dd = eq["equity"]/run_max - 1.0
    max_dd = float(dd.min()) if not dd.empty else np.nan

    # MAR
    mar = float(cagr/abs(max_dd)) if (max_dd<0 and pd.notna(cagr)) else np.nan

    # Sharpe: resample to daily last equity → daily returns → annualize by sqrt(365)
    eq_d = eq.set_index("timestamp")["equity"].resample("1D").last().dropna()
    r = eq_d.pct_change().dropna()
    sharpe_ann = float(r.mean()/r.std()*np.sqrt(365.0)) if (len(r)>=2 and r.std()>0) else np.nan

    return dict(cagr=float(cagr) if cagr==cagr else np.nan,
                max_dd=float(max_dd) if max_dd==max_dd else np.nan,
                mar=float(mar) if mar==mar else np.nan,
                sharpe_ann=float(sharpe_ann) if sharpe_ann==sharpe_ann else np.nan,
                start=start_ts, end=end_ts)

def evaluate_variant(name: str, trades_path: str, equity_path: str) -> Dict[str, object]:
    tr = _read_trades(Path(trades_path))
    eq = _read_equity(Path(equity_path))
    tm = _compute_trade_metrics(tr)
    cm = _compute_curve_metrics(eq)
    return {
        "variant": name, **tm, **cm
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare Donch variants from trades/equity files.")
    parser.add_argument("--variant", nargs=3, action="append",
                        metavar=("NAME","TRADES_CSV","EQUITY_CSV"),
                        help="Add a variant triple. May be repeated.")
    parser.add_argument("--out", default="comparison_metrics.csv", help="Output CSV path.")
    args = parser.parse_args(argv)

    rows: List[Dict[str, object]] = []
    if args.variant:
        for name, t, e in args.variant:
            rows.append(evaluate_variant(name, t, e))
    else:
        defaults = [
            ("priceATR", "trades_priceATR_baseline.csv", "equity_priceATR_baseline.csv"),
            ("avwap_dynamic", "trades_avwap_dynamic_breakout.csv", "equity_avwap_dynamic_breakout.csv"),
        ]
        for name, t, e in defaults:
            if Path(t).exists() and Path(e).exists():
                rows.append(evaluate_variant(name, t, e))

    if not rows:
        print("No variants provided and defaults not found. Use --variant NAME trades.csv equity.csv", file=sys.stderr)
        sys.exit(2)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    # Console preview
    with pd.option_context('display.max_columns', None):
        print(out.round({
            "win_rate": 4, "profit_factor": 3, "cagr": 4, "max_dd": 4, "mar": 3, "sharpe_ann": 3,
            "avg_hold_hours": 2, "avg_win": 2, "avg_loss": 2, "gross_profit": 2, "gross_loss": 2, "net_pnl": 2
        }))

if __name__ == "__main__":
    main()
