#!/usr/bin/env python3
"""
analysis_runner.py — unified analytics for LIVE and BACKTEST trades.

Works with:
  • live exports (trades.csv)
  • backtester aggregates (results/trades_aggregated_*.csv)

It auto-detects columns, computes what it can, and skips sections that
lack inputs (e.g., VWAP/Z/consolidation) without failing.

Usage:
  python analysis_runner.py --csv path/to/trades.csv [--tp_grid] [--debug]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# --- proportion_confint (import or safe fallback) ---
try:
    from statsmodels.stats.proportion import proportion_confint  # type: ignore
except Exception:
    import numpy as np
    from math import sqrt

    def proportion_confint(count, nobs, alpha: float = 0.05, method: str = "wilson"):
        """95% Wilson interval fallback if statsmodels is absent."""
        if nobs == 0:
            return (np.nan, np.nan)
        p = count / nobs
        z = 1.959963984540054  # ~N(0,1) for 95%
        denom = 1 + (z**2) / nobs
        centre = (p + (z**2) / (2 * nobs)) / denom
        radius = z * sqrt(p * (1 - p) / nobs + (z**2) / (4 * nobs**2)) / denom
        return (max(0.0, centre - radius), min(1.0, centre + radius))

# ----------------- small utils -----------------

def wilson_ci(k: int, n: int) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = 1.959963984540054  # 95%
    denom = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + (z*z)/(4*n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def profit_factor(pnls: np.ndarray) -> float:
    if pnls.size == 0:
        return float("nan")
    gp = pnls[pnls > 0].sum()
    gl = pnls[pnls < 0].sum()
    if gl == 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / abs(gl)

def fmt_pct(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "   nan%"
    return f"{x * 100:6.2f}%"

def infer_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
         .fillna(False)
    )

def to_utc(series_or_str):
    return pd.to_datetime(series_or_str, utc=True, errors="coerce")

# ----------------- column resolvers -----------------

_PNL_CANDS = [
    "final_pnl",        # backtester
    "pnl", "PnL", "pnl_usd", "net_pnl", "netPnl",
    "realized_pnl", "realizedPnl", "pnl_total", "profit", "Profit"
]
_ENTRY_TS_CANDS = ["entry_ts","timestamp_at_entry","entry_time","entry","time_entry","timestamp"]
_EXIT_TS_CANDS  = ["exit_ts","time_exited","exit_time","exit","time_exit"]
_EXIT_REASON_CANDS = ["final_reason","exit_reason","reason","exitReason"]  # backtester uses final_reason
_OUTCOME_CANDS = ["trade_outcome","outcome","result"]   # Win/Loss strings

def pick_first(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

# ----------------- loader -----------------

def load_trades(csv_path: Path, debug: bool=False) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path, low_memory=False)
    if debug:
        print("\n[DEBUG] Columns in CSV:")
        print(", ".join(df_raw.columns))

    df = df_raw.copy()

    # ---- timestamps
    ent_col = pick_first(df, _ENTRY_TS_CANDS)
    exi_col = pick_first(df, _EXIT_TS_CANDS)
    df["entry_ts"] = to_utc(df[ent_col]) if ent_col else pd.NaT
    df["exit_ts"]  = to_utc(df[exi_col]) if exi_col else pd.NaT

    # hour-of-day
    if "hour_of_day_at_entry" not in df.columns:
        if "entry_ts" in df.columns:
            df["hour_of_day_at_entry"] = df["entry_ts"].dt.hour
        else:
            df["hour_of_day_at_entry"] = np.nan

    # ---- symbol
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    else:
        df["symbol"] = "UNKNOWN"

    # ---- pnl
    pnl_col = pick_first(df, _PNL_CANDS)
    if pnl_col:
        df["pnl"] = pd.to_numeric(df[pnl_col], errors="coerce")
    else:
        df["pnl"] = np.nan

    # ---- exit reason
    er_col = pick_first(df, _EXIT_REASON_CANDS)
    df["exit_reason"] = df[er_col].astype(str) if er_col else ""

    # ---- is_win
    out_col = pick_first(df, _OUTCOME_CANDS)
    if "is_win" in df.columns:
        df["is_win"] = infer_bool_series(df["is_win"])
    elif out_col:
        o = df[out_col].astype(str).str.strip().str.upper()
        df["is_win"] = o.eq("WIN")
    else:
        # Derive from reason/pnl if outcome not present
        er = df["exit_reason"].str.upper()
        tp_like = er.isin(["TP","TP1","TP_FINAL","TAKE_PROFIT"])
        trail   = er.eq("TRAIL")
        pnl_pos = pd.to_numeric(df["pnl"], errors="coerce") > 0
        df["is_win"] = (tp_like | trail | pnl_pos).fillna(False)

    # ---- optional VWAP/Z/Consolidation (if present)
    if "pct_to_vwap" in df.columns:
        df["pct_to_vwap"] = pd.to_numeric(df["pct_to_vwap"], errors="coerce")
    if "vwap_z_at_entry" in df.columns:
        df["vwap_z_at_entry"] = pd.to_numeric(df["vwap_z_at_entry"], errors="coerce")
    if "vwap_consolidated_at_entry" in df.columns:
        df["vwap_consolidated_at_entry"] = infer_bool_series(df["vwap_consolidated_at_entry"])

    if debug:
        print("\n[DEBUG] First 10 rows:")
        with pd.option_context("display.max_rows", 12, "display.max_columns", 200):
            print(df.head(10))

    return df

# ----------------- sections -----------------

def section_overall(df: pd.DataFrame):
    wr_mask = df["is_win"].isin([True, False])
    n_wr = int(wr_mask.sum())
    wins = int(df.loc[wr_mask, "is_win"].sum()) if n_wr else 0
    wr = wins / n_wr if n_wr else float("nan")
    lwr,upr = wilson_ci(wins, n_wr) if n_wr else (float("nan"), float("nan"))

    pf_mask = df["pnl"].notna()
    n_pf = int(pf_mask.sum())
    pf = profit_factor(df.loc[pf_mask, "pnl"].to_numpy()) if n_pf else float("nan")
    avg = df.loc[pf_mask, "pnl"].mean() if n_pf else float("nan")

    print("\n" + "="*68)
    print(" OVERALL METRICS")
    print("="*68)
    print(f"N (rows): {len(df):<5d}   N (WR): {n_wr:<5d}   Wins: {wins:<5d}   WR: {fmt_pct(wr)}  (95% CI {fmt_pct(lwr)}–{fmt_pct(upr)})")
    if n_pf:
        print(f"N (PF):  {n_pf:<5d}   PF: {pf:0.3f}   Avg PnL: {avg:0.4f}")
    else:
        print("PF/Avg PnL: unavailable (no numeric PnL detected)")

def section_hour_pooling(df: pd.DataFrame, min_hour_n: int, weak_wr: float):
    if "hour_of_day_at_entry" not in df.columns or df["hour_of_day_at_entry"].isna().all():
        print("\n(No hour-of-day info available.)")
        return

    t = (
        df.groupby("hour_of_day_at_entry")
          .agg(n=("is_win","size"), wins=("is_win","sum"))
          .assign(wr=lambda x: x["wins"]/x["n"])
          .reset_index()
          .sort_values("hour_of_day_at_entry")
    )
    print("\n" + "="*68)
    print(" HOUR POOLING  (candidate blocked hours: WR<%.0f%%, n≥%d)" % (weak_wr*100, min_hour_n))
    print("="*68)
    with pd.option_context("display.max_rows", 200):
        print(t.to_string(index=False, formatters={"wr":lambda v: f"{v:0.6f}"}))

    cand = t[(t["n"]>=min_hour_n) & (t["wr"]<weak_wr)]["hour_of_day_at_entry"].tolist()
    if cand:
        kept = df[~df["hour_of_day_at_entry"].isin(cand)]
        n = len(kept); wins = int(kept["is_win"].sum())
        wr = wins/n if n else float("nan")
        pf = profit_factor(kept["pnl"].dropna().to_numpy()) if "pnl" in kept.columns else float("nan")
        avg = kept["pnl"].dropna().mean() if "pnl" in kept.columns else float("nan")
        print(f"\nCandidate blocked hours: {cand}")
        print(f"What-if keep non-weak hours → N={n}  WR={fmt_pct(wr)}  PF={pf:0.3f}  Avg={avg:0.4f}")
    else:
        print("\nNo blocked-hour candidates under current thresholds.")

def section_vwap_dev(df: pd.DataFrame):
    if "pct_to_vwap" not in df.columns:
        print("\n(VWAP distance not available; skipping VWAP grids.)")
        return
    print("\n" + "="*68)
    print(" VWAP ABS DISTANCE GRID  (keep |pct_to_vwap| ≥ t)")
    print("="*68)
    thresholds = [0.025, 0.02, 0.015, 0.01, 0.005]
    rows=[]
    for t in thresholds:
        kept = df[df["pct_to_vwap"].abs() >= t]
        n=len(kept); wins=int(kept["is_win"].sum())
        wr = wins/n if n else float("nan")
        lwr,upr = wilson_ci(wins,n) if n else (float("nan"), float("nan"))
        pf = profit_factor(kept["pnl"].dropna().to_numpy())
        avg = kept["pnl"].dropna().mean() if "pnl" in kept.columns else float("nan")
        gp = kept["pnl"][kept["pnl"]>0].sum(skipna=True)
        gl = kept["pnl"][kept["pnl"]<0].sum(skipna=True).__abs__()
        rows.append([n,wins,wr,lwr,upr,pf,avg,gp,gl,f"pct_to_vwap ≥ {t}", n, len(df)-n])
    out = pd.DataFrame(rows, columns=["n","wins","win_rate","wr_lwr","wr_upr","pf","avg_pnl","gp","gl","rule","kept","dropped"])
    with pd.option_context("display.max_rows", 200):
        print(out.to_string(index=False, formatters={"win_rate":lambda v:f"{v:0.6f}","wr_lwr":lambda v:f"{v:0.6f}","wr_upr":lambda v:f"{v:0.6f}","pf":lambda v:f"{v:0.6f}","avg_pnl":lambda v:f"{v:0.6f}"}))

def section_vwap_z(df: pd.DataFrame) -> None:
    # Need a Z column
    if "vwap_z_at_entry" not in df.columns:
        print("\n(VWAP Z not available; skipping Z veto grid.)\n")
        return

    # Choose which PnL column to use
    pnl_col = (
        "final_pnl" if "final_pnl" in df.columns
        else ("pnl" if "pnl" in df.columns else None)
    )
    if pnl_col is None:
        print("\n(No PnL column found; skipping Z veto grid.)\n")
        return

    zabs = df["vwap_z_at_entry"].abs()
    thresh = [1.0, 1.5, 2.0, 2.5]

    rows = []
    for t in thresh:
        kept = df[zabs <= t]
        dropped = df[zabs > t]

        n = len(kept)
        wins = int(kept.get("is_win", pd.Series(False, index=kept.index)).sum())
        wr = wins / n if n else np.nan
        wr_lwr, wr_upr = proportion_confint(wins, n) if n else (np.nan, np.nan)

        # Gross profit / loss on the chosen PnL column
        gp = kept.loc[kept[pnl_col] > 0, pnl_col].sum(skipna=True)
        gl = kept.loc[kept[pnl_col] < 0, pnl_col].sum(skipna=True)
        gl_abs = abs(gl)
        pf = (gp / gl_abs) if gl_abs > 0 else np.nan
        avg = kept[pnl_col].mean() if n else np.nan

        rows.append({
            "n": n,
            "wins": wins,
            "win_rate": wr,
            "wr_lwr": wr_lwr,
            "wr_upr": wr_upr,
            "pf": pf,
            "avg_pnl": avg,
            "gp": gp,
            "gl": gl_abs,
            "rule": f"|vwap_z_at_entry| ≤ {t}",
            "kept": n,
            "dropped": len(dropped),
        })

    if rows:
        out = pd.DataFrame(rows, columns=[
            "n","wins","win_rate","wr_lwr","wr_upr","pf","avg_pnl","gp","gl","rule","kept","dropped"
        ])
        print("\n" + "="*68)
        print(" VWAP Z EXTREME VETO GRID  (keep |z| ≤ t)")
        print("="*68)
        print(out.to_string(index=False))
        print()


def section_consolidation(df: pd.DataFrame):
    if "vwap_consolidated_at_entry" not in df.columns:
        print("\n(VWAP consolidation flag not available; skipping.)")
        return
    flag = infer_bool_series(df["vwap_consolidated_at_entry"])
    t = (
        pd.DataFrame({"consolidated": flag, "pnl": df.get("pnl", np.nan), "is_win": df["is_win"]})
        .groupby("consolidated")
        .agg(n=("is_win","size"),
             wins=("is_win","sum"),
             wr=("is_win","mean"),
             pf=("pnl", lambda s: profit_factor(s.dropna().to_numpy())),
             avg=("pnl", lambda s: s.dropna().mean() if s.notna().any() else float("nan")))
        .reset_index()
    )
    print("\n" + "="*68)
    print(" VWAP_CONSOLIDATED_AT_ENTRY – WIN RATE BY CATEGORY")
    print("="*68)
    if not t.empty:
        t["wr_pct"] = t["wr"]*100
        with pd.option_context("display.max_rows", 200):
            print(t[["consolidated","n","wins","wr_pct","pf","avg"]].to_string(index=False, formatters={"wr_pct":lambda v:f"{v:6.2f}"}))
    else:
        print("(No data to report.)")

def section_exit_snapshot(df: pd.DataFrame, run_grid: bool):
    if not run_grid:
        print("\n(Payoff what-if grid disabled; pass --tp_grid to enable.)")
        return
    if "exit_reason" not in df.columns:
        print("\n(No exit_reason column; skipping payoff snapshot.)")
        return
    er = df["exit_reason"].astype(str).str.upper()
    tp_like = er.isin(["TP","TP1","TP_FINAL","TAKE_PROFIT"])
    trail   = er.eq("TRAIL")
    sl      = er.isin(["SL","STOP","STOP_LOSS"])
    other   = ~(tp_like | trail | sl)
    print("\n" + "="*68)
    print(" EXIT REASONS SNAPSHOT")
    print("="*68)
    total = len(df)
    for name, m in [("TP-like", tp_like), ("Trail", trail), ("SL", sl), ("Other", other)]:
        print(f"{name:7s}: {int(m.sum()):5d}  ({m.mean()*100:5.1f}%) of {total}")

def section_power(df: pd.DataFrame):
    wr_mask = df["is_win"].isin([True, False])
    n = int(wr_mask.sum())
    if n == 0:
        return
    wr = df.loc[wr_mask, "is_win"].mean()
    lwr, upr = wilson_ci(int(wr*n), n)
    delta = 0.04  # 4 percentage points
    z_alpha = 1.2816  # ~90% one-sided
    z_beta  = 0.8416  # ~80% power
    var = wr*(1-wr)
    n_needed = int(((z_alpha*math.sqrt(2*var) + z_beta*math.sqrt(wr*(1-wr) + (wr+delta)*(1-(wr+delta))))/delta)**2)
    print("\n" + "="*68)
    print(" POWER / SAMPLE-SIZE")
    print("="*68)
    print(f"Observed WR: {fmt_pct(wr)}  95% CI: {fmt_pct(lwr)}–{fmt_pct(upr)}")
    print(f"N needed to detect a +4 pp WR lift at ~80% power (rough): ~{n_needed}")

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to trades CSV (live or backtest).")
    ap.add_argument("--min_hour_n", type=int, default=6, help="Min trades per hour to consider blocking.")
    ap.add_argument("--weak_wr", type=float, default=0.45, help="Hours with WR below this are candidates.")
    ap.add_argument("--tp_grid", action="store_true", help="Print exit reason snapshot (quick sanity).")
    ap.add_argument("--debug", action="store_true", help="Print columns and head(10) for sanity.")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return

    df = load_trades(path, debug=args.debug)

    section_overall(df)
    section_vwap_dev(df)
    section_vwap_z(df)
    section_consolidation(df)
    section_hour_pooling(df, args.min_hour_n, args.weak_wr)
    section_exit_snapshot(df, args.tp_grid)
    section_power(df)

    print("\nDone.")

if __name__ == "__main__":
    main()
