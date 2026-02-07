#!/usr/bin/env python3
"""
research/06_tp_sl_diagnostics.py

Step 06: TP/SL + time-exit diagnostics.

Goals:
  - Quantify exit mix (TP / SL / TIME / OTHER) and how it varies by regime / score.
  - Understand MAE/MFE profiles: how close losers got to TP; how close winners got to SL.
  - Characterize TIME exits (stale trades): win/loss rate, pnl_R distribution, time-in-trade.
  - Optional: join calibrated scores from step 05 and analyze by score deciles.

Inputs:
  - trades CSV: results/trades.clean.csv
  - optional oof_with_calibrated_*.parquet (from step 05), or any file with trade_id + score col

Outputs (in outdir):
  - exit_counts.csv
  - exit_stats.csv
  - time_in_trade_quantiles_by_exit.csv
  - mae_mfe_quantiles_by_exit.csv
  - sl_trades_mfe_vs_tp.csv
  - tp_trades_mae_vs_sl.csv
  - time_exit_breakdown.csv
  - (optional) score_bins_exit_mix.csv
  - (optional) score_bins_stats.csv
  - (optional) plots/*.png

Notes:
  - Assumes 'side' is long (your system currently hardcodes long). Still works if side absent.
  - Uses atr_at_entry + mae_over_atr / mfe_over_atr when present; falls back to absolute excursions if not.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_dt_utc(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True, errors="coerce")
    if pd.api.types.is_numeric_dtype(s):
        x = _to_num(s)
        mx = x.max()
        unit = "ms" if (mx is not None and np.isfinite(mx) and mx > 1e12) else "s"
        return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    out[m] = a[m] / b[m]
    return out


def _exit_class(df: pd.DataFrame) -> pd.Series:
    """
    Robustly map exit_reason/EXIT_FINAL to {TP, SL, TIME, OTHER}.
    EXIT_FINAL: 1=TP, 0=SL, 2=TIME (per your note).
    """
    cls = pd.Series(["OTHER"] * len(df), index=df.index, dtype="object")

    if "EXIT_FINAL" in df.columns:
        ef = _to_num(df["EXIT_FINAL"])
        cls.loc[ef == 1] = "TP"
        cls.loc[ef == 0] = "SL"
        cls.loc[ef == 2] = "TIME"

    if "exit_reason" in df.columns:
        er = df["exit_reason"].astype(str).str.lower()
        cls.loc[er.str.contains("tp", na=False)] = "TP"
        cls.loc[er.str.contains("sl", na=False)] = "SL"
        cls.loc[er.str.contains("time", na=False)] = "TIME"

    return cls


def _quantile_table(df: pd.DataFrame, group_col: str, value_cols: List[str], qs: List[float]) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col, dropna=False):
        row: Dict[str, object] = {"group": g, "n": int(len(sub))}
        for c in value_cols:
            x = _to_num(sub[c]).to_numpy(dtype=np.float64)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                for q in qs:
                    row[f"{c}_q{int(q*100)}"] = np.nan
                row[f"{c}_mean"] = np.nan
                continue
            for q in qs:
                row[f"{c}_q{int(q*100)}"] = float(np.quantile(x, q))
            row[f"{c}_mean"] = float(np.mean(x))
        rows.append(row)
    return pd.DataFrame(rows)


def _maybe_hist(df: pd.DataFrame, col: str, by: str, outpng: str, bins: int = 50) -> None:
    if not HAS_PLT or col not in df.columns or by not in df.columns:
        return
    plt.figure()
    for g, sub in df.groupby(by):
        x = _to_num(sub[col]).to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 10:
            continue
        plt.hist(x, bins=bins, alpha=0.4, label=str(g), density=True)
    plt.legend()
    plt.title(f"{col} by {by}")
    plt.savefig(outpng, dpi=160, bbox_inches="tight")
    plt.close()


def _prob_bins(s: pd.Series, n_bins: int) -> pd.Series:
    x = _to_num(s)
    # equal-frequency bins are more stable than equal-width for skewed scores
    try:
        b = pd.qcut(x, q=n_bins, duplicates="drop")
        return b.astype(str)
    except Exception:
        # fallback to equal-width
        b = pd.cut(x, bins=n_bins)
        return b.astype(str)


# ---------------------------
# Main
# ---------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 06: TP/SL/time-exit diagnostics.")
    p.add_argument("--trades", type=str, default="results/trades.clean.csv")
    p.add_argument("--outdir", type=str, default="research_outputs/06_tp_sl_diagnostics")

    # Optional score join (Step 05 output)
    p.add_argument("--scores", type=str, default="", help="Optional parquet/csv with trade_id + score column(s).")
    p.add_argument("--score-col", type=str, default="", help="Score column to use (e.g. p_lgbm_cal).")
    p.add_argument("--score-bins", type=int, default=10)

    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _ensure_dir(args.outdir)
    plotdir = os.path.join(args.outdir, "plots")
    if not args.no_plots:
        _ensure_dir(plotdir)

    if not os.path.exists(args.trades):
        raise FileNotFoundError(f"Trades file not found: {args.trades}")

    df = pd.read_csv(args.trades, low_memory=False)
    if df.empty:
        raise RuntimeError("Trades CSV is empty.")

    if "trade_id" not in df.columns:
        raise RuntimeError("Trades CSV must contain trade_id.")
    df["trade_id"] = _to_num(df["trade_id"])
    df = df[df["trade_id"].notna()].copy()
    df["trade_id"] = df["trade_id"].astype("int64")
    df = df.drop_duplicates(subset=["trade_id"]).set_index("trade_id", drop=True)

    # timestamps
    if "entry_ts" in df.columns:
        df["entry_ts"] = _to_dt_utc(df["entry_ts"])
    if "exit_ts" in df.columns:
        df["exit_ts"] = _to_dt_utc(df["exit_ts"])

    # core numeric
    for c in ["entry", "exit", "sl", "tp", "atr_at_entry", "mae_over_atr", "mfe_over_atr", "pnl_R"]:
        if c in df.columns:
            df[c] = _to_num(df[c])

    # exit class
    df["exit_class"] = _exit_class(df)

    # time in trade (hours)
    if "entry_ts" in df.columns and "exit_ts" in df.columns:
        dt = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 3600.0
        df["hours_in_trade"] = dt
    else:
        df["hours_in_trade"] = np.nan

    # distances
    # For long: stop distance = entry - sl if sl below entry; but use abs robustly.
    if "entry" in df.columns and "sl" in df.columns:
        df["sl_dist_abs"] = (df["entry"] - df["sl"]).abs()
    else:
        df["sl_dist_abs"] = np.nan

    if "entry" in df.columns and "tp" in df.columns:
        df["tp_dist_abs"] = (df["tp"] - df["entry"]).abs()
    else:
        df["tp_dist_abs"] = np.nan

    # excursions in absolute terms when possible
    if "atr_at_entry" in df.columns and "mae_over_atr" in df.columns:
        df["mae_abs"] = df["mae_over_atr"] * df["atr_at_entry"]
    else:
        df["mae_abs"] = np.nan

    if "atr_at_entry" in df.columns and "mfe_over_atr" in df.columns:
        df["mfe_abs"] = df["mfe_over_atr"] * df["atr_at_entry"]
    else:
        df["mfe_abs"] = np.nan

    # normalize excursions vs planned distances
    df["mfe_frac_tp"] = _safe_div(df["mfe_abs"].to_numpy(dtype=np.float64), df["tp_dist_abs"].to_numpy(dtype=np.float64))
    df["mae_frac_sl"] = _safe_div(df["mae_abs"].to_numpy(dtype=np.float64), df["sl_dist_abs"].to_numpy(dtype=np.float64))

    # ---------------------------
    # 1) Exit counts + basic stats
    # ---------------------------
    exit_counts = df.groupby("exit_class").size().reset_index(name="n").sort_values("n", ascending=False)
    exit_counts.to_csv(os.path.join(args.outdir, "exit_counts.csv"), index=False)

    stats_cols = ["pnl_R", "hours_in_trade", "mae_over_atr", "mfe_over_atr", "mfe_frac_tp", "mae_frac_sl"]
    present = [c for c in stats_cols if c in df.columns]
    exit_stats = df.groupby("exit_class")[present].agg(["count", "mean", "median"]).reset_index()
    exit_stats.to_csv(os.path.join(args.outdir, "exit_stats.csv"), index=False)

    # ---------------------------
    # 2) Quantiles by exit class
    # ---------------------------
    qtab_time = _quantile_table(df, "exit_class", ["hours_in_trade"], qs=[0.05, 0.25, 0.50, 0.75, 0.95])
    qtab_time.to_csv(os.path.join(args.outdir, "time_in_trade_quantiles_by_exit.csv"), index=False)

    qcols = []
    for c in ["mae_over_atr", "mfe_over_atr", "mae_frac_sl", "mfe_frac_tp", "pnl_R"]:
        if c in df.columns:
            qcols.append(c)
    qtab_exc = _quantile_table(df, "exit_class", qcols, qs=[0.05, 0.25, 0.50, 0.75, 0.95])
    qtab_exc.to_csv(os.path.join(args.outdir, "mae_mfe_quantiles_by_exit.csv"), index=False)

    # ---------------------------
    # 3) Key TP/SL diagnostics
    # ---------------------------
    # For SL exits: how much MFE relative to TP (did they almost hit TP before stopping?)
    sl = df[df["exit_class"] == "SL"].copy()
    sl_rows = []
    if not sl.empty and "mfe_frac_tp" in sl.columns:
        x = sl["mfe_frac_tp"].to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x):
            sl_rows.append({"metric": "mfe_frac_tp_mean", "value": float(np.mean(x)), "n": int(len(x))})
            for q in [0.25, 0.50, 0.75, 0.90, 0.95]:
                sl_rows.append({"metric": f"mfe_frac_tp_q{int(q*100)}", "value": float(np.quantile(x, q)), "n": int(len(x))})
            # Useful “near miss” rates
            for thr in [0.25, 0.50, 0.75, 0.90]:
                sl_rows.append({"metric": f"mfe_frac_tp_ge_{thr}", "value": float(np.mean(x >= thr)), "n": int(len(x))})
    pd.DataFrame(sl_rows).to_csv(os.path.join(args.outdir, "sl_trades_mfe_vs_tp.csv"), index=False)

    # For TP exits: how much MAE relative to SL (did winners come close to stopping out?)
    tp = df[df["exit_class"] == "TP"].copy()
    tp_rows = []
    if not tp.empty and "mae_frac_sl" in tp.columns:
        x = tp["mae_frac_sl"].to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x):
            tp_rows.append({"metric": "mae_frac_sl_mean", "value": float(np.mean(x)), "n": int(len(x))})
            for q in [0.25, 0.50, 0.75, 0.90, 0.95]:
                tp_rows.append({"metric": f"mae_frac_sl_q{int(q*100)}", "value": float(np.quantile(x, q)), "n": int(len(x))})
            for thr in [0.50, 0.75, 0.90, 1.00]:
                tp_rows.append({"metric": f"mae_frac_sl_ge_{thr}", "value": float(np.mean(x >= thr)), "n": int(len(x))})
    pd.DataFrame(tp_rows).to_csv(os.path.join(args.outdir, "tp_trades_mae_vs_sl.csv"), index=False)

    # ---------------------------
    # 4) TIME exit deep dive
    # ---------------------------
    time_df = df[df["exit_class"] == "TIME"].copy()
    time_rows = []
    if not time_df.empty:
        pnl = _to_num(time_df["pnl_R"]).to_numpy(dtype=np.float64)
        pnl = pnl[np.isfinite(pnl)]
        if len(pnl):
            time_rows.append({"metric": "n_time", "value": float(len(time_df))})
            time_rows.append({"metric": "mean_pnl_R_time", "value": float(np.mean(pnl))})
            time_rows.append({"metric": "pnl_R_time_q50", "value": float(np.quantile(pnl, 0.50))})
            time_rows.append({"metric": "pnl_R_time_q25", "value": float(np.quantile(pnl, 0.25))})
            time_rows.append({"metric": "pnl_R_time_q75", "value": float(np.quantile(pnl, 0.75))})
            time_rows.append({"metric": "time_exit_winrate_pnlR_gt_0", "value": float(np.mean(pnl > 0))})
        if "hours_in_trade" in time_df.columns:
            h = _to_num(time_df["hours_in_trade"]).to_numpy(dtype=np.float64)
            h = h[np.isfinite(h)]
            if len(h):
                time_rows.append({"metric": "hours_in_trade_time_q50", "value": float(np.quantile(h, 0.50))})
                time_rows.append({"metric": "hours_in_trade_time_q90", "value": float(np.quantile(h, 0.90))})

        # break down TIME exits into profitable vs not (by pnl_R)
        time_df["time_profit_flag"] = np.where(_to_num(time_df["pnl_R"]) > 0, "TIME_PROFIT", "TIME_NOT_PROFIT")
        mix = time_df.groupby("time_profit_flag").agg(
            n=("pnl_R", "size"),
            mean_pnl_R=("pnl_R", "mean"),
            median_pnl_R=("pnl_R", "median"),
            mean_hours=("hours_in_trade", "mean"),
        ).reset_index()
        mix.to_csv(os.path.join(args.outdir, "time_exit_breakdown.csv"), index=False)
    pd.DataFrame(time_rows).to_csv(os.path.join(args.outdir, "time_exit_summary.csv"), index=False)

    # ---------------------------
    # 5) Optional: join scores + analyze by score bins
    # ---------------------------
    if args.scores.strip():
        if not os.path.exists(args.scores):
            raise FileNotFoundError(f"Scores file not found: {args.scores}")

        if args.scores.lower().endswith(".parquet"):
            sc = pd.read_parquet(args.scores)
        else:
            sc = pd.read_csv(args.scores, low_memory=False)

        if "trade_id" not in sc.columns:
            raise RuntimeError("Scores file must contain trade_id.")
        sc["trade_id"] = _to_num(sc["trade_id"])
        sc = sc[sc["trade_id"].notna()].copy()
        sc["trade_id"] = sc["trade_id"].astype("int64")
        sc = sc.drop_duplicates(subset=["trade_id"]).set_index("trade_id", drop=True)

        score_col = args.score_col.strip()
        if not score_col:
            # try to auto pick a calibrated column first
            cand = [c for c in sc.columns if c.endswith("_cal")]
            if cand:
                score_col = cand[0]
            else:
                cand2 = [c for c in sc.columns if c.startswith("p_")]
                if cand2:
                    score_col = cand2[0]
        if not score_col or score_col not in sc.columns:
            raise RuntimeError(f"Could not determine --score-col. Available columns sample: {list(sc.columns)[:20]}")

        dfj = df.join(sc[[score_col]], how="left")
        dfj[score_col] = _to_num(dfj[score_col])

        dfj["score_bin"] = _prob_bins(dfj[score_col], n_bins=int(args.score_bins))

        # Exit mix by score bin
        mix = (
            dfj.groupby(["score_bin", "exit_class"])
            .size()
            .reset_index(name="n")
        )
        # Normalize within bin
        tot = mix.groupby("score_bin")["n"].transform("sum")
        mix["pct_in_bin"] = mix["n"] / tot
        mix.to_csv(os.path.join(args.outdir, "score_bins_exit_mix.csv"), index=False)

        # Stats by score bin
        agg = dfj.groupby("score_bin").agg(
            n=("pnl_R", "size"),
            mean_score=(score_col, "mean"),
            mean_pnl_R=("pnl_R", "mean"),
            sum_pnl_R=("pnl_R", "sum"),
            tp_rate=("exit_class", lambda x: float(np.mean(x == "TP"))),
            sl_rate=("exit_class", lambda x: float(np.mean(x == "SL"))),
            time_rate=("exit_class", lambda x: float(np.mean(x == "TIME"))),
            mean_hours=("hours_in_trade", "mean"),
            mean_mfe_over_atr=("mfe_over_atr", "mean") if "mfe_over_atr" in dfj.columns else ("pnl_R", "mean"),
            mean_mae_over_atr=("mae_over_atr", "mean") if "mae_over_atr" in dfj.columns else ("pnl_R", "mean"),
        ).reset_index()
        agg.to_csv(os.path.join(args.outdir, "score_bins_stats.csv"), index=False)

        if not args.no_plots:
            if "mfe_over_atr" in dfj.columns:
                _maybe_hist(dfj, "mfe_over_atr", "exit_class", os.path.join(plotdir, "mfe_over_atr_by_exit.png"), bins=60)
            if "mae_over_atr" in dfj.columns:
                _maybe_hist(dfj, "mae_over_atr", "exit_class", os.path.join(plotdir, "mae_over_atr_by_exit.png"), bins=60)
            _maybe_hist(dfj, "pnl_R", "exit_class", os.path.join(plotdir, "pnl_R_by_exit.png"), bins=80)

    # Plots without score join
    if not args.no_plots and args.scores.strip() == "":
        if "mfe_over_atr" in df.columns:
            _maybe_hist(df, "mfe_over_atr", "exit_class", os.path.join(plotdir, "mfe_over_atr_by_exit.png"), bins=60)
        if "mae_over_atr" in df.columns:
            _maybe_hist(df, "mae_over_atr", "exit_class", os.path.join(plotdir, "mae_over_atr_by_exit.png"), bins=60)
        _maybe_hist(df, "pnl_R", "exit_class", os.path.join(plotdir, "pnl_R_by_exit.png"), bins=80)

    print(f"[06_tp_sl_diagnostics] DONE. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
