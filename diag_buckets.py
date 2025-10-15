# diag_buckets.py
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import config as cfg

OUTDIR = Path(cfg.RESULTS_DIR) / "diag"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_trades() -> pd.DataFrame:
    t = pd.read_csv(cfg.RESULTS_DIR / "trades.csv", parse_dates=["entry_ts","exit_ts"])
    return t

def load_signals() -> pd.DataFrame:
    s = pq.read_table(cfg.SIGNALS_DIR / "signals.parquet").to_pandas()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True, errors="coerce")
    s = s.rename(columns={"timestamp":"entry_ts"})
    return s

def qbucket(s: pd.Series, q=5, label=None):
    try:
        return pd.qcut(s.astype(float), q, duplicates="drop")
    except Exception:
        return pd.Series([np.nan] * len(s), index=s.index, name=label or s.name)

def main():
    t = load_trades()
    s = load_signals()
    df = t.merge(
        s[["entry_ts","symbol","rs_pct","vol_mult","don_dist_atr","eth_macd_hist_4h"]],
        on=["entry_ts","symbol"], how="left"
    )

    # 1) Exit-reason mix & payoffs
    mix = (df.groupby("exit_reason")["pnl_R"]
            .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
            .sort_values("mean", ascending=False))
    mix.to_csv(OUTDIR / "exit_reason_payoffs.csv")
    print("\n== Exit reason payoffs ==\n", mix)

    # 2) RS deciles
    if "rs_pct" in df.columns:
        df["rs_dec"] = pd.qcut(df["rs_pct"].fillna(-1), 10, labels=False, duplicates="drop")
        rs_tab = df.groupby("rs_dec")["pnl_R"].mean()
        rs_tab.to_csv(OUTDIR / "rs_decile_avgR.csv")
        print("\n== RS deciles (avg R) ==\n", rs_tab)

    # 3) Volume spike strength buckets
    if "vol_mult" in df.columns:
        vol_b = qbucket(df["vol_mult"], 5, "vol_mult")
        vol_tab = df.groupby(vol_b)["pnl_R"].mean()
        vol_tab.to_csv(OUTDIR / "vol_mult_quintile_avgR.csv")
        print("\n== vol_mult quintiles (avg R) ==\n", vol_tab)

    # 4) Donch stretch (distance above breakout at entry, in ATR)
    if "don_dist_atr" in df.columns:
        dd_b = qbucket(df["don_dist_atr"], 5, "don_dist_atr")
        dd_tab = df.groupby(dd_b)["pnl_R"].mean()
        dd_tab.to_csv(OUTDIR / "don_dist_atr_quintile_avgR.csv")
        print("\n== don_dist_atr quintiles (avg R) ==\n", dd_tab)

    # 5) Regime strength bucket
    if "eth_macd_hist_4h" in df.columns:
        rh_b = qbucket(df["eth_macd_hist_4h"], 5, "eth_macd_hist_4h")
        rh_tab = df.groupby(rh_b)["pnl_R"].mean()
        rh_tab.to_csv(OUTDIR / "eth_hist_quintile_avgR.csv")
        print("\n== ETH 4h MACD hist quintiles (avg R) ==\n", rh_tab)

    # Save merged sample for inspection
    df.head(200).to_csv(OUTDIR / "merged_sample_head.csv", index=False)

if __name__ == "__main__":
    main()
