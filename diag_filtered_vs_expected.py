# diag_filtered_vs_expected.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import config as cfg

RESULTS = Path(cfg.RESULTS_DIR)
META    = RESULTS / "meta"
SIGDIR  = Path(cfg.SIGNALS_DIR)

def busy_window_dedup(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Keep first signal per symbol; drop others within +/- busy window."""
    if df.empty: return df
    win = pd.Timedelta(minutes=minutes)
    out = []
    for sym, g in df.sort_values("entry_ts").groupby("symbol"):
        keep_rows = []
        last_kept = pd.Timestamp.min.tz_localize("UTC")
        for _, r in g.iterrows():
            if r.entry_ts >= last_kept + win:
                keep_rows.append(r)
                last_kept = r.entry_ts
        if keep_rows:
            out.append(pd.DataFrame(keep_rows))
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]

def pf(x: pd.Series) -> float:
    gp = x[x > 0].sum()
    gl = -x[x <= 0].sum()
    if gl == 0: return float("inf") if gp > 0 else np.nan
    return float(gp / gl)

def summarize(tag: str, d: pd.DataFrame):
    if d.empty:
        print(f"{tag:>12} | n=0")
        return
    wr = (d["pnl"] > 0).mean()
    print(f"{tag:>12} | n={len(d):4d} | win%={wr*100:5.1f} | PF={pf(d['pnl']):6.2f} | mean R={d['pnl_R'].mean():6.3f}")

def main():
    pstar = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
    preds = pd.read_parquet(META / "oos_predictions.parquet")
    preds["entry_ts"] = pd.to_datetime(preds["entry_ts"], utc=True, errors="coerce")

    # Baseline trades (from the *unfiltered* run on the same signals file)
    base_tr = pd.read_csv(RESULTS / "trades.csv", low_memory=False)
    base_tr["entry_ts"] = pd.to_datetime(base_tr["entry_ts"], utc=True, errors="coerce")

    # Expected subset at p*
    exp = preds.loc[preds["y_proba"] >= pstar, ["symbol","entry_ts"]].copy()
    exp = exp.merge(base_tr, on=["symbol","entry_ts"], how="inner")

    # Dedup by busy window to mimic lockouts (default fallback 480 min)
    busy_min = int(getattr(cfg, "BUSY_WINDOW_MINUTES_FOR_META", 480))
    exp_dedup = busy_window_dedup(exp, minutes=busy_min)

    # Realized filtered trades (after auto_meta_pipeline re-backtest)
    filt_tr = pd.read_csv(RESULTS / "trades.csv", low_memory=False)
    filt_tr["entry_ts"] = pd.to_datetime(filt_tr["entry_ts"], utc=True, errors="coerce")

    print(f"Busy window used for dedup: {busy_min} min")
    print("\n=== Expected (from baseline) vs Realized (filtered) ===")
    summarize("expected", exp)
    summarize("exp_dedup", exp_dedup)
    summarize("realized", filt_tr)

if __name__ == "__main__":
    main()
