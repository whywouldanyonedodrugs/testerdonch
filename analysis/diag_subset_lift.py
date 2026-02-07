# diag_subset_lift.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRED_PATH = ROOT / "results" / "meta" / "oos_predictions.parquet"
TRADES_PATH = ROOT / "results" / "trades.csv"
OUT_DIR = ROOT / "results" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pf(series: pd.Series) -> float:
    gp = series[series > 0].sum()
    gl = -series[series <= 0].sum()
    if gl > 0:
        return float(gp / gl)
    return float("inf") if gp > 0 else np.nan

def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.80

    preds = pd.read_parquet(PRED_PATH)
    trades = pd.read_csv(TRADES_PATH, parse_dates=["entry_ts"])

    preds["entry_ts"] = pd.to_datetime(preds["entry_ts"], utc=True, errors="coerce")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")

    prev = preds["y_true"].mean()
    print(f"Prevalence (baseline AP): {prev:.4f}")  # PR-AUC should exceed this baseline

    df = preds.merge(
        trades[["symbol", "entry_ts", "pnl", "pnl_R"]],
        on=["symbol", "entry_ts"],
        how="inner",
    )

    rows = []
    def add_row(tag: str, d: pd.DataFrame):
        if len(d) == 0:
            rows.append(dict(tag=tag, n=0, win=0.0, pf=np.nan, meanR=np.nan, meanPnL=np.nan))
            return
        win = (d["pnl"] > 0).mean()
        rows.append(dict(
            tag=tag, n=int(len(d)), win=float(win*100.0),
            pf=pf(d["pnl"]), meanR=float(d["pnl_R"].mean()), meanPnL=float(d["pnl"].mean())
        ))

    add_row("ALL", df)
    add_row(f"p>={thr:.2f}", df.loc[df["y_proba"] >= thr])

    # Multi-threshold sweep (quick EV curve on realized trades)
    thresholds = np.round(np.linspace(0.50, 0.95, 10), 2)
    sweep = []
    for t in thresholds:
        d = df.loc[df["y_proba"] >= float(t)]
        sweep.append(dict(threshold=float(t), n=int(len(d)), meanR=float(d["pnl_R"].mean()), meanPnL=float(d["pnl"].mean())))
    sweep_df = pd.DataFrame(sweep).sort_values("threshold")
    sweep_df.to_csv(OUT_DIR / "subset_lift_ev_curve.csv", index=False)

    out = pd.DataFrame(rows)
    print("\n=== Subset lift (realized trades) ===")
    print(out.to_string(index=False))
    out.to_csv(OUT_DIR / f"subset_lift_{thr:.2f}.csv", index=False)
    print(f"\nSaved: {OUT_DIR / f'subset_lift_{thr:.2f}.csv'}")
    print(f"Saved: {OUT_DIR / 'subset_lift_ev_curve.csv'}")

if __name__ == "__main__":
    main()
