# sweep_threshold_and_window.py
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np, pandas as pd

import config as cfg
from meta_model import run_meta
from backtester import run_backtest
import manager as mgr

RESULTS = Path(cfg.RESULTS_DIR); SIGNALS = Path(cfg.SIGNALS_DIR)
META = RESULTS / "meta"; META.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.60,0.65,0.70,0.75,0.80,0.85]
BUSY_MINUTES = [120,180,240,360,480]

def _merge_preds_trades(preds, trades):
    preds = preds.copy(); trades = trades.copy()
    preds["entry_ts"]  = pd.to_datetime(preds["entry_ts"],  utc=True, errors="coerce")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    if "pnl_R" not in trades.columns and {"entry","exit","sl"}.issubset(trades.columns):
        rpu = (trades["entry"] - trades["sl"]).replace(0, np.nan)
        trades["pnl_R"] = (trades["exit"] - trades["entry"]) / rpu
    df = preds.merge(trades[["symbol","entry_ts","pnl_R"]], on=["symbol","entry_ts"], how="left").dropna(subset=["pnl_R"])
    return df

def _filter_signals(preds, t, signals_path):
    keep = preds.loc[preds["y_proba"].ge(float(t)), ["symbol","entry_ts"]].rename(columns={"entry_ts":"timestamp"})
    sig = pd.read_parquet(signals_path)
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
    return sig.merge(keep, on=["symbol","timestamp"], how="inner")

def _dedup_within_symbol_window(sig_df, preds, busy_minutes: int):
    p = preds[["symbol","entry_ts","y_proba"]].copy()
    p["timestamp"] = pd.to_datetime(p["entry_ts"], utc=True)
    sigp = sig_df.merge(p[["symbol","timestamp","y_proba"]], on=["symbol","timestamp"], how="left")
    sigp["y_proba"] = sigp["y_proba"].fillna(0.0)

    out_rows = []
    window = pd.Timedelta(minutes=int(busy_minutes))
    for sym, g in sigp.sort_values(["y_proba","timestamp"], ascending=[False, True]).groupby("symbol"):
        last_kept_ts = None
        for _, row in g.iterrows():
            ts = row["timestamp"]
            if (last_kept_ts is None) or (ts >= last_kept_ts + window):
                out_rows.append(row); last_kept_ts = ts
    out = pd.DataFrame(out_rows).drop(columns=["y_proba"]).sort_values(["timestamp","symbol"])
    return out

def pf(series: pd.Series) -> float:
    gp = series[series > 0].sum(); gl = -series[series <= 0].sum()
    if gl > 0: return float(gp/gl)
    return float("inf") if gp > 0 else np.nan

def kpis(trades_csv: Path) -> dict:
    df = pd.read_csv(trades_csv)
    wr = (df["pnl"] > 0).mean()
    return {
        "n": int(len(df)),
        "win_rate": float(wr),
        "pf": pf(df["pnl"]),
        "mean_R": float(df["pnl_R"].mean()) if "pnl_R" in df.columns else np.nan,
        "net_pnl": float(df["pnl"].sum()) if "pnl" in df.columns else np.nan,
    }

def main():
    # Train/refresh OOS predictions
    run_meta(trades_csv=None, signals_parquet=None, returns_col="pnl_R",
             r_threshold=0.0, blocks=12, k_test=3, embargo=1, max_splits=25,
             outdir=str(META), random_seed=42)
    preds = pd.read_parquet(META / "oos_predictions.parquet")
    trades = pd.read_csv(RESULTS / "trades.csv")
    df = _merge_preds_trades(preds, trades)
    prev = preds["y_true"].mean()
    print(f"Prevalence (baseline AP): {prev:.4f}")

    grid = list(itertools.product(THRESHOLDS, BUSY_MINUTES))
    rows = []

    for t, w in grid:
        # build signals for (t, w)
        sig = _filter_signals(preds, t, SIGNALS / "signals.parquet")
        sig_dedup = _dedup_within_symbol_window(sig, preds, w)
        path = SIGNALS / "signals_sweep.parquet"
        sig_dedup.to_parquet(path, index=False)

        # backtest under your normal constraints
        run_backtest(signals_path=path)
        mgr._postprocess_and_aggregate()

        row = {"threshold": t, "busy_minutes": int(w)}
        row.update(kpis(RESULTS / "trades.csv"))
        print(f"[t={t:.2f}, w={w}] -> n={row['n']} PF={row['pf']:.3f} meanR={row['mean_R']:.4f}")
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["pf","mean_R","n"], ascending=[False, False, False])
    out_path = META / "scheduler_aware_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved sweep table → {out_path}")
    best = out.iloc[0].to_dict()
    with open(META / "scheduler_aware_best.json","w") as f:
        json.dump(best, f, indent=2)
    print("Best combo:", best)

if __name__ == "__main__":
    main()
