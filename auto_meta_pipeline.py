from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

import config as cfg
from backtester import run_backtest
import meta_model as mm

RESULTS = Path(cfg.RESULTS_DIR); RESULTS.mkdir(parents=True, exist_ok=True)
SIGNALS = Path(cfg.SIGNALS_DIR); SIGNALS.mkdir(parents=True, exist_ok=True)

# Defaults if missing in config.py
BUSY_WINDOW_MIN = int(getattr(cfg, "BUSY_WINDOW_MINUTES_FOR_META", 480))  # 8h
MIN_KEPT_AFTER_DEDUP = int(getattr(cfg, "MIN_KEPT_AFTER_DEDUP", 100))
CANDIDATE_THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75]

def infer_signals_file() -> Path:
    cands = sorted(SIGNALS.glob("signals_daily_*.parquet"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    p = SIGNALS / "signals.parquet"
    if p.exists(): return p
    raise FileNotFoundError("No signals file found in signals/.")

def ensure_returns_cols(tr: pd.DataFrame) -> pd.DataFrame:
    tr["entry_ts"] = pd.to_datetime(tr["entry_ts"], utc=True, errors="coerce")
    if "pnl" not in tr.columns:
        exit_price = tr["exit"].fillna(tr["entry"])
        notional_in  = tr["entry"] * tr["qty"]
        notional_out = exit_price * tr["qty"]
        fee_rate = float(getattr(cfg, "FEE_RATE", 0.0006))
        fees = fee_rate * (notional_in.abs() + notional_out.abs())
        tr["fees"] = fees
        tr["pnl"]  = (exit_price - tr["entry"]) * tr["qty"] - fees
    if "pnl_R" not in tr.columns:
        risk_per_unit = (tr["entry"] - tr["sl"]).abs().replace(0, np.nan)
        denom = risk_per_unit * tr["qty"]
        tr["pnl_R"] = tr["pnl"] / denom
    return tr

def busy_window_dedup(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Drop signals that arrive within busy window of a prior kept trade per symbol."""
    if df.empty: return df
    win = pd.Timedelta(minutes=minutes)
    out = []
    for sym, g in df.sort_values("timestamp").groupby("symbol"):
        keep_idx = []
        last_ts = pd.Timestamp.min.tz_localize("UTC")
        for i, r in g.iterrows():
            if r.timestamp >= last_ts + win:
                keep_idx.append(i)
                last_ts = r.timestamp
        out.append(g.loc[keep_idx])
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]

def ev_curve(preds_path: Path, baseline_tr: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    preds = pd.read_parquet(preds_path)
    preds["entry_ts"] = pd.to_datetime(preds["entry_ts"], utc=True, errors="coerce")

    tr_small = baseline_tr[["symbol","entry_ts","pnl_R"]].copy()
    df = preds.merge(tr_small, on=["symbol","entry_ts"], how="inner").dropna(subset=["pnl_R"])

    rows, evs = [], {}
    for t in np.round(np.linspace(0.50, 0.95, 10), 2):
        mask = df["y_proba"] >= t
        if mask.any():
            ev = float(df.loc[mask, "pnl_R"].mean())
            evs[float(t)] = ev
            rows.append({"threshold": float(t), "n": int(mask.sum()), "ev_R": ev})
    return evs, pd.DataFrame(rows)

def filter_and_dedup(preds_path: Path, signals_path: Path, pstar: float) -> Path:
    preds = pd.read_parquet(preds_path)
    preds["entry_ts"] = pd.to_datetime(preds["entry_ts"], utc=True, errors="coerce")

    keep = preds.loc[preds["y_proba"] >= float(pstar),
                     ["symbol","entry_ts"]].rename(columns={"entry_ts": "timestamp"})

    sig = pd.read_parquet(signals_path)
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")

    out = sig.merge(keep, on=["symbol","timestamp"], how="inner")

    # Dedup by busy window (to align with backtester lockouts)
    out = out.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    out = busy_window_dedup(out.rename(columns={"timestamp":"timestamp"}), minutes=BUSY_WINDOW_MIN)
    out_path = SIGNALS / f"signals_filtered_{int(pstar*100):02d}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path)
    return out_path

def main():
    signals_path = infer_signals_file()

    # Always (re)run a baseline backtest on THIS signals file to make EV apples-to-apples
    print(f"[baseline] backtesting on {signals_path.name} to rebuild results/trades.csv …")
    run_backtest(signals_path=signals_path)

    baseline_tr = pd.read_csv(Path(cfg.RESULTS_DIR) / "trades.csv", low_memory=False)
    baseline_tr = ensure_returns_cols(baseline_tr)

    outdir = Path(cfg.RESULTS_DIR) / "meta"
    outdir.mkdir(parents=True, exist_ok=True)

    # Train meta on the same data we just backtested
    mm.run_meta(
        trades_csv=str(Path(cfg.RESULTS_DIR) / "trades.csv"),
        signals_parquet=str(signals_path),
        returns_col="pnl_R",
        r_threshold=0.0,
        blocks=12, k_test=3, embargo=1, max_splits=25,
        outdir=str(outdir),
    )

    preds_path = outdir / "oos_predictions.parquet"
    evs, curve = ev_curve(preds_path, baseline_tr)
    print("EV by threshold:", {k: round(v, 6) for k, v in evs.items()})
    if not evs:
        print("No EV points computed; aborting.")
        return

    # Try thresholds in descending order; require enough signals AFTER dedup
    chosen = None
    for t in CANDIDATE_THRESHOLDS:
        ev = evs.get(t)
        if ev is None: continue
        fpath = filter_and_dedup(preds_path, signals_path, t)
        kept = len(pd.read_parquet(fpath))
        print(f"p*={t:.2f} → kept after dedup: {kept}")
        if kept >= MIN_KEPT_AFTER_DEDUP:
            chosen = (t, ev, fpath, kept)
            break
        chosen = chosen or (t, ev, fpath, kept)  # keep the best we’ve seen even if small

    if chosen is None:
        t = max(evs, key=evs.get); ev = evs[t]
        fpath = filter_and_dedup(preds_path, signals_path, t)
        kept  = len(pd.read_parquet(fpath))
        chosen = (t, ev, fpath, kept)

    best_t, best_ev, filt_path, kept = chosen
    print(f"Chosen p* = {best_t:.2f}  (EV={best_ev:.4f}, kept={kept}) → {filt_path}")

    # Save curve & JSON
    curve.to_csv(outdir / "ev_curve.csv", index=False)
    (outdir / "ev_by_threshold.json").write_text(json.dumps({str(k): v for k, v in evs.items()}, indent=2))

    # Re-backtest on filtered+deduped set
    run_backtest(signals_path=filt_path)

if __name__ == "__main__":
    main()
