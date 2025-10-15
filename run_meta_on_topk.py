# run_meta_on_topk.py
"""
Read results/leaderboard_guarded.csv, take top-K variants by PF then win%,
for each: set config knobs from tag, regenerate (or reuse) signals, run baseline backtest,
train meta-model, pick p* by EV, filter signals, re-backtest, and archive outputs
under variant-specific subfolders.
"""
from __future__ import annotations
import sys, json, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

import config as cfg
from scout import run_scout
from backtester import run_backtest
import meta_model as mm

RESULTS = Path(cfg.RESULTS_DIR); RESULTS.mkdir(parents=True, exist_ok=True)
SIGNALS = Path(cfg.SIGNALS_DIR); SIGNALS.mkdir(parents=True, exist_ok=True)

# ---------------- tag parsing ----------------
def parse_tag(tag: str):
    """
    Example: DON_DAYS-20_VOL-multiple3.0_ENTRY-rebreak_high_PB-retest_SL-3.0_TP-3.0_TE-8.0
    Returns dict with:
      don_days, vol_mode ('multiple'/'quantile'), vol_value (float),
      entry_rule ('rebreak_high'/'close_above_break'),
      pb_model ('retest'/'mean'),
      sl, tp, te (floats)
    """
    m = re.match(r'^DON_DAYS-(?P<dd>\\d+)_VOL-(?P<vol>[^_]+)_ENTRY-(?P<entry>[a-z_]+)_PB-(?P<pb>[a-z_]+)_SL-(?P<sl>[0-9.]+)_TP-(?P<tp>[0-9.]+)_TE-(?P<te>[0-9.]+)$', tag)
    if not m:
        raise ValueError(f"Unrecognized tag: {tag}")
    dd = int(m.group('dd'))
    vol_str = m.group('vol')
    if vol_str.startswith('multiple'):
        vol_mode = 'multiple'
        vol_value = float(vol_str.replace('multiple',''))
    elif vol_str.startswith('quantile'):
        vol_mode = 'quantile'
        vol_value = float(vol_str.replace('quantile',''))
    else:
        raise ValueError(f"Unknown VOL segment: {vol_str}")
    entry = m.group('entry')
    pb    = m.group('pb')
    sl = float(m.group('sl')); tp = float(m.group('tp')); te = float(m.group('te'))
    return dict(don_days=dd, vol_mode=vol_mode, vol_value=vol_value,
                entry_rule=entry, pb_model=pb, sl=sl, tp=tp, te=te)

# ---------------- config setters ----------------
def set_entry_cfg_from_parsed(p):
    cfg.DONCH_BASIS = "days"
    cfg.DON_N_DAYS  = int(p['don_days'])
    cfg.VOL_SPIKE_ENABLED = True
    cfg.VOL_SPIKE_MODE = p['vol_mode']
    if p['vol_mode'] == 'multiple':
        cfg.VOL_MULTIPLE = float(p['vol_value'])
    else:
        cfg.VOL_QUANTILE_Q = float(p['vol_value'])
    if not hasattr(cfg, 'VOL_LOOKBACK_DAYS'): cfg.VOL_LOOKBACK_DAYS = 30
    cfg.PULLBACK_MODEL = p['pb_model']
    cfg.ENTRY_RULE     = p['entry_rule']
    cfg.DON_CONFIRM_CLOSE_ABOVE = (p['entry_rule'] == 'close_above_break')

def set_exit_cfg_from_parsed(p):
    cfg.SL_ATR_MULT = float(p['sl'])
    cfg.TP_ATR_MULT = float(p['tp'])
    cfg.TIME_EXIT_HOURS = float(p['te'])

# ---------------- signals hashing ----------------
def _hash_entry_cfg() -> str:
    payload = dict(
        DONCH_BASIS=cfg.DONCH_BASIS,
        DON_N_DAYS=int(cfg.DON_N_DAYS),
        VOL_SPIKE_MODE=cfg.VOL_SPIKE_MODE,
        VOL_MULTIPLE=float(getattr(cfg, "VOL_MULTIPLE", 3.0)),
        VOL_QUANTILE_Q=float(getattr(cfg, "VOL_QUANTILE_Q", 0.95)),
        VOL_LOOKBACK_DAYS=int(getattr(cfg, "VOL_LOOKBACK_DAYS", 30)),
        PULLBACK_MODEL=cfg.PULLBACK_MODEL,
        ENTRY_RULE=cfg.ENTRY_RULE,
        DON_CONFIRM_CLOSE_ABOVE=bool(cfg.DON_CONFIRM_CLOSE_ABOVE)),
        START=str(cfg.START_DATE), END=str(cfg.END_DATE),
        ATR_TIMEFRAME=str(getattr(cfg, "ATR_TIMEFRAME", "1h")),
        REGIME_TIMEFRAME=str(getattr(cfg, "REGIME_TIMEFRAME", "4h")),
    )
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def ensure_signals() -> Path:
    h = _hash_entry_cfg()
    out_path = SIGNALS / f"signals_daily_{h}.parquet"
    if out_path.exists():
        return out_path
    sig = run_scout()
    sig.to_parquet(out_path, index=False)
    print(f"[scout] wrote {len(sig)} signals → {out_path}")
    return out_path

# ---------------- EV & filtering ----------------
def ev_curve(preds_path: Path, trades_csv: Path) -> tuple[dict, pd.DataFrame]:
    preds = pd.read_parquet(preds_path)
    trades = pd.read_csv(trades_csv, parse_dates=["entry_ts"])
    preds["entry_ts"]  = pd.to_datetime(preds["entry_ts"],  utc=True, errors="coerce")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    df = preds.merge(trades[["symbol","entry_ts","pnl_R"]], on=["symbol","entry_ts"], how="inner").dropna(subset=["pnl_R"])
    thresholds = np.round(np.linspace(0.50, 0.95, 10), 2)
    evs = {}
    rows = []
    for t in thresholds:
        mask = df["y_proba"] >= t
        if mask.any():
            ev = float(df.loc[mask, "pnl_R"].mean())
            evs[float(t)] = ev
            rows.append({"threshold": float(t), "n": int(mask.sum()), "ev_R": ev})
    curve = pd.DataFrame(rows)
    return evs, curve

def filter_signals(preds_path: Path, signals_path: Path, pstar: float, tag_key: str) -> Path:
    preds = pd.read_parquet(preds_path)
    keep = preds.loc[preds["y_proba"] >= float(pstar), ["symbol","entry_ts"]].rename(columns={"entry_ts":"timestamp"})
    sig = pd.read_parquet(signals_path)
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
    out = sig.merge(keep, on=["symbol","timestamp"], how="inner")
    out_path = SIGNALS / f"signals_filtered_{tag_key}_{int(pstar*100):02d}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path)
    return out_path

def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)[:120]

def archive(path: Path, suffix: str) -> Path | None:
    if not path.exists(): return None
    dest = path.with_name(path.stem + f"__{suffix}" + path.suffix)
    try:
        dest.write_bytes(path.read_bytes())
        return dest
    except Exception:
        return None

# ---------------- main ----------------
def main(k: int):
    lb_path = RESULTS / "leaderboard_guarded.csv"
    if not lb_path.exists():
        print("leaderboard_guarded.csv not found. Run the sweep first.")
        sys.exit(1)
    lb = pd.read_csv(lb_path)
    if lb.empty:
        print("Leaderboard is empty."); sys.exit(0)
    lb = lb.sort_values(["profit_factor","win_rate"], ascending=[False, False]).head(k).reset_index(drop=True)

    for _, row in lb.iterrows():
        tag = str(row["tag"])
        print("="*80)
        print("Running meta for:", tag)
        params = parse_tag(tag)

        # Set config
        set_entry_cfg_from_parsed(params)
        set_exit_cfg_from_parsed(params)

        # Ensure signals & run baseline backtest to generate trades.csv
        sig_path = ensure_signals()
        run_backtest(signals_path=sig_path)
        baseline_csv = Path(cfg.RESULTS_DIR) / "trades.csv"

        # Train meta-model into a per-variant folder
        tag_key = sanitize(tag)
        outdir = RESULTS / "meta" / tag_key
        outdir.mkdir(parents=True, exist_ok=True)
        mm.run_meta(
            trades_csv=str(baseline_csv),
            signals_parquet=str(sig_path),
            returns_col="pnl_R",
            r_threshold=0.0,
            blocks=12, k_test=3, embargo=1, max_splits=25,
            outdir=str(outdir),
        )

        preds_path = outdir / "oos_predictions.parquet"
        evs, curve = ev_curve(preds_path, baseline_csv)
        if not evs:
            print("No EV points (preds did not merge with trades) — skipping filtering.")
            continue
        best_t = max(evs, key=evs.get)
        print("EV by threshold:", {k: round(v,6) for k,v in evs.items()})
        print(f"Chosen p* = {best_t:.2f} (EV={evs[best_t]:.4f})")
        curve.to_csv(outdir / "ev_curve.csv", index=False)
        (outdir / "ev_by_threshold.json").write_text(json.dumps({str(k): v for k,v in evs.items()}, indent=2))

        # Filter signals at p* and re-backtest
        filt_path = filter_signals(preds_path, sig_path, best_t, tag_key)
        kept = len(pd.read_parquet(filt_path))
        print(f"Kept {kept} signals @ p*={best_t:.2f} → {filt_path}")
        run_backtest(signals_path=filt_path)

        # Archive outputs for this variant
        archive(Path(cfg.RESULTS_DIR) / "trades.csv", f"{tag_key}__baseline")
        archive(Path(cfg.RESULTS_DIR) / "trades.csv", f"{tag_key}__filtered_{int(best_t*100)}")

if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(k)
