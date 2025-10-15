# sweep_donch_params_guarded.py
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
from typing import Optional, Tuple
import numpy as np, pandas as pd

import config as cfg
from scout import run_scout
from backtester import run_backtest

RESULTS = Path(cfg.RESULTS_DIR)
SIGNALS = Path(cfg.SIGNALS_DIR)
RESULTS.mkdir(parents=True, exist_ok=True)
SIGNALS.mkdir(parents=True, exist_ok=True)

# ------------------ helpers to set config ------------------

def set_entry_cfg(don_days: int,
                  vol_mode: str, vol_mult: float | None, vol_q: float | None,
                  pb_model: str, entry_rule: str, close_above: bool):
    cfg.DONCH_BASIS = "days"
    cfg.DON_N_DAYS  = int(don_days)

    cfg.VOL_SPIKE_ENABLED = True
    cfg.VOL_SPIKE_MODE    = vol_mode  # "multiple" or "quantile"
    cfg.VOL_MULTIPLE      = float(vol_mult) if vol_mult is not None else getattr(cfg, "VOL_MULTIPLE", 2.0)
    cfg.VOL_QUANTILE_Q    = float(vol_q) if vol_q is not None else getattr(cfg, "VOL_QUANTILE_Q", 0.95)
    if not hasattr(cfg, "VOL_LOOKBACK_DAYS"):
        cfg.VOL_LOOKBACK_DAYS = 30

    cfg.PULLBACK_MODEL = pb_model            # "retest" or "mean"
    cfg.ENTRY_RULE     = entry_rule          # "rebreak_high" or "close_above_break"
    cfg.DON_CONFIRM_CLOSE_ABOVE = bool(close_above)

def set_exit_cfg(sl_atr: float, tp_atr: float, te_hours: Optional[float]):
    cfg.SL_ATR_MULT = float(sl_atr)
    cfg.TP_ATR_MULT = float(tp_atr)
    cfg.TIME_EXIT_HOURS = float(te_hours) if te_hours is not None else None

def set_partial_trail_cfg(partial_enabled: bool,
                          partial_ratio: Optional[float],
                          tp1_atr_mult: Optional[float],
                          trail_after_tp1: bool,
                          trail_atr_mult: Optional[float]):
    # Partials
    cfg.PARTIAL_TP_ENABLED = bool(partial_enabled)
    if partial_enabled:
        cfg.PARTIAL_TP_RATIO = float(partial_ratio if partial_ratio is not None else getattr(cfg, "PARTIAL_TP_RATIO", 0.5))
        cfg.PARTIAL_TP1_ATR_MULT = float(tp1_atr_mult if tp1_atr_mult is not None else getattr(cfg, "PARTIAL_TP1_ATR_MULT", 5.0))
    # Trailing
    cfg.TRAIL_AFTER_TP1 = bool(trail_after_tp1)
    if trail_after_tp1:
        cfg.TRAIL_ATR_MULT = float(trail_atr_mult if trail_atr_mult is not None else getattr(cfg, "TRAIL_ATR_MULT", 1.0))

def _hash_entry_cfg() -> str:
    payload = dict(
        DONCH_BASIS=cfg.DONCH_BASIS,
        DON_N_DAYS=int(cfg.DON_N_DAYS),
        VOL_SPIKE_MODE=cfg.VOL_SPIKE_MODE,
        VOL_MULTIPLE=float(cfg.VOL_MULTIPLE),
        VOL_QUANTILE_Q=float(cfg.VOL_QUANTILE_Q),
        VOL_LOOKBACK_DAYS=int(getattr(cfg, "VOL_LOOKBACK_DAYS", 30)),
        PULLBACK_MODEL=cfg.PULLBACK_MODEL,
        ENTRY_RULE=cfg.ENTRY_RULE,
        DON_CONFIRM_CLOSE_ABOVE=bool(cfg.DON_CONFIRM_CLOSE_ABOVE),
        START=str(cfg.START_DATE),
        END=str(cfg.END_DATE),
        ATR_TIMEFRAME=str(getattr(cfg, "ATR_TIMEFRAME", "1h")),
        REGIME_TIMEFRAME=str(getattr(cfg, "REGIME_TIMEFRAME", "4h")),
    )
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def ensure_signals() -> Path:
    """Generate (or reuse) signals for the CURRENT entry config only."""
    h = _hash_entry_cfg()
    out_path = SIGNALS / f"signals_daily_{h}.parquet"
    if out_path.exists():
        return out_path
    print("[scout] generating signals for entry config…")
    sig = run_scout()
    sig.to_parquet(out_path, index=False)
    print(f"[scout] wrote {len(sig)} signals → {out_path}")
    return out_path

# ------------------ metrics & robustness proxies ------------------

def _safe_read_csv(p: Path, **kwargs) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, **kwargs)
    except Exception:
        return pd.DataFrame()

def metrics_from_trades_and_equity(trades_csv: Path, equity_csv: Path) -> dict:
    out = {"n": 0, "win_rate": np.nan, "profit_factor": np.nan, "avg_R": np.nan,
           "max_dd": np.nan, "cagr": np.nan, "mar": np.nan, "daily_sharpe": np.nan,
           "min_block_sharpe": np.nan}

    # ----- Trades-based metrics (R-aware if available) -----
    try:
        tdf = pd.read_csv(trades_csv)
    except Exception:
        tdf = pd.DataFrame()

    if not tdf.empty:
        if "pnl_R" in tdf.columns:
            r = pd.to_numeric(tdf["pnl_R"], errors="coerce").dropna()
            out["n"] = int(len(r))
            out["win_rate"] = float((r > 0).mean()) if len(r) else np.nan
            pos = r[r > 0].sum()
            neg = -r[r < 0].sum()
            out["profit_factor"] = float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else np.nan)
            out["avg_R"] = float(r.mean()) if len(r) else np.nan
        else:
            out["n"] = int(len(tdf))
            if {"entry", "exit"}.issubset(tdf.columns):
                out["win_rate"] = float((tdf["exit"] > tdf["entry"]).mean())

    # ----- Equity-based risk metrics -----
    try:
        edf = pd.read_csv(equity_csv, parse_dates=["timestamp"])
    except Exception:
        edf = pd.DataFrame()

    if not edf.empty and "equity" in edf.columns:
        edf = edf.dropna(subset=["timestamp", "equity"]).copy()
        edf["timestamp"] = pd.to_datetime(edf["timestamp"], utc=True, errors="coerce")
        edf = edf.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Deduplicate timestamps (keep last) to avoid asfreq() errors
        edf = edf[~edf["timestamp"].duplicated(keep="last")]
        edf = edf.set_index("timestamp")

        eq = pd.to_numeric(edf["equity"], errors="coerce").dropna()
        if len(eq) >= 3:
            # Max drawdown (from the full-resolution curve)
            peak = np.maximum.accumulate(eq.values)
            dd = (eq.values / peak - 1.0).min()  # negative
            max_dd = float(-dd)
            out["max_dd"] = max_dd

            # CAGR using first/last timestamps
            t0, t1 = eq.index[0], eq.index[-1]
            years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
            cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[0] > 0 else np.nan
            out["cagr"] = cagr
            out["mar"] = float(cagr / max_dd) if max_dd > 0 else np.nan

            # Daily returns via resample-last (robust to duplicates/irregular spacing)
            daily_eq = eq.resample("1D").last().ffill()
            drets = daily_eq.pct_change().dropna()
            if len(drets) >= 10:
                mu, sd = float(drets.mean()), float(drets.std(ddof=1))
                out["daily_sharpe"] = (mu / sd) * np.sqrt(365) if sd > 0 else np.nan

                # Min block Sharpe across 4 equal calendar blocks
                bvals = []
                for arr in np.array_split(drets.values, 4):
                    if len(arr) >= 5:
                        m = float(np.mean(arr)); s = float(np.std(arr, ddof=1))
                        bvals.append((m / s) * np.sqrt(365) if s > 0 else np.nan)
                if bvals:
                    out["min_block_sharpe"] = float(np.nanmin(bvals))

    return out

def _tag_entry(dd, vm, vmult, vq, er, pb, close_above) -> str:
    vstr = f"{vm}{vmult if vmult is not None else vq}"
    return f"DON_DAYS-{dd}_VOL-{vstr}_ENTRY-{er}_PB-{pb}_CONF-{int(bool(close_above))}"

def _tag_exit(sl, tp, te, p_en, p_ratio, tp1, tr_en, tr_mult) -> str:
    te_str = "None" if (te is None) else f"{float(te):.1f}"
    p_str  = f"PT-{int(bool(p_en))}"
    if p_en:
        p_str += f"@{float(p_ratio):.2f}_TP1x{float(tp1):.1f}"
    tr_str = f"TRL-{int(bool(tr_en))}"
    if tr_en:
        tr_str += f"x{float(tr_mult):.1f}"
    return f"SL-{sl:.1f}_TP-{tp:.1f}_TE-{te_str}_{p_str}_{tr_str}"

# ------------------ main sweep ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end",   type=str, default=None)
    ap.add_argument("--single", action="store_true",
                    help="run only the first variant for a smoke-test")
    ap.add_argument("--use-1m", action="store_true",
                    help="use 1m intrabar tie resolution in sweep (slower)")
    ap.add_argument("--robust-topk", type=int, default=0,
                    help="run reporting.py (PSR/DSR + PBO) on the top-K variants after the sweep")
    args = ap.parse_args()

    if args.start: cfg.START_DATE = args.start
    if args.end:   cfg.END_DATE   = args.end
    cfg.USE_INTRABAR_1M = bool(getattr(args, "use_1m", False))

    # ------ ENTRY GRID ------
    don_days    = [20]
    vol_cfgs    = [("multiple", 2.0, None)]
    pb_models   = ["retest"]
    entry_rules = ["close_above_break"]
    close_above = [True]

    # ------ EXIT GRID ------
    sls = [2.0, 2.5]
    tps = [8.0, 12.0]
    time_exits = [None, 72.0]

    partials = [
        (False, None, None)
    ]
    trailings = [
        (False, None)
    ]

    variants = []
    for dd in don_days:
        for vm, vmult, vq in vol_cfgs:
            for pb in pb_models:
                for er in entry_rules:
                    ent_tag = _tag_entry(dd, vm, vmult, vq, er, pb, close_above[0])
                    set_entry_cfg(dd, vm, vmult, vq, pb, er, close_above[0])
                    sig_path = ensure_signals()
                    if not sig_path.exists():
                        continue
                    for sl in sls:
                        for tp in tps:
                            for te in time_exits:
                                for (p_en, p_ratio, tp1) in partials:
                                    for (tr_en, tr_mult) in trailings:
                                        if tr_en and not p_en:
                                            continue
                                        exit_tag = _tag_exit(sl, tp, te, p_en, p_ratio, tp1, tr_en, tr_mult)
                                        tag = f"{ent_tag}_{exit_tag}"
                                        variants.append((tag, sig_path, sl, tp, te, p_en, p_ratio, tp1, tr_en, tr_mult))

    if args.single:
        variants = variants[:1]
        print(f"Running one variant: {variants[0][0]}")

    print(f"Total variants: {len(variants)}")

    leaderboard = []
    sweeps_dir = RESULTS / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    for (tag, sig_path, sl, tp, te, p_en, p_ratio, tp1, tr_en, tr_mult) in variants:
        set_exit_cfg(sl, tp, te)
        set_partial_trail_cfg(p_en, p_ratio, tp1, tr_en, tr_mult)

        # skip empty-signal files gracefully
        try:
            sig = pd.read_parquet(sig_path)
        except Exception:
            print(f"[run] {tag}\nCould not read signals file: {sig_path}\n")
            continue
        if sig.empty:
            print(f"[run] {tag}\nNo signals to backtest.\n")
            continue

        print(f"[run] {tag}")
        run_backtest(signals_path=sig_path)

        # Save per-variant artifacts
        vdir = sweeps_dir / tag
        vdir.mkdir(parents=True, exist_ok=True)
        trades_csv = RESULTS / "trades.csv"
        equity_csv = RESULTS / "equity.csv"
        if trades_csv.exists(): shutil.copy2(trades_csv, vdir / "trades.csv")
        if equity_csv.exists(): shutil.copy2(equity_csv, vdir / "equity.csv")

        # Collect metrics
        m = metrics_from_trades_and_equity(vdir / "trades.csv", vdir / "equity.csv")
        m.update({
            "tag": tag,
            "sl_atr": sl,
            "tp_atr": tp,
            "time_exit_h": te if te is not None else np.nan,
            "partial": int(bool(p_en)),
            "partial_ratio": p_ratio if p_ratio is not None else np.nan,
            "tp1_atr": tp1 if tp1 is not None else np.nan,
            "trail": int(bool(tr_en)),
            "trail_atr": tr_mult if tr_mult is not None else np.nan,
        })
        leaderboard.append(m)

    lb = pd.DataFrame(leaderboard)
    if lb.empty:
        print("No variants produced trades. Nothing to rank.")
        return

    # Sort by PF, then AvgR, then (lower) MaxDD, then MAR, then WinRate
    sort_cols = [("profit_factor", False), ("avg_R", False), ("max_dd", True), ("mar", False), ("win_rate", False)]
    lb = lb.sort_values([c for c,_ in sort_cols], ascending=[a for _,a in sort_cols])

    out = RESULTS / "leaderboard_guarded.csv"
    lb.to_csv(out, index=False)
    print(f"Saved leaderboard → {out}")
    with pd.option_context("display.max_colwidth", None):
        print(lb.head(15).to_string(index=False))

    # Optional: run reporting on top-K
    if args.robust_topk and args.robust_topk > 0:
        top = lb.head(int(args.robust_topk)).copy()
        print(f"\n[robust] Running reporting.py on top-{len(top)} variants…")
        for _, row in top.iterrows():
            tag = row["tag"]
            vdir = sweeps_dir / tag
            # Temporarily copy this variant's trades back to default location so reporting can see it
            src = vdir / "trades.csv"
            if not src.exists():
                print(f"[robust] skip {tag}: trades.csv missing")
                continue
            shutil.copy2(src, RESULTS / "trades.csv")
            try:
                subprocess.run(
                    ["python", "reporting.py", "--run-all", "--returns-col", "pnl_R",
                     "--variant-cols", "pullback_type", "entry_rule", "don_break_len", "regime_up"],
                    check=False
                )
                # Optionally, copy any reporting outputs/logs to the variant folder
                # (your reporting script mostly prints; if it saves files, add copies here)
            except Exception as e:
                print(f"[robust] reporting failed for {tag}: {e}")

if __name__ == "__main__":
    main()
