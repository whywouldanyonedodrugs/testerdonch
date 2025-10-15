# sweep_donch_params.py
"""
Grid-search Donch strategy knobs (Donchian length, volume filter, TP/SL, etc.)
with:
- Safe handling of empty grids (no crash).
- --single mode to run exactly one configuration.
- Caching of scout signals per (Donch/volume/pullback/entry) to avoid recomputing
  when only exits change.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from scout import run_scout
from backtester import run_backtest
import reporting  # for Sharpe/DSR/PBO helpers

ROOT = Path(__file__).resolve().parent
SIG_DIR = Path(cfg.SIGNALS_DIR)
RES_DIR = Path(cfg.RESULTS_DIR)
VAR_DIR = RES_DIR / "variants"
VAR_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- helpers -----------------

def _ensure_returns_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pnl" not in df.columns and {"entry","exit","qty"}.issubset(df.columns):
        df["pnl"] = (df["exit"] - df["entry"]) * df["qty"]
    if "pnl_R" not in df.columns and {"entry","exit","sl"}.issubset(df.columns):
        rpu = (df["entry"] - df["sl"]).replace(0, np.nan)
        df["pnl_R"] = (df["exit"] - df["entry"]) / rpu
    if "pnl" in df.columns:
        df["is_win"] = (df["pnl"] > 0).astype(int)
    return df

def _slug(d: Dict[str, object]) -> str:
    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            v = f"{v}".rstrip("0").rstrip(".")
        parts.append(f"{k}-{v}")
    return "_".join(parts)

@dataclass
class Variant:
    # scout knobs
    DON_UP_N: int
    DON_CONFIRM_CLOSE_ABOVE: bool
    VOL_SPIKE_ENABLED: bool
    VOL_SPIKE_MULTIPLIER: float
    PULLBACK_MODEL: str      # "retest" | "mean"
    ENTRY_RULE: str          # "rebreak_high" | "close_above_break"
    # exits
    SL_ATR_MULT: float
    TP_ATR_MULT: float
    TIME_EXIT_HOURS: float | None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

# For caching scout results: key that ignores exit parameters
ScoutKey = Tuple[int, bool, bool, float, str, str]

def _scout_key(v: Variant) -> ScoutKey:
    return (v.DON_UP_N, v.DON_CONFIRM_CLOSE_ABOVE, v.VOL_SPIKE_ENABLED,
            v.VOL_SPIKE_MULTIPLIER, v.PULLBACK_MODEL, v.ENTRY_RULE)

def _apply_scout_config_from_key(sk: ScoutKey) -> None:
    dn, cc, ve, vm, pb, er = sk
    cfg.DON_UP_N = int(dn)
    cfg.DON_CONFIRM_CLOSE_ABOVE = bool(cc)
    cfg.VOL_SPIKE_ENABLED = bool(ve)
    cfg.VOL_SPIKE_MULTIPLIER = float(vm)
    cfg.PULLBACK_MODEL = str(pb)
    cfg.ENTRY_RULE = str(er)

def _apply_exit_config(v: Variant) -> None:
    cfg.SL_ATR_MULT = float(v.SL_ATR_MULT)
    cfg.TP_ATR_MULT = float(v.TP_ATR_MULT)
    cfg.TIME_EXIT_HOURS = None if v.TIME_EXIT_HOURS is None else float(v.TIME_EXIT_HOURS)

def _kpis(trades_csv: Path) -> Dict[str, float]:
    if not trades_csv.exists():
        return dict(n=0, win_rate=np.nan, profit_factor=np.nan, mean_R=np.nan, net_pnl=np.nan)
    df = pd.read_csv(trades_csv)
    df = _ensure_returns_cols(df)
    n = len(df)
    if n == 0:
        return dict(n=0, win_rate=np.nan, profit_factor=np.nan, mean_R=np.nan, net_pnl=np.nan)
    x = df["pnl"].astype(float).values if "pnl" in df.columns else np.array([])
    gp = x[x > 0].sum() if x.size else 0.0
    gl = -x[x <= 0].sum() if x.size else 0.0
    pf = (gp / gl) if gl > 0 else np.nan
    wr = (x > 0).mean() if x.size else np.nan
    return dict(n=int(n), win_rate=float(wr), profit_factor=float(pf),
                mean_R=float(df["pnl_R"].mean()) if "pnl_R" in df.columns else np.nan,
                net_pnl=float(df["pnl"].sum()) if "pnl" in df.columns else np.nan)

# ----------------- sweep core -----------------

def sweep(variants: List[Variant]) -> Path:
    if len(variants) == 0:
        # write empty outputs and stop cleanly
        empty_lb = pd.DataFrame(columns=["variant","n","win_rate","profit_factor","mean_R"])
        empty_lb.to_csv(VAR_DIR / "leaderboard.csv", index=False)
        pd.DataFrame().to_csv(VAR_DIR / "trades_all.csv", index=False)
        print("No variants to run. Wrote empty leaderboard/trades_all. Exiting.")
        return VAR_DIR / "trades_all.csv"

    # Cache scout signals per unique scout config
    unique_scout: Dict[ScoutKey, Path] = {}
    leaderboard = []
    combined_rows = []

    # 1) prepare unique scout configs
    scout_keys = sorted({ _scout_key(v) for v in variants })
    print(f"Unique scout configs: {len(scout_keys)}")

    # 2) run scout once per key
    for i, sk in enumerate(scout_keys, 1):
        _apply_scout_config_from_key(sk)
        tag = _slug(dict(
            DON_UP_N=sk[0], DON_CONFIRM_CLOSE_ABOVE=sk[1],
            VOL_SPIKE_ENABLED=sk[2], VOL_SPIKE_MULTIPLIER=sk[3],
            PULLBACK_MODEL=sk[4], ENTRY_RULE=sk[5]
        ))
        sig_path = SIG_DIR / f"signals_{tag}.parquet"
        if sig_path.exists():
            print(f"[{i:02d}/{len(scout_keys)}] Using cached signals: {sig_path}")
        else:
            print(f"[{i:02d}/{len(scout_keys)}] Scouting: {tag}")
            sig = run_scout()
            sig.to_parquet(sig_path, index=False)
            print(f"Signals: {len(sig)} → {sig_path}")
        unique_scout[sk] = sig_path

    # 3) for each variant: apply exits, reuse signals, run backtest
    for j, v in enumerate(variants, 1):
        sk = _scout_key(v)
        sig_path = unique_scout[sk]
        _apply_exit_config(v)
        vtag = _slug(v.as_dict())
        print(f"\n[{j:03d}/{len(variants)}] Backtesting exits for variant: {vtag}")
        try:
            run_backtest(signals_path=sig_path)
            t = pd.read_csv(RES_DIR / "trades.csv", parse_dates=["entry_ts","exit_ts"])
        except Exception as e:
            print(f"[error] backtest failed: {e}")
            t = pd.DataFrame()

        t = _ensure_returns_cols(t)
        for k, val in v.as_dict().items():
            t[k] = val
        t["variant"] = vtag

        (VAR_DIR / f"trades_{vtag}.csv").write_text(t.to_csv(index=False))
        leaderboard.append(dict(variant=vtag, **_kpis(RES_DIR / "trades.csv")))
        combined_rows.append(t)

    # 4) combine + leaderboard
    all_trades = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    all_path = VAR_DIR / "trades_all.csv"
    all_trades.to_csv(all_path, index=False)

    lb = pd.DataFrame(leaderboard)
    if not lb.empty:
        lb = lb.sort_values(["profit_factor","win_rate"], ascending=[False, False])
    else:
        lb = pd.DataFrame(columns=["variant","n","win_rate","profit_factor","mean_R"])
    lb_path = VAR_DIR / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)

    print(f"\nCombined trades saved → {all_path}")
    print(f"Leaderboard saved     → {lb_path}")

    # Optional: daily Sharpe per variant
    try:
        if not all_trades.empty:
            ret_mat = reporting.trades_to_daily_matrix(all_trades, returns_col="pnl_R", variant_cols=["variant"])
            sharpe = ret_mat.apply(lambda s: reporting.sharpe_unannualized(s.values), axis=0).sort_values(ascending=False)
            sharpe_path = VAR_DIR / "sharpe_by_variant.csv"
            sharpe.reset_index().rename(columns={"index":"variant",0:"sharpe"}).to_csv(sharpe_path, index=False)
            print(f"Daily Sharpe (unannualized) saved → {sharpe_path}")
    except Exception as e:
        print(f"[warn] could not compute daily Sharpe matrix: {e}")

    return all_path

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="Run exactly one configuration (use flags below).")
    ap.add_argument("--don", type=int, help="Donchian N (e.g., 40)")
    ap.add_argument("--close-above", type=str, help="Require close above break? true/false")
    ap.add_argument("--vol-enabled", type=str, help="Volume spike enabled? true/false")
    ap.add_argument("--vol-mult", type=float, help="Volume spike multiple (e.g., 2.0)")
    ap.add_argument("--pb", type=str, choices=["retest","mean"], help="Pullback model")
    ap.add_argument("--entry", type=str, choices=["rebreak_high","close_above_break"], help="Entry rule")
    ap.add_argument("--sl", type=float, help="SL ATR multiple")
    ap.add_argument("--tp", type=float, help="TP ATR multiple")
    ap.add_argument("--time-exit", type=float, help="Time exit in hours; omit for None")
    args = ap.parse_args()

    variants: List[Variant] = []

    if args.single:
        if None in (args.don, args.close_above, args.vol_enabled, args.vol_mult, args.pb, args.entry, args.sl, args.tp):
            raise SystemExit("In --single mode you must supply --don --close-above --vol-enabled --vol-mult --pb --entry --sl --tp [--time-exit]")
        def parse_bool(s: str) -> bool:
            return str(s).strip().lower() in {"1","true","t","yes","y"}
        v = Variant(
            DON_UP_N=int(args.don),
            DON_CONFIRM_CLOSE_ABOVE=parse_bool(args.close_above),
            VOL_SPIKE_ENABLED=parse_bool(args.vol_enabled),
            VOL_SPIKE_MULTIPLIER=float(args.vol_mult),
            PULLBACK_MODEL=args.pb,
            ENTRY_RULE=args.entry,
            SL_ATR_MULT=float(args.sl),
            TP_ATR_MULT=float(args.tp),
            TIME_EXIT_HOURS=(float(args.time_exit) if args.time_exit is not None else None),
        )
        variants = [v]
        print("Running one variant:", _slug(v.as_dict()))
    else:
        # Default grid (edit freely)
        don_n = [20, 40, 55]
        confirm_close = [True, False]
        vol_enabled = [True, False]
        vol_mult = [1.5, 2.0, 3.0]
        pb_model = ["retest", "mean"]
        entry_rule = ["rebreak_high", "close_above_break"]
        sl_mult = [1.0, 1.5, 2.0]
        tp_mult = [1.0, 1.5, 2.0]
        time_exit = [None, 4.0, 6.0, 8.0]  # add None to disable

        for dn, cc in itertools.product(don_n, confirm_close):
            for ve, vm in itertools.product(vol_enabled, vol_mult):
                if not ve and vm != vol_mult[0]:
                    continue  # only keep first multiplier if volume disabled
                for pb in pb_model:
                    for er in entry_rule:
                        for sl in sl_mult:
                            for tp in tp_mult:
                                for tx in time_exit:
                                    if dn == 20 and sl == 1.0 and tp == 1.0:
                                        continue
                                    variants.append(Variant(
                                        DON_UP_N=dn, DON_CONFIRM_CLOSE_ABOVE=cc,
                                        VOL_SPIKE_ENABLED=ve, VOL_SPIKE_MULTIPLIER=vm,
                                        PULLBACK_MODEL=pb, ENTRY_RULE=er,
                                        SL_ATR_MULT=sl, TP_ATR_MULT=tp, TIME_EXIT_HOURS=tx,
                                    ))

    print(f"Total variants: {len(variants)}")
    sweep(variants)

if __name__ == "__main__":
    main()
