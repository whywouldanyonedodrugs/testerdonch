#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _try_import_cfg(repo_root: Path):
    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import config as cfg  # type: ignore
        return cfg
    except Exception:
        return None


def _load_btc_vol_hi(repo_root: Path, meta_model_dir: Optional[Path]) -> float:
    default = 1.0
    cfg = _try_import_cfg(repo_root)
    if meta_model_dir is None:
        if cfg is not None and hasattr(cfg, "META_MODEL_DIR"):
            try:
                meta_model_dir = Path(getattr(cfg, "META_MODEL_DIR")).resolve()
            except Exception:
                meta_model_dir = None

    if meta_model_dir is None:
        # common default
        meta_model_dir = (repo_root / "results" / "meta_export").resolve()

    p = meta_model_dir / "regimes_report.json"
    if not p.exists():
        # fallback to config BTC_VOL_HI
        if cfg is not None and hasattr(cfg, "BTC_VOL_HI"):
            try:
                return float(getattr(cfg, "BTC_VOL_HI"))
            except Exception:
                return default
        return default

    rep = json.loads(p.read_text(encoding="utf-8"))
    thr = rep.get("thresholds") or {}
    try:
        return float(thr.get("btc_vol_hi", default))
    except Exception:
        return default


def _find_file(run_dir: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = run_dir / n
        if p.exists():
            return p
    # fallback glob
    for pat in names:
        hits = sorted(run_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _compute_drawdown_from_equity(eq: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    eq = eq.copy()
    eq["timestamp"] = _parse_ts(eq["timestamp"])
    eq = eq.dropna(subset=["timestamp"])
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["equity"]).sort_values("timestamp")
    if eq.empty:
        return float("nan"), float("nan"), eq

    peak = -np.inf
    peak_time = eq["timestamp"].iloc[0]
    max_dd = 0.0
    max_dd_dur_s = 0.0

    dds = []
    for ts, e in zip(eq["timestamp"].to_list(), eq["equity"].to_list()):
        if e >= peak:
            peak = float(e)
            peak_time = ts
        dd = (float(e) / peak - 1.0) if peak != 0 else 0.0
        dds.append(dd)
        if dd < max_dd:
            max_dd = dd
        if dd < 0:
            dur = (ts - peak_time).total_seconds()
            if dur > max_dd_dur_s:
                max_dd_dur_s = dur

    eq["drawdown"] = dds
    return float(max_dd), float(max_dd_dur_s), eq


def _risk_on_mask(tr: pd.DataFrame, btc_vol_hi: float) -> pd.Series:
    # regime_up is in Trade dataclass -> trades.csv
    ru = tr.get("regime_up")
    if ru is None:
        # if missing, treat as all False (conservative)
        return pd.Series([False] * len(tr), index=tr.index)

    if ru.dtype == bool:
        ru_i = ru.astype(int)
    else:
        ru_i = pd.to_numeric(ru, errors="coerce").fillna(0).astype(int)

    # prefer btc_* fields (Trade has btc_trend_slope, btc_vol_regime_level)
    bts = tr["btc_trend_slope"] if "btc_trend_slope" in tr.columns else tr.get("btcusdt_trend_slope")
    bvl = tr["btc_vol_regime_level"] if "btc_vol_regime_level" in tr.columns else tr.get("btcusdt_vol_regime_level")

    bts = pd.to_numeric(bts, errors="coerce") if bts is not None else pd.Series(np.nan, index=tr.index)
    bvl = pd.to_numeric(bvl, errors="coerce") if bvl is not None else pd.Series(np.nan, index=tr.index)

    btc_trend_up = (bts > 0.0)
    btc_vol_high = (bvl >= float(btc_vol_hi))

    # risk_on = 1[(regime_up==1) & (btc_trend_up==1) & (btc_vol_high==0)]
    return (ru_i == 1) & btc_trend_up & (~btc_vol_high)


def _worst_loss_streak(pnl: np.ndarray) -> Tuple[int, float]:
    # consecutive negative pnl streak (by trade order)
    worst_sum = 0.0
    cur_sum = 0.0
    worst_len = 0
    cur_len = 0
    for x in pnl:
        if np.isfinite(x) and x < 0:
            cur_sum += float(x)
            cur_len += 1
            if cur_sum < worst_sum:
                worst_sum = cur_sum
                worst_len = cur_len
        else:
            cur_sum = 0.0
            cur_len = 0
    return int(worst_len), float(worst_sum)


def recompute_one(run_dir: Path, btc_vol_hi: float, use_col: str) -> Dict[str, Any]:
    trades_path = _find_file(run_dir, ["trades.csv", "trades*.csv", "trades.parquet"])
    if trades_path is None:
        return {"setting": run_dir.name, "status": "missing_trades"}

    if trades_path.suffix == ".parquet":
        tr = pd.read_parquet(trades_path)
    else:
        tr = pd.read_csv(trades_path, low_memory=False)

    # timestamps
    if "entry_ts" in tr.columns:
        tr["entry_ts"] = _parse_ts(tr["entry_ts"])
    if "exit_ts" in tr.columns:
        tr["exit_ts"] = _parse_ts(tr["exit_ts"])

    # pnl selection
    pnl_col = None
    if use_col in tr.columns:
        pnl_col = use_col
    else:
        # fallbacks
        for c in ["pnl_R", "pnl"]:
            if c in tr.columns:
                pnl_col = c
                break
    if pnl_col is None:
        return {"setting": run_dir.name, "status": "missing_pnl_col", "trades_file": str(trades_path)}

    pnl_s = pd.to_numeric(tr[pnl_col], errors="coerce")
    pnl = pnl_s.to_numpy()
    n = int(np.isfinite(pnl).sum())

    # risk_on mask from trade snapshot fields
    risk_on = _risk_on_mask(tr, btc_vol_hi)
    risk_off = ~risk_on

    pnl_on = float(np.nansum(pnl[risk_on.to_numpy()]))
    pnl_off = float(np.nansum(pnl[risk_off.to_numpy()]))
    bad_damage = float(np.nansum(pnl[(risk_off.to_numpy()) & (pnl < 0)]))

    # equity / drawdown
    eq_path = _find_file(run_dir, ["equity.csv", "equity*.csv", "equity.parquet"])
    max_dd = float("nan")
    max_dd_dur_s = float("nan")
    eq_out = None
    if eq_path is not None:
        if eq_path.suffix == ".parquet":
            eq = pd.read_parquet(eq_path)
        else:
            eq = pd.read_csv(eq_path)
        if set(["timestamp", "equity"]).issubset(eq.columns):
            max_dd, max_dd_dur_s, eq_out = _compute_drawdown_from_equity(eq)

    # fallback drawdown from pnl if no equity
    if not np.isfinite(max_dd):
        equity = np.nancumsum(np.where(np.isfinite(pnl), pnl, 0.0))
        peak = np.maximum.accumulate(equity) if equity.size else np.array([0.0])
        dd = np.where(peak != 0, equity / peak - 1.0, 0.0)
        max_dd = float(np.min(dd)) if dd.size else 0.0

    # exposure (approx; capped at 1)
    exposure = float("nan")
    exposure_on = float("nan")
    exposure_off = float("nan")
    if "entry_ts" in tr.columns and "exit_ts" in tr.columns:
        valid = tr["entry_ts"].notna() & tr["exit_ts"].notna()
        if valid.any():
            entry = tr.loc[valid, "entry_ts"]
            exit_ = tr.loc[valid, "exit_ts"]
            dur_s = (exit_ - entry).dt.total_seconds().clip(lower=0)
            total_s = (exit_.max() - entry.min()).total_seconds()
            if total_s and total_s > 0:
                exposure = float(min(1.0, dur_s.sum() / total_s))
                # splits
                ro = risk_on.loc[valid]
                exposure_on = float(min(1.0, dur_s[ro].sum() / total_s))
                exposure_off = float(min(1.0, dur_s[~ro].sum() / total_s))

    # basic stats
    num_trades = int(len(tr))
    win_rate = float(np.nanmean(pnl > 0)) if len(pnl) else float("nan")
    avg_trade = float(np.nanmean(pnl)) if len(pnl) else float("nan")
    med_trade = float(np.nanmedian(pnl)) if len(pnl) else float("nan")
    tail_p05 = float(np.nanquantile(pnl, 0.05)) if len(pnl) else float("nan")

    # worst risk-off loss streak
    pnl_off_seq = pnl[risk_off.to_numpy()]
    worst_len_off, worst_sum_off = _worst_loss_streak(pnl_off_seq)

    # write equity_curve.csv if we have eq_out
    if eq_out is not None and not eq_out.empty:
        eq_out[["timestamp", "equity", "drawdown"]].to_csv(run_dir / "equity_curve.csv", index=False)

    out = {
        "setting": run_dir.name,
        "status": "ok",
        "trades_file": str(trades_path),
        "pnl_col": pnl_col,
        "total_pnl": float(np.nansum(pnl)),
        "pnl_risk_on": pnl_on,
        "pnl_risk_off": pnl_off,
        "bad_regime_damage": bad_damage,
        "max_drawdown": float(max_dd),
        "max_dd_duration_seconds": float(max_dd_dur_s),
        "number_of_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "median_trade": med_trade,
        "tail_loss_p05": tail_p05,
        "exposure": exposure,
        "exposure_risk_on": exposure_on,
        "exposure_risk_off": exposure_off,
        "risk_on_trades": int(risk_on.sum()),
        "risk_off_trades": int(risk_off.sum()),
        "worst_loss_streak_len_risk_off": worst_len_off,
        "worst_loss_streak_sum_risk_off": worst_sum_off,
        "btc_vol_hi": float(btc_vol_hi),
    }
    return out


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if pd.isna(v):
                v = ""
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo-root", default="/opt/testerdonch")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--meta-model-dir", default="", help="optional; defaults to config.META_MODEL_DIR or results/meta_export")
    ap.add_argument("--use-pnl-col", default="pnl_R", help="prefer this column (pnl_R recommended), falls back to pnl")
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--topn", type=int, default=15)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sweep_root = (repo_root / args.results_dir / "policy_sweeps" / args.run_id).resolve()
    if not sweep_root.exists():
        raise SystemExit(f"Missing sweep root: {sweep_root}")

    meta_dir = Path(args.meta_model_dir).resolve() if args.meta_model_dir.strip() else None
    btc_vol_hi = _load_btc_vol_hi(repo_root, meta_dir)

    ignore = {"_scoped_signals", "_smoke_run", "_smoke_signals"}
    setting_dirs = [p for p in sweep_root.iterdir() if p.is_dir() and p.name not in ignore]

    rows: List[Dict[str, Any]] = []
    for d in sorted(setting_dirs):
        met = recompute_one(d, btc_vol_hi=btc_vol_hi, use_col=args.use_pnl_col)
        # utility
        if met.get("status") == "ok":
            pon = float(met.get("pnl_risk_on", 0.0))
            poff = float(met.get("pnl_risk_off", 0.0))
            mdd = float(met.get("max_drawdown", 0.0))
            util = pon - args.lam * abs(min(poff, 0.0)) - args.mu * abs(min(mdd, 0.0))
            met["utility"] = float(util)
        else:
            met["utility"] = float("nan")

        # write per-setting recomputed metrics
        (d / "metrics.recomputed.json").write_text(json.dumps(met, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(met)

    df = pd.DataFrame(rows)
    df.to_csv(sweep_root / "summary.recomputed.csv", index=False)

    df_ok = df[df["status"] == "ok"].copy()
    df_ok["utility"] = pd.to_numeric(df_ok["utility"], errors="coerce")
    df_ok = df_ok.sort_values("utility", ascending=False)

    top3 = df_ok.head(3)

    cols = ["setting","utility","total_pnl","pnl_risk_on","pnl_risk_off","bad_regime_damage","max_drawdown",
            "number_of_trades","win_rate","exposure","risk_on_trades","risk_off_trades","worst_loss_streak_sum_risk_off"]
    cols = [c for c in cols if c in df_ok.columns]
    show = df_ok[cols].head(args.topn)

    lines = []
    lines.append(f"# Policy sweep report (recomputed): {args.run_id}")
    lines.append("")
    lines.append(f"Using btc_vol_hi={btc_vol_hi:g} from regimes_report.json (or fallback).")
    lines.append(f"PnL column used: prefer {args.use_pnl_col}, fallback pnl.")
    lines.append("")
    lines.append(f"Utility = PnL_risk_on - {args.lam:g}*abs(min(PnL_risk_off,0)) - {args.mu:g}*abs(min(max_drawdown,0))")
    lines.append("")
    lines.append("## Top 3")
    for _, r in top3.iterrows():
        lines.append(
            f"- {r['setting']}: utility={r['utility']}, total={r.get('total_pnl')}, "
            f"on={r.get('pnl_risk_on')}, off={r.get('pnl_risk_off')}, "
            f"bad_damage={r.get('bad_regime_damage')}, max_dd={r.get('max_drawdown')}"
        )
    lines.append("")
    lines.append(f"## Ranked (top {args.topn})")
    lines.append("")
    lines.append(markdown_table(show))

    (sweep_root / "report.recomputed.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(sweep_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
