#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAS_PLT = True
except Exception:
    HAS_PLT = False


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JT-007: rolling walk-forward OOS validation pack with monthly stability reporting."
    )
    p.add_argument("--trades", default="results/trades.clean.csv")
    p.add_argument("--targets", default="", help="Optional precomputed targets parquet.")
    p.add_argument("--regimes", default="", help="Optional precomputed regimes parquet.")
    p.add_argument("--outdir", default="results/walkforward_oos")
    p.add_argument("--run-id", default="")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")

    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--valid-months", type=int, default=1)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--expanding-train", action="store_true")
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--min-train-trades", type=int, default=1000)
    p.add_argument("--min-valid-trades", type=int, default=100)

    p.add_argument("--target", default="y_good_05")
    p.add_argument("--train-scope", default="ALL", choices=["ALL", "RISK_ON_1"])
    p.add_argument("--fit-scope", default="ALL", choices=["ALL", "RISK_ON_1"])

    p.add_argument("--n-splits", type=int, default=6)
    p.add_argument("--embargo-days", type=float, default=1.0)
    p.add_argument("--min-eval-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-missing", type=float, default=0.95)
    p.add_argument("--min-unique-numeric", type=int, default=5)
    p.add_argument("--time-decay-halflife-days", type=float, default=0.0)
    p.add_argument("--live-safe-features", action="store_true", default=True)

    p.add_argument("--cal-methods", default="none,sigmoid,isotonic")
    p.add_argument("--cal-holdout-frac", type=float, default=0.30)
    p.add_argument("--cal-bins", type=int, default=20)
    p.add_argument("--cal-min-trades", type=int, default=200)
    p.add_argument("--cal-thresholds", type=int, default=51)
    p.add_argument("--cal-sizing-bins", type=int, default=8)
    p.add_argument("--cal-sizing-min-bin-n", type=int, default=25)
    p.add_argument("--cal-min-mult", type=float, default=0.01)
    p.add_argument("--cal-max-mult", type=float, default=1.0)

    p.add_argument("--python", default=sys.executable)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-window-artifacts", action="store_true")
    return p.parse_args()


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Dataclass/type introspection in loaded modules expects the module to exist in sys.modules.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p.resolve()


def _run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[wf] running: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return int(proc.wait())


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    start = _month_floor(ts)
    nxt = start + pd.offsets.MonthBegin(1)
    return nxt - pd.Timedelta(microseconds=1)


def _to_utc_ts(x: object) -> pd.Timestamp:
    t = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"Invalid timestamp: {x}")
    return pd.Timestamp(t)


def _build_windows(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    train_months: int,
    valid_months: int,
    step_months: int,
    expanding_train: bool,
    max_windows: int,
) -> List[WindowSpec]:
    if train_months < 1 or valid_months < 1 or step_months < 1:
        raise ValueError("train/valid/step months must be >= 1")

    start_m = _month_floor(_to_utc_ts(start_ts))
    end_m = _month_floor(_to_utc_ts(end_ts))

    windows: List[WindowSpec] = []
    val_start_m = _month_floor(start_m + pd.DateOffset(months=int(train_months)))
    idx = 0
    while True:
        val_end_m = _month_floor(val_start_m + pd.DateOffset(months=int(valid_months - 1)))
        if val_end_m > end_m:
            break

        if expanding_train:
            tr_start_m = start_m
        else:
            tr_start_m = _month_floor(val_start_m - pd.DateOffset(months=int(train_months)))
        tr_end_m = _month_floor(val_start_m - pd.DateOffset(months=1))

        train_start = max(start_ts, tr_start_m)
        train_end = min(end_ts, _month_end(tr_end_m))
        val_start = max(start_ts, val_start_m)
        val_end = min(end_ts, _month_end(val_end_m))

        wid = f"W{idx:03d}_{val_start.strftime('%Y%m')}"
        windows.append(
            WindowSpec(
                window_id=wid,
                train_start=_to_utc_ts(train_start),
                train_end=_to_utc_ts(train_end),
                val_start=_to_utc_ts(val_start),
                val_end=_to_utc_ts(val_end),
            )
        )
        idx += 1
        if max_windows > 0 and len(windows) >= max_windows:
            break

        val_start_m = _month_floor(val_start_m + pd.DateOffset(months=int(step_months)))

    return windows


def _subset_by_time(df: pd.DataFrame, ts_col: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    m = (df[ts_col] >= lo) & (df[ts_col] <= hi)
    return df.loc[m].copy()


def _pick_best_model(step05_summary: Dict[str, object]) -> Tuple[str, str, float]:
    models = step05_summary.get("models")
    if not isinstance(models, dict) or not models:
        raise RuntimeError("No models found in Step 05 summary.")

    best_key = None
    best_method = "none"
    best_ll = float("inf")
    best_brier = float("inf")
    for model_col, payload in models.items():
        if not isinstance(payload, dict):
            continue
        method = str(payload.get("chosen_method", "none"))
        mt = payload.get("methods_test_metrics")
        if not isinstance(mt, dict):
            continue
        row = mt.get(method, {})
        if not isinstance(row, dict):
            row = {}
        ll = float(row.get("logloss", np.nan))
        br = float(row.get("brier", np.nan))
        ll_key = ll if np.isfinite(ll) else float("inf")
        br_key = br if np.isfinite(br) else float("inf")
        if (ll_key, br_key) < (best_ll, best_brier):
            best_key = str(model_col)
            best_method = method
            best_ll = ll_key
            best_brier = br_key

    if best_key is None:
        # deterministic fallback
        best_key = str(sorted(models.keys())[0])
        payload = models[best_key]
        if isinstance(payload, dict):
            best_method = str(payload.get("chosen_method", "none"))
    return best_key, best_method, best_ll


def _predict_raw(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[:, 1]
        return np.asarray(p, dtype=np.float64)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(x), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-z))
    raise RuntimeError(f"Model {type(model).__name__} cannot produce probabilities.")


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=np.int32)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    out = 0.0
    n = len(y)
    if n <= 0:
        return float("nan")
    for b in range(len(bins) - 1):
        m = idx == b
        if not np.any(m):
            continue
        out += abs(float(np.mean(p[m])) - float(np.mean(y[m]))) * (float(np.sum(m)) / float(n))
    return float(out)


def _rank_lift_table(scored: pd.DataFrame, prob_col: str, y_col: str, pnl_col: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = scored[[prob_col, y_col, pnl_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_p", "win_rate", "mean_pnl_R"]), {
            "top_decile_win_lift": float("nan"),
            "top_decile_mean_pnl_lift": float("nan"),
            "spearman_prob_vs_pnl": float("nan"),
        }

    q = int(min(10, d[prob_col].nunique()))
    if q < 2:
        d["decile"] = 0
    else:
        d["decile"] = pd.qcut(d[prob_col], q=q, labels=False, duplicates="drop")

    tbl = (
        d.groupby("decile", observed=True)
        .agg(
            n=(prob_col, "size"),
            mean_p=(prob_col, "mean"),
            win_rate=(y_col, "mean"),
            mean_pnl_R=(pnl_col, "mean"),
        )
        .reset_index()
        .sort_values("decile")
        .reset_index(drop=True)
    )

    overall_wr = float(d[y_col].mean())
    overall_pnl = float(d[pnl_col].mean())
    top = tbl.tail(1)
    top_wr = float(top["win_rate"].iloc[0]) if not top.empty else float("nan")
    top_pnl = float(top["mean_pnl_R"].iloc[0]) if not top.empty else float("nan")
    spearman = float(d[prob_col].corr(d[pnl_col], method="spearman"))

    return tbl, {
        "top_decile_win_lift": (top_wr / overall_wr) if (np.isfinite(top_wr) and overall_wr > 0) else float("nan"),
        "top_decile_mean_pnl_lift": (top_pnl / overall_pnl) if (np.isfinite(top_pnl) and abs(overall_pnl) > 1e-12) else float("nan"),
        "spearman_prob_vs_pnl": spearman,
    }


def _monthly_metrics(scored: pd.DataFrame, y_col: str, pnl_col: str) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "n",
                "win_rate",
                "mean_p_cal",
                "calibration_abs_gap",
                "mean_pnl_R",
                "top_decile_win_lift",
            ]
        )
    d = scored.copy()
    ts = pd.to_datetime(d["entry_ts"], utc=True, errors="coerce")
    d["month"] = ts.dt.strftime("%Y-%m")
    base = (
        d.groupby("month", observed=True)
        .agg(
            n=("trade_id", "size"),
            win_rate=(y_col, "mean"),
            mean_p_cal=("p_cal", "mean"),
            mean_pnl_R=(pnl_col, "mean"),
        )
        .reset_index()
    )
    base["calibration_abs_gap"] = (base["mean_p_cal"] - base["win_rate"]).abs()

    lifts: List[Dict[str, object]] = []
    for month, sub in d.groupby("month", observed=True):
        _, lift = _rank_lift_table(sub, prob_col="p_cal", y_col=y_col, pnl_col=pnl_col)
        lifts.append({"month": str(month), "top_decile_win_lift": lift["top_decile_win_lift"]})
    lift_df = pd.DataFrame(lifts)
    if not lift_df.empty:
        base = base.merge(lift_df, on="month", how="left")
    else:
        base["top_decile_win_lift"] = np.nan
    return base.sort_values("month").reset_index(drop=True)


def _write_report_md(
    out_path: Path,
    run_meta: Dict[str, object],
    window_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# JT-007 Rolling Walk-Forward OOS Stability Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    for k, v in run_meta.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Window Summary")
    lines.append("")
    if window_df.empty:
        lines.append("No completed windows.")
    else:
        lines.append(window_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Monthly OOS Stability")
    lines.append("")
    if monthly_df.empty:
        lines.append("No monthly OOS rows.")
    else:
        lines.append(monthly_df.to_markdown(index=False))
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_series(df: pd.DataFrame, x: str, y: str, title: str, out_png: Path) -> None:
    if not HAS_PLT or df.empty or y not in df.columns:
        return
    d = df[[x, y]].dropna().copy()
    if d.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(d[x].astype(str), d[y].astype(float), marker="o")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def _write_report_html(
    out_path: Path,
    run_meta: Dict[str, object],
    window_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    assets_rel: str = "assets",
) -> None:
    def _tbl(df: pd.DataFrame, max_rows: int = 200) -> str:
        if df.empty:
            return "<p><em>no rows</em></p>"
        return df.head(max_rows).to_html(index=False, border=0)

    cfg_rows = "\n".join(
        [f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in run_meta.items()]
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>JT-007 Walk-Forward OOS Stability</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f4f4f4; text-align: left; }}
    h1, h2 {{ margin-top: 20px; }}
    .plots img {{ max-width: 100%; margin-bottom: 12px; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>JT-007 Rolling Walk-Forward OOS Stability Report</h1>
  <p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

  <h2>Run Configuration</h2>
  <table>{cfg_rows}</table>

  <h2>Window Summary</h2>
  {_tbl(window_df)}

  <h2>Monthly OOS Stability</h2>
  {_tbl(monthly_df)}

  <h2>Plots</h2>
  <div class="plots">
    <img src="{assets_rel}/window_top_decile_lift.png" alt="Window top decile lift">
    <img src="{assets_rel}/window_ece.png" alt="Window ECE">
    <img src="{assets_rel}/monthly_win_vs_prob.png" alt="Monthly win rate vs mean probability">
    <img src="{assets_rel}/monthly_mean_pnl.png" alt="Monthly mean pnl R">
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _ensure_targets_and_regimes(args: argparse.Namespace, run_dir: Path) -> Tuple[Path, Path]:
    trades = _resolve(args.trades)
    targets = _resolve(args.targets) if args.targets.strip() else run_dir / "_base" / "01_targets" / "targets.parquet"
    regimes = _resolve(args.regimes) if args.regimes.strip() else run_dir / "_base" / "02_regimes" / "regimes.parquet"

    if not targets.exists():
        outdir = targets.parent
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str((REPO_ROOT / "research" / "01_make_targets.py").resolve()),
            "--infile",
            str(trades),
            "--outdir",
            str(outdir),
            "--outfile",
            targets.name,
        ]
        rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=run_dir / "logs" / "01_make_targets.log")
        if rc != 0:
            raise RuntimeError(f"01_make_targets failed rc={rc}")

    if not regimes.exists():
        outdir = regimes.parent
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str((REPO_ROOT / "research" / "02_make_regimes.py").resolve()),
            "--infile",
            str(trades),
            "--outdir",
            str(outdir),
            "--outfile",
            regimes.name,
        ]
        rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=run_dir / "logs" / "02_make_regimes.log")
        if rc != 0:
            raise RuntimeError(f"02_make_regimes failed rc={rc}")

    return targets.resolve(), regimes.resolve()


def main() -> int:
    args = _parse_args()
    rid = args.run_id.strip() or f"jt007_walkforward_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_root = _resolve(args.outdir)
    run_dir = (out_root / rid).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "windows").mkdir(exist_ok=True)
    (run_dir / "aggregate").mkdir(exist_ok=True)

    step04 = _load_module("step04_models_cv", (REPO_ROOT / "research" / "04_models_cv.py").resolve())
    step05 = _load_module("step05_calibration_ev", (REPO_ROOT / "research" / "05_calibration_ev.py").resolve())

    trades_path = _resolve(args.trades)
    if not trades_path.exists():
        raise SystemExit(f"Missing trades file: {trades_path}")

    targets_path, regimes_path = _ensure_targets_and_regimes(args, run_dir)

    trades = step04._read_trades_csv(str(trades_path))
    targets = step04._read_targets_parquet(str(targets_path))
    regimes = step04._read_regimes_parquet(str(regimes_path))

    # Authoritative regime columns from regimes parquet.
    overlap = [c for c in regimes.columns if c in trades.columns]
    trades_join = trades.drop(columns=overlap) if overlap else trades.copy()
    joined = trades_join.join(targets, how="left").join(regimes, how="left")
    joined = step04._drop_duplicate_columns(joined, "joined_frame")
    joined = step04._coalesce_suffix_pairs(joined)
    joined = joined[joined["entry_ts"].notna()].copy()

    if args.target not in joined.columns:
        raise SystemExit(f"Target not found after merge: {args.target}")
    if "pnl_R" not in joined.columns:
        raise SystemExit("Missing pnl_R in trades data.")

    joined[args.target] = pd.to_numeric(joined[args.target], errors="coerce")
    joined["pnl_R"] = pd.to_numeric(joined["pnl_R"], errors="coerce")

    ts_min = _to_utc_ts(joined["entry_ts"].min())
    ts_max = _to_utc_ts(joined["entry_ts"].max())
    start_ts = _to_utc_ts(args.start) if args.start.strip() else ts_min
    end_ts = _to_utc_ts(args.end) if args.end.strip() else ts_max
    start_ts = max(start_ts, ts_min)
    end_ts = min(end_ts, ts_max)
    if end_ts <= start_ts:
        raise SystemExit("Invalid time range after clipping to trades data.")

    windows = _build_windows(
        start_ts=start_ts,
        end_ts=end_ts,
        train_months=int(args.train_months),
        valid_months=int(args.valid_months),
        step_months=int(args.step_months),
        expanding_train=bool(args.expanding_train),
        max_windows=int(args.max_windows),
    )
    if not windows:
        raise SystemExit("No rolling windows generated for selected period.")

    window_rows: List[Dict[str, object]] = []
    all_monthly_rows: List[pd.DataFrame] = []
    all_scored_parts: List[pd.DataFrame] = []

    for w in windows:
        wdir = run_dir / "windows" / w.window_id
        done_path = wdir / "_DONE.json"
        if args.resume and done_path.exists():
            print(f"[wf] resume: skipping {w.window_id}", flush=True)
            try:
                row = json.loads(done_path.read_text(encoding="utf-8"))
                if isinstance(row, dict):
                    window_rows.append(row)
            except Exception:
                pass
            # still gather aggregate inputs if available
            val_scored = wdir / "validation_scored.parquet"
            month_csv = wdir / "monthly_metrics.csv"
            if val_scored.exists():
                try:
                    all_scored_parts.append(pd.read_parquet(val_scored))
                except Exception:
                    pass
            if month_csv.exists():
                try:
                    all_monthly_rows.append(pd.read_csv(month_csv))
                except Exception:
                    pass
            continue

        wdir.mkdir(parents=True, exist_ok=True)
        print(
            f"[wf] window {w.window_id} train={w.train_start}..{w.train_end} val={w.val_start}..{w.val_end}",
            flush=True,
        )

        train_df = _subset_by_time(trades.reset_index(), "entry_ts", w.train_start, w.train_end)
        valid_df = _subset_by_time(joined.reset_index(), "entry_ts", w.val_start, w.val_end)
        train_n = int(len(train_df))
        valid_n = int(len(valid_df))

        if train_n < int(args.min_train_trades) or valid_n < int(args.min_valid_trades):
            row = {
                "window_id": w.window_id,
                "status": "skipped_small_sample",
                "train_start": str(w.train_start),
                "train_end": str(w.train_end),
                "val_start": str(w.val_start),
                "val_end": str(w.val_end),
                "train_n": train_n,
                "valid_n": valid_n,
            }
            done_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            window_rows.append(row)
            continue

        train_ids = set(pd.to_numeric(train_df["trade_id"], errors="coerce").dropna().astype(np.int64).tolist())
        train_targets = targets.loc[targets.index.intersection(train_ids)].reset_index()
        train_regimes = regimes.loc[regimes.index.intersection(train_ids)].reset_index()

        data_dir = wdir / "_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        train_trades_path = data_dir / "trades_train.csv"
        train_targets_path = data_dir / "targets_train.parquet"
        train_regimes_path = data_dir / "regimes_train.parquet"
        train_df.to_csv(train_trades_path, index=False)
        train_targets.to_parquet(train_targets_path, index=False)
        train_regimes.to_parquet(train_regimes_path, index=False)

        m04_dir = wdir / "04_models_cv"
        m05_dir = wdir / "05_calibration_ev"
        m04_dir.mkdir(exist_ok=True)
        m05_dir.mkdir(exist_ok=True)

        cmd04 = [
            args.python,
            str((REPO_ROOT / "research" / "04_models_cv.py").resolve()),
            "--trades",
            str(train_trades_path),
            "--targets",
            str(train_targets_path),
            "--regimes",
            str(train_regimes_path),
            "--outdir",
            str(m04_dir),
            "--target",
            str(args.target),
            "--n-splits",
            str(int(args.n_splits)),
            "--embargo-days",
            str(float(args.embargo_days)),
            "--min-eval-n",
            str(int(args.min_eval_n)),
            "--seed",
            str(int(args.seed)),
            "--max-missing",
            str(float(args.max_missing)),
            "--min-unique-numeric",
            str(int(args.min_unique_numeric)),
            "--train-scope",
            str(args.train_scope),
            "--time-decay-halflife-days",
            str(float(args.time_decay_halflife_days)),
            "--allow-missing-oof",
        ]
        if bool(args.live_safe_features):
            cmd04.append("--live-safe-features")

        rc = _run_cmd(cmd04, cwd=REPO_ROOT, log_path=wdir / "logs_04_models_cv.txt")
        if rc != 0:
            row = {
                "window_id": w.window_id,
                "status": "failed_step04",
                "train_start": str(w.train_start),
                "train_end": str(w.train_end),
                "val_start": str(w.val_start),
                "val_end": str(w.val_end),
                "train_n": train_n,
                "valid_n": valid_n,
            }
            done_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            window_rows.append(row)
            continue

        cmd05 = [
            args.python,
            str((REPO_ROOT / "research" / "05_calibration_ev.py").resolve()),
            "--oof",
            str(m04_dir / "oof_predictions.parquet"),
            "--outdir",
            str(m05_dir),
            "--target",
            str(args.target),
            "--methods",
            str(args.cal_methods),
            "--fit-scope",
            str(args.fit_scope),
            "--holdout-frac",
            str(float(args.cal_holdout_frac)),
            "--n-calib-bins",
            str(int(args.cal_bins)),
            "--min-trades",
            str(int(args.cal_min_trades)),
            "--n-thresholds",
            str(int(args.cal_thresholds)),
            "--sizing-bins",
            str(int(args.cal_sizing_bins)),
            "--sizing-min-bin-n",
            str(int(args.cal_sizing_min_bin_n)),
            "--min-mult",
            str(float(args.cal_min_mult)),
            "--max-mult",
            str(float(args.cal_max_mult)),
            "--no-plots",
        ]
        rc = _run_cmd(cmd05, cwd=REPO_ROOT, log_path=wdir / "logs_05_calibration_ev.txt")
        if rc != 0:
            row = {
                "window_id": w.window_id,
                "status": "failed_step05",
                "train_start": str(w.train_start),
                "train_end": str(w.train_end),
                "val_start": str(w.val_start),
                "val_end": str(w.val_end),
                "train_n": train_n,
                "valid_n": valid_n,
            }
            done_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            window_rows.append(row)
            continue

        s05 = json.loads((m05_dir / "summary.json").read_text(encoding="utf-8"))
        model_col, cal_method, holdout_ll = _pick_best_model(s05)
        model_name = model_col[2:] if model_col.startswith("p_") else model_col
        model_path = m04_dir / f"final_model_{model_name}.joblib"
        if not model_path.exists():
            matches = sorted(m04_dir.glob("final_model_*.joblib"))
            if not matches:
                raise RuntimeError(f"No final model found for {w.window_id}")
            model_path = matches[0]

        manifest = json.loads((m04_dir / "manifest.json").read_text(encoding="utf-8"))
        feat_obj = manifest.get("features", {})
        numeric_cols = list(feat_obj.get("numeric_cols") or [])
        cat_cols = list(feat_obj.get("cat_cols") or [])
        feature_cols = list(dict.fromkeys(numeric_cols + cat_cols))
        if not feature_cols:
            raise RuntimeError(f"{w.window_id}: no feature columns in manifest.")

        # Build validation matrix with exact feature schema.
        v = valid_df.copy()
        v = v[v[args.target].notna()].copy()
        v = v[np.isfinite(v["pnl_R"])].copy()
        if v.empty:
            row = {
                "window_id": w.window_id,
                "status": "skipped_no_valid_target",
                "train_start": str(w.train_start),
                "train_end": str(w.train_end),
                "val_start": str(w.val_start),
                "val_end": str(w.val_end),
                "train_n": train_n,
                "valid_n": valid_n,
            }
            done_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            window_rows.append(row)
            continue

        x = v.copy()
        for c in feature_cols:
            if c not in x.columns:
                x[c] = np.nan
        x = x[feature_cols].copy()
        x = step04._drop_duplicate_columns(x, "X_val")
        for c in numeric_cols:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        for c in cat_cols:
            if c in x.columns:
                x[c] = step04._as_object_with_nan(x[c])
        x = x.where(pd.notna(x), np.nan)

        model = joblib.load(model_path)
        p_raw = _predict_raw(model, x)

        oof = pd.read_parquet(m04_dir / "oof_predictions.parquet")
        if model_col not in oof.columns:
            raise RuntimeError(f"{w.window_id}: model col {model_col} missing in OOF")
        oof[args.target] = pd.to_numeric(oof[args.target], errors="coerce")
        oof[model_col] = pd.to_numeric(oof[model_col], errors="coerce")
        oof = oof[np.isfinite(oof[args.target]) & np.isfinite(oof[model_col])].copy()
        if str(args.fit_scope).upper() == "RISK_ON_1":
            risk_col = "risk_on_1" if "risk_on_1" in oof.columns else ("risk_on" if "risk_on" in oof.columns else None)
            if risk_col is not None:
                r = pd.to_numeric(oof[risk_col], errors="coerce").fillna(0).to_numpy()
                oof = oof.loc[r == 1].copy()
        if oof.empty:
            raise RuntimeError(f"{w.window_id}: empty OOF after fit-scope filtering.")

        y_fit = oof[args.target].to_numpy(dtype=np.int32)
        p_fit = oof[model_col].to_numpy(dtype=np.float64)
        calibrator = step05.fit_calibrator(cal_method, y_fit, p_fit)
        p_cal = calibrator.predict(np.asarray(p_raw, dtype=np.float64))

        y_val = pd.to_numeric(v[args.target], errors="coerce").astype(int).to_numpy()
        pnl_val = pd.to_numeric(v["pnl_R"], errors="coerce").to_numpy(dtype=np.float64)

        scored = pd.DataFrame(
            {
                "trade_id": pd.to_numeric(v["trade_id"], errors="coerce").astype(np.int64),
                "entry_ts": pd.to_datetime(v["entry_ts"], utc=True, errors="coerce"),
                "symbol": v["symbol"].astype(str) if "symbol" in v.columns else "",
                "window_id": w.window_id,
                "target": y_val.astype(np.int16),
                "pnl_R": pnl_val.astype(np.float64),
                "p_raw": np.asarray(p_raw, dtype=np.float64),
                "p_cal": np.asarray(p_cal, dtype=np.float64),
            }
        )
        scored = scored[np.isfinite(scored["p_raw"]) & np.isfinite(scored["p_cal"])].copy()
        if scored.empty:
            raise RuntimeError(f"{w.window_id}: empty scored validation set.")

        # Calibration + rank-lift metrics.
        y = scored["target"].to_numpy(dtype=np.int32)
        p = scored["p_cal"].to_numpy(dtype=np.float64)
        rank_tbl, lift = _rank_lift_table(scored, prob_col="p_cal", y_col="target", pnl_col="pnl_R")
        month_tbl = _monthly_metrics(scored, y_col="target", pnl_col="pnl_R")
        month_tbl.insert(0, "window_id", w.window_id)

        window_metric = {
            "window_id": w.window_id,
            "status": "ok",
            "train_start": str(w.train_start),
            "train_end": str(w.train_end),
            "val_start": str(w.val_start),
            "val_end": str(w.val_end),
            "train_n": train_n,
            "valid_n": valid_n,
            "selected_model_col": model_col,
            "selected_calibration": cal_method,
            "holdout_logloss_selected": (None if not np.isfinite(holdout_ll) else float(holdout_ll)),
            "auc_val": float(step05._safe_auc(y, p)),
            "avg_precision_val": float(step05._safe_ap(y, p)),
            "logloss_val": float(step05._safe_logloss(y, p)),
            "brier_val": float(step05._safe_brier(y, p)),
            "ece_val": float(_ece(y, p, n_bins=int(args.cal_bins))),
            "mean_p_val": float(np.mean(p)),
            "obs_rate_val": float(np.mean(y)),
            "calibration_abs_gap_val": float(abs(np.mean(p) - np.mean(y))),
            "mean_pnl_R_val": float(np.mean(scored["pnl_R"].to_numpy(dtype=np.float64))),
            "win_rate_val": float(np.mean(scored["pnl_R"].to_numpy(dtype=np.float64) > 0.0)),
            "top_decile_win_lift_val": float(lift["top_decile_win_lift"]),
            "top_decile_mean_pnl_lift_val": float(lift["top_decile_mean_pnl_lift"]),
            "spearman_prob_vs_pnl_val": float(lift["spearman_prob_vs_pnl"]),
        }

        scored.to_parquet(wdir / "validation_scored.parquet", index=False)
        scored.to_csv(wdir / "validation_scored.csv.gz", index=False, compression="gzip")
        rank_tbl.to_csv(wdir / "rank_lift_deciles.csv", index=False)
        month_tbl.to_csv(wdir / "monthly_metrics.csv", index=False)
        (wdir / "window_summary.json").write_text(
            json.dumps(window_metric, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        done_path.write_text(json.dumps(window_metric, indent=2, sort_keys=True), encoding="utf-8")

        window_rows.append(window_metric)
        all_scored_parts.append(scored)
        all_monthly_rows.append(month_tbl)

        if not bool(args.keep_window_artifacts):
            # keep compact artifacts by default
            data_dir = wdir / "_data"
            if data_dir.exists():
                for pth in data_dir.glob("*"):
                    try:
                        pth.unlink()
                    except Exception:
                        pass
                try:
                    data_dir.rmdir()
                except Exception:
                    pass

    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(run_dir / "aggregate" / "window_metrics.csv", index=False)
    (run_dir / "aggregate" / "window_metrics.json").write_text(
        json.dumps(window_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    monthly_by_window = (
        pd.concat(all_monthly_rows, ignore_index=True) if all_monthly_rows else pd.DataFrame()
    )
    monthly_by_window.to_csv(run_dir / "aggregate" / "monthly_metrics_by_window.csv", index=False)

    monthly_global = pd.DataFrame()
    if all_scored_parts:
        scored_all = pd.concat(all_scored_parts, ignore_index=True)
        scored_all["entry_ts"] = pd.to_datetime(scored_all["entry_ts"], utc=True, errors="coerce")
        scored_all = scored_all.sort_values(["entry_ts", "window_id"], kind="mergesort")
        # Defensive dedupe if future configs use overlapping validation windows.
        scored_all = scored_all.drop_duplicates(subset=["trade_id"], keep="first").reset_index(drop=True)
        scored_all.to_parquet(run_dir / "aggregate" / "all_validation_scored.parquet", index=False)
        scored_all.to_csv(run_dir / "aggregate" / "all_validation_scored.csv.gz", index=False, compression="gzip")
        monthly_global = _monthly_metrics(scored_all, y_col="target", pnl_col="pnl_R")
    monthly_global.to_csv(run_dir / "aggregate" / "monthly_oos_stability.csv", index=False)

    run_meta = {
        "run_id": rid,
        "trades": str(trades_path),
        "targets": str(targets_path),
        "regimes": str(regimes_path),
        "start": str(start_ts),
        "end": str(end_ts),
        "train_months": int(args.train_months),
        "valid_months": int(args.valid_months),
        "step_months": int(args.step_months),
        "expanding_train": bool(args.expanding_train),
        "windows_total": int(len(windows)),
        "windows_ok": int((window_df["status"] == "ok").sum()) if not window_df.empty and "status" in window_df.columns else 0,
        "target": str(args.target),
        "train_scope": str(args.train_scope),
        "fit_scope": str(args.fit_scope),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")

    assets = run_dir / "aggregate" / "assets"
    _plot_series(window_df, "window_id", "top_decile_win_lift_val", "Top Decile Win Lift by Window", assets / "window_top_decile_lift.png")
    _plot_series(window_df, "window_id", "ece_val", "ECE by Window", assets / "window_ece.png")
    if not monthly_global.empty:
        mg = monthly_global.copy()
        mg = mg.sort_values("month")
        if HAS_PLT:
            plt.figure(figsize=(10, 4))
            plt.plot(mg["month"], mg["win_rate"], marker="o", label="win_rate")
            plt.plot(mg["month"], mg["mean_p_cal"], marker="o", label="mean_p_cal")
            plt.xticks(rotation=45, ha="right")
            plt.title("Monthly OOS: Win Rate vs Mean Predicted Probability")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            assets.mkdir(parents=True, exist_ok=True)
            plt.savefig(assets / "monthly_win_vs_prob.png", dpi=160)
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.bar(mg["month"], mg["mean_pnl_R"])
            plt.xticks(rotation=45, ha="right")
            plt.title("Monthly OOS: Mean PnL_R")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(assets / "monthly_mean_pnl.png", dpi=160)
            plt.close()

    _write_report_md(
        out_path=run_dir / "aggregate" / "monthly_oos_stability_report.md",
        run_meta=run_meta,
        window_df=window_df,
        monthly_df=monthly_global,
    )
    _write_report_html(
        out_path=run_dir / "aggregate" / "monthly_oos_stability_report.html",
        run_meta=run_meta,
        window_df=window_df,
        monthly_df=monthly_global,
        assets_rel="assets",
    )

    print(f"[wf] DONE run_id={rid}", flush=True)
    print(f"[wf] root={run_dir}", flush=True)
    print(f"[wf] windows={len(windows)} ok={run_meta['windows_ok']}", flush=True)
    print(f"[wf] report_md={run_dir / 'aggregate' / 'monthly_oos_stability_report.md'}", flush=True)
    print(f"[wf] report_html={run_dir / 'aggregate' / 'monthly_oos_stability_report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
