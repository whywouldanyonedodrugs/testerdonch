#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scout+backtest for a fixed period with isolated output dirs.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--parquet-dir", required=True)
    p.add_argument("--parquet-1m-dir", default="")
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-root", default="results/simulations")
    p.add_argument("--meta-prob-threshold", type=float, default=None)
    p.add_argument("--meta-online-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--meta-sizing-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--meta-model-dir", default="")
    return p.parse_args()


def main() -> int:
    a = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import config as cfg
    from scout import run_scout
    import backtester

    run_root = Path(a.results_root).resolve() / a.run_id
    signals_dir = run_root / "signals"
    bt_dir = run_root / "backtest"
    run_root.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(parents=True, exist_ok=True)
    bt_dir.mkdir(parents=True, exist_ok=True)

    cfg.START_DATE = str(a.start)
    cfg.END_DATE = str(a.end)
    cfg.PARQUET_DIR = Path(a.parquet_dir).resolve()
    if str(a.parquet_1m_dir).strip():
        p1m = Path(a.parquet_1m_dir).resolve()
        if p1m.exists():
            cfg.PARQUET_1M_DIR = p1m

    cfg.SIGNALS_DIR = signals_dir
    cfg.RESULTS_DIR = bt_dir
    cfg.SCOUT_CLEAN_OUTPUT_DIR = True
    if a.meta_prob_threshold is not None:
        cfg.META_PROB_THRESHOLD = float(a.meta_prob_threshold)
    if a.meta_online_enabled is not None:
        cfg.BT_META_ONLINE_ENABLED = bool(a.meta_online_enabled)
    if a.meta_sizing_enabled is not None:
        cfg.META_SIZING_ENABLED = bool(a.meta_sizing_enabled)
    if str(a.meta_model_dir).strip():
        cfg.META_MODEL_DIR = Path(str(a.meta_model_dir).strip()).resolve()

    started = datetime.now(timezone.utc)
    print(f"[sim] started_utc={started.isoformat()}", flush=True)
    print(f"[sim] run_root={run_root}", flush=True)
    print(f"[sim] period={cfg.START_DATE}..{cfg.END_DATE}", flush=True)
    print(f"[sim] parquet_dir={cfg.PARQUET_DIR}", flush=True)
    print(
        f"[sim] meta: online={getattr(cfg,'BT_META_ONLINE_ENABLED',None)} "
        f"gate={getattr(cfg,'META_PROB_THRESHOLD',None)} "
        f"sizing={getattr(cfg,'META_SIZING_ENABLED',None)} "
        f"model_dir={getattr(cfg,'META_MODEL_DIR',None)}",
        flush=True,
    )

    n = int(run_scout())
    print(f"[sim] scout_signals={n}", flush=True)

    backtester.run_backtest(signals_path=cfg.SIGNALS_DIR)

    trades_path = bt_dir / "trades.csv"
    summary = {
        "run_id": a.run_id,
        "status": "ok",
        "start": str(a.start),
        "end": str(a.end),
        "parquet_dir": str(cfg.PARQUET_DIR),
        "signals_dir": str(signals_dir),
        "backtest_dir": str(bt_dir),
        "signals_rows": n,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if trades_path.exists():
        t = pd.read_csv(trades_path)
        summary["trades_rows"] = int(len(t))
        if len(t) > 0:
            pnl = pd.to_numeric(t.get("pnl"), errors="coerce")
            pnl_r = pd.to_numeric(t.get("pnl_R"), errors="coerce")
            summary["total_pnl_cash"] = float(np.nansum(pnl.values))
            summary["total_pnl_R"] = float(np.nansum(pnl_r.values))
            summary["win_rate"] = float(np.nanmean((pnl > 0).astype(float).values))
        else:
            summary["total_pnl_cash"] = 0.0
            summary["total_pnl_R"] = 0.0
            summary["win_rate"] = float("nan")

    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sim] summary={run_root / 'summary.json'}", flush=True)
    print(f"[sim] done_utc={datetime.now(timezone.utc).isoformat()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
