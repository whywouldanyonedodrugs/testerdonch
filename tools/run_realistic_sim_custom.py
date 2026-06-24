#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scout+backtest for fixed period with custom policy overrides.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--parquet-dir", required=True)
    p.add_argument("--parquet-1m-dir", default="")
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-root", default="results/simulations")

    p.add_argument("--meta-prob-threshold", type=float, default=0.42)
    p.add_argument("--meta-online-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--meta-sizing-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--meta-model-dir", default="/opt/testerdonch/results/offline_releases/20260209_meta_release_v2_live_safe/meta_export_pstar_042")

    p.add_argument("--btc-vol-hi", type=float, default=1.05)
    p.add_argument("--risk-mode", choices=["cash", "percent"], default="cash")
    p.add_argument("--fixed-risk-cash", type=float, default=100.0)
    p.add_argument("--risk-pct", type=float, default=None)

    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument("--tg-auto-chat", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _notify(a: argparse.Namespace, run_label: str, title: str, body: str) -> None:
    try:
        from telegram_notify import TelegramNotifier
        n = TelegramNotifier.from_args(a, run_label=run_label)
        n.send(title, body=body)
    except Exception:
        pass


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

    _notify(a, f"sim:{a.run_id}", "STARTED", f"run_root={run_root}\nperiod={a.start}..{a.end}")

    try:
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

        cfg.BTC_VOL_HI = float(a.btc_vol_hi)
        cfg.RISK_MODE = str(a.risk_mode)
        cfg.FIXED_RISK_CASH = float(a.fixed_risk_cash)
        if a.risk_pct is not None:
            cfg.RISK_PCT = float(a.risk_pct)

        cfg.META_PROB_THRESHOLD = float(a.meta_prob_threshold)
        cfg.BT_META_ONLINE_ENABLED = bool(a.meta_online_enabled)
        cfg.META_SIZING_ENABLED = bool(a.meta_sizing_enabled)
        if str(a.meta_model_dir).strip():
            cfg.META_MODEL_DIR = Path(str(a.meta_model_dir).strip()).resolve()

        started = datetime.now(timezone.utc)
        print(f"[sim] started_utc={started.isoformat()}", flush=True)
        print(f"[sim] run_root={run_root}", flush=True)
        print(f"[sim] period={cfg.START_DATE}..{cfg.END_DATE}", flush=True)
        print(f"[sim] parquet_dir={cfg.PARQUET_DIR}", flush=True)
        print(
            f"[sim] overrides: BTC_VOL_HI={cfg.BTC_VOL_HI} RISK_MODE={cfg.RISK_MODE} "
            f"FIXED_RISK_CASH={cfg.FIXED_RISK_CASH}",
            flush=True,
        )
        print(
            f"[sim] meta: online={cfg.BT_META_ONLINE_ENABLED} gate={cfg.META_PROB_THRESHOLD} "
            f"sizing={cfg.META_SIZING_ENABLED} model_dir={cfg.META_MODEL_DIR}",
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
            "overrides": {
                "BTC_VOL_HI": float(cfg.BTC_VOL_HI),
                "RISK_MODE": str(cfg.RISK_MODE),
                "FIXED_RISK_CASH": float(cfg.FIXED_RISK_CASH),
                "RISK_PCT": float(getattr(cfg, "RISK_PCT", np.nan)),
                "META_PROB_THRESHOLD": float(cfg.META_PROB_THRESHOLD),
                "BT_META_ONLINE_ENABLED": bool(cfg.BT_META_ONLINE_ENABLED),
                "META_SIZING_ENABLED": bool(cfg.META_SIZING_ENABLED),
                "META_MODEL_DIR": str(cfg.META_MODEL_DIR),
            },
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

        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[sim] summary={summary_path}", flush=True)
        print(f"[sim] done_utc={datetime.now(timezone.utc).isoformat()}", flush=True)

        _notify(
            a,
            f"sim:{a.run_id}",
            "DONE",
            f"summary={summary_path}\ntrades_rows={summary.get('trades_rows', 0)}\ntotal_pnl_cash={summary.get('total_pnl_cash', 0.0)}",
        )
        return 0
    except Exception as exc:
        print(f"[sim] FAILED: {type(exc).__name__}: {exc}", flush=True)
        _notify(
            a,
            f"sim:{a.run_id}",
            "FAILED",
            f"reason={type(exc).__name__}: {exc}\nrun_root={run_root}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
