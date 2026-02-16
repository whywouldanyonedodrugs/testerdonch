#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sweep_policy_settings_v2 import (  # type: ignore
    find_trades_file,
    import_cfg,
    latest_from_signals,
    load_scoped_signals,
    preflight_symbols,
    resolve_paths,
    run_backtest_subprocess,
    write_signals_file,
)
from telegram_notify import TelegramNotifier

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_date(s: str) -> str:
    if s.lower() == "latest":
        return "latest"
    datetime.strptime(s, "%Y-%m-%d")
    return s


def _utc_run_id(user_id: str) -> str:
    return user_id.strip() or datetime.now(timezone.utc).strftime("jt013_%Y%m%d_%H%M%S")


def _safe_float(x: object, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _set_low_mem_env() -> None:
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _is_oom_like_rc(rc: object) -> bool:
    try:
        v = int(rc)
    except Exception:
        return False
    return v in (-9, 137)


def _mem_total_gb() -> float:
    p = Path("/proc/meminfo")
    if not p.exists():
        return float("nan")
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
        for line in txt.splitlines():
            if line.startswith("MemTotal:"):
                # kB
                kb = float(line.split()[1])
                return kb / (1024.0 * 1024.0)
    except Exception:
        return float("nan")
    return float("nan")


def _effective_jobs(requested: int, reserve_gb: float, per_worker_gb: float) -> Tuple[int, Dict[str, object]]:
    req = max(1, int(requested))
    total = _mem_total_gb()
    if not np.isfinite(total):
        return req, {
            "requested_jobs": req,
            "effective_jobs": req,
            "mem_total_gb": None,
            "note": "meminfo_unavailable_no_clamp",
        }
    safe = int(max(1.0, np.floor(max(total - float(reserve_gb), 1.0) / max(float(per_worker_gb), 0.5))))
    eff = max(1, min(req, safe))
    note = "clamped_by_mem" if eff < req else "ok"
    return eff, {
        "requested_jobs": req,
        "effective_jobs": eff,
        "mem_total_gb": float(total),
        "mem_reserve_gb": float(reserve_gb),
        "mem_per_worker_gb": float(per_worker_gb),
        "max_jobs_by_mem": int(safe),
        "note": note,
    }


def _max_drawdown(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    eq = np.cumsum(np.nan_to_num(x, nan=0.0))
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(np.nanmin(dd)) if dd.size else float("nan")


def _drawdown_tail_p05(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    eq = np.cumsum(np.nan_to_num(x, nan=0.0))
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    if dd.size == 0:
        return float("nan")
    return float(np.nanquantile(dd, 0.05))


def _empty_metrics_payload(run_dir: Path, *, reason: str) -> Dict[str, Any]:
    return {
        "status": "ok",
        "trades_file": "",
        "metrics_note": str(reason),
        "number_of_trades": 0,
        "total_pnl_R": 0.0,
        "win_rate": float("nan"),
        "max_drawdown_R": float("nan"),
        "drawdown_tail_p05_R": float("nan"),
        "downside_deviation_R": float("nan"),
        "positive_month_ratio": float("nan"),
        "worst_month_mean_pnl_R": float("nan"),
        "worst_window_mean_pnl_R": float("nan"),
        "months_covered": 0,
        "rolling_trades_window": float("nan"),
        "run_dir": str(run_dir),
    }


def _compute_metrics(
    run_dir: Path,
    *,
    rolling_trades_window: int,
) -> Dict[str, Any]:
    trades_path = find_trades_file(run_dir)
    df = pd.read_csv(trades_path, low_memory=False)

    pnl_r_col = None
    for c in ("pnl_R", "pnl_r"):
        if c in df.columns:
            pnl_r_col = c
            break
    if pnl_r_col is None:
        raise RuntimeError(f"No pnl_R-like column in {trades_path}")

    ts_col = None
    for c in ("exit_ts", "closed_at", "timestamp", "ts"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise RuntimeError(f"No exit timestamp column in {trades_path}")

    w = df.copy()
    w["ts"] = pd.to_datetime(w[ts_col], utc=True, errors="coerce")
    w["pnl_R_use"] = pd.to_numeric(w[pnl_r_col], errors="coerce")
    w = w[w["ts"].notna() & w["pnl_R_use"].notna()].copy()
    w = w.sort_values("ts", kind="mergesort")

    pnl = w["pnl_R_use"].to_numpy(dtype=float)
    n = int(len(pnl))
    if n <= 0:
        return {
            "status": "ok",
            "trades_file": str(trades_path),
            "number_of_trades": 0,
            "total_pnl_R": 0.0,
            "win_rate": float("nan"),
            "max_drawdown_R": float("nan"),
            "drawdown_tail_p05_R": float("nan"),
            "downside_deviation_R": float("nan"),
            "positive_month_ratio": float("nan"),
            "worst_month_mean_pnl_R": float("nan"),
            "worst_window_mean_pnl_R": float("nan"),
            "months_covered": 0,
        }

    win_rate = float((pnl > 0).mean())
    total_pnl_r = float(np.nansum(pnl))
    max_dd_r = _max_drawdown(pnl)
    dd_tail_p05_r = _drawdown_tail_p05(pnl)
    downside_dev = float(np.sqrt(np.nanmean(np.minimum(pnl, 0.0) ** 2)))

    w["month"] = w["ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    month_mean = w.groupby("month", sort=True)["pnl_R_use"].mean()
    months_covered = int(len(month_mean))
    positive_month_ratio = float((month_mean > 0).mean()) if months_covered > 0 else float("nan")
    worst_month_mean = float(month_mean.min()) if months_covered > 0 else float("nan")

    roll_n = int(max(2, rolling_trades_window))
    min_periods = int(min(len(pnl), max(2, roll_n // 2)))
    roll = pd.Series(pnl).rolling(window=roll_n, min_periods=min_periods).mean()
    worst_window_mean = float(roll.min()) if roll.notna().any() else float("nan")

    return {
        "status": "ok",
        "trades_file": str(trades_path),
        "number_of_trades": n,
        "total_pnl_R": total_pnl_r,
        "win_rate": win_rate,
        "max_drawdown_R": max_dd_r,
        "drawdown_tail_p05_R": dd_tail_p05_r,
        "downside_deviation_R": downside_dev,
        "positive_month_ratio": positive_month_ratio,
        "worst_month_mean_pnl_R": worst_month_mean,
        "worst_window_mean_pnl_R": worst_window_mean,
        "months_covered": months_covered,
        "rolling_trades_window": roll_n,
    }


@dataclass(frozen=True)
class GridPoint:
    name: str
    block_when_down: bool
    size_when_down: float
    probe_mult: float

    @property
    def overrides(self) -> Dict[str, Any]:
        return {
            "REGIME_BLOCK_WHEN_DOWN": bool(self.block_when_down),
            "REGIME_SIZE_WHEN_DOWN": float(self.size_when_down),
            "RISK_OFF_PROBE_MULT": float(self.probe_mult),
        }


def _build_grid(
    *,
    size_values: List[float],
    probe_values: List[float],
) -> List[GridPoint]:
    out: List[GridPoint] = []
    for p in probe_values:
        out.append(
            GridPoint(
                name=f"block_T__size_100__probe_{str(p).replace('.', '')}",
                block_when_down=True,
                size_when_down=1.0,
                probe_mult=float(p),
            )
        )
    for s in size_values:
        for p in probe_values:
            out.append(
                GridPoint(
                    name=f"block_F__size_{int(round(s * 100)):03d}__probe_{str(p).replace('.', '')}",
                    block_when_down=False,
                    size_when_down=float(s),
                    probe_mult=float(p),
                )
            )
    return out


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("Empty float list")
    return out


def _find_baseline_row(df: pd.DataFrame, cfg) -> pd.Series:
    b = bool(getattr(cfg, "REGIME_BLOCK_WHEN_DOWN", False))
    s = float(getattr(cfg, "REGIME_SIZE_WHEN_DOWN", 0.2))
    p = float(getattr(cfg, "RISK_OFF_PROBE_MULT", 0.01))
    m = (
        (df["REGIME_BLOCK_WHEN_DOWN"].astype(bool) == b)
        & (np.isclose(pd.to_numeric(df["REGIME_SIZE_WHEN_DOWN"], errors="coerce"), s))
        & (np.isclose(pd.to_numeric(df["RISK_OFF_PROBE_MULT"], errors="coerce"), p))
    )
    if bool(m.any()):
        return df.loc[m].iloc[0]
    return df.iloc[0]


def _apply_guardrails(
    df: pd.DataFrame,
    *,
    baseline: pd.Series,
    min_positive_month_ratio: float,
    min_worst_window_mean_pnl_r: float,
    min_trades_frac_vs_baseline: float,
    max_downside_dev_mult_vs_baseline: float,
    max_dd_tail_worsen_frac_vs_baseline: float,
    min_total_pnl_frac_vs_baseline: float,
) -> pd.DataFrame:
    d = df.copy()
    base_pos = _safe_float(baseline.get("positive_month_ratio"))
    base_worst_win = _safe_float(baseline.get("worst_window_mean_pnl_R"))
    base_trades = max(1.0, _safe_float(baseline.get("number_of_trades"), 1.0))
    base_down = max(1e-9, _safe_float(baseline.get("downside_deviation_R"), 1.0))
    base_dd_tail = _safe_float(baseline.get("drawdown_tail_p05_R"), -1.0)
    base_total = _safe_float(baseline.get("total_pnl_R"), 0.0)

    d["soft_positive_month_pass"] = pd.to_numeric(d["positive_month_ratio"], errors="coerce") >= float(min_positive_month_ratio)
    d["soft_worst_window_pass"] = pd.to_numeric(d["worst_window_mean_pnl_R"], errors="coerce") >= float(min_worst_window_mean_pnl_r)
    d["soft_pass_count"] = d[["soft_positive_month_pass", "soft_worst_window_pass"]].sum(axis=1)
    d["soft_positive_month_improved_vs_baseline"] = pd.to_numeric(d["positive_month_ratio"], errors="coerce") > base_pos
    d["soft_worst_window_improved_vs_baseline"] = pd.to_numeric(d["worst_window_mean_pnl_R"], errors="coerce") > base_worst_win
    d["soft_improved_any_vs_baseline"] = d[
        ["soft_positive_month_improved_vs_baseline", "soft_worst_window_improved_vs_baseline"]
    ].any(axis=1)

    d["hard_min_trades_pass"] = pd.to_numeric(d["number_of_trades"], errors="coerce") >= float(min_trades_frac_vs_baseline) * base_trades
    d["hard_downside_pass"] = pd.to_numeric(d["downside_deviation_R"], errors="coerce") <= float(max_downside_dev_mult_vs_baseline) * base_down
    allowed_tail = float(max_dd_tail_worsen_frac_vs_baseline) * max(abs(base_dd_tail), 1e-9)
    d["hard_dd_tail_pass"] = pd.to_numeric(d["drawdown_tail_p05_R"], errors="coerce") >= (base_dd_tail - allowed_tail)

    if base_total > 0:
        d["hard_total_pnl_pass"] = pd.to_numeric(d["total_pnl_R"], errors="coerce") >= float(min_total_pnl_frac_vs_baseline) * base_total
    else:
        d["hard_total_pnl_pass"] = True

    d["hard_proxy_pass"] = d[
        ["hard_min_trades_pass", "hard_downside_pass", "hard_dd_tail_pass", "hard_total_pnl_pass"]
    ].all(axis=1)

    d["candidate_default"] = d["hard_proxy_pass"] & d["soft_improved_any_vs_baseline"]
    d["candidate_conservative"] = d["hard_proxy_pass"]
    return d


def _pick_recommendations(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    default_df = df[df["candidate_default"]].copy()
    if not default_df.empty:
        default_df = default_df.sort_values(
            by=[
                "soft_pass_count",
                "soft_worst_window_pass",
                "soft_positive_month_pass",
                "worst_window_mean_pnl_R",
                "positive_month_ratio",
                "downside_deviation_R",
                "total_pnl_R",
            ],
            ascending=[False, False, False, False, False, True, False],
            kind="mergesort",
        )
        default_row = default_df.iloc[0]
    else:
        default_row = None

    conservative_df = df[df["candidate_conservative"]].copy()
    if not conservative_df.empty:
        if default_row is not None:
            conservative_df = conservative_df[conservative_df["setting"] != default_row["setting"]].copy()
        if conservative_df.empty and default_row is not None:
            conservative_row = default_row
        else:
            conservative_df = conservative_df.sort_values(
                by=[
                    "downside_deviation_R",
                    "drawdown_tail_p05_R",
                    "worst_window_mean_pnl_R",
                    "positive_month_ratio",
                    "total_pnl_R",
                ],
                ascending=[True, False, False, False, False],
                kind="mergesort",
            )
            conservative_row = conservative_df.iloc[0] if not conservative_df.empty else None
    else:
        conservative_row = None
    return default_row, conservative_row


def _variant_override_for_attempt(args: argparse.Namespace, gp: GridPoint, attempt: int) -> Dict[str, Any]:
    perf_overrides = {
        "BT_PROGRESS_ENABLED": False,
        "BT_DECISION_LOG_ENABLED": False,
        "BT_META_REPLAY_ENABLED": False,
        "USE_INTRABAR_1M": False,
    }
    ov = dict(perf_overrides)
    ov.update(gp.overrides)

    if attempt <= 0:
        ov.update(
            {
                "BT_SIGNAL_BATCH_SIZE": int(max(200, args.bt_signal_batch_size)),
                "BT_CACHE_5M_MAX_SYMBOLS": int(max(1, args.bt_cache_5m)),
                "BT_CACHE_1M_MAX_SYMBOLS": int(max(1, args.bt_cache_1m)),
                "BT_CACHE_EQ_MAX_SYMBOLS": int(max(4, args.bt_cache_eq)),
            }
        )
    elif attempt == 1:
        ov.update(
            {
                "BT_SIGNAL_BATCH_SIZE": int(max(200, args.bt_signal_batch_size_retry1)),
                "BT_CACHE_5M_MAX_SYMBOLS": int(max(1, args.bt_cache_5m_retry1)),
                "BT_CACHE_1M_MAX_SYMBOLS": int(max(1, args.bt_cache_1m_retry1)),
                "BT_CACHE_EQ_MAX_SYMBOLS": int(max(4, args.bt_cache_eq_retry1)),
            }
        )
    else:
        ov.update(
            {
                "BT_SIGNAL_BATCH_SIZE": int(max(100, args.bt_signal_batch_size_retry2)),
                "BT_CACHE_5M_MAX_SYMBOLS": int(max(1, args.bt_cache_5m_retry2)),
                "BT_CACHE_1M_MAX_SYMBOLS": int(max(1, args.bt_cache_1m_retry2)),
                "BT_CACHE_EQ_MAX_SYMBOLS": int(max(4, args.bt_cache_eq_retry2)),
            }
        )
    return ov


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JT-013 constrained risk-off containment sweep.")
    p.add_argument("--start", type=_parse_date, default="2023-01-01")
    p.add_argument("--end", type=_parse_date, default="latest")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--on-missing-symbol", choices=["fail", "skip"], default="skip")
    p.add_argument("--smoke-n", type=int, default=0, help="If >0, run only first N scoped signals.")
    p.add_argument("--rolling-trades-window", type=int, default=250)

    p.add_argument("--size-values", type=str, default="0.2,0.35,0.5")
    p.add_argument("--probe-values", type=str, default="0.01,0.05,0.25")

    p.add_argument("--variant-retries", type=int, default=2, help="Retries per variant after non-zero rc.")
    p.add_argument("--retry-until-complete", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-passes", type=int, default=4, help="0 means infinite passes when --retry-until-complete is enabled.")
    p.add_argument("--sleep-sec", type=float, default=20.0)

    p.add_argument("--safe-auto-jobs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mem-reserve-gb", type=float, default=3.0)
    p.add_argument("--mem-per-worker-gb", type=float, default=4.0)

    p.add_argument("--bt-signal-batch-size", type=int, default=2500)
    p.add_argument("--bt-cache-5m", type=int, default=4)
    p.add_argument("--bt-cache-1m", type=int, default=1)
    p.add_argument("--bt-cache-eq", type=int, default=24)

    p.add_argument("--bt-signal-batch-size-retry1", type=int, default=1200)
    p.add_argument("--bt-cache-5m-retry1", type=int, default=2)
    p.add_argument("--bt-cache-1m-retry1", type=int, default=1)
    p.add_argument("--bt-cache-eq-retry1", type=int, default=16)

    p.add_argument("--bt-signal-batch-size-retry2", type=int, default=600)
    p.add_argument("--bt-cache-5m-retry2", type=int, default=1)
    p.add_argument("--bt-cache-1m-retry2", type=int, default=1)
    p.add_argument("--bt-cache-eq-retry2", type=int, default=8)

    p.add_argument("--min-positive-month-ratio", type=float, default=0.45)
    p.add_argument("--min-worst-window-mean-pnl-r", type=float, default=-0.5)
    p.add_argument("--min-trades-frac-vs-baseline", type=float, default=0.35)
    p.add_argument("--max-downside-dev-mult-vs-baseline", type=float, default=1.20)
    p.add_argument("--max-dd-tail-worsen-frac-vs-baseline", type=float, default=0.20)
    p.add_argument("--min-total-pnl-frac-vs-baseline", type=float, default=0.20)

    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument(
        "--tg-auto-chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try Telegram getUpdates to auto-discover chat id when not provided.",
    )
    return p.parse_args()


def main() -> int:
    a = parse_args()
    _set_low_mem_env()
    cfg = import_cfg()
    signals_dir, parquet_dir, results_dir, _meta_model_dir = resolve_paths(cfg)

    rid = _utc_run_id(a.run_id)
    notifier = TelegramNotifier.from_args(a, run_label=f"jt013:{rid}")
    print(f"[jt013] telegram notify: {notifier.status_line()}", flush=True)

    sweep_root = (results_dir / "jt013_risk_off_sweeps" / rid).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    def _handle_sigterm(_signum, _frame) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        end = a.end
        if str(end).lower() == "latest":
            end = latest_from_signals(signals_dir)

        sig = load_scoped_signals(signals_dir, a.start, end)
        sig = preflight_symbols(sig, parquet_dir, a.on_missing_symbol)
        if a.smoke_n and int(a.smoke_n) > 0:
            sig = sig.head(int(a.smoke_n)).copy()

        scoped_dir = write_signals_file(sig, sweep_root / "_scoped_signals")

        requested_jobs = max(1, int(a.jobs))
        mem_diag: Dict[str, object] = {
            "requested_jobs": requested_jobs,
            "effective_jobs": requested_jobs,
            "note": "safe_auto_jobs_disabled",
        }
        effective_jobs = requested_jobs
        if bool(a.safe_auto_jobs):
            effective_jobs, mem_diag = _effective_jobs(
                requested=requested_jobs,
                reserve_gb=float(a.mem_reserve_gb),
                per_worker_gb=float(a.mem_per_worker_gb),
            )

        (sweep_root / "scoped_info.json").write_text(
            json.dumps(
                {
                    "start": a.start,
                    "end": end,
                    "rows": int(len(sig)),
                    "symbols": int(sig["symbol"].nunique()) if "symbol" in sig.columns else 0,
                    "mem_diag": mem_diag,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        size_values = _parse_float_list(a.size_values)
        probe_values = _parse_float_list(a.probe_values)
        gps = _build_grid(size_values=size_values, probe_values=probe_values)

        cfg_baseline = GridPoint(
            name="baseline_cfg",
            block_when_down=bool(getattr(cfg, "REGIME_BLOCK_WHEN_DOWN", False)),
            size_when_down=float(getattr(cfg, "REGIME_SIZE_WHEN_DOWN", 0.2)),
            probe_mult=float(getattr(cfg, "RISK_OFF_PROBE_MULT", 0.01)),
        )
        if not any(
            (g.block_when_down == cfg_baseline.block_when_down)
            and (abs(g.size_when_down - cfg_baseline.size_when_down) < 1e-12)
            and (abs(g.probe_mult - cfg_baseline.probe_mult) < 1e-12)
            for g in gps
        ):
            gps.append(cfg_baseline)

        notifier.send(
            "STARTED",
            body=(
                f"run_id={rid}\nstart={a.start} end={end}\n"
                f"signals_rows={len(sig)} symbols={int(sig['symbol'].nunique()) if 'symbol' in sig.columns else 0}\n"
                f"jobs={requested_jobs} effective_jobs={effective_jobs}\n"
                f"variant_retries={int(a.variant_retries)} retry_until_complete={bool(a.retry_until_complete)}"
            ),
        )
    except KeyboardInterrupt:
        notifier.send(
            "FAILED",
            body=f"run_id={rid}\nstatus=aborted\nreason=interrupt\npath={sweep_root}",
        )
        return 130
    except Exception as exc:
        notifier.send(
            "FAILED",
            body=f"run_id={rid}\nstatus=failed\nreason={type(exc).__name__}: {exc}\npath={sweep_root}",
        )
        raise

    def run_one(gp: GridPoint) -> Dict[str, Any]:
        run_dir = sweep_root / gp.name
        run_dir.mkdir(parents=True, exist_ok=True)
        done = run_dir / "_DONE.json"
        metrics_path = run_dir / "metrics.json"
        if done.exists() and metrics_path.exists():
            try:
                prev = json.loads(metrics_path.read_text(encoding="utf-8"))
                if str(prev.get("status", "")).lower() == "ok":
                    return prev
            except Exception:
                pass

        attempt_rows: List[Dict[str, Any]] = []
        max_attempts = max(1, int(a.variant_retries) + 1)
        last_rc: Optional[int] = None
        last_err: Optional[str] = None
        for attempt in range(max_attempts):
            overrides = _variant_override_for_attempt(a, gp, attempt)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "start": a.start,
                        "end": end,
                        "signals": str(scoped_dir),
                        "grid_point": {
                            "REGIME_BLOCK_WHEN_DOWN": gp.block_when_down,
                            "REGIME_SIZE_WHEN_DOWN": gp.size_when_down,
                            "RISK_OFF_PROBE_MULT": gp.probe_mult,
                        },
                        "attempt": int(attempt),
                        "overrides": overrides,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            try:
                rc, _ = run_backtest_subprocess(
                    run_dir=run_dir,
                    signals_dir=scoped_dir,
                    start=a.start,
                    end=end,
                    overrides=overrides,
                )
            except Exception as exc:
                rc = 999
                last_err = f"{type(exc).__name__}: {exc}"
            last_rc = int(rc)
            log_src = run_dir / "logs.txt"
            if log_src.exists():
                try:
                    shutil.copy2(log_src, run_dir / f"logs.attempt_{attempt}.txt")
                except Exception:
                    pass
            attempt_rows.append(
                {
                    "attempt": int(attempt),
                    "returncode": int(rc),
                    "is_oom_like": _is_oom_like_rc(rc),
                    "overrides": overrides,
                }
            )
            if int(rc) == 0:
                try:
                    met = _compute_metrics(run_dir, rolling_trades_window=int(a.rolling_trades_window))
                except RuntimeError as exc:
                    # Backtester can complete successfully with no executed trades for this variant/window.
                    if "No trades file found under" in str(exc):
                        met = _empty_metrics_payload(run_dir, reason="no_trades_file_no_executions")
                    else:
                        raise
                met["setting"] = gp.name
                met["status"] = "ok"
                met["returncode"] = 0
                met["REGIME_BLOCK_WHEN_DOWN"] = bool(gp.block_when_down)
                met["REGIME_SIZE_WHEN_DOWN"] = float(gp.size_when_down)
                met["RISK_OFF_PROBE_MULT"] = float(gp.probe_mult)
                met["attempts"] = attempt_rows
                metrics_path.write_text(json.dumps(met, indent=2, sort_keys=True), encoding="utf-8")
                done.write_text(json.dumps({"returncode": 0}, indent=2), encoding="utf-8")
                return met
            time.sleep(1.0)

        out = {
            "setting": gp.name,
            "status": "error",
            "returncode": int(last_rc if last_rc is not None else 999),
            "REGIME_BLOCK_WHEN_DOWN": bool(gp.block_when_down),
            "REGIME_SIZE_WHEN_DOWN": float(gp.size_when_down),
            "RISK_OFF_PROBE_MULT": float(gp.probe_mult),
            "attempts": attempt_rows,
            "error": str(last_err or ""),
        }
        metrics_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        done.write_text(json.dumps({"returncode": int(out["returncode"])}, indent=2), encoding="utf-8")
        return out

    pass_idx = 0
    results_map: Dict[str, Dict[str, Any]] = {}
    current_jobs = max(1, int(effective_jobs))
    while True:
        pass_idx += 1
        this_pass_results: List[Dict[str, Any]] = []
        jobs = max(1, int(current_jobs))
        if jobs == 1:
            for gp in gps:
                this_pass_results.append(run_one(gp))
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = [ex.submit(run_one, gp) for gp in gps]
                for fut in as_completed(futs):
                    this_pass_results.append(fut.result())

        for row in this_pass_results:
            results_map[str(row.get("setting"))] = row
        df = pd.DataFrame(list(results_map.values()))
        summary_path = sweep_root / "summary.csv"
        df.to_csv(summary_path, index=False)

        n_total = len(gps)
        n_ok = int((df["status"].astype(str) == "ok").sum()) if "status" in df.columns else 0
        n_err = int((df["status"].astype(str) == "error").sum()) if "status" in df.columns else (n_total - n_ok)
        err_df = df[df.get("status", "").astype(str) == "error"].copy() if "status" in df.columns else pd.DataFrame()
        oom_detected = bool(
            (pd.to_numeric(err_df.get("returncode"), errors="coerce").apply(_is_oom_like_rc)).any()
            if not err_df.empty
            else False
        )
        next_jobs = int(current_jobs)
        if oom_detected and current_jobs > 1:
            next_jobs = max(1, int(current_jobs) - 1)
        (sweep_root / "_PASS_STATUS.json").write_text(
            json.dumps(
                {
                    "pass": int(pass_idx),
                    "total_variants": int(n_total),
                    "ok_variants": int(n_ok),
                    "error_variants": int(n_err),
                    "oom_detected": bool(oom_detected),
                    "jobs_this_pass": int(current_jobs),
                    "jobs_next_pass": int(next_jobs),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if n_err > 0:
            notifier.send(
                "WARN",
                body=(
                    f"run_id={rid}\npass={pass_idx}\n"
                    f"ok={n_ok}/{n_total} errors={n_err}\n"
                    f"oom_detected={oom_detected}\n"
                    f"jobs_this_pass={current_jobs} jobs_next_pass={next_jobs}\n"
                    f"summary={summary_path}\n"
                    f"action={'retrying' if bool(a.retry_until_complete) else 'stopping'}"
                ),
            )

        current_jobs = int(next_jobs)
        if n_err == 0:
            break
        if not bool(a.retry_until_complete):
            break
        if int(a.max_passes) > 0 and pass_idx >= int(a.max_passes):
            break
        time.sleep(max(1.0, float(a.sleep_sec)))

    df = pd.DataFrame(list(results_map.values()))
    summary_path = sweep_root / "summary.csv"
    df.to_csv(summary_path, index=False)
    df_ok = df[df["status"] == "ok"].copy() if ("status" in df.columns) else pd.DataFrame()

    if df_ok.empty:
        (sweep_root / "report.md").write_text("# JT-013\n\nNo successful variants.\n", encoding="utf-8")
        notifier.send(
            "FAILED",
            body=(
                f"run_id={rid}\nstatus=failed\n"
                f"no successful variants\nsummary={summary_path}"
            ),
        )
        print(str(sweep_root))
        return 1

    baseline = _find_baseline_row(df_ok, cfg)
    df_eval = _apply_guardrails(
        df_ok,
        baseline=baseline,
        min_positive_month_ratio=float(a.min_positive_month_ratio),
        min_worst_window_mean_pnl_r=float(a.min_worst_window_mean_pnl_r),
        min_trades_frac_vs_baseline=float(a.min_trades_frac_vs_baseline),
        max_downside_dev_mult_vs_baseline=float(a.max_downside_dev_mult_vs_baseline),
        max_dd_tail_worsen_frac_vs_baseline=float(a.max_dd_tail_worsen_frac_vs_baseline),
        min_total_pnl_frac_vs_baseline=float(a.min_total_pnl_frac_vs_baseline),
    )
    eval_path = sweep_root / "evaluation.csv"
    df_eval.to_csv(eval_path, index=False)

    default_row, conservative_row = _pick_recommendations(df_eval)
    n_total = len(gps)
    n_ok = int((df["status"].astype(str) == "ok").sum()) if "status" in df.columns else 0
    n_err = int((df["status"].astype(str) == "error").sum()) if "status" in df.columns else (n_total - n_ok)

    rec = {
        "run_id": rid,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "start": a.start,
            "end": end,
            "jobs_requested": int(requested_jobs),
            "jobs_effective": int(effective_jobs),
            "signals_rows": int(len(sig)),
            "signals_symbols": int(sig["symbol"].nunique()) if "symbol" in sig.columns else 0,
            "size_values": size_values,
            "probe_values": probe_values,
            "rolling_trades_window": int(a.rolling_trades_window),
            "variant_retries": int(a.variant_retries),
            "retry_until_complete": bool(a.retry_until_complete),
            "max_passes": int(a.max_passes),
            "mem_diag": mem_diag,
        },
        "guardrails": {
            "min_positive_month_ratio": float(a.min_positive_month_ratio),
            "min_worst_window_mean_pnl_r": float(a.min_worst_window_mean_pnl_r),
            "min_trades_frac_vs_baseline": float(a.min_trades_frac_vs_baseline),
            "max_downside_dev_mult_vs_baseline": float(a.max_downside_dev_mult_vs_baseline),
            "max_dd_tail_worsen_frac_vs_baseline": float(a.max_dd_tail_worsen_frac_vs_baseline),
            "min_total_pnl_frac_vs_baseline": float(a.min_total_pnl_frac_vs_baseline),
        },
        "pass_stats": {
            "passes_completed": int(pass_idx),
            "variants_total": int(n_total),
            "variants_ok": int(n_ok),
            "variants_error": int(n_err),
        },
        "baseline_setting": str(baseline.get("setting")),
        "baseline_metrics": baseline.to_dict(),
        "default_profile": default_row.to_dict() if default_row is not None else None,
        "conservative_profile": conservative_row.to_dict() if conservative_row is not None else None,
        "acceptance_check": {
            "improves_failed_soft_gate_any": bool(default_row is not None and bool(default_row.get("soft_improved_any_vs_baseline"))),
            "passes_hard_proxy": bool(default_row is not None and bool(default_row.get("hard_proxy_pass"))),
        },
    }
    rec_path = sweep_root / "recommendation.json"
    rec_path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")

    top_cols = [
        "setting",
        "REGIME_BLOCK_WHEN_DOWN",
        "REGIME_SIZE_WHEN_DOWN",
        "RISK_OFF_PROBE_MULT",
        "positive_month_ratio",
        "worst_month_mean_pnl_R",
        "worst_window_mean_pnl_R",
        "downside_deviation_R",
        "drawdown_tail_p05_R",
        "number_of_trades",
        "total_pnl_R",
        "hard_proxy_pass",
        "soft_improved_any_vs_baseline",
        "soft_pass_count",
    ]
    for c in top_cols:
        if c not in df_eval.columns:
            df_eval[c] = np.nan
    ranked = df_eval.sort_values(
        by=["hard_proxy_pass", "soft_pass_count", "worst_window_mean_pnl_R", "positive_month_ratio", "downside_deviation_R", "total_pnl_R"],
        ascending=[False, False, False, False, True, False],
        kind="mergesort",
    )
    lines: List[str] = []
    lines.append(f"# JT-013 Risk-Off Containment Sweep: {rid}")
    lines.append("")
    lines.append("## Runtime")
    lines.append(f"- passes_completed: `{pass_idx}`")
    lines.append(f"- variants_ok: `{n_ok}/{n_total}`")
    lines.append(f"- variants_error: `{n_err}`")
    lines.append(f"- jobs_requested: `{requested_jobs}` jobs_effective: `{effective_jobs}`")
    lines.append("")
    lines.append("## Baseline")
    lines.append(f"- setting: `{baseline.get('setting')}`")
    lines.append(f"- positive_month_ratio: `{baseline.get('positive_month_ratio')}`")
    lines.append(f"- worst_window_mean_pnl_R: `{baseline.get('worst_window_mean_pnl_R')}`")
    lines.append("")
    lines.append("## Recommendations")
    lines.append(f"- default_profile: `{(default_row.get('setting') if default_row is not None else 'none')}`")
    lines.append(f"- conservative_profile: `{(conservative_row.get('setting') if conservative_row is not None else 'none')}`")
    lines.append("")
    lines.append("## Ranked Candidates")
    lines.append("")
    lines.append(ranked[top_cols].head(20).to_markdown(index=False))
    (sweep_root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    if n_err > 0:
        notifier.send(
            "FAILED",
            body=(
                f"run_id={rid}\nstatus=partial\n"
                f"ok={n_ok}/{n_total} errors={n_err}\n"
                f"recommendation={rec_path}\n"
                f"report={sweep_root / 'report.md'}\n"
                "attention=rerun same command with same --run-id after fixing resource pressure"
            ),
        )
        print(str(sweep_root))
        return 2

    notifier.send(
        "DONE",
        body=(
            f"run_id={rid}\nstatus=ok\nok={n_ok}/{n_total}\n"
            f"default={(default_row.get('setting') if default_row is not None else 'none')}\n"
            f"conservative={(conservative_row.get('setting') if conservative_row is not None else 'none')}\n"
            f"recommendation={rec_path}"
        ),
    )
    print(str(sweep_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
