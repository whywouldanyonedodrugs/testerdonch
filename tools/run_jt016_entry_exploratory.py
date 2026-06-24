#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from run_jt014_regime_conditioned_exits import _compute_variant_metrics  # type: ignore
from sweep_policy_settings_v2 import (  # type: ignore
    import_cfg,
    latest_from_signals,
    load_scoped_signals,
    resolve_paths,
    run_backtest_subprocess,
    write_signals_file,
)
from adverse_research_helpers import filter_df_by_windows, load_windows_file
from telegram_notify import TelegramNotifier


REPO_ROOT = Path(__file__).resolve().parents[1]


CHILD_SCOUT_CODE = r"""
import json, os, sys
from pathlib import Path

repo_root = Path(os.environ["DONCH_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))

import config as cfg

signals_dir = Path(os.environ["DONCH_SIGNALS_DIR"]).resolve()
start = os.environ["DONCH_START"]
end = os.environ["DONCH_END"]
overrides = json.loads(os.environ["DONCH_OVERRIDES_JSON"])

setattr(cfg, "SIGNALS_DIR", signals_dir)
setattr(cfg, "START_DATE", str(start))
setattr(cfg, "END_DATE", str(end))

for k, v in overrides.items():
    setattr(cfg, k, v)

import scout
n = scout.run_scout()
print(f"[jt016] scout rows={n}")
"""


def _parse_date(s: str) -> str:
    if s.lower() == "latest":
        return "latest"
    datetime.strptime(s, "%Y-%m-%d")
    return s


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("empty float list")
    return out


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(float(t)))
    if not out:
        raise ValueError("empty int list")
    return out


def _utc_run_id(user_id: str) -> str:
    return user_id.strip() or datetime.now(timezone.utc).strftime("jt016_%Y%m%d_%H%M%S")


def _safe_float(x: object, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _load_signals_with_symbol(signals_dir: Path, start: str, end: str) -> pd.DataFrame:
    """
    Load scoped signals and guarantee a physical `symbol` column.
    Falls back to hive partition parsing when needed.
    """
    try:
        df = load_scoped_signals(signals_dir, start, end)
        if "symbol" in df.columns:
            return df
    except Exception:
        df = pd.DataFrame()

    parts = sorted(signals_dir.glob("symbol=*/*.parquet"))
    if not parts:
        raise RuntimeError(
            "Signals appear to be missing a physical 'symbol' column, and no hive partition "
            f"dirs like {signals_dir}/symbol=XYZ were found."
        )

    chunks: List[pd.DataFrame] = []
    for fp in parts:
        try:
            sub = pd.read_parquet(fp)
        except Exception:
            continue
        if sub.empty:
            continue
        sym = fp.parent.name.split("=", 1)[1].strip().upper() if "=" in fp.parent.name else ""
        if "symbol" not in sub.columns:
            sub["symbol"] = sym
        else:
            sub["symbol"] = sub["symbol"].astype(str).str.upper().fillna(sym)
        chunks.append(sub)

    if not chunks:
        raise RuntimeError("No readable partition parquet files were found in seed signals directory.")

    out = pd.concat(chunks, ignore_index=True)
    if "timestamp" not in out.columns:
        raise RuntimeError("Signals are missing required 'timestamp' column.")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "symbol"]).copy()

    st = pd.Timestamp(start, tz="UTC")
    en = pd.Timestamp(end, tz="UTC")
    out = out[(out["timestamp"] >= st) & (out["timestamp"] <= en)].copy()
    return out


def _empty_metrics_payload(reason: str) -> Dict[str, Any]:
    return {
        "status": "ok",
        "reason": reason,
        "number_of_trades": 0,
        "total_pnl_cash": 0.0,
        "total_pnl_R": 0.0,
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
        "max_dd_pct": float("nan"),
        "sharpe_daily": float("nan"),
        "sortino_daily": float("nan"),
        "calmar": float("nan"),
        "overall_worst_window_mean_pnl_R": float("nan"),
        "overall_worst_month_mean_pnl_R": float("nan"),
        "overall_positive_month_ratio": float("nan"),
        "risk_off_n_trades": 0,
        "risk_off_total_pnl_cash": 0.0,
        "risk_off_total_pnl_R": 0.0,
        "risk_off_worst_window_mean_pnl_R": float("nan"),
        "risk_off_worst_month_mean_pnl_R": float("nan"),
        "risk_off_positive_month_ratio": float("nan"),
        "risk_off_downside_deviation_R": float("nan"),
        "risk_off_drawdown_tail_p05_R": float("nan"),
        "risk_off_max_dd_R": float("nan"),
    }


def _meta_mode_overrides(meta_mode: str, meta_threshold: Optional[float]) -> Dict[str, Any]:
    mm = str(meta_mode).strip().lower()
    if mm == "off":
        return {
            "BT_META_ONLINE_ENABLED": False,
            "META_SIZING_ENABLED": False,
            "META_PROB_THRESHOLD": None,
        }
    if mm == "size_only":
        return {
            "BT_META_ONLINE_ENABLED": True,
            "META_SIZING_ENABLED": True,
            "META_PROB_THRESHOLD": None,
        }
    if mm == "full":
        out = {
            "BT_META_ONLINE_ENABLED": True,
            "META_SIZING_ENABLED": True,
        }
        if meta_threshold is not None:
            out["META_PROB_THRESHOLD"] = float(meta_threshold)
        return out
    raise ValueError(f"unsupported meta_mode: {meta_mode!r}")


@dataclass(frozen=True)
class EntryVariant:
    name: str
    don_n_days: int
    retest_eps_pct: float
    retest_lookback_bars: int
    pullback_window_hours: int
    vol_multiple: float
    rs_min_percentile: int

    def scout_overrides(self, scout_workers: int) -> Dict[str, Any]:
        return {
            "DON_N_DAYS": int(self.don_n_days),
            "RETEST_EPS_PCT": float(self.retest_eps_pct),
            "RETEST_LOOKBACK_BARS": int(self.retest_lookback_bars),
            "PULLBACK_WINDOW_HOURS": int(self.pullback_window_hours),
            "VOL_MULTIPLE": float(self.vol_multiple),
            "RS_MIN_PERCENTILE": int(self.rs_min_percentile),
            "N_WORKERS": int(max(1, scout_workers)),
            "SCOUT_CLEAN_OUTPUT_DIR": True,
        }


def _build_grid(
    don_days_values: List[int],
    retest_eps_values: List[float],
    retest_lookback_values: List[int],
    pullback_window_hours_values: List[int],
    vol_multiple_values: List[float],
    rs_percentile_values: List[int],
    max_variants: int,
) -> List[EntryVariant]:
    out: List[EntryVariant] = []
    for dn in don_days_values:
        for eps in retest_eps_values:
            for lb in retest_lookback_values:
                for pwh in pullback_window_hours_values:
                    for vm in vol_multiple_values:
                        for rs in rs_percentile_values:
                            name = (
                                f"dn{dn:02d}__eps{int(round(eps*10000)):02d}"
                                f"__lb{lb:03d}__pwh{pwh:03d}__vm{int(round(vm*100)):03d}__rs{rs:02d}"
                            )
                            out.append(
                                EntryVariant(
                                    name=name,
                                    don_n_days=int(dn),
                                    retest_eps_pct=float(eps),
                                    retest_lookback_bars=int(lb),
                                    pullback_window_hours=int(pwh),
                                    vol_multiple=float(vm),
                                    rs_min_percentile=int(rs),
                                )
                            )
    if max_variants > 0:
        return out[: int(max_variants)]
    return out


def _run_scout_subprocess(
    *,
    signals_dir: Path,
    start: str,
    end: str,
    overrides: Dict[str, Any],
    log_path: Path,
) -> int:
    signals_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DONCH_REPO_ROOT"] = str(REPO_ROOT)
    env["DONCH_SIGNALS_DIR"] = str(signals_dir.resolve())
    env["DONCH_START"] = str(start)
    env["DONCH_END"] = str(end)
    env["DONCH_OVERRIDES_JSON"] = json.dumps(overrides, sort_keys=True)
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        p = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD_SCOUT_CODE],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = p.wait()
    return int(rc)


def _run_walkforward(trades_path: Path, wf_root: Path, wf_run_id: str, py: str) -> Tuple[int, Path]:
    wf_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        str(REPO_ROOT / "tools" / "run_walkforward_oos_validation.py"),
        "--trades",
        str(trades_path),
        "--outdir",
        str(wf_root),
        "--run-id",
        wf_run_id,
        "--train-months",
        "6",
        "--valid-months",
        "1",
        "--step-months",
        "1",
        "--train-scope",
        "ALL",
        "--fit-scope",
        "ALL",
        "--target",
        "y_good_05",
        "--resume",
    ]
    log_path = wf_root / f"{wf_run_id}.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = p.wait()
    return int(rc), log_path


def _run_stability_eval(wf_root: Path, wf_run_id: str, py: str) -> Tuple[int, Path, Optional[dict]]:
    cmd = [
        py,
        str(REPO_ROOT / "tools" / "evaluate_walkforward_stability.py"),
        "--root",
        str(wf_root),
        "--run-id",
        wf_run_id,
    ]
    log_path = wf_root / f"{wf_run_id}.jt011_eval.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = p.wait()
    verdict_path = wf_root / wf_run_id / "aggregate" / "stability_verdict.json"
    verdict = None
    if verdict_path.exists():
        try:
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        except Exception:
            verdict = None
    return int(rc), verdict_path, verdict


def _find_trades_file(run_dir: Path) -> Path:
    for cand in [run_dir / "trades.csv", run_dir / "trades.clean.csv", run_dir / "trades.parquet"]:
        if cand.exists():
            return cand
    hits = sorted(run_dir.rglob("trades.csv"))
    if hits:
        return hits[0]
    raise RuntimeError(f"No trades file found under {run_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JT-016 exploratory entry sweep with JT-011 walk-forward checks.")
    p.add_argument("--start", type=_parse_date, default="2023-01-01")
    p.add_argument("--end", type=_parse_date, default="latest")
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--on-missing-symbol", choices=["fail", "skip"], default="skip")
    p.add_argument("--smoke-n", type=int, default=0)

    p.add_argument("--don-days-values", type=str, default="14,20")
    p.add_argument("--retest-eps-values", type=str, default="0.0025,0.0035")
    p.add_argument("--retest-lookback-values", type=str, default="192,288")
    p.add_argument("--pullback-window-hours-values", type=str, default="18,24")
    p.add_argument("--vol-multiple-values", type=str, default="1.8,2.0")
    p.add_argument("--rs-percentile-values", type=str, default="60,70")
    p.add_argument("--max-variants", type=int, default=24)

    p.add_argument("--rolling-trades-window", type=int, default=250)
    p.add_argument("--initial-capital", type=float, default=2000.0)
    p.add_argument("--scout-workers", type=int, default=2)
    p.add_argument("--variant-retries", type=int, default=1)
    p.add_argument("--meta-mode", choices=["off", "size_only", "full"], default="off")
    p.add_argument("--meta-threshold", type=float, default=None)
    p.add_argument("--window-file", type=str, default="")
    p.add_argument("--seed-signals-root", type=str, default="")
    p.add_argument("--policy-neutral", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--policy-block-when-down", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--policy-size-when-down", type=float, default=None)
    p.add_argument("--policy-probe-mult", type=float, default=None)

    p.add_argument("--jt011-top-k", type=int, default=3)
    p.add_argument("--python", type=str, default=sys.executable)

    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument("--tg-auto-chat", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    cfg = import_cfg()
    _signals_dir, _parquet_dir, results_dir, _meta_model_dir = resolve_paths(cfg)

    rid = _utc_run_id(a.run_id)
    notifier = TelegramNotifier.from_args(a, run_label=f"jt016:{rid}")
    print(f"[jt016] telegram notify: {notifier.status_line()}", flush=True)

    run_root = (results_dir / "jt016_entry_sweeps" / rid).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    end = a.end
    if str(end).lower() == "latest":
        end = latest_from_signals(_signals_dir)

    windows = []
    if str(a.window_file).strip():
        windows = load_windows_file(Path(str(a.window_file).strip()))
        if not windows:
            raise RuntimeError(f"window file parsed to zero windows: {a.window_file}")
    seed_root = Path(str(a.seed_signals_root).strip()).resolve() if str(a.seed_signals_root).strip() else None
    if seed_root is not None and (not seed_root.exists()):
        raise FileNotFoundError(f"seed signals root not found: {seed_root}")

    grid = _build_grid(
        don_days_values=_parse_int_list(a.don_days_values),
        retest_eps_values=_parse_float_list(a.retest_eps_values),
        retest_lookback_values=_parse_int_list(a.retest_lookback_values),
        pullback_window_hours_values=_parse_int_list(a.pullback_window_hours_values),
        vol_multiple_values=_parse_float_list(a.vol_multiple_values),
        rs_percentile_values=_parse_int_list(a.rs_percentile_values),
        max_variants=int(a.max_variants),
    )

    notifier.send(
        "STARTED",
        body=(
            f"run_id={rid}\nstart={a.start} end={end}\n"
            f"variants={len(grid)} scout_workers={int(a.scout_workers)}\n"
            f"jt011_top_k={int(a.jt011_top_k)}\n"
            f"meta_mode={a.meta_mode} meta_threshold={a.meta_threshold}\n"
            f"policy_neutral={bool(a.policy_neutral)} windows={len(windows)}\n"
            f"seed_signals_root={(str(seed_root) if seed_root is not None else 'none')}"
        ),
    )

    progress = {"done": 0, "ok": 0, "err": 0}
    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    for idx, gp in enumerate(grid, start=1):
        vdir = run_root / gp.name
        scout_dir = vdir / "signals"
        bt_dir = vdir / "backtest"
        vdir.mkdir(parents=True, exist_ok=True)

        done_path = vdir / "_DONE.json"
        metrics_path = vdir / "metrics.json"
        if done_path.exists() and metrics_path.exists():
            try:
                prev = json.loads(metrics_path.read_text(encoding="utf-8"))
                if str(prev.get("status", "")).lower() == "ok":
                    rows.append(prev)
                    progress["done"] += 1
                    progress["ok"] += 1
                    continue
            except Exception:
                pass

        variant_status = "error"
        out: Dict[str, Any] = {
            "setting": gp.name,
            "status": "error",
            "returncode": 999,
            "don_n_days": gp.don_n_days,
            "retest_eps_pct": gp.retest_eps_pct,
            "retest_lookback_bars": gp.retest_lookback_bars,
            "pullback_window_hours": gp.pullback_window_hours,
            "vol_multiple": gp.vol_multiple,
            "rs_min_percentile": gp.rs_min_percentile,
        }

        max_attempts = max(1, int(a.variant_retries) + 1)
        attempts: List[Dict[str, Any]] = []
        for attempt in range(max_attempts):
            try:
                seed_used = False
                scout_input_dir = scout_dir
                if seed_root is not None:
                    cand = seed_root / gp.name / "signals"
                    if (cand / "signals.parquet").exists() or any(cand.rglob("*.parquet")):
                        seed_used = True
                        scout_input_dir = cand

                if not seed_used:
                    if scout_dir.exists():
                        shutil.rmtree(scout_dir, ignore_errors=True)
                    scout_dir.mkdir(parents=True, exist_ok=True)
                bt_dir.mkdir(parents=True, exist_ok=True)

                if not seed_used:
                    scout_overrides = gp.scout_overrides(int(a.scout_workers))
                    scout_rc = _run_scout_subprocess(
                        signals_dir=scout_dir,
                        start=str(a.start),
                        end=str(end),
                        overrides=scout_overrides,
                        log_path=vdir / f"scout.attempt_{attempt}.log",
                    )
                    if scout_rc != 0:
                        attempts.append({"attempt": attempt, "stage": "scout", "returncode": int(scout_rc)})
                        out["returncode"] = int(scout_rc)
                        continue
                else:
                    attempts.append(
                        {
                            "attempt": attempt,
                            "stage": "scout_seed",
                            "returncode": 0,
                            "seed_dir": str(scout_input_dir),
                        }
                    )

                bt_overrides = {
                    "BT_PROGRESS_ENABLED": False,
                    "BT_DECISION_LOG_ENABLED": False,
                    "BT_META_REPLAY_ENABLED": False,
                    "USE_INTRABAR_1M": False,
                    **_meta_mode_overrides(a.meta_mode, a.meta_threshold),
                }
                if bool(a.policy_neutral):
                    bt_overrides.update(
                        {
                            "REGIME_BLOCK_WHEN_DOWN": False,
                            "REGIME_SIZE_WHEN_DOWN": 1.0,
                            "RISK_OFF_PROBE_MULT": 1.0,
                        }
                    )
                if a.policy_block_when_down is not None:
                    bt_overrides["REGIME_BLOCK_WHEN_DOWN"] = bool(a.policy_block_when_down)
                if a.policy_size_when_down is not None:
                    bt_overrides["REGIME_SIZE_WHEN_DOWN"] = float(a.policy_size_when_down)
                if a.policy_probe_mult is not None:
                    bt_overrides["RISK_OFF_PROBE_MULT"] = float(a.policy_probe_mult)

                bt_signals_dir = scout_input_dir
                scoped_n = None
                if windows:
                    scoped = _load_signals_with_symbol(scout_input_dir, str(a.start), str(end))
                    scoped = filter_df_by_windows(scoped, windows, ts_col="timestamp")
                    scoped_n = int(len(scoped))
                    bt_signals_dir = write_signals_file(
                        scoped,
                        vdir / f"_bt_scoped_signals_attempt_{attempt}",
                    )
                rc_bt, _ = run_backtest_subprocess(
                    run_dir=bt_dir,
                    signals_dir=bt_signals_dir,
                    start=str(a.start),
                    end=str(end),
                    overrides=bt_overrides,
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "stage": "backtest",
                        "returncode": int(rc_bt),
                        "scoped_signals_n": scoped_n,
                        "signals_dir": str(bt_signals_dir),
                        "meta_mode": str(a.meta_mode),
                    }
                )
                out["returncode"] = int(rc_bt)
                if int(rc_bt) != 0:
                    continue

                try:
                    met, win_df = _compute_variant_metrics(
                        run_dir=bt_dir,
                        rolling_trades_window=int(a.rolling_trades_window),
                        initial_capital=float(a.initial_capital),
                    )
                except RuntimeError as exc:
                    if "No trades file found under" in str(exc):
                        met, win_df = _empty_metrics_payload("no_trades_file_no_executions"), pd.DataFrame()
                    else:
                        raise
                if not win_df.empty:
                    win_df.to_csv(vdir / "window_metrics.csv", index=False)

                met.update(
                    {
                        "setting": gp.name,
                        "status": "ok",
                        "returncode": 0,
                        "don_n_days": gp.don_n_days,
                        "retest_eps_pct": gp.retest_eps_pct,
                        "retest_lookback_bars": gp.retest_lookback_bars,
                        "pullback_window_hours": gp.pullback_window_hours,
                        "vol_multiple": gp.vol_multiple,
                        "rs_min_percentile": gp.rs_min_percentile,
                        "meta_mode": str(a.meta_mode),
                        "seed_signals_root": (str(seed_root) if seed_root is not None else None),
                        "window_file": (str(a.window_file).strip() or None),
                        "policy_neutral": bool(a.policy_neutral),
                        "attempts": attempts,
                    }
                )
                out = met
                variant_status = "ok"
                break
            except Exception as exc:
                attempts.append({"attempt": attempt, "stage": "exception", "error": f"{type(exc).__name__}: {exc}"})
                out["error"] = f"{type(exc).__name__}: {exc}"

        out["attempts"] = attempts
        metrics_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        done_path.write_text(json.dumps({"returncode": int(out.get("returncode", 999))}, indent=2), encoding="utf-8")
        rows.append(out)

        progress["done"] += 1
        if variant_status == "ok":
            progress["ok"] += 1
        else:
            progress["err"] += 1

        elapsed_min = max((time.time() - t0) / 60.0, 1e-9)
        rate = progress["done"] / elapsed_min
        eta_min = (len(grid) - progress["done"]) / rate if rate > 0 else float("nan")
        notifier.send(
            "PROGRESS",
            body=(
                f"run_id={rid}\nstage=entry_sweep\n"
                f"done={progress['done']}/{len(grid)} ok={progress['ok']} err={progress['err']}\n"
                f"last={gp.name} status={variant_status}\n"
                f"elapsed_min={elapsed_min:.1f} eta_min={(f'{eta_min:.1f}' if np.isfinite(eta_min) else 'n/a')}"
            ),
        )

    summary = pd.DataFrame(rows)
    summary_path = run_root / "summary.csv"
    summary.to_csv(summary_path, index=False)

    ok = summary[summary["status"] == "ok"].copy() if "status" in summary.columns else pd.DataFrame()
    if ok.empty:
        notifier.send("FAILED", body=f"run_id={rid}\nno successful variants\nsummary={summary_path}")
        print(str(run_root))
        return 1

    # Stage-2: JT-011 checks on top-K by total_pnl_cash
    ok["total_pnl_cash"] = pd.to_numeric(ok.get("total_pnl_cash"), errors="coerce")
    topk = ok.sort_values("total_pnl_cash", ascending=False).head(max(0, int(a.jt011_top_k))).copy()
    wf_rows: List[Dict[str, Any]] = []

    notifier.send(
        "STAGE",
        body=f"run_id={rid}\nstage=jt011_checks\nselected_variants={len(topk)}",
    )

    for i, row in topk.iterrows():
        setting = str(row.get("setting"))
        vdir = run_root / setting
        bt_dir = vdir / "backtest"
        wf_root = vdir / "walkforward_oos"
        wf_run_id = f"{setting}_wf"

        out = {
            "setting": setting,
            "jt011_status": "error",
            "jt011_verdict": None,
            "jt011_failed_gates": None,
            "jt011_recommended_live_risk_mode": None,
        }
        try:
            trades_path = _find_trades_file(bt_dir)
            rc_wf, wf_log = _run_walkforward(trades_path=trades_path, wf_root=wf_root, wf_run_id=wf_run_id, py=str(a.python))
            out["jt011_wf_returncode"] = int(rc_wf)
            out["jt011_wf_log"] = str(wf_log)
            if rc_wf == 0:
                rc_eval, verdict_path, verdict = _run_stability_eval(wf_root=wf_root, wf_run_id=wf_run_id, py=str(a.python))
                out["jt011_eval_returncode"] = int(rc_eval)
                out["jt011_verdict_path"] = str(verdict_path)
                if rc_eval == 0 and isinstance(verdict, dict):
                    out["jt011_status"] = "ok"
                    out["jt011_verdict"] = str(verdict.get("verdict"))
                    out["jt011_failed_gates"] = ",".join([str(x) for x in verdict.get("failed_gates", [])])
                    out["jt011_recommended_live_risk_mode"] = str(verdict.get("recommended_live_risk_mode"))
        except Exception as exc:
            out["jt011_error"] = f"{type(exc).__name__}: {exc}"

        wf_rows.append(out)
        notifier.send(
            "PROGRESS",
            body=(
                f"run_id={rid}\nstage=jt011_checks\nsetting={setting}\n"
                f"status={out.get('jt011_status')} verdict={out.get('jt011_verdict')}"
            ),
        )

    wf_df = pd.DataFrame(wf_rows)
    wf_path = run_root / "jt011_eval.csv"
    wf_df.to_csv(wf_path, index=False)

    # Final recommendation: prefer variants with jt011 pass; fallback to highest total_pnl_cash
    merged = ok.copy()
    if not wf_df.empty:
        merged = merged.merge(wf_df, on="setting", how="left")
    if "jt011_verdict" not in merged.columns:
        merged["jt011_verdict"] = np.nan
    merged["jt011_pass"] = merged["jt011_verdict"].astype(str).str.lower().eq("pass")
    merged = merged.sort_values(["jt011_pass", "total_pnl_cash"], ascending=[False, False], kind="mergesort")
    selected = merged.iloc[0]

    rec = {
        "run_id": rid,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "start": a.start,
            "end": end,
            "variants_total": int(len(grid)),
            "variants_ok": int(len(ok)),
            "variants_error": int(len(grid) - len(ok)),
            "jt011_top_k": int(a.jt011_top_k),
        },
        "recommended_setting": str(selected.get("setting")),
        "recommended_metrics": selected.to_dict(),
        "selection_rule": "jt011_pass_then_total_pnl_cash",
    }
    rec_path = run_root / "recommendation.json"
    rec_path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# JT-016 Entry Exploratory Sweep: {rid}")
    lines.append("")
    lines.append("## Runtime")
    lines.append(f"- variants_ok: `{len(ok)}/{len(grid)}`")
    lines.append(f"- variants_error: `{len(grid)-len(ok)}`")
    lines.append(f"- jt011_top_k: `{int(a.jt011_top_k)}`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- setting: `{selected.get('setting')}`")
    lines.append(f"- jt011_verdict: `{selected.get('jt011_verdict')}`")
    lines.append(f"- total_pnl_cash: `{selected.get('total_pnl_cash')}`")
    lines.append("")
    top_cols = [
        "setting",
        "total_pnl_cash",
        "sharpe_daily",
        "calmar",
        "risk_off_total_pnl_R",
        "risk_off_worst_month_mean_pnl_R",
        "risk_off_worst_window_mean_pnl_R",
        "jt011_verdict",
        "jt011_failed_gates",
    ]
    for c in top_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    lines.append("## Ranked")
    lines.append("")
    lines.append(merged[top_cols].head(20).to_markdown(index=False))
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    notifier.send(
        "DONE",
        body=(
            f"run_id={rid}\nstatus=ok\n"
            f"recommended={selected.get('setting')}\n"
            f"jt011_verdict={selected.get('jt011_verdict')}\n"
            f"recommendation={rec_path}"
        ),
    )
    print(str(run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
