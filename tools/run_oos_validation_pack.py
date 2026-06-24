#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


SCOUT_CODE = r"""
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["DONCH_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))

import config as cfg
from scout import run_scout

cfg.PARQUET_DIR = Path(os.environ["DONCH_PARQUET_DIR"]).resolve()
cfg.SIGNALS_DIR = Path(os.environ["DONCH_SIGNALS_DIR"]).resolve()
cfg.RESULTS_DIR = (cfg.SIGNALS_DIR / "_aux_results").resolve()
cfg.START_DATE = str(os.environ["DONCH_START"])
cfg.END_DATE = str(os.environ["DONCH_END"])
cfg.N_WORKERS = int(os.environ.get("DONCH_SCOUT_WORKERS", "2"))
cfg.SCOUT_BACKEND = str(os.environ.get("DONCH_SCOUT_BACKEND", "thread"))
cfg.SCOUT_CLEAN_OUTPUT_DIR = bool(int(os.environ.get("DONCH_SCOUT_CLEAN", "1")))

cfg.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

n = run_scout()
print(f"[oos] scout_done rows={int(n)} signals_dir={cfg.SIGNALS_DIR}", flush=True)
"""


BACKTEST_CODE = r"""
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["DONCH_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))

import config as cfg
import backtester

cfg.PARQUET_DIR = Path(os.environ["DONCH_PARQUET_DIR"]).resolve()
cfg.SIGNALS_DIR = Path(os.environ["DONCH_SIGNALS_DIR"]).resolve()
cfg.RESULTS_DIR = Path(os.environ["DONCH_OUT_DIR"]).resolve()
cfg.START_DATE = str(os.environ["DONCH_START"])
cfg.END_DATE = str(os.environ["DONCH_END"])
cfg.META_MODEL_DIR = Path(os.environ["DONCH_META_MODEL_DIR"]).resolve()

parq1m = os.environ.get("DONCH_PARQUET_1M_DIR", "").strip()
if parq1m:
    cfg.PARQUET_1M_DIR = Path(parq1m).resolve()

for k, v in json.loads(os.environ["DONCH_OVERRIDES_JSON"]).items():
    setattr(cfg, k, v)

cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
backtester.run_backtest(signals_path=cfg.SIGNALS_DIR)
"""


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: Dict[str, object]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run frozen OOS validation pack (shared signals + multiple backtest variants)."
    )
    p.add_argument("--run-id", default="", help="Sweep-like run id under results/policy_sweeps.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--parquet-dir", default="/opt/parquet/5m")
    p.add_argument("--parquet-1m-dir", default="")
    p.add_argument("--model-dir", default="results/offline_releases/20260209_meta_release_v1/meta_export_pstar_042")
    p.add_argument("--start", default="2025-11-16")
    p.add_argument("--end", default="2026-02-08")
    p.add_argument("--pstar", type=float, default=0.42)
    p.add_argument("--scout-workers", type=int, default=2)
    p.add_argument("--scout-backend", choices=["thread", "process"], default="thread")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-scout", action="store_true")
    return p.parse_args()


def _resolve_under_repo(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p.resolve()


def _run_cmd(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[oos] running: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return int(proc.wait())


def _variants(pstar: float) -> List[Variant]:
    common = {
        "BT_META_REPLAY_ENABLED": False,
        "BT_DECISION_LOG_ENABLED": False,
        "BT_PROGRESS_ENABLED": False,
        "REGIME_BLOCK_WHEN_DOWN": False,
        "RISK_OFF_PROBE_MULT": 0.05,
        "REGIME_SLOPE_FILTER_ENABLED": False,
        "META_GATE_SCOPE": "all",
        "META_GATE_FAIL_CLOSED": False,
    }
    return [
        Variant(
            name="core_no_meta",
            overrides={
                **common,
                "META_PROB_THRESHOLD": None,
                "META_SIZING_ENABLED": False,
                "BT_META_ONLINE_ENABLED": False,
            },
        ),
        Variant(
            name="meta_size_only_no_gate",
            overrides={
                **common,
                "META_PROB_THRESHOLD": None,
                "META_SIZING_ENABLED": True,
                "BT_META_ONLINE_ENABLED": True,
                "META_STRICT_SCHEMA": True,
            },
        ),
        Variant(
            name="meta_gate_pstar_no_size",
            overrides={
                **common,
                "META_PROB_THRESHOLD": float(pstar),
                "META_SIZING_ENABLED": False,
                "BT_META_ONLINE_ENABLED": True,
                "META_STRICT_SCHEMA": True,
            },
        ),
        Variant(
            name="meta_gate_pstar_with_size",
            overrides={
                **common,
                "META_PROB_THRESHOLD": float(pstar),
                "META_SIZING_ENABLED": True,
                "BT_META_ONLINE_ENABLED": True,
                "META_STRICT_SCHEMA": True,
            },
        ),
    ]


def main() -> int:
    args = _parse_args()
    rid = args.run_id.strip() or f"oos_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    results_root = _resolve_under_repo(args.results_dir)
    sweep_root = (results_root / "policy_sweeps" / rid).resolve()
    signals_dir = (sweep_root / "_scoped_signals").resolve()
    stage_file = sweep_root / "_STAGE.txt"
    stage_file.parent.mkdir(parents=True, exist_ok=True)

    parquet_dir = Path(args.parquet_dir).expanduser().resolve()
    if not parquet_dir.exists():
        raise SystemExit(f"Missing --parquet-dir: {parquet_dir}")

    model_dir = _resolve_under_repo(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Missing --model-dir: {model_dir}")

    env_base = os.environ.copy()
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base.setdefault("MALLOC_ARENA_MAX", "2")
    env_base.setdefault("OMP_NUM_THREADS", "1")
    env_base.setdefault("OPENBLAS_NUM_THREADS", "1")
    env_base.setdefault("MKL_NUM_THREADS", "1")
    env_base.setdefault("NUMEXPR_NUM_THREADS", "1")
    env_base["DONCH_REPO_ROOT"] = str(REPO_ROOT)
    env_base["DONCH_PARQUET_DIR"] = str(parquet_dir)
    env_base["DONCH_PARQUET_1M_DIR"] = str(args.parquet_1m_dir).strip()
    env_base["DONCH_SIGNALS_DIR"] = str(signals_dir)
    env_base["DONCH_START"] = str(args.start)
    env_base["DONCH_END"] = str(args.end)
    env_base["DONCH_SCOUT_WORKERS"] = str(max(1, int(args.scout_workers)))
    env_base["DONCH_SCOUT_BACKEND"] = str(args.scout_backend).strip()
    env_base["DONCH_META_MODEL_DIR"] = str(model_dir)

    signals_ready = signals_dir.exists() and any(signals_dir.glob("symbol=*"))
    if not args.skip_scout and not (args.resume and signals_ready):
        stage_file.write_text("SCOUT_RUNNING\n", encoding="utf-8")
        scout_env = dict(env_base)
        scout_env["DONCH_SCOUT_CLEAN"] = "1"
        rc = _run_cmd(
            [args.python, "-u", "-c", SCOUT_CODE],
            log_path=sweep_root / "_scout.log",
            env=scout_env,
        )
        if rc != 0:
            stage_file.write_text("FAILED_SCOUT\n", encoding="utf-8")
            raise SystemExit(f"Scout failed rc={rc}. See {sweep_root / '_scout.log'}")
    else:
        print("[oos] resume: reusing existing _scoped_signals", flush=True)

    if not signals_dir.exists() or not any(signals_dir.glob("symbol=*")):
        stage_file.write_text("FAILED_NO_SIGNALS\n", encoding="utf-8")
        raise SystemExit(f"No partitioned signals under {signals_dir}")

    stage_file.write_text("BACKTEST_RUNNING\n", encoding="utf-8")
    run_info = {
        "run_id": rid,
        "start": str(args.start),
        "end": str(args.end),
        "parquet_dir": str(parquet_dir),
        "model_dir": str(model_dir),
        "pstar": float(args.pstar),
        "variants": [],
    }
    (sweep_root / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    for v in _variants(float(args.pstar)):
        out_dir = sweep_root / v.name
        done_file = out_dir / "_DONE.json"
        if args.resume and done_file.exists():
            try:
                d = json.loads(done_file.read_text(encoding="utf-8"))
                if int(d.get("returncode", 1)) == 0 and (out_dir / "trades.csv").exists():
                    print(f"[oos] resume: skipping {v.name}", flush=True)
                    continue
            except Exception:
                pass

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "start": str(args.start),
                    "end": str(args.end),
                    "signals_dir": str(signals_dir),
                    "overrides": v.overrides,
                    "model_dir": str(model_dir),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        bt_env = dict(env_base)
        bt_env["DONCH_OUT_DIR"] = str(out_dir.resolve())
        bt_env["DONCH_OVERRIDES_JSON"] = json.dumps(v.overrides, sort_keys=True)
        rc = _run_cmd(
            [args.python, "-u", "-c", BACKTEST_CODE],
            log_path=out_dir / "logs.txt",
            env=bt_env,
        )
        done_file.write_text(
            json.dumps({"setting": v.name, "returncode": int(rc)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if rc != 0:
            stage_file.write_text("FAILED_BACKTEST\n", encoding="utf-8")
            raise SystemExit(f"Variant {v.name} failed rc={rc}. See {out_dir / 'logs.txt'}")

    stage_file.write_text("METRICS_RUNNING\n", encoding="utf-8")
    for cmd, log_name in [
        (
            [
                args.python,
                str((REPO_ROOT / "tools" / "recompute_sweep_metrics.py").resolve()),
                "--run-id",
                rid,
                "--results-dir",
                str(results_root.name),
                "--meta-model-dir",
                str(model_dir),
                "--use-pnl-col",
                "pnl_R",
            ],
            "_recompute_metrics.log",
        ),
        (
            [
                args.python,
                str((REPO_ROOT / "tools" / "render_sweep_report.py").resolve()),
                "--run-id",
                rid,
                "--results-dir",
                str(results_root.name),
                "--topn",
                "20",
            ],
            "_render_summary.log",
        ),
        (
            [
                args.python,
                str((REPO_ROOT / "tools" / "render_run_risk_report.py").resolve()),
                "--run-id",
                rid,
                "--results-dir",
                str(results_root.name),
                "--topn-table",
                "20",
                "--topn-plot",
                "8",
            ],
            "_render_risk_report.log",
        ),
    ]:
        rc = _run_cmd(cmd, log_path=sweep_root / log_name, env=env_base)
        if rc != 0:
            stage_file.write_text("FAILED_METRICS\n", encoding="utf-8")
            raise SystemExit(f"Metrics/report stage failed rc={rc}. See {sweep_root / log_name}")

    stage_file.write_text("DONE\n", encoding="utf-8")
    print(f"[oos] DONE run_id={rid}", flush=True)
    print(f"[oos] root={sweep_root}", flush=True)
    print(f"[oos] summary={sweep_root / 'summary.csv'}", flush=True)
    print(f"[oos] risk_report={sweep_root / 'risk_report' / 'report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
