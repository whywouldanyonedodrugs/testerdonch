#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from sweep_policy_settings_v2 import import_cfg, latest_from_signals, resolve_paths  # type: ignore
from telegram_notify import TelegramNotifier


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_date(s: str) -> str:
    if s.lower() == "latest":
        return "latest"
    datetime.strptime(s, "%Y-%m-%d")
    return s


def _utc_run_id(user_id: str) -> str:
    return user_id.strip() or datetime.now(timezone.utc).strftime("jt_pipeline_%Y%m%d_%H%M%S")


def _run_cmd(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        p = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return int(p.wait())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential runner for JT-014b -> JT-015 -> JT-016 with Telegram stage notifications.")
    p.add_argument("--start", type=_parse_date, default="2023-01-01")
    p.add_argument("--end", type=_parse_date, default="latest")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--run-id", type=str, default="")

    # Stage controls
    p.add_argument("--skip-14b", action="store_true")
    p.add_argument("--skip-15", action="store_true")
    p.add_argument("--skip-16", action="store_true")

    # 14b
    p.add_argument("--jt014b-smoke-n", type=int, default=0)
    p.add_argument("--jt014b-variant-retries", type=int, default=2)

    # 15 (policy grid)
    p.add_argument("--jt015-pstar", type=float, default=0.42)
    p.add_argument("--jt015-smoke-n", type=int, default=0)

    # 16 (entry exploratory)
    p.add_argument("--jt016-max-variants", type=int, default=24)
    p.add_argument("--jt016-top-k", type=int, default=3)
    p.add_argument("--jt016-scout-workers", type=int, default=2)

    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument("--tg-auto-chat", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    rid = _utc_run_id(a.run_id)
    notifier = TelegramNotifier.from_args(a, run_label=f"jt-pipeline:{rid}")
    print(f"[jt-pipeline] telegram notify: {notifier.status_line()}", flush=True)

    cfg = import_cfg()
    _signals_dir, _parquet_dir, results_dir, _meta_model_dir = resolve_paths(cfg)

    end = a.end
    if str(end).lower() == "latest":
        end = latest_from_signals(_signals_dir)

    root = (results_dir / "jt_pipeline_runs" / rid).resolve()
    root.mkdir(parents=True, exist_ok=True)

    stages: List[Tuple[str, List[str]]] = []

    if not bool(a.skip_14b):
        cmd14 = [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_jt014_regime_conditioned_exits.py"),
            "--start",
            str(a.start),
            "--end",
            str(end),
            "--jobs",
            str(max(1, int(a.jobs))),
            "--run-id",
            f"{rid}_jt014b",
            "--preset-mode",
            "mechanics",
            "--variant-retries",
            str(max(0, int(a.jt014b_variant_retries))),
            "--policy-probe-mult",
            "0.25",
            "--no-policy-block-when-down",
            "--tg-auto-chat",
        ]
        if int(a.jt014b_smoke_n) > 0:
            cmd14 += ["--smoke-n", str(int(a.jt014b_smoke_n))]
        if str(a.tg_bot_token).strip():
            cmd14 += ["--tg-bot-token", str(a.tg_bot_token).strip()]
        if str(a.tg_chat_id).strip():
            cmd14 += ["--tg-chat-id", str(a.tg_chat_id).strip()]
        stages.append(("jt014b", cmd14))

    if not bool(a.skip_15):
        cmd15 = [
            sys.executable,
            str(REPO_ROOT / "tools" / "sweep_policy_settings_v2.py"),
            "--start",
            str(a.start),
            "--end",
            str(end),
            "--jobs",
            str(max(1, int(a.jobs))),
            "--run-id",
            f"{rid}_jt015",
            "--pstar",
            f"{float(a.jt015_pstar):.6g}",
        ]
        if int(a.jt015_smoke_n) > 0:
            cmd15 += ["--smoke-n", str(int(a.jt015_smoke_n))]
        else:
            cmd15 += ["--smoke-n", "0"]
        stages.append(("jt015", cmd15))

    if not bool(a.skip_16):
        cmd16 = [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_jt016_entry_exploratory.py"),
            "--start",
            str(a.start),
            "--end",
            str(end),
            "--run-id",
            f"{rid}_jt016",
            "--max-variants",
            str(max(1, int(a.jt016_max_variants))),
            "--jt011-top-k",
            str(max(0, int(a.jt016_top_k))),
            "--scout-workers",
            str(max(1, int(a.jt016_scout_workers))),
            "--tg-auto-chat",
        ]
        if str(a.tg_bot_token).strip():
            cmd16 += ["--tg-bot-token", str(a.tg_bot_token).strip()]
        if str(a.tg_chat_id).strip():
            cmd16 += ["--tg-chat-id", str(a.tg_chat_id).strip()]
        stages.append(("jt016", cmd16))

    if not stages:
        print("[jt-pipeline] nothing to run: all stages skipped")
        return 0

    notifier.send(
        "STARTED",
        body=(
            f"pipeline_run_id={rid}\nstart={a.start} end={end}\n"
            f"stages={','.join([s for s,_ in stages])}\nroot={root}"
        ),
    )

    stage_status = []
    t_all = time.time()
    for idx, (name, cmd) in enumerate(stages, start=1):
        t0 = time.time()
        stage_log = root / "logs" / f"{idx:02d}_{name}.log"
        (root / "_STAGE.json").write_text(
            json.dumps(
                {
                    "stage_index": idx,
                    "stage_name": name,
                    "status": "running",
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "cmd": cmd,
                    "log": str(stage_log),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        notifier.send(
            "STAGE_START",
            body=(
                f"pipeline_run_id={rid}\nstage={idx}/{len(stages)}:{name}\n"
                f"log={stage_log}"
            ),
        )
        rc = _run_cmd(cmd, stage_log)
        dt_min = (time.time() - t0) / 60.0
        stage_status.append({"stage": name, "returncode": int(rc), "elapsed_min": float(dt_min), "log": str(stage_log)})

        if rc != 0:
            (root / "_STAGE.json").write_text(
                json.dumps(
                    {
                        "stage_index": idx,
                        "stage_name": name,
                        "status": "failed",
                        "returncode": int(rc),
                        "elapsed_min": float(dt_min),
                        "log": str(stage_log),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (root / "pipeline_status.json").write_text(
                json.dumps(
                    {
                        "pipeline_run_id": rid,
                        "status": "failed",
                        "failed_stage": name,
                        "stage_status": stage_status,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            notifier.send(
                "FAILED",
                body=(
                    f"pipeline_run_id={rid}\nfailed_stage={name}\n"
                    f"returncode={rc}\nelapsed_min={dt_min:.1f}\nlog={stage_log}"
                ),
            )
            print(str(root))
            return int(rc)

        notifier.send(
            "STAGE_DONE",
            body=(
                f"pipeline_run_id={rid}\nstage={idx}/{len(stages)}:{name}\n"
                f"returncode=0\nelapsed_min={dt_min:.1f}\nlog={stage_log}"
            ),
        )

    total_min = (time.time() - t_all) / 60.0
    final = {
        "pipeline_run_id": rid,
        "status": "ok",
        "elapsed_min": float(total_min),
        "stage_status": stage_status,
    }
    (root / "pipeline_status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    notifier.send(
        "DONE",
        body=(
            f"pipeline_run_id={rid}\nstatus=ok\nelapsed_min={total_min:.1f}\n"
            f"root={root}"
        ),
    )
    print(str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
