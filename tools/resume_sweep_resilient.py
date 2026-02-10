#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from telegram_notify import TelegramNotifier


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resume a policy sweep until all variants are successful, then optionally enrich."
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="latest")
    p.add_argument("--jobs", type=int, default=1, help="Sweep parallel jobs (use 1 for OOM safety).")
    p.add_argument("--pstar", type=float, default=0.64)
    p.add_argument("--lambda", dest="lam", type=float, default=2.0)
    p.add_argument("--mu", dest="mu", type=float, default=1.0)
    p.add_argument("--on-missing-symbol", choices=["fail", "skip"], default="skip")
    p.add_argument("--smoke-n", type=int, default=0)

    p.add_argument("--sleep-sec", type=float, default=30.0, help="Sleep between failed/incomplete attempts.")
    p.add_argument("--max-attempts", type=int, default=0, help="0 means infinite retries.")

    p.add_argument("--then-enrich", action="store_true")
    p.add_argument("--enrich-jobs", type=int, default=1)
    p.add_argument("--enrich-retries", type=int, default=2)
    p.add_argument("--funding-stage", choices=["on", "off", "optional"], default="optional")
    p.add_argument("--funding-throttle", type=float, default=0.0)

    p.add_argument("--python", default=sys.executable)
    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument(
        "--tg-auto-chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try Telegram getUpdates to auto-discover chat id when not provided.",
    )
    return p.parse_args()


def run(cmd: List[str], env: Dict[str, str]) -> int:
    print(f"[resilient] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return int(proc.returncode)


def read_summary(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: List[Dict[str, str]], expected: int) -> Dict[str, int]:
    n = len(rows)
    ok = 0
    err = 0
    other = 0
    for r in rows:
        st = str(r.get("status", "")).strip().lower()
        if st == "ok":
            ok += 1
        elif st == "error":
            err += 1
        else:
            other += 1
    missing = max(expected - n, 0)
    return {"rows": n, "ok": ok, "error": err, "other": other, "missing": missing}


def expected_variant_count(pstar: float) -> int:
    # Import from the live sweep module to stay in sync with grid definition.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.sweep_policy_settings_v2 import grid  # type: ignore

    return len(grid(float(pstar)))


def build_low_mem_env(base: Dict[str, str]) -> Dict[str, str]:
    env = dict(base)
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def main() -> int:
    a = parse_args()
    notifier = TelegramNotifier.from_args(a, run_label=f"resume_sweep:{a.run_id}")
    print(f"[resilient] telegram notify: {notifier.status_line()}", flush=True)
    notifier.send(
        "STARTED",
        body=(
            f"run_id={a.run_id}\n"
            f"start={a.start} end={a.end}\n"
            f"jobs={a.jobs} then_enrich={bool(a.then_enrich)}"
        ),
    )

    try:
        expected = expected_variant_count(float(a.pstar))
        print(f"[resilient] expected variants: {expected}", flush=True)

        sweep_root = REPO_ROOT / "results" / "policy_sweeps" / str(a.run_id)
        summary_csv = sweep_root / "summary.csv"

        sweep_cmd = [
            a.python,
            str(REPO_ROOT / "tools" / "sweep_policy_settings_v2.py"),
            "--start",
            str(a.start),
            "--end",
            str(a.end),
            "--jobs",
            str(max(1, int(a.jobs))),
            "--pstar",
            str(a.pstar),
            "--lambda",
            str(a.lam),
            "--mu",
            str(a.mu),
            "--on-missing-symbol",
            str(a.on_missing_symbol),
            "--smoke-n",
            str(int(a.smoke_n)),
            "--run-id",
            str(a.run_id),
        ]

        env = build_low_mem_env(os.environ)

        attempt = 0
        while True:
            attempt += 1
            if a.max_attempts > 0 and attempt > int(a.max_attempts):
                print("[resilient] max-attempts reached", flush=True)
                notifier.send(
                    "FINISHED_WITH_ERRORS",
                    body=(
                        f"run_id={a.run_id}\nstatus=error rc=3\nreason=max_attempts_reached\n"
                        f"attempts={attempt - 1}"
                    ),
                )
                return 3

            print(f"[resilient] attempt {attempt}", flush=True)
            rc = run(sweep_cmd, env)
            rows = read_summary(summary_csv)
            stat = summarize(rows, expected)
            print(
                "[resilient] summary "
                f"rows={stat['rows']} ok={stat['ok']} error={stat['error']} "
                f"other={stat['other']} missing={stat['missing']} rc={rc}",
                flush=True,
            )

            done = (
                rc == 0
                and stat["rows"] >= expected
                and stat["ok"] == expected
                and stat["error"] == 0
                and stat["other"] == 0
            )
            if done:
                print("[resilient] sweep complete", flush=True)
                break

            time.sleep(max(float(a.sleep_sec), 1.0))

        if not a.then_enrich:
            notifier.send(
                "FINISHED",
                body=f"run_id={a.run_id}\nstatus=ok\nattempts={attempt}\nsteps=sweep_only",
            )
            return 0

        enrich_cmd = [
            a.python,
            str(REPO_ROOT / "tools" / "run_sweep_and_enrich.py"),
            "--skip-sweep",
            "--run-id",
            str(a.run_id),
            "--resume",
            "--enrich-jobs",
            str(max(1, int(a.enrich_jobs))),
            "--enrich-retries",
            str(max(0, int(a.enrich_retries))),
            "--funding-stage",
            str(a.funding_stage),
            "--funding-throttle",
            str(float(a.funding_throttle)),
        ]

        if a.tg_bot_token:
            enrich_cmd.extend(["--tg-bot-token", str(a.tg_bot_token)])
        if a.tg_chat_id:
            enrich_cmd.extend(["--tg-chat-id", str(a.tg_chat_id)])
        enrich_cmd.append("--tg-auto-chat" if a.tg_auto_chat else "--no-tg-auto-chat")

        rc = run(enrich_cmd, env)
        notifier.send(
            "FINISHED" if rc == 0 else "FINISHED_WITH_ERRORS",
            body=f"run_id={a.run_id}\nstatus={'ok' if rc == 0 else 'error'} rc={rc}\nattempts={attempt}\nsteps=sweep+enrich",
        )
        return 0 if rc == 0 else rc
    except KeyboardInterrupt:
        notifier.send("INTERRUPTED", body=f"run_id={a.run_id}\nstatus=interrupted (KeyboardInterrupt)")
        raise
    except Exception as exc:
        notifier.send("FAILED", body=f"run_id={a.run_id}\nstatus=failed\nerror={type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
