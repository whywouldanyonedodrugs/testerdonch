#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from telegram_notify import TelegramNotifier


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_run_id(user_id: str) -> str:
    return user_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run policy sweep, then enrich every variant trades output."
    )
    # Sweep args (forwarded)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="latest")
    p.add_argument("--jobs", type=int, default=1, help="Sweep parallel jobs.")
    p.add_argument("--pstar", type=float, default=0.64)
    p.add_argument("--lambda", dest="lam", type=float, default=2.0)
    p.add_argument("--mu", dest="mu", type=float, default=1.0)
    p.add_argument("--on-missing-symbol", choices=["fail", "skip"], default="skip")
    p.add_argument("--smoke-n", type=int, default=0)
    p.add_argument("--run-id", default="", help="If empty, UTC timestamp is used.")

    # Orchestration switches
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-enrich", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if any enrich stage fails.")

    # Enrichment controls
    p.add_argument("--enrich-jobs", type=int, default=1, help="Parallel enrich workers across variants.")
    p.add_argument("--enrich-retries", type=int, default=1, help="Retries per failed variant.")
    p.add_argument(
        "--funding-stage",
        choices=["on", "off", "optional"],
        default="optional",
        help="Funding enrich stage mode.",
    )
    p.add_argument("--funding-throttle", type=float, default=0.0)
    p.add_argument("--max-variants", type=int, default=0, help="If >0, enrich only first N variants (sorted).")

    # Script/executable paths
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--sweep-script", default="tools/sweep_policy_settings_v2.py")
    p.add_argument("--enrich-script", default="tools/enrich_trades_pipeline.py")
    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument(
        "--tg-auto-chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try Telegram getUpdates to auto-discover chat id when not provided.",
    )
    return p.parse_args()


def import_cfg():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import config as cfg  # type: ignore
    return cfg


def find_trades_file(run_dir: Path) -> Path:
    pats = ["trades.clean.csv", "trades.enriched.filled.csv", "trades.enriched.csv", "trades.csv", "*.trades.csv"]
    for pat in pats:
        hits = sorted(run_dir.rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No trades file found under {run_dir}")


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"[orchestrator] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


@dataclass
class EnrichResult:
    variant: str
    status: str
    attempts: int
    seconds: float
    input_trades: Optional[str]
    output_trades: Optional[str]
    note: Optional[str] = None


def enrich_one_variant(
    variant_dir: Path,
    *,
    python_exe: str,
    enrich_script: Path,
    resume: bool,
    retries: int,
    funding_stage: str,
    funding_throttle: float,
) -> EnrichResult:
    started = time.time()
    attempts = 0
    last_err: Optional[str] = None

    try:
        in_trades = find_trades_file(variant_dir)
    except Exception as exc:
        return EnrichResult(
            variant=variant_dir.name,
            status="skipped_no_trades",
            attempts=0,
            seconds=0.0,
            input_trades=None,
            output_trades=None,
            note=f"{type(exc).__name__}: {exc}",
        )

    out_trades = variant_dir / "trades.clean.csv"
    workdir = variant_dir / ".enrich_pipeline"

    base_cmd = [
        python_exe,
        str(enrich_script),
        "--in",
        str(in_trades),
        "--out",
        str(out_trades),
        "--workdir",
        str(workdir),
        "--compat-dir",
        str(variant_dir),
        "--funding-throttle",
        str(funding_throttle),
    ]

    if resume:
        base_cmd.append("--resume")
    if funding_stage == "off":
        base_cmd.append("--skip-funding")
    elif funding_stage == "optional":
        base_cmd.append("--funding-optional")

    max_attempts = max(1, retries + 1)
    for i in range(max_attempts):
        attempts = i + 1
        try:
            run_cmd(base_cmd, REPO_ROOT)
            dt = time.time() - started
            return EnrichResult(
                variant=variant_dir.name,
                status="ok",
                attempts=attempts,
                seconds=dt,
                input_trades=str(in_trades),
                output_trades=str(out_trades),
                note=None,
            )
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if attempts < max_attempts:
                sleep_s = min(20.0, 2.0 * attempts)
                print(
                    f"[orchestrator] enrich failed for {variant_dir.name} attempt {attempts}/{max_attempts}, retrying in {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            else:
                break

    dt = time.time() - started
    return EnrichResult(
        variant=variant_dir.name,
        status="error",
        attempts=attempts,
        seconds=dt,
        input_trades=str(in_trades),
        output_trades=str(out_trades),
        note=last_err,
    )


def list_variant_dirs(sweep_root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in sweep_root.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        ]
    )


def write_enrich_summary(sweep_root: Path, rows: List[EnrichResult]) -> None:
    out_json = sweep_root / "enrichment_summary.json"
    out_csv = sweep_root / "enrichment_summary.csv"
    out_json.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "status",
                "attempts",
                "seconds",
                "input_trades",
                "output_trades",
                "note",
            ],
        )
        wr.writeheader()
        for r in rows:
            wr.writerow(asdict(r))
    print(f"[orchestrator] wrote: {out_json}", flush=True)
    print(f"[orchestrator] wrote: {out_csv}", flush=True)


def main() -> int:
    a = parse_args()
    cfg = import_cfg()

    rid = utc_run_id(a.run_id)
    results_dir = Path(getattr(cfg, "RESULTS_DIR", REPO_ROOT / "results")).resolve()
    sweep_root = (results_dir / "policy_sweeps" / rid).resolve()
    sweep_script = (REPO_ROOT / a.sweep_script).resolve()
    enrich_script = (REPO_ROOT / a.enrich_script).resolve()
    notifier = TelegramNotifier.from_args(a, run_label=f"sweep:{rid}")

    print(f"[orchestrator] telegram notify: {notifier.status_line()}", flush=True)
    notifier.send(
        "STARTED",
        body=(
            f"run_id={rid}\n"
            f"start={a.start} end={a.end}\n"
            f"jobs={a.jobs} enrich_jobs={a.enrich_jobs}\n"
            f"skip_sweep={bool(a.skip_sweep)} skip_enrich={bool(a.skip_enrich)}"
        ),
    )

    try:
        if not a.skip_sweep:
            sweep_cmd = [
                a.python,
                str(sweep_script),
                "--start",
                str(a.start),
                "--end",
                str(a.end),
                "--jobs",
                str(a.jobs),
                "--pstar",
                str(a.pstar),
                "--lambda",
                str(a.lam),
                "--mu",
                str(a.mu),
                "--on-missing-symbol",
                str(a.on_missing_symbol),
                "--smoke-n",
                str(a.smoke_n),
                "--run-id",
                rid,
            ]
            run_cmd(sweep_cmd, REPO_ROOT)
        else:
            if not sweep_root.exists():
                raise FileNotFoundError(f"--skip-sweep was set but run directory does not exist: {sweep_root}")

        print(f"[orchestrator] sweep root: {sweep_root}", flush=True)

        if a.skip_enrich:
            notifier.send("FINISHED", body=f"run_id={rid}\nstatus=ok\nsweep_root={sweep_root}\nsteps=sweep_only")
            return 0

        variants = list_variant_dirs(sweep_root)
        if a.max_variants > 0:
            variants = variants[: int(a.max_variants)]

        if not variants:
            raise RuntimeError(f"No variant directories found in {sweep_root}")

        stage_path = sweep_root / "_ENRICH_STAGE.txt"
        stage_path.write_text(f"ENRICH_START total={len(variants)}\n", encoding="utf-8")

        jobs = max(1, int(a.enrich_jobs))
        rows: List[EnrichResult] = []

        if jobs == 1:
            for i, vdir in enumerate(variants, 1):
                stage_path.write_text(f"ENRICH_RUNNING {i}/{len(variants)} {vdir.name}\n", encoding="utf-8")
                res = enrich_one_variant(
                    vdir,
                    python_exe=a.python,
                    enrich_script=enrich_script,
                    resume=bool(a.resume),
                    retries=int(a.enrich_retries),
                    funding_stage=a.funding_stage,
                    funding_throttle=float(a.funding_throttle),
                )
                rows.append(res)
                print(f"[orchestrator] enrich {res.variant}: {res.status} attempts={res.attempts}", flush=True)
        else:
            fut_to_name: Dict[object, str] = {}
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                for vdir in variants:
                    fut = ex.submit(
                        enrich_one_variant,
                        vdir,
                        python_exe=a.python,
                        enrich_script=enrich_script,
                        resume=bool(a.resume),
                        retries=int(a.enrich_retries),
                        funding_stage=a.funding_stage,
                        funding_throttle=float(a.funding_throttle),
                    )
                    fut_to_name[fut] = vdir.name
                done_n = 0
                for fut in as_completed(fut_to_name):
                    done_n += 1
                    res = fut.result()
                    rows.append(res)
                    stage_path.write_text(
                        f"ENRICH_RUNNING {done_n}/{len(variants)} last={res.variant} status={res.status}\n",
                        encoding="utf-8",
                    )
                    print(f"[orchestrator] enrich {res.variant}: {res.status} attempts={res.attempts}", flush=True)

        write_enrich_summary(sweep_root, rows)

        n_ok = sum(1 for r in rows if r.status == "ok")
        n_err = sum(1 for r in rows if r.status == "error")
        n_skip = len(rows) - n_ok - n_err
        stage_path.write_text(
            f"ENRICH_DONE ok={n_ok} error={n_err} skipped={n_skip}\n",
            encoding="utf-8",
        )
        print(f"[orchestrator] enrich summary: ok={n_ok} error={n_err} skipped={n_skip}", flush=True)

        rc = 2 if (a.strict and n_err > 0) else 0
        notifier.send(
            "FINISHED" if rc == 0 else "FINISHED_WITH_ERRORS",
            body=(
                f"run_id={rid}\n"
                f"status={'ok' if rc == 0 else 'error'} rc={rc}\n"
                f"variants={len(rows)} ok={n_ok} error={n_err} skipped={n_skip}\n"
                f"sweep_root={sweep_root}"
            ),
        )
        return rc

    except KeyboardInterrupt:
        notifier.send("INTERRUPTED", body=f"run_id={rid}\nstatus=interrupted (KeyboardInterrupt)\nsweep_root={sweep_root}")
        raise
    except Exception as exc:
        notifier.send(
            "FAILED",
            body=f"run_id={rid}\nstatus=failed\nerror={type(exc).__name__}: {exc}\nsweep_root={sweep_root}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
