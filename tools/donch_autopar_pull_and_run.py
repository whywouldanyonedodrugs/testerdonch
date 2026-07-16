#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from telegram_notify import TelegramNotifier


REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_RE = re.compile(r"^(autopar_\d{4}-\d{2}-\d{2})(?:\.zip)?$")


def _autopar_enabled() -> bool:
    return os.environ.get("DONCH_AUTOPAR_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PackageRef:
    package_id: str
    path: Path
    is_zip: bool
    mtime: float
    date_key: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().isoformat()


def _to_abs(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p.resolve()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtester-side daily automation for donch_autopar (sync + run parity)."
    )
    p.add_argument(
        "--remote-src",
        default="root@167.235.64.82:/srv/donch-autopar/",
        help="rsync source for live exports.",
    )
    p.add_argument(
        "--sync-dir",
        default="/data/autopar",
        help="Local mirror folder for pulled live exports.",
    )
    p.add_argument("--results-root", default="results/donch_autopar")
    p.add_argument("--state-file", default="", help="Default: <results-root>/.autopar_state.json")
    p.add_argument("--lock-file", default="", help="Default: <results-root>/.autopar.lock")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--daily-script", default="tools/donch_autopar_daily.py")

    p.add_argument("--parquet-dir", default="/opt/parquet/5m")
    p.add_argument("--parquet-1m-dir", default="")
    p.add_argument(
        "--model-dir",
        default="results/offline_releases/20260209_meta_release_v2_live_safe/meta_export_pstar_042",
    )
    p.add_argument("--window-days", type=int, default=3)
    p.add_argument("--scout-workers", type=int, default=2)
    p.add_argument("--scout-backend", choices=["thread", "process"], default="thread")

    p.add_argument("--min-overlap-rate", type=float, default=0.50)
    p.add_argument("--min-enter-agreement", type=float, default=0.85)

    p.add_argument(
        "--max-new-packages",
        type=int,
        default=3,
        help="When not using --latest-only, process up to this many pending packages per run.",
    )
    p.add_argument(
        "--latest-only",
        action="store_true",
        help="Process only the latest pending package.",
    )
    p.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore processed state and run package(s) again.",
    )

    p.add_argument("--resume", action="store_true", help="Forward --resume to donch_autopar_daily.py")
    p.add_argument("--skip-scout", action="store_true")
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument(
        "--daily-extra-arg",
        action="append",
        default=[],
        help="Extra arg forwarded to donch_autopar_daily.py (repeatable).",
    )

    p.add_argument(
        "--skip-sync",
        action="store_true",
        help="Do not run rsync; use already mirrored files in --sync-dir.",
    )
    p.add_argument("--rsync-timeout-sec", type=int, default=120)
    p.add_argument("--rsync-delete", action="store_true", help="Enable --delete-delay in rsync.")
    p.add_argument(
        "--rsync-rsh",
        default="",
        help='Optional rsync remote shell, e.g. \'ssh -i /root/.ssh/id_rsa\'.',
    )
    p.add_argument(
        "--ssh-accept-new-hostkey",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --rsync-rsh is not set, use ssh StrictHostKeyChecking=accept-new.",
    )
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--run-id-prefix", default="", help='Optional prefix, e.g. "bt_".')
    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument(
        "--tg-auto-chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-discover Telegram chat id from getUpdates when chat id is not provided.",
    )
    return p.parse_args()


@contextmanager
def _single_instance_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"lock already held: {lock_path}")
        f.write(str(os.getpid()))
        f.flush()
        try:
            yield
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _load_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"processed": {}, "history": []}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {"processed": {}, "history": []}
        obj.setdefault("processed", {})
        obj.setdefault("history", [])
        return obj
    except Exception:
        return {"processed": {}, "history": []}


def _save_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run_subprocess(cmd: List[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[autopar-sync] running: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return int(proc.wait())


def _sync_live_exports(
    *,
    remote_src: str,
    sync_dir: Path,
    timeout_sec: int,
    delete_enabled: bool,
    rsync_rsh: str,
    log_path: Path,
) -> int:
    sync_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-az", "--timeout", str(max(1, int(timeout_sec)))]
    if rsync_rsh.strip():
        cmd += ["-e", rsync_rsh.strip()]
    if delete_enabled:
        cmd.append("--delete-delay")
    cmd += [remote_src, str(sync_dir) + "/"]
    return _run_subprocess(cmd, cwd=REPO_ROOT, log_path=log_path)


def _pkg_from_path(p: Path) -> Optional[PackageRef]:
    m = PKG_RE.match(p.name)
    if not m:
        return None
    pid = m.group(1)
    try:
        st = p.stat()
    except OSError:
        return None
    date_key = pid.replace("autopar_", "")
    return PackageRef(
        package_id=pid,
        path=p,
        is_zip=p.is_file(),
        mtime=float(st.st_mtime),
        date_key=date_key,
    )


def _discover_packages(sync_dir: Path) -> List[PackageRef]:
    candidates: List[PackageRef] = []
    for p in sync_dir.rglob("autopar_*"):
        if p.is_dir() or (p.is_file() and p.suffix.lower() == ".zip"):
            ref = _pkg_from_path(p)
            if ref is not None:
                candidates.append(ref)

    # deduplicate by package_id; prefer directory over zip
    best: Dict[str, PackageRef] = {}
    for ref in candidates:
        cur = best.get(ref.package_id)
        if cur is None:
            best[ref.package_id] = ref
            continue
        cur_score = (0 if cur.is_zip else 1, cur.mtime)
        new_score = (0 if ref.is_zip else 1, ref.mtime)
        if new_score > cur_score:
            best[ref.package_id] = ref

    rows = list(best.values())
    rows.sort(key=lambda x: (x.date_key, x.mtime))
    return rows


def _pick_pending(
    packages: List[PackageRef],
    state: Dict[str, object],
    *,
    force_reprocess: bool,
    latest_only: bool,
    max_new_packages: int,
) -> List[PackageRef]:
    processed = state.get("processed", {})
    if not isinstance(processed, dict):
        processed = {}
    pending: List[PackageRef] = []
    for p in packages:
        if force_reprocess:
            pending.append(p)
            continue
        rec = processed.get(p.package_id, {})
        status = ""
        if isinstance(rec, dict):
            status = str(rec.get("status", "")).lower().strip()
        if status in ("ok", "warn", "no_signals"):
            continue
        pending.append(p)

    if not pending:
        return []
    if latest_only:
        return [pending[-1]]
    return pending[: max(1, int(max_new_packages))]


def _read_summary_status(run_dir: Path) -> str:
    summary = run_dir / "compare" / "summary.json"
    if not summary.exists():
        return "runner_failed"
    try:
        obj = json.loads(summary.read_text(encoding="utf-8"))
        return str(obj.get("status", "runner_failed"))
    except Exception:
        return "runner_failed"


def _build_run_id(pkg_id: str, prefix: str) -> str:
    px = prefix.strip()
    return f"{px}{pkg_id}" if px else pkg_id


def _run_one_package(
    *,
    pkg: PackageRef,
    args: argparse.Namespace,
    results_root: Path,
    daily_script: Path,
) -> Tuple[int, str, Path]:
    run_id = _build_run_id(pkg.package_id, args.run_id_prefix)
    run_dir = results_root / run_id
    cmd = [
        args.python,
        str(daily_script),
        "--run-id",
        run_id,
        "--results-root",
        str(results_root),
        "--live-input",
        str(pkg.path),
        "--parquet-dir",
        str(Path(args.parquet_dir).expanduser().resolve()),
        "--model-dir",
        str(_to_abs(args.model_dir)),
        "--window-days",
        str(max(1, int(args.window_days))),
        "--scout-workers",
        str(max(1, int(args.scout_workers))),
        "--scout-backend",
        str(args.scout_backend),
        "--min-overlap-rate",
        str(float(args.min_overlap_rate)),
        "--min-enter-agreement",
        str(float(args.min_enter_agreement)),
    ]
    if args.parquet_1m_dir.strip():
        cmd += ["--parquet-1m-dir", str(Path(args.parquet_1m_dir).expanduser().resolve())]
    if args.resume:
        cmd.append("--resume")
    if args.skip_scout:
        cmd.append("--skip-scout")
    if args.skip_backtest:
        cmd.append("--skip-backtest")

    if args.tg_bot_token.strip():
        cmd += ["--tg-bot-token", args.tg_bot_token.strip()]
    if args.tg_chat_id.strip():
        cmd += ["--tg-chat-id", args.tg_chat_id.strip()]
    cmd.append("--tg-auto-chat" if args.tg_auto_chat else "--no-tg-auto-chat")

    for extra in args.daily_extra_arg:
        txt = str(extra).strip()
        if txt:
            cmd.append(txt)

    log_path = results_root / "_autopar_sync_logs" / f"{pkg.package_id}.daily.log"
    rc = _run_subprocess(cmd, cwd=REPO_ROOT, log_path=log_path)
    status = _read_summary_status(run_dir)
    return rc, status, run_dir


def main() -> int:
    a = _parse_args()
    if not _autopar_enabled():
        print(
            "[autopar-sync] disabled: Autopar is legacy and disabled for the QLMG project. "
            "Set DONCH_AUTOPAR_ENABLED=1 only for an intentional manual legacy run.",
            flush=True,
        )
        return 2

    results_root = _to_abs(a.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    state_path = _to_abs(a.state_file) if a.state_file.strip() else (results_root / ".autopar_state.json")
    lock_path = _to_abs(a.lock_file) if a.lock_file.strip() else (results_root / ".autopar.lock")
    sync_dir = Path(a.sync_dir).expanduser().resolve()
    daily_script = _to_abs(a.daily_script)
    if not daily_script.exists():
        raise SystemExit(f"daily script not found: {daily_script}")

    notifier = TelegramNotifier.from_args(a, run_label="autopar-sync")
    print(f"[autopar-sync] telegram notify: {notifier.status_line()}", flush=True)

    try:
        with _single_instance_lock(lock_path):
            state = _load_state(state_path)
            state["last_started_utc"] = _utc_iso()
            _save_state(state_path, state)

            if not a.skip_sync:
                if not a.dry_run:
                    notifier.send("STARTED", body=f"sync {a.remote_src} -> {sync_dir}")
                rc_sync = _sync_live_exports(
                    remote_src=a.remote_src,
                    sync_dir=sync_dir,
                    timeout_sec=a.rsync_timeout_sec,
                    delete_enabled=bool(a.rsync_delete),
                    rsync_rsh=(
                        a.rsync_rsh.strip()
                        or (
                            "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
                            if bool(a.ssh_accept_new_hostkey)
                            else ""
                        )
                    ),
                    log_path=results_root / "_autopar_sync_logs" / "rsync.log",
                )
                if rc_sync != 0:
                    state["last_finished_utc"] = _utc_iso()
                    state["last_sync_rc"] = int(rc_sync)
                    _save_state(state_path, state)
                    notifier.send(
                        "FAILED",
                        body=f"rsync failed rc={rc_sync}\nlog={results_root / '_autopar_sync_logs' / 'rsync.log'}",
                    )
                    return 1

            pkgs = _discover_packages(sync_dir)
            if not pkgs:
                state["last_finished_utc"] = _utc_iso()
                state["last_sync_rc"] = 0
                _save_state(state_path, state)
                print(f"[autopar-sync] no package found under {sync_dir}", flush=True)
                notifier.send("DONE", body=f"no package found under {sync_dir}")
                return 0

            pending = _pick_pending(
                pkgs,
                state,
                force_reprocess=bool(a.force_reprocess),
                latest_only=bool(a.latest_only),
                max_new_packages=max(1, int(a.max_new_packages)),
            )
            if not pending:
                state["last_finished_utc"] = _utc_iso()
                state["last_sync_rc"] = 0
                _save_state(state_path, state)
                msg = f"no pending package; latest={pkgs[-1].package_id}"
                print(f"[autopar-sync] {msg}", flush=True)
                notifier.send("DONE", body=msg)
                return 0

            if a.dry_run:
                names = ", ".join([x.package_id for x in pending])
                print(f"[autopar-sync] dry-run pending: {names}", flush=True)
                notifier.send("DONE", body=f"dry-run pending package(s): {names}")
                return 0

            processed = state.get("processed", {})
            if not isinstance(processed, dict):
                processed = {}
            history = state.get("history", [])
            if not isinstance(history, list):
                history = []

            ok_n, warn_n, fail_n = 0, 0, 0
            for pkg in pending:
                rc, status, run_dir = _run_one_package(
                    pkg=pkg,
                    args=a,
                    results_root=results_root,
                    daily_script=daily_script,
                )
                rec = {
                    "processed_at_utc": _utc_iso(),
                    "source_path": str(pkg.path),
                    "run_id": _build_run_id(pkg.package_id, a.run_id_prefix),
                    "run_dir": str(run_dir),
                    "rc": int(rc),
                    "status": status,
                }
                processed[pkg.package_id] = rec
                history.append({"package_id": pkg.package_id, **rec})
                state["processed"] = processed
                state["history"] = history[-2000:]
                state["last_sync_rc"] = 0
                state["last_finished_utc"] = _utc_iso()
                _save_state(state_path, state)

                if rc != 0 or status == "runner_failed":
                    fail_n += 1
                elif status in ("warn", "no_signals"):
                    warn_n += 1
                else:
                    ok_n += 1

            summary = (
                f"processed={len(pending)} ok={ok_n} warn={warn_n} fail={fail_n}\n"
                f"latest={pending[-1].package_id}\nstate={state_path}"
            )
            if fail_n > 0:
                notifier.send("FAILED", body=summary)
                return 1
            if warn_n > 0:
                notifier.send("WARN", body=summary)
                return 0
            notifier.send("DONE", body=summary)
            return 0
    except RuntimeError as e:
        msg = str(e)
        print(f"[autopar-sync] {msg}", flush=True)
        # Lock held is not an error for timer-driven operation.
        if "lock already held" in msg:
            return 0
        notifier.send("FAILED", body=msg)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
