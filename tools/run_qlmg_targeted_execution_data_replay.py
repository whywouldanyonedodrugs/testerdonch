#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, stable_hash, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402
from tools.run_qlmg_targeted_1m_data_pilot import (  # noqa: E402
    DATASETS,
    OPTIONAL_DATASETS,
    fetch_funding,
    fetch_kline_dataset,
    fetch_open_interest,
    qc_dataset_file,
    write_partition,
)

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_targeted_execution_data_replay_20260627_v1"
LIQSAFE_ROOT = RESULTS_ROOT / "phase_qlmg_simple_alpha_liqsafe_development_20260627_v1_20260627_083845"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
D4_AUDIT_ROOT = RESULTS_ROOT / "phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927"
D4_CANDIDATE_ID = "D4__b4c9487fe82c"
DEFAULT_SEED = 20260627

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "candidate-manifest-normalization",
    "target-window-deduplication-and-tiering",
    "data-source-capability-audit",
    "targeted-1m-download-if-approved",
    "targeted-1m-qc",
    "event-level-reconstruction",
    "one-minute-mark-index-replay",
    "stop-target-liquidation-ordering",
    "funding-oi-verification",
    "family-specific-vs-generic-shock-deconfounding",
    "execution-depth-source-audit",
    "top-of-book-depth-trade-replay-if-available",
    "cost-funding-slippage-stress-refresh",
    "candidate-decision-table",
    "d4-depth-integration",
    "secondary-family-data-plan",
    "next-contracts-and-backlog",
    "decision-report",
    "compact-review-bundle",
    "all",
)

EXPECTED_DEPTH_SCHEMA = {
    "contract_id",
    "family",
    "targeted_1m",
    "top_of_book",
    "shallow_depth",
    "public_trades",
    "liquidation_feed",
    "PIT_sector_map",
    "catalyst_database",
    "listing_lifecycle_metadata",
}

DECISION_STATUSES = {
    "targeted_execution_data_prelead_confirmed",
    "targeted_execution_data_prelead_unresolved",
    "promote_to_family_specific_validation_after_data",
    "not_fairly_tested_missing_data",
    "not_fairly_tested_execution_model_missing",
    "reject_current_translation_only",
    "carry_forward_d4_execution_depth",
    "blocked_by_protocol_issue",
}


@dataclass
class RunContext:
    args: argparse.Namespace
    run_root: Path
    notifier: "RunNotifier"
    start: pd.Timestamp
    end: pd.Timestamp


class RunNotifier:
    def __init__(self, run_root: Path, disabled: bool = False, require_remote: bool = False, allow_no_remote: bool = False) -> None:
        self.run_root = run_root
        self.disabled = disabled
        self.events_path = run_root / "notifications/telegram_events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.remote = None
        self.status = "disabled" if disabled else "unavailable"
        self.missing = "disabled_by_cli" if disabled else ""
        if not disabled and TelegramNotifier is not None:
            class _Args:
                disable_telegram = False
                telegram_dry_run = False
                tg_bot_token = ""
                tg_chat_id = ""
                tg_auto_chat = False
            try:
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-targeted-replay")
                self.status = getattr(self.remote, "status_line", lambda: "enabled")()
                if "disabled" in str(self.status).lower():
                    self.missing = str(self.status)
            except Exception as exc:  # pragma: no cover
                self.remote = None
                self.status = "unavailable"
                self.missing = f"{type(exc).__name__}: {exc}"
        elif not disabled:
            self.missing = "tools.telegram_notify.TelegramNotifier unavailable"
        if require_remote and not self.remote_available and not allow_no_remote:
            raise RuntimeError(f"remote Telegram required but unavailable: {self.missing or self.status}")

    @property
    def remote_available(self) -> bool:
        return (not self.disabled) and self.remote is not None and "enabled" in str(self.status).lower()

    def send(self, title: str, body: str = "", *, level: str = "info") -> bool:
        sent = False
        error = ""
        if self.remote is not None:
            try:
                sent = bool(self.remote.send(title, body))
            except Exception as exc:  # pragma: no cover
                error = f"{type(exc).__name__}: {exc}"
        rec = {"ts_utc": utc_now(), "title": title, "body": body, "level": level, "sent": sent, "status": self.status, "error": error}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        try:
            (self.run_root / "watch_status.json").write_text(json.dumps({"status": "running", "last_event": title, "ts_utc": rec["ts_utc"], "run_root": str(self.run_root)}, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        return sent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG targeted execution-data replay, train-only")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=30.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--download-targeted-1m", action="store_true")
    p.add_argument("--targeted-download-cap-gb", type=float, default=12.0)
    p.add_argument("--use-existing-1m-if-overlap", action="store_true", default=True)
    p.add_argument("--download-depth-if-source-available", action="store_true")
    p.add_argument("--depth-source", default="")
    p.add_argument("--depth-download-cap-gb", type=float, default=20.0)
    p.add_argument("--public-trades-if-source-available", action="store_true")
    p.add_argument("--include-d4", action="store_true", default=True)
    p.add_argument("--include-funding-window", action="store_true", default=True)
    p.add_argument("--include-listing", action="store_true", default=True)
    p.add_argument("--include-generic-shock", action="store_true", default=True)
    p.add_argument("--include-secondary-plan", action="store_true", default=True)
    p.add_argument("--tmux-session-name", default="qlmg_targeted_replay")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--run-root", default="")
    return p.parse_args(argv)


def resolve_run_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.run_root:
        p = Path(args.run_root)
        return (p if p.is_absolute() else REPO / p).resolve(), "explicit_run_root"
    base = (RESULTS_ROOT / DEFAULT_RUN_ID).resolve()
    if args.smoke:
        return (base / "smoke").resolve(), "smoke_subroot"
    if base.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return base.with_name(f"{base.name}_{suffix}"), f"default_root_existed_suffix_{suffix}"
    return base, "default_root_available"


def clamp_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(args.start, utc=True)
    end = min(pd.to_datetime(args.end, utc=True), SCREENING_END)
    if start >= FINAL_HOLDOUT_START or end >= FINAL_HOLDOUT_START:
        raise RuntimeError("requested window overlaps protected QLMG holdout")
    return pd.Timestamp(start), pd.Timestamp(end)


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    rows_list = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows_list:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows_list:
            w.writerow(dict(row))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk = f.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def shell(args: Sequence[str], timeout: float = 120.0) -> str:
    try:
        p = subprocess.run(args, cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def done_path(root: Path, stage: str) -> Path:
    return root / "stage_status" / f"{stage}.done"


def mark_done(root: Path, stage: str) -> None:
    write_text(done_path(root, stage), utc_now())


def required_outputs(root: Path, stage: str) -> list[Path]:
    m = {
        "preflight-and-artifact-freeze": [root / "preflight/preflight_report.md", root / "preflight/frozen_artifact_hashes.json", root / "preflight/input_artifact_manifest.csv"],
        "telegram-and-tmux-setup": [root / "notifications/telegram_readiness_report.md", root / "tmux/watch_commands.md"],
        "seal-guard": [root / "seal/seal_guard_report.md", root / "seal/protected_slice_check.json"],
        "candidate-manifest-normalization": [root / "candidates/canonical_candidate_manifest.csv", root / "candidates/model_level_candidate_manifest.csv"],
        "target-window-deduplication-and-tiering": [root / "windows/target_window_manifest.csv", root / "windows/control_window_manifest.csv", root / "windows/storage_estimate.csv"],
        "data-source-capability-audit": [root / "data_sources/data_source_capability_matrix.csv", root / "data_sources/data_source_capability_report.md"],
        "targeted-1m-download-if-approved": [root / "downloaded_1m/download_manifest.csv", root / "downloaded_1m/download_report.md"],
        "targeted-1m-qc": [root / "downloaded_1m/qc/pilot_coverage_summary.csv", root / "downloaded_1m/qc/pilot_data_qc_report.md"],
        "event-level-reconstruction": [root / "replay/event_level_reconstruction.parquet", root / "replay/event_level_reconstruction_summary.csv"],
        "one-minute-mark-index-replay": [root / "replay/one_minute_mark_replay_summary.csv", root / "replay/event_samples.parquet"],
        "stop-target-liquidation-ordering": [root / "ordering/stop_target_liquidation_ordering_summary.csv"],
        "funding-oi-verification": [root / "funding_oi/funding_oi_verification_summary.csv"],
        "family-specific-vs-generic-shock-deconfounding": [root / "deconfound/family_vs_generic_shock_summary.csv"],
        "execution-depth-source-audit": [root / "depth/depth_source_capability_matrix.csv", root / "depth/depth_procurement_or_live_capture_plan.md"],
        "top-of-book-depth-trade-replay-if-available": [root / "depth/depth_trade_replay_summary.csv"],
        "cost-funding-slippage-stress-refresh": [root / "stress/cost_funding_slippage_stress_summary.csv"],
        "candidate-decision-table": [root / "decision/candidate_decision_table.csv"],
        "d4-depth-integration": [root / "d4/d4_depth_integration_report.md", root / "d4/d4_next_action_contract.json"],
        "secondary-family-data-plan": [root / "secondary/secondary_family_data_plan.csv"],
        "next-contracts-and-backlog": [root / "next_contracts/next_contract_summary.csv", root / "next_contracts/backlog.csv"],
        "decision-report": [root / "QLMG_TARGETED_EXECUTION_DATA_REPLAY_REPORT.md", root / "decision_summary.json"],
        "compact-review-bundle": [root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return m.get(stage, [])


def stage_complete(root: Path, stage: str) -> bool:
    return done_path(root, stage).exists() and all(p.exists() for p in required_outputs(root, stage))


def append_command(root: Path, stage: str) -> None:
    p = root / "command_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": utc_now(), "stage": stage, "argv": sys.argv, "cwd": str(REPO)}, sort_keys=True, default=str) + "\n")


def download_estimate_from_manifest(root: Path) -> float:
    est = read_csv(root / "windows/storage_estimate.csv")
    if est.empty:
        return 0.0
    if "download_selected" in est.columns:
        return float(pd.to_numeric(est.loc[est["download_selected"].astype(str).str.lower().eq("true"), "estimated_compressed_gb"], errors="coerce").fillna(0).sum())
    return float(pd.to_numeric(est.get("estimated_compressed_gb", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())


def estimate_stage_gb(ctx: RunContext, stage: str) -> float:
    if ctx.args.smoke:
        return 0.25 if stage == "targeted-1m-download-if-approved" and ctx.args.download_targeted_1m else 0.1
    if stage == "targeted-1m-download-if-approved" and ctx.args.download_targeted_1m:
        return min(float(ctx.args.targeted_download_cap_gb), max(download_estimate_from_manifest(ctx.run_root), 0.1))
    if stage in {"target-window-deduplication-and-tiering", "event-level-reconstruction", "one-minute-mark-index-replay"}:
        return 1.0
    return 0.2


def ensure_guard(ctx: RunContext, stage: str) -> None:
    snap = resource_snapshot(REPO)
    status = check_resource_guard(
        snap,
        estimated_output_gb=estimate_stage_gb(ctx, stage),
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=20.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", {"stage": stage, **status, "snapshot": snap.__dict__})
    if status["warnings"]:
        ctx.notifier.send("QLMG targeted replay resource warning", f"stage={stage} warnings={status['warnings']}", level="warning")
    if status["status"] != "pass":
        ctx.notifier.send("QLMG targeted replay resource hard stop", f"stage={stage} reasons={status['reasons']}", level="error")
        raise RuntimeError(f"resource guard failed for {stage}: {status}")


def parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def maybe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(value, utc=True))


def validate_window_df(df: pd.DataFrame, cols: Sequence[str]) -> None:
    if not df.empty:
        validate_no_protected(df, cols)


def select_depth_contract_file(root: Path) -> tuple[Path | None, str, pd.DataFrame]:
    candidates = [
        (root / "depth_contracts/depth_contract_summary.csv", "primary_depth_contracts"),
        (root / "contracts/depth_data_contract_summary.csv", "fallback_contracts_depth_data"),
    ]
    usable: list[tuple[Path, str, pd.DataFrame]] = []
    for path, label in candidates:
        df = read_csv(path)
        if not df.empty and EXPECTED_DEPTH_SCHEMA.issubset(set(df.columns)):
            usable.append((path, label, df))
    if not usable:
        return None, "no_usable_depth_contract_schema", pd.DataFrame()
    if len(usable) == 2:
        return usable[0][0], f"used_{usable[0][1]}_both_present_schema_ok", usable[0][2]
    path, label, df = usable[0]
    return path, f"used_{label}_schema_ok", df


def candidate_config_hash(row: Mapping[str, Any]) -> str:
    keys = ["candidate_id", "family", "subfamily", "source_family_preset", "horizon", "target_r", "stop_mult", "risk_bps_override", "best_sizing_model"]
    return stable_hash({k: row.get(k, "") for k in keys})


def is_72h(row: Mapping[str, Any]) -> bool:
    return str(row.get("horizon", "")).lower() == "72h"


def load_corrected_summary() -> pd.DataFrame:
    df = read_csv(LIQSAFE_ROOT / "analysis_corrected_unique_candidate_summary.csv")
    if not df.empty and "candidate_id" in df.columns:
        df["candidate_id"] = df["candidate_id"].astype(str)
    return df


def load_prior_window_plan() -> pd.DataFrame:
    df = read_csv(LIQSAFE_ROOT / "data_plan/targeted_1m_window_plan.csv")
    if not df.empty:
        for c in ["window_start", "window_end"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        if "candidate_id" in df.columns:
            df["candidate_id"] = df["candidate_id"].astype(str)
        validate_window_df(df, ["window_start", "window_end"])
    return df


def d4_counts() -> dict[str, Any]:
    event_path = D4_AUDIT_ROOT / "d4_reconstruction/d4_event_ledger.parquet"
    replay_path = D4_AUDIT_ROOT / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    out = {"d4_event_ledger_exists": event_path.exists(), "d4_1m_replay_exists": replay_path.exists(), "accepted_events": 0, "resolved_events": 0}
    if event_path.exists():
        try:
            out["accepted_events"] = int(len(pd.read_parquet(event_path, columns=["event_id"])))
        except Exception:
            out["accepted_events"] = 0
    if replay_path.exists():
        try:
            rep = pd.read_parquet(replay_path, columns=["event_id"])
            out["resolved_events"] = int(rep["event_id"].nunique()) if "event_id" in rep.columns else int(len(rep))
        except Exception:
            out["resolved_events"] = 0
    return out


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    depth_path, depth_status, _ = select_depth_contract_file(LIQSAFE_ROOT)
    artifacts = []
    for path in [
        LIQSAFE_ROOT / "DETAILED_LIQSAFE_DEVELOPMENT_FINDINGS_REPORT.md",
        LIQSAFE_ROOT / "analysis_corrected_unique_candidate_summary.csv",
        LIQSAFE_ROOT / "data_plan/targeted_1m_window_plan.csv",
        LIQSAFE_ROOT / "triage/all_ideas_preservation_index.csv",
        depth_path if depth_path else LIQSAFE_ROOT / "contracts/depth_data_contract_summary.csv",
        D4_SURVIVAL_ROOT / "D4_SURVIVABILITY_REDESIGN_REPORT.md",
        D4_AUDIT_ROOT / "D4_1M_MARK_REPLAY_FINAL_RESULTS_REPORT.md",
        D4_AUDIT_ROOT / "one_minute_mark/d4_1m_mark_replay_by_window.parquet",
    ]:
        if path and path.exists():
            artifacts.append({"path": str(path), "exists": True, "size_bytes": path.stat().st_size, "sha256": sha256_file(path, max_bytes=64 * 1024 * 1024)})
        else:
            artifacts.append({"path": str(path), "exists": False, "size_bytes": 0, "sha256": ""})
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", artifacts)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", {"artifacts": artifacts, "depth_contract_choice": str(depth_path) if depth_path else "", "depth_contract_status": depth_status})
    write_json(ctx.run_root / "preflight/disk_memory_snapshot.json", snap.__dict__)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\n- free_disk_gb: `{snap.free_gb:.3f}`\n- hard_stop_free_gb: `5`\n- warn_free_gb: `7`\n- stage_output_block_gb: `20`\n- max_output_gb: `{ctx.args.max_output_gb}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight And Artifact Freeze\n\n- run_root: `{ctx.run_root}`\n- liqsafe_root: `{LIQSAFE_ROOT}`\n- d4_survival_root: `{D4_SURVIVAL_ROOT}`\n- d4_audit_root: `{D4_AUDIT_ROOT}`\n- protected_cutoff: `{FINAL_HOLDOUT_START}`\n- screening_end: `{SCREENING_END}`\n- depth_contract_choice: `{depth_path}`\n- depth_contract_status: `{depth_status}`\n- d4_counts: `{d4_counts()}`\n- git_head: `{shell(['git','rev-parse','HEAD'])}`\n- git_status_short: `{shell(['git','status','--short'])[:4000]}`\n")
    if depth_path is None:
        raise RuntimeError("no usable depth contract schema found; fail closed")


def stage_telegram(ctx: RunContext) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\n- disabled: `{ctx.args.disable_telegram}`\n- remote_available: `{ctx.notifier.remote_available}`\n- status: `{ctx.notifier.status}`\n- missing: `{ctx.notifier.missing}`\n- require_telegram: `{ctx.args.require_telegram}`\n- allow_no_telegram: `{ctx.args.allow_no_telegram}`\n")
    write_text(ctx.run_root / "tmux/watch_commands.md", f"# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n")
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nPreferred full command:\n\n```bash\nbash tools/run_qlmg_targeted_execution_data_replay_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --download-targeted-1m --targeted-download-cap-gb {ctx.args.targeted_download_cap_gb:g} --use-existing-1m-if-overlap --include-d4 --include-funding-window --include-listing --include-generic-shock --include-secondary-plan --require-telegram --seed {ctx.args.seed} --launch-tmux\n```\n")
    ctx.notifier.send("QLMG targeted replay stage", "telegram-and-tmux-setup complete")


def stage_seal(ctx: RunContext) -> None:
    good = pd.DataFrame({"window_end": [pd.Timestamp("2025-12-31T23:59:59Z")]})
    validate_no_protected(good, ["window_end"])
    blocked = False
    try:
        validate_no_protected(pd.DataFrame({"window_start": [pd.Timestamp("2026-01-01T00:00:00Z")]}), ["window_start"])
    except Exception:
        blocked = True
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "pre_holdout_read_passed": True, "protected_read_blocked": blocked})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\n- protected_start: `{FINAL_HOLDOUT_START}`\n- generated/read windows must end <= `{SCREENING_END}`\n- protected smoke blocked: `{blocked}`\n")
    if not blocked:
        raise RuntimeError("seal guard smoke did not block protected timestamp")


def stage_candidate_manifest(ctx: RunContext) -> None:
    corrected = load_corrected_summary()
    if corrected.empty:
        raise RuntimeError("corrected unique candidate summary missing")
    corrected = corrected.drop_duplicates("candidate_id", keep="first").copy()
    corrected["targeted_prelead_corrected"] = corrected.get("targeted_prelead_corrected", False).map(parse_bool)
    rows: list[dict[str, Any]] = []
    model_rows = corrected.to_dict("records")
    if ctx.args.include_funding_window:
        fw = corrected[(corrected["family"].astype(str) == "funding_window_orb_failure") & corrected["targeted_prelead_corrected"]].copy()
        fw = fw.sort_values(["beats_refreshed_null", "net_R", "PF"], ascending=[False, False, False]).head(15)
        rows.extend(fw.to_dict("records"))
    if ctx.args.include_listing:
        li = corrected[(corrected["family"].astype(str).isin(["new_perp_listing_event_study", "generic_shock_reversal"])) & corrected["targeted_prelead_corrected"]].copy()
        li = li.sort_values(["beats_refreshed_null", "net_R", "PF"], ascending=[False, False, False]).head(10)
        rows.extend(li.to_dict("records"))
    canonical = pd.DataFrame(rows).drop_duplicates("candidate_id", keep="first") if rows else pd.DataFrame()
    if ctx.args.include_generic_shock:
        contract = LIQSAFE_ROOT / "contracts/generic_shock_reversal_development_contract.json"
        if contract.exists() and "generic_shock_reversal_hypothesis" not in set(canonical.get("candidate_id", pd.Series(dtype=str)).astype(str)):
            generic = {"candidate_id": "generic_shock_reversal_hypothesis", "family": "generic_shock_reversal", "subfamily": "deconfounded_from_funding_listing", "selection_bucket": "hypothesis_preservation", "horizon": "72h", "target_r": 5.0, "stop_mult": 1.0, "events": np.nan, "targeted_prelead_corrected": False}
            canonical = pd.concat([canonical, pd.DataFrame([generic])], ignore_index=True)
    if ctx.args.include_d4:
        d4 = {"candidate_id": D4_CANDIDATE_ID, "family": "D4", "subfamily": "liquidation_safe_carry_forward", "selection_bucket": "mandatory_d4_carry_forward", "horizon": "2h", "target_r": 1.0, "stop_mult": 2.0, "events": d4_counts().get("accepted_events", 4475), "targeted_prelead_corrected": True, "best_sizing_model": "reuse_prior_survivability"}
        canonical = pd.concat([canonical, pd.DataFrame([d4])], ignore_index=True)
    canonical = canonical.drop_duplicates("candidate_id", keep="first").reset_index(drop=True)
    prior_windows = load_prior_window_plan()
    recon_rows = []
    for _, row in canonical.iterrows():
        cid = str(row.get("candidate_id"))
        if cid == D4_CANDIDATE_ID:
            counts = d4_counts()
            status = "exact_reconstruction" if counts.get("accepted_events", 0) and counts.get("resolved_events", 0) else "not_fairly_tested_missing_data"
            reconstructed = int(counts.get("accepted_events", 0) or 0)
            prior_events = int(counts.get("accepted_events", 0) or 0)
        elif cid == "generic_shock_reversal_hypothesis":
            status = "not_fairly_tested_missing_data"
            reconstructed = 0
            prior_events = 0
        else:
            sub = prior_windows[prior_windows.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not prior_windows.empty else pd.DataFrame()
            reconstructed = int(sub.get("event_id", pd.Series(dtype=str)).astype(str).nunique()) if not sub.empty else 0
            prior_events = int(round(maybe_float(row.get("events"), 0)))
            if reconstructed <= 0:
                status = "not_fairly_tested_missing_data"
            elif prior_events and reconstructed == prior_events:
                status = "exact_reconstruction"
            else:
                status = "exact_config_partial_event_reconstruction"
        recon_rows.append({"candidate_id": cid, "prior_event_count": prior_events, "reconstructed_event_count": reconstructed, "reconstruction_status": status, "config_hash": candidate_config_hash(row)})
    recon = pd.DataFrame(recon_rows)
    canonical = canonical.merge(recon, on="candidate_id", how="left")
    canonical["model_row_inflation_removed"] = True
    canonical["source_root"] = str(LIQSAFE_ROOT)
    write_csv(ctx.run_root / "candidates/canonical_candidate_manifest.csv", canonical)
    write_csv(ctx.run_root / "candidates/model_level_candidate_manifest.csv", pd.DataFrame(model_rows))
    write_csv(ctx.run_root / "candidates/candidate_reconstruction_status.csv", recon)
    write_text(ctx.run_root / "candidates/candidate_manifest_report.md", f"# Candidate Manifest Normalization\n\n- corrected unique rows: `{len(corrected)}`\n- canonical candidates: `{len(canonical)}`\n- D4 included: `{ctx.args.include_d4}`\n- model-row inflation removed: `true`\n- reconstruction statuses: `{recon['reconstruction_status'].value_counts().to_dict()}`\n")


def _window_id(row: Mapping[str, Any], prefix: str = "w") -> str:
    key = "|".join(str(row.get(k, "")) for k in ["symbol", "window_start", "window_end", "window_scope"])
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"


def _make_window(candidate: Mapping[str, Any], base: Mapping[str, Any], role: str, scope: str, start: pd.Timestamp, end: pd.Timestamp, priority: int, control_type: str = "") -> dict[str, Any] | None:
    if pd.isna(start) or pd.isna(end) or end >= FINAL_HOLDOUT_START or end <= start:
        return None
    out = {
        "candidate_id": candidate.get("candidate_id"),
        "family": candidate.get("family"),
        "subfamily": candidate.get("subfamily", ""),
        "event_id": base.get("event_id", ""),
        "symbol": str(base.get("symbol", "")),
        "window_start": start,
        "window_end": min(end, SCREENING_END),
        "hours": (min(end, SCREENING_END) - start).total_seconds() / 3600.0,
        "window_role": role,
        "control_type": control_type,
        "window_scope": scope,
        "priority": priority,
        "datasets_requested": "ohlcv_1m;mark_1m;index_1m;premium_1m;open_interest_5m;funding_history",
        "source_event_id": base.get("event_id", ""),
    }
    out["target_window_id"] = _window_id(out)
    return out


def estimate_window_gb(row: Mapping[str, Any]) -> float:
    minutes = max(1.0, maybe_float(row.get("hours"), 0) * 60.0)
    # 1m OHLCV/mark/index/premium + sparse OI/funding. This is deliberately conservative.
    estimated_rows = minutes * 4.2 + 10.0
    return estimated_rows * 160.0 / (1024**3)


def build_window_rows(ctx: RunContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = read_csv(ctx.run_root / "candidates/canonical_candidate_manifest.csv")
    prior = load_prior_window_plan()
    if ctx.args.max_symbols and not prior.empty:
        syms = sorted(prior["symbol"].dropna().astype(str).unique())[: ctx.args.max_symbols]
        prior = prior[prior["symbol"].astype(str).isin(syms)]
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        cid = str(cand.get("candidate_id"))
        if cid in {D4_CANDIDATE_ID, "generic_shock_reversal_hypothesis"}:
            continue
        sub = prior[prior["candidate_id"].astype(str).eq(cid)].sort_values(["symbol", "window_start", "event_id"]) if not prior.empty else pd.DataFrame()
        if ctx.args.smoke:
            sub = sub.head(3)
        for _, w in sub.iterrows():
            start = ts(w["window_start"])
            end = ts(w["window_end"])
            base = w.to_dict()
            core = _make_window(cand, base, "candidate_event", "core_24h", start, min(end, start + pd.Timedelta(hours=28)), 1)
            if core:
                rows.append(core)
            if is_72h(cand):
                anchor = start + pd.Timedelta(hours=4)
                full = _make_window(cand, base, "candidate_event", "full_72h", start, anchor + pd.Timedelta(hours=72), 2)
                if full:
                    rows.append(full)
            same = _make_window(cand, base, "control", "core_24h", start, min(end, start + pd.Timedelta(hours=28)), 3, "same_time_non_signal")
            if same:
                controls.append(same)
            shifted_start = start + pd.Timedelta(days=7)
            shifted_end = end + pd.Timedelta(days=7)
            if shifted_end >= FINAL_HOLDOUT_START:
                shifted_start = start - pd.Timedelta(days=7)
                shifted_end = end - pd.Timedelta(days=7)
            if shifted_start >= ctx.start and shifted_end <= SCREENING_END:
                shifted = _make_window(cand, {**base, "event_id": f"{base.get('event_id')}_shifted"}, "control", "core_24h", shifted_start, shifted_end, 4, "shifted_time")
                if shifted:
                    controls.append(shifted)
            # A deterministic matched proxy from another event in the same symbol or family, if available.
            pool = prior[(prior["candidate_id"].astype(str) != cid) & (prior["symbol"].astype(str) == str(w.get("symbol")))]
            if pool.empty:
                pool = prior[(prior["candidate_id"].astype(str) != cid) & (prior["family"].astype(str) == str(cand.get("family")))].head(1)
            if not pool.empty:
                m = pool.iloc[0].to_dict()
                matched = _make_window(cand, {**m, "event_id": f"{m.get('event_id')}_matched"}, "control", "core_24h", ts(m["window_start"]), ts(m["window_end"]), 5, "matched_symbol_month_vol_oi_funding_proxy")
                if matched:
                    controls.append(matched)
    windows = pd.DataFrame(rows + controls)
    control_df = pd.DataFrame(controls)
    if not windows.empty:
        validate_window_df(windows, ["window_start", "window_end"])
    if not control_df.empty:
        validate_window_df(control_df, ["window_start", "window_end"])
    return windows, control_df


def stage_windows(ctx: RunContext) -> None:
    windows, controls = build_window_rows(ctx)
    if windows.empty:
        write_csv(ctx.run_root / "windows/target_window_manifest.csv", [])
        write_csv(ctx.run_root / "windows/control_window_manifest.csv", [])
        write_csv(ctx.run_root / "windows/window_candidate_map.csv", [])
        write_csv(ctx.run_root / "windows/storage_estimate.csv", [])
        write_text(ctx.run_root / "windows/window_planning_report.md", "# Window Planning\n\nNo reconstructable windows found.\n")
        return
    windows["window_start"] = pd.to_datetime(windows["window_start"], utc=True)
    windows["window_end"] = pd.to_datetime(windows["window_end"], utc=True)
    windows["estimated_compressed_gb"] = windows.apply(estimate_window_gb, axis=1)
    # Deduplicate actual download targets; same-time controls may map to a candidate event window.
    dedup_cols = ["symbol", "window_start", "window_end", "window_scope", "datasets_requested"]
    dedup = windows.sort_values(["priority", "candidate_id", "event_id"]).drop_duplicates(dedup_cols, keep="first").copy()
    dedup["target_window_id"] = dedup.apply(_window_id, axis=1)
    map_df = windows.merge(dedup[dedup_cols + ["target_window_id"]], on=dedup_cols, how="left", suffixes=("_raw", ""))
    cap = float(ctx.args.targeted_download_cap_gb)
    dedup = dedup.sort_values(["priority", "estimated_compressed_gb", "symbol", "window_start"]).reset_index(drop=True)
    dedup["cum_estimated_gb"] = dedup["estimated_compressed_gb"].cumsum()
    dedup["download_selected"] = dedup["cum_estimated_gb"] <= cap
    if ctx.args.smoke:
        dedup.loc[dedup.index >= 10, "download_selected"] = False
    omitted = dedup[~dedup["download_selected"]].copy()
    storage_rows = dedup[["target_window_id", "candidate_id", "family", "symbol", "window_start", "window_end", "window_scope", "window_role", "control_type", "priority", "estimated_compressed_gb", "cum_estimated_gb", "download_selected"]]
    write_csv(ctx.run_root / "windows/target_window_manifest.csv", dedup)
    write_csv(ctx.run_root / "windows/control_window_manifest.csv", controls)
    write_csv(ctx.run_root / "windows/window_candidate_map.csv", map_df)
    write_csv(ctx.run_root / "windows/storage_estimate.csv", storage_rows)
    write_csv(ctx.run_root / "windows/omitted_window_report.csv", omitted)
    write_text(ctx.run_root / "windows/window_planning_report.md", f"# Target Window Planning\n\n- raw candidate/control mappings: `{len(windows)}`\n- deduped download windows: `{len(dedup)}`\n- selected under cap: `{int(dedup['download_selected'].sum())}`\n- omitted by cap: `{len(omitted)}`\n- selected estimated GB: `{float(dedup.loc[dedup['download_selected'], 'estimated_compressed_gb'].sum()):.4f}`\n- cap GB: `{cap}`\n- controls included in manifest: `{len(controls)}`\n- D4 1m redownload policy: reuse prior D4 1m mark replay; D4 depth/trade windows are planned separately.\n")


def path_inventory(root: Path) -> dict[str, Any]:
    exists = root.exists()
    files = list(root.rglob("*.parquet"))[:100] if exists else []
    return {"path": str(root), "exists": exists, "parquet_file_count_sampled": len(files), "size_bytes_sampled": sum(p.stat().st_size for p in files if p.exists())}


def stage_data_sources(ctx: RunContext) -> None:
    rows = [
        path_inventory(Path("/opt/parquet/1m_hot")) | {"dataset": "local_1m_hot", "capability": "partial_existing_overlap"},
        path_inventory(Path("/opt/parquet/1m")) | {"dataset": "local_1m", "capability": "local_store"},
        path_inventory(D4_AUDIT_ROOT / "downloaded_1m") | {"dataset": "prior_d4_downloaded_1m", "capability": "reused_d4_mark_evidence"},
        {"dataset": "bybit_public_ohlcv_mark_index_premium", "path": "https://api.bybit.com/v5/market/*", "exists": True, "capability": "public_api_small_targeted_download", "credentials_required": False},
        {"dataset": "top_of_book_depth_public_trades_history", "path": str(REPO), "exists": False, "capability": "not_found_locally", "credentials_required": "unknown_or_vendor"},
        {"dataset": "liquidation_feed_history", "path": str(REPO), "exists": False, "capability": "not_found_locally", "credentials_required": "unknown_or_vendor"},
    ]
    write_csv(ctx.run_root / "data_sources/data_source_capability_matrix.csv", rows)
    write_text(ctx.run_root / "data_sources/data_source_capability_report.md", "# Data Source Capability Audit\n\nBybit public endpoints are considered usable for bounded 1m OHLCV/mark/index/premium/OI/funding downloads. Historical top-of-book, shallow depth, public trades, and liquidation-feed data were not discovered locally and are treated as procurement/live-capture blockers, not ignored. No secrets are required or logged for public Bybit market data.\n")


def _download_selected_windows(ctx: RunContext, selected: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers["User-Agent"] = "qlmg-targeted-execution-replay/1.0"
    download_root = ctx.run_root / "downloaded_1m"
    datasets = list(DATASETS.keys()) + list(OPTIONAL_DATASETS.keys())
    for n, (_, w) in enumerate(selected.iterrows(), start=1):
        wid = str(w["target_window_id"])
        symbol = str(w["symbol"])
        start = ts(w["window_start"])
        end = ts(w["window_end"])
        for dataset in datasets:
            try:
                if dataset in DATASETS:
                    df, req = fetch_kline_dataset(session, dataset, symbol, start, end)
                    endpoint = DATASETS[dataset]["endpoint"]
                elif dataset == "open_interest_5m":
                    df, req = fetch_open_interest(session, symbol, start, end)
                    endpoint = OPTIONAL_DATASETS[dataset]["endpoint"]
                else:
                    df, req = fetch_funding(session, symbol, start, end)
                    endpoint = OPTIONAL_DATASETS[dataset]["endpoint"]
                if not df.empty:
                    validate_window_df(df, ["timestamp"])
                    path = write_partition(df, download_root, dataset, symbol, wid)
                    status = "ok"
                else:
                    path = Path("")
                    status = "empty"
                manifest.append({"target_window_id": wid, "candidate_id": w.get("candidate_id"), "family": w.get("family"), "symbol": symbol, "dataset": dataset, "endpoint": endpoint, "status": status, "rows": len(df), "requests": req, "path": str(path), "error": ""})
            except Exception as exc:
                failures.append({"target_window_id": wid, "candidate_id": w.get("candidate_id"), "symbol": symbol, "dataset": dataset, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                manifest.append({"target_window_id": wid, "candidate_id": w.get("candidate_id"), "family": w.get("family"), "symbol": symbol, "dataset": dataset, "endpoint": DATASETS.get(dataset, OPTIONAL_DATASETS.get(dataset, {})).get("endpoint", ""), "status": "error", "rows": 0, "requests": 0, "path": "", "error": f"{type(exc).__name__}: {exc}"})
        if n % max(1, int(ctx.args.chunk_size)) == 0:
            ctx.notifier.send("QLMG targeted replay download progress", f"windows_done={n}/{len(selected)}")
    return manifest, failures


def stage_download(ctx: RunContext) -> None:
    manifest_path = ctx.run_root / "downloaded_1m/download_manifest.csv"
    failure_path = ctx.run_root / "downloaded_1m/gaps_and_failures.csv"
    target = read_csv(ctx.run_root / "windows/target_window_manifest.csv")
    if not ctx.args.download_targeted_1m:
        write_csv(manifest_path, [])
        write_csv(failure_path, [{"status": "not_run", "reason": "download_targeted_1m_not_passed"}])
        write_text(ctx.run_root / "downloaded_1m/download_report.md", "# Targeted 1m Download\n\nMode: `not_run`. Audit-only mode was used because `--download-targeted-1m` was not passed.\n")
        return
    if target.empty:
        write_csv(manifest_path, [])
        write_csv(failure_path, [{"status": "blocked", "reason": "no_target_windows"}])
        write_text(ctx.run_root / "downloaded_1m/download_report.md", "# Targeted 1m Download\n\nBlocked: no target windows.\n")
        return
    selected = target[target["download_selected"].map(parse_bool)].copy()
    est = float(pd.to_numeric(selected.get("estimated_compressed_gb", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    if est > float(ctx.args.targeted_download_cap_gb) and not ctx.args.allow_large_output:
        write_csv(manifest_path, [])
        write_csv(failure_path, [{"status": "blocked", "reason": "estimated_download_above_cap", "estimated_gb": est, "cap_gb": ctx.args.targeted_download_cap_gb}])
        write_text(ctx.run_root / "downloaded_1m/download_report.md", f"# Targeted 1m Download\n\nBlocked by cap. Estimated selected GB `{est:.4f}` exceeds cap `{ctx.args.targeted_download_cap_gb}`.\n")
        return
    if ctx.args.dry_run:
        write_csv(manifest_path, [])
        write_csv(failure_path, [{"status": "dry_run", "reason": "download_skipped"}])
        write_text(ctx.run_root / "downloaded_1m/download_report.md", "# Targeted 1m Download\n\nDry run: no external data downloaded.\n")
        return
    manifests, failures = _download_selected_windows(ctx, selected)
    write_csv(manifest_path, manifests)
    write_csv(failure_path, failures)
    ok = sum(1 for r in manifests if r.get("status") == "ok")
    write_text(ctx.run_root / "downloaded_1m/download_report.md", f"# Targeted 1m Download\n\n- mode: `download`\n- windows selected: `{len(selected)}`\n- dataset-window manifest rows: `{len(manifests)}`\n- successful non-empty dataset windows: `{ok}`\n- failures: `{len(failures)}`\n- estimated selected GB: `{est:.4f}`\n- data written under run root only.\n")


def stage_qc(ctx: RunContext) -> None:
    rows = []
    for p in (ctx.run_root / "downloaded_1m").rglob("*.parquet"):
        dataset = p.parts[-3] if len(p.parts) >= 3 else p.parent.parent.name
        symbol = p.parent.name.replace("symbol=", "")
        window_id = p.stem.replace("window=", "")
        rows.append(qc_dataset_file(p, dataset, symbol, window_id))
    write_csv(ctx.run_root / "downloaded_1m/qc/pilot_coverage_summary.csv", rows)
    gaps = [r for r in rows if r.get("status") != "ok" or int(r.get("duplicates", 0) or 0) or int(r.get("gap_count", 0) or 0) or int(r.get("nonpositive_price_count", 0) or 0)]
    write_csv(ctx.run_root / "downloaded_1m/qc/pilot_gap_summary.csv", gaps)
    write_text(ctx.run_root / "downloaded_1m/qc/pilot_data_qc_report.md", f"# Targeted 1m QC\n\n- parquet datasets scanned: `{len(rows)}`\n- issue rows: `{len(gaps)}`\n- no protected rows are allowed; QC uses timestamp checks from downloaded parquet files.\n")


def stage_event_reconstruction(ctx: RunContext) -> None:
    cand = read_csv(ctx.run_root / "candidates/canonical_candidate_manifest.csv")
    map_df = read_csv(ctx.run_root / "windows/window_candidate_map.csv")
    rows: list[dict[str, Any]] = []
    for _, c in cand.iterrows():
        cid = str(c.get("candidate_id"))
        if cid == D4_CANDIDATE_ID:
            counts = d4_counts()
            rows.append({"candidate_id": cid, "family": "D4", "event_id": "D4_PRIOR_REPLAY", "symbol": "MULTI", "decision_ts": "", "reconstruction_status": c.get("reconstruction_status"), "event_count": counts.get("accepted_events", 0), "source": str(D4_AUDIT_ROOT / "one_minute_mark/d4_1m_mark_replay_by_window.parquet")})
            continue
        sub = map_df[map_df.get("candidate_id_raw", map_df.get("candidate_id", pd.Series(dtype=str))).astype(str).eq(cid)] if not map_df.empty else pd.DataFrame()
        if sub.empty:
            rows.append({"candidate_id": cid, "family": c.get("family"), "event_id": "", "symbol": "", "decision_ts": "", "reconstruction_status": c.get("reconstruction_status", "not_fairly_tested_missing_data"), "event_count": 0, "source": "no_window_mapping"})
        else:
            for _, r in sub.head(5000 if not ctx.args.smoke else 50).iterrows():
                start = ts(r.get("window_start"))
                decision = start + pd.Timedelta(hours=4)
                rows.append({"candidate_id": cid, "family": c.get("family"), "subfamily": c.get("subfamily"), "event_id": r.get("event_id_raw", r.get("event_id", "")), "symbol": r.get("symbol_raw", r.get("symbol", "")), "decision_ts": decision, "target_window_id": r.get("target_window_id"), "window_role": r.get("window_role_raw", r.get("window_role", "")), "control_type": r.get("control_type_raw", r.get("control_type", "")), "window_scope": r.get("window_scope_raw", r.get("window_scope", "")), "reconstruction_status": c.get("reconstruction_status"), "source": "liq_safe_window_plan"})
    df = pd.DataFrame(rows)
    if not df.empty and "decision_ts" in df.columns:
        df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce")
    validate_window_df(df, ["decision_ts"])
    out = ctx.run_root / "replay/event_level_reconstruction.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    summary = df.groupby(["candidate_id", "family", "reconstruction_status"], dropna=False).size().reset_index(name="rows") if not df.empty else pd.DataFrame()
    write_csv(ctx.run_root / "replay/event_level_reconstruction_summary.csv", summary)
    write_text(ctx.run_root / "replay/event_level_reconstruction_report.md", f"# Event-Level Reconstruction\n\n- rows: `{len(df)}`\n- candidates: `{df['candidate_id'].nunique() if not df.empty and 'candidate_id' in df.columns else 0}`\n- exact reconstruction is reported separately in `candidates/candidate_reconstruction_status.csv`.\n")


def _download_index(root: Path) -> dict[tuple[str, str], dict[str, Path]]:
    out: dict[tuple[str, str], dict[str, Path]] = {}
    for p in (root / "downloaded_1m").rglob("*.parquet"):
        dataset = p.parts[-3] if len(p.parts) >= 3 else p.parent.parent.name
        symbol = p.parent.name.replace("symbol=", "")
        wid = p.stem.replace("window=", "")
        out.setdefault((wid, symbol), {})[dataset] = p
    return out


def _first_open(path: Path) -> float | None:
    try:
        df = pd.read_parquet(path, columns=["timestamp", "open"])
        if df.empty:
            return None
        return maybe_float(df.sort_values("timestamp").iloc[0].get("open"), np.nan)
    except Exception:
        return None


def _path_metrics(price_path: Path, side: str, entry: float, risk_bps: float, target_r: float = 3.0) -> dict[str, Any]:
    try:
        df = pd.read_parquet(price_path)
        if df.empty:
            return {"replay_status": "empty_path"}
        validate_window_df(df, ["timestamp"])
        high = pd.to_numeric(df.get("high"), errors="coerce")
        low = pd.to_numeric(df.get("low"), errors="coerce")
        risk = max(entry * risk_bps / 10000.0, entry * 0.001)
        if side == "short":
            stop = entry + risk
            target = entry - risk * target_r
            mfe_bps = float(((entry - low.min()) / entry) * 10000.0)
            mae_bps = float(((high.max() - entry) / entry) * 10000.0)
            stop_hit = bool((high >= stop).any())
            target_hit = bool((low <= target).any())
        else:
            stop = entry - risk
            target = entry + risk * target_r
            mfe_bps = float(((high.max() - entry) / entry) * 10000.0)
            mae_bps = float(((entry - low.min()) / entry) * 10000.0)
            stop_hit = bool((low <= stop).any())
            target_hit = bool((high >= target).any())
        net_r = 0.0
        exit_reason = "time"
        if stop_hit and target_hit:
            net_r = -1.0
            exit_reason = "same_bar_or_window_pessimistic_stop"
        elif stop_hit:
            net_r = -1.0
            exit_reason = "stop"
        elif target_hit:
            net_r = float(target_r)
            exit_reason = "target"
        else:
            last = maybe_float(df.sort_values("timestamp").iloc[-1].get("close"), entry)
            gross = (last - entry) if side != "short" else (entry - last)
            net_r = float(gross / risk)
        return {"replay_status": "ok", "entry_price_1m": entry, "risk_bps_used": risk_bps, "target_r_used": target_r, "mfe_bps_1m": mfe_bps, "mae_bps_1m": mae_bps, "net_R_1m_mark_proxy": net_r, "exit_reason_1m": exit_reason, "stop_hit_1m": stop_hit, "target_hit_1m": target_hit}
    except Exception as exc:
        return {"replay_status": f"error:{type(exc).__name__}", "error": str(exc)}


def stage_mark_replay(ctx: RunContext) -> None:
    rec = pd.read_parquet(ctx.run_root / "replay/event_level_reconstruction.parquet") if (ctx.run_root / "replay/event_level_reconstruction.parquet").exists() else pd.DataFrame()
    cand = read_csv(ctx.run_root / "candidates/canonical_candidate_manifest.csv")
    cand_map = {str(r.get("candidate_id")): r for r in cand.to_dict("records")}
    idx = _download_index(ctx.run_root)
    rows: list[dict[str, Any]] = []
    for _, r in rec.iterrows():
        cid = str(r.get("candidate_id"))
        if cid == D4_CANDIDATE_ID:
            rows.append({"candidate_id": cid, "family": "D4", "one_minute_mark_replayed": True, "matched_control": False, "full_hold_replayed": True, "evidence_source": "prior_d4_1m_mark_replay_reused", "replay_status": "reused_prior_d4"})
            continue
        wid = str(r.get("target_window_id", ""))
        sym = str(r.get("symbol", ""))
        paths = idx.get((wid, sym), {})
        mark_path = paths.get("mark_1m") or paths.get("bybit_linear_mark_1m")
        ohlcv_path = paths.get("ohlcv_1m") or paths.get("bybit_linear_ohlcv_1m")
        c = cand_map.get(cid, {})
        role = str(r.get("window_role", ""))
        scope = str(r.get("window_scope", ""))
        side = "short" if "short" in str(c.get("subfamily", "")).lower() or "short" in str(c.get("family", "")).lower() else "long"
        risk_bps = max(50.0, maybe_float(c.get("risk_bps_override"), 150.0) * max(maybe_float(c.get("stop_mult"), 1.0), 0.25))
        target_r = max(0.5, maybe_float(c.get("target_r"), 3.0))
        if mark_path and ohlcv_path:
            entry = _first_open(ohlcv_path)
            if entry and math.isfinite(entry) and entry > 0:
                out = _path_metrics(mark_path, side, entry, risk_bps, target_r)
            else:
                out = {"replay_status": "missing_entry_open"}
        else:
            out = {"replay_status": "missing_1m_mark_or_ohlcv", "missing_mark": not bool(mark_path), "missing_ohlcv": not bool(ohlcv_path)}
        out.update({"candidate_id": cid, "family": r.get("family"), "subfamily": r.get("subfamily"), "event_id": r.get("event_id"), "symbol": sym, "window_role": role, "control_type": r.get("control_type"), "window_scope": scope, "one_minute_mark_replayed": out.get("replay_status") == "ok", "matched_control": role == "control", "full_hold_replayed": scope == "full_72h" and out.get("replay_status") == "ok", "core_24h_replay_available": scope == "core_24h" and out.get("replay_status") == "ok", "full_72h_replay_available": scope == "full_72h" and out.get("replay_status") == "ok"})
        rows.append(out)
    df = pd.DataFrame(rows)
    write_csv(ctx.run_root / "replay/one_minute_mark_replay_summary.csv", df)
    sample = df.head(500) if not df.empty else df
    sample.to_parquet(ctx.run_root / "replay/event_samples.parquet", index=False, compression="zstd")
    grouped = df.groupby(["candidate_id", "family", "window_role", "replay_status"], dropna=False).size().reset_index(name="rows") if not df.empty else pd.DataFrame()
    write_csv(ctx.run_root / "replay/one_minute_mark_replay_by_candidate.csv", grouped)
    write_text(ctx.run_root / "replay/one_minute_mark_index_replay_report.md", f"# 1m Mark/Index Replay\n\n- replay rows: `{len(df)}`\n- rows with 1m mark replay: `{int(df.get('one_minute_mark_replayed', pd.Series(dtype=bool)).sum()) if not df.empty else 0}`\n- D4 policy: reused prior 1m mark replay; no large D4 redownload in this phase.\n")


def stage_ordering(ctx: RunContext) -> None:
    df = read_csv(ctx.run_root / "replay/one_minute_mark_replay_summary.csv")
    if df.empty:
        out = pd.DataFrame()
    else:
        def classify(r: pd.Series) -> str:
            if str(r.get("replay_status")) != "ok":
                return "unresolved_missing_mark_or_ohlcv"
            if parse_bool(r.get("stop_hit_1m")) and parse_bool(r.get("target_hit_1m")):
                return "pessimistic_stop_before_target"
            if parse_bool(r.get("stop_hit_1m")):
                return "stop_before_target"
            if parse_bool(r.get("target_hit_1m")):
                return "target_before_stop"
            return "time_exit_no_hit"
        df["ordering_class"] = df.apply(classify, axis=1)
        out = df.groupby(["candidate_id", "family", "window_role", "ordering_class"], dropna=False).size().reset_index(name="rows")
    write_csv(ctx.run_root / "ordering/stop_target_liquidation_ordering_summary.csv", out)
    write_text(ctx.run_root / "ordering/stop_target_liquidation_ordering_report.md", "# Stop/Target/Liquidation Ordering\n\nMark-primary ordering is computed only when both 1m mark and OHLCV are available. Missing paths are unresolved and cap evidence.\n")


def stage_funding_oi(ctx: RunContext) -> None:
    manifest = read_csv(ctx.run_root / "downloaded_1m/download_manifest.csv")
    rows = []
    for cid, sub in manifest.groupby("candidate_id") if not manifest.empty and "candidate_id" in manifest.columns else []:
        funding = sub[sub["dataset"].astype(str).eq("funding_history")]
        oi = sub[sub["dataset"].astype(str).eq("open_interest_5m")]
        rows.append({"candidate_id": cid, "funding_rows_downloaded": int(pd.to_numeric(funding.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()), "oi_rows_downloaded": int(pd.to_numeric(oi.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()), "funding_anchor_source": "bybit_history_if_rows_else_fallback_flagged", "fallback_schedule_flag": int(pd.to_numeric(funding.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) == 0})
    if not rows:
        rows = [{"candidate_id": "ALL", "funding_rows_downloaded": 0, "oi_rows_downloaded": 0, "funding_anchor_source": "not_downloaded_or_unavailable", "fallback_schedule_flag": True}]
    write_csv(ctx.run_root / "funding_oi/funding_oi_verification_summary.csv", rows)
    write_text(ctx.run_root / "funding_oi/funding_oi_verification_report.md", "# Funding/OI Verification\n\nFunding-window anchors use downloaded Bybit funding history where present. Any missing per-symbol funding rows are flagged as fallback/unknown and cap conclusions.\n")


def stage_deconfound(ctx: RunContext) -> None:
    replay = read_csv(ctx.run_root / "replay/one_minute_mark_replay_summary.csv")
    rows = []
    if not replay.empty:
        for cid, sub in replay.groupby("candidate_id"):
            event = sub[sub["window_role"].astype(str).eq("candidate_event")]
            ctrl = sub[sub["window_role"].astype(str).eq("control")]
            event_r = float(pd.to_numeric(event.get("net_R_1m_mark_proxy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            ctrl_r = float(pd.to_numeric(ctrl.get("net_R_1m_mark_proxy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            rows.append({"candidate_id": cid, "event_signal_R_1m": event_r, "control_signal_R_1m": ctrl_r, "uplift_R_1m": event_r - ctrl_r, "beats_replayed_controls": event_r > ctrl_r and len(ctrl) > 0, "generic_shock_surface_preserved": True})
    write_csv(ctx.run_root / "deconfound/family_vs_generic_shock_summary.csv", rows)
    write_text(ctx.run_root / "deconfound/family_vs_generic_shock_deconfounding_report.md", "# Family-Specific vs Generic Shock Deconfounding\n\nThe generic shock/reversal hypothesis is preserved as a first-class hypothesis. Candidate labels are not rejected merely because a funding/listing family appears to be a generic shock/reversal surface. Replayed controls are required before stronger conclusions.\n")


def stage_depth_audit(ctx: RunContext) -> None:
    candidates = read_csv(ctx.run_root / "candidates/canonical_candidate_manifest.csv")
    windows = read_csv(ctx.run_root / "windows/target_window_manifest.csv")
    symbols = sorted(set(windows.get("symbol", pd.Series(dtype=str)).dropna().astype(str))) if not windows.empty else []
    rows = [
        {"data_type": "top_of_book", "local_source_found": False, "bybit_historical_public_api": False, "likely_vendor_required": True, "forward_live_capture_can_substitute": True},
        {"data_type": "shallow_depth", "local_source_found": False, "bybit_historical_public_api": False, "likely_vendor_required": True, "forward_live_capture_can_substitute": True},
        {"data_type": "public_trades", "local_source_found": False, "bybit_historical_public_api": "limited_current_or_recent_only_unknown", "likely_vendor_required": True, "forward_live_capture_can_substitute": True},
        {"data_type": "liquidation_feed", "local_source_found": False, "bybit_historical_public_api": "not_confirmed_for_history", "likely_vendor_required": True, "forward_live_capture_can_substitute": True},
    ]
    write_csv(ctx.run_root / "depth/depth_source_capability_matrix.csv", rows)
    subset = windows.head(1000).copy() if not windows.empty else pd.DataFrame()
    depth_manifest = subset[["target_window_id", "candidate_id", "family", "symbol", "window_start", "window_end", "window_scope"]].copy() if not subset.empty else pd.DataFrame()
    write_csv(ctx.run_root / "depth/depth_window_manifest.csv", depth_manifest)
    write_text(ctx.run_root / "depth/depth_procurement_or_live_capture_plan.md", f"# Depth / Trade Procurement Or Live Capture Plan\n\n- candidate rows: `{len(candidates)}`\n- exact symbols in first-priority manifest: `{len(symbols)}`\n- required data types: top-of-book, shallow depth, public trades, liquidation-feed history.\n- source options: Tardis/vendor historical archive, any locally configured exchange-transfer archive if later discovered, or forward live capture.\n- official Bybit feasibility: public 1m klines are available; historical depth/trade/liquidation coverage sufficient for this audit is not confirmed by local sources.\n- D4 priority: do not redownload large 1m mark windows; prioritize depth/trade/liquidation-feed windows around prior accepted D4 events and liquidation-risk windows.\n- symbols sample: `{', '.join(symbols[:50])}`\n")


def stage_depth_replay(ctx: RunContext) -> None:
    rows = [{"status": "not_run", "reason": "no_local_top_of_book_depth_public_trade_history_source", "depth_replayed": False, "top_of_book_replayed": False, "public_trades_replayed": False}]
    write_csv(ctx.run_root / "depth/depth_trade_replay_summary.csv", rows)
    write_text(ctx.run_root / "depth/depth_trade_replay_report.md", "# Depth/Trade Replay\n\nNot run. No local historical top-of-book/depth/public-trade source was discovered. This caps candidates at targeted data collection or post-data family-specific validation, depending on other evidence.\n")


def stage_stress(ctx: RunContext) -> None:
    replay = read_csv(ctx.run_root / "replay/one_minute_mark_replay_summary.csv")
    rows = []
    if replay.empty:
        rows.append({"candidate_id": "ALL", "scenario": "base", "net_R": 0.0, "status": "no_replay"})
    else:
        for cid, sub in replay[replay.get("window_role", pd.Series(dtype=str)).astype(str).eq("candidate_event")].groupby("candidate_id"):
            base = float(pd.to_numeric(sub.get("net_R_1m_mark_proxy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            events = max(int(len(sub)), 1)
            for scenario, haircut in [("base", 0.0), ("cost_x1p25", 0.05), ("cost_x1p5", 0.10), ("cost_x2", 0.20), ("plus_10bps", 0.10), ("plus_25bps", 0.25), ("adverse_funding_doubled", 0.05), ("mark_fallback_disabled", 0.0)]:
                rows.append({"candidate_id": cid, "scenario": scenario, "net_R": base - events * haircut, "events": events, "stress_status": "computed_from_1m_proxy" if base else "unresolved_or_zero"})
    write_csv(ctx.run_root / "stress/cost_funding_slippage_stress_summary.csv", rows)
    write_text(ctx.run_root / "stress/cost_funding_slippage_stress_report.md", "# Cost/Funding/Slippage Stress Refresh\n\nStress is refreshed only where 1m replay exists. Base failure rejects current translation only; cost x1.25 blocks prelead status; cost x1.5 marks fragility; cost x2 is a severe warning, not automatic rejection.\n")


def stage_decision_table(ctx: RunContext) -> None:
    candidates = read_csv(ctx.run_root / "candidates/canonical_candidate_manifest.csv")
    replay = read_csv(ctx.run_root / "replay/one_minute_mark_replay_summary.csv")
    deconf = read_csv(ctx.run_root / "deconfound/family_vs_generic_shock_summary.csv")
    depth = read_csv(ctx.run_root / "depth/depth_trade_replay_summary.csv")
    stress = read_csv(ctx.run_root / "stress/cost_funding_slippage_stress_summary.csv")
    beats_controls: dict[str, bool] = {}
    if not deconf.empty and {"candidate_id", "beats_replayed_controls"}.issubset(deconf.columns):
        beats_controls = {str(r["candidate_id"]): parse_bool(r["beats_replayed_controls"]) for _, r in deconf.iterrows()}
    rows = []
    for _, c in candidates.iterrows():
        cid = str(c.get("candidate_id"))
        sub = replay[replay.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not replay.empty else pd.DataFrame()
        events = sub[sub.get("window_role", pd.Series(dtype=str)).astype(str).eq("candidate_event")] if not sub.empty else pd.DataFrame()
        controls = sub[sub.get("window_role", pd.Series(dtype=str)).astype(str).eq("control")] if not sub.empty else pd.DataFrame()
        one_minute = bool(events.get("one_minute_mark_replayed", pd.Series(dtype=bool)).map(parse_bool).any()) if not events.empty else (cid == D4_CANDIDATE_ID)
        controls_replayed = bool(controls.get("one_minute_mark_replayed", pd.Series(dtype=bool)).map(parse_bool).any()) if not controls.empty else False
        full_hold = bool(events.get("full_hold_replayed", pd.Series(dtype=bool)).map(parse_bool).any()) if not events.empty else (cid == D4_CANDIDATE_ID)
        horizon_72 = is_72h(c)
        depth_replayed = False if depth.empty else bool(depth.get("depth_replayed", pd.Series([False])).map(parse_bool).any())
        top_replayed = False if depth.empty else bool(depth.get("top_of_book_replayed", pd.Series([False])).map(parse_bool).any())
        public_replayed = False if depth.empty else bool(depth.get("public_trades_replayed", pd.Series([False])).map(parse_bool).any())
        st = stress[stress.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not stress.empty else pd.DataFrame()
        base = float(pd.to_numeric(st.loc[st.get("scenario", pd.Series(dtype=str)).astype(str).eq("base"), "net_R"], errors="coerce").fillna(0).sum()) if not st.empty else 0.0
        c125 = float(pd.to_numeric(st.loc[st.get("scenario", pd.Series(dtype=str)).astype(str).eq("cost_x1p25"), "net_R"], errors="coerce").fillna(0).sum()) if not st.empty else 0.0
        blocker = []
        if not one_minute:
            blocker.append("missing_1m_mark_replay")
        if not controls_replayed and cid != D4_CANDIDATE_ID:
            blocker.append("matched_controls_not_replayed")
        if controls_replayed and cid != D4_CANDIDATE_ID and not beats_controls.get(cid, False):
            blocker.append("does_not_beat_replayed_controls")
        if horizon_72 and not full_hold:
            blocker.append("full_72h_hold_not_replayed")
        if not top_replayed or not depth_replayed or not public_replayed:
            blocker.append("missing_top_of_book_depth_public_trades")
        if cid == D4_CANDIDATE_ID:
            status = "carry_forward_d4_execution_depth"
            blocker.append("D4_requires_depth_trade_liquidation_feed_evidence")
        elif not one_minute or (horizon_72 and not full_hold) or not controls_replayed:
            status = "targeted_execution_data_prelead_unresolved"
        elif controls_replayed and not beats_controls.get(cid, False):
            status = "reject_current_translation_only"
        elif base <= 0:
            status = "reject_current_translation_only"
        elif c125 <= 0:
            status = "not_fairly_tested_execution_model_missing"
        elif top_replayed and depth_replayed and public_replayed:
            status = "promote_to_family_specific_validation_after_data"
        else:
            status = "targeted_execution_data_prelead_confirmed"
        evidence_level = "1m_mark_plus_controls" if one_minute and controls_replayed else ("prior_d4_1m_replay_reused" if cid == D4_CANDIDATE_ID else "planning_or_partial_replay")
        rows.append({"candidate_id": cid, "family": c.get("family"), "subfamily": c.get("subfamily"), "reconstruction_status": c.get("reconstruction_status"), "evidence_level": evidence_level, "one_minute_mark_replayed": one_minute, "top_of_book_replayed": top_replayed, "depth_replayed": depth_replayed, "public_trades_replayed": public_replayed, "matched_controls_replayed": controls_replayed, "full_hold_replayed": full_hold, "core_24h_replay_available": bool(events.get("core_24h_replay_available", pd.Series(dtype=bool)).map(parse_bool).any()) if not events.empty else False, "full_72h_replay_available": full_hold, "decision_based_on_core_only": bool(horizon_72 and one_minute and not full_hold), "decision_based_on_full_hold": bool(full_hold), "status": status, "main_remaining_blocker": ";".join(dict.fromkeys(blocker))})
    out = pd.DataFrame(rows)
    bad = sorted(set(out["status"]) - DECISION_STATUSES) if not out.empty else []
    if bad:
        raise RuntimeError(f"invalid decision statuses: {bad}")
    write_csv(ctx.run_root / "decision/candidate_decision_table.csv", out)
    write_text(ctx.run_root / "decision/candidate_decision_table_report.md", "# Candidate Decision Table\n\nEvidence-level caps are applied per candidate. Missing 1m mark, matched controls, full-hold replay, depth, top-of-book, public trades, lifecycle, or funding metadata cannot be ignored.\n")


def stage_d4(ctx: RunContext) -> None:
    counts = d4_counts()
    # Depth windows are a compact pointer to prior D4 event windows; no large 1m redownload.
    raw_win = read_csv(D4_AUDIT_ROOT / "targeted_1m/deduped_windows.csv")
    if raw_win.empty:
        raw_win = read_csv(D4_AUDIT_ROOT / "targeted_1m/raw_targeted_windows.csv")
    if not raw_win.empty:
        sample = raw_win.head(1000).copy()
        write_csv(ctx.run_root / "d4/d4_depth_window_manifest.csv", sample)
    else:
        write_csv(ctx.run_root / "d4/d4_depth_window_manifest.csv", [])
    contract = {"contract_id": "d4_targeted_execution_depth_collection_contract", "candidate_id": D4_CANDIDATE_ID, "source_root": str(D4_SURVIVAL_ROOT), "prior_1m_mark_replay_root": str(D4_AUDIT_ROOT), "accepted_events": counts.get("accepted_events"), "resolved_1m_mark_events": counts.get("resolved_events"), "required_next_data": ["top_of_book", "shallow_depth", "public_trades", "liquidation_feed"], "no_live_trading": True, "no_sealed_validation": True, "protected_holdout_start": str(FINAL_HOLDOUT_START)}
    write_json(ctx.run_root / "d4/d4_next_action_contract.json", contract)
    write_text(ctx.run_root / "d4/d4_depth_integration_report.md", f"# D4 Depth Integration\n\n- D4 candidate: `{D4_CANDIDATE_ID}`\n- prior accepted events: `{counts.get('accepted_events')}`\n- prior resolved 1m mark events: `{counts.get('resolved_events')}`\n- D4 1m redownload policy: `reuse_prior_1m_mark_replay`; do not consume this phase's targeted 1m cap unless stale/missing.\n- next action: targeted top-of-book/depth/public-trade/liquidation-feed collection.\n")


def stage_secondary(ctx: RunContext) -> None:
    triage = read_csv(LIQSAFE_ROOT / "triage/all_ideas_preservation_index.csv")
    if triage.empty:
        out = pd.DataFrame([{"family": "unknown", "current_label": "not_fairly_tested_missing_data", "why_not_immediate_prelead": "triage_index_missing", "next_fair_test": "rebuild_preservation_index"}])
    else:
        out = triage[~triage["family"].astype(str).isin(["funding_window_orb_failure", "new_perp_listing_event_study", "D4"])].drop_duplicates(["family", "subfamily"]).head(200).copy()
        out["why_not_in_25_immediate_preleads"] = out.get("main_blocker", "not_selected_for_immediate_targeted_replay")
        out["needed_data_or_entry_redesign"] = out.get("main_blocker", "needs_family_specific_redefinition")
        out["next_fair_test"] = out.get("next_action", "contract_or_data_build_before_replay")
    write_csv(ctx.run_root / "secondary/secondary_family_data_plan.csv", out)
    write_text(ctx.run_root / "secondary/secondary_family_data_plan_report.md", "# Secondary Family Data Plan\n\nSecondary families are preserved and are not rejected by this targeted replay phase. This table records why they were not in the immediate prelead set, path-edge status when available, and the next fair test.\n")


def stage_next_contracts(ctx: RunContext) -> None:
    decisions = read_csv(ctx.run_root / "decision/candidate_decision_table.csv")
    rows = []
    for _, d in decisions.iterrows():
        status = str(d.get("status"))
        if status in {"targeted_execution_data_prelead_confirmed", "targeted_execution_data_prelead_unresolved", "carry_forward_d4_execution_depth"}:
            contract = {"candidate_id": d.get("candidate_id"), "family": d.get("family"), "status": status, "required_next_data": d.get("main_remaining_blocker"), "protected_holdout_start": str(FINAL_HOLDOUT_START), "no_live_trading": True, "no_sealed_validation": True}
            path = ctx.run_root / "next_contracts" / f"{str(d.get('candidate_id')).replace('/', '_')}.json"
            write_json(path, contract)
            rows.append({"candidate_id": d.get("candidate_id"), "family": d.get("family"), "status": status, "contract_path": str(path)})
    write_csv(ctx.run_root / "next_contracts/next_contract_summary.csv", rows)
    secondary = read_csv(ctx.run_root / "secondary/secondary_family_data_plan.csv")
    write_csv(ctx.run_root / "next_contracts/backlog.csv", secondary.head(200) if not secondary.empty else pd.DataFrame())
    write_text(ctx.run_root / "next_contracts/next_contracts_report.md", "# Next Contracts And Backlog\n\nImmediate contracts are limited to targeted execution-data preleads and mandatory D4 carry-forward. Secondary families are preserved in the backlog and are not rejected by this run.\n")


def stage_decision_report(ctx: RunContext) -> None:
    decisions = read_csv(ctx.run_root / "decision/candidate_decision_table.csv")
    counts = decisions["status"].value_counts().to_dict() if not decisions.empty else {}
    final_verdict = "targeted_execution_data_replay_complete" if not decisions.empty else "blocked_by_protocol_issue"
    write_json(ctx.run_root / "decision_summary.json", {"verdict": final_verdict, "protected_holdout_untouched": True, "status_counts": counts, "run_root": str(ctx.run_root), "compact_review_bundle": str(ctx.run_root / "compact_review_bundle")})
    secondary = read_csv(ctx.run_root / "secondary/secondary_family_data_plan.csv")
    sec_table = secondary.head(30).to_markdown(index=False) if not secondary.empty else "No secondary rows."
    dec_table = decisions[["candidate_id", "family", "status", "evidence_level", "main_remaining_blocker"]].head(40).to_markdown(index=False) if not decisions.empty else "No decisions."
    write_text(ctx.run_root / "QLMG_TARGETED_EXECUTION_DATA_REPLAY_REPORT.md", f"# QLMG Targeted Execution Data Replay Report\n\n## Scope\n\nTrain-only targeted execution-data replay. Protected holdout `>= {FINAL_HOLDOUT_START}` was not used. No sealed validation, live readiness, or trading recommendation is made.\n\n## Decision Counts\n\n`{counts}`\n\n## Candidate Decision Table Sample\n\n{dec_table}\n\n## D4\n\nD4 remains mandatory carry-forward for targeted execution-depth/trade/liquidation-feed collection. Prior D4 1m mark replay is reused; D4 does not consume this run's 1m download cap.\n\n## Secondary Families\n\n{sec_table}\n\n## Remaining Blockers\n\nMissing top-of-book/depth/public-trade/liquidation-feed history caps all execution-sensitive candidates. Missing 1m mark/control/full-hold replay caps candidates as unresolved targeted-data preleads, not rejected.\n")


def stage_compact(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_TARGETED_EXECUTION_DATA_REPLAY_REPORT.md",
        "decision_summary.json",
        "preflight/preflight_report.md",
        "candidates/canonical_candidate_manifest.csv",
        "candidates/candidate_reconstruction_status.csv",
        "windows/storage_estimate.csv",
        "windows/omitted_window_report.csv",
        "data_sources/data_source_capability_report.md",
        "downloaded_1m/download_report.md",
        "downloaded_1m/qc/pilot_data_qc_report.md",
        "replay/one_minute_mark_replay_summary.csv",
        "ordering/stop_target_liquidation_ordering_summary.csv",
        "funding_oi/funding_oi_verification_summary.csv",
        "deconfound/family_vs_generic_shock_summary.csv",
        "depth/depth_procurement_or_live_capture_plan.md",
        "stress/cost_funding_slippage_stress_summary.csv",
        "decision/candidate_decision_table.csv",
        "d4/d4_depth_integration_report.md",
        "d4/d4_next_action_contract.json",
        "secondary/secondary_family_data_plan.csv",
        "next_contracts/next_contract_summary.csv",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
        "command_log.jsonl",
    ]
    rows = []
    for rel in include:
        src = ctx.run_root / rel
        if src.exists() and src.is_file():
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            rows.append({"relative_path": rel, "source_path": str(src), "bundle_path": str(dst), "size_bytes": src.stat().st_size})
    # Small samples only; full downloaded datasets are intentionally excluded.
    for rel in ["replay/event_samples.parquet"]:
        src = ctx.run_root / rel
        if src.exists():
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            rows.append({"relative_path": rel, "source_path": str(src), "bundle_path": str(dst), "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nLarge downloaded ledgers and full partitions are intentionally excluded. Use `artifact_path_index.csv` for paths.\n")
    zip_path = ctx.run_root / "compact_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(bundle.parent))


def stage_dispatch(ctx: RunContext, stage: str) -> None:
    if stage == "preflight-and-artifact-freeze":
        stage_preflight(ctx)
    elif stage == "telegram-and-tmux-setup":
        stage_telegram(ctx)
    elif stage == "seal-guard":
        stage_seal(ctx)
    elif stage == "candidate-manifest-normalization":
        stage_candidate_manifest(ctx)
    elif stage == "target-window-deduplication-and-tiering":
        stage_windows(ctx)
    elif stage == "data-source-capability-audit":
        stage_data_sources(ctx)
    elif stage == "targeted-1m-download-if-approved":
        stage_download(ctx)
    elif stage == "targeted-1m-qc":
        stage_qc(ctx)
    elif stage == "event-level-reconstruction":
        stage_event_reconstruction(ctx)
    elif stage == "one-minute-mark-index-replay":
        stage_mark_replay(ctx)
    elif stage == "stop-target-liquidation-ordering":
        stage_ordering(ctx)
    elif stage == "funding-oi-verification":
        stage_funding_oi(ctx)
    elif stage == "family-specific-vs-generic-shock-deconfounding":
        stage_deconfound(ctx)
    elif stage == "execution-depth-source-audit":
        stage_depth_audit(ctx)
    elif stage == "top-of-book-depth-trade-replay-if-available":
        stage_depth_replay(ctx)
    elif stage == "cost-funding-slippage-stress-refresh":
        stage_stress(ctx)
    elif stage == "candidate-decision-table":
        stage_decision_table(ctx)
    elif stage == "d4-depth-integration":
        stage_d4(ctx)
    elif stage == "secondary-family-data-plan":
        stage_secondary(ctx)
    elif stage == "next-contracts-and-backlog":
        stage_next_contracts(ctx)
    elif stage == "decision-report":
        stage_decision_report(ctx)
    elif stage == "compact-review-bundle":
        stage_compact(ctx)
    else:
        raise ValueError(stage)


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        print(f"[resume] skipping {stage}")
        return
    ensure_guard(ctx, stage)
    append_command(ctx.run_root, stage)
    ctx.notifier.send("QLMG targeted replay stage start", stage)
    try:
        stage_dispatch(ctx, stage)
    except Exception as exc:
        ctx.notifier.send("QLMG targeted replay stage failure", f"stage={stage}\n{type(exc).__name__}: {exc}", level="error")
        raise
    mark_done(ctx.run_root, stage)
    ctx.notifier.send("QLMG targeted replay stage complete", stage)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    start, end = clamp_window(args)
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram and not args.smoke, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "start": str(start), "end": str(end), "args": vars(args), "protected_holdout_start": str(FINAL_HOLDOUT_START)})
    notifier.send("QLMG targeted replay run start", f"run_root={run_root}\nstage={args.stage}")
    for stage in stage_list(args.stage):
        run_stage(ctx, stage)
    notifier.send("QLMG targeted replay run complete", f"run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
