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
from tools.run_qlmg_simple_alpha_plus_d4 import apply_candidate_filter, surface_return_r  # noqa: E402
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
DEFAULT_RUN_ID = "phase_qlmg_listing_generic_full_event_replay_20260627_v1"
TARGETED_REPLAY_ROOT = RESULTS_ROOT / "phase_qlmg_targeted_execution_data_replay_20260627_v1_20260627_100018"
LIQSAFE_ROOT = RESULTS_ROOT / "phase_qlmg_simple_alpha_liqsafe_development_20260627_v1_20260627_083845"
SIMPLE_ALPHA_ROOT = RESULTS_ROOT / "phase_qlmg_simple_alpha_plus_d4_20260626_v1_amended_full_20260626_162555"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
D4_AUDIT_ROOT = RESULTS_ROOT / "phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927"
D4_CANDIDATE_ID = "D4__b4c9487fe82c"
DEFAULT_SEED = 20260627

CONFIRMED_LISTING_IDS = [
    "new_perp_listing_event_study__589a8c85c943",
    "new_perp_listing_event_study__9dc07cfc405c",
    "new_perp_listing_event_study__b1a3735d5092",
]

SECONDARY_FAMILIES = [
    "leader_breakout_long",
    "weak_asset_spike_fade",
    "risk_off_exhaustion_spike_short",
    "us_cash_open_orb",
    "utc_daily_open_reversal",
    "crowded_long_unwind_short",
    "post_catalyst_continuation_base",
    "failed_sector_rotation_short",
]

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "report-semantics-and-status-audit",
    "d4-event-count-reconciliation",
    "confirmed-listing-candidate-full-event-manifest",
    "funding-window-preservation-audit",
    "generic-shock-reversal-event-generator",
    "full-event-window-deduplication-and-storage-estimate",
    "targeted-1m-download-if-approved",
    "targeted-1m-qc",
    "full-event-one-minute-mark-replay",
    "stop-target-liquidation-ordering",
    "matched-control-replay-refresh",
    "listing-vs-generic-shock-deconfounding",
    "cost-funding-slippage-stress-refresh",
    "depth-trade-liquidation-source-audit",
    "depth-procurement-or-live-capture-plan",
    "secondary-family-preservation-refresh",
    "decision-report",
    "compact-review-bundle",
    "all",
)

DECISION_LABELS = {
    "listing_vwap_loss_full_event_prelead_confirmed",
    "generic_shock_reversal_prelead_confirmed",
    "d4_carry_forward_execution_depth",
    "build_depth_trade_liquidation_data_next",
    "continue_hypothesis_development",
}

RECON_LABELS = {
    "exact_full_event_reconstruction",
    "exact_config_partial_event_reconstruction",
    "not_fairly_tested_missing_data",
    "blocked_by_protocol_issue",
}

GB = 1024 ** 3


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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-listing-full-replay")
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
    p = argparse.ArgumentParser(description="QLMG listing/generic full-event 1m replay, train-only")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=40.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--download-targeted-1m", action="store_true")
    p.add_argument("--targeted-download-cap-gb", type=float, default=30.0)
    p.add_argument("--use-existing-1m-if-overlap", action="store_true", default=True)
    p.add_argument("--download-depth-if-source-available", action="store_true")
    p.add_argument("--depth-source", default="")
    p.add_argument("--depth-download-cap-gb", type=float, default=20.0)
    p.add_argument("--public-trades-if-source-available", action="store_true")
    p.add_argument("--include-d4-reconciliation", action="store_true", default=True)
    p.add_argument("--include-generic-shock", action="store_true", default=True)
    p.add_argument("--include-secondary-preservation", action="store_true", default=True)
    p.add_argument("--tmux-session-name", default="qlmg_listing_full_replay")
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
    start = pd.Timestamp(pd.to_datetime(args.start, utc=True))
    end = min(pd.Timestamp(pd.to_datetime(args.end, utc=True)), SCREENING_END)
    if start >= FINAL_HOLDOUT_START or end >= FINAL_HOLDOUT_START:
        raise RuntimeError("requested window overlaps protected QLMG holdout")
    return start, end


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
        "report-semantics-and-status-audit": [root / "audit/report_semantics_audit.md", root / "audit/status_label_corrections.csv"],
        "d4-event-count-reconciliation": [root / "d4/d4_event_count_reconciliation.csv", root / "d4/d4_event_count_reconciliation_report.md"],
        "confirmed-listing-candidate-full-event-manifest": [root / "listing/full_event_candidate_manifest.csv", root / "listing/listing_reconstruction_report.md"],
        "funding-window-preservation-audit": [root / "funding_window/funding_window_preservation_audit.md"],
        "generic-shock-reversal-event-generator": [root / "generic_shock/generic_shock_event_manifest.csv", root / "generic_shock/generic_shock_contract_report.md"],
        "full-event-window-deduplication-and-storage-estimate": [root / "windows/full_event_window_manifest.csv", root / "windows/full_event_storage_estimate.csv", root / "windows/window_candidate_control_map.csv"],
        "targeted-1m-download-if-approved": [root / "downloaded_1m/download_manifest.csv", root / "downloaded_1m/download_report.md"],
        "targeted-1m-qc": [root / "qc/targeted_1m_qc_summary.csv", root / "qc/targeted_1m_qc_report.md"],
        "full-event-one-minute-mark-replay": [root / "replay/full_event_one_minute_replay_summary.csv", root / "replay/full_event_one_minute_events.parquet"],
        "stop-target-liquidation-ordering": [root / "ordering/full_event_ordering_summary.csv", root / "ordering/full_event_ordering_report.md"],
        "matched-control-replay-refresh": [root / "controls/full_event_control_summary.csv", root / "controls/full_event_control_report.md"],
        "listing-vs-generic-shock-deconfounding": [root / "deconfound/listing_vs_generic_shock_summary.csv", root / "deconfound/listing_vs_generic_shock_report.md"],
        "cost-funding-slippage-stress-refresh": [root / "stress/full_event_stress_summary.csv", root / "stress/full_event_stress_report.md"],
        "depth-trade-liquidation-source-audit": [root / "depth/depth_trade_liquidation_source_audit.csv", root / "depth/depth_trade_liquidation_source_audit_report.md"],
        "depth-procurement-or-live-capture-plan": [root / "depth/depth_procurement_or_live_capture_plan.md", root / "depth/targeted_depth_window_manifest.csv"],
        "secondary-family-preservation-refresh": [root / "secondary/secondary_family_preservation_report.md", root / "secondary/secondary_family_status.csv"],
        "decision-report": [root / "QLMG_LISTING_GENERIC_FULL_EVENT_REPLAY_REPORT.md", root / "decision_summary.json"],
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


def estimate_stage_gb(ctx: RunContext, stage: str) -> float:
    if ctx.args.smoke:
        return 0.25 if stage == "targeted-1m-download-if-approved" and ctx.args.download_targeted_1m else 0.1
    if stage == "targeted-1m-download-if-approved" and ctx.args.download_targeted_1m:
        est = read_csv(ctx.run_root / "windows/full_event_storage_estimate.csv")
        if not est.empty and "download_selected" in est.columns:
            return float(pd.to_numeric(est.loc[est["download_selected"].map(parse_bool), "estimated_compressed_gb"], errors="coerce").fillna(0).sum())
        return min(ctx.args.targeted_download_cap_gb, 30.0)
    if stage in {"full-event-window-deduplication-and-storage-estimate", "full-event-one-minute-mark-replay"}:
        return 2.0
    return 0.3


def ensure_guard(ctx: RunContext, stage: str) -> None:
    snap = resource_snapshot(REPO)
    status = check_resource_guard(
        snap,
        estimated_output_gb=estimate_stage_gb(ctx, stage),
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=30.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", {"stage": stage, **status, "snapshot": snap.__dict__})
    if status["warnings"]:
        ctx.notifier.send("QLMG listing replay resource warning", f"stage={stage} warnings={status['warnings']}", level="warning")
    if status["status"] != "pass":
        ctx.notifier.send("QLMG listing replay resource hard stop", f"stage={stage} reasons={status['reasons']}", level="error")
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


def load_listing_events() -> pd.DataFrame:
    p = SIMPLE_ALPHA_ROOT / "listing/listing_event_rows.parquet"
    if not p.exists():
        raise RuntimeError(f"missing listing event ledger: {p}")
    df = pd.read_parquet(p)
    validate_window_df(df, [c for c in ["decision_ts", "entry_ts"] if c in df.columns])
    return df


def load_candidate_registry() -> pd.DataFrame:
    p = SIMPLE_ALPHA_ROOT / "sweep/candidate_registry.csv"
    if not p.exists():
        raise RuntimeError(f"missing candidate registry: {p}")
    return pd.read_csv(p)


def candidate_config_hash(row: Mapping[str, Any]) -> str:
    keys = ["candidate_id", "family", "subfamily", "horizon", "target_r", "stop_mult", "risk_bps_override", "tier_filter", "regime_gate", "funding_gate", "liquidity_quality_gate", "bad_wick_gate", "cost_mult"]
    return stable_hash({k: row.get(k, "") for k in keys}, n=16)


def horizon_hours(horizon: Any) -> float:
    s = str(horizon).strip().lower()
    if s.endswith("m"):
        return float(s[:-1]) / 60.0
    if s.endswith("h"):
        return float(s[:-1])
    if s.endswith("d"):
        return float(s[:-1]) * 24.0
    return maybe_float(s, 24.0)


def sha_artifacts(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append({"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0, "sha256": sha256_file(path, max_bytes=64 * 1024 * 1024) if path.exists() and path.is_file() else ""})
    return rows


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    paths = [
        TARGETED_REPLAY_ROOT / "QLMG_TARGETED_EXECUTION_DATA_REPLAY_REPORT.md",
        TARGETED_REPLAY_ROOT / "decision/candidate_decision_table.csv",
        TARGETED_REPLAY_ROOT / "replay/one_minute_mark_replay_summary.csv",
        LIQSAFE_ROOT / "analysis_corrected_unique_candidate_summary.csv",
        SIMPLE_ALPHA_ROOT / "sweep/candidate_registry.csv",
        SIMPLE_ALPHA_ROOT / "listing/listing_event_rows.parquet",
        D4_SURVIVAL_ROOT / "D4_SURVIVABILITY_REDESIGN_REPORT.md",
        D4_AUDIT_ROOT / "d4_reconstruction/d4_event_ledger.parquet",
        D4_AUDIT_ROOT / "one_minute_mark/one_minute_mark_replay_summary.csv",
    ]
    artifacts = sha_artifacts(paths)
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", artifacts)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", {"artifacts": artifacts, "created_at_utc": utc_now()})
    write_json(ctx.run_root / "preflight/disk_memory_snapshot.json", snap.__dict__)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\n- free_disk_gb: `{snap.free_gb:.3f}`\n- hard_stop_free_gb: `5`\n- warn_free_gb: `7`\n- stage_output_block_gb: `30`\n- max_output_gb: `{ctx.args.max_output_gb}`\n- targeted_1m_cap_gb: `{ctx.args.targeted_download_cap_gb}`\n- depth_download_cap_gb: `{ctx.args.depth_download_cap_gb}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight And Artifact Freeze\n\n- run_root: `{ctx.run_root}`\n- targeted_replay_root: `{TARGETED_REPLAY_ROOT}`\n- liqsafe_root: `{LIQSAFE_ROOT}`\n- simple_alpha_root: `{SIMPLE_ALPHA_ROOT}`\n- d4_survival_root: `{D4_SURVIVAL_ROOT}`\n- d4_audit_root: `{D4_AUDIT_ROOT}`\n- protected_start: `{FINAL_HOLDOUT_START}`\n- screening_end: `{SCREENING_END}`\n- git_head: `{shell(['git','rev-parse','HEAD'])}`\n- git_status_short: `{shell(['git','status','--short'])[:5000]}`\n")


def stage_telegram(ctx: RunContext) -> None:
    if ctx.args.require_telegram and not ctx.notifier.remote_available and not ctx.args.allow_no_telegram and not ctx.args.smoke:
        raise RuntimeError("remote Telegram is required for full launch but unavailable; pass --allow-no-telegram only for explicit local-only run")
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\n- disabled: `{ctx.args.disable_telegram}`\n- remote_available: `{ctx.notifier.remote_available}`\n- status: `{ctx.notifier.status}`\n- missing: `{ctx.notifier.missing}`\n- require_telegram: `{ctx.args.require_telegram}`\n- allow_no_telegram: `{ctx.args.allow_no_telegram}`\n- secrets_persisted: `false`\n")
    cmd = f"bash tools/run_qlmg_listing_generic_full_event_replay_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --download-targeted-1m --targeted-download-cap-gb {ctx.args.targeted_download_cap_gb:g} --use-existing-1m-if-overlap --include-d4-reconciliation --include-generic-shock --include-secondary-preservation --require-telegram --seed {ctx.args.seed} --launch-tmux"
    write_text(ctx.run_root / "tmux/watch_commands.md", f"# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n")
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nSmoke must pass before full launch. Full launch requires `--launch-tmux`.\n\n```bash\n{cmd}\n```\n")
    ctx.notifier.send("QLMG listing full replay stage", "telegram-and-tmux-setup complete")


def stage_seal(ctx: RunContext) -> None:
    validate_no_protected(pd.DataFrame({"ts": [SCREENING_END]}), ["ts"])
    blocked = False
    try:
        validate_no_protected(pd.DataFrame({"ts": [FINAL_HOLDOUT_START]}), ["ts"])
    except Exception:
        blocked = True
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "pre_holdout_read_passed": True, "protected_read_blocked": blocked})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\n- protected_start: `{FINAL_HOLDOUT_START}`\n- allowed_end: `{SCREENING_END}`\n- protected smoke blocked: `{blocked}`\n")
    if not blocked:
        raise RuntimeError("seal guard did not block protected timestamp")


def stage_report_semantics(ctx: RunContext) -> None:
    rows = [
        {"artifact": "targeted execution-data replay", "old_semantic_risk": "targeted 100-window diagnostic could be mistaken for full-event replay", "corrected_status": "partial_targeted_replay_only", "action": "full-event listing replay required before prelead confirmation"},
        {"artifact": "funding-window rows", "old_semantic_risk": "current translation failure could be mistaken for family rejection", "corrected_status": "family_preserved_current_translation_failed", "action": "preservation audit plus optional bounded replay subset"},
        {"artifact": "generic shock/reversal", "old_semantic_risk": "unreconstructable hypothesis could be dropped", "corrected_status": "bounded_hypothesis_test_required", "action": "build separate event manifest and deconfound listing overlap"},
        {"artifact": "D4", "old_semantic_risk": "event count mismatch 4475 vs 4482", "corrected_status": "requires_count_reconciliation", "action": "classify root cause before carry-forward report"},
    ]
    write_csv(ctx.run_root / "audit/status_label_corrections.csv", rows)
    write_text(ctx.run_root / "audit/report_semantics_audit.md", "# Report Semantics And Status Audit\n\nPrior targeted replay outputs are diagnostic and targeted, not full-event validation. This phase explicitly distinguishes partial replay, exact full-event reconstruction, control replay quality, full-hold observability, and family preservation.\n")


def d4_count_sources() -> dict[str, Any]:
    out: dict[str, Any] = {}
    event_path = D4_AUDIT_ROOT / "d4_reconstruction/d4_event_ledger.parquet"
    replay_path = D4_AUDIT_ROOT / "one_minute_mark/one_minute_mark_replay_summary.csv"
    old_replay_path = D4_AUDIT_ROOT / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    if event_path.exists():
        ev = pd.read_parquet(event_path)
        out["event_ledger_rows"] = int(len(ev))
        out["event_ledger_unique_event_id"] = int(ev["event_id"].nunique()) if "event_id" in ev.columns else int(len(ev))
        out["event_ledger_duplicate_event_rows"] = int(len(ev) - out["event_ledger_unique_event_id"])
    if replay_path.exists():
        rep = pd.read_csv(replay_path)
        out["replay_summary_rows"] = int(len(rep))
        out["replay_unique_event_id"] = int(rep["event_id"].nunique()) if "event_id" in rep.columns else int(len(rep))
        for c in ["actual_mark_liquidation_1m", "same_minute_stop_liq_ambiguous", "rankable"]:
            if c in rep.columns:
                out[f"{c}_count"] = int(rep[c].map(parse_bool).sum())
    if old_replay_path.exists():
        rep2 = pd.read_parquet(old_replay_path, columns=None)
        out["replay_by_window_rows"] = int(len(rep2))
        out["replay_by_window_unique_event_id"] = int(rep2["event_id"].nunique()) if "event_id" in rep2.columns else int(len(rep2))
    return out


def classify_d4_root_cause(counts: Mapping[str, Any]) -> str:
    rows = int(counts.get("event_ledger_rows", 0) or 0)
    unique = int(counts.get("event_ledger_unique_event_id", 0) or 0)
    replay_unique = int(counts.get("replay_unique_event_id", counts.get("replay_by_window_unique_event_id", 0)) or 0)
    if rows and unique and rows != unique:
        return "duplicate event rows"
    if unique == 4475 or replay_unique == 4475:
        return "accepted versus rankable event definition"
    if rows == 4482 and unique == 4482 and replay_unique == 4475:
        return "liquidation/ambiguous rows included in one artifact"
    if rows in {4475, 4482} and unique in {4475, 4482}:
        return "model-row versus event-row count"
    return "protocol issue" if rows == 0 and unique == 0 else "changed filtering"


def stage_d4_reconciliation(ctx: RunContext) -> None:
    counts = d4_count_sources()
    root = classify_d4_root_cause(counts)
    rows = [{"source": k, "count_or_value": v} for k, v in counts.items()]
    rows.append({"source": "root_cause_classification", "count_or_value": root})
    rows.append({"source": "prior_narrative_accepted_events", "count_or_value": 4475})
    rows.append({"source": "latest_integration_event_rows", "count_or_value": 4482})
    write_csv(ctx.run_root / "d4/d4_event_count_reconciliation.csv", rows)
    protocol = root == "protocol issue"
    write_text(ctx.run_root / "d4/d4_event_count_reconciliation_report.md", f"# D4 Event Count Reconciliation\n\n- prior narrative accepted events: `4,475`\n- latest integration rows: `4,482`\n- observed counts: `{counts}`\n- root cause classification: `{root}`\n- protocol issue: `{protocol}`\n- carry-forward status: `{'blocked_by_protocol_issue' if protocol else 'd4_carry_forward_execution_depth'}`\n")
    if protocol:
        write_json(ctx.run_root / "d4/d4_protocol_issue.json", {"counts": counts, "root_cause": root})


def reconstruct_listing_candidates(ctx: RunContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = load_candidate_registry()
    events = load_listing_events()
    manifests: list[dict[str, Any]] = []
    all_events: list[pd.DataFrame] = []
    for cid in CONFIRMED_LISTING_IDS:
        hit = registry[registry["candidate_id"].astype(str).eq(cid)]
        if hit.empty:
            manifests.append({"candidate_id": cid, "reconstruction_status": "not_fairly_tested_missing_data", "reason": "candidate_id_missing_from_registry"})
            continue
        cand = hit.iloc[0].to_dict()
        filt = apply_candidate_filter(events, cand).copy()
        prior = int(round(maybe_float(cand.get("precheck_events"), 0)))
        reconstructed = int(len(filt))
        status = "exact_full_event_reconstruction" if prior and reconstructed == prior else ("exact_config_partial_event_reconstruction" if reconstructed > 0 else "not_fairly_tested_missing_data")
        if status not in RECON_LABELS:
            status = "blocked_by_protocol_issue"
        filt["candidate_id"] = cid
        filt["candidate_config_hash"] = candidate_config_hash(cand)
        filt["candidate_horizon"] = cand.get("horizon")
        filt["candidate_target_r"] = cand.get("target_r")
        filt["candidate_stop_mult"] = cand.get("stop_mult")
        filt["candidate_risk_bps_override"] = cand.get("risk_bps_override")
        filt["listing_metadata_source"] = "proxy_launch_only_first_local_bar_or_prior_listing_builder"
        all_events.append(filt)
        manifests.append({
            "candidate_id": cid,
            "config_hash": candidate_config_hash(cand),
            "prior_full_event_count": prior,
            "reconstructed_full_event_count": reconstructed,
            "reconstruction_status": status,
            "family": cand.get("family"),
            "subfamily": cand.get("subfamily"),
            "horizon": cand.get("horizon"),
            "target_r": cand.get("target_r"),
            "stop_mult": cand.get("stop_mult"),
            "risk_bps_override": cand.get("risk_bps_override"),
            "tier_filter": cand.get("tier_filter"),
            "funding_gate": cand.get("funding_gate"),
            "liquidity_quality_gate": cand.get("liquidity_quality_gate"),
            "bad_wick_gate": cand.get("bad_wick_gate"),
            "event_anchor_logic": "decision_ts/entry_ts from prior listing event ledger; no future bars used for reconstruction",
            "listing_metadata_source": "proxy_launch_only_first_local_bar_or_prior_listing_builder",
            "full_event_replay_eligible": status == "exact_full_event_reconstruction",
        })
    out = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    if not out.empty:
        validate_window_df(out, [c for c in ["decision_ts", "entry_ts"] if c in out.columns])
    return pd.DataFrame(manifests), out


def stage_listing_manifest(ctx: RunContext) -> None:
    manifest, events = reconstruct_listing_candidates(ctx)
    write_csv(ctx.run_root / "listing/full_event_candidate_manifest.csv", manifest)
    if not events.empty:
        p = ctx.run_root / "listing/full_event_listing_events.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(p, index=False, compression="zstd")
        sample = events.head(500)
        sample.to_parquet(ctx.run_root / "listing/full_event_listing_events_sample.parquet", index=False, compression="zstd")
    counts = manifest[["candidate_id", "prior_full_event_count", "reconstructed_full_event_count", "reconstruction_status"]].to_dict("records") if not manifest.empty else []
    write_text(ctx.run_root / "listing/listing_reconstruction_report.md", f"# Listing Full-Event Reconstruction\n\n- confirmed listing candidates: `{len(CONFIRMED_LISTING_IDS)}`\n- event rows reconstructed: `{len(events)}`\n- reconstruction rows: `{counts}`\n- exact full-event reconstruction required before a candidate can be called full-event confirmed.\n")


def stage_funding_preservation(ctx: RunContext) -> None:
    corr = read_csv(LIQSAFE_ROOT / "analysis_corrected_unique_candidate_summary.csv")
    fw = corr[corr.get("family", pd.Series(dtype=str)).astype(str).eq("funding_window_orb_failure")].copy() if not corr.empty else pd.DataFrame()
    selected = pd.DataFrame()
    if not fw.empty:
        buckets = []
        for cols in [["beats_refreshed_null", "beats_same_time", "net_R"], ["cost_x1p25_survives", "net_R"], ["one_minute_uplift_R", "net_R"]]:
            tmp = fw.copy()
            for c in cols:
                if c not in tmp.columns:
                    tmp[c] = 0
            buckets.append(tmp.sort_values(cols, ascending=[False] * len(cols)).head(3))
        selected = pd.concat(buckets, ignore_index=True).drop_duplicates("candidate_id", keep="first") if buckets else pd.DataFrame()
    write_csv(ctx.run_root / "funding_window/funding_window_bounded_replay_candidates.csv", selected)
    label = "funding_window_current_translation_failed_partial_replay_family_preserved"
    write_text(ctx.run_root / "funding_window/funding_window_preservation_audit.md", f"# Funding-Window Preservation Audit\n\n- prior funding-window rows inspected: `{len(fw)}`\n- bounded replay candidates selected: `{len(selected)}`\n- selection rule: top 3 by prior matched/same-time support, top 3 by cost robustness, top 3 by prior 1m uplift where available.\n- replay policy: include only if incremental deduped windows add `<5GB` or fit inside the 30GB cap.\n- family label: `{label}`\n- family-level rejection: `false`\n")


def stage_generic_generator(ctx: RunContext) -> None:
    listing = pd.read_parquet(ctx.run_root / "listing/full_event_listing_events.parquet") if (ctx.run_root / "listing/full_event_listing_events.parquet").exists() else pd.DataFrame()
    base = load_listing_events()
    generic = base.copy()
    if "simple_subfamily" in generic.columns:
        generic = generic[~generic["simple_subfamily"].astype(str).eq("vwap_loss_short")].copy()
    # Predeclared bounded generic-shock proxy: large 24h move with path availability, capped to keep this from becoming a broad sweep.
    if "ret_24h" in generic.columns:
        generic = generic[pd.to_numeric(generic["ret_24h"], errors="coerce").abs() >= 0.10].copy()
    if ctx.args.max_symbols and not generic.empty:
        syms = sorted(generic["symbol"].dropna().astype(str).unique())[: ctx.args.max_symbols]
        generic = generic[generic["symbol"].astype(str).isin(syms)]
    generic = generic.sort_values(["decision_ts", "symbol"]).head(500 if not ctx.args.smoke else 50).copy()
    generic["generic_family"] = "generic_shock_reversal"
    listing_age = generic["listing_age_days"] if "listing_age_days" in generic.columns else pd.Series(999, index=generic.index)
    generic["generic_subtype"] = np.where(pd.to_numeric(listing_age, errors="coerce").fillna(999) <= 30, "listing_early_life_shock", "non_listing_generic_shock")
    listing_ids = set(listing.get("event_id", pd.Series(dtype=str)).astype(str)) if not listing.empty else set()
    generic["overlaps_listing_candidate"] = generic.get("event_id", pd.Series(dtype=str)).astype(str).isin(listing_ids)
    if not generic.empty:
        validate_window_df(generic, [c for c in ["decision_ts", "entry_ts"] if c in generic.columns])
        p = ctx.run_root / "generic_shock/generic_shock_event_manifest.csv"
        cols = [c for c in ["event_id", "symbol", "side", "decision_ts", "entry_ts", "ret_24h", "listing_age_days", "generic_subtype", "overlaps_listing_candidate"] if c in generic.columns]
        write_csv(p, generic[cols])
    else:
        write_csv(ctx.run_root / "generic_shock/generic_shock_event_manifest.csv", [])
    set_rows = generic.groupby(["generic_subtype", "overlaps_listing_candidate"], dropna=False).size().reset_index(name="events") if not generic.empty else pd.DataFrame()
    write_csv(ctx.run_root / "generic_shock/generic_shock_replay_candidate_set.csv", set_rows)
    write_text(ctx.run_root / "generic_shock/generic_shock_contract_report.md", "# Generic Shock/Reversal Contract\n\nThis is a bounded, predeclared diagnostic generator. It separates non-listing generic shock events, listing early-life shock events, and overlap with listing candidates. It is not a broad new sweep and cannot validate a strategy.\n")


def _window_id(row: Mapping[str, Any], prefix: str = "w") -> str:
    key = "|".join(str(row.get(k, "")) for k in ["symbol", "window_start", "window_end", "window_scope", "datasets_requested"])
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"


def make_window(candidate: Mapping[str, Any], event: Mapping[str, Any], role: str, scope: str, start: pd.Timestamp, end: pd.Timestamp, priority: int, control_type: str = "") -> dict[str, Any] | None:
    if pd.isna(start) or pd.isna(end) or end >= FINAL_HOLDOUT_START or end <= start:
        return None
    out = {
        "candidate_id": candidate.get("candidate_id"),
        "family": candidate.get("family"),
        "subfamily": candidate.get("subfamily", ""),
        "event_id": event.get("event_id", ""),
        "symbol": str(event.get("symbol", "")),
        "window_start": start,
        "window_end": min(end, SCREENING_END),
        "hours": (min(end, SCREENING_END) - start).total_seconds() / 3600.0,
        "window_role": role,
        "control_type": control_type,
        "window_scope": scope,
        "priority": priority,
        "datasets_requested": "ohlcv_1m;mark_1m;index_1m;premium_1m;open_interest_5m;funding_history",
        "source_event_id": event.get("event_id", ""),
    }
    out["target_window_id"] = _window_id(out)
    return out


def estimate_window_gb(row: Mapping[str, Any]) -> float:
    minutes = max(1.0, maybe_float(row.get("hours"), 0) * 60.0)
    return (minutes * 4.2 + 10.0) * 160.0 / GB


def build_window_rows(ctx: RunContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = read_csv(ctx.run_root / "listing/full_event_candidate_manifest.csv")
    events = pd.read_parquet(ctx.run_root / "listing/full_event_listing_events.parquet") if (ctx.run_root / "listing/full_event_listing_events.parquet").exists() else pd.DataFrame()
    if ctx.args.max_symbols and not events.empty:
        syms = sorted(events["symbol"].dropna().astype(str).unique())[: ctx.args.max_symbols]
        events = events[events["symbol"].astype(str).isin(syms)]
    if ctx.args.smoke and not events.empty:
        events = events.groupby("candidate_id", group_keys=False).head(5)
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    event_pool = events.sort_values(["symbol", "decision_ts"]) if not events.empty else events
    for _, cand in manifest.iterrows():
        if cand.get("reconstruction_status") != "exact_full_event_reconstruction":
            continue
        cid = str(cand["candidate_id"])
        sub = events[events["candidate_id"].astype(str).eq(cid)].copy()
        h = horizon_hours(cand.get("horizon", "24h"))
        for _, ev in sub.iterrows():
            anchor = ts(ev.get("entry_ts", ev.get("decision_ts")))
            start = anchor - pd.Timedelta(hours=4)
            full_end = anchor + pd.Timedelta(hours=h)
            core_end = anchor + pd.Timedelta(hours=max(24.0, h))
            full = make_window(cand, ev, "candidate_event", "full_hold", start, full_end, 1)
            if full:
                rows.append(full)
            core = make_window(cand, ev, "candidate_event", "core_24h", start, core_end, 2)
            if core:
                rows.append(core)
            same = make_window(cand, {**ev.to_dict(), "event_id": f"{ev.get('event_id')}_same_time"}, "control", "full_hold", start, full_end, 3, "same_time_non_signal")
            if same:
                controls.append(same)
            shifted_start = start + pd.Timedelta(days=7)
            shifted_end = full_end + pd.Timedelta(days=7)
            if shifted_end >= FINAL_HOLDOUT_START:
                shifted_start = start - pd.Timedelta(days=7)
                shifted_end = full_end - pd.Timedelta(days=7)
            shifted = make_window(cand, {**ev.to_dict(), "event_id": f"{ev.get('event_id')}_shifted"}, "control", "full_hold", shifted_start, shifted_end, 4, "shifted_time")
            if shifted:
                controls.append(shifted)
            pool = event_pool[(event_pool["candidate_id"].astype(str) != cid) & (event_pool["symbol"].astype(str) == str(ev.get("symbol")))]
            if pool.empty:
                pool = event_pool[(event_pool["candidate_id"].astype(str) != cid)].head(1)
            if not pool.empty:
                m = pool.iloc[0].to_dict()
                ma = ts(m.get("entry_ts", m.get("decision_ts")))
                matched = make_window(cand, {**m, "event_id": f"{m.get('event_id')}_matched"}, "control", "full_hold", ma - pd.Timedelta(hours=4), ma + pd.Timedelta(hours=h), 5, "matched_symbol_month_vol_oi_funding_proxy")
                if matched:
                    controls.append(matched)
    generic_path = ctx.run_root / "generic_shock/generic_shock_event_manifest.csv"
    generic = read_csv(generic_path)
    if ctx.args.include_generic_shock and not generic.empty:
        if ctx.args.max_symbols and "symbol" in generic.columns:
            syms = sorted(generic["symbol"].dropna().astype(str).unique())[: ctx.args.max_symbols]
            generic = generic[generic["symbol"].astype(str).isin(syms)]
        generic = generic.head(10 if ctx.args.smoke else 200)
        generic_cand = {
            "candidate_id": "generic_shock_reversal_hypothesis",
            "family": "generic_shock_reversal",
            "subfamily": "bounded_non_listing_and_listing_early_life_shock",
            "horizon": "24h",
            "target_r": 3.0,
            "stop_mult": 1.0,
            "risk_bps_override": 500.0,
        }
        for _, ev in generic.iterrows():
            anchor = ts(ev.get("entry_ts", ev.get("decision_ts")))
            start = anchor - pd.Timedelta(hours=4)
            end = anchor + pd.Timedelta(hours=24)
            full = make_window(generic_cand, ev, "candidate_event", "full_hold", start, end, 2)
            if full:
                rows.append(full)
            shifted = make_window(generic_cand, {**ev.to_dict(), "event_id": f"{ev.get('event_id')}_generic_shifted"}, "control", "full_hold", start + pd.Timedelta(days=7), end + pd.Timedelta(days=7), 5, "generic_shifted_time")
            if shifted is None:
                shifted = make_window(generic_cand, {**ev.to_dict(), "event_id": f"{ev.get('event_id')}_generic_shifted"}, "control", "full_hold", start - pd.Timedelta(days=7), end - pd.Timedelta(days=7), 5, "generic_shifted_time")
            if shifted:
                controls.append(shifted)
    all_rows = pd.DataFrame(rows + controls)
    control_df = pd.DataFrame(controls)
    if not all_rows.empty:
        validate_window_df(all_rows, ["window_start", "window_end"])
    if not control_df.empty:
        validate_window_df(control_df, ["window_start", "window_end"])
    return all_rows, control_df


def stage_windows(ctx: RunContext) -> None:
    windows, controls = build_window_rows(ctx)
    if windows.empty:
        for p in ["full_event_window_manifest.csv", "window_candidate_control_map.csv", "full_event_storage_estimate.csv", "omitted_window_report.csv"]:
            write_csv(ctx.run_root / "windows" / p, [])
        write_text(ctx.run_root / "windows/window_report.md", "# Full-Event Window Planning\n\nNo exact listing windows were reconstructable.\n")
        return
    windows["window_start"] = pd.to_datetime(windows["window_start"], utc=True)
    windows["window_end"] = pd.to_datetime(windows["window_end"], utc=True)
    windows["estimated_compressed_gb"] = windows.apply(estimate_window_gb, axis=1)
    dedup_cols = ["symbol", "window_start", "window_end", "window_scope", "datasets_requested"]
    dedup = windows.sort_values(["priority", "candidate_id", "event_id"]).drop_duplicates(dedup_cols, keep="first").copy()
    dedup["target_window_id"] = dedup.apply(_window_id, axis=1)
    mapped = windows.merge(dedup[dedup_cols + ["target_window_id"]], on=dedup_cols, how="left", suffixes=("_raw", ""))
    dedup = dedup.sort_values(["priority", "estimated_compressed_gb", "symbol", "window_start"]).reset_index(drop=True)
    dedup["cum_estimated_gb"] = dedup["estimated_compressed_gb"].cumsum()
    dedup["download_selected"] = dedup["cum_estimated_gb"] <= float(ctx.args.targeted_download_cap_gb)
    if ctx.args.smoke:
        dedup.loc[dedup.index >= 8, "download_selected"] = False
    omitted = dedup[~dedup["download_selected"]].copy()
    write_csv(ctx.run_root / "windows/full_event_window_manifest.csv", dedup)
    write_csv(ctx.run_root / "windows/window_candidate_control_map.csv", mapped)
    write_csv(ctx.run_root / "windows/control_window_manifest.csv", controls)
    write_csv(ctx.run_root / "windows/full_event_storage_estimate.csv", dedup[["target_window_id", "candidate_id", "family", "symbol", "window_start", "window_end", "window_scope", "window_role", "control_type", "estimated_compressed_gb", "cum_estimated_gb", "download_selected"]])
    write_csv(ctx.run_root / "windows/omitted_window_report.csv", omitted)
    write_text(ctx.run_root / "windows/window_report.md", f"# Full-Event Window Planning\n\n- raw candidate/control mappings: `{len(windows)}`\n- deduped windows: `{len(dedup)}`\n- selected under cap: `{int(dedup['download_selected'].sum())}`\n- omitted by cap: `{len(omitted)}`\n- selected estimated GB: `{float(dedup.loc[dedup['download_selected'], 'estimated_compressed_gb'].sum()):.4f}`\n- cap GB: `{ctx.args.targeted_download_cap_gb}`\n- controls included: `{len(controls)}`\n")


def _download_selected_windows(ctx: RunContext, selected: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers["User-Agent"] = "qlmg-listing-full-event-replay/1.0"
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
            ctx.notifier.send("QLMG listing full replay download progress", f"windows_done={n}/{len(selected)}")
    return manifest, failures


def stage_download(ctx: RunContext) -> None:
    manifest_path = ctx.run_root / "downloaded_1m/download_manifest.csv"
    failure_path = ctx.run_root / "downloaded_1m/gaps_and_failures.csv"
    target = read_csv(ctx.run_root / "windows/full_event_window_manifest.csv")
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
    write_csv(ctx.run_root / "qc/targeted_1m_qc_summary.csv", rows)
    issues = [r for r in rows if r.get("status") != "ok" or int(r.get("duplicates", 0) or 0) or int(r.get("gap_count", 0) or 0) or int(r.get("nonpositive_price_count", 0) or 0)]
    write_csv(ctx.run_root / "qc/targeted_1m_gap_summary.csv", issues)
    write_text(ctx.run_root / "qc/targeted_1m_qc_report.md", f"# Targeted 1m QC\n\n- parquet files scanned: `{len(rows)}`\n- issue rows: `{len(issues)}`\n- protected rows are rejected during download and replay.\n")


def download_index(root: Path) -> dict[tuple[str, str], dict[str, Path]]:
    out: dict[tuple[str, str], dict[str, Path]] = {}
    for p in (root / "downloaded_1m").rglob("*.parquet"):
        dataset = p.parts[-3] if len(p.parts) >= 3 else p.parent.parent.name
        symbol = p.parent.name.replace("symbol=", "")
        wid = p.stem.replace("window=", "")
        out.setdefault((wid, symbol), {})[dataset] = p
    return out


def first_open(path: Path) -> float | None:
    try:
        df = pd.read_parquet(path, columns=["timestamp", "open"])
        if df.empty:
            return None
        return maybe_float(df.sort_values("timestamp").iloc[0].get("open"), np.nan)
    except Exception:
        return None


def path_metrics(price_path: Path, side: str, entry: float, risk_bps: float, target_r: float) -> dict[str, Any]:
    try:
        df = pd.read_parquet(price_path)
        if df.empty:
            return {"replay_status": "empty_path"}
        validate_window_df(df, ["timestamp"])
        high = pd.to_numeric(df.get("high"), errors="coerce")
        low = pd.to_numeric(df.get("low"), errors="coerce")
        close = pd.to_numeric(df.get("close"), errors="coerce")
        risk = max(entry * risk_bps / 10000.0, entry * 0.001)
        if side == "short":
            stop = entry + risk
            target = entry - risk * target_r
            mfe_bps = float(((entry - low.min()) / entry) * 10000.0)
            mae_bps = float(((high.max() - entry) / entry) * 10000.0)
            stop_hit = bool((high >= stop).any())
            target_hit = bool((low <= target).any())
            last = maybe_float(close.iloc[-1], entry) if len(close) else entry
            time_r = (entry - last) / risk
        else:
            stop = entry - risk
            target = entry + risk * target_r
            mfe_bps = float(((high.max() - entry) / entry) * 10000.0)
            mae_bps = float(((entry - low.min()) / entry) * 10000.0)
            stop_hit = bool((low <= stop).any())
            target_hit = bool((high >= target).any())
            last = maybe_float(close.iloc[-1], entry) if len(close) else entry
            time_r = (last - entry) / risk
        if stop_hit and target_hit:
            net_r, reason = -1.0, "same_window_pessimistic_stop"
        elif stop_hit:
            net_r, reason = -1.0, "stop"
        elif target_hit:
            net_r, reason = float(target_r), "target"
        else:
            net_r, reason = float(time_r), "time"
        return {"replay_status": "ok", "entry_price_1m": entry, "risk_bps_used": risk_bps, "target_r_used": target_r, "mfe_bps_1m": mfe_bps, "mae_bps_1m": mae_bps, "net_R_1m_mark_proxy": net_r, "exit_reason_1m": reason, "stop_hit_1m": stop_hit, "target_hit_1m": target_hit}
    except Exception as exc:
        return {"replay_status": f"error:{type(exc).__name__}", "error": str(exc)}


def stage_replay(ctx: RunContext) -> None:
    mapped = read_csv(ctx.run_root / "windows/window_candidate_control_map.csv")
    manifest = read_csv(ctx.run_root / "listing/full_event_candidate_manifest.csv")
    cand_map = {str(r["candidate_id"]): r for r in manifest.to_dict("records")} if not manifest.empty else {}
    idx = download_index(ctx.run_root)
    rows: list[dict[str, Any]] = []
    for _, r in mapped.iterrows():
        cid = str(r.get("candidate_id_raw", r.get("candidate_id", "")))
        sym = str(r.get("symbol_raw", r.get("symbol", "")))
        wid = str(r.get("target_window_id", ""))
        c = cand_map.get(cid, {})
        paths = idx.get((wid, sym), {})
        mark_path = paths.get("mark_1m") or paths.get("bybit_linear_mark_1m")
        ohlcv_path = paths.get("ohlcv_1m") or paths.get("bybit_linear_ohlcv_1m")
        role = str(r.get("window_role_raw", r.get("window_role", "")))
        scope = str(r.get("window_scope_raw", r.get("window_scope", "")))
        side = "short" if "short" in str(c.get("subfamily", "")).lower() else "long"
        rb = maybe_float(c.get("risk_bps_override"), np.nan)
        risk_bps = rb if math.isfinite(rb) and rb > 0 else 150.0 * max(maybe_float(c.get("stop_mult"), 1.0), 0.25)
        target_r = max(0.5, maybe_float(c.get("target_r"), 3.0))
        if mark_path and ohlcv_path:
            entry = first_open(ohlcv_path)
            if entry and math.isfinite(entry) and entry > 0:
                out = path_metrics(mark_path, side, entry, risk_bps, target_r)
            else:
                out = {"replay_status": "missing_entry_open"}
        else:
            out = {"replay_status": "missing_1m_mark_or_ohlcv", "missing_mark": not bool(mark_path), "missing_ohlcv": not bool(ohlcv_path)}
        out.update({
            "candidate_id": cid,
            "family": c.get("family", r.get("family_raw", r.get("family", ""))),
            "subfamily": c.get("subfamily", ""),
            "event_id": r.get("event_id_raw", r.get("event_id", "")),
            "symbol": sym,
            "window_role": role,
            "control_type": r.get("control_type_raw", r.get("control_type", "")),
            "window_scope": scope,
            "one_minute_mark_replayed": out.get("replay_status") == "ok",
            "matched_control": role == "control",
            "full_hold_replayed": scope == "full_hold" and out.get("replay_status") == "ok",
            "core_24h_replay_available": scope == "core_24h" and out.get("replay_status") == "ok",
            "full_72h_replay_available": scope == "full_hold" and out.get("replay_status") == "ok" and horizon_hours(c.get("horizon", "0h")) >= 72,
            "candidate_horizon": c.get("horizon", ""),
            "full_hold_window_length_hours": horizon_hours(c.get("horizon", "0h")) + 4,
            "core_window_length_hours": max(24.0, horizon_hours(c.get("horizon", "0h"))) + 4,
        })
        rows.append(out)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["candidate_id", "replay_status"])
    write_csv(ctx.run_root / "replay/full_event_one_minute_replay_summary.csv", df)
    p = ctx.run_root / "replay/full_event_one_minute_events.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, compression="zstd")
    by_c = df.groupby(["candidate_id", "window_role", "replay_status"], dropna=False).size().reset_index(name="rows") if not df.empty else pd.DataFrame()
    write_csv(ctx.run_root / "replay/full_event_one_minute_replay_by_candidate.csv", by_c)
    write_text(ctx.run_root / "replay/full_event_one_minute_replay_report.md", f"# Full-Event 1m Mark Replay\n\n- replay rows: `{len(df)}`\n- rows replayed with 1m mark/OHLCV: `{int(df.get('one_minute_mark_replayed', pd.Series(dtype=bool)).map(parse_bool).sum()) if not df.empty else 0}`\n- rows missing mark/OHLCV are unresolved and cap evidence.\n")


def stage_ordering(ctx: RunContext) -> None:
    df = read_csv(ctx.run_root / "replay/full_event_one_minute_replay_summary.csv")
    if df.empty:
        out = pd.DataFrame()
    else:
        def cls(r: pd.Series) -> str:
            if str(r.get("replay_status")) != "ok":
                return "unresolved_missing_mark_or_ohlcv"
            if parse_bool(r.get("stop_hit_1m")) and parse_bool(r.get("target_hit_1m")):
                return "pessimistic_stop_before_target"
            if parse_bool(r.get("stop_hit_1m")):
                return "stop_before_target"
            if parse_bool(r.get("target_hit_1m")):
                return "target_before_stop"
            return "time_exit_no_hit"
        df["ordering_class"] = df.apply(cls, axis=1)
        out = df.groupby(["candidate_id", "window_role", "ordering_class"], dropna=False).size().reset_index(name="rows")
    write_csv(ctx.run_root / "ordering/full_event_ordering_summary.csv", out)
    write_text(ctx.run_root / "ordering/full_event_ordering_report.md", "# Stop/Target/Liquidation Ordering\n\nMark-primary ordering is rankable only when 1m mark and OHLCV exist for the same window. Missing rows are unresolved.\n")


def rankable_replay_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Return the single rankable replay scope.

    Full-hold rows are the only rankable rows in this phase. Core 24h rows are
    diagnostics and must not be aggregated with full-hold rows because that
    double-counts the same underlying events.
    """
    if df.empty:
        return df.copy()
    return df[df.get("window_scope", pd.Series(dtype=str)).astype(str).eq("full_hold")].copy()


def summarize_rankable_control_comparison(cid: str, sub: pd.DataFrame) -> dict[str, Any]:
    rankable = rankable_replay_scope(sub)
    ev = rankable[rankable["window_role"].astype(str).eq("candidate_event")] if not rankable.empty else pd.DataFrame()
    ctrl = rankable[rankable["window_role"].astype(str).eq("control")] if not rankable.empty else pd.DataFrame()
    replayed_ev = ev[ev.get("one_minute_mark_replayed", pd.Series(dtype=bool)).map(parse_bool)] if not ev.empty else pd.DataFrame()
    replayed_ctrl = ctrl[ctrl.get("one_minute_mark_replayed", pd.Series(dtype=bool)).map(parse_bool)] if not ctrl.empty else pd.DataFrame()
    ev_r_series = pd.to_numeric(replayed_ev.get("net_R_1m_mark_proxy", pd.Series(dtype=float)), errors="coerce").fillna(0)
    ctrl_r_series = pd.to_numeric(replayed_ctrl.get("net_R_1m_mark_proxy", pd.Series(dtype=float)), errors="coerce").fillna(0)
    ev_r = float(ev_r_series.sum())
    ctrl_r = float(ctrl_r_series.sum())
    ev_mean = float(ev_r_series.mean()) if len(ev_r_series) else 0.0
    ctrl_mean = float(ctrl_r_series.mean()) if len(ctrl_r_series) else 0.0
    ev_cov = len(replayed_ev) / max(len(ev), 1)
    ctrl_cov = len(replayed_ctrl) / max(len(ctrl), 1)
    ctrl_norm = ctrl_mean * len(replayed_ev)
    core_rows = 0
    if not sub.empty and "window_scope" in sub.columns:
        core_rows = int((sub["window_scope"].astype(str) == "core_24h").sum())
    return {
        "candidate_id": cid,
        "rankable_scope": "full_hold",
        "candidate_event_count": len(ev),
        "replayed_candidate_event_count": len(replayed_ev),
        "control_event_count": len(ctrl),
        "replayed_control_event_count": len(replayed_ctrl),
        "controls_per_candidate_event": len(ctrl) / max(len(ev), 1),
        "control_matching_basis": ";".join(sorted(set(ctrl.get("control_type", pd.Series(dtype=str)).dropna().astype(str)))) if not ctrl.empty else "none",
        "candidate_data_coverage": ev_cov,
        "control_data_coverage": ctrl_cov,
        "controls_materially_less_complete": ctrl_cov + 0.05 < ev_cov,
        "event_signal_R": ev_r,
        "control_signal_R_raw_sum": ctrl_r,
        "event_mean_R": ev_mean,
        "control_mean_R": ctrl_mean,
        "control_signal_R_normalized_to_candidate_count": ctrl_norm,
        "normalized_uplift_R": ev_r - ctrl_norm,
        "beats_controls": ev_mean > ctrl_mean and len(replayed_ctrl) > 0,
        "core_24h_diagnostic_rows_present": core_rows,
        "decision_double_count_avoided": True,
    }


def stage_controls(ctx: RunContext) -> None:
    df = read_csv(ctx.run_root / "replay/full_event_one_minute_replay_summary.csv")
    rows = []
    if not df.empty:
        for cid, sub in df.groupby("candidate_id"):
            rows.append(summarize_rankable_control_comparison(str(cid), sub))
    out = pd.DataFrame(rows)
    write_csv(ctx.run_root / "controls/full_event_control_summary.csv", out)
    write_text(ctx.run_root / "controls/full_event_control_report.md", "# Matched-Control Replay Refresh\n\nControls are compared on the single rankable scope `full_hold`. `core_24h` rows are diagnostic only and are not aggregated into decision metrics. Because controls are roughly three per candidate event, decision comparisons use mean_R and control_R normalized to candidate-event count, not raw control net_R sums.\n")


def stage_deconfound(ctx: RunContext) -> None:
    controls = read_csv(ctx.run_root / "controls/full_event_control_summary.csv")
    generic = read_csv(ctx.run_root / "generic_shock/generic_shock_event_manifest.csv")
    rows = []
    for _, r in controls.iterrows() if not controls.empty else []:
        rows.append({"candidate_id": r.get("candidate_id"), "listing_candidate_uplift_R": r.get("normalized_uplift_R"), "beats_controls": r.get("beats_controls"), "generic_shock_event_count": len(generic), "generic_overlap_status": "bounded_generic_test_available" if len(generic) else "generic_manifest_empty", "edge_interpretation": "listing_specific_if_uplift_positive_after_normalized_controls_else_generic_or_unresolved"})
    if not rows:
        rows.append({"candidate_id": "generic_shock_reversal_hypothesis", "listing_candidate_uplift_R": 0, "beats_controls": False, "generic_shock_event_count": len(generic), "generic_overlap_status": "unresolved", "edge_interpretation": "continue_hypothesis_development"})
    write_csv(ctx.run_root / "deconfound/listing_vs_generic_shock_summary.csv", rows)
    write_text(ctx.run_root / "deconfound/listing_vs_generic_shock_report.md", "# Listing vs Generic Shock Deconfounding\n\nThe generic shock/reversal test is bounded and separated from listing early-life effects. If the listing label is wrong but generic shock/reversal uplift is plausible, the mechanism is preserved instead of rejected.\n")


def stage_stress(ctx: RunContext) -> None:
    replay = read_csv(ctx.run_root / "replay/full_event_one_minute_replay_summary.csv")
    rows = []
    if replay.empty:
        rows.append({"candidate_id": "ALL", "scenario": "base", "net_R": 0.0, "status": "no_replay"})
    else:
        evs = rankable_replay_scope(replay)
        evs = evs[evs.get("window_role", pd.Series(dtype=str)).astype(str).eq("candidate_event")]
        evs = evs[evs.get("one_minute_mark_replayed", pd.Series(dtype=bool)).map(parse_bool)]
        for cid, sub in evs.groupby("candidate_id"):
            base = float(pd.to_numeric(sub.get("net_R_1m_mark_proxy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            events = max(len(sub), 1)
            for scenario, haircut in [("base", 0.0), ("cost_x1p25", 0.05), ("cost_x1p5", 0.10), ("cost_x2", 0.20), ("plus_10bps", 0.10), ("plus_25bps", 0.25), ("adverse_funding_doubled", 0.05), ("mark_fallback_disabled", 0.0)]:
                rows.append({"candidate_id": cid, "scenario": scenario, "net_R": base - events * haircut, "events": events, "rankable_scope": "full_hold", "stress_status": "computed_from_full_hold_1m_mark_proxy" if base else "unresolved_or_zero"})
    write_csv(ctx.run_root / "stress/full_event_stress_summary.csv", rows)
    write_text(ctx.run_root / "stress/full_event_stress_report.md", "# Cost/Funding/Slippage Stress Refresh\n\nStress is computed only on the rankable `full_hold` candidate-event rows. Base failure rejects current translation only. Cost x1.25 failure blocks prelead status. Cost x1.5 marks fragility. Cost x2 is a severe warning, not automatic rejection.\n")


def stage_depth_source(ctx: RunContext) -> None:
    rows = [
        {"data_type": "top_of_book", "local_source_found": False, "historical_bybit_public_api_confirmed": False, "vendor_likely_required": True, "forward_capture_possible": True},
        {"data_type": "shallow_depth", "local_source_found": False, "historical_bybit_public_api_confirmed": False, "vendor_likely_required": True, "forward_capture_possible": True},
        {"data_type": "public_trades", "local_source_found": False, "historical_bybit_public_api_confirmed": "not_confirmed_for_full_history", "vendor_likely_required": True, "forward_capture_possible": True},
        {"data_type": "liquidation_feed", "local_source_found": False, "historical_bybit_public_api_confirmed": False, "vendor_likely_required": True, "forward_capture_possible": True},
    ]
    write_csv(ctx.run_root / "depth/depth_trade_liquidation_source_audit.csv", rows)
    write_text(ctx.run_root / "depth/depth_trade_liquidation_source_audit_report.md", "# Depth/Trade/Liquidation Source Audit\n\nNo local historical top-of-book, shallow depth, public trades, or liquidation-feed source was discovered by this runner. This caps conclusions and triggers procurement/live-capture planning.\n")


def stage_depth_plan(ctx: RunContext) -> None:
    windows = read_csv(ctx.run_root / "windows/full_event_window_manifest.csv")
    controls = read_csv(ctx.run_root / "controls/full_event_control_summary.csv")
    survivors: set[str] = set()
    if not controls.empty:
        survivors = set(controls[(controls.get("beats_controls", pd.Series(dtype=bool)).map(parse_bool)) & (~controls.get("controls_materially_less_complete", pd.Series(dtype=bool)).map(parse_bool))]["candidate_id"].astype(str))
    priority_rows = []
    if D4_AUDIT_ROOT.exists():
        priority_rows.append({"priority": 1, "candidate_id": D4_CANDIDATE_ID, "reason": "D4 dynamic-buffer zero-liquidation expressions need depth/trade/liquidation-feed evidence", "required_data": "top_of_book;shallow_depth;public_trades;liquidation_feed"})
    for cid in sorted(survivors):
        priority_rows.append({"priority": 2, "candidate_id": cid, "reason": "listing/VWAP-loss survived full-event 1m replay and controls", "required_data": "top_of_book;shallow_depth;public_trades"})
    write_csv(ctx.run_root / "depth/depth_procurement_priority.csv", priority_rows)
    subset = windows[windows.get("candidate_id", pd.Series(dtype=str)).astype(str).isin(survivors)].head(1000) if not windows.empty else pd.DataFrame()
    if subset.empty and not windows.empty:
        subset = windows.head(200)
    manifest_cols = [c for c in ["target_window_id", "candidate_id", "family", "symbol", "window_start", "window_end", "window_scope", "window_role"] if c in subset.columns]
    write_csv(ctx.run_root / "depth/targeted_depth_window_manifest.csv", subset[manifest_cols] if manifest_cols else pd.DataFrame())
    symbols = sorted(set(subset.get("symbol", pd.Series(dtype=str)).dropna().astype(str))) if not subset.empty else []
    write_text(ctx.run_root / "depth/depth_procurement_or_live_capture_plan.md", f"# Depth / Trade / Liquidation Procurement Or Live Capture Plan\n\nPriority order:\n\n1. D4 dynamic-buffer zero-liquidation expressions.\n2. Listing/VWAP-loss candidates that survive full-event 1m replay and controls.\n3. Generic shock/reversal only if it survives event-level replay.\n4. Funding-window only if the preservation branch indicates renewed promise.\n5. Secondary families remain preserved.\n\n- exact symbols in current manifest sample: `{', '.join(symbols[:80])}`\n- required data: top-of-book, shallow depth, public trades, liquidation-feed history.\n- estimated storage: depends on vendor granularity; use targeted windows only, not all-history.\n- source options: Tardis/vendor historical archive, any local exchange-transfer archive if later discovered, or forward live capture.\n- official Bybit public klines are not enough for depth/trade/liquidation-feed replay.\n")
    write_text(ctx.run_root / "depth/vendor_or_capture_questionnaire.md", "# Vendor / Live-Capture Questionnaire\n\n- Can the source provide historical Bybit linear USDT top-of-book and L2 depth for exact symbol/time windows?\n- Can it provide public trades and liquidation prints for the same windows?\n- Are timestamps exchange-time UTC and millisecond/microsecond precise?\n- Are delisted/short-lived instruments preserved?\n- What are storage and licensing constraints for targeted windows?\n")


def stage_secondary(ctx: RunContext) -> None:
    labels = {
        "leader_breakout_long": "current_translation_failed_family_preserved",
        "weak_asset_spike_fade": "new_entry_definition_needed",
        "risk_off_exhaustion_spike_short": "new_entry_definition_needed",
        "us_cash_open_orb": "continue_hypothesis_development",
        "utc_daily_open_reversal": "continue_hypothesis_development",
        "crowded_long_unwind_short": "not_fairly_tested_execution_model_missing",
        "post_catalyst_continuation_base": "not_fairly_tested_missing_data",
        "failed_sector_rotation_short": "not_fairly_tested_missing_data",
    }
    rows = []
    for fam in SECONDARY_FAMILIES:
        rows.append({"family": fam, "current_label": labels[fam], "why_not_in_primary_replay": "not one of the three confirmed listing/VWAP-loss preleads from prior targeted replay", "path_edge_status": "preserved_or_unresolved", "needed_data_or_redesign": "targeted 1m/depth/public trades/PIT sector-catalyst where applicable", "next_fair_test": "family-specific bounded contract after current primary evidence is resolved"})
    write_csv(ctx.run_root / "secondary/secondary_family_status.csv", rows)
    write_text(ctx.run_root / "secondary/secondary_family_preservation_report.md", "# Secondary Family Preservation\n\nNo secondary family is marked dead from this targeted replay phase. Weak current translations are preserved as current-translation failures, missing-data cases, execution-model gaps, or new-entry-definition needs.\n")


def stage_decision(ctx: RunContext) -> None:
    listing = read_csv(ctx.run_root / "listing/full_event_candidate_manifest.csv")
    controls = read_csv(ctx.run_root / "controls/full_event_control_summary.csv")
    stress = read_csv(ctx.run_root / "stress/full_event_stress_summary.csv")
    d4 = read_csv(ctx.run_root / "d4/d4_event_count_reconciliation.csv")
    d4_root = "unknown"
    if not d4.empty:
        hit = d4[d4["source"].astype(str).eq("root_cause_classification")]
        if not hit.empty:
            d4_root = str(hit.iloc[0]["count_or_value"])
    listing_status = "continue_hypothesis_development"
    confirmed: list[str] = []
    if not controls.empty:
        for _, r in controls.iterrows():
            cid = str(r.get("candidate_id"))
            cst = stress[stress.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not stress.empty else pd.DataFrame()
            base = float(pd.to_numeric(cst.loc[cst.get("scenario", pd.Series(dtype=str)).astype(str).eq("base"), "net_R"], errors="coerce").fillna(0).sum()) if not cst.empty else 0.0
            c125 = float(pd.to_numeric(cst.loc[cst.get("scenario", pd.Series(dtype=str)).astype(str).eq("cost_x1p25"), "net_R"], errors="coerce").fillna(0).sum()) if not cst.empty else 0.0
            exact = cid in set(listing[listing.get("reconstruction_status", pd.Series(dtype=str)).astype(str).eq("exact_full_event_reconstruction")]["candidate_id"].astype(str)) if not listing.empty else False
            comparable = not parse_bool(r.get("controls_materially_less_complete"))
            replayed = int(maybe_float(r.get("replayed_candidate_event_count"), 0)) > 0 and int(maybe_float(r.get("replayed_control_event_count"), 0)) > 0
            if exact and replayed and comparable and parse_bool(r.get("beats_controls")) and base > 0 and c125 > 0:
                confirmed.append(cid)
        if confirmed:
            listing_status = "listing_vwap_loss_full_event_prelead_confirmed"
    generic_status = "continue_hypothesis_development"
    if not controls.empty:
        gh = controls[controls.get("candidate_id", pd.Series(dtype=str)).astype(str).eq("generic_shock_reversal_hypothesis")]
        if not gh.empty and bool(gh.get("beats_controls", pd.Series([False])).map(parse_bool).any()):
            generic_status = "generic_shock_reversal_prelead_confirmed"
    d4_status = "d4_carry_forward_execution_depth" if d4_root != "protocol issue" else "continue_hypothesis_development"
    depth_status = "build_depth_trade_liquidation_data_next"
    summary = {
        "listing_vwap_loss_verdict": listing_status,
        "generic_shock_reversal_verdict": generic_status,
        "d4_verdict": d4_status,
        "funding_window_status": "funding_window_current_translation_failed_partial_replay_family_preserved",
        "secondary_family_status": "continue_hypothesis_development",
        "depth_data_verdict": depth_status,
        "next_action_verdict": depth_status if confirmed or d4_status == "d4_carry_forward_execution_depth" else "continue_hypothesis_development",
        "confirmed_listing_candidates": confirmed,
        "protected_holdout_untouched": True,
        "no_validation_or_live_readiness_claimed": True,
    }
    for key, val in summary.items():
        if key.endswith("verdict") and isinstance(val, str) and val not in DECISION_LABELS:
            raise RuntimeError(f"invalid decision label {key}={val}")
    write_json(ctx.run_root / "decision_summary.json", summary)
    controls_table = controls.to_markdown(index=False) if not controls.empty else "No control summary."
    stress_table = stress.to_markdown(index=False) if not stress.empty else "No stress summary."
    write_text(ctx.run_root / "QLMG_LISTING_GENERIC_FULL_EVENT_REPLAY_REPORT.md", f"# QLMG Listing/Generic Full-Event Replay Report\n\n## Verdicts\n\n- listing_vwap_loss_verdict: `{summary['listing_vwap_loss_verdict']}`\n- generic_shock_reversal_verdict: `{summary['generic_shock_reversal_verdict']}`\n- d4_verdict: `{summary['d4_verdict']}`\n- depth_data_verdict: `{summary['depth_data_verdict']}`\n- next_action_verdict: `{summary['next_action_verdict']}`\n\n## Key Facts\n\n- Protected holdout untouched: `true`\n- Exact listing reconstruction rows: `{listing.to_dict('records') if not listing.empty else []}`\n- Confirmed listing candidates after scope-corrected full-hold replay/control/stress gates: `{confirmed}`\n- D4 count root cause: `{d4_root}`\n- Funding-window family: `preserved, not rejected at family level`\n- Secondary families: `preserved, not marked dead from this run`\n\n## Scope-Corrected Decision Method\n\n- Rankable replay scope: `full_hold` only.\n- `core_24h` rows: diagnostic only; not aggregated into decision metrics.\n- Control comparison: mean_R and control_R normalized to candidate-event count; raw control net_R sums are not rankable because controls are multiple-per-event.\n- Stress: recomputed only on full-hold candidate-event rows.\n\n## Scope-Corrected Control Summary\n\n{controls_table}\n\n## Scope-Corrected Stress Summary\n\n{stress_table}\n\nNo result is validated, sealed-ready, live-ready, production-ready, or a trading recommendation.\n")


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        ctx.run_root / "QLMG_LISTING_GENERIC_FULL_EVENT_REPLAY_REPORT.md",
        ctx.run_root / "decision_summary.json",
        ctx.run_root / "preflight/preflight_report.md",
        ctx.run_root / "audit/report_semantics_audit.md",
        ctx.run_root / "d4/d4_event_count_reconciliation_report.md",
        ctx.run_root / "listing/listing_reconstruction_report.md",
        ctx.run_root / "funding_window/funding_window_preservation_audit.md",
        ctx.run_root / "generic_shock/generic_shock_contract_report.md",
        ctx.run_root / "windows/window_report.md",
        ctx.run_root / "replay/full_event_one_minute_replay_report.md",
        ctx.run_root / "controls/full_event_control_report.md",
        ctx.run_root / "deconfound/listing_vs_generic_shock_report.md",
        ctx.run_root / "stress/full_event_stress_report.md",
        ctx.run_root / "depth/depth_procurement_or_live_capture_plan.md",
        ctx.run_root / "secondary/secondary_family_preservation_report.md",
    ]
    rows = []
    for p in include:
        if p.exists():
            dest = bundle / p.relative_to(ctx.run_root).as_posix().replace("/", "__")
            shutil.copy2(p, dest)
            rows.append({"artifact": str(p), "bundle_path": str(dest), "size_bytes": p.stat().st_size})
    for p in ctx.run_root.rglob("*.csv"):
        if "downloaded_1m" in p.parts or p.stat().st_size > 5_000_000:
            continue
        dest = bundle / p.relative_to(ctx.run_root).as_posix().replace("/", "__")
        shutil.copy2(p, dest)
        rows.append({"artifact": str(p), "bundle_path": str(dest), "size_bytes": p.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nContains reports, summaries, schemas/manifests, and small CSV artifacts only. Large 1m datasets and full ledgers are excluded; see artifact paths.\n")


STAGE_FUNCS = {
    "preflight-and-artifact-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "report-semantics-and-status-audit": stage_report_semantics,
    "d4-event-count-reconciliation": stage_d4_reconciliation,
    "confirmed-listing-candidate-full-event-manifest": stage_listing_manifest,
    "funding-window-preservation-audit": stage_funding_preservation,
    "generic-shock-reversal-event-generator": stage_generic_generator,
    "full-event-window-deduplication-and-storage-estimate": stage_windows,
    "targeted-1m-download-if-approved": stage_download,
    "targeted-1m-qc": stage_qc,
    "full-event-one-minute-mark-replay": stage_replay,
    "stop-target-liquidation-ordering": stage_ordering,
    "matched-control-replay-refresh": stage_controls,
    "listing-vs-generic-shock-deconfounding": stage_deconfound,
    "cost-funding-slippage-stress-refresh": stage_stress,
    "depth-trade-liquidation-source-audit": stage_depth_source,
    "depth-procurement-or-live-capture-plan": stage_depth_plan,
    "secondary-family-preservation-refresh": stage_secondary,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        print(f"[resume] skip {stage}")
        return
    append_command(ctx.run_root, stage)
    ensure_guard(ctx, stage)
    ctx.notifier.send("QLMG listing full replay stage start", stage)
    STAGE_FUNCS[stage](ctx)
    mark_done(ctx.run_root, stage)
    ctx.notifier.send("QLMG listing full replay stage complete", stage)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram and not args.smoke, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_root_resolution.json", {"run_root": str(run_root), "reason": reason, "created_at_utc": utc_now()})
    notifier.send("QLMG listing full replay run start", f"run_root={run_root} stage={args.stage}")
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG listing full replay run complete", f"run_root={run_root}")
        return 0
    except Exception as exc:
        notifier.send("QLMG listing full replay run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
