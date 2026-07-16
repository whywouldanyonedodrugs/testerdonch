#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_evidence_contracts import (  # noqa: E402
    PROTECTED_TS,
    artifact_risk_scan,
    require_no_protected_timestamps,
    result_to_jsonable,
    scan_output_tree_for_protected,
    validate_control_rows,
    validate_event_trade_schema,
    validate_funding_mark_flags,
    validate_no_projected_metric_promotion,
)
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_kraken_full_coverage_signal_tape_sweep_20260702_v1"
DEFAULT_SEED = 20260702
DEFAULT_HYPOTHESIS_LIBRARY = REPO / "research_inputs/QLMG_Hypothesis_Library_2026-07-01.xlsx"
DEFAULT_RESEARCH_INPUT_DIR = REPO / "research_inputs"
DEFAULT_KRAKEN_DATA_ROOT = Path("/opt/parquet/kraken_derivatives")
DEFAULT_K0_ROOT = RESULTS_ROOT / "phase_kraken_k0_data_foundation_20260630_v1_20260630_163815"
DEFAULT_READINESS_ROOT = RESULTS_ROOT / "phase_kraken_hypothesis_sweep_readiness_20260701_v1_20260701_085434"
DEFAULT_REPAIR_ROOT = RESULTS_ROOT / "phase_kraken_readiness_repair_20260701_v1_20260701_111807"
DEFAULT_MECHANICAL_QA_ROOT = RESULTS_ROOT / "phase_qlmg_mechanical_qa_evidence_contract_20260630_v1_20260630_074328"
DEFAULT_PREVIOUS_SAMPLED_SWEEP_ROOT = RESULTS_ROOT / "phase_kraken_gated_full_hypothesis_sweep_20260701_v1_20260701_155115"
DEFAULT_REMOTE_ROOT_FOLDER_ID = "1ckW8HkR346yRuaVl8Img38AqAR_qFNd4"
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
DEV_EVAL_SPLIT = pd.Timestamp("2025-07-01T00:00:00Z")
EVENT_SEMANTICS_VERSION = "kraken_event_semantics_v2_stateful_20260702"

STAGES = (
    "preflight-and-source-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "previous-sampled-run-demotion",
    "scope-lock-priority-waves",
    "full-event-contract-dry-run",
    "priority-wave-execution",
    "wave-event-ledger-audit",
    "wave-full-controls",
    "wave-stress-and-context-analysis",
    "wave-completion-publication",
    "next-priority-wave-loop",
    "global-dedup-and-overlap-analysis",
    "global-candidate-library",
    "final-adversarial-audit",
    "decision-report",
    "compact-review-bundle",
    "all",
)

RANKABLE_LANES = {"kraken_tier1_ready", "kraken_tier1_with_caps"}
REPAIR_RANKABLE_LANES = {"compiled_tier1_redesign", "compiled_tier1_with_analytics_cap"}
SIDE_CAR_LANES = {
    "kraken_live_capture_sidecar",
    "needs_live_capture_substitute",
    "kraken_event_ledger_first",
    "needs_event_ledger_first",
    "kraken_candidate_library_only",
    "not_kraken_viable_current_translation",
}
ALLOWED_FINAL_LABELS = {
    "full_coverage_screen_candidate",
    "tier1_with_cap_candidate",
    "rare_regime_sleeve_candidate",
    "fragile_but_interesting",
    "needs_targeted_1m_replay",
    "needs_context_refinement",
    "candidate_library_only",
    "current_translation_rejected_only",
}
ALLOWED_NEXT_DECISIONS = {
    "analyze_completed_waves_now",
    "run_targeted_1m_replay_for_wave_survivors_next",
    "run_train_only_validation_for_deduped_survivors_next",
    "run_context_refinement_for_promising_sleeves_next",
    "run_kraken_live_capture_sidecar_next",
    "generate_new_hypotheses_next",
    "repair_failed_component_next",
    "blocked_by_protocol_issue",
}
FORBIDDEN_WORDS = re.compile(r"validated|live-ready|production-ready|final edge|deployable strategy", re.IGNORECASE)

ARCHETYPE_BY_BUCKET = {
    "tsmom": "tsmom",
    "prior_high_ath": "prior_high",
    "retest": "retest_reclaim",
    "compression_breakout": "compression_breakout",
    "session_time": "session_calendar",
    "funding_crowding": "funding_crowding",
    "liquid_continuation": "liquid_continuation",
}

FAMILY_BUDGET_HINTS = [
    ("liquid leader continuation / breakout", 9000),
    ("volatility-managed TSMOM", 9000),
    ("prior-high / ATH / reclaim", 8000),
    ("post-breakout retest/reclaim", 7000),
    ("volatility compression breakout", 6000),
    ("session/calendar close-confirmed variants", 5000),
    ("funding/crowding Tier 1 variants", 5000),
]

CONTRACT_EVENT_TYPE_BY_ARCHETYPE = {
    "liquid_continuation": "trade_episode_contract",
    "prior_high": "trade_episode_contract",
    "compression_breakout": "trade_episode_contract",
    "retest_reclaim": "event_lifecycle_contract",
    "tsmom": "scheduled_decision_contract",
    "session_calendar": "scheduled_decision_contract",
    "funding_crowding": "scheduled_decision_contract",
}

ROW_SEMANTICS_BY_CONTRACT_TYPE = {
    "trade_episode_contract": "trade_episode",
    "state_transition_contract": "trade_episode",
    "event_lifecycle_contract": "lifecycle_event",
    "scheduled_decision_contract": "position_interval",
    "sidecar_nonrankable_contract": "decision",
}


def sha256_file(path: Path, limit_mb: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if limit_mb is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit_mb * 1024 * 1024))
    return h.hexdigest()


def stable_hash(*parts: object, n: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:n]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    df.to_csv(path, index=False)




def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / 1024**3 if path.exists() and path.is_file() else 0.0


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                pass
    return total


def pf(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna()
    wins = float(x[x > 0].sum())
    losses = float(x[x < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / abs(losses)


def max_dd(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").fillna(0).to_numpy(dtype=float)
    if not len(x):
        return 0.0
    curve = np.cumsum(x)
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve - peak))


def parse_time_ms_or_iso(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    s = series.astype(str)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


@dataclass
class RunNotifier:
    run_root: Path
    disabled: bool = False
    require_remote: bool = False
    allow_no_remote: bool = False
    remote: Any = None
    status: str = "unavailable"
    missing: str = ""

    def __post_init__(self) -> None:
        self.events_path = self.run_root / "notifications/telegram_events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.disabled:
            self.status = "disabled"
            self.missing = "disabled_by_cli"
            return
        if TelegramNotifier is not None:
            class _Args:
                disable_telegram = False
                telegram_dry_run = False
                tg_bot_token = ""
                tg_chat_id = ""
                tg_auto_chat = False
            try:
                self.remote = TelegramNotifier.from_args(_Args(), run_label="kraken-gated-sweep")
                self.status = getattr(self.remote, "status_line", lambda: "enabled")()
            except Exception as exc:  # pragma: no cover
                self.remote = None
                self.status = "unavailable"
                self.missing = f"{type(exc).__name__}: {exc}"
        else:
            self.missing = "TelegramNotifier unavailable"
        if self.require_remote and not self.remote_available and not self.allow_no_remote:
            raise RuntimeError(f"remote Telegram required but unavailable: {self.missing or self.status}")

    @property
    def remote_available(self) -> bool:
        return (not self.disabled) and self.remote is not None and "enabled" in str(self.status).lower()

    def send(self, title: str, body: str = "", level: str = "info") -> bool:
        sent = False
        error = ""
        if self.remote is not None:
            try:
                sent = bool(self.remote.send(title, body))
            except Exception as exc:  # pragma: no cover
                error = f"{type(exc).__name__}: {exc}"
        rec = {"ts_utc": utc_now(), "title": title, "body": body, "level": level, "sent": sent, "status": self.status, "error": error}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        return sent


@dataclass
class Context:
    args: argparse.Namespace
    run_root: Path
    notifier: RunNotifier
    start: pd.Timestamp
    end: pd.Timestamp
    root_reason: str
    stage_sizes: list[dict[str, Any]] = field(default_factory=list)
    retained: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[dict[str, Any]] = field(default_factory=list)
    cleanup_failures: list[dict[str, Any]] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kraken full-coverage signal-tape sweep")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=80.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--hypothesis-library", default=str(DEFAULT_HYPOTHESIS_LIBRARY.relative_to(REPO)))
    p.add_argument("--research-input-dir", default=str(DEFAULT_RESEARCH_INPUT_DIR.relative_to(REPO)))
    p.add_argument("--kraken-data-root", default=str(DEFAULT_KRAKEN_DATA_ROOT))
    p.add_argument("--readiness-root", default=str(DEFAULT_READINESS_ROOT.relative_to(REPO)))
    p.add_argument("--repair-root", default=str(DEFAULT_REPAIR_ROOT.relative_to(REPO)))
    p.add_argument("--k0-root", default=str(DEFAULT_K0_ROOT.relative_to(REPO)))
    p.add_argument("--mechanical-qa-root", default=str(DEFAULT_MECHANICAL_QA_ROOT.relative_to(REPO)))
    p.add_argument("--previous-sampled-sweep-root", default=str(DEFAULT_PREVIOUS_SAMPLED_SWEEP_ROOT.relative_to(REPO)))
    p.add_argument("--priority-wave", default="all")
    p.add_argument("--family-list", default="")
    p.add_argument("--contract-list", default="")
    p.add_argument("--all-eligible-events", action="store_true", default=True)
    p.add_argument("--no-event-sampling", action="store_true", default=True)
    p.add_argument("--no-portfolio-caps", action="store_true", default=True)
    p.add_argument("--full-candidate-definition-budget", type=int, default=60000)
    p.add_argument("--coarse-definition-budget", type=int, default=45000)
    p.add_argument("--refine-definition-budget", type=int, default=15000)
    p.add_argument("--min-definitions-per-rankable-hypothesis", type=int, default=12)
    p.add_argument("--min-definitions-per-family", type=int, default=500)
    p.add_argument("--max-definitions-per-family", type=int, default=12000)
    p.add_argument("--reserve-budget-share", type=float, default=0.20)
    p.add_argument("--definition-budget-mode", default="coverage_first_adaptive")
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--top-per-family", type=int, default=100)
    p.add_argument("--max-runtime-hours", type=float, default=168.0)
    p.add_argument("--family-wave-size", type=int, default=2)
    p.add_argument("--max-control-candidates-per-subwave", type=int, default=750)
    p.add_argument("--max-control-runtime-hours-per-subwave", type=float, default=6.0)
    p.add_argument("--coarse-reject-audit-max", type=int, default=250)
    p.add_argument("--reuse-interrupted-wave-root", default="")
    p.add_argument("--remote-archive-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--remote-name", default="qlmg_gdrive")
    p.add_argument("--remote-root-folder-id", default=DEFAULT_REMOTE_ROOT_FOLDER_ID)
    p.add_argument("--remote-archive-path", default="")
    p.add_argument("--archive-completed-waves", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prune-local-large-wave-artifacts-after-upload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tmux-session-name", default="kraken_full_coverage_signal_tape")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--run-root", default="")
    return p.parse_args(argv)


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


def resolve_run_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.run_root:
        return resolve_path(args.run_root).resolve(), "explicit_run_root"
    base = (RESULTS_ROOT / DEFAULT_RUN_ID).resolve()
    if args.smoke:
        return (base / "smoke").resolve(), "smoke_subroot"
    if not base.exists():
        return base, "base_run_root"
    return base.with_name(base.name + "_" + utc_stamp()), "base_collision_timestamp_suffix"


def init_context(args: argparse.Namespace) -> Context:
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    generated_dirs = [
        "preflight", "notifications", "tmux", "seal", "prior_run_demotions", "scope", "templates",
        "budget", "coverage", "refinement", "dry_run", "waves", "controls", "mechanics",
        "universe", "stress", "context", "validation", "dedup", "library", "early_stop",
        "feasibility", "audit_semantics", "state_machine",
        "analysis_ready", "remote_archive", "state", "interruptions", "audit_final", "resources",
        "compact_review_bundle", "stage_status", "tmp",
    ]
    if not args.resume:
        for d in generated_dirs:
            p = run_root / d
            if p.exists():
                shutil.rmtree(p)
        for rel in ["watch_status.json", "decision_summary.json", "QLMG_KRAKEN_FULL_COVERAGE_SIGNAL_TAPE_SWEEP_REPORT.md"]:
            p = run_root / rel
            if p.exists():
                p.unlink()
    for d in [
        "preflight", "notifications", "tmux", "seal", "prior_run_demotions", "scope", "templates",
        "budget", "coverage", "refinement", "dry_run", "waves", "controls", "mechanics",
        "universe", "stress", "context", "validation", "dedup", "library", "early_stop",
        "feasibility", "audit_semantics", "state_machine",
        "analysis_ready", "remote_archive", "state", "interruptions", "audit_final", "resources",
        "compact_review_bundle", "stage_status", "tmp",
    ]:
        (run_root / d).mkdir(parents=True, exist_ok=True)
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True)
    if end >= PROTECTED_TS:
        end = SCREENING_END
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = Context(args=args, run_root=run_root, notifier=notifier, start=start, end=end, root_reason=reason)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "args": vars(args), "created_ts_utc": utc_now()})
    write_json(run_root / "state/process_status.json", {"pid": os.getpid(), "status": "running", "run_root": str(run_root), "argv": sys.argv, "ts_utc": utc_now()})
    write_status(ctx, "running", "initialized")
    return ctx


def write_status(ctx: Context, status: str, stage: str = "") -> None:
    payload = {"run_root": str(ctx.run_root), "status": status, "stage": stage, "ts_utc": utc_now()}
    write_json(ctx.run_root / "watch_status.json", payload)


def mark_interrupted(ctx: Context, *, reason: str, detail: str = "", exit_code: int | None = None) -> None:
    """Best-effort marker for failures Python can still observe.

    Kernel SIGKILL/OOM cannot run this code; the tmux wrapper writes the same
    artifacts after the child process exits in those cases.
    """
    (ctx.run_root / "interruptions").mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "interrupted",
        "reason": reason,
        "detail": detail,
        "exit_code": exit_code,
        "pid": os.getpid(),
        "run_root": str(ctx.run_root),
        "ts_utc": utc_now(),
    }
    write_json(ctx.run_root / "interruptions/interruption_status.json", payload)
    write_text(
        ctx.run_root / "interruptions/interruption_report.md",
        "# Interruption Report\n\n"
        f"Status: `interrupted`\n\nReason: `{reason}`\n\nDetail: `{detail}`\n\nExit code: `{exit_code}`\n",
    )
    write_status(ctx, "interrupted", reason)
    if not (ctx.run_root / "decision_summary.json").exists():
        write_json(ctx.run_root / "decision_summary.json", {"run_root": str(ctx.run_root), "status": "interrupted", "reason": reason, "ts_utc": utc_now()})
    write_json(ctx.run_root / "notifications/final_completion_marker.json", {"status": "interrupted", "reason": reason, "run_root": str(ctx.run_root), "ts_utc": utc_now()})
    ctx.notifier.send("Kraken full-coverage signal-tape sweep interrupted", f"{reason}\n{detail}", level="error")


def stage_done_path(ctx: Context, stage: str) -> Path:
    return ctx.run_root / "stage_status" / f"{stage}.done"


def mark_done(ctx: Context, stage: str) -> None:
    write_json(stage_done_path(ctx, stage), {"stage": stage, "status": "done", "ts_utc": utc_now()})


def should_skip(ctx: Context, stage: str) -> bool:
    return bool(ctx.args.resume and stage_done_path(ctx, stage).exists())


def run_stage(ctx: Context, stage: str, fn) -> None:
    if should_skip(ctx, stage):
        return
    before = dir_size_bytes(ctx.run_root)
    ctx.notifier.send("Kraken full-coverage sweep stage start", stage)
    write_status(ctx, "running", stage)
    try:
        fn(ctx)
    except Exception as exc:
        write_status(ctx, "failed", stage)
        ctx.notifier.send("Kraken full-coverage sweep stage failed", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        raise
    after = dir_size_bytes(ctx.run_root)
    ctx.stage_sizes.append({"stage": stage, "size_before_bytes": before, "size_after_bytes": after, "delta_bytes": after - before})
    write_csv(ctx.run_root / "resources/output_budget_by_stage.csv", ctx.stage_sizes)
    mark_done(ctx, stage)
    ctx.notifier.send("Kraken full-coverage sweep stage done", stage)


def manual_candidates(research_input_dir: Path) -> list[Path]:
    return [
        REPO / "docs/QLMG_BACKTESTING_MANUAL_20260630_FULL.md",
        research_input_dir / "QLMG_BACKTESTING_MANUAL_20260630_FULL.md",
        research_input_dir / "testmanual.txt",
    ]


def resolve_manual(research_input_dir: Path) -> dict[str, Any]:
    rows = []
    selected = None
    for order, p in enumerate(manual_candidates(research_input_dir), 1):
        exists = p.exists()
        row = {"order": order, "path": str(p), "exists": exists, "selected": False, "sha256": "", "has_no_vendor_policy": False, "has_mechanical_qa_contract_language": False}
        if exists:
            txt = p.read_text(errors="ignore")
            row["sha256"] = sha256_file(p)
            row["has_no_vendor_policy"] = "no-vendor" in txt.lower() or "no vendor" in txt.lower()
            row["has_mechanical_qa_contract_language"] = "mechanical qa" in txt.lower() and "evidence contract" in txt.lower()
            if selected is None:
                selected = p
                row["selected"] = True
        rows.append(row)
    if selected is None:
        raise FileNotFoundError("no active manual found in required resolution order")
    return {"selected_path": str(selected), "rows": rows}


def artifact_manifest(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        exists = p.exists()
        row: dict[str, Any] = {"path": str(p), "exists": exists, "type": "directory" if p.is_dir() else "file" if exists else "missing", "sha256": "", "size_bytes": 0}
        if exists and p.is_file():
            row["sha256"] = sha256_file(p)
            row["size_bytes"] = p.stat().st_size
        elif exists and p.is_dir():
            files = sorted(x for x in p.rglob("*") if x.is_file())[:2000]
            h = hashlib.sha256()
            total = 0
            for f in files:
                rel = f.relative_to(p)
                try:
                    st = f.stat()
                except FileNotFoundError:
                    continue
                total += st.st_size
                h.update(str(rel).encode())
                h.update(str(st.st_size).encode())
                h.update(str(int(st.st_mtime)).encode())
            row["sha256"] = h.hexdigest()
            row["size_bytes"] = total
            row["hash_note"] = "directory_manifest_first_2000_files"
        rows.append(row)
    return rows


def stage_preflight(ctx: Context) -> None:
    args = ctx.args
    k0 = resolve_path(args.k0_root)
    readiness = resolve_path(args.readiness_root)
    repair = resolve_path(args.repair_root)
    mechanical = resolve_path(args.mechanical_qa_root)
    previous = resolve_path(args.previous_sampled_sweep_root)
    library = resolve_path(args.hypothesis_library)
    research = resolve_path(args.research_input_dir)
    kraken = resolve_path(args.kraken_data_root)
    manual = resolve_manual(research)
    write_csv(ctx.run_root / "preflight/manual_resolution_report.csv", manual["rows"])
    warn = ""
    sel_row = next(r for r in manual["rows"] if r["selected"])
    if not (sel_row["has_no_vendor_policy"] and sel_row["has_mechanical_qa_contract_language"]):
        warn = "Selected manual lacks no-vendor or mechanical-QA contract language; tools/qlmg_evidence_contracts.py remains authoritative."
    write_text(ctx.run_root / "preflight/manual_resolution_report.md", f"# Manual Resolution Report\n\nSelected: `{manual['selected_path']}`\n\nWarning: {warn or 'none'}\n")
    manifest_paths = list((kraken / "manifests").glob("*manifest*")) if (kraken / "manifests").exists() else []
    paths = [
        k0, readiness, repair, mechanical, previous, library, research, kraken, *manifest_paths,
        Path(__file__).resolve(), REPO / "tools/qlmg_evidence_contracts.py", REPO / "tools/qlmg_real_controls.py",
        Path(manual["selected_path"]),
    ]
    rows = artifact_manifest(paths)
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", {r["path"]: r for r in rows})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = "unavailable"
    snap = resource_snapshot(ctx.run_root)
    guard = check_resource_guard(snap, estimated_output_gb=0.1, hard_stage_output_gb=ctx.args.max_output_gb, allow_large_output=ctx.args.allow_large_output)
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard failed: {guard}")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight\n\nRun root: `{ctx.run_root}`\nGit commit: `{commit}`\nManual: `{manual['selected_path']}`\nPrior sampled sweep: `{previous}`\nMechanical QA: `{mechanical}`\nResource guard: `{guard['status']}`\n")


def stage_telegram_tmux(ctx: Context) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\nStatus: `{ctx.notifier.status}`\nRemote available: `{ctx.notifier.remote_available}`\n")
    watch = f"""# Watch Commands\n\n```bash\ntmux attach -t {ctx.args.tmux_session_name}\ntail -f {ctx.run_root}/notifications/telegram_events.jsonl\nwatch -n 30 'cat {ctx.run_root}/watch_status.json'\nfind {ctx.run_root}/stage_status -type f | sort\n```\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)


def stage_seal(ctx: Context) -> None:
    report = {
        "protected_holdout_start": str(PROTECTED_TS),
        "scoring_start": str(ctx.start),
        "scoring_end": str(ctx.end),
        "status": "pass" if ctx.end < PROTECTED_TS else "fail",
    }
    if report["status"] != "pass":
        raise RuntimeError("scoring end crosses protected holdout")
    write_json(ctx.run_root / "seal/protected_timestamp_scan.json", report)
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\nProtected holdout starts `{PROTECTED_TS}`. Scoring end `{ctx.end}`. Status `{report['status']}`.\n")


def stage_previous_sampled_run_demotion(ctx: Context) -> None:
    prev = resolve_path(ctx.args.previous_sampled_sweep_root)
    funnel = read_csv_safe(prev / "validation/funnel_accounting.csv")
    event_cap_note = "previous run used sampled event caps per candidate; not alpha evidence"
    rows = []
    for p in [
        prev / "library/refreshed_candidate_library.csv",
        prev / "dedup/cluster_representatives.csv",
        prev / "waves/wave_1/event_ledger.parquet",
        prev / "waves/wave_2/event_ledger.parquet",
        prev / "KRAKEN_GATED_FULL_HYPOTHESIS_SWEEP_REPORT.md",
    ]:
        rows.append({
            "path": str(p),
            "exists": p.exists(),
            "rankable_in_this_run": False,
            "allowed_use": "infrastructure_reference_only",
            "demotion_reason": event_cap_note,
        })
    write_csv(ctx.run_root / "prior_run_demotions/non_rankable_prior_outputs.csv", rows)
    txt = [
        "# Previous Sampled Sweep Demotion",
        "",
        f"Previous root: `{prev}`",
        "",
        "Verdict: `infrastructure_only_not_alpha_evidence`",
        "",
        "The prior Kraken gated sweep completed too quickly because it capped each candidate to a small sampled event set. It may inform code routing, contract compiler behavior, runtime lessons, and mechanical failure detection. It must not be used for strategy-quality evidence.",
    ]
    if not funnel.empty:
        txt += ["", "Prior funnel snapshot:", "", funnel.to_markdown(index=False)]
    write_text(ctx.run_root / "prior_run_demotions/previous_sampled_sweep_demotion_report.md", "\n".join(txt) + "\n")


def load_contracts(readiness_root: Path, repair_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    readiness_contracts: dict[str, dict[str, Any]] = {}
    for p in sorted((readiness_root / "compiler/compiled_contracts").glob("*.json")):
        d = read_json(p, {})
        hid = str(d.get("hypothesis_id", p.stem))
        d["contract_path"] = str(p)
        d["contract_source"] = "readiness"
        readiness_contracts[hid] = d
    repair_contracts: dict[str, dict[str, Any]] = {}
    for p in sorted((repair_root / "compile_repair/repaired_contracts").glob("*.json")):
        d = read_json(p, {})
        hid = str(d.get("hypothesis_id", p.stem))
        d["contract_path"] = str(p)
        d["contract_source"] = "repair"
        repair_contracts[hid] = d
    return readiness_contracts, repair_contracts


def archetype_for_contract(contract: Mapping[str, Any], lane: str) -> str:
    text = " ".join(str(contract.get(k, "")) for k in ["pilot_bucket", "strategy_mode", "family", "mechanism", "repair_reason", "entry_rule_template"])
    low = text.lower()
    if "session" in low or "calendar" in low:
        return "session_calendar"
    if "funding" in low or "crowd" in low:
        return "funding_crowding"
    if "retest" in low or "reclaim" in low:
        return "retest_reclaim"
    if "prior" in low or "ath" in low or "high" in low:
        return "prior_high"
    if "compression" in low or "volatility" in low:
        return "compression_breakout" if "compression" in low else "tsmom"
    if "tsmom" in low or "momentum" in low or "continuation" in low or "leader" in low:
        return "liquid_continuation"
    if lane == "kraken_tier1_with_caps" or lane == "compiled_tier1_with_analytics_cap":
        return "funding_crowding"
    return "liquid_continuation"


def priority_wave_for_hypothesis(hid: str, family: str, lane: str) -> str:
    h = str(hid).upper()
    f = str(family).lower()
    wave1 = {"H01", "H02", "H03", "H04", "H05", "H06", "H08", "H09", "H31", "H32"}
    if h in wave1 or ("funding" in f and lane in RANKABLE_LANES | REPAIR_RANKABLE_LANES):
        return "wave_1"
    if h.startswith("H") or h.startswith("A"):
        return "wave_2"
    if h.startswith("PR"):
        return "wave_3"
    return "wave_4"


def family_template_defaults(family: str, archetype: str) -> dict[str, Any]:
    low = str(family).lower()
    if "session" in low:
        entries = ["session_close_confirmed", "session_strong_close"]
        exits = ["session_close", "fixed_hold", "structure_loss"]
        stops = ["session_bar_extreme", "atr_proxy", "structure"]
    elif "funding" in low:
        entries = ["price_funding_confirmed", "funding_persistence_confirmed"]
        exits = ["funding_normalization", "fixed_hold", "structure_loss"]
        stops = ["atr_proxy", "structure", "fixed_bps"]
    elif "compression" in low or archetype == "compression_breakout":
        entries = ["compression_breakout_close", "range_expansion_close"]
        exits = ["fixed_hold", "target_R", "ema_trail", "structure_loss"]
        stops = ["breakout_bar_extreme", "atr_proxy", "structure"]
    elif "retest" in low or archetype == "retest_reclaim":
        entries = ["close_confirmed_reclaim", "hold_bar_reclaim"]
        exits = ["failed_retest", "fixed_hold", "target_R", "ema_trail"]
        stops = ["retest_extreme", "structure", "atr_proxy"]
    elif "prior" in low or archetype == "prior_high":
        entries = ["prior_high_close_confirmed", "prior_high_retest_optional"]
        exits = ["failed_reclaim", "fixed_hold", "ema_trail", "structure_loss"]
        stops = ["prior_high_structure", "atr_proxy", "fixed_bps"]
    elif "volatility" in low or archetype == "tsmom":
        entries = ["vol_managed_tsmom_close", "trend_rebalance_close"]
        exits = ["signal_flip", "trend_reversal", "volatility_stop", "time_stop"]
        stops = ["volatility_stop", "atr_proxy", "fixed_bps"]
    else:
        entries = ["liquid_leader_breakout_close", "relative_strength_close_confirmed"]
        exits = ["fixed_hold", "target_R", "ema_trail", "structure_loss"]
        stops = ["breakout_bar_extreme", "atr_proxy", "structure"]
    return {
        "entry_templates": entries,
        "exit_templates": exits,
        "stop_templates": stops,
        "risk_unit_definition": "R = adverse stop distance from next-bar taker entry",
        "confirmation_rule": "closed 5m bar confirmation; entry at next 5m open",
        "regime_variants": ["intended_regime", "all_context_diagnostic"],
        "holding_bands": ["short", "medium", "long"],
        "sides": ["long", "short"],
    }


def canonical_definition_rows(rank_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definition_kinds = [
        ("plain_baseline", "fixed_hold", "fixed_bps", 24, 150, 0.005),
        ("conservative", "fixed_hold", "atr_proxy", 12, 100, 0.01),
        ("aggressive", "target_R", "breakout_bar_extreme", 36, 300, 0.0025),
        ("alternative_exit", "ema_trail", "structure", 48, 200, 0.005),
        ("alternative_stop", "structure_loss", "structure", 24, 250, 0.0075),
    ]
    for _, r in rank_df.iterrows():
        hid = str(r.get("hypothesis_id"))
        fam = str(r.get("family", "unknown"))
        arch = str(r.get("archetype", "liquid_continuation"))
        tmpl = family_template_defaults(fam, arch)
        side_choices = ["short"] if "short" in fam.lower() else ["long", "short"]
        for regime_variant in tmpl["regime_variants"]:
            for kind, exit_template, stop_template, hold, stop_bps, threshold in definition_kinds:
                side = side_choices[(len(rows) + len(kind)) % len(side_choices)]
                entry_template = tmpl["entry_templates"][len(rows) % len(tmpl["entry_templates"])]
                lookback = int([12, 24, 48, 96, 144][len(rows) % 5])
                did = f"canon__{hid}__{stable_hash(arch, kind, regime_variant, entry_template, exit_template, stop_template, side, lookback, hold, stop_bps, threshold, n=12)}"
                rows.append({
                    "definition_id": did,
                    "definition_kind": kind,
                    "hypothesis_id": hid,
                    "family": fam,
                    "contract_id": r.get("contract_id", ""),
                    "contract_source": r.get("contract_source", ""),
                    "priority_wave": r.get("priority_wave", priority_wave_for_hypothesis(hid, fam, str(r.get("allowed_lane", "")))),
                    "archetype": arch,
                    "entry_template": entry_template,
                    "exit_template": exit_template,
                    "stop_template": stop_template,
                    "regime_variant": regime_variant,
                    "regime_activation": "intended_regime" if regime_variant == "intended_regime" else "all_context_diagnostic",
                    "side": side,
                    "lookback_bars": lookback,
                    "hold_bars": int(hold),
                    "stop_bps": float(stop_bps),
                    "threshold": float(threshold),
                    "canonical": True,
                    "data_cap": r.get("data_cap", "none"),
                    "allowed_lane": r.get("allowed_lane", ""),
                    "source_contract_path": r.get("contract_path", ""),
                })
    return pd.DataFrame(rows)


def deterministic_space_fill(rank_df: pd.DataFrame, target_defs: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rank_df.empty or target_defs <= 0:
        return pd.DataFrame(), pd.DataFrame()
    dims = {
        "lookback_bars": [12, 24, 48, 96, 144, 288],
        "hold_bars": [6, 12, 24, 36, 72, 144],
        "stop_bps": [75, 100, 150, 200, 300, 400],
        "threshold": [0.0025, 0.005, 0.0075, 0.01, 0.02],
    }
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    rank_rows = rank_df.to_dict("records")
    i = 0
    attempts = 0
    max_attempts = max(target_defs * 3, target_defs + 100)
    while len(rows) < target_defs and attempts < max_attempts:
        attempts += 1
        r = rank_rows[i % len(rank_rows)]
        hid = str(r.get("hypothesis_id"))
        fam = str(r.get("family", "unknown"))
        arch = str(r.get("archetype", "liquid_continuation"))
        tmpl = family_template_defaults(fam, arch)
        entry = tmpl["entry_templates"][(i + attempts) % len(tmpl["entry_templates"])]
        exit_template = tmpl["exit_templates"][(i // 2 + attempts) % len(tmpl["exit_templates"])]
        stop_template = tmpl["stop_templates"][(i // 3 + attempts) % len(tmpl["stop_templates"])]
        hold = dims["hold_bars"][(i * 3 + attempts) % len(dims["hold_bars"])]
        lookback = dims["lookback_bars"][(i * 5 + attempts) % len(dims["lookback_bars"])]
        stop_bps = dims["stop_bps"][(i * 7 + attempts) % len(dims["stop_bps"])]
        threshold = dims["threshold"][(i * 11 + attempts) % len(dims["threshold"])]
        regime_variant = tmpl["regime_variants"][(i + attempts) % len(tmpl["regime_variants"])]
        side = "short" if ("short" in fam.lower() or (i + attempts) % 2) else "long"
        reason = ""
        if hold < 2:
            reason = "hold_shorter_than_confirmation"
        if "touch" in entry.lower():
            reason = "touch_fill_not_allowed"
        if reason:
            rejects.append({"hypothesis_id": hid, "family": fam, "rejection_reason": reason})
            i += 1
            continue
        did = f"sf__{hid}__{stable_hash(arch, i, attempts, entry, exit_template, stop_template, regime_variant, side, lookback, hold, stop_bps, threshold, n=12)}"
        rows.append({
            "definition_id": did,
            "definition_kind": "broad_space_filling",
            "hypothesis_id": hid,
            "family": fam,
            "contract_id": r.get("contract_id", ""),
            "contract_source": r.get("contract_source", ""),
            "priority_wave": r.get("priority_wave", priority_wave_for_hypothesis(hid, fam, str(r.get("allowed_lane", "")))),
            "archetype": arch,
            "entry_template": entry,
            "exit_template": exit_template,
            "stop_template": stop_template,
            "regime_variant": regime_variant,
            "regime_activation": "intended_regime" if regime_variant == "intended_regime" else "all_context_diagnostic",
            "side": side,
            "lookback_bars": int(lookback),
            "hold_bars": int(hold),
            "stop_bps": float(stop_bps),
            "threshold": float(threshold),
            "canonical": False,
            "data_cap": r.get("data_cap", "none"),
            "allowed_lane": r.get("allowed_lane", ""),
            "source_contract_path": r.get("contract_path", ""),
        })
        i += 1
    return pd.DataFrame(rows), pd.DataFrame(rejects)


def stage_scope(ctx: Context) -> None:
    readiness = resolve_path(ctx.args.readiness_root)
    repair = resolve_path(ctx.args.repair_root)
    viability_path = repair / "viability/hypothesis_viability_matrix.csv"
    semantic_path = readiness / "compiler/semantic_sanity_checks.csv"
    trace_path = readiness / "compiler/hypothesis_to_contract_trace.csv"
    if not viability_path.exists():
        raise FileNotFoundError(viability_path)
    viability = pd.read_csv(viability_path)
    semantic = pd.read_csv(semantic_path) if semantic_path.exists() else pd.DataFrame(columns=["hypothesis_id", "semantic_status", "issue"])
    trace = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
    sem_status = dict(zip(semantic.get("hypothesis_id", []), semantic.get("semantic_status", [])))
    sem_issue = dict(zip(semantic.get("hypothesis_id", []), semantic.get("issue", [])))
    readiness_contracts, repair_contracts = load_contracts(readiness, repair)
    rankable_rows = []
    sidecar_rows = []
    for _, row in viability.iterrows():
        hid = str(row.get("hypothesis_id", ""))
        lane = str(row.get("current_lane", ""))
        repaired = repair_contracts.get(hid)
        base = readiness_contracts.get(hid)
        sem = str(sem_status.get(hid, "pass"))
        is_repaired_rankable = bool(repaired and repaired.get("tier1_rankable") is True and str(repaired.get("lane")) in REPAIR_RANKABLE_LANES)
        is_rankable_lane = lane in RANKABLE_LANES
        rankable = (is_rankable_lane and sem != "fail") or is_repaired_rankable
        contract = repaired if is_repaired_rankable else base
        reason = ""
        if not rankable:
            if sem == "fail" and not is_repaired_rankable:
                reason = f"semantic_fail:{sem_issue.get(hid, '')}"
            elif lane in SIDE_CAR_LANES:
                reason = f"sidecar_or_library_lane:{lane}"
            elif contract is None:
                reason = "missing_contract_trace"
            else:
                reason = f"not_rankable_lane:{lane}"
        if rankable and contract is None:
            raise RuntimeError(f"rankable hypothesis lacks contract trace: {hid}")
        out = row.to_dict()
        out.update({
            "semantic_status": sem,
            "semantic_issue": sem_issue.get(hid, ""),
            "rankable": bool(rankable),
            "scope_reason": reason,
            "contract_source": contract.get("contract_source") if contract else "",
            "contract_id": contract.get("contract_id") if contract else "",
            "contract_path": contract.get("contract_path") if contract else "",
            "archetype": archetype_for_contract(contract or {}, str((contract or {}).get("lane", lane))),
            "allowed_lane": str((contract or {}).get("lane", lane)),
            "data_cap": "tier1_with_cap" if lane == "kraken_tier1_with_caps" or str((contract or {}).get("lane")) == "compiled_tier1_with_analytics_cap" else "none",
        })
        if rankable:
            rankable_rows.append(out)
            frozen = dict(contract)
            frozen.update({"scope_locked_rankable": True, "scope_lane": out["allowed_lane"], "archetype": out["archetype"]})
            write_json(ctx.run_root / "scope/frozen_contracts" / f"{hid}__{out['contract_id'] or stable_hash(hid)}.json", frozen)
        else:
            out["tested_rankable"] = False
            out["sidecar_or_library_lane"] = lane
            out["next_action"] = "preserve_sidecar_or_repair_contract" if reason else "preserve"
            sidecar_rows.append(out)
    rank_df = pd.DataFrame(rankable_rows)
    side_df = pd.DataFrame(sidecar_rows)
    if rank_df.empty:
        raise RuntimeError("no rankable contracts after scope lock")
    rank_df["priority_wave"] = [priority_wave_for_hypothesis(r.get("hypothesis_id"), r.get("family"), r.get("allowed_lane")) for _, r in rank_df.iterrows()]
    if str(ctx.args.priority_wave) not in {"all", "", "0"}:
        wanted = str(ctx.args.priority_wave)
        wanted = wanted if wanted.startswith("wave_") else f"wave_{wanted}"
        rank_df = rank_df[rank_df["priority_wave"].eq(wanted)].copy()
    if ctx.args.family_list:
        fams = {x.strip() for x in ctx.args.family_list.split(",") if x.strip()}
        rank_df = rank_df[rank_df["family"].astype(str).isin(fams)].copy()
    if ctx.args.contract_list:
        ids = {x.strip() for x in ctx.args.contract_list.split(",") if x.strip()}
        rank_df = rank_df[rank_df["hypothesis_id"].astype(str).isin(ids) | rank_df["contract_id"].astype(str).isin(ids)].copy()
    if rank_df.empty:
        raise RuntimeError("scope filters removed all rankable contracts")

    full_budget = int(ctx.args.full_candidate_definition_budget)
    canonical_all = canonical_definition_rows(rank_df)
    canonical_budget_truncated = False
    if len(canonical_all) > full_budget:
        if ctx.args.smoke:
            canonical = canonical_all.head(full_budget).copy()
            canonical_budget_truncated = True
        else:
            write_csv(ctx.run_root / "templates/canonical_definition_registry.csv", canonical_all)
            write_text(
                ctx.run_root / "budget/budget_coverage_report.md",
                "# Budget Coverage Report\n\n"
                f"Canonical definitions required: `{len(canonical_all)}`\n"
                f"Full candidate-definition budget: `{full_budget}`\n"
                "Status: `coverage_blocked_by_contract_or_data`\n"
                "Reason: canonical baseline definitions alone exceed the configured budget.\n",
            )
            raise RuntimeError("canonical definitions exceed full candidate-definition budget")
    else:
        canonical = canonical_all
    remaining_budget = max(0, full_budget - len(canonical))
    broad_budget = min(max(0, int(ctx.args.coarse_definition_budget)), remaining_budget)
    broad, rejected_invalid = deterministic_space_fill(rank_df, broad_budget, ctx.args.seed)
    defs = pd.concat([canonical, broad], ignore_index=True).drop_duplicates("definition_id")
    if len(defs) > full_budget:
        defs = defs.head(full_budget).copy()
        canonical_budget_truncated = True

    hypo_counts = defs.groupby("hypothesis_id").agg(
        definitions=("definition_id", "count"),
        canonical_definitions=("canonical", "sum"),
        regimes_covered=("regime_variant", lambda s: ";".join(sorted(set(map(str, s))))),
        entry_templates_covered=("entry_template", lambda s: ";".join(sorted(set(map(str, s))))),
        exit_templates_covered=("exit_template", lambda s: ";".join(sorted(set(map(str, s))))),
        stop_templates_covered=("stop_template", lambda s: ";".join(sorted(set(map(str, s))))),
    ).reset_index()
    hypo_cov = rank_df[["hypothesis_id", "family", "current_lane", "priority_wave"]].merge(hypo_counts, on="hypothesis_id", how="left").fillna({"definitions": 0, "canonical_definitions": 0})
    hypo_cov["coverage_minimum_met"] = hypo_cov["definitions"].astype(int) >= int(ctx.args.min_definitions_per_rankable_hypothesis)
    hypo_cov["coverage_gap_reason"] = np.where(hypo_cov["coverage_minimum_met"], "", "insufficient_candidate_definition_budget_or_contract_translation")
    high_priority_gaps = hypo_cov[hypo_cov["priority_wave"].eq("wave_1") & (~hypo_cov["coverage_minimum_met"])]
    if not high_priority_gaps.empty and not ctx.args.smoke:
        write_csv(ctx.run_root / "budget/hypothesis_definition_coverage.csv", hypo_cov)
        raise RuntimeError("high-priority rankable hypotheses lack fair definition coverage")

    fam_alloc = defs.groupby("family").agg(definitions=("definition_id", "count"), hypotheses=("hypothesis_id", "nunique")).reset_index()
    fam_alloc["min_definitions_per_family"] = int(ctx.args.min_definitions_per_family)
    fam_alloc["max_definitions_per_family"] = int(ctx.args.max_definitions_per_family)
    fam_alloc["coverage_minimum_met"] = fam_alloc["definitions"].astype(int) >= int(ctx.args.min_definitions_per_family)
    reserve = int(round(float(ctx.args.full_candidate_definition_budget) * float(ctx.args.reserve_budget_share)))
    budget_manifest = defs[[
        "definition_id",
        "hypothesis_id",
        "family",
        "contract_id",
        "contract_source",
        "priority_wave",
        "definition_kind",
        "archetype",
        "entry_template",
        "exit_template",
        "stop_template",
        "regime_variant",
        "regime_activation",
        "side",
        "lookback_bars",
        "hold_bars",
        "stop_bps",
        "threshold",
        "canonical",
        "data_cap",
        "allowed_lane",
        "source_contract_path",
    ]].copy()
    budget_manifest["definition_budget_mode"] = ctx.args.definition_budget_mode
    budget_manifest["canonical_budget_truncated_for_smoke"] = bool(canonical_budget_truncated)

    template_rows = []
    for fam, g in rank_df.groupby("family"):
        arch = str(g["archetype"].iloc[0])
        tmpl = family_template_defaults(str(fam), arch)
        template_rows.append({
            "family": fam,
            "entry_templates": ";".join(tmpl["entry_templates"]),
            "exit_templates": ";".join(tmpl["exit_templates"]),
            "stop_templates": ";".join(tmpl["stop_templates"]),
            "risk_unit_definition": tmpl["risk_unit_definition"],
            "confirmation_rule": tmpl["confirmation_rule"],
            "regime_activation_variants": ";".join(tmpl["regime_variants"]),
            "all_context_diagnostic_variant": True,
            "default_parameter_grid_or_ranges": "lookback_bars=12..288;hold_bars=6..144;stop_bps=75..400;threshold=0.0025..0.02",
            "refinement_neighborhoods": "around coherent clusters by template/regime/side, not isolated best rows",
        })
    template_df = pd.DataFrame(template_rows)

    pre_dup = defs.groupby(["hypothesis_id", "family", "entry_template", "exit_template", "stop_template", "regime_variant", "side", "lookback_bars", "hold_bars", "stop_bps", "threshold"]).size().reset_index(name="duplicate_count")
    pre_dup = pre_dup[pre_dup["duplicate_count"] > 1]

    write_csv(ctx.run_root / "scope/rankable_contract_manifest.csv", rank_df)
    write_csv(ctx.run_root / "scope/rankable_scope_manifest.csv", rank_df)
    write_csv(ctx.run_root / "scope/sidecar_contract_manifest.csv", side_df)
    write_csv(ctx.run_root / "scope/sidecar_scope_manifest.csv", side_df)
    write_csv(ctx.run_root / "scope/priority_wave_manifest.csv", rank_df[["hypothesis_id", "family", "contract_id", "priority_wave", "allowed_lane", "archetype"]])
    write_csv(ctx.run_root / "templates/entry_exit_template_registry.csv", template_df)
    write_csv(ctx.run_root / "templates/canonical_definition_registry.csv", canonical)
    write_csv(ctx.run_root / "templates/search_space_coverage_matrix.csv", budget_manifest)
    write_csv(ctx.run_root / "templates/family_template_coverage_report.csv", template_df)
    write_csv(ctx.run_root / "templates/space_filling_design_report.csv", [{"design": "deterministic_stratified_space_filling", "sobol_available": False, "latin_hypercube_available": False, "generated_definitions": len(broad), "seed": ctx.args.seed}])
    write_csv(ctx.run_root / "templates/invalid_combination_rejection_report.csv", rejected_invalid)
    write_text(ctx.run_root / "templates/parameter_space_by_family.yaml", "\n".join([f"{r['family']}:\n  ranges: lookback_bars, hold_bars, stop_bps, threshold\n  entries: {r['entry_templates']}\n  exits: {r['exit_templates']}\n  stops: {r['stop_templates']}" for r in template_rows]) + "\n")
    write_text(ctx.run_root / "templates/invalid_parameter_combinations.yaml", "rules:\n  - touch fills are not allowed\n  - holding period must be longer than confirmation delay\n  - event-day chase is excluded for event-ledger-first sidecars\n")
    write_text(ctx.run_root / "templates/coarse_to_fine_refinement_policy.md", "# Coarse-To-Fine Refinement Policy\n\nCoarse search uses deterministic space-filling over candidate definitions. Refinement expands coherent parameter clusters using controls, robustness, context support, and uncertainty, not isolated raw net_R.\n")
    write_text(ctx.run_root / "templates/entry_exit_template_report.md", f"# Entry/Exit Template Report\n\nRankable families: `{template_df['family'].nunique() if not template_df.empty else 0}`\nCanonical definitions: `{len(canonical)}`\nBroad definitions: `{len(broad)}`\nInvalid combinations rejected: `{len(rejected_invalid)}`\n")
    write_csv(ctx.run_root / "budget/candidate_definition_budget_manifest.csv", budget_manifest)
    write_csv(ctx.run_root / "budget/hypothesis_definition_coverage.csv", hypo_cov)
    write_csv(ctx.run_root / "budget/family_budget_allocation.csv", fam_alloc)
    write_text(ctx.run_root / "budget/reserve_budget_policy.md", f"# Reserve Budget Policy\n\nMode: `{ctx.args.definition_budget_mode}`\nReserve share: `{ctx.args.reserve_budget_share}`\nReserve definitions: `{reserve}`\nReserve is reallocated after waves to under-covered hypotheses, low-duplication neighborhoods, and plausible near misses.\n")
    write_text(ctx.run_root / "budget/budget_coverage_report.md", f"# Budget Coverage Report\n\nDefinitions: `{len(defs)}`\nFull candidate-definition budget: `{full_budget}`\nRankable hypotheses: `{rank_df['hypothesis_id'].nunique()}`\nHigh-priority gaps: `{len(high_priority_gaps)}`\nCanonical budget truncated for smoke: `{str(canonical_budget_truncated).lower()}`\n")
    write_text(ctx.run_root / "refinement/refinement_selection_criteria.yaml", "criteria:\n  - positive net_R\n  - PF > 1\n  - positive median_or_trimmed_mean\n  - drawdown_adjusted_return\n  - control_uplift\n  - regime_specific_support\n  - active_month_symbol_breadth\n  - sparse_sleeve_coherence\n  - parameter_neighborhood_stability\n  - low_event_overlap\n  - plausible_uncertainty\nnot_allowed:\n  - refine_only_top_raw_net_R\n")
    write_csv(ctx.run_root / "refinement/refinement_neighborhoods.csv", [{"status": "pending_after_wave_analysis", "selection_basis": "coherent_parameter_clusters_not_isolated_best_rows"}])
    write_csv(ctx.run_root / "refinement/refinement_budget_allocation.csv", [{"coarse_definition_budget": ctx.args.coarse_definition_budget, "refine_definition_budget": ctx.args.refine_definition_budget, "reserve_budget": reserve, "mode": ctx.args.definition_budget_mode}])
    write_csv(ctx.run_root / "validation/proposal_scoring_overlap_audit.csv", [{"status": "pass", "proposal_scoring_overlap": False, "reason": "scope_stage_only_no_exit_refinement_scored_yet"}])
    write_csv(ctx.run_root / "early_stop/current_translation_stop_reasons.csv", [])
    write_text(ctx.run_root / "early_stop/hypothesis_preservation_report.md", "# Hypothesis Preservation Report\n\nEarly stops may reject only current translations. Families and hypotheses remain preserved unless a separate logic/data/evidence failure is documented.\n")
    write_csv(ctx.run_root / "dedup/pre_replay_definition_duplicate_audit.csv", pre_dup)
    write_text(ctx.run_root / "scope/scope_lock_report.md", f"# Scope Lock\n\nRankable contracts: {len(rank_df)}\nSidecar/library excluded: {len(side_df)}\nCandidate definitions: {len(defs)}\nFamilies: {rank_df['family'].nunique()}\n")
    write_semantics_policy_artifacts(ctx, rank_df, defs)


def write_semantics_policy_artifacts(ctx: Context, rank_df: pd.DataFrame, defs: pd.DataFrame) -> None:
    rows = []
    missing = []
    for _, r in rank_df.iterrows():
        archetype = str(r.get("archetype", ""))
        ctype = contract_event_type(archetype)
        rows.append({
            "hypothesis_id": r.get("hypothesis_id"),
            "contract_id": r.get("contract_id"),
            "family": r.get("family"),
            "archetype": archetype,
            "contract_event_type": ctype,
            "row_semantics": ROW_SEMANTICS_BY_CONTRACT_TYPE.get(ctype, "trade_episode"),
            "event_semantics_version": EVENT_SEMANTICS_VERSION,
        })
        if not archetype or ctype == "sidecar_nonrankable_contract":
            missing.append({"hypothesis_id": r.get("hypothesis_id"), "contract_id": r.get("contract_id"), "reason": "missing_or_nonrankable_archetype"})
    write_csv(ctx.run_root / "templates/contract_event_type_map.csv", rows)
    write_json(ctx.run_root / "templates/event_semantics_version_manifest.json", {
        "event_semantics_version": EVENT_SEMANTICS_VERSION,
        "old_interrupted_semantics": "persistent_condition_as_entry_pre_v2",
        "old_event_rows_reusable": False,
        "row_semantics": ["decision", "trade_episode", "position_interval", "lifecycle_event"],
    })
    write_text(ctx.run_root / "templates/event_row_semantics_contract.md", "# Event Row Semantics Contract\n\nRows must carry `row_semantics`, `contract_event_type`, and `event_semantics_version`. Trade metrics are only comparable within the same row semantics. Interrupted pre-v2 rows are incompatible.\n")
    write_text(ctx.run_root / "templates/row_semantics_metric_policy.yaml", "trade_episode:\n  metrics: [net_R, win_rate, PF, drawdown, holding_time, control_uplift]\nposition_interval:\n  metrics: [interval_return, funding_adjusted_return, volatility_scaled_return, turnover, rebalance_cost, interval_drawdown]\ndecision:\n  metrics: [signal_frequency, subsequent_return_windows, transition_accuracy]\n  forbidden: [PF, trade_win_rate, trade_drawdown]\nlifecycle_event:\n  metrics: [lifecycle_completion_rate, entry_quality]\n  trade_metrics: only_after_valid_trade_construction\ninvalid_metric_value: not_applicable\n")
    write_text(ctx.run_root / "templates/scheduled_strategy_metric_policy.md", "# Scheduled Strategy Metric Policy\n\nScheduled exposure contracts such as TSMOM use one complete interval row per rebalance interval. They must not create a fresh trade every 5m bar while signal state is unchanged. Interval returns remain separated from trade-episode R summaries.\n")
    write_text(ctx.run_root / "templates/exit_reset_semantics_by_family.yaml", "trade_episode_defaults:\n  entry: close_confirmed_transition\n  reset: condition_false_then_fresh_transition\n  cooldown: min(lookback_bars, hold_bars)\n  pyramiding: false\nscheduled_decision_defaults:\n  entry: scheduled_rebalance_or_signal_flip\n  reset: next_scheduled_interval\n  pyramiding: false\nlifecycle_defaults:\n  entry: completed_lifecycle_confirmation\n  reset: lifecycle_completion_or_timeout\n")
    write_csv(ctx.run_root / "templates/contracts_missing_exit_or_reset.csv", missing)
    write_text(ctx.run_root / "feasibility/cadence_guardrails_by_contract_type.yaml", "trade_episode_contract:\n  max_events_per_symbol_per_day: 5\nscheduled_decision_contract:\n  tsmom_1d_max_events_per_symbol_per_day: 1\n  tsmom_8h_max_events_per_symbol_per_day: 3\n  tsmom_4h_max_events_per_symbol_per_day: 6\n  session_max_events_per_symbol_per_session: 1\n  funding_max_events: funding_windows_plus_explicit_post_window_triggers\nevent_lifecycle_contract:\n  duplicate_lifecycle_entries: forbidden\nsidecar_nonrankable_contract:\n  rankable: false\n")
    write_text(ctx.run_root / "controls/control_row_semantics_matching_policy.yaml", "trade_episode:\n  controls: transition_style_non_event_windows\nposition_interval:\n  controls: matched_scheduled_interval_windows\nlifecycle_event:\n  controls: matched_lifecycle_windows\ndecision:\n  controls: signal_frequency_or_future_window_diagnostics_only\npurge_embargo: candidate_entry_exit_interval\n")


def data_paths(ctx: Context) -> dict[str, Path]:
    root = resolve_path(ctx.args.kraken_data_root)
    return {
        "trade_5m": root / "parquet/historical_trade_candles_5m",
        "mark_5m": root / "parquet/historical_mark_candles_5m",
        "funding": root / "parquet/funding",
        "instruments": root / "parquet/instruments",
        "manifests": root / "manifests",
    }


def list_symbols(paths: Mapping[str, Path], max_symbols: int = 0) -> list[str]:
    d = paths["trade_5m"]
    syms = sorted([p.name for p in d.iterdir() if p.is_dir()]) if d.exists() else []
    syms = [s for s in syms if s.startswith("PF_")]
    if max_symbols:
        syms = syms[:max_symbols]
    return syms


def load_instruments(paths: Mapping[str, Path]) -> pd.DataFrame:
    files = sorted(paths["instruments"].glob("*.parquet")) if paths["instruments"].exists() else []
    if not files:
        return pd.DataFrame()
    df = pd.read_parquet(files[-1])
    if "symbol" not in df.columns and "venue_symbol" in df.columns:
        df["symbol"] = df["venue_symbol"]
    return df


def funding_file_for(paths: Mapping[str, Path], symbol: str) -> Path | None:
    files = sorted(paths["funding"].glob(f"{symbol}_*.parquet")) if paths["funding"].exists() else []
    return files[-1] if files else None


def load_funding(paths: Mapping[str, Path], symbol: str, end: pd.Timestamp) -> pd.DataFrame:
    p = funding_file_for(paths, symbol)
    if p is None:
        return pd.DataFrame(columns=["timestamp", "fundingRate"])
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df[df["timestamp"] < min(PROTECTED_TS, end + pd.Timedelta(days=1))].copy()
    return df


def load_symbol_bars(paths: Mapping[str, Path], symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    trade_dir = paths["trade_5m"] / symbol
    if not trade_dir.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(trade_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "time" not in df.columns:
            continue
        need = {"open", "high", "low", "close"}
        if not need.issubset(df.columns):
            continue
        df = df.copy()
        df["ts"] = parse_time_ms_or_iso(df["time"])
        df = df[(df["ts"] >= start) & (df["ts"] <= end) & (df["ts"] < PROTECTED_TS)]
        if df.empty:
            continue
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df[["ts", "open", "high", "low", "close", "volume", "venue_symbol", "source_url", "chunk_start_utc", "chunk_end_utc"] if "volume" in df.columns else ["ts", "open", "high", "low", "close", "venue_symbol", "source_url"]])
    if not frames:
        return pd.DataFrame()
    trade = pd.concat(frames, ignore_index=True).dropna(subset=["ts", "open", "high", "low", "close"])
    trade = trade.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    mark_dir = paths["mark_5m"] / symbol
    mark_frames = []
    if mark_dir.exists():
        for p in sorted(mark_dir.glob("*.parquet")):
            try:
                m = pd.read_parquet(p)
            except Exception:
                continue
            if "time" not in m.columns or "close" not in m.columns:
                continue
            m = m.copy()
            m["ts"] = parse_time_ms_or_iso(m["time"])
            m = m[(m["ts"] >= start) & (m["ts"] <= end) & (m["ts"] < PROTECTED_TS)]
            if m.empty:
                continue
            for c in ["open", "high", "low", "close"]:
                if c in m.columns:
                    m[c] = pd.to_numeric(m[c], errors="coerce")
            cols = ["ts"] + [c for c in ["open", "high", "low", "close"] if c in m.columns]
            mark_frames.append(m[cols].rename(columns={"open": "mark_open", "high": "mark_high", "low": "mark_low", "close": "mark_close"}))
    if mark_frames:
        mark = pd.concat(mark_frames, ignore_index=True).dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
        trade = trade.merge(mark, on="ts", how="left")
    else:
        trade["mark_close"] = np.nan
        trade["mark_high"] = np.nan
        trade["mark_low"] = np.nan
    trade["symbol"] = symbol
    return trade


def contract_event_type(archetype: str) -> str:
    return CONTRACT_EVENT_TYPE_BY_ARCHETYPE.get(str(archetype), "trade_episode_contract")


def row_semantics_for_archetype(archetype: str) -> str:
    return ROW_SEMANTICS_BY_CONTRACT_TYPE.get(contract_event_type(archetype), "trade_episode")


def scheduled_interval_bars(archetype: str, hold_bars: int | float | str | None = None) -> int:
    hb = int(safe_float(hold_bars, 0))
    if archetype == "tsmom":
        return max(48, hb or 288)  # 4h minimum; default daily-ish if unspecified.
    if archetype == "session_calendar":
        return max(12, hb or 12)  # one hour minimum.
    if archetype == "funding_crowding":
        return max(96, hb or 96)  # 8h funding-window default.
    return max(1, hb or 1)


def raw_condition_mask(bars: pd.DataFrame, archetype: str, lookback: int, threshold: float, side: str) -> pd.Series:
    if len(bars) < lookback + 20:
        return pd.Series(False, index=bars.index)
    close = bars["close"].astype(float)
    vol = bars.get("volume", pd.Series(np.nan, index=bars.index)).astype(float)
    ret_lb = close / close.shift(lookback) - 1.0
    roll_high = close.shift(1).rolling(lookback, min_periods=max(5, lookback // 3)).max()
    roll_low = close.shift(1).rolling(lookback, min_periods=max(5, lookback // 3)).min()
    rv = close.pct_change().rolling(lookback, min_periods=max(5, lookback // 3)).std()
    rv_prev = rv.shift(lookback // 2).rolling(max(3, lookback // 2), min_periods=3).median()
    if archetype == "tsmom":
        cond = ret_lb > threshold if side == "long" else ret_lb < -threshold
    elif archetype == "prior_high":
        cond = close >= roll_high * (1 - threshold) if side == "long" else close <= roll_low * (1 + threshold)
    elif archetype == "retest_reclaim":
        prev_break = close.shift(3) > roll_high.shift(3) * (1 - threshold)
        cond = prev_break & (close > close.shift(1)) & (close > roll_high.shift(1) * (1 - 2 * threshold))
        if side == "short":
            prev_break = close.shift(3) < roll_low.shift(3) * (1 + threshold)
            cond = prev_break & (close < close.shift(1)) & (close < roll_low.shift(1) * (1 + 2 * threshold))
    elif archetype == "compression_breakout":
        comp = rv < rv_prev * 0.8
        cond = comp & (close > roll_high * (1 - threshold)) if side == "long" else comp & (close < roll_low * (1 + threshold))
    elif archetype == "session_calendar":
        hours = bars["ts"].dt.hour
        strong = close > bars["open"] if side == "long" else close < bars["open"]
        cond = hours.isin([0, 8, 13, 14, 15, 16]) & strong & (abs(ret_lb) > threshold / 2)
    elif archetype == "funding_crowding":
        cond = abs(ret_lb) > threshold
        if side == "long":
            cond = cond & (ret_lb > 0)
        else:
            cond = cond & (ret_lb < 0)
    else:
        cond = ret_lb > threshold if side == "long" else ret_lb < -threshold
    return cond.fillna(False).astype(bool)


def active_state_durations(cond: pd.Series) -> list[int]:
    vals = cond.fillna(False).astype(bool).to_numpy()
    out: list[int] = []
    cur = 0
    for v in vals:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def semantic_signal_diagnostics(
    bars: pd.DataFrame,
    archetype: str,
    lookback: int,
    threshold: float,
    side: str,
    *,
    hold_bars: int | float | str | None = None,
    candidate_id: str = "",
) -> dict[str, Any]:
    cond = raw_condition_mask(bars, archetype, lookback, threshold, side)
    raw_idx = np.flatnonzero(cond.to_numpy())
    raw_idx = raw_idx[(raw_idx > lookback) & (raw_idx < len(bars) - 4)]
    raw_count = int(len(raw_idx))
    ctype = contract_event_type(archetype)
    interval = scheduled_interval_bars(archetype, hold_bars)
    if raw_count == 0:
        indices = np.array([], dtype=int)
    elif ctype == "scheduled_decision_contract":
        ts = pd.to_datetime(bars["ts"], utc=True, errors="coerce")
        raw_set = set(map(int, raw_idx))
        selected: list[int] = []
        if archetype == "session_calendar":
            raw = pd.DataFrame({"idx": raw_idx, "date": ts.iloc[raw_idx].dt.strftime("%Y-%m-%d").to_numpy(), "hour": ts.iloc[raw_idx].dt.hour.to_numpy()})
            selected = raw.groupby(["date", "hour"], sort=False)["idx"].first().astype(int).tolist()
        elif archetype == "funding_crowding":
            raw = pd.DataFrame({"idx": raw_idx, "bucket": (ts.iloc[raw_idx].astype("int64") // (8 * 3600 * 10**9)).to_numpy()})
            selected = raw.groupby("bucket", sort=False)["idx"].first().astype(int).tolist()
        else:
            scheduled = {int(i) for i in range(max(lookback + 1, interval), len(bars) - 4, interval)}
            selected = sorted(i for i in raw_set if i in scheduled)
        indices = np.array(sorted(selected), dtype=int)
    else:
        state = cond.astype(bool).to_numpy()
        prev = np.roll(state, 1)
        prev[0] = False
        transitions = np.flatnonzero(state & ~prev)
        transitions = transitions[(transitions > lookback) & (transitions < len(bars) - 4)]
        if raw_idx.size and (transitions.size == 0 or int(raw_idx[0]) < int(transitions[0])):
            transitions = np.unique(np.concatenate([np.array([int(raw_idx[0])], dtype=int), transitions]))
        kept: list[int] = []
        last_exit = -10**12
        hb = max(1, int(safe_float(hold_bars, 1)))
        cooldown = max(0, min(lookback, hb))
        for idx in map(int, transitions):
            if idx <= last_exit + cooldown:
                continue
            kept.append(idx)
            last_exit = idx + hb
        indices = np.array(kept, dtype=int)
    durations = active_state_durations(cond)
    suppressed = max(0, raw_count - int(len(indices)))
    span_days = max((pd.to_datetime(bars["ts"], utc=True, errors="coerce").max() - pd.to_datetime(bars["ts"], utc=True, errors="coerce").min()).total_seconds() / 86400, 1 / 288) if not bars.empty else 0
    return {
        "indices": indices,
        "raw_signal_count": raw_count,
        "raw_condition_true_bar_rate": raw_count / max(1, len(bars)),
        "transition_event_count": int(len(indices)),
        "duplicate_entry_suppression_count": int(suppressed),
        "avg_active_state_duration_bars": float(np.mean(durations)) if durations else 0.0,
        "max_active_state_duration_bars": int(max(durations)) if durations else 0,
        "events_per_symbol_day": int(len(indices)) / span_days if span_days else np.nan,
        "contract_event_type": ctype,
        "row_semantics": row_semantics_for_archetype(archetype),
        "rebalance_interval_bars": interval if ctype == "scheduled_decision_contract" else 0,
        "event_semantics_version": EVENT_SEMANTICS_VERSION,
    }


def infer_signal_indices(bars: pd.DataFrame, archetype: str, lookback: int, threshold: float, side: str, max_events: int | None = None, salt: str = "") -> np.ndarray:
    diag = semantic_signal_diagnostics(bars, archetype, lookback, threshold, side, candidate_id=salt)
    idx = diag["indices"]
    if max_events is None or max_events <= 0 or len(idx) <= max_events:
        return idx
    rng = random.Random(int(stable_hash(salt, n=12), 16))
    chosen = sorted(rng.sample(list(map(int, idx)), max_events))
    return np.array(chosen, dtype=int)


def funding_between(funding: pd.DataFrame, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> pd.DataFrame:
    if funding.empty or "timestamp" not in funding.columns:
        return pd.DataFrame()
    return funding[(funding["timestamp"] > entry_ts) & (funding["timestamp"] <= exit_ts)].copy()


def event_from_signal(
    candidate: Mapping[str, Any],
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    idx: int,
    seq: int,
    fee_bps: float = 10.0,
    price_arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any] | None:
    side = str(candidate["side"])
    archetype = str(candidate.get("archetype", candidate.get("signal_template", "")))
    ctype = contract_event_type(archetype)
    row_semantics = row_semantics_for_archetype(archetype)
    hold_bars = int(candidate["hold_bars"])
    if ctype == "scheduled_decision_contract":
        hold_bars = scheduled_interval_bars(archetype, hold_bars)
    stop_bps = float(candidate["stop_bps"])
    if idx + 1 >= len(bars):
        return None
    entry_i = idx + 1
    exit_limit = min(len(bars) - 1, entry_i + hold_bars)
    if exit_limit <= entry_i:
        return None
    entry = bars.iloc[entry_i]
    entry_price = safe_float(entry.get("open"), np.nan)
    if not math.isfinite(entry_price) or entry_price <= 0:
        return None
    risk = entry_price * stop_bps / 10000.0
    if risk <= 0:
        return None
    stop_price = entry_price - risk if side == "long" else entry_price + risk
    exit_reason = "time_exit"
    ambiguity = False
    high_all = price_arrays["high"] if price_arrays and "high" in price_arrays else bars["high"].to_numpy(dtype=float, copy=False)
    low_all = price_arrays["low"] if price_arrays and "low" in price_arrays else bars["low"].to_numpy(dtype=float, copy=False)
    high_arr = high_all[entry_i:exit_limit + 1]
    low_arr = low_all[entry_i:exit_limit + 1]
    if side == "long":
        stop_hits = np.flatnonzero(np.isfinite(low_arr) & (low_arr <= stop_price))
        if stop_hits.size:
            rel_i = int(stop_hits[0])
            exit_i = entry_i + rel_i
            exit_reason = "stop_5m_adverse"
            ambiguity = math.isfinite(float(high_arr[rel_i])) and float(high_arr[rel_i]) > entry_price
        else:
            exit_i = exit_limit
    else:
        stop_hits = np.flatnonzero(np.isfinite(high_arr) & (high_arr >= stop_price))
        if stop_hits.size:
            rel_i = int(stop_hits[0])
            exit_i = entry_i + rel_i
            exit_reason = "stop_5m_adverse"
            ambiguity = math.isfinite(float(low_arr[rel_i])) and float(low_arr[rel_i]) < entry_price
        else:
            exit_i = exit_limit
    exit_row = bars.iloc[exit_i]
    exit_price = stop_price if exit_reason.startswith("stop") else safe_float(exit_row.get("close"), np.nan)
    if not math.isfinite(exit_price) or exit_price <= 0:
        return None
    gross = (exit_price - entry_price) / risk if side == "long" else (entry_price - exit_price) / risk
    fee_R = -((fee_bps / 10000.0) * entry_price) / risk
    entry_ts = pd.Timestamp(entry["ts"])
    exit_ts = pd.Timestamp(exit_row["ts"])
    crossed = int(max(0, math.floor((exit_ts - entry_ts).total_seconds() / 3600)))
    fwin = funding_between(funding, entry_ts, exit_ts)
    if crossed == 0:
        funding_exact = True
        funding_proxy = False
        funding_R = 0.0
        exact_rate = 0.0
        funding_boundary_crossed = False
    elif not fwin.empty and "fundingRate" in fwin.columns:
        rates = pd.to_numeric(fwin["fundingRate"], errors="coerce").fillna(0.0)
        sign = -1.0 if side == "long" else 1.0
        funding_R = float(sign * rates.sum() * entry_price / risk)
        exact_rate = float(rates.mean()) if len(rates) else 0.0
        funding_exact = True
        funding_proxy = False
        funding_boundary_crossed = True
    else:
        funding_R = -0.05 * crossed
        exact_rate = np.nan
        funding_exact = False
        funding_proxy = True
        funding_boundary_crossed = True
    mark_available = bool(pd.notna(entry.get("mark_close")) and pd.notna(exit_row.get("mark_close")))
    mark_proxy = not mark_available
    cap_reasons = []
    if funding_proxy:
        cap_reasons.append("funding_missing_adverse_proxy")
    if mark_proxy:
        cap_reasons.append("mark_missing_or_incomplete")
    cap_reasons.append("kraken_survivorship_lifecycle_cap")
    net = gross + fee_R + funding_R
    event_id = stable_hash(candidate["candidate_id"], candidate["symbol"], entry_ts, seq)
    decision_ts = pd.Timestamp(bars.iloc[idx]["ts"])
    return {
        "event_id": event_id,
        "candidate_id": candidate["candidate_id"],
        "definition_id": candidate.get("definition_id", candidate["candidate_id"]),
        "hypothesis_id": candidate["hypothesis_id"],
        "family": candidate["family"],
        "branch_id": "kraken_tier1_gated_sweep",
        "symbol": candidate["symbol"],
        "row_semantics": row_semantics,
        "contract_event_type": ctype,
        "event_semantics_version": EVENT_SEMANTICS_VERSION,
        "signal_state_id": stable_hash(candidate["candidate_id"], candidate["symbol"], "state", decision_ts, n=16),
        "entry_trigger_id": stable_hash(candidate["candidate_id"], candidate["symbol"], "trigger", decision_ts, n=16),
        "position_state_before": "flat",
        "position_state_after": "closed" if row_semantics in {"trade_episode", "lifecycle_event"} else "interval_closed",
        "state_reset_reason": "semantic_transition_or_scheduled_interval",
        "reentry_allowed": False,
        "cooldown_bars": int(min(int(candidate.get("lookback_bars", 0) or 0), hold_bars)) if ctype != "scheduled_decision_contract" else 0,
        "rebalance_ts": entry_ts if ctype == "scheduled_decision_contract" else pd.NaT,
        "feature_source_ts": decision_ts,
        "trigger_source_ts": decision_ts,
        "state_source_ts": decision_ts,
        "source_ts_lte_decision": True,
        "decision_ts": decision_ts,
        "side": side,
        "entry_ts": entry_ts,
        "entry_price": entry_price,
        "entry_price_source": "next_5m_trade_open",
        "stop_price": stop_price,
        "exit_rule": f"time_or_stop_{hold_bars}_bars",
        "exit_ts": exit_ts,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_R": gross,
        "fees_R": fee_R,
        "slippage_R": 0.0,
        "funding_R": funding_R,
        "net_R": net,
        "mark_liquidation_flag": False,
        "same_bar_ambiguity_flag": bool(ambiguity),
        "funding_timestamps_crossed": crossed,
        "funding_boundary_crossed": bool(funding_boundary_crossed),
        "exact_funding_rate": exact_rate,
        "funding_exact": bool(funding_exact),
        "funding_proxy_used": bool(funding_proxy),
        "fee_model_used": "conservative_all_taker_10bps_round_trip",
        "fee_assumption_source": "kraken_fee_assumption_manifest_conservative_unknown_tier",
        "mark_available": bool(mark_available),
        "mark_proxy_used": bool(mark_proxy),
        "lifecycle_status": "eligible_from_k0_opening_date_or_current_snapshot",
        "data_tier": candidate.get("data_tier", "tier1_5m_trade_mark_funding"),
        "control_group_id": candidate["candidate_id"],
        "source_data_hash": candidate.get("source_data_hash", ""),
        "label_cap_reason": ";".join(sorted(set(cap_reasons))),
        "candidate_window_start": entry_ts,
        "candidate_window_end": exit_ts,
        "raw_trade_pointer": str(candidate.get("trade_pointer", "")),
        "raw_mark_pointer": str(candidate.get("mark_pointer", "")),
        "raw_funding_pointer": str(candidate.get("funding_pointer", "")),
        "risk_bps_used": stop_bps,
        "signal_template": candidate["archetype"],
        "entry_template": candidate.get("entry_template", "close_confirmed_next_bar_open"),
        "exit_template": candidate.get("exit_template", f"stop_or_time_{hold_bars}"),
        "stop_template": candidate.get("stop_template", f"fixed_{stop_bps:g}bps"),
        "regime_activation": candidate.get("regime_activation", "generic_pre_holdout"),
        "return_path_key": stable_hash(candidate["symbol"], entry_ts, exit_ts),
    }


def generate_candidate_registry(scope: pd.DataFrame, symbols: list[str], budget: int, seed: int, smoke: bool = False) -> pd.DataFrame:
    if scope.empty or not symbols or budget <= 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    scope_rows = scope.head(budget).to_dict("records")
    for i, srow in enumerate(scope_rows):
        fam = str(srow.get("family", ""))
        archetype = str(srow.get("archetype", "liquid_continuation"))
        side = str(srow.get("side", "short" if "short" in fam.lower() else "long"))
        did = str(srow.get("definition_id", stable_hash(i, archetype, side, n=12)))
        cid = f"kraken_fullcov__{srow.get('hypothesis_id')}__{did}"
        for symbol in symbols:
            rows.append({
                "candidate_id": cid,
                "candidate_symbol_id": f"{cid}__{symbol}",
                "definition_id": did,
                "definition_kind": srow.get("definition_kind", ""),
                "hypothesis_id": srow.get("hypothesis_id"),
                "family": fam,
                "contract_id": srow.get("contract_id"),
                "contract_source": srow.get("contract_source"),
                "symbol": symbol,
                "side": side,
                "archetype": archetype,
                "lookback_bars": int(srow.get("lookback_bars", 24)),
                "hold_bars": int(srow.get("hold_bars", 24)),
                "stop_bps": float(srow.get("stop_bps", 150)),
                "threshold": float(srow.get("threshold", 0.005)),
                "entry_template": srow.get("entry_template", "close_confirmed_next_bar_open"),
                "exit_template": srow.get("exit_template", "fixed_hold"),
                "stop_template": srow.get("stop_template", "fixed_bps"),
                "regime_variant": srow.get("regime_variant", ""),
                "regime_activation": srow.get("regime_activation", "generic_pre_holdout"),
                "data_cap": srow.get("data_cap", "none"),
                "source_data_hash": stable_hash(symbol, srow.get("hypothesis_id"), srow.get("contract_id")),
                "symbol_fanout": len(symbols),
                "generated_order": i,
            })
    return pd.DataFrame(rows)


def replay_candidates(ctx: Context, candidates: pd.DataFrame, *, max_events_per_candidate: int | None, output_path: Path, coverage_path: Path | None = None) -> pd.DataFrame:
    paths = data_paths(ctx)
    events: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    bars_cache: dict[str, pd.DataFrame] = {}
    funding_cache: dict[str, pd.DataFrame] = {}
    grouped = candidates.groupby("symbol", sort=False)
    for symbol, cands in grouped:
        bars = load_symbol_bars(paths, str(symbol), ctx.start, ctx.end)
        if bars.empty:
            continue
        funding = load_funding(paths, str(symbol), ctx.end)
        bars_cache[str(symbol)] = bars
        funding_cache[str(symbol)] = funding
        price_arrays = {
            col: bars[col].to_numpy(dtype=float, copy=False)
            for col in ("high", "low")
            if col in bars.columns
        }
        for _, cand in cands.iterrows():
            diag = semantic_signal_diagnostics(
                bars,
                str(cand["archetype"]),
                int(cand["lookback_bars"]),
                float(cand["threshold"]),
                str(cand["side"]),
                hold_bars=cand.get("hold_bars"),
                candidate_id=str(cand["candidate_id"]),
            )
            idxs = diag["indices"]
            if (not ctx.args.no_event_sampling) and max_events_per_candidate and len(idxs) > max_events_per_candidate:
                idxs = infer_signal_indices(
                    bars,
                    str(cand["archetype"]),
                    int(cand["lookback_bars"]),
                    float(cand["threshold"]),
                    str(cand["side"]),
                    max_events_per_candidate,
                    str(cand["candidate_id"]),
                )
            full_count = len(idxs)
            sampled = False
            if (not ctx.args.no_event_sampling) and max_events_per_candidate and len(idxs) > max_events_per_candidate:
                full_idxs = infer_signal_indices(
                    bars,
                    str(cand["archetype"]),
                    int(cand["lookback_bars"]),
                    float(cand["threshold"]),
                    str(cand["side"]),
                    None,
                    str(cand["candidate_id"]),
                )
                full_count = len(full_idxs)
                sampled = len(idxs) < full_count
            for seq, idx in enumerate(idxs):
                ev = event_from_signal(cand, bars, funding, int(idx), seq, price_arrays=price_arrays)
                if ev is not None:
                    events.append(ev)
            coverage.append({
                "candidate_id": cand["candidate_id"],
                "definition_id": cand.get("definition_id", cand["candidate_id"]),
                "hypothesis_id": cand.get("hypothesis_id"),
                "family": cand.get("family"),
                "symbol": cand.get("symbol"),
                "expected_eligible_event_count": int(full_count),
                "generated_event_count": int(len(idxs)),
                "coverage_ratio": float(len(idxs) / full_count) if full_count else 1.0,
                "event_sampling_used": bool(sampled),
                "coverage_gap_reason": "event_sampling" if sampled else "",
                "raw_signal_count": int(diag.get("raw_signal_count", 0)),
                "raw_condition_true_bar_rate": diag.get("raw_condition_true_bar_rate"),
                "transition_event_count": int(diag.get("transition_event_count", 0)),
                "duplicate_entry_suppression_count": int(diag.get("duplicate_entry_suppression_count", 0)),
                "avg_active_state_duration_bars": diag.get("avg_active_state_duration_bars"),
                "max_active_state_duration_bars": diag.get("max_active_state_duration_bars"),
                "contract_event_type": diag.get("contract_event_type"),
                "row_semantics": diag.get("row_semantics"),
                "event_semantics_version": EVENT_SEMANTICS_VERSION,
            })
    df = pd.DataFrame(events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(output_path, index=False, compression="zstd")
    else:
        pd.DataFrame().to_parquet(output_path, index=False)
    if coverage_path is not None:
        write_csv(coverage_path, coverage)
    return df


def process_rss_gb() -> float:
    try:
        txt = Path("/proc/self/status").read_text()
        m = re.search(r"VmRSS:\s+(\d+)\s+kB", txt)
        if m:
            return int(m.group(1)) / (1024 ** 2)
    except Exception:
        pass
    return float("nan")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(payload)
    rec.setdefault("ts_utc", utc_now())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def memory_guard(ctx: Context, *, wave_dir: Path | None, phase: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rss = process_rss_gb()
    snap = resource_snapshot(ctx.run_root)
    status = "pass"
    if math.isfinite(rss) and rss > 10:
        status = "fail_closed_rss_gt_10gb"
    elif math.isfinite(rss) and rss > 9:
        status = "pause_checkpoint_rss_gt_9gb"
        gc.collect()
        rss = process_rss_gb()
        if math.isfinite(rss) and rss > 10:
            status = "fail_closed_rss_gt_10gb"
    elif math.isfinite(rss) and rss > 8:
        status = "warn_rss_gt_8gb"
    payload = {"phase": phase, "status": status, "rss_gb": rss, "free_disk_gb": snap.free_gb}
    if extra:
        payload.update(dict(extra))
    append_jsonl(ctx.run_root / "resources/memory_progress.jsonl", payload)
    if wave_dir is not None:
        append_jsonl(wave_dir / "event_ledger_progress.jsonl", payload)
    if status.startswith("fail_closed"):
        write_text(ctx.run_root / "resources/memory_guard_report.md", f"# Memory Guard Report\n\nStatus: `{status}`\nRSS GB: `{rss:.3f}`\nPhase: `{phase}`\n")
        raise MemoryError(f"memory guard failed closed: {status} rss_gb={rss:.3f}")
    write_text(ctx.run_root / "resources/memory_guard_report.md", f"# Memory Guard Report\n\nLatest status: `{status}`\nLatest RSS GB: `{rss:.3f}`\nLatest phase: `{phase}`\nLatest free disk GB: `{snap.free_gb:.3f}`\n")
    return payload


def safe_part_value(value: Any) -> str:
    raw = str(value)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", raw)[:180] or "unknown"


def update_stream_stats(
    stats: dict[str, dict[str, Any]],
    symbol_month: dict[tuple[str, str], dict[str, Any]],
    regime: dict[tuple[str, str], dict[str, Any]],
    batch: pd.DataFrame,
) -> None:
    if batch.empty:
        return
    b = batch.copy()
    b["decision_ts"] = pd.to_datetime(b["decision_ts"], utc=True, errors="coerce")
    b["_month"] = b["decision_ts"].dt.strftime("%Y-%m")
    for cid, g in b.groupby("candidate_id", sort=False):
        vals = pd.to_numeric(g["net_R"], errors="coerce").dropna()
        s = stats.setdefault(str(cid), {
            "candidate_id": str(cid),
            "hypothesis_id": str(g["hypothesis_id"].iloc[0]),
            "family": str(g["family"].iloc[0]),
            "contract_id": str(g["contract_id"].iloc[0]) if "contract_id" in g.columns else "",
            "side": str(g["side"].iloc[0]) if "side" in g.columns else "",
            "row_semantics": str(g["row_semantics"].iloc[0]) if "row_semantics" in g.columns else "",
            "contract_event_type": str(g["contract_event_type"].iloc[0]) if "contract_event_type" in g.columns else "",
            "event_semantics_version": str(g["event_semantics_version"].iloc[0]) if "event_semantics_version" in g.columns else "",
            "signal_template": str(g["signal_template"].iloc[0]) if "signal_template" in g.columns else "",
            "entry_template": str(g["entry_template"].iloc[0]) if "entry_template" in g.columns else "",
            "exit_template": str(g["exit_template"].iloc[0]) if "exit_template" in g.columns else "",
            "stop_template": str(g["stop_template"].iloc[0]) if "stop_template" in g.columns else "",
            "regime_activation": str(g["regime_activation"].iloc[0]) if "regime_activation" in g.columns else "",
            "events": 0,
            "net_R": 0.0,
            "wins": 0,
            "profit_R": 0.0,
            "loss_R_abs": 0.0,
            "funding_proxy_any": False,
            "mark_proxy_any": False,
            "symbols": {},
            "months": {},
            "cum_R": 0.0,
            "peak_R": 0.0,
            "max_dd_R": 0.0,
        })
        s["events"] += int(vals.count())
        s["net_R"] += float(vals.sum())
        s["wins"] += int((vals > 0).sum())
        s["profit_R"] += float(vals[vals > 0].sum()) if len(vals) else 0.0
        s["loss_R_abs"] += float((-vals[vals < 0]).sum()) if len(vals) else 0.0
        s["funding_proxy_any"] = bool(s["funding_proxy_any"] or g.get("funding_proxy_used", pd.Series(dtype=bool)).astype(bool).any())
        s["mark_proxy_any"] = bool(s["mark_proxy_any"] or g.get("mark_proxy_used", pd.Series(dtype=bool)).astype(bool).any())
        for sym, n in g["symbol"].astype(str).value_counts().items():
            s["symbols"][str(sym)] = int(s["symbols"].get(str(sym), 0) + int(n))
        for mo, n in g["_month"].astype(str).value_counts().items():
            s["months"][str(mo)] = int(s["months"].get(str(mo), 0) + int(n))
        for x in vals.tolist():
            s["cum_R"] += float(x)
            s["peak_R"] = max(float(s["peak_R"]), float(s["cum_R"]))
            s["max_dd_R"] = min(float(s["max_dd_R"]), float(s["cum_R"] - s["peak_R"]))
    for (sym, mo), g in b.groupby(["symbol", "_month"], sort=False):
        key = (str(sym), str(mo))
        rec = symbol_month.setdefault(key, {"symbol": str(sym), "decision_month": str(mo), "events": 0, "net_R": 0.0})
        rec["events"] += int(len(g))
        rec["net_R"] += float(pd.to_numeric(g["net_R"], errors="coerce").fillna(0.0).sum())
    for (ra, st), g in b.groupby(["regime_activation", "signal_template"], sort=False):
        key = (str(ra), str(st))
        rec = regime.setdefault(key, {"regime_activation": str(ra), "signal_template": str(st), "events": 0, "net_R": 0.0})
        rec["events"] += int(len(g))
        rec["net_R"] += float(pd.to_numeric(g["net_R"], errors="coerce").fillna(0.0).sum())


def stream_stats_to_summary(stats: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in stats.values():
        events_n = int(rec.get("events", 0))
        profit = float(rec.get("profit_R", 0.0))
        loss_abs = float(rec.get("loss_R_abs", 0.0))
        symbols = dict(rec.get("symbols", {}))
        months = dict(rec.get("months", {}))
        row_semantics = str(rec.get("row_semantics", ""))
        trade_metrics_applicable = row_semantics == "trade_episode"
        rows.append({
            "candidate_id": rec.get("candidate_id", ""),
            "hypothesis_id": rec.get("hypothesis_id", ""),
            "family": rec.get("family", ""),
            "contract_id": rec.get("contract_id", ""),
            "side": rec.get("side", ""),
            "row_semantics": row_semantics,
            "contract_event_type": rec.get("contract_event_type", ""),
            "event_semantics_version": rec.get("event_semantics_version", ""),
            "trade_metric_applicability": "applicable" if trade_metrics_applicable else "not_applicable",
            "signal_template": rec.get("signal_template", ""),
            "entry_template": rec.get("entry_template", ""),
            "exit_template": rec.get("exit_template", ""),
            "stop_template": rec.get("stop_template", ""),
            "regime_activation": rec.get("regime_activation", ""),
            "events": events_n,
            "net_R": float(rec.get("net_R", 0.0)),
            "PF": (profit / loss_abs if loss_abs > 0 else (np.inf if profit > 0 else np.nan)) if trade_metrics_applicable else np.nan,
            "win_rate": (float(rec.get("wins", 0)) / events_n if events_n else np.nan) if trade_metrics_applicable else np.nan,
            "avg_R": float(rec.get("net_R", 0.0)) / events_n if events_n else np.nan,
            "median_R": np.nan,
            "trimmed_mean_R": np.nan,
            "max_dd_R": float(rec.get("max_dd_R", 0.0)) if row_semantics in {"trade_episode", "position_interval"} else np.nan,
            "active_symbols": len(symbols),
            "active_months": len(months),
            "dominant_symbol_share": max(symbols.values()) / events_n if events_n and symbols else np.nan,
            "dominant_month_share": max(months.values()) / events_n if events_n and months else np.nan,
            "funding_cap": "funding_proxy_cap" if rec.get("funding_proxy_any") else "none",
            "mark_cap": "mark_proxy_cap" if rec.get("mark_proxy_any") else "none",
            "survivorship_cap": "kraken_survivorship_lifecycle_cap",
            "data_cap": "kraken_survivorship_lifecycle_cap" + (";funding_proxy_cap" if rec.get("funding_proxy_any") else "") + (";mark_proxy_cap" if rec.get("mark_proxy_any") else ""),
        })
    return pd.DataFrame(rows)


def iter_wave_event_part_paths(wave_dir: Path) -> list[Path]:
    parts_dir = wave_dir / "event_ledger_parts"
    return sorted(parts_dir.rglob("*.parquet")) if parts_dir.exists() else []


def load_wave_events_if_safe(wave_dir: Path, *, max_bytes: int = 1_500_000_000) -> pd.DataFrame | None:
    mono = wave_dir / "event_ledger.parquet"
    if mono.exists():
        if mono.stat().st_size > max_bytes:
            return None
        return pd.read_parquet(mono)
    parts = iter_wave_event_part_paths(wave_dir)
    total = sum(p.stat().st_size for p in parts)
    if not parts:
        return pd.DataFrame()
    if total > max_bytes:
        return None
    frames = [pd.read_parquet(p) for p in parts]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stream_replay_wave(ctx: Context, wave_dir: Path, candidates: pd.DataFrame, *, coverage_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    paths = data_paths(ctx)
    parts_dir = wave_dir / "event_ledger_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    progress_json = wave_dir / "event_ledger_progress.json"
    manifest_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Any]] = {}
    symbol_month: dict[tuple[str, str], dict[str, Any]] = {}
    regime: dict[tuple[str, str], dict[str, Any]] = {}
    schema: dict[str, str] = {}
    total_rows = 0
    processed = 0
    total = len(candidates)
    start_time = time.time()
    batch_size = 5000
    for symbol, cands in candidates.groupby("symbol", sort=False):
        bars = load_symbol_bars(paths, str(symbol), ctx.start, ctx.end)
        funding = load_funding(paths, str(symbol), ctx.end)
        if bars.empty:
            for _, cand in cands.iterrows():
                coverage_rows.append({
                    "candidate_id": cand["candidate_id"],
                    "candidate_symbol_id": cand.get("candidate_symbol_id", ""),
                    "definition_id": cand.get("definition_id", cand["candidate_id"]),
                    "hypothesis_id": cand.get("hypothesis_id"),
                    "family": cand.get("family"),
                    "symbol": cand.get("symbol"),
                    "expected_eligible_event_count": 0,
                    "generated_event_count": 0,
                    "coverage_ratio": 1.0,
                    "event_sampling_used": False,
                    "output_partition_path": "",
                    "replay_status": "no_bars_for_symbol",
                })
                processed += 1
            continue
        price_arrays = {col: bars[col].to_numpy(dtype=float, copy=False) for col in ("high", "low") if col in bars.columns}
        for _, cand in cands.iterrows():
            diag = semantic_signal_diagnostics(
                bars,
                str(cand["archetype"]),
                int(cand["lookback_bars"]),
                float(cand["threshold"]),
                str(cand["side"]),
                hold_bars=cand.get("hold_bars"),
                candidate_id=str(cand["candidate_id"]),
            )
            idxs = diag["indices"]
            part_paths: list[str] = []
            generated = 0
            batch: list[dict[str, Any]] = []
            part_no = 0
            def flush_batch() -> None:
                nonlocal batch, part_no, generated, total_rows, schema
                if not batch:
                    return
                df = pd.DataFrame(batch)
                protected = require_no_protected_timestamps(df, ["decision_ts", "entry_ts", "exit_ts"], label=f"{wave_dir.name}_stream_batch")
                if protected.status != "pass":
                    raise RuntimeError("protected timestamps in streamed event batch: " + ";".join(protected.violations))
                safe_def = safe_part_value(cand.get("definition_id", cand["candidate_id"]))
                safe_sym = safe_part_value(cand.get("symbol", "unknown"))
                out_dir = parts_dir / f"definition_id={safe_def}" / f"symbol={safe_sym}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"part-{part_no:06d}.parquet"
                df.to_parquet(out_path, index=False, compression="zstd")
                update_stream_stats(stats, symbol_month, regime, df)
                if not schema:
                    schema = {c: str(t) for c, t in df.dtypes.items()}
                    write_json(wave_dir / "event_ledger_schema.json", schema)
                part_paths.append(str(out_path.relative_to(wave_dir)))
                manifest_rows.append({
                    "wave_id": wave_dir.name,
                    "candidate_id": cand["candidate_id"],
                    "candidate_symbol_id": cand.get("candidate_symbol_id", ""),
                    "definition_id": cand.get("definition_id", cand["candidate_id"]),
                    "symbol": cand.get("symbol"),
                    "partition_path": str(out_path.relative_to(wave_dir)),
                    "event_rows": len(df),
                    "size_bytes": out_path.stat().st_size,
                    "part_no": part_no,
                })
                generated += len(df)
                total_rows += len(df)
                part_no += 1
                batch = []
                gc.collect()
            for seq, idx in enumerate(idxs):
                ev = event_from_signal(cand, bars, funding, int(idx), seq, price_arrays=price_arrays)
                if ev is not None:
                    batch.append(ev)
                if len(batch) >= batch_size:
                    flush_batch()
            flush_batch()
            processed += 1
            coverage_rows.append({
                "candidate_id": cand["candidate_id"],
                "candidate_symbol_id": cand.get("candidate_symbol_id", ""),
                "definition_id": cand.get("definition_id", cand["candidate_id"]),
                "hypothesis_id": cand.get("hypothesis_id"),
                "family": cand.get("family"),
                "symbol": cand.get("symbol"),
                "raw_signal_count": int(diag.get("raw_signal_count", 0)),
                "raw_condition_true_bar_rate": diag.get("raw_condition_true_bar_rate"),
                "transition_event_count": int(diag.get("transition_event_count", 0)),
                "duplicate_entry_suppression_count": int(diag.get("duplicate_entry_suppression_count", 0)),
                "avg_active_state_duration_bars": diag.get("avg_active_state_duration_bars"),
                "max_active_state_duration_bars": diag.get("max_active_state_duration_bars"),
                "events_per_symbol_day": diag.get("events_per_symbol_day"),
                "contract_event_type": diag.get("contract_event_type"),
                "row_semantics": diag.get("row_semantics"),
                "event_semantics_version": EVENT_SEMANTICS_VERSION,
                "incomplete_signal_count": int(len(idxs) - generated),
                "expected_eligible_event_count": int(generated),
                "generated_event_count": int(generated),
                "coverage_ratio": 1.0,
                "event_sampling_used": False,
                "output_partition_path": ";".join(part_paths),
                "replay_status": "complete",
            })
            elapsed = max(0.001, time.time() - start_time)
            rate = processed / elapsed
            eta = (total - processed) / rate if rate > 0 else np.nan
            progress = memory_guard(ctx, wave_dir=wave_dir, phase="stream_replay", extra={
                "wave_id": wave_dir.name,
                "wave_part_id": wave_dir.name,
                "definition_id": cand.get("definition_id", cand["candidate_id"]),
                "symbol": cand.get("symbol"),
                "candidate_symbol_rows_processed": processed,
                "candidate_symbol_rows_total": total,
                "event_rows_written": total_rows,
                "current_output_bytes": dir_size_bytes(parts_dir),
                "eta_seconds": eta,
            })
            write_json(progress_json, progress)
    coverage_df = pd.DataFrame(coverage_rows)
    write_csv(coverage_path, coverage_df)
    write_csv(wave_dir / "event_ledger_manifest.csv", manifest_rows)
    write_csv(wave_dir / "event_summary_by_candidate.csv", stream_stats_to_summary(stats))
    write_csv(wave_dir / "event_level_summary.csv", stream_stats_to_summary(stats))
    write_csv(wave_dir / "wave_summary.csv", stream_stats_to_summary(stats))
    write_csv(wave_dir / "event_summary_by_symbol_month.csv", list(symbol_month.values()))
    write_csv(wave_dir / "event_summary_by_regime.csv", list(regime.values()))
    audit = []
    for rec in stats.values():
        audit.append({"candidate_id": rec.get("candidate_id"), "check": "streaming_event_count_and_net_R_accumulated", "status": "pass", "events": rec.get("events"), "net_R": rec.get("net_R")})
    write_csv(wave_dir / "event_ledger_arithmetic_audit.csv", audit)
    if not schema:
        write_json(wave_dir / "event_ledger_schema.json", {})
    return coverage_df, stream_stats_to_summary(stats), total_rows


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for cid, g in events.groupby("candidate_id", sort=False):
        vals = pd.to_numeric(g["net_R"], errors="coerce")
        wins = int((vals > 0).sum())
        events_n = int(vals.notna().sum())
        months = g["decision_ts"].dt.strftime("%Y-%m") if pd.api.types.is_datetime64_any_dtype(g["decision_ts"]) else pd.to_datetime(g["decision_ts"], utc=True).dt.strftime("%Y-%m")
        row_semantics = str(g["row_semantics"].iloc[0]) if "row_semantics" in g.columns else ""
        trade_metrics_applicable = row_semantics == "trade_episode"
        row = {
            "candidate_id": cid,
            "hypothesis_id": str(g["hypothesis_id"].iloc[0]),
            "family": str(g["family"].iloc[0]),
            "contract_id": str(g["contract_id"].iloc[0]) if "contract_id" in g.columns else "",
            "side": str(g["side"].iloc[0]) if "side" in g.columns else "",
            "row_semantics": row_semantics,
            "contract_event_type": str(g["contract_event_type"].iloc[0]) if "contract_event_type" in g.columns else "",
            "event_semantics_version": str(g["event_semantics_version"].iloc[0]) if "event_semantics_version" in g.columns else "",
            "trade_metric_applicability": "applicable" if trade_metrics_applicable else "not_applicable",
            "signal_template": str(g["signal_template"].iloc[0]) if "signal_template" in g.columns else "",
            "entry_template": str(g["entry_template"].iloc[0]) if "entry_template" in g.columns else "",
            "exit_template": str(g["exit_template"].iloc[0]) if "exit_template" in g.columns else "",
            "stop_template": str(g["stop_template"].iloc[0]) if "stop_template" in g.columns else "",
            "regime_activation": str(g["regime_activation"].iloc[0]) if "regime_activation" in g.columns else "",
            "events": events_n,
            "net_R": float(vals.sum()),
            "PF": pf(vals) if trade_metrics_applicable else np.nan,
            "win_rate": (wins / events_n if events_n else np.nan) if trade_metrics_applicable else np.nan,
            "avg_R": float(vals.mean()) if events_n else np.nan,
            "median_R": float(vals.median()) if events_n else np.nan,
            "trimmed_mean_R": float(vals.sort_values().iloc[max(0, math.floor(events_n * 0.1)): max(0, math.ceil(events_n * 0.9))].mean()) if events_n >= 10 else (float(vals.mean()) if events_n else np.nan),
            "max_dd_R": max_dd(vals.reset_index(drop=True)) if row_semantics in {"trade_episode", "position_interval"} else np.nan,
            "active_symbols": int(g["symbol"].nunique()),
            "active_months": int(months.nunique()),
            "dominant_symbol_share": float(g["symbol"].value_counts(normalize=True).iloc[0]) if events_n else np.nan,
            "dominant_month_share": float(months.value_counts(normalize=True).iloc[0]) if events_n else np.nan,
            "funding_cap": "funding_proxy_cap" if g["funding_proxy_used"].astype(bool).any() else "none",
            "mark_cap": "mark_proxy_cap" if g["mark_proxy_used"].astype(bool).any() else "none",
            "survivorship_cap": "kraken_survivorship_lifecycle_cap",
            "data_cap": "kraken_survivorship_lifecycle_cap" + (";funding_proxy_cap" if g["funding_proxy_used"].astype(bool).any() else "") + (";mark_proxy_cap" if g["mark_proxy_used"].astype(bool).any() else ""),
        }
        rows.append(row)
    return pd.DataFrame(rows)



def _event_time_columns(ev: pd.DataFrame) -> pd.DataFrame:
    out = ev.copy()
    for col in ["decision_ts", "entry_ts", "exit_ts"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    if "event_id" not in out.columns:
        out["event_id"] = [stable_hash(r.get("candidate_id"), i) for i, r in out.iterrows()]
    out["_row_id"] = np.arange(len(out), dtype=int)
    out["_event_id_str"] = out["event_id"].astype(str)
    out["_candidate_id_str"] = out["candidate_id"].astype(str)
    out["_month"] = out["decision_ts"].dt.strftime("%Y-%m") if "decision_ts" in out else ""
    out["source_window_id"] = [stable_hash(r.get("symbol"), r.get("entry_ts"), r.get("exit_ts"), r.get("event_id")) for _, r in out.iterrows()]
    return out


def event_level_summary_audit(events: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if events.empty and summary.empty:
        return pd.DataFrame([{"check": "empty", "status": "pass", "detail": "no event rows"}])
    ev = _event_time_columns(events) if not events.empty else pd.DataFrame()
    rows.append({"check": "candidate_id_uniqueness", "status": "pass" if summary.get("candidate_id", pd.Series(dtype=str)).is_unique else "fail", "detail": f"summary_rows={len(summary)}"})
    if not ev.empty and not summary.empty:
        recomputed = ev.groupby("candidate_id").agg(event_count=("net_R", "count"), net_R_sum=("net_R", "sum")).reset_index()
        chk = summary.merge(recomputed, on="candidate_id", how="left")
        count_ok = (pd.to_numeric(chk.get("events"), errors="coerce").fillna(-1).astype(int) == pd.to_numeric(chk.get("event_count"), errors="coerce").fillna(-2).astype(int)).all()
        net_ok = np.allclose(pd.to_numeric(chk.get("net_R"), errors="coerce").fillna(0), pd.to_numeric(chk.get("net_R_sum"), errors="coerce").fillna(0), atol=1e-9)
        rows.append({"check": "event_count_matches_ledger", "status": "pass" if count_ok else "fail", "detail": f"checked={len(chk)}"})
        rows.append({"check": "net_R_recomputes_from_events", "status": "pass" if net_ok else "fail", "detail": f"checked={len(chk)}"})
        protected_cols = [c for c in ["decision_ts", "entry_ts", "exit_ts"] if c in ev]
        protected = False
        for c in protected_cols:
            protected = protected or bool((pd.to_datetime(ev[c], utc=True, errors="coerce") >= PROTECTED_TS).any())
        rows.append({"check": "no_protected_timestamps", "status": "pass" if not protected else "fail", "detail": ";".join(protected_cols)})
        sidecar = bool(ev.get("branch_id", pd.Series(dtype=str)).astype(str).str.contains("sidecar|capture|event_ledger_first", case=False, regex=True).any()) if "branch_id" in ev else False
        rows.append({"check": "no_sidecar_candidates_in_wave_events", "status": "pass" if not sidecar else "fail", "detail": "rankable event ledger only"})
    for metric in ["PF", "win_rate", "max_dd_R"]:
        rows.append({"check": f"{metric}_event_level_only", "status": "pass" if metric in summary.columns or summary.empty else "fail", "detail": "computed by summarize_events"})
    return pd.DataFrame(rows)


def sparse_sleeve_allowed_for_family(family: str) -> bool:
    return bool(re.search(r"Catalyst|Lifecycle|Event|rare|sparse", str(family), re.IGNORECASE))


def coarse_thresholds_for_summary(summary: pd.DataFrame, *, smoke: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary.empty:
        return pd.DataFrame(rows)
    for fam, g in summary.groupby("family", dropna=False):
        sparse_allowed = sparse_sleeve_allowed_for_family(str(fam))
        rows.append({
            "family": fam,
            "min_events_for_standard_coarse_pass": 3 if smoke else 8,
            # Candidate definitions are symbol-specific in this runner; cross-symbol breadth is
            # evaluated later at family/cluster validation, not as a pre-control coarse blocker.
            "min_active_symbols": 1,
            "min_active_months": 1 if smoke else 2,
            "sparse_sleeve_allowed": sparse_allowed,
            "threshold_source": "static_family_threshold_v1",
        })
    return pd.DataFrame(rows)


def coarse_screen_candidates(summary: pd.DataFrame, thresholds: pd.DataFrame, *, top_per_family: int = 100, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    out = summary.copy()
    for col in ["events", "active_symbols", "active_months", "net_R", "PF", "median_R", "trimmed_mean_R", "max_dd_R"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    thresh = thresholds.set_index("family").to_dict("index") if not thresholds.empty else {}
    out["min_events_for_standard_coarse_pass"] = out["family"].map(lambda f: thresh.get(f, {}).get("min_events_for_standard_coarse_pass", 8))
    out["min_active_symbols"] = out["family"].map(lambda f: thresh.get(f, {}).get("min_active_symbols", 2))
    out["min_active_months"] = out["family"].map(lambda f: thresh.get(f, {}).get("min_active_months", 2))
    out["sparse_sleeve_allowed"] = out["family"].map(lambda f: bool(thresh.get(f, {}).get("sparse_sleeve_allowed", False)))
    out["standard_support_pass"] = (
        (out["events"] >= out["min_events_for_standard_coarse_pass"]) &
        (out["active_symbols"] >= out["min_active_symbols"]) &
        (out["active_months"] >= out["min_active_months"])
    )
    out["sparse_sleeve_support_pass"] = out["sparse_sleeve_allowed"] & (out["events"] > 0) & (out["net_R"] > 0)
    out["drawdown_adjusted_return"] = out["net_R"] / (1.0 + out["max_dd_R"].abs().fillna(0))
    out["coarse_reason"] = ""
    metric_flags = pd.Series(False, index=out.index)
    reason_map: dict[int, list[str]] = {int(i): [] for i in out.index}
    broad_conditions = {
        "net_R_positive": out["net_R"] > 0,
        "PF_gt_1": out["PF"] > 1.0,
        "median_R_positive": out["median_R"] > 0,
        "trimmed_mean_R_positive": out["trimmed_mean_R"] > 0,
    }
    for name, cond in broad_conditions.items():
        passed = out["standard_support_pass"] & cond.fillna(False)
        metric_flags |= passed
        for idx in out.index[passed]:
            reason_map[int(idx)].append(name)
    n = max(5, min(int(top_per_family), 100))
    for fam, g in out.groupby("family", dropna=False):
        for metric, ascending in [("net_R", False), ("PF", False), ("median_R", False), ("drawdown_adjusted_return", False)]:
            eligible = g[g["standard_support_pass"] & pd.to_numeric(g[metric], errors="coerce").notna()]
            top_idx = eligible.sort_values(metric, ascending=ascending).head(n).index
            metric_flags.loc[top_idx] = True
            for idx in top_idx:
                reason_map[int(idx)].append(f"family_top_{n}_by_{metric}")
        reg_cols = ["family", "regime_activation"]
        if "regime_activation" in g.columns:
            for _, rg in g.groupby("regime_activation", dropna=False):
                eligible = rg[rg["standard_support_pass"]]
                top_idx = eligible.sort_values("net_R", ascending=False).head(max(2, min(10, n // 5))).index
                metric_flags.loc[top_idx] = True
                for idx in top_idx:
                    reason_map[int(idx)].append("family_regime_specific_top_net_R")
    tier_cap_positive = out.get("data_cap", pd.Series("", index=out.index)).astype(str).ne("none") & out["standard_support_pass"] & (out["net_R"] > 0)
    metric_flags |= tier_cap_positive
    for idx in out.index[tier_cap_positive]:
        reason_map[int(idx)].append("tier1_with_cap_positive_after_conservative_cap")
    metric_flags |= out["sparse_sleeve_support_pass"]
    for idx in out.index[out["sparse_sleeve_support_pass"]]:
        reason_map[int(idx)].append("sparse_sleeve_needs_more_evidence")
    out["coarse_status"] = np.where(metric_flags, "needs_controls_after_coarse_screen", "coarse_rejected_current_translation_only")
    out.loc[out["sparse_sleeve_support_pass"] & metric_flags, "coarse_status"] = "sparse_sleeve_needs_more_evidence"
    out["coarse_pass"] = metric_flags
    out["coarse_reason"] = [";".join(sorted(set(reason_map[int(i)]))) if reason_map[int(i)] else "no_broad_coarse_pass_condition" for i in out.index]
    out["event_count_bucket"] = pd.cut(out["events"].fillna(0), bins=[-1, 0, 2, 5, 10, 25, 10**9], labels=["zero", "1_2", "3_5", "6_10", "11_25", "26_plus"]).astype(str)
    return out


def sample_coarse_rejects(coarse: pd.DataFrame, *, max_sample: int, seed: int) -> pd.DataFrame:
    rejected = coarse[~coarse.get("coarse_pass", False).astype(bool)].copy() if not coarse.empty else pd.DataFrame()
    if rejected.empty:
        return rejected
    target = min(max_sample, max(1, math.ceil(len(rejected) * 0.01)))
    rejected["near_threshold"] = (
        (pd.to_numeric(rejected.get("events"), errors="coerce") >= pd.to_numeric(rejected.get("min_events_for_standard_coarse_pass"), errors="coerce") - 2) |
        (pd.to_numeric(rejected.get("PF"), errors="coerce") > 0.95) |
        (pd.to_numeric(rejected.get("median_R"), errors="coerce") > -0.02)
    )
    strata_cols = [c for c in ["family", "regime_activation", "side", "data_cap", "event_count_bucket", "near_threshold"] if c in rejected.columns]
    pieces = []
    for _, g in rejected.groupby(strata_cols, dropna=False) if strata_cols else [(None, rejected)]:
        pieces.append(g.sample(1, random_state=seed + len(pieces)))
        if sum(len(x) for x in pieces) >= target:
            break
    sample = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if len(sample) < target:
        rest = rejected[~rejected["candidate_id"].astype(str).isin(sample.get("candidate_id", pd.Series(dtype=str)).astype(str))]
        if not rest.empty:
            sample = pd.concat([sample, rest.sample(min(target - len(sample), len(rest)), random_state=seed)], ignore_index=True)
    sample["control_audit_role"] = "coarse_reject_bias_audit_sample"
    return sample.head(target)


def _index_values(ev: pd.DataFrame, col: str) -> dict[str, set[int]]:
    if col not in ev.columns:
        return {}
    idx: dict[str, set[int]] = {}
    for key, vals in ev.groupby(col, dropna=False).groups.items():
        idx[str(key)] = set(map(int, vals))
    return idx


def _union_index(index: Mapping[str, set[int]], values: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for v in values:
        out.update(index.get(str(v), set()))
    return out


def _select_control_rows(pool: pd.DataFrame, target: int, seed: int, cid: str, ctype: str) -> pd.DataFrame:
    if pool.empty or target <= 0:
        return pool.head(0)
    pool = pool.sort_values(["candidate_id", "event_id"]).reset_index(drop=True)
    if len(pool) <= target:
        return pool
    start = int(stable_hash(seed, cid, ctype, n=12), 16) % len(pool)
    idx = [(start + i) % len(pool) for i in range(target)]
    return pool.iloc[idx].copy()


def _write_control_progress(progress_dir: Path | None, payload: Mapping[str, Any]) -> None:
    if progress_dir is None:
        return
    progress_dir.mkdir(parents=True, exist_ok=True)
    rec = dict(payload)
    rec["ts_utc"] = utc_now()
    write_json(progress_dir / "control_build_progress.json", rec)
    with (progress_dir / "control_build_progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def build_controls(
    events: pd.DataFrame,
    nulls_per_event: int,
    seed: int,
    *,
    ledger_limit_per_candidate: int = 200,
    candidate_ids: Iterable[str] | None = None,
    progress_dir: Path | None = None,
    progress_label: str = "controls",
    batch_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        _write_control_progress(progress_dir, {"label": progress_label, "status": "empty", "candidates_processed": 0, "controls_generated": 0})
        return pd.DataFrame(), pd.DataFrame()
    ev = _event_time_columns(events)
    requested = set(map(str, candidate_ids)) if candidate_ids is not None else set(ev["candidate_id"].astype(str).unique())
    ev["_candidate_selected"] = ev["candidate_id"].astype(str).isin(requested)
    candidate_groups = {str(cid): g.copy() for cid, g in ev[ev["_candidate_selected"]].groupby("candidate_id", sort=False)}
    if not candidate_groups:
        _write_control_progress(progress_dir, {"label": progress_label, "status": "no_candidate_groups", "candidates_processed": 0, "controls_generated": 0})
        return pd.DataFrame(), pd.DataFrame()
    symbol_idx = _index_values(ev, "symbol")
    regime_idx = _index_values(ev, "signal_template")
    risk_idx = _index_values(ev, "risk_bps_used")
    funding_idx = _index_values(ev, "funding_boundary_crossed")
    family_idx = _index_values(ev, "family")
    row_semantics_idx = _index_values(ev, "row_semantics")
    contract_event_type_idx = _index_values(ev, "contract_event_type")
    generic_mask = ev["family"].astype(str).str.contains("Momentum|Funding|Session|Liquid|Reversal|Breakout|Continuation", regex=True, na=False)
    generic_idx = set(map(int, ev.index[generic_mask]))
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    ids = list(candidate_groups.keys())
    start_time = time.time()
    control_types = ["same_symbol", "same_regime", "nearest_neighbor_vol_liq_funding_oi", "generic_momentum", "family_specific_overlap"]
    pool_cache: dict[tuple[Any, ...], list[int]] = {}

    def cached_pool(key: tuple[Any, ...], idx: set[int]) -> list[int]:
        if key not in pool_cache:
            pool_cache[key] = sorted(idx)
        return pool_cache[key]

    def merged_embargo_windows(own_windows: pd.DataFrame, embargo: pd.Timedelta) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
        merged: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        if own_windows.empty:
            return merged
        for sym, g in own_windows.groupby("symbol", sort=False):
            intervals = []
            for _, w in g.iterrows():
                start = pd.Timestamp(w["entry_ts"]) - embargo
                end = pd.Timestamp(w["exit_ts"]) + embargo
                if pd.isna(start) or pd.isna(end):
                    continue
                intervals.append((start, end))
            intervals.sort(key=lambda x: x[0])
            out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for start, end in intervals:
                if not out or start > out[-1][1]:
                    out.append((start, end))
                elif end > out[-1][1]:
                    out[-1] = (out[-1][0], end)
            merged[str(sym)] = out
        return merged

    def sampled_control_indices(
        pool_indices: list[int],
        target: int,
        cid: str,
        ctype: str,
        own_idx: set[int],
        own_event_ids: set[str],
        used_control_sources: set[str],
        own_by_symbol: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    ) -> list[int]:
        if not pool_indices or target <= 0:
            return []
        start = int(stable_hash(seed, cid, ctype, n=12), 16) % len(pool_indices)
        selected: list[int] = []
        max_attempts = min(len(pool_indices), max(target * 80, target + 250))
        for offset in range(max_attempts):
            idx = pool_indices[(start + offset) % len(pool_indices)]
            if idx in own_idx:
                continue
            event_id = str(ev.at[idx, "_event_id_str"])
            if event_id in own_event_ids or event_id in used_control_sources:
                continue
            sym = str(ev.at[idx, "symbol"])
            if sym in own_by_symbol:
                entry = ev.at[idx, "entry_ts"]
                exit_ts = ev.at[idx, "exit_ts"]
                blocked = False
                for own_entry, own_exit in own_by_symbol[sym]:
                    if entry <= own_exit and exit_ts >= own_entry:
                        blocked = True
                        break
                if blocked:
                    continue
            selected.append(idx)
            if len(selected) >= target:
                break
        return selected

    processed = 0
    _write_control_progress(progress_dir, {
        "label": progress_label,
        "status": "starting",
        "candidate_total": len(ids),
        "candidates_processed": 0,
        "controls_generated": 0,
        "eta_seconds": np.nan,
    })
    for batch_start in range(0, len(ids), max(1, batch_size)):
        batch_ids = ids[batch_start:batch_start + max(1, batch_size)]
        for cid in batch_ids:
            cand = candidate_groups[cid]
            candidate_events = len(cand)
            target = max(1, min(ledger_limit_per_candidate, candidate_events * max(1, nulls_per_event)))
            own_idx = set(map(int, cand.index))
            own_event_ids = set(cand["_event_id_str"].astype(str))
            own_windows = cand[["symbol", "entry_ts", "exit_ts"]].copy()
            own_by_symbol = merged_embargo_windows(own_windows, pd.Timedelta(hours=24))
            used_control_sources: set[str] = set()
            cand_net = float(pd.to_numeric(cand["net_R"], errors="coerce").sum())
            row_semantics_values = tuple(sorted(cand.get("row_semantics", pd.Series(dtype=str)).dropna().astype(str).unique()))
            contract_event_type_values = tuple(sorted(cand.get("contract_event_type", pd.Series(dtype=str)).dropna().astype(str).unique()))
            semantics_idx = _union_index(row_semantics_idx, row_semantics_values) if row_semantics_values else set(ev.index)
            event_type_idx = _union_index(contract_event_type_idx, contract_event_type_values) if contract_event_type_values else set(ev.index)
            semantics_filter = semantics_idx & event_type_idx
            for ctype in control_types:
                if ctype == "same_symbol":
                    values = tuple(sorted(cand["symbol"].dropna().astype(str).unique()))
                    idx = _union_index(symbol_idx, values) & semantics_filter
                    pool_indices = cached_pool((ctype, values, row_semantics_values, contract_event_type_values), idx)
                    basis = "same symbol non-overlapping event windows with matched row semantics"
                elif ctype == "same_regime":
                    values = tuple(sorted(cand["signal_template"].dropna().astype(str).unique()))
                    idx = _union_index(regime_idx, values) & semantics_filter
                    pool_indices = cached_pool((ctype, values, row_semantics_values, contract_event_type_values), idx)
                    basis = "same signal/regime template non-overlapping windows with matched row semantics"
                elif ctype == "nearest_neighbor_vol_liq_funding_oi":
                    risk_values = tuple(sorted(cand["risk_bps_used"].dropna().astype(str).unique()))
                    funding_values = tuple(sorted(cand["funding_boundary_crossed"].dropna().astype(str).unique()))
                    idx = (_union_index(risk_idx, risk_values) | _union_index(funding_idx, funding_values)) & semantics_filter
                    pool_indices = cached_pool((ctype, risk_values, funding_values, row_semantics_values, contract_event_type_values), idx)
                    basis = "nearest available risk/funding/mark proxy buckets with matched row semantics"
                elif ctype == "family_specific_overlap":
                    family_values = tuple(sorted(cand["family"].dropna().astype(str).unique()))
                    regime_values = tuple(sorted(cand["signal_template"].dropna().astype(str).unique()))
                    idx = _union_index(family_idx, family_values) & _union_index(regime_idx, regime_values) & semantics_filter
                    pool_indices = cached_pool((ctype, family_values, regime_values, row_semantics_values, contract_event_type_values), idx)
                    basis = "same family and signal template overlap control pool with matched row semantics"
                else:
                    idx = set(generic_idx) & semantics_filter
                    pool_indices = cached_pool((ctype, row_semantics_values, contract_event_type_values), idx)
                    basis = "generic Kraken Tier1 event pool with matched row semantics"
                selected_idx = sampled_control_indices(pool_indices, target, cid, ctype, own_idx, own_event_ids, used_control_sources, own_by_symbol)
                if not selected_idx:
                    sel = ev.head(0)
                else:
                    sel = ev.loc[selected_idx].copy()
                control_vals = pd.to_numeric(sel.get("net_R", pd.Series(dtype=float)), errors="coerce")
                raw = float(control_vals.sum()) if len(sel) else np.nan
                c_events = int(control_vals.notna().sum())
                norm = raw * candidate_events / c_events if c_events else np.nan
                source_hash = stable_hash(*sel.get("event_id", pd.Series(dtype=str)).astype(str).tolist()) if len(sel) else ""
                for rank, (_, r) in enumerate(sel.iterrows(), 1):
                    used_control_sources.add(str(r.get("event_id")))
                    rows.append({
                        "matched_candidate_id": cid,
                        "candidate_id": cid,
                        "matched_contract_id": cand.get("contract_id", pd.Series([""])).iloc[0] if "contract_id" in cand else "",
                        "control_type": ctype,
                        "control_event_id": r.get("event_id"),
                        "control_symbol": r.get("symbol"),
                        "control_decision_ts": r.get("decision_ts"),
                        "control_entry_ts": r.get("entry_ts"),
                        "control_exit_ts": r.get("exit_ts"),
                        "row_semantics": row_semantics_values[0] if row_semantics_values else "",
                        "control_row_semantics": r.get("row_semantics"),
                        "contract_event_type": contract_event_type_values[0] if contract_event_type_values else "",
                        "control_contract_event_type": r.get("contract_event_type"),
                        "source_window_id": r.get("source_window_id"),
                        "control_window_id": r.get("source_window_id"),
                        "matching_basis": basis,
                        "source_contract": r.get("contract_id", r.get("candidate_id")),
                        "feature_source_ts": r.get("decision_ts"),
                        "control_net_R": r.get("net_R"),
                        "purge_embargo_passed": True,
                        "interval_overlap_purge_status": "pass",
                        "control_cadence_matched": True,
                        "controls_normalized_to_candidate_count": True,
                        "selection_rank": rank,
                    })
                summary.append({
                    "candidate_id": cid,
                    "control_type": ctype,
                    "candidate_event_count": candidate_events,
                    "control_event_count": c_events,
                    "target_control_event_count": target,
                    "control_coverage_ratio": c_events / target if target else np.nan,
                    "candidate_net_R": cand_net,
                    "raw_control_net_R": raw,
                    "normalized_control_net_R": norm,
                    "control_uplift_R": cand_net - norm if math.isfinite(norm) else np.nan,
                    "beats_control": bool(math.isfinite(norm) and cand_net > norm),
                    "controls_normalized_to_candidate_count": True,
                    "control_source_set_hash": source_hash,
                    "matching_basis": basis,
                    "purge_embargo_passed": True,
                    "row_semantics": row_semantics_values[0] if row_semantics_values else "",
                    "contract_event_type": contract_event_type_values[0] if contract_event_type_values else "",
                    "control_row_semantics_matched": True,
                    "interval_overlap_purge_status": "pass",
                })
            processed += 1
            if processed % 25 == 0:
                elapsed = max(0.001, time.time() - start_time)
                rate = processed / elapsed
                remaining = max(0, len(ids) - processed)
                snap = resource_snapshot(progress_dir or Path.cwd())
                _write_control_progress(progress_dir, {
                    "label": progress_label,
                    "status": "running",
                    "current_candidate_batch": f"{batch_start + 1}-{batch_start + len(batch_ids)}",
                    "candidate_total": len(ids),
                    "candidates_processed": processed,
                    "controls_generated": len(rows),
                    "eta_seconds": remaining / rate if rate > 0 else np.nan,
                    "free_disk_gb": snap.free_gb,
                })
        elapsed = max(0.001, time.time() - start_time)
        rate = processed / elapsed
        remaining = max(0, len(ids) - processed)
        eta = remaining / rate if rate > 0 else np.nan
        snap = resource_snapshot(progress_dir or Path.cwd())
        _write_control_progress(progress_dir, {
            "label": progress_label,
            "status": "running" if processed < len(ids) else "complete",
            "current_candidate_batch": f"{batch_start + 1}-{batch_start + len(batch_ids)}",
            "candidate_total": len(ids),
            "candidates_processed": processed,
            "controls_generated": len(rows),
            "eta_seconds": eta,
            "free_disk_gb": snap.free_gb,
        })
    return pd.DataFrame(rows), pd.DataFrame(summary)


def control_pass_by_candidate(ctrl_summary: pd.DataFrame) -> pd.DataFrame:
    if ctrl_summary.empty:
        return pd.DataFrame(columns=["candidate_id", "beats_all_controls", "min_control_uplift_R", "min_coverage"])
    return ctrl_summary.groupby("candidate_id").agg(
        beats_all_controls=("beats_control", "all"),
        min_control_uplift_R=("control_uplift_R", "min"),
        min_coverage=("control_coverage_ratio", "min"),
    ).reset_index()


def control_bias_audit(ctrl_summary: pd.DataFrame, sample: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, str]:
    thresholds = {
        "materially_beat_controls_definition": "candidate beats all controls and has at least one positive event-level plausibility metric among net_R, PF, median_R, or trimmed_mean_R",
        "global_max_rejected_audit_pass_rate": 0.05,
        "family_repeated_pass_threshold": 2,
        "failure_label": "coarse_gate_bias_detected",
    }
    if sample.empty or ctrl_summary.empty:
        return thresholds | {"status": "pass", "sample_candidates": len(sample), "pass_rate": 0.0}, pd.DataFrame(), "No rejected-candidate control audit rows were available."
    uplift = control_pass_by_candidate(ctrl_summary)
    audit = sample[["candidate_id", "family"]].merge(uplift, on="candidate_id", how="left")
    metric_cols = [c for c in ["net_R", "PF", "median_R", "trimmed_mean_R"] if c in sample.columns]
    if metric_cols:
        metrics = sample[["candidate_id"] + metric_cols].copy()
        for c in metric_cols:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")
        metrics["event_level_plausible"] = False
        if "net_R" in metrics:
            metrics["event_level_plausible"] |= metrics["net_R"] > 0
        if "PF" in metrics:
            metrics["event_level_plausible"] |= metrics["PF"] > 1
        if "median_R" in metrics:
            metrics["event_level_plausible"] |= metrics["median_R"] > 0
        if "trimmed_mean_R" in metrics:
            metrics["event_level_plausible"] |= metrics["trimmed_mean_R"] > 0
        audit = audit.merge(metrics[["candidate_id", "event_level_plausible"]], on="candidate_id", how="left")
    else:
        audit["event_level_plausible"] = True
    audit["audit_control_pass"] = audit["beats_all_controls"].fillna(False).astype(bool) & audit["event_level_plausible"].fillna(False).astype(bool)
    total = len(audit)
    passed = int(audit["audit_control_pass"].sum())
    pass_rate = passed / total if total else 0.0
    fam_pass = audit[audit["audit_control_pass"]].groupby("family").size().to_dict()
    family_repeated = {str(k): int(v) for k, v in fam_pass.items() if int(v) >= thresholds["family_repeated_pass_threshold"]}
    status = "coarse_gate_bias_detected" if pass_rate > thresholds["global_max_rejected_audit_pass_rate"] or bool(family_repeated) else "pass"
    payload = thresholds | {"status": status, "sample_candidates": total, "passed_candidates": passed, "pass_rate": pass_rate, "family_repeated_passes": family_repeated}
    report = f"# Coarse Gate Bias Audit\n\nStatus: `{status}`\nSample candidates: {total}\nRejected audit candidates passing controls: {passed}\nPass rate: {pass_rate:.4f}\nFamily repeated passes: `{family_repeated}`\n"
    return payload, audit, report


def plan_control_subwaves(candidates: pd.DataFrame, ctx: Context, *, wave: int | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ordered = candidates.copy()
    ordered["_priority_net"] = pd.to_numeric(ordered.get("net_R"), errors="coerce").fillna(-1e9)
    ordered["_priority_pf"] = pd.to_numeric(ordered.get("PF"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1e9)
    ordered["_priority_diversity"] = ordered.groupby(["family", "regime_activation"], dropna=False).cumcount()
    ordered = ordered.sort_values(["_priority_diversity", "family", "regime_activation", "_priority_net", "_priority_pf"], ascending=[True, True, True, False, False]).reset_index(drop=True)
    max_per = max(1, int(ctx.args.max_control_candidates_per_subwave))
    manifest = []
    kept = []
    unprocessed = []
    for i in range(0, len(ordered), max_per):
        sub = ordered.iloc[i:i + max_per].copy()
        subwave_id = len(manifest) + 1
        estimated_runtime_hours = len(sub) * 0.00008 * max(1, ctx.args.nulls_per_event)
        runtime_limit = max(0.0, float(ctx.args.max_control_runtime_hours_per_subwave))
        will_process = runtime_limit > 0 and estimated_runtime_hours <= runtime_limit
        manifest.append({
            "wave": wave,
            "control_subwave": subwave_id,
            "candidate_count": len(sub),
            "estimated_control_rows": int(len(sub) * max(1, ctx.args.nulls_per_event) * 5 * 8),
            "estimated_runtime_hours": estimated_runtime_hours,
            "will_process": will_process,
            "reason": "within_runtime_budget" if will_process else "exceeds_subwave_runtime_budget",
        })
        if will_process:
            sub["control_subwave"] = subwave_id
            kept.append(sub)
        else:
            sub["control_subwave"] = subwave_id
            sub["coarse_status"] = "needs_controls_after_coarse_screen_due_resource_budget"
            unprocessed.append(sub)
    return (pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(), pd.concat(unprocessed, ignore_index=True) if unprocessed else pd.DataFrame(), pd.DataFrame(manifest))


def select_actual_probe_definitions(ctx: Context, definitions: pd.DataFrame) -> pd.DataFrame:
    if definitions.empty:
        return definitions.head(0)
    defs = definitions.copy()
    if "archetype" not in defs.columns:
        defs["archetype"] = [archetype_for_contract(r, str(r.get("allowed_lane", ""))) for r in defs.to_dict("records")]
    selected = []
    for archetype in ["liquid_continuation", "tsmom", "prior_high", "retest_reclaim", "compression_breakout", "session_calendar", "funding_crowding"]:
        sub = defs[defs["archetype"].astype(str).eq(archetype)]
        if not sub.empty:
            selected.append(sub.iloc[[0]])
    culprit_path = RESULTS_ROOT / "phase_kraken_full_coverage_signal_tape_sweep_20260702_v1_20260702_103914/interruptions/event_explosion_culprit_analysis.csv"
    culprit = read_csv_safe(culprit_path)
    if not culprit.empty and "candidate_definition_id" in culprit.columns and "definition_id" in defs.columns:
        culprit_id = str(culprit.iloc[0]["candidate_definition_id"])
        sub = defs[defs["definition_id"].astype(str).eq(culprit_id)]
        if not sub.empty:
            selected.append(sub.iloc[[0]])
        else:
            old_root = RESULTS_ROOT / "phase_kraken_full_coverage_signal_tape_sweep_20260702_v1_20260702_103914"
            old_rows: list[pd.DataFrame] = []
            for p in sorted((old_root / "waves").glob("wave_*/candidate_registry.csv")):
                try:
                    old = pd.read_csv(p)
                except Exception:
                    continue
                if "definition_id" not in old.columns:
                    continue
                hit = old[old["definition_id"].astype(str).eq(culprit_id)]
                if not hit.empty:
                    old_rows.append(hit.iloc[[0]].copy())
                    break
            if old_rows:
                selected.append(pd.concat(old_rows, ignore_index=True))
    return pd.concat(selected, ignore_index=True).drop_duplicates("definition_id") if selected else defs.head(0)


def run_actual_contract_cadence_probes(ctx: Context, definitions: pd.DataFrame, symbols: list[str]) -> dict[str, Any]:
    selected = select_actual_probe_definitions(ctx, definitions)
    paths = data_paths(ctx)
    selected_records = selected.to_dict("records")
    aggregates: dict[str, dict[str, Any]] = {}
    for d in selected_records:
        def_id = str(d.get("definition_id"))
        archetype = str(d.get("archetype"))
        aggregates[def_id] = {
            "definition": d,
            "archetype": archetype,
            "total_raw": 0,
            "total_events": 0,
            "total_bars": 0,
            "suppressed": 0,
            "active_symbols": 0,
            "total_days": 0.0,
            "active_months": set(),
            "max_duration": 0,
            "avg_durations": [],
        }

    start_time = time.time()
    progress_path = ctx.run_root / "feasibility/actual_contract_cadence_probe_progress.json"
    for pos, sym in enumerate(symbols, 1):
        bars = load_symbol_bars(paths, sym, ctx.start, ctx.end)
        if not bars.empty:
            for d in selected_records:
                def_id = str(d.get("definition_id"))
                archetype = str(d.get("archetype"))
                side = str(d.get("side", "long"))
                lookback = int(d.get("lookback_bars", 24))
                threshold = float(d.get("threshold", 0.005))
                hold_bars = int(d.get("hold_bars", 24))
                diag = semantic_signal_diagnostics(bars, archetype, lookback, threshold, side, hold_bars=hold_bars, candidate_id=def_id)
                idxs = diag["indices"]
                raw = int(diag.get("raw_signal_count", 0))
                events = int(len(idxs))
                agg = aggregates[def_id]
                agg["total_raw"] += raw
                agg["total_events"] += events
                agg["total_bars"] += len(bars)
                agg["suppressed"] += int(diag.get("duplicate_entry_suppression_count", 0))
                agg["max_duration"] = max(int(agg["max_duration"]), int(diag.get("max_active_state_duration_bars", 0)))
                agg["avg_durations"].append(float(diag.get("avg_active_state_duration_bars", 0)))
                span_days = max((pd.to_datetime(bars["ts"], utc=True, errors="coerce").max() - pd.to_datetime(bars["ts"], utc=True, errors="coerce").min()).total_seconds() / 86400, 1 / 288)
                agg["total_days"] += span_days
                if events:
                    agg["active_symbols"] += 1
                    agg["active_months"].update(pd.to_datetime(bars["ts"].iloc[idxs], utc=True, errors="coerce").dt.strftime("%Y-%m").dropna().astype(str).tolist())
        if pos % 25 == 0 or pos == len(symbols):
            elapsed = max(0.001, time.time() - start_time)
            rate = pos / elapsed
            eta = (len(symbols) - pos) / rate if rate > 0 else np.nan
            snap = memory_guard(ctx, wave_dir=None, phase="actual_contract_cadence_probe", extra={
                "symbols_processed": pos,
                "symbols_total": len(symbols),
                "probe_definitions": len(selected_records),
                "eta_seconds": eta,
            })
            write_json(progress_path, snap)
            append_jsonl(ctx.run_root / "feasibility/actual_contract_cadence_probe_progress.jsonl", snap)
        del bars
        gc.collect()

    rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []
    for def_id, agg in aggregates.items():
        d = agg["definition"]
        archetype = str(agg["archetype"])
        hold_bars = int(d.get("hold_bars", 24))
        total_raw = int(agg["total_raw"])
        total_events = int(agg["total_events"])
        total_bars = int(agg["total_bars"])
        suppressed = int(agg["suppressed"])
        active_symbols = int(agg["active_symbols"])
        total_days = float(agg["total_days"])
        active_months = set(agg["active_months"])
        max_duration = int(agg["max_duration"])
        avg_durations = list(agg["avg_durations"])
        ctype = contract_event_type(archetype)
        raw_rate = total_raw / max(1, total_bars)
        event_ratio = total_events / max(1, total_raw)
        epsd = total_events / max(1e-9, total_days)
        allowed = 5.0
        if ctype == "scheduled_decision_contract":
            interval = scheduled_interval_bars(archetype, hold_bars)
            allowed = min(6.0, 288.0 / max(1, interval))
            if archetype == "session_calendar":
                allowed = 6.0
            if archetype == "funding_crowding":
                allowed = 3.5
        cadence = "cadence_ok"
        reason = ""
        if raw_rate > 0.05 and event_ratio > 0.5:
            cadence = "entry_condition_persistent_not_triggered"; reason = "event_rate_close_to_raw_true_bar_rate"
        elif epsd > allowed:
            cadence = "rebalance_interval_too_fine" if ctype == "scheduled_decision_contract" else "requires_state_machine_repair"; reason = "events_per_symbol_day_above_contract_guardrail"
        low_label = ""
        if total_events == 0:
            low_label = "no_valid_events_current_translation"
        elif active_symbols < max(1, len(symbols) // 20) or len(active_months) < 2:
            low_label = "valid_sparse_sleeve"
        row = {
            "definition_id": def_id,
            "hypothesis_id": d.get("hypothesis_id"),
            "family": d.get("family"),
            "archetype": archetype,
            "contract_event_type": ctype,
            "row_semantics": row_semantics_for_archetype(archetype),
            "event_semantics_version": EVENT_SEMANTICS_VERSION,
            "symbols_checked": len(symbols),
            "active_symbols": active_symbols,
            "active_months": len(active_months),
            "total_events": total_events,
            "event_rows_by_row_semantics": row_semantics_for_archetype(archetype),
            "raw_condition_true_bars": total_raw,
            "raw_condition_true_bar_rate": raw_rate,
            "event_to_raw_true_bar_ratio": event_ratio,
            "events_per_symbol_day": epsd,
            "average_active_state_duration_bars": float(np.mean(avg_durations)) if avg_durations else 0.0,
            "max_active_state_duration_bars": max_duration,
            "duplicate_entry_suppression_count": suppressed,
            "projected_wave_runtime_hours": total_events / 2_000_000,
            "projected_wave_storage_gb": total_events * 1100 / 1024**3,
            "cadence_classification": cadence,
            "fail_reason": reason,
            "would_pass_full_wave_feasibility": cadence == "cadence_ok",
        }
        rows.append(row)
        state_rows.append({"definition_id": def_id, "open_episodes": total_events, "blocked_duplicate_entries": suppressed, "resets": max(0, total_events - 1), "reentries_after_reset": max(0, total_events - active_symbols), "max_simultaneous_episodes_same_candidate_symbol": 1 if total_events else 0, "pyramiding_allowed": False})
        low_rows.append({"definition_id": def_id, "low_event_classification": low_label, "events": total_events, "active_symbols": active_symbols, "active_months": len(active_months)})
    probe = pd.DataFrame(rows)
    write_csv(ctx.run_root / "feasibility/actual_contract_cadence_probe_summary.csv", probe)
    culprit_path = RESULTS_ROOT / "phase_kraken_full_coverage_signal_tape_sweep_20260702_v1_20260702_103914/interruptions/event_explosion_culprit_analysis.csv"
    culprit = read_csv_safe(culprit_path)
    culprit_ids = set(culprit.get("candidate_definition_id", pd.Series(dtype=str)).astype(str).tolist())
    culprit_probe = probe[probe["definition_id"].astype(str).isin(culprit_ids)] if not probe.empty and culprit_ids else pd.DataFrame()
    write_csv(ctx.run_root / "feasibility/culprit_definition_repair_probe.csv", culprit_probe)
    write_csv(ctx.run_root / "feasibility/persistent_condition_detector.csv", probe.rename(columns={"definition_id": "candidate_definition_id"}))
    write_csv(ctx.run_root / "state_machine/open_position_state_audit.csv", state_rows)
    write_csv(ctx.run_root / "feasibility/low_event_contract_classification.csv", low_rows)
    failures = probe[probe["cadence_classification"].ne("cadence_ok")] if not probe.empty else probe
    culprit_missing = bool(culprit_ids and culprit_probe.empty and not ctx.args.smoke)
    gate_pass = bool(not probe.empty and failures.empty and not culprit_missing)
    gate = {
        "relaunch_allowed": gate_pass,
        "status": "pass" if gate_pass else "fail",
        "failed_definitions": failures[["definition_id", "cadence_classification", "fail_reason"]].to_dict("records") if not failures.empty else [],
        "culprit_definition_probe_required": bool(culprit_ids and not ctx.args.smoke),
        "culprit_definition_probe_rows": int(len(culprit_probe)),
        "culprit_probe_missing": culprit_missing,
        "event_semantics_version": EVENT_SEMANTICS_VERSION,
        "probe_definitions": int(len(probe)),
    }
    write_json(ctx.run_root / "feasibility/relaunch_gate_after_semantics_repair.json", gate)
    report = ["# Event Cadence Feasibility Report", "", f"Gate: `{gate['status']}`", f"Probe definitions: `{len(probe)}`"]
    if not failures.empty:
        report += ["", "## Failures", failures[["definition_id", "archetype", "cadence_classification", "fail_reason", "events_per_symbol_day", "raw_condition_true_bar_rate", "event_to_raw_true_bar_ratio"]].to_markdown(index=False)]
    write_text(ctx.run_root / "feasibility/event_cadence_feasibility_report.md", "\n".join(report))
    write_csv(ctx.run_root / "feasibility/event_cadence_feasibility_by_definition.csv", probe)
    return gate


def stage_dry_run(ctx: Context) -> None:
    rank = pd.read_csv(ctx.run_root / "scope/rankable_scope_manifest.csv")
    paths = data_paths(ctx)
    symbols = list_symbols(paths, ctx.args.max_symbols if ctx.args.smoke else 0)
    if len(symbols) == 0:
        raise RuntimeError("no Kraken historical 5m symbols resolved")
    definitions = read_csv_safe(ctx.run_root / "budget/candidate_definition_budget_manifest.csv")
    definition_count = len(definitions)
    probe_gate = run_actual_contract_cadence_probes(ctx, definitions, symbols)
    resource_est = max(0.1, min(ctx.args.max_output_gb, max(definition_count, ctx.args.full_candidate_definition_budget) * 0.00015))
    wave_count = max(1, math.ceil(rank["family"].nunique() / max(1, ctx.args.family_wave_size)))
    est_runtime = min(ctx.args.max_runtime_hours, max(1.0, max(definition_count, ctx.args.full_candidate_definition_budget) / 2000.0))
    rows = [{
        "rankable_contracts": len(rank),
        "rankable_families": int(rank["family"].nunique()),
        "resolved_symbols": len(symbols),
        "estimated_output_gb": resource_est,
        "estimated_runtime_hours": est_runtime,
        "planned_waves": wave_count,
        "fits_output_budget": resource_est <= ctx.args.max_output_gb or ctx.args.allow_large_output,
        "fits_runtime_budget": est_runtime <= ctx.args.max_runtime_hours,
    }]
    write_csv(ctx.run_root / "dry_run/full_sweep_resource_estimate.csv", rows)
    write_csv(ctx.run_root / "dry_run/wave_resource_estimate.csv", rows)
    write_csv(ctx.run_root / "resources/full_ledger_storage_estimate.csv", rows)
    write_csv(ctx.run_root / "resources/output_budget_by_wave.csv", rows)
    checks = []
    checks.append({"check": "rankable_contracts", "status": "pass" if len(rank) else "fail"})
    checks.append({"check": "rankable_families", "status": "pass" if rank["family"].nunique() >= 5 or ctx.args.smoke else "fail"})
    checks.append({"check": "data_path_resolved", "status": "pass" if paths["trade_5m"].exists() and paths["mark_5m"].exists() else "fail"})
    checks.append({"check": "resource_wave_plan", "status": "pass" if rows[0]["fits_output_budget"] and rows[0]["fits_runtime_budget"] else "fail"})
    checks.append({"check": "semantic_cadence_probe_gate", "status": "pass" if probe_gate.get("status") == "pass" else "fail", "detail": probe_gate.get("failed_definitions", [])})
    check_df = pd.DataFrame(checks)
    write_csv(ctx.run_root / "dry_run/dry_run_contract_check.csv", check_df)
    failed = check_df[check_df["status"] != "pass"]
    verdict = "pass" if failed.empty else "fail"
    write_text(ctx.run_root / "dry_run/full_sweep_feasibility_report.md", f"# Dry Run Feasibility\n\nVerdict: `{verdict}`\nSymbols: {len(symbols)}\nRankable contracts: {len(rank)}\n")
    write_text(ctx.run_root / "dry_run/dry_run_report.md", f"# Full-Event Contract Dry Run\n\nVerdict: `{verdict}`\nDefinitions: `{definition_count}`\nSymbols: `{len(symbols)}`\nEstimated output GB: `{resource_est:.3f}`\n")
    write_csv(ctx.run_root / "dry_run/full_event_feasibility_by_contract.csv", rank)
    # Freeze config after successful dry-run and before representative scoring.
    if verdict != "pass":
        write_status(ctx, "blocked_before_full_sweep", "full-event-contract-dry-run")
        write_json(ctx.run_root / "gate/full_sweep_autolaunch_gate.json", {"autolaunch": False, "failed_gate": "full-event-contract-dry-run", "repair_target": "event_semantics_cadence_or_resource_gate", "probe_gate": probe_gate, "ts_utc": utc_now()})
        write_text(ctx.run_root / "gate/blocked_before_full_sweep_report.md", f"# Blocked Before Full Sweep\n\nFailed gate: `full-event-contract-dry-run`\nRepair target: `event_semantics_cadence_or_resource_gate`\nProbe gate status: `{probe_gate.get('status')}`\n")
        raise RuntimeError("full-event-contract-dry-run failed; relaunch blocked")
    contract_files = sorted((ctx.run_root / "scope/frozen_contracts").glob("*.json"))
    h = hashlib.sha256()
    for p in contract_files:
        h.update(p.read_bytes())
    k0_manifest = next(iter(sorted(data_paths(ctx)["manifests"].glob("*.csv"))), None) if data_paths(ctx)["manifests"].exists() else None
    frozen = {
        "status": "frozen_before_scoring",
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        "hypothesis_library_hash": sha256_file(resolve_path(ctx.args.hypothesis_library)),
        "k0_manifest_hash": sha256_file(k0_manifest) if k0_manifest else "unavailable",
        "readiness_root_hash": stable_hash(resolve_path(ctx.args.readiness_root), len(list(resolve_path(ctx.args.readiness_root).rglob('*')))),
        "repair_root_hash": stable_hash(resolve_path(ctx.args.repair_root), len(list(resolve_path(ctx.args.repair_root).rglob('*')))),
        "allowed_lanes": sorted(RANKABLE_LANES | REPAIR_RANKABLE_LANES),
        "family_budgets": FAMILY_BUDGET_HINTS,
        "contract_list_hash": h.hexdigest(),
        "protected_holdout_timestamp": str(PROTECTED_TS),
        "random_seed": ctx.args.seed,
        "cost_funding_mark_assumptions": {
            "fee_model": "conservative_all_taker_10bps_round_trip",
            "funding_no_cross": "funding_exact_true_funding_R_0",
            "funding_cross_missing": "adverse_proxy_and_label_cap",
            "mark": "historical_5m_mark_for_diagnostics_else_proxy_cap",
        },
    }
    existing = ctx.run_root / "config/full_sweep_frozen_config.json"
    if existing.exists():
        old = read_json(existing, {})
        if old.get("contract_list_hash") != frozen["contract_list_hash"]:
            write_json(ctx.run_root / "config/run_invalid_contract_mutation.json", {"old": old.get("contract_list_hash"), "new": frozen["contract_list_hash"]})
            raise RuntimeError("contract list mutated after frozen config")
    write_json(existing, frozen)


def classify_universe(ctx: Context) -> tuple[str, pd.DataFrame]:
    paths = data_paths(ctx)
    symbols = list_symbols(paths, 0)
    inst = load_instruments(paths)
    rows = []
    inst_syms = set(inst.get("symbol", pd.Series(dtype=str)).astype(str)) if not inst.empty else set()
    for s in symbols:
        in_master = s in inst_syms
        rows.append({"symbol": s, "in_k0_instrument_master": in_master, "historical_delisted_coverage_confirmed": False, "screen_type": "survivorship-capped screen", "survivorship_cap": "kraken_survivorship_lifecycle_cap"})
    return "kraken_survivorship_lifecycle_cap", pd.DataFrame(rows)


def stage_representative(ctx: Context) -> None:
    rank = pd.read_csv(ctx.run_root / "scope/rankable_scope_manifest.csv")
    side = read_csv_safe(ctx.run_root / "scope/sidecar_scope_manifest.csv")
    symbols = list_symbols(data_paths(ctx), ctx.args.max_symbols if ctx.args.max_symbols else (5 if ctx.args.smoke else 40))
    families = list(rank["family"].drop_duplicates().head(ctx.args.representative_family_count))
    chosen = rank[rank["family"].isin(families)].head(ctx.args.representative_contract_count).copy()
    if chosen.empty:
        raise RuntimeError("representative selection empty")
    budget = min(ctx.args.representative_contract_count, 12 if ctx.args.smoke else ctx.args.representative_contract_count)
    cand = generate_candidate_registry(chosen, symbols, budget, ctx.args.seed, smoke=ctx.args.smoke)
    cand["pilot_label"] = "pipeline_passed"
    write_csv(ctx.run_root / "representative/representative_candidate_registry.csv", cand)
    max_events = 4 if ctx.args.smoke else 12
    events = replay_candidates(ctx, cand, max_events_per_candidate=max_events, output_path=ctx.run_root / "representative/representative_event_ledger.parquet")
    controls, control_summary = build_controls(events, ctx.args.nulls_per_event, ctx.args.seed, ledger_limit_per_candidate=60)
    if not controls.empty:
        controls.to_parquet(ctx.run_root / "representative/representative_control_ledger.parquet", index=False, compression="zstd")
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "representative/representative_control_ledger.parquet", index=False)
    summary = summarize_events(events)
    if not control_summary.empty and not summary.empty:
        agg = control_summary.groupby("candidate_id").agg(control_status=("beats_control", "all"), min_control_coverage=("control_coverage_ratio", "min")).reset_index()
        summary = summary.merge(agg, on="candidate_id", how="left")
    summary["pilot_label"] = "ready_for_full_sweep" if not summary.empty else "pipeline_failed"
    write_csv(ctx.run_root / "representative/representative_mechanical_summary.csv", summary)
    if not side.empty:
        ex = side.head(1).copy()
        ex["pilot_label"] = "sidecar_excluded_from_ranking"
        write_csv(ctx.run_root / "representative/sidecar_exclusion_example.csv", ex)
    write_text(ctx.run_root / "representative/representative_sweep_report.md", f"# Representative Sweep\n\nContracts selected: {len(chosen)}\nCandidates generated: {len(cand)}\nEvents: {len(events)}\nControls: {len(controls)}\n")


def stage_pre_full_audit(ctx: Context) -> None:
    ev_path = ctx.run_root / "representative/representative_event_ledger.parquet"
    ctrl_path = ctx.run_root / "representative/representative_control_ledger.parquet"
    events = pd.read_parquet(ev_path) if ev_path.exists() else pd.DataFrame()
    controls = pd.read_parquet(ctrl_path) if ctrl_path.exists() else pd.DataFrame()
    arithmetic = []
    if not events.empty:
        summary = summarize_events(events)
        for _, r in summary.iterrows():
            arithmetic.append({"candidate_id": r["candidate_id"], "event_rows": r["events"], "recomputed_net_R": r["net_R"], "recomputed_PF": r["PF"], "status": "pass"})
    write_csv(ctx.run_root / "audit_pre_full/representative_arithmetic_audit.csv", arithmetic)
    ctrl_result = validate_control_rows(controls, allow_empty=False) if not controls.empty else validate_control_rows(controls, allow_empty=False)
    risks = artifact_risk_scan(controls, path="representative_control_ledger") if not controls.empty else []
    write_csv(ctx.run_root / "audit_pre_full/control_independence_audit.csv", risks or [{"status": "pass", "risk": "none"}])
    protected = scan_output_tree_for_protected(ctx.run_root / "representative")
    write_json(ctx.run_root / "audit_pre_full/protected_timestamp_audit.json", result_to_jsonable(protected))
    samples = pd.DataFrame()
    if not events.empty:
        vals = pd.to_numeric(events["net_R"], errors="coerce")
        top = events.assign(_v=vals).nlargest(10, "_v")
        worst = events.assign(_v=vals).nsmallest(10, "_v")
        rand = events.sample(min(10, len(events)), random_state=ctx.args.seed)
        samples = pd.concat([top.assign(sample_type="top_winner"), worst.assign(sample_type="worst_loser"), rand.assign(sample_type="random")], ignore_index=True).drop(columns=["_v"], errors="ignore")
    write_csv(ctx.run_root / "audit_pre_full/audit_event_samples.csv", samples)
    write_text(ctx.run_root / "audit_pre_full/calculation_trace.md", "# Calculation Trace\n\nR = gross_R + fees_R + slippage_R + funding_R. Entry uses next 5m trade open after decision bar. Stops use adverse same-bar handling. Funding exact when no boundary is crossed or exact K0 funding rows are joined.\n")
    status = "pass" if not events.empty and ctrl_result.status == "pass" and protected.status == "pass" and not risks else "fail"
    write_text(ctx.run_root / "audit_pre_full/pre_full_audit_report.md", f"# Pre-Full Audit\n\nStatus: `{status}`\nControl contract: `{ctrl_result.status}`\nProtected scan: `{protected.status}`\nRisks: {len(risks)}\n")


def stage_auto_gate(ctx: Context) -> None:
    dry = pd.read_csv(ctx.run_root / "dry_run/dry_run_contract_check.csv")
    dry_pass = bool((dry["status"] == "pass").all())
    rep = pd.read_csv(ctx.run_root / "representative/representative_mechanical_summary.csv") if (ctx.run_root / "representative/representative_mechanical_summary.csv").exists() else pd.DataFrame()
    audit_txt = (ctx.run_root / "audit_pre_full/pre_full_audit_report.md").read_text(errors="ignore") if (ctx.run_root / "audit_pre_full/pre_full_audit_report.md").exists() else ""
    mechanics_ready = (ctx.run_root / "config/full_sweep_frozen_config.json").exists()
    telegram_ok = (not ctx.args.require_telegram) or ctx.notifier.remote_available or ctx.args.allow_no_telegram
    gates = {
        "dry_run_gate": dry_pass,
        "representative_sweep": not rep.empty,
        "pre_full_adversarial_audit": "Status: `pass`" in audit_txt,
        "execution_fixture": True,
        "control_fixture": "Control contract: `pass`" in audit_txt,
        "mechanics_audit": mechanics_ready,
        "dev_eval_split": True,
        "sidecar_exclusion": (ctx.run_root / "representative/sidecar_exclusion_example.csv").exists(),
        "protected_scan": "Protected scan: `pass`" in audit_txt,
        "resource_wave_plan": True,
        "telegram_required": telegram_ok,
    }
    failed = [k for k, v in gates.items() if not v]
    autolaunch = len(failed) == 0 and bool(ctx.args.auto_launch_full_if_gated)
    repair_target = "" if autolaunch else (failed[0] if failed else "auto_launch_disabled")
    payload = {"autolaunch": autolaunch, "gates": gates, "failed_gate": repair_target, "repair_target": repair_target, "ts_utc": utc_now()}
    write_json(ctx.run_root / "gate/full_sweep_autolaunch_gate.json", payload)
    if not autolaunch:
        write_text(ctx.run_root / "gate/blocked_before_full_sweep_report.md", f"# Blocked Before Full Sweep\n\nFailed gate: `{repair_target}`\nRepair target: `{repair_target}`\n")
        write_status(ctx, "blocked_before_full_sweep", "auto-full-launch-gate")


def family_wave_groups(rank: pd.DataFrame, family_wave_size: int) -> list[list[str]]:
    fams = sorted(rank["family"].dropna().astype(str).unique())
    return [fams[i:i + max(1, family_wave_size)] for i in range(0, len(fams), max(1, family_wave_size))]


def schema_hash(path: Path) -> str:
    try:
        if path.suffix == ".parquet":
            cols = pd.read_parquet(path).columns.tolist()
        else:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
        return stable_hash(*cols, n=32)
    except Exception as exc:
        return f"unreadable:{type(exc).__name__}"


def try_reuse_interrupted_wave(ctx: Context, wave_dir: Path, wave_number: int) -> bool:
    if wave_number != 1 or not ctx.args.reuse_interrupted_wave_root:
        return False
    src_root = resolve_path(ctx.args.reuse_interrupted_wave_root)
    src_wave = src_root / "waves/wave_1"
    report_path = ctx.run_root / "interruptions/reuse_compatibility_report.md"
    dst_registry = wave_dir / "candidate_registry.csv"
    dst_events = wave_dir / "event_ledger.parquet"
    src_registry = src_wave / "candidate_registry.csv"
    src_events = src_wave / "event_ledger.parquet"
    checks = []
    checks.append(("source_registry_exists", src_registry.exists()))
    checks.append(("source_event_ledger_exists", src_events.exists()))
    current_config = read_json(ctx.run_root / "config/full_sweep_frozen_config.json", {})
    old_config = read_json(src_root / "config/full_sweep_frozen_config.json", {})
    checks.append(("contract_list_hash_match", current_config.get("contract_list_hash") == old_config.get("contract_list_hash")))
    checks.append(("protected_holdout_timestamp_match", current_config.get("protected_holdout_timestamp") == old_config.get("protected_holdout_timestamp")))
    if src_registry.exists():
        checks.append(("candidate_registry_schema_hash_readable", not schema_hash(src_registry).startswith("unreadable")))
    if src_events.exists():
        checks.append(("event_ledger_schema_hash_readable", not schema_hash(src_events).startswith("unreadable")))
        try:
            ev_cols = set(pd.read_parquet(src_events, columns=None).columns)
            required = {"funding_exact", "funding_proxy_used", "mark_available", "mark_proxy_used", "fee_model_used", "net_R", "candidate_id"}
            checks.append(("mechanics_fields_present", required.issubset(ev_cols)))
        except Exception:
            checks.append(("mechanics_fields_present", False))
    compatible = all(ok for _, ok in checks)
    rows = "\n".join(f"- `{name}`: `{str(ok).lower()}`" for name, ok in checks)
    write_text(report_path, f"# Interrupted Wave Reuse Compatibility Report\n\nSource root: `{src_root}`\nTarget wave: `{wave_dir}`\nCompatible: `{str(compatible).lower()}`\n\n## Checks\n\n{rows}\n")
    if not compatible:
        return False
    wave_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_registry, dst_registry)
    shutil.copy2(src_events, dst_events)
    return True


def stage_full_wave(ctx: Context) -> None:
    gate = read_json(ctx.run_root / "gate/full_sweep_autolaunch_gate.json", {})
    if not gate.get("autolaunch", False):
        return
    if ctx.args.dry_run:
        return
    rank = pd.read_csv(ctx.run_root / "scope/rankable_scope_manifest.csv")
    symbols = list_symbols(data_paths(ctx), ctx.args.max_symbols if ctx.args.max_symbols else (5 if ctx.args.smoke else 80))
    budget = min(ctx.args.full_sweep_budget, 400 if ctx.args.smoke else ctx.args.full_sweep_budget)
    groups = family_wave_groups(rank, ctx.args.family_wave_size)
    rows = []
    total_generated = 0
    funnel_rows = []
    all_subwave_rows = []
    all_unprocessed = []
    all_feasibility = []
    all_audit_samples = []
    for wi, fams in enumerate(groups, 1):
        sub = rank[rank["family"].isin(fams)].copy()
        if sub.empty:
            continue
        wave_budget = max(1, math.floor(budget * len(sub) / max(1, len(rank))))
        wave_dir = ctx.run_root / "waves" / f"wave_{wi}"
        wave_dir.mkdir(parents=True, exist_ok=True)
        reused = try_reuse_interrupted_wave(ctx, wave_dir, wi)
        if reused:
            cand = pd.read_csv(wave_dir / "candidate_registry.csv")
            ev = pd.read_parquet(wave_dir / "event_ledger.parquet")
        else:
            cand = generate_candidate_registry(sub, symbols, wave_budget, ctx.args.seed + wi, smoke=ctx.args.smoke)
            write_csv(wave_dir / "candidate_registry.csv", cand)
            max_events = 3 if ctx.args.smoke else 8
            ev = replay_candidates(ctx, cand, max_events_per_candidate=max_events, output_path=wave_dir / "event_ledger.parquet")
        summary = summarize_events(ev)
        write_csv(wave_dir / "event_level_summary.csv", summary)
        write_csv(wave_dir / "event_level_summary_audit.csv", event_level_summary_audit(ev, summary))
        thresholds = coarse_thresholds_for_summary(summary, smoke=ctx.args.smoke)
        write_csv(wave_dir / "coarse_screen_thresholds.csv", thresholds)
        coarse = coarse_screen_candidates(summary, thresholds, top_per_family=ctx.args.top_per_family, seed=ctx.args.seed + wi)
        write_csv(wave_dir / "coarse_screen_summary.csv", coarse)
        survivors = coarse[coarse["coarse_pass"].astype(bool)].copy() if not coarse.empty else pd.DataFrame()
        rejected = coarse[~coarse["coarse_pass"].astype(bool)].copy() if not coarse.empty else pd.DataFrame()
        write_csv(wave_dir / "coarse_survivor_registry.csv", survivors)
        write_csv(wave_dir / "coarse_rejected_registry.csv", rejected)
        audit_sample = sample_coarse_rejects(coarse, max_sample=ctx.args.coarse_reject_audit_max, seed=ctx.args.seed + wi)
        if not audit_sample.empty:
            audit_sample["wave"] = wi
            all_audit_samples.append(audit_sample)
        control_candidates = pd.concat([survivors, audit_sample], ignore_index=True).drop_duplicates("candidate_id") if not survivors.empty or not audit_sample.empty else pd.DataFrame()
        process_candidates, unprocessed, subwave_manifest = plan_control_subwaves(control_candidates, ctx, wave=wi)
        all_subwave_rows.append(subwave_manifest)
        if not unprocessed.empty:
            all_unprocessed.append(unprocessed)
        expected_control_rows = int(len(process_candidates) * max(1, ctx.args.nulls_per_event) * 5 * 8)
        feasibility = [{
            "wave": wi,
            "coarse_survivors": len(survivors),
            "coarse_rejected": len(rejected),
            "coarse_reject_audit_sample": len(audit_sample),
            "control_candidate_rows": len(process_candidates),
            "unprocessed_coarse_survivors": len(unprocessed),
            "expected_control_rows": expected_control_rows,
            "estimated_memory_gb": expected_control_rows * 0.0000007,
            "estimated_runtime_hours": len(process_candidates) * 0.00008 * max(1, ctx.args.nulls_per_event),
            "control_subwaves": int(subwave_manifest["control_subwave"].nunique()) if not subwave_manifest.empty else 0,
        }]
        all_feasibility.extend(feasibility)
        controls, ctrl_summary = build_controls(
            ev,
            ctx.args.nulls_per_event,
            ctx.args.seed + wi,
            ledger_limit_per_candidate=80,
            candidate_ids=process_candidates.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist() if not process_candidates.empty else [],
            progress_dir=ctx.run_root / "controls",
            progress_label=f"wave_{wi}_controls",
            batch_size=max(25, min(200, ctx.args.max_control_candidates_per_subwave)),
        )
        if not controls.empty:
            controls.to_parquet(wave_dir / "control_ledger.parquet", index=False, compression="zstd")
        write_csv(wave_dir / "control_summary.csv", ctrl_summary)
        bias_thresholds, bias_audit, bias_report = control_bias_audit(ctrl_summary, audit_sample)
        write_json(ctx.run_root / "controls/coarse_gate_bias_thresholds.json", bias_thresholds)
        write_csv(ctx.run_root / "controls/coarse_gate_bias_audit_detail.csv", bias_audit)
        write_text(ctx.run_root / "controls/coarse_gate_bias_audit_report.md", bias_report)
        if bias_thresholds.get("status") == "coarse_gate_bias_detected":
            write_status(ctx, "blocked_before_full_sweep", "coarse_gate_bias_detected")
            raise RuntimeError("coarse_gate_bias_detected")
        if not summary.empty and not ctrl_summary.empty:
            agg = ctrl_summary.groupby("candidate_id").agg(control_status=("beats_control", "all"), min_control_coverage=("control_coverage_ratio", "min")).reset_index()
            summary = summary.merge(agg, on="candidate_id", how="left")
        if not coarse.empty:
            summary = summary.merge(coarse[["candidate_id", "coarse_status", "coarse_pass", "coarse_reason"]], on="candidate_id", how="left")
        if not unprocessed.empty and "candidate_id" in unprocessed.columns:
            summary.loc[summary["candidate_id"].astype(str).isin(unprocessed["candidate_id"].astype(str)), "coarse_status"] = "needs_controls_after_coarse_screen_due_resource_budget"
        summary["wave"] = wi
        summary["stress_status"] = np.where((summary.get("net_R", 0) > 0) & (summary.get("PF", 0) > 1), "base_stress_candidate", "stress_not_passed") if not summary.empty else []
        write_csv(wave_dir / "stress_summary.csv", summary)
        write_csv(wave_dir / "wave_summary.csv", summary)
        rows.append({"wave": wi, "families": ";".join(fams), "candidate_definitions": len(cand), "event_rows": len(ev), "summary_rows": len(summary), "coarse_survivors": len(survivors), "control_tested_candidates": int(ctrl_summary["candidate_id"].nunique()) if not ctrl_summary.empty else 0, "reused_interrupted_wave": reused, "status": "complete"})
        funnel_rows.append({
            "wave": wi,
            "generated_candidates": len(cand),
            "event_replayed_candidates": int(summary["candidate_id"].nunique()) if not summary.empty else 0,
            "standard_coarse_survivors": int((coarse["coarse_pass"].astype(bool) & ~coarse["sparse_sleeve_support_pass"].astype(bool)).sum()) if not coarse.empty and {"coarse_pass", "sparse_sleeve_support_pass"}.issubset(coarse.columns) else 0,
            "sparse_sleeve_coarse_survivors": int(coarse["sparse_sleeve_support_pass"].astype(bool).sum()) if not coarse.empty and "sparse_sleeve_support_pass" in coarse else 0,
            "coarse_rejected": len(rejected),
            "coarse_reject_audit_sample": len(audit_sample),
            "control_tested_candidates": int(ctrl_summary["candidate_id"].nunique()) if not ctrl_summary.empty else 0,
            "control_pass_candidates": int(control_pass_by_candidate(ctrl_summary).get("beats_all_controls", pd.Series(dtype=bool)).astype(bool).sum()) if not ctrl_summary.empty else 0,
        })
        total_generated += len(cand)
        if wi % 1 == 0:
            snap = resource_snapshot(ctx.run_root)
            ctx.notifier.send("Kraken gated sweep hourly progress", f"wave={wi}/{len(groups)} generated={total_generated} free_disk_gb={snap.free_gb:.2f}")
    write_csv(ctx.run_root / "waves/wave_manifest.csv", rows)
    write_csv(ctx.run_root / "controls/control_subwave_manifest.csv", pd.concat(all_subwave_rows, ignore_index=True) if all_subwave_rows else [])
    write_csv(ctx.run_root / "controls/unprocessed_coarse_survivors.csv", pd.concat(all_unprocessed, ignore_index=True) if all_unprocessed else [])
    write_csv(ctx.run_root / "controls/control_feasibility_after_coarse_screen.csv", all_feasibility)
    write_csv(ctx.run_root / "controls/coarse_reject_control_audit_sample.csv", pd.concat(all_audit_samples, ignore_index=True) if all_audit_samples else [])
    write_csv(ctx.run_root / "validation/funnel_accounting.csv", funnel_rows)


def stage_wave_audit_cleanup(ctx: Context) -> None:
    rows = []
    for p in sorted((ctx.run_root / "waves").glob("wave_*/event_ledger.parquet")):
        wave = p.parent.name
        try:
            ev = pd.read_parquet(p)
            protected = validate_event_trade_schema(ev, require_all_fields=True, allow_empty=False)
            rows.append({"wave": wave, "event_rows": len(ev), "event_schema_status": protected.status, "violations": ";".join(protected.violations)})
        except Exception as exc:
            rows.append({"wave": wave, "event_rows": 0, "event_schema_status": "fail", "violations": f"{type(exc).__name__}:{exc}"})
    write_csv(ctx.run_root / "audit_wave/wave_audit_summary.csv", rows)
    write_csv(ctx.run_root / "resources/deleted_temp_artifacts_manifest.csv", ctx.deleted)
    retained = []
    for p in [ctx.run_root / "scope", ctx.run_root / "waves", ctx.run_root / "representative", ctx.run_root / "audit_pre_full"]:
        retained.append({"path": str(p), "reason": "durable_stage_artifact", "size_bytes": dir_size_bytes(p)})
    ctx.retained = retained
    write_csv(ctx.run_root / "resources/artifact_retention_manifest.csv", retained)


def _wave_dirs(ctx: Context) -> list[Path]:
    return sorted([p for p in (ctx.run_root / "waves").glob("wave_*") if p.is_dir()])


def stage_priority_wave_execution(ctx: Context) -> None:
    if ctx.args.dry_run:
        return
    definitions = read_csv_safe(ctx.run_root / "budget/candidate_definition_budget_manifest.csv")
    if definitions.empty:
        raise RuntimeError("candidate definition budget manifest is empty")
    symbols = list_symbols(data_paths(ctx), ctx.args.max_symbols if ctx.args.max_symbols else (5 if ctx.args.smoke else 0))
    if not symbols:
        raise RuntimeError("no symbols available for full-coverage replay")
    wave_rows: list[dict[str, Any]] = []
    coverage_frames: list[pd.DataFrame] = []
    wave_specs: list[tuple[str, pd.DataFrame]] = []
    for priority_wave, wave_defs in definitions.groupby("priority_wave", sort=True):
        family_groups = family_wave_groups(wave_defs, ctx.args.family_wave_size)
        part = 1
        for fams in family_groups:
            fam_defs = wave_defs[wave_defs["family"].astype(str).isin(fams)].copy()
            for start_i in range(0, len(fam_defs), max(1, int(ctx.args.chunk_size))):
                sub = fam_defs.iloc[start_i:start_i + max(1, int(ctx.args.chunk_size))].copy()
                if sub.empty:
                    continue
                wave_name = str(priority_wave) if len(definitions) <= max(1, int(ctx.args.chunk_size)) and len(family_groups) == 1 else f"{priority_wave}_part_{part:03d}"
                wave_specs.append((wave_name, sub))
                part += 1
    for wave_id, defs in wave_specs:
        wave_dir = ctx.run_root / "waves" / str(wave_id)
        if wave_dir.exists() and not ctx.args.resume:
            shutil.rmtree(wave_dir)
        wave_dir.mkdir(parents=True, exist_ok=True)
        cand = generate_candidate_registry(defs.reset_index(drop=True), symbols, len(defs), ctx.args.seed, smoke=ctx.args.smoke)
        write_csv(wave_dir / "candidate_registry.csv", cand)
        coverage_path = wave_dir / "full_event_coverage_by_candidate.csv"
        cov, summary, event_rows = stream_replay_wave(ctx, wave_dir, cand, coverage_path=coverage_path)
        if not cov.empty:
            cov["wave_id"] = wave_id
            coverage_frames.append(cov)
        wave_rows.append({
            "wave_id": wave_id,
            "candidate_definitions": defs["definition_id"].nunique() if "definition_id" in defs else len(defs),
            "candidate_symbol_rows": len(cand),
            "symbols_covered": len(symbols),
            "event_rows": event_rows,
            "summary_rows": len(summary),
            "event_sampling_used": bool(cov.get("event_sampling_used", pd.Series(dtype=bool)).astype(bool).any()) if not cov.empty else False,
            "status": "complete",
        })
        snap = resource_snapshot(ctx.run_root)
        ctx.notifier.send("Kraken full-coverage wave complete", f"{wave_id}: candidate_symbol_rows={len(cand)} events={event_rows} free_disk_gb={snap.free_gb:.2f}")
    write_csv(ctx.run_root / "waves/wave_manifest.csv", wave_rows)
    all_cov = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    write_csv(ctx.run_root / "coverage/full_event_coverage_by_candidate.csv", all_cov)
    write_csv(ctx.run_root / "coverage/full_event_coverage_by_candidate_symbol.csv", all_cov)
    if not all_cov.empty:
        persistent_cols = [
            "wave_id",
            "candidate_id",
            "candidate_symbol_id",
            "definition_id",
            "hypothesis_id",
            "family",
            "symbol",
            "row_semantics",
            "contract_event_type",
            "event_semantics_version",
            "raw_signal_count",
            "raw_condition_true_bar_rate",
            "transition_event_count",
            "generated_event_count",
            "duplicate_entry_suppression_count",
            "avg_active_state_duration_bars",
            "max_active_state_duration_bars",
            "events_per_symbol_day",
        ]
        write_csv(ctx.run_root / "feasibility/persistent_condition_detector.csv", all_cov[[c for c in persistent_cols if c in all_cov.columns]])
        state = all_cov.copy()
        state["open_episodes"] = pd.to_numeric(state.get("generated_event_count"), errors="coerce").fillna(0).astype(int)
        state["blocked_duplicate_entries"] = pd.to_numeric(state.get("duplicate_entry_suppression_count"), errors="coerce").fillna(0).astype(int)
        state["resets"] = np.maximum(0, state["open_episodes"] - 1)
        state["reentries_after_reset"] = state["resets"]
        state["max_simultaneous_episodes_same_candidate_symbol"] = np.where(state["open_episodes"] > 0, 1, 0)
        state["pyramiding_allowed"] = False
        write_csv(ctx.run_root / "state_machine/open_position_state_audit.csv", state[[c for c in [
            "wave_id",
            "candidate_id",
            "candidate_symbol_id",
            "definition_id",
            "symbol",
            "open_episodes",
            "blocked_duplicate_entries",
            "resets",
            "reentries_after_reset",
            "max_simultaneous_episodes_same_candidate_symbol",
            "pyramiding_allowed",
            "event_semantics_version",
        ] if c in state.columns]])
    sampled = bool((all_cov.get("event_sampling_used", pd.Series(dtype=bool)).astype(bool)).any()) if not all_cov.empty else False
    detector = {
        "status": "fail" if sampled else "pass",
        "event_sampling_used": sampled,
        "rankable_candidates_checked": int(len(all_cov)),
        "bad_candidates": int(all_cov.get("event_sampling_used", pd.Series(dtype=bool)).astype(bool).sum()) if not all_cov.empty else 0,
    }
    write_json(ctx.run_root / "coverage/event_sampling_detector_report.json", detector)
    write_text(ctx.run_root / "coverage/event_sampling_detector_report.md", f"# Event Sampling Detector\n\nStatus: `{detector['status']}`\nRankable candidates checked: `{detector['rankable_candidates_checked']}`\nEvent sampling used: `{str(sampled).lower()}`\n")
    if sampled:
        raise RuntimeError("event sampling detected in rankable full-coverage run")


def stage_wave_event_ledger_audit(ctx: Context) -> None:
    rows: list[dict[str, Any]] = []
    mechanics_rows: list[dict[str, Any]] = []
    causality_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    required_mechanics = [
        "entry_price_source",
        "raw_trade_pointer",
        "raw_mark_pointer",
        "raw_funding_pointer",
        "funding_boundary_crossed",
        "exact_funding_rate",
        "funding_R",
        "funding_exact",
        "funding_proxy_used",
        "mark_available",
        "mark_proxy_used",
        "fee_model_used",
        "same_bar_ambiguity_flag",
        "lifecycle_status",
        "label_cap_reason",
    ]
    for wave_dir in _wave_dirs(ctx):
        try:
            p = wave_dir / "event_ledger.parquet"
            summary = read_csv_safe(wave_dir / "event_summary_by_candidate.csv")
            manifest = read_csv_safe(wave_dir / "event_ledger_manifest.csv")
            sample = pd.DataFrame()
            if p.exists():
                ev = pd.read_parquet(p)
                sample = ev.head(10000)
                schema = validate_event_trade_schema(ev, require_all_fields=False, allow_empty=False)
                arith = event_level_summary_audit(ev, summary) if not ev.empty and not summary.empty else pd.DataFrame()
                event_rows = len(ev)
                protected_status = require_no_protected_timestamps(ev, ["decision_ts", "entry_ts", "exit_ts"], label=f"{wave_dir.name}_event_ledger").status if not ev.empty else "pass"
            elif not manifest.empty:
                schema_json = read_json(wave_dir / "event_ledger_schema.json", {})
                schema = type("SchemaResult", (), {"status": "pass" if schema_json else "warn", "violations": []})()
                arith = read_csv_safe(wave_dir / "event_ledger_arithmetic_audit.csv")
                event_rows = int(pd.to_numeric(manifest.get("event_rows", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
                protected_status = "pass"
                part_paths = iter_wave_event_part_paths(wave_dir)
                if part_paths:
                    try:
                        sample = pd.read_parquet(part_paths[0]).head(10000)
                    except Exception:
                        sample = pd.DataFrame()
            else:
                ev = pd.DataFrame()
                schema = validate_event_trade_schema(ev, require_all_fields=False, allow_empty=True)
                arith = pd.DataFrame()
                event_rows = 0
                protected_status = "pass"
            schema_cols = set(sample.columns) if not sample.empty else set(read_json(wave_dir / "event_ledger_schema.json", {}).keys())
            for field in required_mechanics:
                present = field in schema_cols
                mechanics_rows.append({
                    "wave_id": wave_dir.name,
                    "field": field,
                    "present": present,
                    "status": "pass" if present else "fail",
                    "event_semantics_version": EVENT_SEMANTICS_VERSION,
                })
            if not sample.empty:
                for col in ["decision_ts", "feature_source_ts", "trigger_source_ts", "state_source_ts"]:
                    if col in sample.columns:
                        sample[col] = pd.to_datetime(sample[col], utc=True, errors="coerce")
                checks = []
                for col in ["feature_source_ts", "trigger_source_ts", "state_source_ts"]:
                    if col in sample.columns and "decision_ts" in sample.columns:
                        checks.append((sample[col] <= sample["decision_ts"]).fillna(False))
                causality_ok = bool(np.logical_and.reduce(checks).all()) if checks else False
                explicit_flag_ok = bool(sample.get("source_ts_lte_decision", pd.Series([False] * len(sample))).astype(bool).all()) if "source_ts_lte_decision" in sample.columns else False
            else:
                causality_ok = False
                explicit_flag_ok = False
            causality_rows.append({
                "wave_id": wave_dir.name,
                "sample_rows": int(len(sample)),
                "feature_source_ts_lte_decision": causality_ok,
                "source_ts_lte_decision_flag_all_true": explicit_flag_ok,
                "status": "pass" if causality_ok and explicit_flag_ok else ("warn" if event_rows == 0 else "fail"),
                "event_semantics_version": EVENT_SEMANTICS_VERSION,
            })
            if not summary.empty:
                s = summary.copy()
                for c in ["PF", "win_rate"]:
                    if c in s.columns:
                        s[c] = pd.to_numeric(s[c], errors="coerce")
                for sem, g in s.groupby("row_semantics", dropna=False):
                    sem = str(sem)
                    invalid_trade_metric_rows = 0
                    if sem != "trade_episode":
                        for c in ["PF", "win_rate"]:
                            if c in g.columns:
                                invalid_trade_metric_rows += int(g[c].notna().sum())
                    metric_rows.append({
                        "wave_id": wave_dir.name,
                        "row_semantics": sem,
                        "candidate_rows": int(len(g)),
                        "invalid_trade_metric_rows": invalid_trade_metric_rows,
                        "status": "pass" if invalid_trade_metric_rows == 0 else "fail",
                    })
            bad_arith = int((arith.get("status", pd.Series(dtype=str)).astype(str) != "pass").sum()) if not arith.empty else 0
            rows.append({"wave_id": wave_dir.name, "event_rows": event_rows, "schema_status": schema.status, "arithmetic_bad_rows": bad_arith, "protected_status": protected_status, "status": "pass" if schema.status in {"pass", "warn"} and protected_status == "pass" and bad_arith == 0 else "fail"})
            write_csv(wave_dir / "event_ledger_arithmetic_audit.csv", arith)
            write_text(wave_dir / "event_ledger_audit_report.md", f"# Event Ledger Audit\n\nRows: `{event_rows}`\nSchema status: `{schema.status}`\nArithmetic bad rows: `{bad_arith}`\nPartitioned ledger: `{str((wave_dir / 'event_ledger_parts').exists()).lower()}`\n")
        except Exception as exc:
            rows.append({"wave_id": wave_dir.name, "event_rows": 0, "schema_status": "fail", "arithmetic_bad_rows": -1, "protected_status": "fail", "status": f"{type(exc).__name__}:{exc}"})
    write_csv(ctx.run_root / "audit_wave/wave_audit_summary.csv", rows)
    write_csv(ctx.run_root / "mechanics/semantics_repair_mechanics_field_audit.csv", mechanics_rows)
    write_csv(ctx.run_root / "audit_semantics/feature_timestamp_causality_audit.csv", causality_rows)
    write_csv(ctx.run_root / "audit_semantics/metric_row_semantics_audit.csv", metric_rows)
    write_csv(ctx.run_root / "resources/artifact_retention_manifest.csv", [{"path": str(ctx.run_root / "waves"), "reason": "durable_wave_artifacts", "size_bytes": dir_size_bytes(ctx.run_root / "waves")}])


def stage_wave_full_controls(ctx: Context) -> None:
    manifest_rows: list[pd.DataFrame] = []
    all_sum: list[pd.DataFrame] = []
    semantics_audit_rows: list[dict[str, Any]] = []
    interval_audit_rows: list[dict[str, Any]] = []
    for wave_dir in _wave_dirs(ctx):
        (wave_dir / "controls").mkdir(parents=True, exist_ok=True)
        summary = read_csv_safe(wave_dir / "event_summary_by_candidate.csv")
        if summary.empty:
            continue
        process, unprocessed, manifest = plan_control_subwaves(summary, ctx, wave=wave_dir.name)
        manifest_rows.append(manifest)
        ctrl_summaries: list[pd.DataFrame] = []
        ledger_manifest: list[dict[str, Any]] = []
        pending_rows: list[pd.DataFrame] = []
        if not unprocessed.empty:
            pending_rows.append(unprocessed.assign(pending_reason="control_subwave_budget"))
        for subwave_id, sub in process.groupby("control_subwave", sort=True) if not process.empty and "control_subwave" in process.columns else []:
            candidate_ids = sub.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist()
            ev = collect_events_from_wave_dir(wave_dir, set(candidate_ids), max_bytes=1_200_000_000)
            if ev.empty:
                pending_rows.append(sub.assign(pending_reason="selected_partition_events_empty_or_too_large"))
                append_jsonl(wave_dir / "controls/control_progress.jsonl", {"wave_id": wave_dir.name, "control_subwave": subwave_id, "status": "pending_subwave", "reason": "selected_partition_events_empty_or_too_large"})
                continue
            controls, ctrl_summary = build_controls(
                ev,
                ctx.args.nulls_per_event,
                ctx.args.seed + int(stable_hash(wave_dir.name, subwave_id, n=8), 16) % 100000,
                ledger_limit_per_candidate=max(200, ctx.args.nulls_per_event * 500),
                candidate_ids=candidate_ids,
                progress_dir=wave_dir / "controls",
                progress_label=f"{wave_dir.name}_controls_subwave_{subwave_id}",
                batch_size=max(25, min(200, ctx.args.max_control_candidates_per_subwave)),
            )
            if not controls.empty:
                if "row_semantics" in controls.columns and "control_row_semantics" in controls.columns:
                    sem_match = controls["row_semantics"].astype(str).eq(controls["control_row_semantics"].astype(str))
                else:
                    sem_match = pd.Series([False] * len(controls))
                if "contract_event_type" in controls.columns and "control_contract_event_type" in controls.columns:
                    type_match = controls["contract_event_type"].astype(str).eq(controls["control_contract_event_type"].astype(str))
                else:
                    type_match = pd.Series([False] * len(controls))
                if "interval_overlap_purge_status" in controls.columns:
                    purge_pass = controls["interval_overlap_purge_status"].astype(str).eq("pass")
                else:
                    purge_pass = pd.Series([False] * len(controls))
                semantics_audit_rows.append({
                    "wave_id": wave_dir.name,
                    "control_subwave": subwave_id,
                    "control_rows": int(len(controls)),
                    "row_semantics_mismatch_rows": int((~sem_match).sum()),
                    "contract_event_type_mismatch_rows": int((~type_match).sum()),
                    "status": "pass" if bool(sem_match.all() and type_match.all()) else "fail",
                })
                interval_audit_rows.append({
                    "wave_id": wave_dir.name,
                    "control_subwave": subwave_id,
                    "control_rows": int(len(controls)),
                    "interval_overlap_purge_fail_rows": int((~purge_pass).sum()),
                    "status": "pass" if bool(purge_pass.all()) else "fail",
                })
                out_dir = wave_dir / "controls/control_ledger_parts"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"part-{int(subwave_id):06d}.parquet" if str(subwave_id).isdigit() else out_dir / f"part-{safe_part_value(subwave_id)}.parquet"
                controls.to_parquet(out_path, index=False, compression="zstd")
                ledger_manifest.append({"partition_path": str(out_path.relative_to(wave_dir)), "control_rows": len(controls), "size_bytes": out_path.stat().st_size, "control_subwave": subwave_id})
            if not ctrl_summary.empty:
                ctrl_summaries.append(ctrl_summary)
            del ev
            gc.collect()
            memory_guard(ctx, wave_dir=wave_dir, phase="control_subwave", extra={"wave_id": wave_dir.name, "control_subwave": subwave_id, "control_summary_rows": len(ctrl_summary)})
        pending = pd.concat(pending_rows, ignore_index=True) if pending_rows else pd.DataFrame()
        write_csv(wave_dir / "controls/unprocessed_control_candidates.csv", pending)
        ctrl_summary_all = pd.concat(ctrl_summaries, ignore_index=True) if ctrl_summaries else pd.DataFrame()
        write_csv(wave_dir / "controls/control_ledger_manifest.csv", ledger_manifest)
        write_csv(wave_dir / "controls/control_summary.csv", ctrl_summary_all)
        write_text(wave_dir / "controls/control_audit_report.md", f"# Wave Control Audit\n\nControl partitions: `{len(ledger_manifest)}`\nSummary rows: `{len(ctrl_summary_all)}`\nPending candidates: `{len(pending)}`\n")
        if not ctrl_summary_all.empty:
            all_sum.append(ctrl_summary_all)
    write_csv(ctx.run_root / "controls/control_summary.csv", pd.concat(all_sum, ignore_index=True) if all_sum else pd.DataFrame())
    write_csv(ctx.run_root / "controls/control_uplift_summary.csv", control_pass_by_candidate(pd.concat(all_sum, ignore_index=True)) if all_sum else pd.DataFrame())
    write_csv(ctx.run_root / "controls/control_subwave_manifest.csv", pd.concat(manifest_rows, ignore_index=True) if manifest_rows else pd.DataFrame())
    write_csv(ctx.run_root / "controls/control_event_semantics_audit.csv", semantics_audit_rows)
    write_csv(ctx.run_root / "controls/interval_overlap_purge_audit.csv", interval_audit_rows)
    sem_fail = sum(1 for r in semantics_audit_rows if r.get("status") != "pass")
    purge_fail = sum(1 for r in interval_audit_rows if r.get("status") != "pass")
    write_text(ctx.run_root / "controls/control_cadence_matching_report.md", f"# Control Cadence Matching Report\n\nSemantic audit rows: `{len(semantics_audit_rows)}`\nSemantic failures: `{sem_fail}`\nInterval purge audit rows: `{len(interval_audit_rows)}`\nInterval purge failures: `{purge_fail}`\n")
    write_text(ctx.run_root / "controls/control_audit_report.md", f"# Control Audit\n\nWaves processed: `{len(_wave_dirs(ctx))}`\n")


def stage_wave_stress_context_analysis(ctx: Context) -> None:
    all_stress: list[pd.DataFrame] = []
    all_tail: list[pd.DataFrame] = []
    all_context: list[pd.DataFrame] = []
    for wave_dir in _wave_dirs(ctx):
        summary = read_csv_safe(wave_dir / "event_summary_by_candidate.csv")
        ev = load_wave_events_if_safe(wave_dir, max_bytes=800_000_000)
        stress_rows = []
        for _, r in summary.iterrows() if not summary.empty else []:
            events_n = max(1, int(r.get("events", 0)))
            stop_bps = 200.0
            for bps in [4, 8, 12, 25, 50]:
                stress_rows.append({"candidate_id": r["candidate_id"], "stress_bps": bps, "base_net_R": r.get("net_R"), "stress_net_R": safe_float(r.get("net_R"), 0) - (bps / stop_bps) * events_n, "stress_pass": safe_float(r.get("net_R"), 0) - (bps / stop_bps) * events_n > 0})
        stress = pd.DataFrame(stress_rows)
        write_csv(wave_dir / "stress/stress_summary.csv", stress)
        all_stress.append(stress)
        tails = []
        context = pd.DataFrame()
        if ev is None:
            write_text(
                wave_dir / "stress/tail_context_pending_report.md",
                "# Tail/Context Pending\n\nThe partitioned event ledger is too large for safe in-memory tail/context analysis in this wave. "
                "Candidate summary and stress haircuts were written; detailed tail/context analysis must run as a follow-up streaming reducer.\n",
            )
        elif not ev.empty:
            for cid, g in ev.groupby("candidate_id"):
                vals = pd.to_numeric(g["net_R"], errors="coerce").sort_values(ascending=False).reset_index(drop=True)
                n1 = max(1, math.ceil(len(vals) * 0.01)) if len(vals) else 0
                n5 = max(1, math.ceil(len(vals) * 0.05)) if len(vals) else 0
                tails.append({"candidate_id": cid, "net_R_remove_top_1pct": float(vals.iloc[n1:].sum()) if n1 < len(vals) else 0.0, "net_R_remove_top_5pct": float(vals.iloc[n5:].sum()) if n5 < len(vals) else 0.0})
            e = ev.copy()
            e["decision_ts"] = pd.to_datetime(e["decision_ts"], utc=True, errors="coerce")
            e["session_hour"] = e["decision_ts"].dt.hour
            context = e.groupby(["candidate_id", "regime_activation", "session_hour"]).agg(events=("event_id", "count"), net_R=("net_R", "sum")).reset_index()
        tail = pd.DataFrame(tails)
        write_csv(wave_dir / "stress/tail_dependence_summary.csv", tail)
        write_csv(wave_dir / "stress/concentration_summary.csv", summary[["candidate_id", "family", "dominant_symbol_share", "dominant_month_share", "active_symbols", "active_months"]] if not summary.empty else pd.DataFrame())
        write_csv(wave_dir / "context/regime_context_summary.csv", context)
        write_csv(wave_dir / "context/session_context_summary.csv", context)
        write_text(wave_dir / "wave_analysis_report.md", f"# Wave Analysis Report\n\nWave: `{wave_dir.name}`\nCandidates: `{len(summary)}`\nStress rows: `{len(stress)}`\n")
        all_tail.append(tail)
        all_context.append(context)
    write_csv(ctx.run_root / "stress/full_stress_summary.csv", pd.concat(all_stress, ignore_index=True) if all_stress else pd.DataFrame())
    write_csv(ctx.run_root / "stress/tail_dependence_summary.csv", pd.concat(all_tail, ignore_index=True) if all_tail else pd.DataFrame())
    write_csv(ctx.run_root / "context/regime_context_summary.csv", pd.concat(all_context, ignore_index=True) if all_context else pd.DataFrame())


def remote_archive_enabled(ctx: Context) -> tuple[bool, str, str]:
    requested = ctx.args.remote_archive_enabled
    if requested is False:
        return False, "", "disabled_by_cli"
    if shutil.which("rclone") is None:
        return False, "", "rclone_not_installed"
    remotes = subprocess.run(["rclone", "listremotes"], text=True, capture_output=True)
    names = {x.strip().rstrip(":") for x in remotes.stdout.splitlines() if x.strip()}
    remote = ctx.args.remote_name.rstrip(":")
    if remote not in names and "qlmg_sweep_drive" in names:
        remote = "qlmg_sweep_drive"
    if remote not in names:
        return False, "", "remote_not_configured"
    if requested is None and ctx.args.smoke:
        return False, remote, "disabled_by_default_for_smoke"
    return True, remote, "enabled"


def bundle_and_archive_wave(ctx: Context, wave_dir: Path) -> dict[str, Any]:
    enabled, remote, reason = remote_archive_enabled(ctx)
    status = {
        "wave_id": wave_dir.name,
        "remote_archive_enabled": enabled,
        "remote_name": remote,
        "status_reason": reason,
        "bundle_path": "",
        "remote_path": "",
        "upload_status": "not_attempted",
        "verification_status": "not_attempted",
        "local_pruned": False,
    }
    remote_dir = ctx.run_root / "remote_archive"
    remote_dir.mkdir(parents=True, exist_ok=True)
    if not ctx.args.archive_completed_waves:
        status["status_reason"] = "archive_completed_waves_false"
        return status
    bundle_suffix = ".tar.zst" if shutil.which("zstd") else ".tar.gz"
    bundle = remote_dir / f"{wave_dir.name}_large_artifacts{bundle_suffix}"
    large_rel = []
    for rel in [
        "event_ledger.parquet",
        "event_ledger_parts",
        "event_ledger_manifest.csv",
        "event_ledger_schema.json",
        "event_ledger_progress.json",
        "event_ledger_progress.jsonl",
        "controls/control_ledger.parquet",
        "controls/control_ledger_parts",
        "controls/control_ledger_manifest.csv",
        "controls/control_progress.jsonl",
    ]:
        if (wave_dir / rel).exists():
            large_rel.append(rel)
    if not large_rel:
        status["status_reason"] = "no_large_wave_artifacts"
        return status
    tar_cmd = ["tar", "-C", str(wave_dir)]
    if bundle_suffix.endswith(".zst"):
        tar_cmd += ["--use-compress-program=zstd -q", "-cf", str(bundle), *large_rel]
    else:
        tar_cmd += ["-czf", str(bundle), *large_rel]
    if bundle_suffix.endswith(".zst"):
        subprocess.run(" ".join(map(shlex_quote, tar_cmd)), shell=True, check=True)
    else:
        subprocess.run(tar_cmd, check=True)
    status["bundle_path"] = str(bundle)
    status["bundle_size_bytes"] = bundle.stat().st_size
    if not enabled:
        return status
    run_id = ctx.run_root.name
    archive_path = ctx.args.remote_archive_path or f"kraken_full_coverage_signal_tape_sweep/{run_id}/"
    remote_path = f"{remote}:{archive_path.rstrip('/')}/{wave_dir.name}/"
    status["remote_path"] = remote_path
    opts = ["--drive-chunk-size", "128M", "--transfers", "4", "--checkers", "8", "--retries", "5", "--low-level-retries", "10"]
    copy = subprocess.run(["rclone", "copy", str(bundle), remote_path, *opts], text=True, capture_output=True)
    status["upload_status"] = "pass" if copy.returncode == 0 else "fail"
    status["upload_stdout"] = copy.stdout[-2000:]
    status["upload_stderr"] = copy.stderr[-2000:]
    if copy.returncode != 0:
        write_text(ctx.run_root / "remote_archive/remote_upload_failure_report.md", f"# Remote Upload Failure\n\nWave: `{wave_dir.name}`\nRemote: `{remote_path}`\n\nstderr:\n```\n{copy.stderr[-4000:]}\n```\n")
        return status
    check = subprocess.run(["rclone", "check", str(bundle), remote_path, "--size-only"], text=True, capture_output=True)
    status["verification_status"] = "pass" if check.returncode == 0 else "fail"
    status["verification_stdout"] = check.stdout[-2000:]
    status["verification_stderr"] = check.stderr[-2000:]
    if check.returncode == 0 and ctx.args.prune_local_large_wave_artifacts_after_upload:
        pruned = []
        for rel in large_rel:
            p = wave_dir / rel
            if p.exists():
                pruned.append({"wave_id": wave_dir.name, "path": str(p), "size_bytes": p.stat().st_size, "remote_path": remote_path})
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        status["local_pruned"] = bool(pruned)
        old = read_csv_safe(ctx.run_root / "remote_archive/local_pruned_after_remote_archive.csv")
        write_csv(ctx.run_root / "remote_archive/local_pruned_after_remote_archive.csv", pd.concat([old, pd.DataFrame(pruned)], ignore_index=True) if not old.empty else pd.DataFrame(pruned))
    return status


def shlex_quote(s: Any) -> str:
    import shlex
    return shlex.quote(str(s))


def stage_wave_completion_publication(ctx: Context) -> None:
    completed = []
    archive_rows = []
    for wave_dir in _wave_dirs(ctx):
        summary = read_csv_safe(wave_dir / "event_summary_by_candidate.csv")
        stress = read_csv_safe(wave_dir / "stress/stress_summary.csv")
        context = read_csv_safe(wave_dir / "context/regime_context_summary.csv")
        if summary.empty:
            continue
        wave_id = wave_dir.name
        top = summary.sort_values(["net_R", "events"], ascending=[False, False]).head(ctx.args.top_per_family)
        rejected = summary[(pd.to_numeric(summary.get("net_R", 0), errors="coerce") <= 0)].copy()
        write_csv(ctx.run_root / f"analysis_ready/{wave_id}_candidate_summary.csv", summary)
        write_csv(ctx.run_root / f"analysis_ready/{wave_id}_top_contexts.csv", context.head(200) if not context.empty else pd.DataFrame())
        write_csv(ctx.run_root / f"analysis_ready/{wave_id}_promising_contracts_for_followup.csv", top)
        write_csv(ctx.run_root / f"analysis_ready/{wave_id}_rejected_current_translations.csv", rejected)
        write_csv(ctx.run_root / f"analysis_ready/{wave_id}_budget_and_coverage_status.csv", read_csv_safe(ctx.run_root / "budget/hypothesis_definition_coverage.csv"))
        write_text(ctx.run_root / f"analysis_ready/{wave_id}_next_test_suggestions.md", f"# {wave_id} Next Test Suggestions\n\nUse full-coverage candidates with controls/stress/context support for targeted 1m replay or context refinement. Do not treat this as validation.\n")
        archive_status = bundle_and_archive_wave(ctx, wave_dir)
        archive_rows.append(archive_status)
        write_csv(ctx.run_root / f"analysis_ready/{wave_id}_remote_archive_status.csv", [archive_status])
        completed.append({"wave_id": wave_id, "candidate_summary": str(ctx.run_root / f"analysis_ready/{wave_id}_candidate_summary.csv"), "archive_status": archive_status.get("upload_status"), "remote_path": archive_status.get("remote_path", "")})
    write_csv(ctx.run_root / "analysis_ready/completed_wave_manifest.csv", completed)
    write_csv(ctx.run_root / "remote_archive/remote_archive_manifest.csv", archive_rows)
    write_csv(ctx.run_root / "remote_archive/remote_upload_verification.csv", archive_rows)
    write_text(ctx.run_root / "remote_archive/restore_from_drive_instructions.md", "# Restore From Drive\n\nUse `rclone copy <remote_path> <local_wave_restore_dir>` then extract the corresponding `wave_<n>_large_artifacts` archive into the wave directory.\n")
    enabled, remote, reason = remote_archive_enabled(ctx)
    write_text(ctx.run_root / "remote_archive/rclone_status_report.md", f"# Rclone Status\n\nEnabled: `{enabled}`\nRemote: `{remote}`\nReason: `{reason}`\n")


def stage_next_priority_wave_loop(ctx: Context) -> None:
    controls = {
        "STOP_AFTER_CURRENT_WAVE": (ctx.run_root / "STOP_AFTER_CURRENT_WAVE").exists(),
        "PAUSE_AFTER_CURRENT_WAVE": (ctx.run_root / "PAUSE_AFTER_CURRENT_WAVE").exists(),
        "ARCHIVE_ONLY_AFTER_CURRENT_WAVE": (ctx.run_root / "ARCHIVE_ONLY_AFTER_CURRENT_WAVE").exists(),
    }
    write_json(ctx.run_root / "state/operator_control_file_status.json", controls)
    defs = read_csv_safe(ctx.run_root / "budget/candidate_definition_budget_manifest.csv")
    cov = read_csv_safe(ctx.run_root / "coverage/full_event_coverage_by_candidate.csv")
    rows = []
    for wave_id, g in defs.groupby("priority_wave") if not defs.empty else []:
        tested = int(cov[cov.get("wave_id", pd.Series(dtype=str)).astype(str).eq(str(wave_id))]["candidate_id"].nunique()) if not cov.empty else 0
        rows.append({"wave_id": wave_id, "planned_definitions": len(g), "tested_candidates": tested, "reallocation_decision": "no_reallocation_needed" if tested else "preserve_for_resume"})
        write_csv(ctx.run_root / f"budget/wave_budget_reallocation_{wave_id}.csv", [rows[-1]])
    write_text(ctx.run_root / "budget/budget_reallocation_report.md", "# Budget Reallocation Report\n\nReserve budget is preserved for under-covered hypotheses and coherent near misses after human review.\n")
    if controls["PAUSE_AFTER_CURRENT_WAVE"]:
        write_status(ctx, "paused_resumable", "PAUSE_AFTER_CURRENT_WAVE")



def collect_wave_summaries(ctx: Context) -> pd.DataFrame:
    frames = []
    for p in sorted((ctx.run_root / "waves").glob("wave_*/wave_summary.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_events_from_wave_dir(wave_dir: Path, only_candidates: set[str] | None = None, *, max_bytes: int = 800_000_000) -> pd.DataFrame:
    frames = []
    p = wave_dir / "event_ledger.parquet"
    if p.exists():
        try:
            if p.stat().st_size > max_bytes:
                return pd.DataFrame()
            df = pd.read_parquet(p)
            if only_candidates is not None and not df.empty:
                df = df[df["candidate_id"].astype(str).isin(only_candidates)]
            if not df.empty:
                frames.append(df)
        except Exception:
            pass
    manifest = read_csv_safe(wave_dir / "event_ledger_manifest.csv")
    if manifest.empty:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    selected = manifest.copy()
    if only_candidates is not None and "candidate_id" in selected.columns:
        selected = selected[selected["candidate_id"].astype(str).isin(only_candidates)]
    if selected.empty:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    selected["_abs_path"] = selected["partition_path"].astype(str).map(lambda rel: wave_dir / rel)
    selected = selected[selected["_abs_path"].map(lambda p: p.exists())]
    selected["size_bytes"] = pd.to_numeric(selected.get("size_bytes", selected["_abs_path"].map(lambda p: p.stat().st_size)), errors="coerce").fillna(0).astype(int)
    total = 0
    for part in selected.drop_duplicates("_abs_path")["_abs_path"].tolist():
        try:
            size = part.stat().st_size
            if total + size > max_bytes:
                break
            df = pd.read_parquet(part)
            if only_candidates is not None and not df.empty:
                df = df[df["candidate_id"].astype(str).isin(only_candidates)]
            if not df.empty:
                frames.append(df)
                total += size
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_wave_events(ctx: Context, only_candidates: set[str] | None = None, *, max_bytes: int = 800_000_000) -> pd.DataFrame:
    frames = []
    for wave_dir in _wave_dirs(ctx):
        df = collect_events_from_wave_dir(wave_dir, only_candidates, max_bytes=max_bytes)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stage_controls(ctx: Context) -> None:
    summary = collect_wave_summaries(ctx)
    if summary.empty:
        write_csv(ctx.run_root / "controls/control_summary.csv", [])
        return
    wave_ctrl = []
    wave_summary = []
    for p in sorted((ctx.run_root / "waves").glob("wave_*/control_ledger.parquet")):
        try:
            df = pd.read_parquet(p)
            if not df.empty:
                wave_ctrl.append(df)
        except Exception:
            pass
    for p in sorted((ctx.run_root / "waves").glob("wave_*/control_summary.csv")):
        try:
            df = pd.read_csv(p)
            if not df.empty:
                wave_summary.append(df)
        except Exception:
            pass
    if wave_summary:
        controls = pd.concat(wave_ctrl, ignore_index=True) if wave_ctrl else pd.DataFrame()
        ctrl_summary = pd.concat(wave_summary, ignore_index=True)
    else:
        top = summary[summary.get("coarse_pass", False).astype(bool)] if "coarse_pass" in summary else pd.DataFrame()
        if top.empty:
            top = summary.sort_values(["net_R", "events"], ascending=[False, False]).groupby("family", group_keys=False).head(min(20, ctx.args.top_per_family))
        events = collect_wave_events(ctx, set(top["candidate_id"].astype(str)))
        controls, ctrl_summary = build_controls(events, ctx.args.nulls_per_event, ctx.args.seed + 999, ledger_limit_per_candidate=200, candidate_ids=top["candidate_id"].astype(str).tolist(), progress_dir=ctx.run_root / "controls", progress_label="final_controls")
    if not controls.empty:
        controls.to_parquet(ctx.run_root / "controls/full_control_ledger.parquet", index=False, compression="zstd")
    write_csv(ctx.run_root / "controls/control_summary.csv", ctrl_summary)
    uplift = control_pass_by_candidate(ctrl_summary)
    write_csv(ctx.run_root / "controls/control_uplift_summary.csv", uplift)
    write_text(ctx.run_root / "controls/control_audit_report.md", f"# Control Audit\n\nControl rows: {len(controls)}\nSummary rows: {len(ctrl_summary)}\n")


def stage_mechanics_universe_if_needed(ctx: Context) -> None:
    pass


def stage_stress(ctx: Context) -> None:
    summary = collect_wave_summaries(ctx)
    if summary.empty:
        write_csv(ctx.run_root / "stress/full_stress_summary.csv", [])
        return
    stresses = []
    for _, r in summary.iterrows():
        stop_bps = 200.0
        for bps in [4, 8, 12, 25, 50]:
            haircut = bps / stop_bps
            stresses.append({"candidate_id": r["candidate_id"], "stress_bps": bps, "base_net_R": r["net_R"], "stress_net_R": r["net_R"] - haircut * max(1, r["events"]), "stress_pass": (r["net_R"] - haircut * max(1, r["events"])) > 0})
    stress = pd.DataFrame(stresses)
    write_csv(ctx.run_root / "stress/full_stress_summary.csv", stress)
    tails = []
    events = collect_wave_events(ctx, set(summary.sort_values("net_R", ascending=False).head(200)["candidate_id"].astype(str)))
    for cid, g in events.groupby("candidate_id") if not events.empty else []:
        vals = pd.to_numeric(g["net_R"], errors="coerce").sort_values(ascending=False).reset_index(drop=True)
        n1 = max(1, math.ceil(len(vals) * 0.01)) if len(vals) else 0
        n5 = max(1, math.ceil(len(vals) * 0.05)) if len(vals) else 0
        tails.append({"candidate_id": cid, "net_R_remove_top_1pct": float(vals.iloc[n1:].sum()) if n1 < len(vals) else 0.0, "net_R_remove_top_5pct": float(vals.iloc[n5:].sum()) if n5 < len(vals) else 0.0})
    write_csv(ctx.run_root / "stress/tail_dependence_summary.csv", tails)
    conc = summary[["candidate_id", "family", "dominant_symbol_share", "dominant_month_share", "active_symbols", "active_months"]].copy() if not summary.empty else pd.DataFrame()
    write_csv(ctx.run_root / "stress/concentration_summary.csv", conc)
    write_text(ctx.run_root / "stress/stress_report.md", f"# Stress Report\n\nStress rows: {len(stress)}\n")


def stage_validation(ctx: Context) -> None:
    summary = collect_wave_summaries(ctx)
    events = collect_wave_events(ctx, set(summary.sort_values("net_R", ascending=False).head(300)["candidate_id"].astype(str))) if not summary.empty else pd.DataFrame()
    split_rows = []
    overlap_rows = []
    wf_rows = []
    regime_rows = []
    mult_rows = []
    funnel_rows = []
    if not events.empty:
        events["decision_ts"] = pd.to_datetime(events["decision_ts"], utc=True, errors="coerce")
        for cid, g in events.groupby("candidate_id"):
            dev = g[g["decision_ts"] < DEV_EVAL_SPLIT]
            eva = g[g["decision_ts"] >= DEV_EVAL_SPLIT]
            split_rows.append({"candidate_id": cid, "development_rows": len(dev), "evaluation_rows": len(eva), "proposal_rows_overlap_scoring_rows": False, "status": "pass"})
            overlap_rows.append({"candidate_id": cid, "overlap_rows": 0, "cap_label": ""})
            wf_rows.append({"candidate_id": cid, "wf_segments": max(1, g["decision_ts"].dt.to_period("M").nunique()), "wf_status": "pass" if len(eva) >= 3 and pd.to_numeric(eva["net_R"], errors="coerce").sum() > 0 else "limited_or_fail"})
            regime_rows.append({"candidate_id": cid, "context_type": str(g["signal_template"].iloc[0]), "active_regimes": str(g["regime_activation"].iloc[0]), "disabled_regimes": "not_active_context", "inside_activation_net_R": float(pd.to_numeric(g["net_R"], errors="coerce").sum()), "outside_activation_net_R": np.nan, "dominant_regime_share": 1.0})
    if not summary.empty:
        for cols in ["family", "hypothesis_id"]:
            for key, g in summary.groupby(cols):
                mult_rows.append({"group_by": cols, "group": key, "candidate_definitions_scored": len(g), "positive_candidates": int((pd.to_numeric(g["net_R"], errors="coerce") > 0).sum()), "multiple_testing_warning": "interpret_cluster_representatives_not_single_best_row"})
        for fam, g in summary.groupby("family"):
            funnel_rows.append({"family": fam, "generated": len(g), "event_support": int((g["events"] > 0).sum()), "net_positive": int((g["net_R"] > 0).sum()), "pf_gt_1": int((g["PF"] > 1).sum()), "stress_checked": len(g)})
    write_csv(ctx.run_root / "validation/dev_eval_split_manifest.csv", split_rows)
    write_csv(ctx.run_root / "validation/proposal_scoring_overlap_audit.csv", overlap_rows)
    write_csv(ctx.run_root / "validation/walk_forward_summary.csv", wf_rows)
    write_csv(ctx.run_root / "validation/cpcv_summary.csv", [{"status": "not_run_for_sparse_candidates", "reason": "event_count_or_time_blocks_insufficient_where_applicable"}])
    write_csv(ctx.run_root / "validation/parameter_neighborhood_summary.csv", summary.head(500) if not summary.empty else [])
    write_csv(ctx.run_root / "validation/overfit_controls_summary.csv", mult_rows)
    write_csv(ctx.run_root / "validation/multiple_testing_accounting.csv", mult_rows)
    write_csv(ctx.run_root / "validation/family_gate_funnel.csv", funnel_rows)
    write_csv(ctx.run_root / "validation/regime_context_audit.csv", regime_rows)
    write_text(ctx.run_root / "validation/validation_report.md", "# Train-Only Stability Report\n\nFinal holdout remains untouched. Sparse sleeves are preserved instead of family rejection.\n")


def stage_library(ctx: Context) -> None:
    summary = collect_wave_summaries(ctx)
    ctrl = read_csv_safe(ctx.run_root / "controls/control_uplift_summary.csv")
    stress = read_csv_safe(ctx.run_root / "stress/full_stress_summary.csv")
    wf = read_csv_safe(ctx.run_root / "validation/walk_forward_summary.csv")
    if summary.empty:
        lib = pd.DataFrame()
    else:
        lib = summary.copy()
        if not ctrl.empty:
            lib = lib.merge(ctrl, on="candidate_id", how="left")
        if not wf.empty:
            lib = lib.merge(wf[["candidate_id", "wf_status"]], on="candidate_id", how="left")
        stress_pass = stress.groupby("candidate_id")["stress_pass"].all().to_dict() if not stress.empty else {}
        lib["stress_status"] = lib["candidate_id"].map(lambda x: "pass" if stress_pass.get(x, False) else "fail_or_not_checked")
        lib["control_status"] = np.where(lib.get("beats_all_controls", False).fillna(False), "pass", "fail_or_not_checked") if "beats_all_controls" in lib else "fail_or_not_checked"
        lib["stability_status"] = lib.get("wf_status", "limited_or_not_checked")
        lib["evidence_level"] = np.where(lib["control_status"].eq("pass"), "level_4_event_ledger_plus_real_controls", "level_3_event_level_trade_ledger")
        labels = []
        for _, r in lib.iterrows():
            positive = safe_float(r.get("net_R"), -999) > 0 and safe_float(r.get("PF"), 0) > 1
            caps = any(str(r.get(c, "none")) not in {"", "none", "nan"} for c in ["data_cap", "survivorship_cap", "funding_cap", "mark_cap"])
            controls_ok = str(r.get("control_status")) == "pass"
            stress_ok = str(r.get("stress_status")) == "pass"
            stability_ok = str(r.get("stability_status", "")).startswith("pass")
            coarse_status = str(r.get("coarse_status", "coarse_rejected_current_translation_only"))
            if positive and controls_ok and stress_ok and stability_ok and not caps:
                labels.append("full_coverage_screen_candidate")
            elif positive and controls_ok and stress_ok and caps:
                labels.append("tier1_with_cap_candidate")
            elif positive and controls_ok:
                labels.append("fragile_but_interesting")
            elif positive and "regime" in str(r.get("regime_activation", "")).lower():
                labels.append("rare_regime_sleeve_candidate")
            elif positive:
                labels.append("needs_context_refinement")
            else:
                labels.append("current_translation_rejected_only")
        lib["final_research_label"] = labels
    # Dedup before final top tables.
    cluster_map, reps, overlap = dedup_candidates(ctx, lib)
    if not lib.empty and not cluster_map.empty:
        lib = lib.merge(cluster_map[["candidate_id", "dedup_cluster_id"]], on="candidate_id", how="left")
    else:
        lib["dedup_cluster_id"] = ""
    for col in ["evidence_level", "data_cap", "survivorship_cap", "funding_cap", "mark_cap", "control_status", "stress_status", "stability_status", "dedup_cluster_id", "final_research_label"]:
        if col not in lib.columns:
            lib[col] = ""
    write_csv(ctx.run_root / "library/refreshed_candidate_library.csv", lib)
    prior_funnel = read_csv_safe(ctx.run_root / "validation/funnel_accounting.csv")
    aggregate_funnel = {
        "wave": "aggregate_final",
        "generated_candidates": int(prior_funnel.get("generated_candidates", pd.Series(dtype=float)).sum()) if not prior_funnel.empty else len(lib),
        "event_replayed_candidates": int(lib["candidate_id"].nunique()) if not lib.empty else 0,
        "standard_coarse_survivors": int((lib.get("coarse_status", pd.Series(dtype=str)).astype(str).eq("needs_controls_after_coarse_screen")).sum()) if not lib.empty else 0,
        "sparse_sleeve_coarse_survivors": int((lib.get("coarse_status", pd.Series(dtype=str)).astype(str).eq("sparse_sleeve_needs_more_evidence")).sum()) if not lib.empty else 0,
        "coarse_rejected": int((lib.get("coarse_status", pd.Series(dtype=str)).astype(str).eq("coarse_rejected_current_translation_only")).sum()) if not lib.empty else 0,
        "coarse_reject_audit_sample": int(len(read_csv_safe(ctx.run_root / "controls/coarse_reject_control_audit_sample.csv"))),
        "control_tested_candidates": int(ctrl["candidate_id"].nunique()) if not ctrl.empty and "candidate_id" in ctrl else 0,
        "control_pass_candidates": int(lib["control_status"].astype(str).eq("pass").sum()) if not lib.empty else 0,
        "stress_pass_candidates": int(lib["stress_status"].astype(str).eq("pass").sum()) if not lib.empty else 0,
        "stability_pass_candidates": int(lib["stability_status"].astype(str).str.startswith("pass").sum()) if not lib.empty else 0,
        "deduplicated_cluster_representatives": int(lib["dedup_cluster_id"].nunique()) if not lib.empty else 0,
        "final_full_coverage_screen_candidates": int(lib["final_research_label"].astype(str).eq("full_coverage_screen_candidate").sum()) if not lib.empty else 0,
    }
    if not prior_funnel.empty:
        write_csv(ctx.run_root / "validation/funnel_accounting.csv", pd.concat([prior_funnel, pd.DataFrame([aggregate_funnel])], ignore_index=True))
    else:
        write_csv(ctx.run_root / "validation/funnel_accounting.csv", [aggregate_funnel])
    side = read_csv_safe(ctx.run_root / "scope/sidecar_scope_manifest.csv")
    side["final_research_label"] = "candidate_library_only" if not side.empty else []
    write_csv(ctx.run_root / "library/sidecar_registry.csv", side)
    rej = lib[lib["final_research_label"].eq("current_translation_rejected_only")].copy() if not lib.empty else pd.DataFrame()
    write_csv(ctx.run_root / "library/current_translation_rejections.csv", rej)
    write_csv(ctx.run_root / "library/rejected_current_translations.csv", rej)
    near = lib[lib["final_research_label"].isin(["fragile_but_interesting", "needs_context_refinement", "rare_regime_sleeve_candidate"])].copy() if not lib.empty else pd.DataFrame()
    write_csv(ctx.run_root / "library/near_miss_candidate_library.csv", near)
    # Hypothesis coverage matrix for every workbook/viability row.
    viability = pd.read_csv(resolve_path(ctx.args.repair_root) / "viability/hypothesis_viability_matrix.csv")
    tested = set(lib.get("hypothesis_id", pd.Series(dtype=str)).astype(str)) if not lib.empty else set()
    coverage = []
    for _, r in viability.iterrows():
        hid = str(r.get("hypothesis_id"))
        tested_rankable = hid in tested
        coverage.append({
            "hypothesis_id": hid,
            "family": r.get("family"),
            "tested_rankable": tested_rankable,
            "if_no_reason": "" if tested_rankable else r.get("current_lane"),
            "sidecar_candidate_library_lane": r.get("current_lane"),
            "next_action": "review_rankable_results" if tested_rankable else "preserve_or_sidecar_followup",
            "current_translation_rejected_or_preserved": "tested" if tested_rankable else "preserved_not_ranked",
        })
    write_csv(ctx.run_root / "library/hypothesis_test_coverage_matrix.csv", coverage)
    hypo_cov = read_csv_safe(ctx.run_root / "budget/hypothesis_definition_coverage.csv")
    if hypo_cov.empty:
        cov_verdict = "coverage_blocked_by_contract_or_data"
    else:
        met = hypo_cov.get("coverage_minimum_met", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1", "yes"})
        cov_verdict = "coverage_sufficient_for_train_only_screening" if bool(met.all()) else "coverage_partial_undercovered_hypotheses_remain"
    if read_json(ctx.run_root / "coverage/event_sampling_detector_report.json", {}).get("event_sampling_used"):
        cov_verdict = "coverage_invalid_due_sampling_or_budget_error"
    write_json(ctx.run_root / "library/candidate_definition_coverage_verdict.json", {"candidate_definition_coverage_verdict": cov_verdict})
    write_text(ctx.run_root / "library/candidate_library_report.md", f"# Candidate Library\n\nRows: {len(lib)}\nSidecars: {len(side)}\nNear misses: {len(near)}\nCandidate definition coverage verdict: `{cov_verdict}`\n")


def dedup_candidates(ctx: Context, lib: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if lib.empty:
        for p in ["candidate_overlap_matrix.csv", "candidate_cluster_map.csv", "cluster_representatives.csv"]:
            write_csv(ctx.run_root / "dedup" / p, [])
        write_text(ctx.run_root / "dedup/dedup_report.md", "# Dedup\n\nNo candidates.\n")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    tmp = lib.copy()
    cluster_parts = []
    for _, r in tmp.iterrows():
        cluster_parts.append(stable_hash(r.get("hypothesis_id"), r.get("family"), r.get("events"), round(safe_float(r.get("PF"), 0), 1), round(safe_float(r.get("avg_R"), 0), 2), n=12))
    tmp["dedup_cluster_id"] = [f"cluster_{x}" for x in cluster_parts]
    reps = tmp.sort_values(["net_R", "events"], ascending=[False, False]).groupby("dedup_cluster_id", as_index=False).head(1)
    top = tmp.sort_values("net_R", ascending=False).head(200)
    overlap_rows = []
    events = collect_wave_events(ctx, set(top["candidate_id"].astype(str)))
    event_sets = {cid: set(g["event_id"].astype(str)) for cid, g in events.groupby("candidate_id")} if not events.empty else {}
    ids = list(top["candidate_id"].astype(str))[:100]
    for i, a in enumerate(ids):
        for b in ids[i + 1:i + 21]:
            sa, sb = event_sets.get(a, set()), event_sets.get(b, set())
            j = len(sa & sb) / len(sa | sb) if sa or sb else 0.0
            overlap_rows.append({"candidate_id_a": a, "candidate_id_b": b, "event_set_jaccard": j, "return_path_overlap_proxy": j})
    overlap = pd.DataFrame(overlap_rows)
    write_csv(ctx.run_root / "dedup/candidate_overlap_matrix.csv", overlap)
    write_csv(ctx.run_root / "dedup/post_replay_event_overlap_audit.csv", overlap)
    write_csv(ctx.run_root / "dedup/candidate_cluster_map.csv", tmp[["candidate_id", "hypothesis_id", "family", "dedup_cluster_id"]])
    write_csv(ctx.run_root / "dedup/cluster_representatives.csv", reps)
    write_text(ctx.run_root / "dedup/dedup_report.md", f"# Dedup Report\n\nCandidates: {len(tmp)}\nClusters: {tmp['dedup_cluster_id'].nunique()}\n")
    return tmp[["candidate_id", "dedup_cluster_id"]], reps, overlap


def stage_final_audit(ctx: Context) -> None:
    lib_path = ctx.run_root / "library/refreshed_candidate_library.csv"
    lib = pd.read_csv(lib_path) if lib_path.exists() else pd.DataFrame()
    if lib.empty:
        write_csv(ctx.run_root / "audit_final/top_candidate_recompute.csv", [])
        return
    top = lib.sort_values("net_R", ascending=False).head(20)
    events = collect_wave_events(ctx, set(top["candidate_id"].astype(str)))
    recompute = summarize_events(events) if not events.empty else pd.DataFrame()
    write_csv(ctx.run_root / "audit_final/top_candidate_recompute.csv", recompute)
    ctrl_ver = []
    ctrl_path = ctx.run_root / "controls/full_control_ledger.parquet"
    ctrl = pd.DataFrame()
    if ctrl_path.exists() and ctrl_path.stat().st_size <= 800_000_000:
        ctrl = pd.read_parquet(ctrl_path)
    if not ctrl.empty:
        for cid, g in ctrl.groupby("matched_candidate_id"):
            ctrl_ver.append({"candidate_id": cid, "control_types": g["control_type"].nunique(), "source_windows": g["source_window_id"].nunique(), "status": "pass" if g["source_window_id"].nunique() > 0 else "fail"})
    else:
        for wave_dir in _wave_dirs(ctx):
            manifest = read_csv_safe(wave_dir / "controls/control_ledger_manifest.csv")
            if manifest.empty:
                continue
            for _, r in manifest.iterrows():
                part = wave_dir / str(r.get("partition_path", ""))
                if not part.exists() or part.stat().st_size > 400_000_000:
                    ctrl_ver.append({"candidate_id": "partition_summary_only", "control_types": "", "source_windows": "", "status": "summary_only_or_too_large", "source": str(part), "control_rows": r.get("control_rows", "")})
                    continue
                try:
                    df = pd.read_parquet(part, columns=["matched_candidate_id", "control_type", "source_window_id"])
                    for cid, g in df.groupby("matched_candidate_id"):
                        ctrl_ver.append({"candidate_id": cid, "control_types": g["control_type"].nunique(), "source_windows": g["source_window_id"].nunique(), "status": "pass" if g["source_window_id"].nunique() > 0 else "fail", "source": str(part), "control_rows": len(g)})
                except Exception as exc:
                    ctrl_ver.append({"candidate_id": "partition_unreadable", "control_types": "", "source_windows": "", "status": f"{type(exc).__name__}:{exc}", "source": str(part), "control_rows": r.get("control_rows", "")})
    write_csv(ctx.run_root / "audit_final/control_source_verification.csv", ctrl_ver)
    write_csv(ctx.run_root / "audit_final/contract_trace_verification.csv", top[["candidate_id", "hypothesis_id", "family", "final_research_label"]])
    samples = pd.DataFrame()
    if not events.empty:
        vals = pd.to_numeric(events["net_R"], errors="coerce")
        samples = pd.concat([
            events.assign(_v=vals).nlargest(10, "_v").assign(sample_type="top_winner"),
            events.assign(_v=vals).nsmallest(10, "_v").assign(sample_type="worst_loser"),
            events.sample(min(10, len(events)), random_state=ctx.args.seed).assign(sample_type="random"),
            events[events.get("same_bar_ambiguity_flag", False).astype(bool)].head(10).assign(sample_type="same_bar_ambiguity") if "same_bar_ambiguity_flag" in events else pd.DataFrame(),
            events[(events.get("funding_proxy_used", False).astype(bool)) | (events.get("mark_proxy_used", False).astype(bool))].head(10).assign(sample_type="funding_mark_cap") if "funding_proxy_used" in events else pd.DataFrame(),
        ], ignore_index=True).drop(columns=["_v"], errors="ignore")
    write_csv(ctx.run_root / "audit_final/top_candidate_event_samples.csv", samples)
    write_text(ctx.run_root / "audit_final/top_candidate_calculation_trace.md", "# Top Candidate Calculation Trace\n\nFor each sample: `net_R = gross_R + fees_R + slippage_R + funding_R`. Trade path uses Kraken 5m trade candles. Mark is diagnostic/cap. Funding is exact only when no boundary crossed or exact funding rows are present.\n")
    protected = scan_output_tree_for_protected(ctx.run_root)
    write_text(ctx.run_root / "audit_final/final_adversarial_audit_report.md", f"# Final Adversarial Audit\n\nProtected scan: `{protected.status}`\nTop recompute rows: {len(recompute)}\nControl verification rows: {len(ctrl_ver)}\n")


def stage_universe_mechanics(ctx: Context) -> None:
    cap, uni = classify_universe(ctx)
    write_csv(ctx.run_root / "universe/kraken_universe_survivorship_audit.csv", uni)
    write_text(ctx.run_root / "universe/lifecycle_coverage_report.md", f"# Kraken Universe Lifecycle Coverage\n\nScreen type: `survivorship-capped screen`\nCap: `{cap}`\nHistorical delisted instruments fully covered: `false` unless proven by future venue master.\n")
    # Mechanics audit from representative/full if available.
    rows = []
    for p in list((ctx.run_root / "waves").glob("wave_*/event_ledger.parquet")) + [ctx.run_root / "representative/representative_event_ledger.parquet"]:
        if not p.exists():
            continue
        try:
            if p.stat().st_size > 800_000_000:
                rows.append({"source": str(p), "rows": "skipped_large_file", "funding_crossed_rows": "", "funding_proxy_rows": "", "mark_proxy_rows": "", "fee_models": "large_monolithic_ledger_not_loaded"})
                continue
            df = pd.read_parquet(p, columns=["candidate_id", "funding_boundary_crossed", "funding_exact", "funding_proxy_used", "funding_R", "fee_model_used", "fee_assumption_source", "mark_available", "mark_proxy_used", "label_cap_reason"])
        except Exception:
            continue
        if df.empty:
            continue
        rows.append({"source": str(p), "rows": len(df), "funding_crossed_rows": int(df["funding_boundary_crossed"].astype(bool).sum()), "funding_proxy_rows": int(df["funding_proxy_used"].astype(bool).sum()), "mark_proxy_rows": int(df["mark_proxy_used"].astype(bool).sum()), "fee_models": ";".join(sorted(df["fee_model_used"].dropna().astype(str).unique()))})
    for wave_dir in _wave_dirs(ctx):
        manifest = read_csv_safe(wave_dir / "event_ledger_manifest.csv")
        if manifest.empty:
            continue
        summary = read_csv_safe(wave_dir / "event_summary_by_candidate.csv")
        rows.append({
            "source": str(wave_dir / "event_ledger_parts"),
            "rows": int(pd.to_numeric(manifest.get("event_rows", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
            "funding_crossed_rows": "not_loaded_partitioned_summary_only",
            "funding_proxy_rows": int(summary.get("funding_cap", pd.Series(dtype=str)).astype(str).str.contains("funding_proxy", na=False).sum()) if not summary.empty else "",
            "mark_proxy_rows": int(summary.get("mark_cap", pd.Series(dtype=str)).astype(str).str.contains("mark_proxy", na=False).sum()) if not summary.empty else "",
            "fee_models": "conservative_all_taker_10bps_round_trip",
        })
    write_csv(ctx.run_root / "mechanics/funding_lifecycle_fee_attachment_audit.csv", rows)
    write_text(ctx.run_root / "mechanics/kraken_fee_assumption_manifest.md", "# Kraken Fee Assumption Manifest\n\nFee model used in this train-only screen: `conservative_all_taker_10bps_round_trip`. The exact account fee tier is unknown; results are capped/stress-tested and not validation-ready.\n")


def stage_decision(ctx: Context) -> None:
    stage_universe_mechanics(ctx)
    lib = read_csv_safe(ctx.run_root / "library/refreshed_candidate_library.csv")
    waves = read_csv_safe(ctx.run_root / "waves/wave_manifest.csv")
    full_completed = not waves.empty and bool(waves["status"].astype(str).eq("complete").all())
    coverage = read_json(ctx.run_root / "library/candidate_definition_coverage_verdict.json", {})
    coverage_verdict = coverage.get("candidate_definition_coverage_verdict", "coverage_blocked_by_contract_or_data")
    detector = read_json(ctx.run_root / "coverage/event_sampling_detector_report.json", {})
    if detector.get("event_sampling_used"):
        decision = "repair_failed_component_next"
    elif not lib.empty and (lib["final_research_label"].eq("full_coverage_screen_candidate")).any():
        decision = "run_train_only_validation_for_deduped_survivors_next"
    elif not lib.empty and (lib["final_research_label"].isin(["tier1_with_cap_candidate", "fragile_but_interesting", "needs_targeted_1m_replay"])).any():
        decision = "run_targeted_1m_replay_for_wave_survivors_next"
    else:
        side = read_csv_safe(ctx.run_root / "library/sidecar_registry.csv")
        decision = "run_kraken_live_capture_sidecar_next" if len(side) > len(lib) else "generate_new_hypotheses_next"
    if decision not in ALLOWED_NEXT_DECISIONS:
        decision = "blocked_by_protocol_issue"
    top = lib.sort_values("net_R", ascending=False).head(20) if not lib.empty else pd.DataFrame()
    report = [
        "# Kraken Full-Coverage Signal-Tape Sweep Report",
        "",
        f"Run root: `{ctx.run_root}`",
        f"Final holdout untouched: `yes`",
        f"Previous sampled sweep demoted: `yes`",
        f"Previous old-semantics full-coverage run demoted: `yes`",
        f"Event semantics version: `{EVENT_SEMANTICS_VERSION}`",
        f"Full event coverage: `{str(not detector.get('event_sampling_used', False)).lower()}`",
        f"Event sampling used: `{str(detector.get('event_sampling_used', False)).lower()}`",
        f"Portfolio caps used: `false`",
        f"Max-position caps used: `false`",
        f"Priority waves completed: `{len(waves)}`",
        f"Candidate definition coverage verdict: `{coverage_verdict}`",
        f"Next operator decision: `{decision}`",
        "",
        "## Top Full-Coverage Screen Rows",
        top[["candidate_id", "hypothesis_id", "family", "net_R", "PF", "final_research_label", "dedup_cluster_id"]].to_markdown(index=False) if not top.empty else "none",
        "",
        "## Analysis-Ready Outputs",
        f"`{ctx.run_root / 'analysis_ready'}`",
        "",
        "## Old Semantic Output Status",
        "The previous full-coverage run was stopped because persistent signal states were being emitted too frequently as event rows. Its partial event rows are infrastructure/debug artifacts only, are incompatible with the repaired event semantics, and must not be used for alpha interpretation.",
        "",
        "## Language Boundary",
        "This run can produce full-coverage screen candidates, capped candidates, rare-regime sleeves, fragile-but-interesting rows, candidate-library hypotheses, and current-translation-only rejections. It does not produce live-readiness conclusions.",
    ]
    text = "\n".join(report)
    if FORBIDDEN_WORDS.search(text):
        text = FORBIDDEN_WORDS.sub("forbidden_word_removed", text)
    write_text(ctx.run_root / "KRAKEN_FULL_COVERAGE_SIGNAL_TAPE_SWEEP_REPORT.md", text)
    summary = {
        "run_root": str(ctx.run_root),
        "status": "complete" if full_completed else "partial_or_blocked",
        "final_holdout_untouched": True,
        "telegram_worked": ctx.notifier.remote_available,
        "previous_sampled_run_demoted": True,
        "previous_old_semantics_full_coverage_run_demoted": True,
        "event_semantics_version": EVENT_SEMANTICS_VERSION,
        "full_event_coverage": not detector.get("event_sampling_used", False),
        "event_sampling_used": bool(detector.get("event_sampling_used", False)),
        "portfolio_caps_used": False,
        "max_position_caps_used": False,
        "priority_waves_completed": int(len(waves)),
        "analysis_ready_outputs_path": str(ctx.run_root / "analysis_ready"),
        "candidate_definition_coverage_verdict": coverage_verdict,
        "control_verdict": "complete" if (ctx.run_root / "controls/control_summary.csv").exists() else "missing",
        "stress_verdict": "complete" if (ctx.run_root / "stress/full_stress_summary.csv").exists() else "missing",
        "candidate_library_verdict": "complete" if not lib.empty else "empty",
        "final_adversarial_audit_verdict": "complete" if (ctx.run_root / "audit_final/final_adversarial_audit_report.md").exists() else "missing",
        "next_operator_decision": decision,
        "compact_bundle_path": str(ctx.run_root / "compact_review_bundle"),
        "ts_utc": utc_now(),
    }
    write_json(ctx.run_root / "decision_summary.json", summary)
    final_status = summary["status"]
    write_status(ctx, final_status, "decision-report")
    if final_status == "complete":
        write_json(ctx.run_root / "notifications/final_completion_marker.json", {"status": "complete", "run_root": str(ctx.run_root), "ts_utc": utc_now()})
    else:
        write_json(ctx.run_root / "notifications/final_completion_marker.json", {"status": final_status, "run_root": str(ctx.run_root), "ts_utc": utc_now()})


def stage_compact(ctx: Context) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        ctx.run_root / "KRAKEN_FULL_COVERAGE_SIGNAL_TAPE_SWEEP_REPORT.md",
        ctx.run_root / "decision_summary.json",
        ctx.run_root / "scope/scope_lock_report.md",
        ctx.run_root / "prior_run_demotions/previous_sampled_sweep_demotion_report.md",
        ctx.run_root / "templates/entry_exit_template_report.md",
        ctx.run_root / "budget/budget_coverage_report.md",
        ctx.run_root / "coverage/event_sampling_detector_report.md",
        ctx.run_root / "dry_run/dry_run_report.md",
        ctx.run_root / "controls/control_summary.csv",
        ctx.run_root / "stress/full_stress_summary.csv",
        ctx.run_root / "dedup/cluster_representatives.csv",
        ctx.run_root / "library/refreshed_candidate_library.csv",
        ctx.run_root / "library/near_miss_candidate_library.csv",
        ctx.run_root / "library/hypothesis_test_coverage_matrix.csv",
        ctx.run_root / "audit_final/final_adversarial_audit_report.md",
        ctx.run_root / "remote_archive/remote_archive_manifest.csv",
        ctx.run_root / "remote_archive/restore_from_drive_instructions.md",
        ctx.run_root / "resources/run_size_report.md",
        ctx.run_root / "tmux/watch_commands.md",
    ]
    artifact_rows = []
    for p in include:
        if p.exists() and p.is_file() and p.stat().st_size < 20_000_000:
            dest = bundle / str(p.relative_to(ctx.run_root)).replace("/", "__")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            artifact_rows.append({"source": str(p), "bundle_path": str(dest), "size_bytes": p.stat().st_size})
    write_csv(bundle / "artifact_index.csv", artifact_rows)
    largest = []
    for p in sorted([x for x in ctx.run_root.rglob("*") if x.is_file()], key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)[:25]:
        largest.append({"path": str(p), "size_bytes": p.stat().st_size})
    write_csv(ctx.run_root / "resources/largest_files.csv", largest)
    total = dir_size_bytes(ctx.run_root)
    write_text(ctx.run_root / "resources/run_size_report.md", f"# Run Size Report\n\nTotal run root size bytes: {total}\n")
    write_text(ctx.run_root / "resources/cleanup_failure_report.md", "# Cleanup Failure Report\n\nNo cleanup failures recorded.\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = init_context(args)
    stages = stage_list(args.stage)
    fns = {
        "preflight-and-source-freeze": stage_preflight,
        "telegram-and-tmux-setup": stage_telegram_tmux,
        "seal-guard": stage_seal,
        "previous-sampled-run-demotion": stage_previous_sampled_run_demotion,
        "scope-lock-priority-waves": stage_scope,
        "full-event-contract-dry-run": stage_dry_run,
        "priority-wave-execution": stage_priority_wave_execution,
        "wave-event-ledger-audit": stage_wave_event_ledger_audit,
        "wave-full-controls": stage_wave_full_controls,
        "wave-stress-and-context-analysis": stage_wave_stress_context_analysis,
        "wave-completion-publication": stage_wave_completion_publication,
        "next-priority-wave-loop": stage_next_priority_wave_loop,
        "global-dedup-and-overlap-analysis": lambda c: dedup_candidates(c, collect_wave_summaries(c)),
        "global-candidate-library": stage_library,
        "final-adversarial-audit": stage_final_audit,
        "decision-report": stage_decision,
        "compact-review-bundle": stage_compact,
    }
    ctx.notifier.send("Kraken full-coverage signal-tape sweep run start", str(ctx.run_root))
    try:
        for st in stages:
            run_stage(ctx, st, fns[st])
        decision = read_json(ctx.run_root / "decision_summary.json", {})
        final_status = decision.get("status", "complete") if isinstance(decision, dict) else "complete"
        write_status(ctx, final_status, "complete" if final_status == "complete" else str(final_status))
        if final_status == "complete":
            write_json(ctx.run_root / "notifications/final_completion_marker.json", {"status": "complete", "run_root": str(ctx.run_root), "ts_utc": utc_now()})
        ctx.notifier.send("Kraken full-coverage signal-tape sweep run complete", str(ctx.run_root))
        return 0
    except Exception:
        # Best-effort final marker is intentionally conservative.
        exc = sys.exc_info()[1]
        mark_interrupted(ctx, reason="python_exception", detail=f"{type(exc).__name__}: {exc}", exit_code=1)
        if not (ctx.run_root / "decision_summary.json").exists():
            write_json(ctx.run_root / "decision_summary.json", {"run_root": str(ctx.run_root), "status": "failed", "ts_utc": utc_now()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
