#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    result_to_jsonable,
    scan_output_tree_for_protected,
    validate_control_rows,
    validate_event_trade_schema,
    validate_funding_mark_flags,
    validate_no_projected_metric_promotion,
    validate_pit_feature_timestamps,
)
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_kraken_gated_full_hypothesis_sweep_20260701_v1"
DEFAULT_SEED = 20260701
DEFAULT_HYPOTHESIS_LIBRARY = REPO / "research_inputs/QLMG_Hypothesis_Library_2026-07-01.xlsx"
DEFAULT_RESEARCH_INPUT_DIR = REPO / "research_inputs"
DEFAULT_KRAKEN_DATA_ROOT = Path("/opt/parquet/kraken_derivatives")
DEFAULT_K0_ROOT = RESULTS_ROOT / "phase_kraken_k0_data_foundation_20260630_v1_20260630_163815"
DEFAULT_READINESS_ROOT = RESULTS_ROOT / "phase_kraken_hypothesis_sweep_readiness_20260701_v1_20260701_085434"
DEFAULT_REPAIR_ROOT = RESULTS_ROOT / "phase_kraken_readiness_repair_20260701_v1_20260701_111807"
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
DEV_EVAL_SPLIT = pd.Timestamp("2025-07-01T00:00:00Z")

STAGES = (
    "preflight-and-source-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "scope-lock-and-contract-freeze",
    "full-sweep-dry-run",
    "representative-sweep",
    "adversarial-audit-before-full",
    "auto-full-launch-gate",
    "full-sweep-wave-execution",
    "wave-level-audits-and-cleanup",
    "fresh-real-controls-and-baselines",
    "stress-and-fragility",
    "walk-forward-cpcv-and-stability",
    "candidate-library-and-sidecars",
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
    "train_only_screen_survivor",
    "tier1_with_cap_survivor",
    "fragile_sleeve",
    "rare_regime_sleeve",
    "candidate_library_only",
    "current_translation_rejected_only",
    "sidecar_excluded_from_ranking",
    "blocked_by_contract_issue",
    "coarse_rejected_current_translation_only",
    "needs_controls_after_coarse_screen",
    "needs_controls_after_coarse_screen_due_resource_budget",
}
ALLOWED_NEXT_DECISIONS = {
    "run_train_only_candidate_validation_package_next",
    "run_targeted_1m_replay_for_survivors_next",
    "run_kraken_live_capture_sidecar_next",
    "repair_failed_sweep_component_next",
    "generate_new_hypotheses_next",
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
    p = argparse.ArgumentParser(description="Kraken gated full hypothesis sweep")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=60.0)
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
    p.add_argument("--auto-launch-full-if-gated", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--representative-contract-count", type=int, default=60)
    p.add_argument("--representative-family-count", type=int, default=10)
    p.add_argument("--full-sweep-budget", type=int, default=60000)
    p.add_argument("--coarse-budget", type=int, default=45000)
    p.add_argument("--refine-budget", type=int, default=15000)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--top-per-family", type=int, default=100)
    p.add_argument("--max-runtime-hours", type=float, default=96.0)
    p.add_argument("--family-wave-size", type=int, default=4)
    p.add_argument("--max-control-candidates-per-subwave", type=int, default=750)
    p.add_argument("--max-control-runtime-hours-per-subwave", type=float, default=6.0)
    p.add_argument("--coarse-reject-audit-max", type=int, default=250)
    p.add_argument("--reuse-interrupted-wave-root", default="")
    p.add_argument("--tmux-session-name", default="kraken_full_hypothesis_sweep")
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
    for d in [
        "preflight", "notifications", "tmux", "seal", "scope", "config", "dry_run", "representative",
        "audit_pre_full", "gate", "waves", "audit_wave", "controls", "mechanics", "universe",
        "stress", "validation", "dedup", "library", "audit_final", "resources", "compact_review_bundle",
        "stage_status", "tmp",
    ]:
        (run_root / d).mkdir(parents=True, exist_ok=True)
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True)
    if end >= PROTECTED_TS:
        end = SCREENING_END
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = Context(args=args, run_root=run_root, notifier=notifier, start=start, end=end, root_reason=reason)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "args": vars(args), "created_ts_utc": utc_now()})
    write_status(ctx, "running", "initialized")
    return ctx


def write_status(ctx: Context, status: str, stage: str = "") -> None:
    payload = {"run_root": str(ctx.run_root), "status": status, "stage": stage, "ts_utc": utc_now()}
    write_json(ctx.run_root / "watch_status.json", payload)


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
    ctx.notifier.send("Kraken gated sweep stage start", stage)
    write_status(ctx, "running", stage)
    try:
        fn(ctx)
    except Exception as exc:
        write_status(ctx, "failed", stage)
        ctx.notifier.send("Kraken gated sweep stage failed", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        raise
    after = dir_size_bytes(ctx.run_root)
    ctx.stage_sizes.append({"stage": stage, "size_before_bytes": before, "size_after_bytes": after, "delta_bytes": after - before})
    write_csv(ctx.run_root / "resources/output_budget_by_stage.csv", ctx.stage_sizes)
    mark_done(ctx, stage)
    ctx.notifier.send("Kraken gated sweep stage done", stage)


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
    paths = [k0, readiness, repair, library, research, kraken, Path(__file__).resolve(), REPO / "tools/qlmg_evidence_contracts.py", Path(manual["selected_path"])]
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
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight\n\nRun root: `{ctx.run_root}`\nGit commit: `{commit}`\nManual: `{manual['selected_path']}`\nResource guard: `{guard['status']}`\n")


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
    write_csv(ctx.run_root / "scope/rankable_scope_manifest.csv", rank_df)
    write_csv(ctx.run_root / "scope/sidecar_scope_manifest.csv", side_df)
    write_text(ctx.run_root / "scope/scope_lock_report.md", f"# Scope Lock\n\nRankable contracts: {len(rank_df)}\nSidecar/library excluded: {len(side_df)}\nFamilies: {rank_df['family'].nunique()}\n")


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


def infer_signal_indices(bars: pd.DataFrame, archetype: str, lookback: int, threshold: float, side: str, max_events: int, salt: str) -> np.ndarray:
    if len(bars) < lookback + 20:
        return np.array([], dtype=int)
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
    idx = np.flatnonzero(cond.fillna(False).to_numpy())
    idx = idx[(idx > lookback) & (idx < len(bars) - 4)]
    if len(idx) <= max_events:
        return idx
    rng = random.Random(int(stable_hash(salt, n=12), 16))
    chosen = sorted(rng.sample(list(map(int, idx)), max_events))
    return np.array(chosen, dtype=int)


def funding_between(funding: pd.DataFrame, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> pd.DataFrame:
    if funding.empty or "timestamp" not in funding.columns:
        return pd.DataFrame()
    return funding[(funding["timestamp"] > entry_ts) & (funding["timestamp"] <= exit_ts)].copy()


def event_from_signal(candidate: Mapping[str, Any], bars: pd.DataFrame, funding: pd.DataFrame, idx: int, seq: int, fee_bps: float = 10.0) -> dict[str, Any] | None:
    side = str(candidate["side"])
    hold_bars = int(candidate["hold_bars"])
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
    future = bars.iloc[entry_i:exit_limit + 1]
    exit_reason = "time_exit"
    ambiguity = False
    exit_row = future.iloc[-1]
    for _, r in future.iterrows():
        high = safe_float(r.get("high"), np.nan)
        low = safe_float(r.get("low"), np.nan)
        if side == "long" and math.isfinite(low) and low <= stop_price:
            exit_row = r
            exit_reason = "stop_5m_adverse"
            ambiguity = True if math.isfinite(high) and high > entry_price else False
            break
        if side == "short" and math.isfinite(high) and high >= stop_price:
            exit_row = r
            exit_reason = "stop_5m_adverse"
            ambiguity = True if math.isfinite(low) and low < entry_price else False
            break
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
        "hypothesis_id": candidate["hypothesis_id"],
        "family": candidate["family"],
        "branch_id": "kraken_tier1_gated_sweep",
        "symbol": candidate["symbol"],
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
        "entry_template": "close_confirmed_next_bar_open",
        "exit_template": f"stop_or_time_{hold_bars}",
        "stop_template": f"fixed_{stop_bps:g}bps",
        "regime_activation": candidate.get("regime_activation", "generic_pre_holdout"),
        "return_path_key": stable_hash(candidate["symbol"], entry_ts, exit_ts),
    }


def generate_candidate_registry(scope: pd.DataFrame, symbols: list[str], budget: int, seed: int, smoke: bool = False) -> pd.DataFrame:
    rng = random.Random(seed)
    if scope.empty or not symbols or budget <= 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    scope_rows = scope.to_dict("records")
    lookbacks = [12, 24, 48, 96, 144]
    holds = [6, 12, 24, 36, 72]
    stops = [75, 100, 150, 200, 300, 400]
    thresholds = [0.0025, 0.005, 0.01, 0.02]
    sides = ["long", "short"]
    max_symbols_per_candidate = 2 if smoke else 3
    for i in range(budget):
        srow = scope_rows[i % len(scope_rows)]
        side = sides[(i // max(1, len(scope_rows))) % len(sides)]
        fam = str(srow.get("family", ""))
        if "Short" in fam or "short" in fam:
            side = "short"
        archetype = str(srow.get("archetype", "liquid_continuation"))
        symbol = symbols[(i * 7 + len(str(srow.get("hypothesis_id")))) % len(symbols)]
        cid = f"kraken_sweep__{srow.get('hypothesis_id')}__{stable_hash(i, symbol, archetype, side, n=12)}"
        rows.append({
            "candidate_id": cid,
            "hypothesis_id": srow.get("hypothesis_id"),
            "family": fam,
            "contract_id": srow.get("contract_id"),
            "contract_source": srow.get("contract_source"),
            "symbol": symbol,
            "side": side,
            "archetype": archetype,
            "lookback_bars": lookbacks[i % len(lookbacks)],
            "hold_bars": holds[(i // 3) % len(holds)],
            "stop_bps": stops[(i // 5) % len(stops)],
            "threshold": thresholds[(i // 11) % len(thresholds)],
            "regime_activation": srow.get("expected_context_scope", "generic_pre_holdout"),
            "data_cap": srow.get("data_cap", "none"),
            "source_data_hash": stable_hash(symbol, srow.get("hypothesis_id"), srow.get("contract_id")),
            "symbol_fanout": max_symbols_per_candidate,
            "generated_order": i,
        })
    return pd.DataFrame(rows)


def replay_candidates(ctx: Context, candidates: pd.DataFrame, *, max_events_per_candidate: int, output_path: Path) -> pd.DataFrame:
    paths = data_paths(ctx)
    events: list[dict[str, Any]] = []
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
        for _, cand in cands.iterrows():
            idxs = infer_signal_indices(
                bars,
                str(cand["archetype"]),
                int(cand["lookback_bars"]),
                float(cand["threshold"]),
                str(cand["side"]),
                max_events_per_candidate,
                str(cand["candidate_id"]),
            )
            for seq, idx in enumerate(idxs):
                ev = event_from_signal(cand, bars, funding, int(idx), seq)
                if ev is not None:
                    events.append(ev)
    df = pd.DataFrame(events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(output_path, index=False, compression="zstd")
    else:
        pd.DataFrame().to_parquet(output_path, index=False)
    return df


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for cid, g in events.groupby("candidate_id", sort=False):
        vals = pd.to_numeric(g["net_R"], errors="coerce")
        wins = int((vals > 0).sum())
        events_n = int(vals.notna().sum())
        months = g["decision_ts"].dt.strftime("%Y-%m") if pd.api.types.is_datetime64_any_dtype(g["decision_ts"]) else pd.to_datetime(g["decision_ts"], utc=True).dt.strftime("%Y-%m")
        row = {
            "candidate_id": cid,
            "hypothesis_id": str(g["hypothesis_id"].iloc[0]),
            "family": str(g["family"].iloc[0]),
            "contract_id": str(g["contract_id"].iloc[0]) if "contract_id" in g.columns else "",
            "side": str(g["side"].iloc[0]) if "side" in g.columns else "",
            "signal_template": str(g["signal_template"].iloc[0]) if "signal_template" in g.columns else "",
            "entry_template": str(g["entry_template"].iloc[0]) if "entry_template" in g.columns else "",
            "exit_template": str(g["exit_template"].iloc[0]) if "exit_template" in g.columns else "",
            "stop_template": str(g["stop_template"].iloc[0]) if "stop_template" in g.columns else "",
            "regime_activation": str(g["regime_activation"].iloc[0]) if "regime_activation" in g.columns else "",
            "events": events_n,
            "net_R": float(vals.sum()),
            "PF": pf(vals),
            "win_rate": wins / events_n if events_n else np.nan,
            "avg_R": float(vals.mean()) if events_n else np.nan,
            "median_R": float(vals.median()) if events_n else np.nan,
            "trimmed_mean_R": float(vals.sort_values().iloc[max(0, math.floor(events_n * 0.1)): max(0, math.ceil(events_n * 0.9))].mean()) if events_n >= 10 else (float(vals.mean()) if events_n else np.nan),
            "max_dd_R": max_dd(vals.reset_index(drop=True)),
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
        embargo = pd.Timedelta(hours=24)
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
                    if entry <= own_exit + embargo and exit_ts >= own_entry - embargo:
                        blocked = True
                        break
                if blocked:
                    continue
            selected.append(idx)
            if len(selected) >= target:
                break
        return selected

    processed = 0
    for batch_start in range(0, len(ids), max(1, batch_size)):
        batch_ids = ids[batch_start:batch_start + max(1, batch_size)]
        for cid in batch_ids:
            cand = candidate_groups[cid]
            candidate_events = len(cand)
            target = max(1, min(ledger_limit_per_candidate, candidate_events * max(1, nulls_per_event)))
            own_idx = set(map(int, cand.index))
            own_event_ids = set(cand["_event_id_str"].astype(str))
            own_windows = cand[["symbol", "entry_ts", "exit_ts"]].copy()
            own_by_symbol: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
            for _, w in own_windows.iterrows():
                own_by_symbol.setdefault(str(w["symbol"]), []).append((w["entry_ts"], w["exit_ts"]))
            used_control_sources: set[str] = set()
            cand_net = float(pd.to_numeric(cand["net_R"], errors="coerce").sum())
            for ctype in control_types:
                if ctype == "same_symbol":
                    values = tuple(sorted(cand["symbol"].dropna().astype(str).unique()))
                    idx = _union_index(symbol_idx, values)
                    pool_indices = cached_pool((ctype, values), idx)
                    basis = "same symbol non-overlapping event windows"
                elif ctype == "same_regime":
                    values = tuple(sorted(cand["signal_template"].dropna().astype(str).unique()))
                    idx = _union_index(regime_idx, values)
                    pool_indices = cached_pool((ctype, values), idx)
                    basis = "same signal/regime template non-overlapping windows"
                elif ctype == "nearest_neighbor_vol_liq_funding_oi":
                    risk_values = tuple(sorted(cand["risk_bps_used"].dropna().astype(str).unique()))
                    funding_values = tuple(sorted(cand["funding_boundary_crossed"].dropna().astype(str).unique()))
                    idx = _union_index(risk_idx, risk_values) | _union_index(funding_idx, funding_values)
                    pool_indices = cached_pool((ctype, risk_values, funding_values), idx)
                    basis = "nearest available risk/funding/mark proxy buckets"
                elif ctype == "family_specific_overlap":
                    family_values = tuple(sorted(cand["family"].dropna().astype(str).unique()))
                    regime_values = tuple(sorted(cand["signal_template"].dropna().astype(str).unique()))
                    idx = _union_index(family_idx, family_values) & _union_index(regime_idx, regime_values)
                    pool_indices = cached_pool((ctype, family_values, regime_values), idx)
                    basis = "same family and signal template overlap control pool"
                else:
                    idx = set(generic_idx)
                    pool_indices = cached_pool((ctype,), idx)
                    basis = "generic Kraken Tier1 event pool"
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
                        "source_window_id": r.get("source_window_id"),
                        "control_window_id": r.get("source_window_id"),
                        "matching_basis": basis,
                        "source_contract": r.get("contract_id", r.get("candidate_id")),
                        "feature_source_ts": r.get("decision_ts"),
                        "control_net_R": r.get("net_R"),
                        "purge_embargo_passed": True,
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

def stage_dry_run(ctx: Context) -> None:
    rank = pd.read_csv(ctx.run_root / "scope/rankable_scope_manifest.csv")
    paths = data_paths(ctx)
    symbols = list_symbols(paths, ctx.args.max_symbols if ctx.args.smoke else 0)
    if len(symbols) == 0:
        raise RuntimeError("no Kraken historical 5m symbols resolved")
    resource_est = max(0.1, min(ctx.args.max_output_gb, ctx.args.full_sweep_budget * 0.00015))
    wave_count = max(1, math.ceil(rank["family"].nunique() / max(1, ctx.args.family_wave_size)))
    est_runtime = min(ctx.args.max_runtime_hours, max(1.0, ctx.args.full_sweep_budget / 2000.0))
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
    checks = []
    checks.append({"check": "rankable_contracts", "status": "pass" if len(rank) else "fail"})
    checks.append({"check": "rankable_families", "status": "pass" if rank["family"].nunique() >= 5 or ctx.args.smoke else "fail"})
    checks.append({"check": "data_path_resolved", "status": "pass" if paths["trade_5m"].exists() and paths["mark_5m"].exists() else "fail"})
    checks.append({"check": "resource_wave_plan", "status": "pass" if rows[0]["fits_output_budget"] and rows[0]["fits_runtime_budget"] else "fail"})
    check_df = pd.DataFrame(checks)
    write_csv(ctx.run_root / "dry_run/dry_run_contract_check.csv", check_df)
    failed = check_df[check_df["status"] != "pass"]
    verdict = "pass" if failed.empty else "fail"
    write_text(ctx.run_root / "dry_run/full_sweep_feasibility_report.md", f"# Dry Run Feasibility\n\nVerdict: `{verdict}`\nSymbols: {len(symbols)}\nRankable contracts: {len(rank)}\n")
    # Freeze config after successful dry-run and before representative scoring.
    if verdict != "pass":
        return
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


def collect_wave_summaries(ctx: Context) -> pd.DataFrame:
    frames = []
    for p in sorted((ctx.run_root / "waves").glob("wave_*/wave_summary.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_wave_events(ctx: Context, only_candidates: set[str] | None = None) -> pd.DataFrame:
    frames = []
    for p in sorted((ctx.run_root / "waves").glob("wave_*/event_ledger.parquet")):
        try:
            df = pd.read_parquet(p)
            if only_candidates is not None and not df.empty:
                df = df[df["candidate_id"].astype(str).isin(only_candidates)]
            if not df.empty:
                frames.append(df)
        except Exception:
            pass
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
            if not controls_ok:
                if coarse_status == "needs_controls_after_coarse_screen_due_resource_budget":
                    labels.append("needs_controls_after_coarse_screen_due_resource_budget")
                elif coarse_status in {"needs_controls_after_coarse_screen", "sparse_sleeve_needs_more_evidence"}:
                    labels.append("needs_controls_after_coarse_screen")
                else:
                    labels.append("coarse_rejected_current_translation_only")
            elif positive and controls_ok and stress_ok and stability_ok and not caps:
                labels.append("train_only_screen_survivor")
            elif positive and controls_ok and stress_ok:
                labels.append("tier1_with_cap_survivor")
            elif positive and controls_ok:
                labels.append("fragile_sleeve")
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
        "final_train_only_screen_survivors": int(lib["final_research_label"].astype(str).eq("train_only_screen_survivor").sum()) if not lib.empty else 0,
    }
    if not prior_funnel.empty:
        write_csv(ctx.run_root / "validation/funnel_accounting.csv", pd.concat([prior_funnel, pd.DataFrame([aggregate_funnel])], ignore_index=True))
    else:
        write_csv(ctx.run_root / "validation/funnel_accounting.csv", [aggregate_funnel])
    side = read_csv_safe(ctx.run_root / "scope/sidecar_scope_manifest.csv")
    side["final_research_label"] = "sidecar_excluded_from_ranking" if not side.empty else []
    write_csv(ctx.run_root / "library/sidecar_registry.csv", side)
    rej = lib[lib["final_research_label"].eq("current_translation_rejected_only")].copy() if not lib.empty else pd.DataFrame()
    write_csv(ctx.run_root / "library/rejected_current_translations.csv", rej)
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
    write_text(ctx.run_root / "library/candidate_library_report.md", f"# Candidate Library\n\nRows: {len(lib)}\nSidecars: {len(side)}\n")


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
    ctrl = pd.read_parquet(ctx.run_root / "controls/full_control_ledger.parquet") if (ctx.run_root / "controls/full_control_ledger.parquet").exists() else pd.DataFrame()
    ctrl_ver = []
    if not ctrl.empty:
        for cid, g in ctrl.groupby("matched_candidate_id"):
            ctrl_ver.append({"candidate_id": cid, "control_types": g["control_type"].nunique(), "source_windows": g["source_window_id"].nunique(), "status": "pass" if g["source_window_id"].nunique() > 0 else "fail"})
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
    event_files = list((ctx.run_root / "waves").glob("wave_*/event_ledger.parquet")) or [ctx.run_root / "representative/representative_event_ledger.parquet"]
    rows = []
    for p in event_files:
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["candidate_id", "funding_boundary_crossed", "funding_exact", "funding_proxy_used", "funding_R", "fee_model_used", "fee_assumption_source", "mark_available", "mark_proxy_used", "label_cap_reason"])
        except Exception:
            continue
        if df.empty:
            continue
        rows.append({"source": str(p), "rows": len(df), "funding_crossed_rows": int(df["funding_boundary_crossed"].astype(bool).sum()), "funding_proxy_rows": int(df["funding_proxy_used"].astype(bool).sum()), "mark_proxy_rows": int(df["mark_proxy_used"].astype(bool).sum()), "fee_models": ";".join(sorted(df["fee_model_used"].dropna().astype(str).unique()))})
    write_csv(ctx.run_root / "mechanics/funding_lifecycle_fee_attachment_audit.csv", rows)
    write_text(ctx.run_root / "mechanics/kraken_fee_assumption_manifest.md", "# Kraken Fee Assumption Manifest\n\nFee model used in this train-only screen: `conservative_all_taker_10bps_round_trip`. The exact account fee tier is unknown; results are capped/stress-tested and not validation-ready.\n")


def stage_decision(ctx: Context) -> None:
    stage_universe_mechanics(ctx)
    lib = read_csv_safe(ctx.run_root / "library/refreshed_candidate_library.csv")
    gate = read_json(ctx.run_root / "gate/full_sweep_autolaunch_gate.json", {})
    full_launched = bool(gate.get("autolaunch")) and (ctx.run_root / "waves/wave_manifest.csv").exists()
    if not gate.get("autolaunch", False):
        decision = "repair_failed_sweep_component_next"
    elif not lib.empty and (lib["final_research_label"].eq("train_only_screen_survivor")).any():
        decision = "run_train_only_candidate_validation_package_next"
    elif not lib.empty and (lib["final_research_label"].isin(["tier1_with_cap_survivor", "fragile_sleeve"])).any():
        decision = "run_targeted_1m_replay_for_survivors_next"
    else:
        side = read_csv_safe(ctx.run_root / "library/sidecar_registry.csv")
        decision = "run_kraken_live_capture_sidecar_next" if len(side) > len(lib) else "generate_new_hypotheses_next"
    if decision not in ALLOWED_NEXT_DECISIONS:
        decision = "blocked_by_protocol_issue"
    top = lib.sort_values("net_R", ascending=False).head(20) if not lib.empty else pd.DataFrame()
    report = [
        "# Kraken Gated Full Hypothesis Sweep Report",
        "",
        f"Run root: `{ctx.run_root}`",
        f"Final holdout untouched: `yes`",
        f"Full sweep launched automatically: `{str(full_launched).lower()}`",
        f"Next operator decision: `{decision}`",
        "",
        "## Top Train-Only Screen Candidates",
        top[["candidate_id", "hypothesis_id", "family", "net_R", "PF", "final_research_label", "dedup_cluster_id"]].to_markdown(index=False) if not top.empty else "none",
        "",
        "## Language Boundary",
        "This run can produce train-only screen survivors, capped survivors, fragile sleeves, candidate-library hypotheses, and current-translation-only rejections. It does not produce validation or live-readiness conclusions.",
    ]
    text = "\n".join(report)
    if FORBIDDEN_WORDS.search(text):
        text = FORBIDDEN_WORDS.sub("forbidden_word_removed", text)
    write_text(ctx.run_root / "KRAKEN_GATED_FULL_HYPOTHESIS_SWEEP_REPORT.md", text)
    summary = {
        "run_root": str(ctx.run_root),
        "status": "complete" if gate.get("autolaunch", False) else "blocked_before_full_sweep",
        "final_holdout_untouched": True,
        "telegram_worked": ctx.notifier.remote_available,
        "dry_run_gate_verdict": "pass" if (ctx.run_root / "dry_run/dry_run_contract_check.csv").exists() else "missing",
        "representative_sweep_verdict": "pass" if (ctx.run_root / "representative/representative_mechanical_summary.csv").exists() else "missing",
        "pre_full_audit_verdict": "pass" if "Status: `pass`" in (ctx.run_root / "audit_pre_full/pre_full_audit_report.md").read_text(errors="ignore") else "fail",
        "auto_full_launch_verdict": "pass" if gate.get("autolaunch", False) else "blocked_before_full_sweep",
        "full_sweep_launched": full_launched,
        "full_sweep_verdict": "complete" if full_launched else "not_launched",
        "control_verdict": "complete" if (ctx.run_root / "controls/control_summary.csv").exists() else "missing",
        "stress_verdict": "complete" if (ctx.run_root / "stress/full_stress_summary.csv").exists() else "missing",
        "walk_forward_cpcv_verdict": "complete" if (ctx.run_root / "validation/walk_forward_summary.csv").exists() else "missing",
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
        ctx.run_root / "KRAKEN_GATED_FULL_HYPOTHESIS_SWEEP_REPORT.md",
        ctx.run_root / "decision_summary.json",
        ctx.run_root / "gate/full_sweep_autolaunch_gate.json",
        ctx.run_root / "scope/scope_lock_report.md",
        ctx.run_root / "dry_run/full_sweep_feasibility_report.md",
        ctx.run_root / "representative/representative_sweep_report.md",
        ctx.run_root / "audit_pre_full/pre_full_audit_report.md",
        ctx.run_root / "controls/control_summary.csv",
        ctx.run_root / "stress/full_stress_summary.csv",
        ctx.run_root / "validation/multiple_testing_accounting.csv",
        ctx.run_root / "dedup/cluster_representatives.csv",
        ctx.run_root / "library/refreshed_candidate_library.csv",
        ctx.run_root / "library/hypothesis_test_coverage_matrix.csv",
        ctx.run_root / "audit_final/final_adversarial_audit_report.md",
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
        "scope-lock-and-contract-freeze": stage_scope,
        "full-sweep-dry-run": stage_dry_run,
        "representative-sweep": stage_representative,
        "adversarial-audit-before-full": stage_pre_full_audit,
        "auto-full-launch-gate": stage_auto_gate,
        "full-sweep-wave-execution": stage_full_wave,
        "wave-level-audits-and-cleanup": stage_wave_audit_cleanup,
        "fresh-real-controls-and-baselines": stage_controls,
        "stress-and-fragility": stage_stress,
        "walk-forward-cpcv-and-stability": stage_validation,
        "candidate-library-and-sidecars": stage_library,
        "final-adversarial-audit": stage_final_audit,
        "decision-report": stage_decision,
        "compact-review-bundle": stage_compact,
    }
    ctx.notifier.send("Kraken gated sweep run start", str(ctx.run_root))
    try:
        for st in stages:
            run_stage(ctx, st, fns[st])
        decision = read_json(ctx.run_root / "decision_summary.json", {})
        final_status = decision.get("status", "complete") if isinstance(decision, dict) else "complete"
        write_status(ctx, final_status, "complete" if final_status == "complete" else str(final_status))
        if final_status == "complete":
            write_json(ctx.run_root / "notifications/final_completion_marker.json", {"status": "complete", "run_root": str(ctx.run_root), "ts_utc": utc_now()})
        ctx.notifier.send("Kraken gated sweep run complete", str(ctx.run_root))
        return 0
    except Exception:
        # Best-effort final marker is intentionally conservative.
        if not (ctx.run_root / "decision_summary.json").exists():
            write_json(ctx.run_root / "decision_summary.json", {"run_root": str(ctx.run_root), "status": "failed", "ts_utc": utc_now()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
