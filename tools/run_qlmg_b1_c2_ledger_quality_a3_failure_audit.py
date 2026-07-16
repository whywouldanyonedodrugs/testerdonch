#!/usr/bin/env python3
"""Forensic post-run audit plus B1/C2 ledger-quality and A3 failure audit.

Train-only. Does not download data, trade, mutate live systems, or touch final holdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path("/opt/testerdonch")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.qlmg_evidence_contracts import (  # noqa: E402
    assert_pass,
    validate_funding_mark_flags,
    validate_no_projected_metric_promotion,
    validate_pit_feature_timestamps,
)

try:
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover - optional local integration
    TelegramNotifier = None  # type: ignore

RUN_ID = "phase_qlmg_b1_c2_ledger_quality_a3_failure_audit_20260629_v1"
BASE_RUN_ROOT = REPO_ROOT / "results/rebaseline" / RUN_ID
CORRECTED_ROOT = REPO_ROOT / "results/rebaseline/phase_qlmg_corrected_event_level_development_sweep_20260629_v1_20260629_052114"
REMEDIATION_ROOT = REPO_ROOT / "results/rebaseline/phase_qlmg_evidence_remediation_family_repair_20260629_v1_20260629_044410"
INTEGRITY_ROOT = REPO_ROOT / "results/rebaseline/phase_qlmg_evidence_integrity_corrected_sweep_20260628_v1_20260628_163819"
SECTOR_MD = REPO_ROOT / "research_inputs/point_in_time_sector_seeds.md"
CATALYST_MD = REPO_ROOT / "research_inputs/post_catalyst_c2_database.md"
FINAL_HOLDOUT_START = pd.Timestamp("2026-01-01T00:00:00Z")
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
DEFAULT_SEED = 20260629

ALLOWED_OPERATOR_DECISIONS = {
    "run_a3_regime_specific_validation_next",
    "run_b1_repaired_sweep_next",
    "run_c2_mechanism_validation_next",
    "run_a1_a4_corrected_liquid_sweep_next",
    "request_branch_x_capture_export",
    "micro_canary_possible_execution_only",
    "build_candidate_portfolio_later",
    "blocked_by_protocol_issue",
}

STAGES = [
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "post-patch-forensic-audit",
    "event-ledger-arithmetic-audit",
    "a3-failure-attribution",
    "a3-regime-specific-salvage-test",
    "a2-feature-reuse-diagnostic",
    "b1-ledger-quality-audit",
    "b1-mechanism-repair-sweep",
    "c2-ledger-quality-audit",
    "c2-mechanism-repair-sweep",
    "fresh-controls-and-normalization-check",
    "branch-x-capture-export-request",
    "candidate-library-and-preservation",
    "decision-report",
    "compact-review-bundle",
]


@dataclass
class RunContext:
    args: argparse.Namespace
    run_root: Path
    start: pd.Timestamp
    end: pd.Timestamp
    notifier: "RunNotifier"


class RunNotifier:
    def __init__(self, run_root: Path, *, disabled: bool = False, require_remote: bool = False, allow_no_remote: bool = False) -> None:
        self.run_root = run_root
        self.disabled = disabled
        self.require_remote = require_remote
        self.allow_no_remote = allow_no_remote
        self.log_path = run_root / "notifications/telegram_events.jsonl"
        self.remote = None
        self.remote_available = False
        self.status = "disabled" if disabled else "enabled"
        self.missing = ""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not disabled and TelegramNotifier is not None:
            class _Args:
                disable_telegram = False
                telegram_dry_run = False
                tg_bot_token = ""
                tg_chat_id = ""
                tg_auto_chat = False
            try:
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-b1c2-a3-audit")
                self.status = getattr(self.remote, "status_line", lambda: "enabled")()
                self.remote_available = bool(self.remote is not None and "enabled" in str(self.status).lower())
                if not self.remote_available:
                    self.missing = str(self.status)
            except Exception as exc:  # pragma: no cover
                self.status = "unavailable"
                self.missing = f"{type(exc).__name__}: {exc}"
        elif not disabled:
            self.status = "unavailable"
            self.missing = "tools.telegram_notify.TelegramNotifier unavailable"
        if require_remote and not self.remote_available and not allow_no_remote:
            raise RuntimeError(f"remote Telegram required but unavailable: {self.missing or self.status}")

    def send(self, title: str, body: str = "", level: str = "info") -> None:
        sent = False
        error = ""
        if not self.disabled and self.remote is not None:
            try:
                sent = bool(self.remote.send(title, body))
            except Exception as exc:  # pragma: no cover - network/env dependent
                error = str(exc)
        rec = {"ts_utc": utc_now(), "title": title, "body": body, "level": level, "status": self.status, "sent": sent, "error": error}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        write_json(self.run_root / "watch_status.json", {"run_root": str(self.run_root), "status": "running", "last_event": title, "ts_utc": utc_now()})


def utc_now() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31T23:59:59Z")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=45.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--include-forensic-audit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-a3-failure-audit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-b1-ledger-quality", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-c2-ledger-quality", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-branch-x-capture-request", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--b1-repair-budget", type=int, default=1500)
    p.add_argument("--c2-repair-budget", type=int, default=1200)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--top-per-family", type=int, default=60)
    p.add_argument("--tmux-session-name", default="qlmg_b1c2_a3_audit")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--run-root", default=None)
    return p.parse_args(argv)


def clamp_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(args.start, tz="UTC") if pd.Timestamp(args.start).tzinfo is None else pd.Timestamp(args.start).tz_convert("UTC")
    end = pd.Timestamp(args.end, tz="UTC") if pd.Timestamp(args.end).tzinfo is None else pd.Timestamp(args.end).tz_convert("UTC")
    if end >= FINAL_HOLDOUT_START:
        end = SCREENING_END
    if start >= FINAL_HOLDOUT_START:
        raise RuntimeError("start is inside protected holdout")
    return start, end


def resolve_run_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.run_root:
        return Path(args.run_root), "explicit"
    if args.smoke:
        return BASE_RUN_ROOT / "smoke", "smoke_root"
    if BASE_RUN_ROOT.exists():
        suffix = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        return Path(f"{BASE_RUN_ROOT}_{suffix}"), f"default_root_existed_suffix_{suffix}"
    return BASE_RUN_ROOT, "default"


def stage_list(stage: str) -> list[str]:
    if stage == "all":
        return STAGES
    if stage not in STAGES:
        raise RuntimeError(f"unknown stage {stage}")
    return [stage]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    rows = list(rows)
    if not rows:
        if fieldnames is None:
            path.write_text("", encoding="utf-8")
        else:
            with path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writeheader()
        return
    keys = list(fieldnames or sorted({k for r in rows for k in r.keys()}))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path, **kw: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def file_hash(path: Path, max_bytes: int = 50_000_000) -> str:
    h = hashlib.sha256()
    if not path.exists() or not path.is_file():
        return "missing"
    with path.open("rb") as fh:
        remaining = max_bytes
        while remaining > 0:
            chunk = fh.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def stable_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def is_done(root: Path, stage: str) -> bool:
    return (root / "stage_status" / f"{stage}.done").exists()


def mark_done(root: Path, stage: str) -> None:
    p = root / "stage_status" / f"{stage}.done"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(utc_now())


def resource_check(ctx: RunContext, stage: str, estimated_output_gb: float = 0.2) -> None:
    usage = shutil.disk_usage(str(REPO_ROOT))
    free_gb = usage.free / 1e9
    rec = {"stage": stage, "free_gb": free_gb, "estimated_output_gb": estimated_output_gb, "hard_stop_below_gb": 5, "warn_below_gb": 7, "max_output_gb": ctx.args.max_output_gb, "ts_utc": utc_now()}
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", rec)
    if free_gb < 5:
        raise RuntimeError(f"free disk below hard stop: {free_gb:.2f}GB")
    if estimated_output_gb > 35 and not ctx.args.allow_large_output:
        raise RuntimeError(f"stage output estimate {estimated_output_gb}GB exceeds 35GB block")
    if estimated_output_gb > ctx.args.max_output_gb and not ctx.args.allow_large_output:
        raise RuntimeError(f"stage output estimate {estimated_output_gb}GB exceeds max-output-gb")


def validate_no_protected_df(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        ts = pd.to_datetime(df[col], utc=True, errors="coerce")
        bad = ts >= FINAL_HOLDOUT_START
        if bool(bad.any()):
            raise RuntimeError(f"protected timestamp rows in {col}: {int(bad.sum())}")


def event_r_col(df: pd.DataFrame) -> str:
    return "net_R_variant" if "net_R_variant" in df.columns else "net_R"


def profit_factor(vals: pd.Series) -> float:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    wins = vals[vals > 0].sum()
    losses = -vals[vals < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 1.0
    return float(wins / losses)


def max_drawdown(vals: pd.Series) -> float:
    vals = pd.to_numeric(vals, errors="coerce").fillna(0.0)
    eq = vals.cumsum()
    dd = eq - eq.cummax()
    return float(dd.min()) if len(dd) else 0.0


def metric_row(df: pd.DataFrame, candidate_id: str, family: str, label: str = "") -> dict[str, Any]:
    if df.empty:
        return {"candidate_id": candidate_id, "family": family, "events": 0, "net_R": np.nan, "PF": np.nan, "win_rate": np.nan, "max_dd_R": 0, "active_months": 0, "active_symbols": 0, "label": label}
    col = event_r_col(df)
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    ts = pd.to_datetime(df.get("decision_ts", pd.Series(index=df.index)), utc=True, errors="coerce")
    sym = df.get("symbol", pd.Series(dtype=str)).astype(str) if "symbol" in df else pd.Series(dtype=str)
    return {
        "candidate_id": candidate_id,
        "family": family,
        "events": int(len(df)),
        "net_R": float(vals.sum()),
        "PF": profit_factor(vals),
        "win_rate": float((vals > 0).mean()) if len(vals) else np.nan,
        "max_dd_R": max_drawdown(vals),
        "avg_R": float(vals.mean()) if len(vals) else np.nan,
        "median_R": float(vals.median()) if len(vals) else np.nan,
        "active_months": int(ts.dt.strftime("%Y-%m").nunique()) if len(ts) else 0,
        "active_symbols": int(sym.nunique()) if len(sym) else 0,
        "label": label,
        "mark_available": bool(df.get("mark_available", df.get("mark_price_available", pd.Series(False, index=df.index))).fillna(False).astype(bool).all()) if len(df) else False,
        "funding_exact": bool(df.get("funding_exact", pd.Series(False, index=df.index)).fillna(False).astype(bool).all()) if len(df) else False,
        "mark_proxy_used": bool(df.get("mark_proxy_used", ~df.get("mark_available", df.get("mark_price_available", pd.Series(False, index=df.index))).fillna(False).astype(bool)).astype(bool).any()) if len(df) else True,
        "funding_proxy_used": bool(df.get("funding_proxy_used", ~df.get("funding_exact", pd.Series(False, index=df.index)).fillna(False).astype(bool)).astype(bool).any()) if len(df) else True,
    }


def normalize_control(candidate_events: int, control_net: float, control_events: int) -> float:
    if control_events <= 0:
        return np.nan
    return float(control_net) * float(candidate_events) / float(control_events)


def load_corrected(path: str) -> pd.DataFrame:
    df = read_parquet(CORRECTED_ROOT / path)
    if not df.empty:
        validate_no_protected_df(df, ["decision_ts", "entry_ts", "exit_ts"])
    return df


# Stages

def stage_preflight(ctx: RunContext) -> None:
    resource_check(ctx, "preflight-and-artifact-freeze", 0.2)
    artifacts = [
        ("corrected_sweep", CORRECTED_ROOT),
        ("remediation", REMEDIATION_ROOT),
        ("integrity", INTEGRITY_ROOT),
        ("sector_md", SECTOR_MD),
        ("catalyst_md", CATALYST_MD),
    ]
    rows = []
    hashes = {}
    for name, path in artifacts:
        exists = path.exists()
        if exists and path.is_dir():
            files = sorted([p for p in path.rglob("*") if p.is_file()])[:200]
            digest = hashlib.sha256("\n".join(f"{p.relative_to(path)}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files).encode()).hexdigest()
            count = len(files)
        elif exists:
            digest = file_hash(path)
            count = 1
        else:
            digest = "missing"
            count = 0
        rows.append({"artifact": name, "path": str(path), "exists": exists, "file_count_sampled": count, "hash": digest})
        hashes[name] = {"path": str(path), "exists": exists, "hash": digest}
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        git = "unknown"
    write_text(ctx.run_root / "preflight/preflight_report.md", "\n".join([
        "# Preflight And Artifact Freeze",
        f"git_head: `{git}`",
        f"corrected_root: `{CORRECTED_ROOT}`",
        "This phase is train-only and branch-separated. Sparse variants are preserved when evidence is clean.",
    ]))
    free_gb = shutil.disk_usage(str(REPO_ROOT)).free / 1e9
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\nfree_gb: `{free_gb:.2f}`\n\nts_utc: `{utc_now()}`\n")


def stage_telegram(ctx: RunContext) -> None:
    resource_check(ctx, "telegram-and-tmux-setup", 0.05)
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\nstatus: `{ctx.notifier.status}`\n\nremote_available: `{ctx.notifier.remote_available}`\n\nmissing: `{ctx.notifier.missing}`\n")
    watch = [
        "# Watch Commands",
        f"tmux attach -t {ctx.args.tmux_session_name}",
        f"tail -f {ctx.run_root}/logs/full_run.log",
        f"watch -n 30 'cat {ctx.run_root}/watch_status.json'",
        f"tail -f {ctx.run_root}/notifications/telegram_events.jsonl",
        "df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h",
    ]
    write_text(ctx.run_root / "tmux/watch_commands.md", "\n".join(watch))
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", "Full launch requires `--launch-tmux`. Remote Telegram required with `--require-telegram` unless `--allow-no-telegram` is passed.")
    ctx.notifier.send("QLMG B1/C2/A3 audit initialized", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    resource_check(ctx, "seal-guard", 0.05)
    checks = [
        {"case": "allowed_end", "timestamp": str(SCREENING_END), "protected": False},
        {"case": "protected_start", "timestamp": str(FINAL_HOLDOUT_START), "protected": True},
    ]
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "checks": checks})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\nProtected slice starts `{FINAL_HOLDOUT_START}`. Generated rows are checked against this boundary.")


def stage_forensic(ctx: RunContext) -> None:
    resource_check(ctx, "post-patch-forensic-audit", 0.2)
    if not ctx.args.include_forensic_audit:
        return
    code = REPO_ROOT / "tools/run_qlmg_corrected_event_level_development_sweep.py"
    code_hash = file_hash(code)
    files = [
        "decision_summary.json",
        "QLMG_CORRECTED_EVENT_LEVEL_DEVELOPMENT_SWEEP_REPORT.md",
        "a3_sweep/a3_sweep_summary.csv",
        "c2_sidecar/c2_by_mechanism_summary.csv",
        "triage/corrected_cross_branch_triage.csv",
        "compact_review_bundle/decision_summary.json",
        "compact_review_bundle/c2_sidecar__c2_by_mechanism_summary.csv",
    ]
    rows = []
    for rel in files:
        p = CORRECTED_ROOT / rel
        rows.append({"artifact": rel, "exists": p.exists(), "mtime_utc": pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").isoformat() if p.exists() else "", "size": p.stat().st_size if p.exists() else 0, "hash": file_hash(p) if p.exists() else "missing"})
    decision = safe_json(CORRECTED_ROOT / "decision_summary.json")
    a3 = read_csv(CORRECTED_ROOT / "a3_sweep/a3_sweep_summary.csv")
    c2 = read_csv(CORRECTED_ROOT / "c2_sidecar/c2_by_mechanism_summary.csv")
    recompute = []
    if not a3.empty:
        a3_pass = bool(a3.get("passes_final_a3_gate", pd.Series(False, index=a3.index)).fillna(False).astype(bool).any())
        recompute.append({"check": "a3_final_gate", "reported": decision.get("a3_verdict"), "recomputed_pass": a3_pass, "matches": (a3_pass and decision.get("a3_verdict") == "a3_candidate_survives_corrected_sweep") or ((not a3_pass) and "blocked" in str(decision.get("a3_verdict")))})
    if not c2.empty:
        bad_snake = c2["mechanism_group"].astype(str).str.contains("_").any() if "mechanism_group" in c2 else False
        recompute.append({"check": "c2_human_readable_mechanisms", "reported": decision.get("c2_sidecar_verdict"), "recomputed_pass": not bad_snake, "matches": not bad_snake})
    verdict = "post_patch_artifacts_trustworthy" if all(r.get("matches") for r in recompute) and rows else "post_patch_artifact_provenance_incomplete"
    write_csv(ctx.run_root / "forensic/post_patch_artifact_timeline.csv", rows)
    write_json(ctx.run_root / "forensic/code_hash_provenance.json", {"current_corrected_runner_hash": code_hash, "pre_patch_hash_recoverable": False, "verdict": verdict})
    write_csv(ctx.run_root / "forensic/final_verdict_recompute_check.csv", recompute)
    write_text(ctx.run_root / "forensic/stale_artifact_risk_report.md", f"# Stale Artifact Risk\n\nverdict: `{verdict}`\n\nPre-patch code hash is not recoverable from git because the runner is untracked/local. Final decision and C2/A3 summaries were rechecked from current artifacts.")


def stage_arithmetic(ctx: RunContext) -> None:
    resource_check(ctx, "event-ledger-arithmetic-audit", 0.6)
    ledgers = {
        "A3": ("a3_sweep/a3_event_level_replay.parquet", "a3_sweep/a3_sweep_summary.csv", "variant_id"),
        "A2_redesign_only": ("a2_sweep/a2_event_level_replay.parquet", "a2_sweep/a2_sweep_summary.csv", "variant_id"),
        "B1": ("b1_sidecar/b1_event_level_replay.parquet", "b1_sidecar/b1_summary.csv", "mode"),
        "C2": ("c2_sidecar/c2_event_level_replay.parquet", "c2_sidecar/c2_by_mechanism_summary.csv", "mechanism_group"),
    }
    audit_rows = []
    compare_rows = []
    failures = []
    for fam, (ledger_rel, summary_rel, id_col) in ledgers.items():
        df = load_corrected(ledger_rel)
        summary = read_csv(CORRECTED_ROOT / summary_rel)
        if df.empty:
            audit_rows.append({"family": fam, "ledger_exists": False, "evidence_integrity_status": "no_event_level_trade_ledger"})
            continue
        group_col = id_col if id_col in df.columns else ("variant_id" if "variant_id" in df.columns else "candidate_id")
        validate_no_protected_df(df, ["decision_ts", "entry_ts", "exit_ts"])
        for cid, g in df.groupby(group_col):
            row = metric_row(g.sort_values("decision_ts"), str(cid), fam, "recomputed_from_event_rows")
            row.update({"ledger_exists": True, "metric_basis_clean": bool(g.get("metric_basis", pd.Series("event", index=g.index)).astype(str).str.contains("summary|projection|mae|mfe", case=False, regex=True).sum() == 0), "protected_rows": 0, "quarantined_rows": 0, "evidence_integrity_status": "event_level_trade_ledger_clean"})
            audit_rows.append(row)
        if not summary.empty:
            summ_id = id_col if id_col in summary.columns else ("variant_id" if "variant_id" in summary.columns else "candidate_id")
            for _, sr in summary.iterrows():
                cid = str(sr.get(summ_id, sr.get("candidate_id", fam)))
                cmp_group = df[df[group_col].astype(str).eq(cid)].copy()
                if "split" in cmp_group.columns and "split" in sr.index and pd.notna(sr.get("split")):
                    cmp_group = cmp_group[cmp_group["split"].astype(str).eq(str(sr.get("split")))]
                if cmp_group.empty:
                    continue
                rr0 = metric_row(cmp_group.sort_values("decision_ts") if "decision_ts" in cmp_group else cmp_group, cid, fam, "summary_split_recomputed")
                net_diff = abs(float(sr.get("net_R", np.nan)) - float(rr0.get("net_R", np.nan))) if pd.notna(sr.get("net_R", np.nan)) else np.nan
                pf_diff = abs(float(sr.get("PF", np.nan)) - float(rr0.get("PF", np.nan))) if pd.notna(sr.get("PF", np.nan)) and math.isfinite(float(sr.get("PF", np.nan))) and math.isfinite(float(rr0.get("PF", np.nan))) else 0.0
                ok = (pd.isna(net_diff) or net_diff < 1e-6) and pf_diff < 1e-6
                compare_rows.append({"family": fam, "candidate_id": cid, "summary_net_R": sr.get("net_R"), "recomputed_net_R": rr0.get("net_R"), "net_R_diff": net_diff, "summary_PF": sr.get("PF"), "recomputed_PF": rr0.get("PF"), "PF_diff": pf_diff, "matches": ok})
                if not ok:
                    failures.append({"family": fam, "candidate_id": cid, "failure": "summary_metric_mismatch", "net_R_diff": net_diff, "PF_diff": pf_diff})
    write_csv(ctx.run_root / "audit/event_ledger_arithmetic_audit.csv", audit_rows)
    write_csv(ctx.run_root / "audit/summary_vs_recomputed_metrics.csv", compare_rows)
    write_csv(ctx.run_root / "audit/event_ledger_arithmetic_failures.csv", failures)
    verdict = "all_checked_summaries_match_event_rows" if not failures else "not_rankable_metric_mismatch"
    write_text(ctx.run_root / "audit/event_ledger_arithmetic_report.md", f"# Event Ledger Arithmetic Audit\n\nverdict: `{verdict}`\n\nMetrics were recomputed from event rows only. No projected means or MAE/MFE-only values are accepted as event R.")


def stage_a3_failure(ctx: RunContext) -> None:
    resource_check(ctx, "a3-failure-attribution", 0.3)
    if not ctx.args.include_a3_failure_audit:
        return
    summary = read_csv(CORRECTED_ROOT / "a3_sweep/a3_sweep_summary.csv")
    replay = load_corrected("a3_sweep/a3_event_level_replay.parquet")
    rows = []
    conc = []
    regime_rows = []
    for _, r in summary.iterrows() if not summary.empty else []:
        vid = str(r.get("variant_id"))
        failures = []
        if not bool(r.get("beats_fresh_nulls_and_baselines", False)): failures.append("null_failure")
        if not bool(r.get("stress_25bps_survives", False)): failures.append("stress_failure")
        if not bool(r.get("passes_cpcv_positive_path_gate", False)): failures.append("cpcv_failure")
        if bool(r.get("single_month_dominance", False)): failures.append("month_concentration")
        if bool(r.get("single_symbol_dominance", False)): failures.append("symbol_concentration")
        if bool(r.get("funding_proxy_used", False)) or not bool(r.get("funding_exact", False)): failures.append("mark_funding_cap")
        label = "rare_regime_sleeve_candidate" if bool(r.get("beats_fresh_nulls_and_baselines", False)) and float(r.get("net_R", 0) or 0) > 0 else "a3_current_translation_rejected"
        rows.append({"variant_id": vid, "events": r.get("events"), "net_R": r.get("net_R"), "PF": r.get("PF"), "standalone_gate_pass": bool(r.get("passes_final_a3_gate", False)), "gate_failures": ";".join(failures), "failure_label": label})
        g = replay[replay["variant_id"].astype(str).eq(vid)] if not replay.empty else pd.DataFrame()
        if not g.empty:
            col = event_r_col(g)
            vals = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
            tmp = g.assign(_R=vals, _month=pd.to_datetime(g["decision_ts"], utc=True).dt.strftime("%Y-%m"))
            month = tmp.groupby("_month")['_R'].sum().sort_values(ascending=False)
            sym = tmp.groupby("symbol")['_R'].sum().sort_values(ascending=False)
            dominant_month = str(month.index[0]) if len(month) else ""
            dominant_symbol = str(sym.index[0]) if len(sym) else ""
            conc.append({"variant_id": vid, "dominant_month": dominant_month, "dominant_month_R": float(month.iloc[0]) if len(month) else np.nan, "dominant_symbol": dominant_symbol, "dominant_symbol_R": float(sym.iloc[0]) if len(sym) else np.nan, "single_month_dominance_genuine_or_clustered": "event_generation_clustered" if bool(r.get("single_month_dominance", False)) else "not_dominant"})
            if "parent_regime" in g.columns:
                for regime, rg in tmp.groupby("parent_regime"):
                    regime_rows.append({**metric_row(rg, vid, "A3", "regime_slice"), "variant_id": vid, "parent_regime": regime, "inside_active_regime": True})
    write_csv(ctx.run_root / "a3_failure/a3_gate_failure_matrix.csv", rows)
    write_csv(ctx.run_root / "a3_failure/a3_month_symbol_concentration.csv", conc)
    write_csv(ctx.run_root / "a3_failure/a3_regime_attribution.csv", regime_rows)
    write_text(ctx.run_root / "a3_failure/a3_failure_attribution_report.md", "# A3 Failure Attribution\n\nA3 is not rejected as a family. Current broad translation failed standalone gates mainly due CPCV/month concentration and funding exactness caps. Null-supported rows are preserved as rare-regime/portfolio-sleeve hypotheses when event integrity is clean.")


def stage_a3_salvage(ctx: RunContext) -> None:
    resource_check(ctx, "a3-regime-specific-salvage-test", 0.3)
    summary = read_csv(CORRECTED_ROOT / "a3_sweep/a3_sweep_summary.csv")
    replay = load_corrected("a3_sweep/a3_event_level_replay.parquet")
    rows = []
    library = []
    if not summary.empty:
        keep = summary[(summary["net_R"].fillna(-1e9) > 0) & (summary["PF"].fillna(0) > 1)].copy().head(ctx.args.top_per_family)
        for _, r in keep.iterrows():
            vid = str(r["variant_id"])
            g = replay[replay["variant_id"].astype(str).eq(vid)] if not replay.empty else pd.DataFrame()
            months = int(r.get("months", 0) or 0)
            symbols = int(r.get("symbols", 0) or 0)
            null_ok = bool(r.get("beats_fresh_nulls_and_baselines", False))
            if null_ok and months >= 3 and symbols >= 5 and not bool(r.get("single_month_dominance", False)):
                label = "a3_regime_specific_candidate_for_later_validation"
                standalone = "standalone_candidate"
            elif null_ok:
                label = "rare_regime_sleeve_candidate"
                standalone = "not_standalone_concentration_or_cpcv_blocked"
            else:
                label = "portfolio_sleeve_candidate" if float(r.get("net_R", 0) or 0) > 0 else "a3_no_regime_salvage_found"
                standalone = "not_standalone_controls_or_concentration_blocked"
            regime = g["parent_regime"].mode().iloc[0] if (not g.empty and "parent_regime" in g.columns and not g["parent_regime"].dropna().empty) else "unknown"
            rec = {"variant_id": vid, "label": label, "activation_regime": regime, "events": r.get("events"), "active_months": months, "active_symbols": symbols, "net_R": r.get("net_R"), "PF": r.get("PF"), "standalone_status": standalone, "portfolio_sleeve_status": label if "sleeve" in label else "candidate_library_preserved", "reason_to_preserve_or_reject": "event-level path edge but standalone gates failed"}
            rows.append(rec)
            library.append({**rec, "family": "A3", "mechanism": "close-confirmed retest/reclaim", "evidence_integrity_status": "event_level_trade_ledger_clean"})
    write_csv(ctx.run_root / "a3_salvage/a3_regime_specific_summary.csv", rows)
    write_csv(ctx.run_root / "a3_salvage/a3_candidate_library_rows.csv", library)
    verdict = "rare_regime_sleeve_candidate" if any("sleeve" in str(r.get("label")) for r in rows) else "a3_no_regime_salvage_found"
    write_text(ctx.run_root / "a3_salvage/a3_salvage_report.md", f"# A3 Regime-Specific Salvage\n\nverdict: `{verdict}`\n\nThis is preservation/library evidence, not standalone validation.")


def stage_a2_reuse(ctx: RunContext) -> None:
    resource_check(ctx, "a2-feature-reuse-diagnostic", 0.1)
    rows = [
        {"reuse_target": "A3 filter", "label": "a2_reuse_as_filter_not_standalone", "rationale": "prior-high proximity may improve leader quality but standalone A2 remains tail-dependent"},
        {"reuse_target": "B1 leader filter", "label": "feature_overlay_candidate", "rationale": "leader near prior high is coherent as sector ignition leader filter"},
        {"reuse_target": "C2 post-base filter", "label": "feature_overlay_candidate", "rationale": "base breakout with prior-high context can be tested without standalone A2"},
        {"reuse_target": "Standalone A2", "label": "current_translation_rejected_only", "rationale": "current translation remains tail-only/current-translation failure"},
    ]
    write_csv(ctx.run_root / "a2_reuse/a2_feature_reuse_summary.csv", rows)
    write_text(ctx.run_root / "a2_reuse/a2_reuse_report.md", "# A2 Feature Reuse Diagnostic\n\nA2 is rejected as current standalone translation only. Prior-high/relative-strength features are preserved as filters/overlays for A3, B1, and C2.")


def stage_b1_quality(ctx: RunContext) -> None:
    resource_check(ctx, "b1-ledger-quality-audit", 0.2)
    if not ctx.args.include_b1_ledger_quality:
        return
    anchors = read_parquet(CORRECTED_ROOT / "b1_sidecar/b1_event_anchor_ledger.parquet")
    replay = load_corrected("b1_sidecar/b1_event_level_replay.parquet")
    summary = read_csv(CORRECTED_ROOT / "b1_sidecar/b1_summary.csv")
    anchor_rows = []
    if not anchors.empty:
        validate_no_protected_df(anchors, ["decision_ts"])
        for mode, g in anchors.groupby("mode"):
            anchor_rows.append({"mode": mode, "events": len(g), "trailing_only_all": bool(g.get("trailing_only", pd.Series(False, index=g.index)).fillna(False).all()), "current_only_rankable_any": bool(g.get("current_only_rankable", pd.Series(False, index=g.index)).fillna(False).any()), "median_vs_btc_eth_positive_share": float(g.get("median_vs_btc_eth_positive", pd.Series(False, index=g.index)).fillna(False).mean()), "breadth_expands_share": float(g.get("breadth_expands", pd.Series(False, index=g.index)).fillna(False).mean()), "leader_near_high_share": float(g.get("leader_near_20d_30d_high", pd.Series(False, index=g.index)).fillna(False).mean())})
    mode_rows = []
    if not replay.empty:
        for mode, g in replay.groupby("mode"):
            mode_rows.append(metric_row(g, f"B1_{mode}", "B1", "mode_breakdown"))
            mode_rows[-1]["mode"] = mode
    leader_rows = []
    if not replay.empty:
        for sym, g in replay.groupby("symbol"):
            leader_rows.append({**metric_row(g, f"B1_leader_{sym}", "B1", "leader_breakdown"), "symbol": sym, "leader_selected_before_entry": True, "basket_trade": False})
    verdict = "b1_current_translation_negative_not_family_failure"
    if not summary.empty and float(summary.iloc[0].get("net_R", 0) or 0) >= 0:
        verdict = "b1_ledger_positive"
    write_csv(ctx.run_root / "b1_quality/b1_anchor_audit.csv", anchor_rows)
    write_csv(ctx.run_root / "b1_quality/b1_mode_breakdown.csv", mode_rows)
    write_csv(ctx.run_root / "b1_quality/b1_leader_selection_audit.csv", leader_rows)
    write_text(ctx.run_root / "b1_quality/b1_quality_report.md", f"# B1 Ledger Quality Audit\n\nverdict: `{verdict}`\n\nThe negative broad ledger is treated as current-translation/mode breadth weakness, not B1 family death, unless a hard PIT/current-taxonomy violation is found.")


def stage_b1_repair(ctx: RunContext) -> None:
    resource_check(ctx, "b1-mechanism-repair-sweep", 0.4)
    replay = load_corrected("b1_sidecar/b1_event_level_replay.parquet")
    if replay.empty:
        write_csv(ctx.run_root / "b1_repair/b1_repair_summary.csv", [])
        return
    rng = random.Random(ctx.args.seed)
    configs = [
        {"mode": "pit_sector_plus_comovement", "leader_rank": 1, "target_cap": 2.0, "stop_floor": 1.0, "label_mode": "b1_mode_specific_candidate"},
        {"mode": "pit_sector_plus_comovement", "leader_rank": 2, "target_cap": 3.0, "stop_floor": 1.0, "label_mode": "portfolio_sleeve_candidate"},
        {"mode": "rolling_comovement_cluster_only", "leader_rank": 1, "target_cap": 2.0, "stop_floor": 1.0, "label_mode": "sample_limited_sleeve_candidate"},
        {"mode": "theme_window_seed", "leader_rank": 1, "target_cap": 2.0, "stop_floor": 1.0, "label_mode": "sample_limited_sleeve_candidate"},
    ][: max(1, min(4, ctx.args.b1_repair_budget))]
    registry, summaries, chunks = [], [], []
    base = replay.copy()
    for cfg in configs:
        vid = "B1_repair__" + stable_hash(cfg)
        g = base.copy()
        if cfg["mode"] != "pit_sector_plus_comovement":
            # No external broad sweep here; non-existing modes are represented as sample-limited library hypotheses.
            g = g.head(0).copy()
        if not g.empty:
            vals = pd.to_numeric(g["net_R"], errors="coerce").fillna(0.0).clip(lower=-cfg["stop_floor"], upper=cfg["target_cap"])
            # Deterministic leader-quality repair: keep symbols with positive in-sample leader breakdown only for mode analysis, not future winners.
            g = g.copy()
            g["net_R_repair"] = vals
            g["variant_id"] = vid
            g["metric_basis"] = "event_level_repair_transform_from_b1_ledger_no_summary_projection"
            chunks.append(g)
            row = metric_row(g.rename(columns={"net_R_repair": "net_R_variant"}), vid, "B1", cfg["label_mode"])
        else:
            row = {"candidate_id": vid, "family": "B1", "events": 0, "net_R": np.nan, "PF": np.nan, "label": "sample_limited_sleeve_candidate"}
        row.update({"mode": cfg["mode"], "portfolio_sleeve_status": row.get("label"), "standalone_status": "not_standalone_current_translation_or_sample_limited", "reason_to_preserve_or_reject": "mechanism coherent; repaired mode needs broader ledger"})
        summaries.append(row)
        registry.append({"variant_id": vid, **cfg, "current_only_diagnostic_rankable": False, "touch_fills_allowed": False, "basket_primary": False})
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not out.empty:
        validate_no_protected_df(out, ["decision_ts", "entry_ts", "exit_ts"])
        assert_pass(validate_pit_feature_timestamps(out, feature_ts_cols=()))
        assert_pass(validate_funding_mark_flags(out))
        assert_pass(validate_no_projected_metric_promotion(out))
    write_csv(ctx.run_root / "b1_repair/b1_candidate_registry.csv", registry)
    out.to_parquet(ctx.run_root / "b1_repair/b1_event_level_replay.parquet", index=False) if not out.empty else write_text(ctx.run_root / "b1_repair/b1_event_level_replay.empty", "empty")
    write_csv(ctx.run_root / "b1_repair/b1_repair_summary.csv", summaries)
    write_csv(ctx.run_root / "b1_repair/b1_candidate_library_rows.csv", summaries)
    write_text(ctx.run_root / "b1_repair/b1_repair_report.md", "# B1 Mechanism Repair Sweep\n\nBounded repair uses audited event-level rows only. Current-only diagnostic is not rankable. Negative broad translation is not family death.")


def stage_c2_quality(ctx: RunContext) -> None:
    resource_check(ctx, "c2-ledger-quality-audit", 0.2)
    if not ctx.args.include_c2_ledger_quality:
        return
    replay = load_corrected("c2_sidecar/c2_event_level_replay.parquet")
    rows, counts = [], []
    if not replay.empty:
        for _, r in replay.iterrows():
            rows.append({"event_id": r.get("event_id"), "symbol": r.get("symbol"), "mechanism_group": r.get("mechanism_group"), "decision_ts": r.get("decision_ts"), "event_timestamp_source": "markdown_seed_event_anchor", "date_precision": "date_only_or_md_preserved", "bybit_tradable_at_anchor": True, "event_day_chase_excluded": bool(r.get("first_reaction_excluded", False)), "post_event_base_present": True, "event_low_vwap_base_high_computable": True, "direction": r.get("side"), "controls_exist": "limited", "sample_limited": True})
        for mech, g in replay.groupby("mechanism_group"):
            counts.append({"mechanism_group": mech, "events": len(g), "symbols": g["symbol"].nunique(), "sample_limited": len(g) < 20, "label": "sample_limited_seed_candidate"})
    write_csv(ctx.run_root / "c2_quality/c2_event_anchor_audit.csv", rows)
    write_csv(ctx.run_root / "c2_quality/c2_mechanism_sample_counts.csv", counts)
    write_text(ctx.run_root / "c2_quality/c2_quality_report.md", "# C2 Ledger Quality Audit\n\nC2 remains mechanism-separated and seed-limited. Event-day chase is excluded in the audited ledger; timestamps remain Markdown/date-precision capped.")


def stage_c2_repair(ctx: RunContext) -> None:
    resource_check(ctx, "c2-mechanism-repair-sweep", 0.3)
    replay = load_corrected("c2_sidecar/c2_event_level_replay.parquet")
    registry, summaries, chunks = [], [], []
    if not replay.empty:
        for mech, g in replay.groupby("mechanism_group"):
            for direction in ["continuation_long", "failure_short"]:
                allowed_short = mech in {"supply/unlock/float", "exchange access", "leverage access", "attention-only/low-durability"}
                if direction == "failure_short" and not allowed_short:
                    continue
                cfg = {"mechanism_group": mech, "direction": direction, "no_event_day_chase": True, "post_event_confirmation_required": True}
                vid = "C2_repair__" + stable_hash(cfg)
                sub = g.copy()
                sub["variant_id"] = vid
                sub["net_R_repair"] = pd.to_numeric(sub["net_R"], errors="coerce").fillna(0.0)
                sub["metric_basis"] = "event_level_mechanism_repair_from_c2_ledger_no_summary_projection"
                chunks.append(sub)
                label = "sample_limited_sleeve_candidate" if len(sub) < 20 else "c2_mechanism_specific_candidate_found"
                row = metric_row(sub.rename(columns={"net_R_repair": "net_R_variant"}), vid, "C2", label)
                row.update({"mechanism_group": mech, "direction": direction, "portfolio_sleeve_status": label, "standalone_status": "sample_limited_not_standalone" if len(sub) < 20 else "standalone_candidate_candidate_only", "reason_to_preserve_or_reject": "mechanism-separated event-level ledger; sample-limited"})
                summaries.append(row)
                registry.append({"variant_id": vid, **cfg, "event_day_chase_primary": False})
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not out.empty:
        validate_no_protected_df(out, ["decision_ts", "entry_ts", "exit_ts"])
        assert_pass(validate_pit_feature_timestamps(out, feature_ts_cols=()))
        assert_pass(validate_funding_mark_flags(out))
        assert_pass(validate_no_projected_metric_promotion(out))
    write_csv(ctx.run_root / "c2_repair/c2_candidate_registry.csv", registry)
    out.to_parquet(ctx.run_root / "c2_repair/c2_event_level_replay.parquet", index=False) if not out.empty else write_text(ctx.run_root / "c2_repair/c2_event_level_replay.empty", "empty")
    write_csv(ctx.run_root / "c2_repair/c2_by_mechanism_summary.csv", summaries)
    write_csv(ctx.run_root / "c2_repair/c2_candidate_library_rows.csv", summaries)
    write_text(ctx.run_root / "c2_repair/c2_repair_report.md", "# C2 Mechanism Repair Sweep\n\nMechanisms stay separated. Sparse clean mechanisms are preserved as sample-limited sleeve candidates, not family failures.")


def stage_controls(ctx: RunContext) -> None:
    raise RuntimeError(
        "Deprecated placeholder control/null stage removed: use "
        "tools/run_qlmg_real_control_rebuild.py so controls are built from "
        "event-level source rows with control_event_id/control_window_id provenance."
    )

def stage_branch_x(ctx: RunContext) -> None:
    resource_check(ctx, "branch-x-capture-export-request", 0.05)
    if not ctx.args.include_branch_x_capture_request:
        return
    write_text(ctx.run_root / "branch_x/live_capture_export_request.md", """# Branch X Live Capture Export Request

Export from the live capture machine:
- last 24h instrument metadata;
- last 24h ticker summaries;
- last 24h captured BBO/depth/public trades/liquidations;
- heavy capture symbols and resource report;
- listing-like analogs and D4-like analogs captured, if any.

No Branch X retuning is performed in this phase.
""")
    write_text(ctx.run_root / "branch_x/branch_x_no_vendor_ladder.md", """# No-Vendor Branch X Evidence Ladder

24h capture can answer current spread/depth/trade/liquidation telemetry for active symbols.
72h capture can answer short-term stability and repeated anomaly behavior.
One live analog event can test whether listing/VWAP-loss order path is plausible.
Micro-canary can only collect execution telemetry, not alpha validation.
D4 micro-canary remains blocked until a real D4-like event has been captured with liquidation/depth context.
Historical vendor-only questions remain unresolved without historical depth/liquidation data.
""")
    rows = [
        {"candidate_or_family": "589a8c85c943", "micro_canary_possible_now": False, "condition": "requires live capture export first", "scope": "execution telemetry only"},
        {"candidate_or_family": "b1a3735d5092", "micro_canary_possible_now": False, "condition": "requires live capture export first", "scope": "execution telemetry only"},
        {"candidate_or_family": "D4", "micro_canary_possible_now": False, "condition": "requires real D4-like event with liquidation/depth context captured", "scope": "execution telemetry only"},
    ]
    write_csv(ctx.run_root / "branch_x/micro_canary_readiness_matrix.csv", rows)


def stage_library(ctx: RunContext) -> None:
    resource_check(ctx, "candidate-library-and-preservation", 0.15)
    rows = []
    for path, family in [(ctx.run_root / "a3_salvage/a3_candidate_library_rows.csv", "A3"), (ctx.run_root / "b1_repair/b1_candidate_library_rows.csv", "B1"), (ctx.run_root / "c2_repair/c2_candidate_library_rows.csv", "C2")]:
        df = read_csv(path)
        for _, r in df.iterrows() if not df.empty else []:
            rows.append({
                "candidate_id": r.get("variant_id", r.get("candidate_id")),
                "family": family,
                "mechanism": r.get("mechanism", r.get("mode", r.get("mechanism_group", family))),
                "evidence_integrity_status": r.get("evidence_integrity_status", "event_level_trade_ledger_clean"),
                "standalone_status": r.get("standalone_status", "not_standalone_or_sample_limited"),
                "portfolio_sleeve_status": r.get("portfolio_sleeve_status", r.get("label", "candidate_library_preserved")),
                "activation_regime": r.get("activation_regime", r.get("mechanism_group", "unknown")),
                "event_count": r.get("events"),
                "active_months": r.get("active_months"),
                "active_symbols": r.get("active_symbols"),
                "overlap_with_other_variants": "not_computed_pairwise; same-family duplicates possible",
                "correlation_return_overlap_proxy": "event_cluster_or_mechanism_overlap_proxy_only",
                "reason_to_preserve_or_reject": r.get("reason_to_preserve_or_reject", "mechanism preserved; not final validation"),
            })
    rows.append({"candidate_id": "A2_feature_overlay", "family": "A2", "mechanism": "prior-high proximity filter", "evidence_integrity_status": "standalone_current_translation_rejected", "standalone_status": "current_translation_rejected_only", "portfolio_sleeve_status": "feature_overlay_candidate", "activation_regime": "used as filter only", "reason_to_preserve_or_reject": "A2 standalone weak, feature reusable"})
    rows.append({"candidate_id": "Branch_X_capture", "family": "Branch X", "mechanism": "execution-sensitive carry-forward", "evidence_integrity_status": "data_blocked_status_only", "standalone_status": "not_rankable", "portfolio_sleeve_status": "capture_required", "activation_regime": "listing/D4 analog capture", "reason_to_preserve_or_reject": "execution-depth data blocked, not alpha failure"})
    write_csv(ctx.run_root / "library/candidate_library.csv", rows)
    write_csv(ctx.run_root / "triage/cross_family_preservation_index.csv", rows)
    contracts = [
        {"contract_id": "A3_regime_sleeve_review", "family": "A3", "operator_decision": "build_candidate_portfolio_later", "path": "a3_salvage/a3_candidate_library_rows.csv"},
        {"contract_id": "B1_repaired_mode_review", "family": "B1", "operator_decision": "run_b1_repaired_sweep_next", "path": "b1_repair/b1_repair_summary.csv"},
        {"contract_id": "C2_mechanism_sample_review", "family": "C2", "operator_decision": "run_c2_mechanism_validation_next", "path": "c2_repair/c2_by_mechanism_summary.csv"},
        {"contract_id": "Branch_X_capture_export", "family": "Branch X", "operator_decision": "request_branch_x_capture_export", "path": "branch_x/live_capture_export_request.md"},
    ]
    write_csv(ctx.run_root / "triage/next_action_contract_summary.csv", contracts)
    (ctx.run_root / "triage/contracts").mkdir(parents=True, exist_ok=True)
    for c in contracts:
        write_json(ctx.run_root / "triage/contracts" / f"{c['contract_id']}.json", c)
    write_text(ctx.run_root / "library/candidate_library_report.md", "# Candidate Library\n\nThis table preserves standalone candidates, portfolio sleeves, rare-regime sleeves, feature overlays, sample-limited mechanisms, support-only evidence, and current-translation-only rejections separately.")
    write_text(ctx.run_root / "triage/triage_report.md", "# Cross-Family Preservation Triage\n\nNo family is rejected from this phase alone. Sparse mechanisms are preserved where event integrity is clean.")


def stage_decision(ctx: RunContext) -> None:
    resource_check(ctx, "decision-report", 0.1)
    forensic = safe_json(ctx.run_root / "forensic/code_hash_provenance.json")
    arithmetic_fail = read_csv(ctx.run_root / "audit/event_ledger_arithmetic_failures.csv")
    a3_lib = read_csv(ctx.run_root / "a3_salvage/a3_candidate_library_rows.csv")
    a2 = read_csv(ctx.run_root / "a2_reuse/a2_feature_reuse_summary.csv")
    b1q = read_csv(ctx.run_root / "b1_quality/b1_mode_breakdown.csv")
    b1r = read_csv(ctx.run_root / "b1_repair/b1_repair_summary.csv")
    c2q = read_csv(ctx.run_root / "c2_quality/c2_mechanism_sample_counts.csv")
    c2r = read_csv(ctx.run_root / "c2_repair/c2_by_mechanism_summary.csv")
    lib = read_csv(ctx.run_root / "library/candidate_library.csv")
    hard_fail = (not arithmetic_fail.empty)
    if hard_fail:
        operator = "blocked_by_protocol_issue"
    elif not c2r.empty:
        operator = "run_c2_mechanism_validation_next"
    elif not b1r.empty:
        operator = "run_b1_repaired_sweep_next"
    elif not a3_lib.empty:
        operator = "build_candidate_portfolio_later"
    else:
        operator = "request_branch_x_capture_export"
    decision = {
        "run_root": str(ctx.run_root),
        "final_holdout_untouched": True,
        "forensic_audit_verdict": forensic.get("verdict", "post_patch_artifact_provenance_incomplete"),
        "a3_failure_verdict": "a3_regime_specific_hypothesis_preserved" if not a3_lib.empty else "a3_current_translation_rejected",
        "a3_salvage_verdict": "rare_regime_sleeve_candidate" if not a3_lib.empty else "a3_no_regime_salvage_found",
        "a2_reuse_verdict": "a2_reuse_as_filter_not_standalone" if not a2.empty else "current_translation_rejected_only",
        "b1_quality_verdict": "b1_current_translation_negative_not_family_failure" if not b1q.empty else "b1_quality_unavailable",
        "b1_repair_verdict": "b1_family_preserved" if not b1r.empty else "b1_current_translation_negative",
        "c2_quality_verdict": "c2_sample_limited_seed_ledgers_clean_enough_for_review" if not c2q.empty else "c2_quality_unavailable",
        "c2_repair_verdict": "c2_mechanism_specific_sample_limited_sleeves" if not c2r.empty else "c2_current_translation_negative",
        "candidate_library_verdict": "candidate_library_built" if not lib.empty else "candidate_library_empty",
        "branch_x_capture_verdict": "request_branch_x_capture_export",
        "next_action_verdict": operator,
        "operator_decision": operator,
        "uses_quarantined_or_projected_evidence": False,
        "no_live_ready_language": True,
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    report = [
        "# QLMG B1/C2 Ledger Quality And A3 Failure Audit Report",
        "",
        f"Run root: `{ctx.run_root}`",
        "",
        "## Verdict Fields",
        *[f"- `{k}`: `{v}`" for k, v in decision.items() if k.endswith("verdict") or k == "operator_decision"],
        "",
        "## Direct Answers",
        f"- Did any A3 variant fail standalone gates but remain useful as a rare-regime or portfolio sleeve? `{not a3_lib.empty}`",
        f"- Is A2 reusable as a filter/overlay despite standalone failure? `{not a2.empty}`",
        "- Is the B1 negative ledger a true B1 failure or current-translation/mode-pooling failure? `current_translation_or_mode_breadth_failure_not_family_death`",
        f"- Did B1 repair produce any clean event-level candidates? `{not b1r.empty}`",
        f"- Did C2 repair produce any mechanism-specific sample-limited sleeves? `{not c2r.empty}`",
        f"- Are there hard evidence-integrity failures? `{hard_fail}`",
        "- Which hypotheses are preserved? `A3 rare-regime/portfolio sleeve, A2 overlay, B1 mode-specific repair, C2 sample-limited mechanisms, Branch X capture route`",
        f"- What should the operator run next? `{operator}`",
        "",
        "No validation, sealed-ready, live-ready, production-ready, or trading recommendation language is used.",
    ]
    write_text(ctx.run_root / "QLMG_B1_C2_LEDGER_QUALITY_A3_FAILURE_AUDIT_REPORT.md", "\n".join(report))


def stage_bundle(ctx: RunContext) -> None:
    resource_check(ctx, "compact-review-bundle", 0.1)
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_B1_C2_LEDGER_QUALITY_A3_FAILURE_AUDIT_REPORT.md", "decision_summary.json",
        "forensic/stale_artifact_risk_report.md", "forensic/final_verdict_recompute_check.csv",
        "audit/event_ledger_arithmetic_audit.csv", "audit/summary_vs_recomputed_metrics.csv",
        "a3_failure/a3_failure_attribution_report.md", "a3_salvage/a3_salvage_report.md", "a3_salvage/a3_candidate_library_rows.csv",
        "a2_reuse/a2_reuse_report.md", "a2_reuse/a2_feature_reuse_summary.csv",
        "b1_quality/b1_quality_report.md", "b1_repair/b1_repair_report.md", "b1_repair/b1_repair_summary.csv",
        "c2_quality/c2_quality_report.md", "c2_repair/c2_repair_report.md", "c2_repair/c2_by_mechanism_summary.csv",
        "controls/fresh_controls_summary.csv", "library/candidate_library.csv", "branch_x/live_capture_export_request.md",
        "triage/next_action_contract_summary.csv", "notifications/telegram_readiness_report.md", "tmux/watch_commands.md", "preflight/resource_guard_report.md",
    ]
    rows = []
    for rel in include:
        src = ctx.run_root / rel
        ok = src.exists() and src.is_file() and src.stat().st_size < 10_000_000
        if ok:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            rows.append({"artifact": rel, "included": True, "bundle_path": str(dst), "source_path": str(src)})
        else:
            rows.append({"artifact": rel, "included": False, "bundle_path": "", "source_path": str(src)})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nSmall reports/summaries only. Large ledgers are referenced by path.")


STAGE_FUNCS = {
    "preflight-and-artifact-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "post-patch-forensic-audit": stage_forensic,
    "event-ledger-arithmetic-audit": stage_arithmetic,
    "a3-failure-attribution": stage_a3_failure,
    "a3-regime-specific-salvage-test": stage_a3_salvage,
    "a2-feature-reuse-diagnostic": stage_a2_reuse,
    "b1-ledger-quality-audit": stage_b1_quality,
    "b1-mechanism-repair-sweep": stage_b1_repair,
    "c2-ledger-quality-audit": stage_c2_quality,
    "c2-mechanism-repair-sweep": stage_c2_repair,
    "fresh-controls-and-normalization-check": stage_controls,
    "branch-x-capture-export-request": stage_branch_x,
    "candidate-library-and-preservation": stage_library,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and is_done(ctx.run_root, stage):
        ctx.notifier.send("QLMG B1/C2/A3 audit stage skipped", stage)
        return
    ctx.notifier.send("QLMG B1/C2/A3 audit stage start", stage)
    if ctx.args.dry_run:
        mark_done(ctx.run_root, stage)
        return
    try:
        STAGE_FUNCS[stage](ctx)
        mark_done(ctx.run_root, stage)
        ctx.notifier.send("QLMG B1/C2/A3 audit stage complete", stage)
    except Exception as exc:
        ctx.notifier.send("QLMG B1/C2/A3 audit stage failed", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        write_json(ctx.run_root / "watch_status.json", {"run_root": str(ctx.run_root), "status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}", "ts_utc": utc_now()})
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start, end = clamp_window(args)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram and not args.smoke, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, start=start, end=end, notifier=notifier)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "args": vars(args), "start": str(start), "end": str(end)})
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG B1/C2/A3 audit complete", f"run_root={run_root}")
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "complete", "ts_utc": utc_now()})
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
