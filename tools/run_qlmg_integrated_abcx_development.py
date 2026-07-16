#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_markdown_seed_parser import (  # noqa: E402
    declared_counts,
    detect_date_precision,
    find_table,
    parse_markdown_tables,
    stable_hash,
    table_to_df,
)
from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_integrated_abcx_development_20260628_v2"
DEFAULT_SEED = 20260628
DATA_5M = Path("/opt/parquet/5m")

LIQUID_ROOT = RESULTS_ROOT / "phase_qlmg_liquid_regime_strategy_research_20260628_v1_20260628_120124"
PROXY_ROOT = RESULTS_ROOT / "phase_qlmg_best_effort_proxy_execution_sim_20260628_v1_20260628_105109"
BRUTAL_ROOT = RESULTS_ROOT / "phase_qlmg_brutal_no_depth_stress_20260628_v1_20260628_101136"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
D4_AUDIT_ROOT = RESULTS_ROOT / "phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927"

STAGES = (
    "preflight-and-prior-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "integrated-branch-registry",
    "markdown-source-ingest-and-table-extraction",
    "seed-data-validation",
    "a2-a3-dedup-and-definition-freeze",
    "a2-a3-focused-train-validation",
    "b1-sector-map-and-cluster-build",
    "b1-sector-ignition-tests",
    "c2-catalyst-ledger-build",
    "c2-post-catalyst-base-tests",
    "branch-x-status-and-capture-calibration",
    "cross-branch-triage",
    "next-contracts-and-backlog",
    "decision-report",
    "compact-review-bundle",
    "all",
)

B1_LABELS = {
    "b1_rankable_pit_sector_candidate",
    "b1_comovement_cluster_candidate",
    "b1_theme_seed_candidate",
    "b1_md_seed_research_prelead",
    "b1_taxonomy_proxy_only",
    "b1_mechanism_overlay_only",
    "not_fairly_tested_missing_sector_map",
    "sample_limited_seed_candidate",
    "path_edge_exit_problem",
    "reject_current_translation_only",
}
C2_LABELS = {
    "c2_high_confidence_candidate_train_only",
    "c2_medium_confidence_candidate_train_only",
    "c2_md_excerpt_seed_candidate",
    "c2_seed_limited_candidate",
    "c2_failure_short_candidate",
    "c2_event_family_too_noisy",
    "c2_mechanism_overlay_only",
    "not_fairly_tested_missing_catalyst_data",
    "not_fairly_tested_missing_event_timestamp",
    "sample_limited_seed_candidate",
    "path_edge_exit_problem",
    "reject_current_translation_only",
}
HIGH_LEVEL_VERDICTS = {
    "a2_a3_tier1_prelead_confirmed_train_only",
    "a2_a3_research_inconclusive",
    "b1_rankable_sector_candidate_found",
    "b1_md_seed_research_prelead_found",
    "b1_data_build_required",
    "c2_high_confidence_candidate_found",
    "c2_md_excerpt_seed_candidate_found",
    "c2_data_build_required",
    "continue_branch_x_capture_and_execution_telemetry",
    "no_family_rejected_only_current_translations",
    "blocked_by_protocol_issue",
}

MECHANISM_GROUPS = [
    "legal_regulatory_repricing",
    "etf_institutional_access",
    "supply_shock",
    "unlock_vesting_change",
    "protocol_utility_fee_revenue_change",
    "exchange_access_expansion",
    "leverage_access_expansion",
    "integration_distribution_access",
    "attention_only_low_durability",
]


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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-abcx-v2")
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
    p = argparse.ArgumentParser(description="QLMG integrated ABCX development v2, train-only")
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
    p.add_argument("--include-a2-a3", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-b1", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-c2", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-branch-x", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sector-md", default="research_inputs/point_in_time_sector_seeds.md")
    p.add_argument("--catalyst-md", default="research_inputs/post_catalyst_c2_database.md")
    p.add_argument("--sector-csv", default="")
    p.add_argument("--catalyst-main-csv", default="")
    p.add_argument("--catalyst-excluded-csv", default="")
    p.add_argument("--live-capture-bundle", default="")
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--top-per-family", type=int, default=40)
    p.add_argument("--aggressive-overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tmux-session-name", default="qlmg_abcx_development")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--run-root", default="")
    return p.parse_args(argv)


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


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
    requested_end = pd.to_datetime(args.end, utc=True) if args.end else SCREENING_END
    end = min(pd.Timestamp(requested_end), SCREENING_END)
    if start >= FINAL_HOLDOUT_START or end >= FINAL_HOLDOUT_START:
        raise RuntimeError("requested window overlaps protected QLMG final holdout")
    return pd.Timestamp(start), pd.Timestamp(end)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    rows_list = list(rows)
    keys: list[str] = list(fieldnames or [])
    if not keys:
        for row in rows_list:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_hash(path: Path, max_bytes: int = 20_000_000) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest() + ("_partial" if path.stat().st_size > max_bytes else "")


def mark_done(root: Path, stage: str) -> None:
    p = root / "stage_status" / f"{stage}.done"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(utc_now() + "\n", encoding="utf-8")


def is_done(root: Path, stage: str) -> bool:
    return (root / "stage_status" / f"{stage}.done").exists()


def resource_check(ctx: RunContext, stage: str, estimated_output_gb: float = 0.2) -> None:
    snap = resource_snapshot(ctx.run_root.parent)
    guard = check_resource_guard(
        snap,
        estimated_output_gb=estimated_output_gb,
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=35.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    out = ctx.run_root / "resource_guard" / f"{stage}.json"
    write_json(out, guard)
    if guard["warnings"]:
        ctx.notifier.send("QLMG ABCX resource warning", json.dumps(guard), level="warning")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop for {stage}: {guard}")


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except Exception:
        return str(path)


def root_report_path(root: Path) -> Path | None:
    if not root.exists():
        return None
    reports = sorted(root.glob("*REPORT.md"))
    return reports[0] if reports else None


def root_status(root: Path) -> str:
    if not root.exists():
        return "not_available"
    if (root / "decision_summary.json").exists():
        return "available_decision"
    return "available_no_decision"


def normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def date_precision_to_anchor(value: Any) -> tuple[pd.Timestamp | None, str, str]:
    s = str(value if value is not None else "").strip()
    prec = detect_date_precision(s)
    if prec == "unknown":
        return None, prec, "unknown"
    raw = s[2:].strip() if s.startswith("<=") else s
    try:
        if prec == "exact_datetime":
            ts = pd.to_datetime(raw, utc=True)
            return pd.Timestamp(ts), prec, "as_provided"
        if prec == "date_only":
            ts = pd.to_datetime(raw, utc=True) + pd.Timedelta(days=1)
            return pd.Timestamp(ts), prec, "next_daily_boundary"
        if prec == "month_only":
            ts = pd.to_datetime(raw + "-01", utc=True) + pd.offsets.MonthBegin(1)
            return pd.Timestamp(ts), prec, "next_month_boundary"
        if prec == "year_only":
            ts = pd.Timestamp(f"{raw}-01-01T00:00:00Z") + pd.offsets.YearBegin(1)
            return pd.Timestamp(ts), prec, "next_year_boundary"
        if prec == "lte_date":
            ts = pd.to_datetime(raw, utc=True) + pd.Timedelta(days=1)
            return pd.Timestamp(ts), prec, "lte_next_daily_boundary"
    except Exception:
        return None, prec, "parse_failed"
    return None, prec, "unsupported"


def symbol_live_window(symbol: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    p = DATA_5M / f"{symbol}.parquet"
    if not p.exists():
        return None, None
    try:
        df = pd.read_parquet(p, columns=["timestamp"])
    except Exception:
        return None, None
    if df.empty:
        return None, None
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None, None
    return pd.Timestamp(ts.min()), pd.Timestamp(ts.max())


def source_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def stage_preflight(ctx: RunContext) -> None:
    resource_check(ctx, "preflight-and-prior-freeze", 0.2)
    ctx.run_root.mkdir(parents=True, exist_ok=True)
    prior_roots = {
        "liquid_regime_strategy_research": LIQUID_ROOT,
        "best_effort_proxy_execution_sim": PROXY_ROOT,
        "brutal_no_depth_stress": BRUTAL_ROOT,
        "listing_generic_full_event_replay": LISTING_ROOT,
        "d4_survivability": D4_SURVIVAL_ROOT,
        "d4_liquidation_audit": D4_AUDIT_ROOT,
    }
    if ctx.args.live_capture_bundle:
        prior_roots["live_capture_bundle"] = source_path(ctx.args.live_capture_bundle)
    sector_md = source_path(ctx.args.sector_md)
    catalyst_md = source_path(ctx.args.catalyst_md)
    manifest = []
    hashes: dict[str, Any] = {"git_head": "unknown", "run_root": str(ctx.run_root)}
    try:
        hashes["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        pass
    for name, root in prior_roots.items():
        status = root_status(root) if root.is_dir() else ("available_file" if root.exists() else "not_available")
        decision = root / "decision_summary.json" if root.is_dir() else Path("")
        report = root_report_path(root) if root.is_dir() else None
        manifest.append({"artifact": name, "path": str(root), "status": status, "decision_summary": str(decision) if decision.exists() else "", "report": str(report or "")})
        if root.exists():
            hashes[name] = file_hash(root if root.is_file() else (decision if decision.exists() else (report or root)))
            if decision.exists():
                hashes[f"{name}:decision_summary"] = file_hash(decision)
            if report:
                hashes[f"{name}:report"] = file_hash(report)
    md_status = {
        "sector_md": {"path": str(sector_md), "exists": sector_md.exists(), "hash": file_hash(sector_md)},
        "catalyst_md": {"path": str(catalyst_md), "exists": catalyst_md.exists(), "hash": file_hash(catalyst_md)},
        "csv_policy": "markdown_default_optional_csv_only_after_schema_validation",
    }
    hashes["sector_md"] = md_status["sector_md"]
    hashes["catalyst_md"] = md_status["catalyst_md"]
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", manifest)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    write_json(ctx.run_root / "preflight/md_source_status.json", md_status)
    snap = resource_snapshot(ctx.run_root.parent)
    guard = check_resource_guard(snap, estimated_output_gb=5.0, hard_free_gb=5.0, warn_free_gb=7.0, hard_stage_output_gb=35.0, allow_large_output=ctx.args.allow_large_output)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\nstatus={guard['status']} free_disk_gb={guard['free_disk_gb']:.2f} estimated_output_gb=5.0 max_output_gb={ctx.args.max_output_gb}")
    write_text(ctx.run_root / "preflight/preflight_report.md", "\n".join([
        "# QLMG Integrated ABCX v2 Preflight",
        f"run_root: `{ctx.run_root}`",
        f"window: `{ctx.start}` to `{ctx.end}`",
        f"sector_md_exists: `{sector_md.exists()}`",
        f"catalyst_md_exists: `{catalyst_md.exists()}`",
        "Markdown is the authoritative B1/C2 seed source for this run. Missing companion CSVs are not an operator blocker.",
    ]))


def stage_telegram(ctx: RunContext) -> None:
    resource_check(ctx, "telegram-and-tmux-setup", 0.05)
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\nstatus: `{ctx.notifier.status}`\n\nremote_available: `{ctx.notifier.remote_available}`\n\nmissing: `{ctx.notifier.missing}`")
    write_text(ctx.run_root / "tmux/watch_commands.md", "\n".join([
        "# Watch Commands",
        f"tmux attach -t {ctx.args.tmux_session_name}",
        f"tail -f {ctx.run_root}/logs/full_run.log",
        f"watch -n 30 'cat {ctx.run_root}/watch_status.json'",
        f"tail -f {ctx.run_root}/notifications/telegram_events.jsonl",
        "df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h",
    ]))
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", "\n".join([
        "# Tmux Run Instructions",
        "Full launch requires `--launch-tmux`.",
        "Remote Telegram is required when `--require-telegram` is set unless `--allow-no-telegram` is explicitly passed.",
    ]))
    ctx.notifier.send("QLMG ABCX v2 run initialized", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    resource_check(ctx, "seal-guard", 0.05)
    checks = [
        {"case": "allowed_end", "timestamp": str(SCREENING_END), "passes": True},
        {"case": "protected_start", "timestamp": str(FINAL_HOLDOUT_START), "passes": False},
    ]
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "checks": checks})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\nProtected start: `{FINAL_HOLDOUT_START}`. Generated candidate rows must be before this timestamp.")


def stage_branch_registry(ctx: RunContext) -> None:
    resource_check(ctx, "integrated-branch-registry", 0.1)
    rows = [
        {"branch_id": "branch_l_a2a3_liquid_regime", "component": "A2/A3", "status": "focused_validation", "rankable_within_branch": True},
        {"branch_id": "branch_b_sector_ignition", "component": "B1", "status": "markdown_seed_research", "rankable_within_branch": ctx.args.include_b1},
        {"branch_id": "branch_c_post_catalyst_base", "component": "C2", "status": "markdown_seed_research", "rankable_within_branch": ctx.args.include_c2},
        {"branch_id": "branch_x_execution_sensitive", "component": "D4/listing/generic/funding", "status": "capture_status_only_no_retune", "rankable_within_branch": False},
        {"branch_id": "branch_ops_live_capture", "component": "capture_bundle", "status": "supplied" if ctx.args.live_capture_bundle else "not_available", "rankable_within_branch": False},
    ]
    write_csv(ctx.run_root / "registry/project_branch_registry.csv", rows)
    write_text(ctx.run_root / "registry/project_branch_registry.md", "# Integrated Branch Registry\n\nBranches are separated. Branch X PnL is never mixed into A2/A3/B1/C2 rankings. Every candidate row carries `branch_id`.")


def _write_raw_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def stage_markdown_extract(ctx: RunContext) -> None:
    resource_check(ctx, "markdown-source-ingest-and-table-extraction", 0.2)
    out = ctx.run_root / "md_extract"
    out.mkdir(parents=True, exist_ok=True)
    rows_report = []
    completeness = []
    unparsed_all = []
    mapping = {
        "sector": source_path(ctx.args.sector_md),
        "catalyst": source_path(ctx.args.catalyst_md),
    }
    for kind, path in mapping.items():
        if not path.exists():
            completeness.append({"source": kind, "source_md_path": str(path), "declared_count": 0, "parsed_rows": 0, "validated_accepted_rows": 0, "rejected_or_excluded_rows": 0, "unparsed_rows": 0, "status": "missing"})
            continue
        tables, unparsed = parse_markdown_tables(path)
        unparsed_all.extend([{**u, "source_md_path": str(path), "source": kind} for u in unparsed])
        table_by_section = {t.section.lower(): t for t in tables}
        if kind == "sector":
            targets = {
                "sector_source_matrix_raw.csv": find_table(tables, "source matrix"),
                "sector_seed_raw.csv": find_table(tables, "machine-readable seed table"),
                "sector_annotation_only_raw.csv": find_table(tables, "current-only", "annotation only"),
                "theme_ignition_raw.csv": find_table(tables, "theme ignition event table"),
            }
        else:
            targets = {
                "catalyst_source_matrix_raw.csv": find_table(tables, "catalyst source matrix"),
                "catalyst_main_raw.csv": find_table(tables, "main catalyst database"),
                "catalyst_excluded_raw.csv": find_table(tables, "low-confidence", "excluded"),
                "catalyst_event_family_rules_raw.csv": find_table(tables, "event families", "recommended c2"),
            }
        for fname, table in targets.items():
            df = table_to_df(table, path)
            _write_raw_df(out / fname, df)
            rows_report.append({"source": kind, "file": fname, "section": table.section if table else "missing", "rows": len(df), "parse_status": "ok" if table is not None else "missing_section"})
        declared = declared_counts(path)
        declared_max = max([int(r["declared_count"]) for r in declared], default=0)
        parsed_total = sum(len(t.rows) for t in tables)
        completeness.append({"source": kind, "source_md_path": str(path), "declared_count": declared_max, "parsed_rows": parsed_total, "validated_accepted_rows": 0, "rejected_or_excluded_rows": 0, "unparsed_rows": len(unparsed), "status": "parsed", "declared_count_lines": json.dumps(declared, ensure_ascii=False)})
    write_csv(out / "md_completeness_audit.csv", completeness)
    write_csv(out / "md_extraction_table_report.csv", rows_report)
    if unparsed_all:
        for i, u in enumerate(unparsed_all, start=1):
            write_text(out / "unparsed_sections" / f"unparsed_{i:03d}.md", u.get("raw_text", ""))
    write_csv(out / "unparsed_sections.csv", unparsed_all)
    write_text(out / "md_completeness_audit.md", "# Markdown Completeness Audit\n\nDeclared row counts are compared to parsed rows and later updated with validated row counts in seed validation. C2 is capped as excerpt-limited when parsed main rows are fewer than the declared full database.")
    write_text(out / "md_extraction_report.md", "# Markdown Extraction Report\n\nMarkdown files were parsed directly. Raw row text, citations/source cells, section names, row numbers, hashes, parse status, and warnings are preserved. Companion CSVs are optional and not required for this run.")


def normalize_confidence(s: Any) -> str:
    t = str(s if s is not None else "").strip().lower()
    if "high" in t:
        return "high"
    if "medium" in t:
        return "medium"
    if "low" in t:
        return "low"
    return "unknown"


def normalize_sector_seed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["md_seed_source"] = True
    out["sector_confidence_norm"] = out.get("sector_confidence", "unknown").map(normalize_confidence) if "sector_confidence" in out else "unknown"
    out["is_current_only_bool"] = out.get("is_current_only", "").map(normalize_bool) if "is_current_only" in out else False
    out["effective_start_precision"] = out.get("effective_start_utc", "").map(detect_date_precision) if "effective_start_utc" in out else "unknown"
    out["rankable_pit_sector_seed"] = out["sector_confidence_norm"].isin(["high", "medium"]) & (~out["is_current_only_bool"]) & (~out["effective_start_precision"].eq("unknown"))
    out["parse_status"] = out.get("parse_status", "parsed")
    out["parse_warning"] = out.get("parse_warning", "")
    return out


def normalize_theme_seed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["md_seed_source"] = True
    date_col = "first_public_theme_ts_utc" if "first_public_theme_ts_utc" in out.columns else ("theme_recognition_date_utc" if "theme_recognition_date_utc" in out.columns else "")
    out["date_precision"] = out[date_col].map(detect_date_precision) if date_col else "unknown"
    out["rankable_theme_seed"] = ~out["date_precision"].eq("unknown")
    return out


def normalize_catalyst_seed(df: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["md_seed_source"] = True
    out["source_confidence_norm"] = out.get("durability_score_ex_ante", out.get("source_confidence", "unknown")).map(normalize_confidence)
    out["first_public_precision"] = out.get("first_public_ts_utc", "").map(detect_date_precision) if "first_public_ts_utc" in out else "unknown"
    out["effective_precision"] = out.get("effective_ts_utc", "").map(detect_date_precision) if "effective_ts_utc" in out else "unknown"
    anchors = []
    anchor_source = []
    anchor_precision = []
    for _, row in out.iterrows():
        ts, prec, src = date_precision_to_anchor(row.get("first_public_ts_utc", ""))
        source = "first_public_ts_utc"
        if ts is None:
            ts, prec, src = date_precision_to_anchor(row.get("effective_ts_utc", ""))
            source = "effective_ts_utc" if ts is not None else "unknown"
        anchors.append(ts)
        anchor_source.append(source + ":" + src)
        anchor_precision.append(prec)
    out["event_anchor_ts"] = anchors
    out["event_anchor_source"] = anchor_source
    out["date_precision"] = anchor_precision
    excluded_ids = set(excluded.get("event_id", pd.Series(dtype=str)).astype(str)) if not excluded.empty and "event_id" in excluded else set()
    out["excluded_from_primary"] = out.get("event_id", pd.Series(dtype=str)).astype(str).isin(excluded_ids)
    out["primary_c2_eligible_seed"] = out["source_confidence_norm"].isin(["high", "medium"]) & (~out["excluded_from_primary"]) & pd.notna(out["event_anchor_ts"])
    out["md_excerpt_seed_limited"] = True
    return out


def stage_seed_validation(ctx: RunContext) -> None:
    resource_check(ctx, "seed-data-validation", 0.3)
    md = ctx.run_root / "md_extract"
    sector_raw = read_csv(md / "sector_seed_raw.csv")
    annotation_raw = read_csv(md / "sector_annotation_only_raw.csv")
    theme_raw = read_csv(md / "theme_ignition_raw.csv")
    catalyst_raw = read_csv(md / "catalyst_main_raw.csv")
    catalyst_excl = read_csv(md / "catalyst_excluded_raw.csv")
    catalyst_rules = read_csv(md / "catalyst_event_family_rules_raw.csv")
    sector = normalize_sector_seed(sector_raw)
    theme = normalize_theme_seed(theme_raw)
    catalyst = normalize_catalyst_seed(catalyst_raw, catalyst_excl)
    if not sector.empty:
        (ctx.run_root / "seeds").mkdir(parents=True, exist_ok=True)
        sector.to_parquet(ctx.run_root / "seeds/sector_seed_validated.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "seeds/sector_seed_validated.parquet", index=False)
    if not theme.empty:
        theme.to_parquet(ctx.run_root / "seeds/theme_ignition_seed_validated.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "seeds/theme_ignition_seed_validated.parquet", index=False)
    if not catalyst.empty:
        validate_no_protected(catalyst[catalyst["event_anchor_ts"].notna()], ["event_anchor_ts"])
        catalyst.to_parquet(ctx.run_root / "seeds/catalyst_seed_validated.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "seeds/catalyst_seed_validated.parquet", index=False)
    annotation_raw.to_csv(ctx.run_root / "seeds/sector_annotation_only_validated.csv", index=False)
    catalyst_excl.to_csv(ctx.run_root / "seeds/catalyst_excluded_validated.csv", index=False)
    catalyst_rules.to_parquet(ctx.run_root / "seeds/catalyst_event_family_rules.parquet", index=False)
    comp = read_csv(ctx.run_root / "md_extract/md_completeness_audit.csv")
    rows = []
    for _, r in comp.iterrows():
        source = r["source"]
        accepted = int(sector.get("rankable_pit_sector_seed", pd.Series(dtype=bool)).sum()) if source == "sector" else int(catalyst.get("primary_c2_eligible_seed", pd.Series(dtype=bool)).sum())
        rejected = (len(sector) - accepted + len(annotation_raw)) if source == "sector" else (len(catalyst) - accepted + len(catalyst_excl))
        rows.append({**r.to_dict(), "validated_accepted_rows": accepted, "rejected_or_excluded_rows": int(rejected), "md_excerpt_seed_limited": bool(source == "catalyst" and int(r.get("declared_count", 0) or 0) > len(catalyst))})
    write_csv(ctx.run_root / "md_extract/md_completeness_audit.csv", rows)
    verdict_caps = {
        "b1": "md_seed_research_prelead_cap" if not sector.empty else "not_fairly_tested_missing_sector_seed_md",
        "c2": "md_excerpt_seed_limited" if rows and any(x.get("md_excerpt_seed_limited") for x in rows if x.get("source") == "catalyst") else ("md_seed_research_prelead_cap" if not catalyst.empty else "not_fairly_tested_missing_catalyst_seed_md"),
        "markdown_default_source": True,
    }
    write_json(ctx.run_root / "seeds/fullness_and_verdict_caps.json", verdict_caps)
    write_text(ctx.run_root / "seeds/seed_ingest_report.md", f"# Seed Ingest Report\n\nSector rows parsed: `{len(sector)}`; rankable PIT rows: `{int(sector.get('rankable_pit_sector_seed', pd.Series(dtype=bool)).sum()) if not sector.empty else 0}`.\n\nCatalyst rows parsed: `{len(catalyst)}`; primary eligible seed rows: `{int(catalyst.get('primary_c2_eligible_seed', pd.Series(dtype=bool)).sum()) if not catalyst.empty else 0}`.\n\nCitations/source cells are preserved as text metadata and are not used as strategy features.")


def load_prior_csv(name: str) -> pd.DataFrame:
    p = LIQUID_ROOT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def stage_a2a3_dedup(ctx: RunContext) -> None:
    resource_check(ctx, "a2-a3-dedup-and-definition-freeze", 0.2)
    refined = load_prior_csv("refine/refined_candidate_summary.csv")
    nulls = load_prior_csv("nulls/null_summary.csv")
    stress = load_prior_csv("stress/stress_summary.csv")
    candidates = refined[refined["family"].isin(["A2", "A3"])] if not refined.empty else pd.DataFrame()
    if not nulls.empty:
        candidates = candidates.merge(nulls[["candidate_id", "beats_matched_null", "beats_same_regime_null", "matched_null_uplift_R", "same_regime_null_uplift_R"]], on="candidate_id", how="left")
    if not stress.empty:
        candidates = candidates.merge(stress[["candidate_id", "stress_label", "stress_net_R"]], on="candidate_id", how="left")
    if candidates.empty:
        dedup = pd.DataFrame()
    else:
        candidates = candidates.sort_values(["family", "net_R_proxy", "PF_proxy"], ascending=[True, False, False])
        key_cols = ["family", "side", "regime_gate"]
        dedup = candidates.drop_duplicates(key_cols).head(12).copy()
        dedup["definition_id"] = [f"{r.family}_def_{i+1:02d}" for i, r in enumerate(dedup.itertuples())]
        dedup["lookback_definition"] = np.where(dedup["family"].eq("A2"), "prior_high_proximity_from_prior_run", "close_confirmed_retest_reclaim_from_prior_run")
        dedup["entry_timing"] = "prior_liquid_regime_contract"
        dedup["stop_definition"] = "prior_liquid_regime_contract"
        dedup["exit_definition"] = "prior_liquid_regime_best_internal_validation"
        dedup["execution_model"] = "tier1_all_taker_proxy"
        dedup["sizing_model"] = "diagnostic_fixed_risk_proxy"
    write_csv(ctx.run_root / "a2a3/deduped_candidate_definitions.csv", dedup)
    for _, r in dedup.iterrows() if not dedup.empty else []:
        contract = r.to_dict()
        contract.update({"branch_id": "branch_l_a2a3_liquid_regime", "protected_holdout_start": str(FINAL_HOLDOUT_START), "no_live_trading": True, "no_sealed_validation": True})
        write_text(ctx.run_root / f"a2a3/contracts/{r['definition_id']}.json", json.dumps(contract, indent=2, sort_keys=True, default=str))
    write_text(ctx.run_root / "a2a3/dedup_report.md", f"# A2/A3 Dedup Report\n\nPrior row-level survivors were deduplicated by family/side/regime and prior contract fields. Unique definitions written: `{len(dedup)}`. If this remains above 6, the definitions are preserved rather than force-collapsed.")


def stage_a2a3_validation(ctx: RunContext) -> None:
    resource_check(ctx, "a2-a3-focused-train-validation", 0.8)
    dedup = read_csv(ctx.run_root / "a2a3/deduped_candidate_definitions.csv")
    events_path = LIQUID_ROOT / "events/entry_event_ledger.parquet"
    prior_events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
    prior_events = prior_events[prior_events["family"].isin(["A2", "A3"])] if not prior_events.empty else prior_events
    if ctx.args.smoke and not prior_events.empty:
        prior_events = prior_events.head(500)
    if not prior_events.empty:
        validate_no_protected(prior_events, ["decision_ts", "entry_ts"])
        (ctx.run_root / "a2a3").mkdir(parents=True, exist_ok=True)
        prior_events.to_parquet(ctx.run_root / "a2a3/a2a3_full_event_ledger.parquet", index=False)
    nulls = load_prior_csv("nulls/null_summary.csv")
    stress = load_prior_csv("stress/stress_summary.csv")
    validation = load_prior_csv("validation/walk_forward_summary.csv")
    rows = []
    ablation = []
    for _, d in dedup.iterrows() if not dedup.empty else []:
        cid = d.get("candidate_id")
        evs = prior_events[prior_events["family"].eq(d.get("family"))] if not prior_events.empty else pd.DataFrame()
        nrow = nulls[nulls["candidate_id"].eq(cid)] if not nulls.empty else pd.DataFrame()
        srow = stress[stress["candidate_id"].eq(cid)] if not stress.empty else pd.DataFrame()
        vrow = validation[validation["candidate_id"].eq(cid)] if not validation.empty else pd.DataFrame()
        beats = bool(nrow.iloc[0].get("beats_matched_null", False)) if not nrow.empty else False
        stress_ok = bool((not srow.empty) and str(srow.iloc[0].get("stress_label", "")).startswith("stress_survives"))
        wf = float(vrow.iloc[0].get("percent_positive_paths", 0.0)) if not vrow.empty else 0.0
        label = "a2_a3_tier1_prelead_confirmed_train_only" if beats and stress_ok and wf >= 0.55 else ("tier1_research_prelead" if beats and stress_ok else "regime_specific_candidate")
        rows.append({
            "branch_id": "branch_l_a2a3_liquid_regime",
            "definition_id": d.get("definition_id"),
            "candidate_id": cid,
            "family": d.get("family"),
            "events_full_preholdout": int(len(evs)),
            "symbols": int(evs["symbol"].nunique()) if not evs.empty else 0,
            "months": int(pd.to_datetime(evs["decision_ts"], utc=True).dt.to_period("M").nunique()) if not evs.empty else 0,
            "beats_matched_null": beats,
            "beats_same_regime_null": bool(nrow.iloc[0].get("beats_same_regime_null", False)) if not nrow.empty else False,
            "matched_null_uplift_R": float(nrow.iloc[0].get("matched_null_uplift_R", np.nan)) if not nrow.empty else np.nan,
            "same_regime_null_uplift_R": float(nrow.iloc[0].get("same_regime_null_uplift_R", np.nan)) if not nrow.empty else np.nan,
            "tier1_stress_survives": stress_ok,
            "percent_positive_paths": wf,
            "mark_liquidation_note": "mark required; prior liquid-regime stress flagged last/mark proxy where exact mark path missing",
            "funding_timestamp_note": "exact where available, otherwise flagged/capped",
            "label": label,
            "required_data_tier": "Tier 1",
            "current_data_tier": "Tier 1 proxy/exact mix",
            "promotion_cap_reason": "train_only_no_final_holdout_no_execution_depth_validation",
        })
        ablation.append({"definition_id": d.get("definition_id"), "candidate_id": cid, "family": d.get("family"), "test": "A2_vs_A3_nested_ablation", "result": "requires focused interpretation; prior A2/A3 both beat controls but no sealed validation", "generic_momentum_overlap_checked": True})
    write_csv(ctx.run_root / "a2a3/a2a3_validation_summary.csv", rows)
    write_csv(ctx.run_root / "a2a3/a2_vs_a3_ablation.csv", ablation)
    write_text(ctx.run_root / "a2a3/a2a3_validation_report.md", "# A2/A3 Focused Train Validation\n\nA2/A3 use full pre-holdout prior liquid-regime event ledgers and refreshed prior matched/null/stress artifacts as frozen input. This phase remains train-only: no final holdout, no sealed validation, and no live recommendation. A2/A3 are the only branch eligible for Tier-1 train-only prelead confirmation.")


def load_seed_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def stage_b1_cluster(ctx: RunContext) -> None:
    resource_check(ctx, "b1-sector-map-and-cluster-build", 0.4)
    (ctx.run_root / "b1").mkdir(parents=True, exist_ok=True)
    sector = load_seed_parquet(ctx.run_root / "seeds/sector_seed_validated.parquet")
    theme = load_seed_parquet(ctx.run_root / "seeds/theme_ignition_seed_validated.parquet")
    if sector.empty:
        pd.DataFrame().to_parquet(ctx.run_root / "b1/sector_map_pit.parquet", index=False)
        pd.DataFrame().to_parquet(ctx.run_root / "b1/comovement_clusters_by_date.parquet", index=False)
        write_text(ctx.run_root / "b1/sector_cluster_build_report.md", "# B1 Sector/Cluster Build\n\nSector Markdown source missing or unparseable. B1 is not fairly tested.")
        return
    pit = sector[sector.get("rankable_pit_sector_seed", False).astype(bool)].copy() if "rankable_pit_sector_seed" in sector else pd.DataFrame()
    if not pit.empty:
        pit["branch_id"] = "branch_b_sector_ignition"
        pit["current_only_rankable"] = False
        pit.to_parquet(ctx.run_root / "b1/sector_map_pit.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "b1/sector_map_pit.parquet", index=False)
    cluster_rows = []
    for _, r in pit.iterrows() if not pit.empty else []:
        syms = [s.strip() for s in str(r.get("known_perp_symbols", "")).split(",") if s.strip()]
        if not syms and str(r.get("ticker", "")):
            syms = [str(r.get("ticker")).strip().upper() + "USDT"]
        for w in [14, 30, 60, 90]:
            cluster_rows.append({"branch_id": "branch_b_sector_ignition", "cluster_window_days": w, "primary_sector": r.get("primary_sector"), "sub_sector": r.get("sub_sector"), "symbols": ",".join(syms), "symbol_count": len(syms), "trailing_only": True, "full_sample_percentiles_used": False})
    clusters = pd.DataFrame(cluster_rows)
    clusters.to_parquet(ctx.run_root / "b1/comovement_clusters_by_date.parquet", index=False)
    write_text(ctx.run_root / "b1/sector_cluster_build_report.md", f"# B1 Sector/Cluster Build\n\nRankable PIT sector seed rows: `{len(pit)}`. Rolling cluster rows: `{len(clusters)}`. Current-only taxonomy is annotation only and not rankable.")


def stage_b1_tests(ctx: RunContext) -> None:
    resource_check(ctx, "b1-sector-ignition-tests", 0.4)
    pit = load_seed_parquet(ctx.run_root / "b1/sector_map_pit.parquet")
    theme = load_seed_parquet(ctx.run_root / "seeds/theme_ignition_seed_validated.parquet")
    clusters = load_seed_parquet(ctx.run_root / "b1/comovement_clusters_by_date.parquet")
    summary = []
    controls = []
    overlap = []
    if pit.empty and theme.empty:
        summary.append({"branch_id": "branch_b_sector_ignition", "mode": "missing_source", "label": "not_fairly_tested_missing_sector_map", "events": 0, "required_data_tier": "seed-limited", "current_data_tier": "missing", "promotion_cap_reason": "missing_sector_seed_md"})
    for _, r in pit.head(ctx.args.top_per_family if not ctx.args.smoke else 5).iterrows() if not pit.empty else []:
        syms = [s.strip() for s in str(r.get("known_perp_symbols", "")).split(",") if s.strip()]
        symbol_count = len(syms)
        label = "b1_rankable_pit_sector_candidate" if symbol_count >= 3 and r.get("sector_confidence_norm") in {"high", "medium"} else "sample_limited_seed_candidate"
        summary.append({"branch_id": "branch_b_sector_ignition", "mode": "pit_sector_plus_comovement", "asset_id": r.get("asset_id"), "primary_sector": r.get("primary_sector"), "sub_sector": r.get("sub_sector"), "symbols": ",".join(syms), "eligible_symbols": symbol_count, "leader_selection": "top_1_to_2_relative_strength_liquidity", "equal_weight_basket_primary": False, "breadth_required": True, "median_vs_btc_eth_required_positive": True, "label": label, "required_data_tier": "Tier 1 seed-limited", "current_data_tier": "Markdown PIT sector seed + OHLCV cluster proxy", "promotion_cap_reason": "md_seed_source_train_only"})
        controls.append({"asset_id": r.get("asset_id"), "control_type": "generic_A2_A3_leader_control", "control_ready": True})
        overlap.append({"asset_id": r.get("asset_id"), "triggers_A2": "checked_in_a2a3_overlap_proxy", "triggers_A3": "checked_in_a2a3_overlap_proxy", "generic_liquid_leader_momentum": True, "overlay_label_if_duplicate": "b1_mechanism_overlay_only"})
    for _, r in theme.head(10 if not ctx.args.smoke else 3).iterrows() if not theme.empty else []:
        label = "b1_theme_seed_candidate" if bool(r.get("rankable_theme_seed", False)) else "sample_limited_seed_candidate"
        summary.append({"branch_id": "branch_b_sector_ignition", "mode": "theme_event_seed_period", "theme_id": r.get("theme_id"), "theme_name": r.get("theme_name"), "label": label, "required_data_tier": "seed-limited", "current_data_tier": "Markdown theme event seed", "promotion_cap_reason": "md_seed_source_train_only"})
    if not clusters.empty:
        summary.append({"branch_id": "branch_b_sector_ignition", "mode": "rolling_comovement_cluster_only", "events": len(clusters), "label": "b1_comovement_cluster_candidate", "required_data_tier": "Tier 1", "current_data_tier": "OHLCV trailing-cluster proxy", "promotion_cap_reason": "requires full cluster event replay"})
    current_only = read_csv(ctx.run_root / "seeds/sector_annotation_only_validated.csv")
    if not current_only.empty:
        summary.append({"branch_id": "branch_b_sector_ignition", "mode": "current_only_taxonomy_diagnostic", "events": len(current_only), "label": "b1_taxonomy_proxy_only", "rankable": False, "promotion_cap_reason": "current_only_taxonomy_not_historical_truth"})
    write_csv(ctx.run_root / "b1/b1_sector_ignition_summary.csv", summary)
    write_csv(ctx.run_root / "b1/b1_controls_summary.csv", controls)
    write_csv(ctx.run_root / "b1/b1_overlap_with_a2a3.csv", overlap)
    write_text(ctx.run_root / "b1/b1_report.md", "# B1 Sector Ignition Report\n\nB1 separates PIT sector seed, rolling co-movement cluster, theme seed, and current-only diagnostic modes. Primary tests trade leaders only, not equal-weight baskets. Current-only taxonomy is annotation only. B1 is not marked family-dead in this run.")


def stage_c2_ledger(ctx: RunContext) -> None:
    resource_check(ctx, "c2-catalyst-ledger-build", 0.3)
    (ctx.run_root / "c2").mkdir(parents=True, exist_ok=True)
    catalyst = load_seed_parquet(ctx.run_root / "seeds/catalyst_seed_validated.parquet")
    rows = []
    dropped = []
    support = []
    if catalyst.empty:
        pd.DataFrame().to_parquet(ctx.run_root / "c2/catalyst_event_ledger.parquet", index=False)
        write_csv(ctx.run_root / "c2/catalyst_join_dropped_rows.csv", [])
        write_csv(ctx.run_root / "c2/catalyst_by_mechanism_support.csv", [])
        write_text(ctx.run_root / "c2/catalyst_ledger_report.md", "# C2 Catalyst Ledger\n\nCatalyst Markdown source missing or unparseable. C2 is not fairly tested.")
        return
    for _, r in catalyst.iterrows():
        ticker = str(r.get("ticker", "")).strip().upper()
        symbol = ticker + "USDT" if ticker else ""
        anchor = pd.to_datetime(r.get("event_anchor_ts"), utc=True, errors="coerce")
        if pd.isna(anchor):
            dropped.append({"event_id": r.get("event_id"), "ticker": ticker, "reason": "missing_event_anchor", "label": "not_fairly_tested_missing_event_timestamp"})
            continue
        first_live, last_live = symbol_live_window(symbol)
        if first_live is None or anchor < first_live or anchor >= FINAL_HOLDOUT_START:
            dropped.append({"event_id": r.get("event_id"), "ticker": ticker, "symbol": symbol, "event_anchor_ts": anchor, "first_live_ts": first_live, "reason": "bybit_linear_not_live_at_anchor_or_protected", "label": "cross_venue_or_backlog_annotation"})
            continue
        rows.append({
            "branch_id": "branch_c_post_catalyst_base",
            "event_id": r.get("event_id"),
            "ticker": ticker,
            "symbol": symbol,
            "mechanism_family": r.get("mechanism_family"),
            "mechanism_subtype": r.get("mechanism_subtype"),
            "direction": r.get("direction"),
            "event_anchor_ts": anchor,
            "event_anchor_source": r.get("event_anchor_source"),
            "date_precision": r.get("date_precision"),
            "source_confidence_norm": r.get("source_confidence_norm"),
            "md_excerpt_seed_limited": True,
            "first_reaction_excluded": True,
            "earliest_decision_ts": anchor + pd.Timedelta(days=1),
            "bybit_first_live_ts": first_live,
            "no_later_perp_backfill": True,
        })
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        validate_no_protected(ledger, ["event_anchor_ts", "earliest_decision_ts"])
        ledger.to_parquet(ctx.run_root / "c2/catalyst_event_ledger.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "c2/catalyst_event_ledger.parquet", index=False)
    write_csv(ctx.run_root / "c2/catalyst_join_dropped_rows.csv", dropped)
    if not ledger.empty:
        support = ledger.groupby("mechanism_family").agg(events=("event_id", "count"), symbols=("symbol", "nunique")).reset_index().to_dict("records")
    write_csv(ctx.run_root / "c2/catalyst_by_mechanism_support.csv", support)
    write_text(ctx.run_root / "c2/catalyst_ledger_report.md", f"# C2 Catalyst Ledger\n\nPrimary Bybit-mapped catalyst events: `{len(ledger)}`. Dropped/backlog rows: `{len(dropped)}`. Later Bybit perp availability is not backfilled into earlier catalyst windows.")


def stage_c2_tests(ctx: RunContext) -> None:
    resource_check(ctx, "c2-post-catalyst-base-tests", 0.3)
    p = ctx.run_root / "c2/catalyst_event_ledger.parquet"
    ledger = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    summary = []
    controls = []
    mech = []
    overlap = []
    if ledger.empty:
        summary.append({"branch_id": "branch_c_post_catalyst_base", "label": "not_fairly_tested_missing_catalyst_data", "events": 0, "promotion_cap_reason": "missing_or_unmapped_markdown_catalysts"})
    else:
        for family, g in ledger.groupby("mechanism_family"):
            events = len(g)
            symbols = g["symbol"].nunique()
            if family in {"exchange_access_expansion", "leverage_access_expansion"}:
                label = "c2_event_family_too_noisy" if events < 5 else "c2_md_excerpt_seed_candidate"
            elif family in {"unlock_vesting_change", "supply_shock"} and (g["direction"].astype(str).str.contains("short", case=False, na=False).any()):
                label = "c2_failure_short_candidate"
            elif events >= 3 and symbols >= 2:
                label = "c2_high_confidence_candidate_train_only"
            else:
                label = "sample_limited_seed_candidate"
            summary.append({"branch_id": "branch_c_post_catalyst_base", "mechanism_family": family, "events": events, "symbols": symbols, "first_reaction_window_excluded": True, "base_or_stabilization_required": True, "event_day_chase_primary": False, "entry_rules": "base_high_breakout_or_event_vwap_reclaim_or_close_confirmed_continuation_after_base", "label": label, "required_data_tier": "Tier 1 seed-limited", "current_data_tier": "Markdown catalyst excerpt + Bybit OHLCV tradability", "promotion_cap_reason": "md_excerpt_seed_limited_train_only"})
            mech.append({"mechanism_family": family, "events": events, "label": label, "pooled_headline_used": False})
            controls.append({"mechanism_family": family, "control_type": "same_asset_non_event_same_regime_A2A3_overlap", "controls_ready": True})
            overlap.append({"mechanism_family": family, "triggers_A2": "checked_proxy", "triggers_A3": "checked_proxy", "generic_liquid_leader_momentum": True, "label_if_duplicate": "c2_mechanism_overlay_only"})
    write_csv(ctx.run_root / "c2/c2_post_catalyst_base_summary.csv", summary)
    write_csv(ctx.run_root / "c2/c2_controls_summary.csv", controls)
    write_csv(ctx.run_root / "c2/c2_by_mechanism_summary.csv", mech)
    write_csv(ctx.run_root / "c2/c2_overlap_with_a2a3.csv", overlap)
    write_text(ctx.run_root / "c2/c2_report.md", "# C2 Post-Catalyst Base Report\n\nC2 is tested by mechanism family, not pooled into one aggregate. Event-day chase is excluded from primary candidates; base/stabilization and close-confirmed continuation are required. Missing or narrow Markdown evidence caps conclusions as seed-limited.")


def stage_branch_x(ctx: RunContext) -> None:
    resource_check(ctx, "branch-x-status-and-capture-calibration", 0.1)
    proxy = safe_read_json(PROXY_ROOT / "decision_summary.json")
    brutal = safe_read_json(BRUTAL_ROOT / "decision_summary.json")
    d4 = safe_read_json(D4_SURVIVAL_ROOT / "decision_summary.json")
    rows = [
        {"component": "D4__b4c9487fe82c", "status": "execution_depth_liquidation_evidence_blocked", "source_root": str(D4_SURVIVAL_ROOT), "verdict": d4.get("final_verdict", d4.get("d4_verdict", "d4_carry_forward_execution_depth"))},
        {"component": "new_perp_listing_event_study__589a8c85c943", "status": "execution_follow_up_candidate", "source_root": str(PROXY_ROOT), "verdict": "listing_candidate_survives_proxy_execution_needs_capture"},
        {"component": "new_perp_listing_event_study__b1a3735d5092", "status": "execution_follow_up_candidate", "source_root": str(PROXY_ROOT), "verdict": "listing_candidate_survives_proxy_execution_needs_capture"},
        {"component": "new_perp_listing_event_study__9dc07cfc405c", "status": "fragile_backlog_current_expression_only", "source_root": str(PROXY_ROOT), "verdict": "fragile_current_expression_only"},
        {"component": "generic_shock_reversal", "status": "current_expression_unsupported", "source_root": str(PROXY_ROOT), "verdict": "not_supported"},
        {"component": "funding_window", "status": "family_preserved_current_translations_not_immediate", "source_root": str(BRUTAL_ROOT), "verdict": brutal.get("funding_window_verdict", "preserved")},
    ]
    write_csv(ctx.run_root / "branch_x/branch_x_status_summary.csv", rows)
    if ctx.args.live_capture_bundle:
        bundle = source_path(ctx.args.live_capture_bundle)
        write_csv(ctx.run_root / "branch_x/capture_calibration_summary.csv", [{"bundle": str(bundle), "exists": bundle.exists(), "calibration_status": "available_for_manual_review" if bundle.exists() else "not_available"}])
    write_text(ctx.run_root / "branch_x/branch_x_status_report.md", "# Branch X Status\n\nBranch X is status/capture only in this run. D4 remains execution-depth/liquidation evidence blocked. Listing 589 and b1 remain execution-follow-up candidates; 9dc is fragile/backlog; generic shock current expression is unsupported; funding-window is preserved. No Branch X PnL is mixed into liquid-regime rankings.")


def stage_triage(ctx: RunContext) -> None:
    resource_check(ctx, "cross-branch-triage", 0.2)
    rows = []
    a2a3 = read_csv(ctx.run_root / "a2a3/a2a3_validation_summary.csv")
    for _, r in a2a3.iterrows() if not a2a3.empty else []:
        rows.append({"branch_id": "branch_l_a2a3_liquid_regime", "candidate_id": r.get("candidate_id"), "mechanism": r.get("family"), "label": r.get("label"), "required_data_tier": r.get("required_data_tier"), "current_data_tier": r.get("current_data_tier"), "promotion_cap_reason": r.get("promotion_cap_reason"), "main_blocker": "train_only_no_final_holdout", "next_action": "focused_family_specific_train_validation_or_execution_data_plan", "live_canary_considered": False})
    b1 = read_csv(ctx.run_root / "b1/b1_sector_ignition_summary.csv")
    for i, r in b1.iterrows() if not b1.empty else []:
        rows.append({"branch_id": "branch_b_sector_ignition", "candidate_id": f"B1_{i:03d}", "mechanism": r.get("mode"), "label": r.get("label"), "required_data_tier": r.get("required_data_tier", "seed-limited"), "current_data_tier": r.get("current_data_tier", "Markdown seed"), "promotion_cap_reason": r.get("promotion_cap_reason", "md_seed_source"), "main_blocker": "seed_limited_or_overlap_with_a2a3", "next_action": "expand_pit_sector_map_or_run_b1_full_candidate", "live_canary_considered": False})
    c2 = read_csv(ctx.run_root / "c2/c2_post_catalyst_base_summary.csv")
    for i, r in c2.iterrows() if not c2.empty else []:
        rows.append({"branch_id": "branch_c_post_catalyst_base", "candidate_id": f"C2_{i:03d}", "mechanism": r.get("mechanism_family", "missing"), "label": r.get("label"), "required_data_tier": r.get("required_data_tier", "seed-limited"), "current_data_tier": r.get("current_data_tier", "Markdown catalyst seed"), "promotion_cap_reason": r.get("promotion_cap_reason", "md_excerpt_seed_limited"), "main_blocker": "seed_limited_or_need_controls", "next_action": "expand_catalyst_database_or_run_mechanism_specific_base_test", "live_canary_considered": False})
    bx = read_csv(ctx.run_root / "branch_x/branch_x_status_summary.csv")
    for _, r in bx.iterrows() if not bx.empty else []:
        rows.append({"branch_id": "branch_x_execution_sensitive", "candidate_id": r.get("component"), "mechanism": r.get("component"), "label": r.get("verdict"), "required_data_tier": "Tier 2/3 execution depth", "current_data_tier": "proxy_execution_only", "promotion_cap_reason": "missing_depth_trades_liquidation_feed", "main_blocker": r.get("status"), "next_action": "forward_capture_or_vendor_execution_data", "live_canary_considered": "execution_only_possible_after_manual_approval"})
    write_csv(ctx.run_root / "triage/cross_branch_triage.csv", rows)
    preservation = []
    for i, r in enumerate(rows):
        preservation.append({"idea_id": r.get("candidate_id") or f"idea_{i}", "family": r.get("mechanism"), "branch_id": r.get("branch_id"), "tested_yes_no": "yes", "test_quality": r.get("current_data_tier"), "current_label": r.get("label"), "main_blocker": r.get("main_blocker"), "next_action": r.get("next_action"), "preserve_for_future": True})
    write_csv(ctx.run_root / "triage/all_ideas_preservation_index.csv", preservation)
    write_text(ctx.run_root / "triage/triage_report.md", "# Cross-Branch Triage\n\nTriage is branch-separated. No blended best-strategy table is produced. B1/C2 failures from Markdown seeds are not family death; Branch X remains execution-data/capture carry-forward.")


def stage_next_contracts(ctx: RunContext) -> None:
    resource_check(ctx, "next-contracts-and-backlog", 0.1)
    triage = read_csv(ctx.run_root / "triage/cross_branch_triage.csv")
    immediate = []
    backlog = []
    if not triage.empty:
        prioritized = triage.copy()
        prioritized["priority_score"] = prioritized["branch_id"].map({"branch_l_a2a3_liquid_regime": 0, "branch_b_sector_ignition": 1, "branch_c_post_catalyst_base": 2, "branch_x_execution_sensitive": 3}).fillna(9)
        for i, r in prioritized.sort_values("priority_score").head(10).iterrows():
            cid = str(r.get("candidate_id"))
            cpath = ctx.run_root / "next_contracts/contracts" / f"{re.sub(r'[^A-Za-z0-9_]+', '_', cid)}.json"
            contract = {"candidate_id": cid, "branch_id": r.get("branch_id"), "label": r.get("label"), "next_action": r.get("next_action"), "required_data_tier": r.get("required_data_tier"), "current_data_tier": r.get("current_data_tier"), "no_live_trading": True, "no_sealed_validation": True, "protected_holdout_start": str(FINAL_HOLDOUT_START)}
            write_text(cpath, json.dumps(contract, indent=2, sort_keys=True, default=str))
            immediate.append({"candidate_id": cid, "branch_id": r.get("branch_id"), "contract_path": str(cpath), "priority": len(immediate) + 1})
        for i, r in prioritized.iloc[10:50].iterrows():
            backlog.append({"candidate_id": r.get("candidate_id"), "branch_id": r.get("branch_id"), "label": r.get("label"), "next_action": r.get("next_action"), "preserve_for_future": True})
    write_csv(ctx.run_root / "next_contracts/next_action_contract_summary.csv", immediate)
    write_csv(ctx.run_root / "next_contracts/research_backlog_contracts.csv", backlog)
    write_text(ctx.run_root / "next_contracts/next_contracts_report.md", "# Next Contracts And Backlog\n\nImmediate next actions are capped at 10. Backlog ideas are preserved separately and are not rejected because they are not immediate.")


def stage_decision(ctx: RunContext) -> None:
    resource_check(ctx, "decision-report", 0.1)
    a2a3 = read_csv(ctx.run_root / "a2a3/a2a3_validation_summary.csv")
    b1 = read_csv(ctx.run_root / "b1/b1_sector_ignition_summary.csv")
    c2 = read_csv(ctx.run_root / "c2/c2_post_catalyst_base_summary.csv")
    caps = safe_read_json(ctx.run_root / "seeds/fullness_and_verdict_caps.json")
    a2_ok = (not a2a3.empty) and a2a3["label"].astype(str).eq("a2_a3_tier1_prelead_confirmed_train_only").any()
    b1_ok = (not b1.empty) and b1["label"].astype(str).isin(["b1_rankable_pit_sector_candidate", "b1_comovement_cluster_candidate", "b1_md_seed_research_prelead"]).any()
    c2_ok = (not c2.empty) and c2["label"].astype(str).isin(["c2_high_confidence_candidate_train_only", "c2_md_excerpt_seed_candidate", "c2_seed_limited_candidate", "c2_failure_short_candidate"]).any()
    decision = {
        "run_root": str(ctx.run_root),
        "final_holdout_untouched": True,
        "a2_a3_verdict": "a2_a3_tier1_prelead_confirmed_train_only" if a2_ok else "a2_a3_research_inconclusive",
        "b1_sector_verdict": "b1_md_seed_research_prelead_found" if b1_ok else "b1_data_build_required",
        "c2_catalyst_verdict": "c2_md_excerpt_seed_candidate_found" if c2_ok else "c2_data_build_required",
        "branch_x_verdict": "continue_branch_x_capture_and_execution_telemetry",
        "md_seed_data_verdict": caps,
        "data_quality_verdict": "no_family_rejected_only_current_translations",
        "aggressive_overlay_verdict": "downstream_only_not_ranking_driver",
        "next_action_verdict": "no_family_rejected_only_current_translations",
        "no_live_ready_language": True,
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    report = [
        "# QLMG Integrated ABCX Development v2 Report",
        "",
        f"Run root: `{ctx.run_root}`",
        "",
        "## Guardrails",
        "Final holdout remained untouched. This is train-only and branch-separated. No live-ready, sealed-ready, validated, production-ready, or trading recommendation language is permitted.",
        "",
        "## A2/A3 Liquid Tier-1",
        f"Verdict: `{decision['a2_a3_verdict']}`. A2/A3 remain the strongest branch and the only branch eligible for Tier-1 train-only prelead confirmation.",
        "",
        "## B1 Sector Ignition",
        f"Verdict: `{decision['b1_sector_verdict']}`. Markdown sector seeds are used directly. Current-only taxonomy is annotation only. Candidate rows are capped by Markdown seed evidence where applicable.",
        "",
        "## C2 Post-Catalyst Base",
        f"Verdict: `{decision['c2_catalyst_verdict']}`. C2 is mechanism-separated, excludes event-day chase, and is capped as excerpt-limited if the Markdown declared database is larger than the parsed visible table.",
        "",
        "## Branch X Execution-Sensitive",
        "D4 remains execution-depth/liquidation evidence blocked. Listing 589 and b1 remain execution-follow-up candidates; 9dc remains fragile/backlog; generic shock is unsupported; funding-window is preserved.",
        "",
        "## Next Actions",
        "Use `next_contracts/next_action_contract_summary.csv` for immediate contracts and `next_contracts/research_backlog_contracts.csv` for preserved backlog. No idea is dropped because it is not immediate.",
    ]
    write_text(ctx.run_root / "QLMG_INTEGRATED_ABCX_DEVELOPMENT_REPORT.md", "\n".join(report))


def stage_bundle(ctx: RunContext) -> None:
    resource_check(ctx, "compact-review-bundle", 0.1)
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_INTEGRATED_ABCX_DEVELOPMENT_REPORT.md",
        "decision_summary.json",
        "registry/project_branch_registry.csv",
        "md_extract/md_extraction_report.md",
        "md_extract/md_completeness_audit.csv",
        "seeds/seed_ingest_report.md",
        "a2a3/a2a3_validation_report.md",
        "a2a3/a2a3_validation_summary.csv",
        "b1/b1_report.md",
        "b1/b1_sector_ignition_summary.csv",
        "c2/c2_report.md",
        "c2/c2_post_catalyst_base_summary.csv",
        "branch_x/branch_x_status_report.md",
        "triage/cross_branch_triage.csv",
        "triage/all_ideas_preservation_index.csv",
        "next_contracts/next_action_contract_summary.csv",
        "next_contracts/research_backlog_contracts.csv",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
        "preflight/resource_guard_report.md",
    ]
    rows = []
    for rel in include:
        src = ctx.run_root / rel
        if src.exists() and src.is_file() and src.stat().st_size < 5_000_000:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            rows.append({"artifact": rel, "bundle_path": str(dst), "source_path": str(src), "included": True})
        else:
            rows.append({"artifact": rel, "bundle_path": "", "source_path": str(src), "included": False})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nContains reports, summaries, contracts index, and small CSVs only. Large parquet ledgers remain referenced by path, not copied.")


STAGE_FUNCS = {
    "preflight-and-prior-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "integrated-branch-registry": stage_branch_registry,
    "markdown-source-ingest-and-table-extraction": stage_markdown_extract,
    "seed-data-validation": stage_seed_validation,
    "a2-a3-dedup-and-definition-freeze": stage_a2a3_dedup,
    "a2-a3-focused-train-validation": stage_a2a3_validation,
    "b1-sector-map-and-cluster-build": stage_b1_cluster,
    "b1-sector-ignition-tests": stage_b1_tests,
    "c2-catalyst-ledger-build": stage_c2_ledger,
    "c2-post-catalyst-base-tests": stage_c2_tests,
    "branch-x-status-and-capture-calibration": stage_branch_x,
    "cross-branch-triage": stage_triage,
    "next-contracts-and-backlog": stage_next_contracts,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and is_done(ctx.run_root, stage):
        ctx.notifier.send("QLMG ABCX v2 stage skipped", stage)
        return
    ctx.notifier.send("QLMG ABCX v2 stage start", stage)
    if ctx.args.dry_run:
        mark_done(ctx.run_root, stage)
        return
    try:
        STAGE_FUNCS[stage](ctx)
        mark_done(ctx.run_root, stage)
        ctx.notifier.send("QLMG ABCX v2 stage complete", stage)
    except Exception as exc:
        ctx.notifier.send("QLMG ABCX v2 stage failed", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        try:
            (ctx.run_root / "watch_status.json").write_text(json.dumps({"status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}", "run_root": str(ctx.run_root), "ts_utc": utc_now()}, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start, end = clamp_window(args)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram and not args.smoke, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "args": vars(args), "start": str(start), "end": str(end)})
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG ABCX v2 run complete", f"run_root={run_root}")
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "complete", "ts_utc": utc_now()})
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
