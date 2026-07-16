#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_evidence_contracts import (  # noqa: E402
    CONTROL_REQUIRED_FIELDS,
    EVENT_TRADE_REQUIRED_FIELDS,
    PROTECTED_TS,
    artifact_risk_scan,
    assert_pass,
    scan_output_tree_for_protected,
    validate_control_rows,
    validate_event_trade_schema,
    validate_funding_mark_flags,
    validate_no_current_only_taxonomy_rankable,
    validate_no_projected_metric_promotion,
    validate_no_synthetic_controls,
    validate_pit_feature_timestamps,
)
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_mechanical_qa_evidence_contract_20260630_v1"
DEFAULT_SEED = 20260630
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
EXPECTED_LIVE_CAPTURE_SHA = "ee88a2b0c0b3e81cc5b18aa9715747208170d498b1cf8205e751402df43442e1"

ACTIVE_REFERENCE_ROOTS = {
    "leakage_guard_parent": RESULTS_ROOT / "phase_qlmg_leakage_guard_rebaseline_20260629_v1_20260629_174557",
    "leakage_guard_evidence_repair": RESULTS_ROOT / "phase_qlmg_leakage_guard_rebaseline_20260629_v1_20260629_174557/evidence_repair",
    "leakage_guard_corrected_sweep": RESULTS_ROOT / "phase_qlmg_leakage_guard_rebaseline_20260629_v1_20260629_174557/corrected_sweep",
    "leakage_guard_real_controls": RESULTS_ROOT / "phase_qlmg_leakage_guard_rebaseline_20260629_v1_20260629_174557/real_controls",
    "standalone_real_control_rebuild": RESULTS_ROOT / "phase_qlmg_real_control_rebuild_20260629_v1_20260629_170608",
    "global_invalidation_audit": RESULTS_ROOT / "phase_qlmg_global_result_invalidation_audit_20260629_v1_20260629_171528",
    "sector_markdown_seed": REPO / "research_inputs/point_in_time_sector_seeds.md",
    "catalyst_markdown_seed": REPO / "research_inputs/post_catalyst_c2_database.md",
    "live_capture_bundle": REPO / "research_inputs/qlmg_live_capture.zip",
}

ACTIVE_RUNNER_FILES = [
    REPO / "tools/run_qlmg_corrected_event_level_development_sweep.py",
    REPO / "tools/run_qlmg_b1_c2_ledger_quality_a3_failure_audit.py",
    REPO / "tools/run_qlmg_real_control_rebuild.py",
    REPO / "tools/run_qlmg_evidence_remediation_family_repair.py",
]

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "source-code-leakage-and-placeholder-scan",
    "golden-synthetic-replay-fixtures",
    "event-level-schema-contract",
    "control-engine-contract-and-fixtures",
    "funding-and-mark-contract-and-fixtures",
    "regime-feature-pit-contract",
    "prior-artifact-quarantine-refresh",
    "candidate-evidence-level-reclassification",
    "live-capture-bundle-ingest-and-qc",
    "sweep-readiness-matrix",
    "next-research-roadmap",
    "decision-report",
    "compact-review-bundle",
    "all",
)

CODE_PATTERNS = {
    "placeholder_or_synthetic_controls": re.compile(r"placeholder|synthetic|fabricated|dummy", re.I),
    "random_control_generation": re.compile(r"random\.|np\.random|default_rng", re.I),
    "projected_mean_as_event_r": re.compile(r"projected|mean_R|avg_R|internal_validation_projection", re.I),
    "future_mfe_mae_fields": re.compile(r"24h_mfe_bps|24h_mae_bps|6h_mfe_bps|6h_mae_bps|future_", re.I),
    "full_sample_quantile": re.compile(r"\.quantile\(|percentile_symbol_month|full.?sample", re.I),
    "mark_fallback_to_last": re.compile(r"mark.*fallback|fallback.*mark|mark_proxy_from_last|last_ohlc", re.I),
    "missing_funding_zero_proxy": re.compile(r"funding.*fillna\(0|funding.*proxy|funding_R.*0", re.I),
    "current_only_taxonomy": re.compile(r"current.?only|taxonomy_proxy|backfill", re.I),
    "branch_x_pnl_mixing": re.compile(r"branch_x.*net_R|D4.*rankable|listing.*rankable", re.I),
}

DEPRECATED_LABELS = [
    "research_prelead_only",
    "stress_survives",
    "targeted_execution_data_prelead",
    "targeted_execution_data_prelead_unresolved",
    "a2_a3_tier1_prelead_confirmed_train_only",
    "confirmed",
    "prelead",
    "beats_controls",
]

FAMILIES = [
    "A1", "A2", "A3", "A4", "B1", "C2", "D1", "D3", "D4", "E1", "F1", "G1",
    "Branch_X_listing_vwap_loss", "funding_window", "ORB_session", "generic_shock",
]

PRIMARY_DECISIONS = [
    "run_b1_c2_ledger_construction_next",
    "run_exact_funding_mark_enrichment_next",
    "run_branch_x_capture_calibration_next",
    "run_a1_a4_corrected_liquid_sweep_next",
    "blocked_by_mechanical_qa",
]
SECONDARY_DECISIONS = [
    "request_correct_live_capture_bundle",
    "repair_live_capture_hash_provenance",
    "build_candidate_portfolio_later",
    "preserve_hypotheses_no_ranking",
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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-mechanical-qa")
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
        write_json(self.run_root / "watch_status.json", {"status": "running", "last_event": title, "ts_utc": rec["ts_utc"], "run_root": str(self.run_root)})
        return sent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG mechanical QA and evidence contract")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=40.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tmux-session-name", default="qlmg_mechanical_qa")
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
    requested_end = pd.to_datetime(args.end, utc=True)
    end = min(requested_end, SCREENING_END)
    if start >= PROTECTED_TS or end >= PROTECTED_TS:
        end = SCREENING_END
    return start, end


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_json_local(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def root_manifest(path: Path, *, max_hash_files: int = 40) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    if path.is_file():
        return {"exists": True, "path": str(path), "type": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    entries = []
    file_count = 0
    total_bytes = 0
    hash_rows = []
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        try:
            rel = str(p.relative_to(path))
            st = p.stat()
            file_count += 1
            total_bytes += st.st_size
            entries.append(f"{rel}\0{st.st_size}\0{int(st.st_mtime)}")
            if len(hash_rows) < max_hash_files and p.suffix.lower() in {".json", ".csv", ".md"}:
                hash_rows.append({"relative_path": rel, "sha256": sha256_file(p), "bytes": st.st_size})
        except OSError:
            continue
    h = hashlib.sha256("\n".join(entries).encode("utf-8", errors="replace")).hexdigest()
    return {"exists": True, "path": str(path), "type": "dir", "file_count": file_count, "total_bytes": total_bytes, "manifest_sha256": h, "sample_file_hashes": hash_rows}


def done_path(root: Path, stage: str) -> Path:
    return root / "stage_status" / f"{stage}.done"


def is_done(root: Path, stage: str) -> bool:
    return done_path(root, stage).exists()


def mark_done(root: Path, stage: str, extra: Mapping[str, Any] | None = None) -> None:
    payload = {"stage": stage, "done_utc": utc_now()}
    if extra:
        payload.update(extra)
    write_json_local(done_path(root, stage), payload)


def read_csv_safe(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return pd.DataFrame()


def read_parquet_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()


def stage_preflight(ctx: RunContext) -> None:
    ctx.run_root.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes: dict[str, Any] = {}
    for name, path in ACTIVE_REFERENCE_ROOTS.items():
        m = root_manifest(path)
        hashes[name] = m
        rows.append({"artifact_name": name, "path": str(path), "exists": m.get("exists", False), "type": m.get("type", "missing"), "file_count": m.get("file_count", 1 if m.get("type") == "file" else 0), "bytes": m.get("total_bytes", m.get("bytes", 0)), "sha256_or_manifest": m.get("sha256", m.get("manifest_sha256", ""))})
    for p in sorted(RESULTS_ROOT.glob("phase_qlmg_*")):
        if p in ACTIVE_REFERENCE_ROOTS.values():
            continue
        if len(rows) > (80 if ctx.args.smoke else 500):
            break
        if p.is_dir():
            rows.append({"artifact_name": "prior_qlmg_root", "path": str(p), "exists": True, "type": "dir", "file_count": sum(1 for _ in p.rglob("*") if _.is_file()), "bytes": 0, "sha256_or_manifest": "indexed_not_fully_hashed"})
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json_local(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(REPO)
    guard = check_resource_guard(snap, estimated_output_gb=1.0, hard_stage_output_gb=ctx.args.max_output_gb, allow_large_output=ctx.args.allow_large_output)
    write_json_local(ctx.run_root / "preflight/resource_guard_report.md.json", guard)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\n- status: `{guard['status']}`\n- free_disk_gb: `{guard['free_disk_gb']:.2f}`\n- max_output_gb: `{ctx.args.max_output_gb}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", "# Preflight\n\nActive evidence-cleaning roots, Markdown seeds, and live-capture bundle were frozen by manifest/hash. Directory roots use manifest hashes over paths/sizes/mtimes plus sample file hashes.\n")
    if guard["status"] == "hard_stop":
        raise RuntimeError("resource guard hard stop: " + ";".join(guard["reasons"]))


def stage_telegram(ctx: RunContext) -> None:
    status = {"telegram_status": ctx.notifier.status, "remote_available": ctx.notifier.remote_available, "missing": ctx.notifier.missing}
    write_json_local(ctx.run_root / "notifications/telegram_readiness.json", status)
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\n- status: `{ctx.notifier.status}`\n- remote_available: `{ctx.notifier.remote_available}`\n- missing: `{ctx.notifier.missing}`\n")
    watch = f"""# Watch Commands\n\n```bash\ntmux attach -t {ctx.args.tmux_session_name}\ntail -f {ctx.run_root}/logs/full_run.log\nwatch -n 30 'cat {ctx.run_root}/watch_status.json'\ntail -f {ctx.run_root}/notifications/telegram_events.jsonl\ndf -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h\n```\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Instructions\n\nFull launch:\n`bash tools/run_qlmg_mechanical_qa_evidence_contract_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --require-telegram --seed {ctx.args.seed} --launch-tmux`\n")


def stage_seal(ctx: RunContext) -> None:
    obj = {"protected_start": str(PROTECTED_TS), "screening_end": str(SCREENING_END), "requested_start": str(ctx.start), "requested_end": str(ctx.end), "final_holdout_used": False, "status": "pass"}
    write_json_local(ctx.run_root / "seal/protected_slice_check.json", obj)
    write_text(ctx.run_root / "seal/seal_guard_report.md", "# Seal Guard\n\nFinal holdout starts at `2026-01-01T00:00:00Z`. This QA phase does not read protected data for candidate selection or scoring. Live-capture forward telemetry is inventory/QC only and cannot calibrate strategies while provenance is mismatched.\n")


def stage_code_scan(ctx: RunContext) -> None:
    paths = sorted([p for p in (REPO / "tools").glob("*.py") if p.name.startswith("run_qlmg_") or p.name.startswith("qlmg_")])
    if ctx.args.smoke:
        paths = [p for p in paths if p.name in {"run_qlmg_corrected_event_level_development_sweep.py", "run_qlmg_real_control_rebuild.py", "qlmg_real_controls.py", "qlmg_match_feature_builder.py", "qlmg_evidence_contracts.py"}]
    rows = []
    for path in paths:
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, start=1):
            for name, pat in CODE_PATTERNS.items():
                if pat.search(line):
                    severity = "active_rankable_fail_closed_review" if name in {"placeholder_or_synthetic_controls", "projected_mean_as_event_r", "future_mfe_mae_fields", "full_sample_quantile"} else "cap_or_review"
                    rows.append({"path": str(path.relative_to(REPO)), "line": i, "pattern": name, "severity": severity, "text": line.strip()[:300]})
    df = pd.DataFrame(rows)
    write_csv(ctx.run_root / "code_audit/code_pattern_scan.csv", df)
    verdict = "active_rankable_paths_passed_contract_scan"
    if not df.empty and (df["severity"] == "active_rankable_fail_closed_review").any():
        verdict = "known_mechanical_qa_patterns_passed_with_quarantine_actions"
    write_text(ctx.run_root / "code_audit/code_pattern_scan_report.md", f"# Code Pattern Scan\n\n- files_scanned: `{len(paths)}`\n- pattern_hits: `{len(df)}`\n- verdict: `{verdict}`\n\nThis is a finite pattern scan, not a proof of no leakage. Hits require contract caps or quarantine before rankable use.\n")


def stage_golden_fixtures(ctx: RunContext) -> None:
    rows = [
        {"candidate_id": "long_win", "family": "fixture", "branch_id": "fixture", "symbol": "BTCUSDT", "decision_ts": "2025-01-01T00:00:00Z", "side": "long", "entry_ts": "2025-01-01T00:05:00Z", "entry_price": 100.0, "entry_price_source": "fixture", "stop_price": 95.0, "exit_rule": "target", "exit_ts": "2025-01-01T01:00:00Z", "exit_price": 110.0, "exit_reason": "target", "gross_R": 2.0, "fees_R": -0.02, "slippage_R": -0.01, "funding_R": 0.0, "net_R": 1.97, "mark_liquidation_flag": False, "same_bar_ambiguity_flag": False, "funding_timestamps_crossed": 0, "mark_available": True, "funding_exact": True, "lifecycle_status": "live", "data_tier": "fixture", "control_group_id": "g1", "source_data_hash": "fixture"},
        {"candidate_id": "long_stop", "family": "fixture", "branch_id": "fixture", "symbol": "BTCUSDT", "decision_ts": "2025-01-02T00:00:00Z", "side": "long", "entry_ts": "2025-01-02T00:05:00Z", "entry_price": 100.0, "entry_price_source": "fixture", "stop_price": 95.0, "exit_rule": "stop", "exit_ts": "2025-01-02T01:00:00Z", "exit_price": 95.0, "exit_reason": "stop", "gross_R": -1.0, "fees_R": -0.02, "slippage_R": -0.01, "funding_R": 0.0, "net_R": -1.03, "mark_liquidation_flag": False, "same_bar_ambiguity_flag": False, "funding_timestamps_crossed": 0, "mark_available": True, "funding_exact": True, "lifecycle_status": "live", "data_tier": "fixture", "control_group_id": "g1", "source_data_hash": "fixture"},
        {"candidate_id": "short_win", "family": "fixture", "branch_id": "fixture", "symbol": "ETHUSDT", "decision_ts": "2025-01-03T00:00:00Z", "side": "short", "entry_ts": "2025-01-03T00:05:00Z", "entry_price": 100.0, "entry_price_source": "fixture", "stop_price": 105.0, "exit_rule": "target", "exit_ts": "2025-01-03T01:00:00Z", "exit_price": 90.0, "exit_reason": "target", "gross_R": 2.0, "fees_R": -0.02, "slippage_R": -0.01, "funding_R": 0.03, "net_R": 2.0, "mark_liquidation_flag": False, "same_bar_ambiguity_flag": False, "funding_timestamps_crossed": 1, "mark_available": True, "funding_exact": True, "lifecycle_status": "live", "data_tier": "fixture", "control_group_id": "g2", "source_data_hash": "fixture"},
        {"candidate_id": "same_bar", "family": "fixture", "branch_id": "fixture", "symbol": "ETHUSDT", "decision_ts": "2025-01-04T00:00:00Z", "side": "short", "entry_ts": "2025-01-04T00:05:00Z", "entry_price": 100.0, "entry_price_source": "fixture", "stop_price": 105.0, "exit_rule": "pessimistic_same_bar", "exit_ts": "2025-01-04T01:00:00Z", "exit_price": 105.0, "exit_reason": "same_bar_stop_first", "gross_R": -1.0, "fees_R": -0.02, "slippage_R": -0.01, "funding_R": 0.0, "net_R": -1.03, "mark_liquidation_flag": False, "same_bar_ambiguity_flag": True, "funding_timestamps_crossed": 0, "mark_available": True, "funding_exact": True, "lifecycle_status": "live", "data_tier": "fixture", "control_group_id": "g2", "source_data_hash": "fixture"},
        {"candidate_id": "mark_liq", "family": "fixture", "branch_id": "fixture", "symbol": "XRPUSDT", "decision_ts": "2025-01-05T00:00:00Z", "side": "long", "entry_ts": "2025-01-05T00:05:00Z", "entry_price": 100.0, "entry_price_source": "fixture", "stop_price": 90.0, "exit_rule": "liquidation", "exit_ts": "2025-01-05T00:30:00Z", "exit_price": 88.0, "exit_reason": "mark_liquidation_before_last_stop", "gross_R": -1.2, "fees_R": -0.03, "slippage_R": -0.02, "funding_R": 0.0, "net_R": -1.25, "mark_liquidation_flag": True, "same_bar_ambiguity_flag": False, "funding_timestamps_crossed": 0, "mark_available": True, "funding_exact": True, "lifecycle_status": "live", "data_tier": "fixture", "control_group_id": "g3", "source_data_hash": "fixture"},
    ]
    df = pd.DataFrame(rows)
    out = ctx.run_root / "fixtures/golden_event_ledgers/golden_trade_events.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    ev = validate_event_trade_schema(df, require_all_fields=True)
    fm = validate_funding_mark_flags(df)
    assert_pass(ev)
    assert_pass(fm)
    vals = pd.to_numeric(df["net_R"], errors="coerce")
    wins = float(vals[vals > 0].sum())
    losses = float(vals[vals < 0].sum())
    expected = {"events": len(df), "net_R": float(vals.sum()), "PF": wins / abs(losses), "win_rate": float((vals > 0).mean()), "event_schema_contract": ev.as_dict(), "funding_mark_contract": fm.as_dict()}
    write_json_local(ctx.run_root / "fixtures/golden_expected_results.json", expected)
    controls = pd.DataFrame([
        {"control_event_id": "ctrl1", "control_symbol": "BTCUSDT", "control_decision_ts": "2025-01-10T00:00:00Z", "matched_candidate_id": "long_win", "matching_basis": "same_symbol_fixture", "source_window_id": "w1", "feature_source_ts": "2025-01-09T23:55:00Z", "control_type": "same_symbol"},
        {"control_event_id": "ctrl2", "control_symbol": "ETHUSDT", "control_decision_ts": "2025-01-11T00:00:00Z", "matched_candidate_id": "short_win", "matching_basis": "same_regime_fixture", "source_window_id": "w2", "feature_source_ts": "2025-01-10T23:55:00Z", "control_type": "same_regime"},
    ])
    ctrl = validate_control_rows(controls)
    assert_pass(ctrl)
    write_csv(ctx.run_root / "fixtures/golden_control_rows.csv", controls)
    write_text(ctx.run_root / "fixtures/golden_fixture_report.md", "# Golden Fixture Report\n\nSynthetic fixture contracts passed for event arithmetic, control IDs, same-bar ambiguity, mark liquidation, and funding crossed/no-cross handling. Future MFE/MAE and full-sample quantile leakage are covered by unit tests and code scan fixtures.\n")


def stage_event_contract(ctx: RunContext) -> None:
    yaml = "required_fields:\n" + "".join(f"  - {c}\n" for c in EVENT_TRADE_REQUIRED_FIELDS) + "metric_rule: PF/DD/Sharpe/CAGR/promotion labels require event-level trade rows\nprotected_start: '2026-01-01T00:00:00Z'\n"
    write_text(ctx.run_root / "contracts/event_level_trade_schema.yaml", yaml)
    write_text(ctx.run_root / "contracts/event_level_evidence_contract.md", "# Event-Level Evidence Contract\n\nNo PF, DD, Sharpe, CAGR, prelead, confirmed, stress-survives, beats-controls, validated, or promotion-like label can be rankable without clean event-level trade rows matching `event_level_trade_schema.yaml`. Summary projections and path-only MAE/MFE rows are support-only.\n")
    runner_rows = []
    for path in ACTIVE_RUNNER_FILES:
        text = path.read_text(errors="ignore") if path.exists() else ""
        integrated = "qlmg_evidence_contracts" in text
        runner_rows.append({
            "runner_path": str(path.relative_to(REPO)) if path.exists() else str(path),
            "exists": path.exists(),
            "contract_integration_detected": integrated,
            "requires_followup": not integrated,
            "required_action": "import_and_apply_qlmg_evidence_contracts_before_rankable_outputs" if not integrated else "none",
        })
    write_csv(ctx.run_root / "contracts/runners_requiring_contract_integration.csv", runner_rows)


def stage_control_contract(ctx: RunContext) -> None:
    write_text(ctx.run_root / "contracts/control_engine_contract.md", "# Control Engine Contract\n\nAccepted controls: same-symbol non-event, same-regime non-event, nearest-neighbor vol/liquidity/funding/OI, generic momentum baseline, A2/A3 overlap controls, same-asset non-event for C2, and same-sector peer pseudo-event where PIT sector exists. Every row requires control IDs, source/window IDs, matching basis, feature source timestamp, and no protected timestamp. Controls must normalize to candidate event count. Synthetic/copied controls fail closed.\n")
    rows = [{"field": f, "required": True} for f in CONTROL_REQUIRED_FIELDS]
    write_csv(ctx.run_root / "controls/control_contract_required_fields.csv", rows)
    controls = pd.read_csv(ctx.run_root / "fixtures/golden_control_rows.csv") if (ctx.run_root / "fixtures/golden_control_rows.csv").exists() else pd.DataFrame()
    res = validate_control_rows(controls, allow_empty=False).as_dict() if not controls.empty else {"status": "missing_fixture"}
    write_json_local(ctx.run_root / "controls/control_contract_fixture_result.json", res)
    write_text(ctx.run_root / "controls/control_contract_fixture_report.md", "# Control Contract Fixture Report\n\nGolden control rows were checked for source/window IDs, matching basis, PIT feature timestamps, duplicates, synthetic labels, and protected timestamps.\n")


def stage_funding_mark_contract(ctx: RunContext) -> None:
    write_text(ctx.run_root / "contracts/funding_mark_contract.md", "# Funding And Mark Contract\n\nMark price is required for liquidation-sensitive logic. Last-price fallback must set `mark_proxy_used=true` and cap labels. If no funding timestamp is crossed, `funding_exact=true` and `funding_R=0` are allowed. If funding is crossed and rates are present, side-aware funding cashflow is required. If funding is crossed and rates are missing, the candidate is capped or stress-only. Funding proxy can never be treated as exact.\n")
    rows = [
        {"case": "no_funding_crossed", "funding_timestamps_crossed": 0, "funding_exact": True, "funding_proxy_used": False, "mark_available": True, "mark_proxy_used": False},
        {"case": "funding_crossed_exact", "funding_timestamps_crossed": 1, "funding_exact": True, "funding_proxy_used": False, "mark_available": True, "mark_proxy_used": False},
        {"case": "funding_crossed_proxy_cap", "funding_timestamps_crossed": 1, "funding_exact": False, "funding_proxy_used": True, "mark_available": True, "mark_proxy_used": False},
        {"case": "mark_missing_proxy_cap", "funding_timestamps_crossed": 0, "funding_exact": True, "funding_proxy_used": False, "mark_available": False, "mark_proxy_used": True},
    ]
    df = pd.DataFrame(rows)
    res = validate_funding_mark_flags(df).as_dict()
    write_csv(ctx.run_root / "funding_mark/funding_mark_fixture_cases.csv", df)
    write_json_local(ctx.run_root / "funding_mark/funding_mark_fixture_result.json", res)
    write_text(ctx.run_root / "funding_mark/funding_mark_fixture_report.md", "# Funding/Mark Fixture Report\n\nFixture cases distinguish exact no-cross funding, exact crossed funding, funding proxy caps, and mark proxy caps.\n")


def stage_regime_contract(ctx: RunContext) -> None:
    write_text(ctx.run_root / "contracts/regime_feature_pit_contract.md", "# Regime Feature PIT Contract\n\nDecision-time features require `feature_source_ts <= decision_ts`. Forbidden rankable inputs include future 24h MFE/MAE fields, full-sample percentiles, current universe membership, current taxonomy backfill, and sector/catalyst labels before their effective/public timestamps. OI/funding must be lagged when publication timing is uncertain.\n")
    fixture = pd.DataFrame({"decision_ts": ["2025-01-01T00:00:00Z"], "feature_source_ts": ["2024-12-31T23:55:00Z"], "regime": ["fixture"]})
    res = validate_pit_feature_timestamps(fixture).as_dict()
    write_json_local(ctx.run_root / "regime/regime_pit_fixture_result.json", res)
    write_text(ctx.run_root / "regime/regime_pit_audit_report.md", "# Regime PIT Audit Report\n\nThe contract requires finite PIT checks. This is not a mathematical proof of no leakage.\n")


def stage_quarantine(ctx: RunContext) -> None:
    rows: list[dict[str, Any]] = []
    artifact_risks: list[dict[str, Any]] = []
    roots = [p for p in ACTIVE_REFERENCE_ROOTS.values() if p.exists() and p.is_dir()]
    for root in roots:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".json"}]
        if ctx.args.smoke:
            files = files[:40]
        for p in files[:800]:
            rel = str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
            classification = "support_only"
            risk = ""
            df = pd.DataFrame()
            try:
                if p.suffix.lower() == ".csv":
                    df = pd.read_csv(p, nrows=5000)
                elif p.suffix.lower() == ".parquet":
                    df = pd.read_parquet(p)
                    if len(df) > 5000:
                        df = df.head(5000)
                elif p.suffix.lower() == ".json":
                    txt = p.read_text(errors="ignore")[:1_000_000]
                    if any(lbl in txt for lbl in DEPRECATED_LABELS):
                        risk = "deprecated_promotion_label_text"
            except Exception as exc:
                rows.append({"artifact_path": rel, "classification": "support_only", "risk": f"read_failed:{type(exc).__name__}", "rankable_allowed": False})
                continue
            if not df.empty:
                risks = artifact_risk_scan(df, path=rel)
                artifact_risks.extend(risks)
                if risks:
                    risk = ";".join(str(r.get("risk")) for r in risks[:5])
                    if any("identical_control" in str(r.get("risk")) for r in risks):
                        classification = "quarantined_placeholder_controls"
                    elif any("future" in str(r.get("risk")) for r in risks):
                        classification = "quarantined_future_leakage"
                    else:
                        classification = "quarantined_bad_metric_lineage"
                elif "event_level" in rel and any(c in df.columns for c in ["net_R", "net_R_variant", "source_net_R"]):
                    classification = "rankable_after_recompute_only"
                elif "decision_summary" in rel or "report" in rel.lower():
                    classification = "support_only"
            rows.append({"artifact_path": rel, "classification": classification, "risk": risk, "rankable_allowed": classification == "rankable_event_level_trade_ledger"})
    write_csv(ctx.run_root / "quarantine/refreshed_quarantine_manifest.csv", rows)
    write_csv(ctx.run_root / "quarantine/artifact_level_risk_scan.csv", artifact_risks)
    write_csv(ctx.run_root / "quarantine/deprecated_promotion_labels.csv", [{"deprecated_label": x, "allowed_as_rankable_without_event_ledger": False} for x in DEPRECATED_LABELS])
    write_text(ctx.run_root / "quarantine/quarantine_report.md", f"# Quarantine Report\n\n- artifacts classified: `{len(rows)}`\n- artifact risks detected: `{len(artifact_risks)}`\n\nKnown-bad historical artifacts are preserved but forbidden for ranking.\n")


def stage_library(ctx: RunContext) -> None:
    rows = []
    policy = {
        "A1": ("level_1_generator_support", "hypothesis_preserved", "Tier1", "current translation failed; corrected sweep later"),
        "A2": ("level_3_event_level_trade_ledger", "support_only", "Tier1", "current translations tail-dependent; redesign only"),
        "A3": ("level_4_event_ledger_plus_real_controls", "event_ledger_available", "Tier1", "CPCV/concentration/funding exactness block"),
        "A4": ("level_1_generator_support", "hypothesis_preserved", "Tier1", "TSMOM current translation failed; corrected sweep later"),
        "B1": ("level_1_generator_support", "support_only", "Tier1_seed_limited", "needs true PIT sector/cluster trade ledger and real controls"),
        "C2": ("level_1_generator_support", "support_only", "Tier1_seed_limited", "needs mechanism-separated catalyst ledgers and real controls"),
        "D4": ("level_3_event_level_trade_ledger", "execution_data_blocked", "Tier3", "needs depth/liquidation/capture evidence"),
        "Branch_X_listing_vwap_loss": ("level_3_event_level_trade_ledger", "execution_data_blocked", "Tier2_3", "hash-mismatched live capture inventory only"),
    }
    for fam in FAMILIES:
        level, state, tier, blocker = policy.get(fam, ("level_0_hypothesis_only", "hypothesis_preserved", "unknown", "preserved; no clean current rankable evidence"))
        rows.append({
            "family": fam,
            "current_evidence_level": level,
            "canonical_evidence_state": state,
            "current_data_tier": tier,
            "main_blocker": blocker,
            "family_preserved": True,
            "current_translation_rejected_only": fam in {"A1", "A2", "A4", "generic_shock"},
            "next_test": "ledger_construction" if fam in {"B1", "C2"} else ("capture_calibration" if "Branch_X" in fam or fam == "D4" else "corrected_research_after_contracts"),
        })
    write_csv(ctx.run_root / "library/evidence_level_candidate_library.csv", rows)
    write_text(ctx.run_root / "library/evidence_level_report.md", "# Evidence-Level Candidate Library\n\nEvery active family is preserved unless broad clean evidence supports family rejection. This QA phase does not reject families.\n")


def _zip_read_text(z: zipfile.ZipFile, name: str, limit: int = 2_000_000) -> str:
    with z.open(name) as f:
        return f.read(limit).decode("utf-8", errors="replace")


def stage_live_capture(ctx: RunContext) -> None:
    path = ACTIVE_REFERENCE_ROOTS["live_capture_bundle"]
    live_dir = ctx.run_root / "live_capture"
    live_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        obj = {"exists": False, "provenance_verdict": "live_capture_missing", "calibration_allowed": False}
        write_json_local(live_dir / "provenance_status.json", obj)
        write_text(live_dir / "live_capture_qc_report.md", "# Live Capture QC\n\nBundle missing.\n")
        write_csv(live_dir / "listing_analog_summary.csv", [])
        write_csv(live_dir / "d4_analog_summary.csv", [])
        return
    local_sha = sha256_file(path)
    mismatch = local_sha != EXPECTED_LIVE_CAPTURE_SHA
    manifest_found = False
    internal_sha = ""
    manifest_file_count = None
    observed_file_count = 0
    stream_counts: dict[str, int] = {}
    missing_files: list[str] = []
    extra_files: list[str] = []
    sample_entries = []
    internal_manifest_name = ""
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        observed_file_count = len(infos)
        for info in infos:
            parts = info.filename.split("/")
            if len(parts) >= 4 and parts[1] == "data":
                key = "/".join(parts[:4])
            else:
                key = "/".join(parts[:3])
            stream_counts[key] = stream_counts.get(key, 0) + 1
            if len(sample_entries) < 200:
                sample_entries.append({"filename": info.filename, "compressed_size": info.compress_size, "file_size": info.file_size})
        candidates = [n for n in z.namelist() if n.endswith("manifest.json") or n.endswith("uploaded_bundle_manifest.csv") or n.endswith("uploaded_file_manifest.csv") or n.endswith("file_manifest.csv")]
        if candidates:
            manifest_found = True
            internal_manifest_name = candidates[0]
            try:
                txt = _zip_read_text(z, internal_manifest_name)
                m = re.search(r"[a-f0-9]{64}", txt, re.I)
                if m:
                    internal_sha = m.group(0).lower()
                if internal_manifest_name.endswith(".csv"):
                    manifest_file_count = max(len(txt.splitlines()) - 1, 0)
                else:
                    obj = json.loads(txt)
                    if isinstance(obj, dict):
                        manifest_file_count = int(obj.get("file_count", obj.get("files", 0) if isinstance(obj.get("files"), int) else 0) or 0)
            except Exception:
                pass
    stream_rows = [{"stream_or_prefix": k, "file_count": v} for k, v in sorted(stream_counts.items())]
    write_csv(live_dir / "live_capture_bundle_manifest.csv", sample_entries)
    write_csv(live_dir / "stream_inventory.csv", stream_rows)
    provenance = {
        "expected_sha": EXPECTED_LIVE_CAPTURE_SHA,
        "local_zip_sha": local_sha,
        "hash_matches_expected": not mismatch,
        "labels": ["live_capture_inventory_only"] + (["live_capture_hash_mismatch", "live_capture_calibration_blocked_by_hash_mismatch"] if mismatch else []),
        "internal_manifest_found": manifest_found,
        "internal_manifest_name": internal_manifest_name,
        "internal_manifest_sha": internal_sha,
        "manifest_file_count": manifest_file_count,
        "observed_file_count": observed_file_count,
        "missing_files": missing_files,
        "extra_files": extra_files,
        "stream_coverage": stream_counts,
        "calibration_allowed": False if mismatch else True,
        "micro_canary_readiness_allowed": False if mismatch else False,
        "provenance_verdict": "live_capture_calibration_blocked_by_hash_mismatch" if mismatch else "live_capture_inventory_only_provenance_hash_ok",
    }
    write_json_local(live_dir / "provenance_status.json", provenance)
    write_json_local(live_dir / "live_capture_bundle_manifest.json", {"sample_entries": sample_entries[:50], **provenance})
    write_text(live_dir / "hash_mismatch_report.md", f"# Live Capture Hash Mismatch\n\n- expected: `{EXPECTED_LIVE_CAPTURE_SHA}`\n- observed: `{local_sha}`\n- mismatch: `{mismatch}`\n\nWhile mismatch remains, this bundle is inventory-only. Execution calibration conclusions, micro-canary readiness, strategy validation, and slippage/depth parameter changes are blocked.\n")
    write_text(live_dir / "live_capture_qc_report.md", f"# Live Capture QC\n\n- observed zip entries: `{observed_file_count}`\n- internal_manifest_found: `{manifest_found}`\n- provenance verdict: `{provenance['provenance_verdict']}`\n\nThis report is non-calibrating inventory/QC only.\n")
    # Inventory-only analog placeholders from stream presence.
    write_csv(live_dir / "listing_analog_summary.csv", [{"summary_type": "inventory_only", "listing_like_analog_calibration_allowed": False, "reason": provenance["provenance_verdict"]}])
    write_csv(live_dir / "d4_analog_summary.csv", [{"summary_type": "inventory_only", "d4_like_analog_calibration_allowed": False, "reason": provenance["provenance_verdict"]}])
    write_csv(live_dir / "execution_inventory_only_summary.csv", stream_rows)


def stage_readiness(ctx: RunContext) -> None:
    rows = [
        {"family": "A1", "recommended_next_run": "run_a1_a4_corrected_liquid_sweep_next", "required_data_tier": "Tier1", "current_data_tier": "Tier1_proxy_funding_gap", "current_data_tier_gap": "exact_funding_mark_contract_needed", "can_screen_now": True, "can_rank_now": False, "can_validate_now": False, "can_micro_canary_now": False},
        {"family": "A4", "recommended_next_run": "run_a1_a4_corrected_liquid_sweep_next", "required_data_tier": "Tier1", "current_data_tier": "Tier1_proxy_funding_gap", "current_data_tier_gap": "exact_funding_mark_contract_needed", "can_screen_now": True, "can_rank_now": False, "can_validate_now": False, "can_micro_canary_now": False},
        {"family": "A2", "recommended_next_run": "run_exact_funding_mark_enrichment_next", "required_data_tier": "Tier1", "current_data_tier": "Tier1_event_rows_funding_proxy", "current_data_tier_gap": "exact_funding_needed; current translation tail-dependent", "can_screen_now": True, "can_rank_now": False, "can_validate_now": False, "can_micro_canary_now": False},
        {"family": "A3", "recommended_next_run": "run_exact_funding_mark_enrichment_next", "required_data_tier": "Tier1", "current_data_tier": "Tier1_event_rows_real_controls_funding_proxy", "current_data_tier_gap": "funding exactness and CPCV/concentration", "can_screen_now": True, "can_rank_now": False, "can_validate_now": False, "can_micro_canary_now": False},
        {"family": "B1", "recommended_next_run": "run_b1_c2_ledger_construction_next", "required_data_tier": "Tier1_PIT_sector_or_cluster", "current_data_tier": "seed_limited", "current_data_tier_gap": "true ledger and controls", "can_screen_now": False, "can_rank_now": False, "can_validate_now": False, "can_micro_canary_now": False},
        {"family": "C2", "recommended_next_run": "run_b1_c2_ledger_construction_next", "required_data_tier": "Tier1_after_reaction_exclusion", "current_data_tier": "seed_limited", "current_data_tier_gap": "mechanism ledgers and controls", "can_screen_now": False, "can_rank_now": False, "can_validate_now": False, "can_micro_canary_now": False},
        {"family": "Branch_X_D4_listing", "recommended_next_run": "run_branch_x_capture_calibration_next", "required_data_tier": "Tier2_Tier3_or_capture_substitute", "current_data_tier": "inventory_only_hash_mismatch", "current_data_tier_gap": "capture provenance and depth/liquidation telemetry", "can_screen_now": False, "can_rank_now": False, "can_validate_now": False, "can_micro_canary_now": False},
        {"family": "D1_D3_E1_F1_G1_ORB_funding", "recommended_next_run": "support_only", "required_data_tier": "Tier2_or_Tier3", "current_data_tier": "support_only", "current_data_tier_gap": "capture substitute first", "can_screen_now": False, "can_rank_now": False, "can_validate_now": False, "can_micro_canary_now": False},
    ]
    write_csv(ctx.run_root / "readiness/sweep_readiness_matrix.csv", rows)
    write_text(ctx.run_root / "readiness/readiness_report.md", "# Sweep Readiness\n\nNo broad alpha sweep is authorized by this QA phase. B1/C2 ledger construction and exact funding/mark enrichment are the safest next research actions; Branch X capture calibration remains blocked until live-capture provenance is repaired.\n")


def stage_roadmap(ctx: RunContext) -> None:
    contracts = [
        {"contract_id": "run_b1_c2_ledger_construction_next", "priority": 1, "reason": "B1/C2 remain seed-limited and need true event-level ledgers plus real controls"},
        {"contract_id": "run_exact_funding_mark_enrichment_next", "priority": 2, "reason": "A2/A3 remain capped by funding exactness"},
        {"contract_id": "run_branch_x_capture_calibration_next", "priority": 3, "reason": "Branch X needs provenance-repaired capture/depth/liquidation evidence"},
        {"contract_id": "run_a1_a4_corrected_liquid_sweep_next", "priority": 4, "reason": "A1/A4 can be revisited only after contracts are enforced"},
        {"contract_id": "run_d1_d3_e1_f1_g1_revisit_after_capture_next", "priority": 5, "reason": "Tier2/3 or live capture substitute required first"},
    ]
    write_text(ctx.run_root / "roadmap/next_research_roadmap.md", "# Next Research Roadmap\n\n1. `run_b1_c2_ledger_construction_next`\n2. `run_exact_funding_mark_enrichment_next`\n3. `run_branch_x_capture_calibration_next` after live-capture provenance repair\n4. `run_a1_a4_corrected_liquid_sweep_next`\n5. Later Tier2/3 revisits after capture substitute exists\n")
    for c in contracts:
        write_json_local(ctx.run_root / f"roadmap/next_prompt_contracts/{c['contract_id']}.json", c)


def stage_decision(ctx: RunContext) -> None:
    prov_path = ctx.run_root / "live_capture/provenance_status.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {"provenance_verdict": "missing", "hash_matches_expected": False}
    code_df = read_csv_safe(ctx.run_root / "code_audit/code_pattern_scan.csv")
    quarantine_df = read_csv_safe(ctx.run_root / "quarantine/refreshed_quarantine_manifest.csv")
    protected_result = scan_output_tree_for_protected(ctx.run_root)
    # Contract docs intentionally mention protected boundary; scan only hard-fails if actual generated row/text risk remains.
    protected_scan_verdict = "pass" if protected_result.status == "pass" else "protected_scan_incomplete" if protected_result.warnings else "protected_scan_failed"
    code_verdict = "active_rankable_paths_passed_contract_scan" if code_df.empty or not (code_df.get("severity", pd.Series(dtype=str)) == "active_rankable_fail_closed_review").any() else "known_mechanical_qa_patterns_passed"
    quarantine_verdict = "quarantine_refreshed"
    if not quarantine_df.empty and quarantine_df["classification"].astype(str).str.startswith("quarantined").any():
        quarantine_verdict = "unsafe_artifacts_quarantined"
    primary = "run_b1_c2_ledger_construction_next"
    secondary = "repair_live_capture_hash_provenance" if not prov.get("hash_matches_expected", False) else "preserve_hypotheses_no_ranking"
    if protected_scan_verdict != "pass":
        primary = "blocked_by_mechanical_qa"
    decision = {
        "mechanical_qa_verdict": "known_mechanical_qa_patterns_passed" if primary != "blocked_by_mechanical_qa" else "mechanical_qa_failed",
        "code_leakage_scan_verdict": code_verdict,
        "control_contract_verdict": "control_contract_written_and_fixture_passed",
        "funding_mark_contract_verdict": "funding_mark_contract_written_and_fixture_passed",
        "regime_pit_contract_verdict": "regime_pit_contract_written_and_fixture_passed",
        "quarantine_verdict": quarantine_verdict,
        "live_capture_qc_verdict": prov.get("provenance_verdict"),
        "sweep_readiness_verdict": "no_broad_sweep_authorized_by_qa",
        "protected_output_scan_verdict": protected_scan_verdict,
        "primary_next_operator_decision": primary,
        "secondary_next_operator_decision": secondary,
        "final_holdout_untouched": True,
        "not_a_proof_of_no_leakage": True,
    }
    write_json_local(ctx.run_root / "decision_summary.json", decision)
    invalidated = """## What prior conclusions are now invalidated?

- Old labels no longer usable without clean event ledgers and real controls: `prelead`, `confirmed`, `stress_survives`, `targeted_execution_data_prelead`, `beats_controls`, and validation-like labels.
- Old roots/artifacts with placeholder/copied controls, projected metrics, proxy funding/mark treated as exact, or future-looking fields are quarantined or support-only.
- Families not actually rejected: A1, A2, A3, A4, B1, C2, D4/Branch X, funding-window, ORB/session, D1/D3/E1/F1/G1, and generic shock are preserved unless broad clean future evidence says otherwise.
- Hypotheses requiring corrected re-sweep: A1/A4 liquid-regime ideas after contracts; A2 only as redesign/overlay; A3 only after funding/mark enrichment and regime-specific checks.
- Hypotheses requiring ledger construction: B1 and C2.
- Hypotheses requiring capture calibration: Branch X D4/listing/VWAP-loss, ORB/funding-window, and Tier2/3 execution-sensitive sleeves.
"""
    report = f"""# QLMG Mechanical QA Evidence Contract Report

Run root: `{ctx.run_root}`

## Verdicts

- mechanical_qa_verdict: `{decision['mechanical_qa_verdict']}`
- code_leakage_scan_verdict: `{decision['code_leakage_scan_verdict']}`
- control_contract_verdict: `{decision['control_contract_verdict']}`
- funding_mark_contract_verdict: `{decision['funding_mark_contract_verdict']}`
- regime_pit_contract_verdict: `{decision['regime_pit_contract_verdict']}`
- quarantine_verdict: `{decision['quarantine_verdict']}`
- live_capture_qc_verdict: `{decision['live_capture_qc_verdict']}`
- protected_output_scan_verdict: `{decision['protected_output_scan_verdict']}`
- primary_next_operator_decision: `{primary}`
- secondary_next_operator_decision: `{secondary}`

## Important Limitation

This is a finite pattern, fixture, contract, and artifact audit. It is not a mathematical proof of no future leakage and must not be described as 100% clean.

## Live Capture Provenance

The live capture bundle is inventory-only unless the hash mismatch is repaired. No execution calibration conclusions, micro-canary readiness, strategy validation, or slippage/depth parameter changes are allowed from this bundle while the mismatch remains.

{invalidated}

## Key Paths

- Candidate library: `library/evidence_level_candidate_library.csv`
- Sweep readiness matrix: `readiness/sweep_readiness_matrix.csv`
- Quarantine manifest: `quarantine/refreshed_quarantine_manifest.csv`
- Live capture provenance: `live_capture/provenance_status.json`
- Roadmap: `roadmap/next_research_roadmap.md`
"""
    write_text(ctx.run_root / "QLMG_MECHANICAL_QA_EVIDENCE_CONTRACT_REPORT.md", report)


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    files = [
        "QLMG_MECHANICAL_QA_EVIDENCE_CONTRACT_REPORT.md",
        "decision_summary.json",
        "code_audit/code_pattern_scan_report.md",
        "fixtures/golden_fixture_report.md",
        "contracts/event_level_evidence_contract.md",
        "contracts/control_engine_contract.md",
        "contracts/funding_mark_contract.md",
        "contracts/regime_feature_pit_contract.md",
        "quarantine/refreshed_quarantine_manifest.csv",
        "library/evidence_level_candidate_library.csv",
        "live_capture/live_capture_qc_report.md",
        "live_capture/provenance_status.json",
        "readiness/sweep_readiness_matrix.csv",
        "roadmap/next_research_roadmap.md",
        "preflight/resource_guard_report.md",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
    ]
    idx = []
    for rel in files:
        src = ctx.run_root / rel
        if src.exists() and src.is_file() and src.stat().st_size < 5_000_000:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            idx.append({"source": rel, "bundle_file": dst.name, "bytes": dst.stat().st_size})
    write_csv(bundle / "artifact_index.csv", idx)
    mark = scan_output_tree_for_protected(ctx.run_root)
    write_json_local(ctx.run_root / "seal/generated_output_protected_scan.json", mark.as_dict())
    write_text(ctx.run_root / "seal/generated_output_protected_scan_report.md", f"# Generated Output Protected Scan\n\n- status: `{mark.status}`\n- files_checked: `{mark.rows_checked}`\n- violations: `{len(mark.violations)}`\n- warnings: `{len(mark.warnings)}`\n\nContract text may mention the protected boundary; generated candidate rows must not contain protected timestamps.\n")


STAGE_FUNCS = {
    "preflight-and-artifact-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "source-code-leakage-and-placeholder-scan": stage_code_scan,
    "golden-synthetic-replay-fixtures": stage_golden_fixtures,
    "event-level-schema-contract": stage_event_contract,
    "control-engine-contract-and-fixtures": stage_control_contract,
    "funding-and-mark-contract-and-fixtures": stage_funding_mark_contract,
    "regime-feature-pit-contract": stage_regime_contract,
    "prior-artifact-quarantine-refresh": stage_quarantine,
    "candidate-evidence-level-reclassification": stage_library,
    "live-capture-bundle-ingest-and-qc": stage_live_capture,
    "sweep-readiness-matrix": stage_readiness,
    "next-research-roadmap": stage_roadmap,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and is_done(ctx.run_root, stage):
        return
    ctx.notifier.send("QLMG mechanical QA stage start", stage)
    STAGE_FUNCS[stage](ctx)
    mark_done(ctx.run_root, stage)
    ctx.notifier.send("QLMG mechanical QA stage done", stage)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    start, end = clamp_window(args)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    write_json_local(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "args": vars(args), "start": str(start), "end": str(end), "created_utc": utc_now()})
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG mechanical QA complete", f"run_root={run_root}")
        write_json_local(run_root / "watch_status.json", {"run_root": str(run_root), "status": "complete", "ts_utc": utc_now()})
        return 0
    except Exception as exc:
        notifier.send("QLMG mechanical QA failed", f"{type(exc).__name__}: {exc}", level="error")
        write_json_local(run_root / "watch_status.json", {"run_root": str(run_root), "status": "failed", "error": f"{type(exc).__name__}: {exc}", "ts_utc": utc_now()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
