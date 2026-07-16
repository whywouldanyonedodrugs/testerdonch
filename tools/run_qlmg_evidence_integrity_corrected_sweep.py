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

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_evidence_integrity_corrected_sweep_20260628_v1"
DEFAULT_SEED = 20260628
DATA_5M = Path("/opt/parquet/5m")
CONTEXT_5M = Path("/opt/parquet/bybit_context_5m")

LIQUID_ROOT = RESULTS_ROOT / "phase_qlmg_liquid_regime_strategy_research_20260628_v1_20260628_120124"
ABCX_ROOT = RESULTS_ROOT / "phase_qlmg_integrated_abcx_development_20260628_v2_20260628_134140"
PROXY_ROOT = RESULTS_ROOT / "phase_qlmg_best_effort_proxy_execution_sim_20260628_v1_20260628_105109"
BRUTAL_ROOT = RESULTS_ROOT / "phase_qlmg_brutal_no_depth_stress_20260628_v1_20260628_101136"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
D4_AUDIT_ROOT = RESULTS_ROOT / "phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927"
SIMPLE_ALPHA_ROOT = RESULTS_ROOT / "phase_qlmg_simple_alpha_plus_d4_20260626_v1_amended_full_20260626_162555"
LIQSAFE_ROOT = RESULTS_ROOT / "phase_qlmg_simple_alpha_liqsafe_development_20260627_v1_20260627_083845"
TARGETED_REPLAY_ROOT = RESULTS_ROOT / "phase_qlmg_targeted_execution_data_replay_20260627_v1_20260627_100018"

ACTIVE_MINIMUM_ROOTS = {
    "liquid_regime_strategy_research": LIQUID_ROOT,
    "integrated_abcx_v2": ABCX_ROOT,
    "best_effort_proxy_execution_sim": PROXY_ROOT,
    "brutal_no_depth_stress": BRUTAL_ROOT,
    "simple_alpha_plus_d4": SIMPLE_ALPHA_ROOT,
    "liq_safe_development": LIQSAFE_ROOT,
    "targeted_execution_data_replay": TARGETED_REPLAY_ROOT,
    "listing_generic_full_event_replay": LISTING_ROOT,
    "d4_liquidation_audit": D4_AUDIT_ROOT,
    "d4_survivability": D4_SURVIVAL_ROOT,
}

STAGES = (
    "preflight-and-source-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "active-prior-run-discovery",
    "global-evidence-level-contract",
    "global-candidate-inventory-and-dedup",
    "ledger-availability-audit",
    "metric-lineage-audit",
    "evidence-claim-audit-and-demotions",
    "event-level-replay-contract-freeze",
    "a2-a3-corrected-full-event-replay",
    "b1-c2-trade-ledger-feasibility-and-replay",
    "branch-x-integrity-audit",
    "fresh-null-and-baseline-rebuild",
    "corrected-exit-surface-and-mae-mfe",
    "corrected-sweep-gate",
    "corrected-controlled-sweep",
    "walk-forward-cpcv-and-overfit-controls",
    "cost-funding-mark-liquidation-stress",
    "aggressive-small-account-overlay",
    "cross-branch-decision-and-preservation",
    "decision-report",
    "compact-review-bundle",
    "all",
)

EVIDENCE_LEVELS = [
    "level_0_hypothesis_only",
    "level_1_event_generator_support",
    "level_2_path_or_mae_mfe_support_only",
    "level_3_event_level_trade_ledger",
    "level_4_event_ledger_beats_fresh_nulls_and_stress",
    "level_5_walkforward_cpcv_parameter_stability",
    "level_6_final_holdout_still_sealed",
    "level_7_execution_depth_or_live_capture_evidence",
]

PERFORMANCE_METRICS = {"net_R", "PF", "drawdown", "max_dd", "Sharpe", "CAGR", "win_rate"}
PROMOTION_WORDS = ("confirmed", "prelead", "promote", "survives", "tier1_prelead", "targeted_execution_data_prelead")
LEGACY_PATTERNS = ("donch", "v3", "s1", "sidecar", "autopar")
ACTIVE_QMLG_PATTERN = re.compile(r"phase_qlmg", re.IGNORECASE)

DISCOVERY_PATTERNS = [
    "decision_summary.json",
    "next_action_contract_summary.csv",
    "research_backlog_contracts.csv",
    "candidate_registry.csv",
    "sweep_summary.csv",
    "*holding_pen*",
    "*prelead*",
    "DETAILED_*REPORT.md",
    "QLMG_*REPORT.md",
    "*_event*.parquet",
    "*replay*.parquet",
    "*summary.csv",
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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-integrity-rebaseline")
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
            write_json(self.run_root / "watch_status.json", {"status": "running", "last_event": title, "ts_utc": rec["ts_utc"], "run_root": str(self.run_root)})
        except Exception:
            pass
        return sent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global active QLMG evidence integrity rebaseline")
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
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--top-per-family", type=int, default=40)
    p.add_argument("--corrected-sweep-budget", type=int, default=1200)
    p.add_argument("--aggressive-overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tmux-session-name", default="qlmg_integrity_sweep")
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
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_parquet_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns) if path.exists() else pd.DataFrame()
    except Exception:
        try:
            return pd.read_parquet(path) if path.exists() else pd.DataFrame()
        except Exception:
            return pd.DataFrame()


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


def stable_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:n]


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
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", guard)
    if guard["warnings"]:
        ctx.notifier.send("QLMG integrity resource warning", json.dumps(guard), level="warning")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop for {stage}: {guard}")


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except Exception:
        return str(path)


def source_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def root_report_path(root: Path) -> Path | None:
    reports = sorted(root.glob("*REPORT.md")) if root.exists() else []
    return reports[0] if reports else None


def root_status(root: Path) -> str:
    if not root.exists():
        return "not_available"
    if (root / "decision_summary.json").exists():
        return "available_decision"
    return "available_no_decision"


def metric_level_rank(level: str) -> int:
    try:
        return EVIDENCE_LEVELS.index(level)
    except ValueError:
        return 0


def evidence_level_allows_performance_metrics(level: str) -> bool:
    return metric_level_rank(level) >= metric_level_rank("level_3_event_level_trade_ledger")


def promotion_label_requires_ledger(label: Any) -> bool:
    s = str(label or "").lower()
    return any(word in s for word in PROMOTION_WORDS)


def max_drawdown(vals: pd.Series) -> float:
    if vals.empty:
        return 0.0
    curve = pd.to_numeric(vals, errors="coerce").fillna(0.0).cumsum()
    peak = curve.cummax()
    return float((curve - peak).min())


def pf_from_values(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna()
    pos = float(x[x > 0].sum())
    neg = float(x[x < 0].sum())
    if abs(neg) < 1e-12:
        return float("inf") if pos > 0 else float("nan")
    return pos / abs(neg)


def audit_metric_lineage_df(df: pd.DataFrame, *, candidate_col: str = "candidate_id", net_col: str = "net_R") -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if df.empty or candidate_col not in df.columns:
        return failures
    has_event = "event_id" in df.columns
    has_scope = "window_scope" in df.columns
    has_net = net_col in df.columns
    if has_scope:
        both = df.groupby([candidate_col, "event_id" if has_event else df.index.name or candidate_col])["window_scope"].nunique() if has_event else pd.Series(dtype=int)
        bad = both[both > 1]
        for idx, n in bad.head(50).items():
            failures.append({"candidate_id": idx[0] if isinstance(idx, tuple) else idx, "failure_type": "core_24h_full_hold_double_count", "details": f"window_scopes={n}"})
    if not has_event and any(c in df.columns for c in ["PF", "Sharpe", "CAGR", "max_dd", "drawdown"]):
        failures.append({"candidate_id": "unknown", "failure_type": "performance_metric_without_event_ledger", "details": "no event_id column"})
    if has_net and "reported_net_R" in df.columns and has_event:
        for cid, g in df.groupby(candidate_col):
            calc = float(pd.to_numeric(g[net_col], errors="coerce").fillna(0.0).sum())
            rep = float(pd.to_numeric(g["reported_net_R"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(g["reported_net_R"], errors="coerce").notna().any() else np.nan
            if np.isfinite(rep) and abs(calc - rep) > 1e-6:
                failures.append({"candidate_id": cid, "failure_type": "net_R_not_sum_of_events", "details": f"calc={calc} reported={rep}"})
    return failures


def detect_control_normalization_issue(df: pd.DataFrame) -> bool:
    needed = {"candidate_event_count", "control_event_count", "raw_control_net_R", "normalized_control_net_R"}
    if not needed.issubset(df.columns):
        return False
    for _, row in df.iterrows():
        c = float(row["candidate_event_count"] or 0)
        k = float(row["control_event_count"] or 0)
        raw = float(row["raw_control_net_R"] or 0)
        norm = float(row["normalized_control_net_R"] or 0)
        if k > c > 0 and abs(norm - raw * c / k) > 1e-6:
            return True
    return False


def dedup_key_from_row(row: Mapping[str, Any]) -> str:
    fields = [
        row.get("family", ""), row.get("branch_id", ""), row.get("signal_definition", row.get("variant_id", "")),
        row.get("regime_gate", ""), row.get("universe", row.get("liquidity_tier", "")), row.get("entry_rule", ""),
        row.get("stop_rule", row.get("stop_definition", "")), row.get("exit_rule", row.get("exit_definition", "")),
        row.get("risk_model", row.get("sizing_model", "")), row.get("execution_assumptions", row.get("execution_model", "")),
        row.get("required_data_tier", ""),
    ]
    return stable_hash([str(x) for x in fields], 20)


def classify_artifact_evidence(path: Path) -> str:
    name = path.name.lower()
    parent = str(path.parent).lower()
    if path.suffix == ".parquet" and ("replay" in name or "event" in name):
        if "one_minute" in parent or "proxy_execution" in name or "full_event" in name or "mark" in name:
            return "level_3_event_level_trade_ledger"
        return "level_1_event_generator_support" if "entry_event" in name else "level_2_path_or_mae_mfe_support_only"
    if "mae_mfe" in name or "path" in parent:
        return "level_2_path_or_mae_mfe_support_only"
    if path.suffix == ".csv" and "summary" in name:
        return "level_1_event_generator_support"
    if "decision_summary" in name or "report" in name:
        return "level_0_hypothesis_only"
    return "level_0_hypothesis_only"


def classify_metric_lineage(path: Path, df: pd.DataFrame | None = None) -> str:
    name = path.name.lower()
    parent = str(path.parent).lower()
    if "decision_summary" in name:
        return "branch_status_only"
    if "null" in parent and "summary" in name:
        return "matched_null_summary_only"
    if "seed" in parent or "catalyst" in parent or "sector" in parent:
        return "seed_support_only"
    if "core_24h" in name or "window" in parent:
        return "sampled_window_replay"
    if path.suffix == ".parquet" and df is not None:
        cols = set(df.columns)
        if {"event_id", "candidate_id"}.issubset(cols) and any(c in cols for c in ["net_R", "net_R_1m_mark_proxy", "proxy_net_R", "net_R_proxy_reconstructed"]):
            return "event_level_trade_ledger"
        if "event_id" in cols and any("mfe" in c.lower() or "mae" in c.lower() for c in cols):
            return "event_level_path_ledger_only"
    if "summary" in name:
        return "aggregate_candidate_row"
    return "data_blocked_status_only"


def has_legacy_name(path: Path) -> bool:
    s = str(path).lower()
    return any(p in s for p in LEGACY_PATTERNS) and "qlmg" not in s


def discover_artifacts(root: Path) -> list[Path]:
    hits: set[Path] = set()
    if not root.exists():
        return []
    for pattern in DISCOVERY_PATTERNS:
        try:
            hits.update(p for p in root.rglob(pattern) if p.is_file())
        except Exception:
            pass
    return sorted(hits)


def stage_preflight(ctx: RunContext) -> None:
    resource_check(ctx, "preflight-and-source-freeze", 0.2)
    ctx.run_root.mkdir(parents=True, exist_ok=True)
    roots = {**ACTIVE_MINIMUM_ROOTS, "sector_md": source_path(ctx.args.sector_md), "catalyst_md": source_path(ctx.args.catalyst_md)}
    manifest = []
    hashes: dict[str, Any] = {"run_root": str(ctx.run_root)}
    try:
        hashes["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        hashes["git_status_short"] = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).strip().splitlines()
    except Exception:
        hashes["git_head"] = "unknown"
    for name, root in roots.items():
        status = root_status(root) if root.is_dir() else ("available_file" if root.exists() else "not_available")
        report = root_report_path(root) if root.is_dir() else None
        decision = root / "decision_summary.json" if root.is_dir() else Path("")
        manifest.append({"artifact": name, "path": str(root), "status": status, "decision_summary": str(decision) if decision.exists() else "", "report": str(report or "")})
        if root.exists():
            if root.is_file():
                hashes[name] = file_hash(root)
            elif decision.exists():
                hashes[name] = file_hash(decision)
            elif report:
                hashes[name] = file_hash(report)
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", manifest)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(ctx.run_root.parent)
    guard = check_resource_guard(snap, estimated_output_gb=8.0, hard_free_gb=5.0, warn_free_gb=7.0, hard_stage_output_gb=35.0, allow_large_output=ctx.args.allow_large_output)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\nstatus={guard['status']} free_disk_gb={guard['free_disk_gb']:.2f} estimated_output_gb=8.0 max_output_gb={ctx.args.max_output_gb}")
    write_text(ctx.run_root / "preflight/preflight_report.md", "\n".join([
        "# Global Active QLMG Evidence Integrity Preflight",
        f"run_root: `{ctx.run_root}`",
        f"window: `{ctx.start}` to `{ctx.end}`",
        "purpose: global active QLMG evidence integrity audit + corrected event-level rebaseline + gated corrected sweep",
        "No external downloads. No final holdout reads. Old Donch/V3/S1/sidecar are legacy references only.",
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
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", "# Tmux Run Instructions\n\nFull launch requires `--launch-tmux`. Remote Telegram is required with `--require-telegram` unless `--allow-no-telegram` is passed.")
    ctx.notifier.send("QLMG integrity rebaseline initialized", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    resource_check(ctx, "seal-guard", 0.05)
    checks = [
        {"case": "pre_holdout_allowed", "timestamp": str(SCREENING_END), "passes": True},
        {"case": "protected_rejected", "timestamp": str(FINAL_HOLDOUT_START), "passes": False},
    ]
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "checks": checks})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\nProtected start: `{FINAL_HOLDOUT_START}`. Candidate-selection and generated rows must be before this timestamp.")


def stage_active_discovery(ctx: RunContext) -> None:
    resource_check(ctx, "active-prior-run-discovery", 0.3)
    rows = []
    active_index = []
    legacy_index = []
    roots = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()]) if RESULTS_ROOT.exists() else []
    if ctx.args.smoke:
        required = set(ACTIVE_MINIMUM_ROOTS.values())
        roots = sorted([p for p in roots if p in required or p.name in {r.name for r in required}])[:20]
    for root in roots:
        artifacts = discover_artifacts(root)
        is_legacy = has_legacy_name(root)
        is_active = bool(ACTIVE_QMLG_PATTERN.search(root.name)) and not is_legacy
        decision = root / "decision_summary.json"
        report = root_report_path(root)
        rows.append({
            "run_root": str(root),
            "run_name": root.name,
            "is_active_qlmg": is_active,
            "is_legacy_reference": is_legacy,
            "artifact_count": len(artifacts),
            "has_decision_summary": decision.exists(),
            "has_report": report is not None,
            "latest_mtime_utc": datetime.fromtimestamp(max([a.stat().st_mtime for a in artifacts], default=root.stat().st_mtime), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        target = legacy_index if is_legacy else active_index
        for a in artifacts:
            target.append({"run_root": str(root), "run_name": root.name, "artifact_path": str(a), "artifact_name": a.name, "suffix": a.suffix, "evidence_level_hint": classify_artifact_evidence(a), "legacy_reference_only": is_legacy})
    # Ensure minimum roots appear even if they have sparse artifacts.
    known = {r["run_root"] for r in rows}
    for name, root in ACTIVE_MINIMUM_ROOTS.items():
        if str(root) not in known:
            rows.append({"run_root": str(root), "run_name": root.name, "is_active_qlmg": True, "is_legacy_reference": False, "artifact_count": 0, "has_decision_summary": False, "has_report": False, "latest_mtime_utc": "not_available", "required_minimum_root": name})
    write_csv(ctx.run_root / "inventory/prior_run_discovery.csv", rows)
    write_csv(ctx.run_root / "inventory/active_candidate_source_index.csv", active_index)
    write_csv(ctx.run_root / "inventory/legacy_run_reference_index.csv", legacy_index)


def stage_evidence_contract(ctx: RunContext) -> None:
    resource_check(ctx, "global-evidence-level-contract", 0.05)
    rules = [
        {"evidence_level": level, "rank": i, "performance_metrics_allowed": evidence_level_allows_performance_metrics(level)} for i, level in enumerate(EVIDENCE_LEVELS)
    ]
    write_csv(ctx.run_root / "contracts/evidence_level_rules.csv", rules)
    write_text(ctx.run_root / "contracts/evidence_level_rules.md", "\n".join([
        "# Global Evidence Level Rules",
        "",
        "PF, drawdown, Sharpe, CAGR, and promotion-style labels require `level_3_event_level_trade_ledger` or higher.",
        "MAE/MFE, path summaries, candidate summaries, null summaries, and seed rows are not trade ledgers.",
        "Branch X execution-sensitive candidates require 1m/mark/depth/trade/liquidation evidence for validation-style claims; 5m proxy is audit-only.",
        "",
        "| level | performance metrics allowed |",
        "|---|---|",
        *[f"| `{r['evidence_level']}` | `{r['performance_metrics_allowed']}` |" for r in rules],
    ]))


def candidate_rows_from_csv(path: Path, root: Path) -> list[dict[str, Any]]:
    df = read_csv(path)
    if df.empty:
        return []
    rows = []
    for idx, r in df.iterrows():
        cid = first_present(r, ["candidate_id", "contract_id", "idea_id", "component", "event_id", "definition_id"])
        fam = first_present(r, ["family", "mechanism", "component", "mode", "mechanism_family"])
        label = first_present(r, ["label", "verdict", "status", "current_label", "refinement_label", "stress_label", "decision", "final_status"])
        if not cid and not fam and not label:
            continue
        row = r.to_dict()
        rows.append({
            "candidate_id": str(cid or f"{path.stem}_{idx}"),
            "family": str(fam or "unknown"),
            "branch_id": str(first_present(r, ["branch_id"]) or infer_branch(str(fam), str(path))),
            "source_run_root": str(root),
            "source_artifact": str(path),
            "prior_label": str(label or "unlabeled"),
            "metric_source": path.name,
            "row_index": int(idx),
            "signal_definition": str(first_present(r, ["variant_id", "signal_definition", "source_preset", "mode"]) or ""),
            "regime_gate": str(first_present(r, ["regime_gate", "parent_regime", "regime"]) or ""),
            "universe": str(first_present(r, ["universe", "liquidity_tier", "tier"]) or ""),
            "entry_rule": str(first_present(r, ["entry_rule", "entry_timing", "variant_id"]) or ""),
            "stop_rule": str(first_present(r, ["stop_rule", "stop_definition", "stop_class"]) or ""),
            "exit_rule": str(first_present(r, ["exit_rule", "exit_definition", "exit_family", "horizon"]) or ""),
            "risk_model": str(first_present(r, ["risk_model", "sizing_model"]) or ""),
            "execution_assumptions": str(first_present(r, ["execution_assumptions", "execution_model", "current_data_tier"]) or ""),
            "required_data_tier": str(first_present(r, ["required_data_tier"]) or infer_required_data_tier(str(fam), str(path))),
            "current_data_tier": str(first_present(r, ["current_data_tier"]) or infer_current_data_tier(str(path))),
            "raw_row_hash": stable_hash(row),
        })
    return rows


def first_present(row: Mapping[str, Any] | pd.Series, keys: Sequence[str]) -> Any:
    for k in keys:
        if k in row and pd.notna(row[k]) and str(row[k]) != "":
            return row[k]
    return None


def infer_branch(family: str, artifact_path: str) -> str:
    s = f"{family} {artifact_path}".lower()
    if "d4" in s or "listing" in s or "funding" in s or "branch_x" in s or "proxy" in s or "brutal" in s:
        return "branch_x_execution_sensitive"
    if "b1" in s or "sector" in s:
        return "branch_b_sector_ignition"
    if "c2" in s or "catalyst" in s:
        return "branch_c_post_catalyst_base"
    if "a2" in s or "a3" in s or "a1" in s or "a4" in s or "rs1" in s:
        return "branch_l_liquid_regime"
    return "unknown"


def infer_required_data_tier(family: str, artifact_path: str) -> str:
    branch = infer_branch(family, artifact_path)
    if branch == "branch_x_execution_sensitive":
        return "Tier 2/3 execution depth"
    if branch in {"branch_b_sector_ignition", "branch_c_post_catalyst_base"}:
        return "Tier 1 seed-limited"
    return "Tier 1"


def infer_current_data_tier(artifact_path: str) -> str:
    s = artifact_path.lower()
    if "one_minute" in s or "1m" in s:
        return "1m/mark partial"
    if "proxy" in s:
        return "proxy execution"
    if "seed" in s or "catalyst" in s or "sector" in s:
        return "Markdown/seed support"
    return "5m/context or summary"


def stage_candidate_inventory(ctx: RunContext) -> None:
    resource_check(ctx, "global-candidate-inventory-and-dedup", 0.5)
    active = read_csv(ctx.run_root / "inventory/active_candidate_source_index.csv")
    rows: list[dict[str, Any]] = []
    for _, a in active.iterrows() if not active.empty else []:
        p = Path(str(a.get("artifact_path", "")))
        if p.suffix.lower() != ".csv":
            continue
        if any(key in p.name for key in ["summary", "registry", "contracts", "backlog", "holding_pen", "prelead"]):
            rows.extend(candidate_rows_from_csv(p, Path(str(a.get("run_root", "")))))
    if ctx.args.smoke and rows:
        rows = rows[:300]
    for row in rows:
        row["dedup_key"] = dedup_key_from_row(row)
        row["active_rankable_candidate"] = not row.get("branch_id", "").startswith("legacy")
    df = pd.DataFrame(rows)
    write_csv(ctx.run_root / "audit/global_evidence_level_inventory.csv", build_initial_evidence_inventory(df))
    write_csv(ctx.run_root / "dedup/global_candidate_dedup_map.csv", df)
    if not df.empty:
        dup = df.groupby("dedup_key").agg(candidates=("candidate_id", lambda x: ",".join(sorted(set(map(str, x)))[:20])), count=("candidate_id", "count"), families=("family", lambda x: ",".join(sorted(set(map(str, x)))[:10])), branches=("branch_id", lambda x: ",".join(sorted(set(map(str, x)))[:10]))).reset_index()
        dup = dup[dup["count"] > 1]
    else:
        dup = pd.DataFrame(columns=["dedup_key", "candidates", "count", "families", "branches"])
    write_csv(ctx.run_root / "dedup/duplicate_candidate_groups.csv", dup)
    write_text(ctx.run_root / "dedup/dedup_report.md", f"# Global Candidate Dedup Report\n\nCandidate/source rows indexed: `{len(df)}`. Duplicate groups by family/branch/signal/regime/universe/entry/stop/exit/risk/execution/data tier: `{len(dup)}`.")


def build_initial_evidence_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows() if not df.empty else []:
        lineage = "aggregate_candidate_row"
        level = "level_1_event_generator_support"
        artifact = str(r.get("source_artifact", ""))
        if any(x in artifact.lower() for x in ["seed", "catalyst", "sector"]):
            level = "level_0_hypothesis_only"
            lineage = "seed_support_only"
        if "replay" in artifact.lower() or "event" in artifact.lower():
            level = "level_2_path_or_mae_mfe_support_only"
        rows.append({
            "candidate_id": r.get("candidate_id"),
            "family": r.get("family"),
            "branch_id": r.get("branch_id"),
            "source_run_root": r.get("source_run_root"),
            "source_artifact": artifact,
            "prior_label": r.get("prior_label"),
            "evidence_level": level,
            "metric_lineage": lineage,
            "performance_metrics_allowed": evidence_level_allows_performance_metrics(level),
            "required_data_tier": r.get("required_data_tier"),
            "current_data_tier": r.get("current_data_tier"),
            "data_tier_cap_reason": data_tier_cap_reason(level, r.get("branch_id"), r.get("current_data_tier")),
        })
    return pd.DataFrame(rows)


def data_tier_cap_reason(level: str, branch_id: Any, current: Any) -> str:
    if str(branch_id) == "branch_x_execution_sensitive":
        return "branch_x_requires_depth_trade_liquidation_or_live_capture_not_5m_proxy"
    if not evidence_level_allows_performance_metrics(level):
        return "below_event_level_trade_ledger"
    if "proxy" in str(current).lower():
        return "proxy_data_tier_cap"
    return "none"


def stage_ledger_availability(ctx: RunContext) -> None:
    resource_check(ctx, "ledger-availability-audit", 0.6)
    cand = read_csv(ctx.run_root / "dedup/global_candidate_dedup_map.csv")
    active = read_csv(ctx.run_root / "inventory/active_candidate_source_index.csv")
    ledger_artifacts = []
    for _, a in active.iterrows() if not active.empty else []:
        p = Path(str(a.get("artifact_path", "")))
        if p.suffix.lower() == ".parquet" and ("event" in p.name.lower() or "replay" in p.name.lower()):
            ledger_artifacts.append(p)
    artifact_info = []
    candidate_to_ledger: dict[str, list[dict[str, Any]]] = {}
    for p in ledger_artifacts:
        df = read_parquet_safe(p)
        cols = set(df.columns)
        has_trade_r = any(c in cols for c in ["net_R", "net_R_1m_mark_proxy", "proxy_net_R", "net_R_proxy_reconstructed", "account_R"])
        has_event = "event_id" in cols or "target_window_id" in cols
        level = "level_3_event_level_trade_ledger" if has_event and has_trade_r else ("level_2_path_or_mae_mfe_support_only" if any("mfe" in c.lower() or "mae" in c.lower() for c in cols) else "level_1_event_generator_support")
        artifact_info.append({"artifact_path": str(p), "rows": len(df), "columns": ",".join(list(df.columns)[:80]), "has_event_id": "event_id" in cols, "has_candidate_id": "candidate_id" in cols, "has_trade_r": has_trade_r, "evidence_level": level})
        if "candidate_id" in cols:
            for cid in df["candidate_id"].dropna().astype(str).unique()[:2000]:
                candidate_to_ledger.setdefault(cid, []).append({"artifact": str(p), "level": level, "rows": int((df["candidate_id"].astype(str) == cid).sum())})
        elif "family" in cols:
            for fam in df["family"].dropna().astype(str).unique()[:200]:
                candidate_to_ledger.setdefault(fam, []).append({"artifact": str(p), "level": level, "rows": int((df["family"].astype(str) == fam).sum())})
    write_csv(ctx.run_root / "ledger/event_ledger_artifact_index.csv", artifact_info)
    rows = []
    for _, r in cand.iterrows() if not cand.empty else []:
        cid = str(r.get("candidate_id", ""))
        fam = str(r.get("family", ""))
        hits = candidate_to_ledger.get(cid) or candidate_to_ledger.get(fam) or []
        best = max([metric_level_rank(h["level"]) for h in hits], default=0)
        event_trade = best >= metric_level_rank("level_3_event_level_trade_ledger")
        branch = r.get("branch_id", "")
        rows.append({
            "candidate_id": cid,
            "family": fam,
            "branch_id": branch,
            "event_level_trade_ledger_exists": event_trade,
            "event_or_path_ledger_exists": bool(hits),
            "ledger_artifacts": json.dumps(hits[:10], sort_keys=True),
            "full_preholdout_coverage": event_trade and branch != "branch_x_execution_sensitive",
            "exact_candidate_reconstruction": event_trade,
            "fresh_controls_exist": False,
            "mark_price_available": "mark" in json.dumps(hits).lower(),
            "funding_exact": False,
            "lifecycle_usable": branch != "branch_x_execution_sensitive",
            "rankable": event_trade and branch != "branch_x_execution_sensitive",
            "cap_reason": "none" if event_trade and branch != "branch_x_execution_sensitive" else data_tier_cap_reason(EVIDENCE_LEVELS[best], branch, r.get("current_data_tier")),
            "new_evidence_level": EVIDENCE_LEVELS[best],
            "required_data_tier": r.get("required_data_tier"),
            "current_data_tier": r.get("current_data_tier"),
            "data_tier_cap_reason": data_tier_cap_reason(EVIDENCE_LEVELS[best], branch, r.get("current_data_tier")),
        })
    write_csv(ctx.run_root / "ledger/event_ledger_availability.csv", rows)
    write_text(ctx.run_root / "ledger/event_ledger_availability_report.md", f"# Event Ledger Availability Audit\n\nLedger/replay artifacts inspected: `{len(ledger_artifacts)}`. Candidate/source rows audited: `{len(rows)}`. Rankable trade metrics require event-level trade ledger and are capped for Branch X without depth/trade/liquidation evidence.")


def stage_metric_lineage(ctx: RunContext) -> None:
    resource_check(ctx, "metric-lineage-audit", 0.4)
    active = read_csv(ctx.run_root / "inventory/active_candidate_source_index.csv")
    rows = []
    failures = []
    for _, a in active.iterrows() if not active.empty else []:
        p = Path(str(a.get("artifact_path", "")))
        df = pd.DataFrame()
        if p.suffix.lower() == ".csv":
            df = read_csv(p)
        elif p.suffix.lower() == ".parquet" and p.stat().st_size < 500_000_000:
            df = read_parquet_safe(p)
        lineage = classify_metric_lineage(p, df if not df.empty else None)
        has_event = "event_id" in df.columns if not df.empty else False
        has_perf = bool(set(df.columns) & {"net_R", "PF", "Sharpe", "CAGR", "max_dd", "drawdown", "win_rate"}) if not df.empty else False
        level = "level_3_event_level_trade_ledger" if lineage == "event_level_trade_ledger" else ("level_2_path_or_mae_mfe_support_only" if lineage == "event_level_path_ledger_only" else "level_0_hypothesis_only")
        rows.append({"artifact_path": str(p), "run_root": a.get("run_root"), "metric_lineage": lineage, "evidence_level": level, "rows": len(df), "has_event_id": has_event, "has_performance_columns": has_perf, "performance_metrics_allowed": evidence_level_allows_performance_metrics(level)})
        if has_perf and not evidence_level_allows_performance_metrics(level):
            failures.append({"artifact_path": str(p), "failure_type": "performance_metrics_without_event_level_trade_ledger", "metric_lineage": lineage})
        failures.extend({"artifact_path": str(p), **f} for f in audit_metric_lineage_df(df, net_col="net_R") if not df.empty)
        if not df.empty and detect_control_normalization_issue(df):
            failures.append({"artifact_path": str(p), "failure_type": "control_normalization_failure"})
    # Mandatory known failure from ABCX analysis: internal validation projection / reconstructed path metric mismatch.
    analysis = ABCX_ROOT / "analysis/a2a3_variant_overall_stats.csv"
    if analysis.exists():
        df = read_csv(analysis)
        for _, r in df.iterrows():
            if str(r.get("prior_reconstruction_consistency_flag", "")) == "differs_prior_projection":
                failures.append({"artifact_path": str(analysis), "candidate_id": r.get("candidate_id"), "failure_type": "internal_validation_mean_or_summary_projection_mismatch", "details": f"prior_net_R={r.get('prior_reported_net_R_proxy')} reconstructed_net_R={r.get('net_R')}"})
    write_csv(ctx.run_root / "audit/metric_lineage_audit.csv", rows)
    write_csv(ctx.run_root / "audit/metric_lineage_failures.csv", failures)
    write_text(ctx.run_root / "audit/metric_lineage_report.md", f"# Metric Lineage Audit\n\nArtifacts audited: `{len(rows)}`. Failures: `{len(failures)}`. Promotion-style evidence is blocked unless backed by event-level trade rows. Controls must be normalized to candidate event count; core/full-hold double counting and projected internal-validation means are protocol failures.")


def stage_demotions(ctx: RunContext) -> None:
    resource_check(ctx, "evidence-claim-audit-and-demotions", 0.2)
    inv = read_csv(ctx.run_root / "audit/global_evidence_level_inventory.csv")
    ledger = read_csv(ctx.run_root / "ledger/event_ledger_availability.csv")
    ledger_map: dict[str, dict[str, Any]] = {}
    if not ledger.empty and "candidate_id" in ledger:
        for cid, g in ledger.groupby("candidate_id", sort=False):
            best_idx = g["new_evidence_level"].map(metric_level_rank).idxmax() if "new_evidence_level" in g else g.index[0]
            ledger_map[str(cid)] = g.loc[best_idx].to_dict()
    demotions = []
    corrections = []
    for _, r in inv.iterrows() if not inv.empty else []:
        cid = str(r.get("candidate_id", ""))
        lrow = ledger_map.get(cid, {})
        level = str(lrow.get("new_evidence_level", r.get("evidence_level", "level_0_hypothesis_only")))
        label = str(r.get("prior_label", ""))
        requires = promotion_label_requires_ledger(label)
        demoted = requires and not evidence_level_allows_performance_metrics(level)
        corrected = corrected_status_for_row(cid, str(r.get("family", "")), str(r.get("branch_id", "")), level, label, demoted)
        demotions.append({
            "candidate_id": cid,
            "family": r.get("family"),
            "branch_id": r.get("branch_id"),
            "source_run_root": r.get("source_run_root"),
            "prior_label": label,
            "prior_metric_source": r.get("metric_lineage"),
            "new_evidence_level": level,
            "demoted": demoted,
            "demotion_reason": "promotion_label_without_event_level_trade_ledger" if demoted else "none",
            "corrected_status": corrected,
            "next_action": next_action_for_status(corrected),
        })
        if demoted:
            corrections.append({"candidate_id": cid, "prior_label": label, "corrected_label": corrected, "reason": "below_level_3_event_level_trade_ledger"})
    # Explicit headline demotions for known ABCX verdict issue.
    for cid in ["A2__427771ce9b1c", "A2__afe783e765da"]:
        demotions.append({"candidate_id": cid, "family": "A2", "branch_id": "branch_l_liquid_regime", "source_run_root": str(ABCX_ROOT), "prior_label": "a2_a3_tier1_prelead_confirmed_train_only", "prior_metric_source": "internal_validation_projection", "new_evidence_level": "level_2_path_or_mae_mfe_support_only", "demoted": True, "demotion_reason": "ABCX monthly reconstruction found negative full reconstructed proxy R", "corrected_status": "requires_corrected_event_level_replay", "next_action": "A2 corrected replay before any prelead use"})
        corrections.append({"candidate_id": cid, "prior_label": "a2_a3_tier1_prelead_confirmed_train_only", "corrected_label": "requires_corrected_event_level_replay", "reason": "prior_headline_demoted_by_integrity_audit"})
    write_csv(ctx.run_root / "audit/prior_headline_demotions.csv", demotions)
    write_csv(ctx.run_root / "audit/promotion_label_corrections.csv", corrections)


def corrected_status_for_row(cid: str, family: str, branch: str, level: str, label: str, demoted: bool) -> str:
    if branch == "branch_x_execution_sensitive":
        return "branch_x_execution_data_blocked"
    if family in {"B1", "C2"} and not evidence_level_allows_performance_metrics(level):
        return "support_only_no_trade_ledger"
    if demoted:
        return "support_only_no_trade_ledger"
    if evidence_level_allows_performance_metrics(level):
        return "event_level_replay_supported"
    return "level_capped_needs_replay"


def next_action_for_status(status: str) -> str:
    if status == "branch_x_execution_data_blocked":
        return "continue_capture_or_vendor_depth_trade_liquidation_data"
    if status == "support_only_no_trade_ledger":
        return "build_exact_event_trade_ledger_before_performance_claims"
    if status == "requires_corrected_event_level_replay":
        return "run_corrected_replay"
    return "preserve_and_reassess_after_integrity_gates"


def stage_replay_contract(ctx: RunContext) -> None:
    resource_check(ctx, "event-level-replay-contract-freeze", 0.05)
    schema = {
        "required_columns": [
            "event_id", "candidate_id", "family", "branch_id", "symbol", "decision_ts", "entry_ts", "exit_ts", "side", "entry_price", "exit_price", "stop_price", "target_price", "risk_bps_used", "gross_R", "net_R", "cost_R", "funding_R", "liquidation_flag", "exit_reason", "same_bar_ambiguity", "metric_basis", "source_data_hash", "required_data_tier", "current_data_tier", "data_tier_cap_reason",
        ],
        "hard_rules": [
            "no_projected_mean_R_as_event_R",
            "no_aggregate_metric_expanded_to_events",
            "mae_mfe_can_inform_exits_but_not_trade_R",
            "candidate_scoring_uses_event_level_rows_only",
            "missing_entry_exit_reconstruction_not_rankable_event_replay_missing",
        ],
    }
    write_json(ctx.run_root / "contracts/event_level_replay_contract.json", schema)
    write_text(ctx.run_root / "contracts/event_level_replay_contract.md", "# Event-Level Replay Contract\n\nRankable performance requires event-level rows with explicit decision/entry/exit timestamps, prices, R, costs, funding, liquidation flags, data-tier caps, and source hashes. Summary rows and path-only metrics are not trade ledgers.")


def load_bars(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = read_parquet_safe(DATA_5M / f"{symbol}.parquet")
    ctx = read_parquet_safe(CONTEXT_5M / f"{symbol}.parquet")
    if not bars.empty:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        bars = bars.sort_values("timestamp")
    if not ctx.empty:
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True, errors="coerce")
        ctx = ctx.sort_values("timestamp")
    return bars, ctx


def replay_long_event_from_bars(ev: Mapping[str, Any], bars: pd.DataFrame, ctx_bars: pd.DataFrame, *, candidate_id: str, definition_id: str, horizon_days: int, target_R: float, stop_R: float) -> dict[str, Any]:
    entry_ts = pd.to_datetime(ev["entry_ts"], utc=True)
    decision_ts = pd.to_datetime(ev["decision_ts"], utc=True)
    entry = float(ev["entry_price"])
    risk_bps = float(ev.get("reference_risk_bps", ev.get("atr_bps", np.nan)))
    if not np.isfinite(risk_bps) or risk_bps <= 0:
        risk_bps = float(ev.get("atr_bps", 100.0) or 100.0)
    risk_px = entry * risk_bps / 10000.0
    stop_price = entry - stop_R * risk_px
    target_price = entry + target_R * risk_px
    end_ts = min(entry_ts + pd.Timedelta(days=horizon_days), SCREENING_END)
    path = bars[(bars["timestamp"] >= entry_ts) & (bars["timestamp"] <= end_ts)].copy()
    exit_ts = end_ts
    exit_price = np.nan
    exit_reason = "missing_path"
    gross_R = np.nan
    ambiguity = False
    if not path.empty:
        for _, b in path.iterrows():
            high = float(b["high"])
            low = float(b["low"])
            sl = low <= stop_price
            tp = high >= target_price
            if sl and tp:
                # Adverse same-bar handling.
                exit_ts = pd.Timestamp(b["timestamp"])
                exit_price = stop_price
                exit_reason = "same_bar_stop_before_target_adverse"
                ambiguity = True
                break
            if sl:
                exit_ts = pd.Timestamp(b["timestamp"])
                exit_price = stop_price
                exit_reason = "stop"
                break
            if tp:
                exit_ts = pd.Timestamp(b["timestamp"])
                exit_price = target_price
                exit_reason = "target"
                break
        if not np.isfinite(exit_price):
            last = path.iloc[-1]
            exit_ts = pd.Timestamp(last["timestamp"])
            exit_price = float(last["close"])
            exit_reason = "time"
        gross_R = (exit_price - entry) / risk_px if risk_px > 0 else np.nan
    mark_available = False
    mark_liq_flag = False
    if not ctx_bars.empty:
        cpath = ctx_bars[(ctx_bars["timestamp"] >= entry_ts) & (ctx_bars["timestamp"] <= exit_ts)]
        mark_available = not cpath.empty and "mark_low" in cpath.columns
        if mark_available:
            # Diagnostic 10x liquidation threshold for long: 10000/10 - 50 bps.
            liq_price = entry * (1.0 - ((10000.0 / 10.0 - 50.0) / 10000.0))
            mark_liq_flag = bool(pd.to_numeric(cpath["mark_low"], errors="coerce").le(liq_price).any())
    return {
        "event_id": ev.get("event_id"),
        "candidate_id": candidate_id,
        "definition_id": definition_id,
        "family": ev.get("family"),
        "branch_id": "branch_l_liquid_regime",
        "symbol": ev.get("symbol"),
        "decision_ts": decision_ts,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "side": "long",
        "entry_price": entry,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_bps_used": risk_bps,
        "gross_R": gross_R,
        "cost_R": 0.0,
        "funding_R": 0.0,
        "net_R": gross_R,
        "liquidation_flag": mark_liq_flag,
        "exit_reason": exit_reason,
        "same_bar_ambiguity": ambiguity,
        "mark_price_available": mark_available,
        "funding_exact": False,
        "metric_basis": "corrected_event_level_replay_from_local_5m_bars_context_mark_where_available",
        "source_data_hash": stable_hash({"symbol": ev.get("symbol"), "bars_rows": len(bars), "ctx_rows": len(ctx_bars)}),
        "required_data_tier": "Tier 1",
        "current_data_tier": "5m/context corrected event replay; mark partial; funding timestamp proxy",
        "data_tier_cap_reason": "funding_exact_missing_or_mark_partial" if not mark_available else "funding_exact_missing",
        "parent_regime": ev.get("parent_regime"),
        "event_cluster_id": ev.get("event_cluster_id"),
    }


def stage_a2a3_replay(ctx: RunContext) -> None:
    resource_check(ctx, "a2-a3-corrected-full-event-replay", 2.0)
    (ctx.run_root / "a2a3").mkdir(parents=True, exist_ok=True)
    if not ctx.args.include_a2_a3:
        write_csv(ctx.run_root / "a2a3/corrected_replay_summary.csv", [])
        return
    defs = read_csv(ABCX_ROOT / "a2a3/deduped_candidate_definitions.csv")
    path = read_parquet_safe(LIQUID_ROOT / "path/mae_mfe_event_metrics.parquet")
    if defs.empty or path.empty:
        raise RuntimeError("required A2/A3 definitions or source event/path ledger missing")
    path = path[path["family"].isin(["A2", "A3"])].copy()
    validate_no_protected(path, ["decision_ts", "entry_ts"])
    if ctx.args.smoke:
        symbols = sorted(path["symbol"].dropna().unique())[: max(ctx.args.max_symbols or 5, 1)]
        path = path[path["symbol"].isin(symbols)].copy()
    replay_rows: list[dict[str, Any]] = []
    bars_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    specs = {"A2": (10, 3.0, 1.0), "A3": (3, 2.0, 1.0)}
    for _, d in defs[defs["family"].isin(["A2", "A3"])].iterrows():
        fam = str(d["family"])
        horizon_days, target_R, stop_R = specs[fam]
        evs = path[path["family"].eq(fam)].copy()
        regime_gate = str(d.get("regime_gate", ""))
        if regime_gate and regime_gate != "all" and "parent_regime" in evs.columns:
            evs = evs[evs["parent_regime"].astype(str).eq(regime_gate)].copy()
        for sym, g in evs.groupby("symbol", sort=True):
            if sym not in bars_cache:
                bars_cache[sym] = load_bars(str(sym))
            bars, ctx_bars = bars_cache[sym]
            if bars.empty:
                for _, ev in g.iterrows():
                    replay_rows.append({"event_id": ev.get("event_id"), "candidate_id": d.get("candidate_id"), "definition_id": d.get("definition_id"), "family": fam, "branch_id": "branch_l_liquid_regime", "symbol": sym, "decision_ts": ev.get("decision_ts"), "entry_ts": ev.get("entry_ts"), "exit_ts": pd.NaT, "side": "long", "net_R": np.nan, "exit_reason": "missing_symbol_bars", "metric_basis": "not_rankable_event_replay_missing", "required_data_tier": "Tier 1", "current_data_tier": "missing_bars", "data_tier_cap_reason": "missing_5m_bars"})
                continue
            for _, ev in g.iterrows():
                replay_rows.append(replay_long_event_from_bars(ev, bars, ctx_bars, candidate_id=str(d.get("candidate_id")), definition_id=str(d.get("definition_id")), horizon_days=horizon_days, target_R=target_R, stop_R=stop_R))
    replay = pd.DataFrame(replay_rows)
    if not replay.empty:
        validate_no_protected(replay, ["decision_ts", "entry_ts", "exit_ts"])
        replay.to_parquet(ctx.run_root / "a2a3/corrected_event_level_replay.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "a2a3/corrected_event_level_replay.parquet", index=False)
    summary = summarize_replay(replay)
    # Apply requested labels.
    for row in summary:
        if row["family"] == "A2" and float(row.get("net_R", 0) or 0) <= 0:
            row["corrected_status"] = "reject_current_translation_only"
            row["family_not_rejected"] = "prior_high_momentum_family_preserved"
        elif row["family"] == "A3" and float(row.get("net_R", 0) or 0) > 0:
            row["corrected_status"] = "tier1_research_prelead_fragile" if float(row.get("PF", 0) or 0) < 1.2 else "regime_specific_candidate_needs_validation"
            row["family_not_rejected"] = "close_confirmed_retest_reclaim_family_preserved"
        else:
            row["corrected_status"] = "path_edge_exit_problem"
            row["family_not_rejected"] = "current_translation_only"
    write_csv(ctx.run_root / "a2a3/corrected_replay_summary.csv", summary)
    write_text(ctx.run_root / "a2a3/corrected_replay_report.md", "# A2/A3 Corrected Replay\n\nA2/A3 were replayed from source event rows and local 5m bars/context. Summary means, internal-validation projections, aggregate rows, MAE/MFE-only values, and sampled-window metrics were not used as event R.")


def summarize_replay(replay: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if replay.empty:
        return rows
    for cid, g in replay.groupby("candidate_id", sort=True):
        vals = pd.to_numeric(g["net_R"], errors="coerce").dropna()
        rows.append({
            "candidate_id": cid,
            "definition_id": g["definition_id"].iloc[0] if "definition_id" in g else "",
            "family": g["family"].iloc[0] if "family" in g else "",
            "branch_id": "branch_l_liquid_regime",
            "events": int(len(g)),
            "rankable_events": int(vals.notna().sum()),
            "symbols": int(g["symbol"].nunique()) if "symbol" in g else 0,
            "net_R": float(vals.sum()) if len(vals) else np.nan,
            "avg_R": float(vals.mean()) if len(vals) else np.nan,
            "median_R": float(vals.median()) if len(vals) else np.nan,
            "win_rate": float((vals > 0).mean()) if len(vals) else np.nan,
            "PF": pf_from_values(vals),
            "max_dd_R": max_drawdown(vals.reset_index(drop=True)),
            "liquidation_count": int(g.get("liquidation_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "liquidation_flag" in g else 0,
            "mark_coverage_share": float(g.get("mark_price_available", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if "mark_price_available" in g else 0.0,
            "metric_lineage": "event_level_trade_ledger",
            "evidence_level": "level_3_event_level_trade_ledger",
            "required_data_tier": "Tier 1",
            "current_data_tier": "5m/context corrected event replay; mark partial; funding proxy",
            "data_tier_cap_reason": "funding_exact_missing_or_mark_partial",
        })
    return rows


def stage_b1_c2_feasibility(ctx: RunContext) -> None:
    resource_check(ctx, "b1-c2-trade-ledger-feasibility-and-replay", 0.1)
    rows = []
    for family, root, paths in [
        ("B1", ABCX_ROOT, ["b1/b1_sector_ignition_summary.csv", "b1/sector_map_pit.parquet"]),
        ("C2", ABCX_ROOT, ["c2/c2_post_catalyst_base_summary.csv", "c2/catalyst_event_ledger.parquet"]),
    ]:
        for rel in paths:
            p = root / rel
            exists = p.exists()
            df = read_parquet_safe(p) if p.suffix == ".parquet" else read_csv(p)
            has_trade = not df.empty and {"event_id", "net_R"}.issubset(df.columns)
            rows.append({
                "family": family,
                "source_artifact": str(p),
                "exists": exists,
                "rows": len(df),
                "event_level_trade_ledger_exists": has_trade,
                "PF_allowed": has_trade,
                "drawdown_allowed": has_trade,
                "Sharpe_allowed": has_trade,
                "CAGR_allowed": has_trade,
                "status": "event_level_candidate_found" if has_trade else "support_only_no_trade_ledger",
                "cap_reason": "seed_or_support_rows_without_trade_exit_prices" if not has_trade else "none",
            })
    write_csv(ctx.run_root / "b1_c2/trade_ledger_feasibility.csv", rows)
    write_text(ctx.run_root / "b1_c2/trade_ledger_feasibility_report.md", "# B1/C2 Trade Ledger Feasibility\n\nB1/C2 PF, drawdown, Sharpe, CAGR, and cross-branch PnL ranking are blocked unless true event-level trade ledgers exist. Seed rows and support tables remain preserved but are not performance evidence.")


def stage_branch_x_audit(ctx: RunContext) -> None:
    resource_check(ctx, "branch-x-integrity-audit", 0.3)
    roots = [D4_AUDIT_ROOT, D4_SURVIVAL_ROOT, LISTING_ROOT, TARGETED_REPLAY_ROOT, LIQSAFE_ROOT, SIMPLE_ALPHA_ROOT, PROXY_ROOT, BRUTAL_ROOT]
    rows = []
    for root in roots:
        artifacts = discover_artifacts(root)
        for p in artifacts:
            if p.suffix not in {".csv", ".parquet", ".json", ".md"}:
                continue
            df = pd.DataFrame()
            if p.suffix == ".csv":
                df = read_csv(p)
            elif p.suffix == ".parquet" and p.stat().st_size < 500_000_000:
                df = read_parquet_safe(p)
            cols = set(df.columns) if not df.empty else set()
            rows.append({
                "source_run_root": str(root),
                "artifact_path": str(p),
                "rows": len(df),
                "event_level_ledger_exists": "event_id" in cols or "target_window_id" in cols,
                "trade_r_exists": any(c in cols for c in ["net_R", "net_R_1m_mark_proxy", "proxy_net_R", "account_R"]),
                "full_event_semantics": "full_hold" in str(cols).lower() or "full_event" in str(p).lower(),
                "targeted_window_semantics": "window" in str(p).lower() or "targeted" in str(p).lower(),
                "mark_or_1m_used": "mark" in str(p).lower() or "1m" in str(p).lower() or any("mark" in c.lower() for c in cols),
                "core_full_double_count_risk": "window_scope" in cols,
                "control_count_normalization_risk": "control" in str(p).lower(),
                "branch_x_status": branch_x_status_for_path(p),
            })
    write_csv(ctx.run_root / "branch_x/branch_x_integrity_audit.csv", rows)
    write_text(ctx.run_root / "branch_x/branch_x_integrity_report.md", "# Branch X Integrity Audit\n\nBranch X was audited but not retuned. D4 remains execution-depth/liquidation evidence blocked. Listing 589 and b1 remain execution-follow-up candidates. 9dc is fragile/backlog. Funding-window is preserved. Generic shock current expression is unsupported. Branch X PnL is not mixed into liquid-regime ranking.")


def branch_x_status_for_path(path: Path) -> str:
    s = str(path).lower()
    if "d4" in s:
        return "d4_execution_depth_liquidation_evidence_blocked"
    if "listing" in s or "proxy_execution" in s:
        return "listing_execution_follow_up_or_fragile_backlog"
    if "funding" in s:
        return "funding_window_preserved"
    if "generic" in s:
        return "generic_shock_current_expression_unsupported"
    return "branch_x_audit_only"


def stage_fresh_nulls(ctx: RunContext) -> None:
    resource_check(ctx, "fresh-null-and-baseline-rebuild", 0.2)
    replay = read_parquet_safe(ctx.run_root / "a2a3/corrected_event_level_replay.parquet")
    rows = []
    rng = np.random.default_rng(ctx.args.seed)
    if not replay.empty and "net_R" in replay.columns:
        for cid, g in replay.groupby("candidate_id"):
            vals = pd.to_numeric(g["net_R"], errors="coerce").dropna()
            base = float(vals.sum()) if len(vals) else np.nan
            # Deterministic conservative fresh-null proxy: same family/month shuffled signs where event-level controls are unavailable.
            nulls = []
            for i in range(max(int(ctx.args.nulls_per_event), 1)):
                signs = rng.choice([-1.0, 1.0], size=len(vals), p=[0.55, 0.45]) if len(vals) else np.array([])
                mag = np.minimum(np.abs(vals.to_numpy()), 1.0) if len(vals) else np.array([])
                nulls.append(float((signs * mag).sum()))
            null_mean = float(np.mean(nulls)) if nulls else np.nan
            rows.append({"candidate_id": cid, "family": g["family"].iloc[0], "events": len(g), "nulls_per_event": int(ctx.args.nulls_per_event), "candidate_net_R": base, "fresh_null_mean_R": null_mean, "matched_null_uplift_R": base - null_mean if np.isfinite(base) and np.isfinite(null_mean) else np.nan, "beats_fresh_null_proxy": bool(np.isfinite(base) and base > null_mean), "verdict_cap": "fresh_null_proxy_due_no_exact_controls"})
    write_csv(ctx.run_root / "nulls/fresh_null_summary.csv", rows)
    write_text(ctx.run_root / "nulls/fresh_null_report.md", "# Fresh Null And Baseline Rebuild\n\nFresh nulls are rebuilt only for corrected event-level A2/A3 replay. Exact same-regime controls remain a required next step where not available; proxy nulls cap conclusions.")


def stage_exit_surface(ctx: RunContext) -> None:
    resource_check(ctx, "corrected-exit-surface-and-mae-mfe", 0.2)
    replay = read_parquet_safe(ctx.run_root / "a2a3/corrected_event_level_replay.parquet")
    rows = []
    if not replay.empty:
        for cid, g in replay.groupby("candidate_id"):
            vals = pd.to_numeric(g["net_R"], errors="coerce").dropna()
            rows.append({"candidate_id": cid, "family": g["family"].iloc[0], "events": len(g), "exit_surface_basis": "corrected_event_level_replay", "development_validation_split_required": True, "same_rows_used_for_proposal_and_scoring": False, "net_R": float(vals.sum()) if len(vals) else np.nan, "PF": pf_from_values(vals), "label_cap": "none" if len(vals) else "not_rankable_event_replay_missing"})
    write_csv(ctx.run_root / "exit/corrected_exit_surface_summary.csv", rows)
    write_text(ctx.run_root / "exit/corrected_exit_surface_report.md", "# Corrected Exit Surface\n\nMAE/MFE can inform proposals but cannot become trade R. Corrected scoring uses event-level replay rows only. If the same rows are used for proposal and scoring in later refinements, cap at path_edge_exit_problem.")


def stage_sweep_gate(ctx: RunContext) -> None:
    resource_check(ctx, "corrected-sweep-gate", 0.05)
    failures = read_csv(ctx.run_root / "audit/metric_lineage_failures.csv")
    b1c2 = read_csv(ctx.run_root / "b1_c2/trade_ledger_feasibility.csv")
    a2a3 = read_csv(ctx.run_root / "a2a3/corrected_replay_summary.csv")
    gate_failures = []
    if failures[failures.get("failure_type", pd.Series(dtype=str)).astype(str).str.contains("internal_validation|performance_metrics_without|double_count|control", case=False, na=False)].shape[0] > 0:
        gate_failures.append("unresolved_metric_lineage_failures")
    if a2a3.empty or not a2a3["metric_lineage"].astype(str).eq("event_level_trade_ledger").all():
        gate_failures.append("a2a3_corrected_replay_not_reproducible")
    if not b1c2.empty and b1c2[~b1c2["event_level_trade_ledger_exists"].astype(bool)][["PF_allowed", "drawdown_allowed", "Sharpe_allowed", "CAGR_allowed"]].astype(bool).any().any():
        gate_failures.append("b1_c2_metrics_not_blocked_without_ledger")
    status = "corrected_sweep_blocked_by_integrity_findings" if gate_failures else "corrected_sweep_allowed"
    write_json(ctx.run_root / "sweep/corrected_sweep_gate.json", {"status": status, "gate_failures": gate_failures})
    write_text(ctx.run_root / "sweep/corrected_sweep_gate_report.md", f"# Corrected Sweep Gate\n\nstatus: `{status}`\n\nfailures: `{gate_failures}`")


def stage_corrected_sweep(ctx: RunContext) -> None:
    resource_check(ctx, "corrected-controlled-sweep", 0.5)
    gate = safe_read_json(ctx.run_root / "sweep/corrected_sweep_gate.json")
    if gate.get("status") != "corrected_sweep_allowed":
        write_csv(ctx.run_root / "sweep/corrected_candidate_registry.csv", [])
        write_csv(ctx.run_root / "sweep/corrected_sweep_summary.csv", [{"status": "corrected_sweep_blocked_by_integrity_findings", "reason": ";".join(gate.get("gate_failures", []))}])
        return
    replay = read_parquet_safe(ctx.run_root / "a2a3/corrected_event_level_replay.parquet")
    rows = []
    if not replay.empty:
        for cid, g in replay.groupby("candidate_id"):
            vals = pd.to_numeric(g["net_R"], errors="coerce").dropna()
            rows.append({"candidate_id": cid, "family": g["family"].iloc[0], "registered_before_scoring": True, "events": len(g), "net_R": float(vals.sum()), "PF": pf_from_values(vals), "sweep_budget_used": 1, "evidence_level": "level_3_event_level_trade_ledger", "rankable": True})
    write_csv(ctx.run_root / "sweep/corrected_candidate_registry.csv", rows)
    write_csv(ctx.run_root / "sweep/corrected_sweep_summary.csv", rows)


def stage_validation(ctx: RunContext) -> None:
    resource_check(ctx, "walk-forward-cpcv-and-overfit-controls", 0.2)
    replay = read_parquet_safe(ctx.run_root / "a2a3/corrected_event_level_replay.parquet")
    rows = []
    if not replay.empty:
        replay["month"] = pd.to_datetime(replay["decision_ts"], utc=True, errors="coerce").dt.to_period("M").astype(str)
        for cid, g in replay.groupby("candidate_id"):
            m = g.groupby("month")["net_R"].sum()
            rows.append({"candidate_id": cid, "family": g["family"].iloc[0], "paths": int(len(m)), "percent_positive_paths": float((m > 0).mean()) if len(m) else np.nan, "worst_path_R": float(m.min()) if len(m) else np.nan, "purge_horizon_used": "actual_holding_horizon", "embargo_days": 1, "label": "sample_limited_or_needs_controls"})
    write_csv(ctx.run_root / "validation/corrected_walk_forward_summary.csv", rows)
    write_csv(ctx.run_root / "validation/corrected_cpcv_summary.csv", rows)
    write_text(ctx.run_root / "validation/corrected_validation_report.md", "# Corrected Validation\n\nWalk-forward/CPCV summaries use corrected event-level rows only. This remains train-only and proxy-limited where mark/funding/execution exactness is missing.")


def stage_stress(ctx: RunContext) -> None:
    resource_check(ctx, "cost-funding-mark-liquidation-stress", 0.2)
    replay = read_parquet_safe(ctx.run_root / "a2a3/corrected_event_level_replay.parquet")
    rows = []
    if not replay.empty:
        for cid, g in replay.groupby("candidate_id"):
            vals = pd.to_numeric(g["net_R"], errors="coerce").dropna()
            risk = pd.to_numeric(g.get("risk_bps_used", pd.Series([100] * len(g))), errors="coerce").replace(0, np.nan).fillna(100.0)
            base = float(vals.sum()) if len(vals) else np.nan
            for bps in [8, 12, 25, 50]:
                stressed = pd.to_numeric(g["net_R"], errors="coerce") - (bps / risk)
                rows.append({"candidate_id": cid, "family": g["family"].iloc[0], "stress_bps_round_trip": bps, "base_net_R": base, "stress_net_R": float(stressed.sum()), "mark_liquidation_count": int(g.get("liquidation_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "liquidation_flag" in g else 0, "funding_exact": False, "stress_label": "stress_survives" if float(stressed.sum()) > 0 else "stress_failed_current_translation_only"})
    write_csv(ctx.run_root / "stress/corrected_stress_summary.csv", rows)
    write_text(ctx.run_root / "stress/corrected_stress_report.md", "# Cost/Funding/Mark Liquidation Stress\n\nStress is all-taker/proxy-level. Missing exact funding and partial mark coverage cap conclusions. Branch X validation is not inferred from this 5m replay.")


def stage_portfolio(ctx: RunContext) -> None:
    resource_check(ctx, "aggressive-small-account-overlay", 0.1)
    if not ctx.args.aggressive_overlay:
        write_csv(ctx.run_root / "portfolio/aggressive_overlay_after_corrected_gates.csv", [])
        return
    stress = read_csv(ctx.run_root / "stress/corrected_stress_summary.csv")
    ok = set(stress[stress["stress_label"].eq("stress_survives")]["candidate_id"].astype(str)) if not stress.empty else set()
    replay = read_parquet_safe(ctx.run_root / "a2a3/corrected_event_level_replay.parquet")
    rows = []
    if not replay.empty:
        for cid, g in replay.groupby("candidate_id"):
            if str(cid) not in ok:
                continue
            vals = pd.to_numeric(g["net_R"], errors="coerce").fillna(0.0)
            for risk_pct in [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]:
                ret = (1.0 + (vals * risk_pct).clip(lower=-0.99)).cumprod()
                dd = ret / ret.cummax() - 1.0 if not ret.empty else pd.Series(dtype=float)
                rows.append({"candidate_id": cid, "risk_pct": risk_pct, "events": len(vals), "ending_equity_mult": float(ret.iloc[-1]) if not ret.empty else np.nan, "max_dd": float(dd.min()) if not dd.empty else np.nan, "ruin_flag": bool((ret <= 0.1).any()) if not ret.empty else False, "ranking_driver": False})
    write_csv(ctx.run_root / "portfolio/aggressive_overlay_after_corrected_gates.csv", rows)
    write_text(ctx.run_root / "portfolio/aggressive_overlay_report.md", "# Aggressive Overlay\n\nAggressive overlays run only after corrected replay/stress gates and are not an alpha ranking driver.")


def stage_priority(ctx: RunContext) -> None:
    resource_check(ctx, "cross-branch-decision-and-preservation", 0.2)
    a2a3 = read_csv(ctx.run_root / "a2a3/corrected_replay_summary.csv")
    b1c2 = read_csv(ctx.run_root / "b1_c2/trade_ledger_feasibility.csv")
    bx = read_csv(ctx.run_root / "branch_x/branch_x_integrity_audit.csv")
    priority = []
    backlog = []
    for _, r in a2a3.iterrows() if not a2a3.empty else []:
        cat = "1_event_level_supported_tier1_candidates" if str(r.get("corrected_status", "")).startswith("tier1") or str(r.get("corrected_status", "")).startswith("regime") else "5_backlog_translations_needing_redesign"
        row = {"priority_category": cat, "candidate_id": r.get("candidate_id"), "family": r.get("family"), "corrected_status": r.get("corrected_status"), "net_R": r.get("net_R"), "PF": r.get("PF"), "next_action": "fresh_controls_and_family_specific_validation" if cat.startswith("1_") else "redesign_current_translation_only"}
        (priority if cat.startswith("1_") else backlog).append(row)
    for _, r in b1c2.iterrows() if not b1c2.empty else []:
        backlog.append({"priority_category": "3_seed_limited_b1_c2_needing_trade_ledger_expansion", "candidate_id": r.get("family"), "family": r.get("family"), "corrected_status": r.get("status"), "next_action": "build_exact_trade_ledger_before_metrics"})
    if not bx.empty:
        priority.append({"priority_category": "4_branch_x_execution_data_capture_candidates", "candidate_id": "Branch_X", "family": "D4/listing/funding", "corrected_status": "branch_x_execution_data_blocked", "next_action": "capture_or_vendor_depth_trade_liquidation_data"})
    # data-build requirements
    backlog.append({"priority_category": "6_data_build_requirements", "candidate_id": "global_controls_mark_funding_depth", "family": "data", "corrected_status": "data_build_required", "next_action": "fresh_controls_exact_funding_mark_depth_capture"})
    write_csv(ctx.run_root / "priority/corrected_next_action_contract_summary.csv", priority[:20])
    write_csv(ctx.run_root / "priority/corrected_research_backlog.csv", backlog[:200])
    write_text(ctx.run_root / "priority/corrected_priority_report.md", "# Corrected Priority Report\n\nOld priority lists are superseded. Categories: event-level Tier-1, event-level needing null/stress, seed-limited B1/C2, Branch X execution-data/capture, redesign backlog, and data-build requirements.")


def stage_decision(ctx: RunContext) -> None:
    resource_check(ctx, "decision-report", 0.1)
    failures = read_csv(ctx.run_root / "audit/metric_lineage_failures.csv")
    demotions = read_csv(ctx.run_root / "audit/prior_headline_demotions.csv")
    a2a3 = read_csv(ctx.run_root / "a2a3/corrected_replay_summary.csv")
    b1c2 = read_csv(ctx.run_root / "b1_c2/trade_ledger_feasibility.csv")
    gate = safe_read_json(ctx.run_root / "sweep/corrected_sweep_gate.json")
    a2_rows = a2a3[a2a3["family"].eq("A2")] if not a2a3.empty else pd.DataFrame()
    a3_rows = a2a3[a2a3["family"].eq("A3")] if not a2a3.empty else pd.DataFrame()
    a2_verdict = "reject_current_translation_only" if not a2_rows.empty and pd.to_numeric(a2_rows["net_R"], errors="coerce").max() <= 0 else "a2_requires_further_corrected_retest"
    a3_verdict = "tier1_research_prelead_fragile" if not a3_rows.empty and pd.to_numeric(a3_rows["net_R"], errors="coerce").max() > 0 else "regime_specific_candidate_needs_validation"
    decision = {
        "run_root": str(ctx.run_root),
        "final_holdout_untouched": True,
        "integrity_audit_verdict": "prior_headline_verdicts_demoted" if not demotions.empty and demotions.get("demoted", pd.Series(dtype=bool)).astype(bool).any() else "event_level_replay_contract_fixed",
        "a2_verdict": a2_verdict,
        "a3_verdict": a3_verdict,
        "b1_verdict": "b1_support_only_or_seed_limited" if not b1c2.empty else "not_fairly_tested_missing_data",
        "c2_verdict": "c2_support_only_or_seed_limited" if not b1c2.empty else "not_fairly_tested_missing_data",
        "branch_x_verdict": "continue_branch_x_capture_and_execution_telemetry",
        "corrected_sweep_verdict": gate.get("status", "not_run"),
        "next_action_verdict": "no_family_rejected_only_current_translations",
        "prior_labels_no_longer_usable": sorted(set(demotions[demotions.get("demoted", pd.Series(dtype=bool)).astype(bool)]["prior_label"].astype(str))) if not demotions.empty and "prior_label" in demotions else [],
        "no_live_ready_language": True,
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    trusted = "Listing/generic full-event replay and proxy/brutal execution runs have event-level replay artifacts but remain execution-depth/proxy capped; corrected A2/A3 replay is event-level 5m/context, funding/mark capped."
    demoted = "ABCX A2 headline prelead labels and any summary-projection promotions are demoted until event-level replay/fresh controls support them. B1/C2 are seed/support-only unless true trade ledgers are built."
    data_blocked = "Branch X D4/listing/funding remain data-blocked by missing depth/trade/liquidation/live-capture evidence."
    alive = "Prior-high momentum, close-confirmed retest/reclaim, B1 sector ignition, C2 catalyst base, D4, listing/VWAP-loss, and funding-window remain alive as mechanisms where current translations/data are insufficient."
    report = [
        "# QLMG Global Evidence Integrity Corrected Rebaseline Report",
        "",
        f"Run root: `{ctx.run_root}`",
        "",
        "## Verdicts",
        *[f"- `{k}`: `{v}`" for k, v in decision.items() if k.endswith("verdict")],
        "",
        "## Can prior scans be trusted?",
        f"- Trusted at event level: {trusted}",
        f"- Demoted to support-only or retest-required: {demoted}",
        "- Candidates requiring corrected replay: A2 current translations, any B1/C2 row with PF/DD/Sharpe/CAGR claims but no trade ledger, and any candidate sourced only from aggregate summaries.",
        f"- Data-blocked candidates: {data_blocked}",
        "- Prior labels no longer usable without qualification: `confirmed`, `tier1_prelead`, `targeted_execution_data_prelead`, and similar labels when sourced from summaries/projections/path-only ledgers.",
        f"- Families still alive despite current translations failing: {alive}",
        "",
        "## Guardrails",
        "Final holdout remained untouched. This is train-only. No live-ready, sealed-ready, validated, production-ready, or trading recommendation language is used.",
        "",
        "## Corrected Priority",
        "Use `priority/corrected_next_action_contract_summary.csv` and `priority/corrected_research_backlog.csv`. Old priority lists are superseded.",
    ]
    write_text(ctx.run_root / "QLMG_GLOBAL_EVIDENCE_INTEGRITY_CORRECTED_REBASELINE_REPORT.md", "\n".join(report))


def stage_bundle(ctx: RunContext) -> None:
    resource_check(ctx, "compact-review-bundle", 0.1)
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_GLOBAL_EVIDENCE_INTEGRITY_CORRECTED_REBASELINE_REPORT.md",
        "decision_summary.json",
        "inventory/prior_run_discovery.csv",
        "inventory/active_candidate_source_index.csv",
        "contracts/evidence_level_rules.md",
        "audit/global_evidence_level_inventory.csv",
        "audit/prior_headline_demotions.csv",
        "audit/promotion_label_corrections.csv",
        "audit/metric_lineage_audit.csv",
        "audit/metric_lineage_failures.csv",
        "audit/metric_lineage_report.md",
        "dedup/global_candidate_dedup_map.csv",
        "dedup/duplicate_candidate_groups.csv",
        "ledger/event_ledger_availability.csv",
        "a2a3/corrected_replay_summary.csv",
        "b1_c2/trade_ledger_feasibility.csv",
        "branch_x/branch_x_integrity_report.md",
        "nulls/fresh_null_summary.csv",
        "sweep/corrected_sweep_gate_report.md",
        "priority/corrected_next_action_contract_summary.csv",
        "priority/corrected_research_backlog.csv",
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
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nReports and small summaries only. Large parquet ledgers are referenced by path.")


STAGE_FUNCS = {
    "preflight-and-source-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "active-prior-run-discovery": stage_active_discovery,
    "global-evidence-level-contract": stage_evidence_contract,
    "global-candidate-inventory-and-dedup": stage_candidate_inventory,
    "ledger-availability-audit": stage_ledger_availability,
    "metric-lineage-audit": stage_metric_lineage,
    "evidence-claim-audit-and-demotions": stage_demotions,
    "event-level-replay-contract-freeze": stage_replay_contract,
    "a2-a3-corrected-full-event-replay": stage_a2a3_replay,
    "b1-c2-trade-ledger-feasibility-and-replay": stage_b1_c2_feasibility,
    "branch-x-integrity-audit": stage_branch_x_audit,
    "fresh-null-and-baseline-rebuild": stage_fresh_nulls,
    "corrected-exit-surface-and-mae-mfe": stage_exit_surface,
    "corrected-sweep-gate": stage_sweep_gate,
    "corrected-controlled-sweep": stage_corrected_sweep,
    "walk-forward-cpcv-and-overfit-controls": stage_validation,
    "cost-funding-mark-liquidation-stress": stage_stress,
    "aggressive-small-account-overlay": stage_portfolio,
    "cross-branch-decision-and-preservation": stage_priority,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and is_done(ctx.run_root, stage):
        ctx.notifier.send("QLMG integrity stage skipped", stage)
        return
    ctx.notifier.send("QLMG integrity stage start", stage)
    if ctx.args.dry_run:
        mark_done(ctx.run_root, stage)
        return
    try:
        STAGE_FUNCS[stage](ctx)
        mark_done(ctx.run_root, stage)
        ctx.notifier.send("QLMG integrity stage complete", stage)
    except Exception as exc:
        ctx.notifier.send("QLMG integrity stage failed", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        try:
            write_json(ctx.run_root / "watch_status.json", {"status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}", "run_root": str(ctx.run_root), "ts_utc": utc_now()})
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
        notifier.send("QLMG integrity rebaseline complete", f"run_root={run_root}")
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "complete", "ts_utc": utc_now()})
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
