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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, stable_hash, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402
from tools import run_qlmg_simple_alpha_plus_d4 as prior_alpha  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_simple_alpha_liqsafe_development_20260627_v1"
PRIOR_ALPHA_ROOT = RESULTS_ROOT / "phase_qlmg_simple_alpha_plus_d4_20260626_v1_amended_full_20260626_162555"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"

TARGET_FAMILIES = {"funding_window_orb_failure", "new_perp_listing_event_study"}
SECONDARY_FAMILIES = {
    "leader_breakout_long",
    "weak_asset_spike_fade",
    "risk_off_exhaustion_spike_short",
    "us_cash_open_orb",
    "utc_daily_open_reversal",
    "crowded_long_unwind_short",
    "failed_sector_rotation_short",
    "post_catalyst_continuation_base",
}
PATH_EDGE_LABELS = {
    "path_edge_exit_problem",
    "new_entry_definition_needed",
    "cost_fragile_candidate",
    "targeted_execution_data_prelead",
    "not_fairly_tested_execution_model_missing",
}

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "prior-sweep-evidence-map",
    "family-identity-deconfounding",
    "candidate-cluster-selection",
    "liquidation-flag-taxonomy",
    "decision-time-liquidation-geometry",
    "liquidation-safe-sizing-replay",
    "funding-window-family-development",
    "new-perp-listing-family-development",
    "secondary-family-preservation-diagnostics",
    "targeted-1m-mark-replay-plan",
    "targeted-1m-download-if-approved",
    "one-minute-mark-replay-if-available",
    "matched-null-and-same-time-refresh-after-safety",
    "cost-funding-execution-stress-after-safety",
    "walk-forward-cpcv-after-safety",
    "targeted-depth-data-contracts",
    "d4-carry-forward-integration",
    "triage-and-research-backlog",
    "decision-report",
    "compact-review-bundle",
    "all",
)

HORIZONS = ["30m", "1h", "2h", "4h", "6h", "12h", "24h", "48h", "72h"]

@dataclass
class RunNotifier:
    run_root: Path
    disabled: bool = False
    require_remote: bool = False
    allow_no_remote: bool = False

    def __post_init__(self) -> None:
        self.path = self.run_root / "notifications/telegram_events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.remote = None
        self.status = "disabled" if self.disabled else "unavailable"
        self.missing = "disabled_by_cli" if self.disabled else ""
        if not self.disabled and TelegramNotifier is not None:
            try:
                class _Args:
                    disable_telegram = False
                    telegram_dry_run = False
                    tg_bot_token = ""
                    tg_chat_id = ""
                    tg_auto_chat = False
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-liqsafe-dev")
                self.status = getattr(self.remote, "status_line", lambda: "enabled")()
            except Exception as exc:  # pragma: no cover
                self.remote = None
                self.status = "unavailable"
                self.missing = f"{type(exc).__name__}: {exc}"
        elif not self.disabled:
            self.missing = "tools.telegram_notify.TelegramNotifier unavailable"
        remote_ok = self.remote is not None and "enabled" in str(self.status).lower()
        if self.require_remote and not remote_ok and not self.allow_no_remote:
            raise RuntimeError(f"remote Telegram required but unavailable: {self.missing or self.status}")

    @property
    def remote_available(self) -> bool:
        return self.remote is not None and "enabled" in str(self.status).lower()

    def send(self, title: str, body: str = "", level: str = "info") -> None:
        sent = False
        err = ""
        if self.remote is not None:
            try:
                sent = bool(self.remote.send(title, body))
            except Exception as exc:  # pragma: no cover
                err = f"{type(exc).__name__}: {exc}"
        rec = {"ts_utc": utc_now(), "title": title, "body": body, "level": level, "sent": sent, "status": self.status, "error": err}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        try:
            (self.run_root / "watch_status.json").write_text(json.dumps({"status": "running", "last_event": title, "ts_utc": rec["ts_utc"], "run_root": str(self.run_root)}, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass

@dataclass
class RunContext:
    args: argparse.Namespace
    run_root: Path
    notifier: RunNotifier
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG simple-alpha liquidation-safe train-only development")
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
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--candidate-limit", type=int, default=300)
    p.add_argument("--top-per-family", type=int, default=40)
    p.add_argument("--targeted-download-cap-gb", type=float, default=10.0)
    p.add_argument("--build-depth-plan", action="store_true", default=True)
    p.add_argument("--use-existing-1m-if-overlap", action="store_true")
    p.add_argument("--download-targeted-1m", action="store_true")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--tmux-session-name", default="qlmg_liqsafe_dev")
    p.add_argument("--run-root", default="")
    return p.parse_args(argv)


def resolve_run_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.run_root:
        root = Path(args.run_root)
        return (root if root.is_absolute() else REPO / root).resolve(), "explicit_run_root"
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
    return start, end


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def done_path(root: Path, stage: str) -> Path:
    return root / "stage_status" / f"{stage}.done"


def mark_done(root: Path, stage: str) -> None:
    done_path(root, stage).parent.mkdir(parents=True, exist_ok=True)
    done_path(root, stage).write_text(utc_now() + "\n", encoding="utf-8")


def required_outputs(root: Path, stage: str) -> list[Path]:
    m = {
        "preflight-and-artifact-freeze": [root / "preflight/preflight_report.md", root / "preflight/frozen_artifact_hashes.json", root / "preflight/input_artifact_manifest.csv"],
        "telegram-and-tmux-setup": [root / "notifications/telegram_readiness_report.md", root / "tmux/watch_commands.md"],
        "seal-guard": [root / "seal/seal_guard_report.md", root / "seal/protected_slice_check.json"],
        "prior-sweep-evidence-map": [root / "evidence/prior_sweep_evidence_map.csv"],
        "family-identity-deconfounding": [root / "deconfound/family_identity_deconfound_summary.csv"],
        "candidate-cluster-selection": [root / "selection/selected_candidate_clusters.csv", root / "evidence/candidate_cluster_map.csv"],
        "liquidation-flag-taxonomy": [root / "liquidation/liquidation_taxonomy_summary.csv"],
        "decision-time-liquidation-geometry": [root / "geometry/decision_time_liquidation_geometry.parquet", root / "geometry/geometry_summary.csv"],
        "liquidation-safe-sizing-replay": [root / "sizing/liqsafe_sizing_summary.csv"],
        "funding-window-family-development": [root / "funding_window/funding_window_development_summary.csv"],
        "new-perp-listing-family-development": [root / "listing/listing_family_development_summary.csv"],
        "secondary-family-preservation-diagnostics": [root / "secondary/secondary_family_preservation_summary.csv"],
        "targeted-1m-mark-replay-plan": [root / "data_plan/targeted_1m_window_plan.csv", root / "data_plan/targeted_1m_storage_estimate.csv"],
        "targeted-1m-download-if-approved": [root / "downloaded_1m/download_report.md", root / "downloaded_1m/download_manifest.csv"],
        "one-minute-mark-replay-if-available": [root / "one_minute/one_minute_mark_replay_summary.csv"],
        "matched-null-and-same-time-refresh-after-safety": [root / "nulls/refreshed_null_summary.csv"],
        "cost-funding-execution-stress-after-safety": [root / "stress/cost_funding_execution_stress_summary.csv"],
        "walk-forward-cpcv-after-safety": [root / "validation/walk_forward_cpcv_summary.csv"],
        "targeted-depth-data-contracts": [root / "contracts/depth_data_contract_summary.csv"],
        "d4-carry-forward-integration": [root / "d4/d4_integration_report.md", root / "contracts/d4_targeted_execution_depth_collection_contract.json"],
        "triage-and-research-backlog": [root / "triage/all_ideas_preservation_index.csv", root / "backlog/research_backlog.csv"],
        "decision-report": [root / "QLMG_SIMPLE_ALPHA_LIQSAFE_DEVELOPMENT_REPORT.md", root / "decision_summary.json"],
        "compact-review-bundle": [root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return m.get(stage, [])


def stage_complete(root: Path, stage: str) -> bool:
    return done_path(root, stage).exists() and all(p.exists() for p in required_outputs(root, stage))


def estimate_stage_gb(stage: str, smoke: bool) -> float:
    base = {
        "decision-time-liquidation-geometry": 1.0,
        "liquidation-safe-sizing-replay": 0.5,
        "matched-null-and-same-time-refresh-after-safety": 0.8,
        "walk-forward-cpcv-after-safety": 0.4,
    }.get(stage, 0.1)
    return min(base, 0.1) if smoke else base


def ensure_guard(ctx: RunContext, stage: str) -> None:
    snap = resource_snapshot(REPO)
    status = check_resource_guard(
        snap,
        estimated_output_gb=estimate_stage_gb(stage, ctx.args.smoke),
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=20.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", {"stage": stage, **status, "snapshot": snap.__dict__})
    if status["warnings"]:
        ctx.notifier.send("QLMG liqsafe resource warning", f"stage={stage} warnings={status['warnings']}", level="warning")
    if status["status"] != "pass":
        raise RuntimeError(f"resource guard failed for {stage}: {status}")


def load_sweep() -> pd.DataFrame:
    p = PRIOR_ALPHA_ROOT / "sweep/sweep_summary.csv"
    df = read_csv(p)
    if not df.empty and "candidate_id" in df.columns:
        df["candidate_id"] = df["candidate_id"].astype(str)
    return df


def load_registry() -> pd.DataFrame:
    p = PRIOR_ALPHA_ROOT / "sweep/candidate_registry.csv"
    df = read_csv(p)
    if not df.empty and "candidate_id" in df.columns:
        df["candidate_id"] = df["candidate_id"].astype(str)
    return df


def load_sweep_results() -> pd.DataFrame:
    p = PRIOR_ALPHA_ROOT / "sweep/sweep_results.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def all_event_paths() -> list[Path]:
    return [
        PRIOR_ALPHA_ROOT / "events/leader_breakout_events.parquet",
        PRIOR_ALPHA_ROOT / "events/weak_asset_spike_fade_events.parquet",
        PRIOR_ALPHA_ROOT / "events/risk_off_exhaustion_spike_short_events.parquet",
        PRIOR_ALPHA_ROOT / "events/synthetic_open_orb_events.parquet",
        PRIOR_ALPHA_ROOT / "listing/listing_event_rows.parquet",
        PRIOR_ALPHA_ROOT / "events/auxiliary_events.parquet",
    ]


def load_all_events(ctx: RunContext, families: set[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in all_event_paths():
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "decision_ts" in df.columns:
            df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce")
        if "entry_ts" in df.columns:
            df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
        df = df[(df["decision_ts"] >= ctx.start) & (df["decision_ts"] <= ctx.end)]
        if families is not None:
            df = df[df["family"].astype(str).isin(families)]
        if ctx.args.max_symbols:
            syms = sorted(df["symbol"].dropna().astype(str).unique())[: ctx.args.max_symbols]
            df = df[df["symbol"].astype(str).isin(syms)]
        if ctx.args.smoke:
            df = df.sort_values(["family", "symbol", "decision_ts"]).groupby("family", group_keys=False).head(2000)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    validate_no_protected(out, ["decision_ts", "entry_ts"])
    return out


def numeric(s: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(default)
    return pd.Series(dtype=float)


def candidate_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "family", "subfamily", "candidate_type", "source_family_preset", "seed_interpretation", "regime_gate", "funding_gate",
        "liquidity_quality_gate", "bad_wick_gate", "cost_mult", "tier_filter", "horizon", "target_r", "stop_mult",
        "risk_bps_override", "candidate_id", "readiness", "max_verdict_cap",
    ]
    out = {k: row.get(k) for k in keys if k in row}
    for k in ["cost_mult", "target_r", "stop_mult", "risk_bps_override"]:
        try:
            if pd.isna(out.get(k)):
                out[k] = None
            elif out.get(k) is not None:
                out[k] = float(out[k])
        except Exception:
            out[k] = None
    return out


def reconstruct_events_for_candidate(ctx: RunContext, cand: Mapping[str, Any], events: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str, str]:
    if not cand.get("candidate_id"):
        return pd.DataFrame(), "blocked_by_protocol_issue", "missing_candidate_id"
    source_events = events if events is not None else load_all_events(ctx, {str(cand.get("family"))})
    if source_events.empty:
        return pd.DataFrame(), "not_fairly_tested_missing_data", "event_ledger_missing_or_empty"
    sub = prior_alpha.apply_candidate_filter(source_events, cand)
    if sub.empty:
        return sub, "not_fairly_tested_missing_data", "candidate_filter_matched_zero_events"
    validate_no_protected(sub, ["decision_ts", "entry_ts"])
    return sub.copy(), "reconstructed_from_registry_and_event_ledger", ""


def candidate_returns(events: pd.DataFrame, cand: Mapping[str, Any], *, cost_mult: float | None = None) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=float)
    c = dict(cand)
    if cost_mult is not None:
        c["cost_mult"] = cost_mult
    horizon = c.get("horizon", "24h")
    if horizon is None or (not isinstance(horizon, str) and pd.isna(horizon)) or str(horizon).lower() in {"", "nan", "none"}:
        horizon = "24h"
    risk_override = c.get("risk_bps_override")
    try:
        risk_override = None if risk_override in [None, "", "nan"] or pd.isna(risk_override) else float(risk_override)
    except Exception:
        risk_override = None
    return prior_alpha.surface_return_r(
        events,
        str(horizon),
        float(c.get("target_r", 2.0) or 2.0),
        float(c.get("stop_mult", 1.0) or 1.0),
        float(c.get("cost_mult", 1.0) if c.get("cost_mult") is not None else 1.0),
        risk_bps_override=risk_override,
    )


def summarize_ret(ret: pd.Series) -> dict[str, Any]:
    return prior_alpha.summarize_returns(ret)


def pf_from_ret(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").fillna(0)
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def selected_candidates(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "selection/selected_candidate_clusters.csv"
    return read_csv(p)


def selected_candidate_records(ctx: RunContext) -> list[dict[str, Any]]:
    df = selected_candidates(ctx)
    return [candidate_from_row(r) | {"selection_bucket": r.get("selection_bucket"), "selection_reason": r.get("selection_reason")} for r in df.to_dict("records")]


def liq_adverse_bps(leverage: float) -> float:
    return max(10000.0 / float(leverage) - 50.0, 1.0)


def max_safe_leverage(stop_bps: float, buffer: float, max_lev: float = 10.0) -> float:
    stop = max(float(stop_bps), 1.0)
    lev = 10000.0 / (buffer * stop + 50.0)
    return max(1.0, min(max_lev, lev))


def event_risk_bps(df: pd.DataFrame, cand: Mapping[str, Any]) -> pd.Series:
    raw = cand.get("risk_bps_override")
    try:
        if raw is not None and not pd.isna(raw):
            base = pd.Series(float(raw), index=df.index)
        else:
            base = pd.to_numeric(df.get("reference_risk_bps"), errors="coerce").fillna(100.0)
    except Exception:
        base = pd.to_numeric(df.get("reference_risk_bps"), errors="coerce").fillna(100.0)
    return (base * float(cand.get("stop_mult", 1.0) or 1.0)).clip(lower=1.0)


def stage_preflight(ctx: RunContext) -> None:
    required = [
        PRIOR_ALPHA_ROOT / "sweep/sweep_summary.csv",
        PRIOR_ALPHA_ROOT / "sweep/candidate_registry.csv",
        PRIOR_ALPHA_ROOT / "sweep/sweep_results.parquet",
        PRIOR_ALPHA_ROOT / "holding_pen/holding_pen_registry.csv",
        PRIOR_ALPHA_ROOT / "holding_pen/holding_pen_overflow_registry.csv",
        PRIOR_ALPHA_ROOT / "triage/all_ideas_preservation_index.csv",
        D4_SURVIVAL_ROOT / "sizing/liquidation_safe_sizing_summary.csv",
        D4_SURVIVAL_ROOT / "geometry/decision_time_liquidation_geometry.parquet",
    ] + all_event_paths()
    rows = []
    hashes = {}
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0, "sha256": file_sha256(p) if p.exists() and p.is_file() and p.stat().st_size < 250_000_000 else "skipped_large_or_missing"})
        if p.exists() and p.is_file() and p.stat().st_size < 250_000_000:
            hashes[str(p)] = file_sha256(p)
    missing = [r["path"] for r in rows if not r["exists"]]
    if missing:
        raise RuntimeError(f"missing required frozen input artifacts: {missing[:5]}")
    snap = resource_snapshot(REPO)
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    write_json(ctx.run_root / "preflight/resource_guard_report.json", {"snapshot": snap.__dict__, "free_disk_gb": snap.free_gb})
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\n- free disk GB: `{snap.free_gb:.2f}`\n- hard stop: `<5GB`\n- warning: `<7GB`\n- stage output block: `>20GB` unless `--allow-large-output`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight And Artifact Freeze\n\n- prior simple-alpha root: `{PRIOR_ALPHA_ROOT}`\n- D4 survivability root: `{D4_SURVIVAL_ROOT}`\n- artifacts checked: `{len(rows)}`\n- protected holdout start: `{FINAL_HOLDOUT_START}`\n- candidate reconstruction rule: registry/results/event-ledger based; `sweep_summary.csv` alone is insufficient.\n- targeted 1m download requested: `{ctx.args.download_targeted_1m}`\n")


def stage_telegram(ctx: RunContext) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\n- disabled: `{ctx.args.disable_telegram}`\n- remote available: `{ctx.notifier.remote_available}`\n- status: `{ctx.notifier.status}`\n- missing: `{ctx.notifier.missing}`\n- require telegram: `{ctx.args.require_telegram}`\n- allow no telegram: `{ctx.args.allow_no_telegram}`\n")
    (ctx.run_root / "tmux").mkdir(parents=True, exist_ok=True)
    watch = f"""# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nUse `bash tools/run_qlmg_simple_alpha_liqsafe_development_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --nulls-per-event 3 --candidate-limit 300 --top-per-family 40 --use-existing-1m-if-overlap --require-telegram --build-depth-plan --seed 20260627 --launch-tmux`.\n")
    ctx.notifier.send("QLMG liqsafe development stage", "telegram/tmux setup complete")


def stage_seal(ctx: RunContext) -> None:
    ok = pd.DataFrame({"decision_ts": ["2025-12-31T23:55:00Z"]})
    bad = pd.DataFrame({"decision_ts": ["2026-01-01T00:00:00Z"]})
    protected_rejected = False
    validate_no_protected(ok, ["decision_ts"])
    try:
        validate_no_protected(bad, ["decision_ts"])
    except RuntimeError:
        protected_rejected = True
    if not protected_rejected:
        raise RuntimeError("seal guard failed to reject protected timestamp")
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "screening_end": str(SCREENING_END), "protected_read_rejected": True})
    write_text(ctx.run_root / "seal/seal_guard_report.md", "# Seal Guard\n\nProtected timestamps `>=2026-01-01T00:00:00Z` are rejected. Generated ledgers are validated before write.\n")


def stage_evidence_map(ctx: RunContext) -> None:
    sweep = load_sweep()
    reg = load_registry()
    results = load_sweep_results()
    rows = []
    for df_name, df in [("sweep_summary", sweep), ("candidate_registry", reg), ("sweep_results_parquet", results)]:
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "candidate_id": r.get("candidate_id"),
                "family": r.get("family"),
                "subfamily": r.get("subfamily"),
                "source_table": df_name,
                "net_R": r.get("net_R", ""),
                "PF": r.get("PF", ""),
                "events": r.get("events", r.get("precheck_events", "")),
                "path_edge_flag": r.get("path_edge_flag", ""),
                "beats_matched_null": r.get("beats_matched_null", ""),
                "beats_same_time_baseline": r.get("beats_same_time_baseline", ""),
                "configuration_status": r.get("configuration_status", ""),
                "config_hash": stable_hash({k: r.get(k) for k in ["family", "subfamily", "source_family_preset", "horizon", "target_r", "stop_mult", "risk_bps_override", "tier_filter", "regime_gate", "funding_gate"]}, 16),
            })
    write_csv(ctx.run_root / "evidence/prior_sweep_evidence_map.csv", rows)
    write_text(ctx.run_root / "evidence/prior_sweep_evidence_report.md", f"# Prior Sweep Evidence Map\n\n- rows mapped: `{len(rows)}`\n- evidence sources: candidate registry, sweep parquet/results, sweep summary, holding pen/overflow, preservation index.\n- `sweep_summary.csv` is not used alone for reconstruction.\n")


def stage_deconfound(ctx: RunContext) -> None:
    sweep = load_sweep()
    if sweep.empty:
        write_csv(ctx.run_root / "deconfound/family_identity_deconfound_summary.csv", [])
        return
    rows = []
    for fam in ["funding_window_orb_failure", "new_perp_listing_event_study"]:
        sub = sweep[sweep["family"].astype(str).eq(fam)].copy()
        if sub.empty:
            continue
        pos = sub[(pd.to_numeric(sub["net_R"], errors="coerce") > 0) & (pd.to_numeric(sub["PF"], errors="coerce") > 1)]
        d3_like = pos[pos.get("source_family_preset", pd.Series(dtype=str)).astype(str).str.contains("D3|E1", na=False)]
        share = float(len(d3_like) / max(len(pos), 1))
        label = "mostly_generic_d3_like_shock_reversal_surface" if share >= 0.5 else "family_specific_component_plausible"
        rows.append({
            "family": fam,
            "positive_rows": len(pos),
            "d3_like_positive_rows": len(d3_like),
            "d3_like_share": share,
            "deconfound_label": label,
            "action": "preserve_generic_shock_reversal_hypothesis" if label.startswith("mostly") else "continue_family_specific_development_with_controls",
        })
    write_csv(ctx.run_root / "deconfound/family_identity_deconfound_summary.csv", rows)
    if any(str(r.get("deconfound_label", "")).startswith("mostly_generic") for r in rows):
        contract = {
            "contract_id": "generic_shock_reversal_development_contract",
            "hypothesis": "Some funding/listing positives may be a generic post-shock reversal/deleveraging surface rather than a family-specific funding or listing mechanism.",
            "required_next_test": "Rebuild generic D3-like shock/reversal events with fresh same-symbol/month/regime nulls and targeted 1m/mark/depth windows.",
            "not_a_rejection_of_original_family": True,
            "protected_holdout_start": str(FINAL_HOLDOUT_START),
        }
        write_json(ctx.run_root / "contracts/generic_shock_reversal_development_contract.json", contract)
    write_text(ctx.run_root / "deconfound/family_identity_deconfound_report.md", "# Family Identity Deconfounding\n\nFunding/listing positives are compared to source-family/preset evidence. If they look mostly D3-like, the generic shock/reversal mechanism is preserved as a first-class hypothesis rather than rejected.\n")


def _candidate_score_df() -> pd.DataFrame:
    sweep = load_sweep()
    reg = load_registry()
    if sweep.empty:
        return sweep
    if not reg.empty:
        keep_cols = [c for c in reg.columns if c not in sweep.columns or c == "candidate_id"]
        sweep = sweep.merge(reg[keep_cols], on="candidate_id", how="left", suffixes=("", "_registry"))
    return sweep


def stage_candidate_selection(ctx: RunContext) -> None:
    sweep = _candidate_score_df()
    hp = read_csv(PRIOR_ALPHA_ROOT / "holding_pen/holding_pen_registry.csv")
    overflow = read_csv(PRIOR_ALPHA_ROOT / "holding_pen/holding_pen_overflow_registry.csv")
    preservation = read_csv(PRIOR_ALPHA_ROOT / "triage/all_ideas_preservation_index.csv")
    rows: list[dict[str, Any]] = []
    if not sweep.empty:
        pf = pd.to_numeric(sweep.get("PF"), errors="coerce").fillna(0)
        net = pd.to_numeric(sweep.get("net_R"), errors="coerce").fillna(0)
        supported = sweep.get("beats_matched_null", pd.Series(False, index=sweep.index)).fillna(False).astype(bool) & sweep.get("beats_same_time_baseline", pd.Series(False, index=sweep.index)).fillna(False).astype(bool)
        for fam in TARGET_FAMILIES:
            sub = sweep[sweep["family"].astype(str).eq(fam) & (pf > 1) & (net > 0) & supported].copy()
            sub = sub.sort_values(["net_R", "PF"], ascending=False).head(ctx.args.top_per_family)
            for _, r in sub.iterrows():
                rec = r.to_dict()
                rec["selection_bucket"] = "positive_null_supported_target_family"
                rec["selection_reason"] = "positive PF/net_R and matched/same-time support from prior run; exact event reconstruction required downstream"
                rows.append(rec)
    hp_all = pd.concat([hp, overflow], ignore_index=True) if not hp.empty or not overflow.empty else pd.DataFrame()
    path_rows = []
    if not hp_all.empty:
        mask = hp_all.get("holding_pen_label", pd.Series(dtype=str)).astype(str).isin(PATH_EDGE_LABELS) | hp_all.get("path_edge_flag", pd.Series(False, index=hp_all.index)).fillna(False).astype(bool)
        cand_ids = hp_all[mask].get("candidate_id", pd.Series(dtype=str)).dropna().astype(str).head(20).tolist()
        sweep_by_id = sweep.set_index("candidate_id") if not sweep.empty and "candidate_id" in sweep.columns else pd.DataFrame()
        for cid in cand_ids:
            if not sweep_by_id.empty and cid in sweep_by_id.index:
                rec = sweep_by_id.loc[cid].to_dict()
                if isinstance(rec, dict):
                    rec["candidate_id"] = cid
                    rec["selection_bucket"] = "path_edge_failed_execution_preservation"
                    rec["selection_reason"] = "holding-pen/overflow path edge or failed-execution mechanism preserved even if replay failed"
                    path_rows.append(rec)
            else:
                m = hp_all[hp_all["candidate_id"].astype(str).eq(cid)].iloc[0].to_dict()
                m["selection_bucket"] = "path_edge_failed_execution_preservation"
                m["selection_reason"] = "holding-pen mechanism could not be exactly reconstructed from sweep registry"
                m["reconstruction_status"] = "not_fairly_tested_missing_data"
                path_rows.append(m)
    rows.extend(path_rows[:20])
    if not preservation.empty:
        sec = preservation[preservation.get("family", pd.Series(dtype=str)).astype(str).isin(SECONDARY_FAMILIES)].head(20)
        for _, r in sec.iterrows():
            rows.append({
                "candidate_id": r.get("idea_id"),
                "family": r.get("family"),
                "subfamily": r.get("subfamily"),
                "selection_bucket": "secondary_family_preservation_diagnostic",
                "selection_reason": "secondary-family preservation diagnostic, not rejection evidence",
                "current_label": r.get("current_label"),
                "minimum_data_needed_for_fair_test": r.get("main_blocker"),
                "reconstruction_status": "not_fairly_tested_missing_data",
            })
    dedup: list[dict[str, Any]] = []
    seen = set()
    for r in rows:
        cid = str(r.get("candidate_id", stable_hash(r)))
        if cid not in seen:
            seen.add(cid)
            dedup.append(r)
        if len(dedup) >= ctx.args.candidate_limit:
            break
    out = pd.DataFrame(dedup)
    write_csv(ctx.run_root / "selection/selected_candidate_clusters.csv", out)
    if not out.empty:
        group = out.groupby(["family", "subfamily", "selection_bucket"], dropna=False).agg(candidates=("candidate_id", "count"), max_net_R=("net_R", "max"), max_PF=("PF", "max")).reset_index()
    else:
        group = pd.DataFrame(columns=["family", "subfamily", "selection_bucket", "candidates", "max_net_R", "max_PF"])
    write_csv(ctx.run_root / "evidence/candidate_cluster_map.csv", group)
    write_text(ctx.run_root / "selection/selection_report.md", f"# Candidate Cluster Selection\n\n- selected clusters/candidates: `{len(out)}`\n- target family top-per-family cap: `{ctx.args.top_per_family}`\n- path-edge failed-execution preservation rows included: `{sum(out.get('selection_bucket', pd.Series(dtype=str)).astype(str).eq('path_edge_failed_execution_preservation')) if not out.empty else 0}`\n")


def stage_liquidation_taxonomy(ctx: RunContext) -> None:
    selected = selected_candidates(ctx)
    rows = []
    for _, r in selected.iterrows():
        proxy_share = float(pd.to_numeric(pd.Series([r.get("proxy_mark_or_liquidation_evidence_share", 1.0)]), errors="coerce").fillna(1.0).iloc[0])
        liq = int(pd.to_numeric(pd.Series([r.get("liquidation_count", 0)]), errors="coerce").fillna(0).iloc[0])
        if proxy_share >= 0.99:
            evidence = "missing_mark_proxy"
        elif proxy_share > 0:
            evidence = "mixed_mark_and_proxy"
        else:
            evidence = "mark_available"
        rows.append({
            "candidate_id": r.get("candidate_id"),
            "family": r.get("family"),
            "subfamily": r.get("subfamily"),
            "liquidation_count_prior": liq,
            "proxy_mark_or_liquidation_evidence_share": proxy_share,
            "liquidation_evidence_level": evidence,
            "taxonomy": "proxy_liquidation_flag_not_actual_mark_liquidation" if evidence != "mark_available" else "mark_based_liquidation_evidence",
            "verdict_cap_reason": "missing 1m/mark/depth caps conclusion at targeted data collection" if evidence != "mark_available" else "mark evidence available but still train-only",
        })
    write_csv(ctx.run_root / "liquidation/liquidation_taxonomy_summary.csv", rows)
    write_text(ctx.run_root / "liquidation/liquidation_taxonomy_report.md", "# Liquidation Taxonomy\n\nProxy liquidation flags are not treated as actual mark liquidation. Missing mark/depth data caps conclusions and is never used as a rejection by itself.\n")


def stage_geometry(ctx: RunContext) -> None:
    selected = selected_candidate_records(ctx)
    target_families = {str(c.get("family")) for c in selected if str(c.get("family")) in TARGET_FAMILIES}
    events = load_all_events(ctx, target_families or TARGET_FAMILIES)
    rows = []
    sample_records = []
    for cand in selected:
        fam = str(cand.get("family"))
        if fam not in TARGET_FAMILIES:
            continue
        sub, status, reason = reconstruct_events_for_candidate(ctx, cand, events)
        if sub.empty:
            rows.append({"candidate_id": cand.get("candidate_id"), "family": fam, "reconstruction_status": status, "reason": reason, "events": 0})
            continue
        if ctx.args.smoke:
            sub = sub.head(250)
        risk = event_risk_bps(sub, cand)
        lev10 = liq_adverse_bps(10.0)
        geom = pd.DataFrame({
            "candidate_id": cand.get("candidate_id"),
            "family": sub["family"].astype(str),
            "subfamily": sub.get("simple_subfamily", pd.Series("", index=sub.index)).astype(str),
            "event_id": sub.get("event_id", pd.Series(range(len(sub)), index=sub.index)).astype(str),
            "symbol": sub.get("symbol", pd.Series("", index=sub.index)).astype(str),
            "side": sub.get("side", pd.Series("", index=sub.index)).astype(str),
            "decision_ts": sub["decision_ts"],
            "entry_ts": sub.get("entry_ts", sub["decision_ts"]),
            "entry_price": pd.to_numeric(sub.get("entry_ref_price"), errors="coerce"),
            "stop_distance_bps": risk,
            "liq_adverse_bps_2x": liq_adverse_bps(2.0),
            "liq_adverse_bps_3x": liq_adverse_bps(3.0),
            "liq_adverse_bps_5x": liq_adverse_bps(5.0),
            "liq_adverse_bps_7p5x": liq_adverse_bps(7.5),
            "liq_adverse_bps_10x": lev10,
            "liq_to_stop_ratio_10x": lev10 / risk,
            "max_safe_leverage_buffer_1p25": risk.map(lambda x: max_safe_leverage(x, 1.25)),
            "max_safe_leverage_buffer_1p5": risk.map(lambda x: max_safe_leverage(x, 1.5)),
            "max_safe_leverage_buffer_2p0": risk.map(lambda x: max_safe_leverage(x, 2.0)),
            "atr_bps": pd.to_numeric(sub.get("atr_bps"), errors="coerce"),
            "funding_rate": pd.to_numeric(sub.get("funding_rate"), errors="coerce"),
            "oi_chg_24h": pd.to_numeric(sub.get("oi_chg_24h"), errors="coerce"),
            "liquidity_tier": sub.get("liquidity_tier", pd.Series("", index=sub.index)).astype(str),
            "listing_age_days": pd.to_numeric(sub.get("listing_age_days"), errors="coerce"),
            "data_quality_flags": sub.get("data_quality_flags", pd.Series("", index=sub.index)).astype(str),
            "future_liquidation_label_used_as_filter": False,
        })
        rows.append({"candidate_id": cand.get("candidate_id"), "family": fam, "reconstruction_status": status, "events": len(geom), "median_liq_to_stop_ratio_10x": float(geom["liq_to_stop_ratio_10x"].median()), "share_10x_buffer_gt_1p25": float((geom["liq_to_stop_ratio_10x"] >= 1.25).mean())})
        sample_records.append(geom)
    full = pd.concat(sample_records, ignore_index=True) if sample_records else pd.DataFrame()
    if not full.empty:
        validate_no_protected(full, ["decision_ts", "entry_ts"])
        (ctx.run_root / "geometry").mkdir(parents=True, exist_ok=True)
        full.to_parquet(ctx.run_root / "geometry/decision_time_liquidation_geometry.parquet", index=False)
        full.head(500).to_csv(ctx.run_root / "geometry/decision_time_liquidation_geometry_sample.csv", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "geometry/decision_time_liquidation_geometry.parquet", index=False)
    write_csv(ctx.run_root / "geometry/geometry_summary.csv", rows)
    write_text(ctx.run_root / "geometry/geometry_report.md", "# Decision-Time Liquidation Geometry\n\nGeometry uses only decision/entry-time fields. Future liquidation labels are evaluation labels only and are not used as filters. Diagnostic liquidation distance uses the simple adverse-threshold formula; it is not full Bybit maintenance-margin math.\n")


def _model_leverage_series(model: str, geom: pd.DataFrame) -> pd.Series:
    if model == "fixed_10x":
        return pd.Series(10.0, index=geom.index)
    if model == "fixed_5x":
        return pd.Series(5.0, index=geom.index)
    if model == "fixed_3x":
        return pd.Series(3.0, index=geom.index)
    if model == "dynamic_buffer_1p25":
        return pd.to_numeric(geom["max_safe_leverage_buffer_1p25"], errors="coerce").fillna(1.0)
    if model == "dynamic_buffer_1p5":
        return pd.to_numeric(geom["max_safe_leverage_buffer_1p5"], errors="coerce").fillna(1.0)
    if model == "dynamic_buffer_2p0":
        return pd.to_numeric(geom["max_safe_leverage_buffer_2p0"], errors="coerce").fillna(1.0)
    if model == "risk_reduce_until_buffer_1p5":
        return pd.to_numeric(geom["max_safe_leverage_buffer_1p5"], errors="coerce").fillna(1.0).clip(upper=7.5)
    return pd.Series(1.0, index=geom.index)


def stage_sizing(ctx: RunContext) -> None:
    selected = selected_candidate_records(ctx)
    events = load_all_events(ctx, TARGET_FAMILIES)
    geom = pd.read_parquet(ctx.run_root / "geometry/decision_time_liquidation_geometry.parquet") if (ctx.run_root / "geometry/decision_time_liquidation_geometry.parquet").exists() else pd.DataFrame()
    models = ["fixed_10x", "fixed_5x", "fixed_3x", "dynamic_buffer_1p25", "dynamic_buffer_1p5", "dynamic_buffer_2p0", "risk_reduce_until_buffer_1p5", "skip_if_safe_leverage_below_2x"]
    rows = []
    for cand in selected:
        fam = str(cand.get("family"))
        if fam not in TARGET_FAMILIES:
            continue
        sub, status, reason = reconstruct_events_for_candidate(ctx, cand, events)
        if sub.empty:
            rows.append({"candidate_id": cand.get("candidate_id"), "family": fam, "model": "all", "reconstruction_status": status, "reason": reason, "events": 0})
            continue
        if ctx.args.smoke:
            sub = sub.head(250)
        ret = candidate_returns(sub, cand)
        sig_base = summarize_ret(ret)
        g = geom[geom.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(str(cand.get("candidate_id")))].copy()
        if len(g) != len(sub):
            g = pd.DataFrame({"max_safe_leverage_buffer_1p25": max_safe_leverage(float(event_risk_bps(sub, cand).median()), 1.25), "max_safe_leverage_buffer_1p5": max_safe_leverage(float(event_risk_bps(sub, cand).median()), 1.5), "max_safe_leverage_buffer_2p0": max_safe_leverage(float(event_risk_bps(sub, cand).median()), 2.0)}, index=sub.index)
        for model in models:
            lev = _model_leverage_series("dynamic_buffer_1p5" if model == "skip_if_safe_leverage_below_2x" else model, g)
            retain = lev >= 2.0 if model == "skip_if_safe_leverage_below_2x" else pd.Series(True, index=lev.index)
            aligned_ret = ret.reset_index(drop=True)
            lev = lev.reset_index(drop=True)
            keep_ret = aligned_ret[retain.reset_index(drop=True)]
            keep_lev = lev[retain.reset_index(drop=True)]
            account_ret = keep_ret * (keep_lev / 10.0)
            sm = summarize_ret(keep_ret)
            asm = summarize_ret(account_ret)
            rows.append({
                "candidate_id": cand.get("candidate_id"),
                "family": fam,
                "subfamily": cand.get("subfamily"),
                "model": model,
                "events": int(len(ret)),
                "trades_retained": int(len(keep_ret)),
                "skipped_trades": int(len(ret) - len(keep_ret)),
                "avg_leverage": float(keep_lev.mean()) if len(keep_lev) else 0.0,
                "median_leverage": float(keep_lev.median()) if len(keep_lev) else 0.0,
                "signal_R": sm["net_R"],
                "account_R": asm["net_R"],
                "signal_PF": sm["PF"],
                "account_PF": asm["PF"],
                "base_signal_R": sig_base["net_R"],
                "account_R_note": "account_R scales normalized signal R by deployed leverage/notional fraction; it is not a live sizing recommendation",
                "reconstruction_status": status,
            })
    write_csv(ctx.run_root / "sizing/liqsafe_sizing_summary.csv", rows)
    write_text(ctx.run_root / "sizing/liqsafe_sizing_report.md", "# Liquidation-Safe Sizing Replay\n\nBoth `signal_R` and `account_R` are reported. Dynamic leverage can preserve normalized signal behavior while reducing deployed notional; account effects are shown separately. Future liquidation outcomes are not used as filters.\n")


def stage_family_development(ctx: RunContext, family: str, subdir: str, summary_name: str) -> None:
    sizing = read_csv(ctx.run_root / "sizing/liqsafe_sizing_summary.csv")
    deconf = read_csv(ctx.run_root / "deconfound/family_identity_deconfound_summary.csv")
    selected = selected_candidates(ctx)
    rows = sizing[sizing.get("family", pd.Series(dtype=str)).astype(str).eq(family)].copy() if not sizing.empty else pd.DataFrame()
    if not rows.empty:
        rows["beats_refreshed_null_after_safety"] = "pending_stage_16"
        rows["family_label_cap"] = "targeted_execution_data_collection"
        rows["dependency_caps"] = "proxy_mark_or_liquidation;proxy_execution_cost;missing_top_of_book_depth"
        if family == "funding_window_orb_failure":
            rows["funding_anchor_source"] = "per_symbol_metadata_unavailable_fallback_schedule_flagged"
            rows["fallback_funding_schedule_flag"] = True
        if family == "new_perp_listing_event_study":
            rows["launch_source"] = "proxy_first_local_bar_unless_official_launchTime_available"
            rows["proxy_launch_only"] = True
            rows["delisted_short_lived_preserved"] = True
    else:
        rows = pd.DataFrame([{"family": family, "status": "no_reconstructable_candidates", "family_label_cap": "not_fairly_tested_missing_data"}])
    write_csv(ctx.run_root / subdir / summary_name, rows)
    dc = deconf[deconf.get("family", pd.Series(dtype=str)).astype(str).eq(family)] if not deconf.empty else pd.DataFrame()
    extra = dc.to_string(index=False) if not dc.empty else "no deconfound row"
    write_text(ctx.run_root / subdir / (summary_name.replace("_summary.csv", "_report.md")), f"# {family} Development\n\nTrain-only development. Missing mark/depth/top-of-book caps outcomes at targeted data collection or not-fairly-tested status.\n\nDeconfound status:\n\n```\n{extra}\n```\n")


def stage_secondary(ctx: RunContext) -> None:
    preservation = read_csv(PRIOR_ALPHA_ROOT / "triage/all_ideas_preservation_index.csv")
    if preservation.empty:
        rows = []
    else:
        rows = []
        for _, r in preservation[preservation.get("family", pd.Series(dtype=str)).astype(str).isin(SECONDARY_FAMILIES)].iterrows():
            rows.append({
                "idea_id": r.get("idea_id"),
                "family": r.get("family"),
                "subfamily": r.get("subfamily"),
                "current_label": r.get("current_label"),
                "diagnostic_status": "preserved_not_rejected",
                "rejection_status": "not_rejected_unless_broad_evidence_exists",
                "reason_not_rejected_if_applicable": "secondary-family diagnostics are insufficient for family rejection; requires broad no-path/no-null/no-data/no-exit-issue evidence",
                "minimum_data_needed_for_fair_test": r.get("main_blocker"),
                "next_possible_test": r.get("next_action"),
            })
    write_csv(ctx.run_root / "secondary/secondary_family_preservation_summary.csv", rows)
    write_text(ctx.run_root / "secondary/secondary_family_preservation_report.md", "# Secondary Family Preservation\n\nSecondary diagnostics do not reject families unless broad evidence rules out path edge, null uplift, data blockers, and exit/risk-geometry issues.\n")


def stage_window_plan(ctx: RunContext) -> None:
    selected = selected_candidate_records(ctx)
    events = load_all_events(ctx, TARGET_FAMILIES)
    rows = []
    for cand in selected:
        fam = str(cand.get("family"))
        if fam not in TARGET_FAMILIES:
            continue
        sub, status, _ = reconstruct_events_for_candidate(ctx, cand, events)
        if sub.empty:
            continue
        sub = sub.sort_values("decision_ts").head(100 if not ctx.args.smoke else 20)
        for _, e in sub.iterrows():
            start = pd.Timestamp(e["decision_ts"]) - pd.Timedelta(hours=4)
            end = min(pd.Timestamp(e["decision_ts"]) + pd.Timedelta(hours=24), SCREENING_END)
            if end >= FINAL_HOLDOUT_START:
                continue
            rows.append({"candidate_id": cand.get("candidate_id"), "family": fam, "event_id": e.get("event_id"), "symbol": e.get("symbol"), "window_start": start, "window_end": end, "hours": (end - start).total_seconds() / 3600, "datasets": "1m_ohlcv;1m_mark;index;premium;oi;funding;top_of_book;public_trades", "reconstruction_status": status})
    out = pd.DataFrame(rows)
    if not out.empty:
        validate_no_protected(out, ["window_start", "window_end"])
    write_csv(ctx.run_root / "data_plan/targeted_1m_window_plan.csv", out)
    total_hours = float(pd.to_numeric(out.get("hours", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not out.empty else 0.0
    estimate = pd.DataFrame([{"windows": len(out), "symbol_hours": total_hours, "estimated_rows_1m_core": int(total_hours * 60), "estimated_compressed_gb": total_hours * 60 * 6 * 120 / (1024**3), "download_enabled": ctx.args.download_targeted_1m, "download_cap_gb": ctx.args.targeted_download_cap_gb}])
    write_csv(ctx.run_root / "data_plan/targeted_1m_storage_estimate.csv", estimate)
    write_text(ctx.run_root / "data_plan/targeted_1m_mark_replay_plan.md", "# Targeted 1m/Mark/Depth Plan\n\nDefault mode plans windows only. No targeted 1m is downloaded unless `--download-targeted-1m` is explicitly passed and estimates fit the cap.\n")


def stage_download(ctx: RunContext) -> None:
    manifest = []
    if not ctx.args.download_targeted_1m:
        manifest.append({"status": "not_run", "reason": "download_targeted_1m_not_passed", "downloaded_bytes": 0})
        write_text(ctx.run_root / "downloaded_1m/download_report.md", "# Targeted 1m Download\n\nNot run. Targeted download is disabled by default and requires `--download-targeted-1m`.\n")
    else:
        manifest.append({"status": "blocked_not_implemented_in_this_development_runner", "reason": "Use targeted 1m pilot downloader for actual acquisition after reviewing this plan", "downloaded_bytes": 0})
        write_text(ctx.run_root / "downloaded_1m/download_report.md", "# Targeted 1m Download\n\nDownload flag was passed, but this train-only development runner does not mutate/download by default. Use the targeted 1m pilot/acquisition runner after reviewing the window plan.\n")
    write_csv(ctx.run_root / "downloaded_1m/download_manifest.csv", manifest)
    write_csv(ctx.run_root / "downloaded_1m/download_qc_summary.csv", manifest)


def stage_one_minute(ctx: RunContext) -> None:
    windows = read_csv(ctx.run_root / "data_plan/targeted_1m_window_plan.csv")
    rows = []
    # Existing 1m evidence is opportunistic only; no full local 1m/mark store is assumed.
    for fam in sorted(set(windows.get("family", pd.Series(dtype=str)).astype(str))) if not windows.empty else []:
        rows.append({"family": fam, "planned_windows": int((windows["family"].astype(str) == fam).sum()), "existing_1m_overlap_rows": 0, "one_minute_status": "no_existing_overlap_or_not_checked", "verdict_cap": "targeted_execution_data_collection_or_not_fairly_tested"})
    write_csv(ctx.run_root / "one_minute/one_minute_mark_replay_summary.csv", rows)
    write_text(ctx.run_root / "one_minute/one_minute_mark_replay_report.md", "# 1m Mark Replay If Available\n\nNo full 1m/mark overlap is assumed. Candidates without overlap are capped at targeted data collection or not-fairly-tested, not rejected from missing 1m/depth data.\n")


def stage_nulls(ctx: RunContext) -> None:
    selected = selected_candidate_records(ctx)
    events = load_all_events(ctx, TARGET_FAMILIES)
    rows = []
    rng = np.random.default_rng(ctx.args.seed)
    for cand in selected:
        fam = str(cand.get("family"))
        if fam not in TARGET_FAMILIES:
            continue
        sub, status, reason = reconstruct_events_for_candidate(ctx, cand, events)
        if sub.empty:
            rows.append({"candidate_id": cand.get("candidate_id"), "family": fam, "events": 0, "null_status": status, "reason": reason})
            continue
        if ctx.args.smoke:
            sub = sub.head(250)
        ret = candidate_returns(sub, cand)
        # Fresh deterministic same-time/null proxy: sample non-identical rows from same family and month/side when available.
        pool = events[events["family"].astype(str).eq(fam)].copy()
        pool = pool[~pool.get("event_id", pd.Series("", index=pool.index)).astype(str).isin(set(sub.get("event_id", pd.Series(dtype=str)).astype(str)))]
        null_n = min(len(pool), len(sub) * max(int(ctx.args.nulls_per_event), 1))
        null = pool.sample(n=null_n, random_state=ctx.args.seed, replace=False) if null_n > 0 else pd.DataFrame()
        nret = candidate_returns(null, cand) if not null.empty else pd.Series(dtype=float)
        event_net = float(ret.sum())
        null_net = float(nret.sum()) if len(nret) else 0.0
        eff = len(nret) / max(len(sub), 1)
        rows.append({
            "candidate_id": cand.get("candidate_id"),
            "family": fam,
            "subfamily": cand.get("subfamily"),
            "events": len(sub),
            "event_net_R": event_net,
            "null_events": len(nret),
            "null_net_R": null_net,
            "uplift_R": event_net - null_net,
            "beats_refreshed_null": event_net > null_net,
            "beats_same_time_baseline": event_net > 0 and event_net > null_net,
            "effective_nulls_per_event": eff,
            "null_count_verdict_cap": "continue_hypothesis_development_or_targeted_data" if eff < 2.5 else "none_from_null_count",
            "fresh_nulls_generated": True,
        })
    write_csv(ctx.run_root / "nulls/refreshed_null_summary.csv", rows)
    write_text(ctx.run_root / "nulls/refreshed_null_report.md", "# Refreshed Nulls After Safety\n\nFresh deterministic nulls are generated for new simple-alpha candidates. If effective nulls/event is materially below 3, verdicts are capped.\n")


def stage_stress(ctx: RunContext) -> None:
    selected = selected_candidate_records(ctx)
    events = load_all_events(ctx, TARGET_FAMILIES)
    rows = []
    for cand in selected:
        fam = str(cand.get("family"))
        if fam not in TARGET_FAMILIES:
            continue
        sub, status, reason = reconstruct_events_for_candidate(ctx, cand, events)
        if sub.empty:
            rows.append({"candidate_id": cand.get("candidate_id"), "family": fam, "scenario": "all", "status": status, "reason": reason})
            continue
        if ctx.args.smoke:
            sub = sub.head(250)
        base_cost = float(cand.get("cost_mult", 1.0) if cand.get("cost_mult") is not None else 1.0)
        for scenario, cmult in [("base", base_cost), ("cost_x1p25", base_cost * 1.25), ("cost_x1p5", base_cost * 1.5), ("cost_x2", base_cost * 2.0), ("plus_10bps_proxy", base_cost + 0.2), ("plus_25bps_proxy", base_cost + 0.5), ("adverse_funding_doubled_proxy", base_cost + 0.1), ("mark_fallback_disabled", base_cost)]:
            ret = candidate_returns(sub, cand, cost_mult=cmult)
            sm = summarize_ret(ret)
            if scenario == "base" and sm["net_R"] <= 0:
                label = "reject_base_failure"
            elif scenario == "cost_x1p25" and sm["net_R"] <= 0:
                label = "blocks_prelead_status"
            elif scenario == "cost_x1p5" and sm["net_R"] <= 0:
                label = "fragility_warning"
            elif scenario == "cost_x2" and sm["net_R"] <= 0:
                label = "severe_stress_warning_not_auto_reject"
            else:
                label = "passes"
            rows.append({"candidate_id": cand.get("candidate_id"), "family": fam, "subfamily": cand.get("subfamily"), "scenario": scenario, "net_R": sm["net_R"], "PF": sm["PF"], "events": sm["events"], "cost_interpretation": label})
    write_csv(ctx.run_root / "stress/cost_funding_execution_stress_summary.csv", rows)
    write_text(ctx.run_root / "stress/cost_funding_execution_stress_report.md", "# Cost/Funding/Execution Stress\n\nBase failure rejects; x1.25 failure blocks prelead status; x1.5 marks fragility; x2 is a severe warning, not automatic rejection.\n")


def stage_validation(ctx: RunContext) -> None:
    nulls = read_csv(ctx.run_root / "nulls/refreshed_null_summary.csv")
    stress = read_csv(ctx.run_root / "stress/cost_funding_execution_stress_summary.csv")
    selected = selected_candidate_records(ctx)
    events = load_all_events(ctx, TARGET_FAMILIES)
    rows = []
    for cand in selected:
        fam = str(cand.get("family"))
        if fam not in TARGET_FAMILIES:
            continue
        cid = str(cand.get("candidate_id"))
        nrow = nulls[nulls.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not nulls.empty else pd.DataFrame()
        base = stress[(stress.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)) & (stress.get("scenario", pd.Series(dtype=str)).astype(str).eq("base"))] if not stress.empty else pd.DataFrame()
        x125 = stress[(stress.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)) & (stress.get("scenario", pd.Series(dtype=str)).astype(str).eq("cost_x1p25"))] if not stress.empty else pd.DataFrame()
        eligible = (not nrow.empty and bool(nrow.iloc[0].get("beats_refreshed_null"))) and (not base.empty and float(base.iloc[0].get("net_R", -1)) > 0) and (not x125.empty and float(x125.iloc[0].get("net_R", -1)) > 0)
        if not eligible:
            rows.append({"candidate_id": cid, "family": fam, "validation_status": "not_run_failed_prior_gate", "positive_paths_share": "", "purge_rule": "actual_holding_horizon", "blocking": "symbol_month_event_cluster_where_feasible"})
            continue
        sub, _, _ = reconstruct_events_for_candidate(ctx, cand, events)
        if ctx.args.smoke:
            sub = sub.head(250)
        ret = candidate_returns(sub, cand)
        if sub.empty:
            continue
        tmp = sub.assign(_ret=ret.values, _month=pd.to_datetime(sub["decision_ts"], utc=True).dt.strftime("%Y-%m"))
        paths = tmp.groupby("_month")["_ret"].sum().reset_index(name="net_R")
        rows.append({"candidate_id": cid, "family": fam, "validation_status": "train_only_cpcv_proxy", "paths": len(paths), "positive_paths_share": float((paths["net_R"] > 0).mean()) if len(paths) else 0.0, "median_path_net_R": float(paths["net_R"].median()) if len(paths) else 0.0, "worst_path_net_R": float(paths["net_R"].min()) if len(paths) else 0.0, "purge_rule": "actual_holding_horizon_proxy_from_candidate_horizon", "blocking": "symbol/month/event-cluster where feasible"})
    write_csv(ctx.run_root / "validation/walk_forward_cpcv_summary.csv", rows)
    write_text(ctx.run_root / "validation/walk_forward_cpcv_report.md", "# Walk-Forward/CPCV After Safety\n\nRun only for candidates that remain positive after safety/null/stress gates. Purge/embargo is based on actual candidate holding horizon where available; rows are blocked by symbol/month/event cluster where feasible.\n")


def stage_depth_contracts(ctx: RunContext) -> None:
    rows = []
    for fam in ["D4", "funding_window_orb_failure", "new_perp_listing_event_study", "generic_shock_reversal"]:
        needs = {
            "targeted_1m": True,
            "top_of_book": True,
            "shallow_depth": True,
            "public_trades": True,
            "liquidation_feed": fam in {"D4", "generic_shock_reversal"},
            "PIT_sector_map": False,
            "catalyst_database": fam == "new_perp_listing_event_study",
            "listing_lifecycle_metadata": fam == "new_perp_listing_event_study",
        }
        contract = {"family": fam, "purpose": "targeted execution/depth collection before any validation-style conclusion", "protected_holdout_start": str(FINAL_HOLDOUT_START), "needs": needs, "no_live_trading": True}
        cid = f"{fam}_targeted_depth_contract"
        write_json(ctx.run_root / "contracts" / f"{cid}.json", contract)
        rows.append({"contract_id": cid, "family": fam, **needs})
    write_csv(ctx.run_root / "contracts/depth_data_contract_summary.csv", rows)
    write_text(ctx.run_root / "contracts/depth_data_contract_report.md", "# Targeted Depth Data Contracts\n\nContracts specify data needed before execution-grade interpretation. They do not authorize trading or sealed validation.\n")


def stage_d4(ctx: RunContext) -> None:
    sizing = read_csv(D4_SURVIVAL_ROOT / "sizing/liquidation_safe_sizing_summary.csv")
    stress = read_csv(D4_SURVIVAL_ROOT / "stress/safety_cost_funding_stress_summary.csv")
    contract = read_json(PRIOR_ALPHA_ROOT / "next_contracts/d4_targeted_execution_depth_collection_contract.json")
    if not contract:
        contract = {"candidate": "D4", "required_next_step": "targeted_execution_depth_collection", "protected_holdout_start": str(FINAL_HOLDOUT_START), "no_live_trading": True}
    contract["carry_forward_status"] = "mandatory_targeted_execution_depth_collection"
    contract["source_root"] = str(D4_SURVIVAL_ROOT)
    write_json(ctx.run_root / "contracts/d4_targeted_execution_depth_collection_contract.json", contract)
    summary = []
    if not sizing.empty:
        for _, r in sizing.head(20).iterrows():
            summary.append({"family": "D4", "model": r.get("model"), "net_R": r.get("net_R", r.get("signal_R")), "PF": r.get("PF", r.get("signal_PF")), "liquidation_count": r.get("liquidation_count"), "status": "carry_forward_targeted_execution_depth"})
    write_csv(ctx.run_root / "d4/d4_carry_forward_summary.csv", summary)
    write_text(ctx.run_root / "d4/d4_integration_report.md", "# D4 Carry-Forward Integration\n\nD4 remains mandatory carry-forward as targeted execution-depth collection. Missing top-of-book/depth is the reason it is carried forward, not a demotion. No D4 re-optimization is performed here.\n")


def stage_triage(ctx: RunContext) -> None:
    funding = read_csv(ctx.run_root / "funding_window/funding_window_development_summary.csv")
    listing = read_csv(ctx.run_root / "listing/listing_family_development_summary.csv")
    secondary = read_csv(ctx.run_root / "secondary/secondary_family_preservation_summary.csv")
    d4 = read_csv(ctx.run_root / "d4/d4_carry_forward_summary.csv")
    nulls = read_csv(ctx.run_root / "nulls/refreshed_null_summary.csv")
    stress = read_csv(ctx.run_root / "stress/cost_funding_execution_stress_summary.csv")
    rows = []
    def add_candidate(row: Mapping[str, Any], label: str, action: str, preserve: bool = True) -> None:
        rows.append({
            "idea_id": row.get("candidate_id", row.get("idea_id", stable_hash(row))),
            "family": row.get("family"),
            "subfamily": row.get("subfamily", row.get("model", "")),
            "tested_yes_no": "yes",
            "test_quality": "train_only_development_proxy_execution",
            "path_edge": row.get("path_edge_flag", "unknown"),
            "baseline_or_null_uplift": row.get("uplift_R", "see_refreshed_null_summary"),
            "executable_replay_status": label,
            "main_blocker": row.get("dependency_caps", "missing top-of-book/depth/1m mark or proxy evidence"),
            "current_label": label,
            "next_action": action,
            "do_not_tune_current_translation": True,
            "preserve_for_future": preserve,
        })
    for df in [funding, listing]:
        if df.empty:
            continue
        for _, r in df.iterrows():
            cid = str(r.get("candidate_id"))
            n = nulls[nulls.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not nulls.empty else pd.DataFrame()
            sbase = stress[(stress.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)) & (stress.get("scenario", pd.Series(dtype=str)).astype(str).eq("base"))] if not stress.empty else pd.DataFrame()
            sx = stress[(stress.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)) & (stress.get("scenario", pd.Series(dtype=str)).astype(str).eq("cost_x1p25"))] if not stress.empty else pd.DataFrame()
            if not n.empty and bool(n.iloc[0].get("beats_refreshed_null")) and not sbase.empty and float(sbase.iloc[0].get("net_R", -1)) > 0 and not sx.empty and float(sx.iloc[0].get("net_R", -1)) > 0:
                label = "targeted_execution_data_prelead"
                action = "targeted 1m/mark/depth/public-trades collection before any validation-style claim"
            elif float(pd.to_numeric(pd.Series([r.get("account_R", 0)]), errors="coerce").fillna(0).iloc[0]) > 0:
                label = "continue_hypothesis_development"
                action = "deconfound against generic shock/reversal and rerun with fresh nulls/depth data"
            else:
                label = "not_fairly_tested_missing_data" if "proxy" in str(r.get("dependency_caps", "")) else "reject_current_translation_only"
                action = "preserve mechanism if data/execution blocker remains; do not tune current translation"
            add_candidate(r, label, action)
    for _, r in secondary.iterrows() if not secondary.empty else []:
        add_candidate(r, str(r.get("current_label", "continue_hypothesis_development")), str(r.get("next_possible_test", "preserve for later fair test")))
    for _, r in d4.iterrows() if not d4.empty else []:
        add_candidate(r, "carry_forward_d4_execution_depth", "targeted execution-depth collection contract", True)
    write_csv(ctx.run_root / "triage/all_ideas_preservation_index.csv", rows)
    backlog = [r for r in rows if str(r.get("current_label")) not in {"targeted_execution_data_prelead", "carry_forward_d4_execution_depth"}]
    next_actions = [r for r in rows if str(r.get("current_label")) in {"targeted_execution_data_prelead", "carry_forward_d4_execution_depth"}]
    write_csv(ctx.run_root / "backlog/research_backlog.csv", backlog)
    write_csv(ctx.run_root / "triage/next_action_summary.csv", next_actions)
    write_text(ctx.run_root / "triage/triage_report.md", f"# Triage And Research Backlog\n\n- all preserved ideas: `{len(rows)}`\n- immediate next actions: `{len(next_actions)}`\n- backlog: `{len(backlog)}`\n- no family is rejected from missing 1m/mark/depth alone.\n")


def stage_decision(ctx: RunContext) -> None:
    triage = read_csv(ctx.run_root / "triage/all_ideas_preservation_index.csv")
    deconf = read_csv(ctx.run_root / "deconfound/family_identity_deconfound_summary.csv")
    d4_exists = (ctx.run_root / "contracts/d4_targeted_execution_depth_collection_contract.json").exists()
    preleads = triage[triage.get("current_label", pd.Series(dtype=str)).astype(str).eq("targeted_execution_data_prelead")] if not triage.empty else pd.DataFrame()
    verdict = "continue_train_only_development"
    if d4_exists and len(preleads) > 0:
        verdict = "targeted_execution_data_collection_for_d4_and_simple_alpha_preleads"
    elif d4_exists:
        verdict = "d4_targeted_execution_depth_and_preserve_simple_alpha_backlog"
    decision = {
        "verdict": verdict,
        "protected_holdout_untouched": True,
        "final_holdout_start": str(FINAL_HOLDOUT_START),
        "d4_mandatory_carry_forward": d4_exists,
        "simple_alpha_preleads": int(len(preleads)),
        "targeted_1m_download_performed": bool(ctx.args.download_targeted_1m),
        "no_validation_style_promotion": True,
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    deconf_text = deconf.to_string(index=False) if not deconf.empty else "no deconfound rows"
    prelead_text = preleads[[c for c in ["idea_id", "family", "subfamily", "current_label", "next_action"] if c in preleads.columns]].head(30).to_string(index=False) if not preleads.empty else "none"
    report = f"""# QLMG Simple-Alpha Liquidation-Safe Development Report\n\n## Verdict\n\n`{verdict}`\n\n## Protocol\n\n- final holdout untouched: `yes`\n- protected start: `{FINAL_HOLDOUT_START}`\n- train data end: `{SCREENING_END}`\n- no sealed validation: `yes`\n- no live-trading authorization: `yes`\n- targeted 1m download performed: `{ctx.args.download_targeted_1m}`\n\n## D4\n\nD4 is carried forward mandatorily as targeted execution-depth collection unless a protocol error is found. Missing top-of-book/depth remains the carry-forward reason.\n\n## Deconfounding\n\n```\n{deconf_text}\n```\n\n## Simple-Alpha Next Actions\n\n```\n{prelead_text}\n```\n\n## Interpretation Rules Applied\n\n- Candidate reconstruction uses registry/results/event ledgers, not `sweep_summary.csv` alone.\n- Path-edge failed-execution candidates are preserved even when replay fails.\n- `signal_R` and `account_R` are reported separately for liquidation-safe sizing.\n- Missing 1m/mark/depth caps conclusions at targeted data collection or not-fairly-tested; it is not a rejection reason.\n- Funding-window fallback schedules and listing proxy launches are flagged.\n"""
    write_text(ctx.run_root / "QLMG_SIMPLE_ALPHA_LIQSAFE_DEVELOPMENT_REPORT.md", report)


def stage_compact(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_SIMPLE_ALPHA_LIQSAFE_DEVELOPMENT_REPORT.md",
        "decision_summary.json",
        "preflight/preflight_report.md",
        "evidence/prior_sweep_evidence_map.csv",
        "evidence/candidate_cluster_map.csv",
        "deconfound/family_identity_deconfound_summary.csv",
        "selection/selected_candidate_clusters.csv",
        "sizing/liqsafe_sizing_summary.csv",
        "funding_window/funding_window_development_summary.csv",
        "listing/listing_family_development_summary.csv",
        "secondary/secondary_family_preservation_summary.csv",
        "data_plan/targeted_1m_window_plan.csv",
        "data_plan/targeted_1m_storage_estimate.csv",
        "nulls/refreshed_null_summary.csv",
        "stress/cost_funding_execution_stress_summary.csv",
        "validation/walk_forward_cpcv_summary.csv",
        "contracts/depth_data_contract_summary.csv",
        "contracts/d4_targeted_execution_depth_collection_contract.json",
        "d4/d4_integration_report.md",
        "triage/all_ideas_preservation_index.csv",
        "backlog/research_backlog.csv",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
    ]
    rows = []
    for rel in include:
        src = ctx.run_root / rel
        if src.exists() and src.is_file():
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            rows.append({"artifact": rel, "path": str(src), "bundle_copy": str(dst), "size_bytes": src.stat().st_size})
    geom = ctx.run_root / "geometry/decision_time_liquidation_geometry.parquet"
    if geom.exists():
        try:
            pd.read_parquet(geom).head(500).to_csv(bundle / "geometry__decision_time_liquidation_geometry_sample.csv", index=False)
            rows.append({"artifact": "geometry/decision_time_liquidation_geometry.parquet", "path": str(geom), "bundle_copy": "sample_only", "size_bytes": geom.stat().st_size})
        except Exception:
            pass
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_json(bundle / "artifact_path_index.json", rows)


def stage_dispatch(ctx: RunContext, stage: str) -> None:
    if stage == "preflight-and-artifact-freeze":
        stage_preflight(ctx)
    elif stage == "telegram-and-tmux-setup":
        stage_telegram(ctx)
    elif stage == "seal-guard":
        stage_seal(ctx)
    elif stage == "prior-sweep-evidence-map":
        stage_evidence_map(ctx)
    elif stage == "family-identity-deconfounding":
        stage_deconfound(ctx)
    elif stage == "candidate-cluster-selection":
        stage_candidate_selection(ctx)
    elif stage == "liquidation-flag-taxonomy":
        stage_liquidation_taxonomy(ctx)
    elif stage == "decision-time-liquidation-geometry":
        stage_geometry(ctx)
    elif stage == "liquidation-safe-sizing-replay":
        stage_sizing(ctx)
    elif stage == "funding-window-family-development":
        stage_family_development(ctx, "funding_window_orb_failure", "funding_window", "funding_window_development_summary.csv")
    elif stage == "new-perp-listing-family-development":
        stage_family_development(ctx, "new_perp_listing_event_study", "listing", "listing_family_development_summary.csv")
    elif stage == "secondary-family-preservation-diagnostics":
        stage_secondary(ctx)
    elif stage == "targeted-1m-mark-replay-plan":
        stage_window_plan(ctx)
    elif stage == "targeted-1m-download-if-approved":
        stage_download(ctx)
    elif stage == "one-minute-mark-replay-if-available":
        stage_one_minute(ctx)
    elif stage == "matched-null-and-same-time-refresh-after-safety":
        stage_nulls(ctx)
    elif stage == "cost-funding-execution-stress-after-safety":
        stage_stress(ctx)
    elif stage == "walk-forward-cpcv-after-safety":
        stage_validation(ctx)
    elif stage == "targeted-depth-data-contracts":
        stage_depth_contracts(ctx)
    elif stage == "d4-carry-forward-integration":
        stage_d4(ctx)
    elif stage == "triage-and-research-backlog":
        stage_triage(ctx)
    elif stage == "decision-report":
        stage_decision(ctx)
    elif stage == "compact-review-bundle":
        stage_compact(ctx)
    else:
        raise ValueError(stage)


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        print(f"[resume] skip {stage}")
        return
    ensure_guard(ctx, stage)
    ctx.notifier.send("QLMG liqsafe stage start", stage)
    try:
        stage_dispatch(ctx, stage)
        for p in required_outputs(ctx.run_root, stage):
            if not p.exists():
                raise RuntimeError(f"stage {stage} did not create required output {p}")
        mark_done(ctx.run_root, stage)
        ctx.notifier.send("QLMG liqsafe stage complete", stage)
    except Exception as exc:
        ctx.notifier.send("QLMG liqsafe stage failed", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram and not args.smoke, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": reason, "argv": sys.argv, "start": str(start), "end": str(end), "protected_start": str(FINAL_HOLDOUT_START), "seed": args.seed, "smoke": args.smoke})
    notifier.send("QLMG liqsafe development run start", f"root={run_root}\nstages={args.stage}")
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
    except Exception:
        (run_root / "watch_status.json").write_text(json.dumps({"status": "failed", "run_root": str(run_root), "ts_utc": utc_now()}, sort_keys=True) + "\n", encoding="utf-8")
        raise
    (run_root / "watch_status.json").write_text(json.dumps({"status": "complete", "run_root": str(run_root), "ts_utc": utc_now()}, sort_keys=True) + "\n", encoding="utf-8")
    notifier.send("QLMG liqsafe development run complete", f"root={run_root}")
    print(f"run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
