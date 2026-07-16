#!/usr/bin/env python3
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
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, stable_hash, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402
from tools.run_qlmg_alpha_discovery_marathon import apply_candidate_filter, cost_bps_for_tier, surface_return_r, summarize_returns  # noqa: E402
from tools.run_qlmg_targeted_1m_data_pilot import (  # noqa: E402
    DATASETS,
    OPTIONAL_DATASETS,
    fetch_funding,
    fetch_kline_dataset,
    fetch_open_interest,
)

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_d4_liquidation_execution_audit_20260625_v1"
PRIOR_ROOT = RESULTS_ROOT / "phase_qlmg_alpha_discovery_marathon_20260625_v1_20260625_145339"
PATH_DIAG_ROOT = RESULTS_ROOT / "phase_qlmg_path_diagnostics_exit_surface_20260624_v1_20260624_121522"
TARGETED_1M_ROOT = RESULTS_ROOT / "phase_qlmg_targeted_1m_data_pilot_20260624_v1"
CANDIDATE_ID = "D4__b4c9487fe82c"
DEFAULT_SEED = 20260625

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "d4-contract-and-event-reconstruction",
    "liquidation-flag-taxonomy",
    "mark-path-availability-audit",
    "targeted-1m-window-plan",
    "targeted-1m-download-if-approved",
    "one-minute-mark-replay",
    "stop-before-liquidation-ordering-audit",
    "leverage-and-margin-sensitivity",
    "liquidation-buffer-filter-study",
    "matched-null-refresh",
    "cost-funding-execution-stress-refresh",
    "aggressive-10x-risk-expression-study",
    "decision-report",
    "compact-review-bundle",
    "all",
)

ALLOWED_VERDICTS = {
    "continue_to_targeted_execution_data_collection",
    "promote_to_family_specific_validation_after_data",
    "reject_d4_current_expression",
    "blocked_by_mark_data",
    "blocked_by_execution_depth_data",
    "blocked_by_protocol_issue",
}

HORIZON = "2h"
TARGET_R = 1.0
STOP_MULT = 2.0
COST_MULT = 1.0
RECON_TOL = 1e-6


@dataclass
class RunContext:
    args: argparse.Namespace
    run_root: Path
    notifier: "RunNotifier"
    start: pd.Timestamp
    end: pd.Timestamp


class RunNotifier:
    def __init__(self, run_root: Path, disabled: bool = False) -> None:
        self.run_root = run_root
        self.disabled = disabled
        self.events_path = run_root / "notifications/telegram_events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.notifier = None
        self.status = "disabled"
        self.missing = "disabled_by_cli" if disabled else ""
        if not disabled and TelegramNotifier is not None:
            class _Args:
                tg_bot_token = ""
                tg_chat_id = ""
                tg_auto_chat = False
            try:
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-d4-liq-audit")
                self.status = self.notifier.status_line()
                if "disabled" in self.status.lower():
                    self.missing = self.status
            except Exception as exc:
                self.status = f"disabled: {type(exc).__name__}: {exc}"
                self.missing = self.status
        elif not disabled:
            self.missing = "tools.telegram_notify.TelegramNotifier unavailable"

    @property
    def remote_available(self) -> bool:
        return (not self.disabled) and self.notifier is not None and "enabled" in self.status.lower()

    def send(self, title: str, body: str = "", *, level: str = "info") -> bool:
        sent = False
        if not self.disabled and self.notifier is not None:
            try:
                sent = bool(self.notifier.send(title, body))
            except Exception:
                sent = False
        rec = {"ts_utc": utc_now(), "title": title, "body": body, "level": level, "sent": sent, "status": self.status}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        try:
            watch = {"ts_utc": rec["ts_utc"], "status": "running", "last_event": title, "last_body": body, "run_root": str(self.run_root)}
            (self.run_root / "watch_status.json").write_text(json.dumps(watch, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        return sent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG D4 liquidation and execution audit")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=30.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--download-targeted-1m", action="store_true")
    p.add_argument("--targeted-download-cap-gb", type=float, default=10.0)
    p.add_argument("--tmux-session-name", default="qlmg_d4_liq_audit")
    p.add_argument("--run-root", default="")
    return p.parse_args()


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


def resolve_run_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.run_root:
        p = Path(args.run_root)
        return (p if p.is_absolute() else REPO / p).resolve(), "explicit_run_root"
    base = (RESULTS_ROOT / DEFAULT_RUN_ID).resolve()
    if not base.exists():
        return base, "default_root_available"
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return base.with_name(f"{base.name}_{suffix}"), f"default_root_existed_suffix_{suffix}"


def clamp_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(args.start, utc=True)
    requested_end = pd.to_datetime(args.end, utc=True) if args.end else SCREENING_END
    end = min(pd.Timestamp(requested_end), SCREENING_END)
    if start >= FINAL_HOLDOUT_START or end >= FINAL_HOLDOUT_START:
        raise RuntimeError("requested window overlaps protected QLMG final holdout")
    return start, end


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    if fieldnames is None:
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
    else:
        keys = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


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


def done_path(run_root: Path, stage: str) -> Path:
    return run_root / "stage_status" / f"{stage}.done"


def mark_done(run_root: Path, stage: str) -> None:
    done_path(run_root, stage).parent.mkdir(parents=True, exist_ok=True)
    done_path(run_root, stage).write_text(utc_now() + "\n", encoding="utf-8")


def append_command(run_root: Path, stage: str) -> None:
    path = run_root / "command_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": utc_now(), "stage": stage, "argv": sys.argv}, sort_keys=True) + "\n")


def required_outputs_for_stage(run_root: Path, stage: str) -> list[Path]:
    m = {
        "preflight-and-artifact-freeze": [run_root / "preflight/preflight_report.md", run_root / "preflight/frozen_artifact_hashes.json", run_root / "preflight/resource_guard_report.md"],
        "telegram-and-tmux-setup": [run_root / "notifications/telegram_readiness_report.md", run_root / "tmux/watch_commands.md"],
        "seal-guard": [run_root / "seal/seal_guard_report.md", run_root / "seal/protected_slice_check.json"],
        "d4-contract-and-event-reconstruction": [run_root / "d4_reconstruction/d4_event_ledger.parquet", run_root / "d4_reconstruction/d4_reconstruction_report.md", run_root / "d4_reconstruction/d4_reconstruction_metrics.csv"],
        "liquidation-flag-taxonomy": [run_root / "liquidation/liquidation_taxonomy_summary.csv", run_root / "liquidation/liquidation_taxonomy_report.md"],
        "mark-path-availability-audit": [run_root / "mark_path/mark_path_availability_summary.csv", run_root / "mark_path/mark_path_availability_report.md"],
        "targeted-1m-window-plan": [run_root / "targeted_1m/deduped_windows.csv", run_root / "targeted_1m/storage_estimate.csv", run_root / "targeted_1m/targeted_1m_window_plan.md"],
        "targeted-1m-download-if-approved": [run_root / "downloaded_1m/download_manifest.csv", run_root / "downloaded_1m/download_report.md"],
        "one-minute-mark-replay": [run_root / "one_minute_mark/one_minute_mark_replay_summary.csv", run_root / "one_minute_mark/one_minute_mark_replay_report.md"],
        "stop-before-liquidation-ordering-audit": [run_root / "ordering/stop_liquidation_ordering_summary.csv", run_root / "ordering/stop_liquidation_ordering_report.md"],
        "leverage-and-margin-sensitivity": [run_root / "leverage/leverage_margin_sensitivity_summary.csv", run_root / "leverage/leverage_margin_sensitivity_report.md"],
        "liquidation-buffer-filter-study": [run_root / "filters/liquidation_buffer_filter_summary.csv", run_root / "filters/liquidation_buffer_filter_report.md"],
        "matched-null-refresh": [run_root / "matched_null/d4_refreshed_matched_null_summary.csv", run_root / "matched_null/d4_refreshed_matched_null_report.md"],
        "cost-funding-execution-stress-refresh": [run_root / "stress/d4_cost_funding_execution_stress_summary.csv", run_root / "stress/d4_cost_funding_execution_stress_report.md"],
        "aggressive-10x-risk-expression-study": [run_root / "portfolio/d4_aggressive_10x_risk_expression_summary.csv", run_root / "portfolio/d4_aggressive_10x_risk_expression_report.md"],
        "decision-report": [run_root / "D4_LIQUIDATION_EXECUTION_AUDIT_REPORT.md", run_root / "decision_summary.json"],
        "compact-review-bundle": [run_root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return m.get(stage, [])


def stage_complete(run_root: Path, stage: str) -> bool:
    return done_path(run_root, stage).exists() and all(p.exists() for p in required_outputs_for_stage(run_root, stage))


def estimate_stage_gb(stage: str, ctx: RunContext) -> float:
    if ctx.args.smoke:
        return 0.25
    if stage == "targeted-1m-download-if-approved" and ctx.args.download_targeted_1m:
        return min(float(ctx.args.targeted_download_cap_gb), 20.0)
    if stage in {"d4-contract-and-event-reconstruction", "matched-null-refresh"}:
        return 1.0
    return 0.25


def ensure_guard(ctx: RunContext, stage: str, estimate_gb: float) -> None:
    snap = resource_snapshot(REPO)
    status = check_resource_guard(
        snap,
        estimated_output_gb=estimate_gb,
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=20.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", {"stage": stage, **status, "snapshot": snap.__dict__})
    if status["warnings"]:
        ctx.notifier.send("D4 AUDIT RESOURCE WARNING", f"stage={stage}\n{status}", level="warning")
    if status["status"] != "pass":
        ctx.notifier.send("D4 AUDIT RESOURCE HARD STOP", f"stage={stage}\n{status}", level="error")
        raise RuntimeError(f"resource guard failed for {stage}: {status['reasons']}")


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def parquet_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetDataset(path).schema.names)
    except Exception:
        return []


def load_prior_candidate() -> dict[str, Any]:
    reg = read_csv_safe(PRIOR_ROOT / "sweep/candidate_registry.csv")
    if reg.empty:
        raise FileNotFoundError("missing marathon candidate registry")
    row = reg[reg["candidate_id"].astype(str).eq(CANDIDATE_ID)]
    if row.empty:
        raise RuntimeError(f"missing candidate {CANDIDATE_ID} in registry")
    return row.iloc[0].to_dict()


def load_prior_prelead_row() -> dict[str, Any]:
    df = read_csv_safe(PRIOR_ROOT / "preleads/full_coverage_prelead_summary.csv")
    row = df[df["candidate_id"].astype(str).eq(CANDIDATE_ID)] if not df.empty else pd.DataFrame()
    if row.empty:
        raise RuntimeError(f"missing candidate {CANDIDATE_ID} in full coverage prelead summary")
    return row.iloc[0].to_dict()


def load_event_regime_for_window(ctx: RunContext) -> pd.DataFrame:
    event_path = PRIOR_ROOT / "events/discovery_event_ledger.parquet"
    regime_path = PRIOR_ROOT / "regime/regime_feature_panel.parquet"
    if not event_path.exists() or not regime_path.exists():
        raise FileNotFoundError("missing marathon event or regime parquet")
    cols = parquet_columns(event_path)
    keep = [c for c in cols if c in {
        "event_id", "family", "variant_id", "symbol", "side", "liquidity_tier", "decision_ts", "entry_ts", "entry_ref_price",
        "reference_risk_bps", "atr_bps", "btc_eth_regime", "oi_chg_24h", "funding_rate", "turnover", "range_pct",
        "data_quality_flags", "mark_path_status", "liq_price_10x", "source_run_root", "discovery_scope",
        "2h_path_available", "2h_mfe_bps", "2h_mae_bps", "2h_close_return_bps", "2h_pos1R_before_neg1R", "2h_liquidation_10x",
        "30m_path_available", "30m_mfe_bps", "30m_mae_bps", "30m_close_return_bps", "30m_liquidation_10x",
        "1h_path_available", "1h_mfe_bps", "1h_mae_bps", "1h_close_return_bps", "1h_liquidation_10x",
        "4h_path_available", "4h_mfe_bps", "4h_mae_bps", "4h_close_return_bps", "4h_liquidation_10x",
        "24h_path_available", "24h_mfe_bps", "24h_mae_bps", "24h_close_return_bps", "24h_liquidation_10x",
    }]
    events = pd.read_parquet(event_path, columns=keep)
    events["decision_ts"] = pd.to_datetime(events["decision_ts"], utc=True, errors="coerce")
    events["entry_ts"] = pd.to_datetime(events["entry_ts"], utc=True, errors="coerce")
    events = events[(events["decision_ts"] >= ctx.start) & (events["decision_ts"] <= ctx.end)].copy()
    if ctx.args.max_symbols:
        syms = sorted(events["symbol"].astype(str).unique())[: ctx.args.max_symbols]
        events = events[events["symbol"].isin(syms)].copy()
    if ctx.args.smoke:
        events = events.sort_values(["family", "symbol", "decision_ts"]).groupby("family", group_keys=False).head(4000)
    reg_cols_all = parquet_columns(regime_path)
    reg_keep = [c for c in reg_cols_all if c in {
        "event_id", "feature_ts", "parent_trend_label", "btc_eth_non_deteriorating", "btc_eth_regime_label", "liquidity_quality_label",
        "bad_wick_proxy_label", "price_oi_matrix_24h", "funding_percentile_bucket", "funding_sign_label", "deleveraged_2of4",
        "deleveraged_3of4", "liquidation_proxy_label", "turnover_bucket", "realized_vol_bucket", "session_bucket", "listing_age_bucket",
        "data_integrity_label", "regime_row_hash",
    }]
    reg = pd.read_parquet(regime_path, columns=reg_keep)
    out = events.merge(reg, on="event_id", how="left", suffixes=("", "_regime"))
    validate_no_protected(out, ["decision_ts", "entry_ts", "feature_ts"])
    return out.reset_index(drop=True)


def reconstruct_d4(ctx: RunContext) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    cand = load_prior_candidate()
    prior = load_prior_prelead_row()
    all_rows = load_event_regime_for_window(ctx)
    sub = apply_candidate_filter(all_rows, cand)
    ret = surface_return_r(sub, HORIZON, TARGET_R, STOP_MULT, COST_MULT)
    sub = sub.loc[ret.index].copy()
    sub["candidate_id"] = CANDIDATE_ID
    sub["net_R"] = ret.values
    sub["gross_R_proxy"] = pd.to_numeric(sub.get(f"{HORIZON}_close_return_bps"), errors="coerce").fillna(0.0) / pd.to_numeric(sub.get("reference_risk_bps"), errors="coerce").fillna(100.0).clip(lower=1.0)
    sub["liquidation_flag_proxy_10x"] = sub.get(f"{HORIZON}_liquidation_10x", pd.Series(False, index=sub.index)).fillna(False).astype(bool)
    sm = summarize_returns(ret)
    sm["liquidation_count"] = int(sub["liquidation_flag_proxy_10x"].sum())
    sm["candidate_id"] = CANDIDATE_ID
    sm["prior_events"] = int(prior.get("events", -1))
    sm["prior_net_R"] = float(prior.get("net_R", np.nan))
    sm["prior_PF"] = float(prior.get("PF", np.nan))
    sm["prior_liquidation_count"] = int(prior.get("liquidation_count", -1))
    if not ctx.args.smoke and ctx.start <= pd.Timestamp("2023-01-01T00:00:00Z") and ctx.end >= SCREENING_END:
        compare_reconstruction_metrics(prior, sm)
        sm["reconstruction_status"] = "exact_match_within_tolerance"
    else:
        sm["reconstruction_status"] = "subset_or_smoke_not_full_exact_reconstruction"
    return sub.reset_index(drop=True), sm, prior


def compare_reconstruction_metrics(prior: Mapping[str, Any], recon: Mapping[str, Any], tol: float = RECON_TOL) -> None:
    expected_events = int(prior.get("events", -1))
    actual_events = int(recon.get("events", -2))
    if expected_events != actual_events:
        raise RuntimeError(f"D4 reconstruction event count mismatch expected={expected_events} actual={actual_events}")
    for key in ["net_R", "PF"]:
        a = float(prior.get(key, np.nan))
        b = float(recon.get(key, np.nan))
        if not (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= max(tol, abs(a) * 1e-9)):
            raise RuntimeError(f"D4 reconstruction {key} mismatch expected={a} actual={b}")
    if int(prior.get("liquidation_count", -1)) != int(recon.get("liquidation_count", -2)):
        raise RuntimeError("D4 reconstruction liquidation count mismatch")


def classify_liquidation_row(row: Mapping[str, Any], *, has_one_minute_mark: bool = False, mark_liquidated: bool | None = None, stop_before_liq: bool | None = None) -> str:
    proxy_liq = bool(row.get("liquidation_flag_proxy_10x", row.get(f"{HORIZON}_liquidation_10x", False)))
    if not proxy_liq and mark_liquidated is not True:
        return "no_liquidation_flag"
    if stop_before_liq is True:
        return "stop_would_trigger_before_liquidation"
    if stop_before_liq is False and mark_liquidated is True:
        return "liquidation_before_stop"
    if has_one_minute_mark:
        if mark_liquidated is True:
            return "one_minute_mark_liquidation"
        if mark_liquidated is False:
            return "leverage_assumption_only"
    status = str(row.get("mark_path_status", "")).lower()
    if "mark" in status and ("ok" in status or "available" in status):
        return "five_minute_mark_liquidation"
    if "last_price_proxy" in status:
        return "last_price_proxy_liquidation"
    if not status or "missing" in status or "proxy" in status:
        return "missing_mark_proxy_only"
    return "unknown_unresolved"


def liquidation_adverse_bps_for_leverage(leverage: float, maintenance_margin_fraction: float = 0.005) -> float:
    adverse = max((1.0 / float(leverage)) - float(maintenance_margin_fraction), 0.0)
    return adverse * 10000.0


def stop_distance_bps(df: pd.DataFrame, stop_mult: float = STOP_MULT) -> pd.Series:
    return pd.to_numeric(df.get("reference_risk_bps"), errors="coerce").fillna(100.0).clip(lower=1.0) * float(stop_mult)


def source_artifacts() -> list[Path]:
    return [
        PRIOR_ROOT / "next_contracts/contracts/D4__b4c9487fe82c.json",
        PRIOR_ROOT / "sweep/candidate_registry.csv",
        PRIOR_ROOT / "preleads/full_coverage_prelead_summary.csv",
        PRIOR_ROOT / "events/discovery_event_ledger.parquet",
        PRIOR_ROOT / "regime/regime_feature_panel.parquet",
        PRIOR_ROOT / "matched_null/prelead_matched_null_summary.csv",
        PRIOR_ROOT / "validation/cpcv_summary.csv",
        PRIOR_ROOT / "stress/execution_cost_liquidation_stress_summary.csv",
        PRIOR_ROOT / "portfolio/aggressive_10x_portfolio_summary.csv",
        PRIOR_ROOT / "one_minute/prelead_1m_overlay_summary.csv",
        PATH_DIAG_ROOT / "matched_null/matched_null_path_metrics.parquet",
    ]


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    rows = []
    for p in source_artifacts():
        rows.append({"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0, "sha256_first_100mb": sha256_file(p, 100 * 1024 * 1024) if p.exists() and p.is_file() else ""})
    missing = [r["path"] for r in rows if not r["exists"]]
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", {"candidate_id": CANDIDATE_ID, "source_run_root": str(PRIOR_ROOT), "artifacts": rows, "missing": missing})
    prior = load_prior_prelead_row()
    one = read_csv_safe(PRIOR_ROOT / "one_minute/prelead_1m_overlay_summary.csv")
    one_row = one[one["candidate_id"].astype(str).eq(CANDIDATE_ID)].iloc[0].to_dict() if not one.empty and (one["candidate_id"].astype(str).eq(CANDIDATE_ID)).any() else {}
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard Report\n\n- free disk GB: `{snap.free_gb:.2f}`\n- hard stop free GB: `5`\n- warning free GB: `7`\n- stage output hard stop GB: `20`\n- default max output GB: `{ctx.args.max_output_gb}`\n- targeted download cap GB: `{ctx.args.targeted_download_cap_gb}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight And Artifact Freeze\n\n- candidate: `{CANDIDATE_ID}`\n- prior run root: `{PRIOR_ROOT}`\n- protected holdout starts: `{FINAL_HOLDOUT_START}`\n- requested data window: `{ctx.start}` to `{ctx.end}`\n- prior D4 events: `{prior.get('events')}`\n- prior D4 net_R: `{prior.get('net_R')}`\n- prior D4 PF: `{prior.get('PF')}`\n- prior D4 liquidation_count: `{prior.get('liquidation_count')}`\n- prior proxy mark/liquidation evidence share: `{prior.get('proxy_mark_or_liquidation_evidence_share')}`\n- prior 1m overlay rows: `{one_row.get('pilot_1m_overlap_rows', 'unknown')}`\n- free disk GB: `{snap.free_gb:.2f}`\n- missing required artifacts: `{missing}`\n\nIf missing artifacts are non-empty, downstream exact reconstruction must fail closed.\n")
    if missing:
        raise RuntimeError(f"missing required D4 source artifacts: {missing}")


def stage_telegram(ctx: RunContext) -> None:
    if ctx.args.require_telegram and not ctx.notifier.remote_available and not ctx.args.allow_no_telegram and not ctx.args.smoke:
        raise RuntimeError("remote Telegram is required but unavailable; pass --allow-no-telegram only for explicit local-only runs")
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness Report\n\n- status: `{ctx.notifier.status}`\n- remote_available: `{ctx.notifier.remote_available}`\n- require_telegram: `{ctx.args.require_telegram}`\n- allow_no_telegram: `{ctx.args.allow_no_telegram}`\n- missing/disabled reason: `{ctx.notifier.missing or 'none'}`\n- local log: `notifications/telegram_events.jsonl`\n- secrets persisted: `false`\n")
    watch = f"""# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nNo-download audit:\n\n```bash\nbash tools/run_qlmg_d4_liquidation_execution_audit_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --nulls-per-event {ctx.args.nulls_per_event} --seed {ctx.args.seed}\n```\n\nTargeted-download audit:\n\n```bash\nbash tools/run_qlmg_d4_liquidation_execution_audit_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --nulls-per-event {ctx.args.nulls_per_event} --download-targeted-1m --targeted-download-cap-gb {ctx.args.targeted_download_cap_gb} --seed {ctx.args.seed}\n```\n")
    ctx.notifier.send("D4 LIQUIDATION AUDIT START", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    check = {
        "protected_start": str(FINAL_HOLDOUT_START),
        "allowed_end_inclusive": str(SCREENING_END),
        "requested_start": str(ctx.start),
        "requested_end": str(ctx.end),
        "pre_holdout_read_smoke": bool(ctx.end < FINAL_HOLDOUT_START),
        "protected_read_smoke": "blocked_by_policy",
        "status": "pass" if ctx.end < FINAL_HOLDOUT_START else "fail",
    }
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    if check["status"] != "pass":
        raise RuntimeError("seal guard failed")
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard Report\n\n- protected slice: `{FINAL_HOLDOUT_START}` onward\n- screening/data-acquisition cutoff: `{SCREENING_END}`\n- requested end: `{ctx.end}`\n- protected read smoke: `blocked_by_policy`\n- status: `pass`\n")


def stage_reconstruct(ctx: RunContext) -> None:
    d4, metrics, prior = reconstruct_d4(ctx)
    out_dir = ctx.run_root / "d4_reconstruction"
    out_dir.mkdir(parents=True, exist_ok=True)
    d4.to_parquet(out_dir / "d4_event_ledger.parquet", index=False, compression="zstd")
    trade_cols = [c for c in ["candidate_id", "event_id", "family", "symbol", "side", "liquidity_tier", "decision_ts", "entry_ts", "entry_ref_price", "reference_risk_bps", "net_R", "gross_R_proxy", "liquidation_flag_proxy_10x", "mark_path_status", "data_quality_flags"] if c in d4.columns]
    d4[trade_cols].to_parquet(out_dir / "d4_trade_ledger.parquet", index=False, compression="zstd")
    write_csv(out_dir / "d4_reconstruction_metrics.csv", [metrics])
    write_text(out_dir / "d4_reconstruction_report.md", f"# D4 Reconstruction Report\n\n- candidate: `{CANDIDATE_ID}`\n- reconstruction status: `{metrics.get('reconstruction_status')}`\n- reconstructed events: `{metrics.get('events')}`\n- prior events: `{metrics.get('prior_events')}`\n- reconstructed net_R: `{metrics.get('net_R')}`\n- prior net_R: `{metrics.get('prior_net_R')}`\n- reconstructed PF: `{metrics.get('PF')}`\n- prior PF: `{metrics.get('prior_PF')}`\n- reconstructed liquidation_count: `{metrics.get('liquidation_count')}`\n- prior liquidation_count: `{metrics.get('prior_liquidation_count')}`\n- note: smoke/custom date windows are subset audits and do not claim full exact reconstruction.\n")


def load_d4(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "d4_reconstruction/d4_event_ledger.parquet"
    if not p.exists():
        d4, _, _ = reconstruct_d4(ctx)
        return d4
    return pd.read_parquet(p)


def stage_taxonomy(ctx: RunContext) -> None:
    d4 = load_d4(ctx)
    d4["liquidation_taxonomy"] = [classify_liquidation_row(r) for r in d4.to_dict("records")]
    d4["liquidation_evidence_grade"] = np.where(d4["liquidation_taxonomy"].isin(["one_minute_mark_liquidation", "five_minute_mark_liquidation", "actual_mark_path_liquidation"]), "mark_evidence", np.where(d4["liquidation_taxonomy"].eq("no_liquidation_flag"), "none", "proxy_or_unresolved"))
    out_dir = ctx.run_root / "liquidation"
    out_dir.mkdir(parents=True, exist_ok=True)
    d4[["event_id", "symbol", "decision_ts", "liquidation_flag_proxy_10x", "mark_path_status", "liquidation_taxonomy", "liquidation_evidence_grade"]].to_parquet(out_dir / "liquidation_taxonomy_by_event.parquet", index=False, compression="zstd")
    summ = d4.groupby(["liquidation_taxonomy", "liquidation_evidence_grade"]).size().reset_index(name="events")
    summ.to_csv(out_dir / "liquidation_taxonomy_summary.csv", index=False)
    write_text(out_dir / "liquidation_taxonomy_report.md", "# Liquidation Flag Taxonomy\n\n" + summ.to_markdown(index=False) + "\n\nExisting proxy liquidation flags are not counted as actual mark-path liquidation unless mark evidence is present.\n")


def stage_mark_availability(ctx: RunContext) -> None:
    d4 = load_d4(ctx)
    status = d4.get("mark_path_status", pd.Series("unknown", index=d4.index)).astype(str)
    rows = []
    for k, v in status.value_counts(dropna=False).items():
        rows.append({"mark_path_status": k, "events": int(v), "share": float(v / max(len(d4), 1))})
    mark_available = status.str.contains("mark", case=False, na=False) & status.str.contains("ok|available", case=False, regex=True, na=False)
    rows.append({"mark_path_status": "__any_mark_available__", "events": int(mark_available.sum()), "share": float(mark_available.mean()) if len(d4) else 0.0})
    write_csv(ctx.run_root / "mark_path/mark_path_availability_summary.csv", rows)
    write_text(ctx.run_root / "mark_path/mark_path_availability_report.md", f"# Mark Path Availability Audit\n\n- D4 events checked: `{len(d4)}`\n- mark-available rows: `{int(mark_available.sum())}`\n- mark-available share: `{float(mark_available.mean()) if len(d4) else 0.0:.6f}`\n- conclusion: liquidation evidence remains proxy-grade unless targeted 1m/mark data is downloaded and resolves rows.\n")


def make_windows(ctx: RunContext) -> pd.DataFrame:
    d4 = load_d4(ctx)
    if ctx.args.smoke:
        d4 = d4.head(min(len(d4), 100))
    rows: list[dict[str, Any]] = []
    for _, r in d4.iterrows():
        decision = pd.Timestamp(pd.to_datetime(r["decision_ts"], utc=True))
        exit_ts = decision + pd.Timedelta(hours=2)
        start = decision - pd.Timedelta(hours=4)
        end = exit_ts + pd.Timedelta(hours=24)
        if end >= FINAL_HOLDOUT_START:
            continue
        liq = bool(r.get("liquidation_flag_proxy_10x", False))
        rows.append({
            "window_type": "accepted_d4_event",
            "family": "D4",
            "event_id": r.get("event_id", ""),
            "symbol": r.get("symbol", ""),
            "decision_ts": decision,
            "window_start": start,
            "window_end": end,
            "reason_for_selection": "accepted_d4_trade" + (";liquidation_flagged" if liq else ""),
            "liquidation_flag_proxy_10x": liq,
        })
        if liq:
            rows.append({
                "window_type": "liquidation_flagged_duplicate_priority",
                "family": "D4",
                "event_id": f"liq_{r.get('event_id','')}",
                "source_event_id": r.get("event_id", ""),
                "symbol": r.get("symbol", ""),
                "decision_ts": decision,
                "window_start": start,
                "window_end": end,
                "reason_for_selection": "all_liquidation_flagged_events_priority_subset",
                "liquidation_flag_proxy_10x": True,
            })
        for i in range(max(0, int(ctx.args.nulls_per_event))):
            null = make_null_window(r, i, ctx.args.seed)
            if pd.Timestamp(null["window_end"]) < FINAL_HOLDOUT_START:
                rows.append(null)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["window_id"] = [stable_hash(r, 16) for r in out.to_dict("records")]
        validate_no_protected(out, ["decision_ts", "window_start", "window_end"])
    return out


def make_null_window(row: Mapping[str, Any], ordinal: int, seed: int) -> dict[str, Any]:
    decision = pd.Timestamp(pd.to_datetime(row["decision_ts"], utc=True))
    offsets = [pd.Timedelta(days=7), pd.Timedelta(days=-7), pd.Timedelta(days=14), pd.Timedelta(days=-14), pd.Timedelta(days=21), pd.Timedelta(days=-21)]
    off = offsets[(ordinal + seed) % len(offsets)]
    nd = decision + off
    if nd + pd.Timedelta(hours=26) >= FINAL_HOLDOUT_START:
        nd = decision - abs(off)
    start = nd - pd.Timedelta(hours=4)
    end = nd + pd.Timedelta(hours=26)
    return {
        "window_type": "matched_null_window",
        "family": "D4",
        "event_id": f"null{ordinal}_{row.get('event_id','')}",
        "source_event_id": row.get("event_id", ""),
        "symbol": row.get("symbol", ""),
        "decision_ts": nd,
        "window_start": start,
        "window_end": end,
        "reason_for_selection": "deterministic_same_symbol_offset_matched_null_window",
        "match_level": "same_symbol_offset",
        "liquidation_flag_proxy_10x": False,
    }


def dedupe_windows(df: pd.DataFrame, merge_gap: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)
    out = []
    for sym, sub in df.sort_values(["symbol", "window_start", "window_end"]).groupby("symbol"):
        cur: dict[str, Any] | None = None
        types: set[str] = set()
        event_ids: list[str] = []
        for _, r in sub.iterrows():
            if cur is None:
                cur = {"symbol": sym, "window_start": r["window_start"], "window_end": r["window_end"], "source_window_count": 1}
                types = {str(r.get("window_type", ""))}
                event_ids = [str(r.get("event_id", ""))]
                continue
            if r["window_start"] <= cur["window_end"] + merge_gap:
                cur["window_end"] = max(cur["window_end"], r["window_end"])
                cur["source_window_count"] = int(cur["source_window_count"]) + 1
                types.add(str(r.get("window_type", "")))
                event_ids.append(str(r.get("event_id", "")))
            else:
                cur["window_types"] = ";".join(sorted(types))
                cur["source_event_ids_sample"] = ";".join(event_ids[:25])
                out.append(cur)
                cur = {"symbol": sym, "window_start": r["window_start"], "window_end": r["window_end"], "source_window_count": 1}
                types = {str(r.get("window_type", ""))}
                event_ids = [str(r.get("event_id", ""))]
        if cur is not None:
            cur["window_types"] = ";".join(sorted(types))
            cur["source_event_ids_sample"] = ";".join(event_ids[:25])
            out.append(cur)
    res = pd.DataFrame(out)
    if not res.empty:
        res["window_hours"] = (pd.to_datetime(res["window_end"], utc=True) - pd.to_datetime(res["window_start"], utc=True)).dt.total_seconds() / 3600.0
        res["window_id"] = [stable_hash(r, 16) for r in res.to_dict("records")]
        validate_no_protected(res, ["window_start", "window_end"])
    return res


def estimate_storage_gb(deduped: pd.DataFrame) -> list[dict[str, Any]]:
    total_minutes = float((pd.to_datetime(deduped["window_end"], utc=True) - pd.to_datetime(deduped["window_start"], utc=True)).dt.total_seconds().sum() / 60.0) if not deduped.empty else 0.0
    specs = [("ohlcv_1m", 55), ("mark_1m", 40), ("index_1m", 40), ("premium_1m", 40), ("open_interest_5m", 20), ("funding_history", 8)]
    rows = []
    for dataset, bytes_per_row in specs:
        row_count = total_minutes if dataset not in {"open_interest_5m", "funding_history"} else (total_minutes / 5.0 if dataset == "open_interest_5m" else max(1.0, total_minutes / 480.0) if total_minutes else 0.0)
        b = row_count * bytes_per_row
        rows.append({"dataset": dataset, "estimated_rows": int(row_count), "estimated_compressed_bytes": int(b), "estimated_compressed_gb": b / (1024**3)})
    total = sum(r["estimated_compressed_bytes"] for r in rows)
    rows.append({"dataset": "total_core", "estimated_rows": int(sum(r["estimated_rows"] for r in rows)), "estimated_compressed_bytes": int(total), "estimated_compressed_gb": total / (1024**3)})
    return rows


def stage_window_plan(ctx: RunContext) -> None:
    raw = make_windows(ctx)
    out_dir = ctx.run_root / "targeted_1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "raw_targeted_windows.csv", index=False)
    dedup = dedupe_windows(raw)
    dedup.to_csv(out_dir / "deduped_windows.csv", index=False)
    storage = estimate_storage_gb(dedup)
    write_csv(out_dir / "storage_estimate.csv", storage)
    total_gb = next((float(r["estimated_compressed_gb"]) for r in storage if r["dataset"] == "total_core"), 0.0)
    staged = [
        {"stage": 1, "subset": "liquidation_flagged_events_plus_matched_nulls", "recommended_first": True},
        {"stage": 2, "subset": "all_accepted_D4_events", "recommended_first": False},
        {"stage": 3, "subset": "matched_null_expansion", "recommended_first": False},
        {"stage": 4, "subset": "full_D4_event_universe_only_if_explicitly_approved", "recommended_first": False},
    ]
    write_csv(out_dir / "staged_download_subsets.csv", staged)
    write_text(out_dir / "targeted_1m_window_plan.md", f"# Targeted 1m Window Plan\n\n- raw windows: `{len(raw)}`\n- deduped symbol windows: `{len(dedup)}`\n- estimated core GB: `{total_gb:.4f}`\n- cap GB: `{ctx.args.targeted_download_cap_gb}`\n- fits cap: `{total_gb <= ctx.args.targeted_download_cap_gb}`\n- default action: plan only unless `--download-targeted-1m` is supplied.\n- staged subsets: liquidation-flagged first, accepted events second, null expansion third, full universe only after explicit approval.\n")


def write_partition(df: pd.DataFrame, base: Path, dataset: str, symbol: str, window_id: str) -> Path:
    out_dir = base / dataset / f"symbol={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"window={window_id}.parquet"
    df.to_parquet(p, index=False, compression="zstd")
    return p


def stage_download(ctx: RunContext) -> None:
    out_dir = ctx.run_root / "downloaded_1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["window_id", "symbol", "dataset", "endpoint", "status", "rows", "requests", "path", "error"]
    if not ctx.args.download_targeted_1m:
        write_csv(out_dir / "download_manifest.csv", [], fieldnames=fields)
        write_csv(out_dir / "gaps_and_failures.csv", [], fieldnames=["window_id", "symbol", "dataset", "status", "error"])
        write_text(out_dir / "download_report.md", "# Download Report\n\n- mode: `no_download`\n- downloaded datasets: `none`\n- no external historical data was downloaded.\n")
        return
    dedup = read_csv_safe(ctx.run_root / "targeted_1m/deduped_windows.csv")
    storage = read_csv_safe(ctx.run_root / "targeted_1m/storage_estimate.csv")
    total_gb = float(storage.loc[storage["dataset"].eq("total_core"), "estimated_compressed_gb"].iloc[0]) if not storage.empty and storage["dataset"].eq("total_core").any() else 0.0
    if total_gb > ctx.args.targeted_download_cap_gb and not ctx.args.allow_large_output:
        write_csv(out_dir / "download_manifest.csv", [], fieldnames=fields)
        write_csv(out_dir / "gaps_and_failures.csv", [{"status": "blocked_storage_budget", "error": f"estimated_gb={total_gb:.4f} cap={ctx.args.targeted_download_cap_gb}"}], fieldnames=["window_id", "symbol", "dataset", "status", "error"])
        write_text(out_dir / "download_report.md", f"# Download Report\n\n- status: `blocked_storage_budget`\n- estimated GB: `{total_gb:.4f}`\n- cap GB: `{ctx.args.targeted_download_cap_gb}`\n")
        return
    if ctx.args.smoke:
        dedup = dedup.head(2).copy()
    session = requests.Session()
    session.headers["User-Agent"] = "qlmg-d4-liquidation-audit/1.0"
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for n, row in enumerate(dedup.to_dict("records"), start=1):
        symbol = str(row["symbol"]).upper()
        window_id = str(row.get("window_id") or stable_hash(row, 16))
        start = pd.Timestamp(pd.to_datetime(row["window_start"], utc=True))
        end = pd.Timestamp(pd.to_datetime(row["window_end"], utc=True))
        if end >= FINAL_HOLDOUT_START:
            failures.append({"window_id": window_id, "symbol": symbol, "dataset": "all", "status": "blocked_protected_window", "error": "protected timestamp"})
            continue
        for dataset in list(DATASETS) + list(OPTIONAL_DATASETS):
            status = "ok"
            err = ""
            rows = 0
            reqs = 0
            out_path = ""
            try:
                if dataset in DATASETS:
                    df, reqs = fetch_kline_dataset(session, dataset, symbol, start, end)
                    out_dataset = DATASETS[dataset]["out_dir"]
                    endpoint = DATASETS[dataset]["endpoint"]
                elif dataset == "open_interest_5m":
                    df, reqs = fetch_open_interest(session, symbol, start, end)
                    out_dataset = OPTIONAL_DATASETS[dataset]["out_dir"]
                    endpoint = OPTIONAL_DATASETS[dataset]["endpoint"]
                else:
                    df, reqs = fetch_funding(session, symbol, start, end)
                    out_dataset = OPTIONAL_DATASETS[dataset]["out_dir"]
                    endpoint = OPTIONAL_DATASETS[dataset]["endpoint"]
                if not df.empty:
                    validate_no_protected(df, ["timestamp"])
                    p = write_partition(df, out_dir, out_dataset, symbol, window_id)
                    rows = len(df)
                    out_path = str(p.relative_to(ctx.run_root))
                else:
                    status = "empty"
            except Exception as exc:
                status = "error"
                endpoint = (DATASETS.get(dataset) or OPTIONAL_DATASETS.get(dataset) or {}).get("endpoint", "")
                err = f"{type(exc).__name__}: {exc}"
                failures.append({"window_id": window_id, "symbol": symbol, "dataset": dataset, "status": status, "error": err})
            manifest.append({"window_id": window_id, "symbol": symbol, "dataset": dataset, "endpoint": endpoint, "status": status, "rows": rows, "requests": reqs, "path": out_path, "error": err})
        if n % max(1, ctx.args.chunk_size) == 0 or n == len(dedup):
            write_csv(out_dir / "download_manifest.csv", manifest, fieldnames=fields)
            write_csv(out_dir / "gaps_and_failures.csv", failures, fieldnames=["window_id", "symbol", "dataset", "status", "error"])
            ctx.notifier.send("D4 TARGETED 1M DOWNLOAD PROGRESS", f"windows_done={n}/{len(dedup)}")
    write_text(out_dir / "download_report.md", f"# Download Report\n\n- mode: `download_targeted_1m`\n- windows attempted: `{len(dedup)}`\n- manifest rows: `{len(manifest)}`\n- failures: `{len(failures)}`\n- storage root: `downloaded_1m/`\n")


def _download_index(ctx: RunContext) -> dict[tuple[str, str], Path]:
    manifest = read_csv_safe(ctx.run_root / "downloaded_1m/download_manifest.csv")
    idx: dict[tuple[str, str], Path] = {}
    if manifest.empty:
        return idx
    ok = manifest[manifest["status"].astype(str).eq("ok")].copy()
    for row in ok.to_dict("records"):
        path = str(row.get("path", ""))
        if not path:
            continue
        p = ctx.run_root / path
        if p.exists():
            idx[(str(row.get("window_id")), str(row.get("dataset")))] = p
    return idx


def _assign_dedup_window_ids(raw: pd.DataFrame, dedup: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["window_start"] = pd.to_datetime(out["window_start"], utc=True)
    out["window_end"] = pd.to_datetime(out["window_end"], utc=True)
    dedup = dedup.copy()
    dedup["window_start"] = pd.to_datetime(dedup["window_start"], utc=True)
    dedup["window_end"] = pd.to_datetime(dedup["window_end"], utc=True)
    out["dedup_window_id"] = ""
    out["dedup_window_match_status"] = "not_matched"
    by_symbol = {s: g.sort_values("window_start") for s, g in dedup.groupby("symbol")}
    for i, row in out.iterrows():
        sym = row.get("symbol")
        g = by_symbol.get(sym)
        if g is None or g.empty:
            continue
        hit = g[(g["window_start"] <= row["window_start"]) & (g["window_end"] >= row["window_end"])]
        if hit.empty:
            continue
        # Choose the tightest containing window when overlaps exist.
        h = hit.assign(_span=(hit["window_end"] - hit["window_start"]).dt.total_seconds()).sort_values("_span").iloc[0]
        out.at[i, "dedup_window_id"] = str(h["window_id"])
        out.at[i, "dedup_window_match_status"] = "matched_containing_dedup_window"
    return out


def _full_minute_coverage(df: pd.DataFrame, start_exclusive: pd.Timestamp, end_inclusive: pd.Timestamp) -> tuple[bool, str, pd.DataFrame]:
    if df.empty or "timestamp" not in df.columns:
        return False, "empty", df
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    path = data[(data["timestamp"] > start_exclusive) & (data["timestamp"] <= end_inclusive)].copy()
    if path.empty:
        return False, "no_rows_in_horizon", path
    expected = int((end_inclusive.floor("min") - start_exclusive.floor("min")).total_seconds() // 60)
    if len(path) < max(1, expected - 2):
        return False, f"incomplete_rows_{len(path)}_expected_{expected}", path
    diffs = path["timestamp"].diff().dropna().dt.total_seconds()
    if not diffs.empty and diffs.max() > 120:
        return False, f"minute_gap_gt_120s_{diffs.max()}", path
    if path["timestamp"].min() > start_exclusive + pd.Timedelta(minutes=2):
        return False, "late_first_row", path
    if path["timestamp"].max() < end_inclusive - pd.Timedelta(minutes=2):
        return False, "early_last_row", path
    return True, "full_coverage", path


def _entry_open_from_ohlcv(ohlcv: pd.DataFrame, entry_ts: pd.Timestamp) -> tuple[float | None, str]:
    if ohlcv.empty:
        return None, "missing_ohlcv"
    df = ohlcv.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    hit = df[df["timestamp"] >= entry_ts]
    if hit.empty:
        return None, "missing_entry_open"
    val = pd.to_numeric(hit.iloc[0].get("open"), errors="coerce")
    if pd.isna(val) or float(val) <= 0:
        return None, "nonpositive_entry_open"
    return float(val), "entry_open_from_1m_ohlcv"


def _side_price_levels(side: str, entry_price: float, risk_bps: float, liq_price: float | None = None) -> dict[str, float]:
    stop_bps = float(risk_bps) * STOP_MULT
    target_bps = stop_bps * TARGET_R
    if side == "short":
        stop_price = entry_price * (1.0 + stop_bps / 10000.0)
        target_price = entry_price * (1.0 - target_bps / 10000.0)
        liq = liq_price if liq_price and math.isfinite(float(liq_price)) else entry_price * (1.0 + liquidation_adverse_bps_for_leverage(10.0) / 10000.0)
    else:
        stop_price = entry_price * (1.0 - stop_bps / 10000.0)
        target_price = entry_price * (1.0 + target_bps / 10000.0)
        liq = liq_price if liq_price and math.isfinite(float(liq_price)) else entry_price * (1.0 - liquidation_adverse_bps_for_leverage(10.0) / 10000.0)
    return {"stop_bps": stop_bps, "target_bps": target_bps, "stop_price": float(stop_price), "target_price": float(target_price), "liq_price": float(liq)}


def _replay_single_window(
    *,
    window_row: Mapping[str, Any],
    source_event: Mapping[str, Any],
    ohlcv: pd.DataFrame,
    mark: pd.DataFrame,
) -> dict[str, Any]:
    window_type = str(window_row.get("window_type", ""))
    is_null = window_type == "matched_null_window"
    source_decision = pd.Timestamp(pd.to_datetime(source_event.get("decision_ts"), utc=True))
    source_entry = pd.Timestamp(pd.to_datetime(source_event.get("entry_ts"), utc=True))
    decision_ts = pd.Timestamp(pd.to_datetime(window_row.get("decision_ts"), utc=True)) if is_null else source_decision
    entry_offset = source_entry - source_decision
    entry_ts = decision_ts + entry_offset
    horizon_end = entry_ts + pd.Timedelta(hours=2)
    side = str(source_event.get("side", "long")).lower()
    risk_bps = float(source_event.get("reference_risk_bps", np.nan))
    if not math.isfinite(risk_bps) or risk_bps <= 0:
        return {"mark_replay_status": "fail_closed_invalid_risk_bps"}
    if is_null:
        entry_price, entry_source = _entry_open_from_ohlcv(ohlcv, entry_ts)
        if entry_price is None:
            return {"mark_replay_status": f"fail_closed_{entry_source}", "entry_source": entry_source}
        liq_price = None
    else:
        entry_price = float(source_event.get("entry_ref_price", np.nan))
        entry_source = "source_event_entry_ref_price"
        liq_price = source_event.get("liq_price_10x", np.nan)
        if not math.isfinite(entry_price) or entry_price <= 0:
            return {"mark_replay_status": "fail_closed_invalid_entry_price", "entry_source": entry_source}
    ohlcv_ok, ohlcv_status, ohlcv_path = _full_minute_coverage(ohlcv, entry_ts, horizon_end)
    mark_ok, mark_status, mark_path = _full_minute_coverage(mark, entry_ts, horizon_end)
    if not ohlcv_ok or not mark_ok:
        return {
            "mark_replay_status": "fail_closed_missing_or_incomplete_1m_path",
            "ohlcv_coverage_status": ohlcv_status,
            "mark_coverage_status": mark_status,
            "entry_ts": entry_ts,
            "horizon_end": horizon_end,
            "entry_source": entry_source,
        }
    levels = _side_price_levels(side, entry_price, risk_bps, None if is_null else float(liq_price) if pd.notna(liq_price) else None)
    o = ohlcv_path[["timestamp", "open", "high", "low", "close"]].rename(columns={"high": "last_high", "low": "last_low", "close": "last_close"})
    m = mark_path[["timestamp", "high", "low", "close"]].rename(columns={"high": "mark_high", "low": "mark_low", "close": "mark_close"})
    path = o.merge(m, on="timestamp", how="inner").sort_values("timestamp")
    if len(path) < min(len(ohlcv_path), len(mark_path)) - 2:
        return {"mark_replay_status": "fail_closed_ohlcv_mark_timestamp_mismatch", "entry_source": entry_source}
    exit_ts = path.iloc[-1]["timestamp"]
    exit_price = float(path.iloc[-1]["last_close"])
    exit_reason = "time_exit"
    ordering = "no_liquidation"
    ambiguity = False
    liq_hit_any = False
    stop_hit_any = False
    for _, bar in path.iterrows():
        last_high = float(bar["last_high"])
        last_low = float(bar["last_low"])
        mark_high = float(bar["mark_high"])
        mark_low = float(bar["mark_low"])
        if side == "short":
            liq_hit = mark_high >= levels["liq_price"]
            stop_hit = last_high >= levels["stop_price"]
            target_hit = last_low <= levels["target_price"]
        else:
            liq_hit = mark_low <= levels["liq_price"]
            stop_hit = last_low <= levels["stop_price"]
            target_hit = last_high >= levels["target_price"]
        liq_hit_any = liq_hit_any or liq_hit
        stop_hit_any = stop_hit_any or stop_hit
        if liq_hit and stop_hit:
            exit_ts = bar["timestamp"]
            exit_price = levels["liq_price"]
            exit_reason = "same_minute_stop_liquidation_ambiguous"
            ordering = "same_minute_stop_liquidation_ambiguous"
            ambiguity = True
            break
        if liq_hit:
            exit_ts = bar["timestamp"]
            exit_price = levels["liq_price"]
            exit_reason = "liquidation"
            ordering = "liquidation_before_stop"
            break
        if stop_hit:
            exit_ts = bar["timestamp"]
            exit_price = levels["stop_price"]
            exit_reason = "stop"
            ordering = "stop_before_liquidation" if liq_hit_any else "stop_no_liquidation"
            break
        if target_hit:
            exit_ts = bar["timestamp"]
            exit_price = levels["target_price"]
            exit_reason = "target"
            ordering = "target_before_liquidation" if not liq_hit_any else "target_after_liquidation_seen"
            break
    if side == "short":
        gross_bps = (entry_price - exit_price) / entry_price * 10000.0
        mark_adverse_bps = max(0.0, (float(path["mark_high"].max()) - entry_price) / entry_price * 10000.0)
        mark_favorable_bps = max(0.0, (entry_price - float(path["mark_low"].min())) / entry_price * 10000.0)
    else:
        gross_bps = (exit_price - entry_price) / entry_price * 10000.0
        mark_adverse_bps = max(0.0, (entry_price - float(path["mark_low"].min())) / entry_price * 10000.0)
        mark_favorable_bps = max(0.0, (float(path["mark_high"].max()) - entry_price) / entry_price * 10000.0)
    cost_bps = cost_bps_for_tier(source_event.get("liquidity_tier", "UNKNOWN"))
    net_R = (gross_bps - cost_bps) / max(levels["stop_bps"], 1e-9)
    rankable = (not ambiguity) and exit_reason != "liquidation"
    return {
        "mark_replay_status": "resolved_1m_mark_path",
        "ohlcv_coverage_status": ohlcv_status,
        "mark_coverage_status": mark_status,
        "entry_ts": entry_ts,
        "horizon_end": horizon_end,
        "entry_price_1m": entry_price,
        "entry_source": entry_source,
        "exit_ts_1m": exit_ts,
        "exit_price_1m": exit_price,
        "exit_reason_1m": exit_reason,
        "ordering_class_1m": ordering,
        "same_minute_ambiguity_1m": ambiguity,
        "actual_mark_liquidation_1m": exit_reason == "liquidation",
        "stop_hit_any_1m": stop_hit_any,
        "liq_hit_any_1m": liq_hit_any,
        "stop_price_1m": levels["stop_price"],
        "target_price_1m": levels["target_price"],
        "liq_price_1m": levels["liq_price"],
        "stop_distance_bps_1m": levels["stop_bps"],
        "gross_bps_1m": gross_bps,
        "cost_bps_1m": cost_bps,
        "net_R_1m_mark": net_R,
        "mark_mfe_bps_1m": mark_favorable_bps,
        "mark_mae_bps_1m": mark_adverse_bps,
        "rankable_1m_mark": rankable,
    }


def build_one_minute_replay(ctx: RunContext) -> pd.DataFrame:
    raw = read_csv_safe(ctx.run_root / "targeted_1m/raw_targeted_windows.csv")
    dedup = read_csv_safe(ctx.run_root / "targeted_1m/deduped_windows.csv")
    d4 = load_d4(ctx)
    if raw.empty or dedup.empty or d4.empty:
        return pd.DataFrame()
    raw = raw[raw["window_type"].isin(["accepted_d4_event", "matched_null_window"])].copy()
    raw = _assign_dedup_window_ids(raw, dedup)
    source_by_id = {str(r["event_id"]): r for r in d4.to_dict("records")}
    files = _download_index(ctx)
    rows: list[dict[str, Any]] = []
    grouped = raw.groupby("dedup_window_id", dropna=False)
    for n, (wid, sub) in enumerate(grouped, start=1):
        if not wid:
            for _, row in sub.iterrows():
                rows.append({**row.to_dict(), "mark_replay_status": "fail_closed_no_dedup_window_match"})
            continue
        ohlcv_path = files.get((str(wid), "ohlcv_1m"))
        mark_path = files.get((str(wid), "mark_1m"))
        if ohlcv_path is None or mark_path is None:
            for _, row in sub.iterrows():
                rows.append({**row.to_dict(), "mark_replay_status": "fail_closed_missing_downloaded_ohlcv_or_mark", "dedup_window_id": wid})
            continue
        try:
            ohlcv = pd.read_parquet(ohlcv_path)
            mark = pd.read_parquet(mark_path)
        except Exception as exc:
            for _, row in sub.iterrows():
                rows.append({**row.to_dict(), "mark_replay_status": f"fail_closed_parquet_read_error_{type(exc).__name__}", "dedup_window_id": wid})
            continue
        for _, row in sub.iterrows():
            event_id = str(row.get("event_id", ""))
            raw_source_id = row.get("source_event_id", "")
            if raw_source_id is None or (isinstance(raw_source_id, float) and np.isnan(raw_source_id)) or pd.isna(raw_source_id) or str(raw_source_id).strip() == "":
                source_id = event_id
            else:
                source_id = str(raw_source_id)
            source = source_by_id.get(source_id)
            if source is None:
                rows.append({**row.to_dict(), "mark_replay_status": "fail_closed_missing_source_event", "dedup_window_id": wid})
                continue
            replay = _replay_single_window(window_row=row.to_dict(), source_event=source, ohlcv=ohlcv, mark=mark)
            rows.append({**row.to_dict(), **replay, "source_event_id": source_id, "dedup_window_id": wid, "source_event_net_R": source.get("net_R", np.nan), "source_event_reference_risk_bps": source.get("reference_risk_bps", np.nan)})
        if n % max(1, ctx.args.chunk_size) == 0:
            ctx.notifier.send("D4 1M MARK REPLAY PROGRESS", f"dedup_windows_done={n}/{len(grouped)}")
    out = pd.DataFrame(rows)
    validate_no_protected(out, ["decision_ts", "window_start", "window_end", "entry_ts", "horizon_end", "exit_ts_1m"])
    return out


def stage_one_minute_mark(ctx: RunContext) -> None:
    manifest = read_csv_safe(ctx.run_root / "downloaded_1m/download_manifest.csv")
    prior = read_csv_safe(PRIOR_ROOT / "one_minute/prelead_1m_overlay_summary.csv")
    prior_row = prior[prior["candidate_id"].astype(str).eq(CANDIDATE_ID)].iloc[0].to_dict() if not prior.empty and prior["candidate_id"].astype(str).eq(CANDIDATE_ID).any() else {}
    mark_rows = 0
    ohlcv_rows = 0
    if not manifest.empty:
        mark_rows = int(pd.to_numeric(manifest.loc[manifest["dataset"].eq("mark_1m"), "rows"], errors="coerce").fillna(0).sum())
        ohlcv_rows = int(pd.to_numeric(manifest.loc[manifest["dataset"].eq("ohlcv_1m"), "rows"], errors="coerce").fillna(0).sum())
    replay = build_one_minute_replay(ctx) if mark_rows > 0 and ohlcv_rows > 0 else pd.DataFrame()
    out_dir = ctx.run_root / "one_minute_mark"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not replay.empty:
        replay.to_parquet(out_dir / "d4_1m_mark_replay_by_window.parquet", index=False, compression="zstd")
    accepted = replay[replay.get("window_type", pd.Series(dtype=str)).astype(str).eq("accepted_d4_event")] if not replay.empty else pd.DataFrame()
    resolved = accepted[accepted.get("mark_replay_status", pd.Series(dtype=str)).astype(str).eq("resolved_1m_mark_path")] if not accepted.empty else pd.DataFrame()
    fail_closed = int(len(accepted) - len(resolved)) if not accepted.empty else 0
    actual_liq = int(resolved.get("actual_mark_liquidation_1m", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not resolved.empty else 0
    ambiguous = int(resolved.get("same_minute_ambiguity_1m", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not resolved.empty else 0
    rankable = resolved[resolved.get("rankable_1m_mark", pd.Series(dtype=bool)).fillna(False).astype(bool)] if not resolved.empty else pd.DataFrame()
    status = "no_download_no_new_mark_replay" if mark_rows == 0 else ("event_level_1m_mark_replay_complete" if not replay.empty else "downloaded_mark_rows_available_but_replay_empty")
    rows = [{
        "candidate_id": CANDIDATE_ID,
        "downloaded_mark_1m_rows": mark_rows,
        "downloaded_ohlcv_1m_rows": ohlcv_rows,
        "prior_pilot_overlap_rows": prior_row.get("pilot_1m_overlap_rows", 0),
        "prior_material_same_bar_share": prior_row.get("material_same_bar_share", np.nan),
        "one_minute_mark_replay_status": status,
        "accepted_events": int(len(accepted)),
        "resolved_accepted_events": int(len(resolved)),
        "fail_closed_accepted_events": fail_closed,
        "rankable_accepted_events": int(len(rankable)),
        "actual_mark_liquidation_resolved_events": actual_liq,
        "same_minute_ambiguous_events": ambiguous,
        "net_R_1m_mark_rankable": float(pd.to_numeric(rankable.get("net_R_1m_mark", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not rankable.empty else 0.0,
        "PF_1m_mark_rankable": float(pd.to_numeric(rankable.get("net_R_1m_mark", pd.Series(dtype=float)), errors="coerce").pipe(lambda s: s[s > 0].sum() / max(-s[s < 0].sum(), 1e-12))) if not rankable.empty else 0.0,
        "proxy_liquidation_overstatement_proven": bool(actual_liq == 0 and len(resolved) == len(accepted) and len(accepted) > 0),
    }]
    write_csv(out_dir / "one_minute_mark_replay_summary.csv", rows)
    write_text(out_dir / "one_minute_mark_replay_report.md", f"# One-Minute Mark Replay\n\n- status: `{status}`\n- downloaded 1m mark rows: `{mark_rows}`\n- downloaded 1m OHLCV rows: `{ohlcv_rows}`\n- accepted windows: `{len(accepted)}`\n- resolved accepted events: `{len(resolved)}`\n- fail-closed accepted events: `{fail_closed}`\n- rankable accepted events: `{len(rankable)}`\n- actual mark liquidation events: `{actual_liq}`\n- same-minute ambiguous events: `{ambiguous}`\n- rankable 1m mark net_R: `{rows[0]['net_R_1m_mark_rankable']}`\n- prior pilot overlap rows: `{prior_row.get('pilot_1m_overlap_rows', 0)}`\n- conclusion: event-level 1m mark replay is now the primary liquidation evidence where coverage is complete; missing/incomplete coverage fails closed.\n")


def stage_ordering(ctx: RunContext) -> None:
    out_dir = ctx.run_root / "ordering"
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = ctx.run_root / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    if replay_path.exists():
        rep = pd.read_parquet(replay_path)
        accepted = rep[rep.get("window_type", pd.Series(dtype=str)).astype(str).eq("accepted_d4_event")].copy()
        cls = accepted.get("ordering_class_1m", pd.Series("missing_or_fail_closed", index=accepted.index)).fillna(accepted.get("mark_replay_status", "missing")).astype(str)
        evidence = "1m_mark_replay_primary"
        denom = max(len(accepted), 1)
    else:
        d4 = load_d4(ctx)
        stop_bps = stop_distance_bps(d4)
        mae = pd.to_numeric(d4.get(f"{HORIZON}_mae_bps"), errors="coerce").fillna(0.0)
        liq_proxy = d4.get("liquidation_flag_proxy_10x", pd.Series(False, index=d4.index)).fillna(False).astype(bool)
        stop_hit = mae >= stop_bps
        cls = pd.Series(np.where(~liq_proxy, "no_proxy_liquidation", np.where(stop_hit, "same_bar_or_path_ordering_ambiguous_stop_and_proxy_liq_seen", "proxy_liquidation_without_stop_hit")))
        evidence = "proxy_path_fallback"
        denom = max(len(d4), 1)
    summ = cls.value_counts().reset_index()
    summ.columns = ["ordering_class", "events"]
    summ["share"] = summ["events"] / denom
    summ["evidence_source"] = evidence
    summ.to_csv(out_dir / "stop_liquidation_ordering_summary.csv", index=False)
    write_text(out_dir / "stop_liquidation_ordering_report.md", "# Stop Before Liquidation Ordering Audit\n\n" + summ.to_markdown(index=False) + f"\n\nEvidence source: `{evidence}`. Same-minute conflicts remain fail-closed/ambiguous rather than proof of safe execution.\n")


def stage_leverage(ctx: RunContext) -> None:
    out_dir = ctx.run_root / "leverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = ctx.run_root / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    if replay_path.exists():
        rep = pd.read_parquet(replay_path)
        accepted = rep[(rep.get("window_type", pd.Series(dtype=str)).astype(str).eq("accepted_d4_event")) & (rep.get("mark_replay_status", pd.Series(dtype=str)).astype(str).eq("resolved_1m_mark_path"))]
        mae = pd.to_numeric(accepted.get("mark_mae_bps_1m"), errors="coerce").fillna(0.0)
        evidence = "1m_mark_replay_primary"
    else:
        d4 = load_d4(ctx)
        mae = pd.to_numeric(d4.get(f"{HORIZON}_mae_bps"), errors="coerce").fillna(0.0)
        evidence = "proxy_path_fallback"
    rows = []
    for lev in [2.0, 3.0, 5.0, 7.5, 10.0]:
        threshold = liquidation_adverse_bps_for_leverage(lev)
        liq = mae >= threshold
        rows.append({"leverage": lev, "liquidation_adverse_threshold_bps": threshold, "liquidation_count": int(liq.sum()), "liquidation_share": float(liq.mean()) if len(liq) else 0.0, "evidence_source": evidence, "resolved_events": int(len(mae))})
    write_csv(out_dir / "leverage_margin_sensitivity_summary.csv", rows)
    write_text(out_dir / "leverage_margin_sensitivity_report.md", "# Leverage And Margin Sensitivity\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\nThis uses adverse last-price/proxy path distance where mark path is unavailable, so counts are diagnostic only.\n")


def stage_buffer(ctx: RunContext) -> None:
    out_dir = ctx.run_root / "filters"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    liq_threshold_10x = liquidation_adverse_bps_for_leverage(10.0)
    replay_path = ctx.run_root / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    if replay_path.exists():
        rep = pd.read_parquet(replay_path)
        base = rep[(rep.get("window_type", pd.Series(dtype=str)).astype(str).eq("accepted_d4_event")) & (rep.get("mark_replay_status", pd.Series(dtype=str)).astype(str).eq("resolved_1m_mark_path"))].copy()
        stop_bps = pd.to_numeric(base.get("stop_distance_bps_1m"), errors="coerce").fillna(0.0)
        returns = pd.to_numeric(base.get("net_R_1m_mark"), errors="coerce").fillna(0.0)
        evidence = "1m_mark_replay_primary"
    else:
        base = load_d4(ctx)
        stop_bps = stop_distance_bps(base)
        returns = pd.to_numeric(base.get("net_R"), errors="coerce").fillna(0.0)
        evidence = "proxy_path_fallback"
    for buf in [1.0, 1.25, 1.5, 2.0]:
        keep = liq_threshold_10x >= buf * stop_bps
        sm = summarize_returns(returns[keep])
        rows.append({"buffer_multiple": buf, "events_retained": int(keep.sum()), "events_rejected": int((~keep).sum()), **sm, "safety_filter_status": "diagnostic_not_alpha_filter", "evidence_source": evidence})
    write_csv(out_dir / "liquidation_buffer_filter_summary.csv", rows)
    write_text(out_dir / "liquidation_buffer_filter_report.md", "# Liquidation Buffer Filter Study\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\nThese are safety filters, not tuned alpha filters. Any positive result still requires mark-path evidence.\n")


def load_null_pool(ctx: RunContext) -> pd.DataFrame:
    p = PATH_DIAG_ROOT / "matched_null/matched_null_path_metrics.parquet"
    if not p.exists():
        return pd.DataFrame()
    cols = parquet_columns(p)
    keep = [c for c in cols if c in {"event_id", "family", "variant_id", "symbol", "side", "liquidity_tier", "decision_ts", "entry_ts", "reference_risk_bps", "mark_path_status", "data_quality_flags", f"{HORIZON}_mfe_bps", f"{HORIZON}_mae_bps", f"{HORIZON}_close_return_bps", f"{HORIZON}_liquidation_10x"}]
    df = pd.read_parquet(p, columns=keep)
    df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce")
    df = df[(df["decision_ts"] >= ctx.start) & (df["decision_ts"] <= ctx.end)]
    validate_no_protected(df, ["decision_ts", "entry_ts"])
    return df.reset_index(drop=True)


def sample_matched_nulls(events: pd.DataFrame, pool: pd.DataFrame, nulls_per_event: int, seed: int) -> pd.DataFrame:
    if events.empty or pool.empty or nulls_per_event <= 0:
        return pd.DataFrame()
    rng = random.Random(seed)
    pool = pool.copy()
    pool["month"] = pd.to_datetime(pool["decision_ts"], utc=True).dt.strftime("%Y-%m")
    pool["_row"] = np.arange(len(pool))
    groups: dict[tuple[str, str], np.ndarray] = {k: v["_row"].to_numpy() for k, v in pool.groupby(["symbol", "month"])}
    tier_groups: dict[tuple[str, str], np.ndarray] = {k: v["_row"].to_numpy() for k, v in pool.groupby(["liquidity_tier", "month"])}
    picks: list[int] = []
    for _, ev in events.iterrows():
        month = pd.Timestamp(pd.to_datetime(ev["decision_ts"], utc=True)).strftime("%Y-%m")
        candidates = groups.get((str(ev["symbol"]), month))
        match_level = "same_symbol_month"
        if candidates is None or len(candidates) == 0:
            candidates = tier_groups.get((str(ev.get("liquidity_tier", "")), month))
            match_level = "same_tier_month"
        if candidates is None or len(candidates) == 0:
            candidates = pool["_row"].to_numpy()
            match_level = "global_fallback"
        candidates = list(map(int, candidates))
        rng.shuffle(candidates)
        for idx in candidates[:nulls_per_event]:
            picks.append(idx)
    out = pool.iloc[picks].copy().reset_index(drop=True) if picks else pd.DataFrame()
    if not out.empty:
        out["matched_null_sample_seed"] = seed
    return out


def stage_matched_null(ctx: RunContext) -> None:
    out_dir = ctx.run_root / "matched_null"
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = ctx.run_root / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    if replay_path.exists():
        rep = pd.read_parquet(replay_path)
        events = rep[(rep.get("window_type", pd.Series(dtype=str)).astype(str).eq("accepted_d4_event")) & (rep.get("mark_replay_status", pd.Series(dtype=str)).astype(str).eq("resolved_1m_mark_path")) & (rep.get("rankable_1m_mark", pd.Series(dtype=bool)).fillna(False).astype(bool))].copy()
        nulls = rep[(rep.get("window_type", pd.Series(dtype=str)).astype(str).eq("matched_null_window")) & (rep.get("mark_replay_status", pd.Series(dtype=str)).astype(str).eq("resolved_1m_mark_path")) & (rep.get("rankable_1m_mark", pd.Series(dtype=bool)).fillna(False).astype(bool))].copy()
        event_ret = pd.to_numeric(events.get("net_R_1m_mark"), errors="coerce").fillna(0.0)
        null_ret = pd.to_numeric(nulls.get("net_R_1m_mark"), errors="coerce").fillna(0.0)
        evidence_source = "1m_mark_replay_primary"
        if not nulls.empty:
            nulls.to_parquet(out_dir / "d4_refreshed_matched_null_ledger.parquet", index=False, compression="zstd")
    else:
        events = load_d4(ctx)
        pool = load_null_pool(ctx)
        if ctx.args.smoke:
            events = events.head(min(100, len(events)))
            pool = pool.head(min(5000, len(pool)))
        nulls = sample_matched_nulls(events, pool, ctx.args.nulls_per_event, ctx.args.seed)
        if not nulls.empty:
            nulls.to_parquet(out_dir / "d4_refreshed_matched_null_ledger.parquet", index=False, compression="zstd")
        event_ret = pd.to_numeric(events.get("net_R"), errors="coerce").fillna(0.0)
        null_ret = surface_return_r(nulls, HORIZON, TARGET_R, STOP_MULT, COST_MULT) if not nulls.empty else pd.Series(dtype=float)
        evidence_source = "proxy_path_fallback"
    row = {
        "candidate_id": CANDIDATE_ID,
        "event_count": int(len(events)),
        "null_count": int(len(nulls)),
        "effective_nulls_per_event": float(len(nulls) / max(len(events), 1)),
        "event_mean_R": float(event_ret.mean()) if len(event_ret) else 0.0,
        "null_mean_R": float(null_ret.mean()) if len(null_ret) else 0.0,
        "event_net_R": float(event_ret.sum()),
        "null_net_R": float(null_ret.sum()) if len(null_ret) else 0.0,
        "event_minus_null_net_R": float(event_ret.sum() - (null_ret.sum() if len(null_ret) else 0.0)),
        "beats_refreshed_matched_null": bool(len(null_ret) > 0 and event_ret.mean() > null_ret.mean()),
        "null_support_cap": "full_3_null_support" if len(nulls) >= len(events) * 3 else "limited_null_support_caps_verdict",
        "evidence_source": evidence_source,
    }
    write_csv(out_dir / "d4_refreshed_matched_null_summary.csv", [row])
    write_json(out_dir / "d4_refreshed_matched_null_policy.json", {"nulls_per_event_requested": ctx.args.nulls_per_event, "seed": ctx.args.seed, "match_priority": ["same_symbol_month", "same_tier_month", "global_fallback", "downloaded_window_offsets"], "fresh_nulls_generated": True, "evidence_source": evidence_source})
    write_text(out_dir / "d4_refreshed_matched_null_report.md", f"# D4 Refreshed Matched Null Report\n\n- evidence source: `{evidence_source}`\n- event count: `{row['event_count']}`\n- null count: `{row['null_count']}`\n- effective nulls per event: `{row['effective_nulls_per_event']:.3f}`\n- event net_R: `{row['event_net_R']:.6f}`\n- null net_R: `{row['null_net_R']:.6f}`\n- beats refreshed null: `{row['beats_refreshed_matched_null']}`\n- null support cap: `{row['null_support_cap']}`\n")


def stage_stress(ctx: RunContext) -> None:
    replay_path = ctx.run_root / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    if replay_path.exists():
        rep = pd.read_parquet(replay_path)
        d4 = rep[(rep.get("window_type", pd.Series(dtype=str)).astype(str).eq("accepted_d4_event")) & (rep.get("mark_replay_status", pd.Series(dtype=str)).astype(str).eq("resolved_1m_mark_path")) & (rep.get("rankable_1m_mark", pd.Series(dtype=bool)).fillna(False).astype(bool))].copy()
        base_returns = pd.to_numeric(d4.get("net_R_1m_mark"), errors="coerce").fillna(0.0)
        risk = pd.to_numeric(d4.get("stop_distance_bps_1m"), errors="coerce").fillna(100.0).clip(lower=1.0)
        liq_count = int(d4.get("actual_mark_liquidation_1m", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        evidence_source = "1m_mark_replay_primary"
    else:
        d4 = load_d4(ctx)
        base_returns = pd.to_numeric(d4.get("net_R"), errors="coerce").fillna(0.0)
        risk = pd.to_numeric(d4.get("reference_risk_bps"), errors="coerce").fillna(100.0).clip(lower=1.0) * STOP_MULT
        liq_count = int(d4.get("liquidation_flag_proxy_10x", pd.Series(False, index=d4.index)).fillna(False).astype(bool).sum())
        evidence_source = "proxy_path_fallback"
    rows = []
    for name, cmult, extra_bps in [("base", 1.0, 0.0), ("cost_x1p25", 1.25, 0.0), ("cost_x1p5", 1.5, 0.0), ("cost_x2", 2.0, 0.0), ("add_10bps", 1.0, 10.0), ("add_25bps", 1.0, 25.0), ("funding_doubled_adverse_proxy", 1.0, 0.0), ("mark_fallback_disabled", 1.0, 0.0)]:
        if evidence_source == "1m_mark_replay_primary":
            # Approximate cost multiplier by subtracting the extra cost beyond base tier cost.
            base_cost_bps = pd.to_numeric(d4.get("cost_bps_1m"), errors="coerce").fillna(0.0)
            ret = base_returns - ((base_cost_bps * (cmult - 1.0)) / risk)
        else:
            ret = surface_return_r(d4, HORIZON, TARGET_R, STOP_MULT, COST_MULT * cmult)
        if extra_bps:
            ret = ret - (extra_bps / risk)
        if name == "funding_doubled_adverse_proxy":
            # Funding is already proxy-level here. Apply a small adverse haircut to avoid false precision.
            ret = ret - 0.01
        if name == "mark_fallback_disabled":
            if evidence_source == "proxy_path_fallback":
                ret = ret.mask(d4.get("mark_path_status", pd.Series("", index=d4.index)).astype(str).str.contains("proxy|last_price", case=False, regex=True, na=False), np.nan).dropna()
        sm = summarize_returns(ret)
        rows.append({"scenario": name, "candidate_id": CANDIDATE_ID, **sm, "liquidation_count": liq_count if name != "mark_fallback_disabled" else (liq_count if evidence_source == "1m_mark_replay_primary" else "not_rankable_mark_rows_only"), "evidence_source": evidence_source})
    write_csv(ctx.run_root / "stress/d4_cost_funding_execution_stress_summary.csv", rows)
    write_text(ctx.run_root / "stress/d4_cost_funding_execution_stress_report.md", "# D4 Cost/Funding/Execution Stress Refresh\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\nStress changes accounting only. It does not change entries or thresholds.\n")


def stage_portfolio(ctx: RunContext) -> None:
    replay_path = ctx.run_root / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    if replay_path.exists():
        rep = pd.read_parquet(replay_path)
        d4 = rep[(rep.get("window_type", pd.Series(dtype=str)).astype(str).eq("accepted_d4_event")) & (rep.get("mark_replay_status", pd.Series(dtype=str)).astype(str).eq("resolved_1m_mark_path")) & (rep.get("rankable_1m_mark", pd.Series(dtype=bool)).fillna(False).astype(bool))].copy()
        ret = pd.to_numeric(d4.get("net_R_1m_mark"), errors="coerce").fillna(0.0)
        liq_count = int(d4.get("actual_mark_liquidation_1m", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        evidence_source = "1m_mark_replay_primary"
    else:
        d4 = load_d4(ctx)
        ret = pd.to_numeric(d4.get("net_R"), errors="coerce").fillna(0.0)
        liq_count = int(d4.get("liquidation_flag_proxy_10x", pd.Series(False, index=d4.index)).fillna(False).astype(bool).sum())
        evidence_source = "proxy_path_fallback"
    rows = []
    for equity in [200.0, 500.0, 1000.0]:
        for risk_pct in [0.025, 0.05, 0.10, 0.15, 0.20]:
            curve = equity * (1.0 + ret * risk_pct).clip(lower=0.0).cumprod()
            max_dd = float(((curve / curve.cummax()) - 1.0).min() * 100.0) if len(curve) else 0.0
            rows.append({
                "candidate_id": CANDIDATE_ID,
                "starting_equity": equity,
                "risk_pct": risk_pct,
                "ending_equity": float(curve.iloc[-1]) if len(curve) else equity,
                "max_drawdown_pct": max_dd,
                "ruin_flag": bool((curve <= equity * 0.1).any()) if len(curve) else False,
                "liquidation_count": liq_count,
                "evidence_source": evidence_source,
                "portfolio_overlay_status": "diagnostic_not_live_recommendation",
            })
    write_csv(ctx.run_root / "portfolio/d4_aggressive_10x_risk_expression_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/d4_aggressive_10x_risk_expression_report.md", "# D4 Aggressive 10x Risk Expression\n\n" + pd.DataFrame(rows).head(20).to_markdown(index=False) + "\n\nAggressive sizing cannot create alpha and this overlay is not a live recommendation. Proxy liquidation remains a blocker.\n")


def stage_decision(ctx: RunContext) -> None:
    recon = read_csv_safe(ctx.run_root / "d4_reconstruction/d4_reconstruction_metrics.csv")
    tax = read_csv_safe(ctx.run_root / "liquidation/liquidation_taxonomy_summary.csv")
    mark = read_csv_safe(ctx.run_root / "mark_path/mark_path_availability_summary.csv")
    one = read_csv_safe(ctx.run_root / "one_minute_mark/one_minute_mark_replay_summary.csv")
    null = read_csv_safe(ctx.run_root / "matched_null/d4_refreshed_matched_null_summary.csv")
    stress = read_csv_safe(ctx.run_root / "stress/d4_cost_funding_execution_stress_summary.csv")
    verdict = "continue_to_targeted_execution_data_collection"
    reasons = []
    if recon.empty:
        verdict = "blocked_by_protocol_issue"
        reasons.append("missing_reconstruction_metrics")
    else:
        status = str(recon.iloc[0].get("reconstruction_status", ""))
        if not (ctx.args.smoke or "exact_match" in status or "subset" in status):
            verdict = "blocked_by_protocol_issue"
            reasons.append("reconstruction_not_auditable")
    mark_rows = 0
    if not mark.empty and mark["mark_path_status"].astype(str).eq("__any_mark_available__").any():
        mark_rows = int(mark.loc[mark["mark_path_status"].astype(str).eq("__any_mark_available__"), "events"].iloc[0])
    actual_mark_liq = int(one.iloc[0].get("actual_mark_liquidation_resolved_events", 0)) if not one.empty else 0
    accepted_events = int(one.iloc[0].get("accepted_events", 0)) if not one.empty else 0
    resolved_events = int(one.iloc[0].get("resolved_accepted_events", 0)) if not one.empty else 0
    rankable_events = int(one.iloc[0].get("rankable_accepted_events", 0)) if not one.empty else 0
    fail_closed_events = int(one.iloc[0].get("fail_closed_accepted_events", 0)) if not one.empty else 0
    if accepted_events == 0 or resolved_events == 0:
        verdict = "continue_to_targeted_execution_data_collection" if not ctx.args.download_targeted_1m else "blocked_by_mark_data"
        reasons.append("liquidation_unresolved_proxy_mark_data")
    elif fail_closed_events > 0:
        verdict = "blocked_by_mark_data"
        reasons.append(f"fail_closed_missing_mark_coverage_{fail_closed_events}")
    elif actual_mark_liq > 0:
        verdict = "reject_d4_current_expression"
        reasons.append(f"actual_mark_liquidations_{actual_mark_liq}")
    elif rankable_events < resolved_events:
        verdict = "blocked_by_mark_data"
        reasons.append("same_minute_ambiguity_remaining")
    if not null.empty and not bool(null.iloc[0].get("beats_refreshed_matched_null", False)):
        verdict = "reject_d4_current_expression"
        reasons.append("does_not_beat_refreshed_matched_null")
    if not stress.empty:
        s125 = stress[stress["scenario"].eq("cost_x1p25")]
        if not s125.empty and float(s125.iloc[0].get("net_R", 0.0)) <= 0:
            verdict = "reject_d4_current_expression"
            reasons.append("fails_cost_x1p25")
    if verdict == "promote_to_family_specific_validation_after_data" and (resolved_events < accepted_events or actual_mark_liq > 0):
        verdict = "blocked_by_mark_data"
        reasons.append("promotion_blocked_mark_replay_not_clean")
    if verdict not in ALLOWED_VERDICTS:
        verdict = "blocked_by_protocol_issue"
    summary = {"candidate_id": CANDIDATE_ID, "verdict": verdict, "reasons": reasons, "final_holdout_untouched": True, "protected_start": str(FINAL_HOLDOUT_START), "run_root": str(ctx.run_root), "created_at_utc": utc_now()}
    write_json(ctx.run_root / "decision_summary.json", summary)
    write_text(ctx.run_root / "D4_LIQUIDATION_EXECUTION_AUDIT_REPORT.md", f"# D4 Liquidation Execution Audit Report\n\n## Verdict\n\n`{verdict}`\n\nReasons: `{';'.join(reasons)}`\n\n## Key Facts\n\n- Protected holdout untouched: `true`\n- Candidate: `{CANDIDATE_ID}`\n- Prior root: `{PRIOR_ROOT}`\n- Exact/diagnostic reconstruction status: `{recon.iloc[0].get('reconstruction_status', 'missing') if not recon.empty else 'missing'}`\n- Original mark-available rows: `{mark_rows}`\n- Accepted events in 1m replay: `{accepted_events}`\n- Resolved accepted events: `{resolved_events}`\n- Rankable accepted events: `{rankable_events}`\n- Fail-closed accepted events: `{fail_closed_events}`\n- Actual mark liquidation events: `{actual_mark_liq}`\n- Refreshed matched null beats status: `{null.iloc[0].get('beats_refreshed_matched_null', 'missing') if not null.empty else 'missing'}`\n- Cost x1.25 net_R: `{stress[stress['scenario'].eq('cost_x1p25')].iloc[0].get('net_R', 'missing') if not stress.empty and stress['scenario'].eq('cost_x1p25').any() else 'missing'}`\n\n## Interpretation\n\nD4 is evaluated with event-level 1m mark replay where downloaded coverage is complete. Missing/incomplete mark coverage fails closed. This report does not validate D4, authorize sealed validation, or authorize live trading.\n")
    ctx.notifier.send("D4 LIQUIDATION AUDIT COMPLETE", f"verdict={verdict}\nrun_root={ctx.run_root}")


def stage_compact(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    keep = [
        "D4_LIQUIDATION_EXECUTION_AUDIT_REPORT.md",
        "decision_summary.json",
        "preflight/preflight_report.md",
        "preflight/frozen_artifact_hashes.json",
        "seal/seal_guard_report.md",
        "d4_reconstruction/d4_reconstruction_report.md",
        "d4_reconstruction/d4_reconstruction_metrics.csv",
        "liquidation/liquidation_taxonomy_report.md",
        "liquidation/liquidation_taxonomy_summary.csv",
        "mark_path/mark_path_availability_report.md",
        "targeted_1m/targeted_1m_window_plan.md",
        "targeted_1m/storage_estimate.csv",
        "downloaded_1m/download_report.md",
        "one_minute_mark/one_minute_mark_replay_report.md",
        "one_minute_mark/one_minute_mark_replay_summary.csv",
        "ordering/stop_liquidation_ordering_report.md",
        "leverage/leverage_margin_sensitivity_report.md",
        "filters/liquidation_buffer_filter_report.md",
        "matched_null/d4_refreshed_matched_null_report.md",
        "matched_null/d4_refreshed_matched_null_summary.csv",
        "stress/d4_cost_funding_execution_stress_report.md",
        "portfolio/d4_aggressive_10x_risk_expression_report.md",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
        "command_log.jsonl",
    ]
    rows = []
    for rel in keep:
        src = ctx.run_root / rel
        if src.exists() and src.is_file() and src.stat().st_size <= 10 * 1024 * 1024:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            rows.append({"relative_path": rel, "bundle_path": str(dst.relative_to(ctx.run_root)), "size_bytes": src.stat().st_size, "included": True})
        else:
            rows.append({"relative_path": rel, "bundle_path": "", "size_bytes": src.stat().st_size if src.exists() and src.is_file() else 0, "included": False})
    # Small samples only.
    d4p = ctx.run_root / "d4_reconstruction/d4_event_ledger.parquet"
    if d4p.exists():
        sample = pd.read_parquet(d4p).head(500)
        validate_no_protected(sample, ["decision_ts", "entry_ts"])
        sample.to_parquet(bundle / "d4_event_ledger_sample_500.parquet", index=False, compression="zstd")
        rows.append({"relative_path": "d4_reconstruction/d4_event_ledger.parquet", "bundle_path": "d4_event_ledger_sample_500.parquet", "size_bytes": (bundle / "d4_event_ledger_sample_500.parquet").stat().st_size, "included": "sample_only"})
    replayp = ctx.run_root / "one_minute_mark/d4_1m_mark_replay_by_window.parquet"
    if replayp.exists():
        sample = pd.read_parquet(replayp).head(500)
        validate_no_protected(sample, ["decision_ts", "window_start", "window_end", "entry_ts", "horizon_end", "exit_ts_1m"])
        sample.to_parquet(bundle / "d4_1m_mark_replay_sample_500.parquet", index=False, compression="zstd")
        rows.append({"relative_path": "one_minute_mark/d4_1m_mark_replay_by_window.parquet", "bundle_path": "d4_1m_mark_replay_sample_500.parquet", "size_bytes": (bundle / "d4_1m_mark_replay_sample_500.parquet").stat().st_size, "included": "sample_only"})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_json(bundle / "artifact_path_index.json", {"artifacts": rows})
    zip_path = ctx.run_root / "qlmg_d4_liquidation_execution_audit_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(bundle))


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        print(f"[skip] {stage}")
        return
    ensure_guard(ctx, stage, estimate_stage_gb(stage, ctx))
    append_command(ctx.run_root, stage)
    ctx.notifier.send("D4 AUDIT STAGE START", f"stage={stage}\nrun_root={ctx.run_root}")
    dispatch = {
        "preflight-and-artifact-freeze": stage_preflight,
        "telegram-and-tmux-setup": stage_telegram,
        "seal-guard": stage_seal,
        "d4-contract-and-event-reconstruction": stage_reconstruct,
        "liquidation-flag-taxonomy": stage_taxonomy,
        "mark-path-availability-audit": stage_mark_availability,
        "targeted-1m-window-plan": stage_window_plan,
        "targeted-1m-download-if-approved": stage_download,
        "one-minute-mark-replay": stage_one_minute_mark,
        "stop-before-liquidation-ordering-audit": stage_ordering,
        "leverage-and-margin-sensitivity": stage_leverage,
        "liquidation-buffer-filter-study": stage_buffer,
        "matched-null-refresh": stage_matched_null,
        "cost-funding-execution-stress-refresh": stage_stress,
        "aggressive-10x-risk-expression-study": stage_portfolio,
        "decision-report": stage_decision,
        "compact-review-bundle": stage_compact,
    }
    dispatch[stage](ctx)
    mark_done(ctx.run_root, stage)


def main() -> int:
    args = parse_args()
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": reason, "argv": sys.argv, "created_at_utc": utc_now(), "candidate_id": CANDIDATE_ID, "protected_start": str(FINAL_HOLDOUT_START), "download_targeted_1m": args.download_targeted_1m})
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        return 0
    except Exception as exc:
        notifier.send("D4 LIQUIDATION AUDIT FAILED", f"{type(exc).__name__}: {exc}", level="error")
        (run_root / "watch_status.json").write_text(json.dumps({"ts_utc": utc_now(), "status": "failed", "error": f"{type(exc).__name__}: {exc}", "run_root": str(run_root)}, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
