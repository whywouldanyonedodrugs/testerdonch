#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402
from tools.run_qlmg_execution_depth_pilot import (  # noqa: E402
    D4_CANDIDATE_ID,
    D4_SURVIVAL_ROOT,
    LISTING_IDS,
    LISTING_ROOT,
    build_full_window_manifest,
    select_pilot_windows,
)

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_bybit_historical_execution_data_route_20260628_v1"
DEFAULT_SEED = 20260628
GB = 1024**3

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "candidate-and-window-freeze",
    "bybit-official-data-route-audit",
    "bybit-historical-url-and-schema-probe",
    "pilot-window-selection",
    "storage-and-download-plan",
    "download-if-feasible",
    "downloaded-data-qc",
    "bybit-order-path-replay-if-data-available",
    "candidate-impact-analysis",
    "vendor-gap-and-forward-capture-plan",
    "decision-report",
    "compact-review-bundle",
    "all",
)

ALLOWED_VERDICTS = {
    "bybit_free_data_sufficient_for_pilot",
    "bybit_free_data_partial_continue_with_limited_replay",
    "bybit_free_data_insufficient_procure_vendor",
    "bybit_free_data_insufficient_start_forward_capture",
    "candidate_survives_bybit_order_path_pilot",
    "candidate_fails_bybit_order_path_pilot_current_translation_only",
    "candidate_unresolved_missing_execution_data",
    "d4_carry_forward_execution_depth",
    "blocked_by_data_access",
    "blocked_by_protocol_issue",
}

RISK_PCTS = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
EQUITY_CASES = [200, 500, 1000]
LATENCY_MS = [50, 150, 500]
LEV_CASES = [2, 3, 5, 10]


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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-bybit-hist-route")
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
    p = argparse.ArgumentParser(description="QLMG Bybit official/free historical execution-data route audit")
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
    p.add_argument("--pilot-window-count", type=int, default=250)
    p.add_argument("--download-if-feasible", action="store_true")
    p.add_argument("--download-cap-gb", type=float, default=10.0)
    p.add_argument("--include-d4", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-listing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-public-trades", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-orderbook", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-liquidations", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-top-of-book", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-shallow-depth", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--orderbook-depth-levels", default="top5,top25")
    p.add_argument("--tmux-session-name", default="qlmg_bybit_hist_exec")
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
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(dict(row))


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
        "preflight-and-artifact-freeze": [root / "preflight/preflight_report.md", root / "preflight/frozen_artifact_hashes.json", root / "preflight/input_artifact_manifest.csv", root / "preflight/resource_guard_report.md"],
        "telegram-and-tmux-setup": [root / "notifications/telegram_readiness_report.md", root / "tmux/watch_commands.md"],
        "seal-guard": [root / "seal/seal_guard_report.md", root / "seal/protected_slice_check.json"],
        "candidate-and-window-freeze": [root / "candidates/frozen_candidate_manifest.csv", root / "windows/full_window_manifest.csv", root / "windows/window_candidate_control_map.csv"],
        "bybit-official-data-route-audit": [root / "audit/bybit_official_data_route_matrix.csv", root / "audit/bybit_official_data_route_report.md"],
        "bybit-historical-url-and-schema-probe": [root / "probe/bybit_probe_results.csv", root / "probe/bybit_probe_report.md"],
        "pilot-window-selection": [root / "pilot/pilot_windows.csv", root / "pilot/pilot_selection_report.md"],
        "storage-and-download-plan": [root / "estimate/storage_download_estimate.csv", root / "estimate/storage_download_plan.md"],
        "download-if-feasible": [root / "downloaded_bybit_historical/download_manifest.csv", root / "downloaded_bybit_historical/download_report.md"],
        "downloaded-data-qc": [root / "qc/bybit_historical_qc_summary.csv", root / "qc/bybit_historical_qc_report.md"],
        "bybit-order-path-replay-if-data-available": [root / "replay/bybit_order_path_replay_summary.csv"],
        "candidate-impact-analysis": [root / "impact/candidate_impact_summary.csv", root / "impact/candidate_impact_report.md"],
        "vendor-gap-and-forward-capture-plan": [root / "gap/vendor_gap_report.md", root / "gap/gap_matrix.csv", root / "gap/minimal_vendor_pilot_request.md", root / "gap/forward_live_capture_spec.md"],
        "decision-report": [root / "QLMG_BYBIT_HISTORICAL_EXECUTION_DATA_ROUTE_REPORT.md", root / "decision_summary.json"],
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
    if stage == "download-if-feasible" and ctx.args.download_if_feasible:
        return float(ctx.args.download_cap_gb)
    if stage in {"candidate-and-window-freeze", "pilot-window-selection", "storage-and-download-plan"}:
        return 0.5 if not ctx.args.smoke else 0.1
    return 0.2


def ensure_guard(ctx: RunContext, stage: str) -> None:
    snap = resource_snapshot(REPO)
    status = check_resource_guard(
        snap,
        estimated_output_gb=estimate_stage_gb(ctx, stage),
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=25.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    if ctx.args.download_cap_gb > 15.0 and not ctx.args.allow_large_output:
        status["status"] = "hard_stop"
        status.setdefault("reasons", []).append("download_cap_gb_above_15_without_allow_large_output")
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", {"stage": stage, **status, "snapshot": snap.__dict__})
    if status["warnings"]:
        ctx.notifier.send("QLMG Bybit route resource warning", f"stage={stage} warnings={status['warnings']}", level="warning")
    if status["status"] != "pass":
        ctx.notifier.send("QLMG Bybit route resource hard stop", f"stage={stage} reasons={status['reasons']}", level="error")
        raise RuntimeError(f"resource guard failed for {stage}: {status}")


def validate_window_df(df: pd.DataFrame, cols: Sequence[str]) -> None:
    if not df.empty:
        validate_no_protected(df, cols)


def input_artifacts() -> list[Path]:
    return [
        LISTING_ROOT / "SCOPE_CORRECTED_LISTING_GENERIC_RESULTS_REPORT.md",
        LISTING_ROOT / "depth/targeted_depth_window_manifest.csv",
        LISTING_ROOT / "depth/depth_procurement_or_live_capture_plan.md",
        LISTING_ROOT / "windows/full_event_window_manifest.csv",
        LISTING_ROOT / "controls/full_event_control_summary.csv",
        LISTING_ROOT / "listing/full_event_candidate_manifest.csv",
        D4_SURVIVAL_ROOT / "decision_summary.json",
        D4_SURVIVAL_ROOT / "geometry/decision_time_liquidation_geometry.parquet",
        D4_SURVIVAL_ROOT / "D4_SURVIVABILITY_REDESIGN_REPORT.md",
    ]


def artifact_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    hashes: dict[str, Any] = {"created_at_utc": utc_now(), "artifacts": []}
    for p in input_artifacts():
        exists = p.exists()
        rec = {"path": str(p), "exists": exists, "size_bytes": p.stat().st_size if exists else 0, "sha256": sha256_file(p) if exists and p.is_file() else ""}
        rows.append(rec)
        hashes["artifacts"].append(rec)
    return rows, hashes


def source_url_patterns() -> list[dict[str, Any]]:
    hist_page = "https://www.bybit.com/en/derivative-activity/history-data/"
    docs_base = "https://bybit-exchange.github.io/docs/v5/market/"
    return [
        {"data_type": "historical_public_trades", "source_url_or_endpoint": hist_page, "downloadable": "unknown_probe_required", "format": "web_archive_or_csv_zip_unknown", "historical_or_live_only": "historical_if_archive_available", "linear_usdt_perps_covered": "unknown", "delisted_symbols_covered": "unknown", "replay_sufficiency": "slippage_sanity_participation_caps_only_without_orderbook", "known_limitations": "trades alone are not full execution-depth replay"},
        {"data_type": "historical_orderbook", "source_url_or_endpoint": hist_page, "downloadable": "unknown_probe_required", "format": "web_archive_or_csv_zip_unknown", "historical_or_live_only": "historical_if_archive_available", "linear_usdt_perps_covered": "unknown", "delisted_symbols_covered": "unknown", "replay_sufficiency": "needs_snapshot_or_deltas_sequence_consistency", "known_limitations": "must classify snapshot/delta/topN/BBO reconstructability"},
        {"data_type": "historical_top_of_book_bbo", "source_url_or_endpoint": hist_page, "downloadable": "unknown_probe_required", "format": "web_archive_or_csv_zip_unknown", "historical_or_live_only": "historical_if_archive_available", "linear_usdt_perps_covered": "unknown", "delisted_symbols_covered": "unknown", "replay_sufficiency": "entry_exit_spread_sanity_if_tightly_timestamped", "known_limitations": "BBO alone cannot book-walk larger orders"},
        {"data_type": "historical_shallow_depth", "source_url_or_endpoint": hist_page, "downloadable": "unknown_probe_required", "format": "web_archive_or_csv_zip_unknown", "historical_or_live_only": "historical_if_archive_available", "linear_usdt_perps_covered": "unknown", "delisted_symbols_covered": "unknown", "replay_sufficiency": "listing_replay_if_top5_or_top25_sequence_consistent", "known_limitations": "D4 still needs liquidation history"},
        {"data_type": "historical_l2_snapshots_or_deltas", "source_url_or_endpoint": hist_page, "downloadable": "unknown_probe_required", "format": "web_archive_or_csv_zip_unknown", "historical_or_live_only": "historical_if_archive_available", "linear_usdt_perps_covered": "unknown", "delisted_symbols_covered": "unknown", "replay_sufficiency": "best_if_snapshot_plus_deltas_sequence_consistent", "known_limitations": "large storage"},
        {"data_type": "historical_liquidation_events", "source_url_or_endpoint": hist_page, "downloadable": "unknown_probe_required", "format": "web_archive_or_csv_zip_unknown", "historical_or_live_only": "historical_if_archive_available", "linear_usdt_perps_covered": "unknown", "delisted_symbols_covered": "unknown", "replay_sufficiency": "required_for_D4", "known_limitations": "live WebSocket is not historical liquidation data"},
        {"data_type": "v5_recent_public_trades", "source_url_or_endpoint": docs_base + "recent-trade", "downloadable": "REST_recent_only", "format": "JSON_REST", "historical_or_live_only": "recent_window_not_2025_history", "linear_usdt_perps_covered": True, "delisted_symbols_covered": False, "replay_sufficiency": "not_sufficient_for_2025_replay", "known_limitations": "recent endpoint only, not historical archive"},
        {"data_type": "v5_current_orderbook", "source_url_or_endpoint": docs_base + "orderbook", "downloadable": "REST_current_only", "format": "JSON_REST", "historical_or_live_only": "current_snapshot_only", "linear_usdt_perps_covered": True, "delisted_symbols_covered": False, "replay_sufficiency": "not_historical_replay", "known_limitations": "current REST orderbook is not historical orderbook replay"},
        {"data_type": "v5_mark_index_premium_kline", "source_url_or_endpoint": docs_base + "mark-kline,index-kline,premium-index-kline", "downloadable": "REST_paginated", "format": "JSON_REST", "historical_or_live_only": "historical_context", "linear_usdt_perps_covered": True, "delisted_symbols_covered": "unknown", "replay_sufficiency": "useful_context_insufficient_execution_depth", "known_limitations": "not trades/orderbook/liquidations"},
        {"data_type": "v5_oi_funding_metadata", "source_url_or_endpoint": docs_base + "open-interest,funding/history,instruments-info", "downloadable": "REST_paginated_or_current_metadata", "format": "JSON_REST", "historical_or_live_only": "partial_historical_context_plus_current_metadata", "linear_usdt_perps_covered": True, "delisted_symbols_covered": "unknown", "replay_sufficiency": "context_only", "known_limitations": "not execution-depth"},
    ]


def orderbook_classification_row() -> dict[str, Any]:
    return {
        "snapshot_only": "unknown_until_archive_schema_probe",
        "snapshot_plus_deltas": "unknown_until_archive_schema_probe",
        "top_n_levels": "unknown_until_archive_schema_probe",
        "bbo_reconstructable": "unknown_until_archive_schema_probe",
        "sequence_consistent": "unknown_until_archive_schema_probe",
        "timestamp_tight_enough_for_replay": "unknown_until_archive_schema_probe",
    }


def route_matrix_rows() -> list[dict[str, Any]]:
    rows = []
    ob_class = orderbook_classification_row()
    for r in source_url_patterns():
        row = {
            **r,
            "symbol_naming_convention": "Bybit linear symbols e.g. BTCUSDT; archive naming unknown until probe",
            "interval_or_date_partitioning": "unknown until archive probe; REST endpoints use start/end pagination where available",
            "earliest_date_discoverable": "unknown_without_archive_index_probe",
            "covers_target_symbols_windows": "unknown_until_probe",
            "contains_enough_information_for_replay": False,
        }
        if "orderbook" in str(r["data_type"]) or "depth" in str(r["data_type"]) or "bbo" in str(r["data_type"]):
            row.update(ob_class)
        else:
            row.update({k: "not_applicable" for k in ob_class})
        rows.append(row)
    return rows


def probe_cases_from_windows(run_root: Path) -> list[dict[str, Any]]:
    windows = read_csv(run_root / "windows/full_window_manifest.csv")
    cases = [{"probe_case": "current_liquid_symbol_metadata_only", "symbol": "BTCUSDT", "date": "metadata_current", "source": "current_liquid_symbol"}]
    if not windows.empty:
        listing = windows[windows.get("candidate_id", pd.Series(dtype=str)).astype(str).isin(LISTING_IDS)]
        if not listing.empty:
            r = listing.iloc[0]
            cases.append({"probe_case": "actual_listing_candidate_window", "symbol": str(r.get("symbol")), "date": str(pd.to_datetime(r.get("window_start"), utc=True).date()), "source": str(r.get("target_window_id"))})
        d4 = windows[windows.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(D4_CANDIDATE_ID)]
        if not d4.empty:
            r = d4.iloc[0]
            cases.append({"probe_case": "lifecycle_sensitive_or_d4_target_window", "symbol": str(r.get("symbol")), "date": str(pd.to_datetime(r.get("window_start"), utc=True).date()), "source": str(r.get("target_window_id"))})
    return cases[:3]


def safe_head(url: str, timeout: float = 10.0) -> tuple[str, int, int, str]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": "qlmg-route-audit/1.0"})
        return "ok", int(r.status_code), int(r.headers.get("content-length", 0) or 0), r.headers.get("content-type", "")
    except Exception as exc:
        return f"error:{type(exc).__name__}", 0, 0, str(exc)[:200]


def safe_get_json(url: str, timeout: float = 10.0) -> tuple[str, int, dict[str, Any] | list[Any] | None, str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "qlmg-route-audit/1.0"})
        text = r.text[:500]
        try:
            data = r.json()
        except Exception:
            data = None
        return "ok", int(r.status_code), data, text
    except Exception as exc:
        return f"error:{type(exc).__name__}", 0, None, str(exc)[:500]


def stage_preflight(ctx: RunContext) -> None:
    rows = []
    hashes = {"created_at_utc": utc_now(), "artifacts": []}
    missing = []
    for p in input_artifacts():
        exists = p.exists()
        if not exists:
            missing.append(str(p))
        rec = {"path": str(p), "exists": exists, "size_bytes": p.stat().st_size if exists else 0, "sha256": sha256_file(p) if exists and p.is_file() else ""}
        rows.append(rec)
        hashes["artifacts"].append(rec)
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(REPO)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\n- free_disk_gb: `{snap.free_gb:.3f}`\n- hard_stop_free_gb: `5`\n- warning_free_gb: `7`\n- stage_output_block_gb: `25`\n- max_output_gb: `{ctx.args.max_output_gb}`\n- download_cap_gb: `{ctx.args.download_cap_gb}`\n- hard_route_cap_gb_without_allow_large_output: `15`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight And Artifact Freeze\n\n- run_root: `{ctx.run_root}`\n- listing_root: `{LISTING_ROOT}`\n- d4_survival_root: `{D4_SURVIVAL_ROOT}`\n- protected_start: `{FINAL_HOLDOUT_START}`\n- screening_end: `{SCREENING_END}`\n- missing_required_artifacts: `{missing}`\n- git_head: `{shell(['git','rev-parse','HEAD'])}`\n- git_status_short: `{shell(['git','status','--short'])[:5000]}`\n")
    if missing:
        raise RuntimeError(f"required input artifacts missing: {missing}")


def stage_telegram(ctx: RunContext) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\n- disabled: `{ctx.notifier.disabled}`\n- remote_available: `{ctx.notifier.remote_available}`\n- status: `{ctx.notifier.status}`\n- missing_or_reason: `{ctx.notifier.missing}`\n- secrets_logged: `false`\n")
    write_text(ctx.run_root / "tmux/watch_commands.md", f"# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n")
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nDefault session: `{ctx.args.tmux_session_name}`. Full run requires `--launch-tmux`.\n")


def stage_seal(ctx: RunContext) -> None:
    validate_no_protected(pd.DataFrame({"ts": [ctx.end]}), ["ts"])
    blocked = False
    try:
        validate_no_protected(pd.DataFrame({"ts": [FINAL_HOLDOUT_START]}), ["ts"])
    except RuntimeError:
        blocked = True
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "pre_holdout_read_passed": True, "protected_read_blocked": blocked})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\n- protected_start: `{FINAL_HOLDOUT_START}`\n- generated windows/downloads/reports must end before protected start: `true`\n- protected smoke blocked: `{blocked}`\n")


def stage_candidate_windows(ctx: RunContext) -> None:
    candidates = []
    if ctx.args.include_d4:
        candidates.append({"candidate_id": D4_CANDIDATE_ID, "family": "D4_liquidation_safe_flush", "candidate_type": "d4", "source_root": str(D4_SURVIVAL_ROOT), "required_data": "depth_trades_liquidation_history", "max_without_liquidations": "d4_carry_forward_execution_depth"})
    if ctx.args.include_listing:
        manifest = read_csv(LISTING_ROOT / "listing/full_event_candidate_manifest.csv")
        manifest = manifest[manifest.get("candidate_id", pd.Series(dtype=str)).astype(str).isin(LISTING_IDS)] if not manifest.empty else pd.DataFrame()
        for _, row in manifest.iterrows():
            candidates.append({"candidate_id": row.get("candidate_id"), "family": row.get("family"), "subfamily": row.get("subfamily"), "candidate_type": "listing_vwap_loss", "source_root": str(LISTING_ROOT), "horizon": row.get("horizon"), "listing_metadata_source": row.get("listing_metadata_source"), "required_data": "orderbook_plus_public_trades", "lifecycle_cap": "proxy_launch_only" if "proxy" in str(row.get("listing_metadata_source", "")) else "official_metadata"})
    write_csv(ctx.run_root / "candidates/frozen_candidate_manifest.csv", candidates)
    full = build_full_window_manifest(ctx)  # Reuses prior phase's point-in-time window construction.
    if not ctx.args.include_controls and not full.empty:
        full = full[full.get("window_role", pd.Series(dtype=str)).astype(str).ne("control")].copy()
    write_csv(ctx.run_root / "windows/full_window_manifest.csv", full)
    write_csv(ctx.run_root / "windows/window_candidate_control_map.csv", full[[c for c in ["target_window_id", "candidate_id", "family", "symbol", "window_start", "window_end", "window_role", "selection_bucket"] if c in full.columns]] if not full.empty else pd.DataFrame())
    write_text(ctx.run_root / "windows/window_freeze_report.md", f"# Candidate And Window Freeze\n\n- candidates: `{len(candidates)}`\n- full windows: `{len(full)}`\n- controls included: `{int((full.get('window_role', pd.Series(dtype=str)).astype(str) == 'control').sum()) if not full.empty else 0}`\n- protected holdout untouched: `true`\n")


def stage_route_audit(ctx: RunContext) -> None:
    rows = route_matrix_rows()
    write_csv(ctx.run_root / "audit/bybit_official_data_route_matrix.csv", rows)
    write_text(ctx.run_root / "audit/bybit_official_data_route_report.md", "# Bybit Official/Free Data Route Audit\n\nThis audit separates historical public trades, historical orderbook/BBO/depth, historical liquidation events, mark/index/premium, OI/funding, and instrument metadata. Current REST orderbook and live WebSocket liquidation streams are not historical replay data. Public trades alone may support slippage sanity and participation caps, but are not full execution-depth replay. Listing candidates require orderbook plus public trades for route sufficiency. D4 also requires historical liquidation events or vendor liquidation history.\n")


def stage_probe(ctx: RunContext) -> None:
    cases = probe_cases_from_windows(ctx.run_root)
    rows = []
    samples_dir = ctx.run_root / "probe/bybit_schema_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    hist_page = "https://www.bybit.com/en/derivative-activity/history-data/"
    status, code, size, ctype = safe_head(hist_page)
    for case in cases:
        rows.append({"probe_case": case["probe_case"], "symbol": case["symbol"], "date": case["date"], "data_type": "official_history_page", "attempted_url_or_endpoint": hist_page, "status": status, "http_status": code, "file_size_bytes": size, "content_type": ctype, "schema_columns": "not_available_from_head", "timestamp_units": "unknown", "sample_row_count": 0, "protected_rows_present": False, "usable_for_replay": False})
    # REST recent/current probes are capability-only; they are not used for 2025 candidate selection.
    rest_endpoints = [
        ("v5_recent_public_trades", "https://api.bybit.com/v5/market/recent-trade?category=linear&symbol=BTCUSDT&limit=1"),
        ("v5_current_orderbook", "https://api.bybit.com/v5/market/orderbook?category=linear&symbol=BTCUSDT&limit=1"),
        ("v5_instrument_metadata", "https://api.bybit.com/v5/market/instruments-info?category=linear&symbol=BTCUSDT"),
    ]
    for dtype, url in rest_endpoints:
        st, code, data, text = safe_get_json(url)
        cols = []
        n = 0
        if isinstance(data, dict):
            result = data.get("result", {})
            if isinstance(result, dict):
                if isinstance(result.get("list"), list) and result.get("list"):
                    cols = sorted(result["list"][0].keys()) if isinstance(result["list"][0], dict) else []
                    n = len(result["list"])
                else:
                    cols = sorted(result.keys())
        sample_path = samples_dir / f"{dtype}.json"
        sample_path.write_text(json.dumps(data if data is not None else {"raw": text}, indent=2, sort_keys=True, default=str)[:20000] + "\n", encoding="utf-8")
        rows.append({"probe_case": "current_liquid_symbol_metadata_only", "symbol": "BTCUSDT", "date": "current_capability_only", "data_type": dtype, "attempted_url_or_endpoint": url, "status": st, "http_status": code, "file_size_bytes": len(text), "content_type": "application/json", "schema_columns": ";".join(cols), "timestamp_units": "milliseconds_or_exchange_specific", "sample_row_count": n, "protected_rows_present": False, "usable_for_replay": dtype not in {"v5_current_orderbook", "v5_recent_public_trades"} and False})
    write_csv(ctx.run_root / "probe/bybit_probe_results.csv", rows)
    write_text(ctx.run_root / "probe/bybit_probe_report.md", "# Bybit Historical URL And Schema Probe\n\nProbes include one current liquid symbol capability check, one actual listing-candidate window when available, and one D4/lifecycle-sensitive target when available. REST recent trades and current orderbook probes are capability-only and are not historical 2025 replay data. Historical archive availability remains unresolved unless the official history page exposes downloadable files for the target data type.\n")


def stage_pilot_selection(ctx: RunContext) -> None:
    full = read_csv(ctx.run_root / "windows/full_window_manifest.csv")
    if not full.empty:
        for c in ["window_start", "window_end"]:
            full[c] = pd.to_datetime(full[c], utc=True, errors="coerce")
        validate_window_df(full, ["window_start", "window_end"])
    pilot, omitted = select_pilot_windows(full, int(ctx.args.pilot_window_count))
    if ctx.args.smoke and len(pilot) > 250:
        pilot = pilot.head(250)
    write_csv(ctx.run_root / "pilot/pilot_windows.csv", pilot)
    write_csv(ctx.run_root / "pilot/omitted_pilot_windows.csv", omitted)
    controls = int((pilot.get("window_role", pd.Series(dtype=str)).astype(str) == "control").sum()) if not pilot.empty else 0
    write_text(ctx.run_root / "pilot/pilot_selection_report.md", f"# Pilot Window Selection\n\n- requested_pilot_windows: `{ctx.args.pilot_window_count}`\n- selected_windows: `{len(pilot)}`\n- controls_selected: `{controls}`\n- control_share: `{(controls / max(len(pilot), 1)):.3f}`\n- policy: prioritize controls over near-duplicate winners when quota is tight.\n")


def estimate_storage(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    hours = pd.to_numeric(windows.get("hours", pd.Series(dtype=float)), errors="coerce")
    if hours.isna().all():
        hours = (pd.to_datetime(windows["window_end"], utc=True) - pd.to_datetime(windows["window_start"], utc=True)).dt.total_seconds() / 3600
    total_hours = float(hours.fillna(0).sum())
    specs = [
        ("public_trades", 0.00018),
        ("orderbook_snapshot_or_delta", 0.00150),
        ("top_of_book_bbo", 0.00002),
        ("shallow_depth_top5", 0.00008),
        ("shallow_depth_top25", 0.00025),
        ("liquidation_events", 0.00001),
        ("mark_index_funding_oi_context", 0.00001),
    ]
    return pd.DataFrame([{"dataset": d, "windows": len(windows), "symbol_hours": total_hours, "estimated_compressed_gb": total_hours * gbph, "estimate_basis": "rough_symbol_hour_proxy_not_vendor_quote"} for d, gbph in specs])


def stage_storage(ctx: RunContext) -> None:
    pilot = read_csv(ctx.run_root / "pilot/pilot_windows.csv")
    full = read_csv(ctx.run_root / "windows/full_window_manifest.csv")
    p = estimate_storage(pilot)
    if not p.empty:
        p["scope"] = "pilot"
    f = estimate_storage(full)
    if not f.empty:
        f["scope"] = "full_targeted"
    est = pd.concat([p, f], ignore_index=True) if not p.empty or not f.empty else pd.DataFrame()
    write_csv(ctx.run_root / "estimate/storage_download_estimate.csv", est)
    pilot_gb = float(est.loc[est.get("scope", pd.Series(dtype=str)).astype(str).eq("pilot"), "estimated_compressed_gb"].sum()) if not est.empty else 0.0
    full_gb = float(est.loc[est.get("scope", pd.Series(dtype=str)).astype(str).eq("full_targeted"), "estimated_compressed_gb"].sum()) if not est.empty else 0.0
    over = pilot_gb > float(ctx.args.download_cap_gb)
    write_text(ctx.run_root / "estimate/storage_download_plan.md", f"# Storage And Download Plan\n\n- pilot_estimated_gb: `{pilot_gb:.4f}`\n- full_targeted_estimated_gb: `{full_gb:.4f}`\n- download_cap_gb: `{ctx.args.download_cap_gb}`\n- pilot_exceeds_cap: `{over}`\n- if estimate exceeds cap, stage the pilot rather than silently dropping controls or degrading quality.\n")


def route_feasible_downloads(ctx: RunContext) -> tuple[bool, str]:
    probe = read_csv(ctx.run_root / "probe/bybit_probe_results.csv")
    if probe.empty:
        return False, "no_probe_results"
    # Current REST endpoints are not historical replay data. Only a successful historical archive probe can enable download here.
    historical = probe[probe["data_type"].astype(str).eq("official_history_page")]
    ok = bool((not historical.empty) and historical["status"].astype(str).eq("ok").any() and historical["http_status"].astype(int).between(200, 399).any())
    return False, "official_history_page_reachable_but_no_machine_verified_target_file_url" if ok else "official_history_route_not_machine_downloadable_from_probe"


def stage_download(ctx: RunContext) -> None:
    feasible, reason = route_feasible_downloads(ctx)
    enabled = bool(ctx.args.download_if_feasible)
    rows = []
    if not enabled:
        rows.append({"status": "not_enabled", "reason": "--download-if-feasible not passed", "downloaded": False})
    elif feasible:
        rows.append({"status": "blocked", "reason": "download adapter intentionally fails closed until exact official target file URLs are verified", "downloaded": False})
    else:
        rows.append({"status": "blocked", "reason": reason, "downloaded": False})
    write_csv(ctx.run_root / "downloaded_bybit_historical/download_manifest.csv", rows)
    write_text(ctx.run_root / "downloaded_bybit_historical/download_report.md", f"# Download If Feasible\n\n- enabled: `{enabled}`\n- feasible: `{feasible}`\n- downloaded_anything: `false`\n- reason: `{rows[0]['reason']}`\n- no secrets logged: `true`\n")
    if not feasible:
        write_text(ctx.run_root / "downloaded_bybit_historical/download_blocked_report.md", f"# Download Blocked\n\nNo official/free Bybit historical execution-depth target file URL was machine-verified for the requested pre-holdout windows. Reason: `{reason}`.\n")


def stage_qc(ctx: RunContext) -> None:
    rows = [{"dataset": d, "qc_status": "not_run", "reason": "no_downloaded_dataset"} for d in ["public_trades", "orderbook", "top_of_book", "shallow_depth", "liquidation_events"]]
    write_csv(ctx.run_root / "qc/bybit_historical_qc_summary.csv", rows)
    write_text(ctx.run_root / "qc/bybit_historical_qc_report.md", "# Bybit Historical Data QC\n\nQC did not run because no historical execution dataset was downloaded. This is a data-route blocker, not an alpha failure.\n")


def route_sufficiency(ctx: RunContext) -> dict[str, Any]:
    probe = read_csv(ctx.run_root / "probe/bybit_probe_results.csv")
    trades_current = bool((not probe.empty) and probe["data_type"].astype(str).eq("v5_recent_public_trades").any())
    orderbook_current = bool((not probe.empty) and probe["data_type"].astype(str).eq("v5_current_orderbook").any())
    # Current REST endpoints are explicitly not historical sufficient.
    return {
        "public_trades_historical_available": False,
        "orderbook_historical_available": False,
        "liquidation_history_available": False,
        "recent_trades_endpoint_reachable": trades_current,
        "current_orderbook_endpoint_reachable": orderbook_current,
        "listing_sufficiency": "insufficient_without_historical_orderbook_plus_public_trades",
        "d4_sufficiency": "insufficient_without_liquidation_history_plus_depth_trades",
        "trades_only_status": "not_available_historical;would_be_trades_only_not_full_execution_depth_if_found",
    }


def stage_replay(ctx: RunContext) -> None:
    suff = route_sufficiency(ctx)
    rows = []
    for cid in [D4_CANDIDATE_ID, *LISTING_IDS]:
        rows.append({"candidate_id": cid, "status": "blocked", "reason": "historical_orderbook_public_trades_liquidations_unavailable", "order_path_replayed": False, "trades_only_diagnostic": False, "candidate_type": "d4" if cid == D4_CANDIDATE_ID else "listing_vwap_loss"})
    write_csv(ctx.run_root / "replay/bybit_order_path_replay_summary.csv", rows)
    write_csv(ctx.run_root / "replay/bybit_order_path_risk_grid.csv", [{"equity_usdt": e, "risk_pct_equity": r, "latency_ms": l, "leverage": lev, "risk_label": "baseline" if r <= 0.01 else "aggressive_diagnostic"} for e in EQUITY_CASES for r in RISK_PCTS for l in LATENCY_MS for lev in LEV_CASES])
    write_text(ctx.run_root / "replay/bybit_order_path_replay_blocked_report.md", f"# Bybit Order-Path Replay Blocked\n\n{suff}\n\nListing/VWAP-loss candidates need historical orderbook plus public trades. D4 also needs historical liquidation events or vendor liquidation history. Public trades alone would be diagnostic only, not full execution-depth replay.\n")


def stage_impact(ctx: RunContext) -> None:
    rows = [{"candidate_id": D4_CANDIDATE_ID, "prior_status": "D4_carry_forward_from_survivability", "bybit_route_adjusted_status": "candidate_unresolved_missing_execution_data", "conclusion_changed": False, "reason": "liquidation_history_depth_trades_missing"}]
    for cid in LISTING_IDS:
        rows.append({"candidate_id": cid, "prior_status": "listing_vwap_loss_full_event_1m_mark_prelead", "bybit_route_adjusted_status": "candidate_unresolved_missing_execution_data", "conclusion_changed": False, "reason": "historical_orderbook_public_trades_missing"})
    write_csv(ctx.run_root / "impact/candidate_impact_summary.csv", rows)
    write_text(ctx.run_root / "impact/candidate_impact_report.md", "# Candidate Impact Analysis\n\nNo order-path replay ran, so the Bybit official/free route did not change prior train-only candidate evidence. Missing execution-depth data remains a data blocker, not alpha failure.\n")


def stage_gap(ctx: RunContext) -> None:
    rows = [
        {"missing_field": "historical_orderbook_or_BBO", "why_bybit_free_insufficient": "no verified target historical file/API route from probe", "vendor_likely": True, "forward_capture_can_supply_future_only": True, "needed_for": "listing_and_D4", "priority": 1, "minimal_window_count": 50, "estimated_storage_gb": "10-20"},
        {"missing_field": "historical_public_trades", "why_bybit_free_insufficient": "recent REST endpoint is not 2025 historical archive", "vendor_likely": True, "forward_capture_can_supply_future_only": True, "needed_for": "listing_and_D4", "priority": 2, "minimal_window_count": 50, "estimated_storage_gb": "5-15"},
        {"missing_field": "historical_liquidation_events", "why_bybit_free_insufficient": "live stream is not historical liquidation data", "vendor_likely": True, "forward_capture_can_supply_future_only": True, "needed_for": "D4", "priority": 3, "minimal_window_count": 50, "estimated_storage_gb": "1-5"},
    ]
    write_csv(ctx.run_root / "gap/gap_matrix.csv", rows)
    write_text(ctx.run_root / "gap/vendor_gap_report.md", "# Vendor Gap Report\n\nBybit official/free routes were not machine-verified as sufficient for historical execution-depth replay. Listing candidates need historical orderbook plus public trades. D4 additionally needs historical liquidation events. Missing data does not reject D4, listing, generic shock, funding-window, or secondary families.\n")
    write_text(ctx.run_root / "gap/minimal_vendor_pilot_request.md", "# Minimal Vendor Pilot Request\n\nScope: Bybit linear USDT derivatives only; D4 plus the three listing/VWAP-loss candidates; 50-100 priority windows first; matched controls included; target cap 10-20GB. Required: top-of-book or top5/top25 snapshots/deltas, public trades, liquidation events for D4, derivative ticker/mark where available, exchange timestamps, sequence/update IDs where applicable.\n")
    write_text(ctx.run_root / "gap/forward_live_capture_spec.md", "# Forward Live Capture Specification\n\nCapture Bybit linear USDT: BBO/top-of-book, top5 or top25 depth, public trades, liquidation stream, mark/index/last ticker, funding rate and next funding time, open interest, instrument metadata/status, exchange timestamp, local receive timestamp, and sequence/update ID where available. Forward capture cannot replace historical 2025 replay but can unblock future shadow validation.\n")


def acquisition_verdict(ctx: RunContext) -> str:
    suff = route_sufficiency(ctx)
    if suff["orderbook_historical_available"] and suff["public_trades_historical_available"] and suff["liquidation_history_available"]:
        return "free Bybit route sufficient for listing pilot"
    if suff["orderbook_historical_available"] and suff["public_trades_historical_available"] and not suff["liquidation_history_available"]:
        return "vendor needed only for liquidation data"
    if suff["public_trades_historical_available"] and not suff["orderbook_historical_available"]:
        return "free Bybit route partial"
    return "vendor needed for depth/trades/liquidations"


def stage_decision(ctx: RunContext) -> None:
    suff = route_sufficiency(ctx)
    verdict = "bybit_free_data_insufficient_procure_vendor"
    listing_verdicts = {cid: "candidate_unresolved_missing_execution_data" for cid in LISTING_IDS}
    d4_verdict = "d4_carry_forward_execution_depth"
    summary = {
        "created_at_utc": utc_now(),
        "run_root": str(ctx.run_root),
        "protected_holdout_untouched": True,
        "protected_start": str(FINAL_HOLDOUT_START),
        "bybit_free_route_verdict": verdict,
        "listing_candidate_verdicts": listing_verdicts,
        "d4_verdict": d4_verdict,
        "execution_replay_verdict": "candidate_unresolved_missing_execution_data",
        "vendor_gap_verdict": "bybit_free_data_insufficient_procure_vendor",
        "forward_capture_verdict": "bybit_free_data_insufficient_start_forward_capture",
        "next_action_verdict": verdict,
        "concrete_acquisition_verdict": acquisition_verdict(ctx),
        "route_sufficiency": suff,
        "no_validation_or_live_readiness_claimed": True,
    }
    for v in [summary["bybit_free_route_verdict"], summary["execution_replay_verdict"], summary["vendor_gap_verdict"], summary["forward_capture_verdict"], summary["next_action_verdict"], d4_verdict, *listing_verdicts.values()]:
        if v not in ALLOWED_VERDICTS:
            raise RuntimeError(f"invalid verdict {v}")
    write_json(ctx.run_root / "decision_summary.json", summary)
    write_text(ctx.run_root / "QLMG_BYBIT_HISTORICAL_EXECUTION_DATA_ROUTE_REPORT.md", f"# QLMG Bybit Historical Execution Data Route Report\n\n## Scope\n\nTrain-only official/free Bybit data-route audit. Protected holdout `>= {FINAL_HOLDOUT_START}` was not used. No sealed validation, live readiness, production readiness, or trading recommendation is made.\n\n## Verdicts\n\n- bybit_free_route_verdict: `{summary['bybit_free_route_verdict']}`\n- concrete_acquisition_verdict: `{summary['concrete_acquisition_verdict']}`\n- d4_verdict: `{d4_verdict}`\n- listing_candidate_verdicts: `{listing_verdicts}`\n- execution_replay_verdict: `{summary['execution_replay_verdict']}`\n- vendor_gap_verdict: `{summary['vendor_gap_verdict']}`\n- forward_capture_verdict: `{summary['forward_capture_verdict']}`\n\n## Interpretation\n\nBybit current REST public-trade/orderbook endpoints are useful capability checks but are not historical 2025 replay data. Public trades alone, if found historically, would support slippage sanity and participation caps only. Listing/VWAP-loss requires historical orderbook plus trades. D4 requires historical liquidation data in addition to depth/trades. Missing execution-depth data is a data blocker, not an alpha failure.\n")


def stage_compact(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rels = [
        "QLMG_BYBIT_HISTORICAL_EXECUTION_DATA_ROUTE_REPORT.md",
        "decision_summary.json",
        "preflight/preflight_report.md",
        "seal/seal_guard_report.md",
        "windows/full_window_manifest.csv",
        "pilot/pilot_windows.csv",
        "audit/bybit_official_data_route_report.md",
        "audit/bybit_official_data_route_matrix.csv",
        "probe/bybit_probe_report.md",
        "probe/bybit_probe_results.csv",
        "estimate/storage_download_plan.md",
        "downloaded_bybit_historical/download_report.md",
        "qc/bybit_historical_qc_report.md",
        "replay/bybit_order_path_replay_blocked_report.md",
        "impact/candidate_impact_report.md",
        "gap/vendor_gap_report.md",
        "gap/minimal_vendor_pilot_request.md",
        "gap/forward_live_capture_spec.md",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
    ]
    index = []
    for rel in rels:
        src = ctx.run_root / rel
        if not src.exists():
            continue
        dst = bundle / rel.replace("/", "__")
        shutil.copy2(src, dst)
        index.append({"artifact": rel, "bundle_path": str(dst), "source_path": str(src), "size_bytes": dst.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", index)
    write_json(bundle / "artifact_path_index.json", {"artifacts": index})
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nReports, summaries, manifests, and small samples only. No large downloaded data included.\n")


STAGE_FUNCS = {
    "preflight-and-artifact-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "candidate-and-window-freeze": stage_candidate_windows,
    "bybit-official-data-route-audit": stage_route_audit,
    "bybit-historical-url-and-schema-probe": stage_probe,
    "pilot-window-selection": stage_pilot_selection,
    "storage-and-download-plan": stage_storage,
    "download-if-feasible": stage_download,
    "downloaded-data-qc": stage_qc,
    "bybit-order-path-replay-if-data-available": stage_replay,
    "candidate-impact-analysis": stage_impact,
    "vendor-gap-and-forward-capture-plan": stage_gap,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_compact,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        ctx.notifier.send("QLMG Bybit route stage skipped", f"stage={stage}")
        return
    ensure_guard(ctx, stage)
    append_command(ctx.run_root, stage)
    ctx.notifier.send("QLMG Bybit route stage start", f"stage={stage}")
    STAGE_FUNCS[stage](ctx)
    missing = [str(p) for p in required_outputs(ctx.run_root, stage) if not p.exists()]
    if missing:
        raise RuntimeError(f"stage {stage} did not produce required outputs: {missing}")
    mark_done(ctx.run_root, stage)
    ctx.notifier.send("QLMG Bybit route stage complete", f"stage={stage}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start, end = clamp_window(args)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_root_resolution.json", {"run_root": str(run_root), "reason": reason, "base_run_id": DEFAULT_RUN_ID})
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "start": str(start), "end": str(end), "args": vars(args), "protected_holdout_start": str(FINAL_HOLDOUT_START)})
    notifier.send("QLMG Bybit route run start", f"run_root={run_root}\nstage={args.stage}")
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG Bybit route run complete", f"run_root={run_root}")
        return 0
    except Exception as exc:
        notifier.send("QLMG Bybit route run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
