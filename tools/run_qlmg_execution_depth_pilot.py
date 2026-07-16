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
DEFAULT_RUN_ID = "phase_qlmg_execution_depth_pilot_20260628_v1"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
D4_AUDIT_ROOT = RESULTS_ROOT / "phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927"
D4_CANDIDATE_ID = "D4__b4c9487fe82c"
DEFAULT_SEED = 20260628
GB = 1024**3

LISTING_IDS = [
    "new_perp_listing_event_study__589a8c85c943",
    "new_perp_listing_event_study__9dc07cfc405c",
    "new_perp_listing_event_study__b1a3735d5092",
]

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "candidate-and-window-freeze",
    "official-listing-lifecycle-audit",
    "depth-trade-liquidation-source-capability-audit",
    "pilot-window-selection",
    "storage-and-cost-estimate",
    "free-bybit-data-download-if-enabled",
    "vendor-data-download-if-enabled",
    "depth-trade-data-qc",
    "order-path-replay",
    "candidate-impact-analysis",
    "full-targeted-data-plan",
    "decision-report",
    "compact-review-bundle",
    "all",
)

FINAL_STATUSES = {
    "candidate_survives_depth_pilot",
    "candidate_fails_depth_pilot_current_translation_only",
    "pilot_depth_data_unavailable_procure_vendor",
    "pilot_depth_data_unavailable_start_forward_capture",
    "d4_carry_forward_execution_depth",
    "blocked_by_data_access",
    "blocked_by_protocol_issue",
}

RISK_PCTS = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20]


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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-depth-pilot")
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
    p = argparse.ArgumentParser(description="QLMG execution-depth pilot, train-only")
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
    p.add_argument("--pilot-window-count", type=int, default=250)
    p.add_argument("--download-free-bybit-data", action="store_true")
    p.add_argument("--download-vendor-data-if-configured", action="store_true")
    p.add_argument("--vendor-source", default="")
    p.add_argument("--depth-download-cap-gb", type=float, default=15.0)
    p.add_argument("--include-d4", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-listing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-liquidations", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-public-trades", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-top-of-book", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-shallow-depth", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tmux-session-name", default="qlmg_depth_pilot")
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
        "official-listing-lifecycle-audit": [root / "lifecycle/listing_lifecycle_audit.csv", root / "lifecycle/listing_lifecycle_audit_report.md"],
        "depth-trade-liquidation-source-capability-audit": [root / "sources/depth_trade_liquidation_source_capability_matrix.csv", root / "sources/source_capability_report.md"],
        "pilot-window-selection": [root / "windows/pilot_window_manifest.csv", root / "windows/omitted_window_report.csv", root / "windows/window_selection_report.md"],
        "storage-and-cost-estimate": [root / "storage/storage_estimate.csv", root / "storage/storage_estimate_report.md"],
        "free-bybit-data-download-if-enabled": [root / "downloaded_free_bybit/download_manifest.csv", root / "downloaded_free_bybit/download_report.md"],
        "vendor-data-download-if-enabled": [root / "downloaded_vendor/download_manifest.csv", root / "downloaded_vendor/download_report.md"],
        "depth-trade-data-qc": [root / "qc/depth_trade_data_qc_summary.csv", root / "qc/depth_trade_data_qc_report.md"],
        "order-path-replay": [root / "replay/order_path_replay_summary.csv", root / "replay/order_path_replay_blocked_report.md"],
        "candidate-impact-analysis": [root / "impact/candidate_impact_summary.csv", root / "impact/candidate_impact_report.md"],
        "full-targeted-data-plan": [root / "data_plan/full_targeted_data_plan.csv", root / "depth/depth_procurement_or_live_capture_plan.md"],
        "decision-report": [root / "QLMG_EXECUTION_DEPTH_PILOT_REPORT.md", root / "decision_summary.json"],
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
    if stage in {"free-bybit-data-download-if-enabled", "vendor-data-download-if-enabled"}:
        return float(ctx.args.depth_download_cap_gb) if (ctx.args.download_free_bybit_data or ctx.args.download_vendor_data_if_configured) else 0.1
    if stage in {"candidate-and-window-freeze", "pilot-window-selection", "storage-and-cost-estimate"}:
        return 0.5 if not ctx.args.smoke else 0.1
    return 0.2


def ensure_guard(ctx: RunContext, stage: str) -> None:
    snap = resource_snapshot(REPO)
    status = check_resource_guard(
        snap,
        estimated_output_gb=estimate_stage_gb(ctx, stage),
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=40.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", {"stage": stage, **status, "snapshot": snap.__dict__})
    if status["warnings"]:
        ctx.notifier.send("QLMG depth pilot resource warning", f"stage={stage} warnings={status['warnings']}", level="warning")
    if status["status"] != "pass":
        ctx.notifier.send("QLMG depth pilot resource hard stop", f"stage={stage} reasons={status['reasons']}", level="error")
        raise RuntimeError(f"resource guard failed for {stage}: {status}")


def ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(value, utc=True))


def validate_window_df(df: pd.DataFrame, cols: Sequence[str]) -> None:
    if not df.empty:
        validate_no_protected(df, cols)


def stable_id(*parts: Any, prefix: str = "w") -> str:
    payload = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def path_inventory(path: Path) -> dict[str, Any]:
    exists = path.exists()
    files = [] if not exists else [p for p in path.rglob("*") if p.is_file()]
    total = 0
    for p in files[:20000]:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return {"path": str(path), "exists": exists, "file_count": len(files), "sampled_size_gb": total / GB}


def input_artifacts() -> list[Path]:
    return [
        LISTING_ROOT / "SCOPE_CORRECTED_LISTING_GENERIC_RESULTS_REPORT.md",
        LISTING_ROOT / "depth/targeted_depth_window_manifest.csv",
        LISTING_ROOT / "depth/depth_procurement_or_live_capture_plan.md",
        LISTING_ROOT / "windows/full_event_window_manifest.csv",
        LISTING_ROOT / "controls/full_event_control_summary.csv",
        LISTING_ROOT / "listing/full_event_candidate_manifest.csv",
        LISTING_ROOT / "listing/full_event_listing_events.parquet",
        LISTING_ROOT / "decision_summary.json",
        D4_SURVIVAL_ROOT / "decision_summary.json",
        D4_SURVIVAL_ROOT / "geometry/decision_time_liquidation_geometry.parquet",
        D4_SURVIVAL_ROOT / "sizing/liquidation_safe_sizing_summary.csv",
        D4_SURVIVAL_ROOT / "D4_SURVIVABILITY_REDESIGN_REPORT.md",
    ]


def artifact_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {"created_at_utc": utc_now(), "artifacts": []}
    for p in input_artifacts():
        exists = p.exists()
        rec = {"path": str(p), "exists": exists, "size_bytes": p.stat().st_size if exists else 0, "sha256": sha256_file(p) if exists and p.is_file() else ""}
        rows.append(rec)
        hashes["artifacts"].append(rec)
    return rows, hashes


def load_listing_candidate_manifest() -> pd.DataFrame:
    df = read_csv(LISTING_ROOT / "listing/full_event_candidate_manifest.csv")
    if not df.empty:
        df = df[df.get("candidate_id", pd.Series(dtype=str)).astype(str).isin(LISTING_IDS)].copy()
    return df


def load_listing_windows() -> pd.DataFrame:
    df = read_csv(LISTING_ROOT / "windows/full_event_window_manifest.csv")
    if df.empty:
        return df
    for c in ["window_start", "window_end"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df[df["candidate_id"].astype(str).isin(LISTING_IDS)].copy()
    validate_window_df(df, ["window_start", "window_end"])
    return df


def load_listing_events() -> pd.DataFrame:
    p = LISTING_ROOT / "listing/full_event_listing_events.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df = df[df.get("candidate_id", pd.Series(dtype=str)).astype(str).isin(LISTING_IDS)].copy()
    for c in ["decision_ts", "entry_ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    validate_window_df(df, [c for c in ["decision_ts", "entry_ts"] if c in df.columns])
    return df


def load_d4_geometry() -> pd.DataFrame:
    p = D4_SURVIVAL_ROOT / "geometry/decision_time_liquidation_geometry.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    for c in ["decision_ts", "entry_ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    validate_window_df(df, [c for c in ["decision_ts", "entry_ts"] if c in df.columns])
    return df


def classify_listing_window(row: Mapping[str, Any], events_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    if str(row.get("window_role", "")) == "control":
        ct = str(row.get("control_type", "control") or "control")
        return f"control_{ct}" if ct else "control"
    event_id = str(row.get("source_event_id") or row.get("event_id") or "")
    ev = events_by_id.get(event_id, {})
    if bool(ev.get("72h_liquidation_10x", False)) or bool(ev.get("24h_liquidation_10x", False)):
        return "ambiguous_or_liquidation_sensitive"
    mfe = maybe_float(ev.get("24h_mfe_bps"), maybe_float(ev.get("6h_mfe_bps"), 0.0))
    mae = maybe_float(ev.get("24h_mae_bps"), maybe_float(ev.get("6h_mae_bps"), 0.0))
    close = maybe_float(ev.get("24h_close_return_bps"), maybe_float(ev.get("6h_close_return_bps"), 0.0))
    if mfe >= 1000 or close > 250:
        return "high_positive"
    if mae >= 1000 or close < -250:
        return "worst_adverse"
    return "representative_median"


def make_d4_window(row: Mapping[str, Any], reason: str, priority: int, role: str = "candidate_event") -> dict[str, Any] | None:
    anchor = pd.to_datetime(row.get("entry_ts", row.get("decision_ts")), utc=True, errors="coerce")
    if pd.isna(anchor):
        return None
    start = pd.Timestamp(anchor) - pd.Timedelta(hours=4)
    end = pd.Timestamp(anchor) + pd.Timedelta(hours=24)
    if end >= FINAL_HOLDOUT_START or end <= start:
        return None
    out = {
        "candidate_id": D4_CANDIDATE_ID,
        "family": "D4_liquidation_safe_flush",
        "subfamily": "dynamic_buffer_execution_depth",
        "event_id": row.get("event_id", ""),
        "symbol": row.get("symbol", ""),
        "window_start": start,
        "window_end": min(end, SCREENING_END),
        "hours": (min(end, SCREENING_END) - start).total_seconds() / 3600.0,
        "window_role": role,
        "control_type": "matched_null_control" if role == "control" else "",
        "selection_bucket": reason,
        "window_scope": "execution_depth_24h",
        "priority": priority,
        "datasets_requested": "top_of_book;shallow_depth;public_trades;historical_liquidations;mark_1m;ohlcv_1m",
    }
    out["target_window_id"] = stable_id(out["candidate_id"], out["event_id"], out["window_start"], out["window_end"], role, prefix="dw")
    return out


def build_full_window_manifest(ctx: RunContext) -> pd.DataFrame:
    listing_windows = load_listing_windows()
    listing_events = load_listing_events()
    events_by_id = {str(r.get("event_id")): r for r in listing_events.to_dict("records")} if not listing_events.empty else {}
    rows: list[dict[str, Any]] = []
    if ctx.args.include_listing and not listing_windows.empty:
        for _, r in listing_windows.iterrows():
            rec = r.to_dict()
            rec["selection_bucket"] = classify_listing_window(rec, events_by_id)
            rec["datasets_requested"] = "top_of_book;shallow_depth;public_trades;historical_liquidations;mark_1m;ohlcv_1m"
            rows.append(rec)
    if ctx.args.include_d4:
        d4 = load_d4_geometry()
        if ctx.args.max_symbols and not d4.empty:
            syms = sorted(d4["symbol"].dropna().astype(str).unique())[: ctx.args.max_symbols]
            d4 = d4[d4["symbol"].astype(str).isin(syms)]
        if ctx.args.smoke and not d4.empty:
            d4 = d4.head(30)
        if not d4.empty:
            pools = [
                ("liquidation_sensitive", d4.sort_values("liq_to_stop_ratio_10p0x", ascending=True).head(300), 1),
                ("high_mfe_proxy", d4.sort_values("liq_to_stop_ratio_5p0x", ascending=False).head(300), 2),
                ("worst_mae_proxy", d4.sort_values("stop_distance_bps", ascending=False).head(300), 3),
                ("representative_median", d4.iloc[np.linspace(0, len(d4) - 1, min(300, len(d4))).astype(int)] if len(d4) else d4, 4),
            ]
            for reason, sub, priority in pools:
                for _, row in sub.iterrows():
                    win = make_d4_window(row, reason, priority)
                    if win:
                        rows.append(win)
            # Controls use deterministic shifted windows around the same event. They are not candidate filters.
            control_pool = d4.iloc[np.linspace(0, len(d4) - 1, min(300, len(d4))).astype(int)] if len(d4) else d4
            for _, row in control_pool.iterrows():
                shifted = row.copy()
                shifted["entry_ts"] = pd.to_datetime(row.get("entry_ts"), utc=True) + pd.Timedelta(days=7)
                if pd.to_datetime(shifted["entry_ts"], utc=True) + pd.Timedelta(hours=24) >= FINAL_HOLDOUT_START:
                    shifted["entry_ts"] = pd.to_datetime(row.get("entry_ts"), utc=True) - pd.Timedelta(days=7)
                win = make_d4_window(shifted, "matched_null_control", 5, role="control")
                if win:
                    win["event_id"] = f"{row.get('event_id')}_control"
                    win["target_window_id"] = stable_id(win["candidate_id"], win["event_id"], win["window_start"], win["window_end"], win["window_role"], prefix="dw")
                    rows.append(win)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ["window_start", "window_end"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    validate_window_df(df, ["window_start", "window_end"])
    df = df.dropna(subset=["window_start", "window_end", "symbol"])
    df = df[df["window_end"] > df["window_start"]].copy()
    if ctx.args.max_symbols and not df.empty:
        parts: list[pd.DataFrame] = []
        for _, sub in df.groupby("candidate_id", dropna=False):
            syms = sorted(sub["symbol"].dropna().astype(str).unique())[: ctx.args.max_symbols]
            parts.append(sub[sub["symbol"].astype(str).isin(syms)].copy())
        df = pd.concat(parts, ignore_index=True) if parts else df.head(0)
    df = df.drop_duplicates(subset=["candidate_id", "symbol", "window_start", "window_end", "window_role", "selection_bucket"], keep="first")
    if "target_window_id" not in df.columns:
        df["target_window_id"] = [stable_id(r.candidate_id, r.symbol, r.window_start, r.window_end, r.window_role) for r in df.itertuples()]
    return df.sort_values(["priority", "candidate_id", "window_role", "window_start"]).reset_index(drop=True)


def choose_rows(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    if sub.empty or n <= 0:
        return sub.head(0)
    if len(sub) <= n:
        return sub
    idx = np.linspace(0, len(sub) - 1, n).round().astype(int)
    return sub.iloc[sorted(set(idx))].head(n)


def select_pilot_windows(full: pd.DataFrame, pilot_count: int = 250) -> tuple[pd.DataFrame, pd.DataFrame]:
    if full.empty:
        return full, full
    selected_parts: list[pd.DataFrame] = []
    # Minimum intended quotas: 50 D4 and 50 per listing candidate where available.
    d4 = full[full["candidate_id"].astype(str).eq(D4_CANDIDATE_ID)].copy()
    if not d4.empty:
        d4_controls = d4[d4["window_role"].astype(str).eq("control")]
        d4_events = d4[d4["window_role"].astype(str).ne("control")]
        selected_parts.append(choose_rows(d4_controls.sort_values("priority"), min(20, max(0, len(d4_controls)))))
        remaining = max(0, 50 - sum(len(x) for x in selected_parts if not x.empty and x["candidate_id"].astype(str).eq(D4_CANDIDATE_ID).all()))
        selected_parts.append(choose_rows(d4_events.sort_values(["priority", "window_start"]), remaining))
    for cid in LISTING_IDS:
        sub = full[full["candidate_id"].astype(str).eq(cid)].copy()
        if sub.empty:
            continue
        controls = sub[sub["window_role"].astype(str).eq("control")].copy()
        events = sub[sub["window_role"].astype(str).ne("control")].copy()
        per = min(50, max(10, pilot_count // 5))
        control_n = min(max(per // 2, 15), len(controls))
        selected_parts.append(choose_rows(controls.sort_values(["priority", "window_start"]), control_n))
        remaining = per - control_n
        buckets = ["ambiguous_or_liquidation_sensitive", "worst_adverse", "high_positive", "representative_median"]
        per_bucket = max(1, remaining // max(1, len(buckets)))
        bucket_rows = []
        for b in buckets:
            bucket_rows.append(choose_rows(events[events["selection_bucket"].astype(str).eq(b)].sort_values("window_start"), per_bucket))
        event_sel = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else events.head(0)
        if len(event_sel) < remaining:
            extra = events[~events["target_window_id"].astype(str).isin(set(event_sel.get("target_window_id", pd.Series(dtype=str)).astype(str)))]
            event_sel = pd.concat([event_sel, choose_rows(extra.sort_values(["priority", "window_start"]), remaining - len(event_sel))], ignore_index=True)
        selected_parts.append(event_sel.head(remaining))
    selected = pd.concat([x for x in selected_parts if not x.empty], ignore_index=True) if selected_parts else full.head(0)
    selected = selected.drop_duplicates(subset=["target_window_id"], keep="first")
    if len(selected) > pilot_count:
        # Controls are higher priority than duplicate winners.
        selected["_role_rank"] = np.where(selected["window_role"].astype(str).eq("control"), 0, 1)
        selected = selected.sort_values(["_role_rank", "priority", "candidate_id", "window_start"]).head(pilot_count).drop(columns=["_role_rank"])
    elif len(selected) < pilot_count:
        remaining = full[~full["target_window_id"].astype(str).isin(set(selected["target_window_id"].astype(str)))]
        controls = remaining[remaining["window_role"].astype(str).eq("control")]
        extras = pd.concat([controls, remaining[remaining["window_role"].astype(str).ne("control")]], ignore_index=True)
        selected = pd.concat([selected, extras.head(pilot_count - len(selected))], ignore_index=True)
    selected = selected.drop_duplicates(subset=["target_window_id"], keep="first").head(pilot_count)
    omitted = full[~full["target_window_id"].astype(str).isin(set(selected["target_window_id"].astype(str)))].copy()
    return selected.reset_index(drop=True), omitted.reset_index(drop=True)


def estimate_storage_rows(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if windows.empty:
        return pd.DataFrame()
    hours = pd.to_numeric(windows.get("hours", pd.Series(dtype=float)), errors="coerce").fillna(
        (pd.to_datetime(windows["window_end"], utc=True) - pd.to_datetime(windows["window_start"], utc=True)).dt.total_seconds() / 3600
    )
    total_hours = float(hours.sum())
    # Conservative rough estimates for targeted windows. Not vendor quotes.
    specs = [
        ("historical_top_of_book_bbo", 0.000020),
        ("historical_shallow_depth_top5", 0.000080),
        ("historical_shallow_depth_top25", 0.000250),
        ("historical_public_trades", 0.000180),
        ("historical_liquidation_events", 0.000010),
        ("full_l2_deltas_vendor_only", 0.001500),
    ]
    for dataset, gb_per_symbol_hour in specs:
        rows.append({
            "dataset": dataset,
            "windows": int(len(windows)),
            "symbol_hours": total_hours,
            "estimated_compressed_gb": total_hours * gb_per_symbol_hour,
            "estimate_basis": "rough_targeted_symbol_hour_proxy_not_vendor_quote",
        })
    return pd.DataFrame(rows)


def source_rows(ctx: RunContext) -> list[dict[str, Any]]:
    candidates = [
        ("historical_public_trades", [Path("/data/bybit/trades"), Path("/data/tardis"), Path("/opt/parquet/bybit_trades")], False),
        ("historical_top_of_book", [Path("/data/bybit/book_ticker"), Path("/data/tardis"), Path("/opt/parquet/bybit_bbo")], False),
        ("historical_shallow_depth", [Path("/data/bybit/orderbook"), Path("/data/tardis"), Path("/opt/parquet/bybit_depth")], False),
        ("historical_liquidation_events", [Path("/data/bybit/liquidations"), Path("/data/tardis"), Path("/opt/parquet/bybit_liquidations")], False),
        ("live_liquidation_stream_only", [Path("/data/live/liquidations"), Path("/opt/testerdonch/live")], True),
        ("forward_live_capture_only", [Path("/data/live_capture"), Path("/opt/testerdonch/live_capture")], True),
    ]
    rows = []
    for data_type, paths, live_only in candidates:
        inv = [path_inventory(p) for p in paths]
        found = any(x["exists"] and int(x["file_count"]) > 0 for x in inv)
        rows.append({
            "data_type": data_type,
            "local_source_found": found,
            "historical_usable_for_2025_windows": bool(found and not live_only),
            "live_stream_only": bool(live_only and found),
            "forward_capture_possible": True,
            "bybit_public_api_historical_capability": "not_confirmed_for_depth_or_liquidation_history" if data_type != "historical_public_trades" else "not_enabled_and_not_assumed_sufficient",
            "vendor_likely_required": not bool(found and not live_only),
            "checked_paths": json.dumps(inv, sort_keys=True, default=str),
        })
    rows.append({
        "data_type": "bybit_public_kline_mark_index_funding_oi",
        "local_source_found": True,
        "historical_usable_for_2025_windows": True,
        "live_stream_only": False,
        "forward_capture_possible": False,
        "bybit_public_api_historical_capability": "usable_for_1m_mark_ohlcv_context_only_not_depth_trade_liquidation",
        "vendor_likely_required": False,
        "checked_paths": "existing_prior_1m_runs_and_public_market_endpoints",
    })
    return rows


def depth_data_available(ctx: RunContext) -> bool:
    src = read_csv(ctx.run_root / "sources/depth_trade_liquidation_source_capability_matrix.csv")
    if src.empty:
        return False
    req = {"historical_public_trades", "historical_top_of_book", "historical_shallow_depth"}
    ok = src[src["data_type"].astype(str).isin(req)]
    if ok.empty:
        return False
    return bool(ok["historical_usable_for_2025_windows"].astype(str).str.lower().isin(["true", "1", "yes"]).all())


def stage_preflight(ctx: RunContext) -> None:
    missing = [str(p) for p in input_artifacts() if not p.exists()]
    rows, hashes = artifact_manifest()
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(REPO)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\n- free_disk_gb: `{snap.free_gb:.3f}`\n- hard_stop_free_gb: `5`\n- warning_free_gb: `7`\n- stage_output_block_gb: `40`\n- max_output_gb: `{ctx.args.max_output_gb}`\n- pilot_depth_trade_cap_gb: `{ctx.args.depth_download_cap_gb}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight And Artifact Freeze\n\n- run_root: `{ctx.run_root}`\n- listing_root: `{LISTING_ROOT}`\n- d4_survival_root: `{D4_SURVIVAL_ROOT}`\n- protected_start: `{FINAL_HOLDOUT_START}`\n- screening_end: `{SCREENING_END}`\n- missing_required_artifacts: `{missing}`\n- git_head: `{shell(['git','rev-parse','HEAD'])}`\n- git_status_short: `{shell(['git','status','--short'])[:5000]}`\n")
    if missing:
        raise RuntimeError(f"required input artifacts missing: {missing}")


def stage_telegram(ctx: RunContext) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\n- disabled: `{ctx.notifier.disabled}`\n- remote_available: `{ctx.notifier.remote_available}`\n- status: `{ctx.notifier.status}`\n- missing_or_reason: `{ctx.notifier.missing}`\n- secrets_logged: `false`\n")
    write_text(ctx.run_root / "tmux/watch_commands.md", f"# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n")
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nDefault session: `{ctx.args.tmux_session_name}`. Use `tools/run_qlmg_execution_depth_pilot_tmux.sh` with `--launch-tmux` for full unattended execution.\n")


def stage_seal(ctx: RunContext) -> None:
    pre = pd.DataFrame({"ts": [ctx.end]})
    validate_no_protected(pre, ["ts"])
    blocked = False
    try:
        validate_no_protected(pd.DataFrame({"ts": [FINAL_HOLDOUT_START]}), ["ts"])
    except RuntimeError:
        blocked = True
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "pre_holdout_read_passed": True, "protected_read_blocked": blocked})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\n- protected_start: `{FINAL_HOLDOUT_START}`\n- generated windows must end before protected start: `true`\n- protected smoke blocked: `{blocked}`\n")


def stage_candidate_window_freeze(ctx: RunContext) -> None:
    candidates: list[dict[str, Any]] = []
    if ctx.args.include_d4:
        d4_decision = json.loads((D4_SURVIVAL_ROOT / "decision_summary.json").read_text())
        candidates.append({
            "candidate_id": D4_CANDIDATE_ID,
            "family": "D4_liquidation_safe_flush",
            "subfamily": "dynamic_buffer_execution_depth",
            "source_root": str(D4_SURVIVAL_ROOT),
            "carry_forward_status": d4_decision.get("verdict", "d4_promote_to_targeted_execution_depth_collection"),
            "required_depth_data": "top_of_book;shallow_depth;public_trades;historical_liquidations",
            "max_status_without_depth": "d4_carry_forward_execution_depth",
        })
    if ctx.args.include_listing:
        m = load_listing_candidate_manifest()
        for _, row in m.iterrows():
            candidates.append({
                "candidate_id": row.get("candidate_id"),
                "family": row.get("family"),
                "subfamily": row.get("subfamily"),
                "source_root": str(LISTING_ROOT),
                "horizon": row.get("horizon"),
                "target_r": row.get("target_r"),
                "stop_mult": row.get("stop_mult"),
                "listing_metadata_source": row.get("listing_metadata_source"),
                "lifecycle_cap": "proxy_launch_only" if "proxy" in str(row.get("listing_metadata_source", "")) else "official_listing_metadata_available",
                "required_depth_data": "top_of_book;shallow_depth;public_trades",
                "max_status_without_depth": "pilot_depth_data_unavailable_procure_vendor",
            })
    write_csv(ctx.run_root / "candidates/frozen_candidate_manifest.csv", candidates)
    full = build_full_window_manifest(ctx)
    write_csv(ctx.run_root / "windows/full_window_manifest.csv", full)
    write_csv(ctx.run_root / "windows/window_candidate_control_map.csv", full[[c for c in ["target_window_id", "candidate_id", "family", "symbol", "window_start", "window_end", "window_role", "selection_bucket"] if c in full.columns]] if not full.empty else pd.DataFrame())
    write_text(ctx.run_root / "windows/full_window_freeze_report.md", f"# Candidate And Window Freeze\n\n- candidates: `{len(candidates)}`\n- full windows: `{len(full)}`\n- controls included: `{int((full.get('window_role', pd.Series(dtype=str)).astype(str) == 'control').sum()) if not full.empty else 0}`\n- protected holdout untouched: `true`\n")


def stage_lifecycle(ctx: RunContext) -> None:
    cand = read_csv(ctx.run_root / "candidates/frozen_candidate_manifest.csv")
    rows = []
    for _, row in cand[cand.get("family", pd.Series(dtype=str)).astype(str).eq("new_perp_listing_event_study")].iterrows():
        source = str(row.get("listing_metadata_source", ""))
        proxy_only = "proxy" in source.lower() or not source
        rows.append({
            "candidate_id": row.get("candidate_id"),
            "official_bybit_launch_metadata_available": False,
            "official_status_metadata_available": False,
            "official_delivery_or_delist_metadata_available": False,
            "listing_metadata_source": source or "unknown",
            "proxy_launch_only": proxy_only,
            "classification_cap": "early_life_vwap_loss_or_listing_proxy_prelead" if proxy_only else "official_listing_edge_possible_after_verification",
            "notes": "No official historical launch/delist artifact was found in frozen inputs; do not call this a clean official listing edge." if proxy_only else "official metadata present in frozen inputs",
        })
    write_csv(ctx.run_root / "lifecycle/listing_lifecycle_audit.csv", rows)
    write_text(ctx.run_root / "lifecycle/listing_lifecycle_audit_report.md", "# Official Listing Lifecycle Audit\n\nOfficial Bybit historical launch/listing metadata was not present in the frozen listing candidate rows. Listing candidates therefore keep the `proxy_launch_only` cap and are classified as early-life/VWAP-loss or listing-proxy preleads until official lifecycle data is acquired.\n")


def stage_source_audit(ctx: RunContext) -> None:
    rows = source_rows(ctx)
    write_csv(ctx.run_root / "sources/depth_trade_liquidation_source_capability_matrix.csv", rows)
    hist_ok = any(r["historical_usable_for_2025_windows"] for r in rows if str(r["data_type"]).startswith("historical_"))
    write_text(ctx.run_root / "sources/source_capability_report.md", f"# Depth/Trade/Liquidation Source Capability Audit\n\n- historical execution-depth/trade/liquidation source found locally: `{hist_ok}`\n- source types are separated: historical public trades, historical top-of-book, historical shallow depth, historical liquidation events, live liquidation stream only, forward live capture only.\n- live WebSocket/liquidation capture is not treated as historical liquidation data.\n- Bybit public kline/mark/index/funding/OI data is not sufficient for execution-depth replay.\n")


def stage_pilot_selection(ctx: RunContext) -> None:
    full = read_csv(ctx.run_root / "windows/full_window_manifest.csv")
    if not full.empty:
        for c in ["window_start", "window_end"]:
            full[c] = pd.to_datetime(full[c], utc=True, errors="coerce")
        validate_window_df(full, ["window_start", "window_end"])
    pilot, omitted = select_pilot_windows(full, int(ctx.args.pilot_window_count))
    write_csv(ctx.run_root / "windows/pilot_window_manifest.csv", pilot)
    write_csv(ctx.run_root / "windows/omitted_window_report.csv", omitted)
    rows = []
    if not pilot.empty:
        rows = pilot.groupby(["candidate_id", "window_role", "selection_bucket"], dropna=False).size().reset_index(name="windows").to_dict("records")
    write_csv(ctx.run_root / "windows/pilot_window_composition.csv", rows)
    controls = int((pilot.get("window_role", pd.Series(dtype=str)).astype(str) == "control").sum()) if not pilot.empty else 0
    write_text(ctx.run_root / "windows/window_selection_report.md", f"# Pilot Window Selection\n\n- pilot_window_count_requested: `{ctx.args.pilot_window_count}`\n- pilot_windows_selected: `{len(pilot)}`\n- controls_selected: `{controls}`\n- omitted_windows: `{len(omitted)}`\n- policy: controls are prioritized over additional near-duplicate winners when quotas are tight.\n")


def stage_storage(ctx: RunContext) -> None:
    pilot = read_csv(ctx.run_root / "windows/pilot_window_manifest.csv")
    full = read_csv(ctx.run_root / "windows/full_window_manifest.csv")
    est_pilot = estimate_storage_rows(pilot)
    est_pilot["scope"] = "pilot" if not est_pilot.empty else []
    est_full = estimate_storage_rows(full)
    est_full["scope"] = "full_targeted" if not est_full.empty else []
    est = pd.concat([est_pilot, est_full], ignore_index=True) if not est_pilot.empty or not est_full.empty else pd.DataFrame()
    write_csv(ctx.run_root / "storage/storage_estimate.csv", est)
    pilot_total = float(est.loc[est.get("scope", pd.Series(dtype=str)).astype(str).eq("pilot"), "estimated_compressed_gb"].sum()) if not est.empty else 0.0
    full_total = float(est.loc[est.get("scope", pd.Series(dtype=str)).astype(str).eq("full_targeted"), "estimated_compressed_gb"].sum()) if not est.empty else 0.0
    write_text(ctx.run_root / "storage/storage_estimate_report.md", f"# Storage And Cost Estimate\n\n- pilot_estimated_compressed_gb_all_requested_datasets: `{pilot_total:.4f}`\n- full_targeted_estimated_compressed_gb_all_requested_datasets: `{full_total:.4f}`\n- pilot_cap_gb: `{ctx.args.depth_download_cap_gb}`\n- estimates are rough symbol-hour proxies, not vendor quotes.\n")


def stage_free_bybit_download(ctx: RunContext) -> None:
    enabled = bool(ctx.args.download_free_bybit_data)
    rows = []
    if enabled:
        rows.append({"status": "blocked", "reason": "Bybit public market endpoints do not provide confirmed historical top-of-book/shallow-depth/liquidation history for these 2025 windows in this runner", "downloaded": False})
    else:
        rows.append({"status": "not_enabled", "reason": "--download-free-bybit-data not passed", "downloaded": False})
    write_csv(ctx.run_root / "downloaded_free_bybit/download_manifest.csv", rows)
    write_text(ctx.run_root / "downloaded_free_bybit/download_report.md", f"# Free Bybit Data Download\n\n- enabled: `{enabled}`\n- downloaded_anything: `false`\n- reason: `{rows[0]['reason']}`\n- no secrets logged: `true`\n")


def vendor_source_configured(source: str) -> tuple[bool, str]:
    src = source.strip().lower()
    env_map = {
        "tardis": "TARDIS_API_KEY",
        "amberdata": "AMBERDATA_API_KEY",
        "coinapi": "COINAPI_API_KEY",
        "local_vendor": "QLMG_LOCAL_VENDOR_DEPTH_ROOT",
    }
    if not src:
        return False, "no_vendor_source_requested"
    key = env_map.get(src, "")
    if key and os.environ.get(key):
        return True, f"{src}_configured_via_{key}_presence_only"
    return False, f"{src}_not_configured_or_env_missing"


def stage_vendor_download(ctx: RunContext) -> None:
    enabled = bool(ctx.args.download_vendor_data_if_configured)
    configured, reason = vendor_source_configured(ctx.args.vendor_source)
    rows = []
    if enabled and configured:
        rows.append({"status": "not_implemented_download_adapter", "vendor_source": ctx.args.vendor_source, "reason": "source appears configured but this runner does not execute paid/vendor downloads without a concrete adapter", "downloaded": False})
    elif enabled:
        rows.append({"status": "blocked", "vendor_source": ctx.args.vendor_source, "reason": reason, "downloaded": False})
    else:
        rows.append({"status": "not_enabled", "vendor_source": ctx.args.vendor_source, "reason": "--download-vendor-data-if-configured not passed", "downloaded": False})
    write_csv(ctx.run_root / "downloaded_vendor/download_manifest.csv", rows)
    write_text(ctx.run_root / "downloaded_vendor/download_report.md", f"# Vendor Data Download\n\n- enabled: `{enabled}`\n- vendor_source: `{ctx.args.vendor_source}`\n- configured: `{configured}`\n- downloaded_anything: `false`\n- reason: `{rows[0]['reason']}`\n- secrets_logged: `false`\n")


def stage_qc(ctx: RunContext) -> None:
    rows = [
        {"dataset": "historical_top_of_book", "qc_status": "not_run", "reason": "no_downloaded_dataset"},
        {"dataset": "historical_shallow_depth", "qc_status": "not_run", "reason": "no_downloaded_dataset"},
        {"dataset": "historical_public_trades", "qc_status": "not_run", "reason": "no_downloaded_dataset"},
        {"dataset": "historical_liquidation_events", "qc_status": "not_run", "reason": "no_downloaded_dataset"},
    ]
    write_csv(ctx.run_root / "qc/depth_trade_data_qc_summary.csv", rows)
    write_text(ctx.run_root / "qc/depth_trade_data_qc_report.md", "# Depth/Trade Data QC\n\nQC was not run because no historical execution-depth/trade/liquidation dataset was downloaded or discovered. This is a data-access blocker, not a candidate failure.\n")


def stage_order_path(ctx: RunContext) -> None:
    available = depth_data_available(ctx)
    risk_rows = [{"risk_pct_equity": r, "risk_label": "conservative_baseline" if r <= 0.01 else "aggressive_diagnostic_stress", "live_recommendation": False} for r in RISK_PCTS]
    if not available:
        rows = [{"status": "blocked", "reason": "historical_top_of_book_depth_public_trades_liquidation_history_unavailable", "depth_replayed": False, "top_of_book_replayed": False, "public_trades_replayed": False, "liquidations_replayed": False, **risk_rows[0]}]
        write_csv(ctx.run_root / "replay/order_path_replay_summary.csv", rows)
        write_csv(ctx.run_root / "replay/order_path_risk_grid.csv", risk_rows)
        write_text(ctx.run_root / "replay/order_path_replay_blocked_report.md", "# Order-Path Replay Blocked\n\nHistorical top-of-book, shallow-depth, public-trade, and liquidation-event data were not available locally and were not downloaded from an enabled source. D4 and listing candidates are not rejected from this missing execution data; they remain capped at procurement/live-capture or execution-depth pilot status.\n")
        return
    # Placeholder path for future real adapter; current tests expect schema and conservative risk grid.
    rows = []
    for r in RISK_PCTS:
        rows.append({"status": "not_run_adapter_missing", "reason": "depth_source_available_but_order_path_adapter_not_configured", "depth_replayed": False, "top_of_book_replayed": False, "public_trades_replayed": False, "liquidations_replayed": False, "risk_pct_equity": r, "risk_label": "conservative_baseline" if r <= 0.01 else "aggressive_diagnostic_stress", "live_recommendation": False})
    write_csv(ctx.run_root / "replay/order_path_replay_summary.csv", rows)
    write_csv(ctx.run_root / "replay/order_path_risk_grid.csv", risk_rows)
    write_text(ctx.run_root / "replay/order_path_replay_blocked_report.md", "# Order-Path Replay Blocked\n\nA historical source was detected, but no exchange-faithful order-path adapter was configured for this pilot. Fail closed.\n")


def stage_impact(ctx: RunContext) -> None:
    replay = read_csv(ctx.run_root / "replay/order_path_replay_summary.csv")
    depth_replayed = bool((not replay.empty) and replay.get("depth_replayed", pd.Series([False])).astype(str).str.lower().isin(["true", "1", "yes"]).any())
    prior_controls = read_csv(LISTING_ROOT / "controls/full_event_control_summary.csv")
    rows = []
    if ctx.args.include_d4:
        rows.append({"candidate_id": D4_CANDIDATE_ID, "previous_evidence": "prior_1m_mark_survivability_positive_but_execution_depth_missing", "depth_trade_replay_changed_conclusion": False, "reason": "order_path_replay_not_available", "status_cap": "d4_carry_forward_execution_depth"})
    for cid in LISTING_IDS:
        ctrl = prior_controls[prior_controls.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not prior_controls.empty else pd.DataFrame()
        rows.append({"candidate_id": cid, "previous_evidence": "full_event_1m_mark_prelead_confirmed" if not ctrl.empty else "unknown", "prior_beats_controls": bool(ctrl.get("beats_controls", pd.Series([False])).astype(str).str.lower().isin(["true", "1", "yes"]).any()) if not ctrl.empty else False, "depth_trade_replay_changed_conclusion": depth_replayed, "reason": "order_path_replay_not_available" if not depth_replayed else "order_path_replay_available", "status_cap": "pilot_depth_data_unavailable_procure_vendor" if not depth_replayed else "candidate_survives_depth_pilot"})
    write_csv(ctx.run_root / "impact/candidate_impact_summary.csv", rows)
    write_text(ctx.run_root / "impact/candidate_impact_report.md", "# Candidate Impact Analysis\n\nThis phase did not change candidate conclusions unless historical execution-depth replay ran. Missing depth/trade/liquidation data caps conclusions; it does not reject D4 or listing candidates.\n")


def stage_full_plan(ctx: RunContext) -> None:
    pilot = read_csv(ctx.run_root / "windows/pilot_window_manifest.csv")
    full = read_csv(ctx.run_root / "windows/full_window_manifest.csv")
    priority_rows = []
    if ctx.args.include_d4:
        d4 = pilot[pilot.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(D4_CANDIDATE_ID)] if not pilot.empty else pd.DataFrame()
        priority_rows.append({"priority": 1, "candidate_id": D4_CANDIDATE_ID, "data_types": "historical_top_of_book;historical_shallow_depth;historical_public_trades;historical_liquidation_events", "windows": len(d4), "reason": "D4 liquidation-sensitive execution-depth evidence is first priority"})
    for i, cid in enumerate(LISTING_IDS, start=2):
        sub = pilot[pilot.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not pilot.empty else pd.DataFrame()
        priority_rows.append({"priority": i, "candidate_id": cid, "data_types": "historical_top_of_book;historical_shallow_depth;historical_public_trades", "windows": len(sub), "reason": "listing/VWAP-loss candidate survived full-event 1m replay; controls included in pilot manifest"})
    priority_rows.append({"priority": 5, "candidate_id": "generic_shock_reversal", "data_types": "same_as_listing_if_cheap", "windows": 0, "reason": "include only if cheap after primary D4/listing/control windows"})
    priority_rows.append({"priority": 6, "candidate_id": "funding_window_preservation", "data_types": "backlog_only", "windows": 0, "reason": "preservation/backlog, not default pilot"})
    write_csv(ctx.run_root / "data_plan/full_targeted_data_plan.csv", priority_rows)
    manifest_cols = [c for c in ["target_window_id", "candidate_id", "family", "symbol", "window_start", "window_end", "window_role", "selection_bucket"] if c in pilot.columns]
    write_csv(ctx.run_root / "depth/depth_pilot_window_manifest.csv", pilot[manifest_cols] if manifest_cols else pd.DataFrame())
    full_cols = [c for c in ["target_window_id", "candidate_id", "family", "symbol", "window_start", "window_end", "window_role", "selection_bucket"] if c in full.columns]
    write_csv(ctx.run_root / "depth/depth_full_window_manifest.csv", full[full_cols] if full_cols else pd.DataFrame())
    symbols = sorted(set(pilot.get("symbol", pd.Series(dtype=str)).dropna().astype(str))) if not pilot.empty else []
    write_text(ctx.run_root / "depth/depth_procurement_or_live_capture_plan.md", f"# Depth / Trade / Liquidation Procurement Or Live Capture Plan\n\nPriority order:\n\n1. D4 liquidation-sensitive windows.\n2. Three listing/VWAP-loss candidates that survived full-event 1m replay.\n3. Controls for those listing candidates.\n4. Generic shock/reversal only if cheap to include.\n5. Funding-window only as preservation/backlog.\n\nRequired historical data types: top-of-book/BBO, shallow depth, public trades, liquidation events. A live WebSocket liquidation feed can support forward capture but cannot substitute for historical liquidation data.\n\nExact pilot symbols sample: `{', '.join(symbols[:100])}`\n\nSource options: Tardis/vendor historical archives, any later-discovered local exchange-transfer archive, or forward live capture for future data. Official Bybit public klines are insufficient for execution-depth replay.\n")


def stage_decision(ctx: RunContext) -> None:
    source = read_csv(ctx.run_root / "sources/depth_trade_liquidation_source_capability_matrix.csv")
    replay = read_csv(ctx.run_root / "replay/order_path_replay_summary.csv")
    required_exec_types = {"historical_public_trades", "historical_top_of_book", "historical_shallow_depth", "historical_liquidation_events"}
    exec_source = source[source.get("data_type", pd.Series(dtype=str)).astype(str).isin(required_exec_types)] if not source.empty else pd.DataFrame()
    historical_available = bool((not exec_source.empty) and exec_source.get("historical_usable_for_2025_windows", pd.Series([False])).astype(str).str.lower().isin(["true", "1", "yes"]).all())
    replayed = bool((not replay.empty) and replay.get("depth_replayed", pd.Series([False])).astype(str).str.lower().isin(["true", "1", "yes"]).any())
    if replayed:
        data_verdict = "candidate_survives_depth_pilot"
    elif historical_available:
        data_verdict = "blocked_by_protocol_issue"
    else:
        data_verdict = "pilot_depth_data_unavailable_procure_vendor"
    listing_verdicts = {cid: data_verdict if data_verdict != "candidate_survives_depth_pilot" else "candidate_survives_depth_pilot" for cid in LISTING_IDS}
    d4_verdict = "d4_carry_forward_execution_depth"
    summary = {
        "created_at_utc": utc_now(),
        "run_root": str(ctx.run_root),
        "protected_holdout_untouched": True,
        "protected_start": str(FINAL_HOLDOUT_START),
        "source_availability_verdict": data_verdict,
        "d4_verdict": d4_verdict,
        "listing_candidate_verdicts": listing_verdicts,
        "any_candidate_survived_depth_pilot": bool(replayed),
        "no_validation_or_live_readiness_claimed": True,
        "allowed_statuses": sorted(FINAL_STATUSES),
    }
    if summary["source_availability_verdict"] not in FINAL_STATUSES:
        raise RuntimeError("invalid final status")
    write_json(ctx.run_root / "decision_summary.json", summary)
    rows = []
    for cid, status in [(D4_CANDIDATE_ID, d4_verdict), *listing_verdicts.items()]:
        rows.append({"candidate_id": cid, "status": status, "depth_replayed": replayed, "main_remaining_blocker": "historical_top_of_book_depth_public_trades_liquidations" if not replayed else "none_from_depth_pilot"})
    write_csv(ctx.run_root / "decision/candidate_depth_decision_table.csv", rows)
    write_text(ctx.run_root / "QLMG_EXECUTION_DEPTH_PILOT_REPORT.md", f"# QLMG Execution Depth Pilot Report\n\n## Scope\n\nTrain-only execution-depth pilot. Protected holdout `>= {FINAL_HOLDOUT_START}` was not used. No sealed validation, live readiness, production readiness, or trading recommendation is made.\n\n## Verdicts\n\n- source_availability_verdict: `{summary['source_availability_verdict']}`\n- D4: `{d4_verdict}`\n- listing candidates: `{listing_verdicts}`\n- any candidate survived depth pilot: `{summary['any_candidate_survived_depth_pilot']}`\n\n## Findings\n\n- Pilot windows include candidate and control windows. Controls were prioritized over additional near-duplicate winners when quotas were tight.\n- Listing lifecycle remains capped as `proxy_launch_only` unless official Bybit launch/listing metadata is acquired.\n- Historical public trades, top-of-book, shallow depth, and liquidation-event data are required for order-path replay. Live liquidation streams are not historical liquidation data.\n- If historical execution data is unavailable, D4 and listing candidates are carried forward to procurement/live-capture planning rather than rejected.\n\n## Risk Settings\n\nOrder-path replay risk grid includes conservative baseline risk sizes `0.25%`, `0.50%`, `1.00%` and diagnostic stress sizes `2.50%`, `5%`, `10%`, `15%`, `20%`. Larger settings are not live recommendations.\n")


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rels = [
        "QLMG_EXECUTION_DEPTH_PILOT_REPORT.md",
        "decision_summary.json",
        "preflight/preflight_report.md",
        "preflight/resource_guard_report.md",
        "seal/seal_guard_report.md",
        "candidates/frozen_candidate_manifest.csv",
        "windows/pilot_window_manifest.csv",
        "windows/omitted_window_report.csv",
        "lifecycle/listing_lifecycle_audit_report.md",
        "sources/source_capability_report.md",
        "storage/storage_estimate_report.md",
        "downloaded_free_bybit/download_report.md",
        "downloaded_vendor/download_report.md",
        "qc/depth_trade_data_qc_report.md",
        "replay/order_path_replay_blocked_report.md",
        "impact/candidate_impact_report.md",
        "depth/depth_procurement_or_live_capture_plan.md",
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
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nContains reports, summaries, manifests, and small CSVs only. It does not include downloaded depth/trade datasets.\n")


STAGE_FUNCS = {
    "preflight-and-artifact-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "candidate-and-window-freeze": stage_candidate_window_freeze,
    "official-listing-lifecycle-audit": stage_lifecycle,
    "depth-trade-liquidation-source-capability-audit": stage_source_audit,
    "pilot-window-selection": stage_pilot_selection,
    "storage-and-cost-estimate": stage_storage,
    "free-bybit-data-download-if-enabled": stage_free_bybit_download,
    "vendor-data-download-if-enabled": stage_vendor_download,
    "depth-trade-data-qc": stage_qc,
    "order-path-replay": stage_order_path,
    "candidate-impact-analysis": stage_impact,
    "full-targeted-data-plan": stage_full_plan,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        ctx.notifier.send("QLMG depth pilot stage skipped", f"stage={stage}")
        return
    ensure_guard(ctx, stage)
    append_command(ctx.run_root, stage)
    ctx.notifier.send("QLMG depth pilot stage start", f"stage={stage}")
    STAGE_FUNCS[stage](ctx)
    missing = [str(p) for p in required_outputs(ctx.run_root, stage) if not p.exists()]
    if missing:
        raise RuntimeError(f"stage {stage} did not produce required outputs: {missing}")
    mark_done(ctx.run_root, stage)
    ctx.notifier.send("QLMG depth pilot stage complete", f"stage={stage}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start, end = clamp_window(args)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_root_resolution.json", {"run_root": str(run_root), "reason": reason, "base_run_id": DEFAULT_RUN_ID})
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "start": str(start), "end": str(end), "args": vars(args), "protected_holdout_start": str(FINAL_HOLDOUT_START)})
    notifier.send("QLMG depth pilot run start", f"run_root={run_root}\nstage={args.stage}")
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG depth pilot run complete", f"run_root={run_root}")
        return 0
    except Exception as exc:
        notifier.send("QLMG depth pilot run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
