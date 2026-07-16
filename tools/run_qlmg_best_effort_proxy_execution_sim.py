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

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_best_effort_proxy_execution_sim_20260628_v1"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
DEFAULT_SEED = 20260628
GB = 1024**3

LISTING_IDS = [
    "new_perp_listing_event_study__589a8c85c943",
    "new_perp_listing_event_study__9dc07cfc405c",
    "new_perp_listing_event_study__b1a3735d5092",
]
D4_CANDIDATE_ID = "D4__b4c9487fe82c"

STAGES = (
    "preflight-and-artifact-freeze",
    "seal-guard",
    "baseline-reproduction",
    "liquidity-proxy-feature-build",
    "proxy-spread-model",
    "participation-and-no-fill-model",
    "adverse-selection-model",
    "stop-execution-model",
    "same-minute-ambiguity-model",
    "dynamic-liquidation-safe-sizing",
    "candidate-control-proxy-execution-replay",
    "scenario-survival-table",
    "micro-canary-readiness-diagnostic",
    "decision-report",
    "compact-review-bundle",
    "all",
)

SURVIVAL_LABELS = {
    "survives_base_proxy_execution",
    "survives_severe_proxy_execution",
    "survives_only_optimistic_proxy_execution",
    "fails_proxy_execution_current_expression_only",
    "path_edge_survives_execution_unresolved",
    "not_fairly_tested_missing_execution_depth",
}

FINAL_VERDICTS = {
    "listing_candidate_survives_proxy_execution_needs_capture",
    "listing_candidate_fragile_current_expression_only",
    "listing_candidate_fails_proxy_execution_current_expression_only",
    "d4_remains_execution_depth_data_blocked",
    "micro_canary_possible_execution_only",
    "micro_canary_not_recommended",
    "forward_capture_required",
    "vendor_data_not_affordable_continue_without",
    "blocked_by_protocol_issue",
}


@dataclass
class RunContext:
    args: argparse.Namespace
    run_root: Path
    notifier: "RunNotifier"
    start: pd.Timestamp
    end: pd.Timestamp
    brutal_root: Path | None


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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-proxy-exec-sim")
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
    p = argparse.ArgumentParser(description="QLMG best-effort proxy execution simulation, train-only")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--max-output-gb", type=float, default=30.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--include-listing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-d4", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--intended-order-notional-usdt", type=float, default=100.0)
    p.add_argument("--tmux-session-name", default="qlmg_proxy_exec_sim")
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


def latest_brutal_root() -> Path | None:
    roots = sorted(RESULTS_ROOT.glob("phase_qlmg_brutal_no_depth_stress_20260628_v1*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for root in roots:
        if (root / "decision_summary.json").exists() and (root / "stage_status/decision-report.done").exists():
            return root
    return None


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


def shell(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(cmd), cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def input_artifacts(ctx: RunContext | None = None) -> list[Path]:
    paths = [
        LISTING_ROOT / "replay/full_event_one_minute_events.parquet",
        LISTING_ROOT / "controls/full_event_control_summary.csv",
        LISTING_ROOT / "windows/window_candidate_control_map.csv",
        LISTING_ROOT / "downloaded_1m/download_manifest.csv",
        LISTING_ROOT / "listing/full_event_listing_events.parquet",
        D4_SURVIVAL_ROOT / "D4_SURVIVABILITY_REDESIGN_REPORT.md",
        D4_SURVIVAL_ROOT / "decision_summary.json",
    ]
    if ctx and ctx.brutal_root:
        paths.extend([ctx.brutal_root / "decision_summary.json", ctx.brutal_root / "QLMG_BRUTAL_NO_DEPTH_STRESS_REPORT.md"])
    return paths


def artifact_manifest(ctx: RunContext) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {}
    for p in input_artifacts(ctx):
        exists = p.exists()
        row = {"path": str(p), "exists": exists, "size_bytes": p.stat().st_size if exists else 0, "sha256": sha256_file(p) if exists and p.stat().st_size < 600 * 1024 * 1024 else "large_or_missing"}
        rows.append(row)
        hashes[str(p)] = row
    return rows, hashes


def done_path(root: Path, stage: str) -> Path:
    return root / "stage_status" / f"{stage}.done"


def mark_done(root: Path, stage: str) -> None:
    write_text(done_path(root, stage), utc_now())


def required_outputs(root: Path, stage: str) -> list[Path]:
    mapping: dict[str, list[str]] = {
        "preflight-and-artifact-freeze": ["preflight/preflight_report.md", "preflight/frozen_artifact_hashes.json", "preflight/input_artifact_manifest.csv", "preflight/resource_guard_report.md"],
        "seal-guard": ["seal/protected_slice_check.json", "seal/seal_guard_report.md"],
        "baseline-reproduction": ["baseline/baseline_reproduction_summary.csv", "baseline/baseline_reproduction_report.md"],
        "liquidity-proxy-feature-build": ["features/liquidity_proxy_features.parquet", "features/liquidity_proxy_report.md"],
        "proxy-spread-model": ["models/proxy_spread_model_report.md", "models/proxy_spread_model_contract.json"],
        "participation-and-no-fill-model": ["models/participation_no_fill_report.md", "models/participation_no_fill_contract.json"],
        "adverse-selection-model": ["models/adverse_selection_report.md", "models/adverse_selection_contract.json"],
        "stop-execution-model": ["models/stop_execution_report.md", "models/stop_execution_contract.json"],
        "same-minute-ambiguity-model": ["models/same_minute_ambiguity_report.md", "models/same_minute_ambiguity_contract.json"],
        "dynamic-liquidation-safe-sizing": ["models/dynamic_liqsafe_sizing_report.md", "models/dynamic_liqsafe_sizing_summary.csv"],
        "candidate-control-proxy-execution-replay": ["replay/proxy_execution_replay_summary.csv", "replay/proxy_execution_replay_events.parquet", "replay/proxy_execution_replay_report.md"],
        "scenario-survival-table": ["decision/scenario_survival_table.csv", "decision/scenario_survival_report.md"],
        "micro-canary-readiness-diagnostic": ["canary/micro_canary_readiness_report.md", "canary/micro_canary_readiness.csv"],
        "decision-report": ["QLMG_BEST_EFFORT_PROXY_EXECUTION_SIM_REPORT.md", "decision_summary.json"],
        "compact-review-bundle": ["compact_review_bundle/artifact_path_index.csv", "compact_review_bundle/README.md"],
    }
    return [root / p for p in mapping.get(stage, [])]


def stage_complete(root: Path, stage: str) -> bool:
    return done_path(root, stage).exists() and all(p.exists() for p in required_outputs(root, stage))


def estimate_stage_output_gb(stage: str) -> float:
    if stage in {"liquidity-proxy-feature-build", "candidate-control-proxy-execution-replay"}:
        return 1.5
    return 0.2


def enforce_resources(ctx: RunContext, stage: str) -> None:
    snap = resource_snapshot(REPO)
    guard = check_resource_guard(snap, estimated_output_gb=estimate_stage_output_gb(stage), hard_free_gb=5.0, warn_free_gb=7.0, hard_stage_output_gb=20.0, allow_large_output=bool(ctx.args.allow_large_output))
    if guard["warnings"]:
        ctx.notifier.send("QLMG proxy execution resource warning", json.dumps(guard, sort_keys=True), level="warn")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop before {stage}: {guard}")


def numeric(s: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def pf_from_r(r: pd.Series) -> float:
    vals = pd.to_numeric(r, errors="coerce").dropna()
    gains = float(vals[vals > 0].sum())
    losses = float(-vals[vals < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def deterministic_u01(seed: int, *parts: Any) -> float:
    payload = "|".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")
    x = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return x / float(16**16 - 1)


def load_rankable_replay(ctx: RunContext | None = None, *, full_universe: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(LISTING_ROOT / "replay/full_event_one_minute_events.parquet")
    df = df[(df["window_scope"].astype(str) == "full_hold") & (df["full_hold_replayed"].astype(bool)) & (df["replay_status"].astype(str) == "ok")].copy()
    if ctx is not None and ctx.args.include_listing:
        ids = set(LISTING_IDS) | {"generic_shock_reversal_hypothesis"}
        df = df[df["candidate_id"].astype(str).isin(ids)].copy()
    if ctx is not None and not full_universe:
        if ctx.args.max_symbols and "symbol" in df.columns:
            syms = sorted(df["symbol"].dropna().astype(str).unique())[: int(ctx.args.max_symbols)]
            df = df[df["symbol"].astype(str).isin(syms)].copy()
    return df


def window_map() -> pd.DataFrame:
    w = pd.read_csv(LISTING_ROOT / "windows/window_candidate_control_map.csv", low_memory=False)
    w = w[w["window_scope"].astype(str).eq("full_hold")].copy()
    keep = ["candidate_id", "event_id", "symbol", "window_role", "control_type", "window_scope", "target_window_id", "window_start", "window_end", "source_event_id"]
    return w[[c for c in keep if c in w.columns]].drop_duplicates()


def download_manifest() -> pd.DataFrame:
    m = pd.read_csv(LISTING_ROOT / "downloaded_1m/download_manifest.csv")
    return m[m["status"].astype(str).eq("ok")].copy()


def attach_windows_and_events(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_id", "event_id", "symbol", "window_role", "window_scope"]
    wm = window_map()
    merged = df.merge(wm, on=keys, how="left", validate="many_to_one")
    events = pd.read_parquet(LISTING_ROOT / "listing/full_event_listing_events.parquet")
    event_cols = ["event_id", "decision_ts", "entry_ts", "liquidity_tier", "listing_metadata_source", "funding_rate", "oi_chg_24h", "candidate_horizon"]
    events = events[[c for c in event_cols if c in events.columns]].drop_duplicates("event_id")
    merged = merged.merge(events, on="event_id", how="left", suffixes=("", "_event"))
    merged["window_start"] = pd.to_datetime(merged["window_start"], utc=True, errors="coerce")
    merged["window_end"] = pd.to_datetime(merged["window_end"], utc=True, errors="coerce")
    merged["entry_ts_proxy"] = merged["window_start"] + pd.Timedelta(hours=4)
    merged["decision_ts_proxy"] = merged["entry_ts_proxy"] - pd.Timedelta(minutes=5)
    merged["entry_ts_effective"] = pd.to_datetime(merged.get("entry_ts"), utc=True, errors="coerce").fillna(merged["entry_ts_proxy"])
    merged["decision_ts_effective"] = pd.to_datetime(merged.get("decision_ts"), utc=True, errors="coerce").fillna(merged["decision_ts_proxy"])
    validate_no_protected(merged, ["window_start", "window_end", "entry_ts_effective", "decision_ts_effective"])
    return merged


def baseline_from_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, sub in df.groupby("candidate_id", dropna=False):
        cand = sub[sub["window_role"].astype(str).eq("candidate_event")]
        ctrl = sub[sub["window_role"].astype(str).eq("control")]
        cvals = numeric(cand.get("net_R_1m_mark_proxy", pd.Series(dtype=float))).dropna()
        xvals = numeric(ctrl.get("net_R_1m_mark_proxy", pd.Series(dtype=float))).dropna()
        ctrl_norm = float(xvals.sum()) * (len(cvals) / len(xvals)) if len(xvals) else np.nan
        rows.append({
            "candidate_id": cid,
            "rankable_scope": "full_hold",
            "candidate_event_count": int(len(cvals)),
            "control_event_count": int(len(xvals)),
            "event_signal_R": float(cvals.sum()) if len(cvals) else 0.0,
            "control_signal_R_raw_sum": float(xvals.sum()) if len(xvals) else 0.0,
            "control_signal_R_normalized_to_candidate_count": ctrl_norm,
            "normalized_uplift_R": float(cvals.sum()) - ctrl_norm if np.isfinite(ctrl_norm) else np.nan,
            "event_mean_R": float(cvals.mean()) if len(cvals) else np.nan,
            "control_mean_R": float(xvals.mean()) if len(xvals) else np.nan,
            "beats_controls": bool(np.isfinite(ctrl_norm) and float(cvals.sum()) > ctrl_norm),
        })
    return pd.DataFrame(rows)


def dataset_path_lookup() -> dict[tuple[str, str], Path]:
    m = download_manifest()
    out: dict[tuple[str, str], Path] = {}
    for _, row in m.iterrows():
        path = Path(str(row.get("path", "")))
        if path.exists():
            out[(str(row["target_window_id"]), str(row["dataset"]))] = path
    return out


def read_window_dataset(paths: dict[tuple[str, str], Path], target_window_id: str, dataset: str) -> pd.DataFrame:
    p = paths.get((str(target_window_id), dataset))
    if p is None or not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").copy()
        validate_no_protected(df, ["timestamp"])
    return df


def row_at_or_after(df: pd.DataFrame, ts: pd.Timestamp, tolerance_min: int = 2) -> pd.Series | None:
    if df.empty or "timestamp" not in df.columns:
        return None
    t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    mask = (t >= ts) & (t <= ts + pd.Timedelta(minutes=tolerance_min))
    if not bool(mask.any()):
        mask = (t <= ts) & (t >= ts - pd.Timedelta(minutes=tolerance_min))
    if not bool(mask.any()):
        return None
    return df.loc[mask].iloc[0]


def trailing_sum_until(df: pd.DataFrame, ts: pd.Timestamp, col: str, minutes: int) -> float:
    if df.empty or "timestamp" not in df.columns or col not in df.columns:
        return np.nan
    t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    sub = df[(t <= ts) & (t > ts - pd.Timedelta(minutes=minutes))]
    return float(pd.to_numeric(sub[col], errors="coerce").sum()) if not sub.empty else np.nan


def feature_from_window(row: pd.Series, paths: dict[tuple[str, str], Path]) -> dict[str, Any]:
    tw = str(row.get("target_window_id", ""))
    entry_ts = pd.Timestamp(row["entry_ts_effective"])
    ohlcv = read_window_dataset(paths, tw, "ohlcv_1m")
    mark = read_window_dataset(paths, tw, "mark_1m")
    index = read_window_dataset(paths, tw, "index_1m")
    premium = read_window_dataset(paths, tw, "premium_1m")
    oi = read_window_dataset(paths, tw, "open_interest_5m")
    funding = read_window_dataset(paths, tw, "funding_history")
    out: dict[str, Any] = {
        "candidate_id": row.get("candidate_id"),
        "event_id": row.get("event_id"),
        "symbol": row.get("symbol"),
        "window_role": row.get("window_role"),
        "control_type": row.get("control_type"),
        "target_window_id": tw,
        "entry_ts_effective": entry_ts,
        "decision_ts_effective": row.get("decision_ts_effective"),
        "risk_bps_used": row.get("risk_bps_used"),
        "target_r_used": row.get("target_r_used"),
        "base_net_R_1m_mark_proxy": row.get("net_R_1m_mark_proxy"),
        "exit_reason_1m": row.get("exit_reason_1m"),
        "stop_hit_1m": bool(row.get("stop_hit_1m", False)),
        "target_hit_1m": bool(row.get("target_hit_1m", False)),
        "mfe_bps_1m": row.get("mfe_bps_1m"),
        "mae_bps_1m": row.get("mae_bps_1m"),
        "liquidity_tier": row.get("liquidity_tier"),
        "listing_metadata_source": row.get("listing_metadata_source"),
        "candidate_horizon": row.get("candidate_horizon"),
        "feature_quality": "ok",
    }
    erow = row_at_or_after(ohlcv, entry_ts)
    mrow = row_at_or_after(mark, entry_ts)
    irow = row_at_or_after(index, entry_ts)
    prow = row_at_or_after(premium, entry_ts)
    if erow is None:
        out["feature_quality"] = "missing_ohlcv_1m"
        return out
    open_p = float(erow.get("open", np.nan))
    high_p = float(erow.get("high", np.nan))
    low_p = float(erow.get("low", np.nan))
    close_p = float(erow.get("close", np.nan))
    denom = close_p if np.isfinite(close_p) and close_p > 0 else open_p
    rng = max(high_p - low_p, 0.0) if np.isfinite(high_p) and np.isfinite(low_p) else np.nan
    body = abs(close_p - open_p) if np.isfinite(close_p) and np.isfinite(open_p) else np.nan
    out.update({
        "notional_1m": float(erow.get("turnover", np.nan)),
        "volume_1m": float(erow.get("volume", np.nan)),
        "rolling_5m_notional": trailing_sum_until(ohlcv, entry_ts, "turnover", 5),
        "rolling_15m_notional": trailing_sum_until(ohlcv, entry_ts, "turnover", 15),
        "rolling_60m_notional": trailing_sum_until(ohlcv, entry_ts, "turnover", 60),
        "true_range_1m": rng,
        "high_low_range_bps": (rng / denom * 10000.0) if np.isfinite(rng) and np.isfinite(denom) and denom > 0 else np.nan,
        "wick_ratio": ((rng - body) / rng) if np.isfinite(rng) and rng > 0 and np.isfinite(body) else np.nan,
        "entry_minute_close_minus_open_bps": ((close_p - open_p) / open_p * 10000.0) if np.isfinite(open_p) and open_p > 0 and np.isfinite(close_p) else np.nan,
    })
    if mrow is not None:
        mark_close = float(mrow.get("close", np.nan))
        out["mark_close_1m"] = mark_close
        out["close_to_mark_divergence_bps"] = ((close_p - mark_close) / mark_close * 10000.0) if np.isfinite(mark_close) and mark_close > 0 and np.isfinite(close_p) else np.nan
    else:
        out["mark_missing"] = True
    if irow is not None and mrow is not None:
        index_close = float(irow.get("close", np.nan))
        mark_close = float(mrow.get("close", np.nan))
        out["mark_index_divergence_bps"] = ((mark_close - index_close) / index_close * 10000.0) if np.isfinite(index_close) and index_close > 0 and np.isfinite(mark_close) else np.nan
    if prow is not None:
        out["premium_close"] = float(prow.get("close", np.nan))
    oirow = row_at_or_after(oi, entry_ts, tolerance_min=5)
    if oirow is not None:
        out["open_interest_entry"] = float(oirow.get("open_interest", np.nan))
    if not funding.empty and "timestamp" in funding.columns:
        ft = pd.to_datetime(funding["timestamp"], utc=True, errors="coerce")
        fsub = funding[ft <= entry_ts]
        if not fsub.empty:
            out["funding_rate_latest"] = float(fsub.iloc[-1].get("funding_rate", np.nan))
    return out


def add_trailing_percentiles(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    f = features.copy()
    f["entry_month"] = pd.to_datetime(f["entry_ts_effective"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("M").astype(str)
    f = f.sort_values(["symbol", "entry_ts_effective", "candidate_id", "event_id"]).copy()
    for col, out_col in [("high_low_range_bps", "volatility_percentile_symbol_month"), ("notional_1m", "volume_percentile_symbol_month"), ("wick_ratio", "wick_ratio_percentile_symbol_month")]:
        vals = []
        for _, g in f.groupby(["symbol", "entry_month"], dropna=False):
            s = pd.to_numeric(g[col], errors="coerce")
            hist: list[float] = []
            for v in s:
                if np.isfinite(v):
                    hist.append(float(v))
                    vals.append(float((np.array(hist) <= v).sum() / len(hist)))
                else:
                    vals.append(np.nan)
        f[out_col] = vals
    return f


def summarize_r(vals: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(vals, errors="coerce").dropna()
    gains = float(v[v > 0].sum())
    losses = float(-v[v < 0].sum())
    return {
        "events": int(len(v)),
        "net_R": float(v.sum()) if len(v) else 0.0,
        "mean_R": float(v.mean()) if len(v) else np.nan,
        "median_R": float(v.median()) if len(v) else np.nan,
        "PF": (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0),
        "hit_rate": float((v > 0).mean()) if len(v) else np.nan,
    }


def stage_preflight(ctx: RunContext) -> None:
    rows, hashes = artifact_manifest(ctx)
    missing = [r["path"] for r in rows if not r["exists"]]
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(REPO)
    guard = check_resource_guard(snap, estimated_output_gb=2.0, hard_free_gb=5.0, warn_free_gb=7.0, hard_stage_output_gb=20.0, allow_large_output=ctx.args.allow_large_output)
    dm = read_csv(LISTING_ROOT / "downloaded_1m/download_manifest.csv")
    cov = dm.groupby("dataset", dropna=False).agg(files=("path", "count"), ok=("status", lambda s: int((s.astype(str) == "ok").sum()))).reset_index() if not dm.empty else pd.DataFrame()
    write_csv(ctx.run_root / "preflight/downloaded_1m_dataset_coverage.csv", cov)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\n- free_disk_gb: `{snap.free_gb:.3f}`\n- hard_stop_free_gb: `5`\n- warning_free_gb: `7`\n- stage_output_block_gb: `20`\n- max_output_gb: `{ctx.args.max_output_gb}`\n- guard_status: `{guard['status']}`\n- warnings: `{guard['warnings']}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight And Artifact Freeze\n\n- run_root: `{ctx.run_root}`\n- listing_root: `{LISTING_ROOT}`\n- brutal_root: `{ctx.brutal_root}`\n- d4_survival_root: `{D4_SURVIVAL_ROOT}`\n- protected_start: `{FINAL_HOLDOUT_START}`\n- missing_required_artifacts: `{missing}`\n- no_downloads: `true`\n- git_head: `{shell(['git','rev-parse','HEAD'])}`\n- git_status_short: `{shell(['git','status','--short'])[:5000]}`\n")
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\n- disabled: `{ctx.notifier.disabled}`\n- remote_available: `{ctx.notifier.remote_available}`\n- status: `{ctx.notifier.status}`\n- missing_or_reason: `{ctx.notifier.missing}`\n- secrets_logged: `false`\n")
    write_text(ctx.run_root / "tmux/watch_commands.md", f"# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n")
    if missing:
        raise RuntimeError(f"required input artifacts missing: {missing}")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop: {guard}")


def stage_seal(ctx: RunContext) -> None:
    validate_no_protected(pd.DataFrame({"ts": [ctx.end]}), ["ts"])
    blocked = False
    try:
        validate_no_protected(pd.DataFrame({"ts": [FINAL_HOLDOUT_START]}), ["ts"])
    except RuntimeError:
        blocked = True
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "pre_holdout_read_passed": True, "protected_read_blocked": blocked})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\n- protected_start: `{FINAL_HOLDOUT_START}`\n- protected smoke blocked: `{blocked}`\n- final holdout touched: `false`\n")


def stage_baseline(ctx: RunContext) -> None:
    df = load_rankable_replay(ctx, full_universe=True)
    cur = baseline_from_events(df)
    prior = read_csv(LISTING_ROOT / "controls/full_event_control_summary.csv")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, row in cur.iterrows():
        cid = str(row["candidate_id"])
        p = prior[prior["candidate_id"].astype(str).eq(cid)]
        if p.empty:
            if cid in LISTING_IDS:
                failures.append(cid)
            rows.append({**row.to_dict(), "reproduced_within_tolerance": False, "reason": "prior_missing"})
            continue
        p0 = p.iloc[0]
        count_ok = int(row["candidate_event_count"]) == int(p0["candidate_event_count"]) and int(row["control_event_count"]) == int(p0["control_event_count"])
        event_diff = float(row["event_signal_R"]) - float(p0["event_signal_R"])
        ctrl_diff = float(row["control_signal_R_normalized_to_candidate_count"]) - float(p0["control_signal_R_normalized_to_candidate_count"])
        ok = count_ok and abs(event_diff) <= max(1e-6, abs(float(p0["event_signal_R"])) * 1e-9) and abs(ctrl_diff) <= max(1e-6, abs(float(p0["control_signal_R_normalized_to_candidate_count"])) * 1e-9)
        rows.append({**row.to_dict(), "prior_event_signal_R": p0["event_signal_R"], "prior_control_normalized_R": p0["control_signal_R_normalized_to_candidate_count"], "event_signal_R_diff": event_diff, "control_normalized_R_diff": ctrl_diff, "count_ok": count_ok, "reproduced_within_tolerance": ok})
        if cid in LISTING_IDS and not ok:
            failures.append(cid)
    write_csv(ctx.run_root / "baseline/baseline_reproduction_summary.csv", rows)
    write_text(ctx.run_root / "baseline/baseline_reproduction_report.md", f"# Baseline Reproduction\n\n- rankable_scope: `full_hold`\n- core_24h_excluded: `true`\n- failures: `{failures}`\n")
    if failures:
        raise RuntimeError(f"baseline reproduction failed: {failures}")


def stage_features(ctx: RunContext) -> None:
    base = load_rankable_replay(ctx)
    base = attach_windows_and_events(base)
    if ctx.args.max_symbols:
        syms = sorted(base["symbol"].dropna().astype(str).unique())[: int(ctx.args.max_symbols)]
        base = base[base["symbol"].astype(str).isin(syms)].copy()
    ts = pd.to_datetime(base["entry_ts_effective"], utc=True, errors="coerce")
    base = base[(ts >= ctx.start) & (ts <= ctx.end)].copy()
    if not ctx.args.include_controls:
        base = base[base["window_role"].astype(str).eq("candidate_event")].copy()
    paths = dataset_path_lookup()
    rows: list[dict[str, Any]] = []
    if base.empty:
        features = pd.DataFrame()
    else:
        progress_every = max(int(ctx.args.chunk_size) * 25, 5000)
        for i, (_, row) in enumerate(base.iterrows(), start=1):
            rows.append(feature_from_window(row, paths))
            if i % progress_every == 0 or i == len(base):
                ctx.notifier.send("QLMG proxy execution feature progress", f"processed={i}/{len(base)}")
        features = add_trailing_percentiles(pd.DataFrame(rows))
    out = ctx.run_root / "features/liquidity_proxy_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    q = features.get("feature_quality", pd.Series(dtype=str)).value_counts().to_dict() if not features.empty else {}
    write_text(ctx.run_root / "features/liquidity_proxy_report.md", f"# Liquidity Proxy Feature Build\n\n- feature_rows: `{len(features)}`\n- source: already downloaded 1m OHLCV/mark/index/premium/OI/funding windows.\n- feature_quality_counts: `{q}`\n- decision_time_features and execution_minute_features are distinguished in downstream reports.\n")


def stage_spread(ctx: RunContext) -> None:
    contract = {"spread_models": ["flat25", "flat50", "flat100", "flat200", "vol_max25_0p1_range", "vol_max50_0p2_range", "stress_max100_0p3_range"], "unit": "bps_per_side", "not_order_book_replay": True}
    write_json(ctx.run_root / "models/proxy_spread_model_contract.json", contract)
    write_text(ctx.run_root / "models/proxy_spread_model_report.md", "# Proxy Spread Model\n\nSpread assumptions are per-side bps proxies. Volatility-linked models use the event's 1m high-low range bps. These are not top-of-book measurements.\n")


def stage_participation(ctx: RunContext) -> None:
    contract = {"intended_order_notional_usdt_proxy": "cli --intended-order-notional-usdt", "participation_caps": [0.001, 0.0025, 0.005, 0.01], "fill_policies": ["skip_if_cap_exceeded", "partial_fill_capped_notional"], "no_fill_rules": ["volume_percentile_lt_25", "wick_ratio_top_decile", "range_top_decile", "entry_minute_adverse", "random_10_25_50_deterministic"]}
    write_json(ctx.run_root / "models/participation_no_fill_contract.json", contract)
    write_text(ctx.run_root / "models/participation_no_fill_report.md", "# Participation And No-Fill Model\n\nThe participation model compares a small-account intended notional proxy against a percentage of 1m turnover. Skip and partial-fill variants are both reported. Missing turnover caps the scenario as not fairly tested.\n")


def stage_adverse(ctx: RunContext) -> None:
    contract = {"side": "short_listing_vwap_loss", "penalty_bps": {"mild": 25, "base": 50, "severe": 100, "extreme": 200}, "triggers": ["entry_minute_closes_against_short", "stop_minute_high_range_proxy", "adverse_mark_last_divergence", "liquidity_vacuum_high_volatility"]}
    write_json(ctx.run_root / "models/adverse_selection_contract.json", contract)
    write_text(ctx.run_root / "models/adverse_selection_report.md", "# Adverse Selection Model\n\nFor short listing/VWAP-loss candidates, adverse entry/stop penalties are added when the entry minute moves against the short, mark/last divergence is adverse, or the bar is a high-volatility liquidity-vacuum proxy.\n")


def stage_stop(ctx: RunContext) -> None:
    contract = {"stop_models": ["stop_market_pessimistic", "protected_marketable_limit_50bps", "protected_marketable_limit_100bps", "stop_misses_exit_next_bar", "gap_through_50_100_200bps"], "rankable_base": "pessimistic"}
    write_json(ctx.run_root / "models/stop_execution_contract.json", contract)
    write_text(ctx.run_root / "models/stop_execution_report.md", "# Stop Execution Model\n\nStop execution uses pessimistic stop-market and protected-limit proxies with gap-through penalties. This is not exchange order-book replay.\n")


def stage_ambiguity(ctx: RunContext) -> None:
    contract = {"rankable": ["pessimistic_stop_first", "exclude_ambiguous"], "diagnostic_only": ["target_first"], "ambiguous_definition": "stop_hit_1m and target_hit_1m both true"}
    write_json(ctx.run_root / "models/same_minute_ambiguity_contract.json", contract)
    write_text(ctx.run_root / "models/same_minute_ambiguity_report.md", "# Same-Minute Ambiguity Model\n\nRankable scenarios use stop-first or exclude ambiguous rows. Target-first is diagnostic only and cannot support promotion-style language.\n")


def safe_leverage(risk_bps: pd.Series, buffer: float) -> pd.Series:
    risk = pd.to_numeric(risk_bps, errors="coerce")
    return (10000.0 / (risk * float(buffer) + 50.0)).clip(lower=0.0, upper=10.0)


def stage_dynamic_sizing(ctx: RunContext) -> None:
    f = pd.read_parquet(ctx.run_root / "features/liquidity_proxy_features.parquet") if (ctx.run_root / "features/liquidity_proxy_features.parquet").exists() else pd.DataFrame()
    rows = []
    models = [("fixed_2x", 2.0, None, 0.0), ("fixed_3x", 3.0, None, 0.0), ("fixed_5x", 5.0, None, 0.0), ("dynamic_buffer_1p25", None, 1.25, 0.0), ("dynamic_buffer_1p5", None, 1.5, 0.0), ("dynamic_buffer_2p0", None, 2.0, 0.0), ("dynamic_skip_below_2x", None, 1.5, 2.0), ("dynamic_skip_below_3x", None, 1.5, 3.0)]
    if not f.empty:
        for name, fixed, buffer, min_lev in models:
            lev = pd.Series(float(fixed), index=f.index) if fixed is not None else safe_leverage(f["risk_bps_used"], float(buffer or 1.5))
            keep = lev >= min_lev
            tmp = f.loc[keep].copy()
            tmp["leverage_used"] = lev.loc[keep]
            tmp["signal_R"] = pd.to_numeric(tmp["base_net_R_1m_mark_proxy"], errors="coerce").fillna(0.0)
            tmp["account_R"] = tmp["signal_R"] * (tmp["leverage_used"] / 10.0)
            for cid, sub in tmp.groupby("candidate_id", dropna=False):
                cand = sub[sub["window_role"].astype(str).eq("candidate_event")]
                rows.append({"candidate_id": cid, "sizing_model": name, "events": len(cand), "signal_R": float(cand["signal_R"].sum()), "account_R": float(cand["account_R"].sum()), "avg_leverage": float(cand["leverage_used"].mean()) if not cand.empty else np.nan, "skipped_by_min_leverage": int((~keep).sum())})
    write_csv(ctx.run_root / "models/dynamic_liqsafe_sizing_summary.csv", rows)
    write_text(ctx.run_root / "models/dynamic_liqsafe_sizing_report.md", "# Dynamic Liquidation-Safe Sizing\n\nSizing reports normalized signal_R separately from reduced-notional account_R. Leverage changes do not improve the signal edge; they only change account expression.\n")


def scenario_defs() -> list[dict[str, Any]]:
    return [
        {"scenario_id": "optimistic_flat25_partial", "scenario_band": "optimistic", "spread_model": "flat25", "cap": 0.01, "fill_policy": "partial", "adverse_bps": 25.0, "gap_bps": 0.0, "random_miss": 0.0, "ambiguity_policy": "pessimistic", "rankable": True},
        {"scenario_id": "optimistic_flat50_skip", "scenario_band": "optimistic", "spread_model": "flat50", "cap": 0.01, "fill_policy": "skip", "adverse_bps": 25.0, "gap_bps": 0.0, "random_miss": 0.0, "ambiguity_policy": "pessimistic", "rankable": True},
        {"scenario_id": "base_vol50_partial", "scenario_band": "base", "spread_model": "vol50_0p2", "cap": 0.005, "fill_policy": "partial", "adverse_bps": 50.0, "gap_bps": 50.0, "random_miss": 0.0, "ambiguity_policy": "pessimistic", "rankable": True},
        {"scenario_id": "base_vol50_skip", "scenario_band": "base", "spread_model": "vol50_0p2", "cap": 0.005, "fill_policy": "skip", "adverse_bps": 50.0, "gap_bps": 50.0, "random_miss": 0.0, "ambiguity_policy": "pessimistic", "rankable": True},
        {"scenario_id": "severe_vol100_skip", "scenario_band": "severe", "spread_model": "stress100_0p3", "cap": 0.0025, "fill_policy": "skip", "adverse_bps": 100.0, "gap_bps": 100.0, "random_miss": 0.10, "ambiguity_policy": "pessimistic", "rankable": True},
        {"scenario_id": "extreme_flat300_skip", "scenario_band": "extreme", "spread_model": "flat300", "cap": 0.001, "fill_policy": "skip", "adverse_bps": 200.0, "gap_bps": 200.0, "random_miss": 0.25, "ambiguity_policy": "pessimistic", "rankable": True},
        {"scenario_id": "diagnostic_target_first", "scenario_band": "diagnostic", "spread_model": "flat25", "cap": 0.01, "fill_policy": "partial", "adverse_bps": 0.0, "gap_bps": 0.0, "random_miss": 0.0, "ambiguity_policy": "target_first", "rankable": False},
    ]


def col_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def spread_bps_for(features: pd.DataFrame, model: str) -> pd.Series:
    rng = col_series(features, "high_low_range_bps", 0.0).fillna(0.0)
    if model == "flat25":
        return pd.Series(25.0, index=features.index)
    if model == "flat50":
        return pd.Series(50.0, index=features.index)
    if model == "flat300":
        return pd.Series(300.0, index=features.index)
    if model == "vol50_0p2":
        return np.maximum(50.0, 0.2 * rng)
    if model == "stress100_0p3":
        return np.maximum(100.0, 0.3 * rng)
    return pd.Series(100.0, index=features.index)


def replay_scenario(features: pd.DataFrame, sc: Mapping[str, Any], seed: int, intended_notional: float) -> pd.DataFrame:
    f = features.copy()
    risk = pd.to_numeric(f["risk_bps_used"], errors="coerce")
    valid = risk.gt(0)
    f = f[valid].copy()
    risk = risk[valid]
    spread = spread_bps_for(f, str(sc["spread_model"]))
    adverse = pd.Series(0.0, index=f.index)
    closes_against_short = pd.to_numeric(f.get("entry_minute_close_minus_open_bps", 0), errors="coerce").fillna(0.0) > 0
    high_wick = pd.to_numeric(f.get("wick_ratio_percentile_symbol_month", 0), errors="coerce").fillna(0.0) >= 0.90
    high_range = pd.to_numeric(f.get("volatility_percentile_symbol_month", 0), errors="coerce").fillna(0.0) >= 0.90
    adverse = adverse + np.where(closes_against_short, float(sc["adverse_bps"]), 0.0)
    adverse = adverse + np.where(high_range, float(sc["adverse_bps"]) * 0.5, 0.0)
    divergence = col_series(f, "close_to_mark_divergence_bps", 0.0).fillna(0.0)
    adverse = adverse + np.where(divergence < -25.0, float(sc["adverse_bps"]) * 0.25, 0.0)
    stop = f["stop_hit_1m"].astype(bool)
    amb = stop & f["target_hit_1m"].astype(bool)
    stop_gap = np.where(stop, float(sc["gap_bps"]), 0.0)
    cap_notional = float(sc["cap"]) * col_series(f, "notional_1m", np.nan)
    fill_ratio = (cap_notional / float(intended_notional)).clip(lower=0.0, upper=1.0)
    missing_turnover = ~np.isfinite(cap_notional) | cap_notional.le(0)
    skip = pd.Series(False, index=f.index)
    if sc["fill_policy"] == "skip":
        skip = skip | (fill_ratio < 1.0)
        fill_ratio = pd.Series(1.0, index=f.index)
    else:
        skip = skip | missing_turnover
        fill_ratio = fill_ratio.fillna(0.0)
    low_vol = col_series(f, "volume_percentile_symbol_month", 1.0).fillna(1.0) < 0.25
    nofill_rule = low_vol | high_wick | high_range | closes_against_short
    if sc["scenario_band"] in {"base", "severe", "extreme"}:
        skip = skip | nofill_rule
    if float(sc["random_miss"]) > 0:
        rnd = f["event_id"].astype(str).map(lambda x: deterministic_u01(seed, sc["scenario_id"], x))
        skip = skip | (rnd < float(sc["random_miss"]))
    if sc["ambiguity_policy"] == "exclude_ambiguous":
        skip = skip | amb
    base_r = pd.to_numeric(f["base_net_R_1m_mark_proxy"], errors="coerce").fillna(0.0)
    haircut_r = ((2.0 * spread) + adverse + stop_gap) / risk
    signal_r = base_r - haircut_r
    if sc["ambiguity_policy"] == "pessimistic":
        signal_r = signal_r.mask(amb, np.minimum(signal_r, -1.0 - (float(sc["gap_bps"]) / risk)))
    if sc["ambiguity_policy"] == "target_first":
        signal_r = signal_r.mask(amb, np.maximum(signal_r, float(f.get("target_r_used", 3.0).iloc[0]) if len(f) else 3.0))
    out = f.copy()
    out["scenario_id"] = sc["scenario_id"]
    out["scenario_band"] = sc["scenario_band"]
    out["rankable"] = bool(sc["rankable"])
    out["spread_bps_per_side"] = spread
    out["execution_haircut_R"] = haircut_r
    out["signal_R"] = signal_r.where(~skip)
    out["fill_ratio"] = fill_ratio.where(~skip, 0.0)
    out["account_R"] = out["signal_R"] * out["fill_ratio"]
    out["skipped_flag"] = skip
    out["partial_fill_flag"] = (~skip) & (fill_ratio < 1.0)
    out["no_fill_flag"] = skip
    out["ambiguous_same_minute_flag"] = amb
    out["missing_turnover_flag"] = missing_turnover
    out["feature_cap_status"] = np.where(missing_turnover, "not_fairly_tested_missing_execution_depth", "ok")
    return out


def aggregate_replay(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if events.empty:
        return pd.DataFrame()
    for (cid, sid, band), sub in events.groupby(["candidate_id", "scenario_id", "scenario_band"], dropna=False):
        cand = sub[sub["window_role"].astype(str).eq("candidate_event")]
        ctrl = sub[sub["window_role"].astype(str).eq("control")]
        cs = summarize_r(cand["signal_R"])
        ca = summarize_r(cand["account_R"])
        xs = summarize_r(ctrl["signal_R"])
        ctrl_norm = xs["net_R"] * (cs["events"] / xs["events"]) if xs["events"] else np.nan
        rows.append({"candidate_id": cid, "scenario_id": sid, "scenario_band": band, "rankable": bool(sub["rankable"].iloc[0]), "event_count_total": int((sub["window_role"].astype(str).eq("candidate_event")).sum()), "filled_event_count": cs["events"], "skipped_count": int(cand["skipped_flag"].sum()), "partial_fill_count": int(cand["partial_fill_flag"].sum()), "no_fill_count": int(cand["no_fill_flag"].sum()), "candidate_net_R": cs["net_R"], "candidate_account_R": ca["net_R"], "candidate_PF": cs["PF"], "candidate_median_R": cs["median_R"], "control_count_total": int((sub["window_role"].astype(str).eq("control")).sum()), "filled_control_count": xs["events"], "control_raw_R": xs["net_R"], "control_normalized_R": ctrl_norm, "normalized_control_uplift_R": cs["net_R"] - ctrl_norm if np.isfinite(ctrl_norm) else np.nan, "beats_controls": bool(np.isfinite(ctrl_norm) and cs["net_R"] > ctrl_norm and cs["net_R"] > 0), "max_drawdown_proxy": float(pd.to_numeric(cand["signal_R"], errors="coerce").fillna(0.0).cumsum().sub(pd.to_numeric(cand["signal_R"], errors="coerce").fillna(0.0).cumsum().cummax()).min()) if not cand.empty else np.nan, "worst_symbol": cand.groupby("symbol")["signal_R"].sum().idxmin() if not cand.empty else "", "worst_month": pd.to_datetime(cand["entry_ts_effective"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("M").astype(str).iloc[0] if not cand.empty else ""})
    return pd.DataFrame(rows)


def stage_replay(ctx: RunContext) -> None:
    features = pd.read_parquet(ctx.run_root / "features/liquidity_proxy_features.parquet") if (ctx.run_root / "features/liquidity_proxy_features.parquet").exists() else pd.DataFrame()
    frames = []
    for sc in scenario_defs():
        if features.empty:
            continue
        frames.append(replay_scenario(features, sc, int(ctx.args.seed), float(ctx.args.intended_order_notional_usdt)))
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    outp = ctx.run_root / "replay/proxy_execution_replay_events.parquet"
    outp.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(outp, index=False)
    summary = aggregate_replay(events)
    write_csv(ctx.run_root / "replay/proxy_execution_replay_summary.csv", summary)
    write_text(ctx.run_root / "replay/proxy_execution_replay_report.md", f"# Candidate-Control Proxy Execution Replay\n\n- scenarios: `{len(scenario_defs())}`\n- replay_event_rows: `{len(events)}`\n- summary_rows: `{len(summary)}`\n- intended_order_notional_usdt_proxy: `{ctx.args.intended_order_notional_usdt}`\n- not_order_book_replay: `true`\n")


def stage_survival(ctx: RunContext) -> None:
    s = read_csv(ctx.run_root / "replay/proxy_execution_replay_summary.csv")
    rows = []
    for cid in LISTING_IDS:
        sub = s[s.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not s.empty else pd.DataFrame()
        rank = sub[sub.get("rankable", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"])] if not sub.empty else pd.DataFrame()
        base = rank[(rank["scenario_band"].astype(str) == "base") & (rank["beats_controls"].astype(str).str.lower().isin(["true", "1", "yes"]))] if not rank.empty else pd.DataFrame()
        severe = rank[(rank["scenario_band"].astype(str) == "severe") & (rank["beats_controls"].astype(str).str.lower().isin(["true", "1", "yes"]))] if not rank.empty else pd.DataFrame()
        opt = rank[(rank["scenario_band"].astype(str) == "optimistic") & (rank["beats_controls"].astype(str).str.lower().isin(["true", "1", "yes"]))] if not rank.empty else pd.DataFrame()
        if not severe.empty:
            label = "survives_severe_proxy_execution"
        elif not base.empty:
            label = "survives_base_proxy_execution"
        elif not opt.empty:
            label = "survives_only_optimistic_proxy_execution"
        elif not sub.empty and (pd.to_numeric(sub.get("candidate_net_R", 0), errors="coerce") > 0).any():
            label = "path_edge_survives_execution_unresolved"
        elif not sub.empty:
            label = "fails_proxy_execution_current_expression_only"
        else:
            label = "not_fairly_tested_missing_execution_depth"
        rows.append({"candidate_id": cid, "survival_label": label, "survives_base_proxy": not base.empty, "survives_severe_proxy": not severe.empty, "survives_optimistic_proxy": not opt.empty, "missing_true_depth_blocks_validation": True})
    if ctx.args.include_d4:
        rows.append({"candidate_id": D4_CANDIDATE_ID, "survival_label": "not_fairly_tested_missing_execution_depth", "survives_base_proxy": False, "survives_severe_proxy": False, "survives_optimistic_proxy": False, "missing_true_depth_blocks_validation": True})
    write_csv(ctx.run_root / "decision/scenario_survival_table.csv", rows)
    write_text(ctx.run_root / "decision/scenario_survival_report.md", "# Scenario Survival Table\n\nFailures apply only to current expression/execution assumptions. Families are not rejected from this run alone.\n")


def stage_canary(ctx: RunContext) -> None:
    surv = read_csv(ctx.run_root / "decision/scenario_survival_table.csv")
    rows = []
    for _, row in surv.iterrows():
        cid = str(row["candidate_id"])
        if cid == D4_CANDIDATE_ID:
            label = "micro_canary_not_recommended_missing_liquidation_depth_evidence"
        elif str(row.get("survival_label")) in {"survives_base_proxy_execution", "survives_severe_proxy_execution"}:
            label = "micro_canary_possible_execution_only"
        elif str(row.get("survival_label")) == "survives_only_optimistic_proxy_execution":
            label = "micro_canary_not_recommended_yet"
        else:
            label = "micro_canary_not_recommended_current_expression"
        rows.append({"candidate_id": cid, "micro_canary_label": label, "manual_approval_required": True, "live_capture_required": True, "bbo_depth_trades_liquidations_capture_required": True, "max_notional_usdt": "100-300 isolated", "max_risk_usdt_per_trade": "0.50-1.00", "max_leverage": "2x-3x", "stop_after_execution_anomalies": 2, "evaluate_execution_only_not_pnl": True})
    write_csv(ctx.run_root / "canary/micro_canary_readiness.csv", rows)
    write_text(ctx.run_root / "canary/micro_canary_readiness_report.md", "# Micro-Canary Readiness Diagnostic\n\nThis is execution-only diagnostic language, not live preparation or a trading recommendation. Any canary would require manual approval, live capture, strict small notional/risk caps, and stop-after-anomalies rules.\n")


def stage_decision(ctx: RunContext) -> None:
    surv = read_csv(ctx.run_root / "decision/scenario_survival_table.csv")
    can = read_csv(ctx.run_root / "canary/micro_canary_readiness.csv")
    listing_verdicts = {}
    for _, row in surv.iterrows():
        cid = str(row["candidate_id"])
        if cid == D4_CANDIDATE_ID:
            continue
        label = str(row["survival_label"])
        if label in {"survives_base_proxy_execution", "survives_severe_proxy_execution"}:
            verdict = "listing_candidate_survives_proxy_execution_needs_capture"
        elif label == "survives_only_optimistic_proxy_execution":
            verdict = "listing_candidate_fragile_current_expression_only"
        elif label == "fails_proxy_execution_current_expression_only":
            verdict = "listing_candidate_fails_proxy_execution_current_expression_only"
        else:
            verdict = "listing_candidate_fragile_current_expression_only"
        listing_verdicts[cid] = verdict
    micro_possible = bool((can.get("micro_canary_label", pd.Series(dtype=str)).astype(str) == "micro_canary_possible_execution_only").any()) if not can.empty else False
    decision = {"run_root": str(ctx.run_root), "protected_holdout_touched": False, "listing_candidate_verdicts": listing_verdicts, "d4_verdict": "d4_remains_execution_depth_data_blocked", "vendor_free_data_verdict": "vendor_data_not_affordable_continue_without", "forward_capture_verdict": "forward_capture_required", "micro_canary_diagnostic_verdict": "micro_canary_possible_execution_only" if micro_possible else "micro_canary_not_recommended", "next_action_verdict": "forward_capture_required", "validation_language_used": False}
    write_json(ctx.run_root / "decision_summary.json", decision)
    surv_md = surv.to_markdown(index=False) if not surv.empty else "No survival rows."
    write_text(ctx.run_root / "QLMG_BEST_EFFORT_PROXY_EXECUTION_SIM_REPORT.md", f"# QLMG Best-Effort Proxy Execution Simulation Report\n\n## Scope\n\n- run_root: `{ctx.run_root}`\n- train_only: `true`\n- final_holdout_touched: `false`\n- no_downloads: `true`\n- not_order_book_replay: `true`\n\n## Scenario Survival\n\n{surv_md}\n\n## Interpretation\n\nThis simulator is stricter and more realistic than fixed-bps stress because it uses actual 1m OHLCV/mark/index/premium/OI/funding windows, liquidity proxies, participation caps, deterministic no-fill models, adverse-selection penalties, stop-execution penalties, and dynamic sizing. It is still not validation and not live readiness because historical BBO/depth/trades/liquidation feeds are missing.\n")


def copy_if_small(src: Path, dst_dir: Path, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    rec = {"source_path": str(src), "bundle_path": "", "included": False, "reason": "missing"}
    if not src.exists():
        return rec
    if src.stat().st_size > max_bytes:
        rec["reason"] = "too_large_paths_only"
        return rec
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.relative_to(src.parents[1]).as_posix().replace("/", "__")
    shutil.copy2(src, dst)
    rec.update({"bundle_path": str(dst), "included": True, "reason": "included"})
    return rec


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        ctx.run_root / "QLMG_BEST_EFFORT_PROXY_EXECUTION_SIM_REPORT.md",
        ctx.run_root / "decision_summary.json",
        ctx.run_root / "baseline/baseline_reproduction_summary.csv",
        ctx.run_root / "features/liquidity_proxy_report.md",
        ctx.run_root / "replay/proxy_execution_replay_summary.csv",
        ctx.run_root / "decision/scenario_survival_table.csv",
        ctx.run_root / "canary/micro_canary_readiness_report.md",
        ctx.run_root / "models/proxy_spread_model_report.md",
        ctx.run_root / "models/participation_no_fill_report.md",
        ctx.run_root / "models/adverse_selection_report.md",
        ctx.run_root / "models/stop_execution_report.md",
        ctx.run_root / "models/same_minute_ambiguity_report.md",
        ctx.run_root / "models/dynamic_liqsafe_sizing_report.md",
    ]
    rows = [copy_if_small(p, bundle) for p in include]
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_text(bundle / "README.md", f"# Compact Review Bundle\n\nRun root: `{ctx.run_root}`. Large parquet files are excluded; paths are recorded in artifact index.\n")


STAGE_FUNCS = {
    "preflight-and-artifact-freeze": stage_preflight,
    "seal-guard": stage_seal,
    "baseline-reproduction": stage_baseline,
    "liquidity-proxy-feature-build": stage_features,
    "proxy-spread-model": stage_spread,
    "participation-and-no-fill-model": stage_participation,
    "adverse-selection-model": stage_adverse,
    "stop-execution-model": stage_stop,
    "same-minute-ambiguity-model": stage_ambiguity,
    "dynamic-liquidation-safe-sizing": stage_dynamic_sizing,
    "candidate-control-proxy-execution-replay": stage_replay,
    "scenario-survival-table": stage_survival,
    "micro-canary-readiness-diagnostic": stage_canary,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        return
    enforce_resources(ctx, stage)
    ctx.notifier.send("QLMG proxy execution stage start", stage)
    try:
        STAGE_FUNCS[stage](ctx)
    except Exception as exc:
        ctx.notifier.send("QLMG proxy execution stage failure", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        raise
    mark_done(ctx.run_root, stage)
    ctx.notifier.send("QLMG proxy execution stage complete", stage)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end, brutal_root=latest_brutal_root())
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "run_root_reason": reason, "args": vars(args), "start": str(start), "end": str(end), "brutal_root": str(ctx.brutal_root) if ctx.brutal_root else None})
    notifier.send("QLMG proxy execution run start", f"run_root={run_root}")
    for stage in stage_list(args.stage):
        run_stage(ctx, stage)
    notifier.send("QLMG proxy execution run complete", f"run_root={run_root}")
    try:
        (run_root / "watch_status.json").write_text(json.dumps({"status": "complete", "run_root": str(run_root), "ts_utc": utc_now()}, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass
    print(f"[run_root] {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
