#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
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
    assert_pass,
    result_to_jsonable,
    scan_output_tree_for_protected,
    validate_control_rows,
    validate_event_trade_schema,
    validate_funding_mark_flags,
    validate_no_projected_metric_promotion,
    validate_pit_feature_timestamps,
)
from tools.qlmg_match_feature_builder import enrich_event_pool_with_match_features  # noqa: E402
from tools.qlmg_real_controls import build_real_controls, stable_hash, standardize_event_ledger  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_no_vendor_progress_v2_20260630_v1"
DEFAULT_SEED = 20260630
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
DATA_5M = Path("/opt/parquet/5m")
CONTEXT_5M = Path("/opt/parquet/bybit_context_5m")
FUNDING_HINT_ROOTS = [Path("/opt/parquet/bybit_funding"), Path("/opt/parquet/funding"), Path("/opt/parquet/bybit_context_5m")]
LIVE_CAPTURE_ZIP = REPO / "research_inputs/qlmg_live_capture.zip"
EXPECTED_LIVE_CAPTURE_SHA = "ee88a2b0c0b3e81cc5b18aa9715747208170d498b1cf8205e751402df43442e1"

PREV_NO_VENDOR_ROOT = RESULTS_ROOT / "phase_qlmg_no_vendor_progress_run_20260630_v1_20260630_082124"
MECHANICAL_QA_ROOT = RESULTS_ROOT / "phase_qlmg_mechanical_qa_evidence_contract_20260630_v1_20260630_074328"
LEAKAGE_GUARD_ROOT = RESULTS_ROOT / "phase_qlmg_leakage_guard_rebaseline_20260629_v1_20260629_174557"
EVIDENCE_REPAIR_ROOT = RESULTS_ROOT / "phase_qlmg_evidence_remediation_family_repair_20260629_v1_20260629_044410"
ABCX_ROOT = RESULTS_ROOT / "phase_qlmg_integrated_abcx_development_20260628_v2_20260628_134140"
REAL_CONTROL_ROOT = RESULTS_ROOT / "phase_qlmg_real_control_rebuild_20260629_v1_20260629_170608"
GLOBAL_INVALIDATION_ROOT = RESULTS_ROOT / "phase_qlmg_global_result_invalidation_audit_20260629_v1_20260629_171528"
D4_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"

A3_LEDGER = LEAKAGE_GUARD_ROOT / "corrected_sweep/a3_sweep/a3_event_level_replay.parquet"
A2_LEDGER = LEAKAGE_GUARD_ROOT / "corrected_sweep/a2_sweep/a2_event_level_replay.parquet"
C2_EVENT_LEDGER = ABCX_ROOT / "c2/catalyst_event_ledger.parquet"
C2_REPAIR_LEDGER = EVIDENCE_REPAIR_ROOT / "c2_trade_ledger/c2_event_level_replay.parquet"

QA_REQUIRED_FILES = [
    MECHANICAL_QA_ROOT / "decision_summary.json",
    MECHANICAL_QA_ROOT / "contracts/event_level_evidence_contract.md",
    MECHANICAL_QA_ROOT / "contracts/control_engine_contract.md",
    MECHANICAL_QA_ROOT / "contracts/funding_mark_contract.md",
    MECHANICAL_QA_ROOT / "quarantine/refreshed_quarantine_manifest.csv",
    MECHANICAL_QA_ROOT / "quarantine/deprecated_promotion_labels.csv",
]

ACTIVE_ROOTS = {
    "previous_no_vendor": PREV_NO_VENDOR_ROOT,
    "mechanical_qa": MECHANICAL_QA_ROOT,
    "leakage_guard": LEAKAGE_GUARD_ROOT,
    "real_control_rebuild": REAL_CONTROL_ROOT,
    "global_invalidation_audit": GLOBAL_INVALIDATION_ROOT,
    "evidence_repair": EVIDENCE_REPAIR_ROOT,
    "integrated_abcx": ABCX_ROOT,
    "d4_survivability": D4_ROOT,
    "listing_generic": LISTING_ROOT,
    "data_5m": DATA_5M,
    "context_5m": CONTEXT_5M,
    "live_capture_zip": LIVE_CAPTURE_ZIP,
}

STAGES = (
    "preflight-and-manual-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "previous-no-vendor-run-audit",
    "mark-funding-enrichment-root-cause",
    "true-a1-a4-a3-overlay-candidate-registry",
    "event-level-replay-and-real-controls",
    "tier1-stress-and-sleeve-classification",
    "c2-mechanism-event-expansion",
    "branch-x-capture-calibration",
    "no-vendor-decision-matrix",
    "candidate-library-refresh",
    "decision-report",
    "compact-review-bundle",
    "all",
)

NO_VENDOR_OUTCOMES = [
    "progress_with_current_data",
    "needs_capture_substitute",
    "redesign_to_less_depth_sensitive",
    "discard_current_translation_no_vendor_path",
    "preserve_hypothesis_generate_new_variant",
    "candidate_library_only",
]

ALLOWED_NEXT_DECISIONS = [
    "run_train_only_candidate_validation_package_next",
    "run_a1_a4_redesign_again",
    "run_b1_c2_mechanism_validation_next",
    "branch_x_micro_canary_possible_execution_only",
    "request_72h_branch_x_capture",
    "generate_new_hypotheses_next",
    "discard_no_vendor_current_translations",
    "blocked_by_protocol_issue",
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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-no-vendor-progress-v2")
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
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        write_json(self.run_root / "watch_status.json", {"status": "running", "last_event": title, "ts_utc": rec["ts_utc"], "run_root": str(self.run_root)})
        return sent

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG no-vendor progress v2 candidate development")
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
    p.add_argument("--include-funding-mark-enrichment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-a1-a4-sweep", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-c2-expansion", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-branch-x-capture-calibration", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--a1-budget", type=int, default=2000)
    p.add_argument("--a4-budget", type=int, default=2000)
    p.add_argument("--a3-budget", type=int, default=1000)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--top-per-family", type=int, default=80)
    p.add_argument("--tmux-session-name", default="qlmg_no_vendor_progress_v2")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--run-root", default="")
    p.add_argument("--operator-attested-live-capture", action=argparse.BooleanOptionalAction, default=True)
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
    if start >= PROTECTED_TS or end >= PROTECTED_TS:
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
        writer.writerows(rows_list)

def done_path(ctx: RunContext, stage: str) -> Path:
    return ctx.run_root / "stage_status" / f"{stage}.done"

def mark_done(ctx: RunContext, stage: str) -> None:
    p = done_path(ctx, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"stage": stage, "completed_utc": utc_now()}, sort_keys=True) + "\n")

def file_sha256(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while True:
            n = 1024 * 1024
            if max_bytes is not None:
                remain = max_bytes - read
                if remain <= 0:
                    break
                n = min(n, remain)
            chunk = f.read(n)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()

def read_df(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            return pd.read_parquet(path)
    return pd.read_csv(path)

def pf(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(vals[vals > 0].sum())
    losses = float(vals[vals < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / abs(losses)

def max_dd(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if vals.size == 0:
        return 0.0
    curve = np.cumsum(vals)
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve - peak))

def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default

def split_label(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return np.where(t < pd.Timestamp("2024-07-01T00:00:00Z"), "development", "internal_validation")

def estimate_funding_crosses(entry_ts: pd.Series, exit_ts: pd.Series) -> pd.Series:
    entry = pd.to_datetime(entry_ts, utc=True, errors="coerce")
    exit_ = pd.to_datetime(exit_ts, utc=True, errors="coerce")
    out = []
    for a, b in zip(entry, exit_):
        if pd.isna(a) or pd.isna(b) or b <= a:
            out.append(0)
            continue
        cur = a.floor("8h")
        n = 0
        while cur <= b:
            if cur > a:
                n += 1
            cur += pd.Timedelta(hours=8)
        out.append(n)
    return pd.Series(out, index=entry_ts.index, dtype="int64")

def normalize_event_ledger(df: pd.DataFrame, family: str, branch: str, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["family"] = out.get("family", family)
    out["family"] = out["family"].fillna(family).astype(str)
    out["branch_id"] = out.get("branch_id", branch)
    out["candidate_id"] = out.get("candidate_id", out.get("variant_id", family + "__candidate"))
    out["candidate_id"] = out["candidate_id"].fillna(family + "__candidate").astype(str)
    out["definition_id"] = out.get("definition_id", out["candidate_id"])
    out["event_id"] = out.get("event_id", pd.Series([stable_hash(source, i) for i in range(len(out))], index=out.index))
    for col in ["decision_ts", "entry_ts", "exit_ts"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out = out[out["decision_ts"].notna() & (out["decision_ts"] < PROTECTED_TS)].copy()
    out["side"] = out.get("side", "long")
    for col in ["entry_price", "exit_price", "stop_price", "target_price", "risk_bps_used"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    gross_src = out["gross_R"] if "gross_R" in out.columns else (out["net_R"] if "net_R" in out.columns else pd.Series(0.0, index=out.index))
    fees_src = out["fees_R"] if "fees_R" in out.columns else (out["cost_R"] if "cost_R" in out.columns else pd.Series(0.0, index=out.index))
    out["gross_R"] = pd.to_numeric(gross_src, errors="coerce")
    out["fees_R"] = pd.to_numeric(fees_src, errors="coerce").fillna(0.0)
    out["slippage_R"] = pd.to_numeric(out.get("slippage_R", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    out["funding_R"] = pd.to_numeric(out.get("funding_R", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    if "net_R" not in out.columns:
        out["net_R"] = out["gross_R"] - out["fees_R"] - out["slippage_R"] + out["funding_R"]
    out["net_R"] = pd.to_numeric(out["net_R"], errors="coerce")
    out["exit_rule"] = out.get("exit_rule", "fixed_stop_target_time")
    out["exit_reason"] = out.get("exit_reason", "time")
    out["entry_price_source"] = out.get("entry_price_source", "local_5m_next_bar_open_or_close")
    out["mark_liquidation_flag"] = pd.Series(out.get("mark_liquidation_flag", False), index=out.index).fillna(False).astype(bool)
    out["same_bar_ambiguity_flag"] = pd.Series(out.get("same_bar_ambiguity_flag", False), index=out.index).fillna(False).astype(bool)
    if "funding_timestamps_crossed" not in out.columns:
        out["funding_timestamps_crossed"] = estimate_funding_crosses(out["entry_ts"], out["exit_ts"])
    else:
        out["funding_timestamps_crossed"] = pd.to_numeric(out["funding_timestamps_crossed"], errors="coerce").fillna(0).astype(int)
    out["mark_available"] = pd.Series(out.get("mark_available", out.get("mark_price_available", False)), index=out.index).fillna(False).astype(bool)
    out["funding_exact"] = pd.Series(out.get("funding_exact", False), index=out.index).fillna(False).astype(bool)
    out.loc[out["funding_timestamps_crossed"] == 0, "funding_exact"] = True
    out["mark_proxy_used"] = ~out["mark_available"]
    out["funding_proxy_used"] = ~out["funding_exact"]
    out["lifecycle_status"] = out.get("lifecycle_status", "local_bar_presence_only")
    out["data_tier"] = out.get("data_tier", out.get("required_data_tier", "Tier1"))
    out["control_group_id"] = out.get("control_group_id", out["candidate_id"])
    out["source_data_hash"] = out.get("source_data_hash", stable_hash(source))
    out["metric_basis"] = out.get("metric_basis", "event_level_trade_ledger_no_vendor_v2_train_only")
    out["required_data_tier"] = out.get("required_data_tier", "Tier1")
    out["current_data_tier"] = out.get("current_data_tier", "Tier1_local_5m_context_proxy_funding")
    out["label_cap_reason"] = np.where(out["funding_proxy_used"], "funding_crossed_without_exact_settlement_source", "none")
    out.loc[out["mark_proxy_used"], "label_cap_reason"] = out.loc[out["mark_proxy_used"], "label_cap_reason"].astype(str) + ";mark_proxy_or_missing"
    out["parent_regime"] = out.get("parent_regime", "unknown")
    out["event_cluster_id"] = out.get("event_cluster_id", out["symbol"].astype(str) + "_" + out["decision_ts"].dt.strftime("%Y%m"))
    out["split"] = out.get("split", split_label(out["decision_ts"]))
    out["source_path"] = source
    cols = [
        "event_id", "candidate_id", "definition_id", "family", "branch_id", "symbol", "decision_ts", "entry_ts", "exit_ts", "side",
        "entry_price", "exit_price", "stop_price", "target_price", "risk_bps_used", "gross_R", "fees_R", "slippage_R", "funding_R", "net_R",
        "entry_price_source", "exit_rule", "exit_reason", "mark_liquidation_flag", "same_bar_ambiguity_flag", "funding_timestamps_crossed",
        "mark_available", "funding_exact", "mark_proxy_used", "funding_proxy_used", "lifecycle_status", "data_tier", "control_group_id", "source_data_hash",
        "metric_basis", "required_data_tier", "current_data_tier", "label_cap_reason", "parent_regime", "event_cluster_id", "split", "source_path",
    ]
    return out[[c for c in cols if c in out.columns]].copy()

def load_symbol_5m(symbol: str, start: pd.Timestamp, end: pd.Timestamp, columns: list[str] | None = None) -> pd.DataFrame:
    path = DATA_5M / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_parquet(path, columns=columns)
    except Exception:
        raw = pd.read_parquet(path)
        if columns:
            raw = raw[[c for c in columns if c in raw.columns]].copy()
    if raw.empty or "timestamp" not in raw.columns:
        return pd.DataFrame()
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna() & (df["timestamp"] >= start) & (df["timestamp"] <= end) & (df["timestamp"] < PROTECTED_TS)]
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

def symbol_universe(max_symbols: int = 0) -> list[str]:
    files = sorted(DATA_5M.glob("*.parquet"))
    symbols = [p.stem for p in files]
    if max_symbols and max_symbols > 0:
        return symbols[:max_symbols]
    return symbols

def daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.copy()
    x["date"] = x["timestamp"].dt.floor("D")
    agg = x.groupby("date", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), turnover=("turnover", "sum"), open_interest=("open_interest", "last"),
        funding_rate=("funding_rate", "last"), decision_ts=("timestamp", "last"),
    ).reset_index()
    return agg

def replay_path(symbol: str, side: str, entry_ts: pd.Timestamp, entry_price: float, stop_price: float, target_price: float, max_hold_hours: int) -> dict[str, Any] | None:
    end = min(entry_ts + pd.Timedelta(hours=max_hold_hours), SCREENING_END)
    bars = load_symbol_5m(symbol, entry_ts, end, columns=["timestamp", "open", "high", "low", "close", "turnover", "open_interest", "funding_rate"])
    if bars.empty:
        return None
    risk = abs(entry_price - stop_price)
    if not math.isfinite(risk) or risk <= 0:
        return None
    exit_price = float(bars["close"].iloc[-1])
    exit_ts = pd.Timestamp(bars["timestamp"].iloc[-1])
    reason = "time"
    same_bar = False
    for _, r in bars.iterrows():
        hi = float(r.get("high"))
        lo = float(r.get("low"))
        ts = pd.Timestamp(r.get("timestamp"))
        if side == "long":
            stop_hit = lo <= stop_price
            target_hit = hi >= target_price
            if stop_hit and target_hit:
                exit_price, exit_ts, reason, same_bar = stop_price, ts, "stop_same_bar", True
                break
            if stop_hit:
                exit_price, exit_ts, reason = stop_price, ts, "stop"
                break
            if target_hit:
                exit_price, exit_ts, reason = target_price, ts, "target"
                break
        else:
            stop_hit = hi >= stop_price
            target_hit = lo <= target_price
            if stop_hit and target_hit:
                exit_price, exit_ts, reason, same_bar = stop_price, ts, "stop_same_bar", True
                break
            if stop_hit:
                exit_price, exit_ts, reason = stop_price, ts, "stop"
                break
            if target_hit:
                exit_price, exit_ts, reason = target_price, ts, "target"
                break
    gross = (exit_price - entry_price) / risk if side == "long" else (entry_price - exit_price) / risk
    risk_bps = risk / entry_price * 10000.0 if entry_price else np.nan
    fees_R = 8.0 / risk_bps if risk_bps and math.isfinite(risk_bps) and risk_bps > 0 else np.nan
    return {
        "exit_ts": exit_ts,
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "gross_R": float(gross),
        "fees_R": float(fees_R),
        "slippage_R": 0.0,
        "funding_R": 0.0,
        "net_R": float(gross - fees_R) if math.isfinite(fees_R) else float(gross),
        "risk_bps_used": float(risk_bps),
        "same_bar_ambiguity_flag": bool(same_bar),
    }

_MARK_CONTEXT_CACHE: dict[str, pd.DataFrame] = {}

def load_mark_context(symbol: str) -> pd.DataFrame:
    if symbol in _MARK_CONTEXT_CACHE:
        return _MARK_CONTEXT_CACHE[symbol]
    path = CONTEXT_5M / f"{symbol}.parquet"
    if not path.exists():
        _MARK_CONTEXT_CACHE[symbol] = pd.DataFrame()
        return _MARK_CONTEXT_CACHE[symbol]
    try:
        ctx = pd.read_parquet(path)
    except Exception:
        _MARK_CONTEXT_CACHE[symbol] = pd.DataFrame()
        return _MARK_CONTEXT_CACHE[symbol]
    if ctx.empty:
        _MARK_CONTEXT_CACHE[symbol] = pd.DataFrame()
        return _MARK_CONTEXT_CACHE[symbol]
    ctx = ctx.copy()
    ts_col = "timestamp" if "timestamp" in ctx.columns else ("ts" if "ts" in ctx.columns else None)
    if ts_col is None:
        _MARK_CONTEXT_CACHE[symbol] = pd.DataFrame()
        return _MARK_CONTEXT_CACHE[symbol]
    ctx["timestamp"] = pd.to_datetime(ctx[ts_col], utc=True, errors="coerce")
    for c in ["mark_source_close_ts", "context_source_close_ts", "source_close_ts"]:
        if c in ctx.columns:
            ctx[c] = pd.to_datetime(ctx[c], utc=True, errors="coerce")
    ctx = ctx[ctx["timestamp"].notna() & (ctx["timestamp"] < PROTECTED_TS)].sort_values("timestamp", kind="mergesort")
    _MARK_CONTEXT_CACHE[symbol] = ctx
    return ctx

def mark_available_for(symbol: str, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> bool:
    ctx = load_mark_context(symbol)
    if ctx.empty:
        return False
    mark_cols = [c for c in ctx.columns if "mark" in c.lower() and any(k in c.lower() for k in ["close", "price"])]
    if not mark_cols:
        return False
    span = ctx[(ctx["timestamp"] >= entry_ts) & (ctx["timestamp"] <= exit_ts)]
    if span.empty or span[mark_cols[0]].isna().all():
        return False
    source_cols = [c for c in ["mark_source_close_ts", "context_source_close_ts", "source_close_ts"] if c in span.columns]
    if source_cols:
        src = span[source_cols[0]]
        valid = src.notna()
        if bool(valid.any()):
            return bool((src[valid] <= span.loc[valid, "timestamp"]).all())
    return False

def funding_source_exists() -> bool:
    return any(p.exists() and any(p.glob("*.parquet")) for p in FUNDING_HINT_ROOTS if p.exists())

def candidate_metrics(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for cid, g in events.groupby("candidate_id", sort=False):
        vals = pd.to_numeric(g["net_R"], errors="coerce")
        n = int(vals.notna().sum())
        months = g["decision_ts"].dt.strftime("%Y-%m")
        month_net = vals.groupby(months).sum()
        sym_net = vals.groupby(g["symbol"].astype(str)).sum()
        total = float(vals.sum()) if n else np.nan
        risk_bps = pd.to_numeric(g.get("risk_bps_used"), errors="coerce").replace(0, np.nan)
        stress25 = vals - (25.0 / risk_bps).fillna(0.0)
        stress50 = vals - (50.0 / risk_bps).fillna(0.0)
        funding_cross = pd.to_numeric(g.get("funding_timestamps_crossed"), errors="coerce").fillna(0)
        adverse_funding = vals - ((funding_cross * 2.0) / risk_bps).fillna(0.0)
        rows.append({
            "candidate_id": cid,
            "family": str(g["family"].iloc[0]),
            "branch_id": str(g["branch_id"].iloc[0]),
            "event_count": n,
            "net_R": total,
            "PF": pf(vals),
            "win_rate": float((vals > 0).mean()) if n else np.nan,
            "avg_R": float(vals.mean()) if n else np.nan,
            "median_R": float(vals.median()) if n else np.nan,
            "max_dd_R": max_dd(vals.reset_index(drop=True)),
            "active_months": int(months.nunique()),
            "active_symbols": int(g["symbol"].nunique()),
            "dominant_month_share": float(month_net.max() / total) if n and total and total > 0 and len(month_net) else np.nan,
            "dominant_symbol_share": float(sym_net.max() / total) if n and total and total > 0 and len(sym_net) else np.nan,
            "base_stress_net_R": total,
            "plus25bps_net_R": float(stress25.sum()) if n else np.nan,
            "plus50bps_net_R": float(stress50.sum()) if n else np.nan,
            "adverse_funding_net_R": float(adverse_funding.sum()) if n else np.nan,
            "mark_available": bool(g["mark_available"].fillna(False).astype(bool).all()),
            "funding_exact": bool(g["funding_exact"].fillna(False).astype(bool).all()),
            "mark_proxy_used": bool(g["mark_proxy_used"].fillna(False).astype(bool).any()),
            "funding_proxy_used": bool(g["funding_proxy_used"].fillna(False).astype(bool).any()),
            "label_cap_reason": ";".join(sorted(set(g["label_cap_reason"].dropna().astype(str))))[:300],
            "metric_basis": "event_level_trade_ledger_no_vendor_v2_train_only",
        })
    return pd.DataFrame(rows)

# ---------- stages ----------

def resolve_manual_path() -> Path:
    candidates = [REPO / "docs/QLMG_BACKTESTING_MANUAL_20260630_FULL.md", REPO / "research_inputs/testmanual.txt"]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise RuntimeError("active manual missing: expected docs/QLMG_BACKTESTING_MANUAL_20260630_FULL.md or research_inputs/testmanual.txt")

def stage_preflight(ctx: RunContext) -> None:
    ctx.run_root.mkdir(parents=True, exist_ok=True)
    manual = resolve_manual_path()
    rows = [{"name": "active_manual", "path": str(manual), "exists": True, "is_file": True, "is_dir": False}]
    hashes = {"active_manual": file_sha256(manual)}
    for name, path in ACTIVE_ROOTS.items():
        exists = path.exists()
        rows.append({"name": name, "path": str(path), "exists": exists, "is_file": path.is_file(), "is_dir": path.is_dir()})
        if exists and path.is_file() and path.stat().st_size < 200_000_000:
            hashes[name] = file_sha256(path)
        elif exists and path.is_file():
            hashes[name] = file_sha256(path, max_bytes=32 * 1024 * 1024) + "_first32mb"
    missing_required = [str(p) for p in QA_REQUIRED_FILES if not p.exists()]
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    write_json(ctx.run_root / "preflight/manual_source_status.json", {"manual_path": str(manual), "sha256": hashes["active_manual"]})
    snap = resource_snapshot(ctx.run_root.parent)
    guard = check_resource_guard(snap, estimated_output_gb=2.0 if ctx.args.smoke else 18.0, hard_stage_output_gb=35.0, allow_large_output=ctx.args.allow_large_output)
    write_json(ctx.run_root / "preflight/resource_guard_report.md.json", guard)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", "# Resource Guard\n\n" + json.dumps(guard, indent=2))
    if guard["status"] == "hard_stop":
        raise RuntimeError("resource guard hard stop: " + ";".join(guard["reasons"]))
    if missing_required:
        write_text(ctx.run_root / "preflight/preflight_report.md", "# Preflight\n\nMissing required QA artifacts:\n" + "\n".join(f"- {m}" for m in missing_required))
        raise RuntimeError("missing required mechanical QA artifacts")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight\n\nManual frozen: `{manual}`. Mechanical QA artifacts present. V2 uses local pre-holdout data only.\n")

def stage_telegram(ctx: RunContext) -> None:
    write_json(ctx.run_root / "notifications/telegram_readiness.json", {"status": ctx.notifier.status, "remote_available": ctx.notifier.remote_available, "missing": ctx.notifier.missing})
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\nstatus: {ctx.notifier.status}\nremote_available: {ctx.notifier.remote_available}\n")
    watch = [
        f"tmux attach -t {ctx.args.tmux_session_name}",
        f"tail -f {ctx.run_root}/logs/full_run.log",
        f"watch -n 30 'cat {ctx.run_root}/watch_status.json'",
        f"tail -f {ctx.run_root}/notifications/telegram_events.jsonl",
        "df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h",
    ]
    write_text(ctx.run_root / "tmux/watch_commands.md", "# Watch Commands\n\n" + "\n".join(f"- `{x}`" for x in watch))
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux\n\nDefault session: `{ctx.args.tmux_session_name}`\n")

def stage_seal(ctx: RunContext) -> None:
    check = {"protected_start": str(PROTECTED_TS), "screening_end": str(SCREENING_END), "requested_start": str(ctx.start), "requested_end": str(ctx.end), "status": "pass"}
    if ctx.start >= PROTECTED_TS or ctx.end >= PROTECTED_TS:
        check["status"] = "fail"
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    write_text(ctx.run_root / "seal/seal_guard_report.md", "# Seal Guard\n\nFinal holdout protected from 2026-01-01T00:00:00Z onward. Status: pass.\n")
    if check["status"] != "pass":
        raise RuntimeError("protected slice violation")

def stage_previous_audit(ctx: RunContext) -> None:
    rows: list[dict[str, Any]] = []
    summary_path = PREV_NO_VENDOR_ROOT / "a1_a4/a1_a4_sweep_summary.csv"
    event_path = PREV_NO_VENDOR_ROOT / "a1_a4/a1_a4_event_level_replay.parquet"
    registry_candidates = sorted(PREV_NO_VENDOR_ROOT.rglob("*registry*.csv")) if PREV_NO_VENDOR_ROOT.exists() else []
    summary = read_df(summary_path)
    event = read_df(event_path, columns=["candidate_id", "family", "event_id"])
    for family in ["A1", "A4"]:
        s = summary[summary.get("family", pd.Series(dtype=str)).astype(str).eq(family)] if not summary.empty else pd.DataFrame()
        e = event[event.get("family", pd.Series(dtype=str)).astype(str).eq(family)] if not event.empty else pd.DataFrame()
        rows.append({
            "family": family,
            "prior_summary_exists": summary_path.exists(),
            "prior_event_ledger_exists": event_path.exists(),
            "prior_summary_rows": int(len(s)),
            "prior_event_rows": int(len(e)),
            "prior_unique_candidate_ids": int(e["candidate_id"].nunique()) if not e.empty and "candidate_id" in e.columns else 0,
            "candidate_registry_files_found": ";".join(str(p) for p in registry_candidates),
            "real_registry_found": bool(registry_candidates),
            "budget_interpretation": "event_count_cap_or_single_translation" if int(len(s)) <= 1 else "multiple_candidate_rows",
        })
    verdict = "previous_a1_a4_sweep_was_single_translation_not_full_sweep"
    if registry_candidates and not summary.empty and summary["candidate_id"].nunique() > 20:
        verdict = "previous_a1_a4_sweep_registry_found"
    write_csv(ctx.run_root / "audit/a1_a4_candidate_registry_existence.csv", rows)
    write_text(ctx.run_root / "audit/previous_no_vendor_run_audit.md", "# Previous No-Vendor Run Audit\n\n" + f"Verdict: `{verdict}`\n\n" + pd.DataFrame(rows).to_markdown(index=False))

def stage_mark_funding(ctx: RunContext) -> None:
    if not ctx.args.include_funding_mark_enrichment:
        write_text(ctx.run_root / "funding_mark/root_cause_report.md", "# Mark/Funding Root Cause\n\nSkipped by CLI.\n")
        return
    source_paths = [
        PREV_NO_VENDOR_ROOT / "a1_a4/a1_a4_event_level_replay.parquet",
        PREV_NO_VENDOR_ROOT / "b1/b1_event_level_replay.parquet",
        PREV_NO_VENDOR_ROOT / "c2/c2_event_level_replay.parquet",
        A3_LEDGER,
        A2_LEDGER,
    ]
    parts = []
    for p in source_paths:
        df = read_df(p)
        if not df.empty:
            fam = str(df.get("family", pd.Series([p.parent.name])).iloc[0]) if "family" in df.columns else p.parent.name
            parts.append(normalize_event_ledger(df, fam, "source_recheck", str(p)))
    panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if not panel.empty:
        for (fam, symbol, month), g in panel.assign(month=panel["decision_ts"].dt.strftime("%Y-%m")).groupby(["family", "symbol", "month"], sort=False):
            ctx_exists = (CONTEXT_5M / f"{symbol}.parquet").exists()
            mark_any = bool(g["mark_available"].fillna(False).astype(bool).any())
            funding_crossed = bool((pd.to_numeric(g["funding_timestamps_crossed"], errors="coerce").fillna(0) > 0).any())
            funding_exact_any = bool(g["funding_exact"].fillna(False).astype(bool).any())
            rows.append({
                "family": fam,
                "symbol": symbol,
                "month": month,
                "events": len(g),
                "context_file_exists": ctx_exists,
                "exact_mark_available": mark_any,
                "mark_join_bug": bool(ctx_exists and not mark_any),
                "mark_not_available_for_window": bool((not ctx_exists) and not mark_any),
                "funding_no_cross": bool(not funding_crossed),
                "funding_exact_available": bool(funding_exact_any and funding_crossed),
                "funding_join_bug": bool(funding_source_exists() and funding_crossed and not funding_exact_any),
                "funding_exact_not_available": bool(funding_crossed and not funding_exact_any),
            })
    (ctx.run_root / "funding_mark").mkdir(parents=True, exist_ok=True)
    if not panel.empty:
        panel.to_parquet(ctx.run_root / "funding_mark/enriched_event_panel.parquet", index=False)
        assert_pass(validate_funding_mark_flags(panel))
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "funding_mark/enriched_event_panel.parquet", index=False)
    cov = pd.DataFrame(rows)
    write_csv(ctx.run_root / "funding_mark/coverage_by_family_symbol_month.csv", cov)
    mark_bug = int(cov.get("mark_join_bug", pd.Series(dtype=bool)).sum()) if not cov.empty else 0
    fund_bug = int(cov.get("funding_join_bug", pd.Series(dtype=bool)).sum()) if not cov.empty else 0
    verdict = "join_bug_suspected" if mark_bug or fund_bug else "proxy_caps_due_to_missing_or_unverified_exact_sources"
    write_text(ctx.run_root / "funding_mark/root_cause_report.md", f"# Mark/Funding Root Cause\n\nVerdict: `{verdict}`\n\nRows checked: {len(panel)}. Mark join-bug symbol-months: {mark_bug}. Funding join-bug symbol-months: {fund_bug}. V2 does not silently promote proxy mark/funding to exact.\n")

def make_candidate_registry(ctx: RunContext) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(ctx.args.seed)
    def add(family: str, i: int, params: dict[str, Any]) -> None:
        cid = f"{family}__v2__{stable_hash(family, i, json.dumps(params, sort_keys=True), n=12)}"
        rows.append({"candidate_id": cid, "definition_id": cid, "family": family, "rankable": family in {"A1", "A4"}, **params})
    for i in range(int(ctx.args.a1_budget)):
        add("A1", i, {
            "entry_rule": "close_confirmed_breakout", "touch_fills_allowed": False,
            "lookback_high_days": int([20, 30, 40, 60][i % 4]),
            "smooth_low_buffer": float([0.96, 0.97, 0.98, 0.99][(i // 4) % 4]),
            "vol_contract_max": float([0.08, 0.10, 0.12, 0.15][(i // 16) % 4]),
            "atr_stop_mult": float([1.2, 1.5, 1.8, 2.2][(i // 64) % 4]),
            "target_R": float([2.0, 2.5, 3.0][(i // 256) % 3]),
            "max_hold_hours": int([48, 72, 120][(i // 768) % 3]),
            "a2_prior_high_filter": bool((i // 11) % 2), "parent_regime_filter": bool((i // 17) % 2),
        })
    for i in range(int(ctx.args.a4_budget)):
        add("A4", i, {
            "entry_rule": "vol_managed_tsmom", "touch_fills_allowed": False,
            "lookback_days": int([20, 40, 60, 90, 120][i % 5]),
            "vol_window_days": int([20, 30, 60][(i // 5) % 3]),
            "ret_threshold": float([0.05, 0.10, 0.15, 0.20][(i // 15) % 4]),
            "vol_max": float([0.08, 0.10, 0.12, 0.16][(i // 60) % 4]),
            "atr_stop_mult": float([1.5, 2.0, 2.5][(i // 240) % 3]),
            "target_R": float([2.0, 2.5, 3.0][(i // 720) % 3]),
            "max_hold_hours": int([72, 120, 168][(i // 1440) % 3]),
            "inverse_vol_sizing": True, "parent_regime_filter": bool((i // 13) % 2),
        })
    for i in range(int(ctx.args.a3_budget)):
        add("A3_overlay", i, {
            "entry_rule": "close_confirmed_retest_reclaim_overlay", "touch_fills_allowed": False, "rankable_overlay_only": True,
            "source_idea": "prior_a3_rare_regime", "hold_delay_bars": int([1, 2, 3, 6][i % 4]),
            "no_lower_low_days": int([2, 3, 5][(i // 4) % 3]),
            "target_R": float([1.5, 2.0, 2.5][(i // 12) % 3]),
            "max_hold_hours": int([48, 72, 120][(i // 36) % 3]),
            "a2_prior_high_filter": True,
        })
    reg = pd.DataFrame(rows)
    return reg.sample(frac=1.0, random_state=int(ctx.args.seed)).sort_values(["family", "candidate_id"], kind="mergesort").reset_index(drop=True)

def stage_registry(ctx: RunContext) -> None:
    reg = make_candidate_registry(ctx)
    write_csv(ctx.run_root / "registry/candidate_registry.csv", reg)
    counts = reg.groupby("family").size().reset_index(name="definitions") if not reg.empty else pd.DataFrame()
    write_text(ctx.run_root / "registry/candidate_registry_report.md", "# Candidate Registry\n\nDefinitions are registered before scoring. A2 is used only as an overlay/filter, not standalone rankable.\n\n" + (counts.to_markdown(index=False) if not counts.empty else "No definitions."))

def replay_registry(ctx: RunContext, registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty or not ctx.args.include_a1_a4_sweep:
        return pd.DataFrame()
    symbols = symbol_universe(ctx.args.max_symbols if ctx.args.smoke else 0)
    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover", "open_interest", "funding_rate"]
    rows: list[dict[str, Any]] = []
    max_rows = 1200 if ctx.args.smoke else 180000
    # Keep full registry, but replay a deterministic practical subset per family for resource safety.
    replay_defs_per_family = 20 if ctx.args.smoke else min(160, max(20, int(ctx.args.top_per_family) * 2))
    defs = pd.concat([g.head(replay_defs_per_family) for _, g in registry.groupby("family", sort=False)], ignore_index=True)
    for symbol in symbols:
        if len(rows) >= max_rows:
            break
        df = load_symbol_5m(symbol, ctx.start - pd.Timedelta(days=140), ctx.end, columns=cols)
        if len(df) < 1000:
            continue
        d = daily_bars(df)
        if len(d) < 80:
            continue
        avg_turnover = float(d["turnover"].tail(60).mean()) if d["turnover"].notna().any() else 0.0
        if avg_turnover < 1_000_000 and not ctx.args.smoke:
            continue
        d["ret_20"] = d["close"] / d["close"].shift(20) - 1.0
        d["ret_40"] = d["close"] / d["close"].shift(40) - 1.0
        d["ret_60"] = d["close"] / d["close"].shift(60) - 1.0
        d["ret_90"] = d["close"] / d["close"].shift(90) - 1.0
        d["ret_120"] = d["close"] / d["close"].shift(120) - 1.0
        for lb in [20, 30, 40, 60]:
            d[f"high{lb}"] = d["close"].rolling(lb, min_periods=max(5, lb // 2)).max().shift(1)
        d["low10"] = d["low"].rolling(10, min_periods=5).min().shift(1)
        d["atr10"] = ((d["high"] - d["low"]) / d["close"]).rolling(10, min_periods=5).mean().shift(1)
        d["vol20"] = d["close"].pct_change().rolling(20, min_periods=10).std().shift(1)
        d["vol30"] = d["close"].pct_change().rolling(30, min_periods=15).std().shift(1)
        d["vol60"] = d["close"].pct_change().rolling(60, min_periods=30).std().shift(1)
        candidates = d[(d["decision_ts"] >= ctx.start) & (d["decision_ts"] <= ctx.end)].copy()
        if candidates.empty:
            continue
        for _, cfg in defs.iterrows():
            fam = str(cfg["family"])
            if fam == "A1":
                lb = int(cfg["lookback_high_days"])
                event_rows = candidates[(candidates["close"] > candidates[f"high{lb}"]) & (candidates["low"] > candidates["low10"] * float(cfg["smooth_low_buffer"])) & (candidates["vol20"] <= float(cfg["vol_contract_max"]))].head(8 if ctx.args.smoke else 20)
                side = "long"
                hold = int(cfg["max_hold_hours"])
                target_r = float(cfg["target_R"])
                stop_mult = float(cfg["atr_stop_mult"])
            elif fam == "A4":
                lb = int(cfg["lookback_days"])
                vw = int(cfg["vol_window_days"])
                event_rows = candidates[(candidates[f"ret_{lb}"] > float(cfg["ret_threshold"])) & (candidates[f"vol{vw}"] < float(cfg["vol_max"]))].head(8 if ctx.args.smoke else 20)
                side = "long"
                hold = int(cfg["max_hold_hours"])
                target_r = float(cfg["target_R"])
                stop_mult = float(cfg["atr_stop_mult"])
            else:
                # A3 overlay: delayed close-confirmed reclaim/retest-like condition.
                event_rows = candidates[(candidates["close"] > candidates["high20"] * 0.995) & (candidates["low"] > candidates["low10"] * 0.99)].head(6 if ctx.args.smoke else 12)
                side = "long"
                hold = int(cfg["max_hold_hours"])
                target_r = float(cfg["target_R"])
                stop_mult = 1.5
            for _, r in event_rows.iterrows():
                if len(rows) >= max_rows:
                    break
                decision_ts = pd.Timestamp(r["decision_ts"])
                entry_bars = df[df["timestamp"] > decision_ts]
                if entry_bars.empty:
                    continue
                entry = entry_bars.iloc[0]
                entry_ts = pd.Timestamp(entry["timestamp"])
                entry_price = float(entry["open"] if pd.notna(entry.get("open")) else entry["close"])
                atr = max(safe_float(r.get("atr10"), 0.04), 0.012)
                stop_pct = min(max(atr * stop_mult, 0.02), 0.16)
                stop_price = entry_price * (1.0 - stop_pct)
                target_price = entry_price + target_r * (entry_price - stop_price)
                rep = replay_path(symbol, side, entry_ts, entry_price, stop_price, target_price, hold)
                if rep is None:
                    continue
                rows.append({
                    **rep,
                    "event_id": stable_hash("v2", cfg["candidate_id"], symbol, decision_ts),
                    "candidate_id": cfg["candidate_id"],
                    "definition_id": cfg["definition_id"],
                    "family": fam,
                    "branch_id": "branch_l_liquid_regime_no_vendor_v2",
                    "symbol": symbol,
                    "decision_ts": decision_ts,
                    "entry_ts": entry_ts,
                    "side": side,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "entry_price_source": "local_5m_next_closed_bar_open",
                    "exit_rule": f"{target_r}R_or_{hold}h_time_stop",
                    "parent_regime": str(cfg.get("entry_rule", fam)),
                    "event_cluster_id": f"{symbol}_{decision_ts.strftime('%Y%m')}_{fam}",
                    "mark_available": mark_available_for(symbol, entry_ts, rep["exit_ts"]),
                    "required_data_tier": "Tier1",
                    "current_data_tier": "Tier1_local_5m_context_proxy_funding",
                })
    out = pd.DataFrame(rows)
    return normalize_event_ledger(out, "A1_A4_A3_overlay", "branch_l_liquid_regime_no_vendor_v2", "registry_replay_v2") if not out.empty else out

def recompute_control_summary(candidate_summary: pd.DataFrame, control_ledger: pd.DataFrame) -> pd.DataFrame:
    if candidate_summary.empty or control_ledger.empty:
        return pd.DataFrame()
    rows = []
    for (key, ctype), g in control_ledger.groupby(["candidate_key", "control_type"], sort=False):
        cand = candidate_summary[candidate_summary["candidate_key"].astype(str).eq(str(key))]
        if cand.empty:
            continue
        candidate_events = int(cand["events"].iloc[0])
        candidate_net = float(cand["net_R"].iloc[0])
        vals = pd.to_numeric(g["control_net_R"], errors="coerce")
        raw = float(vals.sum()) if vals.notna().any() else np.nan
        control_events = int(vals.notna().sum())
        norm = raw * candidate_events / control_events if control_events else np.nan
        rows.append({
            "candidate_key": key, "candidate_id": cand["candidate_id"].iloc[0], "family": cand["family"].iloc[0], "control_type": ctype,
            "candidate_event_count": candidate_events, "control_event_count": control_events, "candidate_net_R": candidate_net,
            "raw_control_net_R": raw, "normalized_control_net_R": norm, "control_uplift_R": candidate_net - norm if math.isfinite(norm) else np.nan,
            "beats_control": bool(math.isfinite(norm) and candidate_net > norm), "controls_normalized_to_candidate_count": True,
            "all_control_rows_have_source_ids": bool(g["control_source_row_id"].notna().all() and g["source_window_id"].notna().all()),
        })
    return pd.DataFrame(rows)

def build_overlap_embargo_audit(control_ledger: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    if control_ledger.empty or pool.empty:
        return pd.DataFrame()
    wins = pool[["candidate_key", "symbol", "entry_ts", "exit_ts"]].copy()
    wins["entry_ts"] = pd.to_datetime(wins["entry_ts"], utc=True, errors="coerce")
    wins["exit_ts"] = pd.to_datetime(wins["exit_ts"], utc=True, errors="coerce")
    rows = []
    for _, r in control_ledger.iterrows():
        key = str(r.get("candidate_key"))
        sym = str(r.get("control_symbol"))
        ce = pd.to_datetime(r.get("control_entry_ts"), utc=True, errors="coerce")
        cx = pd.to_datetime(r.get("control_exit_ts"), utc=True, errors="coerce")
        cwin = wins[(wins["candidate_key"].astype(str).eq(key)) & (wins["symbol"].astype(str).eq(sym))]
        violation = False
        embargo_hours = 0.0
        for _, w in cwin.iterrows():
            a, b = w["entry_ts"], w["exit_ts"]
            if pd.isna(a) or pd.isna(b) or pd.isna(ce) or pd.isna(cx):
                continue
            embargo = b - a
            embargo_hours = max(embargo_hours, embargo.total_seconds() / 3600.0)
            if ce <= b + embargo and cx >= a - embargo:
                violation = True
                break
        rows.append({"candidate_key": key, "control_type": r.get("control_type"), "control_event_id": r.get("control_event_id"), "control_symbol": sym, "source_window_id": r.get("source_window_id"), "embargo_hours": embargo_hours, "overlap_or_embargo_violation": violation})
    return pd.DataFrame(rows)

def stage_replay_controls(ctx: RunContext) -> None:
    registry = read_df(ctx.run_root / "registry/candidate_registry.csv")
    events = replay_registry(ctx, registry)
    pdir = ctx.run_root / "replay"
    pdir.mkdir(parents=True, exist_ok=True)
    if events.empty:
        write_text(pdir / "event_level_replay_report.md", "# Event-Level Replay\n\nNo event rows generated.\n")
        write_csv(ctx.run_root / "controls/real_control_summary.csv", [])
        return
    events.to_parquet(pdir / "event_level_replay.parquet", index=False)
    assert_pass(validate_event_trade_schema(events, require_all_fields=False))
    assert_pass(validate_pit_feature_timestamps(events, feature_ts_cols=[]))
    assert_pass(validate_funding_mark_flags(events))
    assert_pass(validate_no_projected_metric_promotion(events))
    pool = standardize_event_ledger(events, str(pdir / "event_level_replay.parquet"), family_hint="A1_A4_A3_overlay")
    enriched, cov = enrich_event_pool_with_match_features(pool, DATA_5M)
    candidate_keys = sorted(enriched["candidate_key"].dropna().astype(str).unique())[: max(1, ctx.args.top_per_family * 8)]
    cand_summary, ctrl_ledger, ctrl_summary = build_real_controls(enriched, candidate_keys=candidate_keys, nulls_per_event=ctx.args.nulls_per_event, seed=ctx.args.seed)
    audit = build_overlap_embargo_audit(ctrl_ledger, enriched)
    if not audit.empty:
        keep_ids = set(audit.loc[~audit["overlap_or_embargo_violation"], ["candidate_key", "control_event_id", "control_type"]].astype(str).agg("|".join, axis=1))
        key_series = ctrl_ledger[["candidate_key", "control_event_id", "control_type"]].astype(str).agg("|".join, axis=1)
        ctrl_ledger = ctrl_ledger[key_series.isin(keep_ids)].copy()
    if not ctrl_ledger.empty:
        ctrl_ledger = ctrl_ledger.sort_values(["candidate_key", "control_type", "selection_rank"], kind="mergesort")
        ctrl_ledger = ctrl_ledger.drop_duplicates(["matched_candidate_id", "control_event_id", "source_window_id"], keep="first").copy()
        ctrl_summary = recompute_control_summary(cand_summary, ctrl_ledger)
        assert_pass(validate_control_rows(ctrl_ledger))
    cdir = ctx.run_root / "controls"
    cdir.mkdir(parents=True, exist_ok=True)
    cand_summary.to_csv(cdir / "candidate_metric_summary.csv", index=False)
    ctrl_ledger.to_parquet(cdir / "control_event_ledger.parquet", index=False)
    write_csv(cdir / "real_control_summary.csv", ctrl_summary)
    write_csv(cdir / "match_feature_coverage.csv", cov)
    write_csv(cdir / "control_overlap_embargo_audit.csv", audit)
    write_text(pdir / "event_level_replay_report.md", f"# Event-Level Replay\n\nRows: {len(events)}. Candidate definitions replayed with local pre-holdout 5m bars.\n")

def stage_stress(ctx: RunContext) -> None:
    pdir = ctx.run_root / "stress"
    pdir.mkdir(parents=True, exist_ok=True)
    ep = ctx.run_root / "replay/event_level_replay.parquet"
    events = pd.read_parquet(ep) if ep.exists() else pd.DataFrame()
    summary = candidate_metrics(events) if not events.empty else pd.DataFrame()
    if not summary.empty:
        summary["label"] = np.where(
            (summary["net_R"] > 0) & (summary["PF"] > 1) & (summary["plus25bps_net_R"] > 0),
            np.where(summary["funding_proxy_used"] | summary["mark_proxy_used"], "tier1_screening_candidate_funding_mark_capped", "tier1_screening_candidate"),
            "current_translation_rejected_only_or_candidate_library",
        )
    write_csv(pdir / "stress_summary.csv", summary)
    rows = []
    for _, r in summary.iterrows() if not summary.empty else []:
        event_count = int(safe_float(r.get("event_count"), 0))
        rows.append({
            "candidate_id": r.get("candidate_id"), "family": r.get("family"), "event_count": event_count,
            "standalone_candidate": bool(event_count >= 50 and safe_float(r.get("net_R"), -1) > 0),
            "portfolio_sleeve_candidate": bool(event_count > 0),
            "rare_regime_sleeve_candidate": bool(event_count < 50),
            "feature_overlay_candidate": bool(r.get("family") == "A3_overlay"),
            "classification": "standalone_candidate" if event_count >= 50 and safe_float(r.get("net_R"), -1) > 0 else ("rare_regime_sleeve_candidate" if event_count < 50 else "candidate_library_only"),
            "label_cap_reason": r.get("label_cap_reason"),
        })
    write_csv(ctx.run_root / "validation/sleeve_classification.csv", rows)
    write_text(ctx.run_root / "stress/stress_report.md", f"# Tier-1 Stress\n\nCandidates summarized: {len(summary)}. All-taker costs, adverse funding caps, and +25/+50 bps diagnostics are applied in event-R space.\n")

def stage_c2_expansion(ctx: RunContext) -> None:
    pdir = ctx.run_root / "c2"
    pdir.mkdir(parents=True, exist_ok=True)
    if not ctx.args.include_c2_expansion:
        write_text(pdir / "c2_expansion_report.md", "# C2 Expansion\n\nSkipped by CLI.\n")
        return
    sources = []
    for pat in ["*catalyst*.csv", "*catalyst*.parquet", "*c2*ledger*.parquet", "*c2*anchor*.csv", "*c2*anchor*.parquet"]:
        for p in RESULTS_ROOT.rglob(pat):
            if p.is_file():
                sources.append(p)
    rows = []
    mechanisms = ["legal_regulatory_repricing", "etf_institutional_access", "protocol_utility_fee_revenue", "supply_unlock_float", "exchange_access", "leverage_access", "integration_distribution"]
    for p in sorted(set(sources)):
        df = read_df(p)
        if df.empty:
            continue
        mech_col = next((c for c in df.columns if "mechanism" in c.lower() or "family" in c.lower() or "event_type" in c.lower()), "")
        if mech_col:
            vals = df[mech_col].fillna("unknown").astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
        else:
            vals = pd.Series(["unknown"] * len(df))
        for mech in mechanisms + ["unknown"]:
            cnt = int(vals.str.contains(mech.split("_")[0], regex=False).sum()) if mech != "unknown" else int((vals == "unknown").sum())
            if cnt:
                rows.append({"source_path": str(p), "mechanism": mech, "rows": cnt, "source_rows": len(df), "date_precision_preserved": True, "bybit_tradability_required_for_primary": True})
    out = pd.DataFrame(rows)
    write_csv(pdir / "event_expansion_summary.csv", out)
    ready = out.groupby("mechanism", as_index=False).agg(source_rows=("rows", "sum")) if not out.empty else pd.DataFrame(columns=["mechanism", "source_rows"])
    if not ready.empty:
        ready["readiness_label"] = np.where(ready["source_rows"] >= 30, "mechanism_event_support_expandable", "sample_limited_seed_candidate")
    write_csv(pdir / "c2_mechanism_readiness.csv", ready)
    write_text(pdir / "c2_expansion_report.md", f"# C2 Mechanism Event Expansion\n\nSources scanned: {len(set(sources))}. C2 is not validated in this stage; sparse mechanisms are preserved.\n")

def read_live_capture_provenance(ctx: RunContext) -> dict[str, Any]:
    path = LIVE_CAPTURE_ZIP
    status = {"zip_path": str(path), "exists": path.exists(), "expected_outer_sha": EXPECTED_LIVE_CAPTURE_SHA, "local_outer_sha": "", "outer_sha_matches": False, "zip_opens_cleanly": False, "internal_manifest_found": False, "manifest_file_count": 0, "observed_file_count": 0, "stream_coverage": {}, "operator_attested": bool(ctx.args.operator_attested_live_capture), "capture_evidence_state": "capture_unusable_corrupt_or_incomplete", "calibration_allowed": False}
    if not path.exists():
        return status
    status["local_outer_sha"] = file_sha256(path)
    status["outer_sha_matches"] = status["local_outer_sha"] == EXPECTED_LIVE_CAPTURE_SHA
    try:
        with zipfile.ZipFile(path) as z:
            status["zip_opens_cleanly"] = z.testzip() is None
            names = z.namelist()
            status["observed_file_count"] = len(names)
            mani_name = next((n for n in names if n.endswith("data/manifests/file_manifest.csv")), "")
            if mani_name:
                status["internal_manifest_found"] = True
                mani = pd.read_csv(io.BytesIO(z.read(mani_name)))
                status["manifest_file_count"] = int(len(mani))
            streams = {}
            for n in names:
                for key in ["orderbook_1", "orderbook_50", "publicTrade", "tickers", "allLiquidation"]:
                    if key in n:
                        streams[key] = streams.get(key, 0) + 1
            status["stream_coverage"] = streams
    except Exception as exc:
        status["zip_error"] = f"{type(exc).__name__}: {exc}"
    coherent = bool(status["zip_opens_cleanly"] and status["internal_manifest_found"] and status["manifest_file_count"] > 0 and any(int(v) > 0 for v in status["stream_coverage"].values()))
    if status["outer_sha_matches"] and coherent:
        status["capture_evidence_state"] = "capture_content_verified"
        status["calibration_allowed"] = True
    elif coherent and status["operator_attested"]:
        status["capture_evidence_state"] = "operator_attested_capture_calibration_capped"
        status["calibration_allowed"] = True
    return status

def stage_branch_x(ctx: RunContext) -> None:
    pdir = ctx.run_root / "branch_x"
    pdir.mkdir(parents=True, exist_ok=True)
    if not ctx.args.include_branch_x_capture_calibration:
        write_text(pdir / "branch_x_capture_calibration_report.md", "# Branch X Capture Calibration\n\nSkipped by CLI.\n")
        return
    prov = read_live_capture_provenance(ctx)
    write_json(pdir / "capture_provenance_status.json", prov)
    streams = prov.get("stream_coverage", {}) or {}
    summary = [
        {"stream": k, "file_count": v, "calibration_use": "capped_distribution_only_not_validation"} for k, v in sorted(streams.items())
    ]
    write_csv(pdir / "capture_calibration_summary.csv", summary)
    listing_ok = bool(prov.get("calibration_allowed") and int(streams.get("publicTrade", 0)) > 0 and (int(streams.get("orderbook_1", 0)) > 0 or int(streams.get("orderbook_50", 0)) > 0))
    d4_ok = bool(listing_ok and int(streams.get("allLiquidation", 0)) > 0)
    write_csv(pdir / "listing_analog_calibration.csv", [{"candidate_group": "listing_vwap_loss_589_b1", "calibration_state": prov.get("capture_evidence_state"), "execution_only_micro_canary_possible": listing_ok, "validation_allowed": False}])
    write_csv(pdir / "d4_analog_calibration.csv", [{"candidate_group": "D4", "calibration_state": prov.get("capture_evidence_state"), "d4_like_liquidation_depth_context_available": d4_ok, "micro_canary_possible": d4_ok, "validation_allowed": False}])
    write_csv(pdir / "micro_canary_readiness_matrix.csv", [
        {"family": "listing_vwap_loss_589_b1", "micro_canary_status": "execution_only_possible_after_separate_human_approval" if listing_ok else "request_72h_branch_x_capture", "not_alpha_validation": True},
        {"family": "D4", "micro_canary_status": "blocked_until_actual_d4_like_capture" if not d4_ok else "execution_only_possible_after_separate_human_approval", "not_alpha_validation": True},
    ])
    write_text(pdir / "branch_x_capture_calibration_report.md", f"# Branch X Capture Calibration\n\nCapture state: `{prov.get('capture_evidence_state')}`. Calibration allowed: `{prov.get('calibration_allowed')}`. This is not validation and does not alter historical alpha parameters.\n")

def stage_decision_matrix(ctx: RunContext) -> None:
    stress = read_df(ctx.run_root / "stress/stress_summary.csv")
    bx = read_df(ctx.run_root / "branch_x/micro_canary_readiness_matrix.csv")
    rows = []
    for fam in ["A1", "A4", "A3_overlay", "C2", "D4", "listing_vwap_loss", "funding_window", "generic_shock"]:
        if fam in {"A1", "A4", "A3_overlay"}:
            sub = stress[stress.get("family", pd.Series(dtype=str)).astype(str).eq(fam)] if not stress.empty else pd.DataFrame()
            outcome = "progress_with_current_data" if not sub.empty else "preserve_hypothesis_generate_new_variant"
        elif fam == "C2":
            outcome = "candidate_library_only"
        elif fam in {"D4", "listing_vwap_loss"}:
            outcome = "needs_capture_substitute"
        elif fam == "generic_shock":
            outcome = "discard_current_translation_no_vendor_path"
        else:
            outcome = "preserve_hypothesis_generate_new_variant"
        rows.append({"family": fam, "no_vendor_outcome": outcome, "waiting_for_vendor_data": False, "family_rejected": False})
    write_csv(ctx.run_root / "decisions/no_vendor_decision_matrix.csv", rows)
    if any(r["no_vendor_outcome"] not in NO_VENDOR_OUTCOMES for r in rows):
        raise RuntimeError("invalid no-vendor outcome")

def stage_library(ctx: RunContext) -> None:
    stress = read_df(ctx.run_root / "stress/stress_summary.csv")
    rows = []
    if not stress.empty:
        for _, r in stress.iterrows():
            event_count = int(safe_float(r.get("event_count"), 0))
            rows.append({
                "candidate_id": r.get("candidate_id"), "family": r.get("family"), "evidence_state": "event_ledger_available" if event_count else "support_only",
                "required_data_tier": "Tier1", "current_data_tier": "Tier1_local_5m_context_proxy_funding", "data_tier_cap_reason": r.get("label_cap_reason"),
                "standalone_status": "possible" if event_count >= 50 and safe_float(r.get("net_R"), -1) > 0 else "not_yet",
                "portfolio_sleeve_status": "candidate_sleeve_possible" if event_count else "not_yet",
                "rare_regime_status": "rare_or_sparse" if event_count < 50 else "not_sparse",
                "feature_overlay_status": "possible_overlay" if r.get("family") == "A3_overlay" else "not_overlay",
                "event_count": event_count, "active_months": int(safe_float(r.get("active_months"), 0)), "active_symbols": int(safe_float(r.get("active_symbols"), 0)),
                "dominant_month_share": r.get("dominant_month_share"), "dominant_symbol_share": r.get("dominant_symbol_share"),
                "return_overlap_proxy": "not_computed_no_vendor_v2", "candidate_correlation_proxy": "not_computed_no_vendor_v2",
                "reason_preserved_or_discarded": "preserved_under_no_vendor_v2_unless_current_translation_discarded",
            })
    rows.extend([
        {"candidate_id": "D4__b4c9487fe82c", "family": "D4", "evidence_state": "capture_substitute_needed", "required_data_tier": "Tier2_or_capture", "current_data_tier": "operator_attested_capture_capped", "data_tier_cap_reason": "D4 needs D4-like capture with liquidation/depth context", "standalone_status": "not_ranked", "portfolio_sleeve_status": "not_ranked", "rare_regime_status": "event_driven", "feature_overlay_status": "none", "event_count": 4475, "active_months": "", "active_symbols": "", "dominant_month_share": "", "dominant_symbol_share": "", "return_overlap_proxy": "", "candidate_correlation_proxy": "", "reason_preserved_or_discarded": "preserved_for_capture_substitute_not_vendor"},
        {"candidate_id": "listing_vwap_loss_589_b1", "family": "Branch_X_listing", "evidence_state": "capture_calibration_candidate", "required_data_tier": "Tier2_or_capture", "current_data_tier": "operator_attested_capture_capped", "data_tier_cap_reason": "execution telemetry only", "standalone_status": "not_ranked", "portfolio_sleeve_status": "candidate_library_only", "rare_regime_status": "listing_event", "feature_overlay_status": "none", "event_count": "", "active_months": "", "active_symbols": "", "dominant_month_share": "", "dominant_symbol_share": "", "return_overlap_proxy": "", "candidate_correlation_proxy": "", "reason_preserved_or_discarded": "preserved_listing_analog_capture_path"},
    ])
    write_csv(ctx.run_root / "library/refreshed_candidate_library.csv", rows)
    contracts_dir = ctx.run_root / "next_contracts/contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    next_rows = [
        {"contract_id": "a1_a4_v2_review", "family": "A1_A4", "next_action": "review_v2_registry_real_controls_and_stress", "data_tier": "Tier1_capped"},
        {"contract_id": "branch_x_capture_72h", "family": "Branch_X", "next_action": "export_or_continue_72h_capture_if_micro_canary_not_ready", "data_tier": "capture_substitute"},
        {"contract_id": "c2_event_expansion", "family": "C2", "next_action": "mechanism_specific_event_expansion_no_validation", "data_tier": "seed_limited"},
    ]
    write_csv(ctx.run_root / "next_contracts/next_action_contract_summary.csv", next_rows)
    for r in next_rows:
        write_json(contracts_dir / f"{r['contract_id']}.json", r)

def top_candidate_samples(ctx: RunContext) -> None:
    pdir = ctx.run_root / "audit_samples"
    pdir.mkdir(parents=True, exist_ok=True)
    ep = ctx.run_root / "replay/event_level_replay.parquet"
    if not ep.exists():
        write_csv(pdir / "top_candidate_event_samples.csv", [])
        write_text(pdir / "top_candidate_calculation_trace.md", "# Calculation Trace\n\nNo event rows available.\n")
        return
    df = pd.read_parquet(ep)
    summaries = candidate_metrics(df).sort_values("net_R", ascending=False).head(20)
    samples = []
    for cid in summaries["candidate_id"].astype(str):
        g = df[df["candidate_id"].astype(str).eq(cid)].copy()
        picks = [g.nlargest(min(10, len(g)), "net_R"), g.nsmallest(min(10, len(g)), "net_R"), g.sample(min(10, len(g)), random_state=ctx.args.seed)]
        if "same_bar_ambiguity_flag" in g.columns:
            picks.append(g[g["same_bar_ambiguity_flag"].fillna(False).astype(bool)].head(10))
        if "mark_proxy_used" in g.columns:
            picks.append(g[g["mark_proxy_used"].fillna(False).astype(bool) | g["funding_proxy_used"].fillna(False).astype(bool)].head(10))
        samples.append(pd.concat(picks, ignore_index=True).drop_duplicates("event_id"))
    out = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    keep = [c for c in ["candidate_id", "family", "event_id", "symbol", "decision_ts", "entry_ts", "exit_ts", "entry_price", "stop_price", "target_price", "exit_price", "gross_R", "fees_R", "slippage_R", "funding_R", "net_R", "risk_bps_used", "exit_reason", "mark_proxy_used", "funding_proxy_used", "source_path"] if c in out.columns]
    write_csv(pdir / "top_candidate_event_samples.csv", out[keep] if keep else out)
    write_text(pdir / "top_candidate_calculation_trace.md", "# Calculation Trace\n\nR = directional price PnL divided by absolute entry-stop risk, minus all-taker fee_R and slippage_R, plus funding_R. Same-bar stop/target ambiguity is adverse: stop wins. Funding-cross rows without exact settlement data are capped and stressed, not treated as exact. Raw bars are in `/opt/parquet/5m/{symbol}.parquet`; mark context is in `/opt/parquet/bybit_context_5m/{symbol}.parquet`.\n")

def stage_report(ctx: RunContext) -> None:
    top_candidate_samples(ctx)
    audit_text = (ctx.run_root / "audit/previous_no_vendor_run_audit.md").read_text() if (ctx.run_root / "audit/previous_no_vendor_run_audit.md").exists() else ""
    fm_text = (ctx.run_root / "funding_mark/root_cause_report.md").read_text() if (ctx.run_root / "funding_mark/root_cause_report.md").exists() else ""
    stress = read_df(ctx.run_root / "stress/stress_summary.csv")
    bx = read_df(ctx.run_root / "branch_x/micro_canary_readiness_matrix.csv")
    c2_ready = read_df(ctx.run_root / "c2/c2_mechanism_readiness.csv")
    telegram_worked = ctx.notifier.remote_available
    has_positive = bool(not stress.empty and ((pd.to_numeric(stress.get("net_R"), errors="coerce") > 0) & (pd.to_numeric(stress.get("PF"), errors="coerce") > 1)).any())
    micro_possible = bool(not bx.empty and bx.get("micro_canary_status", pd.Series(dtype=str)).astype(str).str.contains("execution_only_possible", na=False).any())
    if micro_possible:
        decision = "branch_x_micro_canary_possible_execution_only"
    elif has_positive:
        decision = "run_train_only_candidate_validation_package_next"
    elif not c2_ready.empty:
        decision = "run_b1_c2_mechanism_validation_next"
    else:
        decision = "generate_new_hypotheses_next"
    if decision not in ALLOWED_NEXT_DECISIONS:
        decision = "blocked_by_protocol_issue"
    prev_verdict = "previous_a1_a4_sweep_was_single_translation_not_full_sweep" if "single_translation" in audit_text else "previous_audit_completed"
    fm_verdict = "join_bug_suspected" if "join_bug_suspected" in fm_text else "proxy_caps_due_to_missing_or_unverified_exact_sources"
    decision_json = {
        "run_root": str(ctx.run_root), "final_holdout_untouched": True, "telegram_worked": telegram_worked,
        "previous_no_vendor_audit_verdict": prev_verdict,
        "funding_mark_root_cause_verdict": fm_verdict,
        "a1_a4_redesign_verdict": "registry_based_replay_completed" if (ctx.run_root / "replay/event_level_replay.parquet").exists() else "not_available",
        "a3_overlay_verdict": "overlay_registry_replay_completed" if not stress.empty and (stress["family"].astype(str).eq("A3_overlay")).any() else "not_available",
        "c2_event_expansion_verdict": "completed_seed_limited" if (ctx.run_root / "c2/event_expansion_summary.csv").exists() else "not_available",
        "branch_x_calibration_verdict": "capped_calibration_completed" if (ctx.run_root / "branch_x/capture_calibration_summary.csv").exists() else "not_available",
        "micro_canary_execution_only_possible": micro_possible,
        "no_vendor_decision_verdict": "completed_no_waiting_for_vendor",
        "next_operator_decision": decision,
        "compact_bundle_path": str(ctx.run_root / "compact_review_bundle"),
    }
    write_json(ctx.run_root / "decision_summary.json", decision_json)
    top = stress.sort_values("net_R", ascending=False).head(20) if not stress.empty else pd.DataFrame()
    report = [
        "# QLMG No-Vendor Progress V2 Report", "", f"Run root: `{ctx.run_root}`", "Final holdout untouched: yes", f"Telegram worked: {'yes' if telegram_worked else 'no'}", "",
        "## Previous No-Vendor Audit", prev_verdict, "", "## Funding/Mark Root Cause", fm_verdict, "", "## A1/A4/A3 Results",
        top[["candidate_id", "family", "event_count", "net_R", "PF", "plus25bps_net_R", "mark_proxy_used", "funding_proxy_used", "label"]].to_markdown(index=False) if not top.empty else "No stress summary rows.",
        "", "## C2 Event Expansion", c2_ready.to_markdown(index=False) if not c2_ready.empty else "No C2 expansion rows.", "", "## Branch X Calibration", f"Micro-canary execution-only possible: `{micro_possible}`. This is not validation.", "", "## Next Operator Decision", f"`{decision}`",
    ]
    write_text(ctx.run_root / "QLMG_NO_VENDOR_PROGRESS_V2_REPORT.md", "\n".join(report))
    scan = scan_output_tree_for_protected(ctx.run_root)
    write_json(ctx.run_root / "seal/generated_output_protected_scan.json", result_to_jsonable(scan))
    if not scan.passed:
        raise RuntimeError("generated output protected scan failed: " + ";".join(scan.violations))

def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_NO_VENDOR_PROGRESS_V2_REPORT.md", "decision_summary.json", "preflight/preflight_report.md", "preflight/input_artifact_manifest.csv", "audit/previous_no_vendor_run_audit.md", "audit/a1_a4_candidate_registry_existence.csv", "funding_mark/root_cause_report.md", "funding_mark/coverage_by_family_symbol_month.csv", "registry/candidate_registry_report.md", "stress/stress_summary.csv", "validation/sleeve_classification.csv", "controls/real_control_summary.csv", "controls/control_overlap_embargo_audit.csv", "c2/event_expansion_summary.csv", "c2/c2_mechanism_readiness.csv", "branch_x/capture_calibration_summary.csv", "branch_x/listing_analog_calibration.csv", "branch_x/d4_analog_calibration.csv", "branch_x/micro_canary_readiness_matrix.csv", "decisions/no_vendor_decision_matrix.csv", "library/refreshed_candidate_library.csv", "next_contracts/next_action_contract_summary.csv", "audit_samples/top_candidate_event_samples.csv", "audit_samples/top_candidate_calculation_trace.md",
    ]
    idx = []
    for rel in include:
        src = ctx.run_root / rel
        if src.exists() and src.is_file() and src.stat().st_size < 5_000_000:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            idx.append({"source": rel, "bundle_file": dst.name, "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_index.csv", idx)

STAGE_FUNCS = {
    "preflight-and-manual-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "previous-no-vendor-run-audit": stage_previous_audit,
    "mark-funding-enrichment-root-cause": stage_mark_funding,
    "true-a1-a4-a3-overlay-candidate-registry": stage_registry,
    "event-level-replay-and-real-controls": stage_replay_controls,
    "tier1-stress-and-sleeve-classification": stage_stress,
    "c2-mechanism-event-expansion": stage_c2_expansion,
    "branch-x-capture-calibration": stage_branch_x,
    "no-vendor-decision-matrix": stage_decision_matrix,
    "candidate-library-refresh": stage_library,
    "decision-report": stage_report,
    "compact-review-bundle": stage_bundle,
}

def run_stage(ctx: RunContext, stage: str) -> None:
    if stage == "all":
        for s in stage_list("all"):
            run_stage(ctx, s)
        return
    if ctx.args.resume and done_path(ctx, stage).exists():
        return
    ctx.notifier.send("QLMG no-vendor v2 stage start", stage)
    fn = STAGE_FUNCS[stage]
    if not ctx.args.dry_run:
        fn(ctx)
    mark_done(ctx, stage)
    ctx.notifier.send("QLMG no-vendor v2 stage done", stage)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    start, end = clamp_window(args)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "argv": argv if argv is not None else sys.argv[1:], "start": str(start), "end": str(end), "created_utc": utc_now()})
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "complete", "ts_utc": utc_now()})
        notifier.send("QLMG no-vendor v2 run complete", str(run_root))
        return 0
    except Exception as exc:
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "failed", "error": f"{type(exc).__name__}: {exc}", "ts_utc": utc_now()})
        notifier.send("QLMG no-vendor v2 run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise

if __name__ == "__main__":
    raise SystemExit(main())
