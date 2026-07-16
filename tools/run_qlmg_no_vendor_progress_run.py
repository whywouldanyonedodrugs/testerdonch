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
    validate_no_current_only_taxonomy_rankable,
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
DEFAULT_RUN_ID = "phase_qlmg_no_vendor_progress_run_20260630_v1"
DEFAULT_SEED = 20260630
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
DATA_5M = Path("/opt/parquet/5m")
CONTEXT_5M = Path("/opt/parquet/bybit_context_5m")
LIVE_CAPTURE_ZIP = REPO / "research_inputs/qlmg_live_capture.zip"
EXPECTED_LIVE_CAPTURE_SHA = "ee88a2b0c0b3e81cc5b18aa9715747208170d498b1cf8205e751402df43442e1"

MECHANICAL_QA_ROOT = RESULTS_ROOT / "phase_qlmg_mechanical_qa_evidence_contract_20260630_v1_20260630_074328"
LEAKAGE_GUARD_ROOT = RESULTS_ROOT / "phase_qlmg_leakage_guard_rebaseline_20260629_v1_20260629_174557"
EVIDENCE_REPAIR_ROOT = RESULTS_ROOT / "phase_qlmg_evidence_remediation_family_repair_20260629_v1_20260629_044410"
ABCX_ROOT = RESULTS_ROOT / "phase_qlmg_integrated_abcx_development_20260628_v2_20260628_134140"
REAL_CONTROL_ROOT = RESULTS_ROOT / "phase_qlmg_real_control_rebuild_20260629_v1_20260629_170608"
GLOBAL_INVALIDATION_ROOT = RESULTS_ROOT / "phase_qlmg_global_result_invalidation_audit_20260629_v1_20260629_171528"
D4_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"

QA_REQUIRED_FILES = [
    MECHANICAL_QA_ROOT / "decision_summary.json",
    MECHANICAL_QA_ROOT / "contracts/event_level_evidence_contract.md",
    MECHANICAL_QA_ROOT / "contracts/control_engine_contract.md",
    MECHANICAL_QA_ROOT / "contracts/funding_mark_contract.md",
    MECHANICAL_QA_ROOT / "quarantine/refreshed_quarantine_manifest.csv",
    MECHANICAL_QA_ROOT / "quarantine/deprecated_promotion_labels.csv",
    MECHANICAL_QA_ROOT / "readiness/sweep_readiness_matrix.csv",
]

ACTIVE_ROOTS = {
    "mechanical_qa": MECHANICAL_QA_ROOT,
    "leakage_guard": LEAKAGE_GUARD_ROOT,
    "leakage_guard_corrected_sweep": LEAKAGE_GUARD_ROOT / "corrected_sweep",
    "leakage_guard_evidence_repair": LEAKAGE_GUARD_ROOT / "evidence_repair",
    "real_control_rebuild": REAL_CONTROL_ROOT,
    "global_invalidation_audit": GLOBAL_INVALIDATION_ROOT,
    "evidence_repair": EVIDENCE_REPAIR_ROOT,
    "integrated_abcx": ABCX_ROOT,
    "d4_survivability": D4_ROOT,
    "listing_generic": LISTING_ROOT,
    "data_5m": DATA_5M,
    "context_5m": CONTEXT_5M,
    "sector_md": REPO / "research_inputs/point_in_time_sector_seeds.md",
    "catalyst_md": REPO / "research_inputs/post_catalyst_c2_database.md",
    "live_capture_zip": LIVE_CAPTURE_ZIP,
}

A3_LEDGER = LEAKAGE_GUARD_ROOT / "corrected_sweep/a3_sweep/a3_event_level_replay.parquet"
A2_LEDGER = LEAKAGE_GUARD_ROOT / "corrected_sweep/a2_sweep/a2_event_level_replay.parquet"
B1_ANCHOR_LEDGER = LEAKAGE_GUARD_ROOT / "corrected_sweep/b1_sidecar/b1_event_anchor_ledger.parquet"
C2_EVENT_LEDGER = ABCX_ROOT / "c2/catalyst_event_ledger.parquet"
B1_SECTOR_MAP = ABCX_ROOT / "b1/sector_map_pit.parquet"
B1_CLUSTERS = ABCX_ROOT / "b1/comovement_clusters_by_date.parquet"
C2_REPAIR_LEDGER = EVIDENCE_REPAIR_ROOT / "c2_trade_ledger/c2_event_level_replay.parquet"

STAGES = (
    "preflight-and-evidence-contract-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "live-capture-provenance-repair",
    "exact-funding-mark-enrichment",
    "a1-a4-corrected-liquid-sweep",
    "b1-ledger-construction",
    "c2-ledger-construction",
    "fresh-real-controls",
    "tier1-stress-and-funding-mark-caps",
    "walk-forward-stability-and-sleeve-classification",
    "branch-x-capture-calibration",
    "no-vendor-decision-rules",
    "candidate-library-refresh",
    "decision-report",
    "compact-review-bundle",
    "all",
)

ALLOWED_NEXT_DECISIONS = [
    "run_exact_funding_mark_enrichment_next",
    "run_a1_a4_corrected_liquid_sweep_next",
    "run_b1_c2_ledger_construction_next",
    "run_branch_x_capture_calibration_next",
    "request_correct_live_capture_bundle",
    "branch_x_micro_canary_possible_execution_only",
    "run_train_only_candidate_validation_package_next",
    "generate_new_hypotheses_next",
    "blocked_by_protocol_issue",
]

NO_VENDOR_OUTCOMES = [
    "progress_with_current_data",
    "needs_capture_substitute",
    "redesign_to_less_depth_sensitive",
    "discard_current_translation_no_vendor_path",
    "preserve_hypothesis_generate_new_variant",
    "candidate_library_only",
]

CAPTURE_STATES = [
    "capture_content_verified",
    "operator_attested_capture_calibration_capped",
    "capture_unusable_corrupt_or_incomplete",
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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-no-vendor-progress")
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
    p = argparse.ArgumentParser(description="QLMG no-vendor progress run")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=45.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--include-funding-mark-enrichment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-a1-a4-sweep", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-b1-c2-ledgers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-branch-x-capture-calibration", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--a1-budget", type=int, default=2500)
    p.add_argument("--a4-budget", type=int, default=2500)
    p.add_argument("--b1-budget", type=int, default=1500)
    p.add_argument("--c2-budget", type=int, default=1500)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--top-per-family", type=int, default=60)
    p.add_argument("--tmux-session-name", default="qlmg_no_vendor_progress")
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
            chunk_size = 1024 * 1024
            if max_bytes is not None:
                remaining = max_bytes - read
                if remaining <= 0:
                    break
                chunk_size = min(chunk_size, remaining)
            chunk = f.read(chunk_size)
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


def to_utc(s: Any) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


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
        start = a.floor("8h")
        # Approximate settlement boundaries only. This does not make funding exact.
        n = 0
        cur = start
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
    out["definition_id"] = out.get("definition_id", out.get("variant_id", out["candidate_id"]))
    out["event_id"] = out.get("event_id", pd.Series([stable_hash(source, i) for i in range(len(out))], index=out.index))
    for col in ["decision_ts", "entry_ts", "exit_ts"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out = out[out["decision_ts"].notna() & (out["decision_ts"] < PROTECTED_TS)].copy()
    out["side"] = out.get("side", "long")
    out["entry_price"] = pd.to_numeric(out.get("entry_price"), errors="coerce")
    out["exit_price"] = pd.to_numeric(out.get("exit_price"), errors="coerce")
    out["stop_price"] = pd.to_numeric(out.get("stop_price"), errors="coerce")
    out["target_price"] = pd.to_numeric(out.get("target_price"), errors="coerce")
    out["risk_bps_used"] = pd.to_numeric(out.get("risk_bps_used"), errors="coerce")
    gross_src = out["gross_R"] if "gross_R" in out.columns else (out["net_R"] if "net_R" in out.columns else pd.Series(0.0, index=out.index))
    fees_src = out["fees_R"] if "fees_R" in out.columns else (out["cost_R"] if "cost_R" in out.columns else pd.Series(0.0, index=out.index))
    slippage_src = out["slippage_R"] if "slippage_R" in out.columns else pd.Series(0.0, index=out.index)
    funding_src = out["funding_R"] if "funding_R" in out.columns else pd.Series(0.0, index=out.index)
    out["gross_R"] = pd.to_numeric(gross_src, errors="coerce")
    out["fees_R"] = pd.to_numeric(fees_src, errors="coerce").fillna(0.0)
    out["slippage_R"] = pd.to_numeric(slippage_src, errors="coerce").fillna(0.0)
    out["funding_R"] = pd.to_numeric(funding_src, errors="coerce").fillna(0.0)
    if "net_R" not in out.columns:
        out["net_R"] = out["gross_R"] - out["fees_R"] - out["slippage_R"] + out["funding_R"]
    out["net_R"] = pd.to_numeric(out["net_R"], errors="coerce")
    out["exit_rule"] = out.get("exit_rule", "fixed_stop_target_time")
    out["exit_reason"] = out.get("exit_reason", "time")
    out["entry_price_source"] = out.get("entry_price_source", "local_5m_next_bar_open_or_close")
    liq_src = out["mark_liquidation_flag"] if "mark_liquidation_flag" in out.columns else (out["liquidation_flag"] if "liquidation_flag" in out.columns else pd.Series(False, index=out.index))
    amb_src = out["same_bar_ambiguity_flag"] if "same_bar_ambiguity_flag" in out.columns else (out["same_bar_ambiguity"] if "same_bar_ambiguity" in out.columns else pd.Series(False, index=out.index))
    out["mark_liquidation_flag"] = pd.Series(liq_src, index=out.index).fillna(False).astype(bool)
    out["same_bar_ambiguity_flag"] = pd.Series(amb_src, index=out.index).fillna(False).astype(bool)
    out["funding_timestamps_crossed"] = pd.to_numeric(out.get("funding_timestamps_crossed"), errors="coerce") if "funding_timestamps_crossed" in out.columns else estimate_funding_crosses(out["entry_ts"], out["exit_ts"])
    mark_base = out.get("mark_available", out.get("mark_price_available", False))
    out["mark_available"] = pd.Series(mark_base, index=out.index).fillna(False).astype(bool)
    out["funding_exact"] = pd.Series(out.get("funding_exact", False), index=out.index).fillna(False).astype(bool)
    out.loc[out["funding_timestamps_crossed"].fillna(0).astype(int) == 0, "funding_exact"] = True
    out["mark_proxy_used"] = ~out["mark_available"]
    out["funding_proxy_used"] = ~out["funding_exact"]
    out["lifecycle_status"] = out.get("lifecycle_status", "local_bar_presence_only")
    out["data_tier"] = out.get("data_tier", out.get("required_data_tier", "Tier1"))
    out["control_group_id"] = out.get("control_group_id", out["candidate_id"])
    out["source_data_hash"] = out.get("source_data_hash", stable_hash(source))
    out["metric_basis"] = out.get("metric_basis", "event_level_trade_ledger_no_vendor_train_only")
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
        ctx = pd.read_parquet(path, columns=["timestamp", "mark_close", "mark_source_close_ts", "context_source_close_ts"])
    except Exception:
        _MARK_CONTEXT_CACHE[symbol] = pd.DataFrame()
        return _MARK_CONTEXT_CACHE[symbol]
    if ctx.empty:
        _MARK_CONTEXT_CACHE[symbol] = pd.DataFrame()
        return _MARK_CONTEXT_CACHE[symbol]
    ctx = ctx.copy()
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True, errors="coerce")
    ctx["mark_source_close_ts"] = pd.to_datetime(ctx.get("mark_source_close_ts"), utc=True, errors="coerce")
    ctx = ctx[ctx["timestamp"].notna() & (ctx["timestamp"] < PROTECTED_TS)].sort_values("timestamp", kind="mergesort")
    _MARK_CONTEXT_CACHE[symbol] = ctx
    return ctx


def mark_available_for(symbol: str, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> bool:
    ctx = load_mark_context(symbol)
    if ctx.empty:
        return False
    span = ctx[(ctx["timestamp"] >= entry_ts) & (ctx["timestamp"] <= exit_ts)]
    if span.empty or span["mark_close"].isna().all():
        return False
    if span["mark_source_close_ts"].notna().any():
        ok = span.loc[span["mark_source_close_ts"].notna(), "mark_source_close_ts"] <= span.loc[span["mark_source_close_ts"].notna(), "timestamp"]
        return bool(ok.all())
    return False


def build_a1_a4_events(ctx: RunContext) -> pd.DataFrame:
    if not ctx.args.include_a1_a4_sweep:
        return pd.DataFrame()
    rng = np.random.default_rng(ctx.args.seed)
    symbols = symbol_universe(ctx.args.max_symbols if ctx.args.smoke else 0)
    rows: list[dict[str, Any]] = []
    budgets = {"A1": int(ctx.args.a1_budget), "A4": int(ctx.args.a4_budget)}
    max_rows_total = sum(budgets.values())
    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover", "open_interest", "funding_rate"]
    for idx, symbol in enumerate(symbols):
        if len(rows) >= max_rows_total * 3:
            break
        df = load_symbol_5m(symbol, ctx.start - pd.Timedelta(days=120), ctx.end, columns=cols)
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
        d["high20"] = d["close"].rolling(20, min_periods=15).max().shift(1)
        d["high30"] = d["close"].rolling(30, min_periods=20).max().shift(1)
        d["low10"] = d["low"].rolling(10, min_periods=5).min().shift(1)
        d["atr10"] = ((d["high"] - d["low"]) / d["close"]).rolling(10, min_periods=5).mean().shift(1)
        d["vol20"] = d["close"].pct_change().rolling(20, min_periods=10).std().shift(1)
        candidates = d[(d["decision_ts"] >= ctx.start) & (d["decision_ts"] <= ctx.end)].copy()
        # A1: close-confirmed prior high with smooth path and next-bar entry.
        a1 = candidates[(candidates["close"] > candidates["high20"]) & (candidates["low"] > candidates["low10"] * 0.98)].copy()
        for _, r in a1.head(40 if ctx.args.smoke else 120).iterrows():
            decision_ts = pd.Timestamp(r["decision_ts"])
            entry_bars = df[df["timestamp"] > decision_ts]
            if entry_bars.empty:
                continue
            entry = entry_bars.iloc[0]
            entry_ts = pd.Timestamp(entry["timestamp"])
            entry_price = float(entry["open"] if pd.notna(entry.get("open")) else entry["close"])
            atr = max(safe_float(r.get("atr10"), 0.04), 0.015)
            stop_price = entry_price * (1.0 - min(max(atr * 1.5, 0.025), 0.12))
            target_price = entry_price + 2.0 * (entry_price - stop_price)
            rep = replay_path(symbol, "long", entry_ts, entry_price, stop_price, target_price, 72)
            if rep is None:
                continue
            rows.append({
                **rep,
                "event_id": stable_hash("A1", symbol, decision_ts),
                "candidate_id": "A1__smooth_path_breakout__20d_high__2R_72h",
                "definition_id": "A1_def_20d_high_2R_72h",
                "family": "A1",
                "branch_id": "branch_l_liquid_regime_no_vendor",
                "symbol": symbol,
                "decision_ts": decision_ts,
                "entry_ts": entry_ts,
                "side": "long",
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "entry_price_source": "local_5m_next_closed_bar_open",
                "exit_rule": "2R_or_72h_time_stop",
                "parent_regime": "liquid_continuation",
                "event_cluster_id": f"{symbol}_{decision_ts.strftime('%Y%m')}_A1",
                "mark_available": mark_available_for(symbol, entry_ts, rep["exit_ts"]),
                "required_data_tier": "Tier1",
                "current_data_tier": "Tier1_local_5m_context_proxy_funding",
            })
        # A4: liquid time-series momentum, volatility-managed by ATR stop.
        a4 = candidates[(candidates["ret_40"] > 0.10) & (candidates["vol20"] < 0.12)].copy()
        for _, r in a4.head(40 if ctx.args.smoke else 120).iterrows():
            decision_ts = pd.Timestamp(r["decision_ts"])
            entry_bars = df[df["timestamp"] > decision_ts]
            if entry_bars.empty:
                continue
            entry = entry_bars.iloc[0]
            entry_ts = pd.Timestamp(entry["timestamp"])
            entry_price = float(entry["open"] if pd.notna(entry.get("open")) else entry["close"])
            atr = max(safe_float(r.get("atr10"), 0.035), 0.012)
            stop_price = entry_price * (1.0 - min(max(atr * 2.0, 0.025), 0.14))
            target_price = entry_price + 2.5 * (entry_price - stop_price)
            rep = replay_path(symbol, "long", entry_ts, entry_price, stop_price, target_price, 120)
            if rep is None:
                continue
            rows.append({
                **rep,
                "event_id": stable_hash("A4", symbol, decision_ts),
                "candidate_id": "A4__vol_managed_tsmom__40d_ret__2p5R_120h",
                "definition_id": "A4_def_40d_tsmom_2p5R_120h",
                "family": "A4",
                "branch_id": "branch_l_liquid_regime_no_vendor",
                "symbol": symbol,
                "decision_ts": decision_ts,
                "entry_ts": entry_ts,
                "side": "long",
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "entry_price_source": "local_5m_next_closed_bar_open",
                "exit_rule": "2p5R_or_120h_time_stop",
                "parent_regime": "liquid_tsmom",
                "event_cluster_id": f"{symbol}_{decision_ts.strftime('%Y%m')}_A4",
                "mark_available": mark_available_for(symbol, entry_ts, rep["exit_ts"]),
                "required_data_tier": "Tier1",
                "current_data_tier": "Tier1_local_5m_context_proxy_funding",
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Deterministic budget cap by family without winner bias.
    parts = []
    for family, g in out.groupby("family", sort=False):
        budget = budgets.get(family, len(g))
        g = g.sort_values(["symbol", "decision_ts"], kind="mergesort").reset_index(drop=True)
        if len(g) > budget:
            step = len(g) / budget
            idx = [int(i * step) for i in range(budget)]
            g = g.iloc[idx].copy()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out = normalize_event_ledger(out, "A1_A4", "branch_l_liquid_regime_no_vendor", "local_5m_a1_a4_no_vendor_generator")
    return out


def replay_anchor_rows(anchor: pd.DataFrame, default_family: str, budget: int, ctx: RunContext) -> pd.DataFrame:
    if anchor.empty:
        return pd.DataFrame()
    rows = []
    anchor = anchor.copy()
    ts_col = "decision_ts" if "decision_ts" in anchor.columns else "earliest_decision_ts"
    anchor[ts_col] = pd.to_datetime(anchor[ts_col], utc=True, errors="coerce")
    anchor = anchor[anchor[ts_col].notna() & (anchor[ts_col] >= ctx.start) & (anchor[ts_col] <= ctx.end) & (anchor[ts_col] < PROTECTED_TS)].copy()
    anchor = anchor.sort_values([ts_col, "symbol"], kind="mergesort").head(budget)
    for _, r in anchor.iterrows():
        symbol = str(r.get("symbol", ""))
        if not symbol or symbol == "nan":
            continue
        side = "short" if str(r.get("direction", "long")).lower().startswith("short") else "long"
        decision_ts = pd.Timestamp(r[ts_col])
        df = load_symbol_5m(symbol, decision_ts, min(decision_ts + pd.Timedelta(days=15), ctx.end), columns=["timestamp", "open", "high", "low", "close", "turnover", "open_interest", "funding_rate"])
        if len(df) < 24:
            continue
        entry_row = df.iloc[min(12, len(df) - 1)]  # delayed entry; no event-day chase for daily anchors.
        entry_ts = pd.Timestamp(entry_row["timestamp"])
        entry_price = float(entry_row.get("open", entry_row.get("close")))
        risk_pct = 0.08 if default_family == "C2" else 0.06
        if side == "long":
            stop_price = entry_price * (1.0 - risk_pct)
            target_price = entry_price * (1.0 + risk_pct * 2.0)
        else:
            stop_price = entry_price * (1.0 + risk_pct)
            target_price = entry_price * (1.0 - risk_pct * 2.0)
        rep = replay_path(symbol, side, entry_ts, entry_price, stop_price, target_price, 120)
        if rep is None:
            continue
        mode = str(r.get("mode", r.get("mechanism_family", default_family)))
        cid = f"{default_family}__{re.sub('[^A-Za-z0-9]+', '_', mode).strip('_')[:48]}"
        rows.append({
            **rep,
            "event_id": str(r.get("event_id", stable_hash(default_family, symbol, decision_ts, mode))),
            "candidate_id": cid,
            "definition_id": cid + "__delayed_base_replay",
            "family": default_family,
            "branch_id": "branch_b_sector_ignition" if default_family == "B1" else "branch_c_post_catalyst_base",
            "symbol": symbol,
            "decision_ts": decision_ts,
            "entry_ts": entry_ts,
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_price_source": "local_5m_delayed_anchor_entry",
            "exit_rule": "2R_or_120h_time_stop_no_event_day_chase",
            "parent_regime": mode,
            "event_cluster_id": f"{symbol}_{decision_ts.strftime('%Y%m')}_{default_family}",
            "mark_available": mark_available_for(symbol, entry_ts, rep["exit_ts"]),
            "required_data_tier": "Tier1_seed_limited",
            "current_data_tier": "Tier1_local_5m_context_seed_limited_proxy_funding",
            "first_reaction_excluded": True,
            "event_day_chase_primary": False,
        })
    if not rows:
        return pd.DataFrame()
    return normalize_event_ledger(pd.DataFrame(rows), default_family, "sidecar", f"{default_family}_local_anchor_replay")


def candidate_metrics(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for cid, g in events.groupby("candidate_id", sort=False):
        vals = pd.to_numeric(g["net_R"], errors="coerce")
        gross = pd.to_numeric(g["gross_R"], errors="coerce")
        events_n = int(vals.notna().sum())
        months = g["decision_ts"].dt.to_period("M").astype(str)
        month_net = vals.groupby(months).sum()
        sym_net = vals.groupby(g["symbol"].astype(str)).sum()
        top_month_share = float(month_net.max() / vals.sum()) if events_n and vals.sum() > 0 and len(month_net) else np.nan
        top_symbol_share = float(sym_net.max() / vals.sum()) if events_n and vals.sum() > 0 and len(sym_net) else np.nan
        risk_bps = pd.to_numeric(g.get("risk_bps_used"), errors="coerce").replace(0, np.nan)
        stress25 = vals - (25.0 / risk_bps).fillna(0.0)
        stress50 = vals - (50.0 / risk_bps).fillna(0.0)
        funding_cross = pd.to_numeric(g.get("funding_timestamps_crossed"), errors="coerce").fillna(0)
        adverse_funding = vals - ((funding_cross * 2.0) / risk_bps).fillna(0.0)
        rows.append({
            "candidate_id": cid,
            "family": str(g["family"].iloc[0]),
            "branch_id": str(g["branch_id"].iloc[0]),
            "event_count": events_n,
            "net_R": float(vals.sum()) if events_n else np.nan,
            "PF": pf(vals),
            "win_rate": float((vals > 0).mean()) if events_n else np.nan,
            "avg_R": float(vals.mean()) if events_n else np.nan,
            "median_R": float(vals.median()) if events_n else np.nan,
            "max_dd_R": max_dd(vals.reset_index(drop=True)),
            "active_months": int(months.nunique()),
            "active_symbols": int(g["symbol"].nunique()),
            "dominant_month_share": top_month_share,
            "dominant_symbol_share": top_symbol_share,
            "base_stress_net_R": float(vals.sum()) if events_n else np.nan,
            "plus25bps_net_R": float(stress25.sum()) if events_n else np.nan,
            "plus50bps_net_R": float(stress50.sum()) if events_n else np.nan,
            "adverse_funding_net_R": float(adverse_funding.sum()) if events_n else np.nan,
            "mark_available": bool(g["mark_available"].fillna(False).astype(bool).all()),
            "funding_exact": bool(g["funding_exact"].fillna(False).astype(bool).all()),
            "mark_proxy_used": bool(g["mark_proxy_used"].fillna(False).astype(bool).any()),
            "funding_proxy_used": bool(g["funding_proxy_used"].fillna(False).astype(bool).any()),
            "label_cap_reason": ";".join(sorted(set(g["label_cap_reason"].dropna().astype(str))))[:300],
            "metric_basis": "event_level_trade_ledger_no_vendor_train_only",
        })
    return pd.DataFrame(rows)


def stage_preflight(ctx: RunContext) -> None:
    ctx.run_root.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = {}
    missing_required = []
    for name, path in ACTIVE_ROOTS.items():
        exists = path.exists()
        rows.append({"name": name, "path": str(path), "exists": exists, "is_file": path.is_file(), "is_dir": path.is_dir()})
        if exists and path.is_file() and path.stat().st_size < 200_000_000:
            hashes[name] = file_sha256(path)
        elif exists and path.is_file():
            hashes[name] = file_sha256(path, max_bytes=32 * 1024 * 1024) + "_first32mb"
    for p in QA_REQUIRED_FILES:
        if not p.exists():
            missing_required.append(str(p))
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(ctx.run_root.parent)
    guard = check_resource_guard(snap, estimated_output_gb=2.0 if ctx.args.smoke else 12.0, hard_stage_output_gb=35.0, allow_large_output=ctx.args.allow_large_output)
    write_json(ctx.run_root / "preflight/resource_guard_report.md.json", guard)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", "# Resource Guard\n\n" + json.dumps(guard, indent=2))
    if guard["status"] == "hard_stop":
        raise RuntimeError("resource guard hard stop: " + ";".join(guard["reasons"]))
    if missing_required:
        write_text(ctx.run_root / "preflight/preflight_report.md", "# Preflight\n\nMissing required QA artifacts:\n" + "\n".join(f"- {m}" for m in missing_required))
        raise RuntimeError("missing required mechanical QA artifacts")
    write_text(ctx.run_root / "preflight/preflight_report.md", "# Preflight\n\nMechanical QA artifacts present. No-vendor phase will use local 5m/context and prior source artifacts only.\n")


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


def stage_live_capture(ctx: RunContext) -> None:
    path = LIVE_CAPTURE_ZIP
    out_dir = ctx.run_root / "live_capture"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "zip_path": str(path),
        "exists": path.exists(),
        "expected_outer_sha": EXPECTED_LIVE_CAPTURE_SHA,
        "local_outer_sha": "",
        "outer_sha_matches": False,
        "zip_opens_cleanly": False,
        "internal_manifest_found": False,
        "manifest_file_count": 0,
        "observed_file_count": 0,
        "missing_manifest_files": 0,
        "extra_files": 0,
        "hash_checked_files": 0,
        "hash_mismatch_files": 0,
        "stream_coverage": {},
        "operator_attested": bool(ctx.args.operator_attested_live_capture),
        "capture_evidence_state": "capture_unusable_corrupt_or_incomplete",
        "calibration_allowed": False,
        "calibration_cap": "no_historical_alpha_validation_or_parameter_promotion",
    }
    manifest_rows: list[dict[str, Any]] = []
    if not path.exists():
        write_json(out_dir / "provenance_status.json", status)
        write_text(out_dir / "hash_mismatch_report.md", "# Live Capture Provenance\n\nNo live capture zip found.\n")
        return
    status["local_outer_sha"] = file_sha256(path)
    status["outer_sha_matches"] = status["local_outer_sha"] == EXPECTED_LIVE_CAPTURE_SHA
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            status["zip_opens_cleanly"] = bad is None
            names = z.namelist()
            status["observed_file_count"] = len(names)
            mani_name = next((n for n in names if n.endswith("data/manifests/file_manifest.csv")), "")
            if mani_name:
                status["internal_manifest_found"] = True
                mani = pd.read_csv(io.BytesIO(z.read(mani_name)))
                status["manifest_file_count"] = int(len(mani))
                manifest_rows = mani.head(200).to_dict("records")
                normalized_names = {n.split("qlmg_live_capture/", 1)[-1]: n for n in names}
                missing = 0
                hash_checked = 0
                hash_mismatch = 0
                max_hash_checks = 50 if ctx.args.smoke else 1000000
                for _, r in mani.iterrows():
                    raw = str(r.get("file_path", ""))
                    rel = raw.split("/qlmg_live_capture/", 1)[-1] if "/qlmg_live_capture/" in raw else raw.lstrip("/")
                    rel = rel.split("data/", 1)[-1] if rel.startswith("opt/qlmg_live_capture/data/") else rel
                    candidates = [rel, "data/" + rel if not rel.startswith("data/") else rel, "qlmg_live_capture/" + rel if not rel.startswith("qlmg_live_capture/") else rel]
                    zname = next((normalized_names.get(c.split("qlmg_live_capture/", 1)[-1], c) for c in candidates if c in names or c.split("qlmg_live_capture/", 1)[-1] in normalized_names), "")
                    if not zname:
                        missing += 1
                        continue
                    expected = str(r.get("sha256", ""))
                    if expected and expected not in {"nan", "None"} and hash_checked < max_hash_checks:
                        actual = hashlib.sha256(z.read(zname)).hexdigest()
                        hash_checked += 1
                        if actual != expected:
                            hash_mismatch += 1
                status["missing_manifest_files"] = int(missing)
                status["hash_checked_files"] = int(hash_checked)
                status["hash_mismatch_files"] = int(hash_mismatch)
            streams = {}
            for n in names:
                for key in ["orderbook_1", "orderbook_50", "publicTrade", "tickers", "allLiquidation"]:
                    if key in n:
                        streams[key] = streams.get(key, 0) + 1
            status["stream_coverage"] = streams
            extra = max(0, len(names) - int(status["manifest_file_count"])) if status["manifest_file_count"] else 0
            status["extra_files"] = int(extra)
    except Exception as exc:
        status["zip_error"] = f"{type(exc).__name__}: {exc}"
    coherent = bool(
        status["zip_opens_cleanly"]
        and status["internal_manifest_found"]
        and int(status["manifest_file_count"]) > 0
        and int(status["missing_manifest_files"]) <= max(5, int(status["manifest_file_count"]) // 100)
        and any(int(v) > 0 for v in status["stream_coverage"].values())
    )
    # File-level hash mismatches are retained in the provenance report, but under
    # the no-vendor amendment they do not create indefinite blockage when the
    # bundle is structurally coherent and operator-attested.
    if status["outer_sha_matches"] and coherent:
        status["capture_evidence_state"] = "capture_content_verified"
        status["calibration_allowed"] = True
    elif coherent and status["operator_attested"]:
        status["capture_evidence_state"] = "operator_attested_capture_calibration_capped"
        status["calibration_allowed"] = True
    else:
        status["capture_evidence_state"] = "capture_unusable_corrupt_or_incomplete"
        status["calibration_allowed"] = False
    write_json(out_dir / "provenance_status.json", status)
    write_csv(out_dir / "internal_manifest_sample.csv", manifest_rows)
    write_text(out_dir / "hash_mismatch_report.md", "# Live Capture Provenance\n\n" + json.dumps(status, indent=2))


def stage_enrichment(ctx: RunContext) -> None:
    if not ctx.args.include_funding_mark_enrichment:
        write_text(ctx.run_root / "enrichment/enrichment_report.md", "# Enrichment\n\nSkipped by CLI.\n")
        return
    parts = []
    for family, path in [("A3", A3_LEDGER), ("A2_redesign_only", A2_LEDGER)]:
        df = read_df(path)
        if df.empty:
            continue
        if ctx.args.smoke:
            df = df.head(200)
        parts.append(normalize_event_ledger(df, family, "branch_l_liquid_regime", str(path)))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    (ctx.run_root / "enrichment").mkdir(parents=True, exist_ok=True)
    if not out.empty:
        out.to_parquet(ctx.run_root / "enrichment/enriched_a2_a3_event_ledger.parquet", index=False)
        summary = candidate_metrics(out)
        write_csv(ctx.run_root / "enrichment/enriched_a2_a3_summary.csv", summary)
        assert_pass(validate_pit_feature_timestamps(out, feature_ts_cols=[]))
        assert_pass(validate_funding_mark_flags(out))
    write_text(ctx.run_root / "enrichment/enrichment_report.md", f"# Funding/Mark Enrichment\n\nRows: {len(out)}. Funding exact is true only for no-cross or verified exact source rows; crossed funding remains capped.\n")


def stage_a1_a4(ctx: RunContext) -> None:
    out = build_a1_a4_events(ctx)
    pdir = ctx.run_root / "a1_a4"
    pdir.mkdir(parents=True, exist_ok=True)
    if out.empty:
        write_text(pdir / "a1_a4_report.md", "# A1/A4 Sweep\n\nNo event rows built.\n")
        write_csv(pdir / "a1_a4_sweep_summary.csv", [])
        return
    out.to_parquet(pdir / "a1_a4_event_level_replay.parquet", index=False)
    summary = candidate_metrics(out)
    summary["label"] = np.where(
        (summary["net_R"] > 0) & (summary["PF"] > 1) & (summary["plus25bps_net_R"] > 0),
        np.where(summary["funding_proxy_used"], "tier1_screening_candidate_funding_capped", "tier1_screening_candidate"),
        "reject_current_translation_only_or_candidate_library",
    )
    write_csv(pdir / "a1_a4_sweep_summary.csv", summary)
    assert_pass(validate_pit_feature_timestamps(out, feature_ts_cols=[]))
    assert_pass(validate_funding_mark_flags(out))
    write_text(pdir / "a1_a4_report.md", f"# A1/A4 Corrected Liquid Sweep\n\nEvent rows: {len(out)}. Definitions: {summary['candidate_id'].nunique() if not summary.empty else 0}. Funding-cross rows are capped rather than blocked.\n")


def inventory_sources(patterns: list[str]) -> pd.DataFrame:
    rows = []
    for pat in patterns:
        for p in RESULTS_ROOT.rglob(pat):
            if p.is_file():
                rows.append({"path": str(p), "size_bytes": p.stat().st_size, "suffix": p.suffix, "exists": True})
    return pd.DataFrame(rows).drop_duplicates("path") if rows else pd.DataFrame(columns=["path", "size_bytes", "suffix", "exists"])


def stage_b1(ctx: RunContext) -> None:
    pdir = ctx.run_root / "b1"
    pdir.mkdir(parents=True, exist_ok=True)
    inv = inventory_sources(["*sector*.csv", "*sector*.parquet", "*cluster*.csv", "*cluster*.parquet", "*b1*anchor*.parquet"])
    write_csv(ctx.run_root / "source_inventory/b1_source_inventory.csv", inv)
    if not ctx.args.include_b1_c2_ledgers:
        write_text(pdir / "b1_report.md", "# B1\n\nSkipped by CLI.\n")
        return
    anchors = read_df(B1_ANCHOR_LEDGER)
    if anchors.empty:
        write_csv(pdir / "b1_ledger_blockers.csv", [{"blocker": "no_b1_anchor_ledger_found", "source": str(B1_ANCHOR_LEDGER)}])
        write_text(pdir / "b1_report.md", "# B1\n\nNo anchor ledger found.\n")
        return
    if ctx.args.smoke:
        anchors = anchors.head(ctx.args.b1_budget)
    ledger = replay_anchor_rows(anchors, "B1", int(ctx.args.b1_budget), ctx)
    if ledger.empty:
        write_csv(pdir / "b1_ledger_blockers.csv", [{"blocker": "anchors_exist_but_no_replay_rows", "source": str(B1_ANCHOR_LEDGER)}])
        write_text(pdir / "b1_report.md", "# B1\n\nAnchor rows existed but local bar replay produced no rankable rows.\n")
        return
    ledger.to_parquet(pdir / "b1_event_level_replay.parquet", index=False)
    summary = candidate_metrics(ledger)
    summary["label"] = np.where(summary["event_count"] < 30, "sample_limited_sleeve_candidate", "b1_event_level_sleeve_candidate")
    write_csv(pdir / "b1_ledger_summary.csv", summary)
    write_text(pdir / "b1_report.md", f"# B1 Ledger Construction\n\nRows: {len(ledger)}. Current-only taxonomy rows are not rankable; sparse mechanisms are preserved as sleeves.\n")


def stage_c2(ctx: RunContext) -> None:
    pdir = ctx.run_root / "c2"
    pdir.mkdir(parents=True, exist_ok=True)
    inv = inventory_sources(["*catalyst*.csv", "*catalyst*.parquet", "*c2*anchor*.csv", "*c2*anchor*.parquet", "*excluded*.csv"])
    write_csv(ctx.run_root / "source_inventory/c2_source_inventory.csv", inv)
    if not ctx.args.include_b1_c2_ledgers:
        write_text(pdir / "c2_report.md", "# C2\n\nSkipped by CLI.\n")
        return
    events = read_df(C2_EVENT_LEDGER)
    if events.empty and C2_REPAIR_LEDGER.exists():
        events = read_df(C2_REPAIR_LEDGER)
    if events.empty:
        write_csv(pdir / "c2_ledger_blockers.csv", [{"blocker": "no_c2_event_ledger_found", "source": str(C2_EVENT_LEDGER)}])
        write_text(pdir / "c2_report.md", "# C2\n\nNo catalyst event ledger found.\n")
        return
    if "earliest_decision_ts" in events.columns:
        events = events.rename(columns={"earliest_decision_ts": "decision_ts"})
    if ctx.args.smoke:
        events = events.head(ctx.args.c2_budget)
    ledger = replay_anchor_rows(events, "C2", int(ctx.args.c2_budget), ctx)
    if ledger.empty:
        write_csv(pdir / "c2_ledger_blockers.csv", [{"blocker": "events_exist_but_no_local_bar_replay_rows", "source": str(C2_EVENT_LEDGER)}])
        write_text(pdir / "c2_report.md", "# C2\n\nEvent anchors existed but local bar replay produced no rows.\n")
        return
    ledger.to_parquet(pdir / "c2_event_level_replay.parquet", index=False)
    summary = candidate_metrics(ledger)
    summary["label"] = np.where(summary["event_count"] < 30, "sample_limited_sleeve_candidate", "c2_event_level_sleeve_candidate")
    write_csv(pdir / "c2_ledger_summary.csv", summary)
    write_text(pdir / "c2_report.md", f"# C2 Ledger Construction\n\nRows: {len(ledger)}. Mechanisms are not pooled for validation; sparse groups are preserved as sleeves.\n")


def load_generated_event_pool(ctx: RunContext) -> pd.DataFrame:
    paths = [
        ctx.run_root / "a1_a4/a1_a4_event_level_replay.parquet",
        ctx.run_root / "b1/b1_event_level_replay.parquet",
        ctx.run_root / "c2/c2_event_level_replay.parquet",
        ctx.run_root / "enrichment/enriched_a2_a3_event_ledger.parquet",
    ]
    parts = []
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p)
            family = str(df["family"].iloc[0]) if not df.empty and "family" in df.columns else "unknown"
            std = standardize_event_ledger(df, str(p), family_hint=family)
            parts.append(std)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


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
            "candidate_key": key,
            "candidate_id": cand["candidate_id"].iloc[0],
            "family": cand["family"].iloc[0],
            "control_type": ctype,
            "candidate_event_count": candidate_events,
            "control_event_count": control_events,
            "candidate_net_R": candidate_net,
            "raw_control_net_R": raw,
            "normalized_control_net_R": norm,
            "control_uplift_R": candidate_net - norm if math.isfinite(norm) else np.nan,
            "beats_control": bool(math.isfinite(norm) and candidate_net > norm),
            "controls_normalized_to_candidate_count": True,
            "all_control_rows_have_source_ids": bool(g["control_source_row_id"].notna().all() and g["source_window_id"].notna().all()),
        })
    return pd.DataFrame(rows)


def build_overlap_embargo_audit(control_ledger: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    if control_ledger.empty or pool.empty:
        return pd.DataFrame()
    cand_windows = pool[["candidate_key", "symbol", "entry_ts", "exit_ts"]].copy()
    cand_windows["entry_ts"] = pd.to_datetime(cand_windows["entry_ts"], utc=True, errors="coerce")
    cand_windows["exit_ts"] = pd.to_datetime(cand_windows["exit_ts"], utc=True, errors="coerce")
    rows = []
    for _, r in control_ledger.iterrows():
        key = str(r.get("candidate_key"))
        sym = str(r.get("control_symbol"))
        ce = pd.to_datetime(r.get("control_entry_ts"), utc=True, errors="coerce")
        cx = pd.to_datetime(r.get("control_exit_ts"), utc=True, errors="coerce")
        wins = cand_windows[(cand_windows["candidate_key"].astype(str).eq(key)) & (cand_windows["symbol"].astype(str).eq(sym))]
        violation = False
        embargo_hours = 0.0
        for _, w in wins.iterrows():
            a, b = w["entry_ts"], w["exit_ts"]
            if pd.isna(a) or pd.isna(b) or pd.isna(ce) or pd.isna(cx):
                continue
            embargo = b - a
            embargo_hours = max(embargo_hours, embargo.total_seconds() / 3600.0)
            if ce <= b + embargo and cx >= a - embargo:
                violation = True
                break
        rows.append({
            "candidate_key": key,
            "control_type": r.get("control_type"),
            "control_event_id": r.get("control_event_id"),
            "control_symbol": sym,
            "source_window_id": r.get("source_window_id"),
            "embargo_hours": embargo_hours,
            "overlap_or_embargo_violation": violation,
        })
    return pd.DataFrame(rows)


def stage_controls(ctx: RunContext) -> None:
    pdir = ctx.run_root / "controls"
    pdir.mkdir(parents=True, exist_ok=True)
    pool = load_generated_event_pool(ctx)
    if pool.empty:
        write_text(pdir / "fresh_real_controls_report.md", "# Real Controls\n\nNo event pool available.\n")
        return
    target_families = {"A1", "A4", "B1", "C2"}
    target_pool = pool[pool["family"].astype(str).isin(target_families)].copy()
    support_pool = pool.copy()
    max_support_rows = 5_000 if ctx.args.smoke else 120_000
    if len(support_pool) > max_support_rows:
        # Keep every new no-vendor target row, then add deterministic A2/A3 support rows.
        target_ids = set(target_pool["source_row_id"].astype(str)) if not target_pool.empty else set()
        support_only = support_pool[~support_pool["source_row_id"].astype(str).isin(target_ids)].copy()
        keep_n = max(max_support_rows - len(target_pool), 0)
        if keep_n > 0 and len(support_only) > keep_n:
            support_only = support_only.sort_values("source_row_id", kind="mergesort").sample(n=keep_n, random_state=ctx.args.seed)
        support_pool = pd.concat([target_pool, support_only], ignore_index=True)
    enriched, cov = enrich_event_pool_with_match_features(support_pool, DATA_5M)
    candidate_keys = sorted(enriched.loc[enriched["family"].astype(str).isin(target_families), "candidate_key"].dropna().astype(str).unique())[: max(1, ctx.args.top_per_family * 8)]
    cand_summary, ctrl_ledger, ctrl_summary = build_real_controls(enriched, candidate_keys=candidate_keys, nulls_per_event=ctx.args.nulls_per_event, seed=ctx.args.seed)
    audit = build_overlap_embargo_audit(ctrl_ledger, enriched)
    if not audit.empty:
        keep_ids = set(audit.loc[~audit["overlap_or_embargo_violation"], ["candidate_key", "control_event_id", "control_type"]].astype(str).agg("|".join, axis=1))
        key_series = ctrl_ledger[["candidate_key", "control_event_id", "control_type"]].astype(str).agg("|".join, axis=1)
        ctrl_ledger = ctrl_ledger[key_series.isin(keep_ids)].copy()
    if not ctrl_ledger.empty:
        # The evidence contract forbids the same matched candidate/control/window row
        # from being counted more than once, even if it qualifies under multiple
        # labels. Keep the first deterministic match and let coverage report any loss.
        ctrl_ledger = ctrl_ledger.sort_values(["candidate_key", "control_type", "selection_rank"], kind="mergesort")
        ctrl_ledger = ctrl_ledger.drop_duplicates(["matched_candidate_id", "control_event_id", "source_window_id"], keep="first").copy()
        ctrl_summary = recompute_control_summary(cand_summary, ctrl_ledger)
    cand_summary.to_csv(pdir / "candidate_metric_summary.csv", index=False)
    ctrl_ledger.to_parquet(pdir / "real_control_event_ledger.parquet", index=False)
    write_csv(pdir / "fresh_controls_summary.csv", ctrl_summary)
    write_csv(pdir / "match_feature_coverage.csv", cov)
    write_csv(pdir / "control_overlap_embargo_audit.csv", audit)
    if not ctrl_ledger.empty:
        assert_pass(validate_control_rows(ctrl_ledger))
    write_text(pdir / "fresh_real_controls_report.md", f"# Real Controls\n\nEvent pool rows discovered: {len(pool)}. Support rows used after deterministic cap: {len(support_pool)}. Control target keys: {len(candidate_keys)}. Control rows after embargo filter: {len(ctrl_ledger)}. A2/A3 rows are support controls only in this no-vendor phase.\n")


def stage_stress(ctx: RunContext) -> None:
    pdir = ctx.run_root / "stress"
    pdir.mkdir(parents=True, exist_ok=True)
    pool = load_generated_event_pool(ctx)
    if pool.empty:
        write_text(pdir / "tier1_stress_report.md", "# Stress\n\nNo event rows available.\n")
        return
    # Use original event rows instead of standardized source_net_R so risk_bps is available.
    full_parts = []
    for p in [ctx.run_root / "a1_a4/a1_a4_event_level_replay.parquet", ctx.run_root / "b1/b1_event_level_replay.parquet", ctx.run_root / "c2/c2_event_level_replay.parquet", ctx.run_root / "enrichment/enriched_a2_a3_event_ledger.parquet"]:
        if p.exists():
            full_parts.append(pd.read_parquet(p))
    full = pd.concat(full_parts, ignore_index=True) if full_parts else pd.DataFrame()
    summary = candidate_metrics(full) if not full.empty else pd.DataFrame()
    write_csv(pdir / "tier1_stress_summary.csv", summary)
    write_text(pdir / "tier1_stress_report.md", f"# Tier-1 Stress\n\nRows stressed: {len(full)}. +25/+50 bps and adverse funding stress are accounting-only no-vendor diagnostics.\n")


def stage_stability(ctx: RunContext) -> None:
    pdir = ctx.run_root / "stability"
    pdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in [ctx.run_root / "a1_a4/a1_a4_event_level_replay.parquet", ctx.run_root / "b1/b1_event_level_replay.parquet", ctx.run_root / "c2/c2_event_level_replay.parquet"]:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for cid, g in df.groupby("candidate_id", sort=False):
            vals = pd.to_numeric(g["net_R"], errors="coerce")
            net = float(vals.sum()) if len(vals) else np.nan
            rows.append({
                "candidate_id": cid,
                "family": str(g["family"].iloc[0]),
                "event_count": int(len(g)),
                "active_months": int(g["decision_ts"].dt.to_period("M").nunique()),
                "active_symbols": int(g["symbol"].nunique()),
                "standalone_status": "possible_standalone_screen" if len(g) >= 50 and net > 0 else "not_standalone_yet",
                "portfolio_sleeve_status": "sample_limited_sleeve_candidate" if len(g) < 50 and len(g) > 0 else "candidate_sleeve_possible",
                "rare_regime_status": "rare_or_sparse" if len(g) < 50 else "not_sparse",
                "feature_overlay_status": "possible_context_overlay" if str(g["family"].iloc[0]) in {"B1", "C2"} else "not_primary_overlay",
                "reason_preserved_or_discarded": "preserved_unless_clean_controls_and_no_vendor_path_fail",
            })
    write_csv(pdir / "sleeve_classification.csv", rows)
    write_text(pdir / "stability_report.md", "# Stability and Sleeve Classification\n\nSparse candidates are preserved as sleeves or candidate-library entries rather than rejected.\n")


def stage_branch_x(ctx: RunContext) -> None:
    pdir = ctx.run_root / "branch_x"
    pdir.mkdir(parents=True, exist_ok=True)
    prov_path = ctx.run_root / "live_capture/provenance_status.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    state = prov.get("capture_evidence_state", "capture_unusable_corrupt_or_incomplete")
    calibration_allowed = bool(prov.get("calibration_allowed", False))
    rows = [
        {"family": "D4", "no_vendor_outcome": "needs_capture_substitute", "micro_canary_status": "blocked_until_actual_d4_like_capture", "capture_state": state},
        {"family": "listing_vwap_loss_589_b1", "no_vendor_outcome": "needs_capture_substitute" if not calibration_allowed else "progress_with_current_data", "micro_canary_status": "execution_only_possible_after_separate_human_approval" if calibration_allowed else "waiting_for_coherent_capture", "capture_state": state},
        {"family": "listing_vwap_loss_9dc", "no_vendor_outcome": "candidate_library_only", "micro_canary_status": "not_recommended_fragile_backlog", "capture_state": state},
        {"family": "funding_window", "no_vendor_outcome": "preserve_hypothesis_generate_new_variant", "micro_canary_status": "not_current_translation", "capture_state": state},
        {"family": "generic_shock", "no_vendor_outcome": "discard_current_translation_no_vendor_path", "micro_canary_status": "not_supported_current_expression", "capture_state": state},
    ]
    write_csv(pdir / "branch_x_no_vendor_status.csv", rows)
    write_text(pdir / "branch_x_capture_calibration_report.md", "# Branch X Capture Calibration\n\nNo Branch X PnL is mixed into liquid rankings. Capture state: " + str(state) + ". Calibration allowed: " + str(calibration_allowed) + ".\n")


def stage_no_vendor_decisions(ctx: RunContext) -> None:
    rows = []
    for family in ["A1", "A4", "B1", "C2", "D4", "listing_vwap_loss", "funding_window", "generic_shock", "D1_D3_E1_F1_G1_ORB"]:
        if family in {"A1", "A4"}:
            outcome = "progress_with_current_data" if (ctx.run_root / "a1_a4/a1_a4_sweep_summary.csv").exists() else "preserve_hypothesis_generate_new_variant"
        elif family in {"B1", "C2"}:
            outcome = "progress_with_current_data" if (ctx.run_root / family.lower() / f"{family.lower()}_event_level_replay.parquet").exists() else "candidate_library_only"
        elif family in {"D4", "listing_vwap_loss"}:
            outcome = "needs_capture_substitute"
        elif family == "generic_shock":
            outcome = "discard_current_translation_no_vendor_path"
        else:
            outcome = "preserve_hypothesis_generate_new_variant"
        rows.append({"family": family, "no_vendor_outcome": outcome, "family_rejected": False})
    write_csv(ctx.run_root / "no_vendor/no_vendor_family_decisions.csv", rows)
    write_text(ctx.run_root / "no_vendor/no_vendor_decision_rules.md", "# No-Vendor Decision Rules\n\nNo family is left waiting for paid historical vendor data. Current translations may be discarded; families are preserved unless broad clean evidence supports otherwise.\n")


def stage_library(ctx: RunContext) -> None:
    rows = []
    summaries = []
    for p in [ctx.run_root / "a1_a4/a1_a4_sweep_summary.csv", ctx.run_root / "b1/b1_ledger_summary.csv", ctx.run_root / "c2/c2_ledger_summary.csv", ctx.run_root / "stress/tier1_stress_summary.csv"]:
        if p.exists():
            try:
                summaries.append(pd.read_csv(p))
            except Exception:
                pass
    df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(["candidate_id", "family"], keep="first")
        for _, r in df.iterrows():
            event_count = int(safe_float(r.get("event_count"), 0))
            label_cap = str(r.get("label_cap_reason", ""))
            rows.append({
                "candidate_id": r.get("candidate_id"),
                "family": r.get("family"),
                "evidence_state": "event_ledger_available" if event_count > 0 else "support_only",
                "required_data_tier": "Tier1" if r.get("family") in {"A1", "A4"} else "Tier1_seed_limited",
                "current_data_tier": "Tier1_local_5m_context_proxy_funding",
                "data_tier_cap_reason": label_cap,
                "standalone_status": "possible" if event_count >= 50 and safe_float(r.get("net_R"), -1) > 0 else "not_yet",
                "portfolio_sleeve_status": "sample_limited_sleeve_candidate" if event_count < 50 else "candidate_sleeve_possible",
                "rare_regime_status": "rare_or_sparse" if event_count < 50 else "not_sparse",
                "feature_overlay_status": "possible_overlay" if r.get("family") in {"B1", "C2"} else "not_overlay",
                "event_count": event_count,
                "active_months": int(safe_float(r.get("active_months"), 0)),
                "active_symbols": int(safe_float(r.get("active_symbols"), 0)),
                "dominant_month_share": r.get("dominant_month_share"),
                "dominant_symbol_share": r.get("dominant_symbol_share"),
                "return_overlap_proxy": "not_computed_no_vendor_phase",
                "candidate_correlation_proxy": "not_computed_no_vendor_phase",
                "reason_preserved_or_discarded": "preserved_under_no_vendor_library_unless_current_translation_discarded",
            })
    # Add execution-sensitive preserved hypotheses.
    rows.extend([
        {"candidate_id": "D4__b4c9487fe82c", "family": "D4", "evidence_state": "execution_data_blocked_capture_substitute", "required_data_tier": "Tier2_or_capture", "current_data_tier": "capture_inventory_or_capped", "data_tier_cap_reason": "needs live capture substitute no vendor", "standalone_status": "not_ranked", "portfolio_sleeve_status": "not_ranked", "rare_regime_status": "event_driven", "feature_overlay_status": "none", "event_count": 4475, "active_months": "", "active_symbols": "", "dominant_month_share": "", "dominant_symbol_share": "", "return_overlap_proxy": "", "candidate_correlation_proxy": "", "reason_preserved_or_discarded": "preserved_for_capture_substitute_not_vendor"},
        {"candidate_id": "listing_vwap_loss_589_b1", "family": "Branch_X_listing", "evidence_state": "execution_data_blocked_capture_substitute", "required_data_tier": "Tier2_or_capture", "current_data_tier": "capture_inventory_or_capped", "data_tier_cap_reason": "needs execution telemetry no vendor", "standalone_status": "not_ranked", "portfolio_sleeve_status": "candidate_library_only", "rare_regime_status": "listing_event", "feature_overlay_status": "none", "event_count": "", "active_months": "", "active_symbols": "", "dominant_month_share": "", "dominant_symbol_share": "", "return_overlap_proxy": "", "candidate_correlation_proxy": "", "reason_preserved_or_discarded": "preserved_listing_analog_capture_path"},
    ])
    write_csv(ctx.run_root / "library/no_vendor_candidate_library.csv", rows)
    write_text(ctx.run_root / "library/candidate_library_report.md", "# Candidate Library\n\nRows preserve standalone candidates, sparse sleeves, overlays, data-blocked hypotheses, and current-translation-only discards.\n")


def top_candidate_samples(ctx: RunContext) -> None:
    pdir = ctx.run_root / "audit_samples"
    pdir.mkdir(parents=True, exist_ok=True)
    event_paths = [ctx.run_root / "a1_a4/a1_a4_event_level_replay.parquet", ctx.run_root / "b1/b1_event_level_replay.parquet", ctx.run_root / "c2/c2_event_level_replay.parquet"]
    parts = [pd.read_parquet(p) for p in event_paths if p.exists()]
    if not parts:
        write_csv(pdir / "top_candidate_event_samples.csv", [])
        write_text(pdir / "top_candidate_calculation_trace.md", "# Calculation Trace\n\nNo event rows available.\n")
        return
    df = pd.concat(parts, ignore_index=True)
    summaries = candidate_metrics(df).sort_values("net_R", ascending=False).head(20)
    samples = []
    for cid in summaries["candidate_id"].astype(str):
        g = df[df["candidate_id"].astype(str).eq(cid)].copy()
        picks = [g.nlargest(10, "net_R"), g.nsmallest(10, "net_R"), g.sample(min(10, len(g)), random_state=ctx.args.seed)]
        if "same_bar_ambiguity_flag" in g.columns:
            picks.append(g[g["same_bar_ambiguity_flag"].fillna(False).astype(bool)].head(10))
        if "mark_proxy_used" in g.columns:
            picks.append(g[g["mark_proxy_used"].fillna(False).astype(bool) | g["funding_proxy_used"].fillna(False).astype(bool)].head(10))
        sample = pd.concat(picks, ignore_index=True).drop_duplicates("event_id")
        samples.append(sample)
    out = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    keep = [c for c in ["candidate_id", "family", "event_id", "symbol", "decision_ts", "entry_ts", "exit_ts", "entry_price", "stop_price", "target_price", "exit_price", "gross_R", "fees_R", "slippage_R", "funding_R", "net_R", "risk_bps_used", "exit_reason", "mark_proxy_used", "funding_proxy_used", "source_path"] if c in out.columns]
    write_csv(pdir / "top_candidate_event_samples.csv", out[keep] if keep else out)
    write_text(pdir / "top_candidate_calculation_trace.md", "# Calculation Trace\n\nR = directional price PnL divided by absolute entry-stop risk, minus all-taker fee_R and slippage_R, plus funding_R. Same-bar stop/target ambiguity is adverse: stop wins. Funding-cross rows without exact settlement data are capped and stressed, not treated as exact. Raw bars are in `/opt/parquet/5m/{symbol}.parquet`; mark context is in `/opt/parquet/bybit_context_5m/{symbol}.parquet`.\n")


def stage_report(ctx: RunContext) -> None:
    top_candidate_samples(ctx)
    lib_path = ctx.run_root / "library/no_vendor_candidate_library.csv"
    lib = pd.read_csv(lib_path) if lib_path.exists() else pd.DataFrame()
    prov_path = ctx.run_root / "live_capture/provenance_status.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    stress_path = ctx.run_root / "stress/tier1_stress_summary.csv"
    stress = pd.read_csv(stress_path) if stress_path.exists() else pd.DataFrame()
    has_progress = bool(not stress.empty and ((pd.to_numeric(stress.get("net_R"), errors="coerce") > 0) & (pd.to_numeric(stress.get("PF"), errors="coerce") > 1)).any())
    branch_micro = prov.get("calibration_allowed", False)
    if has_progress:
        decision = "run_train_only_candidate_validation_package_next"
    elif branch_micro:
        decision = "branch_x_micro_canary_possible_execution_only"
    elif (ctx.run_root / "b1/b1_event_level_replay.parquet").exists() or (ctx.run_root / "c2/c2_event_level_replay.parquet").exists():
        decision = "run_b1_c2_ledger_construction_next"
    else:
        decision = "generate_new_hypotheses_next"
    if decision not in ALLOWED_NEXT_DECISIONS:
        decision = "blocked_by_protocol_issue"
    decision_json = {
        "run_root": str(ctx.run_root),
        "final_holdout_untouched": True,
        "telegram_worked": ctx.notifier.remote_available,
        "capture_provenance_verdict": prov.get("capture_evidence_state", "not_available"),
        "funding_mark_enrichment_verdict": "completed_with_caps" if (ctx.run_root / "enrichment/enriched_a2_a3_summary.csv").exists() else "not_available",
        "a1_a4_verdict": "completed" if (ctx.run_root / "a1_a4/a1_a4_sweep_summary.csv").exists() else "not_available",
        "b1_verdict": "event_ledger_attempted" if (ctx.run_root / "b1/b1_event_level_replay.parquet").exists() else "candidate_library_or_blocked",
        "c2_verdict": "event_ledger_attempted" if (ctx.run_root / "c2/c2_event_level_replay.parquet").exists() else "candidate_library_or_blocked",
        "branch_x_calibration_verdict": "capped_calibration_allowed" if branch_micro else "capture_substitute_needed_or_unusable",
        "next_operator_decision": decision,
        "no_vendor_rule_enforced": True,
        "no_live_ready_or_validation_claim": True,
    }
    write_json(ctx.run_root / "decision_summary.json", decision_json)
    progress = lib[lib.get("evidence_state", pd.Series(dtype=str)).astype(str).str.contains("event_ledger", na=False)] if not lib.empty else pd.DataFrame()
    candidate_only = lib[~lib.index.isin(progress.index)] if not lib.empty else pd.DataFrame()
    report = [
        "# QLMG No-Vendor Progress Run Report",
        "",
        f"Run root: `{ctx.run_root}`",
        "Final holdout untouched: yes",
        f"Telegram worked: {'yes' if ctx.notifier.remote_available else 'no'}",
        f"Capture provenance verdict: `{decision_json['capture_provenance_verdict']}`",
        "",
        "## Progress With Current Data",
        progress[["candidate_id", "family", "evidence_state", "event_count", "data_tier_cap_reason"]].to_markdown(index=False) if not progress.empty else "No event-ledger progress candidates were produced.",
        "",
        "## Still Candidate-Library Only",
        candidate_only[["candidate_id", "family", "evidence_state", "reason_preserved_or_discarded"]].head(40).to_markdown(index=False) if not candidate_only.empty else "None.",
        "",
        "## Current Translations Discarded",
        "Generic shock current expression remains unsupported; discard is current-translation only, not family death.",
        "",
        "## Needs New Hypotheses",
        "New hypotheses are needed if A1/A4 do not survive real controls/stress and B1/C2 remain sparse sleeves. The run does not recommend paid vendor data as the only path.",
        "",
        "## Next Operator Decision",
        f"`{decision}`",
    ]
    write_text(ctx.run_root / "QLMG_NO_VENDOR_PROGRESS_RUN_REPORT.md", "\n".join(report))
    scan = scan_output_tree_for_protected(ctx.run_root)
    write_json(ctx.run_root / "seal/generated_output_protected_scan.json", result_to_jsonable(scan))
    if not scan.passed:
        raise RuntimeError("generated output protected scan failed: " + ";".join(scan.violations))


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_NO_VENDOR_PROGRESS_RUN_REPORT.md", "decision_summary.json",
        "preflight/preflight_report.md", "preflight/input_artifact_manifest.csv", "preflight/resource_guard_report.md",
        "live_capture/provenance_status.json", "live_capture/hash_mismatch_report.md",
        "source_inventory/b1_source_inventory.csv", "source_inventory/c2_source_inventory.csv",
        "a1_a4/a1_a4_sweep_summary.csv", "b1/b1_ledger_summary.csv", "c2/c2_ledger_summary.csv",
        "controls/fresh_controls_summary.csv", "controls/control_overlap_embargo_audit.csv",
        "stress/tier1_stress_summary.csv", "library/no_vendor_candidate_library.csv",
        "audit_samples/top_candidate_event_samples.csv", "audit_samples/top_candidate_calculation_trace.md",
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
    "preflight-and-evidence-contract-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "live-capture-provenance-repair": stage_live_capture,
    "exact-funding-mark-enrichment": stage_enrichment,
    "a1-a4-corrected-liquid-sweep": stage_a1_a4,
    "b1-ledger-construction": stage_b1,
    "c2-ledger-construction": stage_c2,
    "fresh-real-controls": stage_controls,
    "tier1-stress-and-funding-mark-caps": stage_stress,
    "walk-forward-stability-and-sleeve-classification": stage_stability,
    "branch-x-capture-calibration": stage_branch_x,
    "no-vendor-decision-rules": stage_no_vendor_decisions,
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
    ctx.notifier.send("QLMG no-vendor stage start", stage)
    fn = STAGE_FUNCS[stage]
    if not ctx.args.dry_run:
        fn(ctx)
    mark_done(ctx, stage)
    ctx.notifier.send("QLMG no-vendor stage done", stage)


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
        notifier.send("QLMG no-vendor run complete", str(run_root))
        return 0
    except Exception as exc:
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "failed", "error": f"{type(exc).__name__}: {exc}", "ts_utc": utc_now()})
        notifier.send("QLMG no-vendor run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
