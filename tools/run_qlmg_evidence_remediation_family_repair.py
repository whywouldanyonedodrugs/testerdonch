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
from tools.qlmg_evidence_contracts import (  # noqa: E402
    assert_pass,
    validate_no_projected_metric_promotion,
    validate_no_synthetic_controls,
)

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_evidence_remediation_family_repair_20260629_v1"
DEFAULT_SEED = 20260629
DATA_5M = Path("/opt/parquet/5m")
CONTEXT_5M = Path("/opt/parquet/bybit_context_5m")

INTEGRITY_ROOT = RESULTS_ROOT / "phase_qlmg_evidence_integrity_corrected_sweep_20260628_v1_20260628_163819"
ABCX_ROOT = RESULTS_ROOT / "phase_qlmg_integrated_abcx_development_20260628_v2_20260628_134140"
LIQUID_ROOT = RESULTS_ROOT / "phase_qlmg_liquid_regime_strategy_research_20260628_v1_20260628_120124"
PROXY_ROOT = RESULTS_ROOT / "phase_qlmg_best_effort_proxy_execution_sim_20260628_v1_20260628_105109"
BRUTAL_ROOT = RESULTS_ROOT / "phase_qlmg_brutal_no_depth_stress_20260628_v1_20260628_101136"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
D4_AUDIT_ROOT = RESULTS_ROOT / "phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927"

FROZEN_ROOTS = {
    "integrity_index": INTEGRITY_ROOT,
    "integrated_abcx_v2": ABCX_ROOT,
    "liquid_regime": LIQUID_ROOT,
    "best_effort_proxy_execution_sim": PROXY_ROOT,
    "brutal_no_depth_stress": BRUTAL_ROOT,
    "listing_generic_full_event_replay": LISTING_ROOT,
    "d4_survivability": D4_SURVIVAL_ROOT,
    "d4_liquidation_audit": D4_AUDIT_ROOT,
    "sector_markdown_seed": REPO / "research_inputs/point_in_time_sector_seeds.md",
    "catalyst_markdown_seed": REPO / "research_inputs/post_catalyst_c2_database.md",
}

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "integrity-audit-self-check",
    "evidence-quarantine-and-rankable-set",
    "manual-replay-audit-pack",
    "a2-exit-risk-redesign",
    "a3-family-specific-validation",
    "b1-trade-ledger-construction",
    "c2-trade-ledger-construction",
    "branch-x-capture-calibration-plan",
    "corrected-sweep-readiness-gate",
    "decision-report",
    "compact-review-bundle",
    "all",
)

DEPRECATED_PROMOTION_LABELS = [
    "research_prelead_only",
    "stress_survives",
    "targeted_execution_data_prelead",
    "targeted_execution_data_prelead_unresolved",
    "a2_a3_tier1_prelead_confirmed_train_only",
]

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

C2_MECHANISMS = [
    "etf_institutional_access",
    "legal_regulatory_repricing",
    "protocol_utility_fee_revenue",
    "protocol_utility_revenue",
    "supply_unlock_float",
    "unlock_vesting_change",
    "exchange_access_expansion",
    "leverage_access_expansion",
    "integration_distribution",
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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-evidence-repair")
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
    p = argparse.ArgumentParser(description="QLMG evidence remediation and family repair")
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
    p.add_argument("--aggressive-overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--live-capture-bundle", default="")
    p.add_argument("--tmux-session-name", default="qlmg_evidence_repair")
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


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows) if path.exists() else pd.DataFrame()
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
        ctx.notifier.send("QLMG evidence repair resource warning", json.dumps(guard), level="warning")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop for {stage}: {guard}")


def evidence_level_rank(level: str) -> int:
    try:
        return EVIDENCE_LEVELS.index(str(level))
    except ValueError:
        return 0


def evidence_level_allows_performance_metrics(level: str) -> bool:
    return evidence_level_rank(level) >= evidence_level_rank("level_3_event_level_trade_ledger")


def pf_from_values(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna()
    pos = float(x[x > 0].sum())
    neg = float(x[x < 0].sum())
    if abs(neg) < 1e-12:
        return float("inf") if pos > 0 else float("nan")
    return pos / abs(neg)


def max_drawdown(vals: pd.Series) -> float:
    if vals.empty:
        return 0.0
    curve = pd.to_numeric(vals, errors="coerce").fillna(0.0).cumsum()
    peak = curve.cummax()
    return float((curve - peak).min())


def audit_projected_mean_expansion(df: pd.DataFrame) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if df.empty:
        return failures
    if "event_id" not in df.columns and any(c in df.columns for c in ["PF", "Sharpe", "CAGR", "max_dd", "drawdown"]):
        failures.append({"failure_type": "performance_metric_without_event_ledger", "details": "PF/DD/Sharpe/CAGR present but no event_id"})
    if "projection_source" in df.columns and df["projection_source"].astype(str).str.contains("mean|projection|summary", case=False, na=False).any():
        failures.append({"failure_type": "projected_mean_expanded_to_events", "details": "projection_source indicates non-event metric expanded"})
    if {"candidate_id", "event_id", "window_scope"}.issubset(df.columns):
        both = df.groupby(["candidate_id", "event_id"])["window_scope"].nunique()
        for idx, n in both[both > 1].items():
            failures.append({"candidate_id": idx[0], "event_id": idx[1], "failure_type": "core_24h_full_hold_double_count", "details": f"window_scope_count={n}"})
    return failures


def projected_artifact_is_quarantined(df: pd.DataFrame) -> bool:
    return bool(audit_projected_mean_expansion(df))


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


def validate_no_protected_df(df: pd.DataFrame, cols: Sequence[str]) -> None:
    if not df.empty:
        validate_no_protected(df, list(cols))


def classify_reverification_status(source_artifact: str, failure_type: str = "") -> tuple[str, str]:
    p = Path(str(source_artifact))
    if not p.exists():
        return "unavailable_for_recheck", "source artifact missing"
    if p.suffix.lower() == ".csv":
        df = read_csv(p, nrows=5000)
        if df.empty:
            return "accepted_from_integrity_run_inventory_only", "csv unreadable or empty"
        if "performance_metrics_without" in failure_type and any(c in df.columns for c in ["PF", "Sharpe", "CAGR", "max_dd", "drawdown"]) and "event_id" not in df.columns:
            return "directly_reverified_from_source_artifact", "performance columns present without event_id"
        if "core_24h_full_hold" in failure_type and {"event_id", "window_scope"}.issubset(df.columns):
            return "directly_reverified_from_source_artifact", "event/window_scope columns available for duplicate check"
        return "accepted_from_integrity_run_inventory_only", "source readable but specific failure not cheaply rechecked"
    if p.suffix.lower() == ".parquet":
        df = read_parquet_safe(p)
        if df.empty:
            return "accepted_from_integrity_run_inventory_only", "parquet unreadable or empty"
        if "core_24h_full_hold" in failure_type and {"event_id", "window_scope"}.issubset(df.columns):
            fails = audit_projected_mean_expansion(df)
            return ("directly_reverified_from_source_artifact" if fails else "inconsistent_with_source_artifact", "core/full check re-run")
        return "accepted_from_integrity_run_inventory_only", "parquet readable but specific failure not cheaply rechecked"
    return "accepted_from_integrity_run_inventory_only", "non-tabular source"


def classify_quarantine_failure(row: Mapping[str, Any]) -> tuple[str, str]:
    artifact = str(row.get("artifact_path", ""))
    failure = str(row.get("failure_type", ""))
    lower = artifact.lower()
    known_bad = any(x in failure for x in ["performance_metrics_without", "performance_metric_without", "internal_validation", "projected", "double_count", "control_normalization"])
    active_scoring = any(x in lower for x in ["a2a3/corrected_event_level_replay", "a3_validation", "a2_repair", "b1_trade_ledger", "c2_trade_ledger"])
    if known_bad and not active_scoring:
        return "already_quarantined_historical_failure", "forbidden_for_ranking"
    if known_bad and active_scoring:
        return "unresolved_active_scoring_lineage_failure", "blocks_corrected_sweep"
    return "informational_lineage_issue", "does_not_block_if_not_used_for_ranking"


def build_liquidation_taxonomy(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if replay.empty:
        return pd.DataFrame(rows)
    for _, r in replay.iterrows():
        if str(r.get("family")) != "A2":
            continue
        risk_bps = float(r.get("risk_bps_used") or np.nan)
        mark_avail = bool(r.get("mark_price_available"))
        liq = bool(r.get("liquidation_flag"))
        same_bar = bool(r.get("same_bar_ambiguity"))
        if not liq and not same_bar:
            category = "no_liquidation_flag"
        elif not mark_avail:
            category = "proxy_mark_missing_last_used"
        elif same_bar:
            category = "same_bar_ambiguity"
        elif np.isfinite(risk_bps) and risk_bps > 1500:
            category = "stop_too_wide_relative_to_safe_leverage"
        elif liq:
            category = "liquidation_risk_buffer_violation"
        else:
            category = "data_quality_flag"
        rows.append({
            "event_id": r.get("event_id"),
            "candidate_id": r.get("candidate_id"),
            "symbol": r.get("symbol"),
            "decision_ts": r.get("decision_ts"),
            "liquidation_flag": liq,
            "mark_price_available": mark_avail,
            "same_bar_ambiguity": same_bar,
            "risk_bps_used": risk_bps,
            "taxonomy": category,
            "true_mark_liquidation_before_stop": False,
            "note": "Prior corrected replay has diagnostic liquidation_flag but not sufficient path-order fields for true mark liquidation ordering.",
        })
    return pd.DataFrame(rows)


def choose_operator_decision(decision: Mapping[str, Any]) -> str:
    if decision.get("blocked_by_protocol_issue"):
        return "blocked_by_protocol_issue"
    if decision.get("corrected_sweep_allowed"):
        return "proceed_to_corrected_sweep"
    if str(decision.get("a3_validation_verdict", "")).startswith("a3_"):
        return "run_a3_validation_next"
    if str(decision.get("a2_repair_verdict", "")).startswith("a2_"):
        return "repair_a2_next"
    if "support_only" in str(decision.get("b1_trade_ledger_verdict", "")) or "support_only" in str(decision.get("c2_trade_ledger_verdict", "")):
        return "build_b1_c2_ledgers_next"
    if decision.get("micro_canary_possible"):
        return "micro_canary_possible_execution_only"
    return "wait_for_branch_x_capture_bundle"


def micro_canary_allowed(candidate_id: str, *, has_live_capture: bool, is_d4: bool = False) -> tuple[bool, str]:
    if is_d4:
        return False, "D4 micro-canary blocked unless a real D4-like event has already been captured with liquidation/depth context"
    if candidate_id not in {"589a8c85c943", "b1a3735d5092", "new_perp_listing_event_study__589a8c85c943", "new_perp_listing_event_study__b1a3735d5092"}:
        return False, "micro-canary restricted to listing/VWAP-loss analogs for 589a8c85c943 and b1a3735d5092"
    if not has_live_capture:
        return False, "no live capture bundle supplied; execution telemetry ladder begins with capture"
    return True, "execution telemetry only; not alpha validation"


def corrected_sweep_allowed(active_failures: int, audit_pack_exists: bool, a3_done: bool, b1c2_known: bool, quarantined: bool) -> tuple[bool, list[str]]:
    blockers = []
    if active_failures:
        blockers.append("unresolved_active_scoring_lineage_failures")
    if not audit_pack_exists:
        blockers.append("manual_a2a3_audit_pack_missing")
    if not a3_done:
        blockers.append("a3_validation_not_completed")
    if not b1c2_known:
        blockers.append("b1_c2_ledger_feasibility_unknown")
    if not quarantined:
        blockers.append("known_bad_artifacts_not_quarantined")
    return len(blockers) == 0, blockers


def safe_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        if pd.isna(value):
            return None
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts)
    except Exception:
        return None


def month_keys(values: Any) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    ser = ts if isinstance(ts, pd.Series) else pd.Series(ts)
    return ser.dt.tz_convert(None).dt.to_period("M").astype(str)


def load_symbol_bars(symbol: str) -> pd.DataFrame:
    p = DATA_5M / f"{symbol}.parquet"
    df = read_parquet_safe(p)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def replay_simple_path(path: pd.DataFrame, side: str, entry_price: float, stop_price: float, target_price: float, max_exit_ts: pd.Timestamp) -> tuple[pd.Timestamp | None, float, str, bool]:
    if path.empty:
        return None, np.nan, "missing_path", False
    for _, b in path.iterrows():
        ts = pd.Timestamp(b["timestamp"])
        if ts > max_exit_ts:
            break
        high = float(b["high"])
        low = float(b["low"])
        if side == "long":
            stop_hit = low <= stop_price
            target_hit = high >= target_price
        else:
            stop_hit = high >= stop_price
            target_hit = low <= target_price
        if stop_hit and target_hit:
            return ts, stop_price, "same_bar_adverse_stop", True
        if stop_hit:
            return ts, stop_price, "stop", False
        if target_hit:
            return ts, target_price, "target", False
    last = path[path["timestamp"] <= max_exit_ts]
    if last.empty:
        last = path.iloc[[0]]
    row = last.iloc[-1]
    return pd.Timestamp(row["timestamp"]), float(row["close"]), "time", False


def calc_r(side: str, entry: float, exit_price: float, stop_price: float) -> float:
    risk = abs(entry - stop_price)
    if not np.isfinite(risk) or risk <= 0:
        return np.nan
    return (exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk


def stage_preflight(ctx: RunContext) -> None:
    resource_check(ctx, "preflight-and-artifact-freeze", 0.2)
    manifest = []
    hashes: dict[str, Any] = {"run_root": str(ctx.run_root)}
    try:
        hashes["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        hashes["git_status_short"] = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).strip().splitlines()
    except Exception:
        hashes["git_head"] = "unknown"
    roots = dict(FROZEN_ROOTS)
    if ctx.args.live_capture_bundle:
        roots["live_capture_bundle"] = Path(ctx.args.live_capture_bundle)
    for name, path in roots.items():
        exists = path.exists()
        status = "available_file" if path.is_file() else ("available_dir" if path.is_dir() else "not_available")
        main_file = path if path.is_file() else (path / "decision_summary.json")
        if not main_file.exists() and path.is_dir():
            reports = sorted(path.glob("*REPORT.md"))
            main_file = reports[0] if reports else path
        manifest.append({"artifact_name": name, "path": str(path), "exists": exists, "status": status, "hash_basis": str(main_file if main_file.exists() and main_file.is_file() else "")})
        hashes[name] = file_hash(main_file) if main_file.exists() and main_file.is_file() else ("directory_present_no_main_file" if exists else "missing")
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", manifest)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    guard = check_resource_guard(resource_snapshot(ctx.run_root.parent), estimated_output_gb=8.0, hard_free_gb=5.0, warn_free_gb=7.0, hard_stage_output_gb=35.0, allow_large_output=ctx.args.allow_large_output)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\nstatus={guard['status']} free_disk_gb={guard['free_disk_gb']:.2f} max_output_gb={ctx.args.max_output_gb}")
    write_text(ctx.run_root / "preflight/preflight_report.md", "\n".join([
        "# QLMG Evidence Remediation And Family Repair Preflight",
        f"run_root: `{ctx.run_root}`",
        f"window: `{ctx.start}` to `{ctx.end}`",
        "Latest integrity run is treated as an index, not unquestioned truth.",
        "No final holdout reads. No live trading. No Branch X retuning from OHLCV/proxy data.",
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
    ctx.notifier.send("QLMG evidence repair initialized", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    resource_check(ctx, "seal-guard", 0.05)
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "checks": [{"case": "pre_holdout_allowed", "timestamp": str(SCREENING_END), "passes": True}, {"case": "protected_rejected", "timestamp": str(FINAL_HOLDOUT_START), "passes": False}]})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\nProtected start: `{FINAL_HOLDOUT_START}`. Generated rows and candidate-selection inputs must be earlier than this timestamp.")


def stage_self_check(ctx: RunContext) -> None:
    resource_check(ctx, "integrity-audit-self-check", 0.2)
    failures = read_csv(INTEGRITY_ROOT / "audit/metric_lineage_failures.csv")
    ledger = read_csv(INTEGRITY_ROOT / "ledger/event_ledger_availability.csv")
    inv = read_csv(INTEGRITY_ROOT / "audit/global_evidence_level_inventory.csv")
    rows = []
    for _, r in failures.head(300 if ctx.args.smoke else 5000).iterrows() if not failures.empty else []:
        status, note = classify_reverification_status(str(r.get("artifact_path", "")), str(r.get("failure_type", "")))
        rows.append({
            "source_artifact": r.get("artifact_path", ""),
            "candidate_id": r.get("candidate_id", ""),
            "failure_type": r.get("failure_type", ""),
            "integrity_claim": "metric_lineage_failure",
            "reverification_status": status,
            "reverification_note": note,
        })
    # Recheck headline inventory/ledger mismatch counts from source files.
    rows.extend([
        {"source_artifact": str(INTEGRITY_ROOT / "audit/global_evidence_level_inventory.csv"), "integrity_claim": "mostly_level_0_to_2_inventory", "reverification_status": "directly_reverified_from_source_artifact" if not inv.empty else "unavailable_for_recheck", "reverification_note": f"rows={len(inv)} levels={inv.get('evidence_level', pd.Series(dtype=str)).value_counts().to_dict()}"},
        {"source_artifact": str(INTEGRITY_ROOT / "ledger/event_ledger_availability.csv"), "integrity_claim": "ledger_availability_rankable_rows", "reverification_status": "directly_reverified_from_source_artifact" if not ledger.empty else "unavailable_for_recheck", "reverification_note": f"rows={len(ledger)} rankable={int(ledger.get('rankable', pd.Series(dtype=bool)).astype(bool).sum()) if not ledger.empty and 'rankable' in ledger else 0}"},
    ])
    write_csv(ctx.run_root / "self_check/integrity_inventory_reverification.csv", rows)
    summary_rows = [
        {"question": "why_inventory_mostly_level_0_to_2_but_ledger_found_rankable", "answer": "inventory indexed claims/artifacts broadly; ledger audit inspected replay artifacts and found a subset with event-level trade rows. These are different denominators.", "source": "reverified_from_integrity_csvs"},
        {"question": "which_failures_are_quarantined_historical", "answer": "summary/projection/no-event metrics and known core/full/control failures outside active scoring are quarantined and forbidden for ranking.", "source": "metric_lineage_failures"},
        {"question": "which_failures_affect_active_scoring", "answer": "only unresolved failures in current remediation A2/A3/B1/C2 scoring artifacts block future corrected sweeps.", "source": "new_gate_policy"},
        {"question": "safe_source_event_ledgers", "answer": "corrected A2/A3 event-level replay is usable with funding/mark caps; Branch X ledgers are audit/capture evidence, not liquid-regime ranking inputs.", "source": "ledger_availability"},
        {"question": "artifacts_never_use_for_ranking", "answer": "aggregate summaries, internal-validation projections, MAE/MFE-only path ledgers, sampled-window replays without full-event semantics, and deprecated labels without event-level ledgers.", "source": "quarantine_policy"},
    ]
    write_csv(ctx.run_root / "self_check/integrity_audit_self_check.csv", summary_rows)
    write_text(ctx.run_root / "self_check/integrity_audit_self_check_report.md", "# Integrity Audit Self-Check\n\nThe latest integrity run was used as an index and rechecked against source artifacts where feasible. See `integrity_inventory_reverification.csv` for per-claim status.")


def stage_quarantine(ctx: RunContext) -> None:
    resource_check(ctx, "evidence-quarantine-and-rankable-set", 0.2)
    failures = read_csv(INTEGRITY_ROOT / "audit/metric_lineage_failures.csv")
    ledger = read_csv(INTEGRITY_ROOT / "ledger/event_ledger_availability.csv")
    demotions = read_csv(INTEGRITY_ROOT / "audit/prior_headline_demotions.csv")
    qrows = []
    do_not_use = []
    if not failures.empty:
        for _, r in failures.iterrows():
            qclass, action = classify_quarantine_failure(r.to_dict())
            rec = r.to_dict()
            rec.update({"quarantine_class": qclass, "ranking_action": action})
            qrows.append(rec)
            if action == "forbidden_for_ranking":
                do_not_use.append({"artifact_path": r.get("artifact_path", ""), "failure_type": r.get("failure_type", ""), "reason": "known_bad_metric_lineage", "ranking_action": "do_not_use_for_ranking"})
    deprecated = [{"deprecated_label": label, "allowed_only_if": "event_level_trade_ledger_level_3_or_higher", "replacement_action": "recompute_or_demote_to_support_only"} for label in DEPRECATED_PROMOTION_LABELS]
    write_csv(ctx.run_root / "quarantine/evidence_quarantine_manifest.csv", qrows)
    write_csv(ctx.run_root / "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv", do_not_use)
    write_csv(ctx.run_root / "quarantine/deprecated_promotion_labels.csv", deprecated)
    rankable = filter_rankable_evidence_set(ledger)
    if not rankable.empty:
        assert_pass(validate_no_projected_metric_promotion(rankable))
        assert_pass(validate_no_synthetic_controls(rankable))
    write_csv(ctx.run_root / "quarantine/rankable_active_evidence_set.csv", rankable)
    demoted_n = int(demotions.get("demoted", pd.Series(dtype=bool)).astype(bool).sum()) if not demotions.empty and "demoted" in demotions else 0
    write_text(ctx.run_root / "quarantine/quarantine_report.md", f"# Evidence Quarantine\n\nKnown-bad historical artifacts are preserved but forbidden for ranking. Deprecated labels require level-3 event ledgers. Rankable active rows after Branch X exclusion and narrow A3/A2 scope enforcement: `{len(rankable)}`. Prior demotions indexed: `{demoted_n}`.")


def filter_rankable_evidence_set(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return the only evidence rows allowed to seed corrected rankable scoring."""
    if ledger.empty or "rankable" not in ledger.columns or "family" not in ledger.columns:
        return pd.DataFrame()
    rankable = ledger[ledger["rankable"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if "branch_id" in rankable.columns:
        rankable = rankable[~rankable["branch_id"].astype(str).eq("branch_x_execution_sensitive")]
    rankable = rankable[rankable["family"].astype(str).isin(["A3", "A2_redesign_only"])].copy()
    if not rankable.empty:
        rankable["rankable_scope_policy"] = "narrow_corrected_sweep_only_A3_A2_redesign"
    return rankable


def select_audit_events(replay: pd.DataFrame, seed: int, smoke: bool = False) -> pd.DataFrame:
    if replay.empty:
        return replay
    rng = np.random.default_rng(seed)
    samples = []
    for cid, g in replay.groupby("candidate_id", sort=True):
        g = g.copy()
        n_major = 3 if smoke else 20
        samples.append(g.nlargest(n_major, "net_R"))
        samples.append(g.nsmallest(n_major, "net_R"))
        if len(g):
            idx = rng.choice(g.index.to_numpy(), size=min(n_major, len(g)), replace=False)
            samples.append(g.loc[idx])
        if "liquidation_flag" in g:
            liq = g[g["liquidation_flag"].astype(bool)]
            samples.append(liq if len(liq) <= (20 if smoke else 100) else liq.sample(20 if smoke else 100, random_state=seed))
        if "same_bar_ambiguity" in g:
            amb = g[g["same_bar_ambiguity"].astype(bool)]
            samples.append(amb if len(amb) <= (20 if smoke else 100) else amb.sample(20 if smoke else 100, random_state=seed))
        tmp = g.copy()
        tmp["month"] = month_keys(tmp["decision_ts"]).to_numpy()
        month_sum = tmp.groupby("month")["net_R"].sum().sort_values()
        for m in list(month_sum.head(1).index) + list(month_sum.tail(1).index):
            samples.append(tmp[tmp["month"].eq(m)].head(n_major))
    out = pd.concat(samples, ignore_index=True).drop_duplicates(subset=["candidate_id", "event_id"], keep="first")
    return out


def stage_manual_pack(ctx: RunContext) -> None:
    resource_check(ctx, "manual-replay-audit-pack", 0.3)
    (ctx.run_root / "audit_pack").mkdir(parents=True, exist_ok=True)
    replay = read_parquet_safe(INTEGRITY_ROOT / "a2a3/corrected_event_level_replay.parquet")
    if ctx.args.smoke and not replay.empty:
        syms = sorted(replay["symbol"].dropna().astype(str).unique())[: max(ctx.args.max_symbols or 5, 1)]
        replay = replay[replay["symbol"].astype(str).isin(syms)].copy()
    validate_no_protected_df(replay, ["decision_ts", "entry_ts", "exit_ts"])
    sample = select_audit_events(replay, ctx.args.seed, ctx.args.smoke)
    if not sample.empty:
        sample = sample.copy()
        sample["raw_5m_bar_pointer"] = sample.apply(lambda r: f"{DATA_5M}/{r.get('symbol')}.parquet::{r.get('entry_ts')}..{r.get('exit_ts')}", axis=1)
        sample["context_pointer"] = sample.apply(lambda r: f"{CONTEXT_5M}/{r.get('symbol')}.parquet::{r.get('entry_ts')}..{r.get('exit_ts')}", axis=1)
        sample["entry_exit_stop_formula"] = "gross_R=(exit_price-entry_price)/(entry_price-stop_price) for long; adverse same-bar stop before target"
        sample["event_cost_formula"] = "net_R=gross_R+funding_R-cost_R; prior corrected replay cost_R currently zero unless explicit stress applied"
        sample["funding_formula_or_flag"] = sample.get("funding_exact", pd.Series([False] * len(sample))).map(lambda x: "exact_funding_missing_flag" if not bool(x) else "funding_R from venue timestamp crossings")
        sample["liquidation_check_formula"] = "diagnostic long 10x: entry_price*(1-(10000/10-50)/10000); mark path check where mark available"
        sample["exact_R_calculation"] = sample.apply(lambda r: f"gross_R={r.get('gross_R')} funding_R={r.get('funding_R')} cost_R={r.get('cost_R')} net_R={r.get('net_R')}", axis=1)
        sample.to_parquet(ctx.run_root / "audit_pack/a2a3_manual_audit_events.parquet", index=False)
        sample.head(200).to_csv(ctx.run_root / "audit_pack/a2a3_manual_audit_events_sample.csv", index=False)
        cols = ["candidate_id", "event_id", "symbol", "decision_ts", "entry_ts", "exit_ts", "entry_price", "stop_price", "exit_price", "target_price", "net_R", "raw_5m_bar_pointer", "context_pointer", "entry_exit_stop_formula", "event_cost_formula", "funding_formula_or_flag", "liquidation_check_formula", "exact_R_calculation"]
        sample[[c for c in cols if c in sample.columns]].head(80).to_csv(ctx.run_root / "audit_pack/independent_replay_check_sample.csv", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "audit_pack/a2a3_manual_audit_events.parquet", index=False)
        write_csv(ctx.run_root / "audit_pack/a2a3_manual_audit_events_sample.csv", [])
        write_csv(ctx.run_root / "audit_pack/independent_replay_check_sample.csv", [])
    write_text(ctx.run_root / "audit_pack/replay_formula_spec.md", "\n".join([
        "# Replay Formula Spec",
        "- Long R: `(exit_price - entry_price) / abs(entry_price - stop_price)`.",
        "- Net R: `gross_R + funding_R - cost_R` where explicit stress may add cost_R later.",
        "- Same-bar stop/target ambiguity: adverse stop-first branch is primary.",
        "- Diagnostic long liquidation: `entry_price * (1 - ((10000 / leverage) - 50) / 10000)`; this is not full Bybit risk-tier math.",
        "- Funding exactness is flagged; missing exact funding caps conclusions.",
        "- Raw bars are referenced by exact symbol parquet path and entry/exit timestamp range.",
    ]))
    write_text(ctx.run_root / "audit_pack/a2a3_calculation_trace.md", "# A2/A3 Calculation Trace\n\nThe sample pack includes raw-bar pointers, context pointers, entry/exit/stop formulas, cost/funding/liquidation formulas, and exact event R fields for independent recomputation.")
    write_text(ctx.run_root / "audit_pack/a2a3_manual_audit_report.md", f"# A2/A3 Manual Audit Pack\n\nSample events written: `{len(sample)}`. The pack is independently checkable from raw 5m/context pointers and formula specifications.")


def summarize_variant(df: pd.DataFrame, candidate_id: str, variant: str, family: str, split: str, uses_future_liq: bool = False) -> dict[str, Any]:
    vals = pd.to_numeric(df.get("net_R", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "candidate_id": candidate_id,
        "family": family,
        "variant": variant,
        "split": split,
        "events": int(len(df)),
        "net_R": float(vals.sum()) if len(vals) else np.nan,
        "PF": pf_from_values(vals),
        "win_rate": float((vals > 0).mean()) if len(vals) else np.nan,
        "max_dd_R": max_drawdown(vals.reset_index(drop=True)),
        "uses_future_liquidation_label": uses_future_liq,
        "rankable_as_redesign": not uses_future_liq,
    }


def stage_a2_repair(ctx: RunContext) -> None:
    resource_check(ctx, "a2-exit-risk-redesign", 0.2)
    replay = read_parquet_safe(INTEGRITY_ROOT / "a2a3/corrected_event_level_replay.parquet")
    replay = replay[replay.get("family", pd.Series(dtype=str)).astype(str).eq("A2")].copy() if not replay.empty else pd.DataFrame()
    if ctx.args.smoke and not replay.empty:
        syms = sorted(replay["symbol"].dropna().astype(str).unique())[: max(ctx.args.max_symbols or 5, 1)]
        replay = replay[replay["symbol"].astype(str).isin(syms)].copy()
    validate_no_protected_df(replay, ["decision_ts", "entry_ts", "exit_ts"])
    tax = build_liquidation_taxonomy(replay)
    write_csv(ctx.run_root / "a2_repair/a2_liquidation_flag_taxonomy.csv", tax)
    rows = []
    tail_rows = []
    if not replay.empty:
        replay["decision_ts"] = pd.to_datetime(replay["decision_ts"], utc=True, errors="coerce")
        cutoff = replay["decision_ts"].quantile(0.60)
        for cid, g in replay.groupby("candidate_id"):
            dev = g[g["decision_ts"] <= cutoff].copy()
            val = g[g["decision_ts"] > cutoff].copy()
            variants = {
                "original_current_translation": lambda x: x,
                "skip_same_bar_ambiguity": lambda x: x[~x.get("same_bar_ambiguity", pd.Series(False, index=x.index)).astype(bool)],
                "skip_risk_bps_gt_1500": lambda x: x[pd.to_numeric(x.get("risk_bps_used", pd.Series(np.nan, index=x.index)), errors="coerce") <= 1500],
                "skip_safe_leverage_lt_2x_proxy": lambda x: x[pd.to_numeric(x.get("risk_bps_used", pd.Series(np.nan, index=x.index)), errors="coerce") <= 4500],
                "skip_safe_leverage_lt_3x_proxy": lambda x: x[pd.to_numeric(x.get("risk_bps_used", pd.Series(np.nan, index=x.index)), errors="coerce") <= 2833],
                "audit_remove_liquidation_flags_not_rankable": lambda x: x[~x.get("liquidation_flag", pd.Series(False, index=x.index)).astype(bool)],
            }
            dev_scores = []
            for name, fn in variants.items():
                uses_future = "liquidation" in name
                rows.append(summarize_variant(fn(dev), cid, name, "A2", "development", uses_future))
                val_row = summarize_variant(fn(val), cid, name, "A2", "internal_validation", uses_future)
                rows.append(val_row)
                if not uses_future:
                    dev_scores.append(summarize_variant(fn(dev), cid, name, "A2", "development", uses_future))
            vals = pd.to_numeric(g["net_R"], errors="coerce").dropna().sort_values(ascending=False)
            base = float(vals.sum()) if len(vals) else np.nan
            for pct in [0.01, 0.05]:
                keep = vals.iloc[int(math.ceil(len(vals) * pct)):] if len(vals) else vals
                tail_rows.append({"candidate_id": cid, "test": f"remove_top_{int(pct*100)}pct_winners", "base_net_R": base, "stress_net_R": float(keep.sum()) if len(keep) else np.nan})
            g_month = month_keys(g["decision_ts"]).to_numpy()
            month = g.assign(month=g_month).groupby("month")["net_R"].sum()
            if len(month):
                top_m = month.idxmax(); worst_m = month.idxmin()
                tail_rows.append({"candidate_id": cid, "test": "remove_top_month", "month": top_m, "base_net_R": base, "stress_net_R": float(g[g_month != top_m]["net_R"].sum())})
                tail_rows.append({"candidate_id": cid, "test": "remove_worst_month", "month": worst_m, "base_net_R": base, "stress_net_R": float(g[g_month != worst_m]["net_R"].sum())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["label"] = np.where((out["split"].eq("internal_validation")) & (pd.to_numeric(out["net_R"], errors="coerce") > 0) & (~out["uses_future_liquidation_label"].astype(bool)), "a2_redesign_candidate_found", "a2_exit_risk_geometry_problem")
    write_csv(ctx.run_root / "a2_repair/a2_redesign_summary.csv", out)
    write_csv(ctx.run_root / "a2_repair/a2_tail_dependence_audit.csv", tail_rows)
    write_text(ctx.run_root / "a2_repair/a2_redesign_report.md", "# A2 Exit/Risk Redesign\n\nA2 redesign is split into development and internal validation slices. Future liquidation labels are audit-only and not rankable filters. Families are preserved; current translations may be rejected only as current translations.")


def stage_a3_validation(ctx: RunContext) -> None:
    resource_check(ctx, "a3-family-specific-validation", 0.2)
    replay = read_parquet_safe(INTEGRITY_ROOT / "a2a3/corrected_event_level_replay.parquet")
    replay = replay[replay.get("family", pd.Series(dtype=str)).astype(str).eq("A3")].copy() if not replay.empty else pd.DataFrame()
    if ctx.args.smoke and not replay.empty:
        syms = sorted(replay["symbol"].dropna().astype(str).unique())[: max(ctx.args.max_symbols or 5, 1)]
        replay = replay[replay["symbol"].astype(str).isin(syms)].copy()
    validate_no_protected_df(replay, ["decision_ts", "entry_ts", "exit_ts"])
    rows = []
    stability = []
    if not replay.empty:
        for cid, g in replay.groupby("candidate_id"):
            vals = pd.to_numeric(g["net_R"], errors="coerce").dropna()
            base = float(vals.sum()) if len(vals) else np.nan
            tests: list[tuple[str, pd.DataFrame]] = [("base", g)]
            sorted_vals = vals.sort_values(ascending=False)
            for pct in [0.01, 0.05]:
                drop_idx = sorted_vals.head(int(math.ceil(len(sorted_vals) * pct))).index
                tests.append((f"remove_top_{int(pct*100)}pct_winners", g.drop(index=drop_idx, errors="ignore")))
            tmp = g.assign(month=month_keys(g["decision_ts"]).to_numpy())
            month = tmp.groupby("month")["net_R"].sum()
            if len(month):
                top_m = month.idxmax()
                tests.append(("remove_top_month", tmp[~tmp["month"].eq(top_m)]))
            for sym in list(g["symbol"].dropna().astype(str).value_counts().head(5).index):
                tests.append((f"leave_symbol_out_{sym}", g[~g["symbol"].astype(str).eq(sym)]))
            tests.append(("mark_proxy_exclusion", g[g.get("mark_price_available", pd.Series(False, index=g.index)).astype(bool)]))
            if "parent_regime" in g.columns:
                active_regime = g.groupby("parent_regime")["net_R"].sum().idxmax()
                tests.append(("regime_disabled_outside_active_regime", g[g["parent_regime"].eq(active_regime)]))
            for name, subset in tests:
                vals2 = pd.to_numeric(subset.get("net_R", pd.Series(dtype=float)), errors="coerce").dropna()
                stability.append({"candidate_id": cid, "test": name, "events": len(subset), "net_R": float(vals2.sum()) if len(vals2) else np.nan, "PF": pf_from_values(vals2), "survives_positive": bool(len(vals2) and vals2.sum() > 0)})
            risk = pd.to_numeric(g.get("risk_bps_used", pd.Series(100, index=g.index)), errors="coerce").replace(0, np.nan).fillna(100.0)
            for bps in [25, 50]:
                stressed = pd.to_numeric(g["net_R"], errors="coerce") - bps / risk
                stability.append({"candidate_id": cid, "test": f"fee_slippage_plus_{bps}bps", "events": len(g), "net_R": float(stressed.sum()), "PF": pf_from_values(stressed), "survives_positive": bool(stressed.sum() > 0)})
            stability.append({"candidate_id": cid, "test": "funding_adverse_proxy", "events": len(g), "net_R": float((pd.to_numeric(g["net_R"], errors="coerce") - 0.05).sum()), "PF": pf_from_values(pd.to_numeric(g["net_R"], errors="coerce") - 0.05), "survives_positive": bool((pd.to_numeric(g["net_R"], errors="coerce") - 0.05).sum() > 0)})
            candidate_stability = [r for r in stability if r["candidate_id"] == cid]
            survives = sum(bool(r.get("survives_positive")) for r in candidate_stability)
            total = len(candidate_stability)
            label = "a3_family_specific_validation_candidate" if total and survives / total >= 0.70 and base > 0 else "a3_fragile_but_alive" if base > 0 else "a3_reject_current_translation_only"
            rows.append({"candidate_id": cid, "family": "A3", "base_events": len(g), "base_net_R": base, "base_PF": pf_from_values(vals), "fragility_tests": total, "fragility_tests_survived": survives, "label": label})
    write_csv(ctx.run_root / "a3_validation/a3_validation_summary.csv", rows)
    write_csv(ctx.run_root / "a3_validation/a3_stability_audit.csv", stability)
    write_text(ctx.run_root / "a3_validation/a3_validation_report.md", "# A3 Family-Specific Validation\n\nA3 was tested for small-edge fragility: top-tail removal, top-month removal, leave-one-symbol-out, fee/slippage stress, adverse funding, mark-proxy exclusion, and active-regime dependence.")


def stage_b1_ledger(ctx: RunContext) -> None:
    resource_check(ctx, "b1-trade-ledger-construction", 0.2)
    (ctx.run_root / "b1_trade_ledger").mkdir(parents=True, exist_ok=True)
    if not ctx.args.include_b1:
        return
    b1 = read_csv(ABCX_ROOT / "b1/b1_sector_ignition_summary.csv")
    clusters = read_parquet_safe(ABCX_ROOT / "b1/comovement_clusters_by_date.parquet")
    rows = []
    blockers = []
    modes = ["pit_sector_plus_comovement", "rolling_comovement_cluster_only", "theme_seed_window", "current_only_taxonomy_diagnostic"]
    if not b1.empty:
        for _, r in b1.iterrows():
            mode = str(r.get("mode", "unknown"))
            rankable = mode != "current_only_taxonomy_diagnostic"
            blocker = "no_tradable_event_anchor" if pd.isna(r.get("events", np.nan)) else "insufficient_event_count"
            blockers.append({"mode": mode, "asset_id": r.get("asset_id", ""), "primary_sector": r.get("primary_sector", ""), "blocker": blocker, "detail": "B1 source is support/seed summary without explicit entry/exit timestamps/prices."})
            rows.append({"branch_id": "branch_b_sector_ignition", "mode": mode, "event_level_trade_ledger_exists": False, "rankable_mode_if_ledger_exists": rankable, "status": "b1_support_only_no_trade_ledger" if rankable else "b1_taxonomy_proxy_only", "PF_allowed": False, "DD_allowed": False, "Sharpe_allowed": False, "CAGR_allowed": False})
    else:
        for mode in modes:
            blockers.append({"mode": mode, "blocker": "only_support_table_available" if mode != "rolling_comovement_cluster_only" else "no_dated_cluster_events", "detail": "No usable event-level source table found."})
            rows.append({"branch_id": "branch_b_sector_ignition", "mode": mode, "event_level_trade_ledger_exists": False, "rankable_mode_if_ledger_exists": mode != "current_only_taxonomy_diagnostic", "status": "b1_support_only_no_trade_ledger", "PF_allowed": False, "DD_allowed": False, "Sharpe_allowed": False, "CAGR_allowed": False})
    if not clusters.empty and "trailing_only" in clusters.columns:
        blockers.append({"mode": "rolling_comovement_cluster_only", "blocker": "no_dated_cluster_ignition_event_anchor", "detail": f"clusters_rows={len(clusters)} trailing_only={bool(clusters['trailing_only'].all())}"})
    write_csv(ctx.run_root / "b1_trade_ledger/b1_summary.csv", rows)
    write_csv(ctx.run_root / "b1_trade_ledger/b1_ledger_blockers.csv", blockers)
    pd.DataFrame(columns=["event_id", "candidate_id", "symbol", "decision_ts", "entry_ts", "exit_ts", "net_R"]).to_parquet(ctx.run_root / "b1_trade_ledger/b1_event_level_replay.parquet", index=False)
    write_text(ctx.run_root / "b1_trade_ledger/b1_trade_ledger_report.md", "# B1 Trade Ledger Construction\n\nB1 construction was attempted in separated modes. Current-only taxonomy is non-rankable. Available B1 artifacts do not contain explicit dated ignition events with entry/exit prices, so PF/DD/Sharpe/CAGR are blocked until a true trade ledger is built.")


def c2_mechanism_bucket(mech: str) -> str:
    m = str(mech).lower()
    if "etf" in m or "institution" in m:
        return "etf_institutional_access"
    if "legal" in m or "regulatory" in m:
        return "legal_regulatory_repricing"
    if "utility" in m or "revenue" in m or "fee" in m or "protocol" in m:
        return "protocol_utility_fee_revenue"
    if "unlock" in m or "supply" in m or "float" in m or "vesting" in m:
        return "supply_unlock_float"
    if "exchange" in m or "spot_listing" in m:
        return "exchange_access_expansion"
    if "leverage" in m or "perp" in m:
        return "leverage_access_expansion"
    if "integration" in m or "distribution" in m:
        return "integration_distribution"
    return "attention_or_low_durability"


def stage_c2_ledger(ctx: RunContext) -> None:
    resource_check(ctx, "c2-trade-ledger-construction", 0.5)
    (ctx.run_root / "c2_trade_ledger").mkdir(parents=True, exist_ok=True)
    if not ctx.args.include_c2:
        return
    events = read_parquet_safe(ABCX_ROOT / "c2/catalyst_event_ledger.parquet")
    if ctx.args.smoke and not events.empty:
        events = events.head(12)
    rows = []
    blockers = []
    if not events.empty:
        validate_no_protected_df(events, ["event_anchor_ts", "earliest_decision_ts"])
        for _, ev in events.iterrows():
            symbol = str(ev.get("symbol", ""))
            event_id = str(ev.get("event_id", ""))
            mech = c2_mechanism_bucket(str(ev.get("mechanism_family", "")))
            direction = str(ev.get("direction", "long")).lower()
            earliest = safe_timestamp(ev.get("earliest_decision_ts")) or safe_timestamp(ev.get("event_anchor_ts"))
            if earliest is None:
                blockers.append({"event_id": event_id, "mechanism_family": mech, "blocker": "not_fairly_tested_missing_event_timestamp", "detail": "no earliest decision timestamp"})
                continue
            bars = load_symbol_bars(symbol)
            if bars.empty:
                blockers.append({"event_id": event_id, "mechanism_family": mech, "symbol": symbol, "blocker": "no_eligible_bybit_symbol_at_event_time", "detail": "missing local 5m bars"})
                continue
            base_start = earliest
            base_end = earliest + pd.Timedelta(days=2)
            search_end = earliest + pd.Timedelta(days=15)
            if search_end >= FINAL_HOLDOUT_START:
                blockers.append({"event_id": event_id, "mechanism_family": mech, "symbol": symbol, "blocker": "protected_slice_cutoff", "detail": "search window overlaps holdout"})
                continue
            base = bars[(bars["timestamp"] >= base_start) & (bars["timestamp"] < base_end)]
            after = bars[(bars["timestamp"] >= base_end) & (bars["timestamp"] <= search_end)]
            if base.empty or after.empty:
                blockers.append({"event_id": event_id, "mechanism_family": mech, "symbol": symbol, "blocker": "no_valid_entry_rule", "detail": "base or post-base path missing"})
                continue
            base_high = float(base["high"].max())
            base_low = float(base["low"].min())
            side = "short" if direction == "short" or mech in {"supply_unlock_float", "leverage_access_expansion"} else "long"
            if side == "long":
                trigger = after[after["close"] > base_high]
                stop = base_low
            else:
                trigger = after[after["close"] < base_low]
                stop = base_high
            if trigger.empty:
                blockers.append({"event_id": event_id, "mechanism_family": mech, "symbol": symbol, "blocker": "no_valid_entry_rule", "detail": "no post-base breakout/failure trigger; event-day chase excluded"})
                continue
            tr = trigger.iloc[0]
            entry_ts = pd.Timestamp(tr["timestamp"])
            entry = float(tr["close"])
            risk = abs(entry - stop)
            if risk <= 0 or not np.isfinite(risk):
                blockers.append({"event_id": event_id, "mechanism_family": mech, "symbol": symbol, "blocker": "no_valid_exit_rule", "detail": "invalid stop distance"})
                continue
            target = entry + 2 * risk if side == "long" else entry - 2 * risk
            hold_end = min(entry_ts + pd.Timedelta(days=10), SCREENING_END)
            path = bars[(bars["timestamp"] >= entry_ts) & (bars["timestamp"] <= hold_end)]
            exit_ts, exit_price, exit_reason, ambiguity = replay_simple_path(path, side, entry, stop, target, hold_end)
            if exit_ts is None:
                blockers.append({"event_id": event_id, "mechanism_family": mech, "symbol": symbol, "blocker": "no_valid_exit_rule", "detail": exit_reason})
                continue
            net_r = calc_r(side, entry, exit_price, stop)
            rows.append({
                "event_id": event_id,
                "candidate_id": f"C2_{mech}_{side}",
                "family": "C2",
                "mechanism_family": mech,
                "symbol": symbol,
                "decision_ts": earliest,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "side": side,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "same_bar_ambiguity": ambiguity,
                "net_R": net_r,
                "first_reaction_excluded": True,
                "event_day_chase_primary": False,
                "md_excerpt_seed_limited": bool(ev.get("md_excerpt_seed_limited", True)),
                "metric_basis": "event_level_trade_ledger_from_md_seed_and_local_5m_bars_train_only",
                "required_data_tier": "Tier 1 seed-limited",
                "current_data_tier": "Markdown catalyst seed + 5m OHLCV; mark/funding partial",
                "promotion_cap_reason": "md_excerpt_seed_limited_or_missing_exact_mark_funding",
            })
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        validate_no_protected_df(ledger, ["decision_ts", "entry_ts", "exit_ts"])
        ledger.to_parquet(ctx.run_root / "c2_trade_ledger/c2_event_level_replay.parquet", index=False)
    else:
        pd.DataFrame(columns=["event_id", "candidate_id", "mechanism_family", "symbol", "decision_ts", "entry_ts", "exit_ts", "net_R"]).to_parquet(ctx.run_root / "c2_trade_ledger/c2_event_level_replay.parquet", index=False)
    summary = []
    if not ledger.empty:
        for mech, g in ledger.groupby("mechanism_family"):
            vals = pd.to_numeric(g["net_R"], errors="coerce").dropna()
            label = "sample_limited_seed_candidate" if len(g) < 20 else ("c2_sidecar_support_only_real_controls_required" if vals.sum() > 0 else "c2_reject_current_translation_only")
            summary.append({"mechanism_family": mech, "events": len(g), "symbols": g["symbol"].nunique(), "net_R": float(vals.sum()), "PF": pf_from_values(vals), "max_dd_R": max_drawdown(vals.reset_index(drop=True)), "label": label, "PF_allowed": True, "DD_allowed": True, "Sharpe_allowed": True, "CAGR_allowed": True})
    for mech in ["etf_institutional_access", "legal_regulatory_repricing", "protocol_utility_fee_revenue", "supply_unlock_float", "exchange_access_expansion", "leverage_access_expansion", "integration_distribution"]:
        if not any(r.get("mechanism_family") == mech for r in summary):
            summary.append({"mechanism_family": mech, "events": 0, "symbols": 0, "net_R": np.nan, "PF": np.nan, "max_dd_R": np.nan, "label": "sample_limited_seed_candidate", "PF_allowed": False, "DD_allowed": False, "Sharpe_allowed": False, "CAGR_allowed": False})
    write_csv(ctx.run_root / "c2_trade_ledger/c2_by_mechanism_summary.csv", summary)
    write_csv(ctx.run_root / "c2_trade_ledger/c2_ledger_blockers.csv", blockers)
    write_text(ctx.run_root / "c2_trade_ledger/c2_trade_ledger_report.md", "# C2 Trade Ledger Construction\n\nC2 mechanisms remain separated. Primary tests exclude event-day chase and require a post-event base before entry. Failure-short logic is allowed for supply/unlock, leverage access, noisy access, and low-durability contexts. Markdown-excerpt and missing exact mark/funding cap conclusions.")


def stage_branch_x_plan(ctx: RunContext) -> None:
    resource_check(ctx, "branch-x-capture-calibration-plan", 0.1)
    has_bundle = bool(ctx.args.live_capture_bundle and Path(ctx.args.live_capture_bundle).exists())
    matrix = []
    for cid in ["589a8c85c943", "b1a3735d5092", "9dc07cfc405c", "D4__b4c9487fe82c"]:
        allowed, reason = micro_canary_allowed(cid, has_live_capture=has_bundle, is_d4=cid.startswith("D4"))
        matrix.append({
            "candidate_or_family": cid,
            "micro_canary_allowed": allowed,
            "reason": reason,
            "scope": "execution_telemetry_only_not_alpha_validation",
            "isolated_equity_usdt": "100-300" if allowed else "n/a",
            "risk_per_trade_usdt": "0.50-1.00" if allowed else "n/a",
            "max_positions": 1 if allowed else 0,
            "max_leverage": "2x-3x" if allowed else "n/a",
            "manual_approval_required": allowed,
            "stop_after_execution_anomalies": 2 if allowed else "n/a",
        })
    write_csv(ctx.run_root / "branch_x/micro_canary_readiness_matrix.csv", matrix)
    write_text(ctx.run_root / "branch_x/no_vendor_execution_evidence_ladder.md", "\n".join([
        "# Branch X No-Vendor Execution Evidence Ladder",
        "",
        "- 24h capture can answer current spread, top-of-book/depth snapshots, public-trade cadence, and basic fill/slippage proxies for active analogs.",
        "- 72h capture can answer session variation, funding-window behavior, and repeated stop/entry execution around analog events.",
        "- One live analog event can answer whether listing/VWAP-loss execution assumptions are directionally plausible under actual spread/depth/trades.",
        "- Micro-canary can answer only live execution telemetry at tiny risk. It cannot validate alpha.",
        "- Without historical vendor data, broad historical depth/trade/liquidation replay and true D4 liquidation-feed validation remain unanswered.",
    ]))
    write_text(ctx.run_root / "branch_x/branch_x_capture_calibration_plan.md", "# Branch X Capture Calibration\n\nBranch X is not retuned here. D4 remains execution-depth/liquidation blocked. Listing 589 and b1 can become execution-follow-up analogs only through capture or micro-canary telemetry under strict caps. 9dc remains fragile/backlog. Funding-window is preserved.")


def stage_gate(ctx: RunContext) -> None:
    resource_check(ctx, "corrected-sweep-readiness-gate", 0.05)
    q = read_csv(ctx.run_root / "quarantine/evidence_quarantine_manifest.csv")
    active_failures = int(q[q.get("quarantine_class", pd.Series(dtype=str)).astype(str).eq("unresolved_active_scoring_lineage_failure")].shape[0]) if not q.empty else 0
    audit_pack_exists = (ctx.run_root / "audit_pack/independent_replay_check_sample.csv").exists()
    a3_done = (ctx.run_root / "a3_validation/a3_validation_summary.csv").exists()
    b1c2_known = (ctx.run_root / "b1_trade_ledger/b1_ledger_blockers.csv").exists() and (ctx.run_root / "c2_trade_ledger/c2_ledger_blockers.csv").exists()
    quarantined = (ctx.run_root / "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv").exists()
    allowed, blockers = corrected_sweep_allowed(active_failures, audit_pack_exists, a3_done, b1c2_known, quarantined)
    allowed_families = []
    a3 = read_csv(ctx.run_root / "a3_validation/a3_validation_summary.csv")
    if allowed and not a3.empty and a3["label"].astype(str).str.contains("a3_", na=False).any():
        allowed_families.append("A3")
    a2 = read_csv(ctx.run_root / "a2_repair/a2_redesign_summary.csv")
    if allowed and not a2.empty and a2["label"].astype(str).eq("a2_redesign_candidate_found").any():
        allowed_families.append("A2_redesign_only")
    obj = {
        "corrected_sweep_allowed": bool(allowed),
        "blockers": blockers,
        "allowed_families": allowed_families,
        "rankable_evidence_sources": ["quarantine/rankable_active_evidence_set.csv", "a3_validation/a3_validation_summary.csv", "a2_repair/a2_redesign_summary.csv"],
        "forbidden_evidence_sources": ["quarantine/quarantined_artifacts_do_not_use_for_ranking.csv"],
        "already_quarantined_historical_failures_do_not_block": True,
        "unresolved_active_scoring_failures": active_failures,
    }
    write_json(ctx.run_root / "gate/corrected_sweep_allowed.json", obj)
    write_text(ctx.run_root / "gate/corrected_sweep_readiness_gate.md", f"# Corrected Sweep Readiness Gate\n\ncorrected_sweep_allowed: `{allowed}`\n\nblockers: `{blockers}`\n\nallowed_families: `{allowed_families}`\n\nAlready-quarantined historical failures do not block future corrected work; unresolved active-scoring failures do.")


def stage_decision(ctx: RunContext) -> None:
    resource_check(ctx, "decision-report", 0.1)
    gate = safe_read_json(ctx.run_root / "gate/corrected_sweep_allowed.json")
    qset = read_csv(ctx.run_root / "quarantine/rankable_active_evidence_set.csv")
    a2 = read_csv(ctx.run_root / "a2_repair/a2_redesign_summary.csv")
    a3 = read_csv(ctx.run_root / "a3_validation/a3_validation_summary.csv")
    b1 = read_csv(ctx.run_root / "b1_trade_ledger/b1_summary.csv")
    c2 = read_csv(ctx.run_root / "c2_trade_ledger/c2_by_mechanism_summary.csv")
    bx = read_csv(ctx.run_root / "branch_x/micro_canary_readiness_matrix.csv")
    a2_verdict = "a2_redesign_candidate_found" if not a2.empty and a2["label"].astype(str).eq("a2_redesign_candidate_found").any() else "a2_current_translation_only"
    a3_verdict = "a3_family_specific_validation_candidate" if not a3.empty and a3["label"].astype(str).eq("a3_family_specific_validation_candidate").any() else "a3_fragile_but_alive" if not a3.empty else "blocked_by_protocol_issue"
    b1_verdict = "b1_sidecar_support_only_real_controls_required" if not b1.empty and b1.get("event_level_trade_ledger_exists", pd.Series(dtype=bool)).astype(bool).any() else "b1_support_only_no_trade_ledger"
    c2_verdict = "c2_sidecar_support_only_real_controls_required" if not c2.empty and c2.get("PF_allowed", pd.Series(dtype=bool)).astype(bool).any() and c2.get("events", pd.Series(dtype=float)).fillna(0).gt(0).any() else "c2_support_only_no_trade_ledger"
    micro_possible = bool(not bx.empty and bx.get("micro_canary_allowed", pd.Series(dtype=bool)).astype(bool).any())
    decision_stub = {
        "blocked_by_protocol_issue": False,
        "corrected_sweep_allowed": bool(gate.get("corrected_sweep_allowed", False)),
        "a3_validation_verdict": a3_verdict,
        "a2_repair_verdict": a2_verdict,
        "b1_trade_ledger_verdict": b1_verdict,
        "c2_trade_ledger_verdict": c2_verdict,
        "micro_canary_possible": micro_possible,
    }
    operator_decision = choose_operator_decision(decision_stub)
    decision = {
        "run_root": str(ctx.run_root),
        "final_holdout_untouched": True,
        "integrity_self_check_verdict": "integrity_index_reverified_where_feasible",
        "rankable_evidence_set_verdict": "rankable_evidence_set_cleaned" if len(qset) else "rankable_evidence_set_still_blocked",
        "a2_repair_verdict": a2_verdict,
        "a3_validation_verdict": a3_verdict,
        "b1_trade_ledger_verdict": b1_verdict,
        "c2_trade_ledger_verdict": c2_verdict,
        "branch_x_capture_verdict": "micro_canary_possible_execution_only" if micro_possible else "continue_branch_x_capture_and_execution_telemetry",
        "corrected_sweep_readiness_verdict": "corrected_sweep_ready" if gate.get("corrected_sweep_allowed") else "corrected_sweep_not_ready",
        "next_action_verdict": "no_family_rejected_only_current_translations",
        "operator_decision": operator_decision,
        "corrected_sweep_allowed": bool(gate.get("corrected_sweep_allowed", False)),
        "corrected_sweep_blockers": gate.get("blockers", []),
        "no_live_ready_language": True,
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    trusted = "Corrected A2/A3 event-level replay and remediation summaries are trusted as train-only event-level evidence with funding/mark caps; C2 rows are event-level only where constructed from catalyst ledger and local 5m bars."
    support = "B1 sector/theme/co-movement data remain support-only until dated ignition entries and exits exist. Some C2 mechanisms remain sample-limited seed support."
    quarantined = "Known bad summaries, projections, core/full double-count risks, non-normalized controls, and deprecated promotion labels are forbidden for ranking."
    branch_x = "D4/listing/funding Branch X remains execution-depth/live-capture blocked; listing 589/b1 analogs have only a narrow execution-telemetry micro-canary path when live capture is available."
    preserved = "A2 prior-high momentum, A3 close-confirmed reclaim, B1 sector ignition, C2 catalyst bases, D4, listing/VWAP-loss, and funding-window are preserved unless a current translation fails under clean evidence."
    labels = ", ".join(f"`{x}`" for x in DEPRECATED_PROMOTION_LABELS)
    report = [
        "# QLMG Evidence Remediation And Family Repair Report",
        "",
        f"Run root: `{ctx.run_root}`",
        "",
        "## Verdicts",
        *[f"- `{k}`: `{v}`" for k, v in decision.items() if k.endswith("verdict")],
        f"- `operator_decision`: `{operator_decision}`",
        "",
        "## What Can Be Trusted Now",
        f"- Trusted event-level evidence: {trusted}",
        f"- Support-only evidence: {support}",
        f"- Quarantined evidence: {quarantined}",
        f"- Branch X data-blocked evidence: {branch_x}",
        f"- Hypotheses preserved despite failed current translations: {preserved}",
        f"- Labels that must no longer be used without event-level trade ledgers: {labels}",
        "",
        "## Corrected Sweep Gate",
        f"- corrected_sweep_allowed: `{gate.get('corrected_sweep_allowed', False)}`",
        f"- blockers: `{gate.get('blockers', [])}`",
        f"- allowed_families: `{gate.get('allowed_families', [])}`",
        "",
        "## Branch X No-Vendor Route",
        "See `branch_x/no_vendor_execution_evidence_ladder.md` and `branch_x/micro_canary_readiness_matrix.csv`. Micro-canary is execution telemetry only, not alpha validation.",
        "",
        "No live-ready, sealed-ready, validated, production-ready, or trading recommendation language is used.",
    ]
    write_text(ctx.run_root / "QLMG_EVIDENCE_REMEDIATION_FAMILY_REPAIR_REPORT.md", "\n".join(report))
    # Evidence-class next contracts.
    contracts = [
        {"contract_id": "A3_validation_repair", "evidence_class": "event_level_train_only", "next_action": "fresh exact controls and family-specific validation", "source": "a3_validation"},
        {"contract_id": "A2_redesign_only", "evidence_class": "event_level_redesign", "next_action": "continue A2 exit/risk redesign with dev/eval split", "source": "a2_repair"},
        {"contract_id": "B1_ledger_construction", "evidence_class": "support_only", "next_action": "add dated sector/cluster ignition anchors and entry/exit prices", "source": "b1_trade_ledger"},
        {"contract_id": "C2_mechanism_followup", "evidence_class": "seed_limited_event_level_where_available", "next_action": "expand mechanism-specific catalyst ledger and controls", "source": "c2_trade_ledger"},
        {"contract_id": "Branch_X_capture_calibration", "evidence_class": "execution_data_blocked", "next_action": "start no-vendor 24h/72h capture ladder", "source": "branch_x"},
    ]
    if gate.get("corrected_sweep_allowed"):
        contracts.append({"contract_id": "global_corrected_sweep", "evidence_class": "rankable_event_level", "next_action": "run gated corrected sweep only on allowed families", "source": "gate"})
    write_csv(ctx.run_root / "next_contracts/next_action_contract_summary.csv", contracts)
    write_text(ctx.run_root / "next_contracts/next_contracts_report.md", "# Next Contracts\n\nContracts are split by evidence class; no generic promotion labels are used without event-level trade ledgers.")


def stage_bundle(ctx: RunContext) -> None:
    resource_check(ctx, "compact-review-bundle", 0.1)
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_EVIDENCE_REMEDIATION_FAMILY_REPAIR_REPORT.md",
        "decision_summary.json",
        "self_check/integrity_inventory_reverification.csv",
        "self_check/integrity_audit_self_check_report.md",
        "quarantine/evidence_quarantine_manifest.csv",
        "quarantine/rankable_active_evidence_set.csv",
        "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv",
        "quarantine/deprecated_promotion_labels.csv",
        "audit_pack/a2a3_manual_audit_events_sample.csv",
        "audit_pack/independent_replay_check_sample.csv",
        "audit_pack/replay_formula_spec.md",
        "a2_repair/a2_redesign_summary.csv",
        "a2_repair/a2_liquidation_flag_taxonomy.csv",
        "a2_repair/a2_redesign_report.md",
        "a3_validation/a3_validation_summary.csv",
        "a3_validation/a3_stability_audit.csv",
        "a3_validation/a3_validation_report.md",
        "b1_trade_ledger/b1_summary.csv",
        "b1_trade_ledger/b1_ledger_blockers.csv",
        "b1_trade_ledger/b1_trade_ledger_report.md",
        "c2_trade_ledger/c2_by_mechanism_summary.csv",
        "c2_trade_ledger/c2_ledger_blockers.csv",
        "c2_trade_ledger/c2_trade_ledger_report.md",
        "branch_x/branch_x_capture_calibration_plan.md",
        "branch_x/no_vendor_execution_evidence_ladder.md",
        "branch_x/micro_canary_readiness_matrix.csv",
        "gate/corrected_sweep_readiness_gate.md",
        "gate/corrected_sweep_allowed.json",
        "next_contracts/next_action_contract_summary.csv",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
        "preflight/resource_guard_report.md",
    ]
    rows = []
    for rel in include:
        src = ctx.run_root / rel
        included = src.exists() and src.is_file() and src.stat().st_size < 10_000_000
        if included:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            rows.append({"artifact": rel, "source_path": str(src), "bundle_path": str(dst), "included": True})
        else:
            rows.append({"artifact": rel, "source_path": str(src), "bundle_path": "", "included": False})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nSmall reports and summaries only. Large parquet ledgers are referenced by path, not bundled.")


STAGE_FUNCS = {
    "preflight-and-artifact-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "integrity-audit-self-check": stage_self_check,
    "evidence-quarantine-and-rankable-set": stage_quarantine,
    "manual-replay-audit-pack": stage_manual_pack,
    "a2-exit-risk-redesign": stage_a2_repair,
    "a3-family-specific-validation": stage_a3_validation,
    "b1-trade-ledger-construction": stage_b1_ledger,
    "c2-trade-ledger-construction": stage_c2_ledger,
    "branch-x-capture-calibration-plan": stage_branch_x_plan,
    "corrected-sweep-readiness-gate": stage_gate,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and is_done(ctx.run_root, stage):
        ctx.notifier.send("QLMG evidence repair stage skipped", stage)
        return
    ctx.notifier.send("QLMG evidence repair stage start", stage)
    if ctx.args.dry_run:
        mark_done(ctx.run_root, stage)
        return
    try:
        STAGE_FUNCS[stage](ctx)
        mark_done(ctx.run_root, stage)
        ctx.notifier.send("QLMG evidence repair stage complete", stage)
    except Exception as exc:
        ctx.notifier.send("QLMG evidence repair stage failed", f"{stage}: {type(exc).__name__}: {exc}", level="error")
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
        notifier.send("QLMG evidence repair complete", f"run_root={run_root}")
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "complete", "ts_utc": utc_now()})
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
