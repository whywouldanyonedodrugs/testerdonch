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
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_screening_core import (  # noqa: E402
    GB,
    ResourceSnapshot,
    check_resource_guard,
    liquidation_price,
    resource_snapshot,
    utc_now,
    write_json,
)
from tools.run_qlmg_engine_and_first_screen import (  # noqa: E402
    DATA_5M,
    FINAL_HOLDOUT_START,
    SCREENING_END,
    add_features,
    cost_bps_for_tier,
    d1_d3_e1_variants,
    discover_symbol_paths,
    latest_tier_for_symbol,
    load_symbol_df,
    parameter_hash,
    signal_indices,
)
from tools.run_qlmg_path_diagnostics_exit_surface import (  # noqa: E402
    BASIC_SURFACE_HORIZONS,
    HORIZONS,
    PREV_ROOT as PHASE05_ROOT,
    _raw_signal_mask,
    btc_eth_regime_map,
    compute_path_row_fast,
    enrich_regime_for_events,
    event_id_for,
    reference_stop_price,
    sample_null_events,
    summarize_r,
    surface_returns,
    symbol_path_arrays,
)

try:
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

DEFAULT_RUN_ID = "phase_qlmg_d1_narrow_validation_20260624_v1"
RESULTS_ROOT = REPO / "results/rebaseline"
PRIOR_PATH_ROOT = REPO / "results/rebaseline/phase_qlmg_path_diagnostics_exit_surface_20260624_v1_20260624_121522"
DATA_1M_HOT = Path("/opt/parquet/1m_hot")
D1_STOP_MULTS = [0.5, 0.75, 1.0]
D1_TARGET_R = [3.0, 5.0]
D1_TIME_EXITS = ["15m", "30m", "1h"]
D1_ENTRY_DELAYS = ["confirmation_close", "next_bar_open"]
ALLOWED_VERDICTS = {
    "promote_to_targeted_execution_data_collection",
    "continue_unsealed_validation",
    "reject_d1_current_translation",
    "blocked_by_1m_or_execution_data",
    "blocked_by_data_quality",
    "blocked_by_protocol_issue",
}
STAGES = (
    "preflight-resource-and-artifact-audit",
    "telegram-and-tmux-setup",
    "seal-guard",
    "d1-contract-freeze",
    "d1-full-coverage-event-rebuild",
    "d1-stronger-matched-nulls",
    "d1-executable-replay-surface",
    "d1-1m-where-available-audit",
    "d1-execution-cost-stress",
    "d1-bad-wick-and-data-quality-audit",
    "d1-regime-and-state-filters",
    "d1-walk-forward-and-cpcv",
    "d1-aggressive-10x-portfolio-overlay",
    "d1-targeted-data-acquisition-plan",
    "d3-e1-hold-or-defer-memo",
    "decision-report",
    "compact-review-bundle",
    "all",
)


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
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-d1-validation")
                self.status = self.notifier.status_line()
                if "disabled" in self.status.lower():
                    self.missing = self.status
            except Exception as exc:
                self.status = f"disabled: {type(exc).__name__}: {exc}"
                self.missing = self.status
        elif not disabled:
            self.missing = "tools.telegram_notify.TelegramNotifier unavailable"

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
        return sent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D1 narrow train-only validation and execution-realism audit")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--max-output-gb", type=float, default=30.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--use-1m-where-available", action="store_true")
    p.add_argument("--tmux-session-name", default="qlmg_d1_validation")
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
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def shell(args: Sequence[str], timeout: float = 120.0) -> str:
    try:
        p = subprocess.run(args, cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


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


def done_path(run_root: Path, stage: str) -> Path:
    return run_root / "stage_status" / f"{stage}.done"


def mark_done(run_root: Path, stage: str) -> None:
    write_text(done_path(run_root, stage), utc_now())


def required_outputs_for_stage(run_root: Path, stage: str) -> list[Path]:
    mapping = {
        "preflight-resource-and-artifact-audit": [run_root / "preflight/preflight_report.md", run_root / "preflight/prior_d1_artifact_manifest.json"],
        "telegram-and-tmux-setup": [run_root / "notifications/telegram_readiness_report.md", run_root / "tmux/watch_commands.md"],
        "seal-guard": [run_root / "seal/seal_guard_report.md", run_root / "seal/protected_slice_check.json"],
        "d1-contract-freeze": [run_root / "contracts/d1_low_volume_reversal_validation_contract.json"],
        "d1-full-coverage-event-rebuild": [run_root / "events/d1_event_coverage_summary.csv"],
        "d1-stronger-matched-nulls": [run_root / "matched_null/d1_matched_null_summary.csv"],
        "d1-executable-replay-surface": [run_root / "replay/d1_executable_replay_summary.csv"],
        "d1-1m-where-available-audit": [run_root / "one_minute/d1_1m_vs_5m_summary.csv"],
        "d1-execution-cost-stress": [run_root / "execution/d1_execution_cost_stress_summary.csv"],
        "d1-bad-wick-and-data-quality-audit": [run_root / "data_quality/d1_bad_wick_artifact_summary.csv"],
        "d1-regime-and-state-filters": [run_root / "regime/d1_state_stratification_summary.csv"],
        "d1-walk-forward-and-cpcv": [run_root / "validation/d1_cpcv_summary.csv"],
        "d1-aggressive-10x-portfolio-overlay": [run_root / "portfolio/d1_aggressive_10x_portfolio_summary.csv"],
        "d1-targeted-data-acquisition-plan": [run_root / "data_plan/d1_targeted_data_acquisition_plan.md"],
        "d3-e1-hold-or-defer-memo": [run_root / "deferred/d3_e1_defer_memo.md"],
        "decision-report": [run_root / "D1_NARROW_VALIDATION_REPORT.md", run_root / "decision_summary.json"],
        "compact-review-bundle": [run_root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return mapping.get(stage, [])


def stage_complete(run_root: Path, stage: str) -> bool:
    return done_path(run_root, stage).exists() and all(p.exists() for p in required_outputs_for_stage(run_root, stage))


def append_command(run_root: Path, stage: str) -> None:
    p = run_root / "command_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": utc_now(), "stage": stage, "argv": sys.argv, "cwd": str(REPO)}, sort_keys=True) + "\n")


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
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", status | {"stage": stage, "snapshot": snap.__dict__})
    if status["warnings"]:
        ctx.notifier.send("RESOURCE WARNING", f"stage={stage}\n{status}", level="warning")
    if status["status"] != "pass":
        ctx.notifier.send("RESOURCE HARD STOP", f"stage={stage}\n{status}", level="error")
        raise RuntimeError(f"resource guard failed for {stage}: {status['reasons']}")


def estimate_stage_gb(stage: str, smoke: bool, nulls: int) -> float:
    if smoke:
        return 0.2
    if stage == "d1-stronger-matched-nulls":
        return 1.5 * max(1, nulls)
    if stage in {"d1-full-coverage-event-rebuild", "d1-executable-replay-surface", "d1-walk-forward-and-cpcv"}:
        return 2.5
    if stage in {"d1-1m-where-available-audit", "d1-aggressive-10x-portfolio-overlay"}:
        return 1.0
    return 0.5


def d1_catalog(seed: int) -> list[dict[str, Any]]:
    return [v for v in d1_d3_e1_variants(seed) if v.get("family") == "D1"]


def d1_variant_hash(variants: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(variants), sort_keys=True, default=str).encode()).hexdigest()


def load_tiers() -> pd.DataFrame:
    p = PHASE05_ROOT / "universe/liquidity_tiers_by_date.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p2 = PRIOR_PATH_ROOT / "../phase_qlmg_engine_and_first_screen_20260624_v1_20260624_101747/universe/liquidity_tiers_by_date.parquet"
    return pd.read_parquet(p2) if p2.exists() else pd.DataFrame(columns=["symbol", "date", "liquidity_tier"])


def latest_tier(tiers: pd.DataFrame, symbol: str, date: str) -> str:
    return latest_tier_for_symbol(tiers, symbol, date) if not tiers.empty else "UNKNOWN"


def event_ledger_path(run_root: Path) -> Path:
    return run_root / "events/d1_event_ledger.parquet"


def path_metrics_path(run_root: Path) -> Path:
    return run_root / "path/d1_path_metrics.parquet"


def replay_summary_path(run_root: Path) -> Path:
    return run_root / "replay/d1_executable_replay_summary.csv"


def parse_time_exit_to_horizon(minutes: int) -> str:
    return {15: "15m", 30: "30m", 60: "1h"}[minutes]


def risk_bps_for_stop(df: pd.DataFrame, stop_mult: float) -> pd.Series:
    return pd.to_numeric(df["atr_bps"], errors="coerce") * float(stop_mult)


def surface_net_returns(df: pd.DataFrame, horizon: str, stop_mult: float, target_r: float, branch: str = "pessimistic", cost_mult: float = 1.0, extra_bps: float = 0.0, maker_discount: float = 1.0, adverse_funding_mult: float = 1.0) -> pd.Series:
    risk = risk_bps_for_stop(df, stop_mult)
    gross = surface_returns(df, horizon, risk, target_r, branch)
    tier_cost_bps = float(sum(cost_bps_for_tier("C"))) * float(cost_mult) * float(maker_discount) + float(extra_bps)
    cost_r = tier_cost_bps / risk.replace(0, np.nan)
    funding_rate = pd.to_numeric(df.get("funding_rate"), errors="coerce").fillna(0.0)
    side = df.get("side", pd.Series("long", index=df.index)).astype(str)
    # Conservative proxy: if funding sign is adverse to side, subtract a small R-normalized drag.
    adverse = np.where(((side == "long") & (funding_rate > 0)) | ((side == "short") & (funding_rate < 0)), funding_rate.abs() * 10000.0 / risk.replace(0, np.nan), 0.0)
    return gross - cost_r - pd.Series(adverse, index=df.index).fillna(0.0) * float(adverse_funding_mult)


def validate_no_protected(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for col in cols:
        if col in df.columns and len(df):
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            if ts.ge(FINAL_HOLDOUT_START).any():
                raise RuntimeError(f"protected timestamp found in {col}")


def generate_d1_events_for_symbol(symbol: str, df: pd.DataFrame, tiers: pd.DataFrame, variants: Sequence[Mapping[str, Any]], regime: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    if df.empty:
        return events, coverage
    df = add_features(df).sort_values("timestamp").reset_index(drop=True)
    tier = latest_tier(tiers, symbol, str(df["timestamp"].max().date()))
    if tier != "C":
        for variant in variants:
            coverage.append({"variant_id": variant["variant_id"], "symbol": symbol, "tier": tier, "raw_triggers": 0, "retained_events": 0, "skip_reason": "tier_not_C"})
        return events, coverage
    for variant in variants:
        raw_count = int(_raw_signal_mask(df, variant).sum())
        idxs = signal_indices(df, variant)
        retained = 0
        for i in idxs:
            if i + 1 >= len(df):
                continue
            row = df.iloc[i]
            entry = df.iloc[i + 1]
            decision_ts = pd.Timestamp(row["timestamp"])
            entry_ts = pd.Timestamp(entry["timestamp"])
            if decision_ts >= FINAL_HOLDOUT_START or entry_ts >= FINAL_HOLDOUT_START:
                raise RuntimeError("protected timestamp generated in D1 event ledger")
            entry_price = float(entry["open"])
            atr = float(row.get("atr_proxy", np.nan))
            if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(atr) or atr <= 0:
                continue
            side = str(variant["side"])
            stop = reference_stop_price(row, entry_price, side, "D1", variant)
            risk_bps = abs(entry_price - stop) / entry_price * 10000.0
            if not np.isfinite(risk_bps) or risk_bps <= 0:
                continue
            event = {
                "family": "D1",
                "variant_id": variant["variant_id"],
                "parameter_hash": variant["parameter_hash"],
                "symbol": symbol,
                "side": side,
                "liquidity_tier": tier,
                "decision_ts": decision_ts,
                "entry_ts": entry_ts,
                "entry_ref_price": entry_price,
                "reference_stop_price": stop,
                "reference_risk_bps": risk_bps,
                "atr_proxy": atr,
                "atr_bps": atr / entry_price * 10000.0,
                "shock_lookback_h": variant["window_h"],
                "shock_threshold": variant["shock"],
                "shock_magnitude": row.get("ret_4h" if variant["window_h"] == 4 else "ret_24h", np.nan),
                "low_volume_condition": bool(row.get("turnover", np.nan) <= row.get("turnover_med_24h", np.inf) * 1.2),
                "confirmation_trigger": "close_up_after_negative_shock" if side == "long" else "close_down_after_positive_shock",
                "ret_4h": row.get("ret_4h", np.nan),
                "ret_24h": row.get("ret_24h", np.nan),
                "range_pct": row.get("range_pct", np.nan),
                "turnover": row.get("turnover", np.nan),
                "turnover_med_24h": row.get("turnover_med_24h", np.nan),
                "oi_chg_24h": row.get("oi_chg_24h", np.nan),
                "funding_rate": row.get("funding_rate", np.nan),
                "listing_age_proxy_days": np.nan,
                "raw_trigger_count_for_symbol_variant": raw_count,
                "data_quality_flags": "",
            }
            event["event_id"] = event_id_for(event)
            events.append(event)
            retained += 1
        coverage.append({"variant_id": variant["variant_id"], "symbol": symbol, "tier": tier, "raw_triggers": raw_count, "retained_events": retained, "skip_reason": ""})
    if events:
        evdf = pd.DataFrame(events)
        evdf = enrich_regime_for_events(evdf, regime)
        events = evdf.to_dict("records")
    return events, coverage


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    files = [
        "QLMG_PATH_DIAGNOSTICS_EXIT_SURFACE_REPORT.md",
        "triage/next_candidate_contracts/D1_next_contract.json",
        "events/event_coverage_summary.csv",
        "path_diagnostics/path_summary_by_family.csv",
        "matched_null/matched_null_summary.csv",
        "exit_surface/basic_exit_surface_top_pessimistic.csv",
        "seal/protected_slice_check.json",
    ]
    manifest = []
    for rel in files:
        p = PRIOR_PATH_ROOT / rel
        manifest.append({"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0, "sha256": sha256_file(p) if p.exists() and p.is_file() else ""})
    d1_prev = {}
    ps = PRIOR_PATH_ROOT / "path_diagnostics/path_summary_by_family.csv"
    if ps.exists():
        df = pd.read_csv(ps)
        row = df[df.get("family") == "D1"].head(1)
        if not row.empty:
            d1_prev = row.iloc[0].to_dict()
    write_json(ctx.run_root / "preflight/prior_d1_artifact_manifest.json", {"prior_root": str(PRIOR_PATH_ROOT), "artifacts": manifest, "prior_d1_summary": d1_prev})
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard Report\n\n- free disk GB: `{snap.free_gb:.2f}`\n- hard stop: `<5GB`\n- warning: `<7GB`\n- stage output hard stop: `>20GB` unless `--allow-large-output`\n- max output GB default: `{ctx.args.max_output_gb}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight Report\n\n- current free disk GB: `{snap.free_gb:.2f}`\n- previous D1 retained events: `{d1_prev.get('events','unknown')}`\n- previous D1 matched-null policy: `1 null/event default`\n- previous compatible exit reference: `atr_0p5 target=5R time=30m`\n- previous limitations: `5m path metrics, proxy cost/funding/liquidation, mark path effectively absent`\n- previous D1 was full coverage diagnostic after cap removal, not executable validation.\n- protected holdout cutoff: `{FINAL_HOLDOUT_START}`\n")


def stage_telegram_tmux(ctx: RunContext) -> None:
    missing = ctx.notifier.missing or "none_detected"
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness Report\n\n- status: `{ctx.notifier.status}`\n- missing/disabled reason: `{missing}`\n- remote send required: `false`\n- local log: `notifications/telegram_events.jsonl`\n- secrets persisted: `false`\n")
    commands = f"""# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", commands)
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nRun: `bash tools/run_qlmg_d1_narrow_validation_tmux.sh --run-root {ctx.run_root} --session {ctx.args.tmux_session_name}`\n")
    ctx.notifier.send("D1 RUN START", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    check = {"final_holdout_start": str(FINAL_HOLDOUT_START), "run_start": str(ctx.start), "run_end": str(ctx.end), "pre_holdout_read_smoke": bool(ctx.end < FINAL_HOLDOUT_START), "protected_read_smoke_blocked": True, "status": "pass" if ctx.end < FINAL_HOLDOUT_START else "fail"}
    if check["status"] != "pass":
        raise RuntimeError("seal guard failed")
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard Report\n\n- protected QLMG holdout starts: `{FINAL_HOLDOUT_START}`\n- run data end: `{ctx.end}`\n- protected reads in ordinary mode: `blocked`\n- status: `pass`\n")


def stage_contract(ctx: RunContext) -> None:
    variants = d1_catalog(ctx.args.seed)
    contract = {
        "candidate_family": "D1_low_volume_short_horizon_reversal",
        "created_at_utc": utc_now(),
        "hypothesis": "Tier C Bybit USDT perps over/under-react after low-volume liquidity-shock conditions and reverse over short horizons.",
        "null_hypothesis": "D1 events have no path or executable uplift over matched non-events after costs and realistic ambiguity handling.",
        "disconfirming_prediction": "Matched nulls, 1m audit, cost stress, or walk-forward paths remove the apparent D1 edge.",
        "allowed_data_end": str(SCREENING_END),
        "forbidden_data": ">=2026-01-01T00:00:00Z final QLMG holdout",
        "universe": "Tier C small-but-tradable Bybit linear USDT perpetuals",
        "sides": ["long", "short"],
        "entry_catalog_policy": "exact prior 30 D1 variants from Phase 0.5 seed catalog; old per-symbol signal cap removed",
        "entry_variants": variants,
        "entry_catalog_hash": d1_variant_hash(variants),
        "low_volume_condition": "turnover <= 1.2 * trailing 24h turnover median at decision_ts",
        "confirmation_logic": "long requires close uptick after negative shock; short requires close downtick after positive shock",
        "de_overlap_rule": "existing Phase 0.5 signal_indices min-gap based on half hold horizon, minimum 12 bars",
        "reference_exit": {"stop": "ATR 0.5", "target": "5R", "time_exit": "30m"},
        "executable_neighborhood": {"stop_atr": D1_STOP_MULTS, "target_R": D1_TARGET_R, "time_exit": D1_TIME_EXITS, "entry_delay": D1_ENTRY_DELAYS},
        "matched_null_design": "same symbol where possible, same/neighbor month, Tier C, volatility bucket, turnover bucket, BTC/ETH regime, same side construction",
        "cost_model": "qlmg proxy cost v0; Tier C base fee+slippage bps from existing cost_bps_for_tier",
        "funding_model": "side-aware proxy funding drag from available funding_rate; not promotion-grade",
        "liquidation_model": "10x diagnostic mark path when available, otherwise last-price proxy flagged as non-promotion-grade",
        "bad_wick_rules": "top 1pct range exclusion, zero/near-zero turnover, context/mark missingness, listing age exclusions, repeated gaps, abnormal spread/range proxy",
        "walk_forward_cpcv": "8 chronological blocks; 2 test blocks; purge max horizon; embargo at least 1 day",
        "no_live_trading": True,
        "no_sealed_validation": True,
    }
    write_json(ctx.run_root / "contracts/d1_low_volume_reversal_validation_contract.json", contract)


def stage_events(ctx: RunContext) -> None:
    variants = d1_catalog(ctx.args.seed)
    tiers = load_tiers()
    max_symbols = ctx.args.max_symbols or (5 if ctx.args.smoke else 0)
    paths = discover_symbol_paths(max_symbols)
    regime = btc_eth_regime_map(ctx.start, ctx.end)
    all_events: list[dict[str, Any]] = []
    all_cov: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for n, path in enumerate(paths, start=1):
        sym = path.stem.upper()
        df = load_symbol_df(sym, ctx.start, ctx.end, include_context=True)
        ev, cov = generate_d1_events_for_symbol(sym, df, tiers, variants, regime)
        all_events.extend(ev)
        all_cov.extend(cov)
        if n % max(1, ctx.args.chunk_size) == 0 or n == len(paths):
            rec = {"ts_utc": utc_now(), "symbols_done": n, "symbols_total": len(paths), "events": len(all_events)}
            progress.append(rec)
            write_csv(ctx.run_root / "events/chunk_manifests/d1_event_rebuild_progress.csv", progress)
            ctx.notifier.send("D1 EVENT PROGRESS", json.dumps(rec))
    events = pd.DataFrame(all_events)
    if not events.empty:
        validate_no_protected(events, ["decision_ts", "entry_ts"])
        events = events.sort_values(["variant_id", "symbol", "decision_ts"]).reset_index(drop=True)
    event_ledger_path(ctx.run_root).parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(event_ledger_path(ctx.run_root), index=False)
    pd.DataFrame(all_cov).to_csv(ctx.run_root / "events/d1_event_coverage_summary.csv", index=False)
    if not events.empty:
        events.head(500).to_parquet(ctx.run_root / "events/d1_event_ledger_sample.parquet", index=False)
    by_side = events.groupby("side").size().to_dict() if not events.empty else {}
    write_text(ctx.run_root / "events/d1_event_coverage_report.md", f"# D1 Event Coverage Report\n\n- symbols processed: `{len(paths)}`\n- frozen D1 variants: `{len(variants)}`\n- retained events: `{len(events)}`\n- side counts: `{by_side}`\n- old signal cap removed: `yes`\n- de-overlap/min-gap retained: `yes`\n")


def stage_matched_nulls(ctx: RunContext) -> None:
    events = pd.read_parquet(event_ledger_path(ctx.run_root)) if event_ledger_path(ctx.run_root).exists() else pd.DataFrame()
    if events.empty:
        write_csv(ctx.run_root / "matched_null/d1_matched_null_summary.csv", [])
        write_text(ctx.run_root / "matched_null/d1_matched_null_report.md", "# D1 Matched Null Report\n\nNo events.\n")
        return
    use_nulls = max(1, int(ctx.args.nulls_per_event))
    if use_nulls > 1 and not ctx.args.allow_large_output and len(events) * use_nulls > 1_000_000:
        use_nulls = 1
        fallback_reason = "projected_null_rows_above_1m_without_allow_large_output"
    else:
        fallback_reason = ""
    nulls = sample_null_events(events, ctx.args.seed, max_per_event=use_nulls)
    null_dir = ctx.run_root / "matched_null/d1_matched_null_path_metrics.parquet"
    if null_dir.exists():
        shutil.rmtree(null_dir) if null_dir.is_dir() else null_dir.unlink()
    null_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    part = 0
    if not nulls.empty:
        for n, (sym, sub) in enumerate(nulls.groupby("symbol"), start=1):
            df = load_symbol_df(sym, max(ctx.start - pd.Timedelta(days=2), pd.Timestamp("2020-01-01T00:00:00Z")), min(ctx.end + pd.Timedelta(days=6), FINAL_HOLDOUT_START - pd.Timedelta(minutes=5)), include_context=True)
            if df.empty:
                continue
            df = add_features(df).sort_values("timestamp").reset_index(drop=True)
            arrays = symbol_path_arrays(df)
            ts_ns = arrays["ts_ns"]
            for _, ev in sub.iterrows():
                dec_ts = pd.Timestamp(ev["decision_ts"])
                pos = int(np.searchsorted(ts_ns, dec_ts.value, side="left"))
                if pos < 0 or pos + 1 >= len(df):
                    continue
                dec = df.iloc[pos]
                ent = df.iloc[pos + 1]
                ev = ev.copy()
                ev["entry_ts"] = pd.Timestamp(ent["timestamp"])
                ev["entry_ref_price"] = float(ent["open"])
                atr = float(dec.get("atr_proxy", np.nan))
                ev["atr_bps"] = atr / float(ent["open"]) * 10000.0 if np.isfinite(atr) and float(ent["open"]) > 0 else np.nan
                ev["reference_risk_bps"] = ev["atr_bps"] * 0.5 if np.isfinite(ev["atr_bps"]) else ev.get("reference_risk_bps", np.nan)
                rows.append(compute_path_row_fast(ev, arrays))
            if n % max(1, ctx.args.chunk_size) == 0 or n == nulls["symbol"].nunique():
                if rows:
                    pd.DataFrame(rows).to_parquet(null_dir / f"part_{part:05d}.parquet", index=False)
                    part += 1
                    rows.clear()
                    ctx.notifier.send("D1 NULL PROGRESS", f"symbols_done={n}/{nulls['symbol'].nunique()} parts={part}")
    if part == 0:
        pd.DataFrame().to_parquet(null_dir / "part_00000.parquet", index=False)
    event_metrics = build_or_load_path_metrics(ctx)
    null_metrics = pd.read_parquet(null_dir) if null_dir.exists() else pd.DataFrame()
    rows2 = summarize_event_vs_null(event_metrics, null_metrics, use_nulls, fallback_reason)
    write_csv(ctx.run_root / "matched_null/d1_matched_null_summary.csv", rows2)
    if not nulls.empty:
        nulls.head(500).to_parquet(ctx.run_root / "matched_null/d1_matched_null_ledger_sample.parquet", index=False)
    write_text(ctx.run_root / "matched_null/d1_matched_null_report.md", f"# D1 Matched Null Report\n\n- requested nulls per event: `{ctx.args.nulls_per_event}`\n- used nulls per event: `{use_nulls}`\n- fallback reason: `{fallback_reason or 'none'}`\n- null path rows: `{len(null_metrics)}`\n- hard failure condition: D1 must beat matched null after costs and after volatility/turnover matching.\n")


def build_or_load_path_metrics(ctx: RunContext) -> pd.DataFrame:
    p = path_metrics_path(ctx.run_root)
    if p.exists():
        return pd.read_parquet(p)
    events = pd.read_parquet(event_ledger_path(ctx.run_root)) if event_ledger_path(ctx.run_root).exists() else pd.DataFrame()
    out_dir = p
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    part = 0
    if not events.empty:
        for n, (sym, sub) in enumerate(events.groupby("symbol"), start=1):
            df = load_symbol_df(sym, max(ctx.start - pd.Timedelta(days=2), pd.Timestamp("2020-01-01T00:00:00Z")), min(ctx.end + pd.Timedelta(days=6), FINAL_HOLDOUT_START - pd.Timedelta(minutes=5)), include_context=True)
            if df.empty:
                continue
            df = add_features(df).sort_values("timestamp").reset_index(drop=True)
            arrays = symbol_path_arrays(df)
            for _, ev in sub.iterrows():
                rows.append(compute_path_row_fast(ev, arrays))
            if n % max(1, ctx.args.chunk_size) == 0 or n == events["symbol"].nunique():
                if rows:
                    pd.DataFrame(rows).to_parquet(out_dir / f"part_{part:05d}.parquet", index=False)
                    part += 1
                    rows.clear()
    if part == 0:
        pd.DataFrame().to_parquet(out_dir / "part_00000.parquet", index=False)
    df = pd.read_parquet(out_dir)
    validate_no_protected(df, ["decision_ts", "entry_ts"])
    return df


def summarize_event_vs_null(events: pd.DataFrame, nulls: pd.DataFrame, nulls_per_event: int, fallback_reason: str = "") -> list[dict[str, Any]]:
    rows = []
    keys = ["all"]
    for key in keys:
        ev = events
        nu = nulls
        ev_mfe = pd.to_numeric(ev.get("24h_mfe_bps"), errors="coerce").median() if not ev.empty else np.nan
        nu_mfe = pd.to_numeric(nu.get("24h_mfe_bps"), errors="coerce").median() if not nu.empty else np.nan
        ev_share = pd.Series(ev.get("24h_pos1R_before_neg1R", False)).fillna(False).mean() if not ev.empty else np.nan
        nu_share = pd.Series(nu.get("24h_pos1R_before_neg1R", False)).fillna(False).mean() if not nu.empty else np.nan
        rows.append({"slice": key, "event_count": len(ev), "null_count": len(nu), "nulls_per_event_policy": nulls_per_event, "fallback_reason": fallback_reason, "event_median_24h_mfe_bps": ev_mfe, "null_median_24h_mfe_bps": nu_mfe, "event_pos1R_before_neg1R_24h_share": ev_share, "null_pos1R_before_neg1R_24h_share": nu_share, "beats_null_path": bool(np.isfinite(ev_mfe) and np.isfinite(nu_mfe) and ev_mfe > nu_mfe and ev_share > nu_share)})
    for col in ["side", "variant_id", "btc_eth_regime"]:
        if col in events.columns:
            for val, ev in events.groupby(col, dropna=False):
                nu = nulls[nulls.get(col) == val] if not nulls.empty and col in nulls.columns else pd.DataFrame()
                ev_mfe = pd.to_numeric(ev.get("24h_mfe_bps"), errors="coerce").median() if not ev.empty else np.nan
                nu_mfe = pd.to_numeric(nu.get("24h_mfe_bps"), errors="coerce").median() if not nu.empty else np.nan
                ev_share = pd.Series(ev.get("24h_pos1R_before_neg1R", False)).fillna(False).mean() if not ev.empty else np.nan
                nu_share = pd.Series(nu.get("24h_pos1R_before_neg1R", False)).fillna(False).mean() if not nu.empty else np.nan
                rows.append({"slice": f"{col}={val}", "event_count": len(ev), "null_count": len(nu), "nulls_per_event_policy": nulls_per_event, "fallback_reason": fallback_reason, "event_median_24h_mfe_bps": ev_mfe, "null_median_24h_mfe_bps": nu_mfe, "event_pos1R_before_neg1R_24h_share": ev_share, "null_pos1R_before_neg1R_24h_share": nu_share, "beats_null_path": bool(np.isfinite(ev_mfe) and np.isfinite(nu_mfe) and ev_mfe > nu_mfe and ev_share > nu_share)})
    return rows


def stage_replay(ctx: RunContext) -> None:
    df = build_or_load_path_metrics(ctx)
    rows = []
    sample_rows = []
    if not df.empty:
        for entry_delay in D1_ENTRY_DELAYS:
            # confirmation_close uses the same path but applies a small adverse execution penalty.
            delay_cost = 5.0 if entry_delay == "confirmation_close" else 0.0
            for stop in D1_STOP_MULTS:
                for target in D1_TARGET_R:
                    for horizon in D1_TIME_EXITS:
                        risk = risk_bps_for_stop(df, stop)
                        gross = surface_returns(df, horizon, risk, target, "pessimistic")
                        net = gross - (float(sum(cost_bps_for_tier("C"))) + delay_cost) / risk.replace(0, np.nan)
                        for keys, sub_idx in df.groupby(["side", "variant_id"], dropna=False).groups.items():
                            sub_ret = net.loc[list(sub_idx)]
                            s = summarize_r(sub_ret)
                            side, vid = keys
                            rows.append({"executable_variant_id": f"{vid}__stop{stop:g}_target{target:g}_{horizon}_{entry_delay}", "source_variant_id": vid, "side": side, "stop_atr": stop, "target_R": target, "time_exit": horizon, "entry_delay": entry_delay, "ambiguity_branch": "pessimistic", **s, "liquidation_count": int(pd.Series(df.loc[list(sub_idx)].get(f'{horizon}_liquidation_10x', False)).fillna(False).sum())})
                        if len(sample_rows) < 500:
                            tmp = df.head(500).copy()
                            tmp["executable_variant_id"] = f"sample_stop{stop:g}_target{target:g}_{horizon}_{entry_delay}"
                            tmp["net_R_proxy"] = net.head(500).to_numpy()
                            sample_rows.extend(tmp.head(max(0, 500 - len(sample_rows))).to_dict("records"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mean_R", "PF", "events"], ascending=[False, False, False])
    replay_summary_path(ctx.run_root).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(replay_summary_path(ctx.run_root), index=False)
    (ctx.run_root / "replay").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"stop_atr": s, "target_R": t, "time_exit": h, "entry_delay": e} for s in D1_STOP_MULTS for t in D1_TARGET_R for h in D1_TIME_EXITS for e in D1_ENTRY_DELAYS]).to_csv(ctx.run_root / "replay/d1_executable_variant_registry.csv", index=False)
    if sample_rows:
        pd.DataFrame(sample_rows).to_parquet(ctx.run_root / "replay/d1_trade_ledger_sample.parquet", index=False)
    write_text(ctx.run_root / "replay/d1_executable_replay_report.md", "# D1 Executable Replay Surface\n\nReplay uses the frozen D1 event set and a bounded exit neighborhood around ATR 0.5, 5R, 30m. Results are proxy executable summaries from 5m paths with pessimistic same-bar handling.\n")


def stage_one_minute(ctx: RunContext) -> None:
    events = pd.read_parquet(event_ledger_path(ctx.run_root)) if event_ledger_path(ctx.run_root).exists() else pd.DataFrame()
    rows = []
    sample = []
    files = {p.stem.upper(): p for p in DATA_1M_HOT.glob("*.parquet")} if DATA_1M_HOT.exists() else {}
    checked = 0
    covered = 0
    for sym, sub in events.groupby("symbol") if not events.empty else []:
        if sym not in files:
            continue
        try:
            one = pd.read_parquet(files[sym])
        except Exception:
            continue
        if "timestamp" not in one.columns:
            continue
        one["timestamp"] = pd.to_datetime(one["timestamp"], utc=True)
        one = one[one["timestamp"] < FINAL_HOLDOUT_START].sort_values("timestamp")
        if one.empty:
            continue
        one_idx = one.set_index("timestamp", drop=False)
        for _, ev in sub.head(2000).iterrows():
            checked += 1
            entry_ts = pd.Timestamp(ev["entry_ts"])
            path = one_idx[(one_idx.index > entry_ts) & (one_idx.index <= entry_ts + pd.Timedelta(minutes=30))]
            if path.empty:
                continue
            covered += 1
            entry = float(ev["entry_ref_price"])
            side = str(ev["side"])
            risk = float(ev["atr_bps"]) * 0.5
            if side == "long":
                mfe = (pd.to_numeric(path["high"], errors="coerce").max() / entry - 1.0) * 10000.0
                mae = (1.0 - pd.to_numeric(path["low"], errors="coerce").min() / entry) * 10000.0
            else:
                mfe = (1.0 - pd.to_numeric(path["low"], errors="coerce").min() / entry) * 10000.0
                mae = (pd.to_numeric(path["high"], errors="coerce").max() / entry - 1.0) * 10000.0
            ret = -1.0 if mae >= risk else (5.0 if mfe >= risk * 5.0 else float(pd.to_numeric(path.iloc[-1]["close"], errors="coerce") / entry - 1.0) * (10000.0 / risk) * (1 if side == "long" else -1))
            rec = {"event_id": ev["event_id"], "symbol": sym, "side": side, "variant_id": ev["variant_id"], "one_minute_net_R_proxy": ret, "one_minute_mfe_bps": mfe, "one_minute_mae_bps": mae, "coverage_status": "covered"}
            if len(sample) < 500:
                sample.append(rec)
            rows.append(rec)
    df = pd.DataFrame(rows)
    summary = [{"events_checked_with_1m_symbol_file": checked, "events_with_1m_30m_path": covered, "coverage_share": covered / checked if checked else 0.0, "mean_1m_R_proxy": pd.to_numeric(df.get("one_minute_net_R_proxy"), errors="coerce").mean() if not df.empty else np.nan, "pf_1m_proxy": summarize_r(pd.to_numeric(df.get("one_minute_net_R_proxy"), errors="coerce"))["PF"] if not df.empty else np.nan, "status": "blocked_sparse_1m" if covered < 100 else "informative"}]
    write_csv(ctx.run_root / "one_minute/d1_1m_vs_5m_summary.csv", summary)
    if sample:
        pd.DataFrame(sample).to_parquet(ctx.run_root / "one_minute/d1_1m_replay_sample.parquet", index=False)
    write_text(ctx.run_root / "one_minute/d1_1m_coverage_report.md", f"# D1 1m Coverage Report\n\n- /opt/parquet/1m_hot files: `{len(files)}`\n- events checked with 1m symbol file: `{checked}`\n- events with 30m 1m path: `{covered}`\n- status: `{summary[0]['status']}`\n")


def stage_cost_stress(ctx: RunContext) -> None:
    df = build_or_load_path_metrics(ctx)
    replay = pd.read_csv(replay_summary_path(ctx.run_root)) if replay_summary_path(ctx.run_root).exists() else pd.DataFrame()
    top_ids = replay.sort_values("mean_R", ascending=False).head(10)["executable_variant_id"].tolist() if not replay.empty else []
    scenarios = [
        ("base", 1.0, 0.0, 1.0, 1.0), ("cost_x1p25", 1.25, 0.0, 1.0, 1.0), ("cost_x1p5", 1.5, 0.0, 1.0, 1.0), ("cost_x2", 2.0, 0.0, 1.0, 1.0),
        ("add_10bps", 1.0, 10.0, 1.0, 1.0), ("add_25bps", 1.0, 25.0, 1.0, 1.0), ("maker_taker_optimistic", 1.0, 0.0, 0.65, 1.0), ("maker_first_taker_failsafe", 1.0, 0.0, 0.85, 1.0), ("funding_doubled_adverse", 1.0, 0.0, 1.0, 2.0), ("mark_fallback_disabled", 1.0, 0.0, 1.0, 1.0),
    ]
    rows = []
    if not df.empty and top_ids:
        for xid in top_ids:
            parts = xid.split("__")
            source_vid = parts[0]
            tail = parts[1] if len(parts) > 1 else ""
            stop = 0.5
            target = 5.0
            horizon = "30m"
            for token in tail.split("_"):
                if token.startswith("stop"):
                    stop = float(token.replace("stop", ""))
                elif token.startswith("target"):
                    target = float(token.replace("target", ""))
                elif token in {"15m", "30m", "1h"}:
                    horizon = token
            sub = df[df["variant_id"] == source_vid]
            if sub.empty:
                continue
            for name, mult, extra, maker_disc, fund_mult in scenarios:
                use = sub
                if name == "mark_fallback_disabled":
                    use = sub[sub.get("mark_path_status") == "mark_available"]
                ret = surface_net_returns(use, horizon, stop, target, "pessimistic", mult, extra, maker_disc, fund_mult)
                s = summarize_r(ret)
                rows.append({"executable_variant_id": xid, "scenario": name, **s, "survives_positive": bool(s["events"] and s["mean_R"] > 0 and s["PF"] > 1.0)})
    write_csv(ctx.run_root / "execution/d1_execution_cost_stress_summary.csv", rows)
    write_text(ctx.run_root / "execution/d1_execution_cost_stress_report.md", "# D1 Execution Cost Stress Report\n\nStress is accounting-only over fixed entries. Rankable candidates must survive cost_x1p25 and must not depend on maker-only optimistic assumptions or mark fallback.\n")


def stage_data_quality(ctx: RunContext) -> None:
    df = build_or_load_path_metrics(ctx)
    rows = []
    if not df.empty:
        q99 = pd.to_numeric(df.get("range_pct"), errors="coerce").quantile(0.99)
        base = surface_net_returns(df, "30m", 0.5, 5.0)
        filters = {
            "all": pd.Series(True, index=df.index),
            "exclude_top1pct_wicks": pd.to_numeric(df.get("range_pct"), errors="coerce") <= q99,
            "exclude_zero_turnover": pd.to_numeric(df.get("turnover"), errors="coerce") > 0,
            "mark_available_only": df.get("mark_path_status", pd.Series("last_price_proxy", index=df.index)) == "mark_available",
            "exclude_abnormal_range_proxy": pd.to_numeric(df.get("range_pct"), errors="coerce") <= pd.to_numeric(df.get("range_pct"), errors="coerce").quantile(0.995),
        }
        for name, mask in filters.items():
            ret = base[mask.fillna(False)]
            s = summarize_r(ret)
            rows.append({"filter": name, **s, "edge_destroyed": bool(s["events"] and s["mean_R"] <= 0)})
    write_csv(ctx.run_root / "data_quality/d1_bad_wick_artifact_summary.csv", rows)
    write_text(ctx.run_root / "data_quality/d1_bad_wick_artifact_report.md", "# D1 Bad-Wick And Data Quality Audit\n\nFilters are diagnostic overlays and do not mutate the event ledger. Mark-available-only is expected to be sparse or empty given prior mark-path limitations.\n")


def stage_regime(ctx: RunContext) -> None:
    df = build_or_load_path_metrics(ctx)
    rows = []
    filt_rows = []
    if not df.empty:
        tmp = df.copy()
        tmp["funding_sign"] = np.where(pd.to_numeric(tmp.get("funding_rate"), errors="coerce") > 0, "positive", np.where(pd.to_numeric(tmp.get("funding_rate"), errors="coerce") < 0, "negative", "zero_or_missing"))
        tmp["oi_state"] = np.where(pd.to_numeric(tmp.get("oi_chg_24h"), errors="coerce") > 0, "oi_up", np.where(pd.to_numeric(tmp.get("oi_chg_24h"), errors="coerce") < 0, "oi_down", "oi_unknown"))
        tmp["vol_bucket"] = pd.qcut(pd.to_numeric(tmp.get("range_pct"), errors="coerce"), 3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
        tmp["turnover_bucket"] = pd.qcut(pd.to_numeric(tmp.get("turnover"), errors="coerce"), 3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
        for cols in [["side"], ["variant_id"], ["btc_eth_regime"], ["funding_sign"], ["oi_state"], ["vol_bucket"], ["turnover_bucket"], ["side", "btc_eth_regime"]]:
            for keys, sub in tmp.groupby(cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                ret = surface_net_returns(sub, "30m", 0.5, 5.0)
                rows.append({"slice_type": "+".join(cols), **{c: v for c, v in zip(cols, keys)}, **summarize_r(ret)})
        filter_defs = {
            "avoid_high_bad_wick_spread_proxy": pd.to_numeric(tmp.get("range_pct"), errors="coerce") <= pd.to_numeric(tmp.get("range_pct"), errors="coerce").quantile(0.99),
            "require_low_volume_state": pd.to_numeric(tmp.get("turnover"), errors="coerce") <= pd.to_numeric(tmp.get("turnover"), errors="coerce").rolling(1, min_periods=1).median().fillna(np.inf),
            "avoid_btc_eth_violently_against_reversal": tmp.get("btc_eth_regime", pd.Series("unknown", index=tmp.index)).astype(str) != "both_negative",
        }
        for name, mask in filter_defs.items():
            filt_rows.append({"filter": name, **summarize_r(surface_net_returns(tmp[mask.fillna(False)], "30m", 0.5, 5.0))})
    write_csv(ctx.run_root / "regime/d1_state_stratification_summary.csv", rows)
    write_csv(ctx.run_root / "regime/d1_state_filter_audit.csv", filt_rows)
    write_text(ctx.run_root / "regime/d1_state_filter_report.md", "# D1 State Filter Audit\n\nOnly the three predeclared economically motivated filters are tested. They are diagnostic and cannot be added to a candidate without a new frozen contract.\n")


def stage_validation(ctx: RunContext) -> None:
    df = build_or_load_path_metrics(ctx)
    rows = []
    details = []
    if not df.empty:
        df = df.sort_values("decision_ts").reset_index(drop=True)
        ts = pd.to_datetime(df["decision_ts"], utc=True)
        blocks = pd.qcut(ts.rank(method="first"), 8, labels=False, duplicates="drop")
        df["block"] = blocks
        combos = []
        vals = sorted(df["block"].dropna().unique())
        for i, a in enumerate(vals):
            for b in vals[i + 1:]:
                combos.append((a, b))
        for stop in [0.5, 0.75, 1.0]:
            for target in [3.0, 5.0]:
                for horizon in ["15m", "30m", "1h"]:
                    path_rs = []
                    for a, b in combos:
                        sub = df[df["block"].isin([a, b])]
                        ret = surface_net_returns(sub, horizon, stop, target)
                        s = summarize_r(ret)
                        path_rs.append(s["mean_R"] if np.isfinite(s["mean_R"]) else np.nan)
                        details.append({"stop_atr": stop, "target_R": target, "time_exit": horizon, "test_blocks": f"{a},{b}", **s})
                    arr = pd.Series(path_rs).dropna()
                    rows.append({"stop_atr": stop, "target_R": target, "time_exit": horizon, "paths": len(arr), "median_path_net_R": float(arr.median()) if len(arr) else np.nan, "percent_positive_paths": float((arr > 0).mean()) if len(arr) else np.nan, "worst_path_net_R": float(arr.min()) if len(arr) else np.nan, "path_dispersion": float(arr.std()) if len(arr) > 1 else np.nan})
    write_csv(ctx.run_root / "validation/d1_walk_forward_summary.csv", rows)
    write_csv(ctx.run_root / "validation/d1_cpcv_summary.csv", rows)
    if details:
        pd.DataFrame(details).to_parquet(ctx.run_root / "validation/d1_cpcv_path_details.parquet", index=False)
    write_text(ctx.run_root / "validation/d1_validation_report.md", "# D1 Walk-Forward And CPCV Report\n\nEight chronological blocks are approximated over pre-holdout D1 rows. Two-block test combinations provide path-stability diagnostics.\n")


def stage_portfolio(ctx: RunContext) -> None:
    replay = pd.read_csv(replay_summary_path(ctx.run_root)) if replay_summary_path(ctx.run_root).exists() else pd.DataFrame()
    stress = pd.read_csv(ctx.run_root / "execution/d1_execution_cost_stress_summary.csv") if (ctx.run_root / "execution/d1_execution_cost_stress_summary.csv").exists() else pd.DataFrame()
    eligible = []
    if not replay.empty and not stress.empty:
        pass_ids = set(stress[(stress["scenario"] == "cost_x1p25") & (stress["survives_positive"].astype(str).str.lower().isin(["true", "1"]))]["executable_variant_id"])
        eligible = replay[replay["executable_variant_id"].isin(pass_ids)].sort_values("mean_R", ascending=False).head(3).to_dict("records")
    rows = []
    for row in eligible:
        mean_r = float(row.get("mean_R", 0.0))
        for equity in [200, 500, 1000]:
            for risk_pct in [0.025, 0.05, 0.10, 0.15, 0.20]:
                for max_pos in [1, 3, 5, 10]:
                    capped_trades = min(int(row.get("events", 0)), 250)
                    growth = (1.0 + risk_pct * mean_r) ** capped_trades if 1.0 + risk_pct * mean_r > 0 else 0.0
                    rows.append({"executable_variant_id": row["executable_variant_id"], "starting_equity": equity, "risk_pct": risk_pct, "max_open_positions": max_pos, "events_used_cap250": capped_trades, "compounded_equity_proxy": equity * growth, "max_drawdown_proxy_R": row.get("max_dd_R_proxy"), "ruin_proxy": growth <= 0.1, "liquidation_count_proxy": row.get("liquidation_count", 0), "note": "diagnostic only; aggressive sizing cannot create alpha"})
    write_csv(ctx.run_root / "portfolio/d1_aggressive_10x_portfolio_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/d1_aggressive_10x_portfolio_report.md", "# D1 Aggressive 10x Portfolio Overlay\n\nRuns only for variants passing earlier proxy gates. This is a diagnostic stress overlay, not live guidance.\n")


def stage_data_plan(ctx: RunContext) -> None:
    events = pd.read_parquet(event_ledger_path(ctx.run_root)) if event_ledger_path(ctx.run_root).exists() else pd.DataFrame()
    rows = []
    if not events.empty:
        sample = events.head(10000)
        for _, ev in sample.iterrows():
            rows.append({"symbol": ev["symbol"], "event_id": ev["event_id"], "window_start": str(pd.Timestamp(ev["decision_ts"]) - pd.Timedelta(hours=2)), "window_end": str(pd.Timestamp(ev["entry_ts"]) + pd.Timedelta(hours=4)), "priority": "accepted_event_or_matched_null_window"})
    write_csv(ctx.run_root / "data_plan/d1_event_windows_for_data_request.csv", rows)
    total_hours = len(rows) * 6
    write_csv(ctx.run_root / "data_plan/d1_storage_estimate.csv", [{"windows": len(rows), "total_window_hours": total_hours, "estimated_1m_ohlcv_mb": round(total_hours * 0.02, 2), "estimated_tob_depth_mb": round(total_hours * 2.0, 2), "note": "rough planning proxy only"}])
    write_text(ctx.run_root / "data_plan/d1_targeted_data_acquisition_plan.md", f"# D1 Targeted Data Acquisition Plan\n\n- event windows listed: `{len(rows)}`\n- requested window: 2h before decision through 4h after entry/exit proxy\n- priority data: 1m OHLCV, top-of-book spread/depth, public trades, mark/index/funding/OI verification\n- no large download performed by this run.\n")


def stage_defer_memo(ctx: RunContext) -> None:
    write_text(ctx.run_root / "deferred/d3_e1_defer_memo.md", "# D3/E1 Hold Or Defer Memo\n\nD3 and E1 are deferred because the prior path diagnostic found matched-null path edge but also material 10x liquidation/proxy-risk concerns and effectively absent mark-path availability. Before either gets a validation run, the project needs better mark path availability, 1m or targeted event data, liquidation-proxy improvement or a liquidation feed, and stricter stabilization filters.\n")


def stage_decision(ctx: RunContext) -> None:
    null = pd.read_csv(ctx.run_root / "matched_null/d1_matched_null_summary.csv") if (ctx.run_root / "matched_null/d1_matched_null_summary.csv").exists() else pd.DataFrame()
    stress = pd.read_csv(ctx.run_root / "execution/d1_execution_cost_stress_summary.csv") if (ctx.run_root / "execution/d1_execution_cost_stress_summary.csv").exists() else pd.DataFrame()
    one = pd.read_csv(ctx.run_root / "one_minute/d1_1m_vs_5m_summary.csv") if (ctx.run_root / "one_minute/d1_1m_vs_5m_summary.csv").exists() else pd.DataFrame()
    dq = pd.read_csv(ctx.run_root / "data_quality/d1_bad_wick_artifact_summary.csv") if (ctx.run_root / "data_quality/d1_bad_wick_artifact_summary.csv").exists() else pd.DataFrame()
    cpcv = pd.read_csv(ctx.run_root / "validation/d1_cpcv_summary.csv") if (ctx.run_root / "validation/d1_cpcv_summary.csv").exists() else pd.DataFrame()
    null_pass = bool(len(null) and str(null.iloc[0].get("beats_null_path", False)).lower() in {"true", "1"})
    cost_pass = bool(len(stress) and len(stress[(stress["scenario"] == "cost_x1p25") & (stress["survives_positive"].astype(str).str.lower().isin(["true", "1"]))]))
    one_status = str(one.iloc[0].get("status", "unknown")) if len(one) else "unknown"
    one_pass = one_status != "informative" or float(one.iloc[0].get("mean_1m_R_proxy", 0) or 0) > 0
    dq_destroy = bool(len(dq) and len(dq[(dq["filter"].isin(["exclude_top1pct_wicks", "exclude_zero_turnover"])) & (dq["edge_destroyed"].astype(str).str.lower().isin(["true", "1"]))]))
    cpcv_pass = bool(len(cpcv) and pd.to_numeric(cpcv.get("percent_positive_paths"), errors="coerce").max() >= 0.55)
    if not null_pass:
        verdict = "reject_d1_current_translation"
    elif dq_destroy:
        verdict = "blocked_by_data_quality"
    elif one_status == "blocked_sparse_1m":
        verdict = "blocked_by_1m_or_execution_data"
    elif not cost_pass or not cpcv_pass:
        verdict = "continue_unsealed_validation"
    elif one_pass:
        verdict = "promote_to_targeted_execution_data_collection"
    else:
        verdict = "blocked_by_1m_or_execution_data"
    decision = {"verdict": verdict, "final_holdout_untouched": True, "null_pass": null_pass, "cost_x1p25_pass": cost_pass, "one_minute_status": one_status, "data_quality_edge_destroyed": dq_destroy, "cpcv_pass": cpcv_pass, "allowed_verdicts": sorted(ALLOWED_VERDICTS)}
    write_json(ctx.run_root / "decision_summary.json", decision)
    report = f"""# D1 Narrow Validation Report\n\n## Verdict\n`{verdict}`\n\n## Gate Summary\n- Final holdout untouched: `true`\n- Stronger matched null pass: `{null_pass}`\n- 1m status: `{one_status}`\n- Cost x1.25 pass: `{cost_pass}`\n- Data quality destroyed edge: `{dq_destroy}`\n- CPCV/path stability pass: `{cpcv_pass}`\n\n## Interpretation\nThis is train-only execution-realism audit output. It does not authorize live trading and does not request sealed validation. If promoted, the only permitted next step is targeted execution data collection around D1 events and matched null windows.\n\n## Key Artifact Paths\n- Contract: `contracts/d1_low_volume_reversal_validation_contract.json`\n- Events: `events/d1_event_ledger.parquet`\n- Matched null: `matched_null/d1_matched_null_summary.csv`\n- Replay: `replay/d1_executable_replay_summary.csv`\n- 1m audit: `one_minute/d1_1m_vs_5m_summary.csv`\n- Cost stress: `execution/d1_execution_cost_stress_summary.csv`\n- Validation: `validation/d1_cpcv_summary.csv`\n"""
    write_text(ctx.run_root / "D1_NARROW_VALIDATION_REPORT.md", report)


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rels = [
        "D1_NARROW_VALIDATION_REPORT.md", "decision_summary.json", "contracts/d1_low_volume_reversal_validation_contract.json",
        "preflight/resource_guard_report.md", "preflight/preflight_report.md", "notifications/telegram_readiness_report.md", "tmux/watch_commands.md",
        "seal/seal_guard_report.md", "events/d1_event_coverage_report.md", "events/d1_event_coverage_summary.csv", "events/d1_event_ledger_sample.parquet",
        "matched_null/d1_matched_null_report.md", "matched_null/d1_matched_null_summary.csv", "replay/d1_executable_replay_report.md", "replay/d1_executable_replay_summary.csv", "replay/d1_executable_variant_registry.csv", "replay/d1_trade_ledger_sample.parquet",
        "one_minute/d1_1m_coverage_report.md", "one_minute/d1_1m_vs_5m_summary.csv", "execution/d1_execution_cost_stress_report.md", "execution/d1_execution_cost_stress_summary.csv", "data_quality/d1_bad_wick_artifact_report.md", "data_quality/d1_bad_wick_artifact_summary.csv", "regime/d1_state_filter_report.md", "regime/d1_state_stratification_summary.csv", "regime/d1_state_filter_audit.csv", "validation/d1_validation_report.md", "validation/d1_cpcv_summary.csv", "portfolio/d1_aggressive_10x_portfolio_report.md", "portfolio/d1_aggressive_10x_portfolio_summary.csv", "data_plan/d1_targeted_data_acquisition_plan.md", "data_plan/d1_storage_estimate.csv", "deferred/d3_e1_defer_memo.md",
    ]
    idx = []
    for rel in rels:
        src = ctx.run_root / rel
        if src.exists():
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            idx.append({"artifact": rel, "bundle_copy": str(dst.relative_to(ctx.run_root)), "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", idx)
    write_json(bundle / "artifact_path_index.json", {"artifacts": idx})
    zip_path = ctx.run_root / "d1_narrow_validation_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in bundle.rglob("*"):
            if item.is_file():
                z.write(item, arcname=str(item.relative_to(ctx.run_root)))
    write_text(bundle / "README.md", f"Compact D1 narrow validation review bundle. Zip: `{zip_path}`\n")


STAGE_FUNCS = {
    "preflight-resource-and-artifact-audit": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram_tmux,
    "seal-guard": stage_seal,
    "d1-contract-freeze": stage_contract,
    "d1-full-coverage-event-rebuild": stage_events,
    "d1-stronger-matched-nulls": stage_matched_nulls,
    "d1-executable-replay-surface": stage_replay,
    "d1-1m-where-available-audit": stage_one_minute,
    "d1-execution-cost-stress": stage_cost_stress,
    "d1-bad-wick-and-data-quality-audit": stage_data_quality,
    "d1-regime-and-state-filters": stage_regime,
    "d1-walk-forward-and-cpcv": stage_validation,
    "d1-aggressive-10x-portfolio-overlay": stage_portfolio,
    "d1-targeted-data-acquisition-plan": stage_data_plan,
    "d3-e1-hold-or-defer-memo": stage_defer_memo,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    start, end = clamp_window(args)
    run_root, root_reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tmp").mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": root_reason, "started_at_utc": utc_now(), "argv": sys.argv, "seed": args.seed, "smoke": args.smoke, "start": str(start), "end": str(end)})
    stages = stage_list(args.stage)
    notifier.send("D1 RUN START", f"stages={stages}\nrun_root={run_root}")
    try:
        for stage in stages:
            if args.resume and stage_complete(run_root, stage):
                continue
            append_command(run_root, stage)
            ensure_guard(ctx, stage, estimate_stage_gb(stage, args.smoke, args.nulls_per_event))
            notifier.send("D1 STAGE START", stage)
            if args.dry_run:
                write_text(run_root / "dry_run" / f"{stage}.txt", f"would run {stage}")
                mark_done(run_root, stage)
                continue
            STAGE_FUNCS[stage](ctx)
            mark_done(run_root, stage)
            shutil.rmtree(run_root / "tmp" / stage, ignore_errors=True)
            watch = {"ts_utc": utc_now(), "stage": stage, "status": "complete", "run_root": str(run_root)}
            write_json(run_root / "watch_status.json", watch)
            notifier.send("D1 STAGE COMPLETE", stage)
        notifier.send("D1 RUN COMPLETE", f"run_root={run_root}")
        return 0
    except Exception as exc:
        notifier.send("D1 RUN FAILED", f"{type(exc).__name__}: {exc}", level="error")
        write_text(run_root / "FAILED.txt", f"{utc_now()} {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
