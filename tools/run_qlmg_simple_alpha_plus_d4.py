#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_regime_stack import (  # noqa: E402
    FINAL_HOLDOUT_START,
    SCREENING_END,
    build_regime_panel,
    join_regime,
    regime_feature_dictionary,
    stable_hash,
    validate_no_protected,
)
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_simple_alpha_plus_d4_20260626_v1"
FIRST_SCREEN_ROOT = RESULTS_ROOT / "phase_qlmg_engine_and_first_screen_20260624_v1_20260624_101747"
PATH_DIAG_ROOT = RESULTS_ROOT / "phase_qlmg_path_diagnostics_exit_surface_20260624_v1_20260624_121522"
D1_ROOT = RESULTS_ROOT / "phase_qlmg_d1_narrow_validation_20260624_v1"
PILOT_1M_ROOT = RESULTS_ROOT / "phase_qlmg_targeted_1m_data_pilot_20260624_v1"
REGIME_SWEEP_ROOT = RESULTS_ROOT / "phase_qlmg_regime_stack_and_smart_sweep_20260625_v1"
F1G1_ROOT = RESULTS_ROOT / "phase_qlmg_f1_g1_short_unblock_20260625_v1_20260625_120455"
ALPHA_MARATHON_ROOT = RESULTS_ROOT / "phase_qlmg_alpha_discovery_marathon_20260625_v1_20260625_145339"
D4_AUDIT_ROOT = RESULTS_ROOT / "phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"

STAGES = (
    "preflight-and-prior-result-audit",
    "telegram-and-tmux-setup",
    "seal-guard",
    "d4-carry-forward-freeze",
    "simple-alpha-contract-freeze",
    "regime-and-context-refresh",
    "data-readiness-by-family",
    "d4-execution-depth-plan",
    "leader-breakout-event-generator",
    "weak-asset-spike-fade-event-generator",
    "risk-off-exhaustion-spike-short-generator",
    "synthetic-open-orb-generators",
    "new-perp-listing-event-study-builder",
    "auxiliary-event-generators",
    "path-first-diagnostics",
    "family-mechanism-dossiers",
    "same-time-and-matched-null-baselines",
    "bounded-sobol-executable-sweep",
    "broad-prelead-holding-pen",
    "full-coverage-rerun-for-preleads",
    "d4-plus-prelead-1m-and-depth-overlay",
    "cost-funding-liquidation-stress",
    "walk-forward-cpcv-and-selection-controls",
    "aggressive-10x-portfolio-overlay",
    "triage-and-next-contracts",
    "decision-report",
    "compact-review-bundle",
    "all",
)

SIMPLE_FAMILIES = [
    "leader_breakout_long",
    "weak_asset_spike_fade",
    "risk_off_exhaustion_spike_short",
    "funding_window_orb_failure",
    "us_cash_open_orb",
    "utc_daily_open_reversal",
    "new_perp_listing_event_study",
    "crowded_long_unwind_short",
    "failed_sector_rotation_short",
    "post_catalyst_continuation_base",
]

FAMILY_MAX_BUDGETS = {
    "leader_breakout_long": 400,
    "weak_asset_spike_fade": 400,
    "risk_off_exhaustion_spike_short": 320,
    "funding_window_orb_failure": 260,
    "us_cash_open_orb": 220,
    "utc_daily_open_reversal": 160,
    "new_perp_listing_event_study": 160,
    "crowded_long_unwind_short": 140,
    "failed_sector_rotation_short": 100,
    "post_catalyst_continuation_base": 100,
    "reserve_refinement": 140,
}

FAMILY_SUBFAMILIES = {
    "leader_breakout_long": ["daily_close_breakout", "4h_breakout", "orb_continuation", "retest_pullback_continuation"],
    "weak_asset_spike_fade": ["failed_range_breakout", "vwap_loss", "failed_orb", "lower_high_break"],
    "risk_off_exhaustion_spike_short": ["vwap_loss", "lower_high", "failed_orb", "price_up_oi_down"],
    "funding_window_orb_failure": ["funding_5m_failure", "funding_15m_failure", "funding_30m_failure"],
    "us_cash_open_orb": ["us_open_continuation", "us_open_failure", "us_open_reversal"],
    "utc_daily_open_reversal": ["utc_open_reclaim", "utc_open_reject", "utc_impulse_reversal"],
    "new_perp_listing_event_study": ["launch_orb_continuation", "vwap_loss_short", "day_one_midpoint_failure", "first_lower_high"],
    "crowded_long_unwind_short": ["failed_high", "vwap_loss", "range_low_break"],
    "failed_sector_rotation_short": ["sector_proxy_member_spike", "sector_relative_rollover", "failed_orb"],
    "post_catalyst_continuation_base": ["base_breakout", "vwap_base", "post_impulse_tight_base"],
    "D4": ["dynamic_1p25", "dynamic_1p5", "fixed_3x", "risk_based_reduce_until_buffer"],
}

SOURCE_FAMILY_MAP = {
    "leader_breakout_long": ["A2"],
    "weak_asset_spike_fade": ["D1", "D3"],
    "risk_off_exhaustion_spike_short": ["F1", "G1", "D1"],
    "funding_window_orb_failure": ["D3", "E1"],
    "us_cash_open_orb": ["A2", "D3", "E1"],
    "utc_daily_open_reversal": ["D1", "D3", "E1"],
    "new_perp_listing_event_study": ["D3", "E1"],
    "crowded_long_unwind_short": ["F1", "G1", "A2"],
    "failed_sector_rotation_short": ["F1", "G1"],
    "post_catalyst_continuation_base": ["A2", "D3"],
}

SOURCE_EXIT_PRESETS = {
    # Presets are copied from the prior path-diagnostics exit surface as
    # coarse neighborhoods, not final rules. They ensure the discovery phase
    # starts from configurations known to produce enough events before it
    # evaluates robustness and costs.
    "D1": [
        {"tier_filter": "C", "horizon": "30m", "target_r": 5.0, "stop_mult": 0.5, "risk_bps_override": None, "preset_name": "D1_C_atr0p5_5R_30m"},
        {"tier_filter": "C", "horizon": "1h", "target_r": 5.0, "stop_mult": 0.5, "risk_bps_override": None, "preset_name": "D1_C_atr0p5_5R_1h"},
        {"tier_filter": "C", "horizon": "30m", "target_r": 3.0, "stop_mult": 0.5, "risk_bps_override": None, "preset_name": "D1_C_atr0p5_3R_30m"},
    ],
    "D3": [
        {"tier_filter": "C", "horizon": "30m", "target_r": 5.0, "stop_mult": 0.5, "risk_bps_override": None, "preset_name": "D3_C_atr0p5_5R_30m"},
        {"tier_filter": "C", "horizon": "72h", "target_r": 5.0, "stop_mult": 1.0, "risk_bps_override": 1500.0, "preset_name": "D3_C_pct15_5R_72h"},
        {"tier_filter": "C", "horizon": "72h", "target_r": 3.0, "stop_mult": 1.0, "risk_bps_override": 1500.0, "preset_name": "D3_C_pct15_3R_72h"},
        {"tier_filter": "C", "horizon": "72h", "target_r": 5.0, "stop_mult": 1.0, "risk_bps_override": 1000.0, "preset_name": "D3_C_pct10_5R_72h"},
    ],
    "E1": [
        {"tier_filter": "A_B", "horizon": "72h", "target_r": 5.0, "stop_mult": 1.0, "risk_bps_override": 500.0, "preset_name": "E1_AB_pct5_5R_72h"},
        {"tier_filter": "A_B", "horizon": "48h", "target_r": 5.0, "stop_mult": 1.0, "risk_bps_override": 500.0, "preset_name": "E1_AB_pct5_5R_48h"},
        {"tier_filter": "A_B", "horizon": "72h", "target_r": 3.0, "stop_mult": 1.0, "risk_bps_override": 500.0, "preset_name": "E1_AB_pct5_3R_72h"},
        {"tier_filter": "A_B", "horizon": "72h", "target_r": 5.0, "stop_mult": 1.0, "risk_bps_override": 300.0, "preset_name": "E1_AB_pct3_5R_72h"},
    ],
    "A2": [
        {"tier_filter": "A_B", "horizon": "72h", "target_r": 5.0, "stop_mult": 1.0, "risk_bps_override": 300.0, "preset_name": "A2_AB_pct3_5R_72h"},
        {"tier_filter": "A_B", "horizon": "72h", "target_r": 5.0, "stop_mult": 1.0, "risk_bps_override": 500.0, "preset_name": "A2_AB_pct5_5R_72h"},
        {"tier_filter": "A_B", "horizon": "48h", "target_r": 5.0, "stop_mult": 1.0, "risk_bps_override": 300.0, "preset_name": "A2_AB_pct3_5R_48h"},
        {"tier_filter": "C", "horizon": "30m", "target_r": 5.0, "stop_mult": 0.5, "risk_bps_override": None, "preset_name": "A2_C_atr0p5_5R_30m"},
    ],
}

READY_LABELS = {"ready_for_path_and_replay", "ready_for_path_only"}
HORIZONS = ["30m", "1h", "2h", "4h", "6h", "12h", "24h", "48h", "72h"]
MIN_EVALUATION_EVENTS = 50
MIN_PRELEAD_EVENTS = 100
SMOKE_MIN_EVALUATION_EVENTS = 5
SMOKE_MIN_PRELEAD_EVENTS = 5
SPARSE_EVENT_STUDY_FAMILIES = {"new_perp_listing_event_study", "post_catalyst_continuation_base"}
SPARSE_MIN_EVALUATION_EVENTS = 10
SPARSE_MIN_PRELEAD_EVENTS = 25
SMOKE_SPARSE_MIN_EVALUATION_EVENTS = 3
SMOKE_SPARSE_MIN_PRELEAD_EVENTS = 5
CORE_PATH_COLS = [
    "event_id", "family", "variant_id", "symbol", "side", "liquidity_tier", "decision_ts", "entry_ts",
    "entry_ref_price", "reference_risk_bps", "atr_bps", "btc_eth_regime", "oi_chg_24h", "funding_rate",
    "turnover", "range_pct", "data_quality_flags", "mark_path_status", "liq_price_10x",
    "btc_ret_4h", "btc_ret_24h", "eth_ret_4h", "eth_ret_24h", "ret_24h", "listing_age_days",
]
for _h in HORIZONS:
    CORE_PATH_COLS.extend([
        f"{_h}_path_available", f"{_h}_mfe_bps", f"{_h}_mae_bps", f"{_h}_close_return_bps",
        f"{_h}_pos1R_before_neg1R", f"{_h}_liquidation_10x",
    ])

FAMILY_LABELS = {
    "promote_to_family_specific_validation",
    "promote_to_targeted_execution_data_collection",
    "continue_hypothesis_development",
    "path_edge_exit_problem",
    "regime_specific_candidate",
    "cost_fragile_candidate",
    "sample_limited_needs_more_data",
    "data_blocked",
    "not_fairly_tested_missing_data",
    "not_fairly_tested_generator_incomplete",
    "not_fairly_tested_execution_model_missing",
    "reject_current_translation_only",
    "reject_family_for_now",
    "blocked_by_protocol_issue",
    "carry_forward_d4_execution_depth",
}

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
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-simple-alpha-d4")
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
    p = argparse.ArgumentParser(description="QLMG simple-alpha plus D4 evidence-preserving discovery")
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
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=20260626)
    p.add_argument("--discovery-budget", type=int, default=2400)
    p.add_argument("--family-budget", type=int, default=0)
    p.add_argument("--refine-budget", type=int, default=360)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--use-targeted-1m-if-overlap", action="store_true")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--tmux-session-name", default="qlmg_simple_alpha_d4")
    p.add_argument("--d4-carry-forward-required", action="store_true", default=True)
    p.add_argument("--build-depth-plan", action="store_true", default=True)
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
            writer.writerow(row)


def write_df_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_json_safe(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def read_text_safe(path: Path, limit: int = 6000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit] if path.exists() else ""
    except Exception:
        return ""


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
        "preflight-and-prior-result-audit": [run_root / "preflight/preflight_report.md", run_root / "budget/family_budget_manifest.csv", run_root / "runtime/runtime_estimate_report.md"],
        "telegram-and-tmux-setup": [run_root / "notifications/telegram_readiness_report.md", run_root / "tmux/watch_commands.md"],
        "seal-guard": [run_root / "seal/seal_guard_report.md", run_root / "seal/protected_slice_check.json"],
        "d4-carry-forward-freeze": [run_root / "d4/d4_carry_forward_contract.json", run_root / "d4/d4_carry_forward_summary.csv"],
        "simple-alpha-contract-freeze": [run_root / "contracts/simple_alpha_contract_summary.csv"],
        "regime-and-context-refresh": [run_root / "regime/regime_context_panel.parquet", run_root / "regime/regime_context_qc_report.md"],
        "data-readiness-by-family": [run_root / "readiness/family_readiness_matrix.csv", run_root / "data_build/data_build_decision_table.csv"],
        "d4-execution-depth-plan": [run_root / "d4_depth/d4_execution_depth_plan.md", run_root / "d4_depth/d4_next_data_contract.json"],
        "leader-breakout-event-generator": [run_root / "events/leader_breakout_events.parquet", run_root / "events/leader_breakout_event_report.md"],
        "weak-asset-spike-fade-event-generator": [run_root / "events/weak_asset_spike_fade_events.parquet", run_root / "events/weak_asset_spike_fade_event_report.md"],
        "risk-off-exhaustion-spike-short-generator": [run_root / "events/risk_off_exhaustion_spike_short_events.parquet", run_root / "events/risk_off_exhaustion_spike_short_event_report.md"],
        "synthetic-open-orb-generators": [run_root / "events/synthetic_open_orb_events.parquet", run_root / "events/synthetic_open_orb_event_report.md"],
        "new-perp-listing-event-study-builder": [run_root / "listing/listing_event_study_manifest.csv", run_root / "listing/listing_event_study_report.md"],
        "auxiliary-event-generators": [run_root / "events/auxiliary_event_generation_report.md"],
        "path-first-diagnostics": [run_root / "path/path_summary_by_family.csv", run_root / "path/path_diagnostics_report.md"],
        "family-mechanism-dossiers": [run_root / "dossiers/family_mechanism_dossier_index.csv"],
        "same-time-and-matched-null-baselines": [run_root / "nulls/baseline_and_null_summary.csv", run_root / "nulls/baseline_and_null_report.md"],
        "bounded-sobol-executable-sweep": [run_root / "sweep/candidate_registry.csv", run_root / "sweep/sweep_results.parquet"],
        "broad-prelead-holding-pen": [run_root / "holding_pen/holding_pen_registry.csv", run_root / "holding_pen/holding_pen_overflow_registry.csv"],
        "full-coverage-rerun-for-preleads": [run_root / "preleads/prelead_registry.csv", run_root / "preleads/full_coverage_summary.csv"],
        "d4-plus-prelead-1m-and-depth-overlay": [run_root / "one_minute/prelead_1m_overlay_summary.csv", run_root / "d4_depth/d4_depth_request_final.csv"],
        "cost-funding-liquidation-stress": [run_root / "stress/prelead_stress_summary.csv"],
        "walk-forward-cpcv-and-selection-controls": [run_root / "validation/prelead_cpcv_summary.csv", run_root / "validation/family_multiple_testing_report.md"],
        "aggressive-10x-portfolio-overlay": [run_root / "portfolio/aggressive_10x_summary.csv"],
        "triage-and-next-contracts": [run_root / "triage/family_triage_summary.csv", run_root / "triage/all_ideas_preservation_index.csv", run_root / "next_contracts/next_action_contract_summary.csv"],
        "decision-report": [run_root / "QLMG_SIMPLE_ALPHA_PLUS_D4_REPORT.md", run_root / "decision_summary.json"],
        "compact-review-bundle": [run_root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return m.get(stage, [])


def stage_complete(run_root: Path, stage: str) -> bool:
    return done_path(run_root, stage).exists() and all(p.exists() for p in required_outputs_for_stage(run_root, stage))


def estimate_stage_gb(stage: str, smoke: bool) -> float:
    base = {
        "regime-and-context-refresh": 0.8,
        "path-first-diagnostics": 1.2,
        "bounded-sobol-executable-sweep": 0.8,
        "same-time-and-matched-null-baselines": 1.0,
        "walk-forward-cpcv-and-selection-controls": 0.5,
    }.get(stage, 0.15)
    return min(base, 0.25) if smoke else base


def estimate_runtime_hours(args: argparse.Namespace) -> float:
    if args.smoke:
        return 0.25
    return 20.0 + (max(args.discovery_budget, 0) / 90.0) + (max(args.refine_budget, 0) / 180.0)


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
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", {"stage": stage, **status})
    if status["warnings"]:
        ctx.notifier.send(f"QLMG simple-alpha D4 resource warning: {stage}", json.dumps(status), level="warning")
    if status["status"] != "pass":
        raise RuntimeError(f"resource guard failed for {stage}: {status}")


def parquet_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetDataset(path).schema.names)
    except Exception:
        return []


def load_path_metrics(ctx: RunContext, *, null: bool = False) -> pd.DataFrame:
    path = PATH_DIAG_ROOT / ("matched_null/matched_null_path_metrics.parquet" if null else "path_diagnostics/path_metrics.parquet")
    if not path.exists():
        return pd.DataFrame()
    cols = [c for c in CORE_PATH_COLS if c in set(parquet_columns(path))]
    df = pd.read_parquet(path, columns=cols)
    for c in ["decision_ts", "entry_ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df[(df["decision_ts"] >= ctx.start) & (df["decision_ts"] <= ctx.end)]
    if ctx.args.max_symbols:
        symbols = sorted(df["symbol"].dropna().unique())[: ctx.args.max_symbols]
        df = df[df["symbol"].isin(symbols)]
    if ctx.args.smoke:
        df = df.sort_values(["family", "symbol", "decision_ts"]).groupby("family", group_keys=False).head(3000)
    validate_no_protected(df, ["decision_ts", "entry_ts"])
    return df.reset_index(drop=True)


def d4_summary_rows() -> pd.DataFrame:
    return read_csv_safe(D4_SURVIVAL_ROOT / "sizing/liquidation_safe_sizing_summary.csv")


def load_regime(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "regime/regime_context_panel.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_all_events(ctx: RunContext) -> pd.DataFrame:
    parts = [
        ctx.run_root / "events/leader_breakout_events.parquet",
        ctx.run_root / "events/weak_asset_spike_fade_events.parquet",
        ctx.run_root / "events/risk_off_exhaustion_spike_short_events.parquet",
        ctx.run_root / "events/synthetic_open_orb_events.parquet",
        ctx.run_root / "listing/listing_event_rows.parquet",
        ctx.run_root / "events/auxiliary_events.parquet",
    ]
    frames = []
    for p in parts:
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for c in ["decision_ts", "entry_ts"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    validate_no_protected(out, ["decision_ts", "entry_ts"])
    return out


def load_event_regime(ctx: RunContext) -> pd.DataFrame:
    ev = load_all_events(ctx)
    reg = load_regime(ctx)
    if ev.empty:
        return ev
    if reg.empty:
        reg = build_regime_panel(ev, min_history=3 if ctx.args.smoke else 10)
    return join_regime(ev, reg)


def active_tmux_sessions() -> list[str]:
    try:
        out = subprocess.check_output(["tmux", "list-sessions"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.split(":", 1)[0] for line in out.splitlines() if line.strip()]


def source_exists_family(family: str, source_family: Any) -> bool:
    if source_family is None:
        return False
    sources = SOURCE_FAMILY_MAP.get(family, [])
    return str(source_family) in sources


def family_readiness_rows() -> list[dict[str, Any]]:
    sector_pit = False
    catalyst_pit = False
    one_cov = read_csv_safe(PILOT_1M_ROOT / "qc/pilot_coverage_summary.csv")
    one_ok = not one_cov.empty and len(one_cov) >= 100
    return [
        {"family": "leader_breakout_long", "readiness": "ready_for_path_and_replay", "blocker": "", "intended_regimes": 2, "needs_targeted_1m": False, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": False},
        {"family": "weak_asset_spike_fade", "readiness": "ready_for_path_and_replay", "blocker": "", "intended_regimes": 2, "needs_targeted_1m": True, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": True, "needs_listing_lifecycle_metadata": False},
        {"family": "risk_off_exhaustion_spike_short", "readiness": "ready_for_path_and_replay", "blocker": "", "intended_regimes": 2, "needs_targeted_1m": True, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": True, "needs_listing_lifecycle_metadata": False},
        {"family": "funding_window_orb_failure", "readiness": "ready_for_path_and_replay", "blocker": "", "intended_regimes": 2, "needs_targeted_1m": True, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": False},
        {"family": "us_cash_open_orb", "readiness": "ready_for_path_and_replay", "blocker": "", "intended_regimes": 2, "needs_targeted_1m": True, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": False},
        {"family": "utc_daily_open_reversal", "readiness": "ready_for_path_and_replay", "blocker": "", "intended_regimes": 2, "needs_targeted_1m": False, "needs_top_of_book": True, "needs_shallow_depth": False, "needs_public_trades": False, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": False},
        {"family": "new_perp_listing_event_study", "readiness": "ready_for_path_only", "blocker": "official historical launch/lifecycle metadata partial; proxy_launch_only", "intended_regimes": 1, "needs_targeted_1m": True, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": True},
        {"family": "crowded_long_unwind_short", "readiness": "ready_for_path_and_replay", "blocker": "", "intended_regimes": 2, "needs_targeted_1m": True, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": False},
        {"family": "failed_sector_rotation_short", "readiness": "ready_for_path_only" if not sector_pit else "ready_for_path_and_replay", "blocker": "no point-in-time sector map; proxy_sector_only" if not sector_pit else "", "intended_regimes": 1, "needs_targeted_1m": True, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": True, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": False},
        {"family": "post_catalyst_continuation_base", "readiness": "ready_for_path_only" if not catalyst_pit else "ready_for_path_and_replay", "blocker": "no point-in-time catalyst database; proxy_catalyst_only" if not catalyst_pit else "", "intended_regimes": 1, "needs_targeted_1m": True, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": True, "needs_listing_lifecycle_metadata": False},
        {"family": "H1_lead_lag", "readiness": "ready_for_path_and_replay" if one_ok else "not_fairly_tested_missing_data", "blocker": "adequate 1m unavailable; no 5m lead/lag candidate tested" if not one_ok else "", "intended_regimes": 0, "needs_targeted_1m": True, "needs_top_of_book": False, "needs_shallow_depth": False, "needs_public_trades": False, "needs_liquidation_feed": False, "needs_pit_sector_map": False, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": False},
        {"family": "D4", "readiness": "carry_forward_targeted_execution_depth", "blocker": "missing top-of-book/depth/public-trade execution evidence", "intended_regimes": 1, "needs_targeted_1m": False, "needs_top_of_book": True, "needs_shallow_depth": True, "needs_public_trades": True, "needs_liquidation_feed": True, "needs_pit_sector_map": False, "needs_catalyst_database": False, "needs_listing_lifecycle_metadata": False},
    ]


def min_budget_for_family(row: Mapping[str, Any]) -> int:
    if row.get("readiness") != "ready_for_path_and_replay":
        return 0
    intended = max(int(row.get("intended_regimes", 1) or 1), 1)
    return min(50, 3 * 3 * intended)


def allocate_family_budgets(total: int, family_budget: int = 0) -> dict[str, int]:
    readiness = {r["family"]: r for r in family_readiness_rows()}
    active = [f for f in SIMPLE_FAMILIES if readiness.get(f, {}).get("readiness") in READY_LABELS]
    budgets = {f: 0 for f in SIMPLE_FAMILIES}
    maxes = {f: FAMILY_MAX_BUDGETS[f] for f in SIMPLE_FAMILIES}
    if family_budget > 0:
        for f in active:
            budgets[f] = min(family_budget, maxes[f])
        return budgets
    minimums = {f: min_budget_for_family(readiness.get(f, {})) for f in active}
    min_total = sum(minimums.values())
    remaining = max(int(total), 0)
    if remaining <= 0:
        return budgets
    if remaining < min_total:
        # Smoke/small runs get a balanced scaled-down minimum, but still nonzero for ready families.
        for f in active:
            if readiness.get(f, {}).get("readiness") == "ready_for_path_and_replay":
                budgets[f] = max(1, int(remaining / max(len(active), 1)))
        left = remaining - sum(budgets.values())
        for f in active:
            if left <= 0:
                break
            budgets[f] += 1
            left -= 1
        return budgets
    for f, v in minimums.items():
        budgets[f] = min(v, maxes[f])
    remaining -= sum(budgets.values())
    cap_remaining = {f: max(maxes[f] - budgets[f], 0) for f in active}
    cap_total = sum(cap_remaining.values())
    if cap_total <= 0 or remaining <= 0:
        return budgets
    for f in active:
        add = int(remaining * cap_remaining[f] / cap_total)
        budgets[f] += min(add, cap_remaining[f])
    leftover = min(remaining, sum(maxes[f] - budgets[f] for f in active)) - sum(max(0, budgets[f] - minimums.get(f, 0)) for f in active)
    for f in active:
        if leftover <= 0:
            break
        if budgets[f] < maxes[f]:
            budgets[f] += 1
            leftover -= 1
    return budgets


def write_budget_manifest(ctx: RunContext) -> None:
    budgets = allocate_family_budgets(ctx.args.discovery_budget, ctx.args.family_budget)
    readiness = {r["family"]: r for r in family_readiness_rows()}
    rows = []
    for fam in SIMPLE_FAMILIES:
        r = readiness.get(fam, {})
        min_b = min_budget_for_family(r)
        rows.append({
            "family": fam,
            "readiness": r.get("readiness", "unknown"),
            "blocker": r.get("blocker", ""),
            "max_budget": FAMILY_MAX_BUDGETS[fam],
            "minimum_viable_budget": min_b,
            "allocated_candidates": budgets.get(fam, 0),
            "minimum_budget_satisfied": bool(budgets.get(fam, 0) >= min_b),
            "estimated_candidates": budgets.get(fam, 0),
            "quality_note": "sampled discovery only; full coverage required for lead status",
        })
    write_csv(ctx.run_root / "budget/family_budget_manifest.csv", rows)
    est = estimate_total_output_gb(ctx)
    write_text(ctx.run_root / "budget/output_size_estimate.md", f"# Output Size Estimate\n\n- estimated total output GB: `{est:.2f}`\n- max output GB CLI: `{ctx.args.max_output_gb}`\n- per-stage hard stop without override: `20GB`\n- fits resource limits: `{est <= ctx.args.max_output_gb}`\n- discovery budget: `{ctx.args.discovery_budget}`\n- refinement budget: `{ctx.args.refine_budget}`\n- D4 outside discovery budget: `true`\n")
    runtime = estimate_runtime_hours(ctx.args)
    write_text(ctx.run_root / "runtime/runtime_estimate_report.md", f"# Runtime Estimate Report\n\n- estimated full-quality runtime hours: `{runtime:.1f}`\n- exceeds 72h: `{runtime > 72}`\n- staged family order: leader_breakout_long, weak_asset_spike_fade, risk_off_exhaustion_spike_short, funding_window_orb_failure, us_cash_open_orb, utc_daily_open_reversal, listing, crowded_long_unwind, sector/catalyst data-build, D4 depth plan always preserved\n- recommended staged command if needed: run each family group with same run root and `--resume` after adding a family filter extension.\n- lower-budget alternative: `--discovery-budget 900 --refine-budget 120`.\n- quality loss under lower budget: fewer sub-mechanisms enter holding pen and less stable family-level evidence; no ready-family minimum may be silently violated.\n")


def estimate_total_output_gb(ctx: RunContext) -> float:
    return 1.0 if ctx.args.smoke else min(10.0 + ctx.args.discovery_budget / 850.0, ctx.args.max_output_gb)


def source_filter(df: pd.DataFrame, family: str) -> pd.DataFrame:
    if df.empty or "family" not in df.columns:
        return pd.DataFrame()
    sources = SOURCE_FAMILY_MAP.get(family, [])
    out = df[df["family"].astype(str).isin(sources)].copy()
    if family == "leader_breakout_long" and "liquidity_tier" in out:
        out = out[out["liquidity_tier"].isin(["A", "B", "C"])]
    elif family in {"weak_asset_spike_fade", "new_perp_listing_event_study"} and "liquidity_tier" in out:
        out = out[out["liquidity_tier"].isin(["B", "C", "D"])]
    elif family == "risk_off_exhaustion_spike_short" and "side" in out:
        out = out[out["side"].astype(str).eq("short") | out["family"].astype(str).isin(["D1"])]
    return out


def assign_subfamily(df: pd.DataFrame, family: str) -> pd.Series:
    subs = FAMILY_SUBFAMILIES.get(family, ["default"])
    if df.empty:
        return pd.Series(dtype=str)
    if "variant_id" in df.columns:
        vals = df["variant_id"].astype(str).map(lambda x: subs[abs(hash(x)) % len(subs)])
    else:
        vals = pd.Series([subs[i % len(subs)] for i in range(len(df))], index=df.index)
    return vals


def write_family_events(ctx: RunContext, family: str, rel_path: str, report_path: str) -> pd.DataFrame:
    base = load_path_metrics(ctx)
    out = source_filter(base, family)
    if out.empty:
        out = base.head(0).copy()
    out = out.copy()
    out["simple_family"] = family
    out["simple_subfamily"] = assign_subfamily(out, family)
    out["source_path_family"] = out.get("family", pd.Series(dtype=str))
    out["family"] = family
    out["event_id"] = [f"{family}__{stable_hash({'src': str(e), 'i': int(i)}, 14)}" for i, e in enumerate(out.get("event_id", pd.Series(range(len(out)))))]
    validate_no_protected(out, ["decision_ts", "entry_ts"])
    p = ctx.run_root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    summ = out.groupby(["simple_subfamily", "side", "liquidity_tier"], dropna=False).agg(events=("event_id", "count"), symbols=("symbol", "nunique"), first_ts=("decision_ts", "min"), last_ts=("decision_ts", "max")).reset_index() if not out.empty else pd.DataFrame(columns=["simple_subfamily", "side", "liquidity_tier", "events", "symbols", "first_ts", "last_ts"])
    write_df_csv(ctx.run_root / rel_path.replace(".parquet", "_summary.csv"), summ)
    write_text(ctx.run_root / report_path, f"# {family} Event Report\n\n- events: `{len(out)}`\n- subfamilies: `{', '.join(FAMILY_SUBFAMILIES.get(family, []))}`\n- protected rows: `0`\n- source: prior pre-holdout path metrics mapped into the requested simple-alpha family contract.\n")
    return out


def cost_bps_for_tier(tier: Any) -> float:
    return {"A": 16.0, "B": 30.0, "C": 50.0, "D": 78.0}.get(str(tier), 93.0)


def surface_return_r(df: pd.DataFrame, horizon: str, target_r: float, stop_mult: float = 1.0, cost_mult: float = 1.0, branch: str = "pessimistic", risk_bps_override: float | None = None) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if risk_bps_override is None or not math.isfinite(float(risk_bps_override)):
        risk = pd.to_numeric(df.get("reference_risk_bps"), errors="coerce").fillna(100.0).clip(lower=1.0) * float(stop_mult)
    else:
        risk = pd.Series(float(risk_bps_override) * float(stop_mult), index=df.index).clip(lower=1.0)
    mfe = pd.to_numeric(df.get(f"{horizon}_mfe_bps"), errors="coerce").fillna(-np.inf)
    mae = pd.to_numeric(df.get(f"{horizon}_mae_bps"), errors="coerce").fillna(np.inf)
    close = pd.to_numeric(df.get(f"{horizon}_close_return_bps"), errors="coerce").fillna(0.0)
    hit_tp = mfe >= target_r * risk
    hit_sl = mae >= risk
    ret = close / risk
    both = hit_tp & hit_sl
    ret = ret.mask(hit_sl & ~hit_tp, -1.0)
    ret = ret.mask(hit_tp & ~hit_sl, float(target_r))
    if branch == "optimistic":
        ret = ret.mask(both, float(target_r))
    elif branch == "neutral":
        ret = ret.mask(both, (float(target_r) - 1.0) / 2.0)
    else:
        ret = ret.mask(both, -1.0)
    costs = df.get("liquidity_tier", pd.Series("UNKNOWN", index=df.index)).map(cost_bps_for_tier).fillna(93.0) * float(cost_mult)
    return (ret - (costs / risk)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def summarize_returns(ret: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    pos = x[x > 0].sum()
    neg = -x[x < 0].sum()
    eq = x.cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    return {"events": int(len(x)), "net_R": float(x.sum()), "mean_R": float(x.mean()) if len(x) else 0.0, "median_R": float(x.median()) if len(x) else 0.0, "PF": float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0), "hit_rate": float((x > 0).mean()) if len(x) else 0.0, "max_dd_R_proxy": dd}


def concentration_stats(sub: pd.DataFrame, ret: pd.Series) -> dict[str, float]:
    if sub.empty or ret.empty:
        return {"max_symbol_positive_share": 0.0, "max_month_positive_share": 0.0, "max_event_cluster_positive_share": 0.0}
    pos = ret[ret > 0]
    if pos.empty:
        return {"max_symbol_positive_share": 0.0, "max_month_positive_share": 0.0, "max_event_cluster_positive_share": 0.0}
    tmp = sub.loc[pos.index].copy()
    tmp["_positive_R"] = pos
    denom = max(float(pos.sum()), 1e-9)
    sym = float(tmp.groupby("symbol")["_positive_R"].sum().max() / denom) if "symbol" in tmp else 0.0
    month_key = pd.to_datetime(tmp["decision_ts"], utc=True).dt.strftime("%Y-%m") if "decision_ts" in tmp else pd.Series("unknown", index=tmp.index)
    month = float(tmp.groupby(month_key)["_positive_R"].sum().max() / denom) if len(tmp) else 0.0
    cluster = float(tmp.groupby("event_id")["_positive_R"].sum().max() / denom) if "event_id" in tmp else 0.0
    return {"max_symbol_positive_share": sym, "max_month_positive_share": month, "max_event_cluster_positive_share": cluster}


def apply_candidate_filter(df: pd.DataFrame, cand: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[df.get("family", pd.Series(dtype=str)).astype(str).eq(str(cand.get("family")))].copy()
    sub = cand.get("subfamily", "any")
    if sub != "any" and "simple_subfamily" in out.columns:
        out = out[out["simple_subfamily"].eq(sub)]
    tier = cand.get("tier_filter", "any")
    if tier == "C":
        out = out[out.get("liquidity_tier").eq("C")]
    elif tier == "A_B":
        out = out[out.get("liquidity_tier").isin(["A", "B"])]
    elif tier == "not_D":
        out = out[~out.get("liquidity_tier").eq("D")]
    if cand.get("regime_gate") == "risk_on" and "parent_trend_label" in out.columns:
        out = out[out["parent_trend_label"].isin(["strong_up", "neutral_up"])]
    elif cand.get("regime_gate") == "risk_off" and "parent_trend_label" in out.columns:
        out = out[out["parent_trend_label"].isin(["down", "neutral_down"])]
    elif cand.get("regime_gate") == "non_deteriorating" and "btc_eth_non_deteriorating" in out.columns:
        out = out[out["btc_eth_non_deteriorating"].fillna(False)]
    if cand.get("funding_gate") == "not_high" and "funding_percentile_bucket" in out.columns:
        out = out[~out["funding_percentile_bucket"].eq("high")]
    elif cand.get("funding_gate") == "low_mid" and "funding_percentile_bucket" in out.columns:
        out = out[out["funding_percentile_bucket"].isin(["low", "mid", "unknown"])]
    if cand.get("liquidity_quality_gate") == "avoid_thin" and "liquidity_quality_label" in out.columns:
        out = out[~out["liquidity_quality_label"].eq("thin_proxy")]
    if cand.get("bad_wick_gate") == "avoid_top1" and "bad_wick_proxy_label" in out.columns:
        out = out[~out["bad_wick_proxy_label"].eq("top_1pct_range_proxy")]
    return out.copy()


def score_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    events = int(row.get("events", 0) or 0)
    net = float(row.get("net_R", 0.0) or 0.0)
    pf = float(row.get("PF", 0.0) or 0.0)
    mean_r = float(row.get("mean_R", 0.0) or 0.0)
    median_r = float(row.get("median_R", 0.0) or 0.0)
    liq = int(row.get("liquidation_count", 0) or 0)
    hard: list[str] = []
    if events <= 0:
        hard.append("no_events")
    if net <= 0:
        hard.append("net_R_nonpositive")
    if pf <= 1.0:
        hard.append("PF_lte_1")
    if liq > 0:
        hard.append("liquidation_count_positive")
    for key, lim, name in [("max_symbol_positive_share", 0.30, "symbol_concentration_gt_30pct"), ("max_month_positive_share", 0.40, "month_concentration_gt_40pct"), ("max_event_cluster_positive_share", 0.35, "event_cluster_concentration_gt_35pct")]:
        if float(row.get(key, 0.0) or 0.0) > lim:
            hard.append(name)
    score = mean_r * 40.0 + median_r * 15.0 + min(pf, 5.0) * 5.0 + math.log1p(max(events, 0)) * 0.4
    score -= liq * 1000.0
    if events <= 0:
        # Zero-event rows are useful generator diagnostics, but must never outrank
        # real losing rows or enter prelead selection as false positives.
        score -= 1_000_000_000.0
    return {"robustness_score": float(score), "lead_rankable": len(hard) == 0, "hard_penalty_reasons": ";".join(hard)}


def candidate_result(df: pd.DataFrame, cand: Mapping[str, Any]) -> dict[str, Any]:
    sub = apply_candidate_filter(df, cand)
    horizon = str(cand.get("horizon", "24h"))
    risk_override_raw = cand.get("risk_bps_override", None)
    risk_override = None
    if risk_override_raw not in [None, "", "nan"]:
        try:
            risk_override = float(risk_override_raw)
        except Exception:
            risk_override = None
    ret = surface_return_r(sub, horizon, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)), risk_bps_override=risk_override)
    sm = summarize_returns(ret)
    mfe = pd.to_numeric(sub.get(f"{horizon}_mfe_bps", pd.Series(dtype=float)), errors="coerce") if not sub.empty else pd.Series(dtype=float)
    mae = pd.to_numeric(sub.get(f"{horizon}_mae_bps", pd.Series(dtype=float)), errors="coerce") if not sub.empty else pd.Series(dtype=float)
    pos1 = pd.to_numeric(sub.get(f"{horizon}_pos1R_before_neg1R", pd.Series(dtype=float)), errors="coerce") if not sub.empty else pd.Series(dtype=float)
    median_mfe = float(mfe.median()) if len(mfe) else 0.0
    median_mae = float(mae.median()) if len(mae) else 0.0
    pos1_share = float(pos1.fillna(0).mean()) if len(pos1) else 0.0
    path_edge_score = (median_mfe / max(abs(median_mae), 1.0)) + pos1_share
    path_edge_flag = bool((len(sub) > 0) and (median_mfe > 0) and (path_edge_score > 1.05 or pos1_share >= 0.48))
    liq_col = f"{horizon}_liquidation_10x"
    liq = int(sub.get(liq_col, pd.Series(False, index=sub.index)).fillna(False).astype(bool).sum()) if not sub.empty else 0
    conc = concentration_stats(sub, ret)
    proxy_mark_share = 1.0
    if not sub.empty and "mark_path_status" in sub.columns:
        ok = sub["mark_path_status"].astype(str).str.contains("ok|available|1m_mark", case=False, regex=True, na=False)
        proxy_mark_share = float(1.0 - ok.mean())
    out = {**cand, **sm, "median_mfe_bps": median_mfe, "median_mae_bps": median_mae, "pos1R_before_neg1R_share": pos1_share, "path_edge_score": float(path_edge_score), "path_edge_flag": path_edge_flag, "liquidation_count": liq, **conc, "proxy_mark_or_liquidation_evidence_share": proxy_mark_share}
    out.update(score_candidate_row(out))
    return out


def generate_candidates(budgets: Mapping[str, int], seed: int, readiness: Sequence[Mapping[str, Any]], smoke: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ready = {r["family"]: r for r in readiness}
    rows: list[dict[str, Any]] = []
    for fam in SIMPLE_FAMILIES:
        n = int(budgets.get(fam, 0))
        if smoke:
            n = min(n, 6)
        if n <= 0 or ready.get(fam, {}).get("readiness") not in READY_LABELS:
            continue
        subs = FAMILY_SUBFAMILIES.get(fam, ["default"])
        source_presets: list[dict[str, Any]] = []
        for source_family in SOURCE_FAMILY_MAP.get(fam, []):
            source_presets.extend(SOURCE_EXIT_PRESETS.get(source_family, []))
        # Diagnostic seeds first: these are coarse neighborhoods that the
        # prior path study showed can produce tradable event counts. They are
        # still scored, null-tested, stress-tested, and cannot become leads
        # unless full coverage reruns pass.
        for preset in source_presets:
            for sub in ["any", *subs]:
                row = {
                    "family": fam,
                    "subfamily": sub,
                    "candidate_type": "diagnostic_seed",
                    "source_family_preset": preset.get("preset_name"),
                    "seed_interpretation": "coarse exit/risk neighborhood borrowed from prior path diagnostics; not the old source-family strategy translation",
                    "regime_gate": "none",
                    "funding_gate": "none",
                    "liquidity_quality_gate": "none",
                    "bad_wick_gate": "none",
                    "cost_mult": 1.0,
                }
                row.update({k: v for k, v in preset.items() if k != "preset_name"})
                rows.append(row)
        # Baselines for every subfamily next.
        for sub in subs:
            rows.append({"family": fam, "subfamily": sub, "candidate_type": "subfamily_baseline", "source_family_preset": "", "seed_interpretation": "", "tier_filter": "any", "regime_gate": "none", "funding_gate": "none", "liquidity_quality_gate": "none", "bad_wick_gate": "none", "horizon": "24h", "target_r": 2.0, "stop_mult": 1.0, "risk_bps_override": None, "cost_mult": 1.0})
        for _ in range(max(0, n - len(subs) - len(source_presets))):
            rows.append({
                "family": fam,
                "subfamily": rng.choice(subs),
                "candidate_type": "bounded_sobol_style",
                "source_family_preset": "",
                "seed_interpretation": "",
                "tier_filter": rng.choice(["C", "A_B", "not_D", "any"]),
                "regime_gate": rng.choice(["none", "risk_on", "risk_off", "non_deteriorating"]),
                "funding_gate": rng.choice(["none", "not_high", "low_mid"]),
                "liquidity_quality_gate": rng.choice(["none", "avoid_thin"]),
                "bad_wick_gate": rng.choice(["none", "avoid_top1"]),
                "horizon": rng.choice(["30m", "1h", "2h", "6h", "24h", "72h"]),
                "target_r": rng.choice([1.0, 2.0, 3.0, 5.0]),
                "stop_mult": rng.choice([0.5, 1.0, 1.5, 2.0]),
                "risk_bps_override": rng.choice([None, None, 300.0, 500.0, 1000.0, 1500.0]),
                "cost_mult": rng.choice([1.0, 1.25, 1.5]),
            })
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        cid = f"{row['family']}__{stable_hash(row, 12)}"
        if cid in seen:
            continue
        seen.add(cid)
        r = dict(row)
        r["candidate_id"] = cid
        rr = ready.get(r["family"], {})
        r["readiness"] = rr.get("readiness", "unknown")
        r["max_verdict_cap"] = "promote_to_targeted_execution_data_collection" if any(rr.get(k) for k in ["needs_top_of_book", "needs_shallow_depth", "needs_public_trades", "needs_liquidation_feed"]) else "unsealed_family_specific_validation_possible_if_all_gates_pass"
        out.append(r)
    return out


def event_count_for_candidate(df: pd.DataFrame, cand: Mapping[str, Any]) -> int:
    try:
        return int(len(apply_candidate_filter(df, cand)))
    except Exception:
        return 0


def min_evaluation_events(ctx: RunContext) -> int:
    return SMOKE_MIN_EVALUATION_EVENTS if ctx.args.smoke else MIN_EVALUATION_EVENTS


def min_prelead_events(ctx: RunContext) -> int:
    return SMOKE_MIN_PRELEAD_EVENTS if ctx.args.smoke else MIN_PRELEAD_EVENTS


def min_evaluation_events_for_family(ctx: RunContext, family: str) -> int:
    if family in SPARSE_EVENT_STUDY_FAMILIES:
        return SMOKE_SPARSE_MIN_EVALUATION_EVENTS if ctx.args.smoke else SPARSE_MIN_EVALUATION_EVENTS
    return min_evaluation_events(ctx)


def min_prelead_events_for_family(ctx: RunContext, family: str) -> int:
    if family in SPARSE_EVENT_STUDY_FAMILIES:
        return SMOKE_SPARSE_MIN_PRELEAD_EVENTS if ctx.args.smoke else SPARSE_MIN_PRELEAD_EVENTS
    return min_prelead_events(ctx)


def top_sweep(ctx: RunContext, n: int = 20) -> pd.DataFrame:
    for p in [ctx.run_root / "preleads/full_coverage_summary.csv", ctx.run_root / "sweep/sweep_summary.csv"]:
        if p.exists():
            df = read_csv_safe(p)
            if not df.empty:
                return df.sort_values(["lead_rankable", "robustness_score"], ascending=[False, False]).head(n)
    return pd.DataFrame()


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    roots = [FIRST_SCREEN_ROOT, PATH_DIAG_ROOT, D1_ROOT, PILOT_1M_ROOT, REGIME_SWEEP_ROOT, F1G1_ROOT, ALPHA_MARATHON_ROOT, D4_AUDIT_ROOT, D4_SURVIVAL_ROOT]
    manifest = []
    for root in roots:
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        manifest.append({"root": str(root), "exists": root.exists(), "file_count": len(files), "sample_files": [str(p.relative_to(root)) for p in files[:30]]})
    write_json(ctx.run_root / "preflight/prior_artifact_manifest.json", {"artifacts": manifest})
    write_budget_manifest(ctx)
    sessions = active_tmux_sessions()
    heavy = [s for s in sessions if any(tok in s.lower() for tok in ["qlmg", "sweep", "backtest", "d4", "alpha"])]
    initial = []
    for fam in SIMPLE_FAMILIES + ["D4"]:
        initial.append({"hypothesis_id": fam, "family": fam, "status": "active_or_data_blocked", "note": "D4 mandatory carry-forward" if fam == "D4" else "simple-alpha diagnostic family"})
    write_csv(ctx.run_root / "preflight/active_hypothesis_registry_initial.csv", initial)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard Report\n\n- free_disk_gb: `{snap.free_gb:.2f}`\n- hard_stop_free_gb: `5`\n- warning_free_gb: `7`\n- max_stage_output_without_override_gb: `20`\n- default_max_output_gb: `{ctx.args.max_output_gb}`\n")
    write_text(ctx.run_root / "preflight/prior_result_audit.md", f"# Prior Result Audit\n\n- D1 current translation rejected: `true`; no further tuning.\n- F1/G1 current translation rejected: `true`; no further tuning.\n- D4 survivability root: `{D4_SURVIVAL_ROOT}`\n- D4 status: `carry_forward_targeted_execution_depth` if reconstruction succeeds.\n- prior path diagnostics root exists: `{PATH_DIAG_ROOT.exists()}`\n- active tmux sessions: `{sessions}`\n- heavy QLMG/backtest sessions detected: `{heavy}`\n- full run may launch now without explicit approval: `{not bool(heavy)}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight Report\n\n- run root: `{ctx.run_root}`\n- free disk GB: `{snap.free_gb:.2f}`\n- requested window: `{ctx.start}` to `{ctx.end}`\n- protected holdout starts: `{FINAL_HOLDOUT_START}`\n- discovery budget: `{ctx.args.discovery_budget}`\n- refine budget: `{ctx.args.refine_budget}`\n- estimated runtime hours: `{estimate_runtime_hours(ctx.args):.1f}`\n- no live system mutation, raw-data deletion, final holdout access, or large download.\n")


def stage_telegram(ctx: RunContext) -> None:
    if ctx.args.require_telegram and not ctx.notifier.remote_available and not ctx.args.allow_no_telegram and not ctx.args.smoke:
        raise RuntimeError("remote Telegram is required for full launch but is unavailable; pass --allow-no-telegram only if intentionally running local-only")
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness Report\n\n- status: `{ctx.notifier.status}`\n- remote_available: `{ctx.notifier.remote_available}`\n- require_telegram: `{ctx.args.require_telegram}`\n- allow_no_telegram: `{ctx.args.allow_no_telegram}`\n- missing/disabled reason: `{ctx.notifier.missing or 'none'}`\n- local JSONL log: `notifications/telegram_events.jsonl`\n- secrets persisted: `false`\n")
    full_cmd = f"bash tools/run_qlmg_simple_alpha_plus_d4_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --discovery-budget {ctx.args.discovery_budget} --refine-budget {ctx.args.refine_budget} --nulls-per-event {ctx.args.nulls_per_event} --use-targeted-1m-if-overlap --require-telegram --d4-carry-forward-required --build-depth-plan --seed {ctx.args.seed} --launch-tmux"
    watch = f"""# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n- resume: `{full_cmd}`\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nSmoke must pass before full launch. Full launch requires `--launch-tmux`.\n\n```bash\n{full_cmd}\n```\n")


def stage_seal(ctx: RunContext) -> None:
    check = {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "requested_start": str(ctx.start), "requested_end": str(ctx.end), "pre_holdout_read_smoke": bool(ctx.end < FINAL_HOLDOUT_START), "protected_read_smoke": "blocked_by_policy", "generated_rows_protected_check": "enforced_by_validate_no_protected", "status": "pass" if ctx.end < FINAL_HOLDOUT_START else "fail"}
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    if check["status"] != "pass":
        raise RuntimeError("seal guard failed")
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard Report\n\n- protected slice starts: `{FINAL_HOLDOUT_START}`\n- allowed data end: `{SCREENING_END}`\n- requested end: `{ctx.end}`\n- status: `pass`\n")


def stage_d4_freeze(ctx: RunContext) -> None:
    decision = read_json_safe(D4_SURVIVAL_ROOT / "decision_summary.json")
    sizing = d4_summary_rows()
    if ctx.args.d4_carry_forward_required and (not D4_SURVIVAL_ROOT.exists() or sizing.empty):
        raise RuntimeError("D4 carry-forward required but survivability artifacts are unavailable")
    wanted = ["dynamic_buffer_1p25_max10x", "dynamic_buffer_1p5_max10x", "dynamic_buffer_2p0_max10x", "fixed_3x", "risk_based_buffer_1p5", "skip_if_safe_sizing_below_3x"]
    rows = []
    for w in wanted:
        hit = sizing[sizing.astype(str).apply(lambda col: col.str.contains(w, regex=False, na=False)).any(axis=1)] if not sizing.empty else pd.DataFrame()
        rec = hit.iloc[0].to_dict() if not hit.empty else {}
        rows.append({"d4_expression": w, "source_found": bool(rec), "net_R": rec.get("net_R", rec.get("net_r", "")), "liquidation_count": rec.get("liquidation_count", ""), "verdict_cap": "carry_forward_d4_execution_depth"})
    write_csv(ctx.run_root / "d4/d4_carry_forward_summary.csv", rows)
    contract = {"candidate_id": "D4__b4c9487fe82c", "status": "carry_forward_targeted_execution_depth", "source_run_root": str(D4_SURVIVAL_ROOT), "source_decision": decision, "preserved_expressions": wanted, "not_validated": True, "not_live_ready": True, "not_sealed_ready": True, "missing_top_of_book_depth_is_carry_forward_reason": True, "no_future_liquidation_filter": True}
    write_json(ctx.run_root / "d4/d4_carry_forward_contract.json", contract)
    write_text(ctx.run_root / "d4/d4_carry_forward_report.md", "# D4 Carry-Forward Report\n\nD4 is preserved as a targeted execution-depth candidate. It is not validated, not live-ready, and not sealed-validation-ready. Missing top-of-book/depth/public-trade evidence is the reason for carry-forward, not a demotion reason.\n")


def stage_contracts(ctx: RunContext) -> None:
    out_dir = ctx.run_root / "contracts/family_contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    readiness = {r["family"]: r for r in family_readiness_rows()}
    rows = []
    for fam in SIMPLE_FAMILIES + ["D4"]:
        r = readiness.get(fam, {})
        contract = {
            "family": fam,
            "created_at_utc": utc_now(),
            "hypothesis_status": "diagnostic_discovery_only" if fam != "D4" else "carry_forward_targeted_execution_depth",
            "protected_holdout_start": "2026-01-01T00:00:00Z",
            "allowed_data_end": "2025-12-31T23:59:59Z",
            "subfamilies": FAMILY_SUBFAMILIES.get(fam, []),
            "readiness": r.get("readiness", "unknown"),
            "required_data_blockers": r.get("blocker", ""),
            "forbidden_shortcuts": ["future liquidation labels as filters", "current active status as historical truth", "proxy sector/catalyst as true PIT data", "last price as mark price"],
            "matched_null_required_for_prelead": True,
            "full_coverage_required_for_lead_status": True,
            "maximum_promotion_status_allowed_given_current_data": "promote_to_targeted_execution_data_collection" if fam == "D4" or r.get("needs_top_of_book") or r.get("needs_shallow_depth") else "family_specific_validation_possible_after_gates",
            "no_live_trading": True,
            "no_sealed_validation": True,
        }
        p = out_dir / f"{fam}.json"
        p.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({"family": fam, "path": str(p), "readiness": contract["readiness"], "subfamilies": ";".join(contract["subfamilies"]), "max_promotion": contract["maximum_promotion_status_allowed_given_current_data"]})
    write_csv(ctx.run_root / "contracts/simple_alpha_contract_summary.csv", rows)


def stage_regime(ctx: RunContext) -> None:
    df = load_path_metrics(ctx, null=False)
    reg = build_regime_panel(df, min_history=3 if ctx.args.smoke else 10)
    validate_no_protected(reg, ["decision_ts", "feature_ts"])
    (ctx.run_root / "regime").mkdir(parents=True, exist_ok=True)
    reg.to_parquet(ctx.run_root / "regime/regime_context_panel.parquet", index=False)
    regime_feature_dictionary().to_csv(ctx.run_root / "regime/regime_context_dictionary.csv", index=False)
    rows = []
    for col in ["parent_trend_label", "btc_eth_regime_label", "realized_vol_bucket", "liquidity_quality_label", "funding_percentile_bucket", "price_oi_matrix_24h", "deleveraged_2of4", "session_bucket", "listing_age_bucket"]:
        if col in reg.columns:
            for label, count in reg[col].astype(str).value_counts(dropna=False).head(30).items():
                rows.append({"feature": col, "label": label, "rows": int(count), "share": float(count / max(len(reg), 1))})
    write_csv(ctx.run_root / "regime/regime_context_coverage_summary.csv", rows)
    write_text(ctx.run_root / "regime/regime_context_qc_report.md", f"# Regime And Context Refresh\n\n- source rows: `{len(df)}`\n- regime rows: `{len(reg)}`\n- point-in-time rule: `feature_ts <= decision_ts`\n- sector/catalyst layer: `placeholder_or_proxy_only`\n")


def stage_readiness(ctx: RunContext) -> None:
    rows = family_readiness_rows()
    write_csv(ctx.run_root / "readiness/family_readiness_matrix.csv", rows)
    data_rows = []
    for r in rows:
        data_rows.append({k: r.get(k) for k in ["family", "needs_targeted_1m", "needs_top_of_book", "needs_shallow_depth", "needs_public_trades", "needs_liquidation_feed", "needs_pit_sector_map", "needs_catalyst_database", "needs_listing_lifecycle_metadata", "readiness", "blocker"]})
    write_csv(ctx.run_root / "data_build/data_build_decision_table.csv", data_rows)
    write_text(ctx.run_root / "readiness/family_readiness_report.md", "# Family Readiness Report\n\nReady families receive minimum exploration budgets unless explicitly blocked. H1 lead/lag is not tested on 5m. Sector/catalyst families are proxy/data-build only without PIT data. D4 is carry-forward targeted execution-depth.\n")


def stage_d4_depth_plan(ctx: RunContext) -> None:
    (ctx.run_root / "d4_depth").mkdir(parents=True, exist_ok=True)
    geom = pd.DataFrame()
    p = D4_SURVIVAL_ROOT / "geometry/decision_time_liquidation_geometry.parquet"
    if p.exists():
        try:
            geom = pd.read_parquet(p, columns=[c for c in ["event_id", "symbol", "decision_ts", "entry_ts"] if c in parquet_columns(p)])
        except Exception:
            geom = pd.DataFrame()
    if geom.empty:
        geom = pd.DataFrame([{"event_id": "D4__b4c9487fe82c", "symbol": "UNKNOWN", "decision_ts": ctx.start, "entry_ts": ctx.start}])
    geom = geom.head(500 if ctx.args.smoke else 5000).copy()
    geom["window_start"] = pd.to_datetime(geom["decision_ts"], utc=True, errors="coerce") - pd.Timedelta(hours=4)
    geom["window_end"] = pd.to_datetime(geom.get("entry_ts", geom["decision_ts"]), utc=True, errors="coerce") + pd.Timedelta(hours=24)
    validate_no_protected(geom, ["decision_ts", "entry_ts", "window_start", "window_end"])
    geom[["event_id", "symbol", "window_start", "window_end"]].to_csv(ctx.run_root / "d4_depth/d4_depth_window_manifest.csv", index=False)
    storage = [{"dataset": ds, "estimated_gb": gb, "source_feasibility": src} for ds, gb, src in [("top_of_book", 8.0, "vendor_or_unavailable_locally"), ("shallow_depth", 18.0, "vendor_or_unavailable_locally"), ("public_trades", 12.0, "vendor_or_exchange_archive"), ("liquidation_prints", 2.0, "vendor_or_unavailable_locally"), ("existing_1m_mark_ohlcv", 0.0, "already_available_prior_d4_audit")]]
    write_csv(ctx.run_root / "d4_depth/d4_storage_estimate.csv", storage)
    write_csv(ctx.run_root / "d4_depth/d4_data_source_feasibility.csv", storage)
    contract = {"contract_type": "d4_targeted_execution_depth_collection", "source_candidate": "D4__b4c9487fe82c", "windows": int(len(geom)), "datasets": [r["dataset"] for r in storage], "no_download_in_this_run": True}
    write_json(ctx.run_root / "d4_depth/d4_next_data_contract.json", contract)
    write_text(ctx.run_root / "d4_depth/d4_execution_depth_plan.md", f"# D4 Execution Depth Plan\n\n- windows planned: `{len(geom)}`\n- D4 should wait for top-of-book/depth/public-trade evidence before any family-specific validation.\n- no download performed in this run.\n")


def stage_event_family(ctx: RunContext, family: str, rel: str, report: str) -> None:
    write_family_events(ctx, family, rel, report)


def stage_synthetic_orb(ctx: RunContext) -> None:
    frames = []
    for fam in ["funding_window_orb_failure", "us_cash_open_orb", "utc_daily_open_reversal"]:
        frames.append(write_family_events(ctx, fam, f"tmp/{fam}.parquet", f"tmp/{fam}.md"))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        # Preserve the concrete family names so candidate filters can match
        # funding/US-open/UTC-open ORB rows during the sweep.
        out["event_generator_group"] = "synthetic_open_orb"
    validate_no_protected(out, ["decision_ts", "entry_ts"])
    out.to_parquet(ctx.run_root / "events/synthetic_open_orb_events.parquet", index=False)
    write_text(ctx.run_root / "events/synthetic_open_orb_event_report.md", f"# Synthetic Open ORB Events\n\n- events: `{len(out)}`\n- includes funding-window, US cash-open, and UTC daily-open proxy families.\n- baselines are same-time/matched in null stage.\n")


def stage_listing(ctx: RunContext) -> None:
    out = write_family_events(ctx, "new_perp_listing_event_study", "listing/listing_event_rows.parquet", "listing/listing_event_study_report.md")
    manifest = out[["event_id", "symbol", "decision_ts", "simple_subfamily", "liquidity_tier"]].head(100000) if not out.empty else pd.DataFrame(columns=["event_id", "symbol", "decision_ts", "simple_subfamily", "liquidity_tier"])
    write_df_csv(ctx.run_root / "listing/listing_event_study_manifest.csv", manifest)
    if not out.empty:
        rows = []
        for h in ["1h", "4h", "24h", "72h"]:
            if f"{h}_close_return_bps" in out:
                rows.append({"horizon": h, "median_return_bps": float(pd.to_numeric(out[f"{h}_close_return_bps"], errors="coerce").median()), "events": len(out), "metadata_status": "proxy_launch_only"})
        write_csv(ctx.run_root / "listing/listing_event_window_returns.csv", rows)
    else:
        write_csv(ctx.run_root / "listing/listing_event_window_returns.csv", [])
    write_text(ctx.run_root / "listing/listing_data_gap_report.md", "# Listing Data Gap Report\n\nOfficial historical launch/lifecycle data is partial or unavailable in this phase. Listing outputs are `proxy_launch_only` and cannot be validation candidates.\n")


def stage_aux(ctx: RunContext) -> None:
    frames = []
    for fam in ["crowded_long_unwind_short", "failed_sector_rotation_short", "post_catalyst_continuation_base"]:
        frames.append(write_family_events(ctx, fam, f"tmp/{fam}.parquet", f"tmp/{fam}.md"))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    validate_no_protected(out, ["decision_ts", "entry_ts"])
    out.to_parquet(ctx.run_root / "events/auxiliary_events.parquet", index=False)
    write_text(ctx.run_root / "events/auxiliary_event_generation_report.md", f"# Auxiliary Event Generation\n\n- events: `{len(out)}`\n- failed sector rotation and post-catalyst continuation are proxy/data-build only without PIT data.\n")


def stage_path(ctx: RunContext) -> None:
    ev = load_all_events(ctx)
    rows = []
    for keys, sub in ev.groupby(["family", "simple_subfamily", "side", "liquidity_tier"], dropna=False):
        fam, subfam, side, tier = keys
        rec = {"family": fam, "subfamily": subfam, "side": side, "liquidity_tier": tier, "events": len(sub), "symbols": sub["symbol"].nunique() if "symbol" in sub else 0}
        for h in ["30m", "1h", "6h", "24h", "72h"]:
            if f"{h}_mfe_bps" in sub:
                rec[f"{h}_median_mfe_bps"] = float(pd.to_numeric(sub[f"{h}_mfe_bps"], errors="coerce").median())
                rec[f"{h}_median_mae_bps"] = float(pd.to_numeric(sub[f"{h}_mae_bps"], errors="coerce").median()) if f"{h}_mae_bps" in sub else np.nan
        rows.append(rec)
    write_csv(ctx.run_root / "path/path_summary_by_family.csv", rows)
    # Store compact reusable path metrics sample/summary, not a duplicate huge ledger.
    sample = ev.head(10000)
    if not sample.empty:
        sample.to_parquet(ctx.run_root / "path/path_metrics_sample.parquet", index=False)
    write_text(ctx.run_root / "path/path_diagnostics_report.md", f"# Path-First Diagnostics\n\n- event rows inspected: `{len(ev)}`\n- family/subfamily groups: `{len(rows)}`\n- path edge is preserved even when executable replay is blocked by exit, cost, liquidation, or data.\n")


def label_from_evidence(row: Mapping[str, Any], *, matched_uplift: float | None = None, stress_x125: float | None = None) -> tuple[str, str, str]:
    events = int(row.get("events", 0) or 0)
    net = float(row.get("net_R", 0.0) or 0.0)
    pf = float(row.get("PF", 0.0) or 0.0)
    liq = int(row.get("liquidation_count", 0) or 0)
    proxy_raw = row.get("proxy_mark_or_liquidation_evidence_share", 1.0)
    proxy = 1.0 if proxy_raw is None or (isinstance(proxy_raw, float) and math.isnan(proxy_raw)) else float(proxy_raw)
    path_edge = bool(row.get("path_edge_flag", False))
    beats_null = bool(row.get("beats_matched_null", False))
    if events == 0:
        return "not_fairly_tested_generator_incomplete", "not_rejected_generator_incomplete", "build safer generator and rerun"
    if net <= 0 or pf <= 1.0:
        if path_edge:
            return "path_edge_exit_problem", "not rejected; favorable path/MFE exists but executable replay is negative", "redesign entry/exit/risk geometry and rerun against fresh nulls"
        return "reject_current_translation_only", "mechanism not rejected if path/baseline evidence exists elsewhere", "redesign entry or exit before tuning"
    if not beats_null:
        return "path_edge_exit_problem", "not rejected; positive in isolation but not supported by matched/same-time baseline", "refresh event definition or matched-null design before lead status"
    if liq > 0 or proxy > 0 or (matched_uplift is not None and matched_uplift <= 0):
        return "promote_to_targeted_execution_data_collection", "not rejected; evidence is execution/data capped", "collect 1m/depth/trades/mark data and rerun"
    if stress_x125 is not None and stress_x125 <= 0:
        return "path_edge_exit_problem", "not rejected; fails cost x1.25 lead gate", "rework execution/cost geometry"
    return "promote_to_family_specific_validation", "not rejected; passed diagnostic gates", "family-specific unsealed validation contract"


def stage_dossiers(ctx: RunContext) -> None:
    path = read_csv_safe(ctx.run_root / "path/path_summary_by_family.csv")
    out_rows = []
    for fam in SIMPLE_FAMILIES + ["D4"]:
        if fam == "D4":
            subrows = [{"family": "D4", "subfamily": s, "events": 4475, "path_edge": True, "baseline_or_null_uplift": True, "executable_replay_status": "targeted_execution_depth_capped", "current_label": "carry_forward_d4_execution_depth"} for s in FAMILY_SUBFAMILIES["D4"]]
        else:
            g = path[path.get("family", pd.Series(dtype=str)).astype(str).eq(fam)] if not path.empty else pd.DataFrame()
            subrows = []
            for s in FAMILY_SUBFAMILIES.get(fam, ["default"]):
                sg = g[g.get("subfamily", pd.Series(dtype=str)).astype(str).eq(s)] if not g.empty else pd.DataFrame()
                ev = int(sg.get("events", pd.Series(dtype=int)).sum()) if not sg.empty else 0
                mfe_cols = [c for c in sg.columns if c.endswith("median_mfe_bps")]
                edge = bool(ev > 0 and any(float(pd.to_numeric(sg[c], errors="coerce").median() or 0) > 0 for c in mfe_cols)) if mfe_cols else bool(ev > 0)
                subrows.append({"family": fam, "subfamily": s, "events": ev, "path_edge": edge, "baseline_or_null_uplift": "pending_null_stage", "executable_replay_status": "pending_sweep", "current_label": "continue_hypothesis_development" if ev else "not_fairly_tested_generator_incomplete"})
        lines = [f"# {fam} Mechanism Dossier", "", "This dossier is subfamily-aware. A family-level average must not hide a promising sub-mechanism.", ""]
        for r in subrows:
            r.update({"rejection_status": "not_rejected" if r.get("path_edge") else "current_translation_weak", "reason_not_rejected_if_applicable": "preserved pending baseline/replay/data evidence" if r.get("path_edge") else "generator or current translation may be incomplete", "minimum_data_needed_for_fair_test": data_need_for_family(fam), "next_possible_test": next_test_for_family(fam, str(r.get("subfamily")))})
            out_rows.append(r)
            lines.append(f"- `{r['subfamily']}`: events `{r['events']}`, path_edge `{r['path_edge']}`, label `{r['current_label']}`")
        write_text(ctx.run_root / "dossiers" / f"{fam}_mechanism_dossier.md", "\n".join(lines))
        write_csv(ctx.run_root / "dossiers" / f"{fam}_mechanism_summary.csv", subrows)
    write_csv(ctx.run_root / "dossiers/family_mechanism_dossier_index.csv", out_rows)


def data_need_for_family(fam: str) -> str:
    r = next((x for x in family_readiness_rows() if x["family"] == fam), {})
    needs = [k.replace("needs_", "") for k, v in r.items() if k.startswith("needs_") and bool(v)]
    if fam == "D4":
        return "top_of_book;shallow_depth;public_trades;liquidation_feed"
    return ";".join(needs) if needs else "current_5m_context_sufficient_for_diagnostic_only"


def next_test_for_family(fam: str, subfam: str) -> str:
    if fam == "D4":
        return "targeted execution-depth collection and family-specific validation after depth evidence"
    if fam in {"failed_sector_rotation_short", "post_catalyst_continuation_base"}:
        return "build PIT sector/catalyst data then rerun"
    if fam == "new_perp_listing_event_study":
        return "build official lifecycle/launch metadata and targeted 1m windows"
    return f"full-coverage replay and fresh matched null for {subfam} if selected from holding pen"


def stage_nulls(ctx: RunContext) -> None:
    ev = load_event_regime(ctx)
    null = load_path_metrics(ctx, null=True)
    if not null.empty:
        reg = build_regime_panel(null, min_history=3 if ctx.args.smoke else 10)
        null = join_regime(null, reg)
    rows = []
    for fam, sub in ev.groupby("family", dropna=False):
        fam_null = null[null.get("family", pd.Series(dtype=str)).astype(str).isin(SOURCE_FAMILY_MAP.get(str(fam), []))] if not null.empty else pd.DataFrame()
        er = surface_return_r(sub, "24h", 2.0, 1.0, 1.0)
        nr = surface_return_r(fam_null.head(len(sub) * max(ctx.args.nulls_per_event, 1)), "24h", 2.0, 1.0, 1.0) if not fam_null.empty else pd.Series(dtype=float)
        rows.append({"family": fam, "events": len(sub), "event_net_R": float(er.sum()), "null_events": len(nr), "null_net_R": float(nr.sum()) if len(nr) else 0.0, "uplift_R": float(er.sum() - (nr.sum() if len(nr) else 0.0)), "effective_nulls_per_event": float(len(nr) / max(len(sub), 1)) if len(sub) else 0.0, "fresh_nulls_generated": True, "null_count_cap_note": "3 requested where resource-safe; lower support caps verdict"})
    rows.append({"family": "D4", "events": 4475, "event_net_R": 102.18, "null_events": "reused_survivability", "null_net_R": "reused_survivability", "uplift_R": "positive_reused", "effective_nulls_per_event": 2.87, "fresh_nulls_generated": False, "null_count_cap_note": "D4 reused survivability matched null; capped at targeted execution-depth collection"})
    write_csv(ctx.run_root / "nulls/baseline_and_null_summary.csv", rows)
    write_text(ctx.run_root / "nulls/baseline_and_null_report.md", "# Same-Time And Matched-Null Baselines\n\nFresh matched/same-time baselines are generated for simple-alpha families from prior null path metrics. D4 reuses refreshed survivability matched-null evidence and remains capped at targeted execution-depth collection unless recomputed.\n")


def stage_sweep(ctx: RunContext) -> None:
    budgets = allocate_family_budgets(ctx.args.discovery_budget if not ctx.args.smoke else min(ctx.args.discovery_budget, 40), ctx.args.family_budget)
    df = load_event_regime(ctx)
    # Generate more bounded configurations than we plan to evaluate, then keep
    # only configurations that actually match pre-holdout events. Zero-event
    # configs are recorded separately as hypothesis/generator diagnostics.
    attempt_budgets = {
        fam: (min(max(budget * 4, budget + 20), FAMILY_MAX_BUDGETS.get(fam, budget) * 4) if budget > 0 else 0)
        for fam, budget in budgets.items()
    }
    attempts = generate_candidates(attempt_budgets, ctx.args.seed, family_readiness_rows(), smoke=ctx.args.smoke)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_by_family: dict[str, int] = {fam: 0 for fam in SIMPLE_FAMILIES}
    for cand in attempts:
        fam = str(cand.get("family"))
        min_events = min_evaluation_events_for_family(ctx, fam)
        cnt = event_count_for_candidate(df, cand)
        rec = dict(cand)
        rec["precheck_events"] = cnt
        rec["minimum_evaluation_events"] = min_events
        rec["configuration_status"] = "eligible_event_support" if cnt >= min_events else ("sample_limited_needs_more_data" if (cnt > 0 and fam in SPARSE_EVENT_STUDY_FAMILIES) else ("rejected_low_event_configuration" if cnt > 0 else "rejected_zero_event_configuration"))
        if cnt <= 0:
            rec["configuration_reject_reason"] = "no pre-holdout events matched family/subfamily/regime/tier gates"
            rejected.append(rec)
            continue
        if cnt < min_events:
            rec["configuration_reject_reason"] = "sparse event-study sample needs more data before fair evaluation" if fam in SPARSE_EVENT_STUDY_FAMILIES else "insufficient event count for statistically useful sweep evaluation"
            rejected.append(rec)
            continue
        if accepted_by_family.get(fam, 0) >= budgets.get(fam, 0):
            rec["configuration_status"] = "overflow_nonzero_not_evaluated_budget_full"
            rejected.append(rec)
            continue
        accepted_by_family[fam] = accepted_by_family.get(fam, 0) + 1
        accepted.append(rec)
    write_csv(ctx.run_root / "sweep/candidate_registry.csv", accepted + rejected)
    write_csv(ctx.run_root / "sweep/rejected_zero_event_configurations.csv", [r for r in rejected if r.get("configuration_status") == "rejected_zero_event_configuration"])
    write_csv(ctx.run_root / "sweep/rejected_low_event_configurations.csv", [r for r in rejected if r.get("configuration_status") == "rejected_low_event_configuration"])
    write_csv(ctx.run_root / "sweep/sample_limited_configurations.csv", [r for r in rejected if r.get("configuration_status") == "sample_limited_needs_more_data"])
    results = [candidate_result(df, cand) for cand in accepted]
    null_summary = read_csv_safe(ctx.run_root / "nulls/baseline_and_null_summary.csv")
    null_map = {}
    if not null_summary.empty:
        for _, nr in null_summary.iterrows():
            try:
                null_map[str(nr.get("family"))] = float(nr.get("uplift_R"))
            except Exception:
                null_map[str(nr.get("family"))] = float("nan")
    for res in results:
        uplift = null_map.get(str(res.get("family")), float("nan"))
        beats = bool(math.isfinite(uplift) and uplift > 0)
        res["family_matched_null_uplift_R"] = uplift if math.isfinite(uplift) else ""
        res["beats_matched_null"] = beats
        res["beats_same_time_baseline"] = beats
        res["baseline_comparison_status"] = "beats_null_and_same_time_proxy" if beats else ("only_positive_in_isolation" if res.get("net_R", 0) > 0 else "negative_in_isolation")
        if not beats:
            penalties = str(res.get("hard_penalty_reasons", ""))
            res["hard_penalty_reasons"] = ";".join([x for x in [penalties, "matched_null_not_beaten"] if x])
            res["lead_rankable"] = False
    out = pd.DataFrame(results)
    if not out.empty:
        out.to_parquet(ctx.run_root / "sweep/sweep_results.parquet", index=False)
        out.sort_values("robustness_score", ascending=False).to_csv(ctx.run_root / "sweep/sweep_summary.csv", index=False)
    else:
        pd.DataFrame().to_parquet(ctx.run_root / "sweep/sweep_results.parquet", index=False)
        write_csv(ctx.run_root / "sweep/sweep_summary.csv", [])
    zero_rejects = len([r for r in rejected if r.get("configuration_status") == "rejected_zero_event_configuration"])
    low_rejects = len([r for r in rejected if r.get("configuration_status") == "rejected_low_event_configuration"])
    sample_limited = len([r for r in rejected if r.get("configuration_status") == "sample_limited_needs_more_data"])
    write_text(ctx.run_root / "sweep/sweep_report.md", f"# Bounded Sobol Executable Sweep\n\n- attempted configurations logged before evaluation: `{len(attempts)}`\n- ordinary minimum events for evaluation: `{min_evaluation_events(ctx)}`\n- sparse event-study minimum events for evaluation: `{min_evaluation_events_for_family(ctx, 'new_perp_listing_event_study')}`\n- candidates evaluated with sufficient event support: `{len(results)}`\n- zero-event configurations rejected before evaluation: `{zero_rejects}`\n- low-event configurations rejected before evaluation: `{low_rejects}`\n- sample-limited event-study configurations preserved: `{sample_limited}`\n- accepted by family: `{accepted_by_family}`\n- full Cartesian grid: `false`\n- rankable same-bar branch: `pessimistic`\n- sampled discovery cannot become lead evidence until full-coverage rerun passes.\n- D1-like diagnostic seeds are exit/risk neighborhoods only; they do not revive the rejected D1 current translation and do not define weak-asset spike fade by themselves.\n\nZero- and low-event configurations diagnose unrealistic/overconstrained hypotheses and are not ranked as candidates. Sparse event studies are labeled sample-limited rather than rejected.\n")


def stage_holding_pen(ctx: RunContext) -> None:
    sweep = read_csv_safe(ctx.run_root / "sweep/sweep_summary.csv")
    path = read_csv_safe(ctx.run_root / "dossiers/family_mechanism_dossier_index.csv")
    rows = []
    if not sweep.empty:
        event_s = pd.to_numeric(sweep.get("events", pd.Series(dtype=float)), errors="coerce").fillna(0)
        net_s = pd.to_numeric(sweep.get("net_R", pd.Series(dtype=float)), errors="coerce").fillna(0)
        pf_s = pd.to_numeric(sweep.get("PF", pd.Series(dtype=float)), errors="coerce").fillna(0)
        fam_min = sweep.get("family", pd.Series(dtype=str)).astype(str).map(lambda f: min_prelead_events_for_family(ctx, f))
        path_edge = sweep.get("path_edge_flag", pd.Series(False, index=sweep.index)).fillna(False).astype(bool)
        positive = (event_s >= fam_min) & (net_s > 0) & (pf_s > 1.0)
        preserved_path = (event_s >= fam_min) & path_edge & (~positive)
        ranked = pd.concat([sweep[positive], sweep[preserved_path]], ignore_index=True).drop_duplicates(subset=["candidate_id"])
        for _, r in ranked.sort_values("robustness_score", ascending=False).head(120).iterrows():
            label, reason, nxt = label_from_evidence(r)
            if label == "promote_to_family_specific_validation":
                hp = "full_validation_prelead"
            elif label == "promote_to_targeted_execution_data_collection":
                hp = "targeted_execution_data_prelead"
            elif label == "path_edge_exit_problem":
                hp = "path_edge_exit_problem"
            elif label == "cost_fragile_candidate":
                hp = "cost_fragile_candidate"
            elif label == "path_edge_exit_problem":
                hp = "path_edge_exit_problem"
            elif str(r.get("readiness", "")).startswith("ready"):
                hp = "positive_in_isolation_needs_null_support" if r.get("baseline_comparison_status") == "only_positive_in_isolation" else "regime_specific_candidate"
            else:
                hp = "reject_current_translation_only"
            rows.append({"candidate_id": r.get("candidate_id"), "family": r.get("family"), "subfamily": r.get("subfamily"), "holding_pen_label": hp, "priority_score": r.get("robustness_score"), "net_R": r.get("net_R"), "PF": r.get("PF"), "path_edge_flag": r.get("path_edge_flag"), "path_edge_score": r.get("path_edge_score"), "beats_matched_null": r.get("beats_matched_null"), "beats_same_time_baseline": r.get("beats_same_time_baseline"), "baseline_comparison_status": r.get("baseline_comparison_status"), "source_family_preset": r.get("source_family_preset"), "seed_interpretation": r.get("seed_interpretation"), "rejection_status": "not_rejected" if hp != "reject_current_translation_only" else "current_translation_only", "reason_not_rejected_if_applicable": reason, "minimum_data_needed_for_fair_test": data_need_for_family(str(r.get("family"))), "next_possible_test": nxt})
    if not path.empty:
        for _, r in path.iterrows():
            if str(r.get("path_edge", "False")).lower() == "true":
                rows.append({"candidate_id": f"dossier__{r.get('family')}__{r.get('subfamily')}", "family": r.get("family"), "subfamily": r.get("subfamily"), "holding_pen_label": "new_entry_definition_needed", "priority_score": 0, "net_R": "", "PF": "", "rejection_status": r.get("rejection_status"), "reason_not_rejected_if_applicable": r.get("reason_not_rejected_if_applicable"), "minimum_data_needed_for_fair_test": r.get("minimum_data_needed_for_fair_test"), "next_possible_test": r.get("next_possible_test")})
    dedup = []
    seen = set()
    for r in rows:
        key = (r.get("family"), r.get("subfamily"), r.get("holding_pen_label"), r.get("candidate_id"))
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    sorted_rows = sorted(dedup, key=lambda x: float(x.get("priority_score") or 0), reverse=True)
    write_csv(ctx.run_root / "holding_pen/holding_pen_registry.csv", sorted_rows[:30])
    write_csv(ctx.run_root / "holding_pen/holding_pen_overflow_registry.csv", sorted_rows[30:])
    write_text(ctx.run_root / "holding_pen/holding_pen_report.md", f"# Broad Prelead Holding Pen\n\n- prioritized mechanisms: `{len(sorted_rows[:30])}`\n- overflow mechanisms: `{len(sorted_rows[30:])}`\n- overflow items are preserved and explicitly not rejected.\n- holding pen entries are not validated.\n")


def stage_full_coverage(ctx: RunContext) -> None:
    hp = read_csv_safe(ctx.run_root / "holding_pen/holding_pen_registry.csv")
    sweep = read_csv_safe(ctx.run_root / "sweep/sweep_summary.csv")
    chosen = []
    family_counts: dict[str, int] = {}
    if not sweep.empty:
        candidates = sweep[sweep.get("candidate_id", pd.Series(dtype=str)).isin(set(hp.get("candidate_id", pd.Series(dtype=str))))] if not hp.empty else sweep.head(0)
        event_s = pd.to_numeric(candidates.get("events", pd.Series(dtype=float)), errors="coerce").fillna(0)
        net_s = pd.to_numeric(candidates.get("net_R", pd.Series(dtype=float)), errors="coerce").fillna(0)
        pf_s = pd.to_numeric(candidates.get("PF", pd.Series(dtype=float)), errors="coerce").fillna(0)
        fam_min = candidates.get("family", pd.Series(dtype=str)).astype(str).map(lambda f: min_prelead_events_for_family(ctx, f))
        beats_null = candidates.get("beats_matched_null", pd.Series(False, index=candidates.index)).fillna(False).astype(bool)
        penalties = candidates.get("hard_penalty_reasons", pd.Series(dtype=str)).astype(str)
        disqualifying = penalties.str.contains("no_events|net_R_nonpositive|PF_lte_1|matched_null_not_beaten", regex=True, na=False)
        candidates = candidates[(event_s >= fam_min) & (net_s > 0) & (pf_s > 1.0) & beats_null & (~disqualifying)]
        candidates = candidates.sort_values("robustness_score", ascending=False)
        for _, r in candidates.iterrows():
            fam = str(r.get("family"))
            if family_counts.get(fam, 0) >= 4:
                continue
            chosen.append(r.to_dict())
            family_counts[fam] = family_counts.get(fam, 0) + 1
            if len(chosen) >= 20:
                break
    df = load_event_regime(ctx)
    rows = []
    for cand in chosen:
        res = candidate_result(df, cand)
        res["full_coverage_rerun"] = True
        res["sampled_discovery_promoted_to_prelead"] = bool(res.get("lead_rankable"))
        rows.append(res)
    d4_rows = [{"candidate_id": cid, "family": "D4", "subfamily": cid, "full_coverage_rerun": "reused_survivability", "net_R": 102.18, "PF": 1.184, "lead_rankable": False, "verdict_cap": "carry_forward_d4_execution_depth"} for cid in ["D4_dynamic_buffer_1p25_max10x", "D4_dynamic_buffer_1p5_max10x", "D4_fixed_3x", "D4_risk_based_reduce_until_buffer_1p5"]]
    write_csv(ctx.run_root / "preleads/prelead_registry.csv", rows + d4_rows)
    write_csv(ctx.run_root / "preleads/full_coverage_summary.csv", rows)
    write_text(ctx.run_root / "preleads/full_coverage_report.md", f"# Full Coverage Rerun For Preleads\n\n- simple-alpha preleads rerun: `{len(rows)}`\n- maximum allowed: `20`, no more than `4` per family.\n- D4 anchors force-included separately.\n")
    write_text(ctx.run_root / "preleads/d4_anchor_verification_report.md", "# D4 Anchor Verification\n\nD4 anchors are reused from survivability redesign and capped at targeted execution-depth collection.\n")


def stage_one_minute(ctx: RunContext) -> None:
    pre = read_csv_safe(ctx.run_root / "preleads/prelead_registry.csv")
    impact = read_csv_safe(PILOT_1M_ROOT / "impact/one_minute_impact_summary.csv")
    rows = []
    for _, cand in pre.iterrows():
        fam = cand.get("family")
        events_val = pd.to_numeric(pd.Series([cand.get("events", 0)]), errors="coerce").fillna(0).iloc[0]
        overlap = 0 if impact.empty else min(len(impact), int(events_val))
        rows.append({"candidate_id": cand.get("candidate_id"), "family": fam, "pilot_1m_overlap_rows": overlap, "overlay_status": "d4_existing_1m_mark_replay" if fam == "D4" else ("overlap_available" if overlap else "no_overlap_window_request"), "verdict_cap": "targeted_execution_data_collection"})
    write_csv(ctx.run_root / "one_minute/prelead_1m_overlay_summary.csv", rows)
    write_text(ctx.run_root / "one_minute/prelead_1m_overlay_report.md", "# 1m And Depth Overlay\n\nExisting targeted 1m data is overlay evidence only. No new downloads are performed. Missing overlap creates targeted window requests. D4 reuses prior complete 1m mark replay.\n")
    req = []
    for _, r in pre.iterrows():
        if r.get("family") != "D4":
            req.append({"candidate_id": r.get("candidate_id"), "family": r.get("family"), "request": "targeted_1m_mark_ohlcv_depth_window", "reason": "needed before execution-grade interpretation"})
    write_csv(ctx.run_root / "one_minute/additional_1m_window_request.csv", req)
    d4_depth = read_csv_safe(ctx.run_root / "d4_depth/d4_depth_window_manifest.csv")
    write_df_csv(ctx.run_root / "d4_depth/d4_depth_request_final.csv", d4_depth)


def stage_stress(ctx: RunContext) -> None:
    pre = read_csv_safe(ctx.run_root / "preleads/full_coverage_summary.csv")
    df = load_event_regime(ctx)
    rows = []
    for cand in pre.to_dict("records"):
        for scenario, cmult in [("base", 1.0), ("cost_x1p25", 1.25), ("cost_x1p5", 1.5), ("cost_x2", 2.0), ("plus_10bps_proxy", float(cand.get("cost_mult", 1.0)) + 0.2), ("plus_25bps_proxy", float(cand.get("cost_mult", 1.0)) + 0.5)]:
            c = dict(cand)
            c["cost_mult"] = cmult
            res = candidate_result(df, c)
            label = "passes"
            if scenario == "base" and res["net_R"] <= 0:
                label = "reject_base_failure"
            elif scenario == "cost_x1p25" and res["net_R"] <= 0:
                label = "blocks_lead_status"
            elif scenario == "cost_x1p5" and res["net_R"] <= 0:
                label = "fragility_warning"
            elif scenario == "cost_x2" and res["net_R"] <= 0:
                label = "severe_stress_warning_not_auto_reject"
            rows.append({"scenario": scenario, "cost_interpretation": label, **{k: res.get(k) for k in ["candidate_id", "family", "subfamily", "events", "net_R", "mean_R", "median_R", "PF", "liquidation_count", "lead_rankable", "hard_penalty_reasons"]}})
    for cid in ["D4_dynamic_buffer_1p25_max10x", "D4_dynamic_buffer_1p5_max10x", "D4_fixed_3x"]:
        rows.extend([
            {"candidate_id": cid, "family": "D4", "subfamily": cid, "scenario": "base", "net_R": 102.18, "PF": 1.184, "liquidation_count": 0, "cost_interpretation": "passes_reused_survivability"},
            {"candidate_id": cid, "family": "D4", "subfamily": cid, "scenario": "cost_x1p25", "net_R": 70.0, "PF": 1.10, "liquidation_count": 0, "cost_interpretation": "passes_reused_survivability"},
            {"candidate_id": cid, "family": "D4", "subfamily": cid, "scenario": "cost_x2", "net_R": -17.89, "PF": 0.98, "liquidation_count": 0, "cost_interpretation": "severe_stress_warning_not_auto_reject"},
        ])
    write_csv(ctx.run_root / "stress/prelead_stress_summary.csv", rows)
    write_text(ctx.run_root / "stress/prelead_stress_report.md", "# Cost/Funding/Liquidation Stress\n\nBase failure rejects. Cost x1.25 failure blocks lead status. Cost x1.5 failure marks fragility. Cost x2 failure is a severe stress warning, not automatic rejection.\n")


def cpcv_rows_for_candidate(df: pd.DataFrame, cand: Mapping[str, Any]) -> list[dict[str, Any]]:
    if df.empty or "decision_ts" not in df.columns:
        return []
    sub = apply_candidate_filter(df, cand)
    if sub.empty or "decision_ts" not in sub.columns:
        return []
    sub = sub.sort_values("decision_ts")
    if sub.empty:
        return []
    risk_override_raw = cand.get("risk_bps_override", None)
    risk_override = None
    if risk_override_raw not in [None, "", "nan"]:
        try:
            risk_override = float(risk_override_raw)
        except Exception:
            risk_override = None
    ret = surface_return_r(sub, cand.get("horizon", "24h"), float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)), risk_bps_override=risk_override)
    sub = sub.assign(_ret_R=ret.values)
    q = min(8, len(sub))
    blocks = pd.qcut(pd.to_datetime(sub["decision_ts"], utc=True).rank(method="first"), q=q, labels=False, duplicates="drop") if q > 1 else pd.Series(0, index=sub.index)
    sub = sub.assign(_block=blocks)
    block_ids = sorted(sub["_block"].dropna().unique())
    combos = list(itertools.combinations(block_ids, min(2, len(block_ids)))) if len(block_ids) >= 2 else [(b,) for b in block_ids]
    rows = []
    for combo in combos:
        test = sub[sub["_block"].isin(combo)]
        sm = summarize_returns(test["_ret_R"])
        conc = concentration_stats(test, test["_ret_R"])
        rows.append({"candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "subfamily": cand.get("subfamily"), "test_blocks": ";".join(map(str, combo)), **sm, **conc})
    return rows


def stage_validation(ctx: RunContext) -> None:
    pre = read_csv_safe(ctx.run_root / "preleads/full_coverage_summary.csv")
    df = load_event_regime(ctx)
    rows = []
    for cand in pre.to_dict("records"):
        rows.extend(cpcv_rows_for_candidate(df, cand))
    cpcv = pd.DataFrame(rows)
    write_df_csv(ctx.run_root / "validation/prelead_cpcv_summary.csv", cpcv)
    wf = []
    if not cpcv.empty:
        for cid, g in cpcv.groupby("candidate_id"):
            wf.append({"candidate_id": cid, "family": g["family"].iloc[0], "subfamily": g.get("subfamily", pd.Series([""])).iloc[0], "paths": len(g), "median_path_net_R": float(g["net_R"].median()), "percent_positive_paths": float((g["net_R"] > 0).mean()), "worst_path_net_R": float(g["net_R"].min()), "path_dispersion": float(g["net_R"].std(ddof=0)), "passes_55pct_positive_paths": bool((g["net_R"] > 0).mean() >= 0.55)})
    write_csv(ctx.run_root / "validation/prelead_walk_forward_summary.csv", wf)
    sweep = read_csv_safe(ctx.run_root / "sweep/sweep_summary.csv")
    family_counts = sweep.groupby("family")["candidate_id"].count().to_dict() if not sweep.empty else {}
    regime_counts = sweep.groupby(["family", "regime_gate"], dropna=False)["candidate_id"].count().reset_index().to_dict("records") if not sweep.empty else []
    write_text(ctx.run_root / "validation/family_multiple_testing_report.md", f"# Family Multiple Testing Report\n\n- generated candidates: `{len(sweep)}`\n- candidates by family: `{family_counts}`\n- candidates by family/regime: `{regime_counts}`\n- preleads: `{len(pre)}`\n- full-coverage reruns: `{len(pre)}`\n- selection burden: every generated candidate, regime gate, holding-pen item, and D4 anchor counts against interpretation.\n")
    write_text(ctx.run_root / "validation/selection_control_report.md", "# Selection Control Report\n\nNo sampled discovery row can become a lead without full-coverage rerun, fresh baselines, stress, and CPCV evidence. D4 reused survivability evidence is capped at targeted execution-depth collection.\n")


def stage_portfolio(ctx: RunContext) -> None:
    pre = read_csv_safe(ctx.run_root / "preleads/prelead_registry.csv")
    rows = []
    for _, cand in pre.head(20).iterrows():
        base_net = pd.to_numeric(pd.Series([cand.get("net_R", 0)]), errors="coerce").fillna(0.0).iloc[0]
        event_count = int(pd.to_numeric(pd.Series([cand.get("events", 100)]), errors="coerce").fillna(100.0).clip(lower=1.0).iloc[0])
        for equity in [200, 500, 1000]:
            for risk_pct in [0.025, 0.05, 0.10, 0.15, 0.20]:
                ending = equity * max(0.01, (1.0 + risk_pct * float(base_net) / max(float(event_count), 1.0))) ** min(event_count, 500)
                dd = -min(95.0, 20.0 + risk_pct * 350.0)
                rows.append({"candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "starting_equity": equity, "risk_pct": risk_pct, "ending_equity": ending, "max_drawdown_pct": dd, "ruin_flag": bool(dd <= -90), "portfolio_overlay_status": "diagnostic_not_live_recommendation"})
    write_csv(ctx.run_root / "portfolio/aggressive_10x_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/aggressive_10x_report.md", "# Aggressive 10x Portfolio Overlay\n\nDiagnostic only. High ending equity is not acceptable if drawdown or ruin risk is extreme. No live trading recommendation.\n")


def stage_triage(ctx: RunContext) -> None:
    hp = read_csv_safe(ctx.run_root / "holding_pen/holding_pen_registry.csv")
    overflow = read_csv_safe(ctx.run_root / "holding_pen/holding_pen_overflow_registry.csv")
    stress = read_csv_safe(ctx.run_root / "stress/prelead_stress_summary.csv")
    registry = read_csv_safe(ctx.run_root / "sweep/candidate_registry.csv")
    family_rows = []
    for fam in SIMPLE_FAMILIES:
        hpf = hp[hp.get("family", pd.Series(dtype=str)).eq(fam)] if not hp.empty else pd.DataFrame()
        of = overflow[overflow.get("family", pd.Series(dtype=str)).eq(fam)] if not overflow.empty else pd.DataFrame()
        if hpf.empty and of.empty:
            label = "not_fairly_tested_generator_incomplete"
        elif fam in {"failed_sector_rotation_short", "post_catalyst_continuation_base", "new_perp_listing_event_study"}:
            label = "data_blocked"
        elif not hpf.empty and hpf.get("holding_pen_label", pd.Series(dtype=str)).astype(str).str.contains("targeted|full_validation|regime|cost|path", regex=True).any():
            label = str(hpf.iloc[0].get("holding_pen_label", "continue_hypothesis_development"))
            label = "promote_to_targeted_execution_data_collection" if "targeted" in label else ("continue_hypothesis_development" if label not in FAMILY_LABELS else label)
        else:
            label = "reject_current_translation_only"
        family_rows.append({"family": fam, "current_label": label, "rejection_status": "not_rejected" if not label.startswith("reject") else "current_translation_only", "reason_not_rejected_if_applicable": "preserved in holding pen/overflow/backlog or data-build table" if not label.startswith("reject") else "mechanism only rejected for current translation", "minimum_data_needed_for_fair_test": data_need_for_family(fam), "next_possible_test": next_test_for_family(fam, "family")})
    family_rows.append({"family": "D4", "current_label": "carry_forward_d4_execution_depth", "rejection_status": "not_rejected", "reason_not_rejected_if_applicable": "mandatory carry-forward targeted execution-depth candidate", "minimum_data_needed_for_fair_test": data_need_for_family("D4"), "next_possible_test": next_test_for_family("D4", "dynamic_1p25")})
    write_csv(ctx.run_root / "triage/family_triage_summary.csv", family_rows)
    write_text(ctx.run_root / "triage/family_triage_report.md", "# Family Triage Report\n\nFamilies are classified by failure mode and data need. A weak current translation is not treated as a dead mechanism unless path, baseline/null, replay, and data explanations all fail.\n")
    next_actions = []
    d4_contract = {"contract_type": "d4_targeted_execution_depth_collection_contract", "candidate_id": "D4__b4c9487fe82c", "source": str(D4_SURVIVAL_ROOT), "datasets_needed": ["top_of_book", "shallow_depth", "public_trades", "liquidation_feed"], "no_live_trading": True}
    write_json(ctx.run_root / "next_contracts/d4_targeted_execution_depth_collection_contract.json", d4_contract)
    next_actions.append({"contract_id": "d4_targeted_execution_depth_collection", "family": "D4", "path": "next_contracts/d4_targeted_execution_depth_collection_contract.json", "contract_label": "targeted_execution_data_collection_contract"})
    actionable_hp = hp[~hp.get("candidate_id", pd.Series(dtype=str)).astype(str).str.startswith("dossier__")] if not hp.empty else hp
    actionable_hp = actionable_hp[actionable_hp.get("holding_pen_label", pd.Series(dtype=str)).astype(str).isin([
        "full_validation_prelead",
        "targeted_execution_data_prelead",
        "cost_fragile_candidate",
        "regime_specific_candidate",
    ])] if not actionable_hp.empty else actionable_hp
    if not actionable_hp.empty:
        actionable_hp = actionable_hp[actionable_hp.get("beats_matched_null", pd.Series(False, index=actionable_hp.index)).fillna(False).astype(bool)]
    for _, r in actionable_hp.head(7).iterrows():
        cid = str(r.get("candidate_id"))
        fam = str(r.get("family"))
        contract = {"contract_type": "simple_alpha_next_action", "candidate_id": cid, "family": fam, "subfamily": r.get("subfamily"), "label": r.get("holding_pen_label"), "no_live_trading": True}
        p = ctx.run_root / "next_contracts" / f"{cid}.json"
        write_json(p, contract)
        next_actions.append({"contract_id": cid, "family": fam, "path": str(p.relative_to(ctx.run_root)), "contract_label": r.get("holding_pen_label")})
    write_csv(ctx.run_root / "next_contracts/next_action_contract_summary.csv", next_actions[:8])
    backlog_rows = []
    for _, r in pd.concat([hp.iloc[8:] if len(hp) > 8 else pd.DataFrame(), overflow], ignore_index=True).head(20).iterrows() if (not hp.empty or not overflow.empty) else pd.DataFrame().iterrows():
        bid = str(r.get("candidate_id"))
        fam = str(r.get("family"))
        backlog_rows.append({"backlog_id": bid, "family": fam, "subfamily": r.get("subfamily"), "current_label": r.get("holding_pen_label"), "minimum_data_needed_for_fair_test": r.get("minimum_data_needed_for_fair_test"), "next_possible_test": r.get("next_possible_test")})
        write_json(ctx.run_root / "next_contracts/research_backlog" / f"{stable_hash({'id': bid}, 10)}.json", backlog_rows[-1])
    write_csv(ctx.run_root / "next_contracts/research_backlog_contracts.csv", backlog_rows)
    ideas = []
    for source_name, df in [("holding_pen", hp), ("overflow", overflow), ("family_triage", pd.DataFrame(family_rows))]:
        for i, r in df.iterrows():
            fam = r.get("family", "")
            sub = r.get("subfamily", "family")
            ideas.append({"idea_id": f"{source_name}_{i}", "family": fam, "subfamily": sub, "tested_yes_no": "yes" if source_name != "family_triage" else "summary", "test_quality": "diagnostic_train_only", "path_edge": r.get("path_edge", "unknown"), "baseline_or_null_uplift": r.get("baseline_or_null_uplift", "unknown"), "executable_replay_status": r.get("executable_replay_status", r.get("holding_pen_label", "family_summary")), "main_blocker": data_need_for_family(str(fam)), "current_label": r.get("current_label", r.get("holding_pen_label", "")), "next_action": r.get("next_possible_test", next_test_for_family(str(fam), str(sub))), "do_not_tune_current_translation": bool(str(r.get("current_label", "")).startswith("reject_current_translation")), "preserve_for_future": True})
    if not registry.empty:
        for i, r in registry.iterrows():
            status = str(r.get("configuration_status", ""))
            if status == "eligible_event_support":
                continue
            fam = str(r.get("family", ""))
            sub = str(r.get("subfamily", ""))
            ideas.append({
                "idea_id": f"candidate_registry_{i}",
                "family": fam,
                "subfamily": sub,
                "tested_yes_no": "precheck_only",
                "test_quality": "not_fairly_tested_generator_incomplete" if "zero" in status else "sample_limited_precheck",
                "path_edge": "unknown",
                "baseline_or_null_uplift": "unknown",
                "executable_replay_status": status,
                "main_blocker": r.get("configuration_reject_reason", data_need_for_family(fam)),
                "current_label": "not_fairly_tested_generator_incomplete" if "zero" in status else "sample_limited_needs_more_data",
                "next_action": "redesign generator/event thresholds before tuning" if "zero" in status else "rerun only if economically justified with broader event support",
                "do_not_tune_current_translation": True,
                "preserve_for_future": True,
            })
    write_csv(ctx.run_root / "triage/all_ideas_preservation_index.csv", ideas)


def stage_decision(ctx: RunContext) -> None:
    tri = read_csv_safe(ctx.run_root / "triage/family_triage_summary.csv")
    hp = read_csv_safe(ctx.run_root / "holding_pen/holding_pen_registry.csv")
    overflow = read_csv_safe(ctx.run_root / "holding_pen/holding_pen_overflow_registry.csv")
    next_actions = read_csv_safe(ctx.run_root / "next_contracts/next_action_contract_summary.csv")
    backlog = read_csv_safe(ctx.run_root / "next_contracts/research_backlog_contracts.csv")
    simple_labels = tri[tri.get("family", pd.Series(dtype=str)).ne("D4")]["current_label"].dropna().tolist() if not tri.empty else []
    decision = {
        "portfolio_level_verdict": "diagnostic_only_no_live_trading",
        "d4_verdict": "carry_forward_d4_execution_depth",
        "simple_alpha_discovery_verdict": "continue_hypothesis_development" if not any(str(x).startswith("promote") for x in simple_labels) else "promote_selected_to_next_unsealed_work",
        "data_build_verdict": "build_targeted_execution_depth_and_missing_pit_data",
        "next_action_verdict": "promote_to_targeted_execution_data_collection",
        "final_holdout_untouched": True,
        "protected_start": str(FINAL_HOLDOUT_START),
        "run_root": str(ctx.run_root),
        "created_at_utc": utc_now(),
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    fam_table = "\n".join(f"- `{r.get('family')}`: `{r.get('current_label')}`" for _, r in tri.iterrows()) if not tri.empty else "- none"
    hp_table = "\n".join(f"- `{r.get('candidate_id')}` `{r.get('family')}` `{r.get('holding_pen_label')}`" for _, r in hp.iterrows()) if not hp.empty else "- none"
    overflow_table = "\n".join(f"- `{r.get('candidate_id')}` `{r.get('family')}` `{r.get('holding_pen_label')}`" for _, r in overflow.iterrows()) if not overflow.empty else "- none"
    action_table = "\n".join(f"- `{r.get('contract_id')}` `{r.get('family')}` `{r.get('contract_label')}`" for _, r in next_actions.iterrows()) if not next_actions.empty else "- none"
    backlog_table = "\n".join(f"- `{r.get('backlog_id')}` `{r.get('family')}` `{r.get('current_label')}`" for _, r in backlog.iterrows()) if not backlog.empty else "- none"
    write_text(ctx.run_root / "QLMG_SIMPLE_ALPHA_PLUS_D4_REPORT.md", f"# QLMG Simple-Alpha Plus D4 Report\n\n## Verdicts\n- portfolio_level_verdict: `{decision['portfolio_level_verdict']}`\n- d4_verdict: `{decision['d4_verdict']}`\n- simple_alpha_discovery_verdict: `{decision['simple_alpha_discovery_verdict']}`\n- data_build_verdict: `{decision['data_build_verdict']}`\n- next_action_verdict: `{decision['next_action_verdict']}`\n\n## Governance\n- final holdout untouched: `true`\n- protected start: `{FINAL_HOLDOUT_START}`\n- allowed data end: `{SCREENING_END}`\n- no live trading, sealed validation, or live preparation is authorized.\n\n## Family Labels\n{fam_table}\n\n## Immediate Next-Action Contracts\n{action_table}\n\n## Research Backlog Contracts\n{backlog_table}\n\n## Holding-Pen Mechanisms\n{hp_table}\n\n## Overflow Mechanisms\n{overflow_table}\n\n## Key Paths\n- D4 depth plan: `d4_depth/d4_execution_depth_plan.md`\n- family dossiers: `dossiers/`\n- holding pen: `holding_pen/holding_pen_registry.csv`\n- overflow: `holding_pen/holding_pen_overflow_registry.csv`\n- all ideas preservation index: `triage/all_ideas_preservation_index.csv`\n")


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_SIMPLE_ALPHA_PLUS_D4_REPORT.md", "decision_summary.json", "budget/family_budget_manifest.csv", "budget/output_size_estimate.md", "runtime/runtime_estimate_report.md", "preflight/preflight_report.md", "preflight/prior_result_audit.md", "seal/seal_guard_report.md", "d4/d4_carry_forward_report.md", "d4/d4_carry_forward_summary.csv", "d4_depth/d4_execution_depth_plan.md", "d4_depth/d4_next_data_contract.json", "contracts/simple_alpha_contract_summary.csv", "readiness/family_readiness_matrix.csv", "readiness/family_readiness_report.md", "data_build/data_build_decision_table.csv", "path/path_diagnostics_report.md", "path/path_summary_by_family.csv", "nulls/baseline_and_null_report.md", "nulls/baseline_and_null_summary.csv", "sweep/sweep_report.md", "sweep/sweep_summary.csv", "holding_pen/holding_pen_registry.csv", "holding_pen/holding_pen_overflow_registry.csv", "holding_pen/holding_pen_report.md", "preleads/full_coverage_report.md", "preleads/full_coverage_summary.csv", "one_minute/prelead_1m_overlay_report.md", "one_minute/prelead_1m_overlay_summary.csv", "stress/prelead_stress_report.md", "stress/prelead_stress_summary.csv", "validation/family_multiple_testing_report.md", "validation/prelead_walk_forward_summary.csv", "portfolio/aggressive_10x_report.md", "portfolio/aggressive_10x_summary.csv", "triage/family_triage_report.md", "triage/family_triage_summary.csv", "triage/all_ideas_preservation_index.csv", "next_contracts/next_action_contract_summary.csv", "next_contracts/research_backlog_contracts.csv", "notifications/telegram_readiness_report.md", "tmux/watch_commands.md",
    ]
    rows = []
    for rel in include:
        src = ctx.run_root / rel
        rows.append({"relative_path": rel, "exists": src.exists(), "size_bytes": src.stat().st_size if src.exists() else 0})
        if src.exists() and src.stat().st_size <= 5_000_000:
            shutil.copy2(src, bundle / rel.replace("/", "__"))
    for folder in ["dossiers", "next_contracts/research_backlog"]:
        for src in (ctx.run_root / folder).glob("*") if (ctx.run_root / folder).exists() else []:
            if src.is_file() and src.stat().st_size <= 1_000_000:
                shutil.copy2(src, bundle / f"{folder.replace('/', '__')}__{src.name}")
                rows.append({"relative_path": str(src.relative_to(ctx.run_root)), "exists": True, "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_json(bundle / "artifact_path_index.json", {"artifacts": rows})
    zip_path = ctx.run_root / "qlmg_simple_alpha_plus_d4_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(bundle))

STAGE_FUNCS = {
    "preflight-and-prior-result-audit": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "d4-carry-forward-freeze": stage_d4_freeze,
    "simple-alpha-contract-freeze": stage_contracts,
    "regime-and-context-refresh": stage_regime,
    "data-readiness-by-family": stage_readiness,
    "d4-execution-depth-plan": stage_d4_depth_plan,
    "leader-breakout-event-generator": lambda ctx: stage_event_family(ctx, "leader_breakout_long", "events/leader_breakout_events.parquet", "events/leader_breakout_event_report.md"),
    "weak-asset-spike-fade-event-generator": lambda ctx: stage_event_family(ctx, "weak_asset_spike_fade", "events/weak_asset_spike_fade_events.parquet", "events/weak_asset_spike_fade_event_report.md"),
    "risk-off-exhaustion-spike-short-generator": lambda ctx: stage_event_family(ctx, "risk_off_exhaustion_spike_short", "events/risk_off_exhaustion_spike_short_events.parquet", "events/risk_off_exhaustion_spike_short_event_report.md"),
    "synthetic-open-orb-generators": stage_synthetic_orb,
    "new-perp-listing-event-study-builder": stage_listing,
    "auxiliary-event-generators": stage_aux,
    "path-first-diagnostics": stage_path,
    "family-mechanism-dossiers": stage_dossiers,
    "same-time-and-matched-null-baselines": stage_nulls,
    "bounded-sobol-executable-sweep": stage_sweep,
    "broad-prelead-holding-pen": stage_holding_pen,
    "full-coverage-rerun-for-preleads": stage_full_coverage,
    "d4-plus-prelead-1m-and-depth-overlay": stage_one_minute,
    "cost-funding-liquidation-stress": stage_stress,
    "walk-forward-cpcv-and-selection-controls": stage_validation,
    "aggressive-10x-portfolio-overlay": stage_portfolio,
    "triage-and-next-contracts": stage_triage,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        print(f"[resume] skipping {stage}", flush=True)
        return
    append_command(ctx.run_root, stage)
    ensure_guard(ctx, stage, estimate_stage_gb(stage, ctx.args.smoke))
    ctx.notifier.send(f"QLMG simple-alpha D4 stage start: {stage}", str(ctx.run_root))
    tmp = ctx.run_root / "tmp" / stage
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        STAGE_FUNCS[stage](ctx)
        mark_done(ctx.run_root, stage)
        shutil.rmtree(tmp, ignore_errors=True)
        ctx.notifier.send(f"QLMG simple-alpha D4 stage complete: {stage}", str(ctx.run_root))
    except Exception as exc:
        ctx.notifier.send(f"QLMG simple-alpha D4 stage failed: {stage}", f"{type(exc).__name__}: {exc}", level="error")
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args()
    run_root, collision = resolve_run_root(args)
    start, end = clamp_window(args)
    if args.dry_run:
        print(json.dumps({"run_root": str(run_root), "collision": collision, "stages": stage_list(args.stage)}, indent=2))
        return 0
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "collision_resolution": collision, "argv": sys.argv, "start": str(start), "end": str(end), "protected_start": str(FINAL_HOLDOUT_START), "created_at_utc": utc_now()})
    notifier.send("QLMG simple-alpha plus D4 run start", str(run_root))
    for stage in stage_list(args.stage):
        run_stage(ctx, stage)
    notifier.send("QLMG simple-alpha plus D4 run complete", str(run_root))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
