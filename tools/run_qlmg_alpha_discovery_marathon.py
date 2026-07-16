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
    REGIME_LAYERS,
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
DEFAULT_RUN_ID = "phase_qlmg_alpha_discovery_marathon_20260625_v1"
FIRST_SCREEN_ROOT = RESULTS_ROOT / "phase_qlmg_engine_and_first_screen_20260624_v1_20260624_101747"
PATH_DIAG_ROOT = RESULTS_ROOT / "phase_qlmg_path_diagnostics_exit_surface_20260624_v1_20260624_121522"
D1_ROOT = RESULTS_ROOT / "phase_qlmg_d1_narrow_validation_20260624_v1"
PILOT_1M_ROOT = RESULTS_ROOT / "phase_qlmg_targeted_1m_data_pilot_20260624_v1"
F1G1_ROOT = RESULTS_ROOT / "phase_qlmg_f1_g1_short_unblock_20260625_v1_20260625_120455"

STAGES = (
    "preflight-and-prior-result-audit",
    "telegram-and-tmux-setup",
    "seal-guard",
    "scoring-and-promotion-rule-fix",
    "data-availability-and-family-readiness",
    "regime-stack-refresh",
    "family-generator-contracts",
    "discovery-event-generation",
    "path-first-opportunity-screen",
    "regime-conditional-cube",
    "bounded-sobol-entry-exit-search",
    "full-coverage-rerun-for-preleads",
    "matched-null-validation-for-preleads",
    "targeted-1m-overlay-for-preleads",
    "execution-cost-liquidation-stress",
    "walk-forward-cpcv-and-overfit-controls",
    "aggressive-10x-portfolio-overlay",
    "new-hypothesis-synthesis",
    "next-candidate-contracts",
    "decision-report",
    "compact-review-bundle",
    "all",
)

ALLOWED_VERDICTS = {
    "promote_one_or_more_to_family_specific_validation",
    "promote_to_targeted_execution_data_collection",
    "build_sector_catalyst_data_next",
    "continue_alpha_discovery_needed",
    "reject_current_translations_and_research_new_hypotheses",
    "blocked_by_data_or_execution",
    "blocked_by_protocol_issue",
}

FAMILY_MAX_BUDGETS = {
    "A1_A3_A4": 350,
    "A2": 150,
    "D3_D4_E1": 350,
    "H3": 200,
    "F1_G1": 250,
    "H1": 150,
    "B1_C2": 250,
    "adaptive_refinement": 150,
}

HORIZONS = ["30m", "1h", "2h", "4h", "6h", "12h", "24h", "48h", "72h"]
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
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-alpha-marathon")
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
    p = argparse.ArgumentParser(description="QLMG alpha discovery marathon")
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
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--discovery-budget", type=int, default=1800)
    p.add_argument("--family-budget", type=int, default=0, help="optional per-family cap override for smoke/debug")
    p.add_argument("--refine-budget", type=int, default=240)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--use-targeted-1m-if-overlap", action="store_true")
    p.add_argument("--launch-tmux", action="store_true", help="accepted for wrapper compatibility; runner itself does not spawn tmux")
    p.add_argument("--tmux-session-name", default="qlmg_alpha_marathon")
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
        "preflight-and-prior-result-audit": [run_root / "preflight/preflight_report.md", run_root / "budget/family_budget_manifest.csv", run_root / "budget/output_size_estimate.md"],
        "telegram-and-tmux-setup": [run_root / "notifications/telegram_readiness_report.md", run_root / "tmux/watch_commands.md"],
        "seal-guard": [run_root / "seal/seal_guard_report.md", run_root / "seal/protected_slice_check.json"],
        "scoring-and-promotion-rule-fix": [run_root / "scoring/scoring_contract.json", run_root / "scoring/scoring_rule_report.md"],
        "data-availability-and-family-readiness": [run_root / "readiness/family_readiness_matrix.csv", run_root / "readiness/family_readiness_report.md"],
        "regime-stack-refresh": [run_root / "regime/regime_feature_panel.parquet", run_root / "regime/regime_stack_qc_report.md"],
        "family-generator-contracts": [run_root / "contracts/family_contract_summary.csv"],
        "discovery-event-generation": [run_root / "events/discovery_event_ledger.parquet", run_root / "events/discovery_event_summary.csv"],
        "path-first-opportunity-screen": [run_root / "path_screen/path_opportunity_summary.csv", run_root / "path_screen/path_opportunity_report.md"],
        "regime-conditional-cube": [run_root / "cubes/regime_conditional_cube.parquet", run_root / "cubes/regime_conditional_cube_summary.csv"],
        "bounded-sobol-entry-exit-search": [run_root / "sweep/candidate_registry.csv", run_root / "sweep/sweep_results.parquet"],
        "full-coverage-rerun-for-preleads": [run_root / "preleads/full_coverage_prelead_summary.csv"],
        "matched-null-validation-for-preleads": [run_root / "matched_null/prelead_matched_null_summary.csv", run_root / "matched_null/fresh_matched_null_policy.json"],
        "targeted-1m-overlay-for-preleads": [run_root / "one_minute/prelead_1m_overlay_summary.csv"],
        "execution-cost-liquidation-stress": [run_root / "stress/execution_cost_liquidation_stress_summary.csv"],
        "walk-forward-cpcv-and-overfit-controls": [run_root / "validation/cpcv_summary.csv", run_root / "validation/family_multiple_testing_report.md"],
        "aggressive-10x-portfolio-overlay": [run_root / "portfolio/aggressive_10x_portfolio_summary.csv"],
        "new-hypothesis-synthesis": [run_root / "hypotheses/new_hypothesis_synthesis.md", run_root / "hypotheses/hypothesis_candidates.csv"],
        "next-candidate-contracts": [run_root / "next_contracts/next_contract_summary.csv"],
        "decision-report": [run_root / "QLMG_ALPHA_DISCOVERY_MARATHON_REPORT.md", run_root / "decision_summary.json"],
        "compact-review-bundle": [run_root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return m.get(stage, [])


def stage_complete(run_root: Path, stage: str) -> bool:
    return done_path(run_root, stage).exists() and all(p.exists() for p in required_outputs_for_stage(run_root, stage))


def estimate_stage_gb(stage: str, smoke: bool) -> float:
    base = {
        "regime-stack-refresh": 0.8,
        "discovery-event-generation": 1.2,
        "bounded-sobol-entry-exit-search": 0.8,
        "matched-null-validation-for-preleads": 1.0,
        "walk-forward-cpcv-and-overfit-controls": 0.5,
    }.get(stage, 0.15)
    return min(base, 0.25) if smoke else base


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
        ctx.notifier.send(f"QLMG alpha marathon resource warning: {stage}", json.dumps(status), level="warning")
    if status["status"] != "pass":
        raise RuntimeError(f"resource guard failed for {stage}: {status}")


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_text_safe(path: Path, limit: int = 6000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit] if path.exists() else ""
    except Exception:
        return ""


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
        df = df.sort_values(["family", "symbol", "decision_ts"]).groupby("family", group_keys=False).head(4000)
    validate_no_protected(df, ["decision_ts", "entry_ts"])
    return df.reset_index(drop=True)


def load_events(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "events/discovery_event_ledger.parquet"
    return pd.read_parquet(p) if p.exists() else load_path_metrics(ctx, null=False)


def load_regime(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "regime/regime_feature_panel.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_event_regime(ctx: RunContext, *, null: bool = False) -> pd.DataFrame:
    base = load_path_metrics(ctx, null=null) if null else load_events(ctx)
    reg = build_regime_panel(base, min_history=3 if ctx.args.smoke else 10) if null else load_regime(ctx)
    if reg.empty and not base.empty:
        reg = build_regime_panel(base, min_history=3 if ctx.args.smoke else 10)
    return join_regime(base, reg)


def allocate_family_budgets(total: int, family_budget: int = 0) -> dict[str, int]:
    total = max(int(total), 0)
    if family_budget > 0:
        return {k: min(int(family_budget), v) for k, v in FAMILY_MAX_BUDGETS.items()}
    max_total = sum(FAMILY_MAX_BUDGETS.values())
    if total >= max_total:
        return dict(FAMILY_MAX_BUDGETS)
    raw = {k: int(total * v / max_total) for k, v in FAMILY_MAX_BUDGETS.items()}
    remainder = total - sum(raw.values())
    for k in FAMILY_MAX_BUDGETS:
        if remainder <= 0:
            break
        raw[k] += 1
        remainder -= 1
    return raw


def one_minute_adequate() -> tuple[bool, int, float]:
    cov = read_csv_safe(PILOT_1M_ROOT / "qc/pilot_coverage_summary.csv")
    if cov.empty:
        return False, 0, 0.0
    ok = cov[cov.get("status", pd.Series(dtype=str)).astype(str).eq("ok")] if "status" in cov.columns else cov
    ratio = float(pd.to_numeric(ok.get("coverage_ratio", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not ok.empty else 0.0
    return len(ok) >= 100 and ratio >= 0.95, int(len(ok)), ratio


def family_readiness_rows() -> list[dict[str, Any]]:
    one_ok, one_rows, one_ratio = one_minute_adequate()
    f1g1_decision = read_text_safe(F1G1_ROOT / "decision_summary.json", 1000)
    sector_pit = False
    catalyst_pit = False
    rows = [
        {"family": "D3", "readiness": "ready_for_discovery", "reason": "prior path edge exists; 5m path metrics available", "max_verdict_cap": "promote_to_targeted_execution_data_collection", "proxy_sector_only": False},
        {"family": "D4", "readiness": "ready_for_discovery", "reason": "inferred from D3/E1 deleveraging/reset proxies", "max_verdict_cap": "promote_to_targeted_execution_data_collection", "proxy_sector_only": False},
        {"family": "E1", "readiness": "ready_for_discovery", "reason": "prior path edge exists; mark/liquidation remains proxy-grade", "max_verdict_cap": "promote_to_targeted_execution_data_collection", "proxy_sector_only": False},
        {"family": "A1", "readiness": "path_only", "reason": "no true A1 event ledger in path diagnostics; proxy from A2-like continuation events", "max_verdict_cap": "continue_alpha_discovery_needed", "proxy_sector_only": False},
        {"family": "A3", "readiness": "path_only", "reason": "post-breakout pullback path ledger not separately available", "max_verdict_cap": "continue_alpha_discovery_needed", "proxy_sector_only": False},
        {"family": "A4", "readiness": "path_only", "reason": "liquid continuation variant uses A2 proxy evidence only", "max_verdict_cap": "continue_alpha_discovery_needed", "proxy_sector_only": False},
        {"family": "A2", "readiness": "ready_for_discovery", "reason": "prior A2 path metrics available but matched-null was weak", "max_verdict_cap": "continue_alpha_discovery_needed", "proxy_sector_only": False},
        {"family": "H3", "readiness": "path_only", "reason": "volatility interruption/reversal proxy can use D3/E1 path metrics; no dedicated H3 generator", "max_verdict_cap": "continue_alpha_discovery_needed", "proxy_sector_only": False},
        {"family": "F1", "readiness": "ready_for_discovery" if F1G1_ROOT.exists() else "blocked_do_not_run", "reason": "short-unblock artifacts exist; current translation rejected" if F1G1_ROOT.exists() else "missing short-unblock artifacts", "max_verdict_cap": "continue_alpha_discovery_needed", "proxy_sector_only": False, "prior_decision": f1g1_decision[:200]},
        {"family": "G1", "readiness": "ready_for_discovery" if F1G1_ROOT.exists() else "blocked_do_not_run", "reason": "short-unblock artifacts exist; current translation rejected" if F1G1_ROOT.exists() else "missing short-unblock artifacts", "max_verdict_cap": "continue_alpha_discovery_needed", "proxy_sector_only": False, "prior_decision": f1g1_decision[:200]},
        {"family": "H1", "readiness": "ready_for_discovery" if one_ok else "needs_targeted_1m", "reason": f"1m coverage rows={one_rows}, avg coverage={one_ratio:.3f}; H1 cannot be tested as 5m lead/lag", "max_verdict_cap": "targeted_execution_data_collection_contract", "proxy_sector_only": False},
        {"family": "B1", "readiness": "ready_for_discovery" if sector_pit and catalyst_pit else "needs_sector_map", "reason": "no point-in-time sector/catalyst database found; cluster proxy allowed only as exploratory", "max_verdict_cap": "build_sector_catalyst_data_next", "proxy_sector_only": True},
        {"family": "C2", "readiness": "ready_for_discovery" if sector_pit and catalyst_pit else "needs_catalyst_database", "reason": "no point-in-time catalyst database found; post-catalyst base cannot be ranked as true catalyst strategy", "max_verdict_cap": "build_sector_catalyst_data_next", "proxy_sector_only": True},
    ]
    return rows


def write_budget_manifest(ctx: RunContext) -> None:
    alloc = allocate_family_budgets(ctx.args.discovery_budget, ctx.args.family_budget)
    readiness = {r["family"]: r for r in family_readiness_rows()}
    rows: list[dict[str, Any]] = []
    groups = {
        "A1_A3_A4": ["A1", "A3", "A4"],
        "A2": ["A2"],
        "D3_D4_E1": ["D3", "D4", "E1"],
        "H3": ["H3"],
        "F1_G1": ["F1", "G1"],
        "H1": ["H1"],
        "B1_C2": ["B1", "C2"],
        "adaptive_refinement": ["adaptive_refinement"],
    }
    for group, fams in groups.items():
        blocked = [f for f in fams if readiness.get(f, {}).get("readiness", "ready_for_discovery") not in {"ready_for_discovery", "path_only"}]
        status = "blocked_or_contract_only" if blocked and group in {"H1", "B1_C2"} else "active_or_proxy_limited"
        rows.append({
            "budget_group": group,
            "families": ";".join(fams),
            "max_budget": FAMILY_MAX_BUDGETS[group],
            "allocated_candidates": alloc.get(group, 0),
            "blocked_families": ";".join(blocked),
            "status": status,
            "quality_note": "full coverage rerun required before prelead ranking",
        })
    write_csv(ctx.run_root / "budget/family_budget_manifest.csv", rows)
    est = estimate_total_output_gb(ctx)
    write_text(ctx.run_root / "budget/output_size_estimate.md", f"# Output Size Estimate\n\n- estimated total output GB: `{est:.2f}`\n- max output GB CLI: `{ctx.args.max_output_gb}`\n- per-stage hard stop without override: `20GB`\n- fits resource limits: `{est <= ctx.args.max_output_gb}`\n- discovery budget: `{ctx.args.discovery_budget}`\n- refinement budget: `{ctx.args.refine_budget}`\n- note: discovery sampling cannot become lead evidence until full-coverage rerun passes.\n")


def estimate_total_output_gb(ctx: RunContext) -> float:
    return 1.0 if ctx.args.smoke else min(8.0 + ctx.args.discovery_budget / 1000.0, ctx.args.max_output_gb)


def cost_bps_for_tier(tier: Any) -> float:
    return {"A": 16.0, "B": 30.0, "C": 50.0, "D": 78.0}.get(str(tier), 93.0)


def surface_return_r(df: pd.DataFrame, horizon: str, target_r: float, stop_mult: float = 1.0, cost_mult: float = 1.0, branch: str = "pessimistic") -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    risk = pd.to_numeric(df.get("reference_risk_bps"), errors="coerce").fillna(100.0).clip(lower=1.0) * float(stop_mult)
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
    ret = ret - (costs / risk)
    return ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def summarize_returns(ret: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    pos = x[x > 0].sum()
    neg = -x[x < 0].sum()
    eq = x.cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    return {
        "events": int(len(x)),
        "net_R": float(x.sum()),
        "mean_R": float(x.mean()) if len(x) else 0.0,
        "median_R": float(x.median()) if len(x) else 0.0,
        "PF": float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0),
        "hit_rate": float((x > 0).mean()) if len(x) else 0.0,
        "max_dd_R_proxy": dd,
    }


def apply_candidate_filter(df: pd.DataFrame, cand: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    fam = str(cand.get("family", ""))
    out = df
    if fam in {"D3", "E1", "A2"}:
        out = out[out["family"].eq(fam)]
    elif fam == "D4":
        out = out[out["family"].isin(["D3", "E1"])]
    elif fam in {"A1", "A3", "A4"}:
        out = out[out["family"].eq("A2")]
    elif fam == "H3":
        out = out[out["family"].isin(["D3", "E1", "D1"])]
    elif fam in {"B1", "C2"}:
        out = out[out["family"].isin(["A2", "D3", "E1"])]
    elif fam in {"H1", "F1", "G1"}:
        return out.iloc[0:0].copy()
    tier = cand.get("tier_filter", "any")
    if tier == "C":
        out = out[out.get("liquidity_tier").eq("C")]
    elif tier == "A_B":
        out = out[out.get("liquidity_tier").isin(["A", "B"])]
    elif tier == "not_D":
        out = out[~out.get("liquidity_tier").eq("D")]
    if cand.get("parent_gate") == "non_deteriorating" and "btc_eth_non_deteriorating" in out.columns:
        out = out[out["btc_eth_non_deteriorating"].fillna(False)]
    elif cand.get("parent_gate") == "up_or_neutral" and "parent_trend_label" in out.columns:
        out = out[out["parent_trend_label"].isin(["strong_up", "neutral_up"])]
    elif cand.get("parent_gate") == "stress_ok" and "parent_trend_label" in out.columns:
        out = out[~out["parent_trend_label"].eq("down")]
    if cand.get("deleveraged_gate") and "deleveraged_2of4" in out.columns:
        out = out[out["deleveraged_2of4"].fillna(False)]
    if cand.get("funding_gate") == "not_high" and "funding_percentile_bucket" in out.columns:
        out = out[~out["funding_percentile_bucket"].eq("high")]
    elif cand.get("funding_gate") == "low_mid" and "funding_percentile_bucket" in out.columns:
        out = out[out["funding_percentile_bucket"].isin(["low", "mid", "unknown"])]
    if cand.get("price_oi_gate") == "price_down_oi_down" and "price_oi_matrix_24h" in out.columns:
        out = out[out["price_oi_matrix_24h"].eq("price_down_oi_down")]
    elif cand.get("price_oi_gate") == "not_price_down_oi_up" and "price_oi_matrix_24h" in out.columns:
        out = out[~out["price_oi_matrix_24h"].eq("price_down_oi_up")]
    if cand.get("liquidity_quality_gate") == "avoid_thin" and "liquidity_quality_label" in out.columns:
        out = out[~out["liquidity_quality_label"].eq("thin_proxy")]
    if cand.get("bad_wick_gate") == "avoid_top1" and "bad_wick_proxy_label" in out.columns:
        out = out[~out["bad_wick_proxy_label"].eq("top_1pct_range_proxy")]
    return out.copy()


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
    month = float(tmp.groupby(pd.to_datetime(tmp["decision_ts"], utc=True).dt.to_period("M"))["_positive_R"].sum().max() / denom)
    if "event_id" in tmp:
        cluster = float(tmp.groupby("event_id")["_positive_R"].sum().max() / denom)
    else:
        cluster = 0.0
    return {"max_symbol_positive_share": sym, "max_month_positive_share": month, "max_event_cluster_positive_share": cluster}


def score_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    events = int(row.get("events", 0) or 0)
    net = float(row.get("net_R", 0.0) or 0.0)
    pf = float(row.get("PF", 0.0) or 0.0)
    mean_r = float(row.get("mean_R", 0.0) or 0.0)
    median_r = float(row.get("median_R", 0.0) or 0.0)
    liq = int(row.get("liquidation_count", 0) or 0)
    sym_share = float(row.get("max_symbol_positive_share", 0.0) or 0.0)
    month_share = float(row.get("max_month_positive_share", 0.0) or 0.0)
    cluster_share = float(row.get("max_event_cluster_positive_share", 0.0) or 0.0)
    hard: list[str] = []
    if events <= 0:
        hard.append("no_events")
    if net <= 0:
        hard.append("net_R_nonpositive")
    if pf <= 1.0:
        hard.append("PF_lte_1")
    if liq > 0:
        hard.append("liquidation_count_positive")
    if sym_share > 0.30:
        hard.append("symbol_concentration_gt_30pct")
    if month_share > 0.40:
        hard.append("month_concentration_gt_40pct")
    if cluster_share > 0.35:
        hard.append("event_cluster_concentration_gt_35pct")
    score = mean_r * 40.0 + median_r * 15.0 + min(pf, 5.0) * 5.0 + math.log1p(events) * 0.4
    score -= liq * 1000.0 + sym_share * 6.0 + month_share * 4.0 + cluster_share * 4.0
    return {"robustness_score": float(score), "lead_rankable": len(hard) == 0, "hard_penalty_reasons": ";".join(hard)}


def candidate_result(df: pd.DataFrame, cand: Mapping[str, Any]) -> dict[str, Any]:
    sub = apply_candidate_filter(df, cand)
    horizon = str(cand.get("horizon", "24h"))
    ret = surface_return_r(sub, horizon, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
    sm = summarize_returns(ret)
    liq_col = f"{horizon}_liquidation_10x"
    liq = int(sub.get(liq_col, pd.Series(False, index=sub.index)).fillna(False).astype(bool).sum()) if not sub.empty else 0
    conc = concentration_stats(sub, ret)
    proxy_mark_share = 1.0
    if not sub.empty and "mark_path_status" in sub.columns:
        ok = sub["mark_path_status"].astype(str).str.contains("ok|available", case=False, regex=True, na=False)
        proxy_mark_share = float(1.0 - ok.mean())
    out = {**cand, **sm, "liquidation_count": liq, **conc, "proxy_mark_or_liquidation_evidence_share": proxy_mark_share}
    out.update(score_candidate_row(out))
    return out


def generate_candidates(budgets: Mapping[str, int], seed: int, readiness: Sequence[Mapping[str, Any]], smoke: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ready = {r["family"]: r for r in readiness}
    group_families = {
        "D3_D4_E1": ["D3", "D4", "E1"],
        "A1_A3_A4": ["A1", "A3", "A4"],
        "A2": ["A2"],
        "H3": ["H3"],
        "F1_G1": ["F1", "G1"],
        "H1": ["H1"],
        "B1_C2": ["B1", "C2"],
    }
    rows: list[dict[str, Any]] = []
    for group, fams in group_families.items():
        n = int(budgets.get(group, 0))
        if smoke:
            n = min(n, 8)
        active_fams = [f for f in fams if ready.get(f, {}).get("readiness") in {"ready_for_discovery", "path_only"}]
        if not active_fams:
            continue
        for fam in active_fams:
            rows.append({"candidate_type": "no_regime_baseline", "budget_group": group, "family": fam, "tier_filter": "any", "parent_gate": "none", "deleveraged_gate": False, "funding_gate": "none", "price_oi_gate": "none", "liquidity_quality_gate": "none", "bad_wick_gate": "none", "horizon": "24h", "target_r": 2.0, "stop_mult": 1.0, "cost_mult": 1.0})
        for _ in range(max(0, n - len(active_fams))):
            fam = rng.choice(active_fams)
            rows.append({
                "candidate_type": "regime_conditioned",
                "budget_group": group,
                "family": fam,
                "tier_filter": rng.choice(["C", "A_B", "not_D", "any"]),
                "parent_gate": rng.choice(["none", "non_deteriorating", "up_or_neutral", "stress_ok"]),
                "deleveraged_gate": rng.choice([False, True]) if fam in {"D3", "D4", "E1", "H3"} else False,
                "funding_gate": rng.choice(["none", "not_high", "low_mid"]),
                "price_oi_gate": rng.choice(["none", "price_down_oi_down", "not_price_down_oi_up"]),
                "liquidity_quality_gate": rng.choice(["none", "avoid_thin"]),
                "bad_wick_gate": rng.choice(["none", "avoid_top1"]),
                "horizon": rng.choice(["30m", "1h", "2h", "6h", "24h", "72h"]),
                "target_r": rng.choice([1.0, 2.0, 3.0, 5.0]),
                "stop_mult": rng.choice([0.5, 1.0, 1.5, 2.0]),
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
        r["proxy_sector_only"] = bool(ready.get(r["family"], {}).get("proxy_sector_only", False))
        r["readiness"] = ready.get(r["family"], {}).get("readiness", "unknown")
        r["max_verdict_cap"] = ready.get(r["family"], {}).get("max_verdict_cap", "unknown")
        out.append(r)
    return out


def top_sweep(ctx: RunContext, n: int = 12) -> pd.DataFrame:
    paths = [ctx.run_root / "preleads/full_coverage_prelead_summary.csv", ctx.run_root / "sweep/sweep_summary.csv"]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                return df.sort_values(["lead_rankable", "robustness_score"], ascending=[False, False]).head(n)
    return pd.DataFrame()


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    roots = [FIRST_SCREEN_ROOT, PATH_DIAG_ROOT, D1_ROOT, PILOT_1M_ROOT, F1G1_ROOT]
    manifest = []
    for root in roots:
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        manifest.append({"root": str(root), "exists": root.exists(), "file_count": len(files), "sample_files": [str(p.relative_to(root)) for p in files[:25]]})
    write_json(ctx.run_root / "preflight/prior_artifact_manifest.json", {"artifacts": manifest})
    write_budget_manifest(ctx)
    path_report = read_text_safe(PATH_DIAG_ROOT / "QLMG_PATH_DIAGNOSTICS_EXIT_SURFACE_REPORT.md", 2000)
    f1g1_decision = read_text_safe(F1G1_ROOT / "decision_summary.json", 1000)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard Report\n\n- free_disk_gb: `{snap.free_gb:.2f}`\n- hard_stop_free_gb: `5`\n- warning_free_gb: `7`\n- max_stage_output_without_override_gb: `20`\n- default_max_output_gb: `{ctx.args.max_output_gb}`\n")
    write_text(ctx.run_root / "preflight/prior_result_audit.md", f"# Prior Result Audit\n\n- path diagnostics root: `{PATH_DIAG_ROOT}`\n- path diagnostic excerpt available: `{bool(path_report)}`\n- F1/G1 decision excerpt: `{f1g1_decision[:300]}`\n- prior D1 is not rescued here; D1 may appear only as historical/path context.\n- prior matched-null summaries are not sufficient for prelead validation; this run performs candidate-specific fresh matching from available null path rows.\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight Report\n\n- run root: `{ctx.run_root}`\n- current free disk GB: `{snap.free_gb:.2f}`\n- requested window: `{ctx.start}` to `{ctx.end}`\n- protected holdout starts: `{FINAL_HOLDOUT_START}`\n- discovery budget: `{ctx.args.discovery_budget}`\n- refinement budget: `{ctx.args.refine_budget}`\n- no final holdout, live system, or large download is used.\n")


def stage_telegram(ctx: RunContext) -> None:
    if ctx.args.require_telegram and not ctx.notifier.remote_available and not ctx.args.allow_no_telegram and not ctx.args.smoke:
        raise RuntimeError("remote Telegram is required for full launch but is unavailable; pass --allow-no-telegram only if intentionally running local-only")
    missing = ctx.notifier.missing or "none"
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness Report\n\n- status: `{ctx.notifier.status}`\n- remote_available: `{ctx.notifier.remote_available}`\n- require_telegram: `{ctx.args.require_telegram}`\n- allow_no_telegram: `{ctx.args.allow_no_telegram}`\n- missing/disabled reason: `{missing}`\n- local JSONL log: `notifications/telegram_events.jsonl`\n- secrets persisted: `false`\n")
    watch = f"""# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)
    full_cmd = f"bash tools/run_qlmg_alpha_discovery_marathon_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --discovery-budget {ctx.args.discovery_budget} --refine-budget {ctx.args.refine_budget} --nulls-per-event {ctx.args.nulls_per_event} --use-targeted-1m-if-overlap --require-telegram --seed {ctx.args.seed} --launch-tmux"
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nThe wrapper must not launch a full unattended run unless invoked with `--launch-tmux`.\n\nFull command:\n\n```bash\n{full_cmd}\n```\n")


def stage_seal(ctx: RunContext) -> None:
    check = {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "requested_start": str(ctx.start), "requested_end": str(ctx.end), "pre_holdout_read_smoke": bool(ctx.end < FINAL_HOLDOUT_START), "protected_read_smoke": "blocked_by_policy", "status": "pass" if ctx.end < FINAL_HOLDOUT_START else "fail"}
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    if check["status"] != "pass":
        raise RuntimeError("seal guard failed")
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard Report\n\n- protected slice starts: `{FINAL_HOLDOUT_START}`\n- allowed data end: `{SCREENING_END}`\n- requested end: `{ctx.end}`\n- status: `pass`\n")


def stage_scoring(ctx: RunContext) -> None:
    contract = {
        "negative_net_R_can_rank_as_lead": False,
        "PF_lte_1_can_rank_as_lead": False,
        "liquidation_count_positive_blocks_validation_style_lead": True,
        "proxy_mark_liquidation_max_verdict": "promote_to_targeted_execution_data_collection",
        "proxy_lifecycle_or_execution_cost_max_verdict": "promote_to_targeted_execution_data_collection",
        "missing_top_of_book_depth_max_verdict": "promote_to_targeted_execution_data_collection",
        "concentration_limits": {"symbol_positive_R_share": 0.30, "month_positive_R_share": 0.40, "event_cluster_positive_R_share": 0.35},
        "matched_null_failure_blocks_lead": True,
        "isolated_parameter_point_blocks_lead": True,
        "sampled_discovery_requires_full_coverage_rerun": True,
        "same_bar_rankable_branch": "pessimistic",
    }
    write_json(ctx.run_root / "scoring/scoring_contract.json", contract)
    write_text(ctx.run_root / "scoring/scoring_rule_report.md", "# Scoring And Promotion Rule Fix\n\nThe score is diagnostic only. Hard lead gates reject nonpositive net_R, PF <= 1, liquidation, concentration breaches, matched-null failure, sampled-only evidence, and isolated one-point neighborhoods. Optimistic same-bar branches are never rankable.\n")


def stage_readiness(ctx: RunContext) -> None:
    rows = family_readiness_rows()
    write_csv(ctx.run_root / "readiness/family_readiness_matrix.csv", rows)
    write_text(ctx.run_root / "readiness/family_readiness_report.md", "# Family Readiness Report\n\n- H1 is not tested as a 5m lead/lag candidate; it requires adequate 1m coverage.\n- B1/C2 cannot be ranked as true catalyst/sector strategies without point-in-time sector/catalyst data. Proxy rotation outputs are `proxy_sector_only`.\n- F1/G1 use prior short-unblock evidence and remain capped by current translation quality.\n")


def stage_regime(ctx: RunContext) -> None:
    df = load_path_metrics(ctx, null=False)
    reg = build_regime_panel(df, min_history=3 if ctx.args.smoke else 10)
    validate_no_protected(reg, ["decision_ts", "feature_ts"])
    (ctx.run_root / "regime").mkdir(parents=True, exist_ok=True)
    reg.to_parquet(ctx.run_root / "regime/regime_feature_panel.parquet", index=False)
    regime_feature_dictionary().to_csv(ctx.run_root / "regime/regime_feature_dictionary.csv", index=False)
    REGIME_ROWS = []
    for col in ["parent_trend_label", "btc_eth_regime_label", "realized_vol_bucket", "liquidity_quality_label", "funding_percentile_bucket", "price_oi_matrix_24h", "deleveraged_2of4", "session_bucket", "listing_age_bucket"]:
        if col in reg.columns:
            for label, count in reg[col].astype(str).value_counts(dropna=False).head(30).items():
                REGIME_ROWS.append({"feature": col, "label": label, "rows": int(count), "share": float(count / max(len(reg), 1))})
    write_csv(ctx.run_root / "regime/regime_feature_coverage_summary.csv", REGIME_ROWS)
    write_text(ctx.run_root / "regime/regime_stack_qc_report.md", f"# Regime Stack Refresh Report\n\n- source rows: `{len(df)}`\n- regime rows: `{len(reg)}`\n- point-in-time rule: `feature_ts <= decision_ts`\n- sector/catalyst layer: `schema_only_or_proxy_only`\n")


def stage_contracts(ctx: RunContext) -> None:
    out_dir = ctx.run_root / "contracts/family_contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    readiness = {r["family"]: r for r in family_readiness_rows()}
    families = ["A1", "A3", "A4", "A2", "D3", "D4", "E1", "H3", "F1", "G1", "H1", "B1", "C2"]
    rows = []
    for fam in families:
        contract = {
            "family": fam,
            "created_at_utc": utc_now(),
            "hypothesis_status": "diagnostic_discovery_only",
            "protected_holdout_start": "2026-01-01T00:00:00Z",
            "allowed_data_end": "2025-12-31T23:59:59Z",
            "readiness": readiness.get(fam, {}).get("readiness", "unknown"),
            "proxy_sector_only": bool(readiness.get(fam, {}).get("proxy_sector_only", False)),
            "max_verdict_cap": readiness.get(fam, {}).get("max_verdict_cap", "unknown"),
            "no_live_trading": True,
            "no_sealed_validation": True,
            "entry_generation": "use existing path/event artifacts or family-specific prior source; no final holdout",
            "matched_null_required": True,
            "full_coverage_required_for_prelead": True,
            "cost_funding_liquidation": "proxy-grade unless local data proves otherwise",
        }
        path = out_dir / f"{fam}.json"
        path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({"family": fam, "path": str(path), "readiness": contract["readiness"], "proxy_sector_only": contract["proxy_sector_only"], "max_verdict_cap": contract["max_verdict_cap"]})
    write_csv(ctx.run_root / "contracts/family_contract_summary.csv", rows)


def stage_events(ctx: RunContext) -> None:
    df = load_path_metrics(ctx, null=False)
    validate_no_protected(df, ["decision_ts", "entry_ts"])
    out = df.copy()
    out["source_run_root"] = str(PATH_DIAG_ROOT)
    out["discovery_scope"] = "full_prior_path_metrics_with_requested_window_filters"
    (ctx.run_root / "events").mkdir(parents=True, exist_ok=True)
    out.to_parquet(ctx.run_root / "events/discovery_event_ledger.parquet", index=False)
    summary = out.groupby(["family", "side", "liquidity_tier"], dropna=False).agg(events=("event_id", "count"), symbols=("symbol", "nunique"), first_ts=("decision_ts", "min"), last_ts=("decision_ts", "max")).reset_index()
    write_df_csv(ctx.run_root / "events/discovery_event_summary.csv", summary)
    write_text(ctx.run_root / "events/discovery_event_report.md", f"# Discovery Event Generation\n\n- rows written: `{len(out)}`\n- old signal cap: `not applied in this stage; using prior full path metrics within selected date/symbol window`\n- final holdout rows: `0`\n")


def stage_path_screen(ctx: RunContext) -> None:
    df = load_events(ctx)
    rows = []
    for keys, sub in df.groupby(["family", "side", "liquidity_tier"], dropna=False):
        fam, side, tier = keys
        rec = {"family": fam, "side": side, "liquidity_tier": tier, "events": len(sub), "symbols": sub["symbol"].nunique()}
        for h in ["30m", "1h", "6h", "24h", "72h"]:
            if f"{h}_mfe_bps" in sub:
                rec[f"{h}_median_mfe_bps"] = float(pd.to_numeric(sub[f"{h}_mfe_bps"], errors="coerce").median())
                rec[f"{h}_median_mae_bps"] = float(pd.to_numeric(sub[f"{h}_mae_bps"], errors="coerce").median())
                rec[f"{h}_pos1R_before_neg1R_share"] = float(sub.get(f"{h}_pos1R_before_neg1R", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean())
        rows.append(rec)
    write_csv(ctx.run_root / "path_screen/path_opportunity_summary.csv", rows)
    write_text(ctx.run_root / "path_screen/path_opportunity_report.md", "# Path-First Opportunity Screen\n\nThis stage measures path asymmetry only. It does not declare executable strategy validity. Families failing later matched-null validation cannot advance.\n")


def stage_cube(ctx: RunContext) -> None:
    df = load_event_regime(ctx)
    dims = [d for d in ["family", "side", "liquidity_tier", "parent_trend_label", "realized_vol_bucket", "liquidity_quality_label", "price_oi_matrix_24h", "funding_percentile_bucket", "deleveraged_2of4", "session_bucket"] if d in df.columns]
    rows = []
    for keys, sub in df.groupby(dims, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(dims, keys))
        rec.update({
            "events": int(len(sub)),
            "mean_24h_mfe_bps": float(pd.to_numeric(sub.get("24h_mfe_bps"), errors="coerce").mean()),
            "median_24h_mfe_bps": float(pd.to_numeric(sub.get("24h_mfe_bps"), errors="coerce").median()),
            "mean_24h_mae_bps": float(pd.to_numeric(sub.get("24h_mae_bps"), errors="coerce").mean()),
            "pos1R_before_neg1R_24h_share": float(sub.get("24h_pos1R_before_neg1R", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "liquidation_10x_24h_count": int(sub.get("24h_liquidation_10x", pd.Series(False, index=sub.index)).fillna(False).astype(bool).sum()),
        })
        rows.append(rec)
    cube = pd.DataFrame(rows)
    if not cube.empty:
        cube = cube.sort_values(["events", "mean_24h_mfe_bps"], ascending=[False, False])
    (ctx.run_root / "cubes").mkdir(parents=True, exist_ok=True)
    cube.to_parquet(ctx.run_root / "cubes/regime_conditional_cube.parquet", index=False)
    write_df_csv(ctx.run_root / "cubes/regime_conditional_cube_summary.csv", cube.head(500))
    write_text(ctx.run_root / "cubes/regime_conditional_cube_report.md", f"# Regime Conditional Cube\n\n- joined rows: `{len(df)}`\n- cube rows: `{len(cube)}`\n- rare labels are diagnostics-only unless full-coverage rerun and null validation pass.\n")


def stage_sweep(ctx: RunContext) -> None:
    readiness = family_readiness_rows()
    budgets = allocate_family_budgets(ctx.args.discovery_budget if not ctx.args.smoke else min(ctx.args.discovery_budget, 40), ctx.args.family_budget)
    cands = generate_candidates(budgets, ctx.args.seed, readiness, smoke=ctx.args.smoke)
    write_csv(ctx.run_root / "sweep/candidate_registry.csv", cands)
    df = load_event_regime(ctx)
    rows = [candidate_result(df, c) for c in cands]
    res = pd.DataFrame(rows)
    (ctx.run_root / "sweep").mkdir(parents=True, exist_ok=True)
    res.to_parquet(ctx.run_root / "sweep/sweep_results.parquet", index=False)
    sort_cols = ["lead_rankable", "robustness_score"] if "lead_rankable" in res else ["robustness_score"]
    res_sorted = res.sort_values(sort_cols, ascending=[False, False] if len(sort_cols) == 2 else [False]) if not res.empty else res
    write_df_csv(ctx.run_root / "sweep/sweep_summary.csv", res_sorted.head(500))
    write_text(ctx.run_root / "sweep/sweep_report.md", f"# Bounded Sobol-Style Entry/Exit Search\n\n- candidates logged before evaluation: `{len(cands)}`\n- candidates evaluated: `{len(res)}`\n- Cartesian broad grid: `false`\n- rankable branch: `pessimistic`\n- capped/sampled discovery requires full-coverage rerun before any prelead label.\n")


def stage_full_coverage(ctx: RunContext) -> None:
    sweep = pd.read_parquet(ctx.run_root / "sweep/sweep_results.parquet") if (ctx.run_root / "sweep/sweep_results.parquet").exists() else pd.DataFrame()
    if sweep.empty:
        write_csv(ctx.run_root / "preleads/full_coverage_prelead_summary.csv", [])
        write_text(ctx.run_root / "preleads/full_coverage_prelead_report.md", "# Full Coverage Prelead Rerun\n\nNo sweep rows.\n")
        return
    active = sweep[sweep["lead_rankable"].fillna(False)] if "lead_rankable" in sweep.columns else pd.DataFrame()
    if active.empty:
        # Keep a small positive diagnostic list, but mark it non-rankable.
        active = sweep[sweep["net_R"].gt(0)].sort_values("robustness_score", ascending=False).head(6)
    else:
        active = active.sort_values("robustness_score", ascending=False).groupby("family", group_keys=False).head(3).head(12)
    df = load_event_regime(ctx)
    rows = []
    for cand in active.to_dict("records"):
        res = candidate_result(df, cand)
        res["full_coverage_status"] = "pass" if res.get("lead_rankable") else "fail_hard_gate"
        res["sampled_discovery_promoted_to_prelead"] = bool(res.get("lead_rankable"))
        rows.append(res)
    out = pd.DataFrame(rows)
    write_df_csv(ctx.run_root / "preleads/full_coverage_prelead_summary.csv", out)
    write_text(ctx.run_root / "preleads/full_coverage_prelead_report.md", f"# Full Coverage Prelead Rerun\n\n- candidates rerun: `{len(out)}`\n- rule: sampled/capped discovery can only advance after this full eligible pre-holdout rerun passes hard gates.\n")


def matched_null_for_candidate(event_df: pd.DataFrame, null_df: pd.DataFrame, cand: Mapping[str, Any], nulls_per_event: int, seed: int) -> dict[str, Any]:
    ev = apply_candidate_filter(event_df, cand)
    nu_pool = apply_candidate_filter(null_df, cand)
    h = cand.get("horizon", "24h")
    if ev.empty or nu_pool.empty:
        ev_ret = surface_return_r(ev, h, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
        ev_sm = summarize_returns(ev_ret)
        return {"candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "event_count": ev_sm["events"], "null_count": 0, "effective_nulls_per_event": 0.0, "event_mean_R": ev_sm["mean_R"], "null_mean_R": np.nan, "event_minus_null_mean_R": np.nan, "beats_fresh_matched_null": False}
    target_n = min(len(nu_pool), max(1, int(nulls_per_event)) * len(ev))
    rng = np.random.default_rng(seed + int(stable_hash(cand.get("candidate_id", ""), 8), 16) % 100000)
    take_idx = rng.choice(nu_pool.index.to_numpy(), size=target_n, replace=False if target_n <= len(nu_pool) else True)
    nu = nu_pool.loc[take_idx]
    ev_ret = surface_return_r(ev, h, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
    nu_ret = surface_return_r(nu, h, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
    ev_sm = summarize_returns(ev_ret)
    nu_sm = summarize_returns(nu_ret)
    eff = float(len(nu) / max(len(ev), 1))
    return {
        "candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "event_count": ev_sm["events"], "null_count": nu_sm["events"], "effective_nulls_per_event": eff,
        "event_mean_R": ev_sm["mean_R"], "null_mean_R": nu_sm["mean_R"], "event_net_R": ev_sm["net_R"], "null_net_R": nu_sm["net_R"],
        "event_minus_null_mean_R": ev_sm["mean_R"] - nu_sm["mean_R"], "event_minus_null_net_R": ev_sm["net_R"] - nu_sm["net_R"],
        "beats_fresh_matched_null": bool(ev_sm["mean_R"] > nu_sm["mean_R"] and ev_sm["net_R"] > nu_sm["net_R"]),
        "null_support_cap": "full_null_support" if eff >= 3 else "limited_null_support_caps_verdict",
    }


def stage_matched_null(ctx: RunContext) -> None:
    pre = top_sweep(ctx, 12)
    ev_df = load_event_regime(ctx, null=False)
    nu_df = load_event_regime(ctx, null=True)
    rows = [matched_null_for_candidate(ev_df, nu_df, c, ctx.args.nulls_per_event, ctx.args.seed) for c in pre.to_dict("records")]
    write_csv(ctx.run_root / "matched_null/prelead_matched_null_summary.csv", rows)
    write_json(ctx.run_root / "matched_null/fresh_matched_null_policy.json", {"requested_nulls_per_event": ctx.args.nulls_per_event, "source": "candidate_specific_fresh_resample_from_prior_null_path_rows", "prior_one_null_summary_reused": False, "cap_if_effective_nulls_lt_3": "continue_alpha_or_targeted_data_only"})
    write_text(ctx.run_root / "matched_null/prelead_matched_null_report.md", "# Fresh Matched Null Validation For Preleads\n\nCandidate-specific null rows are freshly sampled from the prior null path-metric pool, not reused from prior summary tables. If effective null support is below 3/event, verdicts are capped as specified.\n")


def stage_one_minute(ctx: RunContext) -> None:
    pre = top_sweep(ctx, 12)
    impact = read_csv_safe(PILOT_1M_ROOT / "impact/one_minute_impact_summary.csv")
    rows = []
    ev_df = load_event_regime(ctx, null=False)
    for cand in pre.to_dict("records"):
        sub = apply_candidate_filter(ev_df, cand)
        ids = set(sub.get("event_id", pd.Series(dtype=str)).astype(str).head(20000))
        hit = impact[impact.get("event_id", pd.Series(dtype=str)).astype(str).isin(ids)] if not impact.empty and "event_id" in impact.columns else pd.DataFrame()
        rows.append({"candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "events_checked": len(ids), "pilot_1m_overlap_rows": int(len(hit)), "targeted_1m_status": "overlap_available" if len(hit) else "no_overlap_request_needed", "material_same_bar_share": float(hit.get("material_same_bar_resolution_needed", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(hit) else np.nan})
    write_csv(ctx.run_root / "one_minute/prelead_1m_overlay_summary.csv", rows)
    write_text(ctx.run_root / "one_minute/prelead_1m_overlay_report.md", "# Targeted 1m Overlay For Preleads\n\nExisting targeted 1m pilot data is overlay evidence only. No new downloads are performed. Missing overlap caps execution-grade conclusions.\n")
    write_csv(ctx.run_root / "one_minute/additional_1m_window_request.csv", [])


def stage_stress(ctx: RunContext) -> None:
    pre = top_sweep(ctx, 12)
    df = load_event_regime(ctx)
    rows = []
    for cand in pre.to_dict("records"):
        for scenario, cmult in [("base", 1.0), ("cost_x1p25", 1.25), ("cost_x1p5", 1.5), ("cost_x2", 2.0), ("plus_10bps_proxy", float(cand.get("cost_mult", 1.0)) + 0.2), ("plus_25bps_proxy", float(cand.get("cost_mult", 1.0)) + 0.5)]:
            c = dict(cand)
            c["cost_mult"] = cmult
            res = candidate_result(df, c)
            rows.append({"scenario": scenario, **{k: res.get(k) for k in ["candidate_id", "family", "events", "net_R", "mean_R", "median_R", "PF", "liquidation_count", "lead_rankable", "hard_penalty_reasons"]}})
    write_csv(ctx.run_root / "stress/execution_cost_liquidation_stress_summary.csv", rows)
    write_text(ctx.run_root / "stress/execution_cost_liquidation_stress_report.md", "# Execution Cost And Liquidation Stress\n\nCandidates failing base cost * 1.25, showing liquidation, or depending on proxy mark/liquidation cannot advance beyond targeted data collection.\n")


def cpcv_rows_for_candidate(df: pd.DataFrame, cand: Mapping[str, Any]) -> list[dict[str, Any]]:
    sub = apply_candidate_filter(df, cand).sort_values("decision_ts")
    if sub.empty:
        return []
    h = cand.get("horizon", "24h")
    ret = surface_return_r(sub, h, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
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
        rows.append({"candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "test_blocks": ";".join(map(str, combo)), **sm, **conc})
    return rows


def stage_validation(ctx: RunContext) -> None:
    pre = top_sweep(ctx, 12)
    df = load_event_regime(ctx)
    rows = []
    for cand in pre.to_dict("records"):
        rows.extend(cpcv_rows_for_candidate(df, cand))
    cpcv = pd.DataFrame(rows)
    write_df_csv(ctx.run_root / "validation/cpcv_summary.csv", cpcv)
    wf = []
    if not cpcv.empty:
        for cid, g in cpcv.groupby("candidate_id"):
            wf.append({"candidate_id": cid, "family": g["family"].iloc[0], "paths": len(g), "median_path_net_R": float(g["net_R"].median()), "percent_positive_paths": float((g["net_R"] > 0).mean()), "worst_path_net_R": float(g["net_R"].min()), "path_dispersion": float(g["net_R"].std(ddof=0)), "passes_55pct_positive_paths": bool((g["net_R"] > 0).mean() >= 0.55)})
    write_csv(ctx.run_root / "validation/walk_forward_summary.csv", wf)
    sweep = pd.read_parquet(ctx.run_root / "sweep/sweep_results.parquet") if (ctx.run_root / "sweep/sweep_results.parquet").exists() else pd.DataFrame()
    prelead = read_csv_safe(ctx.run_root / "preleads/full_coverage_prelead_summary.csv")
    family_counts = sweep.groupby("family")["candidate_id"].count().to_dict() if not sweep.empty else {}
    regime_counts = sweep.groupby(["family", "parent_gate"], dropna=False)["candidate_id"].count().reset_index().to_dict("records") if not sweep.empty else []
    write_json(ctx.run_root / "validation/family_multiple_testing_summary.json", {"generated_candidates": int(len(sweep)), "candidates_by_family": family_counts, "candidates_by_family_regime": regime_counts, "preleads": int(len(prelead)), "full_coverage_reruns": int(len(prelead)), "selection_burden_note": "All generated candidates count against interpretation; preleads require full coverage, null, stress, and CPCV gates."})
    write_text(ctx.run_root / "validation/family_multiple_testing_report.md", f"# Family Multiple Testing Report\n\n- generated candidates: `{len(sweep)}`\n- candidates by family: `{family_counts}`\n- preleads: `{len(prelead)}`\n- full-coverage reruns: `{len(prelead)}`\n- selection burden: all generated candidates and regime gates are disclosed; no best-run-only interpretation is allowed.\n")
    write_text(ctx.run_root / "validation/validation_report.md", f"# Walk-Forward/CPCV And Overfit Controls\n\n- CPCV rows: `{len(cpcv)}`\n- purge/embargo: block-level proxy; this is diagnostic, not final validation.\n")


def stage_portfolio(ctx: RunContext) -> None:
    pre = top_sweep(ctx, 12)
    matched = read_csv_safe(ctx.run_root / "matched_null/prelead_matched_null_summary.csv")
    valid = read_csv_safe(ctx.run_root / "validation/walk_forward_summary.csv")
    stress = read_csv_safe(ctx.run_root / "stress/execution_cost_liquidation_stress_summary.csv")
    df = load_event_regime(ctx)
    rows = []
    for cand in pre.to_dict("records"):
        cid = cand.get("candidate_id")
        m = matched[matched.get("candidate_id", pd.Series(dtype=str)).eq(cid)] if not matched.empty else pd.DataFrame()
        v = valid[valid.get("candidate_id", pd.Series(dtype=str)).eq(cid)] if not valid.empty else pd.DataFrame()
        s = stress[(stress.get("candidate_id", pd.Series(dtype=str)).eq(cid)) & (stress.get("scenario", pd.Series(dtype=str)).eq("cost_x1p25"))] if not stress.empty else pd.DataFrame()
        eligible = (not m.empty and bool(m.iloc[0].get("beats_fresh_matched_null", False)) and float(m.iloc[0].get("effective_nulls_per_event", 0)) >= 1.0 and not v.empty and bool(v.iloc[0].get("passes_55pct_positive_paths", False)) and not s.empty and float(s.iloc[0].get("net_R", -1)) > 0)
        if not eligible:
            rows.append({"candidate_id": cid, "family": cand.get("family"), "portfolio_overlay_status": "skipped_failed_prior_gates"})
            continue
        sub = apply_candidate_filter(df, cand).sort_values("decision_ts")
        ret = surface_return_r(sub, cand.get("horizon", "24h"), float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
        for equity in [200, 500, 1000]:
            for risk_pct in [0.025, 0.05, 0.10, 0.15, 0.20]:
                eq = float(equity)
                peak = eq
                max_dd = 0.0
                ruin = False
                for r in ret.head(10000):
                    eq += eq * risk_pct * float(r)
                    peak = max(peak, eq)
                    max_dd = min(max_dd, (eq / peak) - 1.0)
                    if eq <= equity * 0.1:
                        ruin = True
                        break
                rows.append({"candidate_id": cid, "family": cand.get("family"), "starting_equity": equity, "risk_pct": risk_pct, "ending_equity": eq, "max_drawdown_pct": max_dd * 100.0, "ruin_flag": ruin, "portfolio_overlay_status": "diagnostic_not_live_recommendation"})
    write_csv(ctx.run_root / "portfolio/aggressive_10x_portfolio_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/aggressive_10x_portfolio_report.md", "# Aggressive 10x Portfolio Overlay\n\nRuns only for candidates passing prior gates. Aggressive sizing cannot create alpha and does not authorize live trading.\n")


def stage_hypotheses(ctx: RunContext) -> None:
    path = read_csv_safe(ctx.run_root / "path_screen/path_opportunity_summary.csv")
    rows = []
    if not path.empty:
        for _, r in path.sort_values("24h_median_mfe_bps", ascending=False).head(6).iterrows():
            rows.append({"hypothesis_id": f"hyp_{len(rows)+1}", "source_family": r.get("family"), "observation": f"High median 24h MFE in tier {r.get('liquidity_tier')} side {r.get('side')}", "next_step": "only formulate a narrow contract if matched-null and data-quality evidence support it"})
    write_csv(ctx.run_root / "hypotheses/hypothesis_candidates.csv", rows)
    write_text(ctx.run_root / "hypotheses/new_hypothesis_synthesis.md", "# New Hypothesis Synthesis\n\nGenerated hypotheses are exploratory and limited to at most six observations from path/regime evidence. They are not strategy approvals.\n")


def stage_next_contracts(ctx: RunContext) -> None:
    pre = read_csv_safe(ctx.run_root / "preleads/full_coverage_prelead_summary.csv")
    matched = read_csv_safe(ctx.run_root / "matched_null/prelead_matched_null_summary.csv")
    valid = read_csv_safe(ctx.run_root / "validation/walk_forward_summary.csv")
    stress = read_csv_safe(ctx.run_root / "stress/execution_cost_liquidation_stress_summary.csv")
    out_dir = ctx.run_root / "next_contracts/contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    created = 0
    if not pre.empty:
        for cand in pre.sort_values("robustness_score", ascending=False).to_dict("records"):
            if created >= 3:
                break
            cid = cand.get("candidate_id")
            m = matched[matched.get("candidate_id", pd.Series(dtype=str)).eq(cid)] if not matched.empty else pd.DataFrame()
            v = valid[valid.get("candidate_id", pd.Series(dtype=str)).eq(cid)] if not valid.empty else pd.DataFrame()
            sbase = stress[(stress.get("candidate_id", pd.Series(dtype=str)).eq(cid)) & (stress.get("scenario", pd.Series(dtype=str)).eq("cost_x1p25"))] if not stress.empty else pd.DataFrame()
            pass_null = not m.empty and bool(m.iloc[0].get("beats_fresh_matched_null", False))
            eff_null = float(m.iloc[0].get("effective_nulls_per_event", 0)) if not m.empty else 0.0
            pass_val = not v.empty and bool(v.iloc[0].get("passes_55pct_positive_paths", False))
            pass_stress = not sbase.empty and float(sbase.iloc[0].get("net_R", -1)) > 0
            if not (pass_null and pass_val and pass_stress):
                continue
            proxy_evidence = float(cand.get("proxy_mark_or_liquidation_evidence_share", 1.0) or 1.0) > 0.0 or eff_null < 3
            label = "targeted_execution_data_collection_contract" if proxy_evidence else "family_specific_validation_contract"
            if cand.get("proxy_sector_only"):
                label = "sector_catalyst_data_build_contract"
            contract = {"candidate_id": cid, "family": cand.get("family"), "label": label, "created_at_utc": utc_now(), "source_run_root": str(ctx.run_root), "no_live_trading": True, "no_sealed_validation_without_new_contract": True, "reason": "passed marathon gates but evidence caps still apply"}
            p = out_dir / f"{cid}.json"
            p.write_text(json.dumps(contract, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            rows.append({"candidate_id": cid, "family": cand.get("family"), "contract_label": label, "path": str(p)})
            created += 1
    if not rows:
        rows.append({"candidate_id": "", "family": "", "contract_label": "reject_current_translation_no_contract", "path": ""})
    write_csv(ctx.run_root / "next_contracts/next_contract_summary.csv", rows)
    write_text(ctx.run_root / "next_contracts/next_contract_report.md", f"# Next Candidate Contracts\n\n- contracts created: `{created}`\n- maximum contracts allowed: `3`\n- no contract authorizes live trading or sealed validation.\n")


def stage_decision(ctx: RunContext) -> None:
    contracts = read_csv_safe(ctx.run_root / "next_contracts/next_contract_summary.csv")
    readiness = read_csv_safe(ctx.run_root / "readiness/family_readiness_matrix.csv")
    if contracts.empty:
        verdict = "blocked_by_protocol_issue"
    elif contracts["contract_label"].eq("family_specific_validation_contract").any():
        verdict = "promote_one_or_more_to_family_specific_validation"
    elif contracts["contract_label"].eq("targeted_execution_data_collection_contract").any():
        verdict = "promote_to_targeted_execution_data_collection"
    elif contracts["contract_label"].eq("sector_catalyst_data_build_contract").any() or (not readiness.empty and readiness.get("proxy_sector_only", pd.Series(dtype=bool)).fillna(False).any()):
        verdict = "build_sector_catalyst_data_next"
    elif contracts["contract_label"].eq("reject_current_translation_no_contract").all():
        verdict = "continue_alpha_discovery_needed"
    else:
        verdict = "continue_alpha_discovery_needed"
    if verdict not in ALLOWED_VERDICTS:
        verdict = "blocked_by_protocol_issue"
    write_json(ctx.run_root / "decision_summary.json", {"verdict": verdict, "final_holdout_untouched": True, "protected_start": str(FINAL_HOLDOUT_START), "run_root": str(ctx.run_root), "created_at_utc": utc_now()})
    pre = read_csv_safe(ctx.run_root / "preleads/full_coverage_prelead_summary.csv")
    matched = read_csv_safe(ctx.run_root / "matched_null/prelead_matched_null_summary.csv")
    write_text(ctx.run_root / "QLMG_ALPHA_DISCOVERY_MARATHON_REPORT.md", f"# QLMG Alpha Discovery Marathon Report\n\n## Verdict\n`{verdict}`\n\n## Governance\n- final holdout untouched: `true`\n- protected start: `{FINAL_HOLDOUT_START}`\n- allowed data end: `{SCREENING_END}`\n- no live trading, sealed validation, or live preparation is authorized.\n\n## Counts\n- preleads full-coverage rerun: `{len(pre)}`\n- matched-null rows: `{len(matched)}`\n\n## Evidence Caps\n- B1/C2 sector/catalyst evidence is proxy-only unless PIT data exists.\n- H1 is data-request-only unless adequate 1m exists.\n- Proxy mark/liquidation, proxy lifecycle, proxy cost, or missing top-of-book/depth caps any positive result at targeted execution data collection.\n\n## Key Artifacts\n- budget manifest: `budget/family_budget_manifest.csv`\n- scoring contract: `scoring/scoring_contract.json`\n- readiness matrix: `readiness/family_readiness_matrix.csv`\n- family multiple-testing: `validation/family_multiple_testing_report.md`\n- next contracts: `next_contracts/next_contract_summary.csv`\n")


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_ALPHA_DISCOVERY_MARATHON_REPORT.md", "decision_summary.json", "budget/family_budget_manifest.csv", "budget/output_size_estimate.md", "preflight/preflight_report.md", "preflight/prior_result_audit.md", "seal/seal_guard_report.md", "scoring/scoring_rule_report.md", "scoring/scoring_contract.json", "readiness/family_readiness_report.md", "readiness/family_readiness_matrix.csv", "regime/regime_stack_qc_report.md", "regime/regime_feature_coverage_summary.csv", "contracts/family_contract_summary.csv", "events/discovery_event_summary.csv", "path_screen/path_opportunity_report.md", "path_screen/path_opportunity_summary.csv", "cubes/regime_conditional_cube_report.md", "cubes/regime_conditional_cube_summary.csv", "sweep/sweep_report.md", "sweep/sweep_summary.csv", "preleads/full_coverage_prelead_report.md", "preleads/full_coverage_prelead_summary.csv", "matched_null/prelead_matched_null_report.md", "matched_null/prelead_matched_null_summary.csv", "one_minute/prelead_1m_overlay_report.md", "one_minute/prelead_1m_overlay_summary.csv", "stress/execution_cost_liquidation_stress_report.md", "stress/execution_cost_liquidation_stress_summary.csv", "validation/validation_report.md", "validation/walk_forward_summary.csv", "validation/family_multiple_testing_report.md", "portfolio/aggressive_10x_portfolio_report.md", "portfolio/aggressive_10x_portfolio_summary.csv", "hypotheses/new_hypothesis_synthesis.md", "hypotheses/hypothesis_candidates.csv", "next_contracts/next_contract_report.md", "next_contracts/next_contract_summary.csv", "notifications/telegram_readiness_report.md", "tmux/watch_commands.md",
    ]
    rows = []
    for rel in include:
        src = ctx.run_root / rel
        rows.append({"relative_path": rel, "exists": src.exists(), "size_bytes": src.stat().st_size if src.exists() else 0})
        if src.exists() and src.stat().st_size <= 5_000_000:
            shutil.copy2(src, bundle / rel.replace("/", "__"))
    for src in (ctx.run_root / "next_contracts/contracts").glob("*.json") if (ctx.run_root / "next_contracts/contracts").exists() else []:
        shutil.copy2(src, bundle / f"next_contracts__{src.name}")
        rows.append({"relative_path": str(src.relative_to(ctx.run_root)), "exists": True, "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_json(bundle / "artifact_path_index.json", {"artifacts": rows})
    zip_path = ctx.run_root / "qlmg_alpha_discovery_marathon_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(bundle))


STAGE_FUNCS = {
    "preflight-and-prior-result-audit": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "scoring-and-promotion-rule-fix": stage_scoring,
    "data-availability-and-family-readiness": stage_readiness,
    "regime-stack-refresh": stage_regime,
    "family-generator-contracts": stage_contracts,
    "discovery-event-generation": stage_events,
    "path-first-opportunity-screen": stage_path_screen,
    "regime-conditional-cube": stage_cube,
    "bounded-sobol-entry-exit-search": stage_sweep,
    "full-coverage-rerun-for-preleads": stage_full_coverage,
    "matched-null-validation-for-preleads": stage_matched_null,
    "targeted-1m-overlay-for-preleads": stage_one_minute,
    "execution-cost-liquidation-stress": stage_stress,
    "walk-forward-cpcv-and-overfit-controls": stage_validation,
    "aggressive-10x-portfolio-overlay": stage_portfolio,
    "new-hypothesis-synthesis": stage_hypotheses,
    "next-candidate-contracts": stage_next_contracts,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        print(f"[resume] skipping {stage}", flush=True)
        return
    append_command(ctx.run_root, stage)
    ensure_guard(ctx, stage, estimate_stage_gb(stage, ctx.args.smoke))
    ctx.notifier.send(f"QLMG alpha marathon stage start: {stage}", str(ctx.run_root))
    tmp = ctx.run_root / "tmp" / stage
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        STAGE_FUNCS[stage](ctx)
        mark_done(ctx.run_root, stage)
        shutil.rmtree(tmp, ignore_errors=True)
        ctx.notifier.send(f"QLMG alpha marathon stage complete: {stage}", str(ctx.run_root))
    except Exception as exc:
        ctx.notifier.send(f"QLMG alpha marathon stage failed: {stage}", f"{type(exc).__name__}: {exc}", level="error")
        raise


def main() -> int:
    args = parse_args()
    run_root, reason = resolve_run_root(args)
    start, end = clamp_window(args)
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": reason, "argv": sys.argv, "start": str(start), "end": str(end), "seed": args.seed, "smoke": args.smoke, "created_at_utc": utc_now(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True).stdout.strip()})
    notifier.send("QLMG alpha marathon run start", str(run_root))
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG alpha marathon run complete", str(run_root))
        return 0
    except Exception as exc:
        notifier.send("QLMG alpha marathon run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
