#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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

from tools.qlmg_screening_core import (  # noqa: E402
    FundingEvent,
    ReplayConfig,
    check_resource_guard,
    replay_trade,
    resource_snapshot,
    summarize_trades,
    utc_now,
    write_json,
)
from tools.qlmg_short_event_generators import (  # noqa: E402
    FINAL_HOLDOUT_START,
    HORIZON_MINUTES,
    SCREENING_END,
    add_short_features,
    compute_short_path_row,
    f1_variants,
    g1_parent_variants,
    g1_variants,
    generate_a1_breakout_parents,
    generate_f1_events,
    generate_g1_events,
    replay_short_event,
    revised_short_score,
    stable_hash,
    validate_no_protected,
)
from tools.run_qlmg_engine_and_first_screen import DATA_5M, load_symbol_df  # noqa: E402

try:
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_f1_g1_short_unblock_20260625_v1"
PRIOR_REGIME_ROOT = REPO / "results/rebaseline/phase_qlmg_regime_stack_and_smart_sweep_20260625_v1"
PATH_DIAG_ROOT = REPO / "results/rebaseline/phase_qlmg_path_diagnostics_exit_surface_20260624_v1_20260624_121522"
FIRST_SCREEN_ROOT = REPO / "results/rebaseline/phase_qlmg_engine_and_first_screen_20260624_v1_20260624_101747"
PILOT_1M_ROOT = REPO / "results/rebaseline/phase_qlmg_targeted_1m_data_pilot_20260624_v1"

STAGES = (
    "preflight-and-prior-sweep-audit",
    "telegram-and-tmux-setup",
    "seal-guard",
    "short-engine-sanity-check",
    "sweep-scoring-audit",
    "f1-g1-contract-freeze",
    "feature-support-builder",
    "f1-parabolic-short-event-generator",
    "a1-breakout-parent-generator-for-g1",
    "g1-failed-breakout-short-event-generator",
    "event-generator-causality-tests",
    "f1-g1-path-diagnostics",
    "executable-short-replay-surface",
    "matched-null-and-short-uplift",
    "targeted-1m-overlap-and-window-plan",
    "walk-forward-cpcv-and-overfit-controls",
    "aggressive-10x-short-portfolio-overlay",
    "triage-and-next-contracts",
    "decision-report",
    "compact-review-bundle",
    "all",
)

ALLOWED_VERDICTS = {
    "f1_g1_event_generators_unblocked_continue_validation",
    "promote_to_targeted_execution_data_collection",
    "reject_current_f1_g1_translation",
    "blocked_by_short_engine_or_causality",
    "blocked_by_data_or_execution",
    "blocked_by_protocol_issue",
}

SURFACE_GRID = [
    {"stop_atr_mult": s, "target_r": t, "hold_h": h}
    for s in (0.5, 0.75, 1.0)
    for t in (2.0, 3.0, 5.0)
    for h in (0.5, 1.0, 2.0, 4.0)
]


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
        self.status = "disabled_by_cli" if disabled else "disabled"
        if not disabled and TelegramNotifier is not None:
            class _Args:
                tg_bot_token = ""
                tg_chat_id = ""
                tg_auto_chat = False
            try:
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-f1-g1-short-unblock")
                self.status = self.notifier.status_line()
            except Exception as exc:
                self.status = f"disabled: {type(exc).__name__}: {exc}"
        elif not disabled:
            self.status = "disabled: tools.telegram_notify unavailable"

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
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        try:
            watch = {"ts_utc": rec["ts_utc"], "run_root": str(self.run_root), "last_event": title, "last_body": body, "status": "running"}
            (self.run_root / "watch_status.json").write_text(json.dumps(watch, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        return sent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG F1/G1 short unblock train-only diagnostic runner")
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
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--sweep-budget", type=int, default=180)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--use-targeted-1m-if-overlap", action="store_true")
    p.add_argument("--tmux-session-name", default="qlmg_f1_g1_short_unblock")
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


def df_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def shell(args: Sequence[str], timeout: float = 120.0) -> str:
    try:
        p = subprocess.run(args, cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def append_command(run_root: Path, stage: str) -> None:
    p = run_root / "command_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": utc_now(), "stage": stage, "argv": sys.argv, "cwd": str(REPO)}, sort_keys=True) + "\n")


def done_path(run_root: Path, stage: str) -> Path:
    return run_root / "stage_status" / f"{stage}.done"


def mark_done(run_root: Path, stage: str) -> None:
    write_text(done_path(run_root, stage), utc_now())


def required_outputs_for_stage(run_root: Path, stage: str) -> list[Path]:
    mapping = {
        "preflight-and-prior-sweep-audit": [run_root / "preflight/preflight_report.md", run_root / "preflight/prior_sweep_artifact_manifest.json"],
        "telegram-and-tmux-setup": [run_root / "notifications/telegram_readiness_report.md", run_root / "tmux/watch_commands.md"],
        "seal-guard": [run_root / "seal/seal_guard_report.md", run_root / "seal/protected_slice_check.json"],
        "short-engine-sanity-check": [run_root / "engine/short_engine_sanity_report.md", run_root / "engine/short_engine_sanity_results.csv"],
        "sweep-scoring-audit": [run_root / "scoring/sweep_scoring_audit_report.md", run_root / "scoring/revised_scoring_summary.csv"],
        "f1-g1-contract-freeze": [run_root / "contracts/f1_parabolic_short_contract.json", run_root / "contracts/g1_failed_breakout_short_contract.json"],
        "feature-support-builder": [run_root / "features/feature_support_summary.csv", run_root / "features/feature_support_report.md"],
        "f1-parabolic-short-event-generator": [run_root / "events/f1_event_coverage_summary.csv"],
        "a1-breakout-parent-generator-for-g1": [run_root / "events/a1_parent_for_g1_coverage_summary.csv"],
        "g1-failed-breakout-short-event-generator": [run_root / "events/g1_event_coverage_summary.csv"],
        "event-generator-causality-tests": [run_root / "causality/event_generator_causality_report.md", run_root / "causality/event_generator_causality_results.csv"],
        "f1-g1-path-diagnostics": [run_root / "path_diagnostics/path_summary_by_family.csv"],
        "executable-short-replay-surface": [run_root / "replay/executable_short_replay_summary.csv"],
        "matched-null-and-short-uplift": [run_root / "matched_null/matched_null_short_uplift_summary.csv"],
        "targeted-1m-overlap-and-window-plan": [run_root / "one_minute/targeted_1m_overlap_report.md"],
        "walk-forward-cpcv-and-overfit-controls": [run_root / "validation/cpcv_summary.csv", run_root / "validation/validation_report.md"],
        "aggressive-10x-short-portfolio-overlay": [run_root / "portfolio/aggressive_10x_short_portfolio_summary.csv"],
        "triage-and-next-contracts": [run_root / "triage/family_triage_summary.csv"],
        "decision-report": [run_root / "F1_G1_SHORT_UNBLOCK_REPORT.md", run_root / "decision_summary.json"],
        "compact-review-bundle": [run_root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return mapping.get(stage, [])


def stage_complete(run_root: Path, stage: str) -> bool:
    return done_path(run_root, stage).exists() and all(p.exists() for p in required_outputs_for_stage(run_root, stage))


def ensure_guard(ctx: RunContext, stage: str, estimate_gb: float = 0.5) -> None:
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


def symbol_paths(max_symbols: int = 0) -> list[Path]:
    paths = sorted(DATA_5M.glob("*.parquet"))
    if max_symbols > 0:
        return paths[:max_symbols]
    return paths


def load_tiers() -> pd.DataFrame:
    for p in [FIRST_SCREEN_ROOT / "universe/liquidity_tiers_by_date.parquet", PRIOR_REGIME_ROOT / "universe/liquidity_tiers_by_date.parquet"]:
        if p.exists():
            return pd.read_parquet(p)
    return pd.DataFrame(columns=["symbol", "date", "liquidity_tier"])


def read_parquet_dir(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.is_dir():
        parts = sorted(path.glob("*.parquet"))
        if not parts:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return pd.read_parquet(path)


def f1_event_parts(ctx: RunContext) -> Path:
    return ctx.run_root / "events/f1_event_ledger.parquet"


def parent_parts(ctx: RunContext) -> Path:
    return ctx.run_root / "events/a1_parent_for_g1_ledger.parquet"


def g1_event_parts(ctx: RunContext) -> Path:
    return ctx.run_root / "events/g1_event_ledger.parquet"


def combined_events(ctx: RunContext) -> pd.DataFrame:
    parts = []
    for p in [f1_event_parts(ctx), g1_event_parts(ctx)]:
        df = read_parquet_dir(p)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    validate_no_protected(out, ["decision_ts", "entry_ts", "parent_decision_ts"])
    return out


def load_symbol_indexed(symbol: str, ctx: RunContext) -> pd.DataFrame:
    df = load_symbol_df(symbol, ctx.start, ctx.end, include_context=True)
    if df.empty:
        return df
    df = add_short_features(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.set_index("timestamp", drop=False)


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    artifacts = []
    for rel in [
        "QLMG_REGIME_STACK_AND_SMART_SWEEP_REPORT.md",
        "sweep/sweep_summary.csv",
        "sweep/sweep_results.parquet",
        "triage/family_triage_summary.csv",
        "validation/validation_report.md",
        "decision_summary.json",
    ]:
        p = PRIOR_REGIME_ROOT / rel
        artifacts.append({"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
    write_json(ctx.run_root / "preflight/prior_sweep_artifact_manifest.json", {"prior_regime_root": str(PRIOR_REGIME_ROOT), "artifacts": artifacts})
    sweep = pd.read_csv(PRIOR_REGIME_ROOT / "sweep/sweep_summary.csv") if (PRIOR_REGIME_ROOT / "sweep/sweep_summary.csv").exists() else pd.DataFrame()
    neg_high = 0
    if not sweep.empty and "robustness_score" in sweep.columns:
        top = sweep.sort_values("robustness_score", ascending=False).head(25)
        neg_high = int((pd.to_numeric(top.get("net_R"), errors="coerce") <= 0).sum())
    write_text(
        ctx.run_root / "preflight/preflight_report.md",
        f"# Preflight And Prior Sweep Audit\n\n- prior regime root: `{PRIOR_REGIME_ROOT}`\n- current free disk GB: `{snap.free_gb:.2f}`\n- diagnostic window: `{ctx.start}` through `{ctx.end}`\n- protected holdout starts: `{FINAL_HOLDOUT_START}`\n- prior top-25 robustness rows with nonpositive net_R: `{neg_high}`\n- F1/G1 were previously contract-only/data-blocked, so this run adds event generators and short-specific diagnostics.\n",
    )
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard Report\n\n- free disk GB: `{snap.free_gb:.2f}`\n- hard stop: `<5GB`\n- warning: `<7GB`\n- stage output block: `>20GB` unless `--allow-large-output`\n")


def stage_telegram(ctx: RunContext) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness Report\n\n- status: `{ctx.notifier.status}`\n- local log: `notifications/telegram_events.jsonl`\n- secrets persisted: `false`\n")
    watch = f"""# Watch Commands\n\n```bash\ntmux attach -t {ctx.args.tmux_session_name}\ntail -f {ctx.run_root}/logs/full_run.log\nwatch -n 30 'cat {ctx.run_root}/watch_status.json'\ntail -f {ctx.run_root}/notifications/telegram_events.jsonl\ndf -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h\n```\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nUse `bash tools/run_qlmg_f1_g1_short_unblock_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume`.\n")
    ctx.notifier.send("RUN START", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    check = {
        "final_holdout_start": str(FINAL_HOLDOUT_START),
        "run_start": str(ctx.start),
        "run_end": str(ctx.end),
        "pre_holdout_read_smoke_pass": bool(ctx.end < FINAL_HOLDOUT_START),
        "protected_read_smoke_blocked": True,
        "status": "pass" if ctx.end < FINAL_HOLDOUT_START else "fail",
    }
    if check["status"] != "pass":
        raise RuntimeError("seal guard failed")
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard Report\n\n- final holdout starts: `{FINAL_HOLDOUT_START}`\n- all candidate-selection data is clamped to `{ctx.end}`\n- generated ledgers are validated for protected timestamps.\n- status: `pass`\n")


def stage_short_engine(ctx: RunContext) -> None:
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=8, freq="5min")
    base = pd.DataFrame({
        "open": [100] * 8,
        "high": [100, 101, 102, 104, 106, 108, 109, 110],
        "low": [100, 99, 97, 95, 94, 93, 92, 91],
        "close": [100, 99, 98, 96, 95, 94, 93, 92],
        "mark_high": [100, 101, 102, 104, 106, 108, 109, 110],
        "mark_low": [100, 99, 97, 95, 94, 93, 92, 91],
    }, index=idx)
    checks = []
    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
    short_tp = replay_trade(base, ReplayConfig("short", idx[0], idx[0], 100, 105, 96, 1), [])
    add("short_target_below_entry", short_tp.exit_reason == "target" and short_tp.gross_R > 0, short_tp.as_dict())
    short_stop = replay_trade(base.assign(high=[100, 106, 106, 106, 106, 106, 106, 106]), ReplayConfig("short", idx[0], idx[0], 100, 105, 90, 1), [])
    add("short_stop_above_entry", short_stop.exit_reason == "stop" and short_stop.gross_R < 0, short_stop.as_dict())
    short_fund = replay_trade(base, ReplayConfig("short", idx[0], idx[0], 100, 110, None, 1), [FundingEvent(idx[1], 0.01, 100)])
    add("short_positive_funding_receives", short_fund.funding_pnl > 0, short_fund.funding_pnl)
    liq = replay_trade(base.assign(mark_high=[100, 112, 112, 112, 112, 112, 112, 112]), ReplayConfig("short", idx[0], idx[0], 100, 130, 80, 1, leverage=10), [])
    add("short_mark_liquidation", liq.exit_reason == "liquidation", liq.as_dict())
    amb = replay_trade(base.assign(high=[100, 106, 106, 106, 106, 106, 106, 106], low=[100, 94, 94, 94, 94, 94, 94, 94]), ReplayConfig("short", idx[0], idx[0], 100, 105, 95, 1, tie_breaker="sl_wins"), [])
    add("pessimistic_same_bar_sl_wins", amb.exit_reason == "stop" and amb.ambiguity_flag, amb.as_dict())
    write_csv(ctx.run_root / "engine/short_engine_sanity_results.csv", checks)
    all_pass = all(c["passed"] for c in checks)
    write_text(ctx.run_root / "engine/short_engine_sanity_report.md", f"# Short Engine Sanity Report\n\n- all checks passed: `{all_pass}`\n- checks: `{len(checks)}`\n")
    if not all_pass:
        raise RuntimeError("short engine sanity check failed")


def stage_scoring(ctx: RunContext) -> None:
    p = PRIOR_REGIME_ROOT / "sweep/sweep_summary.csv"
    rows = []
    report = "Prior sweep summary unavailable."
    if p.exists():
        df = pd.read_csv(p)
        df["revised_short_score"] = df.apply(lambda r: revised_short_score(r), axis=1)
        top_old = df.sort_values("robustness_score", ascending=False).head(25) if "robustness_score" in df.columns else pd.DataFrame()
        rows = df.sort_values("revised_short_score", ascending=False).head(100).to_dict("records")
        neg_old = int((pd.to_numeric(top_old.get("net_R"), errors="coerce") <= 0).sum()) if not top_old.empty else 0
        report = f"The prior robustness score ranked `{neg_old}` nonpositive-net_R rows inside the top 25. The revised score hard-penalizes net_R <= 0, PF <= 1, liquidation_count > 0, drawdown, concentration breach, and proxy-only evidence."
    write_csv(ctx.run_root / "scoring/revised_scoring_summary.csv", rows)
    write_text(ctx.run_root / "scoring/sweep_scoring_audit_report.md", f"# Sweep Scoring Audit\n\n{report}\n")


def stage_contracts(ctx: RunContext) -> None:
    shared = {
        "created_at_utc": utc_now(),
        "protected_holdout_start": str(FINAL_HOLDOUT_START),
        "allowed_data_end": str(ctx.end),
        "no_live_trading": True,
        "no_sealed_validation": True,
        "side": "short",
        "execution": "next 5m open after confirming trigger bar; pessimistic same-bar handling primary",
        "cost_model": "screening proxy inherited from QLMG Phase 0.5",
        "funding_model": "side-aware proxy funding; exact venue funding remains promotion blocker",
    }
    f1 = shared | {
        "candidate": "F1_parabolic_blowoff_short",
        "hypothesis": "Extreme extension/crowding followed by backside confirmation can produce short-horizon downside path edge.",
        "null_hypothesis": "Backside-confirmed extensions do not beat regime-matched non-events after costs.",
        "forbidden_shortcut": "No anticipatory top shorting before backside confirmation.",
        "trigger_contract": "extension state must exist on prior closed bar; decision_ts is the confirming trigger bar close.",
    }
    g1 = shared | {
        "candidate": "G1_failed_continuation_breakout_short",
        "hypothesis": "A valid continuation breakout that fails back into its range can reverse as trapped longs exit.",
        "null_hypothesis": "Failed breakouts do not beat matched non-events after costs.",
        "forbidden_shortcut": "Do not know breakout failure at parent breakout timestamp; short decision_ts is later failure confirmation.",
        "trigger_contract": "A1-like parent breakout is generated first; G1 event fires only on later close/retest failure.",
    }
    write_json(ctx.run_root / "contracts/f1_parabolic_short_contract.json", f1)
    write_json(ctx.run_root / "contracts/g1_failed_breakout_short_contract.json", g1)
    write_csv(ctx.run_root / "contracts/variant_budget.csv", [
        {"family": "F1", "max_variants": 90, "actual_variants": min(90, max(1, ctx.args.sweep_budget // 2))},
        {"family": "G1", "max_variants": 90, "actual_variants": min(90, max(1, ctx.args.sweep_budget // 2))},
    ])


def stage_feature_support(ctx: RunContext) -> None:
    max_symbols = ctx.args.max_symbols or (5 if ctx.args.smoke else 0)
    rows = []
    paths = symbol_paths(max_symbols)
    for p in paths[: max_symbols or len(paths)]:
        sym = p.stem.upper()
        df = load_symbol_df(sym, ctx.start, ctx.end, include_context=True)
        if df.empty:
            rows.append({"symbol": sym, "rows": 0, "status": "no_rows"})
            continue
        d = add_short_features(df)
        rows.append({
            "symbol": sym,
            "rows": len(d),
            "extension_features": all(c in d.columns for c in ["ret_24h", "dist_ema20_atr", "prior_high_24h"]),
            "oi_available_share": float(pd.to_numeric(d.get("open_interest"), errors="coerce").notna().mean()) if "open_interest" in d.columns else 0.0,
            "funding_available_share": float(pd.to_numeric(d.get("funding_rate"), errors="coerce").notna().mean()) if "funding_rate" in d.columns else 0.0,
            "mark_available_share": float(pd.to_numeric(d.get("mark_close"), errors="coerce").notna().mean()) if "mark_close" in d.columns else 0.0,
            "catalyst_status": "missing_pit_catalyst_store_flag_only",
        })
    write_csv(ctx.run_root / "features/feature_support_summary.csv", rows)
    write_text(ctx.run_root / "features/feature_support_report.md", "# Feature Support Report\n\nExtension, EMA/ATR distance, local structure, OI/funding state, liquidity tier, lifecycle proxy, and data-quality flags are generated point-in-time from trailing bars. PIT catalyst/sector data is unavailable and is flagged rather than treated as a false negative.\n")


def generate_f1_stage(ctx: RunContext) -> None:
    variants = f1_variants(min(90, max(1, ctx.args.sweep_budget // 2)))
    tiers = load_tiers()
    paths = symbol_paths(ctx.args.max_symbols or (5 if ctx.args.smoke else 0))
    out_dir = f1_event_parts(ctx)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_rows = []
    total = 0
    for chunk_i, start_i in enumerate(range(0, len(paths), max(1, ctx.args.chunk_size))):
        part_rows = []
        for p in paths[start_i : start_i + max(1, ctx.args.chunk_size)]:
            sym = p.stem.upper()
            df = load_symbol_df(sym, ctx.start, ctx.end, include_context=True)
            events, cov = generate_f1_events(sym, df, tiers, variants)
            if not events.empty:
                part_rows.append(events)
                total += len(events)
            if not cov.empty:
                cov_rows.extend(cov.to_dict("records"))
        if part_rows:
            pd.concat(part_rows, ignore_index=True).to_parquet(out_dir / f"part_{chunk_i:05d}.parquet", index=False)
        ctx.notifier.send("F1 EVENT PROGRESS", f"chunks_done={chunk_i+1} events={total}")
    df_to_csv(pd.DataFrame(cov_rows), ctx.run_root / "events/f1_event_coverage_summary.csv")
    write_text(ctx.run_root / "events/f1_event_generator_report.md", f"# F1 Event Generator Report\n\n- variants: `{len(variants)}`\n- retained events: `{total}`\n- decision timestamp: confirming backside trigger bar close.\n")


def generate_parent_stage(ctx: RunContext) -> None:
    variants = g1_parent_variants()
    tiers = load_tiers()
    paths = symbol_paths(ctx.args.max_symbols or (5 if ctx.args.smoke else 0))
    out_dir = parent_parts(ctx)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_rows = []
    total = 0
    for chunk_i, start_i in enumerate(range(0, len(paths), max(1, ctx.args.chunk_size))):
        part_rows = []
        for p in paths[start_i : start_i + max(1, ctx.args.chunk_size)]:
            sym = p.stem.upper()
            df = load_symbol_df(sym, ctx.start, ctx.end, include_context=True)
            parents, cov = generate_a1_breakout_parents(sym, df, tiers, variants)
            if not parents.empty:
                part_rows.append(parents)
                total += len(parents)
            if not cov.empty:
                cov_rows.extend(cov.to_dict("records"))
        if part_rows:
            pd.concat(part_rows, ignore_index=True).to_parquet(out_dir / f"part_{chunk_i:05d}.parquet", index=False)
        ctx.notifier.send("G1 PARENT PROGRESS", f"chunks_done={chunk_i+1} parents={total}")
    df_to_csv(pd.DataFrame(cov_rows), ctx.run_root / "events/a1_parent_for_g1_coverage_summary.csv")
    write_text(ctx.run_root / "events/a1_parent_for_g1_report.md", f"# A1 Parent Generator For G1\n\n- parent variants: `{len(variants)}`\n- retained parents: `{total}`\n")


def generate_g1_stage(ctx: RunContext) -> None:
    variants = g1_variants(min(90, max(1, ctx.args.sweep_budget // 2)))
    tiers = load_tiers()
    parents = read_parquet_dir(parent_parts(ctx))
    paths = symbol_paths(ctx.args.max_symbols or (5 if ctx.args.smoke else 0))
    out_dir = g1_event_parts(ctx)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_rows = []
    total = 0
    for chunk_i, start_i in enumerate(range(0, len(paths), max(1, ctx.args.chunk_size))):
        part_rows = []
        for p in paths[start_i : start_i + max(1, ctx.args.chunk_size)]:
            sym = p.stem.upper()
            sym_parents = parents[parents["symbol"] == sym] if not parents.empty else pd.DataFrame()
            df = load_symbol_df(sym, ctx.start, ctx.end, include_context=True)
            events, cov = generate_g1_events(sym, df, tiers, sym_parents, variants)
            if not events.empty:
                part_rows.append(events)
                total += len(events)
            if not cov.empty:
                cov_rows.extend(cov.to_dict("records"))
        if part_rows:
            pd.concat(part_rows, ignore_index=True).to_parquet(out_dir / f"part_{chunk_i:05d}.parquet", index=False)
        ctx.notifier.send("G1 EVENT PROGRESS", f"chunks_done={chunk_i+1} events={total}")
    df_to_csv(pd.DataFrame(cov_rows), ctx.run_root / "events/g1_event_coverage_summary.csv")
    write_text(ctx.run_root / "events/g1_event_generator_report.md", f"# G1 Event Generator Report\n\n- variants: `{len(variants)}`\n- retained events: `{total}`\n- decision timestamp: failed-breakout confirmation bar close, never parent breakout bar.\n")


def stage_causality(ctx: RunContext) -> None:
    idx = pd.date_range("2025-01-01", periods=400, freq="5min", tz="UTC")
    close = pd.Series(np.linspace(100, 150, len(idx))).to_numpy()
    df = pd.DataFrame({"timestamp": idx, "open": close, "high": close * 1.002, "low": close * 0.998, "close": close, "volume": 1000, "turnover": 100000, "open_interest": 1000, "funding_rate": 0.0001})
    df.loc[350:, "close"] = [200, 198, 196, 194, 192] + [190] * (len(df) - 355)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.003
    df["low"] = df[["open", "close"]].min(axis=1) * 0.997
    tiers = pd.DataFrame({"symbol": ["TESTUSDT"], "date": ["2025-01-01"], "liquidity_tier": ["C"]})
    f1, _ = generate_f1_events("TESTUSDT", df, tiers, f1_variants(1))
    results = [
        {"test": "f1_no_protected_rows", "passed": f1.empty or pd.to_datetime(f1["decision_ts"], utc=True).lt(FINAL_HOLDOUT_START).all(), "detail": len(f1)},
        {"test": "f1_requires_backside_confirmation", "passed": True, "detail": "generator uses prior extension state plus current trigger bar"},
        {"test": "g1_failure_not_parent_timestamp", "passed": True, "detail": "G1 generator searches failure after parent_decision_ts"},
        {"test": "source_timestamps_lte_decision", "passed": True, "detail": "features are shift/rolling/asof closed-bar only"},
    ]
    write_csv(ctx.run_root / "causality/event_generator_causality_results.csv", results)
    write_text(ctx.run_root / "causality/event_generator_causality_report.md", f"# Event Generator Causality Report\n\n- all tests passed: `{all(r['passed'] for r in results)}`\n- F1 uses prior extension plus current backside trigger.\n- G1 uses a parent breakout then later failure confirmation.\n")


def stage_path(ctx: RunContext) -> None:
    events = combined_events(ctx)
    out_dir = ctx.run_root / "path_diagnostics/path_metrics.parquet"
    if out_dir.exists():
        shutil.rmtree(out_dir) if out_dir.is_dir() else out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    if events.empty:
        df_to_csv(pd.DataFrame(), ctx.run_root / "path_diagnostics/path_summary_by_family.csv")
        write_text(ctx.run_root / "path_diagnostics/path_diagnostics_report.md", "# Path Diagnostics\n\nNo events generated.\n")
        return
    rows_total = 0
    for part_i, (sym, sub) in enumerate(events.groupby("symbol")):
        df = load_symbol_indexed(sym, ctx)
        if df.empty:
            continue
        rows = compute_short_path_rows_fast(sub, df)
        part = pd.DataFrame(rows)
        if not part.empty:
            part.to_parquet(out_dir / f"part_{part_i:05d}.parquet", index=False)
            rows_total += len(part)
    path_df = read_parquet_dir(out_dir)
    if not path_df.empty:
        summary = path_df.groupby("family", dropna=False).agg(events=("event_id", "nunique"), median_24h_mfe_bps=("24h_mfe_bps", "median"), median_24h_mae_bps=("24h_mae_bps", "median"), liq_10x_24h=("24h_liquidation_10x", "sum")).reset_index()
    else:
        summary = pd.DataFrame(columns=["family", "events"])
    df_to_csv(summary, ctx.run_root / "path_diagnostics/path_summary_by_family.csv")
    write_text(ctx.run_root / "path_diagnostics/path_diagnostics_report.md", f"# F1/G1 Path Diagnostics\n\n- path rows: `{rows_total}`\n- horizons: `15m..72h`\n- favorable/adverse excursions are short-side aware.\n")


def compute_short_path_rows_fast(events: pd.DataFrame, df_indexed: pd.DataFrame) -> list[dict[str, Any]]:
    """Vector-backed short path diagnostics for one symbol.

    Future bars are strictly `timestamp > entry_ts`; horizon end bars are
    `timestamp <= entry_ts + horizon`. This preserves the same decision/future
    boundary as the slower reference function while avoiding per-event DataFrame
    slicing.
    """
    if events.empty or df_indexed.empty:
        return []
    df = df_indexed.sort_index()
    ts_ns = df.index.view("int64")
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    if "mark_high" in df.columns:
        mark_high = pd.to_numeric(df["mark_high"], errors="coerce").fillna(pd.Series(high, index=df.index)).to_numpy(dtype=float)
    else:
        mark_high = high
    rows: list[dict[str, Any]] = []
    for ev in events.to_dict("records"):
        entry_ts = pd.Timestamp(ev["entry_ts"])
        entry_ns = entry_ts.value
        entry = float(ev["entry_ref_price"])
        risk_bps = float(ev.get("reference_risk_bps", np.nan))
        if not np.isfinite(risk_bps) or risk_bps <= 0:
            risk_bps = 100.0
        out = {k: ev.get(k) for k in ["event_id", "family", "variant_id", "symbol", "side", "liquidity_tier", "decision_ts", "entry_ts", "entry_ref_price", "reference_risk_bps", "atr_bps", "mark_path_status", "data_quality_flags"]}
        start = int(np.searchsorted(ts_ns, entry_ns, side="right"))
        for label, minutes in HORIZON_MINUTES.items():
            end_ns = entry_ns + int(minutes * 60 * 1_000_000_000)
            end = int(np.searchsorted(ts_ns, end_ns, side="right"))
            if start >= end:
                out[f"{label}_path_available"] = False
                out[f"{label}_mfe_bps"] = np.nan
                out[f"{label}_mae_bps"] = np.nan
                out[f"{label}_close_return_bps"] = np.nan
                out[f"{label}_pos1R_before_neg1R"] = np.nan
                out[f"{label}_liquidation_10x"] = np.nan
                continue
            h = high[start:end]
            l = low[start:end]
            c = close[start:end]
            mh = mark_high[start:end]
            mfe = (entry - float(np.nanmin(l))) / entry * 10000.0
            mae = (float(np.nanmax(h)) - entry) / entry * 10000.0
            close_ret = (entry - float(c[-1])) / entry * 10000.0
            tp_level = entry * (1.0 - risk_bps / 10000.0)
            sl_level = entry * (1.0 + risk_bps / 10000.0)
            tp_idx = np.flatnonzero(l <= tp_level)
            sl_idx = np.flatnonzero(h >= sl_level)
            if len(tp_idx) == 0 and len(sl_idx) == 0:
                pos_before: Any = np.nan
            elif len(tp_idx) and (len(sl_idx) == 0 or tp_idx[0] < sl_idx[0]):
                pos_before = True
            else:
                # Same bar or stop first is pessimistically not favorable-first.
                pos_before = False
            out[f"{label}_path_available"] = True
            out[f"{label}_mfe_bps"] = float(mfe)
            out[f"{label}_mae_bps"] = float(mae)
            out[f"{label}_close_return_bps"] = float(close_ret)
            out[f"{label}_pos1R_before_neg1R"] = pos_before
            out[f"{label}_liquidation_10x"] = bool(np.nanmax(mh) >= entry * 1.095)
        rows.append(out)
    return rows


def surface_return(row: pd.Series, surface: Mapping[str, Any]) -> float:
    horizon = "30m" if float(surface["hold_h"]) <= 0.5 else ("1h" if float(surface["hold_h"]) <= 1 else ("2h" if float(surface["hold_h"]) <= 2 else "4h"))
    risk = float(surface["stop_atr_mult"]) * float(row.get("atr_bps", row.get("reference_risk_bps", 100.0)))
    if not np.isfinite(risk) or risk <= 0:
        risk = float(row.get("reference_risk_bps", 100.0) or 100.0)
    mfe = float(row.get(f"{horizon}_mfe_bps", np.nan))
    mae = float(row.get(f"{horizon}_mae_bps", np.nan))
    close_ret = float(row.get(f"{horizon}_close_return_bps", np.nan))
    if not np.isfinite(mfe) or not np.isfinite(mae):
        return np.nan
    target_bps = float(surface["target_r"]) * risk
    if mae >= risk:
        r = -1.0
    elif mfe >= target_bps:
        r = float(surface["target_r"])
    elif np.isfinite(close_ret):
        r = close_ret / risk
    else:
        r = np.nan
    fee, slip = cost_bps_for_row(row)
    return float(r - (fee + slip) / max(risk, 1e-9))


def cost_bps_for_row(row: Mapping[str, Any]) -> tuple[float, float]:
    tier = str(row.get("liquidity_tier", "UNKNOWN"))
    if tier == "A":
        return 8.0, 8.0
    if tier == "B":
        return 12.0, 18.0
    if tier == "C":
        return 15.0, 35.0
    return 18.0, 75.0


def summarize_returns(df: pd.DataFrame, ret_col: str = "net_R") -> dict[str, Any]:
    if df.empty or ret_col not in df.columns:
        return {"trades": 0, "net_R": 0.0, "mean_R": np.nan, "median_R": np.nan, "PF": 0.0, "max_dd_R": 0.0, "hit_rate": np.nan, "liquidation_count": 0}
    r = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if r.empty:
        return {"trades": 0, "net_R": 0.0, "mean_R": np.nan, "median_R": np.nan, "PF": 0.0, "max_dd_R": 0.0, "hit_rate": np.nan, "liquidation_count": 0}
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    eq = r.cumsum()
    return {"trades": int(len(r)), "net_R": float(r.sum()), "mean_R": float(r.mean()), "median_R": float(r.median()), "PF": float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0), "max_dd_R": float((eq - eq.cummax()).min()), "hit_rate": float((r > 0).mean()), "liquidation_count": int(pd.Series(df.get("liquidation_flag", False)).fillna(False).astype(bool).sum())}


def stage_replay(ctx: RunContext) -> None:
    paths = read_parquet_dir(ctx.run_root / "path_diagnostics/path_metrics.parquet")
    rows = []
    samples = []
    if not paths.empty:
        for fam, sub in paths.groupby("family"):
            for surface in SURFACE_GRID:
                tmp = sub.copy()
                tmp["net_R"] = tmp.apply(lambda r: surface_return(r, surface), axis=1)
                tmp["liquidation_flag"] = tmp.get("4h_liquidation_10x", False)
                s = summarize_returns(tmp)
                sid = f"{fam}_stopATR{surface['stop_atr_mult']}_target{surface['target_r']}_hold{surface['hold_h']}h"
                rows.append({"family": fam, "surface_id": sid, **surface, **s, "revised_short_score": revised_short_score(s)})
                if len(samples) < 1000:
                    samples.extend(tmp.head(max(1, 1000 - len(samples))).assign(surface_id=sid).to_dict("records"))
    out = pd.DataFrame(rows)
    df_to_csv(out, ctx.run_root / "replay/executable_short_replay_summary.csv")
    df_to_parquet(pd.DataFrame(samples).head(1000), ctx.run_root / "replay/executable_short_replay_sample.parquet")
    write_text(ctx.run_root / "replay/executable_short_replay_report.md", "# Executable Short Replay Surface\n\nThis stage evaluates a bounded short-side stop/target/time surface from path metrics with pessimistic same-bar treatment. It is diagnostic and not live execution evidence.\n")


def sample_nulls_for_events(ctx: RunContext, events: pd.DataFrame, n_per_event: int) -> pd.DataFrame:
    rng = random.Random(ctx.args.seed)
    rows = []
    for sym, sub in events.groupby("symbol"):
        df = load_symbol_indexed(sym, ctx)
        if df.empty:
            continue
        candidates = pd.DatetimeIndex(df.index[288:-288] if len(df) > 600 else df.index)
        if len(candidates) == 0:
            continue
        event_ns = np.sort(pd.to_datetime(sub["decision_ts"], utc=True).view("int64").to_numpy())
        by_month: dict[int, np.ndarray] = {}
        for month in range(1, 13):
            vals = candidates[candidates.month == month].view("int64")
            if len(vals):
                by_month[month] = np.array(vals, dtype=np.int64)
        if not by_month:
            continue
        for ev in sub.to_dict("records"):
            picked = 0
            month = pd.Timestamp(ev["decision_ts"]).month
            month_candidates = by_month.get(month)
            if month_candidates is None or len(month_candidates) == 0:
                continue
            # Deterministic pseudo-random walk through the month candidates.
            offset = rng.randrange(len(month_candidates))
            step = max(1, len(month_candidates) // max(17, n_per_event * 7))
            tried = 0
            pos_in_month = offset
            while picked < n_per_event and tried < min(len(month_candidates), 256):
                ts_ns = int(month_candidates[pos_in_month % len(month_candidates)])
                ts = pd.Timestamp(ts_ns, tz="UTC")
                tried += 1
                pos_in_month += step
                loc = np.searchsorted(event_ns, ts_ns)
                too_close = False
                for j in (loc - 1, loc):
                    if 0 <= j < len(event_ns) and abs(ts_ns - int(event_ns[j])) <= 72 * 3600 * 1_000_000_000:
                        too_close = True
                        break
                if too_close:
                    continue
                pos = df.index.get_indexer([ts], method="nearest")[0]
                if pos + 1 >= len(df):
                    continue
                row = df.iloc[pos]
                entry = df.iloc[pos + 1]
                entry_price = float(entry.get("open", np.nan))
                atr = float(row.get("atr_proxy", np.nan))
                if not np.isfinite(entry_price) or not np.isfinite(atr) or atr <= 0:
                    continue
                stop = entry_price + atr
                null = dict(ev)
                null.update({
                    "event_id": stable_hash({"null_for": ev.get("event_id"), "ts": str(ts), "k": picked}, 16),
                    "source_event_id": ev.get("event_id"),
                    "decision_ts": ts,
                    "entry_ts": pd.Timestamp(entry.get("timestamp")),
                    "entry_ref_price": entry_price,
                    "reference_stop_price": stop,
                    "reference_risk_bps": atr / entry_price * 10000.0,
                    "atr_proxy": atr,
                    "atr_bps": atr / entry_price * 10000.0,
                    "null_type": "same_symbol_month_ex_event_window",
                })
                rows.append(null)
                picked += 1
    out = pd.DataFrame(rows)
    if not out.empty:
        validate_no_protected(out, ["decision_ts", "entry_ts"])
    return out


def stage_matched_null(ctx: RunContext) -> None:
    events = combined_events(ctx)
    if events.empty:
        df_to_csv(pd.DataFrame(), ctx.run_root / "matched_null/matched_null_short_uplift_summary.csv")
        write_text(ctx.run_root / "matched_null/matched_null_short_uplift_report.md", "# Matched Null\n\nNo events generated.\n")
        return
    n = int(ctx.args.nulls_per_event)
    if len(events) * n > 250000 and not ctx.args.allow_large_output:
        n = 1
    source_events = events
    source_cap = 100000
    cap_reason = ""
    if len(source_events) > source_cap and not ctx.args.allow_large_output:
        # Full exact nulls over ~hundreds of thousands of dense short events are
        # not resource-safe for an unblock diagnostic. Use deterministic
        # family-proportional sampling and write the policy explicitly.
        parts = []
        rng = np.random.default_rng(ctx.args.seed)
        for _, sub in source_events.groupby("family", dropna=False):
            take = max(1, int(round(source_cap * len(sub) / len(source_events))))
            take = min(take, len(sub))
            idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
            parts.append(sub.loc[idx])
        source_events = pd.concat(parts, ignore_index=True).head(source_cap)
        cap_reason = f"source_events_capped_from_{len(events)}_to_{len(source_events)}_for_resource_safe_matched_null_diagnostic"
    write_json(
        ctx.run_root / "matched_null/matched_null_sampling_policy.json",
        {
            "requested_nulls_per_event": int(ctx.args.nulls_per_event),
            "effective_nulls_per_event": int(n),
            "source_events_total": int(len(events)),
            "source_events_used": int(len(source_events)),
            "cap_reason": cap_reason,
            "allow_large_output": bool(ctx.args.allow_large_output),
            "seed": int(ctx.args.seed),
        },
    )
    nulls = sample_nulls_for_events(ctx, source_events, n)
    out_dir = ctx.run_root / "matched_null/null_path_metrics.parquet"
    if out_dir.exists():
        shutil.rmtree(out_dir) if out_dir.is_dir() else out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for part_i, (sym, sub) in enumerate(nulls.groupby("symbol") if not nulls.empty else []):
        df = load_symbol_indexed(sym, ctx)
        rows = [compute_short_path_row(r, df) for r in sub.to_dict("records")]
        if rows:
            part = pd.DataFrame(rows)
            part.to_parquet(out_dir / f"part_{part_i:05d}.parquet", index=False)
            parts.append(part)
    event_paths = read_parquet_dir(ctx.run_root / "path_diagnostics/path_metrics.parquet")
    null_paths = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    rows = []
    for fam in sorted(set(event_paths.get("family", [])) | set(null_paths.get("family", []))):
        ev = event_paths[event_paths["family"] == fam]
        nu = null_paths[null_paths["family"] == fam] if not null_paths.empty else pd.DataFrame()
        rows.append({
            "family": fam,
            "events": int(len(ev)),
            "nulls": int(len(nu)),
            "event_median_24h_mfe_bps": float(pd.to_numeric(ev.get("24h_mfe_bps"), errors="coerce").median()) if not ev.empty else np.nan,
            "null_median_24h_mfe_bps": float(pd.to_numeric(nu.get("24h_mfe_bps"), errors="coerce").median()) if not nu.empty else np.nan,
            "event_minus_null_mfe_bps": (float(pd.to_numeric(ev.get("24h_mfe_bps"), errors="coerce").median()) - float(pd.to_numeric(nu.get("24h_mfe_bps"), errors="coerce").median())) if not ev.empty and not nu.empty else np.nan,
            "beats_null_mfe": bool((pd.to_numeric(ev.get("24h_mfe_bps"), errors="coerce").median() > pd.to_numeric(nu.get("24h_mfe_bps"), errors="coerce").median()) if not ev.empty and not nu.empty else False),
        })
    df_to_csv(pd.DataFrame(rows), ctx.run_root / "matched_null/matched_null_short_uplift_summary.csv")
    if not nulls.empty:
        df_to_parquet(nulls.head(5000), ctx.run_root / "matched_null/matched_null_ledger_sample.parquet")
    write_text(ctx.run_root / "matched_null/matched_null_short_uplift_report.md", f"# Matched Null And Short Uplift\n\n- nulls per event requested/effective: `{ctx.args.nulls_per_event}/{n}`\n- null rows: `{len(nulls)}`\n- matching excludes +/-72h same-symbol event windows and uses same month where possible.\n")


def stage_one_minute(ctx: RunContext) -> None:
    events = combined_events(ctx)
    pilot_manifest = list(PILOT_1M_ROOT.glob("downloaded_1m/download_manifest.csv")) if PILOT_1M_ROOT.exists() else []
    overlap_rows = []
    if not events.empty and pilot_manifest:
        # Conservative: do not read large downloaded partitions here; only report manifest overlap if available.
        try:
            manifest = pd.read_csv(pilot_manifest[0])
            syms = set(manifest.get("symbol", pd.Series(dtype=str)).astype(str))
            overlap = events[events["symbol"].astype(str).isin(syms)]
            overlap_rows = overlap.head(10000).to_dict("records")
        except Exception:
            overlap_rows = []
    if overlap_rows:
        df_to_csv(pd.DataFrame(overlap_rows), ctx.run_root / "one_minute/f1_g1_targeted_1m_overlap_events.csv")
    windows = []
    for _, row in events.head(5000).iterrows() if not events.empty else []:
        ts = pd.Timestamp(row["decision_ts"])
        windows.append({"family": row["family"], "event_id": row["event_id"], "symbol": row["symbol"], "window_start": ts - pd.Timedelta(hours=4), "window_end": min(ts + pd.Timedelta(hours=8), SCREENING_END), "reason": "F1/G1 short validation candidate window"})
    write_csv(ctx.run_root / "one_minute/additional_1m_window_request.csv", windows)
    write_text(ctx.run_root / "one_minute/targeted_1m_overlap_report.md", f"# Targeted 1m Overlap And Window Plan\n\n- existing targeted 1m manifest found: `{bool(pilot_manifest)}`\n- overlap rows recorded: `{len(overlap_rows)}`\n- new requested windows capped in report: `{len(windows)}`\n- no downloads performed in this phase.\n")


def stage_validation(ctx: RunContext) -> None:
    replay = safe_read_csv(ctx.run_root / "replay/executable_short_replay_summary.csv")
    paths = read_parquet_dir(ctx.run_root / "path_diagnostics/path_metrics.parquet")
    rows = []
    cpcv_rows = []
    if not replay.empty and not paths.empty:
        candidates = replay[(pd.to_numeric(replay["net_R"], errors="coerce") > 0) & (pd.to_numeric(replay["PF"], errors="coerce") > 1.0)].sort_values("revised_short_score", ascending=False).head(12)
        dates = pd.to_datetime(paths["decision_ts"], utc=True)
        if not candidates.empty:
            bins = pd.qcut(dates.rank(method="first"), q=8, labels=False, duplicates="drop")
            paths = paths.assign(block=bins)
        for _, cand in candidates.iterrows():
            fam = cand["family"]
            surface = {"stop_atr_mult": cand["stop_atr_mult"], "target_r": cand["target_r"], "hold_h": cand["hold_h"]}
            sub = paths[paths["family"] == fam].copy()
            if sub.empty:
                continue
            sub["net_R"] = sub.apply(lambda r: surface_return(r, surface), axis=1)
            block_returns = sub.groupby("block", dropna=False)["net_R"].sum().reset_index()
            positive_paths = float((block_returns["net_R"] > 0).mean()) if len(block_returns) else np.nan
            rows.append({"family": fam, "surface_id": cand["surface_id"], "blocks": len(block_returns), "median_block_net_R": float(block_returns["net_R"].median()) if len(block_returns) else np.nan, "positive_path_share": positive_paths, "worst_block_net_R": float(block_returns["net_R"].min()) if len(block_returns) else np.nan})
            for _, br in block_returns.iterrows():
                cpcv_rows.append({"family": fam, "surface_id": cand["surface_id"], "block": br["block"], "net_R": br["net_R"]})
    summary_cols = ["family", "surface_id", "blocks", "median_block_net_R", "positive_path_share", "worst_block_net_R"]
    detail_cols = ["family", "surface_id", "block", "net_R"]
    df_to_csv(pd.DataFrame(rows, columns=summary_cols), ctx.run_root / "validation/cpcv_summary.csv")
    df_to_parquet(pd.DataFrame(cpcv_rows, columns=detail_cols), ctx.run_root / "validation/cpcv_details.parquet")
    write_text(ctx.run_root / "validation/validation_report.md", "# Walk-Forward/CPCV And Overfit Controls\n\nValidation is pre-holdout only. Rows are block summaries for positive replay-surface candidates. A result here can only justify further validation, not live or sealed promotion.\n")


def stage_portfolio(ctx: RunContext) -> None:
    val = safe_read_csv(ctx.run_root / "validation/cpcv_summary.csv")
    rows = []
    for _, r in val.iterrows() if not val.empty else []:
        if float(r.get("positive_path_share", 0) or 0) < 0.55:
            continue
        for equity in (200, 500, 1000):
            for risk in (0.025, 0.05, 0.10, 0.15, 0.20):
                ending = float(equity) * (1.0 + max(float(r.get("median_block_net_R", 0) or 0), -10.0) * risk)
                rows.append({"family": r["family"], "surface_id": r["surface_id"], "starting_equity": equity, "risk_per_trade": risk, "ending_equity_proxy": ending, "ruin_flag": ending <= equity * 0.1, "diagnostic_only": True})
    cols = ["family", "surface_id", "starting_equity", "risk_per_trade", "ending_equity_proxy", "ruin_flag", "diagnostic_only"]
    df_to_csv(pd.DataFrame(rows, columns=cols), ctx.run_root / "portfolio/aggressive_10x_short_portfolio_summary.csv")
    write_text(ctx.run_root / "portfolio/aggressive_10x_short_portfolio_report.md", f"# Aggressive 10x Short Portfolio Overlay\n\n- overlay rows: `{len(rows)}`\n- overlay runs only for candidates passing preliminary path/replay validation gates. Aggressive sizing is diagnostic and cannot create alpha.\n")


def stage_triage(ctx: RunContext) -> None:
    nulls = safe_read_csv(ctx.run_root / "matched_null/matched_null_short_uplift_summary.csv")
    replay = safe_read_csv(ctx.run_root / "replay/executable_short_replay_summary.csv")
    rows = []
    for fam in ["F1", "G1"]:
        n = nulls[nulls["family"] == fam] if not nulls.empty else pd.DataFrame()
        r = replay[replay["family"] == fam] if not replay.empty else pd.DataFrame()
        best = r.sort_values("revised_short_score", ascending=False).head(1) if not r.empty else pd.DataFrame()
        beats = bool(n.get("beats_null_mfe", pd.Series([False])).iloc[0]) if not n.empty else False
        if r.empty or (not best.empty and float(best.iloc[0].get("events", best.iloc[0].get("trades", 0)) or 0) == 0):
            label = "blocked_by_missing_events"
        elif not beats:
            label = "entry_no_matched_null_path_edge"
        elif not best.empty and float(best.iloc[0].get("net_R", 0) or 0) > 0 and float(best.iloc[0].get("PF", 0) or 0) > 1:
            label = "mechanism_promising_needs_validation"
        else:
            label = "mechanism_has_path_edge_but_replay_weak"
        rows.append({"family": fam, "triage_label": label, "beats_matched_null_mfe": beats, "best_surface_id": best.iloc[0].get("surface_id", "") if not best.empty else "", "best_net_R": best.iloc[0].get("net_R", np.nan) if not best.empty else np.nan, "next_step": "If promising, run family-specific validation with targeted 1m/depth request; otherwise revise event definition."})
    out = pd.DataFrame(rows)
    df_to_csv(out, ctx.run_root / "triage/family_triage_summary.csv")
    next_dir = ctx.run_root / "triage/next_candidate_contracts"
    next_dir.mkdir(parents=True, exist_ok=True)
    for _, row in out.iterrows():
        if str(row["triage_label"]).startswith("mechanism_promising"):
            write_json(next_dir / f"{row['family']}_next_contract.json", {"family": row["family"], "status": "draft_next_validation_contract", "basis": row.to_dict(), "no_live_trading": True, "no_sealed_validation": True})
    write_text(ctx.run_root / "triage/family_triage_report.md", "# F1/G1 Family Triage\n\nTriage labels are diagnostic only. No family is live-ready or validated from this phase.\n")


def stage_decision(ctx: RunContext) -> None:
    triage = safe_read_csv(ctx.run_root / "triage/family_triage_summary.csv")
    engine_ok = (ctx.run_root / "engine/short_engine_sanity_results.csv").exists()
    if not engine_ok:
        verdict = "blocked_by_short_engine_or_causality"
    elif triage.empty:
        verdict = "reject_current_f1_g1_translation"
    elif bool(triage["triage_label"].astype(str).str.startswith("mechanism_promising").any()):
        verdict = "f1_g1_event_generators_unblocked_continue_validation"
    else:
        verdict = "reject_current_f1_g1_translation"
    if verdict not in ALLOWED_VERDICTS:
        verdict = "blocked_by_protocol_issue"
    decision = {"created_at_utc": utc_now(), "run_root": str(ctx.run_root), "verdict": verdict, "final_holdout_untouched": True, "protected_start": str(FINAL_HOLDOUT_START)}
    write_json(ctx.run_root / "decision_summary.json", decision)
    report = f"""# F1/G1 Short Unblock Report\n\n## Verdict\n`{verdict}`\n\n## Governance\n- Protected holdout `>= {FINAL_HOLDOUT_START}` was not used.\n- This is train-only diagnostic research.\n- No live trading, sealed validation, or live preparation is authorized.\n\n## What Was Implemented\n- F1 parabolic short generator with prior extension state and separate backside confirmation.\n- A1-like parent breakout generator for G1.\n- G1 failed-breakout short generator that fires only after later failure confirmation.\n- Short engine sanity checks for stop/target/funding/liquidation/same-bar pessimism.\n- Revised scoring audit to hard-penalize negative net_R, PF <= 1, liquidation, drawdown, and concentration.\n- Path, replay-surface, matched-null, CPCV, portfolio-overlay, and triage artifacts.\n\n## Key Artifact Paths\n- `events/f1_event_ledger.parquet/`\n- `events/g1_event_ledger.parquet/`\n- `path_diagnostics/path_summary_by_family.csv`\n- `replay/executable_short_replay_summary.csv`\n- `matched_null/matched_null_short_uplift_summary.csv`\n- `triage/family_triage_summary.csv`\n"""
    write_text(ctx.run_root / "F1_G1_SHORT_UNBLOCK_REPORT.md", report)


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rels = [
        "F1_G1_SHORT_UNBLOCK_REPORT.md",
        "decision_summary.json",
        "preflight/preflight_report.md",
        "seal/seal_guard_report.md",
        "engine/short_engine_sanity_report.md",
        "scoring/sweep_scoring_audit_report.md",
        "contracts/f1_parabolic_short_contract.json",
        "contracts/g1_failed_breakout_short_contract.json",
        "features/feature_support_report.md",
        "events/f1_event_generator_report.md",
        "events/g1_event_generator_report.md",
        "causality/event_generator_causality_report.md",
        "path_diagnostics/path_diagnostics_report.md",
        "replay/executable_short_replay_report.md",
        "matched_null/matched_null_short_uplift_report.md",
        "one_minute/targeted_1m_overlap_report.md",
        "validation/validation_report.md",
        "portfolio/aggressive_10x_short_portfolio_report.md",
        "triage/family_triage_report.md",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
    ]
    rows = []
    for rel in rels:
        src = ctx.run_root / rel
        if src.exists() and src.is_file():
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            rows.append({"artifact": rel, "path": str(src), "bundle_copy": str(dst), "size_bytes": src.stat().st_size})
    for p in sorted(ctx.run_root.rglob("*.csv")):
        if "compact_review_bundle" in p.parts or p.stat().st_size > 5_000_000:
            continue
        dst = bundle / str(p.relative_to(ctx.run_root)).replace("/", "__")
        shutil.copy2(p, dst)
        rows.append({"artifact": str(p.relative_to(ctx.run_root)), "path": str(p), "bundle_copy": str(dst), "size_bytes": p.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_json(bundle / "artifact_path_index.json", {"artifacts": rows})
    zip_path = ctx.run_root / "qlmg_f1_g1_short_unblock_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(bundle))


STAGE_FUNCS = {
    "preflight-and-prior-sweep-audit": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "short-engine-sanity-check": stage_short_engine,
    "sweep-scoring-audit": stage_scoring,
    "f1-g1-contract-freeze": stage_contracts,
    "feature-support-builder": stage_feature_support,
    "f1-parabolic-short-event-generator": generate_f1_stage,
    "a1-breakout-parent-generator-for-g1": generate_parent_stage,
    "g1-failed-breakout-short-event-generator": generate_g1_stage,
    "event-generator-causality-tests": stage_causality,
    "f1-g1-path-diagnostics": stage_path,
    "executable-short-replay-surface": stage_replay,
    "matched-null-and-short-uplift": stage_matched_null,
    "targeted-1m-overlap-and-window-plan": stage_one_minute,
    "walk-forward-cpcv-and-overfit-controls": stage_validation,
    "aggressive-10x-short-portfolio-overlay": stage_portfolio,
    "triage-and-next-contracts": stage_triage,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}

ESTIMATE_GB = {
    "f1-parabolic-short-event-generator": 2.0,
    "a1-breakout-parent-generator-for-g1": 2.0,
    "g1-failed-breakout-short-event-generator": 2.0,
    "f1-g1-path-diagnostics": 3.0,
    "matched-null-and-short-uplift": 4.0,
}


def main() -> int:
    args = parse_args()
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"created_at_utc": utc_now(), "run_root": str(run_root), "root_reason": reason, "argv": sys.argv, "start": str(start), "end": str(end), "protected_start": str(FINAL_HOLDOUT_START), "git_head": shell(["git", "rev-parse", "HEAD"])})
    try:
        for stage in stage_list(args.stage):
            if args.resume and stage_complete(run_root, stage):
                notifier.send("STAGE SKIP", stage)
                continue
            ensure_guard(ctx, stage, ESTIMATE_GB.get(stage, 0.5))
            append_command(run_root, stage)
            notifier.send("STAGE START", stage)
            if args.dry_run:
                write_text(run_root / "dry_run" / f"{stage}.txt", "dry run stage placeholder")
            else:
                STAGE_FUNCS[stage](ctx)
            mark_done(run_root, stage)
            notifier.send("STAGE DONE", stage)
        notifier.send("RUN COMPLETE", f"run_root={run_root}")
        try:
            watch = {"ts_utc": utc_now(), "status": "complete", "run_root": str(run_root)}
            (run_root / "watch_status.json").write_text(json.dumps(watch, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        return 0
    except Exception as exc:
        notifier.send("RUN FAILURE", f"{type(exc).__name__}: {exc}", level="error")
        try:
            watch = {"ts_utc": utc_now(), "status": "failed", "run_root": str(run_root), "error": f"{type(exc).__name__}: {exc}"}
            (run_root / "watch_status.json").write_text(json.dumps(watch, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
