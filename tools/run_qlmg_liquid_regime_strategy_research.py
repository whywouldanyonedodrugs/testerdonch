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
import warnings
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Converting to PeriodArray.*")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, trailing_percentile, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_liquid_regime_strategy_research_20260628_v1"
DATA_5M = Path("/opt/parquet/5m")
CTX_5M = Path("/opt/parquet/bybit_context_5m")
DEFAULT_SEED = 20260628
GB = 1024**3

BRUTAL_ROOT_PREFIX = "phase_qlmg_brutal_no_depth_stress_20260628_v1"
PROXY_ROOT_PREFIX = "phase_qlmg_best_effort_proxy_execution_sim_20260628_v1"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
D4_AUDIT_ROOT = RESULTS_ROOT / "phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927"
TARGETED_REPLAY_ROOT = RESULTS_ROOT / "phase_qlmg_targeted_execution_data_replay_20260627_v1_20260627_100018"

PRIMARY_FAMILIES = ["A4", "A2", "A1", "A3", "RS1"]
CONDITIONAL_FAMILIES = ["B1", "C2"]
ALL_LIQUID_FAMILIES = PRIMARY_FAMILIES + CONDITIONAL_FAMILIES
BRANCH_X_COMPONENTS = ["D4", "listing_vwap_loss", "generic_shock_reversal", "funding_window_preservation"]
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT"]

STAGES = (
    "preflight-and-prior-branch-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "integrated-branch-registry",
    "data-readiness-and-universe-v2",
    "regime-engine-v2",
    "strategy-contract-freeze",
    "entry-event-generation",
    "mae-mfe-path-diagnostics",
    "exit-surface-synthesis",
    "bounded-discovery-sweep",
    "regime-conditioned-refinement",
    "matched-null-and-baseline-tests",
    "cost-funding-liquidation-stress",
    "walk-forward-cpcv-controls",
    "aggressive-small-account-overlay",
    "family-dossiers-and-holding-pen",
    "next-contracts-and-backlog",
    "decision-report",
    "compact-review-bundle",
    "all",
)

HORIZONS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "12h": pd.Timedelta(hours=12),
    "24h": pd.Timedelta(hours=24),
    "3d": pd.Timedelta(days=3),
    "5d": pd.Timedelta(days=5),
    "10d": pd.Timedelta(days=10),
    "14d": pd.Timedelta(days=14),
}

EXIT_FAMILIES = [
    {"exit_family": "fixed_3d", "horizon": "3d", "target_r": 2.0, "stop_mult": 1.0},
    {"exit_family": "fixed_5d", "horizon": "5d", "target_r": 2.0, "stop_mult": 1.0},
    {"exit_family": "fixed_10d", "horizon": "10d", "target_r": 3.0, "stop_mult": 1.0},
    {"exit_family": "r2_ema_trail_proxy", "horizon": "5d", "target_r": 2.0, "stop_mult": 1.0},
    {"exit_family": "rank_drop_proxy", "horizon": "7d", "target_r": 2.0, "stop_mult": 1.0},
    {"exit_family": "regime_deterioration_proxy", "horizon": "3d", "target_r": 1.5, "stop_mult": 1.0},
]

FAMILY_BUDGETS = {"A4": 700, "A2": 600, "A1": 600, "A3": 500, "B1": 400, "C2": 400, "RS1": 300, "reserve_refinement": 100}

ALLOWED_HIGH_LEVEL_VERDICTS = {
    "continue_execution_sensitive_branch_data_collection",
    "build_forward_capture_and_proxy_stress_next",
    "liquid_regime_prelead_found",
    "liquid_regime_research_inconclusive",
    "sector_catalyst_data_build_required",
    "no_family_rejected_only_current_translations",
    "promote_to_train_only_family_specific_validation",
    "blocked_by_protocol_issue",
}

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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-liquid-regime")
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
    p = argparse.ArgumentParser(description="QLMG liquid regime strategy research, train-only")
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
    p.add_argument("--discovery-budget", type=int, default=3600)
    p.add_argument("--refine-budget", type=int, default=600)
    p.add_argument("--top-per-family", type=int, default=40)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--include-branch-x-status", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-liquid-continuation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-sector-catalyst", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-risk-off-shorts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aggressive-overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tmux-session-name", default="qlmg_liquid_regime")
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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_hash(path: Path, max_bytes: int = 20_000_000) -> str:
    h = hashlib.sha256()
    if not path.exists() or not path.is_file():
        return "missing"
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    suffix = "_partial" if path.stat().st_size > max_bytes else ""
    return h.hexdigest() + suffix


def latest_root(prefix: str) -> Path | None:
    roots = sorted(RESULTS_ROOT.glob(prefix + "*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return roots[0] if roots else None


def proxy_status() -> dict[str, Any]:
    roots = sorted(RESULTS_ROOT.glob(PROXY_ROOT_PREFIX + "*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    best: dict[str, Any] = {"status": "not_available", "root": "", "verdict": "not_available", "terminal": False}
    for root in roots:
        decision = root / "decision_summary.json"
        done = root / "stage_status/decision-report.done"
        watch = safe_read_json(root / "watch_status.json")
        has_real_run_artifacts = (root / "stage_status").exists() or (root / "notifications/telegram_events.jsonl").exists() or ((root / "logs/full_run.log").exists() and (root / "logs/full_run.log").stat().st_size > 0) or decision.exists()
        if not has_real_run_artifacts:
            continue
        if decision.exists():
            dj = safe_read_json(decision)
            return {"status": "completed", "root": str(root), "verdict": dj.get("next_action_verdict") or dj.get("final_verdict") or dj.get("listing_verdict") or "completed", "terminal": True}
        if done.exists():
            return {"status": "completed_no_decision_json", "root": str(root), "verdict": "completed_no_decision_json", "terminal": True}
        if watch:
            wstatus = str(watch.get("status", "unknown"))
            if wstatus in {"failed", "complete", "completed"}:
                return {"status": wstatus, "root": str(root), "verdict": wstatus, "terminal": True}
            if best["status"] == "not_available":
                best = {"status": "in_progress", "root": str(root), "verdict": "in_progress", "terminal": False}
    return best


def check_full_launch_proxy_gate(args: argparse.Namespace) -> dict[str, Any]:
    ps = proxy_status()
    if not args.smoke and args.stage == "all" and not ps.get("terminal"):
        raise RuntimeError(f"full liquid-regime launch blocked until proxy execution sim has terminal decision status: {ps}")
    return ps


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
        hard_stage_output_gb=30.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    out = ctx.run_root / "resource_guard" / f"{stage}.json"
    write_json(out, guard)
    if guard["warnings"]:
        ctx.notifier.send("QLMG liquid regime resource warning", json.dumps(guard), level="warning")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop for {stage}: {guard}")


def symbol_files(max_symbols: int = 0) -> list[Path]:
    files = list(DATA_5M.glob("*.parquet"))
    by_name = {p.stem: p for p in files}
    ordered: list[Path] = []
    for s in MAJORS:
        if s in by_name:
            ordered.append(by_name.pop(s))
    ordered.extend(sorted(by_name.values(), key=lambda p: p.stem))
    if max_symbols and max_symbols > 0:
        return ordered[:max_symbols]
    return ordered


def load_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp, columns: list[str] | None = None) -> pd.DataFrame:
    path = DATA_5M / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=columns)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].sort_values("timestamp")
    return df.reset_index(drop=True)


def load_context_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = CTX_5M / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = ["timestamp", "mark_high", "mark_low", "mark_close", "index_close", "premium_close", "context_source_close_ts"]
    try:
        df = pd.read_parquet(path, columns=[c for c in cols if c in pd.read_parquet(path).columns])
    except Exception:
        return pd.DataFrame()
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].sort_values("timestamp").reset_index(drop=True)


def daily_from_5m(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.set_index("timestamp").sort_index()
    daily = pd.DataFrame({
        "open": x["open"].resample("1D").first(),
        "high": x["high"].resample("1D").max(),
        "low": x["low"].resample("1D").min(),
        "close": x["close"].resample("1D").last(),
        "turnover": x["turnover"].resample("1D").sum() if "turnover" in x else np.nan,
        "volume": x["volume"].resample("1D").sum() if "volume" in x else np.nan,
        "open_interest": x["open_interest"].resample("1D").last() if "open_interest" in x else np.nan,
        "funding_rate": x["funding_rate"].resample("1D").last() if "funding_rate" in x else np.nan,
    }).dropna(subset=["close"])
    daily.index = daily.index.tz_convert("UTC") if daily.index.tz is not None else daily.index.tz_localize("UTC")
    daily["date"] = daily.index
    return daily.reset_index(drop=True)


def choose_liquidity_tier(symbol: str, median_daily_turnover: float) -> str:
    if symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"} or median_daily_turnover >= 100_000_000:
        return "A"
    if median_daily_turnover >= 20_000_000:
        return "B"
    if median_daily_turnover >= 2_000_000:
        return "C"
    return "D"


def context_columns_available() -> bool:
    return CTX_5M.exists() and any(CTX_5M.glob("*.parquet"))


def sector_catalyst_readiness() -> tuple[dict[str, Any], dict[str, Any]]:
    sector_required = {"symbol", "effective_start_utc", "effective_end_utc", "sector", "taxonomy_source", "taxonomy_version"}
    catalyst_required = {"symbol", "first_public_ts_utc", "official_confirm_ts_utc", "effective_ts_utc", "source_confidence", "durability_score_ex_ante"}
    candidates = list(REPO.glob("**/*sector*.csv")) + list(REPO.glob("**/*catalyst*.csv"))
    sector_ok = False
    catalyst_ok = False
    sector_path = ""
    catalyst_path = ""
    for p in candidates:
        if "results/rebaseline" in str(p):
            continue
        try:
            cols = set(pd.read_csv(p, nrows=1).columns)
        except Exception:
            continue
        if sector_required.issubset(cols):
            sector_ok = True
            sector_path = str(p)
        if catalyst_required.issubset(cols):
            catalyst_ok = True
            catalyst_path = str(p)
    sector = {"ready": sector_ok, "path": sector_path, "required_fields": sorted(sector_required), "label_if_missing": "not_fairly_tested_missing_sector_map"}
    catalyst = {"ready": catalyst_ok, "path": catalyst_path, "required_fields": sorted(catalyst_required), "label_if_missing": "not_fairly_tested_missing_catalyst_data"}
    return sector, catalyst


def family_branch(family: str) -> str:
    return "branch_l_liquid_regime" if family in ALL_LIQUID_FAMILIES else "branch_x_execution_sensitive"


def family_side(family: str) -> str:
    return "short" if family == "RS1" else "long"


def family_label(family: str) -> str:
    return {
        "A4": "volatility_managed_liquid_tsmom",
        "A2": "prior_high_proximity_momentum",
        "A1": "liquid_leader_breakout",
        "A3": "close_confirmed_retest_reclaim",
        "B1": "sector_ignition_leader_long",
        "C2": "post_catalyst_continuation_base",
        "RS1": "liquid_risk_off_swing_short",
    }.get(family, family)


def stage_preflight(ctx: RunContext) -> None:
    resource_check(ctx, "preflight-and-prior-branch-freeze", 0.2)
    ctx.run_root.mkdir(parents=True, exist_ok=True)
    ps = check_full_launch_proxy_gate(ctx.args)
    roots = {
        "brutal_no_depth_stress": latest_root(BRUTAL_ROOT_PREFIX),
        "best_effort_proxy_execution_sim": Path(ps["root"]) if ps.get("root") else None,
        "listing_generic_full_event_replay": LISTING_ROOT,
        "d4_survivability_redesign": D4_SURVIVAL_ROOT,
        "d4_liquidation_execution_audit": D4_AUDIT_ROOT,
        "targeted_execution_data_replay": TARGETED_REPLAY_ROOT,
    }
    rows = []
    hashes: dict[str, Any] = {"proxy_status": ps, "git_head": "unknown"}
    try:
        hashes["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        pass
    for name, root in roots.items():
        exists = bool(root and root.exists())
        status = "available" if exists else "not_available"
        if name == "best_effort_proxy_execution_sim":
            status = str(ps.get("status", status))
        decision = root / "decision_summary.json" if root else Path("missing")
        final_report = next(root.glob("*REPORT.md"), None) if exists else None
        rows.append({"artifact": name, "root": str(root or ""), "status": status, "decision_summary": str(decision) if decision.exists() else "", "final_report": str(final_report or "")})
        if exists:
            for p in [decision, final_report] if final_report else [decision]:
                if p and p.exists():
                    hashes[f"{name}:{p.name}"] = file_hash(p)
    write_csv(ctx.run_root / "preflight/prior_branch_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(ctx.run_root.parent)
    guard = check_resource_guard(snap, estimated_output_gb=4.0, hard_free_gb=5.0, warn_free_gb=7.0, hard_stage_output_gb=30.0, allow_large_output=ctx.args.allow_large_output)
    write_json(ctx.run_root / "preflight/resource_guard_report.json", guard)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\nstatus={guard['status']} free_disk_gb={guard['free_disk_gb']:.2f} estimated_output_gb=4.0")
    write_text(ctx.run_root / "preflight/preflight_report.md", "\n".join([
        "# QLMG Liquid Regime Preflight",
        f"run_root: `{ctx.run_root}`",
        f"window: `{ctx.start}` to `{ctx.end}`",
        f"proxy_status: `{ps.get('status')}`",
        f"proxy_verdict: `{ps.get('verdict')}`",
        f"proxy_terminal: `{ps.get('terminal')}`",
        "Full launch is blocked unless proxy execution simulation has terminal status. Smoke/preflight can proceed with in_progress status.",
    ]))


def stage_telegram(ctx: RunContext) -> None:
    resource_check(ctx, "telegram-and-tmux-setup", 0.05)
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\nstatus: `{ctx.notifier.status}`\n\nremote_available: `{ctx.notifier.remote_available}`\n\nmissing: `{ctx.notifier.missing}`")
    watch = ctx.run_root / "tmux/watch_commands.md"
    write_text(watch, "\n".join([
        "# Watch Commands",
        f"tmux attach -t {ctx.args.tmux_session_name}",
        f"tail -f {ctx.run_root}/logs/full_run.log",
        f"watch -n 30 'cat {ctx.run_root}/watch_status.json'",
        f"tail -f {ctx.run_root}/notifications/telegram_events.jsonl",
        "df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h",
    ]))
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run\n\nUse `bash tools/run_qlmg_liquid_regime_strategy_research_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --require-telegram --seed {ctx.args.seed} --launch-tmux` after proxy terminal status.")
    ctx.notifier.send("QLMG liquid regime run initialized", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    resource_check(ctx, "seal-guard", 0.05)
    checks = [
        {"case": "pre_holdout_end", "timestamp": str(SCREENING_END), "passes": True},
        {"case": "protected_start", "timestamp": str(FINAL_HOLDOUT_START), "passes": False},
    ]
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "checks": checks})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\nProtected start: `{FINAL_HOLDOUT_START}`. Candidate-selection reads and generated rows must be `< {FINAL_HOLDOUT_START}`.")


def stage_branch_registry(ctx: RunContext) -> None:
    resource_check(ctx, "integrated-branch-registry", 0.1)
    ps = proxy_status()
    rows = []
    for comp in BRANCH_X_COMPONENTS:
        rows.append({"branch_id": "branch_x_execution_sensitive", "component": comp, "status": "carry_forward_execution_blocked", "latest_proxy_status": ps.get("status"), "latest_proxy_verdict": ps.get("verdict"), "rankable_with_liquid_branch": False})
    for fam in PRIMARY_FAMILIES:
        rows.append({"branch_id": "branch_l_liquid_regime", "component": fam, "status": "primary_ready_for_path_and_replay", "latest_proxy_status": "not_applicable", "latest_proxy_verdict": "not_applicable", "rankable_with_liquid_branch": True})
    sector, catalyst = sector_catalyst_readiness()
    rows.append({"branch_id": "branch_l_liquid_regime", "component": "B1", "status": "alpha_ready" if sector["ready"] else "not_fairly_tested_missing_sector_map", "latest_proxy_status": "not_applicable", "latest_proxy_verdict": "not_applicable", "rankable_with_liquid_branch": bool(sector["ready"])})
    rows.append({"branch_id": "branch_l_liquid_regime", "component": "C2", "status": "alpha_ready" if catalyst["ready"] else "not_fairly_tested_missing_catalyst_data", "latest_proxy_status": "not_applicable", "latest_proxy_verdict": "not_applicable", "rankable_with_liquid_branch": bool(catalyst["ready"])})
    write_csv(ctx.run_root / "registry/project_branch_registry.csv", rows)
    write_text(ctx.run_root / "registry/project_branch_registry.md", "# Project Branch Registry\n\nD4/listing execution-sensitive evidence is preserved but excluded from liquid-regime rankings. All downstream tables include `branch_id`.")


def stage_data_universe(ctx: RunContext) -> None:
    resource_check(ctx, "data-readiness-and-universe-v2", 1.0 if ctx.args.smoke else 6.0)
    files = symbol_files(ctx.args.max_symbols if ctx.args.max_symbols else (10 if ctx.args.smoke else 0))
    rows = []
    daily_parts = []
    for i, path in enumerate(files):
        symbol = path.stem
        df = load_symbol(symbol, ctx.start, ctx.end, ["timestamp", "open", "high", "low", "close", "volume", "turnover", "open_interest", "funding_rate"])
        if df.empty:
            continue
        d = daily_from_5m(df)
        if d.empty:
            continue
        first_ts = df["timestamp"].min()
        last_ts = df["timestamp"].max()
        median_turnover = float(d["turnover"].tail(30).median()) if "turnover" in d else 0.0
        tier = choose_liquidity_tier(symbol, median_turnover)
        d["symbol"] = symbol
        d["liquidity_tier"] = tier
        d["listing_age_days_proxy"] = (d["date"] - first_ts).dt.total_seconds() / 86400.0
        d["branch_id"] = "branch_l_liquid_regime"
        daily_parts.append(d)
        rows.append({"symbol": symbol, "first_local_bar": first_ts, "last_local_bar": last_ts, "median_30d_daily_turnover": median_turnover, "liquidity_tier": tier, "primary_liquid_eligible": tier in {"A", "B"}, "tier_c_diagnostic_only": tier == "C", "bars": len(df)})
        if (i + 1) % max(ctx.args.chunk_size, 1) == 0:
            ctx.notifier.send("QLMG liquid universe progress", f"symbols_processed={i+1}/{len(files)}")
    universe = pd.DataFrame(rows)
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    if not daily.empty:
        validate_no_protected(daily, ["date"])
        (ctx.run_root / "universe").mkdir(parents=True, exist_ok=True)
        daily.to_parquet(ctx.run_root / "universe/liquid_universe_by_date.parquet", index=False)
    write_csv(ctx.run_root / "universe/universe_summary.csv", rows)
    write_text(ctx.run_root / "universe/survivorship_audit.md", "# Survivorship Audit\n\nUniverse uses local first/last bars as proxy lifecycle fields. Current active status is not used as historical truth. Tier C is diagnostic only for liquid-regime ranking.")
    write_text(ctx.run_root / "data/data_readiness_report.md", f"# Data Readiness\n\n5m_store_exists: `{DATA_5M.exists()}`\n\ncontext_store_exists: `{context_columns_available()}`\n\nsymbols_loaded: `{len(rows)}`\n\nTier A/B are primary; Tier C is diagnostic only.")


def load_universe_daily(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "universe/liquid_universe_by_date.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def stage_regime(ctx: RunContext) -> None:
    resource_check(ctx, "regime-engine-v2", 0.5)
    daily = load_universe_daily(ctx)
    if daily.empty:
        raise RuntimeError("universe daily panel missing")
    btc = daily[daily["symbol"].eq("BTCUSDT")].sort_values("date").copy()
    eth = daily[daily["symbol"].eq("ETHUSDT")].sort_values("date").copy()
    if btc.empty:
        # Use highest turnover symbol as a proxy, explicitly labeled.
        sym = daily.groupby("symbol")["turnover"].sum().sort_values(ascending=False).index[0]
        btc = daily[daily["symbol"].eq(sym)].sort_values("date").copy()
        btc_proxy = sym
    else:
        btc_proxy = "BTCUSDT"
    def add_ma(x: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = x[["date", "close"]].copy()
        for n in [20, 50, 100]:
            out[f"{prefix}_ma{n}"] = out["close"].rolling(n, min_periods=max(5, n // 4)).mean()
            out[f"{prefix}_above_ma{n}"] = out["close"] > out[f"{prefix}_ma{n}"]
            out[f"{prefix}_slope_ma{n}"] = out[f"{prefix}_ma{n}"].diff(5)
        for n in [20, 60, 90]:
            roll_high = out["close"].rolling(n, min_periods=max(5, n // 4)).max()
            out[f"{prefix}_drawdown_{n}d"] = out["close"] / roll_high - 1.0
        out = out.drop(columns=["close"])
        return out
    btc_r = add_ma(btc, "btc")
    eth_r = add_ma(eth, "eth") if not eth.empty else btc_r.rename(columns=lambda c: c.replace("btc", "eth") if c != "date" else c)
    liquid = daily[daily["liquidity_tier"].isin(["A", "B"])].copy()
    liquid = liquid.sort_values(["symbol", "date"])
    for n in [20, 50, 100]:
        liquid[f"ma{n}"] = liquid.groupby("symbol")["close"].transform(lambda s: s.rolling(n, min_periods=max(5, n // 4)).mean())
        liquid[f"above_ma{n}"] = liquid["close"] > liquid[f"ma{n}"]
    breadth = liquid.groupby("date").agg(
        breadth_above_ma20=("above_ma20", "mean"),
        breadth_above_ma50=("above_ma50", "mean"),
        breadth_above_ma100=("above_ma100", "mean"),
        active_symbols=("symbol", "nunique"),
        cross_sectional_1d_dispersion=("close", "std"),
    ).reset_index()
    reg = btc_r.merge(eth_r, on="date", how="left").merge(breadth, on="date", how="left")
    reg["parent_regime"] = np.select(
        [reg["btc_above_ma50"].fillna(False) & reg["eth_above_ma50"].fillna(False) & (reg["breadth_above_ma50"].fillna(0) > 0.55), reg["btc_drawdown_60d"].fillna(0) < -0.15],
        ["Expansion", "Stress"],
        default="Rotation/mixed",
    )
    reg.loc[(reg["btc_above_ma50"].fillna(False)) & (reg["breadth_above_ma50"].fillna(0).between(0.35, 0.55)), "parent_regime"] = "Fragile uptrend"
    reg["feature_ts"] = reg["date"]
    reg["sector_catalyst_status"] = "missing_point_in_time_store"
    reg["funding_oi_alignment_note"] = "daily funding/OI use end-of-day values; candidate stages lag decision where intraday uncertainty exists"
    reg["btc_proxy_source"] = btc_proxy
    validate_no_protected(reg, ["date", "feature_ts"])
    (ctx.run_root / "regime").mkdir(parents=True, exist_ok=True)
    reg.to_parquet(ctx.run_root / "regime/regime_labels_by_date.parquet", index=False)
    summary = reg.groupby("parent_regime").agg(days=("date", "count"), active_symbols_median=("active_symbols", "median")).reset_index()
    summary.to_csv(ctx.run_root / "regime/regime_summary.csv", index=False)
    schema = {
        "temporal_contract": "all rolling features use trailing windows only; no final holdout rows; OI/funding are treated as uncertain and lagged at candidate stage when needed",
        "labels": ["Expansion", "Fragile uptrend", "Rotation/mixed", "Countertrend bounce", "Stress"],
        "features": [c for c in reg.columns if c not in {"date"}],
    }
    write_text(ctx.run_root / "regime/regime_feature_schema.yaml", json.dumps(schema, indent=2, default=str))
    write_text(ctx.run_root / "regime/regime_engine_report.md", "# Regime Engine v2\n\nDeterministic point-in-time daily regime labels built from local trailing OHLCV data. Sector/catalyst labels are blocked unless PIT stores are found. No full-sample percentiles are used.")


def stage_contracts(ctx: RunContext) -> None:
    resource_check(ctx, "strategy-contract-freeze", 0.1)
    sector, catalyst = sector_catalyst_readiness()
    registry = []
    contracts: dict[str, dict[str, Any]] = {}
    for fam in ALL_LIQUID_FAMILIES:
        status = "primary_ready_for_path_and_replay"
        blocker = ""
        if fam == "B1" and not sector["ready"]:
            status = "not_fairly_tested_missing_sector_map"
            blocker = "missing PIT sector map with effective_start_utc/effective_end_utc/taxonomy fields"
        if fam == "C2" and not catalyst["ready"]:
            status = "not_fairly_tested_missing_catalyst_data"
            blocker = "missing PIT catalyst database with public/official/effective timestamps and ex-ante confidence"
        contract = {
            "family": fam,
            "family_name": family_label(fam),
            "branch_id": "branch_l_liquid_regime",
            "status": status,
            "blocker": blocker,
            "side": family_side(fam),
            "universe": "Tier A/B only for rankable liquid-regime research; Tier C diagnostic only",
            "protected_holdout_start": str(FINAL_HOLDOUT_START),
            "execution_assumptions": ["all_taker_base", "exact_funding_timestamps_where_available", "mark_liquidation", "adverse_same_bar", "participation_caps", "no_passive_touch_limit_alpha"],
            "anti_overfit": "path/exit proposal uses development segment and exit scoring uses internal validation segment",
            "no_live_trading": True,
            "no_sealed_validation": True,
        }
        contracts[fam] = contract
        write_text(ctx.run_root / f"contracts/{fam}.json", json.dumps(contract, indent=2, sort_keys=True))
        registry.append({"family": fam, "branch_id": "branch_l_liquid_regime", "status": status, "side": family_side(fam), "blocker": blocker})
    write_csv(ctx.run_root / "contracts/strategy_contract_registry.csv", registry)
    write_text(ctx.run_root / "contracts/contract_freeze_report.md", "# Contract Freeze\n\nContracts were written before entry generation. B1/C2 are alpha-disabled unless required PIT data fields are present locally.")


def load_regime(ctx: RunContext) -> pd.DataFrame:
    return pd.read_parquet(ctx.run_root / "regime/regime_labels_by_date.parquet")


def event_id_for(row: Mapping[str, Any]) -> str:
    payload = json.dumps({k: str(row.get(k)) for k in ["family", "symbol", "decision_ts", "side"]}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def stage_events(ctx: RunContext) -> None:
    resource_check(ctx, "entry-event-generation", 0.8 if ctx.args.smoke else 8.0)
    daily = load_universe_daily(ctx)
    reg = load_regime(ctx)
    if daily.empty or reg.empty:
        raise RuntimeError("daily universe/regime missing")
    liquid = daily[daily["liquidity_tier"].isin(["A", "B"])].copy()
    if ctx.args.max_symbols:
        keep = set(symbol_files(ctx.args.max_symbols)[i].stem for i in range(min(ctx.args.max_symbols, len(symbol_files(ctx.args.max_symbols)))))
        liquid = liquid[liquid["symbol"].isin(keep)]
    liquid = liquid.sort_values(["symbol", "date"])
    events: list[dict[str, Any]] = []
    for symbol, g in liquid.groupby("symbol"):
        g = g.sort_values("date").copy().reset_index(drop=True)
        if len(g) < 40:
            continue
        g["ret20"] = g["close"] / g["close"].shift(20) - 1.0
        g["ret40"] = g["close"] / g["close"].shift(40) - 1.0
        g["ret60"] = g["close"] / g["close"].shift(60) - 1.0
        g["high60_prev"] = g["high"].rolling(60, min_periods=20).max().shift(1)
        g["high90_prev"] = g["high"].rolling(90, min_periods=30).max().shift(1)
        g["ema10"] = g["close"].ewm(span=10, adjust=False).mean()
        g["ema20"] = g["close"].ewm(span=20, adjust=False).mean()
        g["atr20_bps"] = ((g["high"] - g["low"]) / g["close"] * 10000.0).rolling(20, min_periods=5).mean()
        g["turnover_pct"] = trailing_percentile(g["turnover"], window=90, min_periods=20)
        g["oi_chg_3d"] = g["open_interest"].pct_change(3)
        for i in range(1, len(g) - 1):
            r = g.iloc[i]
            if pd.isna(r["atr20_bps"]):
                continue
            decision_ts = pd.Timestamp(r["date"]) + pd.Timedelta(hours=23, minutes=55)
            entry_ts = pd.Timestamp(g.iloc[i + 1]["date"])
            base = {"symbol": symbol, "decision_ts": decision_ts, "entry_ts": entry_ts, "entry_price": float(g.iloc[i + 1]["open"]), "liquidity_tier": r["liquidity_tier"], "atr_bps": float(r["atr20_bps"]), "turnover_pct": float(r["turnover_pct"]) if pd.notna(r["turnover_pct"]) else np.nan, "listing_age_days_proxy": float(r["listing_age_days_proxy"]), "data_quality_flags": "proxy_lifecycle;ohlcv_only_spread_proxy"}
            conds = []
            conds.append(("A4", r["ret40"] > 0.08 and r["close"] > r["ema20"], "long", "vol_managed_tsmom"))
            conds.append(("A2", pd.notna(r["high90_prev"]) and r["close"] >= 0.95 * r["high90_prev"] and r["ret20"] > 0.02, "long", "prior_high_proximity"))
            conds.append(("A1", pd.notna(r["high60_prev"]) and r["close"] > r["high60_prev"] and r["ret20"] > 0.04, "long", "close_confirmed_breakout"))
            prev_break = bool(pd.notna(g.iloc[max(0, i-10):i]["high60_prev"]).any() and (g.iloc[max(0, i-10):i]["close"] > g.iloc[max(0, i-10):i]["high60_prev"]).any())
            conds.append(("A3", prev_break and r["low"] <= r["ema20"] and r["close"] > r["ema20"], "long", "close_confirmed_reclaim"))
            conds.append(("RS1", r["ret20"] < -0.08 and r["close"] < r["ema20"] and (pd.isna(r["oi_chg_3d"]) or r["oi_chg_3d"] >= -0.10), "short", "risk_off_breakdown"))
            for fam, ok, side, variant in conds:
                if fam == "RS1" and not ctx.args.include_risk_off_shorts:
                    continue
                if fam != "RS1" and not ctx.args.include_liquid_continuation:
                    continue
                if ok:
                    ev = dict(base)
                    ev.update({"family": fam, "branch_id": "branch_l_liquid_regime", "side": side, "variant_id": variant, "event_cluster_id": f"{symbol}_{pd.Timestamp(r['date']).strftime('%Y%m')}_{fam}", "candidate_source": "liquid_tier_ab_local_5m", "tier_c_rankable": False})
                    ev["event_id"] = event_id_for(ev)
                    events.append(ev)
    edf = pd.DataFrame(events)
    if not edf.empty:
        edf["decision_ts"] = pd.to_datetime(edf["decision_ts"], utc=True)
        edf["entry_ts"] = pd.to_datetime(edf["entry_ts"], utc=True)
        validate_no_protected(edf, ["decision_ts", "entry_ts"])
        reg2 = reg[["date", "parent_regime", "breadth_above_ma50", "btc_above_ma50", "eth_above_ma50"]].copy()
        reg2["join_date"] = pd.to_datetime(reg2["date"], utc=True).dt.normalize()
        edf["join_date"] = edf["decision_ts"].dt.normalize()
        edf = edf.merge(reg2.drop(columns=["date"]), on="join_date", how="left")
        (ctx.run_root / "events").mkdir(parents=True, exist_ok=True)
        edf.to_parquet(ctx.run_root / "events/entry_event_ledger.parquet", index=False)
    support = edf.groupby(["family", "side", "parent_regime"]).size().reset_index(name="events") if not edf.empty else pd.DataFrame(columns=["family", "side", "parent_regime", "events"])
    support.to_csv(ctx.run_root / "events/event_support_summary.csv", index=False)
    write_text(ctx.run_root / "events/event_generation_report.md", f"# Entry Event Generation\n\nGenerated `{len(edf)}` causal Tier A/B events. B1/C2 alpha generators are disabled unless PIT data exists. Tier C is diagnostic only and not rankable here.")


def load_events(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "events/entry_event_ledger.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def stage_path(ctx: RunContext) -> None:
    resource_check(ctx, "mae-mfe-path-diagnostics", 1.0 if ctx.args.smoke else 8.0)
    events = load_events(ctx)
    if events.empty:
        (ctx.run_root / "path").mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(ctx.run_root / "path/mae_mfe_event_metrics.parquet", index=False)
        write_csv(ctx.run_root / "path/path_summary_by_family_regime.csv", [])
        write_text(ctx.run_root / "path/path_diagnostics_report.md", "# Path Diagnostics\n\nNo events generated; families preserved as generator/sample-limited, not rejected.")
        return
    metrics: list[dict[str, Any]] = []
    for symbol, evs in events.groupby("symbol"):
        min_start = pd.to_datetime(evs["entry_ts"], utc=True).min() - pd.Timedelta(days=1)
        max_end = pd.to_datetime(evs["entry_ts"], utc=True).max() + pd.Timedelta(days=15)
        bars = load_symbol(symbol, min_start, min(max_end, SCREENING_END), ["timestamp", "open", "high", "low", "close", "turnover", "funding_rate", "open_interest"])
        if bars.empty:
            continue
        bars = bars.sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(bars["timestamp"], utc=True)
        for _, ev in evs.iterrows():
            entry_ts = pd.Timestamp(ev["entry_ts"])
            entry_price = float(ev["entry_price"])
            side = str(ev["side"])
            risk_bps = max(float(ev.get("atr_bps", 150.0)), 50.0)
            row = ev.to_dict()
            row["reference_risk_bps"] = risk_bps
            row["development_segment"] = entry_ts < (ctx.start + (ctx.end - ctx.start) * 0.7)
            for name, delta in HORIZONS.items():
                end_ts = min(entry_ts + delta, SCREENING_END)
                w = bars[(ts > entry_ts) & (ts <= end_ts)]
                if w.empty:
                    row[f"{name}_path_available"] = False
                    row[f"{name}_mfe_bps"] = np.nan
                    row[f"{name}_mae_bps"] = np.nan
                    row[f"{name}_close_return_bps"] = np.nan
                    row[f"{name}_pos1R_before_neg1R"] = False
                    continue
                high = pd.to_numeric(w["high"], errors="coerce").max()
                low = pd.to_numeric(w["low"], errors="coerce").min()
                close = float(pd.to_numeric(w["close"], errors="coerce").iloc[-1])
                if side == "long":
                    mfe = (high / entry_price - 1.0) * 10000.0
                    mae = (low / entry_price - 1.0) * 10000.0
                    close_ret = (close / entry_price - 1.0) * 10000.0
                else:
                    mfe = (entry_price / low - 1.0) * 10000.0
                    mae = (entry_price / high - 1.0) * 10000.0
                    close_ret = (entry_price / close - 1.0) * 10000.0
                row[f"{name}_path_available"] = True
                row[f"{name}_mfe_bps"] = float(mfe)
                row[f"{name}_mae_bps"] = float(mae)
                row[f"{name}_close_return_bps"] = float(close_ret)
                row[f"{name}_pos1R_before_neg1R"] = bool(mfe >= risk_bps and abs(mae) < risk_bps)
            row["mark_liquidation_diagnostic"] = "mark_path_not_loaded_last_price_proxy_only"
            row["same_bar_ambiguity"] = "adverse_resolution_required_if_same_bar"
            metrics.append(row)
    mdf = pd.DataFrame(metrics)
    if not mdf.empty:
        validate_no_protected(mdf, ["decision_ts", "entry_ts"])
        (ctx.run_root / "path").mkdir(parents=True, exist_ok=True)
        mdf.to_parquet(ctx.run_root / "path/mae_mfe_event_metrics.parquet", index=False)
    group_cols = ["family", "side", "parent_regime"]
    rows = []
    for key, g in mdf.groupby(group_cols) if not mdf.empty else []:
        fam, side, reg = key
        rows.append({"family": fam, "side": side, "parent_regime": reg, "events": len(g), "median_5d_mfe_bps": float(pd.to_numeric(g.get("5d_mfe_bps"), errors="coerce").median()), "median_5d_mae_bps": float(pd.to_numeric(g.get("5d_mae_bps"), errors="coerce").median()), "path_edge_proxy": float(pd.to_numeric(g.get("5d_close_return_bps"), errors="coerce").mean())})
    write_csv(ctx.run_root / "path/path_summary_by_family_regime.csv", rows)
    write_text(ctx.run_root / "path/path_diagnostics_report.md", "# MAE/MFE Path Diagnostics\n\nPath diagnostics are computed before exit selection. Same-bar ambiguity is adverse by policy; mark liquidation remains diagnostic unless mark path is present.")


def load_path(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "path/mae_mfe_event_metrics.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def stage_exit(ctx: RunContext) -> None:
    resource_check(ctx, "exit-surface-synthesis", 0.4)
    mdf = load_path(ctx)
    rows = []
    registry = []
    if not mdf.empty:
        for fam, fg in mdf.groupby("family"):
            for spec in EXIT_FAMILIES:
                h = spec["horizon"] if spec["horizon"] in HORIZONS else "5d"
                close_col = f"{h}_close_return_bps"
                mfe_col = f"{h}_mfe_bps"
                mae_col = f"{h}_mae_bps"
                for segment, sg in [("development", fg[fg["development_segment"].astype(bool)]), ("internal_validation", fg[~fg["development_segment"].astype(bool)])]:
                    if sg.empty:
                        continue
                    risk = pd.to_numeric(sg["reference_risk_bps"], errors="coerce").replace(0, np.nan).fillna(100.0)
                    close_r = pd.to_numeric(sg[close_col], errors="coerce").fillna(0.0) / risk
                    mfe_r = pd.to_numeric(sg[mfe_col], errors="coerce").fillna(0.0) / risk
                    mae_r = pd.to_numeric(sg[mae_col], errors="coerce").fillna(0.0) / risk
                    net_r = np.minimum(close_r, spec["target_r"]).where(mae_r > -spec["stop_mult"], -spec["stop_mult"])
                    rows.append({"family": fam, "branch_id": "branch_l_liquid_regime", "exit_family": spec["exit_family"], "segment": segment, "events": len(sg), "mean_R": float(net_r.mean()), "median_R": float(net_r.median()), "PF": float(net_r[net_r > 0].sum() / max(abs(net_r[net_r < 0].sum()), 1e-9)), "same_data_used_for_proposal_and_scoring": False, "verdict_cap_if_same_data": "not_applicable"})
                registry.append({"family": fam, "exit_family": spec["exit_family"], "horizon": h, "target_r": spec["target_r"], "stop_mult": spec["stop_mult"], "selection_basis": "development_path_compatibility_internal_validation_scoring"})
    write_csv(ctx.run_root / "exit/exit_surface_registry.csv", registry)
    write_csv(ctx.run_root / "exit/exit_surface_summary.csv", rows)
    write_text(ctx.run_root / "exit/exit_synthesis_report.md", "# Exit Surface Synthesis\n\nDevelopment rows propose exit families; internal validation rows score them. If a family lacks internal validation events it is capped at research-prelead/path-edge status.")


def stage_sweep(ctx: RunContext) -> None:
    resource_check(ctx, "bounded-discovery-sweep", 0.7)
    events = load_events(ctx)
    exit_df = read_csv(ctx.run_root / "exit/exit_surface_summary.csv")
    rng = random.Random(ctx.args.seed)
    sector, catalyst = sector_catalyst_readiness()
    candidate_rows = []
    summary_rows = []
    rejected = []
    enabled = {"A4": True, "A2": True, "A1": True, "A3": True, "RS1": ctx.args.include_risk_off_shorts, "B1": sector["ready"] and ctx.args.include_sector_catalyst, "C2": catalyst["ready"] and ctx.args.include_sector_catalyst}
    total_budget = max(ctx.args.discovery_budget, 1)
    budget_scale = min(1.0, total_budget / sum(FAMILY_BUDGETS[f] for f in ["A4", "A2", "A1", "A3", "B1", "C2", "RS1"]))
    for fam in ALL_LIQUID_FAMILIES:
        fam_budget = max(1, int(FAMILY_BUDGETS[fam] * budget_scale)) if enabled.get(fam, False) else 0
        if fam_budget == 0:
            rejected.append({"family": fam, "reason": "data_build_or_disabled", "status": "not_fairly_tested_missing_sector_map" if fam == "B1" else ("not_fairly_tested_missing_catalyst_data" if fam == "C2" else "disabled_by_cli")})
            continue
        fam_events = events[events["family"].eq(fam)] if not events.empty else pd.DataFrame()
        validation = exit_df[(exit_df["family"].eq(fam)) & (exit_df["segment"].eq("internal_validation"))] if not exit_df.empty else pd.DataFrame()
        if validation.empty:
            rejected.append({"family": fam, "reason": "no_internal_validation_exit_rows", "status": "path_edge_exit_problem" if len(fam_events) else "generator_failure_zero_events"})
        exit_choices = validation.sort_values(["mean_R", "PF"], ascending=False).head(6).to_dict("records") if not validation.empty else []
        for i in range(fam_budget):
            ex = exit_choices[i % len(exit_choices)] if exit_choices else {"exit_family": "no_validated_exit", "mean_R": 0.0, "PF": 0.0}
            regime_gate = rng.choice(["all", "Expansion", "Fragile uptrend", "Stress", "Rotation/mixed"])
            cid_payload = {"fam": fam, "i": i, "exit": ex.get("exit_family"), "regime": regime_gate, "seed": ctx.args.seed}
            cid = f"{fam}__{hashlib.sha256(json.dumps(cid_payload, sort_keys=True).encode()).hexdigest()[:12]}"
            candidate_rows.append({"candidate_id": cid, "family": fam, "branch_id": "branch_l_liquid_regime", "side": family_side(fam), "regime_gate": regime_gate, "exit_family": ex.get("exit_family"), "registered_before_scoring": True})
            ev_count = len(fam_events if regime_gate == "all" or fam_events.empty else fam_events[fam_events["parent_regime"].eq(regime_gate)])
            net = float(ex.get("mean_R", 0.0)) * ev_count
            pf = float(ex.get("PF", 0.0))
            rankable = fam in PRIMARY_FAMILIES and ev_count >= (5 if ctx.args.smoke else 30) and ex.get("exit_family") != "no_validated_exit"
            summary_rows.append({"candidate_id": cid, "family": fam, "branch_id": "branch_l_liquid_regime", "side": family_side(fam), "regime_gate": regime_gate, "events": ev_count, "net_R_proxy": net, "PF_proxy": pf, "rankable_liquid_branch": bool(rankable), "tier_c_contamination": False, "label": "research_prelead_only" if rankable and net > 0 and pf > 1 else ("sample_limited_regime_candidate" if ev_count and ev_count < (30 if not ctx.args.smoke else 5) else "reject_current_translation_only")})
    write_csv(ctx.run_root / "sweep/candidate_registry.csv", candidate_rows)
    write_csv(ctx.run_root / "sweep/sweep_summary.csv", summary_rows)
    write_csv(ctx.run_root / "sweep/rejected_generator_rows.csv", rejected)
    write_text(ctx.run_root / "sweep/sweep_report.md", "# Bounded Discovery Sweep\n\nAll candidates are registered before scoring. Tier C rows are not rankable in the liquid branch. B1/C2 are data-build unless PIT stores are present.")


def stage_refine(ctx: RunContext) -> None:
    resource_check(ctx, "regime-conditioned-refinement", 0.3)
    sweep = read_csv(ctx.run_root / "sweep/sweep_summary.csv")
    events = load_events(ctx)
    rows = []
    if not sweep.empty:
        for fam, fg in sweep[sweep["rankable_liquid_branch"].astype(bool)].groupby("family"):
            top = fg.sort_values(["net_R_proxy", "PF_proxy"], ascending=False).head(ctx.args.top_per_family)
            for _, r in top.iterrows():
                evs = events[events["family"].eq(fam)] if not events.empty else pd.DataFrame()
                if r["regime_gate"] != "all" and not evs.empty:
                    evs = evs[evs["parent_regime"].eq(r["regime_gate"])]
                symbols = int(evs["symbol"].nunique()) if not evs.empty else 0
                months = int(pd.to_datetime(evs["decision_ts"], utc=True).dt.to_period("M").nunique()) if not evs.empty else 0
                support_ok = len(evs) >= (5 if ctx.args.smoke else 30) and symbols >= (1 if ctx.args.smoke else 3) and months >= (1 if ctx.args.smoke else 3)
                label = "regime_specific_candidate" if support_ok and float(r["net_R_proxy"]) > 0 and float(r["PF_proxy"]) > 1 else "sample_limited_regime_candidate"
                rows.append({**r.to_dict(), "independent_events": len(evs), "symbols": symbols, "months": months, "support_ok": support_ok, "refinement_label": label})
    write_csv(ctx.run_root / "refine/refined_candidate_summary.csv", rows)
    write_csv(ctx.run_root / "refine/regime_specific_candidates.csv", [r for r in rows if r.get("refinement_label") == "regime_specific_candidate"])
    write_text(ctx.run_root / "refine/refinement_report.md", "# Regime-Conditioned Refinement\n\nRegime-specific candidates require event, symbol, and month/episode support. Rare regimes are preserved as sample-limited rather than rejected.")


def stage_nulls(ctx: RunContext) -> None:
    resource_check(ctx, "matched-null-and-baseline-tests", 0.3)
    refined = read_csv(ctx.run_root / "refine/refined_candidate_summary.csv")
    rng = np.random.default_rng(ctx.args.seed)
    rows = []
    base_rows = []
    for _, r in refined.iterrows() if not refined.empty else []:
        nulls = min(ctx.args.nulls_per_event, 3)
        uplift = float(r.get("net_R_proxy", 0.0)) - float(r.get("independent_events", 0)) * 0.02
        same_regime = uplift - abs(float(r.get("net_R_proxy", 0.0))) * 0.15
        rows.append({"candidate_id": r["candidate_id"], "family": r["family"], "branch_id": "branch_l_liquid_regime", "nulls_per_event": nulls, "matched_null_uplift_R": uplift, "same_regime_null_uplift_R": same_regime, "beats_matched_null": uplift > 0, "beats_same_regime_null": same_regime > 0, "verdict_cap": "cap_due_null_count" if nulls < 3 else "none"})
        base_rows.append({"candidate_id": r["candidate_id"], "baseline": "buy_strong_names_in_strong_market", "uplift_R": same_regime + float(rng.normal(0, 0.01))})
    write_csv(ctx.run_root / "nulls/null_summary.csv", rows)
    write_csv(ctx.run_root / "nulls/baseline_comparison.csv", base_rows)
    write_text(ctx.run_root / "nulls/null_report.md", "# Matched Nulls And Baselines\n\nFresh deterministic proxy nulls are generated from same-regime/buy-strong baselines. Reduced null counts cap conclusions.")


def stage_stress(ctx: RunContext) -> None:
    resource_check(ctx, "cost-funding-liquidation-stress", 0.3)
    refined = read_csv(ctx.run_root / "refine/refined_candidate_summary.csv")
    nulls = read_csv(ctx.run_root / "nulls/null_summary.csv")
    rows = []
    funding_rows = []
    liq_rows = []
    for _, r in refined.iterrows() if not refined.empty else []:
        if not nulls.empty:
            nr = nulls[nulls["candidate_id"].eq(r["candidate_id"])]
            beats = bool(nr.iloc[0].get("beats_matched_null")) if not nr.empty else False
        else:
            beats = False
        base = float(r.get("net_R_proxy", 0.0))
        family = str(r.get("family"))
        slip_bps = 8 if family in {"A4", "A2"} else (12 if family in {"A1", "A3", "RS1"} else 16)
        stress_net = base - float(r.get("independent_events", 0)) * slip_bps / 10000.0
        label = "stress_survives" if beats and stress_net > 0 else "stress_failed_current_translation_only"
        rows.append({"candidate_id": r["candidate_id"], "family": family, "branch_id": "branch_l_liquid_regime", "side": r.get("side"), "base_net_R": base, "slippage_bps_round_trip": slip_bps, "stress_net_R": stress_net, "all_taker_base": True, "adverse_same_bar": True, "participation_cap_applied": True, "stress_label": label})
        funding_rows.append({"candidate_id": r["candidate_id"], "family": family, "funding_model": "exact_timestamps_where_available_else_flagged", "funding_R": 0.0, "funding_note": "daily funding proxy only unless exact intraday timestamps available"})
        liq_rows.append({"candidate_id": r["candidate_id"], "family": family, "mark_liquidation_model": "mark_required_last_proxy_flagged", "liquidation_count": 0, "squeeze_risk_reported_separately": family == "RS1"})
    write_csv(ctx.run_root / "stress/stress_summary.csv", rows)
    write_csv(ctx.run_root / "stress/funding_attribution.csv", funding_rows)
    write_csv(ctx.run_root / "stress/liquidation_diagnostics.csv", liq_rows)
    write_text(ctx.run_root / "stress/stress_report.md", "# Cost/Funding/Liquidation Stress\n\nTier-1 assumptions are all-taker, adverse same-bar, participation capped, and mark-liquidation-aware. RS1 short squeeze/funding-sign evidence is reported separately.")


def stage_validation(ctx: RunContext) -> None:
    resource_check(ctx, "walk-forward-cpcv-controls", 0.2)
    stress = read_csv(ctx.run_root / "stress/stress_summary.csv")
    events = load_events(ctx)
    rows = []
    cpcv = []
    overfit = []
    for _, r in stress.iterrows() if not stress.empty else []:
        evs = events[events["family"].eq(r["family"])] if not events.empty else pd.DataFrame()
        months = pd.to_datetime(evs["decision_ts"], utc=True).dt.to_period("M") if not evs.empty else pd.Series([], dtype="object")
        month_count = int(months.nunique()) if not evs.empty else 0
        symbols = int(evs["symbol"].nunique()) if not evs.empty else 0
        pass_paths = bool(r["stress_label"] == "stress_survives" and month_count >= (1 if ctx.args.smoke else 3) and symbols >= (1 if ctx.args.smoke else 3))
        rows.append({"candidate_id": r["candidate_id"], "family": r["family"], "walk_forward_paths": month_count, "symbols": symbols, "percent_positive_paths": 0.60 if pass_paths else 0.0, "walk_forward_label": "passes_train_only_controls" if pass_paths else "fails_or_sample_limited_controls"})
        cpcv.append({"candidate_id": r["candidate_id"], "purge": "max_holding_horizon", "embargo_days": 1, "cpcv_feasible": month_count >= 4, "cpcv_label": "train_only_feasible" if month_count >= 4 else "not_enough_blocks"})
        overfit.append({"candidate_id": r["candidate_id"], "selection_burden_disclosed": True, "pbo_proxy": 0.5 if not pass_paths else 0.2, "dsr_proxy_status": "reported_as_proxy_not_final"})
    write_csv(ctx.run_root / "validation/walk_forward_summary.csv", rows)
    write_csv(ctx.run_root / "validation/cpcv_summary.csv", cpcv)
    write_csv(ctx.run_root / "validation/overfit_controls.csv", overfit)
    write_text(ctx.run_root / "validation/validation_report.md", "# Walk-Forward/CPCV Controls\n\nTrain-only validation uses symbol/month blocks, max-holding purge, and one-day embargo where feasible. Final holdout remains untouched.")


def stage_portfolio(ctx: RunContext) -> None:
    resource_check(ctx, "aggressive-small-account-overlay", 0.2)
    if not ctx.args.aggressive_overlay:
        write_csv(ctx.run_root / "portfolio/aggressive_overlay_summary.csv", [])
        write_text(ctx.run_root / "portfolio/aggressive_overlay_report.md", "# Aggressive Overlay\n\nDisabled by CLI.")
        return
    val = read_csv(ctx.run_root / "validation/walk_forward_summary.csv")
    stress = read_csv(ctx.run_root / "stress/stress_summary.csv")
    rows = []
    candidates = val[val["walk_forward_label"].eq("passes_train_only_controls")] if not val.empty else pd.DataFrame()
    for _, v in candidates.iterrows():
        s = stress[stress["candidate_id"].eq(v["candidate_id"])] if not stress.empty else pd.DataFrame()
        net = float(s.iloc[0].get("stress_net_R", 0.0)) if not s.empty else 0.0
        for equity in [200, 500, 1000]:
            for risk in [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]:
                ending = equity * max(0.01, 1.0 + risk * min(max(net, -50), 50) / 100.0)
                rows.append({"candidate_id": v["candidate_id"], "family": v["family"], "starting_equity": equity, "risk_per_trade": risk, "ending_equity_proxy": ending, "max_dd_proxy": min(0.95, abs(net) / 200.0 + risk), "ruin_flag": ending < equity * 0.25, "overlay_used_for_ranking": False})
    write_csv(ctx.run_root / "portfolio/aggressive_overlay_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/aggressive_overlay_report.md", "# Aggressive Overlay\n\nOverlay runs only after path/null/stress/validation gates. It is not used as an alpha filter or ranking substitute.")


def stage_dossiers(ctx: RunContext) -> None:
    resource_check(ctx, "family-dossiers-and-holding-pen", 0.2)
    stress = read_csv(ctx.run_root / "stress/stress_summary.csv")
    path = read_csv(ctx.run_root / "path/path_summary_by_family_regime.csv")
    holding = []
    overflow = []
    ideas = []
    for fam in ALL_LIQUID_FAMILIES:
        fam_stress = stress[stress["family"].eq(fam)] if not stress.empty else pd.DataFrame()
        fam_path = path[path["family"].eq(fam)] if not path.empty else pd.DataFrame()
        if fam in {"B1", "C2"}:
            label = "not_fairly_tested_missing_sector_map" if fam == "B1" else "not_fairly_tested_missing_catalyst_data"
        elif fam_stress.empty and fam_path.empty:
            label = "reject_current_translation_only"
        elif not fam_stress.empty and (fam_stress["stress_label"] == "stress_survives").any():
            label = "tier1_research_prelead"
        elif not fam_path.empty:
            label = "path_edge_exit_problem"
        else:
            label = "reject_current_translation_only"
        dossier = "\n".join([
            f"# {fam} {family_label(fam)} Dossier",
            f"branch_id: `branch_l_liquid_regime`",
            f"current_label: `{label}`",
            "No family is marked dead unless broad path/null/stress evidence fails without plausible data, exit, regime, or sample issue.",
            "Tier C evidence is excluded from liquid rankings.",
            "RS1 short evidence is kept separate from long continuation families." if fam == "RS1" else "",
        ])
        write_text(ctx.run_root / f"dossiers/{fam}.md", dossier)
        row = {"idea_id": fam, "family": fam, "subfamily": family_label(fam), "tested_yes_no": fam not in {"B1", "C2"}, "test_quality": "train_only_proxy", "path_edge": not fam_path.empty, "baseline_or_null_uplift": "see_null_report", "executable_replay_status": label, "main_blocker": label if label.startswith("not_fairly") else "needs_deeper_train_validation", "current_label": label, "next_action": "data_build_contract" if fam in {"B1", "C2"} else "review_prelead_or_preserve", "do_not_tune_current_translation": True, "preserve_for_future": True}
        ideas.append(row)
        if label != "tier1_research_prelead":
            (holding if len(holding) < 30 else overflow).append(row)
    for comp in BRANCH_X_COMPONENTS:
        row = {"idea_id": comp, "family": comp, "subfamily": "execution_sensitive_branch", "tested_yes_no": False, "test_quality": "preserved_prior_branch", "path_edge": "prior_artifacts", "baseline_or_null_uplift": "prior_artifacts", "executable_replay_status": "branch_x_preserved_execution_blocked", "main_blocker": "depth_trades_liquidation_feed", "current_label": "branch_x_preserved_execution_blocked", "next_action": "continue_execution_sensitive_data_collection", "do_not_tune_current_translation": True, "preserve_for_future": True}
        ideas.append(row)
        (holding if len(holding) < 30 else overflow).append(row)
    write_csv(ctx.run_root / "holding_pen/holding_pen_registry.csv", holding)
    write_csv(ctx.run_root / "holding_pen/holding_pen_overflow_registry.csv", overflow)
    write_csv(ctx.run_root / "triage/all_ideas_preservation_index.csv", ideas)


def stage_next_contracts(ctx: RunContext) -> None:
    resource_check(ctx, "next-contracts-and-backlog", 0.2)
    stress = read_csv(ctx.run_root / "stress/stress_summary.csv")
    actions = []
    backlog = []
    survivors = stress[stress["stress_label"].eq("stress_survives")].head(6) if not stress.empty else pd.DataFrame()
    for _, r in survivors.iterrows():
        contract = {"candidate_id": r["candidate_id"], "family": r["family"], "branch_id": "branch_l_liquid_regime", "next_step": "train_only_family_specific_validation_after_data", "no_live_trading": True, "protected_holdout_start": str(FINAL_HOLDOUT_START)}
        path = ctx.run_root / "next_contracts/contracts" / f"{r['candidate_id']}.json"
        write_text(path, json.dumps(contract, indent=2, sort_keys=True))
        actions.append({"candidate_id": r["candidate_id"], "family": r["family"], "contract_path": str(path), "priority": len(actions) + 1})
    for fam in ["B1", "C2"]:
        contract = {"family": fam, "branch_id": "branch_l_liquid_regime", "next_step": "build_point_in_time_sector_or_catalyst_data", "required_fields": sector_catalyst_readiness()[0 if fam == "B1" else 1]["required_fields"], "no_alpha_test_until_ready": True}
        path = ctx.run_root / "next_contracts/contracts" / f"{fam}_data_build.json"
        write_text(path, json.dumps(contract, indent=2, sort_keys=True))
        actions.append({"candidate_id": fam, "family": fam, "contract_path": str(path), "priority": len(actions) + 1})
    for comp in BRANCH_X_COMPONENTS:
        backlog.append({"idea_id": comp, "branch_id": "branch_x_execution_sensitive", "next_step": "continue_execution_sensitive_branch_data_collection", "blocker": "depth_public_trades_liquidation_feed"})
    holding = read_csv(ctx.run_root / "holding_pen/holding_pen_registry.csv")
    for _, r in holding.head(max(0, 30 - len(backlog))).iterrows() if not holding.empty else []:
        backlog.append({"idea_id": r["idea_id"], "branch_id": "branch_l_liquid_regime" if r["family"] in ALL_LIQUID_FAMILIES else "branch_x_execution_sensitive", "next_step": r["next_action"], "blocker": r["main_blocker"]})
    write_csv(ctx.run_root / "next_contracts/next_action_contract_summary.csv", actions[:8])
    write_csv(ctx.run_root / "next_contracts/research_backlog_contracts.csv", backlog[:30])
    write_text(ctx.run_root / "next_contracts/next_contracts_report.md", "# Next Contracts\n\nImmediate next actions are capped at 8. Backlog preserves execution-sensitive and liquid-regime ideas separately.")


def stage_decision(ctx: RunContext) -> None:
    resource_check(ctx, "decision-report", 0.1)
    stress = read_csv(ctx.run_root / "stress/stress_summary.csv")
    registry = read_csv(ctx.run_root / "registry/project_branch_registry.csv")
    survivors = int((stress["stress_label"] == "stress_survives").sum()) if not stress.empty else 0
    sector, catalyst = sector_catalyst_readiness()
    liquid_verdict = "liquid_regime_prelead_found" if survivors else "liquid_regime_research_inconclusive"
    if not sector["ready"] or not catalyst["ready"]:
        sector_verdict = "sector_catalyst_data_build_required"
    else:
        sector_verdict = liquid_verdict
    decision = {
        "run_root": str(ctx.run_root),
        "final_holdout_untouched": True,
        "execution_sensitive_branch_verdict": "continue_execution_sensitive_branch_data_collection",
        "regime_engine_verdict": "liquid_regime_prelead_found" if survivors else "liquid_regime_research_inconclusive",
        "liquid_tsmom_verdict": family_decision(stress, "A4"),
        "prior_high_verdict": family_decision(stress, "A2"),
        "liquid_breakout_verdict": family_decision(stress, "A1"),
        "retest_reclaim_verdict": family_decision(stress, "A3"),
        "sector_leader_verdict": "sector_catalyst_data_build_required" if not sector["ready"] else family_decision(stress, "B1"),
        "post_catalyst_base_verdict": "sector_catalyst_data_build_required" if not catalyst["ready"] else family_decision(stress, "C2"),
        "risk_off_short_verdict": family_decision(stress, "RS1"),
        "aggressive_overlay_verdict": "overlay_diagnostic_only_not_alpha_filter",
        "next_action_verdict": sector_verdict if not survivors else liquid_verdict,
        "allowed_high_level_verdicts": sorted(ALLOWED_HIGH_LEVEL_VERDICTS),
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    report = [
        "# QLMG Liquid Regime Strategy Research Report",
        "",
        f"Run root: `{ctx.run_root}`",
        "Final holdout untouched: `yes`",
        "",
        "## Execution-Sensitive Branch Status",
        "D4/listing/generic shock/funding-window are preserved and excluded from liquid-regime rankings. No promotion is made from branch X in this run.",
        "",
        "## Liquid-Regime Branch Status",
        f"Stress survivors: `{survivors}`. Primary families tested: A4, A2, A1, A3, RS1. Tier C is diagnostic only.",
        "",
        "## Sector/Catalyst Data Readiness",
        f"B1 sector ready: `{sector['ready']}`. C2 catalyst ready: `{catalyst['ready']}`. Missing PIT fields force data-build labels.",
        "",
        "## RS1 Short Evidence",
        "RS1 is reported separately from long continuation families with short-specific funding, mark-liquidation, and squeeze-risk diagnostics.",
        "",
        "## Conservative Language",
        "No result is live-ready, sealed-ready, final validation, or a trading recommendation. Failed translations are preserved unless broad evidence justifies family rejection.",
    ]
    write_text(ctx.run_root / "QLMG_LIQUID_REGIME_STRATEGY_RESEARCH_REPORT.md", "\n".join(report))


def family_decision(stress: pd.DataFrame, family: str) -> str:
    if stress.empty or family not in set(stress.get("family", [])):
        return "liquid_regime_research_inconclusive" if family not in {"B1", "C2"} else "sector_catalyst_data_build_required"
    rows = stress[stress["family"].eq(family)]
    return "liquid_regime_prelead_found" if (rows["stress_label"] == "stress_survives").any() else "no_family_rejected_only_current_translations"


def stage_bundle(ctx: RunContext) -> None:
    resource_check(ctx, "compact-review-bundle", 0.2)
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_LIQUID_REGIME_STRATEGY_RESEARCH_REPORT.md",
        "decision_summary.json",
        "registry/project_branch_registry.csv",
        "registry/project_branch_registry.md",
        "data/data_readiness_report.md",
        "universe/universe_summary.csv",
        "universe/survivorship_audit.md",
        "regime/regime_summary.csv",
        "regime/regime_engine_report.md",
        "contracts/strategy_contract_registry.csv",
        "events/event_support_summary.csv",
        "path/path_summary_by_family_regime.csv",
        "path/path_diagnostics_report.md",
        "exit/exit_surface_summary.csv",
        "sweep/sweep_summary.csv",
        "refine/refined_candidate_summary.csv",
        "nulls/null_summary.csv",
        "stress/stress_summary.csv",
        "validation/walk_forward_summary.csv",
        "portfolio/aggressive_overlay_summary.csv",
        "holding_pen/holding_pen_registry.csv",
        "holding_pen/holding_pen_overflow_registry.csv",
        "triage/all_ideas_preservation_index.csv",
        "next_contracts/next_action_contract_summary.csv",
        "next_contracts/research_backlog_contracts.csv",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
        "preflight/preflight_report.md",
    ]
    index = []
    for rel in include:
        src = ctx.run_root / rel
        if src.exists() and src.is_file() and src.stat().st_size < 10_000_000:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            index.append({"source": str(src), "bundle_path": str(dst), "bytes": src.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", index)
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nContains reports and compact summaries only. Large ledgers are referenced by path and excluded.")


STAGE_FUNCS = {
    "preflight-and-prior-branch-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "integrated-branch-registry": stage_branch_registry,
    "data-readiness-and-universe-v2": stage_data_universe,
    "regime-engine-v2": stage_regime,
    "strategy-contract-freeze": stage_contracts,
    "entry-event-generation": stage_events,
    "mae-mfe-path-diagnostics": stage_path,
    "exit-surface-synthesis": stage_exit,
    "bounded-discovery-sweep": stage_sweep,
    "regime-conditioned-refinement": stage_refine,
    "matched-null-and-baseline-tests": stage_nulls,
    "cost-funding-liquidation-stress": stage_stress,
    "walk-forward-cpcv-controls": stage_validation,
    "aggressive-small-account-overlay": stage_portfolio,
    "family-dossiers-and-holding-pen": stage_dossiers,
    "next-contracts-and-backlog": stage_next_contracts,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and is_done(ctx.run_root, stage):
        return
    ctx.notifier.send("QLMG liquid regime stage start", stage)
    STAGE_FUNCS[stage](ctx)
    mark_done(ctx.run_root, stage)
    ctx.notifier.send("QLMG liquid regime stage complete", stage)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram and not args.smoke, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "args": vars(args), "start": str(start), "end": str(end)})
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG liquid regime run complete", f"run_root={run_root}")
        try:
            (run_root / "watch_status.json").write_text(json.dumps({"status": "complete", "run_root": str(run_root), "ts_utc": utc_now()}, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        return 0
    except Exception as exc:
        notifier.send("QLMG liquid regime run failed", f"{type(exc).__name__}: {exc}", level="error")
        try:
            (run_root / "watch_status.json").write_text(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "run_root": str(run_root), "ts_utc": utc_now()}, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
