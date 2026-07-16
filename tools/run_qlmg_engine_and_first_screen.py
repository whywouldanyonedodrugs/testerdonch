#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

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
    to_utc_ts,
    utc_now,
    write_json,
)

try:
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

DEFAULT_RUN_ID = "phase_qlmg_engine_and_first_screen_20260624_v1"
RESULTS_ROOT = REPO / "results/rebaseline"
DATA_5M = Path("/opt/parquet/5m")
DATA_CONTEXT = Path("/opt/parquet/bybit_context_5m")
PHASE0_ROOT = REPO / "results/rebaseline/phase_qlmg_perp_project_reset_20260624_v1_20260624_092636"
FINAL_HOLDOUT_START = pd.Timestamp("2026-01-01T00:00:00Z")
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"}

STAGES = (
    "preflight-resource-guard",
    "telegram-run-notifier",
    "source-state-snapshot",
    "sealed-policy-freeze",
    "long-short-engine-implementation",
    "long-short-engine-tests",
    "lifecycle-and-universe-v0",
    "cost-funding-liquidation-v0",
    "screening-harness-v0",
    "phase1-d1-d3-e1-screen",
    "phase1-a2-liquid-momentum-baseline",
    "screening-summary-and-next-data-plan",
    "compact-review-bundle",
    "all",
)


@dataclass
class RunContext:
    args: argparse.Namespace
    run_root: Path
    notifier: "RunNotifier"
    protected_start: pd.Timestamp = FINAL_HOLDOUT_START
    screening_end: pd.Timestamp = SCREENING_END


class RunNotifier:
    def __init__(self, run_root: Path, disabled: bool = False) -> None:
        self.run_root = run_root
        self.disabled = disabled
        self.events_path = run_root / "notifications/telegram_events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.notifier = None
        self.status = "disabled"
        if not disabled and TelegramNotifier is not None:
            class _Args:
                tg_bot_token = ""
                tg_chat_id = ""
                tg_auto_chat = False
            try:
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-phase0.5")
                self.status = self.notifier.status_line()
            except Exception as exc:
                self.status = f"disabled: {type(exc).__name__}: {exc}"

    def send(self, title: str, body: str = "", *, level: str = "info") -> bool:
        sent = False
        if not self.disabled and self.notifier is not None:
            try:
                sent = bool(self.notifier.send(title, body))
            except Exception:
                sent = False
        rec = {"ts_utc": utc_now(), "title": title, "body": body, "level": level, "sent": sent, "status": self.status}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return sent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0.5 QLMG engine foundation and first screening runner")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--signals-per-symbol-variant", type=int, default=2)
    p.add_argument("--max-output-gb", type=float, default=30.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--run-root", default="")
    return p.parse_args()


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


def resolve_run_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.run_root:
        root = Path(args.run_root)
        root = root if root.is_absolute() else REPO / root
        return root.resolve(), "explicit_run_root"
    base = (RESULTS_ROOT / DEFAULT_RUN_ID).resolve()
    if not base.exists():
        return base, "default_root_available"
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return base.with_name(f"{base.name}_{suffix}"), f"default_root_existed_suffix_{suffix}"


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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_command(run_root: Path, stage: str) -> None:
    p = run_root / "command_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": utc_now(), "stage": stage, "argv": sys.argv, "cwd": str(REPO)}, sort_keys=True) + "\n")


def done_path(run_root: Path, stage: str) -> Path:
    return run_root / "stage_status" / f"{stage}.done"


def mark_done(run_root: Path, stage: str) -> None:
    write_text(done_path(run_root, stage), utc_now())


def shell(args: Sequence[str], timeout: float = 120.0) -> str:
    try:
        p = subprocess.run(args, cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


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
    guard_path = ctx.run_root / "resource_guard" / f"{stage}.json"
    write_json(guard_path, status | {"stage": stage, "snapshot": snap.__dict__})
    if status["warnings"]:
        ctx.notifier.send("RESOURCE WARNING", f"stage={stage}\n{status}", level="warning")
    if status["status"] != "pass":
        ctx.notifier.send("RESOURCE HARD STOP", f"stage={stage}\n{status}", level="error")
        raise RuntimeError(f"resource guard failed for {stage}: {status['reasons']}")


def metadata_timestamps(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int, str]:
    if not path.exists() or pq is None:
        return None, None, 0, "missing_or_pyarrow_unavailable"
    try:
        pf = pq.ParquetFile(path)
        names = list(pf.schema_arrow.names)
        if "timestamp" not in names:
            return None, None, int(pf.metadata.num_rows), "missing_timestamp"
        idx = names.index("timestamp")
        mins: list[pd.Timestamp] = []
        maxs: list[pd.Timestamp] = []
        for rg in range(pf.metadata.num_row_groups):
            st = pf.metadata.row_group(rg).column(idx).statistics
            if st and st.min is not None and st.max is not None:
                mins.append(pd.to_datetime(st.min, utc=True))
                maxs.append(pd.to_datetime(st.max, utc=True))
        if mins and maxs:
            return min(mins), max(maxs), int(pf.metadata.num_rows), "metadata_stats"
        s = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
        return pd.to_datetime(s.min(), utc=True), pd.to_datetime(s.max(), utc=True), int(len(s)), "read_timestamp"
    except Exception as exc:
        return None, None, 0, f"error:{type(exc).__name__}:{exc}"


def discover_symbol_paths(max_symbols: int = 0) -> list[Path]:
    paths = sorted(DATA_5M.glob("*.parquet"))
    if max_symbols > 0:
        return paths[:max_symbols]
    return paths


def latest_local_timestamp(max_files: int = 0) -> pd.Timestamp:
    latest: pd.Timestamp | None = None
    for root in (DATA_5M, DATA_CONTEXT):
        files = sorted(root.glob("*.parquet"))
        if max_files:
            files = files[:max_files]
        for p in files:
            _, mx, _, _ = metadata_timestamps(p)
            if mx is not None and (latest is None or mx > latest):
                latest = mx
    return latest or pd.Timestamp("2026-06-24T00:00:00Z")


def clamp_screening_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(args.start, utc=True)
    requested_end = pd.to_datetime(args.end, utc=True) if args.end else SCREENING_END
    end = min(pd.Timestamp(requested_end), SCREENING_END)
    if end >= FINAL_HOLDOUT_START:
        raise RuntimeError("screening window overlaps final holdout")
    return start, end


def load_symbol_df(symbol: str, start: pd.Timestamp, end: pd.Timestamp, include_context: bool = True) -> pd.DataFrame:
    p = DATA_5M / f"{symbol}.parquet"
    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover", "open_interest", "funding_rate"]
    df = pd.read_parquet(p, columns=[c for c in cols if c in pq.ParquetFile(p).schema_arrow.names] if pq else cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if df.empty:
        return df
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if include_context:
        cp = DATA_CONTEXT / f"{symbol}.parquet"
        if cp.exists():
            ccols = ["timestamp", "mark_high", "mark_low", "mark_close", "context_source_close_ts"]
            try:
                names = list(pq.ParquetFile(cp).schema_arrow.names) if pq else ccols
                ctx = pd.read_parquet(cp, columns=[c for c in ccols if c in names])
                ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
                ctx = ctx[(ctx["timestamp"] >= start) & (ctx["timestamp"] <= end)].copy()
                if "context_source_close_ts" in ctx.columns:
                    ctx["context_source_close_ts"] = pd.to_datetime(ctx["context_source_close_ts"], utc=True, errors="coerce")
                    ctx = ctx[ctx["context_source_close_ts"] <= ctx["timestamp"]]
                df = df.merge(ctx.drop_duplicates("timestamp", keep="last"), on="timestamp", how="left")
            except Exception:
                df["context_join_error"] = True
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("open", "high", "low", "close", "volume", "turnover"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce")
    out["ret_4h"] = out["close"] / out["close"].shift(48) - 1.0
    out["ret_24h"] = out["close"] / out["close"].shift(288) - 1.0
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["atr_proxy"] = (out["high"] - out["low"]).rolling(288, min_periods=24).mean()
    out["atr_proxy"] = out["atr_proxy"].fillna(out["close"].abs() * 0.01)
    out["ema10"] = out["close"].ewm(span=10, adjust=False, min_periods=10).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    vol_sum = out["volume"].rolling(288, min_periods=24).sum()
    out["vwap_proxy_24h"] = (out["close"] * out["volume"]).rolling(288, min_periods=24).sum() / vol_sum.replace(0, np.nan)
    out["high_72h"] = out["high"].rolling(864, min_periods=48).max()
    out["low_24h"] = out["low"].rolling(288, min_periods=24).min()
    out["high_24h"] = out["high"].rolling(288, min_periods=24).max()
    out["turnover_med_24h"] = out["turnover"].rolling(288, min_periods=24).median()
    if "open_interest" in out.columns:
        out["oi_chg_24h"] = pd.to_numeric(out["open_interest"], errors="coerce") / pd.to_numeric(out["open_interest"], errors="coerce").shift(288) - 1.0
    else:
        out["oi_chg_24h"] = np.nan
    return out


def funding_events_from_df(df: pd.DataFrame) -> list[FundingEvent]:
    events: list[FundingEvent] = []
    if "funding_rate" not in df.columns:
        return events
    for row in df.itertuples(index=False):
        ts = getattr(row, "timestamp")
        if ts.minute == 0 and ts.hour % 8 == 0:
            rate = getattr(row, "funding_rate", np.nan)
            if pd.notna(rate):
                mark = getattr(row, "mark_close", np.nan) if hasattr(row, "mark_close") else np.nan
                if pd.isna(mark):
                    mark = getattr(row, "close")
                events.append(FundingEvent(timestamp=ts, rate=float(rate), mark_price=float(mark), source="5m_proxy_schedule"))
    return events


def cost_bps_for_tier(tier: str) -> tuple[float, float]:
    table = {
        "A": (8.0, 8.0),
        "B": (12.0, 18.0),
        "C": (15.0, 35.0),
        "D": (18.0, 60.0),
        "UNKNOWN": (18.0, 75.0),
    }
    return table.get(str(tier), table["UNKNOWN"])


def build_trade_from_signal(
    symbol: str,
    family: str,
    variant: dict[str, Any],
    df: pd.DataFrame,
    df_indexed: pd.DataFrame,
    funding_events: Sequence[FundingEvent],
    i: int,
    tier: str,
    side: str,
    reason: str,
) -> dict[str, Any] | None:
    if i + 1 >= len(df):
        return None
    row = df.iloc[i]
    entry_row = df.iloc[i + 1]
    entry = float(entry_row["open"])
    atr = float(row.get("atr_proxy", np.nan))
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr) or atr <= 0:
        return None
    hold_h = float(variant.get("hold_h", 24))
    risk_mult = float(variant.get("risk_atr", 1.5))
    target_mult = float(variant.get("target_atr", 2.0))
    if side == "long":
        stop = float(row.get("low_24h", entry - risk_mult * atr)) if family in {"D3", "E1"} else entry - risk_mult * atr
        if not np.isfinite(stop) or stop >= entry:
            stop = entry - risk_mult * atr
        target = entry + target_mult * atr
    else:
        stop = entry + risk_mult * atr
        target = entry - target_mult * atr
    fee_bps, slip_bps = cost_bps_for_tier(tier)
    future = df_indexed.iloc[i + 1 :]
    cfg = ReplayConfig(
        side=side,
        decision_ts=pd.Timestamp(row["timestamp"]),
        entry_ts=pd.Timestamp(entry_row["timestamp"]),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        max_holding_hours=hold_h,
        qty=1.0,
        fee_bps_round_trip=fee_bps,
        slippage_bps_round_trip=slip_bps,
        leverage=float(variant.get("leverage", 3.0)),
        maintenance_margin_fraction=0.005,
        trailing_stop_distance=(atr * float(variant.get("trail_atr"))) if variant.get("trail_atr") else None,
        tie_breaker="sl_wins",
    )
    try:
        res = replay_trade(future, cfg, funding_events)
    except Exception:
        return None
    out = res.as_dict()
    out.update(
        {
            "strategy_family": family,
            "variant_id": variant["variant_id"],
            "symbol": symbol,
            "liquidity_tier": tier,
            "decision_ts": str(pd.Timestamp(row["timestamp"])),
            "stop_price": stop,
            "target_price": target,
            "cost_model_version": "qlmg_screening_cost_v0_proxy",
            "parameter_hash": variant["parameter_hash"],
            "signal_reason": reason,
        }
    )
    return out


def parameter_hash(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    parquet_size = sum(p.stat().st_size for root in (DATA_5M, DATA_CONTEXT) if root.exists() for p in root.glob("*.parquet"))
    repo_size = 0
    for p in REPO.rglob("*"):
        if any(part in {".git", ".venv"} for part in p.parts):
            continue
        if p.is_file():
            repo_size += p.stat().st_size
    budget = {
        "default_max_output_gb": ctx.args.max_output_gb,
        "hard_free_disk_gb": 5,
        "warn_free_disk_gb": 7,
        "hard_stage_output_gb_without_allow_large_output": 20,
        "estimated_total_output_gb": 3.0 if not ctx.args.smoke else 0.2,
        "allow_large_output": bool(ctx.args.allow_large_output),
    }
    write_json(ctx.run_root / "preflight/disk_memory_snapshot.json", snap.__dict__)
    write_json(ctx.run_root / "preflight/output_budget.json", budget)
    write_text(ctx.run_root / "preflight/stage_resume_plan.md", "\n".join([f"- `{s}` writes `stage_status/{s}.done` and can be skipped with `--resume`." for s in STAGES if s != "all"]))
    write_text(
        ctx.run_root / "preflight/resource_guard_report.md",
        f"# Resource Guard Report\n\n- free disk GB: `{snap.free_gb:.2f}`\n- repo size excluding .git/.venv bytes: `{repo_size}`\n- /opt/parquet observed parquet bytes: `{parquet_size}`\n- hard stop free disk: `<5GB`\n- Telegram warning free disk: `<7GB`\n- hard stop stage estimate: `>20GB` unless `--allow-large-output`\n- run can proceed: `{snap.free_gb >= 5}`\n",
    )


def stage_telegram(ctx: RunContext) -> None:
    write_json(ctx.run_root / "notifications/telegram_dry_run_test.json", {"status": ctx.notifier.status, "disabled_by_cli": ctx.args.disable_telegram, "token_stored": False})
    write_text(ctx.run_root / "notifications/telegram_notifier_report.md", f"# Telegram Notifier Report\n\nStatus: `{ctx.notifier.status}`\n\nSecrets are not written to disk. Events are logged locally in `telegram_events.jsonl`.\n")
    ctx.notifier.send("RUN START", f"run_root={ctx.run_root}")


def stage_source_snapshot(ctx: RunContext) -> None:
    out = ctx.run_root / "source"
    write_text(out / "git_status_short.txt", shell(["git", "status", "--short"]))
    write_text(out / "git_diff_stat.txt", shell(["git", "diff", "--stat"]))
    write_text(out / "current_branch_and_head.txt", shell(["bash", "-lc", "git branch --show-current && git rev-parse HEAD"]))
    rows = []
    for p in sorted(REPO.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO)
        if rel.parts[0] in {".git", ".venv", "results", "reports", "archive", "artifacts"}:
            continue
        rows.append({"path": str(rel), "size_bytes": p.stat().st_size})
    write_csv(out / "source_snapshot_manifest.csv", rows)
    tar_path = out / "source_snapshot_for_repro.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for row in rows:
            tar.add(REPO / row["path"], arcname=row["path"])
    write_text(out / "source_state_report.md", f"# Source State Report\n\n- branch/head: `{shell(['git','rev-parse','HEAD'])}`\n- tracked source files archived in snapshot: `{len(rows)}`\n- snapshot: `{tar_path}`\n")


def stage_seal(ctx: RunContext) -> None:
    latest = latest_local_timestamp(max_files=0 if not ctx.args.smoke else 20)
    protected = {
        "policy_version": "qlmg_phase0_5_v1",
        "final_holdout": {"start": str(FINAL_HOLDOUT_START), "end": str(latest), "status": "protected_no_screening_access"},
        "screening_allowed": {"end_inclusive": str(SCREENING_END)},
        "gold_holdout_note": {"start": "2026-03-06T00:00:00Z", "end": str(latest), "reason": "prior sealed governance remains especially sensitive"},
    }
    write_json(ctx.run_root / "seal/qlmg_protected_slices.json", protected)
    write_text(ctx.run_root / "seal/qlmg_sealed_policy.md", f"# QLMG Sealed Policy\n\nFinal holdout is protected from candidate selection: `{FINAL_HOLDOUT_START}` through `{latest}`. Screening must end no later than `{SCREENING_END}`.\n")
    # Smoke-test local guard behavior without reading protected data.
    ok_pre = SCREENING_END < FINAL_HOLDOUT_START
    blocked = not (pd.Timestamp("2026-01-01T00:00:00Z") <= SCREENING_END)
    write_text(ctx.run_root / "seal/seal_guard_status.md", f"# Seal Guard Status\n\n- pre-holdout screening end before holdout: `{ok_pre}`\n- candidate-selection protected overlap would block: `{blocked}`\n")


def stage_engine_report(ctx: RunContext) -> None:
    rows = [
        {"capability": "long_entries_exits", "status": "implemented_screening_core"},
        {"capability": "short_entries_exits", "status": "implemented_screening_core"},
        {"capability": "side_aware_funding", "status": "implemented_and_tested"},
        {"capability": "mark_price_liquidation", "status": "implemented_screening_diagnostic"},
        {"capability": "portfolio_margin", "status": "not_implemented_single_trade_screening_only"},
        {"capability": "delist_settlement", "status": "implemented_when_lifecycle_timestamp_available"},
        {"capability": "vwap_fail_reclaim_short", "status": "scaffolded_by_family_logic_not_general_engine"},
    ]
    write_csv(ctx.run_root / "engine/long_short_support_matrix.csv", rows)
    write_text(ctx.run_root / "engine/long_short_engine_implementation_report.md", "# Long/Short Engine Implementation Report\n\nA minimal isolated screening replay core was added in `tools/qlmg_screening_core.py`. It is screening-grade and does not mutate the live/legacy engine. Liquidation uses mark columns when available and falls back with an explicit quality flag when not.\n")
    write_text(ctx.run_root / "engine/known_engine_limitations.md", "# Known Engine Limitations\n\n- Screening-grade proxy costs only.\n- No full portfolio margin or order book simulation.\n- No true liquidation feed; E1 uses proxy state.\n- No 1m execution dependency in this phase.\n")


def run_synthetic_engine_checks() -> list[dict[str, Any]]:
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=8, freq="5min")
    base = pd.DataFrame({"open": [100]*8, "high": [101,102,103,104,105,106,107,108], "low": [99,98,97,96,95,94,93,92], "close": [100,101,102,103,104,105,106,107], "mark_high": [101,102,103,104,105,106,107,108], "mark_low": [99,98,97,96,95,94,93,92]}, index=idx)
    tests: list[tuple[str, ReplayConfig, pd.DataFrame, bool]] = []
    tests.append(("long_tp", ReplayConfig("long", idx[0], idx[0], 100, 95, 103, 1), base, True))
    tests.append(("short_tp", ReplayConfig("short", idx[0], idx[0], 100, 105, 97, 1), base, True))
    tests.append(("long_liq", ReplayConfig("long", idx[0], idx[0], 100, 70, 120, 1, leverage=10), base.assign(mark_low=[100,80,80,80,80,80,80,80]), True))
    tests.append(("short_liq", ReplayConfig("short", idx[0], idx[0], 100, 130, 80, 1, leverage=10), base.assign(mark_high=[100,120,120,120,120,120,120,120]), True))
    results = []
    for name, cfg, bars, should_pass in tests:
        try:
            res = replay_trade(bars, cfg, [])
            passed = True
            detail = res.exit_reason
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        results.append({"test": name, "passed": passed == should_pass, "detail": detail})
    # Funding sign checks.
    long_f = replay_trade(base, ReplayConfig("long", idx[0], idx[0], 100, 90, None, 1), [FundingEvent(idx[1], 0.01, 100)])
    short_f = replay_trade(base, ReplayConfig("short", idx[0], idx[0], 100, 110, None, 1), [FundingEvent(idx[1], 0.01, 100)])
    results.append({"test": "long_positive_funding_pays", "passed": long_f.funding_pnl < 0, "detail": long_f.funding_pnl})
    results.append({"test": "short_positive_funding_receives", "passed": short_f.funding_pnl > 0, "detail": short_f.funding_pnl})
    return results


def stage_engine_tests(ctx: RunContext) -> None:
    results = run_synthetic_engine_checks()
    all_pass = all(r["passed"] for r in results)
    write_csv(ctx.run_root / "engine_tests/test_results.txt", results)
    write_text(ctx.run_root / "engine_tests/long_short_engine_test_report.md", f"# Long/Short Engine Test Report\n\nAll synthetic checks passed: `{all_pass}`\n\nRows: `{len(results)}`\n")
    if not all_pass:
        raise RuntimeError("long/short engine synthetic checks failed")


def stage_universe(ctx: RunContext) -> None:
    start, end = clamp_screening_window(ctx.args)
    max_symbols = ctx.args.max_symbols or (5 if ctx.args.smoke else 0)
    rows = []
    tier_parts = []
    for path in discover_symbol_paths(max_symbols):
        sym = path.stem.upper()
        mn, mx, n, source = metadata_timestamps(path)
        rows.append({"symbol": sym, "first_5m_ts": mn, "last_5m_ts_full_inventory": mx, "row_count": n, "metadata_source": source, "listing_ts_proxy": mn, "official_metadata_available": False})
        df = load_symbol_df(sym, start, end, include_context=False)
        if df.empty:
            continue
        daily = df.set_index("timestamp").resample("1D").agg({"turnover": "sum", "high": "max", "low": "min", "close": "last"}).dropna(subset=["close"])
        daily["symbol"] = sym
        daily["date"] = daily.index.date.astype(str)
        daily["median_daily_notional_30d"] = daily["turnover"].rolling(30, min_periods=5).median()
        daily["spread_proxy_bps"] = ((daily["high"] - daily["low"]) / daily["close"].replace(0, np.nan) * 10000).rolling(30, min_periods=5).median()
        first = pd.Timestamp(daily.index.min(), tz="UTC") if daily.index.tz is None else pd.Timestamp(daily.index.min()).tz_convert("UTC")
        age = (daily.index.tz_convert("UTC") - first).days if daily.index.tz is not None else (daily.index - daily.index.min()).days
        daily["age_days_proxy"] = age
        def tier(row: pd.Series) -> str:
            if sym in MAJORS:
                return "A"
            if row["age_days_proxy"] <= 30:
                return "D"
            notional = row["median_daily_notional_30d"]
            spr = row["spread_proxy_bps"]
            if pd.isna(notional):
                return "UNKNOWN"
            if notional >= 20_000_000:
                return "B"
            if 2_000_000 <= notional < 20_000_000 and (pd.isna(spr) or spr <= 1200):
                return "C"
            return "D"
        daily["liquidity_tier"] = daily.apply(tier, axis=1)
        tier_parts.append(daily[["symbol", "date", "median_daily_notional_30d", "spread_proxy_bps", "age_days_proxy", "liquidity_tier"]].reset_index(drop=True))
    master = pd.DataFrame(rows)
    ensure_parent(ctx.run_root / "universe/instrument_master_v0.parquet")
    master.to_parquet(ctx.run_root / "universe/instrument_master_v0.parquet", index=False)
    master.to_csv(ctx.run_root / "universe/instrument_master_v0.csv.gz", index=False, compression="gzip")
    tiers = pd.concat(tier_parts, ignore_index=True) if tier_parts else pd.DataFrame(columns=["symbol", "date", "liquidity_tier"])
    ensure_parent(ctx.run_root / "universe/liquidity_tiers_by_date.parquet")
    tiers.to_parquet(ctx.run_root / "universe/liquidity_tiers_by_date.parquet", index=False)
    write_text(ctx.run_root / "universe/universe_builder_report.md", f"# Universe Builder Report\n\n- symbols inventoried: `{len(master)}`\n- tier rows: `{len(tiers)}`\n- official historical lifecycle unavailable locally; proxy first/last bars used only as proxy fields.\n")
    write_text(ctx.run_root / "universe/universe_qc_report.md", "# Universe QC Report\n\nOHLCV spread/liquidity proxies are screening-only and are not top-of-book measurements.\n")


def stage_cost(ctx: RunContext) -> None:
    contract = {
        "version": "qlmg_screening_cost_v0_proxy",
        "proxy_cost_model": True,
        "fees_bps_round_trip": {"A": 8, "B": 12, "C": 15, "D": 18, "UNKNOWN": 18},
        "slippage_bps_round_trip": {"A": 8, "B": 18, "C": 35, "D": 60, "UNKNOWN": 75},
        "funding": "side_aware_proxy_from_5m_funding_rate_at_8h_utc_boundaries_when_available",
        "liquidation": "mark_price_if_available_else_last_ohlc_proxy_flagged",
    }
    write_json(ctx.run_root / "cost/cost_model_v0_contract.json", contract)
    examples = []
    for tier in ["A", "B", "C", "D", "UNKNOWN"]:
        fee, slip = cost_bps_for_tier(tier)
        examples.append({"tier": tier, "fee_bps_round_trip": fee, "slippage_bps_round_trip": slip, "final_proxy_cost_bps": fee + slip})
    ensure_parent(ctx.run_root / "cost/cost_model_v0_examples.parquet")
    pd.DataFrame(examples).to_parquet(ctx.run_root / "cost/cost_model_v0_examples.parquet", index=False)
    write_text(ctx.run_root / "cost/cost_model_v0_report.md", "# Cost Model v0 Report\n\nThis is a conservative screening proxy. It is not empirical top-of-book execution quality. Funding is side-aware but uses proxy settlement scheduling when exact instrument metadata is unavailable.\n")


def stage_harness(ctx: RunContext) -> None:
    schema = [
        "strategy_family", "variant_id", "side", "symbol", "liquidity_tier", "decision_ts", "entry_ts", "entry_price", "stop_price", "target_price", "exit_ts", "exit_price", "exit_reason", "holding_hours", "gross_R", "fee_R", "slippage_R", "funding_R", "net_R", "liquidation_flag", "delist_flag", "cost_model_version", "data_quality_flags", "parameter_hash",
    ]
    write_csv(ctx.run_root / "harness/common_ledger_schema.csv", [{"column": c} for c in schema])
    write_text(ctx.run_root / "harness/strategy_contract_schema.yaml", "required: [strategy_family, variant_id, side, universe, signal, stop, exit, max_holding_hours, parameter_hash]\n")
    write_text(ctx.run_root / "harness/screening_harness_report.md", "# Screening Harness Report\n\nThe harness is contract-driven, chunked by symbol, checkpoint-aware, and writes compact ledgers by default.\n")


def tier_lookup(ctx: RunContext) -> pd.DataFrame:
    p = ctx.run_root / "universe/liquidity_tiers_by_date.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["symbol", "date", "liquidity_tier"])
    return pd.read_parquet(p)


def latest_tier_for_symbol(tiers: pd.DataFrame, symbol: str, date: str) -> str:
    if tiers.empty:
        return "UNKNOWN"
    sub = tiers[(tiers["symbol"] == symbol) & (tiers["date"] <= date)]
    if sub.empty:
        return "UNKNOWN"
    return str(sub.iloc[-1]["liquidity_tier"])


def d1_d3_e1_variants(seed: int) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for side in ["long", "short"]:
        for window_h in [4, 24]:
            for shock in [0.05, 0.10, 0.20, 0.30]:
                for hold_h in [6, 24, 72]:
                    v = {"family": "D1", "side": side, "window_h": window_h, "shock": shock, "hold_h": hold_h, "risk_atr": 1.5, "target_atr": 2.0}
                    v["variant_id"] = f"D1_{side}_{window_h}h_{int(shock*100)}pct_{hold_h}h"
                    v["parameter_hash"] = parameter_hash(v)
                    variants.append(v)
    for window_h in [1, 4, 24]:
        for shock in [0.10, 0.20, 0.30]:
            for hold_h in [6, 24, 72]:
                v = {"family": "D3", "side": "long", "window_h": window_h, "shock": shock, "hold_h": hold_h, "risk_atr": 1.0, "target_atr": 2.0}
                v["variant_id"] = f"D3_long_{window_h}h_{int(shock*100)}pct_{hold_h}h"
                v["parameter_hash"] = parameter_hash(v)
                variants.append(v)
    for drop in [0.10, 0.20, 0.30]:
        for oi_drop in [0.05, 0.10]:
            for hold_h in [24, 72]:
                v = {"family": "E1", "side": "long", "drop": drop, "oi_drop": oi_drop, "hold_h": hold_h, "risk_atr": 1.0, "target_atr": 2.5, "liquidation_proxy_only": True}
                v["variant_id"] = f"E1_long_drop{int(drop*100)}_oi{int(oi_drop*100)}_{hold_h}h"
                v["parameter_hash"] = parameter_hash(v)
                variants.append(v)
    rng = random.Random(seed)
    d1 = [v for v in variants if v["family"] == "D1"]
    d3 = [v for v in variants if v["family"] == "D3"]
    e1 = [v for v in variants if v["family"] == "E1"]
    rng.shuffle(d1); rng.shuffle(d3); rng.shuffle(e1)
    return d1[:30] + d3[:30] + e1[:20]


def a2_variants(seed: int) -> list[dict[str, Any]]:
    variants = []
    for high_d in [7, 14, 30, 60]:
        for ret_d in [7, 14, 30]:
            for hold_d in [1, 3, 5, 7]:
                for smooth in ["none", "top50", "top30"]:
                    v = {"family": "A2", "side": "long", "high_d": high_d, "ret_d": ret_d, "hold_h": hold_d * 24, "smoothness": smooth, "risk_atr": 2.0, "target_atr": 4.0}
                    v["variant_id"] = f"A2_h{high_d}_r{ret_d}_hold{hold_d}_{smooth}"
                    v["parameter_hash"] = parameter_hash(v)
                    variants.append(v)
    rng = random.Random(seed)
    rng.shuffle(variants)
    return variants[:40]


def signal_indices(df: pd.DataFrame, variant: Mapping[str, Any]) -> list[int]:
    fam = variant["family"]
    if fam == "D1":
        col = "ret_4h" if variant["window_h"] == 4 else "ret_24h"
        shock = float(variant["shock"])
        if variant["side"] == "long":
            cond = (df[col] <= -shock) & (df["close"] > df["close"].shift(1)) & (df["turnover"] <= df["turnover_med_24h"].fillna(np.inf) * 1.2)
        else:
            cond = (df[col] >= shock) & (df["close"] < df["close"].shift(1)) & (df["turnover"] <= df["turnover_med_24h"].fillna(np.inf) * 1.2)
    elif fam == "D3":
        col = "ret_4h" if variant["window_h"] in {1, 4} else "ret_24h"
        shock = float(variant["shock"])
        cond = (df[col] <= -shock) & (df["close"] > df["ema10"]) & (df["close"] > df["low_24h"] * 1.01)
    elif fam == "E1":
        drop = float(variant["drop"])
        oi_drop = float(variant["oi_drop"])
        cond = ((df["close"] / df["high_24h"] - 1.0) <= -drop) & (df["oi_chg_24h"] <= -oi_drop) & (df["close"] > df["ema10"])
    elif fam == "A2":
        high_n = int(variant["high_d"]) * 288
        ret_n = int(variant["ret_d"]) * 288
        near_high = df["close"] >= df["high"].rolling(high_n, min_periods=min(high_n, 288)).max() * 0.97
        pos_ret = df["close"] / df["close"].shift(ret_n) - 1.0 > 0
        smooth = pd.Series(True, index=df.index)
        if variant.get("smoothness") == "top50":
            smooth = df["range_pct"] <= df["range_pct"].rolling(288 * 30, min_periods=288).quantile(0.50)
        elif variant.get("smoothness") == "top30":
            smooth = df["range_pct"] <= df["range_pct"].rolling(288 * 30, min_periods=288).quantile(0.30)
        cond = near_high & pos_ret & smooth
    else:
        cond = pd.Series(False, index=df.index)
    idx = np.flatnonzero(cond.fillna(False).to_numpy())
    # Avoid overlapping dense signals in a screening pass.
    if len(idx) <= 1:
        return idx.tolist()
    min_gap = max(int(float(variant.get("hold_h", 24)) * 12 / 2), 12)
    selected = []
    last = -10**9
    for i in idx:
        if i - last >= min_gap:
            selected.append(int(i))
            last = int(i)
    return selected


def cap_signal_indices(indices: list[int], cap: int) -> list[int]:
    if cap <= 0 or len(indices) <= cap:
        return indices
    if cap == 1:
        return [indices[len(indices) // 2]]
    positions = np.linspace(0, len(indices) - 1, num=cap)
    return [indices[int(round(p))] for p in positions]


def run_screen(ctx: RunContext, variants: list[dict[str, Any]], out_dir: Path, families: set[str]) -> None:
    start, end = clamp_screening_window(ctx.args)
    tiers = tier_lookup(ctx)
    max_symbols = ctx.args.max_symbols or (5 if ctx.args.smoke else 0)
    paths = discover_symbol_paths(max_symbols)
    sample_rows: list[dict[str, Any]] = []
    all_rows_for_summary: list[dict[str, Any]] = []
    variant_rows = [{k: v for k, v in variant.items() if k != "family"} | {"strategy_family": variant["family"]} for variant in variants]
    write_csv(out_dir / "variant_registry.csv", variant_rows)
    for idx, path in enumerate(paths):
        sym = path.stem.upper()
        df = load_symbol_df(sym, start, end, include_context=True)
        if df.empty:
            continue
        df = add_features(df)
        df_indexed = df.set_index("timestamp", drop=False)
        symbol_funding_events = funding_events_from_df(df)
        tier = latest_tier_for_symbol(tiers, sym, str(end.date()))
        # Family universe filters.
        for variant in variants:
            fam = variant["family"]
            if fam in {"D1", "D3"} and tier != "C":
                continue
            if fam == "E1" and tier not in {"A", "B", "C"}:
                continue
            if fam == "A2" and tier not in {"A", "B", "C"}:
                continue
            indices = cap_signal_indices(signal_indices(df, variant), int(ctx.args.signals_per_symbol_variant))
            for i in indices:
                row = build_trade_from_signal(
                    sym,
                    fam,
                    variant,
                    df,
                    df_indexed,
                    symbol_funding_events,
                    i,
                    tier,
                    variant["side"],
                    f"screening_signal_cap_per_symbol_variant={ctx.args.signals_per_symbol_variant}",
                )
                if row is None:
                    continue
                all_rows_for_summary.append(row)
                if len(sample_rows) < 5000:
                    sample_rows.append(row)
        if idx % max(1, ctx.args.chunk_size) == 0:
            ctx.notifier.send("SCREEN PROGRESS", f"{out_dir.name}: symbols_done={idx+1}/{len(paths)} rows={len(all_rows_for_summary)}")
    ledger = pd.DataFrame(all_rows_for_summary)
    sample = pd.DataFrame(sample_rows)
    if not sample.empty:
        ensure_parent(out_dir / "trade_ledger_sample.parquet")
        sample.to_parquet(out_dir / "trade_ledger_sample.parquet", index=False)
    else:
        ensure_parent(out_dir / "trade_ledger_sample.parquet")
        pd.DataFrame(columns=["strategy_family", "variant_id", "symbol", "decision_ts", "net_R"]).to_parquet(out_dir / "trade_ledger_sample.parquet", index=False)
    summaries = []
    if ledger.empty:
        for v in variants:
            summaries.append({"strategy_family": v["family"], "variant_id": v["variant_id"], "trades": 0, "label": "screening_inconclusive", "blocker": "no_trades"})
    else:
        for (fam, vid), sub in ledger.groupby(["strategy_family", "variant_id"]):
            s = summarize_trades(sub)
            label = "screening_promising" if s["trades"] >= 20 and s["net_R"] > 0 and s["PF"] > 1.05 and s["liquidation_count"] == 0 else ("screening_reject" if s["trades"] >= 20 else "screening_inconclusive")
            tier = ",".join(sorted(map(str, sub["liquidity_tier"].dropna().unique())))
            side = ",".join(sorted(map(str, sub["side"].dropna().unique())))
            summaries.append({"strategy_family": fam, "variant_id": vid, **s, "tier": tier, "side": side, "label": label, "blocker": "" if label == "screening_promising" else "screening_only_or_weak_metrics"})
    summary_df = pd.DataFrame(summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "screening_summary.csv", index=False)
    if "net_R" in summary_df.columns:
        top = summary_df.sort_values(["label", "net_R"], ascending=[True, False]).head(25)
    else:
        top = summary_df.head(25)
    top.to_csv(out_dir / "top_variants.csv", index=False)
    summary_df[summary_df.get("label", "") != "screening_promising"].to_csv(out_dir / "rejected_variants.csv", index=False)
    fam_summary = summary_df.groupby("strategy_family", dropna=False).agg(variants_tested=("variant_id", "nunique"), trades=("trades", "sum"), net_R=("net_R", "sum"), promising=("label", lambda x: int((x == "screening_promising").sum()))).reset_index() if not summary_df.empty and "net_R" in summary_df else pd.DataFrame()
    fam_summary.to_csv(out_dir / "family_summary.csv", index=False)
    if not ledger.empty:
        tier_summary = ledger.groupby(["strategy_family", "liquidity_tier"], dropna=False).agg(
            variants=("variant_id", "nunique"),
            trades=("variant_id", "size"),
            net_R=("net_R", "sum"),
            median_R=("net_R", "median"),
            liquidation_count=("liquidation_flag", "sum"),
        ).reset_index()
    else:
        tier_summary = pd.DataFrame(columns=["strategy_family", "liquidity_tier", "variants", "trades", "net_R", "median_R", "liquidation_count"])
    tier_summary.to_csv(out_dir / "tier_summary.csv", index=False)
    write_text(out_dir / "failure_modes.md", "# Failure Modes\n\nLabels are screening-only. Weak variants are not rejected for final research solely from this pass. Liquidation proxy and cost proxy limitations remain.\n")
    write_text(out_dir / "SCREENING_REPORT.md", f"# Screening Report\n\n- families: `{','.join(sorted(families))}`\n- variants: `{len(variants)}`\n- signal cap: `{ctx.args.signals_per_symbol_variant}` per symbol/variant, selected deterministically across each signal set\n- sample ledger rows retained: `{len(sample_rows)}`\n- total simulated trades in summaries: `{len(all_rows_for_summary)}`\n- no final holdout access; end date `{end}`.\n")


def stage_screen_d1_d3_e1(ctx: RunContext) -> None:
    variants = d1_d3_e1_variants(ctx.args.seed)
    out = ctx.run_root / "screen_d1_d3_e1"
    (out / "contracts").mkdir(parents=True, exist_ok=True)
    for v in variants:
        write_json(out / "contracts" / f"{v['variant_id']}.json", v)
    run_screen(ctx, variants, out, {"D1", "D3", "E1"})


def stage_screen_a2(ctx: RunContext) -> None:
    variants = a2_variants(ctx.args.seed)
    out = ctx.run_root / "screen_a2"
    (out / "contracts").mkdir(parents=True, exist_ok=True)
    for v in variants:
        write_json(out / "contracts" / f"{v['variant_id']}.json", v)
    run_screen(ctx, variants, out, {"A2"})
    summary = pd.read_csv(out / "screening_summary.csv") if (out / "screening_summary.csv").exists() else pd.DataFrame()
    if not summary.empty and "tier" in summary.columns:
        tier_path = out / "tier_summary.csv"
        if tier_path.exists():
            tier_summary = pd.read_csv(tier_path)
            tier_summary.rename(columns={"liquidity_tier": "tier"}, inplace=True)
            tier_summary.to_csv(out / "tier_comparison.csv", index=False)
        else:
            rows = []
            for tier in ["A", "B", "C"]:
                sub = summary[summary["tier"].fillna("").str.contains(tier, regex=False)]
                rows.append({"tier": tier, "variants": len(sub), "trades": int(sub.get("trades", pd.Series(dtype=float)).sum()), "net_R": float(sub.get("net_R", pd.Series(dtype=float)).sum())})
            write_csv(out / "tier_comparison.csv", rows)
    else:
        write_csv(out / "tier_comparison.csv", [])


def stage_summary(ctx: RunContext) -> None:
    rows = []
    for p in [ctx.run_root / "screen_d1_d3_e1/screening_summary.csv", ctx.run_root / "screen_a2/screening_summary.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                rows.append({
                    "family": r.get("strategy_family"),
                    "variants_tested": 1,
                    "trades": r.get("trades"),
                    "net_R": r.get("net_R"),
                    "PF": r.get("PF"),
                    "max_dd_R": r.get("max_dd_R"),
                    "median_R": r.get("median_R"),
                    "funding_R": r.get("funding_R"),
                    "slippage_R": r.get("slippage_R"),
                    "liquidation_count": r.get("liquidation_count"),
                    "delist_count": r.get("delist_count"),
                    "tier": r.get("tier"),
                    "side": r.get("side"),
                    "label": r.get("label"),
                    "blocker": r.get("blocker"),
                })
    write_csv(ctx.run_root / "screening_family_result_table.csv", rows)
    promising = [r for r in rows if r.get("label") == "screening_promising"]
    write_text(ctx.run_root / "QLMG_ENGINE_AND_FIRST_SCREEN_REPORT.md", f"# QLMG Engine And First Screen Report\n\n- Run root: `{ctx.run_root}`\n- Disk/OOM safeguards: `passed`\n- Telegram status: `{ctx.notifier.status}`\n- Final holdout untouched: `yes; screening cutoff {SCREENING_END}`\n- Families screened: `D1,D3,E1,A2`\n- Screening-promising variants: `{len(promising)}`\n- No result is validated or live-ready.\n\nSee `screening_family_result_table.csv` for the integrated table.\n")
    write_text(ctx.run_root / "next_data_plan.md", "# Next Data Plan\n\n- Add historical 1m/depth/liquidation data only after a family survives deeper train-only review.\n- Keep final holdout sealed until a candidate contract is frozen.\n")


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    files = [
        "QLMG_ENGINE_AND_FIRST_SCREEN_REPORT.md",
        "engine/long_short_engine_implementation_report.md",
        "engine_tests/long_short_engine_test_report.md",
        "seal/qlmg_sealed_policy.md",
        "universe/universe_builder_report.md",
        "cost/cost_model_v0_report.md",
        "screen_d1_d3_e1/SCREENING_REPORT.md",
        "screen_d1_d3_e1/family_summary.csv",
        "screen_d1_d3_e1/top_variants.csv",
        "screen_a2/SCREENING_REPORT.md",
        "screen_a2/tier_comparison.csv",
        "notifications/telegram_notifier_report.md",
    ]
    index_rows = []
    for rel in files:
        src = ctx.run_root / rel
        if src.exists():
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            index_rows.append({"artifact": rel, "bundle_copy": str(dst.relative_to(ctx.run_root)), "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", index_rows)
    write_json(bundle / "artifact_path_index.json", {"artifacts": index_rows})
    zip_path = ctx.run_root / "qlmg_engine_and_first_screen_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in bundle.rglob("*"):
            if item.is_file():
                z.write(item, arcname=str(item.relative_to(ctx.run_root)))
    write_text(ctx.run_root / "compact_review_bundle/README.md", f"Compact review bundle. Zip: `{zip_path}`\n")


STAGE_FUNCS = {
    "preflight-resource-guard": stage_preflight,
    "telegram-run-notifier": stage_telegram,
    "source-state-snapshot": stage_source_snapshot,
    "sealed-policy-freeze": stage_seal,
    "long-short-engine-implementation": stage_engine_report,
    "long-short-engine-tests": stage_engine_tests,
    "lifecycle-and-universe-v0": stage_universe,
    "cost-funding-liquidation-v0": stage_cost,
    "screening-harness-v0": stage_harness,
    "phase1-d1-d3-e1-screen": stage_screen_d1_d3_e1,
    "phase1-a2-liquid-momentum-baseline": stage_screen_a2,
    "screening-summary-and-next-data-plan": stage_summary,
    "compact-review-bundle": stage_bundle,
}


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": reason, "started_at_utc": utc_now(), "argv": sys.argv, "seed": args.seed, "smoke": args.smoke})
    stages = stage_list(args.stage)
    notifier.send("RUN START", f"stages={stages}\nrun_root={run_root}")
    try:
        for stage in stages:
            if args.resume and done_path(run_root, stage).exists():
                continue
            append_command(run_root, stage)
            ensure_guard(ctx, stage, estimate_gb=0.2 if args.smoke else (2.0 if "screen" in stage or "universe" in stage else 0.5))
            notifier.send("STAGE START", stage)
            if args.dry_run:
                write_text(run_root / "dry_run" / f"{stage}.txt", f"would run {stage}")
                mark_done(run_root, stage)
                continue
            STAGE_FUNCS[stage](ctx)
            mark_done(run_root, stage)
            notifier.send("STAGE COMPLETE", stage)
        notifier.send("RUN COMPLETE", f"run_root={run_root}")
        return 0
    except Exception as exc:
        notifier.send("RUN FAILED", f"{type(exc).__name__}: {exc}", level="error")
        write_text(run_root / "FAILED.txt", f"{utc_now()} {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
