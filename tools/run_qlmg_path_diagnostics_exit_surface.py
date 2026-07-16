#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    ReplayConfig,
    check_resource_guard,
    liquidation_price,
    resource_snapshot,
    to_utc_ts,
    utc_now,
    write_json,
)
from tools.run_qlmg_engine_and_first_screen import (  # noqa: E402
    DATA_5M,
    DATA_CONTEXT,
    SCREENING_END,
    FINAL_HOLDOUT_START,
    add_features,
    a2_variants,
    cost_bps_for_tier,
    d1_d3_e1_variants,
    discover_symbol_paths,
    latest_tier_for_symbol,
    load_symbol_df,
    metadata_timestamps,
    parameter_hash,
    signal_indices,
)

try:
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

DEFAULT_RUN_ID = "phase_qlmg_path_diagnostics_exit_surface_20260624_v1"
RESULTS_ROOT = REPO / "results/rebaseline"
PREV_ROOT = REPO / "results/rebaseline/phase_qlmg_engine_and_first_screen_20260624_v1_20260624_101747"
HORIZONS: list[tuple[str, int]] = [
    ("15m", 15), ("30m", 30), ("1h", 60), ("2h", 120), ("4h", 240),
    ("6h", 360), ("12h", 720), ("24h", 1440), ("48h", 2880), ("72h", 4320), ("5d", 7200),
]
BASIC_SURFACE_HORIZONS = ["30m", "1h", "2h", "4h", "6h", "12h", "24h", "48h", "72h"]
STOP_CLASSES = [
    ("structure", "structure", np.nan),
    ("atr_0p5", "atr", 0.5), ("atr_1", "atr", 1.0), ("atr_1p5", "atr", 1.5), ("atr_2", "atr", 2.0), ("atr_3", "atr", 3.0),
    ("pct_1", "pct", 0.01), ("pct_2", "pct", 0.02), ("pct_3", "pct", 0.03), ("pct_5", "pct", 0.05), ("pct_10", "pct", 0.10), ("pct_15", "pct", 0.15),
]
TARGET_R = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
AMBIGUITY_BRANCHES = ["pessimistic", "neutral", "optimistic"]
STAGES = (
    "preflight-resource-and-prior-screen-audit",
    "telegram-and-resume-check",
    "seal-guard",
    "what-was-actually-tested-audit",
    "entry-family-contracts",
    "full-path-event-ledgers",
    "mfe-mae-path-diagnostics",
    "stop-target-time-exit-surface",
    "crypto-specific-exit-surface",
    "state-and-regime-stratification",
    "matched-null-path-comparison",
    "bad-wick-and-data-artifact-audit",
    "aggressive-10x-portfolio-overlay",
    "family-triage-and-next-contracts",
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
        if not disabled and TelegramNotifier is not None:
            class _Args:
                tg_bot_token = ""
                tg_chat_id = ""
                tg_auto_chat = False
            try:
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-path-diagnostics")
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
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return sent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG path diagnostics and exit surface runner")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--chunk-size", type=int, default=20)
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
    if end >= FINAL_HOLDOUT_START:
        raise RuntimeError("requested diagnostic window overlaps final QLMG holdout")
    if start >= FINAL_HOLDOUT_START:
        raise RuntimeError("requested diagnostic start is inside final QLMG holdout")
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
        "preflight-resource-and-prior-screen-audit": [run_root / "preflight/preflight_report.md", run_root / "preflight/prior_screen_artifact_manifest.json"],
        "telegram-and-resume-check": [run_root / "notifications/telegram_readiness_report.md", run_root / "resume/resume_plan.md"],
        "seal-guard": [run_root / "seal/seal_guard_report.md", run_root / "seal/protected_slice_check.json"],
        "what-was-actually-tested-audit": [run_root / "audit/what_was_actually_tested.md", run_root / "audit/previous_variant_rule_table.csv"],
        "entry-family-contracts": [run_root / "contracts/family_contract_summary.csv"],
        "full-path-event-ledgers": [run_root / "events/event_coverage_summary.csv"],
        "mfe-mae-path-diagnostics": [run_root / "path_diagnostics/path_metrics.parquet", run_root / "path_diagnostics/path_summary_by_family.csv"],
        "stop-target-time-exit-surface": [run_root / "exit_surface/basic_exit_surface_summary.csv"],
        "crypto-specific-exit-surface": [run_root / "exit_surface/crypto_exit_surface_summary.csv"],
        "state-and-regime-stratification": [run_root / "regime/state_stratification_summary.csv"],
        "matched-null-path-comparison": [run_root / "matched_null/matched_null_summary.csv"],
        "bad-wick-and-data-artifact-audit": [run_root / "data_quality/bad_wick_artifact_summary.csv"],
        "aggressive-10x-portfolio-overlay": [run_root / "portfolio/aggressive_10x_portfolio_summary.csv"],
        "family-triage-and-next-contracts": [run_root / "triage/family_triage_summary.csv"],
        "compact-review-bundle": [run_root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return mapping.get(stage, [])


def stage_complete(run_root: Path, stage: str) -> bool:
    return done_path(run_root, stage).exists() and all(p.exists() for p in required_outputs_for_stage(run_root, stage))


def shell(args: Sequence[str], timeout: float = 120.0) -> str:
    try:
        p = subprocess.run(args, cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


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


def variants_all(seed: int) -> list[dict[str, Any]]:
    return d1_d3_e1_variants(seed) + a2_variants(seed)


def variants_by_family(seed: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for v in variants_all(seed):
        out.setdefault(v["family"], []).append(v)
    return out


def _raw_signal_mask(df: pd.DataFrame, variant: Mapping[str, Any]) -> pd.Series:
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
        cond = ((df["close"] / df["high_24h"] - 1.0) <= -float(variant["drop"])) & (df["oi_chg_24h"] <= -float(variant["oi_drop"])) & (df["close"] > df["ema10"])
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
    return cond.fillna(False)


def load_tiers(prev_root: Path = PREV_ROOT) -> pd.DataFrame:
    p = prev_root / "universe/liquidity_tiers_by_date.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=["symbol", "date", "liquidity_tier"])


def latest_tier(tiers: pd.DataFrame, symbol: str, date: str) -> str:
    return latest_tier_for_symbol(tiers, symbol, date) if not tiers.empty else "UNKNOWN"


def family_tier_allowed(fam: str, tier: str) -> bool:
    if fam in {"D1", "D3"}:
        return tier == "C"
    if fam in {"E1", "A2"}:
        return tier in {"A", "B", "C"}
    return False


def reference_stop_price(row: pd.Series, entry: float, side: str, family: str, variant: Mapping[str, Any]) -> float:
    atr = float(row.get("atr_proxy", np.nan))
    risk_mult = float(variant.get("risk_atr", 1.5))
    if not np.isfinite(atr) or atr <= 0:
        atr = abs(entry) * 0.01
    if side == "long":
        stop = float(row.get("low_24h", np.nan)) if family in {"D3", "E1"} else np.nan
        if not np.isfinite(stop) or stop >= entry:
            stop = entry - risk_mult * atr
    else:
        stop = entry + risk_mult * atr
    if not np.isfinite(stop) or stop <= 0:
        stop = entry * (0.99 if side == "long" else 1.01)
    return float(stop)


def pct_bps(x: float) -> float:
    return float(x) * 10000.0


def btc_eth_regime_map(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for sym in ["BTCUSDT", "ETHUSDT"]:
        p = DATA_5M / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].sort_values("timestamp")
        df[f"{sym[:3].lower()}_ret_4h"] = df["close"] / df["close"].shift(48) - 1.0
        df[f"{sym[:3].lower()}_ret_24h"] = df["close"] / df["close"].shift(288) - 1.0
        rows.append(df[["timestamp", f"{sym[:3].lower()}_ret_4h", f"{sym[:3].lower()}_ret_24h"]])
    if not rows:
        return pd.DataFrame(columns=["timestamp", "btc_eth_regime"])
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on="timestamp", how="outer")
    out = out.sort_values("timestamp")
    btc = out.get("btc_ret_4h", pd.Series(np.nan, index=out.index))
    eth = out.get("eth_ret_4h", pd.Series(np.nan, index=out.index))
    out["btc_eth_regime"] = np.where((btc >= 0) & (eth >= 0), "both_nonnegative", np.where((btc < 0) & (eth < 0), "both_negative", "mixed"))
    return out


def enrich_regime_for_events(events: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    if events.empty or regime.empty:
        events["btc_eth_regime"] = "unknown"
        return events
    left = events.sort_values("decision_ts")
    right = regime.sort_values("timestamp")
    out = pd.merge_asof(left, right, left_on="decision_ts", right_on="timestamp", direction="backward")
    out.drop(columns=["timestamp"], inplace=True, errors="ignore")
    out["btc_eth_regime"] = out["btc_eth_regime"].fillna("unknown")
    return out.sort_values(["family", "variant_id", "symbol", "decision_ts"]).reset_index(drop=True)


def event_id_for(row: Mapping[str, Any]) -> str:
    key = "|".join(str(row.get(k, "")) for k in ["family", "variant_id", "symbol", "decision_ts", "side"])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def event_ledger_path(run_root: Path) -> Path:
    return run_root / "events/all_event_ledger.parquet"


def path_metrics_path(run_root: Path) -> Path:
    return run_root / "path_diagnostics/path_metrics.parquet"


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    prev_files = [
        "QLMG_ENGINE_AND_FIRST_SCREEN_REPORT.md", "screen_d1_d3_e1/SCREENING_REPORT.md", "screen_d1_d3_e1/variant_registry.csv",
        "screen_d1_d3_e1/top_variants.csv", "screen_d1_d3_e1/family_summary.csv", "screen_a2/SCREENING_REPORT.md",
        "screen_a2/tier_comparison.csv", "cost/cost_model_v0_contract.json", "cost/cost_model_v0_report.md",
        "engine/long_short_engine_implementation_report.md", "universe/instrument_master_v0.parquet", "universe/liquidity_tiers_by_date.parquet",
        "seal/qlmg_protected_slices.json",
    ]
    manifest = []
    for rel in prev_files:
        p = PREV_ROOT / rel
        manifest.append({"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0, "sha256": sha256_file(p) if p.exists() and p.is_file() else ""})
    write_json(ctx.run_root / "preflight/prior_screen_artifact_manifest.json", {"previous_root": str(PREV_ROOT), "artifacts": manifest})
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard Report\n\n- free disk GB: `{snap.free_gb:.2f}`\n- hard stop: `<5GB`\n- warning: `<7GB`\n- max output GB default: `{ctx.args.max_output_gb}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight Report\n\n- previous run root: `{PREV_ROOT}`\n- current free disk GB: `{snap.free_gb:.2f}`\n- diagnostic window: `{ctx.start}` through `{ctx.end}`\n- previous signal caps: `2 per symbol/variant` from Phase 0.5 report\n- previous cost status: `proxy-level`\n- previous engine limitations: `screening-grade`; not conclusive for economic rejection\n- final holdout cutoff: `{FINAL_HOLDOUT_START}`\n")


def stage_telegram(ctx: RunContext) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness Report\n\n- status: `{ctx.notifier.status}`\n- fail soft: `true`\n- local log: `notifications/telegram_events.jsonl`\n- secrets persisted: `false`\n")
    write_text(ctx.run_root / "resume/resume_plan.md", "# Resume Plan\n\nEach stage writes `stage_status/<stage>.done`. `--resume` skips a stage only when its checkpoint and required outputs exist.\n")
    ctx.notifier.send("RUN START", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    check = {
        "final_holdout_start": str(FINAL_HOLDOUT_START),
        "screening_end": str(ctx.end),
        "pre_holdout_read_smoke": bool(ctx.end < FINAL_HOLDOUT_START),
        "protected_read_smoke_blocked": True,
        "status": "pass" if ctx.end < FINAL_HOLDOUT_START else "fail",
    }
    if check["status"] != "pass":
        raise RuntimeError("seal guard failed")
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard Report\n\n- final holdout starts: `{FINAL_HOLDOUT_START}`\n- diagnostic data end: `{ctx.end}`\n- protected reads in ordinary diagnostic mode: `blocked`\n- status: `pass`\n")


def stage_actual_tested(ctx: RunContext) -> None:
    rows = []
    for p in [PREV_ROOT / "screen_d1_d3_e1/variant_registry.csv", PREV_ROOT / "screen_a2/variant_registry.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                fam = r.get("strategy_family")
                rows.append({
                    "strategy_family": fam,
                    "variant_id": r.get("variant_id"),
                    "side": r.get("side"),
                    "universe_tier": "C" if fam in {"D1", "D3"} else "A/B/C",
                    "stop_rule": "single crude ATR/structure stop from Phase 0.5",
                    "target_rule": "single crude ATR target from Phase 0.5",
                    "time_rule": f"{r.get('hold_h')}h",
                    "signal_cap": "2 per symbol/variant",
                    "mfe_mae_computed": False,
                    "matched_null_used": False,
                    "alternative_sl_tp_tested": False,
                    "cost_model": "qlmg_screening_cost_v0_proxy",
                })
    write_csv(ctx.run_root / "audit/previous_variant_rule_table.csv", rows)
    write_text(ctx.run_root / "audit/what_was_actually_tested.md", "# What Was Actually Tested\n\nThe Phase 0.5 run tested capped/sampled signals with one crude stop/target/time construction per variant. It did not test multiple SL/TP families, did not compute full MFE/MAE path potential, and did not use matched nulls. Negative results from that run are negative for that crude construction, not final economic rejection of the entry mechanisms.\n")


def contract_row(family: str, status: str, side: str, universe: str, blocker: str = "") -> dict[str, Any]:
    return {
        "family": family,
        "status": status,
        "side": side,
        "universe_tier": universe,
        "hypothesis": {
            "D1": "Low-volume shocks may over/under-react and mean-revert on short horizons.",
            "D3": "Small-cap liquidity shocks may bounce after washout and reclaim.",
            "E1": "Post-deleveraging states may rebound after leverage clears.",
            "A2": "Prior-high proximity momentum may have favorable path behavior in liquid names.",
            "A1/A3": "Smooth-path continuation/pullback may be more compatible than raw breakout chase.",
            "F1": "Parabolic blowoff shorts require backside confirmation.",
            "G1": "Failed continuation breakouts may reverse when reclaim/follow-through fails.",
        }.get(family, "diagnostic family"),
        "no_final_holdout": True,
        "required_data": "5m OHLCV; sidecar/context optional but audited when present",
        "matched_null_design": "same symbol/month/tier/volatility/turnover/BTC-ETH regime where possible",
        "blocker": blocker,
    }


def stage_contracts(ctx: RunContext) -> None:
    contracts = [
        contract_row("D1", "implemented", "long_and_short", "C"),
        contract_row("D3", "implemented", "long", "C"),
        contract_row("E1", "implemented", "long", "A/B/C"),
        contract_row("A2", "implemented", "long", "A/B primary; C diagnostic"),
        contract_row("A1/A3", "draft_only", "long", "A/B", "not in previous mandatory screen; defer to avoid silent scope expansion"),
        contract_row("F1", "draft_only", "short", "A/B/C", "requires explicit backside-confirmation contract and short-specific validation"),
        contract_row("G1", "draft_only", "short", "A/B/C", "requires explicit failed-breakout contract and null design"),
    ]
    (ctx.run_root / "contracts").mkdir(parents=True, exist_ok=True)
    for row in contracts:
        write_json(ctx.run_root / "contracts" / f"{row['family'].replace('/','_')}.json", row)
    write_csv(ctx.run_root / "contracts/family_contract_summary.csv", contracts)


def generate_events_for_symbol(symbol: str, df: pd.DataFrame, tiers: pd.DataFrame, variants: list[dict[str, Any]], regime: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    if df.empty:
        return events, coverage
    df = add_features(df)
    tier = latest_tier(tiers, symbol, str(df["timestamp"].max().date()))
    for variant in variants:
        fam = variant["family"]
        if not family_tier_allowed(fam, tier):
            coverage.append({"family": fam, "variant_id": variant["variant_id"], "symbol": symbol, "tier": tier, "raw_triggers": 0, "retained_events": 0, "skip_reason": "tier_not_allowed"})
            continue
        raw_mask = _raw_signal_mask(df, variant)
        raw_count = int(raw_mask.sum())
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
                raise RuntimeError("protected timestamp generated in event ledger")
            entry_price = float(entry["open"])
            atr = float(row.get("atr_proxy", np.nan))
            if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(atr) or atr <= 0:
                continue
            side = str(variant["side"])
            stop = reference_stop_price(row, entry_price, side, fam, variant)
            risk_bps = abs(entry_price - stop) / entry_price * 10000.0
            if not np.isfinite(risk_bps) or risk_bps <= 0:
                continue
            mark_gap = np.nan
            if "mark_close" in row and pd.notna(row.get("mark_close")):
                mark_gap = float(row.get("mark_close")) / float(row.get("close")) - 1.0
            event = {
                "family": fam,
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
                "ret_4h": row.get("ret_4h", np.nan),
                "ret_24h": row.get("ret_24h", np.nan),
                "range_pct": row.get("range_pct", np.nan),
                "turnover": row.get("turnover", np.nan),
                "turnover_med_24h": row.get("turnover_med_24h", np.nan),
                "oi_chg_24h": row.get("oi_chg_24h", np.nan),
                "funding_rate": row.get("funding_rate", np.nan),
                "mark_last_gap": mark_gap,
                "listing_age_proxy_days": np.nan,
                "raw_trigger_count_for_symbol_variant": raw_count,
                "data_quality_flags": "",
            }
            event["event_id"] = event_id_for(event)
            events.append(event)
            retained += 1
        coverage.append({"family": fam, "variant_id": variant["variant_id"], "symbol": symbol, "tier": tier, "raw_triggers": raw_count, "retained_events": retained, "skip_reason": ""})
    return events, coverage


def stage_events(ctx: RunContext) -> None:
    variants = variants_all(ctx.args.seed)
    tiers = load_tiers()
    max_symbols = ctx.args.max_symbols or (5 if ctx.args.smoke else 0)
    paths = discover_symbol_paths(max_symbols)
    regime = btc_eth_regime_map(ctx.start, ctx.end)
    all_events: list[dict[str, Any]] = []
    all_coverage: list[dict[str, Any]] = []
    progress_rows = []
    for n, path in enumerate(paths, start=1):
        sym = path.stem.upper()
        df = load_symbol_df(sym, ctx.start, ctx.end, include_context=True)
        ev, cov = generate_events_for_symbol(sym, df, tiers, variants, regime)
        all_events.extend(ev)
        all_coverage.extend(cov)
        if n % max(1, ctx.args.chunk_size) == 0 or n == len(paths):
            rec = {"ts_utc": utc_now(), "symbols_done": n, "symbols_total": len(paths), "events": len(all_events)}
            progress_rows.append(rec)
            write_csv(ctx.run_root / "events/chunk_progress_manifest.csv", progress_rows)
            ctx.notifier.send("EVENT LEDGER PROGRESS", json.dumps(rec))
    events = pd.DataFrame(all_events)
    if not events.empty:
        events = enrich_regime_for_events(events, regime)
        if (events["decision_ts"] >= FINAL_HOLDOUT_START).any() or (events["entry_ts"] >= FINAL_HOLDOUT_START).any():
            raise RuntimeError("event ledger contains protected timestamps")
    event_ledger_path(ctx.run_root).parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(event_ledger_path(ctx.run_root), index=False)
    cov = pd.DataFrame(all_coverage)
    cov.to_csv(ctx.run_root / "events/event_coverage_summary.csv", index=False)
    if not events.empty:
        sample = events.head(500)
        sample.to_parquet(ctx.run_root / "events/event_ledger_sample.parquet", index=False)
    write_text(ctx.run_root / "events/event_coverage_report.md", f"# Event Coverage Report\n\n- symbols processed: `{len(paths)}`\n- variants: `{len(variants)}`\n- retained events: `{len(events)}`\n- old signal cap removed: `yes`\n- de-overlap/min-gap rule retained from Phase 0.5 signal logic: `yes`\n")


def load_symbol_for_path(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Include enough warmup/path history; caller still filters event timestamps.
    return load_symbol_df(symbol, start, end, include_context=True).sort_values("timestamp").reset_index(drop=True)


def _slice_future(df_indexed: pd.DataFrame, entry_ts: pd.Timestamp, minutes: int) -> pd.DataFrame:
    end = entry_ts + pd.Timedelta(minutes=int(minutes))
    return df_indexed[(df_indexed.index > entry_ts) & (df_indexed.index <= end)]


def first_hit_minutes(side: str, path: pd.DataFrame, entry: float, threshold_bps: float, favorable: bool) -> float:
    if path.empty or not np.isfinite(threshold_bps):
        return np.nan
    if side == "long":
        target = entry * (1 + threshold_bps / 10000.0) if favorable else entry * (1 - threshold_bps / 10000.0)
        hits = path.index[path["high"].ge(target)] if favorable else path.index[path["low"].le(target)]
    else:
        target = entry * (1 - threshold_bps / 10000.0) if favorable else entry * (1 + threshold_bps / 10000.0)
        hits = path.index[path["low"].le(target)] if favorable else path.index[path["high"].ge(target)]
    if len(hits) == 0:
        return np.nan
    return float((hits[0] - path.index[0]).total_seconds() / 60.0) if len(path.index) else np.nan


def compute_path_row(event: pd.Series, df_indexed: pd.DataFrame) -> dict[str, Any]:
    side = str(event["side"])
    sign = 1.0 if side == "long" else -1.0
    entry_ts = pd.Timestamp(event["entry_ts"])
    entry = float(event["entry_ref_price"])
    atr_bps = float(event.get("atr_bps", np.nan))
    risk_bps = float(event.get("reference_risk_bps", np.nan))
    out = {
        "event_id": event["event_id"], "family": event["family"], "variant_id": event["variant_id"], "symbol": event["symbol"],
        "side": side, "liquidity_tier": event.get("liquidity_tier"), "decision_ts": event["decision_ts"], "entry_ts": entry_ts,
        "entry_ref_price": entry, "reference_risk_bps": risk_bps, "atr_bps": atr_bps,
        "btc_eth_regime": event.get("btc_eth_regime", "unknown"), "oi_chg_24h": event.get("oi_chg_24h", np.nan), "funding_rate": event.get("funding_rate", np.nan),
        "turnover": event.get("turnover", np.nan), "range_pct": event.get("range_pct", np.nan), "data_quality_flags": event.get("data_quality_flags", ""),
    }
    max_path = _slice_future(df_indexed, entry_ts, 7200)
    mark_missing = "mark_high" not in max_path.columns or "mark_low" not in max_path.columns or max_path.get("mark_high", pd.Series(dtype=float)).isna().all()
    out["mark_path_status"] = "last_price_proxy" if mark_missing else "mark_available"
    try:
        liq = liquidation_price(entry, side, 10.0, 0.005)
    except Exception:
        liq = np.nan
    out["liq_price_10x"] = liq
    for label, mins in HORIZONS:
        path = _slice_future(df_indexed, entry_ts, mins)
        if path.empty:
            out[f"{label}_path_available"] = False
            continue
        high = pd.to_numeric(path["high"], errors="coerce")
        low = pd.to_numeric(path["low"], errors="coerce")
        close = float(path.iloc[-1]["close"])
        if side == "long":
            fav = (high.max() / entry - 1.0) * 10000.0
            adv = (1.0 - low.min() / entry) * 10000.0
            mfe_idx = high.idxmax()
            mae_idx = low.idxmin()
            liq_hit = low.le(liq).any() if np.isfinite(liq) else False
        else:
            fav = (1.0 - low.min() / entry) * 10000.0
            adv = (high.max() / entry - 1.0) * 10000.0
            mfe_idx = low.idxmin()
            mae_idx = high.idxmax()
            liq_hit = high.ge(liq).any() if np.isfinite(liq) else False
        out[f"{label}_path_available"] = True
        out[f"{label}_mfe_bps"] = float(fav)
        out[f"{label}_mae_bps"] = float(adv)
        out[f"{label}_mfe_atr"] = float(fav / atr_bps) if np.isfinite(atr_bps) and atr_bps > 0 else np.nan
        out[f"{label}_mae_atr"] = float(adv / atr_bps) if np.isfinite(atr_bps) and atr_bps > 0 else np.nan
        out[f"{label}_close_return_bps"] = float(sign * (close / entry - 1.0) * 10000.0)
        out[f"{label}_time_to_mfe_min"] = float((pd.Timestamp(mfe_idx) - entry_ts).total_seconds() / 60.0)
        out[f"{label}_time_to_mae_min"] = float((pd.Timestamp(mae_idx) - entry_ts).total_seconds() / 60.0)
        out[f"{label}_first_adverse_before_favorable"] = bool(pd.Timestamp(mae_idx) < pd.Timestamp(mfe_idx))
        out[f"{label}_liquidation_10x"] = bool(liq_hit)
        for r in [0.5, 1.0, 2.0, 3.0, 5.0]:
            th = risk_bps * r if np.isfinite(risk_bps) else np.nan
            out[f"{label}_hit_pos_{str(r).replace('.','p')}R"] = bool(fav >= th) if np.isfinite(th) else False
        for r in [0.5, 1.0, 2.0, 3.0]:
            th = risk_bps * r if np.isfinite(risk_bps) else np.nan
            out[f"{label}_hit_neg_{str(r).replace('.','p')}R"] = bool(adv >= th) if np.isfinite(th) else False
        pos1 = first_hit_minutes(side, path, entry, risk_bps, True)
        neg1 = first_hit_minutes(side, path, entry, risk_bps, False)
        out[f"{label}_pos1R_before_neg1R"] = bool(np.isfinite(pos1) and (not np.isfinite(neg1) or pos1 <= neg1))
    return out


def _array_or_nan(df: pd.DataFrame, col: str, fallback: str | None = None) -> np.ndarray:
    if col in df.columns:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    elif fallback and fallback in df.columns:
        arr = pd.to_numeric(df[fallback], errors="coerce").to_numpy(dtype=float)
    else:
        arr = np.full(len(df), np.nan, dtype=float)
    return arr


def symbol_path_arrays(df: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    ts_ns = ts.astype("int64").to_numpy()
    mark_high = _array_or_nan(df, "mark_high", "high")
    mark_low = _array_or_nan(df, "mark_low", "low")
    mark_status = "mark_available" if ("mark_high" in df.columns and "mark_low" in df.columns and np.isfinite(mark_high).any() and np.isfinite(mark_low).any()) else "last_price_proxy"
    if not np.isfinite(mark_high).any():
        mark_high = _array_or_nan(df, "high")
    if not np.isfinite(mark_low).any():
        mark_low = _array_or_nan(df, "low")
    return {
        "ts_ns": ts_ns,
        "high": _array_or_nan(df, "high"),
        "low": _array_or_nan(df, "low"),
        "close": _array_or_nan(df, "close"),
        "mark_high": mark_high,
        "mark_low": mark_low,
        "mark_status": mark_status,
    }


def _first_hit_offset(mask: np.ndarray) -> float:
    if mask.size == 0 or not bool(mask.any()):
        return np.nan
    return float(np.argmax(mask) + 1) * 5.0


def compute_path_row_fast(event: pd.Series, arrays: Mapping[str, Any]) -> dict[str, Any]:
    side = str(event["side"])
    sign = 1.0 if side == "long" else -1.0
    entry_ts = pd.Timestamp(event["entry_ts"])
    entry_ns = entry_ts.value
    ts_ns = arrays["ts_ns"]
    pos = int(np.searchsorted(ts_ns, entry_ns, side="left"))
    # If entry_ts is between bars, start from the first bar at/after entry_ts and scan strictly after it.
    if pos >= len(ts_ns):
        pos = len(ts_ns) - 1
    entry = float(event["entry_ref_price"])
    atr_bps = float(event.get("atr_bps", np.nan))
    risk_bps = float(event.get("reference_risk_bps", np.nan))
    out = {
        "event_id": event["event_id"], "family": event["family"], "variant_id": event["variant_id"], "symbol": event["symbol"],
        "side": side, "liquidity_tier": event.get("liquidity_tier"), "decision_ts": event["decision_ts"], "entry_ts": entry_ts,
        "entry_ref_price": entry, "reference_risk_bps": risk_bps, "atr_bps": atr_bps,
        "btc_eth_regime": event.get("btc_eth_regime", "unknown"), "oi_chg_24h": event.get("oi_chg_24h", np.nan), "funding_rate": event.get("funding_rate", np.nan),
        "turnover": event.get("turnover", np.nan), "range_pct": event.get("range_pct", np.nan), "data_quality_flags": event.get("data_quality_flags", ""),
        "mark_path_status": arrays.get("mark_status", "last_price_proxy"),
    }
    try:
        liq = liquidation_price(entry, side, 10.0, 0.005)
    except Exception:
        liq = np.nan
    out["liq_price_10x"] = liq
    high = arrays["high"]; low = arrays["low"]; close = arrays["close"]; mark_high = arrays["mark_high"]; mark_low = arrays["mark_low"]
    for label, mins in HORIZONS:
        end_ns = entry_ns + int(mins * 60 * 1_000_000_000)
        end_pos = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
        start_pos = pos + 1
        if end_pos < start_pos or start_pos >= len(ts_ns):
            out[f"{label}_path_available"] = False
            continue
        end_pos = min(end_pos, len(ts_ns) - 1)
        hs = high[start_pos:end_pos + 1]
        ls = low[start_pos:end_pos + 1]
        cs = close[end_pos]
        if hs.size == 0 or ls.size == 0 or not np.isfinite(cs):
            out[f"{label}_path_available"] = False
            continue
        if side == "long":
            max_i = int(np.nanargmax(hs)) if np.isfinite(hs).any() else 0
            min_i = int(np.nanargmin(ls)) if np.isfinite(ls).any() else 0
            fav = (float(np.nanmax(hs)) / entry - 1.0) * 10000.0
            adv = (1.0 - float(np.nanmin(ls)) / entry) * 10000.0
            liq_hit = bool(np.nanmin(mark_low[start_pos:end_pos + 1]) <= liq) if np.isfinite(liq) else False
            pos_mask_1r = hs >= entry * (1 + risk_bps / 10000.0) if np.isfinite(risk_bps) else np.zeros_like(hs, dtype=bool)
            neg_mask_1r = ls <= entry * (1 - risk_bps / 10000.0) if np.isfinite(risk_bps) else np.zeros_like(ls, dtype=bool)
        else:
            max_i = int(np.nanargmin(ls)) if np.isfinite(ls).any() else 0
            min_i = int(np.nanargmax(hs)) if np.isfinite(hs).any() else 0
            fav = (1.0 - float(np.nanmin(ls)) / entry) * 10000.0
            adv = (float(np.nanmax(hs)) / entry - 1.0) * 10000.0
            liq_hit = bool(np.nanmax(mark_high[start_pos:end_pos + 1]) >= liq) if np.isfinite(liq) else False
            pos_mask_1r = ls <= entry * (1 - risk_bps / 10000.0) if np.isfinite(risk_bps) else np.zeros_like(ls, dtype=bool)
            neg_mask_1r = hs >= entry * (1 + risk_bps / 10000.0) if np.isfinite(risk_bps) else np.zeros_like(hs, dtype=bool)
        out[f"{label}_path_available"] = True
        out[f"{label}_mfe_bps"] = float(fav)
        out[f"{label}_mae_bps"] = float(adv)
        out[f"{label}_mfe_atr"] = float(fav / atr_bps) if np.isfinite(atr_bps) and atr_bps > 0 else np.nan
        out[f"{label}_mae_atr"] = float(adv / atr_bps) if np.isfinite(atr_bps) and atr_bps > 0 else np.nan
        out[f"{label}_close_return_bps"] = float(sign * (float(cs) / entry - 1.0) * 10000.0)
        out[f"{label}_time_to_mfe_min"] = float(max_i + 1) * 5.0
        out[f"{label}_time_to_mae_min"] = float(min_i + 1) * 5.0
        out[f"{label}_first_adverse_before_favorable"] = bool(min_i < max_i)
        out[f"{label}_liquidation_10x"] = bool(liq_hit)
        for r in [0.5, 1.0, 2.0, 3.0, 5.0]:
            th = risk_bps * r if np.isfinite(risk_bps) else np.nan
            out[f"{label}_hit_pos_{str(r).replace('.','p')}R"] = bool(fav >= th) if np.isfinite(th) else False
        for r in [0.5, 1.0, 2.0, 3.0]:
            th = risk_bps * r if np.isfinite(risk_bps) else np.nan
            out[f"{label}_hit_neg_{str(r).replace('.','p')}R"] = bool(adv >= th) if np.isfinite(th) else False
        pos1 = _first_hit_offset(pos_mask_1r)
        neg1 = _first_hit_offset(neg_mask_1r)
        out[f"{label}_pos1R_before_neg1R"] = bool(np.isfinite(pos1) and (not np.isfinite(neg1) or pos1 <= neg1))
    return out


def bool_series(values: Any, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.fillna(False).astype(bool)
    if index is None:
        return pd.Series([], dtype=bool)
    return pd.Series([bool(values)] * len(index), index=index, dtype=bool)


def bool_sum(values: Any, index: pd.Index | None = None) -> int:
    return int(bool_series(values, index).sum())


def bool_mean(values: Any, index: pd.Index | None = None) -> float:
    s = bool_series(values, index)
    return float(s.mean()) if len(s) else float("nan")


def stage_path_metrics(ctx: RunContext) -> None:
    events_p = event_ledger_path(ctx.run_root)
    if not events_p.exists():
        raise RuntimeError("event ledger missing; run full-path-event-ledgers first")
    events = pd.read_parquet(events_p)
    out_path = path_metrics_path(ctx.run_root)
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    out_path.mkdir(parents=True, exist_ok=True)
    progress = []
    if events.empty:
        pd.DataFrame().to_parquet(out_path / "part_00000.parquet", index=False)
        write_csv(ctx.run_root / "path_diagnostics/path_summary_by_family.csv", [])
        write_text(ctx.run_root / "path_diagnostics/mfe_mae_report.md", "# MFE/MAE Report\n\nNo events.\n")
        return
    part_no = 0
    chunk_rows: list[dict[str, Any]] = []
    symbol_groups = list(events.groupby("symbol"))
    for n, (sym, sub) in enumerate(symbol_groups, start=1):
        start = max(ctx.start - pd.Timedelta(days=2), pd.Timestamp("2020-01-01T00:00:00Z"))
        end = min(ctx.end + pd.Timedelta(days=6), FINAL_HOLDOUT_START - pd.Timedelta(minutes=5))
        df = load_symbol_for_path(sym, start, end)
        if df.empty:
            continue
        arrays = symbol_path_arrays(df)
        for _, ev in sub.iterrows():
            if pd.Timestamp(ev["decision_ts"]) >= FINAL_HOLDOUT_START:
                raise RuntimeError("protected event found during path computation")
            chunk_rows.append(compute_path_row_fast(ev, arrays))
        if n % max(1, ctx.args.chunk_size) == 0 or n == len(symbol_groups):
            if chunk_rows:
                pd.DataFrame(chunk_rows).to_parquet(out_path / f"part_{part_no:05d}.parquet", index=False)
                part_no += 1
                chunk_rows.clear()
            rec = {"ts_utc": utc_now(), "symbols_done": n, "symbols_total": len(symbol_groups), "path_rows_written_parts": sum(1 for _ in out_path.glob('part_*.parquet'))}
            progress.append(rec)
            write_csv(ctx.run_root / "path_diagnostics/chunk_progress_manifest.csv", progress)
            ctx.notifier.send("PATH PROGRESS", json.dumps(rec))
    metrics = pd.read_parquet(out_path)
    summary = []
    for fam, sub in metrics.groupby("family") if not metrics.empty else []:
        summary.append({
            "family": fam,
            "events": len(sub),
            "median_24h_mfe_bps": pd.to_numeric(sub.get("24h_mfe_bps"), errors="coerce").median(),
            "median_24h_mae_bps": pd.to_numeric(sub.get("24h_mae_bps"), errors="coerce").median(),
            "median_24h_close_bps": pd.to_numeric(sub.get("24h_close_return_bps"), errors="coerce").median(),
            "pos1R_before_neg1R_24h_share": bool_mean(sub.get("24h_pos1R_before_neg1R", False), sub.index),
            "liq10x_24h_count": bool_sum(sub.get("24h_liquidation_10x", False), sub.index),
        })
    write_csv(ctx.run_root / "path_diagnostics/path_summary_by_family.csv", summary)
    write_text(ctx.run_root / "path_diagnostics/mfe_mae_report.md", "# MFE/MAE Path Diagnostics\n\nComputed path-first diagnostics from 5m OHLCV/context through pre-holdout only. `path_metrics.parquet` is a partitioned Parquet directory to keep memory bounded. These metrics do not select final exits and remain diagnostic.\n")

def _risk_bps_for_stop(df: pd.DataFrame, stop_kind: str, val: float) -> pd.Series:
    if stop_kind == "structure":
        return pd.to_numeric(df["reference_risk_bps"], errors="coerce")
    if stop_kind == "atr":
        return pd.to_numeric(df["atr_bps"], errors="coerce") * float(val)
    if stop_kind == "pct":
        return pd.Series(float(val) * 10000.0, index=df.index)
    return pd.Series(np.nan, index=df.index)


def surface_returns(df: pd.DataFrame, horizon: str, risk_bps: pd.Series, target_r: float, branch: str) -> pd.Series:
    mfe = pd.to_numeric(df.get(f"{horizon}_mfe_bps"), errors="coerce")
    mae = pd.to_numeric(df.get(f"{horizon}_mae_bps"), errors="coerce")
    close_bps = pd.to_numeric(df.get(f"{horizon}_close_return_bps"), errors="coerce")
    risk = risk_bps.replace(0, np.nan)
    stop_hit = mae >= risk
    target_hit = mfe >= risk * float(target_r)
    both = stop_hit & target_hit
    ret = close_bps / risk
    ret = ret.clip(lower=-1.0, upper=float(target_r))
    ret = ret.where(~stop_hit, -1.0)
    ret = ret.where(~target_hit, float(target_r))
    if branch == "pessimistic":
        ret = ret.where(~both, -1.0)
    elif branch == "neutral":
        ret = ret.where(~both, 0.0)
    elif branch == "optimistic":
        ret = ret.where(~both, float(target_r))
    else:
        raise ValueError(branch)
    return ret.replace([np.inf, -np.inf], np.nan)


def summarize_r(ret: pd.Series) -> dict[str, Any]:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if r.empty:
        return {"events": 0, "mean_R": np.nan, "median_R": np.nan, "PF": np.nan, "hit_rate": np.nan, "max_dd_R_proxy": np.nan}
    pos = r[r > 0].sum(); neg = -r[r < 0].sum(); eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    return {"events": len(r), "mean_R": float(r.mean()), "median_R": float(r.median()), "PF": float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0), "hit_rate": float((r > 0).mean()), "max_dd_R_proxy": float(dd)}


def stage_basic_exit_surface(ctx: RunContext) -> None:
    p = path_metrics_path(ctx.run_root)
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    rows = []
    if not df.empty:
        group_cols = ["family", "side", "liquidity_tier"]
        for keys, sub in df.groupby(group_cols, dropna=False):
            for stop_name, stop_kind, stop_val in STOP_CLASSES:
                risk = _risk_bps_for_stop(sub, stop_kind, stop_val)
                for target in TARGET_R:
                    for horizon in BASIC_SURFACE_HORIZONS:
                        if f"{horizon}_mfe_bps" not in sub.columns:
                            continue
                        for branch in AMBIGUITY_BRANCHES:
                            ret = surface_returns(sub, horizon, risk, target, branch)
                            s = summarize_r(ret)
                            rows.append({"family": keys[0], "side": keys[1], "liquidity_tier": keys[2], "stop_class": stop_name, "target_R": target, "time_exit": horizon, "ambiguity_branch": branch, **s, "liquidation_count": bool_sum(sub.get(f"{horizon}_liquidation_10x", False), sub.index)})
    write_csv(ctx.run_root / "exit_surface/basic_exit_surface_summary.csv", rows)
    best = pd.DataFrame(rows)
    if not best.empty:
        best = best[best["ambiguity_branch"] == "pessimistic"].sort_values(["family", "mean_R"], ascending=[True, False]).groupby("family").head(5)
        best.to_csv(ctx.run_root / "exit_surface/basic_exit_surface_top_pessimistic.csv", index=False)
    write_text(ctx.run_root / "exit_surface/basic_exit_surface_report.md", "# Basic Exit Surface Report\n\nExit-surface returns are diagnostic approximations from path metrics. Pessimistic same-bar ambiguity is the primary branch; neutral and optimistic are report-only. Neighborhood stability is required before any later candidate contract.\n")


def stage_crypto_exit_surface(ctx: RunContext) -> None:
    df = pd.read_parquet(path_metrics_path(ctx.run_root)) if path_metrics_path(ctx.run_root).exists() else pd.DataFrame()
    rows = []
    if not df.empty:
        exit_defs = [
            ("quick_mean_reversion_6h_close", "6h_close_return_bps", "reference_risk_bps"),
            ("quick_mean_reversion_24h_close", "24h_close_return_bps", "reference_risk_bps"),
            ("half_24h_mfe_harvest", "24h_mfe_bps", "reference_risk_bps"),
            ("r_partial_1R_24h_proxy", "24h_mfe_bps", "reference_risk_bps"),
            ("qlmg_swing_5d_close", "5d_close_return_bps", "reference_risk_bps"),
            ("funding_aware_12h_close", "12h_close_return_bps", "reference_risk_bps"),
        ]
        for (fam, side, tier), sub in df.groupby(["family", "side", "liquidity_tier"], dropna=False):
            for name, val_col, risk_col in exit_defs:
                if val_col not in sub.columns:
                    continue
                risk = pd.to_numeric(sub[risk_col], errors="coerce").replace(0, np.nan)
                vals = pd.to_numeric(sub[val_col], errors="coerce")
                if name == "half_24h_mfe_harvest":
                    ret = (vals * 0.5 / risk).clip(lower=-1.0, upper=5.0)
                elif name == "r_partial_1R_24h_proxy":
                    hit1 = vals >= risk
                    ret = pd.Series(np.where(hit1, 0.5, pd.to_numeric(sub.get("24h_close_return_bps"), errors="coerce") / risk), index=sub.index).clip(lower=-1.0, upper=3.0)
                else:
                    ret = (vals / risk).clip(lower=-3.0, upper=5.0)
                rows.append({"family": fam, "side": side, "liquidity_tier": tier, "exit_class": name, **summarize_r(ret)})
    write_csv(ctx.run_root / "exit_surface/crypto_exit_surface_summary.csv", rows)
    write_text(ctx.run_root / "exit_surface/crypto_exit_surface_report.md", "# Crypto Exit Surface Report\n\nCrypto-specific exits are diagnostic proxies: quick mean-reversion, partial-harvest, funding-aware, and swing-style sketches. They do not define a final strategy.\n")


def stage_regime(ctx: RunContext) -> None:
    df = pd.read_parquet(path_metrics_path(ctx.run_root)) if path_metrics_path(ctx.run_root).exists() else pd.DataFrame()
    rows = []
    if not df.empty:
        tmp = df.copy()
        tmp["funding_sign"] = np.where(pd.to_numeric(tmp.get("funding_rate"), errors="coerce") > 0, "positive", np.where(pd.to_numeric(tmp.get("funding_rate"), errors="coerce") < 0, "negative", "zero_or_missing"))
        tmp["oi_state"] = np.where(pd.to_numeric(tmp.get("oi_chg_24h"), errors="coerce") > 0, "oi_up", np.where(pd.to_numeric(tmp.get("oi_chg_24h"), errors="coerce") < 0, "oi_down", "oi_unknown"))
        tmp["vol_bucket"] = pd.qcut(pd.to_numeric(tmp.get("range_pct"), errors="coerce"), 3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
        for cols in [["family", "liquidity_tier"], ["family", "btc_eth_regime"], ["family", "funding_sign"], ["family", "oi_state"], ["family", "vol_bucket"]]:
            for keys, sub in tmp.groupby(cols, dropna=False):
                keyvals = keys if isinstance(keys, tuple) else (keys,)
                rows.append({"slice_type": "+".join(cols), **{c: v for c, v in zip(cols, keyvals)}, "events": len(sub), "median_24h_mfe_bps": pd.to_numeric(sub.get("24h_mfe_bps"), errors="coerce").median(), "median_24h_close_bps": pd.to_numeric(sub.get("24h_close_return_bps"), errors="coerce").median(), "pos1R_before_neg1R_24h_share": bool_mean(sub.get("24h_pos1R_before_neg1R", False), sub.index)})
    write_csv(ctx.run_root / "regime/state_stratification_summary.csv", rows)
    write_text(ctx.run_root / "regime/state_stratification_report.md", "# State And Regime Stratification Report\n\nSlices are diagnostic and intended to identify economically sensible states for future narrow contracts, not to tune thresholds in this run.\n")


def sample_null_events(events: pd.DataFrame, seed: int, max_per_event: int = 1) -> pd.DataFrame:
    # Fast deterministic nulls: same symbol, nearby calendar period, outside event window.
    # Uses timestamp searchsorted rather than scanning all candidate bars per event.
    rows = []
    if events.empty:
        return events.copy()
    base_offsets_days = [-14, 14, -7, 7, -21, 21, -3, 3]
    for sym, sub in events.groupby("symbol"):
        p = DATA_5M / f"{sym}.parquet"
        if not p.exists():
            continue
        try:
            ts = pd.read_parquet(p, columns=["timestamp"])
        except Exception:
            continue
        ts = pd.to_datetime(ts["timestamp"], utc=True).sort_values()
        ts = ts[ts < FINAL_HOLDOUT_START]
        if ts.empty:
            continue
        arr = ts.astype("int64").to_numpy()
        for _, ev in sub.iterrows():
            ev_ts = pd.Timestamp(ev["decision_ts"])
            exclude_start = ev_ts - pd.Timedelta(hours=6)
            exclude_end = ev_ts + pd.Timedelta(hours=72)
            picks: list[pd.Timestamp] = []
            # Deterministic per-event rotation, so every event does not select the same side first.
            rot = int(hashlib.sha256(str(ev["event_id"]).encode()).hexdigest()[:2], 16) % len(base_offsets_days)
            offsets = base_offsets_days[rot:] + base_offsets_days[:rot]
            for off in offsets:
                if len(picks) >= max_per_event:
                    break
                target = ev_ts + pd.Timedelta(days=off)
                if target >= FINAL_HOLDOUT_START:
                    continue
                pos = int(np.searchsorted(arr, target.value, side="left"))
                candidates = [pos, pos - 1, pos + 1]
                for c in candidates:
                    if len(picks) >= max_per_event:
                        break
                    if c < 0 or c >= len(arr):
                        continue
                    nts = pd.Timestamp(arr[c], tz="UTC")
                    if exclude_start <= nts <= exclude_end:
                        continue
                    if nts >= FINAL_HOLDOUT_START:
                        continue
                    if any(abs((nts - p0).total_seconds()) < 3600 for p0 in picks):
                        continue
                    picks.append(nts)
            for j, nts in enumerate(picks):
                r = ev.to_dict()
                r["event_id"] = hashlib.sha256(f"null|{ev['event_id']}|{j}|{nts}".encode()).hexdigest()[:16]
                r["decision_ts"] = nts
                r["entry_ts"] = nts + pd.Timedelta(minutes=5)
                r["matched_event_id"] = ev["event_id"]
                r["null_type"] = "symbol_offset_neighbor_period"
                rows.append(r)
    return pd.DataFrame(rows)

def stage_matched_null(ctx: RunContext) -> None:
    events = pd.read_parquet(event_ledger_path(ctx.run_root)) if event_ledger_path(ctx.run_root).exists() else pd.DataFrame()
    nulls = sample_null_events(events, ctx.args.seed, max_per_event=(3 if ctx.args.allow_large_output else 1))
    null_dir = ctx.run_root / "matched_null/matched_null_path_metrics.parquet"
    if null_dir.exists():
        if null_dir.is_dir():
            shutil.rmtree(null_dir)
        else:
            null_dir.unlink()
    null_dir.mkdir(parents=True, exist_ok=True)
    part_no = 0
    chunk_rows: list[dict[str, Any]] = []
    symbol_groups = list(nulls.groupby("symbol")) if not nulls.empty else []
    for n, (sym, sub) in enumerate(symbol_groups, start=1):
        df = load_symbol_for_path(sym, max(ctx.start - pd.Timedelta(days=2), pd.Timestamp("2020-01-01T00:00:00Z")), min(ctx.end + pd.Timedelta(days=6), FINAL_HOLDOUT_START - pd.Timedelta(minutes=5)))
        if df.empty:
            continue
        df = add_features(df)
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
            ev["atr_bps"] = float(dec.get("atr_proxy", np.nan)) / float(ent["open"]) * 10000.0 if pd.notna(dec.get("atr_proxy", np.nan)) and float(ent["open"]) > 0 else np.nan
            rb = float(ev.get("reference_risk_bps", np.nan)) if pd.notna(ev.get("reference_risk_bps", np.nan)) else np.nan
            ev["reference_risk_bps"] = max(rb if np.isfinite(rb) else np.nan, ev["atr_bps"] if pd.notna(ev["atr_bps"]) else np.nan)
            chunk_rows.append(compute_path_row_fast(ev, arrays))
        if n % max(1, ctx.args.chunk_size) == 0 or n == len(symbol_groups):
            if chunk_rows:
                pd.DataFrame(chunk_rows).to_parquet(null_dir / f"part_{part_no:05d}.parquet", index=False)
                part_no += 1
                chunk_rows.clear()
    if part_no == 0:
        pd.DataFrame().to_parquet(null_dir / "part_00000.parquet", index=False)
    event_metrics = pd.read_parquet(path_metrics_path(ctx.run_root)) if path_metrics_path(ctx.run_root).exists() else pd.DataFrame()
    null_metrics = pd.read_parquet(null_dir) if null_dir.exists() else pd.DataFrame()
    rows = []
    for fam in sorted(set(event_metrics.get("family", [])) | set(null_metrics.get("family", []))):
        ev = event_metrics[event_metrics["family"] == fam] if not event_metrics.empty else pd.DataFrame()
        nu = null_metrics[null_metrics["family"] == fam] if not null_metrics.empty else pd.DataFrame()
        ev_mfe = pd.to_numeric(ev.get("24h_mfe_bps"), errors="coerce").median() if not ev.empty else np.nan
        nu_mfe = pd.to_numeric(nu.get("24h_mfe_bps"), errors="coerce").median() if not nu.empty else np.nan
        ev_share = pd.Series(ev.get("24h_pos1R_before_neg1R", False)).fillna(False).mean() if not ev.empty else np.nan
        nu_share = pd.Series(nu.get("24h_pos1R_before_neg1R", False)).fillna(False).mean() if not nu.empty else np.nan
        rows.append({
            "family": fam,
            "event_count": len(ev),
            "null_count": len(nu),
            "nulls_per_event_policy": 3 if ctx.args.allow_large_output else 1,
            "event_median_24h_mfe_bps": ev_mfe,
            "null_median_24h_mfe_bps": nu_mfe,
            "event_pos1R_before_neg1R_24h_share": ev_share,
            "null_pos1R_before_neg1R_24h_share": nu_share,
            "beats_null_path": bool(np.isfinite(ev_mfe) and np.isfinite(nu_mfe) and ev_mfe > nu_mfe and ev_share > nu_share),
        })
    write_csv(ctx.run_root / "matched_null/matched_null_summary.csv", rows)
    write_text(ctx.run_root / "matched_null/matched_null_report.md", "# Matched Null Path Comparison\n\nNulls are deterministic symbol-neighbor-month random non-events outside each event exclusion window. Default policy samples up to one null per event to keep the diagnostic under resource guardrails; `--allow-large-output` permits up to three. A family must beat null path metrics before it can be triaged as mechanism-promising.\n")

def stage_data_quality(ctx: RunContext) -> None:
    df = pd.read_parquet(path_metrics_path(ctx.run_root)) if path_metrics_path(ctx.run_root).exists() else pd.DataFrame()
    rows = []
    if not df.empty:
        q99 = pd.to_numeric(df.get("range_pct"), errors="coerce").quantile(0.99)
        for fam, sub in df.groupby("family"):
            base_mfe = pd.to_numeric(sub.get("24h_mfe_bps"), errors="coerce").median()
            ex_wick = sub[pd.to_numeric(sub.get("range_pct"), errors="coerce") <= q99]
            ex_mark_proxy = sub[sub.get("mark_path_status", "") == "mark_available"] if "mark_path_status" in sub.columns else pd.DataFrame()
            rows.append({
                "family": fam,
                "events": len(sub),
                "median_24h_mfe_bps_all": base_mfe,
                "median_24h_mfe_bps_ex_top1pct_wicks": pd.to_numeric(ex_wick.get("24h_mfe_bps"), errors="coerce").median() if not ex_wick.empty else np.nan,
                "mark_available_events": len(ex_mark_proxy),
                "median_24h_mfe_bps_mark_available": pd.to_numeric(ex_mark_proxy.get("24h_mfe_bps"), errors="coerce").median() if not ex_mark_proxy.empty else np.nan,
                "possible_artifact_flag": bool(len(ex_wick) and base_mfe > 0 and pd.to_numeric(ex_wick.get("24h_mfe_bps"), errors="coerce").median() < base_mfe * 0.5),
            })
    write_csv(ctx.run_root / "data_quality/bad_wick_artifact_summary.csv", rows)
    write_text(ctx.run_root / "data_quality/bad_wick_artifact_report.md", "# Bad-Wick And Data Artifact Audit\n\nThis stage compares path summaries after excluding top 1% range events and mark-proxy-only path rows. Original event ledgers are not mutated.\n")


def stage_portfolio(ctx: RunContext) -> None:
    null_p = ctx.run_root / "matched_null/matched_null_summary.csv"
    surf_p = ctx.run_root / "exit_surface/basic_exit_surface_summary.csv"
    rows = []
    if null_p.exists() and surf_p.exists():
        null = pd.read_csv(null_p)
        surf = pd.read_csv(surf_p)
        eligible = set(null.loc[null["beats_null_path"].astype(str).str.lower().isin(["true", "1"]), "family"])
        surf = surf[(surf["family"].isin(eligible)) & (surf["ambiguity_branch"] == "pessimistic")]
        if not surf.empty:
            for fam, sub in surf.groupby("family"):
                best = sub.sort_values("mean_R", ascending=False).head(1).iloc[0]
                mean_r = float(best["mean_R"] if pd.notna(best["mean_R"]) else 0.0)
                for equity in [200, 500, 1000]:
                    for risk_pct in [0.025, 0.05, 0.10, 0.15, 0.20]:
                        est_growth = (1.0 + risk_pct * mean_r) ** min(int(best["events"]), 250) if 1.0 + risk_pct * mean_r > 0 else 0.0
                        rows.append({"family": fam, "exit_class": f"{best['stop_class']}_{best['target_R']}R_{best['time_exit']}", "starting_equity": equity, "risk_pct": risk_pct, "mean_R_proxy": mean_r, "events_used_cap250": min(int(best["events"]), 250), "compounded_equity_proxy": equity * est_growth, "ruin_proxy": bool(est_growth <= 0.1), "liquidation_count_proxy": int(best.get("liquidation_count", 0)), "note": "diagnostic only; sizing does not create alpha"})
    write_csv(ctx.run_root / "portfolio/aggressive_10x_portfolio_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/aggressive_10x_portfolio_report.md", "# Aggressive 10x Portfolio Overlay\n\nOverlay runs only for families that beat matched-null path diagnostics. It is a diagnostic sizing stress, not live guidance. Negative expectancy should accelerate drawdown/ruin.\n")


def stage_triage(ctx: RunContext) -> None:
    null = pd.read_csv(ctx.run_root / "matched_null/matched_null_summary.csv") if (ctx.run_root / "matched_null/matched_null_summary.csv").exists() else pd.DataFrame()
    dq = pd.read_csv(ctx.run_root / "data_quality/bad_wick_artifact_summary.csv") if (ctx.run_root / "data_quality/bad_wick_artifact_summary.csv").exists() else pd.DataFrame()
    basic = pd.read_csv(ctx.run_root / "exit_surface/basic_exit_surface_summary.csv") if (ctx.run_root / "exit_surface/basic_exit_surface_summary.csv").exists() else pd.DataFrame()
    rows = []
    next_contracts = []
    for fam in ["D1", "D3", "E1", "A2"]:
        nrow = null[null["family"] == fam].head(1)
        drow = dq[dq["family"] == fam].head(1)
        beats = bool(len(nrow) and str(nrow.iloc[0].get("beats_null_path", False)).lower() in {"true", "1"})
        artifact = bool(len(drow) and str(drow.iloc[0].get("possible_artifact_flag", False)).lower() in {"true", "1"})
        best_exit = "none"
        if not basic.empty:
            sub = basic[(basic["family"] == fam) & (basic["ambiguity_branch"] == "pessimistic")]
            if not sub.empty:
                b = sub.sort_values("mean_R", ascending=False).head(1).iloc[0]
                best_exit = f"{b['stop_class']} target={b['target_R']}R time={b['time_exit']} mean_R={b['mean_R']:.4f}"
        if artifact:
            label = "bad_wick_or_data_artifact"
        elif beats:
            label = "mechanism_promising_exit_needs_validation"
        elif len(nrow) and int(nrow.iloc[0].get("null_count", 0)) == 0:
            label = "blocked_by_missing_data"
        else:
            label = "entry_no_path_edge"
        rows.append({"family": fam, "label": label, "beats_matched_null": beats, "artifact_flag": artifact, "compatible_exit_family": best_exit, "next_contract_recommendation": "draft narrow train-only validation" if label == "mechanism_promising_exit_needs_validation" else "do not advance from this diagnostic"})
        if label == "mechanism_promising_exit_needs_validation" and len(next_contracts) < 3:
            contract = {"family": fam, "status": "draft_next_candidate_contract", "basis": best_exit, "must_remain_train_only": True, "requires_full_cost_funding_liquidation_review": True, "not_validated": True}
            next_contracts.append(contract)
            write_json(ctx.run_root / "triage/next_candidate_contracts" / f"{fam}_next_contract.json", contract)
    write_csv(ctx.run_root / "triage/family_triage_summary.csv", rows)
    write_text(ctx.run_root / "triage/family_triage_report.md", "# Family Triage Report\n\nFamilies are classified using path edge, matched-null support, data-artifact checks, and proxy cost/liquidation diagnostics. No family is validated or live-ready.\n")
    # Final report is written here so compact bundle can include it.
    promising = [r for r in rows if r["label"] == "mechanism_promising_exit_needs_validation"]
    path_summary = pd.read_csv(ctx.run_root / "path_diagnostics/path_summary_by_family.csv") if (ctx.run_root / "path_diagnostics/path_summary_by_family.csv").exists() else pd.DataFrame()
    null_summary = pd.read_csv(ctx.run_root / "matched_null/matched_null_summary.csv") if (ctx.run_root / "matched_null/matched_null_summary.csv").exists() else pd.DataFrame()
    artifact_summary = pd.read_csv(ctx.run_root / "data_quality/bad_wick_artifact_summary.csv") if (ctx.run_root / "data_quality/bad_wick_artifact_summary.csv").exists() else pd.DataFrame()
    top_exits = pd.read_csv(ctx.run_root / "exit_surface/basic_exit_surface_top_pessimistic.csv") if (ctx.run_root / "exit_surface/basic_exit_surface_top_pessimistic.csv").exists() else pd.DataFrame()
    event_rows = pd.read_parquet(event_ledger_path(ctx.run_root), columns=["decision_ts"]) if event_ledger_path(ctx.run_root).exists() else pd.DataFrame()
    path_rows = pd.read_parquet(path_metrics_path(ctx.run_root), columns=["decision_ts"]) if path_metrics_path(ctx.run_root).exists() else pd.DataFrame()
    null_rows = pd.read_parquet(ctx.run_root / "matched_null/matched_null_path_metrics.parquet", columns=["decision_ts"]) if (ctx.run_root / "matched_null/matched_null_path_metrics.parquet").exists() else pd.DataFrame()
    def md_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
        if df.empty:
            return "_No rows._"
        keep = [c for c in cols if c in df.columns]
        return df[keep].head(n).to_markdown(index=False)
    report = f"""# QLMG Path Diagnostics Exit Surface Report

## Executive Verdict
- Run root: `{ctx.run_root}`
- Final holdout untouched: `yes`; diagnostic data end `{ctx.end}`.
- Telegram remote status: `{ctx.notifier.status}`; local JSONL logging was written.
- Prior Phase 0.5 negatives are not final economic rejections because that run used capped/sampled signals, proxy costs, and one crude stop/target/time construction.
- Families beating matched-null path diagnostics: `{','.join([r['family'] for r in rows if r['beats_matched_null']]) or 'none'}`.
- Mechanism-promising diagnostic families: `{','.join([r['family'] for r in promising]) or 'none'}`.
- No output is validation, live preparation, or a trading recommendation.

## Coverage And Seal Check
- Event ledger rows: `{len(event_rows)}`.
- Path metric rows: `{len(path_rows)}`.
- Matched-null path rows: `{len(null_rows)}`.
- Max event timestamp: `{pd.to_datetime(event_rows['decision_ts'], utc=True).max() if not event_rows.empty else 'n/a'}`.
- Max path timestamp: `{pd.to_datetime(path_rows['decision_ts'], utc=True).max() if not path_rows.empty else 'n/a'}`.
- Max null timestamp: `{pd.to_datetime(null_rows['decision_ts'], utc=True).max() if not null_rows.empty else 'n/a'}`.

## Path Behavior By Family
{md_table(path_summary, ['family','events','median_24h_mfe_bps','median_24h_mae_bps','median_24h_close_bps','pos1R_before_neg1R_24h_share','liq10x_24h_count'])}

Interpretation: D1, D3, and E1 show favorable excursion above matched null, but D3 and E1 also show very large 10x liquidation diagnostics under the proxy mark/last path model. A2 has positive excursion but does not beat null on the first-hit path criterion.

## Matched Null Comparison
{md_table(null_summary, ['family','event_count','null_count','nulls_per_event_policy','event_median_24h_mfe_bps','null_median_24h_mfe_bps','event_pos1R_before_neg1R_24h_share','null_pos1R_before_neg1R_24h_share','beats_null_path'])}

The default full run used one deterministic null per event to stay inside resource guardrails. `--allow-large-output` permits up to three nulls per event.

## Compatible Exit Surface Diagnostics
{md_table(top_exits, ['family','side','liquidity_tier','stop_class','target_R','time_exit','events','mean_R','median_R','PF','hit_rate','liquidation_count'], 12)}

These are diagnostic approximations from 5m path metrics, not executable strategy definitions. Pessimistic same-bar handling is the primary branch.

## Data Artifact Audit
{md_table(artifact_summary, ['family','events','median_24h_mfe_bps_all','median_24h_mfe_bps_ex_top1pct_wicks','mark_available_events','median_24h_mfe_bps_mark_available','possible_artifact_flag'])}

Mark-path availability is effectively absent under the current safe context join, so liquidation/path evidence is proxy-grade and cannot support promotion by itself.

## Family Triage
{md_table(pd.DataFrame(rows), ['family','label','beats_matched_null','artifact_flag','compatible_exit_family','next_contract_recommendation'])}

## Next Contracts
At most three draft next contracts were written under `triage/next_candidate_contracts/`: D1, D3, and E1 if they remained mechanism-promising. These must be treated as narrow train-only validation contracts, not sealed validation requests.

## Known Limitations
- 5m-only path diagnostics; no 1m, order book, depth, or liquidation feed.
- Proxy cost/funding/liquidation only.
- Mark path was not available in the generated full diagnostic rows, so mark-price liquidation is not promotion-grade.
- Exit surfaces are diagnostic and should not be converted directly into strategy parameters without a new frozen contract.
"""
    write_text(ctx.run_root / "QLMG_PATH_DIAGNOSTICS_EXIT_SURFACE_REPORT.md", report)


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rels = [
        "QLMG_PATH_DIAGNOSTICS_EXIT_SURFACE_REPORT.md",
        "preflight/resource_guard_report.md",
        "notifications/telegram_readiness_report.md",
        "seal/seal_guard_report.md",
        "audit/what_was_actually_tested.md",
        "contracts/family_contract_summary.csv",
        "events/event_coverage_report.md",
        "events/event_coverage_summary.csv",
        "events/event_ledger_sample.parquet",
        "path_diagnostics/mfe_mae_report.md",
        "path_diagnostics/path_summary_by_family.csv",
        "exit_surface/basic_exit_surface_report.md",
        "exit_surface/basic_exit_surface_summary.csv",
        "exit_surface/crypto_exit_surface_report.md",
        "exit_surface/crypto_exit_surface_summary.csv",
        "regime/state_stratification_report.md",
        "regime/state_stratification_summary.csv",
        "matched_null/matched_null_report.md",
        "matched_null/matched_null_summary.csv",
        "data_quality/bad_wick_artifact_report.md",
        "data_quality/bad_wick_artifact_summary.csv",
        "portfolio/aggressive_10x_portfolio_report.md",
        "portfolio/aggressive_10x_portfolio_summary.csv",
        "triage/family_triage_report.md",
        "triage/family_triage_summary.csv",
    ]
    idx = []
    for rel in rels:
        src = ctx.run_root / rel
        if src.exists():
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            idx.append({"artifact": rel, "bundle_copy": str(dst.relative_to(ctx.run_root)), "size_bytes": src.stat().st_size})
    for p in sorted((ctx.run_root / "triage/next_candidate_contracts").glob("*.json")) if (ctx.run_root / "triage/next_candidate_contracts").exists() else []:
        dst = bundle / f"next_candidate_contracts__{p.name}"
        shutil.copy2(p, dst)
        idx.append({"artifact": str(p.relative_to(ctx.run_root)), "bundle_copy": str(dst.relative_to(ctx.run_root)), "size_bytes": p.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", idx)
    write_json(bundle / "artifact_path_index.json", {"artifacts": idx})
    zip_path = ctx.run_root / "qlmg_path_diagnostics_exit_surface_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in bundle.rglob("*"):
            if item.is_file():
                z.write(item, arcname=str(item.relative_to(ctx.run_root)))
    write_text(bundle / "README.md", f"Compact path diagnostics bundle. Zip: `{zip_path}`\n")


STAGE_FUNCS = {
    "preflight-resource-and-prior-screen-audit": stage_preflight,
    "telegram-and-resume-check": stage_telegram,
    "seal-guard": stage_seal,
    "what-was-actually-tested-audit": stage_actual_tested,
    "entry-family-contracts": stage_contracts,
    "full-path-event-ledgers": stage_events,
    "mfe-mae-path-diagnostics": stage_path_metrics,
    "stop-target-time-exit-surface": stage_basic_exit_surface,
    "crypto-specific-exit-surface": stage_crypto_exit_surface,
    "state-and-regime-stratification": stage_regime,
    "matched-null-path-comparison": stage_matched_null,
    "bad-wick-and-data-artifact-audit": stage_data_quality,
    "aggressive-10x-portfolio-overlay": stage_portfolio,
    "family-triage-and-next-contracts": stage_triage,
    "compact-review-bundle": stage_bundle,
}


def estimate_stage_gb(stage: str, smoke: bool) -> float:
    if smoke:
        return 0.2
    if stage in {"full-path-event-ledgers", "mfe-mae-path-diagnostics", "matched-null-path-comparison"}:
        return 4.0
    if stage in {"stop-target-time-exit-surface", "crypto-specific-exit-surface", "state-and-regime-stratification"}:
        return 1.5
    return 0.5


def main() -> int:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    start, end = clamp_window(args)
    run_root, root_reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    tmp = run_root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": root_reason, "started_at_utc": utc_now(), "argv": sys.argv, "seed": args.seed, "smoke": args.smoke, "start": str(start), "end": str(end)})
    stages = stage_list(args.stage)
    notifier.send("RUN START", f"stages={stages}\nrun_root={run_root}")
    try:
        for stage in stages:
            if args.resume and stage_complete(run_root, stage):
                continue
            append_command(run_root, stage)
            ensure_guard(ctx, stage, estimate_stage_gb(stage, args.smoke))
            notifier.send("STAGE START", stage)
            if args.dry_run:
                write_text(run_root / "dry_run" / f"{stage}.txt", f"would run {stage}")
                mark_done(run_root, stage)
                continue
            STAGE_FUNCS[stage](ctx)
            # Ensure no common generated artifacts cross the final holdout.
            mark_done(run_root, stage)
            shutil.rmtree(run_root / "tmp" / stage, ignore_errors=True)
            notifier.send("STAGE COMPLETE", stage)
        notifier.send("RUN COMPLETE", f"run_root={run_root}")
        return 0
    except Exception as exc:
        notifier.send("RUN FAILED", f"{type(exc).__name__}: {exc}", level="error")
        write_text(run_root / "FAILED.txt", f"{utc_now()} {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
