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
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Converting to PeriodArray.*")
warnings.filterwarnings("ignore", message="Converting to PeriodArray.*", category=UserWarning)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_regime_stack import (  # noqa: E402
    FINAL_HOLDOUT_START,
    SCREENING_END,
    REGIME_LAYERS,
    build_regime_panel,
    join_regime,
    label_overlap_matrix,
    regime_feature_dictionary,
    stable_hash,
    validate_no_protected,
)
from tools.qlmg_strategy_contracts import allocate_sweep_budget, write_contracts  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_regime_stack_and_smart_sweep_20260625_v1"
FIRST_SCREEN_ROOT = REPO / "results/rebaseline/phase_qlmg_engine_and_first_screen_20260624_v1_20260624_101747"
PATH_DIAG_ROOT = REPO / "results/rebaseline/phase_qlmg_path_diagnostics_exit_surface_20260624_v1_20260624_121522"
D1_ROOT = REPO / "results/rebaseline/phase_qlmg_d1_narrow_validation_20260624_v1"
PILOT_1M_ROOT = REPO / "results/rebaseline/phase_qlmg_targeted_1m_data_pilot_20260624_v1"

STAGES = (
    "preflight-and-prior-artifact-audit",
    "telegram-and-tmux-setup",
    "seal-guard",
    "regime-stack-contract",
    "regime-feature-builder",
    "regime-label-qc-and-causality-audit",
    "strategy-family-contracts-with-regime-preconditions",
    "conditional-performance-cubes",
    "smart-sobol-regime-entry-exit-sweep",
    "local-neighborhood-refinement",
    "matched-null-and-regime-uplift-validation",
    "targeted-1m-overlay-for-shortlisted-candidates",
    "walk-forward-cpcv-and-overfit-controls",
    "aggressive-10x-regime-conditioned-portfolio-overlay",
    "strategy-regime-triage-and-next-contracts",
    "sector-catalyst-data-scaffold",
    "decision-report",
    "compact-review-bundle",
    "all",
)

ALLOWED_VERDICTS = {
    "promote_one_or_more_to_family_specific_validation",
    "promote_to_targeted_execution_data_collection",
    "continue_regime_stack_development",
    "pivot_to_sector_catalyst_data_build",
    "reject_current_regime_conditioned_translations",
    "blocked_by_data_or_execution",
    "blocked_by_protocol_issue",
}

HORIZONS = ["30m", "1h", "2h", "4h", "6h", "12h", "24h", "48h", "72h"]
CORE_PATH_COLS = [
    "event_id", "family", "variant_id", "symbol", "side", "liquidity_tier", "decision_ts", "entry_ts",
    "entry_ref_price", "reference_risk_bps", "atr_bps", "btc_eth_regime", "oi_chg_24h", "funding_rate",
    "turnover", "range_pct", "data_quality_flags", "mark_path_status", "liq_price_10x",
]
for _h in HORIZONS:
    CORE_PATH_COLS.extend([f"{_h}_path_available", f"{_h}_mfe_bps", f"{_h}_mae_bps", f"{_h}_close_return_bps", f"{_h}_pos1R_before_neg1R", f"{_h}_liquidation_10x"])


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
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-regime-sweep")
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
        try:
            watch = {"ts_utc": rec["ts_utc"], "status": "running", "last_event": title, "last_body": body, "run_root": str(self.run_root)}
            (self.run_root / "watch_status.json").write_text(json.dumps(watch, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        return sent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG regime stack and smart regime-conditioned sweep")
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
    p.add_argument("--sweep-budget", type=int, default=360)
    p.add_argument("--local-refine-budget", type=int, default=72)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--use-targeted-1m", action="store_true")
    p.add_argument("--tmux-session-name", default="qlmg_regime_sweep")
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


def done_path(run_root: Path, stage: str) -> Path:
    return run_root / "stage_status" / f"{stage}.done"


def mark_done(run_root: Path, stage: str) -> None:
    done_path(run_root, stage).parent.mkdir(parents=True, exist_ok=True)
    done_path(run_root, stage).write_text(utc_now() + "\n", encoding="utf-8")


def append_command(run_root: Path, stage: str) -> None:
    rec = {"ts_utc": utc_now(), "stage": stage, "argv": sys.argv}
    path = run_root / "command_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def required_outputs_for_stage(run_root: Path, stage: str) -> list[Path]:
    m = {
        "preflight-and-prior-artifact-audit": [run_root / "preflight/preflight_report.md", run_root / "preflight/prior_artifact_manifest.json"],
        "telegram-and-tmux-setup": [run_root / "notifications/telegram_readiness_report.md", run_root / "tmux/watch_commands.md"],
        "seal-guard": [run_root / "seal/seal_guard_report.md", run_root / "seal/protected_slice_check.json"],
        "regime-stack-contract": [run_root / "regime_contract/regime_stack_contract.json", run_root / "regime_contract/regime_feature_dictionary.csv"],
        "regime-feature-builder": [run_root / "regime_features/regime_feature_panel.parquet", run_root / "regime_features/regime_feature_coverage_summary.csv"],
        "regime-label-qc-and-causality-audit": [run_root / "regime_qc/regime_label_qc_report.md", run_root / "regime_qc/regime_label_frequency.csv"],
        "strategy-family-contracts-with-regime-preconditions": [run_root / "contracts/strategy_regime_contract_summary.csv"],
        "conditional-performance-cubes": [run_root / "cubes/conditional_performance_cube.parquet", run_root / "cubes/conditional_cube_summary.csv"],
        "smart-sobol-regime-entry-exit-sweep": [run_root / "sweep/sweep_candidate_registry.csv", run_root / "sweep/sweep_results.parquet"],
        "local-neighborhood-refinement": [run_root / "refinement/refinement_registry.csv", run_root / "refinement/refinement_summary.csv"],
        "matched-null-and-regime-uplift-validation": [run_root / "matched_null/matched_null_validation_summary.csv"],
        "targeted-1m-overlay-for-shortlisted-candidates": [run_root / "one_minute/one_minute_overlay_summary.csv"],
        "walk-forward-cpcv-and-overfit-controls": [run_root / "validation/walk_forward_summary.csv", run_root / "validation/cpcv_summary.csv"],
        "aggressive-10x-regime-conditioned-portfolio-overlay": [run_root / "portfolio/aggressive_10x_portfolio_summary.csv"],
        "strategy-regime-triage-and-next-contracts": [run_root / "triage/family_triage_summary.csv"],
        "sector-catalyst-data-scaffold": [run_root / "sector_catalyst/sector_catalyst_data_scaffold.md"],
        "decision-report": [run_root / "QLMG_REGIME_STACK_AND_SMART_SWEEP_REPORT.md", run_root / "decision_summary.json"],
        "compact-review-bundle": [run_root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return m.get(stage, [])


def stage_complete(run_root: Path, stage: str) -> bool:
    return done_path(run_root, stage).exists() and all(p.exists() for p in required_outputs_for_stage(run_root, stage))


def estimate_stage_gb(stage: str, smoke: bool) -> float:
    base = {
        "regime-feature-builder": 0.5,
        "conditional-performance-cubes": 0.6,
        "smart-sobol-regime-entry-exit-sweep": 0.8,
        "matched-null-and-regime-uplift-validation": 0.7,
        "walk-forward-cpcv-and-overfit-controls": 0.4,
    }.get(stage, 0.1)
    return min(base, 0.2) if smoke else base


def ensure_guard(ctx: RunContext, stage: str, estimate_gb: float) -> None:
    snap = resource_snapshot(REPO)
    status = check_resource_guard(snap, estimated_output_gb=estimate_gb, allow_large_output=ctx.args.allow_large_output)
    path = ctx.run_root / "resource_guard" / f"{stage}.json"
    write_json(path, {"stage": stage, **status})
    if status["warnings"]:
        ctx.notifier.send(f"QLMG regime sweep resource warning: {stage}", json.dumps(status), level="warning")
    if status["status"] != "pass":
        raise RuntimeError(f"resource guard failed for {stage}: {status}")


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = __import__("hashlib").sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_text_safe(path: Path, limit: int = 4000) -> str:
    return path.read_text(encoding="utf-8")[:limit] if path.exists() else ""


def path_metrics_path(null: bool = False) -> Path:
    if null:
        return PATH_DIAG_ROOT / "matched_null/matched_null_path_metrics.parquet"
    return PATH_DIAG_ROOT / "path_diagnostics/path_metrics.parquet"


def load_path_metrics(ctx: RunContext, *, null: bool = False) -> pd.DataFrame:
    path = path_metrics_path(null)
    if not path.exists():
        return pd.DataFrame()
    cols = CORE_PATH_COLS
    # Keep only columns physically present in older parquet schemas.
    try:
        import pyarrow.parquet as pq
        schema = pq.ParquetDataset(path).schema
        available = set(schema.names)
        cols = [c for c in cols if c in available]
    except Exception:
        pass
    df = pd.read_parquet(path, columns=cols)
    for c in ["decision_ts", "entry_ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df[(df["decision_ts"] >= ctx.start) & (df["decision_ts"] <= ctx.end)]
    if ctx.args.max_symbols:
        symbols = sorted(df["symbol"].dropna().unique())[: ctx.args.max_symbols]
        df = df[df["symbol"].isin(symbols)]
    if ctx.args.smoke:
        df = df.sort_values(["family", "symbol", "decision_ts"]).groupby("family", group_keys=False).head(5000)
    validate_no_protected(df, ["decision_ts", "entry_ts"])
    return df.reset_index(drop=True)


def cost_bps_for_tier(tier: Any) -> float:
    return {"A": 16.0, "B": 30.0, "C": 50.0, "D": 78.0}.get(str(tier), 93.0)


def surface_return_r(df: pd.DataFrame, horizon: str, target_r: float, stop_mult: float = 1.0, cost_mult: float = 1.0, branch: str = "pessimistic") -> pd.Series:
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
    out = df
    fam = cand.get("family")
    if fam == "D4":
        out = out[out["family"].isin(["D3", "E1"])]
    elif fam == "A1_A3":
        out = out[out["family"].eq("A2")]
    elif fam in {"F1", "G1"}:
        return out.iloc[0:0].copy()
    else:
        out = out[out["family"].eq(fam)]
    tier = cand.get("tier_filter", "any")
    if tier == "C":
        out = out[out.get("liquidity_tier").eq("C")]
    elif tier == "A_B":
        out = out[out.get("liquidity_tier").isin(["A", "B"])]
    if cand.get("parent_gate") == "non_deteriorating" and "btc_eth_non_deteriorating" in out.columns:
        out = out[out["btc_eth_non_deteriorating"].fillna(False)]
    elif cand.get("parent_gate") == "up_or_neutral" and "parent_trend_label" in out.columns:
        out = out[out["parent_trend_label"].isin(["strong_up", "neutral_up"])]
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
    return out.copy()


def candidate_result(df: pd.DataFrame, cand: Mapping[str, Any]) -> dict[str, Any]:
    sub = apply_candidate_filter(df, cand)
    horizon = str(cand.get("horizon", "24h"))
    ret = surface_return_r(sub, horizon, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
    sm = summarize_returns(ret)
    liq_col = f"{horizon}_liquidation_10x"
    liq = int(sub.get(liq_col, pd.Series(False, index=sub.index)).fillna(False).astype(bool).sum()) if not sub.empty else 0
    pos = ret[ret > 0]
    if not sub.empty and not pos.empty:
        tmp = sub.loc[pos.index].assign(_positive_R=pos)
        sym_share = float(tmp.groupby("symbol")['_positive_R'].sum().max() / max(float(pos.sum()), 1e-9)) if "symbol" in tmp.columns else 0.0
        month_share = float(tmp.groupby(pd.to_datetime(tmp["decision_ts"], utc=True).dt.to_period("M"))["_positive_R"].sum().max() / max(float(pos.sum()), 1e-9))
    else:
        sym_share = 0.0
        month_share = 0.0
    score = sm["mean_R"] * 35 + sm["PF"] * 5 + sm["hit_rate"] * 10 - liq * 0.02 - sym_share * 5 - month_share * 3 + math.log1p(sm["events"]) * 0.5
    return {**cand, **sm, "liquidation_count": liq, "max_symbol_positive_share": sym_share, "max_month_positive_share": month_share, "robustness_score": float(score), "evaluation_status": "ok" if sm["events"] > 0 else "no_events"}


def generate_candidates(budget: int, seed: int, include_shorts: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    alloc = allocate_sweep_budget(budget, include_shorts=include_shorts)
    families_by_group = {"D3_D4_E1": ["D3", "D4", "E1"], "A1_A3": ["A1_A3"], "A2": ["A2"], "F1_G1": ["F1", "G1"]}
    rows: list[dict[str, Any]] = []
    # Explicit no-regime baselines first.
    for fam in ["D3", "D4", "E1", "A1_A3", "A2"] + (["F1", "G1"] if include_shorts else []):
        rows.append({"candidate_id": "", "family": fam, "candidate_type": "no_regime_baseline", "tier_filter": "any", "parent_gate": "none", "deleveraged_gate": False, "funding_gate": "none", "price_oi_gate": "none", "liquidity_quality_gate": "none", "horizon": "24h", "target_r": 2.0, "stop_mult": 1.0, "cost_mult": 1.0})
    for group, count in alloc.items():
        fams = families_by_group[group]
        for _ in range(max(0, count)):
            fam = rng.choice(fams)
            rows.append({
                "candidate_id": "",
                "family": fam,
                "candidate_type": "regime_conditioned",
                "tier_filter": rng.choice(["C", "A_B", "any"] if fam in {"A2", "A1_A3"} else ["C", "any"]),
                "parent_gate": rng.choice(["none", "non_deteriorating", "up_or_neutral"]),
                "deleveraged_gate": rng.choice([False, True]) if fam in {"D3", "D4", "E1"} else False,
                "funding_gate": rng.choice(["none", "not_high", "low_mid"]),
                "price_oi_gate": rng.choice(["none", "price_down_oi_down", "not_price_down_oi_up"]),
                "liquidity_quality_gate": rng.choice(["none", "avoid_thin"]),
                "horizon": rng.choice(["30m", "1h", "6h", "24h", "72h"]),
                "target_r": rng.choice([1.0, 2.0, 3.0, 5.0]),
                "stop_mult": rng.choice([0.5, 1.0, 1.5, 2.0]),
                "cost_mult": rng.choice([1.0, 1.25, 1.5]),
            })
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = {k: v for k, v in row.items() if k != "candidate_id"}
        cid = f"{row['family']}__{stable_hash(payload, 12)}"
        if cid in seen:
            continue
        seen.add(cid)
        row = dict(row)
        row["candidate_id"] = cid
        out.append(row)
    return out[:budget + 8]


def write_df_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    roots = [FIRST_SCREEN_ROOT, PATH_DIAG_ROOT, D1_ROOT, PILOT_1M_ROOT]
    manifest = []
    for root in roots:
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        manifest.append({"root": str(root), "exists": root.exists(), "file_count": len(files), "sample_files": [str(p.relative_to(root)) for p in files[:20]]})
    write_json(ctx.run_root / "preflight/prior_artifact_manifest.json", {"artifacts": manifest})
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard Report\n\n- free_disk_gb: `{snap.free_gb:.2f}`\n- hard_stop_free_gb: `5`\n- warning_free_gb: `7`\n- max_stage_output_without_override_gb: `20`\n- run_window: `{ctx.start}` to `{ctx.end}`\n")
    interp = """# Prior Result Interpretation

- D1 current translation is closed and is not rerun except as a negative-control reference.
- D3/D4/E1 remain plausible but require regime-aware validation.
- A2 unconditional screen failed, but A1/A2/A3 liquid continuation may still need parent-trend/breadth regime filtering.
- Final holdout starts `2026-01-01T00:00:00Z`; no final holdout access is allowed.
- Public trades, top-of-book/depth, and true liquidation feeds remain missing; execution evidence is not promotion-grade.
"""
    write_text(ctx.run_root / "preflight/prior_result_interpretation.md", interp)
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight Report\n\n- first screen root: `{FIRST_SCREEN_ROOT}`\n- path diagnostic root: `{PATH_DIAG_ROOT}`\n- D1 validation root: `{D1_ROOT}`\n- targeted 1m pilot root: `{PILOT_1M_ROOT}`\n- current free disk GB: `{snap.free_gb:.2f}`\n- final holdout cutoff: `{FINAL_HOLDOUT_START}`\n- previous screens were diagnostic/screening-grade and not conclusive for live readiness.\n")


def stage_telegram_tmux(ctx: RunContext) -> None:
    missing = ctx.notifier.missing or "none"
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness Report\n\n- status: `{ctx.notifier.status}`\n- missing/disabled reason: `{missing}`\n- local log: `notifications/telegram_events.jsonl`\n- secrets persisted: `false`\n")
    watch = f"""# Watch Commands

- `tmux attach -t {ctx.args.tmux_session_name}`
- `tail -f {ctx.run_root}/logs/full_run.log`
- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`
- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`
- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`
"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nUse `bash tools/run_qlmg_regime_stack_and_smart_sweep_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --sweep-budget {ctx.args.sweep_budget} --local-refine-budget {ctx.args.local_refine_budget} --nulls-per-event {ctx.args.nulls_per_event} --use-targeted-1m --seed {ctx.args.seed}`.\n")


def stage_seal(ctx: RunContext) -> None:
    check = {
        "protected_start": str(FINAL_HOLDOUT_START),
        "allowed_end": str(SCREENING_END),
        "requested_start": str(ctx.start),
        "requested_end": str(ctx.end),
        "pre_holdout_read_smoke": bool(ctx.end < FINAL_HOLDOUT_START),
        "protected_read_smoke": "blocked_by_policy",
        "status": "pass" if ctx.end < FINAL_HOLDOUT_START else "fail",
    }
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    if check["status"] != "pass":
        raise RuntimeError("seal guard failed")
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard Report\n\n- protected slice starts: `{FINAL_HOLDOUT_START}`\n- this run may use data through: `{SCREENING_END}`\n- requested end: `{ctx.end}`\n- generated ledger protected-row checks: `required at every stage`\n- status: `pass`\n")


def stage_regime_contract(ctx: RunContext) -> None:
    write_json(ctx.run_root / "regime_contract/regime_stack_contract.json", {"contract_frozen": True, "created_at_utc": utc_now(), "protected_holdout_start": str(FINAL_HOLDOUT_START), "layers": REGIME_LAYERS, "pit_rules": ["feature_ts <= decision_ts", "rolling percentiles use trailing rows only", "cross-sectional labels cannot use current-active future universe", "missing required labels fail closed or become diagnostics-only"]})
    write_csv(ctx.run_root / "regime_contract/regime_layer_summary.csv", REGIME_LAYERS)
    regime_feature_dictionary().to_csv(ctx.run_root / "regime_contract/regime_feature_dictionary.csv", index=False)


def stage_regime_features(ctx: RunContext) -> None:
    df = load_path_metrics(ctx, null=False)
    reg = build_regime_panel(df, min_history=3 if ctx.args.smoke else 10)
    (ctx.run_root / "regime_features").mkdir(parents=True, exist_ok=True)
    reg.to_parquet(ctx.run_root / "regime_features/regime_feature_panel.parquet", index=False)
    summary = []
    for col in ["parent_trend_label", "btc_eth_regime_label", "realized_vol_bucket", "liquidity_quality_label", "funding_percentile_bucket", "price_oi_matrix_24h", "deleveraged_2of4", "session_bucket", "listing_age_bucket"]:
        if col in reg.columns:
            vc = reg[col].astype(str).value_counts(dropna=False).head(20)
            for label, count in vc.items():
                summary.append({"feature": col, "label": label, "rows": int(count), "share": float(count / max(len(reg), 1))})
    write_csv(ctx.run_root / "regime_features/regime_feature_coverage_summary.csv", summary)
    write_text(ctx.run_root / "regime_features/regime_feature_builder_report.md", f"# Regime Feature Builder Report\n\n- source path metrics rows: `{len(df)}`\n- regime rows: `{len(reg)}`\n- source: prior path diagnostics event/path rows, not final holdout.\n- sector/catalyst labels: `schema_only_blocked_no_pit_store`\n")


def load_regime(ctx: RunContext) -> pd.DataFrame:
    path = ctx.run_root / "regime_features/regime_feature_panel.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_event_regime(ctx: RunContext, *, null: bool = False) -> pd.DataFrame:
    df = load_path_metrics(ctx, null=null)
    if null:
        reg = build_regime_panel(df, min_history=3 if ctx.args.smoke else 10)
    else:
        reg = load_regime(ctx)
        if reg.empty:
            reg = build_regime_panel(df, min_history=3 if ctx.args.smoke else 10)
    return join_regime(df, reg)


def stage_regime_qc(ctx: RunContext) -> None:
    reg = load_regime(ctx)
    violations = []
    if not reg.empty:
        if (pd.to_datetime(reg["feature_ts"], utc=True) > pd.to_datetime(reg["decision_ts"], utc=True)).any():
            violations.append({"violation": "feature_ts_after_decision_ts", "severity": "major"})
        if (pd.to_datetime(reg["decision_ts"], utc=True) >= FINAL_HOLDOUT_START).any():
            violations.append({"violation": "protected_timestamp", "severity": "major"})
    freq = []
    for col in ["parent_trend_label", "realized_vol_bucket", "liquidity_quality_label", "funding_percentile_bucket", "price_oi_matrix_24h", "session_bucket"]:
        if col in reg.columns:
            for label, count in reg[col].astype(str).value_counts(dropna=False).items():
                freq.append({"feature": col, "label": label, "rows": int(count), "share": float(count / max(len(reg), 1))})
    write_csv(ctx.run_root / "regime_qc/regime_label_frequency.csv", freq)
    overlap = label_overlap_matrix(reg, ["parent_trend_label", "realized_vol_bucket", "funding_percentile_bucket", "price_oi_matrix_24h", "session_bucket"])
    write_df_csv(ctx.run_root / "regime_qc/regime_label_overlap_matrix.csv", overlap)
    write_csv(ctx.run_root / "regime_qc/causality_violations.csv", violations, fieldnames=["violation", "severity"])
    write_text(ctx.run_root / "regime_qc/regime_label_qc_report.md", f"# Regime Label QC Report\n\n- regime rows: `{len(reg)}`\n- causality violations: `{len(violations)}`\n- major timestamp leakage: `{any(v.get('severity') == 'major' for v in violations)}`\n- stale/missing sidecar handling: labels retain explicit unknown/flagged states.\n")
    if any(v.get("severity") == "major" for v in violations):
        raise RuntimeError("major regime timestamp leakage detected")


def stage_strategy_contracts(ctx: RunContext) -> None:
    families = ["D3", "D4", "E1", "A1_A3", "A2", "F1", "G1"]
    rows = write_contracts(ctx.run_root, families)
    write_csv(ctx.run_root / "contracts/strategy_regime_contract_summary.csv", rows)


def stage_cubes(ctx: RunContext) -> None:
    df = load_event_regime(ctx)
    dims = ["family", "side", "liquidity_tier", "parent_trend_label", "realized_vol_bucket", "liquidity_quality_label", "price_oi_matrix_24h", "funding_percentile_bucket", "deleveraged_2of4", "session_bucket", "listing_age_bucket"]
    dims = [d for d in dims if d in df.columns]
    grouped = df.groupby(dims, dropna=False)
    rows = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(dims, keys))
        rec.update({
            "events": int(len(sub)),
            "mean_24h_mfe_bps": float(pd.to_numeric(sub.get("24h_mfe_bps"), errors="coerce").mean()),
            "median_24h_mfe_bps": float(pd.to_numeric(sub.get("24h_mfe_bps"), errors="coerce").median()),
            "mean_24h_mae_bps": float(pd.to_numeric(sub.get("24h_mae_bps"), errors="coerce").mean()),
            "pos1R_before_neg1R_24h_share": float(sub.get("24h_pos1R_before_neg1R", pd.Series(False, index=sub.index)).fillna(False).astype(bool).mean()),
            "liq10x_24h_count": int(sub.get("24h_liquidation_10x", pd.Series(False, index=sub.index)).fillna(False).astype(bool).sum()),
        })
        rows.append(rec)
    cube = pd.DataFrame(rows).sort_values(["events", "mean_24h_mfe_bps"], ascending=[False, False]) if rows else pd.DataFrame()
    (ctx.run_root / "cubes").mkdir(parents=True, exist_ok=True)
    cube.to_parquet(ctx.run_root / "cubes/conditional_performance_cube.parquet", index=False)
    summary = cube.head(500) if not cube.empty else cube
    write_df_csv(ctx.run_root / "cubes/conditional_cube_summary.csv", summary)
    write_text(ctx.run_root / "cubes/conditional_cube_report.md", f"# Conditional Cube Report\n\n- joined event rows: `{len(df)}`\n- cube rows: `{len(cube)}`\n- labels with rare support should be diagnostics-only unless event counts are adequate.\n- A1/A3 lacks a true event ledger and is treated as proxy/limited in later sweep stages.\n")


def evaluate_candidates(df: pd.DataFrame, candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [candidate_result(df, c) for c in candidates]
    return pd.DataFrame(rows)


def stage_sweep(ctx: RunContext) -> None:
    df = load_event_regime(ctx)
    candidates = generate_candidates(ctx.args.sweep_budget if not ctx.args.smoke else min(ctx.args.sweep_budget, 20), ctx.args.seed, include_shorts=False)
    write_csv(ctx.run_root / "sweep/sweep_candidate_registry.csv", candidates)
    res = evaluate_candidates(df, candidates)
    (ctx.run_root / "sweep").mkdir(parents=True, exist_ok=True)
    res.to_parquet(ctx.run_root / "sweep/sweep_results.parquet", index=False)
    write_df_csv(ctx.run_root / "sweep/sweep_summary.csv", res.sort_values("robustness_score", ascending=False).head(300))
    fam_counts = res.groupby("family")["candidate_id"].count().to_dict() if not res.empty else {}
    write_text(ctx.run_root / "sweep/sweep_report.md", f"# Sweep Report\n\n- candidates logged: `{len(candidates)}`\n- candidates evaluated: `{len(res)}`\n- family counts: `{fam_counts}`\n- F1/G1: `contract_only_data_blocked_no_safe_event_generation_in_this_run`\n- A1/A3: `proxy_from_A2_event_set_until_true_event_ledger_exists`\n")


def top_shortlist(ctx: RunContext, max_n: int = 5) -> pd.DataFrame:
    paths = [ctx.run_root / "refinement/refinement_summary.csv", ctx.run_root / "sweep/sweep_summary.csv"]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                return df.sort_values("robustness_score", ascending=False).head(max_n)
    return pd.DataFrame()


def stage_refinement(ctx: RunContext) -> None:
    sweep_path = ctx.run_root / "sweep/sweep_results.parquet"
    sweep = pd.read_parquet(sweep_path) if sweep_path.exists() else pd.DataFrame()
    if sweep.empty:
        write_csv(ctx.run_root / "refinement/refinement_registry.csv", [])
        write_csv(ctx.run_root / "refinement/refinement_summary.csv", [])
        write_text(ctx.run_root / "refinement/refinement_report.md", "# Refinement Report\n\nNo sweep candidates available.\n")
        return
    selected = []
    for fams, n in [(["D3", "D4", "E1"], 2), (["A1_A3"], 1), (["A2"], 1), (["F1", "G1"], 1)]:
        sub = sweep[sweep["family"].isin(fams) & sweep["events"].gt(0)].sort_values("robustness_score", ascending=False).head(n)
        selected.extend(sub.to_dict("records"))
    neighbors: list[dict[str, Any]] = []
    for rec in selected:
        base = {k: rec[k] for k in ["family", "tier_filter", "parent_gate", "deleveraged_gate", "funding_gate", "price_oi_gate", "liquidity_quality_gate"] if k in rec}
        count = 0
        for h in ["30m", "1h", "6h", "24h", "72h"]:
            for target in [1.0, 2.0, 3.0, 5.0]:
                if count >= 12:
                    break
                row = {**base, "candidate_type": "local_refinement", "horizon": h, "target_r": target, "stop_mult": rec.get("stop_mult", 1.0), "cost_mult": rec.get("cost_mult", 1.0)}
                row["candidate_id"] = f"{row['family']}__refine__{stable_hash(row, 12)}"
                neighbors.append(row)
                count += 1
            if count >= 12:
                break
    if ctx.args.local_refine_budget:
        neighbors = neighbors[: ctx.args.local_refine_budget if not ctx.args.smoke else min(ctx.args.local_refine_budget, 6)]
    write_csv(ctx.run_root / "refinement/refinement_registry.csv", neighbors)
    df = load_event_regime(ctx)
    res = evaluate_candidates(df, neighbors) if neighbors else pd.DataFrame()
    write_df_csv(ctx.run_root / "refinement/refinement_summary.csv", res)
    write_text(ctx.run_root / "refinement/refinement_report.md", f"# Refinement Report\n\n- selected base candidates: `{len(selected)}`\n- neighbor variants: `{len(neighbors)}`\n- isolated one-point results should be rejected if neighbors do not show similar direction.\n")


def stage_matched_null(ctx: RunContext) -> None:
    shortlist = top_shortlist(ctx, 5)
    event_df = load_event_regime(ctx)
    null_df = load_event_regime(ctx, null=True)
    rows = []
    for cand in shortlist.to_dict("records"):
        ev_sub = apply_candidate_filter(event_df, cand)
        nu_sub = apply_candidate_filter(null_df, cand)
        h = cand.get("horizon", "24h")
        ev_ret = surface_return_r(ev_sub, h, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
        nu_ret = surface_return_r(nu_sub, h, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
        ev = summarize_returns(ev_ret)
        nu = summarize_returns(nu_ret)
        rows.append({
            "candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "event_count": ev["events"], "null_count": nu["events"],
            "event_mean_R": ev["mean_R"], "null_mean_R": nu["mean_R"], "event_net_R": ev["net_R"], "null_net_R": nu["net_R"],
            "event_minus_null_mean_R": ev["mean_R"] - nu["mean_R"], "beats_regime_matched_null": bool(ev["mean_R"] > nu["mean_R"] and ev["net_R"] > nu["net_R"]),
            "nulls_per_event_effective": 1, "null_policy_note": "existing_prior_matched_null_path_metrics_have_one_null_per_event; additional nulls require recomputation",
        })
    write_csv(ctx.run_root / "matched_null/matched_null_validation_summary.csv", rows)
    write_text(ctx.run_root / "matched_null/matched_null_validation_report.md", f"# Matched Null And Regime Uplift Validation Report\n\n- shortlisted candidates checked: `{len(rows)}`\n- null source: prior path-diagnostics matched-null path metrics\n- requested nulls per event: `{ctx.args.nulls_per_event}`\n- effective nulls per event: `1` when using frozen prior null path metrics; this is a limitation and is promotion-blocking for direct validation claims.\n")


def stage_one_minute(ctx: RunContext) -> None:
    shortlist = top_shortlist(ctx, 5)
    impact_path = PILOT_1M_ROOT / "impact/one_minute_impact_summary.csv"
    impact = read_csv_safe(impact_path)
    rows = []
    if ctx.args.use_targeted_1m and not impact.empty and not shortlist.empty:
        event_df = load_event_regime(ctx)
        for cand in shortlist.to_dict("records"):
            sub = apply_candidate_filter(event_df, cand)
            ids = set(sub.get("event_id", pd.Series(dtype=str)).astype(str).head(5000))
            hit = impact[impact.get("event_id", pd.Series(dtype=str)).astype(str).isin(ids)] if "event_id" in impact.columns else pd.DataFrame()
            rows.append({"candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "shortlist_events_checked": len(ids), "pilot_1m_overlap_rows": int(len(hit)), "mean_1m_mfe_bps": float(pd.to_numeric(hit.get("one_m_mfe_bps", pd.Series(dtype=float)), errors="coerce").mean()) if not hit.empty else np.nan, "mean_1m_mae_bps": float(pd.to_numeric(hit.get("one_m_mae_bps", pd.Series(dtype=float)), errors="coerce").mean()) if not hit.empty else np.nan, "material_same_bar_share": float(hit.get("material_same_bar_resolution_needed", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if not hit.empty else np.nan})
    write_csv(ctx.run_root / "one_minute/one_minute_overlay_summary.csv", rows)
    write_text(ctx.run_root / "one_minute/one_minute_overlay_report.md", f"# Targeted 1m Overlay Report\n\n- use targeted 1m flag: `{ctx.args.use_targeted_1m}`\n- pilot impact artifact exists: `{impact_path.exists()}`\n- overlay rows: `{len(rows)}`\n- if shortlisted events are not covered, use `additional_1m_window_request.csv` before execution-grade validation.\n")
    write_csv(ctx.run_root / "one_minute/additional_1m_window_request.csv", [])


def cpcv_rows_for_candidate(df: pd.DataFrame, cand: Mapping[str, Any]) -> list[dict[str, Any]]:
    sub = apply_candidate_filter(df, cand).sort_values("decision_ts")
    if sub.empty:
        return []
    h = cand.get("horizon", "24h")
    ret = surface_return_r(sub, h, float(cand.get("target_r", 2.0)), float(cand.get("stop_mult", 1.0)), float(cand.get("cost_mult", 1.0)))
    sub = sub.assign(_ret_R=ret.values)
    blocks = pd.qcut(pd.to_datetime(sub["decision_ts"], utc=True).rank(method="first"), q=min(8, len(sub)), labels=False, duplicates="drop")
    sub = sub.assign(_block=blocks)
    rows = []
    block_ids = sorted(sub["_block"].dropna().unique())
    combos = list(itertools.combinations(block_ids, min(2, len(block_ids)))) if len(block_ids) >= 2 else [(b,) for b in block_ids]
    for combo in combos:
        test = sub[sub["_block"].isin(combo)]
        sm = summarize_returns(test["_ret_R"])
        pos = test[test["_ret_R"] > 0]
        sym_share = 0.0
        if not pos.empty:
            sym_share = float(pos.groupby("symbol")['_ret_R'].sum().max() / max(float(pos["_ret_R"].sum()), 1e-9))
        rows.append({"candidate_id": cand.get("candidate_id"), "family": cand.get("family"), "test_blocks": ";".join(map(str, combo)), **sm, "max_symbol_positive_share": sym_share})
    return rows


def stage_validation(ctx: RunContext) -> None:
    shortlist = top_shortlist(ctx, 5)
    df = load_event_regime(ctx)
    rows = []
    for cand in shortlist.to_dict("records"):
        rows.extend(cpcv_rows_for_candidate(df, cand))
    cpcv = pd.DataFrame(rows)
    write_df_csv(ctx.run_root / "validation/cpcv_summary.csv", cpcv)
    wf_rows = []
    if not cpcv.empty:
        for cid, g in cpcv.groupby("candidate_id"):
            wf_rows.append({"candidate_id": cid, "paths": len(g), "median_path_net_R": float(g["net_R"].median()), "percent_positive_paths": float((g["net_R"] > 0).mean()), "worst_path_net_R": float(g["net_R"].min()), "path_dispersion": float(g["net_R"].std(ddof=0)), "passes_55pct_positive_paths": bool((g["net_R"] > 0).mean() >= 0.55)})
    write_csv(ctx.run_root / "validation/walk_forward_summary.csv", wf_rows)
    write_csv(ctx.run_root / "validation/overfit_control_summary.csv", [{"control": "PBO_DSR", "status": "not_implemented_transparent_cpcv_proxy_used", "limitation": "selection_bias_not_fully_estimated"}])
    write_text(ctx.run_root / "validation/validation_report.md", f"# Validation Report\n\n- CPCV rows: `{len(cpcv)}`\n- purge/embargo model: `block-level proxy; exact event overlap purging not fully implemented in v1`\n- hard validation claims are not allowed from this run.\n")


def stage_portfolio(ctx: RunContext) -> None:
    shortlist = top_shortlist(ctx, 5)
    matched = read_csv_safe(ctx.run_root / "matched_null/matched_null_validation_summary.csv")
    valid = read_csv_safe(ctx.run_root / "validation/walk_forward_summary.csv")
    df = load_event_regime(ctx)
    rows = []
    for cand in shortlist.to_dict("records"):
        cid = cand.get("candidate_id")
        mrow = matched[matched.get("candidate_id", pd.Series(dtype=str)).eq(cid)] if not matched.empty else pd.DataFrame()
        vrow = valid[valid.get("candidate_id", pd.Series(dtype=str)).eq(cid)] if not valid.empty else pd.DataFrame()
        eligible = (not mrow.empty and bool(mrow.iloc[0].get("beats_regime_matched_null", False))) and (not vrow.empty and bool(vrow.iloc[0].get("passes_55pct_positive_paths", False)))
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
                for r in ret.head(5000):
                    eq += eq * risk_pct * float(r)
                    peak = max(peak, eq)
                    max_dd = min(max_dd, (eq / peak) - 1.0)
                    if eq <= equity * 0.1:
                        ruin = True
                        break
                rows.append({"candidate_id": cid, "family": cand.get("family"), "starting_equity": equity, "risk_pct": risk_pct, "ending_equity": eq, "max_drawdown_pct": max_dd * 100, "ruin_flag": ruin, "portfolio_overlay_status": "diagnostic_not_live_recommendation"})
    write_csv(ctx.run_root / "portfolio/aggressive_10x_portfolio_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/aggressive_10x_portfolio_report.md", "# Aggressive 10x Portfolio Overlay Report\n\nThis is a diagnostic overlay only. Aggressive sizing does not create alpha and does not authorize live trading.\n")


def stage_triage(ctx: RunContext) -> None:
    sweep = read_csv_safe(ctx.run_root / "sweep/sweep_summary.csv")
    matched = read_csv_safe(ctx.run_root / "matched_null/matched_null_validation_summary.csv")
    valid = read_csv_safe(ctx.run_root / "validation/walk_forward_summary.csv")
    rows = []
    next_dir = ctx.run_root / "triage/next_candidate_contracts"
    next_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for fam in ["D3", "D4", "E1", "A1_A3", "A2", "F1", "G1"]:
        f = sweep[sweep.get("family", pd.Series(dtype=str)).eq(fam)] if not sweep.empty else pd.DataFrame()
        best = f.sort_values("robustness_score", ascending=False).head(1) if not f.empty else pd.DataFrame()
        label = "blocked_by_missing_data" if fam in {"F1", "G1"} else "continue_regime_development"
        reason = "no evaluated rows"
        cid = ""
        if not best.empty:
            cid = str(best.iloc[0].get("candidate_id"))
            reason = "regime-conditioned screening evidence requires matched-null and validation confirmation"
            m = matched[matched.get("candidate_id", pd.Series(dtype=str)).eq(cid)] if not matched.empty else pd.DataFrame()
            v = valid[valid.get("candidate_id", pd.Series(dtype=str)).eq(cid)] if not valid.empty else pd.DataFrame()
            if not m.empty and bool(m.iloc[0].get("beats_regime_matched_null", False)):
                label = "promote_to_targeted_execution_data_collection" if fam in {"D3", "D4", "E1"} else "continue_regime_development"
                if not v.empty and bool(v.iloc[0].get("passes_55pct_positive_paths", False)):
                    label = "promote_to_family_specific_validation"
            elif not m.empty:
                label = "reject_current_translation"
        rows.append({"family": fam, "best_candidate_id": cid, "label": label, "reason": reason})
        if label in {"promote_to_family_specific_validation", "promote_to_targeted_execution_data_collection"} and created < 3 and not best.empty:
            contract = best.iloc[0].to_dict()
            contract.update({"frozen_next_phase_status": "draft_contract_requires_human_review", "no_live_trading": True, "no_sealed_validation_without_new_contract": True})
            (next_dir / f"{fam}_next_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            created += 1
    write_csv(ctx.run_root / "triage/family_triage_summary.csv", rows)
    write_text(ctx.run_root / "triage/strategy_regime_triage_report.md", f"# Strategy Regime Triage Report\n\n- families triaged: `{len(rows)}`\n- next candidate contracts created: `{created}`\n- no family is validated or live-ready from this run.\n")


def stage_sector_scaffold(ctx: RunContext) -> None:
    write_text(ctx.run_root / "sector_catalyst/sector_catalyst_data_scaffold.md", "# Sector/Catalyst Data Scaffold\n\nNo local point-in-time sector/catalyst store is assumed. B1/C2 alpha tests remain blocked until sector membership and catalyst events are timestamped with source/confidence fields.\n")
    write_csv(ctx.run_root / "sector_catalyst/sector_map_schema.csv", [{"column": c, "description": d} for c, d in [("symbol", "Bybit symbol"), ("sector", "point-in-time sector label"), ("valid_from_ts", "known valid start"), ("valid_to_ts", "known valid end"), ("source", "source URI/name"), ("confidence", "manual/source confidence")]])
    write_csv(ctx.run_root / "sector_catalyst/catalyst_event_schema.csv", [{"column": c, "description": d} for c, d in [("symbol", "Bybit symbol"), ("event_ts", "event timestamp known to market"), ("mechanism", "access/leverage/legal/supply/utility/attention"), ("source", "source URI/name"), ("confidence", "confidence score"), ("visible_ts", "timestamp strategy could know event")]])
    write_text(ctx.run_root / "sector_catalyst/b1_c2_future_test_design.md", "# B1/C2 Future Test Design\n\nBuild point-in-time sector and catalyst stores first. Event-day chase is deprioritized; post-catalyst bases need visibility timestamps and mechanism labels before testing.\n")


def stage_decision(ctx: RunContext) -> None:
    triage = read_csv_safe(ctx.run_root / "triage/family_triage_summary.csv")
    if triage.empty:
        verdict = "blocked_by_protocol_issue"
    elif triage["label"].eq("promote_to_family_specific_validation").any():
        verdict = "promote_one_or_more_to_family_specific_validation"
    elif triage["label"].eq("promote_to_targeted_execution_data_collection").any():
        verdict = "promote_to_targeted_execution_data_collection"
    elif triage["label"].eq("continue_regime_development").any():
        verdict = "continue_regime_stack_development"
    elif triage["label"].eq("blocked_by_missing_data").all():
        verdict = "blocked_by_data_or_execution"
    else:
        verdict = "reject_current_regime_conditioned_translations"
    if verdict not in ALLOWED_VERDICTS:
        verdict = "blocked_by_protocol_issue"
    write_json(ctx.run_root / "decision_summary.json", {"verdict": verdict, "final_holdout_untouched": True, "protected_start": str(FINAL_HOLDOUT_START), "run_root": str(ctx.run_root), "created_at_utc": utc_now()})
    write_text(ctx.run_root / "QLMG_REGIME_STACK_AND_SMART_SWEEP_REPORT.md", f"# QLMG Regime Stack And Smart Sweep Report\n\n## Verdict\n`{verdict}`\n\n## Governance\n- final holdout untouched: `true`\n- protected start: `{FINAL_HOLDOUT_START}`\n- allowed data end: `{SCREENING_END}`\n- no live trading or sealed validation is authorized.\n\n## Evidence Artifacts\n- regime contract: `regime_contract/regime_stack_contract.json`\n- regime features: `regime_features/regime_feature_panel.parquet`\n- conditional cube: `cubes/conditional_performance_cube.parquet`\n- sweep results: `sweep/sweep_results.parquet`\n- matched null validation: `matched_null/matched_null_validation_summary.csv`\n- validation: `validation/cpcv_summary.csv`\n- triage: `triage/family_triage_summary.csv`\n\n## Limitations\n- A1/A3 is proxy-limited unless a true event ledger is built.\n- F1/G1 are contract-only unless safe short event generation is added.\n- True top-of-book, public trades, and liquidation feed are missing.\n")


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_REGIME_STACK_AND_SMART_SWEEP_REPORT.md", "decision_summary.json", "preflight/preflight_report.md", "preflight/resource_guard_report.md", "regime_contract/regime_stack_contract.json", "regime_contract/regime_feature_dictionary.csv", "regime_qc/regime_label_qc_report.md", "cubes/conditional_cube_report.md", "cubes/conditional_cube_summary.csv", "sweep/sweep_report.md", "sweep/sweep_summary.csv", "refinement/refinement_report.md", "refinement/refinement_summary.csv", "matched_null/matched_null_validation_report.md", "matched_null/matched_null_validation_summary.csv", "one_minute/one_minute_overlay_report.md", "one_minute/one_minute_overlay_summary.csv", "validation/validation_report.md", "validation/walk_forward_summary.csv", "portfolio/aggressive_10x_portfolio_report.md", "portfolio/aggressive_10x_portfolio_summary.csv", "triage/strategy_regime_triage_report.md", "triage/family_triage_summary.csv", "sector_catalyst/sector_catalyst_data_scaffold.md", "notifications/telegram_readiness_report.md", "tmux/watch_commands.md", "seal/seal_guard_report.md",
    ]
    rows = []
    for rel in include:
        src = ctx.run_root / rel
        rows.append({"relative_path": rel, "exists": src.exists(), "size_bytes": src.stat().st_size if src.exists() else 0})
        if src.exists() and src.stat().st_size <= 5_000_000:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
    # Copy next contracts if present.
    next_dir = ctx.run_root / "triage/next_candidate_contracts"
    if next_dir.exists():
        for src in next_dir.glob("*.json"):
            dst = bundle / f"next_candidate_contracts__{src.name}"
            shutil.copy2(src, dst)
            rows.append({"relative_path": str(src.relative_to(ctx.run_root)), "exists": True, "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_json(bundle / "artifact_path_index.json", {"artifacts": rows})
    zip_path = ctx.run_root / "qlmg_regime_stack_and_smart_sweep_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(bundle))


STAGE_FUNCS = {
    "preflight-and-prior-artifact-audit": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram_tmux,
    "seal-guard": stage_seal,
    "regime-stack-contract": stage_regime_contract,
    "regime-feature-builder": stage_regime_features,
    "regime-label-qc-and-causality-audit": stage_regime_qc,
    "strategy-family-contracts-with-regime-preconditions": stage_strategy_contracts,
    "conditional-performance-cubes": stage_cubes,
    "smart-sobol-regime-entry-exit-sweep": stage_sweep,
    "local-neighborhood-refinement": stage_refinement,
    "matched-null-and-regime-uplift-validation": stage_matched_null,
    "targeted-1m-overlay-for-shortlisted-candidates": stage_one_minute,
    "walk-forward-cpcv-and-overfit-controls": stage_validation,
    "aggressive-10x-regime-conditioned-portfolio-overlay": stage_portfolio,
    "strategy-regime-triage-and-next-contracts": stage_triage,
    "sector-catalyst-data-scaffold": stage_sector_scaffold,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        print(f"[resume] skipping {stage}", flush=True)
        return
    append_command(ctx.run_root, stage)
    ensure_guard(ctx, stage, estimate_stage_gb(stage, ctx.args.smoke))
    ctx.notifier.send(f"QLMG regime sweep stage start: {stage}", str(ctx.run_root))
    tmp = ctx.run_root / "tmp" / stage
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        STAGE_FUNCS[stage](ctx)
        mark_done(ctx.run_root, stage)
        shutil.rmtree(tmp, ignore_errors=True)
        ctx.notifier.send(f"QLMG regime sweep stage complete: {stage}", str(ctx.run_root))
    except Exception as exc:
        ctx.notifier.send(f"QLMG regime sweep stage failed: {stage}", f"{type(exc).__name__}: {exc}", level="error")
        raise


def main() -> int:
    args = parse_args()
    run_root, reason = resolve_run_root(args)
    start, end = clamp_window(args)
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": reason, "argv": sys.argv, "start": str(start), "end": str(end), "seed": args.seed, "smoke": args.smoke, "created_at_utc": utc_now()})
    notifier.send("QLMG regime sweep run start", str(run_root))
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG regime sweep run complete", str(run_root))
        return 0
    except Exception as exc:
        notifier.send("QLMG regime sweep run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
