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

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_brutal_no_depth_stress_20260628_v1"
LISTING_ROOT = RESULTS_ROOT / "phase_qlmg_listing_generic_full_event_replay_20260627_v1_20260627_115829"
D4_SURVIVAL_ROOT = RESULTS_ROOT / "phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"
DEFAULT_SEED = 20260628
GB = 1024**3

LISTING_IDS = [
    "new_perp_listing_event_study__589a8c85c943",
    "new_perp_listing_event_study__9dc07cfc405c",
    "new_perp_listing_event_study__b1a3735d5092",
]
D4_CANDIDATE_ID = "D4__b4c9487fe82c"

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "candidate-and-control-reconstruction",
    "baseline-replay-reproduction",
    "no-depth-execution-stress-design",
    "slippage-and-participation-stress",
    "stop-execution-and-same-minute-stress",
    "missed-fill-and-gap-through-stress",
    "liquidation-buffer-and-dynamic-sizing-stress",
    "control-normalized-stress-comparison",
    "aggressive-small-account-overlay",
    "candidate-survival-decision-table",
    "data-value-of-information-report",
    "decision-report",
    "compact-review-bundle",
    "all",
)

FINAL_VERDICTS = {
    "listing_candidate_survives_brutal_stress_needs_depth",
    "listing_candidate_survives_moderate_stress_needs_depth",
    "listing_candidate_fails_brutal_stress_current_expression_only",
    "listing_candidate_unresolved_execution_depth_needed",
    "d4_remains_execution_depth_carry_forward",
    "vendor_pilot_high_priority",
    "vendor_pilot_low_priority",
    "forward_capture_high_priority",
    "micro_canary_possible_execution_only",
    "micro_canary_not_recommended",
    "blocked_by_protocol_issue",
}

SURVIVAL_LABELS = {
    "survives_brutal_no_depth_stress",
    "survives_moderate_stress_only",
    "fails_brutal_no_depth_stress_current_expression_only",
    "path_edge_survives_but_execution_unresolved",
    "not_fairly_tested_missing_execution_depth",
    "blocked_by_protocol_issue",
}

RISK_PCTS = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
EQUITY_CASES = [200, 500, 1000]


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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-brutal-no-depth-stress")
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
        rec = {
            "ts_utc": utc_now(),
            "title": title,
            "body": body,
            "level": level,
            "sent": sent,
            "status": self.status,
            "error": error,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        try:
            (self.run_root / "watch_status.json").write_text(
                json.dumps({"status": "running", "last_event": title, "ts_utc": rec["ts_utc"], "run_root": str(self.run_root)}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        return sent


def parse_grid_float(text: str) -> list[float]:
    vals: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return vals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG brutal no-depth execution stress, train-only")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=30.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--include-d4", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-listing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--slippage-bps-grid", default="25,50,100,200,300")
    p.add_argument("--participation-grid", default="0.001,0.0025,0.005,0.01")
    p.add_argument("--missed-fill-grid", default="0,0.1,0.25,0.5")
    p.add_argument("--latency-grid-ms", default="50,150,500")
    p.add_argument("--tmux-session-name", default="qlmg_brutal_stress")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--run-root", default="")
    args = p.parse_args(argv)
    args.slippage_bps_values = parse_grid_float(args.slippage_bps_grid)
    args.participation_values = parse_grid_float(args.participation_grid)
    args.missed_fill_values = parse_grid_float(args.missed_fill_grid)
    args.latency_ms_values = [int(float(x)) for x in parse_grid_float(args.latency_grid_ms)]
    return args


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
    end = min(pd.to_datetime(args.end, utc=True), SCREENING_END)
    if start >= FINAL_HOLDOUT_START or end >= FINAL_HOLDOUT_START:
        raise RuntimeError("requested window overlaps protected QLMG holdout")
    return pd.Timestamp(start), pd.Timestamp(end)


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    rows_list = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows_list:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(dict(row))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def shell(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(cmd), cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def input_artifacts(include_d4: bool = False) -> list[Path]:
    paths = [
        LISTING_ROOT / "SCOPE_CORRECTED_LISTING_GENERIC_RESULTS_REPORT.md",
        LISTING_ROOT / "replay/full_event_one_minute_events.parquet",
        LISTING_ROOT / "controls/full_event_control_summary.csv",
        LISTING_ROOT / "stress/full_event_stress_summary.csv",
        LISTING_ROOT / "ordering/full_event_ordering_summary.csv",
        LISTING_ROOT / "listing/full_event_candidate_manifest.csv",
        LISTING_ROOT / "listing/full_event_listing_events.parquet",
    ]
    if include_d4:
        paths.extend([D4_SURVIVAL_ROOT / "D4_SURVIVABILITY_REDESIGN_REPORT.md", D4_SURVIVAL_ROOT / "decision_summary.json"])
    return paths


def artifact_manifest(include_d4: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {}
    for p in input_artifacts(include_d4):
        exists = p.exists()
        row = {"path": str(p), "exists": exists, "size_bytes": p.stat().st_size if exists else 0, "sha256": sha256_file(p) if exists and p.stat().st_size < 500 * 1024 * 1024 else "large_or_missing"}
        rows.append(row)
        hashes[str(p)] = row
    return rows, hashes


def done_path(root: Path, stage: str) -> Path:
    return root / "stage_status" / f"{stage}.done"


def mark_done(root: Path, stage: str) -> None:
    write_text(done_path(root, stage), utc_now())


def required_outputs(root: Path, stage: str) -> list[Path]:
    mapping: dict[str, list[str]] = {
        "preflight-and-artifact-freeze": ["preflight/preflight_report.md", "preflight/frozen_artifact_hashes.json", "preflight/input_artifact_manifest.csv", "preflight/resource_guard_report.md"],
        "telegram-and-tmux-setup": ["notifications/telegram_readiness_report.md", "tmux/watch_commands.md", "tmux/tmux_run_instructions.md"],
        "seal-guard": ["seal/protected_slice_check.json", "seal/seal_guard_report.md"],
        "candidate-and-control-reconstruction": ["reconstruction/reconstruction_summary.csv", "reconstruction/reconstruction_report.md"],
        "baseline-replay-reproduction": ["baseline/baseline_reproduction_summary.csv", "baseline/baseline_reproduction_report.md"],
        "no-depth-execution-stress-design": ["stress_design/stress_design_contract.json", "stress_design/stress_design_report.md"],
        "slippage-and-participation-stress": ["stress/slippage_participation_summary.csv", "stress/slippage_participation_report.md"],
        "stop-execution-and-same-minute-stress": ["stress/stop_execution_summary.csv", "stress/stop_execution_report.md"],
        "missed-fill-and-gap-through-stress": ["stress/missed_fill_gap_summary.csv", "stress/missed_fill_gap_report.md"],
        "liquidation-buffer-and-dynamic-sizing-stress": ["stress/liquidation_dynamic_sizing_summary.csv", "stress/liquidation_dynamic_sizing_report.md"],
        "control-normalized-stress-comparison": ["comparison/control_normalized_stress_summary.csv", "comparison/control_normalized_stress_report.md"],
        "aggressive-small-account-overlay": ["portfolio/aggressive_overlay_summary.csv", "portfolio/aggressive_overlay_report.md"],
        "candidate-survival-decision-table": ["decision/candidate_survival_table.csv", "decision/candidate_survival_report.md"],
        "data-value-of-information-report": ["voi/data_value_of_information_summary.csv", "voi/data_value_of_information_report.md"],
        "decision-report": ["QLMG_BRUTAL_NO_DEPTH_STRESS_REPORT.md", "decision_summary.json"],
        "compact-review-bundle": ["compact_review_bundle/artifact_path_index.csv", "compact_review_bundle/README.md"],
    }
    return [root / p for p in mapping.get(stage, [])]


def stage_complete(root: Path, stage: str) -> bool:
    return done_path(root, stage).exists() and all(p.exists() for p in required_outputs(root, stage))


def estimate_stage_output_gb(stage: str) -> float:
    if stage in {"slippage-and-participation-stress", "missed-fill-and-gap-through-stress"}:
        return 0.20
    if stage == "compact-review-bundle":
        return 0.05
    return 0.10


def enforce_resources(ctx: RunContext, stage: str) -> None:
    snap = resource_snapshot(REPO)
    guard = check_resource_guard(
        snap,
        estimated_output_gb=estimate_stage_output_gb(stage),
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=20.0,
        allow_large_output=bool(ctx.args.allow_large_output),
    )
    if guard["warnings"]:
        ctx.notifier.send("QLMG brutal stress resource warning", json.dumps(guard, sort_keys=True), level="warn")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop before {stage}: {guard}")


def load_listing_events() -> pd.DataFrame:
    events = pd.read_parquet(LISTING_ROOT / "listing/full_event_listing_events.parquet")
    validate_no_protected(events, ["decision_ts", "entry_ts"])
    return events


def load_replay_full_hold(ctx: RunContext | None = None) -> pd.DataFrame:
    df = pd.read_parquet(LISTING_ROOT / "replay/full_event_one_minute_events.parquet")
    mask = (df["window_scope"].astype(str) == "full_hold") & (df["full_hold_replayed"].astype(bool)) & (df["replay_status"].astype(str) == "ok")
    df = df.loc[mask].copy()
    if ctx is not None and ctx.args.include_listing:
        ids = set(LISTING_IDS)
        if not ctx.args.include_d4:
            df = df[df["candidate_id"].astype(str).isin(ids | {"generic_shock_reversal_hypothesis"})]
    if ctx is not None and ctx.args.max_symbols and "symbol" in df.columns:
        symbols = sorted(df["symbol"].dropna().astype(str).unique())[: int(ctx.args.max_symbols)]
        df = df[df["symbol"].astype(str).isin(symbols)].copy()
    if ctx is not None:
        events = load_listing_events()[["event_id", "decision_ts", "entry_ts", "turnover", "range_pct", "candidate_id", "listing_metadata_source"]].rename(columns={"candidate_id": "event_candidate_id"})
        df = df.merge(events, on="event_id", how="left")
        if "decision_ts" in df.columns:
            start, end = ctx.start, ctx.end
            ts = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce")
            df = df[(ts.isna()) | ((ts >= start) & (ts <= end))].copy()
            validate_no_protected(df, ["decision_ts", "entry_ts"])
    return df


def numeric(s: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def deterministic_u01(seed: int, *parts: Any) -> float:
    payload = "|".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")
    x = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return x / float(16**16 - 1)


def apply_missed_fill_mask(df: pd.DataFrame, probability: float, seed: int, scenario_id: str, *, adverse: bool = False) -> pd.Series:
    if probability <= 0:
        return pd.Series(False, index=df.index)
    base = df["event_id"].astype(str).map(lambda x: deterministic_u01(seed, scenario_id, x)) < probability
    if not adverse:
        return base
    # Adverse fill model misses profitable/high-MFE rows first. It is diagnostic, not rankable for promotion.
    mfe = numeric(df.get("mfe_bps_1m", pd.Series(index=df.index, dtype=float)))
    threshold = mfe.quantile(max(0.0, min(1.0, 1.0 - probability))) if len(mfe.dropna()) else np.inf
    return (mfe >= threshold) | base


def participation_retained_mask(df: pd.DataFrame, cap: float) -> tuple[pd.Series, str, float]:
    if "turnover" not in df.columns or numeric(df["turnover"]).le(0).all():
        return pd.Series(False, index=df.index), "not_fairly_tested_missing_execution_depth", 0.0
    turnover = numeric(df["turnover"])
    if cap <= 0.001:
        q = 0.50
    elif cap <= 0.0025:
        q = 0.25
    elif cap <= 0.005:
        q = 0.10
    else:
        q = 0.05
    threshold = float(turnover.quantile(q))
    return turnover >= threshold, "proxy_turnover_participation_cap", threshold


def pf_from_r(r: pd.Series) -> float:
    vals = pd.to_numeric(r, errors="coerce").dropna()
    gains = float(vals[vals > 0].sum())
    losses = float(-vals[vals < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def month_concentration(df: pd.DataFrame, r_col: str) -> float:
    if "decision_ts" not in df.columns or df.empty:
        return np.nan
    tmp = df.copy()
    tmp["_r"] = pd.to_numeric(tmp[r_col], errors="coerce").fillna(0.0)
    pos = float(tmp.loc[tmp["_r"] > 0, "_r"].sum())
    if pos <= 0:
        return 0.0
    ts = pd.to_datetime(tmp["decision_ts"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("M")
    month_pos = tmp.assign(_month=ts).groupby("_month")["_r"].apply(lambda x: x[x > 0].sum())
    return float(month_pos.max() / pos) if not month_pos.empty else np.nan


def summarize_group(df: pd.DataFrame, r_col: str = "stressed_R") -> dict[str, Any]:
    vals = pd.to_numeric(df.get(r_col, pd.Series(dtype=float)), errors="coerce").dropna()
    events = int(len(vals))
    net = float(vals.sum()) if events else 0.0
    return {
        "events": events,
        "net_R": net,
        "mean_R": float(vals.mean()) if events else np.nan,
        "median_R": float(vals.median()) if events else np.nan,
        "PF": pf_from_r(vals),
        "hit_rate": float((vals > 0).mean()) if events else np.nan,
        "max_single_loss_R": float(vals.min()) if events else np.nan,
        "positive_R": float(vals[vals > 0].sum()) if events else 0.0,
    }


def stress_dataframe(
    df: pd.DataFrame,
    *,
    seed: int,
    scenario_id: str,
    entry_slip_bps: float = 0.0,
    exit_slip_bps: float = 0.0,
    gap_bps: float = 0.0,
    participation_cap: float | None = None,
    missed_fill_prob: float = 0.0,
    adverse_missed_fill: bool = False,
    candidate_only_adverse: bool = False,
    stop_ambiguity_penalty_r: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    risk = pd.to_numeric(out["risk_bps_used"], errors="coerce")
    valid_risk = risk.gt(0) & np.isfinite(risk)
    invalid_risk_count = int((~valid_risk).sum())
    out = out.loc[valid_risk].copy()
    if out.empty:
        return out.assign(stressed_R=pd.Series(dtype=float)), {"invalid_risk_rows": invalid_risk_count, "retained_after_participation": 0, "missed_fill_rows": 0, "participation_status": "not_applicable"}
    risk = pd.to_numeric(out["risk_bps_used"], errors="coerce")
    haircut = (float(entry_slip_bps) + float(exit_slip_bps) + float(gap_bps)) / risk
    out["base_R"] = pd.to_numeric(out["net_R_1m_mark_proxy"], errors="coerce").fillna(0.0)
    out["execution_haircut_R"] = haircut
    out["stressed_R"] = out["base_R"] - out["execution_haircut_R"]
    if stop_ambiguity_penalty_r:
        amb = out["stop_hit_1m"].astype(bool) & out["target_hit_1m"].astype(bool)
        out.loc[amb, "stressed_R"] = out.loc[amb, "stressed_R"] - float(stop_ambiguity_penalty_r)
    participation_status = "not_applied"
    participation_threshold = np.nan
    if participation_cap is not None:
        keep, participation_status, participation_threshold = participation_retained_mask(out, float(participation_cap))
        out = out.loc[keep].copy()
    retained_after_participation = len(out)
    missed = pd.Series(False, index=out.index)
    if missed_fill_prob > 0 and not out.empty:
        if candidate_only_adverse:
            cand = out["window_role"].astype(str).eq("candidate_event")
            missed.loc[cand] = apply_missed_fill_mask(out.loc[cand], missed_fill_prob, seed, scenario_id, adverse=True)
            ctrl = out["window_role"].astype(str).ne("candidate_event")
            missed.loc[ctrl] = apply_missed_fill_mask(out.loc[ctrl], missed_fill_prob, seed, scenario_id, adverse=False)
        else:
            missed = apply_missed_fill_mask(out, missed_fill_prob, seed, scenario_id, adverse=adverse_missed_fill)
        out = out.loc[~missed].copy()
    meta = {
        "invalid_risk_rows": invalid_risk_count,
        "retained_after_participation": retained_after_participation,
        "missed_fill_rows": int(missed.sum()) if len(missed) else 0,
        "participation_status": participation_status,
        "participation_threshold_turnover": participation_threshold,
    }
    return out, meta


def summarize_candidate_control(stressed: pd.DataFrame, scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cid, sub in stressed.groupby("candidate_id", dropna=False):
        cand = sub[sub["window_role"].astype(str) == "candidate_event"]
        ctrl = sub[sub["window_role"].astype(str) == "control"]
        cs = summarize_group(cand)
        xs = summarize_group(ctrl)
        normalized_ctrl = xs["net_R"] * (cs["events"] / xs["events"]) if xs["events"] else np.nan
        uplift = cs["net_R"] - normalized_ctrl if np.isfinite(normalized_ctrl) else np.nan
        rows.append({
            **dict(scenario),
            "candidate_id": cid,
            "candidate_events": cs["events"],
            "control_events": xs["events"],
            "candidate_net_R": cs["net_R"],
            "candidate_mean_R": cs["mean_R"],
            "candidate_median_R": cs["median_R"],
            "candidate_PF": cs["PF"],
            "candidate_hit_rate": cs["hit_rate"],
            "control_raw_net_R": xs["net_R"],
            "control_normalized_net_R": normalized_ctrl,
            "control_mean_R": xs["mean_R"],
            "control_PF": xs["PF"],
            "normalized_uplift_R": uplift,
            "beats_controls_after_stress": bool(np.isfinite(uplift) and uplift > 0 and cs["net_R"] > 0),
            "month_concentration_positive_R": month_concentration(cand, "stressed_R"),
        })
    return rows


def baseline_from_events(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cid, sub in df.groupby("candidate_id", dropna=False):
        cand = sub[sub["window_role"].astype(str) == "candidate_event"]
        ctrl = sub[sub["window_role"].astype(str) == "control"]
        cs = summarize_group(cand.rename(columns={"net_R_1m_mark_proxy": "stressed_R"}))
        xs = summarize_group(ctrl.rename(columns={"net_R_1m_mark_proxy": "stressed_R"}))
        normalized_ctrl = xs["net_R"] * (cs["events"] / xs["events"]) if xs["events"] else np.nan
        rows.append({
            "candidate_id": cid,
            "rankable_scope": "full_hold",
            "candidate_event_count": cs["events"],
            "control_event_count": xs["events"],
            "event_signal_R": cs["net_R"],
            "control_signal_R_raw_sum": xs["net_R"],
            "control_signal_R_normalized_to_candidate_count": normalized_ctrl,
            "normalized_uplift_R": cs["net_R"] - normalized_ctrl if np.isfinite(normalized_ctrl) else np.nan,
            "event_mean_R": cs["mean_R"],
            "control_mean_R": xs["mean_R"],
            "beats_controls": bool(np.isfinite(normalized_ctrl) and cs["net_R"] > normalized_ctrl),
        })
    return pd.DataFrame(rows)


def stage_preflight(ctx: RunContext) -> None:
    rows, hashes = artifact_manifest(ctx.args.include_d4)
    missing = [r["path"] for r in rows if not r["exists"]]
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(REPO)
    guard = check_resource_guard(snap, estimated_output_gb=1.0, hard_free_gb=5.0, warn_free_gb=7.0, hard_stage_output_gb=20.0, allow_large_output=ctx.args.allow_large_output)
    write_text(
        ctx.run_root / "preflight/resource_guard_report.md",
        f"# Resource Guard\n\n- free_disk_gb: `{snap.free_gb:.3f}`\n- hard_stop_free_gb: `5`\n- warning_free_gb: `7`\n- stage_output_block_gb: `20`\n- max_output_gb: `{ctx.args.max_output_gb}`\n- guard_status: `{guard['status']}`\n- warnings: `{guard['warnings']}`\n",
    )
    write_text(
        ctx.run_root / "preflight/preflight_report.md",
        f"# Preflight And Artifact Freeze\n\n- run_root: `{ctx.run_root}`\n- listing_root: `{LISTING_ROOT}`\n- d4_survival_root: `{D4_SURVIVAL_ROOT}`\n- protected_start: `{FINAL_HOLDOUT_START}`\n- screening_end: `{SCREENING_END}`\n- missing_required_artifacts: `{missing}`\n- include_d4: `{ctx.args.include_d4}`\n- full_hold_only_rankable: `true`\n- git_head: `{shell(['git','rev-parse','HEAD'])}`\n- git_status_short: `{shell(['git','status','--short'])[:5000]}`\n",
    )
    if missing:
        raise RuntimeError(f"required input artifacts missing: {missing}")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop: {guard}")


def stage_telegram(ctx: RunContext) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\n- disabled: `{ctx.notifier.disabled}`\n- remote_available: `{ctx.notifier.remote_available}`\n- status: `{ctx.notifier.status}`\n- missing_or_reason: `{ctx.notifier.missing}`\n- secrets_logged: `false`\n")
    write_text(ctx.run_root / "tmux/watch_commands.md", f"# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n")
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Run Instructions\n\nDefault session: `{ctx.args.tmux_session_name}`. Use `tools/run_qlmg_brutal_no_depth_stress_tmux.sh` with `--launch-tmux` for full unattended execution.\n")


def stage_seal(ctx: RunContext) -> None:
    validate_no_protected(pd.DataFrame({"ts": [ctx.end]}), ["ts"])
    blocked = False
    try:
        validate_no_protected(pd.DataFrame({"ts": [FINAL_HOLDOUT_START]}), ["ts"])
    except RuntimeError:
        blocked = True
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "pre_holdout_read_passed": True, "protected_read_blocked": blocked})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\n- protected_start: `{FINAL_HOLDOUT_START}`\n- run_end: `{ctx.end}`\n- protected smoke blocked: `{blocked}`\n- final holdout touched: `false`\n")


def stage_reconstruction(ctx: RunContext) -> None:
    df = load_replay_full_hold(ctx)
    event_cols = ["decision_ts", "entry_ts"]
    validate_no_protected(df, event_cols)
    rows: list[dict[str, Any]] = []
    for cid, sub in df.groupby("candidate_id", dropna=False):
        rows.append({
            "candidate_id": cid,
            "full_hold_rows": len(sub),
            "candidate_event_rows": int((sub["window_role"].astype(str) == "candidate_event").sum()),
            "control_rows": int((sub["window_role"].astype(str) == "control").sum()),
            "core_24h_rows_rankable": 0,
            "full_hold_only_rankable": True,
            "risk_bps_missing_rows": int(pd.to_numeric(sub["risk_bps_used"], errors="coerce").isna().sum()),
            "turnover_missing_rows": int(pd.to_numeric(sub.get("turnover", pd.Series(index=sub.index)), errors="coerce").isna().sum()),
            "reconstruction_status": "full_hold_event_control_rows_reconstructed",
        })
    write_csv(ctx.run_root / "reconstruction/reconstruction_summary.csv", rows)
    write_text(ctx.run_root / "reconstruction/reconstruction_report.md", f"# Candidate And Control Reconstruction\n\n- loaded full-hold rankable rows: `{len(df)}`\n- core_24h rows excluded from rankable metrics: `true`\n- included listing candidates: `{ctx.args.include_listing}`\n- included D4 diagnostic: `{ctx.args.include_d4}`\n- protected holdout rows detected: `false`\n")


def stage_baseline(ctx: RunContext) -> None:
    # Baseline reproduction must compare against the frozen prior full universe,
    # even during smoke. Smoke bounds only later stress/runtime, not the guard.
    df = load_replay_full_hold(None)
    if not ctx.args.include_d4:
        df = df[df["candidate_id"].astype(str).isin(set(LISTING_IDS) | {"generic_shock_reversal_hypothesis"})].copy()
    cur = baseline_from_events(df)
    prior = read_csv(LISTING_ROOT / "controls/full_event_control_summary.csv")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, row in cur.iterrows():
        cid = row["candidate_id"]
        p = prior[prior["candidate_id"].astype(str).eq(str(cid))]
        if p.empty:
            rows.append({**row.to_dict(), "prior_match_status": "no_prior_control_row", "reproduced_within_tolerance": False})
            if str(cid).startswith("new_perp_listing_event_study"):
                failures.append(str(cid))
            continue
        p0 = p.iloc[0]
        diff_event = float(row["event_signal_R"]) - float(p0["event_signal_R"])
        diff_ctrl = float(row["control_signal_R_normalized_to_candidate_count"]) - float(p0["control_signal_R_normalized_to_candidate_count"])
        count_ok = int(row["candidate_event_count"]) == int(p0["candidate_event_count"]) and int(row["control_event_count"]) == int(p0["control_event_count"])
        tol_event = max(1e-6, abs(float(p0["event_signal_R"])) * 1e-9)
        tol_ctrl = max(1e-6, abs(float(p0["control_signal_R_normalized_to_candidate_count"])) * 1e-9)
        ok = count_ok and abs(diff_event) <= tol_event and abs(diff_ctrl) <= tol_ctrl
        rows.append({**row.to_dict(), "prior_event_signal_R": p0["event_signal_R"], "prior_control_normalized_R": p0["control_signal_R_normalized_to_candidate_count"], "event_signal_R_diff": diff_event, "control_normalized_R_diff": diff_ctrl, "count_ok": count_ok, "reproduced_within_tolerance": ok})
        if str(cid).startswith("new_perp_listing_event_study") and not ok:
            failures.append(str(cid))
    write_csv(ctx.run_root / "baseline/baseline_reproduction_summary.csv", rows)
    write_text(ctx.run_root / "baseline/baseline_reproduction_report.md", f"# Baseline Replay Reproduction\n\n- baseline source: `controls/full_event_control_summary.csv`\n- rankable scope: `full_hold`\n- core_24h double count avoided: `true`\n- reproduced listing candidates: `{len([r for r in rows if str(r.get('candidate_id')).startswith('new_perp_listing_event_study') and r.get('reproduced_within_tolerance')])}`\n- reproduction_failures: `{failures}`\n")
    if failures:
        raise RuntimeError(f"baseline reproduction failed for listing candidates: {failures}")


def stage_stress_design(ctx: RunContext) -> None:
    contract = {
        "phase": "QLMG brutal no-depth stress v1",
        "train_only": True,
        "protected_holdout_start": str(FINAL_HOLDOUT_START),
        "rankable_scope": "full_hold_only",
        "slippage_bps_grid": ctx.args.slippage_bps_values,
        "participation_grid": ctx.args.participation_values,
        "missed_fill_grid": ctx.args.missed_fill_values,
        "latency_grid_ms": ctx.args.latency_ms_values,
        "haircut_formula": "R_haircut=(entry_slip_bps+exit_slip_bps+gap_bps)/risk_bps_used",
        "participation_model": "proxy turnover quantile retention; missing turnover means not_fairly_tested_missing_execution_depth",
        "control_normalization": "control R normalized to candidate-event count before comparison",
        "stress_bands": {
            "moderate": "<=50 bps entry and exit, 0.5% cap, pessimistic same-minute, no adverse missed-fill",
            "severe": "100 bps entry and exit, 0.25% cap, 50 bps gap, 10% symmetric missed-fill",
            "extreme": "200 bps+ entry and exit, 0.1% cap, adverse missed-fill, 100 bps+ gap",
        },
        "depth_missing_policy": "data blocker, not alpha rejection",
    }
    write_json(ctx.run_root / "stress_design/stress_design_contract.json", contract)
    write_text(ctx.run_root / "stress_design/stress_design_report.md", "# No-Depth Execution Stress Design\n\nThe stress bridge applies punitive proxy execution haircuts to existing full-hold 1m mark/OHLCV replay rows. It is not exchange-faithful depth replay. Missing order book, public trade, and liquidation feed evidence remains a data blocker.\n")


def scenario_rows_slippage_participation(ctx: RunContext) -> list[dict[str, Any]]:
    rows = []
    for slip in ctx.args.slippage_bps_values:
        for cap in ctx.args.participation_values:
            band = "moderate" if slip <= 50 and cap >= 0.005 else "severe" if slip <= 100 and cap >= 0.0025 else "extreme"
            rows.append({"scenario_id": f"slip{slip:g}_cap{cap:g}", "scenario_family": "slippage_participation", "stress_band": band, "entry_slip_bps": slip, "exit_slip_bps": slip, "gap_bps": 0.0, "participation_cap": cap, "missed_fill_prob": 0.0, "adverse_missed_fill": False, "candidate_only_adverse": False, "rankable": True})
    return rows


def stage_slippage_participation(ctx: RunContext) -> None:
    df = load_replay_full_hold(ctx)
    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for sc in scenario_rows_slippage_participation(ctx):
        stressed, meta = stress_dataframe(df, seed=ctx.args.seed, scenario_id=sc["scenario_id"], entry_slip_bps=sc["entry_slip_bps"], exit_slip_bps=sc["exit_slip_bps"], participation_cap=sc["participation_cap"])
        meta_rows.append({**sc, **meta})
        rows.extend(summarize_candidate_control(stressed, sc))
    out = pd.DataFrame(rows)
    write_csv(ctx.run_root / "stress/slippage_participation_summary.csv", out)
    write_csv(ctx.run_root / "stress/slippage_participation_meta.csv", meta_rows)
    pos = int(out["beats_controls_after_stress"].sum()) if not out.empty else 0
    write_text(ctx.run_root / "stress/slippage_participation_report.md", f"# Slippage And Participation Stress\n\n- scenarios: `{len(meta_rows)}`\n- candidate/scenario rows: `{len(out)}`\n- positive after normalized controls: `{pos}`\n- participation is a proxy turnover-retention stress because true depth is missing.\n")


def stage_stop_execution(ctx: RunContext) -> None:
    df = load_replay_full_hold(ctx)
    rows: list[dict[str, Any]] = []
    scenarios = [
        {"scenario_id": "same_minute_pessimistic", "scenario_family": "stop_execution", "stress_band": "moderate", "entry_slip_bps": 25.0, "exit_slip_bps": 25.0, "gap_bps": 0.0, "participation_cap": 0.005, "stop_ambiguity_penalty_r": 1.0, "rankable": True},
        {"scenario_id": "stop_market_gap_50bps", "scenario_family": "stop_execution", "stress_band": "severe", "entry_slip_bps": 50.0, "exit_slip_bps": 50.0, "gap_bps": 50.0, "participation_cap": 0.0025, "stop_ambiguity_penalty_r": 1.0, "rankable": True},
        {"scenario_id": "target_first_diagnostic_only", "scenario_family": "stop_execution", "stress_band": "diagnostic", "entry_slip_bps": 25.0, "exit_slip_bps": 25.0, "gap_bps": 0.0, "participation_cap": 0.005, "stop_ambiguity_penalty_r": 0.0, "rankable": False},
    ]
    for sc in scenarios:
        stressed, _meta = stress_dataframe(df, seed=ctx.args.seed, scenario_id=sc["scenario_id"], entry_slip_bps=sc["entry_slip_bps"], exit_slip_bps=sc["exit_slip_bps"], gap_bps=sc["gap_bps"], participation_cap=sc["participation_cap"], stop_ambiguity_penalty_r=sc["stop_ambiguity_penalty_r"])
        rows.extend(summarize_candidate_control(stressed, sc))
    out = pd.DataFrame(rows)
    write_csv(ctx.run_root / "stress/stop_execution_summary.csv", out)
    amb = df["stop_hit_1m"].astype(bool) & df["target_hit_1m"].astype(bool)
    write_text(ctx.run_root / "stress/stop_execution_report.md", f"# Stop Execution And Same-Minute Stress\n\n- same-minute stop/target ambiguous full-hold rows: `{int(amb.sum())}`\n- pessimistic ambiguity branch is rankable: `true`\n- target-first branch is diagnostic-only: `true`\n")


def stage_missed_fill_gap(ctx: RunContext) -> None:
    df = load_replay_full_hold(ctx)
    rows: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for p in ctx.args.missed_fill_values:
        if p == 0:
            continue
        scenarios.append({"scenario_id": f"sym_miss{p:g}_gap50", "scenario_family": "missed_fill_gap", "stress_band": "severe" if p <= 0.10 else "extreme", "entry_slip_bps": 50.0, "exit_slip_bps": 50.0, "gap_bps": 50.0, "participation_cap": 0.0025, "missed_fill_prob": p, "adverse_missed_fill": False, "candidate_only_adverse": False, "rankable": True})
        scenarios.append({"scenario_id": f"adverse_miss{p:g}_gap100", "scenario_family": "missed_fill_gap", "stress_band": "extreme", "entry_slip_bps": 100.0, "exit_slip_bps": 100.0, "gap_bps": 100.0, "participation_cap": 0.001, "missed_fill_prob": p, "adverse_missed_fill": True, "candidate_only_adverse": True, "rankable": False})
    for sc in scenarios:
        stressed, _meta = stress_dataframe(df, seed=ctx.args.seed, scenario_id=sc["scenario_id"], entry_slip_bps=sc["entry_slip_bps"], exit_slip_bps=sc["exit_slip_bps"], gap_bps=sc["gap_bps"], participation_cap=sc["participation_cap"], missed_fill_prob=sc["missed_fill_prob"], adverse_missed_fill=sc["adverse_missed_fill"], candidate_only_adverse=sc["candidate_only_adverse"])
        rows.extend(summarize_candidate_control(stressed, sc))
    out = pd.DataFrame(rows)
    write_csv(ctx.run_root / "stress/missed_fill_gap_summary.csv", out)
    write_text(ctx.run_root / "stress/missed_fill_gap_report.md", f"# Missed-Fill And Gap-Through Stress\n\n- scenarios: `{len(scenarios)}`\n- deterministic seed: `{ctx.args.seed}`\n- symmetric missed-fill stress is rankable; candidate-only adverse fill stress is diagnostic-only.\n")


def safe_leverage_from_risk(risk_bps: pd.Series, buffer: float) -> pd.Series:
    risk = pd.to_numeric(risk_bps, errors="coerce")
    lev = 10000.0 / (risk * float(buffer) + 50.0)
    return lev.clip(lower=0.0, upper=10.0)


def stage_liq_dynamic_sizing(ctx: RunContext) -> None:
    df = load_replay_full_hold(ctx)
    rows: list[dict[str, Any]] = []
    models = [
        ("fixed_10x", 10.0, None),
        ("fixed_5x", 5.0, None),
        ("fixed_3x", 3.0, None),
        ("dynamic_buffer_1p25x_stop", None, 1.25),
        ("dynamic_buffer_1p5x_stop", None, 1.5),
        ("dynamic_buffer_2p0x_stop", None, 2.0),
        ("risk_based_reduce_until_buffer", None, 1.5),
    ]
    for name, fixed_lev, buffer in models:
        tmp = df.copy()
        if fixed_lev is not None:
            lev = pd.Series(float(fixed_lev), index=tmp.index)
        else:
            lev = safe_leverage_from_risk(tmp["risk_bps_used"], float(buffer or 1.5))
        tmp["leverage_used"] = lev
        tmp["signal_R"] = pd.to_numeric(tmp["net_R_1m_mark_proxy"], errors="coerce").fillna(0.0)
        tmp["account_R"] = tmp["signal_R"] * (tmp["leverage_used"] / 10.0)
        tmp = tmp[tmp["leverage_used"] >= 1.0].copy()
        tmp["stressed_R"] = tmp["account_R"]
        for cid, sub in tmp.groupby("candidate_id", dropna=False):
            cand = sub[sub["window_role"].astype(str) == "candidate_event"]
            ctrl = sub[sub["window_role"].astype(str) == "control"]
            cs = summarize_group(cand, "stressed_R")
            ss = summarize_group(cand.assign(stressed_R=cand["signal_R"]), "stressed_R")
            xs = summarize_group(ctrl, "stressed_R")
            norm = xs["net_R"] * (cs["events"] / xs["events"]) if xs["events"] else np.nan
            rows.append({
                "candidate_id": cid,
                "sizing_model": name,
                "stress_band": "diagnostic",
                "candidate_events": cs["events"],
                "signal_R": ss["net_R"],
                "account_R": cs["net_R"],
                "control_account_R_normalized": norm,
                "account_uplift_R": cs["net_R"] - norm if np.isfinite(norm) else np.nan,
                "PF_account": cs["PF"],
                "avg_leverage_used": float(cand["leverage_used"].mean()) if not cand.empty else np.nan,
                "median_leverage_used": float(cand["leverage_used"].median()) if not cand.empty else np.nan,
                "leverage_reduces_notional": bool((cand["leverage_used"] < 10.0).any()) if not cand.empty else False,
                "listing_liquidation_evidence_status": "not_available_true_feed_no_depth_diagnostic_only",
            })
    write_csv(ctx.run_root / "stress/liquidation_dynamic_sizing_summary.csv", rows)
    write_text(ctx.run_root / "stress/liquidation_dynamic_sizing_report.md", "# Liquidation Buffer And Dynamic Sizing Stress\n\nDynamic sizing reports both normalized `signal_R` and reduced-notional `account_R`. Leverage changes are not allowed to improve signal edge; they only change account expression. True liquidation-feed history remains missing.\n")


def stage_comparison(ctx: RunContext) -> None:
    frames = []
    for p in [ctx.run_root / "stress/slippage_participation_summary.csv", ctx.run_root / "stress/stop_execution_summary.csv", ctx.run_root / "stress/missed_fill_gap_summary.csv"]:
        df = read_csv(p)
        if not df.empty:
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_csv(ctx.run_root / "comparison/control_normalized_stress_summary.csv", all_df)
    if all_df.empty:
        txt = "No stress comparison rows were available."
    else:
        rankable = all_df[all_df.get("rankable", pd.Series([True] * len(all_df))).astype(str).str.lower().isin(["true", "1", "yes"])]
        by_band = rankable.groupby(["candidate_id", "stress_band"], dropna=False).agg(rows=("scenario_id", "count"), positive=("beats_controls_after_stress", "sum"), max_uplift=("normalized_uplift_R", "max"), max_net_R=("candidate_net_R", "max")).reset_index()
        write_csv(ctx.run_root / "comparison/control_normalized_stress_by_band.csv", by_band)
        txt = f"Rankable scenario rows: `{len(rankable)}`. Control R is normalized to candidate-event count before comparison; raw control sums are not rankable."
    write_text(ctx.run_root / "comparison/control_normalized_stress_report.md", f"# Control-Normalized Stress Comparison\n\n{txt}\n")


def stage_portfolio(ctx: RunContext) -> None:
    comp = read_csv(ctx.run_root / "comparison/control_normalized_stress_summary.csv")
    rows: list[dict[str, Any]] = []
    if not comp.empty:
        candidates = comp[(comp.get("rankable", pd.Series([True] * len(comp))).astype(str).str.lower().isin(["true", "1", "yes"])) & (pd.to_numeric(comp.get("candidate_net_R", 0), errors="coerce") > 0)].copy()
        candidates = candidates.sort_values(["stress_band", "candidate_id", "candidate_net_R"], ascending=[True, True, False]).head(120)
        for _, cand in candidates.iterrows():
            n = max(int(float(cand.get("candidate_events", 0) or 0)), 1)
            mean_r = float(cand.get("candidate_mean_R", 0) or 0)
            for equity in EQUITY_CASES:
                for risk_pct in RISK_PCTS:
                    clipped_trade_growth = max(0.01, 1.0 + risk_pct * mean_r)
                    ending = float(equity) * (clipped_trade_growth ** min(n, 500))
                    rows.append({
                        "candidate_id": cand.get("candidate_id"),
                        "scenario_id": cand.get("scenario_id"),
                        "stress_band": cand.get("stress_band"),
                        "starting_equity": equity,
                        "risk_pct_equity": risk_pct,
                        "risk_label": "conservative_baseline" if risk_pct <= 0.01 else "aggressive_diagnostic_stress",
                        "ending_equity_proxy": ending,
                        "events_used_capped_at_500": min(n, 500),
                        "max_drawdown_proxy_unavailable_without_path_equity_curve": True,
                        "live_recommendation": False,
                    })
    write_csv(ctx.run_root / "portfolio/aggressive_overlay_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/aggressive_overlay_report.md", "# Aggressive Small-Account Overlay\n\nThis is a coarse proxy overlay from stressed mean R, not an order-book execution simulation and not a live sizing recommendation. Conservative baseline risk sizes include 0.25%, 0.50%, and 1.00%; larger risk sizes are diagnostic stress only.\n")


def classify_candidate(comp: pd.DataFrame, cid: str) -> tuple[str, str, str]:
    sub = comp[comp["candidate_id"].astype(str).eq(cid)].copy()
    if sub.empty:
        return "not_fairly_tested_missing_execution_depth", "no stress rows", "listing_candidate_unresolved_execution_depth_needed"
    rankable = sub[sub.get("rankable", pd.Series([True] * len(sub))).astype(str).str.lower().isin(["true", "1", "yes"])]
    moderate = rankable[(rankable["stress_band"].astype(str) == "moderate") & (rankable["beats_controls_after_stress"].astype(str).str.lower().isin(["true", "1", "yes"]))]
    severe = rankable[(rankable["stress_band"].astype(str) == "severe") & (rankable["beats_controls_after_stress"].astype(str).str.lower().isin(["true", "1", "yes"]))]
    extreme = rankable[(rankable["stress_band"].astype(str) == "extreme") & (rankable["beats_controls_after_stress"].astype(str).str.lower().isin(["true", "1", "yes"]))]
    if not severe.empty:
        return "survives_brutal_no_depth_stress", "positive net_R and control-normalized uplift under severe no-depth stress", "listing_candidate_survives_brutal_stress_needs_depth"
    if not moderate.empty:
        return "survives_moderate_stress_only", "positive net_R and control-normalized uplift under moderate no-depth stress but not severe", "listing_candidate_survives_moderate_stress_needs_depth"
    base_like = rankable[(rankable["scenario_family"].astype(str) == "slippage_participation") & (pd.to_numeric(rankable["candidate_net_R"], errors="coerce") > 0)]
    if not base_like.empty:
        return "path_edge_survives_but_execution_unresolved", "candidate remains positive in isolation but does not retain control-normalized uplift under punitive stress", "listing_candidate_unresolved_execution_depth_needed"
    return "fails_brutal_no_depth_stress_current_expression_only", "punitive no-depth proxy stress kills current expression; family not rejected", "listing_candidate_fails_brutal_stress_current_expression_only"


def stage_survival(ctx: RunContext) -> None:
    comp = read_csv(ctx.run_root / "comparison/control_normalized_stress_summary.csv")
    rows: list[dict[str, Any]] = []
    for cid in LISTING_IDS:
        label, reason, verdict = classify_candidate(comp, cid)
        rows.append({
            "candidate_id": cid,
            "survival_label": label,
            "reason": reason,
            "final_verdict_component": verdict,
            "true_depth_available": False,
            "public_trades_available": False,
            "missing_execution_depth_blocks_validation_language": True,
            "current_expression_only_failure_if_failed": label == "fails_brutal_no_depth_stress_current_expression_only",
        })
    if ctx.args.include_d4:
        rows.append({
            "candidate_id": D4_CANDIDATE_ID,
            "survival_label": "path_edge_survives_but_execution_unresolved",
            "reason": "D4 diagnostic is carried forward; no true liquidation-feed/depth replay in this bridge",
            "final_verdict_component": "d4_remains_execution_depth_carry_forward",
            "true_depth_available": False,
            "public_trades_available": False,
            "missing_execution_depth_blocks_validation_language": True,
            "current_expression_only_failure_if_failed": False,
        })
    write_csv(ctx.run_root / "decision/candidate_survival_table.csv", rows)
    write_text(ctx.run_root / "decision/candidate_survival_report.md", "# Candidate Survival Decision Table\n\nSurvival labels are punitive proxy diagnostics only. A failed stress label applies to the current expression, not the family. Missing true depth/trade/liquidation data remains a data blocker.\n")


def stage_voi(ctx: RunContext) -> None:
    comp = read_csv(ctx.run_root / "comparison/control_normalized_stress_summary.csv")
    rows: list[dict[str, Any]] = []
    for cid in LISTING_IDS:
        sub = comp[comp.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(cid)] if not comp.empty else pd.DataFrame()
        if sub.empty:
            dom = "missing_depth"
        else:
            severe = sub[sub.get("stress_band", pd.Series(dtype=str)).astype(str).eq("severe")]
            pos_severe = bool((severe.get("beats_controls_after_stress", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"])).any()) if not severe.empty else False
            dom = "vendor_depth_needed_to_measure_participation_and_slippage" if not pos_severe else "depth_data_still_needed_to_confirm_punitive_proxy"
        rows.append({
            "candidate_id": cid,
            "dominant_uncertainty": dom,
            "recommended_data": "historical_top_of_book;shallow_depth;public_trades",
            "vendor_pilot_priority": "high",
            "forward_capture_priority": "high",
            "reason": "No-depth stress cannot establish exchange-faithful slippage, participation, or stop execution.",
        })
    if ctx.args.include_d4:
        rows.append({"candidate_id": D4_CANDIDATE_ID, "dominant_uncertainty": "true_liquidation_feed_and_depth", "recommended_data": "historical_liquidations;top_of_book;shallow_depth;public_trades", "vendor_pilot_priority": "high", "forward_capture_priority": "high", "reason": "D4 requires liquidation history in addition to depth/trades."})
    write_csv(ctx.run_root / "voi/data_value_of_information_summary.csv", rows)
    write_text(ctx.run_root / "voi/data_value_of_information_report.md", "# Data Value Of Information\n\nThe most valuable next data is historical top-of-book/shallow-depth/public trades for listing candidates, and liquidation history plus depth/trades for D4. The no-depth bridge is not a substitute for vendor/live execution data.\n")


def stage_decision_report(ctx: RunContext) -> None:
    surv = read_csv(ctx.run_root / "decision/candidate_survival_table.csv")
    comp = read_csv(ctx.run_root / "comparison/control_normalized_stress_by_band.csv")
    verdicts = sorted(set(surv.get("final_verdict_component", pd.Series(dtype=str)).dropna().astype(str))) if not surv.empty else ["blocked_by_protocol_issue"]
    if any("survives_brutal" in x for x in surv.get("survival_label", pd.Series(dtype=str)).astype(str)):
        overall = "listing_candidate_survives_brutal_stress_needs_depth"
        micro = "micro_canary_possible_execution_only"
    elif any("survives_moderate" in x for x in surv.get("survival_label", pd.Series(dtype=str)).astype(str)):
        overall = "listing_candidate_survives_moderate_stress_needs_depth"
        micro = "micro_canary_not_recommended"
    elif not surv.empty:
        overall = "listing_candidate_unresolved_execution_depth_needed"
        micro = "micro_canary_not_recommended"
    else:
        overall = "blocked_by_protocol_issue"
        micro = "micro_canary_not_recommended"
    decision = {
        "run_root": str(ctx.run_root),
        "protected_holdout_touched": False,
        "rankable_scope": "full_hold_only",
        "overall_listing_verdict": overall,
        "d4_verdict": "d4_remains_execution_depth_carry_forward" if ctx.args.include_d4 else "not_included",
        "vendor_data_verdict": "vendor_pilot_high_priority",
        "forward_capture_verdict": "forward_capture_high_priority",
        "micro_canary_verdict": micro,
        "allowed_verdicts_only": True,
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    table_md = surv.to_markdown(index=False) if not surv.empty else "No survival rows."
    band_md = comp.to_markdown(index=False) if not comp.empty else "No band summary rows."
    write_text(
        ctx.run_root / "QLMG_BRUTAL_NO_DEPTH_STRESS_REPORT.md",
        f"# QLMG Brutal No-Depth Stress Report\n\n"
        f"## Scope\n\n- run_root: `{ctx.run_root}`\n- train_only: `true`\n- final_holdout_touched: `false`\n- rankable_scope: `full_hold` only; `core_24h` excluded.\n- no external data downloaded: `true`\n- execution-depth status: no true order book, public-trade, or liquidation-feed replay.\n\n"
        f"## Candidate Survival\n\n{table_md}\n\n"
        f"## Stress Band Summary\n\n{band_md}\n\n"
        f"## Interpretation\n\nThis bridge applies harsh proxy haircuts to existing 1m mark/OHLCV full-hold replay. It can identify expressions that are robust enough to justify vendor/live execution-data work, but it cannot validate exchange-faithful execution. Candidate failures are current-expression-only failures; they are not family-level rejection. D4 remains an execution-depth/liquidation-data carry-forward only when included.\n\n"
        f"## Next Step\n\nRun a vendor or forward-capture pilot for historical top-of-book, shallow depth, public trades, and liquidation history on the prioritized windows before any validation-style language is considered.\n",
    )


def copy_if_small(src: Path, dst_dir: Path, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    rec = {"source_path": str(src), "bundle_path": "", "included": False, "reason": "missing"}
    if not src.exists():
        return rec
    if src.stat().st_size > max_bytes:
        rec["reason"] = "too_large_paths_only"
        return rec
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.relative_to(src.parents[1] if src.is_relative_to(src.parents[1]) else src.parent).as_posix().replace("/", "__")
    shutil.copy2(src, dst)
    rec.update({"bundle_path": str(dst), "included": True, "reason": "included"})
    return rec


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        ctx.run_root / "QLMG_BRUTAL_NO_DEPTH_STRESS_REPORT.md",
        ctx.run_root / "decision_summary.json",
        ctx.run_root / "preflight/resource_guard_report.md",
        ctx.run_root / "reconstruction/reconstruction_report.md",
        ctx.run_root / "baseline/baseline_reproduction_summary.csv",
        ctx.run_root / "stress_design/stress_design_contract.json",
        ctx.run_root / "stress/slippage_participation_summary.csv",
        ctx.run_root / "stress/stop_execution_summary.csv",
        ctx.run_root / "stress/missed_fill_gap_summary.csv",
        ctx.run_root / "stress/liquidation_dynamic_sizing_summary.csv",
        ctx.run_root / "comparison/control_normalized_stress_summary.csv",
        ctx.run_root / "decision/candidate_survival_table.csv",
        ctx.run_root / "voi/data_value_of_information_summary.csv",
        ctx.run_root / "notifications/telegram_readiness_report.md",
    ]
    rows = [copy_if_small(p, bundle) for p in include]
    write_csv(bundle / "artifact_path_index.csv", rows)
    write_text(bundle / "README.md", f"# Compact Review Bundle\n\nRun root: `{ctx.run_root}`. Large source replay parquet files are not copied; paths are preserved in the artifact index and preflight manifest.\n")


STAGE_FUNCS = {
    "preflight-and-artifact-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "candidate-and-control-reconstruction": stage_reconstruction,
    "baseline-replay-reproduction": stage_baseline,
    "no-depth-execution-stress-design": stage_stress_design,
    "slippage-and-participation-stress": stage_slippage_participation,
    "stop-execution-and-same-minute-stress": stage_stop_execution,
    "missed-fill-and-gap-through-stress": stage_missed_fill_gap,
    "liquidation-buffer-and-dynamic-sizing-stress": stage_liq_dynamic_sizing,
    "control-normalized-stress-comparison": stage_comparison,
    "aggressive-small-account-overlay": stage_portfolio,
    "candidate-survival-decision-table": stage_survival,
    "data-value-of-information-report": stage_voi,
    "decision-report": stage_decision_report,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and stage_complete(ctx.run_root, stage):
        return
    enforce_resources(ctx, stage)
    ctx.notifier.send("QLMG brutal no-depth stress stage start", stage)
    try:
        STAGE_FUNCS[stage](ctx)
    except Exception as exc:
        ctx.notifier.send("QLMG brutal no-depth stress stage failure", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        raise
    mark_done(ctx.run_root, stage)
    ctx.notifier.send("QLMG brutal no-depth stress stage complete", stage)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "run_root_reason": reason, "args": vars(args), "start": str(start), "end": str(end)})
    notifier.send("QLMG brutal no-depth stress run start", f"run_root={run_root}")
    for stage in stage_list(args.stage):
        run_stage(ctx, stage)
    notifier.send("QLMG brutal no-depth stress run complete", f"run_root={run_root}")
    try:
        (run_root / "watch_status.json").write_text(json.dumps({"status": "complete", "run_root": str(run_root), "ts_utc": utc_now()}, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass
    print(f"[run_root] {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
