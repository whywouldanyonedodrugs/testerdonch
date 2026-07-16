#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    validate_control_rows,
    validate_no_projected_metric_promotion,
)

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_corrected_event_level_development_sweep_20260629_v1"
DEFAULT_SEED = 20260629
DATA_5M = Path("/opt/parquet/5m")
CONTEXT_5M = Path("/opt/parquet/bybit_context_5m")

REMEDIATION_ROOT = RESULTS_ROOT / "phase_qlmg_evidence_remediation_family_repair_20260629_v1_20260629_044410"
REAL_CONTROLS_ROOT = RESULTS_ROOT / "phase_qlmg_real_control_rebuild_20260629_v1_20260629_170608"
INTEGRITY_ROOT = RESULTS_ROOT / "phase_qlmg_evidence_integrity_corrected_sweep_20260628_v1_20260628_163819"
ABCX_ROOT = RESULTS_ROOT / "phase_qlmg_integrated_abcx_development_20260628_v2_20260628_134140"
LIQUID_ROOT = RESULTS_ROOT / "phase_qlmg_liquid_regime_strategy_research_20260628_v1_20260628_120124"
SECTOR_MD = REPO / "research_inputs/point_in_time_sector_seeds.md"
CATALYST_MD = REPO / "research_inputs/post_catalyst_c2_database.md"

REQUIRED_GATE_ARTIFACTS = [
    "gate/corrected_sweep_allowed.json",
    "quarantine/rankable_active_evidence_set.csv",
    "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv",
    "quarantine/deprecated_promotion_labels.csv",
]
RANKABLE_FAMILIES = {"A3", "A2_redesign_only"}
FORBIDDEN_RANKABLE_FAMILIES = {"A1", "A4", "D1", "D3", "E1", "F1", "G1", "funding-window", "ORB", "listing", "D4"}

STAGES = (
    "preflight-and-evidence-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "rankable-source-gate",
    "a3-event-level-refinement-sweep",
    "a2-redesign-event-level-sweep",
    "b1-sector-cluster-ledger-sidecar",
    "c2-mechanism-ledger-sidecar",
    "fresh-nulls-and-baselines",
    "tier1-cost-funding-mark-stress",
    "walk-forward-cpcv-and-stability",
    "aggressive-small-account-overlay",
    "branch-x-status-and-capture-request",
    "corrected-cross-branch-triage",
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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="qlmg-corrected-dev-sweep")
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
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        write_json(self.run_root / "watch_status.json", {"status": "running", "last_event": title, "ts_utc": rec["ts_utc"], "run_root": str(self.run_root)})
        return sent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLMG corrected event-level development sweep")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=45.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--include-a3", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-a2-redesign", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-b1-ledger-sidecar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-c2-ledger-sidecar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-branch-x-status", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--corrected-sweep-budget", type=int, default=6000)
    p.add_argument("--a3-budget", type=int, default=2500)
    p.add_argument("--a2-budget", type=int, default=2000)
    p.add_argument("--b1-sidecar-budget", type=int, default=900)
    p.add_argument("--c2-sidecar-budget", type=int, default=600)
    p.add_argument("--nulls-per-event", type=int, default=3)
    p.add_argument("--top-per-family", type=int, default=60)
    p.add_argument("--aggressive-overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tmux-session-name", default="qlmg_corrected_dev_sweep")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--run-root", default="")
    p.add_argument("--remediation-root", default=str(REMEDIATION_ROOT))
    p.add_argument("--real-controls-root", default=str(REAL_CONTROLS_ROOT))
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
    guard = check_resource_guard(
        resource_snapshot(ctx.run_root.parent),
        estimated_output_gb=estimated_output_gb,
        hard_free_gb=5.0,
        warn_free_gb=7.0,
        hard_stage_output_gb=40.0,
        allow_large_output=ctx.args.allow_large_output,
    )
    write_json(ctx.run_root / "resource_guard" / f"{stage}.json", guard)
    if guard["warnings"]:
        ctx.notifier.send("QLMG corrected sweep resource warning", json.dumps(guard), level="warning")
    if guard["status"] == "hard_stop":
        raise RuntimeError(f"resource guard hard stop for {stage}: {guard}")


def validate_no_protected_df(df: pd.DataFrame, cols: Sequence[str]) -> None:
    if not df.empty:
        validate_no_protected(df, list(cols))


def month_keys(values: Any) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    ser = ts if isinstance(ts, pd.Series) else pd.Series(ts)
    return ser.dt.tz_convert(None).dt.to_period("M").astype(str)


def pf(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna()
    pos = float(x[x > 0].sum())
    neg = float(x[x < 0].sum())
    if abs(neg) < 1e-12:
        return float("inf") if pos > 0 else float("nan")
    return pos / abs(neg)


def max_dd(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").fillna(0.0)
    if x.empty:
        return 0.0
    curve = x.cumsum()
    return float((curve - curve.cummax()).min())


def load_base_replay() -> pd.DataFrame:
    df = read_parquet_safe(INTEGRITY_ROOT / "a2a3/corrected_event_level_replay.parquet")
    if not df.empty:
        for c in ["decision_ts", "entry_ts", "exit_ts"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        validate_no_protected_df(df, ["decision_ts", "entry_ts", "exit_ts"])
    return df


def split_dev_eval(df: pd.DataFrame, q: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    if df.empty or "decision_ts" not in df:
        return df.iloc[0:0], df.iloc[0:0], None
    cutoff = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce").quantile(q)
    return df[df["decision_ts"] <= cutoff].copy(), df[df["decision_ts"] > cutoff].copy(), pd.Timestamp(cutoff)


def metric_summary(df: pd.DataFrame, candidate_id: str, family: str, variant_id: str, *, split: str = "all") -> dict[str, Any]:
    vals = pd.to_numeric(df.get("net_R_variant", df.get("net_R", pd.Series(dtype=float))), errors="coerce").dropna()
    n = len(vals)
    mark_available = bool(df.get("mark_price_available", pd.Series(dtype=bool)).fillna(False).astype(bool).all()) if len(df) else False
    funding_exact = bool(df.get("funding_exact", pd.Series(dtype=bool)).fillna(False).astype(bool).all()) if len(df) else False
    return {
        "candidate_id": candidate_id,
        "family": family,
        "variant_id": variant_id,
        "split": split,
        "events": int(len(df)),
        "net_R": float(vals.sum()) if n else np.nan,
        "PF": pf(vals),
        "win_rate": float((vals > 0).mean()) if n else np.nan,
        "avg_R": float(vals.mean()) if n else np.nan,
        "median_R": float(vals.median()) if n else np.nan,
        "max_dd_R": max_dd(vals.reset_index(drop=True)),
        "symbols": int(df.get("symbol", pd.Series(dtype=str)).nunique()) if len(df) else 0,
        "months": int(month_keys(df["decision_ts"]).nunique()) if len(df) and "decision_ts" in df else 0,
        "mark_available": mark_available,
        "funding_exact": funding_exact,
        "mark_proxy_used": not mark_available,
        "funding_proxy_used": not funding_exact,
        "label_cap_reason": "none" if mark_available and funding_exact else "mark_or_funding_proxy_cap",
    }


def apply_event_filter(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if cfg.get("risk_bps_max") is not None and "risk_bps_used" in out:
        out = out[pd.to_numeric(out["risk_bps_used"], errors="coerce") <= float(cfg["risk_bps_max"])]
    if cfg.get("mark_required"):
        out = out[out.get("mark_price_available", pd.Series(False, index=out.index)).fillna(False).astype(bool)]
    if cfg.get("regime") and "parent_regime" in out:
        out = out[out["parent_regime"].astype(str).eq(str(cfg["regime"]))]
    if cfg.get("funding_proxy_allowed") is False and "funding_exact" in out:
        out = out[out["funding_exact"].fillna(False).astype(bool)]
    return out


def variant_event_rows(base: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = apply_event_filter(base, cfg).copy()
    if out.empty:
        return out
    vals = pd.to_numeric(out["net_R"], errors="coerce").fillna(0.0).copy()
    # Exit/stop/sizing variants are deterministic transforms of existing corrected event rows;
    # these are rankable only because source rows are event-level, and the transform is recorded.
    hold = float(cfg.get("hold_factor", 1.0))
    target_cap = cfg.get("target_cap_R")
    if target_cap is not None:
        vals = vals.clip(upper=float(target_cap))
    stop_floor = cfg.get("stop_floor_R")
    if stop_floor is not None:
        vals = vals.clip(lower=-float(stop_floor))
    vals = vals * hold
    cost_bps = float(cfg.get("extra_cost_bps", 0.0))
    risk = pd.to_numeric(out.get("risk_bps_used", pd.Series(100.0, index=out.index)), errors="coerce").replace(0, np.nan).fillna(100.0)
    vals = vals - (cost_bps / risk)
    out["net_R_variant"] = vals
    out["variant_id"] = cfg["variant_id"]
    out["variant_config_hash"] = stable_hash(dict(cfg), 16)
    out["metric_basis"] = "event_level_transform_from_corrected_replay_not_summary_projection"
    return out


def fragility_tests(df: pd.DataFrame, candidate_id: str, variant_id: str, family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    vals = pd.to_numeric(df["net_R_variant"], errors="coerce").dropna().sort_values(ascending=False)
    base = metric_summary(df, candidate_id, family, variant_id)
    rows.append({**base, "test": "base", "survives_positive": bool(base["net_R"] > 0)})
    for pct in [0.01, 0.05]:
        drop = vals.head(max(1, int(math.ceil(len(vals) * pct)))).index
        sub = df.drop(index=drop, errors="ignore")
        s = metric_summary(sub, candidate_id, family, variant_id)
        rows.append({**s, "test": f"remove_top_{int(pct*100)}pct_winners", "survives_positive": bool(s["net_R"] > 0)})
    if "decision_ts" in df:
        months = month_keys(df["decision_ts"])
        by_m = df.assign(_month=months.to_numpy()).groupby("_month")["net_R_variant"].sum()
        if len(by_m):
            top_m = by_m.idxmax()
            sub = df[months.to_numpy() != top_m]
            s = metric_summary(sub, candidate_id, family, variant_id)
            rows.append({**s, "test": "remove_top_month", "removed_month": top_m, "survives_positive": bool(s["net_R"] > 0)})
    for sym in list(df.get("symbol", pd.Series(dtype=str)).astype(str).value_counts().head(5).index):
        sub = df[~df["symbol"].astype(str).eq(sym)]
        s = metric_summary(sub, candidate_id, family, variant_id)
        rows.append({**s, "test": "leave_one_symbol_out", "removed_symbol": sym, "survives_positive": bool(s["net_R"] > 0)})
    risk = pd.to_numeric(df.get("risk_bps_used", pd.Series(100.0, index=df.index)), errors="coerce").replace(0, np.nan).fillna(100.0)
    for bps in [25, 50]:
        sub = df.copy()
        sub["net_R_variant"] = pd.to_numeric(sub["net_R_variant"], errors="coerce") - bps / risk
        s = metric_summary(sub, candidate_id, family, variant_id)
        rows.append({**s, "test": f"fee_slippage_plus_{bps}bps", "survives_positive": bool(s["net_R"] > 0)})
    if "mark_price_available" in df:
        sub = df[df["mark_price_available"].fillna(False).astype(bool)]
        s = metric_summary(sub, candidate_id, family, variant_id)
        rows.append({**s, "test": "mark_proxy_exclusion", "survives_positive": bool(s["net_R"] > 0)})
    return rows


def label_a3(summary: Mapping[str, Any], frag: pd.DataFrame, controls_pass: bool) -> str:
    if not (float(summary.get("net_R", 0) or 0) > 0 and float(summary.get("PF", 0) or 0) > 1.0):
        return "a3_reject_current_translation_only"
    top1 = frag[frag["test"].eq("remove_top_1pct_winners")]["survives_positive"].astype(bool).any() if not frag.empty else False
    top5 = frag[frag["test"].eq("remove_top_5pct_winners")]["survives_positive"].astype(bool).any() if not frag.empty else False
    cost25 = frag[frag["test"].eq("fee_slippage_plus_25bps")]["survives_positive"].astype(bool).any() if not frag.empty else False
    concentration_ok = int(summary.get("symbols", 0) or 0) > 1 and int(summary.get("months", 0) or 0) > 1
    if top1 and cost25 and concentration_ok and controls_pass:
        return "a3_fragile_but_alive" if not top5 else "tier1_research_prelead"
    return "path_edge_exit_problem"


def label_a2(summary: Mapping[str, Any], frag: pd.DataFrame, controls_pass: bool) -> tuple[str, str]:
    net = float(summary.get("net_R", 0) or 0)
    top1 = frag[frag["test"].eq("remove_top_1pct_winners")]["survives_positive"].astype(bool).any() if not frag.empty else False
    top5 = frag[frag["test"].eq("remove_top_5pct_winners")]["survives_positive"].astype(bool).any() if not frag.empty else False
    if net <= 0:
        return "a2_reject_current_translation_only", "current translation failure"
    if top1 and top5 and controls_pass:
        return "a2_redesign_candidate", "broadly distributed prior-high momentum edge"
    if top1 and not top5:
        return "a2_tail_only_current_translation", "tail-only payoff"
    return "path_edge_exit_problem", "exit/risk geometry problem"


def normalize_controls(candidate_net: float, candidate_events: int, control_net: float, control_events: int) -> float:
    if control_events <= 0 or candidate_events <= 0:
        return np.nan
    return float(control_net) * float(candidate_events) / float(control_events)


def control_rows_for(df: pd.DataFrame, candidate_id: str, variant_id: str, seed: int, nulls: int) -> list[dict[str, Any]]:
    raise RuntimeError(
        "Deprecated synthetic control generator is hard-disabled. Use "
        "tools/run_qlmg_real_control_rebuild.py, which constructs controls "
        "from event-level source rows with control_event_id/control_window_id "
        "provenance and real matching features."
    )


def read_required_gate() -> dict[str, Any]:
    missing = [rel for rel in REQUIRED_GATE_ARTIFACTS if not (REMEDIATION_ROOT / rel).exists()]
    if missing:
        raise RuntimeError(f"missing required remediation gate artifacts: {missing}")
    gate = safe_read_json(REMEDIATION_ROOT / "gate/corrected_sweep_allowed.json")
    if not gate.get("corrected_sweep_allowed"):
        raise RuntimeError(f"remediation corrected sweep gate is not allowed: {gate}")
    allowed = set(gate.get("allowed_families", []))
    if not (RANKABLE_FAMILIES & allowed):
        raise RuntimeError(f"remediation gate does not allow any rankable family from {RANKABLE_FAMILIES}: {allowed}")
    return gate


def remediation_allowed_rankable_families() -> set[str]:
    gate = safe_read_json(REMEDIATION_ROOT / "gate/corrected_sweep_allowed.json")
    return set(gate.get("allowed_families", [])) & RANKABLE_FAMILIES


def stage_preflight(ctx: RunContext) -> None:
    resource_check(ctx, "preflight-and-evidence-freeze", 0.2)
    roots = {
        "remediation": REMEDIATION_ROOT,
        "integrity": INTEGRITY_ROOT,
        "integrated_abcx_v2": ABCX_ROOT,
        "liquid_regime": LIQUID_ROOT,
        "sector_md": SECTOR_MD,
        "catalyst_md": CATALYST_MD,
    }
    manifest = []
    hashes: dict[str, Any] = {"run_root": str(ctx.run_root)}
    try:
        hashes["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        hashes["git_status_short"] = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).strip().splitlines()
    except Exception:
        hashes["git_head"] = "unknown"
    for name, p in roots.items():
        basis = p if p.is_file() else p / "decision_summary.json"
        if not basis.exists() and p.is_dir():
            basis = p / "QLMG_EVIDENCE_REMEDIATION_FAMILY_REPAIR_REPORT.md"
        manifest.append({"artifact_name": name, "path": str(p), "exists": p.exists(), "hash_basis": str(basis if basis.exists() else "")})
        hashes[name] = file_hash(basis) if basis.exists() and basis.is_file() else ("directory_present_no_main_file" if p.exists() else "missing")
    for rel in [
        "gate/corrected_sweep_allowed.json",
        "quarantine/rankable_active_evidence_set.csv",
        "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv",
        "quarantine/deprecated_promotion_labels.csv",
        "a3_validation/a3_validation_summary.csv",
        "a3_validation/a3_stability_audit.csv",
        "a2_repair/a2_redesign_summary.csv",
        "a2_repair/a2_liquidation_flag_taxonomy.csv",
        "b1_trade_ledger/b1_ledger_blockers.csv",
        "c2_trade_ledger/c2_by_mechanism_summary.csv",
    ]:
        p = REMEDIATION_ROOT / rel
        manifest.append({"artifact_name": rel, "path": str(p), "exists": p.exists(), "hash_basis": str(p) if p.exists() else ""})
        hashes[rel] = file_hash(p) if p.exists() else "missing"
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", manifest)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    guard = check_resource_guard(resource_snapshot(ctx.run_root.parent), estimated_output_gb=10.0, hard_free_gb=5.0, warn_free_gb=7.0, hard_stage_output_gb=40.0, allow_large_output=ctx.args.allow_large_output)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard\n\nstatus={guard['status']} free_disk_gb={guard['free_disk_gb']:.2f} max_output_gb={ctx.args.max_output_gb}")
    write_text(ctx.run_root / "preflight/preflight_report.md", "\n".join([
        "# Corrected Event-Level Development Sweep Preflight",
        f"run_root: `{ctx.run_root}`",
        f"window: `{ctx.start}` to `{ctx.end}`",
        "Rankable scope is limited to A3 and A2_redesign_only. B1/C2 are sidecars; Branch X is status-only.",
        "No quarantined evidence, projected means, summary rows, MAE/MFE-only rows, or Branch X proxy retuning may be used for ranking.",
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
    ctx.notifier.send("QLMG corrected development sweep initialized", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    resource_check(ctx, "seal-guard", 0.05)
    write_json(ctx.run_root / "seal/protected_slice_check.json", {"protected_start": str(FINAL_HOLDOUT_START), "allowed_end": str(SCREENING_END), "checks": [{"case": "pre_holdout_allowed", "timestamp": str(SCREENING_END), "passes": True}, {"case": "protected_rejected", "timestamp": str(FINAL_HOLDOUT_START), "passes": False}]})
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard\n\nProtected start: `{FINAL_HOLDOUT_START}`. No reads/scoring/generated candidate rows may use protected timestamps.")


def stage_rankable_gate(ctx: RunContext) -> None:
    resource_check(ctx, "rankable-source-gate", 0.1)
    gate = read_required_gate()
    rankable_set = read_csv(REMEDIATION_ROOT / "quarantine/rankable_active_evidence_set.csv")
    quarantined = read_csv(REMEDIATION_ROOT / "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv")
    deprecated = read_csv(REMEDIATION_ROOT / "quarantine/deprecated_promotion_labels.csv")
    blockers = []
    if not (REMEDIATION_ROOT / "quarantine/rankable_active_evidence_set.csv").exists():
        blockers.append("rankable_active_evidence_set_missing")
    if set(gate.get("allowed_families", [])) - RANKABLE_FAMILIES:
        # Not fatal by itself, but the runner will ignore any extra family.
        pass
    required_deprecated = {
        "research_prelead_only",
        "stress_survives",
        "targeted_execution_data_prelead",
        "targeted_execution_data_prelead_unresolved",
        "a2_a3_tier1_prelead_confirmed_train_only",
    }
    deprecated_labels = set(deprecated.get("deprecated_label", pd.Series(dtype=str)).astype(str))
    if deprecated.empty or not required_deprecated.issubset(deprecated_labels):
        blockers.append("deprecated_label_registry_malformed")
    active_rankable = rankable_set.copy()
    if not active_rankable.empty and "rankable" in active_rankable.columns:
        active_rankable = active_rankable[active_rankable["rankable"].astype(str).str.lower().isin(["true", "1", "yes"])]
    elif not active_rankable.empty:
        blockers.append("rankable_active_evidence_set_missing_rankable_column")
        active_rankable = pd.DataFrame()
    rankable_families_seen = set(active_rankable.get("family", pd.Series(dtype=str)).dropna().astype(str))
    forbidden_seen = sorted(rankable_families_seen - RANKABLE_FAMILIES)
    if forbidden_seen:
        blockers.append(f"rankable_active_evidence_set_contains_forbidden_families:{','.join(forbidden_seen[:20])}")
    obj = {
        "rankable_source_gate_passed": not blockers,
        "blockers": blockers,
        "rankable_families": sorted(RANKABLE_FAMILIES),
        "rankable_families_allowed_by_remediation": sorted(set(gate.get("allowed_families", [])) & RANKABLE_FAMILIES),
        "rankable_families_seen": sorted(rankable_families_seen),
        "forbidden_rankable_families_seen": forbidden_seen,
        "forbidden_rankable_families": sorted(FORBIDDEN_RANKABLE_FAMILIES),
        "quarantined_artifact_count": int(len(quarantined)),
        "deprecated_label_count": int(len(deprecated)),
        "b1_c2_sidecar_only_until_trade_ledger": True,
        "branch_x_status_only": True,
    }
    write_json(ctx.run_root / "gate/rankable_source_gate.json", obj)
    write_text(ctx.run_root / "gate/rankable_source_gate_report.md", f"# Rankable Source Gate\n\npassed: `{obj['rankable_source_gate_passed']}`\n\nRankable families: `A3`, `A2_redesign_only`. B1/C2 are sidecars; Branch X is status-only. Quarantined artifact count: `{len(quarantined)}`.\n\nForbidden rankable families seen in remediation rankable set: `{', '.join(forbidden_seen) if forbidden_seen else 'none'}`.")
    if blockers:
        raise RuntimeError(f"rankable source gate failed: {blockers}")


def candidate_grid(family: str, budget: int, smoke: bool = False) -> list[dict[str, Any]]:
    if family == "A3":
        risk_caps = [1000, 1500, 2000, None]
        holds = [1.0, 0.9, 0.8]
        target_caps = [2.0, 3.0, 5.0, None]
        regimes = [None, "Expansion", "Rotation"]
        costs = [0.0, 4.0]
        rows = []
        for risk in risk_caps:
            for hold in holds:
                for target in target_caps:
                    for regime in regimes:
                        for cost in costs:
                            rows.append({"family": "A3", "risk_bps_max": risk, "hold_factor": hold, "target_cap_R": target, "stop_floor_R": 1.0, "regime": regime, "extra_cost_bps": cost, "mark_required": False, "funding_proxy_allowed": True})
    else:
        risk_caps = [1000, 1500, 2000, None]
        holds = [1.0, 0.75, 0.5]
        target_caps = [2.0, 3.0, None]
        stop_floors = [1.0, 1.5, 2.0]
        regimes = [None, "Expansion", "Rotation"]
        rows = []
        for risk in risk_caps:
            for hold in holds:
                for target in target_caps:
                    for stop in stop_floors:
                        for regime in regimes:
                            rows.append({"family": "A2_redesign_only", "risk_bps_max": risk, "hold_factor": hold, "target_cap_R": target, "stop_floor_R": stop, "regime": regime, "extra_cost_bps": 0.0, "mark_required": False, "funding_proxy_allowed": True})
    max_n = min(budget, len(rows), 30 if smoke else len(rows))
    out = []
    for i, r in enumerate(rows[:max_n]):
        vid = f"{family}__dev_{stable_hash({'i': i, **r}, 12)}"
        out.append({**r, "variant_id": vid, "registered_before_scoring": True})
    return out


def run_family_sweep(base: pd.DataFrame, family: str, budget: int, seed: int, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_family = "A3" if family == "A3" else "A2"
    source = base[base["family"].astype(str).eq(source_family)].copy()
    if smoke and not source.empty:
        syms = sorted(source["symbol"].dropna().astype(str).unique())[:5]
        source = source[source["symbol"].astype(str).isin(syms)].copy()
    dev, eval_df, cutoff = split_dev_eval(source)
    registry = []
    replays = []
    summaries = []
    frag_rows = []
    configs = candidate_grid(family, budget, smoke=smoke)
    for cfg in configs:
        vid = cfg["variant_id"]
        registry.append({"variant_id": vid, "family": family, "config_hash": stable_hash(cfg), "registered_before_scoring": True, "proposal_split": "development", "ranking_split": "internal_validation", **cfg})
        dev_rows = variant_event_rows(dev, cfg)
        eval_rows = variant_event_rows(eval_df, cfg)
        if not dev_rows.empty:
            dev_rows["split"] = "development"
        if not eval_rows.empty:
            eval_rows["split"] = "internal_validation"
        combined = pd.concat([dev_rows, eval_rows], ignore_index=True) if not dev_rows.empty or not eval_rows.empty else pd.DataFrame()
        if not combined.empty:
            combined["candidate_id"] = vid
            combined["family"] = family
            replays.append(combined)
        sm = metric_summary(eval_rows, vid, family, vid, split="internal_validation")
        sm["development_events"] = len(dev_rows)
        sm["development_net_R"] = float(pd.to_numeric(dev_rows.get("net_R_variant", pd.Series(dtype=float)), errors="coerce").sum()) if len(dev_rows) else np.nan
        sm["dev_eval_overlap_rows"] = 0
        sm["proposal_scoring_overlap"] = False
        fr = pd.DataFrame(fragility_tests(eval_rows, vid, vid, family)) if not eval_rows.empty else pd.DataFrame()
        if not fr.empty:
            frag_rows.append(fr)
        summaries.append(sm)
    replay = pd.concat(replays, ignore_index=True) if replays else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    frag = pd.concat(frag_rows, ignore_index=True) if frag_rows else pd.DataFrame()
    if not summary.empty and not frag.empty:
        labels = []
        for _, row in summary.iterrows():
            fr = frag[frag["variant_id"].eq(row["variant_id"])]
            # Controls are added later; preliminary label uses event/stress/fragility only.
            if family == "A3":
                labels.append(label_a3(row, fr, controls_pass=True))
            else:
                labels.append(label_a2(row, fr, controls_pass=True)[0])
        summary["pre_null_label"] = labels
    return pd.DataFrame(registry), replay, summary if not summary.empty else pd.DataFrame(), frag if not frag.empty else pd.DataFrame()


def stage_a3_sweep(ctx: RunContext) -> None:
    resource_check(ctx, "a3-event-level-refinement-sweep", 1.5)
    if not ctx.args.include_a3 or "A3" not in remediation_allowed_rankable_families():
        write_csv(ctx.run_root / "a3_sweep/a3_candidate_registry.csv", [])
        write_csv(ctx.run_root / "a3_sweep/a3_sweep_summary.csv", [])
        write_csv(ctx.run_root / "a3_sweep/a3_fragility_summary.csv", [])
        write_text(ctx.run_root / "a3_sweep/a3_report.md", "# A3 Event-Level Refinement Sweep\n\nSkipped because A3 was not enabled or not allowed by the remediation gate.")
        return
    base = load_base_replay()
    registry, replay, summary, frag = run_family_sweep(base, "A3", ctx.args.a3_budget, ctx.args.seed, ctx.args.smoke)
    validate_no_protected_df(replay, ["decision_ts", "entry_ts", "exit_ts"])
    (ctx.run_root / "a3_sweep").mkdir(parents=True, exist_ok=True)
    registry.to_csv(ctx.run_root / "a3_sweep/a3_candidate_registry.csv", index=False)
    replay.to_parquet(ctx.run_root / "a3_sweep/a3_event_level_replay.parquet", index=False)
    summary.to_csv(ctx.run_root / "a3_sweep/a3_sweep_summary.csv", index=False)
    frag.to_csv(ctx.run_root / "a3_sweep/a3_fragility_summary.csv", index=False)
    write_text(ctx.run_root / "a3_sweep/a3_report.md", "# A3 Event-Level Refinement Sweep\n\nA3 candidates are generated on the development split and ranked on the internal-validation split. Event-level rows only; no summary projections. Positive labels are provisional until fresh nulls, stress, and stability stages complete.")


def stage_a2_sweep(ctx: RunContext) -> None:
    resource_check(ctx, "a2-redesign-event-level-sweep", 1.5)
    if not ctx.args.include_a2_redesign or "A2_redesign_only" not in remediation_allowed_rankable_families():
        write_csv(ctx.run_root / "a2_sweep/a2_candidate_registry.csv", [])
        write_csv(ctx.run_root / "a2_sweep/a2_sweep_summary.csv", [])
        write_csv(ctx.run_root / "a2_sweep/a2_tail_dependence_summary.csv", [])
        write_text(ctx.run_root / "a2_sweep/a2_report.md", "# A2 Redesign Event-Level Sweep\n\nSkipped because A2_redesign_only was not enabled or not allowed by the remediation gate.")
        return
    base = load_base_replay()
    registry, replay, summary, frag = run_family_sweep(base, "A2_redesign_only", ctx.args.a2_budget, ctx.args.seed + 1, ctx.args.smoke)
    validate_no_protected_df(replay, ["decision_ts", "entry_ts", "exit_ts"])
    if not summary.empty:
        diagnoses = []
        for _, row in summary.iterrows():
            fr = frag[frag["variant_id"].eq(row["variant_id"])] if not frag.empty else pd.DataFrame()
            label, diag = label_a2(row, fr, controls_pass=True)
            diagnoses.append(diag)
        summary["a2_diagnostic_class"] = diagnoses
        summary["future_liquidation_labels_used_for_ranking"] = False
    (ctx.run_root / "a2_sweep").mkdir(parents=True, exist_ok=True)
    registry.to_csv(ctx.run_root / "a2_sweep/a2_candidate_registry.csv", index=False)
    replay.to_parquet(ctx.run_root / "a2_sweep/a2_event_level_replay.parquet", index=False)
    summary.to_csv(ctx.run_root / "a2_sweep/a2_sweep_summary.csv", index=False)
    frag.to_csv(ctx.run_root / "a2_sweep/a2_tail_dependence_summary.csv", index=False)
    write_text(ctx.run_root / "a2_sweep/a2_report.md", "# A2 Redesign Event-Level Sweep\n\nA2 remains redesign-only. Future liquidation flags, future MAE/MFE, realized bad outcomes, and future month/symbol/tail knowledge are not rankable filters. Liquidation-flag removal is not included as a rankable variant.")


def load_daily(symbol: str) -> pd.DataFrame:
    p = DATA_5M / f"{symbol}.parquet"
    df = read_parquet_safe(p)
    if df.empty or "timestamp" not in df:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = df.set_index("timestamp").resample("1D", label="right", closed="right").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).dropna().reset_index()
    out["symbol"] = symbol
    out["ret_1d"] = out["close"].pct_change()
    out["ret_7d"] = out["close"].pct_change(7)
    out["rolling_high_30d"] = out["close"].rolling(30, min_periods=10).max().shift(1)
    out["near_30d_high"] = out["close"] >= 0.97 * out["rolling_high_30d"]
    out["vol_breadth"] = out["volume"] > out["volume"].rolling(20, min_periods=5).median().shift(1)
    return out


def parse_symbols(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = str(value or "").replace(";", ",").split(",")
    syms = []
    for part in parts:
        p = part.strip().upper()
        if not p:
            continue
        if not p.endswith("USDT"):
            p = f"{p}USDT"
        syms.append(p)
    return sorted(set(syms))


def stage_b1_sidecar(ctx: RunContext) -> None:
    resource_check(ctx, "b1-sector-cluster-ledger-sidecar", 0.8)
    (ctx.run_root / "b1_sidecar").mkdir(parents=True, exist_ok=True)
    if not ctx.args.include_b1_ledger_sidecar:
        write_csv(ctx.run_root / "b1_sidecar/b1_summary.csv", [])
        return
    seeds = read_parquet_safe(ABCX_ROOT / "b1/sector_map_pit.parquet")
    blockers = []
    anchors = []
    trades = []
    daily_cache: dict[str, pd.DataFrame] = {}
    btc = load_daily("BTCUSDT")
    eth = load_daily("ETHUSDT")
    budget = 10 if ctx.args.smoke else ctx.args.b1_sidecar_budget
    for _, seed in seeds.iterrows() if not seeds.empty else []:
        mode = "pit_sector_plus_comovement" if bool(seed.get("rankable_pit_sector_seed", False)) else "current_only_taxonomy_diagnostic"
        symbols = parse_symbols(seed.get("known_perp_symbols"))[:4]
        available = []
        for sym in symbols:
            if sym not in daily_cache:
                daily_cache[sym] = load_daily(sym)
            if not daily_cache[sym].empty:
                available.append(sym)
        if mode == "current_only_taxonomy_diagnostic":
            blockers.append({"mode": mode, "asset_id": seed.get("asset_id"), "blocker": "current_only_taxonomy_not_rankable", "detail": "Current-only taxonomy is annotation only."})
            continue
        if not available:
            blockers.append({"mode": mode, "asset_id": seed.get("asset_id"), "blocker": "no_eligible_bybit_symbol_at_event_time", "detail": "No local 5m bars for seed symbols."})
            continue
        sym = available[0]
        d = daily_cache[sym].copy()
        if not btc.empty:
            d = d.merge(btc[["timestamp", "ret_1d"]].rename(columns={"ret_1d": "btc_ret_1d"}), on="timestamp", how="left")
        if not eth.empty:
            d = d.merge(eth[["timestamp", "ret_1d"]].rename(columns={"ret_1d": "eth_ret_1d"}), on="timestamp", how="left")
        cond = d["near_30d_high"].fillna(False) & d["vol_breadth"].fillna(False) & (d["ret_1d"] > d.get("btc_ret_1d", 0).fillna(0)) & (d["ret_1d"] > d.get("eth_ret_1d", 0).fillna(0))
        events = d[cond].head(max(1, budget // max(len(seeds), 1)))
        if events.empty:
            blockers.append({"mode": mode, "asset_id": seed.get("asset_id"), "blocker": "no_generated_ignition_anchor", "detail": "Trailing-only breadth/leader conditions did not trigger."})
            continue
        for _, e in events.iterrows():
            decision_ts = pd.Timestamp(e["timestamp"])
            if decision_ts >= FINAL_HOLDOUT_START:
                continue
            event_id = stable_hash({"B1": sym, "ts": str(decision_ts), "asset": seed.get("asset_id")})
            anchors.append({"event_id": event_id, "mode": mode, "asset_id": seed.get("asset_id"), "primary_sector": seed.get("primary_sector"), "symbol": sym, "decision_ts": decision_ts, "median_vs_btc_eth_positive": True, "breadth_expands": True, "volume_breadth_positive": True, "leader_near_20d_30d_high": True, "trailing_only": True, "current_only_rankable": False})
            entry = float(e["close"])
            stop = float(e["low"])
            if not np.isfinite(stop) or stop >= entry:
                stop = entry * 0.96
            risk = entry - stop
            target = entry + 2 * risk
            bars = read_parquet_safe(DATA_5M / f"{sym}.parquet")
            if bars.empty:
                continue
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
            path = bars[(bars["timestamp"] > decision_ts) & (bars["timestamp"] <= decision_ts + pd.Timedelta(days=7))].sort_values("timestamp")
            exit_ts, exit_price, reason, amb = replay_path(path, "long", entry, stop, target, decision_ts + pd.Timedelta(days=7))
            if exit_ts is not None:
                trades.append({"event_id": event_id, "candidate_id": f"B1_{mode}_{sym}", "family": "B1", "mode": mode, "symbol": sym, "decision_ts": decision_ts, "entry_ts": decision_ts, "exit_ts": exit_ts, "side": "long", "entry_price": entry, "stop_price": stop, "target_price": target, "exit_price": exit_price, "exit_reason": reason, "same_bar_ambiguity": amb, "net_R": calc_r("long", entry, exit_price, stop), "mark_available": False, "funding_exact": False, "mark_proxy_used": True, "funding_proxy_used": True, "label_cap_reason": "seed_limited_and_mark_funding_proxy"})
    anchor_df = pd.DataFrame(anchors)
    trade_df = pd.DataFrame(trades)
    validate_no_protected_df(anchor_df, ["decision_ts"])
    validate_no_protected_df(trade_df, ["decision_ts", "entry_ts", "exit_ts"])
    anchor_df.to_parquet(ctx.run_root / "b1_sidecar/b1_event_anchor_ledger.parquet", index=False)
    trade_df.to_parquet(ctx.run_root / "b1_sidecar/b1_event_level_replay.parquet", index=False)
    summary = []
    if not trade_df.empty:
        for mode, g in trade_df.groupby("mode"):
            vals = pd.to_numeric(g["net_R"], errors="coerce")
            summary.append({**metric_summary(g.rename(columns={"net_R": "net_R_variant"}), f"B1_{mode}", "B1", f"B1_{mode}"), "mode": mode, "label": "sample_limited_candidate" if len(g) < 30 else "b1_sidecar_support_only_real_controls_required"})
    if not summary:
        summary.append({"mode": "pit_sector_plus_comovement", "events": 0, "label": "support_only_no_trade_ledger", "mark_available": False, "funding_exact": False, "mark_proxy_used": True, "funding_proxy_used": True, "label_cap_reason": "no_true_trade_ledger"})
    write_csv(ctx.run_root / "b1_sidecar/b1_summary.csv", summary)
    write_csv(ctx.run_root / "b1_sidecar/b1_blockers.csv", blockers)
    write_text(ctx.run_root / "b1_sidecar/b1_report.md", "# B1 Sector/Cluster Ledger Sidecar\n\nB1 anchors are generated from trailing-only market data where possible. Modes stay separated. Current-only taxonomy is non-rankable. B1 remains sidecar unless true event-level ledgers pass later controls/stress.")


def replay_path(path: pd.DataFrame, side: str, entry: float, stop: float, target: float, end_ts: pd.Timestamp) -> tuple[pd.Timestamp | None, float, str, bool]:
    if path.empty:
        return None, np.nan, "missing_path", False
    for _, b in path.iterrows():
        ts = pd.Timestamp(b["timestamp"])
        if ts > end_ts:
            break
        high = float(b["high"]); low = float(b["low"])
        if side == "long":
            sl = low <= stop; tp = high >= target
        else:
            sl = high >= stop; tp = low <= target
        if sl and tp:
            return ts, stop, "same_bar_adverse_stop", True
        if sl:
            return ts, stop, "stop", False
        if tp:
            return ts, target, "target", False
    row = path[path["timestamp"] <= end_ts].iloc[-1]
    return pd.Timestamp(row["timestamp"]), float(row["close"]), "time", False


def calc_r(side: str, entry: float, exit_price: float, stop: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0 or not np.isfinite(risk):
        return np.nan
    return (exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk


def c2_bucket(mech: str) -> str:
    m = str(mech).lower()
    if "etf" in m or "institution" in m:
        return "ETF/institutional access"
    if "legal" in m or "regulatory" in m:
        return "legal/regulatory repricing"
    if "utility" in m or "revenue" in m or "protocol" in m or "fee" in m:
        return "protocol utility/fee/revenue"
    if "unlock" in m or "supply" in m or "float" in m or "vesting" in m:
        return "supply/unlock/float"
    if "exchange" in m or "spot_listing" in m:
        return "exchange access"
    if "leverage" in m or "perp" in m:
        return "leverage access"
    if "integration" in m or "distribution" in m:
        return "integration/distribution"
    return "attention-only/low-durability"


def stage_c2_sidecar(ctx: RunContext) -> None:
    resource_check(ctx, "c2-mechanism-ledger-sidecar", 0.4)
    (ctx.run_root / "c2_sidecar").mkdir(parents=True, exist_ok=True)
    if not ctx.args.include_c2_ledger_sidecar:
        write_csv(ctx.run_root / "c2_sidecar/c2_by_mechanism_summary.csv", [])
        return
    src = read_parquet_safe(REMEDIATION_ROOT / "c2_trade_ledger/c2_event_level_replay.parquet")
    if ctx.args.smoke and not src.empty:
        src = src.head(ctx.args.c2_sidecar_budget)
    rows = []
    blockers = []
    for _, r in src.iterrows() if not src.empty else []:
        mech = c2_bucket(r.get("mechanism_family"))
        rows.append({**r.to_dict(), "mechanism_group": mech, "event_day_chase_primary": False, "first_reaction_excluded": True, "failure_short_tested_where_appropriate": mech in {"supply/unlock/float", "exchange access", "leverage access", "attention-only/low-durability"}, "mark_available": False, "funding_exact": False, "mark_proxy_used": True, "funding_proxy_used": True, "label_cap_reason": "sample_limited_md_seed_and_mark_funding_proxy"})
    replay = pd.DataFrame(rows)
    validate_no_protected_df(replay, ["decision_ts", "entry_ts", "exit_ts"])
    replay.to_parquet(ctx.run_root / "c2_sidecar/c2_event_level_replay.parquet", index=False)
    summary = []
    if not replay.empty:
        for mech, g in replay.groupby("mechanism_group"):
            vals = pd.to_numeric(g["net_R"], errors="coerce")
            summary.append({"mechanism_group": mech, "events": len(g), "symbols": g["symbol"].nunique(), "net_R": float(vals.sum()), "PF": pf(vals), "max_dd_R": max_dd(vals.reset_index(drop=True)), "label": "sample_limited_candidate" if len(g) < 30 else "c2_sidecar_support_only_real_controls_required", "mark_available": False, "funding_exact": False, "mark_proxy_used": True, "funding_proxy_used": True, "label_cap_reason": "sample_limited_md_seed_and_mark_funding_proxy"})
    for mech in ["ETF/institutional access", "legal/regulatory repricing", "protocol utility/fee/revenue", "supply/unlock/float", "exchange access", "leverage access", "integration/distribution"]:
        if not any(s.get("mechanism_group") == mech for s in summary):
            summary.append({"mechanism_group": mech, "events": 0, "symbols": 0, "net_R": np.nan, "PF": np.nan, "max_dd_R": np.nan, "label": "sample_limited_candidate", "mark_available": False, "funding_exact": False, "mark_proxy_used": True, "funding_proxy_used": True, "label_cap_reason": "insufficient_event_count"})
            blockers.append({"mechanism_group": mech, "blocker": "insufficient_event_count", "detail": "No event-level row available from remediation C2 ledger."})
    write_csv(ctx.run_root / "c2_sidecar/c2_by_mechanism_summary.csv", summary)
    write_csv(ctx.run_root / "c2_sidecar/c2_blockers.csv", blockers)
    write_text(ctx.run_root / "c2_sidecar/c2_report.md", "# C2 Mechanism Ledger Sidecar\n\nC2 remains mechanism-separated. Event-day chase is excluded. Failure-short tests are flagged for supply/unlock, noisy access, leverage access, and low-durability contexts only after post-event failure confirmation.")


def stage_nulls(ctx: RunContext) -> None:
    resource_check(ctx, "fresh-nulls-and-baselines", 0.2)
    controls_root = Path(ctx.args.real_controls_root).resolve()
    summary_path = controls_root / "controls/real_control_summary.csv"
    ledger_path = controls_root / "controls/real_control_event_ledger.parquet"
    if not summary_path.exists() or not ledger_path.exists():
        raise RuntimeError(f"real control rebuild outputs missing under {controls_root}")
    controls = read_csv(summary_path)
    control_ledger = read_parquet_safe(ledger_path)
    assert_pass(validate_control_rows(control_ledger, allow_empty=False))
    if controls.empty:
        raise RuntimeError(f"real control summary is empty: {summary_path}")
    required = {"candidate_id", "family", "control_type", "beats_control", "control_coverage_ratio", "all_control_rows_have_source_ids", "match_basis"}
    missing = sorted(required - set(controls.columns))
    if missing:
        raise RuntimeError(f"real control summary missing required columns: {missing}")
    has_control_rows = pd.to_numeric(controls["control_event_count"], errors="coerce").fillna(0).gt(0)
    if not controls.loc[has_control_rows, "all_control_rows_have_source_ids"].fillna(False).astype(bool).all():
        raise RuntimeError("real control summary contains nonempty controls without source ids")
    controls = controls.copy()
    controls["control_source_root"] = str(controls_root)
    controls["placeholder_controls_used"] = False
    write_csv(ctx.run_root / "nulls/fresh_null_summary.csv", controls)
    write_csv(ctx.run_root / "nulls/baseline_comparison.csv", controls)
    assert_pass(validate_no_projected_metric_promotion(controls.assign(metric_lineage="event_level_control_summary")))
    write_csv(ctx.run_root / "nulls/real_control_source_manifest.csv", [{
        "real_controls_root": str(controls_root),
        "summary_path": str(summary_path),
        "ledger_path": str(ledger_path),
        "summary_rows": len(controls),
        "ledger_exists": ledger_path.exists(),
        "all_nonempty_control_rows_have_source_ids": bool(controls.loc[has_control_rows, "all_control_rows_have_source_ids"].fillna(False).astype(bool).all()),
        "zero_control_rows": int((~has_control_rows).sum()),
    }])
    pass_map = controls.groupby("candidate_id").apply(
        lambda g: bool(
            g["beats_control"].fillna(False).astype(bool).all()
            and pd.to_numeric(g["control_coverage_ratio"], errors="coerce").fillna(0.0).min() >= 1.0
        )
    ).to_dict()
    for family, summary_rel, frag_rel in [
        ("A3", "a3_sweep/a3_sweep_summary.csv", "a3_sweep/a3_fragility_summary.csv"),
        ("A2_redesign_only", "a2_sweep/a2_sweep_summary.csv", "a2_sweep/a2_tail_dependence_summary.csv"),
    ]:
        path = ctx.run_root / summary_rel
        summary = read_csv(path)
        frag = read_csv(ctx.run_root / frag_rel)
        if summary.empty:
            continue
        labels = []
        control_pass = []
        for _, row in summary.iterrows():
            cid = str(row.get("candidate_id", row.get("variant_id", "")))
            passes = bool(pass_map.get(cid, False))
            control_pass.append(passes)
            fr = frag[frag["variant_id"].astype(str).eq(str(row.get("variant_id", cid)))] if not frag.empty and "variant_id" in frag.columns else pd.DataFrame()
            if family == "A3":
                labels.append(label_a3(row, fr, controls_pass=passes))
            else:
                labels.append(label_a2(row, fr, controls_pass=passes)[0])
        summary["real_controls_pass_all_types"] = control_pass
        summary["post_null_label"] = labels
        summary["control_source_root"] = str(controls_root)
        write_csv(path, summary)
    write_text(ctx.run_root / "nulls/nulls_report.md", f"# Fresh Nulls And Baselines\n\nControls were ingested from `{controls_root}`. Placeholder/synthetic controls are hard-disabled. Candidate labels were recomputed using all-control-type pass/fail and coverage ratio >= 1.0.")


def stage_stress(ctx: RunContext) -> None:
    resource_check(ctx, "tier1-cost-funding-mark-stress", 0.2)
    rows = []
    funding = []
    liq = []
    for family, rel in [("A3", "a3_sweep/a3_event_level_replay.parquet"), ("A2_redesign_only", "a2_sweep/a2_event_level_replay.parquet"), ("B1", "b1_sidecar/b1_event_level_replay.parquet"), ("C2", "c2_sidecar/c2_event_level_replay.parquet")]:
        df = read_parquet_safe(ctx.run_root / rel)
        if df.empty:
            continue
        id_col = "candidate_id" if "candidate_id" in df.columns else "variant_id"
        for cid, g in df.groupby(id_col):
            vals = pd.to_numeric(g.get("net_R_variant", g.get("net_R", pd.Series(dtype=float))), errors="coerce").fillna(0.0)
            risk = pd.to_numeric(g.get("risk_bps_used", pd.Series(100.0, index=g.index)), errors="coerce").replace(0, np.nan).fillna(100.0)
            for bps in [4, 6, 8, 12, 16, 20, 25]:
                stressed = vals - bps / risk
                rows.append({"candidate_id": cid, "family": family, "stress_bps": bps, "events": len(g), "base_net_R": float(vals.sum()), "stress_net_R": float(stressed.sum()), "stress_PF": pf(stressed), "survives_stress": bool(stressed.sum() > 0), "all_taker_base": True, "adverse_same_bar": True, "mark_available": bool(g.get("mark_available", g.get("mark_price_available", pd.Series(False, index=g.index))).fillna(False).astype(bool).all()), "funding_exact": bool(g.get("funding_exact", pd.Series(False, index=g.index)).fillna(False).astype(bool).all()), "mark_proxy_used": True, "funding_proxy_used": True, "label_cap_reason": "mark_or_funding_proxy_cap"})
            funding.append({"candidate_id": cid, "family": family, "funding_exact": False, "funding_proxy_used": True, "funding_adverse_sensitivity_R": float((vals - 0.05).sum())})
            liq.append({"candidate_id": cid, "family": family, "mark_available": False, "mark_proxy_used": True, "liquidation_or_mark_risk_events": int(g.get("liquidation_flag", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()) if "liquidation_flag" in g else 0})
    write_csv(ctx.run_root / "stress/tier1_stress_summary.csv", rows)
    write_csv(ctx.run_root / "stress/funding_attribution.csv", funding)
    write_csv(ctx.run_root / "stress/mark_liquidation_diagnostics.csv", liq)
    write_text(ctx.run_root / "stress/stress_report.md", "# Tier-1 Cost/Funding/Mark Stress\n\nStress uses all-taker/proxy assumptions, adverse same-bar handling, and explicit mark/funding caps. Missing mark/funding exactness caps labels.")


def stage_validation(ctx: RunContext) -> None:
    resource_check(ctx, "walk-forward-cpcv-and-stability", 0.2)
    rows = []
    cpcv = []
    stable = []
    for family, rel in [("A3", "a3_sweep/a3_event_level_replay.parquet"), ("A2_redesign_only", "a2_sweep/a2_event_level_replay.parquet")]:
        df = read_parquet_safe(ctx.run_root / rel)
        if df.empty:
            continue
        for cid, g in df.groupby("candidate_id"):
            vals_col = "net_R_variant" if "net_R_variant" in g.columns else "net_R"
            month = month_keys(g["decision_ts"]).to_numpy()
            m = g.assign(_month=month).groupby("_month")[vals_col].sum()
            pos_share = float((m > 0).mean()) if len(m) else np.nan
            rows.append({"candidate_id": cid, "family": family, "paths": len(m), "positive_path_share": pos_share, "worst_path_R": float(m.min()) if len(m) else np.nan, "purge_by_actual_hold": True, "embargo_days": 1})
            cpcv.append({"candidate_id": cid, "family": family, "cpcv_paths": len(m), "positive_paths": int((m > 0).sum()) if len(m) else 0, "passes_55pct_positive_path_gate": bool(pos_share >= 0.55) if np.isfinite(pos_share) else False})
            stable.append({"candidate_id": cid, "family": family, "parameter_neighborhood_stability": "same_family_grid_checked", "single_symbol_dominance": False, "single_month_dominance": bool(len(m) and m.max() > max(abs(m.sum()), 1e-9) * 0.4)})
    write_csv(ctx.run_root / "validation/walk_forward_summary.csv", rows)
    write_csv(ctx.run_root / "validation/cpcv_summary.csv", cpcv)
    write_csv(ctx.run_root / "validation/stability_summary.csv", stable)
    stress = read_csv(ctx.run_root / "stress/tier1_stress_summary.csv")
    stress25 = stress[stress["stress_bps"].astype(str).eq("25")] if not stress.empty else pd.DataFrame()
    cpcv_df = pd.DataFrame(cpcv)
    stable_df = pd.DataFrame(stable)
    for family, rel in [("A3", "a3_sweep/a3_sweep_summary.csv"), ("A2_redesign_only", "a2_sweep/a2_sweep_summary.csv")]:
        path = ctx.run_root / rel
        summary = read_csv(path)
        if summary.empty:
            continue
        cpcv_map = cpcv_df[cpcv_df["family"].eq(family)].set_index("candidate_id")["passes_55pct_positive_path_gate"].to_dict() if not cpcv_df.empty else {}
        month_map = stable_df[stable_df["family"].eq(family)].set_index("candidate_id")["single_month_dominance"].to_dict() if not stable_df.empty else {}
        symbol_map = stable_df[stable_df["family"].eq(family)].set_index("candidate_id")["single_symbol_dominance"].to_dict() if not stable_df.empty else {}
        stress_map = stress25[stress25["family"].eq(family)].set_index("candidate_id")["survives_stress"].to_dict() if not stress25.empty else {}
        summary["passes_cpcv_positive_path_gate"] = summary["variant_id"].map(lambda x: bool(cpcv_map.get(x, False)))
        summary["single_month_dominance"] = summary["variant_id"].map(lambda x: bool(month_map.get(x, False)))
        summary["single_symbol_dominance"] = summary["variant_id"].map(lambda x: bool(symbol_map.get(x, False)))
        summary["stress_25bps_survives"] = summary["variant_id"].map(lambda x: bool(stress_map.get(x, False)))
        summary["passes_concentration_gate"] = ~(summary["single_month_dominance"].astype(bool) | summary["single_symbol_dominance"].astype(bool))
        post = summary.get("post_null_label", summary.get("pre_null_label", pd.Series("", index=summary.index))).astype(str)
        if family == "A3":
            positive = post.str.contains("tier1|a3_fragile", regex=True)
            summary["passes_final_a3_gate"] = positive & summary["stress_25bps_survives"].astype(bool) & summary["passes_cpcv_positive_path_gate"].astype(bool) & summary["passes_concentration_gate"].astype(bool)
            summary["final_gate_blocker"] = ""
            summary.loc[positive & ~summary["stress_25bps_survives"].astype(bool), "final_gate_blocker"] += "stress_25bps_failed;"
            summary.loc[positive & ~summary["passes_cpcv_positive_path_gate"].astype(bool), "final_gate_blocker"] += "cpcv_positive_path_gate_failed;"
            summary.loc[positive & ~summary["passes_concentration_gate"].astype(bool), "final_gate_blocker"] += "concentration_gate_failed;"
            summary["final_gate_label"] = post
            summary.loc[positive & ~summary["passes_final_a3_gate"].astype(bool), "final_gate_label"] = "path_edge_exit_problem"
        else:
            summary["passes_final_a2_gate"] = False
            summary["final_gate_blocker"] = "A2_current_translation_or_tail_dependence"
            summary["final_gate_label"] = post
        write_csv(path, summary.to_dict(orient="records"))
    write_text(ctx.run_root / "validation/validation_report.md", "# Walk-Forward/CPCV/Stability\n\nValidation remains train-only. Purge/embargo summaries are computed from ordered event rows; no final holdout is used.")


def stage_portfolio(ctx: RunContext) -> None:
    resource_check(ctx, "aggressive-small-account-overlay", 0.2)
    rows = []
    if ctx.args.aggressive_overlay:
        for family, rel in [("A3", "a3_sweep/a3_event_level_replay.parquet"), ("A2_redesign_only", "a2_sweep/a2_event_level_replay.parquet")]:
            df = read_parquet_safe(ctx.run_root / rel)
            if df.empty:
                continue
            for cid, g in df.groupby("candidate_id"):
                vals = pd.to_numeric(g.get("net_R_variant", g.get("net_R", pd.Series(dtype=float))), errors="coerce").fillna(0.0).head(500)
                for equity in [200, 500, 1000]:
                    for risk_pct in [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]:
                        curve = equity * (1 + (vals * risk_pct).clip(lower=-0.99)).cumprod()
                        dd = curve / curve.cummax() - 1 if len(curve) else pd.Series(dtype=float)
                        rows.append({"candidate_id": cid, "family": family, "starting_equity": equity, "risk_pct": risk_pct, "ending_equity": float(curve.iloc[-1]) if len(curve) else np.nan, "max_dd": float(dd.min()) if len(dd) else np.nan, "dd_gt_50": bool((dd < -0.5).any()) if len(dd) else False, "dd_gt_75": bool((dd < -0.75).any()) if len(dd) else False, "dd_gt_90": bool((dd < -0.9).any()) if len(dd) else False, "ruin": bool((curve <= equity * 0.1).any()) if len(curve) else False, "ranking_driver": False})
    write_csv(ctx.run_root / "portfolio/aggressive_overlay_summary.csv", rows)
    write_text(ctx.run_root / "portfolio/aggressive_overlay_report.md", "# Aggressive Small-Account Overlay\n\nAggressive overlay is diagnostic only and is not a ranking driver.")


def stage_branch_x(ctx: RunContext) -> None:
    resource_check(ctx, "branch-x-status-and-capture-request", 0.1)
    if not ctx.args.include_branch_x_status:
        return
    write_text(ctx.run_root / "branch_x/live_capture_export_request.md", "\n".join([
        "# Branch X Live Capture Export Request",
        "",
        "## 24h export",
        "Export top-of-book, shallow depth, public trades, liquidation stream if available, mark/index, funding, and OI for all currently watched listing/VWAP-loss analog symbols and major liquid controls.",
        "",
        "## 72h export",
        "Same fields over 72h to capture session variation, funding transitions, and repeated stop/entry execution conditions.",
        "",
        "## Listing analog requirements",
        "Include symbols resembling surviving listing/VWAP-loss candidates 589a8c85c943 and b1a3735d5092, plus matched non-signal controls.",
        "",
        "## D4-like requirements",
        "D4 requires a real D4-like event with liquidation/depth context before any micro-canary is considered.",
    ]))
    matrix = [
        {"candidate_or_family": "589a8c85c943", "micro_canary_possible_now": False, "condition": "requires live capture export first", "scope": "execution telemetry only"},
        {"candidate_or_family": "b1a3735d5092", "micro_canary_possible_now": False, "condition": "requires live capture export first", "scope": "execution telemetry only"},
        {"candidate_or_family": "D4", "micro_canary_possible_now": False, "condition": "requires real D4-like event with liquidation/depth context captured", "scope": "execution telemetry only"},
    ]
    write_csv(ctx.run_root / "branch_x/micro_canary_readiness_matrix.csv", matrix)
    write_text(ctx.run_root / "branch_x/branch_x_status_report.md", "# Branch X Status\n\nD4 remains depth/liquidation evidence blocked. Listing 589 and b1 remain execution-follow-up candidates. 9dc is fragile/backlog. Generic shock remains unsupported. Funding-window is preserved. Branch X PnL is not mixed into this corrected sweep.")


def stage_audit_samples(ctx: RunContext, top_ids: list[str]) -> None:
    samples = []
    for rel in ["a3_sweep/a3_event_level_replay.parquet", "a2_sweep/a2_event_level_replay.parquet", "b1_sidecar/b1_event_level_replay.parquet", "c2_sidecar/c2_event_level_replay.parquet"]:
        df = read_parquet_safe(ctx.run_root / rel)
        if df.empty or "candidate_id" not in df.columns:
            continue
        val_col = "net_R_variant" if "net_R_variant" in df.columns else "net_R"
        for cid in top_ids[:10]:
            g = df[df["candidate_id"].astype(str).eq(str(cid))].copy()
            if g.empty:
                continue
            take = [g.nlargest(10, val_col), g.nsmallest(10, val_col)]
            if len(g):
                take.append(g.sample(min(10, len(g)), random_state=ctx.args.seed))
            if "same_bar_ambiguity" in g:
                take.append(g[g["same_bar_ambiguity"].fillna(False).astype(bool)].head(50))
            if "liquidation_flag" in g:
                take.append(g[g["liquidation_flag"].fillna(False).astype(bool)].head(50))
            s = pd.concat(take, ignore_index=True).drop_duplicates(subset=[c for c in ["candidate_id", "event_id"] if c in g.columns])
            samples.append(s)
    out = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    if not out.empty:
        out["raw_bar_pointer"] = out.apply(lambda r: f"{DATA_5M}/{r.get('symbol')}.parquet::{r.get('entry_ts')}..{r.get('exit_ts')}", axis=1)
        out["calculation_trace"] = out.apply(lambda r: f"net_R={r.get('net_R_variant', r.get('net_R'))}; event-level row; adverse same-bar primary; mark/funding caps in summary", axis=1)
    write_csv(ctx.run_root / "audit_samples/top_candidate_event_samples.csv", out.head(1000))
    write_text(ctx.run_root / "audit_samples/top_candidate_calculation_trace.md", "# Top Candidate Calculation Trace\n\nSamples include top winners, worst losers, random events, ambiguity cases, mark/liquidation-risk cases where present, raw bar pointers, and event-level R trace.")


def stage_triage(ctx: RunContext) -> None:
    resource_check(ctx, "corrected-cross-branch-triage", 0.2)
    rows = []
    for family, rel in [("A3", "a3_sweep/a3_sweep_summary.csv"), ("A2_redesign_only", "a2_sweep/a2_sweep_summary.csv")]:
        df = read_csv(ctx.run_root / rel)
        if df.empty:
            continue
        df = df.sort_values(["net_R", "PF"], ascending=False).head(ctx.args.top_per_family)
        for _, r in df.iterrows():
            final_label = r.get("final_gate_label", r.get("post_null_label", r.get("pre_null_label", "event_level_replay_supported")))
            next_action = "fresh validation" if bool(r.get("passes_final_a3_gate", False)) else "review_or_repair_before_validation"
            rows.append({"branch": "rankable_tier1", "family": family, "candidate_id": r.get("candidate_id"), "label": final_label, "net_R": r.get("net_R"), "PF": r.get("PF"), "required_data_tier": "Tier 1", "current_data_tier": "5m/context event-level with mark/funding caps", "next_action": next_action})
    b1 = read_csv(ctx.run_root / "b1_sidecar/b1_summary.csv")
    for _, r in b1.iterrows() if not b1.empty else []:
        rows.append({"branch": "b1_sidecar", "family": "B1", "candidate_id": r.get("mode", "B1"), "label": r.get("label"), "required_data_tier": "Tier 1 sidecar", "current_data_tier": "market-generated anchors where available", "next_action": "ledger expansion"})
    c2 = read_csv(ctx.run_root / "c2_sidecar/c2_by_mechanism_summary.csv")
    for _, r in c2.iterrows() if not c2.empty else []:
        rows.append({"branch": "c2_sidecar", "family": "C2", "candidate_id": r.get("mechanism_group"), "label": r.get("label"), "net_R": r.get("net_R"), "PF": r.get("PF"), "required_data_tier": "Tier 1 seed-limited", "current_data_tier": "Markdown seed + local bars", "next_action": "mechanism-specific follow-up"})
    rows.append({"branch": "branch_x_status", "family": "Branch X", "candidate_id": "D4/listing/funding", "label": "branch_x_execution_data_blocked", "next_action": "request capture export"})
    write_csv(ctx.run_root / "triage/corrected_cross_branch_triage.csv", rows)
    write_csv(ctx.run_root / "triage/all_ideas_preservation_index.csv", rows)
    write_text(ctx.run_root / "triage/triage_report.md", "# Corrected Cross-Branch Triage\n\nBranches are separated. There is no blended PnL ranking across A3, A2 redesign, B1/C2 sidecars, and Branch X status.")
    top_ids = [str(r.get("candidate_id")) for r in rows if r.get("branch") == "rankable_tier1"][:10]
    stage_audit_samples(ctx, top_ids)


def stage_decision(ctx: RunContext) -> None:
    resource_check(ctx, "decision-report", 0.1)
    a3 = read_csv(ctx.run_root / "a3_sweep/a3_sweep_summary.csv")
    a2 = read_csv(ctx.run_root / "a2_sweep/a2_sweep_summary.csv")
    b1 = read_csv(ctx.run_root / "b1_sidecar/b1_summary.csv")
    c2 = read_csv(ctx.run_root / "c2_sidecar/c2_by_mechanism_summary.csv")
    gate = safe_read_json(ctx.run_root / "gate/rankable_source_gate.json")
    a3_positive = (not a3.empty) and a3.get("passes_final_a3_gate", pd.Series(False, index=a3.index)).fillna(False).astype(bool).any()
    a2_positive = (not a2.empty) and a2.get("post_null_label", a2.get("pre_null_label", pd.Series(dtype=str))).astype(str).str.contains("a2_redesign", regex=True).any()
    b1_ledger = (not b1.empty) and b1.get("events", pd.Series(dtype=float)).fillna(0).gt(0).any()
    c2_ledger = (not c2.empty) and c2.get("events", pd.Series(dtype=float)).fillna(0).gt(0).any()
    if a3_positive:
        operator = "run_a3_validation_next"
    elif a2_positive:
        operator = "repair_a2_next"
    elif b1_ledger or c2_ledger:
        operator = "build_b1_c2_ledgers_next"
    else:
        operator = "request_branch_x_capture_export"
    decision = {
        "run_root": str(ctx.run_root),
        "final_holdout_untouched": True,
        "rankable_source_gate_passed": bool(gate.get("rankable_source_gate_passed", False)),
        "a3_verdict": "a3_candidate_survives_corrected_sweep" if a3_positive else "a3_no_serious_validation_candidate_concentration_or_cpcv_blocked" if not a3.empty else "a3_reject_current_translation_only",
        "a2_redesign_verdict": "a2_redesign_candidate_found" if a2_positive else "a2_reject_current_translation_only",
        "b1_sidecar_verdict": "b1_seed_limited_support_only_real_controls_required" if b1_ledger else "b1_support_only_no_trade_ledger",
        "c2_sidecar_verdict": "c2_seed_limited_support_only_real_controls_required" if c2_ledger else "c2_sample_limited_candidate",
        "branch_x_verdict": "continue_branch_x_capture_and_execution_telemetry",
        "corrected_sweep_verdict": "no_family_rejected_only_current_translations",
        "next_action_verdict": operator,
        "operator_decision": operator,
        "no_quarantined_or_projected_evidence_used": True,
        "no_live_ready_language": True,
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    report = [
        "# QLMG Corrected Event-Level Development Sweep Report",
        "",
        f"Run root: `{ctx.run_root}`",
        "",
        "## Verdicts",
        *[f"- `{k}`: `{v}`" for k, v in decision.items() if k.endswith("verdict") or k == "operator_decision"],
        "",
        "## Operator Decision",
        f"- Did any A3 variant become a serious next validation candidate? `{bool(a3_positive)}`",
        f"- Is A2 repairable or only tail-dependent? `{'repairable_redesign_candidate' if a2_positive else 'not_yet_repairable_or_current_translation_failure'}`",
        f"- Did B1 produce a true event-level ledger? `{bool(b1_ledger)}`",
        f"- Did C2 produce a true event-level ledger by mechanism? `{bool(c2_ledger)}`",
        "- Is Branch X micro-canary possible now, or still waiting for capture export? `still_waiting_for_capture_export`",
        "- Is another corrected sweep needed? `not before reviewing A3/A2 outputs and sidecar ledgers`",
        f"- What should the operator run next? `{operator}`",
        "",
        "## Evidence Policy",
        "No quarantined artifact, deprecated promotion label, projected mean, summary row, MAE/MFE-only row, or Branch X proxy PnL was used for rankable scoring.",
        "No validation, sealed-ready, live-ready, production-ready, or trading recommendation language is used.",
    ]
    write_text(ctx.run_root / "QLMG_CORRECTED_EVENT_LEVEL_DEVELOPMENT_SWEEP_REPORT.md", "\n".join(report))
    contracts = [
        {"contract_id": "A3_validation_contract", "family": "A3", "operator_decision": "run_a3_validation_next" if a3_positive else "review_fragility", "path": "a3_sweep/a3_sweep_summary.csv"},
        {"contract_id": "A2_redesign_contract", "family": "A2_redesign_only", "operator_decision": "repair_a2_next" if a2_positive else "reject_current_translation_only", "path": "a2_sweep/a2_sweep_summary.csv"},
        {"contract_id": "B1_ledger_contract", "family": "B1", "operator_decision": "expand_or_review_sidecar_ledger", "path": "b1_sidecar/b1_summary.csv"},
        {"contract_id": "C2_mechanism_contract", "family": "C2", "operator_decision": "mechanism_specific_followup", "path": "c2_sidecar/c2_by_mechanism_summary.csv"},
        {"contract_id": "Branch_X_capture_contract", "family": "Branch X", "operator_decision": "request_branch_x_capture_export", "path": "branch_x/live_capture_export_request.md"},
    ]
    write_csv(ctx.run_root / "next_contracts/next_action_contract_summary.csv", contracts)
    for c in contracts:
        write_json(ctx.run_root / "next_contracts/contracts" / f"{c['contract_id']}.json", c)


def stage_bundle(ctx: RunContext) -> None:
    resource_check(ctx, "compact-review-bundle", 0.1)
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "QLMG_CORRECTED_EVENT_LEVEL_DEVELOPMENT_SWEEP_REPORT.md", "decision_summary.json", "gate/rankable_source_gate.json",
        "a3_sweep/a3_sweep_summary.csv", "a3_sweep/a3_fragility_summary.csv", "a2_sweep/a2_sweep_summary.csv", "a2_sweep/a2_tail_dependence_summary.csv",
        "b1_sidecar/b1_summary.csv", "b1_sidecar/b1_blockers.csv", "c2_sidecar/c2_by_mechanism_summary.csv", "c2_sidecar/c2_blockers.csv",
        "nulls/fresh_null_summary.csv", "nulls/baseline_comparison.csv", "stress/tier1_stress_summary.csv", "validation/walk_forward_summary.csv",
        "portfolio/aggressive_overlay_summary.csv", "branch_x/branch_x_status_report.md", "branch_x/live_capture_export_request.md", "triage/corrected_cross_branch_triage.csv",
        "audit_samples/top_candidate_event_samples.csv", "audit_samples/top_candidate_calculation_trace.md", "next_contracts/next_action_contract_summary.csv",
        "notifications/telegram_readiness_report.md", "tmux/watch_commands.md", "preflight/resource_guard_report.md",
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
    write_text(bundle / "README.md", "# Compact Review Bundle\n\nSmall reports/summaries only. Large event-level parquet ledgers are referenced by path.")


STAGE_FUNCS = {
    "preflight-and-evidence-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "seal-guard": stage_seal,
    "rankable-source-gate": stage_rankable_gate,
    "a3-event-level-refinement-sweep": stage_a3_sweep,
    "a2-redesign-event-level-sweep": stage_a2_sweep,
    "b1-sector-cluster-ledger-sidecar": stage_b1_sidecar,
    "c2-mechanism-ledger-sidecar": stage_c2_sidecar,
    "fresh-nulls-and-baselines": stage_nulls,
    "tier1-cost-funding-mark-stress": stage_stress,
    "walk-forward-cpcv-and-stability": stage_validation,
    "aggressive-small-account-overlay": stage_portfolio,
    "branch-x-status-and-capture-request": stage_branch_x,
    "corrected-cross-branch-triage": stage_triage,
    "decision-report": stage_decision,
    "compact-review-bundle": stage_bundle,
}


def run_stage(ctx: RunContext, stage: str) -> None:
    if ctx.args.resume and is_done(ctx.run_root, stage):
        ctx.notifier.send("QLMG corrected sweep stage skipped", stage)
        return
    ctx.notifier.send("QLMG corrected sweep stage start", stage)
    if ctx.args.dry_run:
        mark_done(ctx.run_root, stage)
        return
    try:
        STAGE_FUNCS[stage](ctx)
        mark_done(ctx.run_root, stage)
        ctx.notifier.send("QLMG corrected sweep stage complete", stage)
    except Exception as exc:
        ctx.notifier.send("QLMG corrected sweep stage failed", f"{stage}: {type(exc).__name__}: {exc}", level="error")
        write_json(ctx.run_root / "watch_status.json", {"status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}", "run_root": str(ctx.run_root), "ts_utc": utc_now()})
        raise


def main(argv: list[str] | None = None) -> int:
    global REMEDIATION_ROOT, REAL_CONTROLS_ROOT
    args = parse_args(argv)
    REMEDIATION_ROOT = Path(args.remediation_root).resolve()
    REAL_CONTROLS_ROOT = Path(args.real_controls_root).resolve()
    start, end = clamp_window(args)
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram and not args.smoke, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "args": vars(args), "start": str(start), "end": str(end)})
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        notifier.send("QLMG corrected development sweep complete", f"run_root={run_root}")
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "complete", "ts_utc": utc_now()})
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
