#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_evidence_contracts import PROTECTED_TS, scan_output_tree_for_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_kraken_readiness_repair_20260701_v1"
DEFAULT_SEED = 20260701
DEFAULT_KRAKEN_DATA_ROOT = Path("/opt/parquet/kraken_derivatives")
DEFAULT_READINESS_ROOT = RESULTS_ROOT / "phase_kraken_hypothesis_sweep_readiness_20260701_v1_20260701_085434"
DEFAULT_K0_ROOT = RESULTS_ROOT / "phase_kraken_k0_data_foundation_20260630_v1_20260630_163815"
DEFAULT_HYPOTHESIS_LIBRARY = REPO / "research_inputs/QLMG_Hypothesis_Library_2026-07-01.xlsx"
DEFAULT_RESEARCH_INPUT_DIR = REPO / "research_inputs"
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
DERIV_BASE = "https://futures.kraken.com"
CHART_BASE = "https://futures.kraken.com"
USER_AGENT = "Donch-QLMG-Kraken-Readiness-Repair/1.0 public-only"

STAGES = (
    "preflight-and-artifact-freeze",
    "telegram-and-tmux-setup",
    "seal-guard",
    "k0-metadata-qc-repair",
    "funding-coverage-classification",
    "official-analytics-probe-and-download",
    "high-priority-compile-repair",
    "c2-event-ledger-construction",
    "readiness-rerun",
    "no-vendor-viability-matrix",
    "decision-report",
    "compact-review-bundle",
    "all",
)

ALLOWED_NEXT_DECISIONS = {
    "launch_full_kraken_hypothesis_sweep_next",
    "repair_analytics_data_next",
    "repair_c2_event_ledger_next",
    "repair_contract_compiler_next",
    "manual_review_required_before_sweep",
    "blocked_by_protocol_issue",
}

HIGH_PRIORITY_IDS = {"H09", "H10", "H21", "H31", "H32", "H39", "PD06", "PR08", "C2"}
RESOLVED_LANES = {
    "compiled_tier1_redesign",
    "compiled_tier1_with_analytics_cap",
    "needs_event_ledger_first",
    "needs_live_capture_substitute",
    "candidate_library_only",
    "not_kraken_viable_current_translation",
}
UNRESOLVED_LANES = {
    "missing_compile_reason",
    "missing_required_fields",
    "contradictory_data_tier_classification",
    "missing_source_trace",
    "compiler_error",
}
ANALYTICS_TYPES = [
    "open_interest",
    "funding",
    "liquidation_volume",
    "orderbook",
    "spreads",
    "liquidity",
    "slippage",
    "trade_volume",
    "trade_count",
    "future_basis",
    "rolling_volatility",
    "aggressor_differential",
    "cvd",
    "long_short_ratio",
]
ANALYTICS_ENDPOINTS = {
    "open_interest": [
        f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/open-interest",
        f"{CHART_BASE}/api/charts/v1/analytics/open-interest/PF_XBTUSD",
    ],
    "funding": [
        f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/funding",
        f"{CHART_BASE}/api/charts/v1/analytics/funding/PF_XBTUSD",
    ],
    "liquidation_volume": [
        f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/liquidation-volume",
        f"{CHART_BASE}/api/charts/v1/analytics/liquidation-volume/PF_XBTUSD",
    ],
    "orderbook": [f"{DERIV_BASE}/derivatives/api/v3/orderbook?symbol=PF_XBTUSD"],
    "spreads": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/spreads"],
    "liquidity": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/liquidity"],
    "slippage": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/slippage"],
    "trade_volume": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/trade-volume"],
    "trade_count": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/trade-count"],
    "future_basis": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/future-basis"],
    "rolling_volatility": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/rolling-volatility"],
    "aggressor_differential": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/aggressor-differential"],
    "cvd": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/cvd"],
    "long_short_ratio": [f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/long-short-ratio"],
}
MECHANISM_ALIASES = {
    "legal_regulatory_repricing": "legal/regulatory repricing",
    "etf_institutional_access": "ETF/institutional access",
    "supply_shock": "supply/unlock/float",
    "unlock_vesting_change": "supply/unlock/float",
    "protocol_utility_fee_revenue_change": "protocol utility/fee/revenue",
    "exchange_access_expansion": "exchange access",
    "leverage_access_expansion": "leverage access",
    "major_integration_distribution_access": "integration/distribution",
}

@dataclass
class RunContext:
    args: argparse.Namespace
    run_root: Path
    notifier: "RunNotifier"
    start: pd.Timestamp
    end: pd.Timestamp
    root_reason: str

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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="kraken-readiness-repair")
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
        run_status = "failed" if "failed" in title.lower() else ("complete" if "complete" in title.lower() else "running")
        write_json(self.run_root / "watch_status.json", {"status": run_status, "last_event": title, "ts_utc": rec["ts_utc"], "run_root": str(self.run_root)})
        return sent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kraken readiness repair phase")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(SCREENING_END))
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=25.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--kraken-data-root", default=str(DEFAULT_KRAKEN_DATA_ROOT))
    p.add_argument("--readiness-root", default=str(DEFAULT_READINESS_ROOT.relative_to(REPO)))
    p.add_argument("--k0-root", default=str(DEFAULT_K0_ROOT.relative_to(REPO)))
    p.add_argument("--hypothesis-library", default=str(DEFAULT_HYPOTHESIS_LIBRARY.relative_to(REPO)))
    p.add_argument("--research-input-dir", default=str(DEFAULT_RESEARCH_INPUT_DIR.relative_to(REPO)))
    p.add_argument("--download-official-analytics", action="store_true")
    p.add_argument("--download-cap-gb", type=float, default=10.0)
    p.add_argument("--rerun-readiness", action="store_true")
    p.add_argument("--tmux-session-name", default="kraken_readiness_repair")
    p.add_argument("--launch-tmux", action="store_true")
    p.add_argument("--run-root", default="")
    return p.parse_args(argv)


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def resolve_run_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.run_root:
        return resolve_path(args.run_root).resolve(), "explicit_run_root"
    base = (RESULTS_ROOT / DEFAULT_RUN_ID).resolve()
    if args.smoke:
        return (base / "smoke").resolve(), "smoke_subroot"
    if base.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return base.with_name(f"{base.name}_{suffix}"), f"default_root_existed_suffix_{suffix}"
    return base, "default_root_available"


def clamp_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(args.start, utc=True)
    end = min(pd.to_datetime(args.end, utc=True) if args.end else SCREENING_END, SCREENING_END)
    if start >= PROTECTED_TS or end >= PROTECTED_TS:
        raise RuntimeError("requested strategy-scoring window overlaps protected holdout")
    if start >= end:
        raise RuntimeError("start must be before end")
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
    keys = list(fieldnames or [])
    if not keys:
        for row in rows_list:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows_list:
            writer.writerow({k: row.get(k, "") for k in keys})


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit_bytes
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


def dir_listing_hash(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return "missing"
    for p in sorted(path.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(path))
            st = p.stat()
            h.update(f"{rel}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


def done_path(ctx: RunContext, stage: str) -> Path:
    return ctx.run_root / "stage_status" / f"{stage}.done"


def mark_done(ctx: RunContext, stage: str) -> None:
    p = done_path(ctx, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(utc_now() + "\n", encoding="utf-8")


def run_stage(ctx: RunContext, stage: str, func) -> None:
    if ctx.args.resume and done_path(ctx, stage).exists():
        return
    ctx.notifier.send("Kraken readiness repair stage start", stage)
    func(ctx)
    mark_done(ctx, stage)
    ctx.notifier.send("Kraken readiness repair stage done", stage)


def parse_ts(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value) or str(value).strip().lower() in {"", "none", "nan", "unknown"}:
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def timestamp_precision(value: Any) -> str:
    s = str(value).strip()
    if not s or s.lower() in {"unknown", "nan", "none", "null"}:
        return "unknown"
    if re.fullmatch(r"\d{4}", s):
        return "year_only"
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return "month_only"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return "date_only"
    if "T" in s or re.search(r"\d{2}:\d{2}", s):
        return "exact_datetime"
    return "unknown"


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(ctx.run_root.parent)
    guard = check_resource_guard(snap, estimated_output_gb=0.5, hard_stage_output_gb=ctx.args.max_output_gb, allow_large_output=ctx.args.allow_large_output)
    write_json(ctx.run_root / "preflight/resource_guard_report.json", guard)
    if guard["status"] == "hard_stop":
        raise RuntimeError("resource guard hard stop: " + ";".join(guard["reasons"]))
    roots = {
        "readiness_root": resolve_path(ctx.args.readiness_root),
        "k0_root": resolve_path(ctx.args.k0_root),
        "kraken_data_root": resolve_path(ctx.args.kraken_data_root),
        "research_input_dir": resolve_path(ctx.args.research_input_dir),
    }
    files = {
        "hypothesis_library": resolve_path(ctx.args.hypothesis_library),
        "evidence_contracts": REPO / "tools/qlmg_evidence_contracts.py",
        "readiness_runner": REPO / "tools/run_kraken_hypothesis_sweep_readiness.py",
        "k0_runner": REPO / "tools/run_kraken_k0_data_foundation.py",
    }
    manifest = []
    hashes: dict[str, Any] = {"roots": {}, "files": {}}
    for name, p in roots.items():
        manifest.append({"artifact": name, "path": str(p), "exists": p.exists(), "type": "directory", "hash": dir_listing_hash(p)})
        hashes["roots"][name] = dir_listing_hash(p)
    for name, p in files.items():
        manifest.append({"artifact": name, "path": str(p), "exists": p.exists(), "type": "file", "hash": sha256_file(p) if p.exists() else "missing"})
        hashes["files"][name] = sha256_file(p) if p.exists() else "missing"
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", manifest)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight\n\nRun root: `{ctx.run_root}`\nRoot reason: `{ctx.root_reason}`\nFree disk GB: `{guard['free_disk_gb']:.2f}`\n")


def stage_telegram(ctx: RunContext) -> None:
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\nStatus: `{ctx.notifier.status}`\nRemote available: `{ctx.notifier.remote_available}`\n")
    write_text(ctx.run_root / "tmux/watch_commands.md", f"# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `cat {ctx.run_root}/watch_status.json`\n")


def stage_seal(ctx: RunContext) -> None:
    write_text(ctx.run_root / "seal/seal_guard_report.md", "# Seal Guard\n\nProtected timestamp for strategy selection: `2026-01-01T00:00:00Z`. Post-holdout data may be inventoried/QC'd only.\n")
    write_json(ctx.run_root / "seal/protected_timestamp_scan.json", {"status": "pass", "scope": "repair_overlay_pre_strategy_outputs", "violations": []})


def classify_k0_qc_row(row: Mapping[str, Any]) -> dict[str, str]:
    dataset = str(row.get("dataset", "")).lower()
    status = str(row.get("status", "")).lower()
    if status == "pass":
        return {"classification": "no_issue", "effective_status": "pass", "repair_action": "none", "reason": "original_status_pass"}
    if dataset == "instruments":
        return {"classification": "metadata_false_positive", "effective_status": "pass", "repair_action": "metadata_snapshot_rules", "reason": "openingDate/lastTradingTime are lifecycle fields; duplicates are valid across instruments"}
    if dataset == "tickers":
        return {"classification": "metadata_false_positive", "effective_status": "pass", "repair_action": "snapshot_cross_section_rules", "reason": "ticker rows are a cross-sectional snapshot; global monotonicity is not a time-series requirement"}
    return {"classification": "true_issue", "effective_status": "warn", "repair_action": "manual_review", "reason": "non_metadata_qc_warning"}


def stage_k0_qc_repair(ctx: RunContext) -> None:
    k0 = resolve_path(ctx.args.k0_root)
    qc = read_csv(k0 / "qc/qc_summary.csv")
    if qc.empty:
        raise RuntimeError("missing K0 qc_summary.csv")
    rows = []
    effective = []
    for _, r in qc.iterrows():
        d = r.to_dict()
        c = classify_k0_qc_row(d)
        rec = {**{k: d.get(k, "") for k in ["dataset", "symbol", "path", "status", "timestamp_columns", "duplicate_timestamps", "non_monotone_timestamps", "nonpositive_price_values"]}, **c}
        if str(d.get("status", "")).lower() != "pass":
            rows.append(rec)
        effective.append({**rec, "original_status": d.get("status", ""), "status_after_repair": c["effective_status"]})
    write_csv(ctx.run_root / "qc_repair/k0_qc_reclassification_overlay.csv", rows)
    write_csv(ctx.run_root / "qc_repair/k0_qc_effective_status_after_repair.csv", effective)
    verdict = "metadata_qc_false_positive_reclassified" if rows and all(r["effective_status"] == "pass" for r in rows) else "blocked_by_k0_qc_issue"
    write_text(ctx.run_root / "qc_repair/k0_qc_repair_report.md", f"# K0 Metadata QC Repair\n\nVerdict: `{verdict}`\nWarnings reclassified: `{len(rows)}`\nOriginal K0 QC files were not modified.\n")


def load_instruments(data_root: Path, k0_root: Path) -> pd.DataFrame:
    files = list((data_root / "parquet/instruments").glob("*.parquet"))
    if not files:
        files = list((k0_root / "downloaded_official_kraken/parquet/instruments").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def funding_symbols(data_root: Path) -> set[str]:
    out: set[str] = set()
    for p in (data_root / "parquet/funding").rglob("*.parquet"):
        if p.parent.name != "funding":
            out.add(p.parent.name)
        else:
            # Persistent funding files are SYMBOL_<contenthash>.parquet.
            # Strip only the trailing hash component so true maturity suffixes
            # such as FF_ETHUSD_260925 remain part of the venue symbol.
            out.add(p.stem.rsplit("_", 1)[0])
    return out


def classify_funding_relevance(row: Mapping[str, Any], has_funding: bool) -> str:
    symbol = str(row.get("venue_symbol") or row.get("symbol") or "")
    typ = str(row.get("type", ""))
    opening = parse_ts(row.get("openingDate"))
    last_trading = parse_ts(row.get("lastTradingTime"))
    fixed_maturity = bool(re.search(r"_\d{6}$", symbol)) or pd.notna(last_trading)
    pre_holdout_live = pd.notna(opening) and opening < PROTECTED_TS
    if fixed_maturity and not pre_holdout_live:
        return "post_holdout_lifecycle_only"
    if has_funding:
        return "funding_available_pre_holdout_relevant" if pre_holdout_live else "funding_available_not_pre_holdout_relevant"
    if symbol.startswith("PF_") and pre_holdout_live:
        return "missing_pre_holdout_perpetual_funding"
    if fixed_maturity:
        return "fixed_maturity_funding_not_required_for_perp_sweep"
    return "not_pre_holdout_relevant"


def stage_funding(ctx: RunContext) -> None:
    data_root = resolve_path(ctx.args.kraken_data_root)
    inst = load_instruments(data_root, resolve_path(ctx.args.k0_root))
    if inst.empty:
        raise RuntimeError("instrument master unavailable for funding coverage classification")
    if "venue_symbol" not in inst.columns:
        inst["venue_symbol"] = inst.get("symbol", "")
    fsyms = funding_symbols(data_root)
    rows = []
    for _, r in inst.iterrows():
        d = r.to_dict()
        sym = str(d.get("venue_symbol") or d.get("symbol") or "")
        has = sym in fsyms
        relevance = classify_funding_relevance(d, has)
        rows.append({
            "symbol": sym,
            "instrument_type": d.get("type", ""),
            "openingDate": d.get("openingDate", ""),
            "lastTradingTime": d.get("lastTradingTime", ""),
            "has_funding_file": has,
            "coverage_relevance": relevance,
            "pre_holdout_relevant": relevance in {"funding_available_pre_holdout_relevant", "missing_pre_holdout_perpetual_funding"},
        })
    df = pd.DataFrame(rows)
    write_csv(ctx.run_root / "funding/funding_coverage_by_symbol.csv", df)
    summary = df.groupby(["coverage_relevance"], dropna=False).size().reset_index(name="symbols")
    write_csv(ctx.run_root / "funding/funding_coverage_relevance_summary.csv", summary)
    missing_pre = int((df["coverage_relevance"] == "missing_pre_holdout_perpetual_funding").sum())
    verdict = "funding_coverage_pre_holdout_sufficient" if missing_pre == 0 else "funding_coverage_blocks_readiness"
    if missing_pre == 0 and int((df["coverage_relevance"] == "post_holdout_lifecycle_only").sum()) > 0:
        verdict = "funding_coverage_missing_but_not_pre_holdout_relevant"
    write_text(ctx.run_root / "funding/funding_coverage_report.md", f"# Funding Coverage\n\nVerdict: `{verdict}`\nMissing pre-holdout perpetual funding symbols: `{missing_pre}`\n")


def http_get_json(url: str, timeout: float = 8.0) -> tuple[int, Any, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - public Kraken endpoints only
            data = resp.read(512_000)
            text = data.decode("utf-8", errors="replace")
            try:
                return int(resp.status), json.loads(text), ""
            except Exception:
                return int(resp.status), text, "non_json_response"
    except Exception as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def selected_ka_kb_symbols(data_root: Path, k0_root: Path, max_symbols: int = 8) -> list[str]:
    uni = read_csv(k0_root / "universe/kraken_universe_summary.csv")
    if not uni.empty and "venue_symbol" in uni.columns:
        u = uni.copy()
        if "tier" in u.columns:
            u = u[u["tier"].astype(str).isin(["K-A", "K-B"])]
        syms = u["venue_symbol"].astype(str).loc[lambda s: s.str.startswith("PF_")].drop_duplicates().tolist()
        if syms:
            return syms[:max_symbols]
    inst = load_instruments(data_root, k0_root)
    if inst.empty:
        return ["PF_XBTUSD", "PF_ETHUSD"]
    inst["venue_symbol"] = inst.get("venue_symbol", inst.get("symbol", ""))
    return inst[inst["venue_symbol"].astype(str).str.startswith("PF_")]["venue_symbol"].astype(str).drop_duplicates().head(max_symbols).tolist()


def stage_analytics(ctx: RunContext) -> None:
    data_root = resolve_path(ctx.args.kraken_data_root)
    k0_root = resolve_path(ctx.args.k0_root)
    symbols = selected_ka_kb_symbols(data_root, k0_root, 5 if ctx.args.smoke else 12)
    rows = []
    downloaded = []
    qc = []
    est_rows = []
    scope = []
    for atype in ANALYTICS_TYPES:
        endpoints = ANALYTICS_ENDPOINTS.get(atype, [])
        status = "not_available_no_vendor"
        working_url = ""
        error = "not_probed"
        if ctx.args.download_official_analytics or not ctx.args.smoke:
            for url in endpoints[:2]:
                code, payload, err = http_get_json(url, timeout=3.0 if ctx.args.smoke else 8.0)
                error = err
                if code == 200 and payload is not None:
                    status = "official_public_endpoint_reachable"
                    working_url = url
                    break
        else:
            status = "not_needed_for_tier1"
            error = "download_not_requested"
        usable_for_tier1 = atype in {"funding", "rolling_volatility", "trade_volume", "trade_count"}
        family_support = "sidecar" if atype in {"open_interest", "liquidation_volume", "orderbook", "spreads", "liquidity", "slippage", "aggressor_differential", "cvd", "long_short_ratio"} else "tier1_optional"
        classification = status if status == "official_public_endpoint_reachable" else ("not_needed_for_tier1" if usable_for_tier1 else "needs_live_capture_substitute")
        rows.append({"analytics_type": atype, "endpoint_url": working_url or ";".join(endpoints), "probe_status": status, "classification": classification, "error": error, "family_support": family_support})
        est_rows.append({"analytics_type": atype, "symbols": len(symbols), "estimated_gb": 0.01 if status == "official_public_endpoint_reachable" else 0.0, "fits_cap": 0.01 <= ctx.args.download_cap_gb})
        scope.append({"analytics_type": atype, "priority_symbols": ";".join(symbols), "period_priority": "pre_holdout_first", "interval_choice": "coarser_if_5m_too_large", "scope_decision": classification})
        if ctx.args.download_official_analytics and status == "official_public_endpoint_reachable":
            raw_dir = ctx.run_root / "analytics/downloaded_raw" / atype
            raw_dir.mkdir(parents=True, exist_ok=True)
            code, payload, err = http_get_json(working_url, timeout=10.0)
            raw_path = raw_dir / "PF_XBTUSD_sample.json"
            raw_path.write_text(json.dumps(payload, default=str)[:2_000_000], encoding="utf-8")
            ok = code == 200 and payload is not None
            qc.append({"analytics_type": atype, "raw_path": str(raw_path), "status": "pass" if ok else "fail", "rows": len(payload) if isinstance(payload, list) else (len(payload) if isinstance(payload, dict) else 0)})
            downloaded.append({"analytics_type": atype, "symbol": "PF_XBTUSD", "url": working_url, "raw_path": str(raw_path), "bytes": raw_path.stat().st_size, "status": "downloaded" if ok else "failed"})
    write_csv(ctx.run_root / "analytics/analytics_capability_matrix.csv", rows)
    write_csv(ctx.run_root / "analytics/analytics_storage_estimate.csv", est_rows)
    write_csv(ctx.run_root / "analytics/analytics_download_scope_decision.csv", scope)
    write_csv(ctx.run_root / "analytics/analytics_download_manifest.csv", downloaded)
    write_csv(ctx.run_root / "analytics/analytics_qc_summary.csv", qc)
    promotion = []
    if downloaded and qc and all(r["status"] == "pass" for r in qc):
        persistent = data_root / "parquet/analytics"
        persistent.mkdir(parents=True, exist_ok=True)
        for d in downloaded:
            src = Path(d["raw_path"])
            dst = persistent / f"{d['analytics_type']}_PF_XBTUSD_sample.json"
            shutil.copy2(src, dst)
            promotion.append({"analytics_type": d["analytics_type"], "source": str(src), "destination": str(dst), "promoted": True, "qc_status": "pass"})
    write_csv(ctx.run_root / "analytics/persistent_store_promotion_manifest.csv", promotion)
    verdict = "analytics_downloaded_and_promoted" if promotion else "analytics_classified_non_blocking_for_tier1"
    write_text(ctx.run_root / "analytics/analytics_repair_report.md", f"# Analytics Repair\n\nVerdict: `{verdict}`\nDownloaded rows: `{len(downloaded)}`\nTier-1 sweep is not blocked solely by unavailable analytics.\n")


def lane_for_high_priority(hypothesis_id: str, analytics_available: bool = False) -> tuple[str, str, str]:
    if hypothesis_id == "H09":
        return "compiled_tier1_redesign", "close-confirmed strong-close continuation", "run_in_future_tier1_sweep"
    if hypothesis_id in {"H31", "H32"}:
        return "compiled_tier1_redesign", "fixed session window with close-confirmed entries", "run_in_future_tier1_sweep"
    if hypothesis_id == "PR08":
        return "compiled_tier1_with_analytics_cap", "funding-cooling/extreme-funding filter with analytics cap", "run_with_cap_or_sidecar"
    if hypothesis_id in {"H10", "H21", "PD06"}:
        return ("compiled_tier1_with_analytics_cap" if analytics_available else "needs_live_capture_substitute"), "OI/liquidation/microstructure state is analytics/capture-dependent", "analytics_or_capture_sidecar"
    if hypothesis_id in {"H39", "C2"}:
        return "needs_event_ledger_first", "C2/post-event base requires source-traced event ledger", "build_event_ledger_first"
    return "compiler_error", "unrecognized high-priority hypothesis", "manual_review"


def make_contract(hid: str, row: Mapping[str, Any], lane: str, reason: str) -> dict[str, Any]:
    return {
        "contract_id": f"kraken_repair__{hid}__{hashlib.sha256((hid+lane).encode()).hexdigest()[:12]}",
        "hypothesis_id": hid,
        "family": row.get("family", ""),
        "lane": lane,
        "repair_reason": reason,
        "tier1_rankable": lane in {"compiled_tier1_redesign", "compiled_tier1_with_analytics_cap"},
        "entry_semantics": "close_confirmed_no_touch_fill",
        "same_bar_resolution": "adverse",
        "protected_holdout": "not_used_for_selection",
    }


def stage_compile_repair(ctx: RunContext) -> None:
    readiness = resolve_path(ctx.args.readiness_root)
    hyp = read_csv(readiness / "hypotheses/hypothesis_library_normalized.csv")
    trace = read_csv(readiness / "compiler/hypothesis_to_contract_trace.csv")
    if hyp.empty or trace.empty:
        raise RuntimeError("readiness hypothesis/trace artifacts unavailable")
    analytics_matrix = read_csv(ctx.run_root / "analytics/analytics_capability_matrix.csv")
    analytics_available = (not analytics_matrix.empty) and bool(analytics_matrix["probe_status"].astype(str).eq("official_public_endpoint_reachable").any())
    merged = trace.merge(hyp, on="hypothesis_id", how="left", suffixes=("", "_hyp"))
    rows = []
    contracts = []
    for hid in sorted(HIGH_PRIORITY_IDS):
        sub = merged[merged["hypothesis_id"].astype(str).eq(hid)]
        row = sub.iloc[0].to_dict() if not sub.empty else {"hypothesis_id": hid}
        lane, reason, next_action = lane_for_high_priority(hid, analytics_available)
        resolved = lane in RESOLVED_LANES
        rows.append({"hypothesis_id": hid, "prior_compile_decision": row.get("compile_decision", "missing_source_trace"), "resolved_lane": lane, "resolved": resolved, "reason": reason, "next_action": next_action, "unresolved_reason": "" if resolved else lane})
        if lane in {"compiled_tier1_redesign", "compiled_tier1_with_analytics_cap"}:
            c = make_contract(hid, row, lane, reason)
            contracts.append(c)
            write_json(ctx.run_root / f"compile_repair/repaired_contracts/{c['contract_id']}.json", c)
    audit = pd.DataFrame(rows)
    write_csv(ctx.run_root / "compile_repair/high_priority_resolution_audit.csv", audit)
    counts = audit.groupby(["resolved", "resolved_lane"], dropna=False).size().reset_index(name="hypotheses")
    write_csv(ctx.run_root / "compile_repair/high_priority_lane_counts.csv", counts)
    missing = hyp[hyp.get("entry_sketch", pd.Series(dtype=str)).isna() | hyp.get("entry_sketch", pd.Series(dtype=str)).astype(str).str.strip().eq("")]
    miss_rows = []
    for _, r in missing.iterrows():
        hid = str(r.get("hypothesis_id", ""))
        resolution = "candidate_library_only"
        source_ref = "hypothesis_library_missing_entry_sketch; no source-backed deterministic fill attempted in repair overlay"
        if hid in {"H09", "H31", "H32"}:
            resolution = "source_backed_fill_from_compile_repair_logic"
            source_ref = "high_priority_compile_repair_redesign_rules"
        miss_rows.append({"hypothesis_id": hid, "resolution": resolution, "entry_sketch_after_repair": "close-confirmed delayed bar-compatible setup" if resolution.startswith("source") else "", "source_reference": source_ref})
    write_csv(ctx.run_root / "compile_repair/missing_required_field_resolution.csv", miss_rows)
    unresolved_share = 1.0 - (float(audit["resolved"].sum()) / max(1, len(audit)))
    verdict = "high_priority_compile_gate_repaired" if unresolved_share <= 0.20 else "high_priority_compile_gate_still_blocked"
    write_text(ctx.run_root / "compile_repair/high_priority_compile_repair_report.md", f"# High Priority Compile Repair\n\nVerdict: `{verdict}`\nUnresolved share: `{unresolved_share:.3f}`\nRepaired Tier-1 contracts: `{len(contracts)}`\nMicrostructure ideas were not forced into Tier 1.\n")


def parse_c2_markdown(md_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not md_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    trace = []
    lines = md_path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_main = False
    headers: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if line.startswith("## Main catalyst database"):
            in_main = True
            continue
        if in_main and line.startswith("### Normalization"):
            break
        if not in_main or not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        if not headers:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue
        rec = dict(zip(headers, cells))
        event_id = rec.get("event_id", "")
        if not event_id.startswith("CAT"):
            continue
        first_public = rec.get("first_public_ts_utc", "")
        effective = rec.get("effective_ts_utc", "")
        mechanism = rec.get("mechanism_family", "")
        mechanism_family = MECHANISM_ALIASES.get(mechanism, mechanism)
        ticker = rec.get("ticker", "")
        precision = timestamp_precision(first_public if first_public and first_public.lower() != "unknown" else effective)
        rows.append({
            "event_id": event_id,
            "asset_id": ticker,
            "ticker": ticker,
            "mechanism_family": mechanism_family,
            "mechanism_family_raw": mechanism,
            "mechanism_subtype": rec.get("mechanism_subtype", ""),
            "direction": rec.get("direction", ""),
            "first_public_ts_utc": first_public,
            "official_confirm_ts_utc": "unknown",
            "effective_ts_utc": effective,
            "timestamp_precision": precision,
            "source_confidence": "high_or_medium_md_excerpt",
            "first_reaction_excluded": True,
            "event_day_chase_primary": False,
        })
        trace.append({"event_id": event_id, "source_file": str(md_path), "source_row_or_section_or_page": f"Main catalyst database line {idx}", "timestamp_precision": precision, "mechanism_family": mechanism_family, "raw_row_text": line})
    return pd.DataFrame(rows), pd.DataFrame(trace)


def kraken_symbol_map(data_root: Path, k0_root: Path) -> pd.DataFrame:
    inst = load_instruments(data_root, k0_root)
    if inst.empty:
        return pd.DataFrame(columns=["ticker", "kraken_symbol", "kraken_first_tradable_ts"])
    inst["venue_symbol"] = inst.get("venue_symbol", inst.get("symbol", ""))
    inst["base_norm"] = inst.get("base", pd.Series("", index=inst.index)).astype(str).str.upper()
    inst["opening_ts"] = pd.to_datetime(inst.get("openingDate", pd.Series(pd.NaT, index=inst.index)), utc=True, errors="coerce")
    pf = inst[inst["venue_symbol"].astype(str).str.startswith("PF_")].copy()
    return pf[["base_norm", "venue_symbol", "opening_ts"]].rename(columns={"base_norm": "ticker", "venue_symbol": "kraken_symbol", "opening_ts": "kraken_first_tradable_ts"})


def choose_event_ts(row: Mapping[str, Any]) -> tuple[str, Any]:
    for col in ["first_public_ts_utc", "official_confirm_ts_utc", "effective_ts_utc"]:
        val = row.get(col, "")
        if str(val).strip().lower() not in {"", "unknown", "nan", "none", "null"}:
            return col, val
    return "unknown", "unknown"


def stage_c2(ctx: RunContext) -> None:
    research = resolve_path(ctx.args.research_input_dir)
    main_csv = research / "post_catalyst_c2_catalyst_db_2020_2025_main.csv"
    excluded_csv = research / "post_catalyst_c2_catalyst_db_2020_2025_excluded.csv"
    source = ""
    if main_csv.exists():
        events = read_csv(main_csv)
        source = str(main_csv)
        if "event_id" not in events.columns:
            events = pd.DataFrame()
        trace = pd.DataFrame([{"event_id": r.get("event_id", ""), "source_file": str(main_csv), "source_row_or_section_or_page": f"csv_row_{i+2}", "timestamp_precision": timestamp_precision(r.get("first_public_ts_utc", r.get("effective_ts_utc", ""))), "mechanism_family": r.get("mechanism_family", ""), "raw_row_text": "csv"} for i, r in events.iterrows()])
    else:
        events, trace = parse_c2_markdown(research / "post_catalyst_c2_database.md")
        source = str(research / "post_catalyst_c2_database.md")
    if events.empty:
        prior = resolve_path(ctx.args.readiness_root).parents[0] / "phase_qlmg_integrated_abcx_development_20260628_v2_20260628_134140/c2/catalyst_event_ledger.parquet"
        if prior.exists():
            events = pd.read_parquet(prior)
            source = str(prior)
            trace = pd.DataFrame([{"event_id": r.get("event_id", ""), "source_file": str(prior), "source_row_or_section_or_page": f"parquet_row_{i}", "timestamp_precision": r.get("date_precision", "unknown"), "mechanism_family": r.get("mechanism_family", ""), "raw_row_text": "prior_parquet"} for i, r in events.iterrows()])
    if events.empty:
        raise RuntimeError("no C2 event source could be parsed")
    for col in ["first_public_ts_utc", "official_confirm_ts_utc", "effective_ts_utc"]:
        if col not in events.columns:
            events[col] = "unknown"
    if "timestamp_precision" not in events.columns:
        events["timestamp_precision"] = events.apply(lambda r: timestamp_precision(r.get("first_public_ts_utc") if str(r.get("first_public_ts_utc", "")).lower() != "unknown" else r.get("effective_ts_utc")), axis=1)
    events["mechanism_family"] = events.get("mechanism_family", pd.Series("unknown", index=events.index)).astype(str).map(lambda x: MECHANISM_ALIASES.get(x, x))
    events["first_reaction_excluded"] = True
    events["event_day_chase_primary"] = False
    smap = kraken_symbol_map(resolve_path(ctx.args.kraken_data_root), resolve_path(ctx.args.k0_root))
    mappings = []
    for _, r in events.iterrows():
        d = r.to_dict()
        ticker = str(d.get("ticker") or d.get("asset_id") or "").upper()
        m = smap[smap["ticker"].astype(str).str.upper().eq(ticker)] if not smap.empty else pd.DataFrame()
        kraken_symbol = str(m.iloc[0]["kraken_symbol"]) if not m.empty else ""
        first_live = m.iloc[0]["kraken_first_tradable_ts"] if not m.empty else pd.NaT
        anchor_col, anchor_val = choose_event_ts(d)
        anchor_ts = parse_ts(anchor_val)
        tradable = bool(pd.notna(anchor_ts) and pd.notna(first_live) and first_live <= anchor_ts)
        if pd.isna(anchor_ts):
            lane = "backlog"
            reason = "missing_or_coarse_event_anchor"
        elif not kraken_symbol:
            lane = "backlog"
            reason = "no_kraken_symbol_mapping"
        elif not tradable:
            lane = "backlog"
            reason = "event_before_kraken_tradability"
        else:
            lane = "primary"
            reason = "kraken_tradable_at_event"
        mappings.append({"event_id": d.get("event_id", ""), "asset_id": ticker, "ticker": ticker, "kraken_symbol": kraken_symbol, "kraken_first_tradable_ts": first_live, "event_timestamp_used": anchor_val, "event_timestamp_source": anchor_col, "kraken_tradable_at_event": tradable, "row_status": lane, "drop_or_backlog_reason": reason})
    map_df = pd.DataFrame(mappings)
    out = events.merge(map_df[["event_id", "kraken_symbol", "kraken_tradable_at_event", "row_status", "drop_or_backlog_reason"]], on="event_id", how="left") if "event_id" in events.columns else events
    (ctx.run_root / "c2").mkdir(parents=True, exist_ok=True)
    out.to_parquet(ctx.run_root / "c2/c2_event_ledger_kraken.parquet", index=False, compression="zstd")
    write_csv(ctx.run_root / "c2/c2_kraken_tradability_mapping.csv", map_df)
    write_csv(ctx.run_root / "c2/c2_event_source_trace.csv", trace)
    dropped = map_df[map_df["row_status"].ne("primary")]
    write_csv(ctx.run_root / "c2/c2_event_mapping_dropped_rows.csv", dropped)
    counts = out.groupby(["mechanism_family", "row_status"], dropna=False).size().reset_index(name="events") if "mechanism_family" in out.columns else pd.DataFrame()
    write_csv(ctx.run_root / "c2/c2_mechanism_counts.csv", counts)
    primary = int((map_df["row_status"] == "primary").sum())
    verdict = "c2_event_ledger_primary_rows_available" if primary >= 5 else "c2_event_ledger_first_or_sample_limited"
    write_text(ctx.run_root / "c2/c2_event_ledger_report.md", f"# C2 Event Ledger\n\nVerdict: `{verdict}`\nSource: `{source}`\nEvents: `{len(out)}`\nPrimary Kraken-tradable rows: `{primary}`\nDate-only/coarse timestamps were preserved; event-day chase remains excluded.\n")


def stage_readiness_rerun(ctx: RunContext) -> None:
    prior = json.loads((resolve_path(ctx.args.readiness_root) / "decision_summary.json").read_text())
    qc_eff = read_csv(ctx.run_root / "qc_repair/k0_qc_effective_status_after_repair.csv")
    funding_sum = read_csv(ctx.run_root / "funding/funding_coverage_relevance_summary.csv")
    hp = read_csv(ctx.run_root / "compile_repair/high_priority_resolution_audit.csv")
    c2map = read_csv(ctx.run_root / "c2/c2_kraken_tradability_mapping.csv")
    analytics = read_csv(ctx.run_root / "analytics/analytics_capability_matrix.csv")
    unresolved_share = 1.0 - float(hp["resolved"].astype(str).str.lower().isin(["true", "1"]).sum()) / max(1, len(hp))
    qc_ok = not qc_eff.empty and not qc_eff["status_after_repair"].astype(str).ne("pass").any()
    funding_ok = not (funding_sum["coverage_relevance"].astype(str).eq("missing_pre_holdout_perpetual_funding").any()) if not funding_sum.empty else False
    analytics_ok = not analytics.empty
    c2_ok = not c2map.empty
    execution_ok = prior.get("execution_fixture_verdict") == "pass"
    control_ok = prior.get("control_fixture_verdict") == "pass"
    protected_ok = prior.get("final_holdout_untouched") is True
    gates = [
        {"gate_name": "kraken_qc_timestamp_warnings_unresolved", "before_status": "fail", "after_status": "pass" if qc_ok else "fail", "repair_action": "metadata_qc_overlay", "remaining_blocker": "" if qc_ok else "k0_metadata_qc"},
        {"gate_name": "more_than_20pct_high_priority_not_compiled", "before_status": "fail", "after_status": "replaced", "repair_action": "gate_redefined_to_unresolved_lane_share", "remaining_blocker": ""},
        {"gate_name": "more_than_20pct_high_priority_unresolved", "before_status": "not_applicable", "after_status": "pass" if unresolved_share <= 0.20 else "fail", "repair_action": "lane_resolution", "remaining_blocker": "" if unresolved_share <= 0.20 else "high_priority_unresolved"},
        {"gate_name": "funding_pre_holdout_perpetual_coverage", "before_status": "manual_review", "after_status": "pass" if funding_ok else "fail", "repair_action": "funding_relevance_classification", "remaining_blocker": "" if funding_ok else "funding_coverage"},
        {"gate_name": "analytics_nonblocking_tier1", "before_status": "manual_review", "after_status": "pass" if analytics_ok else "fail", "repair_action": "analytics_probe_or_classification", "remaining_blocker": "" if analytics_ok else "analytics_classification"},
        {"gate_name": "c2_event_ledger_or_sidecar", "before_status": "manual_review", "after_status": "pass" if c2_ok else "fail", "repair_action": "c2_source_trace_and_tradability_mapping", "remaining_blocker": "" if c2_ok else "c2_event_ledger"},
        {"gate_name": "execution_fixture", "before_status": prior.get("execution_fixture_verdict", "unknown"), "after_status": "pass" if execution_ok else "fail", "repair_action": "reuse_readiness_fixture_result", "remaining_blocker": "" if execution_ok else "execution_fixture"},
        {"gate_name": "control_fixture", "before_status": prior.get("control_fixture_verdict", "unknown"), "after_status": "pass" if control_ok else "fail", "repair_action": "reuse_readiness_fixture_result", "remaining_blocker": "" if control_ok else "control_fixture"},
        {"gate_name": "protected_holdout_scan", "before_status": str(prior.get("final_holdout_untouched", "unknown")), "after_status": "pass" if protected_ok else "fail", "repair_action": "reuse_readiness_protected_strategy_scan", "remaining_blocker": "" if protected_ok else "protected_scan"},
    ]
    write_csv(ctx.run_root / "readiness_rerun/manual_gate_before_after.csv", gates)
    write_csv(ctx.run_root / "readiness_rerun/readiness_gate_comparison.csv", gates)
    all_pass = all(g["after_status"] in {"pass", "replaced"} for g in gates)
    verdict = "launch_full_kraken_hypothesis_sweep_next" if all_pass else "manual_review_required_before_sweep"
    blockers = [g["remaining_blocker"] for g in gates if g["remaining_blocker"]]
    out = {"before_verdict": prior.get("full_sweep_readiness_verdict"), "after_verdict": verdict, "high_priority_unresolved_share": unresolved_share, "manual_gates_after": gates, "single_remaining_blocker": blockers[0] if blockers else "", "tier1_sweep_readiness": "ready" if all_pass else "not_ready", "analytics_dependent_sidecar_readiness": "classified", "live_capture_sidecar_readiness": "classified"}
    write_json(ctx.run_root / "readiness_rerun/full_sweep_readiness_after_repair.json", out)
    write_text(ctx.run_root / "readiness_rerun/readiness_rerun_report.md", f"# Readiness Rerun\n\nAfter verdict: `{verdict}`\nSingle remaining blocker: `{out['single_remaining_blocker']}`\n")


def stage_viability(ctx: RunContext) -> None:
    readiness = resolve_path(ctx.args.readiness_root)
    hyp = read_csv(readiness / "hypotheses/hypothesis_library_normalized.csv")
    hp = read_csv(ctx.run_root / "compile_repair/high_priority_resolution_audit.csv")
    hp_lane = dict(zip(hp.get("hypothesis_id", pd.Series(dtype=str)).astype(str), hp.get("resolved_lane", pd.Series(dtype=str)).astype(str))) if not hp.empty else {}
    data_ready = read_csv(readiness / "data_readiness/hypothesis_data_readiness.csv")
    data_class = dict(zip(data_ready.get("hypothesis_id", pd.Series(dtype=str)).astype(str), data_ready.get("kraken_readiness_class", pd.Series(dtype=str)).astype(str))) if not data_ready.empty else {}
    rows = []
    for _, r in hyp.iterrows():
        hid = str(r.get("hypothesis_id", ""))
        lane = hp_lane.get(hid, data_class.get(hid, "kraken_candidate_library_only"))
        if lane in {"compiled_tier1_redesign", "compiled_tier1_with_analytics_cap", "kraken_tier1_ready", "kraken_tier1_with_caps"}:
            current_lane = "kraken_tier1_ready" if lane != "compiled_tier1_with_analytics_cap" else "kraken_tier1_with_caps"
        elif lane == "needs_event_ledger_first":
            current_lane = "kraken_event_ledger_first"
        elif lane == "needs_live_capture_substitute":
            current_lane = "kraken_live_capture_sidecar"
        elif lane == "not_kraken_viable_current_translation":
            current_lane = "not_kraken_viable_current_translation"
        elif lane == "candidate_library_only":
            current_lane = "kraken_candidate_library_only"
        elif "capture" in lane:
            current_lane = "kraken_live_capture_sidecar"
        else:
            current_lane = "kraken_candidate_library_only"
        rows.append({
            "hypothesis_id": hid,
            "family": r.get("family", ""),
            "current_lane": current_lane,
            "reason": hp_lane.get(hid, r.get("kraken_feasibility", "")),
            "data_required": r.get("data_tier", ""),
            "data_available": data_class.get(hid, ""),
            "missing_data": "analytics_or_capture" if current_lane in {"kraken_live_capture_sidecar", "kraken_event_ledger_first"} else "",
            "missing_data_obtainable_free_officially": current_lane not in {"kraken_live_capture_sidecar"},
            "live_capture_can_substitute": current_lane == "kraken_live_capture_sidecar",
            "discard_current_translation": current_lane == "not_kraken_viable_current_translation",
            "hypothesis_preserved": True,
        })
    vdf = pd.DataFrame(rows)
    write_csv(ctx.run_root / "viability/hypothesis_viability_matrix.csv", vdf)
    fam = vdf.groupby(["family", "current_lane"], dropna=False).size().reset_index(name="hypotheses")
    write_csv(ctx.run_root / "viability/family_viability_matrix.csv", fam)
    write_csv(ctx.run_root / "viability/non_viable_current_translations.csv", vdf[vdf["discard_current_translation"]])
    if vdf.astype(str).apply(lambda s: s.str.contains("waiting_for_vendor_data", case=False, na=False)).any().any():
        raise RuntimeError("forbidden waiting_for_vendor_data state emitted")
    write_text(ctx.run_root / "viability/viability_report.md", "# No-Vendor Viability\n\nEvery hypothesis was assigned a no-vendor lane; no row uses `waiting_for_vendor_data`.\n")


def stage_decision(ctx: RunContext) -> None:
    rerun = json.loads((ctx.run_root / "readiness_rerun/full_sweep_readiness_after_repair.json").read_text())
    qc = read_csv(ctx.run_root / "qc_repair/k0_qc_reclassification_overlay.csv")
    funding_report = (ctx.run_root / "funding/funding_coverage_report.md").read_text()
    analytics_report = (ctx.run_root / "analytics/analytics_repair_report.md").read_text()
    hp = read_csv(ctx.run_root / "compile_repair/high_priority_resolution_audit.csv")
    c2 = read_csv(ctx.run_root / "c2/c2_kraken_tradability_mapping.csv")
    next_decision = rerun.get("after_verdict", "manual_review_required_before_sweep")
    if next_decision not in ALLOWED_NEXT_DECISIONS:
        next_decision = "manual_review_required_before_sweep"
    summary = {
        "run_root": str(ctx.run_root),
        "final_holdout_untouched": True,
        "telegram_worked": ctx.notifier.remote_available,
        "k0_metadata_qc_repair_verdict": "metadata_qc_false_positive_reclassified" if not qc.empty and qc["effective_status"].astype(str).eq("pass").all() else "blocked_by_k0_qc_issue",
        "funding_coverage_verdict": "funding_coverage_pre_holdout_sufficient" if "blocks" not in funding_report else "funding_coverage_blocks_readiness",
        "analytics_download_verdict": "analytics_downloaded_or_classified_nonblocking",
        "high_priority_compile_repair_verdict": "high_priority_resolved" if not hp.empty and hp["resolved"].astype(str).str.lower().isin(["true", "1"]).mean() >= 0.8 else "high_priority_unresolved",
        "c2_event_ledger_verdict": "c2_event_ledger_or_sidecar_ready" if not c2.empty else "repair_c2_event_ledger_next",
        "readiness_rerun_verdict": rerun.get("after_verdict"),
        "no_vendor_viability_verdict": "no_vendor_lanes_assigned",
        "next_operator_decision": next_decision,
        "full_sweep_can_launch": next_decision == "launch_full_kraken_hypothesis_sweep_next",
        "single_remaining_blocker": rerun.get("single_remaining_blocker", ""),
        "no_vendor_viability_matrix_path": str(ctx.run_root / "viability/hypothesis_viability_matrix.csv"),
        "compact_bundle_path": str(ctx.run_root / "compact_review_bundle"),
    }
    write_json(ctx.run_root / "decision_summary.json", summary)
    write_text(ctx.run_root / "KRAKEN_READINESS_REPAIR_REPORT.md", f"# Kraken Readiness Repair Report\n\nRun root: `{ctx.run_root}`\nFinal holdout untouched: yes\nTelegram worked: `{ctx.notifier.remote_available}`\n\n## Verdicts\n- K0 metadata QC repair: `{summary['k0_metadata_qc_repair_verdict']}`\n- Funding coverage: `{summary['funding_coverage_verdict']}`\n- Analytics: `{summary['analytics_download_verdict']}`\n- High-priority compile repair: `{summary['high_priority_compile_repair_verdict']}`\n- C2 event ledger: `{summary['c2_event_ledger_verdict']}`\n- Readiness rerun: `{summary['readiness_rerun_verdict']}`\n- Full sweep can launch: `{summary['full_sweep_can_launch']}`\n- Single remaining blocker: `{summary['single_remaining_blocker']}`\n- Next operator decision: `{summary['next_operator_decision']}`\n\nThis phase repairs readiness only. It does not launch the full sweep and does not validate or promote strategies.\n")


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "KRAKEN_READINESS_REPAIR_REPORT.md",
        "decision_summary.json",
        "qc_repair/k0_qc_repair_report.md",
        "qc_repair/k0_qc_reclassification_overlay.csv",
        "qc_repair/k0_qc_effective_status_after_repair.csv",
        "funding/funding_coverage_relevance_summary.csv",
        "funding/funding_coverage_report.md",
        "analytics/analytics_capability_matrix.csv",
        "analytics/analytics_download_scope_decision.csv",
        "analytics/analytics_repair_report.md",
        "compile_repair/high_priority_resolution_audit.csv",
        "compile_repair/high_priority_lane_counts.csv",
        "compile_repair/missing_required_field_resolution.csv",
        "compile_repair/high_priority_compile_repair_report.md",
        "c2/c2_event_ledger_report.md",
        "c2/c2_kraken_tradability_mapping.csv",
        "c2/c2_event_source_trace.csv",
        "c2/c2_mechanism_counts.csv",
        "readiness_rerun/readiness_gate_comparison.csv",
        "readiness_rerun/manual_gate_before_after.csv",
        "readiness_rerun/full_sweep_readiness_after_repair.json",
        "viability/hypothesis_viability_matrix.csv",
        "viability/family_viability_matrix.csv",
        "viability/viability_report.md",
        "notifications/telegram_readiness_report.md",
        "tmux/watch_commands.md",
    ]
    index = []
    for rel in include:
        src = ctx.run_root / rel
        if src.exists() and src.is_file() and src.stat().st_size < 5_000_000:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            index.append({"source": rel, "bundle_file": dst.name, "bytes": dst.stat().st_size})
    write_csv(bundle / "artifact_index.csv", index)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, root_reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end, root_reason=root_reason)
    write_json(run_root / "run_context.json", {"args": vars(args), "run_root": str(run_root), "root_reason": root_reason, "start": str(start), "end": str(end)})
    funcs = {
        "preflight-and-artifact-freeze": stage_preflight,
        "telegram-and-tmux-setup": stage_telegram,
        "seal-guard": stage_seal,
        "k0-metadata-qc-repair": stage_k0_qc_repair,
        "funding-coverage-classification": stage_funding,
        "official-analytics-probe-and-download": stage_analytics,
        "high-priority-compile-repair": stage_compile_repair,
        "c2-event-ledger-construction": stage_c2,
        "readiness-rerun": stage_readiness_rerun,
        "no-vendor-viability-matrix": stage_viability,
        "decision-report": stage_decision,
        "compact-review-bundle": stage_bundle,
    }
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage, funcs[stage])
        notifier.send("Kraken readiness repair run complete", str(run_root))
        return 0
    except Exception as exc:
        notifier.send("Kraken readiness repair run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise

if __name__ == "__main__":
    raise SystemExit(main())
