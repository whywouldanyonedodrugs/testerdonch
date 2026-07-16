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
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_screening_core import (  # noqa: E402
    check_resource_guard,
    resource_snapshot,
    utc_now,
    write_json,
)
from tools.run_qlmg_engine_and_first_screen import (  # noqa: E402
    DATA_5M,
    DATA_CONTEXT,
    FINAL_HOLDOUT_START,
    SCREENING_END,
)

try:
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_qlmg_targeted_1m_data_pilot_20260624_v1"
PRIOR_ROOT = REPO / "results/rebaseline/phase_qlmg_path_diagnostics_exit_surface_20260624_v1_20260624_121522"
DATA_1M_HOT = Path("/opt/parquet/1m_hot")
DATA_1M = Path("/opt/parquet/1m")
BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")
REQUEST_TIMEOUT = 20
RATE_LIMIT_RET_CODES = {10006, 10018, 10016}

STAGES = (
    "preflight-resource-and-api-audit",
    "telegram-and-tmux-setup",
    "seal-guard",
    "prior-d3-e1-event-window-extraction",
    "window-dedup-and-storage-estimate",
    "bybit-1m-source-capability-audit",
    "pilot-window-selection",
    "pilot-download-if-safe",
    "pilot-data-qc",
    "one-minute-impact-replay-sample",
    "full-acquisition-decision",
    "compact-review-bundle",
    "all",
)

DATASETS = {
    "ohlcv_1m": {
        "endpoint": "/v5/market/kline",
        "interval": "1",
        "columns": ["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        "out_dir": "bybit_linear_ohlcv_1m",
    },
    "mark_1m": {
        "endpoint": "/v5/market/mark-price-kline",
        "interval": "1",
        "columns": ["timestamp", "open", "high", "low", "close"],
        "out_dir": "bybit_linear_mark_1m",
    },
    "index_1m": {
        "endpoint": "/v5/market/index-price-kline",
        "interval": "1",
        "columns": ["timestamp", "open", "high", "low", "close"],
        "out_dir": "bybit_linear_index_1m",
    },
    "premium_1m": {
        "endpoint": "/v5/market/premium-index-price-kline",
        "interval": "1",
        "columns": ["timestamp", "open", "high", "low", "close"],
        "out_dir": "bybit_linear_premium_1m",
    },
}
OPTIONAL_DATASETS = {
    "open_interest_5m": {"endpoint": "/v5/market/open-interest", "out_dir": "bybit_linear_open_interest_5m"},
    "funding_history": {"endpoint": "/v5/market/funding/history", "out_dir": "bybit_linear_funding_history"},
}

ALLOWED_VERDICTS = {
    "expand_targeted_1m_acquisition",
    "pilot_enough_continue_with_d3_d4_e1_validation",
    "do_not_download_more_1m_now",
    "blocked_api_or_source_unavailable",
    "blocked_storage_budget",
    "blocked_protocol_issue",
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
                self.notifier = TelegramNotifier.from_args(_Args(), run_label="qlmg-1m-pilot")
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
    p = argparse.ArgumentParser(description="Targeted QLMG 1m data pilot for D3/D4/E1 windows")
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
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--max-events-per-family", type=int, default=100)
    p.add_argument("--nulls-per-event", type=int, default=1)
    p.add_argument("--pilot-download-cap-gb", type=float, default=5.0)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--download", action="store_true")
    g.add_argument("--no-download", action="store_true")
    p.add_argument("--tmux-session-name", default="qlmg_1m_pilot")
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
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def shell(args: Sequence[str], timeout: float = 120.0) -> str:
    try:
        p = subprocess.run(args, cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


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


def done_path(run_root: Path, stage: str) -> Path:
    return run_root / "stage_status" / f"{stage}.done"


def mark_done(run_root: Path, stage: str) -> None:
    write_text(done_path(run_root, stage), utc_now())


def append_command(run_root: Path, stage: str) -> None:
    p = run_root / "command_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": utc_now(), "stage": stage, "argv": sys.argv, "cwd": str(REPO)}, sort_keys=True) + "\n")


def required_outputs_for_stage(run_root: Path, stage: str) -> list[Path]:
    mapping = {
        "preflight-resource-and-api-audit": [run_root / "preflight/preflight_report.md", run_root / "preflight/local_1m_inventory.csv", run_root / "preflight/api_capability_report.md"],
        "telegram-and-tmux-setup": [run_root / "notifications/telegram_readiness_report.md", run_root / "tmux/watch_commands.md"],
        "seal-guard": [run_root / "seal/seal_guard_report.md", run_root / "seal/protected_slice_check.json"],
        "prior-d3-e1-event-window-extraction": [run_root / "windows/raw_event_windows.csv", run_root / "windows/window_extraction_report.md"],
        "window-dedup-and-storage-estimate": [run_root / "windows/deduped_windows.csv", run_root / "windows/storage_estimate.csv"],
        "bybit-1m-source-capability-audit": [run_root / "source_capability/source_capability_report.md", run_root / "source_capability/sample_download_manifest.csv"],
        "pilot-window-selection": [run_root / "pilot/pilot_windows.csv", run_root / "pilot/pilot_selection_report.md"],
        "pilot-download-if-safe": [run_root / "downloaded_1m/download_manifest.csv", run_root / "downloaded_1m/download_report.md"],
        "pilot-data-qc": [run_root / "qc/pilot_data_qc_report.md", run_root / "qc/pilot_coverage_summary.csv"],
        "one-minute-impact-replay-sample": [run_root / "impact/one_minute_impact_summary.csv", run_root / "impact/one_minute_impact_report.md"],
        "full-acquisition-decision": [run_root / "FULL_1M_ACQUISITION_DECISION.md", run_root / "decision_summary.json"],
        "compact-review-bundle": [run_root / "compact_review_bundle/artifact_path_index.csv"],
    }
    return mapping.get(stage, [])


def stage_complete(run_root: Path, stage: str) -> bool:
    return done_path(run_root, stage).exists() and all(p.exists() for p in required_outputs_for_stage(run_root, stage))


def estimate_stage_gb(stage: str, smoke: bool, download: bool, cap_gb: float) -> float:
    if smoke:
        return 0.1 if not download else min(0.5, cap_gb)
    if stage == "pilot-download-if-safe" and download:
        return min(float(cap_gb), 20.0)
    if stage in {"prior-d3-e1-event-window-extraction", "window-dedup-and-storage-estimate", "pilot-window-selection"}:
        return 0.5
    if stage in {"pilot-data-qc", "one-minute-impact-replay-sample"}:
        return 1.0 if download else 0.2
    return 0.2


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


def validate_no_protected(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for col in cols:
        if col in df.columns and len(df):
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            if ts.ge(FINAL_HOLDOUT_START).any():
                raise RuntimeError(f"protected timestamp found in {col}")


def to_ms(ts: Any) -> int:
    t = pd.Timestamp(pd.to_datetime(ts, utc=True))
    return int(t.timestamp() * 1000)


def from_ms(value: Any) -> pd.Timestamp:
    return pd.to_datetime(int(value), unit="ms", utc=True)


def path_time_inventory(root: Path) -> dict[str, Any]:
    exists = root.exists()
    files = sorted(root.glob("*.parquet")) if exists else []
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None
    rows = 0
    sample_schema: list[str] = []
    for p in files[:20]:
        try:
            if pq is not None:
                pf = pq.ParquetFile(p)
                rows += int(pf.metadata.num_rows)
                names = list(pf.schema_arrow.names)
                if not sample_schema:
                    sample_schema = names
                if "timestamp" in names:
                    idx = names.index("timestamp")
                    for rg in range(pf.metadata.num_row_groups):
                        st = pf.metadata.row_group(rg).column(idx).statistics
                        if st and st.min is not None and st.max is not None:
                            a = pd.to_datetime(st.min, utc=True)
                            b = pd.to_datetime(st.max, utc=True)
                            min_ts = a if min_ts is None else min(min_ts, a)
                            max_ts = b if max_ts is None else max(max_ts, b)
            else:
                df = pd.read_parquet(p, columns=["timestamp"])
                rows += len(df)
                a = pd.to_datetime(df["timestamp"].min(), utc=True)
                b = pd.to_datetime(df["timestamp"].max(), utc=True)
                min_ts = a if min_ts is None else min(min_ts, a)
                max_ts = b if max_ts is None else max(max_ts, b)
        except Exception:
            continue
    return {
        "path": str(root),
        "exists": exists,
        "file_count": len(files),
        "sampled_file_count_for_timestamps": min(len(files), 20),
        "sampled_rows": rows,
        "min_timestamp_sampled": str(min_ts) if min_ts is not None else "",
        "max_timestamp_sampled": str(max_ts) if max_ts is not None else "",
        "size_bytes": sum(p.stat().st_size for p in files) if exists else 0,
        "schema_sample": ";".join(sample_schema),
    }


def bybit_get(session: requests.Session, endpoint: str, params: Mapping[str, Any], retries: int = 4) -> dict[str, Any]:
    url = BYBIT_BASE_URL + endpoint
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=dict(params), timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(min(8.0, 1.0 + attempt))
                continue
            r.raise_for_status()
            data = r.json()
            ret = int(data.get("retCode", -1))
            if ret != 0:
                if ret in RATE_LIMIT_RET_CODES:
                    time.sleep(min(8.0, 1.0 + attempt))
                    continue
                raise RuntimeError(f"retCode={data.get('retCode')} retMsg={data.get('retMsg')}")
            return data.get("result", {}) or {}
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 + attempt * 0.5)
    raise RuntimeError(f"Bybit request failed {endpoint} params={dict(params)} err={last_err}")


def normalize_kline_rows(rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(columns))
    df = pd.DataFrame(rows, columns=list(columns))
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    for c in columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def normalize_oi_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    out = []
    for r in rows:
        ts = r.get("timestamp") or r.get("time")
        oi = r.get("openInterest") or r.get("open_interest")
        if ts is None or oi is None:
            continue
        out.append({"timestamp": from_ms(ts), "open_interest": pd.to_numeric(oi, errors="coerce")})
    return pd.DataFrame(out).dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last") if out else pd.DataFrame(columns=["timestamp", "open_interest"])


def normalize_funding_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    out = []
    for r in rows:
        ts = r.get("fundingRateTimestamp") or r.get("timestamp")
        rate = r.get("fundingRate") or r.get("funding_rate")
        if ts is None or rate is None:
            continue
        out.append({"timestamp": from_ms(ts), "funding_rate": pd.to_numeric(rate, errors="coerce")})
    return pd.DataFrame(out).dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last") if out else pd.DataFrame(columns=["timestamp", "funding_rate"])


def fetch_kline_dataset(session: requests.Session, dataset: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp, limit: int = 1000) -> tuple[pd.DataFrame, int]:
    spec = DATASETS[dataset]
    start = pd.Timestamp(start).floor("min")
    cursor_end = pd.Timestamp(end).ceil("min")
    all_frames = []
    requests_made = 0
    # Bybit returns newest-first rows for bounded kline requests. Page backwards by end timestamp
    # so the earliest part of a long event window is not silently dropped.
    while cursor_end >= start:
        params = {"category": "linear", "symbol": symbol, "interval": spec["interval"], "start": to_ms(start), "end": to_ms(cursor_end), "limit": limit}
        result = bybit_get(session, spec["endpoint"], params)
        requests_made += 1
        rows = result.get("list", []) or []
        if not rows:
            break
        frame = normalize_kline_rows(rows, spec["columns"])
        if frame.empty:
            break
        all_frames.append(frame)
        earliest = pd.Timestamp(frame["timestamp"].min())
        if earliest <= start:
            break
        next_end = earliest - pd.Timedelta(minutes=1)
        if next_end >= cursor_end:
            break
        cursor_end = next_end
        if requests_made > 10000:
            raise RuntimeError("download pagination safety stop")
        time.sleep(0.03)
    if not all_frames:
        return pd.DataFrame(columns=spec["columns"]), requests_made
    df = pd.concat(all_frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)
    return df, requests_made


def fetch_open_interest(session: requests.Session, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    rows_all: list[Mapping[str, Any]] = []
    cursor = None
    requests_made = 0
    for _ in range(100):
        params: dict[str, Any] = {"category": "linear", "symbol": symbol, "intervalTime": "5min", "startTime": to_ms(start), "endTime": to_ms(end), "limit": 200}
        if cursor:
            params["cursor"] = cursor
        result = bybit_get(session, "/v5/market/open-interest", params)
        requests_made += 1
        rows = result.get("list", []) or []
        rows_all.extend(rows)
        cursor = result.get("nextPageCursor") or ""
        if not cursor or not rows:
            break
        time.sleep(0.03)
    df = normalize_oi_rows(rows_all)
    if not df.empty:
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)
    return df, requests_made


def fetch_funding(session: requests.Session, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    rows_all: list[Mapping[str, Any]] = []
    cur_end = to_ms(end)
    requests_made = 0
    for _ in range(100):
        result = bybit_get(session, "/v5/market/funding/history", {"category": "linear", "symbol": symbol, "endTime": cur_end, "limit": 200})
        requests_made += 1
        rows = result.get("list", []) or []
        if not rows:
            break
        rows_all.extend(rows)
        ts_vals = [int(r.get("fundingRateTimestamp")) for r in rows if r.get("fundingRateTimestamp") is not None]
        if not ts_vals:
            break
        oldest = min(ts_vals)
        if oldest <= to_ms(start):
            break
        cur_end = oldest - 1
        time.sleep(0.03)
    df = normalize_funding_rows(rows_all)
    if not df.empty:
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)
    return df, requests_made


def month_key(ts: Any) -> str:
    return pd.Timestamp(pd.to_datetime(ts, utc=True)).strftime("%Y-%m")


def window_hash(row: Mapping[str, Any]) -> str:
    key = "|".join(str(row.get(k, "")) for k in ["window_type", "family", "event_id", "symbol", "window_start", "window_end"])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def read_prior_path_metrics() -> pd.DataFrame:
    p = PRIOR_ROOT / "path_diagnostics/path_metrics.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing prior path metrics: {p}")
    return pd.read_parquet(p)



def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def stage_preflight(ctx: RunContext) -> None:
    snap = resource_snapshot(REPO)
    inv = [path_time_inventory(p) for p in [DATA_1M_HOT, DATA_1M, DATA_5M, DATA_CONTEXT]]
    write_csv(ctx.run_root / "preflight/local_1m_inventory.csv", inv)
    prior_files = [
        PRIOR_ROOT / "QLMG_PATH_DIAGNOSTICS_EXIT_SURFACE_REPORT.md",
        PRIOR_ROOT / "path_diagnostics/path_summary_by_family.csv",
        PRIOR_ROOT / "matched_null/matched_null_summary.csv",
        PRIOR_ROOT / "events/all_event_ledger.parquet",
        PRIOR_ROOT / "path_diagnostics/path_metrics.parquet",
        PRIOR_ROOT / "triage/family_triage_summary.csv",
        PRIOR_ROOT / "seal/protected_slice_check.json",
    ]
    prior_rows = []
    for p in prior_files:
        prior_rows.append({"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0, "sha256_first_100mb": sha256_file(p, 100 * 1024 * 1024) if p.exists() and p.is_file() else ""})
    write_json(ctx.run_root / "preflight/prior_artifact_manifest.json", {"prior_root": str(PRIOR_ROOT), "artifacts": prior_rows})
    helper_hits = shell(["bash", "-lc", "rg -n 'market/kline|mark-price-kline|index-price-kline|premium-index-price-kline|open-interest|funding/history' pull.py pull5.py enrich_trades_funding_bybit.py tools 2>/dev/null | head -80"], timeout=30)
    env_flags = {k: bool(os.getenv(k)) for k in ["BYBIT_API_KEY", "BYBIT_API_SECRET", "BYBIT_BASE_URL", "TG_BOT_TOKEN", "TELEGRAM_BOT_TOKEN", "TG_CHAT_ID", "TELEGRAM_CHAT_ID"]}
    write_text(ctx.run_root / "preflight/api_capability_report.md", f"# API Capability Preflight\n\n- public Bybit base URL: `{BYBIT_BASE_URL}`\n- public market endpoints require credentials: `false`\n- local Bybit helper references found: `{bool(helper_hits.strip())}`\n- env presence flags only, no secrets printed: `{env_flags}`\n\n```text\n{helper_hits[:4000]}\n```\n")
    write_text(ctx.run_root / "preflight/resource_guard_report.md", f"# Resource Guard Report\n\n- free disk GB: `{snap.free_gb:.2f}`\n- hard stop: `<5GB`\n- warning: `<7GB`\n- stage output hard stop: `>20GB` unless `--allow-large-output`\n- pilot cap GB: `{ctx.args.pilot_download_cap_gb}`\n")
    write_text(ctx.run_root / "preflight/preflight_report.md", f"# Preflight Report\n\n- run root: `{ctx.run_root}`\n- current free disk GB: `{snap.free_gb:.2f}`\n- /opt/parquet/1m_hot files: `{inv[0]['file_count']}`\n- /opt/parquet/1m files: `{inv[1]['file_count']}`\n- prior path diagnostic root: `{PRIOR_ROOT}`\n- protected holdout cutoff: `{FINAL_HOLDOUT_START}`\n- no final holdout download/selection permitted.\n")


def stage_telegram_tmux(ctx: RunContext) -> None:
    missing = ctx.notifier.missing or "none_detected"
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness Report\n\n- status: `{ctx.notifier.status}`\n- missing/disabled reason: `{missing}`\n- local log: `notifications/telegram_events.jsonl`\n- secrets persisted: `false`\n")
    watch = f"""# Watch Commands\n\n- `tmux attach -t {ctx.args.tmux_session_name}`\n- `tail -f {ctx.run_root}/logs/full_run.log`\n- `watch -n 30 'cat {ctx.run_root}/watch_status.json'`\n- `tail -f {ctx.run_root}/notifications/telegram_events.jsonl`\n- `df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h`\n"""
    write_text(ctx.run_root / "tmux/watch_commands.md", watch)
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux Instructions\n\nRun: `bash tools/run_qlmg_targeted_1m_data_pilot_tmux.sh --tmux-session-name {ctx.args.tmux_session_name} --stage all --resume --max-events-per-family {ctx.args.max_events_per_family} --nulls-per-event {ctx.args.nulls_per_event} {'--download' if ctx.args.download else '--no-download'} --pilot-download-cap-gb {ctx.args.pilot_download_cap_gb}`\n")
    ctx.notifier.send("QLMG 1M PILOT START", f"run_root={ctx.run_root}")


def stage_seal(ctx: RunContext) -> None:
    pre_ok = ctx.end < FINAL_HOLDOUT_START
    protected_blocked = True
    protected_attempt = {"window_start": str(FINAL_HOLDOUT_START), "window_end": str(FINAL_HOLDOUT_START + pd.Timedelta(hours=1)), "blocked": True}
    check = {"final_holdout_start": str(FINAL_HOLDOUT_START), "allowed_end_inclusive": str(SCREENING_END), "requested_start": str(ctx.start), "requested_end": str(ctx.end), "pre_holdout_selection_passes": bool(pre_ok), "protected_selection_smoke": protected_attempt, "status": "pass" if pre_ok and protected_blocked else "fail"}
    if check["status"] != "pass":
        raise RuntimeError("seal guard failed")
    write_json(ctx.run_root / "seal/protected_slice_check.json", check)
    write_text(ctx.run_root / "seal/seal_guard_report.md", f"# Seal Guard Report\n\n- protected slice: `{FINAL_HOLDOUT_START}` onward\n- selected/downloadable windows must end before: `{SCREENING_END}`\n- protected smoke selection: `blocked`\n- status: `pass`\n")


def build_windows_from_path_metrics(df: pd.DataFrame, smoke: bool = False) -> pd.DataFrame:
    keep = df[df["family"].isin(["D3", "E1"])].copy()
    if smoke:
        keep = keep.groupby("family", group_keys=False).head(12)
    rows: list[dict[str, Any]] = []
    for _, r in keep.iterrows():
        decision_ts = pd.Timestamp(r["decision_ts"])
        anchor = decision_ts
        start = anchor - pd.Timedelta(hours=4)
        end = decision_ts + pd.Timedelta(hours=24)
        if end >= FINAL_HOLDOUT_START:
            continue
        family = str(r["family"])
        rows.append({
            "window_type": "event",
            "family": family,
            "event_id": r.get("event_id", ""),
            "symbol": str(r["symbol"]),
            "window_start": start,
            "window_end": end,
            "decision_ts": decision_ts,
            "shock_ts": "",
            "source_run_root": str(PRIOR_ROOT),
            "reason_for_selection": f"prior_{family}_path_diagnostic_event",
            "prior_24h_mfe_bps": r.get("24h_mfe_bps", np.nan),
            "prior_24h_mae_bps": r.get("24h_mae_bps", np.nan),
            "prior_pos1R_before_neg1R_24h": r.get("24h_pos1R_before_neg1R", np.nan),
            "prior_matched_null_uplift": "available_in_summary_not_per_event",
            "liquidation_proxy_flag": bool(r.get("24h_liquidation_10x", False)),
            "data_quality_flags": r.get("data_quality_flags", ""),
        })
    # D4-like diagnostic subset, not a new strategy family.
    d4 = keep[((pd.to_numeric(keep.get("oi_chg_24h"), errors="coerce") <= 0) | (pd.to_numeric(keep.get("funding_rate"), errors="coerce") <= 0)) & (pd.to_numeric(keep.get("24h_mfe_bps"), errors="coerce") > pd.to_numeric(keep.get("24h_mae_bps"), errors="coerce"))]
    if smoke:
        d4 = d4.head(8)
    for _, r in d4.iterrows():
        decision_ts = pd.Timestamp(r["decision_ts"])
        start = decision_ts - pd.Timedelta(hours=4)
        end = decision_ts + pd.Timedelta(hours=24)
        if end >= FINAL_HOLDOUT_START:
            continue
        rows.append({
            "window_type": "event",
            "family": "D4_like_inferred",
            "event_id": f"d4_{r.get('event_id','')}",
            "symbol": str(r["symbol"]),
            "window_start": start,
            "window_end": end,
            "decision_ts": decision_ts,
            "shock_ts": "",
            "source_run_root": str(PRIOR_ROOT),
            "reason_for_selection": "diagnostic_inferred_from_D3_E1_price_oi_funding_reset_proxy",
            "prior_24h_mfe_bps": r.get("24h_mfe_bps", np.nan),
            "prior_24h_mae_bps": r.get("24h_mae_bps", np.nan),
            "prior_pos1R_before_neg1R_24h": r.get("24h_pos1R_before_neg1R", np.nan),
            "prior_matched_null_uplift": "not_a_strategy_family",
            "liquidation_proxy_flag": bool(r.get("24h_liquidation_10x", False)),
            "data_quality_flags": str(r.get("data_quality_flags", "")) + ";D4_like_inferred_not_strategy",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["family", "symbol", "decision_ts"]).reset_index(drop=True)
        validate_no_protected(out, ["window_start", "window_end", "decision_ts"])
    return out


def stage_extract_windows(ctx: RunContext) -> None:
    df = read_prior_path_metrics()
    df = df[(pd.to_datetime(df["decision_ts"], utc=True) >= ctx.start) & (pd.to_datetime(df["decision_ts"], utc=True) <= ctx.end)]
    if ctx.args.max_symbols:
        syms = sorted(df["symbol"].astype(str).unique())[: ctx.args.max_symbols]
        df = df[df["symbol"].isin(syms)]
    windows = build_windows_from_path_metrics(df, smoke=ctx.args.smoke)
    write_csv(ctx.run_root / "windows/raw_event_windows.csv", windows.to_dict("records"))
    counts = windows.groupby("family").size().to_dict() if not windows.empty else {}
    write_text(ctx.run_root / "windows/window_extraction_report.md", f"# Window Extraction Report\n\n- prior root: `{PRIOR_ROOT}`\n- families extracted: `D3,E1,D4_like_inferred`\n- raw windows: `{len(windows)}`\n- counts by family: `{counts}`\n- D4-like rows are inferred diagnostic windows only and cannot create strategy conclusions.\n- protected windows rejected: `yes`\n")


def dedupe_windows(df: pd.DataFrame, merge_gap: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = []
    df = df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)
    for sym, sub in df.sort_values(["symbol", "window_start", "window_end"]).groupby("symbol"):
        cur: dict[str, Any] | None = None
        families: set[str] = set()
        event_ids: list[str] = []
        for _, r in sub.iterrows():
            if cur is None:
                cur = {"symbol": sym, "window_start": r["window_start"], "window_end": r["window_end"], "source_event_count": 1}
                families = {str(r["family"])}
                event_ids = [str(r.get("event_id", ""))]
                continue
            if r["window_start"] <= cur["window_end"] + merge_gap:
                cur["window_end"] = max(cur["window_end"], r["window_end"])
                cur["source_event_count"] = int(cur["source_event_count"]) + 1
                families.add(str(r["family"]))
                event_ids.append(str(r.get("event_id", "")))
            else:
                cur["families"] = ";".join(sorted(families))
                cur["source_event_ids_sample"] = ";".join(event_ids[:20])
                out.append(cur)
                cur = {"symbol": sym, "window_start": r["window_start"], "window_end": r["window_end"], "source_event_count": 1}
                families = {str(r["family"])}
                event_ids = [str(r.get("event_id", ""))]
        if cur is not None:
            cur["families"] = ";".join(sorted(families))
            cur["source_event_ids_sample"] = ";".join(event_ids[:20])
            out.append(cur)
    res = pd.DataFrame(out)
    if not res.empty:
        res["window_hours"] = (pd.to_datetime(res["window_end"], utc=True) - pd.to_datetime(res["window_start"], utc=True)).dt.total_seconds() / 3600.0
        validate_no_protected(res, ["window_start", "window_end"])
    return res


def estimate_storage(deduped: pd.DataFrame) -> list[dict[str, Any]]:
    total_minutes = float((pd.to_datetime(deduped["window_end"], utc=True) - pd.to_datetime(deduped["window_start"], utc=True)).dt.total_seconds().sum() / 60.0) if not deduped.empty else 0.0
    specs = [
        ("ohlcv_1m", 7, 55),
        ("mark_1m", 5, 40),
        ("index_1m", 5, 40),
        ("premium_1m", 5, 40),
        ("open_interest_5m", 2, 20),
        ("funding_history", 2, 8),
        ("public_trades_optional", 0, 0),
        ("top_of_book_optional", 0, 0),
    ]
    rows = []
    for dataset, cols, bytes_per_row in specs:
        if dataset == "open_interest_5m":
            row_count = total_minutes / 5.0
        elif dataset == "funding_history":
            row_count = max(1.0, total_minutes / (8.0 * 60.0)) if total_minutes else 0.0
        elif bytes_per_row == 0:
            row_count = 0.0
        else:
            row_count = total_minutes
        est_bytes = row_count * bytes_per_row
        rows.append({"dataset": dataset, "estimated_rows": int(row_count), "estimated_compressed_bytes": int(est_bytes), "estimated_compressed_gb": est_bytes / (1024**3), "columns_or_fields": cols, "source_status": "planned" if bytes_per_row else "not_available_in_this_pilot"})
    rows.append({"dataset": "total_core", "estimated_rows": int(total_minutes * 4 + total_minutes / 5.0), "estimated_compressed_bytes": int(sum(r["estimated_compressed_bytes"] for r in rows if r["dataset"] not in {"public_trades_optional", "top_of_book_optional"})), "estimated_compressed_gb": sum(float(r["estimated_compressed_gb"]) for r in rows if r["dataset"] not in {"public_trades_optional", "top_of_book_optional"}), "columns_or_fields": "", "source_status": "estimated"})
    return rows


def stage_dedup_storage(ctx: RunContext) -> None:
    raw = pd.read_csv(ctx.run_root / "windows/raw_event_windows.csv") if (ctx.run_root / "windows/raw_event_windows.csv").exists() else pd.DataFrame()
    dedup = dedupe_windows(raw)
    write_csv(ctx.run_root / "windows/deduped_windows.csv", dedup.to_dict("records"))
    storage = estimate_storage(dedup)
    write_csv(ctx.run_root / "windows/storage_estimate.csv", storage)
    total_gb = next((r["estimated_compressed_gb"] for r in storage if r["dataset"] == "total_core"), 0.0)
    raw_count = len(raw)
    dedup_count = len(dedup)
    hours = float(dedup.get("window_hours", pd.Series(dtype=float)).sum()) if not dedup.empty else 0.0
    write_text(ctx.run_root / "windows/storage_estimate_report.md", f"# Storage Estimate Report\n\n- raw window count: `{raw_count}`\n- deduped symbol-window count: `{dedup_count}`\n- total symbol-hours: `{hours:.2f}`\n- estimated core compressed GB: `{float(total_gb):.4f}`\n- fits 5GB cap: `{float(total_gb) <= 5.0}`\n- fits 20GB cap: `{float(total_gb) <= 20.0}`\n- public trades/top-of-book: `not available from local source in this pilot`\n")


def capability_sample_windows(ctx: RunContext) -> pd.DataFrame:
    dedup_path = ctx.run_root / "windows/deduped_windows.csv"
    if dedup_path.exists():
        dedup = pd.read_csv(dedup_path)
        if not dedup.empty:
            return dedup.head(2).copy()
    return pd.DataFrame([
        {"symbol": "BTCUSDT", "window_start": pd.Timestamp("2025-01-01T00:00:00Z"), "window_end": pd.Timestamp("2025-01-01T00:15:00Z"), "families": "capability", "source_event_count": 0},
        {"symbol": "ETHUSDT", "window_start": pd.Timestamp("2025-01-01T00:00:00Z"), "window_end": pd.Timestamp("2025-01-01T00:15:00Z"), "families": "capability", "source_event_count": 0},
    ])


def stage_source_capability(ctx: RunContext) -> None:
    windows = capability_sample_windows(ctx)
    session = requests.Session()
    session.headers["User-Agent"] = "qlmg-targeted-1m-pilot/1.0"
    rows = []
    schemas = []
    for _, w in windows.iterrows():
        symbol = str(w["symbol"]).upper()
        start = pd.Timestamp(pd.to_datetime(w["window_start"], utc=True))
        end = min(pd.Timestamp(pd.to_datetime(w["window_end"], utc=True)), start + pd.Timedelta(minutes=15))
        for dataset in list(DATASETS) + list(OPTIONAL_DATASETS):
            status = "ok"
            error = ""
            nrows = 0
            reqs = 0
            cols: list[str] = []
            try:
                if dataset in DATASETS:
                    df, reqs = fetch_kline_dataset(session, dataset, symbol, start, end, limit=5)
                elif dataset == "open_interest_5m":
                    df, reqs = fetch_open_interest(session, symbol, start, end)
                else:
                    df, reqs = fetch_funding(session, symbol, start, end)
                nrows = len(df)
                cols = list(df.columns)
                if nrows == 0:
                    status = "empty"
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                df = pd.DataFrame()
            rows.append({"symbol": symbol, "dataset": dataset, "endpoint": (DATASETS.get(dataset) or OPTIONAL_DATASETS.get(dataset) or {}).get("endpoint", ""), "status": status, "rows": nrows, "requests": reqs, "error": error})
            schemas.append({"symbol": symbol, "dataset": dataset, "columns": ";".join(cols), "row_count": nrows, "status": status})
    write_csv(ctx.run_root / "source_capability/sample_download_manifest.csv", rows)
    write_csv(ctx.run_root / "source_capability/sample_schema_report.md.tmp.csv", schemas)
    write_text(ctx.run_root / "source_capability/sample_schema_report.md", "# Sample Schema Report\n\n" + pd.DataFrame(schemas).to_markdown(index=False))
    ok_core = all(any(r["dataset"] == ds and r["status"] in {"ok", "empty"} for r in rows) for ds in DATASETS)
    write_text(ctx.run_root / "source_capability/source_capability_report.md", f"# Source Capability Report\n\n- Bybit public kline endpoint tested: `yes`\n- Bybit mark-price kline endpoint tested: `yes`\n- Bybit index-price kline endpoint tested: `yes`\n- Bybit premium-index kline endpoint tested: `yes`\n- OI/funding verification endpoints tested: `yes`\n- core dataset API reachable: `{ok_core}`\n- credentials required: `false`\n- secrets logged: `false`\n- public trade/top-of-book source: `not found locally; not downloaded`\n")


def select_family_pilot(raw: pd.DataFrame, family: str, limit: int, seed: int) -> pd.DataFrame:
    sub = raw[raw["family"] == family].copy()
    if sub.empty or limit <= 0:
        return sub.head(0)
    sub["month"] = pd.to_datetime(sub["decision_ts"], utc=True).dt.strftime("%Y-%m")
    sub["path_strength"] = pd.to_numeric(sub.get("prior_24h_mfe_bps"), errors="coerce") - pd.to_numeric(sub.get("prior_24h_mae_bps"), errors="coerce")
    # Keep both strong and weak prior cases while spreading across symbols/months.
    strong = sub.sort_values("path_strength", ascending=False).groupby(["symbol", "month"], group_keys=False).head(1)
    weak = sub.sort_values("path_strength", ascending=True).groupby(["symbol", "month"], group_keys=False).head(1)
    cand = pd.concat([strong, weak], ignore_index=True).drop_duplicates("event_id")
    if len(cand) < limit:
        rest = sub.drop(index=cand.index, errors="ignore")
        cand = pd.concat([cand, rest], ignore_index=True).drop_duplicates("event_id")
    cand = cand.sample(frac=1.0, random_state=seed).head(limit).sort_values(["family", "symbol", "decision_ts"])
    return cand


def make_null_window_for_event(row: pd.Series, ordinal: int, seed: int) -> dict[str, Any]:
    start = pd.Timestamp(pd.to_datetime(row["window_start"], utc=True))
    end = pd.Timestamp(pd.to_datetime(row["window_end"], utc=True))
    # Deterministic offsets around event month. Keep pre-holdout and avoid exact event window.
    offsets = [pd.Timedelta(days=7), pd.Timedelta(days=-7), pd.Timedelta(days=14), pd.Timedelta(days=-14)]
    off = offsets[(ordinal + seed) % len(offsets)]
    ns = start + off
    ne = end + off
    if ne >= FINAL_HOLDOUT_START:
        ns = start - abs(off)
        ne = end - abs(off)
    if ns < pd.Timestamp("2020-01-01T00:00:00Z"):
        ns = start + pd.Timedelta(days=3)
        ne = end + pd.Timedelta(days=3)
    return {
        "window_type": "matched_null",
        "family": row["family"],
        "event_id": f"null{ordinal}_{row['event_id']}",
        "source_event_id": row["event_id"],
        "symbol": row["symbol"],
        "window_start": ns,
        "window_end": ne,
        "decision_ts": ns + (pd.Timestamp(pd.to_datetime(row["decision_ts"], utc=True)) - start),
        "reason_for_selection": "deterministic_same_symbol_offset_null_window",
        "match_level": "same_symbol_offset_window",
    }


def stage_pilot_selection(ctx: RunContext) -> None:
    raw = pd.read_csv(ctx.run_root / "windows/raw_event_windows.csv") if (ctx.run_root / "windows/raw_event_windows.csv").exists() else pd.DataFrame()
    max_base = 20 if ctx.args.smoke else int(ctx.args.max_events_per_family)
    d3_lim = min(max_base, 300)
    e1_lim = min(max_base, 300)
    d4_lim = min(max_base // 2 if not ctx.args.smoke else 10, 50 if max_base <= 100 else 300)
    chosen = []
    for fam, lim in [("D3", d3_lim), ("E1", e1_lim), ("D4_like_inferred", d4_lim)]:
        part = select_family_pilot(raw, fam, lim, ctx.args.seed)
        chosen.append(part)
    events = pd.concat([p for p in chosen if not p.empty], ignore_index=True) if chosen else pd.DataFrame()
    events["window_type"] = "event" if not events.empty else []
    null_rows: list[dict[str, Any]] = []
    if not events.empty:
        for _, r in events.iterrows():
            for i in range(max(0, int(ctx.args.nulls_per_event))):
                null_rows.append(make_null_window_for_event(r, i, ctx.args.seed))
    nulls = pd.DataFrame(null_rows)
    pilot = pd.concat([events, nulls], ignore_index=True, sort=False) if not nulls.empty else events
    if not pilot.empty:
        validate_no_protected(pilot, ["window_start", "window_end", "decision_ts"])
        pilot["window_id"] = [window_hash(r) for r in pilot.to_dict("records")]
        pilot = pilot.sort_values(["window_type", "family", "symbol", "window_start"]).reset_index(drop=True)
    write_csv(ctx.run_root / "pilot/pilot_windows.csv", pilot.to_dict("records"))
    counts = pilot.groupby(["window_type", "family"]).size().to_dict() if not pilot.empty else {}
    write_text(ctx.run_root / "pilot/pilot_selection_report.md", f"# Pilot Selection Report\n\n- max events per family: `{ctx.args.max_events_per_family}`\n- nulls per event: `{ctx.args.nulls_per_event}`\n- selected windows: `{len(pilot)}`\n- counts: `{counts}`\n- selection includes strong and weak prior path cases, plus liquidation-proxy flags where present.\n")


def write_partition(df: pd.DataFrame, base: Path, dataset: str, symbol: str, window_id: str) -> Path:
    out_dir = base / dataset / f"symbol={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"window={window_id}.parquet"
    df.to_parquet(p, index=False, compression="zstd")
    return p


def pilot_estimated_gb(run_root: Path) -> float:
    pilot = pd.read_csv(run_root / "pilot/pilot_windows.csv") if (run_root / "pilot/pilot_windows.csv").exists() else pd.DataFrame()
    if pilot.empty:
        return 0.0
    ded = dedupe_windows(pilot.rename(columns={"window_type": "_window_type"})) if "window_start" in pilot.columns else pd.DataFrame()
    storage = estimate_storage(ded) if not ded.empty else []
    return float(next((r["estimated_compressed_gb"] for r in storage if r["dataset"] == "total_core"), 0.0))


def stage_download(ctx: RunContext) -> None:
    pilot = pd.read_csv(ctx.run_root / "pilot/pilot_windows.csv") if (ctx.run_root / "pilot/pilot_windows.csv").exists() else pd.DataFrame()
    download_root = ctx.run_root / "downloaded_1m"
    download_root.mkdir(parents=True, exist_ok=True)
    manifest_path = download_root / "download_manifest.csv"
    failure_path = download_root / "gaps_and_failures.csv"
    manifest_fields = ["window_id", "window_type", "family", "symbol", "dataset", "endpoint", "status", "rows", "requests", "path", "error"]
    failure_fields = ["window_id", "symbol", "dataset", "status", "error", "reason", "estimated_gb", "cap_gb"]
    if ctx.args.no_download or not ctx.args.download:
        write_csv(manifest_path, [], fieldnames=manifest_fields)
        write_csv(failure_path, [], fieldnames=failure_fields)
        write_text(download_root / "download_report.md", "# Download Report\n\n- mode: `no_download`\n- downloaded datasets: `none`\n- no external historical data was downloaded.\n")
        return
    est_gb = pilot_estimated_gb(ctx.run_root)
    if est_gb > float(ctx.args.pilot_download_cap_gb) and not ctx.args.allow_large_output:
        write_csv(manifest_path, [], fieldnames=manifest_fields)
        write_csv(failure_path, [{"reason": "estimated_pilot_size_above_cap", "estimated_gb": est_gb, "cap_gb": ctx.args.pilot_download_cap_gb}], fieldnames=failure_fields)
        write_text(download_root / "download_report.md", f"# Download Report\n\n- mode: `blocked_storage_budget`\n- estimated GB: `{est_gb:.4f}`\n- cap GB: `{ctx.args.pilot_download_cap_gb}`\n")
        return
    session = requests.Session()
    session.headers["User-Agent"] = "qlmg-targeted-1m-pilot/1.0"
    existing = set()
    if manifest_path.exists() and ctx.args.resume:
        try:
            old = pd.read_csv(manifest_path)
            existing = set(old[old.get("status") == "ok"].apply(lambda r: f"{r['window_id']}|{r['dataset']}", axis=1)) if not old.empty else set()
        except Exception:
            existing = set()
    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    req_count = 0
    for n, row in enumerate(pilot.to_dict("records"), start=1):
        symbol = str(row["symbol"]).upper()
        window_id = str(row.get("window_id") or window_hash(row))
        start = pd.Timestamp(pd.to_datetime(row["window_start"], utc=True))
        end = pd.Timestamp(pd.to_datetime(row["window_end"], utc=True))
        if end >= FINAL_HOLDOUT_START:
            failures.append({"window_id": window_id, "symbol": symbol, "dataset": "all", "status": "blocked_protected_window"})
            continue
        for dataset in list(DATASETS) + list(OPTIONAL_DATASETS):
            key = f"{window_id}|{dataset}"
            if key in existing:
                continue
            status = "ok"
            err = ""
            out_path = ""
            rows = 0
            reqs = 0
            try:
                if dataset in DATASETS:
                    df, reqs = fetch_kline_dataset(session, dataset, symbol, start, end)
                    out_dataset = DATASETS[dataset]["out_dir"]
                elif dataset == "open_interest_5m":
                    df, reqs = fetch_open_interest(session, symbol, start, end)
                    out_dataset = OPTIONAL_DATASETS[dataset]["out_dir"]
                else:
                    df, reqs = fetch_funding(session, symbol, start, end)
                    out_dataset = OPTIONAL_DATASETS[dataset]["out_dir"]
                req_count += reqs
                if not df.empty:
                    validate_no_protected(df, ["timestamp"])
                    p = write_partition(df, download_root, out_dataset, symbol, window_id)
                    out_path = str(p.relative_to(ctx.run_root))
                    rows = len(df)
                else:
                    status = "empty"
            except Exception as exc:
                status = "error"
                err = f"{type(exc).__name__}: {exc}"
                failures.append({"window_id": window_id, "symbol": symbol, "dataset": dataset, "status": status, "error": err})
            manifests.append({"window_id": window_id, "window_type": row.get("window_type", ""), "family": row.get("family", ""), "symbol": symbol, "dataset": dataset, "endpoint": (DATASETS.get(dataset) or OPTIONAL_DATASETS.get(dataset) or {}).get("endpoint", ""), "status": status, "rows": rows, "requests": reqs, "path": out_path, "error": err})
        if n % max(1, ctx.args.chunk_size) == 0 or n == len(pilot):
            write_csv(manifest_path, manifests, fieldnames=manifest_fields)
            write_csv(failure_path, failures, fieldnames=failure_fields)
            ctx.notifier.send("1M PILOT DOWNLOAD PROGRESS", f"windows_done={n}/{len(pilot)} requests={req_count}")
    write_csv(manifest_path, manifests, fieldnames=manifest_fields)
    write_csv(failure_path, failures, fieldnames=failure_fields)
    ok = sum(1 for r in manifests if r["status"] == "ok")
    errors = sum(1 for r in manifests if r["status"] == "error")
    write_text(download_root / "download_report.md", f"# Download Report\n\n- mode: `download`\n- windows attempted: `{len(pilot)}`\n- dataset-window rows in manifest: `{len(manifests)}`\n- successful non-empty downloads: `{ok}`\n- errors: `{errors}`\n- requests made: `{req_count}`\n- data stored under this run root only.\n")


def qc_dataset_file(path: Path, dataset: str, symbol: str, window_id: str, expected_rows: int | None = None) -> dict[str, Any]:
    try:
        df = pd.read_parquet(path)
        if "timestamp" not in df.columns:
            return {"dataset": dataset, "symbol": symbol, "window_id": window_id, "path": str(path), "status": "missing_timestamp", "rows": len(df), "expected_rows": expected_rows or 0, "coverage_ratio": 0.0}
        ts = pd.to_datetime(df["timestamp"], utc=True)
        dupes = int(ts.duplicated().sum())
        gaps = 0
        if len(ts) > 1 and dataset.endswith("1m"):
            gaps = int((ts.sort_values().diff().dropna() > pd.Timedelta(minutes=1)).sum())
        bad_price = 0
        # Premium index values can legitimately be negative. Only price-like datasets require positive OHLC.
        if "premium" not in dataset and "funding" not in dataset and "open_interest" not in dataset:
            for c in ["open", "high", "low", "close"]:
                if c in df.columns:
                    bad_price += int((pd.to_numeric(df[c], errors="coerce") <= 0).sum())
        coverage = float(len(df) / expected_rows) if expected_rows else np.nan
        return {"dataset": dataset, "symbol": symbol, "window_id": window_id, "path": str(path), "status": "ok", "rows": len(df), "expected_rows": expected_rows or 0, "coverage_ratio": coverage, "min_ts": str(ts.min()), "max_ts": str(ts.max()), "duplicates": dupes, "gap_count": gaps, "nonpositive_price_count": bad_price}
    except Exception as exc:
        return {"dataset": dataset, "symbol": symbol, "window_id": window_id, "path": str(path), "status": "error", "error": f"{type(exc).__name__}: {exc}", "expected_rows": expected_rows or 0, "coverage_ratio": 0.0}


def stage_qc(ctx: RunContext) -> None:
    download_root = ctx.run_root / "downloaded_1m"
    pilot = read_csv_safe(ctx.run_root / "pilot/pilot_windows.csv")
    expected: dict[str, dict[str, int]] = {}
    if not pilot.empty:
        for _, r in pilot.iterrows():
            wid = str(r.get("window_id"))
            start = pd.Timestamp(pd.to_datetime(r["window_start"], utc=True))
            end = pd.Timestamp(pd.to_datetime(r["window_end"], utc=True))
            mins = max(0, int((end - start).total_seconds() // 60) + 1)
            expected[wid] = {
                "bybit_linear_ohlcv_1m": mins,
                "bybit_linear_mark_1m": mins,
                "bybit_linear_index_1m": mins,
                "bybit_linear_premium_1m": mins,
                "bybit_linear_open_interest_5m": max(1, mins // 5),
                "bybit_linear_funding_history": 0,
            }
    rows: list[dict[str, Any]] = []
    for p in download_root.rglob("*.parquet"):
        dataset = p.parts[-3] if len(p.parts) >= 3 else p.parent.parent.name
        symbol = p.parent.name.replace("symbol=", "")
        window_id = p.stem.replace("window=", "")
        rows.append(qc_dataset_file(p, dataset, symbol, window_id, expected.get(window_id, {}).get(dataset)))
    write_csv(ctx.run_root / "qc/pilot_coverage_summary.csv", rows)
    gap_rows = [r for r in rows if int(r.get("gap_count", 0) or 0) > 0 or int(r.get("duplicates", 0) or 0) > 0 or int(r.get("nonpositive_price_count", 0) or 0) > 0 or r.get("status") != "ok" or (r.get("expected_rows", 0) and float(r.get("coverage_ratio", 0) or 0) < 0.95)]
    gap_fields = list(rows[0].keys()) if rows else ["dataset", "symbol", "window_id", "path", "status", "rows", "expected_rows", "coverage_ratio", "min_ts", "max_ts", "duplicates", "gap_count", "nonpositive_price_count", "error"]
    write_csv(ctx.run_root / "qc/pilot_gap_summary.csv", gap_rows, fieldnames=gap_fields)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    full_cov = sum(1 for r in rows if r.get("expected_rows", 0) and float(r.get("coverage_ratio", 0) or 0) >= 0.95)
    write_text(ctx.run_root / "qc/pilot_data_qc_report.md", f"# Pilot Data QC Report\n\n- parquet files checked: `{len(rows)}`\n- ok files: `{ok}`\n- files with >=95% expected coverage: `{full_cov}`\n- files with gaps/duplicates/bad values/errors/low coverage: `{len(gap_rows)}`\n- premium index values may be negative and are not treated as bad prices.\n- no protected timestamps allowed by runner before write.\n")


def read_download_for_window(run_root: Path, dataset_dir: str, symbol: str, window_id: str) -> pd.DataFrame:
    p = run_root / "downloaded_1m" / dataset_dir / f"symbol={symbol}" / f"window={window_id}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def compute_1m_path_impact(row: Mapping[str, Any], ohlcv: pd.DataFrame, mark: pd.DataFrame) -> dict[str, Any]:
    if ohlcv.empty:
        return {"status": "missing_ohlcv_1m"}
    ohlcv = ohlcv.sort_values("timestamp")
    decision_ts = pd.Timestamp(pd.to_datetime(row.get("decision_ts"), utc=True)) if row.get("decision_ts") else pd.Timestamp(pd.to_datetime(row["window_start"], utc=True)) + pd.Timedelta(hours=4)
    future = ohlcv[pd.to_datetime(ohlcv["timestamp"], utc=True) >= decision_ts]
    if future.empty:
        return {"status": "no_future_1m_after_decision"}
    entry = float(pd.to_numeric(future.iloc[0].get("open"), errors="coerce"))
    if not np.isfinite(entry) or entry <= 0:
        return {"status": "bad_entry_price"}
    high = pd.to_numeric(future["high"], errors="coerce").max()
    low = pd.to_numeric(future["low"], errors="coerce").min()
    family = str(row.get("family", ""))
    side = "long"  # D3/E1/D4-like pilot focus is long flush/bounce behavior.
    mfe = (high / entry - 1.0) * 10000.0 if side == "long" else (1.0 - low / entry) * 10000.0
    mae = (1.0 - low / entry) * 10000.0 if side == "long" else (high / entry - 1.0) * 10000.0
    mark_rows = len(mark) if not mark.empty else 0
    mark_liq_proxy = np.nan
    if not mark.empty:
        mark_low = pd.to_numeric(mark.get("low"), errors="coerce").min()
        mark_liq_proxy = (1.0 - mark_low / entry) * 10000.0
    return {"status": "ok", "family": family, "one_m_mfe_bps": mfe, "one_m_mae_bps": mae, "one_m_path_rows": len(future), "mark_rows": mark_rows, "mark_adverse_bps_proxy": mark_liq_proxy, "material_same_bar_resolution_needed": bool(mae > 0 and mfe > 0)}


def stage_impact(ctx: RunContext) -> None:
    pilot = pd.read_csv(ctx.run_root / "pilot/pilot_windows.csv") if (ctx.run_root / "pilot/pilot_windows.csv").exists() else pd.DataFrame()
    manifest = read_csv_safe(ctx.run_root / "downloaded_1m/download_manifest.csv")
    rows = []
    if not pilot.empty and not manifest.empty:
        events = pilot[pilot.get("window_type") == "event"].head(500)
        for _, r in events.iterrows():
            symbol = str(r["symbol"]).upper()
            window_id = str(r["window_id"])
            ohlcv = read_download_for_window(ctx.run_root, DATASETS["ohlcv_1m"]["out_dir"], symbol, window_id)
            mark = read_download_for_window(ctx.run_root, DATASETS["mark_1m"]["out_dir"], symbol, window_id)
            rec = {"window_id": window_id, "event_id": r.get("event_id", ""), "family": r.get("family", ""), "symbol": symbol, "prior_24h_mfe_bps": r.get("prior_24h_mfe_bps", np.nan), "prior_24h_mae_bps": r.get("prior_24h_mae_bps", np.nan)}
            rec.update(compute_1m_path_impact(r, ohlcv, mark))
            rows.append(rec)
    write_csv(ctx.run_root / "impact/one_minute_impact_summary.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    med_mfe = float(pd.to_numeric(pd.Series([r.get("one_m_mfe_bps") for r in ok]), errors="coerce").median()) if ok else np.nan
    med_mae = float(pd.to_numeric(pd.Series([r.get("one_m_mae_bps") for r in ok]), errors="coerce").median()) if ok else np.nan
    material_share = np.nan
    if ok:
        prior = pd.to_numeric(pd.Series([r.get("prior_24h_mfe_bps") for r in ok]), errors="coerce")
        one = pd.to_numeric(pd.Series([r.get("one_m_mfe_bps") for r in ok]), errors="coerce")
        if prior.notna().any():
            material_share = float(((one - prior).abs() > 250).mean())
    material_text = "yes" if np.isfinite(material_share) and material_share > 0.25 else "no_or_unresolved"
    write_text(ctx.run_root / "impact/one_minute_impact_report.md", f"# 1m Impact Replay Sample\n\n- event windows with 1m impact rows: `{len(rows)}`\n- ok rows: `{len(ok)}`\n- median 1m MFE bps: `{med_mfe}`\n- median 1m MAE bps: `{med_mae}`\n- material MFE difference share vs prior 24h path: `{material_share}`\n- 1m materially changes conclusions: `{material_text}`\n- same-bar ambiguity assessed from 1m path availability, not from 5m-only assumptions.\n")


def stage_decision(ctx: RunContext) -> None:
    download_manifest = read_csv_safe(ctx.run_root / "downloaded_1m/download_manifest.csv")
    failures = read_csv_safe(ctx.run_root / "downloaded_1m/gaps_and_failures.csv")
    qc = read_csv_safe(ctx.run_root / "qc/pilot_coverage_summary.csv")
    impact = read_csv_safe(ctx.run_root / "impact/one_minute_impact_summary.csv")
    storage = read_csv_safe(ctx.run_root / "windows/storage_estimate.csv")
    total_gb = float(storage[storage.get("dataset") == "total_core"].iloc[0].get("estimated_compressed_gb", 0.0)) if not storage.empty and len(storage[storage.get("dataset") == "total_core"]) else 0.0
    downloaded_ok = bool(len(download_manifest) and (download_manifest.get("status") == "ok").any())
    core_errors = int((download_manifest.get("status") == "error").sum()) if not download_manifest.empty and "status" in download_manifest.columns else 0
    if not qc.empty and "dataset" in qc.columns:
        core_qc = qc[qc["dataset"].isin(["bybit_linear_ohlcv_1m", "bybit_linear_mark_1m", "bybit_linear_index_1m", "bybit_linear_premium_1m"])]
        qc_ok = bool(len(core_qc) and (core_qc.get("status") == "ok").all() and (pd.to_numeric(core_qc.get("coverage_ratio"), errors="coerce").fillna(1.0) >= 0.95).all() and (pd.to_numeric(core_qc.get("gap_count"), errors="coerce").fillna(0) == 0).all())
    else:
        qc_ok = False
    impact_ok = bool(len(impact) and (impact.get("status") == "ok").any())
    material = False
    if impact_ok:
        one_m = pd.to_numeric(impact.get("one_m_mfe_bps"), errors="coerce")
        prior = pd.to_numeric(impact.get("prior_24h_mfe_bps"), errors="coerce")
        material = bool(((one_m - prior).abs() > 250).mean() > 0.25) if prior.notna().any() else True
    if total_gb > ctx.args.pilot_download_cap_gb and not ctx.args.allow_large_output:
        verdict = "blocked_storage_budget"
    elif ctx.args.no_download or not ctx.args.download:
        verdict = "do_not_download_more_1m_now"
    elif not downloaded_ok and core_errors:
        verdict = "blocked_api_or_source_unavailable"
    elif downloaded_ok and qc_ok and material:
        verdict = "expand_targeted_1m_acquisition"
    elif downloaded_ok and qc_ok and impact_ok:
        verdict = "pilot_enough_continue_with_d3_d4_e1_validation"
    elif downloaded_ok and not qc_ok:
        verdict = "blocked_api_or_source_unavailable"
    else:
        verdict = "do_not_download_more_1m_now"
    decision = {"verdict": verdict, "allowed_verdicts": sorted(ALLOWED_VERDICTS), "final_holdout_untouched": True, "download_mode": bool(ctx.args.download), "estimated_core_gb": total_gb, "downloaded_ok": downloaded_ok, "qc_ok": qc_ok, "impact_ok": impact_ok, "one_minute_material_change_detected": material, "core_errors": core_errors}
    write_json(ctx.run_root / "decision_summary.json", decision)
    write_text(ctx.run_root / "FULL_1M_ACQUISITION_DECISION.md", f"# Full 1m Acquisition Decision\n\n## Verdict\n`{verdict}`\n\n## Evidence\n- final holdout untouched: `true`\n- download mode: `{bool(ctx.args.download)}`\n- estimated core GB: `{total_gb:.4f}`\n- downloaded ok: `{downloaded_ok}`\n- QC ok: `{qc_ok}`\n- impact rows ok: `{impact_ok}`\n- material 1m change detected: `{material}`\n- core API errors: `{core_errors}`\n\nThis is an acquisition decision only. It does not validate D3/E1/D4-like strategy logic.\n")


def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rels = [
        "FULL_1M_ACQUISITION_DECISION.md", "decision_summary.json", "preflight/preflight_report.md", "preflight/resource_guard_report.md", "preflight/local_1m_inventory.csv", "preflight/api_capability_report.md", "notifications/telegram_readiness_report.md", "tmux/watch_commands.md", "seal/seal_guard_report.md", "windows/window_extraction_report.md", "windows/storage_estimate.csv", "windows/storage_estimate_report.md", "pilot/pilot_selection_report.md", "pilot/pilot_windows.csv", "source_capability/source_capability_report.md", "source_capability/sample_download_manifest.csv", "source_capability/sample_schema_report.md", "downloaded_1m/download_manifest.csv", "downloaded_1m/download_report.md", "downloaded_1m/gaps_and_failures.csv", "qc/pilot_data_qc_report.md", "qc/pilot_coverage_summary.csv", "qc/pilot_gap_summary.csv", "impact/one_minute_impact_summary.csv", "impact/one_minute_impact_report.md",
    ]
    idx = []
    for rel in rels:
        src = ctx.run_root / rel
        if src.exists():
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            idx.append({"artifact": rel, "bundle_copy": str(dst.relative_to(ctx.run_root)), "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_path_index.csv", idx)
    write_json(bundle / "artifact_path_index.json", {"artifacts": idx})
    write_text(bundle / "README.md", "Compact targeted 1m pilot review bundle. Downloaded datasets are intentionally excluded; use artifact index paths.\n")
    zip_path = ctx.run_root / "qlmg_targeted_1m_data_pilot_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in bundle.rglob("*"):
            if item.is_file():
                z.write(item, arcname=str(item.relative_to(ctx.run_root)))



def download_dependent_stage(stage: str) -> bool:
    return stage in {"pilot-download-if-safe", "pilot-data-qc", "one-minute-impact-replay-sample", "full-acquisition-decision", "compact-review-bundle"}


def completed_in_no_download_mode(run_root: Path) -> bool:
    p = run_root / "downloaded_1m/download_report.md"
    if not p.exists():
        return False
    try:
        return "mode: `no_download`" in p.read_text(encoding="utf-8")
    except Exception:
        return False


def run_stage(ctx: RunContext, stage: str) -> None:
    if stage == "preflight-resource-and-api-audit":
        stage_preflight(ctx)
    elif stage == "telegram-and-tmux-setup":
        stage_telegram_tmux(ctx)
    elif stage == "seal-guard":
        stage_seal(ctx)
    elif stage == "prior-d3-e1-event-window-extraction":
        stage_extract_windows(ctx)
    elif stage == "window-dedup-and-storage-estimate":
        stage_dedup_storage(ctx)
    elif stage == "bybit-1m-source-capability-audit":
        stage_source_capability(ctx)
    elif stage == "pilot-window-selection":
        stage_pilot_selection(ctx)
    elif stage == "pilot-download-if-safe":
        stage_download(ctx)
    elif stage == "pilot-data-qc":
        stage_qc(ctx)
    elif stage == "one-minute-impact-replay-sample":
        stage_impact(ctx)
    elif stage == "full-acquisition-decision":
        stage_decision(ctx)
    elif stage == "compact-review-bundle":
        stage_bundle(ctx)
    else:
        raise ValueError(stage)


def main() -> int:
    args = parse_args()
    if not args.download and not args.no_download:
        args.no_download = True
    run_root, reason = resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    start, end = clamp_window(args)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": reason, "argv": sys.argv, "start": str(start), "end": str(end), "download": bool(args.download), "seed": args.seed, "created_at_utc": utc_now()})
    stages = stage_list(args.stage)
    try:
        for stage in stages:
            if args.resume and stage_complete(run_root, stage):
                if args.download and download_dependent_stage(stage) and completed_in_no_download_mode(run_root):
                    notifier.send("1M PILOT STAGE RERUN", f"{stage} previous checkpoint was no-download mode")
                else:
                    notifier.send("1M PILOT STAGE SKIP", stage)
                    continue
            notifier.send("1M PILOT STAGE START", stage)
            append_command(run_root, stage)
            ensure_guard(ctx, stage, estimate_stage_gb(stage, args.smoke, args.download, args.pilot_download_cap_gb))
            if args.dry_run:
                write_text(run_root / "dry_run" / f"{stage}.md", f"Would run stage `{stage}`.\n")
            else:
                run_stage(ctx, stage)
            mark_done(run_root, stage)
            notifier.send("1M PILOT STAGE DONE", stage)
        notifier.send("1M PILOT COMPLETE", f"run_root={run_root}")
        return 0
    except Exception as exc:
        notifier.send("1M PILOT FAILED", f"{type(exc).__name__}: {exc}", level="error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
