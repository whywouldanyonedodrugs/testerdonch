#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import time
import urllib.error
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

from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402

try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

RESULTS_ROOT = REPO / "results/rebaseline"
DEFAULT_RUN_ID = "phase_kraken_k0_data_foundation_20260630_v1"
DEFAULT_SEED = 20260630
PERSISTENT_ROOT = Path("/opt/parquet/kraken_derivatives")
MECHANICAL_QA_ROOT = RESULTS_ROOT / "phase_qlmg_mechanical_qa_evidence_contract_20260630_v1_20260630_074328"
NO_VENDOR_ROOT = RESULTS_ROOT / "phase_qlmg_no_vendor_progress_run_20260630_v1_20260630_082124"
MANUAL_CANDIDATES = [REPO / "docs/QLMG_BACKTESTING_MANUAL_20260630_FULL.md", REPO / "research_inputs/testmanual.txt"]
PROTECTED_STRATEGY_SELECTION_TS = pd.Timestamp("2026-01-01T00:00:00Z")
DERIV_BASE = "https://futures.kraken.com"
CHART_BASE = "https://futures.kraken.com"
USER_AGENT = "Donch-QLMG-Kraken-K0/1.0 public-only"

STAGES = (
    "preflight-and-source-freeze",
    "telegram-and-tmux-setup",
    "kraken-official-endpoint-probe",
    "kraken-schema-contracts",
    "kraken-storage-and-download-plan",
    "kraken-official-data-download",
    "kraken-data-qc",
    "kraken-instrument-master-and-lifecycle",
    "kraken-bar-mark-funding-analytics-panel",
    "kraken-universe-and-liquidity-tiers",
    "kraken-bybit-portability-matrix",
    "kraken-strategy-readiness-matrix",
    "kraken-live-capture-spec",
    "kraken-backtest-engine-gap-report",
    "decision-report",
    "compact-review-bundle",
    "all",
)

ALLOWED_NEXT_DECISIONS = {
    "run_kraken_k1_strategy_screen_next",
    "continue_kraken_official_download_next",
    "build_kraken_live_capture_next",
    "build_kraken_backtest_engine_next",
    "blocked_by_kraken_public_data_access",
    "blocked_by_protocol_issue",
}

NO_VENDOR_CLASSES = {
    "kraken_progress_with_official_data",
    "kraken_needs_live_capture_substitute",
    "kraken_redesign_to_less_depth_sensitive",
    "kraken_candidate_library_only",
    "kraken_discard_current_translation_no_vendor_path",
}

PRIVATE_OR_ORDER_PATH_TOKENS = ("/private", "/sendorder", "/cancel", "/editorder", "/batchorder", "/accounts", "/fills", "/openpositions")

ENDPOINT_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "instruments": [
        {"url": f"{DERIV_BASE}/derivatives/api/v3/instruments", "params": {}},
    ],
    "tickers": [
        {"url": f"{DERIV_BASE}/derivatives/api/v3/tickers", "params": {}},
    ],
    "historical_funding": [
        {"url": f"{DERIV_BASE}/derivatives/api/v3/historicalfundingrates", "params": {"symbol": "PF_XBTUSD"}},
        {"url": f"{DERIV_BASE}/derivatives/api/v3/historical-funding-rates", "params": {"symbol": "PF_XBTUSD"}},
    ],
    "candles_trade_1m": [
        {"url": f"{CHART_BASE}/api/charts/v1/trade/PF_XBTUSD/1m", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/candles/PF_XBTUSD/1m", "params": {}},
    ],
    "mark_candles_1m": [
        {"url": f"{CHART_BASE}/api/charts/v1/mark/PF_XBTUSD/1m", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/mark-price/PF_XBTUSD/1m", "params": {}},
    ],
    "analytics_open_interest": [
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/open-interest", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/open-interest/PF_XBTUSD", "params": {}},
    ],
    "analytics_funding": [
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/funding", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/funding/PF_XBTUSD", "params": {}},
    ],
    "analytics_liquidation_volume": [
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/liquidation-volume", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/liquidation-volume/PF_XBTUSD", "params": {}},
    ],
    "analytics_spreads": [
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/spreads", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/spreads/PF_XBTUSD", "params": {}},
    ],
    "analytics_liquidity": [
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/liquidity", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/liquidity/PF_XBTUSD", "params": {}},
    ],
    "analytics_slippage": [
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/slippage", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/slippage/PF_XBTUSD", "params": {}},
    ],
    "analytics_trade_volume": [
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/trade-volume", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/trade-volume/PF_XBTUSD", "params": {}},
    ],
    "analytics_trade_count": [
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/PF_XBTUSD/trade-count", "params": {}},
        {"url": f"{CHART_BASE}/api/charts/v1/analytics/trade-count/PF_XBTUSD", "params": {}},
    ],
    "public_execution_events": [
        {"url": f"{DERIV_BASE}/derivatives/api/v3/history", "params": {"symbol": "PF_XBTUSD"}},
        {"url": f"{DERIV_BASE}/derivatives/api/v3/tradehistory", "params": {"symbol": "PF_XBTUSD"}},
        {"url": f"{DERIV_BASE}/derivatives/api/v3/public-executions", "params": {"symbol": "PF_XBTUSD"}},
    ],
    "public_order_events": [
        {"url": f"{DERIV_BASE}/derivatives/api/v3/public-order-events", "params": {"symbol": "PF_XBTUSD"}},
        {"url": f"{DERIV_BASE}/derivatives/api/v3/orderbook", "params": {"symbol": "PF_XBTUSD"}},
    ],
    "recent_trade_history": [
        {"url": f"{DERIV_BASE}/derivatives/api/v3/trades", "params": {"symbol": "PF_XBTUSD"}},
        {"url": f"{DERIV_BASE}/derivatives/api/v3/history", "params": {"symbol": "PF_XBTUSD"}},
    ],
}

DATA_FAMILY_MAP = {
    "candles": ["candles_trade_1m"],
    "mark": ["mark_candles_1m"],
    "historical_funding": ["historical_funding"],
    "analytics_open_interest": ["analytics_open_interest"],
    "analytics_funding": ["analytics_funding"],
    "analytics_liquidation_volume": ["analytics_liquidation_volume"],
    "analytics_orderbook_spreads_liquidity_slippage": ["public_order_events", "analytics_spreads", "analytics_liquidity", "analytics_slippage"],
    "trade_volume_count": ["analytics_trade_volume", "analytics_trade_count"],
    "public_execution_events": ["public_execution_events"],
    "public_order_events": ["public_order_events"],
    "recent_trade_history": ["recent_trade_history"],
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
                self.remote = TelegramNotifier.from_args(_Args(), run_label="kraken-k0-foundation")
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
    p = argparse.ArgumentParser(description="Kraken K0 official-data foundation")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--max-output-gb", type=float, default=40.0)
    p.add_argument("--allow-large-output", action="store_true")
    p.add_argument("--disable-telegram", action="store_true")
    p.add_argument("--require-telegram", action="store_true")
    p.add_argument("--allow-no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--download-official-data", action="store_true")
    p.add_argument("--download-cap-gb", type=float, default=25.0)
    p.add_argument("--include-candles", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-funding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-mark-events", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-analytics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-public-executions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-public-order-events", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-lifecycle-status", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--historical-bar-backfill", action="store_true")
    p.add_argument("--backfill-start", default="2023-01-01")
    p.add_argument("--backfill-end", default="")
    p.add_argument("--backfill-resolution", default="5m")
    p.add_argument("--backfill-chunk-hours", type=int, default=120)
    p.add_argument("--backfill-max-symbols", type=int, default=0)
    p.add_argument("--backfill-symbols", default="")
    p.add_argument("--backfill-perpetual-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--backfill-progress-every", type=int, default=500)
    p.add_argument("--backfill-telegram-every", type=int, default=1000)
    p.add_argument("--backfill-include-trade-candles", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--backfill-include-mark-candles", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tmux-session-name", default="kraken_k0_foundation")
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
    if args.smoke:
        return (base / "smoke").resolve(), "smoke_subroot"
    if base.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return base.with_name(f"{base.name}_{suffix}"), f"default_root_existed_suffix_{suffix}"
    return base, "default_root_available"

def parse_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True) if args.end else pd.Timestamp.now(tz="UTC")
    if end < start:
        raise RuntimeError("end before start")
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
        writer.writerows(rows_list)

def done_path(ctx: RunContext, stage: str) -> Path:
    return ctx.run_root / "stage_status" / f"{stage}.done"

def mark_done(ctx: RunContext, stage: str) -> None:
    p = done_path(ctx, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"stage": stage, "completed_utc": utc_now()}, sort_keys=True) + "\n")

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stable_hash(value: str, n: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]

def is_public_safe_url(url: str) -> bool:
    low = url.lower()
    return low.startswith("https://") and not any(tok in low for tok in PRIVATE_OR_ORDER_PATH_TOKENS)

def public_http_get(url: str, params: Mapping[str, Any] | None = None, timeout: float = 15.0) -> tuple[int, bytes, dict[str, str], str]:
    if not is_public_safe_url(url):
        raise RuntimeError(f"blocked private/order/non-https endpoint: {url}")
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
    full_url = url + (("?" + query) if query else "")
    req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/csv,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(2_000_000)
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return int(resp.status), body, headers, full_url
    except urllib.error.HTTPError as exc:
        body = exc.read(512_000)
        headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        return int(exc.code), body, headers, full_url
    except Exception as exc:
        return 0, str(exc).encode("utf-8", errors="replace"), {}, full_url

def json_loads_maybe(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except Exception:
        return None

def flatten_records(obj: Any) -> list[dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x if isinstance(x, dict) else {"value": x} for x in obj]
    if isinstance(obj, dict):
        for key in ["instruments", "tickers", "rates", "history", "candles", "data", "result"]:
            val = obj.get(key)
            if isinstance(val, list):
                return [x if isinstance(x, dict) else {"value": x} for x in val]
            if isinstance(val, dict):
                nested = flatten_records(val)
                if nested:
                    return nested
        if obj:
            return [obj]
    return []

def infer_schema(obj: Any) -> dict[str, str]:
    recs = flatten_records(obj)
    if not recs:
        return {}
    keys: dict[str, str] = {}
    for r in recs[:20]:
        for k, v in r.items():
            keys.setdefault(str(k), type(v).__name__)
    return keys

def timestamp_range_from_records(records: list[dict[str, Any]]) -> tuple[str, str, str]:
    ts_cols = ["time", "timestamp", "date", "datetime", "effectiveTime", "lastUpdateTime", "last"]
    vals = []
    unit = "unknown"
    for r in records:
        for c in ts_cols:
            if c in r:
                vals.append(r[c])
                break
    parsed = []
    for v in vals[:10000]:
        if isinstance(v, (int, float)):
            unit = "ms" if float(v) > 10_000_000_000 else "s"
            parsed.append(pd.to_datetime(v, unit=unit, utc=True, errors="coerce"))
        else:
            parsed.append(pd.to_datetime(v, utc=True, errors="coerce"))
    clean = [p for p in parsed if not pd.isna(p)]
    if not clean:
        return "", "", unit
    return str(min(clean)), str(max(clean)), unit

def safe_symbol_from_instrument(row: Mapping[str, Any]) -> str:
    for k in ["symbol", "instrument", "pair", "name"]:
        v = row.get(k)
        if v:
            return str(v)
    return ""

def parse_kraken_symbol(symbol: str) -> dict[str, Any]:
    s = str(symbol)
    parts = s.split("_")
    typ = parts[0] if parts else ""
    tail = parts[1] if len(parts) > 1 else s
    base = tail[:-3] if len(tail) > 3 and tail.endswith("USD") else tail
    quote = "USD" if tail.endswith("USD") else "unknown"
    return {"venue_symbol": s, "prefix": typ, "base_asset": "XBT" if base == "XBT" else base, "quote_asset": quote, "display_symbol": ("BTC" if base == "XBT" else base) + ("/" + quote if quote != "unknown" else "")}

def manual_path() -> Path | None:
    for p in MANUAL_CANDIDATES:
        if p.exists():
            return p
    return None

def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def read_json_or_empty(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def stage_preflight(ctx: RunContext) -> None:
    ctx.run_root.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = {}
    for name, path in {
        "mechanical_qa": MECHANICAL_QA_ROOT,
        "active_manual": manual_path() or Path("missing_manual"),
        "latest_bybit_no_vendor": NO_VENDOR_ROOT,
    }.items():
        exists = path.exists()
        rows.append({"name": name, "path": str(path), "exists": exists, "is_file": path.is_file(), "is_dir": path.is_dir()})
        if exists and path.is_file() and path.stat().st_size < 200_000_000:
            hashes[name] = file_sha256(path)
    # local Kraken reports, if any
    for p in sorted(REPO.rglob("*kraken*"))[:200]:
        if p.is_file() and "phase_kraken_k0_data_foundation" not in str(p):
            rows.append({"name": "local_kraken_source", "path": str(p), "exists": True, "is_file": True, "is_dir": False})
    write_csv(ctx.run_root / "preflight/input_artifact_manifest.csv", rows)
    write_json(ctx.run_root / "preflight/frozen_artifact_hashes.json", hashes)
    snap = resource_snapshot(ctx.run_root.parent)
    est = min(float(ctx.args.download_cap_gb), 35.0) if ctx.args.download_official_data else 2.0
    guard = check_resource_guard(snap, estimated_output_gb=est, hard_stage_output_gb=35.0, allow_large_output=ctx.args.allow_large_output)
    # Resource choice is intentional: the tmux wrapper uses nice/ionice because Bybit runs may coexist.
    guard["resource_choice"] = "nice_ionice_when_launched_by_tmux; proceed_if_disk_and_load_guards_pass"
    write_json(ctx.run_root / "preflight/resource_guard_report.md.json", guard)
    write_text(ctx.run_root / "preflight/resource_guard_report.md", "# Resource Guard\n\n" + json.dumps(guard, indent=2))
    if guard["status"] == "hard_stop":
        raise RuntimeError("resource guard hard stop: " + ";".join(guard["reasons"]))
    public_guard = {"kraken_api_keys_read": False, "private_endpoints_allowed": False, "order_endpoints_allowed": False, "public_only": True}
    write_json(ctx.run_root / "preflight/public_only_key_guard.json", public_guard)
    write_text(ctx.run_root / "preflight/preflight_report.md", "# Preflight\n\nKraken K0 uses official/free public endpoints only. It does not read Kraken API keys and does not call private or order endpoints.\n")

def stage_telegram(ctx: RunContext) -> None:
    write_json(ctx.run_root / "notifications/telegram_readiness.json", {"status": ctx.notifier.status, "remote_available": ctx.notifier.remote_available, "missing": ctx.notifier.missing})
    write_text(ctx.run_root / "notifications/telegram_readiness_report.md", f"# Telegram Readiness\n\nstatus: {ctx.notifier.status}\nremote_available: {ctx.notifier.remote_available}\n")
    watch = [
        f"tmux attach -t {ctx.args.tmux_session_name}",
        f"tail -f {ctx.run_root}/logs/full_run.log",
        f"watch -n 30 'cat {ctx.run_root}/watch_status.json'",
        f"tail -f {ctx.run_root}/notifications/telegram_events.jsonl",
        "df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h",
    ]
    write_text(ctx.run_root / "tmux/watch_commands.md", "# Watch Commands\n\n" + "\n".join(f"- `{x}`" for x in watch))
    write_text(ctx.run_root / "tmux/tmux_run_instructions.md", f"# Tmux\n\nDefault session: `{ctx.args.tmux_session_name}`. Wrapper runs with nice/ionice when available.\n")

def stage_probe(ctx: RunContext) -> None:
    pdir = ctx.run_root / "probes/schema_samples"
    pdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, candidates in ENDPOINT_CANDIDATES.items():
        best_status = -1
        chosen = None
        for idx, cand in enumerate(candidates):
            status, body, headers, full_url = public_http_get(cand["url"], cand.get("params", {}), timeout=10.0)
            obj = json_loads_maybe(body)
            records = flatten_records(obj)
            schema = infer_schema(obj)
            earliest, latest, ts_unit = timestamp_range_from_records(records)
            sample_name = f"{family}_{idx}_{status}.json"
            write_json(pdir / sample_name, {"url": full_url, "status": status, "headers": dict(list(headers.items())[:20]), "schema": schema, "sample": obj if isinstance(obj, (dict, list)) else body.decode('utf-8', errors='replace')[:2000]})
            row = {
                "endpoint_family": family,
                "url": full_url,
                "method": "GET",
                "http_status": status,
                "works": bool(status == 200 and (records or schema)),
                "schema_sample_path": str(Path("probes/schema_samples") / sample_name),
                "timestamp_units": ts_unit,
                "earliest_available_ts_sample": earliest,
                "latest_available_ts_sample": latest,
                "sample_row_count": len(records),
                "pagination": "unknown_probe_only",
                "rate_limit_behavior": headers.get("x-ratelimit-remaining", headers.get("retry-after", "not_visible")),
                "auth_required": status in {401, 403},
                "sufficiency": "K0" if status == 200 else "unavailable_or_support_only",
            }
            rows.append(row)
            if status == 200 and (records or schema):
                chosen = row
                break
            best_status = max(best_status, status)
        if chosen is None and candidates:
            pass
    df = pd.DataFrame(rows)
    write_csv(ctx.run_root / "probes/endpoint_capability_matrix.csv", df)
    good = int(df.get("works", pd.Series(dtype=bool)).sum()) if not df.empty else 0
    write_text(ctx.run_root / "probes/endpoint_probe_report.md", f"# Endpoint Probe Report\n\nEndpoint candidates probed: {len(df)}. Working public endpoints: {good}. Live probe behavior is implementation source of truth; documentation discrepancies are retained in the matrix.\n")

def stage_contracts(ctx: RunContext) -> None:
    cdir = ctx.run_root / "contracts"
    cdir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "kraken_instrument_master_schema.yaml": ["venue", "venue_symbol", "display_symbol", "base_asset", "quote_asset", "settlement_currency", "contract_type", "tick_size", "min_order_size", "contract_size", "opening_date", "status", "margin_class", "max_leverage", "source_confidence"],
        "kraken_market_data_schema.yaml": ["venue_symbol", "ts", "open", "high", "low", "close", "volume", "source_endpoint", "timestamp_unit"],
        "kraken_funding_schema.yaml": ["venue_symbol", "funding_ts", "funding_rate", "period_hours", "funding_per_hour", "funding_8h_equiv", "source_endpoint"],
        "kraken_mark_schema.yaml": ["venue_symbol", "ts", "mark_price", "index_price", "trigger_basis", "source_endpoint"],
        "kraken_analytics_schema.yaml": ["venue_symbol", "ts", "analytics_type", "raw_value", "normalized_value", "source_endpoint"],
        "kraken_lifecycle_schema.yaml": ["venue_symbol", "interval_start", "interval_end", "status", "source_endpoint", "confidence"],
    }
    for name, fields in schemas.items():
        write_text(cdir / name, "fields:\n" + "\n".join(f"  - {f}" for f in fields))
    write_text(cdir / "kraken_symbol_mapping_contract.md", "# Kraken Symbol Mapping Contract\n\nUse `venue_symbol` exactly as Kraken returns it, e.g. `PF_XBTUSD`. `display_symbol` is derived only for human reports. Never map Bybit symbols directly into Kraken routes.\n")
    write_text(cdir / "kraken_backtest_mechanics_contract.md", "# Kraken Backtest Mechanics Contract\n\nKraken is not Bybit. Contracts must explicitly model venue_symbol, display_symbol, base/quote, settlement currency, collateral/PnL currency, Multi-M collateral uncertainty, collateral haircuts, margin class, max leverage, one-way net position per maturity, funding accrual/settlement behavior, Last/Mark/Index trigger basis, price-protected IOC/partial-fill market semantics, fee schedule uncertainty, and liquidation/closeout/assignment/unwind fields where public data allow. Unknowns are caps, not defaults.\n")

def chosen_endpoints() -> dict[str, dict[str, Any]]:
    # Placeholder; live chosen endpoints are read from matrix by stages. Kept for tests and defaults.
    return {fam: cands[0] for fam, cands in ENDPOINT_CANDIDATES.items() if cands}

def load_probe_matrix(ctx: RunContext) -> pd.DataFrame:
    return read_csv_or_empty(ctx.run_root / "probes/endpoint_capability_matrix.csv")

def selected_symbols(ctx: RunContext) -> list[str]:
    inst = read_json_or_empty(ctx.run_root / "probes/schema_samples/instruments_0_200.json")
    recs = flatten_records(inst.get("sample") if isinstance(inst, dict) else inst)
    symbols = [safe_symbol_from_instrument(r) for r in recs]
    symbols = [s for s in symbols if s]
    if not symbols:
        symbols = ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_ADAUSD"]
    # Prefer perpetual futures prefix if available while preserving venue order.
    symbols = list(dict.fromkeys([s for s in symbols if s.startswith("PF_")] + symbols))
    if ctx.args.max_symbols and ctx.args.max_symbols > 0:
        symbols = symbols[: ctx.args.max_symbols]
    return symbols[:5] if ctx.args.smoke else symbols

def backfill_symbols(ctx: RunContext) -> list[str]:
    if ctx.args.backfill_symbols:
        symbols = [s.strip() for s in str(ctx.args.backfill_symbols).split(",") if s.strip()]
        return symbols
    symbols = selected_symbols(ctx)
    if ctx.args.backfill_perpetual_only:
        pf = [s for s in symbols if s.startswith("PF_")]
        if pf:
            symbols = pf
    if ctx.args.backfill_max_symbols and ctx.args.backfill_max_symbols > 0:
        symbols = symbols[: ctx.args.backfill_max_symbols]
    return symbols[: min(len(symbols), 2)] if ctx.args.smoke else symbols

def resolution_minutes(resolution: str) -> int:
    r = str(resolution).strip().lower()
    if r.endswith("m"):
        return max(1, int(r[:-1]))
    if r.endswith("h"):
        return max(1, int(r[:-1]) * 60)
    return 1

def effective_backfill_chunk_hours(ctx: RunContext) -> int:
    # Keep chunks below Kraken's 2,000 candle response window.
    max_hours = max(1, int((1900 * resolution_minutes(ctx.args.backfill_resolution)) / 60))
    return max(1, min(int(ctx.args.backfill_chunk_hours), max_hours))

def historical_chunk_key(dataset: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp, resolution: str) -> str:
    return f"{dataset}|{symbol}|{start.isoformat()}|{end.isoformat()}|{resolution}"

def load_completed_historical_chunks(ctx: RunContext) -> set[str]:
    keys: set[str] = set()
    for rel in ["download/historical_bar_backfill_progress.csv", "download/download_manifest.csv"]:
        df = read_csv_or_empty(ctx.run_root / rel)
        if df.empty:
            continue
        needed = {"dataset", "symbol", "chunk_start", "chunk_end", "resolution", "status"}
        if not needed.issubset(set(df.columns)):
            continue
        ok = df[df["status"].astype(str).eq("downloaded")]
        for _, r in ok.iterrows():
            keys.add(historical_chunk_key(str(r["dataset"]), str(r["symbol"]), pd.Timestamp(r["chunk_start"]), pd.Timestamp(r["chunk_end"]), str(r["resolution"])))
    return keys

def disk_status_line(path: Path) -> str:
    snap = resource_snapshot(path)
    if hasattr(snap, "free_bytes"):
        free = float(snap.free_bytes) / (1024**3)
        used = float(snap.used_bytes) / (1024**3)
        total = float(snap.total_bytes) / (1024**3)
    else:
        free = float(snap.get("free_disk_gb", 0.0) or 0.0)
        used = float(snap.get("used_disk_gb", 0.0) or 0.0)
        total = float(snap.get("total_disk_gb", 0.0) or 0.0)
    return f"disk_free_gb={free:.1f}; disk_used_gb={used:.1f}; disk_total_gb={total:.1f}"

def backfill_window(ctx: RunContext) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(ctx.args.backfill_start, tz="UTC")
    end = pd.Timestamp(ctx.args.backfill_end, tz="UTC") if ctx.args.backfill_end else pd.Timestamp.utcnow().tz_convert("UTC")
    if end <= start:
        raise RuntimeError("--backfill-end must be after --backfill-start")
    return start, end

def pre_holdout_seconds(start: pd.Timestamp, end: pd.Timestamp) -> float:
    usable_end = min(end, PROTECTED_STRATEGY_SELECTION_TS)
    return max(0.0, (usable_end - start).total_seconds())

def stage_storage(ctx: RunContext) -> None:
    matrix = load_probe_matrix(ctx)
    syms = selected_symbols(ctx)
    days = max(1, int(math.ceil((ctx.end - ctx.start).total_seconds() / 86400.0)))
    rows = []
    # Several official public endpoints are retention-limited or return a bounded recent window.
    # Estimate against the observed probe retention instead of pretending all routes provide
    # full requested history; retention limitations are reported separately.
    for dataset, per_symbol_daily_mb in [("instruments", 0.1), ("tickers", 0.1), ("funding", 0.02), ("candles", 1.5), ("mark", 1.0), ("analytics", 0.5), ("events", 0.2)]:
        mult = len(syms) if dataset not in {"instruments", "tickers"} else 1
        estimate_days = days
        if dataset in {"candles", "mark", "analytics", "events"}:
            estimate_days = min(days, 3)
        if dataset == "funding":
            estimate_days = min(days, 370)
        est_gb = per_symbol_daily_mb * estimate_days * mult / 1024.0
        rows.append({"dataset": dataset, "symbol_count": mult, "days": estimate_days, "requested_days": days, "estimated_gb": round(est_gb, 4), "staging_path": str(ctx.run_root / "downloaded_official_kraken" / dataset), "persistent_path": str(PERSISTENT_ROOT / ("parquet/" + dataset if dataset != "events" else "parquet/events"))})
    if ctx.args.historical_bar_backfill:
        bf_start, bf_end = backfill_window(ctx)
        bf_days = max(1, int(math.ceil((bf_end - bf_start).total_seconds() / 86400.0)))
        bf_symbols = backfill_symbols(ctx)
        resolution_factor = 1.0 if str(ctx.args.backfill_resolution) == "1m" else (1.0 / 5.0 if str(ctx.args.backfill_resolution) == "5m" else 1.0)
        # Empirical K0 sample was roughly 0.04-0.08 MB per symbol-day for
        # 1m trade+mark Parquet+raw. Scale 5m by row count.
        if ctx.args.backfill_include_trade_candles:
            rows.append({"dataset": f"historical_trade_candles_{ctx.args.backfill_resolution}", "symbol_count": len(bf_symbols), "days": bf_days, "requested_days": bf_days, "estimated_gb": round(0.05 * resolution_factor * bf_days * len(bf_symbols) / 1024.0, 4), "staging_path": str(ctx.run_root / "downloaded_official_kraken/historical_trade_candles"), "persistent_path": str(PERSISTENT_ROOT / "parquet/historical_trade_candles")})
        if ctx.args.backfill_include_mark_candles:
            rows.append({"dataset": f"historical_mark_candles_{ctx.args.backfill_resolution}", "symbol_count": len(bf_symbols), "days": bf_days, "requested_days": bf_days, "estimated_gb": round(0.08 * resolution_factor * bf_days * len(bf_symbols) / 1024.0, 4), "staging_path": str(ctx.run_root / "downloaded_official_kraken/historical_mark_candles"), "persistent_path": str(PERSISTENT_ROOT / "parquet/historical_mark_candles")})
    estimate = pd.DataFrame(rows)
    total = float(estimate["estimated_gb"].sum()) if not estimate.empty else 0.0
    write_csv(ctx.run_root / "download/storage_estimate.csv", estimate)
    write_text(ctx.run_root / "download/download_plan.md", f"# Kraken Download Plan\n\nEstimated output: {total:.3f} GB. Cap: {ctx.args.download_cap_gb:.3f} GB. Download enabled: {ctx.args.download_official_data}. Retention-limited endpoints do not block other datasets.\n")
    write_text(ctx.run_root / "download/persistent_store_plan.md", f"# Persistent Store Plan\n\nRoot: `{PERSISTENT_ROOT}`\n\nLayout: `raw/<dataset>/<symbol>/<chunk>.jsonl.gz`, `parquet/instruments/`, `parquet/candles/`, `parquet/funding/`, `parquet/analytics/`, `parquet/events/`, `manifests/`, `qc/`.\n")
    # Retention matrix from live probe matrix.
    ret_rows = []
    for data_family, endpoint_families in DATA_FAMILY_MAP.items():
        sub = matrix[matrix.get("endpoint_family", pd.Series(dtype=str)).isin(endpoint_families)] if not matrix.empty else pd.DataFrame()
        works = sub[sub.get("works", pd.Series(dtype=bool)).astype(bool)] if not sub.empty and "works" in sub.columns else pd.DataFrame()
        best = works.iloc[0].to_dict() if not works.empty else (sub.iloc[0].to_dict() if not sub.empty else {})
        status = int(best.get("http_status", 0) or 0) if best else 0
        endpoint_url = str(best.get("url", ""))
        if status != 200:
            ret_class = "unavailable_or_unprobed"
        elif "/orderbook" in endpoint_url:
            ret_class = "current_snapshot_only_not_historical_depth"
        elif data_family in {"recent_trade_history", "public_execution_events", "public_order_events"}:
            ret_class = "recent_only_or_retention_limited"
        elif data_family in {"candles", "mark"}:
            ret_class = "partial_historical_or_recent_window_sample"
        else:
            ret_class = "full_or_partial_historical"
        ret_rows.append({
            "data_family": data_family,
            "endpoint_url_used": best.get("url", ""),
            "authentication_required": bool(best.get("auth_required", False)),
            "earliest_timestamp_available": best.get("earliest_available_ts_sample", ""),
            "latest_timestamp_available": best.get("latest_available_ts_sample", ""),
            "symbol_count": len(syms),
            "row_count": best.get("sample_row_count", 0),
            "paging_behavior": best.get("pagination", "unknown"),
            "rate_limit_behavior": best.get("rate_limit_behavior", "not_visible"),
            "retention_classification": ret_class,
            "tier1_usability": data_family in {"candles", "historical_funding", "analytics_open_interest"} and status == 200,
            "tier2_3_usability": data_family in {"analytics_orderbook_spreads_liquidity_slippage", "public_execution_events", "public_order_events"} and status == 200,
            "family_support": "liquid_family_support" if data_family in {"candles", "historical_funding", "analytics_open_interest"} else "execution_sensitive_support_or_capture_needed",
        })
    write_csv(ctx.run_root / "download/kraken_retention_depth_matrix.csv", ret_rows)

    # Official support page says a historical funding export exists. K0 records
    # whether the public support/export path is reachable; it does not use private credentials.
    export_url = "https://support.kraken.com/articles/export-historical-funding-rates"
    status, body, headers, full_url = public_http_get(export_url, {}, timeout=10.0)
    export_rows = [{
        "url": full_url,
        "page_reachable": status == 200,
        "http_status": status,
        "manual_action_required": True,
        "expected_output_format": "CSV export per Kraken support article if available to operator",
        "would_improve_historical_funding_exactness": True,
        "automated_without_private_credentials": False,
    }]
    write_csv(ctx.run_root / "download/kraken_funding_export_audit.csv", export_rows)
    write_text(ctx.run_root / "download/kraken_funding_export_audit.md", "# Kraken Funding Export Audit\n\n" + json.dumps(export_rows[0], indent=2))
    write_json(ctx.run_root / "download/download_resume_state.json", {"download_enabled": ctx.args.download_official_data, "estimated_gb": total, "cap_gb": ctx.args.download_cap_gb, "status": "planned"})
    if ctx.args.download_official_data and total > ctx.args.download_cap_gb and not ctx.args.allow_large_output:
        raise RuntimeError(f"download estimate {total:.3f} GB exceeds cap {ctx.args.download_cap_gb:.3f} GB")

def raw_write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(payload)
    return file_sha256(path)

def json_to_parquet(path: Path, obj: Any, extra: Mapping[str, Any] | None = None) -> int:
    recs = flatten_records(obj)
    if not recs and isinstance(obj, dict):
        recs = [obj]
    if not recs:
        return 0
    df = pd.DataFrame(recs)
    for k, v in (extra or {}).items():
        df[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(df)

def working_url_for(ctx: RunContext, family: str) -> tuple[str, dict[str, Any]] | tuple[str, None]:
    matrix = load_probe_matrix(ctx)
    if matrix.empty:
        return "", None
    sub = matrix[(matrix["endpoint_family"].astype(str) == family) & (matrix["works"].astype(str).str.lower().isin(["true", "1"]))]
    if sub.empty:
        return "", None
    row = sub.iloc[0].to_dict()
    url = str(row.get("url", ""))
    parsed = urllib.parse.urlparse(url)
    params = dict(urllib.parse.parse_qsl(parsed.query))
    base = urllib.parse.urlunparse(parsed._replace(query=""))
    return base, params

def download_one(ctx: RunContext, dataset: str, family: str, symbol: str | None = None) -> dict[str, Any]:
    base, params = working_url_for(ctx, family)
    if not base:
        return {"dataset": dataset, "endpoint_family": family, "symbol": symbol or "", "status": "skipped_no_working_endpoint"}
    params = dict(params or {})
    if symbol:
        # Preserve endpoint-specific parameter if already present; replace path PF_XBTUSD for chart URLs.
        if "PF_XBTUSD" in base:
            base = base.replace("PF_XBTUSD", symbol)
        elif "symbol" in params:
            params["symbol"] = symbol
    status, body, headers, full_url = public_http_get(base, params, timeout=30.0)
    raw_path = ctx.run_root / "downloaded_official_kraken/raw" / dataset / (symbol or "all") / f"chunk_{stable_hash(full_url)}.jsonl.gz"
    raw_sha = raw_write(raw_path, body)
    obj = json_loads_maybe(body)
    parquet_path = ctx.run_root / "downloaded_official_kraken/parquet" / dataset / f"{symbol or 'all'}_{stable_hash(full_url)}.parquet"
    rows = json_to_parquet(parquet_path, obj, {"source_url": full_url, "venue_symbol": symbol or ""}) if status == 200 else 0
    parquet_sha = file_sha256(parquet_path) if parquet_path.exists() else ""
    return {"dataset": dataset, "endpoint_family": family, "symbol": symbol or "", "url": full_url, "http_status": status, "raw_path": str(raw_path), "raw_sha256": raw_sha, "parquet_path": str(parquet_path) if parquet_path.exists() else "", "parquet_sha256": parquet_sha, "rows": rows, "status": "downloaded" if status == 200 else "http_error"}

def download_chart_chunk(ctx: RunContext, dataset: str, family: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    base, params = working_url_for(ctx, family)
    if not base:
        return {"dataset": dataset, "endpoint_family": family, "symbol": symbol, "status": "skipped_no_working_endpoint", "chunk_start": str(start), "chunk_end": str(end)}
    params = dict(params or {})
    if "PF_XBTUSD" in base:
        base = base.replace("PF_XBTUSD", symbol)
    if "/1m" in base and str(ctx.args.backfill_resolution) != "1m":
        base = base.replace("/1m", f"/{ctx.args.backfill_resolution}")
    elif "symbol" in params:
        params["symbol"] = symbol
    params["from"] = int(start.timestamp())
    params["to"] = int(end.timestamp())
    params["count"] = 2000
    status, body, headers, full_url = public_http_get(base, params, timeout=30.0)
    chunk_id = stable_hash(f"{symbol}|{dataset}|{int(start.timestamp())}|{int(end.timestamp())}|{full_url}")
    raw_path = ctx.run_root / "downloaded_official_kraken/raw" / dataset / symbol / f"{start.strftime('%Y%m%dT%H%M%S')}_{chunk_id}.jsonl.gz"
    raw_sha = raw_write(raw_path, body)
    obj = json_loads_maybe(body)
    parquet_path = ctx.run_root / "downloaded_official_kraken/parquet" / dataset / symbol / f"{start.strftime('%Y%m%dT%H%M%S')}_{chunk_id}.parquet"
    rows = json_to_parquet(parquet_path, obj, {
        "source_url": full_url,
        "venue_symbol": symbol,
        "chunk_start_utc": start.isoformat(),
        "chunk_end_utc": end.isoformat(),
        "resolution": ctx.args.backfill_resolution,
        "historical_backfill": True,
        "rankable_pre_holdout": bool(end <= PROTECTED_STRATEGY_SELECTION_TS),
        "contains_protected_period": bool(end > PROTECTED_STRATEGY_SELECTION_TS),
    }) if status == 200 else 0
    parquet_sha = file_sha256(parquet_path) if parquet_path.exists() else ""
    return {
        "dataset": dataset,
        "endpoint_family": family,
        "symbol": symbol,
        "url": full_url,
        "http_status": status,
        "raw_path": str(raw_path),
        "raw_sha256": raw_sha,
        "parquet_path": str(parquet_path) if parquet_path.exists() else "",
        "parquet_sha256": parquet_sha,
        "rows": rows,
        "chunk_start": start.isoformat(),
        "chunk_end": end.isoformat(),
        "resolution": ctx.args.backfill_resolution,
        "rankable_pre_holdout": bool(end <= PROTECTED_STRATEGY_SELECTION_TS),
        "contains_protected_period": bool(end > PROTECTED_STRATEGY_SELECTION_TS),
        "status": "downloaded" if status == 200 else "http_error",
    }

def download_historical_bar_backfill(ctx: RunContext, manifest: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    if not ctx.args.historical_bar_backfill:
        return
    bf_start, bf_end = backfill_window(ctx)
    symbols = backfill_symbols(ctx)
    chunk_hours = effective_backfill_chunk_hours(ctx)
    datasets: list[tuple[str, str]] = []
    if ctx.args.backfill_include_trade_candles:
        datasets.append((f"historical_trade_candles_{ctx.args.backfill_resolution}", "candles_trade_1m"))
    if ctx.args.backfill_include_mark_candles:
        datasets.append((f"historical_mark_candles_{ctx.args.backfill_resolution}", "mark_candles_1m"))
    progress_path = ctx.run_root / "download/historical_bar_backfill_progress.csv"
    progress_rows = read_csv_or_empty(progress_path).to_dict("records")
    completed = load_completed_historical_chunks(ctx)
    # Precompute work units so progress/ETA-style status is meaningful.
    work_units: list[tuple[str, str, str, pd.Timestamp, pd.Timestamp]] = []
    for symbol in symbols:
        cur = bf_start
        while cur < bf_end:
            nxt = min(cur + pd.Timedelta(hours=chunk_hours), bf_end)
            for dataset, family in datasets:
                work_units.append((dataset, family, symbol, cur, nxt))
            cur = nxt
    total_units = len(work_units)
    skipped_units = 0
    processed_units = 0
    downloaded_rows = int(sum(float(r.get("rows", 0) or 0) for r in progress_rows if str(r.get("status")) == "downloaded"))
    started_monotonic = time.monotonic()
    ctx.notifier.send(
        "Kraken K0 historical backfill start",
        f"symbols={len(symbols)}; datasets={','.join(d for d, _ in datasets)}; chunks={total_units}; resolution={ctx.args.backfill_resolution}; {disk_status_line(ctx.run_root)}",
    )
    for dataset, family, symbol, cur, nxt in work_units:
        key = historical_chunk_key(dataset, symbol, cur, nxt, ctx.args.backfill_resolution)
        if key in completed:
            skipped_units += 1
            processed_units += 1
            continue
        rec = download_chart_chunk(ctx, dataset, family, symbol, cur, nxt)
        manifest.append(rec)
        progress_rows.append(rec)
        processed_units += 1
        if rec.get("status") == "downloaded":
            downloaded_rows += int(float(rec.get("rows", 0) or 0))
            completed.add(key)
        else:
            errors.append(rec)
        progress_every = max(1, int(ctx.args.backfill_progress_every))
        telegram_every = max(1, int(ctx.args.backfill_telegram_every))
        if processed_units % progress_every == 0:
            write_csv(progress_path, progress_rows)
        if processed_units % telegram_every == 0:
            elapsed = max(1.0, time.monotonic() - started_monotonic)
            rate = processed_units / elapsed
            remaining = max(0, total_units - processed_units)
            eta_sec = remaining / rate if rate > 0 else 0.0
            ctx.notifier.send(
                "Kraken K0 historical backfill progress",
                f"processed={processed_units}/{total_units} ({processed_units / max(total_units, 1):.1%}); skipped_resume={skipped_units}; rows={downloaded_rows}; errors={len(errors)}; rate_chunks_per_min={rate * 60:.1f}; eta_hours={eta_sec / 3600:.2f}; elapsed_hours={elapsed / 3600:.2f}; symbol={symbol}; {disk_status_line(ctx.run_root)}",
            )
    write_csv(progress_path, progress_rows)
    elapsed = max(1.0, time.monotonic() - started_monotonic)
    ctx.notifier.send(
        "Kraken K0 historical backfill done",
        f"processed={processed_units}/{total_units}; skipped_resume={skipped_units}; rows={downloaded_rows}; errors={len(errors)}; rate_chunks_per_min={(processed_units / elapsed) * 60:.1f}; elapsed_hours={elapsed / 3600:.2f}; {disk_status_line(ctx.run_root)}",
    )
    if progress_rows:
        df = pd.DataFrame(progress_rows)
        summary = df.groupby(["dataset", "symbol", "status"], dropna=False).agg(
            chunks=("status", "size"),
            rows=("rows", "sum"),
            rankable_pre_holdout_chunks=("rankable_pre_holdout", "sum"),
            protected_or_post_holdout_chunks=("contains_protected_period", "sum"),
        ).reset_index()
        summary["approx_coverage_days"] = summary["chunks"].astype(float) * chunk_hours / 24.0
        summary["approx_rankable_pre_holdout_days"] = summary["rankable_pre_holdout_chunks"].astype(float) * chunk_hours / 24.0
        write_csv(ctx.run_root / "download/historical_bar_backfill_summary.csv", summary)

def stage_download(ctx: RunContext) -> None:
    ddir = ctx.run_root / "download"
    ddir.mkdir(parents=True, exist_ok=True)
    errors = []
    manifest = []
    if not ctx.args.download_official_data:
        write_csv(ddir / "download_manifest.csv", [])
        write_csv(ddir / "download_errors.csv", [])
        write_text(ddir / "download_skipped_report.md", "# Download\n\nSkipped because --download-official-data was not passed.\n")
        return
    syms = selected_symbols(ctx)
    # Priority 1
    for dataset, family, symbolized in [("instruments", "instruments", False), ("tickers", "tickers", False)]:
        rec = download_one(ctx, dataset, family, None)
        manifest.append(rec)
        if rec.get("status") != "downloaded":
            errors.append(rec)
    if ctx.args.include_funding:
        for s in syms:
            rec = download_one(ctx, "funding", "historical_funding", s)
            manifest.append(rec)
            if rec.get("status") != "downloaded":
                errors.append(rec)
    if ctx.args.include_candles:
        for s in syms:
            rec = download_one(ctx, "candles", "candles_trade_1m", s)
            manifest.append(rec)
            if rec.get("status") != "downloaded":
                errors.append(rec)
    if ctx.args.include_mark_events:
        for s in syms:
            rec = download_one(ctx, "mark", "mark_candles_1m", s)
            manifest.append(rec)
            if rec.get("status") != "downloaded":
                errors.append(rec)
    if ctx.args.include_analytics:
        for family in ["analytics_open_interest", "analytics_funding", "analytics_liquidation_volume", "analytics_spreads", "analytics_liquidity", "analytics_slippage", "analytics_trade_volume", "analytics_trade_count"]:
            for s in syms[: max(1, min(len(syms), 10 if not ctx.args.smoke else 3))]:
                rec = download_one(ctx, "analytics", family, s)
                manifest.append(rec)
                if rec.get("status") != "downloaded":
                    errors.append(rec)
    if ctx.args.include_public_executions:
        for s in syms[: max(1, min(len(syms), 5))]:
            rec = download_one(ctx, "events", "public_execution_events", s)
            manifest.append(rec)
            if rec.get("status") != "downloaded":
                errors.append(rec)
    if ctx.args.include_public_order_events:
        for s in syms[: max(1, min(len(syms), 5))]:
            rec = download_one(ctx, "events", "public_order_events", s)
            manifest.append(rec)
            if rec.get("status") != "downloaded":
                errors.append(rec)
    download_historical_bar_backfill(ctx, manifest, errors)
    write_csv(ddir / "download_manifest.csv", manifest)
    write_csv(ddir / "download_errors.csv", errors)
    write_json(ddir / "download_resume_state.json", {"status": "complete", "manifest_rows": len(manifest), "error_rows": len(errors), "completed_utc": utc_now()})
    # Persist a verified copy under the Kraken derivatives data root. Raw
    # staging remains intact; this is an additive copy, not destructive promotion.
    persist = []
    for rec in manifest:
        if rec.get("status") != "downloaded":
            continue
        dataset = str(rec.get("dataset") or "unknown")
        raw_symbol = rec.get("symbol")
        symbol = "all" if raw_symbol is None or (isinstance(raw_symbol, float) and math.isnan(raw_symbol)) or str(raw_symbol).lower() == "nan" or str(raw_symbol) == "" else str(raw_symbol)
        raw_src = Path(str(rec.get("raw_path") or ""))
        pq_src = Path(str(rec.get("parquet_path") or ""))
        raw_dst = PERSISTENT_ROOT / "raw" / dataset / symbol / raw_src.name if raw_src.exists() else Path("")
        if pq_src.exists() and pq_src.parent.name == symbol:
            pq_dst = PERSISTENT_ROOT / "parquet" / dataset / symbol / pq_src.name
        else:
            pq_dst = PERSISTENT_ROOT / "parquet" / dataset / pq_src.name if pq_src.exists() else Path("")
        if raw_src.exists():
            raw_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_src, raw_dst)
        if pq_src.exists():
            pq_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pq_src, pq_dst)
        persist.append({
            "dataset": dataset,
            "symbol": symbol,
            "staged_raw_path": str(raw_src) if raw_src.exists() else "",
            "persistent_raw_path": str(raw_dst) if raw_src.exists() else "",
            "staged_parquet_path": str(pq_src) if pq_src.exists() else "",
            "persistent_parquet_path": str(pq_dst) if pq_src.exists() else "",
            "raw_sha256": rec.get("raw_sha256"),
            "persistent_raw_sha256": file_sha256(raw_dst) if raw_src.exists() else "",
            "parquet_sha256": rec.get("parquet_sha256"),
            "persistent_parquet_sha256": file_sha256(pq_dst) if pq_src.exists() else "",
        })
    (PERSISTENT_ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ddir / "download_manifest.csv", PERSISTENT_ROOT / "manifests" / f"{ctx.run_root.name}_download_manifest.csv")
    write_csv(ddir / "persistent_store_manifest.csv", persist)

def stage_qc(ctx: RunContext) -> None:
    manifest = read_csv_or_empty(ctx.run_root / "download/download_manifest.csv")
    rows = []
    issues = []
    for _, rec in manifest.iterrows() if not manifest.empty else []:
        p = Path(str(rec.get("parquet_path", "")))
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            issues.append({"path": str(p), "issue": f"parquet_read_error:{type(exc).__name__}"})
            continue
        ts_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "timestamp"])]
        duplicate_ts = 0
        non_monotone = False
        nonpositive_price = 0
        if ts_cols:
            ts = pd.to_datetime(df[ts_cols[0]], utc=True, errors="coerce")
            duplicate_ts = int(ts.duplicated().sum())
            non_monotone = bool((ts.dropna().sort_values().to_numpy() != ts.dropna().to_numpy()).any()) if ts.notna().sum() > 1 else False
        price_cols = [c for c in df.columns if c.lower() in {"price", "mark", "markprice", "last", "open", "high", "low", "close"}]
        for c in price_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            nonpositive_price += int((vals <= 0).sum())
        rows.append({"dataset": rec.get("dataset"), "symbol": rec.get("symbol"), "path": str(p), "rows": len(df), "timestamp_columns": ";".join(ts_cols), "duplicate_timestamps": duplicate_ts, "non_monotone_timestamps": non_monotone, "nonpositive_price_values": nonpositive_price, "status": "pass" if duplicate_ts == 0 and not non_monotone and nonpositive_price == 0 else "warn"})
        if duplicate_ts or non_monotone or nonpositive_price:
            issues.append({"path": str(p), "issue": "timestamp_or_price_qc_warning", "duplicate_timestamps": duplicate_ts, "non_monotone": non_monotone, "nonpositive_price_values": nonpositive_price})
    write_csv(ctx.run_root / "qc/qc_summary.csv", rows)
    write_csv(ctx.run_root / "qc/qc_issues.csv", issues)
    write_text(ctx.run_root / "qc/qc_report.md", f"# QC Report\n\nDatasets checked: {len(rows)}. Issues/warnings: {len(issues)}. Protected strategy-selection holdout remains a future strategy-selection policy, not a K0 download block.\n")

def stage_instrument_master(ctx: RunContext) -> None:
    manifest = read_csv_or_empty(ctx.run_root / "download/download_manifest.csv")
    p = ""
    if not manifest.empty:
        rows = manifest[(manifest["dataset"].astype(str) == "instruments") & manifest["parquet_path"].notna()]
        if not rows.empty:
            p = str(rows.iloc[0]["parquet_path"])
    if not p:
        # Try schema sample from probe.
        sample = read_json_or_empty(ctx.run_root / "probes/schema_samples/instruments_0_200.json")
        obj = sample.get("sample") if isinstance(sample, dict) else sample
        recs = flatten_records(obj)
        df = pd.DataFrame(recs)
    else:
        df = pd.read_parquet(p)
    records = []
    for _, r in df.iterrows() if not df.empty else []:
        sym = safe_symbol_from_instrument(r)
        if not sym:
            continue
        parsed = parse_kraken_symbol(sym)
        records.append({
            "venue": "kraken_derivatives", **parsed,
            "settlement_currency": r.get("settlementCurrency", r.get("quote", "USD")),
            "collateral_wallet_type": "Multi-M_or_unknown",
            "contract_type": r.get("type", r.get("contractType", "unknown")),
            "tick_size": r.get("tickSize", r.get("tick_size", "")),
            "min_order_size": r.get("contractSize", r.get("tradeSize", "")),
            "contract_size": r.get("contractSize", ""),
            "opening_date": r.get("openingDate", r.get("effectiveDate", "")),
            "status": r.get("tradeable", r.get("status", "unknown")),
            "margin_class": r.get("marginType", "unknown"),
            "max_leverage": r.get("maxLeverage", "unknown"),
            "source_confidence": "official_public_endpoint" if p else "probe_sample",
        })
    master = pd.DataFrame(records)
    outdir = ctx.run_root / "instrument_master"
    outdir.mkdir(parents=True, exist_ok=True)
    master.to_parquet(outdir / "kraken_instrument_master.parquet", index=False)
    intervals = master[["venue_symbol", "opening_date", "status", "source_confidence"]].copy() if not master.empty else pd.DataFrame(columns=["venue_symbol", "opening_date", "status", "source_confidence"])
    if not intervals.empty:
        intervals["interval_start"] = intervals["opening_date"]
        intervals["interval_end"] = "unknown"
    intervals.to_parquet(outdir / "kraken_lifecycle_intervals.parquet", index=False)
    write_text(outdir / "instrument_master_report.md", f"# Kraken Instrument Master\n\nRows: {len(master)}. Missing delist dates are unknown, not inferred as never delisted.\n")

def stage_panel(ctx: RunContext) -> None:
    manifest = read_csv_or_empty(ctx.run_root / "download/download_manifest.csv")
    parts = []
    if not manifest.empty:
        for _, rec in manifest.iterrows():
            if str(rec.get("dataset")) not in {"candles", "mark", "funding", "analytics"}:
                continue
            p = Path(str(rec.get("parquet_path", "")))
            if p.exists():
                df = pd.read_parquet(p)
                df["dataset"] = rec.get("dataset")
                df["venue_symbol"] = rec.get("symbol", df.get("venue_symbol", ""))
                parts.append(df.head(5000 if ctx.args.smoke else 100000))
    panel = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    pdir = ctx.run_root / "panels"
    pdir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(pdir / "kraken_k0_panel.parquet", index=False)
    feature_rows = [
        {"feature": "funding_per_hour", "definition": "funding rate normalized by observed period when exact cadence known; otherwise capped proxy"},
        {"feature": "funding_8h_equiv", "definition": "funding_per_hour * 8, not directly comparable to Bybit without cap"},
        {"feature": "oi_usd_or_proxy", "definition": "open-interest analytics where official field exists"},
        {"feature": "liquidity_spread_proxy", "definition": "analytics spread/liquidity/slippage endpoints when available"},
    ]
    write_csv(pdir / "kraken_feature_dictionary.csv", feature_rows)
    write_text(pdir / "panel_build_report.md", f"# K0 Panel\n\nRows: {len(panel)}. Raw Kraken funding/OI are not compared directly to Bybit; normalized feature dictionary records cap semantics.\n")

def stage_universe(ctx: RunContext) -> None:
    master_path = ctx.run_root / "instrument_master/kraken_instrument_master.parquet"
    master = pd.read_parquet(master_path) if master_path.exists() else pd.DataFrame()
    rows = []
    priority = {"XBT", "BTC", "ETH", "SOL", "XRP", "HYPE", "ADA", "DOGE", "LINK", "AVAX", "BNB"}
    for _, r in master.iterrows() if not master.empty else []:
        base = str(r.get("base_asset", ""))
        tier = "K-A" if base in priority else "K-B"
        if str(r.get("status", "")).lower() in {"false", "closed", "delisted"}:
            tier = "K-D"
        rows.append({"venue_symbol": r.get("venue_symbol"), "base_asset": base, "tier": tier, "default_threshold_basis": "kraken_official_instrument_presence_initial_K0", "relaxed_small_account_threshold": True})
    df = pd.DataFrame(rows)
    udir = ctx.run_root / "universe"
    udir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(udir / "kraken_liquidity_tiers_by_date.parquet", index=False)
    write_csv(udir / "kraken_universe_summary.csv", df)
    write_text(udir / "universe_report.md", f"# Kraken Universe\n\nRows: {len(df)}. K-A/K-B are initial K0 tiers and must be refined using Kraken volume/OI/spread analytics before strategy ranking.\n")

def stage_portability(ctx: RunContext) -> None:
    families = ["A1", "A2", "A3", "A4", "B1", "C2", "D1_D3", "D4", "E1", "F1_G1", "Branch_X_listing", "ORB_funding_window", "generic_shock"]
    rows = []
    for fam in families:
        if fam in {"A1", "A2", "A3", "A4"}:
            cls = "kraken_progress_with_official_data"
            first = "K1 liquid Tier A/B official candles+funding+OI screen"
        elif fam in {"B1", "C2"}:
            cls = "kraken_candidate_library_only"
            first = "migrate only if symbol/event breadth exists and first reaction exclusions are enforced"
        elif fam in {"D4", "E1", "Branch_X_listing", "ORB_funding_window"}:
            cls = "kraken_needs_live_capture_substitute"
            first = "capture/redesign before ranking"
        else:
            cls = "kraken_redesign_to_less_depth_sensitive"
            first = "delayed close-confirmed redesign or library only"
        rows.append({"family": fam, "current_bybit_evidence_level": "see QLMG evidence contract; not imported as Kraken evidence", "kraken_data_sufficiency": cls, "expected_kraken_suitability": "conditional", "required_data_tier": "K-Tier1" if fam in {"A1", "A2", "A3", "A4"} else "K-Tier2_or_capture", "no_vendor_path": cls, "likely_first_kraken_test": first, "kraken_family_classification": cls})
    write_csv(ctx.run_root / "portability/kraken_bybit_family_portability.csv", rows)
    write_text(ctx.run_root / "portability/portability_report.md", "# Kraken/Bybit Portability\n\nBybit PnL is not Kraken evidence. A1/A2/A3/A4 migrate first; execution-sensitive families need capture or redesign. No family is left waiting for vendor data.\n")

def k1_eligibility(ctx: RunContext) -> dict[str, Any]:
    master_path = ctx.run_root / "instrument_master/kraken_instrument_master.parquet"
    panel_path = ctx.run_root / "panels/kraken_k0_panel.parquet"
    master_exists = master_path.exists() and (len(pd.read_parquet(master_path)) > 0 if master_path.exists() else False)
    panel_exists = panel_path.exists() and (len(pd.read_parquet(panel_path)) > 0 if panel_path.exists() else False)
    dl_manifest = read_csv_or_empty(ctx.run_root / "download/download_manifest.csv")
    downloaded_rows = int((dl_manifest.get("status", pd.Series(dtype=str)).astype(str).eq("downloaded")).sum()) if not dl_manifest.empty else 0
    qc = read_csv_or_empty(ctx.run_root / "qc/qc_summary.csv")
    universe = read_csv_or_empty(ctx.run_root / "universe/kraken_universe_summary.csv")
    ret = read_csv_or_empty(ctx.run_root / "download/kraken_retention_depth_matrix.csv")
    candles = bool((ret.get("data_family", pd.Series(dtype=str)).astype(str).eq("candles") & ret.get("tier1_usability", pd.Series(dtype=bool)).astype(bool)).any()) if not ret.empty else False
    funding = bool((ret.get("data_family", pd.Series(dtype=str)).astype(str).eq("historical_funding") & ret.get("tier1_usability", pd.Series(dtype=bool)).astype(bool)).any()) if not ret.empty else False
    oi = bool((ret.get("data_family", pd.Series(dtype=str)).astype(str).eq("analytics_open_interest") & ret.get("tier1_usability", pd.Series(dtype=bool)).astype(bool)).any()) if not ret.empty else False
    for ticker_path in (ctx.run_root / "downloaded_official_kraken/parquet/tickers").glob("*.parquet"):
        ticker_df = pd.read_parquet(ticker_path)
        if "openInterest" in ticker_df.columns and ticker_df["openInterest"].notna().any():
            oi = True
            break
    ka_kb = int(universe[universe.get("tier", pd.Series(dtype=str)).astype(str).isin(["K-A", "K-B"])].shape[0]) if not universe.empty else 0
    hist = read_csv_or_empty(ctx.run_root / "download/historical_bar_backfill_summary.csv")
    hist_symbols = 0
    max_hist_days = 0.0
    if not hist.empty and {"dataset", "symbol", "status", "approx_coverage_days"}.issubset(set(hist.columns)):
        day_col = "approx_rankable_pre_holdout_days" if "approx_rankable_pre_holdout_days" in hist.columns else "approx_coverage_days"
        trade = hist[
            hist["dataset"].astype(str).str.startswith("historical_trade_candles")
            & hist["status"].astype(str).eq("downloaded")
        ].copy()
        if not trade.empty:
            max_hist_days = float(trade[day_col].max())
            hist_symbols = int(trade[trade[day_col].astype(float) >= 1000.0]["symbol"].nunique())
    historical_candles_usable = hist_symbols >= 2
    critical_qc = bool((qc.get("status", pd.Series(dtype=str)).astype(str).eq("fail")).any()) if not qc.empty else False
    eligible = bool(downloaded_rows > 0 and master_exists and panel_exists and candles and historical_candles_usable and funding and oi and ka_kb >= 2 and not critical_qc)
    return {"k1_can_start": eligible, "downloaded_dataset_chunks": downloaded_rows, "instrument_master": master_exists, "k0_panel": panel_exists, "candles_usable": candles, "historical_candles_usable": historical_candles_usable, "historical_candle_symbols_1000d_plus": hist_symbols, "max_historical_candle_coverage_days": max_hist_days, "funding_or_proxy_available": funding, "oi_or_analytics_proxy_available": oi, "ka_kb_symbol_count": ka_kb, "critical_qc_issue": critical_qc, "k1_scope": "A4,A1,A2_filter,A3; B1/C2 conditional" if eligible else "blocked_or_continue_K0"}

def stage_readiness(ctx: RunContext) -> None:
    elig = k1_eligibility(ctx)
    rows = []
    for fam in ["A4", "A1", "A2_filter", "A3", "B1", "C2", "D4", "Branch_X", "ORB_funding_window"]:
        can_screen = fam in {"A4", "A1", "A2_filter", "A3"} and bool(elig["k1_can_start"])
        rows.append({"family": fam, "can_screen_now": can_screen, "can_rank_train_only_now": False, "can_validate_train_only_now": False, "needs_capture": fam in {"D4", "Branch_X", "ORB_funding_window"}, "needs_private_live_connector": False, "needs_manual_kraken_account_check": True, "needs_strategy_redesign": fam not in {"A4", "A1", "A2_filter", "A3"}, "next_action": "K1_screen" if can_screen else "data_or_capture_or_redesign"})
    write_csv(ctx.run_root / "readiness/kraken_strategy_readiness_matrix.csv", rows)
    write_json(ctx.run_root / "readiness/k1_eligibility.json", elig)
    write_text(ctx.run_root / "readiness/readiness_report.md", "# Kraken Strategy Readiness\n\n" + json.dumps(elig, indent=2))

def stage_live_capture(ctx: RunContext) -> None:
    universe = read_csv_or_empty(ctx.run_root / "universe/kraken_universe_summary.csv")
    symbols = universe["venue_symbol"].dropna().astype(str).head(30).tolist() if not universe.empty and "venue_symbol" in universe.columns else ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_ADAUSD"]
    rows = [{"venue_symbol": s, "priority": i + 1, "capture_reason": "K-A/K-B or default priority"} for i, s in enumerate(symbols)]
    write_csv(ctx.run_root / "live_capture/kraken_capture_symbol_list.csv", rows)
    spec = "# Kraken Live Capture Spec\n\nPublic streams only. Capture instruments/status hourly, tickers every 10-60s, orderbook top levels, trades, funding/OI ticker fields, and liquidation/termination/block-like public fields if present. No orders. No private trading endpoints. Telegram summaries and disk guards required.\n"
    write_text(ctx.run_root / "live_capture/kraken_live_capture_spec.md", spec)
    write_text(ctx.run_root / "live_capture/kraken_live_capture_agent_prompt.md", spec + "\nPriority symbols:\n" + "\n".join(f"- {s}" for s in symbols[:20]))
    write_text(ctx.run_root / "live_capture/kraken_capture_storage_estimate.md", "# Capture Storage Estimate\n\nStart with selected K-A/K-B symbols. Expect orderbook/trades to dominate storage; enforce disk guard and rotating chunk manifests.\n")

def stage_engine_gap(ctx: RunContext) -> None:
    rows = [
        {"gap": "symbol_mapping", "required_change": "venue_symbol/display_symbol separation"},
        {"gap": "settlement_pnl", "required_change": "USD settlement and collateral/PnL currency model"},
        {"gap": "funding", "required_change": "Kraken funding cadence/accrual contract"},
        {"gap": "mark_liquidation", "required_change": "Kraken mark/trigger/closeout semantics"},
        {"gap": "position_model", "required_change": "one-way net position per maturity"},
        {"gap": "orders", "required_change": "price-protected IOC/partial-fill semantics"},
        {"gap": "fees", "required_change": "Kraken fee schedule source and caps"},
        {"gap": "event_ledger", "required_change": "Kraken data-tier and currency fields"},
    ]
    write_csv(ctx.run_root / "engine/kraken_required_code_changes.csv", rows)
    write_text(ctx.run_root / "engine/kraken_engine_gap_report.md", "# Kraken Engine Gap Report\n\nKraken cannot reuse Bybit mechanics. Required code changes are listed in `kraken_required_code_changes.csv`.\n")
    write_text(ctx.run_root / "account/kraken_account_mechanics_checklist.md", "# Kraken Account Mechanics Checklist\n\n- EEA derivatives access confirmed: manual unknown\n- Retail/pro classification: manual unknown\n- Max leverage: manual unknown\n- Multi-M collateral wallet needed: likely/manual\n- USD settlement / PnL accounting: required\n- One-way net position model: required\n- Fee schedule source: official Kraken/manual confirmation\n- Demo environment: manual check\n- Private connector later: separate prompt, not K0\n")

def stage_report(ctx: RunContext) -> None:
    probe = read_csv_or_empty(ctx.run_root / "probes/endpoint_capability_matrix.csv")
    dl = read_csv_or_empty(ctx.run_root / "download/download_manifest.csv")
    qc = read_csv_or_empty(ctx.run_root / "qc/qc_summary.csv")
    elig = read_json_or_empty(ctx.run_root / "readiness/k1_eligibility.json")
    tg_ready = read_json_or_empty(ctx.run_root / "notifications/telegram_readiness.json")
    telegram_worked = bool(ctx.notifier.remote_available or tg_ready.get("remote_available", False))
    working = int(probe.get("works", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1"]).sum()) if not probe.empty and "works" in probe.columns else 0
    downloaded = int((dl.get("status", pd.Series(dtype=str)).astype(str).eq("downloaded")).sum()) if not dl.empty else 0
    critical_qc = bool(elig.get("critical_qc_issue", False)) if isinstance(elig, dict) else False
    if working == 0:
        primary = "blocked_by_kraken_public_data_access"
        secondary = "build_kraken_live_capture_next"
    elif downloaded == 0 and ctx.args.download_official_data:
        primary = "continue_kraken_official_download_next"
        secondary = "build_kraken_backtest_engine_next"
    elif bool(elig.get("k1_can_start", False)):
        primary = "run_kraken_k1_strategy_screen_next"
        secondary = "build_kraken_live_capture_next"
    else:
        primary = "continue_kraken_official_download_next" if not ctx.args.download_official_data or downloaded == 0 else "build_kraken_backtest_engine_next"
        secondary = "build_kraken_live_capture_next"
    if primary not in ALLOWED_NEXT_DECISIONS:
        primary = "blocked_by_protocol_issue"
    decision = {
        "run_root": str(ctx.run_root),
        "final_holdout_strategy_selection_protected": True,
        "telegram_worked": telegram_worked,
        "endpoint_probe_verdict": "working_public_endpoints_found" if working else "no_working_public_endpoints_found",
        "data_download_verdict": "downloaded" if downloaded else ("not_requested" if not ctx.args.download_official_data else "no_downloads_completed"),
        "data_qc_verdict": "critical_qc_issue" if critical_qc else "qc_complete_or_no_critical_issue",
        "instrument_master_verdict": "built" if (ctx.run_root / "instrument_master/kraken_instrument_master.parquet").exists() else "not_built",
        "k0_panel_verdict": "built" if (ctx.run_root / "panels/kraken_k0_panel.parquet").exists() else "not_built",
        "kraken_universe_verdict": "built" if (ctx.run_root / "universe/kraken_universe_summary.csv").exists() else "not_built",
        "portability_verdict": "built",
        "strategy_readiness_verdict": "k1_can_start" if bool(elig.get("k1_can_start", False)) else "continue_k0_or_engine_work",
        "live_capture_spec_verdict": "built",
        "primary_next_operator_decision": primary,
        "secondary_next_operator_decision": secondary,
        "live_capture_spec_path": str(ctx.run_root / "live_capture/kraken_live_capture_agent_prompt.md"),
        "engine_gap_report_path": str(ctx.run_root / "engine/kraken_engine_gap_report.md"),
        "compact_bundle_path": str(ctx.run_root / "compact_review_bundle"),
    }
    write_json(ctx.run_root / "decision_summary.json", decision)
    report = [
        "# Kraken K0 Data Foundation Report", "", f"Run root: `{ctx.run_root}`", "Final holdout strategy-selection protected: yes", f"Telegram worked: {'yes' if ctx.notifier.remote_available else 'no'}", "",
        "## Endpoint Probe", f"Working endpoint probes: {working}", "", "## Download", f"Downloaded dataset chunks: {downloaded}", "", "## K1 Eligibility", json.dumps(elig, indent=2), "", "## Next Operator Decision", f"Primary: `{primary}`", f"Secondary: `{secondary}`", "", "No strategy validation, live readiness, production readiness, or trading recommendation is claimed.",
    ]
    write_text(ctx.run_root / "KRAKEN_K0_DATA_FOUNDATION_REPORT.md", "\n".join(report))

def stage_bundle(ctx: RunContext) -> None:
    bundle = ctx.run_root / "compact_review_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    include = [
        "KRAKEN_K0_DATA_FOUNDATION_REPORT.md", "decision_summary.json", "preflight/preflight_report.md", "preflight/resource_guard_report.md", "probes/endpoint_capability_matrix.csv", "probes/endpoint_probe_report.md", "download/storage_estimate.csv", "download/download_plan.md", "download/kraken_retention_depth_matrix.csv", "download/download_manifest.csv", "download/download_errors.csv", "download/persistent_store_manifest.csv", "download/kraken_funding_export_audit.csv", "download/kraken_funding_export_audit.md", "qc/qc_summary.csv", "qc/qc_issues.csv", "instrument_master/instrument_master_report.md", "universe/kraken_universe_summary.csv", "portability/kraken_bybit_family_portability.csv", "readiness/kraken_strategy_readiness_matrix.csv", "readiness/k1_eligibility.json", "live_capture/kraken_live_capture_spec.md", "live_capture/kraken_live_capture_agent_prompt.md", "engine/kraken_engine_gap_report.md", "engine/kraken_required_code_changes.csv", "account/kraken_account_mechanics_checklist.md",
    ]
    idx = []
    for rel in include:
        src = ctx.run_root / rel
        if src.exists() and src.is_file() and src.stat().st_size < 5_000_000:
            dst = bundle / rel.replace("/", "__")
            shutil.copy2(src, dst)
            idx.append({"source": rel, "bundle_file": dst.name, "size_bytes": src.stat().st_size})
    # include schema contracts and samples small only
    for src in list((ctx.run_root / "contracts").glob("*"))[:50] + list((ctx.run_root / "probes/schema_samples").glob("*.json"))[:50]:
        if src.is_file() and src.stat().st_size < 1_000_000:
            dst = bundle / str(src.relative_to(ctx.run_root)).replace("/", "__")
            shutil.copy2(src, dst)
            idx.append({"source": str(src.relative_to(ctx.run_root)), "bundle_file": dst.name, "size_bytes": src.stat().st_size})
    write_csv(bundle / "artifact_index.csv", idx)

STAGE_FUNCS = {
    "preflight-and-source-freeze": stage_preflight,
    "telegram-and-tmux-setup": stage_telegram,
    "kraken-official-endpoint-probe": stage_probe,
    "kraken-schema-contracts": stage_contracts,
    "kraken-storage-and-download-plan": stage_storage,
    "kraken-official-data-download": stage_download,
    "kraken-data-qc": stage_qc,
    "kraken-instrument-master-and-lifecycle": stage_instrument_master,
    "kraken-bar-mark-funding-analytics-panel": stage_panel,
    "kraken-universe-and-liquidity-tiers": stage_universe,
    "kraken-bybit-portability-matrix": stage_portability,
    "kraken-strategy-readiness-matrix": stage_readiness,
    "kraken-live-capture-spec": stage_live_capture,
    "kraken-backtest-engine-gap-report": stage_engine_gap,
    "decision-report": stage_report,
    "compact-review-bundle": stage_bundle,
}

def run_stage(ctx: RunContext, stage: str) -> None:
    if stage == "all":
        for s in stage_list("all"):
            run_stage(ctx, s)
        return
    if ctx.args.resume and done_path(ctx, stage).exists():
        return
    ctx.notifier.send("Kraken K0 stage start", stage)
    if not ctx.args.dry_run:
        STAGE_FUNCS[stage](ctx)
    mark_done(ctx, stage)
    ctx.notifier.send("Kraken K0 stage done", stage)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root, reason = resolve_run_root(args)
    start, end = parse_window(args)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_context.json", {"run_root": str(run_root), "root_reason": reason, "argv": argv if argv is not None else sys.argv[1:], "start": str(start), "end": str(end), "created_utc": utc_now()})
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    try:
        for stage in stage_list(args.stage):
            run_stage(ctx, stage)
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "complete", "ts_utc": utc_now()})
        notifier.send("Kraken K0 run complete", str(run_root))
        return 0
    except Exception as exc:
        write_json(run_root / "watch_status.json", {"run_root": str(run_root), "status": "failed", "error": f"{type(exc).__name__}: {exc}", "ts_utc": utc_now()})
        notifier.send("Kraken K0 run failed", f"{type(exc).__name__}: {exc}", level="error")
        raise

if __name__ == "__main__":
    raise SystemExit(main())
