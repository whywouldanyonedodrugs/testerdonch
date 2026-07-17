#!/usr/bin/env python3
"""Acquire bounded official evidence for Kraken candle base-volume semantics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tools.kraken_candle_volume_authority import (
        aggregate_execution_interval, canonical_hash, validate_candle_interval,
        validate_semantic_versions,
    )
except ModuleNotFoundError:
    from kraken_candle_volume_authority import (
        aggregate_execution_interval, canonical_hash, validate_candle_interval,
        validate_semantic_versions,
    )


SYMBOLS = ("PF_XBTUSD", "PF_ETHUSD", "PF_XRPUSD", "PF_AAVEUSD")
INTERVAL_STARTS_MS = (1784277000000, 1784277300000, 1784277600000)
USER_AGENT = "Donch-Stage2C1-VolumeAuthority/1.0"
SAFE_RESPONSE_HEADERS = {
    "accept-ranges", "age", "cache-control", "cf-cache-status", "content-encoding",
    "content-length", "content-range", "content-type", "date", "etag", "expires",
    "is-truncated", "last-modified", "location", "next-continuation-token", "server",
    "strict-transport-security", "vary", "x-archive-orig-cf-cache-status",
    "x-archive-orig-content-encoding", "x-archive-orig-content-length",
    "x-archive-orig-content-type", "x-archive-src", "x-content-type-options",
}
OFFICIAL_URLS = (
    ("requested_market_candles", "https://docs.kraken.com/api/docs/futures-api/charts/candles"),
    ("requested_trade_history", "https://docs.kraken.com/api/docs/futures-api/trading/get-history"),
    ("linear_specs_current", "https://support.kraken.com/articles/4844359082772-linear-multi-collateral-derivatives-contract-specifications"),
    ("eea_specs_current", "https://support.kraken.com/articles/perpetual-contract-specifications-for-clients-in-the-eea"),
)
ARCHIVED_URLS = (
    ("market_candles_docs_20260414", "20260414100136", "https://docs.kraken.com/api/docs/futures-api/charts/candles/"),
    ("legacy_trade_history_docs_20260216", "20260216012157", "https://docs.kraken.com/api/docs/futures-api/trading/get-history/"),
    ("public_execution_docs_20260414", "20260414100712", "https://docs.kraken.com/api/docs/futures-api/history/get-public-execution-events/"),
    ("linear_specs_20230223", "20230223084816", "https://support.kraken.com/hc/en-us/articles/4844359082772-Linear-Multi-Collateral-Perpetual-Contract-Specifications"),
    ("linear_specs_20231210", "20231210001614", "https://support.kraken.com/hc/en-us/articles/4844359082772-Linear-Multi-Collateral-Perpetual-Contract-Specifications"),
    ("linear_specs_20240417", "20240417123420", "https://support.kraken.com/hc/en-us/articles/4844359082772-Linear-Multi-Collateral-Perpetual-Contract-Specifications"),
    ("linear_specs_20241207", "20241207021651", "https://support.kraken.com/hc/en-us/articles/4844359082772-Linear-Multi-Collateral-Perpetual-Contract-Specifications"),
    ("linear_specs_20250208", "20250208030305", "https://support.kraken.com/hc/en-us/articles/4844359082772-Linear-Multi-Collateral-Perpetual-Contract-Specifications"),
    ("linear_specs_20250323", "20250323232448", "https://support.kraken.com/hc/en-us/articles/4844359082772-Linear-Multi-Collateral-Perpetual-Contract-Specifications"),
    ("linear_specs_20250617", "20250617054715", "https://support.kraken.com/articles/4844359082772-linear-multi-collateral-derivatives-contract-specifications"),
    ("linear_specs_20250815", "20250815212134", "https://support.kraken.com/articles/4844359082772-linear-multi-collateral-derivatives-contract-specifications"),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch(url: str) -> tuple[bytes, dict[str, str], int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/html,*/*"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            headers = {key: value for key, value in response.headers.items() if key.lower() in SAFE_RESPONSE_HEADERS}
            return response.read(), headers, int(response.status), response.geturl()
    except urllib.error.HTTPError as exc:
        headers = {key: value for key, value in exc.headers.items() if key.lower() in SAFE_RESPONSE_HEADERS}
        return exc.read(), headers, int(exc.code), exc.geturl()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(payload)
    temp.replace(path)


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.cell: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"td", "th"}:
            self.in_cell = True
            self.cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.in_cell:
            self.row.append(" ".join("".join(self.cell).split()))
            self.in_cell = False
        elif tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell.append(data)


def parse_spec_rows(payload: bytes, snapshot_ts: str, source_sha256: str) -> list[dict[str, str]]:
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    parser = TableParser()
    parser.feed(payload.decode("utf-8", errors="strict"))
    output: list[dict[str, str]] = []
    for row in parser.rows:
        symbol = row[0].rstrip("*") if row else ""
        offset = 1
        if len(row) > 2 and row[1] in {"Active", "Inactive"}:
            offset = 2
        if not symbol.startswith("PF_") or len(row) <= offset + 1:
            continue
        base_label, min_lot = row[offset], row[offset + 1]
        base = base_label.rsplit("(", 1)[-1].rstrip(")*").strip() if "(" in base_label else base_label
        try:
            float(min_lot.replace(",", ""))
        except ValueError:
            continue
        output.append({
            "symbol": symbol, "snapshot_ts": snapshot_ts,
            "base_currency": base, "base_currency_label": base_label,
            "min_lot": min_lot.replace(",", ""), "source_sha256": source_sha256,
        })
    return output


def acquire_sources(output: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    source_dir = output / "sources" / "official_docs"
    ledger: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, str]] = []
    for source_id, url in OFFICIAL_URLS:
        body, headers, status, final_url = fetch(url)
        base = source_dir / source_id
        write_atomic(base.with_suffix(".body"), body)
        write_atomic(base.with_suffix(".headers.json"), (json.dumps(headers, indent=2, sort_keys=True) + "\n").encode())
        ledger.append({
            "source_id": source_id, "requested_url": url, "final_url": final_url,
            "accessed_utc": utc_now(), "http_status": status, "bytes": len(body),
            "sha256": sha256_bytes(body), "archived_snapshot_ts": "",
            "body_path": str(base.with_suffix(".body").relative_to(output)),
        })
        if source_id == "linear_specs_current" and status == 200:
            semantic_rows.extend(parse_spec_rows(body, "2026-07-17T00:00:00Z", sha256_bytes(body)))
    for source_id, timestamp, original in ARCHIVED_URLS:
        url = f"https://web.archive.org/web/{timestamp}id_/{original}"
        body, headers, status, final_url = fetch(url)
        base = source_dir / source_id
        write_atomic(base.with_suffix(".body"), body)
        write_atomic(base.with_suffix(".headers.json"), (json.dumps(headers, indent=2, sort_keys=True) + "\n").encode())
        ledger.append({
            "source_id": source_id, "requested_url": url, "final_url": final_url,
            "accessed_utc": utc_now(), "http_status": status, "bytes": len(body),
            "sha256": sha256_bytes(body), "archived_snapshot_ts": timestamp,
            "body_path": str(base.with_suffix(".body").relative_to(output)),
        })
        if source_id.startswith("linear_specs_") and status == 200:
            semantic_rows.extend(parse_spec_rows(
                body, pd.to_datetime(timestamp, format="%Y%m%d%H%M%S", utc=True).isoformat(), sha256_bytes(body),
            ))
    return ledger, pd.DataFrame(semantic_rows)


def acquire_calibration(output: Path) -> pd.DataFrame:
    raw_dir = output / "sources" / "calibration_raw"
    rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        candle_url = (
            f"https://futures.kraken.com/api/charts/v1/trade/{symbol}/5m?"
            + urllib.parse.urlencode({"from": INTERVAL_STARTS_MS[0] // 1000, "to": (INTERVAL_STARTS_MS[-1] + 300_000) // 1000})
        )
        candle_body, candle_headers, candle_status, _ = fetch(candle_url)
        if candle_status != 200:
            raise ValueError(f"candle request failed for {symbol}: {candle_status}")
        write_atomic(raw_dir / f"{symbol}_candles.json", candle_body)
        write_atomic(raw_dir / f"{symbol}_candles.headers.json", (json.dumps(candle_headers, indent=2, sort_keys=True) + "\n").encode())
        candles = json.loads(candle_body).get("candles", [])
        for start_ms in INTERVAL_STARTS_MS:
            pages: list[dict[str, Any]] = []
            token = ""
            page_number = 0
            while True:
                params: dict[str, Any] = {
                    "since": start_ms, "before": start_ms + 300_000,
                    "sort": "asc", "count": 1000,
                }
                if token:
                    params["continuation_token"] = token
                url = f"https://futures.kraken.com/api/history/v3/market/{symbol}/executions?{urllib.parse.urlencode(params)}"
                body, headers, status, _ = fetch(url)
                if status != 200:
                    raise ValueError(f"execution request failed for {symbol}:{start_ms}: {status}")
                write_atomic(raw_dir / f"{symbol}_{start_ms}_executions_{page_number:03d}.json", body)
                write_atomic(raw_dir / f"{symbol}_{start_ms}_executions_{page_number:03d}.headers.json", (json.dumps(headers, indent=2, sort_keys=True) + "\n").encode())
                pages.append(json.loads(body))
                truncated = str(headers.get("is-truncated", headers.get("Is-Truncated", "false"))).lower() == "true"
                token = headers.get("next-continuation-token", headers.get("Next-Continuation-Token", ""))
                page_number += 1
                if not truncated:
                    break
                if not token or page_number > 100:
                    raise ValueError("truncated execution history has no safe continuation")
            execution_volume, trade_count = aggregate_execution_interval(
                pages, symbol=symbol, start_ms=start_ms, end_ms=start_ms + 300_000,
            )
            audit = validate_candle_interval(
                candles, symbol=symbol, start_ms=start_ms, execution_volume=execution_volume,
            )
            audit.update({
                "interval_end_ms": start_ms + 300_000, "public_execution_count": trade_count,
                "page_count": page_number, "truncated_pages_followed": page_number - 1,
                "candle_response_sha256": sha256_bytes(candle_body),
                "execution_page_hashes": ";".join(
                    sha256_bytes((raw_dir / f"{symbol}_{start_ms}_executions_{index:03d}.json").read_bytes())
                    for index in range(page_number)
                ),
                "current_calibration_used_for_alpha": False,
            })
            rows.append(audit)
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def build(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger, semantic_raw = acquire_sources(output)
    calibration = acquire_calibration(output)
    if len(calibration) < 12 or not calibration["exact_match"].all():
        raise ValueError("base-volume calibration did not pass every frozen interval")
    semantic = validate_semantic_versions(semantic_raw)
    write_csv(output / "KRAKEN_CANDLE_VOLUME_SOURCE_LEDGER.csv", ledger)
    write_csv(output / "KRAKEN_CANDLE_VOLUME_CALIBRATION.csv", calibration)
    write_csv(output / "KRAKEN_CANDLE_VOLUME_AUTHORITY.csv", semantic)
    authority_hash = canonical_hash({
        "calibration": calibration.to_dict("records"),
        "semantic_rows": semantic.astype(str).to_dict("records"),
        "source_hashes": sorted(row["sha256"] for row in ledger),
        "field": "base_volume", "exact_quote_volume": "unavailable",
    })
    (output / "KRAKEN_CANDLE_VOLUME_VALIDATION.md").write_text(
        "# Kraken Candle Volume Validation\n\n"
        f"Authority hash: `{authority_hash}`.\n\n"
        f"All {len(calibration)} frozen five-minute intervals matched complete public execution quantity exactly. "
        "The public execution interval is half-open and every execution UID was unique. The observed PF contract "
        "specification rows express base currency and minimum lot in base units. Historical semantic rows are "
        "versioned by archived official source; events preceding a symbol's first observed official version or "
        "with inconsistent base/min-lot observations must fail closed.\n\n"
        "`volume` is authorized as base quantity only for listed PF symbol/semantic intervals. Exact quote volume "
        "is unavailable. Current 2026 calibration was not joined to alpha, strategy outcomes, parameters, or ranks.\n",
        encoding="utf-8",
    )
    (output / "LIQUIDITY_PROXY_CONTRACT.md").write_text(
        "# Liquidity Proxy Contract\n\n"
        "Field: `close_based_usd_volume_proxy`. Five-minute value is `trade_close_5m * verified_base_volume_5m`; "
        "daily value is its UTC-day sum. Membership is top 100 by the prior 30 calendar days' median, requiring "
        "at least 20 valid prior days and ranking once daily using data through the prior UTC day.\n\n"
        "This is a causal cohort-hygiene proxy. It is not exact quote volume, not traded USD notional, not capacity, "
        "not spread/depth evidence, and not executable-liquidity or slippage evidence.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
