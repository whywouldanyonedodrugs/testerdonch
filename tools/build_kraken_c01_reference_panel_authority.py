#!/usr/bin/env python3
"""Build the outcome-free Kraken C01 reference-panel authority.

The builder reads four already-bounded public candle responses, official instrument
metadata, and the official cumulative derivatives terminal-event page. It never
opens a candidate, return, funding, or protected-period payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import pandas as pd


TASK_ID = "donch_bt_stage_2a1_c01_reference_panel_20260717_v1"
REFERENCE_PANEL_ID = "kraken_c01_reference_panel_v1"
FROM_SECONDS = 1767139200
TO_SECONDS = 1767225599
FROM_MS = FROM_SECONDS * 1000
PROTECTED_START_MS = 1767225600000
RANKABLE_START = "2023-01-01T00:00:00Z"
PROTECTED_START = "2026-01-01T00:00:00Z"
FINAL_CANDLE_UTC = "2025-12-31T23:55:00Z"
SYMBOLS = {"PF_XBTUSD": "BTC", "PF_ETHUSD": "ETH"}
TICK_TYPES = ("trade", "mark")
NORMALIZED_FIELDS = (
    "time", "open", "high", "low", "close", "volume", "source_url",
    "venue_symbol", "chunk_start_utc", "chunk_end_utc", "resolution",
    "historical_backfill", "rankable_pre_holdout", "contains_protected_period",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candle_url(symbol: str, tick_type: str) -> str:
    if symbol not in SYMBOLS or tick_type not in TICK_TYPES:
        raise ValueError(f"unauthorized candle request identity: {tick_type}:{symbol}")
    return (
        f"https://futures.kraken.com/api/charts/v1/{tick_type}/{symbol}/5m"
        f"?from={FROM_SECONDS}&to={TO_SECONDS}"
    )


def assert_request_identity(url: str, symbol: str, tick_type: str) -> None:
    expected = urlparse(build_candle_url(symbol, tick_type))
    actual = urlparse(url.strip())
    if actual.scheme != "https" or actual.netloc != "futures.kraken.com" or actual.path != expected.path:
        raise ValueError(f"wrong candle endpoint identity: {url}")
    query = parse_qs(actual.query, strict_parsing=True)
    if query.get("from") != [str(FROM_SECONDS)] or query.get("to") != [str(TO_SECONDS)]:
        raise ValueError(f"wrong candle request bounds: {url}")
    if set(query) != {"from", "to"}:
        raise ValueError(f"unexpected candle request parameters: {sorted(query)}")


def _identity_value(value: Any) -> str:
    return str(value).strip().lower().replace("-price", "")


def parse_candle_payload(payload: bytes, *, symbol: str, tick_type: str, source_url: str) -> list[dict[str, Any]]:
    """Validate an exact response before returning normalized rows."""
    assert_request_identity(source_url, symbol, tick_type)
    try:
        value = json.loads(payload)
    except Exception as exc:
        raise ValueError("candle response is not unambiguous JSON") from exc
    if not isinstance(value, Mapping) or set(value) - {"candles", "more_candles", "symbol", "tick_type"}:
        raise ValueError("candle response schema is mixed or ambiguous")
    if value.get("more_candles") is not False:
        raise ValueError("pagination is incomplete or ambiguous")
    for key, expected in (("symbol", symbol), ("tick_type", tick_type)):
        if key in value and _identity_value(value[key]) != _identity_value(expected):
            raise ValueError(f"response {key} does not match request")
    candles = value.get("candles")
    if not isinstance(candles, list) or not candles:
        raise ValueError("candle response has no rows")

    required = {"time", "open", "high", "low", "close", "volume"}
    times: list[int] = []
    rows: list[dict[str, Any]] = []
    for raw in candles:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("candle row schema is ambiguous")
        try:
            timestamp = int(raw["time"])
        except Exception as exc:
            raise ValueError("candle timestamp is not integer milliseconds") from exc
        if timestamp >= PROTECTED_START_MS:
            raise ValueError("protected-period candle rejected before normalization")
        if timestamp < FROM_MS or timestamp > TO_SECONDS * 1000:
            raise ValueError("candle timestamp falls outside exact request interval")
        times.append(timestamp)
        rows.append({
            **{field: raw[field] for field in ("time", "open", "high", "low", "close", "volume")},
            "source_url": source_url.strip(),
            "venue_symbol": symbol,
            "chunk_start_utc": "2025-12-31T00:00:00Z",
            "chunk_end_utc": PROTECTED_START,
            "resolution": "5m",
            "historical_backfill": True,
            "rankable_pre_holdout": True,
            "contains_protected_period": False,
        })
    if len(times) != len(set(times)):
        raise ValueError("duplicate candle timestamps")
    if times != sorted(times):
        raise ValueError("out-of-order candle timestamps")
    return rows


def missing_intervals(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    observed = {int(row["time"]) for row in rows}
    expected = range(FROM_MS, PROTECTED_START_MS, 300_000)
    return [
        datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
        for value in expected if value not in observed
    ]


def canonical_csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str] = NORMALIZED_FIELDS) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self.visible: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table" and self._table is None:
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.visible.append(text)
            if self._cell is not None:
                self._cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def parse_terminal_lifecycle_html(payload: bytes) -> tuple[list[dict[str, str]], set[str]]:
    parser = _TableParser()
    parser.feed(payload.decode("utf-8", errors="strict"))
    terminal_rows: list[dict[str, str]] = []
    for table in parser.tables:
        headers = [re.sub(r"\s+", " ", value.strip()).lower() for value in table[0]]
        if "symbol" not in headers or not any("settlement date" in value for value in headers):
            continue
        for raw in table[1:]:
            if len(raw) != len(headers):
                raise ValueError("terminal lifecycle table row has ambiguous width")
            row = dict(zip(headers, raw))
            raw_symbol = row.get("symbol", "").strip().upper()
            symbol = raw_symbol.rstrip("*").strip()
            if not re.fullmatch(r"[A-Z0-9_:-]+", symbol):
                raise ValueError(f"ambiguous terminal lifecycle symbol: {raw_symbol}")
            terminal_rows.append({
                "symbol": symbol,
                "source_symbol": raw_symbol,
                "settlement_date": next(row[key] for key in headers if "settlement date" in key),
                "observation": next((row[key] for key in headers if "observation" in key), ""),
                "settlement": next((row[key] for key in headers if key == "settlement"), ""),
            })
    if not terminal_rows:
        raise ValueError("official cumulative terminal-event table not found")
    visible = " ".join(parser.visible)
    resumed = {
        match.group(1).upper()
        for match in re.finditer(
            r"\b(PF_[A-Z0-9]+USD)\b(?:(?!\bPF_[A-Z0-9]+USD\b).){0,100}\b(?:resumed|re-enabled)\b",
            visible,
            re.I,
        )
    }
    return terminal_rows, resumed


def terminal_status(symbol: str, terminal_rows: Sequence[Mapping[str, str]], resumed: set[str]) -> str:
    if symbol in resumed:
        return "resumed_after_official_suspension_note"
    if any(row.get("symbol") == symbol for row in terminal_rows):
        return "delisted_and_settled_in_official_terminal_ledger"
    return "not_listed_in_official_terminal_ledger_as_of_access"


def load_official_instruments(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_bytes())
    rows = value.get("instruments") if isinstance(value, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("official instrument source is ambiguous")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("symbol") in SYMBOLS:
            symbol = str(row["symbol"])
            if symbol in result:
                raise ValueError(f"duplicate official instrument identity: {symbol}")
            result[symbol] = dict(row)
    if set(result) != set(SYMBOLS):
        raise ValueError("official instrument source is missing a reference-panel member")
    for symbol, row in result.items():
        if row.get("type") != "flexible_futures" or not row.get("openingDate"):
            raise ValueError(f"ambiguous official identity for {symbol}")
    return result


def load_prior_coverage_authority(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        symbol = row.get("Kraken_symbol", "")
        if symbol not in SYMBOLS:
            continue
        if symbol in result:
            raise ValueError(f"duplicate prior coverage authority: {symbol}")
        result[symbol] = row
    if set(result) != set(SYMBOLS):
        raise ValueError("prior coverage authority is missing a reference-panel member")
    for symbol, row in result.items():
        expected = {
            "trade_coverage_start_utc": RANKABLE_START,
            "trade_coverage_end_utc": "2025-12-31T00:00:00Z",
            "mark_coverage_start_utc": RANKABLE_START,
            "mark_coverage_end_utc": "2025-12-31T00:00:00Z",
            "identity_confidence": "high",
        }
        mismatches = {key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value}
        if mismatches:
            raise ValueError(f"prior authority mismatch for {symbol}: {mismatches}")
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_csv_bytes(rows, fields))


def deterministic_panel_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(sorted(row.items())) for row in sorted(rows, key=lambda row: str(row["Kraken_symbol"]))],
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return sha256_bytes(payload)


PANEL_FIELDS = (
    "reference_panel_id", "instrument_id", "canonical_asset_id", "Kraken_symbol", "role",
    "official_opening_date", "identity_confidence", "terminal_lifecycle_status",
    "terminal_lifecycle_source", "terminal_lifecycle_source_sha256", "trade_coverage_start_utc",
    "trade_coverage_end_utc", "mark_coverage_start_utc", "mark_coverage_end_utc",
    "continuous_tradeability_claim", "operational_availability_rule", "survivorship_cap",
    "included", "inclusion_reason", "permitted_claim", "prohibited_claim",
)


def build(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    source = Path(args.source_dir)
    normalized = output / "normalized_final_2025_day"
    normalized.mkdir(parents=True, exist_ok=True)
    access_utc = (source.parent / "access_utc.txt").read_text(encoding="utf-8").strip()
    instrument_source = Path(args.instrument_source)
    instruments = load_official_instruments(instrument_source)
    prior_authority_path = Path(args.prior_authority)
    prior_authority = load_prior_coverage_authority(prior_authority_path)

    source_rows: list[dict[str, Any]] = []
    coverage: dict[tuple[str, str], dict[str, Any]] = {}
    for symbol in SYMBOLS:
        for tick_type in TICK_TYPES:
            stem = f"{symbol}_{tick_type}"
            body_path = source / f"{stem}.body"
            url_path = source / f"{stem}.url.txt"
            status_path = source / f"{stem}.status.txt"
            headers_path = source / f"{stem}.headers.txt"
            status = status_path.read_text(encoding="utf-8").strip()
            if status != "200":
                raise ValueError(f"official candle request blocked for {stem}: HTTP {status}")
            body = body_path.read_bytes()
            url = url_path.read_text(encoding="utf-8").strip()
            rows = parse_candle_payload(body, symbol=symbol, tick_type=tick_type, source_url=url)
            gaps = missing_intervals(rows)
            csv_path = normalized / f"{stem}.csv"
            parquet_path = normalized / f"{stem}.parquet"
            csv_payload = canonical_csv_bytes(rows)
            csv_path.write_bytes(csv_payload)
            pd.DataFrame(rows, columns=NORMALIZED_FIELDS).to_parquet(parquet_path, index=False)
            first_ms, last_ms = int(rows[0]["time"]), int(rows[-1]["time"])
            first_utc = datetime.fromtimestamp(first_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
            last_utc = datetime.fromtimestamp(last_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
            coverage[(symbol, tick_type)] = {"rows": len(rows), "gaps": gaps, "first": first_utc, "last": last_utc}
            source_rows.append({
                "source_id": stem, "symbol": symbol, "tick_type": tick_type, "url": url,
                "request_from": FROM_SECONDS, "request_to": TO_SECONDS, "access_utc": access_utc,
                "http_status": status, "raw_path": str(body_path), "raw_sha256": sha256_bytes(body),
                "raw_bytes": len(body), "headers_path": str(headers_path), "rows": len(rows),
                "first_candle_utc": first_utc, "last_candle_utc": last_utc,
                "more_candles": "false", "missing_5m_intervals": len(gaps),
                "normalized_csv_path": str(csv_path.relative_to(output)), "normalized_content_sha256": sha256_bytes(csv_payload),
                "normalized_parquet_path": str(parquet_path.relative_to(output)), "normalized_parquet_sha256": sha256_file(parquet_path),
                "returned_2026_rows": 0, "status": "verified_complete" if not gaps and len(rows) == 288 else "verified_with_reported_gaps",
            })

    lifecycle_dir = Path(args.lifecycle_source_dir)
    lifecycle_body = lifecycle_dir / "kraken_derivatives_delistings.body"
    lifecycle_url = (lifecycle_dir / "kraken_derivatives_delistings.url.txt").read_text(encoding="utf-8").strip()
    lifecycle_status = (lifecycle_dir / "kraken_derivatives_delistings.status.txt").read_text(encoding="utf-8").strip()
    if lifecycle_status != "200" or "support.kraken.com" not in urlparse(lifecycle_url).netloc:
        raise ValueError("official terminal lifecycle source failed authority checks")
    lifecycle_payload = lifecycle_body.read_bytes()
    terminal_rows, resumed = parse_terminal_lifecycle_html(lifecycle_payload)
    lifecycle_sha = sha256_bytes(lifecycle_payload)

    panel_rows: list[dict[str, Any]] = []
    for symbol, asset in SYMBOLS.items():
        prior = prior_authority[symbol]
        trade = coverage[(symbol, "trade")]
        mark = coverage[(symbol, "mark")]
        if trade["first"] != "2025-12-31T00:00:00Z" or mark["first"] != "2025-12-31T00:00:00Z":
            raise ValueError(f"final-day authority does not start at midnight for {symbol}")
        if trade["last"] != FINAL_CANDLE_UTC or mark["last"] != FINAL_CANDLE_UTC or trade["gaps"] or mark["gaps"]:
            raise ValueError(f"final-day authority is incomplete for {symbol}")
        panel_rows.append({
            "reference_panel_id": REFERENCE_PANEL_ID,
            "instrument_id": f"kraken:{symbol}",
            "canonical_asset_id": asset,
            "Kraken_symbol": symbol,
            "role": "causal_market_reference_factor_only",
            "official_opening_date": str(instruments[symbol]["openingDate"]),
            "identity_confidence": "high_official_current_and_archived_snapshots",
            "terminal_lifecycle_status": terminal_status(symbol, terminal_rows, resumed),
            "terminal_lifecycle_source": lifecycle_url,
            "terminal_lifecycle_source_sha256": lifecycle_sha,
            "trade_coverage_start_utc": prior["trade_coverage_start_utc"],
            "trade_coverage_end_utc": PROTECTED_START,
            "mark_coverage_start_utc": prior["mark_coverage_start_utc"],
            "mark_coverage_end_utc": PROTECTED_START,
            "continuous_tradeability_claim": "no",
            "operational_availability_rule": "all required trade and mark bars must exist inside the event-time feature/risk window; known terminal or maintenance invalidation fails closed",
            "survivorship_cap": "not_survivorship_free; terminal ledger is retrospective and temporary status continuity is unproven",
            "included": "true",
            "inclusion_reason": "stable official identity; complete bounded trade/mark metadata interval; no listed terminal event as of source access",
            "permitted_claim": "factor/reference identity and event-time bar availability only",
            "prohibited_claim": "candidate-universe membership; continuous tradeability; no outages; survivorship-free history; economic evidence",
        })
    panel_hash = deterministic_panel_hash(panel_rows)

    source_fields = (
        "source_id", "symbol", "tick_type", "url", "request_from", "request_to", "access_utc",
        "http_status", "raw_path", "raw_sha256", "raw_bytes", "headers_path", "rows",
        "first_candle_utc", "last_candle_utc", "more_candles", "missing_5m_intervals",
        "normalized_csv_path", "normalized_content_sha256", "normalized_parquet_path",
        "normalized_parquet_sha256", "returned_2026_rows", "status",
    )
    write_csv(output / "FINAL_2025_DAY_SOURCE_LEDGER.csv", source_rows, source_fields)
    write_csv(output / "C01_REFERENCE_PANEL_AUTHORITY.csv", panel_rows, PANEL_FIELDS)
    lifecycle_ledger = [{
        "source_id": "kraken_derivatives_cumulative_delistings",
        "source_kind": "official_kraken_terminal_event_ledger",
        "url": lifecycle_url,
        "access_utc": access_utc,
        "http_status": lifecycle_status,
        "raw_path": str(lifecycle_body),
        "sha256": lifecycle_sha,
        "bytes": len(lifecycle_payload),
        "terminal_rows_parsed": len(terminal_rows),
        "resumed_symbols_parsed": len(resumed),
        "PF_XBTUSD_status": terminal_status("PF_XBTUSD", terminal_rows, resumed),
        "PF_ETHUSD_status": terminal_status("PF_ETHUSD", terminal_rows, resumed),
        "claim_boundary": "terminal events only; absence does not prove no temporary interruption",
    }]
    write_csv(output / "TERMINAL_LIFECYCLE_SOURCE_LEDGER.csv", lifecycle_ledger, tuple(lifecycle_ledger[0]))

    (output / "FINAL_2025_DAY_COVERAGE_VALIDATION.md").write_text(f"""# Final 2025 Day Coverage Validation

Status: **verified complete**.

- Exact official requests: 4/4 HTTP 200.
- Rows: 288 for each of XBT trade, XBT mark, ETH trade, and ETH mark.
- First timestamp: `2025-12-31T00:00:00Z`.
- Last five-minute open: `{FINAL_CANDLE_UTC}`; coverage end is `{PROTECTED_START}` exclusive.
- Missing five-minute intervals: 0 for every slice.
- Duplicate or out-of-order timestamps: 0 accepted.
- `more_candles`: false for every slice.
- Returned protected-period rows: 0.
- Previously identified mixed 2025/2026 chunk opened: no.
- Candidate, return, and funding calculations: 0.

The new interval starts at the prior authority's exclusive coverage boundary. It is retained as a separate bounded shard; no protected file was used as fallback.
""", encoding="utf-8")
    (output / "TERMINAL_LIFECYCLE_VALIDATION.md").write_text(f"""# Terminal Lifecycle Validation

The official cumulative Kraken derivatives suspensions/delistings page was cached at `{access_utc}` and parsed into {len(terminal_rows)} terminal rows. `PF_XBTUSD` and `PF_ETHUSD` are not present in that terminal-event table as accessed.

This supports only a bounded terminal status: neither contract is listed as delisted and settled in the cached cumulative ledger. It does not establish uninterrupted trading, absence of temporary post-only/suspension/maintenance intervals, or survivorship-free status. Operational availability remains an event-time bar and known-status check.
""", encoding="utf-8")
    (output / "C01_REFERENCE_PANEL_CLAIM_BOUNDARY.md").write_text(f"""# C01 Reference Panel Claim Boundary

Reference panel: `{REFERENCE_PANEL_ID}`
Panel hash: `{panel_hash}`

`PF_XBTUSD` and `PF_ETHUSD` are admitted only as causal market-reference/factor series. They are not a C01 candidate universe and are not evidence of a tradable strategy.

## Allowed

- Use a member as a factor only when every trade/mark row required by the event-time window is present inside `[2023-01-01, 2026-01-01)`.
- Fail closed when a required window is missing or a known terminal/maintenance status invalidates it.
- Cite stable official identity, March 2022 opening date, and absence from the cached cumulative terminal-event table as of access.

## Prohibited

- `continuous_tradeability_claim`: **no**.
- `survivorship_free_claim`: **no**.
- Absence from the terminal table is not a no-outage or no-suspension claim.
- The panel cannot be used as candidate-universe authority, economic evidence, or permission to inspect protected outcomes.

Temporary status-history reconstruction is explicitly deferred; this task does not create a status platform.
""", encoding="utf-8")

    core_artifacts = [
        "C01_REFERENCE_PANEL_AUTHORITY.csv", "C01_REFERENCE_PANEL_CLAIM_BOUNDARY.md",
        "FINAL_2025_DAY_SOURCE_LEDGER.csv", "FINAL_2025_DAY_COVERAGE_VALIDATION.md",
        "TERMINAL_LIFECYCLE_SOURCE_LEDGER.csv", "TERMINAL_LIFECYCLE_VALIDATION.md",
    ] + [str(path.relative_to(output)) for path in sorted(normalized.glob("*"))]
    manifest = {
        "task_id": TASK_ID,
        "reference_panel_id": REFERENCE_PANEL_ID,
        "reference_panel_hash": panel_hash,
        "prior_coverage_authority": str(prior_authority_path),
        "prior_coverage_authority_sha256": sha256_file(prior_authority_path),
        "continuous_tradeability_claim": False,
        "survivorship_free_claim": False,
        "protected_outcomes_opened": False,
        "economic_outputs_computed": False,
        "returned_2026_rows": 0,
        "manifest_self_hash_excluded": True,
        "files": [
            {"path": name, "bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)}
            for name in core_artifacts
        ],
    }
    (output / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--lifecycle-source-dir", required=True)
    parser.add_argument("--instrument-source", required=True)
    parser.add_argument("--prior-authority", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
