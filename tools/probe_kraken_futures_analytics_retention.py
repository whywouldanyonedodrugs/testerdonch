#!/usr/bin/env python3
"""Bounded, outcome-free retention probe for Kraken Futures analytics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlparse, parse_qs

import requests


TASK_ID = "donch_bt_stage_7a_minimal_futures_analytics_retention_20260717_v2"
BASE_URL = "https://futures.kraken.com/api/charts/v1/analytics"
PROTECTED_START = 1767225600  # 2026-01-01T00:00:00Z
MAX_REQUESTS = 48
MAX_BYTES = 50 * 1024 * 1024
SYMBOLS = ("PF_XBTUSD", "PF_ETHUSD")
ANALYTICS_TYPES = ("open-interest", "funding", "liquidation-volume", "future-basis")
WINDOWS = (
    ("2023", 1686787200, 1686873600),
    ("2024", 1718409600, 1718496000),
    ("2025", 1752537600, 1752624000),
)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    symbol: str
    analytics_type: str
    year: str
    interval: int
    since: int
    to: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso(ts: int | float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_matrix() -> list[RequestSpec]:
    rows: list[RequestSpec] = []
    for symbol in SYMBOLS:
        for metric in ANALYTICS_TYPES:
            for year, since, to in WINDOWS:
                rid = f"{symbol.lower()}__{metric}__{year}"
                rows.append(RequestSpec(rid, symbol, metric, year, 3600, since, to))
    if len(rows) != 24 or len({r.request_id for r in rows}) != 24:
        raise RuntimeError("frozen request matrix is not exactly 24 unique cells")
    return rows


def build_url(spec: RequestSpec) -> str:
    if spec.symbol not in SYMBOLS or spec.analytics_type not in ANALYTICS_TYPES:
        raise ValueError("request outside frozen symbol/type allowlist")
    if spec.interval != 3600 or spec.since >= spec.to or spec.to > PROTECTED_START:
        raise ValueError("request lacks a valid explicit pre-protected bound")
    query = urlencode({"interval": spec.interval, "since": spec.since, "to": spec.to})
    url = f"{BASE_URL}/{spec.symbol}/{spec.analytics_type}?{query}"
    validate_url(url)
    return url


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "futures.kraken.com":
        raise ValueError("non-authorized host")
    if not parsed.path.startswith("/api/charts/v1/analytics/"):
        raise ValueError("non-authorized endpoint")
    query = parse_qs(parsed.query)
    if set(query) != {"interval", "since", "to"}:
        raise ValueError("request must contain only explicit interval/since/to")
    since, to = int(query["since"][0]), int(query["to"][0])
    if int(query["interval"][0]) != 3600 or since >= to or to > PROTECTED_START:
        raise ValueError("unsafe request bounds")


def schema_keys(value: Any, prefix: str = "") -> list[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.add(name)
            keys.update(schema_keys(child, name))
    elif isinstance(value, list) and value:
        keys.update(schema_keys(value[0], f"{prefix}[]"))
    return sorted(keys)


def extract_timestamps(payload: Any) -> list[float]:
    if not isinstance(payload, dict):
        return []
    result = payload.get("result")
    if not isinstance(result, dict):
        return []
    values = result.get("timestamp", [])
    if not isinstance(values, list):
        return []
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def normalize_timestamps(values: list[float]) -> tuple[list[float], str]:
    if not values:
        return [], "unavailable"
    median = sorted(values)[len(values) // 2]
    if median >= 100_000_000_000:
        return [v / 1000 for v in values], "inferred_milliseconds"
    if median >= 100_000_000:
        return values, "verified_documented_seconds"
    return values, "ambiguous"


def count_null_nonfinite(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, float) and not math.isfinite(value):
        return 1
    if isinstance(value, dict):
        return sum(count_null_nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_null_nonfinite(v) for v in value)
    return 0


def classify_response(spec: RequestSpec, status: int, content_type: str, body: bytes) -> dict[str, Any]:
    row: dict[str, Any] = {
        "classification": "request_failed", "schema_keys": "", "row_count": 0,
        "first_timestamp": "", "last_timestamp": "", "minimum_timestamp": "",
        "maximum_timestamp": "", "timestamp_unit_status": "unavailable",
        "null_nonfinite_count": "", "server_error_fields": "", "lower_bound_honored": False,
        "upper_bound_honored": False, "protected_2026_rows": 0,
    }
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        if status in {400, 404}:
            row["classification"] = "unsupported_type_or_symbol"
        return row
    row["schema_keys"] = "|".join(schema_keys(payload))
    errors = payload.get("errors", []) if isinstance(payload, dict) else []
    row["server_error_fields"] = json.dumps(errors, sort_keys=True, separators=(",", ":"))
    values, unit = normalize_timestamps(extract_timestamps(payload))
    row["timestamp_unit_status"] = unit
    row["row_count"] = len(values)
    if values:
        row.update({
            "first_timestamp": iso(values[0]), "last_timestamp": iso(values[-1]),
            "minimum_timestamp": iso(min(values)), "maximum_timestamp": iso(max(values)),
            "lower_bound_honored": min(values) >= spec.since,
            "upper_bound_honored": max(values) <= spec.to,
            "protected_2026_rows": sum(v >= PROTECTED_START for v in values),
        })
        # Do not inspect values when a server returns protected timestamps.
        if row["protected_2026_rows"]:
            row["classification"] = "recent_only_or_bound_ignored"
            return row
        row["null_nonfinite_count"] = count_null_nonfinite(payload)
        if unit == "ambiguous":
            row["classification"] = "schema_or_unit_ambiguous"
        elif not row["lower_bound_honored"] or not row["upper_bound_honored"]:
            row["classification"] = "recent_only_or_bound_ignored"
        elif status == 200 and not errors:
            row["classification"] = "verified_historical_rows"
        return row
    if status == 200 and isinstance(payload, dict) and isinstance(payload.get("result"), dict) and not errors:
        row["classification"] = "empty_valid_response"
        row["lower_bound_honored"] = True
        row["upper_bound_honored"] = True
    elif status in {400, 404}:
        row["classification"] = "unsupported_type_or_symbol"
    elif status == 200:
        row["classification"] = "schema_or_unit_ambiguous"
    return row


class Budget:
    def __init__(self) -> None:
        self.requests = 0
        self.bytes = 0

    def charge(self, byte_count: int) -> None:
        if self.requests + 1 > MAX_REQUESTS or self.bytes + byte_count > MAX_BYTES:
            raise RuntimeError("authorized request or byte budget exceeded")
        self.requests += 1
        self.bytes += byte_count


def fetch(url: str, timeout: float = 30) -> tuple[int, dict[str, str], bytes]:
    validate_url(url)
    response = requests.get(url, headers={"Accept": "application/json", "User-Agent": "donch-retention-probe/1.0"}, timeout=timeout)
    return response.status_code, dict(response.headers), response.content


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = columns or sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_pass(root: Path, pass_no: int, matrix: list[RequestSpec], budget: Budget,
             reader: Callable[[str], tuple[int, dict[str, str], bytes]] = fetch) -> list[dict[str, Any]]:
    rows = []
    out = root / "raw" / "responses" / f"pass{pass_no}"
    out.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(matrix):
        url = build_url(spec)
        started = utc_now()
        try:
            status, headers, body = reader(url)
            budget.charge(len(body))
            body_path = out / f"{spec.request_id}.body"
            headers_path = out / f"{spec.request_id}.headers.json"
            body_path.write_bytes(body)
            headers_path.write_text(json.dumps(headers, indent=2, sort_keys=True) + "\n")
            parsed = classify_response(spec, status, headers.get("Content-Type", ""), body)
            row = {**asdict(spec), "pass": pass_no, "request_url": url, "request_started_utc": started,
                   "http_status": status, "response_content_type": headers.get("Content-Type", ""),
                   "response_bytes": len(body), "response_sha256": sha256_bytes(body), **parsed}
        except requests.RequestException as exc:
            budget.charge(0)
            row = {**asdict(spec), "pass": pass_no, "request_url": url, "request_started_utc": started,
                   "http_status": "", "response_content_type": "", "response_bytes": 0,
                   "response_sha256": "", "classification": "request_failed", "schema_keys": "",
                   "row_count": 0, "first_timestamp": "", "last_timestamp": "",
                   "minimum_timestamp": "", "maximum_timestamp": "", "timestamp_unit_status": "unavailable",
                   "null_nonfinite_count": "", "server_error_fields": type(exc).__name__,
                   "lower_bound_honored": False, "upper_bound_honored": False, "protected_2026_rows": 0}
        rows.append(row)
        if index + 1 < len(matrix):
            time.sleep(0.25)
    return rows


def compare_replay(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    right = {row["request_id"]: row for row in second}
    result = []
    fields = ("classification", "schema_keys", "row_count", "minimum_timestamp", "maximum_timestamp")
    for left in first:
        other = right[left["request_id"]]
        stable = all(left.get(k) == other.get(k) for k in fields)
        result.append({"request_id": left["request_id"], "structurally_stable": stable,
                       "response_hash_equal": left.get("response_sha256") == other.get("response_sha256"),
                       "changed_fields": "|".join(k for k in fields if left.get(k) != other.get(k)),
                       "pass1_sha256": left.get("response_sha256", ""), "pass2_sha256": other.get("response_sha256", "")})
    return result


def decision(matrix: list[dict[str, Any]], replay: list[dict[str, Any]]) -> str:
    valid = sum(r["classification"] == "verified_historical_rows" and bool(r["upper_bound_honored"]) for r in matrix)
    stable = all(r["structurally_stable"] for r in replay)
    if valid == 24 and stable:
        return "ready_for_bounded_historical_analytics_audit"
    if valid == 0:
        return "historical_public_analytics_unavailable"
    return "partial_historical_analytics_requires_review"


def manifest(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"):
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path),
                     "distribution": "local_only" if path.parts[-3:-2] == ("responses",) or "raw/responses" in path.as_posix() else "reviewable"})
    payload = {"task_id": TASK_ID, "generated_utc": utc_now(), "artifacts": rows}
    (root / "ARTIFACT_MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    budget = Budget()
    first = run_pass(root, 1, matrix, budget)
    second = run_pass(root, 2, matrix, budget)
    ledger = first + second
    replay = compare_replay(first, second)
    final_decision = decision(first, replay)
    write_csv(root / "KRAKEN_ANALYTICS_MINIMAL_REQUEST_LEDGER.csv", ledger)
    retention = [{**{k: row[k] for k in ("request_id", "symbol", "analytics_type", "year", "since", "to")},
                  "classification": row["classification"], "row_count": row["row_count"],
                  "minimum_timestamp": row["minimum_timestamp"], "maximum_timestamp": row["maximum_timestamp"],
                  "upper_bound_honored": row["upper_bound_honored"], "protected_2026_rows": row["protected_2026_rows"]}
                 for row in first]
    write_csv(root / "KRAKEN_ANALYTICS_RETENTION_MATRIX.csv", retention)
    write_csv(root / "KRAKEN_ANALYTICS_REPLAY_COMPARISON.csv", replay)
    docs = []
    for path in sorted((root / "raw" / "docs").glob("*")):
        docs.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path),
                     "authority": "official_kraken_documentation", "access_note": "live page access restricted; schema snapshot records official indexed content"})
    write_csv(root / "KRAKEN_ANALYTICS_DOC_SNAPSHOT_INDEX.csv", docs)
    type_status = {
        metric: (
            "bounded_probe_historical_rows_verified_not_full_history"
            if all(r["classification"] == "verified_historical_rows" for r in first if r["analytics_type"] == metric)
            else "bounded_probe_empty_or_partial"
        )
        for metric in ANALYTICS_TYPES
    }
    readiness = [{"capability": metric, "prior_status": "unavailable" if metric != "funding" else "narrow_late_2025_exact_only",
                  "probe_status": type_status[metric], "authority_decision": final_decision,
                  "economic_use_authorized": "no"} for metric in ANALYTICS_TYPES]
    write_csv(root / "HYPOTHESIS_DATA_READINESS_UPDATE.csv", readiness)
    (root / "KRAKEN_ANALYTICS_SCHEMA_AND_UNIT_NOTE.md").write_text(
        "# Kraken Analytics Schema and Unit Note\n\n"
        "- Request `since` and `to`: **verified** as epoch seconds from official documentation.\n"
        "- Request `interval=3600`: **verified** as a documented allowed interval.\n"
        "- Response timestamp unit: recorded per cell as documented seconds, inferred milliseconds, ambiguous, or unavailable.\n"
        "- Open-interest value unit: **unavailable**.\n- Funding value unit and sign: **unavailable**.\n"
        "- Liquidation-volume value unit: **unavailable**.\n- Future-basis value unit and sign: **unavailable**.\n\n"
        "No value semantics were inferred from prices, returns, or outcomes.\n")
    counts = {c: sum(r["classification"] == c for r in first) for c in sorted({r["classification"] for r in first})}
    (root / "KRAKEN_ANALYTICS_AUTHORITY_DECISION.md").write_text(
        f"# Kraken Analytics Authority Decision\n\nDecision: `{final_decision}`\n\n"
        f"First-pass classifications: `{json.dumps(counts, sort_keys=True)}`.\n\n"
        f"Requests: `{budget.requests}`; downloaded response bytes: `{budget.bytes}`. "
        f"Structurally stable replay cells: `{sum(r['structurally_stable'] for r in replay)}/24`.\n\n"
        "This decision authorizes no full acquisition, signal design, or economic work.\n")
    (root / "VALIDATION.md").write_text(
        f"# Validation\n\n- Frozen first pass: 24 requests.\n- Exact replay: 24 requests.\n"
        f"- Request budget used: {budget.requests}/48.\n- Response bytes: {budget.bytes}/{MAX_BYTES}.\n"
        f"- Protected rows returned: {sum(int(r['protected_2026_rows']) for r in ledger)}.\n"
        "- Economic outputs computed: no.\n- Protected outcomes inspected: no.\n")
    (root / "REVIEW.md").write_text(
        "# Independent Review Checklist\n\nPending independent read-only review of documentation provenance, exact matrix, "
        "bounds, classifications, replay, capability claims, and absence of economic/protected outputs.\n")
    (root / "COMPLETION.md").write_text(
        f"# Completion\n\nStatus: `{final_decision}`\n\nThe bounded source-authority probe completed without economic analysis.\n")
    next_action = ("A separately authorized bounded historical analytics audit may be designed; do not acquire full history yet."
                   if final_decision == "ready_for_bounded_historical_analytics_audit" else
                   "Do not begin full historical acquisition. Resolve the partial/unavailable public analytics authority first.")
    (root / "NEXT_ACTION.md").write_text(f"# Next Action\n\n{next_action}\n")
    manifest(root)
    print(json.dumps({"decision": final_decision, "requests": budget.requests, "bytes": budget.bytes,
                      "valid_cells": sum(r["classification"] == "verified_historical_rows" for r in first)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
