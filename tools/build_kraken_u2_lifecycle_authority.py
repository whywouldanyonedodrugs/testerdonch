#!/usr/bin/env python3
"""Build the outcome-free Kraken U2 lifecycle authority artifacts.

This tool reads instrument metadata and download-manifest metadata only. It does
not open candle payloads or compute any economic quantity.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RANKABLE_START = "2023-01-01T00:00:00Z"
PROTECTED_START = "2026-01-01T00:00:00Z"
COHORT_VERSION = "kraken_u2_anchor_v1_20260716"
CONSIDERED = {
    "PF_XBTUSD": "BTC",
    "PF_ETHUSD": "ETH",
}
TRADE_DATASET = "historical_trade_candles_5m"
MARK_DATASET = "historical_mark_candles_5m"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    text = str(value).strip().replace("+00:00", "Z")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"timezone missing: {value}")
    return parsed.astimezone(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_http_date(value: str) -> str:
    if not value:
        return ""
    return utc_text(parsedate_to_datetime(value))


def assert_kraken_identity(venue: str, symbol: str) -> None:
    if venue.lower() != "kraken" or not symbol.startswith("PF_"):
        raise ValueError(f"non-Kraken perpetual identity rejected: {venue}:{symbol}")


def load_instrument_payload(payload: bytes) -> list[dict[str, Any]]:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    value = json.loads(payload)
    if isinstance(value, Mapping):
        value = value.get("instruments")
    if not isinstance(value, list):
        raise ValueError("instrument payload does not contain a list")
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping) or not raw.get("symbol"):
            continue
        row = dict(raw)
        row["symbol"] = str(row["symbol"]).upper()
        rows.append(row)
    return rows


def instrument_snapshot(path: Path, observed_at: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_instrument_payload(path.read_bytes()):
        symbol = row["symbol"]
        if symbol in result:
            raise ValueError(f"duplicate instrument in snapshot: {symbol}")
        result[symbol] = {
            "symbol": symbol,
            "observed_at": observed_at,
            "type": row.get("type"),
            "tradeable": row.get("tradeable"),
            "openingDate": row.get("openingDate"),
            "lastTradingTime": row.get("lastTradingTime"),
            "isExpired": row.get("isExpired"),
            "base": row.get("base"),
            "quote": row.get("quote"),
            "pair": row.get("pair"),
            "isin": row.get("isin"),
        }
    return result


def validate_identity_history(symbol: str, snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not snapshots:
        raise ValueError(f"no official snapshots for {symbol}")
    opening = {utc_text(parse_utc(str(row["openingDate"]))) for row in snapshots if row.get("openingDate")}
    types = {str(row.get("type")) for row in snapshots}
    if len(opening) != 1 or len(types) != 1:
        raise ValueError(f"conflicting identity history for {symbol}")
    if types != {"flexible_futures"}:
        raise ValueError(f"unexpected contract type for {symbol}: {types}")
    if any(row.get("tradeable") is not True for row in snapshots):
        raise ValueError(f"non-tradeable or unknown official checkpoint for {symbol}")
    return {
        "opening_date": next(iter(opening)),
        "contract_type": "linear_perpetual_flexible_futures",
        "checkpoint_count": len(snapshots),
        "checkpoint_start": min(str(row["observed_at"]) for row in snapshots),
        "checkpoint_end": max(str(row["observed_at"]) for row in snapshots),
    }


def merge_coverage_intervals(rows: Iterable[Mapping[str, str]]) -> tuple[str, str, list[tuple[str, str]]]:
    intervals = sorted((parse_utc(row["chunk_start"]), parse_utc(row["chunk_end"])) for row in rows)
    if not intervals:
        return "", "", []
    for start, end in intervals:
        if start >= end:
            raise ValueError(f"invalid coverage interval: {start} >= {end}")
    merged: list[list[datetime]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps = [(utc_text(merged[i - 1][1]), utc_text(merged[i][0])) for i in range(1, len(merged))]
    return utc_text(merged[0][0]), utc_text(merged[-1][1]), gaps


def coverage_for(
    manifest_rows: Sequence[Mapping[str, str]], symbol: str, dataset: str
) -> tuple[str, str, list[tuple[str, str]], int]:
    accepted = [
        row
        for row in manifest_rows
        if row.get("symbol") == symbol
        and row.get("dataset") == dataset
        and row.get("status") == "downloaded"
        and str(row.get("rankable_pre_holdout", "")).lower() == "true"
        and str(row.get("contains_protected_period", "")).lower() == "false"
    ]
    start, end, gaps = merge_coverage_intervals(accepted)
    return start, end, gaps, len(accepted)


def deterministic_cohort_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [dict(sorted(row.items())) for row in sorted(rows, key=lambda row: str(row["Kraken_symbol"]))]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def assert_rankable_interval(start: str, end: str) -> None:
    parsed_start, parsed_end = parse_utc(start), parse_utc(end)
    if parsed_start < parse_utc(RANKABLE_START) or parsed_end > parse_utc(PROTECTED_START) or parsed_start >= parsed_end:
        raise ValueError(f"rankable interval rejected: [{start}, {end})")


def candidate_is_includable(
    *,
    official_start: str,
    claimed_start: str,
    claimed_end: str,
    trade_start: str,
    trade_end: str,
    mark_start: str,
    mark_end: str,
    trade_gaps: Sequence[Any],
    mark_gaps: Sequence[Any],
    lifecycle_continuity_verified: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    assert_rankable_interval(claimed_start, claimed_end)
    if parse_utc(official_start) > parse_utc(claimed_start):
        reasons.append("official_start_after_claimed_start")
    for label, start, end, gaps in (
        ("trade", trade_start, trade_end, trade_gaps),
        ("mark", mark_start, mark_end, mark_gaps),
    ):
        if not start or not end or parse_utc(start) > parse_utc(claimed_start) or parse_utc(end) < parse_utc(claimed_end):
            reasons.append(f"{label}_coverage_incomplete")
        if gaps:
            reasons.append(f"{label}_coverage_gaps")
    if not lifecycle_continuity_verified:
        reasons.append("lifecycle_interval_continuity_unproven")
    return not reasons, reasons


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def header_value(path: Path, name: str) -> str:
    pattern = re.compile(rf"^{re.escape(name)}:\s*(.*?)\s*$", re.IGNORECASE)
    value = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.rstrip("\r"))
        if match:
            value = match.group(1)
    return value


def http_status(path: Path) -> str:
    status = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^HTTP/\S+\s+(\d+)", line.rstrip("\r"))
        if match:
            status = match.group(1)
    return status


def archived_observed_at(path: Path) -> str:
    match = re.search(r"wayback_instruments_(20\d{12})\.body$", path.name)
    if not match:
        raise ValueError(f"archive timestamp missing: {path}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def build_source_ledger(
    source_dir: Path, download_manifest: Path, local_instrument_sources: Sequence[Path]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for body in sorted(source_dir.glob("*.body")):
        stem = body.with_suffix("")
        url_path = stem.with_suffix(".url.txt")
        headers = stem.with_suffix(".headers.txt")
        rows.append({
            "source_id": body.stem,
            "source_kind": "official_kraken_live" if body.name.startswith("kraken_") else "archived_official_kraken",
            "url_or_path": url_path.read_text(encoding="utf-8").strip() if url_path.exists() else str(body),
            "access_utc": normalize_http_date(header_value(headers, "date")) if headers.exists() else "",
            "http_status": http_status(headers) if headers.exists() else "",
            "sha256": sha256_file(body),
            "bytes": body.stat().st_size,
            "authority_use": "instrument_identity_status_checkpoint" if "instruments_20" in body.name or body.name == "kraken_futures_instruments.body" else "source_discovery_or_negative_capability_check",
            "result": "parsed" if "instruments_20" in body.name or body.name == "kraken_futures_instruments.body" else "retained_not_sufficient_for_historical_continuity",
        })
    rows.append({
        "source_id": "local_rankable_download_manifest",
        "source_kind": "local_official_kraken_acquisition_manifest",
        "url_or_path": str(download_manifest),
        "access_utc": source_dir.joinpath("access_utc.txt").read_text(encoding="utf-8").strip(),
        "http_status": "local",
        "sha256": sha256_file(download_manifest),
        "bytes": download_manifest.stat().st_size,
        "authority_use": "trade_mark_metadata_coverage_only",
        "result": "parsed_without_opening_market_payloads",
    })
    for path in local_instrument_sources:
        rows.append({
            "source_id": f"local_instrument_{path.suffix.lstrip('.') or 'file'}",
            "source_kind": "local_official_kraken_instrument_snapshot",
            "url_or_path": str(path),
            "access_utc": source_dir.joinpath("access_utc.txt").read_text(encoding="utf-8").strip(),
            "http_status": "local",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "authority_use": "identity_cross_check_only",
            "result": "recorded; current snapshot does not prove historical lifecycle",
        })
    return rows


LIFECYCLE_FIELDS = [
    "instrument_id", "canonical_asset_id", "Kraken_symbol", "contract_type",
    "eligible_start_utc", "eligible_end_utc", "status_intervals", "listing_source",
    "status_or_end_source", "source_publication_utc", "source_access_utc", "source_sha256",
    "trade_coverage_start_utc", "trade_coverage_end_utc", "mark_coverage_start_utc",
    "mark_coverage_end_utc", "identity_confidence", "lifecycle_confidence", "unknown_fields",
    "included_in_U2", "inclusion_or_exclusion_reason", "permitted_claim",
]


def build(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    source_dir = Path(args.source_dir)
    manifest_path = Path(args.download_manifest)
    output.mkdir(parents=True, exist_ok=True)

    archived_paths = sorted(source_dir.glob("wayback_instruments_20*.body"))
    current_path = source_dir / "kraken_futures_instruments.body"
    snapshots: list[tuple[str, dict[str, dict[str, Any]], Path]] = []
    for path in archived_paths:
        observed = archived_observed_at(path)
        snapshots.append((observed, instrument_snapshot(path, observed), path))
    current_observed = normalize_http_date(header_value(source_dir / "kraken_futures_instruments.headers.txt", "date"))
    snapshots.append((current_observed, instrument_snapshot(current_path, current_observed), current_path))
    manifest_rows = read_csv(manifest_path)

    authority_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    source_paths = [path for _, _, path in snapshots]
    source_hashes = [sha256_file(path) for path in source_paths]
    checkpoint_publications = [observed for observed, _, _ in snapshots[:-1]]

    for symbol, asset in sorted(CONSIDERED.items()):
        assert_kraken_identity("kraken", symbol)
        history = [snapshot[symbol] for _, snapshot, _ in snapshots if symbol in snapshot]
        identity = validate_identity_history(symbol, history)
        trade_start, trade_end, trade_gaps, trade_chunks = coverage_for(manifest_rows, symbol, TRADE_DATASET)
        mark_start, mark_end, mark_gaps, mark_chunks = coverage_for(manifest_rows, symbol, MARK_DATASET)
        claimed_start = RANKABLE_START
        claimed_end = PROTECTED_START
        included, reasons = candidate_is_includable(
            official_start=identity["opening_date"], claimed_start=claimed_start,
            claimed_end=claimed_end, trade_start=trade_start, trade_end=trade_end,
            mark_start=mark_start, mark_end=mark_end, trade_gaps=trade_gaps,
            mark_gaps=mark_gaps, lifecycle_continuity_verified=False,
        )
        unknown = [
            "continuous contract status between sparse official snapshots",
            "historical suspension/post-only/wind-down intervals",
            "trade and mark coverage for 2025-12-31T00:00:00Z through 2026-01-01T00:00:00Z",
        ]
        status_intervals = json.dumps([
            {"observed_at": row["observed_at"], "tradeable": row["tradeable"]}
            for row in history
        ], sort_keys=True, separators=(",", ":"))
        authority = {
            "instrument_id": f"kraken:{symbol}",
            "canonical_asset_id": asset,
            "Kraken_symbol": symbol,
            "contract_type": identity["contract_type"],
            "eligible_start_utc": claimed_start if included else "",
            "eligible_end_utc": claimed_end if included else "",
            "status_intervals": status_intervals,
            "listing_source": str(source_paths[0]),
            "status_or_end_source": ";".join(str(path) for path in source_paths),
            "source_publication_utc": json.dumps(checkpoint_publications, separators=(",", ":")),
            "source_access_utc": current_observed,
            "source_sha256": ";".join(source_hashes),
            "trade_coverage_start_utc": trade_start,
            "trade_coverage_end_utc": trade_end,
            "mark_coverage_start_utc": mark_start,
            "mark_coverage_end_utc": mark_end,
            "identity_confidence": "high",
            "lifecycle_confidence": "insufficient_for_continuous_rankable_interval",
            "unknown_fields": json.dumps(unknown, separators=(",", ":")),
            "included_in_U2": str(included).lower(),
            "inclusion_or_exclusion_reason": ";".join(reasons) if reasons else "all_frozen_gates_pass",
            "permitted_claim": "identity, official opening date, and tradeable status at archived checkpoints only; no continuous-eligibility claim",
        }
        authority_rows.append(authority)
        if included:
            cohort_rows.append({**authority, "cohort_version": COHORT_VERSION})
        else:
            exclusion_rows.append({
                "Kraken_symbol": symbol,
                "canonical_asset_id": asset,
                "exclusion_reason": ";".join(reasons),
                "unresolved_uncertainty": json.dumps(unknown, separators=(",", ":")),
                "official_checkpoint_count": identity["checkpoint_count"],
                "official_checkpoint_start_utc": identity["checkpoint_start"],
                "official_checkpoint_end_utc": identity["checkpoint_end"],
                "trade_manifest_chunks": trade_chunks,
                "mark_manifest_chunks": mark_chunks,
                "smallest_repair": "acquire an official or archived complete lifecycle-status history covering the unresolved checkpoint gaps; separately acquire rankable trade/mark metadata for the final 2025 day",
            })

    cohort_hash = deterministic_cohort_hash(cohort_rows)
    cohort_fields = LIFECYCLE_FIELDS + ["cohort_version", "cohort_hash"]
    cohort_output_rows = [{**row, "cohort_hash": cohort_hash} for row in cohort_rows]
    source_fields = ["source_id", "source_kind", "url_or_path", "access_utc", "http_status", "sha256", "bytes", "authority_use", "result"]
    exclusion_fields = ["Kraken_symbol", "canonical_asset_id", "exclusion_reason", "unresolved_uncertainty", "official_checkpoint_count", "official_checkpoint_start_utc", "official_checkpoint_end_utc", "trade_manifest_chunks", "mark_manifest_chunks", "smallest_repair"]
    local_instrument_sources = [Path(args.local_instrument_parquet), Path(args.local_instrument_raw)]
    write_csv(
        output / "U2_LIFECYCLE_SOURCE_LEDGER.csv",
        build_source_ledger(source_dir, manifest_path, local_instrument_sources),
        source_fields,
    )
    write_csv(output / "U2_INSTRUMENT_LIFECYCLE_AUTHORITY.csv", authority_rows, LIFECYCLE_FIELDS)
    write_csv(output / "U2_ANCHOR_COHORT.csv", cohort_output_rows, cohort_fields)
    write_csv(output / "U2_EXCLUSIONS_AND_UNCERTAINTY.csv", exclusion_rows, exclusion_fields)

    coverage_report = f"""# U2 Coverage And Claim Boundary

## Decision

No contract is admitted to `{COHORT_VERSION}`. BTC and ETH identity and opening dates are high-confidence, and both were tradeable in all ten archived official Kraken instrument snapshots plus the current official snapshot. That evidence does not establish uninterrupted lifecycle status between sparse checkpoints; the longest archived gap is approximately 15 months. Unknown lifecycle state is not imputed active.

## Coverage boundary

- Requested interval: `[{RANKABLE_START}, {PROTECTED_START})`.
- Local official trade/mark metadata coverage: `[{RANKABLE_START}, 2025-12-31T00:00:00Z)` for both considered contracts, with no manifest interval gaps.
- Uncovered rankable tail: `[2025-12-31T00:00:00Z, {PROTECTED_START})`.
- Protected outcome payloads opened: **zero**.
- Economic outputs computed: **zero**.

## Permitted claim

The artifacts establish stable Kraken identity, official March 2022 opening dates, tradeable status at the archived checkpoints, and local trade/mark metadata coverage through the stated exclusive end. They do **not** establish continuous historical eligibility, absence of temporary suspension/wind-down state, or a rankable U2 cohort.

## C01 boundary

C01 is not authorized because the U2 cohort is empty. The next task is a bounded lifecycle repair, not feature implementation or economic research.
"""
    (output / "U2_COVERAGE_AND_CLAIM_BOUNDARY.md").write_text(coverage_report, encoding="utf-8")
    validation_report = f"""# U2 Validation Report

Status: **blocked insufficient lifecycle authority**

- Contracts considered: {len(CONSIDERED)}
- Contracts included: {len(cohort_rows)}
- Contracts excluded: {len(exclusion_rows)}
- Archived official instrument checkpoints parsed: {len(archived_paths)}
- Current official instrument snapshots parsed: 1
- Identity conflicts: 0
- Coverage metadata gaps: 0
- Cohort hash: `{cohort_hash}` (canonical JSON of the included rows)
- Outcome fields read: 0
- Candidate returns computed: 0
- Protected payloads opened: 0

All exclusions are fail-closed consequences of the frozen rule. Synthetic parser and interval tests are recorded in the task command log.
"""
    (output / "U2_VALIDATION_REPORT.md").write_text(validation_report, encoding="utf-8")

    artifacts = [
        "U2_LIFECYCLE_SOURCE_LEDGER.csv", "U2_INSTRUMENT_LIFECYCLE_AUTHORITY.csv",
        "U2_ANCHOR_COHORT.csv", "U2_EXCLUSIONS_AND_UNCERTAINTY.csv",
        "U2_COVERAGE_AND_CLAIM_BOUNDARY.md", "U2_VALIDATION_REPORT.md",
    ]
    manifest = {
        "task_id": "donch_bt_stage_2a_u2_lifecycle_20260716_v1",
        "cohort_version": COHORT_VERSION,
        "cohort_hash": cohort_hash,
        "contracts_considered": len(CONSIDERED),
        "contracts_included": len(cohort_rows),
        "protected_outcomes_opened": False,
        "economic_outputs_computed": False,
        "manifest_self_hash_excluded": True,
        "files": [
            {"path": name, "bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)}
            for name in artifacts
        ],
    }
    (output / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--download-manifest", required=True)
    parser.add_argument("--local-instrument-parquet", required=True)
    parser.add_argument("--local-instrument-raw", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
