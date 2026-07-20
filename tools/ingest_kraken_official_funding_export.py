#!/usr/bin/env python3
"""Fail-closed ingestion of Kraken's human-transferred funding CSV export.

Protected rows are routed as opaque raw bytes after parsing only timestamp and
tradeable identity. Rate fields are converted to Decimal only for rankable rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any


RANKABLE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
PROTECTED_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
EXPECTED_HEADER = b"timestamp,tradeable,absolute_rate,relative_rate"
CSV_NAME = re.compile(r"^exports/([A-Z0-9_]+)\.csv$")
TIMESTAMP = re.compile(rb"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):00:00$")
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1000.0


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def parse_hour(raw: bytes) -> datetime:
    match = TIMESTAMP.fullmatch(raw)
    if not match:
        raise ValueError("timestamp is not an exact UTC hour")
    return datetime(*(int(part) for part in match.groups()), tzinfo=timezone.utc)


def decimal_rate(raw: bytes) -> Decimal:
    try:
        value = Decimal(raw.decode("ascii"))
    except (InvalidOperation, UnicodeDecodeError) as exc:
        raise ValueError("invalid decimal rate") from exc
    if not value.is_finite():
        raise ValueError("non-finite decimal rate")
    return value


def safe_member_kind(info: zipfile.ZipInfo) -> str:
    name = info.filename
    path = PurePosixPath(name)
    if name.startswith("/") or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise RuntimeError(f"unsafe ZIP path: {name!r}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and stat.S_ISLNK(mode):
        raise RuntimeError(f"ZIP symlink rejected: {name!r}")
    if info.flag_bits & 0x1:
        raise RuntimeError(f"encrypted ZIP member rejected: {name!r}")
    if info.file_size > MAX_MEMBER_BYTES:
        raise RuntimeError(f"unreasonable ZIP member size: {name!r}")
    ratio = info.file_size / max(1, info.compress_size)
    if ratio > MAX_COMPRESSION_RATIO:
        raise RuntimeError(f"unreasonable ZIP compression ratio: {name!r}")
    if info.is_dir():
        return "directory"
    if CSV_NAME.fullmatch(name):
        return "funding_csv"
    if name.startswith("__MACOSX/") and PurePosixPath(name).name.startswith("._"):
        return "appledouble_metadata_excluded"
    if name == "exports/.DS_Store":
        return "finder_metadata_excluded"
    raise RuntimeError(f"unexpected ZIP payload: {name!r}")


def deterministic_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for name, payload in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100444 << 16
            target.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.chmod(path, 0o444)


def load_universes(path: Path) -> tuple[set[str], set[str], set[str]]:
    reconciled: set[str] = set()
    k0: set[str] = set()
    campaign: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = row["PF_symbol"]
            reconciled.add(symbol)
            if row["included"] == "True":
                k0.add(symbol)
            if row["final_campaign_eligible"] == "True":
                campaign.add(symbol)
    return reconciled, k0, campaign


def ingest(source_zip: Path, output: Path, universe_csv: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reconciled, k0, campaign = load_universes(universe_csv)
    inventory: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    rankable_members: list[tuple[str, bytes]] = []
    protected_members: list[tuple[str, bytes]] = []
    pre_rankable_members: list[tuple[str, bytes]] = []
    invalid_members: list[tuple[str, bytes]] = []
    export_symbols: set[str] = set()

    with zipfile.ZipFile(source_zip) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise RuntimeError(f"duplicate ZIP member names: {duplicates[:5]}")
        for info in infos:
            kind = safe_member_kind(info)
            payload = archive.read(info)
            if len(payload) != info.file_size:
                raise RuntimeError(f"member length mismatch: {info.filename}")
            inventory.append({
                "member_name": info.filename, "member_kind": kind,
                "compressed_bytes": info.compress_size, "uncompressed_bytes": info.file_size,
                "compression_ratio": round(info.file_size / max(1, info.compress_size), 6),
                "uncompressed_sha256": sha256_bytes(payload),
                "exclusion_reason": (
                    "Finder AppleDouble metadata; not a funding CSV" if kind == "appledouble_metadata_excluded"
                    else "Finder directory metadata; not a funding CSV" if kind == "finder_metadata_excluded"
                    else ""
                ),
            })
            if kind != "funding_csv":
                continue
            symbol = CSV_NAME.fullmatch(info.filename).group(1)  # type: ignore[union-attr]
            export_symbols.add(symbol)
            lines = payload.splitlines(keepends=True)
            if not lines or lines[0].rstrip(b"\r\n") != EXPECTED_HEADER:
                raise RuntimeError(f"header mismatch: {info.filename}")
            rankable = [EXPECTED_HEADER + b"\n"]
            protected = [EXPECTED_HEADER + b"\n"]
            pre_rankable = [EXPECTED_HEADER + b"\n"]
            invalid = [EXPECTED_HEADER + b"\n"]
            first: datetime | None = None
            last: datetime | None = None
            prior: datetime | None = None
            rankable_count = protected_count = pre_rankable_count = invalid_count = gap_hours = zero_pairs = 0
            for line_number, raw_line in enumerate(lines[1:], start=2):
                raw = raw_line.rstrip(b"\r\n")
                prefix = raw.split(b",", 2)
                if len(prefix) != 3:
                    invalid.append(raw + b"\n"); invalid_count += 1; continue
                raw_ts, raw_tradeable, rate_tail = prefix
                try:
                    stamp = parse_hour(raw_ts)
                    tradeable = raw_tradeable.decode("ascii")
                except (ValueError, UnicodeDecodeError):
                    invalid.append(raw + b"\n"); invalid_count += 1; continue
                if tradeable != symbol:
                    raise RuntimeError(f"filename/tradeable mismatch: {info.filename}:{line_number}")
                if prior is not None:
                    if stamp <= prior:
                        raise RuntimeError(f"duplicate or unordered timestamp: {info.filename}:{line_number}")
                    gap_hours += max(0, int((stamp - prior).total_seconds() // 3600) - 1)
                first = first or stamp; last = stamp; prior = stamp
                if stamp >= PROTECTED_START:
                    # The rate tail stays opaque: it is neither split nor converted.
                    protected.append(raw + b"\n"); protected_count += 1
                    continue
                rate_fields = rate_tail.split(b",")
                if len(rate_fields) != 2:
                    raise RuntimeError(f"rankable field count mismatch: {info.filename}:{line_number}")
                absolute, relative = map(decimal_rate, rate_fields)
                if (absolute == 0) != (relative == 0):
                    raise RuntimeError(f"zero/zero mismatch: {info.filename}:{line_number}")
                if absolute == 0:
                    zero_pairs += 1
                elif absolute.is_signed() != relative.is_signed():
                    raise RuntimeError(f"rate sign mismatch: {info.filename}:{line_number}")
                if stamp < RANKABLE_START:
                    pre_rankable.append(raw + b"\n"); pre_rankable_count += 1
                else:
                    rankable.append(raw + b"\n"); rankable_count += 1
            if invalid_count:
                invalid_members.append((f"unknown_or_invalid/{symbol}.csv", b"".join(invalid)))
            rankable_payload = b"".join(rankable)
            protected_payload = b"".join(protected)
            pre_rankable_payload = b"".join(pre_rankable)
            rankable_members.append((f"rankable_2023_2025/{symbol}.csv", rankable_payload))
            if protected_count:
                protected_members.append((f"protected_2026_plus_quarantine/{symbol}.csv", protected_payload))
            if pre_rankable_count:
                pre_rankable_members.append((f"pre_rankable_before_2023_excluded/{symbol}.csv", pre_rankable_payload))
            coverage.append({
                "symbol": symbol, "source_member": info.filename, "source_rows": len(lines) - 1,
                "first_timestamp_utc": first.isoformat().replace("+00:00", "Z") if first else "",
                "last_timestamp_utc": last.isoformat().replace("+00:00", "Z") if last else "",
                "rankable_rows": rankable_count, "protected_rows": protected_count,
                "pre_rankable_excluded_rows": pre_rankable_count,
                "unknown_or_invalid_rows": invalid_count, "missing_hour_count": gap_hours,
                "rankable_zero_zero_rows": zero_pairs,
                "rankable_member_sha256": sha256_bytes(rankable_payload),
                "protected_member_sha256": sha256_bytes(protected_payload) if protected_count else "",
            })
            ledger.extend([
                {"symbol": symbol, "partition": "rankable_2023_2025", "row_count": rankable_count, "numeric_rate_parsing": "Decimal"},
                {"symbol": symbol, "partition": "protected_2026_plus_quarantine", "row_count": protected_count, "numeric_rate_parsing": "none_opaque_raw_line"},
                {"symbol": symbol, "partition": "pre_rankable_before_2023_excluded", "row_count": pre_rankable_count, "numeric_rate_parsing": "Decimal_schema_only_not_calibration"},
                {"symbol": symbol, "partition": "unknown_or_invalid", "row_count": invalid_count, "numeric_rate_parsing": "none"},
            ])

    anchors = {
        "PF_DEXEUSD": ("297f5badb99207349f71307e0496cf35463454f69a107fe429d8cf402c8ad0f8", 8664, 745),
        "PF_OPENUSD": ("6fc1d276a9f4207a28be9f2504192360785abb6a3d9bd3ee6d91a8b4b0671c31", 2743, 745),
        "PF_ZBTUSD": ("c2877d184c5a5e7e185b7d6d76601fb5d31c436b096c6c8f2da69386cd052315", 2073, 745),
    }
    by_member = {row["member_name"]: row for row in inventory}
    by_symbol = {row["symbol"]: row for row in coverage}
    for symbol, (digest, rows, protected_rows) in anchors.items():
        member = by_member.get(f"exports/{symbol}.csv")
        if member is None or member["uncompressed_sha256"] != digest:
            raise RuntimeError(f"sample anchor hash mismatch: {symbol}")
        if by_symbol[symbol]["source_rows"] != rows or by_symbol[symbol]["protected_rows"] != protected_rows:
            raise RuntimeError(f"sample anchor count mismatch: {symbol}")

    campaign_mapping = []
    for symbol in sorted(campaign):
        present = symbol in export_symbols
        campaign_mapping.append({
            "campaign_symbol": symbol, "export_member": f"exports/{symbol}.csv" if present else "",
            "mapping_status": "mapped_exact_filename_and_tradeable" if present else "mechanically_excluded_missing_export_symbol",
            "mechanical_exclusion_reason": "" if present else "no official export member",
        })
    missing_campaign = sorted(campaign - export_symbols)
    if missing_campaign:
        raise RuntimeError(f"campaign symbols lack export coverage: {missing_campaign}")

    rankable_zip = output / "kraken_funding_rankable_2023_2025.zip"
    protected_zip = output / "kraken_funding_protected_2026_plus_quarantine.zip"
    invalid_zip = output / "kraken_funding_unknown_or_invalid.zip"
    pre_rankable_zip = output / "kraken_funding_pre_rankable_before_2023_excluded.zip"
    deterministic_zip(rankable_zip, rankable_members)
    deterministic_zip(protected_zip, protected_members)
    deterministic_zip(invalid_zip, invalid_members)
    deterministic_zip(pre_rankable_zip, pre_rankable_members)

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        path = output / name
        fields = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)

    write_csv("OFFICIAL_EXPORT_MEMBER_INVENTORY.csv", inventory)
    write_csv("OFFICIAL_EXPORT_COVERAGE.csv", sorted(coverage, key=lambda row: row["symbol"]))
    write_csv("OFFICIAL_EXPORT_PARTITION_LEDGER.csv", sorted(ledger, key=lambda row: (row["symbol"], row["partition"])))
    write_csv("CAMPAIGN_SYMBOL_FUNDING_MAPPING.csv", campaign_mapping)
    result = {
        "source": {
            "path": str(source_zip), "byte_size": source_zip.stat().st_size,
            "sha256": sha256_file(source_zip), "member_count": len(inventory),
            "compressed_bytes": sum(row["compressed_bytes"] for row in inventory),
            "uncompressed_bytes": sum(row["uncompressed_bytes"] for row in inventory),
            "authority_classification": "human_transferred_official_export+official_support_page_provenance+local_content_hash",
        },
        "inventory": {"export_symbols": len(export_symbols), "reconciled_identities": len(reconciled), "k0_included_identities": len(k0), "campaign_symbols": len(campaign)},
        "partitions": {
            "rankable": {"path": str(rankable_zip), "sha256": sha256_file(rankable_zip), "rows": sum(row["rankable_rows"] for row in coverage), "symbols": sum(row["rankable_rows"] > 0 for row in coverage)},
            "protected": {"path": str(protected_zip), "sha256": sha256_file(protected_zip), "rows": sum(row["protected_rows"] for row in coverage), "symbols": sum(row["protected_rows"] > 0 for row in coverage)},
            "unknown_or_invalid": {"path": str(invalid_zip), "sha256": sha256_file(invalid_zip), "rows": sum(row["unknown_or_invalid_rows"] for row in coverage)},
            "pre_rankable_excluded": {"path": str(pre_rankable_zip), "sha256": sha256_file(pre_rankable_zip), "rows": sum(row["pre_rankable_excluded_rows"] for row in coverage)},
        },
        "sample_anchors_verified": sorted(anchors),
        "campaign_mapping": {"mapped": len(campaign_mapping), "mechanically_excluded": 0},
        "protected_funding_rows_opened_for_partition": True,
        "protected_funding_values_used_for_statistics": 0,
        "protected_strategy_price_or_return_rows_opened": 0,
    }
    (output / "INGESTION_RESULT.json").write_bytes(canonical_bytes(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--universe-csv", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(ingest(args.source_zip, args.output, args.universe_csv), sort_keys=True))
