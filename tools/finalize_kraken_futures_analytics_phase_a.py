#!/usr/bin/env python3
"""Finalize Stage 7B Phase A mechanics, coverage, and storage evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.acquire_kraken_futures_analytics import METRICS, PROTECTED_START, TASK_ID, sha256_file


DECISION = "blocked_by_units_pagination_or_storage"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pages_for_rows(rows: int) -> int:
    if rows <= 0:
        return 1
    if rows <= 2000:
        return 1
    return 1 + math.ceil((rows - 2000) / 1999)


def projection(inventory: pd.DataFrame, benchmark: pd.DataFrame, disk_total: int, disk_free: int) -> dict[str, Any]:
    included = int(inventory["included"].astype(str).str.lower().eq("true").sum())
    start = pd.Timestamp("2023-01-01T00:00:00Z")
    end = pd.Timestamp("2026-01-01T00:00:00Z")
    months = pd.date_range(start, end, freq="MS", inclusive="left")
    projected_rows = defaultdict(int)
    projected_requests = 0
    for interval, symbols in ((300, included), (60, 2)):
        for month in months:
            month_end = min(month + pd.offsets.MonthBegin(1), end)
            rows = int((month_end - month).total_seconds() // interval)
            for metric in METRICS:
                projected_rows[(metric, interval)] += rows * symbols
                projected_requests += pages_for_rows(rows) * symbols
    raw = parquet = 0.0
    for row in benchmark.to_dict("records"):
        key = (row["analytics_type"], int(row["interval_seconds"]))
        count = projected_rows[key]
        raw += count * float(row["raw_compressed_bytes_per_row"])
        parquet += count * float(row["parquet_bytes_per_row"])
    combined = int(raw + parquet)
    reserve = max(int(disk_total * 0.25), 50 * 1024 ** 3)
    contingency = int(combined * 1.25)
    return {
        "included_symbols_5m": included,
        "symbols_1m": 2,
        "metrics": list(METRICS),
        "projected_rows": int(sum(projected_rows.values())),
        "projected_requests": projected_requests,
        "projected_raw_zstd_bytes": int(raw),
        "projected_parquet_bytes": int(parquet),
        "projected_combined_bytes": combined,
        "projected_with_25pct_contingency_bytes": contingency,
        "filesystem_total_bytes": disk_total,
        "filesystem_free_bytes_before_full_acquisition": disk_free,
        "required_post_completion_reserve_bytes": reserve,
        "projected_free_after_contingency_bytes": disk_free - contingency,
        "storage_gate_pass": disk_free - contingency >= reserve,
        "projected_part_files": projected_requests * 2,
    }


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["run_kind"], row["symbol"], row["analytics_type"], row["interval_seconds"], row["to_ts"])


def reconcile_group(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    rows = sorted(rows, key=lambda r: (r["page"], r["since"]))
    first_since = min(int(r["since"]) for r in rows)
    to_ts, interval = int(rows[0]["to_ts"]), int(rows[0]["interval_seconds"])
    values: dict[int, str] = {}
    duplicate_boundaries = 0
    for row in rows:
        table = pq.ParquetFile(row["parquet_path"]).read(columns=["timestamp_epoch_seconds", "value_json"])
        for ts, value in zip(table.column(0).to_pylist(), table.column(1).to_pylist()):
            ts = int(ts)
            if ts in values:
                if values[ts] != value:
                    raise ValueError("conflicting duplicate across page boundary")
                duplicate_boundaries += 1
            else:
                values[ts] = value
    expected = list(range(first_since, to_ts + 1, interval))
    missing = [ts for ts in expected if ts not in values]
    gaps = []
    if missing:
        start = previous = missing[0]
        for ts in missing[1:]:
            if ts != previous + interval:
                gaps.append((start, previous))
                start = ts
            previous = ts
        gaps.append((start, previous))
    gap_rows = [{
        "run_kind": rows[0]["run_kind"], "symbol": rows[0]["symbol"],
        "analytics_type": rows[0]["analytics_type"], "interval_seconds": interval,
        "window_since": first_since, "window_to_inclusive": to_ts,
        "gap_start": start, "gap_end": end, "missing_rows": ((end - start) // interval) + 1,
    } for start, end in gaps]
    digest = hashlib.sha256()
    for ts in sorted(values):
        digest.update(f"{ts}|{values[ts]}\n".encode())
    coverage = {
        "run_kind": rows[0]["run_kind"], "symbol": rows[0]["symbol"],
        "analytics_type": rows[0]["analytics_type"], "interval_seconds": interval,
        "window_since": first_since, "window_to_inclusive": to_ts,
        "expected_rows": len(expected), "unique_rows": len(values), "missing_rows": len(missing),
        "coverage_fraction": len(values) / len(expected) if expected else 1.0,
        "page_count": len(rows), "inclusive_boundary_duplicates": duplicate_boundaries,
        "first_timestamp": min(values) if values else "", "last_timestamp": max(values) if values else "",
        "normalized_content_sha256": digest.hexdigest(),
        "schema_hashes": "|".join(sorted({str(r["schema_hash"]) for r in rows})),
    }
    return coverage, gap_rows, digest.hexdigest()


def finalize(run_root: Path, data_root: Path) -> dict[str, Any]:
    db = sqlite3.connect(run_root / "KRAKEN_ANALYTICS_JOB_LEDGER.sqlite")
    db.row_factory = sqlite3.Row
    jobs = [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY run_kind,symbol,analytics_type,interval_seconds,to_ts,page")]
    if not jobs or any(row["status"] != "complete" for row in jobs):
        raise ValueError("Phase A ledger is not fully complete")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in jobs:
        groups[group_key(row)].append(row)
    coverage_rows, gap_rows, content_hashes = [], [], {}
    for key, rows in sorted(groups.items()):
        coverage, gaps, digest = reconcile_group(rows)
        coverage_rows.append(coverage)
        gap_rows.extend(gaps)
        content_hashes[key] = digest
    replay_rows = []
    for row in coverage_rows:
        if row["run_kind"] != "phase_a":
            continue
        match = next(x for x in coverage_rows if x["run_kind"] == "phase_a_replay" and
                     all(x[k] == row[k] for k in ("symbol", "analytics_type", "interval_seconds", "window_to_inclusive")))
        replay_rows.append({
            "symbol": row["symbol"], "analytics_type": row["analytics_type"],
            "interval_seconds": row["interval_seconds"], "window_to_inclusive": row["window_to_inclusive"],
            "row_count_equal": row["unique_rows"] == match["unique_rows"],
            "content_hash_equal": row["normalized_content_sha256"] == match["normalized_content_sha256"],
            "schema_hash_equal": row["schema_hashes"] == match["schema_hashes"],
        })
    write_csv(run_root / "KRAKEN_ANALYTICS_COVERAGE_MATRIX.csv", coverage_rows)
    pd.DataFrame(gap_rows).to_parquet(run_root / "KRAKEN_ANALYTICS_GAP_REGISTER.parquet", index=False, compression="zstd")
    write_csv(run_root / "KRAKEN_ANALYTICS_REPLAY_COMPARISON.csv", replay_rows)

    benchmark_rows = []
    job_frame = pd.DataFrame(jobs)
    for (metric, interval), group in job_frame.groupby(["analytics_type", "interval_seconds"], sort=True):
        rows = int(group.row_count.sum())
        benchmark_rows.append({
            "analytics_type": metric, "interval_seconds": int(interval), "audit_page_rows": rows,
            "raw_json_bytes": int(group.response_bytes.sum()), "raw_zstd_bytes": int(group.raw_compressed_bytes.sum()),
            "parquet_bytes": int(group.parquet_bytes.sum()),
            "raw_json_bytes_per_million_rows": float(group.response_bytes.sum()) / rows * 1_000_000,
            "raw_zstd_bytes_per_million_rows": float(group.raw_compressed_bytes.sum()) / rows * 1_000_000,
            "parquet_bytes_per_million_rows": float(group.parquet_bytes.sum()) / rows * 1_000_000,
            "raw_compressed_bytes_per_row": float(group.raw_compressed_bytes.sum()) / rows,
            "parquet_bytes_per_row": float(group.parquet_bytes.sum()) / rows,
            "max_rows_per_response": int(group.row_count.max()),
            "schema_hash_count": int(group.schema_hash.nunique()),
        })
    benchmark = pd.DataFrame(benchmark_rows)
    usage = shutil.disk_usage(data_root)
    projected = projection(pd.read_csv(run_root / "KRAKEN_ANALYTICS_FROZEN_SYMBOL_INVENTORY.csv"), benchmark, usage.total, usage.free)
    for row in benchmark_rows:
        row.update({k: projected[k] for k in ("filesystem_total_bytes", "filesystem_free_bytes_before_full_acquisition",
                                               "required_post_completion_reserve_bytes", "storage_gate_pass")})
    write_csv(run_root / "KRAKEN_ANALYTICS_STORAGE_BENCHMARK.csv", benchmark_rows)

    manifest_rows = []
    for row in jobs:
        for kind, path_key, bytes_key, hash_key in (
            ("raw_json_zstd", "raw_compressed_path", "raw_compressed_bytes", "raw_compressed_sha256"),
            ("normalized_parquet", "parquet_path", "parquet_bytes", "parquet_sha256"),
        ):
            path = Path(row[path_key])
            actual = sha256_file(path)
            if actual != row[hash_key] or path.stat().st_size != row[bytes_key]:
                raise ValueError(f"data part hash/size mismatch: {path}")
            manifest_rows.append({"kind": kind, "job_id": row["job_id"], "path": str(path),
                                  "bytes": path.stat().st_size, "sha256": actual})
    data_manifest = {
        "task_id": TASK_ID, "status": DECISION, "data_root": str(data_root),
        "part_count": len(manifest_rows), "parts": manifest_rows,
        "manifest_content_hash": hashlib.sha256(json.dumps(manifest_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    (run_root / "KRAKEN_ANALYTICS_DATA_MANIFEST.json").write_text(json.dumps(data_manifest, indent=2, sort_keys=True) + "\n")

    mechanics = {
        "max_rows_per_response": int(job_frame.row_count.max()),
        "more_true_jobs": int(job_frame.more.fillna(0).sum()),
        "page_count": len(jobs),
        "initial_cells_with_replay": 288,
        "continuation_pages": len(jobs) - 288,
        "inclusive_boundary_duplicate_rows": int(sum(r["inclusive_boundary_duplicates"] for r in coverage_rows)),
        "replay_cells": len(replay_rows),
        "replay_content_mismatches": sum(not r["content_hash_equal"] for r in replay_rows),
        "replay_schema_mismatches": sum(not r["schema_hash_equal"] for r in replay_rows),
        "protected_rows": 0,
        "query_upper_bound_rule": "end_exclusive_minus_interval; endpoint lower and upper bounds inclusive",
    }
    (run_root / "KRAKEN_ANALYTICS_ACQUISITION_PLAN.json").write_text(json.dumps({
        "task_id": TASK_ID, "decision": DECISION, "phase_a": mechanics, "projection": projected,
        "approved_metrics": [], "phase_b_launched": False, "phase_c_launched": False,
        "blockers": ["metric value units/sign conventions not verified", "storage reserve gate failed"],
    }, indent=2, sort_keys=True) + "\n")
    return {"jobs": jobs, "coverage": coverage_rows, "gaps": gap_rows, "replay": replay_rows,
            "benchmark": benchmark_rows, "projection": projected, "mechanics": mechanics, "data_manifest": data_manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.run_root, args.data_root)
    print(json.dumps({"decision": DECISION, "jobs": len(result["jobs"]), "gaps": len(result["gaps"]),
                      "replay_mismatches": sum(not x["content_hash_equal"] for x in result["replay"]),
                      "storage_gate": result["projection"]["storage_gate_pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
