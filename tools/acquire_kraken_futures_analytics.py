#!/usr/bin/env python3
"""Resumable, bounded acquisition for approved Kraken Futures analytics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import signal
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


TASK_ID = "donch_bt_stage_7b_resumable_analytics_acquisition_20260717_v1"
BASE_URL = "https://futures.kraken.com/api/charts/v1/analytics"
TRAIN_START = 1672531200
PROTECTED_START = 1767225600
METRICS = ("open-interest", "liquidation-volume", "future-basis")
INTERVALS = (60, 300)
STATUSES = ("planned", "running", "raw_verified", "normalized", "validated", "complete", "retryable_error", "blocked_error")
AUDIT_WINDOWS = (
    ("2023_start", 1672531200, 1673136000),
    ("2023_end", 1703462400, 1704067200),
    ("2024_mid", 1717200000, 1717804800),
    ("2025_end", 1766534400, 1767225600),
)


@dataclass(frozen=True)
class JobSpec:
    run_kind: str
    symbol: str
    analytics_type: str
    interval_seconds: int
    since: int
    to: int
    page: int = 0

    @property
    def job_id(self) -> str:
        value = f"{self.run_kind}|{self.symbol}|{self.analytics_type}|{self.interval_seconds}|{self.since}|{self.to}|{self.page}"
        return hashlib.sha256(value.encode()).hexdigest()[:24]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def validate_spec(spec: JobSpec) -> None:
    if not spec.symbol.startswith("PF_") or spec.analytics_type not in METRICS or spec.interval_seconds not in INTERVALS:
        raise ValueError("job outside approved symbol/type/interval contract")
    if spec.since < TRAIN_START or spec.since >= spec.to or spec.to >= PROTECTED_START:
        raise ValueError("job outside explicit rankable boundary")


def request_url(spec: JobSpec) -> str:
    validate_spec(spec)
    query = urlencode({"interval": spec.interval_seconds, "since": spec.since, "to": spec.to})
    url = f"{BASE_URL}/{spec.symbol}/{spec.analytics_type}?{query}"
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    if parsed.scheme != "https" or parsed.netloc != "futures.kraken.com" or set(q) != {"interval", "since", "to"}:
        raise ValueError("unsafe analytics request")
    return url


def audit_specs(symbols: list[str], run_kind: str = "phase_a") -> list[JobSpec]:
    if len(symbols) != 6 or symbols[:2] != ["PF_XBTUSD", "PF_ETHUSD"] or len(set(symbols)) != 6:
        raise ValueError("Phase A requires the frozen six-symbol audit inventory")
    # Kraken's `to` is inclusive; translate each frozen end-exclusive window
    # to the final allowed interval timestamp before constructing the URL.
    return [JobSpec(run_kind, symbol, metric, interval, since, end_exclusive - interval)
            for symbol in symbols for metric in METRICS for interval in INTERVALS
            for _, since, end_exclusive in AUDIT_WINDOWS]


def monthly_specs(symbols: list[str], metrics: list[str], interval: int, run_kind: str) -> list[JobSpec]:
    if not metrics or any(metric not in METRICS for metric in metrics):
        raise ValueError("full acquisition requires approved analytics metrics")
    months = pd.date_range("2023-01-01T00:00:00Z", "2026-01-01T00:00:00Z", freq="MS", inclusive="left")
    specs = []
    for symbol in symbols:
        for metric in metrics:
            for month in months:
                end = min(month + pd.offsets.MonthBegin(1), pd.Timestamp("2026-01-01T00:00:00Z"))
                specs.append(JobSpec(run_kind, symbol, metric, interval, int(month.timestamp()), int(end.timestamp()) - interval))
    return specs


def validate_acquisition_approval(path: Path) -> list[str]:
    plan = json.loads(path.read_text())
    metrics = list(plan.get("approved_metrics", []))
    if not plan.get("projection", {}).get("storage_gate_pass", False):
        raise ValueError("full acquisition storage gate is not approved")
    if not metrics or any(metric not in METRICS for metric in metrics):
        raise ValueError("full acquisition unit/semantic gate is not approved")
    return metrics


def schema_hash(payload: Any) -> str:
    def shape(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: shape(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [shape(value[0])] if value else []
        return type(value).__name__
    return canonical_hash(shape(payload))


def normalized_rows(spec: JobSpec, payload: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("timestamp"), list) or "data" not in result:
        raise ValueError("analytics schema missing result.timestamp or result.data")
    timestamps = result["timestamp"]
    if not all(isinstance(v, int) for v in timestamps):
        raise ValueError("timestamp unit/type ambiguous")
    if any(v < TRAIN_START or v > spec.to or v >= PROTECTED_START for v in timestamps):
        # Fail before traversing data values.
        raise PermissionError("response contains out-of-bound or protected timestamp")
    if any(v % spec.interval_seconds for v in timestamps):
        raise ValueError("timestamps are not interval aligned")
    data = result["data"]
    values: list[Any]
    value_field: str
    if spec.analytics_type == "future-basis":
        if not isinstance(data, dict) or not isinstance(data.get("basis"), list):
            raise ValueError("future-basis schema missing data.basis")
        values, value_field = data["basis"], "basis"
    elif spec.analytics_type == "open-interest":
        if not isinstance(data, list):
            raise ValueError("open-interest data is not a list")
        values, value_field = data, "open_interest"
    else:
        if not isinstance(data, list):
            raise ValueError("liquidation-volume data is not a list")
        values, value_field = data, "liquidation_volume"
    if len(values) != len(timestamps):
        raise ValueError("timestamp/value length mismatch")
    rows = []
    for row_number, (ts, value) in enumerate(zip(timestamps, values)):
        value_json = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        row = {
            "timestamp_utc": pd.Timestamp(ts, unit="s", tz="UTC"),
            "timestamp_epoch_seconds": ts,
            "value_field": value_field,
            "value_json": value_json,
            "analytics_type": spec.analytics_type,
            "symbol": spec.symbol,
            "interval_seconds": spec.interval_seconds,
            "source_job_id": spec.job_id,
            "source_request_id": spec.job_id,
            "request_since": spec.since,
            "request_to": spec.to,
            "semantic_status": "source_authorized_economic_interpretation_blocked",
        }
        if spec.analytics_type == "open-interest":
            if not isinstance(value, list) or len(value) != 4 or not all(isinstance(item, str) for item in value):
                raise ValueError("open-interest row is not the exact four-string tuple")
            row.update({f"value_{index}_raw": item for index, item in enumerate(value)})
        elif spec.analytics_type == "liquidation-volume":
            if not isinstance(value, str):
                raise ValueError("liquidation-volume row is not an exact scalar string")
            row["value_raw"] = value
        else:
            # Retain every aligned field returned under data, not only basis.
            for field, field_values in sorted(data.items()):
                if not isinstance(field_values, list) or len(field_values) != len(timestamps):
                    raise ValueError(f"future-basis field {field!r} is not timestamp aligned")
                field_value = field_values[row_number]
                row[f"{field}_raw"] = field_value if isinstance(field_value, str) else json.dumps(
                    field_value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
        rows.append(row)
    columns = ["timestamp_utc", "timestamp_epoch_seconds", "value_field", "value_json", "analytics_type",
               "symbol", "interval_seconds", "source_job_id", "source_request_id", "request_since", "request_to", "semantic_status"]
    extra = sorted({key for row in rows for key in row} - set(columns))
    frame = pd.DataFrame(rows, columns=columns + extra)
    return frame, bool(result.get("more", False))


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    ordered = frame.sort_values(["timestamp_epoch_seconds", "value_json"], kind="stable")
    counts = ordered.groupby("timestamp_epoch_seconds", sort=False)["value_json"].nunique(dropna=False)
    if (counts > 1).any():
        raise ValueError("conflicting duplicate timestamp values")
    duplicates = int(ordered.duplicated("timestamp_epoch_seconds", keep="first").sum())
    return ordered.drop_duplicates("timestamp_epoch_seconds", keep="first").reset_index(drop=True), duplicates


def continuation_spec(spec: JobSpec, frame: pd.DataFrame, more: bool) -> JobSpec | None:
    if not more:
        return None
    if frame.empty:
        # The endpoint can report more=true for a valid empty historical slice.
        # There is no safe cursor to follow; record the anomaly and stop this cell.
        return None
    last = int(frame["timestamp_epoch_seconds"].max())
    if last <= spec.since or last >= spec.to:
        raise ValueError("pagination made no safe progress")
    # Kraken lower bounds are inclusive; retain the boundary and deduplicate later.
    return JobSpec(spec.run_kind, spec.symbol, spec.analytics_type, spec.interval_seconds, last, spec.to, spec.page + 1)


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, run_kind TEXT NOT NULL, analytics_type TEXT NOT NULL, symbol TEXT NOT NULL,
            interval_seconds INTEGER NOT NULL, since INTEGER NOT NULL, to_ts INTEGER NOT NULL, page INTEGER NOT NULL,
            request_url_without_secrets TEXT NOT NULL, status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
            started_utc TEXT, completed_utc TEXT, http_status INTEGER, response_bytes INTEGER,
            raw_sha256 TEXT, raw_compressed_path TEXT, raw_compressed_bytes INTEGER, raw_compressed_sha256 TEXT,
            row_count INTEGER, first_timestamp INTEGER, last_timestamp INTEGER, schema_hash TEXT,
            parquet_path TEXT, parquet_bytes INTEGER, parquet_sha256 TEXT, validation_status TEXT,
            duplicate_boundary_rows INTEGER DEFAULT 0, more INTEGER, error_class TEXT, error_message TEXT
        )""")
        self.db.commit()

    def add(self, spec: JobSpec) -> None:
        validate_spec(spec)
        self.db.execute("""INSERT OR IGNORE INTO jobs
            (job_id,run_kind,analytics_type,symbol,interval_seconds,since,to_ts,page,request_url_without_secrets,status)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (spec.job_id, spec.run_kind, spec.analytics_type, spec.symbol, spec.interval_seconds,
             spec.since, spec.to, spec.page, request_url(spec), "planned"))
        self.db.commit()

    def reset_stale_running(self) -> int:
        cur = self.db.execute("UPDATE jobs SET status='planned', error_class='stale_running_recovered' WHERE status='running'")
        self.db.commit()
        return cur.rowcount

    def row(self, job_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row

    def update(self, job_id: str, **values: Any) -> None:
        if "status" in values and values["status"] not in STATUSES:
            raise ValueError("invalid job status")
        sql = ",".join(f"{k}=?" for k in values)
        self.db.execute(f"UPDATE jobs SET {sql} WHERE job_id=?", (*values.values(), job_id))
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def verified_complete(self, spec: JobSpec) -> bool:
        row = self.row(spec.job_id)
        recoverable_empty_more = (
            row["status"] == "blocked_error"
            and row["error_message"] == "more=true with no continuation timestamp"
            and row["row_count"] == 0
            and row["more"] == 1
            and row["validation_status"] == "hash_schema_bounds_pass"
        )
        if recoverable_empty_more:
            self.update(spec.job_id, status="complete", validation_status="hash_schema_bounds_pass_empty_more_no_cursor",
                        error_class="pagination_anomaly", error_message="more=true with empty response; no safe continuation")
            row = self.row(spec.job_id)
        recoverable_materialized = (
            row["status"] != "complete"
            and bool(row["raw_compressed_path"] and row["raw_compressed_sha256"])
            and bool(row["parquet_path"] and row["parquet_sha256"])
        )
        if recoverable_materialized:
            raw = Path(row["raw_compressed_path"])
            parquet = Path(row["parquet_path"])
            if raw.is_file() and parquet.is_file() and sha256_file(raw) == row["raw_compressed_sha256"] and sha256_file(parquet) == row["parquet_sha256"]:
                table = pq.ParquetFile(parquet).read(columns=["timestamp_epoch_seconds", "source_job_id"])
                timestamps = table.column("timestamp_epoch_seconds").to_pylist()
                jobs = set(table.column("source_job_id").to_pylist())
                if (table.num_rows == int(row["row_count"] or 0) and jobs <= {spec.job_id}
                        and all(spec.since <= int(ts) <= spec.to and int(ts) < PROTECTED_START for ts in timestamps)):
                    self.update(spec.job_id, status="complete", validation_status="recovered_materialized_parts_hash_schema_bounds_pass",
                                error_class="crash_recovery", error_message="verified finalized raw/parquet parts rebound after interruption",
                                completed_utc=utc_now())
                    row = self.row(spec.job_id)
        if row["status"] != "complete":
            return False
        for path_key, hash_key in (("raw_compressed_path", "raw_compressed_sha256"), ("parquet_path", "parquet_sha256")):
            path = Path(row[path_key] or "")
            if not path.is_file() or sha256_file(path) != row[hash_key]:
                self.update(spec.job_id, status="blocked_error", error_class="verified_part_hash_mismatch")
                return False
        return True

    def rows(self, run_kind: str | None = None) -> list[dict[str, Any]]:
        query, args = ("SELECT * FROM jobs ORDER BY run_kind,symbol,analytics_type,interval_seconds,since,page", ())
        if run_kind:
            query, args = query.replace(" ORDER", " WHERE run_kind=? ORDER"), (run_kind,)
        return [dict(r) for r in self.db.execute(query, args).fetchall()]


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def raw_path(data_root: Path, spec: JobSpec) -> Path:
    dt = datetime.fromtimestamp(spec.since, timezone.utc)
    return data_root / "raw" / spec.analytics_type / spec.symbol / str(spec.interval_seconds) / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{spec.job_id}.json.zst"


def parquet_path(data_root: Path, spec: JobSpec) -> Path:
    dt = datetime.fromtimestamp(spec.since, timezone.utc)
    return data_root / "normalized" / spec.analytics_type / f"interval={spec.interval_seconds}" / f"symbol={spec.symbol}" / f"year={dt.year:04d}" / f"month={dt.month:02d}" / f"part-{spec.job_id}.parquet"


def compress_zstd(data: bytes) -> bytes:
    return bytes(pa.compress(data, codec="zstd", asbytes=True))


def write_parquet(frame: pd.DataFrame, path: Path, metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **{k.encode(): v.encode() for k, v in metadata.items()}})
    pq.write_table(table, temp, compression="zstd", use_dictionary=True, row_group_size=250_000)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def approved_unit_metrics(statuses: dict[str, str]) -> list[str]:
    allowed = {"verified", "bounded_by_official_schema_and_repository_invariant"}
    return sorted(metric for metric in METRICS if statuses.get(metric) in allowed)


def storage_reserve_pass(total_bytes: int, free_bytes: int, projected_bytes: int, free_inodes: int,
                         projected_inodes: int) -> bool:
    reserve = max(int(total_bytes * 0.25), 50 * 1024 ** 3)
    return free_bytes - int(projected_bytes * 1.25) >= reserve and free_inodes > projected_inodes


def compact_month_parts(parts: list[Path], output: Path) -> None:
    if not parts or output.exists():
        raise ValueError("compaction requires source parts and a new output")
    parents = {part.parent.resolve() for part in parts}
    if len(parents) != 1 or output.parent.resolve() not in parents:
        raise ValueError("compaction cannot cross metric/symbol/interval/month partition")
    frames = [pq.ParquetFile(part).read().to_pandas() for part in sorted(parts)]
    frame = pd.concat(frames, ignore_index=True)
    frame, _ = deduplicate(frame)
    write_parquet(frame, output, {"task_id": TASK_ID, "compaction": "same_partition_atomic_no_source_delete"})
    check = pq.ParquetFile(output).read()
    if check.num_rows != len(frame):
        raise ValueError("compacted Parquet verification failed")


def default_fetch(url: str, timeout: float = 60) -> tuple[int, dict[str, str], bytes]:
    response = requests.get(url, headers={"Accept": "application/json", "User-Agent": "donch-stage7b-acquisition/1.0"}, timeout=timeout)
    return response.status_code, dict(response.headers), response.content


class StopFlag:
    requested = False

    def install(self) -> None:
        def handler(_signum: int, _frame: Any) -> None:
            self.requested = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


class Acquirer:
    def __init__(self, ledger: Ledger, data_root: Path, fetcher: Callable[[str], tuple[int, dict[str, str], bytes]] = default_fetch,
                 max_attempts: int = 3, throttle_seconds: float = 0.25, max_response_bytes: int = 50 * 1024 * 1024,
                 progress_callback: Callable[[JobSpec], bool] | None = None):
        self.ledger, self.data_root, self.fetcher = ledger, data_root, fetcher
        self.max_attempts, self.throttle_seconds, self.max_response_bytes = max_attempts, throttle_seconds, max_response_bytes
        self.progress_callback = progress_callback
        self.stop = StopFlag()

    def run(self, initial: Iterable[JobSpec]) -> None:
        self.stop.install()
        queue = list(initial)
        for spec in queue:
            self.ledger.add(spec)
        index = 0
        while index < len(queue):
            spec = queue[index]
            if self.stop.requested:
                break
            if self.ledger.verified_complete(spec):
                index += 1
                continue
            continuation = self.process(spec)
            if continuation is not None:
                self.ledger.add(continuation)
                queue.append(continuation)
            if self.progress_callback is not None and not self.progress_callback(spec):
                self.stop.requested = True
                break
            index += 1
            if index < len(queue):
                time.sleep(self.throttle_seconds)

    def process(self, spec: JobSpec) -> JobSpec | None:
        row = self.ledger.row(spec.job_id)
        attempt = int(row["attempt_count"] or 0) + 1
        self.ledger.update(spec.job_id, status="running", attempt_count=attempt, started_utc=utc_now(), error_class=None, error_message=None)
        try:
            status, _headers, body = self.fetcher(request_url(spec))
            if len(body) > self.max_response_bytes:
                raise ValueError("response byte cap exceeded")
            if status == 429 or status >= 500:
                if attempt < self.max_attempts:
                    self.ledger.update(spec.job_id, status="retryable_error", http_status=status, response_bytes=len(body))
                    time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))) + random.Random(spec.job_id + str(attempt)).random() * 0.1)
                    return self.process(spec)
                raise RuntimeError(f"repeated retryable HTTP {status}")
            if status in {400, 404}:
                self.ledger.update(spec.job_id, status="blocked_error", http_status=status, response_bytes=len(body),
                                   error_class="unsupported_type_or_symbol", error_message="public endpoint returned unsupported")
                return None
            if status != 200:
                raise RuntimeError(f"unexpected HTTP {status}")
            payload = json.loads(body)
            frame, more = normalized_rows(spec, payload)
            frame["source_schema_hash"] = schema_hash(payload)
            frame, duplicates = deduplicate(frame)
            compressed = compress_zstd(body)
            rp, pp = raw_path(self.data_root, spec), parquet_path(self.data_root, spec)
            if rp.exists() or pp.exists():
                raise FileExistsError("refusing to overwrite unverified existing part")
            atomic_write(rp, compressed)
            self.ledger.update(spec.job_id, status="raw_verified", http_status=status, response_bytes=len(body),
                               raw_sha256=sha256_bytes(body), raw_compressed_path=str(rp), raw_compressed_bytes=len(compressed),
                               raw_compressed_sha256=sha256_file(rp), row_count=len(frame),
                               first_timestamp=int(frame.timestamp_epoch_seconds.min()) if len(frame) else None,
                               last_timestamp=int(frame.timestamp_epoch_seconds.max()) if len(frame) else None,
                               schema_hash=schema_hash(payload), duplicate_boundary_rows=duplicates, more=int(more))
            write_parquet(frame, pp, {"task_id": TASK_ID, "source_job_id": spec.job_id, "raw_sha256": sha256_bytes(body)})
            self.ledger.update(spec.job_id, status="normalized", parquet_path=str(pp), parquet_bytes=pp.stat().st_size, parquet_sha256=sha256_file(pp))
            check = pq.ParquetFile(pp).read()
            if check.num_rows != len(frame) or sha256_file(rp) != self.ledger.row(spec.job_id)["raw_compressed_sha256"]:
                raise ValueError("raw/parquet verification failed")
            validation = "hash_schema_bounds_pass_empty_more_no_cursor" if more and frame.empty else "hash_schema_bounds_pass"
            self.ledger.update(spec.job_id, status="complete", validation_status=validation, completed_utc=utc_now(),
                               error_class="pagination_anomaly" if more and frame.empty else None,
                               error_message="more=true with empty response; no safe continuation" if more and frame.empty else None)
            return continuation_spec(spec, frame, more)
        except PermissionError as exc:
            self.ledger.update(spec.job_id, status="blocked_error", error_class=type(exc).__name__, error_message=str(exc), completed_utc=utc_now())
            raise
        except Exception as exc:
            self.ledger.update(spec.job_id, status="blocked_error", error_class=type(exc).__name__, error_message=str(exc), completed_utc=utc_now())
            raise


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def load_frozen_symbols(freeze_path: Path) -> tuple[list[str], str]:
    value = json.loads(freeze_path.read_text())
    inventory = freeze_path.with_name(freeze_path.name.removesuffix(".freeze.json"))
    if sha256_file(inventory) != value["inventory_sha256"]:
        raise ValueError("frozen inventory hash mismatch")
    return list(value["audit_symbols"]), value["inventory_sha256"]


def load_included_symbols(freeze_path: Path) -> list[str]:
    inventory = freeze_path.with_name(freeze_path.name.removesuffix(".freeze.json"))
    frame = pd.read_csv(inventory)
    included = frame["included"].astype(str).str.lower().eq("true")
    return sorted(frame.loc[included, "PF_symbol"].astype(str).unique())


def export_request_ledger(ledger: Ledger, output: Path) -> None:
    frame = pd.DataFrame(ledger.rows())
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")


def main() -> int:
    parser = argparse.ArgumentParser()
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--phase-a", action="store_true")
    phases.add_argument("--phase-b", action="store_true")
    phases.add_argument("--phase-c", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--approval-plan", type=Path)
    args = parser.parse_args()
    symbols, inventory_hash = load_frozen_symbols(args.freeze)
    ledger = Ledger(args.run_root / "KRAKEN_ANALYTICS_JOB_LEDGER.sqlite")
    stale = ledger.reset_stale_running()
    started = time.monotonic()
    if args.phase_a:
        specs = audit_specs(symbols, "phase_a")
        if args.replay:
            specs += audit_specs(symbols, "phase_a_replay")
    else:
        if args.approval_plan is None:
            raise SystemExit("Phase B/C requires an independently reviewed approval plan")
        metrics = validate_acquisition_approval(args.approval_plan)
        if args.phase_b:
            specs = monthly_specs(load_included_symbols(args.freeze), metrics, 300, "phase_b")
        else:
            specs = monthly_specs(["PF_XBTUSD", "PF_ETHUSD"], metrics, 60, "phase_c")
    Acquirer(ledger, args.data_root).run(specs)
    export_request_ledger(ledger, args.run_root / "KRAKEN_ANALYTICS_REQUEST_LEDGER.parquet")
    summary = {"task_id": TASK_ID, "inventory_hash": inventory_hash, "stale_jobs_reset": stale,
               "jobs": len(ledger.rows()), "peak_rss_bytes": peak_rss_bytes(), "runtime_seconds": time.monotonic() - started}
    (args.run_root / "phase_a_execution_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
