from __future__ import annotations

import csv
import json
import math
import os
import resource
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .canonical import atomic_write_bytes, atomic_write_json, canonical_hash, sha256_file


UTC = timezone.utc
PROTECTED_START_MS = 1_767_225_600_000
ALLOWED_CANDLE_COLUMNS = frozenset({
    "time", "open", "high", "low", "close", "volume", "source_url", "venue_symbol",
    "chunk_start_utc", "chunk_end_utc", "resolution", "historical_backfill",
    "rankable_pre_holdout", "contains_protected_period",
})
EMPTY_CANDLE_COLUMNS = frozenset({
    "candles", "more_candles", "source_url", "venue_symbol", "chunk_start_utc",
    "chunk_end_utc", "resolution", "historical_backfill", "rankable_pre_holdout",
    "contains_protected_period",
})
PROHIBITED_CACHE_TOKENS = frozenset({
    "future_return", "forward_return", "post_entry", "gross_bps", "net_bps", "pnl",
    "candidate_rank", "selection_score", "outer_fold_value", "control_outcome",
})
BASE_SOURCE_ROLES = frozenset({
    "price_and_instrument_source_manifest", "funding_partition_manifest",
    "rankable_funding_package_manifest", "rankable_funding_package",
    "kraken_acquisition_manifest", "campaign_universe_reconciliation",
    "terminal_lifecycle_source_ledger",
})
KDA02B_SOURCE_ROLES = frozenset({
    "stage20_kda02b_event_tape_manifest", "stage20_kda02b_fold_local_thresholds",
    "stage20_kda02b_mechanical_cell_skips", "stage20_kda02b_attempt_registry",
})


class ProductionCacheError(RuntimeError):
    pass


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _absolute(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _record(path: Path, *, role: str, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.relative_to(relative_to).as_posix() if relative_to is not None and path.is_relative_to(relative_to) else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


@dataclass(frozen=True)
class SourcePart:
    dataset: str
    symbol: str
    path: str
    sha256: str
    rows: int
    chunk_start: str
    chunk_end: str

    def payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset, "symbol": self.symbol, "path": self.path,
            "sha256": self.sha256, "rows": self.rows,
            "chunk_start": self.chunk_start, "chunk_end": self.chunk_end,
        }


class ProductionCacheCompiler:
    """Code-owned, payoff-firewalled compiler for the Stage-23 source cache.

    The cache is a content-addressed virtual columnar cache: its artifacts are
    canonical per-symbol indexes over the physically verified Kraken parquet
    parts.  No OHLC payload is copied, and launch resolves only the listed
    hashes.  This preserves the existing source bytes while making formula,
    schema, membership, funding, fold and protected-boundary changes invalidate
    every semantic key.
    """

    FORMULA_REGISTRY = {
        "five_minute_bar": "open_ts=time_ms; close_ts=time_ms+300000; source_close_ts=close_ts; no duplicate time; lexical source-hash tie is prohibited",
        "gap": "expected open_ts increment exactly 300000ms; never bridge",
        "daily_validity": "UTC day; finite OHLC; first/last expected bar; >=274/288 rows; no missing run >15m",
        "canonical_quote_notional": "linear PF base-unit candle volume multiplied by completed trade close; aggregate only completed bars",
        "lagged_liquidity": "median canonical quote-notional over prior 30 completed UTC days, shifted one complete UTC day",
        "membership": "187 bound current-roster/bar-existence cohort; listing age>=30d; known terminal event excludes at and after event",
        "rank": "average ties; lexical symbol is identity-only deterministic ordering, never tie-breaking the score",
        "funding": "exact Stage19 hourly absolute-rate rows; start/end alignments remain separate",
        "row_identity": "sha256(symbol,time,OHLCV,physical_source_sha256)",
        "sort": "symbol UTF-8 ascending, time integer ascending, physical_source_sha256 ascending",
        "duplicate": "same symbol/time with unequal OHLCV is common-integrity failure; byte-equal duplicates collapse to one row with all source hashes retained",
        "missingness": "explicit unavailable_data; no zero fill, forward fill, backfill, or survivor substitution",
        "protected_boundary": "time < 1767225600000 and every decision/entry/actual exit < 2026-01-01T00:00:00Z",
    }

    def __init__(self, authority_path: Path, output_root: Path, repository_root: Path) -> None:
        self.authority_path = authority_path
        self.output_root = output_root
        self.repository_root = repository_root
        self.accessed: list[dict[str, Any]] = []

    def _read_json(self, path: Path, role: str) -> Any:
        self.accessed.append(_record(path, role=role))
        return json.loads(path.read_text(encoding="utf-8"))

    def _authority(self) -> tuple[dict[str, Any], dict[str, Path]]:
        authority = self._read_json(self.authority_path, "execution_input_authority")
        roles: dict[str, Path] = {}
        for source in authority.get("source_records", ()):
            path = _absolute(self.repository_root, source["path"])
            if not path.is_file() or path.stat().st_size != int(source["bytes"]) or sha256_file(path) != source["sha256"]:
                raise ProductionCacheError(f"physical source authority mismatch: {source.get('role')}")
            role = str(source["role"])
            roles[role] = path
            self.accessed.append(_record(path, role=role))
        required = BASE_SOURCE_ROLES | KDA02B_SOURCE_ROLES
        if set(roles) != required:
            raise ProductionCacheError("execution authority source-role inventory is incomplete or broadened")
        return authority, roles

    def _kda02b(self, roles: Mapping[str, Path]) -> dict[str, Any]:
        """Verify Stage-20 pre-outcome inputs without opening event payloads."""
        manifest = self._read_json(roles["stage20_kda02b_event_tape_manifest"], "stage20_kda02b_event_tape_manifest_decoded")
        if manifest.get("status") != "pass" or manifest.get("economic_outcome_reader_opened") is not False or manifest.get("Capitalcom_payload_opened") is not False or int(manifest.get("protected_rows_opened", -1)) != 0:
            raise ProductionCacheError("Stage-20 KDA02B pre-outcome manifest is not outcome-firewalled")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 187:
            raise ProductionCacheError("Stage-20 KDA02B event-tape inventory is not exactly 187 symbols")
        symbols: list[str] = []
        total_rows = 0
        inventory: list[dict[str, Any]] = []
        for row in files:
            if set(row) != {"bytes", "path", "rows", "sha256", "symbol"}:
                raise ProductionCacheError("Stage-20 KDA02B event-tape record schema differs")
            path = Path(str(row["path"]))
            if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                raise ProductionCacheError(f"Stage-20 KDA02B event-tape hash mismatch: {row.get('symbol')}")
            if "/preoutcome/event_tapes/events/" not in path.as_posix():
                raise ProductionCacheError("Stage-20 KDA02B event tape is outside the authorized pre-outcome tree")
            symbols.append(str(row["symbol"])); total_rows += int(row["rows"])
            inventory.append({key: row[key] for key in ("symbol", "rows", "bytes", "sha256")})
        if len(set(symbols)) != 187 or int(manifest.get("event_rows", -1)) != total_rows:
            raise ProductionCacheError("Stage-20 KDA02B event-tape identity/count reconciliation failed")
        thresholds = self._read_json(roles["stage20_kda02b_fold_local_thresholds"], "stage20_kda02b_fold_local_thresholds_decoded")
        skips = self._read_json(roles["stage20_kda02b_mechanical_cell_skips"], "stage20_kda02b_mechanical_cell_skips_decoded")
        attempts = self._read_json(roles["stage20_kda02b_attempt_registry"], "stage20_kda02b_attempt_registry_decoded")
        if thresholds.get("method") != "Hyndman-Fan type 7 on full eligible inner-training PIT observations" or not isinstance(thresholds.get("models"), Mapping):
            raise ProductionCacheError("Stage-20 KDA02B fold-local threshold contract differs")
        if not isinstance(skips.get("skips"), list) or len(skips["skips"]) != int(manifest.get("mechanical_cell_model_skips", -1)):
            raise ProductionCacheError("Stage-20 KDA02B mechanical skip reconciliation failed")
        if not isinstance(attempts, Mapping):
            raise ProductionCacheError("Stage-20 KDA02B attempt registry is invalid")
        return {
            "symbols": len(symbols), "event_rows": total_rows,
            "event_tape_inventory_sha256": canonical_hash(inventory),
            "fold_local_thresholds_sha256": sha256_file(roles["stage20_kda02b_fold_local_thresholds"]),
            "mechanical_skips_sha256": sha256_file(roles["stage20_kda02b_mechanical_cell_skips"]),
            "attempt_registry_sha256": sha256_file(roles["stage20_kda02b_attempt_registry"]),
            "event_payloads_opened": 0, "economic_outcomes_opened": False,
        }

    def _symbols(self, unit_manifest: Mapping[str, Any], universe_path: Path) -> tuple[str, ...]:
        anchors = tuple(sorted(str(item["symbol"]) for item in unit_manifest["anchor_source_files"]))
        if len(anchors) != 187 or int(unit_manifest["campaign_symbols"]) != 187 or len(set(anchors)) != 187:
            raise ProductionCacheError("price authority does not bind exactly 187 unique campaign symbols")
        with universe_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        universe = tuple(sorted(row["PF_symbol"] for row in rows if _bool(row["final_campaign_eligible"])))
        if universe != anchors:
            raise ProductionCacheError("campaign universe reconciliation differs from the price authority")
        return anchors

    def _source_parts(self, manifest_path: Path, symbols: set[str]) -> dict[str, list[SourcePart]]:
        parts: dict[str, list[SourcePart]] = defaultdict(list)
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                dataset = str(row["dataset"])
                symbol = str(row["symbol"])
                if dataset not in {"historical_trade_candles_5m", "historical_mark_candles_5m"} or symbol not in symbols:
                    continue
                if row["status"] != "downloaded" or not _bool(row["rankable_pre_holdout"]) or _bool(row["contains_protected_period"]):
                    continue
                path = Path(row["parquet_path"])
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ProductionCacheError(f"rankable parquet source is absent: {path}")
                expected = str(row["parquet_sha256"])
                if len(expected) != 64:
                    raise ProductionCacheError("acquisition manifest has an invalid parquet hash")
                end = str(row["chunk_end"])
                if end and _parse_iso(end) > datetime(2026, 1, 1, tzinfo=UTC):
                    raise ProductionCacheError("rankable source part crosses the protected boundary")
                parts[symbol].append(SourcePart(dataset, symbol, str(path), expected, int(row["rows"]), str(row["chunk_start"]), end))
        if set(parts) != symbols:
            raise ProductionCacheError("one or more campaign symbols lack rankable trade/mark parts")
        for symbol, source_parts in parts.items():
            datasets = {part.dataset for part in source_parts}
            if datasets != {"historical_trade_candles_5m", "historical_mark_candles_5m"}:
                raise ProductionCacheError(f"trade/mark source pair is incomplete: {symbol}")
            source_parts.sort(key=lambda item: (item.dataset, item.chunk_start, item.chunk_end, item.sha256))
        return parts

    def _verify_parquet_parts(self, parts: Mapping[str, Sequence[SourcePart]], *, physical_hashes: bool) -> dict[str, Any]:
        import pyarrow.parquet as pq
        import numpy as np

        rows = 0; bytes_read = 0; verified = 0; unique_rows = 0; explicit_empty_parts = 0
        stream_inventory: list[dict[str, Any]] = []
        daily_trade_rows: list[dict[str, Any]] = []
        for symbol in sorted(parts):
            by_dataset: dict[str, list[SourcePart]] = defaultdict(list)
            for part in parts[symbol]:
                by_dataset[part.dataset].append(part)
            symbol_streams: dict[str, dict[str, Any]] = {}
            trade_days: dict[int, list[float]] = {}
            trade_day_times: dict[int, list[Any]] = defaultdict(list)
            for dataset, source_parts in sorted(by_dataset.items()):
                previous_time: int | None = None
                previous_row: tuple[float, ...] | None = None
                gaps: list[dict[str, int]] = []
                duplicates = 0
                first_time: int | None = None
                stream_rows = 0
                for part in source_parts:
                    path = Path(part.path)
                    if physical_hashes and sha256_file(path) != part.sha256:
                        raise ProductionCacheError(f"parquet content hash mismatch: {path}")
                    parquet = pq.ParquetFile(path)
                    columns = set(parquet.schema_arrow.names)
                    if columns == EMPTY_CANDLE_COLUMNS:
                        table = parquet.read()
                        payload = table.to_pylist()
                        if part.rows != 1 or payload != [{
                            "candles": [], "more_candles": False,
                            "source_url": payload[0]["source_url"], "venue_symbol": symbol,
                            "chunk_start_utc": payload[0]["chunk_start_utc"], "chunk_end_utc": payload[0]["chunk_end_utc"],
                            "resolution": "5m", "historical_backfill": True,
                            "rankable_pre_holdout": True, "contains_protected_period": False,
                        }]:
                            raise ProductionCacheError("noncanonical empty candle source marker")
                        rows += part.rows; bytes_read += path.stat().st_size; verified += 1; explicit_empty_parts += 1
                        continue
                    if columns != ALLOWED_CANDLE_COLUMNS or any(token in name.lower() for token in PROHIBITED_CACHE_TOKENS for name in columns):
                        raise ProductionCacheError("candle schema is missing, broadened, or contains a prohibited outcome field")
                    if parquet.metadata.num_rows != part.rows:
                        raise ProductionCacheError("parquet row count differs from acquisition manifest")
                    table = parquet.read(columns=["time", "open", "high", "low", "close", "volume"])
                    times = table["time"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                    if len(times) != part.rows or not len(times):
                        raise ProductionCacheError("rankable parquet part is empty or has a decoded row mismatch")
                    if int(times[-1]) >= PROTECTED_START_MS or int(times[0]) < 1_672_531_200_000:
                        raise ProductionCacheError("physical candle timestamp crosses the authorized rankable partition")
                    differences = np.diff(times)
                    if np.any(differences <= 0):
                        raise ProductionCacheError("physical candle part is unsorted or contains an internal duplicate")
                    internal_gaps = np.flatnonzero(differences != 300_000)
                    gaps.extend({"left_ms": int(times[index]), "right_ms": int(times[index + 1])} for index in internal_gaps)
                    current_first = tuple(float(table[name][0].as_py()) for name in ("open", "high", "low", "close", "volume"))
                    current_last = tuple(float(table[name][-1].as_py()) for name in ("open", "high", "low", "close", "volume"))
                    if any(not math.isfinite(value) for value in (*current_first, *current_last)):
                        raise ProductionCacheError("physical candle boundary contains a nonfinite value")
                    duplicate_first = False
                    if previous_time is not None:
                        if int(times[0]) == previous_time:
                            if current_first != previous_row:
                                raise ProductionCacheError("overlapping candle parts disagree at the shared timestamp")
                            duplicates += 1
                            duplicate_first = True
                        elif int(times[0]) < previous_time:
                            raise ProductionCacheError("candle source parts overlap beyond one byte-equal boundary row")
                        elif int(times[0]) - previous_time != 300_000:
                            gaps.append({"left_ms": previous_time, "right_ms": int(times[0])})
                    if dataset == "historical_trade_candles_5m":
                        begin = 1 if duplicate_first else 0
                        selected_times = times[begin:]
                        closes = table["close"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)[begin:]
                        volumes = table["volume"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)[begin:]
                        notionals = closes * volumes
                        if np.any(~np.isfinite(notionals)) or np.any(notionals < 0):
                            raise ProductionCacheError("canonical quote-notional input is nonfinite or negative")
                        day_values = selected_times // 86_400_000
                        unique_days, starts, counts = np.unique(day_values, return_index=True, return_counts=True)
                        sums = np.add.reduceat(notionals, starts)
                        for day, start, count, subtotal in zip(unique_days, starts, counts, sums):
                            day = int(day); start = int(start); count = int(count)
                            aggregate = trade_days.setdefault(day, [0.0, 0.0, float(selected_times[start]), float(selected_times[start + count - 1])])
                            aggregate[0] += float(subtotal); aggregate[1] += count
                            aggregate[2] = min(aggregate[2], float(selected_times[start])); aggregate[3] = max(aggregate[3], float(selected_times[start + count - 1]))
                            trade_day_times[day].append(selected_times[start:start + count])
                    first_time = int(times[0]) if first_time is None else first_time
                    previous_time = int(times[-1]); previous_row = current_last
                    rows += part.rows; bytes_read += path.stat().st_size; verified += 1; stream_rows += part.rows
                stream = {
                    "symbol": symbol, "dataset": dataset,
                    "physical_rows": stream_rows, "byte_equal_boundary_duplicates": duplicates,
                    "unique_rows": stream_rows - duplicates, "first_time_ms": first_time,
                    "last_time_ms": previous_time, "temporal_gap_count": len(gaps),
                    "temporal_gap_inventory_sha256": canonical_hash(gaps),
                }
                stream_inventory.append(stream); symbol_streams[dataset] = stream
                unique_rows += stream["unique_rows"]
            trade = symbol_streams["historical_trade_candles_5m"]
            mark = symbol_streams["historical_mark_candles_5m"]
            comparison_fields = ("unique_rows", "first_time_ms", "last_time_ms", "temporal_gap_count", "temporal_gap_inventory_sha256")
            stream_inventory.append({
                "symbol": symbol, "dataset": "trade_mark_schedule_reconciliation",
                "matching_fields": [field for field in comparison_fields if trade[field] == mark[field]],
                "differing_fields": [field for field in comparison_fields if trade[field] != mark[field]],
                "missingness_behavior": "decision or accounting path requiring an absent exact stream timestamp is unavailable_data; never fill or substitute",
            })
            for day, (quote_notional, count, first, last) in sorted(trade_days.items()):
                first_ms = int(first); last_ms = int(last); count_i = int(count)
                timestamps = np.unique(np.concatenate(trade_day_times[day]))
                maximum_gap_ms = int(np.diff(timestamps).max()) if len(timestamps) > 1 else 0
                daily_trade_rows.append({
                    "symbol": symbol, "day_open_ms": day * 86_400_000,
                    "canonical_quote_notional": quote_notional, "five_minute_rows": count_i,
                    "first_open_ms": first_ms, "last_open_ms": last_ms,
                    "maximum_gap_ms": maximum_gap_ms,
                    "valid_day": count_i >= 274 and first_ms == day * 86_400_000 and last_ms == day * 86_400_000 + 86_100_000 and maximum_gap_ms <= 900_000,
                })
        return {
            "physical_parts": verified, "source_rows": rows, "unique_source_rows": unique_rows,
            "source_bytes": bytes_read, "stream_inventory": stream_inventory,
            "stream_inventory_sha256": canonical_hash(stream_inventory),
            "explicit_empty_prelisting_parts": explicit_empty_parts,
            "daily_trade_rows": daily_trade_rows,
        }

    def _funding(self, path: Path, symbols: Sequence[str]) -> dict[str, Any]:
        inventory = []
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ProductionCacheError(f"funding ZIP CRC failure: {bad}")
            names = set(archive.namelist())
            for symbol in symbols:
                name = f"rankable_2023_2025/{symbol}.csv"
                if name not in names:
                    raise ProductionCacheError(f"exact funding partition missing symbol: {symbol}")
                with archive.open(name) as source:
                    header = source.readline().decode("utf-8").strip().split(",")
                    if header != ["timestamp", "tradeable", "absolute_rate", "relative_rate"]:
                        raise ProductionCacheError("funding partition schema differs from Stage19")
                    count = 0; first = None; last = None
                    for raw in source:
                        if not raw.strip():
                            continue
                        fields = raw.decode("utf-8").rstrip("\n").split(",")
                        if len(fields) != 4 or fields[1] != symbol:
                            raise ProductionCacheError("funding row identity is invalid")
                        timestamp = _parse_iso(fields[0].replace(" ", "T") + "Z")
                        if timestamp >= datetime(2026, 1, 1, tzinfo=UTC):
                            raise ProductionCacheError("funding partition contains a protected row")
                        rate = float(fields[2])
                        if not math.isfinite(rate):
                            raise ProductionCacheError("funding partition contains a nonfinite absolute rate")
                        first = first or timestamp; last = timestamp; count += 1
                inventory.append({"symbol": symbol, "rows": count, "first_timestamp": first.isoformat() if first else None, "last_timestamp": last.isoformat() if last else None})
        return {"symbols": len(inventory), "rows": sum(item["rows"] for item in inventory), "inventory_sha256": canonical_hash(inventory), "inventory": inventory}

    def _pit_membership(self, universe_path: Path, lifecycle_ledger_path: Path, symbols: Sequence[str], daily_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        with universe_path.open("r", encoding="utf-8", newline="") as handle:
            universe_rows = {row["PF_symbol"]: row for row in csv.DictReader(handle) if _bool(row["final_campaign_eligible"])}
        if set(universe_rows) != set(symbols):
            raise ProductionCacheError("PIT lifecycle inventory differs from the exact campaign symbols")
        with lifecycle_ledger_path.open("r", encoding="utf-8", newline="") as handle:
            ledger_rows = list(csv.DictReader(handle))
        if len(ledger_rows) != 1 or ledger_rows[0].get("source_kind") != "official_kraken_terminal_event_ledger":
            raise ProductionCacheError("terminal lifecycle source ledger schema or cardinality differs")
        raw_lifecycle = _absolute(self.repository_root, ledger_rows[0]["raw_path"])
        if not raw_lifecycle.is_file() or raw_lifecycle.stat().st_size != int(ledger_rows[0]["bytes"]) or sha256_file(raw_lifecycle) != ledger_rows[0]["sha256"]:
            raise ProductionCacheError("terminal lifecycle physical source differs from its ledger")
        self.accessed.append(_record(raw_lifecycle, role="terminal_lifecycle_physical_source"))
        lifecycle: dict[str, tuple[int, int | None]] = {}
        for symbol in symbols:
            row = universe_rows[symbol]
            opening = int(_parse_iso(row["opening_or_start"]).timestamp() * 1000)
            terminal = int(_parse_iso(row["terminal_or_end"]).timestamp() * 1000) if row["terminal_or_end"] else None
            lifecycle[symbol] = (opening, terminal)
        by_symbol: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
        for row in daily_rows:
            by_symbol[str(row["symbol"])][int(row["day_open_ms"])] = row
        day_ms = 86_400_000
        rankable_days = range(1_672_531_200_000, PROTECTED_START_MS, day_ms)
        output_rows: list[dict[str, Any]] = []
        unavailable_counts: defaultdict[str, int] = defaultdict(int)
        for day in rankable_days:
            notionals: dict[str, float] = {}
            for symbol in symbols:
                opening, terminal = lifecycle[symbol]
                if day < opening + 30 * day_ms or (terminal is not None and day >= terminal):
                    unavailable_counts["lifecycle_or_listing_age"] += 1; continue
                current = by_symbol[symbol].get(day)
                history = [by_symbol[symbol].get(day - offset * day_ms) for offset in range(1, 31)]
                if current is None or int(current["five_minute_rows"]) <= 0:
                    unavailable_counts["current_day_absent"] += 1; continue
                if any(row is None or not row["valid_day"] for row in history):
                    unavailable_counts["lagged_30d_history_incomplete"] += 1; continue
                values = sorted(float(row["canonical_quote_notional"]) for row in history if row is not None)
                value = median(values)
                if not math.isfinite(value) or value <= 0:
                    unavailable_counts["lagged_quote_notional_invalid"] += 1; continue
                notionals[symbol] = value
            groups: dict[float, list[str]] = defaultdict(list)
            for symbol, value in notionals.items():
                groups[value].append(symbol)
            ranks: dict[str, float] = {}
            position = 1
            for value in sorted(groups, reverse=True):
                members = sorted(groups[value])
                rank = (position + position + len(members) - 1) / 2.0
                ranks.update({symbol: rank for symbol in members}); position += len(members)
            for symbol in sorted(notionals, key=lambda item: (ranks[item], item)):
                output_rows.append({
                    "day_open_ms": day, "symbol": symbol,
                    "lagged_30d_median_canonical_quote_notional": notionals[symbol],
                    "average_liquidity_rank": ranks[symbol],
                    "eligible_population": len(notionals),
                    "top_10": ranks[symbol] <= 10, "top_20": ranks[symbol] <= 20, "top_40": ranks[symbol] <= 40,
                    "decision_count_5m": int(by_symbol[symbol][day]["five_minute_rows"]),
                })
        if not output_rows:
            raise ProductionCacheError("PIT membership compiler produced no eligible decisions")
        root = self.output_root / "pit"
        path = root / "PIT_DAILY_MEMBERSHIP.jsonl"
        atomic_write_bytes(path, b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n" for row in output_rows))
        artifact = {"path": path.relative_to(self.output_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        return {
            "daily_source_rows": len(daily_rows), "membership_rows": len(output_rows),
            "eligible_decisions_5m": sum(int(row["decision_count_5m"]) for row in output_rows),
            "unavailable_counts": dict(sorted(unavailable_counts.items())),
            "lifecycle_inventory_sha256": canonical_hash({key: value for key, value in sorted(lifecycle.items())}),
            "terminal_lifecycle_ledger_sha256": sha256_file(lifecycle_ledger_path), "terminal_lifecycle_physical_source_sha256": sha256_file(raw_lifecycle),
            "membership_content_sha256": canonical_hash(output_rows), "artifact": artifact,
            "average_tie_rank": True, "liquidity_lag_days": 1, "liquidity_window_days": 30,
        }

    def _sample_rows(self, parts: Mapping[str, Sequence[SourcePart]]) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq

        selected_symbols = (sorted(parts)[0], sorted(parts)[len(parts) // 2], sorted(parts)[-1])
        samples = []
        for symbol in selected_symbols:
            trade = [
                part for part in parts[symbol]
                if part.dataset == "historical_trade_candles_5m"
                and set(pq.ParquetFile(part.path).schema_arrow.names) == ALLOWED_CANDLE_COLUMNS
            ]
            if not trade:
                raise ProductionCacheError(f"sample symbol has no physical candle rows: {symbol}")
            for part in (trade[0], trade[len(trade) // 2], trade[-1]):
                table = pq.read_table(part.path, columns=["time", "open", "high", "low", "close", "volume"])
                indexes = sorted({0, table.num_rows // 2, table.num_rows - 1})
                for index in indexes:
                    row = {name: table[name][index].as_py() for name in table.column_names}
                    timestamp = int(row["time"])
                    if timestamp >= PROTECTED_START_MS:
                        raise ProductionCacheError("sampled candle crosses protected boundary")
                    numeric = [float(row[name]) for name in ("open", "high", "low", "close", "volume")]
                    if any(not math.isfinite(value) for value in numeric) or min(numeric[:4]) <= 0:
                        raise ProductionCacheError("sampled candle has invalid numeric content")
                    samples.append({"symbol": symbol, "source_sha256": part.sha256, "row_index": index, "time": timestamp, "row_identity_sha256": canonical_hash([symbol, timestamp, *map(str, numeric), part.sha256])})
        return samples

    def _fold_reconciliation(self, fold_graph: Mapping[str, Any], pit_artifact: Mapping[str, Any]) -> dict[str, Any]:
        path = self.output_root / str(pit_artifact["path"])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        partitions: list[dict[str, Any]] = []
        for outer in fold_graph["outer_folds"]:
            outer_start = int(_parse_iso(outer["outer_evaluation_start"]).timestamp() * 1000)
            outer_end = int(_parse_iso(outer["outer_evaluation_end_exclusive"]).timestamp() * 1000)
            outer_count = sum(int(row["decision_count_5m"]) for row in rows if outer_start <= int(row["day_open_ms"]) < outer_end)
            partitions.append({
                "phase": "outer_evaluation", "outer_fold_id": outer["outer_fold_id"], "inner_fold_id": None,
                "evaluation_start": outer["outer_evaluation_start"], "evaluation_end_exclusive": outer["outer_evaluation_end_exclusive"],
                "purge_days": outer["purge_days"], "embargo_days": outer["embargo_days"], "eligible_decisions_5m": outer_count,
            })
            for inner in outer["inner_folds"]:
                start = int(_parse_iso(inner["validation_start"]).timestamp() * 1000)
                end = int(_parse_iso(inner["validation_end_exclusive"]).timestamp() * 1000)
                count = sum(int(row["decision_count_5m"]) for row in rows if start <= int(row["day_open_ms"]) < end)
                partitions.append({
                    "phase": "inner_validation", "outer_fold_id": outer["outer_fold_id"], "inner_fold_id": inner["inner_fold_id"],
                    "training_start": inner["training_start"], "training_end_exclusive": inner["training_latest_exit_exclusive"],
                    "evaluation_start": inner["validation_start"], "evaluation_end_exclusive": inner["validation_end_exclusive"],
                    "purge_before_validation_days": inner["purge_before_validation_days"], "eligible_decisions_5m": count,
                })
        if len(partitions) != 8 + sum(len(outer["inner_folds"]) for outer in fold_graph["outer_folds"]) or any(row["eligible_decisions_5m"] <= 0 for row in partitions):
            raise ProductionCacheError("fold/decision reconciliation is incomplete")
        return {
            "outer": 8, "inner_total": len(partitions) - 8, "partitions": partitions,
            "partition_inventory_sha256": canonical_hash(partitions), "empty_fold_rule": fold_graph["empty_fold_rule"],
        }

    def build(self, *, physical_hashes: bool = True) -> dict[str, Any]:
        started_wall = time.perf_counter(); started_cpu = time.process_time()
        authority, roles = self._authority()
        kda02b = self._kda02b(roles)
        unit = self._read_json(roles["price_and_instrument_source_manifest"], "price_unit_manifest")
        symbols = self._symbols(unit, roles["campaign_universe_reconciliation"])
        parts = self._source_parts(roles["kraken_acquisition_manifest"], set(symbols))
        compiler_identity = canonical_hash({
            "authority_sha256": sha256_file(self.authority_path), "compiler_sha256": sha256_file(Path(__file__)),
            "formula_registry": self.FORMULA_REGISTRY, "physical_hashes": physical_hashes,
        })
        state_path = self.output_root / "BUILD_STATE.json"
        checkpoint = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        source_path = self.output_root / "SOURCE_VERIFICATION.json"; pit_path = self.output_root / "PIT_SUMMARY.json"
        resumable = (
            checkpoint.get("state") in {"pit_committed", "committed"} and checkpoint.get("compiler_identity_sha256") == compiler_identity
            and source_path.is_file() and pit_path.is_file()
            and sha256_file(source_path) == checkpoint.get("source_verification_sha256")
            and sha256_file(pit_path) == checkpoint.get("pit_summary_sha256")
        )
        if resumable:
            verified = json.loads(source_path.read_text(encoding="utf-8")); pit = json.loads(pit_path.read_text(encoding="utf-8"))
            pit_artifact = self.output_root / str(pit["artifact"]["path"])
            if not pit_artifact.is_file() or pit_artifact.stat().st_size != pit["artifact"]["bytes"] or sha256_file(pit_artifact) != pit["artifact"]["sha256"]:
                raise ProductionCacheError("resumable PIT checkpoint artifact drift")
        else:
            atomic_write_json(state_path, {"schema": "stage23_production_cache_build_state_v1", "state": "source_verification_running", "compiler_identity_sha256": compiler_identity})
            verified = self._verify_parquet_parts(parts, physical_hashes=physical_hashes)
            daily_trade_rows = verified.pop("daily_trade_rows")
            pit = self._pit_membership(roles["campaign_universe_reconciliation"], roles["terminal_lifecycle_source_ledger"], symbols, daily_trade_rows)
            atomic_write_json(source_path, verified); atomic_write_json(pit_path, pit)
            atomic_write_json(state_path, {
                "schema": "stage23_production_cache_build_state_v1", "state": "pit_committed",
                "compiler_identity_sha256": compiler_identity, "source_verification_sha256": sha256_file(source_path),
                "pit_summary_sha256": sha256_file(pit_path), "pit_artifact_sha256": pit["artifact"]["sha256"],
            })
        funding = self._funding(roles["rankable_funding_package"], symbols)
        fold_graph_path = self.authority_path.parent / "FOLD_GRAPH.json"
        fold_graph = self._read_json(fold_graph_path, "fold_graph")
        if len(fold_graph.get("outer_folds", ())) != 8 or sha256_file(fold_graph_path) != authority["fold_graph_sha256"]:
            raise ProductionCacheError("fold graph does not bind exactly eight outer folds")
        folds = self._fold_reconciliation(fold_graph, pit["artifact"])
        semantic_inputs = {
            "authority_sha256": sha256_file(self.authority_path),
            "source_record_inventory_sha256": canonical_hash(authority["source_records"]),
            "formula_registry": self.FORMULA_REGISTRY,
            "candle_schema": sorted(ALLOWED_CANDLE_COLUMNS),
            "rankable_interval": authority["rankable_interval"],
            "protected_cutoff": "2026-01-01T00:00:00Z",
            "pit_universe_sha256": authority["pit_universe_sha256"],
            "funding_manifest_sha256": authority["funding_manifest_sha256"],
            "fold_graph_sha256": authority["fold_graph_sha256"],
        }
        artifact_root = self.output_root / "symbol_indexes"
        artifact_rows = []
        for symbol in symbols:
            payload = {
                "schema": "stage23_production_symbol_cache_index_v1", "symbol": symbol,
                "semantic_inputs_sha256": canonical_hash(semantic_inputs),
                "source_parts": [part.payload() for part in parts[symbol]],
                "source_part_inventory_sha256": canonical_hash([part.payload() for part in parts[symbol]]),
                "row_identity_rule": self.FORMULA_REGISTRY["row_identity"],
                "protected_cutoff": semantic_inputs["protected_cutoff"],
            }
            path = artifact_root / f"{symbol}.json"
            atomic_write_json(path, payload)
            artifact_rows.append({"symbol": symbol, "path": path.relative_to(self.output_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path), "semantic_cache_key": canonical_hash({"symbol": symbol, **semantic_inputs, "parts": payload["source_part_inventory_sha256"]})})
        samples = self._sample_rows(parts)
        elapsed = time.perf_counter() - started_wall
        manifest = {
            "schema": "stage23_production_semantic_cache_manifest_v1",
            "platform": "kraken_native_linear_pf",
            "construction": "code-owned content-addressed virtual columnar cache over physically verified Kraken parts",
            "authority_sha256": sha256_file(self.authority_path),
            "semantic_inputs": semantic_inputs,
            "campaign_symbols": len(symbols),
            "source_verification": verified,
            "pit_membership": pit,
            "stage20_kda02b_preoutcome": kda02b,
            "funding": {key: value for key, value in funding.items() if key != "inventory"},
            "funding_inventory": funding["inventory"],
            "folds": folds,
            "artifacts": [pit["artifact"], *artifact_rows],
            "artifact_inventory_sha256": canonical_hash([pit["artifact"], *artifact_rows]),
            "sample_recomputation": samples,
            "sample_recomputation_sha256": canonical_hash(samples),
            "canonical_sort": self.FORMULA_REGISTRY["sort"],
            "duplicate_rule": self.FORMULA_REGISTRY["duplicate"],
            "gap_rule": self.FORMULA_REGISTRY["gap"],
            "missingness_rule": self.FORMULA_REGISTRY["missingness"],
            "prohibited_cache_columns": sorted(PROHIBITED_CACHE_TOKENS),
            "outcome_firewall": {"post_entry_payoff_reader": "closed", "candidate_ranking_reader": "closed", "protected_rows_opened": 0, "capitalcom_payload_opened": False},
            "build_metrics": {"cache_bytes": pit["artifact"]["bytes"] + sum(row["bytes"] for row in artifact_rows), "source_bytes_verified": verified["source_bytes"], "source_rows_indexed": verified["source_rows"]},
            "access_inventory_sha256": canonical_hash(self.accessed),
            "status": "pass",
        }
        atomic_write_json(self.output_root / "PRODUCTION_CACHE_MANIFEST.json", manifest)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        atomic_write_json(self.output_root / "BUILD_METRICS.json", {
            "schema": "stage23_production_cache_build_metrics_v1", "wall_seconds": elapsed,
            "cpu_seconds": time.process_time() - started_cpu, "maximum_rss_bytes": int(usage.ru_maxrss) * 1024,
            "cache_bytes": manifest["build_metrics"]["cache_bytes"],
            "physical_read_bytes": verified["source_bytes"], "source_rows": verified["source_rows"],
            "status": "pass",
        })
        atomic_write_json(self.output_root / "BUILD_STATE.json", {"schema": "stage23_production_cache_build_state_v1", "state": "committed", "compiler_identity_sha256": compiler_identity, "source_verification_sha256": sha256_file(source_path), "pit_summary_sha256": sha256_file(pit_path), "pit_artifact_sha256": pit["artifact"]["sha256"], "artifact_inventory_sha256": manifest["artifact_inventory_sha256"], "manifest_sha256": sha256_file(self.output_root / "PRODUCTION_CACHE_MANIFEST.json")})
        return manifest


def replay_audit(first_root: Path, second_root: Path) -> dict[str, Any]:
    left = json.loads((first_root / "PRODUCTION_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    right = json.loads((second_root / "PRODUCTION_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    stable_fields = ("semantic_inputs", "campaign_symbols", "source_verification", "pit_membership", "stage20_kda02b_preoutcome", "funding", "funding_inventory", "folds", "artifacts", "artifact_inventory_sha256", "sample_recomputation", "sample_recomputation_sha256")
    mismatches = [field for field in stable_fields if left[field] != right[field]]
    byte_identical_manifests = sha256_file(first_root / "PRODUCTION_CACHE_MANIFEST.json") == sha256_file(second_root / "PRODUCTION_CACHE_MANIFEST.json")
    return {"schema": "stage23_production_cache_replay_audit_v1", "stable_fields": list(stable_fields), "mismatches": mismatches, "byte_identical_manifests": byte_identical_manifests, "byte_identical_artifacts": left["artifacts"] == right["artifacts"], "equivalent_columnar_sources": left["source_verification"] == right["source_verification"], "pass": not mismatches and byte_identical_manifests}


__all__ = ["ALLOWED_CANDLE_COLUMNS", "PROHIBITED_CACHE_TOKENS", "ProductionCacheCompiler", "ProductionCacheError", "replay_audit"]
