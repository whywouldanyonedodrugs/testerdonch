from __future__ import annotations

import json
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a1_state import initial_state
from .canonical import canonical_hash, sha256_file
from .engine_types import ContextInputs, FamilyInput, FundingInput, KRAKEN_PLATFORM, PROTECTED_START, SignalBar
from .family_engines import a1_compression, a3_starter_retest
from .production_cache import SourcePart
from .production_inputs import (
    ProductionInputError,
    _a2_proximity_feature_arrays,
    _context,
    _load_daily_bars,
    _load_trade_bars,
    _snapshot,
    _thresholds,
)
from .production_population_tables import A1PopulationTableAuthority, A3PopulationTableAuthority, PopulationTableError
from .synthetic import with_source_authority


UTC = timezone.utc
RANKABLE_START = datetime(2023, 1, 1, tzinfo=UTC)
SUPPORTED_FAMILIES = frozenset({
    "A4_TSMOM_V7",
    "A1_COMPRESSION_V2",
    "A2_PRIOR_HIGH_RS_CONTEXT_V1",
    "A3_STARTER_RETEST_V3",
})


class LazyProductionInputError(ProductionInputError):
    """A requested production locator cannot be proven from bound authorities."""


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise LazyProductionInputError("production locator timestamp is timezone-naive")
    return result.astimezone(UTC)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LazyProductionInputError(f"{label} is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise LazyProductionInputError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LazyProductionInputError(f"{label} is not a JSON object")
    return payload


def _resolve(root: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class FamilyDecisionLocator:
    family_id: str
    phase: str
    outer_fold_id: str
    inner_fold_id: str | None
    symbol: str
    decision_ts: datetime
    executable_attempt_id: str | None = None
    canonical_economic_address_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_ts", _utc(self.decision_ts))
        if self.family_id not in SUPPORTED_FAMILIES:
            raise LazyProductionInputError("lazy A1-A4 adapter received an unsupported family")
        if self.phase not in {"inner_validation", "outer_evaluation"}:
            raise LazyProductionInputError("production locator phase is invalid")
        if self.phase == "inner_validation" and not self.inner_fold_id:
            raise LazyProductionInputError("inner-validation locator lacks its inner fold")
        if self.phase == "outer_evaluation" and self.inner_fold_id is not None:
            raise LazyProductionInputError("outer-evaluation locator unexpectedly names an inner fold")
        if not self.symbol.startswith("PF_") or self.symbol.upper().startswith("CAPITAL"):
            raise LazyProductionInputError("production locator is not a Kraken PF symbol")
        if (self.executable_attempt_id is None) != (self.canonical_economic_address_sha256 is None):
            raise LazyProductionInputError("locator configuration identity is only partially bound")
        if self.canonical_economic_address_sha256 is not None and (
            len(self.canonical_economic_address_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.canonical_economic_address_sha256)
        ):
            raise LazyProductionInputError("locator economic-address hash is invalid")

    @property
    def partition_key(self) -> tuple[str, str, str | None]:
        return self.phase, self.outer_fold_id, self.inner_fold_id

    def identity_payload(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "phase": self.phase,
            "outer_fold_id": self.outer_fold_id,
            "inner_fold_id": self.inner_fold_id,
            "symbol": self.symbol,
            "decision_ts": self.decision_ts.isoformat(),
            "executable_attempt_id": self.executable_attempt_id,
            "canonical_economic_address_sha256": self.canonical_economic_address_sha256,
        }


def _fold_partitions(fold_graph: Mapping[str, Any]) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    output: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    outer_folds = fold_graph.get("outer_folds")
    if not isinstance(outer_folds, list) or len(outer_folds) != 8:
        raise LazyProductionInputError("fold graph is not the frozen eight-outer graph")
    for outer in outer_folds:
        outer_id = str(outer["outer_fold_id"])
        evaluation_start = _utc(outer["outer_evaluation_start"])
        evaluation_end = _utc(outer["outer_evaluation_end_exclusive"])
        outer_row = {
            "phase": "outer_evaluation",
            "outer_fold_id": outer_id,
            "inner_fold_id": None,
            "training_start": _utc(outer["development_start"]),
            "training_end_exclusive": evaluation_start - timedelta(days=int(outer["purge_days"])),
            "evaluation_start": evaluation_start,
            "evaluation_end_exclusive": evaluation_end,
        }
        output[("outer_evaluation", outer_id, None)] = outer_row
        inner_folds = outer.get("inner_folds")
        if not isinstance(inner_folds, list):
            raise LazyProductionInputError("outer fold has no inner-fold inventory")
        for inner in inner_folds:
            inner_id = str(inner["inner_fold_id"])
            row = {
                "phase": "inner_validation",
                "outer_fold_id": outer_id,
                "inner_fold_id": inner_id,
                "training_start": _utc(inner["training_start"]),
                "training_end_exclusive": _utc(inner["training_latest_exit_exclusive"]),
                "evaluation_start": _utc(inner["validation_start"]),
                "evaluation_end_exclusive": _utc(inner["validation_end_exclusive"]),
            }
            key = ("inner_validation", outer_id, inner_id)
            if key in output:
                raise LazyProductionInputError("fold graph contains a duplicate partition")
            output[key] = row
    if len(output) != 132:
        raise LazyProductionInputError("fold graph is not the frozen 124-inner/eight-outer graph")
    return output


class LazyProductionFamilyInputAdapter:
    """Resolve one A1-A4 ``FamilyInput`` from the full verified population.

    The Stage-23 cache remains virtual: this adapter reads only the requested
    symbol's authorized parquet ranges, plus the point-in-time context needed
    by the registered family engines.  A1/A3 threshold vectors remain mmap
    views over their existing tables.  No payoff or post-entry outcome reader
    is reachable from this module.
    """

    def __init__(
        self,
        *,
        repository_root: Path,
        execution_authority_path: Path,
        virtual_cache_manifest_path: Path,
        virtual_cache_root: Path,
        fold_graph_path: Path,
        a1_population_manifest_path: Path,
        a3_population_manifest_path: Path,
        expected_sha256: Mapping[str, str],
        maximum_threshold_cache_entries: int = 1,
        construction_mode: str = "shadow_no_outcome",
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.execution_authority_path = execution_authority_path.resolve()
        self.virtual_cache_manifest_path = virtual_cache_manifest_path.resolve()
        self.virtual_cache_root = virtual_cache_root.resolve()
        self.fold_graph_path = fold_graph_path.resolve()
        self.a1_population_manifest_path = a1_population_manifest_path.resolve()
        self.a3_population_manifest_path = a3_population_manifest_path.resolve()
        if construction_mode != "shadow_no_outcome":
            raise LazyProductionInputError(
                "Stage24 adapter permits only shadow_no_outcome; authorized economic post-entry bars require a separate launch adapter"
            )
        self.construction_mode = construction_mode
        if maximum_threshold_cache_entries != 1:
            raise LazyProductionInputError("threshold cache must remain bounded to one locator")
        self._verify_explicit_hashes(expected_sha256)
        self.authority = _read_json(self.execution_authority_path, label="execution input authority")
        self.virtual_manifest = _read_json(self.virtual_cache_manifest_path, label="Stage23 virtual cache manifest")
        self.fold_graph = _read_json(self.fold_graph_path, label="fold graph")
        self._verify_authorities()
        self.partitions = _fold_partitions(self.fold_graph)
        self._artifact_by_symbol = {
            str(row["symbol"]): dict(row)
            for row in self.virtual_manifest["artifacts"]
            if row.get("symbol")
        }
        self._pit_record = next(
            (dict(row) for row in self.virtual_manifest["artifacts"] if row.get("path") == "pit/PIT_DAILY_MEMBERSHIP.jsonl"),
            None,
        )
        if self._pit_record is None:
            raise LazyProductionInputError("virtual cache omits PIT membership")
        self._pit_by_day = self._load_pit()
        a1_population_root = self.a1_population_manifest_path.parent.parent.parent
        a3_population_root = self.a3_population_manifest_path.parent.parent.parent
        try:
            self._a1 = A1PopulationTableAuthority(a1_population_root, self.a1_population_manifest_path)
            self._a3 = A3PopulationTableAuthority(a3_population_root, self.a3_population_manifest_path)
        except (OSError, KeyError, ValueError, PopulationTableError) as exc:
            raise LazyProductionInputError("population-table authority verification failed") from exc
        self._population_roots = {
            "A1_COMPRESSION_V2": a1_population_root,
            "A3_STARTER_RETEST_V3": a3_population_root,
        }
        self._index_cache: dict[str, tuple[SourcePart, ...]] = {}
        self._verified_source_paths: set[str] = set()
        self._daily_cache: dict[tuple[str, datetime], tuple[Any, ...]] = {}
        self._bars_cache: OrderedDict[tuple[str, datetime], tuple[Any, ...]] = OrderedDict()
        self._funding_cache: OrderedDict[tuple[str, datetime], tuple[tuple[datetime, str, float], ...]] = OrderedDict()
        self._threshold_cache: OrderedDict[tuple[Any, ...], tuple[dict[str, Any], list[dict[str, Any]]]] = OrderedDict()

    @classmethod
    def from_launch_population_authority(
        cls,
        *,
        launch_authority_path: Path,
        launch_authority_sha256: str,
        repository_root: Path,
        construction_mode: str = "shadow_no_outcome",
    ) -> "LazyProductionFamilyInputAdapter":
        """Create the adapter from one final, hash-bound launch authority."""
        from .launch_population_authority import validate_launch_population_authority

        if len(launch_authority_sha256) != 64 or sha256_file(launch_authority_path) != launch_authority_sha256:
            raise LazyProductionInputError("launch population authority SHA-256 differs")
        payload = _read_json(launch_authority_path, label="launch population authority")
        try:
            validate_launch_population_authority(payload, verify_files=True)
        except Exception as exc:
            raise LazyProductionInputError("launch population authority validation failed") from exc
        expected_inventory = canonical_hash({key: value for key, value in payload.items() if key != "authority_inventory_sha256"})
        if payload.get("authority_inventory_sha256") != expected_inventory:
            raise LazyProductionInputError("launch population authority inventory differs")
        records = {
            "execution_input_authority": payload["execution_input_authority"],
            "virtual_cache_manifest": payload["virtual_cache"],
            "fold_graph": payload["fold_graph"],
            "a1_population_manifest": payload["a1_population_table"],
            "a3_population_manifest": payload["a3_population_table"],
        }
        return cls(
            repository_root=repository_root,
            execution_authority_path=Path(str(records["execution_input_authority"]["path"])),
            virtual_cache_manifest_path=Path(str(records["virtual_cache_manifest"]["path"])),
            virtual_cache_root=Path(str(records["virtual_cache_manifest"]["root"])),
            fold_graph_path=Path(str(records["fold_graph"]["path"])),
            a1_population_manifest_path=Path(str(records["a1_population_manifest"]["path"])),
            a3_population_manifest_path=Path(str(records["a3_population_manifest"]["path"])),
            expected_sha256={key: str(value["sha256"]) for key, value in records.items()},
            construction_mode=construction_mode,
        )

    def _verify_explicit_hashes(self, expected: Mapping[str, str]) -> None:
        paths = {
            "execution_input_authority": self.execution_authority_path,
            "virtual_cache_manifest": self.virtual_cache_manifest_path,
            "fold_graph": self.fold_graph_path,
            "a1_population_manifest": self.a1_population_manifest_path,
            "a3_population_manifest": self.a3_population_manifest_path,
        }
        if set(expected) != set(paths):
            raise LazyProductionInputError("explicit adapter hash inventory is incomplete or broadened")
        for label, path in paths.items():
            value = str(expected[label])
            if len(value) != 64 or sha256_file(path) != value:
                raise LazyProductionInputError(f"{label} SHA-256 differs")

    def _verify_authorities(self) -> None:
        if self.authority.get("schema") not in {"stage23_execution_input_authority_v1", "stage24_execution_input_authority_v2"}:
            raise LazyProductionInputError("execution input authority schema is unsupported")
        if self.authority.get("platform") != KRAKEN_PLATFORM or self.authority.get("protected_rows_opened") != 0:
            raise LazyProductionInputError("execution input authority is not sealed Kraken rankable input")
        if self.authority.get("economic_outcomes_opened") is not False or self.authority.get("capitalcom_payload_opened") is not False:
            raise LazyProductionInputError("execution input authority outcome firewall is open")
        if self.virtual_manifest.get("schema") != "stage23_production_semantic_cache_manifest_v1" or self.virtual_manifest.get("status") != "pass":
            raise LazyProductionInputError("virtual cache manifest schema/status differs")
        firewall = self.virtual_manifest.get("outcome_firewall", {})
        if firewall.get("protected_rows_opened") != 0 or firewall.get("post_entry_payoff_reader") != "closed" or firewall.get("capitalcom_payload_opened") is not False:
            raise LazyProductionInputError("virtual cache outcome firewall is open")
        artifacts = self.virtual_manifest.get("artifacts")
        if not isinstance(artifacts, list) or canonical_hash(artifacts) != self.virtual_manifest.get("artifact_inventory_sha256"):
            raise LazyProductionInputError("virtual cache artifact inventory differs")
        semantic = self.virtual_manifest.get("semantic_inputs", {})
        for field in ("fold_graph_sha256", "funding_manifest_sha256", "pit_universe_sha256", "source_record_inventory_sha256"):
            authority_value = self.authority.get(field)
            if field == "source_record_inventory_sha256":
                authority_value = canonical_hash(self.authority.get("source_records", ()))
            if semantic.get(field) != authority_value:
                raise LazyProductionInputError(f"virtual cache and execution authority disagree on {field}")
        if sha256_file(self.fold_graph_path) != self.authority.get("fold_graph_sha256"):
            raise LazyProductionInputError("fold graph differs from execution authority")
        roles: set[str] = set()
        for record in (*self.authority.get("source_records", ()), *self.authority.get("kda02b_authority_records", ())):
            role = str(record.get("role", ""))
            if role in roles:
                raise LazyProductionInputError("execution authority contains a duplicate source role")
            roles.add(role)
            path = _resolve(self.repository_root, record.get("path", ""))
            if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
                raise LazyProductionInputError(f"execution authority source differs: {role}")
        required = {"price_and_instrument_source_manifest", "rankable_funding_package", "campaign_universe_reconciliation"}
        if not required <= roles:
            raise LazyProductionInputError("execution authority omits a required price/funding/universe role")
        a1 = _read_json(self.a1_population_manifest_path, label="A1 population manifest")
        a3 = _read_json(self.a3_population_manifest_path, label="A3 population manifest")
        pit_hash = self.virtual_manifest.get("pit_membership", {}).get("membership_content_sha256")
        if a1.get("schema") != "stage24_a1_exact_pit_population_table_v1" or a1.get("pit_content_sha256") != pit_hash or a1.get("protected_rows") != 0:
            raise LazyProductionInputError("A1 population manifest is not bound to virtual PIT authority")
        if a3.get("schema") != "stage24_a3_exact_pit_first_crossing_table_v1" or a3.get("protected_rows") != 0:
            raise LazyProductionInputError("A3 population manifest is invalid")

    def _load_pit(self) -> dict[int, tuple[dict[str, Any], ...]]:
        assert self._pit_record is not None
        path = self.virtual_cache_root / str(self._pit_record["path"])
        if not path.is_file() or path.stat().st_size != int(self._pit_record["bytes"]) or sha256_file(path) != self._pit_record["sha256"]:
            raise LazyProductionInputError("PIT membership physical bytes differ")
        by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[str, int]] = set()
        ordered_rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                symbol = str(row["symbol"]); day = int(row["day_open_ms"])
                key = symbol, day
                if key in seen or day >= int(PROTECTED_START.timestamp() * 1000):
                    raise LazyProductionInputError("PIT membership is duplicated or protected")
                normalized = dict(row)
                seen.add(key); by_day[day].append(normalized); ordered_rows.append(normalized)
        expected_rows = int(self.virtual_manifest["pit_membership"]["membership_rows"])
        if len(seen) != expected_rows or canonical_hash(ordered_rows) != self.virtual_manifest["pit_membership"]["membership_content_sha256"]:
            raise LazyProductionInputError("PIT membership content identity differs")
        return {day: tuple(rows) for day, rows in by_day.items()}

    def iter_decision_locators(
        self,
        *,
        family_id: str,
        partition_key: tuple[str, str, str | None],
        config: Mapping[str, Any],
        executable_attempt_id: str,
        canonical_economic_address_sha256: str,
    ):
        """Yield the deterministic PIT schedule for one frozen configuration.

        A2 is intentionally separate: its locators are the already frozen A1/A3
        parent events selected inside each inner fold.  Callers must construct
        those exact parent-bound locators after parent selection; enumerating
        the generic PIT grid as A2 opportunities would broaden the contract.
        """
        if family_id not in SUPPORTED_FAMILIES:
            raise LazyProductionInputError("locator enumeration received an unsupported family")
        if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
            raise LazyProductionInputError("A2 decision locators are parent-event-bound and are not a generic PIT schedule")
        partition = self.partitions.get(partition_key)
        if partition is None:
            raise LazyProductionInputError("locator enumeration partition is absent")
        top_n = config.get("PIT_liquidity_top_n")
        if top_n not in {10, 20, 40}:
            raise LazyProductionInputError("frozen configuration lacks a supported PIT top-N")
        if not executable_attempt_id:
            raise LazyProductionInputError("locator enumeration lacks executable attempt identity")
        if len(canonical_economic_address_sha256) != 64 or any(character not in "0123456789abcdef" for character in canonical_economic_address_sha256):
            raise LazyProductionInputError("locator enumeration economic address is invalid")
        start = partition["evaluation_start"]; end = partition["evaluation_end_exclusive"]
        for day_ms in sorted(self._pit_by_day):
            day = datetime.fromtimestamp(day_ms / 1000, tz=UTC)
            if not start <= day < end:
                continue
            selected = sorted(
                (row for row in self._pit_by_day[day_ms] if bool(row[f"top_{top_n}"])),
                key=lambda row: (float(row["average_liquidity_rank"]), str(row["symbol"])),
            )
            decision_count = max((int(row["decision_count_5m"]) for row in selected), default=0)
            if any(int(row["decision_count_5m"]) != decision_count for row in selected) or decision_count not in {1, 288}:
                raise LazyProductionInputError("PIT day has a mixed or invalid decision schedule")
            if family_id == "A4_TSMOM_V7":
                rebalance = config.get("rebalance")
                if rebalance not in {"8h", "1d"}:
                    raise LazyProductionInputError("A4 configuration has an invalid rebalance clock")
                offsets = (0,) if rebalance == "1d" or decision_count == 1 else (0, 8 * 60, 16 * 60)
            else:
                offsets = (0,) if decision_count == 1 else tuple(range(0, 24 * 60, 5))
            for minute_offset in offsets:
                decision = day + timedelta(minutes=minute_offset)
                if not start <= decision < end:
                    continue
                for row in selected:
                    yield FamilyDecisionLocator(
                        family_id=family_id,
                        phase=partition_key[0],
                        outer_fold_id=partition_key[1],
                        inner_fold_id=partition_key[2],
                        symbol=str(row["symbol"]),
                        decision_ts=decision,
                        executable_attempt_id=executable_attempt_id,
                        canonical_economic_address_sha256=canonical_economic_address_sha256,
                    )

    def _partition(self, locator: FamilyDecisionLocator) -> dict[str, Any]:
        partition = self.partitions.get(locator.partition_key)
        if partition is None:
            raise LazyProductionInputError("locator partition is absent from the frozen fold graph")
        decision = locator.decision_ts
        if decision.second or decision.microsecond or decision.minute % 5:
            raise LazyProductionInputError("decision locator is not an exact five-minute open")
        if not (partition["evaluation_start"] <= decision < partition["evaluation_end_exclusive"] < PROTECTED_START):
            raise LazyProductionInputError("decision locator is outside its evaluation partition")
        day_ms = int(decision.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        rows = self._pit_by_day.get(day_ms, ())
        selected = next((row for row in rows if row["symbol"] == locator.symbol), None)
        if selected is None:
            raise LazyProductionInputError("symbol-decision locator is not point-in-time eligible")
        decision_count = int(selected["decision_count_5m"])
        if decision_count == 1 and decision != decision.replace(hour=0, minute=0):
            raise LazyProductionInputError("partial-day PIT schedule does not contain this decision")
        if decision_count not in {1, 288}:
            raise LazyProductionInputError("PIT decision schedule is invalid")
        return dict(partition)

    def _parts(self, symbol: str) -> tuple[SourcePart, ...]:
        cached = self._index_cache.get(symbol)
        if cached is not None:
            return cached
        record = self._artifact_by_symbol.get(symbol)
        if record is None:
            raise LazyProductionInputError("symbol is absent from the virtual cache")
        path = self.virtual_cache_root / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise LazyProductionInputError("virtual symbol index bytes differ")
        payload = _read_json(path, label="virtual symbol index")
        if payload.get("schema") != "stage23_production_symbol_cache_index_v1" or payload.get("symbol") != symbol or payload.get("protected_cutoff") != "2026-01-01T00:00:00Z":
            raise LazyProductionInputError("virtual symbol index schema/identity differs")
        if payload.get("semantic_inputs_sha256") != canonical_hash(self.virtual_manifest.get("semantic_inputs", {})):
            raise LazyProductionInputError("virtual symbol index semantic authority differs")
        rows = payload.get("source_parts")
        if not isinstance(rows, list) or canonical_hash(rows) != payload.get("source_part_inventory_sha256"):
            raise LazyProductionInputError("virtual symbol source-part inventory differs")
        parts = tuple(SourcePart(
            str(row["dataset"]), str(row["symbol"]), str(row["path"]), str(row["sha256"]),
            int(row["rows"]), str(row["chunk_start"]), str(row["chunk_end"]),
        ) for row in rows)
        if any(part.symbol != symbol for part in parts):
            raise LazyProductionInputError("virtual symbol index contains a foreign symbol")
        self._index_cache[symbol] = parts
        return parts

    def _verified_trade_parts(self, symbol: str, start: datetime, end: datetime) -> tuple[SourcePart, ...]:
        selected = []
        for part in self._parts(symbol):
            if part.dataset != "historical_trade_candles_5m":
                continue
            part_start = _utc(part.chunk_start)
            part_end = _utc(part.chunk_end) if part.chunk_end else PROTECTED_START
            if part_end < start or part_start >= end:
                continue
            path = Path(part.path)
            identity = str(path)
            if identity not in self._verified_source_paths:
                if not path.is_file() or sha256_file(path) != part.sha256:
                    raise LazyProductionInputError("bound trade-candle source bytes differ")
                self._verified_source_paths.add(identity)
            selected.append(part)
        if not selected:
            raise LazyProductionInputError("requested symbol interval has no authorized trade source")
        return tuple(selected)

    def _bars(self, symbol: str, cutoff: datetime) -> tuple[Any, ...]:
        cutoff = _utc(cutoff)
        key = symbol, cutoff
        cached = self._bars_cache.get(key)
        if cached is not None:
            self._bars_cache.move_to_end(key)
            return cached
        parts = self._verified_trade_parts(symbol, RANKABLE_START, cutoff)
        rows = _load_trade_bars(parts, RANKABLE_START, cutoff)
        if not rows:
            raise LazyProductionInputError("authorized trade stream decoded no bars")
        if any(row.close_ts > cutoff for row in rows):
            raise LazyProductionInputError("trade loader admitted a not-yet-completed bar")
        self._bars_cache[key] = rows
        self._bars_cache.move_to_end(key)
        while len(self._bars_cache) > 3:
            removable = next((item for item in self._bars_cache if item[0] not in {"PF_XBTUSD", "PF_ETHUSD", symbol}), next(iter(self._bars_cache)))
            self._bars_cache.pop(removable)
        return rows

    def _daily(self, symbol: str, cutoff: datetime) -> tuple[Any, ...]:
        cutoff = _utc(cutoff)
        day = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
        availability_boundary = day if cutoff == day else day + timedelta(microseconds=1)
        key = symbol, availability_boundary
        cached = self._daily_cache.get(key)
        if cached is not None:
            return cached
        parts = self._verified_trade_parts(symbol, RANKABLE_START, cutoff)
        rows = tuple(row for row in _load_daily_bars(parts, RANKABLE_START, cutoff) if row.close_ts < cutoff)
        if any(row.close_ts >= cutoff for row in rows):
            raise LazyProductionInputError("daily loader admitted a not-yet-completed day")
        # Daily context changes only at a UTC-day boundary. Keep one requested
        # day so sequential five-minute locators cannot grow memory unbounded.
        if self._daily_cache and next(iter(self._daily_cache))[1] != key[1]:
            self._daily_cache.clear()
        self._daily_cache[key] = rows
        return rows

    def _funding(self, symbol: str, cutoff: datetime) -> tuple[tuple[datetime, str, float], ...]:
        cutoff = _utc(cutoff)
        key = symbol, cutoff
        cached = self._funding_cache.get(key)
        if cached is not None:
            self._funding_cache.move_to_end(key); return cached
        record = next((row for row in self.authority["source_records"] if row.get("role") == "rankable_funding_package"), None)
        if record is None:
            raise LazyProductionInputError("rankable funding package authority is absent")
        rows: list[tuple[datetime, str, float]] = []
        with zipfile.ZipFile(_resolve(self.repository_root, record["path"])) as archive:
            with archive.open(f"rankable_2023_2025/{symbol}.csv") as source:
                header = source.readline().decode("utf-8").strip().split(",")
                if header != ["timestamp", "tradeable", "absolute_rate", "relative_rate"]:
                    raise LazyProductionInputError("funding partition schema differs")
                for raw in source:
                    fields = raw.decode("utf-8").rstrip("\n").split(",")
                    if len(fields) != 4 or fields[1] != symbol:
                        raise LazyProductionInputError("funding row identity differs")
                    timestamp = _utc(fields[0].replace(" ", "T") + "Z")
                    if timestamp > cutoff:
                        break
                    if timestamp >= PROTECTED_START:
                        raise LazyProductionInputError("protected funding row opened")
                    rows.append((timestamp, fields[2], float(fields[3])))
        frozen = tuple(rows)
        self._funding_cache[key] = frozen
        while len(self._funding_cache) > 4:
            self._funding_cache.popitem(last=False)
        return frozen

    def _execution_schedule_bar(self, symbol: str, decision: datetime, completed_bars: Sequence[SignalBar]) -> SignalBar:
        """Bind only the next open timestamp; never decode its future OHLC."""
        import pyarrow.parquet as pq

        if not completed_bars or completed_bars[-1].close_ts != decision:
            raise LazyProductionInputError("decision has no exact last completed trade close")
        decision_ms = int(decision.timestamp() * 1000)
        found = 0
        for part in self._verified_trade_parts(symbol, decision, decision + timedelta(minutes=5)):
            path = Path(part.path)
            schema = set(pq.ParquetFile(path).schema_arrow.names)
            if "time" not in schema:
                continue
            table = pq.read_table(path, columns=["time"], filters=[("time", "==", decision_ms)])
            found += len(table)
        if found < 1:
            raise LazyProductionInputError("verified execution-open timestamp is absent")
        last_close = float(completed_bars[-1].close)
        return SignalBar(
            decision,
            decision + timedelta(minutes=5),
            last_close,
            last_close,
            last_close,
            last_close,
            decision,
            decision,
            True,
            True,
            None,
        )

    def _context_inputs(self, locator: FamilyDecisionLocator, partition: Mapping[str, Any], snapshot: Mapping[str, Any]) -> ContextInputs:
        daily = {symbol: self._daily(symbol, locator.decision_ts) for symbol in self._artifact_by_symbol}
        pit_rows_by_day = {day: rows for day, rows in self._pit_by_day.items()}
        exact_funding = self._funding(locator.symbol, locator.decision_ts)
        return _context(
            locator.decision_ts,
            daily,
            locator.symbol,
            snapshot,
            partition["training_start"],
            partition["training_end_exclusive"],
            tuple((timestamp, relative * 10000.0) for timestamp, _, relative in exact_funding),
            pit_rows_by_day,
        )

    def _table_populations(self, family_id: str, symbol: str, decile: int, partition: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        start = partition["training_start"]; end = partition["training_end_exclusive"]
        populations: dict[str, Any] = {}; unavailable: list[dict[str, Any]] = []
        names: list[str] = []
        if family_id == "A1_COMPRESSION_V2":
            names.extend(a1_compression.impulse_population_key(window, scope, side) for window in ("6h", "12h", "1d", "3d", "7d") for side in (-1, 1) for scope in ("symbol_side", "liquidity_decile_side", "global_side"))
            names.extend(a1_compression.contraction_population_key(base, baseline, scope) for base in ("2h", "6h", "12h", "1d", "3d") for scope in ("symbol", "liquidity_decile", "global_PIT") for baseline in ("adjacent_equal_duration", "trailing_5x_base_duration"))
            names.extend(a1_compression.smoothness_population_key(base, scope) for base in ("2h", "6h", "12h", "1d", "3d") for scope in ("symbol", "liquidity_decile", "global_PIT"))
            authority: Any = self._a1
        elif family_id == "A3_STARTER_RETEST_V3":
            names.extend(a3_starter_retest.breakout_population_key(lookback, atr, scope, side) for lookback in (5, 10, 20, 60) for atr in (10, 20, 40, 60) for side in (-1, 1) for scope in ("symbol_side", "liquidity_decile_side", "global_side"))
            authority = self._a3
        else:
            return populations, unavailable
        for name in names:
            try:
                populations[name] = authority.population(name, target_symbol=symbol, target_decile=decile, training_start=start, training_end=end)
            except PopulationTableError as exc:
                if "feature is absent" in str(exc):
                    raise LazyProductionInputError(
                        f"registered {family_id} population signature is absent from the bound compiler output: {name}"
                    ) from exc
                unavailable.append({"feature_signature": name, "status": "unavailable_data", "reason": str(exc)})
        return populations, unavailable

    def _populations(self, locator: FamilyDecisionLocator, partition: Mapping[str, Any], decile: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        key = (locator.family_id, locator.symbol, locator.partition_key, partition["training_start"], partition["training_end_exclusive"])
        cached = self._threshold_cache.get(key)
        if cached is not None:
            return cached
        table, unavailable = self._table_populations(locator.family_id, locator.symbol, decile, partition)
        if locator.family_id in {"A2_PRIOR_HIGH_RS_CONTEXT_V1", "A4_TSMOM_V7"}:
            target_bars = self._bars(locator.symbol, locator.decision_ts)
            bars_by_symbol = {locator.symbol: target_bars, "PF_XBTUSD": self._bars("PF_XBTUSD", locator.decision_ts), "PF_ETHUSD": self._bars("PF_ETHUSD", locator.decision_ts)}
            daily = {symbol: self._daily(symbol, locator.decision_ts) for symbol in bars_by_symbol}
            proximity = _a2_proximity_feature_arrays(target_bars, daily[locator.symbol]) if locator.family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None
            calculated, missing = _thresholds(
                bars_by_symbol,
                daily,
                target=locator.symbol,
                training_start=partition["training_start"],
                training_end=partition["training_end_exclusive"],
                skip_a1=True,
                skip_a3=True,
                a2_proximity_arrays=proximity,
            )
            prefix = "A2_" if locator.family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else "A4_"
            table.update({name: value for name, value in calculated.items() if name.startswith(prefix)})
            unavailable.extend(row for row in missing if str(row.get("feature_signature", "")).startswith(prefix))
        result = table, unavailable
        self._threshold_cache[key] = result
        while len(self._threshold_cache) > 1:
            self._threshold_cache.popitem(last=False)
        return result

    def frame(self, locator: FamilyDecisionLocator) -> FamilyInput:
        partition = self._partition(locator)
        day_ms = int(locator.decision_ts.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        day_rows = self._pit_by_day[day_ms]
        snapshot = _snapshot(day_rows, locator.decision_ts)
        target_decile = int(snapshot["lagged_liquidity_deciles"][locator.symbol])
        populations, unavailable = self._populations(locator, partition, target_decile)
        completed = tuple(row for row in self._bars(locator.symbol, locator.decision_ts) if locator.decision_ts - timedelta(days=181) <= row.open_ts and row.close_ts <= locator.decision_ts)
        schedule_bar = self._execution_schedule_bar(locator.symbol, locator.decision_ts, completed)
        bars = (*completed, schedule_bar)
        daily = tuple(row for row in self._daily(locator.symbol, locator.decision_ts) if row.close_ts < locator.decision_ts)
        funding = tuple(
            FundingInput(timestamp, timestamp, absolute)
            for timestamp, absolute, _ in self._funding(locator.symbol, locator.decision_ts)
            if partition["training_start"] <= timestamp <= locator.decision_ts
        )
        context = self._context_inputs(locator, partition, snapshot)
        metadata = {
            "production_input": True,
            "lazy_full_population_adapter": "stage24_lazy_full_population_family_input_v1",
            "construction_mode": self.construction_mode,
            "requested_family_id": locator.family_id,
            "evaluation_start": partition["evaluation_start"],
            "evaluation_end_exclusive": partition["evaluation_end_exclusive"],
            "eligible_days": int((partition["evaluation_end_exclusive"] - partition["evaluation_start"]).days),
            "eligible_symbol_seconds": float((partition["evaluation_end_exclusive"] - partition["evaluation_start"]).total_seconds()),
            "base_gap_allowance_bps_per_hour": 0.25,
            "stress_gap_allowance_bps_per_hour": 0.50,
            "pit_universe_snapshot": snapshot,
            "campaign_partition": partition,
            "a1_persistent_state": initial_state().payload(),
            "a1_state_origin": "history_rebuild_at_complete_frame_start",
            "execution_schedule_identity": canonical_hash({"symbol": locator.symbol, "entry_open_ts": locator.decision_ts.isoformat(), "source": "verified_virtual_trade_schedule"}),
            "shadow_ohlc_attestation": {
                "real_rows_max_close_ts": locator.decision_ts,
                "execution_open_timestamp_only": True,
                "execution_open_price_source": "last_completed_trade_close_placeholder",
                "post_entry_real_ohlc_rows": 0,
            },
            "decision_locator": locator.identity_payload(),
            "decision_locator_sha256": canonical_hash(locator.identity_payload()),
            "partition_identity_sha256": canonical_hash({
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in partition.items()
            }),
            "source_part_inventory_sha256": canonical_hash([part.payload() for part in self._parts(locator.symbol)]),
            "feature_signature_unavailable": tuple(unavailable),
            "virtual_cache_manifest_sha256": sha256_file(self.virtual_cache_manifest_path),
            "a1_population_manifest_sha256": sha256_file(self.a1_population_manifest_path),
            "a3_population_manifest_sha256": sha256_file(self.a3_population_manifest_path),
            "protected_rows": 0,
            "economic_outcomes_opened": False,
            "capitalcom_payload_opened": False,
        }
        frame = FamilyInput(
            KRAKEN_PLATFORM,
            locator.symbol,
            str(locator.inner_fold_id or locator.outer_fold_id),
            locator.decision_ts,
            bars,
            daily,
            funding,
            populations,
            context,
            metadata,
        )
        frame = with_source_authority(frame, self.authority)
        frame.validate()
        return frame


__all__ = [
    "FamilyDecisionLocator",
    "LazyProductionFamilyInputAdapter",
    "LazyProductionInputError",
    "SUPPORTED_FAMILIES",
]
