from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .canonical import canonical_hash, sha256_file
from .engine_types import ContextInputs, FamilyInput, FundingInput, KRAKEN_PLATFORM
from .family_engines.kda02b_adjudication import cell_contract_sha256
from .kda02b_population_index import (
    KDA02BPopulationIndexError,
    PopulationExpectations,
    PRODUCTION_EXPECTATIONS,
    validate_kda02b_lazy_population_index,
)
from .kda02b_production import (
    FEATURE_CONTRACT_SHA256,
    KDA_REQUIRED_FIELDS_AND_UNITS,
    _feature_history,
    _guarded_feature_frame,
    _schedule_bars,
)
from .production_cache import ALLOWED_CANDLE_COLUMNS, EMPTY_CANDLE_COLUMNS, ProductionCacheCompiler, SourcePart
from .production_inputs import _funding_rows, _load_trade_bars
from .synthetic import with_source_authority


UTC = timezone.utc
SHADOW_MODE = "shadow_no_outcome"
ECONOMIC_MODE = "economic_authorized"


class KDA02BLazyFamilyInputError(RuntimeError):
    pass


@dataclass(frozen=True)
class KDA02BLazyFamilyInputRecord:
    status: str
    event_id: str
    cell_id: str
    model_id: str
    outer_fold_id: str
    symbol: str
    decision_ts: datetime
    event_locator_sha256: str
    unavailable_reason: str | None
    frame: FamilyInput | None


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise KDA02BLazyFamilyInputError("KDA02B lazy input timestamp is timezone-naive")
    return result.astimezone(UTC)


def _authority_context(authority_path: Path, repository_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    required_bindings = (
        "source_manifest_sha256", "pit_universe_sha256", "funding_manifest_sha256",
        "cache_contract_sha256", "fold_graph_sha256",
    )
    if any(
        not isinstance(authority.get(key), str)
        or len(authority[key]) != 64
        or any(character not in "0123456789abcdef" for character in authority[key])
        for key in required_bindings
    ):
        raise KDA02BLazyFamilyInputError("execution-input authority lacks a valid cache binding")
    roles: dict[str, Path] = {}
    records = [*authority.get("source_records", ()), *authority.get("kda02b_authority_records", ())]
    for record in records:
        role = str(record.get("role", ""))
        raw_path = Path(str(record.get("path", "")))
        path = raw_path if raw_path.is_absolute() else repository_root / raw_path
        if role in roles:
            raise KDA02BLazyFamilyInputError(f"duplicate execution-input role: {role}")
        if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise KDA02BLazyFamilyInputError(f"execution-input authority drift: {role}")
        roles[role] = path
    required_roles = {
        "kraken_acquisition_manifest", "rankable_funding_package",
        "stage8a_feature_cache_manifest", "stage8a_shared_feature_schema",
        "stage20_kda02b_fold_local_thresholds",
    }
    if not required_roles <= set(roles):
        raise KDA02BLazyFamilyInputError(f"KDA02B lazy FamilyInput roles are absent: {sorted(required_roles - set(roles))}")
    return authority, roles


def _source_parts(
    authority_path: Path,
    repository_root: Path,
    roles: Mapping[str, Path],
    symbols: set[str],
) -> dict[str, list[SourcePart]]:
    compiler = ProductionCacheCompiler(authority_path, Path("."), repository_root)
    try:
        return compiler._source_parts(roles["kraken_acquisition_manifest"], symbols)
    except Exception as exc:
        raise KDA02BLazyFamilyInputError("KDA02B price-source authority cannot be compiled") from exc


def _part_overlaps(part: SourcePart, start: datetime, end: datetime) -> bool:
    part_start = _utc(part.chunk_start) if part.chunk_start else datetime(2023, 1, 1, tzinfo=UTC)
    part_end = _utc(part.chunk_end) if part.chunk_end else datetime(2026, 1, 1, tzinfo=UTC)
    return part_start < end and part_end > start


def _verify_symbol_parts(parts: Sequence[SourcePart]) -> None:
    import pyarrow.parquet as pq

    if {part.dataset for part in parts} != {"historical_trade_candles_5m", "historical_mark_candles_5m"}:
        raise KDA02BLazyFamilyInputError("KDA02B trade/mark source pair is incomplete")
    for part in parts:
        path = Path(part.path)
        if not path.is_file() or path.stat().st_size <= 0 or sha256_file(path) != part.sha256:
            raise KDA02BLazyFamilyInputError(f"KDA02B candle source hash differs: {path}")
        parquet = pq.ParquetFile(path)
        columns = frozenset(parquet.schema_arrow.names)
        if columns not in {ALLOWED_CANDLE_COLUMNS, EMPTY_CANDLE_COLUMNS} or parquet.metadata.num_rows != part.rows:
            raise KDA02BLazyFamilyInputError(f"KDA02B candle source schema/count differs: {path}")


def _economic_bars(parts: Sequence[SourcePart], decision: datetime, horizon: str) -> tuple[Any, ...]:
    hours = int(str(horizon).removesuffix("h"))
    end = decision + timedelta(hours=hours + 1, minutes=15)
    selected = tuple(part for part in parts if _part_overlaps(part, decision, end))
    bars = _load_trade_bars(selected, decision, end)
    if not bars or bars[0].open_ts != decision:
        raise KDA02BLazyFamilyInputError("authorized KDA02B economic execution bars do not start at the exact decision")
    if bars[-1].open_ts < decision + timedelta(hours=hours + 1):
        raise KDA02BLazyFamilyInputError("authorized KDA02B economic execution bars do not cover delay plus horizon")
    return bars


class KDA02BLazyFamilyInputAdapter:
    """Stream exact KDA02B index rows into bounded, authority-backed FamilyInput objects."""

    def __init__(
        self,
        *,
        index_root: Path,
        authority_path: Path,
        repository_root: Path,
        mode: str = SHADOW_MODE,
        economic_authorization_sha256: str | None = None,
        expectations: PopulationExpectations = PRODUCTION_EXPECTATIONS,
    ) -> None:
        if mode not in {SHADOW_MODE, ECONOMIC_MODE}:
            raise KDA02BLazyFamilyInputError(f"unsupported KDA02B FamilyInput mode: {mode}")
        if mode == ECONOMIC_MODE and (
            not isinstance(economic_authorization_sha256, str)
            or len(economic_authorization_sha256) != 64
            or any(character not in "0123456789abcdef" for character in economic_authorization_sha256)
        ):
            raise KDA02BLazyFamilyInputError("economic KDA02B input access lacks an explicit hash-bound authorization")
        try:
            self.index_manifest = validate_kda02b_lazy_population_index(index_root, expectations)
        except KDA02BPopulationIndexError as exc:
            raise KDA02BLazyFamilyInputError("KDA02B lazy population authority failed validation") from exc
        self.index_root = index_root
        self.authority_path = authority_path
        self.repository_root = repository_root
        self.mode = mode
        self.economic_authorization_sha256 = economic_authorization_sha256
        self.expectations = expectations
        self.authority, self.roles = _authority_context(authority_path, repository_root)
        recorded_authority = self.index_manifest["source_records"]["execution_input_authority"]
        if recorded_authority.get("sha256") != sha256_file(authority_path):
            raise KDA02BLazyFamilyInputError("KDA02B index and execution-input authority differ")
        feature_manifest = json.loads(self.roles["stage8a_feature_cache_manifest"].read_text(encoding="utf-8"))
        partitions = feature_manifest.get("partitions")
        if not isinstance(partitions, list):
            raise KDA02BLazyFamilyInputError("Stage8A feature partition manifest is invalid")
        self.feature_partitions = {str(row.get("symbol")): row for row in partitions}
        thresholds = json.loads(self.roles["stage20_kda02b_fold_local_thresholds"].read_text(encoding="utf-8"))
        if not isinstance(thresholds.get("models"), Mapping):
            raise KDA02BLazyFamilyInputError("KDA02B fold-local threshold authority is invalid")
        self.models = thresholds["models"]
        self.source_parts = _source_parts(authority_path, repository_root, self.roles, set(self.feature_partitions))
        self._verified_part_symbols: set[str] = set()
        self.last_reconciliation: dict[str, Any] | None = None

    def _index_rows(self) -> Iterator[dict[str, Any]]:
        import pyarrow.parquet as pq

        record = next(row for row in self.index_manifest["files"] if row["role"] == "event_index")
        path = self.index_root / str(record["path"])
        current_symbol: str | None = None
        buffered: list[dict[str, Any]] = []
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                symbol = str(row["symbol"])
                if current_symbol is not None and symbol != current_symbol:
                    for item in sorted(buffered, key=lambda value: (
                        str(value["outer_fold_id"]), _utc(value["decision_ts"]),
                        str(value["cell_id"]), str(value["event_id"]),
                    )):
                        yield item
                    buffered = []
                current_symbol = symbol
                buffered.append(row)
        for item in sorted(buffered, key=lambda value: (
            str(value["outer_fold_id"]), _utc(value["decision_ts"]),
            str(value["cell_id"]), str(value["event_id"]),
        )):
            yield item

    def _record(self, row: Mapping[str, Any], frame: FamilyInput | None) -> KDA02BLazyFamilyInputRecord:
        status = str(row["status"])
        return KDA02BLazyFamilyInputRecord(
            status=status,
            event_id=str(row["event_id"]), cell_id=str(row["cell_id"]),
            model_id=str(row["model_id"]), outer_fold_id=str(row["outer_fold_id"]),
            symbol=str(row["symbol"]), decision_ts=_utc(row["decision_ts"]),
            event_locator_sha256=str(row["event_locator_sha256"]),
            unavailable_reason=str(row["unavailable_reason"]) if status == "typed_unavailable" else None,
            frame=frame,
        )

    def stream(
        self,
        *,
        cell_id: str | None = None,
        outer_fold_id: str | None = None,
    ) -> Iterator[KDA02BLazyFamilyInputRecord]:
        """Yield the complete index or one exact registered cell/fold slice."""
        if cell_id is not None and not cell_id:
            raise KDA02BLazyFamilyInputError("KDA02B cell filter is empty")
        if outer_fold_id is not None and not outer_fold_id:
            raise KDA02BLazyFamilyInputError("KDA02B fold filter is empty")
        current_symbol: str | None = None
        feature_frame: Any = None
        economic_funding: tuple[tuple[datetime, str, float], ...] = ()
        shared_history_key: tuple[str, datetime] | None = None
        shared_bars_key: tuple[datetime, str] | None = None
        shared_history: tuple[Mapping[str, Any], ...] = ()
        shared_bars: tuple[Any, ...] = ()
        counts = Counter()
        prior_order: tuple[str, str, datetime, str, str] | None = None
        for row in self._index_rows():
            symbol = str(row["symbol"]); fold = str(row["outer_fold_id"]); decision = _utc(row["decision_ts"])
            if cell_id is not None and str(row["cell_id"]) != cell_id:
                continue
            if outer_fold_id is not None and fold != outer_fold_id:
                continue
            order = (symbol, fold, decision, str(row["cell_id"]), str(row["event_id"]))
            if prior_order is not None and order <= prior_order:
                raise KDA02BLazyFamilyInputError("KDA02B lazy input order is not strict symbol/fold/decision/cell/event")
            prior_order = order
            status = str(row["status"]); counts[status] += 1
            if status == "typed_unavailable":
                yield self._record(row, None)
                continue
            if status != "eligible":
                raise KDA02BLazyFamilyInputError(f"unknown KDA02B lazy input status: {status}")
            if symbol != current_symbol:
                current_symbol = symbol
                partition = self.feature_partitions.get(symbol)
                if partition is None or partition.get("sha256") != row.get("feature_partition_sha256"):
                    raise KDA02BLazyFamilyInputError("KDA02B event-to-Stage8A feature partition binding differs")
                feature_frame = _guarded_feature_frame(partition)
                if symbol not in self._verified_part_symbols:
                    _verify_symbol_parts(self.source_parts[symbol])
                    self._verified_part_symbols.add(symbol)
                economic_funding = (
                    _funding_rows(self.roles["rankable_funding_package"], symbol)
                    if self.mode == ECONOMIC_MODE else ()
                )
                shared_history_key = None
                shared_bars_key = None
            feature_ordinal = int(row["feature_row_ordinal"])
            if not 0 <= feature_ordinal < len(feature_frame):
                raise KDA02BLazyFamilyInputError("KDA02B feature-row ordinal is outside its partition")
            physical_timestamp = _utc(feature_frame.iloc[feature_ordinal]["timestamp_utc"])
            if physical_timestamp != _utc(row["feature_timestamp_utc"]) or physical_timestamp + timedelta(minutes=5) != decision:
                raise KDA02BLazyFamilyInputError("KDA02B feature-row locator does not bind the exact decision")
            model_id = str(row["model_id"])
            model = self.models.get(model_id)
            if not isinstance(model, Mapping):
                raise KDA02BLazyFamilyInputError(f"KDA02B fold-local model is absent: {model_id}")
            evaluation_start = _utc(model["evaluation_start"]); evaluation_end = _utc(model["evaluation_end"])
            history_key = (model_id, decision)
            if history_key != shared_history_key:
                shared_history = _feature_history(feature_frame, evaluation_start, decision)
                final = shared_history[-1]
                if final.get("eligible") is not True:
                    raise KDA02BLazyFamilyInputError("eligible KDA02B index row resolved to a causally ineligible feature")
                shared_history_key = history_key
                shared_bars_key = None
            bars_key = (decision, "shadow" if self.mode == SHADOW_MODE else str(row["horizon"]))
            if bars_key != shared_bars_key:
                final = shared_history[-1]
                end = decision + timedelta(hours=7, minutes=15)
                overlapping = tuple(part for part in self.source_parts[symbol] if _part_overlaps(part, decision, end))
                if self.mode == SHADOW_MODE:
                    shared_bars = _schedule_bars(overlapping, decision, float(final["trade_close"]))
                else:
                    shared_bars = _economic_bars(overlapping, decision, str(row["horizon"]))
                shared_bars_key = bars_key
            partition = {
                "phase": "kda02b_adjudication", "outer_fold_id": fold, "inner_fold_id": None,
                "training_start": _utc(model["training_start"]),
                "training_end_exclusive": _utc(model["training_end"]),
                "evaluation_start": evaluation_start, "evaluation_end_exclusive": evaluation_end,
            }
            funding_end = decision + timedelta(hours=int(str(row["horizon"]).removesuffix("h")) + 1, minutes=15)
            funding = tuple(
                FundingInput(timestamp, timestamp, absolute)
                for timestamp, absolute, _relative in economic_funding
                if decision <= timestamp <= funding_end
            )
            cell = str(row["cell_id"])
            metadata = {
                "production_input": True,
                "kda02b_lazy_population_input": True,
                "input_mode": self.mode,
                "shadow_no_outcome_execution_schedule_only": self.mode == SHADOW_MODE,
                "real_post_entry_price_rows_opened": 0 if self.mode == SHADOW_MODE else len(shared_bars),
                "economic_outcomes_opened": self.mode == ECONOMIC_MODE,
                "economic_authorization_sha256": self.economic_authorization_sha256,
                "protected_rows": 0,
                "campaign_partition": partition,
                "evaluation_start": evaluation_start,
                "evaluation_end_exclusive": evaluation_end,
                "eligible_days": int((evaluation_end - evaluation_start).days),
                "eligible_symbol_seconds": float((evaluation_end - evaluation_start).total_seconds() * self.expectations.eligible_symbols),
                "base_gap_allowance_bps_per_hour": 0.25,
                "stress_gap_allowance_bps_per_hour": 0.50,
                "stage20_cell_id": cell,
                "stage20_cell_contract_sha256": cell_contract_sha256(cell),
                "stage20_model_id": model_id,
                "stage20_event_id": str(row["event_id"]),
                "stage20_translation_id": str(row["translation_id"]),
                "stage20_tape_side": str(row["side"]),
                "stage20_tape_horizon": str(row["horizon"]),
                "kda02b_feature_history": shared_history,
                "fold_thresholds": {
                    key: float(value) for key, value in model.get("thresholds", {}).items()
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                },
                "kda02b_feature_partition_sha256": str(row["feature_partition_sha256"]),
                "kda02b_feature_schema_sha256": sha256_file(self.roles["stage8a_shared_feature_schema"]),
                "kda02b_feature_contract_sha256": FEATURE_CONTRACT_SHA256,
                "kda02b_required_fields_and_units": KDA_REQUIRED_FIELDS_AND_UNITS,
                "kda02b_authority_inventory_sha256": self.authority.get("kda02b_authority_inventory_sha256"),
                "kda02b_event_locator_sha256": str(row["event_locator_sha256"]),
                "kda02b_retention_boundary_identity_sha256": str(row["retention_boundary_identity_sha256"]),
                "execution_schedule_identity": canonical_hash({
                    "symbol": symbol, "open_ts": [bar.open_ts.isoformat() for bar in shared_bars],
                    "source": "verified_trade_timestamp_column_only" if self.mode == SHADOW_MODE else "authorized_trade_ohlc",
                }),
            }
            frame = FamilyInput(
                KRAKEN_PLATFORM, symbol, model_id, decision, shared_bars, (), funding,
                {}, ContextInputs(), metadata,
            )
            frame = with_source_authority(frame, self.authority)
            frame.validate()
            yield self._record(row, frame)
        expected = self.index_manifest["counts"]
        if cell_id is None and outer_fold_id is None and (
            counts["eligible"] != expected["eligible_event_rows"]
            or counts["typed_unavailable"] != expected["unavailable_event_rows"]
        ):
            raise KDA02BLazyFamilyInputError("KDA02B lazy FamilyInput terminal count differs")
        self.last_reconciliation = {
            "eligible_frames": counts["eligible"],
            "typed_unavailable_rows_without_frames": counts["typed_unavailable"],
            "cell_id": cell_id,
            "outer_fold_id": outer_fold_id,
            "mode": self.mode,
            "economic_outcomes_opened": self.mode == ECONOMIC_MODE,
            "status": "pass",
        }


__all__ = [
    "ECONOMIC_MODE", "KDA02BLazyFamilyInputAdapter", "KDA02BLazyFamilyInputError",
    "KDA02BLazyFamilyInputRecord", "SHADOW_MODE",
]
