from __future__ import annotations

import json
import gzip
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_bytes, atomic_write_json, canonical_hash, canonical_json_bytes, sha256_file
from .engine_types import ContextInputs, DailyBar, ExactPopulationTableView, ExactPopulationView, FamilyInput, FundingInput, SignalBar, ThresholdPopulation, _mark_validated_frame
from .family_engines.common import EngineInputError, require_utc


class CacheBuildError(RuntimeError):
    pass


def _read_payload(path: Path, encoding: str) -> Mapping[str, Any]:
    if encoding == "canonical_json":
        raw = path.read_bytes()
    elif encoding == "canonical_json_gzip_mtime0":
        try:
            raw = gzip.decompress(path.read_bytes())
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            raise CacheBuildError("compressed semantic-cache frame is invalid") from exc
    else:
        raise CacheBuildError(f"unsupported semantic-cache frame encoding: {encoding}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheBuildError("semantic-cache frame JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise CacheBuildError("semantic-cache frame is not a JSON object")
    return payload


def _read_reference_payload(
    path: Path,
    *,
    cache_root: Path,
    components_by_path: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    reference = _read_payload(path, "canonical_json")
    if set(reference) != {"schema", "shared_payload_path", "shared_payload_sha256", "campaign_partition"} or reference.get("schema") != "family_input_reference_v1":
        raise CacheBuildError("semantic-cache frame reference schema differs")
    relative = Path(str(reference["shared_payload_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise CacheBuildError("semantic-cache shared payload path is unsafe")
    record = components_by_path.get(relative.as_posix())
    if record is None or record.get("sha256") != reference.get("shared_payload_sha256"):
        raise CacheBuildError("semantic-cache frame reference is not bound to a registered component")
    component_path = cache_root / relative
    if not component_path.is_file() or component_path.stat().st_size != record.get("bytes") or sha256_file(component_path) != record.get("sha256"):
        raise CacheBuildError("semantic-cache shared payload bytes differ")
    payload = dict(_read_payload(component_path, str(record.get("encoding"))))
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("campaign_partition") is not None:
        raise CacheBuildError("semantic-cache shared payload contains an unbound partition")
    payload["metadata"] = {**metadata, "campaign_partition": reference["campaign_partition"]}
    return payload


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise CacheBuildError("cache timestamp is not an ISO-8601 string")
    try:
        return require_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise CacheBuildError("cache timestamp is invalid") from exc


def _restore_metadata(value: Any, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {str(item_key): _restore_metadata(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return tuple(_restore_metadata(item) for item in value)
    if isinstance(value, str) and (key or "").endswith(("_ts", "_start", "_end", "_exclusive", "_until")):
        try:
            return _datetime(value)
        except CacheBuildError:
            return value
    return value


def _decode_population_values(value: object, cache_root: Path | None, *, external_verified: bool) -> Sequence[float]:
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    if isinstance(value, Mapping) and value.get("schema") == "exact_float64_population_table_view_v1":
        expected = {
            "schema", "values_path", "values_sha256", "timestamps_path", "timestamps_sha256",
            "symbols_path", "symbols_sha256", "deciles_path", "deciles_sha256", "dtypes",
            "physical_count", "training_start_ms", "training_end_ms", "selected_count", "unique_count", "minimum_unique_count_verified", "value_multiplier",
            "symbol_code", "liquidity_decile",
        }
        if set(value) != expected or value.get("dtypes") != {"values": "float64_le", "timestamps": "int64_le", "symbols": "uint16_le", "deciles": "uint8"} or cache_root is None:
            raise CacheBuildError("external threshold table identity is invalid")
        return ExactPopulationTableView(
            values_path=str(value["values_path"]), values_sha256=str(value["values_sha256"]),
            timestamps_path=str(value["timestamps_path"]), timestamps_sha256=str(value["timestamps_sha256"]),
            symbols_path=str(value["symbols_path"]), symbols_sha256=str(value["symbols_sha256"]),
            deciles_path=str(value["deciles_path"]), deciles_sha256=str(value["deciles_sha256"]),
            physical_count=int(value["physical_count"]), training_start_ms=int(value["training_start_ms"]),
            training_end_ms=int(value["training_end_ms"]), selected_count=int(value["selected_count"]),
            unique_count=None if value["unique_count"] is None else int(value["unique_count"]),
            minimum_unique_count_verified=int(value["minimum_unique_count_verified"]),
            value_multiplier=int(value["value_multiplier"]),
            symbol_code=None if value["symbol_code"] is None else int(value["symbol_code"]),
            liquidity_decile=None if value["liquidity_decile"] is None else int(value["liquidity_decile"]),
            root=str(cache_root), physical_verified=external_verified,
        )
    if not isinstance(value, Mapping) or value.get("schema") != "exact_sorted_float64_population_view_v1":
        raise CacheBuildError("threshold population values have an unsupported schema")
    expected = {
        "schema", "relative_path", "sha256", "dtype", "physical_count",
        "start_index", "end_index", "unique_count",
    }
    if set(value) != expected or value.get("dtype") != "float64_le" or cache_root is None:
        raise CacheBuildError("external threshold population identity is invalid")
    return ExactPopulationView(
        relative_path=str(value["relative_path"]), sha256=str(value["sha256"]),
        physical_count=int(value["physical_count"]), start_index=int(value["start_index"]),
        end_index=int(value["end_index"]), unique_count=int(value["unique_count"]),
        root=str(cache_root), physical_verified=external_verified,
    )


def decode_family_input(payload: Mapping[str, Any], *, cache_root: Path | None = None, external_components_verified: bool = False) -> FamilyInput:
    expected = {"platform", "symbol", "fold_id", "decision_ts", "five_minute_bars", "daily_bars", "funding", "threshold_populations", "context", "metadata"}
    if set(payload) != expected:
        raise CacheBuildError("semantic-cache frame has unknown or missing top-level fields")
    bars = tuple(
        SignalBar(
            _datetime(row["open_ts"]), _datetime(row["close_ts"]), float(row["open"]), float(row["high"]),
            float(row["low"]), float(row["close"]), _datetime(row["source_close_ts"]),
            _datetime(row["feature_available_ts"]), bool(row["lifecycle_valid"]), bool(row["pit_eligible"]),
            None if row.get("quote_notional") is None else float(row["quote_notional"]),
        )
        for row in payload["five_minute_bars"]
    )
    def daily(rows: Sequence[Mapping[str, Any]]) -> tuple[DailyBar, ...]:
        return tuple(DailyBar(_datetime(row["close_ts"]), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), _datetime(row["source_close_ts"]), _datetime(row["feature_available_ts"]), bool(row["valid_day"])) for row in rows)
    funding = tuple(FundingInput(_datetime(row["row_timestamp"]), _datetime(row["publication_ts"]), str(row["absolute_rate_usd_per_contract_unit"]), str(row["source_partition"])) for row in payload["funding"])
    populations = {
        str(name): ThresholdPopulation(
            _decode_population_values(row["values"], cache_root, external_verified=external_components_verified), tuple(str(value) for value in row["unique_symbols"]), str(row["scope"]),
            _datetime(row["training_start"]), _datetime(row["training_end_exclusive"]), _datetime(row["source_close_ts"]),
            _datetime(row["feature_available_ts"]), str(row["source_sha256"]),
        )
        for name, row in payload["threshold_populations"].items()
    }
    raw_context = payload["context"]
    context = ContextInputs(
        btc_daily=daily(raw_context["btc_daily"]),
        eth_daily=daily(raw_context["eth_daily"]),
        symbol_daily=daily(raw_context["symbol_daily"]),
        breadth_history=tuple(float(value) for value in raw_context["breadth_history"]),
        dispersion_history=tuple(float(value) for value in raw_context["dispersion_history"]),
        breadth_history_by_lookback={int(key): tuple(float(value) for value in values) for key, values in raw_context["breadth_history_by_lookback"].items()},
        dispersion_history_by_lookback={int(key): tuple(float(value) for value in values) for key, values in raw_context["dispersion_history_by_lookback"].items()},
        cross_section_returns={str(key): float(value) for key, value in raw_context["cross_section_returns"].items()},
        cross_section_returns_by_lookback={int(key): {str(symbol): float(value) for symbol, value in values.items()} for key, values in raw_context["cross_section_returns_by_lookback"].items()},
        cross_section_liquidity_deciles={str(key): int(value) for key, value in raw_context["cross_section_liquidity_deciles"].items()},
        parent_universe=tuple(str(value) for value in raw_context["parent_universe"]),
        funding_burden_history=tuple(float(value) for value in raw_context["funding_burden_history"]),
        funding_burden_current=None if raw_context["funding_burden_current"] is None else float(raw_context["funding_burden_current"]),
        as_of_ts=None if raw_context["as_of_ts"] is None else _datetime(raw_context["as_of_ts"]),
        source_close_ts=None if raw_context["source_close_ts"] is None else _datetime(raw_context["source_close_ts"]),
        feature_available_ts=None if raw_context["feature_available_ts"] is None else _datetime(raw_context["feature_available_ts"]),
        source_sha256=raw_context["source_sha256"],
    )
    frame = FamilyInput(
        str(payload["platform"]), str(payload["symbol"]), str(payload["fold_id"]), _datetime(payload["decision_ts"]),
        bars, daily(payload["daily_bars"]), funding, populations, context, _restore_metadata(payload["metadata"]),
    )
    frame.validate()
    return frame


def _authority_bindings(authority: Mapping[str, Any]) -> dict[str, str]:
    keys = ("source_manifest_sha256", "pit_universe_sha256", "funding_manifest_sha256", "cache_contract_sha256", "fold_graph_sha256")
    result = {key: str(authority[key]) for key in keys}
    if any(len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in result.values()):
        raise CacheBuildError("execution-input authority contains an invalid binding")
    return result


def _campaign_partition(frame: FamilyInput) -> dict[str, Any]:
    """Return and validate the exact fold partition attached by the source compiler."""
    raw = frame.metadata.get("campaign_partition")
    if not isinstance(raw, Mapping):
        raise CacheBuildError("source-compiled frame lacks a campaign partition")
    required = {
        "phase", "outer_fold_id", "inner_fold_id", "training_start",
        "training_end_exclusive", "evaluation_start", "evaluation_end_exclusive",
    }
    if set(raw) != required:
        raise CacheBuildError("campaign partition has unknown or missing fields")
    partition = dict(raw)
    if partition["phase"] not in {"inner_validation", "outer_evaluation", "kda02b_adjudication", "synthetic_canary"}:
        raise CacheBuildError("campaign partition has an unsupported phase")
    start = require_utc(partition["training_start"])
    end = require_utc(partition["training_end_exclusive"])
    evaluation_start = require_utc(partition["evaluation_start"])
    evaluation_end = require_utc(partition["evaluation_end_exclusive"])
    if not (start < end <= evaluation_start < evaluation_end):
        raise CacheBuildError("campaign partition temporal ordering is invalid")
    if not (evaluation_start <= require_utc(frame.decision_ts) < evaluation_end):
        raise CacheBuildError("frame decision is outside its exact campaign partition")
    for population in frame.threshold_populations.values():
        if require_utc(population.training_start) != start or require_utc(population.training_end_exclusive) != end:
            raise CacheBuildError("threshold population does not bind the partition training interval")
    return partition


def build_semantic_cache(
    cache_root: Path,
    execution_input_authority: Mapping[str, Any],
    frames: Sequence[FamilyInput],
    *,
    authority_root: Path,
    synthetic_only: bool = False,
) -> Path:
    """Persist canonical frame payloads after physical source-authority verification."""
    writer = SemanticCacheWriter(
        cache_root,
        execution_input_authority,
        authority_root=authority_root,
        synthetic_only=synthetic_only,
    )
    for frame in frames:
        writer.add(frame)
    return writer.finalize()


class SemanticCacheWriter:
    """Streaming canonical cache writer; retains records, never full frames."""

    def __init__(
        self,
        cache_root: Path,
        execution_input_authority: Mapping[str, Any],
        *,
        authority_root: Path,
        synthetic_only: bool = False,
    ) -> None:
        self.cache_root = cache_root
        self.execution_input_authority = execution_input_authority
        self.synthetic_only = bool(synthetic_only)
        self.bindings = _authority_bindings(execution_input_authority)
        records_by_role = {}
        for record in execution_input_authority.get("source_records", ()):
            raw = Path(str(record["path"]))
            path = raw if raw.is_absolute() else authority_root / raw
            if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
                raise CacheBuildError(f"source authority mismatch: {record.get('role')}")
            records_by_role[str(record["role"])] = record
        if not records_by_role:
            raise CacheBuildError("cache construction has no verified source authorities")
        self.source_composite = canonical_hash(execution_input_authority["source_records"])
        self.artifacts: list[dict[str, Any]] = []
        self.typed_unavailable: list[dict[str, Any]] = []
        self.components: dict[str, dict[str, Any]] = {}
        self.paths: set[str] = set()

    def add_unavailable(
        self,
        *,
        family_id: str,
        partition: Mapping[str, Any],
        reason: str,
        authority_sha256: str,
    ) -> dict[str, Any]:
        if not reason or len(authority_sha256) != 64:
            raise CacheBuildError("typed cache unavailability lacks its reason or authority")
        required = {
            "phase", "outer_fold_id", "inner_fold_id", "training_start",
            "training_end_exclusive", "evaluation_start", "evaluation_end_exclusive",
        }
        if set(partition) != required:
            raise CacheBuildError("typed cache unavailability has an invalid partition")
        canonical_partition = _content_partition(partition)
        record = {
            "family_id": family_id,
            "status": "unavailable_data",
            "reason": reason,
            "authority_sha256": authority_sha256,
            "campaign_partition": canonical_partition,
        }
        identity = canonical_hash(record)
        if any(item["unavailable_identity_sha256"] == identity for item in self.typed_unavailable):
            return next(item for item in self.typed_unavailable if item["unavailable_identity_sha256"] == identity)
        stored = {**record, "unavailable_identity_sha256": identity}
        self.typed_unavailable.append(stored)
        return stored

    def add(self, frame: FamilyInput) -> dict[str, Any]:
        frame.validate()
        partition = _campaign_partition(frame)
        provenance = frame.metadata.get("source_authority")
        expected_provenance = {
            **self.bindings,
            "source_record_inventory_sha256": self.source_composite,
            "rankable_funding_package_sha256": self.execution_input_authority.get("rankable_funding_package_sha256"),
        }
        if provenance != expected_provenance:
            raise CacheBuildError("frame was not derived under the exact execution-input authority")
        for raw_component in frame.metadata.get("cache_authority_components", ()):
            if not isinstance(raw_component, Mapping) or set(raw_component) != {"path", "bytes", "sha256", "encoding"}:
                raise CacheBuildError("frame cache-authority component record is invalid")
            relative_path = Path(str(raw_component["path"]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise CacheBuildError("frame cache-authority component path is unsafe")
            physical = self.cache_root / relative_path
            if raw_component["encoding"] != "canonical_json" or not physical.is_file() or physical.stat().st_size != int(raw_component["bytes"]) or sha256_file(physical) != raw_component["sha256"]:
                raise CacheBuildError("frame cache-authority component bytes differ")
            component = {
                **dict(raw_component),
                "shared_content_sha256": canonical_hash({"path": relative_path.as_posix(), "sha256": raw_component["sha256"], "encoding": raw_component["encoding"]}),
            }
            previous = self.components.get(relative_path.as_posix())
            if previous is not None and previous != component:
                raise CacheBuildError("frame cache-authority component identity conflicts")
            self.components[relative_path.as_posix()] = component
        for population in frame.threshold_populations.values():
            values = population.values
            if not isinstance(values, (ExactPopulationView, ExactPopulationTableView)):
                continue
            if values.root is None or Path(values.root).resolve() != self.cache_root.resolve():
                raise CacheBuildError("external threshold component is outside this cache root")
            records = values.component_records() if isinstance(values, ExactPopulationTableView) else ((values.relative_path, values.sha256, "npy_float64_le_sorted_v1"),)
            for raw_relative, expected_hash, encoding in records:
                relative_path = Path(raw_relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise CacheBuildError("external threshold component path is unsafe")
                physical = self.cache_root / relative_path
                component = {
                    "path": relative_path.as_posix(), "bytes": physical.stat().st_size if physical.is_file() else -1,
                    "sha256": expected_hash, "encoding": encoding,
                    "shared_content_sha256": canonical_hash({"path": relative_path.as_posix(), "sha256": expected_hash, "encoding": encoding}),
                }
                previous = self.components.get(relative_path.as_posix())
                if previous is not None:
                    if previous != component:
                        raise CacheBuildError("external threshold component identity conflicts")
                    continue
                if not physical.is_file() or sha256_file(physical) != expected_hash:
                    raise CacheBuildError("external threshold component bytes differ")
                self.components[relative_path.as_posix()] = component
        payload = frame.content_payload()
        content_hash = canonical_hash(payload)
        relative = Path("frames") / f"{content_hash}.ref.json"
        if relative.as_posix() in self.paths:
            raise CacheBuildError("semantic cache contains a duplicate frame content identity")
        shared_payload = dict(payload)
        metadata = dict(shared_payload["metadata"])
        metadata["campaign_partition"] = None
        shared_payload["metadata"] = metadata
        shared_content_hash = canonical_hash(shared_payload)
        shared_relative = Path("components") / f"{shared_content_hash}.json.gz"
        shared_path = self.cache_root / shared_relative
        if shared_relative.as_posix() not in self.components:
            atomic_write_bytes(shared_path, gzip.compress(canonical_json_bytes(shared_payload), compresslevel=6, mtime=0))
            self.components[shared_relative.as_posix()] = {
                "path": shared_relative.as_posix(),
                "bytes": shared_path.stat().st_size,
                "sha256": sha256_file(shared_path),
                "encoding": "canonical_json_gzip_mtime0",
                "shared_content_sha256": shared_content_hash,
            }
        path = self.cache_root / relative
        reference = {
            "schema": "family_input_reference_v1",
            "shared_payload_path": shared_relative.as_posix(),
            "shared_payload_sha256": self.components[shared_relative.as_posix()]["sha256"],
            "campaign_partition": _content_partition(partition),
        }
        atomic_write_bytes(path, canonical_json_bytes(reference))
        decoded = decode_family_input(_read_reference_payload(
            path,
            cache_root=self.cache_root,
            components_by_path=self.components,
        ), cache_root=self.cache_root, external_components_verified=True)
        if decoded.content_sha256() != content_hash:
            raise CacheBuildError("cache encode/decode replay mismatch")
        record = {
            "path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "encoding": "family_input_reference_v1",
            "frame_content_sha256": content_hash, "fold_id": frame.fold_id, "symbol": frame.symbol,
            "decision_ts": require_utc(frame.decision_ts).isoformat(),
            "campaign_partition": _content_partition(partition),
            "shared_payload_path": shared_relative.as_posix(),
            "shared_content_sha256": shared_content_hash,
        }
        self.paths.add(relative.as_posix())
        self.artifacts.append(record)
        return record

    def finalize(self) -> Path:
        artifacts = sorted(
            self.artifacts,
            key=lambda record: (
                str(record["fold_id"]),
                str(record["symbol"]),
                str(record["decision_ts"]),
                str(record["frame_content_sha256"]),
            ),
        )
        if not artifacts:
            raise CacheBuildError("semantic cache cannot be empty")
        manifest = {
            "schema": self.execution_input_authority["cache_manifest_contract"]["schema"],
            "platform": self.execution_input_authority["platform"],
            "rankable_interval": self.execution_input_authority["rankable_interval"],
            **self.bindings,
            "rankable_funding_package_sha256": self.execution_input_authority.get("rankable_funding_package_sha256"),
            "source_record_inventory_sha256": self.source_composite,
            "artifacts": artifacts,
            "artifact_inventory_sha256": canonical_hash(artifacts),
            "components": sorted(self.components.values(), key=lambda row: str(row["path"])),
            "component_inventory_sha256": canonical_hash(sorted(self.components.values(), key=lambda row: str(row["path"]))),
            "typed_unavailable": sorted(
                self.typed_unavailable,
                key=lambda row: (
                    str(row["family_id"]),
                    str(row["campaign_partition"]["phase"]),
                    str(row["campaign_partition"]["outer_fold_id"]),
                    str(row["campaign_partition"]["inner_fold_id"]),
                ),
            ),
            "construction": "canonical decoded FamilyInput payload from exact physically verified authorities",
            "synthetic_only": self.synthetic_only,
        }
        manifest_path = self.cache_root / "SEMANTIC_CACHE_MANIFEST.json"
        atomic_write_json(manifest_path, manifest)
        return manifest_path

def decoded_frame_with_locator(
    path: Path,
    relative: str,
    bindings: Mapping[str, str],
    *,
    encoding: str = "canonical_json",
    cache_root: Path | None = None,
    components_by_path: Mapping[str, Mapping[str, Any]] | None = None,
) -> FamilyInput:
    if encoding == "family_input_reference_v1":
        if cache_root is None or components_by_path is None:
            raise CacheBuildError("semantic-cache reference decoder lacks its component inventory")
        payload = _read_reference_payload(path, cache_root=cache_root, components_by_path=components_by_path)
    else:
        payload = _read_payload(path, encoding)
    frame = decode_family_input(payload, cache_root=cache_root, external_components_verified=True)
    located = replace(frame, metadata={**frame.metadata, "cache_artifact_path": relative, "cache_bindings": dict(bindings)})
    _mark_validated_frame(located)
    return located


def _content_partition(partition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: require_utc(value).isoformat() if key.endswith(("_start", "_exclusive")) else value
        for key, value in partition.items()
    }


__all__ = ["CacheBuildError", "SemanticCacheWriter", "build_semantic_cache", "decode_family_input", "decoded_frame_with_locator"]
