from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .engine_types import ContextInputs, DailyBar, FamilyInput, FundingInput, SignalBar, ThresholdPopulation
from .family_engines.common import EngineInputError, require_utc


class CacheBuildError(RuntimeError):
    pass


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
    if isinstance(value, str) and (key or "").endswith(("_ts", "_start", "_end", "_exclusive")):
        try:
            return _datetime(value)
        except CacheBuildError:
            return value
    return value


def decode_family_input(payload: Mapping[str, Any]) -> FamilyInput:
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
            tuple(float(value) for value in row["values"]), tuple(str(value) for value in row["unique_symbols"]), str(row["scope"]),
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
    bindings = _authority_bindings(execution_input_authority)
    records_by_role = {}
    for record in execution_input_authority.get("source_records", ()):
        path = authority_root / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise CacheBuildError(f"source authority mismatch: {record.get('role')}")
        records_by_role[str(record["role"])] = record
    if not records_by_role:
        raise CacheBuildError("cache construction has no verified source authorities")
    source_composite = canonical_hash(execution_input_authority["source_records"])
    artifact_root = cache_root / "frames"
    artifacts = []
    for frame in sorted(frames, key=lambda item: (item.fold_id, item.symbol, item.decision_ts.isoformat(), item.content_sha256())):
        frame.validate()
        partition = _campaign_partition(frame)
        provenance = frame.metadata.get("source_authority")
        expected_provenance = {
            **bindings,
            "source_record_inventory_sha256": source_composite,
            "rankable_funding_package_sha256": execution_input_authority.get("rankable_funding_package_sha256"),
        }
        if provenance != expected_provenance:
            raise CacheBuildError("frame was not derived under the exact execution-input authority")
        payload = frame.content_payload()
        content_hash = canonical_hash(payload)
        relative = Path("frames") / f"{content_hash}.json"
        path = cache_root / relative
        atomic_write_json(path, payload)
        decoded = decode_family_input(json.loads(path.read_text(encoding="utf-8")))
        if decoded.content_sha256() != content_hash:
            raise CacheBuildError("cache encode/decode replay mismatch")
        artifacts.append({
            "path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "frame_content_sha256": content_hash, "fold_id": frame.fold_id, "symbol": frame.symbol,
            "decision_ts": require_utc(frame.decision_ts).isoformat(),
            "campaign_partition": _content_partition(partition),
        })
    if not artifacts:
        raise CacheBuildError("semantic cache cannot be empty")
    manifest = {
        "schema": execution_input_authority["cache_manifest_contract"]["schema"],
        "platform": execution_input_authority["platform"],
        "rankable_interval": execution_input_authority["rankable_interval"],
        **bindings,
        "rankable_funding_package_sha256": execution_input_authority.get("rankable_funding_package_sha256"),
        "source_record_inventory_sha256": source_composite,
        "artifacts": artifacts,
        "artifact_inventory_sha256": canonical_hash(artifacts),
        "construction": "canonical decoded FamilyInput payload from exact physically verified authorities",
        "synthetic_only": bool(synthetic_only),
    }
    manifest_path = cache_root / "SEMANTIC_CACHE_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def decoded_frame_with_locator(path: Path, relative: str, bindings: Mapping[str, str]) -> FamilyInput:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = decode_family_input(payload)
    return replace(frame, metadata={**frame.metadata, "cache_artifact_path": relative, "cache_bindings": dict(bindings)})


def _content_partition(partition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: require_utc(value).isoformat() if key.endswith(("_start", "_exclusive")) else value
        for key, value in partition.items()
    }


__all__ = ["CacheBuildError", "build_semantic_cache", "decode_family_input", "decoded_frame_with_locator"]
