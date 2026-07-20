from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .canonical import atomic_write_bytes, atomic_write_json, atomic_write_jsonl, canonical_hash, canonical_json_bytes, sha256_file
from .budget import optimize_budget
from .controls import compile_controls, coverage_rows
from .family_engines import ENGINES
from .generator import GENERATOR_ID, GENERATOR_SEED, approximate_discrepancy, approximate_nearest_neighbor, point, stream
from .schema import (
    CAMPAIGN_ID,
    FAMILY_ORDER,
    SCHEMA_VERSION,
    SchemaError,
    axis_fixture_config,
    baseline_config,
    complexity,
    economic_address,
    family_schemas,
    generation_levels,
    marginal_levels,
    mapped_config,
    normalize_config,
    schema_document,
    schema_hash,
    search_axes,
)
from .selection import safe_pruning_policy


EXPECTED_HASHES = {
    "stage21_v1_zip": "d5c7c8a99e55e6c0fa27a366d9cd838b19bdecb5320ffb02c936d10c5d14e960",
    "stage21_v2_strategy_registry": "7221956dbc520797b178da3a744ee35fc3d3c6115c4ca7ea5b6f7e1bc4903ea1",
    "stage21_v2_control_registry": "491d341e915e7598e1672c337a6cbedc1c80ddf59cd84043ee98a5e0571cab0f",
    "stage21_v3_contract": "b7ea01f0537bae1efac468bc5a6c8f1ec292c5fe1edddbd98b13444339382f0e",
    "stage21_v4_addendum": "3985e316ba175a96e6d3a969f8a1671d1aa318e0de092ee7763975bc594cd87a",
    "stage21_v5_closure": "6b21272f61ef808f65f59d3c26dd70ff8f6808a94bd69b87149f2858fbdd5c18",
    "stage21_v5_zip": "0f7bb1ff3d62b2cfb0f1e14a6cfbd6a52d0bfd3d26e7331ed69f694dd570d8b9",
    "stage22_task": "58fbec98558850dd8cd39c00cf4b9fc57d38c32cdf5064d97f0e598766fcd169",
}

EXPECTED_SOURCE_EXECUTABLE = 2913
EXPECTED_SOURCE_CONDITIONAL = 608
EXPECTED_LEGACY_PROJECTIONS = 3776
EXPECTED_FINAL_CONTROLS = 800


@dataclass(frozen=True)
class SourcePaths:
    stage21_v1_zip: Path
    stage21_v5_zip: Path
    strategy_registry: Path
    control_registry: Path
    v3_contract: Path
    v4_addendum: Path
    v5_closure: Path
    stage22_task: Path

    def hash_paths(self) -> dict[str, Path]:
        return {
            "stage21_v1_zip": self.stage21_v1_zip,
            "stage21_v2_strategy_registry": self.strategy_registry,
            "stage21_v2_control_registry": self.control_registry,
            "stage21_v3_contract": self.v3_contract,
            "stage21_v4_addendum": self.v4_addendum,
            "stage21_v5_closure": self.v5_closure,
            "stage21_v5_zip": self.stage21_v5_zip,
            "stage22_task": self.stage22_task,
        }


def verify_sources(paths: SourcePaths) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for key, path in paths.hash_paths().items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != EXPECTED_HASHES[key]:
            raise ValueError(f"source hash mismatch for {key}: {actual}")
        verified[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    strategy_rows = sum(1 for line in paths.strategy_registry.open("rb") if line.strip())
    control_rows = sum(1 for line in paths.control_registry.open("rb") if line.strip())
    if strategy_rows != 3521 or control_rows != 800:
        raise ValueError(f"source registry count mismatch: {strategy_rows}/{control_rows}")
    verified["source_row_counts"] = {"strategy": strategy_rows, "control": control_rows}
    return verified


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    tmp = Path(raw_tmp)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    tmp = Path(raw_tmp)
    try:
        table = pa.Table.from_pylist([dict(row) for row in rows])
        pq.write_table(table, tmp, compression="zstd", version="2.6", write_statistics=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _source_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("source JSONL contains non-object")
    return rows


def _legacy_config(row: Mapping[str, Any], direction: str | None = None, exact_parent_id: str | None = None) -> dict[str, Any]:
    family = str(row["family_id"])
    config = dict(row["config"])
    if family == "A4_TSMOM_V7":
        config.update({"ATR_window_days_for_ATR_exits": 20, "volatility_estimator": "close_to_close"})
    elif family == "A1_COMPRESSION_V2":
        config.update({
            "ATR_window_days": 20,
            "contraction_baseline": "adjacent_equal_duration",
            "impulse_rank_scope": "symbol_side",
            "shape_rank_scope": "symbol",
        })
    elif family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        config.pop("parent_binding")
        if exact_parent_id is None:
            raise ValueError("A2 legacy projection requires one exact projected parent")
        enabled = any((
            config["BTC_ETH_context"] != "none",
            config["RS_rank"] != "none",
            config["breadth_dispersion"] != "none",
            config["proximity_rank"] != "none",
            config["reclaim_state"] != "none",
        ))
        if not enabled:
            config["overlay_action"] = "parent_only"
        config.update({
            "parent_binding_mode": "source_attempt",
            "parent_source_attempt_id": exact_parent_id,
            "ATR_window_days_for_proximity": 20,
            "BTC_ETH_drawdown_lookback_days": 60,
            "BTC_ETH_trend_pair_days": "20_60",
            "BTC_ETH_volatility_lookback_days": 20,
            "RS_population_scope": "global_PIT",
            "breadth_return_lookback_days": 20,
            "dispersion_return_lookback_days": 20,
        })
    elif family == "A3_STARTER_RETEST_V3":
        if direction not in {"long", "short"}:
            raise ValueError("A3 legacy projection requires a direction")
        config.update({"direction": direction, "ATR_window_days": 20, "breakout_rank_scope": "symbol_side"})
    return normalize_config(family, config)


def _selection_role(row: Mapping[str, Any]) -> str:
    family = row["family_id"]
    lane = row["lane"]
    context = row.get("config", {}).get("context_overlay")
    if family in {"A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"} and context in {"prior_high_RS", "BTC_ETH", "breadth_dispersion"}:
        return "source_prior_anchor_not_main_beam"
    if lane == "anchor_ablation":
        return "mechanism_anchor_or_ablation"
    if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        return "legacy_exact_parent_overlay_anchor"
    if family == "KDA02B_SURVIVOR_ADJUDICATION_V1":
        return "fixed_adjudication"
    return "legacy_screening_eligible"


def normalize_legacy(source_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    status_counts = Counter(str(row.get("status")) for row in source_rows)
    if status_counts["registered_executable"] != EXPECTED_SOURCE_EXECUTABLE or status_counts["registered_conditional"] != EXPECTED_SOURCE_CONDITIONAL:
        raise ValueError(f"legacy source status counts differ: {status_counts}")
    ledger: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    source_to_projection_count: Counter[str] = Counter()
    legacy_seen: dict[str, str] = {}
    source_projection_ids: dict[str, tuple[str, ...]] = {}
    for source_index, row in enumerate(source_rows, start=1):
        family = str(row["family_id"])
        source_attempt = str(row["attempt_id"])
        if row["status"] != "registered_executable":
            continue
        count = 2 if family == "A3_STARTER_RETEST_V3" else 1
        source_projection_ids[source_attempt] = tuple(f"{family}:S22:L:{source_index:04d}:{number}" for number in range(1, count + 1))
    duplicate_count = 0
    for source_index, row in enumerate(source_rows, start=1):
        family = str(row["family_id"])
        source_attempt = str(row["attempt_id"])
        source_payload_hash = canonical_hash(row)
        if row["status"] == "registered_conditional":
            parent_mismatch = False
            if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                parent_mismatch = not str(row["config"]["parent_binding"]).startswith(str(row["config"]["parent_family"]) + ":")
            ledger.append({
                "source_row_index": source_index,
                "source_attempt_id": source_attempt,
                "family": family,
                "source_status": row["status"],
                "normalization_status": "inherited_conditional_not_executed_in_new_broad_screen",
                "projection_count": 0,
                "source_payload_sha256": source_payload_hash,
                "lineage_defect": "parent_family_parent_binding_conflict" if parent_mismatch else None,
            })
            continue
        if family == "A3_STARTER_RETEST_V3":
            projection_dimensions: tuple[tuple[str | None, str | None], ...] = (("long", None), ("short", None))
        elif family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
            source_parent = str(row["config"]["parent_binding"])
            parent_ids = source_projection_ids.get(source_parent)
            if not parent_ids:
                raise ValueError(f"A2 exact source parent is not executable: {source_parent}")
            projection_dimensions = tuple((None, parent_id) for parent_id in parent_ids)
        else:
            projection_dimensions = ((None, None),)
        projection_ids = []
        for projection_number, (direction, exact_parent_id) in enumerate(projection_dimensions, start=1):
            config = _legacy_config(row, direction, exact_parent_id)
            _, address = economic_address(family, config)
            attempt_id = f"{family}:S22:L:{source_index:04d}:{projection_number}"
            disposition = "execute_once"
            duplicate_of = None
            if address in legacy_seen:
                disposition = "multiplicity_only_duplicate"
                duplicate_of = legacy_seen[address]
                duplicate_count += 1
            else:
                legacy_seen[address] = attempt_id
            projection = {
                "campaign_id": CAMPAIGN_ID,
                "executable_attempt_id": attempt_id,
                "source_attempt_id": source_attempt,
                "family_id": family,
                "lane": "legacy_projection",
                "selection_role": _selection_role(row),
                "status": "registered_executable",
                "execution_disposition": disposition,
                "duplicate_of_executable_attempt_id": duplicate_of,
                "canonical_economic_address_sha256": address,
                "schema_sha256": schema_hash(),
                "config": config,
                "lineage": {
                    "source_registry_sha256": EXPECTED_HASHES["stage21_v2_strategy_registry"],
                    "source_payload_sha256": source_payload_hash,
                    "source_lane": row["lane"],
                    "source_status": row["status"],
                    "projection_rule": "a3_directional_split_v1" if direction else "typed_legacy_projection_v1",
                },
            }
            if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                template_id = canonical_hash({"mode": "source_attempt", "parent_executable_attempt_id": exact_parent_id})
                projection.update({
                    "parent_binding_template_id": template_id,
                    "resolved_parent_executable_attempt_id": exact_parent_id,
                    "parent_only_counterpart_id": canonical_hash({"template": template_id, "role": "parent_only"}),
                    "overlay_counterpart_id": canonical_hash({"template": template_id, "overlay_address": address, "role": "context_overlay"}),
                })
            projections.append(projection)
            projection_ids.append(attempt_id)
            source_to_projection_count[source_attempt] += 1
        ledger.append({
            "source_row_index": source_index,
            "source_attempt_id": source_attempt,
            "family": family,
            "source_status": row["status"],
            "normalization_status": "normalized_executable_projection",
            "projection_count": len(projection_ids),
            "projection_ids_json": json.dumps(projection_ids, separators=(",", ":")),
            "source_payload_sha256": source_payload_hash,
            "lineage_defect": None,
        })
    if len(ledger) != 3521 or len(projections) != EXPECTED_LEGACY_PROJECTIONS:
        raise ValueError(f"legacy normalization count mismatch: {len(ledger)}/{len(projections)}")
    audit = {
        "source_rows": len(ledger),
        "source_reported_executable": status_counts["registered_executable"],
        "source_reported_conditional": status_counts["registered_conditional"],
        "executable_projections": len(projections),
        "a3_directional_split_additional_projections": sum(1 for row in source_rows if row["status"] == "registered_executable" and row["family_id"] == "A3_STARTER_RETEST_V3"),
        "a2_exact_a3_parent_split_additional_projections": sum(1 for row in source_rows if row["status"] == "registered_executable" and row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and row["config"]["parent_family"] == "A3_STARTER_RETEST_V3"),
        "duplicate_projection_count": duplicate_count,
        "unique_economic_addresses": len({row["canonical_economic_address_sha256"] for row in projections}),
        "conditional_replacement_budget": EXPECTED_SOURCE_CONDITIONAL,
    }
    return ledger, projections, audit


def generate_new_broad(existing_rows: Sequence[Mapping[str, Any]], broad_counts: Mapping[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    existing_addresses = {str(row["canonical_economic_address_sha256"]) for row in existing_rows}
    rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    family_summary: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        if family not in broad_counts:
            continue
        specs = search_axes(family)
        accepted = 0
        rejected_invalid = 0
        rejected_duplicate = 0
        accepted_points: list[tuple[float, ...]] = []
        for stream_index, coordinates in stream(family, len(specs)):
            raw = mapped_config(family, coordinates)
            status = "accepted"
            reason = None
            normalized: dict[str, Any] | None = None
            address = None
            try:
                normalized = normalize_config(family, raw)
                _, address = economic_address(family, normalized)
                if address in existing_addresses:
                    status = "rejected_duplicate"
                    reason = "canonical_economic_address_already_registered"
                    rejected_duplicate += 1
            except SchemaError as exc:
                status = "rejected_invalid"
                reason = str(exc)
                rejected_invalid += 1
            coordinate_rows.append({
                "family": family,
                "stream_index": stream_index,
                "generator_id": GENERATOR_ID,
                "generator_seed": GENERATOR_SEED,
                "coordinates_json": json.dumps(list(coordinates), separators=(",", ":")),
                "mapped_values_json": json.dumps(raw, sort_keys=True, separators=(",", ":")),
                "normalized_config_json": json.dumps(normalized, sort_keys=True, separators=(",", ":")) if normalized is not None else None,
                "status": status,
                "reason": reason,
                "canonical_economic_address_sha256": address,
            })
            if status != "accepted":
                if stream_index >= 1_000_000:
                    raise RuntimeError(f"valid-domain exhaustion for {family}")
                continue
            assert normalized is not None and address is not None
            accepted += 1
            accepted_points.append(coordinates)
            attempt_id = f"{family}:S22:B:{accepted:05d}"
            new_row = {
                "campaign_id": CAMPAIGN_ID,
                "executable_attempt_id": attempt_id,
                "source_attempt_id": None,
                "family_id": family,
                "lane": "broad_space_filling",
                "selection_role": "conditional_parent_overlay_template" if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else "main_broad_screening",
                "status": "registered_conditional_parent_slot" if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else "registered_executable",
                "execution_disposition": "execute_if_parent_available" if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else "execute_once",
                "duplicate_of_executable_attempt_id": None,
                "canonical_economic_address_sha256": address,
                "schema_sha256": schema_hash(),
                "config": normalized,
                "lineage": {
                    "generator_id": GENERATOR_ID,
                    "generator_seed": GENERATOR_SEED,
                    "stream_index": stream_index,
                    "coordinate_sha256": canonical_hash(list(coordinates)),
                    "purpose": "v4_broad_coverage_plus_conditional_replacement",
                },
            }
            if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                parent_slot = f"{normalized['parent_family']}:{normalized['parent_fold_id']}:beam:{int(normalized['parent_beam_rank']):02d}"
                template_id = canonical_hash({"mode": "beam_slot", "parent_slot": parent_slot})
                new_row.update({
                    "parent_binding_template_id": template_id,
                    "resolved_parent_executable_attempt_id": None,
                    "parent_only_counterpart_id": canonical_hash({"template": template_id, "role": "parent_only"}),
                    "overlay_counterpart_id": canonical_hash({"template": template_id, "overlay_address": address, "role": "context_overlay"}),
                })
            rows.append(new_row)
            existing_addresses.add(address)
            if accepted == broad_counts[family]:
                family_summary[family] = {
                    "accepted": accepted,
                    "coordinates_consumed": stream_index + 1,
                    "rejected_invalid": rejected_invalid,
                    "rejected_duplicate": rejected_duplicate,
                    "approximate_one_dimensional_discrepancy": approximate_discrepancy(accepted_points),
                    "approximate_nearest_neighbor": approximate_nearest_neighbor(accepted_points),
                }
                break
    if len(rows) != sum(broad_counts.values()):
        raise AssertionError("new broad count mismatch")
    return rows, coordinate_rows, family_summary


def _coverage(final_rows: Sequence[Mapping[str, Any]], broad_families: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    marginal: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    unrepresented: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        family_rows = [row for row in final_rows if row["family_id"] == family]
        coverage_rows_for_family = [row for row in family_rows if row["lane"] == "broad_space_filling"] if family in broad_families else family_rows
        configs = [row["config"] for row in coverage_rows_for_family]
        family_marginal_failures = 0
        for field, level in marginal_levels(family):
            count = sum(config.get(field) == level for config in configs)
            minimum = 50 if family in broad_families else 1
            status = "pass" if count >= minimum else "fail"
            family_marginal_failures += status == "fail"
            marginal.append({"family": family, "field": field, "level_json": json.dumps(level, separators=(",", ":")), "count": count, "minimum": minimum, "status": status})
            if status == "fail":
                unrepresented.append({"family": family, "region_type": "marginal", "field_a": field, "value_a_json": json.dumps(level), "field_b": None, "value_b_json": None, "reason": "below_minimum_coverage"})
        family_pair_failures = 0
        for field_a, field_b in family_schemas[family].priority_pairs:
            spec_a = family_schemas[family].axis_map[field_a]
            spec_b = family_schemas[family].axis_map[field_b]
            for value_a in generation_levels(family, spec_a):
                for value_b in generation_levels(family, spec_b):
                    candidate = baseline_config(family)
                    candidate[field_a] = value_a
                    candidate[field_b] = value_b
                    logically_valid = True
                    try:
                        normalized = normalize_config(family, candidate)
                        logically_valid = normalized[field_a] == value_a and normalized[field_b] == value_b
                    except SchemaError:
                        logically_valid = False
                    count = sum(config.get(field_a) == value_a and config.get(field_b) == value_b for config in configs)
                    minimum = 20 if logically_valid and family in broad_families else 0 if not logically_valid else 1
                    status = "unavailable_logical_constraint" if not logically_valid else "pass" if count >= minimum else "fail"
                    family_pair_failures += status == "fail"
                    pairwise.append({
                        "family": family,
                        "axis_a": field_a,
                        "value_a_json": json.dumps(value_a, separators=(",", ":")),
                        "axis_b": field_b,
                        "value_b_json": json.dumps(value_b, separators=(",", ":")),
                        "count": count,
                        "minimum": minimum,
                        "status": status,
                    })
                    if status == "fail":
                        unrepresented.append({"family": family, "region_type": "priority_pair", "field_a": field_a, "value_a_json": json.dumps(value_a), "field_b": field_b, "value_b_json": json.dumps(value_b), "reason": "below_minimum_coverage"})
        summary[family] = {
            "attempt_rows": len(family_rows),
            "broad_coverage_rows": len(coverage_rows_for_family),
            "unique_addresses": len({row["canonical_economic_address_sha256"] for row in family_rows}),
            "marginal_failures": int(family_marginal_failures),
            "priority_pair_failures": int(family_pair_failures),
        }
    return marginal, pairwise, unrepresented, summary


def _semantic_coverage() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    semantic: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for spec in family_schemas[family].axes:
            levels = spec.allowed_values if spec.allowed_values is not None else ("<exact_string>",)
            for value in levels:
                fixture_value = "A1_COMPRESSION_V2:anchor_ablation:0001" if value == "<exact_string>" else value
                config = axis_fixture_config(family, spec.name, fixture_value)
                normalized = normalize_config(family, config)
                active = normalized[spec.name] is not None
                semantic.append({
                    "family": family,
                    "field": spec.name,
                    "value_json": json.dumps(value, separators=(",", ":")),
                    "formula_id": spec.formula_id,
                    "classification": spec.classification,
                    "active_fixture": active,
                    "focused_test": f"test_every_axis_value__{family}__{spec.name}",
                    "status": "pass" if active else "inactive_by_contract",
                })
            baseline = baseline_config(family)
            normalized_baseline = normalize_config(family, baseline)
            truth.append({
                "family": family,
                "field": spec.name,
                "branch": "baseline",
                "active": normalized_baseline[spec.name] is not None,
                "config_sha256": canonical_hash(baseline),
            })
            fixture = axis_fixture_config(family, spec.name, next(iter(levels)) if next(iter(levels)) != "<exact_string>" else "A1_COMPRESSION_V2:anchor_ablation:0001")
            normalized_fixture = normalize_config(family, fixture)
            truth.append({
                "family": family,
                "field": spec.name,
                "branch": "forced_active_fixture",
                "active": normalized_fixture[spec.name] is not None,
                "config_sha256": canonical_hash(fixture),
            })
    invalid = [
        {"case_id": "unknown_field", "family": "A4_TSMOM_V7", "expected": "SchemaError", "contract": "unknown fields fail packet compilation"},
        {"case_id": "a3_exposure_over_one", "family": "A3_STARTER_RETEST_V3", "expected": "SchemaError", "contract": "starter_fraction+add_fraction<=1"},
        {"case_id": "a2_empty_components", "family": "A2_PRIOR_HIGH_RS_CONTEXT_V1", "expected": "SchemaError", "contract": "at least one context component"},
        {"case_id": "a2_parent_mismatch", "family": "A2_PRIOR_HIGH_RS_CONTEXT_V1", "expected": "SchemaError", "contract": "source parent family matches exact parent ID"},
        {"case_id": "wrong_scalar_type", "family": "A1_COMPRESSION_V2", "expected": "SchemaError", "contract": "typed value enforcement"},
    ]
    return semantic, truth, invalid


def _legacy_control_lineage(path: Path) -> list[dict[str, Any]]:
    rows = _source_rows(path)
    return [{
        "source_row_index": index,
        "source_control_attempt_id": row.get("control_attempt_id") or row.get("control_template_id") or row.get("control_id") or f"source-control-{index:04d}",
        "source_payload_sha256": canonical_hash(row),
        "source_registry_sha256": EXPECTED_HASHES["stage21_v2_control_registry"],
        "stage22_status": "immutable_lineage_superseded_by_finite_stage22_control_table",
    } for index, row in enumerate(rows, start=1)]


def compile_deterministic(paths: SourcePaths, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    authority = verify_sources(paths)
    atomic_write_json(output_root / "SOURCE_AUTHORITY.json", {"schema": "stage22_source_authority_v1", "status": "pass", "sources": authority})
    schema_doc = schema_document()
    atomic_write_json(output_root / "FAMILY_AXIS_SCHEMA.json", schema_doc)
    atomic_write_bytes(output_root / "FAMILY_AXIS_SCHEMA.sha256", (sha256_file(output_root / "FAMILY_AXIS_SCHEMA.json") + "  FAMILY_AXIS_SCHEMA.json\n").encode("utf-8"))

    source_rows = _source_rows(paths.strategy_registry)
    legacy_ledger, legacy_projections, legacy_audit = normalize_legacy(source_rows)
    _write_parquet(output_root / "LEGACY_NORMALIZATION_LEDGER.parquet", legacy_ledger)
    atomic_write_jsonl(output_root / "LEGACY_EXECUTABLE_PROJECTION.jsonl", legacy_projections)
    duplicate_report = (
        "# Legacy duplicate and supersession audit\n\n"
        f"- Source rows: `{legacy_audit['source_rows']}`\n"
        f"- Source-reported executable: `{legacy_audit['source_reported_executable']}`\n"
        f"- Source-reported conditional and not executed: `{legacy_audit['source_reported_conditional']}`\n"
        f"- Typed executable projections: `{legacy_audit['executable_projections']}`\n"
        f"- Added projections from A3 directional split: `{legacy_audit['a3_directional_split_additional_projections']}`\n"
        f"- Duplicate projections executed once: `{legacy_audit['duplicate_projection_count']}`\n\n"
        "The 608 outcome-conditional refinement rows remain immutable lineage and are not treated as unconditional. "
        "Their multiplicity is replaced by ex-ante broad coordinates. Provenance fields never enter the economic-address hash.\n"
    )
    atomic_write_bytes(output_root / "LEGACY_DUPLICATE_AND_SUPERSESSION_AUDIT.md", duplicate_report.encode("utf-8"))

    existing_counts = Counter(row["family_id"] for row in legacy_projections)
    budget = optimize_budget(dict(existing_counts))
    broad_counts = budget["new_broad_counts"]
    atomic_write_json(output_root / "OUTCOME_FREE_BUDGET_OPTIMIZER.json", budget)
    new_rows, raw_coordinates, generator_summary = generate_new_broad(legacy_projections, broad_counts)
    final_rows = sorted(
        [*legacy_projections, *new_rows],
        key=lambda row: (FAMILY_ORDER.index(row["family_id"]), 0 if row["lane"] == "legacy_projection" else 1, row["executable_attempt_id"]),
    )
    expected_final_attempts = int(budget["target_total_attempt_rows"])
    if len(final_rows) != expected_final_attempts:
        raise AssertionError(f"final attempt count {len(final_rows)} != {expected_final_attempts}")
    if len({row["executable_attempt_id"] for row in final_rows}) != len(final_rows):
        raise AssertionError("duplicate executable attempt IDs")
    atomic_write_jsonl(output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", final_rows)
    execution_rows = [row for row in final_rows if row["execution_disposition"] in {"execute_once", "execute_if_parent_available"}]
    if len({row["canonical_economic_address_sha256"] for row in execution_rows}) != len(execution_rows):
        raise AssertionError("execute-once registry still contains duplicate economic addresses")
    atomic_write_jsonl(output_root / "FINAL_EXECUTION_REGISTRY.jsonl", execution_rows)
    a2_counterparts = [{
        "a2_executable_attempt_id": row["executable_attempt_id"],
        "parent_binding_mode": row["config"]["parent_binding_mode"],
        "parent_binding_template_id": row["parent_binding_template_id"],
        "parent_family": row["config"]["parent_family"],
        "parent_fold_id": row["config"].get("parent_fold_id"),
        "parent_beam_rank": row["config"].get("parent_beam_rank"),
        "resolved_parent_executable_attempt_id": row["resolved_parent_executable_attempt_id"],
        "missing_parent_behavior": "unavailable_no_parent",
        "parent_only_counterpart_id": row["parent_only_counterpart_id"],
        "overlay_counterpart_id": row["overlay_counterpart_id"],
        "overlay_economic_address_sha256": row["canonical_economic_address_sha256"],
        "atomic_binding_compiler": "stage22_a2_parent_counterpart_v1",
    } for row in final_rows if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and row["execution_disposition"] != "multiplicity_only_duplicate"]
    atomic_write_jsonl(output_root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl", a2_counterparts)
    _write_parquet(output_root / "RAW_SPACE_FILLING_COORDINATES.parquet", raw_coordinates)

    marginal, pairwise, unrepresented, coverage_summary = _coverage(final_rows, set(broad_counts))
    _write_csv(output_root / "SEARCH_SPACE_COVERAGE_MATRIX.csv", ("family", "field", "level_json", "count", "minimum", "status"), marginal)
    _write_parquet(output_root / "PAIRWISE_COVERAGE_MATRIX.parquet", pairwise)
    _write_csv(output_root / "UNREPRESENTED_VALID_REGIONS.csv", ("family", "region_type", "field_a", "value_a_json", "field_b", "value_b_json", "reason"), unrepresented)

    semantic, truth, invalid = _semantic_coverage()
    _write_csv(output_root / "SEMANTIC_COVERAGE_MATRIX.csv", ("family", "field", "value_json", "formula_id", "classification", "active_fixture", "focused_test", "status"), semantic)
    _write_csv(output_root / "ACTIVE_IF_TRUTH_TABLE.csv", ("family", "field", "branch", "active", "config_sha256"), truth)
    _write_csv(output_root / "INVALID_COMBINATION_MATRIX.csv", ("case_id", "family", "expected", "contract"), invalid)

    fixtures = {
        "schema": "stage22_registry_replay_fixtures_v1",
        "generator": {"id": GENERATOR_ID, "seed": GENERATOR_SEED},
        "fixed_points": {
            family: [{"stream_index": index, "coordinates": list(point(family, index, min(4, len(search_axes(family)))))} for index in range(3)]
            for family in broad_counts
        },
        "registry": {
            "rows": len(final_rows),
            "first_row": final_rows[0],
            "last_row": final_rows[-1],
            "sha256": sha256_file(output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"),
        },
    }
    atomic_write_json(output_root / "REGISTRY_REPLAY_FIXTURES.json", fixtures)

    source_control_rows = _source_rows(paths.control_registry)
    controls = compile_controls(source_control_rows)
    atomic_write_jsonl(output_root / "FINAL_CONTROL_REGISTRY.jsonl", controls)
    control_coverage = list(coverage_rows(controls))
    _write_csv(output_root / "CONTROL_COVERAGE_MATRIX.csv", ("family", "control_id", "rows", "folds", "beam_slots", "unique_addresses", "status"), control_coverage)
    atomic_write_json(output_root / "CONTROL_REPLAY_FIXTURES.json", {
        "schema": "stage22_control_replay_fixtures_v1",
        "rows": len(controls),
        "first_row": controls[0],
        "last_row": controls[-1],
        "sha256": sha256_file(output_root / "FINAL_CONTROL_REGISTRY.jsonl"),
    })
    _write_parquet(output_root / "LEGACY_CONTROL_LINEAGE.parquet", _legacy_control_lineage(paths.control_registry))

    engine_dir = output_root / "ENGINE_CONTRACTS"
    for family in FAMILY_ORDER:
        atomic_write_json(engine_dir / f"{family}.json", {"family": family, **ENGINES[family].contract()})
    # Import after module initialization to avoid a compiler/validator import
    # cycle; these probes execute the real dispatcher and accounting path.
    from .validators import accounting_probe, control_engine_probe, selection_route_probe, semantic_engine_probe
    engine_probe = semantic_engine_probe()
    control_probe = control_engine_probe()
    route_probe = selection_route_probe()
    account_probe = accounting_probe()
    if not all(item["pass"] for item in (engine_probe, control_probe, route_probe, account_probe)):
        raise AssertionError("an executable semantic/control/accounting/selection probe failed")
    _write_csv(output_root / "ENGINE_COVERAGE_MATRIX.csv", ("family", "field", "value_json", "formula_id", "interpreter", "fixture", "ledger_rows", "status"), engine_probe["coverage"])
    _write_csv(output_root / "CONTROL_EXECUTION_COVERAGE_MATRIX.csv", ("family", "control_id", "dispatcher", "ledger_rows", "observation_rows", "status"), control_probe["coverage"])
    _write_csv(output_root / "SELECTION_ROUTE_MATRIX.csv", ("family", "case", "route", "status"), route_probe["rows"])
    exit_rows = [row for row in engine_probe["coverage"] if row["field"] in {"exit", "fixed_target_R", "direction", "add_fraction", "adjudication_variant"}]
    _write_csv(output_root / "EXIT_ACCOUNTING_MATRIX.csv", ("family", "field", "value_json", "formula_id", "interpreter", "fixture", "ledger_rows", "status"), exit_rows)
    atomic_write_json(output_root / "ENGINE_PROBE_AUDIT.json", engine_probe)
    atomic_write_json(output_root / "CONTROL_ENGINE_PROBE_AUDIT.json", control_probe)
    atomic_write_json(output_root / "SELECTION_ROUTE_PROBE_AUDIT.json", route_probe)
    atomic_write_json(output_root / "ACCOUNTING_PROBE_AUDIT.json", account_probe)
    atomic_write_json(output_root / "SAFE_PRUNING_POLICY.json", safe_pruning_policy())
    atomic_write_json(output_root / "SHARED_SEMANTIC_CACHE_CONTRACT.json", {
        "schema": "stage22_shared_semantic_cache_contract_v1",
        "cache_objects": ["decision_calendar", "PIT_universe_and_lagged_liquidity", "daily_and_5m_feature_panels", "prior_highs_lows", "ATR_and_volatility", "breadth_dispersion", "BTC_ETH_context", "funding", "entry_exit_lookup_arrays", "event_keys"],
        "hash_inputs": ["source_manifest_sha256", "content_sha256", "family_axis_schema_sha256", "feature_formula_id", "rankable_interval", "protected_cutoff", "PIT_universe_hash", "funding_manifest_hash", "selected_event_key_hash"],
        "does_not_invalidate": ["report_wrapper", "narrative_text", "presentation_only_format"],
        "must_invalidate": ["feature_meaning", "source_manifest", "schema", "protected_boundary", "selected_event_keys", "funding_semantics", "PIT_universe"],
        "unknown_input": "fail_closed",
    })
    atomic_write_json(output_root / "HISTORICAL_LINEAGE_DECISIONS.json", {
        "schema": "stage22_historical_lineage_decisions_v1",
        "programme_exposure_class": "program_exposed_historical",
        "historical_translation_decisions_immutable": {
            "tsmom_v6": "defer_current_translation",
            "a1_compression": "defer_current_translation",
            "prior_high_reclaim_v2": "defer_current_translation",
            "breakout_retest_v2": "current_translation_rejected_only",
            "KDA01": "KDA01_level3_repaired_no_primary_pass_stop",
            "KDA02A": "KDA02_level3_no_primary_pass_stop",
            "KDA02B": "execution_sensitive_candidate",
            "KDA02C": "sample_limited_prospective_candidate",
            "KDX01": "translation_rejected"
        },
        "interpretation": "Stage 22 compiles materially new typed addresses; it does not rewrite or rescue any historical terminal decision",
        "claim_limits": ["not independent validation", "not live-ready", "no deployment claim"],
    })
    result = {
        "campaign_id": CAMPAIGN_ID,
        "schema_sha256": schema_hash(),
        "legacy_audit": legacy_audit,
        "budget_optimizer": budget,
        "new_broad_counts": broad_counts,
        "generator_summary": generator_summary,
        "final_attempt_rows": len(final_rows),
        "final_execution_rows": len(execution_rows),
        "final_unique_economic_addresses": len({row["canonical_economic_address_sha256"] for row in final_rows}),
        "final_control_rows": len(controls),
        "a2_parent_counterpart_rows": len(a2_counterparts),
        "coverage_summary": coverage_summary,
        "unrepresented_valid_region_count": len(unrepresented),
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
    }
    atomic_write_json(output_root / "COMPILER_SUMMARY.json", result)
    return result
