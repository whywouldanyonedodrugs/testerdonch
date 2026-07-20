from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_hash, canonical_json_bytes
from .engine_types import FamilyInput
from .schema import CAMPAIGN_ID, OUTER_FOLDS


CONTROL_IDS: dict[str, tuple[str, ...]] = {
    "A4_TSMOM_V7": ("A4_SIGN_PERMUTED_MAIN_NULL", "A4_GENERIC_SIGNED_RETURN", "A4_VOL_SCALING_REMOVED", "A4_PATH_COMPONENT_REMOVED", "A4_CONTEXT_REMOVED"),
    "A1_COMPRESSION_V2": ("A1_MATCHED_PSEUDO_EVENT_MAIN_NULL", "A1_PRICE_ONLY_IMPULSE", "A1_CONTRACTION_REMOVED", "A1_SMOOTHNESS_REMOVED", "A1_CONTEXT_REMOVED"),
    "A2_PRIOR_HIGH_RS_CONTEXT_V1": ("A2_CONTEXT_PERMUTED_MAIN_NULL", "A2_PARENT_ONLY", "A2_PRIOR_HIGH_REMOVED", "A2_RS_REMOVED", "A2_EXTERNAL_CONTEXT_REMOVED"),
    "A3_STARTER_RETEST_V3": ("A3_RETEST_TIME_PERMUTED_MAIN_NULL", "A3_STARTER_ONLY", "A3_MATCHED_PSEUDO_EVENT", "A3_CONFIRMATION_REMOVED", "A3_CONTEXT_REMOVED"),
}


def effective_seed(payload: Mapping[str, Any]) -> int:
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(dict(payload))).digest()[:8], "big", signed=False)


def compile_controls(source_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rebind the immutable 800 source controls to the five-slot Stage-22 beam."""
    if len(source_rows) != 800:
        raise ValueError("the immutable source control registry must contain 800 rows")
    source_index = {
        (str(row["family_id"]), str(row["outer_fold_id"]), int(row["deterministic_beam_slot"]), str(row["control_id"])): row
        for row in source_rows
    }
    rows: list[dict[str, Any]] = []
    for family in CONTROL_IDS:
        for fold in OUTER_FOLDS:
            for beam_rank in range(1, 6):
                parent_slot = f"{family}:{fold}:beam:{beam_rank:02d}"
                for control_id in CONTROL_IDS[family]:
                    source = source_index.get((family, fold, beam_rank, control_id))
                    if source is None:
                        raise ValueError(f"source control is absent: {family}/{fold}/{beam_rank}/{control_id}")
                    identity = {
                        "campaign_id": CAMPAIGN_ID,
                        "family": family,
                        "fold": fold,
                        "parent_slot": parent_slot,
                        "control_id": control_id,
                        "replicate_index": 0,
                        "transformation_allocator_version": "stage22_exact_control_dispatch_v2",
                    }
                    seed = effective_seed(identity)
                    address_payload = {
                        **identity,
                        "effective_seed": seed,
                        "missing_parent_behavior": "unavailable_no_parent",
                        "duplicate_behavior": "unavailable_duplicate_address",
                    }
                    rows.append({
                        "control_attempt_id": canonical_hash({"stage22_control": address_payload}),
                        "control_id": control_id,
                        "family": family,
                        "fold": fold,
                        "parent_slot": parent_slot,
                        "beam_rank": beam_rank,
                        "replicate_index": 0,
                        "effective_seed": seed,
                        "transformation_allocator_version": "stage22_exact_control_dispatch_v2",
                        "missing_parent_behavior": "unavailable_no_parent",
                        "duplicate_behavior": "unavailable_duplicate_address",
                        "economic_address_sha256": canonical_hash(address_payload),
                        "status": "registered_conditional",
                        "lineage": {
                            "source_control_template_address_sha256": source["control_template_address_sha256"],
                            "source_prior_control_template_address_sha256": source["prior_control_template_address_sha256"],
                            "source_seed": source["seed"],
                        },
                    })
    if len(rows) != 800 or len({row["economic_address_sha256"] for row in rows}) != 800:
        raise AssertionError("Stage-22 control reconciliation failed")
    return rows


def coverage_rows(rows: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for family, control_ids in CONTROL_IDS.items():
        family_rows = [row for row in rows if row["family"] == family]
        for control_id in control_ids:
            selected = [row for row in family_rows if row["control_id"] == control_id]
            yield {
                "family": family,
                "control_id": control_id,
                "rows": len(selected),
                "folds": len({row["fold"] for row in selected}),
                "beam_slots": len({row["parent_slot"] for row in selected}),
                "unique_addresses": len({row["economic_address_sha256"] for row in selected}),
                "status": "pass" if len(selected) == 40 and len({row["economic_address_sha256"] for row in selected}) == 40 else "fail",
            }


def _permute_metadata(frames: Sequence[FamilyInput], *, group_fields: Sequence[str], value_field: str, seed: int) -> list[FamilyInput]:
    grouped: dict[tuple[Any, ...], list[FamilyInput]] = {}
    for frame in frames:
        key = tuple(frame.metadata[field] for field in group_fields)
        grouped.setdefault(key, []).append(frame)
    output: list[FamilyInput] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        group = sorted(grouped[key], key=lambda item: (item.symbol, item.decision_ts.isoformat()))
        values = [item.metadata[value_field] for item in group]
        random.Random(effective_seed({"base_seed": seed, "group": list(key), "field": value_field})).shuffle(values)
        for frame, value in zip(group, values):
            metadata = dict(frame.metadata); metadata[value_field] = value
            output.append(replace(frame, metadata=metadata))
    return sorted(output, key=lambda item: (item.symbol, item.decision_ts.isoformat()))


def transformed_inputs(control_id: str, frames: Sequence[FamilyInput], seed: int) -> tuple[list[FamilyInput], str]:
    """Return executable raw inputs and the exact engine-control branch.

    No return, fill, funding, exposure, or net field is mutated.  The transformed
    raw/config input is subsequently rerun through the production dispatcher.
    """
    if control_id == "A4_SIGN_PERMUTED_MAIN_NULL":
        return _permute_metadata(frames, group_fields=("control_symbol", "control_year"), value_field="control_signal_sign", seed=seed), control_id
    if control_id == "A2_CONTEXT_PERMUTED_MAIN_NULL":
        return _permute_metadata(frames, group_fields=("control_symbol", "control_utc_month", "control_liquidity_decile"), value_field="control_context_vector", seed=seed), control_id
    if control_id == "A3_RETEST_TIME_PERMUTED_MAIN_NULL":
        return _permute_metadata(frames, group_fields=("control_symbol", "control_utc_quarter", "control_side"), value_field="control_add_lag_bars", seed=seed), control_id
    if control_id in {"A1_MATCHED_PSEUDO_EVENT_MAIN_NULL", "A3_MATCHED_PSEUDO_EVENT"}:
        allocated: set[str] = set()
        selected: list[FamilyInput] = []
        for parent in sorted(frames, key=lambda item: (item.symbol, item.decision_ts.isoformat())):
            candidates = [item for item in parent.metadata.get("pseudo_candidate_frames", ()) if isinstance(item, FamilyInput) and item.decision_ts.isoformat() not in allocated]
            if not candidates:
                continue
            chosen = min(candidates, key=lambda item: canonical_hash({"seed": seed, "control_id": control_id, "parent": parent.decision_ts.isoformat(), "candidate": item.decision_ts.isoformat()}))
            allocated.add(chosen.decision_ts.isoformat())
            selected.append(chosen)
        return selected, control_id
    supported = {control for controls in CONTROL_IDS.values() for control in controls}
    if control_id not in supported:
        raise ValueError(f"unsupported registered control: {control_id}")
    return list(frames), control_id


def execute_control(
    control_row: Mapping[str, Any],
    parent_row: Mapping[str, Any],
    frames: Sequence[FamilyInput],
    *,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    parent_binding: Mapping[str, Any] | None = None,
    parent_frames: Sequence[FamilyInput] | None = None,
) -> dict[str, Any]:
    if control_row.get("family") != parent_row.get("family_id"):
        raise ValueError("control family differs from parent")
    transformed, engine_control_id = transformed_inputs(str(control_row["control_id"]), frames, int(control_row["effective_seed"]))
    from .executor import dispatch_registered_attempt
    result = dispatch_registered_attempt(
        parent_row,
        transformed,
        registry_by_id=registry_by_id,
        parent_binding=parent_binding,
        parent_frames=parent_frames,
        control_id=engine_control_id,
    )
    result["control_attempt_id"] = control_row["control_attempt_id"]
    result["control_economic_address_sha256"] = control_row["economic_address_sha256"]
    return result


__all__ = ["CONTROL_IDS", "compile_controls", "coverage_rows", "effective_seed", "execute_control", "transformed_inputs"]
