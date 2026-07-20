from __future__ import annotations

import hashlib
import random
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_hash, canonical_json_bytes
from .schema import CAMPAIGN_ID, FAMILY_ORDER, OUTER_FOLDS


CONTROL_TYPES: dict[str, tuple[str, ...]] = {
    "A4_TSMOM_V7": ("sign_permutation", "signed_return_only", "no_smoothness", "unscaled_exposure", "context_null"),
    "A1_COMPRESSION_V2": ("matched_pseudo_event", "price_only_impulse", "no_contraction", "no_smoothness", "context_null"),
    "A2_PRIOR_HIGH_RS_CONTEXT_V1": ("context_vector_permutation", "parent_only", "no_prior_level", "no_RS", "no_macro_breadth_dispersion"),
    "A3_STARTER_RETEST_V3": ("retest_time_permutation", "starter_only", "price_only_breakout", "zero_add", "context_null"),
    "KDA02B_SURVIVOR_ADJUDICATION_V1": ("matched_pseudo_event", "price_only", "OI_removed", "liquidation_removed", "generic_structure_control"),
}


def effective_seed(payload: Mapping[str, Any]) -> int:
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(dict(payload))).digest()[:8], "big", signed=False)


def compile_controls() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for fold in OUTER_FOLDS:
            for beam_rank in range(1, 9):
                parent_slot = f"{family}:{fold}:beam:{beam_rank:02d}"
                for control_id in CONTROL_TYPES[family]:
                    replicate_index = 0
                    identity = {
                        "campaign_id": CAMPAIGN_ID,
                        "family": family,
                        "fold": fold,
                        "parent_slot": parent_slot,
                        "control_id": control_id,
                        "replicate_index": replicate_index,
                        "transformation_allocator_version": "stage22_control_allocator_v1",
                    }
                    seed = effective_seed(identity)
                    address_payload = {
                        **identity,
                        "effective_seed": seed,
                        "missing_parent_behavior": "unavailable_no_parent",
                        "duplicate_behavior": "unavailable_duplicate_address",
                    }
                    address = canonical_hash(address_payload)
                    rows.append({
                        "control_attempt_id": f"S22:{family}:C:{fold}:B{beam_rank:02d}:{control_id}",
                        "control_id": control_id,
                        "family": family,
                        "fold": fold,
                        "parent_slot": parent_slot,
                        "replicate_index": replicate_index,
                        "effective_seed": seed,
                        "transformation_allocator_version": "stage22_control_allocator_v1",
                        "missing_parent_behavior": "unavailable_no_parent",
                        "duplicate_behavior": "unavailable_duplicate_address",
                        "economic_address_sha256": address,
                        "status": "registered_conditional",
                    })
    if len(rows) != 1600:
        raise AssertionError(f"control count is {len(rows)}, expected 1600")
    addresses = [row["economic_address_sha256"] for row in rows]
    if len(addresses) != len(set(addresses)):
        raise AssertionError("duplicate control economic addresses")
    return rows


def coverage_rows(rows: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["family"] == family]
        for control_id in CONTROL_TYPES[family]:
            selected = [row for row in family_rows if row["control_id"] == control_id]
            yield {
                "family": family,
                "control_id": control_id,
                "rows": len(selected),
                "folds": len({row["fold"] for row in selected}),
                "beam_slots": len({row["parent_slot"] for row in selected}),
                "unique_addresses": len({row["economic_address_sha256"] for row in selected}),
                "status": "pass" if len(selected) == 64 and len({row["economic_address_sha256"] for row in selected}) == 64 else "fail",
            }


def _group_permutation(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], value_field: str, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    output: list[dict[str, Any]] = []
    for group_key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        group = sorted(grouped[group_key], key=lambda row: str(row["event_id"]))
        values = [row[value_field] for row in group]
        group_seed = effective_seed({"base_seed": seed, "group": list(group_key), "value_field": value_field})
        random.Random(group_seed).shuffle(values)
        for row, value in zip(group, values):
            row[value_field] = value
            output.append(row)
    return sorted(output, key=lambda row: str(row["event_id"]))


def apply_control(control_id: str, events: Sequence[Mapping[str, Any]], effective_seed_value: int) -> list[dict[str, Any]]:
    """Apply the registered deterministic transformation to synthetic or materialized parent rows."""
    source = [dict(row) for row in events]
    if control_id == "sign_permutation":
        signed = []
        for row in source:
            value = float(row["signal_scalar"])
            row["signal_sign"] = 0 if value == 0 else 1 if value > 0 else -1
            row["signal_magnitude"] = abs(value)
            signed.append(row)
        permuted = _group_permutation(signed, ("symbol", "year"), "signal_sign", effective_seed_value)
        for row in permuted:
            row["signal_scalar"] = row["signal_sign"] * row["signal_magnitude"]
        return permuted
    if control_id == "context_vector_permutation":
        return _group_permutation(source, ("symbol", "utc_month", "liquidity_decile"), "context_vector", effective_seed_value)
    if control_id == "retest_time_permutation":
        return _group_permutation(source, ("symbol", "utc_quarter", "side"), "add_lag_bars", effective_seed_value)
    if control_id == "matched_pseudo_event":
        allocated: set[str] = set()
        output = []
        for row in sorted(source, key=lambda item: str(item["event_id"])):
            candidates = [candidate for candidate in row.get("pseudo_candidates", []) if str(candidate["candidate_ts"]) not in allocated]
            if not candidates:
                updated = dict(row)
                updated["control_status"] = "unavailable_control"
                output.append(updated)
                continue
            chosen = min(candidates, key=lambda candidate: canonical_hash({"seed": effective_seed_value, "control_id": control_id, "parent_event_id": row["event_id"], "candidate_ts": candidate["candidate_ts"]}))
            allocated.add(str(chosen["candidate_ts"]))
            updated = dict(row)
            updated["pseudo_event"] = chosen
            updated["control_status"] = "complete"
            output.append(updated)
        return output
    field_nulls = {
        "signed_return_only": {"signal_estimator": "signed_return"},
        "no_smoothness": {"smoothness_component": None},
        "unscaled_exposure": {"exposure_multiplier": 1.0},
        "context_null": {"context_multiplier": 1.0},
        "price_only_impulse": {"contraction_component": None, "smoothness_component": None},
        "no_contraction": {"contraction_component": None},
        "parent_only": {"context_multiplier": 1.0, "context_vector": None},
        "no_prior_level": {"prior_level_component": None},
        "no_RS": {"relative_strength_component": None},
        "no_macro_breadth_dispersion": {"BTC_ETH_component": None, "breadth_component": None, "dispersion_component": None},
        "starter_only": {"add_fraction": 0.0},
        "price_only_breakout": {"retest_component": None, "context_multiplier": 1.0},
        "zero_add": {"add_fraction": 0.0},
        "price_only": {"open_interest_component": None, "liquidation_component": None, "derivatives_state_predicate": None},
        "OI_removed": {"open_interest_component": None},
        "liquidation_removed": {"liquidation_component": None},
        "generic_structure_control": {"derivatives_state_predicate": False},
    }
    if control_id not in field_nulls:
        raise ValueError(f"unsupported registered control: {control_id}")
    return [{**row, **field_nulls[control_id]} for row in sorted(source, key=lambda item: str(item["event_id"]))]
