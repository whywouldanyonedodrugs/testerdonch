from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .canonical import canonical_hash, canonical_json_bytes
from .engine_types import FamilyInput
from .family_engines import a4_tsmom, kda02b_adjudication
from .family_engines.common import require_utc
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


def _frame_index(frames: Sequence[FamilyInput]) -> dict[tuple[str, str], FamilyInput]:
    result = {(frame.symbol, frame.decision_ts.isoformat()): frame for frame in frames}
    if len(result) != len(frames):
        raise ValueError("control input has duplicate symbol/decision frames")
    return result


def _liquidity_decile(frame: FamilyInput) -> int:
    snapshot = frame.metadata.get("pit_universe_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("control frame lacks a verified PIT universe snapshot")
    deciles = snapshot.get("lagged_liquidity_deciles")
    if not isinstance(deciles, Mapping) or frame.symbol not in deciles or not 1 <= int(deciles[frame.symbol]) <= 10:
        raise ValueError("control frame lacks a bound lagged-liquidity decile")
    return int(deciles[frame.symbol])


def _pcg_permute(values: Sequence[Any], seed_payload: Mapping[str, Any]) -> list[Any]:
    if not values:
        return []
    order = np.random.Generator(np.random.PCG64(effective_seed(seed_payload))).permutation(len(values))
    return [values[int(index)] for index in order]


def maximum_holding_for_parent(parent_row: Mapping[str, Any]) -> timedelta:
    config = parent_row["config"]
    if parent_row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1":
        horizon = kda02b_adjudication.cell_contract(str(config["stage20_cell_id"]))["axes"]["horizon"]
        return {"1h": timedelta(hours=1), "3h": timedelta(hours=3), "6h": timedelta(hours=6)}[horizon]
    exit_name = str(config.get("exit", ""))
    if exit_name.startswith("time_"):
        amount = int(exit_name.split("_")[1][:-1])
        return timedelta(hours=amount) if exit_name.endswith("h") else timedelta(days=amount)
    # Structural, signal-reversal, stop and trail exits share the engine's
    # frozen ten-day maximum evaluation horizon.
    return timedelta(days=10)


def _hashable(value: Any) -> Any:
    if isinstance(value, datetime):
        return require_utc(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _hashable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_hashable(item) for item in value]
    return value


def matched_pseudo_event_directives(
    parent_observations: Sequence[Any],
    candidate_frames: Sequence[FamilyInput],
    *,
    parent_sides: Mapping[str, int],
    control_id: str,
    seed: int,
    control_address: str,
    maximum_holding: timedelta,
) -> tuple[list[FamilyInput], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    """Exact side/hour/lagged-liquidity matched allocator with explicit misses."""
    frames = _frame_index(candidate_frames)
    allocated_decisions: set[str] = set()
    selected: list[FamilyInput] = []
    directives: dict[str, Mapping[str, Any]] = {}
    unavailable: list[dict[str, Any]] = []
    for parent in sorted(parent_observations, key=lambda item: item.event_id):
        parent_frame = frames.get((parent.symbol, parent.decision_ts.isoformat()))
        if parent_frame is None:
            unavailable.append({"parent_event_id": parent.event_id, "status": "unavailable_parent_frame"}); continue
        side = parent_sides.get(parent.event_id)
        if side not in (-1, 1):
            raise ValueError("matched pseudo allocator lacks the exact parent side")
        month = parent.decision_ts.strftime("%Y-%m")
        stratum = (parent.symbol, month, parent.decision_ts.weekday() * 24 + parent.decision_ts.hour, _liquidity_decile(parent_frame), side)
        candidates = []
        for frame in candidate_frames:
            decision_key = frame.decision_ts.isoformat()
            if decision_key in allocated_decisions or frame.symbol != parent.symbol:
                continue
            if frame.decision_ts.strftime("%Y-%m") != month or frame.decision_ts.weekday() * 24 + frame.decision_ts.hour != stratum[2] or _liquidity_decile(frame) != stratum[3]:
                continue
            if parent.decision_ts - maximum_holding <= frame.decision_ts <= parent.decision_ts + maximum_holding:
                continue
            candidates.append(frame)
        if not candidates:
            unavailable.append({"parent_event_id": parent.event_id, "status": "unavailable_no_matched_pseudo_event", "stratum": list(stratum)}); continue
        chosen = min(candidates, key=lambda frame: (canonical_hash({"seed": seed, "symbol": frame.symbol, "month": month, "side": side, "hour_of_week": stratum[2], "liquidity_decile": stratum[3], "decision_ts": frame.decision_ts.isoformat()}), frame.decision_ts))
        allocated_decisions.add(chosen.decision_ts.isoformat())
        selected.append(chosen)
        directives[chosen.content_sha256()] = {"allocator": "matched_pseudo_event_allocator_v2", "parent_event_id": parent.event_id, "side": side, "matched_decision_ts": chosen.decision_ts}
    return selected, directives, unavailable


def derive_control_inputs(
    control_row: Mapping[str, Any],
    parent_row: Mapping[str, Any],
    parent_result: Mapping[str, Any],
    frames: Sequence[FamilyInput],
) -> tuple[list[FamilyInput], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    control_id = str(control_row["control_id"]); seed = int(control_row["effective_seed"])
    frame_index = _frame_index(frames); observations = list(parent_result["observations"])
    ledger_by_event = {row["event_id"]: row for row in parent_result["ledger"] if row.get("status") == "complete"}
    directives: dict[str, Mapping[str, Any]] = {}
    if control_id in {"A1_MATCHED_PSEUDO_EVENT_MAIN_NULL", "A3_MATCHED_PSEUDO_EVENT"}:
        parent_sides = {event_id: int(row.get("engine_event", {}).get("side", 0)) for event_id, row in ledger_by_event.items()}
        selected, directives, unavailable = matched_pseudo_event_directives(observations, frames, parent_sides=parent_sides, control_id=control_id, seed=20260721, control_address=str(control_row["economic_address_sha256"]), maximum_holding=maximum_holding_for_parent(parent_row))
        for frame in selected:
            directive = dict(directives[frame.content_sha256()])
            parent_ledger = ledger_by_event.get(str(directive["parent_event_id"]))
            if parent_ledger is None:
                raise ValueError("matched pseudo allocator lost its exact parent ledger")
            side = parent_ledger.get("engine_event", {}).get("side")
            if side not in (-1, 1):
                raise ValueError("matched pseudo allocator parent side is unavailable")
            directives[frame.content_sha256()] = {**directive, "side": side}
        return selected, directives, unavailable
    if control_id == "A4_SIGN_PERMUTED_MAIN_NULL":
        grouped_frames: dict[tuple[str, int], list[FamilyInput]] = {}
        for frame in frames:
            grouped_frames.setdefault((frame.symbol, frame.decision_ts.year), []).append(frame)
        for group, group_frames in sorted(grouped_frames.items()):
            ordered_frames = sorted(group_frames, key=lambda frame: frame.decision_ts)
            signs = [a4_tsmom.scalar_estimator_sign(frame, parent_row["config"]) for frame in ordered_frames]
            values = _pcg_permute(signs, {"base_seed": 20260721, "control_id": control_id, "symbol": group[0], "year": group[1]})
            for frame, sign in zip(ordered_frames, values):
                directives[frame.content_sha256()] = {"signal_sign": sign, "allocator": "registered_PCG64_complete_sign_vector_v1"}
        return list(frames), directives, []

    grouped: dict[tuple[Any, ...], list[tuple[Any, FamilyInput, Any]]] = {}
    for observation in observations:
        frame = frame_index.get((observation.symbol, observation.decision_ts.isoformat()))
        ledger = ledger_by_event.get(observation.event_id)
        if frame is None or ledger is None:
            raise ValueError("control derivation cannot reconcile parent event/frame/ledger")
        if control_id == "A2_CONTEXT_PERMUTED_MAIN_NULL":
            value = tuple(tuple(item) for item in ledger["context_components"]); group = (observation.symbol, observation.decision_ts.strftime("%Y-%m"), _liquidity_decile(frame))
        elif control_id == "A3_RETEST_TIME_PERMUTED_MAIN_NULL":
            retest = ledger.get("retest") or {}; add_index = retest.get("add_entry_index"); starter_index = ledger["engine_event"]["entry_index"]
            value = "NO_ADD" if add_index is None else int(add_index) - int(starter_index)
            group = (observation.symbol, f"{observation.decision_ts.year}Q{(observation.decision_ts.month - 1) // 3 + 1}", ledger["engine_event"]["side"])
        else:
            continue
        grouped.setdefault(group, []).append((observation, frame, value))
    for group, rows in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        ordered = sorted(rows, key=lambda row: row[0].event_id)
        values = _pcg_permute([row[2] for row in ordered], {"base_seed": 20260721, "control_id": control_id, "group": list(group)})
        for (observation, frame, _), value in zip(ordered, values):
            field = "signal_sign" if control_id.startswith("A4_") else "context_vector" if control_id.startswith("A2_") else "add_lag_bars"
            directives[frame.content_sha256()] = {field: value, "parent_event_id": observation.event_id, "allocator": "registered_PCG64_without_replacement_v2"}
    return list(frames), directives, []


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
    from .executor import dispatch_registered_attempt
    parent_result = dispatch_registered_attempt(parent_row, frames, registry_by_id=registry_by_id, parent_binding=parent_binding, parent_frames=parent_frames)
    transformed, directives, unavailable = derive_control_inputs(control_row, parent_row, parent_result, frames)
    result = dispatch_registered_attempt(
        parent_row,
        transformed,
        registry_by_id=registry_by_id,
        parent_binding=parent_binding,
        parent_frames=parent_frames,
        control_id=str(control_row["control_id"]),
        control_directives=directives,
    )
    result["control_attempt_id"] = control_row["control_attempt_id"]
    result["control_economic_address_sha256"] = control_row["economic_address_sha256"]
    result["allocation_unavailable"] = unavailable
    result["exact_parent_result_sha256"] = canonical_hash(_hashable({"observations": [item.event_id for item in parent_result["observations"]], "ledger": parent_result["ledger"]}))
    return result


__all__ = ["CONTROL_IDS", "compile_controls", "coverage_rows", "derive_control_inputs", "effective_seed", "execute_control", "matched_pseudo_event_directives", "maximum_holding_for_parent"]
