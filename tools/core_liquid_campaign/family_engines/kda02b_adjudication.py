from __future__ import annotations

from typing import Any, Mapping

from .common import EngineInputError


ENGINE_ID = "kda02b_adjudication_engine_v1"


def apply_variant(features: Mapping[str, Any], variant: str) -> dict[str, Any]:
    result = dict(features)
    if variant == "identity_replay":
        return result
    if variant == "price_only":
        return {key: value for key, value in result.items() if key.startswith("price_") or key in {"event_id", "side"}}
    if variant == "OI_removed":
        result["open_interest_component"] = None
        return result
    if variant == "liquidation_removed":
        result["liquidation_component"] = None
        return result
    if variant == "generic_structure_control":
        result["derivatives_state_predicate"] = False
        return result
    if variant == "stress_cost_32bps":
        result["round_trip_cost_bps"] = 32.0
        return result
    if variant in {"funding_start_alignment", "funding_end_alignment", "funding_zero", "entry_delay_15m", "entry_delay_60m"}:
        result["adjudication_override"] = variant
        return result
    raise EngineInputError(f"unsupported KDA02B adjudication variant: {variant}")


def contract() -> dict[str, Any]:
    return {
        "engine_id": ENGINE_ID,
        "event_type": "exact_stage20_cell_adjudication",
        "event_identity": "preserved Stage20 cell/event identity plus immutable adjudication variant",
        "side_grammar": "inherits exact Stage20 cell side",
        "entry_exit": "inherits exact Stage20 event unless the registered delay variant changes entry",
        "accounting": "separate base/stress cost and funding alignment variants; no generic platform mechanics",
        "non_overlap": "inherits exact accepted Stage20 economic episodes",
        "controls": ["matched_pseudo_event", "price_only", "OI_removed", "liquidation_removed", "generic_structure_control"],
        "stress_tests": ["32bps", "funding_start_alignment", "funding_end_alignment", "funding_zero", "entry_delay_15m", "entry_delay_60m"],
    }


__all__ = ["apply_variant", "contract"]
