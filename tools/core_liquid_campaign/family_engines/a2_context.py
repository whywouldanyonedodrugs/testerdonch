from __future__ import annotations

from typing import Any, Mapping

from ..canonical import canonical_hash
from .common import EngineInputError, component_threshold


ENGINE_ID = "a2_context_engine_v1"


def component_vector(config: Mapping[str, Any], raw: Mapping[str, float]) -> list[tuple[str, float]]:
    components: list[tuple[str, float]] = []
    proximity = component_threshold(float(raw.get("proximity", 0.0)), str(config["proximity_rank"]))
    if proximity is not None:
        components.append(("proximity", proximity))
    rs = component_threshold(float(raw.get("RS", 0.0)), str(config["RS_rank"]))
    if rs is not None:
        components.append(("RS", rs))
    reclaim = config["reclaim_state"]
    if reclaim != "none":
        value = float(raw.get("reclaim", 0.0))
        components.append(("reclaim", value if reclaim == "continuous_distance" else (1.0 if value >= 0.5 else 0.0)))
    btc_eth = config["BTC_ETH_context"]
    if btc_eth != "none":
        components.append((f"BTC_ETH_{btc_eth}", float(raw["BTC_ETH"])))
    breadth = config["breadth_dispersion"]
    if breadth in ("breadth", "both"):
        components.append(("breadth", float(raw["breadth"])))
    if breadth in ("dispersion", "both"):
        components.append(("dispersion", 1.0 - float(raw["dispersion"])))
    if not components or any(not 0.0 <= value <= 1.0 for _, value in components):
        raise EngineInputError("A2 component vector is empty or outside [0,1]")
    return components


def overlay_multiplier(action: str, components: list[tuple[str, float]]) -> float:
    values = [value for _, value in components]
    if action == "parent_only":
        return 1.0
    if action == "permission":
        return 1.0 if all(value >= 0.5 for value in values) else 0.0
    if action == "linear_size_0_to_1":
        result = 1.0
        for value in values:
            result *= max(0.0, min(1.0, 2.0 * value - 1.0))
        return result
    if action == "tercile_size":
        minimum = min(values)
        return 0.0 if minimum < 1.0 / 3.0 else 0.5 if minimum < 2.0 / 3.0 else 1.0
    raise EngineInputError(f"unknown A2 overlay action: {action}")


def parent_slot_id(parent_family: str, fold: str, beam_rank: int) -> str:
    return f"{parent_family}:{fold}:beam:{beam_rank:02d}"


def counterpart_ids(config_hash: str, parent_identity: str) -> tuple[str, str]:
    base = {"config_hash": config_hash, "parent_identity": parent_identity}
    return canonical_hash({**base, "role": "parent_only"}), canonical_hash({**base, "role": "context_overlay"})


def paired_uplift(parent_net: float, overlay_net: float, parent_event_id: str, overlay_event_id: str, exposure_equal: bool, occupancy_equal: bool) -> float:
    if parent_event_id != overlay_event_id or not exposure_equal or not occupancy_equal:
        raise EngineInputError("A2 paired uplift requires identical event identity, exposure and occupancy")
    return overlay_net - parent_net


def contract() -> dict[str, Any]:
    return {
        "engine_id": ENGINE_ID,
        "event_type": "parent_bound_context_overlay",
        "event_identity": "identical to exact parent event; overlay and parent-only counterparts are frozen atomically",
        "parent_grammar": "exact source parent for retained legacy rows or exact family/outer-fold/beam-rank slot for new templates",
        "missing_parent": "unavailable_no_parent; never reassigned",
        "features": ["prior-level proximity/reclaim", "BTC-relative strength", "BTC/ETH context", "PIT breadth", "PIT dispersion"],
        "side_grammar": "inherits exact parent side",
        "entry_exit": "inherits exact parent timestamps and fills",
        "accounting": "event IDs, gross move, costs, funding and occupancy are identical except registered permission/size multiplier",
        "controls": ["context_vector_permutation", "parent_only", "no_prior_level", "no_RS", "no_macro_breadth_dispersion"],
        "promotion": "registered paired uplift and main context-permutation null",
    }


__all__ = ["component_vector", "contract", "counterpart_ids", "overlay_multiplier", "paired_uplift", "parent_slot_id"]
