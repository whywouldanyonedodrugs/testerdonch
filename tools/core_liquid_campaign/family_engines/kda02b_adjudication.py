from __future__ import annotations

import itertools
import math
from datetime import timedelta
from typing import Any, Mapping

from ..canonical import canonical_hash
from ..engine_types import FamilyInput
from .common import EngineInputError, require_utc


ENGINE_ID = "kda02b_adjudication_engine_v2"
AXIS_LEVELS = {
    "oi_axis": ("raw_oi_log_change", "fold_local_percentile"),
    "price_axis": ("raw_bps", "fold_local_rank"),
    "price_state": ("negative", "positive"),
    "branch": ("continuation", "reversal"),
    "horizon": ("1h", "3h", "6h"),
    "liquidation_context": ("continuous_intensity", "present_absent"),
}


def cell_contract(cell_id: str) -> dict[str, Any]:
    try:
        number = int(cell_id.removeprefix("KDA02B_"))
    except ValueError as exc:
        raise EngineInputError("invalid KDA02B cell ID") from exc
    combinations = tuple(itertools.product(*AXIS_LEVELS.values()))
    if not 1 <= number <= len(combinations) or cell_id != f"KDA02B_{number:03d}":
        raise EngineInputError("KDA02B cell is outside the frozen 96-cell grammar")
    axes = dict(zip(AXIS_LEVELS, combinations[number - 1]))
    return {
        "cell_id": cell_id,
        "family": "KDA02B",
        "axes": axes,
        "price_window": "completed_1h_ending_at_decision_source_close",
        "oi_raw_range": [-0.12, -0.01],
        "raw_price_absolute_bps_range": [14.0, 500.0],
        "rank_rules": {"oi": "q0_to_q20", "trade_abs": "q80_to_q100", "mark_abs": "q80_to_q100", "liquidation": "q80_to_q100"},
        "episode": "first false-to-true all-mandatory completed-bar onset after rearm",
        "decision_availability_lag_minutes": 5,
    }


def cell_contract_sha256(cell_id: str) -> str:
    return canonical_hash(cell_contract(cell_id))


def apply_variant(features: Mapping[str, Any], variant: str) -> dict[str, Any]:
    result = dict(features)
    if variant == "identity_replay":
        return result
    if variant == "price_only":
        return {"price": result["price"]}
    if variant == "OI_removed":
        result.pop("oi", None); return result
    if variant == "liquidation_removed":
        result.pop("liquidation", None); return result
    if variant == "generic_structure_control":
        return {"generic_structure": bool(result.get("generic_structure", False))}
    if variant in {"stress_cost_32bps", "funding_start_alignment", "funding_end_alignment", "funding_zero", "entry_delay_15m", "entry_delay_60m"}:
        return result
    raise EngineInputError(f"unsupported KDA02B adjudication variant: {variant}")


def _finite_number(source: Mapping[str, Any], key: str) -> float:
    value = source.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise EngineInputError(f"KDA02B raw feature is absent or nonfinite: {key}")
    return float(value)


def _between(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper


def _predicates(source: Mapping[str, Any], axes: Mapping[str, str], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    trade_bps = _finite_number(source, "trade_return_1h") * 10000.0
    mark_bps = _finite_number(source, "mark_return_1h") * 10000.0
    sign = -1.0 if axes["price_state"] == "negative" else 1.0
    price = trade_bps * sign > 0 and mark_bps * sign > 0
    if axes["price_axis"] == "raw_bps":
        price = price and _between(abs(trade_bps), 14.0, 500.0) and _between(abs(mark_bps), 14.0, 500.0)
    else:
        price = price and _between(abs(trade_bps), _finite_number(thresholds, "trade_abs_q80"), _finite_number(thresholds, "trade_abs_q100")) and _between(abs(mark_bps), _finite_number(thresholds, "mark_abs_q80"), _finite_number(thresholds, "mark_abs_q100"))
    oi_value = _finite_number(source, "oi_log_change_1h")
    oi = _between(oi_value, -0.12, -0.01) if axes["oi_axis"] == "raw_oi_log_change" else _between(oi_value, _finite_number(thresholds, "oi_q0"), _finite_number(thresholds, "oi_q20"))
    if axes["liquidation_context"] == "present_absent":
        liquidation = _finite_number(source, "liquidation_base_units_1h") > 0
    else:
        if source.get("liquidation_normalization_valid") is not True:
            liquidation = False
        else:
            value = _finite_number(source, "liquidation_intensity_robust_z")
            liquidation = _between(value, _finite_number(thresholds, "liquidation_q80"), _finite_number(thresholds, "liquidation_q100"))
    return {"price": price, "oi": oi, "liquidation": liquidation}


def _raw_feature_history(frame: FamilyInput) -> tuple[Mapping[str, Any], ...]:
    raw = frame.metadata.get("kda02b_feature_history")
    if not isinstance(raw, (tuple, list)) or len(raw) < 2 or any(not isinstance(item, Mapping) for item in raw):
        raise EngineInputError("KDA02B sequential raw feature history is absent")
    history = tuple(raw)
    decision = require_utc(frame.decision_ts)
    previous_close = None
    for source in history:
        source_close = source.get("source_close_ts"); available = source.get("feature_available_ts")
        if not hasattr(source_close, "tzinfo") or not hasattr(available, "tzinfo"):
            raise EngineInputError("KDA02B feature timestamps are absent")
        if require_utc(source_close) > decision or require_utc(available) > decision:
            raise EngineInputError("KDA02B derivative feature is not point-in-time available")
        if previous_close is not None and require_utc(source_close) <= previous_close:
            raise EngineInputError("KDA02B feature history is not strictly sorted")
        previous_close = require_utc(source_close)
    if require_utc(history[-1]["feature_available_ts"]) != decision:
        raise EngineInputError("KDA02B final feature row does not bind the exact decision")
    return history


def _is_valid(source: Mapping[str, Any]) -> bool:
    return all(source.get(key) is True for key in ("eligible", "known_lifecycle_mask", "trade_coverage", "mark_coverage", "analytics_coverage"))


def onset_indices(history: tuple[Mapping[str, Any], ...], axes: Mapping[str, str], thresholds: Mapping[str, Any], variant: str) -> list[int]:
    """Port of Stage-20: rearm only on all-false; gaps/invalid rows are barriers."""
    onsets: list[int] = []
    armed = False
    previous_close = None
    for index, source in enumerate(history):
        close = require_utc(source["source_close_ts"])
        gap = previous_close is None or close - previous_close != timedelta(minutes=5)
        previous_close = close
        if gap or not _is_valid(source):
            armed = False
            if not _is_valid(source):
                continue
        predicates = apply_variant(_predicates(source, axes, thresholds), variant)
        all_true = all(predicates.values())
        all_false = not any(predicates.values())
        if all_false:
            armed = True
        elif all_true and armed:
            onsets.append(index); armed = False
    return onsets


def evaluate(frame: FamilyInput, config: Mapping[str, Any], *, control_id: str | None = None, control_directive: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Compute the exact frozen KDA02B predicates from bound raw derivative features."""
    frame.validate()
    cell_id = str(config["stage20_cell_id"])
    contract = cell_contract(cell_id)
    expected_contract = cell_contract_sha256(cell_id)
    if frame.metadata.get("stage20_cell_contract_sha256") != expected_contract:
        raise EngineInputError("KDA02B cell contract hash does not match the code-owned frozen grammar")
    history = _raw_feature_history(frame)
    thresholds = frame.metadata.get("fold_thresholds")
    if not isinstance(thresholds, Mapping):
        raise EngineInputError("KDA02B fold-local threshold model is absent")
    axes = contract["axes"]
    variant = str(control_id or config["adjudication_variant"])
    if variant == "generic_structure_control":
        if not isinstance(control_directive, Mapping) or control_directive.get("allocator") != "matched_pseudo_event_allocator_v2" or control_directive.get("matched_decision_ts") != require_utc(frame.decision_ts):
            raise EngineInputError("KDA02B generic structure requires a bound matched-pseudo allocator directive")
        current_predicates = {"matched_pseudo_event": True}
    else:
        onsets = onset_indices(history, axes, thresholds, variant)
        if not onsets or onsets[-1] != len(history) - 1:
            return []
        current_predicates = apply_variant(_predicates(history[-1], axes, thresholds), variant)
    if not all(current_predicates.values()):
        return []
    sign = -1 if axes["price_state"] == "negative" else 1
    side = sign if axes["branch"] == "continuation" else -sign
    trigger = require_utc(frame.decision_ts)
    delay = timedelta(minutes=15 if variant == "entry_delay_15m" else 60 if variant == "entry_delay_60m" else 0)
    target = trigger + delay
    entry_index = next((index for index, bar in enumerate(frame.five_minute_bars) if require_utc(bar.open_ts) >= target), None)
    if entry_index is None or require_utc(frame.five_minute_bars[entry_index].open_ts) - target > timedelta(minutes=10):
        return []
    alignment = "start_inclusive_end_exclusive" if variant == "funding_start_alignment" else "start_exclusive_end_inclusive" if variant == "funding_end_alignment" else "minimum_of_registered_start_end"
    return [{
        "event_id": canonical_hash({"cell_contract_sha256": expected_contract, "symbol": frame.symbol, "decision_ts": trigger.isoformat(), "variant": variant}),
        "side": side,
        "decision_ts": trigger,
        "entry_index": entry_index,
        "exit": f"time_{axes['horizon']}",
        "cost_bps": 32.0 if variant == "stress_cost_32bps" else 14.0,
        "funding_alignment": alignment,
        "funding_zero": variant == "funding_zero",
        "context_multiplier": 1.0,
        "stage20_cell_contract_sha256": expected_contract,
        "predicate_components": tuple(sorted(current_predicates)),
    }]


def contract() -> dict[str, Any]:
    return {
        "engine_id": ENGINE_ID,
        "event_type": "exact_stage20_cell_adjudication",
        "event_identity": "SHA256(cell-contract,symbol,decision_ts,adjudication-variant)",
        "features": "raw completed 1h trade/mark return, OI log change, liquidation base units/intensity, exact fold-local thresholds; current and previous rows are evaluated in code",
        "side_grammar": "agreed trade/mark sign; continuation follows and reversal opposes",
        "entry_exit": "decision plus registered 0/15/60-minute delay to next authorized open; exact cell horizon 1h/3h/6h to next open",
        "accounting": "base/stress costs; conservative minimum of the two registered funding alignments unless a specific alignment variant is under adjudication",
        "non_overlap": "actual executable exit timestamp",
        "controls": ["price_only", "OI_removed", "liquidation_removed", "generic_structure_control"],
        "stress_tests": ["32bps", "funding_start_alignment", "funding_end_alignment", "funding_zero", "entry_delay_15m", "entry_delay_60m"],
    }


__all__ = ["apply_variant", "cell_contract", "cell_contract_sha256", "contract", "evaluate"]
