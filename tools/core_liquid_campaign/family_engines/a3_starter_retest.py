from __future__ import annotations

from typing import Any

from ..canonical import canonical_hash
from .common import EngineInputError


ENGINE_ID = "a3_starter_retest_engine_v1"


def breakout_magnitude(close: float, level: float, atr: float, direction: str) -> float:
    if atr <= 0:
        raise EngineInputError("A3 breakout ATR must be positive")
    if direction == "long":
        return (close - level) / atr
    if direction == "short":
        return (level - close) / atr
    raise EngineInputError("A3 direction must be long or short")


def retest_state(direction: str, level: float, atr: float, depth: float, previous_close: float, close: float) -> str:
    lower, upper = level - depth * atr, level + depth * atr
    if direction == "long":
        if close < lower:
            return "invalidated"
        if previous_close > upper and lower <= close <= upper and close < previous_close:
            return "activated"
        if close > level and close > previous_close:
            return "reclaimed"
    elif direction == "short":
        if close > upper:
            return "invalidated"
        if previous_close < lower and lower <= close <= upper and close > previous_close:
            return "activated"
        if close < level and close < previous_close:
            return "reclaimed"
    else:
        raise EngineInputError("A3 direction must be long or short")
    return "waiting"


def event_id(symbol: str, direction: str, crossing_ts: str, frozen_level: float, config_hash: str) -> str:
    return canonical_hash({"event_type": "directional_starter_retest_parent", "symbol": symbol, "direction": direction, "crossing_ts": crossing_ts, "frozen_level": frozen_level, "config_hash": config_hash})


def parent_weights(starter_fraction: float, add_fraction: float, starter_net: float, add_net: float | None) -> float:
    if starter_fraction + add_fraction > 1.0 + 1e-12:
        raise EngineInputError("A3 parent exposure exceeds one")
    if add_net is None:
        add_fraction = 0.0
    return starter_fraction * starter_net + add_fraction * (add_net or 0.0)


def contract() -> dict[str, Any]:
    return {
        "engine_id": ENGINE_ID,
        "event_type": "directional_starter_retest_parent",
        "event_identity": "SHA256(symbol,direction,crossing_ts,frozen_level,config_hash)",
        "side_grammar": "long and short are separate executable identities; symmetric portfolio is derived report only",
        "entry": "starter at next authorized open after final confirmation; optional add at next authorized open after first valid reclaim",
        "exit": "each leg applies the registered exit independently; parent exit is max actual leg exit",
        "non_overlap": "one parent observation; next parent only when its entry is at or after prior actual parent exit",
        "accounting": "starter_fraction*starter_net + add_fraction*add_net; absent add has zero exposure and is not a trade",
        "threshold_populations": "breakout magnitude uses explicit symbol/liquidity-decile/global side scope",
        "controls": ["retest_time_permutation", "starter_only", "price_only_breakout", "zero_add", "context_null"],
        "stress_tests": ["32bps", "15m delay", "no-retest component"],
    }


__all__ = ["breakout_magnitude", "contract", "event_id", "parent_weights", "retest_state"]
