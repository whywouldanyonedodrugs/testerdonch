from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..canonical import canonical_hash
from .common import EngineInputError, log_return, path_smoothness, sample_standard_deviation


ENGINE_ID = "a1_compression_engine_v1"


def realized_volatility(closes: Sequence[float]) -> float:
    returns = [log_return(left, right) for left, right in zip(closes, closes[1:])]
    return sample_standard_deviation(returns)


def features(impulse_closes: Sequence[float], base_closes: Sequence[float], baseline_closes: Sequence[float], side: int) -> dict[str, float]:
    if side not in (-1, 1):
        raise EngineInputError("A1 side must be long or short")
    if len(base_closes) != len(baseline_closes):
        raise EngineInputError("contraction baseline must match base duration")
    baseline_vol = realized_volatility(baseline_closes)
    if baseline_vol <= 0:
        raise EngineInputError("collapsed contraction baseline")
    return {
        "side_signed_impulse": side * log_return(impulse_closes[0], impulse_closes[-1]),
        "contraction_ratio": realized_volatility(base_closes) / baseline_vol,
        "base_smoothness": path_smoothness(base_closes),
    }


def confirmation_pass(closes: Sequence[float], frozen_extreme: float, side: int, confirmation: str) -> bool:
    required = 1 if confirmation == "one_close" else 2
    if confirmation == "close_plus_bounded_15m_delay":
        required = 2
    if len(closes) < required:
        return False
    selected = closes[-required:]
    return all(value > frozen_extreme for value in selected) if side == 1 else all(value < frozen_extreme for value in selected)


def event_id(symbol: str, side: int, impulse_start: str, impulse_end: str, base_start: str, base_end: str, confirmation_ts: str) -> str:
    return canonical_hash({
        "event_type": "impulse_base_confirmation_episode",
        "symbol": symbol,
        "side": side,
        "impulse_start": impulse_start,
        "impulse_end": impulse_end,
        "base_start": base_start,
        "base_end": base_end,
        "confirmation_ts": confirmation_ts,
    })


def contract() -> dict[str, Any]:
    return {
        "engine_id": ENGINE_ID,
        "event_type": "impulse_base_confirmation_episode",
        "event_identity": "SHA256(family,symbol,side,impulse_start,impulse_end,base_start,base_end,confirmation_ts)",
        "features": ["side_signed_impulse", "contraction_ratio", "base_path_smoothness"],
        "side_grammar": "long, short, or one definition evaluating both sides separately",
        "entry": "first qualifying confirmation completed close; next authorized trade open",
        "exit": "registered time/base-failure/ATR stop/trail/target; completed-close trigger and next-open fill",
        "non_overlap": "definition-symbol chronological acceptance using actual executable exit",
        "accounting": "one episode is one observation; equal-event then equal-market-day aggregation",
        "threshold_populations": "impulse and shape populations are explicit config axes",
        "controls": ["matched_pseudo_event", "price_only_impulse", "no_contraction", "no_smoothness", "context_null"],
        "stress_tests": ["32bps", "entry_delay_15m", "strict_PIT_membership"],
    }


__all__ = ["confirmation_pass", "contract", "event_id", "features"]
