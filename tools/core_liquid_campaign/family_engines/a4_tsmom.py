from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..canonical import canonical_hash
from .common import EngineInputError, close_to_close_volatility, ema, log_return, parkinson_volatility, path_smoothness


ENGINE_ID = "a4_tsmom_engine_v1"


def volatility(config: Mapping[str, Any], closes: Sequence[float], highs: Sequence[float] | None = None, lows: Sequence[float] | None = None) -> float:
    if config["volatility_estimator"] == "close_to_close":
        return close_to_close_volatility(closes)
    if highs is None or lows is None:
        raise EngineInputError("Parkinson estimator requires completed highs and lows")
    return parkinson_volatility(highs, lows)


def signal_scalar(
    config: Mapping[str, Any],
    closes: Sequence[float],
    *,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    prior_high: float | None = None,
    prior_low: float | None = None,
    atr: float | None = None,
    ensemble_components: Sequence[float] | None = None,
) -> float:
    estimator = config["signal_estimator"]
    vol = volatility(config, closes, highs, lows)
    if not math.isfinite(vol) or vol <= 0:
        raise EngineInputError("volatility estimator is unavailable")
    if estimator == "signed_return":
        return log_return(closes[0], closes[-1]) / vol
    if estimator == "ema_slope":
        series = ema(closes, max(2, len(closes) // 2))
        reference = series[max(0, len(series) - 289)]
        return log_return(reference, series[-1]) / vol
    if estimator == "breakout_distance_rank":
        if prior_high is None or prior_low is None or atr is None or atr <= 0:
            raise EngineInputError("breakout-distance estimator requires prior high, prior low and ATR")
        close = closes[-1]
        return max((close - prior_high) / atr, 0.0) + min((close - prior_low) / atr, 0.0)
    if estimator == "equal_weight_ensemble":
        if ensemble_components is None or len(ensemble_components) != 3:
            raise EngineInputError("ensemble requires three pre-winsorized, MAD-scaled components")
        if any(not math.isfinite(float(value)) for value in ensemble_components):
            raise EngineInputError("ensemble component unavailable")
        return sum(float(value) for value in ensemble_components) / 3.0
    raise EngineInputError(f"unknown estimator: {estimator}")


def side_from_scalar(scalar: float, direction: str) -> int:
    if scalar == 0:
        return 0
    if direction == "long_short":
        return 1 if scalar > 0 else -1
    if direction == "long_flat":
        return 1 if scalar > 0 else 0
    if direction == "short_flat":
        return -1 if scalar < 0 else 0
    raise EngineInputError(f"unknown direction: {direction}")


def event_id(symbol: str, decision_ts: str, config_hash: str, side: int) -> str:
    return canonical_hash({"event_type": "definition_symbol_episode", "symbol": symbol, "decision_ts": decision_ts, "config_hash": config_hash, "side": side})


def contract() -> dict[str, Any]:
    return {
        "engine_id": ENGINE_ID,
        "event_type": "definition_symbol_episode",
        "event_identity": "SHA256(event_type,symbol,decision_ts,config_hash,side)",
        "features": ["signed_return", "ema_slope", "breakout_distance_rank", "equal_weight_ensemble", "path_smoothness", "close_to_close_volatility", "parkinson_volatility"],
        "side_grammar": "strict scalar sign; exact zero is flat",
        "entry": "registered UTC rebalance decision; next authorized trade open",
        "exit": "registered time, signal reversal, or completed-close ATR trail; next authorized trade open",
        "non_overlap": "definition-symbol chronological acceptance using actual executable exit",
        "accounting": "cohort equal-weight with frozen volatility/context exposure; per-episode costs and exact funding",
        "threshold_populations": "fold-local registered scope; no scope fallback",
        "aggregate_metrics": "cohort-derived UTC-day means with empty cohorts omitted as no opportunity",
        "controls": ["sign_permutation", "signed_return_only", "no_smoothness", "unscaled_exposure", "context_null"],
        "stress_tests": ["32bps", "entry_delay_15m", "funding_boundary_alignments"],
        "removed_axis": "signal_rank_scope",
        "removed_axis_reason": "no source-defined economic consumer",
    }


__all__ = ["contract", "event_id", "path_smoothness", "side_from_scalar", "signal_scalar", "volatility"]
