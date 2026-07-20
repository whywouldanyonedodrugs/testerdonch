from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

from ..canonical import canonical_hash
from ..engine_types import FamilyInput
from .common import EngineInputError, ema, log_return, path_smoothness, percentile_from_population, require_utc, sample_standard_deviation, type7_quantile, wilder_atr


ENGINE_ID = "a4_tsmom_engine_v1"


def volatility(config: Mapping[str, Any], closes: Sequence[float], highs: Sequence[float] | None = None, lows: Sequence[float] | None = None) -> float:
    if config["volatility_estimator"] == "close_to_close":
        returns = [log_return(left, right) for left, right in zip(closes, closes[1:])]
        return sample_standard_deviation(returns) * math.sqrt(365.0)
    if highs is None or lows is None:
        raise EngineInputError("Parkinson estimator requires completed highs and lows")
    terms = []
    for high, low in zip(highs, lows):
        if high <= 0 or low <= 0 or high < low:
            raise EngineInputError("invalid Parkinson daily high/low")
        terms.append(math.log(high / low) ** 2)
    return math.sqrt((365.0 / (4.0 * math.log(2.0))) * (sum(terms) / len(terms)))


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


def _mad_scaled(value: float, population: Sequence[float]) -> float:
    finite = sorted(float(item) for item in population if math.isfinite(float(item)))
    if len(finite) < 30 or len(set(finite)) < 20:
        raise EngineInputError("A4 ensemble population minimum is not met")
    lower, upper = type7_quantile(finite, 0.01), type7_quantile(finite, 0.99)
    winsorized = min(upper, max(lower, value))
    center = median(finite)
    mad = median(abs(item - center) for item in finite)
    if mad <= 0:
        raise EngineInputError("A4 ensemble MAD is zero")
    return (winsorized - center) / mad


def evaluate(frame: FamilyInput, config: Mapping[str, Any], *, control_id: str | None = None) -> list[dict[str, Any]]:
    """Evaluate the registered rebalance, estimator, gates and frozen exposure."""
    frame.validate()
    decision = require_utc(frame.decision_ts)
    if config["rebalance"] == "1d" and (decision.hour, decision.minute) != (0, 0):
        return []
    if config["rebalance"] == "8h" and (decision.hour not in {0, 8, 16} or decision.minute != 0):
        return []
    lookback = int(config["lookback_days"])
    vol_window = int(config["vol_window_days"])
    required = max(lookback + 1, vol_window + 1)
    if len(frame.daily_bars) < required:
        raise EngineInputError("A4 daily history is incomplete")
    selected = frame.daily_bars[-required:]
    closes = [bar.close for bar in selected]
    highs = [bar.high for bar in selected]
    lows = [bar.low for bar in selected]
    signal_closes = closes[-(lookback + 1):]
    estimator_config = dict(config)
    if control_id == "A4_GENERIC_SIGNED_RETURN":
        estimator_config["signal_estimator"] = "signed_return"
    ensemble = None
    if estimator_config["signal_estimator"] == "equal_weight_ensemble":
        signed = log_return(signal_closes[0], signal_closes[-1])
        ema_series = ema(signal_closes, max(2, len(signal_closes) // 2))
        slope = log_return(ema_series[0], ema_series[-1])
        atr = wilder_atr(highs[-(min(20, len(highs) - 1) + 1):], lows[-(min(20, len(lows) - 1) + 1):], closes[-(min(20, len(closes) - 1) + 1):], min(20, len(closes) - 1))
        breakout = max((closes[-1] - max(highs[-(lookback + 1):-1])) / atr, 0.0) + min((closes[-1] - min(lows[-(lookback + 1):-1])) / atr, 0.0)
        ensemble = [
            _mad_scaled(signed, frame.threshold_populations["A4_ensemble:signed_return"].values),
            _mad_scaled(slope, frame.threshold_populations["A4_ensemble:ema_slope"].values),
            _mad_scaled(breakout, frame.threshold_populations["A4_ensemble:breakout_distance_rank"].values),
        ]
    atr_window = int(config["ATR_window_days_for_ATR_exits"] or 20)
    atr_daily = frame.daily_bars[-(atr_window + 1):]
    atr = wilder_atr([bar.high for bar in atr_daily], [bar.low for bar in atr_daily], [bar.close for bar in atr_daily], atr_window)
    scalar = signal_scalar(
        estimator_config,
        signal_closes,
        highs=highs[-len(signal_closes):],
        lows=lows[-len(signal_closes):],
        prior_high=max(highs[-(lookback + 1):-1]),
        prior_low=min(lows[-(lookback + 1):-1]),
        atr=atr,
        ensemble_components=ensemble,
    )
    if control_id == "A4_SIGN_PERMUTED_MAIN_NULL":
        sign = int(frame.metadata["control_signal_sign"])
        if sign not in (-1, 0, 1):
            raise EngineInputError("permuted A4 sign is invalid")
        scalar = abs(scalar) * sign
    side = side_from_scalar(scalar, str(config["direction"]))
    if side == 0:
        return []
    if config["path_smoothness_quantile_min"] != "none" and control_id != "A4_PATH_COMPONENT_REMOVED":
        intraday_closes = [bar.close for bar in frame.five_minute_bars if require_utc(bar.close_ts) <= decision]
        expected = lookback * 288
        if len(intraday_closes) < expected:
            raise EngineInputError("A4 path smoothness history is incomplete")
        smoothness = path_smoothness(intraday_closes[-expected:])
        population = frame.threshold_populations["A4_path_smoothness"]
        population.validate()
        _, passes = percentile_from_population(smoothness, population.values, str(config["path_smoothness_quantile_min"]))
        if not passes:
            return []
    realized = volatility(config, closes[-(vol_window + 1):], highs[-(vol_window + 1):], lows[-(vol_window + 1):])
    target = config["annualized_vol_target"]
    exposure = 1.0 if target == "none" or control_id == "A4_VOL_SCALING_REMOVED" else min(2.0, max(0.25, float(target) / realized))
    context_multiplier = 1.0
    if config["context_overlay"] != "none" and control_id != "A4_CONTEXT_REMOVED":
        from .a2_context import named_context_multiplier
        context_multiplier = named_context_multiplier(frame, str(config["context_overlay"]), side)
    entry_index = next((index for index, bar in enumerate(frame.five_minute_bars) if require_utc(bar.open_ts) >= decision), None)
    if entry_index is None:
        raise EngineInputError("A4 entry open is unavailable")
    if (require_utc(frame.five_minute_bars[entry_index].open_ts) - decision).total_seconds() > 600:
        raise EngineInputError("A4 entry open exceeds ten-minute lookup")
    return [{
        "event_id": event_id(frame.symbol, decision.isoformat(), canonical_hash(config), side),
        "side": side,
        "decision_ts": decision,
        "entry_index": entry_index,
        "atr": atr if str(config["exit"]).startswith("ATR_") else None,
        "signal_scalar": scalar,
        "exposure": exposure,
        "context_multiplier": context_multiplier,
    }]


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


__all__ = ["contract", "evaluate", "event_id", "path_smoothness", "side_from_scalar", "signal_scalar", "volatility"]
