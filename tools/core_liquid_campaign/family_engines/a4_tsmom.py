from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

from ..canonical import canonical_hash
from ..engine_types import FamilyInput
from .common import (
    EngineInputError,
    ema,
    log_return,
    path_smoothness,
    percentile_from_population,
    require_utc,
    sample_standard_deviation,
    type7_quantile,
    wilder_atr,
)


ENGINE_ID = "a4_tsmom_engine_v2"
FIVE_MINUTE_PERIODS_PER_YEAR = 365 * 288


def ensemble_population_key(component: str, lookback_days: int, volatility_estimator: str) -> str:
    return f"A4_ensemble:{component}:lookback={lookback_days}:volatility={volatility_estimator}"


def smoothness_population_key(lookback_days: int) -> str:
    return f"A4_path_smoothness:lookback={lookback_days}"


def volatility(
    config: Mapping[str, Any],
    closes: Sequence[float],
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
) -> float:
    """Frozen five-minute annualized volatility estimators."""
    if config["volatility_estimator"] == "close_to_close":
        returns = [log_return(left, right) for left, right in zip(closes, closes[1:])]
        return sample_standard_deviation(returns) * math.sqrt(FIVE_MINUTE_PERIODS_PER_YEAR)
    if highs is None or lows is None or len(highs) != len(lows) or len(highs) != len(closes):
        raise EngineInputError("Parkinson estimator requires aligned completed highs and lows")
    terms = []
    for high, low in zip(highs, lows):
        if high <= 0 or low <= 0 or high < low:
            raise EngineInputError("invalid Parkinson five-minute high/low")
        terms.append(math.log(high / low) ** 2)
    if len(terms) < 2:
        raise EngineInputError("Parkinson estimator history is incomplete")
    return math.sqrt((FIVE_MINUTE_PERIODS_PER_YEAR / (4.0 * math.log(2.0))) * (sum(terms) / len(terms)))


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
    realized_vol: float | None = None,
) -> float:
    estimator = config["signal_estimator"]
    vol = float(realized_vol) if realized_vol is not None else volatility(config, closes, highs, lows)
    if not math.isfinite(vol) or vol <= 0:
        raise EngineInputError("volatility estimator is unavailable")
    if estimator == "signed_return":
        return log_return(closes[0], closes[-1]) / vol
    if estimator == "ema_slope":
        span = int(config["lookback_days"]) * 288
        if len(closes) < span + 289:
            raise EngineInputError("EMA slope requires span plus a t-minus-one-day reference")
        series = ema(closes, span)
        return log_return(series[-289], series[-1]) / vol
    if estimator == "breakout_distance_rank":
        if prior_high is None or prior_low is None or atr is None or atr <= 0:
            raise EngineInputError("breakout-distance estimator requires prior high, prior low and ATR")
        close = closes[-1]
        return max((close - prior_high) / atr, 0.0) + min((close - prior_low) / atr, 0.0)
    if estimator == "equal_weight_ensemble":
        if ensemble_components is None or len(ensemble_components) not in {2, 3}:
            raise EngineInputError("ensemble requires the registered two or three scaled components")
        if any(not math.isfinite(float(value)) for value in ensemble_components):
            raise EngineInputError("ensemble component unavailable")
        return sum(float(value) for value in ensemble_components) / len(ensemble_components)
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
    lower, upper = type7_quantile(finite, 0.05), type7_quantile(finite, 0.95)
    fitted = [min(upper, max(lower, item)) for item in finite]
    winsorized = min(upper, max(lower, value))
    center = median(fitted)
    mad = median(abs(item - center) for item in fitted)
    if mad <= 0:
        raise EngineInputError("A4 ensemble MAD is zero")
    return (winsorized - center) / mad


def _estimator_state(frame: FamilyInput, config: Mapping[str, Any], control_id: str | None = None) -> dict[str, Any] | None:
    """Compute the pre-return A4 estimator state for one registered decision."""
    frame.validate()
    frame.require_pit_top_n(int(config["PIT_liquidity_top_n"]))
    decision = require_utc(frame.decision_ts)
    if config["rebalance"] == "1d" and (decision.hour, decision.minute) != (0, 0):
        return None
    if config["rebalance"] == "8h" and (decision.hour not in {0, 8, 16} or decision.minute != 0):
        return None
    lookback = int(config["lookback_days"])
    vol_window = int(config["vol_window_days"])
    completed = [bar for bar in frame.five_minute_bars if require_utc(bar.close_ts) <= decision]
    required_5m = max(lookback * 288 + 289, vol_window * 288 + 1)
    if len(completed) < required_5m:
        raise EngineInputError("A4 completed five-minute history is incomplete")
    closes = [bar.close for bar in completed]
    highs = [bar.high for bar in completed]
    lows = [bar.low for bar in completed]
    return_count = lookback * 288 + 1
    return_closes = closes[-return_count:]
    return_highs = highs[-return_count:]
    return_lows = lows[-return_count:]
    ema_closes = closes[-(lookback * 288 + 289):]
    available_daily = [bar for bar in frame.daily_bars if require_utc(bar.close_ts) < decision]
    if len(available_daily) < max(lookback, 21):
        raise EngineInputError("A4 prior daily range/ATR history is incomplete")
    prior = available_daily[-lookback:]
    atr_bars = available_daily[-21:]
    atr20 = wilder_atr([bar.high for bar in atr_bars], [bar.low for bar in atr_bars], [bar.close for bar in atr_bars], 20)

    estimator_config = dict(config)
    if control_id == "A4_GENERIC_SIGNED_RETURN":
        estimator_config["signal_estimator"] = "signed_return"
    ensemble = None
    if estimator_config["signal_estimator"] == "equal_weight_ensemble":
        signal_vol = volatility(config, return_closes, return_highs, return_lows)
        ema_series = ema(ema_closes, lookback * 288)
        raw_components = {
            "signed_return": log_return(return_closes[0], return_closes[-1]) / signal_vol,
            "ema_slope": log_return(ema_series[-289], ema_series[-1]) / signal_vol,
            "breakout_distance_rank": max((closes[-1] - max(bar.high for bar in prior)) / atr20, 0.0)
            + min((closes[-1] - min(bar.low for bar in prior)) / atr20, 0.0),
        }
        names = ("signed_return", "ema_slope") if control_id == "A4_PATH_COMPONENT_REMOVED" else tuple(raw_components)
        ensemble = []
        for name in names:
            population = frame.threshold_populations[ensemble_population_key(name, lookback, str(config["volatility_estimator"]))]
            population.validate(decision_ts=decision)
            ensemble.append(_mad_scaled(raw_components[name], population.values))
    scalar_input = ema_closes if estimator_config["signal_estimator"] == "ema_slope" else return_closes
    scalar = signal_scalar(
        estimator_config,
        scalar_input,
        highs=return_highs if len(return_highs) == len(scalar_input) else highs[-len(scalar_input):],
        lows=return_lows if len(return_lows) == len(scalar_input) else lows[-len(scalar_input):],
        prior_high=max(bar.high for bar in prior),
        prior_low=min(bar.low for bar in prior),
        atr=atr20,
        ensemble_components=ensemble,
        realized_vol=volatility(config, return_closes, return_highs, return_lows),
    )
    return {
        "decision": decision,
        "lookback": lookback,
        "vol_window": vol_window,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "available_daily": available_daily,
        "scalar": scalar,
    }


def scalar_estimator_sign(frame: FamilyInput, config: Mapping[str, Any]) -> int:
    """Return the complete pre-return estimator sign, including exact zero."""
    state = _estimator_state(frame, config)
    if state is None:
        raise EngineInputError("frame is not on the registered A4 rebalance calendar")
    scalar = float(state["scalar"])
    return 0 if scalar == 0 else (1 if scalar > 0 else -1)


def evaluate(
    frame: FamilyInput,
    config: Mapping[str, Any],
    *,
    control_id: str | None = None,
    control_directive: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one exact rebalance decision from completed five-minute inputs."""
    state = _estimator_state(frame, config, control_id)
    if state is None:
        return []
    decision = state["decision"]
    lookback = int(state["lookback"])
    vol_window = int(state["vol_window"])
    closes = state["closes"]
    highs = state["highs"]
    lows = state["lows"]
    available_daily = state["available_daily"]
    scalar = float(state["scalar"])
    generic_signed = control_id == "A4_GENERIC_SIGNED_RETURN"
    if control_id == "A4_SIGN_PERMUTED_MAIN_NULL":
        sign = None if control_directive is None else control_directive.get("signal_sign")
        if sign not in (-1, 0, 1):
            raise EngineInputError("derived permuted A4 sign is absent or invalid")
        scalar = abs(scalar) * int(sign)
    side = side_from_scalar(scalar, str(config["direction"]))
    if side == 0:
        return []

    if config["path_smoothness_quantile_min"] != "none" and control_id != "A4_PATH_COMPONENT_REMOVED" and not generic_signed:
        expected = lookback * 288
        smoothness = path_smoothness(closes[-(expected + 1):])
        population = frame.threshold_populations[smoothness_population_key(lookback)]
        population.validate(decision_ts=decision)
        _, passes = percentile_from_population(smoothness, population.values, str(config["path_smoothness_quantile_min"]))
        if not passes:
            return []

    vol_count = vol_window * 288 + 1
    realized = volatility(config, closes[-vol_count:], highs[-vol_count:], lows[-vol_count:])
    target = config["annualized_vol_target"]
    exposure = 1.0 if target == "none" or control_id in {"A4_VOL_SCALING_REMOVED", "A4_GENERIC_SIGNED_RETURN"} else min(2.0, max(0.25, float(target) / realized))
    context_multiplier = 1.0
    if config["context_overlay"] != "none" and control_id not in {"A4_CONTEXT_REMOVED", "A4_GENERIC_SIGNED_RETURN"}:
        from .a2_context import named_context_multiplier

        context_multiplier = named_context_multiplier(frame, str(config["context_overlay"]), side)
    entry_index = next((index for index, bar in enumerate(frame.five_minute_bars) if require_utc(bar.open_ts) >= decision), None)
    if entry_index is None:
        raise EngineInputError("A4 entry open is unavailable")
    if (require_utc(frame.five_minute_bars[entry_index].open_ts) - decision).total_seconds() > 600:
        raise EngineInputError("A4 entry open exceeds ten-minute lookup")
    atr_window = int(config["ATR_window_days_for_ATR_exits"] or 20)
    exit_atr_bars = available_daily[-(atr_window + 1):]
    exit_atr = wilder_atr([bar.high for bar in exit_atr_bars], [bar.low for bar in exit_atr_bars], [bar.close for bar in exit_atr_bars], atr_window)
    return [{
        "event_id": event_id(frame.symbol, decision.isoformat(), canonical_hash(config), side),
        "cohort_id": canonical_hash({"config": canonical_hash(config), "decision_ts": decision.isoformat()}),
        "side": side,
        "decision_ts": decision,
        "entry_index": entry_index,
        "atr": exit_atr if str(config["exit"]).startswith("ATR_") else None,
        "signal_scalar": scalar,
        "exposure": exposure,
        "context_multiplier": context_multiplier,
    }]


def contract() -> dict[str, Any]:
    return {
        "engine_id": ENGINE_ID,
        "event_type": "definition_symbol_episode",
        "event_identity": "SHA256(event_type,symbol,decision_ts,config_hash,side)",
        "features": ["5m_signed_return", "5m_ema_slope", "daily_prior_range_breakout_distance", "q05_q95_MAD_ensemble", "5m_path_smoothness", "5m_close_to_close_volatility", "5m_parkinson_volatility"],
        "side_grammar": "strict scalar sign; exact zero is flat",
        "entry": "registered UTC rebalance decision; next authorized trade open",
        "exit": "registered time, signal reversal, or completed-close ATR trail; next authorized trade open",
        "non_overlap": "definition-symbol chronological acceptance using actual executable exit",
        "accounting": "sum frozen equal-cohort weighted symbol contributions once; mean nonempty cohorts by UTC day",
        "threshold_populations": "fold-training-bound symbol-local scope; no fallback",
        "controls": ["derived sign permutation", "signed-return only with vol/path/context disabled", "no smoothness/breakout component", "unscaled exposure", "context null"],
        "removed_axis": "signal_rank_scope",
        "removed_axis_reason": "no source-defined economic consumer",
    }


__all__ = ["FIVE_MINUTE_PERIODS_PER_YEAR", "contract", "ensemble_population_key", "evaluate", "event_id", "path_smoothness", "scalar_estimator_sign", "side_from_scalar", "signal_scalar", "smoothness_population_key", "volatility"]
