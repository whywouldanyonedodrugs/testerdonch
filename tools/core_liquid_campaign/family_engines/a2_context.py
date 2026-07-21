from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

from ..canonical import canonical_hash
from ..engine_types import DailyBar, FamilyInput
from .common import EngineInputError, average_rank_percentiles, component_threshold, log_return, require_utc, sample_standard_deviation, weak_percentile, wilder_atr


ENGINE_ID = "a2_context_engine_v1"


def proximity_population_key(lookback_days: int, atr_window_days: int, side: int) -> str:
    return f"A2_proximity:lookback={lookback_days}:atr={atr_window_days}:side={side}"


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
    if any(not 0.0 <= value <= 1.0 for _, value in components):
        raise EngineInputError("A2 component vector is outside [0,1]")
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
        if not values:
            return 1.0
        minimum = min(values)
        return 0.0 if minimum < 1.0 / 3.0 else 0.5 if minimum < 2.0 / 3.0 else 1.0
    raise EngineInputError(f"unknown A2 overlay action: {action}")


def parent_slot_id(parent_family: str, fold: str, beam_rank: int) -> str:
    if parent_family not in {"A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"} or not 1 <= beam_rank <= 5:
        raise EngineInputError("A2 parent slot is outside the frozen family/five-slot beam")
    return f"{parent_family}:{fold}:beam:{beam_rank:02d}"


def counterpart_ids(config_hash: str, parent_identity: str) -> tuple[str, str]:
    base = {"config_hash": config_hash, "parent_identity": parent_identity}
    return canonical_hash({**base, "role": "parent_only"}), canonical_hash({**base, "role": "context_overlay"})


def _return(bars: Sequence[DailyBar], days: int) -> float:
    if len(bars) < days + 1:
        raise EngineInputError("context daily return history is incomplete")
    return log_return(bars[-days - 1].close, bars[-1].close)


def _timeseries_percentile(value: float, history: Sequence[float]) -> float:
    return weak_percentile(value, history)


def _btc_eth_component(frame: FamilyInput, config: Mapping[str, Any], side: int) -> float:
    mode = str(config["BTC_ETH_context"])
    values = []
    for asset, bars in (("BTC", frame.context.btc_daily), ("ETH", frame.context.eth_daily)):
        if mode == "trend":
            short, long = (int(item) for item in str(config["BTC_ETH_trend_pair_days"]).split("_"))
            short_return, long_return = _return(bars, short), _return(bars, long)
            short_p = _timeseries_percentile(short_return, frame.threshold_populations[f"A2_{asset}_trend_{short}"].values)
            long_p = _timeseries_percentile(long_return, frame.threshold_populations[f"A2_{asset}_trend_{long}"].values)
            component = (short_p + long_p) / 2.0
            values.append(component if side == 1 else 1.0 - component)
        elif mode == "volatility":
            window = int(config["BTC_ETH_volatility_lookback_days"])
            returns = [log_return(left.close, right.close) for left, right in zip(bars[-(window + 1):], bars[-window:])]
            volatility = sample_standard_deviation(returns) * math.sqrt(365.0)
            values.append(1.0 - _timeseries_percentile(volatility, frame.threshold_populations[f"A2_{asset}_volatility_{window}"].values))
        elif mode == "drawdown":
            window = int(config["BTC_ETH_drawdown_lookback_days"])
            if len(bars) < window:
                raise EngineInputError("BTC/ETH drawdown history is incomplete")
            drawdown = 1.0 - bars[-1].close / max(bar.close for bar in bars[-window:])
            percentile = _timeseries_percentile(drawdown, frame.threshold_populations[f"A2_{asset}_drawdown_{window}"].values)
            values.append(1.0 - percentile if side == 1 else percentile)
        else:
            raise EngineInputError(f"unsupported BTC/ETH context: {mode}")
    return sum(values) / 2.0


def raw_component_percentiles(frame: FamilyInput, config: Mapping[str, Any], side: int) -> dict[str, float]:
    """Compute A2 percentiles from completed raw histories at the parent decision."""
    decision = require_utc(frame.decision_ts)
    symbol_daily = frame.context.symbol_daily or frame.daily_bars
    if any(require_utc(bar.close_ts) >= decision for bar in symbol_daily):
        raise EngineInputError("A2 daily context is not point-in-time safe")
    result: dict[str, float] = {}
    completed_intraday = [bar for bar in frame.five_minute_bars if require_utc(bar.close_ts) <= decision]
    if len(completed_intraday) < 2:
        raise EngineInputError("A2 intraday decision history is incomplete")
    if config["proximity_rank"] != "none" or config["reclaim_state"] != "none":
        lookback = int(config["prior_high_lookback_days"])
        atr_window = int(config["ATR_window_days_for_proximity"])
        if len(symbol_daily) < max(lookback, atr_window + 1):
            raise EngineInputError("A2 prior-level history is incomplete")
        level = max(bar.high for bar in symbol_daily[-lookback:]) if side == 1 else min(bar.low for bar in symbol_daily[-lookback:])
        atr_bars = symbol_daily[-(atr_window + 1):]
        atr = wilder_atr([bar.high for bar in atr_bars], [bar.low for bar in atr_bars], [bar.close for bar in atr_bars], atr_window)
        current = completed_intraday[-1].close
        proximity = (current - level) / atr if side == 1 else (level - current) / atr
        result["proximity"] = _timeseries_percentile(proximity, frame.threshold_populations[proximity_population_key(lookback, atr_window, side)].values)
        if config["reclaim_state"] == "continuous_distance":
            result["reclaim"] = result["proximity"]
        elif config["reclaim_state"] == "first_close_above":
            previous = completed_intraday[-2].close
            crossed = previous <= level < current if side == 1 else previous >= level > current
            result["reclaim"] = 1.0 if crossed else 0.0
    if config["RS_rank"] != "none":
        lookback = int(config["RS_lookback_days"])
        symbol_return = _return(symbol_daily, lookback)
        btc_return = _return(frame.context.btc_daily, lookback)
        raw_rs = side * (symbol_return - btc_return)
        scope = str(config["RS_population_scope"])
        candidates = dict(frame.context.cross_section_returns_by_lookback.get(lookback, {}))
        if not candidates:
            raise EngineInputError("A2 registered RS lookback has no bound cross-sectional snapshot")
        if scope == "liquidity_decile":
            decile = frame.context.cross_section_liquidity_deciles.get(frame.symbol)
            candidates = {symbol: value for symbol, value in candidates.items() if frame.context.cross_section_liquidity_deciles.get(symbol) == decile}
        elif scope == "parent_liquidity_universe":
            candidates = {symbol: value for symbol, value in candidates.items() if symbol in frame.context.parent_universe}
        if len(candidates) < 5:
            raise EngineInputError("A2 cross-sectional RS population has fewer than five symbols")
        ordered_symbols = sorted(candidates)
        values = [side * (float(candidates[symbol]) - btc_return) for symbol in ordered_symbols]
        ranks = average_rank_percentiles(values)
        if frame.symbol in candidates:
            result["RS"] = ranks[ordered_symbols.index(frame.symbol)]
        else:
            extended = values + [raw_rs]
            result["RS"] = average_rank_percentiles(extended)[-1]
    if config["BTC_ETH_context"] != "none":
        result["BTC_ETH"] = _btc_eth_component(frame, config, side)
    breadth_mode = str(config["breadth_dispersion"])
    if breadth_mode in {"breadth", "both"}:
        lookback = int(config["breadth_return_lookback_days"])
        returns = frame.context.cross_section_returns_by_lookback.get(lookback)
        if not returns or len(returns) < 5:
            raise EngineInputError("A2 registered breadth lookback has insufficient PIT members")
        current = sum(float(value) > 0 for value in returns.values()) / len(returns)
        history = frame.context.breadth_history_by_lookback.get(lookback)
        if history is None:
            raise EngineInputError("A2 registered breadth lookback has no fold-training history")
        percentile = _timeseries_percentile(current, history)
        result["breadth"] = percentile if side == 1 else 1.0 - percentile
    if breadth_mode in {"dispersion", "both"}:
        lookback = int(config["dispersion_return_lookback_days"])
        returns = frame.context.cross_section_returns_by_lookback.get(lookback)
        if not returns or len(returns) < 5:
            raise EngineInputError("A2 registered dispersion lookback has insufficient PIT members")
        values = [float(value) for value in returns.values()]
        center = median(values)
        current = median(abs(value - center) for value in values)
        history = frame.context.dispersion_history_by_lookback.get(lookback)
        if history is None:
            raise EngineInputError("A2 registered dispersion lookback has no fold-training history")
        result["dispersion"] = _timeseries_percentile(current, history)
    return result


def evaluate_overlay(frame: FamilyInput, config: Mapping[str, Any], parent_event: Mapping[str, Any], *, control_id: str | None = None, control_directive: Mapping[str, Any] | None = None) -> dict[str, Any]:
    side = int(parent_event["side"])
    raw = raw_component_percentiles(frame, config, side)
    effective = dict(config)
    if control_id == "A2_PRIOR_HIGH_REMOVED":
        effective["proximity_rank"] = "none"
        effective["reclaim_state"] = "none"
    elif control_id == "A2_RS_REMOVED":
        effective["RS_rank"] = "none"
    elif control_id == "A2_EXTERNAL_CONTEXT_REMOVED":
        effective["BTC_ETH_context"] = "none"
        effective["breadth_dispersion"] = "none"
    if control_id == "A2_PARENT_ONLY":
        multiplier, components = 1.0, []
    elif control_id == "A2_CONTEXT_PERMUTED_MAIN_NULL":
        supplied = None if control_directive is None else control_directive.get("context_vector")
        if not isinstance(supplied, (list, tuple)):
            raise EngineInputError("derived permuted A2 context vector is absent")
        components = [(str(name), float(value)) for name, value in supplied]
        multiplier = overlay_multiplier(str(config["overlay_action"]), components)
    else:
        components = component_vector(effective, raw)
        multiplier = overlay_multiplier(str(config["overlay_action"]), components)
    return {
        **dict(parent_event),
        "context_components": components,
        "context_multiplier": multiplier,
        "parent_only_counterpart_id": parent_event.get("parent_only_counterpart_id"),
        "overlay_counterpart_id": parent_event.get("overlay_counterpart_id"),
    }


def named_context_multiplier(frame: FamilyInput, overlay: str, side: int) -> float:
    if overlay == "none":
        return 1.0
    if overlay == "funding_veto":
        if frame.context.funding_burden_current is None:
            raise EngineInputError("current funding burden is unavailable")
        current = side * float(frame.context.funding_burden_current)
        percentile = weak_percentile(current, tuple(side * float(value) for value in frame.context.funding_burden_history))
        return 1.0 if percentile < 0.90 else 0.0
    if overlay == "prior_high_RS":
        config = {
            "proximity_rank": "q50",
            "reclaim_state": "none",
            "prior_high_lookback_days": 60,
            "ATR_window_days_for_proximity": 20,
            "RS_rank": "q50",
            "RS_lookback_days": 20,
            "RS_population_scope": "global_PIT",
            "BTC_ETH_context": "none",
            "breadth_dispersion": "none",
        }
        components = component_vector(config, raw_component_percentiles(frame, config, side))
        return overlay_multiplier("permission", components)
    if overlay in {"BTC_ETH", "breadth_dispersion"}:
        config = {
            "proximity_rank": "none",
            "reclaim_state": "none",
            "RS_rank": "none",
            "BTC_ETH_context": "trend" if overlay == "BTC_ETH" else "none",
            "BTC_ETH_trend_pair_days": "20_60",
            "breadth_dispersion": "both" if overlay == "breadth_dispersion" else "none",
            "breadth_return_lookback_days": 20,
            "dispersion_return_lookback_days": 20,
        }
        components = component_vector(config, raw_component_percentiles(frame, config, side))
        if overlay == "breadth_dispersion":
            by_name = dict(components)
            combined = 0.5 * by_name["breadth"] + 0.5 * by_name["dispersion"]
            return max(0.0, min(1.0, 2.0 * combined - 1.0))
        return overlay_multiplier("permission", components)
    raise EngineInputError(f"unsupported named context overlay: {overlay}")


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


__all__ = ["component_vector", "contract", "counterpart_ids", "evaluate_overlay", "named_context_multiplier", "overlay_multiplier", "paired_uplift", "parent_slot_id", "proximity_population_key", "raw_component_percentiles"]
