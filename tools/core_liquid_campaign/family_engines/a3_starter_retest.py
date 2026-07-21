from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

from ..canonical import canonical_hash
from ..engine_types import FamilyInput, require_contiguous_5m, require_contiguous_daily
from .common import EngineInputError, percentile_from_population, require_utc, wilder_atr


ENGINE_ID = "a3_starter_retest_engine_v1"


def breakout_population_key(lookback_days: int, atr_window_days: int, scope: str, side: int) -> str:
    return f"A3_breakout:lookback={lookback_days}:atr={atr_window_days}:scope={scope}:side={side}"


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


@dataclass(frozen=True)
class RetestResult:
    status: str
    activation_index: int | None
    reclaim_index: int | None
    add_entry_index: int | None


def run_retest_state_machine(
    frame: FamilyInput,
    *,
    direction: str,
    level: float,
    atr: float,
    depth: float,
    starter_entry_index: int,
    starter_exit_ts: object,
    window: str,
) -> RetestResult:
    duration = {"6h": timedelta(hours=6), "1d": timedelta(days=1), "3d": timedelta(days=3)}.get(window)
    if duration is None:
        raise EngineInputError(f"unsupported A3 retest window: {window}")
    bars = frame.five_minute_bars
    starter_entry_ts = require_utc(bars[starter_entry_index].open_ts)
    deadline = starter_entry_ts + duration
    exit_ts = require_utc(starter_exit_ts)  # type: ignore[arg-type]
    state = "waiting_activation"
    activation_index: int | None = None
    for index in range(starter_entry_index + 1, len(bars) - 1):
        bar = bars[index]
        if require_utc(bar.close_ts) >= deadline or require_utc(bar.close_ts) >= exit_ts:
            break
        prior = bars[index - 1]
        transition = retest_state(direction, level, atr, depth, prior.close, bar.close)
        if transition == "invalidated":
            return RetestResult("unavailable_invalidated", activation_index, None, None)
        if state == "waiting_activation" and transition == "activated":
            state = "activated"
            activation_index = index
            continue
        activation_close = bars[int(activation_index)].close if activation_index is not None else None
        beyond_activation = activation_close is not None and (bar.close > activation_close if direction == "long" else bar.close < activation_close)
        if state == "activated" and index > int(activation_index) and transition == "reclaimed" and beyond_activation:
            add_entry_index = index + 1
            if add_entry_index >= len(bars):
                return RetestResult("unavailable_missing_open", activation_index, index, None)
            if require_utc(bars[add_entry_index].open_ts) - require_utc(bar.close_ts) > timedelta(minutes=10):
                return RetestResult("unavailable_missing_open", activation_index, index, None)
            if require_utc(bars[add_entry_index].open_ts) >= deadline or require_utc(bars[add_entry_index].open_ts) >= exit_ts:
                return RetestResult("unavailable_after_window_or_starter_exit", activation_index, index, None)
            return RetestResult("complete", activation_index, index, add_entry_index)
    return RetestResult("unavailable_no_reclaim", activation_index, None, None)


def _population(frame: FamilyInput, lookback: int, atr_window: int, scope: str, side: int):
    name = breakout_population_key(lookback, atr_window, scope, side)
    try:
        population = frame.threshold_populations[name]
    except KeyError as exc:
        raise EngineInputError(f"missing threshold population: {name}") from exc
    population.validate(pooled="global" in scope or "liquidity_decile" in scope, decision_ts=frame.decision_ts)
    return population.values


def evaluate(frame: FamilyInput, config: Mapping[str, Any], *, control_id: str | None = None, control_directive: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Generate directional starter events and retain a stateful retest plan."""
    frame.validate()
    frame.require_pit_top_n(int(config["PIT_liquidity_top_n"]))
    require_contiguous_5m(frame.five_minute_bars)
    direction = str(config["direction"])
    side = 1 if direction == "long" else -1
    lookback = int(config["breakout_lookback_days"])
    daily = tuple(bar for bar in frame.daily_bars if require_utc(bar.close_ts) < require_utc(frame.decision_ts))
    if len(daily) < max(lookback, int(config["ATR_window_days"] or 20) + 1):
        raise EngineInputError("A3 daily history is incomplete")
    require_contiguous_daily(daily[-max(lookback, int(config["ATR_window_days"] or 20) + 1):])
    level = max(bar.high for bar in daily[-lookback:]) if side == 1 else min(bar.low for bar in daily[-lookback:])
    atr_window = int(config["ATR_window_days"] or 20)
    atr_daily = daily[-(atr_window + 1):]
    atr = wilder_atr([bar.high for bar in atr_daily], [bar.low for bar in atr_daily], [bar.close for bar in atr_daily], atr_window)
    bars = frame.five_minute_bars
    if control_id == "A3_RETEST_TIME_PERMUTED_MAIN_NULL" and (not isinstance(control_directive, Mapping) or "add_lag_bars" not in control_directive):
        raise EngineInputError("A3 retest permutation lacks a derived add-lag directive")
    if control_id == "A3_MATCHED_PSEUDO_EVENT":
        if not isinstance(control_directive, Mapping) or control_directive.get("allocator") != "matched_pseudo_event_allocator_v2" or control_directive.get("side") != side or control_directive.get("matched_decision_ts") != require_utc(frame.decision_ts):
            raise EngineInputError("A3 matched pseudo event lacks an exact derived allocator directive")
        entry_index = next((i for i, bar in enumerate(bars) if require_utc(bar.open_ts) >= require_utc(frame.decision_ts)), None)
        if entry_index is None:
            return []
        return [{
            "event_id": canonical_hash({"control": control_id, "parent_event_id": control_directive["parent_event_id"], "symbol": frame.symbol, "decision_ts": require_utc(frame.decision_ts).isoformat()}),
            "side": side, "decision_ts": require_utc(frame.decision_ts), "entry_index": entry_index, "level": level, "atr": atr,
            "magnitude": breakout_magnitude(bars[max(0, entry_index - 1)].close, level, atr, direction),
            "starter_fraction": float(config["starter_fraction"]), "add_fraction": float(config["add_fraction"]),
            "retest_depth": config.get("retest_depth_ATR"), "retest_window": config.get("retest_window"), "context_multiplier": 1.0,
            "matched_parent_event_id": control_directive["parent_event_id"],
        }]
    events: list[dict[str, Any]] = []
    for index in range(1, len(bars) - 4):
        prior, crossing = bars[index - 1], bars[index]
        pseudo_match = control_id == "A3_MATCHED_PSEUDO_EVENT" and require_utc(crossing.close_ts) == require_utc(frame.decision_ts)
        crossed = pseudo_match or (prior.close <= level < crossing.close if side == 1 else prior.close >= level > crossing.close)
        if not crossed:
            continue
        magnitude = breakout_magnitude(crossing.close, level, atr, direction)
        if control_id not in {"A3_CONFIRMATION_REMOVED", "A3_MATCHED_PSEUDO_EVENT"}:
            _, passes = percentile_from_population(magnitude, _population(frame, lookback, atr_window, str(config["breakout_rank_scope"]), side), str(config["breakout_rank_min"]))
            if not passes:
                continue
        confirmation = str(config["confirmation"])
        if control_id in {"A3_CONFIRMATION_REMOVED", "A3_MATCHED_PSEUDO_EVENT"} or confirmation == "one_close":
            confirmation_indices = (index,)
        elif confirmation == "two_closes":
            confirmation_indices = (index, index + 1)
        elif confirmation == "close_plus_15m_delay":
            confirmation_indices = (index, index + 3)
        else:
            raise EngineInputError(f"unsupported A3 confirmation: {confirmation}")
        selected = [bars[item] for item in confirmation_indices]
        final_confirmation_ts = require_utc(selected[-1].close_ts)
        if final_confirmation_ts > require_utc(frame.decision_ts):
            break
        if confirmation == "close_plus_15m_delay" and require_utc(selected[-1].close_ts) - require_utc(selected[0].close_ts) != timedelta(minutes=15):
            raise EngineInputError("A3 delayed confirmation is not exactly fifteen minutes")
        if not all(side * (bar.close - level) > 0 for bar in selected):
            continue
        if final_confirmation_ts != require_utc(frame.decision_ts):
            continue
        entry_index = confirmation_indices[-1] + 1
        if require_utc(bars[entry_index].open_ts) - require_utc(selected[-1].close_ts) > timedelta(minutes=10):
            continue
        context_multiplier = 1.0
        if config["context_overlay"] != "none" and control_id != "A3_CONTEXT_REMOVED":
            from .a2_context import named_context_multiplier
            context_multiplier = named_context_multiplier(frame, str(config["context_overlay"]), side)
        events.append({
            "event_id": event_id(frame.symbol, direction, require_utc(crossing.close_ts).isoformat(), level, canonical_hash(config)),
            "side": side,
            "decision_ts": require_utc(selected[-1].close_ts),
            "entry_index": entry_index,
            "level": level,
            "atr": atr,
            "magnitude": magnitude,
            "starter_fraction": 1.0 if control_id == "A3_STARTER_ONLY" else float(config["starter_fraction"]),
            "add_fraction": 0.0 if control_id == "A3_STARTER_ONLY" else float(config["add_fraction"]),
            "retest_depth": config.get("retest_depth_ATR"),
            "retest_window": config.get("retest_window"),
            "context_multiplier": context_multiplier,
            "control_add_lag_bars": control_directive.get("add_lag_bars") if control_id == "A3_RETEST_TIME_PERMUTED_MAIN_NULL" and isinstance(control_directive, Mapping) else None,
        })
    return events


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


__all__ = ["RetestResult", "breakout_magnitude", "breakout_population_key", "contract", "evaluate", "event_id", "parent_weights", "retest_state", "run_retest_state_machine"]
