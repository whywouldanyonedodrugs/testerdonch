from __future__ import annotations

import math
from datetime import timedelta
from typing import Any, Mapping, Sequence

from ..canonical import canonical_hash
from ..engine_types import FamilyInput, SignalBar, require_contiguous_5m, require_contiguous_daily
from .common import EngineInputError, log_return, path_smoothness, percentile_from_population, percentile_from_prevalidated_sorted, require_utc, sample_standard_deviation, wilder_atr


ENGINE_ID = "a1_compression_engine_v1"


def impulse_population_key(window: str, scope: str, side: int) -> str:
    return f"A1_impulse:window={window}:scope={scope}:side={side}"


def contraction_population_key(base_duration: str, baseline: str, scope: str) -> str:
    return f"A1_contraction:base={base_duration}:baseline={baseline}:scope={scope}"


def smoothness_population_key(base_duration: str, scope: str) -> str:
    return f"A1_smoothness:base={base_duration}:scope={scope}"


def realized_volatility(closes: Sequence[float]) -> float:
    returns = [log_return(left, right) for left, right in zip(closes, closes[1:])]
    return sample_standard_deviation(returns)


def features(impulse_closes: Sequence[float], base_closes: Sequence[float], baseline_closes: Sequence[float], side: int) -> dict[str, float]:
    if side not in (-1, 1):
        raise EngineInputError("A1 side must be long or short")
    if len(baseline_closes) not in {len(base_closes), 5 * len(base_closes)}:
        raise EngineInputError("contraction baseline must be equal-duration or trailing-five-times duration")
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


def _bar_count(duration: str) -> int:
    units = {"h": 12, "d": 288}
    try:
        return int(duration[:-1]) * units[duration[-1]]
    except (KeyError, ValueError) as exc:
        raise EngineInputError(f"unsupported A1 clock: {duration}") from exc


def _population(frame: FamilyInput, name: str) -> Sequence[float]:
    try:
        population = frame.threshold_populations[name]
    except KeyError as exc:
        raise EngineInputError(f"missing threshold population: {name}") from exc
    population.validate(pooled="global" in name or "liquidity_decile" in name, decision_ts=frame.decision_ts)
    return population.values


def _decision_contiguous_segment(frame: FamilyInput) -> tuple[tuple[SignalBar, ...], bool]:
    """Return the gap-free component containing the registered decision.

    Earlier components cannot supply rolling history after a gap.  A later
    component cannot supply an entry/exit for the decision.  The boolean marks
    whether history was rebuilt after a preceding gap.
    """
    bars = frame.five_minute_bars
    decision_index = next((index for index, bar in enumerate(bars) if require_utc(bar.close_ts) == require_utc(frame.decision_ts)), None)
    if decision_index is None:
        return tuple(bars), False
    start = decision_index
    while start > 0 and require_utc(bars[start].open_ts) - require_utc(bars[start - 1].open_ts) == timedelta(minutes=5):
        start -= 1
    end = decision_index + 1
    while end < len(bars) and require_utc(bars[end].open_ts) - require_utc(bars[end - 1].open_ts) == timedelta(minutes=5):
        end += 1
    return tuple(bars[start:end]), start > 0


def rearm_ready(sides: Sequence[int], owning_side: int | None, percentiles: Mapping[int, float]) -> bool:
    """Strict q50 reset rule used by the symmetric A1 state machine."""
    if owning_side is not None:
        return float(percentiles.get(owning_side, math.inf)) < 0.50
    return all(float(percentiles.get(side, math.inf)) < 0.50 for side in sides)


def evaluate(frame: FamilyInput, config: Mapping[str, Any], *, control_id: str | None = None, control_directive: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run the complete A1 episode state machine from completed five-minute bars."""
    frame.validate()
    frame.require_pit_top_n(int(config["PIT_liquidity_top_n"]))
    persisted_state = frame.metadata.get("a1_persistent_state")
    if frame.metadata.get("production_input") is True:
        from ..a1_state import A1PersistentState

        if not isinstance(persisted_state, Mapping):
            raise EngineInputError("production A1 frame lacks its persisted state")
        persisted_state = A1PersistentState(**persisted_state)
        persisted_state.validate()
        if persisted_state.state in {"episode_owned_long", "episode_owned_short", "base", "confirmation"}:
            raise EngineInputError("production A1 frame begins inside an active episode without a complete episode checkpoint")
    bars, rebuilt_after_gap = _decision_contiguous_segment(frame)
    require_contiguous_5m(bars)
    impulse_n = _bar_count(str(config["impulse_window"]))
    base_n = _bar_count(str(config["base_duration"]))
    baseline_n = base_n if config["contraction_baseline"] == "adjacent_equal_duration" else 5 * base_n
    if control_id == "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL":
        if not isinstance(control_directive, Mapping) or control_directive.get("allocator") != "matched_pseudo_event_allocator_v2":
            raise EngineInputError("A1 matched pseudo event lacks its derived allocator directive")
        side = control_directive.get("side")
        if side not in (-1, 1) or control_directive.get("matched_decision_ts") != require_utc(frame.decision_ts):
            raise EngineInputError("A1 matched pseudo event directive is not bound to this frame")
        entry_index = next((i for i, bar in enumerate(bars) if require_utc(bar.open_ts) >= require_utc(frame.decision_ts)), None)
        decision_index = next((i for i, bar in enumerate(bars) if require_utc(bar.close_ts) == require_utc(frame.decision_ts)), None)
        if entry_index is None or decision_index is None or decision_index < base_n:
            return []
        base_bars = bars[decision_index - base_n + 1:decision_index + 1]
        atr_window = int(config["ATR_window_days"] or 20)
        available_daily = [bar for bar in frame.daily_bars if require_utc(bar.close_ts) < require_utc(frame.decision_ts)]
        atr_daily = available_daily[-(atr_window + 1):]
        require_contiguous_daily(atr_daily)
        atr = wilder_atr([bar.high for bar in atr_daily], [bar.low for bar in atr_daily], [bar.close for bar in atr_daily], atr_window) if str(config["exit"]).startswith("ATR_") else None
        return [{
            "event_id": canonical_hash({"control": control_id, "parent_event_id": control_directive["parent_event_id"], "symbol": frame.symbol, "decision_ts": require_utc(frame.decision_ts).isoformat()}),
            "side": int(side), "decision_ts": require_utc(frame.decision_ts), "entry_index": entry_index,
            "structural_level": (min(bar.close for bar in base_bars) if side == 1 else max(bar.close for bar in base_bars)) if config["exit"] == "base_failure" else None,
            "atr": atr, "context_multiplier": 1.0, "matched_parent_event_id": control_directive["parent_event_id"],
        }]
    if control_id == "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL":
        raise AssertionError("matched pseudo control returned through the direct branch")
    else:
        sides = (1, -1) if config["direction"] == "symmetric" else (1 if config["direction"] == "long" else -1,)
    threshold = int(str(config["impulse_rank_min"])[1:]) / 100.0
    prepared_populations: dict[str, Sequence[float]] = {}

    def prepared_population(name: str) -> Sequence[float]:
        if name not in prepared_populations:
            raw = _population(frame, name)
            prepared_populations[name] = raw if callable(getattr(raw, "weak_percentile", None)) else tuple(sorted(float(value) for value in raw))
        return prepared_populations[name]

    if persisted_state is None:
        armed = {side: not rebuilt_after_gap for side in sides}
        owning_side: int | None = None
    else:
        armed = {side: persisted_state.state == "armed" and not rebuilt_after_gap for side in sides}
        owning_side = persisted_state.owner
    if rebuilt_after_gap:
        recorded_owner = (
            persisted_state.owner
            if persisted_state is not None and hasattr(persisted_state, "owner")
            else frame.metadata.get("a1_pre_gap_owning_side")
        )
        if recorded_owner not in (*sides, None):
            raise EngineInputError("A1 pre-gap owning side is absent or invalid")
        if "a1_pre_gap_owning_side" not in frame.metadata and persisted_state is None:
            raise EngineInputError("A1 gap rebuild lacks the persisted owning-side state")
        owning_side = None if recorded_owner is None else int(recorded_owner)
    history_restore_index: int | None = None
    previous_percentile = {side: -math.inf for side in sides}
    episodes: list[dict[str, Any]] = []
    minimum_index = max(impulse_n, baseline_n)
    index = minimum_index
    while index + base_n + 4 < len(bars):
        if history_restore_index is not None and index > history_restore_index:
            for candidate_side in sides:
                armed[candidate_side] = True
            owning_side = None
            history_restore_index = None
        side_candidates: list[tuple[float, int, float]] = []
        percentiles: dict[int, float] = {}
        scores: dict[int, float] = {}
        for side in sides:
            impulse = [bar.close for bar in bars[index - impulse_n:index + 1]]
            score = side * log_return(impulse[0], impulse[-1])
            population_name = impulse_population_key(str(config["impulse_window"]), str(config["impulse_rank_scope"]), side)
            percentile, passes = percentile_from_prevalidated_sorted(score, prepared_population(population_name), str(config["impulse_rank_min"]))
            percentiles[side] = percentile
            scores[side] = score
            previous = previous_percentile[side]
            previous_percentile[side] = percentile
            if not armed[side]:
                continue
            pseudo_match = control_id == "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL" and require_utc(bars[index].close_ts) == require_utc(frame.decision_ts)
            crossed = pseudo_match or (passes and previous < threshold)
            if crossed:
                side_candidates.append((percentile - threshold, side, score))
        if not any(armed.values()):
            if persisted_state is not None and persisted_state.state == "cooldown" and persisted_state.cooldown_until is not None and require_utc(bars[index].close_ts) < require_utc(persisted_state.cooldown_until):
                index += 1
                continue
            if rearm_ready(sides, owning_side, percentiles):
                history_restore_index = index
            index += 1
            continue
        if len(side_candidates) == 2 and math.isclose(side_candidates[0][0], side_candidates[1][0], rel_tol=0.0, abs_tol=0.0):
            for side in sides:
                armed[side] = False
            index += 1
            continue
        if not side_candidates:
            index += 1
            continue
        _, side, impulse_score = max(side_candidates, key=lambda item: item[0])
        for blocked_side in sides:
            armed[blocked_side] = False
        owning_side = side
        history_restore_index = None
        impulse_bars = bars[index - impulse_n:index + 1]
        extreme = max(bar.close for bar in impulse_bars) if side == 1 else min(bar.close for bar in impulse_bars)
        base_start = index + 1
        base_end = base_start + base_n
        base_bars = bars[base_start:base_end]
        if len(base_bars) != base_n:
            break
        if any(side * (bar.close - extreme) > 0 for bar in base_bars):
            index = base_end
            continue
        baseline_bars = bars[base_start - baseline_n:base_start]
        if len(baseline_bars) != baseline_n:
            index += 1
            continue
        computed = features(
            [bar.close for bar in impulse_bars],
            [bar.close for bar in base_bars],
            [bar.close for bar in baseline_bars],
            side,
        )
        contraction_ok = True
        if config["contraction_rank_max"] != "none" and control_id not in {"A1_CONTRACTION_REMOVED", "A1_PRICE_ONLY_IMPULSE"}:
            contraction_percentile, _ = percentile_from_prevalidated_sorted(
                computed["contraction_ratio"],
                prepared_population(contraction_population_key(str(config["base_duration"]), str(config["contraction_baseline"]), str(config["shape_rank_scope"]))),
            )
            contraction_ok = contraction_percentile <= int(str(config["contraction_rank_max"])[1:]) / 100.0
        smoothness_ok = True
        if config["smoothness_rank_min"] != "none" and control_id not in {"A1_SMOOTHNESS_REMOVED", "A1_PRICE_ONLY_IMPULSE"}:
            _, smoothness_ok = percentile_from_prevalidated_sorted(
                computed["base_smoothness"],
                prepared_population(smoothness_population_key(str(config["base_duration"]), str(config["shape_rank_scope"]))),
                str(config["smoothness_rank_min"]),
            )
        if not contraction_ok or not smoothness_ok:
            index = base_end
            continue
        first_confirmation = base_end
        confirmation = str(config["confirmation"])
        if confirmation == "one_close":
            confirmation_indices = (first_confirmation,)
        elif confirmation == "two_closes":
            confirmation_indices = (first_confirmation, first_confirmation + 1)
        elif confirmation == "close_plus_bounded_15m_delay":
            confirmation_indices = (first_confirmation, first_confirmation + 3)
        else:
            raise EngineInputError(f"unsupported A1 confirmation: {confirmation}")
        if max(confirmation_indices) >= len(bars):
            break
        confirmation_bars = [bars[item] for item in confirmation_indices]
        final_confirmation_ts = require_utc(confirmation_bars[-1].close_ts)
        if final_confirmation_ts > require_utc(frame.decision_ts):
            break
        if confirmation == "close_plus_bounded_15m_delay" and require_utc(confirmation_bars[1].close_ts) - require_utc(confirmation_bars[0].close_ts) != timedelta(minutes=15):
            raise EngineInputError("A1 delayed confirmation is not exactly fifteen minutes")
        if not all(side * (bar.close - extreme) > 0 for bar in confirmation_bars):
            index = max(confirmation_indices) + 1
            continue
        if final_confirmation_ts != require_utc(frame.decision_ts):
            index = max(confirmation_indices) + 1
            continue
        final_confirmation = confirmation_bars[-1]
        entry_index = max(confirmation_indices) + 1
        if entry_index >= len(bars) or require_utc(bars[entry_index].open_ts) - require_utc(final_confirmation.close_ts) > timedelta(minutes=10):
            index = entry_index
            continue
        structural_level = min(bar.close for bar in base_bars) if side == 1 else max(bar.close for bar in base_bars)
        atr_window = int(config["ATR_window_days"] or 20)
        available_daily = [bar for bar in frame.daily_bars if require_utc(bar.close_ts) < final_confirmation_ts]
        daily = available_daily[-(atr_window + 1):]
        require_contiguous_daily(daily)
        atr = wilder_atr([bar.high for bar in daily], [bar.low for bar in daily], [bar.close for bar in daily], atr_window) if str(config["exit"]).startswith("ATR_") else None
        context_multiplier = 1.0
        if config["context_overlay"] != "none" and control_id not in {"A1_CONTEXT_REMOVED", "A1_PRICE_ONLY_IMPULSE", "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL"}:
            from .a2_context import named_context_multiplier
            context_multiplier = named_context_multiplier(frame, str(config["context_overlay"]), side)
        episodes.append({
            "event_id": event_id(frame.symbol, side, require_utc(impulse_bars[0].close_ts).isoformat(), require_utc(impulse_bars[-1].close_ts).isoformat(), require_utc(base_bars[0].close_ts).isoformat(), require_utc(base_bars[-1].close_ts).isoformat(), require_utc(final_confirmation.close_ts).isoformat()),
            "side": side,
            "decision_ts": require_utc(final_confirmation.close_ts),
            "entry_index": entry_index,
            "structural_level": structural_level if config["exit"] == "base_failure" else None,
            "atr": atr,
            "impulse_score": impulse_score,
            "features": computed,
            "context_multiplier": context_multiplier,
            "a1_state_generation": getattr(persisted_state, "state_generation", None),
        })
        index = entry_index
    return episodes


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


__all__ = ["confirmation_pass", "contract", "contraction_population_key", "evaluate", "event_id", "features", "impulse_population_key", "rearm_ready", "smoothness_population_key"]
