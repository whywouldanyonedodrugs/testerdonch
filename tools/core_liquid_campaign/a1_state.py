from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Mapping, Sequence

from .family_engines.common import EngineInputError, require_utc


A1_STATES = frozenset({
    "armed",
    "episode_owned_long",
    "episode_owned_short",
    "base",
    "confirmation",
    "disarmed",
    "history_rebuild",
    "cooldown",
    "terminal_episode_reason",
})
ACTIVE_STATES = frozenset({"episode_owned_long", "episode_owned_short", "base", "confirmation"})


@dataclass(frozen=True)
class A1PersistentState:
    state: str
    owner: int | None
    cooldown_until: datetime | None
    last_valid_ts: datetime | None
    state_generation: int
    terminal_episode_reason: str | None = None
    rearm_eligible_after_ts: datetime | None = None

    def validate(self) -> None:
        if self.state not in A1_STATES:
            raise EngineInputError("unknown A1 persistent state")
        if self.owner not in (-1, 1, None):
            raise EngineInputError("A1 persistent owner is invalid")
        if self.state == "episode_owned_long" and self.owner != 1:
            raise EngineInputError("A1 long-owned state lacks its long owner")
        if self.state == "episode_owned_short" and self.owner != -1:
            raise EngineInputError("A1 short-owned state lacks its short owner")
        if self.state == "cooldown" and self.cooldown_until is None:
            raise EngineInputError("A1 cooldown state lacks its deadline")
        if self.state == "terminal_episode_reason" and not self.terminal_episode_reason:
            raise EngineInputError("A1 terminal state lacks its reason")
        if self.state_generation < 0:
            raise EngineInputError("A1 state generation is negative")
        for value in (self.cooldown_until, self.last_valid_ts, self.rearm_eligible_after_ts):
            if value is not None:
                require_utc(value)

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def initial_state() -> A1PersistentState:
    return A1PersistentState("history_rebuild", None, None, None, 0)


def _next(current: A1PersistentState, timestamp: datetime, **changes: object) -> A1PersistentState:
    timestamp = require_utc(timestamp)
    if current.last_valid_ts is not None and timestamp <= require_utc(current.last_valid_ts):
        raise EngineInputError("A1 transition timestamp is not strictly increasing")
    result = replace(
        current,
        last_valid_ts=timestamp,
        state_generation=current.state_generation + 1,
        **changes,
    )
    result.validate()
    return result


def transition(
    state: A1PersistentState,
    *,
    timestamp: datetime,
    action: str,
    percentiles: Mapping[int, float] | None = None,
    required_sides: Sequence[int] = (1, -1),
    side: int | None = None,
    cooldown_until: datetime | None = None,
    terminal_reason: str | None = None,
) -> A1PersistentState:
    """Apply one persisted A1 transition; no implicit same-timestamp rearm."""
    state.validate()
    timestamp = require_utc(timestamp)
    if action == "gap":
        reason = "temporal_gap" if state.state in ACTIVE_STATES else state.terminal_episode_reason
        return _next(
            state,
            timestamp,
            state="history_rebuild",
            terminal_episode_reason=reason,
            rearm_eligible_after_ts=None,
        )
    if action == "history_complete":
        if state.state != "history_rebuild":
            raise EngineInputError("A1 history completion outside history_rebuild")
        return _next(state, timestamp, state="disarmed", rearm_eligible_after_ts=timestamp)
    if action == "rearm":
        if state.state not in {"disarmed", "terminal_episode_reason"}:
            raise EngineInputError("A1 rearm outside a disarmed state")
        if state.rearm_eligible_after_ts is not None and timestamp <= require_utc(state.rearm_eligible_after_ts):
            raise EngineInputError("A1 cannot rearm on the history-restoration timestamp")
        values = percentiles or {}
        if state.owner is None:
            ready = all(float(values.get(candidate, float("inf"))) < 0.50 for candidate in required_sides)
        else:
            ready = float(values.get(state.owner, float("inf"))) < 0.50
        if not ready:
            return _next(state, timestamp)
        if state.cooldown_until is not None and timestamp < require_utc(state.cooldown_until):
            return _next(state, timestamp, state="cooldown")
        return _next(
            state,
            timestamp,
            state="armed",
            owner=None,
            cooldown_until=None,
            terminal_episode_reason=None,
            rearm_eligible_after_ts=None,
        )
    if action == "cooldown_expired":
        if state.state != "cooldown" or state.cooldown_until is None or timestamp < require_utc(state.cooldown_until):
            raise EngineInputError("A1 cooldown has not expired")
        return _next(state, timestamp, state="disarmed", rearm_eligible_after_ts=timestamp)
    if action == "trigger":
        if state.state != "armed" or side not in (-1, 1):
            raise EngineInputError("A1 trigger is outside armed state or lacks a side")
        return _next(state, timestamp, state="episode_owned_long" if side == 1 else "episode_owned_short", owner=side)
    if action == "base":
        if state.state not in {"episode_owned_long", "episode_owned_short"}:
            raise EngineInputError("A1 base transition lacks an owned episode")
        return _next(state, timestamp, state="base")
    if action == "confirmation":
        if state.state != "base":
            raise EngineInputError("A1 confirmation transition lacks a base")
        return _next(state, timestamp, state="confirmation")
    if action == "episode_terminal":
        if state.state not in ACTIVE_STATES:
            raise EngineInputError("A1 terminal transition lacks an active episode")
        reason = str(terminal_reason or "completed")
        return _next(
            state,
            timestamp,
            state="terminal_episode_reason",
            terminal_episode_reason=reason,
            cooldown_until=cooldown_until,
            rearm_eligible_after_ts=timestamp,
        )
    raise EngineInputError(f"unknown A1 transition action: {action}")


__all__ = ["A1PersistentState", "A1_STATES", "ACTIVE_STATES", "initial_state", "transition"]
