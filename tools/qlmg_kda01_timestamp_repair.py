"""Pure KDA01 execution-timestamp repair primitives."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping

import pandas as pd

from tools.qlmg_kda01_contract_closure import ExecutionAvailability
from tools.qlmg_kraken_derivatives_state import PROTECTED_START, TRAIN_START

GRID = pd.Timedelta(minutes=5)
MAX_ENTRY_DELAY = pd.Timedelta(minutes=10)
MAX_EXIT_DELAY = pd.Timedelta(minutes=10)


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def normalized_bar_times(values: Iterable[Any]) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(list(values), utc=True)).sort_values().unique()
    if len(times) and (times[0] < TRAIN_START or times[-1] >= PROTECTED_START):
        raise ValueError("non-rankable timestamp entered repaired execution availability")
    return times


def expected_entry_at_or_after(decision_ts: Any) -> pd.Timestamp:
    """First five-minute grid timestamp at or after causal availability."""
    return _utc(decision_ts).ceil("5min")


def locate_repaired_execution(
    decision_ts: Any, timeout_hours: int, bar_times: pd.DatetimeIndex
) -> ExecutionAvailability:
    if timeout_hours not in {1, 6}:
        raise ValueError("unsupported frozen KDA01 timeout")
    decision = _utc(decision_ts)
    expected = expected_entry_at_or_after(decision)
    entry_index = int(bar_times.searchsorted(decision, side="left"))
    if entry_index >= len(bar_times):
        return ExecutionAvailability(expected, None, None, None, None, None, "missing_entry_bar")
    entry = pd.Timestamp(bar_times[entry_index])
    entry_delay = (entry - expected).total_seconds() / 60
    if entry_delay > MAX_ENTRY_DELAY.total_seconds() / 60:
        return ExecutionAvailability(expected, entry, entry_delay, None, None, None, "entry_delay_exceeded")
    target = entry + pd.Timedelta(hours=timeout_hours)
    if target >= PROTECTED_START:
        return ExecutionAvailability(expected, entry, entry_delay, target, None, None, "protected_boundary_crossing")
    exit_index = int(bar_times.searchsorted(target, side="left"))
    if exit_index >= len(bar_times):
        return ExecutionAvailability(expected, entry, entry_delay, target, None, None, "missing_exit_bar")
    exit_ts = pd.Timestamp(bar_times[exit_index])
    exit_delay = (exit_ts - target).total_seconds() / 60
    status = "eligible" if exit_delay <= MAX_EXIT_DELAY.total_seconds() / 60 else "exit_delay_exceeded"
    return ExecutionAvailability(expected, entry, entry_delay, target, exit_ts, exit_delay, status)


def repaired_execution_records(
    events: pd.DataFrame,
    definitions: pd.DataFrame,
    bars_by_symbol: Mapping[str, pd.DatetimeIndex],
) -> pd.DataFrame:
    """Apply repaired availability and unchanged definition-local actual-exit overlap."""
    required_events = {"event_id", "economic_address", "branch_id", "symbol", "decision_ts"}
    required_definitions = {"definition_id", "definition_contract_hash", "branch_id", "timeout_hours"}
    if missing := sorted(required_events - set(events.columns)):
        raise ValueError(f"missing execution event inputs: {missing}")
    if missing := sorted(required_definitions - set(definitions.columns)):
        raise ValueError(f"missing execution definition inputs: {missing}")
    rows: list[dict[str, Any]] = []
    for definition in definitions.sort_values("definition_id", kind="mergesort").itertuples():
        branch_events = events.loc[events.branch_id.eq(definition.branch_id)].sort_values(
            ["symbol", "decision_ts", "event_id"], kind="mergesort"
        )
        for event in branch_events.itertuples():
            bars = bars_by_symbol.get(event.symbol, pd.DatetimeIndex([], tz="UTC"))
            availability = locate_repaired_execution(event.decision_ts, int(definition.timeout_hours), bars)
            rows.append({
                "definition_id": definition.definition_id,
                "definition_contract_hash": definition.definition_contract_hash,
                "branch_id": event.branch_id,
                "event_id": event.event_id,
                "economic_address": event.economic_address,
                "symbol": event.symbol,
                "decision_ts": _utc(event.decision_ts),
                "year": _utc(event.decision_ts).year,
                **asdict(availability),
            })
    result = pd.DataFrame(rows).sort_values(
        ["definition_id", "symbol", "entry_ts", "decision_ts", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    result["accepted"] = False
    result["prior_event_id"] = ""
    result["prior_exit_ts"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    for indices in result.groupby(["definition_id", "symbol"], sort=False).groups.values():
        open_until: pd.Timestamp | None = None
        prior_event = ""
        for index in indices:
            if result.at[index, "status"] != "eligible":
                continue
            entry = pd.Timestamp(result.at[index, "entry_ts"])
            if open_until is not None and entry < open_until:
                result.at[index, "status"] = "actual_position_overlap"
                result.at[index, "prior_event_id"] = prior_event
                result.at[index, "prior_exit_ts"] = open_until
                continue
            result.at[index, "accepted"] = True
            open_until = pd.Timestamp(result.at[index, "exit_ts"])
            prior_event = str(result.at[index, "event_id"])
    if result.duplicated(["definition_id", "event_id"]).any():
        raise ValueError("duplicate repaired definition-event record")
    return result
