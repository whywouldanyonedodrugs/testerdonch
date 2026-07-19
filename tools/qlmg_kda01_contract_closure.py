"""Outcome-free KDA01 v2 inference-cluster and execution-availability contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from tools.qlmg_kda01_v2 import TRANSLATION_ID
from tools.qlmg_kraken_derivatives_state import PROTECTED_START, TRAIN_START, stable_hash


CONTRACT_VERSION = "kda01_level3_contract_v2_20260719"
GRID = pd.Timedelta(minutes=5)
MAX_ENTRY_DELAY = pd.Timedelta(minutes=10)
MAX_EXIT_DELAY = pd.Timedelta(minutes=10)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("KDA01 closure timestamp must be timezone-aware")
    return timestamp.tz_convert("UTC")


def market_cluster_identity(attempt: str, parent_onset: Any) -> tuple[str, str, str, pd.Timestamp]:
    """Return date, daily identity, block identity, and six-hour UTC block."""
    if attempt not in {"primary", "robustness"}:
        raise ValueError("unsupported KDA01 attempt")
    onset = _utc_timestamp(parent_onset)
    if not (TRAIN_START <= onset < PROTECTED_START):
        raise ValueError("non-rankable KDA01 parent onset")
    onset_date = onset.strftime("%Y-%m-%d")
    block = onset.floor("6h")
    day_id = stable_hash({
        "translation_id": TRANSLATION_ID,
        "attempt": attempt,
        "parent_onset_utc_date": onset_date,
    })
    block_id = stable_hash({
        "translation_id": TRANSLATION_ID,
        "attempt": attempt,
        "parent_onset_utc_6h_block": block.isoformat(),
    })
    return onset_date, day_id, block_id, block


def attach_market_cluster_identity(events: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Join immutable parent onsets and add cross-symbol cluster identities."""
    required_events = {"event_id", "parent_episode_id", "attempt", "symbol", "decision_ts"}
    required_episodes = {"parent_episode_id", "attempt", "parent_onset_ts"}
    if missing := sorted(required_events - set(events.columns)):
        raise ValueError(f"missing event identity inputs: {missing}")
    if missing := sorted(required_episodes - set(episodes.columns)):
        raise ValueError(f"missing parent identity inputs: {missing}")
    if episodes.parent_episode_id.duplicated().any():
        raise ValueError("duplicate parent episode identity")
    parent = episodes[["parent_episode_id", "attempt", "parent_onset_ts"]].rename(
        columns={"attempt": "parent_attempt"}
    )
    out = events.merge(parent, on="parent_episode_id", how="left", validate="many_to_one")
    if out.parent_onset_ts.isna().any() or not out.attempt.eq(out.parent_attempt).all():
        raise ValueError("event-to-parent identity mismatch")
    rows = [market_cluster_identity(row.attempt, row.parent_onset_ts) for row in out.itertuples()]
    out["parent_onset_utc_date"] = [row[0] for row in rows]
    out["market_day_cluster_id"] = [row[1] for row in rows]
    out["market_6h_cluster_id"] = [row[2] for row in rows]
    out["parent_onset_utc_6h_block"] = [row[3] for row in rows]
    out = out.drop(columns="parent_attempt").sort_values(
        ["attempt", "parent_onset_ts", "symbol", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(out) != len(events) or out.event_id.duplicated().any():
        raise ValueError("cluster identity row reconciliation failed")
    return out


def expected_next_open(decision_ts: Any) -> pd.Timestamp:
    """First five-minute UTC grid timestamp strictly after decision."""
    decision = _utc_timestamp(decision_ts)
    return decision.floor("5min") + GRID


def normalized_bar_times(values: Iterable[Any]) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(list(values), utc=True)).sort_values().unique()
    if len(times) and (times[0] < TRAIN_START or times[-1] >= PROTECTED_START):
        raise ValueError("non-rankable timestamp entered execution availability")
    return times


def _first_index(times: pd.DatetimeIndex, target: pd.Timestamp, *, strictly_after: bool) -> int:
    return int(times.searchsorted(target, side="right" if strictly_after else "left"))


@dataclass(frozen=True)
class ExecutionAvailability:
    expected_entry_ts: pd.Timestamp
    entry_ts: pd.Timestamp | None
    entry_delay_minutes: float | None
    exit_target_ts: pd.Timestamp | None
    exit_ts: pd.Timestamp | None
    exit_delay_minutes: float | None
    status: str


def locate_execution_availability(
    decision_ts: Any, timeout_hours: int, bar_times: pd.DatetimeIndex
) -> ExecutionAvailability:
    """Resolve entry/exit timestamps without opening any price column."""
    if timeout_hours not in {1, 6}:
        raise ValueError("unsupported frozen KDA01 timeout")
    decision = _utc_timestamp(decision_ts)
    expected = expected_next_open(decision)
    entry_index = _first_index(bar_times, decision, strictly_after=True)
    if entry_index >= len(bar_times):
        return ExecutionAvailability(expected, None, None, None, None, None, "missing_entry_bar")
    entry = pd.Timestamp(bar_times[entry_index])
    entry_delay = (entry - expected).total_seconds() / 60
    if entry_delay > MAX_ENTRY_DELAY.total_seconds() / 60:
        return ExecutionAvailability(expected, entry, entry_delay, None, None, None, "entry_delay_exceeded")
    target = entry + pd.Timedelta(hours=timeout_hours)
    if target >= PROTECTED_START:
        return ExecutionAvailability(expected, entry, entry_delay, target, None, None, "protected_boundary_crossing")
    exit_index = _first_index(bar_times, target, strictly_after=False)
    if exit_index >= len(bar_times):
        return ExecutionAvailability(expected, entry, entry_delay, target, None, None, "missing_exit_bar")
    exit_ts = pd.Timestamp(bar_times[exit_index])
    exit_delay = (exit_ts - target).total_seconds() / 60
    status = "eligible" if exit_delay <= MAX_EXIT_DELAY.total_seconds() / 60 else "exit_delay_exceeded"
    return ExecutionAvailability(expected, entry, entry_delay, target, exit_ts, exit_delay, status)


def execution_records(
    events: pd.DataFrame,
    definitions: pd.DataFrame,
    bars_by_symbol: Mapping[str, pd.DatetimeIndex],
) -> pd.DataFrame:
    """Fan out events to frozen definitions, then apply definition-local non-overlap."""
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
            availability = locate_execution_availability(event.decision_ts, int(definition.timeout_hours), bars)
            rows.append({
                "definition_id": definition.definition_id,
                "definition_contract_hash": definition.definition_contract_hash,
                "branch_id": event.branch_id,
                "event_id": event.event_id,
                "economic_address": event.economic_address,
                "symbol": event.symbol,
                "decision_ts": _utc_timestamp(event.decision_ts),
                "year": _utc_timestamp(event.decision_ts).year,
                **asdict(availability),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["definition_id", "symbol", "entry_ts", "decision_ts", "event_id"], kind="mergesort").reset_index(drop=True)
    result["accepted"] = False
    result["prior_event_id"] = ""
    result["prior_exit_ts"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    for (_, _), indices in result.groupby(["definition_id", "symbol"], sort=False).groups.items():
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
        raise ValueError("duplicate definition-event execution record")
    return result


def frozen_contract_hash(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("level3_contract_hash", None)
    return stable_hash(payload)
