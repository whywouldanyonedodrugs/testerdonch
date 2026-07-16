#!/usr/bin/env python3
"""Reusable signal-state primitives for point-in-time QLMG research.

The module deliberately does not know strategy economics. Raw signal generation
and event execution remain family-owned; this module owns immutable tape hashes,
PIT parent projections, and definition-local non-overlap state.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


SIGNAL_STATE_CONTRACT_VERSION = "signal_state_contract_v1_20260715"


class SignalStateContractError(ValueError):
    """Raised when a signal-state operation cannot satisfy the frozen contract."""


def _canonical_value(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        timestamp = value
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        return timestamp.isoformat()
    if isinstance(value, np.datetime64):
        return _canonical_value(pd.Timestamp(value))
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return format(number, ".17g")
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))) if isinstance(value, (set, frozenset)) else items
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_frame_hash(frame: pd.DataFrame, *, sort_fields: Sequence[str], columns: Sequence[str] | None = None) -> str:
    """Hash canonical sorted row content, never storage metadata or incidental order."""
    missing_sort = [field for field in sort_fields if field not in frame.columns]
    if missing_sort:
        raise SignalStateContractError("missing_sort_fields:" + ",".join(missing_sort))
    selected = list(columns) if columns is not None else sorted(frame.columns)
    missing = [field for field in selected if field not in frame.columns]
    if missing:
        raise SignalStateContractError("missing_hash_fields:" + ",".join(missing))
    if frame.empty:
        return stable_hash([])
    ordered = frame.sort_values(list(sort_fields), kind="mergesort", na_position="last")
    rows = [{field: _canonical_value(row[field]) for field in selected} for row in ordered.to_dict("records")]
    return stable_hash(rows)


def freeze_raw_signal_tape(
    raw_signals: pd.DataFrame,
    *,
    address_col: str = "raw_signal_address_hash",
    sort_fields: Sequence[str] = ("raw_policy_hash", "symbol", "entry_ts", "raw_signal_address_hash"),
) -> tuple[pd.DataFrame, str]:
    """Validate and freeze a parent-neutral raw tape with unique addresses."""
    required = {address_col, "symbol", "decision_ts", "entry_ts"}
    missing = sorted(required - set(raw_signals.columns))
    if missing:
        raise SignalStateContractError("missing_raw_signal_fields:" + ",".join(missing))
    blank = raw_signals[address_col].isna() | raw_signals[address_col].astype(str).str.strip().eq("")
    if bool(blank.any()):
        raise SignalStateContractError(f"blank_raw_signal_addresses={int(blank.sum())}")
    duplicates = raw_signals.duplicated(address_col, keep=False)
    if bool(duplicates.any()):
        raise SignalStateContractError(f"duplicate_raw_signal_addresses={int(duplicates.sum())}")
    frozen = raw_signals.sort_values(list(sort_fields), kind="mergesort").reset_index(drop=True).copy()
    content_hash = canonical_frame_hash(frozen, sort_fields=sort_fields)
    frozen["raw_signal_hash"] = content_hash
    frozen["signal_state_contract_version"] = SIGNAL_STATE_CONTRACT_VERSION
    return frozen, content_hash


def project_parent_policies(
    frozen_raw: pd.DataFrame,
    policies: Sequence[Mapping[str, Any]],
    *,
    is_allowed: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
    feature_ts_col: str = "parent_feature_ts",
) -> tuple[pd.DataFrame, str]:
    """Create stateless PIT parent-policy projections from one immutable raw tape."""
    required = {"raw_signal_address_hash", "raw_signal_hash", "decision_ts", feature_ts_col}
    missing = sorted(required - set(frozen_raw.columns))
    if missing:
        raise SignalStateContractError("missing_projection_source_fields:" + ",".join(missing))
    decision_ts = pd.to_datetime(frozen_raw["decision_ts"], utc=True, errors="coerce")
    feature_ts = pd.to_datetime(frozen_raw[feature_ts_col], utc=True, errors="coerce")
    invalid_ts = decision_ts.isna() | feature_ts.isna() | feature_ts.gt(decision_ts)
    if bool(invalid_ts.any()):
        raise SignalStateContractError(f"parent_projection_pit_violations={int(invalid_ts.sum())}")
    rows: list[dict[str, Any]] = []
    for policy in sorted(policies, key=lambda row: str(row.get("selected_key_policy_hash", ""))):
        policy_hash = str(policy.get("selected_key_policy_hash", "")).strip()
        if not policy_hash:
            raise SignalStateContractError("blank_selected_key_policy_hash")
        for raw in frozen_raw.to_dict("records"):
            if not is_allowed(raw, policy):
                continue
            row = {**raw, **dict(policy)}
            row["selected_key_policy_hash"] = policy_hash
            row["candidate_key"] = "SIGK_" + stable_hash({
                "selected_key_policy_hash": policy_hash,
                "raw_signal_address_hash": raw["raw_signal_address_hash"],
            })[:24]
            rows.append(row)
    projected = pd.DataFrame(rows)
    if projected.empty:
        projected = pd.DataFrame(columns=[*frozen_raw.columns, "selected_key_policy_hash", "candidate_key"])
    else:
        projected = projected.sort_values(
            ["selected_key_policy_hash", "symbol", "entry_ts", "candidate_key"], kind="mergesort"
        ).reset_index(drop=True)
        duplicates = projected.duplicated("candidate_key", keep=False)
        if bool(duplicates.any()):
            raise SignalStateContractError(f"duplicate_projected_candidate_keys={int(duplicates.sum())}")
    projection_hash = canonical_frame_hash(
        projected,
        sort_fields=("selected_key_policy_hash", "symbol", "entry_ts", "candidate_key"),
    )
    projected["projection_hash"] = projection_hash
    projected["signal_state_contract_version"] = SIGNAL_STATE_CONTRACT_VERSION
    return projected, projection_hash


def suppress_repeated_unresolved_setups(
    signals: pd.DataFrame,
    *,
    setup_id_col: str = "setup_sequence_id",
    order_fields: Sequence[str] = ("symbol", "decision_ts", "raw_signal_address_hash"),
) -> pd.DataFrame:
    """Keep one completion per mechanical setup, without any holding-period state."""
    required = {setup_id_col, *order_fields}
    missing = sorted(required - set(signals.columns))
    if missing:
        raise SignalStateContractError("missing_unresolved_setup_fields:" + ",".join(missing))
    ordered = signals.sort_values(list(order_fields), kind="mergesort")
    return ordered.drop_duplicates(setup_id_col, keep="first").reset_index(drop=True)


def simulate_definition_non_overlap(
    candidates: pd.DataFrame,
    definition: Mapping[str, Any],
    execute_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply non-overlap using actual exits and state local to one definition."""
    required = {"candidate_key", "symbol", "entry_ts"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise SignalStateContractError("missing_candidate_fields:" + ",".join(missing))
    definition_id = str(definition.get("definition_id", "")).strip()
    if not definition_id:
        raise SignalStateContractError("blank_definition_id")
    accepted: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    open_trade: dict[str, dict[str, Any]] = {}
    ordered = candidates.sort_values(["entry_ts", "symbol", "candidate_key"], kind="mergesort")
    for key in ordered.to_dict("records"):
        entry_ts = pd.Timestamp(key["entry_ts"])
        prior = open_trade.get(str(key["symbol"]))
        if prior is not None and entry_ts < pd.Timestamp(prior["exit_ts"]):
            skips.append({
                "definition_id": definition_id,
                "candidate_key": key["candidate_key"],
                "symbol": key["symbol"],
                "entry_ts": entry_ts,
                "prior_trade_id": prior["event_id"],
                "prior_entry_ts": prior["entry_ts"],
                "prior_actual_exit_ts": prior["exit_ts"],
                "skip_reason": "same_symbol_same_definition_position_actually_open",
            })
            continue
        event, exclusion = execute_fn(key, definition)
        if exclusion is not None:
            exclusions.append({**dict(exclusion), "definition_id": definition_id, "candidate_key": key["candidate_key"]})
            continue
        if event is None:
            raise SignalStateContractError("executor_returned_no_event_or_exclusion")
        event_row = {**dict(key), **dict(event), "definition_id": definition_id}
        if "exit_ts" not in event_row:
            raise SignalStateContractError("accepted_event_missing_exit_ts")
        exit_ts = pd.Timestamp(event_row["exit_ts"])
        if exit_ts < entry_ts:
            raise SignalStateContractError("accepted_event_exit_before_entry")
        event_row.setdefault("event_id", "SIGE_" + stable_hash({
            "definition_id": definition_id,
            "candidate_key": key["candidate_key"],
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
        })[:24])
        accepted.append(event_row)
        open_trade[str(key["symbol"])] = event_row
    return pd.DataFrame(accepted), pd.DataFrame(skips), pd.DataFrame(exclusions)


def simulate_all_definitions(
    projected: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    execute_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate each definition with a newly allocated, isolated state machine."""
    accepted_frames: list[pd.DataFrame] = []
    skip_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    for definition in sorted(definitions, key=lambda row: str(row.get("definition_id", ""))):
        policy_hash = str(definition.get("selected_key_policy_hash", "")).strip()
        selected = projected[projected["selected_key_policy_hash"].astype(str).eq(policy_hash)].copy()
        accepted, skips, exclusions = simulate_definition_non_overlap(selected, definition, execute_fn)
        if not accepted.empty:
            accepted_frames.append(accepted)
        if not skips.empty:
            skip_frames.append(skips)
        if not exclusions.empty:
            exclusion_frames.append(exclusions)
    return (
        pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame(),
        pd.concat(skip_frames, ignore_index=True) if skip_frames else pd.DataFrame(),
        pd.concat(exclusion_frames, ignore_index=True) if exclusion_frames else pd.DataFrame(),
    )


def build_rankable_contract_manifest(
    *,
    raw_signals: pd.DataFrame,
    projected: pd.DataFrame,
    accepted: pd.DataFrame,
    skips: pd.DataFrame,
    exclusions: pd.DataFrame,
    eligible_definition_rows: int,
) -> dict[str, Any]:
    """Build the evidence manifest required before a run can claim rankable output."""
    raw_hashes = set(raw_signals.get("raw_signal_hash", pd.Series(dtype=str)).dropna().astype(str))
    projection_hashes = set(projected.get("projection_hash", pd.Series(dtype=str)).dropna().astype(str))
    if len(raw_hashes) != 1 or len(projection_hashes) != 1:
        raise SignalStateContractError("raw_or_projection_tape_not_frozen")
    accepted_hash = canonical_frame_hash(
        accepted,
        sort_fields=("definition_id", "symbol", "entry_ts", "candidate_key"),
    )
    reconciled = int(eligible_definition_rows) == len(accepted) + len(skips) + len(exclusions)
    return {
        "signal_state_contract_version": SIGNAL_STATE_CONTRACT_VERSION,
        "raw_signal_hash": next(iter(raw_hashes)),
        "projection_hash": next(iter(projection_hashes)),
        "accepted_trade_hash": accepted_hash,
        "raw_signal_count": len(raw_signals),
        "eligible_definition_rows": int(eligible_definition_rows),
        "accepted_trade_count": len(accepted),
        "non_overlap_skip_count": len(skips),
        "outcome_exclusion_count": len(exclusions),
        "raw_tape_frozen_before_outcomes": True,
        "projection_frozen_before_outcomes": True,
        "non_overlap_reconciled": reconciled,
        "no_mutable_state_shared_across_definitions": True,
    }
