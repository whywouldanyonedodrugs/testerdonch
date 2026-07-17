#!/usr/bin/env python3
"""Outcome-free machine contract for the frozen C01 Level-3 run."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any

import numpy as np


TRAIN_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2026, 1, 1, tzinfo=timezone.utc)
PRIMARY_MODEL = "btc_eth_ols_daily_v1"
ROBUSTNESS_MODEL = "btc_only_ols_daily_v1"
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_RESAMPLES = 10_000

BRANCHES = (
    ("positive_smooth_long", "long", "smooth", "positive", "onset_next_open"),
    ("negative_smooth_short", "short", "smooth", "negative", "onset_next_open"),
    (
        "positive_jump_completed_failure_short", "short", "jump_dominated", "positive",
        "completed_failure_within_24h_then_next_open",
    ),
    (
        "negative_jump_completed_failure_long", "long", "jump_dominated", "negative",
        "completed_failure_within_24h_then_next_open",
    ),
)
TIMEOUTS = (("6h", 6, "primary"), ("24h", 24, "robustness"))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def definition_register() -> list[dict[str, Any]]:
    """Return all 16 frozen model x branch x timeout definitions."""
    rows: list[dict[str, Any]] = []
    for model, model_role, prefix in (
        (PRIMARY_MODEL, "primary", "p"),
        (ROBUSTNESS_MODEL, "robustness_only", "r"),
    ):
        for branch_index, (branch, side, path, sign, entry_rule) in enumerate(BRANCHES, start=1):
            for timeout, timeout_hours, timeout_role in TIMEOUTS:
                policy = {
                    "model": model,
                    "branch": branch,
                    "side": side,
                    "path_state": path,
                    "shock_sign": sign,
                    "entry_rule": entry_rule,
                    "jump_confirmation_window_hours": 24 if path == "jump_dominated" else None,
                    "timeout_hours": timeout_hours,
                    "fixed_notional": True,
                }
                rows.append({
                    "definition_id": f"c01_l3_{prefix}{branch_index:02d}_{timeout}",
                    "model": model,
                    "model_role": model_role,
                    "branch": branch,
                    "side": side,
                    "path_state": path,
                    "shock_sign": sign,
                    "entry_rule": entry_rule,
                    "timeout": timeout,
                    "timeout_hours": timeout_hours,
                    "timeout_role": timeout_role,
                    "definition_policy_hash": canonical_hash(policy),
                    "registered_even_if_zero_trades": True,
                })
    ids = [row["definition_id"] for row in rows]
    hashes = [row["definition_policy_hash"] for row in rows]
    if len(rows) != 16 or len(set(ids)) != 16 or len(set(hashes)) != 16:
        raise ValueError("C01 definition register is not exactly 16 unique policies")
    return rows


def _utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return result.astimezone(timezone.utc)


def interval_is_wholly_train_eligible(required_timestamps: Mapping[str, Any]) -> bool:
    """Require every declared confirmation/entry/monitoring/funding/execution time inside train."""
    required = {
        "onset_ts", "confirmation_ts", "entry_ts", "last_stop_monitor_ts",
        "timeout_ts", "funding_accounting_end_ts", "exit_execution_ts",
    }
    if set(required_timestamps) != required or any(value is None for value in required_timestamps.values()):
        return False
    timestamps = [_utc(value) for value in required_timestamps.values()]
    return all(TRAIN_START <= value < TRAIN_END for value in timestamps)


@dataclass(frozen=True)
class NonOverlapResult:
    accepted: tuple[dict[str, Any], ...]
    skipped: tuple[dict[str, Any], ...]


def definition_local_non_overlap(rows: Iterable[Mapping[str, Any]]) -> NonOverlapResult:
    """Apply actual-exit non-overlap independently by definition and symbol."""
    work = [dict(row) for row in rows]
    required = {"definition_id", "symbol", "economic_address", "onset_ts", "entry_ts", "actual_exit_ts"}
    if any(not required.issubset(row) for row in work):
        raise ValueError("non-overlap row is missing required identity or actual-exit fields")
    addresses = [str(row["economic_address"]) for row in work]
    if len(addresses) != len(set(addresses)):
        raise ValueError("duplicate economic address")
    work.sort(key=lambda row: (
        str(row["definition_id"]), str(row["symbol"]), _utc(row["onset_ts"]),
        str(row["economic_address"]),
    ))
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    active: dict[tuple[str, str], dict[str, Any]] = {}
    for row in work:
        entry_ts = _utc(row["entry_ts"])
        exit_ts = _utc(row["actual_exit_ts"])
        if exit_ts <= entry_ts:
            raise ValueError("actual executable exit must follow entry")
        key = (str(row["definition_id"]), str(row["symbol"]))
        prior = active.get(key)
        if prior is not None and entry_ts < _utc(prior["actual_exit_ts"]):
            skipped.append({
                **row,
                "skip_reason": "same_symbol_definition_position_still_open",
                "prior_economic_address": prior["economic_address"],
                "prior_entry_ts": prior["entry_ts"],
                "prior_actual_exit_ts": prior["actual_exit_ts"],
            })
            continue
        accepted.append(row)
        active[key] = row
    if len(accepted) + len(skipped) != len(work):
        raise AssertionError("non-overlap reconciliation failed")
    return NonOverlapResult(tuple(accepted), tuple(skipped))


def fixed_notional_net_bps(
    *, entry_price: float, exit_price: float, side: str,
    fee_bps: float, slippage_bps: float, funding_cashflow_bps: float,
) -> dict[str, float]:
    values = (entry_price, exit_price, fee_bps, slippage_bps, funding_cashflow_bps)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("non-finite fixed-notional input")
    if entry_price <= 0 or exit_price <= 0 or fee_bps < 0 or slippage_bps < 0:
        raise ValueError("invalid fixed-notional price or cost")
    direction = {"long": 1.0, "short": -1.0}.get(side)
    if direction is None:
        raise ValueError("side must be long or short")
    gross = direction * (exit_price / entry_price - 1.0) * 10_000.0
    net = gross - fee_bps - slippage_bps + funding_cashflow_bps
    return {
        "gross_return_bps": gross,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "funding_cashflow_bps": funding_cashflow_bps,
        "net_return_bps": net,
    }


def funding_partition(row: Mapping[str, Any]) -> str:
    exact = int(row.get("exact_boundary_count", -1))
    imputed = int(row.get("imputed_boundary_count", -1))
    if exact < 0 or imputed < 0:
        raise ValueError("funding boundary counts are required")
    if exact == 0 and imputed == 0:
        return "zero_boundary"
    if exact > 0 and imputed == 0:
        return "fully_exact"
    if exact == 0 and imputed > 0:
        return "fully_imputed"
    return "mixed"


def partition_funding_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output = {name: [] for name in ("fully_exact", "mixed", "fully_imputed", "zero_boundary")}
    for row in rows:
        item = dict(row)
        output[funding_partition(item)].append(item)
    return output


def canonical_episode_bootstrap_mean_ci(
    episode_values_bps: Mapping[str, Sequence[float]], *,
    seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    if resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("C01 cluster bootstrap must use exactly 10,000 resamples")
    episodes = sorted(episode_values_bps)
    if not episodes or any(not episode_values_bps[key] for key in episodes):
        raise ValueError("non-empty canonical episodes are required")
    arrays = [np.asarray(episode_values_bps[key], dtype=float) for key in episodes]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("non-finite bootstrap value")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled = rng.integers(0, len(arrays), size=len(arrays))
        means[index] = np.concatenate([arrays[position] for position in sampled]).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


LEVEL3_GATE_NAMES = (
    "executed_trades_at_least_100",
    "each_calendar_year_at_least_20",
    "positive_mean_net_bps",
    "positive_median_net_bps",
    "bootstrap_lower_bound_at_least_minus_5_bps",
    "max_symbol_pnl_share_at_most_25pct",
    "max_episode_pnl_share_at_most_10pct",
    "max_year_positive_pnl_share_at_most_70pct",
    "stress_mean_at_least_minus_10_bps",
)


def level3_gate_flags(metrics: Mapping[str, Any]) -> dict[str, bool]:
    years = metrics.get("trade_count_by_year", {})
    return {
        "executed_trades_at_least_100": int(metrics["executed_trades"]) >= 100,
        "each_calendar_year_at_least_20": all(int(years.get(year, 0)) >= 20 for year in (2023, 2024, 2025)),
        "positive_mean_net_bps": float(metrics["mean_net_bps"]) > 0,
        "positive_median_net_bps": float(metrics["median_net_bps"]) > 0,
        "bootstrap_lower_bound_at_least_minus_5_bps": float(metrics["bootstrap_ci_lower_bps"]) >= -5,
        "max_symbol_pnl_share_at_most_25pct": float(metrics["max_symbol_pnl_share"]) <= 0.25,
        "max_episode_pnl_share_at_most_10pct": float(metrics["max_episode_pnl_share"]) <= 0.10,
        "max_year_positive_pnl_share_at_most_70pct": float(metrics["max_year_positive_pnl_share"]) <= 0.70,
        "stress_mean_at_least_minus_10_bps": float(metrics["stress_mean_net_bps"]) >= -10,
    }


def definitions_permitted_for_level4(
    definition_metrics: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Only primary definitions can pass; robustness rows cannot rescue them."""
    passed: list[str] = []
    for row in definition_metrics:
        if row.get("model") != PRIMARY_MODEL:
            continue
        flags = level3_gate_flags(row)
        if set(flags) != set(LEVEL3_GATE_NAMES):
            raise AssertionError("Level-3 gate registry mismatch")
        if all(flags.values()):
            passed.append(str(row["definition_id"]))
    return sorted(passed)


def select_matched_non_event(
    event: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Choose the nearest frozen-caliper match without widening."""
    event_ts = _utc(event["onset_ts"])
    event_vol = abs(float(event["lagged_volatility_24h"]))
    if event_vol <= 0:
        raise ValueError("event lagged volatility must be positive")
    matches: list[tuple[float, datetime, str, dict[str, Any]]] = []
    for raw in candidates:
        row = dict(raw)
        if (
            row.get("symbol") != event.get("symbol")
            or int(row.get("calendar_year", -1)) != int(event.get("calendar_year", -2))
            or row.get("direction") != event.get("direction")
            or bool(row.get("inside_same_symbol_c01_episode", True))
        ):
            continue
        timestamp = _utc(row["timestamp"])
        if abs((timestamp - event_ts).total_seconds()) < 48 * 3600:
            continue
        control_vol = abs(float(row["lagged_volatility_24h"]))
        vol_distance = abs(control_vol - event_vol) / event_vol
        btc_distance = abs(abs(float(row["btc_return_6h_bps"])) - abs(float(event["btc_return_6h_bps"])))
        eth_distance = abs(abs(float(row["eth_return_6h_bps"])) - abs(float(event["eth_return_6h_bps"])))
        if vol_distance > 0.20 or btc_distance > 50 or eth_distance > 50:
            continue
        score = vol_distance + btc_distance / 50.0 + eth_distance / 50.0
        matches.append((score, timestamp, str(row.get("control_address", "")), row))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return matches[0][3]


ALLOWED_GAP_STATUSES = {"unavailable", "irrecoverable", "deferred_with_exact_task"}


def validate_package_disposition(disposition: Mapping[str, Any]) -> None:
    if disposition.get("protocol_disposition") != "closed_by_claim_narrowing":
        raise ValueError("package protocol disposition is not narrowed")
    if disposition.get("package_role") != "strategic_and_continuity_review_only":
        raise ValueError("package role is too broad")
    if disposition.get("package_release_ready_for_independent_reproduction") is not False:
        raise ValueError("claim narrowing cannot make the package release-ready")
    gaps = disposition.get("missing_items")
    if not isinstance(gaps, list) or not gaps:
        raise ValueError("package gaps must remain explicit")
    for gap in gaps:
        if gap.get("status") not in ALLOWED_GAP_STATUSES:
            raise ValueError("missing evidence cannot be converted to pass")
        if gap.get("status") == "deferred_with_exact_task" and not gap.get("exact_task"):
            raise ValueError("deferred gap requires an exact task")
