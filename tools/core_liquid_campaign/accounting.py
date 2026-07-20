from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from .family_engines.common import EngineInputError, require_utc


EXIT_PRIORITY = ("structural_or_ATR_stop", "trail", "fixed_target", "signal_reversal", "time")


@dataclass(frozen=True)
class TradeBar:
    open_ts: datetime
    close_ts: datetime
    open: float
    close: float
    high: float | None = None
    low: float | None = None
    lifecycle_valid: bool = True
    source_close_ts: datetime | None = None
    feature_available_ts: datetime | None = None

    def validate(self) -> None:
        if require_utc(self.open_ts) >= require_utc(self.close_ts):
            raise EngineInputError("bar open_ts must precede close_ts")
        if not all(math.isfinite(value) and value > 0 for value in (self.open, self.close)):
            raise EngineInputError("trade bar prices must be positive finite values")
        if not self.lifecycle_valid:
            raise EngineInputError("trade bar is outside valid instrument lifecycle")
        if self.high is not None and (not math.isfinite(self.high) or self.high < max(self.open, self.close)):
            raise EngineInputError("invalid trade-bar high")
        if self.low is not None and (not math.isfinite(self.low) or self.low > min(self.open, self.close) or self.low <= 0):
            raise EngineInputError("invalid trade-bar low")
        for timestamp in (self.source_close_ts, self.feature_available_ts):
            if timestamp is not None and require_utc(timestamp) > require_utc(self.close_ts):
                raise EngineInputError("trade bar uses information unavailable at close")


@dataclass(frozen=True)
class FundingPayment:
    payment_ts: datetime
    publication_ts: datetime
    rate_bps: float
    source_partition: str = "exact"

    def validate(self) -> None:
        if require_utc(self.publication_ts) > require_utc(self.payment_ts):
            raise EngineInputError("funding publication occurs after payment")
        if self.source_partition != "exact":
            raise EngineInputError("rankable accounting requires exact funding partition")
        if not math.isfinite(self.rate_bps):
            raise EngineInputError("funding rate must be finite")


@dataclass(frozen=True)
class LegResult:
    status: str
    entry_ts: datetime
    exit_ts: datetime | None
    entry_price: float
    exit_price: float | None
    side: int
    exit_reason: str | None
    gross_bps: float | None
    cost_bps: float | None
    funding_bps: float | None
    favorable_funding_bps: float | None
    gap_allowance_bps: float | None
    net_bps: float | None
    reportable_net_bps: float | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["entry_ts"] = require_utc(self.entry_ts).isoformat().replace("+00:00", "Z")
        value["exit_ts"] = require_utc(self.exit_ts).isoformat().replace("+00:00", "Z") if self.exit_ts else None
        return value


def _duration(exit_name: str) -> timedelta | None:
    mapping = {
        "time_1h": timedelta(hours=1),
        "time_3h": timedelta(hours=3),
        "time_6h": timedelta(hours=6),
        "time_1d": timedelta(days=1),
        "time_3d": timedelta(days=3),
        "time_5d": timedelta(days=5),
        "time_10d": timedelta(days=10),
    }
    return mapping.get(exit_name)


def _atr_multiple(exit_name: str) -> float | None:
    if exit_name.startswith("ATR_stop_") or exit_name.startswith("ATR_trail_"):
        return float(exit_name.rsplit("_", 1)[1])
    return None


def _funding_cashflow(payments: Iterable[FundingPayment], entry_ts: datetime, exit_ts: datetime, side: int, alignment: str) -> float:
    if alignment == "minimum_of_registered_start_end":
        materialized = tuple(payments)
        return min(
            _funding_cashflow(materialized, entry_ts, exit_ts, side, "start_inclusive_end_exclusive"),
            _funding_cashflow(materialized, entry_ts, exit_ts, side, "start_exclusive_end_inclusive"),
        )
    total = 0.0
    for payment in sorted(payments, key=lambda item: require_utc(item.payment_ts)):
        payment.validate()
        timestamp = require_utc(payment.payment_ts)
        if alignment == "start_inclusive_end_exclusive":
            included = require_utc(entry_ts) <= timestamp < require_utc(exit_ts)
        elif alignment == "start_exclusive_end_inclusive":
            included = require_utc(entry_ts) < timestamp <= require_utc(exit_ts)
        else:
            raise EngineInputError(f"unknown funding alignment: {alignment}")
        if included:
            total += -side * payment.rate_bps
    return total


def simulate_leg(
    bars: Sequence[TradeBar],
    *,
    entry_index: int,
    side: int,
    exit_name: str,
    atr: float | None,
    fixed_target_r: float | None,
    structural_level: float | None,
    signal_reversal_close_ts: set[datetime] | None,
    funding: Sequence[FundingPayment],
    cost_bps: float,
    funding_alignment: str,
    evaluation_end_exclusive: datetime,
    exposure: float = 1.0,
    gap_allowance_bps: float = 0.0,
    maximum_fill_delay: timedelta = timedelta(minutes=10),
    evaluation_start: datetime | None = None,
) -> LegResult:
    if side not in (-1, 1):
        raise EngineInputError("side must be -1 or 1")
    if not 0 <= entry_index < len(bars):
        raise EngineInputError("entry index outside bars")
    if any(require_utc(left.open_ts) >= require_utc(right.open_ts) for left, right in zip(bars, bars[1:])):
        raise EngineInputError("bars must be strictly sorted")
    for bar in bars:
        bar.validate()
    entry = bars[entry_index]
    entry_ts = require_utc(entry.open_ts)
    boundary = require_utc(evaluation_end_exclusive)
    start = require_utc(evaluation_start) if evaluation_start is not None else None
    if entry_ts >= boundary or (start is not None and entry_ts < start):
        raise EngineInputError("entry is outside evaluation boundary")
    if not math.isfinite(exposure) or exposure < 0:
        raise EngineInputError("leg exposure must be nonnegative and finite")
    if not math.isfinite(gap_allowance_bps) or gap_allowance_bps > 0:
        raise EngineInputError("funding gap allowance must be zero or adverse")
    if cost_bps < 0 or not math.isfinite(cost_bps):
        raise EngineInputError("round-trip cost must be nonnegative and finite")
    entry_price = entry.open
    multiple = _atr_multiple(exit_name)
    if multiple is not None and (atr is None or not math.isfinite(atr) or atr <= 0):
        raise EngineInputError("ATR exit requires positive ATR")
    if fixed_target_r is not None and not exit_name.startswith("ATR_stop_"):
        raise EngineInputError("fixed target requires ATR stop exit")
    deadline = entry_ts + (_duration(exit_name) or timedelta(days=10))
    initial_stop = entry_price - side * multiple * float(atr) if multiple is not None else None
    target = entry_price + side * fixed_target_r * multiple * float(atr) if fixed_target_r is not None and multiple is not None else None
    best_close = entry.close
    reversal = {require_utc(item) for item in (signal_reversal_close_ts or set())}
    trigger_reason: str | None = None
    fill_bar: TradeBar | None = None
    for index in range(entry_index, len(bars) - 1):
        bar = bars[index]
        next_bar = bars[index + 1]
        best_close = max(best_close, bar.close) if side == 1 else min(best_close, bar.close)
        triggers: set[str] = set()
        if structural_level is not None and side * (bar.close - structural_level) < 0:
            triggers.add("structural_or_ATR_stop")
        if exit_name.startswith("ATR_stop_") and initial_stop is not None and side * (bar.close - initial_stop) <= 0:
            triggers.add("structural_or_ATR_stop")
        if exit_name.startswith("ATR_trail_") and atr is not None:
            trail = best_close - side * multiple * atr
            if side * (bar.close - trail) <= 0:
                triggers.add("trail")
        if target is not None and side * (bar.close - target) >= 0:
            triggers.add("fixed_target")
        if exit_name == "signal_reversal" and require_utc(bar.close_ts) in reversal:
            triggers.add("signal_reversal")
        if require_utc(next_bar.open_ts) >= deadline:
            triggers.add("time")
        if triggers:
            trigger_reason = next(reason for reason in EXIT_PRIORITY if reason in triggers)
            if require_utc(next_bar.open_ts) - require_utc(bar.close_ts) > maximum_fill_delay:
                return LegResult("unavailable_missing_open", entry_ts, None, entry_price, None, side, trigger_reason, None, None, None, None, None, None, None)
            fill_bar = next_bar
            break
    if fill_bar is None or require_utc(fill_bar.open_ts) >= boundary:
        return LegResult("unavailable_boundary_crossing", entry_ts, None, entry_price, None, side, None, None, None, None, None, None, None, None)
    exit_ts = require_utc(fill_bar.open_ts)
    gross = side * (fill_bar.open / entry_price - 1.0) * 10000.0
    exact_funding = _funding_cashflow(funding, entry_ts, exit_ts, side, funding_alignment)
    adverse_funding = min(exact_funding, 0.0)
    favorable_funding = max(exact_funding, 0.0)
    weighted_gross = exposure * gross
    weighted_cost = exposure * cost_bps
    weighted_adverse_funding = exposure * (adverse_funding + gap_allowance_bps)
    weighted_favorable_funding = exposure * favorable_funding
    selection_net = weighted_gross - weighted_cost + weighted_adverse_funding
    reportable_net = selection_net + weighted_favorable_funding
    return LegResult(
        "complete",
        entry_ts,
        exit_ts,
        entry_price,
        fill_bar.open,
        side,
        trigger_reason,
        weighted_gross,
        weighted_cost,
        weighted_adverse_funding,
        weighted_favorable_funding,
        exposure * gap_allowance_bps,
        selection_net,
        reportable_net,
    )


def aggregate_parent_legs(starter: LegResult, starter_fraction: float, add: LegResult | None, add_fraction: float) -> dict[str, Any]:
    if starter.status != "complete" or starter.net_bps is None or starter.exit_ts is None:
        raise EngineInputError("starter leg must be complete")
    if starter_fraction + add_fraction > 1.0 + 1e-12:
        raise EngineInputError("parent fractions exceed one")
    valid_add = add is not None and add.status == "complete" and add.net_bps is not None and add.exit_ts is not None
    if not valid_add:
        add_fraction = 0.0
    net = starter_fraction * starter.net_bps + add_fraction * (add.net_bps if valid_add else 0.0)
    reportable = starter_fraction * float(starter.reportable_net_bps) + add_fraction * (float(add.reportable_net_bps) if valid_add else 0.0)
    gross = starter_fraction * float(starter.gross_bps) + add_fraction * (float(add.gross_bps) if valid_add else 0.0)
    cost = starter_fraction * float(starter.cost_bps) + add_fraction * (float(add.cost_bps) if valid_add else 0.0)
    funding = starter_fraction * float(starter.funding_bps) + add_fraction * (float(add.funding_bps) if valid_add else 0.0)
    favorable = starter_fraction * float(starter.favorable_funding_bps) + add_fraction * (float(add.favorable_funding_bps) if valid_add else 0.0)
    gap = starter_fraction * float(starter.gap_allowance_bps) + add_fraction * (float(add.gap_allowance_bps) if valid_add else 0.0)
    exits = [starter.exit_ts] + ([add.exit_ts] if valid_add else [])
    return {
        "event_count": 1,
        "parent_exit_ts": max(exits),
        "gross_bps": gross,
        "cost_bps": cost,
        "funding_bps": funding,
        "favorable_funding_bps_report_only": favorable,
        "gap_allowance_bps": gap,
        "net_bps": net,
        "reportable_net_bps": reportable,
        "add_status": "complete" if valid_add else "unavailable_add",
    }


def contract() -> dict[str, Any]:
    return {
        "price_roles": {"trade_close": "triggers/features", "trade_open": "fills", "mark": "not substituted", "index": "not substituted"},
        "costs": {"base_round_trip_bps_per_leg": 14.0, "stress_round_trip_bps_per_leg": 32.0},
        "funding": "exact signed cashflow at registered boundary alignment; favourable funding is report-only and cannot activate, rank, rescue, or select; adverse exact funding and registered adverse gap allowance enter selection net",
        "trigger": "completed close only",
        "fill": "next lifecycle-valid authorized trade open at or after trigger with an enforced ten-minute maximum lookup delay",
        "simultaneous_exit_priority": list(EXIT_PRIORITY),
        "boundary": "drop/censor; never artificial close",
        "non_overlap": "actual executable exit timestamp",
    }
