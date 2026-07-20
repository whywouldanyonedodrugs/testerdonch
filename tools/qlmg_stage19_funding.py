#!/usr/bin/env python3
"""Decimal-safe Stage 19 funding arithmetic and calibration primitives."""

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping


HOUR = Decimal(3600)
BPS = Decimal(10000)


def type7(values: Iterable[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    if not ordered or probability < 0 or probability > 1:
        raise ValueError("invalid type-7 quantile input")
    if len(ordered) == 1:
        return ordered[0]
    h = Decimal(len(ordered) - 1) * probability
    lower = int(h)
    fraction = h - lower
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def equal_symbol_weighted_quantile(
    sorted_values_by_symbol: Mapping[str, list[Decimal]], probability: Decimal
) -> Decimal:
    """Quantile of an empirical mixture giving every symbol equal total mass.

    The inverse CDF is linearly interpolated between the two weighted empirical
    order statistics bracketing `probability`; each observation has weight
    `1 / (symbol_count * observations_for_symbol)`.
    """
    if not sorted_values_by_symbol or probability < 0 or probability > 1:
        raise ValueError("invalid equal-symbol mixture")
    points: list[tuple[Decimal, Decimal]] = []
    symbol_weight = Decimal(1) / Decimal(len(sorted_values_by_symbol))
    for values in sorted_values_by_symbol.values():
        if not values:
            raise ValueError("empty symbol distribution")
        observation_weight = symbol_weight / Decimal(len(values))
        points.extend((value, observation_weight) for value in values)
    points.sort(key=lambda item: item[0])
    cumulative = Decimal(0)
    previous_value = points[0][0]
    previous_cumulative = Decimal(0)
    for value, weight in points:
        next_cumulative = cumulative + weight
        if probability <= next_cumulative:
            if next_cumulative == previous_cumulative:
                return value
            fraction = (probability - previous_cumulative) / (next_cumulative - previous_cumulative)
            return previous_value + fraction * (value - previous_value)
        previous_value = value
        previous_cumulative = next_cumulative
        cumulative = next_cumulative
    return points[-1][0]


def overlap_seconds(entry: datetime, exit_: datetime, start: datetime, end: datetime) -> Decimal:
    if any(value.tzinfo is None for value in (entry, exit_, start, end)):
        raise ValueError("timestamps must be timezone-aware")
    if exit_ <= entry or end <= start:
        raise ValueError("invalid interval")
    seconds = max(0.0, (min(exit_, end) - max(entry, start)).total_seconds())
    return Decimal(str(seconds))


def exact_cashflow_bps(
    position_sign: int, absolute_rate: Decimal, held_fraction: Decimal,
    entry_trade_open_usd_per_contract_unit: Decimal,
) -> Decimal:
    if position_sign not in (-1, 1):
        raise ValueError("position_sign must be -1 or +1")
    if not absolute_rate.is_finite() or not held_fraction.is_finite() or not entry_trade_open_usd_per_contract_unit.is_finite():
        raise ValueError("non-finite arithmetic input")
    if held_fraction < 0 or held_fraction > 1 or entry_trade_open_usd_per_contract_unit <= 0:
        raise ValueError("invalid arithmetic input")
    return -Decimal(position_sign) * absolute_rate * held_fraction / entry_trade_open_usd_per_contract_unit * BPS


def period_for_row(row_timestamp: datetime, alignment: str) -> tuple[datetime, datetime]:
    if row_timestamp.tzinfo is None or row_timestamp.minute or row_timestamp.second or row_timestamp.microsecond:
        raise ValueError("row timestamp must be an aware UTC-hour boundary")
    if alignment == "alignment_start":
        return row_timestamp, row_timestamp + timedelta(hours=1)
    if alignment == "alignment_end":
        return row_timestamp - timedelta(hours=1), row_timestamp
    raise ValueError("unknown alignment")


def dual_alignment_cashflow_bps(
    *, entry: datetime, exit_: datetime, position_sign: int, entry_trade_open: Decimal,
    absolute_rates: Mapping[datetime, Decimal], base_gap_bps_per_hour: Decimal,
    stress_gap_bps_per_hour: Decimal,
) -> dict[str, Decimal]:
    if (not base_gap_bps_per_hour.is_finite() or not stress_gap_bps_per_hour.is_finite()
            or base_gap_bps_per_hour < 0 or stress_gap_bps_per_hour < base_gap_bps_per_hour):
        raise ValueError("gap allowances must be finite, nonnegative, and stress>=base")
    output: dict[str, Decimal] = {}
    for alignment in ("alignment_start", "alignment_end"):
        cursor = entry.replace(minute=0, second=0, microsecond=0)
        if alignment == "alignment_end":
            cursor += timedelta(hours=1)
        signed = Decimal(0)
        missing_hours = Decimal(0)
        while True:
            start, end = period_for_row(cursor, alignment)
            fraction = overlap_seconds(entry, exit_, start, end) / HOUR
            if fraction > 0:
                if cursor in absolute_rates:
                    signed += exact_cashflow_bps(position_sign, absolute_rates[cursor], fraction, entry_trade_open)
                else:
                    missing_hours += fraction
            if end >= exit_:
                break
            cursor += timedelta(hours=1)
        output[f"signed_{alignment}_bps"] = signed
        output[f"missing_{alignment}_hours"] = missing_hours
    output["adverse_exact_funding_bps"] = min(
        Decimal(0), output["signed_alignment_start_bps"], output["signed_alignment_end_bps"]
    )
    worst_missing = max(output["missing_alignment_start_hours"], output["missing_alignment_end_hours"])
    output["base_gap_cost_bps"] = -base_gap_bps_per_hour * worst_missing
    output["stress_gap_cost_bps"] = -stress_gap_bps_per_hour * worst_missing
    return output


class Stage19FundingEngine:
    """Hash-bound campaign adapter for the immutable rankable funding package."""

    def __init__(
        self, rankable_zip: Path, expected_package_sha256: str,
        allowance_table: Path, expected_allowance_sha256: str,
    ) -> None:
        if self._sha256(rankable_zip) != expected_package_sha256:
            raise RuntimeError("rankable funding package hash mismatch")
        if self._sha256(allowance_table) != expected_allowance_sha256:
            raise RuntimeError("funding allowance table hash mismatch")
        self.rankable_zip = rankable_zip
        self._rates_cache: dict[str, Mapping[datetime, Decimal]] = {}
        self.allowances: dict[str, tuple[Decimal, Decimal]] = {}
        with allowance_table.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                symbol = row["symbol"]
                base = Decimal(row["base_gap_allowance_bps_per_hour"])
                stress = Decimal(row["stress_gap_allowance_bps_per_hour"])
                if symbol in self.allowances or not base.is_finite() or not stress.is_finite() or base < 0 or stress < base:
                    raise RuntimeError("invalid or duplicate funding allowance")
                self.allowances[symbol] = (base, stress)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_symbol(self, symbol: str) -> Mapping[datetime, Decimal]:
        if symbol in self._rates_cache:
            return self._rates_cache[symbol]
        member = f"rankable_2023_2025/{symbol}.csv"
        rates: dict[datetime, Decimal] = {}
        with zipfile.ZipFile(self.rankable_zip) as archive:
            if member not in archive.namelist():
                raise RuntimeError(f"rankable funding member missing: {symbol}")
            with archive.open(member) as raw:
                for row in csv.DictReader(io.TextIOWrapper(raw, encoding="ascii", newline="")):
                    stamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    if not datetime(2023, 1, 1, tzinfo=timezone.utc) <= stamp < datetime(2026, 1, 1, tzinfo=timezone.utc):
                        raise RuntimeError("rankable funding timestamp boundary violation")
                    if row["tradeable"] != symbol or stamp in rates:
                        raise RuntimeError("funding identity or duplicate timestamp violation")
                    rate = Decimal(row["absolute_rate"])
                    if not rate.is_finite():
                        raise RuntimeError("non-finite absolute funding rate")
                    rates[stamp] = rate
        frozen = MappingProxyType(rates)
        self._rates_cache[symbol] = frozen
        return frozen

    def evaluate_trade(
        self, *, symbol: str, entry: datetime, exit_: datetime,
        position_sign: int, entry_trade_open: Decimal,
    ) -> dict[str, Decimal]:
        rankable_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        protected_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        if (entry.tzinfo is None or exit_.tzinfo is None or entry.utcoffset() != timedelta(0)
                or exit_.utcoffset() != timedelta(0) or not rankable_start <= entry < exit_ < protected_start):
            raise RuntimeError("trade interval outside aware-UTC rankable boundary")
        if symbol not in self.allowances:
            raise RuntimeError(f"campaign symbol allowance missing: {symbol}")
        base, stress = self.allowances[symbol]
        return dual_alignment_cashflow_bps(
            entry=entry, exit_=exit_, position_sign=position_sign,
            entry_trade_open=entry_trade_open, absolute_rates=self.load_symbol(symbol),
            base_gap_bps_per_hour=base, stress_gap_bps_per_hour=stress,
        )
