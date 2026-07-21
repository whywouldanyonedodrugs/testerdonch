from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from .canonical import canonical_hash
from .family_engines.common import EngineInputError, require_utc


KRAKEN_PLATFORM = "kraken_native_linear_pf"
RANKABLE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
PROTECTED_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _content_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return require_utc(value).isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return _content_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _content_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_content_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise EngineInputError(f"unsupported cache-content value: {type(value).__name__}")


@dataclass(frozen=True)
class SignalBar:
    """A completed, point-in-time-safe Kraken trade bar used by an engine."""

    open_ts: datetime
    close_ts: datetime
    open: float
    high: float
    low: float
    close: float
    source_close_ts: datetime
    feature_available_ts: datetime
    lifecycle_valid: bool = True
    pit_eligible: bool = True
    quote_notional: float | None = None

    def validate(self) -> None:
        open_ts = require_utc(self.open_ts)
        close_ts = require_utc(self.close_ts)
        source_close = require_utc(self.source_close_ts)
        available = require_utc(self.feature_available_ts)
        if not (RANKABLE_START <= open_ts < close_ts < PROTECTED_START):
            raise EngineInputError("signal bar crosses the rankable/protected boundary")
        if source_close > close_ts or available > close_ts:
            raise EngineInputError("signal bar contains future information")
        if not self.lifecycle_valid:
            raise EngineInputError("signal bar is outside valid instrument lifecycle")
        if not self.pit_eligible:
            raise EngineInputError("signal bar is outside the point-in-time eligible universe")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise EngineInputError("invalid signal-bar OHLC")

    def trade_bar(self):
        from .accounting import TradeBar
        self.validate()
        return TradeBar(
            self.open_ts,
            self.close_ts,
            self.open,
            self.close,
            high=self.high,
            low=self.low,
            lifecycle_valid=self.lifecycle_valid,
            source_close_ts=self.source_close_ts,
            feature_available_ts=self.feature_available_ts,
        )


@dataclass(frozen=True)
class FundingInput:
    row_timestamp: datetime
    publication_ts: datetime
    absolute_rate_usd_per_contract_unit: str
    source_partition: str = "exact"

    def validate(self) -> None:
        timestamp = require_utc(self.row_timestamp)
        if timestamp.minute or timestamp.second or timestamp.microsecond:
            raise EngineInputError("funding row timestamp is not an exact UTC-hour boundary")
        if require_utc(self.publication_ts) > timestamp:
            raise EngineInputError("funding publication occurs after its registered row timestamp")
        if self.source_partition != "exact":
            raise EngineInputError("rankable accounting requires exact funding partition")
        try:
            rate = Decimal(self.absolute_rate_usd_per_contract_unit)
        except (InvalidOperation, ValueError) as exc:
            raise EngineInputError("funding absolute rate is not a canonical decimal") from exc
        if not rate.is_finite():
            raise EngineInputError("funding absolute rate must be finite")


@dataclass(frozen=True)
class DailyBar:
    close_ts: datetime
    open: float
    high: float
    low: float
    close: float
    source_close_ts: datetime
    feature_available_ts: datetime
    valid_day: bool = True

    def validate(self, decision_ts: datetime) -> None:
        close_ts = require_utc(self.close_ts)
        if close_ts >= require_utc(decision_ts):
            raise EngineInputError("daily input is not completed before decision")
        if require_utc(self.source_close_ts) > require_utc(decision_ts) or require_utc(self.feature_available_ts) > require_utc(decision_ts):
            raise EngineInputError("daily input is not available at decision")
        if not self.valid_day:
            raise EngineInputError("daily input fails the frozen valid-day rule")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise EngineInputError("invalid daily OHLC")


@dataclass(frozen=True)
class ThresholdPopulation:
    values: tuple[float, ...]
    unique_symbols: tuple[str, ...] = ()
    scope: str = "symbol"
    training_start: datetime | None = None
    training_end_exclusive: datetime | None = None
    source_close_ts: datetime | None = None
    feature_available_ts: datetime | None = None
    source_sha256: str | None = None

    def validate(self, *, pooled: bool = False, decision_ts: datetime | None = None) -> None:
        finite = tuple(value for value in self.values if isinstance(value, (int, float)))
        if len(finite) < 30 or len(set(finite)) < 20:
            raise EngineInputError("threshold population minimum is not met")
        if pooled and len(set(self.unique_symbols)) < 5:
            raise EngineInputError("pooled threshold population has fewer than five symbols")
        if None in {self.training_start, self.training_end_exclusive, self.source_close_ts, self.feature_available_ts}:
            raise EngineInputError("threshold population lacks training/availability timestamps")
        start = require_utc(self.training_start)  # type: ignore[arg-type]
        end = require_utc(self.training_end_exclusive)  # type: ignore[arg-type]
        source_close = require_utc(self.source_close_ts)  # type: ignore[arg-type]
        available = require_utc(self.feature_available_ts)  # type: ignore[arg-type]
        if not (RANKABLE_START <= start < end <= PROTECTED_START) or source_close >= end:
            raise EngineInputError("threshold population training interval is invalid")
        if decision_ts is not None and (end > require_utc(decision_ts) or source_close > require_utc(decision_ts) or available > require_utc(decision_ts)):
            raise EngineInputError("threshold population is not available at decision")
        if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise EngineInputError("threshold population source provenance is absent")


@dataclass(frozen=True)
class ContextInputs:
    """Raw component histories; engines fit percentiles instead of accepting ranks."""

    btc_daily: tuple[DailyBar, ...] = ()
    eth_daily: tuple[DailyBar, ...] = ()
    symbol_daily: tuple[DailyBar, ...] = ()
    breadth_history: tuple[float, ...] = ()
    dispersion_history: tuple[float, ...] = ()
    breadth_history_by_lookback: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
    dispersion_history_by_lookback: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
    cross_section_returns: Mapping[str, float] = field(default_factory=dict)
    cross_section_returns_by_lookback: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    cross_section_liquidity_deciles: Mapping[str, int] = field(default_factory=dict)
    parent_universe: tuple[str, ...] = ()
    funding_burden_history: tuple[float, ...] = ()
    funding_burden_current: float | None = None
    as_of_ts: datetime | None = None
    source_close_ts: datetime | None = None
    feature_available_ts: datetime | None = None
    source_sha256: str | None = None

    def validate(self, decision_ts: datetime) -> None:
        populated = bool(
            self.btc_daily or self.eth_daily or self.symbol_daily or self.breadth_history
            or self.dispersion_history or self.breadth_history_by_lookback
            or self.dispersion_history_by_lookback or self.cross_section_returns
            or self.cross_section_returns_by_lookback or self.funding_burden_history
            or self.funding_burden_current is not None
        )
        if not populated:
            return
        if None in {self.as_of_ts, self.source_close_ts, self.feature_available_ts}:
            raise EngineInputError("context snapshot lacks point-in-time timestamps")
        decision = require_utc(decision_ts)
        as_of = require_utc(self.as_of_ts)  # type: ignore[arg-type]
        source_close = require_utc(self.source_close_ts)  # type: ignore[arg-type]
        available = require_utc(self.feature_available_ts)  # type: ignore[arg-type]
        if as_of != decision or source_close > decision or available > decision:
            raise EngineInputError("context snapshot is not exact-as-of the frame decision")
        if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise EngineInputError("context snapshot source provenance is absent")
        for bars in (self.btc_daily, self.eth_daily, self.symbol_daily):
            for bar in bars:
                bar.validate(decision)
        for mapping in (self.cross_section_returns, *self.cross_section_returns_by_lookback.values()):
            if any(not isinstance(value, (int, float)) for value in mapping.values()):
                raise EngineInputError("context cross-section contains a nonnumeric value")
        for histories in (self.breadth_history_by_lookback, self.dispersion_history_by_lookback):
            for lookback, history in histories.items():
                if not isinstance(lookback, int) or lookback <= 0 or len(history) < 30 or any(not isinstance(value, (int, float)) for value in history):
                    raise EngineInputError("context lookback history is incomplete or invalid")


@dataclass(frozen=True)
class FamilyInput:
    """Raw/cache input accepted by the code-owned family dispatcher.

    No net return, candidate rank, or pre-aggregated economic observation is a
    valid field.  Every result is regenerated from completed bars and exact
    funding payments by the registered engine and accounting path.
    """

    platform: str
    symbol: str
    fold_id: str
    decision_ts: datetime
    five_minute_bars: tuple[SignalBar, ...]
    daily_bars: tuple[DailyBar, ...]
    funding: tuple[FundingInput, ...]
    threshold_populations: Mapping[str, ThresholdPopulation] = field(default_factory=dict)
    context: ContextInputs = field(default_factory=ContextInputs)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        decision = require_utc(self.decision_ts)
        if self.platform != KRAKEN_PLATFORM:
            raise EngineInputError("only the Kraken native linear PF adapter is permitted")
        if not (RANKABLE_START <= decision < PROTECTED_START):
            raise EngineInputError("decision timestamp is outside the rankable interval")
        if not self.symbol or self.symbol.upper().startswith("CAPITAL"):
            raise EngineInputError("invalid or Capital.com symbol source")
        if not self.five_minute_bars:
            raise EngineInputError("five-minute bars are required")
        previous: datetime | None = None
        for bar in self.five_minute_bars:
            bar.validate()
            if previous is not None and require_utc(bar.open_ts) <= previous:
                raise EngineInputError("five-minute bars are not strictly sorted")
            previous = require_utc(bar.open_ts)
        for bar in self.daily_bars:
            bar.validate(decision)
        for payment in self.funding:
            payment.validate()
            if not (RANKABLE_START <= require_utc(payment.row_timestamp) < PROTECTED_START):
                raise EngineInputError("funding payment crosses protected boundary")
        for population in self.threshold_populations.values():
            population.validate(decision_ts=decision)
        self.context.validate(decision)

    def require_pit_top_n(self, top_n: int) -> None:
        snapshot = self.metadata.get("pit_universe_snapshot")
        if not isinstance(snapshot, Mapping):
            raise EngineInputError("point-in-time universe snapshot is absent")
        available = snapshot.get("feature_available_ts"); source_close = snapshot.get("source_close_ts"); as_of = snapshot.get("as_of_ts")
        if not all(isinstance(value, datetime) for value in (available, source_close, as_of)) or require_utc(as_of) != require_utc(self.decision_ts) or require_utc(available) > require_utc(self.decision_ts) or require_utc(source_close) > require_utc(self.decision_ts):
            raise EngineInputError("point-in-time universe is not exact and available at decision")
        source_hash = snapshot.get("source_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise EngineInputError("point-in-time universe source hash is absent")
        ranks = snapshot.get("lagged_liquidity_ranks")
        eligible = snapshot.get("eligible_symbols")
        notionals = snapshot.get("lagged_quote_notional")
        top_n_sets = snapshot.get("top_n_symbols")
        if not isinstance(eligible, (tuple, list)) or len(set(eligible)) != len(eligible) or not isinstance(ranks, Mapping) or set(ranks) != set(eligible) or self.symbol not in ranks:
            raise EngineInputError("symbol is absent from point-in-time liquidity ranks")
        if sorted(int(value) for value in ranks.values()) != list(range(1, len(eligible) + 1)):
            raise EngineInputError("point-in-time liquidity ranks are not a complete deterministic ordering")
        if not isinstance(notionals, Mapping) or set(notionals) != set(eligible) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0 for value in notionals.values()):
            raise EngineInputError("point-in-time lagged quote-notional population is incomplete")
        if not isinstance(top_n_sets, Mapping) or str(top_n) not in top_n_sets:
            raise EngineInputError("registered PIT top-N roster was not compiled")
        expected_roster = tuple(symbol for symbol, _ in sorted(ranks.items(), key=lambda item: (int(item[1]), str(item[0])))[:top_n])
        if tuple(top_n_sets[str(top_n)]) != expected_roster:
            raise EngineInputError("registered PIT top-N roster differs from the bound rank ordering")
        rank = ranks[self.symbol]
        if not isinstance(rank, int) or rank < 1 or rank > top_n or self.symbol not in expected_roster:
            raise EngineInputError("symbol is outside the registered PIT liquidity top-N")

    def content_payload(self) -> dict[str, Any]:
        """Canonical cache payload, excluding only its non-semantic cache locator."""
        metadata = {
            key: value
            for key, value in self.metadata.items()
            if key not in {"cache_artifact_path", "cache_bindings"}
        }
        return _content_value({
            "platform": self.platform,
            "symbol": self.symbol,
            "fold_id": self.fold_id,
            "decision_ts": self.decision_ts,
            "five_minute_bars": self.five_minute_bars,
            "daily_bars": self.daily_bars,
            "funding": self.funding,
            "threshold_populations": self.threshold_populations,
            "context": self.context,
            "metadata": metadata,
        })

    def content_sha256(self) -> str:
        return canonical_hash(self.content_payload())

    def bars_at_or_after(self, trigger_close_ts: datetime, *, maximum_delay: timedelta = timedelta(minutes=10)) -> tuple[TradeBar, ...]:
        trigger = require_utc(trigger_close_ts)
        result = tuple(bar.trade_bar() for bar in self.five_minute_bars if require_utc(bar.open_ts) >= trigger)
        if not result or require_utc(result[0].open_ts) - trigger > maximum_delay:
            raise EngineInputError("no lifecycle-valid authorized open within ten minutes")
        return result


def require_contiguous_5m(bars: Sequence[SignalBar]) -> None:
    for left, right in zip(bars, bars[1:]):
        if require_utc(right.open_ts) - require_utc(left.open_ts) != timedelta(minutes=5):
            raise EngineInputError("required five-minute history is not contiguous")


__all__ = [
    "ContextInputs",
    "DailyBar",
    "FamilyInput",
    "FundingInput",
    "KRAKEN_PLATFORM",
    "PROTECTED_START",
    "RANKABLE_START",
    "SignalBar",
    "ThresholdPopulation",
    "require_contiguous_5m",
]
