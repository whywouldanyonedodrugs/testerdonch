from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
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
    payment_ts: datetime
    publication_ts: datetime
    rate_bps: float
    source_partition: str = "exact"

    def validate(self) -> None:
        if require_utc(self.publication_ts) > require_utc(self.payment_ts):
            raise EngineInputError("funding publication occurs after payment")
        if self.source_partition != "exact":
            raise EngineInputError("rankable accounting requires exact funding partition")
        if not isinstance(self.rate_bps, (int, float)):
            raise EngineInputError("funding rate must be numeric")


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

    def validate(self, *, pooled: bool = False) -> None:
        finite = tuple(value for value in self.values if isinstance(value, (int, float)))
        if len(finite) < 30 or len(set(finite)) < 20:
            raise EngineInputError("threshold population minimum is not met")
        if pooled and len(set(self.unique_symbols)) < 5:
            raise EngineInputError("pooled threshold population has fewer than five symbols")


@dataclass(frozen=True)
class ContextInputs:
    """Raw component histories; engines fit percentiles instead of accepting ranks."""

    btc_daily: tuple[DailyBar, ...] = ()
    eth_daily: tuple[DailyBar, ...] = ()
    symbol_daily: tuple[DailyBar, ...] = ()
    breadth_history: tuple[float, ...] = ()
    dispersion_history: tuple[float, ...] = ()
    cross_section_returns: Mapping[str, float] = field(default_factory=dict)
    cross_section_liquidity_deciles: Mapping[str, int] = field(default_factory=dict)
    parent_universe: tuple[str, ...] = ()
    funding_burden_history: tuple[float, ...] = ()


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
            if not (RANKABLE_START <= require_utc(payment.payment_ts) < PROTECTED_START):
                raise EngineInputError("funding payment crosses protected boundary")

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
