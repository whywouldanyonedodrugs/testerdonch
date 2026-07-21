from __future__ import annotations

import math
from bisect import bisect_right
from datetime import datetime, timezone
from typing import Iterable, Sequence


class EngineInputError(ValueError):
    pass


def require_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise EngineInputError("timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


def require_available(feature_available_ts: datetime, decision_ts: datetime) -> None:
    if require_utc(feature_available_ts) > require_utc(decision_ts):
        raise EngineInputError("feature_available_ts exceeds decision_ts")


def log_return(start: float, end: float) -> float:
    if not math.isfinite(start) or not math.isfinite(end) or start <= 0 or end <= 0:
        raise EngineInputError("prices must be positive finite values")
    return math.log(end / start)


def sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise EngineInputError("sample standard deviation requires at least two values")
    if any(not math.isfinite(value) for value in values):
        raise EngineInputError("non-finite value")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def close_to_close_volatility(closes: Sequence[float]) -> float:
    if len(closes) < 3:
        raise EngineInputError("close-to-close volatility requires at least three closes")
    returns = [log_return(left, right) for left, right in zip(closes, closes[1:])]
    return sample_standard_deviation(returns) * math.sqrt(365.0 * 288.0)


def parkinson_volatility(highs: Sequence[float], lows: Sequence[float]) -> float:
    if len(highs) != len(lows) or len(highs) < 2:
        raise EngineInputError("Parkinson volatility requires matched high/low observations")
    terms = []
    for high, low in zip(highs, lows):
        if not math.isfinite(high) or not math.isfinite(low) or high <= 0 or low <= 0 or high < low:
            raise EngineInputError("invalid completed high/low observation")
        terms.append(math.log(high / low) ** 2)
    return math.sqrt(sum(terms) / len(terms) / (4.0 * math.log(2.0))) * math.sqrt(365.0 * 288.0)


def ema(values: Sequence[float], span: int) -> list[float]:
    if span <= 0 or len(values) < span:
        raise EngineInputError("EMA history shorter than span")
    alpha = 2.0 / (span + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        if not math.isfinite(value) or value <= 0:
            raise EngineInputError("EMA price must be positive and finite")
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def path_smoothness(closes: Sequence[float]) -> float:
    if len(closes) < 2:
        raise EngineInputError("path smoothness needs at least two closes")
    numerator = abs(log_return(closes[0], closes[-1]))
    denominator = sum(abs(log_return(left, right)) for left, right in zip(closes, closes[1:]))
    return 0.0 if denominator == 0 else numerator / denominator


def type7_quantile(values: Sequence[float], probability: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite or not 0 <= probability <= 1:
        raise EngineInputError("invalid Type-7 quantile input")
    if len(finite) == 1:
        return finite[0]
    h = (len(finite) - 1) * probability
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    return finite[lower] + (h - lower) * (finite[upper] - finite[lower])


def type7_quantile_with_negative_infinity(values: Sequence[float], probability: float) -> float:
    """Frozen Type-7 arithmetic where explicit empty folds are negative infinity."""
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0 <= probability <= 1 or any(math.isnan(value) or value == math.inf for value in ordered):
        raise EngineInputError("invalid Type-7-with-negative-infinity input")
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * probability
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    fraction = h - lower
    if fraction == 0:
        return ordered[lower]
    if ordered[lower] == -math.inf or ordered[upper] == -math.inf:
        return -math.inf
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def weak_percentile(value: float, population: Sequence[float]) -> float:
    finite = [float(item) for item in population if math.isfinite(float(item))]
    if len(finite) < 30 or len(set(finite)) < 20:
        raise EngineInputError("threshold population fails minimums")
    return sum(item <= value for item in finite) / len(finite)


def weak_percentile_prevalidated_sorted(value: float, population: Sequence[float]) -> float:
    """Exact weak percentile for a caller-validated finite sorted population."""
    if not population:
        raise EngineInputError("prevalidated threshold population is empty")
    return bisect_right(population, float(value)) / len(population)


def average_rank_percentiles(values: Sequence[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(float(value)) for value in values):
        raise EngineInputError("cross-sectional ranking requires at least two finite values")
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average_one_based_rank = ((cursor + 1) + end) / 2.0
        normalized = average_one_based_rank / len(values)
        for _, index in ordered[cursor:end]:
            result[index] = normalized
        cursor = end
    return result


def liquidity_decile(normalized_rank: float) -> int:
    if not 0.0 <= normalized_rank <= 1.0:
        raise EngineInputError("normalized rank must be in [0,1]")
    return 1 + min(9, math.floor(10.0 * normalized_rank))


def component_threshold(value: float, level: str) -> float | None:
    if not 0.0 <= value <= 1.0:
        raise EngineInputError("component percentile outside [0,1]")
    if level == "none":
        return None
    if level == "continuous":
        return value
    if level.startswith("q"):
        threshold = int(level[1:]) / 100.0
        return 1.0 if value >= threshold else 0.0
    raise EngineInputError(f"unknown component threshold: {level}")


def arithmetic_mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise EngineInputError("empty arithmetic mean")
    return sum(items) / len(items)


def wilder_atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], window: int) -> float:
    if window <= 0 or len(highs) != len(lows) or len(highs) != len(closes) or len(closes) < window + 1:
        raise EngineInputError("Wilder ATR requires window+1 consecutive completed daily bars")
    true_ranges = []
    for index in range(1, len(closes)):
        high, low, previous = float(highs[index]), float(lows[index]), float(closes[index - 1])
        if min(high, low, previous) <= 0 or high < low:
            raise EngineInputError("invalid daily input for ATR")
        true_ranges.append(max(high - low, abs(high - previous), abs(low - previous)))
    atr = sum(true_ranges[:window]) / window
    for value in true_ranges[window:]:
        atr = (atr * (window - 1) + value) / window
    return atr


def percentile_from_population(value: float, population: Sequence[float], threshold_name: str | None = None) -> tuple[float, bool]:
    percentile = weak_percentile(value, population)
    if threshold_name is None or threshold_name == "none":
        return percentile, True
    if not threshold_name.startswith("q"):
        raise EngineInputError(f"unknown quantile threshold: {threshold_name}")
    return percentile, percentile >= int(threshold_name[1:]) / 100.0


def percentile_from_prevalidated_sorted(value: float, population: Sequence[float], threshold_name: str | None = None) -> tuple[float, bool]:
    percentile = weak_percentile_prevalidated_sorted(value, population)
    if threshold_name is None or threshold_name == "none":
        return percentile, True
    if not threshold_name.startswith("q"):
        raise EngineInputError(f"unknown quantile threshold: {threshold_name}")
    return percentile, percentile >= int(threshold_name[1:]) / 100.0
