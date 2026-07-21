from __future__ import annotations

import csv
import json
import math
import zipfile
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .a1_state import initial_state
from .cache import SemanticCacheWriter
from .canonical import atomic_write_json, canonical_hash, sha256_file
from .engine_types import ContextInputs, DailyBar, FamilyInput, FundingInput, KRAKEN_PLATFORM, SignalBar, ThresholdPopulation
from .engine_types import ExactPopulationView
from .family_engines import a1_compression, a2_context, a3_starter_retest, a4_tsmom
from .family_engines.common import EngineInputError, ema, log_return, path_smoothness, sample_standard_deviation, wilder_atr
from .production_cache import ALLOWED_CANDLE_COLUMNS, ProductionCacheCompiler, ProductionCacheError, SourcePart
from .production_population_tables import (
    A1PopulationTableAuthority,
    A1PopulationTableCompiler,
    A3PopulationTableAuthority,
    A3PopulationTableCompiler,
    PopulationTableError,
)
from .synthetic import with_source_authority


UTC = timezone.utc
FAMILIES = (
    "A4_TSMOM_V7",
    "A1_COMPRESSION_V2",
    "A2_PRIOR_HIGH_RS_CONTEXT_V1",
    "A3_STARTER_RETEST_V3",
    "KDA02B_SURVIVOR_ADJUDICATION_V1",
)


class ProductionInputError(RuntimeError):
    pass


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _population(
    values: Sequence[float],
    *,
    symbols: Sequence[str],
    scope: str,
    training_start: datetime,
    training_end: datetime,
    identity: Mapping[str, Any],
    store: "ExactPopulationStore | None" = None,
) -> ThresholdPopulation | None:
    finite = tuple(float(value) for value in values if math.isfinite(float(value)))
    if len(finite) < 30 or len(set(finite)) < 20:
        return None
    stored_values: Sequence[float] = finite if store is None else store.add(finite, identity=identity)
    value_identity: Any = stored_values.identity_payload() if isinstance(stored_values, ExactPopulationView) else finite
    return ThresholdPopulation(
        stored_values,
        tuple(sorted(set(symbols))),
        scope,
        training_start,
        training_end,
        training_end - timedelta(minutes=5),
        training_end,
        canonical_hash({"identity": identity, "values": value_identity, "symbols": sorted(set(symbols))}),
    )


class ExactPopulationStore:
    """Content-address exact sorted populations outside repeated frame JSON."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.root = cache_root / "populations"

    def add(self, values: Sequence[float], *, identity: Mapping[str, Any]) -> ExactPopulationView:
        import numpy as np

        array = np.asarray(values, dtype="<f8")
        if array.ndim != 1 or not bool(np.isfinite(array).all()):
            raise ProductionInputError("threshold population store received nonfinite values")
        array.sort(kind="mergesort")
        value_hash = canonical_hash({"values_sha256": __import__("hashlib").sha256(array.tobytes(order="C")).hexdigest(), "count": len(array), "dtype": "float64_le"})
        relative = Path("populations") / f"{value_hash}.npy"
        path = self.cache_root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                __import__("os").fsync(handle.fileno())
            temporary.replace(path)
        return ExactPopulationView(
            relative.as_posix(), sha256_file(path), len(array), 0, len(array),
            int(np.unique(array).size), str(self.cache_root),
        )


def _valid_daily(bars: Sequence[SignalBar]) -> tuple[DailyBar, ...]:
    grouped: dict[datetime, list[SignalBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.open_ts.replace(hour=0, minute=0, second=0, microsecond=0)].append(bar)
    rows = []
    for day, members in sorted(grouped.items()):
        members = sorted(members, key=lambda item: item.open_ts)
        if len(members) < 274 or members[0].open_ts != day or members[-1].open_ts != day + timedelta(hours=23, minutes=55):
            continue
        if any(right.open_ts - left.open_ts > timedelta(minutes=15) for left, right in zip(members, members[1:])):
            continue
        close_ts = day + timedelta(days=1)
        rows.append(DailyBar(
            close_ts,
            members[0].open,
            max(item.high for item in members),
            min(item.low for item in members),
            members[-1].close,
            close_ts,
            close_ts,
            True,
        ))
    return tuple(rows)


def _load_trade_bars(parts: Sequence[SourcePart], start: datetime, end: datetime) -> tuple[SignalBar, ...]:
    import pyarrow.parquet as pq

    start_ms = int(start.timestamp() * 1000); end_ms = int(end.timestamp() * 1000)
    rows: dict[int, tuple[float, float, float, float, float]] = {}
    for part in parts:
        if part.dataset != "historical_trade_candles_5m":
            continue
        parquet = pq.ParquetFile(part.path)
        if set(parquet.schema_arrow.names) != ALLOWED_CANDLE_COLUMNS:
            continue
        table = pq.read_table(part.path, columns=["time", "open", "high", "low", "close", "volume"], filters=[("time", ">=", start_ms), ("time", "<", end_ms)])
        for raw in table.to_pylist():
            timestamp = int(raw["time"])
            values = tuple(float(raw[name]) for name in ("open", "high", "low", "close", "volume"))
            if timestamp in rows and rows[timestamp] != values:
                raise ProductionInputError("overlapping trade parts disagree")
            rows[timestamp] = values
    output = []
    for timestamp, (open_price, high, low, close, volume) in sorted(rows.items()):
        open_ts = datetime.fromtimestamp(timestamp / 1000, tz=UTC); close_ts = open_ts + timedelta(minutes=5)
        output.append(SignalBar(open_ts, close_ts, open_price, high, low, close, close_ts, close_ts, True, True, close * volume))
    return tuple(output)


def _load_daily_bars(parts: Sequence[SourcePart], start: datetime, end: datetime) -> tuple[DailyBar, ...]:
    """Load valid daily OHLC directly, without retaining millions of bar objects."""
    import numpy as np
    import pyarrow.parquet as pq

    start_ms = int(start.timestamp() * 1000); end_ms = int(end.timestamp() * 1000)
    chunks: list[tuple[Any, ...]] = []
    previous_time: int | None = None; previous_values: tuple[float, ...] | None = None
    for part in parts:
        if part.dataset != "historical_trade_candles_5m":
            continue
        parquet = pq.ParquetFile(part.path)
        if set(parquet.schema_arrow.names) != ALLOWED_CANDLE_COLUMNS:
            continue
        table = pq.read_table(part.path, columns=["time", "open", "high", "low", "close"], filters=[("time", ">=", start_ms), ("time", "<", end_ms)])
        if not len(table):
            continue
        arrays = [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in ("time", "open", "high", "low", "close")]
        times = arrays[0].astype("<i8", copy=False); values = [array.astype("<f8", copy=False) for array in arrays[1:]]
        offset = 0
        current_first = tuple(float(array[0]) for array in values)
        if previous_time is not None and int(times[0]) == previous_time:
            if current_first != previous_values:
                raise ProductionInputError("overlapping trade parts disagree")
            offset = 1
        elif previous_time is not None and int(times[0]) < previous_time:
            raise ProductionInputError("trade parts overlap beyond their shared boundary")
        chunks.append((times[offset:], *(array[offset:] for array in values)))
        previous_time = int(times[-1]); previous_values = tuple(float(array[-1]) for array in values)
    if not chunks:
        return ()
    columns = [np.concatenate([chunk[index] for chunk in chunks]) for index in range(5)]
    times, opens, highs, lows, closes = columns
    days = times // 86_400_000
    unique_days, begins, counts = np.unique(days, return_index=True, return_counts=True)
    result = []
    for day, begin, count in zip(unique_days, begins, counts):
        begin = int(begin); count = int(count); stop = begin + count
        selected_times = times[begin:stop]
        day_open_ms = int(day) * 86_400_000
        if count < 274 or int(selected_times[0]) != day_open_ms or int(selected_times[-1]) != day_open_ms + 86_100_000 or (len(selected_times) > 1 and int(np.diff(selected_times).max()) > 900_000):
            continue
        close_ts = datetime.fromtimestamp((day_open_ms + 86_400_000) / 1000, tz=UTC)
        result.append(DailyBar(
            close_ts, float(opens[begin]), float(np.max(highs[begin:stop])), float(np.min(lows[begin:stop])),
            float(closes[stop - 1]), close_ts, close_ts, True,
        ))
    return tuple(result)


def _snapshot(rows: Sequence[Mapping[str, Any]], decision: datetime) -> dict[str, Any]:
    day_ms = int(decision.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    selected = [row for row in rows if int(row["day_open_ms"]) == day_ms]
    selected.sort(key=lambda row: (float(row["average_liquidity_rank"]), str(row["symbol"])))
    if not selected or selected[0]["symbol"] != "PF_XBTUSD":
        raise ProductionInputError("PF_XBTUSD is not the exact top-ranked PIT symbol at the production probe")
    eligible = tuple(str(row["symbol"]) for row in selected)
    ranks = {str(row["symbol"]): float(row["average_liquidity_rank"]) for row in selected}
    notionals = {str(row["symbol"]): float(row["lagged_30d_median_canonical_quote_notional"]) for row in selected}
    deciles = {
        symbol: 1 + min(9, int((ranks[symbol] - 1) * 10 / len(eligible)))
        for symbol in eligible
    }
    payload = {
        "as_of_ts": decision,
        "source_close_ts": decision,
        "feature_available_ts": decision,
        "eligible_symbols": eligible,
        "lagged_liquidity_ranks": ranks,
        "lagged_quote_notional": notionals,
        "lagged_liquidity_deciles": deciles,
        "top_n_symbols": {
            str(count): tuple(symbol for symbol in eligible if ranks[symbol] <= count)
            for count in (10, 20, 40)
        },
    }
    payload["source_sha256"] = canonical_hash({
        **payload,
        "as_of_ts": decision.isoformat(),
        "source_close_ts": decision.isoformat(),
        "feature_available_ts": decision.isoformat(),
    })
    return payload


def _returns(daily: Sequence[DailyBar], lookback: int) -> list[float]:
    return [log_return(daily[index - lookback].close, daily[index].close) for index in range(lookback, len(daily))]


def _a2_proximity_feature_arrays(
    bars: Sequence[SignalBar],
    daily: Sequence[DailyBar],
) -> tuple[Any, dict[str, Any]]:
    """Compute the exact A2 prior-level proximity values once per physical bar."""
    import numpy as np

    ordered_bars = tuple(sorted(bars, key=lambda row: row.open_ts))
    ordered_daily = tuple(sorted(daily, key=lambda row: row.close_ts))
    times = np.asarray([int(row.open_ts.timestamp() * 1000) for row in ordered_bars], dtype="<i8")
    closes = np.asarray([row.close for row in ordered_bars], dtype="<f8")
    if len(times) != len(np.unique(times)) or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ProductionInputError("A2 physical bars are duplicated or unsorted")
    arrays = {
        a2_context.proximity_population_key(lookback, atr_window, side): np.full(len(times), np.nan, dtype="<f8")
        for lookback in (20, 60, 120, 250)
        for atr_window in (10, 20, 40, 60)
        for side in (-1, 1)
    }
    for index, last_daily in enumerate(ordered_daily):
        day_ms = int(last_daily.close_ts.timestamp() * 1000)
        begin = int(np.searchsorted(times, day_ms, side="left"))
        end = int(np.searchsorted(times, day_ms + 86_400_000, side="left"))
        if begin == end:
            continue
        for lookback in (20, 60, 120, 250):
            if index + 1 < lookback:
                continue
            prior = ordered_daily[index - lookback + 1:index + 1]
            if prior[-1].close_ts - prior[0].close_ts != timedelta(days=lookback - 1):
                continue
            levels = {1: max(row.high for row in prior), -1: min(row.low for row in prior)}
            for atr_window in (10, 20, 40, 60):
                if index + 1 < atr_window + 1:
                    continue
                atr_rows = ordered_daily[index - atr_window:index + 1]
                if atr_rows[-1].close_ts - atr_rows[0].close_ts != timedelta(days=atr_window):
                    continue
                try:
                    atr = wilder_atr(
                        [row.high for row in atr_rows],
                        [row.low for row in atr_rows],
                        [row.close for row in atr_rows],
                        atr_window,
                    )
                except EngineInputError:
                    continue
                for side in (-1, 1):
                    level = levels[side]
                    arrays[a2_context.proximity_population_key(lookback, atr_window, side)][begin:end] = (
                        (closes[begin:end] - level) / atr
                        if side == 1 else
                        (level - closes[begin:end]) / atr
                    )
    return times, arrays


def _funding_rows(path: Path, symbol: str) -> tuple[tuple[datetime, str, float], ...]:
    rows = []
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"rankable_2023_2025/{symbol}.csv") as source:
            header = source.readline().decode("utf-8").strip().split(",")
            if header != ["timestamp", "tradeable", "absolute_rate", "relative_rate"]:
                raise ProductionInputError("funding partition schema differs")
            for raw in source:
                fields = raw.decode("utf-8").rstrip("\n").split(",")
                if len(fields) != 4 or fields[1] != symbol:
                    raise ProductionInputError("funding row identity differs")
                timestamp = _utc(fields[0].replace(" ", "T") + "Z")
                if timestamp >= datetime(2026, 1, 1, tzinfo=UTC):
                    raise ProductionInputError("protected funding row opened")
                rows.append((timestamp, fields[2], float(fields[3])))
    return tuple(rows)


def _thresholds(
    bars_by_symbol: Mapping[str, Sequence[SignalBar]],
    daily_by_symbol: Mapping[str, Sequence[DailyBar]],
    *,
    target: str,
    training_start: datetime,
    training_end: datetime,
    store: ExactPopulationStore | None = None,
    skip_a1: bool = False,
    skip_a3: bool = False,
    a2_proximity_arrays: tuple[Any, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, ThresholdPopulation], list[dict[str, Any]]]:
    symbols = tuple(sorted(bars_by_symbol))
    if target not in bars_by_symbol or target not in daily_by_symbol:
        raise ProductionInputError(f"threshold target is absent: {target}")
    target_bars = tuple(bar for bar in bars_by_symbol[target] if training_start <= bar.open_ts and bar.close_ts < training_end)
    target_daily = tuple(bar for bar in daily_by_symbol[target] if training_start < bar.close_ts < training_end)
    result: dict[str, ThresholdPopulation] = {}
    unavailable: list[dict[str, Any]] = []

    def add(name: str, values: Sequence[float], scope: str = "symbol", population_symbols: Sequence[str] = (target,)) -> None:
        population = _population(values, symbols=population_symbols, scope=scope, training_start=training_start, training_end=training_end, identity={"feature_signature": name, "training_start": training_start.isoformat(), "training_end": training_end.isoformat(), "scope": scope, "symbols": sorted(set(population_symbols))}, store=store)
        if population is None:
            unavailable.append({"feature_signature": name, "status": "unavailable_data", "reason": "minimum_30_rows_20_unique_not_met", "rows": len(values)})
        else:
            result[name] = population

    impulse_counts = {} if skip_a1 else {"6h": 72, "12h": 144, "1d": 288, "3d": 864, "7d": 2016}
    for window, count in impulse_counts.items():
        for side in (-1, 1):
            symbol_values = [
                side * log_return(target_bars[index - count].close, target_bars[index].close)
                for index in range(count, len(target_bars))
                if target_bars[index].open_ts - target_bars[index - count].open_ts == timedelta(minutes=5 * count)
            ]
            for scope in ("symbol_side", "liquidity_decile_side", "global_side"):
                values = list(symbol_values)
                population_symbols: Sequence[str] = (target,)
                if scope != "symbol_side":
                    values = []
                    for symbol in symbols:
                        rows = tuple(bar for bar in bars_by_symbol[symbol] if training_start <= bar.open_ts and bar.close_ts < training_end)
                        values.extend(
                            side * log_return(rows[index - count].close, rows[index].close)
                            for index in range(count, len(rows))
                            if rows[index].open_ts - rows[index - count].open_ts == timedelta(minutes=5 * count)
                        )
                    population_symbols = symbols
                add(a1_compression.impulse_population_key(window, scope, side), values, scope, population_symbols)

    base_counts = {} if skip_a1 else {"2h": 24, "6h": 72, "12h": 144, "1d": 288, "3d": 864}
    closes = [bar.close for bar in target_bars]
    def shape_values(rows: Sequence[SignalBar], count: int) -> tuple[dict[str, list[float]], list[float]]:
        local_closes = [bar.close for bar in rows]
        contractions: dict[str, list[float]] = {}
        for baseline_name, multiple in (("adjacent_equal_duration", 1), ("trailing_5x_base_duration", 5)):
            values = []
            baseline_count = count * multiple
            minimum_index = count + baseline_count - 1
            for index in range(minimum_index, len(local_closes)):
                if rows[index].open_ts - rows[index - minimum_index].open_ts != timedelta(minutes=5 * minimum_index):
                    continue
                base_closes = local_closes[index - count + 1:index + 1]
                baseline_closes = local_closes[index - count - baseline_count + 1:index - count + 1]
                try:
                    values.append(a1_compression.features(base_closes, base_closes, baseline_closes, 1)["contraction_ratio"])
                except EngineInputError:
                    continue
            contractions[baseline_name] = values
        smooth = []
        for index in range(count - 1, len(local_closes)):
            if rows[index].open_ts - rows[index - count + 1].open_ts != timedelta(minutes=5 * (count - 1)):
                continue
            try:
                smooth.append(path_smoothness(local_closes[index - count + 1:index + 1]))
            except EngineInputError:
                continue
        return contractions, smooth

    for base_name, count in base_counts.items():
        contraction_by_baseline, smoothness = shape_values(target_bars, count)
        for scope in ("symbol", "liquidity_decile", "global_PIT"):
            scoped_contractions = contraction_by_baseline
            scoped_smoothness = smoothness
            population_symbols: Sequence[str] = (target,)
            if scope != "symbol":
                scoped_contractions = {name: [] for name in contraction_by_baseline}
                scoped_smoothness = []
                for symbol in symbols:
                    rows = tuple(bar for bar in bars_by_symbol[symbol] if training_start <= bar.open_ts and bar.close_ts < training_end)
                    item_contractions, item_smoothness = shape_values(rows, count)
                    for name, values in item_contractions.items():
                        scoped_contractions[name].extend(values)
                    scoped_smoothness.extend(item_smoothness)
                population_symbols = symbols
            for baseline_name, values in contraction_by_baseline.items():
                add(a1_compression.contraction_population_key(base_name, baseline_name, scope), scoped_contractions[baseline_name], scope, population_symbols)
            add(a1_compression.smoothness_population_key(base_name, scope), scoped_smoothness, scope, population_symbols)

    if a2_proximity_arrays is not None:
        import numpy as np

        a2_times, a2_arrays = a2_proximity_arrays
        start_ms = int(training_start.timestamp() * 1000); end_ms = int(training_end.timestamp() * 1000)
        time_mask = (a2_times >= start_ms) & (a2_times + 300_000 < end_ms)
        for lookback in (20, 60, 120, 250):
            for atr_window in (10, 20, 40, 60):
                for side in (-1, 1):
                    name = a2_context.proximity_population_key(lookback, atr_window, side)
                    raw = a2_arrays[name]
                    add(name, np.asarray(raw[time_mask & np.isfinite(raw)], dtype="<f8"))

    for lookback in (() if a2_proximity_arrays is not None and skip_a3 else (20, 60, 120, 250)):
        for atr_window in (10, 20, 40, 60):
            for side in (-1, 1):
                values = []
                breakout_by_symbol: dict[str, list[float]] = {}
                for symbol in symbols:
                    symbol_daily = tuple(bar for bar in daily_by_symbol[symbol] if training_start < bar.close_ts < training_end)
                    symbol_bars = tuple(bar for bar in bars_by_symbol[symbol] if training_start <= bar.open_ts and bar.close_ts < training_end)
                    bars_by_day: dict[datetime, list[tuple[int, SignalBar]]] = defaultdict(list)
                    for bar_index, bar in enumerate(symbol_bars):
                        bars_by_day[bar.open_ts.replace(hour=0, minute=0, second=0, microsecond=0)].append((bar_index, bar))
                    crossings: list[float] = []
                    for index in range(max(lookback, atr_window + 1), len(symbol_daily) + 1):
                        required_days = max(lookback, atr_window + 1)
                        if symbol_daily[index - 1].close_ts - symbol_daily[index - required_days].close_ts != timedelta(days=required_days - 1):
                            continue
                        history_end = symbol_daily[index - 1].close_ts
                        day = history_end.replace(hour=0, minute=0, second=0, microsecond=0)
                        indexed_day_bars = sorted(bars_by_day.get(day, ()), key=lambda row: row[1].open_ts)
                        if not indexed_day_bars:
                            continue
                        day_bars = [row for _, row in indexed_day_bars]
                        prior = symbol_daily[index - lookback:index]
                        atr_rows = symbol_daily[index - atr_window - 1:index]
                        try:
                            atr = wilder_atr([row.high for row in atr_rows], [row.low for row in atr_rows], [row.close for row in atr_rows], atr_window)
                        except EngineInputError:
                            continue
                        level = max(row.high for row in prior) if side == 1 else min(row.low for row in prior)
                        if symbol == target:
                            values.extend((bar.close - level) / atr if side == 1 else (level - bar.close) / atr for bar in day_bars)
                        if not skip_a3:
                            first_index = indexed_day_bars[0][0]
                            previous = symbol_bars[first_index - 1] if first_index > 0 else None
                            for bar in day_bars:
                                crossed = previous is not None and (previous.close <= level < bar.close if side == 1 else previous.close >= level > bar.close)
                                if crossed:
                                    crossings.append((bar.close - level) / atr if side == 1 else (level - bar.close) / atr)
                                    break
                                previous = bar
                    breakout_by_symbol[symbol] = crossings
                add(a2_context.proximity_population_key(lookback, atr_window, side), values)
                if not skip_a3:
                    for scope in ("symbol_side", "liquidity_decile_side", "global_side"):
                        scoped = breakout_by_symbol[target] if scope == "symbol_side" else [value for symbol in symbols for value in breakout_by_symbol[symbol]]
                        add(a3_starter_retest.breakout_population_key(lookback, atr_window, scope, side), scoped, scope, symbols if scope != "symbol_side" else (target,))

    for asset, symbol in (("BTC", "PF_XBTUSD"), ("ETH", "PF_ETHUSD")):
        rows = tuple(bar for bar in daily_by_symbol[symbol] if training_start < bar.close_ts < training_end)
        for days in (5, 20, 60, 120):
            add(f"A2_{asset}_trend_{days}", [value for value in _returns(rows, days) for _ in range(288)])
        for days in (10, 20, 40, 60):
            values = []
            for index in range(days, len(rows)):
                returns = [log_return(left.close, right.close) for left, right in zip(rows[index - days:index], rows[index - days + 1:index + 1])]
                if len(returns) >= 2:
                    values.append(sample_standard_deviation(returns) * math.sqrt(365.0))
            add(f"A2_{asset}_volatility_{days}", [value for value in values for _ in range(288)])
        for days in (20, 60, 120):
            values = [1.0 - rows[index].close / max(row.close for row in rows[index - days + 1:index + 1]) for index in range(days - 1, len(rows))]
            add(f"A2_{asset}_drawdown_{days}", [value for value in values for _ in range(288)])

    absolute_move_prefix = [0.0]
    for left, right in zip(closes, closes[1:]):
        absolute_move_prefix.append(absolute_move_prefix[-1] + abs(log_return(left, right)))
    for lookback in (5, 10, 20, 40, 60, 90, 120, 180):
        count = lookback * 288
        smoothness = []
        for index in range(count, len(closes), 96):
            if target_bars[index].open_ts - target_bars[index - count].open_ts != timedelta(minutes=5 * count):
                continue
            path = absolute_move_prefix[index] - absolute_move_prefix[index - count]
            if path > 0:
                smoothness.append(abs(log_return(closes[index - count], closes[index])) / path)
        add(a4_tsmom.smoothness_population_key(lookback), smoothness)
        if len(closes) <= max(count + 288, 21 * 288):
            for estimator in ("close_to_close", "Parkinson"):
                for component in ("signed_return", "ema_slope", "breakout_distance_rank"):
                    add(a4_tsmom.ensemble_population_key(component, lookback, estimator), ())
            continue
        import numpy as np

        close_array = np.asarray(closes, dtype=np.float64)
        log_returns = np.diff(np.log(close_array))
        return_sum = np.concatenate(([0.0], np.cumsum(log_returns)))
        return_square_sum = np.concatenate(([0.0], np.cumsum(log_returns * log_returns)))
        high_array = np.asarray([row.high for row in target_bars], dtype=np.float64)
        low_array = np.asarray([row.low for row in target_bars], dtype=np.float64)
        parkinson_terms = np.log(high_array / low_array) ** 2
        parkinson_sum = np.concatenate(([0.0], np.cumsum(parkinson_terms)))
        ema_series = ema(closes, count)
        for estimator in ("close_to_close", "Parkinson"):
            for component in ("signed_return", "ema_slope", "breakout_distance_rank"):
                values = []
                for index in range(max(count + 288, 21 * 288), len(target_bars), 96):
                    try:
                        required = max(count + 288, 21 * 288)
                        if target_bars[index].open_ts - target_bars[index - required].open_ts != timedelta(minutes=5 * required):
                            continue
                        if estimator == "close_to_close":
                            sample_count = count
                            total = return_sum[index] - return_sum[index - count]
                            square_total = return_square_sum[index] - return_square_sum[index - count]
                            variance = (square_total - total * total / sample_count) / (sample_count - 1)
                            vol = math.sqrt(max(0.0, variance)) * math.sqrt(a4_tsmom.FIVE_MINUTE_PERIODS_PER_YEAR)
                        else:
                            total = parkinson_sum[index + 1] - parkinson_sum[index - count]
                            vol = math.sqrt((a4_tsmom.FIVE_MINUTE_PERIODS_PER_YEAR / (4.0 * math.log(2.0))) * (total / (count + 1)))
                        if vol <= 0:
                            continue
                        if component == "signed_return":
                            value = log_return(closes[index - count], closes[index]) / vol
                        elif component == "ema_slope":
                            value = log_return(ema_series[index - 288], ema_series[index]) / vol
                        else:
                            day_index = min(len(target_daily) - 1, index // 288)
                            prior = target_daily[max(0, day_index - lookback):day_index]
                            atr_rows = target_daily[max(0, day_index - 21):day_index]
                            if len(prior) < lookback or len(atr_rows) < 21:
                                continue
                            atr = wilder_atr([row.high for row in atr_rows], [row.low for row in atr_rows], [row.close for row in atr_rows], 20)
                            value = max((closes[index] - max(row.high for row in prior)) / atr, 0.0) + min((closes[index] - min(row.low for row in prior)) / atr, 0.0)
                        values.append(value)
                    except (EngineInputError, IndexError):
                        continue
                add(a4_tsmom.ensemble_population_key(component, lookback, estimator), values)
    return result, unavailable


def _context(
    decision: datetime,
    daily_by_symbol: Mapping[str, Sequence[DailyBar]],
    target_symbol: str,
    pit_snapshot: Mapping[str, Any],
    training_start: datetime,
    training_end: datetime,
    funding_relative: Sequence[tuple[datetime, float]],
    pit_rows_by_day: Mapping[int, Sequence[Mapping[str, Any]]],
) -> ContextInputs:
    histories = {symbol: tuple(row for row in rows if row.close_ts < decision) for symbol, rows in daily_by_symbol.items()}
    current_eligible = set(str(symbol) for symbol in pit_snapshot["eligible_symbols"])
    lookbacks = (5, 10, 20, 60, 120, 250)
    by_lookback: dict[int, dict[str, float]] = {}
    for lookback in lookbacks:
        by_lookback[lookback] = {
            symbol: log_return(rows[-lookback - 1].close, rows[-1].close)
            for symbol, rows in histories.items()
            if symbol in current_eligible and len(rows) >= lookback + 1
            and rows[-1].close_ts - rows[-lookback - 1].close_ts == timedelta(days=lookback)
        }
    breadth_history: dict[int, tuple[float, ...]] = {}
    dispersion_history: dict[int, tuple[float, ...]] = {}
    daily_maps = {symbol: {row.close_ts: row for row in rows} for symbol, rows in daily_by_symbol.items()}
    for lookback in (5, 20, 60):
        breadth = []; dispersion = []
        for day_ms, membership_rows in sorted(pit_rows_by_day.items()):
            day = datetime.fromtimestamp(day_ms / 1000, tz=UTC)
            if not (training_start <= day < training_end):
                continue
            values = []
            for membership in membership_rows:
                symbol = str(membership["symbol"])
                end_row = daily_maps.get(symbol, {}).get(day)
                start_row = daily_maps.get(symbol, {}).get(day - timedelta(days=lookback))
                if end_row is not None and start_row is not None:
                    values.append(log_return(start_row.close, end_row.close))
            if len(values) >= 5:
                repeats = next((int(row["decision_count_5m"]) for row in membership_rows if row["symbol"] == "PF_XBTUSD"), 0)
                breadth_value = sum(value > 0 for value in values) / len(values)
                center = median(values); dispersion_value = median(abs(value - center) for value in values)
                breadth.extend([breadth_value] * repeats)
                dispersion.extend([dispersion_value] * repeats)
        breadth_history[lookback] = tuple(breadth)
        dispersion_history[lookback] = tuple(dispersion)
    deciles = pit_snapshot["lagged_liquidity_deciles"]
    payload = {
        "decision": decision.isoformat(),
        "symbols": sorted(histories),
        "returns": by_lookback,
        "training_start": training_start.isoformat(),
        "training_end": training_end.isoformat(),
    }
    available_funding = [value for timestamp, value in funding_relative if training_start <= timestamp <= decision]
    return ContextInputs(
        btc_daily=histories["PF_XBTUSD"],
        eth_daily=histories["PF_ETHUSD"],
        symbol_daily=histories[target_symbol],
        breadth_history=breadth_history.get(20, ()),
        dispersion_history=dispersion_history.get(20, ()),
        breadth_history_by_lookback=breadth_history,
        dispersion_history_by_lookback=dispersion_history,
        cross_section_returns=by_lookback[20],
        cross_section_returns_by_lookback=by_lookback,
        cross_section_liquidity_deciles={symbol: int(deciles[symbol]) for symbol in histories if symbol in deciles},
        parent_universe=tuple(symbol for symbol in pit_snapshot["top_n_symbols"]["40"] if symbol in histories),
        funding_burden_history=tuple(available_funding[-720:-1]),
        funding_burden_current=available_funding[-1] if available_funding else None,
        as_of_ts=decision,
        source_close_ts=decision,
        feature_available_ts=decision,
        source_sha256=canonical_hash(payload),
    )


class ProductionFamilyInputBuilder:
    """Sole Stage-24 adapter from verified physical authority to FamilyInput."""

    def __init__(self, *, authority_path: Path, fold_graph_path: Path, output_root: Path, repository_root: Path) -> None:
        self.authority_path = authority_path
        self.fold_graph_path = fold_graph_path
        self.output_root = output_root
        self.repository_root = repository_root

    def build(self) -> dict[str, Any]:
        verifier = ProductionCacheCompiler(self.authority_path, self.output_root / "source_audit", self.repository_root)
        authority, roles = verifier._authority()
        unit = verifier._read_json(roles["price_and_instrument_source_manifest"], "price_unit_manifest")
        symbols = verifier._symbols(unit, roles["campaign_universe_reconciliation"])
        parts = verifier._source_parts(roles["kraken_acquisition_manifest"], set(symbols))
        verified = verifier._verify_parquet_parts(parts, physical_hashes=True)
        daily_trade_rows = verified.pop("daily_trade_rows")
        pit = verifier._pit_membership(roles["campaign_universe_reconciliation"], roles["terminal_lifecycle_source_ledger"], symbols, daily_trade_rows)
        verifier._funding(roles["rankable_funding_package"], symbols)
        kda = verifier._kda02b(roles)
        fold_graph = json.loads(self.fold_graph_path.read_text(encoding="utf-8"))
        if sha256_file(self.fold_graph_path) != authority["fold_graph_sha256"] or len(fold_graph.get("outer_folds", ())) != 8:
            raise ProductionInputError("fold graph authority mismatch")
        pit_rows = [json.loads(line) for line in (verifier.output_root / pit["artifact"]["path"]).read_text(encoding="utf-8").splitlines() if line]
        pit_rows_by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in pit_rows:
            pit_rows_by_day[int(row["day_open_ms"])].append(row)
        context_symbols = ("PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_DOGEUSD", "PF_ADAUSD", "PF_LINKUSD", "PF_AVAXUSD")
        selected_symbols = ("PF_XBTUSD", "PF_ADAUSD", "PF_AVAXUSD")
        full_bars = {
            symbol: _load_trade_bars(parts[symbol], datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
            for symbol in context_symbols
        }
        daily = {symbol: _valid_daily(rows) for symbol, rows in full_bars.items()}
        for symbol in symbols:
            if symbol not in daily:
                daily[symbol] = _load_daily_bars(parts[symbol], datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
        exact_funding_by_symbol = {
            symbol: _funding_rows(roles["rankable_funding_package"], symbol)
            for symbol in selected_symbols
        }
        a2_proximity_by_symbol = {
            symbol: _a2_proximity_feature_arrays(full_bars[symbol], daily[symbol])
            for symbol in selected_symbols
        }
        cache_root = self.output_root / "semantic_cache"
        population_store = ExactPopulationStore(cache_root)
        a1_table_build = A1PopulationTableCompiler(cache_root, parts, pit_rows).build()
        a1_population_authority = A1PopulationTableAuthority(cache_root, Path(a1_table_build["manifest_path"]))
        a3_table_build = A3PopulationTableCompiler(cache_root, parts, pit_rows, daily).build()
        a3_population_authority = A3PopulationTableAuthority(cache_root, Path(a3_table_build["manifest_path"]))
        writer = SemanticCacheWriter(cache_root, authority, authority_root=self.repository_root, synthetic_only=False)
        matrix: list[dict[str, Any]] = []
        feature_audit: list[dict[str, Any]] = []
        available_feature_signatures: set[str] = set()
        threshold_cache: dict[tuple[datetime, datetime, str, int], tuple[dict[str, ThresholdPopulation], list[dict[str, Any]]]] = {}
        partitions: list[dict[str, Any]] = []
        for outer in fold_graph["outer_folds"]:
            evaluation_start = _utc(outer["outer_evaluation_start"]); evaluation_end = _utc(outer["outer_evaluation_end_exclusive"])
            training_start = _utc(outer["development_start"]); training_end = evaluation_start - timedelta(days=int(outer["purge_days"]))
            partitions.append({
                "phase": "outer_evaluation", "outer_fold_id": outer["outer_fold_id"], "inner_fold_id": None,
                "training_start": training_start, "training_end_exclusive": training_end,
                "evaluation_start": evaluation_start, "evaluation_end_exclusive": evaluation_end,
                "decision": evaluation_start + timedelta(days=19),
            })
            for inner in outer["inner_folds"]:
                inner_evaluation_start = _utc(inner["validation_start"])
                inner_evaluation_end = _utc(inner["validation_end_exclusive"])
                partitions.append({
                    "phase": "inner_validation", "outer_fold_id": outer["outer_fold_id"],
                    "inner_fold_id": inner["inner_fold_id"],
                    "training_start": _utc(inner["training_start"]),
                    "training_end_exclusive": _utc(inner["training_latest_exit_exclusive"]),
                    "evaluation_start": inner_evaluation_start,
                    "evaluation_end_exclusive": inner_evaluation_end,
                    "decision": inner_evaluation_start + timedelta(days=19),
                })
        partitions.sort(key=lambda row: (
            row["training_end_exclusive"], row["evaluation_start"], row["phase"],
            row["outer_fold_id"], str(row["inner_fold_id"]),
        ))
        active_threshold_boundary: tuple[datetime, datetime] | None = None
        for partition_with_decision in partitions:
            partition = {key: value for key, value in partition_with_decision.items() if key != "decision"}
            decision = partition_with_decision["decision"]
            training_start = partition["training_start"]; training_end = partition["training_end_exclusive"]
            boundary = (training_start, training_end)
            if boundary != active_threshold_boundary:
                threshold_cache.clear()
                active_threshold_boundary = boundary
            snapshot = _snapshot(pit_rows, decision)
            history_start = decision - timedelta(days=181)
            frame_hashes: dict[str, str] = {}
            for symbol in selected_symbols:
                target_decile = int(snapshot["lagged_liquidity_deciles"][symbol])
                threshold_key = (training_start, training_end, symbol, target_decile)
                if threshold_key not in threshold_cache:
                    non_a1_populations, threshold_unavailable = _thresholds(
                        full_bars,
                        daily,
                        target=symbol,
                        training_start=training_start,
                        training_end=training_end,
                        store=population_store,
                        skip_a1=True,
                        skip_a3=True,
                        a2_proximity_arrays=a2_proximity_by_symbol[symbol],
                    )
                    a1_populations: dict[str, ThresholdPopulation] = {}
                    for window in ("6h", "12h", "1d", "3d", "7d"):
                        for side in (-1, 1):
                            for scope in ("symbol_side", "liquidity_decile_side", "global_side"):
                                name = a1_compression.impulse_population_key(window, scope, side)
                                a1_populations[name] = a1_population_authority.population(
                                    name, target_symbol=symbol, target_decile=target_decile,
                                    training_start=training_start, training_end=training_end,
                                )
                    for base in ("2h", "6h", "12h", "1d", "3d"):
                        for scope in ("symbol", "liquidity_decile", "global_PIT"):
                            for baseline in ("adjacent_equal_duration", "trailing_5x_base_duration"):
                                name = a1_compression.contraction_population_key(base, baseline, scope)
                                a1_populations[name] = a1_population_authority.population(
                                    name, target_symbol=symbol, target_decile=target_decile,
                                    training_start=training_start, training_end=training_end,
                                )
                            name = a1_compression.smoothness_population_key(base, scope)
                            a1_populations[name] = a1_population_authority.population(
                                name, target_symbol=symbol, target_decile=target_decile,
                                training_start=training_start, training_end=training_end,
                            )
                    a3_populations: dict[str, ThresholdPopulation] = {}
                    for lookback in (20, 60, 120, 250):
                        for atr_window in (10, 20, 40, 60):
                            for side in (-1, 1):
                                for scope in ("symbol_side", "liquidity_decile_side", "global_side"):
                                    name = a3_starter_retest.breakout_population_key(lookback, atr_window, scope, side)
                                    try:
                                        a3_populations[name] = a3_population_authority.population(
                                            name, target_symbol=symbol, target_decile=target_decile,
                                            training_start=training_start, training_end=training_end,
                                        )
                                    except PopulationTableError as exc:
                                        threshold_unavailable.append({
                                            "feature_signature": name,
                                            "status": "unavailable_data",
                                            "reason": str(exc),
                                            "rows": 0,
                                        })
                    threshold_cache[threshold_key] = (
                        {**non_a1_populations, **a1_populations, **a3_populations},
                        threshold_unavailable,
                    )
                populations, unavailable = threshold_cache[threshold_key]
                available_feature_signatures.update(populations)
                feature_audit.extend({
                    "phase": partition["phase"],
                    "outer_fold_id": partition["outer_fold_id"],
                    "inner_fold_id": partition["inner_fold_id"],
                    "symbol": symbol,
                    **item,
                } for item in unavailable)
                symbol_probe_bars = tuple(
                    bar for bar in full_bars[symbol]
                    if history_start <= bar.open_ts <= decision
                )
                if not symbol_probe_bars or symbol_probe_bars[-1].open_ts != decision:
                    raise ProductionInputError(f"real entry schedule open is absent: {symbol}")
                exact_funding = exact_funding_by_symbol[symbol]
                context = _context(
                    decision,
                    daily,
                    symbol,
                    snapshot,
                    training_start,
                    training_end,
                    tuple((timestamp, relative * 10000.0) for timestamp, _, relative in exact_funding),
                    pit_rows_by_day,
                )
                metadata = {
                    "production_input": True,
                    "evaluation_start": partition["evaluation_start"],
                    "evaluation_end_exclusive": partition["evaluation_end_exclusive"],
                    "eligible_days": int((partition["evaluation_end_exclusive"] - partition["evaluation_start"]).days),
                    "eligible_symbol_seconds": float((partition["evaluation_end_exclusive"] - partition["evaluation_start"]).total_seconds()),
                    "base_gap_allowance_bps_per_hour": 0.25,
                    "stress_gap_allowance_bps_per_hour": 0.50,
                    "pit_universe_snapshot": snapshot,
                    "campaign_partition": partition,
                    "a1_persistent_state": initial_state().payload(),
                    "a1_state_origin": "history_rebuild_at_complete_frame_start",
                    "execution_schedule_identity": canonical_hash({"symbol": symbol, "entry_open_ts": decision.isoformat(), "source": "verified_trade_schedule"}),
                    "protected_rows": 0,
                    "economic_outcomes_opened": False,
                    "cache_authority_components": tuple({
                        "path": Path(build["manifest_path"]).relative_to(cache_root).as_posix(),
                        "bytes": Path(build["manifest_path"]).stat().st_size,
                        "sha256": build["manifest_sha256"],
                        "encoding": "canonical_json",
                    } for build in (a1_table_build, a3_table_build)),
                }
                frame = FamilyInput(
                    KRAKEN_PLATFORM,
                    symbol,
                    str(partition["inner_fold_id"] or partition["outer_fold_id"]),
                    decision,
                    symbol_probe_bars,
                    tuple(row for row in daily[symbol] if row.close_ts < decision),
                    tuple(
                        FundingInput(timestamp, timestamp, absolute)
                        for timestamp, absolute, _ in exact_funding
                        if training_start <= timestamp <= decision
                    ),
                    populations,
                    context,
                    metadata,
                )
                frame = with_source_authority(frame, authority)
                frame.validate()
                record = writer.add(frame)
                frame_hashes[symbol] = record["frame_content_sha256"]
            for family in FAMILIES:
                if family == "KDA02B_SURVIVOR_ADJUDICATION_V1":
                    reason = "authorized Stage20 event tape has event identities but no raw decision-time derivative feature columns"
                    unavailable_record = writer.add_unavailable(
                        family_id=family,
                        partition=partition,
                        reason=reason,
                        authority_sha256=kda["event_tape_inventory_sha256"],
                    )
                    matrix.append({"family": family, "phase": partition["phase"], "outer_fold_id": partition["outer_fold_id"], "inner_fold_id": partition["inner_fold_id"], "status": "unavailable_data", "reason": reason, "authority_sha256": kda["event_tape_inventory_sha256"], "unavailable_identity_sha256": unavailable_record["unavailable_identity_sha256"]})
                else:
                    matrix.append({"family": family, "phase": partition["phase"], "outer_fold_id": partition["outer_fold_id"], "inner_fold_id": partition["inner_fold_id"], "status": "available", "frame_content_sha256_by_symbol": frame_hashes})
        manifest_path = writer.finalize()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "schema": "stage24_production_family_input_build_v1",
            "status": "pass",
            "cache_manifest": str(manifest_path),
            "cache_manifest_sha256": sha256_file(manifest_path),
            "family_fold_input_matrix": matrix,
            "family_fold_rows": len(matrix),
            "available_rows": sum(row["status"] == "available" for row in matrix),
            "typed_unavailable_rows": sum(row["status"] == "unavailable_data" for row in matrix),
            "feature_signatures_available": sorted(available_feature_signatures),
            "feature_signature_unavailable": feature_audit,
            "source_verification_sha256": canonical_hash(verified),
            "pit_membership_sha256": pit["membership_content_sha256"],
            "protected_rows": 0,
            "economic_outcomes_opened": False,
            "capitalcom_payload_opened": False,
            "source_index_cache_emitted": False,
            "cache_authority_schema": manifest["schema"],
            "a1_population_table_manifest_sha256": a1_table_build["manifest_sha256"],
            "a1_population_table_rows": a1_table_build["rows"],
            "a3_population_table_manifest_sha256": a3_table_build["manifest_sha256"],
            "a3_population_table_rows": sum(int(record["rows"]) for record in a3_table_build["features"].values()),
        }
        atomic_write_json(self.output_root / "PRODUCTION_FAMILY_INPUT_BUILD.json", report)
        return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build the authority-bound Stage 24 production FamilyInput cache")
    result.add_argument("--authority", type=Path, required=True)
    result.add_argument("--fold-graph", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--repository-root", type=Path, default=Path.cwd())
    return result


def main() -> int:
    args = parser().parse_args()
    report = ProductionFamilyInputBuilder(
        authority_path=args.authority,
        fold_graph_path=args.fold_graph,
        output_root=args.output,
        repository_root=args.repository_root,
    ).build()
    print(json.dumps({"status": report["status"], "cache_manifest_sha256": report["cache_manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FAMILIES", "ProductionFamilyInputBuilder", "ProductionInputError", "main"]
