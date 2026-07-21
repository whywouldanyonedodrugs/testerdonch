from __future__ import annotations

import csv
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .a1_state import initial_state, transition
from .cache import SemanticCacheWriter
from .canonical import atomic_write_json, canonical_hash, sha256_file
from .engine_types import ContextInputs, DailyBar, FamilyInput, FundingInput, KRAKEN_PLATFORM, SignalBar, ThresholdPopulation
from .family_engines import a1_compression, a2_context, a3_starter_retest, a4_tsmom
from .family_engines.common import EngineInputError, ema, log_return, path_smoothness, sample_standard_deviation, wilder_atr
from .production_cache import ALLOWED_CANDLE_COLUMNS, ProductionCacheCompiler, ProductionCacheError, SourcePart
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
) -> ThresholdPopulation | None:
    finite = tuple(float(value) for value in values if math.isfinite(float(value)))
    if len(finite) < 30 or len(set(finite)) < 20:
        return None
    return ThresholdPopulation(
        finite,
        tuple(sorted(set(symbols))),
        scope,
        training_start,
        training_end,
        training_end - timedelta(minutes=5),
        training_end,
        canonical_hash({"identity": identity, "values": finite, "symbols": sorted(set(symbols))}),
    )


def _valid_daily(bars: Sequence[SignalBar]) -> tuple[DailyBar, ...]:
    grouped: dict[datetime, list[SignalBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.open_ts.replace(hour=0, minute=0, second=0, microsecond=0)].append(bar)
    rows = []
    for day, members in sorted(grouped.items()):
        members = sorted(members, key=lambda item: item.open_ts)
        if len(members) != 288 or members[0].open_ts != day or members[-1].open_ts != day + timedelta(hours=23, minutes=55):
            continue
        if any(right.open_ts - left.open_ts != timedelta(minutes=5) for left, right in zip(members, members[1:])):
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
    training_start: datetime,
    training_end: datetime,
) -> tuple[dict[str, ThresholdPopulation], list[dict[str, Any]]]:
    symbols = tuple(sorted(bars_by_symbol))
    target = "PF_XBTUSD"
    target_bars = tuple(bar for bar in bars_by_symbol[target] if training_start <= bar.open_ts and bar.close_ts < training_end)
    target_daily = tuple(bar for bar in daily_by_symbol[target] if training_start < bar.close_ts < training_end)
    result: dict[str, ThresholdPopulation] = {}
    unavailable: list[dict[str, Any]] = []

    def add(name: str, values: Sequence[float], scope: str = "symbol", population_symbols: Sequence[str] = (target,)) -> None:
        population = _population(values, symbols=population_symbols, scope=scope, training_start=training_start, training_end=training_end, identity={"feature_signature": name, "training_end": training_end.isoformat()})
        if population is None:
            unavailable.append({"feature_signature": name, "status": "unavailable_data", "reason": "minimum_30_rows_20_unique_not_met", "rows": len(values)})
        else:
            result[name] = population

    impulse_counts = {"6h": 72, "12h": 144, "1d": 288, "3d": 864, "7d": 2016}
    for window, count in impulse_counts.items():
        for side in (-1, 1):
            symbol_values = [side * log_return(target_bars[index - count].close, target_bars[index].close) for index in range(count, len(target_bars), 12)]
            for scope in ("symbol_side", "liquidity_decile_side", "global_side"):
                values = list(symbol_values)
                population_symbols: Sequence[str] = (target,)
                if scope != "symbol_side":
                    values = []
                    for symbol in symbols:
                        rows = tuple(bar for bar in bars_by_symbol[symbol] if training_start <= bar.open_ts and bar.close_ts < training_end)
                        values.extend(side * log_return(rows[index - count].close, rows[index].close) for index in range(count, len(rows), 72))
                    population_symbols = symbols
                add(a1_compression.impulse_population_key(window, scope, side), values, scope, population_symbols)

    base_counts = {"2h": 24, "6h": 72, "12h": 144, "1d": 288, "3d": 864}
    closes = [bar.close for bar in target_bars]
    close_returns = [log_return(left, right) for left, right in zip(closes, closes[1:])]
    return_prefix = [0.0]
    square_prefix = [0.0]
    absolute_move_prefix = [0.0]
    for value, left, right in zip(close_returns, closes, closes[1:]):
        return_prefix.append(return_prefix[-1] + value)
        square_prefix.append(square_prefix[-1] + value * value)
        absolute_move_prefix.append(absolute_move_prefix[-1] + abs(right - left))

    def rolling_std(start: int, end: int) -> float:
        count = end - start
        total = return_prefix[end] - return_prefix[start]
        square = square_prefix[end] - square_prefix[start]
        variance = (square - total * total / count) / (count - 1)
        return math.sqrt(max(0.0, variance))

    for base_name, count in base_counts.items():
        contraction_by_baseline: dict[str, list[float]] = {}
        for baseline_name, multiple in (("adjacent_equal_duration", 1), ("trailing_5x_base_duration", 5)):
            values = []
            for index in range(count * (multiple + 1), len(closes), 24):
                base_vol = rolling_std(index - count, index)
                prior_vol = rolling_std(index - count * (multiple + 1), index - count)
                if prior_vol > 0:
                    values.append(base_vol / prior_vol)
            contraction_by_baseline[baseline_name] = values
        smoothness = []
        for index in range(count, len(closes), 24):
            path = absolute_move_prefix[index] - absolute_move_prefix[index - count]
            if path > 0:
                smoothness.append(abs(closes[index] - closes[index - count]) / path)
        for scope in ("symbol", "liquidity_decile", "global_PIT"):
            for baseline_name, values in contraction_by_baseline.items():
                add(a1_compression.contraction_population_key(base_name, baseline_name, scope), values, scope, symbols if scope != "symbol" else (target,))
            add(a1_compression.smoothness_population_key(base_name, scope), smoothness, scope, symbols if scope != "symbol" else (target,))

    for lookback in (20, 60, 120, 250):
        for atr_window in (10, 20, 40, 60):
            for side in (-1, 1):
                values = []
                for index in range(max(lookback, atr_window + 1), len(target_daily)):
                    prior = target_daily[index - lookback:index]
                    atr_rows = target_daily[index - atr_window - 1:index]
                    try:
                        atr = wilder_atr([row.high for row in atr_rows], [row.low for row in atr_rows], [row.close for row in atr_rows], atr_window)
                    except EngineInputError:
                        continue
                    current = target_daily[index].close
                    level = max(row.high for row in prior) if side == 1 else min(row.low for row in prior)
                    values.append((current - level) / atr if side == 1 else (level - current) / atr)
                add(a2_context.proximity_population_key(lookback, atr_window, side), values)
                for scope in ("symbol_side", "liquidity_decile_side", "global_side"):
                    add(a3_starter_retest.breakout_population_key(lookback, atr_window, scope, side), values, scope, symbols if scope != "symbol_side" else (target,))

    for asset, symbol in (("BTC", "PF_XBTUSD"), ("ETH", "PF_ETHUSD")):
        rows = tuple(bar for bar in daily_by_symbol[symbol] if training_start < bar.close_ts < training_end)
        for days in (5, 20, 60, 120):
            add(f"A2_{asset}_trend_{days}", _returns(rows, days))
        for days in (10, 20, 40, 60):
            values = []
            for index in range(days, len(rows)):
                returns = [log_return(left.close, right.close) for left, right in zip(rows[index - days:index], rows[index - days + 1:index + 1])]
                if len(returns) >= 2:
                    values.append(sample_standard_deviation(returns) * math.sqrt(365.0))
            add(f"A2_{asset}_volatility_{days}", values)
        for days in (20, 60, 120):
            values = [1.0 - rows[index].close / max(row.close for row in rows[index - days + 1:index + 1]) for index in range(days - 1, len(rows))]
            add(f"A2_{asset}_drawdown_{days}", values)

    for lookback in (5, 10, 20, 40, 60, 90, 120, 180):
        count = lookback * 288
        smoothness = []
        for index in range(count, len(closes), 288):
            path = absolute_move_prefix[index] - absolute_move_prefix[index - count]
            if path > 0:
                smoothness.append(abs(closes[index] - closes[index - count]) / path)
        add(a4_tsmom.smoothness_population_key(lookback), smoothness)
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
                for index in range(max(count + 288, 21 * 288), len(target_bars), 288):
                    try:
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
) -> ContextInputs:
    histories = {symbol: tuple(row for row in rows if row.close_ts < decision) for symbol, rows in daily_by_symbol.items()}
    lookbacks = (5, 10, 20, 60, 120, 250)
    by_lookback: dict[int, dict[str, float]] = {}
    for lookback in lookbacks:
        by_lookback[lookback] = {
            symbol: log_return(rows[-lookback - 1].close, rows[-1].close)
            for symbol, rows in histories.items() if len(rows) >= lookback + 1
        }
    breadth_history: dict[int, tuple[float, ...]] = {}
    dispersion_history: dict[int, tuple[float, ...]] = {}
    training_daily = {symbol: tuple(row for row in rows if training_start < row.close_ts < training_end) for symbol, rows in daily_by_symbol.items()}
    for lookback in (5, 20, 60):
        breadth = []; dispersion = []
        maximum = max((len(rows) for rows in training_daily.values()), default=0)
        for index in range(lookback, maximum):
            values = [log_return(rows[index - lookback].close, rows[index].close) for rows in training_daily.values() if len(rows) > index]
            if len(values) >= 5:
                breadth.append(sum(value > 0 for value in values) / len(values))
                center = median(values); dispersion.append(median(abs(value - center) for value in values))
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
        selected_symbols = ("PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_DOGEUSD", "PF_ADAUSD", "PF_LINKUSD", "PF_AVAXUSD")
        full_bars = {
            symbol: _load_trade_bars(parts[symbol], datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
            for symbol in selected_symbols
        }
        daily = {symbol: _valid_daily(rows) for symbol, rows in full_bars.items()}
        exact_funding_by_symbol = {
            symbol: _funding_rows(roles["rankable_funding_package"], symbol)
            for symbol in selected_symbols
        }
        cache_root = self.output_root / "semantic_cache"
        writer = SemanticCacheWriter(cache_root, authority, authority_root=self.repository_root, synthetic_only=False)
        matrix: list[dict[str, Any]] = []
        feature_audit: list[dict[str, Any]] = []
        available_feature_signatures: set[str] = set()
        threshold_cache: dict[tuple[datetime, datetime], tuple[dict[str, ThresholdPopulation], list[dict[str, Any]]]] = {}
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
        for partition_with_decision in partitions:
            partition = {key: value for key, value in partition_with_decision.items() if key != "decision"}
            decision = partition_with_decision["decision"]
            training_start = partition["training_start"]; training_end = partition["training_end_exclusive"]
            snapshot = _snapshot(pit_rows, decision)
            threshold_key = (training_start, training_end)
            if threshold_key not in threshold_cache:
                threshold_cache[threshold_key] = _thresholds(full_bars, daily, training_start=training_start, training_end=training_end)
            populations, unavailable = threshold_cache[threshold_key]
            available_feature_signatures.update(populations)
            feature_audit.extend({
                "phase": partition["phase"],
                "outer_fold_id": partition["outer_fold_id"],
                "inner_fold_id": partition["inner_fold_id"],
                **item,
            } for item in unavailable)
            history_start = decision - timedelta(days=181)
            probe_bars = tuple(bar for bar in full_bars["PF_XBTUSD"] if history_start <= bar.open_ts <= decision)
            if not probe_bars or probe_bars[-1].open_ts != decision:
                raise ProductionInputError("real entry schedule open is absent")
            frame_hashes: dict[str, str] = {}
            for symbol in selected_symbols:
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
                )
                state = transition(initial_state(), timestamp=decision - timedelta(minutes=5), action="history_complete")
                state = transition(state, timestamp=decision, action="rearm", percentiles={1: 0.49, -1: 0.49})
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
                    "a1_persistent_state": state.payload(),
                    "execution_schedule_identity": canonical_hash({"symbol": symbol, "entry_open_ts": decision.isoformat(), "source": "verified_trade_schedule"}),
                    "protected_rows": 0,
                    "economic_outcomes_opened": False,
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
                    matrix.append({"family": family, "phase": partition["phase"], "outer_fold_id": partition["outer_fold_id"], "inner_fold_id": partition["inner_fold_id"], "status": "unavailable_data", "reason": "authorized Stage20 event tape has event identities but no raw decision-time derivative feature columns", "authority_sha256": kda["event_tape_inventory_sha256"]})
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
        }
        atomic_write_json(self.output_root / "PRODUCTION_FAMILY_INPUT_BUILD.json", report)
        return report


__all__ = ["FAMILIES", "ProductionFamilyInputBuilder", "ProductionInputError"]
