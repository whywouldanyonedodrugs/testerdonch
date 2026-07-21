from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .engine_types import ContextInputs, DailyBar, FamilyInput, FundingInput, KRAKEN_PLATFORM, SignalBar, ThresholdPopulation
from .canonical import canonical_hash
from .schema import baseline_config


UTC = timezone.utc


def population(start: float = -0.20, step: float = 0.01, *, scope: str = "symbol") -> ThresholdPopulation:
    values = tuple(start + step * index for index in range(40))
    return ThresholdPopulation(
        values,
        tuple(f"S{index:02d}" for index in range(10)),
        scope,
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2024, 12, 31, 23, 55, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
        canonical_hash({"synthetic_threshold_population": values, "scope": scope}),
    )


def _signal_bar(open_ts: datetime, price: float, next_price: float | None = None) -> SignalBar:
    close = price if next_price is None else next_price
    high = max(price, close) + 0.1
    low = min(price, close) - 0.1
    close_ts = open_ts + timedelta(minutes=5)
    return SignalBar(open_ts, close_ts, price, high, low, close, close_ts, close_ts)


def daily_history(anchor: datetime, count: int = 400, *, base: float = 100.0, trend: float = 0.03) -> tuple[DailyBar, ...]:
    rows = []
    for index in range(count):
        close_ts = anchor - timedelta(days=count - index)
        close = base + trend * index + math.sin(index / 3.0) * 0.4
        open_price = close - math.sin(index / 2.0) * 0.1
        rows.append(DailyBar(close_ts, open_price, max(open_price, close) + 1.0, min(open_price, close) - 1.0, close, close_ts, close_ts))
    return tuple(rows)


def context_inputs(anchor: datetime, symbol_daily: tuple[DailyBar, ...]) -> ContextInputs:
    btc = daily_history(anchor, 400, base=30000.0, trend=5.0)
    eth = daily_history(anchor, 400, base=2000.0, trend=0.7)
    returns = {"SYN": 0.08, "BTC": 0.04, "ETH": 0.05, "S1": -0.02, "S2": 0.01, "S3": 0.03}
    by_lookback = {
        days: {symbol: value * (days / 20.0) for symbol, value in returns.items()}
        for days in (5, 10, 20, 60, 120, 250)
    }
    breadth_history = {
        days: tuple(0.20 + index * 0.015 + days * 0.0001 for index in range(40))
        for days in (5, 20, 60)
    }
    dispersion_history = {
        days: tuple(0.01 + index * 0.001 + days * 0.00001 for index in range(40))
        for days in (5, 20, 60)
    }
    return ContextInputs(
        btc_daily=btc,
        eth_daily=eth,
        symbol_daily=symbol_daily,
        breadth_history=tuple(0.20 + index * 0.015 for index in range(40)),
        dispersion_history=tuple(0.01 + index * 0.001 for index in range(40)),
        breadth_history_by_lookback=breadth_history,
        dispersion_history_by_lookback=dispersion_history,
        cross_section_returns=returns,
        cross_section_returns_by_lookback=by_lookback,
        cross_section_liquidity_deciles={symbol: 5 for symbol in returns},
        parent_universe=tuple(returns),
        funding_burden_history=tuple(index * 0.05 for index in range(40)),
        funding_burden_current=0.1,
        as_of_ts=anchor,
        source_close_ts=anchor,
        feature_available_ts=anchor,
        source_sha256=canonical_hash({"synthetic_context": anchor.isoformat(), "symbols": sorted(returns)}),
    )


def threshold_populations() -> dict[str, ThresholdPopulation]:
    from .family_engines.a1_compression import contraction_population_key, impulse_population_key, smoothness_population_key as a1_smoothness_key
    from .family_engines.a2_context import proximity_population_key
    from .family_engines.a3_starter_retest import breakout_population_key
    from .family_engines.a4_tsmom import ensemble_population_key, smoothness_population_key as a4_smoothness_key

    result: dict[str, ThresholdPopulation] = {}
    for side in (-1, 1):
        for scope in ("symbol_side", "liquidity_decile_side", "global_side"):
            for window in ("6h", "12h", "1d", "3d", "7d"):
                result[impulse_population_key(window, scope, side)] = population(0.005, 0.001, scope=scope)
            for lookback in (5, 10, 20, 60):
                for atr_window in (10, 20, 40, 60):
                    result[breakout_population_key(lookback, atr_window, scope, side)] = population(0.01, 0.01, scope=scope)
        for lookback in (20, 60, 120, 250):
            for atr_window in (10, 20, 40, 60):
                result[proximity_population_key(lookback, atr_window, side)] = population(-2.0, 0.1)
    for scope in ("symbol", "liquidity_decile", "global_PIT"):
        for base in ("2h", "6h", "12h", "1d", "3d"):
            for baseline in ("adjacent_equal_duration", "trailing_5x_base_duration"):
                result[contraction_population_key(base, baseline, scope)] = population(0.05, 0.02, scope=scope)
            result[a1_smoothness_key(base, scope)] = population(0.0, 0.02, scope=scope)
    for lookback in (5, 10, 20, 40, 60, 90, 120, 180):
        result[a4_smoothness_key(lookback)] = population(0.0, 0.02)
        for volatility_estimator in ("close_to_close", "Parkinson"):
            for component in ("signed_return", "ema_slope", "breakout_distance_rank"):
                result[ensemble_population_key(component, lookback, volatility_estimator)] = population(-0.20, 0.01)
    for asset in ("BTC", "ETH"):
        for days in (5, 20, 60, 120):
            result[f"A2_{asset}_trend_{days}"] = population(-0.20, 0.01)
        for days in (10, 20, 40, 60):
            result[f"A2_{asset}_volatility_{days}"] = population(0.01, 0.005)
        for days in (20, 60, 120):
            result[f"A2_{asset}_drawdown_{days}"] = population(0.0, 0.01)
    return result


def _metadata(anchor: datetime) -> dict[str, Any]:
    eligible = ("SYN", *(f"PIT{index:02d}" for index in range(1, 50)))
    ranks = {symbol: index for index, symbol in enumerate(eligible, start=1)}
    return {
        "evaluation_start": anchor - timedelta(days=1),
        "evaluation_end_exclusive": datetime(2026, 1, 1, tzinfo=UTC),
        "eligible_days": 365,
        "eligible_symbol_seconds": 365.0 * 86400.0,
        "base_gap_allowance_bps_per_hour": 0.25,
        "stress_gap_allowance_bps_per_hour": 0.50,
        "current_breadth": 0.70,
        "current_dispersion": 0.02,
        "current_funding_burden_bps": 0.1,
        "control_symbol": "SYN",
        "control_year": anchor.year,
        "control_signal_sign": 1,
        "control_utc_month": anchor.strftime("%Y-%m"),
        "control_liquidity_decile": 5,
        "control_context_vector": (("RS", 0.8),),
        "control_utc_quarter": f"{anchor.year}Q{(anchor.month - 1) // 3 + 1}",
        "control_side": 1,
        "control_add_lag_bars": 3,
        "pit_universe_snapshot": {
            "as_of_ts": anchor,
            "source_close_ts": anchor,
            "feature_available_ts": anchor,
            "source_sha256": canonical_hash({"synthetic_pit_universe": anchor.isoformat()}),
            "eligible_symbols": eligible,
            "lagged_liquidity_ranks": ranks,
            "lagged_quote_notional": {symbol: float(1_000_000 - index) for index, symbol in enumerate(eligible)},
            "lagged_liquidity_deciles": {symbol: 1 + min(9, (index * 10) // len(eligible)) for index, symbol in enumerate(eligible)},
            "top_n_symbols": {str(count): eligible[:count] for count in (10, 20, 40)},
        },
        "synthetic_only": True,
        "campaign_partition": {
            "phase": "synthetic_canary",
            "outer_fold_id": "2025Q2",
            "inner_fold_id": "SYNTHETIC",
            "training_start": datetime(2023, 1, 1, tzinfo=UTC),
            "training_end_exclusive": datetime(2025, 1, 1, tzinfo=UTC),
            "evaluation_start": anchor - timedelta(days=1),
            "evaluation_end_exclusive": datetime(2026, 1, 1, tzinfo=UTC),
        },
    }


def a4_frame(config: Mapping[str, Any] | None = None, *, signal_sign: int | None = None, anchor: datetime | None = None) -> FamilyInput:
    config = dict(config or baseline_config("A4_TSMOM_V7")); anchor = anchor or datetime(2025, 6, 1, tzinfo=UTC)
    lookback = int(config["lookback_days"]); vol_window = int(config["vol_window_days"])
    history_bars = max(20 * 288, lookback * 288 + 289, vol_window * 288 + 1)
    start = anchor - timedelta(minutes=5 * history_bars)
    sign = float(signal_sign) if signal_sign is not None else (-1.0 if config["direction"] == "short_flat" else 1.0)
    prices = [500.0 + sign * 0.0008 * index + 0.03 * math.sin(index / 17.0) for index in range(history_bars + 3005)]
    bars = tuple(_signal_bar(start + timedelta(minutes=5 * index), prices[index], prices[index + 1]) for index in range(len(prices) - 1))
    daily = daily_history(anchor, max(400, lookback + 80), base=1000.0, trend=2.0 * sign)
    metadata = _metadata(anchor)
    return FamilyInput(KRAKEN_PLATFORM, "SYN", "2025Q2", anchor, bars, daily, (FundingInput(anchor + timedelta(hours=8), anchor + timedelta(hours=8), "1.0"),), threshold_populations(), context_inputs(anchor, daily), metadata)


def a1_frame(config: Mapping[str, Any] | None = None, *, side: int | None = None, anchor: datetime | None = None) -> FamilyInput:
    config = dict(config or baseline_config("A1_COMPRESSION_V2")); anchor = anchor or datetime(2025, 6, 1, tzinfo=UTC)
    impulse = {"6h": 72, "12h": 144, "1d": 288, "3d": 864, "7d": 2016}[str(config["impulse_window"])]
    base = {"2h": 24, "6h": 72, "12h": 144, "1d": 288, "3d": 864}[str(config["base_duration"])]
    baseline = base if config["contraction_baseline"] == "adjacent_equal_duration" else 5 * base
    candidate = max(impulse, baseline) + 2
    selected_side = side or (-1 if config["direction"] == "short" else 1)
    total = candidate + base + 4 + 3005
    values = [100.0 + (0.15 if index % 2 else -0.15) for index in range(total + 1)]
    values[candidate] = 105.0 if selected_side == 1 else 95.0
    for index in range(candidate + 1, candidate + base + 1):
        drift = (index - candidate) / max(1, base)
        values[index] = (104.5 + 0.2 * drift) if selected_side == 1 else (95.5 - 0.2 * drift)
    confirmation = candidate + base + 1
    for offset in range(4):
        values[confirmation + offset] = (106.0 + 0.02 * offset) if selected_side == 1 else (94.0 - 0.02 * offset)
    for index in range(confirmation + 4, len(values)):
        values[index] = values[confirmation + 3] + selected_side * 0.001 * (index - confirmation - 3)
    final_confirmation_index = confirmation - 1 if config["confirmation"] == "one_close" else confirmation if config["confirmation"] == "two_closes" else confirmation + 2
    start = anchor - timedelta(minutes=5 * (final_confirmation_index + 1))
    bars = tuple(_signal_bar(start + timedelta(minutes=5 * index), values[index], values[index + 1]) for index in range(total))
    daily = daily_history(anchor, 400)
    return FamilyInput(KRAKEN_PLATFORM, "SYN", "2025Q2", anchor, bars, daily, (), threshold_populations(), context_inputs(anchor, daily), _metadata(anchor))


def a3_frame(config: Mapping[str, Any] | None = None, *, anchor: datetime | None = None) -> FamilyInput:
    config = dict(config or baseline_config("A3_STARTER_RETEST_V3")); anchor = anchor or datetime(2025, 6, 1, tzinfo=UTC)
    side = 1 if config["direction"] == "long" else -1
    daily = list(daily_history(anchor, 400, base=99.0, trend=0.0))
    # Freeze an exact 100 prior high/low while retaining nonzero true range.
    daily = tuple(DailyBar(bar.close_ts, 99.0 if side == 1 else 101.0, 100.0 if side == 1 else 102.0, 98.0 if side == 1 else 100.0, 99.0 if side == 1 else 101.0, bar.source_close_ts, bar.feature_available_ts) for bar in daily)
    values = [99.0 if side == 1 else 101.0] * 4005
    values[1] = 99.0 if side == 1 else 101.0
    values[2] = 101.2 if side == 1 else 98.8
    values[4] = 101.3 if side == 1 else 98.7
    values[5] = 102.0 if side == 1 else 98.0
    values[6] = 100.5
    values[7] = 101.5 if side == 1 else 99.5
    values[8] = 102.0 if side == 1 else 98.0
    for index in range(9, len(values)):
        values[index] = values[8] + side * 0.001 * (index - 8)
    final_confirmation_index = 1 if config["confirmation"] == "one_close" else 2 if config["confirmation"] == "two_closes" else 4
    start = anchor - timedelta(minutes=5 * (final_confirmation_index + 1))
    bars = tuple(_signal_bar(start + timedelta(minutes=5 * index), values[index], values[index + 1]) for index in range(len(values) - 1))
    metadata = _metadata(anchor); metadata["control_side"] = side
    return FamilyInput(KRAKEN_PLATFORM, "SYN", "2025Q2", anchor, bars, daily, (), threshold_populations(), context_inputs(anchor, daily), metadata)


def kda_frame(config: Mapping[str, Any] | None = None, *, anchor: datetime | None = None) -> FamilyInput:
    config = dict(config or baseline_config("KDA02B_SURVIVOR_ADJUDICATION_V1")); anchor = anchor or datetime(2025, 6, 1, tzinfo=UTC)
    from .family_engines.kda02b_adjudication import cell_contract, cell_contract_sha256
    contract = cell_contract(str(config["stage20_cell_id"])); axes = contract["axes"]
    prices = [100.0 + 0.01 * index for index in range(700)]
    bars = tuple(_signal_bar(anchor + timedelta(minutes=5 * index), prices[index], prices[index + 1]) for index in range(len(prices) - 1))
    daily = daily_history(anchor, 100)
    metadata = _metadata(anchor)
    sign = -1.0 if axes["price_state"] == "negative" else 1.0
    raw = {"trade_return_1h": sign * 0.02, "mark_return_1h": sign * 0.02, "oi_log_change_1h": -0.02, "liquidation_base_units_1h": 1.0, "liquidation_intensity_robust_z": 2.0, "liquidation_normalization_valid": True, "generic_structure_predicate": True, "source_close_ts": anchor - timedelta(minutes=5), "feature_available_ts": anchor, "eligible": True, "known_lifecycle_mask": True, "trade_coverage": True, "mark_coverage": True, "analytics_coverage": True}
    prior = {**raw, "trade_return_1h": 0.0, "mark_return_1h": 0.0, "oi_log_change_1h": 0.0, "liquidation_base_units_1h": 0.0, "liquidation_intensity_robust_z": 0.0, "source_close_ts": anchor - timedelta(minutes=10), "feature_available_ts": anchor - timedelta(minutes=5)}
    metadata.update({"stage20_cell_id": config["stage20_cell_id"], "stage20_cell_contract_sha256": cell_contract_sha256(str(config["stage20_cell_id"])), "kda02b_feature_history": (prior, raw), "fold_thresholds": {"trade_abs_q80": 14.0, "trade_abs_q100": 500.0, "mark_abs_q80": 14.0, "mark_abs_q100": 500.0, "oi_q0": -0.12, "oi_q20": -0.01, "liquidation_q80": 1.0, "liquidation_q100": 5.0}})
    return FamilyInput(KRAKEN_PLATFORM, "SYN", "2025Q2", anchor, bars, daily, (), threshold_populations(), context_inputs(anchor, daily), metadata)


def frame_for_family(family: str, config: Mapping[str, Any]) -> FamilyInput:
    if family == "A4_TSMOM_V7": return a4_frame(config)
    if family == "A1_COMPRESSION_V2": return a1_frame(config)
    if family == "A3_STARTER_RETEST_V3": return a3_frame(config)
    if family == "KDA02B_SURVIVOR_ADJUDICATION_V1": return kda_frame(config)
    if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1": return a1_frame(baseline_config("A1_COMPRESSION_V2"))
    raise ValueError(family)


def with_source_authority(frame: FamilyInput, authority: Mapping[str, Any]) -> FamilyInput:
    bindings = {
        key: authority[key]
        for key in ("source_manifest_sha256", "pit_universe_sha256", "funding_manifest_sha256", "cache_contract_sha256", "fold_graph_sha256")
    }
    provenance = {
        **bindings,
        "source_record_inventory_sha256": canonical_hash(authority["source_records"]),
        "rankable_funding_package_sha256": authority.get("rankable_funding_package_sha256"),
    }
    return replace(frame, metadata={**frame.metadata, "source_authority": provenance})


__all__ = ["a1_frame", "a3_frame", "a4_frame", "context_inputs", "daily_history", "frame_for_family", "kda_frame", "population", "threshold_populations", "with_source_authority"]
