"""Pure frozen KDA01 Level-3 economic calculations."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260719


def branch_side(branch_id: str) -> int:
    mapping = {
        "positive_efficient_continuation": 1,
        "negative_efficient_continuation": -1,
        "positive_completed_failure": -1,
        "negative_completed_failure": 1,
    }
    suffix = branch_id.removeprefix("primary_").removeprefix("robustness_")
    if suffix not in mapping:
        raise ValueError(f"unknown frozen branch side: {branch_id}")
    return mapping[suffix]


def score_open_prices(entry: float, exit_: float, side: int) -> tuple[float, float, float]:
    values = (float(entry), float(exit_))
    if side not in {-1, 1} or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("invalid frozen open-price return input")
    gross = side * (values[1] / values[0] - 1.0) * 10_000.0
    return gross, gross - 14.0, gross - 32.0


def equal_cluster_returns(trades: pd.DataFrame, cluster: str) -> pd.DataFrame:
    required = {"definition_id", cluster, "gross_bps", "base_net_bps", "stress_net_bps"}
    if missing := required - set(trades.columns):
        raise ValueError(f"missing cluster inputs: {sorted(missing)}")
    return trades.groupby(["definition_id", cluster], sort=True, as_index=False).agg(
        trade_count=("event_id", "size"),
        gross_bps=("gross_bps", "mean"),
        base_net_bps=("base_net_bps", "mean"),
        stress_net_bps=("stress_net_bps", "mean"),
    )


def cluster_bootstrap(values: Iterable[float]) -> tuple[np.ndarray, float, float]:
    source = np.asarray(list(values), dtype=float)
    if not len(source) or not np.isfinite(source).all():
        raise ValueError("invalid cluster-bootstrap source")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    batch = 250
    for start in range(0, BOOTSTRAP_RESAMPLES, batch):
        count = min(batch, BOOTSTRAP_RESAMPLES - start)
        output[start:start + count] = source[rng.integers(0, len(source), size=(count, len(source)))].mean(axis=1)
    return output, float(np.percentile(output, 2.5)), float(np.percentile(output, 97.5))


def positive_contribution_share(trades: pd.DataFrame, group: str, value: str = "base_net_bps") -> float:
    grouped = trades.groupby(group, sort=True)[value].sum()
    positive = grouped[grouped > 0]
    denominator = float(positive.sum())
    if not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return float(positive.max() / denominator)


def gate_flags(row: dict[str, Any]) -> dict[str, bool]:
    finite = lambda name: math.isfinite(float(row[name]))
    flags = {
        "executed_trades_ge_100": int(row["accepted_count"]) >= 100,
        "each_year_ge_20": all(int(row[f"trades_{year}"]) >= 20 for year in (2023, 2024, 2025)),
        "equal_day_base_mean_positive": finite("equal_day_base_mean_bps") and float(row["equal_day_base_mean_bps"]) > 0,
        "equal_day_base_median_positive": finite("equal_day_base_median_bps") and float(row["equal_day_base_median_bps"]) > 0,
        "bootstrap_lower_ge_minus5": finite("bootstrap_lower_bps") and float(row["bootstrap_lower_bps"]) >= -5,
        "market_day_share_le_10pct": finite("market_day_positive_share") and float(row["market_day_positive_share"]) <= .10,
        "symbol_share_le_25pct": finite("symbol_positive_share") and float(row["symbol_positive_share"]) <= .25,
        "year_share_le_70pct": finite("year_positive_share") and float(row["year_positive_share"]) <= .70,
        "equal_day_stress_mean_ge_minus10": finite("equal_day_stress_mean_bps") and float(row["equal_day_stress_mean_bps"]) >= -10,
    }
    flags["all_gates_pass"] = all(flags.values())
    return flags
