#!/usr/bin/env python3
"""Pure validation helpers for Kraken candle base-volume authority.

The module has no strategy or outcome interface.  It validates bounded public
execution/candle observations and constructs the explicitly named lagged
close-based USD volume proxy used only for C01 cohort hygiene.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROXY_FIELD = "close_based_usd_volume_proxy"
FORBIDDEN_CLAIMS = {"traded_usd_notional", "quote_volume", "capacity", "executable_liquidity"}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def exact_decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid exact decimal: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite exact decimal: {value!r}")
    return result


def aggregate_execution_interval(
    pages: Sequence[Mapping[str, Any]], *, symbol: str, start_ms: int, end_ms: int,
) -> tuple[Decimal, int]:
    """Aggregate one side of each public execution exactly once."""
    if end_ms <= start_ms or start_ms % 300_000 or end_ms % 300_000:
        raise ValueError("calibration interval must be an aligned positive five-minute interval")
    seen: set[str] = set()
    total = Decimal("0")
    rows = 0
    for page in pages:
        elements = page.get("elements")
        if not isinstance(elements, list):
            raise ValueError("public execution page has no elements list")
        for element in elements:
            if not isinstance(element, Mapping):
                raise ValueError("invalid public execution element")
            event = element.get("event", {})
            execution = event.get("Execution", {}).get("execution", {}) if isinstance(event, Mapping) else {}
            uid = str(execution.get("uid", ""))
            timestamp = int(execution.get("timestamp", -1))
            tradeable = str(execution.get("makerOrder", {}).get("tradeable", ""))
            if not uid or uid in seen:
                raise ValueError("missing or duplicate public execution UID")
            if tradeable != symbol:
                raise ValueError(f"wrong public execution symbol: {tradeable!r}")
            if not start_ms <= timestamp < end_ms:
                raise ValueError("public execution outside requested half-open interval")
            quantity = exact_decimal(execution.get("quantity"))
            if quantity <= 0:
                raise ValueError("public execution quantity must be positive")
            seen.add(uid)
            total += quantity
            rows += 1
    return total, rows


def validate_candle_interval(
    candles: Iterable[Mapping[str, Any]], *, symbol: str, start_ms: int,
    execution_volume: Decimal,
) -> dict[str, Any]:
    matching = [row for row in candles if int(row.get("time", -1)) == start_ms]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one matching candle for {symbol}:{start_ms}")
    candle_volume = exact_decimal(matching[0].get("volume"))
    difference = execution_volume - candle_volume
    return {
        "symbol": symbol,
        "interval_start_ms": start_ms,
        "execution_volume": format(execution_volume, "f"),
        "candle_volume": format(candle_volume, "f"),
        "exact_difference": format(difference, "f"),
        "exact_match": difference == 0,
    }


def validate_semantic_versions(rows: pd.DataFrame) -> pd.DataFrame:
    """Fail closed where a PF symbol's observed official base/min-lot changes."""
    required = {"symbol", "snapshot_ts", "base_currency", "min_lot", "source_sha256"}
    if not required.issubset(rows.columns):
        raise ValueError(f"semantic rows missing: {sorted(required - set(rows.columns))}")
    work = rows.copy()
    work["snapshot_ts"] = pd.to_datetime(work["snapshot_ts"], utc=True, errors="raise")
    work["min_lot_normalized"] = work["min_lot"].map(lambda value: format(exact_decimal(value).normalize(), "f"))
    work["semantic_tuple"] = list(zip(work["base_currency"].astype(str), work["min_lot_normalized"]))
    counts = work.groupby("symbol", sort=True)["semantic_tuple"].nunique()
    work["semantic_consistent"] = work["symbol"].map(counts).eq(1)
    work["semantic_version"] = work.apply(
        lambda row: canonical_hash({
            "symbol": row["symbol"], "base_currency": row["base_currency"],
            "min_lot": row["min_lot_normalized"], "source_sha256": row["source_sha256"],
        }), axis=1,
    )
    return work.drop(columns="semantic_tuple").sort_values(["symbol", "snapshot_ts"], kind="mergesort")


def daily_close_based_proxy(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "source_open_ts", "close", "volume"}
    if not required.issubset(bars.columns):
        raise ValueError(f"proxy bars missing: {sorted(required - set(bars.columns))}")
    work = bars.copy()
    work["source_open_ts"] = pd.to_datetime(work["source_open_ts"], utc=True, errors="raise")
    work["close"] = pd.to_numeric(work["close"], errors="raise")
    work["volume"] = pd.to_numeric(work["volume"], errors="raise")
    if (work[["close", "volume"]] < 0).any().any() or work["source_open_ts"].duplicated().any():
        raise ValueError("proxy bars contain negative values or duplicate timestamps")
    work[PROXY_FIELD] = work["close"] * work["volume"]
    work["utc_day"] = work["source_open_ts"].dt.floor("D")
    return (
        work.groupby(["symbol", "utc_day"], sort=True, as_index=False)[PROXY_FIELD]
        .sum().sort_values(["utc_day", "symbol"], kind="mergesort").reset_index(drop=True)
    )


def lagged_top_n_membership(
    daily: pd.DataFrame, *, top_n: int = 100, lookback_days: int = 30, minimum_valid_days: int = 20,
) -> pd.DataFrame:
    """Rank once per UTC day using only complete days strictly before it."""
    required = {"symbol", "utc_day", PROXY_FIELD}
    if not required.issubset(daily.columns):
        raise ValueError(f"daily proxy missing: {sorted(required - set(daily.columns))}")
    work = daily.copy()
    work["utc_day"] = pd.to_datetime(work["utc_day"], utc=True, errors="raise").dt.floor("D")
    if work.duplicated(["symbol", "utc_day"]).any():
        raise ValueError("duplicate symbol-day proxy row")
    days = sorted(work["utc_day"].unique())
    output: list[dict[str, Any]] = []
    for day in days:
        day = pd.Timestamp(day)
        prior = work[(work["utc_day"] < day) & (work["utc_day"] >= day - pd.Timedelta(days=lookback_days))]
        stats = prior.groupby("symbol", sort=True)[PROXY_FIELD].agg(["median", "count"]).reset_index()
        stats = stats[stats["count"] >= minimum_valid_days].copy()
        stats = stats.sort_values(["median", "symbol"], ascending=[False, True], kind="mergesort")
        stats["rank"] = range(1, len(stats) + 1)
        for row in stats.itertuples(index=False):
            output.append({
                "utc_day": day, "symbol": row.symbol,
                "prior_30d_median_close_based_usd_volume_proxy": row.median,
                "valid_prior_days": int(row.count), "rank": int(row.rank),
                "top_100_eligible": int(row.rank) <= top_n,
            })
    return pd.DataFrame(output)


def assert_proxy_claim_boundary(columns: Iterable[str], contract_text: str) -> None:
    names = {str(value).lower() for value in columns}
    forbidden = sorted(token for token in FORBIDDEN_CLAIMS if token in names)
    if forbidden:
        raise ValueError(f"forbidden liquidity claims: {forbidden}")
    if PROXY_FIELD not in names or PROXY_FIELD not in contract_text:
        raise ValueError("close-based USD volume proxy label is missing")
    for claim in ("not exact quote volume", "not capacity", "prior UTC day"):
        if claim.lower() not in contract_text.lower():
            raise ValueError(f"proxy claim boundary missing: {claim}")
