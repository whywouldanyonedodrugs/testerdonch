from __future__ import annotations

import re
from typing import Any, Callable, Mapping

import pandas as pd

from tools.qlmg_rankable_source_contract import (
    RankableSourceContractError,
    filter_rankable_source_rows,
    read_rankable_source_payload,
)


CAPITALCOM_PLATFORM = "Capital.com"
PRICE_POINTS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = (
    "platform",
    "platform_epic",
    "instrument_type",
    "instrument_name",
    "currency",
    "contract_form",
    *(f"bid_{point}" for point in PRICE_POINTS),
    *(f"ask_{point}" for point in PRICE_POINTS),
    "bar_start_utc",
    "bar_end_utc",
    "availability_utc",
    "calendar_id",
    "metadata_snapshot_hash",
    "expiry_or_undated",
    "market_status",
    "volume_semantic_status",
    "financing_status",
    "corporate_action_status",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXPLICIT_STATUS_VALUES = {"known", "unknown", "not_applicable"}
_EXECUTABLE_MARKET_STATUSES = {"open", "tradeable", "tradable"}
_RANKABLE_START = pd.Timestamp("2023-01-01T00:00:00Z")
_PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")


def _explicit_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def _require_nonblank(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().isin({"", "nan", "none"}).any():
            raise RankableSourceContractError(f"Capital.com {column} missing")


def validate_capitalcom_bid_ask_bars(
    frame: pd.DataFrame,
    *,
    authority: Mapping[str, Any],
) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise RankableSourceContractError(f"Capital.com required fields missing: {missing}")
    out = filter_rankable_source_rows(
        frame,
        selected_platform="capital.com",
        event_time_column="bar_end_utc",
    )
    if len(out) != len(frame):
        raise RankableSourceContractError("Capital.com payload contains pre-train or wrong-platform rows")

    identity_columns = (
        "platform_epic",
        "instrument_type",
        "instrument_name",
        "currency",
        "contract_form",
        "calendar_id",
        "expiry_or_undated",
        "market_status",
    )
    _require_nonblank(out, identity_columns)

    for column in ("bar_start_utc", "bar_end_utc", "availability_utc"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
        if out[column].isna().any():
            raise RankableSourceContractError(f"Capital.com {column} invalid")
    if (out["bar_start_utc"] >= out["bar_end_utc"]).any():
        raise RankableSourceContractError("Capital.com bar interval invalid")
    if (out["availability_utc"] < out["bar_end_utc"]).any():
        raise RankableSourceContractError("Capital.com availability precedes bar completion")
    if (out["bar_start_utc"] < _RANKABLE_START).any() or (out["availability_utc"] >= _PROTECTED_START).any():
        raise RankableSourceContractError("Capital.com row interval is outside the rankable boundary")
    manifest_start = pd.to_datetime(authority.get("minimum_event_time_utc"), utc=True, errors="coerce")
    manifest_maximum = pd.to_datetime(authority.get("maximum_event_time_utc"), utc=True, errors="coerce")
    manifest_end = pd.to_datetime(authority.get("interval_end_utc_exclusive"), utc=True, errors="coerce")
    if pd.isna(manifest_start) or (out["bar_end_utc"] < manifest_start).any():
        raise RankableSourceContractError("Capital.com payload precedes manifest interval")
    if pd.notna(manifest_maximum) and (out["bar_end_utc"] > manifest_maximum).any():
        raise RankableSourceContractError("Capital.com payload exceeds manifest maximum")
    if pd.notna(manifest_end) and (out["bar_end_utc"] >= manifest_end).any():
        raise RankableSourceContractError("Capital.com payload exceeds manifest exclusive end")

    price_columns = [f"{side}_{point}" for side in ("bid", "ask") for point in PRICE_POINTS]
    for column in price_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any() or (out[column] <= 0).any():
            raise RankableSourceContractError(f"Capital.com {column} invalid")
    for side in ("bid", "ask"):
        if (out[f"{side}_high"] < out[[f"{side}_open", f"{side}_close", f"{side}_low"]].max(axis=1)).any():
            raise RankableSourceContractError(f"Capital.com {side} OHLC high ordering invalid")
        if (out[f"{side}_low"] > out[[f"{side}_open", f"{side}_close", f"{side}_high"]].min(axis=1)).any():
            raise RankableSourceContractError(f"Capital.com {side} OHLC low ordering invalid")
    for point in PRICE_POINTS:
        if (out[f"ask_{point}"] < out[f"bid_{point}"]).any():
            raise RankableSourceContractError(f"Capital.com bid/ask ordering invalid at {point}")

    metadata_hash = out["metadata_snapshot_hash"].astype(str).str.strip().str.lower()
    if not metadata_hash.map(lambda value: bool(_SHA256_PATTERN.fullmatch(value))).all():
        raise RankableSourceContractError("Capital.com metadata snapshot hash invalid")
    volume_status = out["volume_semantic_status"].astype(str).str.strip().str.lower()
    if not volume_status.isin({"verified", "unverified"}).all():
        raise RankableSourceContractError("Capital.com volume semantic status invalid")
    if volume_status.eq("verified").any() and not _explicit_true(authority.get("volume_semantics_verified", False)):
        raise RankableSourceContractError("Capital.com volume semantics lack fixture authority")
    for column in ("financing_status", "corporate_action_status"):
        status = out[column].astype(str).str.strip().str.lower()
        if not status.isin(_EXPLICIT_STATUS_VALUES).all():
            raise RankableSourceContractError(f"Capital.com {column} must be explicit known or unknown")

    out["platform"] = CAPITALCOM_PLATFORM
    out["price_semantics"] = "otc_cfd_bid_ask"
    return out.sort_values(["availability_utc", "bar_end_utc", "platform_epic"], kind="mergesort").reset_index(drop=True)


def load_capitalcom_bid_ask_bars(
    authority: Mapping[str, Any] | None,
    *,
    payload_reader: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    frame = read_rankable_source_payload(
        authority,
        selected_platform="capital.com",
        payload_reader=payload_reader,
    )
    return validate_capitalcom_bid_ask_bars(frame, authority=authority or {})


def hypothetical_execution_price(
    row: Mapping[str, Any],
    *,
    side: str,
    price_point: str = "open",
) -> float:
    side_key = str(side).strip().lower()
    point = str(price_point).strip().lower()
    if side_key not in {"buy", "sell"} or point not in PRICE_POINTS:
        raise RankableSourceContractError("Capital.com execution side or price point invalid")
    quote_side = "ask" if side_key == "buy" else "bid"
    value = float(row[f"{quote_side}_{point}"])
    if value <= 0:
        raise RankableSourceContractError("Capital.com execution quote invalid")
    return value


def first_executable_target_bar(
    bars: pd.DataFrame,
    *,
    earliest_availability_utc: Any,
) -> pd.Series:
    earliest = pd.to_datetime(earliest_availability_utc, utc=True, errors="coerce")
    if pd.isna(earliest):
        raise RankableSourceContractError("Capital.com executable target time invalid")
    eligible = bars[
        bars["availability_utc"].ge(earliest)
        & bars["market_status"].astype(str).str.strip().str.lower().isin(_EXECUTABLE_MARKET_STATUSES)
    ]
    if eligible.empty:
        raise RankableSourceContractError("Capital.com no executable target quote available")
    return eligible.sort_values(["availability_utc", "bar_end_utc", "platform_epic"], kind="mergesort").iloc[0]
