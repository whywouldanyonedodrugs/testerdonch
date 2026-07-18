from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, TypeVar

import pandas as pd


RANKABLE_START_UTC = datetime(2023, 1, 1, tzinfo=timezone.utc)
PROTECTED_START_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)
RANKABLE_PURPOSE = "rankable_research"
RECOGNIZED_PLATFORMS = {"kraken", "capital.com"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


class RankableSourceContractError(RuntimeError):
    pass


def canonical_platform(value: Any) -> str:
    platform = str(value or "").strip().lower()
    if platform not in RECOGNIZED_PLATFORMS:
        raise RankableSourceContractError(f"rankable source platform rejected: {value!r}")
    return platform


def _utc(value: Any, field: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise RankableSourceContractError(f"rankable source {field} missing")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RankableSourceContractError(f"rankable source {field} invalid") from exc
    if parsed.tzinfo is None:
        raise RankableSourceContractError(f"rankable source {field} must be UTC-aware")
    return parsed.astimezone(timezone.utc)


def _require_sha256(authority: Mapping[str, Any], field: str) -> str:
    value = str(authority.get(field, "")).strip().lower()
    if not _SHA256_PATTERN.fullmatch(value):
        raise RankableSourceContractError(f"rankable source {field} missing or invalid")
    return value


def assert_rankable_source_authority(
    authority: Mapping[str, Any] | None,
    *,
    selected_platform: str,
    funding: bool = False,
) -> None:
    """Fail closed on a source manifest before its payload reader is called."""
    if not isinstance(authority, Mapping):
        raise RankableSourceContractError("rankable source authority missing or unprovable")

    platform = canonical_platform(authority.get("platform"))
    if platform != canonical_platform(selected_platform):
        raise RankableSourceContractError("rankable source selected adapter mismatch")
    if str(authority.get("purpose", "")).strip() != RANKABLE_PURPOSE:
        raise RankableSourceContractError("rankable source purpose rejected")
    if not str(authority.get("source_dataset_id", "")).strip():
        raise RankableSourceContractError("rankable source dataset identity missing")
    _require_sha256(authority, "schema_hash")
    _require_sha256(authority, "content_sha256")

    start = _utc(authority.get("minimum_event_time_utc"), "minimum_event_time_utc")
    maximum = authority.get("maximum_event_time_utc")
    end_exclusive = authority.get("interval_end_utc_exclusive")
    if bool(str(maximum or "").strip()) == bool(str(end_exclusive or "").strip()):
        raise RankableSourceContractError(
            "rankable source must declare exactly one maximum timestamp or exclusive interval end"
        )
    if start < RANKABLE_START_UTC:
        raise RankableSourceContractError("rankable source pre-2023 or mixed interval rejected")
    if start >= PROTECTED_START_UTC:
        raise RankableSourceContractError("rankable source protected interval rejected")

    if str(end_exclusive or "").strip():
        end = _utc(end_exclusive, "interval_end_utc_exclusive")
        if end <= start or end > PROTECTED_START_UTC:
            raise RankableSourceContractError("rankable source exclusive interval end rejected")
    else:
        end = _utc(maximum, "maximum_event_time_utc")
        if end < start or end >= PROTECTED_START_UTC:
            raise RankableSourceContractError("rankable source maximum timestamp rejected")

    if funding and str(authority.get("funding_type", "")).strip().lower() != "exact":
        raise RankableSourceContractError("rankable source funding type rejected")


def read_rankable_source_payload(
    authority: Mapping[str, Any] | None,
    *,
    selected_platform: str,
    payload_reader: Callable[[], _T],
    funding: bool = False,
) -> _T:
    assert_rankable_source_authority(
        authority,
        selected_platform=selected_platform,
        funding=funding,
    )
    return payload_reader()


def filter_rankable_source_rows(
    frame: pd.DataFrame,
    *,
    selected_platform: str,
    event_time_column: str,
    platform_column: str = "platform",
) -> pd.DataFrame:
    """Filter non-selected and pre-train rows before rankable downstream use."""
    if platform_column not in frame.columns or event_time_column not in frame.columns:
        raise RankableSourceContractError("rankable source row identity or event time missing")
    platform = canonical_platform(selected_platform)
    row_platform = frame[platform_column].map(canonical_platform)
    event_time = pd.to_datetime(frame[event_time_column], utc=True, errors="coerce")
    if event_time.isna().any():
        raise RankableSourceContractError("rankable source row event time invalid")
    if (event_time >= pd.Timestamp(PROTECTED_START_UTC)).any():
        raise RankableSourceContractError("rankable source payload contains protected rows")
    keep = row_platform.eq(platform) & event_time.ge(pd.Timestamp(RANKABLE_START_UTC))
    return frame.loc[keep].copy()


def assert_legacy_kraken_rankable_authority(
    authority: Mapping[str, Any] | None,
    *,
    source_label: str,
    funding: bool = False,
) -> None:
    """Preserve the existing Kraken manifest contract without changing identities."""
    prefix = "rankable file authority"
    if not isinstance(authority, Mapping):
        raise RankableSourceContractError(f"{prefix} missing or unprovable: {source_label}")
    if str(authority.get("purpose", "")) != RANKABLE_PURPOSE:
        raise RankableSourceContractError(f"{prefix} purpose rejected: {source_label}")
    if str(authority.get("venue", "")).strip().lower() != "kraken":
        raise RankableSourceContractError(f"{prefix} venue rejected: {source_label}")
    contains_protected = str(authority.get("contains_protected_period", "false")).strip().lower() in {"true", "1"}
    rankable_pre_holdout = str(authority.get("rankable_pre_holdout", "false")).strip().lower() in {"true", "1"}
    if contains_protected or not rankable_pre_holdout:
        raise RankableSourceContractError(f"{prefix} status rejected: {source_label}")
    start = pd.to_datetime(authority.get("start_ts"), utc=True, errors="coerce")
    end = pd.to_datetime(authority.get("end_ts"), utc=True, errors="coerce")
    end_exclusive = str(authority.get("end_ts_exclusive", "false")).strip().lower() in {"true", "1"}
    protected = pd.Timestamp(PROTECTED_START_UTC)
    if pd.isna(start) or pd.isna(end) or start > end or end > protected or (end == protected and not end_exclusive):
        raise RankableSourceContractError(f"{prefix} interval rejected: {source_label}")
    if funding and str(authority.get("funding_type", "")) != "exact":
        raise RankableSourceContractError(f"{prefix} funding type rejected: {source_label}")


def directed_cross_platform_contract_id(
    *,
    source_platform: str,
    source_instrument_id: str,
    target_platform: str,
    target_instrument_id: str,
    contract_version: str,
) -> str:
    payload = {
        "contract_version": str(contract_version).strip(),
        "source_instrument_id": str(source_instrument_id).strip(),
        "source_platform": canonical_platform(source_platform),
        "target_instrument_id": str(target_instrument_id).strip(),
        "target_platform": canonical_platform(target_platform),
    }
    if not all(payload.values()):
        raise RankableSourceContractError("directed cross-platform identity field missing")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
