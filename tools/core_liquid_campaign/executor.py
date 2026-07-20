from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .canonical import sha256_file
from .schema import CAMPAIGN_ID, economic_address, normalize_config
from .selection import EventObservation, aggregate_streaming


RANKABLE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
PROTECTED_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class ExecutionAuthorization:
    campaign_id: str
    manifest_sha256: str
    approval_request_sha256: str
    supplied_external_approval_sha256: str | None
    economic_run_authorized: bool

    def require(self, expected_manifest_sha256: str, expected_approval_request_sha256: str) -> None:
        if self.campaign_id != CAMPAIGN_ID:
            raise AuthorizationError("campaign identity mismatch")
        if self.manifest_sha256 != expected_manifest_sha256 or self.approval_request_sha256 != expected_approval_request_sha256:
            raise AuthorizationError("manifest/approval-request hash mismatch")
        if not self.economic_run_authorized or not self.supplied_external_approval_sha256:
            raise AuthorizationError("exact external economic approval is absent")


@dataclass(frozen=True)
class CacheAuthority:
    platform: str
    interval_start: datetime
    interval_end_exclusive: datetime
    source_manifest_sha256: str
    content_sha256: str
    pit_universe_sha256: str
    funding_manifest_sha256: str
    protected_row_count: int
    capitalcom_row_count: int

    def validate(self) -> None:
        if self.platform != "kraken_native_linear_pf":
            raise AuthorizationError("only Kraken native linear PF cache authority is allowed")
        if self.interval_start.tzinfo is None or self.interval_end_exclusive.tzinfo is None:
            raise AuthorizationError("cache interval timestamps must be timezone-aware")
        if self.interval_start < RANKABLE_START or self.interval_end_exclusive > PROTECTED_START or self.interval_start >= self.interval_end_exclusive:
            raise AuthorizationError("cache interval crosses the rankable/protected boundary")
        if self.protected_row_count != 0 or self.capitalcom_row_count != 0:
            raise AuthorizationError("protected or Capital.com rows are present")
        for name, value in (
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("content_sha256", self.content_sha256),
            ("pit_universe_sha256", self.pit_universe_sha256),
            ("funding_manifest_sha256", self.funding_manifest_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise AuthorizationError(f"invalid {name}")


def validate_registered_attempt(row: Mapping[str, Any]) -> None:
    if row.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("registered attempt campaign mismatch")
    family = str(row["family_id"])
    normalized = normalize_config(family, row["config"])
    if normalized != row["config"]:
        raise ValueError("registered attempt config is not canonical")
    _, address = economic_address(family, normalized)
    if address != row["canonical_economic_address_sha256"]:
        raise ValueError("registered attempt economic-address mismatch")


def execute_registered_attempt(
    row: Mapping[str, Any],
    events: Iterable[EventObservation],
    *,
    cache_authority: CacheAuthority,
    authorization: ExecutionAuthorization,
    expected_manifest_sha256: str,
    expected_approval_request_sha256: str,
) -> dict[str, Any]:
    """Future approved entry point. Stage 22 tests only its fail-closed boundary."""
    authorization.require(expected_manifest_sha256, expected_approval_request_sha256)
    cache_authority.validate()
    validate_registered_attempt(row)
    result = aggregate_streaming(events)
    return {
        "campaign_id": CAMPAIGN_ID,
        "executable_attempt_id": row["executable_attempt_id"],
        "canonical_economic_address_sha256": row["canonical_economic_address_sha256"],
        "cache_content_sha256": cache_authority.content_sha256,
        "aggregate": result,
    }


def synthetic_probe_attempt(row: Mapping[str, Any], events: Iterable[EventObservation]) -> dict[str, Any]:
    validate_registered_attempt(row)
    return {"synthetic_only": True, "economic_outcomes_opened": False, "aggregate": aggregate_streaming(events)}
