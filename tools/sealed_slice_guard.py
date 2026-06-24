#!/usr/bin/env python3
"""Fail-closed guard for sealed research slices.

This module intentionally does not read market data. It only checks requested date
windows against a local sealed-slice registry and validates that any overlapping
candidate-selection access is backed by an explicit frozen contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO / "reports/sealed_slices/sealed_slice_registry.json"
DEFAULT_DD_ROOT = REPO / "reports/project_handover_due_diligence_20260621"
DEFAULT_SEAL_MANIFEST = DEFAULT_DD_ROOT / "seal_status/2026_03_06_to_2026_06_18_seal_manifest.json"
DEFAULT_HASH_MANIFEST = DEFAULT_DD_ROOT / "data_manifests/data_manifest_hashes.json"

CANDIDATE_SELECTION_PURPOSES = {
    "candidate_selection",
    "model_selection",
    "ranking",
    "tuning",
    "thresholding",
    "calibration",
    "holdout_review",
    "research_selection",
}


@dataclass(frozen=True)
class SealedSlice:
    slice_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    status: str
    manifest_hashes: Mapping[str, Any]
    external_attestation_status: str
    source_registry_path: str


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _parse_start(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="raise")


def _parse_end(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    # Date-only end arguments are inclusive through end-of-day.
    if len(str(value)) <= 10:
        ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts


def _overlaps(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    return a_start <= b_end and b_start <= a_end


def register_default_sealed_slice(registry_path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Register the locally known sealed-slice metadata from due-diligence artifacts."""
    if not DEFAULT_SEAL_MANIFEST.exists():
        raise FileNotFoundError(f"missing seal manifest: {DEFAULT_SEAL_MANIFEST}")
    if not DEFAULT_HASH_MANIFEST.exists():
        raise FileNotFoundError(f"missing data manifest hashes: {DEFAULT_HASH_MANIFEST}")
    seal = _read_json(DEFAULT_SEAL_MANIFEST)
    hashes = _read_json(DEFAULT_HASH_MANIFEST)
    sl = seal["slice"]
    entry = {
        "slice_id": "sealed_2026_03_06_to_2026_06_18",
        "start": sl["start"],
        "end": sl["end"],
        "status": "sealed_pending_external_attestation",
        "external_attestation_status": seal.get("status", "pending_external_attestation"),
        "created_at_utc": _utc(),
        "policy": (
            "No reading, summarizing, plotting, ranking, tuning, thresholding, calibrating, "
            "or candidate-selection inspection of this slice unless a candidate contract is frozen "
            "and explicitly declares this sealed slice."
        ),
        "manifest_hashes": hashes,
        "source_seal_manifest": str(DEFAULT_SEAL_MANIFEST.relative_to(REPO)),
        "source_data_manifest_hashes": str(DEFAULT_HASH_MANIFEST.relative_to(REPO)),
        "attestation_required": True,
        "attestation_artifact": str((DEFAULT_DD_ROOT / "seal_status/live_team_attestation_template.md").relative_to(REPO)),
    }
    registry = {
        "registry_version": 1,
        "updated_at_utc": _utc(),
        "slices": [entry],
    }
    _write_json(registry_path, registry)
    return registry


def load_registry(registry_path: Path = DEFAULT_REGISTRY) -> list[SealedSlice]:
    if not registry_path.exists():
        return []
    raw = _read_json(registry_path)
    out: list[SealedSlice] = []
    for item in raw.get("slices", []):
        out.append(
            SealedSlice(
                slice_id=str(item["slice_id"]),
                start=_parse_start(str(item["start"])),
                end=_parse_end(str(item["end"])),
                status=str(item.get("status", "unknown")),
                manifest_hashes=item.get("manifest_hashes", {}),
                external_attestation_status=str(item.get("external_attestation_status", "unknown")),
                source_registry_path=str(registry_path),
            )
        )
    return out


def overlapping_sealed_slices(start: str | pd.Timestamp, end: str | pd.Timestamp, registry_path: Path = DEFAULT_REGISTRY) -> list[SealedSlice]:
    start_ts = _parse_start(str(start)) if not isinstance(start, pd.Timestamp) else start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end_ts = _parse_end(str(end)) if not isinstance(end, pd.Timestamp) else end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    return [sl for sl in load_registry(registry_path) if _overlaps(start_ts, end_ts, sl.start, sl.end)]


def _contract_allows_slice(contract: Mapping[str, Any], sealed_slice: SealedSlice, purpose: str) -> tuple[bool, str]:
    if not bool(contract.get("contract_frozen", False)):
        return False, "contract_frozen_not_true"
    allowed = contract.get("sealed_slice_access", [])
    if not isinstance(allowed, list):
        return False, "sealed_slice_access_not_list"
    for entry in allowed:
        if not isinstance(entry, dict):
            continue
        if entry.get("slice_id") != sealed_slice.slice_id:
            continue
        purposes = set(map(str, entry.get("allowed_purposes", [])))
        if purpose not in purposes and "candidate_selection" not in purposes:
            return False, "purpose_not_allowed_by_contract"
        if not bool(entry.get("frozen_before_access", False)):
            return False, "frozen_before_access_not_true"
        return True, "allowed"
    return False, "slice_not_declared_in_contract"


def assert_sealed_slice_access_allowed(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    purpose: str,
    contract_path: str | Path | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> None:
    """Raise RuntimeError when sealed-slice access is not allowed.

    Non-candidate-selection purposes are still reported as safe by default here;
    callers should pass a candidate-selection purpose for any action that can
    influence models, thresholds, rankings, or strategy interpretation.
    """
    registry = Path(registry_path)
    overlaps = overlapping_sealed_slices(start, end, registry)
    if not overlaps:
        return
    normalized_purpose = str(purpose)
    if normalized_purpose not in CANDIDATE_SELECTION_PURPOSES:
        return
    if contract_path is None:
        ids = ",".join(sl.slice_id for sl in overlaps)
        raise RuntimeError(f"sealed-slice access denied for {normalized_purpose}: {ids}; no frozen contract supplied")
    cp = Path(contract_path)
    if not cp.exists():
        raise RuntimeError(f"sealed-slice access denied: frozen contract path does not exist: {cp}")
    contract = _read_json(cp)
    failures = []
    for sl in overlaps:
        ok, reason = _contract_allows_slice(contract, sl, normalized_purpose)
        if not ok:
            failures.append(f"{sl.slice_id}:{reason}")
    if failures:
        raise RuntimeError("sealed-slice access denied: " + ";".join(failures))


def add_sealed_slice_access_to_contract(contract_path: Path, *, start: str, end: str, purpose: str, registry_path: Path = DEFAULT_REGISTRY) -> None:
    """Mark a contract as frozen and explicitly authorize overlapping sealed slices."""
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    contract = _read_json(contract_path)
    contract["contract_frozen"] = True
    current = contract.get("sealed_slice_access", [])
    if not isinstance(current, list):
        current = []
    known = {(x.get("slice_id"), tuple(x.get("allowed_purposes", []))) for x in current if isinstance(x, dict)}
    for sl in overlapping_sealed_slices(start, end, registry_path):
        entry = {
            "slice_id": sl.slice_id,
            "allowed_purposes": [purpose],
            "frozen_before_access": True,
            "registered_data_manifest_hashes": sl.manifest_hashes,
            "external_attestation_status_at_freeze": sl.external_attestation_status,
            "note": "This does not imply live/remote non-use attestation is complete; it only records that this candidate contract was frozen before local sealed-slice access.",
        }
        key = (entry["slice_id"], tuple(entry["allowed_purposes"]))
        if key not in known:
            current.append(entry)
    contract["sealed_slice_access"] = current
    contract["contract_frozen_at_utc"] = contract.get("contract_frozen_at_utc") or _utc()
    _write_json(contract_path, contract)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register and enforce sealed-slice access policy.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register-default")
    check = sub.add_parser("check")
    check.add_argument("--start", required=True)
    check.add_argument("--end", required=True)
    check.add_argument("--purpose", required=True)
    check.add_argument("--contract-path", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "register-default":
        registry = register_default_sealed_slice()
        print(DEFAULT_REGISTRY)
        print(json.dumps(registry, indent=2, sort_keys=True, default=str))
        return 0
    if args.cmd == "check":
        assert_sealed_slice_access_allowed(
            start=args.start,
            end=args.end,
            purpose=args.purpose,
            contract_path=args.contract_path or None,
        )
        print("sealed-slice access check passed")
        return 0
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
