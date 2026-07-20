"""Cryptographic launch-gate records for the approved Stage 20 campaign."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]

from tools.qlmg_stage20_campaign import (
    APPROVAL_SHA256, CAMPAIGN_ID, FUNDING_PACKAGE, FUNDING_PACKAGE_SHA256,
    MANIFEST_SHA256, PACKET_SHA256, STAGE19_REL, START_COMMIT,
    Stage20Error, atomic_json, authority_audit, file_sha256, stable_hash, utc_now,
)
from tools.qlmg_stage19_funding import Stage19FundingEngine
from tools.validate_stage19_campaign_packet import validate as validate_stage19


GATE_SCHEMA = "stage20_bound_gate_v1"


def binding_context() -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "repository_commit": START_COMMIT,
        "approval_sha256": APPROVAL_SHA256,
        "campaign_manifest_sha256": MANIFEST_SHA256,
        "approval_packet_sha256": PACKET_SHA256,
        "approved_phases": [2, 3, 4, 5],
        "approved_lanes": ["KDA02B", "KDA02C", "KDX01"],
        "executable_cells": 186,
        "inherited_non_executable_attempts": 42,
        "programme_attempts": 228,
    }


def artifact_inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted({Path(item).resolve() for item in paths}, key=str):
        if not path.is_file():
            raise Stage20Error(f"bound gate artifact missing: {path}")
        records.append({"path": str(path), "bytes": path.stat().st_size,
                        "sha256": file_sha256(path)})
    return records


def gate_record(gate_id: str, artifacts: Iterable[Path], assertions: dict[str, Any],
                *, status: str = "pass") -> dict[str, Any]:
    if status != "pass":
        raise Stage20Error("only passing decisions may become launch gate records")
    payload = {
        "schema": GATE_SCHEMA, "gate_id": gate_id, "status": status,
        "created_at_utc": utc_now(), "binding_context": binding_context(),
        "artifacts": artifact_inventory(artifacts), "assertions": assertions,
    }
    payload["binding_sha256"] = stable_hash(payload)
    return payload


def write_gate(path: Path, gate_id: str, artifacts: Iterable[Path],
               assertions: dict[str, Any]) -> dict[str, Any]:
    record = gate_record(gate_id, artifacts, assertions)
    atomic_json(path, record)
    return record


def validate_gate(path: Path, expected_gate_id: str) -> dict[str, Any]:
    record = json.loads(path.read_text())
    binding = record.pop("binding_sha256", None)
    if (record.get("schema") != GATE_SCHEMA or record.get("gate_id") != expected_gate_id
            or record.get("status") != "pass" or record.get("binding_context") != binding_context()
            or binding != stable_hash(record)):
        raise Stage20Error(f"bound gate identity or signature invalid: {expected_gate_id}")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Stage20Error(f"bound gate artifact inventory absent: {expected_gate_id}")
    for item in artifacts:
        target = Path(item["path"])
        if (not target.is_file() or target.stat().st_size != int(item["bytes"])
                or file_sha256(target) != item["sha256"]):
            raise Stage20Error(f"bound gate artifact drift: {expected_gate_id}:{target}")
    return {**record, "binding_sha256": binding}


def assert_gate_binds(gate: dict[str, Any], path: Path) -> None:
    target = path.resolve()
    matches = [row for row in gate.get("artifacts", []) if Path(row["path"]) == target]
    if len(matches) != 1:
        raise Stage20Error(f"gate {gate.get('gate_id')} does not bind artifact: {target}")
    row = matches[0]
    if (target.stat().st_size != int(row["bytes"]) or file_sha256(target) != row["sha256"]):
        raise Stage20Error(f"gate-bound artifact drift: {target}")


def build_source_manifest(output: Path, paths: Iterable[Path]) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != START_COMMIT:
        raise Stage20Error("source-manifest repository commit drift")
    payload = {
        "schema": "stage20_reviewed_source_manifest_v1", "status": "frozen",
        "created_at_utc": utc_now(), "binding_context": binding_context(),
        "files": artifact_inventory(paths),
    }
    payload["manifest_sha256"] = stable_hash(payload)
    atomic_json(output, payload)
    return payload


def validate_source_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    signature = value.pop("manifest_sha256", None)
    if (value.get("schema") != "stage20_reviewed_source_manifest_v1"
            or value.get("status") != "frozen"
            or value.get("binding_context") != binding_context()
            or signature != stable_hash(value)):
        raise Stage20Error("reviewed source manifest identity invalid")
    for item in value.get("files", []):
        target = Path(item["path"])
        if (not target.is_file() or target.stat().st_size != int(item["bytes"])
                or file_sha256(target) != item["sha256"]):
            raise Stage20Error(f"reviewed source drift: {target}")
    return {**value, "manifest_sha256": signature}


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repository, text=True).strip()


def final_launch_boundary_audit(*, approval: Path, source_manifest: dict[str, Any],
                                event_root: Path, launch_gate: dict[str, Any],
                                run_root: Path, fetch_remote: bool = True) -> dict[str, Any]:
    """Repeat every immutable and operational input check immediately before launch."""
    if source_manifest.get("binding_context") != binding_context():
        raise Stage20Error("final source binding context drift")
    audit = authority_audit(ROOT, approval)
    head = _git_output(ROOT, "rev-parse", "HEAD")
    canonical = Path("/opt/testerdonch")
    if fetch_remote:
        fetched = subprocess.run(["git", "fetch", "origin", "main"], cwd=canonical,
                                 capture_output=True, text=True)
        if fetched.returncode:
            raise Stage20Error("final repository authority fetch failed")
    canonical_head = _git_output(canonical, "rev-parse", "HEAD")
    origin_main = _git_output(canonical, "rev-parse", "origin/main")
    canonical_branch = _git_output(canonical, "branch", "--show-current")
    if (head != START_COMMIT or canonical_head != START_COMMIT or origin_main != START_COMMIT
            or canonical_branch != "main"):
        raise Stage20Error("final repository authority drift")

    readiness = validate_stage19(ROOT / STAGE19_REL)
    if readiness.get("synthetic_canary") != "pass" or readiness.get("registered_cells") != 186:
        raise Stage20Error("final Stage 19 dependency/runtime validation failed")
    funding_contract_path = ROOT / STAGE19_REL / "FUNDING_COST_AND_COVERAGE_CONTRACT.json"
    funding_contract = json.loads(funding_contract_path.read_text())
    funding = Stage19FundingEngine(
        FUNDING_PACKAGE, FUNDING_PACKAGE_SHA256,
        ROOT / STAGE19_REL / "FUNDING_GAP_ALLOWANCE_TABLE.csv",
        funding_contract["gap_allowance_table_sha256"],
    )
    if len(funding.allowances) != 187:
        raise Stage20Error("final funding runtime coverage mismatch")

    event_manifest_path = event_root / "PREOUTCOME_EVENT_TAPE_MANIFEST.json"
    event_manifest = json.loads(event_manifest_path.read_text())
    files = event_manifest.get("files", [])
    if (event_manifest.get("status") != "pass" or event_manifest.get("registered_cells") != 186
            or event_manifest.get("protected_rows_opened") != 0
            or event_manifest.get("economic_outcome_reader_opened") is not False
            or event_manifest.get("Capitalcom_payload_opened") is not False
            or len(files) != 187 or len({row.get("symbol") for row in files}) != 187
            or any(not row.get("symbol") for row in files)):
        raise Stage20Error("final event-partition authority failed")
    for row in files:
        path = Path(row["path"])
        if (not path.is_file() or path.stat().st_size != int(row["bytes"])
                or file_sha256(path) != row["sha256"]):
            raise Stage20Error(f"final event partition hash drift: {path}")

    expected_limits = {
        "max_workers": 4, "max_wall_seconds": 14400,
        "max_rss_bytes": 5 * 1024**3, "max_output_bytes": 5 * 1024**3,
        "heartbeat_seconds": 1800,
    }
    if launch_gate.get("assertions", {}).get("runtime_limits") != expected_limits:
        raise Stage20Error("final launch runtime-limit binding drift")
    if shutil.disk_usage(run_root).free < expected_limits["max_output_bytes"]:
        raise Stage20Error("final launch free-disk gate failed")
    record = {
        "schema": "stage20_atomic_launch_boundary_v1", "status": "pass",
        "verified_at_utc": utc_now(), "binding_context": binding_context(),
        "authority_audit": audit, "reviewed_source_manifest_sha256": source_manifest["manifest_sha256"],
        "final_launch_gate_binding_sha256": launch_gate["binding_sha256"],
        "repository": {"worktree_head": head, "canonical_head": canonical_head,
                       "origin_main": origin_main, "canonical_branch": canonical_branch},
        "funding_package_sha256": file_sha256(FUNDING_PACKAGE),
        "funding_symbols": len(funding.allowances),
        "event_manifest_sha256": file_sha256(event_manifest_path),
        "event_partitions_verified": len(files), "runtime_limits": expected_limits,
        "protected_rows_opened": 0, "Capitalcom_payload_opened": False,
    }
    record["audit_sha256"] = stable_hash(record)
    atomic_json(run_root / "LAUNCH_BOUNDARY_VERIFICATION.json", record)
    return record
