#!/usr/bin/env python3
"""Small, outcome-agnostic state engine for registered QLMG research campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REQUIRED_MANIFEST_FIELDS = {
    "campaign_id", "repository_and_data_hashes", "hypotheses", "fold_schedule",
    "search_spaces", "selection_algorithm", "candidate_beam", "cost_and_execution",
    "multiplicity", "phase_permissions", "resource_limits", "stop_conditions",
    "review_requirements", "archive_and_handoff",
}
STAGE14_REQUIRED_MANIFEST_FIELDS = {
    "campaign_id", "repository_and_data_hashes", "ready_hypotheses", "fold_schedule",
    "search_spaces", "selection_algorithm", "cost_and_execution", "multiplicity",
    "phase_permissions", "resource_limits", "stop_conditions",
}
FAMILY_STOPS = {
    "mechanically_unavailable", "no_development_candidate", "search_budget_exhausted",
    "mechanism_underidentified", "family_specific_defect",
}
GLOBAL_STOPS = {
    "shared_authority_failure", "protected_exposure", "shared_timestamp_defect",
    "unsafe_git_or_storage", "shared_replay_failure",
}
APPROVAL_REQUIRED_PHASES = {2, 3, 4, 5, 6, 7}
TRUSTED_APPROVAL_SHA256 = {
    "human_approval_kraken_derivatives_campaign_001_20260720_v1":
        "c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b",
}
RANKABLE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
RANKABLE_END = datetime(2026, 1, 1, tzinfo=timezone.utc)


class CampaignContractError(ValueError):
    """Raised when a campaign action violates the registered contract."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ids(items: Iterable[dict[str, Any]], field: str) -> list[str]:
    values = [str(item.get(field, "")) for item in items]
    if any(not value for value in values) or len(set(values)) != len(values):
        raise CampaignContractError(f"{field} values must be present and unique")
    return values


def manifest_hypotheses(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if "ready_hypotheses" in manifest:
        if "hypotheses" in manifest:
            raise CampaignContractError("manifest hypothesis alias substitution rejected")
        spaces = {str(item["family_id"]): str(item["search_space_id"])
                  for item in manifest.get("search_spaces", [])}
        rows = []
        for hypothesis_id in manifest["ready_hypotheses"]:
            family = str(hypothesis_id).split("_", 1)[0]
            if family not in spaces:
                raise CampaignContractError(f"Stage-14 hypothesis lacks search space: {hypothesis_id}")
            rows.append({"hypothesis_id": str(hypothesis_id),
                         "search_space_id": spaces[family],
                         "programme_exposure_class": manifest.get("programme_exposure_class")})
        return rows
    return list(manifest.get("hypotheses", []))


def candidate_beam_limit(manifest: dict[str, Any]) -> int:
    if "ready_hypotheses" in manifest:
        if "candidate_beam" in manifest:
            raise CampaignContractError("manifest candidate-beam alias substitution rejected")
        value = manifest.get("resource_limits", {}).get("candidate_beam_per_family")
    else:
        value = manifest.get("candidate_beam", {}).get("max_retained_per_hypothesis")
    if not isinstance(value, int) or value < 1:
        raise CampaignContractError("candidate beam must be a positive integer")
    return value


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CampaignContractError(f"timestamp must be UTC Z format: {value!r}")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CampaignContractError(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo != timezone.utc:
        raise CampaignContractError(f"timestamp must be UTC: {value!r}")
    return parsed


def validate_manifest(manifest: dict[str, Any]) -> None:
    stage14_schema = "ready_hypotheses" in manifest
    required = STAGE14_REQUIRED_MANIFEST_FIELDS if stage14_schema else REQUIRED_MANIFEST_FIELDS
    missing = sorted(required - manifest.keys())
    if missing:
        raise CampaignContractError(f"missing manifest fields: {missing}")
    hypotheses = manifest_hypotheses(manifest)
    hypothesis_ids = set(_ids(hypotheses, "hypothesis_id"))
    search_ids = set(_ids(manifest["search_spaces"], "search_space_id"))
    for hypothesis in hypotheses:
        if hypothesis.get("search_space_id") not in search_ids:
            raise CampaignContractError(f"unknown search space for {hypothesis['hypothesis_id']}")
        if hypothesis.get("programme_exposure_class") not in {
            "campaign_sealed_outer_fold", "program_exposed_historical", "protected_prospective"
        }:
            raise CampaignContractError("invalid programme exposure class")
    permissions = manifest["phase_permissions"]
    if any(str(phase) not in {str(i) for i in range(8)} for phase in permissions):
        raise CampaignContractError("phase permissions must use phases 0 through 7")
    fold_ids = _ids(manifest["fold_schedule"], "fold_id")
    if len(fold_ids) != len(set(fold_ids)):
        raise CampaignContractError("fold ids must be unique")
    for fold in manifest["fold_schedule"]:
        if fold.get("hypothesis_id") not in hypothesis_ids:
            raise CampaignContractError("fold references unknown hypothesis")
        start = parse_utc(fold["development_start"]); development_end = parse_utc(fold["development_end"])
        embargo_end = parse_utc(fold["embargo_end"]); evaluation_start = parse_utc(fold["evaluation_start"])
        evaluation_end = parse_utc(fold["evaluation_end"])
        if not (RANKABLE_START <= start < development_end <= embargo_end <= evaluation_start < evaluation_end <= RANKABLE_END):
            raise CampaignContractError("fold ordering or rankable boundary invalid")
    if manifest.get("economic_run_authorized_by_manifest", False):
        raise CampaignContractError("readiness manifest cannot self-authorize economics")
    candidate_beam_limit(manifest)


def build_dag(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    nodes: list[dict[str, Any]] = []
    for hypothesis in manifest_hypotheses(manifest):
        hid = hypothesis["hypothesis_id"]
        phase_0 = f"{hid}:phase_0"; phase_1 = f"{hid}:phase_1"
        nodes.extend([
            {"node_id": phase_0, "hypothesis_id": hid, "phase": 0, "fold_id": None, "depends_on": []},
            {"node_id": phase_1, "hypothesis_id": hid, "phase": 1, "fold_id": None, "depends_on": [phase_0]},
        ])
        previous = phase_1
        for fold in [item for item in manifest["fold_schedule"] if item["hypothesis_id"] == hid]:
            fold_id = fold["fold_id"]
            development = f"{fold_id}:phase_2"
            cell_nodes = [f"{fold_id}:cell:{cell_id}" for cell_id in next(
                item["registered_cell_ids"] for item in manifest["search_spaces"]
                if item["search_space_id"] == hypothesis["search_space_id"])]
            nodes.extend({"node_id": cell, "hypothesis_id": hid, "phase": 2,
                          "fold_id": fold_id, "depends_on": [development]} for cell in cell_nodes)
            freeze = f"{fold_id}:phase_3"
            evaluation = f"{fold_id}:phase_4"
            nodes.extend([
                {"node_id": development, "hypothesis_id": hid, "phase": 2, "fold_id": fold_id, "depends_on": [previous]},
                {"node_id": freeze, "hypothesis_id": hid, "phase": 3, "fold_id": fold_id, "depends_on": cell_nodes or [development]},
                {"node_id": evaluation, "hypothesis_id": hid, "phase": 4, "fold_id": fold_id, "depends_on": [freeze]},
            ])
            previous = evaluation
        for phase in (5, 6, 7):
            node_id = f"{hid}:phase_{phase}"
            nodes.append({"node_id": node_id, "hypothesis_id": hid, "phase": phase,
                          "fold_id": None, "depends_on": [previous]})
            previous = node_id
    return nodes


def validate_explored_cells(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    spaces = {space["search_space_id"]: set(space["registered_cell_ids"])
              for space in manifest["search_spaces"]}
    seen: set[tuple[str, str, str]] = set()
    hypotheses = {item["hypothesis_id"]: item for item in manifest_hypotheses(manifest)}
    folds = {item["fold_id"]: item for item in manifest["fold_schedule"]}
    for row in rows:
        key = (row["hypothesis_id"], row["fold_id"], row["cell_id"])
        if key in seen:
            raise CampaignContractError(f"duplicate explored cell: {key}")
        seen.add(key)
        hypothesis = hypotheses.get(row["hypothesis_id"]); fold = folds.get(row["fold_id"])
        if not hypothesis or not fold or fold["hypothesis_id"] != row["hypothesis_id"]:
            raise CampaignContractError("explored cell hypothesis/fold binding invalid")
        if hypothesis["search_space_id"] != row["search_space_id"]:
            raise CampaignContractError("explored cell search-space binding invalid")
        if row["search_space_id"] not in spaces or row["cell_id"] not in spaces[row["search_space_id"]]:
            raise CampaignContractError(f"unregistered search cell: {row['cell_id']}")


def validate_freeze(freeze: dict[str, Any], target_fold: dict[str, Any], folds: list[dict[str, Any]]) -> None:
    by_id = {fold["fold_id"]: fold for fold in folds}
    frozen_at = parse_utc(freeze["frozen_at"])
    if frozen_at > parse_utc(target_fold["evaluation_start"]):
        raise CampaignContractError("freeze occurs after outer evaluation begins")
    for source_id in freeze["source_fold_ids"]:
        source = by_id[source_id]
        if source["hypothesis_id"] != target_fold["hypothesis_id"]:
            raise CampaignContractError("cross-hypothesis fold cannot influence freeze")
        if parse_utc(source["evaluation_end"]) > frozen_at:
            raise CampaignContractError("source fold was not fully usable before freeze")


def validate_candidate_beam(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    limit = candidate_beam_limit(manifest)
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["hypothesis_id"], row["fold_id"])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > limit:
            raise CampaignContractError(f"candidate beam exceeded: {key}")


def apply_stop(state: dict[str, Any], reason: str, hypothesis_id: str | None = None) -> dict[str, Any]:
    result = json.loads(json.dumps(state))
    if reason in GLOBAL_STOPS:
        result["global_stop"] = {"reason": reason}
    elif reason in FAMILY_STOPS:
        if not hypothesis_id:
            raise CampaignContractError("family stop requires hypothesis_id")
        result.setdefault("family_stops", {})[hypothesis_id] = {"reason": reason}
    else:
        raise CampaignContractError(f"unregistered stop reason: {reason}")
    result["generation"] = int(result.get("generation", 0)) + 1
    return result


def enforce_phase_permission(manifest: dict[str, Any], phase: int, *, state: dict[str, Any] | None = None,
                             hypothesis_id: str | None = None, approval: dict[str, Any] | None = None,
                             approval_packet: dict[str, Any] | None = None,
                             approval_raw_bytes: bytes | None = None,
                             expected_approval_sha256: str | None = None,
                             manifest_raw_bytes: bytes | None = None,
                             approval_packet_raw_bytes: bytes | None = None,
                             launch_constraints: dict[str, Any] | None = None) -> None:
    if state and state.get("global_stop"):
        raise CampaignContractError("campaign is globally stopped")
    if state and hypothesis_id and hypothesis_id in state.get("family_stops", {}):
        raise CampaignContractError(f"hypothesis is stopped: {hypothesis_id}")
    if phase in APPROVAL_REQUIRED_PHASES:
        if state is None or not hypothesis_id:
            raise CampaignContractError("state and hypothesis_id are required for approved phases")
        if state.get("campaign_id") != manifest["campaign_id"] or state.get("manifest_sha256") != sha256_bytes(canonical_bytes(manifest)):
            raise CampaignContractError("campaign state identity mismatch")
        validate_external_approval(
            manifest, approval, approval_packet, phase, hypothesis_id,
            approval_raw_bytes=approval_raw_bytes,
            expected_approval_sha256=expected_approval_sha256,
            manifest_raw_bytes=manifest_raw_bytes,
            approval_packet_raw_bytes=approval_packet_raw_bytes,
            launch_constraints=launch_constraints,
        )
        external_override = (
            approval is not None
            and {"campaign_manifest_canonical_sha256", "approval_packet_canonical_sha256"}.issubset(approval)
            and approval_packet is not None
            and "phases_requested" in approval_packet
            and "ready_lanes" in approval_packet
        )
        if not manifest["phase_permissions"].get(str(phase), False) and not external_override:
            raise CampaignContractError("false readiness permission requires exact Stage-14 external approval")
        return
    if not manifest["phase_permissions"].get(str(phase), False):
        raise CampaignContractError(f"phase {phase} is not authorized")


def complete_node(manifest: dict[str, Any], state: dict[str, Any], node_id: str, *,
                  approval: dict[str, Any] | None = None,
                  approval_packet: dict[str, Any] | None = None,
                  approval_raw_bytes: bytes | None = None,
                  expected_approval_sha256: str | None = None,
                  manifest_raw_bytes: bytes | None = None,
                  approval_packet_raw_bytes: bytes | None = None,
                  launch_constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    nodes = {node["node_id"]: node for node in build_dag(manifest)}
    if node_id not in nodes:
        raise CampaignContractError(f"unknown DAG node: {node_id}")
    node = nodes[node_id]
    enforce_phase_permission(manifest, node["phase"], state=state,
                             hypothesis_id=node["hypothesis_id"], approval=approval,
                             approval_packet=approval_packet,
                             approval_raw_bytes=approval_raw_bytes,
                             expected_approval_sha256=expected_approval_sha256,
                             manifest_raw_bytes=manifest_raw_bytes,
                             approval_packet_raw_bytes=approval_packet_raw_bytes,
                             launch_constraints=launch_constraints)
    completed = set(state.get("completed_nodes", []))
    if node_id in completed:
        raise CampaignContractError("DAG node already completed")
    if not set(node["depends_on"]).issubset(completed):
        raise CampaignContractError("DAG node dependencies are incomplete")
    result = json.loads(json.dumps(state))
    result["completed_nodes"] = sorted(completed | {node_id})
    result["generation"] = int(result.get("generation", 0)) + 1
    return result


def validate_external_approval(manifest: dict[str, Any], approval: dict[str, Any] | None,
                               approval_packet: dict[str, Any] | None, phase: int,
                               hypothesis_id: str | None, *,
                               approval_raw_bytes: bytes | None = None,
                               expected_approval_sha256: str | None = None,
                               manifest_raw_bytes: bytes | None = None,
                               approval_packet_raw_bytes: bytes | None = None,
                               launch_constraints: dict[str, Any] | None = None) -> None:
    required = {"approval_id", "status", "human_authorized", "authorized_by", "authorized_at",
                "campaign_id", "campaign_manifest_sha256", "approval_packet_sha256",
                "approved_phases", "approved_hypotheses", "repository_and_data_hashes",
                "cost_and_execution_sha256"}
    if not approval or required - approval.keys() or not approval_packet:
        raise CampaignContractError("separate complete human approval artifact required")
    if approval["status"] != "approved" or approval["human_authorized"] is not True:
        raise CampaignContractError("approval artifact is not explicitly human-authorized")
    if approval["campaign_id"] != manifest["campaign_id"]:
        raise CampaignContractError("approval campaign mismatch")
    canonical_manifest_hash = sha256_bytes(canonical_bytes(manifest))
    canonical_packet_hash = sha256_bytes(canonical_bytes(approval_packet))
    dual_hash_contract = {
        "campaign_manifest_canonical_sha256", "approval_packet_canonical_sha256"
    }.issubset(approval)
    if dual_hash_contract:
        if approval_raw_bytes is None or not expected_approval_sha256:
            raise CampaignContractError("externally anchored human-approval bytes are required")
        trusted_hash = TRUSTED_APPROVAL_SHA256.get(str(approval.get("approval_id", "")))
        if trusted_hash is None or expected_approval_sha256 != trusted_hash:
            raise CampaignContractError("human-approval artifact lacks a trusted external anchor")
        if sha256_bytes(approval_raw_bytes) != expected_approval_sha256:
            raise CampaignContractError("human-approval artifact file-byte hash mismatch")
        if json.loads(approval_raw_bytes) != approval:
            raise CampaignContractError("parsed human approval differs from anchored bytes")
        if manifest_raw_bytes is None or approval_packet_raw_bytes is None:
            raise CampaignContractError("raw packet and manifest bytes are required by approval")
        if approval["campaign_manifest_sha256"] != sha256_bytes(manifest_raw_bytes):
            raise CampaignContractError("approval manifest file-byte hash mismatch")
        if approval["approval_packet_sha256"] != sha256_bytes(approval_packet_raw_bytes):
            raise CampaignContractError("approval packet file-byte hash mismatch")
        if approval["campaign_manifest_canonical_sha256"] != canonical_manifest_hash:
            raise CampaignContractError("approval manifest canonical hash mismatch")
        if approval["approval_packet_canonical_sha256"] != canonical_packet_hash:
            raise CampaignContractError("approval packet canonical hash mismatch")
    else:
        # Preserve the older Stage-13 schema for its own fixtures. It cannot be
        # substituted into a Stage-14 dual-hash approval.
        if approval["campaign_manifest_sha256"] != canonical_manifest_hash:
            raise CampaignContractError("approval manifest hash mismatch")
        if approval["approval_packet_sha256"] != canonical_packet_hash:
            raise CampaignContractError("approval packet hash mismatch")
    if approval_packet.get("campaign_manifest_sha256") != approval["campaign_manifest_sha256"]:
        raise CampaignContractError("packet does not bind the approved campaign manifest")
    if approval["repository_and_data_hashes"] != manifest["repository_and_data_hashes"]:
        raise CampaignContractError("approval authority hashes mismatch")
    if approval["cost_and_execution_sha256"] != sha256_bytes(canonical_bytes(manifest["cost_and_execution"])):
        raise CampaignContractError("approval cost contract mismatch")
    stage14_schema = "phases_requested" in approval_packet or "ready_lanes" in approval_packet
    if stage14_schema:
        if "phase_permissions_requested" in approval_packet or "candidate_list" in approval_packet:
            raise CampaignContractError("Stage-14 packet alias-field substitution rejected")
        if not isinstance(approval_packet.get("phases_requested"), list) or not isinstance(approval_packet.get("ready_lanes"), list):
            raise CampaignContractError("complete Stage-14 packet schema required")
        requested = {int(value) for value in approval_packet["phases_requested"]}
        packet_hypotheses = set(approval_packet["ready_lanes"])
    else:
        if "phases_requested" in approval_packet or "ready_lanes" in approval_packet:
            raise CampaignContractError("legacy packet alias-field substitution rejected")
        requested = {int(key) for key, value in approval_packet.get("phase_permissions_requested", {}).items() if value}
        packet_hypotheses = set(approval_packet.get("candidate_list", []))
    if requested != set(approval["approved_phases"]):
        raise CampaignContractError("packet and approval phase scopes differ")
    if packet_hypotheses != set(approval["approved_hypotheses"]):
        raise CampaignContractError("packet and approval hypothesis scopes differ")
    if phase not in requested or phase not in approval["approved_phases"] or not hypothesis_id or hypothesis_id not in approval["approved_hypotheses"] or hypothesis_id not in packet_hypotheses:
        raise CampaignContractError("phase or hypothesis outside external approval scope")
    supplemental = approval.get("supplemental_binding_constraints")
    if supplemental:
        if launch_constraints is None:
            raise CampaignContractError("supplemental launch constraints are required")
        funding_rule = supplemental.get("funding_boundary_coverage", {})
        coverage = launch_constraints.get("funding_coverage", {})
        weighted = float(coverage.get("campaign_weighted", -1))
        if not math.isfinite(weighted) or not 0.0 <= weighted <= 1.0:
            raise CampaignContractError("campaign-weighted funding coverage must be finite in [0,1]")
        if weighted < float(funding_rule.get("minimum_campaign_weighted", 1)):
            raise CampaignContractError("campaign-weighted funding coverage is inadequate")
        per_fold = coverage.get("by_hypothesis_fold", {})
        expected_folds = {item["fold_id"] for item in manifest["fold_schedule"]}
        values = [float(per_fold[key]) for key in expected_folds] if set(per_fold) == expected_folds else []
        if (set(per_fold) != expected_folds
                or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
                or any(value < float(funding_rule.get("minimum_per_hypothesis_fold", 1)) for value in values)):
            raise CampaignContractError("per-hypothesis-fold funding coverage is inadequate")
        if coverage.get("missing_boundary_policy") != funding_rule.get("missing_boundary_policy"):
            raise CampaignContractError("funding missing-boundary policy mismatch")
        if coverage.get("selection_use") != funding_rule.get("selection_use"):
            raise CampaignContractError("funding coverage selection-use policy mismatch")
        notification = launch_constraints.get("telegram", {})
        required_notifications = supplemental.get("telegram_notifications", {}).get("required_before_outcome_read", False)
        if required_notifications and not all(notification.get(key) is True for key in (
            "secure_configuration_present", "dry_run_delivered", "heartbeat_delivered", "stop_alert_delivered"
        )):
            raise CampaignContractError("Telegram launch validation is incomplete")
        if notification.get("secret_values_logged_or_archived") is not False:
            raise CampaignContractError("Telegram secret-handling attestation missing")
    parse_utc(approval["authorized_at"])


def initial_state(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest)
    return {"campaign_id": manifest["campaign_id"], "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
            "generation": 0, "completed_nodes": [], "family_stops": {}, "global_stop": None}


def load_or_initialize(state_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = initial_state(manifest)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with (state_path.parent / f".{state_path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if state_path.exists():
            current = json.loads(state_path.read_text())
            if current["manifest_sha256"] != expected["manifest_sha256"]:
                raise CampaignContractError("state manifest hash mismatch")
            return current
        atomic_write_json(state_path, expected)
        return expected


def commit_state(state_path: Path, state: dict[str, Any], expected_generation: int,
                 manifest: dict[str, Any], *, approval: dict[str, Any] | None = None,
                 approval_packet: dict[str, Any] | None = None,
                 approval_raw_bytes: bytes | None = None,
                 expected_approval_sha256: str | None = None,
                 manifest_raw_bytes: bytes | None = None,
                 approval_packet_raw_bytes: bytes | None = None,
                 launch_constraints: dict[str, Any] | None = None) -> None:
    with (state_path.parent / f".{state_path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(state_path.read_text())
        if current["generation"] != expected_generation:
            raise CampaignContractError("stale campaign state generation")
        if state["generation"] != expected_generation + 1:
            raise CampaignContractError("state generation must advance exactly once")
        manifest_hash = sha256_bytes(canonical_bytes(manifest))
        if current["manifest_sha256"] != manifest_hash:
            raise CampaignContractError("persisted state does not match supplied manifest")
        if state["campaign_id"] != current["campaign_id"] or state["manifest_sha256"] != current["manifest_sha256"]:
            raise CampaignContractError("immutable state identity changed")
        known_nodes = {node["node_id"] for node in build_dag(manifest)}
        if not set(state["completed_nodes"]).issubset(known_nodes):
            raise CampaignContractError("state contains unknown completed DAG node")
        if not set(current["completed_nodes"]).issubset(state["completed_nodes"]):
            raise CampaignContractError("completed DAG nodes cannot be removed")
        if current.get("global_stop") and state.get("global_stop") != current["global_stop"]:
            raise CampaignContractError("global stop cannot be cleared or changed")
        for key, value in current.get("family_stops", {}).items():
            if state.get("family_stops", {}).get(key) != value:
                raise CampaignContractError("family stop cannot be cleared or changed")
        added_nodes = set(state["completed_nodes"]) - set(current["completed_nodes"])
        global_changed = state.get("global_stop") != current.get("global_stop")
        added_family_stops = set(state.get("family_stops", {})) - set(current.get("family_stops", {}))
        if sum((bool(added_nodes), global_changed, bool(added_family_stops))) != 1:
            raise CampaignContractError("commit must contain exactly one node transition or one stop transition")
        if added_nodes:
            if len(added_nodes) != 1:
                raise CampaignContractError("only one DAG node may complete per transaction")
            node = {item["node_id"]: item for item in build_dag(manifest)}[next(iter(added_nodes))]
            if not set(node["depends_on"]).issubset(current["completed_nodes"]):
                raise CampaignContractError("committed DAG node dependencies are incomplete")
            enforce_phase_permission(manifest, node["phase"], state=current,
                                     hypothesis_id=node["hypothesis_id"], approval=approval,
                                     approval_packet=approval_packet,
                                     approval_raw_bytes=approval_raw_bytes,
                                     expected_approval_sha256=expected_approval_sha256,
                                     manifest_raw_bytes=manifest_raw_bytes,
                                     approval_packet_raw_bytes=approval_packet_raw_bytes,
                                     launch_constraints=launch_constraints)
        elif global_changed:
            if not state.get("global_stop") or state["global_stop"].get("reason") not in GLOBAL_STOPS:
                raise CampaignContractError("invalid global stop transition")
        else:
            if len(added_family_stops) != 1:
                raise CampaignContractError("only one family stop may be added per transaction")
            key = next(iter(added_family_stops)); stop = state["family_stops"][key]
            if key not in {item["hypothesis_id"] for item in manifest_hypotheses(manifest)} or stop.get("reason") not in FAMILY_STOPS:
                raise CampaignContractError("invalid family stop transition")
        atomic_write_json(state_path, state)


def validate_resource_usage(manifest: dict[str, Any], usage: dict[str, float]) -> None:
    for key, limit in manifest["resource_limits"].items():
        if key in usage and usage[key] > limit:
            raise CampaignContractError(f"resource budget exceeded: {key}")


def heartbeat(campaign_id: str, generation: int, now: str | None = None) -> dict[str, Any]:
    return {"campaign_id": campaign_id, "generation": generation,
            "heartbeat_ts": now or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


def reconcile_artifacts(records: list[dict[str, str]], root: Path) -> None:
    for record in records:
        path = root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise CampaignContractError(f"artifact reconciliation failed: {record['path']}")
