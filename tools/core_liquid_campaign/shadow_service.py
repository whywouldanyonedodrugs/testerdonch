from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .campaign import CampaignContractError, CampaignOrchestrator, STAGE_GRAPH, _jsonable, require_population_authority_bindings
from .controls import execute_control, reconcile_control_duplicates
from .executor import CacheAuthority
from .kda02b_lazy_family_input import KDA02BLazyFamilyInputAdapter
from .lazy_production_inputs import LazyProductionFamilyInputAdapter
from .population_execution import LaunchPopulationSchedule
from .runtime import ResourceLimits
from .shadow_campaign import BoundedShadowKDA02BAdapter, BoundedShadowPopulationSchedule, ShadowCampaignAuthorization, ShadowCampaignPacketError
from .shadow_payoff import ShadowPayoffProvider
from .terminal import terminal_package, verify_terminal_inventory


class ShadowAuthorizationError(PermissionError):
    pass


class FinalPacketInterfaceCanaryAuthorization:
    """Exact final-manifest authority that can authorize only a no-outcome interface canary."""

    def __init__(self, spec_path: Path) -> None:
        self.spec_path = spec_path
        self.spec = json.loads(spec_path.read_text(encoding="utf-8"))
        packet = self.spec.get("final_campaign_packet")
        if not isinstance(packet, Mapping):
            raise ShadowAuthorizationError("final packet canary binding is absent")
        self.packet = dict(packet)
        self.repository_root = Path(str(self.spec.get("repository_root", "")))
        self.manifest_path = self._require_record(self.packet.get("manifest"), "manifest")
        self.approval_request_path = self._require_record(self.packet.get("approval_request"), "approval_request")
        self.external_approval_path = self._require_record(self.packet.get("shadow_authorization"), "shadow_authorization")

    @staticmethod
    def _require_record(raw: object, label: str) -> Path:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "bytes", "role", "sha256"}:
            raise ShadowAuthorizationError(f"final packet canary record is malformed: {label}")
        path = Path(str(raw.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != int(raw.get("bytes", -1))
            or sha256_file(path) != raw.get("sha256")
        ):
            raise ShadowAuthorizationError(f"final packet canary bytes differ: {label}")
        return path

    def require(self) -> dict[str, Any]:
        if (
            self.spec.get("schema") != "stage24_final_packet_interface_canary_spec_v1"
            or self.spec.get("mode") != "shadow_no_outcome"
            or self.spec.get("economic_outcomes_authorized") is not False
            or self.spec.get("protected_outcomes_authorized") is not False
            or self.spec.get("capitalcom_payload_access") is not False
            or int(self.spec.get("workers", -1)) != 1
            or not 5 <= int(self.spec.get("heartbeat_seconds", -1)) <= 60
        ):
            raise ShadowAuthorizationError("final packet canary scope is invalid or broadened")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        request = json.loads(self.approval_request_path.read_text(encoding="utf-8"))
        approval = json.loads(self.external_approval_path.read_text(encoding="utf-8"))
        manifest_hash = sha256_file(self.manifest_path)
        request_hash = sha256_file(self.approval_request_path)
        if (
            manifest.get("schema") != "stage24_final_executable_campaign_manifest_v1"
            or manifest.get("authorization_state") != "awaiting_one_exact_external_human_launch_approval"
            or manifest.get("economic_outcomes_opened") is not False
            or manifest.get("capitalcom_payload_opened") is not False
            or int(manifest.get("protected_rows_opened", -1)) != 0
            or request.get("final_campaign_manifest_sha256") != manifest_hash
        ):
            raise ShadowAuthorizationError("regenerated final manifest/request is not a closed pre-approval packet")
        if (
            approval.get("authorization") != "execute_exact_stage24_final_packet_interface_shadow_no_outcome"
            or approval.get("final_campaign_manifest_sha256") != manifest_hash
            or approval.get("final_human_approval_request_sha256") != request_hash
            or approval.get("economic_outcomes_authorized") is not False
            or approval.get("protected_outcomes_authorized") is not False
            or approval.get("capitalcom_payload_access") is not False
        ):
            raise ShadowAuthorizationError("exact final packet no-outcome authorization differs")
        head = subprocess.run(
            ["git", "-C", str(self.repository_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if head != manifest.get("repository", {}).get("implementation_commit"):
            raise ShadowAuthorizationError("final packet canary repository commit differs")
        packet_root = self.manifest_path.parent
        for record in manifest.get("artifact_dependencies", ()):
            path = packet_root / str(record.get("path", ""))
            if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
                raise ShadowAuthorizationError(f"final packet dependency differs: {record.get('path')}")
        inventory_path = self._require_record(self.packet.get("packet_inventory"), "packet_inventory")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        files = inventory.get("files")
        if not isinstance(files, list) or inventory.get("inventory_sha256") != canonical_hash(files):
            raise ShadowAuthorizationError("final packet inventory identity differs")
        for record in files:
            path = packet_root / str(record.get("path", ""))
            if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
                raise ShadowAuthorizationError(f"final packet inventory bytes differ: {record.get('path')}")
        require_population_authority_bindings(manifest, packet_root)
        return manifest


def _notify_bound_stop(telegram: Any, *, service_identity: str, status: str, reason: str, resumable: bool) -> bool:
    sanitized_reason = " ".join(str(reason).split())[:1_000]
    return bool(telegram.bound_stop({
        "service_identity": service_identity,
        "status": status,
        "reason": sanitized_reason,
        "resumable": resumable,
    }))


class ProductionShadowCampaignOrchestrator(CampaignOrchestrator):
    """Real stage graph with controls decoded by the real lazy population adapter."""

    def _control_jobs(self, execution, controls, registry, cache, beams, bindings):  # type: ignore[no-untyped-def]
        if self.population_schedule is None or self.population_adapter is None:
            raise CampaignContractError("shadow controls require the launch-population schedule and FamilyInput adapter")
        by_id = {row["executable_attempt_id"]: row for row in execution}
        beam_by_slot = {row["parent_slot"]: row for row in beams}
        parents_by_slot = {slot: by_id[beam["executable_attempt_id"]] for slot, beam in beam_by_slot.items()}
        resolved_controls = reconcile_control_duplicates(controls, parents_by_slot)
        atomic_write_json(self.run_root / "FINAL_CONTROL_EXECUTION_REGISTRY.json", {
            "schema": "stage24_shadow_population_control_execution_registry_v1",
            "rows": resolved_controls,
            "registry_sha256": canonical_hash(resolved_controls),
            "input_path": "LaunchPopulationSchedule->LazyProductionFamilyInputAdapter",
            "benchmark_probe_frames_used": 0,
            "status": "frozen_before_shadow_control_values",
        })
        for control in resolved_controls:
            beam = beam_by_slot.get(control["parent_slot"])
            job_id = f"control:{control['control_attempt_id']}"
            if beam is None:
                yield job_id, (lambda control=control, job_id=job_id: {
                    "status": "unavailable_no_parent", "control_attempt_id": control["control_attempt_id"],
                    "control_id": control["control_id"], "family_id": control["family"],
                    "outer_fold_id": control["fold"], "effective_seed": control["effective_seed"],
                    "parent_slot": control["parent_slot"], "registered_job_id": job_id,
                    "registered_attempt_id": control["control_attempt_id"],
                    "input_path": "launch_population_unavailable_no_parent",
                    "benchmark_probe_frames_used": 0,
                })
                continue
            parent = by_id[beam["executable_attempt_id"]]

            def task(control=control, parent=parent, job_id=job_id):
                if control["execution_status"] == "unavailable_duplicate_address":
                    return {
                        "status": "unavailable_duplicate_address", "control_attempt_id": control["control_attempt_id"],
                        "duplicate_of_control_attempt_id": control["duplicate_of_control_attempt_id"],
                        "control_id": control["control_id"], "family_id": control["family"],
                        "outer_fold_id": control["fold"], "effective_seed": control["effective_seed"],
                        "parent_slot": control["parent_slot"], "registered_job_id": job_id,
                        "registered_attempt_id": parent["executable_attempt_id"],
                        "input_path": "launch_population_duplicate_reconciliation",
                        "benchmark_probe_frames_used": 0,
                    }
                parent_binding = bindings.get(parent["executable_attempt_id"])
                parent_row = None
                if parent["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and parent_binding is not None:
                    parent_row = registry.get(str(parent_binding.get("parent_executable_attempt_id")))
                schedule_frame = getattr(self.population_schedule, "frame", None)
                locators = []
                frames = []
                for locator in self.population_schedule.iter_locators(
                    parent,
                    phase="outer_evaluation", outer_fold_id=str(control["fold"]), inner_fold_id=None,
                    parent_attempt=parent_row,
                ):
                    locators.append(locator)
                    frames.append(
                        schedule_frame(locator) if callable(schedule_frame) else self.population_adapter.frame(locator)
                    )
                locators = tuple(locators); frames = tuple(frames)
                if not frames:
                    raise CampaignContractError("shadow control population slice is empty")
                parent_frames = None
                if parent["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                    if parent_row is None:
                        raise CampaignContractError("shadow A2 control lacks its exact parent")
                    parent_frames = tuple(self.population_adapter.frame(replace(
                        locator,
                        family_id=str(parent_row["family_id"]),
                        executable_attempt_id=str(parent_row["executable_attempt_id"]),
                        canonical_economic_address_sha256=str(parent_row["canonical_economic_address_sha256"]),
                    )) for locator in locators)
                result = execute_control(
                    control, parent, frames, registry_by_id=registry,
                    parent_binding=parent_binding,
                    parent_frames=parent_frames,
                    payoff_provider=self.payoff_provider,
                )
                if parent["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                    paired = result.get("paired_control", {})
                    if set(paired.get("parent_event_ids", ())) != set(paired.get("control_event_ids", ())):
                        raise CampaignContractError("A2 control event identities differ from its exact parent")
                attestation = getattr(self.payoff_provider, "attestation", None)
                return {
                    **_jsonable(result), "registered_job_id": job_id,
                    "registered_attempt_id": parent["executable_attempt_id"],
                    "control_attempt_id": control["control_attempt_id"], "control_id": control["control_id"],
                    "family_id": control["family"], "outer_fold_id": control["fold"],
                    "effective_seed": control["effective_seed"], "parent_slot": control["parent_slot"],
                    "input_path": "LaunchPopulationSchedule->LazyProductionFamilyInputAdapter",
                    "population_frame_count": len(frames), "benchmark_probe_frames_used": 0,
                    "shadow_attestation": attestation() if callable(attestation) else None,
                }
            yield job_id, task


def _verify_shadow_worker_evidence(campaign_root: Path) -> dict[str, Any]:
    materialized = 0
    provider_versions: set[str] = set()
    for stage in (
        "kda02b_adjudication", "outer_evaluation", "conditional_materialization", "conditional_controls",
        "event_locator_outer_evaluation", "event_locator_conditional_materialization",
        "event_locator_conditional_controls",
    ):
        artifact_root = campaign_root / stage / "artifacts"
        for path in sorted(artifact_root.glob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8")).get("result", {})
            payloads = result.get("batch_results") if isinstance(result.get("batch_results"), list) else (result,)
            for payload in payloads:
                for row in payload.get("ledger", ()):
                    if row.get("status") != "complete":
                        continue
                    if (
                        row.get("shadow_only") is not True
                        or row.get("economic_outcome_opened") is not False
                        or row.get("real_post_entry_rows_opened") != 0
                        or row.get("real_funding_rows_opened") != 0
                        or row.get("actual_accounting_path_executed") is not True
                    ):
                        raise ShadowAuthorizationError("worker materialization crossed or omitted the shadow outcome firewall")
                    provider_versions.add(str(row.get("provider_version")))
                    materialized += 1
    if materialized < 1 or provider_versions != {"stage24_deterministic_synthetic_post_entry_v1"}:
        raise ShadowAuthorizationError("real stage graph has no reconciled ShadowPayoffProvider worker evidence")
    return {
        "materialized_shadow_ledger_rows": materialized,
        "provider_versions": sorted(provider_versions),
        "economic_outcomes_opened": False,
        "real_post_entry_rows_opened": 0,
        "real_funding_rows_opened": 0,
    }


def _write_shadow_bound_stop(
    run_root: Path,
    *,
    generation: int,
    attempt_id: str | None = None,
    attempt_ids: list[str] | None = None,
    control_ids: list[str] | None = None,
    all_workers_stopped: bool,
) -> dict[str, Any]:
    root = run_root / "terminal_bound_stops" / f"generation-{generation:06d}"
    if (root / "TERMINAL_ARTIFACT_INVENTORY.json").is_file():
        return verify_terminal_inventory(root)
    terminal_package(
        root,
        attempt_ids=list(attempt_ids or ([attempt_id] if attempt_id is not None else [])),
        control_ids=list(control_ids or []),
        attempt_rows=[],
        control_rows=[],
        routes=[],
        forensics=[],
        all_workers_stopped=all_workers_stopped,
        bound_stop=True,
        job_reconciliation={"pass": False, "reason": "resumable_shadow_bound_stop"},
    )
    return verify_terminal_inventory(root)


class ShadowAuthorization:
    """Exact Stage-24 no-outcome authorization; never authorizes economic payoff reads."""

    def __init__(self, spec_path: Path) -> None:
        self.spec_path = spec_path

    @staticmethod
    def _bound_file(record: Mapping[str, Any], *, repository_root: Path) -> Path:
        raw = Path(str(record.get("path", "")))
        path = raw if raw.is_absolute() else repository_root / raw
        if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise ShadowAuthorizationError(f"shadow authority file mismatch: {record.get('role')}")
        return path

    @staticmethod
    def _verify_reused_inventory(section: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        files = section.get("files")
        if not isinstance(files, list) or int(section.get("file_count", -1)) != len(files):
            raise ShadowAuthorizationError(f"reused {label} inventory count differs")
        for record in files:
            if not isinstance(record, Mapping):
                raise ShadowAuthorizationError(f"reused {label} inventory record is invalid")
            path = Path(str(record.get("path", "")))
            if (
                not path.is_file()
                or path.stat().st_size != int(record.get("bytes", -1))
                or sha256_file(path) != record.get("sha256")
            ):
                raise ShadowAuthorizationError(f"reused {label} evidence bytes differ")
        if canonical_hash(files) != section.get("inventory_sha256"):
            raise ShadowAuthorizationError(f"reused {label} inventory hash differs")
        return {
            "label": label,
            "file_count": len(files),
            "inventory_sha256": str(section["inventory_sha256"]),
            "verified_paths_sha256": canonical_hash(sorted(str(record["path"]) for record in files)),
        }

    def require(self) -> dict[str, Any]:
        if not self.spec_path.is_file():
            raise ShadowAuthorizationError("shadow service specification is absent")
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        if spec.get("schema") != "stage24_shadow_service_spec_v2" or spec.get("mode") != "shadow_no_outcome":
            raise ShadowAuthorizationError("shadow service mode/schema differs")
        if spec.get("economic_outcomes_authorized") is not False or spec.get("protected_outcomes_authorized") is not False or spec.get("capitalcom_payload_access") is not False:
            raise ShadowAuthorizationError("shadow service specification broadens outcome authority")
        repository_root = Path(str(spec["repository_root"]))
        authority_record = spec.get("stage24_task")
        if not isinstance(authority_record, Mapping):
            raise ShadowAuthorizationError("Stage 24 task binding is absent")
        self._bound_file(authority_record, repository_root=repository_root)
        if authority_record.get("sha256") != "9e546e9376408f97bc3bc2ef2862c06e746864eacbd9a5a70f0e71680eeeccdf":
            raise ShadowAuthorizationError("Stage 24 task hash differs")
        continuation = spec.get("continuation_task")
        if continuation is not None:
            if not isinstance(continuation, Mapping):
                raise ShadowAuthorizationError("Stage 24 continuation task binding is invalid")
            self._bound_file(continuation, repository_root=repository_root)
            if continuation.get("sha256") != "6da86984b890314ea4422c8787d5c6de282342385c6505005358bac31e3493d3":
                raise ShadowAuthorizationError("Stage 24 continuation task hash differs")
        for record in spec.get("bound_files", ()):
            self._bound_file(record, repository_root=repository_root)
        reused = spec.get("reused_evidence_authority")
        reused_verification = None
        if reused is not None:
            if not isinstance(reused, Mapping):
                raise ShadowAuthorizationError("reused evidence authority record is invalid")
            reused_path = self._bound_file(reused, repository_root=repository_root)
            reused_payload = json.loads(reused_path.read_text(encoding="utf-8"))
            if (
                reused_payload.get("schema") != "stage24_reused_shadow_evidence_authority_v1"
                or reused_payload.get("status") != "pass"
                or reused_payload.get("economic_outcomes_opened") is not False
            ):
                raise ShadowAuthorizationError("reused shadow evidence authority is invalid")
            verified_sections = []
            all_paths: list[str] = []
            for label, expected_count in (
                ("structural_inner_development", 3_969),
                ("kda02b_adjudication", 3),
                ("representative_benchmark", 2_401),
            ):
                section = reused_payload.get(label)
                if not isinstance(section, Mapping):
                    raise ShadowAuthorizationError(f"reused {label} evidence is absent")
                verified_sections.append(self._verify_reused_inventory(section, label=label))
                if int(section.get("file_count", -1)) != expected_count:
                    raise ShadowAuthorizationError(f"reused {label} frozen file count differs")
                all_paths.extend(str(record["path"]) for record in section["files"])
            if len(all_paths) != 6_373 or len(set(all_paths)) != 6_373:
                raise ShadowAuthorizationError("reused evidence total or unique file count differs")
            if (
                int(reused_payload["structural_inner_development"].get("marker_count", -1)) != 1_984
                or int(reused_payload["structural_inner_development"].get("artifact_count", -1)) != 1_984
                or int(reused_payload["representative_benchmark"].get("completed_count", -1)) != 1_200
                or reused_payload.get("markers_deleted_rewritten_or_recomputed") is not False
            ):
                raise ShadowAuthorizationError("reused evidence reconciliation differs")
            delta_record = reused_payload.get("commit_delta_impact_audit")
            if not isinstance(delta_record, Mapping):
                raise ShadowAuthorizationError("reused evidence lacks the commit-delta audit binding")
            delta_path = Path(str(delta_record.get("path", "")))
            delta_sha256 = str(delta_record.get("sha256", ""))
            if (
                len(delta_sha256) != 64
                or any(character not in "0123456789abcdef" for character in delta_sha256)
                or delta_sha256 != "f97675dbbf45e267243dd011d13834ecd9b18a721900ca1ed92101b0b1a12e0d"
                or not delta_path.is_file()
                or sha256_file(delta_path) != delta_sha256
            ):
                raise ShadowAuthorizationError("commit-delta audit hash reference differs")
            reused_verification = {
                "schema": "stage24_reused_evidence_startup_verification_v1",
                "launch_generation_scope": "once_before_worker_supervisor_start",
                "reused_evidence_authority_sha256": str(reused["sha256"]),
                "sections": verified_sections,
                "total_file_count": len(all_paths),
                "unique_file_count": len(set(all_paths)),
                "all_paths_sha256": canonical_hash(sorted(all_paths)),
                "commit_delta_impact_audit_sha256": delta_sha256,
                "status": "pass",
            }
            reused_verification["verification_sha256"] = canonical_hash(reused_verification)
        actual_commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        reviewed_commit = str(spec.get("reviewed_commit", ""))
        if actual_commit != reviewed_commit:
            raise ShadowAuthorizationError("live worktree is not the exact reviewed commit")
        packet = spec.get("shadow_campaign_packet")
        if not isinstance(packet, Mapping):
            raise ShadowAuthorizationError("bounded shadow campaign packet binding is absent")
        for name, record in packet.items():
            if name == "packet_root":
                continue
            if not isinstance(record, Mapping):
                raise ShadowAuthorizationError(f"shadow campaign packet record is invalid: {name}")
            self._bound_file(record, repository_root=repository_root)
        if spec.get("cache_classification") != "benchmark_probe_only" or spec.get("cache_is_launch_input_authority") is not False:
            raise ShadowAuthorizationError("shadow service cache is not isolated as a benchmark-only probe")
        if spec.get("payoff_provider") != "ShadowPayoffProvider":
            raise ShadowAuthorizationError("shadow payoff provider binding differs")
        try:
            BoundedShadowPopulationSchedule(object(), spec.get("population_slice_policy", {}))
        except ShadowCampaignPacketError as exc:
            raise ShadowAuthorizationError(str(exc)) from exc
        if spec.get("identity_bindings", {}).get("population_slice_policy_sha256") != canonical_hash(spec["population_slice_policy"]):
            raise ShadowAuthorizationError("shadow population slice policy binding differs")
        try:
            BoundedShadowKDA02BAdapter(object(), spec.get("kda02b_slice_policy", {}))
        except ShadowCampaignPacketError as exc:
            raise ShadowAuthorizationError(str(exc)) from exc
        if spec.get("identity_bindings", {}).get("kda02b_slice_policy_sha256") != canonical_hash(spec["kda02b_slice_policy"]):
            raise ShadowAuthorizationError("shadow KDA02B slice policy binding differs")
        if not 1 <= int(spec.get("workers", 0)) <= 4:
            raise ShadowAuthorizationError("shadow worker bound differs")
        if reused_verification is not None:
            spec["_startup_reused_evidence_verification"] = reused_verification
        return spec


def _run_shadow_service(spec: Mapping[str, Any], telegram: Any) -> dict[str, Any]:
    run_root = Path(str(spec["run_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    startup_verification = spec.get("_startup_reused_evidence_verification")
    if isinstance(startup_verification, Mapping):
        atomic_write_json(run_root / "REUSED_EVIDENCE_STARTUP_VERIFICATION.json", startup_verification)
    state_path = run_root / "SHADOW_CAMPAIGN_STATE.json"
    if state_path.is_file():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            prior["terminal_inventory"] = verify_terminal_inventory(run_root / "campaign" / "terminal")
            if prior.get("identity_bindings_sha256") != canonical_hash(spec["identity_bindings"]):
                raise ShadowAuthorizationError("completed shadow state identity differs from the service specification")
            return prior
    state = {
        "schema": "stage24_shadow_campaign_state_v2",
        "status": "running",
        "service_identity": spec["service_identity"],
        "identity_bindings_sha256": canonical_hash(spec["identity_bindings"]),
        "stage_graph": list(STAGE_GRAPH),
        "cache_classification": "benchmark_probe_only",
        "cache_is_launch_input_authority": False,
        "payoff_provider": "ShadowPayoffProvider",
        "economic_outcomes_opened": False,
        "protected_outcomes_opened": False,
        "capitalcom_payload_opened": False,
        "telegram_preflight": "pass",
    }
    atomic_write_json(state_path, state)
    telegram.launch({
        "service_identity": spec["service_identity"],
        "status": "active",
        "run_root": str(run_root),
        "workers": int(spec["workers"]),
    })
    repository_root = Path(str(spec["repository_root"]))
    try:
        authorization = ShadowCampaignAuthorization(spec, repository_root)
        manifest = authorization.require()
    except ShadowCampaignPacketError as exc:
        raise ShadowAuthorizationError(str(exc)) from exc
    packet_root = Path(str(spec["shadow_campaign_packet"]["packet_root"]))
    cache_path = Path(str(spec["shadow_campaign_packet"]["cache_manifest"]["path"]))
    cache = CacheAuthority(cache_path, cache_path.parent)
    limits = ResourceLimits(
        max_workers=int(spec["workers"]),
        max_jobs_in_flight=int(spec["workers"]),
        max_rss_bytes=10 * 1024**3,
        max_output_bytes=24 * 1024**3,
        minimum_free_disk_bytes=8 * 1024**3,
        heartbeat_seconds=int(spec.get("heartbeat_seconds", 1800)),
        graceful_stop_seconds=300,
        wall_time_seconds=None,
    )
    scenario_record = spec["shadow_campaign_packet"].get("synthetic_scenario_matrix")
    scenario_payload = (
        json.loads(Path(str(scenario_record["path"])).read_text(encoding="utf-8"))
        if isinstance(scenario_record, Mapping) else None
    )
    scenario_rows = [] if scenario_payload is None else scenario_payload.get("assignments", ())
    provider = ShadowPayoffProvider(
        str(spec["synthetic_provider_version"]),
        scenario_by_attempt={str(row["executable_attempt_id"]): str(row["scenario"]) for row in scenario_rows},
        concentration_symbol_by_attempt={
            str(row["executable_attempt_id"]): str(row["concentration_symbol"])
            for row in scenario_rows if row.get("concentration_symbol")
        },
    )
    launch_record = manifest["launch_population_authority"]
    launch_path = Path(str(launch_record["path"]))
    complete_schedule = LaunchPopulationSchedule(launch_path, str(launch_record["sha256"]))
    population_adapter = LazyProductionFamilyInputAdapter.from_launch_population_authority(
        launch_authority_path=launch_path,
        launch_authority_sha256=str(launch_record["sha256"]),
        repository_root=repository_root,
        construction_mode="shadow_no_outcome",
    )
    population_schedule = BoundedShadowPopulationSchedule(
        complete_schedule, spec["population_slice_policy"], population_adapter=population_adapter,
    )
    kda_record = manifest["kda02b_lazy_population_authority"]
    kda_path = Path(str(kda_record["path"]))
    complete_kda_adapter = KDA02BLazyFamilyInputAdapter(
        index_root=kda_path.parent,
        authority_path=Path(str(spec["shadow_campaign_packet"]["execution_input_authority"]["path"])),
        repository_root=repository_root,
        mode="shadow_no_outcome",
    )
    kda_adapter = BoundedShadowKDA02BAdapter(complete_kda_adapter, spec["kda02b_slice_policy"])
    orchestrator = ProductionShadowCampaignOrchestrator(
        packet_root=packet_root,
        run_root=run_root / "campaign",
        repository_root=repository_root,
        cache_authority=cache,
        authorization=authorization,
        heartbeat=telegram.heartbeat,
        limits=limits,
        payoff_provider=provider,
        population_schedule=population_schedule,
        population_adapter=population_adapter,
        kda02b_population_adapter=kda_adapter,
    )
    if spec["population_slice_policy"].get("schema") == "stage24_shadow_event_locator_policy_v2":
        reused_record = spec.get("reused_evidence_authority")
        if not isinstance(reused_record, Mapping):
            raise ShadowAuthorizationError("event-locator resume lacks its reused evidence authority")
        reused_payload = json.loads(Path(str(reused_record["path"])).read_text(encoding="utf-8"))
        orchestrator.reuse_structural_shadow = True
        orchestrator.structural_evidence_run_root = Path(str(reused_payload["prior_campaign_run_root"]))
        orchestrator.inner_development_stage_name = "event_locator_inner_development"
        orchestrator.conditional_refinement_stage_name = "event_locator_conditional_refinement"
        orchestrator.a2_inner_development_stage_name = "event_locator_a2_inner_development"
        orchestrator.outer_evaluation_stage_name = "event_locator_outer_evaluation"
        orchestrator.conditional_controls_stage_name = "event_locator_conditional_controls"
        orchestrator.conditional_materialization_stage_name = "event_locator_conditional_materialization"
    try:
        campaign_state = orchestrator.run()
    except CampaignContractError as exc:
        campaign_state_path = run_root / "campaign" / "CAMPAIGN_STAGE_STATE.json"
        campaign_state = json.loads(campaign_state_path.read_text(encoding="utf-8")) if campaign_state_path.is_file() else {}
        strategy = [json.loads(line) for line in (packet_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines() if line]
        controls = [json.loads(line) for line in (packet_root / "FINAL_CONTROL_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines() if line]
        bound_terminal = _write_shadow_bound_stop(
            run_root,
            generation=int(campaign_state.get("generation", 0)),
            attempt_ids=[canonical_hash({"registered_registry_index": index, "registered_row": row}) for index, row in enumerate(strategy)],
            control_ids=[str(row["control_attempt_id"]) for row in controls],
            all_workers_stopped=True,
        )
        state.update({
            "status": "global_resumable_bound_stop",
            "reason": str(exc),
            "health_release": False,
            "all_workers_stopped": True,
            "resumable": True,
            "terminal_bound_stop_inventory": bound_terminal,
        })
        atomic_write_json(state_path, state)
        _notify_bound_stop(
            telegram,
            service_identity=str(spec["service_identity"]),
            status=str(state["status"]),
            reason=str(state["reason"]),
            resumable=True,
        )
        return state
    if campaign_state.get("status") != "complete":
        raise ShadowAuthorizationError("real CampaignOrchestrator did not reach terminal completion")
    terminal_verification = verify_terminal_inventory(run_root / "campaign" / "terminal")
    stage_reconciliation = json.loads((run_root / "campaign" / "STAGE_JOB_RECONCILIATION.json").read_text(encoding="utf-8"))
    if stage_reconciliation.get("pass") is not True:
        raise ShadowAuthorizationError("real CampaignOrchestrator stage-marker reconciliation failed")
    health_stage = (
        "event_locator_inner_development"
        if spec["population_slice_policy"].get("schema") == "stage24_shadow_event_locator_policy_v2"
        else "inner_development"
    )
    health_state = json.loads((run_root / "campaign" / health_stage / "SUPERVISOR_STATE.json").read_text(encoding="utf-8"))
    if health_state.get("first_real_unit_reconciled") is not True or health_state.get("health_release") is not True or int(health_state.get("heartbeat_success_count", 0)) < 1:
        raise ShadowAuthorizationError("real CampaignOrchestrator health release lacks a reconciled unit or scheduled heartbeat")
    worker_evidence = _verify_shadow_worker_evidence(run_root / "campaign")
    state.update({
        "status": "complete",
        "campaign_orchestrator": "actual_production_stage_graph",
        "completed_stages": campaign_state.get("completed_stages"),
        "first_real_registered_unit_reconciled": True,
        "scheduled_heartbeat_delivered": int(health_state.get("heartbeat_success_count", 0)) >= 1,
        "health_release": True,
        "stage_job_reconciliation_sha256": sha256_file(run_root / "campaign" / "STAGE_JOB_RECONCILIATION.json"),
        "terminal_inventory": terminal_verification,
        "shadow_payoff_provider_attestation": provider.attestation(),
        "shadow_worker_evidence": worker_evidence,
    })
    atomic_write_json(state_path, state)
    telegram.complete({
        "service_identity": spec["service_identity"],
        "status": state["status"],
        "run_root": str(run_root),
        "health_release": state["health_release"],
    })
    hold = float(spec.get("hold_after_health_seconds", 0.0))
    if hold > 0:
        time.sleep(hold)
    return state


def run_shadow_service(spec_path: Path) -> dict[str, Any]:
    """Run the service and notify on every post-preflight exceptional stop."""
    spec = ShadowAuthorization(spec_path).require()
    from tools.run_stage22_core_liquid_campaign import TelegramTransport
    telegram = TelegramTransport()
    if telegram.preflight() is not True:
        raise ShadowAuthorizationError("secure Telegram preflight failed")
    try:
        return _run_shadow_service(spec, telegram)
    except Exception as exc:
        try:
            run_root = Path(str(spec["run_root"]))
            service_identity = str(spec["service_identity"])
            state_path = run_root / "SHADOW_CAMPAIGN_STATE.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
            state.update({
                "status": "pre_outcome_launch_failure",
                "reason": str(exc),
                "health_release": False,
                "resumable": False,
                "economic_outcomes_opened": False,
            })
            run_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(state_path, state)
            _notify_bound_stop(
                telegram,
                service_identity=service_identity,
                status=str(state["status"]),
                reason=str(state["reason"]),
                resumable=False,
            )
        except Exception:
            # Preserve the owning exception; a transport failure must not mask
            # the launch/resource/authorization failure that stopped work.
            pass
        raise


def run_final_packet_interface_canary(spec_path: Path) -> dict[str, Any]:
    """Exercise one registered unit through the final manifest without opening economic outcomes."""
    from tools.run_stage22_core_liquid_campaign import TelegramTransport

    authorization = FinalPacketInterfaceCanaryAuthorization(spec_path)
    spec = authorization.spec
    manifest = authorization.require()
    telegram = TelegramTransport()
    if telegram.preflight() is not True:
        raise ShadowAuthorizationError("secure Telegram preflight failed")
    run_root = Path(str(spec["run_root"])); run_root.mkdir(parents=True, exist_ok=True)
    service_identity = str(spec["service_identity"])
    telegram.launch({"service_identity": service_identity, "status": "shadow_no_outcome", "run_root": str(run_root), "workers": 1})
    try:
        packet_root = authorization.manifest_path.parent
        resolved = require_population_authority_bindings(manifest, packet_root)
        launch_record, launch_path = resolved["launch_population_authority"]
        _kda_record, kda_path = resolved["kda02b_lazy_population_authority"]
        cache_path = FinalPacketInterfaceCanaryAuthorization._require_record(
            authorization.packet.get("cache_manifest"), "cache_manifest",
        )
        execution_input_path = FinalPacketInterfaceCanaryAuthorization._require_record(
            authorization.packet.get("execution_input_authority"), "execution_input_authority",
        )
        cache = CacheAuthority(cache_path, cache_path.parent)
        complete_schedule = LaunchPopulationSchedule(launch_path, str(launch_record["sha256"]))
        population_adapter = LazyProductionFamilyInputAdapter.from_launch_population_authority(
            launch_authority_path=launch_path,
            launch_authority_sha256=str(launch_record["sha256"]),
            repository_root=authorization.repository_root,
            construction_mode="shadow_no_outcome",
        )
        schedule = BoundedShadowPopulationSchedule(
            complete_schedule,
            {
                "schema": "stage24_shadow_population_slice_policy_v1",
                "selection": "first_authority_order_locator_on_each_distinct_UTC_day",
                "maximum_distinct_days_per_attempt_partition": 1,
                "economic_values_used_for_selection": False,
                "benchmark_frame_values_used": False,
                "full_launch_population_authority_preserved": True,
            },
            population_adapter=population_adapter,
        )
        kda_adapter = KDA02BLazyFamilyInputAdapter(
            index_root=kda_path.parent,
            authority_path=execution_input_path,
            repository_root=authorization.repository_root,
            mode="shadow_no_outcome",
        )
        limits = ResourceLimits(
            max_workers=1, max_jobs_in_flight=1, max_rss_bytes=10 * 1024**3,
            max_output_bytes=256 * 1024**2, minimum_free_disk_bytes=8 * 1024**3,
            heartbeat_seconds=int(spec["heartbeat_seconds"]),
            graceful_stop_seconds=300, wall_time_seconds=None,
        )
        orchestrator = CampaignOrchestrator(
            packet_root=packet_root, run_root=run_root / "campaign",
            repository_root=authorization.repository_root, cache_authority=cache,
            authorization=authorization, heartbeat=telegram.heartbeat, limits=limits,
            payoff_provider=ShadowPayoffProvider("stage24-final-packet-interface-canary-v1"),
            population_schedule=schedule, population_adapter=population_adapter,
            kda02b_population_adapter=kda_adapter,
        )
        parsed, execution, _controls, registry, cache_payload = orchestrator._authority()
        candidate = next((row for row in execution if row.get("family_id") == "A4_TSMOM_V7"), None)
        partition = next((
            record.get("campaign_partition", {}) for record in cache_payload.get("artifacts", ())
            if record.get("campaign_partition", {}).get("phase") == "inner_validation"
        ), None)
        if candidate is None or not isinstance(partition, Mapping):
            raise ShadowAuthorizationError("final packet canary has no production-shaped registered unit")
        job_id = "final-packet-interface-canary:" + str(candidate["executable_attempt_id"])
        atomic_write_json(orchestrator.stage_state, {
            "schema": "stage22_campaign_stage_state_v1", "campaign_id": parsed["campaign_id"],
            "stage_graph": list(STAGE_GRAPH), "completed_stages": ["cache_validation"], "status": "running",
        })
        task = orchestrator._population_execution_job(
            candidate, registry, job_id, phase="inner_validation",
            outer_fold_id=str(partition["outer_fold_id"]), inner_fold_id=str(partition["inner_fold_id"]),
        )
        supervisor = orchestrator._run_jobs("final_packet_interface_canary", iter(((job_id, task),)), require_health_release=True)
        if (
            supervisor.get("status") != "complete"
            or int(supervisor.get("completed_count", 0)) != 1
            or supervisor.get("first_real_unit_reconciled") is not True
            or supervisor.get("health_release") is not True
            or int(supervisor.get("heartbeat_success_count", 0)) < 1
            or supervisor.get("worker_pids")
            or supervisor.get("all_workers_stopped") is not True
        ):
            raise ShadowAuthorizationError("final packet interface canary did not reconcile one healthy unit")
        result = {
            "schema": "stage24_final_packet_interface_canary_result_v1",
            "status": "complete", "mode": "shadow_no_outcome",
            "service_identity": service_identity, "manifest_sha256": sha256_file(authorization.manifest_path),
            "registered_production_shaped_units": 1, "first_real_unit_reconciled": True,
            "health_release": True, "scheduled_heartbeat_delivered": True,
            "both_population_authorities_constructed": True, "all_workers_stopped": True,
            "worker_pids": [], "economic_outcomes_opened": False,
            "protected_outcomes_opened": False, "capitalcom_payload_opened": False,
        }
        atomic_write_json(run_root / "FINAL_PACKET_INTERFACE_CANARY_RESULT.json", result)
        telegram.complete({"service_identity": service_identity, "status": "complete", "run_root": str(run_root), "health_release": True})
        return result
    except Exception as exc:
        _notify_bound_stop(telegram, service_identity=service_identity, status="pre_outcome_canary_bound_stop", reason=str(exc), resumable=False)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the exact Stage 24 production shadow service")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_shadow_service(args.spec), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ShadowAuthorization", "ShadowAuthorizationError", "_write_shadow_bound_stop", "run_shadow_service"]
