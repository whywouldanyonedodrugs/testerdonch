from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .campaign import CampaignContractError, CampaignOrchestrator, STAGE_GRAPH, _jsonable
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
                locators = tuple(self.population_schedule.iter_locators(
                    parent,
                    phase="outer_evaluation", outer_fold_id=str(control["fold"]), inner_fold_id=None,
                    parent_attempt=parent_row,
                ))
                frames = tuple(self.population_adapter.frame(locator) for locator in locators)
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
    for stage in ("kda02b_adjudication", "outer_evaluation", "conditional_materialization", "conditional_controls"):
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
        for record in spec.get("bound_files", ()):
            self._bound_file(record, repository_root=repository_root)
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
        return spec


def run_shadow_service(spec_path: Path) -> dict[str, Any]:
    spec = ShadowAuthorization(spec_path).require()
    from tools.run_stage22_core_liquid_campaign import TelegramTransport

    run_root = Path(str(spec["run_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "SHADOW_CAMPAIGN_STATE.json"
    if state_path.is_file():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            prior["terminal_inventory"] = verify_terminal_inventory(run_root / "campaign" / "terminal")
            if prior.get("identity_bindings_sha256") != canonical_hash(spec["identity_bindings"]):
                raise ShadowAuthorizationError("completed shadow state identity differs from the service specification")
            return prior
    telegram = TelegramTransport()
    if telegram.preflight() is not True:
        raise ShadowAuthorizationError("secure Telegram preflight failed")
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
    provider = ShadowPayoffProvider(str(spec["synthetic_provider_version"]))
    launch_record = manifest["launch_population_authority"]
    launch_path = Path(str(launch_record["path"]))
    complete_schedule = LaunchPopulationSchedule(launch_path, str(launch_record["sha256"]))
    population_schedule = BoundedShadowPopulationSchedule(complete_schedule, spec["population_slice_policy"])
    population_adapter = LazyProductionFamilyInputAdapter.from_launch_population_authority(
        launch_authority_path=launch_path,
        launch_authority_sha256=str(launch_record["sha256"]),
        repository_root=repository_root,
        construction_mode="shadow_no_outcome",
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
        return state
    if campaign_state.get("status") != "complete":
        raise ShadowAuthorizationError("real CampaignOrchestrator did not reach terminal completion")
    terminal_verification = verify_terminal_inventory(run_root / "campaign" / "terminal")
    stage_reconciliation = json.loads((run_root / "campaign" / "STAGE_JOB_RECONCILIATION.json").read_text(encoding="utf-8"))
    if stage_reconciliation.get("pass") is not True:
        raise ShadowAuthorizationError("real CampaignOrchestrator stage-marker reconciliation failed")
    health_state = json.loads((run_root / "campaign" / "inner_development" / "SUPERVISOR_STATE.json").read_text(encoding="utf-8"))
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
    hold = float(spec.get("hold_after_health_seconds", 0.0))
    if hold > 0:
        time.sleep(hold)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the exact Stage 24 production shadow service")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_shadow_service(args.spec), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ShadowAuthorization", "ShadowAuthorizationError", "_write_shadow_bound_stop", "run_shadow_service"]
