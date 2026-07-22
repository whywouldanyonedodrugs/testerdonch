from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .controls import execute_control, matched_pseudo_event_directives, maximum_holding_for_parent, reconcile_control_duplicates
from .executor import CacheAuthority, ExecutionAuthorization, PayoffProvider, dispatch_registered_attempt, execute_cached_synthetic_attempt, execute_registered_attempt
from .family_engines.common import EngineInputError, require_utc
from .lazy_production_inputs import LazyProductionFamilyInputAdapter
from .population_execution import LaunchPopulationSchedule
from .runtime import LazySupervisor, ResourceLimits
from .schema import CAMPAIGN_ID, FAMILY_ORDER, OUTER_FOLDS, complexity, economic_address, family_schemas, normalize_config
from .selection import (
    beam_order_key,
    aggregate_streaming,
    day_cluster_bootstrap_q05,
    deduplicate_event_overlap,
    inner_fold_summary,
    materialization_policy,
    registered_bootstrap_seed,
    resolve_region_overlap,
    selection_role_eligible,
    select_beam,
    stable_neighborhoods,
)
from .terminal import campaign_terminal_records, terminal_package, verify_terminal_inventory
from .launch_population_authority import validate_launch_population_authority
from .kda02b_population_index import validate_kda02b_lazy_population_index
from .kda02b_lazy_family_input import KDA02BLazyFamilyInputAdapter


STAGE_GRAPH = (
    "cache_validation",
    "inner_development",
    "kda02b_adjudication",
    "plateau_and_refinement",
    "conditional_refinement_execution",
    "parent_beam_freeze",
    "a2_parent_resolution",
    "a2_inner_development",
    "final_beam_freeze",
    "outer_evaluation",
    "conditional_controls",
    "conditional_materialization",
    "terminal_reconciliation",
)


class CampaignContractError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return require_utc(value).isoformat()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"status": "unavailable_empty_fold"} if value == -math.inf else {"status": "invalid_nonfinite"}
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CampaignOrchestrator:
    """The complete frozen Stage-22 stage graph; no caller-selected economics."""

    def __init__(
        self,
        *,
        packet_root: Path,
        run_root: Path,
        repository_root: Path,
        cache_authority: CacheAuthority,
        authorization: ExecutionAuthorization,
        heartbeat: Callable[[Mapping[str, Any]], bool],
        limits: ResourceLimits,
        payoff_provider: PayoffProvider | None = None,
        population_schedule: LaunchPopulationSchedule | None = None,
        population_adapter: LazyProductionFamilyInputAdapter | None = None,
        kda02b_population_adapter: KDA02BLazyFamilyInputAdapter | None = None,
    ) -> None:
        self.packet_root = packet_root; self.run_root = run_root; self.repository_root = repository_root
        self.cache_authority = cache_authority; self.authorization = authorization; self.heartbeat = heartbeat; self.limits = limits
        self.payoff_provider = payoff_provider
        self.population_schedule = population_schedule
        self.population_adapter = population_adapter
        self.kda02b_population_adapter = kda02b_population_adapter
        if (population_schedule is None) != (population_adapter is None):
            raise CampaignContractError("population schedule and FamilyInput adapter must be supplied together")
        self.stage_state = run_root / "CAMPAIGN_STAGE_STATE.json"

    def _authority(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
        manifest = self.authorization.require()
        population_binding = manifest.get("launch_population_authority")
        kda_population_binding = manifest.get("kda02b_lazy_population_authority")
        if not isinstance(population_binding, Mapping) or not isinstance(kda_population_binding, Mapping):
            raise CampaignContractError("complete launch population authorities are absent")
        population_path = Path(str(population_binding.get("path", "")))
        if not population_path.is_absolute():
            population_path = self.packet_root / population_path
        if (
            not population_path.is_file()
            or population_path.stat().st_size != int(population_binding.get("bytes", -1))
            or sha256_file(population_path) != population_binding.get("sha256")
        ):
            raise CampaignContractError("A1-A4 launch population authority bytes differ")
        population = json.loads(population_path.read_text(encoding="utf-8"))
        validate_launch_population_authority(population)
        kda_manifest_path = Path(str(kda_population_binding.get("path", "")))
        if not kda_manifest_path.is_absolute():
            kda_manifest_path = self.packet_root / kda_manifest_path
        if (
            not kda_manifest_path.is_file()
            or kda_manifest_path.stat().st_size != int(kda_population_binding.get("bytes", -1))
            or sha256_file(kda_manifest_path) != kda_population_binding.get("sha256")
        ):
            raise CampaignContractError("KDA02B launch population authority bytes differ")
        validate_kda02b_lazy_population_index(kda_manifest_path.parent)
        expected_cache = manifest.get("primary_hashes", {}).get("cache_authority_manifest") or manifest.get("primary_hashes", {}).get("production_cache_manifest")
        if expected_cache != sha256_file(self.cache_authority.manifest_path):
            raise CampaignContractError("launch cache authority differs from the reviewed campaign manifest")
        paths = {
            "strategy": self.packet_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl",
            "execution": self.packet_root / "FINAL_EXECUTION_REGISTRY.jsonl",
            "controls": self.packet_root / "FINAL_CONTROL_REGISTRY.jsonl",
            "counterparts": self.packet_root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl",
        }
        expected = manifest["primary_hashes"]
        bindings = {"strategy": "strategy_registry", "execution": "execution_registry", "controls": "control_registry", "counterparts": "a2_counterpart_registry"}
        for name, path in paths.items():
            if not path.is_file() or sha256_file(path) != expected[bindings[name]]:
                raise CampaignContractError(f"launch registry drift: {name}")
        execution = _read_jsonl(paths["execution"]); controls = _read_jsonl(paths["controls"])
        registry = {row["executable_attempt_id"]: row for row in execution}
        if len(registry) != len(execution):
            raise CampaignContractError("duplicate executable attempt ID at launch")
        cache = json.loads(self.cache_authority.manifest_path.read_text(encoding="utf-8"))
        self.cache_authority.preload_frames(manifest)
        return manifest, execution, controls, registry, cache

    @staticmethod
    def _partition(record: Mapping[str, Any]) -> Mapping[str, Any]:
        partition = record.get("campaign_partition")
        if not isinstance(partition, Mapping):
            raise CampaignContractError("cache artifact lacks its frozen campaign partition")
        return partition

    def _artifacts(self, cache: Mapping[str, Any], *, phase: str, outer_fold_id: str | None = None, inner_fold_id: str | None = None) -> list[str]:
        selected = []
        for record in cache["artifacts"]:
            partition = self._partition(record)
            if partition.get("phase") != phase:
                continue
            if outer_fold_id is not None and partition.get("outer_fold_id") != outer_fold_id:
                continue
            if inner_fold_id is not None and partition.get("inner_fold_id") != inner_fold_id:
                continue
            selected.append(str(record["path"]))
        return sorted(selected)

    def _run_jobs(self, stage: str, jobs: Iterable[tuple[str, Callable[[], Any]]], *, require_health_release: bool = False) -> dict[str, Any]:
        root = self.run_root / stage
        bindings = {
            "manifest_sha256": sha256_file(self.authorization.manifest_path),
            "approval_request_sha256": sha256_file(self.authorization.approval_request_path),
            "external_approval_sha256": sha256_file(self.authorization.external_approval_path),
            "cache_manifest_sha256": sha256_file(self.cache_authority.manifest_path),
        }
        def validator(job_id: str, result: Any) -> bool:
            if not isinstance(result, Mapping) or result.get("registered_job_id") != job_id or result.get("registered_attempt_id") is None or result.get("status") != "complete" or not isinstance(result.get("aggregate"), Mapping):
                return False
            if result.get("cache_manifest_sha256") != bindings["cache_manifest_sha256"] or result.get("external_approval_sha256") != bindings["external_approval_sha256"]:
                return False
            campaign_state = json.loads(self.stage_state.read_text(encoding="utf-8"))
            if campaign_state.get("campaign_id") != CAMPAIGN_ID or campaign_state.get("status") != "running":
                return False
            self.authorization.require()
            return True
        state = LazySupervisor(root, self.limits, heartbeat=self.heartbeat, real_unit_validator=validator, identity_bindings=bindings).run(jobs, require_health_release=require_health_release)
        if state["status"] != "complete" or state.get("failed"):
            raise CampaignContractError(f"stage bound stop: {stage}/{state['status']}")
        if require_health_release and state.get("health_release") is not True:
            raise CampaignContractError(f"stage completed before required health release: {stage}")
        return state

    def _execution_job(
        self,
        row: Mapping[str, Any],
        artifact_paths: Sequence[str],
        registry: Mapping[str, Mapping[str, Any]],
        job_id: str,
        *,
        parent_binding: Mapping[str, Any] | None = None,
        parent_artifact_paths: Sequence[str] | None = None,
        materialize: bool = False,
    ) -> Callable[[], Any]:
        def task() -> dict[str, Any]:
            if row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1" and row["config"]["adjudication_variant"] == "generic_structure_control":
                identity = next((candidate for candidate in registry.values() if candidate["family_id"] == row["family_id"] and candidate["config"]["stage20_cell_id"] == row["config"]["stage20_cell_id"] and candidate["config"]["adjudication_variant"] == "identity_replay"), None)
                if identity is None:
                    raise CampaignContractError("KDA02B generic control has no exact identity parent")
                identity_result = execute_registered_attempt(identity, cache_authority=self.cache_authority, authorization=self.authorization, artifact_paths=artifact_paths, registry_by_id=registry, payoff_provider=self.payoff_provider)
                manifest = self.authorization.require(); _, frames = self.cache_authority.load_frames(manifest, artifact_paths)
                sides = {ledger["event_id"]: int(ledger["engine_event"]["side"]) for ledger in identity_result["ledger"] if ledger.get("status") == "complete"}
                selected, directives, unavailable = matched_pseudo_event_directives(
                    identity_result["observations"], frames, parent_sides=sides,
                    control_id="KDA02B_GENERIC_STRUCTURE_CONTROL", seed=20260721,
                    control_address=row["canonical_economic_address_sha256"],
                    maximum_holding=maximum_holding_for_parent(row),
                )
                path_by_hash = {frame.content_sha256(): path for frame, path in zip(frames, artifact_paths)}
                selected_paths = [path_by_hash[frame.content_sha256()] for frame in selected]
                if selected_paths:
                    result = execute_registered_attempt(row, cache_authority=self.cache_authority, authorization=self.authorization, artifact_paths=selected_paths, registry_by_id=registry, control_directives=directives, payoff_provider=self.payoff_provider)
                else:
                    result = {"status": "unavailable_no_matched_pseudo_event", "campaign_id": manifest["campaign_id"], "executable_attempt_id": row["executable_attempt_id"], "canonical_economic_address_sha256": row["canonical_economic_address_sha256"], "observations": [], "ledger": [], "aggregate": {}}
                result["allocation_unavailable"] = unavailable
            else:
                result = execute_registered_attempt(
                    row, cache_authority=self.cache_authority, authorization=self.authorization, artifact_paths=artifact_paths,
                    registry_by_id=registry, parent_binding=parent_binding, parent_artifact_paths=parent_artifact_paths,
                    payoff_provider=self.payoff_provider,
                )
            if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and result.get("status") == "complete":
                manifest = self.authorization.require(); _, frames = self.cache_authority.load_frames(manifest, artifact_paths)
                null_control = {
                    "family": row["family_id"], "control_id": "A2_CONTEXT_PERMUTED_MAIN_NULL",
                    "effective_seed": 20260721, "economic_address_sha256": canonical_hash({"development_main_null": row["canonical_economic_address_sha256"]}),
                    "control_attempt_id": canonical_hash({"development_main_null_attempt": row["executable_attempt_id"]}),
                    "execution_status": "execute_once", "resolved_parent_executable_attempt_id": row["executable_attempt_id"],
                }
                result["development_main_null"] = execute_control(
                    null_control, row, frames, registry_by_id=registry, parent_binding=parent_binding,
                    parent_frames=frames if parent_artifact_paths is not None else None,
                    payoff_provider=self.payoff_provider,
                )
            if not materialize:
                observations = list(result.pop("observations", ()))
                result.pop("ledger", None)
                cohort_sums: dict[tuple[str, str], float] = {}
                for observation in observations:
                    cohort = observation.cohort_id or observation.event_id
                    key = (observation.market_day, cohort)
                    cohort_sums[key] = cohort_sums.get(key, 0.0) + float(observation.base_net_bps)
                day_values = {
                    day: sum(value for (item_day, _), value in cohort_sums.items() if item_day == day) / sum(item_day == day for item_day, _ in cohort_sums)
                    for day in sorted({key[0] for key in cohort_sums})
                }
                result.update({
                    "materialization": "aggregate_only",
                    "observation_count": len(observations),
                    "event_ids": sorted(observation.event_id for observation in observations),
                    "day_base_net_bps": day_values,
                })
            else:
                result["materialization"] = "full_registered_ledger"
            return {**_jsonable(result), "registered_job_id": job_id, "registered_attempt_id": row["executable_attempt_id"]}
        return task

    def _population_execution_job(
        self,
        row: Mapping[str, Any],
        registry: Mapping[str, Mapping[str, Any]],
        job_id: str,
        *,
        phase: str,
        outer_fold_id: str,
        inner_fold_id: str | None,
        parent_binding: Mapping[str, Any] | None = None,
        materialize: bool = False,
    ) -> Callable[[], Any]:
        """Stream one registered attempt over its complete lazy PIT partition."""
        if self.population_schedule is None or self.population_adapter is None:
            raise CampaignContractError("complete launch population execution is not configured")
        schedule = self.population_schedule
        adapter = self.population_adapter

        def task() -> dict[str, Any]:
            parent_row = None
            if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and parent_binding is not None and parent_binding.get("status") == "bound":
                parent_row = registry.get(str(parent_binding.get("parent_executable_attempt_id")))
            observations = []
            ledgers: list[dict[str, Any]] = []
            unavailable: list[dict[str, Any]] = []
            paired_parent_ids: set[str] = set()
            paired_overlay_ids: set[str] = set()
            paired_ids: set[str] = set()
            paired_uplift_by_day: dict[str, list[float]] = {}
            paired_parent_attempts: set[str] = set()
            main_null_by_day: dict[str, list[float]] = {}
            main_null_parent_ids: set[str] = set()
            main_null_control_ids: set[str] = set()
            locator_count = 0
            for locator in schedule.iter_locators(
                row,
                phase=phase,
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                parent_attempt=parent_row,
            ):
                locator_count += 1
                try:
                    frame = adapter.frame(locator)
                    parent_frame = None
                    if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                        if parent_row is None:
                            raise CampaignContractError("bound A2 population locator has no exact parent row")
                        parent_frame = adapter.frame(replace(
                            locator,
                            family_id=str(parent_row["family_id"]),
                            executable_attempt_id=str(parent_row["executable_attempt_id"]),
                            canonical_economic_address_sha256=str(parent_row["canonical_economic_address_sha256"]),
                        ))
                    result = dispatch_registered_attempt(
                        row,
                        (frame,),
                        registry_by_id=registry,
                        parent_binding=parent_binding,
                        parent_frames=(parent_frame,) if parent_frame is not None else None,
                        payoff_provider=self.payoff_provider,
                    )
                except EngineInputError as exc:
                    unavailable.append({
                        "status": "unavailable_data",
                        "symbol": locator.symbol,
                        "decision_ts": locator.decision_ts.isoformat(),
                        "reason": str(exc),
                        "locator_identity_sha256": canonical_hash({
                            "family_id": locator.family_id,
                            "phase": locator.phase,
                            "outer_fold_id": locator.outer_fold_id,
                            "inner_fold_id": locator.inner_fold_id,
                            "symbol": locator.symbol,
                            "decision_ts": locator.decision_ts.isoformat(),
                        }),
                    })
                    continue
                selected = list(result.get("observations", ()))
                # A1 reconstructs its state from the complete trailing history.
                # A lazy decision locator owns only episodes completed at this
                # decision; earlier reconstructed episodes belong to earlier
                # locators and must not be duplicated.
                if row["family_id"] == "A1_COMPRESSION_V2":
                    selected = [item for item in selected if require_utc(item.decision_ts) == locator.decision_ts]
                observations.extend(selected)
                if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                    paired = result.get("paired_parent", {})
                    paired_parent_attempts.add(str(paired.get("parent_executable_attempt_id")))
                    paired_parent_ids.update(str(value) for value in paired.get("parent_event_ids", ()))
                    paired_overlay_ids.update(str(value) for value in paired.get("overlay_event_ids", ()))
                    paired_ids.update(str(value) for value in paired.get("paired_event_ids", ()))
                    for day, value in paired.get("paired_uplift_by_utc_day", {}).items():
                        paired_uplift_by_day.setdefault(str(day), []).append(float(value))
                    null_control = {
                        "family": row["family_id"], "control_id": "A2_CONTEXT_PERMUTED_MAIN_NULL",
                        "effective_seed": 20260721,
                        "economic_address_sha256": canonical_hash({"development_main_null": row["canonical_economic_address_sha256"]}),
                        "control_attempt_id": canonical_hash({"development_main_null_attempt": row["executable_attempt_id"]}),
                        "execution_status": "execute_once", "resolved_parent_executable_attempt_id": row["executable_attempt_id"],
                    }
                    null_result = execute_control(
                        null_control, row, (frame,), registry_by_id=registry,
                        parent_binding=parent_binding, parent_frames=(parent_frame,),
                        payoff_provider=self.payoff_provider,
                    )
                    null_paired = null_result.get("paired_control", {})
                    main_null_parent_ids.update(str(value) for value in null_paired.get("parent_event_ids", ()))
                    main_null_control_ids.update(str(value) for value in null_paired.get("control_event_ids", ()))
                    for day, value in null_paired.get("parent_minus_control_by_utc_day", {}).items():
                        main_null_by_day.setdefault(str(day), []).append(float(value))
                if materialize:
                    selected_ids = {item.event_id for item in selected}
                    ledgers.extend(item for item in result.get("ledger", ()) if item.get("event_id") in selected_ids)

            accepted = []
            last_exit: dict[str, datetime] = {}
            for item in sorted(observations, key=lambda value: (value.symbol, value.entry_ts, value.event_id)):
                if item.symbol not in last_exit or item.entry_ts >= last_exit[item.symbol]:
                    accepted.append(item)
                    last_exit[item.symbol] = item.exit_ts
            aggregate = aggregate_streaming(iter(accepted))
            cohort_sums: dict[tuple[str, str], float] = {}
            for observation in accepted:
                key = (observation.market_day, observation.cohort_id or observation.event_id)
                cohort_sums[key] = cohort_sums.get(key, 0.0) + float(observation.base_net_bps)
            day_values = {
                day: sum(value for (item_day, _), value in cohort_sums.items() if item_day == day)
                / sum(item_day == day for item_day, _ in cohort_sums)
                for day in sorted({key[0] for key in cohort_sums})
            }
            payload: dict[str, Any] = {
                "status": "complete",
                "campaign_id": CAMPAIGN_ID,
                "executable_attempt_id": row["executable_attempt_id"],
                "canonical_economic_address_sha256": row["canonical_economic_address_sha256"],
                "aggregate": aggregate,
                "observation_count": len(accepted),
                "event_ids": sorted(item.event_id for item in accepted),
                "day_base_net_bps": day_values,
                "population_locator_count": locator_count,
                "typed_unavailable_observation_count": len(unavailable),
                "typed_unavailable_observations": unavailable,
                "materialization": "full_registered_ledger" if materialize else "aggregate_only",
                "cache_manifest_sha256": sha256_file(self.cache_authority.manifest_path),
                "external_approval_sha256": sha256_file(self.authorization.external_approval_path),
                "registered_job_id": job_id,
                "registered_attempt_id": row["executable_attempt_id"],
            }
            if materialize:
                payload.update({"observations": accepted, "ledger": ledgers})
            if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                payload["paired_parent"] = {
                    "parent_executable_attempt_id": next(iter(paired_parent_attempts)) if len(paired_parent_attempts) == 1 else None,
                    "parent_event_ids": sorted(paired_parent_ids),
                    "overlay_event_ids": sorted(paired_overlay_ids),
                    "paired_event_ids": sorted(paired_ids),
                    "parent_event_identity_match": paired_parent_ids == paired_overlay_ids,
                    "paired_coverage": len(paired_ids) / len(paired_parent_ids) if paired_parent_ids else 0.0,
                    "paired_uplift_by_utc_day": {day: sum(values) / len(values) for day, values in sorted(paired_uplift_by_day.items())},
                }
                payload["development_main_null"] = {"paired_control": {
                    "parent_event_ids": sorted(main_null_parent_ids),
                    "control_event_ids": sorted(main_null_control_ids),
                    "coverage": len(main_null_control_ids) / len(main_null_parent_ids) if main_null_parent_ids else 0.0,
                    "parent_minus_control_by_utc_day": {day: sum(values) / len(values) for day, values in sorted(main_null_by_day.items())},
                }}
            attestation = getattr(self.payoff_provider, "attestation", None)
            if callable(attestation):
                payload["shadow_attestation"] = attestation()
            return _jsonable(payload)
        return task

    def _inner_jobs(self, execution: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any]) -> Iterable[tuple[str, Callable[[], Any]]]:
        non_overlay = [row for row in execution if row["family_id"] not in {"A2_PRIOR_HIGH_RS_CONTEXT_V1", "KDA02B_SURVIVOR_ADJUDICATION_V1"}]
        for outer in OUTER_FOLDS:
            inner_ids = sorted({str(self._partition(record).get("inner_fold_id")) for record in cache["artifacts"] if self._partition(record).get("phase") == "inner_validation" and self._partition(record).get("outer_fold_id") == outer})
            if not inner_ids:
                raise CampaignContractError(f"no inner-fold cache artifacts for {outer}")
            for inner in inner_ids:
                artifacts = self._artifacts(cache, phase="inner_validation", outer_fold_id=outer, inner_fold_id=inner)
                for row in non_overlay:
                    job_id = f"inner:{outer}:{inner}:{row['executable_attempt_id']}"
                    if self.population_schedule is not None:
                        yield job_id, self._population_execution_job(
                            row, registry, job_id, phase="inner_validation",
                            outer_fold_id=outer, inner_fold_id=inner,
                        )
                    else:
                        yield job_id, self._execution_job(row, artifacts, registry, job_id)

    @staticmethod
    def _completed_payloads(stage_root: Path) -> list[dict[str, Any]]:
        payloads = []
        for marker_path in sorted((stage_root / "markers").glob("*.json")):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            artifact = stage_root / marker["artifact"]
            if sha256_file(artifact) != marker["artifact_sha256"]:
                raise CampaignContractError("stage artifact hash mismatch during reconciliation")
            payloads.append(json.loads(artifact.read_text(encoding="utf-8"))["result"])
        return payloads

    def _freeze_beams(self, execution: Sequence[Mapping[str, Any]], *, output_name: str = "FROZEN_BEAM_REGISTRY.json") -> list[dict[str, Any]]:
        rows_by_id = {row["executable_attempt_id"]: row for row in execution}
        payloads = self._completed_payloads(self.run_root / "inner_development")
        a2_root = self.run_root / "a2_inner_development"
        if a2_root.exists():
            payloads.extend(self._completed_payloads(a2_root))
        refinement_root = self.run_root / "conditional_refinement"
        if refinement_root.exists():
            payloads.extend(self._completed_payloads(refinement_root))
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for payload in payloads:
            _, outer, inner, _ = payload["registered_job_id"].split(":", 3)
            grouped.setdefault((payload["registered_attempt_id"], outer), []).append({"inner": inner, **payload})
        corroborated_parents: dict[str, set[str]] = {fold: set() for fold in OUTER_FOLDS}
        for (attempt_id, outer), parts in grouped.items():
            row = rows_by_id.get(attempt_id)
            if row is None or row["family_id"] != "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                continue
            paired = [part.get("paired_parent", {}) for part in parts]
            day_values = [float(value) for item in paired for value in item.get("paired_uplift_by_utc_day", {}).values()]
            coverage = min((float(item.get("paired_coverage", 0.0)) for item in paired), default=0.0)
            identity_match = bool(paired) and all(item.get("parent_event_identity_match") is True for item in paired)
            q05 = day_cluster_bootstrap_q05(day_values, registered_bootstrap_seed(row["canonical_economic_address_sha256"], outer)) if day_values else -math.inf
            if identity_match and coverage >= 0.70 and day_values and sum(day_values) / len(day_values) > 0 and q05 > 0:
                corroborated_parents[outer].update(str(item["parent_executable_attempt_id"]) for item in paired if item.get("parent_executable_attempt_id"))
        beams: list[dict[str, Any]] = []
        audit_surface: list[dict[str, Any]] = []
        for outer in OUTER_FOLDS:
            for family in FAMILY_ORDER:
                if family == "KDA02B_SURVIVOR_ADJUDICATION_V1":
                    continue
                candidates = []
                for attempt_id, row in rows_by_id.items():
                    if row["family_id"] != family or (attempt_id, outer) not in grouped:
                        continue
                    parts = grouped[(attempt_id, outer)]
                    available_parts = [
                        part for part in parts
                        if part.get("status") == "complete"
                        and isinstance(part.get("aggregate"), Mapping)
                        and "observation_count" in part
                    ]
                    values = [
                        part["aggregate"].get("base_net_bps") if int(part.get("observation_count", 0)) else None
                        for part in sorted(parts, key=lambda part: part["inner"])
                    ]
                    stress_values = [part["aggregate"].get("stress_net_bps") for part in available_parts if int(part.get("observation_count", 0)) and part["aggregate"].get("stress_net_bps") is not None]
                    summary = inner_fold_summary(values)
                    finite = [value for value in values if value is not None]
                    day_values = [float(value) for part in available_parts for value in part.get("day_base_net_bps", {}).values()]
                    event_ids = [event_id for part in available_parts for event_id in part.get("event_ids", ())]
                    paired = [part.get("paired_parent", {}) for part in parts]
                    paired_days = [float(value) for item in paired for value in item.get("paired_uplift_by_utc_day", {}).values()]
                    main_nulls = [part.get("development_main_null", {}).get("paired_control", {}) for part in parts]
                    main_null_days = [float(value) for item in main_nulls for value in item.get("parent_minus_control_by_utc_day", {}).values()]
                    main_null_coverage = min((float(item.get("coverage", 0.0)) for item in main_nulls), default=0.0)
                    main_null_q05 = day_cluster_bootstrap_q05(main_null_days, registered_bootstrap_seed(row["canonical_economic_address_sha256"], f"{outer}:A2_MAIN_NULL")) if main_null_days else -math.inf
                    candidate = {
                        **row, "base_net_bps": median(finite) if finite else -math.inf,
                        "stress_net_bps": median(stress_values) if stress_values else -math.inf,
                        "inner_nonempty_fraction": summary["nonempty_fraction"], "p20_inner_fold": summary["p20_with_negative_infinity"],
                        "accepted_trades": sum(int(part.get("observation_count", 0)) for part in available_parts), "market_days": sum(len(part.get("day_base_net_bps", {})) for part in available_parts),
                        "threshold_coverage": min((part["aggregate"].get("threshold_coverage") or 0.0 for part in available_parts), default=0.0),
                        "opportunity_frequency": median([part["aggregate"].get("opportunity_frequency_per_30d") or 0.0 for part in available_parts]) if available_parts else 0.0,
                        "median_inner_fold": summary["median_with_negative_infinity"],
                        "day_cluster_bootstrap_q05": day_cluster_bootstrap_q05(day_values, registered_bootstrap_seed(row["canonical_economic_address_sha256"], outer)) if day_values else -math.inf,
                        "complexity": complexity(family, row["config"]), "event_ids": event_ids,
                        "context_corroborated": attempt_id in corroborated_parents[outer],
                    }
                    if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                        candidate.update({
                            "paired_uplift_mean": sum(paired_days) / len(paired_days) if paired_days else -math.inf,
                            "paired_uplift_bootstrap_q05": day_cluster_bootstrap_q05(paired_days, registered_bootstrap_seed(row["canonical_economic_address_sha256"], outer)) if paired_days else -math.inf,
                            "paired_coverage": min((float(item.get("paired_coverage", 0.0)) for item in paired), default=0.0),
                            "parent_event_identity_match": bool(paired) and all(item.get("parent_event_identity_match") is True for item in paired),
                            "a2_main_null_mean": sum(main_null_days) / len(main_null_days) if main_null_days else -math.inf,
                            "a2_main_null_bootstrap_q05": main_null_q05,
                            "a2_main_null_coverage": main_null_coverage,
                            "a2_main_null_pass": bool(main_null_days) and main_null_coverage >= 0.70 and sum(main_null_days) / len(main_null_days) > 0 and main_null_q05 > 0,
                        })
                    candidates.append(candidate)
                if not candidates:
                    continue
                plateau_candidates = [candidate for candidate in candidates if selection_role_eligible(candidate)]
                regions, _ = resolve_region_overlap(stable_neighborhoods(family, plateau_candidates))
                supported = {address for region in regions for address in region["member_ids"]}
                support = {address: region["support"] for region in regions for address in region["member_ids"]}
                for candidate in candidates:
                    candidate["stable_region"] = candidate["canonical_economic_address_sha256"] in supported
                    candidate["plateau_support_count"] = support.get(candidate["canonical_economic_address_sha256"], 0)
                deduplicated_rows, event_overlap_rejections = deduplicate_event_overlap(select_beam(candidates, width=len(candidates)))
                selected_rows = deduplicated_rows[:5]
                selected_addresses = {row["canonical_economic_address_sha256"] for row in selected_rows}
                for candidate in candidates:
                    checks = {
                        "stable_region": bool(candidate["stable_region"]),
                        "accepted_trades": int(candidate["accepted_trades"]) >= 30,
                        "market_days": int(candidate["market_days"]) >= 20,
                        "base_positive": math.isfinite(float(candidate["base_net_bps"])) and float(candidate["base_net_bps"]) > 0,
                        "threshold_coverage": float(candidate["threshold_coverage"]) >= 0.70,
                        "selection_role": selection_role_eligible(candidate),
                    }
                    if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                        checks.update({
                            "paired_uplift": float(candidate.get("paired_uplift_mean", -math.inf)) > 0 and float(candidate.get("paired_uplift_bootstrap_q05", -math.inf)) > 0 and float(candidate.get("paired_coverage", 0.0)) >= 0.70 and candidate.get("parent_event_identity_match") is True,
                            "main_null": candidate.get("a2_main_null_pass") is True,
                        })
                    failed_checks = [name for name, passed in checks.items() if not passed]
                    within_ten_percent = {
                        "accepted_trades": int(candidate["accepted_trades"]) >= 27,
                        "market_days": int(candidate["market_days"]) >= 18,
                        "threshold_coverage": float(candidate["threshold_coverage"]) >= 0.63,
                    }
                    near_miss = len(failed_checks) == 1 and failed_checks[0] in within_ten_percent and within_ten_percent[failed_checks[0]]
                    audit_surface.append({
                        "family_id": family, "outer_fold_id": outer,
                        "canonical_economic_address_sha256": candidate["canonical_economic_address_sha256"],
                        "beam_survivor": candidate["canonical_economic_address_sha256"] in selected_addresses,
                        "passed": all(checks.values()), "near_miss": near_miss,
                        "near_miss_rule": "one_failed_nonintegrity_gate_within_10pct" if near_miss else None,
                        "failure_class": "pass" if all(checks.values()) else failed_checks[0],
                        "failed_eligibility_gate_count": len(failed_checks),
                        "integrity_valid": True,
                        "mechanism_anchor": candidate.get("selection_role") == "mechanism_anchor_or_ablation",
                        "main_component_null": candidate.get("selection_role") in {"main_component_null", "fixed_adjudication"} or candidate.get("config", {}).get("adjudication_variant") != "identity_replay" and candidate.get("family_id") == "KDA02B_SURVIVOR_ADJUDICATION_V1",
                        "required_by_control_or_forensic": candidate["canonical_economic_address_sha256"] in selected_addresses,
                        "event_overlap_status": next((item for item in event_overlap_rejections if item["canonical_economic_address_sha256"] == candidate["canonical_economic_address_sha256"]), None),
                    })
                for rank, selected in enumerate(selected_rows, 1):
                    beams.append({"outer_fold_id": outer, "family_id": family, "beam_rank": rank, "parent_slot": f"{family}:{outer}:beam:{rank:02d}", "executable_attempt_id": selected["executable_attempt_id"], "canonical_economic_address_sha256": selected["canonical_economic_address_sha256"], "selection_key": list(beam_order_key(selected))})
        atomic_write_json(self.run_root / output_name, {"schema": "stage22_frozen_beam_registry_v1", "rows": beams, "status": "frozen_before_dependent_outcomes"})
        if output_name == "FROZEN_BEAM_REGISTRY.json":
            atomic_write_json(self.run_root / "DEVELOPMENT_SELECTION_SURFACE.json", {"schema": "stage22_development_selection_surface_v1", "rows": audit_surface, "status": "frozen"})
        return beams

    @staticmethod
    def _execution_id_address_inventory(execution: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "registry_index": index,
                "executable_attempt_id": str(row["executable_attempt_id"]),
                "canonical_economic_address_sha256": str(row["canonical_economic_address_sha256"]),
            }
            for index, row in enumerate(execution)
        ]

    @classmethod
    def _bind_frozen_execution(
        cls,
        execution: Sequence[Mapping[str, Any]],
        runtime_rows: Sequence[Mapping[str, Any]],
    ) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
        combined = [*execution, *runtime_rows]
        if cls._execution_id_address_inventory(combined) != cls._execution_id_address_inventory(execution):
            raise CampaignContractError("runtime execution inventory differs from the frozen Stage-21 V5 registry")
        registry = {str(row["executable_attempt_id"]): row for row in combined}
        if len(registry) != len(combined):
            raise CampaignContractError("runtime execution inventory contains a duplicate executable attempt ID")
        return combined, registry

    def _freeze_refinements(self, execution: Sequence[Mapping[str, Any]], _beams: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        inventory = self._execution_id_address_inventory(execution)
        rows: list[dict[str, Any]] = []
        atomic_write_json(self.run_root / "CONDITIONAL_REFINEMENT_REGISTRY.json", {
            "schema": "stage24_stage21_v5_runtime_refinement_closure_v1",
            "campaign_id": CAMPAIGN_ID,
            "rows": rows,
            "row_count": 0,
            "registry_sha256": canonical_hash(rows),
            "adaptive_refinement": "disabled",
            "reason": "Do not perform adaptive refinement. All 6,144 additions are pre-outcome broad Sobol attempts.",
            "source_execution_registry_rows": len(execution),
            "source_executable_attempt_id_count": len({row["executable_attempt_id"] for row in execution}),
            "source_canonical_economic_address_count": len({row["canonical_economic_address_sha256"] for row in execution}),
            "source_execution_registry_id_address_inventory_sha256": canonical_hash(inventory),
            "source_execution_registry_row_inventory_sha256": canonical_hash(list(execution)),
            "registration_boundary": "frozen Stage-21 V5 registry before any shadow or economic outcome",
            "status": "frozen_no_runtime_refinement",
        })
        return rows

    def _a2_bindings(self, execution: Sequence[Mapping[str, Any]], beams: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        slot = {row["parent_slot"]: row for row in beams}
        by_id = {row["executable_attempt_id"]: row for row in execution}
        bindings = {}
        for row in execution:
            if row["family_id"] != "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                continue
            if row["config"]["parent_binding_mode"] == "source_attempt":
                parent_id = row["resolved_parent_executable_attempt_id"]
            else:
                requested_slot = f"{row['config']['parent_family']}:{row['config']['parent_fold_id']}:beam:{int(row['config']['parent_beam_rank']):02d}"
                if canonical_hash({"mode": "beam_slot", "parent_slot": requested_slot}) != row["parent_binding_template_id"]:
                    raise CampaignContractError("A2 parent slot template hash does not match its typed fields")
                parent = slot.get(requested_slot)
                if parent is None:
                    bindings[row["executable_attempt_id"]] = {
                        "status": "unavailable_no_parent",
                        "parent_binding_template_id": row["parent_binding_template_id"],
                        "parent_executable_attempt_id": None,
                        "parent_only_counterpart_id": row["parent_only_counterpart_id"],
                        "overlay_counterpart_id": row["overlay_counterpart_id"],
                    }
                    continue
                parent_id = parent["executable_attempt_id"]
            parent_row = by_id.get(parent_id)
            if parent_row is None or parent_row.get("family_id") != row["config"]["parent_family"]:
                raise CampaignContractError("A2 resolved parent is absent or has the wrong family")
            component_fields = ("proximity_rank", "RS_rank", "reclaim_state", "BTC_ETH_context", "breadth_dispersion")
            binding = {
                "status": "bound",
                "parent_binding_template_id": row["parent_binding_template_id"], "parent_executable_attempt_id": parent_id,
                "parent_only_counterpart_id": row["parent_only_counterpart_id"], "overlay_counterpart_id": row["overlay_counterpart_id"],
                "parent_family": parent_row["family_id"],
                "parent_fold_id": row["config"].get("parent_fold_id"),
                "parent_beam_rank": row["config"].get("parent_beam_rank"),
                "parent_side": parent_row["config"].get("direction"),
                "component_set": {name: row["config"].get(name) for name in component_fields},
                "exposure_action": row["config"].get("overlay_action"),
            }
            binding["binding_sha256"] = canonical_hash({"a2_executable_attempt_id": row["executable_attempt_id"], **binding})
            bindings[row["executable_attempt_id"]] = binding
        atomic_write_json(self.run_root / "A2_RESOLVED_PARENT_BINDINGS.json", {"schema": "stage23_a2_resolved_parent_bindings_v2", "rows": [{"a2_executable_attempt_id": key, **value} for key, value in sorted(bindings.items())], "missing_parent": "unavailable_no_parent", "status": "frozen_before_A2_outcomes", "resolution_sha256": canonical_hash([{"a2_executable_attempt_id": key, **value} for key, value in sorted(bindings.items())])})
        return bindings

    def run(self) -> dict[str, Any]:
        manifest, execution, controls, registry, cache = self._authority()
        state = {"schema": "stage22_campaign_stage_state_v1", "campaign_id": manifest["campaign_id"], "stage_graph": list(STAGE_GRAPH), "completed_stages": ["cache_validation"], "status": "running"}
        atomic_write_json(self.stage_state, state)
        self._run_jobs("inner_development", self._inner_jobs(execution, registry, cache), require_health_release=True); state["completed_stages"].append("inner_development"); atomic_write_json(self.stage_state, state)
        self._run_jobs("kda02b_adjudication", self._kda_jobs(execution, registry, cache)); state["completed_stages"].append("kda02b_adjudication"); atomic_write_json(self.stage_state, state)
        centers = self._freeze_beams(execution, output_name="FROZEN_REFINEMENT_CENTER_REGISTRY.json")
        refinement_records = self._freeze_refinements(execution, centers)
        refinements = [row for row in refinement_records if row.get("status") == "registered_refinement"]
        combined_execution, combined_registry = self._bind_frozen_execution(execution, refinements)
        self._run_jobs("conditional_refinement", self._refinement_jobs(refinements, combined_registry, cache))
        state["completed_stages"].extend(["plateau_and_refinement", "conditional_refinement_execution"]); atomic_write_json(self.stage_state, state)
        parent_beams = self._freeze_beams(combined_execution, output_name="FROZEN_PARENT_BEAM_REGISTRY.json"); state["completed_stages"].append("parent_beam_freeze"); atomic_write_json(self.stage_state, state)
        bindings = self._a2_bindings(combined_execution, parent_beams); state["completed_stages"].append("a2_parent_resolution"); atomic_write_json(self.stage_state, state)
        self._run_jobs("a2_inner_development", self._a2_inner_jobs(combined_execution, combined_registry, cache, bindings)); state["completed_stages"].append("a2_inner_development"); atomic_write_json(self.stage_state, state)
        beams = self._freeze_beams(combined_execution); state["completed_stages"].append("final_beam_freeze"); atomic_write_json(self.stage_state, state)
        # The same exact worker path is used for A2, outer folds and controls;
        # their job generators are determined solely by the frozen beam and
        # parent-slot registries. Empty parents persist unavailable markers.
        selected_ids = {row["executable_attempt_id"] for row in beams}
        if any(row["family_id"] != "KDA02B_SURVIVOR_ADJUDICATION_V1" for row in combined_execution) and not selected_ids:
            raise CampaignContractError("no deterministic beam survived; outer/control work is unavailable")
        # Actual outer/control jobs are intentionally constructed only after
        # the preceding immutable stage files exist, preventing eager work.
        outer_jobs = self._outer_jobs(combined_execution, combined_registry, cache, beams, bindings)
        self._run_jobs("outer_evaluation", outer_jobs); state["completed_stages"].append("outer_evaluation"); atomic_write_json(self.stage_state, state)
        self._run_jobs("conditional_controls", self._control_jobs(combined_execution, controls, combined_registry, cache, beams, bindings)); state["completed_stages"].append("conditional_controls"); atomic_write_json(self.stage_state, state)
        outer_payloads = self._completed_payloads(self.run_root / "outer_evaluation")
        surface = json.loads((self.run_root / "DEVELOPMENT_SELECTION_SURFACE.json").read_text(encoding="utf-8"))["rows"]
        materialized = materialization_policy(surface)
        atomic_write_json(self.run_root / "MATERIALIZATION_REGISTRY.json", {"schema": "stage22_materialization_registry_v1", "addresses": materialized, "status": "frozen"})
        self._run_jobs("conditional_materialization", self._materialization_jobs(combined_execution, combined_registry, cache, materialized, bindings))
        state["completed_stages"].append("conditional_materialization")
        kda_payloads = self._completed_payloads(self.run_root / "kda02b_adjudication")
        control_payloads = self._completed_payloads(self.run_root / "conditional_controls")
        stage_payloads = []
        for stage_name in ("inner_development", "kda02b_adjudication", "conditional_refinement", "a2_inner_development", "outer_evaluation", "conditional_materialization"):
            stage_payloads.extend(self._completed_payloads(self.run_root / stage_name))
        strategy_rows = _read_jsonl(self.packet_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl")
        terminal = campaign_terminal_records(
            strategy_rows=strategy_rows, execution_rows=combined_execution, control_registry=controls,
            stage_payloads=stage_payloads, outer_payloads=outer_payloads, control_payloads=control_payloads,
            fold_order=OUTER_FOLDS,
        )
        expected_jobs = {
            "inner_development": {job for job, _ in self._inner_jobs(execution, registry, cache)},
            "kda02b_adjudication": {job for job, _ in self._kda_jobs(execution, registry, cache)},
            "conditional_refinement": {job for job, _ in self._refinement_jobs(refinements, combined_registry, cache)},
            "a2_inner_development": {job for job, _ in self._a2_inner_jobs(combined_execution, combined_registry, cache, bindings)},
            "outer_evaluation": {job for job, _ in self._outer_jobs(combined_execution, combined_registry, cache, beams, bindings)},
            "conditional_controls": {job for job, _ in self._control_jobs(combined_execution, controls, combined_registry, cache, beams, bindings)},
            "conditional_materialization": {job for job, _ in self._materialization_jobs(combined_execution, combined_registry, cache, materialized, bindings)},
        }
        job_rows = []
        all_workers_stopped = True
        for stage_name, expected in expected_jobs.items():
            stage_root = self.run_root / stage_name
            observed = set()
            for marker_path in sorted((stage_root / "markers").glob("*.json")):
                marker = json.loads(marker_path.read_text(encoding="utf-8")); observed.add(str(marker["job_id"]))
            supervisor = json.loads((stage_root / "SUPERVISOR_STATE.json").read_text(encoding="utf-8"))
            stopped = supervisor.get("all_workers_stopped") is True and not supervisor.get("worker_pids") and int(supervisor.get("in_flight_count", 0)) == 0
            all_workers_stopped = all_workers_stopped and stopped
            job_rows.append({"stage": stage_name, "expected": len(expected), "observed": len(observed), "missing": sorted(expected - observed), "extra": sorted(observed - expected), "all_workers_stopped": stopped, "supervisor_generation": supervisor.get("generation"), "pass": expected == observed and stopped})
        job_reconciliation = {"schema": "stage23_stage_job_reconciliation_v1", "stages": job_rows, "pass": all(row["pass"] for row in job_rows)}
        atomic_write_json(self.run_root / "STAGE_JOB_RECONCILIATION.json", job_reconciliation)
        expected_attempt_ids = [canonical_hash({"registered_registry_index": index, "registered_row": row}) for index, row in enumerate(strategy_rows)]
        expected_control_ids = [str(row["control_attempt_id"]) for row in controls]
        terminal_result = terminal_package(
            self.run_root / "terminal", attempt_ids=expected_attempt_ids, control_ids=expected_control_ids,
            attempt_rows=terminal["attempt_rows"], control_rows=terminal["control_rows"],
            routes=terminal["routes"], forensics=terminal["forensics"], all_workers_stopped=all_workers_stopped,
            job_reconciliation=job_reconciliation,
        )
        terminal_inventory_verification = verify_terminal_inventory(self.run_root / "terminal")
        reconciliation = {
            "registered_configuration_rows": len(strategy_rows), "execution_registry_rows": len(execution),
            "conditional_refinement_rows": len(refinements), "control_registry_rows": len(controls),
            "kda02b_adjudication_jobs": len(kda_payloads), "beam_rows": len(beams),
            "outer_jobs": len(outer_payloads), "completed_stage_count": len(state["completed_stages"]) + 1,
            "terminal_package": terminal_result, "terminal_inventory_verification": terminal_inventory_verification,
            "control_gate_inventory": terminal["control_gate_inventory"],
        }
        atomic_write_json(self.run_root / "TERMINAL_RECONCILIATION.json", {"schema": "stage23_terminal_reconciliation_v2", **reconciliation, "status": "pass"})
        state.update({"completed_stages": [*state["completed_stages"], "terminal_reconciliation"], "status": "complete"}); atomic_write_json(self.stage_state, state)
        inventory_rows = []
        excluded = {"CAMPAIGN_ARTIFACT_INVENTORY.json"}
        for path in sorted(self.run_root.rglob("*")):
            relative = path.relative_to(self.run_root).as_posix()
            if path.is_file() and relative not in excluded:
                inventory_rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        campaign_inventory = {"schema": "stage23_complete_campaign_artifact_inventory_v1", "files": inventory_rows, "excluded_to_avoid_self_reference": sorted(excluded), "inventory_sha256": canonical_hash(inventory_rows)}
        atomic_write_json(self.run_root / "CAMPAIGN_ARTIFACT_INVENTORY.json", campaign_inventory)
        return state

    def _a2_inner_jobs(self, execution: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any], bindings: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, Callable[[], Any]]]:
        overlays = [row for row in execution if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and row["executable_attempt_id"] in bindings]
        for outer in OUTER_FOLDS:
            inner_ids = sorted({str(self._partition(record).get("inner_fold_id")) for record in cache["artifacts"] if self._partition(record).get("phase") == "inner_validation" and self._partition(record).get("outer_fold_id") == outer})
            for inner in inner_ids:
                artifacts = self._artifacts(cache, phase="inner_validation", outer_fold_id=outer, inner_fold_id=inner)
                for row in overlays:
                    binding = bindings[row["executable_attempt_id"]]
                    if row["config"].get("parent_binding_mode") == "beam_slot" and row["config"].get("parent_fold_id") != outer:
                        continue
                    job_id = f"inner:{outer}:{inner}:{row['executable_attempt_id']}"
                    if binding.get("status") != "bound":
                        yield job_id, (lambda row=row, job_id=job_id, binding=binding: {
                            "status": "unavailable_no_parent",
                            "registered_attempt_id": row["executable_attempt_id"],
                            "registered_job_id": job_id,
                            "parent_slot": binding.get("parent_slot"),
                            "aggregate": {}, "observation_count": 0,
                            "day_base_net_bps": {}, "event_ids": [],
                            "materialization": "explicit_empty_unavailable_observation",
                        })
                        continue
                    if self.population_schedule is not None:
                        yield job_id, self._population_execution_job(
                            row, registry, job_id, phase="inner_validation",
                            outer_fold_id=outer, inner_fold_id=inner,
                            parent_binding=binding,
                        )
                    else:
                        yield job_id, self._execution_job(row, artifacts, registry, job_id, parent_binding=binding, parent_artifact_paths=artifacts)

    def _kda_jobs(self, execution: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any]) -> Iterable[tuple[str, Callable[[], Any]]]:
        rows = [row for row in execution if row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1"]
        if self.kda02b_population_adapter is not None:
            for row in rows:
                cell_id = str(row["config"]["stage20_cell_id"])
                job_id = f"kda02b:{row['executable_attempt_id']}"

                def population_task(row=row, cell_id=cell_id, job_id=job_id):
                    observations = []
                    ledger: list[dict[str, Any]] = []
                    unavailable: list[dict[str, Any]] = []
                    eligible_records = 0
                    for record in self.kda02b_population_adapter.stream(cell_id=cell_id):
                        if record.status == "typed_unavailable":
                            unavailable.append({
                                "status": "unavailable_data",
                                "reason": record.unavailable_reason,
                                "event_id": record.event_id,
                                "cell_id": record.cell_id,
                                "model_id": record.model_id,
                                "outer_fold_id": record.outer_fold_id,
                                "symbol": record.symbol,
                                "decision_ts": record.decision_ts.isoformat(),
                                "event_locator_sha256": record.event_locator_sha256,
                            })
                            continue
                        if record.frame is None:
                            raise CampaignContractError("eligible KDA02B population record has no FamilyInput")
                        eligible_records += 1
                        directives = None
                        if row["config"]["adjudication_variant"] == "generic_structure_control":
                            directives = {
                                record.frame.content_sha256(): {
                                    "allocator": "matched_pseudo_event_allocator_v2",
                                    "matched_decision_ts": record.frame.decision_ts,
                                }
                            }
                        dispatched = dispatch_registered_attempt(
                            row, (record.frame,), registry_by_id=registry,
                            control_directives=directives,
                            payoff_provider=self.payoff_provider,
                        )
                        observations.extend(dispatched.get("observations", ()))
                        ledger.extend(dispatched.get("ledger", ()))
                    accepted = []
                    last_exit: dict[str, datetime] = {}
                    for item in sorted(observations, key=lambda value: (value.symbol, value.entry_ts, value.event_id)):
                        if item.symbol not in last_exit or item.entry_ts >= last_exit[item.symbol]:
                            accepted.append(item); last_exit[item.symbol] = item.exit_ts
                    return _jsonable({
                        "status": "complete",
                        "family_id": row["family_id"],
                        "registered_attempt_id": row["executable_attempt_id"],
                        "registered_job_id": job_id,
                        "canonical_economic_address_sha256": row["canonical_economic_address_sha256"],
                        "aggregate": aggregate_streaming(iter(accepted)),
                        "observations": accepted,
                        "ledger": ledger,
                        "observation_count": len(accepted),
                        "event_ids": sorted(item.event_id for item in accepted),
                        "population_eligible_records": eligible_records,
                        "typed_unavailable_observation_count": len(unavailable),
                        "typed_unavailable_observations": unavailable,
                        "materialization": "full_registered_ledger",
                        "cache_manifest_sha256": sha256_file(self.cache_authority.manifest_path),
                        "external_approval_sha256": sha256_file(self.authorization.external_approval_path),
                        "shadow_attestation": self.payoff_provider.attestation() if callable(getattr(self.payoff_provider, "attestation", None)) else None,
                    })

                yield job_id, population_task
            return
        artifacts = self._artifacts(cache, phase="kda02b_adjudication")
        if rows and not artifacts:
            unavailable = [
                item for item in cache.get("typed_unavailable", ())
                if item.get("family_id") == "KDA02B_SURVIVOR_ADJUDICATION_V1"
                and item.get("status") == "unavailable_data"
            ]
            if not unavailable:
                raise CampaignContractError("exact Stage-20 KDA02B adjudication cache is absent without typed authority reason")
            reasons = {(str(item.get("reason")), str(item.get("authority_sha256"))) for item in unavailable}
            if len(reasons) != 1:
                raise CampaignContractError("KDA02B typed cache unavailability is inconsistent")
            reason, authority_sha256 = next(iter(reasons))
            for row in rows:
                job_id = f"kda02b:{row['executable_attempt_id']}"
                yield job_id, (lambda row=row, job_id=job_id, reason=reason, authority_sha256=authority_sha256: {
                    "status": "unavailable_data",
                    "registered_attempt_id": row["executable_attempt_id"],
                    "registered_job_id": job_id,
                    "family_id": row["family_id"],
                    "reason": reason,
                    "authority_sha256": authority_sha256,
                    "aggregate": {},
                    "observation_count": 0,
                    "day_base_net_bps": {},
                    "event_ids": [],
                    "materialization": "explicit_empty_unavailable_observation",
                })
            return
        for row in rows:
            cell_id = str(row["config"]["stage20_cell_id"])
            cell_artifacts = sorted(
                str(record["path"])
                for record in cache["artifacts"]
                if self._partition(record).get("phase") == "kda02b_adjudication"
                and record.get("kda02b_stage20_cell_id") == cell_id
            )
            if len(cell_artifacts) != 9:
                raise CampaignContractError(f"KDA02B exact nine-fold cache binding is incomplete: {cell_id}")
            job_id = f"kda02b:{row['executable_attempt_id']}"
            yield job_id, self._execution_job(row, cell_artifacts, registry, job_id, materialize=True)

    def _refinement_jobs(self, refinements: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any]) -> Iterable[tuple[str, Callable[[], Any]]]:
        for outer in OUTER_FOLDS:
            inner_ids = sorted({str(self._partition(record).get("inner_fold_id")) for record in cache["artifacts"] if self._partition(record).get("phase") == "inner_validation" and self._partition(record).get("outer_fold_id") == outer})
            for inner in inner_ids:
                artifacts = self._artifacts(cache, phase="inner_validation", outer_fold_id=outer, inner_fold_id=inner)
                for row in refinements:
                    job_id = f"inner:{outer}:{inner}:{row['executable_attempt_id']}"
                    yield job_id, self._execution_job(row, artifacts, registry, job_id)

    def _outer_jobs(self, execution: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any], beams: Sequence[Mapping[str, Any]], bindings: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, Callable[[], Any]]]:
        by_id = {row["executable_attempt_id"]: row for row in execution}
        for beam in sorted(beams, key=lambda row: (row["outer_fold_id"], row["family_id"], row["beam_rank"])):
            outer = beam["outer_fold_id"]; row = by_id[beam["executable_attempt_id"]]
            artifacts = self._artifacts(cache, phase="outer_evaluation", outer_fold_id=outer)
            if not artifacts:
                raise CampaignContractError(f"outer cache artifacts are absent: {outer}")
            job_id = f"outer:{outer}:{row['executable_attempt_id']}"
            if self.population_schedule is not None:
                task = self._population_execution_job(
                    row, registry, job_id, phase="outer_evaluation",
                    outer_fold_id=outer, inner_fold_id=None,
                    parent_binding=bindings.get(row["executable_attempt_id"]), materialize=True,
                )
            else:
                task = self._execution_job(row, artifacts, registry, job_id, parent_binding=bindings.get(row["executable_attempt_id"]), parent_artifact_paths=artifacts if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None, materialize=True)
            def wrapped(task=task, row=row, outer=outer, job_id=job_id):
                result = task(); result.update({"family_id": row["family_id"], "outer_fold_id": outer, "canonical_economic_address_sha256": row["canonical_economic_address_sha256"]}); return result
            yield job_id, wrapped

    def _materialization_jobs(self, execution: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any], addresses: Sequence[str], bindings: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, Callable[[], Any]]]:
        by_address = {row["canonical_economic_address_sha256"]: row for row in execution}
        aggregate_source: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        for stage_name in ("inner_development", "conditional_refinement", "a2_inner_development"):
            stage_root = self.run_root / stage_name
            if not stage_root.exists():
                continue
            for payload in self._completed_payloads(stage_root):
                job = str(payload.get("registered_job_id", ""))
                pieces = job.split(":", 3)
                if len(pieces) == 4 and pieces[0] == "inner" and isinstance(payload.get("aggregate"), Mapping):
                    aggregate_source[(pieces[1], pieces[2], str(payload.get("registered_attempt_id")))] = payload
        for address in sorted(set(addresses)):
            row = by_address.get(address)
            if row is None:
                raise CampaignContractError("materialization address is absent from the executable registry")
            for outer in OUTER_FOLDS:
                inner_ids = sorted({str(self._partition(record).get("inner_fold_id")) for record in cache["artifacts"] if self._partition(record).get("phase") == "inner_validation" and self._partition(record).get("outer_fold_id") == outer})
                for inner in inner_ids:
                    artifacts = self._artifacts(cache, phase="inner_validation", outer_fold_id=outer, inner_fold_id=inner)
                    job_id = f"materialize:{outer}:{inner}:{row['executable_attempt_id']}"
                    source = aggregate_source.get((outer, inner, str(row["executable_attempt_id"])))
                    if source is None:
                        raise CampaignContractError("materialization has no exact aggregate-only source")
                    if self.population_schedule is not None:
                        task = self._population_execution_job(
                            row, registry, job_id, phase="inner_validation",
                            outer_fold_id=outer, inner_fold_id=inner,
                            parent_binding=bindings.get(row["executable_attempt_id"]), materialize=True,
                        )
                    else:
                        task = self._execution_job(row, artifacts, registry, job_id, parent_binding=bindings.get(row["executable_attempt_id"]), parent_artifact_paths=artifacts if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None, materialize=True)
                    def verified(task=task, source=source):
                        result = task()
                        left = canonical_hash(source.get("aggregate", {}))
                        right = canonical_hash(result.get("aggregate", {}))
                        if left != right:
                            raise CampaignContractError("aggregate/materialized recomputation mismatch")
                        result.update({"aggregate_materialized_equal": True, "aggregate_only_sha256": left, "materialized_aggregate_sha256": right})
                        return result
                    yield job_id, verified

    def _control_jobs(self, execution: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any], beams: Sequence[Mapping[str, Any]], bindings: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, Callable[[], Any]]]:
        by_id = {row["executable_attempt_id"]: row for row in execution}; beam_by_slot = {row["parent_slot"]: row for row in beams}
        parents_by_slot = {slot: by_id[beam["executable_attempt_id"]] for slot, beam in beam_by_slot.items()}
        resolved_controls = reconcile_control_duplicates(controls, parents_by_slot)
        atomic_write_json(self.run_root / "FINAL_CONTROL_EXECUTION_REGISTRY.json", {"schema": "stage23_final_control_execution_registry_v1", "rows": resolved_controls, "registry_sha256": canonical_hash(resolved_controls), "status": "frozen_before_control_outcomes"})
        for control in resolved_controls:
            beam = beam_by_slot.get(control["parent_slot"])
            if beam is None:
                job_id = f"control:{control['control_attempt_id']}"
                yield job_id, (lambda control=control, job_id=job_id: {
                    "status": "unavailable_no_parent", "control_attempt_id": control["control_attempt_id"],
                    "control_id": control["control_id"], "family_id": control["family"],
                    "outer_fold_id": control["fold"], "effective_seed": control["effective_seed"],
                    "parent_slot": control["parent_slot"], "registered_job_id": job_id,
                    "registered_attempt_id": control["control_attempt_id"],
                })
                continue
            parent = by_id[beam["executable_attempt_id"]]; artifacts = self._artifacts(cache, phase="outer_evaluation", outer_fold_id=control["fold"])
            job_id = f"control:{control['control_attempt_id']}"
            def task(control=control, parent=parent, artifacts=artifacts, job_id=job_id):
                if control["execution_status"] == "unavailable_duplicate_address":
                    return {
                        "status": "unavailable_duplicate_address", "control_attempt_id": control["control_attempt_id"],
                        "duplicate_of_control_attempt_id": control["duplicate_of_control_attempt_id"],
                        "control_id": control["control_id"], "family_id": control["family"],
                        "outer_fold_id": control["fold"], "effective_seed": control["effective_seed"],
                        "parent_slot": control["parent_slot"], "registered_job_id": job_id,
                        "registered_attempt_id": parent["executable_attempt_id"],
                    }
                manifest = self.authorization.require(); _, frames = self.cache_authority.load_frames(manifest, artifacts)
                result = execute_control(control, parent, frames, registry_by_id=registry, parent_binding=bindings.get(parent["executable_attempt_id"]), parent_frames=frames if parent["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None, payoff_provider=self.payoff_provider)
                return {
                    **_jsonable(result), "registered_job_id": job_id,
                    "registered_attempt_id": parent["executable_attempt_id"],
                    "control_attempt_id": control["control_attempt_id"], "control_id": control["control_id"],
                    "family_id": control["family"], "outer_fold_id": control["fold"],
                    "effective_seed": control["effective_seed"], "parent_slot": control["parent_slot"],
                }
            yield job_id, task


def synthetic_production_path_job(
    row: Mapping[str, Any],
    *,
    cache_authority: CacheAuthority,
    campaign_manifest: Mapping[str, Any],
    artifact_paths: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
    job_id: str,
    parent_binding: Mapping[str, Any] | None = None,
    parent_artifact_paths: Sequence[str] | None = None,
) -> Callable[[], Any]:
    def task() -> dict[str, Any]:
        result = execute_cached_synthetic_attempt(
            row,
            cache_authority=cache_authority,
            campaign_manifest=campaign_manifest,
            artifact_paths=artifact_paths,
            registry_by_id=registry,
            parent_binding=parent_binding,
            parent_artifact_paths=parent_artifact_paths,
        )
        return {**_jsonable(result), "registered_job_id": job_id, "registered_attempt_id": row["executable_attempt_id"]}
    return task


__all__ = ["CampaignContractError", "CampaignOrchestrator", "STAGE_GRAPH", "synthetic_production_path_job"]
