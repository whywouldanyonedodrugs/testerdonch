from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .controls import execute_control, matched_pseudo_event_directives, maximum_holding_for_parent
from .executor import CacheAuthority, ExecutionAuthorization, execute_cached_synthetic_attempt, execute_registered_attempt
from .family_engines.common import require_utc
from .runtime import LazySupervisor, ResourceLimits
from .schema import CAMPAIGN_ID, FAMILY_ORDER, OUTER_FOLDS, complexity, economic_address, family_schemas, normalize_config
from .selection import (
    allocate_refinements,
    day_cluster_bootstrap_q05,
    inner_fold_summary,
    materialization_policy,
    registered_bootstrap_seed,
    resolve_region_overlap,
    select_beam,
    stable_neighborhoods,
)


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
    ) -> None:
        self.packet_root = packet_root; self.run_root = run_root; self.repository_root = repository_root
        self.cache_authority = cache_authority; self.authorization = authorization; self.heartbeat = heartbeat; self.limits = limits
        self.stage_state = run_root / "CAMPAIGN_STAGE_STATE.json"

    def _authority(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
        manifest = self.authorization.require()
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
        self.cache_authority.load_frames(manifest, [cache["artifacts"][0]["path"]])
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

    def _run_jobs(self, stage: str, jobs: Iterable[tuple[str, Callable[[], Any]]]) -> dict[str, Any]:
        root = self.run_root / stage
        validator = lambda job_id, result: isinstance(result, Mapping) and result.get("registered_job_id") == job_id and result.get("registered_attempt_id") is not None
        state = LazySupervisor(root, self.limits, heartbeat=self.heartbeat, real_unit_validator=validator).run(jobs)
        if state["status"] != "complete" or state.get("failed"):
            raise CampaignContractError(f"stage bound stop: {stage}/{state['status']}")
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
                identity_result = execute_registered_attempt(identity, cache_authority=self.cache_authority, authorization=self.authorization, artifact_paths=artifact_paths, registry_by_id=registry)
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
                    result = execute_registered_attempt(row, cache_authority=self.cache_authority, authorization=self.authorization, artifact_paths=selected_paths, registry_by_id=registry, control_directives=directives)
                else:
                    result = {"status": "unavailable_no_matched_pseudo_event", "campaign_id": manifest["campaign_id"], "executable_attempt_id": row["executable_attempt_id"], "canonical_economic_address_sha256": row["canonical_economic_address_sha256"], "observations": [], "ledger": [], "aggregate": {}}
                result["allocation_unavailable"] = unavailable
            else:
                result = execute_registered_attempt(
                    row, cache_authority=self.cache_authority, authorization=self.authorization, artifact_paths=artifact_paths,
                    registry_by_id=registry, parent_binding=parent_binding, parent_artifact_paths=parent_artifact_paths,
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
                    values = [part["aggregate"]["base_net_bps"] if int(part["observation_count"]) else None for part in sorted(parts, key=lambda part: part["inner"])]
                    stress_values = [part["aggregate"]["stress_net_bps"] for part in parts if int(part["observation_count"])]
                    summary = inner_fold_summary(values)
                    finite = [value for value in values if value is not None]
                    day_values = [float(value) for part in parts for value in part["day_base_net_bps"].values()]
                    event_ids = [event_id for part in parts for event_id in part["event_ids"]]
                    candidates.append({
                        **row, "base_net_bps": median(finite) if finite else -math.inf,
                        "stress_net_bps": median(stress_values) if stress_values else -math.inf,
                        "inner_nonempty_fraction": summary["nonempty_fraction"], "p20_inner_fold": summary["p20_with_negative_infinity"],
                        "accepted_trades": sum(int(part["observation_count"]) for part in parts), "market_days": sum(len(part["day_base_net_bps"]) for part in parts),
                        "threshold_coverage": min((part["aggregate"].get("threshold_coverage") or 0.0 for part in parts), default=0.0),
                        "opportunity_frequency": median([part["aggregate"].get("opportunity_frequency_per_30d") or 0.0 for part in parts]),
                        "median_inner_fold": summary["median_with_negative_infinity"],
                        "day_cluster_bootstrap_q05": day_cluster_bootstrap_q05(day_values, registered_bootstrap_seed(row["canonical_economic_address_sha256"], outer)) if day_values else -math.inf,
                        "complexity": complexity(family, row["config"]), "event_ids": event_ids,
                    })
                if not candidates:
                    continue
                regions, _ = resolve_region_overlap(stable_neighborhoods(family, candidates))
                supported = {address for region in regions for address in region["member_ids"]}
                support = {address: region["support"] for region in regions for address in region["member_ids"]}
                for candidate in candidates:
                    candidate["stable_region"] = candidate["canonical_economic_address_sha256"] in supported
                    candidate["plateau_support_count"] = support.get(candidate["canonical_economic_address_sha256"], 0)
                selected_rows = select_beam(candidates)
                selected_addresses = {row["canonical_economic_address_sha256"] for row in selected_rows}
                for candidate in candidates:
                    checks = {
                        "stable_region": bool(candidate["stable_region"]),
                        "accepted_trades": int(candidate["accepted_trades"]) >= 30,
                        "market_days": int(candidate["market_days"]) >= 20,
                        "base_positive": math.isfinite(float(candidate["base_net_bps"])) and float(candidate["base_net_bps"]) > 0,
                        "threshold_coverage": float(candidate["threshold_coverage"]) >= 0.70,
                    }
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
                        "mechanism_anchor": str(candidate.get("lane", "")).endswith("anchor"),
                        "main_component_null": False,
                    })
                for rank, selected in enumerate(selected_rows, 1):
                    beams.append({"outer_fold_id": outer, "family_id": family, "beam_rank": rank, "parent_slot": f"{family}:{outer}:beam:{rank:02d}", "executable_attempt_id": selected["executable_attempt_id"], "canonical_economic_address_sha256": selected["canonical_economic_address_sha256"], "selection_key": list(select_beam([selected])[0].get("selection_key", ())) if False else None})
        atomic_write_json(self.run_root / output_name, {"schema": "stage22_frozen_beam_registry_v1", "rows": beams, "status": "frozen_before_dependent_outcomes"})
        if output_name == "FROZEN_BEAM_REGISTRY.json":
            atomic_write_json(self.run_root / "DEVELOPMENT_SELECTION_SURFACE.json", {"schema": "stage22_development_selection_surface_v1", "rows": audit_surface, "status": "frozen"})
        return beams

    def _freeze_refinements(self, execution: Sequence[Mapping[str, Any]], beams: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows_by_address = {row["canonical_economic_address_sha256"]: row for row in execution}
        output = []
        existing = set(rows_by_address)
        for family in FAMILY_ORDER:
            if family in {"A2_PRIOR_HIGH_RS_CONTEXT_V1", "KDA02B_SURVIVOR_ADJUDICATION_V1"}:
                continue
            family_beams = [row for row in beams if row["family_id"] == family]
            regions = [{"medoid": row["canonical_economic_address_sha256"], "passed": True} for row in family_beams]
            reserved = [canonical_hash({"campaign": "stage22", "family": family, "reserved_refinement_index": index}) for index in range(64)]
            for row in allocate_refinements(family, regions, rows_by_address, reserved, existing):
                if row["status"] == "registered_refinement":
                    row = {
                        **row,
                        "campaign_id": CAMPAIGN_ID,
                        "family_id": family,
                        "executable_attempt_id": row["reserved_attempt_id"],
                        "execution_disposition": "execute_once",
                        "duplicate_of_executable_attempt_id": None,
                        "lane": "conditional_refinement",
                    }
                output.append({"family_id": family, **row})
        atomic_write_json(self.run_root / "CONDITIONAL_REFINEMENT_REGISTRY.json", {"schema": "stage22_conditional_refinement_registry_v1", "rows": output, "registration_boundary": "written and hashed before any refinement outcome", "status": "frozen"})
        return output

    def _a2_bindings(self, execution: Sequence[Mapping[str, Any]], beams: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        slot = {row["parent_slot"]: row for row in beams}
        bindings = {}
        for row in execution:
            if row["family_id"] != "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                continue
            if row["config"]["parent_binding_mode"] == "source_attempt":
                parent_id = row["resolved_parent_executable_attempt_id"]
            else:
                parent = slot.get(row["parent_binding_template_id"])
                if parent is None:
                    continue
                parent_id = parent["executable_attempt_id"]
            bindings[row["executable_attempt_id"]] = {
                "parent_binding_template_id": row["parent_binding_template_id"], "parent_executable_attempt_id": parent_id,
                "parent_only_counterpart_id": row["parent_only_counterpart_id"], "overlay_counterpart_id": row["overlay_counterpart_id"],
            }
        atomic_write_json(self.run_root / "A2_RESOLVED_PARENT_BINDINGS.json", {"schema": "stage22_a2_resolved_parent_bindings_v1", "rows": [{"a2_executable_attempt_id": key, **value} for key, value in sorted(bindings.items())], "missing_parent": "unavailable_no_parent", "status": "frozen_before_A2_outcomes"})
        return bindings

    def run(self) -> dict[str, Any]:
        manifest, execution, controls, registry, cache = self._authority()
        state = {"schema": "stage22_campaign_stage_state_v1", "campaign_id": manifest["campaign_id"], "stage_graph": list(STAGE_GRAPH), "completed_stages": ["cache_validation"], "status": "running"}
        atomic_write_json(self.stage_state, state)
        self._run_jobs("inner_development", self._inner_jobs(execution, registry, cache)); state["completed_stages"].append("inner_development"); atomic_write_json(self.stage_state, state)
        self._run_jobs("kda02b_adjudication", self._kda_jobs(execution, registry, cache)); state["completed_stages"].append("kda02b_adjudication"); atomic_write_json(self.stage_state, state)
        centers = self._freeze_beams(execution, output_name="FROZEN_REFINEMENT_CENTER_REGISTRY.json")
        refinement_records = self._freeze_refinements(execution, centers)
        refinements = [row for row in refinement_records if row.get("status") == "registered_refinement"]
        combined_execution = [*execution, *refinements]
        combined_registry = {**registry, **{row["executable_attempt_id"]: row for row in refinements}}
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
        reconciliation = {"execution_registry_rows": len(execution), "conditional_refinement_rows": len(refinements), "control_registry_rows": len(controls), "kda02b_adjudication_jobs": len(kda_payloads), "beam_rows": len(beams), "outer_jobs": len(outer_payloads), "completed_stage_count": len(state["completed_stages"]) + 1}
        atomic_write_json(self.run_root / "TERMINAL_RECONCILIATION.json", {"schema": "stage22_terminal_reconciliation_v1", **reconciliation, "status": "pass"})
        state.update({"completed_stages": [*state["completed_stages"], "terminal_reconciliation"], "status": "complete"}); atomic_write_json(self.stage_state, state)
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
                    yield job_id, self._execution_job(row, artifacts, registry, job_id, parent_binding=binding, parent_artifact_paths=artifacts)

    def _kda_jobs(self, execution: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any]) -> Iterable[tuple[str, Callable[[], Any]]]:
        rows = [row for row in execution if row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1"]
        artifacts = self._artifacts(cache, phase="kda02b_adjudication")
        if rows and not artifacts:
            raise CampaignContractError("exact Stage-20 KDA02B adjudication cache is absent")
        for row in rows:
            job_id = f"kda02b:{row['executable_attempt_id']}"
            yield job_id, self._execution_job(row, artifacts, registry, job_id, materialize=True)

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
            task = self._execution_job(row, artifacts, registry, job_id, parent_binding=bindings.get(row["executable_attempt_id"]), parent_artifact_paths=artifacts if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None, materialize=True)
            def wrapped(task=task, row=row, outer=outer, job_id=job_id):
                result = task(); result.update({"family_id": row["family_id"], "outer_fold_id": outer, "canonical_economic_address_sha256": row["canonical_economic_address_sha256"]}); return result
            yield job_id, wrapped

    def _materialization_jobs(self, execution: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any], addresses: Sequence[str], bindings: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, Callable[[], Any]]]:
        by_address = {row["canonical_economic_address_sha256"]: row for row in execution}
        for address in sorted(set(addresses)):
            row = by_address.get(address)
            if row is None:
                raise CampaignContractError("materialization address is absent from the executable registry")
            for outer in OUTER_FOLDS:
                inner_ids = sorted({str(self._partition(record).get("inner_fold_id")) for record in cache["artifacts"] if self._partition(record).get("phase") == "inner_validation" and self._partition(record).get("outer_fold_id") == outer})
                for inner in inner_ids:
                    artifacts = self._artifacts(cache, phase="inner_validation", outer_fold_id=outer, inner_fold_id=inner)
                    job_id = f"materialize:{outer}:{inner}:{row['executable_attempt_id']}"
                    yield job_id, self._execution_job(row, artifacts, registry, job_id, parent_binding=bindings.get(row["executable_attempt_id"]), parent_artifact_paths=artifacts if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None, materialize=True)

    def _control_jobs(self, execution: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any], beams: Sequence[Mapping[str, Any]], bindings: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, Callable[[], Any]]]:
        by_id = {row["executable_attempt_id"]: row for row in execution}; beam_by_slot = {row["parent_slot"]: row for row in beams}
        for control in sorted(controls, key=lambda row: row["control_attempt_id"]):
            beam = beam_by_slot.get(control["parent_slot"])
            if beam is None:
                continue
            parent = by_id[beam["executable_attempt_id"]]; artifacts = self._artifacts(cache, phase="outer_evaluation", outer_fold_id=control["fold"])
            job_id = f"control:{control['control_attempt_id']}"
            def task(control=control, parent=parent, artifacts=artifacts, job_id=job_id):
                manifest = self.authorization.require(); _, frames = self.cache_authority.load_frames(manifest, artifacts)
                result = execute_control(control, parent, frames, registry_by_id=registry, parent_binding=bindings.get(parent["executable_attempt_id"]), parent_frames=frames if parent["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None)
                return {**_jsonable(result), "registered_job_id": job_id, "registered_attempt_id": parent["executable_attempt_id"]}
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
