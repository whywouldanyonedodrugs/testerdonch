#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from tools.core_liquid_campaign.campaign import CampaignOrchestrator
from tools.core_liquid_campaign.canonical import atomic_write_json, canonical_hash, sha256_file
from tools.core_liquid_campaign.kda02b_lazy_family_input import KDA02BLazyFamilyInputAdapter
from tools.core_liquid_campaign.runtime import LazySupervisor, ResourceLimits
from tools.core_liquid_campaign.shadow_campaign import BoundedShadowKDA02BAdapter
from tools.core_liquid_campaign.shadow_payoff import ShadowPayoffProvider


CELL_ID = "KDA02B_009"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _task(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]):
    kda_record = spec["shadow_campaign_packet"]["kda02b_population_authority"]
    complete = KDA02BLazyFamilyInputAdapter(
        index_root=Path(str(kda_record["path"])).parent,
        authority_path=Path(str(spec["shadow_campaign_packet"]["execution_input_authority"]["path"])),
        repository_root=Path(str(spec["repository_root"])),
        mode="shadow_no_outcome",
    )
    bounded = BoundedShadowKDA02BAdapter(complete, spec["kda02b_slice_policy"])
    orchestrator = CampaignOrchestrator.__new__(CampaignOrchestrator)
    orchestrator.kda02b_population_adapter = bounded
    orchestrator.cache_authority = SimpleNamespace(
        manifest_path=Path(str(spec["shadow_campaign_packet"]["cache_manifest"]["path"]))
    )
    orchestrator.authorization = SimpleNamespace(
        external_approval_path=Path(str(spec["shadow_campaign_packet"]["external_authorization"]["path"]))
    )
    orchestrator.payoff_provider = ShadowPayoffProvider("stage24-kda02b-all-variants-invariance")
    registry = {str(row["executable_attempt_id"]): row for row in rows}
    return next(iter(orchestrator._kda_jobs(tuple(rows), registry, {})))


def _artifact(root: Path) -> tuple[dict[str, Any], str]:
    markers = sorted((root / "markers").glob("*.json"))
    if len(markers) != 1:
        raise RuntimeError("KDA02B invariance supervisor did not reconcile exactly one batch marker")
    marker = json.loads(markers[0].read_text(encoding="utf-8"))
    path = root / str(marker["artifact"])
    if sha256_file(path) != marker["artifact_sha256"]:
        raise RuntimeError("KDA02B invariance artifact hash differs")
    return json.loads(path.read_text(encoding="utf-8"))["result"], marker["artifact_sha256"]


def _run(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], root: Path, workers: int, *, stop: bool = False):
    job_id, task = _task(spec, rows)
    limits = ResourceLimits(
        max_workers=workers,
        max_jobs_in_flight=workers,
        max_rss_bytes=10 * 1024**3,
        max_output_bytes=2 * 1024**3,
        minimum_free_disk_bytes=8 * 1024**3,
        heartbeat_seconds=1800,
        graceful_stop_seconds=300,
        wall_time_seconds=None,
    )
    supervisor = LazySupervisor(
        root,
        limits,
        heartbeat=lambda _payload: True,
        real_unit_validator=lambda candidate, result: candidate == job_id and result.get("batch_size") == 11,
        identity_bindings={"test": "stage24_kda02b_denominator_resume_v1"},
    )
    state = supervisor.run(iter(((job_id, task),)), stop_after_completions=1 if stop else None)
    result, artifact_sha = _artifact(root)
    return state, result, artifact_sha


def validate(spec_path: Path, output_root: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    packet_root = Path(str(spec["shadow_campaign_packet"]["packet_root"]))
    rows = [
        row for row in _read_jsonl(packet_root / "FINAL_EXECUTION_REGISTRY.jsonl")
        if row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1"
        and row["config"]["stage20_cell_id"] == CELL_ID
    ]
    if len(rows) != 11 or len({row["config"]["adjudication_variant"] for row in rows}) != 11:
        raise RuntimeError("exact 11-row KDA02B adjudication registry is absent")
    one_state, one_result, one_sha = _run(spec, rows, output_root / "worker-1", 1)
    four_state, four_result, four_sha = _run(spec, tuple(reversed(rows)), output_root / "worker-4-reversed", 4)
    stopped, stopped_result, stopped_sha = _run(spec, rows, output_root / "restart", 2, stop=True)
    resumed, resumed_result, resumed_sha = _run(spec, tuple(reversed(rows)), output_root / "restart", 2)
    result_hashes = {
        "worker_1": canonical_hash(one_result),
        "worker_4_reversed": canonical_hash(four_result),
        "restart_before_resume": canonical_hash(stopped_result),
        "restart_after_resume": canonical_hash(resumed_result),
    }
    variants = sorted(row["adjudication_variant"] for row in one_result["batch_results"])
    denominator_pass = all(
        row["denominator_reconciliation"]["aggregate_materialized_equal"] is True
        and row["stage20_denominator_contract"]["eligible_days"] == 823
        and row["stage20_denominator_contract"]["eligible_symbols"] == 187
        for row in one_result["batch_results"]
    )
    passed = (
        len(set(result_hashes.values())) == 1
        and one_state["status"] == "complete"
        and four_state["status"] == "complete"
        and stopped["status"] == "graceful_bound_stop"
        and resumed["status"] == "complete"
        and len(one_result["batch_results"]) == 11
        and denominator_pass
        and one_sha == four_sha == stopped_sha == resumed_sha
    )
    report = {
        "schema": "stage24_kda02b_all_variants_invariance_v1",
        "status": "pass" if passed else "fail",
        "mode": "shadow_no_outcome",
        "economic_outcomes_opened": False,
        "protected_outcomes_opened": False,
        "cell_id": CELL_ID,
        "registered_variant_count": len(rows),
        "completed_variant_count": len(one_result["batch_results"]),
        "variants": variants,
        "worker_count_invariant": result_hashes["worker_1"] == result_hashes["worker_4_reversed"],
        "input_order_invariant": result_hashes["worker_1"] == result_hashes["worker_4_reversed"],
        "restart_idempotent": result_hashes["restart_before_resume"] == result_hashes["restart_after_resume"],
        "aggregate_materialized_equal_all_variants": denominator_pass,
        "result_hashes": result_hashes,
        "artifact_sha256": {
            "worker_1": one_sha,
            "worker_4_reversed": four_sha,
            "restart_before_resume": stopped_sha,
            "restart_after_resume": resumed_sha,
        },
        "supervisor_status": {
            "worker_1": one_state["status"],
            "worker_4_reversed": four_state["status"],
            "restart_before_resume": stopped["status"],
            "restart_after_resume": resumed["status"],
        },
        "report_sha256": canonical_hash({
            "variants": variants,
            "result_hashes": result_hashes,
            "artifact_sha256": [one_sha, four_sha, stopped_sha, resumed_sha],
        }),
    }
    if not passed:
        raise RuntimeError(f"KDA02B all-variant invariance failed: {report}")
    atomic_write_json(output_root / "KDA02B_ALL_VARIANTS_AND_INVARIANCE.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all 11 KDA02B variants and resume invariance")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError("KDA02B validation output root already contains evidence")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = validate(args.spec, args.output_root)
    print(json.dumps({"status": report["status"], "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
