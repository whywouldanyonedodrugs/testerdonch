#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq

from tools.core_liquid_campaign.campaign import CampaignOrchestrator
from tools.core_liquid_campaign.canonical import atomic_write_json, canonical_hash, sha256_file
from tools.core_liquid_campaign.kda02b_lazy_family_input import KDA02BLazyFamilyInputAdapter
from tools.core_liquid_campaign.shadow_campaign import BoundedShadowKDA02BAdapter
from tools.core_liquid_campaign.shadow_payoff import ShadowPayoffProvider


TARGET_ATTEMPT = "KDA02B_SURVIVOR_ADJUDICATION_V1:S22:L:3313:1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_trace(spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    packet_root = Path(str(spec["shadow_campaign_packet"]["packet_root"]))
    rows = _read_jsonl(packet_root / "FINAL_EXECUTION_REGISTRY.jsonl")
    row = next(item for item in rows if item["executable_attempt_id"] == TARGET_ATTEMPT)
    kda_record = spec["shadow_campaign_packet"]["kda02b_population_authority"]
    kda_manifest_path = Path(str(kda_record["path"]))
    kda_manifest = json.loads(kda_manifest_path.read_text(encoding="utf-8"))
    event_record = next(item for item in kda_manifest["files"] if item["role"] == "event_index")
    event_path = kda_manifest_path.parent / str(event_record["path"])
    table = pq.read_table(
        event_path,
        columns=["event_id", "cell_id", "model_id", "outer_fold_id", "status"],
        filters=[("cell_id", "=", str(row["config"]["stage20_cell_id"]))],
    )
    index_rows = table.to_pylist()
    statuses: dict[str, int] = {}
    for item in index_rows:
        statuses[str(item["status"])] = statuses.get(str(item["status"]), 0) + 1
    complete = KDA02BLazyFamilyInputAdapter(
        index_root=kda_manifest_path.parent,
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
    orchestrator.payoff_provider = ShadowPayoffProvider("stage24-kda02b-denominator-trace")
    job_id, task = next(iter(orchestrator._kda_jobs((row,), {TARGET_ATTEMPT: row}, {})))
    batch = task()
    result = batch["batch_results"][0]
    denominator = result["denominator_reconciliation"]
    stage20_code = Path(spec["repository_root"]) / "tools/qlmg_stage20_campaign.py"
    return {
        "schema": "stage24_kda02b_denominator_pipeline_trace_v1",
        "status": "pass",
        "mode": "shadow_no_outcome",
        "economic_outcomes_opened": False,
        "protected_outcomes_opened": False,
        "target": {
            "attempt_id": TARGET_ATTEMPT,
            "job_id": job_id,
            "cell_id": row["config"]["stage20_cell_id"],
            "adjudication_variant": row["config"]["adjudication_variant"],
            "canonical_economic_address_sha256": row["canonical_economic_address_sha256"],
        },
        "authority": {
            "kda02b_population_manifest": {"path": str(kda_manifest_path), "sha256": sha256_file(kda_manifest_path)},
            "event_index": {"path": str(event_path), "sha256": sha256_file(event_path)},
            "stage20_event_tape_manifest": kda_manifest["source_records"]["event_manifest"],
            "stage20_fold_thresholds": kda_manifest["source_records"]["fold_thresholds"],
            "stage20_executable_accounting_code": {"path": str(stage20_code), "sha256": sha256_file(stage20_code)},
        },
        "pipeline": {
            "raw_authority_rows": len(index_rows),
            "raw_authority_unique_event_identities": len({str(item["event_id"]) for item in index_rows}),
            "pit_eligible_rows": statuses.get("eligible", 0),
            "typed_unavailable_rows": statuses.get("typed_unavailable", 0),
            "data_complete_shadow_frames": int(result["population_eligible_records"]),
            "bounded_frames_by_outer_fold": bounded.last_reconciliation["eligible_records_by_outer_fold"],
            "episode_constructed_observations": int(result["generated_observation_count"]),
            "joined_deduplicated_economic_event_identities": len(set(result["event_ids"])),
            "overlap_suppressed_observations": int(result["overlap_suppressed_observation_count"]),
            "side_counts": result["side_counts"],
            "component_filter": row["config"]["adjudication_variant"],
            "zero_exposure_observations": denominator["zero_exposure_observation_count"],
            "aggregate_numerators": denominator["aggregate_numerators"],
            "aggregate_denominators": denominator["aggregate_denominators"],
            "streaming_aggregate_sha256": denominator["streaming_aggregate_sha256"],
            "materialized_aggregate_sha256": denominator["materialized_aggregate_sha256"],
            "aggregate_materialized_equal": denominator["aggregate_materialized_equal"],
        },
        "stage20_denominator_semantics": result["stage20_denominator_contract"],
        "first_divergent_boundary": {
            "location": "KDA02BLazyFamilyInputAdapter metadata construction before engine dispatch",
            "prior_behavior": "occupancy used 117 currently KDA-eligible symbols instead of Stage20's full 187-symbol campaign denominator",
            "required_behavior": "bind every source quarter to Stage20's 187-symbol denominator; after cross-fold non-overlap, validate the 90/91/92-day quarter denominators, sum the nine disjoint intervals, and bind that one job denominator to every accepted observation",
            "downstream_exception_boundary": "CampaignOrchestrator._kda_jobs aggregate_streaming rejected the still-mixed 90/91/92-day per-quarter denominators",
            "economic_semantics_changed": False,
        },
        "trace_sha256": canonical_hash({
            "attempt_id": TARGET_ATTEMPT,
            "raw_rows": len(index_rows),
            "statuses": statuses,
            "denominator_contract": result["stage20_denominator_contract"],
            "denominator_reconciliation": denominator,
        }),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace the Stage24 KDA02B denominator boundary without economic outcomes")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = build_trace(args.spec)
    atomic_write_json(args.output, trace)
    print(json.dumps({"status": trace["status"], "output": str(args.output), "trace_sha256": trace["trace_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
