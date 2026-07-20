#!/usr/bin/env python3
"""Fail-closed validation for the Stage 21 no-outcome campaign packet."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = Path("/opt/testerdonch/results/rebaseline/phase_kraken_derivatives_campaign_stage20_20260720_v01")
ROOT = HERE.parents[3]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


required = [
    "STAGE20_RESEARCH_ADEQUACY_AUDIT.md", "STAGE20_RESEARCH_ADEQUACY_AUDIT.json",
    "STAGE20_SEARCH_COVERAGE_MATRIX.csv", "STAGE20_FOLD_AND_CANDIDATE_SUMMARY.csv",
    "STAGE20_TRANSLATION_PROVENANCE.jsonl",
    "CORE_CAMPAIGN_MECHANISM_CARDS.json", "OUTCOME_FREE_MEASUREMENT_PLAN.json",
    "REGISTERED_DISCOVERY_SPACE.json", "CONTEXT_AND_ABLATION_REGISTRY.json",
    "FULL_REGISTERED_CONFIGURATION_REGISTRY.jsonl",
    "INNER_OUTER_FOLD_MAP.json", "SELECTION_AND_PLATEAU_CONTRACT.json",
    "CONTROL_AND_FORENSIC_CONTRACT.json", "UNATTENDED_RUNTIME_AND_RECOVERY_CONTRACT.json",
    "RESOURCE_PROJECTION.json", "CAMPAIGN_MANIFEST.json", "HUMAN_APPROVAL_REQUEST.json",
    "FULL_CONTROL_TEMPLATE_REGISTRY.jsonl", "CAMPAIGN_EXECUTION_SPEC.json",
    "synthetic_campaign_supervisor.py", "recompute_stage20_candidate_summary.py",
    "NO_OUTCOME_RUNTIME_BENCHMARK.json", "NO_OUTCOME_RUNTIME_RESUME_BENCHMARK.json",
    "NO_OUTCOME_BOUND_STOP_CANARY.json", "NO_OUTCOME_BOUND_STOP_RECOVERY.json",
]
for name in required:
    require((HERE / name).is_file() and (HERE / name).stat().st_size > 0, f"missing {name}")

audit = load("STAGE20_RESEARCH_ADEQUACY_AUDIT.json")
provenance_rows = [json.loads(line) for line in (HERE / "STAGE20_TRANSLATION_PROVENANCE.jsonl").read_text(encoding="utf-8").splitlines()]
require(len(provenance_rows) == 186 and len({row["cell_id"] for row in provenance_rows}) == 186, "translation provenance ledger drift")
require(audit["translation_cell_provenance_ledger"]["sha256"] == sha(HERE / "STAGE20_TRANSLATION_PROVENANCE.jsonl"), "translation provenance hash drift")
for name, expected in audit["terminal_artifact_hashes"].items():
    if name == "terminal_handoff_zip":
        continue
    require(sha(RUN / name) == expected, f"Stage20 artifact drift: {name}")

with (HERE / "STAGE20_FOLD_AND_CANDIDATE_SUMMARY.csv").open(newline="", encoding="utf-8") as handle:
    candidates = list(csv.DictReader(handle))
require(len(candidates) == 45, "candidate summary must have 45 rows")
require(len({row["cell_id"] for row in candidates}) == 19, "candidate summary must have 19 unique cells")
require({row["outer_quarter"] for row in candidates} == {f"{year}Q{quarter}" for year in (2023, 2024, 2025) for quarter in range(1, 5) if not (year == 2023 and quarter < 4)}, "quarter coverage drift")
for row in candidates:
    require(abs(float(row["development_aggregate_base_net_mean_bps"]) - float(row["development_base_net_recomputed_mean_bps"])) < 1e-10, "development recomputation mismatch")
    require(abs(float(row["outer_aggregate_base_net_mean_bps"]) - float(row["outer_base_net_recomputed_mean_bps"])) < 1e-10, "outer recomputation mismatch")
    for prefix in ("development_", "outer_"):
        separate = max(float(row[prefix + key]) for key in ("symbol_contribution", "day_contribution", "year_contribution"))
        require(abs(separate - float(row[prefix + "symbol_day_year_contribution"])) < 1e-10, "concentration decomposition mismatch")
with tempfile.TemporaryDirectory() as directory:
    reproduced = Path(directory) / "candidate_summary.csv"
    subprocess.run(["/opt/testerdonch/.venv/bin/python", str(HERE / "recompute_stage20_candidate_summary.py"), "--output", str(reproduced)], check=True)
    require(sha(reproduced) == sha(HERE / "STAGE20_FOLD_AND_CANDIDATE_SUMMARY.csv"), "candidate summary byte replay mismatch")

space = load("REGISTERED_DISCOVERY_SPACE.json")
expected_attempts = {"A4_TSMOM_V7": 864, "A1_COMPRESSION_V2": 1040,
                     "A2_PRIOR_HIGH_RS_CONTEXT_V1": 704, "A3_STARTER_RETEST_V3": 704,
                     "KDA02B_SURVIVOR_ADJUDICATION_V1": 209, "total": 3521}
require(space["new_attempt_budget"] == expected_attempts, "attempt budget drift")
for family in space["families"]:
    require(family["screening"] + family["anchors_ablations"] + family["refinement_reserve"] == expected_attempts[family["family_id"]], f"family attempt arithmetic drift: {family['family_id']}")
registry_lines = [json.loads(line) for line in (HERE / "FULL_REGISTERED_CONFIGURATION_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines()]
require(len(registry_lines) == 3521, "full configuration registry row drift")
require(len({row["attempt_id"] for row in registry_lines}) == 3521, "duplicate attempt ID")
require(len({row["canonical_economic_address_sha256"] for row in registry_lines}) == 3521, "duplicate economic address")
require(space["full_configuration_registry"]["sha256"] == sha(HERE / "FULL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"), "configuration registry hash drift")
require(sum(row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1" for row in registry_lines) == 209, "KDA02B attempt drift")
for family_id, total in expected_attempts.items():
    if family_id == "total": continue
    require(sum(row["family_id"] == family_id for row in registry_lines) == total, f"registry family count drift: {family_id}")
for row in registry_lines:
    if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        require(row["config"]["parent_binding"] in {"A1_COMPRESSION_V2:anchor_ablation:0001", "A3_STARTER_RETEST_V3:anchor_ablation:0001"}, "A2 parent binding drift")
expected_a4_source = {
    "A4_TSMOM_V7:anchor_ablation:0001": (20, 0.2, "1d", "long_flat", "time_1d"),
    "A4_TSMOM_V7:anchor_ablation:0002": (60, 0.2, "1d", "long_flat", "time_1d"),
}
for attempt_id, expected in expected_a4_source.items():
    row = next(item for item in registry_lines if item["attempt_id"] == attempt_id)
    config = row["config"]
    require((config["lookback_days"], config["annualized_vol_target"], config["rebalance"], config["direction"], config["exit"]) == expected, f"A4 source anchor drift: {attempt_id}")
    require(row["provenance"] == "source_prior_partial_anchor_and_single_signal_null" and row["source_or_null_reference"].startswith("tsmom_v6_"), f"A4 source provenance drift: {attempt_id}")
kda_variants = {row["config"]["adjudication_variant"] for row in registry_lines if row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1"}
require(len(kda_variants) == 11 and "funding_zero" in kda_variants and "base_cost_14bps" not in kda_variants, "KDA distinct variant drift")
controls = load("CONTROL_AND_FORENSIC_CONTRACT.json")
control_templates = [json.loads(line) for line in (HERE / "FULL_CONTROL_TEMPLATE_REGISTRY.jsonl").read_text(encoding="utf-8").splitlines()]
require(len(control_templates) == 800 and len({row["control_template_address_sha256"] for row in control_templates}) == 800, "control template registry drift")
require(controls["control_template_registry"]["sha256"] == sha(HERE / "FULL_CONTROL_TEMPLATE_REGISTRY.jsonl"), "control registry hash drift")
require(controls["family_control_slots"]["A4_TSMOM_V7"][0]["control_id"] == "A4_SIGN_PERMUTED_MAIN_NULL", "A4 main null missing")
require(controls["family_control_slots"]["A3_STARTER_RETEST_V3"][0]["control_id"] == "A3_RETEST_TIME_PERMUTED_MAIN_NULL", "A3 main null missing")
require(all(len(rows) == 5 for rows in controls["family_control_slots"].values()), "family control slot drift")
require(len({(row["family_id"], row["control_id"]) for row in control_templates}) == 20, "family-specific control template drift")
semantics = space["feature_and_execution_semantics"]
require("w_i=e_i*c_i/|I_t|" in semantics["A4_weighted_aggregation"], "A4 aggregation discretion remains")
require("Trade count counts parents" in semantics["A3_leg_accounting"] and "parent_exit_ts=max leg exits" in semantics["A3_leg_accounting"], "A3 parent/leg discretion remains")

folds = load("INNER_OUTER_FOLD_MAP.json")
require(len(folds["outer_folds"]) == 8, "must have eight forward quarterly folds")
require([row["outer_fold_id"] for row in folds["outer_folds"]] == ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"], "outer fold order drift")
for outer in folds["outer_folds"]:
    require(outer["outer_evaluation_end_exclusive"] <= "2026-01-01T00:00:00Z", "protected boundary violation")
    require(len(outer["inner_folds"]) >= 5, "too few registered inner folds")
    require(all(inner["validation_end_exclusive"] < outer["outer_evaluation_start"] for inner in outer["inner_folds"]), "inner/outer leakage")

runtime = load("UNATTENDED_RUNTIME_AND_RECOVERY_CONTRACT.json")
resource = load("RESOURCE_PROJECTION.json")
require(runtime["limits"] == resource["limits"], "resource/runtime limit drift")
require(runtime["scheduler"]["workers"] == 4 and runtime["scheduler"]["max_in_flight"] == 4, "scheduler bound drift")
require(runtime["notifications"]["heartbeat_seconds"] == 1800, "heartbeat drift")
benchmarks = {name: load(name) for name in ["NO_OUTCOME_RUNTIME_BENCHMARK.json", "NO_OUTCOME_RUNTIME_RESUME_BENCHMARK.json", "NO_OUTCOME_BOUND_STOP_CANARY.json", "NO_OUTCOME_BOUND_STOP_RECOVERY.json"]}
for name, row in benchmarks.items():
    require(row["outcome_reader_opened"] is False and row["protected_rows_opened"] == 0 and row["Capitalcom_payload_opened"] is False, f"benchmark firewall failure: {name}")
    require(row["registry_sha256"] == sha(HERE / "FULL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"), f"benchmark registry binding failure: {name}")
require(benchmarks["NO_OUTCOME_RUNTIME_BENCHMARK.json"]["status"] == "terminal_complete" and benchmarks["NO_OUTCOME_RUNTIME_BENCHMARK.json"]["completed_jobs"] == 3521, "cold benchmark failure")
require(benchmarks["NO_OUTCOME_RUNTIME_RESUME_BENCHMARK.json"]["status"] == "terminal_complete" and benchmarks["NO_OUTCOME_RUNTIME_RESUME_BENCHMARK.json"]["idempotent_resume_markers"] == 3521, "resume benchmark failure")
require(benchmarks["NO_OUTCOME_BOUND_STOP_CANARY.json"]["status"] == "bound_stop", "bound-stop canary failure")
require(benchmarks["NO_OUTCOME_BOUND_STOP_RECOVERY.json"]["status"] == "terminal_complete" and benchmarks["NO_OUTCOME_BOUND_STOP_RECOVERY.json"]["idempotent_resume_markers"] >= 100, "bound-stop recovery failure")

manifest = load("CAMPAIGN_MANIFEST.json")
approval = load("HUMAN_APPROVAL_REQUEST.json")
require(manifest["status"] == "frozen_not_authorized" and not manifest["economic_run_authorized_by_manifest"], "manifest self-authorization")
require(approval["status"] == "exact_human_approval_required", "approval status drift")
require(approval["campaign_manifest_sha256"] == sha(HERE / "CAMPAIGN_MANIFEST.json"), "approval/manifest hash mismatch")
require(approval["economic_dependency_sha256"] == manifest["economic_dependency_sha256"], "economic dependency mismatch")
for name, expected in manifest["contract_file_sha256"].items():
    require(sha(HERE / name) == expected, f"contract hash mismatch: {name}")

physical_dependencies = {
    "kraken_k0_download_manifest": Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"),
    "stage8a_feature_cache_manifest": ROOT / "docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1/KDA_FEATURE_CACHE_MANIFEST.json",
    "shared_funding_panel_manifest": Path("/opt/testerdonch/results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1/funding/shared_funding_panel_manifest.csv"),
    "rankable_funding_package": Path("/opt/testerdonch/results/rebaseline/phase_kraken_local_official_funding_export_20260720_v4/kraken_funding_rankable_2023_2025.zip"),
    "stage19_funding_contract": ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2/FUNDING_COST_AND_COVERAGE_CONTRACT.json",
    "stage19_gap_allowance_table": ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2/FUNDING_GAP_ALLOWANCE_TABLE.csv",
    "tier1_prior_input_manifest": Path("/opt/testerdonch/results/rebaseline/phase_kraken_uncapped_tier1_two_family_sweep_repaired_memorysafe_20260706_v1_20260706_133455/preflight/input_manifest.json"),
    "tier1_prior_frozen_artifact_hashes": Path("/opt/testerdonch/results/rebaseline/phase_kraken_uncapped_tier1_two_family_sweep_repaired_memorysafe_20260706_v1_20260706_133455/preflight/frozen_artifact_hashes.json"),
    "tsmom_v6_definition_manifest": Path("/opt/testerdonch/results/rebaseline/phase_kraken_uncapped_tier1_two_family_sweep_repaired_memorysafe_20260706_v1_20260706_133455/tsmom/redesign/tsmom_curated_sweep_definitions_v6.csv"),
    "first_wave_program_decision_register": Path("/opt/testerdonch/results/rebaseline/phase_kraken_first_wave_closure_review_20260717_v1/PROGRAM_DECISION_REGISTER.csv"),
    "first_wave_authority_and_scope": Path("/opt/testerdonch/results/rebaseline/phase_kraken_first_wave_closure_review_20260717_v1/AUTHORITY_AND_SCOPE.md"),
    "first_wave_lineage_map": Path("/opt/testerdonch/results/rebaseline/phase_kraken_first_wave_closure_review_20260717_v1/PRIOR_AND_FIRST_WAVE_LINEAGE_MAP.csv"),
    "stage20_external_human_approval": RUN / "HUMAN_APPROVAL.json",
    "stage20_campaign_manifest": ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2/CAMPAIGN_MANIFEST.json",
    "stage20_approval_packet": ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2/FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json",
}
for key, path in physical_dependencies.items():
    require(path.is_file(), f"missing economic dependency: {key}")
    require(sha(path) == manifest["economic_dependency_sha256"][key], f"economic dependency drift: {key}")

require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == manifest["repository_base_commit"], "repository base drift during precommit validation")
print(json.dumps({"status": "pass", "required_files": len(required), "candidate_rows": len(candidates),
                  "unique_selected_cells": 19, "new_attempts": 3521, "outer_folds": 8,
                  "physical_dependencies_verified": len(physical_dependencies), "protected_rows_opened": 0,
                  "Capitalcom_payload_opened": False}, sort_keys=True))
