#!/usr/bin/env python3
"""Build deterministic Stage 13 outcome-free campaign-readiness records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qlmg_research_campaign import atomic_write_json, build_dag, canonical_bytes, sha256_bytes, sha256_file, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
STAGE8 = ROOT / "docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1"
FIXED_AT = "2026-07-19T00:00:00Z"
HASHES = {
    "repository_base_commit": "a457742c50e27071c7a9f4d7f4dc4c5677534ea7",
    "analytics_manifest_sha256": "f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d",
    "cohort_sha256": "5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636",
    "feature_contract_sha256": "4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4",
    "generator_contract_sha256": "c285af918e317722e91e3c1df5e44eb48abf27c97fedfe251b0881e3aa8f0017",
    "protocol_v2_sha256": "4effe3e9608377876486b1583854a7e1479c8e93d6de1661f13d35842bbc6a73",
    "family_redefinition_policy_sha256": "d07b0d290e59d17f7d6a587e2a31e6e550468f9ad403250cd6183c549e685335",
    "campaign_protocol_sha256": "7091156df9bf815a001423b63d673d2d4217f2616fd40e76a261f218a2d614df",
}

CANDIDATES = [
    ("KDA02B_v2_oi_vacuum_redevelopment", "KDA02B", "derivatives_state", "S_KDA02B", "program_exposed_historical"),
    ("KDA02C_v1_purge_breadth_context", "KDA02C", "derivatives_state", "S_KDA02C", "program_exposed_historical"),
    ("KDX01_v1_downside_completed_derivatives_state_rejection", "KDX01", "derivatives_state", "S_KDX01", "program_exposed_historical"),
    ("C17_v1_executed_catalyst_state", "C17", "catalyst", "S_C17", "program_exposed_historical"),
]


def oi_counts() -> dict:
    path = STAGE8 / "KDA02_EVENT_COUNT_MATRIX.csv"
    yearly: dict[str, int] = {}
    symbols: set[str] = set()
    directions: dict[str, int] = {}
    total = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["definition_id"] != "kda02_primary_oi_vacuum":
                continue
            count = int(row["event_count"]); total += count
            yearly[row["year"]] = yearly.get(row["year"], 0) + count
            directions[row["direction"]] = directions.get(row["direction"], 0) + count
            symbols.add(row["symbol"])
    return {"onset_event_count": total, "year_counts": yearly, "direction_counts": directions,
            "symbol_count": len(symbols), "source": str(path.relative_to(ROOT))}


def cells(prefix: str, axes: dict[str, list[str]]) -> list[dict]:
    rows = [{}]
    for name, values in axes.items():
        rows = [dict(row, **{name: value}) for row in rows for value in values]
    return [{"cell_id": f"{prefix}_{index:03d}", "parameters": row} for index, row in enumerate(rows, 1)]


def search_spaces() -> list[dict]:
    specs = [
        ("S_KDA02B", "KDA02B", {"oi_form": ["within_symbol_60d_closed_5m_percentile_rank_of_oi_log_change_1h", "oi_log_change_1h_lt_0"], "price_displacement_cap": ["abs_trade_return_1h_lte_1_times_realized_vol_24h", "abs_trade_return_1h_lte_2_times_realized_vol_24h"], "payoff": ["opposite_sign_of_trade_return_1h_at_decision", "same_sign_as_trade_return_1h_at_decision"]}),
        ("S_KDA02C", "KDA02C", {"purge_state": ["kda02_primary_completed", "kda02_robust_completed"], "breadth_window": ["completed_onsets_in_trailing_15m", "completed_onsets_in_trailing_60m"], "breadth_form": ["count_divided_by_PIT_eligible_cohort_size", "development_fold_only_tercile_of_count_divided_by_PIT_eligible_cohort_size"]}),
        ("S_KDX01", "KDX01", {"completed_state": ["kda01_primary_failure", "kda02_primary_completed", "kda03_negative_completed_basis_rejection"], "confirmation": ["trade_5m_and_mark_1h_same_downside_sign", "mark_1h_downside_sign_only"], "horizon": ["1h_from_actual_entry", "6h_from_actual_entry"]}),
        ("S_C17", "C17", {"execution_state": ["audited_effective_timestamp_lte_decision_ts", "audited_announcement_timestamp_lte_decision_ts_control"], "participation": ["PIT_leader_response", "PIT_eligible_cohort_breadth"], "branch": ["continuation_same_sign_as_catalyst_state", "failure_opposite_sign_after_initial_response"]}),
    ]
    result = []
    for sid, family, axes in specs:
        registered = cells(family, axes)
        result.append({"search_space_id": sid, "family_id": family, "axes": axes,
                       "registered_cells": registered,
                       "registered_cell_ids": [row["cell_id"] for row in registered],
                       "explored_cell_budget": len(registered), "complexity": "main_effects_and_one_registered_interaction_only"})
    return result


def folds() -> list[dict]:
    quarters = [
        ("2023Q4", "2023-09-30T18:00:00Z", "2023-10-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        ("2024Q1", "2023-12-31T18:00:00Z", "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z"),
        ("2024Q2", "2024-03-31T18:00:00Z", "2024-04-01T00:00:00Z", "2024-07-01T00:00:00Z"),
        ("2024Q3", "2024-06-30T18:00:00Z", "2024-07-01T00:00:00Z", "2024-10-01T00:00:00Z"),
        ("2024Q4", "2024-09-30T18:00:00Z", "2024-10-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        ("2025Q1", "2024-12-31T18:00:00Z", "2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z"),
        ("2025Q2", "2025-03-31T18:00:00Z", "2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z"),
        ("2025Q3", "2025-06-30T18:00:00Z", "2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
        ("2025Q4", "2025-09-30T18:00:00Z", "2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    ]
    rows = []
    for hid, _, _, _, _ in CANDIDATES:
        for label, development_end, start, end in quarters:
            rows.append({"fold_id": f"{hid}:{label}", "hypothesis_id": hid,
                         "development_start": "2023-04-01T00:00:00Z", "development_end": development_end,
                         "embargo_start": development_end, "embargo_end": start,
                         "evaluation_start": start, "evaluation_end": end,
                         "embargo": "six_hours_plus_runtime_assertion_of_episode_nonoverlap_using_actual_executable_exit",
                         "exposure_class": "program_exposed_historical",
                         "independent_validation_claim": False})
    return rows


def readiness() -> dict:
    common = {"decision_semantics": "completed_5m source state; decision_ts=source timestamp+5m; source_close_ts<=decision_ts",
              "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
              "cost_break_even": "14 bps base and 32 bps stress all-in round trip before signed exact funding; inherited from Stage 9 contract, with no outcome data read here",
              "programme_exposure_class": "program_exposed_historical", "independent_validation_claim": False,
              "source_data_semantic_authority":"CAMPAIGN_MANIFEST.json#repository_and_data_hashes",
              "fold_schedule_ref":"CAMPAIGN_MANIFEST.json#fold_schedule",
              "mechanical_stop_rules":["required_measurement_missing","source_close_after_decision","protected_row_detected","search_cell_unregistered","cost_contract_unbound"],
              "resource_projection":{"max_workers":1,"phase_0_1_wall_minutes":30,"phase_2_5_upper_bound":"campaign shared limit; benchmark required"}}
    oi = oi_counts()
    return {"generated_at": FIXED_AT, "economic_outputs_computed": False, "candidates": [
        dict(common, hypothesis_id=CANDIDATES[0][0], parent_family_id="KDA02", parent_translation_ids=["KDA02A", "KDA02B"],
             mechanism="leveraged holders reduce open interest without a liquidation burst; dealer and inventory rebalancing may create reversal or continuation conditional on price displacement",
             compelled_actor="leveraged derivatives holders and liquidity providers absorbing position reduction", main_null="OI decline is ordinary churn with no incremental state information",
             raw_measurement="oi_log_change_1h in base-unit OI; price displacement trade_return_1h versus realized_vol_24h; no bps conversion applies to OI quantity",
             raw_magnitude_status="not_summarized_in_existing_outcome_free_matrix; native OI-change and price-displacement distributions required",
             event_frequency=oi, duration={"status":"not_measured", "reason":"Stage 8 matrix records false-to-true onsets, not state duration"},
             search_space_id="S_KDA02B", allowed_search_lanes=["response_surface","component_incrementality","horizon_payoff"], complexity_budget=8, multiplicity_family="KDA02_inherited",
             component_controls=["price_displacement_only", "OI_change_only", "liquidation_present_vs_absent", "direction_balance"],
             measurement_weaknesses=["analytics semantics are inferred_authoritative_v1 rather than exchange-documented units", "current-roster/lifecycle-capped cohort", "OI is unsigned base-unit quantity and does not identify actor direction", "OI retention truncates exact history and creates the first usable onset boundary", "duration unavailable from count matrix"],
             phase_1_status="blocked_raw_magnitude_duration_retention_boundary_and_semantic_limit_acceptance_required"),
        dict(common, hypothesis_id=CANDIDATES[1][0], parent_family_id="KDA02", parent_translation_ids=["KDA02A", "KDA02C"],
             contamination_label="post_hoc_context_hypothesis", mechanism="simultaneous completed purge states proxy a market-wide forced-deleveraging balance sheet rather than an isolated instrument event",
             compelled_actor="cross-market leveraged holders and shared liquidity providers", main_null="breadth is a mechanical consequence of common price beta and adds no information",
             raw_measurement="cross-sectional count/share of PIT-eligible symbols in a completed purge state; unit is symbols and cohort share",
             event_frequency={"status":"not_materialized_for_new_breadth_translation"}, duration={"status":"not_materialized"},
             search_space_id="S_KDA02C", allowed_search_lanes=["response_surface","component_incrementality"], complexity_budget=8, multiplicity_family="KDA02_inherited_post_hoc",
             component_controls=["isolated_completed_purge", "market_return", "eligible_cohort_size", "major_vs_alt"],
             measurement_weaknesses=["post-hoc origin", "breadth series not yet materialized", "current roster is not survivorship-free"],
             phase_1_status="blocked_measurement_materialization_required"),
        dict(common, hypothesis_id=CANDIDATES[2][0], parent_family_id="KDA01_KDA02_KDA03_cross_family", parent_translation_ids=["KDA01", "KDA02A", "KDA03_NEG_REJECTION_6H"],
             contamination_label="cross_family_program_exposed_redevelopment", mechanism="a downside completed state jointly visible in price, mark, OI, basis or liquidation can identify forced risk reduction after an incomplete crowded move",
             compelled_actor="long-side leveraged holders, margin-constrained traders, and liquidity providers", main_null="the combination restates contemporaneous downside price movement",
             raw_measurement="registered completed-state components in native OI quantity, decimal basis (bps=decimal*10000), liquidation base-unit flow, and return units",
             event_frequency={"status":"not_materialized_for_new_cross_family_translation"}, duration={"status":"not_materialized"},
             search_space_id="S_KDX01", allowed_search_lanes=["component_incrementality","sparse_model","horizon_payoff"], complexity_budget=12, multiplicity_family="KDA_cross_family_inherited",
             component_controls=["price_only", "mark_confirmation_removed", "OI_removed", "basis_removed", "liquidation_removed"],
             measurement_weaknesses=["explicit prior outcome contamination", "new joint-state incidence not materialized", "liquidation side is only a proxy"],
             phase_1_status="blocked_measurement_materialization_required"),
        dict(common, hypothesis_id=CANDIDATES[3][0], parent_family_id="C17", parent_translation_ids=["C2", "H39", "H40", "H41", "H42"],
             mechanism="an executed change in access, legality, supply, fees, utility, or distribution compels repricing; participation distinguishes continuation from trapped anticipation",
             compelled_actor="holders, users, issuers, market makers, and access-constrained investors directly affected by the executed change", main_null="ordinary price structure explains the response without catalyst execution or participation state",
             raw_measurement="100 submitted source records consolidated to 98 logical records: 59 high-confidence, 27 medium-confidence, 12 excluded; no market response opened",
             event_frequency={"logical_seed_records":98, "high_confidence":59, "medium_confidence":27, "excluded":12, "status":"source_verified_seed_not_closed_census"},
             duration={"status":"not_applicable_until_execution-state timestamps are audited"},
             search_space_id="S_C17", allowed_search_lanes=["response_surface","component_incrementality"], complexity_budget=8, multiplicity_family="catalyst_lineage_inherited_not_fully_recoverable",
             component_controls=["matched_non_event_date", "announcement_vs_execution", "leader_removed", "breadth_removed", "failed_catalyst"],
             measurement_weaknesses=["sample-limited seed", "not a closed census", "timestamp phase/effective access/duplicates require audit"],
             phase_1_status="blocked_independent_timestamp_and_fold_authority_required", future_route="separate_campaign_not_derivatives_batch")
    ]}


def manifest(implementation_commit: str = "UNCOMMITTED_STAGE13") -> dict:
    spaces = search_spaces()
    authorities = dict(HASHES, repository_implementation_commit=implementation_commit,
                       campaign_engine_sha256=sha256_file(ROOT / "tools/qlmg_research_campaign.py"),
                       readiness_builder_sha256=sha256_file(Path(__file__).resolve()))
    return {"campaign_id":"kraken_research_campaign_001_readiness", "version":"1.0", "mode":"measurement_only",
            "economic_run_authorized_by_manifest":False, "repository_and_data_hashes":authorities,
            "hypotheses":[{"hypothesis_id":hid,"family_id":family,"lane":lane,"search_space_id":sid,"programme_exposure_class":exposure} for hid,family,lane,sid,exposure in CANDIDATES],
            "fold_schedule":folds(), "search_spaces":spaces,
            "selection_algorithm":{"method":"predeclared_pareto","objectives":["cross_fold_sign_consistency","cost_margin","coverage","simplicity"],"tie_break":["lower_complexity","larger_effective_cluster_count","lexical_cell_id"],"backward_information_flow":False},
            "candidate_beam":{"max_retained_per_hypothesis":5,"max_outer_evaluated_per_hypothesis":3},
            "cost_and_execution":{"instrument":"Kraken linear derivatives","authority":"docs/agent/task_archive/20260719_donch_bt_stage_9_kda02_purge_adjudication_conditional_level3_20260719_v1/KDA02_FINAL_LEVEL3_CONTRACT.json","base_all_in_round_trip_bps_before_funding":14,"stress_all_in_round_trip_bps_before_funding":32,"entry":"first authorized PF 5m trade-bar open at or after decision_ts","exit":"first authorized PF 5m trade-bar open at or after exit target","maximum_entry_delay_minutes":10,"maximum_exit_delay_minutes":10,"non_overlap":"translation-local symbol-local using actual executable exit","funding":"include signed cashflow at exact venue timestamps; missing exact required funding fails closed","required_components":["taker_fee","spread_crossing","slippage","funding"],"same_bar_heroics":False},
            "multiplicity":{"method":"family_level_complete_cell_registry_plus_selection_bias_reporting","inherit_parent_attempts":True,"historical_attempt_counts_not_fully_recoverable_are_never_zero":True},
            "phase_permissions":{str(i):i in (0,1) for i in range(8)},
            "resource_limits":{"wall_seconds":14400,"max_workers":4,"max_disk_bytes":5368709120,"max_cells_total":sum(len(x["registered_cell_ids"]) for x in spaces)},
            "stop_conditions":{"family_only":["mechanically_unavailable","no_development_candidate","search_budget_exhausted","mechanism_underidentified","family_specific_defect"],"global":["shared_authority_failure","protected_exposure","shared_timestamp_defect","unsafe_git_or_storage","shared_replay_failure"]},
            "review_requirements":{"before_phase_2":"independent_protocol_code_readiness_review","before_outer_fold":"manifest_and_freeze_hash_check","after_campaign":"artifact_reconciliation_and_claim_review"},
            "archive_and_handoff":{"task_archive_required":True,"artifact_manifest_required":True,"approved_drive_target":"docs/agent/DRIVE_HANDOFF_TARGET.md","overwrite":False}}


def build(output: Path, implementation_commit: str = "UNCOMMITTED_STAGE13") -> None:
    output.mkdir(parents=True, exist_ok=True)
    campaign = manifest(implementation_commit); validate_manifest(campaign)
    ready = readiness()
    redefinitions = {"generated_at":FIXED_AT,"records":[
        {"parent_family_id":"KDA02","parent_translation_ids":["KDA02A","KDA02B"],"new_translation_id":CANDIDATES[0][0],"material_difference":"continuous OI-vacuum response surface without requiring liquidation","search_space_id":"S_KDA02B","programme_exposure_class":"program_exposed_historical","multiplicity_family":"KDA02_inherited","independent_evidence_plan":"future unseen prospective period; 2023-2025 cannot qualify"},
        {"parent_family_id":"KDA02","parent_translation_ids":["KDA02A","KDA02C"],"new_translation_id":CANDIDATES[1][0],"material_difference":"cross-sectional purge breadth as predecision context","search_space_id":"S_KDA02C","programme_exposure_class":"program_exposed_historical","multiplicity_family":"KDA02_inherited_post_hoc","independent_evidence_plan":"future unseen prospective period after translation freeze"},
        {"parent_family_id":"KDA01_KDA02_KDA03_cross_family","parent_translation_ids":["KDA01","KDA02A","KDA03_NEG_REJECTION_6H"],"new_translation_id":CANDIDATES[2][0],"material_difference":"downside completed-state synthesis with component incrementality","search_space_id":"S_KDX01","programme_exposure_class":"program_exposed_historical","multiplicity_family":"KDA_cross_family_inherited","independent_evidence_plan":"future unseen prospective period; explicit contaminated development"},
        {"parent_family_id":"C17","parent_translation_ids":["C2","H39","H40","H41","H42"],"new_translation_id":CANDIDATES[3][0],"material_difference":"executed/effective catalyst state crossed with leader or breadth participation","search_space_id":"S_C17","programme_exposure_class":"program_exposed_historical","multiplicity_family":"catalyst_lineage_inherited_not_fully_recoverable","independent_evidence_plan":"separate audited catalyst campaign and future unseen prospective period"}]}
    explored = {"campaign_id":campaign["campaign_id"],"rows":[],"note":"No Phase 2 cells explored; every permitted future cell is registered in CAMPAIGN_MANIFEST.json."}
    beam = {"campaign_id":campaign["campaign_id"],"rows":[],"note":"No candidates scored or retained; economics are unauthorized."}
    state = {"campaign_id":campaign["campaign_id"],"manifest_sha256":sha256_bytes(canonical_bytes(campaign)),"generation":0,"completed_nodes":[n["node_id"] for n in build_dag(campaign) if n["phase"] == 0],"family_stops":{},"global_stop":None,"phase_1_readiness_records_built":True,"phase_2_admission":{"KDA02B_v2_oi_vacuum_redevelopment":"blocked_raw_magnitude_duration_retention_boundary_and_semantic_limit_acceptance","KDA02C_v1_purge_breadth_context":"blocked_breadth_materialization","KDX01_v1_downside_completed_derivatives_state_rejection":"blocked_joint_state_materialization","C17_v1_executed_catalyst_state":"separate_blocked_timestamp_fold_authority"}}
    approval = {"packet_id":"kraken_derivatives_campaign_001_phase2_5_approval_request","status":"human_approval_required_not_authorized","candidate_list":[x[0] for x in CANDIDATES[:3]],"C17_excluded_route":"separate campaign after independent timestamp and fold authority","repository_and_data_hashes":campaign["repository_and_data_hashes"],"campaign_manifest_sha256":state["manifest_sha256"],"quarterly_forward_fold_schedule":[x for x in campaign["fold_schedule"] if x["hypothesis_id"] in [c[0] for c in CANDIDATES[:3]]],"search_spaces":[x for x in campaign["search_spaces"] if x["family_id"]!="C17"],"candidate_beam":campaign["candidate_beam"],"selection_algorithm":campaign["selection_algorithm"],"multiplicity":campaign["multiplicity"],"cost_and_execution":campaign["cost_and_execution"],"cost_and_execution_sha256":sha256_bytes(canonical_bytes(campaign["cost_and_execution"])),"fold_schedule_sha256":sha256_bytes(canonical_bytes([x for x in campaign["fold_schedule"] if x["hypothesis_id"] in [c[0] for c in CANDIDATES[:3]]])),"search_spaces_sha256":sha256_bytes(canonical_bytes([x for x in campaign["search_spaces"] if x["family_id"]!="C17"])),"phase_permissions_requested":{"2":True,"3":True,"4":True,"5":True,"6":False,"7":False},"preconditions":["resolve every Phase 1 admission blocker without outcomes","all repository/data/code/cost hashes still match this packet","independent review acceptance","human approval artifact names this packet hash"],"resource_limits":campaign["resource_limits"],"estimated_wall_time":"up to 4 hours after input sizing benchmark; stop at resource limit","review_checkpoints":campaign["review_requirements"],"stop_conditions":campaign["stop_conditions"],"archive_and_handoff":campaign["archive_and_handoff"],"economic_run_authorized":False,"protected_outcome_access":False}
    objects = {"CAMPAIGN_MANIFEST.json":campaign,"CAMPAIGN_DAG.json":{"nodes":build_dag(campaign)},"PHASE0_1_READINESS.json":ready,"FAMILY_REDEFINITION_RECORDS.json":redefinitions,"SEARCH_SPACE_REGISTRY.json":{"search_spaces":campaign["search_spaces"]},"EXPLORED_CELL_REGISTRY.json":explored,"CANDIDATE_BEAM_REGISTRY.json":beam,"CAMPAIGN_STATE.json":state,"HEARTBEAT.json":{"campaign_id":campaign["campaign_id"],"generation":0,"heartbeat_ts":FIXED_AT,"status":"phase_0_1_readiness_built"},"FOLD_AND_EXPOSURE_MAP.json":{"rows":campaign["fold_schedule"]},"RESOURCE_PROJECTION.json":{"limits":campaign["resource_limits"],"estimated_wall_time":"up to 4 hours after input sizing benchmark","phase_0_1_actual":"stdlib metadata/count aggregation only"},"FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json":approval}
    for name, value in objects.items(): atomic_write_json(output/name, value)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--implementation-commit",default="UNCOMMITTED_STAGE13"); args=parser.parse_args(); build(args.output,args.implementation_commit)
