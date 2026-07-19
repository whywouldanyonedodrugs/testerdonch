#!/usr/bin/env python3
"""Build the deterministic, non-authorizing Stage 14 campaign packet."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"docs/agent/task_archive/20260719_donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1"
LOCAL=Path("/opt/parquet/kraken_derivatives/analytics/stage14_phase1_v1")
FIXED="2026-07-19T00:00:00Z"


def h(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): digest.update(b)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()


def jwrite(path: Path,value: object) -> None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def space(space_id: str,family: str,axes: dict[str,list[str]],budget: int,depth: int) -> dict:
    combos=list(itertools.product(*axes.values()))
    if len(combos)!=budget: raise ValueError(f"{space_id} budget mismatch")
    cells=[]
    for i,values in enumerate(combos,1): cells.append({"cell_id":f"{family}_{i:03d}","parameters":dict(zip(axes,values))})
    return {"search_space_id":space_id,"family_id":family,"axes":axes,"registered_cells":cells,"registered_cell_ids":[x["cell_id"] for x in cells],
            "maximum_cells":budget,"continuous_surface_first":True,"fold_local_rank_only":True,"maximum_interaction_depth":depth,"historical_multiplicity_inherited":True}


def folds(lanes: list[str]) -> list[dict]:
    quarters=[("2023Q4","2023-09-30T18:00:00Z","2023-10-01T00:00:00Z","2024-01-01T00:00:00Z"),("2024Q1","2023-12-31T18:00:00Z","2024-01-01T00:00:00Z","2024-04-01T00:00:00Z"),("2024Q2","2024-03-31T18:00:00Z","2024-04-01T00:00:00Z","2024-07-01T00:00:00Z"),("2024Q3","2024-06-30T18:00:00Z","2024-07-01T00:00:00Z","2024-10-01T00:00:00Z"),("2024Q4","2024-09-30T18:00:00Z","2024-10-01T00:00:00Z","2025-01-01T00:00:00Z"),("2025Q1","2024-12-31T18:00:00Z","2025-01-01T00:00:00Z","2025-04-01T00:00:00Z"),("2025Q2","2025-03-31T18:00:00Z","2025-04-01T00:00:00Z","2025-07-01T00:00:00Z"),("2025Q3","2025-06-30T18:00:00Z","2025-07-01T00:00:00Z","2025-10-01T00:00:00Z"),("2025Q4","2025-09-30T18:00:00Z","2025-10-01T00:00:00Z","2026-01-01T00:00:00Z")]
    return [{"fold_id":f"{lane}:{label}","hypothesis_id":lane,"development_start":"2023-04-01T00:00:00Z","development_end":dev_end,"embargo_start":dev_end,"embargo_end":start,"evaluation_start":start,"evaluation_end":end,"embargo":"six_hours_plus_actual_executable_exit_nonoverlap","exposure_class":"program_exposed_historical","independent_validation_claim":False} for lane in lanes for label,dev_end,start,end in quarters]


def build(implementation_commit: str = "UNCOMMITTED_STAGE14") -> None:
    measurement=json.loads((LOCAL/"PHASE1_MEASUREMENT.json").read_text()); decisions=json.loads((ARCHIVE/"PHASE1_ADMISSION_DECISIONS.json").read_text())["decisions"]
    lanes=[x for x,v in decisions.items() if v=="phase_2_ready"]
    if not lanes: raise ValueError("no ready lane")
    expected=["KDA02B_v2_oi_vacuum_redevelopment","KDA02C_v1_purge_breadth_context","KDX01_v1_downside_completed_derivatives_state_rejection"]
    if lanes!=expected: raise ValueError("unexpected ready lane order")
    spaces=[
      space("S_KDA02B_V2","KDA02B",{"oi_axis":["raw_oi_log_change_continuous","fold_local_percentile_surface"],"price_displacement_axis":["raw_bps_continuous","fold_local_rank_continuous"],"price_state":["negative","positive"],"branch":["continuation","reversal"],"horizon":["1h","3h","6h"],"liquidation_context":["continuous_intensity","present_absent"]},96,3),
      space("S_KDA02C_V1","KDA02C",{"purge_identity":["primary_z2","robust_pct95"],"diagnostic_window":["5m","15m","30m","60m"],"breadth_form":["raw_share_continuous","fold_local_rank_continuous","isolated_vs_nonisolated_increment"],"direction":["negative","positive"]},48,3),
      space("S_KDX01_V1","KDX01",{"component_ladder":["trade_mark_structural","trade_mark_structural_oi","trade_mark_structural_oi_liquidation","trade_mark_structural_oi_basis_level","trade_mark_structural_oi_basis_change","trade_mark_structural_oi_breadth","trade_mark_structural_oi_liquidation_basis_change"],"component_scaling":["raw_unit_continuous","fold_local_rank_continuous"],"direction":["continuation","reversal"],"horizon":["1h","3h","6h"]},84,6),
    ]
    total=sum(x["maximum_cells"] for x in spaces)
    if total!=228 or sum(len(x["registered_cell_ids"]) for x in spaces)!=total: raise ValueError("global cell registry mismatch")
    fold_rows=folds(lanes)
    local_files=[]
    for path in sorted(LOCAL.rglob("*")):
        if path.is_file(): local_files.append({"path":str(path.relative_to(LOCAL)),"bytes":path.stat().st_size,"sha256":h(path)})
    local_manifest={"root":str(LOCAL),"files":local_files,"file_count":len(local_files),"economic_outputs_computed":False,"protected_rows_opened":0}
    jwrite(ARCHIVE/"LOCAL_STATE_TAPE_MANIFEST.json",local_manifest)
    authorities={"repository_start_commit":"e14bbd0d26c14e48a347481f170fcfe8851df625","stage13_campaign_manifest_sha256":"09f4a534dcd799bf252e6e47037af58ad2177256d976b37defd8c257d2949e27","stage13_approval_request_sha256":"ed27fc30d39155d6b08d63539a22d34b50d19ab3c909396a8719c443057fcd5c","protocol_v2_sha256":"4effe3e9608377876486b1583854a7e1479c8e93d6de1661f13d35842bbc6a73","family_policy_sha256":"d07b0d290e59d17f7d6a587e2a31e6e550468f9ad403250cd6183c549e685335","campaign_protocol_sha256":"7091156df9bf815a001423b63d673d2d4217f2616fd40e76a261f218a2d614df","route_policy_sha256":"c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa","analytics_manifest_sha256":"f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d","authorized_cohort_sha256":"5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636","feature_contract_sha256":"4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4","local_state_tape_manifest_sha256":h(ARCHIVE/"LOCAL_STATE_TAPE_MANIFEST.json"),"phase1_measurement_sha256":h(LOCAL/"PHASE1_MEASUREMENT.json"),"funding_and_cost_contract_sha256":h(ARCHIVE/"CAMPAIGN_FUNDING_AND_COST_CONTRACT.md"),"search_space_builder_sha256":h(Path(__file__)),"measurement_primitive_code_sha256":h(ROOT/"tools/qlmg_derivatives_phase1.py"),"measurement_builder_sha256":h(ROOT/"tools/build_derivatives_phase1_closure.py"),"measurement_finalizer_sha256":h(ROOT/"tools/finalize_derivatives_phase1_closure.py"),"closure_validator_sha256":h(ROOT/"tools/validate_stage14_closure.py")}
    selection={"method":"predeclared_nested_pareto","objectives":["base_net_expectancy_after_authorized_costs","median_inner_fold_utility","worst_inner_fold_utility","cross_fold_sign_and_stability","effective_independent_cluster_count","left_tail_and_drawdown_proxy_by_archetype","opportunity_frequency","capital_occupancy","execution_margin","complexity","candidate_correlation"],"candidate_beam_per_family":5,"global_candidate_beam":12,"deterministic_tie_break":["higher_worst_inner_fold_utility","higher_effective_independent_cluster_count","lower_complexity","lexical_cell_id"],"backward_information_flow":False}
    cost={"instrument":"Kraken linear PF derivatives","base_round_trip_bps_pre_funding":14,"stress_round_trip_bps_pre_funding":32,"primary_metric":"base net expectancy after 14 bps and signed exact-or-conservative-adverse funding","funding_partitions":["zero_boundary","exact","mixed","imputed"],"published_panel_coverage":"2024-01-01 through 2025-12-21 required-boundary panel; exact source concentrated from 2025-06-26","required_preoutcome_2023_remedy":"apply frozen model 0054af0e with unchanged parameters to every registered 2023 boundary using PIT inputs before opening outcomes; do not retrain or relabel missing as zero-boundary; event/fold fails closed at the registered coverage floor","sensitivities":["32bps_plus_conservative","exact_boundary_only_slice","zero_funding_diagnostic","severe_adverse_imputation"],"imputation_cannot_activate_gate_or_improve_selection":True,"funding_model_hash":"0054af0ee40740e39739bfade92f342867bb208a4fe7ed15b151a8a0a838d072","execution":"first authorized PF 5m open at or after decision/target, maximum 10 minute delay","non_overlap":"translation-symbol local using actual executable exit","same_bar_heroics":False}
    replay=float(measurement["benchmark"]["finalization_replay_wall_seconds"]); safety=1.5; workers=4; wall_cap=14400
    performance_cell_ceiling=int(wall_cap*workers/(replay*safety)); projected_wall=total*replay*safety/workers
    if total>performance_cell_ceiling: raise ValueError("semantic cell registry exceeds measured wall budget")
    per_cell_output_cap=1_048_576; shared_output_cap=2_147_483_648; projected_disk=shared_output_cap+total*per_cell_output_cap
    resources={"benchmark":measurement["benchmark"],"derivation":{"registered_cells_are_complete_mechanism_cartesian_products":True,"single_worker_full_replay_seconds_used_as_conservative_per_cell_cost":replay,"runtime_safety_factor":safety,"workers":workers,"performance_cell_ceiling":performance_cell_ceiling,"registered_cells":total,"projected_wall_seconds":projected_wall,"per_cell_output_cap_bytes":per_cell_output_cap,"shared_output_cap_bytes":shared_output_cap,"projected_disk_bytes":projected_disk},"maximum_cells_by_family":{"KDA02B":96,"KDA02C":48,"KDX01":84},"maximum_total_cells":228,"max_workers":workers,"wall_seconds":wall_cap,"max_disk_bytes":5368709120,"candidate_beam_per_family":5,"complexity_budget_max_interaction_depth":6,"projection":"mechanism-complete 96/48/84 registry is below the benchmark-derived cell ceiling; each cell and shared writer is separately hard-capped; stop before either four hours or 5 GiB"}
    manifest={"campaign_id":"kraken_derivatives_campaign_001_stage14_ready","version":"2.0","generated_at":FIXED,"mode":"future_phase2_5_request_only","economic_run_authorized_by_manifest":False,"ready_hypotheses":lanes,"C17_excluded":True,"repository_and_data_hashes":authorities,"search_spaces":spaces,"fold_schedule":fold_rows,"selection_algorithm":selection,"cost_and_execution":cost,"multiplicity":{"complete_registry_required":True,"counts_every_response_bin_feature_subset_model_rule_direction_horizon_context_cell":True,"inherit_parent_and_program_attempts":True,"unrecoverable_historical_attempt_count_never_zero":True},"resource_limits":resources,"phase_permissions":{"0":True,"1":True,"2":False,"3":False,"4":False,"5":False,"6":False,"7":False},"stop_conditions":{"family_only":["mechanically_unavailable","mechanism_underidentified","insufficient_incidence","family_specific_defect","no_development_candidate"],"global":["authority_mismatch","protected_exposure","shared_timestamp_defect","unsafe_git_or_storage","deterministic_replay_failure"]},"programme_exposure_class":"program_exposed_historical","independent_validation_claim":False,"human_approval_artifact_required":True}
    manifest["repository_and_data_hashes"]["repository_implementation_commit"]=implementation_commit
    jwrite(ARCHIVE/"CAMPAIGN_MANIFEST.json",manifest); manifest_hash=h(ARCHIVE/"CAMPAIGN_MANIFEST.json")
    registry={"generated_at":FIXED,"derivation":"mechanism-complete finite raw-unit/fold-local-rank axes, admitted only below RESOURCE_PROJECTION benchmark ceiling; no outcome selected an axis","search_spaces":spaces,"maximum_total_cells":228,"all_cells_registered":True,"search_spaces_sha256":hashlib.sha256(canonical(spaces)).hexdigest()}
    jwrite(ARCHIVE/"SEARCH_SPACE_REGISTRY.json",registry); jwrite(ARCHIVE/"FOLD_AND_EXPOSURE_MAP.json",{"rows":fold_rows,"programme_exposed_historical":True,"not_independent_validation":True})
    jwrite(ARCHIVE/"RESOURCE_PROJECTION.json",resources)
    packet={"packet_id":"kraken_derivatives_campaign_001_stage14_phase2_5_approval_request","status":"human_approval_required_not_authorized","ready_lanes":lanes,"C17_excluded":True,"phases_requested":[2,3,4,5],"campaign_manifest_sha256":manifest_hash,"authorities":authorities,"search_space_registry_sha256":h(ARCHIVE/"SEARCH_SPACE_REGISTRY.json"),"fold_and_exposure_map_sha256":h(ARCHIVE/"FOLD_AND_EXPOSURE_MAP.json"),"resource_projection_sha256":h(ARCHIVE/"RESOURCE_PROJECTION.json"),"cost_and_execution":cost,"selection_algorithm":selection,"programme_exposed_historical":True,"not_independent_validation":True,"economic_run_authorized":False,"protected_outcome_access":False,"unattended_scope":"one later hash-matched Phase 2-5 campaign without routine inter-fold approval; all stops remain binding","authorization_preconditions":["external human approval artifact explicitly names this final packet SHA-256","repository/data/code/cost/search/fold hashes match","independent Stage14 review accepted","outcome-free frozen-model boundary panel extension covers registered 2023-2025 events at the predeclared coverage floor","no protected or Capital.com access"]}
    jwrite(ARCHIVE/"FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json",packet); packet_hash=h(ARCHIVE/"FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json")
    (ARCHIVE/"FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md").write_text(f"""# Future Derivatives Campaign Approval Packet

Status: `human_approval_required_not_authorized`.

Ready lanes: {', '.join(lanes)}. C17 is excluded. Requested future authority is exactly Phases 2-5 for one hash-matched unattended campaign; no Phase 2 execution is authorized by this packet.

Campaign manifest SHA-256: `{manifest_hash}`<br>
Approval packet JSON SHA-256: `{packet_hash}`

The campaign is `programme_exposed_historical` and is not independent validation. An external human-approval artifact must explicitly name the approval packet JSON hash above. The 228-cell registry, 27 quarterly fold records, 14/32 bps plus partitioned-funding contract, candidate beams, resource caps, family/global stops, and deterministic tie-break are binding.
""",encoding="utf-8")
    registry_root=ROOT/"docs/agent/research_campaigns/kraken_derivatives_campaign_001_stage14_ready"
    registry_root.mkdir(parents=True,exist_ok=True)
    jwrite(registry_root/"CAMPAIGN_READINESS_REGISTRY.json",{"campaign_id":manifest["campaign_id"],"status":"phase_1_closed_human_approval_required","ready_lanes":lanes,"C17_status":"separate_blocked_route_not_in_campaign","campaign_manifest_path":str((ARCHIVE/"CAMPAIGN_MANIFEST.json").relative_to(ROOT)),"campaign_manifest_sha256":manifest_hash,"approval_packet_path":str((ARCHIVE/"FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json").relative_to(ROOT)),"approval_packet_sha256":packet_hash,"phase_2_5_authorized":False,"historical_terminal_decisions_changed":False})
    print(json.dumps({"campaign_manifest_sha256":manifest_hash,"approval_packet_sha256":packet_hash,"search_registry_sha256":h(ARCHIVE/"SEARCH_SPACE_REGISTRY.json"),"cells":total,"lanes":lanes},sort_keys=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--implementation-commit",default="UNCOMMITTED_STAGE14"); args=parser.parse_args(); build(args.implementation_commit)
