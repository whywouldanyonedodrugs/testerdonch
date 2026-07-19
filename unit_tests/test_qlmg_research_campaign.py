import json
import ast
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.qlmg_research_campaign import (
    CampaignContractError, apply_stop, build_dag, canonical_bytes, commit_state, complete_node,
    enforce_phase_permission,
    initial_state, load_or_initialize, reconcile_artifacts, sha256_bytes, sha256_file,
    heartbeat, validate_candidate_beam, validate_explored_cells, validate_freeze,
    validate_manifest, validate_resource_usage,
)


STAGE14 = Path(__file__).resolve().parents[1] / "docs/agent/task_archive/20260719_donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1"
STAGE15 = Path(__file__).resolve().parents[1] / "docs/agent/task_archive/20260720_donch_bt_stage_15_unattended_derivatives_campaign_20260720_v1"


def stage15_binding():
    manifest_path = STAGE14 / "CAMPAIGN_MANIFEST.json"
    packet_path = STAGE14 / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json"
    manifest_raw = manifest_path.read_bytes(); packet_raw = packet_path.read_bytes()
    current_manifest = json.loads(manifest_raw); packet = json.loads(packet_raw)
    approval_raw = (STAGE15 / "HUMAN_APPROVAL.json").read_bytes()
    approval = json.loads(approval_raw)
    minimum = approval["supplemental_binding_constraints"]["funding_boundary_coverage"]
    constraints = {
        "funding_coverage": {
            "campaign_weighted": 0.97,
            "by_hypothesis_fold": {fold["fold_id"]: 0.96 for fold in current_manifest["fold_schedule"]},
            "missing_boundary_policy": minimum["missing_boundary_policy"],
            "selection_use": minimum["selection_use"],
        },
        "telegram": {
            "secure_configuration_present": True,
            "dry_run_delivered": True,
            "heartbeat_delivered": True,
            "stop_alert_delivered": True,
            "secret_values_logged_or_archived": False,
        },
    }
    return current_manifest, packet, approval, approval_raw, manifest_raw, packet_raw, constraints


def manifest():
    return {
        "campaign_id": "test", "repository_and_data_hashes": {},
        "hypotheses": [{"hypothesis_id": "H1", "search_space_id": "S1",
                        "programme_exposure_class": "program_exposed_historical"}],
        "fold_schedule": [{"fold_id": "F1", "hypothesis_id": "H1",
                           "development_start": "2023-01-01T00:00:00Z",
                           "development_end": "2023-06-30T00:00:00Z",
                           "embargo_end": "2023-07-01T00:00:00Z",
                           "evaluation_start": "2023-07-01T00:00:00Z",
                           "evaluation_end": "2023-10-01T00:00:00Z"}],
        "search_spaces": [{"search_space_id": "S1", "registered_cell_ids": ["c1", "c2"]}],
        "selection_algorithm": {}, "candidate_beam": {"max_retained_per_hypothesis": 1},
        "cost_and_execution": {}, "multiplicity": {},
        "phase_permissions": {str(i): i < 2 for i in range(8)},
        "resource_limits": {"wall_seconds": 10}, "stop_conditions": {},
        "review_requirements": {}, "archive_and_handoff": {},
        "economic_run_authorized_by_manifest": False,
    }


class CampaignTests(unittest.TestCase):
    def test_manifest_and_dag(self):
        validate_manifest(manifest())
        dag = build_dag(manifest())
        self.assertEqual(len(dag), 10)
        self.assertEqual([node["phase"] for node in dag[:3]], [0, 1, 2])
        self.assertEqual(dag[2]["fold_id"], "F1")

    def test_missing_field_and_protected_fold_fail_closed(self):
        bad = manifest(); del bad["multiplicity"]
        with self.assertRaises(CampaignContractError): validate_manifest(bad)
        bad = manifest(); bad["fold_schedule"][0]["evaluation_end"] = "2026-04-01T00:00:00Z"
        with self.assertRaises(CampaignContractError): validate_manifest(bad)
        bad = manifest(); bad["fold_schedule"][0]["evaluation_end"] = "2023-06-01T00:00:00Z"
        with self.assertRaises(CampaignContractError): validate_manifest(bad)

    def test_only_registered_cells_and_beam(self):
        valid = [{"hypothesis_id": "H1", "fold_id": "F1", "search_space_id": "S1", "cell_id": "c1"}]
        validate_explored_cells(manifest(), valid)
        invalid = [dict(valid[0], cell_id="hidden")]
        with self.assertRaises(CampaignContractError): validate_explored_cells(manifest(), invalid)
        invalid = [dict(valid[0], hypothesis_id="H2")]
        with self.assertRaises(CampaignContractError): validate_explored_cells(manifest(), invalid)
        with self.assertRaises(CampaignContractError): validate_candidate_beam(manifest(), [valid[0], dict(valid[0], cell_id="c2")])

    def test_outer_fold_cannot_flow_backward(self):
        fold = manifest()["fold_schedule"][0]
        with self.assertRaises(CampaignContractError):
            validate_freeze({"frozen_at": "2023-06-30T00:00:00Z", "source_fold_ids": ["F1"]}, fold, [fold])

    def test_phase_permission_and_stop_isolation(self):
        enforce_phase_permission(manifest(), 1)
        with self.assertRaises(CampaignContractError): enforce_phase_permission(manifest(), 2)
        authorized_in_manifest = manifest(); authorized_in_manifest["phase_permissions"]["2"] = True
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(authorized_in_manifest, 2, hypothesis_id="H1")
        packet = {"campaign_manifest_sha256":sha256_bytes(canonical_bytes(authorized_in_manifest)),
                  "candidate_list":["H1"], "phase_permissions_requested":{"2":True}}
        approval = {"approval_id":"human-1", "status":"approved", "human_authorized":True,
                    "authorized_by":"named-human", "authorized_at":"2026-07-19T00:00:00Z",
                    "campaign_id":"test", "campaign_manifest_sha256":packet["campaign_manifest_sha256"],
                    "approval_packet_sha256":sha256_bytes(canonical_bytes(packet)),
                    "approved_phases":[2], "approved_hypotheses":["H1"],
                    "repository_and_data_hashes":authorized_in_manifest["repository_and_data_hashes"],
                    "cost_and_execution_sha256":sha256_bytes(canonical_bytes(authorized_in_manifest["cost_and_execution"]))}
        enforce_phase_permission(authorized_in_manifest, 2, state=initial_state(authorized_in_manifest), hypothesis_id="H1", approval=approval,
                                 approval_packet=packet)
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(manifest(), 1, state={"global_stop":{"reason":"shared_replay_failure"}})
        state = initial_state(manifest())
        family = apply_stop(state, "no_development_candidate", "H1")
        self.assertIsNone(family["global_stop"])
        global_stop = apply_stop(state, "shared_replay_failure")
        self.assertEqual(global_stop["family_stops"], {})

    def test_idempotent_resume_and_transaction_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            first = load_or_initialize(path, manifest())
            self.assertEqual(first, load_or_initialize(path, manifest()))
            forged = dict(first, generation=1, completed_nodes=["H1:phase_1"])
            with self.assertRaises(CampaignContractError): commit_state(path, forged, 0, manifest())
            updated = complete_node(manifest(), first, "H1:phase_0")
            commit_state(path, updated, 0, manifest())
            with self.assertRaises(CampaignContractError): commit_state(path, dict(updated, generation=1), 0, manifest())

    def test_dag_transition_requires_dependencies_and_respects_stops(self):
        state = initial_state(manifest())
        with self.assertRaises(CampaignContractError): complete_node(manifest(), state, "H1:phase_1")
        state = complete_node(manifest(), state, "H1:phase_0")
        state = complete_node(manifest(), state, "H1:phase_1")
        stopped = apply_stop(state, "no_development_candidate", "H1")
        with self.assertRaises(CampaignContractError): complete_node(manifest(), stopped, "F1:phase_2")

    def test_artifact_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "a"; path.write_text("x")
            reconcile_artifacts([{"path": "a", "sha256": sha256_file(path)}], root)
            with self.assertRaises(CampaignContractError):
                reconcile_artifacts([{"path": "a", "sha256": "0" * 64}], root)

    def test_resource_limit_and_deterministic_heartbeat(self):
        validate_resource_usage(manifest(), {"wall_seconds": 10})
        with self.assertRaises(CampaignContractError):
            validate_resource_usage(manifest(), {"wall_seconds": 11})
        self.assertEqual(heartbeat("test", 2, "2026-07-19T00:00:00Z")["generation"], 2)

    def test_builder_is_deterministic_and_outcome_reader_free(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "tools/build_research_campaign_readiness.py").read_text()
        imports = {alias.name for node in ast.walk(ast.parse(source))
                   if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        self.assertTrue(imports.isdisjoint({"pandas", "pyarrow", "polars", "numpy"}))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for output in (first, second):
                subprocess.run(["python3", "tools/build_research_campaign_readiness.py", "--output", output],
                               cwd=root, check=True)
            one = {p.name: p.read_bytes() for p in Path(first).iterdir()}
            two = {p.name: p.read_bytes() for p in Path(second).iterdir()}
            self.assertEqual(one, two)

    def test_stage14_external_approval_overrides_false_readiness_only_after_exact_validation(self):
        current_manifest, packet, approval, approval_raw, manifest_raw, packet_raw, constraints = stage15_binding()
        hypothesis = approval["approved_hypotheses"][0]
        state = initial_state(current_manifest)
        enforce_phase_permission(
            current_manifest, 2, state=state, hypothesis_id=hypothesis,
            approval=approval, approval_packet=packet,
            approval_raw_bytes=approval_raw,
            expected_approval_sha256="c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b",
            manifest_raw_bytes=manifest_raw, approval_packet_raw_bytes=packet_raw,
            launch_constraints=constraints,
        )
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(current_manifest, 2, state=state, hypothesis_id=hypothesis)

    def test_stage14_binding_rejects_raw_or_canonical_tampering_and_alias_substitution(self):
        current_manifest, packet, approval, approval_raw, manifest_raw, packet_raw, constraints = stage15_binding()
        hypothesis = approval["approved_hypotheses"][0]; state = initial_state(current_manifest)
        common = dict(state=state, hypothesis_id=hypothesis, approval=approval,
                      approval_packet=packet, approval_raw_bytes=approval_raw,
                      expected_approval_sha256="c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b",
                      manifest_raw_bytes=manifest_raw,
                      approval_packet_raw_bytes=packet_raw, launch_constraints=constraints)
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(current_manifest, 2, **dict(common, approval_packet_raw_bytes=packet_raw + b" "))
        changed_manifest = json.loads(json.dumps(current_manifest)); changed_manifest["campaign_id"] += "x"
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(changed_manifest, 2, **common)
        aliases = json.loads(json.dumps(packet))
        aliases["phase_permissions_requested"] = {str(value): True for value in aliases.pop("phases_requested")}
        aliases["candidate_list"] = aliases.pop("ready_lanes")
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(current_manifest, 2, **dict(common, approval_packet=aliases))

    def test_stage14_binding_rejects_phase_hypothesis_state_telegram_and_coverage_bypass(self):
        current_manifest, packet, approval, approval_raw, manifest_raw, packet_raw, constraints = stage15_binding()
        hypothesis = approval["approved_hypotheses"][0]; state = initial_state(current_manifest)
        common = dict(state=state, hypothesis_id=hypothesis, approval=approval,
                      approval_packet=packet, approval_raw_bytes=approval_raw,
                      expected_approval_sha256="c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b",
                      manifest_raw_bytes=manifest_raw,
                      approval_packet_raw_bytes=packet_raw, launch_constraints=constraints)
        for phase in (6, 7):
            with self.assertRaises(CampaignContractError):
                enforce_phase_permission(current_manifest, phase, **common)
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(current_manifest, 2, **dict(common, hypothesis_id="C17"))
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(current_manifest, 2, **dict(common, state=None))
        missing_telegram = json.loads(json.dumps(constraints)); missing_telegram["telegram"]["dry_run_delivered"] = False
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(current_manifest, 2, **dict(common, launch_constraints=missing_telegram))
        low_coverage = json.loads(json.dumps(constraints))
        low_coverage["funding_coverage"]["campaign_weighted"] = 0.949
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(current_manifest, 2, **dict(common, launch_constraints=low_coverage))
        low_fold = json.loads(json.dumps(constraints))
        first_fold = next(iter(low_fold["funding_coverage"]["by_hypothesis_fold"]))
        low_fold["funding_coverage"]["by_hypothesis_fold"][first_fold] = 0.899
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(current_manifest, 2, **dict(common, launch_constraints=low_fold))
        for invalid in ("NaN", "Infinity", "-Infinity", -0.01, 1.01):
            invalid_campaign = json.loads(json.dumps(constraints))
            invalid_campaign["funding_coverage"]["campaign_weighted"] = invalid
            with self.assertRaisesRegex(CampaignContractError, "finite in"):
                enforce_phase_permission(current_manifest, 2, **dict(common, launch_constraints=invalid_campaign))
            invalid_fold = json.loads(json.dumps(constraints))
            invalid_fold["funding_coverage"]["by_hypothesis_fold"][first_fold] = invalid
            with self.assertRaisesRegex(CampaignContractError, "per-hypothesis-fold"):
                enforce_phase_permission(current_manifest, 2, **dict(common, launch_constraints=invalid_fold))

    def test_false_readiness_cannot_be_overridden_by_legacy_or_rewritten_approval(self):
        old_manifest = manifest()
        packet = {"campaign_manifest_sha256": sha256_bytes(canonical_bytes(old_manifest)),
                  "candidate_list": ["H1"], "phase_permissions_requested": {"2": True}}
        approval = {"approval_id": "human-1", "status": "approved", "human_authorized": True,
                    "authorized_by": "named-human", "authorized_at": "2026-07-19T00:00:00Z",
                    "campaign_id": "test", "campaign_manifest_sha256": packet["campaign_manifest_sha256"],
                    "approval_packet_sha256": sha256_bytes(canonical_bytes(packet)),
                    "approved_phases": [2], "approved_hypotheses": ["H1"],
                    "repository_and_data_hashes": {},
                    "cost_and_execution_sha256": sha256_bytes(canonical_bytes({}))}
        with self.assertRaises(CampaignContractError):
            enforce_phase_permission(old_manifest, 2, state=initial_state(old_manifest),
                                     hypothesis_id="H1", approval=approval, approval_packet=packet)
        current_manifest, packet, approval, approval_raw, manifest_raw, packet_raw, constraints = stage15_binding()
        rewritten = json.loads(json.dumps(approval)); rewritten["approved_phases"].append(6)
        with self.assertRaisesRegex(CampaignContractError, "parsed human approval differs"):
            enforce_phase_permission(current_manifest, 6, state=initial_state(current_manifest),
                                     hypothesis_id=approval["approved_hypotheses"][0], approval=rewritten,
                                     approval_packet=packet, approval_raw_bytes=approval_raw,
                                     expected_approval_sha256="c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b",
                                     manifest_raw_bytes=manifest_raw, approval_packet_raw_bytes=packet_raw,
                                     launch_constraints=constraints)
        widened_packet = json.loads(json.dumps(packet)); widened_packet["phases_requested"].append(6)
        widened_packet_raw = canonical_bytes(widened_packet)
        coordinated = json.loads(json.dumps(approval)); coordinated["approved_phases"].append(6)
        coordinated["approval_packet_sha256"] = sha256_bytes(widened_packet_raw)
        coordinated["approval_packet_canonical_sha256"] = sha256_bytes(canonical_bytes(widened_packet))
        coordinated_raw = canonical_bytes(coordinated)
        with self.assertRaisesRegex(CampaignContractError, "file-byte hash mismatch"):
            enforce_phase_permission(current_manifest, 6, state=initial_state(current_manifest),
                                     hypothesis_id=approval["approved_hypotheses"][0], approval=coordinated,
                                     approval_packet=widened_packet, approval_raw_bytes=coordinated_raw,
                                     expected_approval_sha256="c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b",
                                     manifest_raw_bytes=manifest_raw, approval_packet_raw_bytes=widened_packet_raw,
                                     launch_constraints=constraints)


if __name__ == "__main__":
    unittest.main()
