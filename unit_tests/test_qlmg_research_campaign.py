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


if __name__ == "__main__":
    unittest.main()
