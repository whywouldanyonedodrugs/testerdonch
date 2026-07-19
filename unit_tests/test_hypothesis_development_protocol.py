import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "docs" / "agent"
STAGE11 = (
    AGENT
    / "task_archive"
    / "20260719_donch_bt_stage_11_kda03_basis_shock_20260719_v2"
)


class HypothesisDevelopmentProtocolTest(unittest.TestCase):
    def test_policy_route_vocabulary_is_unchanged(self):
        policy_path = AGENT / "RESEARCH_GATE_ROUTING_POLICY.json"
        self.assertEqual(
            hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa",
        )
        policy = json.loads(policy_path.read_text())
        self.assertEqual(policy["version"], "1.0")
        self.assertEqual(
            set(policy["routing_statuses"]),
            {
                "unconditional_control_candidate",
                "conditional_context_candidate_unvalidated",
                "convex_tail_candidate_unvalidated",
                "execution_sensitive_candidate",
                "narrow_sleeve_candidate",
                "sample_limited_prospective_candidate",
                "mechanically_unavailable",
                "translation_rejected",
                "blocked_by_data_or_authority",
            },
        )

    def test_protocol_has_forward_only_seven_phase_contract(self):
        protocol = json.loads((AGENT / "HYPOTHESIS_DEVELOPMENT_PROTOCOL.json").read_text())
        self.assertEqual(protocol["version"], "1.0")
        self.assertFalse(protocol["economic_run_authorized"])
        self.assertFalse(protocol["protected_outcome_access_authorized"])
        self.assertFalse(protocol["route_vocabulary_changed"])
        self.assertEqual([phase["phase"] for phase in protocol["phases"]], list(range(7)))

        admission = set(protocol["economic_run_admission"]["fields"])
        self.assertEqual(
            admission,
            {
                "measurement_semantics_valid",
                "event_frequency_consistent_with_mechanism",
                "raw_magnitude_economically_relevant",
                "actor_or_structural_mechanism_identified",
                "development_route_or_explicit_one_shot_reason",
                "horizon_justified",
                "component_controls_defined",
                "payoff_archetype_frozen",
            },
        )
        phase2 = protocol["phases"][2]
        self.assertTrue(phase2["separate_exact_human_approval_required"])
        self.assertTrue(phase2["all_explored_cells_must_be_registered"])
        self.assertFalse(phase2["confirmatory_evidence"])
        self.assertFalse(phase2["same_sample_rescue_allowed"])
        self.assertFalse(protocol["phases"][3]["backward_information_transfer_allowed"])
        self.assertEqual(
            protocol["phases"][4]["evaluation_rule"],
            "evaluate_frozen_translation_on_next_untouched_rankable_block",
        )
        self.assertFalse(protocol["phases"][5]["future_block_exposure_to_earlier_design_allowed"])

    def test_limitation_tags_are_descriptive_and_complete(self):
        registry = json.loads((AGENT / "EVIDENCE_LIMITATION_TAGS.json").read_text())
        self.assertTrue(registry["separate_from_research_route"])
        self.assertFalse(registry["changes_route_or_evidence_level"])
        self.assertEqual(registry["promotion_effect"], "none")
        required = {
            "small_sample",
            "few_independent_clusters",
            "high_variance",
            "wide_cluster_uncertainty",
            "threshold_sensitive",
            "cost_sensitive",
            "symbol_concentrated",
            "time_concentrated",
            "context_dependent",
            "measurement_underidentified",
            "proxy_side_or_reference",
            "mechanically_sparse",
            "lifecycle_capped",
            "funding_incomplete",
            "not_control_eligible",
        }
        self.assertTrue(required.issubset(registry["tags"]))

    def test_stage11_authority_and_primary_routes_remain_exact(self):
        manifest_path = STAGE11 / "ARTIFACT_MANIFEST.json"
        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "07a9a18f75320d44703b42b3ed0d0a03fe143bba89c24a875ec5e4ac6a9b2856",
        )
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(
            manifest["manifest_content_hash"],
            "0bc85e5056db8ddb38e7977761e2fe657647cf2cb632e0447153bf64e1cd3af7",
        )
        self.assertEqual(
            manifest["contract_hash"],
            "5f1abdc7e21ab9e3a6851b21930a7af745cbb865be334f8be45c9a561d2411e7",
        )
        self.assertEqual(manifest["terminal_status"], "KDA03_level3_routes_assigned")

        with (STAGE11 / "run_outputs" / "KDA03_LEVEL3_DEFINITION_DECISIONS.csv").open() as fh:
            primary = [row for row in csv.DictReader(fh) if row["attempt"] == "primary"]
        self.assertEqual(len(primary), 12)
        routes = [row["policy_route"] for row in primary]
        self.assertEqual(routes.count("translation_rejected"), 11)
        self.assertEqual(routes.count("sample_limited_prospective_candidate"), 1)
        retained = next(row for row in primary if row["policy_route"] != "translation_rejected")
        self.assertEqual(
            retained["definition_id"],
            "kda03_v1_primary_negative_completed_basis_impulse_rejection_timeout_6h",
        )
        self.assertEqual(retained["control_eligible"], "False")

    def test_closure_documents_exact_bounded_metrics_and_no_promotion(self):
        lessons = (ROOT / "KDA03_STAGE11_LESSONS.md").read_text()
        for value in ("+9.1570", "+2.9323", "-8.2953", "-8.8430"):
            self.assertIn(value, lessons)
        for marker in (
            "not_control_eligible",
            "unvalidated",
            "not live-ready",
            "no same-sample threshold rescue",
        ):
            self.assertIn(marker.lower(), lessons.lower())

    def test_prior_terminal_decisions_remain_visible_and_immutable(self):
        continuity = (AGENT / "CURRENT_CONTINUITY.md").read_text()
        for decision in (
            "level3_no_primary_pass_stop",
            "C03_PIT_authority_unavailable",
            "C16_flow_authority_unavailable",
            "KDA01_level3_repaired_no_primary_pass_stop",
            "KDA02_level3_no_primary_pass_stop",
            "KDA03_level3_routes_assigned",
        ):
            self.assertIn(decision, continuity)
        self.assertIn("These exact run decisions remain immutable.", continuity)


if __name__ == "__main__":
    unittest.main()
