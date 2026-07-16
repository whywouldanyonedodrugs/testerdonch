import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from tools import qlmg_evidence_contracts as evidence
from tools import qlmg_signal_state_contract as state
from tools import run_kraken_cross_family_signal_state_repair_campaign as campaign


class SignalStateContractTests(unittest.TestCase):
    def test_contract_version_is_shared_with_evidence_validator(self):
        self.assertEqual(state.SIGNAL_STATE_CONTRACT_VERSION, evidence.SIGNAL_STATE_CONTRACT_VERSION)

    @staticmethod
    def raw_tape() -> pd.DataFrame:
        base = pd.Timestamp("2025-01-01T00:00:00Z")
        rows = [
            {
                "raw_policy_hash": "raw",
                "raw_signal_address_hash": "raw-a",
                "setup_sequence_id": "setup-a",
                "symbol": "PF_XUSD",
                "decision_ts": base,
                "parent_feature_ts": base - pd.Timedelta(hours=4),
                "entry_ts": base + pd.Timedelta(minutes=5),
                "parent_state": "both_down",
            },
            {
                "raw_policy_hash": "raw",
                "raw_signal_address_hash": "raw-b",
                "setup_sequence_id": "setup-b",
                "symbol": "PF_XUSD",
                "decision_ts": base + pd.Timedelta(days=1),
                "parent_feature_ts": base + pd.Timedelta(hours=20),
                "entry_ts": base + pd.Timedelta(days=1, minutes=5),
                "parent_state": "mixed",
            },
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def candidates(offset_hours=(0, 24, 73), policy="policy") -> pd.DataFrame:
        base = pd.Timestamp("2025-01-01T00:00:00Z")
        return pd.DataFrame([
            {
                "candidate_key": f"candidate-{number}",
                "selected_key_policy_hash": policy,
                "symbol": "PF_XUSD",
                "decision_ts": base + pd.Timedelta(hours=offset) - pd.Timedelta(minutes=5),
                "entry_ts": base + pd.Timedelta(hours=offset),
            }
            for number, offset in enumerate(offset_hours)
        ])

    @staticmethod
    def definition(definition_id="definition", exit_policy="early_stop", policy="policy"):
        return {
            "definition_id": definition_id,
            "selected_key_policy_hash": policy,
            "exit_policy": exit_policy,
        }

    @staticmethod
    def executor(key, definition):
        holds = {
            "early_stop": pd.Timedelta(hours=12),
            "early_structural": pd.Timedelta(hours=18),
            "fixed_72h": pd.Timedelta(hours=72),
            "seven_day": pd.Timedelta(days=7),
        }
        hold = holds[definition["exit_policy"]]
        return {
            "entry_ts": key["entry_ts"],
            "exit_ts": key["entry_ts"] + hold,
            "exit_reason": definition["exit_policy"],
        }, None

    def test_early_stop_allows_valid_reentry(self):
        accepted, skips, exclusions = state.simulate_definition_non_overlap(
            self.candidates((0, 24)), self.definition(exit_policy="early_stop"), self.executor
        )
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)
        self.assertTrue(exclusions.empty)

    def test_early_structural_exit_allows_valid_reentry(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(
            self.candidates((0, 24)), self.definition(exit_policy="early_structural"), self.executor
        )
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_fixed_hold_expiry_allows_later_signal(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(
            self.candidates((0, 73)), self.definition(exit_policy="fixed_72h"), self.executor
        )
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_parent_policy_projection_is_nested_and_pit(self):
        frozen, _ = state.freeze_raw_signal_tape(self.raw_tape())
        policies = [
            {"selected_key_policy_hash": "broad", "parent_policy": "all"},
            {"selected_key_policy_hash": "strict", "parent_policy": "strict"},
        ]
        projected, projection_hash = state.project_parent_policies(
            frozen,
            policies,
            is_allowed=lambda raw, policy: policy["parent_policy"] == "all" or raw["parent_state"] == "both_down",
        )
        strict = set(projected[projected.selected_key_policy_hash.eq("strict")].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq("broad")].raw_signal_address_hash)
        self.assertEqual(strict, {"raw-a"})
        self.assertEqual(broad, {"raw-a", "raw-b"})
        self.assertTrue(strict < broad)
        self.assertEqual(len(projection_hash), 64)

    def test_parent_projection_fails_on_future_feature(self):
        raw = self.raw_tape()
        raw.loc[0, "parent_feature_ts"] = raw.loc[0, "decision_ts"] + pd.Timedelta(minutes=1)
        frozen, _ = state.freeze_raw_signal_tape(raw)
        with self.assertRaisesRegex(state.SignalStateContractError, "parent_projection_pit_violations"):
            state.project_parent_policies(
                frozen,
                [{"selected_key_policy_hash": "broad"}],
                is_allowed=lambda raw, policy: True,
            )

    def test_repeated_unresolved_setup_is_suppressed_without_hold_block(self):
        raw = self.raw_tape()
        duplicate = raw.iloc[0].copy()
        duplicate["raw_signal_address_hash"] = "raw-a-repeat"
        duplicate["decision_ts"] += pd.Timedelta(hours=4)
        duplicate["entry_ts"] += pd.Timedelta(hours=4)
        combined = pd.concat([raw, duplicate.to_frame().T], ignore_index=True)
        deduplicated = state.suppress_repeated_unresolved_setups(combined)
        self.assertEqual(set(deduplicated.setup_sequence_id), {"setup-a", "setup-b"})
        self.assertIn("raw-b", set(deduplicated.raw_signal_address_hash))

    def test_deterministic_rerun_parity(self):
        candidates = self.candidates((0, 24, 73))
        definition = self.definition(exit_policy="early_stop")
        first = state.simulate_definition_non_overlap(candidates, definition, self.executor)
        second = state.simulate_definition_non_overlap(candidates.sample(frac=1, random_state=7), definition, self.executor)
        for left, right, sort_fields in zip(first, second, (("definition_id", "candidate_key"), ("definition_id", "candidate_key"), ("definition_id", "candidate_key"))):
            if left.empty and right.empty:
                continue
            self.assertEqual(
                state.canonical_frame_hash(left, sort_fields=sort_fields),
                state.canonical_frame_hash(right, sort_fields=sort_fields),
            )

    def test_no_cross_definition_state_sharing(self):
        candidates = self.candidates((0, 24))
        definitions = [
            self.definition("long_definition", "seven_day"),
            self.definition("short_definition", "early_stop"),
        ]
        accepted, skips, _ = state.simulate_all_definitions(candidates, definitions, self.executor)
        counts = accepted.groupby("definition_id").size().to_dict()
        self.assertEqual(counts, {"long_definition": 1, "short_definition": 2})
        self.assertEqual(skips.groupby("definition_id").size().to_dict(), {"long_definition": 1})

    def test_complete_skip_ledger_contains_actual_prior_exit(self):
        _, skips, _ = state.simulate_definition_non_overlap(
            self.candidates((0, 24)), self.definition(exit_policy="seven_day"), self.executor
        )
        self.assertEqual(len(skips), 1)
        row = skips.iloc[0]
        self.assertEqual(row.skip_reason, "same_symbol_same_definition_position_actually_open")
        self.assertEqual(row.prior_actual_exit_ts, pd.Timestamp("2025-01-08T00:00:00Z"))
        self.assertTrue(row.prior_trade_id)

    def test_rankable_manifest_passes_and_missing_hash_fails_closed(self):
        frozen, _ = state.freeze_raw_signal_tape(self.raw_tape())
        projected, _ = state.project_parent_policies(
            frozen,
            [{"selected_key_policy_hash": "policy"}],
            is_allowed=lambda raw, policy: True,
        )
        accepted, skips, exclusions = state.simulate_definition_non_overlap(
            projected,
            self.definition(exit_policy="early_stop"),
            self.executor,
        )
        manifest = state.build_rankable_contract_manifest(
            raw_signals=frozen,
            projected=projected,
            accepted=accepted,
            skips=skips,
            exclusions=exclusions,
            eligible_definition_rows=len(projected),
        )
        self.assertTrue(evidence.validate_rankable_signal_state_contract(manifest).passed)
        del manifest["raw_signal_hash"]
        result = evidence.validate_rankable_signal_state_contract(manifest)
        self.assertFalse(result.passed)
        self.assertIn("raw_signal_hash", ";".join(result.violations))

    def test_rankable_manifest_reconciliation_fails_closed(self):
        manifest = {
            "signal_state_contract_version": evidence.SIGNAL_STATE_CONTRACT_VERSION,
            "raw_signal_hash": "a" * 64,
            "projection_hash": "b" * 64,
            "accepted_trade_hash": "c" * 64,
            "raw_signal_count": 2,
            "eligible_definition_rows": 3,
            "accepted_trade_count": 1,
            "non_overlap_skip_count": 1,
            "outcome_exclusion_count": 0,
            "raw_tape_frozen_before_outcomes": True,
            "projection_frozen_before_outcomes": True,
            "non_overlap_reconciled": True,
            "no_mutable_state_shared_across_definitions": True,
        }
        result = evidence.validate_rankable_signal_state_contract(manifest)
        self.assertFalse(result.passed)
        self.assertIn("non_overlap_reconciliation_mismatch", result.violations)

    def test_repository_scan_finds_only_known_direct_active_preblocks(self):
        scan = pd.DataFrame(campaign.repository_scan())
        direct = set(scan[scan.classification.eq("directly_affected_pre_outcome_signal_suppression")].runner)
        self.assertEqual(direct, set(campaign.DIRECT_RUNNERS))
        self.assertEqual(int(scan.pre_outcome_signal_suppression.sum()), 4)

    def test_campaign_gate_stays_closed_with_six_unresolved_roots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            decision = campaign.run(root)
            self.assertFalse(decision["new_family_launch_allowed"])
            self.assertEqual(decision["directly_affected_roots"], 3)
            self.assertEqual(decision["downstream_affected_roots"], 3)
            self.assertEqual(decision["unresolved_registry_count"], 6)
            gate = pd.read_json(root / "campaign/new_family_launch_gate.json", typ="series")
            self.assertFalse(bool(gate["new_family_launch_allowed"]))
            self.assertEqual(gate["next_prompt_target"], "LFBS only")


if __name__ == "__main__":
    unittest.main()
