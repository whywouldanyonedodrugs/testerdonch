import inspect
import unittest

import pandas as pd

from tools import qlmg_signal_state_contract as state
from tools import run_kraken_lfbs_signal_state_repaired_lineage as repaired
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs


class LFBSSignalStateRepairedLineageTests(unittest.TestCase):
    @staticmethod
    def candidates() -> pd.DataFrame:
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        return pd.DataFrame([
            {"candidate_key": "a", "selected_key_policy_hash": "p", "symbol": "PF_XUSD", "decision_ts": start - pd.Timedelta(minutes=5), "entry_ts": start},
            {"candidate_key": "b", "selected_key_policy_hash": "p", "symbol": "PF_XUSD", "decision_ts": start + pd.Timedelta(hours=23, minutes=55), "entry_ts": start + pd.Timedelta(hours=24)},
            {"candidate_key": "c", "selected_key_policy_hash": "p", "symbol": "PF_XUSD", "decision_ts": start + pd.Timedelta(hours=72, minutes=55), "entry_ts": start + pd.Timedelta(hours=73)},
        ])

    @staticmethod
    def definition(definition_id: str, exit_policy: str) -> dict:
        return {"definition_id": definition_id, "selected_key_policy_hash": "p", "exit_policy": exit_policy}

    @staticmethod
    def execute(key, definition):
        holds = {
            "initial_stop": pd.Timedelta(hours=8),
            "close_back_above_failed_level": pd.Timedelta(hours=16),
            "fixed_72h_comparator": pd.Timedelta(hours=72),
            "atr_trailing_exit": pd.Timedelta(days=7),
        }
        return {"entry_ts": key["entry_ts"], "exit_ts": key["entry_ts"] + holds[definition["exit_policy"]], "exit_policy": definition["exit_policy"], "exit_reason": definition["exit_policy"]}, None

    def test_frozen_manifest_and_hashes_unchanged(self):
        current = lfbs.frozen_manifest()
        original = pd.read_csv(repaired.OLD_SCREEN / "manifest/failed_breakout_short_definitions.csv")
        pd.testing.assert_frame_equal(current.reset_index(drop=True), original.reset_index(drop=True), check_dtype=False)

    def test_raw_enumerator_has_no_parent_filter_or_hold_state(self):
        source = inspect.getsource(lfbs.enumerate_raw_signals)
        self.assertNotIn("parent_state(spec", source)
        self.assertNotIn("blocked_until", source)
        self.assertNotIn("Timedelta(hours=72)", source)

    def test_early_stop_allows_reentry_inside_72_hours(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(self.candidates().iloc[:2], self.definition("stop", "initial_stop"), self.execute)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_early_structural_close_allows_reentry(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(self.candidates().iloc[:2], self.definition("structure", "close_back_above_failed_level"), self.execute)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_fixed_72h_expiry_allows_later_signal(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(self.candidates().iloc[[0, 2]], self.definition("fixed", "fixed_72h_comparator"), self.execute)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_parent_projection_strict_is_raw_subset(self):
        now = pd.Timestamp("2025-01-01T00:00:00Z")
        raw = pd.DataFrame([
            {"raw_policy_hash": "r", "raw_signal_address_hash": "a", "setup_sequence_id": "a", "symbol": "PF_XUSD", "decision_ts": now, "entry_ts": now, "parent_feature_ts": now, "parent_available": True, "parent_state": "both_down", "reference_days": 60, "failure_bars": 3},
            {"raw_policy_hash": "r", "raw_signal_address_hash": "b", "setup_sequence_id": "b", "symbol": "PF_YUSD", "decision_ts": now, "entry_ts": now, "parent_feature_ts": now, "parent_available": True, "parent_state": "not_both_down", "reference_days": 60, "failure_bars": 3},
        ])
        frozen, _ = state.freeze_raw_signal_tape(raw)
        policies = [
            {"selected_key_policy_hash": "strict", "reference_days": 60, "failure_bars": 3, "parent_context": "fragile_countertrend_stress"},
            {"selected_key_policy_hash": "broad", "reference_days": 60, "failure_bars": 3, "parent_context": "all_regime_comparator"},
        ]
        projected, _ = state.project_parent_policies(frozen, policies, is_allowed=lambda source, policy: policy["parent_context"] == "all_regime_comparator" or source["parent_state"] == "both_down")
        strict = set(projected[projected.selected_key_policy_hash.eq("strict")].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq("broad")].raw_signal_address_hash)
        self.assertEqual(strict, {"a"})
        self.assertEqual(broad, {"a", "b"})

    def test_single_policy_downstream_replay_does_not_require_absent_comparator(self):
        source = inspect.getsource(repaired.run_period_engine)
        self.assertIn("not_applicable_single_policy_replay", source)
        self.assertIn("if strict_rows.empty or broad_rows.empty", source)

    def test_downstream_resume_requires_completed_screen(self):
        source = inspect.getsource(repaired.resume_downstream_lineage)
        self.assertIn('screen_summary.get("status") != "complete"', source)
        self.assertIn("run_presample(presample_root, resume=True)", source)

    def test_deterministic_replay_and_no_cross_definition_state(self):
        definitions = [self.definition("fixed", "fixed_72h_comparator"), self.definition("early", "initial_stop")]
        first = state.simulate_all_definitions(self.candidates().iloc[:2], definitions, self.execute)
        second = state.simulate_all_definitions(self.candidates().iloc[:2].sample(frac=1, random_state=5), definitions, self.execute)
        self.assertEqual(state.canonical_frame_hash(first[0], sort_fields=("definition_id", "candidate_key")), state.canonical_frame_hash(second[0], sort_fields=("definition_id", "candidate_key")))
        counts = first[0].groupby("definition_id").size().to_dict()
        self.assertEqual(counts, {"early": 2, "fixed": 1})

    def test_economic_address_is_definition_local_unique(self):
        row = {"symbol": "PF_XUSD", "decision_ts": "2025-01-01T00:00:00Z", "entry_ts": "2025-01-01T00:05:00Z", "initial_stop": 110.0, "risk_denominator": 10.0, "exit_policy": "fixed_72h_comparator", "maximum_exit_ts": "2025-01-04T00:05:00Z"}
        addresses = [repaired.economic_address({**row, "symbol": symbol}) for symbol in ("PF_XUSD", "PF_YUSD")]
        self.assertEqual(len(addresses), len(set(addresses)))

    def test_rankable_contract_assertion_precedes_cost_scoring(self):
        source = inspect.getsource(repaired.run_period_engine)
        self.assertLess(source.index("assert_rankable_signal_state_contract"), source.index("attach_costs(accepted"))
        self.assertLess(source.index('write_csv(root / "controls/control_key_manifest.csv"'), source.index("materialize_controls"))


if __name__ == "__main__":
    unittest.main()
