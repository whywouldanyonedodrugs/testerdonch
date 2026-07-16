import inspect
import unittest

import pandas as pd

from tools import qlmg_signal_state_contract as state
from tools import run_kraken_rfbs_signal_state_repaired as repaired
from tools import run_kraken_rfbs_signal_state_repaired_materialization as downstream
from tools import run_kraken_riskoff_failed_bounce_short_screen as base


class RFBSSignalStateRepairedTests(unittest.TestCase):
    @staticmethod
    def candidates(offsets=(0, 24, 73)):
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        return pd.DataFrame([
            {"candidate_key": f"k{number}", "selected_key_policy_hash": "p", "symbol": "PF_XUSD", "decision_ts": start + pd.Timedelta(hours=offset) - pd.Timedelta(minutes=5), "entry_ts": start + pd.Timedelta(hours=offset)}
            for number, offset in enumerate(offsets)
        ])

    @staticmethod
    def execute(key, definition):
        holds = {"stop": 8, "daily_ema10_close": 16, "fixed_72h": 72, "swing_high_trail_7d": 168}
        exit_ts = key["entry_ts"] + pd.Timedelta(hours=holds[definition["exit_policy"]])
        return {"entry_ts": key["entry_ts"], "exit_ts": exit_ts}, None

    def definition(self, definition_id, exit_policy):
        return {"definition_id": definition_id, "selected_key_policy_hash": "p", "exit_policy": exit_policy}

    def test_frozen_manifest_is_source_equal(self):
        current = base.frozen_manifest()
        old = pd.read_csv(repaired.SOURCE_ROOT / "manifest/riskoff_failed_bounce_definitions.csv")
        pd.testing.assert_frame_equal(current.reset_index(drop=True), old.reset_index(drop=True), check_dtype=False)

    def test_raw_enumerator_has_no_preblock_or_parent_filter(self):
        source = inspect.getsource(base.enumerate_raw_signals)
        self.assertNotIn("blocked_until", source)
        self.assertNotIn("Timedelta(days=7)", source)
        self.assertNotIn("parent_allowed", source)

    def test_early_stop_and_ema_allow_reentry(self):
        for exit_policy in ("stop", "daily_ema10_close"):
            accepted, skips, _ = state.simulate_definition_non_overlap(self.candidates((0, 24)), self.definition(exit_policy, exit_policy), self.execute)
            self.assertEqual(len(accepted), 2)
            self.assertTrue(skips.empty)

    def test_fixed_expiry_allows_later_signal(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(self.candidates((0, 73)), self.definition("fixed", "fixed_72h"), self.execute)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_higher_high_reset_and_confirmation_expiry_remain_tested(self):
        source = inspect.getsource(base.confirmation_sequences)
        self.assertIn("peak_index = cursor", source)
        self.assertIn("bars_after_peak > confirmation_bars", source)

    def test_parent_projection_nesting(self):
        now = pd.Timestamp("2025-01-01T00:00:00Z")
        raw = pd.DataFrame([
            {"raw_policy_hash": "r", "raw_signal_address_hash": "a", "symbol": "X", "decision_ts": now, "entry_ts": now, "parent_feature_ts": now, "parent_state": "both_down"},
            {"raw_policy_hash": "r", "raw_signal_address_hash": "b", "symbol": "Y", "decision_ts": now, "entry_ts": now, "parent_feature_ts": now, "parent_state": "mixed_at_least_one_down"},
        ])
        frozen, _ = state.freeze_raw_signal_tape(raw)
        policies = [{"selected_key_policy_hash": "strict", "parent_policy": "strict_both_down_stress"}, {"selected_key_policy_hash": "broad", "parent_policy": "broader_fragile_countertrend_stress"}]
        projected, _ = state.project_parent_policies(frozen, policies, is_allowed=lambda row, policy: base.parent_allowed(policy["parent_policy"], row["parent_state"]))
        strict = set(projected[projected.selected_key_policy_hash.eq("strict")].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq("broad")].raw_signal_address_hash)
        self.assertEqual(strict, {"a"})
        self.assertEqual(broad, {"a", "b"})

    def test_deterministic_and_definition_local(self):
        definitions = [self.definition("slow", "swing_high_trail_7d"), self.definition("fast", "stop")]
        first = state.simulate_all_definitions(self.candidates((0, 24)), definitions, self.execute)[0]
        second = state.simulate_all_definitions(self.candidates((0, 24)).sample(frac=1, random_state=4), definitions, self.execute)[0]
        self.assertEqual(state.canonical_frame_hash(first, sort_fields=("definition_id", "candidate_key")), state.canonical_frame_hash(second, sort_fields=("definition_id", "candidate_key")))
        self.assertEqual(first.groupby("definition_id").size().to_dict(), {"fast": 2, "slow": 1})

    def test_economic_addresses_definition_scoped_unique(self):
        row = {"symbol": "PF_XUSD", "decision_ts": "2025-01-01T00:00:00Z", "entry_ts": "2025-01-01T00:05:00Z", "initial_stop": 110.0, "risk_denominator": 10.0, "exit_policy": "fixed_72h", "maximum_exit_ts": "2025-01-04T00:05:00Z"}
        self.assertNotEqual(repaired.economic_address(row), repaired.economic_address({**row, "symbol": "PF_YUSD"}))

    def test_freezes_precede_outcomes_and_campaign_closes_downstream_only(self):
        source = inspect.getsource(repaired.run)
        self.assertLess(source.index("assert_rankable_signal_state_contract"), source.index("attach_costs(accepted"))
        self.assertLess(source.index('write_csv(root / "controls/control_key_manifest.csv"'), source.index("for control in control_keys.to_dict"))
        self.assertNotIn("update_campaign(root, final)", source)
        closure = inspect.getsource(downstream._close_campaign)
        self.assertIn('"unresolved_registry_count": 0', closure)
        self.assertIn("Cross-family repair campaign closure and continuity reconciliation only", closure)


if __name__ == "__main__":
    unittest.main()
