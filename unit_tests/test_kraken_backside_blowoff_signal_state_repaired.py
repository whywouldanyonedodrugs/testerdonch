import inspect
import unittest

import pandas as pd

from tools import qlmg_signal_state_contract as state
from tools import run_kraken_backside_blowoff_short_screen as base
from tools import run_kraken_backside_blowoff_signal_state_repaired as repaired


class BacksideBlowoffSignalStateRepairedTests(unittest.TestCase):
    @staticmethod
    def candidates(offsets=(0, 24, 73)):
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        return pd.DataFrame([
            {"candidate_key": f"k{number}", "selected_key_policy_hash": "p", "symbol": "PF_XUSD", "decision_ts": start + pd.Timedelta(hours=offset)-pd.Timedelta(minutes=5), "entry_ts": start+pd.Timedelta(hours=offset)}
            for number, offset in enumerate(offsets)
        ])

    @staticmethod
    def definition(definition_id, exit_policy):
        return {"definition_id": definition_id, "selected_key_policy_hash": "p", "exit_policy": exit_policy}

    @staticmethod
    def execute(key, definition):
        holds = {"stop": 8, "daily_ema10_close": 16, "fixed_72h": 72, "swing_high_trail_7d": 168}
        exit_ts = key["entry_ts"] + pd.Timedelta(hours=holds[definition["exit_policy"]])
        return {"entry_ts": key["entry_ts"], "exit_ts": exit_ts, "exit_reason": definition["exit_policy"]}, None

    def test_frozen_manifest_is_exactly_24_and_source_equal(self):
        current = base.frozen_manifest()
        old = pd.read_csv(repaired.SOURCE_ROOT / "manifest/backside_blowoff_short_definitions.csv")
        pd.testing.assert_frame_equal(current.reset_index(drop=True), old.reset_index(drop=True), check_dtype=False)

    def test_raw_enumerator_has_no_position_or_maximum_hold_preblock(self):
        source = inspect.getsource(base.enumerate_raw_signals)
        self.assertNotIn("blocked_until", source)
        self.assertNotIn("Timedelta(days=7)", source)
        self.assertNotIn("parent_context", source)

    def test_early_stop_allows_reentry_inside_seven_days(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(self.candidates((0, 24)), self.definition("stop", "stop"), self.execute)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_early_ema_exit_allows_reentry(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(self.candidates((0, 24)), self.definition("ema", "daily_ema10_close"), self.execute)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_fixed_72h_expiry_allows_later_signal(self):
        accepted, skips, _ = state.simulate_definition_non_overlap(self.candidates((0, 73)), self.definition("fixed", "fixed_72h"), self.execute)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(skips.empty)

    def test_confirmation_expiry_and_higher_high_reset(self):
        frame = pd.DataFrame({
            "decision_ts": pd.date_range("2025-01-01", periods=5, freq="4h", tz="UTC"),
            "open": [100.0]*5, "high": [110.0, 112.0, 108.0, 107.0, 106.0], "low": [90.0, 91.0, 90.0, 89.0, 88.0],
            "close": [108.0, 110.0, 100.0, 99.0, 88.0], "extension_40_5": [True, False, False, False, False], "extension_70_10": [False]*5,
        })
        known = pd.date_range("2025-01-01T00:05:00Z", periods=193, freq="5min")
        work = pd.DataFrame({"known_ts": known, "volume": 1.0, "vwap_num": 100.0})
        self.assertEqual(base.confirmation_sequences(frame, work, "rise_40pct_5d_3atr", 1), [])
        found = base.confirmation_sequences(frame, work, "rise_40pct_5d_3atr", 3)
        self.assertEqual(found[0]["peak_index"], 1)
        self.assertEqual(found[0]["decision_index"], 4)

    def test_strict_parent_projection_is_raw_subset(self):
        now = pd.Timestamp("2025-01-01T00:00:00Z")
        raw = pd.DataFrame([
            {"raw_policy_hash": "r", "raw_signal_address_hash": "a", "setup_sequence_id": "a", "symbol": "PF_XUSD", "decision_ts": now, "entry_ts": now, "parent_feature_ts": now, "parent_state": "both_down"},
            {"raw_policy_hash": "r", "raw_signal_address_hash": "b", "setup_sequence_id": "b", "symbol": "PF_YUSD", "decision_ts": now, "entry_ts": now, "parent_feature_ts": now, "parent_state": "mixed"},
        ])
        frozen, _ = state.freeze_raw_signal_tape(raw)
        projected, _ = state.project_parent_policies(frozen, [{"selected_key_policy_hash": "strict", "parent_context": "fragile_countertrend_stress"}, {"selected_key_policy_hash": "broad", "parent_context": "all_regime_comparator"}], is_allowed=lambda source, policy: policy["parent_context"] == "all_regime_comparator" or source["parent_state"] == "both_down")
        strict = set(projected[projected.selected_key_policy_hash.eq("strict")].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq("broad")].raw_signal_address_hash)
        self.assertEqual(strict, {"a"})
        self.assertEqual(broad, {"a", "b"})

    def test_deterministic_replay_and_no_cross_definition_state(self):
        candidates = self.candidates((0, 24))
        definitions = [self.definition("slow", "swing_high_trail_7d"), self.definition("fast", "stop")]
        first = state.simulate_all_definitions(candidates, definitions, self.execute)
        second = state.simulate_all_definitions(candidates.sample(frac=1, random_state=3), definitions, self.execute)
        self.assertEqual(state.canonical_frame_hash(first[0], sort_fields=("definition_id", "candidate_key")), state.canonical_frame_hash(second[0], sort_fields=("definition_id", "candidate_key")))
        self.assertEqual(first[0].groupby("definition_id").size().to_dict(), {"fast": 2, "slow": 1})

    def test_economic_addresses_are_unique_within_definition(self):
        row = {"symbol": "PF_XUSD", "decision_ts": "2025-01-01T00:00:00Z", "entry_ts": "2025-01-01T00:05:00Z", "initial_stop": 110.0, "risk_denominator": 10.0, "exit_policy": "fixed_72h", "maximum_exit_ts": "2025-01-04T00:05:00Z"}
        addresses = [repaired.economic_address({**row, "symbol": symbol}) for symbol in ("PF_XUSD", "PF_YUSD")]
        self.assertEqual(len(addresses), len(set(addresses)))

    def test_contract_and_control_freeze_precede_cost_and_control_outcomes(self):
        source = inspect.getsource(repaired.run)
        self.assertLess(source.index("assert_rankable_signal_state_contract"), source.index("attach_costs(accepted"))
        self.assertLess(source.index('write_csv(root / "controls/control_key_manifest.csv"'), source.index("for control in control_keys.to_dict"))
        self.assertIn('len(set(PRESERVED_IDS)-set(summary.definition_id))', source)
        self.assertLess(source.index("compact_bundle(root)"), source.index("update_campaign(root, final)"))

    def test_campaign_update_uses_canonical_schema_fields(self):
        source = inspect.getsource(repaired.update_campaign)
        self.assertIn('dependency.loc[affected, "replay_status"]', source)
        self.assertNotIn('dependency.loc[affected, "status"]', source)
        self.assertIn('drop(columns=["status"]', source)
        self.assertIn('drop(columns=["next_allowed_action"]', source)


if __name__ == "__main__":
    unittest.main()
