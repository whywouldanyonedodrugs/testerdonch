from __future__ import annotations

import inspect
import unittest

import pandas as pd

from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools.run_kraken_prior_high_v2_survivor_preflight import (
    PROFILE,
    event_key_sets,
    parameter_similarity,
)
from tools.run_kraken_prior_high_v2_materialization_profile import PROFILE as PROFILE_V2, PAIRS, DRY_DEFINITIONS
from tools import run_kraken_prior_high_v2_full_targeted as full_targeted
from tools import run_kraken_prior_high_v2_control_matching_repair as control_repair


class PriorHighV2SurvivorPreflightTests(unittest.TestCase):
    def test_full_targeted_symmetric_forensics_has_full_and_partitioned_scopes(self):
        paired = pd.DataFrame({
            "candidate_definition_id": ["d1", "d1"], "control_class": ["same_symbol", "same_symbol"],
            "funding_mode": ["severe_imputed", "severe_imputed"], "slippage_round_trip_bps": [12, 12],
            "funding_partition": ["both_fully_exact", "imputed_or_mixed"], "period": ["2024", "2025_h2"],
            "paired_uplift_R": [1.0, -0.25], "symbol_month": ["A|2024-01", "B|2025-08"],
            "candidate_symbol": ["A", "B"], "month": ["2024-01", "2025-08"],
        })
        summary, _ = full_targeted.symmetric_forensics(paired)
        scopes = set(zip(summary["funding_partition"], summary["period"]))
        self.assertIn(("all", "full_train"), scopes)
        self.assertIn(("both_fully_exact", "full_train"), scopes)
        self.assertIn(("all", "2024"), scopes)

    def test_full_targeted_freezes_all_control_keys_before_outcomes(self):
        source = inspect.getsource(full_targeted.run)
        self.assertLess(source.index("control_key_manifest.csv"), source.index("evaluate_control_outcomes"))
        self.assertIn("len(definitions) * len(profile.CONTROL_CLASSES)", source)

    def test_control_repair_precheck_does_not_read_outcomes(self):
        source = inspect.getsource(control_repair.precheck)
        self.assertNotIn("prior_high_event_from_address", source)
        self.assertNotIn("prior_high_exit_plan", source)
        self.assertIn("prior_high_atr_at_decision", source)
        self.assertIn("evaluate_parent_regime_gate", source)
        self.assertIn("evaluate_funding_gate", source)

    def test_control_repair_uses_uniform_adequacy_contract(self):
        source = inspect.getsource(control_repair.run)
        self.assertIn("coverage.outcome_coverage >= .70", source)
        self.assertIn("coverage.completed_control_outcomes >= 15", source)
    def test_profile_is_registered_as_dry_run_only(self) -> None:
        self.assertIn(PROFILE, runner.PHASE_PROFILES)
        stages = runner.PHASE_PROFILES[PROFILE]["stages"]
        self.assertIn("prior-high-v2-targeted-materialization-profile-dry-run", stages)
        self.assertNotIn("prior-high-v2-targeted-materialization", stages)

    def test_v2_profile_is_registered_with_bounded_dry_run(self) -> None:
        self.assertIn(PROFILE_V2, runner.PHASE_PROFILES)
        stages = runner.PHASE_PROFILES[PROFILE_V2]["stages"]
        self.assertIn("prior-high-v2-targeted-materialization-profile-v2-dry-run", stages)
        self.assertEqual(set(DRY_DEFINITIONS), {"prior_high_v2_022", "prior_high_v2_038", "prior_high_v2_023"})
        self.assertEqual(PAIRS, {"prior_high_v2_022": "prior_high_v2_038", "prior_high_v2_035": "prior_high_v2_019"})

    def test_event_identity_ignores_candidate_specific_event_id(self) -> None:
        events = pd.DataFrame([
            {"candidate_definition_id": "a", "symbol": "PF_XBTUSD", "decision_ts": "2025-01-01T00:00:00Z", "event_id": "a1"},
            {"candidate_definition_id": "b", "symbol": "PF_XBTUSD", "decision_ts": "2025-01-01T00:00:00Z", "event_id": "b9"},
        ])
        keys = event_key_sets(events)
        self.assertEqual(keys["a"], keys["b"])

    def test_parameter_similarity_is_bounded_and_exit_sensitive(self) -> None:
        left = pd.Series({"signal_type": "x", "exit_template": "time"})
        same = parameter_similarity(left, left.copy())
        right = left.copy()
        right["exit_template"] = "trail"
        changed = parameter_similarity(left, right)
        self.assertEqual(same, 1.0)
        self.assertGreaterEqual(changed, 0.0)
        self.assertLess(changed, same)


if __name__ == "__main__":
    unittest.main()
