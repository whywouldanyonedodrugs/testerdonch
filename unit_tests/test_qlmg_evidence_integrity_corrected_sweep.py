from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from tools import run_qlmg_evidence_integrity_corrected_sweep as mod
from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, validate_no_protected


class EvidenceIntegrityCorrectedSweepTests(unittest.TestCase):
    def test_protected_slice_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"decision_ts": [FINAL_HOLDOUT_START]}), ["decision_ts"])

    def test_projected_mean_r_cannot_pass_as_event_r(self) -> None:
        df = pd.DataFrame({"candidate_id": ["x"], "PF": [1.4], "net_R": [10.0]})
        failures = mod.audit_metric_lineage_df(df)
        self.assertTrue(any(f["failure_type"] == "performance_metric_without_event_ledger" for f in failures))
        self.assertFalse(mod.evidence_level_allows_performance_metrics("level_2_path_or_mae_mfe_support_only"))

    def test_core_and_full_hold_double_count_blocked(self) -> None:
        df = pd.DataFrame({
            "candidate_id": ["x", "x"],
            "event_id": ["e1", "e1"],
            "window_scope": ["core_24h", "full_hold"],
            "net_R": [1.0, 1.2],
        })
        failures = mod.audit_metric_lineage_df(df)
        self.assertTrue(any(f["failure_type"] == "core_24h_full_hold_double_count" for f in failures))

    def test_controls_must_be_normalized(self) -> None:
        bad = pd.DataFrame({"candidate_event_count": [10], "control_event_count": [30], "raw_control_net_R": [9.0], "normalized_control_net_R": [9.0]})
        good = pd.DataFrame({"candidate_event_count": [10], "control_event_count": [30], "raw_control_net_R": [9.0], "normalized_control_net_R": [3.0]})
        self.assertTrue(mod.detect_control_normalization_issue(bad))
        self.assertFalse(mod.detect_control_normalization_issue(good))

    def test_pf_dd_sharpe_without_trade_ledger_blocked(self) -> None:
        self.assertFalse(mod.evidence_level_allows_performance_metrics("level_0_hypothesis_only"))
        self.assertFalse(mod.evidence_level_allows_performance_metrics("level_1_event_generator_support"))
        self.assertFalse(mod.evidence_level_allows_performance_metrics("level_2_path_or_mae_mfe_support_only"))
        self.assertTrue(mod.evidence_level_allows_performance_metrics("level_3_event_level_trade_ledger"))

    def test_duplicate_row_level_variants_share_dedup_key(self) -> None:
        row1 = {"family": "A2", "branch_id": "branch_l", "variant_id": "v", "regime_gate": "all", "entry_rule": "e", "stop_rule": "s", "exit_rule": "x", "risk_model": "r", "execution_assumptions": "taker", "required_data_tier": "Tier 1"}
        row2 = dict(row1)
        row2["candidate_id"] = "different_row_id"
        self.assertEqual(mod.dedup_key_from_row(row1), mod.dedup_key_from_row(row2))

    def test_branch_x_pnl_mixed_into_a2a3_ranking_blocked_by_branch(self) -> None:
        row = {"candidate_id": "D4__x", "family": "D4", "branch_id": "branch_x_execution_sensitive", "current_data_tier": "5m proxy"}
        reason = mod.data_tier_cap_reason("level_3_event_level_trade_ledger", row["branch_id"], row["current_data_tier"])
        self.assertIn("branch_x_requires", reason)

    def test_metric_lineage_net_r_must_sum_events(self) -> None:
        df = pd.DataFrame({"candidate_id": ["x", "x"], "event_id": ["e1", "e2"], "net_R": [1.0, 2.0], "reported_net_R": [5.0, 5.0]})
        failures = mod.audit_metric_lineage_df(df)
        self.assertTrue(any(f["failure_type"] == "net_R_not_sum_of_events" for f in failures))

    def test_run_root_collision_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / mod.DEFAULT_RUN_ID
            base.mkdir()
            args = SimpleNamespace(run_root="", smoke=False)
            with mock.patch.object(mod, "RESULTS_ROOT", Path(td)):
                root, reason = mod.resolve_run_root(args)
        self.assertIn("default_root_existed_suffix", reason)
        self.assertNotEqual(root.name, mod.DEFAULT_RUN_ID)

    def test_tmux_wrapper_launch_gate_text(self) -> None:
        txt = Path("tools/run_qlmg_evidence_integrity_corrected_sweep_tmux.sh").read_text()
        self.assertIn("--launch-tmux", txt)
        self.assertIn("remote Telegram required", txt)


if __name__ == "__main__":
    unittest.main()
