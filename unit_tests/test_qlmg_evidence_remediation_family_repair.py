from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from tools import run_qlmg_evidence_remediation_family_repair as mod
from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, validate_no_protected


class EvidenceRemediationFamilyRepairTests(unittest.TestCase):
    def test_protected_slice_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"decision_ts": [FINAL_HOLDOUT_START]}), ["decision_ts"])

    def test_projected_mean_expanded_to_events_is_quarantined(self) -> None:
        df = pd.DataFrame({
            "candidate_id": ["x", "x"],
            "event_id": ["e1", "e2"],
            "net_R": [0.5, 0.5],
            "projection_source": ["summary_mean_R_projection", "summary_mean_R_projection"],
            "PF": [1.5, 1.5],
        })
        failures = mod.audit_projected_mean_expansion(df)
        self.assertTrue(any(f["failure_type"] == "projected_mean_expanded_to_events" for f in failures))
        self.assertTrue(mod.projected_artifact_is_quarantined(df))

    def test_performance_metrics_blocked_without_event_level_ledger(self) -> None:
        self.assertFalse(mod.evidence_level_allows_performance_metrics("level_2_path_or_mae_mfe_support_only"))
        self.assertTrue(mod.evidence_level_allows_performance_metrics("level_3_event_level_trade_ledger"))

    def test_core_full_double_count_detected(self) -> None:
        df = pd.DataFrame({
            "candidate_id": ["x", "x"],
            "event_id": ["e1", "e1"],
            "window_scope": ["core_24h", "full_hold"],
            "net_R": [1.0, 1.2],
        })
        failures = mod.audit_projected_mean_expansion(df)
        self.assertTrue(any(f["failure_type"] == "core_24h_full_hold_double_count" for f in failures))

    def test_control_normalization_issue_detected(self) -> None:
        bad = pd.DataFrame({"candidate_event_count": [10], "control_event_count": [30], "raw_control_net_R": [9.0], "normalized_control_net_R": [9.0]})
        good = pd.DataFrame({"candidate_event_count": [10], "control_event_count": [30], "raw_control_net_R": [9.0], "normalized_control_net_R": [3.0]})
        self.assertTrue(mod.detect_control_normalization_issue(bad))
        self.assertFalse(mod.detect_control_normalization_issue(good))

    def test_quarantine_historical_failure_not_active_blocker(self) -> None:
        row = {"artifact_path": "/tmp/old/sweep_summary.csv", "failure_type": "performance_metrics_without_event_level_trade_ledger"}
        qclass, action = mod.classify_quarantine_failure(row)
        self.assertEqual(qclass, "already_quarantined_historical_failure")
        self.assertEqual(action, "forbidden_for_ranking")

    def test_active_scoring_failure_blocks_gate(self) -> None:
        row = {"artifact_path": "/tmp/a3_validation/summary.csv", "failure_type": "performance_metrics_without_event_level_trade_ledger"}
        qclass, action = mod.classify_quarantine_failure(row)
        self.assertEqual(qclass, "unresolved_active_scoring_lineage_failure")
        self.assertEqual(action, "blocks_corrected_sweep")

    def test_corrected_sweep_gate_blocks_active_failures_only(self) -> None:
        allowed, blockers = mod.corrected_sweep_allowed(1, True, True, True, True)
        self.assertFalse(allowed)
        self.assertIn("unresolved_active_scoring_lineage_failures", blockers)
        allowed, blockers = mod.corrected_sweep_allowed(0, True, True, True, True)
        self.assertTrue(allowed)
        self.assertEqual(blockers, [])

    def test_rankable_evidence_set_is_narrow_a3_a2_only(self) -> None:
        ledger = pd.DataFrame({
            "candidate_id": ["a3", "a2", "f1", "d4"],
            "family": ["A3", "A2_redesign_only", "F1", "D4"],
            "branch_id": ["branch_l", "branch_l", "branch_l", "branch_x_execution_sensitive"],
            "rankable": [True, True, True, True],
        })
        out = mod.filter_rankable_evidence_set(ledger)
        self.assertEqual(set(out["family"]), {"A3", "A2_redesign_only"})
        self.assertTrue(out["rankable_scope_policy"].eq("narrow_corrected_sweep_only_A3_A2_redesign").all())

    def test_a2_liquidation_taxonomy_decomposes_flags(self) -> None:
        df = pd.DataFrame({
            "family": ["A2", "A2", "A2"],
            "event_id": ["a", "b", "c"],
            "candidate_id": ["x", "x", "x"],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "decision_ts": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "liquidation_flag": [True, True, True],
            "mark_price_available": [False, True, True],
            "same_bar_ambiguity": [False, True, False],
            "risk_bps_used": [100, 100, 2000],
        })
        out = mod.build_liquidation_taxonomy(df)
        self.assertIn("proxy_mark_missing_last_used", set(out["taxonomy"]))
        self.assertIn("same_bar_ambiguity", set(out["taxonomy"]))
        self.assertIn("stop_too_wide_relative_to_safe_leverage", set(out["taxonomy"]))

    def test_micro_canary_restricted_to_two_listing_analogs(self) -> None:
        ok, reason = mod.micro_canary_allowed("589a8c85c943", has_live_capture=True)
        self.assertTrue(ok)
        self.assertIn("execution telemetry", reason)
        ok, reason = mod.micro_canary_allowed("9dc07cfc405c", has_live_capture=True)
        self.assertFalse(ok)
        ok, reason = mod.micro_canary_allowed("D4__b4c9487fe82c", has_live_capture=True, is_d4=True)
        self.assertFalse(ok)
        self.assertIn("D4 micro-canary blocked", reason)

    def test_c2_mechanism_bucket_separates_families(self) -> None:
        self.assertEqual(mod.c2_mechanism_bucket("etf_institutional_access"), "etf_institutional_access")
        self.assertEqual(mod.c2_mechanism_bucket("major_unlock"), "supply_unlock_float")
        self.assertEqual(mod.c2_mechanism_bucket("leverage_access_expansion"), "leverage_access_expansion")

    def test_operator_decision_priority(self) -> None:
        self.assertEqual(mod.choose_operator_decision({"corrected_sweep_allowed": True}), "proceed_to_corrected_sweep")
        self.assertEqual(mod.choose_operator_decision({"corrected_sweep_allowed": False, "a3_validation_verdict": "a3_fragile_but_alive"}), "run_a3_validation_next")
        self.assertEqual(mod.choose_operator_decision({"blocked_by_protocol_issue": True}), "blocked_by_protocol_issue")

    def test_run_root_collision_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / mod.DEFAULT_RUN_ID
            base.mkdir()
            args = SimpleNamespace(run_root="", smoke=False)
            with mock.patch.object(mod, "RESULTS_ROOT", Path(td)):
                root, reason = mod.resolve_run_root(args)
        self.assertIn("default_root_existed_suffix", reason)
        self.assertNotEqual(root.name, mod.DEFAULT_RUN_ID)

    def test_tmux_wrapper_launch_and_telegram_gates(self) -> None:
        txt = Path("tools/run_qlmg_evidence_remediation_family_repair_tmux.sh").read_text()
        self.assertIn("--launch-tmux", txt)
        self.assertIn("remote Telegram required", txt)


if __name__ == "__main__":
    unittest.main()
