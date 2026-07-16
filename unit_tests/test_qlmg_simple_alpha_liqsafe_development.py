from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tools import run_qlmg_simple_alpha_liqsafe_development as mod
from tools.qlmg_regime_stack import validate_no_protected


class QLMGSimpleAlphaLiqsafeDevelopmentTests(unittest.TestCase):
    def test_protected_slice_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"decision_ts": ["2026-01-01T00:00:00Z"]}), ["decision_ts"])

    def test_candidate_from_row_preserves_config_not_summary_only(self):
        row = {
            "family": "funding_window_orb_failure",
            "subfamily": "funding_30m_failure",
            "horizon": "30m",
            "target_r": "5",
            "stop_mult": "0.5",
            "risk_bps_override": "",
            "candidate_id": "abc",
            "net_R": 999,
        }
        cand = mod.candidate_from_row(row)
        self.assertEqual(cand["candidate_id"], "abc")
        self.assertEqual(cand["family"], "funding_window_orb_failure")
        self.assertNotIn("net_R", cand)

    def test_reconstruct_events_uses_filter_semantics(self):
        args = SimpleNamespace(max_symbols=0, smoke=False)
        ctx = SimpleNamespace(args=args, start=pd.Timestamp("2025-01-01T00:00:00Z"), end=pd.Timestamp("2025-12-31T23:59:59Z"))
        events = pd.DataFrame({
            "family": ["funding_window_orb_failure", "new_perp_listing_event_study"],
            "simple_subfamily": ["funding_30m_failure", "first_lower_high"],
            "liquidity_tier": ["C", "C"],
            "decision_ts": pd.to_datetime(["2025-01-02T00:00:00Z", "2025-01-02T01:00:00Z"], utc=True),
            "entry_ts": pd.to_datetime(["2025-01-02T00:05:00Z", "2025-01-02T01:05:00Z"], utc=True),
        })
        cand = {"candidate_id": "c1", "family": "funding_window_orb_failure", "subfamily": "funding_30m_failure", "tier_filter": "C"}
        sub, status, reason = mod.reconstruct_events_for_candidate(ctx, cand, events)
        self.assertEqual(status, "reconstructed_from_registry_and_event_ledger")
        self.assertEqual(reason, "")
        self.assertEqual(len(sub), 1)

    def test_path_edge_failed_execution_labels_are_preserved(self):
        self.assertIn("path_edge_exit_problem", mod.PATH_EDGE_LABELS)
        self.assertIn("cost_fragile_candidate", mod.PATH_EDGE_LABELS)

    def test_signal_and_account_r_are_distinct_under_leverage_scaling(self):
        ret = pd.Series([1.0, -0.5, 2.0])
        signal = ret.sum()
        account = (ret * 0.5).sum()
        self.assertEqual(signal, 2.5)
        self.assertEqual(account, 1.25)
        self.assertLess(account, signal)

    def test_dynamic_leverage_buffer_decreases_with_stop_distance(self):
        tight = mod.max_safe_leverage(100.0, 1.25)
        wide = mod.max_safe_leverage(1000.0, 1.25)
        self.assertGreater(tight, wide)
        self.assertLessEqual(tight, 10.0)

    def test_download_disabled_by_default(self):
        args = mod.parse_args([])
        self.assertFalse(args.download_targeted_1m)
        self.assertEqual(args.seed, 20260627)
        self.assertEqual(args.nulls_per_event, 3)
        self.assertEqual(args.candidate_limit, 300)

    def test_d4_mandatory_carry_forward_contract_path(self):
        self.assertTrue(str(mod.D4_SURVIVAL_ROOT).endswith("phase_qlmg_d4_survivability_redesign_20260625_v1_20260626_133332"))

    def test_tmux_wrapper_requires_launch_flag(self):
        text = Path("tools/run_qlmg_simple_alpha_liqsafe_development_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("Full tmux launch not started", text)

    def test_required_outputs_include_d4_contract(self):
        with tempfile.TemporaryDirectory() as td:
            outs = mod.required_outputs(Path(td), "d4-carry-forward-integration")
            self.assertTrue(any("d4_targeted_execution_depth_collection_contract.json" in str(p) for p in outs))


if __name__ == "__main__":
    unittest.main()
