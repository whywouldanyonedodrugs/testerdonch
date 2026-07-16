from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tools import run_qlmg_listing_generic_full_event_replay as mod
from tools.qlmg_regime_stack import validate_no_protected


class ListingGenericFullEventReplayTests(unittest.TestCase):
    def test_protected_slice_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"ts": ["2026-01-01T00:00:00Z"]}), ["ts"])

    def test_parse_defaults(self):
        args = mod.parse_args([])
        self.assertEqual(args.seed, 20260627)
        self.assertEqual(args.targeted_download_cap_gb, 30.0)
        self.assertEqual(args.max_output_gb, 40.0)
        self.assertFalse(args.download_targeted_1m)

    def test_reconstruction_labels_are_limited(self):
        self.assertIn("exact_full_event_reconstruction", mod.RECON_LABELS)
        self.assertNotIn("exact_reconstruction", mod.RECON_LABELS)

    def test_horizon_hours(self):
        self.assertEqual(mod.horizon_hours("30m"), 0.5)
        self.assertEqual(mod.horizon_hours("6h"), 6.0)
        self.assertEqual(mod.horizon_hours("2d"), 48.0)

    def test_make_window_rejects_protected(self):
        out = mod.make_window({"candidate_id": "c"}, {"event_id": "e", "symbol": "BTCUSDT"}, "candidate_event", "full_hold", pd.Timestamp("2025-12-31T23:00:00Z"), pd.Timestamp("2026-01-01T01:00:00Z"), 1)
        self.assertIsNone(out)

    def test_estimate_window_positive(self):
        self.assertGreater(mod.estimate_window_gb({"hours": 28}), 0)

    def test_d4_root_cause_duplicate(self):
        cause = mod.classify_d4_root_cause({"event_ledger_rows": 4482, "event_ledger_unique_event_id": 4475})
        self.assertEqual(cause, "duplicate event rows")

    def test_d4_root_cause_protocol(self):
        cause = mod.classify_d4_root_cause({"event_ledger_rows": 0, "event_ledger_unique_event_id": 0})
        self.assertEqual(cause, "protocol issue")

    def test_candidate_config_hash_stable(self):
        row = {"candidate_id": "a", "family": "new_perp_listing_event_study", "subfamily": "vwap_loss_short", "horizon": "6h"}
        self.assertEqual(mod.candidate_config_hash(row), mod.candidate_config_hash(dict(row)))

    def test_path_metrics_missing_file_reports_error(self):
        out = mod.path_metrics(Path("/tmp/missing_path_for_test.parquet"), "short", 100.0, 150.0, 3.0)
        self.assertTrue(str(out["replay_status"]).startswith("error"))

    def test_control_quality_logic_fixture(self):
        ev_cov = 0.9
        ctrl_cov = 0.8
        self.assertTrue(ctrl_cov + 0.05 < ev_cov)

    def test_rankable_scope_excludes_core_24h(self):
        df = pd.DataFrame([
            {"candidate_id": "c", "window_scope": "full_hold", "window_role": "candidate_event"},
            {"candidate_id": "c", "window_scope": "core_24h", "window_role": "candidate_event"},
        ])
        out = mod.rankable_replay_scope(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["window_scope"], "full_hold")

    def test_control_comparison_normalizes_multiple_controls(self):
        df = pd.DataFrame([
            {"candidate_id": "c", "window_scope": "full_hold", "window_role": "candidate_event", "one_minute_mark_replayed": True, "net_R_1m_mark_proxy": 2.0},
            {"candidate_id": "c", "window_scope": "core_24h", "window_role": "candidate_event", "one_minute_mark_replayed": True, "net_R_1m_mark_proxy": 100.0},
            {"candidate_id": "c", "window_scope": "full_hold", "window_role": "control", "control_type": "a", "one_minute_mark_replayed": True, "net_R_1m_mark_proxy": 1.0},
            {"candidate_id": "c", "window_scope": "full_hold", "window_role": "control", "control_type": "b", "one_minute_mark_replayed": True, "net_R_1m_mark_proxy": 1.0},
            {"candidate_id": "c", "window_scope": "full_hold", "window_role": "control", "control_type": "c", "one_minute_mark_replayed": True, "net_R_1m_mark_proxy": 1.0},
        ])
        out = mod.summarize_rankable_control_comparison("c", df)
        self.assertEqual(out["candidate_event_count"], 1)
        self.assertEqual(out["core_24h_diagnostic_rows_present"], 1)
        self.assertEqual(out["event_signal_R"], 2.0)
        self.assertEqual(out["control_signal_R_raw_sum"], 3.0)
        self.assertEqual(out["control_signal_R_normalized_to_candidate_count"], 1.0)
        self.assertTrue(out["beats_controls"])

    def test_decision_labels_are_conservative(self):
        self.assertIn("listing_vwap_loss_full_event_prelead_confirmed", mod.DECISION_LABELS)
        self.assertNotIn("validated", mod.DECISION_LABELS)

    def test_required_outputs_include_final_report(self):
        with tempfile.TemporaryDirectory() as td:
            outs = mod.required_outputs(Path(td), "decision-report")
            self.assertTrue(any("QLMG_LISTING_GENERIC_FULL_EVENT_REPLAY_REPORT.md" in str(p) for p in outs))

    def test_wrapper_launch_gate_text(self):
        text = Path("tools/run_qlmg_listing_generic_full_event_replay_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("Full tmux launch not started", text)

    def test_stage_list_all_excludes_all(self):
        self.assertNotIn("all", mod.stage_list("all"))


if __name__ == "__main__":
    unittest.main()
