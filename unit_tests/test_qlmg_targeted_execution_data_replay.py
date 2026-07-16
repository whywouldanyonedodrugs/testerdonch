from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tools import run_qlmg_targeted_execution_data_replay as mod
from tools.qlmg_regime_stack import validate_no_protected


class TargetedExecutionDataReplayTests(unittest.TestCase):
    def test_protected_slice_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"window_end": ["2026-01-01T00:00:00Z"]}), ["window_end"])

    def test_parse_defaults_include_download_cap_and_d4(self):
        args = mod.parse_args([])
        self.assertEqual(args.seed, 20260627)
        self.assertEqual(args.targeted_download_cap_gb, 12.0)
        self.assertTrue(args.include_d4)
        self.assertFalse(args.download_targeted_1m)

    def test_depth_contract_fallback_schema_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "contracts/depth_data_contract_summary.csv"
            p.parent.mkdir(parents=True)
            pd.DataFrame([{c: True for c in mod.EXPECTED_DEPTH_SCHEMA}]).to_csv(p, index=False)
            path, status, df = mod.select_depth_contract_file(root)
            self.assertEqual(path, p)
            self.assertIn("fallback", status)
            self.assertFalse(df.empty)

    def test_depth_contract_fail_closed_without_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "contracts/depth_data_contract_summary.csv"
            p.parent.mkdir(parents=True)
            pd.DataFrame([{"contract_id": "x"}]).to_csv(p, index=False)
            path, status, df = mod.select_depth_contract_file(root)
            self.assertIsNone(path)
            self.assertEqual(status, "no_usable_depth_contract_schema")
            self.assertTrue(df.empty)

    def test_candidate_config_hash_stable(self):
        row = {"candidate_id": "a", "family": "funding", "horizon": "72h", "target_r": 5, "stop_mult": 1}
        self.assertEqual(mod.candidate_config_hash(row), mod.candidate_config_hash(dict(row)))

    def test_window_creation_rejects_protected(self):
        cand = {"candidate_id": "c", "family": "f"}
        base = {"event_id": "e", "symbol": "BTCUSDT"}
        out = mod._make_window(cand, base, "candidate_event", "core_24h", pd.Timestamp("2025-12-31T23:00:00Z"), pd.Timestamp("2026-01-01T01:00:00Z"), 1)
        self.assertIsNone(out)

    def test_window_estimate_positive(self):
        row = {"hours": 28}
        self.assertGreater(mod.estimate_window_gb(row), 0)

    def test_72h_detection(self):
        self.assertTrue(mod.is_72h({"horizon": "72h"}))
        self.assertFalse(mod.is_72h({"horizon": "24h"}))

    def test_decision_status_set_has_required_caps(self):
        self.assertIn("targeted_execution_data_prelead_unresolved", mod.DECISION_STATUSES)
        self.assertIn("carry_forward_d4_execution_depth", mod.DECISION_STATUSES)

    def test_d4_counts_function_shape(self):
        counts = mod.d4_counts()
        self.assertIn("accepted_events", counts)
        self.assertIn("resolved_events", counts)

    def test_download_requires_explicit_flag(self):
        args = mod.parse_args([])
        self.assertFalse(args.download_targeted_1m)
        args2 = mod.parse_args(["--download-targeted-1m"])
        self.assertTrue(args2.download_targeted_1m)

    def test_wrapper_requires_launch_flag(self):
        text = Path("tools/run_qlmg_targeted_execution_data_replay_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("Full tmux launch not started", text)

    def test_required_outputs_include_decision_table(self):
        with tempfile.TemporaryDirectory() as td:
            outs = mod.required_outputs(Path(td), "candidate-decision-table")
            self.assertTrue(any("candidate_decision_table.csv" in str(p) for p in outs))

    def test_mark_path_metrics_missing_status(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.parquet"
            out = mod._path_metrics(missing, "long", 100.0, 100.0, 3.0)
            self.assertTrue(str(out.get("replay_status")).startswith("error"))

    def test_no_model_row_inflation_by_drop_duplicates(self):
        df = pd.DataFrame([
            {"candidate_id": "a", "family": "f", "targeted_prelead_corrected": True},
            {"candidate_id": "a", "family": "f", "targeted_prelead_corrected": True},
        ])
        self.assertEqual(len(df.drop_duplicates("candidate_id")), 1)


if __name__ == "__main__":
    unittest.main()
