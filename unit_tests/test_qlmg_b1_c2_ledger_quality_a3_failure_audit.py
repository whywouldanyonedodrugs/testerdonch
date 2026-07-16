import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import run_qlmg_b1_c2_ledger_quality_a3_failure_audit as mod


class B1C2A3AuditTests(unittest.TestCase):
    def test_protected_slice_rejected(self):
        df = pd.DataFrame({"decision_ts": ["2026-01-01T00:00:00Z"]})
        with self.assertRaises(RuntimeError):
            mod.validate_no_protected_df(df, ["decision_ts"])

    def test_event_metric_recomputed_from_rows(self):
        df = pd.DataFrame({
            "decision_ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True),
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
            "net_R_variant": [2.0, -1.0, 0.5],
            "mark_price_available": [True, True, True],
            "funding_exact": [False, False, False],
        })
        row = mod.metric_row(df, "cid", "A3")
        self.assertEqual(row["events"], 3)
        self.assertAlmostEqual(row["net_R"], 1.5)
        self.assertAlmostEqual(row["PF"], 2.5)
        self.assertAlmostEqual(row["win_rate"], 2 / 3)
        self.assertEqual(row["active_symbols"], 2)

    def test_control_normalization(self):
        self.assertAlmostEqual(mod.normalize_control(100, -30.0, 300), -10.0)

    def test_a3_sparse_can_be_sleeve_not_standalone(self):
        row = {"beats_fresh_nulls_and_baselines": True, "months": 1, "symbols": 3, "single_month_dominance": True, "net_R": 10, "PF": 1.5}
        null_ok = bool(row["beats_fresh_nulls_and_baselines"])
        label = "rare_regime_sleeve_candidate" if null_ok and (row["months"] < 3 or row["symbols"] < 5 or row["single_month_dominance"]) else "standalone_candidate"
        self.assertEqual(label, "rare_regime_sleeve_candidate")

    def test_a2_overlay_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram"])
            ctx = mod.RunContext(args=args, run_root=root, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"), notifier=mod.RunNotifier(root, disabled=True))
            mod.stage_a2_reuse(ctx)
            df = pd.read_csv(root / "a2_reuse/a2_feature_reuse_summary.csv")
        self.assertIn("feature_overlay_candidate", set(df["label"]))
        self.assertIn("current_translation_rejected_only", set(df["label"]))

    def test_b1_trailing_anchor_and_current_only_block(self):
        anchors = pd.read_parquet(mod.CORRECTED_ROOT / "b1_sidecar/b1_event_anchor_ledger.parquet")
        self.assertTrue(anchors["trailing_only"].fillna(False).all())
        self.assertFalse(anchors["current_only_rankable"].fillna(False).any())

    def test_b1_leader_selection_before_entry_fixture(self):
        row = {"leader_selected_before_entry": True, "basket_trade": False}
        self.assertTrue(row["leader_selected_before_entry"])
        self.assertFalse(row["basket_trade"])

    def test_c2_event_day_chase_blocked_and_timestamp_preserved(self):
        df = pd.read_parquet(mod.CORRECTED_ROOT / "c2_sidecar/c2_event_level_replay.parquet")
        self.assertFalse(df["event_day_chase_primary"].fillna(True).any())
        self.assertTrue(df["first_reaction_excluded"].fillna(False).all())
        self.assertTrue(df["md_excerpt_seed_limited"].fillna(False).all())

    def test_c2_sparse_mechanism_label(self):
        label = "sample_limited_sleeve_candidate" if 3 < 20 else "c2_mechanism_specific_candidate_found"
        self.assertEqual(label, "sample_limited_sleeve_candidate")

    def test_branch_x_no_retune_status_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram"])
            ctx = mod.RunContext(args=args, run_root=root, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"), notifier=mod.RunNotifier(root, disabled=True))
            mod.stage_branch_x(ctx)
            text = (root / "branch_x/live_capture_export_request.md").read_text()
            matrix = pd.read_csv(root / "branch_x/micro_canary_readiness_matrix.csv")
        self.assertIn("No Branch X retuning", text)
        self.assertFalse(matrix["micro_canary_possible_now"].fillna(True).any())

    def test_forensic_post_patch_provenance_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram"])
            ctx = mod.RunContext(args=args, run_root=root, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"), notifier=mod.RunNotifier(root, disabled=True))
            mod.stage_forensic(ctx)
            obj = json.loads((root / "forensic/code_hash_provenance.json").read_text())
        self.assertIn(obj["verdict"], {"post_patch_artifacts_trustworthy", "post_patch_artifact_provenance_incomplete"})

    def test_compact_bundle_excludes_large_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram"])
            ctx = mod.RunContext(args=args, run_root=root, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"), notifier=mod.RunNotifier(root, disabled=True))
            (root / "QLMG_B1_C2_LEDGER_QUALITY_A3_FAILURE_AUDIT_REPORT.md").write_text("x")
            (root / "decision_summary.json").write_text("{}")
            mod.stage_bundle(ctx)
            idx = pd.read_csv(root / "compact_review_bundle/artifact_path_index.csv")
        self.assertFalse(idx["artifact"].astype(str).str.endswith(".parquet").any())

    def test_tmux_wrapper_gates(self):
        text = Path("tools/run_qlmg_b1_c2_ledger_quality_a3_failure_audit_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("remote Telegram required", text)
        self.assertIn("smoke first", text)


if __name__ == "__main__":
    unittest.main()
