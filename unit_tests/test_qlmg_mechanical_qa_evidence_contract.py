import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools import qlmg_evidence_contracts as contracts
from tools import run_qlmg_mechanical_qa_evidence_contract as mod


class EvidenceContractsTest(unittest.TestCase):
    def test_event_schema_requires_metric_ledger_fields(self):
        df = pd.DataFrame({"candidate_id": ["c"], "family": ["A3"], "decision_ts": ["2025-01-01T00:00:00Z"], "net_R": [1.0]})
        res = contracts.validate_event_trade_schema(df, require_all_fields=True)
        self.assertFalse(res.passed)
        self.assertIn("missing_event_trade_fields", ";".join(res.violations))

    def test_control_rows_require_source_and_window_ids(self):
        df = pd.DataFrame({
            "control_event_id": ["x"],
            "control_symbol": ["BTCUSDT"],
            "control_decision_ts": ["2025-01-01T00:00:00Z"],
            "matched_candidate_id": ["c"],
            "matching_basis": ["same_symbol"],
            "source_window_id": ["w"],
            "feature_source_ts": ["2024-12-31T23:55:00Z"],
        })
        self.assertTrue(contracts.validate_control_rows(df).passed)
        bad = df.drop(columns=["source_window_id"])
        self.assertFalse(contracts.validate_control_rows(bad).passed)

    def test_placeholder_controls_blocked(self):
        df = pd.DataFrame({
            "control_event_id": ["x"], "control_symbol": ["BTCUSDT"], "control_decision_ts": ["2025-01-01T00:00:00Z"],
            "matched_candidate_id": ["c"], "matching_basis": ["synthetic placeholder"], "source_window_id": ["w"], "feature_source_ts": ["2024-12-31T23:55:00Z"],
        })
        res = contracts.validate_control_rows(df)
        self.assertFalse(res.passed)
        self.assertIn("synthetic_or_placeholder_control_rows", ";".join(res.violations))

    def test_projected_mean_promotion_blocked(self):
        df = pd.DataFrame({"candidate_id": ["c"], "label": ["prelead_confirmed"], "metric_lineage": ["summary_projection_only"], "PF": [2.0]})
        res = contracts.validate_no_projected_metric_promotion(df)
        self.assertFalse(res.passed)
        self.assertIn("promotion_without_event_trade_ledger", ";".join(res.violations))

    def test_future_mfe_mae_feature_blocked(self):
        df = pd.DataFrame({"decision_ts": ["2025-01-01T00:00:00Z"], "feature_source_ts": ["2024-12-31T23:55:00Z"], "24h_mfe_bps": [100]})
        res = contracts.validate_pit_feature_timestamps(df)
        self.assertFalse(res.passed)
        self.assertIn("future_path_fields_present", ";".join(res.violations))

    def test_full_sample_quantile_leakage_scan_pattern(self):
        text = 'threshold = df["x"].quantile(0.95)'
        self.assertTrue(mod.CODE_PATTERNS["full_sample_quantile"].search(text))

    def test_mark_and_funding_proxy_exactness_blocked(self):
        df = pd.DataFrame({"funding_exact": [True], "funding_proxy_used": [True], "mark_available": [True], "mark_proxy_used": [True]})
        res = contracts.validate_funding_mark_flags(df)
        self.assertFalse(res.passed)
        joined = ";".join(res.violations)
        self.assertIn("funding_proxy_treated_exact", joined)
        self.assertIn("mark_proxy_treated_available", joined)

    def test_exact_funding_no_cross_case_allowed(self):
        df = pd.DataFrame({"funding_timestamps_crossed": [0], "funding_exact": [True], "funding_proxy_used": [False], "mark_available": [True], "mark_proxy_used": [False]})
        self.assertTrue(contracts.validate_funding_mark_flags(df).passed)

    def test_current_only_taxonomy_cannot_be_rankable(self):
        df = pd.DataFrame({"rankable": [True], "taxonomy_source": ["current-only taxonomy proxy"]})
        res = contracts.validate_no_current_only_taxonomy_rankable(df)
        self.assertFalse(res.passed)

    def test_protected_holdout_rejected(self):
        df = pd.DataFrame({"decision_ts": ["2026-01-01T00:00:00Z"]})
        res = contracts.require_no_protected_timestamps(df)
        self.assertFalse(res.passed)

    def test_artifact_scan_detects_identical_controls(self):
        df = pd.DataFrame({
            "candidate_id": ["c", "c"],
            "control_type": ["same_symbol", "same_regime"],
            "normalized_control_net_R": [1.23, 1.23],
        })
        risks = contracts.artifact_risk_scan(df, path="x.csv")
        self.assertTrue(any(r["risk"] == "identical_control_values_across_types" for r in risks))


class MechanicalQARunnerTest(unittest.TestCase):
    def make_ctx(self, root: Path) -> mod.RunContext:
        args = mod.parse_args(["--run-root", str(root), "--disable-telegram"])
        return mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-15", tz="UTC"))

    def test_live_capture_hash_mismatch_inventory_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zpath = root / "capture.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("qlmg_live_capture/data/manifests/file_manifest.csv", "path,size\na,1\n")
                z.writestr("qlmg_live_capture/data/raw/orderbook/2026-06-28/BTCUSDT/file.jsonl.gz", b"x")
            ctx = self.make_ctx(root / "run")
            refs = dict(mod.ACTIVE_REFERENCE_ROOTS)
            refs["live_capture_bundle"] = zpath
            with patch.object(mod, "ACTIVE_REFERENCE_ROOTS", refs):
                mod.stage_live_capture(ctx)
            prov = json.loads((ctx.run_root / "live_capture/provenance_status.json").read_text())
            self.assertFalse(prov["hash_matches_expected"])
            self.assertIn("live_capture_hash_mismatch", prov["labels"])
            self.assertFalse(prov["calibration_allowed"])

    def test_runner_writes_required_contracts_and_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "run"
            zpath = base / "capture.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("qlmg_live_capture/data/manifests/file_manifest.csv", "path,size\na,1\n")
            refs = {k: (base / k) for k in mod.ACTIVE_REFERENCE_ROOTS}
            for p in refs.values():
                p.mkdir(parents=True, exist_ok=True)
            refs["live_capture_bundle"] = zpath
            with patch.object(mod, "ACTIVE_REFERENCE_ROOTS", refs):
                self.assertEqual(mod.main(["--run-root", str(root), "--stage", "all", "--disable-telegram", "--smoke"]), 0)
            self.assertTrue((root / "contracts/event_level_trade_schema.yaml").exists())
            self.assertTrue((root / "quarantine/deprecated_promotion_labels.csv").exists())
            decision = json.loads((root / "decision_summary.json").read_text())
            self.assertIn(decision["primary_next_operator_decision"], mod.PRIMARY_DECISIONS)
            self.assertTrue(decision["not_a_proof_of_no_leakage"])

    def test_tmux_wrapper_has_launch_and_telegram_gates(self):
        text = Path("tools/run_qlmg_mechanical_qa_evidence_contract_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("remote Telegram required", text)
        self.assertIn("smoke first", text)

    def test_protected_output_scan_completes_on_current_run_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "safe.csv").write_text("decision_ts,value\n2025-01-01T00:00:00Z,1\n")
            res = contracts.scan_output_tree_for_protected(root)
            self.assertTrue(res.passed)
            self.assertGreaterEqual(res.rows_checked, 1)


if __name__ == "__main__":
    unittest.main()
