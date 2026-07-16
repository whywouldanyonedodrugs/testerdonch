import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools import qlmg_evidence_contracts as contracts
from tools import run_qlmg_no_vendor_progress_run as mod


class NoVendorProgressHelpersTest(unittest.TestCase):
    def test_protected_slice_rejected(self):
        args = mod.parse_args(["--start", "2026-01-01", "--disable-telegram"])
        with self.assertRaises(RuntimeError):
            mod.clamp_window(args)

    def test_funding_no_cross_exact_and_cross_proxy(self):
        df = pd.DataFrame({
            "event_id": ["a", "b"],
            "candidate_id": ["c", "c"],
            "family": ["A1", "A1"],
            "branch_id": ["b", "b"],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "decision_ts": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "entry_ts": ["2025-01-01T00:05:00Z", "2025-01-01T00:05:00Z"],
            "exit_ts": ["2025-01-01T01:00:00Z", "2025-01-01T12:00:00Z"],
            "side": ["long", "long"],
            "entry_price": [100.0, 100.0],
            "exit_price": [101.0, 101.0],
            "stop_price": [95.0, 95.0],
            "target_price": [110.0, 110.0],
            "risk_bps_used": [500.0, 500.0],
            "gross_R": [0.2, 0.2],
            "net_R": [0.18, 0.18],
            "mark_available": [True, True],
        })
        out = mod.normalize_event_ledger(df, "A1", "branch", "fixture")
        self.assertTrue(bool(out.loc[0, "funding_exact"]))
        self.assertFalse(bool(out.loc[0, "funding_proxy_used"]))
        self.assertFalse(bool(out.loc[1, "funding_exact"]))
        self.assertTrue(bool(out.loc[1, "funding_proxy_used"]))
        self.assertTrue(contracts.validate_funding_mark_flags(out).passed)

    def test_no_vendor_outcome_vocab(self):
        self.assertIn("discard_current_translation_no_vendor_path", mod.NO_VENDOR_OUTCOMES)
        self.assertIn("preserve_hypothesis_generate_new_variant", mod.NO_VENDOR_OUTCOMES)
        self.assertNotIn("waiting_for_vendor", mod.NO_VENDOR_OUTCOMES)

    def test_capture_state_vocab_and_allowed_decision(self):
        self.assertIn("operator_attested_capture_calibration_capped", mod.CAPTURE_STATES)
        self.assertIn("branch_x_micro_canary_possible_execution_only", mod.ALLOWED_NEXT_DECISIONS)
        self.assertIn("run_train_only_candidate_validation_package_next", mod.ALLOWED_NEXT_DECISIONS)
        self.assertNotIn("run_candidate_validation_package_next", mod.ALLOWED_NEXT_DECISIONS)

    def test_overlap_embargo_audit_detects_overlap(self):
        pool = pd.DataFrame({
            "candidate_key": ["cand"],
            "symbol": ["BTCUSDT"],
            "entry_ts": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
            "exit_ts": pd.to_datetime(["2025-01-02T00:00:00Z"], utc=True),
        })
        controls = pd.DataFrame({
            "candidate_key": ["cand"],
            "control_type": ["same_symbol"],
            "control_event_id": ["ctrl"],
            "control_symbol": ["BTCUSDT"],
            "control_entry_ts": pd.to_datetime(["2025-01-01T12:00:00Z"], utc=True),
            "control_exit_ts": pd.to_datetime(["2025-01-01T18:00:00Z"], utc=True),
            "source_window_id": ["w"],
        })
        audit = mod.build_overlap_embargo_audit(controls, pool)
        self.assertTrue(bool(audit.loc[0, "overlap_or_embargo_violation"]))


class NoVendorProgressRunnerTest(unittest.TestCase):
    def make_ctx(self, root: Path) -> mod.RunContext:
        args = mod.parse_args(["--run-root", str(root), "--disable-telegram", "--smoke"])
        return mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-15", tz="UTC"))

    def test_preflight_fails_closed_missing_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            ctx = self.make_ctx(root)
            with patch.object(mod, "QA_REQUIRED_FILES", [Path(tmp) / "missing.json"]):
                with self.assertRaises(RuntimeError):
                    mod.stage_preflight(ctx)

    def test_live_capture_attested_capped_for_coherent_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zpath = root / "capture.zip"
            payload = b"hello"
            sha = __import__("hashlib").sha256(payload).hexdigest()
            manifest = f"file_path,stream,symbol,date,start_ts,end_ts,row_message_count,size_bytes,sha256,compression,status,quality_flags\n/opt/qlmg_live_capture/data/raw/publicTrade/BTCUSDT/test.jsonl,publicTrade,BTCUSDT,2026-06-28,2026-06-28T00:00:00Z,2026-06-28T00:01:00Z,1,{len(payload)},{sha},none,ok,\n"
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("qlmg_live_capture/data/manifests/file_manifest.csv", manifest)
                z.writestr("qlmg_live_capture/data/raw/publicTrade/BTCUSDT/test.jsonl", payload)
            ctx = self.make_ctx(root / "run")
            with patch.object(mod, "LIVE_CAPTURE_ZIP", zpath):
                mod.stage_live_capture(ctx)
            prov = json.loads((ctx.run_root / "live_capture/provenance_status.json").read_text())
            self.assertEqual(prov["capture_evidence_state"], "operator_attested_capture_calibration_capped")
            self.assertTrue(prov["calibration_allowed"])

    def test_library_has_portfolio_sleeve_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            ctx = self.make_ctx(root)
            (root / "stress").mkdir(parents=True)
            pd.DataFrame({
                "candidate_id": ["c"], "family": ["A1"], "event_count": [10], "net_R": [1.0], "active_months": [2], "active_symbols": [1],
                "dominant_month_share": [0.5], "dominant_symbol_share": [1.0], "label_cap_reason": ["funding"]
            }).to_csv(root / "stress/tier1_stress_summary.csv", index=False)
            mod.stage_library(ctx)
            lib = pd.read_csv(root / "library/no_vendor_candidate_library.csv")
            for col in ["standalone_status", "portfolio_sleeve_status", "rare_regime_status", "feature_overlay_status", "reason_preserved_or_discarded"]:
                self.assertIn(col, lib.columns)

    def test_tmux_wrapper_has_launch_smoke_and_telegram_gates(self):
        text = Path("tools/run_qlmg_no_vendor_progress_run_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("remote Telegram required", text)
        self.assertIn("smoke first", text)
        self.assertIn("run_qlmg_no_vendor_progress_run.py", text)


if __name__ == "__main__":
    unittest.main()
