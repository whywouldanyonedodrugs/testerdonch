import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import tools.run_project_deep_cleanup_20260624 as mod


class ProjectDeepCleanupTests(unittest.TestCase):
    def test_run_root_suffixes_when_default_exists(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / mod.DEFAULT_RUN_ID).mkdir()
            args = SimpleNamespace(run_root="", results_root=str(base), run_id=mod.DEFAULT_RUN_ID)
            root, reason = mod.run_root_from_args(args)
            self.assertNotEqual(root, base / mod.DEFAULT_RUN_ID)
            self.assertIn("default_root_existed_created_suffix", reason)

    def test_guard_refuses_raw_data_and_source_paths(self):
        run_root = mod.REPO / "results/rebaseline/phase_project_deep_cleanup_20260624_v1"
        archive_root = mod.REPO / "archive/legacy_donch_research_20260624"
        for p in [
            Path("/opt/parquet/5m"),
            mod.REPO / "tools",
            mod.REPO / "unit_tests",
            mod.REPO / "docs/QLMG_PERP_PROJECT_STATE.md",
            mod.REPO / "live",
        ]:
            ok, _reason = mod.guard_deletion(p, run_root, archive_root)
            self.assertFalse(ok)

    def test_legacy_result_root_is_candidate_but_active_qlmg_is_not(self):
        run_root = mod.REPO / "results/rebaseline/phase_project_deep_cleanup_20260624_v1"
        archive_root = mod.REPO / "archive/legacy_donch_research_20260624"
        legacy = mod.REPO / "results/rebaseline/phase_state_transition_v2_20260421_track_b_sidecar_patched"
        if legacy.exists():
            ok, reason = mod.guard_deletion(legacy, run_root, archive_root)
            self.assertTrue(ok, reason)
        active = mod.REPO / "results/rebaseline/phase_qlmg_perp_project_reset_20260624_v1_20260624_092636"
        if active.exists():
            ok, _reason = mod.guard_deletion(active, run_root, archive_root)
            self.assertFalse(ok)

    def test_archive_selection_keeps_small_reports_not_heavy_ledgers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = root / "PHASE_REPORT.md"
            ledger = root / "decision_ledger.csv"
            report.write_text("summary\n", encoding="utf-8")
            ledger.write_text("x\n", encoding="utf-8")
            self.assertTrue(mod.archive_should_copy(report))
            self.assertFalse(mod.archive_should_copy(ledger))

    def test_redundant_index_with_gzip_is_candidate(self):
        with tempfile.TemporaryDirectory(dir=mod.REPO) as td:
            p = Path(td) / "trade_log_and_ledger_file_index.txt"
            p.write_text("x\n", encoding="utf-8")
            p.with_suffix(p.suffix + ".gz").write_bytes(b"gz")
            ok, reason = mod.classify_candidate(p, mod.REPO / "results/rebaseline/phase_project_deep_cleanup_20260624_v1", mod.REPO / "archive/legacy_donch_research_20260624")
            self.assertTrue(ok)
            self.assertEqual(reason, "redundant_uncompressed_index_with_gzip")


if __name__ == "__main__":
    unittest.main()
