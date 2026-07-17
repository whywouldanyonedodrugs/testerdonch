import tempfile
import unittest
from pathlib import Path

from tools import build_kraken_first_wave_closure_review as stage5a
from tools import validate_kraken_first_wave_closure_review as validator


class FirstWaveClosureReviewTests(unittest.TestCase):
    def test_source_allowlist_rejects_parquet(self):
        with self.assertRaisesRegex(ValueError, "non-text source prohibited"):
            stage5a.assert_source_allowed(stage5a.C01 / "TRADE_LEDGER.parquet")

    def test_source_allowlist_rejects_unrelated_economic_root(self):
        with self.assertRaisesRegex(ValueError, "outside Stage 5A allowlist"):
            stage5a.assert_source_allowed(stage5a.ROOT / "results/rebaseline/unrelated/decision_summary.json")

    def test_authoritative_decisions_are_frozen(self):
        self.assertEqual(stage5a.read_json(stage5a.C01 / "RUN_MANIFEST.json")["family_decision"], "level3_no_primary_pass_stop")
        self.assertEqual(stage5a.read_json(stage5a.C02 / "RUN_MANIFEST.json")["decision"], "level3_no_primary_pass_stop")
        self.assertEqual(stage5a.read_json(stage5a.C03 / "PHASE_B_OMISSION_AUDIT.json")["reason"], "C03_PIT_authority_unavailable")

    def test_compact_metrics_preserves_all_frozen_definitions(self):
        rows = stage5a.compact_metrics("C01", stage5a.C01) + stage5a.compact_metrics("C02", stage5a.C02)
        self.assertEqual(len(rows), 20)
        self.assertFalse(any(row["all_level3_gates_pass"] == "True" for row in rows))

    def test_build_is_non_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "existing").write_text("x")
            with self.assertRaises(FileExistsError):
                stage5a.build(root)

    def test_actual_closed_package_passes_independent_validation(self):
        result = validator.validate(validator.DEFAULT_PACKAGE)
        self.assertEqual(result["status"], "approve")
        self.assertEqual(result["definition_rows_verified"], 20)


if __name__ == "__main__":
    unittest.main()
