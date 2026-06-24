import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import tools.run_qlmg_perp_project_reset as mod


class QlmgPerpProjectResetTests(unittest.TestCase):
    def test_run_root_suffixes_when_default_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / mod.DEFAULT_RUN_ID).mkdir()
            args = SimpleNamespace(run_root="", results_root=str(root), run_id=mod.DEFAULT_RUN_ID)
            out, reason = mod.run_root_from_args(args)
            self.assertNotEqual(out, root / mod.DEFAULT_RUN_ID)
            self.assertIn("default_root_existed_created_suffix", reason)

    def test_safe_cache_guard_refuses_durable_artifacts(self):
        ok, reason = mod.is_safe_cache_candidate(Path("results/rebaseline/some_manifest_cache"))
        self.assertFalse(ok)
        self.assertIn("not_allowlisted", reason)
        ok, reason = mod.is_safe_cache_candidate(Path("results/rebaseline/run/shared_cache"))
        self.assertTrue(ok)
        self.assertIn("allowlisted_cache_name", reason)
        ok, reason = mod.is_safe_cache_candidate(Path("results/rebaseline/run/ledger_cache"))
        self.assertFalse(ok)

    def test_data_inventory_handles_empty_one_minute_store(self):
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "1m"
            empty.mkdir()
            rows, summary = mod.scan_parquet_root(empty, "fixture_1m")
            self.assertEqual(rows, [])
            self.assertTrue(summary["exists"])
            self.assertEqual(summary["file_count"], 0)
            self.assertEqual(summary["symbol_count"], 0)

    def test_strategy_contracts_include_required_policy_fields(self):
        text = mod.contract_text("D1_low_volume_short_horizon_reversal", mod.STRATEGY_FAMILIES["D1_low_volume_short_horizon_reversal"])
        for token in [
            "direction:",
            "required_data:",
            "forbidden_shortcuts:",
            "validation_plan:",
            "promotion_gates:",
            "data_blockers:",
        ]:
            self.assertIn(token, text)

    def test_short_support_matrix_contains_funding_and_liquidation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod.stage_short_audit(root, SimpleNamespace())
            df = pd.read_csv(root / "shorts/short_support_matrix.csv")
            caps = set(df["capability"])
            self.assertIn("funding_sign_by_side", caps)
            self.assertIn("mark_price_liquidation_long_short", caps)

    def test_sealed_policy_is_draft_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            pd.DataFrame(
                [{"dataset": "fixture", "latest_timestamp": "2026-06-18 23:55:00+00:00"}]
            ).to_csv(root / "data/data_store_inventory.csv", index=False)
            mod.stage_sealed_policy(root, SimpleNamespace())
            registry = (root / "seal/sealed_registry_draft.json").read_text()
            self.assertIn('"status": "draft"', registry)
            self.assertIn("Phase 0", (root / "seal/new_sealed_policy.md").read_text())

    def test_metadata_probe_can_be_skipped_without_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod.stage_data_acquisition(root, SimpleNamespace(skip_metadata_probes=True, metadata_timeout=0.01))
            self.assertTrue((root / "data_acquisition/data_gap_matrix.csv").exists())
            cache = root / "data_acquisition/metadata_probe_cache"
            self.assertTrue(cache.exists())
            self.assertEqual(list(cache.glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
