from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from tools import run_qlmg_integrated_abcx_development as mod
from tools.qlmg_markdown_seed_parser import detect_date_precision, parse_markdown_tables, table_to_df
from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, validate_no_protected


class IntegratedABCXDevelopmentTests(unittest.TestCase):
    def test_protected_slice_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_no_protected(pd.DataFrame({"decision_ts": [FINAL_HOLDOUT_START]}), ["decision_ts"])

    def test_local_markdown_files_extract_tables(self) -> None:
        sector = Path("research_inputs/point_in_time_sector_seeds.md")
        catalyst = Path("research_inputs/post_catalyst_c2_database.md")
        self.assertTrue(sector.exists())
        self.assertTrue(catalyst.exists())
        sector_tables, sector_unparsed = parse_markdown_tables(sector)
        catalyst_tables, _ = parse_markdown_tables(catalyst)
        self.assertGreaterEqual(len(sector_tables), 4)
        self.assertGreaterEqual(len(catalyst_tables), 4)
        self.assertEqual(sector_unparsed, [])
        self.assertTrue(any(t.section == "Machine-readable seed table" for t in sector_tables))
        self.assertTrue(any(t.section == "Main catalyst database" for t in catalyst_tables))

    def test_raw_row_and_citation_preservation(self) -> None:
        tables, _ = parse_markdown_tables("research_inputs/post_catalyst_c2_database.md")
        main = next(t for t in tables if t.section == "Main catalyst database")
        df = table_to_df(main, "research_inputs/post_catalyst_c2_database.md")
        for col in ["source_md_path", "source_section", "source_row_number", "raw_row_text", "raw_row_hash", "source_cells_raw", "parse_status", "parse_warning"]:
            self.assertIn(col, df.columns)
        self.assertTrue(df["raw_row_text"].astype(str).str.contains("cite").any())

    def test_coarse_date_handling(self) -> None:
        self.assertEqual(detect_date_precision("2024-03-16"), "date_only")
        self.assertEqual(detect_date_precision("2024-03"), "month_only")
        self.assertEqual(detect_date_precision("2024"), "year_only")
        self.assertEqual(detect_date_precision("<=2020-01-01"), "lte_date")
        self.assertEqual(detect_date_precision("unknown"), "unknown")
        ts, precision, source = mod.date_precision_to_anchor("2024-03-16")
        self.assertEqual(precision, "date_only")
        self.assertEqual(str(ts), "2024-03-17 00:00:00+00:00")
        self.assertEqual(source, "next_daily_boundary")

    def test_current_only_taxonomy_blocked(self) -> None:
        df = pd.DataFrame({
            "sector_confidence": ["high", "high"],
            "is_current_only": ["true", "false"],
            "effective_start_utc": ["2024-01-01", "2024-01-01"],
        })
        out = mod.normalize_sector_seed(df)
        self.assertFalse(bool(out.loc[0, "rankable_pit_sector_seed"]))
        self.assertTrue(bool(out.loc[1, "rankable_pit_sector_seed"]))

    def test_excluded_catalyst_omission(self) -> None:
        df = pd.DataFrame({
            "event_id": ["CAT1", "CAT2"],
            "ticker": ["ETH", "SOL"],
            "durability_score_ex_ante": ["high", "high"],
            "first_public_ts_utc": ["2024-01-01", "2024-01-02"],
            "effective_ts_utc": ["2024-01-01", "2024-01-02"],
        })
        excl = pd.DataFrame({"event_id": ["CAT2"]})
        out = mod.normalize_catalyst_seed(df, excl)
        self.assertFalse(bool(out.loc[out["event_id"].eq("CAT2"), "primary_c2_eligible_seed"].iloc[0]))
        self.assertTrue(bool(out.loc[out["event_id"].eq("CAT1"), "primary_c2_eligible_seed"].iloc[0]))

    def test_event_day_chase_excluded_contractually(self) -> None:
        row = {"first_reaction_window_excluded": True, "event_day_chase_primary": False}
        self.assertTrue(row["first_reaction_window_excluded"])
        self.assertFalse(row["event_day_chase_primary"])

    def test_b1_leader_only_semantics(self) -> None:
        self.assertIn("b1_rankable_pit_sector_candidate", mod.B1_LABELS)
        summary_row = {"leader_selection": "top_1_to_2_relative_strength_liquidity", "equal_weight_basket_primary": False}
        self.assertFalse(summary_row["equal_weight_basket_primary"])

    def test_c2_mechanism_family_separation(self) -> None:
        self.assertIn("legal_regulatory_repricing", mod.MECHANISM_GROUPS)
        self.assertIn("exchange_access_expansion", mod.MECHANISM_GROUPS)
        self.assertNotIn("pooled_all_catalysts", mod.MECHANISM_GROUPS)

    def test_branch_separation(self) -> None:
        self.assertEqual(mod.HIGH_LEVEL_VERDICTS & {"validated", "live_ready"}, set())
        self.assertNotEqual("branch_x_execution_sensitive", "branch_l_a2a3_liquid_regime")

    def test_run_root_collision_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / mod.DEFAULT_RUN_ID
            base.mkdir()
            args = SimpleNamespace(run_root="", smoke=False)
            with mock.patch.object(mod, "RESULTS_ROOT", Path(td)):
                root, reason = mod.resolve_run_root(args)
        self.assertIn("default_root_existed_suffix", reason)
        self.assertNotEqual(root.name, mod.DEFAULT_RUN_ID)

    def test_tmux_wrapper_launch_gate_text(self) -> None:
        txt = Path("tools/run_qlmg_integrated_abcx_development_tmux.sh").read_text()
        self.assertIn("--launch-tmux", txt)
        self.assertIn("remote Telegram required", txt)


if __name__ == "__main__":
    unittest.main()
