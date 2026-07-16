import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools import run_qlmg_corrected_event_level_development_sweep as mod


class CorrectedEventLevelDevelopmentSweepTests(unittest.TestCase):
    def make_remediation_root(self) -> tempfile.TemporaryDirectory:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / "gate").mkdir(parents=True)
        (root / "quarantine").mkdir(parents=True)
        (root / "gate/corrected_sweep_allowed.json").write_text(json.dumps({
            "corrected_sweep_allowed": True,
            "allowed_families": ["A3", "A2_redesign_only"],
            "blockers": [],
        }))
        pd.DataFrame([{"candidate_id": "a3", "family": "A3", "rankable": True}]).to_csv(root / "quarantine/rankable_active_evidence_set.csv", index=False)
        pd.DataFrame([{"artifact": "bad_summary.csv", "reason": "summary_projection_only"}]).to_csv(root / "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv", index=False)
        pd.DataFrame([
            {"deprecated_label": "research_prelead_only"},
            {"deprecated_label": "stress_survives"},
            {"deprecated_label": "targeted_execution_data_prelead"},
            {"deprecated_label": "targeted_execution_data_prelead_unresolved"},
            {"deprecated_label": "a2_a3_tier1_prelead_confirmed_train_only"},
        ]).to_csv(root / "quarantine/deprecated_promotion_labels.csv", index=False)
        return tmp

    def test_protected_slice_rejected(self):
        df = pd.DataFrame({"decision_ts": ["2026-01-01T00:00:00Z"]})
        with self.assertRaises(Exception):
            mod.validate_no_protected_df(df, ["decision_ts"])

    def test_missing_remediation_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(mod, "REMEDIATION_ROOT", Path(tmp)):
                with self.assertRaises(RuntimeError):
                    mod.read_required_gate()

    def test_rankable_gate_requires_allowed_families(self):
        tmp = self.make_remediation_root()
        self.addCleanup(tmp.cleanup)
        with patch.object(mod, "REMEDIATION_ROOT", Path(tmp.name)):
            gate = mod.read_required_gate()
        self.assertEqual(set(gate["allowed_families"]), {"A3", "A2_redesign_only"})

    def test_rankable_scope_excludes_forbidden_families(self):
        self.assertEqual(mod.RANKABLE_FAMILIES, {"A3", "A2_redesign_only"})
        self.assertIn("D4", mod.FORBIDDEN_RANKABLE_FAMILIES)
        self.assertIn("listing", mod.FORBIDDEN_RANKABLE_FAMILIES)
        self.assertNotIn("D4", mod.RANKABLE_FAMILIES)
        self.assertNotIn("listing", mod.RANKABLE_FAMILIES)

    def test_quarantined_deprecated_evidence_blocking_registry(self):
        tmp = self.make_remediation_root()
        self.addCleanup(tmp.cleanup)
        with patch.object(mod, "REMEDIATION_ROOT", Path(tmp.name)):
            args = mod.parse_args(["--run-root", str(Path(tmp.name) / "run"), "--disable-telegram"])
            root, _ = mod.resolve_run_root(args)
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
            mod.stage_rankable_gate(ctx)
            obj = json.loads((root / "gate/rankable_source_gate.json").read_text())
        self.assertTrue(obj["rankable_source_gate_passed"])
        self.assertEqual(obj["deprecated_label_count"], 5)

    def test_synthetic_control_helper_hard_disabled(self):
        df = pd.DataFrame({"net_R_variant": [1.0, -0.5]})
        with self.assertRaisesRegex(RuntimeError, "Deprecated synthetic control generator"):
            mod.control_rows_for(df, "candidate", "variant", 1, 3)

    def test_rankable_gate_blocks_forbidden_rankable_family_rows(self):
        tmp = self.make_remediation_root()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pd.DataFrame([
            {"candidate_id": "a3", "family": "A3", "rankable": True},
            {"candidate_id": "f1", "family": "F1", "rankable": True},
        ]).to_csv(root / "quarantine/rankable_active_evidence_set.csv", index=False)
        with patch.object(mod, "REMEDIATION_ROOT", root):
            args = mod.parse_args(["--run-root", str(root / "run"), "--disable-telegram"])
            run_root, _ = mod.resolve_run_root(args)
            ctx = mod.RunContext(args=args, run_root=run_root, notifier=mod.RunNotifier(run_root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
            with self.assertRaisesRegex(RuntimeError, "forbidden_families"):
                mod.stage_rankable_gate(ctx)

    def test_split_dev_eval_has_no_overlap(self):
        df = pd.DataFrame({
            "decision_ts": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
            "value": range(10),
        })
        dev, eval_df, cutoff = mod.split_dev_eval(df, 0.6)
        self.assertLess(dev["decision_ts"].max(), eval_df["decision_ts"].min())
        self.assertLessEqual(dev["decision_ts"].max(), cutoff)

    def test_a3_candidate_grid_is_rankable_family_only(self):
        rows = mod.candidate_grid("A3", 10, smoke=True)
        self.assertTrue(rows)
        self.assertEqual({r["family"] for r in rows}, {"A3"})

    def test_a2_candidate_grid_is_rankable_family_only(self):
        rows = mod.candidate_grid("A2_redesign_only", 10, smoke=True)
        self.assertTrue(rows)
        self.assertEqual({r["family"] for r in rows}, {"A2_redesign_only"})

    def test_a2_future_liquidation_flag_not_rankable_filter(self):
        base = pd.DataFrame({
            "candidate_id": ["a", "b"],
            "family": ["A2", "A2"],
            "decision_ts": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
            "net_R": [1.0, -0.5],
            "risk_bps_used": [100.0, 100.0],
            "liquidation_flag": [True, False],
        })
        cfg = {"family": "A2_redesign_only", "variant_id": "a2_test", "risk_bps_max": None, "hold_factor": 1.0, "target_cap_R": None, "extra_cost_bps": 0.0}
        out = mod.variant_event_rows(base, cfg)
        self.assertEqual(len(out), 2)
        self.assertTrue(out["liquidation_flag"].any())

    def test_a3_fragility_gate_label(self):
        frag = pd.DataFrame({
            "test": ["remove_top_1pct_winners", "remove_top_5pct_winners", "fee_slippage_plus_25bps"],
            "survives_positive": [True, False, True],
        })
        label = mod.label_a3({"net_R": 10.0, "PF": 1.2, "symbols": 3, "months": 4}, frag, controls_pass=True)
        self.assertEqual(label, "a3_fragile_but_alive")

    def test_normalized_controls_scale_to_candidate_event_count(self):
        self.assertAlmostEqual(mod.normalize_controls(10.0, 100, -30.0, 300), -10.0)

    def test_stage_nulls_ingests_real_controls_and_relabels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            controls_root = Path(tmp) / "real_controls"
            (controls_root / "controls").mkdir(parents=True)
            pd.DataFrame([
                {
                    "candidate_id": "A3__x",
                    "family": "A3",
                    "control_type": "same_symbol",
                    "control_event_count": 2,
                    "beats_control": True,
                    "control_coverage_ratio": 1.0,
                    "all_control_rows_have_source_ids": True,
                    "match_basis": "same symbol",
                },
                {
                    "candidate_id": "A3__x",
                    "family": "A3",
                    "control_type": "same_regime",
                    "control_event_count": 2,
                    "beats_control": True,
                    "control_coverage_ratio": 1.0,
                    "all_control_rows_have_source_ids": True,
                    "match_basis": "same regime",
                },
            ]).to_csv(controls_root / "controls/real_control_summary.csv", index=False)
            pd.DataFrame([{"control_event_id": "c1"}]).to_parquet(controls_root / "controls/real_control_event_ledger.parquet", index=False)
            (root / "a3_sweep").mkdir(parents=True)
            pd.DataFrame([{
                "candidate_id": "A3__x",
                "variant_id": "A3__x",
                "family": "A3",
                "events": 10,
                "net_R": 5.0,
                "PF": 1.2,
                "symbols": 3,
                "months": 4,
                "pre_null_label": "path_edge_exit_problem",
            }]).to_csv(root / "a3_sweep/a3_sweep_summary.csv", index=False)
            pd.DataFrame([
                {"variant_id": "A3__x", "test": "remove_top_1pct_winners", "survives_positive": True},
                {"variant_id": "A3__x", "test": "remove_top_5pct_winners", "survives_positive": True},
                {"variant_id": "A3__x", "test": "fee_slippage_plus_25bps", "survives_positive": True},
            ]).to_csv(root / "a3_sweep/a3_fragility_summary.csv", index=False)
            (root / "a2_sweep").mkdir(parents=True)
            pd.DataFrame().to_csv(root / "a2_sweep/a2_sweep_summary.csv", index=False)
            pd.DataFrame().to_csv(root / "a2_sweep/a2_tail_dependence_summary.csv", index=False)
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram", "--real-controls-root", str(controls_root)])
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
            mod.stage_nulls(ctx)
            out = pd.read_csv(root / "nulls/fresh_null_summary.csv")
            self.assertFalse(out["placeholder_controls_used"].astype(bool).any())
            relabel = pd.read_csv(root / "a3_sweep/a3_sweep_summary.csv")
            self.assertTrue(relabel["real_controls_pass_all_types"].astype(bool).iloc[0])

    def test_metric_summary_requires_event_level_input_for_metrics(self):
        df = pd.DataFrame({
            "decision_ts": pd.to_datetime(["2025-01-01", "2025-02-01"], utc=True),
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "net_R_variant": [1.0, -0.5],
            "mark_price_available": [False, False],
            "funding_exact": [False, False],
        })
        row = mod.metric_summary(df, "cid", "A3", "v1")
        self.assertEqual(row["events"], 2)
        self.assertFalse(row["mark_available"])
        self.assertFalse(row["funding_exact"])
        self.assertTrue(row["mark_proxy_used"])
        self.assertTrue(row["funding_proxy_used"])
        self.assertEqual(row["label_cap_reason"], "mark_or_funding_proxy_cap")

    def test_b1_symbol_parser_handles_lists_and_strings(self):
        self.assertEqual(mod.parse_symbols("BTCUSDT, ETHUSDT; SOLUSDT"), ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.assertEqual(mod.parse_symbols(["BTCUSDT", "ETHUSDT"]), ["BTCUSDT", "ETHUSDT"])

    def test_c2_mechanism_separation_bucket(self):
        self.assertEqual(mod.c2_bucket("ETF approval and institutional access"), "ETF/institutional access")
        self.assertEqual(mod.c2_bucket("unlock supply expansion"), "supply/unlock/float")
        self.assertEqual(mod.c2_bucket("exchange listing access"), "exchange access")

    def test_wrapper_launch_and_telegram_gates(self):
        text = Path("tools/run_qlmg_corrected_event_level_development_sweep_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("remote Telegram required", text)
        self.assertIn("smoke first", text)


if __name__ == "__main__":
    unittest.main()
