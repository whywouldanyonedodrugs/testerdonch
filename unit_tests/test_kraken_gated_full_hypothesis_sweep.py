from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tools import run_kraken_gated_full_hypothesis_sweep as sweep


class KrakenGatedFullHypothesisSweepTests(unittest.TestCase):
    def test_manual_resolution_prefers_docs_then_research_then_testmanual(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_repo = sweep.REPO
            try:
                sweep.REPO = root
                research = root / "research_inputs"
                docs = root / "docs"
                research.mkdir()
                docs.mkdir()
                (research / "testmanual.txt").write_text("no-vendor mechanical qa evidence contract")
                (research / "QLMG_BACKTESTING_MANUAL_20260630_FULL.md").write_text("no-vendor mechanical qa evidence contract research")
                (docs / "QLMG_BACKTESTING_MANUAL_20260630_FULL.md").write_text("no-vendor mechanical qa evidence contract docs")
                res = sweep.resolve_manual(research)
                self.assertTrue(res["selected_path"].endswith("docs/QLMG_BACKTESTING_MANUAL_20260630_FULL.md"))
            finally:
                sweep.REPO = old_repo

    def test_semantic_fail_excluded_without_repair(self):
        with tempfile.TemporaryDirectory() as td:
            rr = Path(td)
            ready = rr / "ready"
            repair = rr / "repair"
            (ready / "compiler/compiled_contracts").mkdir(parents=True)
            (repair / "viability").mkdir(parents=True)
            pd.DataFrame([{ "hypothesis_id": "H15", "family": "Funding", "current_lane": "kraken_tier1_ready"}, {"hypothesis_id":"H03", "family":"Liquid continuation / Momentum", "current_lane":"kraken_tier1_ready"}]).to_csv(repair / "viability/hypothesis_viability_matrix.csv", index=False)
            pd.DataFrame([{ "hypothesis_id": "H15", "semantic_status": "fail", "issue": "touch_fill_entry_not_allowed_tier1"}, {"hypothesis_id":"H03", "semantic_status":"pass", "issue":""}]).to_csv(ready / "compiler/semantic_sanity_checks.csv", index=False)
            pd.DataFrame([{ "hypothesis_id": "H15"}, {"hypothesis_id":"H03"}]).to_csv(ready / "compiler/hypothesis_to_contract_trace.csv", index=False)
            (ready / "compiler/compiled_contracts/kraken__H15__x.json").write_text('{"hypothesis_id":"H15","contract_id":"c15","family":"Funding"}')
            (ready / "compiler/compiled_contracts/kraken__H03__x.json").write_text('{"hypothesis_id":"H03","contract_id":"c03","family":"Liquid continuation / Momentum"}')
            ctx = SimpleNamespace(args=SimpleNamespace(readiness_root=str(ready), repair_root=str(repair)), run_root=rr / "run")
            ctx.run_root.mkdir()
            (ctx.run_root / "scope/frozen_contracts").mkdir(parents=True)
            sweep.stage_scope(ctx)
            side = pd.read_csv(ctx.run_root / "scope/sidecar_scope_manifest.csv")
            self.assertEqual(side.iloc[0]["rankable"], False)
            self.assertIn("semantic_fail", side.iloc[0]["scope_reason"])

    def test_repaired_contract_overrides_semantic_fail_when_tier1_rankable(self):
        with tempfile.TemporaryDirectory() as td:
            rr = Path(td)
            ready = rr / "ready"
            repair = rr / "repair"
            (ready / "compiler/compiled_contracts").mkdir(parents=True)
            (repair / "viability").mkdir(parents=True)
            (repair / "compile_repair/repaired_contracts").mkdir(parents=True)
            pd.DataFrame([{"hypothesis_id": "PD06", "family": "Funding", "current_lane": "kraken_tier1_with_caps"}]).to_csv(repair / "viability/hypothesis_viability_matrix.csv", index=False)
            pd.DataFrame([{"hypothesis_id": "PD06", "semantic_status": "fail", "issue": "microstructure_hypothesis_not_tier1_rankable"}]).to_csv(ready / "compiler/semantic_sanity_checks.csv", index=False)
            pd.DataFrame([{"hypothesis_id": "PD06"}]).to_csv(ready / "compiler/hypothesis_to_contract_trace.csv", index=False)
            (ready / "compiler/compiled_contracts/kraken__PD06__x.json").write_text('{"hypothesis_id":"PD06","contract_id":"old","family":"Funding"}')
            (repair / "compile_repair/repaired_contracts/kraken_repair__PD06.json").write_text('{"hypothesis_id":"PD06","contract_id":"repair","family":"Funding","tier1_rankable":true,"lane":"compiled_tier1_with_analytics_cap"}')
            ctx = SimpleNamespace(args=SimpleNamespace(readiness_root=str(ready), repair_root=str(repair)), run_root=rr / "run")
            ctx.run_root.mkdir()
            (ctx.run_root / "scope/frozen_contracts").mkdir(parents=True)
            sweep.stage_scope(ctx)
            rank = pd.read_csv(ctx.run_root / "scope/rankable_scope_manifest.csv")
            self.assertEqual(len(rank), 1)
            self.assertEqual(rank.iloc[0]["contract_source"], "repair")

    def test_funding_no_cross_is_exact_zero(self):
        bars = pd.DataFrame({
            "ts": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:05:00Z", "2025-01-01T00:10:00Z", "2025-01-01T00:15:00Z"], utc=True),
            "open": [100, 101, 102, 103], "high": [101, 102, 103, 104], "low": [99, 100, 101, 102], "close": [100.5, 101.5, 102.5, 103.5],
            "mark_close": [100.5, 101.5, 102.5, 103.5], "mark_high": [101, 102, 103, 104], "mark_low": [99, 100, 101, 102],
        })
        cand = {"candidate_id": "c", "hypothesis_id": "h", "family": "f", "symbol": "PF_XBTUSD", "side": "long", "hold_bars": 1, "stop_bps": 100, "archetype": "liquid_continuation"}
        ev = sweep.event_from_signal(cand, bars, pd.DataFrame(), 0, 1)
        self.assertIsNotNone(ev)
        self.assertFalse(ev["funding_boundary_crossed"])
        self.assertTrue(ev["funding_exact"])
        self.assertEqual(ev["funding_R"], 0.0)

    def test_funding_cross_missing_caps_and_adverse_proxy(self):
        times = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
        bars = pd.DataFrame({"ts": times, "open": 100, "high": 101, "low": 99.5, "close": 100.5, "mark_close": 100.5, "mark_high": 101, "mark_low": 99.5})
        cand = {"candidate_id": "c", "hypothesis_id": "h", "family": "f", "symbol": "PF_XBTUSD", "side": "long", "hold_bars": 15, "stop_bps": 100, "archetype": "liquid_continuation"}
        ev = sweep.event_from_signal(cand, bars, pd.DataFrame(), 0, 1)
        self.assertTrue(ev["funding_boundary_crossed"])
        self.assertFalse(ev["funding_exact"])
        self.assertTrue(ev["funding_proxy_used"])
        self.assertIn("funding_missing_adverse_proxy", ev["label_cap_reason"])

    def test_real_controls_have_required_ids_and_purge_flag(self):
        events = pd.DataFrame([
            {"event_id":"e1","candidate_id":"c1","symbol":"S1","decision_ts":"2025-01-01T00:00:00Z","entry_ts":"2025-01-01T00:05:00Z","exit_ts":"2025-01-01T01:00:00Z","net_R":1.0,"signal_template":"a","family":"F","risk_bps_used":100,"funding_boundary_crossed":False},
            {"event_id":"e2","candidate_id":"c2","symbol":"S1","decision_ts":"2025-01-03T00:00:00Z","entry_ts":"2025-01-03T00:05:00Z","exit_ts":"2025-01-03T01:00:00Z","net_R":-0.5,"signal_template":"a","family":"F","risk_bps_used":100,"funding_boundary_crossed":False},
            {"event_id":"e3","candidate_id":"c3","symbol":"S2","decision_ts":"2025-01-04T00:00:00Z","entry_ts":"2025-01-04T00:05:00Z","exit_ts":"2025-01-04T01:00:00Z","net_R":0.2,"signal_template":"b","family":"F","risk_bps_used":200,"funding_boundary_crossed":True},
        ])
        ledger, summary = sweep.build_controls(events, 1, 1)
        self.assertFalse(ledger.empty)
        for col in ["control_event_id", "source_window_id", "matching_basis", "purge_embargo_passed"]:
            self.assertIn(col, ledger.columns)

    def test_event_level_summary_audit_recomputes_counts_and_net_r(self):
        events = pd.DataFrame([
            {"event_id":"e1","candidate_id":"c1","hypothesis_id":"h","family":"F","symbol":"S1","decision_ts":"2025-01-01T00:00:00Z","entry_ts":"2025-01-01T00:05:00Z","exit_ts":"2025-01-01T00:10:00Z","net_R":1.0,"funding_proxy_used":False,"mark_proxy_used":False},
            {"event_id":"e2","candidate_id":"c1","hypothesis_id":"h","family":"F","symbol":"S2","decision_ts":"2025-02-01T00:00:00Z","entry_ts":"2025-02-01T00:05:00Z","exit_ts":"2025-02-01T00:10:00Z","net_R":-0.25,"funding_proxy_used":False,"mark_proxy_used":False},
        ])
        summary = sweep.summarize_events(pd.DataFrame(events))
        audit = sweep.event_level_summary_audit(events, summary)
        self.assertTrue((audit["status"] == "pass").all())

    def test_coarse_gate_blocks_one_event_standard_but_preserves_sparse_sleeve(self):
        summary = pd.DataFrame([
            {"candidate_id":"std","hypothesis_id":"h1","family":"Liquid continuation / Momentum","side":"long","events":1,"active_symbols":1,"active_months":1,"net_R":5.0,"PF":99.0,"median_R":5.0,"trimmed_mean_R":5.0,"max_dd_R":0.0,"regime_activation":"r","data_cap":"none"},
            {"candidate_id":"sparse","hypothesis_id":"h2","family":"Catalyst / Lifecycle / Event","side":"long","events":1,"active_symbols":1,"active_months":1,"net_R":1.0,"PF":99.0,"median_R":1.0,"trimmed_mean_R":1.0,"max_dd_R":0.0,"regime_activation":"r","data_cap":"none"},
        ])
        thresholds = sweep.coarse_thresholds_for_summary(summary, smoke=False)
        coarse = sweep.coarse_screen_candidates(summary, thresholds, top_per_family=5)
        labels = dict(zip(coarse["candidate_id"], coarse["coarse_status"]))
        self.assertEqual(labels["std"], "coarse_rejected_current_translation_only")
        self.assertEqual(labels["sparse"], "sparse_sleeve_needs_more_evidence")

    def test_control_subwave_budget_preserves_unprocessed_survivors(self):
        with tempfile.TemporaryDirectory() as td:
            candidates = pd.DataFrame([
                {"candidate_id":f"c{i}","family":"F","regime_activation":"r","net_R":float(i),"PF":1.2}
                for i in range(5)
            ])
            ctx = SimpleNamespace(args=SimpleNamespace(max_control_candidates_per_subwave=2, max_control_runtime_hours_per_subwave=0.000001, nulls_per_event=3))
            kept, unprocessed, manifest = sweep.plan_control_subwaves(candidates, ctx, wave=1)
            self.assertTrue(kept.empty)
            self.assertEqual(len(unprocessed), 5)
            self.assertTrue((unprocessed["coarse_status"] == "needs_controls_after_coarse_screen_due_resource_budget").all())
            self.assertFalse(manifest["will_process"].any())

    def test_candidate_limited_controls_do_not_process_all_candidates(self):
        events = pd.DataFrame([
            {"event_id":"e1","candidate_id":"c1","symbol":"S1","decision_ts":"2025-01-01T00:00:00Z","entry_ts":"2025-01-01T00:05:00Z","exit_ts":"2025-01-01T01:00:00Z","net_R":1.0,"signal_template":"a","family":"F","risk_bps_used":100,"funding_boundary_crossed":False},
            {"event_id":"e2","candidate_id":"c2","symbol":"S1","decision_ts":"2025-01-03T00:00:00Z","entry_ts":"2025-01-03T00:05:00Z","exit_ts":"2025-01-03T01:00:00Z","net_R":-0.5,"signal_template":"a","family":"F","risk_bps_used":100,"funding_boundary_crossed":False},
            {"event_id":"e3","candidate_id":"c3","symbol":"S2","decision_ts":"2025-01-04T00:00:00Z","entry_ts":"2025-01-04T00:05:00Z","exit_ts":"2025-01-04T01:00:00Z","net_R":0.2,"signal_template":"b","family":"F","risk_bps_used":200,"funding_boundary_crossed":True},
        ])
        _, summary = sweep.build_controls(events, 1, 1, candidate_ids=["c1"])
        self.assertEqual(set(summary["candidate_id"]), {"c1"})

    def test_coarse_gate_bias_threshold_detects_rejected_control_passes(self):
        sample = pd.DataFrame([{"candidate_id":f"c{i}","family":"F"} for i in range(10)])
        rows = []
        for i in range(10):
            for ctype in ["same_symbol", "same_regime"]:
                rows.append({"candidate_id":f"c{i}","control_type":ctype,"beats_control": i == 0, "control_uplift_R": 1.0 if i == 0 else -1.0, "control_coverage_ratio":1.0})
        payload, _, _ = sweep.control_bias_audit(pd.DataFrame(rows), sample)
        self.assertEqual(payload["status"], "coarse_gate_bias_detected")

    def test_negative_reject_beating_worse_controls_is_not_material_bias(self):
        sample = pd.DataFrame([
            {"candidate_id":"c1","family":"F","net_R":-1.0,"PF":0.5,"median_R":-0.1,"trimmed_mean_R":-0.1},
            {"candidate_id":"c2","family":"F","net_R":-0.5,"PF":0.7,"median_R":-0.1,"trimmed_mean_R":-0.1},
        ])
        rows = []
        for cid in ["c1", "c2"]:
            for ctype in ["same_symbol", "same_regime"]:
                rows.append({"candidate_id":cid,"control_type":ctype,"beats_control": True, "control_uplift_R": 0.1, "control_coverage_ratio":1.0})
        payload, audit, _ = sweep.control_bias_audit(pd.DataFrame(rows), sample)
        self.assertEqual(payload["status"], "pass")
        self.assertFalse(audit["audit_control_pass"].any())

    def test_tmux_wrapper_requires_launch(self):
        txt = Path("tools/run_kraken_gated_full_hypothesis_sweep_tmux.sh").read_text()
        self.assertIn("refusing to launch tmux without --launch-tmux", txt)
        self.assertIn("run_kraken_gated_full_hypothesis_sweep.py", txt)

    def test_forbidden_language_regex(self):
        self.assertIsNotNone(sweep.FORBIDDEN_WORDS.search("validated"))
        self.assertIsNone(sweep.FORBIDDEN_WORDS.search("train-only screen survivor"))


if __name__ == "__main__":
    unittest.main()
