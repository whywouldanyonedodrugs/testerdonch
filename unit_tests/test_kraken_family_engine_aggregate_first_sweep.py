from __future__ import annotations

import tempfile
import unittest
import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tools import run_kraken_family_engine_aggregate_first_sweep as sweep
from tools import kraken_aggregate_cache_layer as cache_layer


class KrakenFamilyEngineAggregateFirstSweepTests(unittest.TestCase):
    def bars(self, n: int = 600) -> pd.DataFrame:
        ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
        px = pd.Series([100 + i * 0.03 + (i % 17) * 0.01 for i in range(n)], dtype=float)
        return pd.DataFrame({
            "ts": ts,
            "open": px,
            "high": px + 0.8,
            "low": px - 0.8,
            "close": px + 0.2,
            "volume": 1000.0,
            "mark_close": px + 0.1,
            "mark_high": px + 0.7,
            "mark_low": px - 0.7,
            "symbol": "PF_XBTUSD",
        })

    def candidate(self, engine_id: str = "liquid_continuation_breakout_engine", archetype: str = "liquid_continuation") -> dict:
        return {
            "candidate_id": f"c_{engine_id}",
            "definition_id": f"d_{engine_id}",
            "hypothesis_id": "H01",
            "family": "Liquid continuation / Momentum",
            "family_engine_id": engine_id,
            "archetype": archetype,
            "symbol": "PF_XBTUSD",
            "side": "long",
            "lookback_bars": 12,
            "hold_bars": 24,
            "stop_bps": 100.0,
            "threshold": 0.001,
            "entry_template": "close_confirmed",
            "exit_template": "fixed_hold",
            "stop_template": "fixed_bps",
            "regime_activation": "all_context_diagnostic",
            "symbol_universe_hash": "u",
            "parameter_vector_hash": "p",
        }

    def test_all_required_family_engines_expose_methods(self):
        expected = {
            "scheduled_tsmom_engine",
            "liquid_continuation_breakout_engine",
            "prior_high_reclaim_engine",
            "retest_reclaim_lifecycle_engine",
            "compression_breakout_engine",
            "session_calendar_engine",
            "funding_crowding_engine",
        }
        self.assertEqual(set(sweep.ENGINES), expected)
        for engine in sweep.ENGINES.values():
            for method in [
                "build_candidate_masks",
                "enumerate_valid_event_addresses",
                "compute_exact_aggregate_metrics",
                "regenerate_materialized_ledger",
                "build_matched_controls",
                "run_family_specific_stress",
            ]:
                self.assertTrue(callable(getattr(engine, method)))
            self.assertFalse("infer_signal_indices" in engine.enumerate_valid_event_addresses.__qualname__)

    def test_aggregate_summary_equals_materialized_ledger(self):
        bars = self.bars()
        funding = pd.DataFrame()
        for engine_id, archetype in sweep.ENGINE_ARCHETYPES.items():
            cand = self.candidate(engine_id, archetype)
            engine = sweep.ENGINES[engine_id]
            agg = engine.compute_exact_aggregate_metrics(bars, funding, cand)
            mat = engine.regenerate_materialized_ledger(bars, funding, cand)
            recomputed = sweep.summarize_materialized_candidate(mat, cand, engine)
            self.assertEqual(int(agg["events"]), int(recomputed["events"]), engine_id)
            self.assertAlmostEqual(float(agg["net_R"]), float(recomputed["net_R"]), places=10, msg=engine_id)
            self.assertEqual(agg["event_sampling_used"], False)
            self.assertEqual(agg["aggregate_metric_basis"], "exact_from_all_events")

    def test_event_address_reproducibility_fields_can_be_written(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(kraken_data_root=str(root), smoke=True, chunk_size=10, seed=1), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-01-02", tz="UTC"))
            # Directly write a minimal manifest row using the public schema expectation.
            row = {
                "candidate_definition_id": "d1",
                "family_engine_id": "scheduled_tsmom_engine",
                "engine_contract_hash": sweep.ENGINES["scheduled_tsmom_engine"].engine_contract_hash(),
                "event_semantics_version": sweep.EVENT_SEMANTICS_VERSION,
                "row_semantics": sweep.ENGINES["scheduled_tsmom_engine"].row_semantics(),
                "symbol_universe_hash": "u",
                "time_window_hash": "t",
                "parameter_vector_hash": "p",
                "event_address_generation_rule": "scheduled_tsmom_engine.enumerate_valid_event_addresses",
                "feature_panel_manifest_hash": "f",
            }
            sweep.write_csv(root / "audit/event_address_reproducibility_manifest.csv", [row])
            out = pd.read_csv(root / "audit/event_address_reproducibility_manifest.csv")
            self.assertEqual(out.iloc[0]["event_semantics_version"], sweep.EVENT_SEMANTICS_VERSION)
            self.assertIn("enumerate_valid_event_addresses", out.iloc[0]["event_address_generation_rule"])

    def test_protected_timestamp_event_is_rejected(self):
        bars = self.bars(20)
        bars["ts"] = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
        cand = self.candidate()
        engine = sweep.ENGINES["liquid_continuation_breakout_engine"]
        ledger = engine.regenerate_materialized_ledger(bars, pd.DataFrame(), cand)
        self.assertTrue(ledger.empty)

    def test_protected_dated_chunks_are_skipped_before_read(self):
        files = [
            Path("/tmp/PF_XBTUSD/20251231T000000_before.parquet"),
            Path("/tmp/PF_XBTUSD/20260101T000000_protected.parquet"),
            Path("/tmp/PF_XBTUSD/20260110T000000_after.parquet"),
            Path("/tmp/PF_XBTUSD/no_timestamp.parquet"),
        ]
        kept = [p.name for p in sweep.pre_holdout_files(files)]
        self.assertIn("20251231T000000_before.parquet", kept)
        self.assertIn("no_timestamp.parquet", kept)
        self.assertNotIn("20260101T000000_protected.parquet", kept)
        self.assertNotIn("20260110T000000_after.parquet", kept)

    def test_row_semantics_invalid_metric_not_zero_for_lifecycle(self):
        bars = self.bars()
        cand = self.candidate("retest_reclaim_lifecycle_engine", "retest_reclaim")
        engine = sweep.ENGINES["retest_reclaim_lifecycle_engine"]
        agg = engine.compute_exact_aggregate_metrics(bars, pd.DataFrame(), cand)
        self.assertEqual(agg["row_semantics"], "lifecycle_event")
        self.assertTrue(pd.isna(agg["PF"]))
        self.assertTrue(pd.isna(agg["win_rate"]))

    def test_adaptive_materialization_budget_caps_initial_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "coarse").mkdir(parents=True)
            rows = []
            for i in range(600):
                rows.append({"candidate_id": f"c{i}", "definition_id": f"d{i}", "family_engine_id": "scheduled_tsmom_engine", "family": "F", "events": 10, "net_R": float(i), "median_R": 0.1, "row_semantics": "position_interval"})
            pd.DataFrame(rows).to_parquet(root / "coarse/all_candidate_aggregate_summary.parquet", index=False)
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(materialization_budget=1000, audit_sample_budget=300, smoke=False, seed=7))
            sweep.stage_selection(ctx)
            selected = pd.read_csv(root / "selection/materialization_selection.csv")
            audit = pd.read_csv(root / "selection/audit_sample_selection.csv")
            self.assertLessEqual(len(selected), 250)
            self.assertLessEqual(len(audit), 150)
            report = (root / "selection/materialization_budget_adaptation_report.md").read_text()
            self.assertIn("pending_materialization", report)

    def test_controls_are_materialized_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "materialized").mkdir(parents=True)
            sweep.write_csv(root / "materialized/materialized_ledger_manifest.csv", [{"candidate_id": "c1", "path": "materialized/missing.parquet", "event_rows": 0}])
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(nulls_per_event=1, seed=1))
            sweep.stage_controls(ctx)
            pre = pd.read_csv(root / "controls/control_materialization_precondition_report.csv")
            self.assertFalse(bool(pre.iloc[0]["controls_allowed"]))
            self.assertIn("materialized", (root / "controls/control_audit_report.md").read_text())

    def test_benchmark_cached_aggregate_loads_symbol_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "benchmark").mkdir(parents=True)
            (root / "audit").mkdir(parents=True)
            calls = {"bars": 0}
            old_bars = sweep.load_symbol_bars
            old_funding = sweep.load_funding
            try:
                def fake_bars(paths, symbol, start, end):
                    calls["bars"] += 1
                    return self.bars(120)
                sweep.load_symbol_bars = fake_bars
                sweep.load_funding = lambda paths, symbol, end: pd.DataFrame()
                ctx = SimpleNamespace(
                    run_root=root,
                    start=pd.Timestamp("2025-01-01", tz="UTC"),
                    end=pd.Timestamp("2025-01-02", tz="UTC"),
                    args=SimpleNamespace(kraken_data_root=str(root)),
                )
                candidates = pd.DataFrame([
                    self.candidate("liquid_continuation_breakout_engine", "liquid_continuation"),
                    self.candidate("scheduled_tsmom_engine", "tsmom"),
                ])
                candidates["symbol"] = "PF_XBTUSD"
                out = sweep.aggregate_benchmark_candidates_cached(ctx, candidates)
                self.assertEqual(calls["bars"], 1)
                self.assertEqual(len(out), 2)
                self.assertTrue((root / "benchmark/engine_benchmark_progress.json").exists())
            finally:
                sweep.load_symbol_bars = old_bars
                sweep.load_funding = old_funding

    def test_tmux_wrapper_requires_launch_and_uses_new_runner(self):
        txt = Path("tools/run_kraken_family_engine_aggregate_first_sweep_tmux.sh").read_text()
        self.assertIn("refusing to launch tmux without --launch-tmux", txt)
        self.assertIn("run_kraken_family_engine_aggregate_first_sweep.py", txt)
        self.assertIn("phase_kraken_engine_wave_v0_tranche_tsmom_retest_p1_20260704_v1", txt)

    def test_compact_stage_writes_final_complete_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "notifications").mkdir(parents=True)
            sweep.write_json(root / "decision_summary.json", {"status": "complete"})
            sweep.write_text(root / "KRAKEN_FAMILY_ENGINE_AGGREGATE_FIRST_REPAIR_REPORT.md", "# Report\n")
            ctx = SimpleNamespace(run_root=root)
            sweep.stage_compact(ctx)
            watch = json.loads((root / "watch_status.json").read_text())
            marker = json.loads((root / "notifications/final_completion_marker.json").read_text())
            self.assertEqual(watch["status"], "complete")
            self.assertEqual(watch["stage"], "compact-review-bundle")
            self.assertEqual(marker["status"], "complete")

    def test_benchmark_gate_contradiction_blocks_go(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prior = root / "prior"
            run = root / "run"
            (prior / "benchmark").mkdir(parents=True)
            (prior / "budget").mkdir()
            (prior / "seal").mkdir()
            (run / "gate").mkdir(parents=True)
            sweep.write_json(prior / "benchmark/full_budget_projection_analysis.json", {
                "projected_aggregate_evaluation_hours": 5004.0,
                "projected_runtime_bucket_from_actual_budget": "above_168_hours",
                "engines_with_no_planned_definitions": ["compression_breakout_engine"],
            })
            sweep.write_text(prior / "benchmark/operator_go_no_go_after_engine_benchmark.md", "Status: `go`\n")
            sweep.write_csv(prior / "benchmark/aggregate_materialized_probe_audit.csv", [{"status": "pass"}])
            sweep.write_csv(prior / "budget/candidate_definition_budget_manifest.csv", [{"family_engine_id": "scheduled_tsmom_engine"}])
            sweep.write_json(prior / "seal/protected_timestamp_scan_classification.json", {"scoring_or_event_artifact_protected_timestamp_violation_observed": False})
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(prior_benchmark_root=str(prior)))
            sweep.stage_benchmark_gate_repair(ctx)
            rules = json.loads((run / "gate/repaired_benchmark_gate_rules.json").read_text())
            self.assertEqual(rules["gate_status"], "no_go")
            self.assertTrue(rules["conditions"]["projected_runtime_gt_168h"])
            self.assertTrue(rules["conditions"]["required_rankable_engine_zero_definitions"])

    def test_candidate_compiler_repair_covers_all_required_engines(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            readiness = root / "readiness"
            repair = root / "repair"
            run = root / "run"
            (readiness / "compiler").mkdir(parents=True)
            repair.mkdir()
            for rel in ["engines/engine_contracts", "audit", "templates", "compiler", "budget"]:
                (run / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_csv(readiness / "compiler/hypothesis_to_contract_trace.csv", [{
                "hypothesis_id": "H_ONLY",
                "family": "TSMOM only",
                "contract_id": "c1",
                "lane": "kraken_tier1_ready",
                "entry_sketch": "volatility-managed TSMOM",
            }])
            args = SimpleNamespace(readiness_root=str(readiness), repair_root=str(repair), smoke=True, candidate_definition_budget=70, coarse_definition_budget=50, seed=7)
            ctx = SimpleNamespace(run_root=run, args=args)
            sweep.stage_candidate_compiler_and_budget_repair(ctx)
            cov = pd.read_csv(run / "budget/required_engine_coverage_audit.csv")
            self.assertEqual(set(cov["family_engine_id"]), set(sweep.REQUIRED_RANKABLE_ENGINES))
            self.assertTrue((cov["status"] == "pass").all())

    def test_family_universe_fanout_reduces_candidate_symbol_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prior = root / "prior"
            run = root / "run"
            (prior / "benchmark").mkdir(parents=True)
            (run / "panels").mkdir(parents=True)
            (run / "budget").mkdir()
            (run / "universe").mkdir()
            sweep.write_json(prior / "benchmark/engine_benchmark_progress.json", {"total_symbols": 301})
            panel = [{"symbol": f"PF_{i}USD", "bar_rows": 1000 - i, "funding_rows": 100 if i % 2 == 0 else 0, "status": "available", "end_ts": "2025-12-31T23:55:00Z"} for i in range(12)]
            sweep.write_csv(run / "panels/aggregate_panel_manifest.csv", panel)
            defs = []
            for engine_id in sweep.REQUIRED_RANKABLE_ENGINES:
                defs.append({"definition_id": f"d_{engine_id}", "hypothesis_id": "H", "family": "F", "family_engine_id": engine_id})
            sweep.write_csv(run / "budget/repaired_candidate_definition_budget_manifest.csv", defs)
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(prior_benchmark_root=str(prior), smoke=True, max_symbols=5, kraken_data_root=str(root)))
            sweep.stage_family_universe_fanout_repair(ctx)
            counts = pd.read_csv(run / "universe/family_candidate_symbol_counts.csv")
            self.assertTrue((counts["candidate_symbol_rows_after"] < counts["candidate_symbol_rows_before"]).all())
            self.assertTrue((counts["after_symbol_count"] <= 5).all())

    def test_protected_scan_classifies_metadata_separately_from_scoring_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["preflight", "seal", "benchmark"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_text(root / "preflight/input_artifact_manifest.csv", "path\n/results/foo_20260701\n")
            sweep.write_csv(root / "benchmark/safe_rows.csv", [{"decision_ts": "2025-12-31T23:55:00Z", "net_R": 1.0}])
            ctx = SimpleNamespace(run_root=root)
            sweep.stage_protected_scan_classification_repair(ctx)
            scoring = json.loads((root / "seal/protected_scoring_artifact_scan.json").read_text())
            report = (root / "seal/protected_scan_classification_report.md").read_text()
            self.assertEqual(scoring["status"], "pass")
            self.assertIn("Metadata/path-only", report)

    def test_operator_report_requires_materialization_benchmark(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["decision", "benchmark", "budget", "universe"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_json(root / "benchmark/repaired_full_budget_projection_analysis.json", {"gate_status": "go", "projected_aggregate_evaluation_hours": 10})
            sweep.write_csv(root / "budget/required_engine_coverage_audit.csv", [{"family_engine_id": e, "status": "pass"} for e in sweep.REQUIRED_RANKABLE_ENGINES])
            sweep.write_csv(root / "universe/family_candidate_symbol_counts.csv", [{"family_engine_id": e, "after_symbol_count": 1} for e in sweep.REQUIRED_RANKABLE_ENGINES])
            ctx = SimpleNamespace(run_root=root)
            sweep.stage_operator_go_no_go_report(ctx)
            decision = json.loads((root / "decision/operator_go_no_go_decision.json").read_text())
            self.assertEqual(decision["next_operator_decision"], "repair_materialization_controls_next")

    def test_phase_profile_reproducibility_defaults_to_vectorized_repair(self):
        args = sweep.parse_args([])
        self.assertEqual(args.phase_profile, "vectorized_priority_repair_20260703_v1")
        stages = sweep.active_stage_list(args)
        self.assertIn("seal-guard-and-protected-scan", stages)
        self.assertIn("priority-tranche-go-no-go", stages)
        self.assertEqual(sweep.active_run_id(args), "phase_kraken_aggfirst_vectorized_priority_repair_20260703_v1")
        old = sweep.parse_args(["--phase-profile", "aggregate_first_repair_20260702_v1"])
        self.assertIn("benchmark-gate-repair", sweep.active_stage_list(old))

    def test_v2_phase_profile_stage_list_and_prior_profiles_callable(self):
        args = sweep.parse_args(["--phase-profile", "vectorized_priority_repair_20260703_v2"])
        self.assertEqual(sweep.active_run_id(args), "phase_kraken_aggfirst_vectorized_priority_repair_20260703_v2")
        self.assertEqual(sweep.active_stage_list(args), [
            "preflight-and-source-freeze",
            "seal-guard-and-protected-scan",
            "benchmark-and-representativeness-repair",
            "true-batch-vectorization-repair",
            "hypothesis-routing-and-doe-repair",
            "control-matching-and-control-cap-repair",
            "candidate-overlap-clustering",
            "v0-and-engine-wave-tranche-planning",
            "benchmark-rerun",
            "priority-tranche-go-no-go",
            "decision-report",
            "compact-review-bundle",
        ])
        self.assertIn("smoke_integration_20260702_v1", sweep.PHASE_PROFILES)
        self.assertIn("aggregate_first_benchmark_20260702_v1", sweep.PHASE_PROFILES)
        self.assertIn("vectorized_priority_repair_20260703_v1", sweep.PHASE_PROFILES)

    def test_engine_wave_profile_stage_list_and_prior_profiles_callable(self):
        args = sweep.parse_args(["--phase-profile", "engine_wave_v0_tranche_20260703_v1"])
        self.assertEqual(sweep.active_run_id(args), "phase_kraken_engine_wave_v0_tranche_20260703_v1")
        self.assertEqual(sweep.active_stage_list(args), [
            "preflight-and-source-freeze",
            "seal-guard-and-protected-scan",
            "prelaunch-contract-eligibility-gate",
            "engine-wave-plan-freeze",
            "engine-wave-loop",
            "per-wave-aggregate-evaluation",
            "per-wave-selection-and-clustering",
            "per-wave-materialized-ledger-regeneration",
            "per-wave-aggregate-vs-materialized-audit",
            "per-wave-real-controls",
            "per-wave-stress-context-stability",
            "per-wave-analysis-ready-publication",
            "per-wave-archive-and-cleanup",
            "cross-wave-summary",
            "decision-report",
            "compact-review-bundle",
        ])
        self.assertIn("vectorized_priority_repair_20260703_v2", sweep.PHASE_PROFILES)
        self.assertIn("aggregate_first_repair_20260702_v1", sweep.PHASE_PROFILES)

    def test_prelaunch_gate_excludes_nonrankable_with_preservation_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            prior = Path(td) / "prior"
            for rel in ["prelaunch", "priority_tranche", "compiler"]:
                (prior / rel).mkdir(parents=True, exist_ok=True)
            root.mkdir()
            sweep.write_csv(prior / "priority_tranche/priority_tranche_definition_manifest.csv", [
                {"definition_id": "d_ok", "hypothesis_id": "H01", "family": "Liquid continuation", "allowed_lane": "compiled_for_future_full_sweep", "family_engine_id": "liquid_continuation_breakout_engine", "archetype": "liquid_continuation", "entry_template": "liquid_continuation_close_confirmed", "data_cap": "none"},
                {"definition_id": "d_capture", "hypothesis_id": "D4", "family": "Liquidation microstructure", "allowed_lane": "needs_live_capture_substitute", "family_engine_id": "funding_crowding_engine", "archetype": "funding_crowding", "entry_template": "funding_crowding_close_confirmed", "data_cap": "live_capture_required"},
                {"definition_id": "d_touch", "hypothesis_id": "ORB", "family": "ORB scalp", "allowed_lane": "compiled_for_future_full_sweep", "family_engine_id": "session_calendar_engine", "archetype": "session_calendar", "entry_template": "touch_fill_orb", "data_cap": "none"},
            ])
            sweep.write_csv(prior / "compiler/hypothesis_to_engine_translation_audit.csv", [
                {"hypothesis_id": "H01", "family_engine_id": "liquid_continuation_breakout_engine", "priority_lane": "priority_tier1_full_train_candidate", "translation_type": "direct strategy translation", "translation_rankable": True},
                {"hypothesis_id": "D4", "family_engine_id": "funding_crowding_engine", "priority_lane": "forward_capture_sidecar", "translation_type": "sidecar/capture translation", "translation_rankable": False},
                {"hypothesis_id": "ORB", "family_engine_id": "session_calendar_engine", "priority_lane": "priority_tier1_full_train_candidate", "translation_type": "direct strategy translation", "translation_rankable": True},
            ])
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(prior_benchmark_root=str(prior)))
            sweep.stage_prelaunch_contract_eligibility_gate(ctx)
            eligible = pd.read_csv(root / "prelaunch/eligible_definition_manifest.csv")
            excluded = pd.read_csv(root / "prelaunch/excluded_nonrankable_contracts.csv")
            self.assertEqual(set(eligible["definition_id"]), {"d_ok"})
            self.assertIn("forward_capture_sidecar", set(excluded["preservation_state"]))
            self.assertFalse((excluded["preservation_state"].astype(str) == "family_rejected").any())

    def test_engine_wave_plan_freezes_materialization_budgets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            prior = Path(td) / "prior"
            for rel in ["prelaunch", "engine_wave", "mechanics", "panels", "universe", "engines/engine_contracts", "audit", "templates", "priority_tranche"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            (prior / "priority_tranche").mkdir(parents=True, exist_ok=True)
            rows = []
            for engine_id, budget in sweep.ENGINE_WAVE_MATERIALIZATION_BUDGETS.items():
                for i in range(3):
                    rows.append({"definition_id": f"{engine_id}_{i}", "hypothesis_id": f"H{i}", "family": "f", "allowed_lane": "compiled_for_future_full_sweep", "family_engine_id": engine_id, "archetype": "tsmom", "entry_template": "tsmom_close_confirmed", "prelaunch_eligible": True})
            sweep.write_csv(root / "prelaunch/eligible_definition_manifest.csv", rows)
            sweep.write_csv(root / "prelaunch/excluded_nonrankable_contracts.csv", [])
            sweep.write_csv(prior / "priority_tranche/priority_tranche_definition_manifest.csv", rows)
            sweep.write_csv(prior / "priority_tranche/v0_tranche_plan.csv", [{"family_engine_id": e, "planned_definitions": 3, "runtime_hours_estimate": 1.0} for e in sweep.ENGINE_WAVE_ORDER])
            sweep.write_csv(prior / "priority_tranche/engine_wave_tranche_plan.csv", [{"family_engine_id": e, "engine_wave_id": f"{e}__wave_001", "planned_definitions": 3, "runtime_hours_estimate": 1.0} for e in sweep.ENGINE_WAVE_ORDER])
            args = SimpleNamespace(prior_benchmark_root=str(prior), smoke=True, max_symbols=0, kraken_data_root=str(Path(td) / "kraken"), max_output_gb=80, seed=7)
            ctx = SimpleNamespace(run_root=root, args=args, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
            sweep.stage_engine_wave_plan_freeze(ctx)
            budgets = pd.read_csv(root / "engine_wave/per_wave_materialization_budget.csv")
            self.assertTrue((budgets["budget_is_event_cap"].astype(str).str.lower() == "false").all())
            self.assertTrue((budgets["materialization_budget"] <= 2).all())
            self.assertTrue((root / "mechanics/account_fee_scenario_manifest.csv").exists())
            fee = pd.read_csv(root / "mechanics/account_fee_scenario_manifest.csv")
            self.assertIn("operator_attested_current_account_state", set(fee["fee_source"]))

    def test_seal_guard_writes_scoring_scan_and_ignores_metadata_dates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["preflight", "seal", "benchmark"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_text(root / "preflight/input_artifact_manifest.csv", "path\n/results/foo_20260703\n")
            sweep.write_csv(root / "benchmark/safe_rows.csv", [{"decision_ts": "2025-12-31T23:55:00Z"}])
            ctx = SimpleNamespace(run_root=root, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-12-31", tz="UTC"))
            sweep.stage_seal_guard_and_protected_scan(ctx)
            self.assertTrue((root / "seal/protected_timestamp_scan.json").exists())
            scoring = json.loads((root / "seal/protected_scoring_artifact_scan.json").read_text())
            self.assertEqual(scoring["status"], "pass")

    def test_real_null_controls_are_not_self_matches(self):
        bars = self.bars(500)
        wave = 100 + pd.Series([float(__import__("math").sin(i / 8.0)) for i in range(len(bars))])
        bars["open"] = wave
        bars["close"] = wave + 0.05
        bars["high"] = bars["close"] + 0.5
        bars["low"] = bars["open"] - 0.5
        bars["mark_close"] = bars["close"]
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand["hold_bars"] = 48
        cand["threshold"] = 0.001
        engine = sweep.ENGINES["scheduled_tsmom_engine"]
        ledger = engine.regenerate_materialized_ledger(bars, pd.DataFrame(), cand)
        controls, summary, diag = sweep.build_real_null_controls(cand, engine, bars, pd.DataFrame(), ledger, nulls_per_event=1)
        self.assertEqual(diag.get("zero_control_reason", ""), "")
        self.assertFalse(controls.empty)
        self.assertFalse(bool(controls["self_match"].any()))
        self.assertFalse(summary.empty)

    def test_priority_lane_classification_has_data_tiers_and_no_vendor_waiting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            readiness = root / "readiness"; repair = root / "repair"; run = root / "run"
            (readiness / "compiler").mkdir(parents=True)
            repair.mkdir()
            (run / "priority").mkdir(parents=True)
            sweep.write_csv(readiness / "compiler/hypothesis_to_contract_trace.csv", [
                {"hypothesis_id": "H01", "family": "Volatility-managed TSMOM", "contract_id": "c1", "lane": "kraken_tier1_ready", "entry_sketch": "TSMOM"},
                {"hypothesis_id": "C2", "family": "C2 catalyst ETF institutional access", "contract_id": "c2", "lane": "needs_event_ledger_first", "entry_sketch": "catalyst"},
            ])
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(readiness_root=str(readiness), repair_root=str(repair)))
            sweep.stage_priority_lane_classification(ctx)
            lanes = pd.read_csv(run / "priority/hypothesis_priority_lane.csv")
            for col in ["current_data_tier", "required_data_tier", "current_testability", "forward_capture_upgrade_path", "lane_assignment_reason"]:
                self.assertIn(col, lanes.columns)
            self.assertFalse(lanes.astype(str).apply(lambda c: c.str.contains("waiting_for_vendor_data", case=False, regex=False)).any().any())
            self.assertIn("event_ledger_first_sidecar", set(lanes["priority_lane"]))

    def test_priority_tranche_budget_and_pending_compute_are_written(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["priority", "priority_tranche", "budget", "universe", "panels"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            lanes = []
            for engine_id, meta in sweep.ENGINE_REPAIR_CONTRACTS.items():
                lanes.append({"hypothesis_id": meta["hypothesis_id"], "family": meta["family"], "contract_id": f"c_{engine_id}", "contract_source": "test", "allowed_lane": "kraken_tier1_ready", "archetype": meta["archetype"], "family_engine_id": engine_id, "priority_lane": "priority_tier1_full_train_candidate"})
            sweep.write_csv(root / "priority/hypothesis_priority_lane.csv", lanes)
            panel = [{"symbol": f"PF_{i}USD", "bar_rows": 1000 - i, "funding_rows": 100, "status": "available", "end_ts": "2025-12-31T23:55:00Z"} for i in range(10)]
            sweep.write_csv(root / "panels/aggregate_panel_manifest.csv", panel)
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(smoke=True, max_symbols=5, seed=7, kraken_data_root=str(root)))
            sweep.stage_priority_tranche_budget_construction(ctx)
            defs = pd.read_csv(root / "priority_tranche/priority_tranche_definition_manifest.csv")
            self.assertFalse(defs.empty)
            self.assertTrue((root / "priority_tranche/pending_compute_manifest.csv").exists())

    def test_symbol_stratified_benchmark_design_rejects_one_symbol(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["benchmark", "panels", "engines/engine_contracts", "audit", "templates", "universe"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_csv(root / "panels/aggregate_panel_manifest.csv", [{"symbol": "PF_XBTUSD", "bar_rows": 100, "funding_rows": 10, "status": "available", "start_ts": "2025-01-01", "end_ts": "2025-01-02"}])
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(smoke=False, max_symbols=0, kraken_data_root=str(root)))
            sweep.stage_benchmark_and_representativeness_repair(ctx)
            report = (root / "benchmark/symbol_representativeness_report.md").read_text()
            self.assertIn("Symbol-stratified benchmark design pass: `False`", report)

    def test_hypothesis_routing_and_doe_writes_canonical_first_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            readiness = root / "readiness"; repair = root / "repair"; run = root / "run"
            (readiness / "compiler").mkdir(parents=True)
            repair.mkdir()
            for rel in ["priority", "priority_tranche", "compiler", "budget", "universe", "panels"]:
                (run / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_csv(readiness / "compiler/hypothesis_to_contract_trace.csv", [
                {"hypothesis_id": "H02", "family": "Volatility-managed TSMOM", "contract_id": "c1", "lane": "kraken_tier1_ready", "entry_sketch": "TSMOM"},
                {"hypothesis_id": "C2", "family": "C2 catalyst ETF institutional access", "contract_id": "c2", "lane": "needs_event_ledger_first", "entry_sketch": "catalyst"},
            ])
            sweep.write_csv(run / "panels/aggregate_panel_manifest.csv", [{"symbol": f"PF_{i}USD", "bar_rows": 1000 - i, "funding_rows": 100, "status": "available", "end_ts": "2025-12-31T23:55:00Z"} for i in range(8)])
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(readiness_root=str(readiness), repair_root=str(repair), smoke=True, max_symbols=5, seed=11, kraken_data_root=str(root)))
            sweep.stage_hypothesis_routing_and_doe_repair(ctx)
            self.assertTrue((run / "compiler/hypothesis_to_engine_translation_audit.csv").exists())
            canonical = pd.read_csv(run / "priority_tranche/canonical_definition_manifest.csv")
            self.assertFalse(canonical.empty)
            self.assertTrue((canonical["canonical"].astype(bool)).all())
            self.assertTrue((run / "priority_tranche/doe_sampling_manifest.csv").exists())

    def test_v2_go_no_go_prefers_v0_when_it_fits_and_never_launches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["gate", "benchmark", "controls", "priority", "priority_tranche", "analysis_ready", "candidate_library", "seal", "audit", "compiler"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_json(root / "benchmark/vectorized_full_budget_projection.json", {"aggregate_materialized_audit_pass": True, "vectorized_scalar_audit_pass": True, "benchmark_mode": "cached_scalar_benchmark", "median_speedup_vs_prior_repair": 1.2})
            sweep.write_json(root / "benchmark/priority_tranche_total_runtime_projection.json", {"projected_total_hours": 200, "runtime_gate": "no_go_over_168h"})
            sweep.write_json(root / "benchmark/full_repaired_total_runtime_projection.json", {"projected_total_hours": 400, "runtime_gate": "no_go_over_168h"})
            sweep.write_csv(root / "priority_tranche/v0_tranche_plan.csv", [{"family_engine_id": "scheduled_tsmom_engine", "runtime_hours_estimate": 10}])
            sweep.write_csv(root / "priority_tranche/engine_wave_tranche_plan.csv", [{"family_engine_id": "scheduled_tsmom_engine", "runtime_hours_estimate": 10}])
            sweep.write_csv(root / "benchmark/symbol_stratified_materialization_controls_stress_benchmark.csv", [
                {"candidate_id": "c1", "symbol": "PF_XBTUSD", "row_semantics": "position_interval", "status": "pass"},
                {"candidate_id": "c2", "symbol": "PF_ETHUSD", "row_semantics": "trade_episode", "status": "pass"},
                {"candidate_id": "c3", "symbol": "PF_SOLUSD", "row_semantics": "lifecycle_event", "status": "pass"},
            ])
            sweep.write_text(root / "benchmark/symbol_representativeness_report.md", "Symbol-stratified materialization/control/stress benchmark pass: `True`\n")
            sweep.write_text(root / "audit/vectorized_exactness_failure_report.md", "Failures: `0`\n")
            sweep.write_csv(root / "controls/control_zero_row_diagnostics.csv", [{"candidate_id": "c", "controls_row_count": 10}])
            sweep.write_csv(root / "controls/control_self_match_leakage_audit.csv", [{"candidate_id": "c", "self_match_rows": 0}])
            sweep.write_csv(root / "controls/control_cap_diagnostics.csv", [{"candidate_id": "c", "cap_scope": "benchmark_diagnostic_only"}])
            sweep.write_csv(root / "priority/hypothesis_priority_lane.csv", [{"hypothesis_id": "H", "priority_lane": "priority_tier1_full_train_candidate"}])
            sweep.write_csv(root / "priority_tranche/priority_tranche_definition_manifest.csv", [{"definition_id": "d", "hypothesis_id": "H", "family_engine_id": "scheduled_tsmom_engine"}])
            sweep.write_csv(root / "compiler/hypothesis_to_engine_translation_audit.csv", [{"hypothesis_id": "H", "family_engine_id": "scheduled_tsmom_engine", "priority_lane": "priority_tier1_full_train_candidate"}])
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(phase_profile="vectorized_priority_repair_20260703_v2"))
            sweep.stage_priority_tranche_go_no_go(ctx)
            gate = json.loads((root / "gate/priority_tranche_go_no_go.json").read_text())
            self.assertEqual(gate["next_operator_decision"], "launch_priority_tranche_v0_next")
            self.assertFalse(gate["priority_tranche_launched"])
            self.assertFalse(gate["full_sweep_launched"])
            lib = pd.read_csv(root / "candidate_library/vectorized_priority_repair_candidate_status_update.csv")
            self.assertFalse(lib.astype(str).apply(lambda c: c.str.contains("waiting_for_vendor_data", case=False, regex=False)).any().any())
            self.assertFalse(bool(lib.get("aggregate_only_survivor_label", pd.Series([False])).astype(bool).any()))

    def test_priority_go_no_go_blocks_zero_controls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["gate", "benchmark", "controls", "priority", "priority_tranche", "analysis_ready", "candidate_library", "seal"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_json(root / "benchmark/vectorized_full_budget_projection.json", {"aggregate_materialized_audit_pass": True, "vectorized_scalar_audit_pass": True})
            sweep.write_json(root / "benchmark/priority_tranche_total_runtime_projection.json", {"runtime_gate": "go_under_72h", "projected_total_hours": 10})
            sweep.write_json(root / "benchmark/full_repaired_total_runtime_projection.json", {"runtime_gate": "no_go_over_168h", "projected_total_hours": 200})
            sweep.write_csv(root / "benchmark/priority_tranche_materialization_controls_stress_benchmark.csv", [{"candidate_id": "c", "status": "pass"}])
            sweep.write_csv(root / "controls/control_zero_row_diagnostics.csv", [{"candidate_id": "c", "controls_row_count": 0, "zero_control_reason": "insufficient_non_event_windows"}])
            sweep.write_csv(root / "priority/hypothesis_priority_lane.csv", [{"hypothesis_id": "H", "priority_lane": "priority_tier1_full_train_candidate"}])
            sweep.write_csv(root / "priority_tranche/priority_tranche_definition_manifest.csv", [{"definition_id": "d", "hypothesis_id": "H", "family_engine_id": "scheduled_tsmom_engine"}])
            ctx = SimpleNamespace(run_root=root)
            sweep.stage_priority_tranche_go_no_go(ctx)
            gate = json.loads((root / "gate/priority_tranche_go_no_go.json").read_text())
            self.assertEqual(gate["next_operator_decision"], "repair_control_matching_next")
            ev = pd.read_csv(root / "analysis_ready/evidence_level_assignment.csv")
            self.assertTrue((ev["evidence_level"] == "level_1_generator_support").all())

    def test_structural_repair_profile_stage_list_and_run_id(self):
        args = sweep.parse_args(["--phase-profile", "engine_wave_structural_repair_20260703_v1", "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), "phase_kraken_engine_wave_structural_repair_20260703_v1")
        self.assertEqual(sweep.active_stage_list(args), list(sweep.STRUCTURAL_REPAIR_STAGES))
        self.assertIn("engine_wave_v0_tranche_20260703_v1", sweep.PHASE_PROFILES)
        self.assertIn("vectorized_priority_repair_20260703_v2", sweep.PHASE_PROFILES)

    def test_tsmom_retest_profile_stage_list_run_id_and_stop_flag(self):
        args = sweep.parse_args([
            "--phase-profile", "engine_wave_tsmom_retest_20260703_v1",
            "--stage", "all",
            "--family-list", "scheduled_tsmom_engine",
            "--stop-after-current-engine-wave",
        ])
        self.assertEqual(sweep.active_run_id(args), "phase_kraken_engine_wave_v0_tranche_tsmom_retest_20260703_v1")
        self.assertEqual(sweep.active_stage_list(args), list(sweep.ENGINE_WAVE_V0_TRANCHE_STAGES))
        self.assertTrue(args.stop_after_current_engine_wave)

    def test_tsmom_retest_p1_profile_stage_list_run_id_and_dependency_arg(self):
        args = sweep.parse_args([
            "--phase-profile", "engine_wave_tsmom_retest_p1_20260704_v1",
            "--stage", "all",
            "--family-list", "scheduled_tsmom_engine",
            "--stop-after-current-engine-wave",
            "--p1-repair-root", "results/rebaseline/phase_kraken_engine_wave_p1_protocol_repair_20260704_v1",
        ])
        self.assertEqual(sweep.active_run_id(args), "phase_kraken_engine_wave_v0_tranche_tsmom_retest_p1_20260704_v1")
        self.assertEqual(sweep.active_stage_list(args), list(sweep.ENGINE_WAVE_V0_TRANCHE_STAGES))
        self.assertTrue(sweep.is_tsmom_retest_profile(args.phase_profile))
        self.assertIn("p1_repair_root", vars(args))

    def write_repair_dependency_roots(self, root: Path) -> tuple[Path, Path, Path]:
        structural = root / "structural"
        stat = root / "stat"
        p1 = root / "p1"
        for rel in [
            "contracts", "audit", "controls", "gate", "resume",
        ]:
            (structural / rel).mkdir(parents=True, exist_ok=True)
        for rel in [
            "preflight", "validation", "controls", "mechanics", "funding",
        ]:
            (stat / rel).mkdir(parents=True, exist_ok=True)
        for rel in [
            "gate", "mechanics", "funding", "controls", "validation",
        ]:
            (p1 / rel).mkdir(parents=True, exist_ok=True)
        sweep.write_text(structural / "contracts/candidate_identity_contract.md", "ok")
        sweep.write_text(structural / "contracts/parameter_vector_contract.md", "ok")
        sweep.write_csv(structural / "audit/original_selected_vs_materialized_audit.csv", [{"status": "pass"}])
        sweep.write_text(structural / "controls/control_schema_repair_report.md", "ok")
        sweep.write_json(structural / "gate/wave_gate_fail_closed_contract.json", {"ok": True})
        sweep.write_text(structural / "resume/stale_wave_gate_invalidation_report.md", "ok")
        sweep.write_json(structural / "decision_summary.json", {
            "status": "complete",
            "candidate_identity_repaired": True,
            "parameter_preservation_repaired": True,
            "aggregate_materialized_audit_repaired": True,
            "control_schema_repaired": True,
            "wave_gate_fail_closed_repaired": True,
            "resume_stale_gate_invalidation_repaired": True,
        })
        sweep.write_text(stat / "preflight/structural_repair_dependency_check.md", "ok")
        sweep.write_csv(stat / "validation/dev_eval_split_manifest.csv", [{"status": "pass"}])
        sweep.write_text(stat / "controls/interval_overlap_purge_policy.md", "ok")
        sweep.write_csv(stat / "controls/control_cap_diagnostics.csv", [{"status": "pass"}])
        sweep.write_csv(stat / "mechanics/evidence_cap_application_audit.csv", [{"status": "pass"}])
        sweep.write_csv(stat / "funding/funding_exactness_summary.csv", [{"status": "pass"}])
        sweep.write_json(stat / "decision_summary.json", {
            "status": "complete",
            "structural_dependency_pass": True,
            "dev_eval_repair": True,
            "interval_purge_repair": True,
            "control_cap_repair": True,
            "mechanics_cap_policy": True,
            "funding_exactness_classification": True,
        })
        sweep.write_json(p1 / "gate/rerun_readiness_gate.json", {
            "unresolved_p0_count": 0,
            "unresolved_p1_count": 0,
            "active_p2_cap_count": 1,
        })
        sweep.write_csv(p1 / "mechanics/evidence_cap_application_audit.csv", [{"status": "pass"}])
        sweep.write_text(p1 / "mechanics/fee_formula_policy.md", "ok")
        sweep.write_csv(p1 / "mechanics/fee_model_audit.csv", [{"formula_repaired": True}])
        sweep.write_csv(p1 / "mechanics/mark_liquidation_lifecycle_audit.csv", [{"status": "active_cap"}])
        sweep.write_csv(p1 / "funding/funding_exactness_code_audit.csv", [{"status": "pass"}])
        sweep.write_text(p1 / "funding/funding_cap_impact_report.md", "ok")
        sweep.write_csv(p1 / "controls/control_cap_truthfulness_audit.csv", [{"status": "pass"}])
        sweep.write_csv(p1 / "validation/multiple_testing_label_gate_audit.csv", [{"status": "pass"}])
        sweep.write_json(p1 / "decision_summary.json", {
            "status": "complete",
            "P1_MECH_LABEL_001_fixed": True,
            "P1_FEE_FORMULA_001_fixed": True,
            "P1_FUNDING_PROXY_001_fixed": True,
            "P1_CTRL_CAP_001_fixed": True,
            "P1_STAT_MTEST_001_fixed": True,
            "P2_MARK_LIFECYCLE_001_status": "active_cap",
            "unresolved_p0_count": 0,
            "unresolved_p1_count": 0,
            "active_p2_cap_count": 1,
        })
        return structural, stat, p1

    def test_repaired_dependency_check_requires_structural_and_stat_roots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            structural, stat, p1 = self.write_repair_dependency_roots(root)
            run = root / "run"
            (run / "preflight").mkdir(parents=True)
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(structural_repair_root=str(structural), stat_protocol_repair_root=str(stat), p1_repair_root=str(p1)))
            self.assertTrue(sweep.stage_repaired_dependency_check(ctx))
            report = (run / "preflight/repaired_dependency_check.md").read_text()
            self.assertIn("Overall status: `pass`", report)
            bad = root / "bad"
            (bad / "preflight").mkdir(parents=True)
            bad_ctx = SimpleNamespace(run_root=bad, args=SimpleNamespace(structural_repair_root=str(structural), stat_protocol_repair_root=str(root / "missing_stat"), p1_repair_root=str(p1)))
            self.assertFalse(sweep.stage_repaired_dependency_check(bad_ctx))

    def test_tsmom_retest_wave_plan_contains_only_scheduled_tsmom(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "run"
            src = root / "src"
            for rel in ["prelaunch", "engine_wave", "mechanics", "priority_tranche"]:
                (run / rel).mkdir(parents=True, exist_ok=True)
            (src / "priority_tranche").mkdir(parents=True, exist_ok=True)
            (src / "compiler").mkdir(parents=True, exist_ok=True)
            eligible = [
                {**self.candidate("scheduled_tsmom_engine", "tsmom"), "definition_id": "d_tsmom", "prelaunch_eligible": True, "allowed_lane": "priority_tier1_full_train_candidate"},
                {**self.candidate("liquid_continuation_breakout_engine", "liquid_continuation"), "definition_id": "d_liq", "prelaunch_eligible": True, "allowed_lane": "priority_tier1_full_train_candidate"},
            ]
            sweep.write_csv(run / "prelaunch/eligible_definition_manifest.csv", eligible)
            sweep.write_csv(run / "prelaunch/excluded_nonrankable_contracts.csv", [])
            sweep.write_csv(src / "priority_tranche/priority_tranche_definition_manifest.csv", eligible)
            sweep.write_csv(src / "priority_tranche/v0_tranche_plan.csv", [
                {"family_engine_id": "scheduled_tsmom_engine", "planned_definitions": 10, "runtime_hours_estimate": 1},
                {"family_engine_id": "liquid_continuation_breakout_engine", "planned_definitions": 10, "runtime_hours_estimate": 1},
            ])
            sweep.write_csv(src / "priority_tranche/engine_wave_tranche_plan.csv", [
                {"family_engine_id": "scheduled_tsmom_engine", "planned_definitions": 10, "runtime_hours_estimate": 1},
                {"family_engine_id": "liquid_continuation_breakout_engine", "planned_definitions": 10, "runtime_hours_estimate": 1},
            ])
            old_panel, old_contracts, old_universe, old_tsmom_universe = sweep.stage_panel_build, sweep.stage_family_engine_contracts, sweep.stage_family_universe_fanout_repair, sweep.stage_tsmom_retest_universe_fanout
            try:
                sweep.stage_panel_build = lambda ctx: None
                sweep.stage_family_engine_contracts = lambda ctx: None
                sweep.stage_family_universe_fanout_repair = lambda ctx: None
                sweep.stage_tsmom_retest_universe_fanout = lambda ctx: sweep.write_csv(ctx.run_root / "universe/family_universe_symbols.csv", [{"family_engine_id": "scheduled_tsmom_engine", "symbol": "PF_XBTUSD"}])
                ctx = SimpleNamespace(
                    run_root=run,
                    args=SimpleNamespace(
                        phase_profile="engine_wave_tsmom_retest_20260703_v1",
                        family_list="scheduled_tsmom_engine",
                        smoke=False,
                        max_output_gb=80,
                        prior_benchmark_root=str(src),
                    ),
                )
                sweep.stage_engine_wave_plan_freeze(ctx)
            finally:
                sweep.stage_panel_build, sweep.stage_family_engine_contracts, sweep.stage_family_universe_fanout_repair, sweep.stage_tsmom_retest_universe_fanout = old_panel, old_contracts, old_universe, old_tsmom_universe
            plan = pd.read_csv(run / "engine_wave/engine_wave_execution_plan.csv")
            self.assertEqual(plan["family_engine_id"].tolist(), ["scheduled_tsmom_engine"])
            self.assertTrue((run / "engine_wave/tsmom_retest_materialization_budget.csv").exists())

    def test_protected_scan_timestamp_column_predicate_excludes_candidate_ids(self):
        self.assertTrue(sweep.is_timestamp_like_column_name("decision_ts"))
        self.assertTrue(sweep.is_timestamp_like_column_name("feature_source_ts"))
        self.assertFalse(sweep.is_timestamp_like_column_name("candidate_definition_id"))
        self.assertFalse(sweep.is_timestamp_like_column_name("candidate_symbol_id"))

    def test_parameter_preservation_from_selection_to_materialization_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["materialized", "mechanics"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            cand = sweep.identity_enriched_record({
                **self.candidate("scheduled_tsmom_engine", "tsmom"),
                "symbol": "__universe__",
                "selected_symbol_universe": "PF_XBTUSD;PF_ETHUSD",
                "materialization_scope": "definition_universe",
                "original_selected_events": 0,
                "original_selected_net_R": 0.0,
            })
            old_bars = sweep.load_symbol_bars
            old_funding = sweep.load_funding
            try:
                def fake_bars(paths, symbol, start, end):
                    b = self.bars(160)
                    b["symbol"] = symbol
                    return b
                sweep.load_symbol_bars = fake_bars
                sweep.load_funding = lambda paths, symbol, end: pd.DataFrame()
                ctx = SimpleNamespace(
                    run_root=root,
                    start=pd.Timestamp("2025-01-01", tz="UTC"),
                    end=pd.Timestamp("2025-02-01", tz="UTC"),
                    args=SimpleNamespace(kraken_data_root=str(root), smoke=True),
                )
                manifest, coverage = sweep.materialize_wave_candidates(ctx, pd.DataFrame([cand]), root / "materialized")
            finally:
                sweep.load_symbol_bars = old_bars
                sweep.load_funding = old_funding
            self.assertFalse(manifest.empty)
            self.assertEqual(str(manifest.iloc[0]["parameter_vector_hash"]), cand["parameter_vector_hash"])
            self.assertEqual(str(manifest.iloc[0]["parameter_vector_json"]), cand["parameter_vector_json"])
            self.assertEqual(str(manifest.iloc[0]["materialization_scope"]), "definition_universe")
            self.assertEqual(int(manifest.iloc[0]["symbol_count_materialized"]), 2)
            self.assertFalse(bool(coverage.iloc[0]["event_sampling_used"]))

    def test_wave001_parameter_loss_reproducer_cannot_pass_original_selected_audit(self):
        fixture = json.loads(Path("unit_tests/fixtures/wave001_parameter_loss_reproducer.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "wave/event_ledgers").mkdir(parents=True, exist_ok=True)
            (root / "wave/audit").mkdir(parents=True, exist_ok=True)
            selected_example = fixture["selected_aggregate_example"]
            wrong_example = fixture["wrong_materialized_example"]
            selected = pd.DataFrame([{
                **self.candidate("scheduled_tsmom_engine", "tsmom"),
                "candidate_definition_id": selected_example["candidate_definition_id"],
                "candidate_id": selected_example["candidate_definition_id"],
                "parameter_vector_hash": selected_example["parameter_vector_hash"],
                "parameter_vector_json": "{\"fixture\":\"selected\"}",
                "events": selected_example["events"],
                "net_R": selected_example["net_R"],
                "gross_R": selected_example["net_R"],
                "fees_R": 0.0,
                "funding_R": 0.0,
                "active_symbols": 1,
                "original_selected_events": selected_example["events"],
                "original_selected_net_R": selected_example["net_R"],
                "original_selected_gross_R": selected_example["net_R"],
                "original_selected_fees_R": 0.0,
                "original_selected_funding_R": 0.0,
            }])
            per_row = wrong_example["net_R"] / wrong_example["events"]
            ledger = pd.DataFrame({
                "event_id": [f"e{i}" for i in range(wrong_example["events"])],
                "candidate_definition_id": selected_example["candidate_definition_id"],
                "candidate_id": selected_example["candidate_definition_id"],
                "parameter_vector_hash": selected_example["parameter_vector_hash"],
                "symbol": "PF_XBTUSD",
                "net_R": per_row,
                "gross_R": per_row,
                "fees_R": 0.0,
                "funding_R": 0.0,
            })
            ledger_path = root / "wave/event_ledgers/definition__fixture.parquet"
            ledger.to_parquet(ledger_path, index=False)
            manifest = pd.DataFrame([{
                "candidate_id": selected_example["candidate_definition_id"],
                "candidate_definition_id": selected_example["candidate_definition_id"],
                "parameter_vector_hash": selected_example["parameter_vector_hash"],
                "family_engine_id": "scheduled_tsmom_engine",
                "path": str(ledger_path.relative_to(root)),
                "materialization_scope": "definition_universe",
                "symbol_count_materialized": 1,
                "event_rows": wrong_example["events"],
            }])
            ctx = SimpleNamespace(run_root=root)
            audit = sweep.aggregate_vs_materialized_wave_audit(ctx, selected, manifest, root / "wave/audit")
            self.assertTrue(audit["status"].astype(str).eq("fail").any())
            failed_metrics = set(audit[audit["status"].astype(str).eq("fail")]["metric"])
            self.assertIn("events", failed_metrics)
            self.assertIn("net_R", failed_metrics)

    def test_candidate_identity_contract_distinguishes_definition_and_symbol_rows(self):
        base = sweep.identity_enriched_record({**self.candidate(), "symbol": "__universe__"})
        base["candidate_symbol_id"] = ""
        symbol_row = sweep.identity_enriched_record({**self.candidate(), "symbol": "PF_ETHUSD"}, symbol="PF_ETHUSD")
        self.assertEqual(base["candidate_definition_id"], symbol_row["candidate_definition_id"])
        self.assertNotEqual(base["candidate_symbol_id"], symbol_row["candidate_symbol_id"])
        duplicate_definition_row = {**base}
        audited = pd.DataFrame([base, duplicate_definition_row])
        audit = sweep.candidate_identity_audit_frame(audited, context="identity_test")
        duplicate_status = audit[audit["check"].astype(str).eq("duplicate_definition_parameter_rows_without_symbol_context")].iloc[0]["status"]
        self.assertEqual(duplicate_status, "fail")
        symbol_only = pd.DataFrame([symbol_row, sweep.identity_enriched_record({**self.candidate(), "symbol": "PF_SOLUSD"}, symbol="PF_SOLUSD")])
        audit_symbol = sweep.candidate_identity_audit_frame(symbol_only, context="symbol_context")
        duplicate_symbol_status = audit_symbol[audit_symbol["check"].astype(str).eq("duplicate_definition_parameter_rows_without_symbol_context")].iloc[0]["status"]
        self.assertEqual(duplicate_symbol_status, "pass")

    def tsmom_definition_row(self) -> dict:
        return {
            "definition_id": "sf__H02__duplicate_merge_fixture",
            "hypothesis_id": "H02",
            "family": "Volatility-managed TSMOM",
            "contract_id": "c_h02",
            "contract_source": "unit_test",
            "allowed_lane": "priority_tier1_full_train_candidate",
            "family_engine_id": "scheduled_tsmom_engine",
            "archetype": "tsmom",
            "definition_kind": "space_filling",
            "entry_template": "tsmom_close_confirmed",
            "exit_template": "fixed_hold",
            "stop_template": "fixed_bps",
            "regime_activation": "intended_regime",
            "regime_variant": "intended_regime",
            "side": "long",
            "lookback_bars": 24,
            "hold_bars": 288,
            "stop_bps": 75.0,
            "threshold": 0.005,
            "canonical": False,
            "data_cap": "",
            "priority_tranche": "priority_tranche_001",
            "priority_tranche_source": "unit_test",
            "priority_lane": "priority_tier1_full_train_candidate",
        }

    def duplicate_h02_translation_rows(self) -> list[dict]:
        base = {
            "hypothesis_id": "H02",
            "family_engine_id": "scheduled_tsmom_engine",
            "priority_lane": "priority_tier1_full_train_candidate",
            "current_data_tier": "kraken_tier1_ready",
            "required_data_tier": "kraken_tier1_ready",
            "current_testability": "rankable_current_kraken_data",
            "forward_capture_upgrade_path": "",
            "translation_type": "direct strategy translation",
            "translation_rankable": True,
            "translated_engine": "scheduled_tsmom_engine",
            "translation_reason": "archetype maps directly to engine",
        }
        return [
            {**base, "source_family": "Liquid continuation / Momentum"},
            {**base, "source_family": "Volatility-managed TSMOM"},
        ]

    def test_prelaunch_collapses_compatible_duplicate_translation_rows_before_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            run = root / "run"
            sweep.write_csv(src / "priority_tranche/priority_tranche_definition_manifest.csv", [self.tsmom_definition_row()])
            sweep.write_csv(src / "compiler/hypothesis_to_engine_translation_audit.csv", self.duplicate_h02_translation_rows())
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(prior_benchmark_root=str(src)))
            sweep.stage_prelaunch_contract_eligibility_gate(ctx)

            eligible = pd.read_csv(run / "prelaunch/eligible_definition_manifest.csv")
            self.assertEqual(len(eligible), 1)
            self.assertEqual(eligible["definition_id"].nunique(), 1)
            trans_audit = pd.read_csv(run / "prelaunch/translation_key_uniqueness_audit.csv")
            self.assertIn("compatible_duplicate_collapsed", set(trans_audit["action"].astype(str)))
            post_audit = pd.read_csv(run / "prelaunch/post_merge_definition_identity_audit.csv")
            self.assertFalse(post_audit["status"].astype(str).eq("fail").any())

            candidates = sweep.generate_candidate_registry_for_family_universe(
                eligible,
                {"scheduled_tsmom_engine": ["PF_XBTUSD", "PF_ETHUSD"]},
                len(eligible),
            )
            self.assertEqual(len(candidates), 2)
            uniqueness = sweep.candidate_symbol_uniqueness_audit_frame(candidates, context="unit_test_registry")
            self.assertFalse(uniqueness["status"].astype(str).eq("fail").any())

    def test_prelaunch_contradictory_duplicate_translation_rows_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            run = root / "run"
            bad_translations = self.duplicate_h02_translation_rows()
            bad_translations[1] = {**bad_translations[1], "priority_lane": "forward_capture_sidecar"}
            sweep.write_csv(src / "priority_tranche/priority_tranche_definition_manifest.csv", [self.tsmom_definition_row()])
            sweep.write_csv(src / "compiler/hypothesis_to_engine_translation_audit.csv", bad_translations)
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(prior_benchmark_root=str(src)))
            with self.assertRaisesRegex(RuntimeError, "contradictory duplicate translation"):
                sweep.stage_prelaunch_contract_eligibility_gate(ctx)
            trans_audit = pd.read_csv(run / "prelaunch/translation_key_uniqueness_audit.csv")
            self.assertTrue(trans_audit["status"].astype(str).eq("fail").any())

    def test_duplicate_candidate_symbol_rows_fail_before_aggregate_scope_can_double_count(self):
        row = sweep.identity_enriched_record({
            **self.candidate("scheduled_tsmom_engine", "tsmom"),
            "symbol": "PF_XBTUSD",
            "events": 100,
            "event_count": 100,
            "net_R": 12.5,
            "gross_R": 15.0,
            "fees_R": -1.0,
            "funding_R": -1.5,
        }, symbol="PF_XBTUSD")
        aggregate = pd.DataFrame([row, dict(row)])
        audit = sweep.candidate_symbol_uniqueness_audit_frame(aggregate, context="unit_test_aggregate")
        self.assertTrue(audit["status"].astype(str).eq("fail").any())
        with self.assertRaisesRegex(RuntimeError, "duplicate candidate-symbol identity"):
            sweep.aggregate_definition_scope(aggregate)

    def test_control_contract_aliases_equality_and_feature_timestamp(self):
        bars = self.bars(500)
        wave = 100 + pd.Series([float(__import__("math").sin(i / 8.0)) for i in range(len(bars))])
        bars["open"] = wave
        bars["close"] = wave + 0.05
        bars["high"] = bars["close"] + 0.5
        bars["low"] = bars["open"] - 0.5
        bars["mark_close"] = bars["close"]
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand["hold_bars"] = 48
        cand["threshold"] = 0.001
        engine = sweep.ENGINES["scheduled_tsmom_engine"]
        ledger = engine.regenerate_materialized_ledger(bars, pd.DataFrame(), cand)
        controls, summary, diag = sweep.build_real_null_controls(cand, engine, bars, pd.DataFrame(), ledger, nulls_per_event=1)
        self.assertFalse(controls.empty)
        for col in ["control_event_id", "control_symbol", "control_decision_ts", "matched_candidate_id", "matching_basis", "source_window_id", "feature_source_ts", "event_id", "symbol", "decision_ts"]:
            self.assertIn(col, controls.columns)
        self.assertFalse(controls["control_event_id"].isna().any())
        self.assertEqual(len(controls["control_event_id"]), controls["control_event_id"].nunique())
        self.assertTrue((controls["control_symbol"].astype(str) == controls["symbol"].astype(str)).all())
        self.assertTrue((pd.to_datetime(controls["feature_source_ts"], utc=True) <= pd.to_datetime(controls["control_decision_ts"], utc=True)).all())
        audit = sweep.control_contract_field_audit(controls)
        self.assertFalse(audit["status"].astype(str).eq("fail").any())

    def test_failed_wave_gate_downgrades_analysis_ready_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave_dirs = {
                "analysis": root / "analysis_ready/wave",
            }
            wave_dirs["analysis"].mkdir(parents=True)
            aggregate = pd.DataFrame([sweep.identity_enriched_record({
                **self.candidate("scheduled_tsmom_engine", "tsmom"),
                "symbol": "PF_XBTUSD",
                "events": 10,
                "net_R": 5.0,
                "gross_R": 5.0,
                "fees_R": 0.0,
                "funding_R": 0.0,
                "active_months": 1,
            })])
            selected = sweep.aggregate_definition_scope(aggregate).head(1)
            control_summary = pd.DataFrame([{"candidate_id": selected.iloc[0]["candidate_definition_id"], "control_count": 10}])
            gate = {"wave_gate_status": "repair_wave_component_next"}
            ctx = SimpleNamespace(run_root=root)
            sweep.write_wave_analysis_ready(ctx, "wave", wave_dirs, aggregate, selected, pd.DataFrame(), pd.DataFrame(), control_summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), gate)
            lib = pd.read_csv(root / "analysis_ready/wave/candidate_library_update.csv")
            self.assertTrue((lib["candidate_library_state"].astype(str) == "blocked_by_wave_component_issue").all())
            self.assertFalse(lib["evidence_level"].astype(str).isin(["level_4_event_ledger_plus_real_controls", "level_5"]).any())
            self.assertFalse(lib["candidate_library_state"].astype(str).eq("train_only_candidate_pending_validation").any())

    def test_resume_invalidation_rejects_stale_wave_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = root / "old"
            run = root / "run"
            (old / "waves/scheduled_tsmom_engine__wave_001").mkdir(parents=True)
            (run / "resume").mkdir(parents=True)
            sweep.write_json(old / "waves/scheduled_tsmom_engine__wave_001/wave_gate_decision.json", {
                "wave_gate_status": "pass",
                "wave_gate_logic_version": "old",
            })
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(failed_wave_root=str(old)))
            sweep.stage_resume_invalidation_repair(ctx)
            report = json.loads((run / "resume/stale_wave_gate_invalidation_report.json").read_text())
            self.assertTrue(report["stale_gate_detected"])
            self.assertEqual(report["required_action"], "invalidate_old_gate_and_rerun_wave")

    def test_single_symbol_materialization_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["materialized", "mechanics"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            cand = sweep.identity_enriched_record({
                **self.candidate("liquid_continuation_breakout_engine", "liquid_continuation"),
                "symbol": "PF_XBTUSD",
                "materialization_scope": "single_symbol_diagnostic_only",
                "selected_symbol_universe": "",
            })
            old_bars = sweep.load_symbol_bars
            old_funding = sweep.load_funding
            try:
                sweep.load_symbol_bars = lambda paths, symbol, start, end: self.bars(220)
                sweep.load_funding = lambda paths, symbol, end: pd.DataFrame()
                ctx = SimpleNamespace(
                    run_root=root,
                    start=pd.Timestamp("2025-01-01", tz="UTC"),
                    end=pd.Timestamp("2025-02-01", tz="UTC"),
                    args=SimpleNamespace(kraken_data_root=str(root), smoke=True),
                )
                manifest, _ = sweep.materialize_wave_candidates(ctx, pd.DataFrame([cand]), root / "materialized")
            finally:
                sweep.load_symbol_bars = old_bars
                sweep.load_funding = old_funding
            self.assertFalse(manifest.empty)
            self.assertEqual(str(manifest.iloc[0]["materialization_scope"]), "single_symbol_diagnostic_only")
            self.assertTrue(bool(manifest.iloc[0]["single_symbol_diagnostic_only"]))

    def test_stat_protocol_structural_dependency_failure_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            missing = Path(td) / "missing_structural"
            (root / "preflight").mkdir(parents=True)
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(structural_repair_root=str(missing)))
            ok = sweep.stage_structural_repair_dependency_check(ctx)
            self.assertFalse(ok)
            report = (root / "preflight/structural_repair_dependency_check.md").read_text()
            self.assertIn("Overall status: `fail`", report)

    def test_proposal_list_ignores_internal_eval_returns(self):
        aggregate = pd.DataFrame([
            {"candidate_id": "c1", "family_engine_id": "scheduled_tsmom_engine", "row_semantics": "position_interval"},
            {"candidate_id": "c2", "family_engine_id": "scheduled_tsmom_engine", "row_semantics": "position_interval"},
        ])
        months = pd.DataFrame([
            {"candidate_id": "c1", "symbol": "PF_XBTUSD", "month": "2024-01", "events": 10, "net_R": 10.0},
            {"candidate_id": "c2", "symbol": "PF_XBTUSD", "month": "2024-01", "events": 10, "net_R": 5.0},
            {"candidate_id": "c1", "symbol": "PF_XBTUSD", "month": "2024-02", "events": 10, "net_R": -1000.0},
            {"candidate_id": "c2", "symbol": "PF_XBTUSD", "month": "2024-02", "events": 10, "net_R": 1000.0},
        ])
        first = sweep.proposal_only_selection(aggregate, months, budget=1)
        changed = months.copy()
        changed.loc[changed["month"].eq("2024-02") & changed["candidate_id"].eq("c1"), "net_R"] = 100000.0
        changed.loc[changed["month"].eq("2024-02") & changed["candidate_id"].eq("c2"), "net_R"] = -100000.0
        second = sweep.proposal_only_selection(aggregate, changed, budget=1)
        self.assertEqual(first.iloc[0]["candidate_id"], "c1")
        self.assertEqual(second.iloc[0]["candidate_id"], "c1")
        self.assertEqual(first.iloc[0]["selection_segment_used"], "proposal_train_segment")

    def test_sparse_internal_eval_applies_sample_limited_cap(self):
        summary = pd.DataFrame([
            {"candidate_id": "c1", "train_internal_segment": "proposal_train_segment", "events": 100, "net_R": 5, "active_symbols": 3, "active_months": 4},
            {"candidate_id": "c1", "train_internal_segment": "internal_eval_train_segment", "events": 2, "net_R": 1, "active_symbols": 1, "active_months": 1},
        ])
        caps = sweep.apply_dev_eval_caps(summary, min_events=20, min_symbols=2, min_months=2)
        self.assertEqual(caps.iloc[0]["coverage_status"], "internal_eval_sample_limited_cap")

    def test_full_train_selection_gets_bias_cap_in_overlap_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stopped = root / "stopped"
            run = root / "run"
            wave = stopped / "waves/scheduled_tsmom_engine__wave_001"
            for rel in ["aggregate", "selection"]:
                (wave / rel).mkdir(parents=True, exist_ok=True)
            run.mkdir()
            aggregate = pd.DataFrame([{"candidate_id": "c_old", "family_engine_id": "scheduled_tsmom_engine", "row_semantics": "position_interval"}])
            aggregate.to_parquet(wave / "aggregate/all_candidate_aggregate_summary.parquet", index=False)
            pd.DataFrame([
                {"candidate_id": "c_old", "symbol": "PF_XBTUSD", "month": "2024-01", "events": 10, "net_R": 1.0},
                {"candidate_id": "c_old", "symbol": "PF_XBTUSD", "month": "2024-02", "events": 10, "net_R": 1.0},
            ]).to_parquet(wave / "aggregate/candidate_symbol_month_summary.parquet", index=False)
            pd.DataFrame([{"candidate_id": "c_old"}]).to_csv(wave / "selection/materialization_selection.csv", index=False)
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(stopped_engine_wave_root=str(stopped), smoke=True, seed=1))
            sweep.stage_dev_eval_selection_repair(ctx)
            overlap = pd.read_csv(run / "validation/proposal_scoring_overlap_audit.csv")
            self.assertIn("aggregate_screen_only_full_train_selection_bias_cap", set(overlap["cap_applied"].astype(str)))

    def test_interval_purge_and_embargo_remove_contaminated_controls(self):
        events = pd.DataFrame([{
            "candidate_id": "c1", "family_engine_id": "scheduled_tsmom_engine", "symbol": "PF_XBTUSD",
            "entry_ts": "2025-01-01T00:00:00Z", "exit_ts": "2025-01-01T01:00:00Z", "hold_bars": 12,
        }])
        controls = pd.DataFrame([
            {"matched_candidate_id": "c1", "control_symbol": "PF_XBTUSD", "entry_ts": "2025-01-01T00:30:00Z", "exit_ts": "2025-01-01T01:30:00Z"},
            {"matched_candidate_id": "c1", "control_symbol": "PF_XBTUSD", "entry_ts": "2025-01-01T01:20:00Z", "exit_ts": "2025-01-01T01:40:00Z"},
            {"matched_candidate_id": "c1", "control_symbol": "PF_XBTUSD", "entry_ts": "2025-01-02T00:00:00Z", "exit_ts": "2025-01-02T01:00:00Z"},
        ])
        audit = sweep.audit_interval_purge(events, controls)
        self.assertEqual(int(audit.iloc[0]["controls_removed_by_interval_overlap"]), 1)
        self.assertEqual(int(audit.iloc[0]["controls_removed_by_embargo"]), 1)
        self.assertEqual(audit.iloc[0]["purge_status"], "fail")

    def test_control_cap_reporting_is_truthful(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stopped = root / "stopped"
            run = root / "run"
            wave = stopped / "waves/scheduled_tsmom_engine__wave_001"
            (wave / "controls/control_ledger").mkdir(parents=True)
            (run / "controls").mkdir(parents=True)
            pd.DataFrame([{"candidate_id": "c1"}]).to_csv(wave / "controls/control_summary.csv", index=False)
            pd.DataFrame([{"matched_candidate_id": "c1", "control_event_id": f"ctrl{i}"} for i in range(6)]).to_parquet(wave / "controls/control_ledger/control_ledger.parquet", index=False)
            pd.DataFrame([{"candidate_id": "c1", "controls_removed_by_interval_overlap": 1, "controls_removed_by_embargo": 1}]).to_csv(run / "controls/interval_overlap_purge_audit.csv", index=False)
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(stopped_engine_wave_root=str(stopped), seed=7))
            sweep.stage_control_cap_reporting_repair(ctx)
            diag = pd.read_csv(run / "controls/control_cap_diagnostics.csv")
            self.assertEqual(int(diag.iloc[0]["total_matched_controls_before_cap"]), 6)
            self.assertEqual(int(diag.iloc[0]["controls_removed_by_interval_purge"]), 2)
            self.assertEqual(int(diag.iloc[0]["controls_retained"]), 4)

    def test_mechanics_caps_block_clean_level4(self):
        level, reason = sweep.apply_evidence_cap_gate({
            "requested_evidence_level": "level_4_event_ledger_plus_real_controls",
            "interval_purge_failed": True,
            "funding_proxy_used": True,
            "slippage_R": 0.0,
            "mark_liquidation_diagnostic_only": True,
        })
        self.assertNotEqual(level, "level_4_event_ledger_plus_real_controls")
        self.assertIn("interval_overlap_purge_failed", reason)
        self.assertIn("funding_proxy_cap", reason)
        self.assertIn("base_no_slippage_event_ledger_requires_stress", reason)

    def test_fee_scenario_and_funding_exactness_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stopped = root / "stopped"
            kraken = root / "kraken"
            run = root / "run"
            wave = stopped / "waves/scheduled_tsmom_engine__wave_001"
            for rel in ["materialized/event_ledgers"]:
                (wave / rel).mkdir(parents=True, exist_ok=True)
            (kraken / "parquet/funding").mkdir(parents=True)
            (run / "mechanics").mkdir(parents=True)
            ledger = pd.DataFrame([{
                "candidate_id": "c1", "funding_proxy_used": True, "slippage_R": 0.0, "index_price_source": "unavailable",
            }])
            ledger.to_parquet(wave / "materialized/event_ledgers/c1.parquet", index=False)
            pd.DataFrame([{"path": "waves/scheduled_tsmom_engine__wave_001/materialized/event_ledgers/c1.parquet"}]).to_csv(wave / "materialized/materialized_ledger_manifest.csv", index=False)
            pd.DataFrame([{"timestamp": "2025-06-26T08:00:00Z", "fundingRate": 0.1, "venue_symbol": "PF_XBTUSD"}]).to_parquet(kraken / "parquet/funding/PF_XBTUSD.parquet", index=False)
            ctx = SimpleNamespace(run_root=run, start=pd.Timestamp("2024-01-01", tz="UTC"), args=SimpleNamespace(stopped_engine_wave_root=str(stopped), kraken_data_root=str(kraken), smoke=False))
            sweep.stage_mechanics_cap_and_fee_model_repair(ctx)
            sweep.stage_funding_exactness_classification(ctx)
            fee = pd.read_csv(run / "mechanics/fee_model_audit.csv")
            self.assertIn("kraken_current_account_zero_fee", set(fee["fee_scenario"]))
            self.assertIn("fee_reversion_standard", set(fee["fee_scenario"]))
            funding = pd.read_csv(run / "funding/funding_coverage_by_symbol_date.csv")
            self.assertEqual(funding.iloc[0]["funding_exactness_class"], "partial_exact_with_proxy_needed")

    def test_p1_protocol_repair_profile_stage_list_and_run_id(self):
        args = sweep.parse_args(["--phase-profile", "engine_wave_p1_protocol_repair_20260704_v1", "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), "phase_kraken_engine_wave_p1_protocol_repair_20260704_v1")
        self.assertEqual(sweep.active_stage_list(args), list(sweep.P1_PROTOCOL_REPAIR_STAGES))
        self.assertIn("engine_wave_v0_tranche_20260703_v1", sweep.PHASE_PROFILES)

    def test_wave_label_consumes_mechanics_caps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave_root = root / "waves/wave"
            ledger_dir = wave_root / "materialized/event_ledgers"
            for rel in ["controls", "mechanics", "analysis_ready/wave"]:
                (wave_root / rel if rel != "analysis_ready/wave" else root / rel).mkdir(parents=True, exist_ok=True)
            ledger_dir.mkdir(parents=True)
            ev = pd.DataFrame([{
                "candidate_id": "c1", "candidate_definition_id": "c1", "symbol": "PF_XBTUSD",
                "funding_proxy_used": True, "funding_label_cap_reason": "funding_proxy_selection_cap",
                "slippage_R": 0.0, "mark_proxy_used": False, "index_price_source": "unavailable",
                "mark_liquidation_flag": False, "same_bar_ambiguity_flag": False,
                "label_cap_reason": "kraken_survivorship_lifecycle_cap;funding_missing_adverse_proxy",
            }])
            ev.to_parquet(ledger_dir / "c1.parquet", index=False)
            manifest = pd.DataFrame([{"candidate_id": "c1", "candidate_definition_id": "c1", "path": "waves/wave/materialized/event_ledgers/c1.parquet"}])
            selected = pd.DataFrame([{"candidate_id": "c1", "candidate_definition_id": "c1", "evidence_cap_reason": ""}])
            controls = pd.DataFrame([{"candidate_id": "c1", "control_event_count": 10}])
            wave_dirs = {
                "root": wave_root,
                "controls": wave_root / "controls",
                "analysis": root / "analysis_ready/wave",
            }
            sweep.write_csv(wave_root / "controls/control_cap_diagnostics.csv", [{"candidate_definition_id": "c1", "cap_applied": False, "final_evidence_eligible": True}])
            sweep.write_csv(wave_root / "controls/interval_overlap_purge_audit.csv", [{"candidate_id": "c1", "purge_status": "pass"}])
            ctx = SimpleNamespace(run_root=root)
            cap = sweep.collect_wave_evidence_cap_audit(ctx, wave_dirs, manifest, selected, controls, pd.DataFrame())
            self.assertTrue(bool(cap.iloc[0]["mechanics_cap_active"]))
            self.assertFalse(bool(cap.iloc[0]["clean_evidence_allowed"]))
            self.assertEqual(cap.iloc[0]["evidence_level_contract"], "level_4_event_ledger_plus_real_controls__mechanics_capped")

    def test_fee_R_round_trip_entry_exit_notional(self):
        comp = sweep.compute_round_trip_fee_R(
            100.0,
            110.0,
            10.0,
            qty=2.0,
            contract_size_or_multiplier=3.0,
            entry_fee_bps_per_side=10.0,
            exit_fee_bps_per_side=20.0,
        )
        expected = -(((100.0 * 2.0 * 3.0) * 0.001) + ((110.0 * 2.0 * 3.0) * 0.002)) / (10.0 * 2.0 * 3.0)
        self.assertAlmostEqual(comp["fee_R"], expected, places=12)
        short_comp = sweep.compute_round_trip_fee_R(110.0, 100.0, 10.0, qty=1.0, contract_size_or_multiplier=1.0, entry_fee_bps_per_side=10.0)
        self.assertAlmostEqual(short_comp["fee_R"], -((110.0 * 0.001) + (100.0 * 0.001)) / 10.0, places=12)

    def test_funding_proxy_uses_venue_boundaries_or_blocks_clean_selection(self):
        entry = pd.Timestamp("2025-01-01T00:00:00Z")
        exit_ = pd.Timestamp("2025-01-02T00:00:00Z")
        fields = sweep.funding_components_for_event(pd.DataFrame(), entry, exit_, side="long", entry_price=100.0, risk_price=1.0)
        self.assertEqual(fields["funding_boundary_count_proxy"], 3)
        self.assertNotEqual(fields["funding_boundary_count_proxy"], 24)
        self.assertTrue(fields["funding_proxy_used"])
        self.assertEqual(fields["funding_R"], 0.0)
        self.assertEqual(fields["funding_R_used_for_selection"], 0.0)
        self.assertAlmostEqual(fields["funding_proxy_R"], -0.15, places=12)
        self.assertAlmostEqual(fields["adverse_missing_funding_proxy_R"], -0.15, places=12)
        self.assertEqual(fields["funding_base_mode"], "gross_minus_fees_missing_exact_funding_not_charged")
        self.assertIn("funding_proxy_selection_cap", fields["funding_label_cap_reason"])
        no_cross = sweep.funding_components_for_event(pd.DataFrame(), entry, entry + pd.Timedelta(hours=1), side="long", entry_price=100.0, risk_price=1.0)
        self.assertTrue(no_cross["funding_exact"])
        self.assertEqual(no_cross["funding_R"], 0.0)

    def test_control_cap_reports_available_before_retained(self):
        all_idxs = pd.Series(range(sweep.CONTROL_RETAINED_MAX + 250)).to_numpy()
        retained = sweep.retain_control_indices(all_idxs, sweep.CONTROL_RETAINED_MAX)
        diag = {
            "total_matched_controls_available": len(all_idxs),
            "controls_retained": len(retained),
            "cap_applied": len(all_idxs) > len(retained),
            "cap_method": "deterministic_time_stratified_even_spacing",
            "final_evidence_eligible": not (len(all_idxs) > len(retained)),
        }
        self.assertGreater(diag["total_matched_controls_available"], diag["controls_retained"])
        self.assertTrue(diag["cap_applied"])
        self.assertEqual(diag["cap_method"], "deterministic_time_stratified_even_spacing")
        self.assertFalse(diag["final_evidence_eligible"])

    def test_level4_blocks_without_multiple_testing_neighborhood_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave_root = root / "waves/wave"
            ledger_dir = wave_root / "materialized/event_ledgers"
            for rel in ["controls", "mechanics", "analysis_ready/wave"]:
                (wave_root / rel if rel != "analysis_ready/wave" else root / rel).mkdir(parents=True, exist_ok=True)
            ledger_dir.mkdir(parents=True)
            pd.DataFrame([{
                "candidate_id": "c1", "candidate_definition_id": "c1", "symbol": "PF_XBTUSD",
                "funding_proxy_used": False, "slippage_R": -0.01, "mark_proxy_used": False,
                "index_price_source": "available", "mark_liquidation_flag": True,
                "same_bar_ambiguity_flag": False, "label_cap_reason": "",
            }]).to_parquet(ledger_dir / "c1.parquet", index=False)
            wave_dirs = {"root": wave_root, "controls": wave_root / "controls", "analysis": root / "analysis_ready/wave"}
            sweep.write_csv(wave_root / "controls/control_cap_diagnostics.csv", [{"candidate_definition_id": "c1", "cap_applied": False, "final_evidence_eligible": True}])
            sweep.write_csv(wave_root / "controls/interval_overlap_purge_audit.csv", [{"candidate_id": "c1", "purge_status": "pass"}])
            cap = sweep.collect_wave_evidence_cap_audit(
                SimpleNamespace(run_root=root),
                wave_dirs,
                pd.DataFrame([{"candidate_id": "c1", "candidate_definition_id": "c1", "path": "waves/wave/materialized/event_ledgers/c1.parquet"}]),
                pd.DataFrame([{"candidate_id": "c1", "candidate_definition_id": "c1"}]),
                pd.DataFrame([{"candidate_id": "c1", "control_event_count": 10}]),
                pd.DataFrame(),
            )
            self.assertIn("parameter_neighborhood_or_fold_support_missing", cap.iloc[0]["label_cap_reason"])
            self.assertNotEqual(cap.iloc[0]["evidence_level_contract"], "level_4_event_ledger_plus_real_controls")

    def test_resume_config_hash_change_invalidates_wave_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["resume", "waves/wave/resume", "engine_wave", "prelaunch", "mechanics", "funding"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            sweep.write_csv(root / "engine_wave/wave_definition_manifest.csv", [{"definition_id": "d1"}])
            args = SimpleNamespace(
                phase_profile="engine_wave_v0_tranche_20260703_v1",
                k0_root=str(root / "k0"),
                repair_root=str(root / "repair"),
                structural_repair_root=str(root / "structural"),
                stat_protocol_repair_root=str(root / "stat"),
            )
            for d in ["k0", "repair", "structural", "stat"]:
                (root / d).mkdir()
            ctx = SimpleNamespace(run_root=root, args=args)
            wave_dirs = {"root": root / "waves/wave"}
            _, digest1 = sweep.build_wave_gate_hash_manifest(ctx, "wave", wave_dirs)
            sweep.write_csv(root / "engine_wave/wave_definition_manifest.csv", [{"definition_id": "d2"}])
            _, digest2 = sweep.build_wave_gate_hash_manifest(ctx, "wave", wave_dirs)
            self.assertNotEqual(digest1, digest2)

    def test_mark_lifecycle_cap_blocks_clean_claims(self):
        level, reason = sweep.apply_evidence_cap_gate({
            "requested_evidence_level": "level_4_event_ledger_plus_real_controls",
            "mark_liquidation_diagnostic_only": True,
            "slippage_R": -0.01,
        })
        self.assertEqual(level, "level_4_event_ledger_plus_real_controls__mechanics_capped")
        self.assertIn("mark_liquidation_diagnostic_only_cap", reason)

    def test_tsmom_p1_top_level_outputs_copy_required_audits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave_root = root / "waves/scheduled_tsmom_engine__wave_001"
            dirs = {
                "root": wave_root,
                "aggregate": wave_root / "aggregate",
                "audit": wave_root / "audit",
                "controls": wave_root / "controls",
                "stress": wave_root / "stress",
                "materialized": wave_root / "materialized",
                "analysis": root / "analysis_ready/scheduled_tsmom_engine__wave_001",
            }
            for p in dirs.values():
                p.mkdir(parents=True, exist_ok=True)
            sweep.write_csv(dirs["audit"] / "original_selected_vs_materialized_audit.csv", [{"status": "pass"}])
            sweep.write_csv(dirs["aggregate"] / "candidate_symbol_uniqueness_audit.csv", [{"status": "pass"}])
            sweep.write_csv(dirs["controls"] / "control_contract_field_audit.csv", [{"status": "pass"}])
            sweep.write_csv(dirs["controls"] / "interval_overlap_purge_audit.csv", [{"purge_status": "pass"}])
            sweep.write_csv(dirs["controls"] / "control_cap_diagnostics.csv", [{"candidate_id": "c1", "total_matched_controls_before_cap": 10, "controls_retained": 10}])
            sweep.write_csv(dirs["controls"] / "control_self_match_leakage_audit.csv", [{"self_match_rows": 0}])
            sweep.write_csv(dirs["root"] / "mechanics/evidence_cap_application_audit.csv", [{"candidate_id": "c1", "mechanics_cap_active": True}])
            sweep.write_csv(dirs["stress"] / "fee_scenario_stress_summary.csv", [{"candidate_id": "c1", "fee_scenario": "kraken_current_account_zero_fee"}])
            sweep.write_json(dirs["root"] / "wave_gate_decision.json", {"wave_gate_status": "pass"})
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(phase_profile="engine_wave_tsmom_retest_p1_20260704_v1"))
            sweep.write_tsmom_retest_top_level_outputs(ctx, "scheduled_tsmom_engine__wave_001", dirs)
            self.assertTrue((root / "audit/original_selected_vs_materialized_audit.csv").exists())
            self.assertTrue((root / "audit/candidate_symbol_uniqueness_audit.csv").exists())
            self.assertTrue((root / "controls/control_cap_diagnostics.csv").exists())
            self.assertTrue((root / "analysis_ready/scheduled_tsmom_engine__wave_001/mechanics_cap_summary.csv").exists())
            self.assertTrue((root / "analysis_ready/scheduled_tsmom_engine__wave_001/fee_scenario_stress_summary.csv").exists())
            self.assertTrue((root / "analysis_ready/scheduled_tsmom_engine__wave_001/funding_cap_summary.csv").exists())

    def test_strict_bool_parser_string_false_and_ambiguous(self):
        self.assertFalse(sweep.parse_bool_value("False"))
        self.assertFalse(sweep.parse_bool_value("0"))
        self.assertTrue(sweep.parse_bool_value("yes"))
        with self.assertRaises(ValueError):
            sweep.parse_bool_value("maybe")
        self.assertTrue(sweep.parse_bool_value("maybe", default=True))
        parsed = sweep.parse_bool_series(pd.Series(["False", "true", "0", "1"]))
        self.assertEqual(parsed.tolist(), [False, True, False, True])

    def test_event_sampling_gate_uses_strict_bool_parser(self):
        self.assertFalse(sweep.any_true(pd.Series(["False", "0", False]), default=True))
        self.assertTrue(sweep.any_true(pd.Series(["ambiguous"]), default=True))
        self.assertFalse(sweep.any_true(pd.Series(["ambiguous"]), default=False))

    def test_dev_eval_cap_label_active_blocks_clean_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "selection"
            aggregate = pd.DataFrame([{
                "candidate_definition_id": "d1",
                "candidate_id": "d1",
                "parameter_vector_hash": "p1",
                "hypothesis_id": "H02",
                "family": "TSMOM",
                "family_engine_id": "scheduled_tsmom_engine",
                "row_semantics": "position_interval",
                "symbol": "PF_XBTUSD",
                "events": 30,
                "net_R": 10.0,
                "gross_R": 11.0,
                "fees_R": -1.0,
                "funding_R": 0.0,
                "median_R": 0.1,
                "active_months": 2,
            }])
            sym_month = pd.DataFrame([
                {"candidate_definition_id": "d1", "candidate_id": "d1", "symbol": "PF_XBTUSD", "month": "2025-01", "events": 25, "net_R": 5.0},
                {"candidate_definition_id": "d1", "candidate_id": "d1", "symbol": "PF_XBTUSD", "month": "2025-02", "events": 1, "net_R": 0.1},
            ])
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(phase_profile=sweep.TSMOM_RETEST_P1_CANARY_PHASE_PROFILE, seed=1))
            selected, _, _ = sweep.select_wave_materialization(
                ctx,
                "scheduled_tsmom_engine__wave_001",
                aggregate,
                {"materialization_budget": 1, "near_miss_budget": 0, "audit_sample_budget": 0},
                out,
                sym_month,
            )
            self.assertTrue(bool(selected.iloc[0]["internal_eval_sample_limited_cap"]))
            self.assertEqual(selected.iloc[0]["candidate_status"], "sample_limited_pending_validation")
            self.assertIn("internal_eval_sample_limited_cap", selected.iloc[0]["evidence_cap_reason"])

    def test_control_purge_before_cap_accounting(self):
        bars = self.bars(40)
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand["hold_bars"] = 3
        engine = sweep.ENGINES["scheduled_tsmom_engine"]
        ledger = pd.DataFrame([{
            "candidate_id": cand["candidate_id"],
            "candidate_definition_id": cand["definition_id"],
            "symbol": "PF_XBTUSD",
            "decision_ts": bars.iloc[1]["ts"],
            "entry_ts": bars.iloc[2]["ts"],
            "exit_ts": bars.iloc[5]["ts"],
            "net_R": 0.1,
        }])
        original = sweep.enumerate_null_control_indices
        try:
            sweep.enumerate_null_control_indices = lambda *args, **kwargs: (np.array([0, 1, 2, 6, 10, 15]), "synthetic_nulls")
            _, _, diag = sweep.build_real_null_controls(cand, engine, bars, pd.DataFrame(), ledger, nulls_per_event=1)
        finally:
            sweep.enumerate_null_control_indices = original
        self.assertEqual(diag["total_raw_null_opportunities"], 6)
        self.assertIn("eligible_after_purge_embargo", diag)
        self.assertLessEqual(diag["retained_after_cap"], diag["eligible_after_purge_embargo"])
        self.assertEqual(diag["controls_retained_pre_path"], diag["retained_after_cap"])

    def test_level5_requires_real_stability_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave_root = root / "waves/wave"
            ledger_dir = wave_root / "materialized/event_ledgers"
            for rel in ["stress", "context", "validation", "concentration"]:
                (wave_root / rel).mkdir(parents=True, exist_ok=True)
            ledger_dir.mkdir(parents=True)
            ledger = pd.DataFrame([
                {
                    "candidate_id": "c1",
                    "candidate_definition_id": "c1",
                    "family_engine_id": "scheduled_tsmom_engine",
                    "symbol": "PF_XBTUSD",
                    "decision_ts": "2025-01-01T00:00:00Z",
                    "net_R": 0.1,
                    "fees_R": -0.01,
                    "funding_R": 0.0,
                },
                {
                    "candidate_id": "c1",
                    "candidate_definition_id": "c1",
                    "family_engine_id": "scheduled_tsmom_engine",
                    "symbol": "PF_XBTUSD",
                    "decision_ts": "2025-08-01T00:00:00Z",
                    "net_R": 0.2,
                    "fees_R": -0.01,
                    "funding_R": 0.0,
                },
            ])
            ledger.to_parquet(ledger_dir / "c1.parquet", index=False)
            manifest = pd.DataFrame([{"candidate_id": "c1", "path": "waves/wave/materialized/event_ledgers/c1.parquet"}])
            _, _, stability, _ = sweep.run_wave_stress_context_stability(
                SimpleNamespace(run_root=root),
                manifest,
                {
                    "stress": wave_root / "stress",
                    "context": wave_root / "context",
                    "validation": wave_root / "validation",
                    "concentration": wave_root / "concentration",
                },
            )
            self.assertFalse(stability["evidence_level"].astype(str).str.contains("level_5", regex=False).any())
            self.assertFalse(bool(stability.iloc[0]["wf_cpcv_neighborhood_pass"]))

    def test_load_funding_reads_all_relevant_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            funding_dir = root / "parquet/funding"
            funding_dir.mkdir(parents=True)
            for i in range(5):
                pd.DataFrame([{
                    "timestamp": f"2025-01-0{i + 1}T00:00:00Z",
                    "fundingRate": i + 1,
                }]).to_parquet(funding_dir / f"PF_XBTUSD_2025010{i + 1}T000000.parquet", index=False)
            paths = {"funding": funding_dir}
            out = sweep.load_funding(paths, "PF_XBTUSD", pd.Timestamp("2025-01-31", tz="UTC"))
            self.assertEqual(len(out), 5)
            self.assertEqual(float(out["fundingRate"].sum()), 15.0)

    def test_fee_scenario_methodology_is_algebraic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kraken = root / "kraken"
            (kraken / "parquet/funding").mkdir(parents=True)
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(kraken_data_root=str(kraken)))
            sweep.stage_p2_cheap_hardening(ctx)
            text = (root / "mechanics/fee_scenario_methodology_report.md").read_text()
            self.assertIn("fee_scenario_algebraic_adjustment_not_full_replay", text)

    def test_canary_profile_limits_definitions_not_symbols_or_events(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prelaunch = root / "prelaunch/eligible_definition_manifest.csv"
            rows = []
            for i, kind in enumerate(["conservative", "plain_baseline", "aggressive", "intended_regime", "normal"]):
                rows.append({
                    "definition_id": f"d{i}",
                    "candidate_definition_id": f"d{i}",
                    "candidate_id": f"d{i}",
                    "parameter_vector_hash": f"p{i}",
                    "hypothesis_id": "H02" if i == 0 else "A4",
                    "family": "TSMOM",
                    "family_engine_id": "scheduled_tsmom_engine",
                    "definition_kind": kind,
                    "canonical": True,
                    "prelaunch_eligible": "True",
                })
            sweep.write_csv(prelaunch, rows)
            ctx = SimpleNamespace(
                run_root=root,
                args=SimpleNamespace(
                    phase_profile=sweep.TSMOM_RETEST_P1_CANARY_PHASE_PROFILE,
                    smoke=False,
                    max_wave_definitions=3,
                    canary_selection_mode="ultra_canary",
                    canary_materialization_budget=3,
                    canary_audit_sample_budget=2,
                    max_output_gb=80,
                    family_list="scheduled_tsmom_engine",
                ),
            )
            sweep.stage_engine_wave_plan_freeze(ctx)
            manifest = pd.read_csv(root / "canary/tsmom_canary_definition_manifest.csv")
            budget_report = (root / "canary/tsmom_canary_budget_report.md").read_text()
            self.assertEqual(len(manifest), 3)
            self.assertIn("Limits apply to candidate definitions only", budget_report)
            self.assertIn("Event sampling: `forbidden`", budget_report)
            self.assertIn("Symbol cap: `forbidden outside smoke`", budget_report)

    def test_canary_rejects_max_symbols_unless_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            args = sweep.parse_args([
                "--phase-profile", sweep.TSMOM_RETEST_P1_CANARY_PHASE_PROFILE,
                "--stage", "all",
                "--run-root", str(Path(td) / "run"),
                "--max-symbols", "5",
            ])
            with self.assertRaises(RuntimeError):
                sweep.init_context(args)

    def test_canary_hard_gate_stage_parity_with_tsmom_p1(self):
        canary = sweep.active_stage_list(sweep.parse_args(["--phase-profile", sweep.TSMOM_RETEST_P1_CANARY_PHASE_PROFILE, "--stage", "all"]))
        full = sweep.active_stage_list(sweep.parse_args(["--phase-profile", sweep.TSMOM_RETEST_P1_PHASE_PROFILE, "--stage", "all"]))
        for stage in [
            "prelaunch-contract-eligibility-gate",
            "engine-wave-plan-freeze",
            "engine-wave-loop",
            "cross-wave-summary",
            "decision-report",
            "compact-review-bundle",
        ]:
            self.assertIn(stage, canary)
            self.assertIn(stage, full)

    def test_tsmom_p1_canary_harness_repair_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.TSMOM_P1_CANARY_HARNESS_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.TSMOM_P1_CANARY_HARNESS_REPAIR_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.TSMOM_P1_CANARY_HARNESS_REPAIR_STAGES))

    def test_tsmom_canary_runtime_futility_repair_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.TSMOM_CANARY_RUNTIME_FUTILITY_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.TSMOM_CANARY_RUNTIME_FUTILITY_REPAIR_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.TSMOM_CANARY_RUNTIME_FUTILITY_REPAIR_STAGES))

    def test_semantic_duplicate_canary_deduplication_and_alias_mapping(self):
        base = {
            "definition_id": "d_h02",
            "candidate_definition_id": "c_h02",
            "hypothesis_id": "H02",
            "family_engine_id": "scheduled_tsmom_engine",
            "parameter_vector_hash": "same_param",
            "side": "long",
            "lookback_bars": 12,
            "hold_bars": 24,
            "stop_bps": 0,
            "threshold": 0,
            "entry_template": "tsmom_close_confirmed",
            "exit_template": "fixed_hold",
            "stop_template": "scheduled_rebalance",
            "regime_activation": "all_context_diagnostic",
            "row_semantics": "position_interval",
            "selected_symbol_universe_hash": "pit_u",
        }
        alias = {**base, "definition_id": "d_a4", "candidate_definition_id": "c_a4", "hypothesis_id": "A4"}
        deduped, alias_map, audit = sweep.deduplicate_tsmom_canary_definitions(pd.DataFrame([base, alias]))
        self.assertEqual(len(deduped), 1)
        self.assertEqual(len(alias_map), 1)
        self.assertIn("semantic_duplicate_collapsed", set(audit["status"].astype(str)))

    def test_materialized_futility_gate_stops_all_bad_candidates_without_family_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave_root = root / "waves/scheduled_tsmom_engine__wave_001"
            ledger_path = wave_root / "materialized/event_ledgers/negative.parquet"
            ledger_path.parent.mkdir(parents=True)
            ev = pd.DataFrame({
                "decision_ts": pd.date_range("2025-01-01", periods=120, freq="1h", tz="UTC"),
                "symbol": ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD"] * 40,
                "net_R": [-0.10] * 120,
                "gross_R": [-0.08] * 120,
                "fees_R": [-0.01] * 120,
                "funding_R": [-0.01] * 120,
                "slippage_R": [0.0] * 120,
                "regime_label": ["all_context_diagnostic"] * 120,
            })
            ev.to_parquet(ledger_path, index=False)
            manifest = pd.DataFrame([{
                "candidate_id": "c1",
                "candidate_definition_id": "c1",
                "parameter_vector_hash": "p1",
                "family_engine_id": "scheduled_tsmom_engine",
                "path": str(ledger_path.relative_to(root)),
                "materialization_scope": "definition_universe",
            }])
            ctx = SimpleNamespace(run_root=root)
            wave_dirs = {"root": wave_root}
            decision_summary, decision = sweep.write_materialized_futility_gate(ctx, wave_dirs, manifest)
            self.assertTrue(decision["futility_stop"])
            self.assertFalse(decision["family_rejected"])
            self.assertEqual(decision["pending_definitions_preserved_as"], "pending_compute")
            self.assertEqual(len(decision_summary), 1)

    def test_diagnostic_control_cap_blocks_clean_evidence_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave = root / "waves/w"
            event_path = wave / "materialized/event_ledgers/c.parquet"
            event_path.parent.mkdir(parents=True)
            pd.DataFrame({
                "decision_ts": pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC"),
                "symbol": ["PF_XBTUSD"] * 3,
                "net_R": [0.1, 0.2, -0.1],
                "funding_proxy_used": [False] * 3,
                "slippage_R": [-0.01] * 3,
                "mark_proxy_used": [False] * 3,
                "same_bar_ambiguity_flag": [False] * 3,
                "index_price_source": ["available"] * 3,
                "mark_liquidation_flag": [True] * 3,
                "label_cap_reason": [""] * 3,
            }).to_parquet(event_path, index=False)
            wave_dirs = {"root": wave, "controls": wave / "controls"}
            (wave / "controls").mkdir(parents=True)
            sweep.write_csv(wave / "controls/control_cap_diagnostics.csv", [{
                "candidate_definition_id": "c",
                "diagnostic_only_not_evidence": True,
                "cap_applied": False,
                "final_evidence_eligible": False,
            }])
            sweep.write_csv(wave / "controls/interval_overlap_purge_audit.csv", [])
            manifest = pd.DataFrame([{"candidate_id": "c", "candidate_definition_id": "c", "path": str(event_path.relative_to(root))}])
            control_summary = pd.DataFrame([{"candidate_id": "c", "control_evidence_label": "control_diagnostic_only_not_evidence"}])
            audit = sweep.collect_wave_evidence_cap_audit(SimpleNamespace(run_root=root), wave_dirs, manifest, pd.DataFrame(), control_summary, pd.DataFrame())
            reason = str(audit.iloc[0]["label_cap_reason"])
            self.assertIn("control_diagnostic_only_not_evidence", reason)
            self.assertFalse(bool(audit.iloc[0]["clean_evidence_allowed"]))

    def test_fast_canary_rejects_max_symbols_unless_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            args = sweep.parse_args([
                "--phase-profile", sweep.TSMOM_RETEST_P1_ULTRA_CANARY_FAST_PHASE_PROFILE,
                "--stage", "all",
                "--run-root", str(Path(td) / "run"),
                "--max-symbols", "5",
            ])
            with self.assertRaises(RuntimeError):
                sweep.init_context(args)

    def test_fast_canary_definition_limit_does_not_cap_symbol_or_event_scope(self):
        rows = []
        for i in range(5):
            rows.append({
                "definition_id": f"d{i}",
                "candidate_definition_id": f"c{i}",
                "hypothesis_id": "H02" if i == 0 else "A4",
                "family_engine_id": "scheduled_tsmom_engine",
                "parameter_vector_hash": f"p{i}",
                "side": "long",
                "lookback_bars": 12 + i,
                "hold_bars": 24,
                "entry_template": "tsmom_close_confirmed",
                "exit_template": "fixed_hold",
                "stop_template": "scheduled_rebalance",
                "regime_activation": "all_context_diagnostic",
                "row_semantics": "position_interval",
                "selected_symbol_universe": "PF_XBTUSD;PF_ETHUSD;PF_SOLUSD",
            })
        selected = sweep.select_tsmom_canary_definitions(pd.DataFrame(rows), 3)
        self.assertEqual(len(selected), 3)
        self.assertTrue((selected["selected_symbol_universe"].astype(str).str.contains("PF_ETHUSD")).all())

    def test_gate_accounting_repair_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.GATE_ACCOUNTING_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.GATE_ACCOUNTING_REPAIR_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.GATE_ACCOUNTING_REPAIR_STAGES))

    def test_materialized_futility_stop_cannot_continue_automation(self):
        ctx = SimpleNamespace(args=SimpleNamespace(phase_profile=sweep.TSMOM_RETEST_P1_ULTRA_CANARY_FAST_PHASE_PROFILE))
        action = sweep.wave_next_action_for_gate(ctx, hard_gate_pass=True, materialized_futility_stop=True)
        self.assertNotEqual(action, "continue_next_engine_wave_next")
        self.assertEqual(action, "analyze_tsmom_wave_now")

    def test_missing_p1_artifact_consumption_blocks_dependency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p1 = root / "p1"
            run = root / "run"
            p1.mkdir()
            ctx = SimpleNamespace(run_root=run, args=SimpleNamespace(p1_repair_root=str(p1), phase_profile=sweep.TSMOM_RETEST_P1_ULTRA_CANARY_FAST_PHASE_PROFILE))
            ok = sweep.consume_p1_dependency_artifacts(ctx)
            self.assertFalse(ok)
            pass_ok, blockers = sweep.p1_dependency_artifacts_pass(ctx)
            self.assertFalse(pass_ok)
            self.assertTrue(any("missing_p1_dependency_artifact" in b for b in blockers))

    def test_diagnostic_control_partial_retention_not_reported_all_controls(self):
        bars = self.bars(700)
        wave = 100 + pd.Series([float(np.sin(i / 7.0)) for i in range(len(bars))])
        bars["open"] = wave
        bars["close"] = wave + 0.04
        bars["high"] = bars["close"] + 0.6
        bars["low"] = bars["open"] - 0.6
        bars["mark_close"] = bars["close"]
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand["hold_bars"] = 48
        cand["threshold"] = 0.0005
        engine = sweep.ENGINES["scheduled_tsmom_engine"]
        ledger = engine.regenerate_materialized_ledger(bars, pd.DataFrame(), cand)
        controls, _, diag = sweep.build_real_null_controls(cand, engine, bars, pd.DataFrame(), ledger, nulls_per_event=3, retention_limit=5, diagnostic_fast_limit=5)
        self.assertEqual(diag["control_mode"], "diagnostic")
        self.assertEqual(diag["cap_scope"], "diagnostic_only_not_evidence")
        self.assertFalse(bool(diag["final_evidence_eligible"]))
        if diag["diagnostic_retention_partial"]:
            self.assertNotEqual(diag["cap_method"], "all_controls")
        self.assertLessEqual(len(controls), 5)

    def test_same_bar_stop_has_effective_nonzero_interval(self):
        ts = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
        bars = pd.DataFrame({
            "ts": ts,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.5, 101.0, 100.5, 100.5],
            "low": [99.5, 98.0, 99.0, 99.0],
            "close": [100.0, 99.5, 100.0, 100.0],
            "mark_close": [100.0, 99.5, 100.0, 100.0],
        })
        cand = self.candidate("liquid_continuation_breakout_engine", "liquid_continuation")
        cand["stop_bps"] = 100.0
        cand = sweep.identity_enriched_record(cand, sweep.ENGINES["liquid_continuation_breakout_engine"], symbol="PF_XBTUSD")
        addr = sweep.EventAddress(cand["candidate_id"], cand["definition_id"], "H01", "F", "liquid_continuation_breakout_engine", "PF_XBTUSD", 0, 0, ts[0], 1, 2, "trade_episode", "trade_episode_contract")
        ev = sweep.event_from_address(cand, bars, pd.DataFrame(), addr)
        self.assertIsNotNone(ev)
        self.assertEqual(pd.Timestamp(ev["exit_ts"]), pd.Timestamp(ev["entry_ts"]))
        self.assertGreater(pd.Timestamp(ev["effective_exit_ts_for_purge"]), pd.Timestamp(ev["entry_ts"]))
        audit = sweep.same_bar_interval_semantics_audit_frame(pd.DataFrame([ev]))
        self.assertEqual(audit.iloc[0]["status"], "pass")

    def test_candidate_library_current_state_is_unique_with_history(self):
        rows = [
            {"candidate_definition_id": "c1", "parameter_vector_hash": "p1", "candidate_library_state": "pending_compute", "evidence_level": "level_1"},
            {"candidate_definition_id": "c1", "parameter_vector_hash": "p1", "candidate_library_state": "near_miss_preserved", "evidence_level": "level_2"},
            {"candidate_definition_id": "c2", "parameter_vector_hash": "p2", "candidate_library_state": "pending_compute", "evidence_level": "level_1"},
        ]
        current, history, audit = sweep.deduplicate_candidate_library_current_state(rows)
        self.assertEqual(len(current[current["candidate_definition_id"].eq("c1")]), 1)
        self.assertEqual(len(history), 3)
        self.assertEqual(audit.iloc[0]["status"], "pass")

    def test_canary_dedup_enriches_nonblank_parameter_lineage(self):
        rows = [
            {"definition_id": "d_h02", "candidate_definition_id": "c_h02", "hypothesis_id": "H02", "family_engine_id": "scheduled_tsmom_engine", "side": "long", "lookback_bars": 12, "hold_bars": 48, "stop_bps": 100, "threshold": 0.001, "entry_template": "tsmom_close_confirmed", "exit_template": "fixed_hold", "stop_template": "scheduled_rebalance", "regime_activation": "all_context_diagnostic", "selected_symbol_universe_hash": "pit"},
            {"definition_id": "d_a4", "candidate_definition_id": "c_a4", "hypothesis_id": "A4", "family_engine_id": "scheduled_tsmom_engine", "side": "long", "lookback_bars": 12, "hold_bars": 48, "stop_bps": 100, "threshold": 0.001, "entry_template": "tsmom_close_confirmed", "exit_template": "fixed_hold", "stop_template": "scheduled_rebalance", "regime_activation": "all_context_diagnostic", "selected_symbol_universe_hash": "pit"},
        ]
        deduped, alias, audit = sweep.deduplicate_tsmom_canary_definitions(pd.DataFrame(rows))
        self.assertEqual(len(deduped), 1)
        self.assertEqual(len(alias), 1)
        self.assertFalse(audit["parameter_vector_hash"].astype(str).str.strip().isin(["", "nan", "None"]).any())
        self.assertFalse(audit["parameter_vector_json"].astype(str).str.strip().isin(["", "nan", "None"]).any())
        self.assertIn("alias_parameter_vector_hash", alias.columns)

    def test_positive_evidence_gate_fails_closed_without_required_audits(self):
        clean, reasons = sweep.positive_evidence_gate_for_candidate(
            cid="c1",
            hard_gate_pass=True,
            cap_row={},
            control_cap=pd.DataFrame(),
            control_summary=pd.DataFrame([{"candidate_id": "c1", "control_event_count": 10}]),
            gate={},
        )
        self.assertFalse(clean)
        self.assertIn("missing_mechanics_cap_audit", reasons)
        self.assertIn("missing_control_cap_diagnostics", reasons)

    def test_streaming_fold_summary_matches_materialized_sample(self):
        bars = self.bars(300)
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand["hold_bars"] = 48
        cand["threshold"] = 0.0005
        cand = sweep.identity_enriched_record(cand, sweep.ENGINES["scheduled_tsmom_engine"], symbol="PF_XBTUSD")
        accs = {"symbol_month": {}, "regime": {}, "year": {}, "month": {}, "fold": {}}
        metrics = sweep.compute_streaming_aggregate_and_group_summaries(sweep.ENGINES["scheduled_tsmom_engine"], bars, pd.DataFrame(), cand, accs)
        ledger = sweep.ENGINES["scheduled_tsmom_engine"].regenerate_materialized_ledger(bars, pd.DataFrame(), cand)
        self.assertEqual(int(metrics["events"]), len(ledger))
        self.assertAlmostEqual(float(metrics["net_R"]), float(pd.to_numeric(ledger["net_R"], errors="coerce").sum()), places=10)
        fold = sweep.finalize_streaming_summary_acc(accs["fold"])
        self.assertEqual(int(fold["events"].sum()) if not fold.empty else 0, len(ledger))

    def test_tsmom_candidate_set_audit_redesign_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.TSMOM_CANDIDATE_SET_AUDIT_REDESIGN_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.TSMOM_CANDIDATE_SET_AUDIT_REDESIGN_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.TSMOM_CANDIDATE_SET_AUDIT_REDESIGN_STAGES))

    def test_external_candidate_definition_manifest_prelaunch_uses_redesigned_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "redesigned.csv"
            rows = sweep.redesigned_tsmom_rows("ultra").head(2)
            rows.to_csv(manifest, index=False)
            ctx = SimpleNamespace(
                run_root=root / "run",
                args=SimpleNamespace(candidate_definition_manifest=str(manifest)),
            )
            sweep.stage_prelaunch_contract_eligibility_gate(ctx)
            eligible = pd.read_csv(ctx.run_root / "prelaunch/eligible_definition_manifest.csv")
            self.assertEqual(len(eligible), 2)
            self.assertTrue(eligible["definition_id"].astype(str).str.startswith("redesigned_tsmom_ultra").all())
            self.assertTrue(eligible["uses_external_candidate_definition_manifest"].astype(str).str.lower().isin({"true", "1"}).all())
            self.assertFalse(eligible["parameter_vector_hash"].astype(str).str.strip().isin(["", "nan", "None"]).any())

    def test_tsmom_universe_ordering_detects_alphabetical_not_liquidity(self):
        panel = pd.DataFrame([
            {"symbol": "PF_ZZZUSD", "bar_rows": 3000, "funding_rows": 100, "status": "available"},
            {"symbol": "PF_AAAUSD", "bar_rows": 10, "funding_rows": 1, "status": "available"},
        ])
        verdict = sweep.infer_tsmom_universe_ordering(["PF_AAAUSD", "PF_ZZZUSD"], panel)
        self.assertTrue(verdict["alphabetical_or_truncated"])
        self.assertFalse(verdict["pit_liquidity_based"])
        self.assertEqual(verdict["ordering_basis"], "alphabetical_or_symbol_name_order")

    def test_expected_major_audit_explains_eligible_excluded_symbol(self):
        panel = pd.DataFrame([
            {"symbol": "PF_XBTUSD", "bar_rows": 1000, "funding_rows": 100, "status": "available"},
            {"symbol": "PF_ETHUSD", "bar_rows": 900, "funding_rows": 90, "status": "available"},
            {"symbol": "PF_SOLUSD", "bar_rows": 800, "funding_rows": 80, "status": "available"},
        ])
        audit = sweep.tsmom_expected_major_audit(["PF_ETHUSD", "PF_XBTUSD"], panel)
        sol = audit[audit["symbol"].eq("PF_SOLUSD")].iloc[0]
        self.assertTrue(bool(sol["eligible_by_panel"]))
        self.assertFalse(bool(sol["included"]))
        self.assertIn("eligible_but_excluded", sol["inclusion_exclusion_reason"])

    def test_tsmom_time_unit_audit_flags_5m_churn(self):
        selected = pd.DataFrame([{
            "candidate_id": "c",
            "candidate_definition_id": "c",
            "side": "long",
            "lookback_bars": 12,
            "hold_bars": 24,
            "events": 1000,
            "active_symbols": 10,
            "active_months": 2,
        }])
        audit = sweep.tsmom_time_unit_audit_frame(selected)
        self.assertEqual(audit.iloc[0]["source_bar_timeframe"], "5m")
        self.assertLess(float(audit.iloc[0]["lookback_days"]), 1.0)
        self.assertEqual(audit.iloc[0]["time_unit_verdict"], "short_horizon_5m_churn")

    def test_redesigned_tsmom_definitions_have_explicit_units_and_unique_hashes(self):
        ultra = sweep.redesigned_tsmom_rows("ultra")
        audit = sweep.validate_redesigned_tsmom_manifest(ultra)
        self.assertFalse(audit["status"].astype(str).eq("fail").any())
        self.assertEqual(len(ultra), len(set(ultra["parameter_vector_hash"].astype(str))))
        for col in [
            "candidate_definition_id", "liquidity_tier", "universe_policy", "bar_timeframe",
            "lookback_days", "lookback_4h_bars", "rebalance_interval", "hold_interval",
            "vol_window_days", "vol_target", "parent_regime_gate", "funding_gate",
            "fee_scenario_handling", "why_this_definition",
        ]:
            self.assertIn(col, ultra.columns)
            self.assertFalse(ultra[col].astype(str).str.strip().isin(["", "nan", "None"]).any())

    def test_redesigned_tsmom_manifest_rejects_missing_time_units(self):
        bad = sweep.redesigned_tsmom_rows("ultra").head(1).drop(columns=["lookback_days"])
        audit = sweep.validate_redesigned_tsmom_manifest(bad)
        self.assertTrue(audit[audit["check"].eq("required_field_lookback_days")]["status"].eq("fail").any())

    def test_redesigned_execution_binding_audit_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.TSMOM_REDESIGNED_EXECUTION_BINDING_AUDIT_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.TSMOM_REDESIGNED_EXECUTION_BINDING_AUDIT_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.TSMOM_REDESIGNED_EXECUTION_BINDING_AUDIT_STAGES))

    def test_redesigned_manifest_schema_accepts_repaired_side_semantics(self):
        ultra = sweep.redesigned_tsmom_rows("ultra")
        audit = sweep.redesigned_manifest_schema_audit(ultra, manifest_name="ultra")
        side_row = audit[audit["check"].eq("engine_supported_side_values")].iloc[0]
        self.assertEqual(side_row["status"], "pass")
        self.assertEqual(int(side_row["bad_rows"]), 0)

    def test_redesigned_manifest_schema_flags_unknown_side(self):
        bad = sweep.redesigned_tsmom_rows("ultra")
        bad.loc[0, "side"] = "bad_side"
        audit = sweep.redesigned_manifest_schema_audit(bad, manifest_name="ultra")
        side_row = audit[audit["check"].eq("engine_supported_side_values")].iloc[0]
        self.assertEqual(side_row["status"], "fail")
        self.assertEqual(int(side_row["bad_rows"]), 1)

    def test_redesigned_binding_repair_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.TSMOM_REDESIGNED_BINDING_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.TSMOM_REDESIGNED_BINDING_REPAIR_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.TSMOM_REDESIGNED_BINDING_REPAIR_STAGES))

    def test_manifest_field_binding_blocks_report_only_active_fields(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = SimpleNamespace(run_root=Path(td), args=SimpleNamespace())
            rows = sweep.manifest_field_binding_rows(ctx)
            blockers = rows[rows["launch_blocker"].astype(bool)]
            self.assertIn("universe_policy", set(blockers["field"]))
            self.assertIn("parent_regime_gate", set(blockers["field"]))
            self.assertIn("funding_gate", set(blockers["field"]))
            self.assertIn("vol_target", set(blockers["field"]))

    def test_redesigned_daily_cadence_does_not_fire_every_5m_on_engine_addresses(self):
        bars = self.bars(13000)
        row = sweep.redesigned_tsmom_rows("ultra").iloc[0].to_dict()
        row = sweep.identity_enriched_record({**row, "symbol": "PF_XBTUSD"}, sweep.ENGINES["scheduled_tsmom_engine"], symbol="PF_XBTUSD")
        addresses = sweep.ENGINES["scheduled_tsmom_engine"].enumerate_valid_event_addresses(bars, row)
        self.assertGreater(len(addresses), 2)
        ts = pd.Series([a.decision_ts for a in addresses]).sort_values()
        min_spacing_minutes = ts.diff().dropna().dt.total_seconds().div(60).min()
        self.assertGreaterEqual(float(min_spacing_minutes), 24 * 60)

    def test_redesigned_manifest_loader_marks_external_manifest_source(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ultra.csv"
            sweep.redesigned_tsmom_rows("ultra").to_csv(path, index=False)
            ctx = SimpleNamespace(run_root=Path(td) / "run", args=SimpleNamespace())
            loaded = sweep.load_candidate_definition_manifest(ctx, path)
            self.assertEqual(len(loaded), 5)
            self.assertTrue(loaded["uses_external_candidate_definition_manifest"].astype(str).str.lower().isin({"true", "1"}).all())
            self.assertTrue(loaded["candidate_definition_id"].astype(str).str.startswith("redesigned_tsmom_ultra").all())

    def test_pit_major_anchor_not_forced_when_inactive_and_tier_ab_not_forced(self):
        checkpoint = pd.Timestamp("2025-06-01", tz="UTC")
        panel_rows = [{
            "symbol": "PF_XBTUSD",
            "first_ts": "2025-01-01T00:00:00Z",
            "last_ts": "2025-03-01T00:00:00Z",
            "bar_rows": 1,
            "funding_rows": 0,
        }]
        for i in range(41):
            panel_rows.append({
                "symbol": f"PF_ALT{i:02d}USD",
                "first_ts": "2025-01-01T00:00:00Z",
                "last_ts": "2025-12-31T00:00:00Z",
                "bar_rows": 10000 - i,
                "funding_rows": 0,
            })
        panel = pd.DataFrame(panel_rows)
        top_symbols, top_audit = sweep.tsmom_policy_symbols_at_checkpoint({"universe_policy": "pit_liquidity_top_majors"}, panel, checkpoint)
        xbt = top_audit[top_audit["symbol"].eq("PF_XBTUSD")].iloc[0]
        self.assertFalse(bool(xbt["eligible"]))
        self.assertFalse(bool(xbt["included"]))
        self.assertNotIn("PF_XBTUSD", top_symbols)
        tier_symbols, tier_audit = sweep.tsmom_policy_symbols_at_checkpoint({"universe_policy": "pit_liquidity_tier_ab"}, panel, checkpoint)
        self.assertEqual(len(tier_symbols), sweep.TSMOM_TIER_AB_UNIVERSE_LIMIT)
        self.assertNotIn("PF_XBTUSD", tier_symbols)
        xbt_tier = tier_audit[tier_audit["symbol"].eq("PF_XBTUSD")].iloc[0]
        self.assertEqual(xbt_tier["inclusion_exclusion_reason"], "after_symbol_panel_end")

    def test_funding_gate_missing_data_modes(self):
        ts = pd.Timestamp("2025-06-01", tz="UTC")
        exclude = sweep.evaluate_funding_gate({"funding_gate": "exclude_top_decile_positive_funding"}, pd.DataFrame(), ts)
        aware = sweep.evaluate_funding_gate({"funding_gate": "funding_aware_cap"}, pd.DataFrame(), ts)
        required = sweep.evaluate_funding_gate({"funding_gate": "funding_extreme_required"}, pd.DataFrame(), ts)
        self.assertTrue(exclude["allowed"])
        self.assertEqual(exclude["cap"], "funding_gate_unavailable_cap")
        self.assertTrue(aware["allowed"])
        self.assertEqual(aware["cap"], "funding_aware_cap_missing_or_insufficient")
        self.assertFalse(required["allowed"])
        self.assertEqual(required["skip_reason"], "funding_extreme_unavailable_skip")

    def test_exclude_top_20pct_positive_funding_gate(self):
        ts = pd.Timestamp("2025-06-01T00:00:00Z")
        funding = pd.DataFrame({
            "timestamp": pd.date_range("2025-05-01T00:00:00Z", periods=40, freq="8h", tz="UTC"),
            "fundingRate": np.linspace(-0.001, 0.001, 40),
            "relativeFundingRate": np.linspace(-0.001, 0.001, 40),
        })
        res = sweep.evaluate_funding_gate({"funding_gate": "exclude_top_20pct_positive_funding"}, funding, ts)
        self.assertFalse(res["allowed"])
        self.assertEqual(res["skip_reason"], "top_20pct_positive_funding_skip")
        missing = sweep.evaluate_funding_gate({"funding_gate": "exclude_top_20pct_positive_funding"}, pd.DataFrame(), ts)
        self.assertTrue(missing["allowed"])
        self.assertEqual(missing["cap"], "funding_gate_unavailable_cap")

    def test_raw_and_scaled_r_are_preserved_for_vol_target_overlay(self):
        bars = self.bars(800)
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand.update({
            "side": "long_flat",
            "threshold": 0.0,
            "lookback_bars": 12,
            "hold_bars": 24,
            "vol_window_days": 1.0,
            "vol_target": 0.2,
        })
        cand = sweep.identity_enriched_record(cand, sweep.ENGINES["scheduled_tsmom_engine"], symbol="PF_XBTUSD")
        ledger = sweep.ENGINES["scheduled_tsmom_engine"].regenerate_materialized_ledger(bars, pd.DataFrame(), cand)
        self.assertFalse(ledger.empty)
        for col in ["raw_gross_R", "raw_fee_R", "raw_funding_R", "raw_slippage_R", "raw_net_R", "vol_scale", "scaled_net_R"]:
            self.assertIn(col, ledger.columns)
        self.assertTrue(np.allclose(pd.to_numeric(ledger["net_R"], errors="coerce"), pd.to_numeric(ledger["scaled_net_R"], errors="coerce")))
        self.assertTrue((pd.to_numeric(ledger["vol_scale"], errors="coerce") > 0).all())

    def test_parent_gate_missing_warmup_skips_without_future_fill(self):
        with tempfile.TemporaryDirectory() as td:
            bars = self.bars(100)
            result = sweep.evaluate_parent_regime_gate(
                {"parent_regime_gate": "btc_eth_trend_up", "kraken_data_root": td},
                bars,
                pd.Timestamp("2025-01-02", tz="UTC"),
            )
            self.assertFalse(result["allowed"])
            self.assertEqual(result["skip_reason"], "parent_gate_warmup_skip")
            self.assertEqual(result["feature_source_ts"], str(pd.Timestamp("2025-01-02", tz="UTC")))

    def test_v2_policy_hashes_are_included_in_parameter_hash(self):
        v1 = sweep.redesigned_tsmom_rows("ultra")
        v2 = sweep.redesigned_tsmom_v2_rows("ultra")
        for col in [
            "universe_policy_hash",
            "parent_regime_policy_hash",
            "funding_gate_policy_hash",
            "vol_target_sizing_policy_hash",
            "side_semantics_policy_hash",
        ]:
            self.assertIn(col, v2.columns)
            self.assertFalse(v2[col].astype(str).str.strip().isin(["", "nan", "None"]).any())
        self.assertEqual(len(v2), len(set(v2["parameter_vector_hash"].astype(str))))
        merged = v1[["candidate_definition_id", "parameter_vector_hash"]].merge(
            v2[["candidate_definition_id", "parameter_vector_hash"]],
            on="candidate_definition_id",
            suffixes=("_v1", "_v2"),
        )
        self.assertTrue((merged["parameter_vector_hash_v1"] != merged["parameter_vector_hash_v2"]).all())

    def test_short_diagnostic_side_is_capped_and_not_primary(self):
        self.assertEqual(sweep.tsmom_side_direction("short_diagnostic"), "short")
        self.assertIn("short_diagnostic_cap", sweep.tsmom_side_label_cap("short_diagnostic"))

    def test_universe_binding_repair_v2_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.TSMOM_UNIVERSE_BINDING_REPAIR_V2_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.TSMOM_UNIVERSE_BINDING_REPAIR_V2_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.TSMOM_UNIVERSE_BINDING_REPAIR_V2_STAGES))

    def test_max_symbols_rejected_outside_smoke_for_any_profile(self):
        with tempfile.TemporaryDirectory() as td:
            args = sweep.parse_args([
                "--phase-profile", sweep.TSMOM_UNIVERSE_BINDING_REPAIR_V2_PHASE_PROFILE,
                "--stage", "all",
                "--run-root", str(Path(td) / "run"),
                "--max-symbols", "3",
            ])
            with self.assertRaises(RuntimeError):
                sweep.init_context(args)

    def test_expected_major_present_in_panel_not_marked_missing_by_v3_policy(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = SimpleNamespace(args=SimpleNamespace(kraken_data_root=td), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-12-31", tz="UTC"))
            checkpoint = pd.Timestamp("2025-06-01", tz="UTC")
            panel = pd.DataFrame([
                {"symbol": "PF_XBTUSD", "start_ts": "2024-01-01T00:00:00Z", "end_ts": "2025-12-31T00:00:00Z", "status": "available", "bar_rows": 100, "funding_rows": 10},
                {"symbol": "PF_ETHUSD", "start_ts": "2024-01-01T00:00:00Z", "end_ts": "2025-12-31T00:00:00Z", "status": "available", "bar_rows": 90, "funding_rows": 10},
                {"symbol": "PF_ALTUSD", "start_ts": "2024-01-01T00:00:00Z", "end_ts": "2025-12-31T00:00:00Z", "status": "available", "bar_rows": 80, "funding_rows": 0},
            ])
            symbols, audit = sweep.tsmom_policy_symbols_at_checkpoint_v3(ctx, {"universe_policy": "pit_liquidity_top_majors", "top_major_target_size": 2}, panel, checkpoint)
            self.assertIn("PF_XBTUSD", symbols)
            row = audit[audit["symbol"].eq("PF_XBTUSD")].iloc[0]
            self.assertTrue(bool(row["eligible_at_checkpoint"]))
            self.assertNotEqual(row["inclusion_exclusion_reason"], "symbol_missing_from_panel")

    def test_v3_top_major_and_tier_ab_target_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = SimpleNamespace(args=SimpleNamespace(kraken_data_root=td), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-12-31", tz="UTC"))
            checkpoint = pd.Timestamp("2025-06-01", tz="UTC")
            rows = []
            for sym in [*sweep.REDESIGNED_EXPECTED_MAJORS, *[f"PF_ALT{i:02d}USD" for i in range(50)]]:
                rows.append({"symbol": sym, "start_ts": "2024-01-01T00:00:00Z", "end_ts": "2025-12-31T00:00:00Z", "status": "available", "bar_rows": 100, "funding_rows": 0})
            panel = pd.DataFrame(rows)
            top_symbols, _ = sweep.tsmom_policy_symbols_at_checkpoint_v3(ctx, {"universe_policy": "pit_liquidity_top_majors", "top_major_target_size": 8}, panel, checkpoint)
            tier_symbols, _ = sweep.tsmom_policy_symbols_at_checkpoint_v3(ctx, {"universe_policy": "pit_liquidity_tier_ab", "tier_ab_target_size": 30}, panel, checkpoint)
            self.assertEqual(len(top_symbols), 8)
            self.assertEqual(len(tier_symbols), 30)
            self.assertTrue(set(top_symbols).issubset(set(panel["symbol"])))

    def test_universe_alphabetical_first_n_detector(self):
        all_symbols = ["PF_BUSD", "PF_AUSD", "PF_CUSD"]
        self.assertTrue(sweep.universe_symbols_are_alphabetical_first_n(["PF_AUSD", "PF_BUSD"], all_symbols))
        self.assertFalse(sweep.universe_symbols_are_alphabetical_first_n(["PF_CUSD", "PF_AUSD"], all_symbols))

    def test_v3_manifest_adds_universe_policy_version_and_target_hashes(self):
        v3 = sweep.redesigned_tsmom_v3_rows("ultra")
        self.assertIn("universe_policy_version", v3.columns)
        self.assertIn("top_major_target_size", v3.columns)
        self.assertIn("tier_ab_target_size", v3.columns)
        self.assertFalse(v3["parameter_vector_hash"].astype(str).str.strip().isin(["", "nan", "None"]).any())
        self.assertEqual(len(v3), len(set(v3["parameter_vector_hash"].astype(str))))

    def test_fast_mechanical_event_semantics_repair_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.TSMOM_FAST_MECHANICAL_EVENT_SEMANTICS_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.TSMOM_FAST_MECHANICAL_EVENT_SEMANTICS_REPAIR_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.TSMOM_FAST_MECHANICAL_EVENT_SEMANTICS_REPAIR_STAGES))

    def test_streaming_aggregate_preserves_active_tsmom_fields(self):
        bars = self.bars(900)
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand.update({
            "side": "long_flat",
            "lookback_bars": 12,
            "hold_bars": 48,
            "funding_gate": "funding_aware_cap",
            "parent_regime_gate": "all_context_diagnostic",
            "vol_target": 0.25,
            "vol_window_days": 1,
            "universe_policy": "pit_liquidity_top_majors",
            "bar_timeframe": "4h",
        })
        cand = sweep.identity_enriched_record(cand, sweep.ENGINES["scheduled_tsmom_engine"], symbol="PF_XBTUSD")
        agg = sweep.ENGINES["scheduled_tsmom_engine"].compute_exact_aggregate_metrics(bars, pd.DataFrame(), cand)
        self.assertEqual(agg["funding_gate"], "funding_aware_cap")
        self.assertEqual(agg["vol_target"], 0.25)
        self.assertEqual(agg["universe_policy"], "pit_liquidity_top_majors")
        self.assertIn("raw_net_R", agg)
        self.assertIn("scaled_net_R", agg)

    def test_event_row_contains_gate_and_vol_target_binding_fields(self):
        bars = self.bars(900)
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand.update({
            "side": "long_flat",
            "threshold": 0.0,
            "lookback_bars": 12,
            "hold_bars": 48,
            "funding_gate": "funding_aware_cap",
            "parent_regime_gate": "all_context_diagnostic",
            "vol_target": 0.25,
            "vol_window_days": 1,
        })
        cand = sweep.identity_enriched_record(cand, sweep.ENGINES["scheduled_tsmom_engine"], symbol="PF_XBTUSD")
        ledger = sweep.ENGINES["scheduled_tsmom_engine"].regenerate_materialized_ledger(bars, pd.DataFrame(), cand)
        self.assertFalse(ledger.empty)
        row = ledger.iloc[0]
        for col in [
            "funding_gate_pass", "funding_feature_source_ts", "funding_exact_available", "funding_missing_action",
            "parent_regime_gate", "parent_gate_pass", "parent_gate_feature_source_ts",
            "vol_scale", "raw_net_R", "scaled_net_R",
        ]:
            self.assertIn(col, ledger.columns)
        self.assertEqual(row["funding_gate"], "funding_aware_cap")
        self.assertEqual(row["funding_gate_status"], "missing_exact_allowed_capped")

    def test_tsmom_event_count_bounds_detect_5m_firing(self):
        cand = self.candidate("scheduled_tsmom_engine", "tsmom")
        cand.update({"candidate_definition_id": "c", "candidate_id": "c", "parameter_vector_hash": "p", "hold_bars": 288})
        events = pd.DataFrame([
            {"candidate_definition_id": "c", "parameter_vector_hash": "p", "symbol": "PF_XBTUSD", "decision_ts": "2025-01-01T00:00:00Z"},
            {"candidate_definition_id": "c", "parameter_vector_hash": "p", "symbol": "PF_XBTUSD", "decision_ts": "2025-01-01T00:05:00Z"},
        ])
        audit = sweep.tsmom_event_count_theoretical_vs_actual(events, pd.DataFrame([cand]))
        self.assertTrue(audit["daily_8h_4h_fires_every_5m"].any())
        self.assertTrue(audit["status"].eq("fail").any())

    def test_tsmom_overlap_audit_detects_overlapping_intervals(self):
        events = pd.DataFrame([
            {"candidate_definition_id": "c", "parameter_vector_hash": "p", "symbol": "PF_XBTUSD", "entry_ts": "2025-01-01T00:00:00Z", "exit_interval_end_ts": "2025-01-01T08:00:00Z"},
            {"candidate_definition_id": "c", "parameter_vector_hash": "p", "symbol": "PF_XBTUSD", "entry_ts": "2025-01-01T04:00:00Z", "exit_interval_end_ts": "2025-01-01T12:00:00Z"},
        ])
        audit = sweep.tsmom_overlap_audit_frame(events)
        self.assertTrue(audit["overlapping_exposure_accounting_cap"].any())
        self.assertTrue(audit["status"].eq("fail").any())

    def test_known_boundary_funding_R_fixture(self):
        funding = pd.DataFrame({
            "timestamp": pd.to_datetime(["2025-01-01T08:00:00Z", "2025-01-01T16:00:00Z"], utc=True),
            "fundingRate": [0.001, -0.0005],
        })
        fields = sweep.funding_components_for_event(
            funding,
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-01T16:00:00Z"),
            side="long",
            entry_price=100.0,
            risk_price=10.0,
        )
        self.assertEqual(fields["funding_boundary_count_exact"], 2)
        self.assertFalse(fields["funding_proxy_used"])
        self.assertAlmostEqual(fields["funding_R"], -((0.001 - 0.0005) * 100.0 / 10.0), places=12)

    def test_exact_funding_uses_relative_funding_rate_when_present(self):
        funding = pd.DataFrame({
            "timestamp": pd.to_datetime(["2025-01-01T08:00:00Z"], utc=True),
            "fundingRate": [2.5],
            "relativeFundingRate": [0.000025],
        })
        fields = sweep.funding_components_for_event(
            funding,
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-01T08:00:00Z"),
            side="long",
            entry_price=100.0,
            risk_price=10.0,
        )
        self.assertEqual(fields["funding_rate_source"], "relativeFundingRate")
        self.assertAlmostEqual(fields["funding_R"], -(0.000025 * 100.0 / 10.0), places=12)

    def test_vol_target_fixture_non_unit_scales_preserve_raw_R(self):
        ts = pd.date_range("2025-01-01", periods=400, freq="5min", tz="UTC")
        low_vol = pd.DataFrame({"ts": ts, "close": 100.0 + np.arange(400) * 0.001})
        high_vol = pd.DataFrame({"ts": ts, "close": 100.0 + np.sin(np.arange(400)) * 20.0})
        cand = {"vol_target": 0.5, "vol_window_days": 1, "vol_scale_min": 0.25, "vol_scale_max": 2.0}
        low = sweep.vol_target_scale_for_event(cand, low_vol, ts[-1])
        high = sweep.vol_target_scale_for_event(cand, high_vol, ts[-1])
        self.assertGreater(low["vol_scale"], 1.0)
        self.assertLess(high["vol_scale"], 1.0)
        raw = {"raw_gross_R": 1.0, "raw_fee_R": -0.1, "raw_funding_R": 0.05, "raw_slippage_R": 0.0}
        scaled_net = (raw["raw_gross_R"] + raw["raw_fee_R"] + raw["raw_funding_R"] + raw["raw_slippage_R"]) * high["vol_scale"]
        self.assertNotAlmostEqual(scaled_net, raw["raw_gross_R"] + raw["raw_fee_R"] + raw["raw_funding_R"], places=6)

    def test_scope_aware_audit_compares_matching_train_internal_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "waves/scheduled_tsmom_engine__wave_001/audit"
            agg_dir = root / "waves/scheduled_tsmom_engine__wave_001/aggregate"
            mat_dir = root / "waves/scheduled_tsmom_engine__wave_001/materialized/event_ledgers"
            out_dir.mkdir(parents=True)
            agg_dir.mkdir(parents=True)
            mat_dir.mkdir(parents=True)
            ctx = SimpleNamespace(run_root=root)
            sweep.write_csv(root / "validation/dev_eval_split_manifest.csv", [{"split_ts": "2025-07-01T00:00:00Z"}])
            aggregate = pd.DataFrame([{
                "candidate_id": "c", "candidate_definition_id": "c", "candidate_symbol_id": "cs", "definition_id": "d",
                "hypothesis_id": "H", "family": "F", "family_engine_id": "scheduled_tsmom_engine", "symbol": "PF_XBTUSD",
                "parameter_vector_hash": "p", "events": 2, "net_R": 3.0, "gross_R": 4.0, "fees_R": -1.0, "funding_R": 0.0,
                "raw_net_R": 3.0, "raw_gross_R": 4.0, "raw_fee_R": -1.0, "raw_funding_R": 0.0,
                "scaled_net_R": 3.0, "scaled_gross_R": 4.0, "scaled_fee_R": -1.0, "scaled_funding_R": 0.0,
            }])
            aggregate.to_parquet(agg_dir / "all_candidate_aggregate_summary.parquet", index=False)
            months = pd.DataFrame([
                {"candidate_id": "c", "candidate_definition_id": "c", "symbol": "PF_XBTUSD", "month": "2025-06", "events": 1, "net_R": 1.0, "gross_R": 2.0, "fees_R": -1.0, "funding_R": 0.0, "raw_net_R": 1.0, "scaled_net_R": 1.0},
                {"candidate_id": "c", "candidate_definition_id": "c", "symbol": "PF_XBTUSD", "month": "2025-07", "events": 1, "net_R": 2.0, "gross_R": 2.0, "fees_R": 0.0, "funding_R": 0.0, "raw_net_R": 2.0, "scaled_net_R": 2.0},
            ])
            months.to_parquet(agg_dir / "candidate_symbol_month_summary.parquet", index=False)
            ledger = pd.DataFrame([
                {"candidate_id": "c", "candidate_definition_id": "c", "parameter_vector_hash": "p", "family_engine_id": "scheduled_tsmom_engine", "symbol": "PF_XBTUSD", "decision_ts": "2025-06-15T00:00:00Z", "net_R": 1.0, "gross_R": 2.0, "fees_R": -1.0, "funding_R": 0.0, "raw_net_R": 1.0, "raw_gross_R": 2.0, "raw_fee_R": -1.0, "raw_funding_R": 0.0, "scaled_net_R": 1.0, "scaled_gross_R": 2.0, "scaled_fee_R": -1.0, "scaled_funding_R": 0.0},
                {"candidate_id": "c", "candidate_definition_id": "c", "parameter_vector_hash": "p", "family_engine_id": "scheduled_tsmom_engine", "symbol": "PF_XBTUSD", "decision_ts": "2025-07-15T00:00:00Z", "net_R": 2.0, "gross_R": 2.0, "fees_R": 0.0, "funding_R": 0.0, "raw_net_R": 2.0, "raw_gross_R": 2.0, "raw_fee_R": 0.0, "raw_funding_R": 0.0, "scaled_net_R": 2.0, "scaled_gross_R": 2.0, "scaled_fee_R": 0.0, "scaled_funding_R": 0.0},
            ])
            ledger_path = mat_dir / "c.parquet"
            ledger.to_parquet(ledger_path, index=False)
            candidates = pd.DataFrame([{"candidate_id": "c", "candidate_definition_id": "c", "parameter_vector_hash": "p"}])
            manifest = pd.DataFrame([{"candidate_id": "c", "candidate_definition_id": "c", "parameter_vector_hash": "p", "family_engine_id": "scheduled_tsmom_engine", "materialization_scope": "definition_universe", "symbol_count_materialized": 1, "event_rows": 2, "path": str(ledger_path.relative_to(root))}])
            audit = sweep.aggregate_vs_materialized_wave_audit(ctx, candidates, manifest, out_dir)
            self.assertFalse(audit[audit["status"].eq("fail")].shape[0])
            self.assertIn("proposal_train_segment", set(audit["scope"]))
            self.assertIn("internal_eval_train_segment", set(audit["scope"]))

    def test_pre_run_lineage_gate_fails_missing_vol_scale_bounds(self):
        rows = sweep.redesigned_tsmom_v3_rows("ultra").copy()
        rows = rows.drop(columns=[c for c in ["vol_scale_min", "vol_scale_max"] if c in rows.columns])
        audit = sweep.pre_run_parameter_lineage_gate_frame(rows, context="v3_missing_vol_bounds")
        failed = set(audit[audit["status"].eq("fail")]["check"].astype(str))
        self.assertIn("required_field_vol_scale_min", failed)
        self.assertIn("required_field_vol_scale_max", failed)
        self.assertIn("parameter_json_contains_policy_hashes_and_vol_bounds", failed)

    def test_v4_hashes_include_vol_scale_bounds_and_policy_hashes(self):
        v3 = sweep.redesigned_tsmom_v3_rows("ultra")
        v4 = sweep.redesigned_tsmom_v4_rows("ultra")
        merged = v4.merge(v3[["candidate_definition_id", "parameter_vector_hash"]], on="candidate_definition_id", how="left", suffixes=("_v4", "_v3"))
        self.assertTrue((merged["parameter_vector_hash_v4"].astype(str) != merged["parameter_vector_hash_v3"].astype(str)).all())
        payload = sweep.parameter_json_payload(v4.iloc[0]["parameter_vector_json"])
        for key in [
            "vol_scale_min", "vol_scale_max", "universe_policy_hash", "parent_regime_policy_hash",
            "funding_gate_policy_hash", "vol_target_sizing_policy_hash", "side_semantics_policy_hash",
        ]:
            self.assertIn(key, payload)
        self.assertEqual(payload["vol_scale_min"], 0.25)
        self.assertEqual(payload["vol_scale_max"], 2.0)
        self.assertTrue(sweep.pre_run_parameter_lineage_gate_frame(v4, context="v4")["status"].eq("pass").all())

    def test_pre_run_lineage_gate_bound_to_future_tsmom_canary_profiles(self):
        self.assertIn("prelaunch-contract-eligibility-gate", sweep.active_stage_list(sweep.parse_args(["--phase-profile", sweep.TSMOM_RETEST_P1_ULTRA_CANARY_FAST_PHASE_PROFILE, "--stage", "all"])))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(run_root=root, args=SimpleNamespace(phase_profile=sweep.TSMOM_ULTRA_LINEAGE_FUTILITY_TRIAGE_PHASE_PROFILE))
            for rel in ["gate", "redesign", "lineage"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            sweep.stage_vol_scale_lineage_contract_repair(ctx)
            sweep.stage_pre_run_lineage_gate_repair(ctx)
            binding = pd.read_csv(root / "gate/pre_run_lineage_gate_profile_binding_audit.csv")
            self.assertIn(sweep.TSMOM_RETEST_P1_ULTRA_CANARY_FAST_PHASE_PROFILE, set(binding["phase_profile"]))
            self.assertTrue(binding["bound_before_aggregate_materialization"].astype(str).str.lower().isin({"true", "1"}).all())

    def test_run_level_tsmom_convenience_audits_are_emitted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(run_root=root)
            for rel in ["semantics", "funding", "regime", "sizing", "audit"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            events = pd.DataFrame([{
                "candidate_definition_id": "c",
                "parameter_vector_hash": "p",
                "symbol": "PF_XBTUSD",
                "event_id": "e1",
                "decision_ts": "2025-01-01T00:00:00Z",
                "entry_ts": "2025-01-01T00:00:00Z",
                "exit_interval_end_ts": "2025-01-01T08:00:00Z",
                "funding_gate": "funding_aware_cap",
                "funding_gate_status": "pass",
                "funding_gate_pass": True,
                "funding_feature_source_ts": "2025-01-01T00:00:00Z",
                "funding_exact_available": True,
                "funding_missing_action": "",
                "funding_label_cap_reason": "",
                "parent_regime_gate": "btc_eth_parent_trend",
                "parent_gate_status": "pass",
                "parent_gate_pass": True,
                "parent_gate_feature_source_ts": "2025-01-01T00:00:00Z",
                "parent_btc_feature_source_ts": "2025-01-01T00:00:00Z",
                "parent_eth_feature_source_ts": "2025-01-01T00:00:00Z",
                "parent_gate_missing_action": "",
                "vol_scale": 0.5,
                "vol_target": 0.35,
                "vol_target_status": "pass",
                "raw_net_R": 1.0,
                "scaled_net_R": 0.5,
                "raw_gross_R": 1.2,
                "scaled_gross_R": 0.6,
                "raw_fee_R": -0.1,
                "scaled_fee_R": -0.05,
                "raw_funding_R": -0.1,
                "scaled_funding_R": -0.05,
                "funding_boundary_count_exact": 1,
                "funding_boundary_count_proxy": 0,
            }])
            defs = pd.DataFrame([{"candidate_definition_id": "c", "parameter_vector_hash": "p", "hold_bars": 96}])
            sweep.write_run_level_tsmom_convenience_audits(ctx, events, defs)
            for rel in [
                "semantics/tsmom_event_count_sanity_audit.csv",
                "semantics/tsmom_overlap_audit.csv",
                "funding/funding_gate_event_row_binding_audit.csv",
                "regime/parent_gate_event_row_binding_audit.csv",
                "sizing/vol_target_event_row_binding_audit.csv",
                "funding/funding_r_sanity_audit.csv",
                "funding/funding_boundary_count_audit.csv",
                "audit/run_level_convenience_audit_output_report.md",
            ]:
                self.assertTrue((root / rel).exists(), rel)

    def test_ultra_futility_triage_decision_does_not_recommend_longer_run_when_all_weak(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(run_root=root)
            (root / "triage").mkdir(parents=True, exist_ok=True)
            sweep.write_json(root / "triage/ultra_definition_futility_decision.json", {
                "all_current_ultra_definitions_materially_negative": True,
                "all_materialized_definitions_materially_negative": True,
                "current_ultra_definition_set_classification": "current_ultra_definition_set_weak_pending_redesign_or_alternative_tsmom_definitions",
                "nonmaterialized_aggregate_definition_worth_targeted_materialization": False,
            })
            sweep.write_csv(root / "triage/definition_failure_mode_classification.csv", [{"candidate_definition_id": "c", "dominant_failure_mode": "gross alpha negative"}])
            sweep.stage_rerun_or_redesign_decision(ctx)
            decision = sweep.read_json(root / "triage/rerun_or_redesign_decision.json", {})
            self.assertEqual(decision["recommended_next_workflow_step"], "current_ultra_set_weak_redesign_definitions_next")

    def test_standard_v4_pretriage_profile_stage_list(self):
        stages = sweep.active_stage_list(sweep.parse_args(["--phase-profile", sweep.TSMOM_STANDARD_V4_AGG_PRETRIAGE_PHASE_PROFILE, "--stage", "all"]))
        self.assertEqual(stages, list(sweep.TSMOM_STANDARD_V4_AGG_PRETRIAGE_STAGES))
        self.assertNotIn("engine-wave-loop", stages)
        self.assertNotIn("per-wave-materialized-ledger-regeneration", stages)

    def test_standard_v4_runtime_gate_blocks_aggregate_when_over_30_minutes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ["prelaunch", "triage"]:
                (root / rel).mkdir(parents=True, exist_ok=True)
            defs = pd.DataFrame([
                {"candidate_definition_id": "d1", "hypothesis_id": "H02", "side": "long", "universe_policy": "pit_liquidity_top_majors", "lookback_bars": 2880, "hold_bars": 288, "rebalance_interval": "1d"},
            ])
            sweep.write_csv(root / "prelaunch/standard_v4_definition_manifest.csv", defs)
            sweep.write_json(root / "triage/standard_v4_runtime_estimate.json", {"aggregate_screen_allowed": False, "estimated_minutes": 60.0})
            ctx = SimpleNamespace(run_root=root)
            sweep.stage_standard_v4_aggregate_only_screen(ctx)
            screen = pd.read_csv(root / "triage/standard_v4_aggregate_screen.csv")
            self.assertEqual(screen.iloc[0]["aggregate_only_evidence_label"], "aggregate_screen_not_run_runtime_gate")
            self.assertIn("runtime_estimate_exceeds_30_minutes", screen.iloc[0]["skip_reason"])

    def test_standard_v4_nonfutility_requires_positive_threshold_not_less_bad(self):
        weak = {
            "net_R": -1.0,
            "gross_R": 2.0,
            "zero_fee_net_R": -0.5,
            "no_funding_net_R": -0.25,
            "gross_R_per_1000_events": 0.1,
            "events_per_symbol_month": 10,
            "active_months": 12,
            "side": "long",
        }
        strong = {
            "net_R": 10.0,
            "gross_R": 50.0,
            "zero_fee_net_R": 20.0,
            "no_funding_net_R": 20.0,
            "gross_R_per_1000_events": 2.0,
            "events_per_symbol_month": 20,
            "active_months": 12,
            "side": "long",
        }
        self.assertNotEqual(sweep.classify_standard_v4_definition(weak), "aggregate_nonfutile_materialization_candidate")
        self.assertEqual(sweep.classify_standard_v4_definition(strong), "aggregate_nonfutile_materialization_candidate")

    def test_uncapped_two_family_profile_stage_list_has_no_wall_clock_gate(self):
        args = sweep.parse_args(["--phase-profile", sweep.UNCAPPED_TWO_FAMILY_PHASE_PROFILE, "--stage", "all"])
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.UNCAPPED_TWO_FAMILY_STAGES))
        self.assertNotIn("runtime-estimate", stages)
        self.assertNotIn("standard-v4-aggregate-only-screen", stages)
        self.assertIn("tsmom-v4-v5-aggregate-screen", stages)
        self.assertIn("prior-high-reclaim-mechanical-canary", stages)

    def test_uncapped_two_family_profile_rejects_non_smoke_max_symbols(self):
        with tempfile.TemporaryDirectory() as td:
            args = sweep.parse_args([
                "--phase-profile", sweep.UNCAPPED_TWO_FAMILY_PHASE_PROFILE,
                "--stage", "preflight-and-source-freeze",
                "--run-root", td,
                "--max-symbols", "5",
                "--disable-telegram",
            ])
            with self.assertRaisesRegex(RuntimeError, "max-symbols is smoke-only"):
                sweep.init_context(args)

    def test_tsmom_v5_curated_manifest_contract(self):
        defs = sweep.build_tsmom_v5_curated_manifest()
        self.assertEqual(len(defs), 64)
        self.assertFalse(defs["parameter_vector_hash"].duplicated().any())
        self.assertTrue((pd.to_numeric(defs["lookback_days"], errors="coerce") >= 10).all())
        self.assertTrue((pd.to_numeric(defs["hold_bars"], errors="coerce") >= 288).all())
        self.assertFalse(defs["bar_timeframe"].astype(str).str.lower().eq("5m").any())
        ranked = defs[defs["universe_policy"].astype(str).eq("pit_liquidity_tier_ab")]
        self.assertTrue((pd.to_numeric(ranked["rank_top_n"], errors="coerce") > 0).all())
        self.assertFalse(defs["entry_template"].astype(str).str.contains("generic_v0", case=False, regex=False).any())

    def test_prior_high_v1_manifests_are_curated_and_time_explicit(self):
        mech = sweep.build_prior_high_mechanical_manifest()
        full = sweep.build_prior_high_sweep_manifest()
        self.assertEqual(len(mech), 6)
        self.assertEqual(len(full), 64)
        self.assertFalse(full["bar_timeframe"].astype(str).str.lower().eq("5m").any())
        self.assertTrue(full["entry_template"].astype(str).str.contains("close_confirmed_next_bar").all())
        self.assertTrue(full["exit_template"].astype(str).str.contains("atr").all())
        self.assertTrue(full["exit_template"].astype(str).str.contains("structure").all())
        self.assertTrue(full["exit_template"].astype(str).str.contains("vwap").all())

    def test_prior_high_exit_binding_repair_profile_stage_list(self):
        args = sweep.parse_args(["--phase-profile", sweep.PRIOR_HIGH_EXIT_BINDING_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(tuple(sweep.active_stage_list(args)), sweep.PRIOR_HIGH_EXIT_BINDING_REPAIR_STAGES)
        self.assertIn("non-smoke-launch-readiness-check", sweep.active_stage_list(args))

    def test_prior_high_v2_manifest_is_bound_and_hash_clean(self):
        mech = sweep.build_prior_high_mechanical_manifest_v2()
        full = sweep.build_prior_high_sweep_manifest_v2()
        self.assertEqual(len(mech), 6)
        self.assertGreaterEqual(len(full), 36)
        self.assertLessEqual(len(full), 64)
        self.assertFalse(full["parameter_vector_hash"].duplicated().any())
        self.assertFalse(full["candidate_definition_id"].astype(str).str.contains("prior_high_v1_", regex=False).any())
        self.assertFalse(full["bar_timeframe"].astype(str).str.lower().eq("5m").any())
        self.assertTrue((pd.to_numeric(full["lookback_value"], errors="coerce") >= 20).all())
        self.assertTrue((pd.to_numeric(full["hold_bars"], errors="coerce") >= 288).all())
        self.assertTrue(full["entry_template"].astype(str).str.contains("close_confirmed_next_bar").all())
        self.assertTrue(full["same_bar_policy_hash"].astype(str).str.len().gt(0).all())
        self.assertTrue(full["stop_fill_policy_hash"].astype(str).str.len().gt(0).all())
        audit = sweep.prior_high_v2_manifest_audit(full, context="test")
        self.assertFalse(audit["status"].astype(str).eq("fail").any(), audit.to_string())

    def test_prior_high_atr_timeframe_resolution_fails_closed(self):
        daily = sweep.prior_high_v2_base_record(
            1, signal_type="prior_high_breakout", side="long", timeframe="daily", lookback_days=20,
            hold="1d", universe_policy="pit_liquidity_top_majors", parent_gate="", funding_gate="funding_aware_cap",
            atr_window_days=14, atr_stop_mult=0.75, atr_trail_mult=1.5, structure_buffer_atr=0.25,
            vwap_type="daily", vwap_anchor_policy="current_session", exit_module="atr_initial_stop",
        )
        self.assertEqual(sweep.prior_high_atr_resolution(daily)["atr_lookback_bars_resolved"], 14)
        four_h = sweep.prior_high_v2_base_record(
            2, signal_type="prior_high_breakout", side="long", timeframe="4h", lookback_days=20,
            hold="3d", universe_policy="pit_liquidity_tier_ab", parent_gate="", funding_gate="funding_aware_cap",
            atr_window_days=14, atr_stop_mult=1.5, atr_trail_mult=1.5, structure_buffer_atr=0.25,
            vwap_type="session", vwap_anchor_policy="utc_8h_session", exit_module="atr_initial_stop",
        )
        self.assertEqual(sweep.prior_high_atr_resolution(four_h)["atr_lookback_bars_resolved"], 84)
        daily_atr_for_4h = dict(four_h, atr_bar_timeframe="1d", atr_lookback_bars_resolved=14)
        self.assertEqual(sweep.prior_high_atr_resolution(daily_atr_for_4h)["atr_lookback_bars_resolved"], 14)
        missing_unit = dict(daily)
        missing_unit.pop("atr_window_unit", None)
        self.assertEqual(sweep.prior_high_atr_resolution(missing_unit)["status"], "fail")
        bad_5m = dict(daily, atr_bar_timeframe="5m")
        self.assertEqual(sweep.prior_high_atr_resolution(bad_5m)["status"], "fail")

    def test_prior_high_exit_module_fixtures_are_executable(self):
        fixtures = sweep.prior_high_exit_fixture_audit_frame()
        self.assertFalse(fixtures["status"].astype(str).eq("fail").any(), fixtures.to_string())
        expected = {"atr_initial_stop", "atr_trailing_stop", "structure_stop", "vwap_entry_filter", "vwap_exit", "time_exit"}
        self.assertEqual(set(fixtures["fixture_module"]), expected)
        self.assertFalse(fixtures["metadata_only"].astype(bool).any())

    def test_prior_high_side_semantics_long_flat_and_short_diagnostic(self):
        self.assertEqual(sweep.prior_high_side_direction("long_flat"), "long")
        self.assertEqual(sweep.prior_high_side_direction("short_diagnostic"), "short")
        cand = sweep.build_prior_high_mechanical_manifest_v2().iloc[1].to_dict()
        cand["lookback_bars"] = 50
        bars = self.bars(2000)
        masks = sweep.ENGINES["prior_high_reclaim_engine"].build_candidate_masks(bars, cand)
        self.assertIn("entry_mask", masks)

    def test_prior_high_stale_v1_manifest_rejected_for_explicit_launch_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            v1 = root / "prior_high_reclaim_sweep_definitions_v1.csv"
            sweep.write_csv(v1, sweep.build_prior_high_sweep_manifest())
            args = sweep.parse_args([
                "--phase-profile", sweep.UNCAPPED_TWO_FAMILY_PHASE_PROFILE,
                "--stage", "prior-high-reclaim-v1-aggregate-screen",
                "--run-root", str(root / "run"),
                "--prior-high-definition-manifest", str(v1),
                "--disable-telegram",
            ])
            ctx = SimpleNamespace(run_root=root / "run", args=args)
            with self.assertRaisesRegex(RuntimeError, "stale v1/generic"):
                sweep.load_prior_high_v2_sweep_manifest(ctx)

    def test_prior_high_daily_cadence_does_not_fire_every_5m(self):
        bars = self.bars(2000)
        cand = sweep.build_prior_high_mechanical_manifest_v2().iloc[0].to_dict()
        cand["lookback_bars"] = 50
        cand["hold_bars"] = 288
        engine = sweep.ENGINES["prior_high_reclaim_engine"]
        addrs = engine.enumerate_valid_event_addresses(bars, cand)
        if len(addrs) >= 2:
            spacings = [
                int((pd.Timestamp(addrs[i].decision_ts) - pd.Timestamp(addrs[i - 1].decision_ts)).total_seconds() // 60)
                for i in range(1, len(addrs))
            ]
            self.assertGreaterEqual(min(spacings), 24 * 60)

    def test_prior_high_mechanical_failure_blocks_aggregate_stage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "prior_high/mechanical_canary").mkdir(parents=True)
            (root / "prior_high/redesign").mkdir(parents=True)
            sweep.write_json(root / "prior_high/mechanical_canary/prior_high_reclaim_mechanical_gate_decision.json", {"status": "fail"})
            sweep.write_csv(root / "prior_high/redesign/prior_high_reclaim_sweep_definitions_v1.csv", sweep.build_prior_high_sweep_manifest())
            ctx = SimpleNamespace(run_root=root)
            sweep.stage_prior_high_reclaim_v1_aggregate_screen(ctx)
            decision = sweep.read_json(root / "prior_high/decision.json", {})
            self.assertEqual(decision["next_operator_decision"], "repair_prior_high_component_next")
            cls = pd.read_csv(root / "prior_high/ultra_definition_classification.csv")
            self.assertEqual(cls.iloc[0]["classification"], "pending_compute_interrupted")

    def test_uncapped_two_family_aggregate_only_labels_are_not_materialized_evidence(self):
        row = {
            "net_R": 10.0,
            "gross_R": 50.0,
            "zero_fee_net_R": 20.0,
            "no_funding_net_R": 20.0,
            "gross_R_per_1000_events": 2.0,
            "events_per_symbol_month": 20,
            "active_months": 12,
            "side": "long",
        }
        self.assertEqual(sweep.classify_two_family_aggregate_row(row), "aggregate_nonfutile_materialization_candidate")
        screen = pd.DataFrame([{**row, "candidate_definition_id": "d"}])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(run_root=root)
            sweep.stage_standard_v4_cost_turnover_funding_triage_to_path(ctx, screen, root / "summary.csv")
            out = pd.read_csv(root / "summary.csv")
            self.assertEqual(out.iloc[0]["aggregate_only_evidence_label"], "aggregate_only_diagnostic_not_strategy_evidence")

    def test_repaired_two_family_profile_stage_list_includes_pre_restart_audit(self):
        args = sweep.parse_args(["--phase-profile", sweep.REPAIRED_TWO_FAMILY_PHASE_PROFILE, "--stage", "all"])
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.REPAIRED_TWO_FAMILY_STAGES))
        self.assertIn("global-pre-restart-audit", stages)
        self.assertIn("tsmom-v4-v5-aggregate-screen", stages)
        self.assertNotIn("runtime-estimate", stages)

    def test_repaired_tsmom_top_n_filter_is_event_time_not_static_pruning(self):
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "rank_audit.csv"
            candidates = pd.DataFrame([
                {
                    "candidate_definition_id": "d",
                    "parameter_vector_hash": "p",
                    "symbol": sym,
                    "rank_top_n": 2,
                    "rank_metric": "trailing_return",
                }
                for sym in ["PF_AUSD", "PF_BUSD", "PF_CUSD", "PF_DUSD"]
            ])
            out = sweep.apply_tsmom_rank_top_n_filter(candidates, pd.DataFrame(), audit_path)
            self.assertEqual(len(out), len(candidates))
            audit = pd.read_csv(audit_path)
            self.assertEqual(audit.iloc[0]["status"], "pass_event_time_per_decision_filter")
            self.assertFalse(bool(audit.iloc[0]["static_proxy_filter_used"]))

    def test_repaired_tsmom_top_n_changes_by_decision_timestamp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trade_root = root / "parquet/historical_trade_candles_5m"
            panel_path = root / "panel.csv"
            symbols = ["PF_AUSD", "PF_BUSD", "PF_CUSD"]
            ts = pd.date_range("2025-01-01T00:00:00Z", periods=4 * 24 * 12, freq="5min")
            for sym in symbols:
                sym_dir = trade_root / sym
                sym_dir.mkdir(parents=True, exist_ok=True)
                base = np.ones(len(ts)) * 100.0
                if sym == "PF_AUSD":
                    base[: 2 * 24 * 12] = np.linspace(100, 130, 2 * 24 * 12)
                    base[2 * 24 * 12 :] = np.linspace(100, 80, len(ts) - 2 * 24 * 12)
                elif sym == "PF_BUSD":
                    base[: 2 * 24 * 12] = np.linspace(100, 90, 2 * 24 * 12)
                    base[2 * 24 * 12 :] = np.linspace(100, 140, len(ts) - 2 * 24 * 12)
                else:
                    base = np.linspace(100, 101, len(ts))
                bars = pd.DataFrame({"time": ts, "open": base, "high": base + 1, "low": base - 1, "close": base, "volume": 1000.0})
                bars.to_parquet(sym_dir / "bars_20250101T000000.parquet", index=False)
            panel = pd.DataFrame({
                "symbol": symbols,
                "start_ts": ["2025-01-01T00:00:00Z"] * 3,
                "end_ts": ["2025-12-31T00:00:00Z"] * 3,
                "status": ["available"] * 3,
                "bar_rows": [100, 100, 100],
            })
            sweep.write_csv(panel_path, panel)
            candidate = {
                "candidate_definition_id": "tsmom_ranked",
                "family_engine_id": "scheduled_tsmom_engine",
                "universe_policy": "pit_liquidity_tier_ab",
                "tier_ab_target_size": 3,
                "rank_top_n": 1,
                "rank_metric": "trailing_return",
                "lookback_days": 1,
                "parameter_vector_hash": "ranked_hash",
                "universe_policy_hash": "policy_hash",
                "pit_panel_manifest_path": str(panel_path),
                "kraken_data_root": str(root),
                "run_start_ts": "2025-01-01T00:00:00Z",
                "run_end_ts": "2025-01-05T00:00:00Z",
            }
            sweep._PIT_POLICY_SYMBOLS_CACHE.clear()
            sweep._RANK_TOP_N_DECISION_CACHE.clear()
            sweep._RANK_SYMBOL_BARS_CACHE.clear()
            first = sweep.rank_top_n_symbols_for_decision(candidate, pd.Timestamp("2025-01-02T12:00:00Z"))
            second = sweep.rank_top_n_symbols_for_decision(candidate, pd.Timestamp("2025-01-04T00:00:00Z"))
            self.assertEqual(first, ("PF_AUSD",))
            self.assertEqual(second, ("PF_BUSD",))

    def test_repaired_prior_high_registry_uses_pit_policy_not_generic_fanout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = sweep.parse_args([
                "--phase-profile", sweep.REPAIRED_TWO_FAMILY_PHASE_PROFILE,
                "--stage", "all",
                "--run-root", str(root),
                "--kraken-data-root", str(root / "data"),
                "--disable-telegram",
            ])
            ctx = SimpleNamespace(args=args, run_root=root, start=pd.Timestamp("2025-01-01T00:00:00Z"), end=pd.Timestamp("2025-02-01T00:00:00Z"))
            panel = pd.DataFrame({
                "symbol": ["PF_XBTUSD", "PF_ETHUSD", "PF_AUSD", "PF_BUSD", "PF_CUSD"],
                "start_ts": ["2024-01-01T00:00:00Z"] * 5,
                "end_ts": ["2025-12-31T00:00:00Z"] * 5,
                "status": ["available"] * 5,
                "bar_rows": [100, 100, 100, 100, 100],
            })
            (root / "panels").mkdir(parents=True, exist_ok=True)
            sweep.write_csv(root / "panels/aggregate_panel_manifest.csv", panel)
            row = sweep.prior_high_v2_base_record(
                1,
                signal_type="prior_high_breakout",
                side="long",
                timeframe="daily",
                lookback_days=20,
                hold="1d",
                universe_policy="pit_liquidity_tier_ab",
                parent_gate="btc_eth_trend_up_or_neutral",
                funding_gate="funding_aware_cap",
                atr_window_days=14,
                atr_stop_mult=0.75,
                atr_trail_mult=1.5,
                structure_buffer_atr=0.25,
                vwap_type="daily",
                vwap_anchor_policy="current_session",
                exit_module="atr_initial_stop",
                version_prefix="prior_high_v3",
                universe_policy_version="prior_high_pit_liquidity_v3_repaired_20260705",
            )
            row["tier_ab_target_size"] = 2
            registry = sweep.generate_candidate_registry_for_family_universe(pd.DataFrame([row]), {"prior_high_reclaim_engine": panel["symbol"].tolist()}, 0, panel=panel, ctx=ctx)
            self.assertEqual(registry["symbol"].nunique(), 2)
            self.assertLess(registry["symbol"].nunique(), len(panel))
            self.assertTrue(registry["pit_universe_event_time_check"].astype(bool).all())

    def test_repaired_two_family_v6_and_prior_high_v3_manifest_contracts(self):
        v6 = sweep.build_tsmom_v6_curated_manifest()
        self.assertGreaterEqual(len(v6), 96)
        self.assertLessEqual(len(v6), 128)
        self.assertFalse(v6["parameter_vector_hash"].duplicated().any())
        ranked = v6[v6["universe_policy"].astype(str).eq("pit_liquidity_tier_ab")]
        self.assertFalse(ranked.empty)
        self.assertTrue((pd.to_numeric(ranked["rank_top_n"], errors="coerce") > 0).all())
        self.assertFalse(v6["bar_timeframe"].astype(str).str.lower().eq("5m").any())
        v3 = sweep.build_prior_high_sweep_manifest_v3()
        self.assertGreaterEqual(len(v3), 72)
        self.assertLessEqual(len(v3), 96)
        self.assertFalse(v3["parameter_vector_hash"].duplicated().any())
        self.assertTrue(v3["candidate_definition_id"].astype(str).str.startswith("prior_high_v3_").all())
        self.assertFalse(v3["candidate_definition_id"].astype(str).str.contains("prior_high_v1_", regex=False).any())

    def test_memory_throughput_profiles_registered(self):
        self.assertIn(sweep.MEMORY_THROUGHPUT_REPAIR_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertIn(sweep.MEMORYSAFE_TWO_FAMILY_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertIn("previous-oom-interruption-ingest", sweep.MEMORY_THROUGHPUT_REPAIR_STAGES)
        self.assertIn("memory-safe-aggregate-repair-audit", sweep.MEMORY_THROUGHPUT_REPAIR_STAGES)
        self.assertTrue(sweep.is_memorysafe_two_family_profile(sweep.MEMORYSAFE_TWO_FAMILY_PHASE_PROFILE))

    def test_cache_lifecycle_clears_rank_and_pit_caches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(run_root=root)
            sweep._RANK_SYMBOL_BARS_CACHE[("r", "s", "a", "b")] = pd.DataFrame({"x": [1]})
            sweep._RANK_TOP_N_DECISION_CACHE[("r", "u", "p", "d", "m", 5, 20, "h")] = ("PF_XBTUSD",)
            sweep._PIT_POLICY_SYMBOLS_CACHE[("r", "u", "a", "b", "d", "h")] = ("PF_XBTUSD",)
            sweep._PANEL_MANIFEST_CACHE["keep"] = pd.DataFrame({"symbol": ["PF_XBTUSD"]})
            sweep.clear_aggregate_caches(ctx, reason="unit_test", scope="batch")
            self.assertFalse(sweep._RANK_SYMBOL_BARS_CACHE)
            self.assertFalse(sweep._RANK_TOP_N_DECISION_CACHE)
            self.assertFalse(sweep._PIT_POLICY_SYMBOLS_CACHE)
            self.assertIn("keep", sweep._PANEL_MANIFEST_CACHE)
            audit = pd.read_csv(root / "performance/cache_lifecycle_audit.csv")
            self.assertEqual(audit.iloc[-1]["reason"], "unit_test")

    def test_memory_guard_clean_stop_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notifier = SimpleNamespace(send=lambda *args, **kwargs: None)
            args = SimpleNamespace(memory_soft_limit_gb=0.000001)
            ctx = SimpleNamespace(run_root=root, args=args, notifier=notifier)
            pending = pd.DataFrame([{"candidate_definition_id": "d1", "parameter_vector_hash": "p1"}])
            with self.assertRaises(sweep.MemoryGuardStop):
                sweep.memory_guard_check(ctx, stage="unit_memory_guard", processed=0, total=1, pending_candidates=pending)
            decision = json.loads((root / "decision_summary.json").read_text())
            self.assertEqual(decision["status"], "interrupted_resource_guard")
            self.assertTrue((root / "interruptions/memory_guard_stop_report.md").exists())
            interrupted = pd.read_csv(root / "interruptions/pending_compute_interrupted.csv")
            self.assertEqual(interrupted.iloc[0]["candidate_status"], "pending_compute_interrupted")

    def test_memorysafe_aggregate_shards_match_legacy_candidate_totals(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / "data"
            bars = self.bars(420).rename(columns={"ts": "time"})
            sym_dir = data_root / "parquet/historical_trade_candles_5m/PF_XBTUSD"
            sym_dir.mkdir(parents=True)
            bars.to_parquet(sym_dir / "PF_XBTUSD_20250101T000000.parquet", index=False)
            base = self.candidate("liquid_continuation_breakout_engine", "liquid_continuation")
            rows = []
            for i, threshold in enumerate([0.001, 0.002, 0.003]):
                rec = dict(base)
                rec["definition_id"] = f"d_mem_{i}"
                rec["threshold"] = threshold
                rec["candidate_definition_id"] = f"d_mem_{i}"
                rec = sweep.identity_enriched_record(rec, sweep.ENGINES["liquid_continuation_breakout_engine"], symbol="PF_XBTUSD")
                rows.append(rec)
            candidates = pd.DataFrame(rows)
            notifier = SimpleNamespace(send=lambda *args, **kwargs: None)
            common_args = {
                "kraken_data_root": str(data_root),
                "smoke": True,
                "chunk_size": 10,
                "aggregate_batch_size": 2,
                "memory_soft_limit_gb": 1000.0,
            }
            legacy_ctx = SimpleNamespace(
                run_root=root / "legacy",
                args=SimpleNamespace(**common_args, phase_profile=sweep.REPAIRED_TWO_FAMILY_PHASE_PROFILE),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-01-02 12:00", tz="UTC"),
                notifier=notifier,
            )
            memory_ctx = SimpleNamespace(
                run_root=root / "memorysafe",
                args=SimpleNamespace(**common_args, phase_profile=sweep.MEMORYSAFE_TWO_FAMILY_PHASE_PROFILE),
                start=legacy_ctx.start,
                end=legacy_ctx.end,
                notifier=notifier,
            )
            for p in [legacy_ctx.run_root, memory_ctx.run_root]:
                for rel in ["coarse", "audit", "performance", "resources", "interruptions"]:
                    (p / rel).mkdir(parents=True, exist_ok=True)
            legacy_agg, _, _ = sweep.aggregate_candidates_to_dir(legacy_ctx, candidates, legacy_ctx.run_root / "coarse")
            mem_agg, _, _ = sweep.aggregate_candidates_to_dir(memory_ctx, candidates, memory_ctx.run_root / "coarse")
            legacy = legacy_agg.sort_values("candidate_definition_id").reset_index(drop=True)
            mem = mem_agg.sort_values("candidate_definition_id").reset_index(drop=True)
            self.assertEqual(legacy["events"].astype(int).tolist(), mem["events"].astype(int).tolist())
            np.testing.assert_allclose(legacy["net_R"].astype(float), mem["net_R"].astype(float), rtol=1e-12, atol=1e-12)
            self.assertTrue((memory_ctx.run_root / "coarse/aggregate_shards/candidate_aggregate").exists())
            self.assertTrue((memory_ctx.run_root / "resources/memory_usage_timeseries.csv").exists())

    def test_oom_interrupted_root_is_not_resumable_evidence(self):
        root = sweep.oom_interrupted_two_family_root()
        self.assertIn("phase_kraken_uncapped_tier1_two_family_sweep_repaired_20260705_v2", root.name)
        self.assertFalse(bool(sweep.read_json(root / "decision_summary.json", {}).get("status") == "complete"))

    def test_cache_acceleration_profile_matches_optimisation_plan_stages(self):
        self.assertIn(sweep.CACHE_ACCELERATION_FOUNDATION_PHASE_PROFILE, sweep.PHASE_PROFILES)
        expected = [
            "preflight-and-active-run-snapshot",
            "cache-contracts-and-leak-boundary",
            "candidate-registry-and-shard-manifest",
            "heartbeat-and-hotspot-timers",
            "decision-calendar-cache",
            "universe-membership-cache",
            "tsmom-decision-input-caches",
            "tsmom-interval-outcome-cache",
            "prior-high-feature-and-outcome-cache-contract",
            "cache-vs-scalar-exactness-benchmark",
            "cache-performance-benchmark",
            "parallelism-readiness-benchmark",
            "accelerated-two-family-launch-readiness",
            "decision-report",
            "compact-review-bundle",
        ]
        self.assertEqual(list(sweep.CACHE_ACCELERATION_FOUNDATION_STAGES), expected)
        self.assertTrue(sweep.optimisation_plan_path().exists())
        text = sweep.read_optimisation_plan_text()
        for stage in expected[:-2]:
            self.assertIn(stage, text)

    def test_cache_layer_freeze_and_leak_guards(self):
        keys = pd.DataFrame([
            {"candidate_identity_hash": "c1", "symbol_id": "PF_XBTUSD", "decision_ts": "2025-01-01T00:00:00Z"},
            {"candidate_identity_hash": "c1", "symbol_id": "PF_ETHUSD", "decision_ts": "2025-01-01T00:00:00Z"},
        ])
        frozen, key_hash = cache_layer.freeze_selected_event_keys(keys)
        self.assertTrue(frozen["selected_event_keys_frozen"].all())
        self.assertEqual(frozen["selected_event_keys_hash"].nunique(), 1)
        self.assertTrue(key_hash)
        decision_inputs = pd.DataFrame([
            {"decision_ts": "2025-01-02T00:00:00Z", "feature_available_ts": "2025-01-01T23:55:00Z"},
            {"decision_ts": "2025-01-02T00:00:00Z", "feature_available_ts": "2025-01-02T00:00:00Z"},
        ])
        audit = cache_layer.assert_decision_input_no_leak(decision_inputs)
        self.assertFalse(audit["status"].astype(str).eq("fail").any())
        bad_inputs = pd.DataFrame([{"decision_ts": "2025-01-02T00:00:00Z", "feature_available_ts": "2025-01-02T00:05:00Z"}])
        bad_audit = cache_layer.assert_decision_input_no_leak(bad_inputs)
        self.assertTrue(bad_audit["status"].astype(str).eq("fail").any())
        access = pd.DataFrame([{"cache_class": "outcome", "access_phase": "selection", "selected_event_keys_frozen": False}])
        leak = cache_layer.assert_outcome_cache_not_used_pre_freeze(access)
        self.assertEqual(leak.iloc[0]["status"], "fail")

    def test_cache_shard_manifest_hash_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            df = pd.DataFrame([
                {"candidate_identity_hash": "c2", "symbol_id": "PF_ETHUSD", "decision_ts": "2025-01-01T00:00:00Z", "v": 2},
                {"candidate_identity_hash": "c1", "symbol_id": "PF_XBTUSD", "decision_ts": "2025-01-01T00:00:00Z", "v": 1},
            ])
            manifest = cache_layer.write_atomic_shard(
                root / "tmp/shard.parquet",
                root / "final/shard.parquet",
                df,
                {"code_hash": "code", "config_hash": "cfg"},
                sort_keys=["candidate_identity_hash", "symbol_id", "decision_ts"],
            )
            self.assertEqual(manifest["hash_basis"], "canonical_sorted_csv_rows")
            validation = cache_layer.validate_shard_manifest(root / "final/shard.parquet.manifest.json", expected={"code_hash": "code"})
            self.assertEqual(validation["status"], "pass")
            bad = cache_layer.validate_shard_manifest(root / "final/shard.parquet.manifest.json", expected={"code_hash": "other"})
            self.assertEqual(bad["status"], "fail")

    def test_cache_shard_manifest_validation_uses_recorded_sort_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            df = pd.DataFrame([
                {"candidate_definition_id": "d1", "candidate_symbol_id": "d1__PF_B", "symbol": "PF_B", "parameter_vector_hash": "p"},
                {"candidate_definition_id": "d1", "candidate_symbol_id": "d1__PF_A", "symbol": "PF_A", "parameter_vector_hash": "p"},
            ])
            manifest = cache_layer.write_atomic_shard(
                root / "tmp/registry.parquet",
                root / "registry/registry.parquet",
                df,
                {"code_hash": "code", "config_hash": "cfg", "input_manifest_hash": "input"},
                sort_keys=["candidate_definition_id", "candidate_symbol_id", "symbol", "parameter_vector_hash"],
            )
            expected_keys = ["candidate_definition_id", "candidate_symbol_id", "symbol", "parameter_vector_hash"]
            self.assertEqual(manifest["canonical_sort_keys"], expected_keys)
            validation = cache_layer.validate_shard_manifest(
                root / "registry/registry.parquet.manifest.json",
                expected={"code_hash": "code", "config_hash": "cfg", "input_manifest_hash": "input"},
            )
            self.assertEqual(validation["status"], "pass")

    def test_tsmom_semantic_cache_contract_repair_profile_registered(self):
        self.assertIn(sweep.TSMOM_SEMANTIC_CACHE_CONTRACT_REPAIR_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_SEMANTIC_CACHE_CONTRACT_REPAIR_STAGES),
            [
                "preflight-and-source-freeze",
                "tsmom-semantic-cache-contract-repair",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        profile = sweep.PHASE_PROFILES[sweep.TSMOM_SEMANTIC_CACHE_CONTRACT_REPAIR_PHASE_PROFILE]
        self.assertIn("TSMOM-only", profile["description"])

    def test_semantic_cache_hash_ignores_harmless_wrapper_change(self):
        rows = {r["case"]: r for r in sweep.semantic_cache_reuse_matrix_rows()}
        harmless = rows["harmless_wrapper_reporting_profile_code_hash_change"]
        self.assertEqual(harmless["status"], "pass")
        self.assertTrue(harmless["semantic_hash_equal"])
        self.assertTrue(harmless["expected_reuse_allowed"])

    def test_semantic_cache_hash_invalidates_rank_semantic_changes(self):
        rows = {r["case"]: r for r in sweep.semantic_cache_reuse_matrix_rows()}
        for case in [
            "rank_lookback_changed",
            "rank_metric_changed",
            "rank_computation_version_changed",
            "schema_version_changed",
            "data_manifest_hash_changed",
            "protected_boundary_changed",
        ]:
            self.assertEqual(rows[case]["status"], "pass", case)
            self.assertFalse(rows[case]["semantic_hash_equal"], case)
            self.assertFalse(rows[case]["expected_reuse_allowed"], case)

    def test_outcome_semantic_manifest_requires_selected_event_key_hash(self):
        df = pd.DataFrame([{
            "candidate_definition_id": "d1",
            "candidate_symbol_id": "d1__PF_XBTUSD",
            "symbol": "PF_XBTUSD",
            "decision_ts": "2025-01-01T00:00:00Z",
            "raw_net_R": 0.1,
        }])
        with self.assertRaises(ValueError):
            cache_layer.semantic_cache_manifest(
                cache_class="tsmom_interval_outcome",
                df=df,
                sort_keys=["candidate_definition_id", "candidate_symbol_id", "symbol", "decision_ts"],
                policy_fields={"interval_policy_hash": "i"},
                input_manifest_hash="input",
                protected_train_boundary="2026-01-01T00:00:00Z",
            )

    def test_decision_input_semantic_manifest_rejects_outcome_fields(self):
        df = pd.DataFrame([{
            "decision_ts": "2025-01-01T00:00:00Z",
            "feature_available_ts": "2024-12-31T23:55:00Z",
            "raw_net_R": 0.1,
        }])
        with self.assertRaises(ValueError):
            cache_layer.semantic_cache_manifest(
                cache_class="tsmom_rank_feature",
                df=df,
                sort_keys=["decision_ts"],
                policy_fields={"rank_metric": "trailing_return"},
                input_manifest_hash="input",
                protected_train_boundary="2026-01-01T00:00:00Z",
            )

    def test_semantic_cache_manifest_deterministic_under_repeated_write_read(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            df = pd.DataFrame([
                {"ts": "2025-01-01T00:05:00Z", "close": 101.0},
                {"ts": "2025-01-01T00:00:00Z", "close": 100.0},
            ])
            kwargs = {
                "cache_class": "tsmom_rank_feature",
                "policy_fields": {"symbol": "PF_XBTUSD", "lookback_days": 20, "rank_metric": "trailing_return"},
                "input_manifest_hash": "input",
                "data_hash": "data",
                "protected_train_boundary": "2026-01-01T00:00:00Z",
                "sort_keys": ["ts"],
            }
            first = cache_layer.write_semantic_cache_shard(root / "tmp/a.parquet", root / "cache/a.parquet", df, **kwargs)
            second = cache_layer.write_semantic_cache_shard(root / "tmp/b.parquet", root / "cache/b.parquet", df, **kwargs)
            for key in ["semantic_cache_hash", "schema_hash", "content_hash", "row_count"]:
                self.assertEqual(first[key], second[key])
            loaded = json.loads((root / "cache/a.parquet.manifest.json").read_text())
            validation = cache_layer.validate_semantic_cache_manifest(loaded, expected={"semantic_cache_hash": first["semantic_cache_hash"]})
            self.assertEqual(validation["status"], "pass")

    def test_optimisation_completion_audit_contains_required_docs_items(self):
        reqs = sweep.cache_foundation_required_artifacts()
        paths = {r["required_artifact_or_behavior"] for r in reqs}
        self.assertIn("contracts/cache_class_contract.md", paths)
        self.assertIn("cache_manifests/tsmom_interval_return_manifest.csv", paths)
        self.assertIn("audit/optimisation_plan_completion_audit.csv", paths)
        self.assertIn("prelaunch/accelerated_two_family_launch_readiness.json", paths)

    def test_active_boundary_profile_registered(self):
        self.assertIn(sweep.ACTIVE_BOUNDARY_CACHE_SHADOW_PHASE_PROFILE, sweep.PHASE_PROFILES)
        expected = [
            "preflight-and-active-run-snapshot",
            "tsmom-aggregate-boundary-monitor",
            "active-run-boundary-stop",
            "active-tsmom-aggregate-harvest",
            "cache-shadow-candidate-registry-freeze",
            "cache-shadow-tsmom-evaluation",
            "cache-vs-active-scalar-exactness-audit",
            "cache-performance-audit",
            "prior-high-cache-deferred-routing-decision",
            "next-launch-recommendation",
            "decision-report",
            "compact-review-bundle",
        ]
        self.assertEqual(list(sweep.ACTIVE_BOUNDARY_CACHE_SHADOW_STAGES), expected)

    def test_boundary_stop_does_not_stop_before_tsmom_done(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = sweep.ACTIVE_BOUNDARY_REFERENCE_RUN_ROOT
            try:
                sweep.ACTIVE_BOUNDARY_REFERENCE_RUN_ROOT = root / "active"
                (sweep.ACTIVE_BOUNDARY_REFERENCE_RUN_ROOT / "stage_status").mkdir(parents=True)
                sweep.write_json(sweep.ACTIVE_BOUNDARY_REFERENCE_RUN_ROOT / "watch_status.json", {"stage": "tsmom-v4-v5-aggregate-screen", "status": "running"})
                ctx = SimpleNamespace(run_root=root / "phase", args=SimpleNamespace(smoke=False), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
                (ctx.run_root / "active").mkdir(parents=True)
                sweep.stage_active_run_boundary_stop(ctx)
                text = (ctx.run_root / "active/boundary_stop_report.md").read_text()
                self.assertIn("not_stopped_boundary_not_reached", text)
            finally:
                sweep.ACTIVE_BOUNDARY_REFERENCE_RUN_ROOT = old

    def test_mock_boundary_harvest_registry_shadow_exactness(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(
                run_root=root / "phase",
                args=SimpleNamespace(smoke=True),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-02-01", tz="UTC"),
            )
            for rel in ["active", "baseline", "shadow", "audit", "performance", "prior_high", "prelaunch", "stage_status", "tmp"]:
                (ctx.run_root / rel).mkdir(parents=True, exist_ok=True)
            sweep.create_mock_active_boundary_run(ctx)
            sweep.stage_active_run_boundary_stop(ctx)
            sweep.stage_active_tsmom_aggregate_harvest(ctx)
            sweep.stage_cache_shadow_candidate_registry_freeze(ctx)
            sweep.stage_cache_shadow_tsmom_evaluation(ctx)
            sweep.stage_cache_vs_active_scalar_exactness_audit(ctx)
            sweep.stage_prior_high_cache_deferred_routing_decision(ctx)
            exact = pd.read_csv(ctx.run_root / "audit/cache_vs_active_scalar_exactness.csv")
            self.assertFalse(exact["status"].astype(str).eq("fail").any())
            registry = pd.read_csv(ctx.run_root / "shadow/cache_shadow_registry_vs_scalar_baseline_audit.csv")
            self.assertEqual(registry.iloc[0]["status"], "pass")
            routing = (ctx.run_root / "prior_high/prior_high_cache_deferred_routing_report.md").read_text()
            self.assertIn("does not claim prior-high acceleration readiness", routing)

    def test_cache_vs_active_scalar_exactness_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(
                run_root=root / "phase",
                args=SimpleNamespace(smoke=True),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-02-01", tz="UTC"),
            )
            for rel in ["active", "baseline", "shadow", "audit", "tmp"]:
                (ctx.run_root / rel).mkdir(parents=True, exist_ok=True)
            sweep.create_mock_active_boundary_run(ctx)
            sweep.stage_active_tsmom_aggregate_harvest(ctx)
            sweep.stage_cache_shadow_candidate_registry_freeze(ctx)
            sweep.stage_cache_shadow_tsmom_evaluation(ctx)
            shadow_path = ctx.run_root / "shadow/cache_tsmom_aggregate_summary.parquet"
            shadow = pd.read_parquet(shadow_path)
            shadow.loc[0, "net_R"] = float(shadow.loc[0, "net_R"]) + 1.0
            shadow.to_parquet(shadow_path, index=False)
            sweep.stage_cache_vs_active_scalar_exactness_audit(ctx)
            exact = pd.read_csv(ctx.run_root / "audit/cache_vs_active_scalar_exactness.csv")
            failed = exact[exact["status"].astype(str).eq("fail")]
            self.assertIn("net_R", set(failed["check"].astype(str)))

    def test_v6_setup_streaming_repair_profile_registered_and_setup_only(self):
        self.assertIn(sweep.V6_SETUP_STREAMING_MEMORY_REPAIR_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.V6_SETUP_STREAMING_MEMORY_REPAIR_STAGES),
            ["preflight-and-source-freeze", "v6-setup-streaming-benchmark", "decision-report", "compact-review-bundle"],
        )
        source = inspect.getsource(sweep.stage_v6_setup_streaming_benchmark)
        self.assertNotIn("aggregate_candidates_to_dir(", source)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir(", source)
        self.assertIn("write_atomic_shard", source)
        self.assertIn("v6_candidate_registry_shards", source)

    def test_representative_v6_cache_exactness_profile_registered(self):
        self.assertIn(sweep.REPRESENTATIVE_V6_CACHE_EXACTNESS_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.REPRESENTATIVE_V6_CACHE_EXACTNESS_STAGES),
            [
                "preflight-and-source-freeze",
                "representative-v6-registry-freeze",
                "representative-v6-scalar-evaluation",
                "representative-v6-cache-evaluation",
                "representative-v6-exactness-and-performance-audit",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        source = inspect.getsource(sweep.cache_evaluate_representative_v6)
        self.assertIn("selected_event_keys_hash", source)
        self.assertIn("event_from_address", source)
        self.assertNotIn("active_scalar_tsmom_aggregate", source)

    def test_representative_v6_definition_sample_includes_ranked_and_liquid(self):
        defs = pd.DataFrame([
            {"candidate_definition_id": "liquid_long", "universe_policy": "pit_liquidity_top_majors", "side": "long", "rank_top_n": 0, "rank_metric": ""},
            {"candidate_definition_id": "liquid_flat", "universe_policy": "pit_liquidity_top_majors", "side": "long_flat", "rank_top_n": 0, "rank_metric": ""},
            {"candidate_definition_id": "rank3_trail", "universe_policy": "pit_liquidity_tier_ab", "side": "long", "rank_top_n": 3, "rank_metric": "trailing_return"},
            {"candidate_definition_id": "rank5_risk", "universe_policy": "pit_liquidity_tier_ab", "side": "long", "rank_top_n": 5, "rank_metric": "risk_adjusted_return"},
        ])
        sample = sweep.representative_v6_definition_ids(defs, max_defs=4)
        self.assertIn("liquid_long", sample)
        self.assertIn("liquid_flat", sample)
        self.assertIn("rank3_trail", sample)
        self.assertIn("rank5_risk", sample)

    def test_tsmom_v6_accelerated_profile_registered_and_tsmom_only(self):
        self.assertIn(sweep.TSMOM_V6_ACCELERATED_AGGREGATE_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_V6_ACCELERATED_AGGREGATE_STAGES),
            [
                "preflight-and-source-freeze",
                "tsmom-v6-accelerated-registry-freeze",
                "tsmom-v6-decision-input-cache-build",
                "tsmom-v6-selected-event-key-freeze",
                "tsmom-v6-interval-outcome-cache-build",
                "tsmom-v6-accelerated-aggregate-benchmark",
                "tsmom-v6-accelerated-gate-and-runtime-projection",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        profile = sweep.PHASE_PROFILES[sweep.TSMOM_V6_ACCELERATED_AGGREGATE_PHASE_PROFILE]
        self.assertIn("TSMOM-only", profile["description"])
        self.assertNotIn("prior-high", " ".join(sweep.TSMOM_V6_ACCELERATED_AGGREGATE_STAGES))

    def test_tsmom_v6_accelerated_path_forbids_memorysafe_scalar_fallback(self):
        source = inspect.getsource(sweep.accelerated_tsmom_v6_cache_aggregate)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir", source)
        self.assertIn("independent_accelerated_evaluation", source)
        self.assertIn("accelerated_tsmom_scalar_fallback_used", source)

    def test_tsmom_v6_topn_audit_detects_alphabetical_static_panel(self):
        symbols = ["PF_AAVEUSD", "PF_ADAUSD", "PF_AVAXUSD", "PF_XBTUSD"]
        self.assertTrue(sweep.universe_symbols_are_alphabetical_first_n(["PF_AAVEUSD", "PF_ADAUSD"], symbols))
        self.assertFalse(sweep.universe_symbols_are_alphabetical_first_n(["PF_XBTUSD", "PF_ADAUSD"], symbols))

    def test_tsmom_v6_cache_performance_repair_profile_registered(self):
        self.assertIn(sweep.TSMOM_V6_CACHE_PERFORMANCE_REPAIR_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_V6_CACHE_PERFORMANCE_REPAIR_STAGES),
            [
                "preflight-and-source-freeze",
                "tsmom-v6-cache-bottleneck-ingest",
                "tsmom-v6-cache-reuse-repair-audit",
                "tsmom-v6-representative-cache-performance-benchmark",
                "tsmom-v6-medium-cache-scaling-benchmark",
                "tsmom-v6-warm-cache-scaling-benchmark",
                "tsmom-v6-cache-performance-gate",
                "decision-report",
                "compact-review-bundle",
            ],
        )

    def test_tsmom_v6_reused_cache_builder_has_no_scalar_fallback(self):
        source = inspect.getsource(sweep.build_tsmom_v6_decision_input_caches_reused)
        self.assertIn("rank_panel_cache", source)
        self.assertIn("topn_cache", source)
        self.assertIn("parent_cache", source)
        self.assertIn("funding_cache", source)
        self.assertIn("vol_cache", source)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir", source)

    def test_tsmom_v6_accelerated_full_aggregate_profile_registered(self):
        self.assertIn(sweep.TSMOM_V6_ACCELERATED_FULL_AGGREGATE_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_V6_ACCELERATED_FULL_AGGREGATE_STAGES),
            [
                "preflight-and-source-freeze",
                "full-tsmom-v6-registry-freeze",
                "full-tsmom-v6-cache-validation",
                "full-tsmom-v6-accelerated-aggregate-run",
                "full-tsmom-v6-exactness-sentinel",
                "full-tsmom-v6-gate-and-decision",
                "decision-report",
                "compact-review-bundle",
            ],
        )

    def test_tsmom_v6_accelerated_full_run_has_no_scalar_fallback(self):
        source = inspect.getsource(sweep.stage_full_tsmom_v6_accelerated_aggregate_run)
        self.assertIn("run_scalar=False", source)
        self.assertIn("run_tsmom_v6_reused_cache_benchmark", source)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir", source)

    def test_rank_symbol_bars_uses_close_only_window_loader(self):
        source = inspect.getsource(sweep.rank_symbol_bars_for_candidate)
        self.assertIn("load_symbol_rank_close_window", source)
        self.assertNotIn("load_symbol_bars(paths", source)

    def test_cache_empty_event_summary_matches_scalar_zero_event_summary(self):
        cand = {
            "candidate_id": "c",
            "candidate_definition_id": "c",
            "candidate_symbol_id": "c__PF_XBTUSD",
            "definition_id": "d",
            "family_engine_id": "scheduled_tsmom_engine",
            "family": "TSMOM",
            "symbol": "PF_XBTUSD",
            "parameter_vector_hash": "p",
        }
        scalar = sweep.StreamingAggregate(cand, sweep.ENGINES["scheduled_tsmom_engine"]).as_candidate_summary()
        cached = sweep.aggregate_events_from_cache(cand, pd.DataFrame())
        for key in ["events", "data_cap", "funding_cap", "mark_cap", "candidate_status"]:
            self.assertEqual(cached[key], scalar[key])

    def test_v6_setup_missing_shard_manifest_fails_resume_validation(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.parquet.manifest.json"
            result = sweep.validate_streaming_setup_shard_manifest(
                missing,
                expected_code_hash="code",
                expected_config_hash="config",
                expected_input_manifest_hash="input",
            )
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["reason"], "missing_manifest")

    def test_v6_setup_stale_code_config_source_hash_rejects_shard_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            df = pd.DataFrame([
                {"candidate_definition_id": "d1", "candidate_symbol_id": "d1__PF_XBTUSD", "symbol": "PF_XBTUSD", "parameter_vector_hash": "p1"},
            ])
            cache_layer.write_atomic_shard(
                root / "tmp/shard.parquet",
                root / "registry/shard.parquet",
                df,
                {"code_hash": "code_a", "config_hash": "config_a", "input_manifest_hash": "input_a"},
                sort_keys=["candidate_definition_id", "candidate_symbol_id", "symbol", "parameter_vector_hash"],
            )
            ok = sweep.validate_streaming_setup_shard_manifest(
                root / "registry/shard.parquet.manifest.json",
                expected_code_hash="code_a",
                expected_config_hash="config_a",
                expected_input_manifest_hash="input_a",
            )
            self.assertEqual(ok["status"], "pass")
            stale = sweep.validate_streaming_setup_shard_manifest(
                root / "registry/shard.parquet.manifest.json",
                expected_code_hash="code_b",
                expected_config_hash="config_a",
                expected_input_manifest_hash="input_a",
            )
            self.assertEqual(stale["status"], "fail")
            self.assertIn("expected_code_hash_mismatch", stale["reason"])

    def test_tsmom_calendar_universe_cache_builder_profile_registered(self):
        self.assertIn(sweep.TSMOM_CALENDAR_UNIVERSE_CACHE_BUILDER_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_CALENDAR_UNIVERSE_CACHE_BUILDER_STAGES),
            [
                "preflight-and-source-freeze",
                "tsmom-calendar-universe-cache-builder",
                "decision-report",
                "compact-review-bundle",
            ],
        )

    def test_tsmom_calendar_specs_are_unique_by_semantics_not_candidate_rows(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = sweep.Context(
                args=SimpleNamespace(
                    kraken_data_root="/tmp/kraken",
                    k0_root="k0",
                    structural_repair_root="struct",
                    stat_protocol_repair_root="stat",
                    p1_repair_root="p1",
                ),
                run_root=Path(td),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-02-01", tz="UTC"),
                notifier=SimpleNamespace(status={}),
            )
            rows = pd.DataFrame([
                {"candidate_definition_id": "d1", "symbol": "PF_XBTUSD", "bar_timeframe": "5m", "rebalance_interval": "daily", "lookback_days": 20, "lookback_bars": 5760, "hold_interval": "1d", "hold_bars": 288},
                {"candidate_definition_id": "d1", "symbol": "PF_ETHUSD", "bar_timeframe": "5m", "rebalance_interval": "daily", "lookback_days": 20, "lookback_bars": 5760, "hold_interval": "1d", "hold_bars": 288},
                {"candidate_definition_id": "d2", "symbol": "PF_XBTUSD", "bar_timeframe": "5m", "rebalance_interval": "daily", "lookback_days": 40, "lookback_bars": 11520, "hold_interval": "3d", "hold_bars": 864},
            ])
            specs = sweep.tsmom_decision_calendar_specs_from_registry(rows, ctx)
            self.assertEqual(len(specs), 2)
            self.assertEqual(int(specs["candidate_symbol_row_count"].sum()), 3)

    def test_tsmom_calendar_rows_respect_protected_boundary(self):
        bars = pd.DataFrame({
            "ts": pd.date_range("2025-12-31 22:00:00", periods=80, freq="5min", tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
        })
        spec = {
            "calendar_hash": "cal",
            "bar_timeframe": "5m",
            "rebalance_interval": "1h",
            "lookback_bars": 2,
            "hold_bars": 2,
            "interval_bars": 1,
        }
        rows = sweep.tsmom_calendar_rows_for_symbol_spec(bars, spec, "PF_XBTUSD")
        self.assertTrue(rows)
        self.assertTrue(all(pd.Timestamp(r["decision_ts"]) < sweep.PROTECTED_TS for r in rows))

    def test_calendar_universe_builder_does_not_call_aggregate_or_materialization(self):
        source = inspect.getsource(sweep.stage_tsmom_calendar_universe_cache_builder)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir", source)
        self.assertNotIn("stage_full_tsmom_v6_accelerated_aggregate_run", source)
        self.assertNotIn("stage_tsmom_targeted_materialization_if_nonfutile", source)
        self.assertIn("tsmom_policy_symbols_at_checkpoint_v3", source)

    def test_precomputed_liquidity_chunk_proxy_matches_scalar_proxy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trade = root / "parquet/historical_trade_candles_5m"
            funding = root / "parquet/funding"
            trade.mkdir(parents=True)
            funding.mkdir(parents=True)
            sym = "PF_XBTUSD"
            for name, size, base in [
                ("PF_XBTUSD_20250101T000000.parquet", 101, trade),
                ("PF_XBTUSD_20250115T000000.parquet", 203, trade),
                ("PF_XBTUSD_20250220T000000.parquet", 307, trade),
                ("PF_XBTUSD_20250110T000000.parquet", 41, funding),
                ("PF_XBTUSD_20250210T000000.parquet", 53, funding),
            ]:
                (base / name).write_bytes(b"x" * size)
            ctx = SimpleNamespace(
                args=SimpleNamespace(kraken_data_root=str(root)),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-03-01", tz="UTC"),
            )
            panel = pd.DataFrame([{"symbol": sym, "first_trade_ts": "2024-01-01T00:00:00Z", "last_trade_ts": "2025-12-31T00:00:00Z", "status": "available"}])
            paths = {
                "root": root,
                "trade_5m": trade,
                "alt_trade": root / "parquet/trade",
                "funding": funding,
            }
            checkpoint = pd.Timestamp("2025-02-01T00:00:00Z")
            scalar = sweep.trailing_30d_liquidity_proxy(paths, sym, checkpoint)
            chunks = sweep.tsmom_precompute_liquidity_chunk_metadata(ctx, panel)
            cached = sweep.trailing_30d_liquidity_proxy_from_chunk_metadata(chunks, sym, checkpoint)
            for key in ["trailing_30d_trade_file_count", "trailing_30d_funding_file_count", "trailing_30d_trade_file_bytes", "trailing_30d_funding_file_bytes", "pit_liquidity_proxy_score"]:
                self.assertEqual(cached[key], scalar[key])

    def test_precomputed_panel_lifecycle_matches_scalar_lifecycle(self):
        panel = pd.DataFrame([
            {"symbol": "PF_XBTUSD", "status": "available", "start_ts": "2025-01-01T00:00:00Z", "end_ts": "2025-12-31T00:00:00Z", "bar_rows": 10},
            {"symbol": "PF_NEWUSD", "status": "available", "start_ts": "2025-02-01T00:00:00Z", "end_ts": "2025-12-31T00:00:00Z", "bar_rows": 10},
            {"symbol": "PF_ZEROUSD", "status": "available", "start_ts": "2025-01-01T00:00:00Z", "end_ts": "2025-12-31T00:00:00Z", "bar_rows": 0},
        ])
        lifecycle = sweep.precompute_panel_lifecycle(panel)
        for sym, ts in [
            ("PF_XBTUSD", pd.Timestamp("2025-01-15T00:00:00Z")),
            ("PF_NEWUSD", pd.Timestamp("2025-01-15T00:00:00Z")),
            ("PF_ZEROUSD", pd.Timestamp("2025-01-15T00:00:00Z")),
            ("PF_MISSING", pd.Timestamp("2025-01-15T00:00:00Z")),
        ]:
            self.assertEqual(
                sweep.panel_lifecycle_eligible_precomputed(lifecycle, panel, sym, ts),
                sweep.panel_lifecycle_eligible_at(panel, sym, ts),
            )

    def test_tsmom_rank_topn_cache_builder_profile_registered(self):
        self.assertIn(sweep.TSMOM_RANK_TOPN_CACHE_BUILDER_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_RANK_TOPN_CACHE_BUILDER_STAGES),
            [
                "preflight-and-source-freeze",
                "tsmom-rank-topn-cache-builder",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        profile = sweep.PHASE_PROFILES[sweep.TSMOM_RANK_TOPN_CACHE_BUILDER_PHASE_PROFILE]
        self.assertIn("TSMOM-only", profile["description"])

    def test_rank_topn_tie_policy_is_stable_hash_not_alphabetical(self):
        source = inspect.getsource(sweep.rank_top_n_symbols_for_decision)
        self.assertIn("tie_break_hash", source)
        self.assertIn("stable_hash", source)
        self.assertNotIn('sort_values(["score", "symbol"]', source)

    def test_rank_topn_builder_does_not_build_outcomes_or_aggregate(self):
        source = inspect.getsource(sweep.stage_tsmom_rank_topn_cache_builder)
        self.assertNotIn("stage_tsmom_v6_interval_outcome_cache_build", source)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir", source)
        self.assertNotIn("stage_full_tsmom_v6_accelerated_aggregate_run", source)
        self.assertNotIn("stage_tsmom_targeted_materialization_if_nonfutile", source)

    def test_tsmom_mask_selected_key_builder_profile_registered(self):
        self.assertIn(sweep.TSMOM_MASK_SELECTED_KEY_BUILDER_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_MASK_SELECTED_KEY_BUILDER_STAGES),
            [
                "preflight-and-source-freeze",
                "tsmom-mask-selected-key-builder",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        profile = sweep.PHASE_PROFILES[sweep.TSMOM_MASK_SELECTED_KEY_BUILDER_PHASE_PROFILE]
        self.assertIn("TSMOM-only", profile["description"])

    def test_mask_selected_key_builder_does_not_build_outcomes_or_aggregate(self):
        source = inspect.getsource(sweep.stage_tsmom_mask_selected_key_builder)
        self.assertNotIn("stage_tsmom_v6_interval_outcome_cache_build", source)
        self.assertNotIn("stage_full_tsmom_v6_accelerated_aggregate_run", source)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir", source)
        self.assertNotIn("event_from_address(", source)
        self.assertIn("semantic_cache_manifest", source)
        self.assertIn("selected_event_key_hash=\"\"", source)

    def test_selected_event_key_columns_exclude_outcome_fields(self):
        self.assertFalse(set(sweep.SELECTED_EVENT_KEY_COLUMNS) & set(cache_layer.OUTCOME_FIELDS))

    def test_outcome_manifest_without_selected_key_hash_is_blocked(self):
        with self.assertRaises(ValueError):
            cache_layer.semantic_cache_manifest(
                cache_class="tsmom_interval_outcome",
                df=pd.DataFrame(columns=["decision_ts"]),
                sort_keys=["decision_ts"],
                policy_fields={"test": "blocked"},
                input_manifest_hash="input",
                protected_train_boundary=str(sweep.PROTECTED_TS),
                selected_event_key_hash="",
            )

    def test_tsmom_outcome_grouped_aggregate_profile_registered(self):
        self.assertIn(sweep.TSMOM_OUTCOME_GROUPED_AGGREGATE_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_OUTCOME_GROUPED_AGGREGATE_STAGES),
            [
                "preflight-and-source-freeze",
                "tsmom-outcome-grouped-aggregate",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        profile = sweep.PHASE_PROFILES[sweep.TSMOM_OUTCOME_GROUPED_AGGREGATE_PHASE_PROFILE]
        self.assertIn("TSMOM-only", profile["description"])

    def test_grouped_tsmom_aggregate_preserves_raw_and_scaled_components(self):
        candidates = pd.DataFrame([
            {
                "candidate_definition_id": "d1",
                "candidate_symbol_id": "d1__PF_XBTUSD",
                "symbol": "PF_XBTUSD",
                "parameter_vector_hash": "p1",
                "family_engine_id": "scheduled_tsmom_engine",
                "side": "long",
            },
            {
                "candidate_definition_id": "d2",
                "candidate_symbol_id": "d2__PF_ETHUSD",
                "symbol": "PF_ETHUSD",
                "parameter_vector_hash": "p2",
                "family_engine_id": "scheduled_tsmom_engine",
                "side": "long",
            },
        ])
        outcomes = pd.DataFrame([
            {
                "candidate_definition_id": "d1",
                "candidate_symbol_id": "d1__PF_XBTUSD",
                "symbol": "PF_XBTUSD",
                "parameter_vector_hash": "p1",
                "decision_ts": "2025-01-01T00:00:00Z",
                "gross_R": 2.0,
                "fees_R": -0.1,
                "funding_R": -0.2,
                "slippage_R": 0.0,
                "net_R": 1.7,
                "raw_gross_R": 1.0,
                "raw_fee_R": -0.05,
                "raw_funding_R": -0.1,
                "raw_slippage_R": 0.0,
                "raw_net_R": 0.85,
                "scaled_gross_R": 2.0,
                "scaled_fee_R": -0.1,
                "scaled_funding_R": -0.2,
                "scaled_slippage_R": 0.0,
                "scaled_net_R": 1.7,
                "funding_timestamps_crossed": 1,
                "funding_boundary_count_exact": 1,
                "funding_boundary_count_proxy": 0,
                "funding_exact": True,
                "funding_proxy_used": False,
                "label_cap_reason": "kraken_survivorship_lifecycle_cap",
            },
            {
                "candidate_definition_id": "d1",
                "candidate_symbol_id": "d1__PF_XBTUSD",
                "symbol": "PF_XBTUSD",
                "parameter_vector_hash": "p1",
                "decision_ts": "2025-01-02T00:00:00Z",
                "gross_R": -1.0,
                "fees_R": -0.1,
                "funding_R": 0.0,
                "slippage_R": 0.0,
                "net_R": -1.1,
                "raw_gross_R": -0.5,
                "raw_fee_R": -0.05,
                "raw_funding_R": 0.0,
                "raw_slippage_R": 0.0,
                "raw_net_R": -0.55,
                "scaled_gross_R": -1.0,
                "scaled_fee_R": -0.1,
                "scaled_funding_R": 0.0,
                "scaled_slippage_R": 0.0,
                "scaled_net_R": -1.1,
                "funding_timestamps_crossed": 0,
                "funding_boundary_count_exact": 0,
                "funding_boundary_count_proxy": 0,
                "funding_exact": False,
                "funding_proxy_used": True,
                "label_cap_reason": "funding_proxy_cap",
            },
        ])
        grouped = sweep.grouped_tsmom_aggregate_summary(candidates, outcomes)
        row = grouped[grouped["candidate_symbol_id"].eq("d1__PF_XBTUSD")].iloc[0]
        self.assertEqual(int(row["events"]), 2)
        self.assertAlmostEqual(float(row["gross_R"]), 1.0)
        self.assertAlmostEqual(float(row["raw_gross_R"]), 0.5)
        self.assertAlmostEqual(float(row["scaled_net_R"]), 0.6)
        self.assertEqual(int(row["funding_boundary_count"]), 1)
        self.assertEqual(int(row["exact_funding_count"]), 1)
        self.assertEqual(int(row["proxy_funding_count"]), 1)
        zero = grouped[grouped["candidate_symbol_id"].eq("d2__PF_ETHUSD")].iloc[0]
        self.assertEqual(int(zero["events"]), 0)
        self.assertFalse(bool(zero["event_sampling_used"]))

    def test_outcome_grouped_stage_does_not_call_scalar_aggregate_fallback(self):
        source = inspect.getsource(sweep.stage_tsmom_outcome_grouped_aggregate)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir", source)
        self.assertNotIn("stage_full_tsmom_v6_accelerated_aggregate_run", source)

    def test_tsmom_end_to_end_scaling_benchmark_profile_registered(self):
        self.assertIn(sweep.TSMOM_END_TO_END_SCALING_BENCHMARK_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.TSMOM_END_TO_END_SCALING_BENCHMARK_STAGES),
            [
                "preflight-and-source-freeze",
                "tsmom-end-to-end-scaling-benchmark",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        profile = sweep.PHASE_PROFILES[sweep.TSMOM_END_TO_END_SCALING_BENCHMARK_PHASE_PROFILE]
        self.assertIn("TSMOM-only", profile["description"])

    def test_tsmom_end_to_end_scaling_benchmark_does_not_launch_full_paths(self):
        source = inspect.getsource(sweep.stage_tsmom_end_to_end_scaling_benchmark)
        self.assertNotIn("stage_full_tsmom_v6_accelerated_aggregate_run", source)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir(", source)
        self.assertNotIn("scalar_tsmom_v6_benchmark_events(", source)
        self.assertNotIn("scalar_selected_tsmom_event_keys(", source)

    def test_path_tree_size_bytes_counts_nested_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").write_bytes(b"123")
            (root / "nested").mkdir()
            (root / "nested" / "b").write_bytes(b"45")
            self.assertEqual(sweep.path_tree_size_bytes(root), 5)

    def test_full_tsmom_v6_cache_dry_run_profile_registered(self):
        self.assertIn(sweep.FULL_TSMOM_V6_CACHE_DRY_RUN_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.FULL_TSMOM_V6_CACHE_DRY_RUN_STAGES),
            [
                "preflight-and-source-freeze",
                "full-tsmom-v6-cache-dry-run",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        profile = sweep.PHASE_PROFILES[sweep.FULL_TSMOM_V6_CACHE_DRY_RUN_PHASE_PROFILE]
        self.assertIn("TSMOM-only", profile["description"])

    def test_full_tsmom_v6_cache_dry_run_does_not_build_outcomes_or_aggregate(self):
        source = inspect.getsource(sweep.stage_full_tsmom_v6_cache_dry_run)
        self.assertNotIn("build_tsmom_outcomes_from_frozen_keys(", source)
        self.assertNotIn("stage_full_tsmom_v6_accelerated_aggregate_run(", source)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir(", source)
        self.assertNotIn("scalar_tsmom_v6_benchmark_events(", source)

    def test_full_tsmom_v6_aggregate_profile_registered(self):
        self.assertIn(sweep.FULL_TSMOM_V6_AGGREGATE_PHASE_PROFILE, sweep.PHASE_PROFILES)
        self.assertEqual(
            list(sweep.FULL_TSMOM_V6_AGGREGATE_STAGES),
            [
                "preflight-and-source-freeze",
                "full-tsmom-v6-aggregate-run",
                "decision-report",
                "compact-review-bundle",
            ],
        )
        profile = sweep.PHASE_PROFILES[sweep.FULL_TSMOM_V6_AGGREGATE_PHASE_PROFILE]
        self.assertIn("TSMOM-only", profile["description"])

    def test_full_tsmom_v6_aggregate_avoids_old_full_route_and_caps(self):
        source = inspect.getsource(sweep.stage_full_tsmom_v6_aggregate_run)
        self.assertNotIn("stage_full_tsmom_v6_accelerated_aggregate_run(", source)
        self.assertNotIn("run_tsmom_v6_reused_cache_benchmark(", source)
        self.assertNotIn("aggregate_candidates_memorysafe_to_dir(", source)
        self.assertNotIn("event_sampling_used=True", source)
        self.assertNotIn("event_cap_used=True", source)

    def test_tsmom_signal_mask_matches_completed_bar_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = sweep.Context(
                args=SimpleNamespace(kraken_data_root="/tmp/kraken"),
                run_root=Path(td),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-01-02", tz="UTC"),
                notifier=SimpleNamespace(status={}),
            )
            bars = pd.DataFrame({
                "ts": pd.to_datetime([
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:05:00Z",
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:15:00Z",
                ], utc=True),
                "open": [100.0, 100.0, 110.0, 90.0],
                "high": [101.0, 101.0, 111.0, 91.0],
                "low": [99.0, 99.0, 109.0, 89.0],
                "close": [100.0, 100.0, 110.0, 90.0],
            })
            registry = pd.DataFrame([{
                "candidate_definition_id": "d_tsmom",
                "candidate_symbol_id": "d_tsmom__PF_XBTUSD",
                "candidate_id": "d_tsmom__PF_XBTUSD",
                "definition_id": "d_tsmom",
                "hypothesis_id": "H02",
                "family": "Volatility-managed TSMOM",
                "family_engine_id": "scheduled_tsmom_engine",
                "symbol": "PF_XBTUSD",
                "symbol_id": "PF_XBTUSD",
                "lookback_bars": 2,
                "hold_bars": 48,
                "threshold": 0.05,
                "side": "long",
            }])
            events = pd.DataFrame([
                {
                    "signal_policy_hash": "sig",
                    "candidate_definition_id": "d_tsmom",
                    "candidate_symbol_id": "d_tsmom__PF_XBTUSD",
                    "symbol_id": "PF_XBTUSD",
                    "symbol": "PF_XBTUSD",
                    "decision_ts": "2025-01-01T00:10:00+00:00",
                    "idx": 2,
                    "entry_idx": 3,
                    "exit_idx": 3,
                    "seq": 99,
                },
                {
                    "signal_policy_hash": "sig",
                    "candidate_definition_id": "d_tsmom",
                    "candidate_symbol_id": "d_tsmom__PF_XBTUSD",
                    "symbol_id": "PF_XBTUSD",
                    "symbol": "PF_XBTUSD",
                    "decision_ts": "2025-01-01T00:15:00+00:00",
                    "idx": 3,
                    "entry_idx": 4,
                    "exit_idx": 4,
                    "seq": 100,
                },
            ])
            original_loader = sweep.load_symbol_bars
            original_signal_loader = sweep.load_symbol_signal_bars
            try:
                sweep.load_symbol_bars = lambda paths, symbol, start, end: bars.copy()
                sweep.load_symbol_signal_bars = lambda paths, symbol, start, end: bars[["ts", "close"]].copy()
                signal, passed = sweep.tsmom_signal_masks_for_calendar_events(ctx, events, registry, {})
            finally:
                sweep.load_symbol_bars = original_loader
                sweep.load_symbol_signal_bars = original_signal_loader
            self.assertEqual(signal["signal_pass"].tolist(), [True, False])
            self.assertEqual(len(passed), 1)
            self.assertEqual(int(passed["idx"].iloc[0]), 2)
            self.assertEqual(int(passed["seq"].iloc[0]), 0)
            self.assertLessEqual(pd.Timestamp(passed["signal_feature_available_ts"].iloc[0]), pd.Timestamp(passed["decision_ts"].iloc[0]))

    def test_rank_feature_specs_are_unique_by_semantics_not_candidate_rows(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = sweep.Context(
                args=SimpleNamespace(
                    kraken_data_root="/tmp/kraken",
                    k0_root="k0",
                    structural_repair_root="struct",
                    stat_protocol_repair_root="stat",
                    p1_repair_root="p1",
                ),
                run_root=Path(td),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-02-01", tz="UTC"),
                notifier=SimpleNamespace(status={}),
            )
            rows = pd.DataFrame([
                {"candidate_definition_id": "d1", "symbol": "PF_XBTUSD", "bar_timeframe": "4h", "rebalance_interval": "8h", "lookback_days": 20, "lookback_bars": 5760, "hold_interval": "3d", "hold_bars": 864, "rank_top_n": 5, "rank_metric": "trailing_return"},
                {"candidate_definition_id": "d1", "symbol": "PF_ETHUSD", "bar_timeframe": "4h", "rebalance_interval": "8h", "lookback_days": 20, "lookback_bars": 5760, "hold_interval": "3d", "hold_bars": 864, "rank_top_n": 5, "rank_metric": "trailing_return"},
                {"candidate_definition_id": "d2", "symbol": "PF_XBTUSD", "bar_timeframe": "4h", "rebalance_interval": "8h", "lookback_days": 40, "lookback_bars": 11520, "hold_interval": "3d", "hold_bars": 864, "rank_top_n": 5, "rank_metric": "trailing_return"},
            ])
            specs = sweep.tsmom_rank_feature_specs_from_registry(rows, ctx)
            self.assertEqual(len(specs), 2)
            self.assertEqual(int(specs["candidate_symbol_row_count"].sum()), 3)

    def test_rank_feature_scope_uses_pit_universe_symbols_not_registry_subset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "audit").mkdir(parents=True)
            pd.DataFrame([
                {
                    "universe_cache_hash": "u1",
                    "selected_symbols": "PF_XBTUSD;PF_ETHUSD;PF_ZROUSD",
                }
            ]).to_csv(root / "audit/universe_decision_audit.csv", index=False)
            symbols = sweep.tsmom_rank_population_symbols_from_universe_audit(root, ["u1"])
            self.assertEqual(symbols, ["PF_ETHUSD", "PF_XBTUSD", "PF_ZROUSD"])
            ctx = sweep.Context(
                args=SimpleNamespace(
                    kraken_data_root="/tmp/kraken",
                    k0_root="k0",
                    structural_repair_root="struct",
                    stat_protocol_repair_root="stat",
                    p1_repair_root="p1",
                ),
                run_root=root,
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-02-01", tz="UTC"),
                notifier=SimpleNamespace(status={}),
            )
            rows = pd.DataFrame([
                {"candidate_definition_id": "d1", "symbol": "PF_XBTUSD", "bar_timeframe": "4h", "rebalance_interval": "8h", "lookback_days": 20, "lookback_bars": 5760, "rank_top_n": 3, "rank_metric": "trailing_return"},
                {"candidate_definition_id": "d1", "symbol": "PF_ETHUSD", "bar_timeframe": "4h", "rebalance_interval": "8h", "lookback_days": 20, "lookback_bars": 5760, "rank_top_n": 3, "rank_metric": "trailing_return"},
            ])
            registry_scope = sweep.stable_hash("PF_ETHUSD", "PF_XBTUSD", n=32)
            pit_scope = sweep.stable_hash(*symbols, n=32)
            registry_specs = sweep.tsmom_rank_feature_specs_from_registry(rows, ctx, symbol_scope_hash=registry_scope)
            pit_specs = sweep.tsmom_rank_feature_specs_from_registry(rows, ctx, symbol_scope_hash=pit_scope)
            self.assertNotEqual(registry_specs["rank_feature_hash"].iloc[0], pit_specs["rank_feature_hash"].iloc[0])
            self.assertEqual(pit_specs["symbol_scope_hash"].iloc[0], pit_scope)
            self.assertEqual(pit_specs["symbol_scope"].iloc[0], "pit_universe_selected_symbol_union_for_ranked_policies")

    def test_rank_feature_rows_use_completed_bars_strictly_before_decision(self):
        bars = pd.DataFrame({
            "ts": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T12:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "close": [100.0, 110.0, 1000.0],
        })
        rows = sweep.rank_feature_rows_from_close_window(
            bars,
            [pd.Timestamp("2025-01-02T00:00:00Z")],
            rank_feature_hash="rank",
            symbol="PF_XBTUSD",
            lookback_days=2,
            rank_metric="trailing_return",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_ts"], "2025-01-01T12:00:00+00:00")
        self.assertAlmostEqual(rows[0]["rank_score"], 0.10)
        self.assertLess(pd.Timestamp(rows[0]["feature_available_ts"]), pd.Timestamp(rows[0]["decision_ts"]))

    def write_targeted_tsmom_fixture_roots(self, root: Path, *, mutate_shortlist: bool = False) -> tuple[Path, Path, Path]:
        survivor = root / "survivor"
        full = root / "full"
        outcome = root / "outcome"
        (survivor / "selection").mkdir(parents=True)
        (full / "aggregate").mkdir(parents=True)
        (outcome / "cache").mkdir(parents=True)
        selected_hash = "selected_hash"
        shortlist_ids = list(sweep.TSMOM_V6_TARGETED_SHORTLIST_IDS)
        if mutate_shortlist:
            shortlist_ids[-1] = "tsmom_v6_999"
        shortlist_rows = []
        aggregate_rows = []
        definition_rows = []
        outcome_rows = []
        for i, cid in enumerate(shortlist_ids):
            parameter_hash = f"hash_{cid}"
            event_count = i + 2
            raw_net = -1.0 if cid == "tsmom_v6_059" else 1.0
            scaled_net = 2.0
            shortlist_rows.append({
                "candidate_definition_id": cid,
                "parameter_vector_hash": parameter_hash,
                "selected_event_key_hash": selected_hash,
                "actual_event_rows": event_count,
                "raw_net_R": raw_net,
                "scaled_net_R": scaled_net,
            })
            aggregate_rows.append({"candidate_definition_id": cid, "parameter_vector_hash": parameter_hash})
            definition_rows.append({"candidate_definition_id": cid, "events": event_count})
            for j in range(event_count):
                outcome_rows.append({
                    "candidate_definition_id": cid,
                    "candidate_symbol_id": f"{cid}__PF_XBTUSD",
                    "symbol": "PF_XBTUSD",
                    "decision_ts": f"2025-01-{j + 1:02d}T00:00:00+00:00",
                    "event_id": f"{cid}_{j}",
                    "seq": j,
                    "parameter_vector_hash": parameter_hash,
                    "selected_event_keys_hash": selected_hash,
                    "scaled_net_R": 0.1,
                    "raw_net_R": 0.1,
                    "scaled_gross_R": 0.2,
                    "scaled_fee_R": -0.01,
                    "scaled_funding_R": -0.02,
                    "scaled_slippage_R": 0.0,
                    "vol_scale": 1.0,
                    "funding_exact": True,
                    "funding_proxy_used": False,
                    "funding_boundary_crossed": True,
                    "parent_gate_pass": True,
                    "parent_gate_feature_source_ts": "2024-12-31T20:00:00+00:00",
                })
        deferred_rows = [{"candidate_definition_id": cid} for cid in sweep.TSMOM_V6_TARGETED_DEFERRED_DUPLICATE_IDS]
        sweep.write_csv(survivor / "selection/survivor_shortlist.csv", shortlist_rows)
        sweep.write_csv(survivor / "selection/rejected_or_deferred_candidates.csv", deferred_rows)
        sweep.write_csv(survivor / "selection/duplicate_cluster_report.csv", [{"cluster_id": "c"}])
        sweep.write_json(full / "decision_summary.json", {
            "status": "complete",
            "exactness_sentinel_pass": True,
            "mismatch_count": 0,
            "protected_interval_violations": 0,
            "decision_input_leak_violations": 0,
            "static_topn_failures": 0,
            "selected_event_key_hash": selected_hash,
        })
        sweep.write_csv(full / "aggregate/tsmom_v6_full_aggregate_summary.csv", aggregate_rows)
        sweep.write_csv(full / "aggregate/tsmom_v6_definition_level_aggregate_summary.csv", definition_rows)
        sweep.write_json(outcome / "decision_summary.json", {"status": "complete", "exactness_pass": True, "mismatch_count": 0})
        sweep.write_csv(outcome / "cache/tsmom_interval_outcome_manifest.csv", [{"selected_event_key_hash": selected_hash, "status": "pass"}])
        pd.DataFrame(outcome_rows).to_parquet(outcome / "cache/tsmom_interval_outcome.parquet", index=False)
        return survivor, full, outcome

    def test_tsmom_v6_targeted_materialization_profiles_are_registered(self):
        dry = sweep.parse_args(["--phase-profile", sweep.TSMOM_V6_TARGETED_MATERIALIZATION_PROFILE_PHASE_PROFILE, "--stage", "all"])
        launch = sweep.parse_args(["--phase-profile", sweep.TSMOM_V6_TARGETED_MATERIALIZATION_CONTROLS_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(dry), sweep.TSMOM_V6_TARGETED_MATERIALIZATION_PROFILE_RUN_ID)
        self.assertEqual(sweep.active_stage_list(dry), list(sweep.TSMOM_V6_TARGETED_MATERIALIZATION_PROFILE_STAGES))
        self.assertEqual(sweep.active_run_id(launch), sweep.TSMOM_V6_TARGETED_MATERIALIZATION_CONTROLS_RUN_ID)
        self.assertIn("tsmom-v6-targeted-lineage-gate", sweep.active_stage_list(launch))
        self.assertIn("tsmom-v6-targeted-materialization", sweep.active_stage_list(launch))
        self.assertNotIn("prior-high-reclaim-v1-aggregate-screen", sweep.active_stage_list(launch))

    def test_targeted_tsmom_lineage_gate_freezes_exact_shortlist_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            survivor, full, outcome = self.write_targeted_tsmom_fixture_roots(root)
            ctx = SimpleNamespace(
                run_root=root / "run",
                args=SimpleNamespace(
                    candidate_definition_manifest=str(survivor / "selection/survivor_shortlist.csv"),
                    prior_benchmark_root=str(full),
                    cache_root=str(outcome),
                    phase_profile=sweep.TSMOM_V6_TARGETED_MATERIALIZATION_PROFILE_PHASE_PROFILE,
                ),
            )
            lineage, audit, freeze = sweep.validate_tsmom_v6_targeted_lineage(ctx, write_outputs=True)
            self.assertEqual(lineage["status"], "pass")
            self.assertEqual(lineage["shortlisted_definitions_frozen"], sorted(sweep.TSMOM_V6_TARGETED_SHORTLIST_IDS))
            self.assertEqual(lineage["deferred_duplicates_frozen"], sorted(sweep.TSMOM_V6_TARGETED_DEFERRED_DUPLICATE_IDS))
            self.assertTrue(audit["status"].eq("pass").all())
            self.assertEqual(len(freeze), 7)
            self.assertTrue((ctx.run_root / "preflight/shortlist_lineage_audit.csv").exists())

    def test_targeted_tsmom_lineage_gate_fails_closed_on_shortlist_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            survivor, full, outcome = self.write_targeted_tsmom_fixture_roots(root, mutate_shortlist=True)
            ctx = SimpleNamespace(
                run_root=root / "run",
                args=SimpleNamespace(
                    candidate_definition_manifest=str(survivor / "selection/survivor_shortlist.csv"),
                    prior_benchmark_root=str(full),
                    cache_root=str(outcome),
                    phase_profile=sweep.TSMOM_V6_TARGETED_MATERIALIZATION_PROFILE_PHASE_PROFILE,
                ),
            )
            lineage, audit, _ = sweep.validate_tsmom_v6_targeted_lineage(ctx, write_outputs=True)
            self.assertEqual(lineage["status"], "fail")
            self.assertIn("shortlist_exact_expected_ids", set(audit.loc[audit["status"].eq("fail"), "check"]))

    def test_a1_compression_binding_cache_dry_run_profile_registered(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_BINDING_CACHE_DRY_RUN_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_BINDING_CACHE_DRY_RUN_RUN_ID)
        self.assertEqual(sweep.active_stage_list(args), list(sweep.A1_COMPRESSION_BINDING_CACHE_DRY_RUN_STAGES))
        self.assertIn("a1-compression-binding-cache-dry-run", sweep.active_stage_list(args))
        self.assertNotIn("full-tsmom-v6-aggregate-run", sweep.active_stage_list(args))
        self.assertNotIn("tsmom-v6-targeted-materialization", sweep.active_stage_list(args))

    def test_a1_compression_stage_does_not_launch_aggregate_or_outcomes(self):
        source = inspect.getsource(sweep.stage_a1_compression_binding_cache_dry_run)
        forbidden = [
            "stage_full_tsmom_v6_aggregate_run(",
            "stage_tsmom_outcome_grouped_aggregate(",
            "stage_tsmom_v6_targeted_materialization(",
            "aggregate_candidates_memorysafe_to_dir(",
            "build_tsmom_outcomes_from_frozen_keys(",
        ]
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_a1_universe_policy_binding_is_explicit_not_text_inference(self):
        row = {"universe_policy": "majors_only", "leader_rank_metric": "relative_strength_20d", "leader_top_n": 5}
        ctx = SimpleNamespace(start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2025-12-31", tz="UTC"), args=SimpleNamespace(kraken_data_root=str(sweep.DEFAULT_KRAKEN_DATA_ROOT)), run_root=Path("/tmp/a1_test"))
        bound = sweep.a1_engine_candidate(row, ctx)
        self.assertEqual(bound["universe_policy"], "pit_liquidity_top_majors")
        self.assertEqual(bound["engine_route"], "explicit_a1_h06_h12_h13_cache_binding_no_legacy_text_inference")
        self.assertEqual(bound["rank_metric"], "relative_strength")

    def test_a1_universe_policy_binding_idempotent_and_unknown_fail_closed(self):
        ctx = SimpleNamespace(start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2025-12-31", tz="UTC"), args=SimpleNamespace(kraken_data_root=str(sweep.DEFAULT_KRAKEN_DATA_ROOT)), run_root=Path("/tmp/a1_test"))
        major = sweep.a1_engine_candidate({"universe_policy": "pit_liquidity_top_majors", "leader_rank_metric": "relative_strength_20d", "leader_top_n": 5}, ctx)
        self.assertEqual(major["universe_policy"], "pit_liquidity_top_majors")
        self.assertEqual(major["a1_universe_binding_status"], "active_idempotent_pit_top_major_policy")
        tier = sweep.a1_engine_candidate({"universe_policy": "pit_liquidity_tier_ab", "leader_rank_metric": "relative_strength_20d", "leader_top_n": 5}, ctx)
        self.assertEqual(tier["universe_policy"], "pit_liquidity_tier_ab")
        self.assertEqual(tier["a1_universe_binding_status"], "active_idempotent_pit_tier_ab_policy")
        with self.assertRaises(RuntimeError):
            sweep.a1_engine_candidate({"universe_policy": "unknown_policy", "leader_rank_metric": "relative_strength_20d", "leader_top_n": 5}, ctx)

    def test_a1_decision_input_cache_rejects_outcome_fields(self):
        df = pd.DataFrame([{
            "candidate_definition_id": "a1",
            "symbol": "PF_XBTUSD",
            "decision_ts": "2025-01-01T00:00:00Z",
            "feature_available_ts": "2024-12-31T20:00:00Z",
            "raw_net_R": 1.0,
        }])
        with self.assertRaises(ValueError):
            cache_layer.semantic_cache_manifest(
                cache_class="a1_breakout_signal",
                df=df,
                sort_keys=["candidate_definition_id", "symbol", "decision_ts"],
                policy_fields={"test": "outcome_field_rejection"},
                input_manifest_hash="input",
                protected_train_boundary=str(sweep.PROTECTED_TS),
            )

    def test_a1_compression_mechanical_canary_profile_registered(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_MECHANICAL_CANARY_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_MECHANICAL_CANARY_RUN_ID)
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.A1_COMPRESSION_MECHANICAL_CANARY_STAGES))
        self.assertIn("a1-compression-mechanical-canary", stages)
        self.assertNotIn("full-tsmom-v6-aggregate-run", stages)
        self.assertNotIn("tsmom-v6-targeted-materialization", stages)

    def test_a1_compression_full_train_aggregate_profile_registered_and_isolated(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_FULL_TRAIN_AGGREGATE_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_FULL_TRAIN_AGGREGATE_RUN_ID)
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.A1_COMPRESSION_FULL_TRAIN_AGGREGATE_STAGES))
        self.assertIn("a1-compression-full-train-aggregate", stages)
        self.assertNotIn("full-tsmom-v6-aggregate-run", stages)
        self.assertNotIn("tsmom-v6-targeted-materialization", stages)
        self.assertNotIn("prior-high-reclaim-v1-aggregate-screen", stages)

    def test_a1_semantic_cache_repair_profile_registered(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_SEMANTIC_CACHE_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_SEMANTIC_CACHE_REPAIR_RUN_ID)
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.A1_COMPRESSION_SEMANTIC_CACHE_REPAIR_STAGES))
        self.assertIn("a1-compression-semantic-cache-builder-repair", stages)
        self.assertNotIn("a1-compression-full-train-aggregate", stages)
        self.assertNotIn("tsmom-v6-targeted-materialization", stages)

    def test_a1_full_train_cache_dry_run_profile_registered(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_FULL_TRAIN_CACHE_DRY_RUN_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_FULL_TRAIN_CACHE_DRY_RUN_RUN_ID)
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.A1_COMPRESSION_FULL_TRAIN_CACHE_DRY_RUN_STAGES))
        self.assertIn("a1-compression-full-train-cache-dry-run", stages)
        self.assertNotIn("a1-compression-full-train-aggregate", stages)
        self.assertNotIn("tsmom-v6-targeted-materialization", stages)

    def test_a1_true_streaming_cache_repair_profile_registered(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_TRUE_STREAMING_CACHE_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_TRUE_STREAMING_CACHE_REPAIR_RUN_ID)
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.A1_COMPRESSION_TRUE_STREAMING_CACHE_REPAIR_STAGES))
        self.assertIn("a1-compression-true-streaming-cache-builder-repair", stages)
        self.assertNotIn("a1-compression-full-train-aggregate", stages)
        self.assertNotIn("tsmom-v6-targeted-materialization", stages)

    def test_a1_selected_key_compiler_repair_profile_registered(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_SELECTED_KEY_COMPILER_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_SELECTED_KEY_COMPILER_REPAIR_RUN_ID)
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.A1_COMPRESSION_SELECTED_KEY_COMPILER_REPAIR_STAGES))
        self.assertIn("a1-compression-selected-key-compiler-repair", stages)
        self.assertNotIn("a1-compression-full-train-aggregate", stages)
        self.assertNotIn("tsmom-v6-targeted-materialization", stages)

    def test_a1_true_streaming_profile_uses_streaming_writer(self):
        main_source = inspect.getsource(sweep.main)
        self.assertIn('"a1-compression-true-streaming-cache-builder-repair": stage_a1_compression_true_streaming_cache_builder_repair', main_source)
        source = inspect.getsource(sweep.stage_a1_compression_true_streaming_cache_builder_repair)
        self.assertIn("a1_build_true_streaming_cache_dry_run", source)
        self.assertNotIn("stage_a1_compression_full_train_aggregate(", source)
        self.assertNotIn("a1_outcome_event_from_selected(", source)

    def test_a1_streaming_memory_guard_blocks_pass(self):
        source = inspect.getsource(sweep.stage_a1_compression_true_streaming_cache_builder_repair)
        self.assertIn("memory_guard_triggered", source)
        self.assertIn("and not memory_guard_triggered", source)
        self.assertIn("full_aggregate_launched", source)
        self.assertIn("outcome_cache_built", source)

    def test_a1_full_train_cache_dry_run_does_not_build_outcomes_or_aggregate(self):
        source = inspect.getsource(sweep.stage_a1_compression_full_train_cache_dry_run)
        forbidden = [
            "a1_outcome_event_from_selected(",
            "a1_materialized_summary(",
            "to_parquet(",
            "write_a1_cache_table(ctx, \"full_train_interval_outcome",
            "stage_a1_compression_full_train_aggregate(",
        ]
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn('output_prefix="full_train"', source)

    def test_a1_full_train_stage_is_fail_closed_until_cache_dry_run(self):
        main_source = inspect.getsource(sweep.main)
        self.assertIn('"a1-compression-full-train-aggregate": stage_a1_compression_full_train_aggregate_cached', main_source)
        blocker_source = inspect.getsource(sweep.stage_a1_compression_full_train_aggregate_cached)
        self.assertIn("blocked until", blocker_source)
        self.assertNotIn("a1_feature_bundle(", blocker_source)

    def test_a1_semantic_cache_builder_does_not_call_feature_bundle_loop(self):
        source = inspect.getsource(sweep.a1_build_semantic_cache_tables)
        self.assertNotIn("a1_feature_bundle(", source)
        self.assertIn("a1_raw_feature_bundle_from_completed", source)

    def test_a1_selected_key_policy_hash_excludes_exit_only_fields(self):
        base = {
            "entry_spec_id": "entry_1",
            "definition_lane": "a1_impulse",
            "side": "long",
            "decision_timeframe": "4h",
            "universe_policy": "tier_ab_liquid_strict",
            "leader_rank_metric": "relative_strength_20d",
            "leader_top_n": 5,
            "parent_regime_gate": "btc_eth_trend_up",
            "funding_gate": "funding_aware_cap",
            "entry_policy_hash": "entry_hash",
            "universe_policy_hash": "universe_hash",
            "rank_policy_hash": "rank_hash",
            "regime_policy_hash": "regime_hash",
            "funding_policy_hash": "funding_hash",
            "exit_policy_id": "fixed_hold_3d_atr_stop_1p0",
            "hold_interval": "3d",
            "atr_stop_mult": 1.0,
            "parameter_vector_hash": "pv1",
            "candidate_definition_id": "a1_exit_1",
        }
        changed_exit = dict(base, exit_policy_id="atr_trail_2x_time_5d", hold_interval="5d", atr_stop_mult=2.0, parameter_vector_hash="pv2", candidate_definition_id="a1_exit_2")
        self.assertEqual(sweep.a1_selected_key_policy_hash(base), sweep.a1_selected_key_policy_hash(changed_exit))

    def test_a1_selected_key_policy_hash_changes_on_entry_field(self):
        base = {
            "entry_spec_id": "entry_1",
            "definition_lane": "a1_impulse",
            "side": "long",
            "decision_timeframe": "4h",
            "universe_policy": "tier_ab_liquid_strict",
            "leader_rank_metric": "relative_strength_20d",
            "leader_top_n": 5,
            "parent_regime_gate": "btc_eth_trend_up",
            "funding_gate": "funding_aware_cap",
            "entry_policy_hash": "entry_hash",
            "universe_policy_hash": "universe_hash",
            "rank_policy_hash": "rank_hash",
        }
        changed_entry = dict(base, leader_top_n=10)
        self.assertNotEqual(sweep.a1_selected_key_policy_hash(base), sweep.a1_selected_key_policy_hash(changed_entry))

    def test_a1_selected_key_compiler_source_avoids_per_definition_universe_append(self):
        source = inspect.getsource(sweep.a1_build_selected_key_compiler_dry_run)
        self.assertIn("selected_key_policy_hash", source)
        self.assertIn("groups", source)
        self.assertNotIn('rows_by_class["liquid_universe"].append', source)
        self.assertIn("universe_summary_rows.append", source)

    def test_a1_selected_key_compiler_shard_count_guard_present(self):
        source = inspect.getsource(sweep.a1_build_selected_key_compiler_dry_run)
        self.assertIn("max_open_buffer_rows > 10000", source)
        self.assertIn("shard_count > 10000", source)
        stage_source = inspect.getsource(sweep.stage_a1_compression_selected_key_compiler_repair)
        self.assertIn("shard_count < 10000", stage_source)

    def test_a1_selected_key_compiler_uses_prepared_funding_gate(self):
        source = inspect.getsource(sweep.a1_build_selected_key_compiler_dry_run)
        self.assertIn("a1_prepare_funding_frame", source)
        self.assertIn("a1_funding_gate_from_prepared", source)
        self.assertNotIn("a1_funding_gate(d,", source)

    def test_a1_feature_mask_compiler_repair_profile_registered(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_FEATURE_MASK_COMPILER_REPAIR_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_FEATURE_MASK_COMPILER_REPAIR_RUN_ID)
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.A1_COMPRESSION_FEATURE_MASK_COMPILER_REPAIR_STAGES))
        self.assertIn("a1-compression-feature-mask-compiler-repair", stages)
        self.assertNotIn("a1-compression-full-train-aggregate", stages)
        self.assertNotIn("tsmom-v6-targeted-materialization", stages)

    def test_a1_feature_mask_compiler_does_not_call_raw_bundle_hotpath(self):
        source = inspect.getsource(sweep.a1_build_feature_mask_compiler_dry_run)
        self.assertIn("a1_feature_mask_table_from_completed", source)
        self.assertIn("feature_mask_cache", source)
        self.assertIn("selected_key_policy_hash", source)
        self.assertNotIn("a1_raw_feature_bundle_from_completed", source)
        self.assertNotIn("a1_outcome_event_from_selected(", source)
        self.assertNotIn("stage_a1_compression_full_train_aggregate(", source)

    def test_a1_feature_mask_stage_is_benchmark_only(self):
        source = inspect.getsource(sweep.stage_a1_compression_feature_mask_compiler_repair)
        self.assertIn("single_full_window_spec_profile.csv", source)
        self.assertIn("balanced_32_spec_profile.csv", source)
        self.assertIn("a1_build_feature_mask_compiler_dry_run", source)
        self.assertNotIn("scale_512", source)
        self.assertNotIn("a1_outcome_event_from_selected(", source)
        self.assertNotIn("stage_a1_compression_full_train_aggregate(", source)
        self.assertIn("full_aggregate_launched", source)
        self.assertIn("outcome_cache_built", source)

    def test_a1_feature_mask_table_matches_scalar_feature_boundaries(self):
        ts = pd.date_range("2025-01-01", periods=80, freq="1D", tz="UTC")
        close = np.linspace(100.0, 220.0, len(ts))
        completed = pd.DataFrame({
            "source_ts": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(len(ts)),
        })
        defn = {
            "candidate_definition_id": "a1_fixture",
            "definition_lane": "a1_impulse_base_breakout",
            "decision_timeframe": "1d",
            "base_window_days": 10,
            "impulse_lookback_days": 20,
            "impulse_required": True,
            "impulse_return_threshold": 0.01,
            "compression_required": False,
            "breakout_trigger": "close_confirmed_breakout",
            "signal_policy_hash": "signal_fixture",
        }
        decisions = [ts[30], ts[50]]
        table = sweep.a1_feature_mask_table_from_completed(
            defn,
            completed,
            decisions,
            feature_spec_hash="feature_fixture",
            symbol_id="PF_FIXTUREUSD",
            window_id="fixture_window",
        )
        self.assertEqual(len(table), len(decisions))
        for decision in decisions:
            raw = sweep.a1_raw_feature_bundle_from_completed(defn, completed, decision)
            row = table[table["decision_ts"].astype(str).eq(pd.Timestamp(decision).isoformat())].iloc[0]
            for col in ["base_high", "base_low", "base_width_pct", "impulse_return", "rv", "rv_percentile", "range_slope", "current_close"]:
                self.assertAlmostEqual(float(row[col]), float(raw[col]), places=12)
            raw_signal = sweep.a1_signal_pass_from_raw_features(defn, raw)
            self.assertEqual(bool(row["signal_pass"]), bool(raw_signal[0]))
            self.assertLessEqual(pd.Timestamp(row["feature_available_ts"]), pd.Timestamp(decision))

    def test_a1_feature_mask_semantic_cache_signature_fields_present(self):
        source = inspect.getsource(sweep.a1_build_feature_mask_compiler_dry_run)
        for token in ["feature_spec_hash", "symbol_id", "decision_timeframe", "window_id", "feature_mask_content_hash"]:
            self.assertIn(token, source)
        self.assertIn("feature_mask_manifest_rows.append", source)

    def test_a1_production_sharded_aggregate_profile_registered(self):
        args = sweep.parse_args(["--phase-profile", sweep.A1_COMPRESSION_PRODUCTION_SHARDED_AGGREGATE_PHASE_PROFILE, "--stage", "all"])
        self.assertEqual(sweep.active_run_id(args), sweep.A1_COMPRESSION_PRODUCTION_SHARDED_AGGREGATE_RUN_ID)
        stages = sweep.active_stage_list(args)
        self.assertEqual(stages, list(sweep.A1_COMPRESSION_PRODUCTION_SHARDED_AGGREGATE_STAGES))
        self.assertIn("a1-compression-production-shard-plan", stages)
        self.assertIn("a1-compression-first-pack-sharded-aggregate", stages)
        self.assertIn("a1-compression-first-pack-reducer", stages)
        self.assertNotIn("a1-compression-full-train-aggregate", stages)
        self.assertNotIn("tsmom-v6-targeted-materialization", stages)

    def test_a1_first_pack_selection_is_lane_balanced_and_not_first_row(self):
        lanes = [
            "a1_impulse_base_breakout",
            "a1_plus_compression",
            "h12_rv_compression_breakout",
            "h13_flat_range_escape",
            "h06_vcp_like_contraction",
        ]
        rows = []
        for lane in lanes:
            for idx, count in enumerate([1, 100, 50], start=1):
                rows.append({
                    "shard_id": f"{lane}_{idx}",
                    "selected_key_policy_hash": f"{lane}_hash_{idx}",
                    "feature_spec_hash": f"feature_{idx}",
                    "definition_lane": lane,
                    "entry_spec_id": f"{lane}_entry_{idx:03d}",
                    "definition_count": 8,
                    "exit_policy_count": 8,
                    "candidate_definition_ids": "",
                    "exit_policy_ids": "",
                    "prior_selected_event_rows": count,
                    "first_pack": False,
                    "first_pack_role": "",
                    "selection_basis": "fixture",
                    "shard_status": "planned",
                })
        rows.append({**rows[0], "definition_lane": "short_diagnostic", "selected_key_policy_hash": "short_hash", "entry_spec_id": "short_entry", "prior_selected_event_rows": 1000})
        selected = sweep.a1_select_first_pack_shards(pd.DataFrame(rows))
        self.assertEqual(len(selected), 10)
        self.assertFalse(selected["definition_lane"].astype(str).eq("short_diagnostic").any())
        counts = selected.groupby("definition_lane").size().to_dict()
        self.assertTrue(all(counts.get(lane) == 2 for lane in lanes))
        self.assertFalse(selected["entry_spec_id"].astype(str).str.endswith("_001").any())
        self.assertEqual(set(selected["first_pack_role"].astype(str)), {"broad_or_high_event_count", "standard_representative"})

    def test_a1_first_pack_reducer_requires_exactly_ten_shards(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = SimpleNamespace(run_root=Path(td))
            expected = pd.DataFrame([{"shard_id": f"shard_{i}"} for i in range(9)])
            with self.assertRaises(RuntimeError):
                sweep.a1_reduce_first_pack_shards(ctx, expected)

    def test_a1_finalized_aggregate_shard_validation_rejects_hash_mismatch_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard_dir = root / "aggregate_shards/shard_a"
            shard_dir.mkdir(parents=True)
            aggregate = pd.DataFrame([{"candidate_definition_id": "a", "exit_policy_id": "x", "events": 1, "net_R": 1.234567890123}])
            aggregate.to_csv(shard_dir / "aggregate.csv", index=False)
            aggregate_written = pd.read_csv(shard_dir / "aggregate.csv")
            pd.DataFrame([{"status": "pass"}]).to_csv(shard_dir / "selected_key_manifest.csv", index=False)
            pd.DataFrame([{"status": "pass"}]).to_csv(shard_dir / "outcome_cache_manifest.csv", index=False)
            manifest = {
                "shard_id": "shard_a",
                "status": "complete",
                "aggregate_content_hash": sweep.canonical_frame_hash(aggregate_written, sort_keys=["candidate_definition_id", "exit_policy_id"]),
            }
            (shard_dir / "shard_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(sweep.a1_validate_finalized_aggregate_shard(root, "shard_a")["status"], "pass")
            bad = dict(manifest, aggregate_content_hash="bad")
            (shard_dir / "shard_manifest.json").write_text(json.dumps(bad))
            self.assertEqual(sweep.a1_validate_finalized_aggregate_shard(root, "shard_a")["status"], "fail")
            (shard_dir / "selected_key_manifest.csv").unlink()
            self.assertEqual(sweep.a1_validate_finalized_aggregate_shard(root, "shard_a")["reason"], "missing_required_shard_files")

    def test_a1_sharded_aggregate_source_guards(self):
        exec_source = inspect.getsource(sweep.a1_execute_economic_shard)
        reducer_source = inspect.getsource(sweep.a1_reduce_first_pack_shards)
        stage_source = inspect.getsource(sweep.stage_a1_compression_first_pack_sharded_aggregate)
        self.assertIn("selected_key_freeze_ts", exec_source)
        self.assertIn("outcome_start_ts", exec_source)
        self.assertIn("outcome_after_freeze", exec_source)
        self.assertIn("len(complete) != 10", reducer_source)
        self.assertIn("10/10 complete shard manifests", reducer_source)
        self.assertNotIn("stage_a1_compression_full_train_aggregate", stage_source)
        self.assertNotIn("full_train_interval_outcomes", exec_source)

    def test_a1_cache_reuse_audit_distinguishes_rebuilt_shard_local_paths(self):
        rows = sweep.a1_cache_reuse_rows_from_shard_result({
            "shard_id": "shard_a",
            "feature_mask_cache_source": "rebuilt_shard_local_feature_mask_inputs_for_selected_key",
            "feature_mask_cache_reused": False,
            "rebuilt_shard_local_feature_mask_cache": True,
            "selected_key_cache_source": "rebuilt_shard_local_selected_key_cache",
            "rebuilt_shard_local_selected_key_cache": True,
            "selected_event_key_hash": "abc",
            "outcome_cache_manifest_status": "pass",
        })
        by_class = {row["cache_class"]: row for row in rows}
        self.assertFalse(by_class["feature_mask"]["reused"])
        self.assertTrue(by_class["feature_mask"]["rebuilt"])
        self.assertFalse(by_class["selected_key"]["reused"])
        self.assertTrue(by_class["selected_key"]["rebuilt"])
        self.assertTrue(by_class["outcome"]["newly_built"])

    def test_a1_prepared_funding_gate_is_point_in_time(self):
        funding = pd.DataFrame([
            {"timestamp": f"2025-01-{day:02d}T00:00:00Z", "fundingRate": 0.0001 * day}
            for day in range(1, 32)
        ])
        prepared = sweep.a1_prepare_funding_frame(funding)
        result = sweep.a1_funding_gate_from_prepared(
            {"funding_gate": "exclude_top_decile_positive_funding"},
            prepared,
            pd.Timestamp("2025-02-01T00:00:00Z"),
        )
        self.assertEqual(result["status"], "filtered")
        self.assertLess(pd.Timestamp(result["feature_source_ts"]), pd.Timestamp("2025-02-01T00:00:00Z", tz="UTC"))

    def test_a1_vectorized_rank_scores_match_scalar(self):
        ts = pd.date_range("2025-01-01", periods=60, freq="1D", tz="UTC")
        close = np.linspace(100.0, 160.0, len(ts))
        panel = {"ts_ns": ts.astype("int64").to_numpy(), "close": close}
        decisions = [pd.Timestamp("2025-01-25T00:00:00Z"), pd.Timestamp("2025-02-15T00:00:00Z")]
        for metric in ["relative_strength", "risk_adjusted_return"]:
            vector = sweep.a1_rank_scores_for_decisions(panel, decisions, 20, metric)
            for decision in decisions:
                scalar_score, scalar_ts = sweep.a1_rank_score_from_panel(panel, decision, 20, metric)
                row = vector[pd.Timestamp(decision).isoformat()]
                self.assertLess(abs(row["score"] - scalar_score), 1e-9)
                self.assertEqual(row["source_ts"], scalar_ts)

    def test_a1_full_train_window_buffers_max_hold_before_protected_boundary(self):
        ctx = SimpleNamespace(
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2025-12-31T23:59:59Z", tz="UTC"),
        )
        manifest = pd.DataFrame([{"time_stop_days": 15, "hold_value": 15}])
        windows = sweep.a1_full_train_window_manifest(ctx, manifest)
        decision_end = pd.Timestamp(windows.iloc[0]["decision_end"])
        self.assertLess(decision_end + pd.Timedelta(days=15), sweep.PROTECTED_TS)
        self.assertEqual(windows.iloc[0]["protected_interval_policy"], "decision_window_ends_before_protected_boundary_by_max_declared_hold")

    def test_a1_full_train_uses_operator_supplied_prior_benchmark_root(self):
        source = inspect.getsource(sweep.stage_a1_compression_full_train_aggregate)
        self.assertIn("active_prior_benchmark_root(ctx.args)", source)
        self.assertNotIn("active_prior_benchmark_root(ctx)", source)

    def test_a1_selected_key_hash_consistency_fails_closed_on_blank_manifest_column(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "binding"
            out = Path(td) / "audit"
            (root / "cache/selected_event_key_shards").mkdir(parents=True)
            h = "abc123"
            (root / "decision_summary.json").write_text(json.dumps({"selected_event_key_hash": h}))
            (root / "cache/selected_event_key_freeze_summary.json").write_text(json.dumps({"selected_event_key_hash": h}))
            pd.DataFrame([{
                "selected_event_key_hash": "",
                "policy_fields": json.dumps({"selected_event_key_hash": h}),
            }]).to_csv(root / "cache/selected_event_key_manifest.csv", index=False)
            pd.DataFrame([{
                "candidate_identity_hash": "cid",
                "symbol_id": "PF_XBTUSD",
                "decision_ts": "2025-01-01T00:00:00Z",
                "selected_event_keys_hash": h,
            }]).to_csv(root / "cache/selected_event_key_shards/selected_event_key_part_000.csv", index=False)
            ok, expected = sweep.a1_selected_key_hash_consistency_audit(root, out)
            self.assertFalse(ok)
            self.assertEqual(expected, h)
            audit = pd.read_csv(out / "selected_key_hash_consistency_audit.csv")
            self.assertIn("selected_event_key_manifest_top_level", set(audit.loc[audit["status"].eq("fail"), "source"]))

    def test_a1_write_selected_event_key_manifest_includes_top_level_hash(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = SimpleNamespace(
                run_root=Path(td),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-01-02", tz="UTC"),
                args=SimpleNamespace(kraken_data_root=str(sweep.DEFAULT_KRAKEN_DATA_ROOT)),
            )
            df = pd.DataFrame([{
                "candidate_identity_hash": "cid",
                "symbol_id": "PF_XBTUSD",
                "decision_ts": "2025-01-01T00:00:00Z",
                "selected_event_keys_hash": "hash123",
            }])
            row = sweep.write_a1_cache_table(
                ctx,
                "selected_event_key",
                "selected_event_key",
                df,
                ["candidate_identity_hash", "symbol_id", "decision_ts"],
                {"family": "a1_test"},
                selected_event_key_hash="hash123",
            )
            self.assertEqual(row["selected_event_key_hash"], "hash123")
            manifest = pd.read_csv(Path(td) / "cache/selected_event_key_manifest.csv")
            self.assertEqual(str(manifest["selected_event_key_hash"].iloc[0]), "hash123")

    def test_a1_semantic_selected_key_hash_consistency_passes_when_all_sources_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "cache/a1_selected_event_key_shards").mkdir(parents=True)
            h = "a1_hash_123"
            (root / "cache/a1_selected_event_key_freeze_summary.json").write_text(json.dumps({"selected_event_key_hash": h}))
            pd.DataFrame([{
                "selected_event_key_hash": h,
                "policy_fields": json.dumps({"selected_event_key_hash": h}),
            }]).to_csv(root / "cache/a1_selected_event_key_manifest.csv", index=False)
            pd.DataFrame([{
                "candidate_identity_hash": "cid",
                "symbol_id": "PF_XBTUSD",
                "decision_ts": "2025-01-01T00:00:00Z",
                "selected_event_keys_hash": h,
            }]).to_csv(root / "cache/a1_selected_event_key_shards/a1_selected_event_key_part_000.csv", index=False)
            ctx = SimpleNamespace(run_root=root)
            audit = sweep.a1_selected_key_consistency_from_current_root(ctx)
            self.assertTrue(audit["status"].astype(str).eq("pass").all())

    def test_a1_full_train_selected_key_hash_consistency_includes_decision_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "cache/full_train_selected_event_key_shards").mkdir(parents=True)
            h = "full_train_hash_123"
            (root / "decision_summary.json").write_text(json.dumps({"selected_event_key_hash": h}))
            (root / "cache/full_train_selected_event_key_freeze_summary.json").write_text(json.dumps({"selected_event_key_hash": h}))
            pd.DataFrame([{
                "selected_event_key_hash": h,
                "policy_fields": json.dumps({"selected_event_key_hash": h}),
            }]).to_csv(root / "cache/full_train_selected_event_key_manifest.csv", index=False)
            pd.DataFrame([{
                "candidate_identity_hash": "cid",
                "symbol_id": "PF_XBTUSD",
                "decision_ts": "2025-01-01T00:00:00Z",
                "selected_event_keys_hash": h,
            }]).to_csv(root / "cache/full_train_selected_event_key_shards/full_train_selected_event_key_part_000.csv", index=False)
            ctx = SimpleNamespace(run_root=root)
            audit = sweep.a1_selected_key_consistency_from_current_root(ctx, cache_name="full_train_selected_event_key", include_decision_summary=True)
            self.assertTrue(audit["status"].astype(str).eq("pass").all())
            self.assertIn("decision_summary", set(audit["source"].astype(str)))

    def test_a1_streaming_shard_manifest_complete_rejects_temp_missing_and_stale(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = SimpleNamespace(
                run_root=Path(td),
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-01-02", tz="UTC"),
                args=SimpleNamespace(kraken_data_root=str(sweep.DEFAULT_KRAKEN_DATA_ROOT)),
            )
            rows = [{
                "candidate_definition_id": "a1",
                "symbol_id": "PF_XBTUSD",
                "decision_ts": "2025-01-01T00:00:00Z",
                "feature_available_ts": "2024-12-31T20:00:00Z",
            }]
            complete = sweep.a1_write_streaming_cache_shard(
                ctx,
                cache_name="test_signal",
                cache_class="a1_breakout_signal",
                rows=rows,
                sort_keys=["candidate_definition_id", "symbol_id", "decision_ts"],
                policy_fields={"test": "complete"},
                shard_index=0,
            )
            self.assertTrue(sweep.a1_streaming_manifest_is_complete(complete))
            stale = dict(complete)
            stale["semantic_cache_hash"] = "stale"
            self.assertFalse(sweep.a1_streaming_manifest_is_complete(stale))
            missing = dict(complete)
            missing["shard_path"] = ""
            self.assertFalse(sweep.a1_streaming_manifest_is_complete(missing))
            temp = sweep.a1_write_streaming_cache_shard(
                ctx,
                cache_name="test_selected",
                cache_class="selected_event_key",
                rows=[{
                    "candidate_identity_hash": "cid",
                    "symbol_id": "PF_XBTUSD",
                    "decision_ts": "2025-01-01T00:00:00Z",
                }],
                sort_keys=["candidate_identity_hash", "symbol_id", "decision_ts"],
                policy_fields={"test": "temp"},
                shard_index=0,
                temp_shard=True,
            )
            self.assertFalse(sweep.a1_streaming_manifest_is_complete(temp))

    def test_a1_semantic_manifest_stale_or_missing_fails_closed(self):
        df = pd.DataFrame([{
            "candidate_definition_id": "a1",
            "symbol": "PF_XBTUSD",
            "decision_ts": "2025-01-01T00:00:00Z",
            "feature_available_ts": "2024-12-31T20:00:00Z",
        }])
        manifest = cache_layer.semantic_cache_manifest(
            cache_class="a1_breakout_signal",
            df=df,
            sort_keys=["candidate_definition_id", "symbol", "decision_ts"],
            policy_fields={"signal": "breakout"},
            input_manifest_hash="input_hash",
            protected_train_boundary=str(sweep.PROTECTED_TS),
        )
        self.assertEqual(cache_layer.validate_semantic_cache_manifest(manifest)["status"], "pass")
        stale = dict(manifest)
        stale["semantic_cache_hash"] = "stale"
        self.assertEqual(cache_layer.validate_semantic_cache_manifest(stale)["status"], "fail")
        missing = dict(manifest)
        missing.pop("content_hash")
        self.assertEqual(cache_layer.validate_semantic_cache_manifest(missing)["status"], "fail")

    def test_a1_streaming_parent_gate_hotpath_uses_semantic_table(self):
        source = inspect.getsource(sweep.a1_build_true_streaming_cache_dry_run)
        self.assertIn("a1_parent_gate_semantic_table_for_window", source)
        self.assertIn("parent_gate_table_cache", source)
        self.assertNotIn("a1_parent_gate(d,", source)
        self.assertNotIn('pd.to_datetime(bars["ts"]', source)

    def test_a1_parent_gate_prepared_lookup_uses_completed_source_ts(self):
        frame = pd.DataFrame([
            {"source_ts": "2025-01-01T00:00:00Z", "close": 100.0, "sma_40d": 95.0, "ret_20d": 0.02, "up": True, "down": False},
            {"source_ts": "2025-01-01T04:00:00Z", "close": 101.0, "sma_40d": 96.0, "ret_20d": 0.03, "up": True, "down": False},
        ])
        prepared = sweep.a1_prepare_parent_gate_frame(frame)
        result = sweep.a1_parent_gate_from_prepared_frames(
            {"parent_regime_gate": "btc_trend_up"},
            pd.Timestamp("2025-01-01T03:00:00Z"),
            prepared,
            prepared,
        )
        self.assertTrue(result["allowed"])
        self.assertEqual(str(pd.Timestamp(result["feature_source_ts"])), str(pd.Timestamp("2025-01-01T00:00:00Z", tz="UTC")))
        self.assertTrue(result["source_ts_lte_decision"])

    def test_a1_parent_gate_semantic_cache_key_includes_policy_and_window(self):
        source = inspect.getsource(sweep.a1_build_true_streaming_cache_dry_run)
        self.assertIn('str(d.get("parent_gate_policy_hash", ""))', source)
        self.assertIn('str(d.get("parent_regime_gate", ""))', source)
        self.assertIn("window_id", source)
        self.assertIn("timeframe", source)


if __name__ == "__main__":
    unittest.main()
