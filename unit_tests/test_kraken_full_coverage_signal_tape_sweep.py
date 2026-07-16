from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from tools import run_kraken_full_coverage_signal_tape_sweep as sweep


class KrakenFullCoverageSignalTapeSweepTests(unittest.TestCase):
    def test_safety_flags_with_no_prefix_parse_true(self):
        args = sweep.parse_args(["--no-event-sampling", "--no-portfolio-caps", "--all-eligible-events"])
        self.assertTrue(args.no_event_sampling)
        self.assertTrue(args.no_portfolio_caps)
        self.assertTrue(args.all_eligible_events)

    def test_persistent_condition_generates_transition_events_not_true_bars(self):
        n = 500
        bars = pd.DataFrame({
            "ts": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
            "open": range(100, 100 + n),
            "high": range(101, 101 + n),
            "low": range(99, 99 + n),
            "close": range(100, 100 + n),
            "volume": 1.0,
        })
        diag = sweep.semantic_signal_diagnostics(bars, "liquid_continuation", 12, 0.001, "long", hold_bars=24, candidate_id="x")
        self.assertGreater(diag["raw_signal_count"], 100)
        self.assertLess(diag["transition_event_count"], diag["raw_signal_count"] / 10)
        self.assertEqual(diag["contract_event_type"], "trade_episode_contract")
        self.assertEqual(diag["row_semantics"], "trade_episode")
        self.assertGreater(diag["duplicate_entry_suppression_count"], 0)

    def test_tsmom_uses_rebalance_cadence_not_every_5m_true_bar(self):
        n = 576
        bars = pd.DataFrame({
            "ts": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
            "open": [100 + i * 0.05 for i in range(n)],
            "high": [101 + i * 0.05 for i in range(n)],
            "low": [99 + i * 0.05 for i in range(n)],
            "close": [100 + i * 0.05 for i in range(n)],
            "volume": 1.0,
        })
        diag = sweep.semantic_signal_diagnostics(bars, "tsmom", 12, 0.001, "long", hold_bars=288, candidate_id="x")
        self.assertEqual(diag["contract_event_type"], "scheduled_decision_contract")
        self.assertEqual(diag["row_semantics"], "position_interval")
        self.assertLessEqual(diag["events_per_symbol_day"], 1.5)
        self.assertLess(diag["transition_event_count"], diag["raw_signal_count"] / 20)

    def test_canonical_definition_registry_covers_each_rankable_hypothesis(self):
        rank = pd.DataFrame([
            {"hypothesis_id": "H01", "family": "Liquid continuation / Momentum", "archetype": "liquid_continuation", "contract_id": "c1", "contract_source": "test", "priority_wave": "wave_1"},
            {"hypothesis_id": "H03", "family": "Prior-high / ATH / reclaim", "archetype": "prior_high", "contract_id": "c3", "contract_source": "test", "priority_wave": "wave_1"},
        ])
        out = sweep.canonical_definition_rows(rank)
        self.assertFalse(out.empty)
        self.assertEqual(set(out["hypothesis_id"]), {"H01", "H03"})
        self.assertTrue((out.groupby("hypothesis_id")["definition_id"].nunique() >= 7).all())
        self.assertIn("all_context_diagnostic", set(out["regime_variant"]))

    def test_space_filling_is_deterministic_and_rejects_no_touch_fills(self):
        rank = pd.DataFrame([
            {"hypothesis_id": "H02", "family": "Liquid continuation / Momentum", "archetype": "liquid_continuation", "contract_id": "c2", "contract_source": "test", "priority_wave": "wave_1"},
        ])
        a, rej_a = sweep.deterministic_space_fill(rank, 20, 123)
        b, rej_b = sweep.deterministic_space_fill(rank, 20, 123)
        self.assertEqual(a["definition_id"].tolist(), b["definition_id"].tolist())
        self.assertEqual(len(a), 20)
        self.assertIsInstance(rej_a, pd.DataFrame)
        self.assertIsInstance(rej_b, pd.DataFrame)

    def test_candidate_registry_fans_out_definitions_to_all_symbols(self):
        defs = pd.DataFrame([{
            "definition_id": "d1",
            "definition_kind": "plain_baseline",
            "hypothesis_id": "H01",
            "family": "Liquid continuation / Momentum",
            "contract_id": "c1",
            "contract_source": "test",
            "archetype": "liquid_continuation",
            "side": "long",
            "lookback_bars": 12,
            "hold_bars": 6,
            "stop_bps": 100,
            "threshold": 0.001,
        }])
        reg = sweep.generate_candidate_registry(defs, ["PF_A", "PF_B", "PF_C"], budget=1, seed=1)
        self.assertEqual(len(reg), 3)
        self.assertEqual(reg["candidate_id"].nunique(), 1)
        self.assertEqual(set(reg["symbol"]), {"PF_A", "PF_B", "PF_C"})
        self.assertTrue((reg["symbol_fanout"] == 3).all())

    def test_replay_candidates_writes_full_coverage_without_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data/parquet/historical_trade_candles_5m/PF_XBTUSD"
            mark = root / "data/parquet/historical_mark_candles_5m/PF_XBTUSD"
            fund = root / "data/parquet/funding"
            data.mkdir(parents=True)
            mark.mkdir(parents=True)
            fund.mkdir(parents=True)
            bars = pd.DataFrame({
                "time": (pd.date_range("2025-01-01", periods=240, freq="5min", tz="UTC").view("int64") // 1_000_000).astype("int64"),
                "open": [100 + i * 0.1 for i in range(240)],
                "high": [100.5 + i * 0.1 for i in range(240)],
                "low": [99.5 + i * 0.1 for i in range(240)],
                "close": [100.2 + i * 0.1 for i in range(240)],
                "volume": 1000,
                "venue_symbol": "PF_XBTUSD",
                "source_url": "fixture",
                "chunk_start_utc": "2025-01-01",
                "chunk_end_utc": "2025-01-02",
            })
            bars.to_parquet(data / "part.parquet", index=False)
            bars.to_parquet(mark / "part.parquet", index=False)
            ctx = SimpleNamespace(
                args=SimpleNamespace(kraken_data_root=str(root / "data"), no_event_sampling=True),
                start=pd.Timestamp("2025-01-01T00:00:00Z"),
                end=pd.Timestamp("2025-01-02T00:00:00Z"),
            )
            candidates = pd.DataFrame([{
                "candidate_id": "c1", "definition_id": "d1", "hypothesis_id": "H01", "family": "Liquid continuation / Momentum",
                "contract_id": "ct", "contract_source": "test", "symbol": "PF_XBTUSD", "side": "long", "archetype": "liquid_continuation",
                "lookback_bars": 12, "hold_bars": 6, "stop_bps": 100, "threshold": 0.001,
                "entry_template": "close", "exit_template": "fixed_hold", "stop_template": "fixed_bps", "regime_activation": "all_context_diagnostic",
                "data_cap": "none", "source_data_hash": "h",
            }])
            events = sweep.replay_candidates(ctx, candidates, max_events_per_candidate=None, output_path=root / "events.parquet", coverage_path=root / "coverage.csv")
            cov = pd.read_csv(root / "coverage.csv")
            self.assertFalse(events.empty)
            self.assertFalse(cov["event_sampling_used"].astype(bool).any())
            self.assertEqual(float(cov.iloc[0]["coverage_ratio"]), 1.0)
            self.assertIn("row_semantics", events.columns)
            self.assertEqual(set(events["event_semantics_version"]), {sweep.EVENT_SEMANTICS_VERSION})

    def test_stream_replay_writes_partitioned_ledger_progress_and_candidate_symbol_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data/parquet/historical_trade_candles_5m/PF_XBTUSD"
            mark = root / "data/parquet/historical_mark_candles_5m/PF_XBTUSD"
            fund = root / "data/parquet/funding"
            data.mkdir(parents=True)
            mark.mkdir(parents=True)
            fund.mkdir(parents=True)
            bars = pd.DataFrame({
                "time": (pd.date_range("2025-01-01", periods=240, freq="5min", tz="UTC").view("int64") // 1_000_000).astype("int64"),
                "open": [100 + i * 0.1 for i in range(240)],
                "high": [100.5 + i * 0.1 for i in range(240)],
                "low": [99.5 + i * 0.1 for i in range(240)],
                "close": [100.2 + i * 0.1 for i in range(240)],
                "volume": 1000,
                "venue_symbol": "PF_XBTUSD",
                "source_url": "fixture",
                "chunk_start_utc": "2025-01-01",
                "chunk_end_utc": "2025-01-02",
            })
            bars.to_parquet(data / "part.parquet", index=False)
            bars.to_parquet(mark / "part.parquet", index=False)
            run_root = root / "run"
            run_root.mkdir()
            ctx = SimpleNamespace(
                run_root=run_root,
                args=SimpleNamespace(kraken_data_root=str(root / "data"), no_event_sampling=True),
                start=pd.Timestamp("2025-01-01T00:00:00Z"),
                end=pd.Timestamp("2025-01-02T00:00:00Z"),
            )
            candidates = pd.DataFrame([{
                "candidate_id": "c1", "candidate_symbol_id": "c1__PF_XBTUSD", "definition_id": "d1", "hypothesis_id": "H01",
                "family": "Liquid continuation / Momentum", "contract_id": "ct", "contract_source": "test", "symbol": "PF_XBTUSD",
                "side": "long", "archetype": "liquid_continuation", "lookback_bars": 12, "hold_bars": 6,
                "stop_bps": 100, "threshold": 0.001, "entry_template": "close", "exit_template": "fixed_hold",
                "stop_template": "fixed_bps", "regime_activation": "all_context_diagnostic", "data_cap": "none",
                "source_data_hash": "h",
            }])
            wave = run_root / "waves/wave_test"
            cov, summary, rows = sweep.stream_replay_wave(ctx, wave, candidates, coverage_path=wave / "coverage.csv")
            self.assertGreater(rows, 0)
            self.assertFalse(cov["event_sampling_used"].astype(bool).any())
            self.assertEqual(float(cov.iloc[0]["coverage_ratio"]), 1.0)
            self.assertGreater(int(cov.iloc[0]["duplicate_entry_suppression_count"]), 0)
            self.assertEqual(cov.iloc[0]["event_semantics_version"], sweep.EVENT_SEMANTICS_VERSION)
            self.assertTrue((wave / "event_ledger_parts").exists())
            self.assertTrue((wave / "event_ledger_manifest.csv").exists())
            self.assertTrue((wave / "event_ledger_progress.jsonl").exists())
            self.assertTrue((run_root / "resources/memory_progress.jsonl").exists())
            self.assertFalse(summary.empty)

    def test_memory_guard_fails_closed_above_10gb_and_writes_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = SimpleNamespace(run_root=root)
            with mock.patch.object(sweep, "process_rss_gb", return_value=10.5):
                with self.assertRaises(MemoryError):
                    sweep.memory_guard(ctx, wave_dir=None, phase="fixture")
            report = (root / "resources/memory_guard_report.md").read_text()
            self.assertIn("fail_closed_rss_gt_10gb", report)

    def test_event_from_signal_uses_first_adverse_stop_without_row_iteration(self):
        bars = pd.DataFrame({
            "ts": pd.date_range("2025-01-01", periods=8, freq="5min", tz="UTC"),
            "open": [100, 100, 100, 100, 100, 100, 100, 100],
            "high": [101, 101, 101, 101, 101, 101, 101, 101],
            "low": [99, 99, 99.5, 98, 97, 96, 95, 94],
            "close": [100, 100, 100, 99, 98, 97, 96, 95],
            "volume": 1.0,
            "mark_close": [100] * 8,
            "mark_high": [101] * 8,
            "mark_low": [99] * 8,
            "symbol": "PF_XBTUSD",
        })
        candidate = {
            "candidate_id": "c_stop",
            "definition_id": "d_stop",
            "hypothesis_id": "H01",
            "family": "Liquid continuation / Momentum",
            "symbol": "PF_XBTUSD",
            "side": "long",
            "hold_bars": 5,
            "stop_bps": 100,
            "archetype": "liquid_continuation",
            "entry_template": "close",
            "exit_template": "fixed_hold",
            "stop_template": "fixed_bps",
            "regime_activation": "all_context_diagnostic",
        }
        ev = sweep.event_from_signal(candidate, bars, pd.DataFrame(), idx=1, seq=0)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["exit_reason"], "stop_5m_adverse")
        self.assertEqual(str(ev["exit_ts"]), "2025-01-01 00:15:00+00:00")
        self.assertTrue(ev["same_bar_ambiguity_flag"])
        self.assertEqual(ev["row_semantics"], "trade_episode")
        self.assertEqual(ev["event_semantics_version"], sweep.EVENT_SEMANTICS_VERSION)

    def test_non_trade_row_semantics_blank_trade_only_metrics(self):
        events = pd.DataFrame({
            "candidate_id": ["c1", "c1"],
            "hypothesis_id": ["H02", "H02"],
            "family": ["TSMOM", "TSMOM"],
            "side": ["long", "long"],
            "row_semantics": ["position_interval", "position_interval"],
            "contract_event_type": ["scheduled_decision_contract", "scheduled_decision_contract"],
            "event_semantics_version": [sweep.EVENT_SEMANTICS_VERSION, sweep.EVENT_SEMANTICS_VERSION],
            "signal_template": ["tsmom", "tsmom"],
            "entry_template": ["scheduled", "scheduled"],
            "exit_template": ["rebalance", "rebalance"],
            "stop_template": ["vol", "vol"],
            "regime_activation": ["all", "all"],
            "decision_ts": pd.date_range("2025-01-01", periods=2, freq="1D", tz="UTC"),
            "symbol": ["PF_XBTUSD", "PF_XBTUSD"],
            "net_R": [1.0, -0.2],
            "funding_proxy_used": [False, False],
            "mark_proxy_used": [False, False],
        })
        summary = sweep.summarize_events(events)
        self.assertEqual(summary.iloc[0]["trade_metric_applicability"], "not_applicable")
        self.assertTrue(pd.isna(summary.iloc[0]["PF"]))
        self.assertTrue(pd.isna(summary.iloc[0]["win_rate"]))

    def test_controls_match_row_semantics_and_contract_type(self):
        ts = pd.date_range("2025-01-01", periods=6, freq="1h", tz="UTC")
        events = pd.DataFrame({
            "event_id": [f"e{i}" for i in range(6)],
            "candidate_id": ["a", "a", "b", "b", "c", "c"],
            "contract_id": ["cta", "cta", "ctb", "ctb", "ctc", "ctc"],
            "family": ["Liquid continuation / Momentum"] * 4 + ["TSMOM"] * 2,
            "signal_template": ["liquid_continuation"] * 4 + ["tsmom"] * 2,
            "symbol": ["PF_XBTUSD", "PF_ETHUSD", "PF_BTCUSD", "PF_SOLUSD", "PF_XBTUSD", "PF_ETHUSD"],
            "decision_ts": ts,
            "entry_ts": ts,
            "exit_ts": ts + pd.Timedelta(minutes=30),
            "net_R": [1, 1, -0.2, 0.1, 0.4, -0.1],
            "risk_bps_used": [100] * 6,
            "funding_boundary_crossed": [False] * 6,
            "row_semantics": ["trade_episode"] * 4 + ["position_interval"] * 2,
            "contract_event_type": ["trade_episode_contract"] * 4 + ["scheduled_decision_contract"] * 2,
        })
        controls, summary = sweep.build_controls(events, 1, 7, candidate_ids=["a"], ledger_limit_per_candidate=10)
        self.assertFalse(controls.empty)
        self.assertTrue(controls["row_semantics"].eq(controls["control_row_semantics"]).all())
        self.assertTrue(controls["contract_event_type"].eq(controls["control_contract_event_type"]).all())
        self.assertFalse(summary.empty)
        self.assertTrue(summary["control_row_semantics_matched"].astype(bool).all())

    def test_bundle_archive_does_not_prune_when_verification_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wave = root / "waves/wave_1"
            (wave / "controls").mkdir(parents=True)
            (wave / "event_ledger.parquet").write_bytes(b"event")
            (wave / "controls/control_ledger.parquet").write_bytes(b"control")
            ctx = SimpleNamespace(
                run_root=root,
                args=SimpleNamespace(archive_completed_waves=True, remote_archive_enabled=False, remote_name="missing", remote_archive_path="", prune_local_large_wave_artifacts_after_upload=True, smoke=False),
            )
            status = sweep.bundle_and_archive_wave(ctx, wave)
            self.assertEqual(status["upload_status"], "not_attempted")
            self.assertTrue((wave / "event_ledger.parquet").exists())
            self.assertTrue((wave / "controls/control_ledger.parquet").exists())

    def test_tmux_wrapper_requires_launch_and_uses_new_runner(self):
        txt = Path("tools/run_kraken_full_coverage_signal_tape_sweep_tmux.sh").read_text()
        self.assertIn("refusing to launch tmux without --launch-tmux", txt)
        self.assertIn("run_kraken_full_coverage_signal_tape_sweep.py", txt)

    def test_non_resume_init_clears_generated_run_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            stale = root / "waves/wave_1/event_ledger.parquet"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")
            done = root / "stage_status/old.done"
            done.parent.mkdir(parents=True)
            done.write_text("stale")
            args = sweep.parse_args(["--run-root", str(root), "--disable-telegram"])
            sweep.init_context(args)
            self.assertFalse(stale.exists())
            self.assertFalse(done.exists())
            self.assertTrue((root / "stage_status").exists())


if __name__ == "__main__":
    unittest.main()
