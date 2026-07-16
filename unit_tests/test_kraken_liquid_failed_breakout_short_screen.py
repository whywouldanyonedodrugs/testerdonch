import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import run_kraken_liquid_failed_breakout_short_screen as short
from tools import qlmg_evidence_contracts as contracts


class LiquidFailedBreakoutShortTests(unittest.TestCase):
    def test_manifest_is_frozen_24_and_exit_fanout_deduplicates_to_eight_keys(self):
        manifest = short.frozen_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 8)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())

    def test_short_stop_fill_is_adverse_on_gap(self):
        gap = pd.Series({"open": 110.0, "high": 112.0})
        touch = pd.Series({"open": 99.0, "high": 105.0})
        self.assertEqual(short.stop_fill_short(gap, 105.0), 110.0)
        self.assertEqual(short.stop_fill_short(touch, 105.0), 105.0)

    def test_evaluation_boundary_exit_fails_closed(self):
        frame = pd.DataFrame({
            "entry_ts": ["2025-12-31T20:00:00Z"],
            "exit_ts": ["2025-12-31T23:55:00Z"],
            "exit_reason": ["fixed_72h_or_data_horizon"],
        })
        result = contracts.validate_evaluation_window_intervals(
            frame, window_start="2025-07-01T00:00:00Z", window_end="2026-01-01T00:00:00Z"
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("artificial_horizon" in value for value in result.violations))

    def test_exit_policy_does_not_change_selected_key_hash(self):
        manifest = short.frozen_manifest()
        group = manifest.groupby(["reference_days", "failure_bars", "parent_context"])
        self.assertTrue(all(g.selected_key_policy_hash.nunique() == 1 for _, g in group))

    def test_control_address_ignores_class_label(self):
        row = {"symbol": "PF_XBTUSD", "decision_ts": "2025-01-01T00:00:00Z", "entry_ts": "2025-01-01T00:05:00Z", "initial_stop": 101.0, "risk_denominator": 1.0, "exit_policy": "fixed_72h_comparator", "maximum_exit_ts": "2025-01-04T00:05:00Z"}
        first = short.control_address_hash({**row, "control_class": "a"})
        second = short.control_address_hash({**row, "control_class": "b"})
        self.assertEqual(first, second)

    def test_daily_reference_at_same_close_is_excluded(self):
        four = pd.DataFrame({"decision_ts": pd.to_datetime(["2025-01-02T00:00:00Z", "2025-01-02T04:00:00Z"]), "close": [10.0, 10.0]})
        daily = pd.DataFrame({"daily_source_ts": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"]), "prior_high_20d": [9.0, 11.0]})
        joined = short.with_daily_features(four, daily)
        self.assertEqual(joined.prior_high_20d.tolist(), [9.0, 11.0])

    def test_real_resample_feature_join_preserves_trade_close_name(self):
        ts = pd.date_range("2024-01-01", periods=61 * 288, freq="5min", tz="UTC")
        bars = pd.DataFrame({"ts": ts, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1.0, "mark_close": 10.0})
        four, daily = short.signal_bars(bars)
        joined = short.with_daily_features(four, daily)
        self.assertTrue({"open", "high", "low", "close"}.issubset(joined.columns))
        self.assertFalse(any(column.endswith("_x") for column in joined.columns))

    def test_same_regime_slice_retains_precomputed_control_masks(self):
        pool = pd.DataFrame({"parent_state": ["both_down", "other"], "bearish_reversal": [True, False]})
        same_regime = pool[pool.parent_state.eq("both_down")]
        self.assertEqual(same_regime[same_regime.bearish_reversal].shape[0], 1)

    def test_canonical_sequence_ignores_repeated_above_level_closes(self):
        frame = pd.DataFrame({"close": [11.0, 12.0, 13.0, 9.0], "prior_high_60d": [10.0] * 4})
        self.assertEqual(short.canonical_failure_sequences(frame, "prior_high_60d", 3), [(0, 3, 10.0)])

    def test_canonical_sequence_expires_then_rearms_on_later_breakout(self):
        frame = pd.DataFrame({"close": [11.0, 12.0, 13.0, 14.0, 9.0, 11.0, 9.0], "prior_high_60d": [10.0] * 7})
        self.assertEqual(short.canonical_failure_sequences(frame, "prior_high_60d", 3), [(5, 6, 10.0)])

    def test_canonical_sequence_rearms_only_after_failure_resolution(self):
        frame = pd.DataFrame({"close": [11.0, 12.0, 9.0, 11.0, 9.0], "prior_high_60d": [10.0] * 5})
        self.assertEqual(short.canonical_failure_sequences(frame, "prior_high_60d", 3), [(0, 2, 10.0), (3, 4, 10.0)])

    def test_control_builder_loads_history_once_per_symbol_not_per_candidate(self):
        source = inspect.getsource(short.build_controls)
        self.assertEqual(source.count("runner.load_symbol_bars("), 1)
        self.assertLess(source.index("runner.load_symbol_bars("), source.index("for key in candidates"))
        self.assertIn("control_key_symbol_shards", source)

    def test_compact_bundle_excludes_raw_ledgers_caches_logs_and_process_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text_inputs = {
                "decision_summary.json": "{}\n",
                "contract/failed_breakout_short_contract.md": "contract\n",
            }
            csv_inputs = (
                "manifest/failed_breakout_short_definitions.csv",
                "manifest/pit_panel.csv",
                "audit/exactness_sentinel.csv",
                "economics/definition_summary.csv",
                "economics/cost_funding_attribution.csv",
                "forensics/concentration_and_removal.csv",
                "forensics/parameter_neighborhood.csv",
                "decision/candidate_decisions.csv",
            )
            for relative, content in text_inputs.items():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
            for relative in csv_inputs:
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("status\npass\n")
            event = root / "materialized/event_ledgers/lfbs_v1_001.csv"; event.parent.mkdir(parents=True); event.write_text("event_id\nE1\n")
            outcome_manifest = root / "aggregate_shards/s1/shard_manifest.json"; outcome_manifest.parent.mkdir(parents=True); outcome_manifest.write_text(json.dumps({"shard_id": "s1", "status": "complete", "selected_rows": 1, "outcome_rows": 1}))
            selected_manifest = root / "selected_key_shards/s1/selected_key_manifest.json"; selected_manifest.parent.mkdir(parents=True); selected_manifest.write_text(json.dumps({"shard_id": "s1", "status": "frozen", "rows": 1}))
            cache = root / "selected_key_symbol_shards/PF_XBTUSD.parquet"; cache.parent.mkdir(parents=True); cache.write_bytes(b"raw-cache")
            (root / "main.pid").write_text("1\n")
            log = root / "logs/run.log"; log.parent.mkdir(); log.write_text("log\n")
            comparison = pd.DataFrame([{"definition_id": "lfbs_v1_001", "control_classes": "same_symbol", "cost_mode": "base", "control_economic_address_hash": "h1", "paired_rows": 1, "candidate_mean_R": 1.0, "control_mean_R": 0.0, "unique_address_uplift_R": 1.0}])
            address = pd.DataFrame([{"definition_id": "lfbs_v1_001", "control_economic_address_hash": "h1", "class_labels": "same_symbol", "class_count": 1, "duplicated_address_counted_independently": 0}])

            bundle = short.build_compact_review_bundle(root, comparison, address)
            names = {path.name for path in bundle.iterdir()}
            self.assertEqual(len(names), len(short.COMPACT_REVIEW_FILES) + 1)
            self.assertIn("bundle_manifest.csv", names)
            self.assertFalse(any("selected_key_symbol_shards" in name for name in names))
            self.assertFalse(any("event_ledgers" in name for name in names))
            self.assertFalse(any(name.endswith(".pid") or name.endswith(".log") for name in names))


if __name__ == "__main__": unittest.main()
