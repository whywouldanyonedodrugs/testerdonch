from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools import build_kda01_v2_prerun_freeze as runner
from tools.qlmg_kda01_v2 import (
    FEATURE_EXTENSION_HASH, GENERATOR_HASH, deterministic_episode_id,
    extend_causal_features, generate_parent_episodes_and_events, parent_mask,
    progress_classification,
)


def base_frame(rows: int = 100) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp_utc": ts, "oi_log_change_1h": 0.01, "trade_return_1h": 0.02,
        "mark_return_1h": 0.02, "path_efficiency_1h": 0.60,
        "basis_decimal": 0.001, "basis_level_robust_z": 0.0,
        "basis_level_percentile": 0.50, "basis_level_normalization_valid": True,
        "oi_change_robust_z": 0.0, "oi_change_percentile": 0.50,
        "oi_change_normalization_valid": True, "price_progress_percentile": 0.80,
        "price_progress_normalization_valid": True,
        "eligible": True, "known_lifecycle_mask": True, "trade_coverage": True,
        "mark_coverage": True, "analytics_coverage": True,
        "trade_high": 101.0, "trade_low": 99.0, "trade_close": 100.0,
        "mark_high": 101.0, "mark_low": 99.0, "mark_close": 100.0,
    })
    return frame


def activate_episode(frame: pd.DataFrame, onset: int = 20, direction: int = 1) -> None:
    sign = float(direction)
    frame.loc[onset, ["trade_return_1h", "mark_return_1h"]] = 0.02 * sign
    frame.loc[onset, "basis_decimal"] = 0.001 * sign
    frame.loc[onset, "basis_level_robust_z"] = 2.0 * sign
    frame.loc[onset, "basis_level_percentile"] = 0.95 if direction > 0 else 0.05
    frame.loc[onset, "oi_change_robust_z"] = 2.0
    frame.loc[onset, "oi_change_percentile"] = 0.95
    frame.loc[onset:onset+40, "oi_change_robust_z"] = frame.loc[onset:onset+40, "oi_change_robust_z"].clip(lower=1.1)
    frame.loc[onset:onset+40, "basis_level_robust_z"] = 1.1 * sign
    frame.loc[onset, "basis_level_robust_z"] = 2.0 * sign


class KDA01V2PreRunTests(unittest.TestCase):
    def test_contract_hashes_are_stable(self):
        self.assertEqual(len(FEATURE_EXTENSION_HASH), 64)
        self.assertEqual(len(GENERATOR_HASH), 64)
        self.assertEqual(deterministic_episode_id("PF_XBTUSD", "primary", 1, "2024-01-01T00:00Z"), deterministic_episode_id("PF_XBTUSD", "primary", 1, "2024-01-01T00:00Z"))

    def test_causal_oi_and_price_progress_normalization(self):
        ts = pd.date_range("2023-01-01", periods=80, freq="D", tz="UTC")
        frame = pd.DataFrame({
            "timestamp_utc": ts, "oi_log_change_1h": np.linspace(.01, .08, 80),
            "trade_return_1h": np.linspace(.02, .09, 80), "mark_return_1h": np.linspace(.02, .09, 80), "path_efficiency_1h": .6,
            "basis_decimal": .001, "basis_level_robust_z": 2.1,
            "basis_level_percentile": .96, "basis_level_normalization_valid": True,
            "eligible": True, "known_lifecycle_mask": True, "trade_coverage": True,
            "mark_coverage": True, "analytics_coverage": True,
        })
        result = extend_causal_features(frame)
        self.assertFalse(result.loc[29, "oi_change_normalization_valid"])
        self.assertTrue(result.loc[30, "oi_change_normalization_valid"])
        self.assertFalse(result.loc[59, "price_progress_normalization_valid"])
        self.assertTrue(result.loc[60, "price_progress_normalization_valid"])

    def test_future_rows_cannot_change_prior_scores(self):
        ts = pd.date_range("2023-01-01", periods=80, freq="D", tz="UTC")
        frame = pd.DataFrame({"timestamp_utc":ts,"oi_log_change_1h":np.linspace(.01,.08,80),"trade_return_1h":np.linspace(.02,.09,80),"mark_return_1h":np.linspace(.02,.09,80),"path_efficiency_1h":.6,"basis_decimal":.001,"basis_level_robust_z":2.1,"basis_level_percentile":.96,"basis_level_normalization_valid":True,"eligible":True,"known_lifecycle_mask":True,"trade_coverage":True,"mark_coverage":True,"analytics_coverage":True})
        first = extend_causal_features(frame.iloc[:70])
        frame.loc[70:, "oi_log_change_1h"] = 1e9
        second = extend_causal_features(frame)
        pd.testing.assert_series_equal(first.oi_change_robust_z.reset_index(drop=True), second.oi_change_robust_z.iloc[:70].reset_index(drop=True))

    def test_price_progress_is_absent_outside_preprogress_parent_state(self):
        frame=base_frame(2)
        frame.loc[:,"oi_change_robust_z"]=[2.0,2.0]
        frame.loc[:,"basis_level_robust_z"]=[2.0,2.0]
        frame.loc[:,"basis_decimal"]=[.001,-.001]
        frame.loc[:,"basis_level_percentile"]=[.95,.95]
        from tools.qlmg_kda01_v2 import _preprogress_parent_mask
        self.assertTrue(_preprogress_parent_mask(frame,"primary",1).iloc[0])
        self.assertFalse(_preprogress_parent_mask(frame,"primary",1).iloc[1])

    def test_directional_basis_coherence_and_exact_boundaries(self):
        frame = base_frame(2)
        frame.loc[:, "oi_change_robust_z"] = 2.0
        frame.loc[:, "basis_level_robust_z"] = [2.0, -2.0]
        frame.loc[:, "basis_decimal"] = [0.001, -0.001]
        frame.loc[:, ["trade_return_1h", "mark_return_1h"]] = [[.01,.01],[-.01,-.01]]
        self.assertTrue(parent_mask(frame, "primary", 1).iloc[0])
        self.assertTrue(parent_mask(frame, "primary", -1).iloc[1])
        frame.loc[0, "basis_decimal"] = -0.001
        self.assertFalse(parent_mask(frame, "primary", 1).iloc[0])

    def test_robustness_tail_boundaries(self):
        frame = base_frame(2)
        frame.loc[:, "oi_change_percentile"] = .95
        frame.loc[:, "basis_level_percentile"] = [.95, .05]
        frame.loc[:, "basis_decimal"] = [.001, -.001]
        frame.loc[:, ["trade_return_1h", "mark_return_1h"]] = [[.01,.01],[-.01,-.01]]
        self.assertTrue(parent_mask(frame, "robustness", 1).iloc[0])
        self.assertTrue(parent_mask(frame, "robustness", -1).iloc[1])

    def test_progress_threshold_precedence(self):
        frame = base_frame(3)
        frame.loc[:, "price_progress_percentile"] = [.75,.25,.50]
        frame.loc[:, "path_efficiency_1h"] = [.50,.60,.30]
        self.assertEqual(progress_classification(frame).tolist(), ["efficient_progress","deteriorating_progress","intermediate"])

    def test_nonfinite_path_efficiency_cannot_enter_parent_state(self):
        frame=base_frame(1)
        frame.loc[0,["oi_change_robust_z","basis_level_robust_z"]]=[2.0,2.0]
        frame.loc[0,"path_efficiency_1h"]=np.nan
        self.assertFalse(parent_mask(frame,"primary",1).iloc[0])

    def test_one_efficient_and_one_structural_failure_per_episode(self):
        frame = base_frame(); activate_episode(frame)
        frame.loc[25, "price_progress_percentile"] = .25
        frame.loc[33, ["trade_close", "mark_close"]] = [98.0, 98.0]
        episodes, events = generate_parent_episodes_and_events(frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r")
        primary = events[events.attempt.eq("primary")]
        self.assertEqual(primary.event_type.value_counts().to_dict(), {"efficient_crowding_continuation":1,"completed_structural_failure":1})
        self.assertEqual(episodes[episodes.attempt.eq("primary")].shape[0], 1)

    def test_touch_trade_only_and_sign_flip_do_not_confirm_failure(self):
        for trade_close, mark_close in ((99.0,99.0),(98.0,100.0),(100.0,100.0)):
            frame=base_frame(); activate_episode(frame); frame.loc[25,"price_progress_percentile"]=.25
            frame.loc[33,["trade_close","mark_close"]]=[trade_close,mark_close]
            _,events=generate_parent_episodes_and_events(frame,symbol="PF_TESTUSD",semantic_hash="s",analytics_manifest_hash="a",cohort_hash="c",source_refs="r")
            self.assertFalse(events.event_type.eq("completed_structural_failure").any())

    def test_failure_after_six_hour_deadline_is_excluded(self):
        frame=base_frame(110); activate_episode(frame); frame.loc[25,"price_progress_percentile"]=.25
        frame.loc[94,["trade_close","mark_close"]]=[98.0,98.0]
        _,events=generate_parent_episodes_and_events(frame,symbol="PF_TESTUSD",semantic_hash="s",analytics_manifest_hash="a",cohort_hash="c",source_refs="r")
        self.assertFalse(events.event_type.eq("completed_structural_failure").any())

    def test_episode_ending_before_one_hour_has_no_failure_window(self):
        frame=base_frame(); activate_episode(frame)
        frame.loc[21:, ["oi_change_robust_z", "basis_level_robust_z"]] = 0.0
        frame.loc[22, "price_progress_percentile"] = .25
        frame.loc[33, ["trade_close", "mark_close"]] = [98.0,98.0]
        episodes,events=generate_parent_episodes_and_events(frame,symbol="PF_TESTUSD",semantic_hash="s",analytics_manifest_hash="a",cohort_hash="c",source_refs="r")
        self.assertFalse(episodes[episodes.attempt.eq("primary")].impulse_complete.iloc[0])
        self.assertFalse(events.event_type.eq("completed_structural_failure").any())

    def test_episode_reset_requires_sixty_minutes_without_parent(self):
        frame=base_frame(130); activate_episode(frame,20); activate_episode(frame,80)
        episodes,_=generate_parent_episodes_and_events(frame,symbol="PF_TESTUSD",semantic_hash="s",analytics_manifest_hash="a",cohort_hash="c",source_refs="r")
        self.assertEqual(len(episodes[episodes.attempt.eq("primary")]),2)

    def test_hysteresis_ends_after_thirty_minutes_outside(self):
        frame=base_frame(); activate_episode(frame)
        episodes,_=generate_parent_episodes_and_events(frame,symbol="PF_TESTUSD",semantic_hash="s",analytics_manifest_hash="a",cohort_hash="c",source_refs="r")
        row=episodes[episodes.attempt.eq("primary")].iloc[0]
        self.assertEqual(row.episode_end_ts, frame.timestamp_utc.iloc[66] + pd.Timedelta(minutes=5))

    def test_primary_and_robustness_are_separate_attempts(self):
        frame=base_frame(); activate_episode(frame)
        episodes,events=generate_parent_episodes_and_events(frame,symbol="PF_TESTUSD",semantic_hash="s",analytics_manifest_hash="a",cohort_hash="c",source_refs="r")
        self.assertEqual(set(episodes.attempt),{"primary","robustness"})
        self.assertEqual(set(events.attempt),{"primary","robustness"})
        self.assertEqual(len(set(events.economic_address)),len(events))

    def test_deterministic_replay_and_zero_duplicates(self):
        frame=base_frame(); activate_episode(frame)
        first=generate_parent_episodes_and_events(frame,symbol="PF_TESTUSD",semantic_hash="s",analytics_manifest_hash="a",cohort_hash="c",source_refs="r")
        second=generate_parent_episodes_and_events(frame,symbol="PF_TESTUSD",semantic_hash="s",analytics_manifest_hash="a",cohort_hash="c",source_refs="r")
        pd.testing.assert_frame_equal(first[0],second[0]); pd.testing.assert_frame_equal(first[1],second[1])
        self.assertFalse(first[1].event_id.duplicated().any()); self.assertFalse(first[1].economic_address.duplicated().any())

    def test_pretrain_and_protected_rows_fail_closed(self):
        for timestamp in ("2022-12-31T23:55Z","2026-01-01T00:00Z"):
            frame=base_frame(); frame.loc[0,"timestamp_utc"]=pd.Timestamp(timestamp)
            with self.assertRaisesRegex(ValueError,"non-rankable"):
                extend_causal_features(frame)

    def test_definition_register_never_uses_robustness_to_rescue(self):
        gates=pd.DataFrame([{"branch_id":b,"mechanically_feasible":b==runner.PRIMARY_BRANCHES[0]} for b in runner.PRIMARY_BRANCHES])
        definitions=runner.definition_register(gates)
        self.assertEqual(len(definitions),4)
        self.assertFalse(definitions.can_rescue_primary.any())

    def test_source_has_no_economic_runner_or_outcome_reader(self):
        source=Path(runner.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("outcome_reader",source)
        self.assertNotIn("candidate_return",source)
        self.assertNotIn("forward_return",source)
        self.assertNotIn("calculate_pnl",source)

    def test_atomic_partition_and_bounded_reducer_contract(self):
        source=Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("os.replace(temp, shard)",source)
        self.assertIn("SET memory_limit='1GB'",source)
        self.assertNotIn("all_events",source)

    def test_non_bar_manifest_rows_are_skipped_before_ohlc_read(self):
        source=Path(runner.__file__).read_text(encoding="utf-8")
        schema_guard=source.index("if not set(columns).issubset(schema):")
        parquet_read=source.index("read(columns=columns)")
        self.assertLess(schema_guard,parquet_read)

    def test_zero_count_shards_are_not_passed_to_duckdb_union(self):
        source=Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('if int(manifest["episode_count"]):',source)
        self.assertIn('if int(manifest["event_count"]):',source)

    def test_old_family_overlap_uses_allowlisted_identity_columns(self):
        source=Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('allowed_symbols = ("symbol", "PF_symbol")',source)
        self.assertIn('allowed_times = ("decision_ts", "dominant_bar_close_ts", "impulse_onset_ts", "entry_ts")',source)
        self.assertIn("unsafe old-family identity schema",source)


if __name__ == "__main__":
    unittest.main()
