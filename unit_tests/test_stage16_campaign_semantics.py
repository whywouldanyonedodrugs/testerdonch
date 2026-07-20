import copy
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.qlmg_stage16_campaign import (
    OutcomeReadSpy, Stage16ContractError, assign_right_closed_bin,
    build_inner_folds, build_translation_registry, canonical_bytes, canonical_sha256, file_sha256,
    collapse_edges, deterministic_beam, directional_breadth, dominates, economic_address, funding_net_bps,
    kdx_episode_trace, quantile_type7, resolve_fixed_execution, side_for_kda02b,
    side_for_kda02c, synthetic_canary, synthetic_metrics, validate_episode_duration, validate_translation_registry,
)
from tools.validate_stage16_campaign_packet import validate


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_16_complete_campaign_semantics_git_cleanup_20260720_v1"


class Stage16SemanticsTests(unittest.TestCase):
    def test_outcome_read_spy_fails_closed(self):
        spy = OutcomeReadSpy()
        self.assertEqual(spy.read("synthetic/features", ["x"], [{"x": 1}]), [{"x": 1}])
        for path, columns in (("synthetic/forward_returns", ["x"]), ("synthetic/features", ["pnl"])):
            with self.assertRaises(Stage16ContractError):
                spy.read(path, columns, [])

    def test_type7_bins_are_deterministic_and_fail_closed(self):
        values = list(range(11))
        self.assertEqual(quantile_type7(values, .5), 5)
        edges = collapse_edges([quantile_type7(values, p) for p in (0, .2, .4, .6, .8, 1)])
        self.assertEqual(edges, [0, 2, 4, 6, 8, 10])
        self.assertEqual(assign_right_closed_bin(2, edges), 0)
        self.assertEqual(assign_right_closed_bin(2.1, edges), 1)
        with self.assertRaises(Stage16ContractError): collapse_edges([1, 1, 1])
        with self.assertRaises(Stage16ContractError): assign_right_closed_bin(math.nan, edges)

    def test_inner_folds_latest_six_complete_months(self):
        early = build_inner_folds("2023-04-01T00:00:00Z", "2023-09-30T18:00:00Z")
        self.assertEqual([row["inner_fold_id"] for row in early], ["M_202305", "M_202306", "M_202307", "M_202308"])
        later = build_inner_folds("2023-04-01T00:00:00Z", "2024-09-30T18:00:00Z")
        self.assertEqual(len(later), 6)
        self.assertEqual(later[-1]["inner_fold_id"], "M_202408")
        self.assertTrue(all(row["purge_embargo_hours"] >= row["maximum_horizon_hours"] + row["maximum_episode_overlap_hours"] for row in later))
        validate_episode_duration("2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z")
        with self.assertRaises(Stage16ContractError):
            validate_episode_duration("2024-01-01T00:00:00Z", "2024-01-01T06:00:01Z")

    def test_all_translations_complete_unique_and_stable(self):
        registry = build_translation_registry()
        validate_translation_registry(registry)
        self.assertEqual(registry["cells_by_family"], {"KDA02B": 96, "KDA02C": 48, "KDX01": 42})
        self.assertEqual(canonical_sha256(registry), canonical_sha256(build_translation_registry()))
        bad = copy.deepcopy(registry); del bad["cells"][0]["exit"]
        with self.assertRaises(Stage16ContractError): validate_translation_registry(bad)
        changed = copy.deepcopy(registry)
        changed["cells"][0]["feature_contract"]["exact_thresholds"]["oi"] = "changed"
        with self.assertRaises(Stage16ContractError): validate_translation_registry(changed)
        retired = registry["removed_from_executable_registry"]["attempts"]
        self.assertEqual(len(retired), 42)
        self.assertTrue(all(row["can_enter_translation_or_beam"] is False for row in retired))

    def test_side_and_native_symbol_identities(self):
        self.assertEqual(side_for_kda02b("continuation", 2, 1), "long")
        self.assertEqual(side_for_kda02b("continuation", -2, -1), "short")
        self.assertEqual(side_for_kda02b("reversal", 2, 1), "short")
        self.assertEqual(side_for_kda02b("reversal", -2, -1), "long")
        self.assertIsNone(side_for_kda02b("continuation", 2, -1))
        self.assertEqual(side_for_kda02c("negative"), "long")
        self.assertEqual(side_for_kda02c("positive"), "short")
        registry = build_translation_registry()
        c = next(row for row in registry["cells"] if row["family"] == "KDA02C")
        self.assertIn("underlying frozen completed-purge", c["instrument_mapping"]["rule"])
        x = next(row for row in registry["cells"] if row["family"] == "KDX01")
        self.assertIn("long", x["side_mapping"]["rule"])

    def test_KDA02C_windows_use_parent_direction_and_distinct_trailing_intervals(self):
        events = [
            {"base_event_id": "a", "decision_source_ts": "2024-01-01T00:00:00Z", "parent_direction": -1, "purge_identity": "primary_z2"},
            {"base_event_id": "b", "decision_source_ts": "2024-01-01T00:10:00Z", "parent_direction": -1, "purge_identity": "primary_z2"},
            {"base_event_id": "c", "decision_source_ts": "2024-01-01T00:25:00Z", "parent_direction": -1, "purge_identity": "primary_z2"},
            {"base_event_id": "wrong-side", "decision_source_ts": "2024-01-01T00:29:00Z", "parent_direction": 1, "purge_identity": "primary_z2"},
        ]
        expected = {5: 0, 15: 1, 30: 2, 60: 3}
        for window, count in expected.items():
            result = directional_breadth(events, "2024-01-01T00:30:00Z", window, "negative", "primary_z2", 100)
            self.assertEqual(result["directional_onset_count"], count)
            self.assertEqual(result["directional_onset_share"], count/100)

    def test_KDX_onset_delayed_components_flicker_reclaim_and_reset(self):
        rows = [
            {"source_close_ts": "2024-01-01T00:00:00Z", "price": True, "oi": False, "completed_trade_mark_reclaim": False},
            {"source_close_ts": "2024-01-01T00:05:00Z", "price": True, "oi": True, "completed_trade_mark_reclaim": False},
            {"source_close_ts": "2024-01-01T00:10:00Z", "price": False, "oi": True, "completed_trade_mark_reclaim": False},
            {"source_close_ts": "2024-01-01T00:15:00Z", "price": True, "oi": True, "completed_trade_mark_reclaim": True},
            {"source_close_ts": "2024-01-01T00:20:00Z", "price": True, "oi": True, "completed_trade_mark_reclaim": False},
            {"source_close_ts": "2024-01-01T00:25:00Z", "price": False, "oi": True, "completed_trade_mark_reclaim": False},
            {"source_close_ts": "2024-01-01T00:30:00Z", "price": True, "oi": True, "completed_trade_mark_reclaim": False},
        ]
        self.assertEqual(kdx_episode_trace(rows, ["price", "oi"]), [
            {"event": "onset", "source_close_ts": "2024-01-01T00:05:00Z"},
            {"event": "completed_reclaim", "source_close_ts": "2024-01-01T00:15:00Z"},
            {"event": "onset", "source_close_ts": "2024-01-01T00:30:00Z"},
        ])

    def test_execution_delay_fold_and_protected_boundaries(self):
        resolved = resolve_fixed_execution("2024-01-01T00:00:00Z", "1h", ["2024-01-01T00:05:00Z", "2024-01-01T01:05:00Z"], "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z")
        self.assertEqual(resolved["entry_ts"], "2024-01-01T00:05:00Z")
        with self.assertRaises(Stage16ContractError):
            resolve_fixed_execution("2024-01-01T00:00:00Z", "1h", ["2024-01-01T00:15:00Z", "2024-01-01T01:05:00Z"], "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z")
        with self.assertRaises(Stage16ContractError):
            resolve_fixed_execution("2025-12-31T23:00:00Z", "1h", ["2025-12-31T23:05:00Z", "2026-01-01T00:05:00Z"], "2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z")

    def test_economic_address_stability_and_identity_sensitivity(self):
        address = economic_address("a" * 64, "EVENT1", "PF_XBTUSD", "long", "2024-01-01T00:00:00Z", "F1")
        self.assertEqual(address, economic_address("a" * 64, "EVENT1", "PF_XBTUSD", "long", "2024-01-01T00:00:00Z", "F1"))
        self.assertNotEqual(address, economic_address("a" * 64, "EVENT2", "PF_XBTUSD", "long", "2024-01-01T00:00:00Z", "F1"))
        self.assertNotEqual(address, economic_address("a" * 64, "EVENT1", "PF_XBTUSD", "short", "2024-01-01T00:00:00Z", "F1"))

    def test_pareto_missing_is_worst_and_beam_is_deterministic(self):
        common = {"median_inner_fold_base_net_mean_bps": 1, "p20_inner_fold_base_net_mean_bps": 1,
                  "aggregate_base_net_mean_bps": 1, "cluster_bootstrap_lower_bound_bps": 1,
                  "left_tail_utility_bps": 1, "opportunity_frequency_per_30d": 1,
                  "execution_margin_bps": 1, "complexity": 1,
                  "market_day_returns": {"d1": 1, "d2": 2, "d3": 3}, "integrity_pass": True,
                  "accepted_trade_count": 30, "independent_market_day_clusters": 20,
                  "independent_utc_hour_clusters": 20}
        a = dict(common, canonical_translation_id="A")
        b = dict(common, canonical_translation_id="B", aggregate_base_net_mean_bps=None)
        self.assertTrue(dominates(a, b))
        candidates = [dict(common, canonical_translation_id=value) for value in ("B", "A")]
        self.assertEqual([row["canonical_translation_id"] for row in deterministic_beam(candidates)], ["A", "B"])
        invalid = dict(common, canonical_translation_id="X", aggregate_base_net_mean_bps=math.nan)
        self.assertEqual(deterministic_beam([invalid]), [])

    def test_funding_sign_adverse_imputation_and_missingness(self):
        boundary = [{"funding_rate": .001, "boundary_notional_over_entry_notional": 1}]
        self.assertEqual(funding_net_bps(20, "long", boundary, []), -4)
        self.assertEqual(funding_net_bps(20, "short", boundary, []), 16)
        self.assertEqual(funding_net_bps(20, "long", [], [10]), -26)
        self.assertEqual(funding_net_bps(20, "long", [], [-10], mode="stress"), -76)
        with self.assertRaises(Stage16ContractError): funding_net_bps(20, "long", [], [math.nan])

    def test_synthetic_end_to_end_canary(self):
        result = synthetic_canary()
        self.assertTrue(result["read_spy_forward_rejected"])
        self.assertEqual(result["registered_cells"], 186)
        self.assertEqual(result["KDA02B_sides"], ["long", "short", None])
        self.assertEqual(result["KDA02C_identity"], {"symbol": "PF_SOLUSD", "side": "long"})
        self.assertEqual(result["KDX01_identity"]["side"], "long")
        self.assertTrue(result["metrics_complete"])
        self.assertIn("candidate_beam_high_correlation", sum(result["beam_tags"], []))
        self.assertFalse(result["economic_outputs_computed"])

    def test_inner_fold_metric_uses_equal_days_not_trade_weight(self):
        values = [("F1", "d1", 0), ("F1", "d1", 10), ("F1", "d2", 20), ("F2", "d3", 30), ("F3", "d4", 40)]
        rows = [{"event_id": f"e{i}", "day": day, "utc_hour": f"h{i}", "symbol": "PF_XBTUSD", "year": "2024", "inner_fold": fold, "base_net_bps": value, "stress_net_bps": value-2, "holding_seconds": 60} for i, (fold, day, value) in enumerate(values)]
        result = synthetic_metrics(rows, eligible_calendar_days=10, eligible_symbols=1, eligible_interval_seconds=10000, complexity=2)
        self.assertEqual(result["median_inner_fold_base_net_mean_bps"], 30)
        self.assertEqual(result["p20_inner_fold_base_net_mean_bps"], 19.5)

    def test_packet_replay_and_launch_validator(self):
        authority_snapshot = {path: path.read_bytes() for path in ARCHIVE.iterdir() if path.is_file()}
        try:
            with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
                snapshots = []
                for target in (one, two):
                    subprocess.run(["python3", "tools/build_stage16_campaign_packet.py", "--implementation-commit", "TEST"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
                    snapshots.append({name: (ARCHIVE / name).read_bytes() for name in ["CAMPAIGN_MANIFEST.json", "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json", "ECONOMIC_TRANSLATION_REGISTRY.json"]})
                self.assertEqual(snapshots[0], snapshots[1])
        finally:
            for path, payload in authority_snapshot.items():
                path.write_bytes(payload)
        result = validate(ARCHIVE)
        self.assertTrue(result["packet_semantics_complete"])
        self.assertTrue(result["campaign_engine_can_execute_without_discretion"])
        self.assertTrue(result["external_human_approval_still_required"])

    def test_validator_rejects_hash_drift_and_self_authorization(self):
        validate(ARCHIVE)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ARCHIVE.iterdir():
                if path.is_file(): (root / path.name).write_bytes(path.read_bytes())
            packet_path = root / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json"
            packet = json.loads(packet_path.read_text()); packet["economic_run_authorized"] = True
            packet_path.write_text(json.dumps(packet))
            with self.assertRaises(Stage16ContractError): validate(root)

    def test_validator_rejects_semantic_mutations_even_when_hashes_rebound(self):
        mutations = [
            ("INNER_FOLD_MAP.json", lambda x: x["outer_folds"][0].update(purge_interval_hours=6)),
            ("DEVELOPMENT_METRIC_CONTRACT.json", lambda x: x["metrics"].pop("execution_margin_bps")),
            ("ECONOMIC_TRANSLATION_REGISTRY.json", lambda x: x["cells"][144]["feature_contract"]["causal_reclaim"].pop("reference_timestamp")),
            ("ECONOMIC_TRANSLATION_REGISTRY.json", lambda x: x["removed_from_executable_registry"]["attempts"][0].update(can_enter_translation_or_beam=True)),
            ("FUNDING_COST_AND_COVERAGE_CONTRACT.json", lambda x: x.pop("primary_imputed_boundary_bps")),
            ("TELEGRAM_AND_SUPERVISION_CONTRACT.json", lambda x: x["later_launch"].update(heartbeat_minutes=31)),
            ("RESOURCE_PROJECTION.json", lambda x: x.pop("max_memory_bytes")),
        ]
        for filename, mutate in mutations:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for path in ARCHIVE.iterdir():
                    if path.is_file(): (root / path.name).write_bytes(path.read_bytes())
                target = root / filename; value = json.loads(target.read_text()); mutate(value); target.write_bytes(canonical_bytes(value))
                manifest_path = root / "CAMPAIGN_MANIFEST.json"; manifest = json.loads(manifest_path.read_text())
                manifest["dependency_file_sha256"][filename] = file_sha256(target)
                manifest["dependency_canonical_sha256"][filename] = canonical_sha256(value)
                manifest_path.write_bytes(canonical_bytes(manifest))
                packet_path = root / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json"; packet = json.loads(packet_path.read_text())
                packet["dependency_file_sha256"][filename] = file_sha256(target)
                packet["dependency_canonical_sha256"][filename] = canonical_sha256(value)
                packet["campaign_manifest_file_sha256"] = file_sha256(manifest_path)
                packet["campaign_manifest_canonical_sha256"] = canonical_sha256(manifest)
                payload = {key: value for key, value in packet.items() if key != "packet_payload_canonical_sha256"}
                packet["packet_payload_canonical_sha256"] = canonical_sha256(payload)
                packet_path.write_bytes(canonical_bytes(packet))
                with self.assertRaises(Stage16ContractError): validate(root)


if __name__ == "__main__":
    unittest.main()
