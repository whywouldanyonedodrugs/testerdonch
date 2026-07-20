import tempfile
import unittest
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

from tools.qlmg_stage20_campaign import (
    Stage20Error, atomic_json, choose_beam,
    kda02b_symbol_events, kdx_symbol_events, merge_kda02c_symbol_partitions,
    nonoverlap, registered_rank_edges,
)
from tools.stage20_phase2_5_canary import run as run_phase2_5_canary
from tools.qlmg_stage20_launch_gates import validate_gate, write_gate
from tools.run_stage20_economic_campaign import (
    _cell_metrics, _score_job, maybe_release_health,
    write_attempt_and_multiplicity_registries,
)


class Stage20CampaignTests(unittest.TestCase):
    def test_kda02c_rows_remain_in_native_symbol_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{"symbol": "PF_X", "family": "KDA02B", "value": 1}]).to_parquet(
                root / "PF_X.parquet", index=False
            )
            files = [{"symbol": "PF_X", "path": str(root / "PF_X.parquet"),
                      "rows": 1, "bytes": 0, "sha256": "stale"}]
            additions = pd.DataFrame([{"symbol": "PF_X", "family": "KDA02C", "value": 2}])
            merge_kda02c_symbol_partitions(additions, root, files)
            merged = pd.read_parquet(root / "PF_X.parquet")
            self.assertEqual(merged.family.tolist(), ["KDA02B", "KDA02C"])
            self.assertEqual(files[0]["rows"], 2)
            with self.assertRaises(Stage20Error):
                merge_kda02c_symbol_partitions(
                    pd.DataFrame([{"symbol": "PF_Y", "family": "KDA02C", "value": 3}]), root, files
                )

    def test_phase2_5_canary_is_synthetic_and_deterministic(self):
        result = run_phase2_5_canary()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["synthetic_only"])
        self.assertEqual(result["real_economic_outcomes_computed"], 0)
        self.assertTrue(result["bounded_lazy_scheduler"])
        self.assertTrue(result["real_funding_integration"])
        self.assertTrue(result["idempotent_recovery"])
        self.assertTrue(result["graceful_pre_submission_bound_stop"])

    def test_attempt_multiplicity_reconciles_exact_approved_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_attempt_and_multiplicity_registries(Path(directory))
            self.assertEqual(result["executable_attempts"], 186)
            self.assertEqual(result["inherited_non_executable_KDX_attempts"], 42)
            self.assertEqual(result["programme_attempts"], 228)

    def test_outer_scoring_fails_before_atomic_freeze(self):
        with self.assertRaisesRegex(Stage20Error, "before atomic freeze"):
            _score_job({"symbol": "PF_SYNTH", "model_id": "Q_2024Q1", "family": "KDA02B"})

    def test_bound_gate_rejects_status_only_and_artifact_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "reviewed.txt"
            artifact.write_text("reviewed\n")
            gate = root / "gate.json"
            atomic_json(gate, {"status": "pass"})
            with self.assertRaises(Stage20Error):
                validate_gate(gate, "synthetic_supervisor_canary")
            write_gate(gate, "synthetic_supervisor_canary", [artifact], {"canary": "pass"})
            self.assertEqual(validate_gate(gate, "synthetic_supervisor_canary")["status"], "pass")
            artifact.write_text("drifted\n")
            with self.assertRaisesRegex(Stage20Error, "artifact drift"):
                validate_gate(gate, "synthetic_supervisor_canary")

    def test_empty_inner_fold_is_preserved_and_blocks_stability_ranking(self):
        entry = pd.date_range("2024-01-01T00:00:00Z", periods=30, freq="24h")
        frame = pd.DataFrame({
            "cell_id": "C", "source_model_id": "I_PRESENT", "symbol": "PF_X",
            "entry_ts": entry, "exit_ts": entry + pd.Timedelta(hours=1),
            "base_net_bps": 5.0, "stress_net_bps": 3.0,
            "base_net_alignment_start_bps": 5.0,
            "base_net_alignment_end_bps": 5.0,
        })
        row = _cell_metrics(frame, [{"cell_id": "C", "family": "KDA02B",
                                     "canonical_translation_id": "T", "complexity": 1}],
                            ["I_PRESENT", "I_EMPTY"], 30, 30 * 86400)[0]
        self.assertEqual(row["unavailable_inner_fold_count"], 1)
        self.assertEqual(row["inner_fold_observations"][1]["status"], "empty_unavailable")
        self.assertTrue(pd.isna(row["p20_inner_fold_base_net_mean_bps"]))
        self.assertEqual(choose_beam([row]), [])

    def test_health_release_requires_heartbeat_and_reconciled_real_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "cell.parquet"
            pd.DataFrame({"x": [1]}).to_parquet(artifact, index=False)
            from tools.qlmg_stage20_campaign import file_sha256
            state = {"generation": 1, "health_release_status": "pending",
                     "first_scheduled_heartbeat_delivered": False,
                     "first_reconciled_real_cell": {
                         "cell_id": "C", "job_id": "J",
                         "files": [{"path": str(artifact), "sha256": file_sha256(artifact)}],
                     }}
            atomic_json(root / "CAMPAIGN_STATE.json", state)
            maybe_release_health(Namespace(run_root=root), state)
            self.assertFalse((root / "HEALTH_RELEASE.json").exists())
            atomic_json(root / "HEARTBEAT.json", {"status": "healthy"})
            state["first_scheduled_heartbeat_delivered"] = True
            maybe_release_health(Namespace(run_root=root), state)
            self.assertEqual(json.loads((root / "HEALTH_RELEASE.json").read_text())["status"], "pass")

    def test_stage20_entrypoints_do_not_require_ambient_pythonpath(self):
        scripts = ["stage20_phase2_5_canary.py", "validate_stage20_event_replay.py",
                   "validate_stage20_telegram.py", "run_stage20_economic_campaign.py"]
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            for name in scripts:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).parents[1] / "tools" / name), "--help"],
                    cwd=directory, env=environment, capture_output=True, text=True,
                )
                self.assertEqual(completed.returncode, 0, msg=f"{name}: {completed.stderr}")

    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_json(path, {"generation": 1})
            atomic_json(path, {"generation": 2})
            self.assertEqual(path.read_text(), '{\n  "generation": 2\n}\n')

    def test_registered_rank_edges_are_type7_and_require_three_unique_bins(self):
        edges = registered_rank_edges(pd.Series(range(11)))
        self.assertEqual(edges, {"q0": 0.0, "q20": 2.0, "q80": 8.0, "q100": 10.0})
        with self.assertRaises(Stage20Error):
            registered_rank_edges(pd.Series([1.0] * 20))

    def test_nonoverlap_uses_actual_exit_per_definition_and_symbol(self):
        frame = pd.DataFrame([
            {"translation_id": "A", "symbol": "X", "event_id": "1", "decision_ts": pd.Timestamp("2023-01-01T00:00Z"), "entry_ts": pd.Timestamp("2023-01-01T00:05Z"), "exit_ts": pd.Timestamp("2023-01-01T01:05Z")},
            {"translation_id": "A", "symbol": "X", "event_id": "2", "decision_ts": pd.Timestamp("2023-01-01T00:30Z"), "entry_ts": pd.Timestamp("2023-01-01T00:35Z"), "exit_ts": pd.Timestamp("2023-01-01T01:35Z")},
            {"translation_id": "B", "symbol": "X", "event_id": "3", "decision_ts": pd.Timestamp("2023-01-01T00:30Z"), "entry_ts": pd.Timestamp("2023-01-01T00:35Z"), "exit_ts": pd.Timestamp("2023-01-01T01:35Z")},
        ])
        self.assertEqual(nonoverlap(frame).event_id.tolist(), ["1", "3"])

    def test_beam_requires_frozen_positive_eligibility(self):
        base = {
            "canonical_translation_id": "A", "integrity_pass": True, "accepted_trade_count": 30,
            "independent_market_day_clusters": 20, "independent_utc_hour_clusters": 20,
            "aggregate_base_net_mean_bps": 1.0, "median_inner_fold_base_net_mean_bps": 1.0,
            "p20_inner_fold_base_net_mean_bps": 1.0, "cluster_bootstrap_lower_bound_bps": 1.0,
            "left_tail_utility_bps": 1.0, "opportunity_frequency_per_30d": 1.0,
            "execution_margin_bps": 1.0, "complexity": 1, "market_day_returns": {"d1": 1, "d2": 2, "d3": 3},
        }
        self.assertEqual([row["canonical_translation_id"] for row in choose_beam([base])], ["A"])
        bad = dict(base, canonical_translation_id="B", aggregate_base_net_mean_bps=0)
        self.assertEqual(choose_beam([bad]), [])

    def test_registered_event_generation_is_causal_and_rearms(self):
        ts = pd.date_range("2024-01-01", periods=8, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "timestamp_utc": ts, "eligible": True, "known_lifecycle_mask": True,
            "trade_coverage": True, "mark_coverage": True, "analytics_coverage": True,
            "trade_return_1h": [-.002, -.002, 0, -.002, -.002, -.002, -.002, -.002],
            "mark_return_1h": [-.002, -.002, 0, -.002, -.002, -.002, -.002, -.002],
            "oi_log_change_1h": [-.02, -.02, 0, -.02, -.02, -.02, -.02, -.02],
            "liquidation_base_units_1h": [1, 1, 0, 1, 1, 1, 1, 1],
            "liquidation_intensity_robust_z": 3.0, "liquidation_normalization_valid": True,
            "basis_bps": -20.0, "basis_change_1h": -.002,
            "basis_level_normalization_valid": True, "basis_change_normalization_valid": True,
            "trade_close": [90, 89, 88, 87, 88, 89, 91, 92],
            "mark_close": [90, 89, 88, 87, 88, 89, 91, 92], "breadth_share": .03,
        })
        thresholds = {"oi_q0": -.1, "oi_q20": -.015, "trade_abs_q0": 0, "trade_abs_q80": 15,
                      "trade_abs_q100": 100, "mark_abs_q0": 0, "mark_abs_q80": 15,
                      "mark_abs_q100": 100, "liquidation_q0": 0, "liquidation_q80": 2,
                      "liquidation_q100": 10, "basis_level_q0": -100, "basis_level_q20": -15,
                      "basis_change_q0": -100, "basis_change_q20": -15,
                      "breadth_q0": 0, "breadth_q80": .02, "breadth_q100": 1}
        bcell = {"canonical_translation_id": "B", "cell_id": "B1", "search_axes": {
            "price_state": "negative", "price_axis": "raw_bps", "oi_axis": "raw_oi_log_change",
            "liquidation_context": "present_absent", "branch": "reversal", "horizon": "1h"}}
        events = kda02b_symbol_events(frame, "PF_X", bcell, thresholds, ts[0], ts[-1] + pd.Timedelta(minutes=5))
        self.assertEqual(events.decision_ts.tolist(), [ts[3] + pd.Timedelta(minutes=5)])
        self.assertTrue(events.side.eq("long").all())

        xcell = {"canonical_translation_id": "X", "cell_id": "X1", "search_axes": {
            "component_scaling": "raw_unit", "horizon": "1h"}, "feature_contract": {
            "required_components": ["downside_trade_displacement", "downside_mark_displacement",
                                    "oi_contraction", "completed_trade_mark_reclaim"]}}
        xevents = kdx_symbol_events(frame, "PF_X", xcell, thresholds, ts[0], ts[-1] + pd.Timedelta(minutes=10))
        self.assertEqual(xevents.decision_ts.tolist(), [ts[4] + pd.Timedelta(minutes=5)])


if __name__ == "__main__":
    unittest.main()
