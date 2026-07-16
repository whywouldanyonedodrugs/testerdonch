from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.run_qlmg_d1_narrow_validation import (
    FINAL_HOLDOUT_START,
    d1_catalog,
    d1_variant_hash,
    done_path,
    event_ledger_path,
    parse_time_exit_to_horizon,
    risk_bps_for_stop,
    stage_complete,
    surface_net_returns,
    validate_no_protected,
)
from tools.run_qlmg_path_diagnostics_exit_surface import sample_null_events


class D1NarrowValidationTests(unittest.TestCase):
    def test_seal_boundary_constant(self) -> None:
        self.assertEqual(str(FINAL_HOLDOUT_START), "2026-01-01 00:00:00+00:00")

    def test_d1_catalog_is_frozen_30_variants(self) -> None:
        cats = d1_catalog(20260624)
        self.assertEqual(len(cats), 30)
        self.assertTrue(all(v["family"] == "D1" for v in cats))
        self.assertEqual(d1_variant_hash(cats), d1_variant_hash(d1_catalog(20260624)))

    def test_stage_complete_requires_checkpoint_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done_path(root, "d1-full-coverage-event-rebuild").parent.mkdir(parents=True)
            done_path(root, "d1-full-coverage-event-rebuild").write_text("done")
            self.assertFalse(stage_complete(root, "d1-full-coverage-event-rebuild"))
            (root / "events").mkdir()
            pd.DataFrame({"x": [1]}).to_parquet(event_ledger_path(root), index=False)
            (root / "events/d1_event_coverage_summary.csv").write_text("x\n")
            self.assertTrue(stage_complete(root, "d1-full-coverage-event-rebuild"))

    def test_validate_no_protected_raises(self) -> None:
        df = pd.DataFrame({"decision_ts": [pd.Timestamp("2026-01-01", tz="UTC")]})
        with self.assertRaises(RuntimeError):
            validate_no_protected(df, ["decision_ts"])

    def test_surface_net_returns_side_aware_cost(self) -> None:
        df = pd.DataFrame({
            "atr_bps": [100.0],
            "30m_mfe_bps": [500.0],
            "30m_mae_bps": [10.0],
            "30m_close_return_bps": [100.0],
            "funding_rate": [0.0],
            "side": ["long"],
        })
        r = surface_net_returns(df, "30m", 0.5, 5.0)
        self.assertTrue(np.isfinite(float(r.iloc[0])))
        self.assertLess(float(r.iloc[0]), 5.0)  # cost deducted

    def test_parse_time_exit(self) -> None:
        self.assertEqual(parse_time_exit_to_horizon(15), "15m")
        self.assertEqual(parse_time_exit_to_horizon(30), "30m")
        self.assertEqual(parse_time_exit_to_horizon(60), "1h")

    def test_risk_bps_for_stop(self) -> None:
        df = pd.DataFrame({"atr_bps": [100.0, 200.0]})
        self.assertEqual(list(risk_bps_for_stop(df, 0.5)), [50.0, 100.0])

    def test_null_sampling_empty_is_deterministic(self) -> None:
        events = pd.DataFrame([{"event_id": "x", "symbol": "NOFILEUSDT", "decision_ts": pd.Timestamp("2025-01-01", tz="UTC"), "entry_ts": pd.Timestamp("2025-01-01 00:05", tz="UTC"), "side": "long", "family": "D1", "variant_id": "v"}])
        a = sample_null_events(events, 20260624, max_per_event=3)
        b = sample_null_events(events, 20260624, max_per_event=3)
        self.assertEqual(len(a), len(b))

    def test_tmux_wrapper_exists(self) -> None:
        self.assertTrue(Path("tools/run_qlmg_d1_narrow_validation_tmux.sh").exists())


if __name__ == "__main__":
    unittest.main()
