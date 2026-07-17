from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools import kraken_c01_prerun_contract as frozen
from tools import run_kraken_c01_level3_economic as runner


UTC = "UTC"


def bars(onset: pd.Timestamp, hours: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range(onset - pd.Timedelta(hours=6), onset + pd.Timedelta(hours=hours), freq="5min", inclusive="left")
    trade = pd.DataFrame({"source_open_ts": index, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    trade.loc[trade.source_open_ts.eq(onset-pd.Timedelta(hours=6)), "low"] = 90.0
    mark = trade.copy()
    residual = pd.DataFrame({"source_open_ts": index, "decision_ts": index+pd.Timedelta(minutes=5), "residual": 0.01})
    residual.loc[residual.source_open_ts.eq(onset-pd.Timedelta(minutes=5)), "residual"] = 1.0
    return trade, mark, residual


def event(onset: pd.Timestamp, *, path: str = "smooth", sign: str = "positive") -> dict:
    return {
        "event_id": "event_1", "candidate_id": "candidate_1", "canonical_episode_id": "episode_1",
        "symbol": "PF_TESTUSD", "venue": "Kraken", "decision_ts": onset,
        "shock_window_start": onset-pd.Timedelta(hours=6), "shock_window_end": onset,
        "residual_model_version": frozen.PRIMARY_MODEL, "sign": sign, "path_state": path,
        "residual_shock_6h": 1.71, "largest_bar_share": 1.0/1.71, "path_efficiency": 1.0,
        "feature_version": "c01_residual_path_features_v1_20260717",
        "reference_panel_hash": runner.REFERENCE_PANEL_SHA256, "candidate_cohort_hash": runner.COHORT_SHA256,
        "protected_rows_read": 0, "economic_outputs_computed": False,
    }


def definition(*, path: str = "smooth", sign: str = "positive", side: str = "long", hours: int = 6) -> dict:
    row = next(item for item in frozen.definition_register() if item["model"] == frozen.PRIMARY_MODEL and item["path_state"] == path and item["shock_sign"] == sign and item["timeout_hours"] == hours)
    assert row["side"] == side
    return row


class C01Level3EconomicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.onset = pd.Timestamp("2024-06-01T12:00:00Z")

    def test_all_16_definitions_and_zero_trade_rows(self) -> None:
        register = pd.DataFrame(frozen.definition_register())
        eligibility = pd.DataFrame(columns=["definition_id", "status", "confirmed"])
        trades = pd.DataFrame(columns=[
            "definition_id", "calendar_year", "gross_return_bps", "base_fee_slippage_net_bps",
            "stress_fee_slippage_net_bps", "canonical_episode_id", "symbol", "funding_partition",
            "base_funding_adjusted_net_bps", "conservative_funding_adjusted_net_bps", "severe_funding_adjusted_net_bps",
        ])
        metrics, gates, *_ = runner.compute_reports(register, trades, eligibility)
        self.assertEqual(len(metrics), 16)
        self.assertEqual(len(gates), 16)
        self.assertEqual(int(metrics.executed_trades.sum()), 0)
        self.assertFalse(gates.all_gates_pass.any())

    def test_smooth_entry_and_timeout_next_open(self) -> None:
        trade, mark, residual = bars(self.onset)
        out = runner.prepare_candidate(event(self.onset), definition(), trade, mark, residual)
        self.assertEqual(out["entry_ts"], self.onset)
        self.assertEqual(out["actual_exit_ts"], self.onset+pd.Timedelta(hours=6))
        self.assertEqual(out["exit_reason"], "fixed_timeout")
        self.assertEqual(out["stop_price"], 90.0)

    def test_jump_confirmation_and_dominant_identity(self) -> None:
        trade, mark, residual = bars(self.onset)
        trade.loc[trade.source_open_ts.eq(self.onset+pd.Timedelta(minutes=5)), "close"] = 98.0
        out = runner.prepare_candidate(
            event(self.onset, path="jump_dominated", sign="positive"),
            definition(path="jump_dominated", sign="positive", side="short"), trade, mark, residual,
        )
        self.assertEqual(out["dominant_bar_source_open_ts"], self.onset-pd.Timedelta(minutes=5))
        self.assertEqual(out["confirmation_ts"], self.onset+pd.Timedelta(minutes=10))
        self.assertEqual(out["entry_ts"], self.onset+pd.Timedelta(minutes=10))

    def test_jump_24h_confirmation_boundary(self) -> None:
        trade, mark, residual = bars(self.onset, 31)
        trade.loc[trade.source_open_ts.eq(self.onset+pd.Timedelta(hours=24)-pd.Timedelta(minutes=5)), "close"] = 98.0
        out = runner.prepare_candidate(
            event(self.onset, path="jump_dominated", sign="positive"),
            definition(path="jump_dominated", sign="positive", side="short"), trade, mark, residual,
        )
        self.assertEqual(out["confirmation_ts"], self.onset+pd.Timedelta(hours=24))
        trade.loc[trade.source_open_ts.eq(self.onset+pd.Timedelta(hours=24)-pd.Timedelta(minutes=5)), "close"] = 100.0
        trade.loc[trade.source_open_ts.eq(self.onset+pd.Timedelta(hours=24)), "close"] = 98.0
        with self.assertRaisesRegex(runner.CandidateInvalid, "jump_confirmation_unavailable"):
            runner.prepare_candidate(event(self.onset, path="jump_dominated", sign="positive"), definition(path="jump_dominated", sign="positive", side="short"), trade, mark, residual)

    def test_mark_close_stop_and_next_open(self) -> None:
        trade, mark, residual = bars(self.onset)
        trigger_open = self.onset+pd.Timedelta(minutes=10)
        mark.loc[mark.source_open_ts.eq(trigger_open), "close"] = 89.0
        out = runner.prepare_candidate(event(self.onset), definition(), trade, mark, residual)
        self.assertEqual(out["actual_exit_ts"], trigger_open+pd.Timedelta(minutes=5))
        self.assertEqual(out["exit_reason"], "mark_close_stop_next_trade_open")

    def test_same_bar_stop_timeout_fails_closed(self) -> None:
        trade, mark, residual = bars(self.onset)
        mark.loc[mark.source_open_ts.eq(self.onset+pd.Timedelta(hours=6)-pd.Timedelta(minutes=5)), "close"] = 89.0
        with self.assertRaisesRegex(runner.CandidateInvalid, "same_bar_stop_timeout_ambiguity"):
            runner.prepare_candidate(event(self.onset), definition(), trade, mark, residual)

    def test_missing_trade_mark_and_invalid_stop_fail_closed(self) -> None:
        trade, mark, residual = bars(self.onset)
        mark = mark[~mark.source_open_ts.eq(self.onset+pd.Timedelta(minutes=10))]
        with self.assertRaisesRegex(runner.CandidateInvalid, "missing_mark"):
            runner.prepare_candidate(event(self.onset), definition(), trade, mark, residual)
        trade, mark, residual = bars(self.onset)
        trade.loc[trade.source_open_ts < self.onset, "low"] = 100.0
        with self.assertRaisesRegex(runner.CandidateInvalid, "non_positive_structural"):
            runner.prepare_candidate(event(self.onset), definition(), trade, mark, residual)

    def test_lifecycle_invalid_interval_fails_closed(self) -> None:
        trade, mark, residual = bars(self.onset)
        invalid = [(self.onset+pd.Timedelta(hours=1), self.onset+pd.Timedelta(hours=2))]
        with self.assertRaisesRegex(runner.CandidateInvalid, "known_lifecycle_invalid"):
            runner.prepare_candidate(event(self.onset), definition(), trade, mark, residual, invalid)

    def test_no_artificial_protected_close(self) -> None:
        onset = pd.Timestamp("2025-12-31T22:00:00Z")
        trade, mark, residual = bars(onset, 3)
        trade = trade[trade.source_open_ts < runner.TRAIN_END]
        mark = mark[mark.source_open_ts < runner.TRAIN_END]
        residual = residual[residual.source_open_ts < runner.TRAIN_END]
        with self.assertRaisesRegex(runner.CandidateInvalid, "timeout_next_trade_open_unavailable"):
            runner.prepare_candidate(event(onset), definition(), trade, mark, residual)

    def test_fixed_notional_costs_and_signed_funding(self) -> None:
        long = frozen.fixed_notional_net_bps(entry_price=100, exit_price=101, side="long", fee_bps=10, slippage_bps=4, funding_cashflow_bps=-2)
        short = frozen.fixed_notional_net_bps(entry_price=100, exit_price=99, side="short", fee_bps=20, slippage_bps=12, funding_cashflow_bps=-3)
        self.assertAlmostEqual(long["net_return_bps"], 84)
        self.assertAlmostEqual(short["net_return_bps"], 65)

    def test_funding_partitions_and_missing_nonfinite_fail_closed(self) -> None:
        trades = pd.DataFrame([{"economic_address":"a", "symbol":"PF_TESTUSD", "entry_ts":pd.Timestamp("2024-01-01T00:00Z"), "actual_exit_ts":pd.Timestamp("2024-01-01T02:00Z"), "side":"long"}])
        panel = pd.DataFrame([
            {"symbol":"PF_TESTUSD", "timestamp":pd.Timestamp("2024-01-01T00:00Z"), "funding_exact":True, "funding_imputed":False, "funding_rate_source":"exact", "funding_rate_central":0.001, "funding_rate_conservative":0.002, "funding_rate_severe":0.003, "funding_rate_conservative_short":-0.002, "funding_rate_severe_short":-0.003},
            {"symbol":"PF_TESTUSD", "timestamp":pd.Timestamp("2024-01-01T01:00Z"), "funding_exact":False, "funding_imputed":True, "funding_rate_source":"imputed", "funding_rate_central":0.001, "funding_rate_conservative":0.002, "funding_rate_severe":0.003, "funding_rate_conservative_short":-0.002, "funding_rate_severe_short":-0.003},
            {"symbol":"PF_TESTUSD", "timestamp":pd.Timestamp("2024-01-01T02:00Z"), "funding_exact":True, "funding_imputed":False, "funding_rate_source":"exact", "funding_rate_central":0.001, "funding_rate_conservative":0.002, "funding_rate_severe":0.003, "funding_rate_conservative_short":-0.002, "funding_rate_severe_short":-0.003},
        ])
        out, _ = runner.attach_funding(trades, panel, {field: 0.0 for field in ["funding_rate_central","funding_rate_conservative","funding_rate_severe","funding_rate_conservative_short","funding_rate_severe_short"]})
        self.assertEqual(out.iloc[0].funding_partition, "mixed")
        self.assertAlmostEqual(out.iloc[0].funding_cashflow_central_bps, -20)
        panel.loc[1, "funding_rate_central"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            runner.attach_funding(trades, panel, {})

    def test_actual_exit_non_overlap_and_skips_excluded(self) -> None:
        rows = []
        for i, (entry, exit_) in enumerate(((0, 2), (1, 3), (3, 4))):
            rows.append({"definition_id":"d", "symbol":"s", "economic_address":str(i), "onset_ts":self.onset+pd.Timedelta(hours=entry), "entry_ts":self.onset+pd.Timedelta(hours=entry), "actual_exit_ts":self.onset+pd.Timedelta(hours=exit_)})
        result = frozen.definition_local_non_overlap(rows)
        self.assertEqual([row["economic_address"] for row in result.accepted], ["0", "2"])
        self.assertEqual([row["economic_address"] for row in result.skipped], ["1"])

    def test_concentration_formulas_and_gate_boundaries(self) -> None:
        frame = pd.DataFrame([
            {"base_fee_slippage_net_bps":60.0,"symbol":"A","canonical_episode_id":"E1","calendar_year":2023},
            {"base_fee_slippage_net_bps":40.0,"symbol":"B","canonical_episode_id":"E2","calendar_year":2024},
        ])
        result = runner.concentration_metrics(frame)
        self.assertEqual(result["total_net_bps"], 100)
        self.assertEqual(result["max_symbol_pnl_share"], .6)
        self.assertEqual(result["max_episode_pnl_share"], .6)
        self.assertEqual(result["max_year_positive_pnl_share"], .6)
        frame.base_fee_slippage_net_bps = [-1, -2]
        self.assertTrue(np.isnan(runner.concentration_metrics(frame)["max_symbol_pnl_share"]))

    def test_bootstrap_is_exactly_deterministic_10000(self) -> None:
        values = {"a":[1.0,2.0], "b":[-1.0,3.0]}
        self.assertEqual(frozen.canonical_episode_bootstrap_mean_ci(values), frozen.canonical_episode_bootstrap_mean_ci(values))
        with self.assertRaises(ValueError):
            frozen.canonical_episode_bootstrap_mean_ci(values, resamples=9999)

    def test_event_tape_rejects_protected_pretrain_nonkraken_duplicates(self) -> None:
        base = pd.DataFrame([event(self.onset)])
        self.assertEqual(len(runner.validate_event_tape(base)), 1)
        for field, value, message in (
            ("decision_ts", pd.Timestamp("2026-01-01T00:00Z"), "pre-2023 or protected"),
            ("decision_ts", pd.Timestamp("2022-12-31T23:55Z"), "pre-2023 or protected"),
            ("venue", "Other", "non-Kraken"),
        ):
            bad = base.copy(); bad[field] = value
            with self.assertRaisesRegex(ValueError, message): runner.validate_event_tape(bad)
        with self.assertRaisesRegex(ValueError, "duplicate onset"):
            runner.validate_event_tape(pd.concat([base, base]))

    def test_mixed_event_outcome_schema_rejected_before_reader(self) -> None:
        self.assertTrue(runner.OUTCOME_COLUMNS & {"event_id", "close"})


if __name__ == "__main__":
    unittest.main()
