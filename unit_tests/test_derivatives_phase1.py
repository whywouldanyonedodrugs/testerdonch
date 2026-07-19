import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools.qlmg_derivatives_phase1 import (
    FEATURE_COLUMNS, GRAMMAR_LADDERS, OutcomeReadSpy, episode_table,
    oi_retention_gap_counts, onset_mask, reconcile_universe, validate_grammar,
)


class DerivativesPhase1Tests(unittest.TestCase):
    def test_outcome_spy_rejects_unregistered_and_protected(self):
        spy = OutcomeReadSpy()
        with self.assertRaisesRegex(ValueError, "outcome firewall"):
            spy.read(Path("unused"), ["forward_return_1h"], kind="feature")
        row = {column: [False] for column in FEATURE_COLUMNS}
        row["timestamp_utc"] = [pd.Timestamp("2026-01-01", tz="UTC")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.parquet"; pd.DataFrame(row).to_parquet(path)
            with patch("tools.qlmg_derivatives_phase1.FEATURE_ROOT",Path(tmp)), patch("tools.qlmg_derivatives_phase1.pd.read_parquet") as reader:
                with self.assertRaisesRegex(ValueError, "protected row"):
                    spy.read(path, FEATURE_COLUMNS, kind="feature")
                reader.assert_not_called()
        with self.assertRaisesRegex(ValueError,"unknown reader kind"):
            spy.read(Path("unused"),[],kind="unknown")

    def test_false_to_true_requires_valid_contiguous_false_predecessor(self):
        ts = pd.Series(pd.to_datetime(["2023-03-01 00:00Z", "2023-03-01 00:05Z", "2023-03-01 00:10Z", "2023-03-01 00:20Z"]))
        state = pd.Series([True, False, True, True])
        valid = pd.Series([True, True, True, True])
        self.assertEqual(onset_mask(state, valid, ts).tolist(), [False, False, True, False])

    def test_retention_boundary_true_state_is_not_onset(self):
        ts = pd.Series(pd.date_range("2023-01-01", periods=5, freq="5min", tz="UTC"))
        state = pd.Series([False, True, True, False, True])
        valid = pd.Series([False, True, True, True, True])
        self.assertEqual(onset_mask(state, valid, ts).tolist(), [False, False, False, False, True])

    def test_OI_retention_gaps_measure_OI_not_grid(self):
        ts=pd.Series(pd.date_range("2023-01-01",periods=5,freq="5min",tz="UTC")); oi=pd.Series([1.0,2.0,None,None,3.0])
        self.assertEqual(oi_retention_gap_counts(ts,oi),(1,2))

    def test_episode_duration_and_gap(self):
        ts = pd.Series(pd.date_range("2023-01-01", periods=7, freq="5min", tz="UTC"))
        state = pd.Series([False, True, True, False, False, True, False])
        valid = pd.Series(True, index=state.index)
        episodes = episode_table("PF_XBTUSD", ts, state, valid)
        self.assertEqual(episodes.duration_minutes.tolist(), [10, 5])
        self.assertEqual(float(episodes.minutes_since_prior_episode.iloc[1]), 10.0)

    def test_pit_denominator_is_row_local(self):
        eligible = pd.DataFrame({"a":[True,False],"b":[True,True],"c":[False,True]})
        state = pd.DataFrame({"a":[True,False],"b":[False,True],"c":[False,True]})
        denominator=eligible.sum(axis=1); share=(state & eligible).sum(axis=1)/denominator
        self.assertEqual(denominator.tolist(), [2,2]); self.assertEqual(share.tolist(), [.5,1.0])

    def test_grammar_bounded_and_contamination_is_external_label(self):
        cells=validate_grammar(); self.assertEqual(cells,list(GRAMMAR_LADDERS))
        self.assertTrue(all(len(x)<=6 and {"trade_downside","mark_downside","structural_rejection"}.issubset(x) for x in cells))
        with self.assertRaises(ValueError): validate_grammar([("trade_downside","mark_downside")])

    def test_universe_reconciliation_exact_one_reason(self):
        rows=[]
        for i in range(479):
            rows.append({"PF_symbol":f"PF_{i}USD","canonical_asset_id":str(i),"trade_coverage_day_count":1,
                         "mark_coverage_day_count":1,"included":True,"exclusion_reason":""})
        for row in rows[-19:]: row["included"] = False
        out=reconcile_universe(pd.DataFrame(rows),{"PF_0USD","PF_1USD"})
        self.assertEqual(len(out),479); self.assertEqual(int(out.final_campaign_eligible.sum()),2)
        self.assertFalse(out.campaign_exclusion_reason.eq("").any())

    def test_packet_budget_arithmetic_and_stop_isolation(self):
        family_budgets={"KDA02B":96,"KDA02C":48,"KDX01":84}
        self.assertEqual(sum(family_budgets.values()),228)
        family_stops={"KDX01":"mechanism_underidentified"}
        self.assertNotIn("KDA02B",family_stops); self.assertNotIn("KDA02C",family_stops)


if __name__ == "__main__":
    unittest.main()
