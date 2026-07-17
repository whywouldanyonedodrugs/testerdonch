import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools import build_kraken_c03_pit_context as c03
from tools.kraken_candle_volume_authority import lagged_top_n_membership


class C03Tests(unittest.TestCase):
    def day(self, value="2024-01-02"):
        return pd.Timestamp(value, tz="UTC")

    def test_opening_and_unknown_preservation(self):
        opening=[pd.Timestamp("2024-01-02T12:00:00Z")]
        self.assertEqual(c03.classify_day(self.day("2024-01-01"),openings=opening,terminal_days=[],resumed_without_boundary=False,identity_collision=False)[0],"verified_ineligible")
        self.assertEqual(c03.classify_day(self.day(),openings=opening,terminal_days=[],resumed_without_boundary=False,identity_collision=False)[0],"unknown")

    def test_explicit_status_transition_boundaries(self):
        valid=((pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2024-01-04T00:00Z")),)
        invalid=((pd.Timestamp("2024-01-02T00:00Z"),pd.Timestamp("2024-01-03T00:00Z")),)
        args=dict(openings=[pd.Timestamp("2023-01-01T00:00Z")],terminal_days=[],resumed_without_boundary=False,identity_collision=False,verified_full_day_intervals=valid,invalid_intervals=invalid)
        self.assertEqual(c03.classify_day(self.day("2024-01-01"),**args)[0],"verified_eligible")
        self.assertEqual(c03.classify_day(self.day("2024-01-02"),**args)[0],"verified_ineligible")
        self.assertEqual(c03.classify_day(self.day("2024-01-04"),**args)[0],"unknown")

    def test_settlement_and_unbounded_resumption(self):
        args=dict(openings=[pd.Timestamp("2023-01-01T00:00Z")],terminal_days=[pd.Timestamp("2024-01-02T00:00Z")],identity_collision=False)
        self.assertEqual(c03.classify_day(self.day(),resumed_without_boundary=False,**args)[0],"verified_ineligible")
        self.assertEqual(c03.classify_day(self.day(),resumed_without_boundary=True,**args)[0],"unknown")

    def test_later_tradeable_snapshot_requires_unknown_resumption_boundary(self):
        history=[("2024-02-01T00:00:00Z",{"tradeable":True},Path("snapshot"))]
        terminal=[pd.Timestamp("2024-01-01T00:00:00Z")]
        self.assertTrue(c03.has_unbounded_resumption(history,terminal,False))

    def test_collision_fails_closed(self):
        result=c03.classify_day(self.day(),openings=[pd.Timestamp("2023-01-01T00:00Z")],terminal_days=[],resumed_without_boundary=False,identity_collision=True)
        self.assertEqual(result[0],"unknown")

    def test_bar_presence_never_changes_lifecycle_state(self):
        kwargs=dict(openings=[pd.Timestamp("2023-01-01T00:00Z")],terminal_days=[],resumed_without_boundary=False,identity_collision=False)
        self.assertEqual(c03.classify_day(self.day(),**kwargs)[0],"unknown")

    def test_feasibility_exact_thresholds(self):
        rows=[]
        for index in range(40):
            rows.append({"date":self.day(),"status_class":"unknown" if index<8 else "verified_eligible","trade_bar_available":True,"mark_bar_available":True,"identity_collision":False})
        matrix,gate=c03.feasibility(pd.DataFrame(rows))
        daily=matrix.iloc[0]; self.assertTrue(daily.usable); self.assertEqual(daily.unknown_share,0.20)
        rows[0]["status_class"]="unknown"; rows[8]["status_class"]="unknown"
        self.assertFalse(c03.feasibility(pd.DataFrame(rows))[0].iloc[0].usable)

    def test_prior_day_liquidity_membership(self):
        daily=pd.DataFrame({"symbol":["A"]*21+["B"]*21,"utc_day":list(pd.date_range("2024-01-01",periods=21,tz="UTC"))*2,"close_based_usd_volume_proxy":[100]*21+[50]*21})
        ranked=lagged_top_n_membership(daily,top_n=1,lookback_days=30,minimum_valid_days=20)
        final=ranked[ranked.utc_day.eq(pd.Timestamp("2024-01-21",tz="UTC"))]
        self.assertEqual(final.loc[final.top_100_eligible,"symbol"].tolist(),["A"])

    def test_causal_breadth_arithmetic_and_denominator(self):
        values=pd.Series([1.0]*10+[-1.0]*10)
        result=c03.breadth_metrics(values,20)
        self.assertTrue(result["available"]); self.assertEqual(result["signed_breadth"],0)
        self.assertFalse(c03.breadth_metrics(values.iloc[:15],20)["available"])

    def test_survivor_controls_are_not_pit_labels(self):
        self.assertNotEqual("current_roster_survivor_breadth_control","pit_membership_authority")
        self.assertNotEqual("aggregate_bar_existence_breadth_control","pit_membership_authority")

    def test_safe_event_loader_uses_only_requested_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"events.csv"; pd.DataFrame({"event_id":["e"],"symbol":["PF_X"],"decision_ts":["2024-01-01T00:00Z"],"net_bps":[99]}).to_csv(path,index=False)
            with patch.object(c03.pd,"read_csv",wraps=c03.pd.read_csv) as reader:
                got=c03.safe_event_identity(path,("event_id","symbol","decision_ts"))
            self.assertNotIn("net_bps",got)
            self.assertEqual(reader.call_args.kwargs["usecols"],["event_id","symbol","decision_ts"])
            with self.assertRaisesRegex(ValueError,"non-identity"):
                c03.safe_event_identity(path,("event_id","net_bps"))

    def test_safe_join_rejects_protected_rows(self):
        events=pd.DataFrame({"event_id":["e"],"symbol":["PF_X"],"decision_ts":["2026-01-01T00:00Z"]})
        features=pd.DataFrame({"symbol":["PF_X"],"feature_available_ts":["2025-12-31T23:00Z"],"available":[True]})
        with self.assertRaisesRegex(ValueError,"protected"):
            c03.attach_feature_availability(events,features)

    def test_hash_replay_is_deterministic(self):
        self.assertEqual(c03.canonical_hash({"b":2,"a":1}),c03.canonical_hash({"a":1,"b":2}))


if __name__=="__main__": unittest.main()
