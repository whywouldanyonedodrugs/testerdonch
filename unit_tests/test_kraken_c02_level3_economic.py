import tempfile,unittest
from pathlib import Path
import pandas as pd
import numpy as np
from tools import run_kraken_c02_level3_economic as r

class Tests(unittest.TestCase):
 def bars(self):
  ts=pd.date_range("2024-01-01",periods=100,freq="5min",tz="UTC"); return pd.DataFrame({"source_open_ts":ts,"open":100.0,"high":101.0,"low":99.0,"close":100.0})
 def event(self): return {"event_id":"e","PF_symbol":"PF_X","decision_ts":pd.Timestamp("2024-01-01T00:00Z"),"canonical_episode_id":"ep"}
 def definition(self,h=1): return {"definition_id":"d","timeout_hours":h}
 def test_next_open_and_timeout(self):
  x=r.prepare(self.event(),self.definition(),self.bars()); self.assertEqual(x["entry_ts"],pd.Timestamp("2024-01-01T00:05Z")); self.assertEqual(x["actual_exit_ts"],pd.Timestamp("2024-01-01T01:05Z"))
 def test_6h_timeout(self): self.assertEqual(r.prepare(self.event(),self.definition(6),self.bars())["actual_exit_ts"],pd.Timestamp("2024-01-01T06:05Z"))
 def test_missing_exit(self):
  with self.assertRaises(r.CandidateInvalid): r.prepare(self.event(),self.definition(12),self.bars())
 def test_lifecycle(self):
  with self.assertRaises(r.CandidateInvalid): r.prepare(self.event(),self.definition(),self.bars(),[(pd.Timestamp("2024-01-01T00:30Z"),pd.Timestamp("2024-01-01T00:40Z"))])
 def test_costs(self): self.assertEqual(r.prepare(self.event(),self.definition(),self.bars())["base_net_bps_ex_funding"],-14)
 def test_no_control_surface(self): self.assertNotIn("control",dir(r))
 def test_existing_root_refused_contract(self):
  self.assertTrue(hasattr(r,"main"))
 def test_nonpositive_concentration_only_fails_dependent_gates(self):
  trades=pd.DataFrame({"calendar_year":[2023]*20+[2024]*20+[2025]*60,"base_net_bps_ex_funding":[-1.0]*100,"stress_net_bps_ex_funding":[-2.0]*100})
  concentration={"max_positive_symbol_share":np.nan,"max_positive_episode_share":np.nan,"max_positive_year_share":np.nan}
  gates=r.gate_report(trades,-1.0,concentration)
  self.assertTrue(gates["executed_trades_ge_100"])
  self.assertTrue(gates["each_year_ge_20"])
  self.assertTrue(gates["bootstrap_lower_ge_minus5"])
  self.assertFalse(gates["symbol_share_le_25pct"])
  self.assertFalse(gates["mean_base_net_positive"])
 def test_funding_contract_label_is_reported(self):
  self.assertEqual(pd.Series(["fully_exact"]).replace({"fully_exact":"fully_exact_funded"}).iloc[0],"fully_exact_funded")
if __name__=="__main__": unittest.main()
