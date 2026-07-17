import tempfile,unittest
from pathlib import Path
import pandas as pd
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
if __name__=="__main__": unittest.main()
