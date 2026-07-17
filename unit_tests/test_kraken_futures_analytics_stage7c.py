import json, tempfile, unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools.acquire_kraken_futures_analytics import PROTECTED_START, sha256_bytes, sha256_file
from tools.run_kraken_futures_analytics_stage7c import (
    CORE, SEMANTIC_STATUS, Unit, build_shards, build_units, cleanup_staging,
    compact_parquet, create_bundle, final_paths, storage, unit_specs, verify_unit_manifest,
)

class Stage7CTests(unittest.TestCase):
    def inventory(self, root):
        symbols=["PF_XBTUSD","PF_ETHUSD"]+[f"PF_S{i:03d}USD" for i in range(458)]
        rows=[{"frozen_order":i+1,"PF_symbol":s,"included":True} for i,s in enumerate(symbols)]
        p=Path(root)/"inventory.csv"; pd.DataFrame(rows).to_csv(p,index=False); return p

    def test_deterministic_shards_and_schedule(self):
        with tempfile.TemporaryDirectory() as td:
            first=build_shards(self.inventory(td)); second=build_shards(self.inventory(td))
            pd.testing.assert_frame_equal(first,second)
            self.assertEqual(len(first),460); self.assertLessEqual(first.groupby("shard_id").size().max(),32)
            self.assertEqual(tuple(first.iloc[:2].symbol),CORE)
            units=build_units(first)
            self.assertEqual(len(units),1836)
            self.assertTrue(all(u.interval==60 for u in units[:108]))
            self.assertTrue(all(u.interval==300 and u.shard_id=="000" for u in units[108:216]))
            self.assertEqual(len({u.unit_id for u in units}),len(units))

    def test_specs_are_strictly_preprotected(self):
        unit=Unit(0,60,"000",2025,12,"open-interest",CORE)
        specs=unit_specs(unit)
        self.assertEqual(len(specs),2)
        self.assertTrue(all(s.to < PROTECTED_START for s in specs))
        self.assertTrue(all(s.to == PROTECTED_START-60 for s in specs))

    def synthetic_rows(self, root):
        root=Path(root); rows=[]
        for i,symbol in enumerate(CORE):
            raw=b'{"result":{"timestamp":[],"data":[],"more":false}}'
            rp=root/f"{i}.json.zst"; rp.write_bytes(bytes(pa.compress(raw,codec="zstd",asbytes=True)))
            frame=pd.DataFrame({"timestamp_utc":[pd.Timestamp("2023-01-01T00:00:00Z")],"timestamp_epoch_seconds":[1672531200],
                "value_field":["liquidation_volume"],"value_json":[f'"{i}"'],"analytics_type":["liquidation-volume"],
                "symbol":[symbol],"interval_seconds":[300],"source_job_id":[f"j{i}"],"request_since":[1672531200],
                "request_to":[1672531500],"semantic_status":[SEMANTIC_STATUS],"value_raw":[str(i)]})
            pp=root/f"{i}.parquet"; frame.to_parquet(pp,index=False)
            rows.append({"job_id":f"j{i}","status":"complete","raw_compressed_path":str(rp),"raw_compressed_sha256":sha256_file(rp),
                "raw_sha256":sha256_bytes(raw),"raw_compressed_bytes":rp.stat().st_size,"response_bytes":len(raw),"row_count":1,
                "first_timestamp":1672531200,"last_timestamp":1672531200,"parquet_path":str(pp),"symbol":symbol})
        return rows

    def test_bundle_extract_verify_compact_and_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); rows=self.synthetic_rows(root); out=root/"final/data.parquet"
            digest,size,count=compact_parquet(rows,out)
            self.assertEqual(count,2); self.assertEqual(sha256_file(out),digest)
            bundle=root/"bundle/source.tar.zst"; meta,index=create_bundle(rows,bundle)
            self.assertTrue(meta["verified_extraction"]); self.assertEqual(len(index),2)
            # Deterministic rerun verifies, never overwrites.
            self.assertEqual(create_bundle(rows,bundle)[0]["bundle_sha256"],meta["bundle_sha256"])
            manifest=root/"unit.json"; manifest.write_text(json.dumps({"final_parquet":{"path":str(out),"bytes":size,"sha256":digest},
                "raw_bundle":{"path":str(bundle),"bytes":bundle.stat().st_size,"sha256":sha256_file(bundle)}}))
            self.assertIsNotNone(verify_unit_manifest(manifest))
            self.assertEqual(cleanup_staging(rows),4)
            self.assertIsNotNone(verify_unit_manifest(manifest))

    def test_crash_before_manifest_is_not_complete(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(verify_unit_manifest(Path(td)/"missing.json"))

    def test_storage_thresholds_and_paths(self):
        with tempfile.TemporaryDirectory() as td:
            value=storage(Path(td)); self.assertEqual(value["warning_threshold_bytes"],int(value["total_bytes"]*.25)); self.assertEqual(value["hard_stop_threshold_bytes"],int(value["total_bytes"]*.20))
            u=Unit(0,300,"001",2023,1,"future-basis",("PF_X",)); paths=final_paths(Path(td),u)
            self.assertIn("shard=001",str(paths[0])); self.assertTrue(str(paths[1]).endswith("source.tar.zst"))

    def test_no_economic_fields(self):
        forbidden={"signal","return","pnl","rank","mae","mfe","label"}
        import tools.run_kraken_futures_analytics_stage7c as module
        self.assertFalse(forbidden & set(vars(module)))

if __name__=="__main__": unittest.main()
