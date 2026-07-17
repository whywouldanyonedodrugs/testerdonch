#!/usr/bin/env python3
"""Stage 7C exact-scope shard-month Kraken analytics acquisition."""
from __future__ import annotations

import argparse, hashlib, io, json, os, resource, shutil, sqlite3, sys, tarfile, tempfile, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.acquire_kraken_futures_analytics import (
    Acquirer, JobSpec, Ledger, METRICS, PROTECTED_START, TRAIN_START,
    export_request_ledger, sha256_bytes, sha256_file, utc_now,
)
from tools.telegram_notify import TelegramNotifier, load_telegram_env_files

TASK_ID = "donch_bt_stage_7c_resume_analytics_acquisition_20260717_v1"
SEMANTIC_STATUS = "source_authorized_economic_interpretation_blocked"
CORE = ("PF_XBTUSD", "PF_ETHUSD")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def publish_or_verify(path: Path, data: bytes) -> None:
    if path.exists():
        if sha256_file(path) != sha256_bytes(data):
            raise ValueError(f"existing finalized object differs: {path}")
        return
    atomic_bytes(path, data)


def build_shards(inventory_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(inventory_path)
    frame = frame[frame.included.astype(str).str.lower().eq("true")].sort_values("frozen_order", kind="stable")
    symbols = frame.PF_symbol.astype(str).tolist()
    if len(symbols) != 460 or not set(CORE) <= set(symbols):
        raise ValueError("frozen included inventory is not the authorized 460-symbol set")
    ordered = list(CORE) + [symbol for symbol in symbols if symbol not in CORE]
    rows=[]
    for index, symbol in enumerate(ordered):
        shard_number = 0 if index < 2 else 1 + ((index - 2) // 32)
        rows.append({"shard_id":f"{shard_number:03d}","shard_order":shard_number,
                     "symbol_order":index,"symbol":symbol,"is_core":index < 2})
    result=pd.DataFrame(rows)
    if result.groupby("shard_id").size().max() > 32 or result.symbol.duplicated().any():
        raise ValueError("invalid deterministic shard assignment")
    return result


@dataclass(frozen=True)
class Unit:
    order: int; interval: int; shard_id: str; year: int; month: int; metric: str; symbols: tuple[str,...]
    @property
    def unit_id(self) -> str:
        return hashlib.sha256(f"{self.interval}|{self.shard_id}|{self.year:04d}-{self.month:02d}|{self.metric}".encode()).hexdigest()[:20]
    @property
    def run_kind(self) -> str: return f"stage7c_{self.unit_id}"


def build_units(shards: pd.DataFrame) -> list[Unit]:
    months=pd.date_range("2023-01-01T00:00:00Z","2026-01-01T00:00:00Z",freq="MS",inclusive="left")
    groups={sid:tuple(g.sort_values("symbol_order").symbol) for sid,g in shards.groupby("shard_id",sort=True)}
    sequence=[(60,"000",groups["000"]),(300,"000",groups["000"])]
    sequence += [(300,sid,symbols) for sid,symbols in groups.items() if sid != "000"]
    units=[]; order=0
    for interval,sid,symbols in sequence:
        for month in months:
            for metric in METRICS:
                units.append(Unit(order,interval,sid,month.year,month.month,metric,symbols)); order+=1
    return units


def unit_specs(unit: Unit) -> list[JobSpec]:
    start=pd.Timestamp(year=unit.year,month=unit.month,day=1,tz="UTC")
    end=min(start+pd.offsets.MonthBegin(1),pd.Timestamp("2026-01-01T00:00:00Z"))
    return [JobSpec(unit.run_kind,symbol,unit.metric,unit.interval,int(start.timestamp()),int(end.timestamp())-unit.interval)
            for symbol in unit.symbols]


def final_paths(root: Path, unit: Unit) -> tuple[Path,Path,Path,Path]:
    part=root/"normalized"/f"analytics_type={unit.metric}"/f"interval={unit.interval}"/f"year={unit.year:04d}"/f"month={unit.month:02d}"/f"shard={unit.shard_id}"/"data.parquet"
    bundle=root/"raw_bundles"/f"analytics_type={unit.metric}"/f"interval={unit.interval}"/f"year={unit.year:04d}"/f"month={unit.month:02d}"/f"shard={unit.shard_id}"/"source.tar.zst"
    bundle_manifest=bundle.with_name("source.manifest.json")
    unit_manifest=root/"unit_manifests"/f"{unit.order:04d}_{unit.unit_id}.json"
    return part,bundle,bundle_manifest,unit_manifest


def unit_rows(ledger: Ledger, unit: Unit) -> list[dict[str,Any]]:
    return ledger.rows(unit.run_kind)


def validate_terminal_jobs(rows: list[dict[str,Any]], expected_symbols: tuple[str,...]) -> None:
    initial={row["symbol"] for row in rows if row["page"]==0}
    if initial != set(expected_symbols): raise ValueError("unit initial-job coverage mismatch")
    for row in rows:
        if row["status"] == "complete": continue
        if row["status"] == "blocked_error" and row["error_class"] == "unsupported_type_or_symbol": continue
        raise ValueError(f"unit has nonterminal job {row['job_id']} status={row['status']}")


def compact_parquet(rows: list[dict[str,Any]], output: Path) -> tuple[str,int,int]:
    paths=[Path(row["parquet_path"]) for row in rows if row["status"]=="complete"]
    frames=[pq.read_table(path).to_pandas() for path in paths]
    if frames:
        frame=pd.concat(frames,ignore_index=True,sort=False)
        compare=frame.groupby(["symbol","timestamp_epoch_seconds"],sort=False)["value_json"].nunique(dropna=False)
        if (compare>1).any(): raise ValueError("conflicting cross-page duplicate")
        frame=frame.sort_values(["timestamp_epoch_seconds","symbol","source_job_id"],kind="stable")
        frame=frame.drop_duplicates(["symbol","timestamp_epoch_seconds"],keep="first").reset_index(drop=True)
    else:
        frame=pd.DataFrame(columns=["timestamp_utc","timestamp_epoch_seconds","value_field","value_json","analytics_type","symbol","interval_seconds","source_job_id","request_since","request_to","semantic_status"])
    table=pa.Table.from_pandas(frame,preserve_index=False)
    metadata={**(table.schema.metadata or {}),b"task_id":TASK_ID.encode(),b"semantic_status":SEMANTIC_STATUS.encode()}
    table=table.replace_schema_metadata(metadata)
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(f".{output.name}.{os.getpid()}.tmp")
    pq.write_table(table,tmp,compression="zstd",row_group_size=250_000,use_dictionary=True)
    with tmp.open("rb") as handle: os.fsync(handle.fileno())
    if output.exists():
        if sha256_file(output)!=sha256_file(tmp): raise ValueError("existing final Parquet differs")
        tmp.unlink()
    else: os.replace(tmp,output)
    check=pq.read_table(output)
    if check.num_rows!=len(frame): raise ValueError("final Parquet row verification failed")
    return sha256_file(output),output.stat().st_size,len(frame)


def raw_index(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    result=[]
    for row in rows:
        if row["status"]!="complete": continue
        path=Path(row["raw_compressed_path"])
        if sha256_file(path)!=row["raw_compressed_sha256"]: raise ValueError("staged raw hash mismatch")
        restored=bytes(pa.decompress(path.read_bytes(),int(row["response_bytes"]),codec="zstd",asbytes=True))
        if sha256_bytes(restored)!=row["raw_sha256"]: raise ValueError("staged uncompressed raw hash mismatch")
        result.append({"member_path":f"responses/{row['job_id']}.json.zst","request_id":row["job_id"],
                       "response_bytes":row["response_bytes"],"uncompressed_sha256":row["raw_sha256"],
                       "compressed_member_sha256":row["raw_compressed_sha256"],"compressed_bytes":row["raw_compressed_bytes"],
                       "row_count":row["row_count"],"first_timestamp":row["first_timestamp"],"last_timestamp":row["last_timestamp"],
                       "source_path":str(path)})
    return sorted(result,key=lambda x:x["request_id"])


def create_bundle(rows: list[dict[str,Any]], output: Path) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    index=raw_index(rows)
    index_bytes=b"".join((json.dumps({k:v for k,v in item.items() if k!='source_path'},sort_keys=True,separators=(",",":"))+"\n").encode() for item in index)
    with tempfile.TemporaryDirectory(dir=output.parent if output.parent.exists() else None) as td:
        tar_path=Path(td)/"source.tar"
        with tarfile.open(tar_path,"w",format=tarfile.PAX_FORMAT) as tar:
            info=tarfile.TarInfo("INDEX.jsonl"); info.size=len(index_bytes); info.mtime=0; info.mode=0o444
            tar.addfile(info,io.BytesIO(index_bytes))
            for item in index:
                data=Path(item["source_path"]).read_bytes(); info=tarfile.TarInfo(item["member_path"])
                info.size=len(data); info.mtime=0; info.mode=0o444; tar.addfile(info,io.BytesIO(data))
        tar_bytes=tar_path.read_bytes(); compressed=bytes(pa.compress(tar_bytes,codec="zstd",asbytes=True))
        publish_or_verify(output,compressed)
        restored=bytes(pa.decompress(output.read_bytes(),len(tar_bytes),codec="zstd",asbytes=True))
        verify_tar=Path(td)/"verify.tar"; verify_tar.write_bytes(restored)
        extract=Path(td)/"extract"; extract.mkdir()
        with tarfile.open(verify_tar,"r") as tar:
            if any(name.startswith("/") or ".." in Path(name).parts for name in tar.getnames()): raise ValueError("unsafe bundle member")
            tar.extractall(extract, filter="data")
        if (extract/"INDEX.jsonl").read_bytes()!=index_bytes: raise ValueError("bundle index verification failed")
        for item in index:
            if sha256_file(extract/item["member_path"])!=item["compressed_member_sha256"]: raise ValueError("bundle member verification failed")
    manifest={"bundle_path":str(output),"bundle_sha256":sha256_file(output),"bundle_bytes":output.stat().st_size,
              "member_count":len(index),"index_sha256":sha256_bytes(index_bytes),"verified_extraction":True}
    return manifest,index


def verify_unit_manifest(path: Path) -> dict[str,Any]|None:
    if not path.is_file(): return None
    value=json.loads(path.read_text())
    for key in ("final_parquet","raw_bundle"):
        item=value[key]; p=Path(item["path"])
        if not p.is_file() or p.stat().st_size!=item["bytes"] or sha256_file(p)!=item["sha256"]: return None
    return value


def cleanup_staging(rows: list[dict[str,Any]]) -> int:
    count=0
    for row in rows:
        for key in ("raw_compressed_path","parquet_path"):
            path=Path(row[key] or "")
            if path.is_file(): path.unlink(); count+=1
    return count


def storage(root: Path) -> dict[str,int|bool]:
    root.mkdir(parents=True,exist_ok=True)
    usage=shutil.disk_usage(root); stat=os.statvfs(root)
    warning=int(usage.total*.25); hard=int(usage.total*.20)
    return {"total_bytes":usage.total,"free_bytes":usage.free,"warning_threshold_bytes":warning,"hard_stop_threshold_bytes":hard,
            "free_inodes":stat.f_favail,"warning":usage.free<warning,"hard_stop":usage.free<hard}


class Worker:
    def __init__(self,args:argparse.Namespace):
        self.args=args; self.run=args.run_root; self.data=args.data_root; self.run.mkdir(parents=True,exist_ok=True); self.data.mkdir(parents=True,exist_ok=True)
        self.ledger=Ledger(self.run/"KRAKEN_ANALYTICS_JOB_LEDGER.sqlite"); self.started=time.monotonic(); self.last_notice=0.0
        self.last_heartbeat=0.0; self.current_unit: Unit|None=None
        load_telegram_env_files(); self.notifier=TelegramNotifier.from_args(args,run_label="Stage7C Kraken analytics")
        self.shards=build_shards(args.inventory); self.units=build_units(self.shards)
    def heartbeat(self,unit:Unit|None,status:str,last_error:str="") -> None:
        done=sum(verify_unit_manifest(final_paths(self.data,u)[3]) is not None for u in self.units)
        jobs=self.ledger.rows(); st=storage(self.data)
        value={"task_id":TASK_ID,"ts_utc":utc_now(),"status":status,"unit_id":unit.unit_id if unit else None,
               "metric":unit.metric if unit else None,"interval":unit.interval if unit else None,"shard":unit.shard_id if unit else None,
               "year_month":f"{unit.year:04d}-{unit.month:02d}" if unit else None,"units_complete":done,"units_total":len(self.units),
               "jobs_total":len(jobs),"jobs_complete":sum(r['status']=='complete' for r in jobs),"requests":sum(int(r['attempt_count'] or 0) for r in jobs),
               "rows":sum(int(r['row_count'] or 0) for r in jobs),"response_bytes":sum(int(r['response_bytes'] or 0) for r in jobs),
               "rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,"elapsed_seconds":time.monotonic()-self.started,"last_error":last_error,
               "telegram_enabled":self.notifier.enabled,**st}
        atomic_json(self.run/"KRAKEN_ANALYTICS_HEARTBEAT.json",value)
        now=time.monotonic()
        if now-self.last_notice>=3600 or status in {"started","complete","storage_hard_stop","failed"}:
            self.notifier.send(status,f"units {done}/{len(self.units)}; jobs {value['jobs_complete']}/{value['jobs_total']}; rows {value['rows']}; free {st['free_bytes']} bytes")
            self.last_notice=now
        self.last_heartbeat=now
    def request_progress(self, _spec: JobSpec) -> bool:
        st=storage(self.data)
        if st["hard_stop"]:
            self.heartbeat(self.current_unit,"storage_hard_stop")
            return False
        if time.monotonic()-self.last_heartbeat >= 300:
            self.heartbeat(self.current_unit,"running_storage_warning" if st["warning"] else "running")
        return True
    def prepare(self):
        self.shards.to_csv(self.run/"KRAKEN_ANALYTICS_STAGE7C_SYMBOL_SHARDS.csv",index=False)
        atomic_json(self.run/"KRAKEN_ANALYTICS_ACQUISITION_PLAN.json",{"task_id":TASK_ID,"inventory_sha256":sha256_file(self.args.inventory),
            "shard_map_hash":canonical_hash(self.shards.to_dict('records')),"units":len(self.units),"final_data_files":len(self.units)*2,
            "metrics":list(METRICS),"intervals":{"60":list(CORE),"300":"all_460_frozen"},"protected_end_exclusive":PROTECTED_START,
            "semantic_status":SEMANTIC_STATUS,"economic_use_authorized":False,"projection":self.args.projected_bytes})
    def run_all(self):
        self.prepare(); self.heartbeat(None,"started")
        for unit in self.units:
            self.current_unit=unit
            part,bundle,bundle_manifest_path,unit_manifest_path=final_paths(self.data,unit)
            existing=verify_unit_manifest(unit_manifest_path)
            if existing:
                cleanup_staging(unit_rows(self.ledger,unit)); continue
            st=storage(self.data)
            if st["hard_stop"]:
                self.heartbeat(unit,"storage_hard_stop"); return "acquisition_partial_resume_ready"
            specs=unit_specs(unit); self.ledger.reset_stale_running(); acquirer=Acquirer(
                self.ledger,self.data/"staging",throttle_seconds=self.args.throttle,progress_callback=self.request_progress
            )
            acquirer.run(specs)
            if acquirer.stop.requested:
                self.heartbeat(unit,"interrupted"); return "acquisition_partial_resume_ready"
            rows=unit_rows(self.ledger,unit); validate_terminal_jobs(rows,unit.symbols)
            pq_hash,pq_bytes,pq_rows=compact_parquet(rows,part)
            bundle_meta,index=create_bundle(rows,bundle)
            atomic_json(bundle_manifest_path,{"task_id":TASK_ID,"unit_id":unit.unit_id,**bundle_meta})
            manifest={"task_id":TASK_ID,"status":"complete","unit":asdict(unit),"unit_id":unit.unit_id,
                      "final_parquet":{"path":str(part),"bytes":pq_bytes,"sha256":pq_hash,"rows":pq_rows},
                      "raw_bundle":{"path":str(bundle),"bytes":bundle_meta['bundle_bytes'],"sha256":bundle_meta['bundle_sha256'],"members":len(index)},
                      "bundle_manifest":{"path":str(bundle_manifest_path),"sha256":sha256_file(bundle_manifest_path)},
                      "unsupported_symbols":sorted({r['symbol'] for r in rows if r['status']=='blocked_error'}),"published_utc":utc_now()}
            atomic_json(unit_manifest_path,manifest)
            if verify_unit_manifest(unit_manifest_path) is None: raise ValueError("published unit failed verification")
            cleanup_staging(rows); export_request_ledger(self.ledger,self.run/"KRAKEN_ANALYTICS_REQUEST_LEDGER.parquet")
            self.heartbeat(unit,"running")
        self.finalize(); self.heartbeat(None,"complete"); return "historical_analytics_acquisition_complete"
    def finalize(self):
        manifests=[json.loads(p.read_text()) for p in sorted((self.data/"unit_manifests").glob("*.json"))]
        if len(manifests)!=len(self.units): raise ValueError("cannot finalize incomplete unit set")
        files=[]
        for item in manifests:
            for key in ("final_parquet","raw_bundle"):
                files.append({"kind":key,**item[key],"unit_id":item["unit_id"]})
        atomic_json(self.run/"KRAKEN_ANALYTICS_DATA_MANIFEST.json",{"task_id":TASK_ID,"files":files,"file_count":len(files),"content_hash":canonical_hash(files)})
        indexes=[]
        for p in sorted((self.data/"raw_bundles").rglob("source.tar.zst")):
            indexes.append({"bundle_path":str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
        pd.DataFrame(indexes).to_parquet(self.run/"KRAKEN_ANALYTICS_RAW_BUNDLE_INDEX.parquet",index=False,compression="zstd")
        export_request_ledger(self.ledger,self.run/"KRAKEN_ANALYTICS_REQUEST_LEDGER.parquet")


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--inventory",type=Path,required=True); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--projected-bytes",type=int,default=14_338_289_997); p.add_argument("--throttle",type=float,default=.25)
    p.add_argument("--tg-bot-token",default=""); p.add_argument("--tg-chat-id",default=""); p.add_argument("--tg-auto-chat",action="store_true")
    return p.parse_args()


def main():
    args=parse_args(); st=storage(args.data_root); projected=st["free_bytes"]-args.projected_bytes
    if projected < st["warning_threshold_bytes"]: raise SystemExit(f"prestart storage gate failed: projected={projected} threshold={st['warning_threshold_bytes']}")
    worker=Worker(args)
    try:
        status=worker.run_all(); print(json.dumps({"status":status,"heartbeat":str(args.run_root/'KRAKEN_ANALYTICS_HEARTBEAT.json')},sort_keys=True)); return 0 if status!="blocked_by_runtime_storage_or_endpoint" else 2
    except Exception as exc:
        worker.heartbeat(None,"failed",f"{type(exc).__name__}: {exc}"); worker.notifier.send("failed",f"{type(exc).__name__}: {exc}"); raise

if __name__=="__main__": raise SystemExit(main())
