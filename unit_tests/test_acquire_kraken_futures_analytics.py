import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.acquire_kraken_futures_analytics import (
    Acquirer, JobSpec, Ledger, PROTECTED_START, TRAIN_START, audit_specs,
    approved_unit_metrics, compact_month_parts, continuation_spec, deduplicate,
    monthly_specs, normalized_rows, request_url, storage_reserve_pass,
    validate_acquisition_approval, write_parquet,
)


class AnalyticsAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.spec = JobSpec("test", "PF_XBTUSD", "open-interest", 300, TRAIN_START, TRAIN_START + 3600)

    def payload(self, timestamps=None, data=None, more=False):
        timestamps = timestamps if timestamps is not None else [TRAIN_START, TRAIN_START + 300]
        data = data if data is not None else [["1", "2", "3", "4"], ["5", "6", "7", "8"]]
        return {"result": {"timestamp": timestamps, "data": data, "more": more}, "errors": []}

    def test_phase_a_matrix_is_exact(self):
        symbols = ["PF_XBTUSD", "PF_ETHUSD", "PF_1INCHUSD", "PF_AAVEUSD", "PF_ADAUSD", "PF_ALGOUSD"]
        rows = audit_specs(symbols)
        self.assertEqual(len(rows), 144)
        self.assertEqual(len({x.job_id for x in rows}), 144)
        self.assertTrue(all(x.to < PROTECTED_START for x in rows))
        final = [x for x in rows if x.since == 1766534400]
        self.assertTrue(all(x.to == PROTECTED_START - x.interval_seconds for x in final))

    def test_request_requires_explicit_preprotected_bounds(self):
        url = request_url(self.spec)
        self.assertIn("since=1672531200", url)
        self.assertIn("to=1672534800", url)
        with self.assertRaises(ValueError):
            request_url(JobSpec("x", "PF_XBTUSD", "open-interest", 300, TRAIN_START, PROTECTED_START + 1))
        with self.assertRaises(ValueError):
            request_url(JobSpec("x", "PF_XBTUSD", "open-interest", 300, TRAIN_START, PROTECTED_START))

    def test_more_uses_inclusive_last_timestamp(self):
        frame, more = normalized_rows(self.spec, self.payload(more=True))
        nxt = continuation_spec(self.spec, frame, more)
        self.assertEqual(nxt.since, TRAIN_START + 300)
        self.assertEqual(nxt.to, self.spec.to)

    def test_empty_more_has_no_invented_cursor(self):
        frame, more = normalized_rows(self.spec, self.payload([], [], True))
        self.assertIsNone(continuation_spec(self.spec, frame, more))

    def test_protected_or_pre2023_rejected_before_values(self):
        class ExplodingData(dict):
            def get(self, *args, **kwargs):
                raise AssertionError("data values opened")
        payload = {"result": {"timestamp": [PROTECTED_START], "data": ExplodingData(), "more": False}}
        with self.assertRaises(PermissionError):
            normalized_rows(self.spec, payload)

    def test_empty_and_schema_drift(self):
        frame, more = normalized_rows(self.spec, self.payload([], [], False))
        self.assertTrue(frame.empty)
        self.assertFalse(more)
        with self.assertRaises(ValueError):
            normalized_rows(self.spec, {"result": {"timestamp": []}})

    def test_conflicting_and_identical_duplicates(self):
        frame = pd.DataFrame({"timestamp_epoch_seconds": [1, 1], "value_json": ["1", "1"]})
        out, count = deduplicate(frame)
        self.assertEqual(len(out), 1)
        self.assertEqual(count, 1)
        frame.loc[1, "value_json"] = "2"
        with self.assertRaises(ValueError):
            deduplicate(frame)

    def test_metric_schemas(self):
        for metric, data in (("liquidation-volume", ["1", "2"]), ("future-basis", {"basis": ["1", "2"]})):
            spec = JobSpec("x", "PF_ETHUSD", metric, 300, TRAIN_START, TRAIN_START + 3600)
            frame, _ = normalized_rows(spec, self.payload(data=data))
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.analytics_type.iloc[0], metric)

    def test_raw_semantic_fields_are_lossless(self):
        frame, _ = normalized_rows(self.spec, self.payload())
        self.assertEqual(frame.loc[0, ["value_0_raw", "value_1_raw", "value_2_raw", "value_3_raw"]].tolist(), ["1", "2", "3", "4"])
        self.assertEqual(frame.semantic_status.unique().tolist(), ["source_authorized_economic_interpretation_blocked"])
        basis = JobSpec("x", "PF_XBTUSD", "future-basis", 300, TRAIN_START, TRAIN_START + 3600)
        payload = self.payload(data={"basis": ["0.1", "0.2"], "usdValue": ["10", "20"]})
        out, _ = normalized_rows(basis, payload)
        self.assertEqual(out[["basis_raw", "usdValue_raw"]].iloc[0].tolist(), ["0.1", "10"])

    def test_stale_job_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "jobs.sqlite")
            ledger.add(self.spec)
            ledger.update(self.spec.job_id, status="running")
            self.assertEqual(ledger.reset_stale_running(), 1)
            self.assertEqual(ledger.row(self.spec.job_id)["status"], "planned")

    def test_atomic_hash_resume_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = Ledger(root / "jobs.sqlite")
            body = json.dumps(self.payload()).encode()
            acq = Acquirer(ledger, root / "data", fetcher=lambda _url: (200, {}, body), throttle_seconds=0)
            acq.run([self.spec])
            self.assertTrue(ledger.verified_complete(self.spec))
            before = dict(ledger.row(self.spec.job_id))
            acq.run([self.spec])
            after = dict(ledger.row(self.spec.job_id))
            self.assertEqual(before["attempt_count"], after["attempt_count"])

    def test_progress_callback_can_stop_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = Ledger(root / "jobs.sqlite")
            body = json.dumps(self.payload()).encode()
            calls = []
            acquirer = Acquirer(ledger, root / "data", fetcher=lambda _url: (200, {}, body), throttle_seconds=0,
                                progress_callback=lambda spec: (calls.append(spec.job_id) or False))
            acquirer.run([self.spec])
            self.assertTrue(acquirer.stop.requested)
            self.assertEqual(len(calls), 1)
            self.assertEqual(ledger.row(self.spec.job_id)["status"], "complete")

    def test_post_normalization_crash_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = Ledger(root / "jobs.sqlite")
            body = json.dumps(self.payload()).encode()
            acq = Acquirer(ledger, root / "data", fetcher=lambda _url: (200, {}, body), throttle_seconds=0)
            acq.run([self.spec])
            ledger.update(self.spec.job_id, status="running", completed_utc=None)
            self.assertTrue(ledger.verified_complete(self.spec))
            self.assertEqual(ledger.row(self.spec.job_id)["error_class"], "crash_recovery")

    def test_exact_retry_does_not_mutate_url(self):
        with tempfile.TemporaryDirectory() as td:
            urls = []
            def fetch(url):
                urls.append(url)
                if len(urls) == 1:
                    return 500, {}, b"{}"
                return 200, {}, json.dumps(self.payload()).encode()
            ledger = Ledger(Path(td) / "jobs.sqlite")
            Acquirer(ledger, Path(td) / "data", fetcher=fetch, throttle_seconds=0).run([self.spec])
            self.assertEqual(urls[0], urls[1])

    def test_unsupported_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "jobs.sqlite")
            Acquirer(ledger, Path(td) / "data", fetcher=lambda _url: (404, {}, b"{}"), throttle_seconds=0).run([self.spec])
            row = ledger.row(self.spec.job_id)
            self.assertEqual(row["status"], "blocked_error")
            self.assertEqual(row["error_class"], "unsupported_type_or_symbol")

    def test_zero_economic_fields(self):
        description = Ledger(Path(tempfile.mkdtemp()) / "jobs.sqlite").db.execute("SELECT * FROM jobs").description or []
        columns = {column[0] for column in description}
        forbidden = {"pnl", "return", "mae", "mfe", "rank", "label"}
        self.assertFalse(forbidden & columns)

    def test_unit_and_storage_gates_fail_closed(self):
        statuses = {"open-interest": "unavailable", "liquidation-volume": "verified", "future-basis": "blocked"}
        self.assertEqual(approved_unit_metrics(statuses), ["liquidation-volume"])
        self.assertFalse(storage_reserve_pass(150 * 1024**3, 55 * 1024**3, 10 * 1024**3, 1_000_000, 10))
        self.assertFalse(storage_reserve_pass(150 * 1024**3, 80 * 1024**3, 10 * 1024**3, 5, 10))
        with tempfile.TemporaryDirectory() as td:
            plan = Path(td) / "plan.json"
            plan.write_text(json.dumps({"approved_metrics": ["open-interest"], "projection": {"storage_gate_pass": False}}))
            with self.assertRaises(ValueError):
                validate_acquisition_approval(plan)

    def test_monthly_full_scope_planner_is_strictly_bounded(self):
        specs = monthly_specs(["PF_XBTUSD", "PF_ETHUSD"], ["future-basis"], 60, "phase_c")
        self.assertEqual(len(specs), 72)
        self.assertTrue(all(spec.since >= TRAIN_START and spec.to < PROTECTED_START for spec in specs))

    def test_monthly_compaction_is_partition_local_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as td:
            partition = Path(td) / "metric" / "interval=300" / "symbol=PF_XBTUSD" / "year=2023" / "month=01"
            partition.mkdir(parents=True)
            frame, _ = normalized_rows(self.spec, self.payload())
            first, second = partition / "part-a.parquet", partition / "part-b.parquet"
            write_parquet(frame.iloc[:1], first, {"x": "1"})
            write_parquet(frame, second, {"x": "2"})
            output = partition / "compacted.parquet"
            compact_month_parts([first, second], output)
            self.assertEqual(len(pd.read_parquet(output)), 2)
            other = Path(td) / "other" / "part.parquet"
            other.parent.mkdir(); write_parquet(frame, other, {"x": "3"})
            with self.assertRaises(ValueError):
                compact_month_parts([first, other], partition / "bad.parquet")


if __name__ == "__main__":
    unittest.main()
