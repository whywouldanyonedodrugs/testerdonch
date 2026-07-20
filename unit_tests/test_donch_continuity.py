import json
import tempfile
import unittest
from pathlib import Path

from tools.donch_continuity import (
    ContinuityError,
    ContinuityPointerStale,
    LocalContinuityStore,
    PROHIBITED_CONFIRMATIONS,
    bootstrap_ledger,
    encoded_json,
    generate_daily_digest,
    publish_update,
    self_hash,
    sha256_bytes,
    validate_event,
    validate_local_ledger,
    validate_pointer,
    validate_snapshot,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded_json(value))


def snapshot(sequence, when, task_id, commit):
    value = {
        "active_campaign": None,
        "approval_packet_hash": None,
        "as_of_utc": when,
        "campaign_manifest_hash": None,
        "current_blockers": [],
        "data_authority_changes": [],
        "human_approval_status": "not_applicable",
        "incidents_and_authorized_protected_actions": [],
        "last_drive_handoff": "https://drive.google.com/drive/folders/test",
        "last_task_archive": f"docs/agent/task_archive/{task_id}",
        "last_task_id": task_id,
        "last_task_status": "complete",
        "next_authorized_action": "await next exact task",
        "origin_main": commit,
        "pending_source_refresh_items": [],
        "project_source_refresh_watermark": {"date": "2026-07-20", "sequence": sequence},
        "repository_main": commit,
        "repository_root": "/opt/testerdonch",
        "schema_version": "1.0",
        "sequence": sequence,
        "snapshot_sha256": None,
        "terminal_decisions": [],
        "working_tree_status": "reported_clean",
    }
    value["snapshot_sha256"] = self_hash(value, "snapshot_sha256")
    return value


def pointer(value, filename):
    data = encoded_json(value)
    return {
        "as_of_utc": value["as_of_utc"],
        "last_task_id": value["last_task_id"],
        "schema_version": "1.0",
        "sequence": value["sequence"],
        "snapshot_path": f"snapshots/{filename}",
        "snapshot_sha256": sha256_bytes(data),
    }


def event(sequence, when, task_id, before, after):
    compact = when.replace("-", "").replace(":", "")
    value = {
        "blockers_added": [], "blockers_closed": [], "campaign_packet_changes": [],
        "data_authority_changes": [], "drive_handoff": "https://drive.google.com/drive/folders/test",
        "event_id": f"event_{sequence:06d}_{compact}_{task_id.replace('_', '-')}",
        "event_sha256": None, "event_time_utc": when, "event_type": "task_complete",
        "incidents_or_authorized_protected_actions": [], "material_changes": ["synthetic change"],
        "next_authorized_action": "await next exact task",
        "prohibited_content_confirmed_absent": sorted(PROHIBITED_CONFIRMATIONS),
        "project_source_refresh_reason": "", "project_source_refresh_required": False,
        "repository_main_after": after, "repository_main_before": before,
        "schema_version": "1.0", "sequence": sequence,
        "task_archive": f"docs/agent/task_archive/{task_id}", "task_id": task_id,
        "task_status": "complete", "terminal_decisions_added": [],
    }
    value["event_sha256"] = self_hash(value, "event_sha256")
    return value


class CorruptReadStore(LocalContinuityStore):
    def read_bytes(self, relative):
        data = super().read_bytes(relative)
        return data + b"corrupt" if relative == "SCHEMA.json" else data


class PointerFailureStore(LocalContinuityStore):
    def replace_pointer(self, local, expected_current_sha256):
        raise ContinuityPointerStale("continuity_pointer_stale: synthetic pointer failure")


class DonchContinuityTests(unittest.TestCase):
    COMMIT0 = "0" * 40
    COMMIT1 = "1" * 40

    def prepare_bootstrap(self, base, store_class=LocalContinuityStore):
        inputs = base / "inputs"
        readme = inputs / "README.md"
        schema = inputs / "SCHEMA.json"
        readme.parent.mkdir(parents=True)
        readme.write_text("# Test continuity\n")
        schema.write_text('{"type":"object"}\n')
        state = snapshot(0, "2026-07-20T00:00:00Z", "stage19", self.COMMIT0)
        snapshot_path = inputs / "state_000000_20260720T000000Z.json"
        pointer_path = inputs / "CURRENT_STATE_POINTER.json"
        write_json(snapshot_path, state)
        write_json(pointer_path, pointer(state, snapshot_path.name))
        store = store_class(base / "ledger")
        return store, readme, schema, snapshot_path, pointer_path

    def bootstrap(self, base):
        store, readme, schema, snapshot_path, pointer_path = self.prepare_bootstrap(base)
        result = bootstrap_ledger(store, readme, schema, snapshot_path, pointer_path)
        self.assertEqual(result["sequence"], 0)
        return store

    def prepare_update(self, base, sequence=1):
        inputs = base / "update"
        when = "2026-07-20T01:00:00Z"
        task_id = "material_task"
        state = snapshot(sequence, when, task_id, self.COMMIT1)
        snapshot_path = inputs / f"state_{sequence:06d}_20260720T010000Z.json"
        pointer_path = inputs / "CURRENT_STATE_POINTER.json"
        event_path = inputs / f"event_{sequence:06d}.json"
        write_json(snapshot_path, state)
        write_json(pointer_path, pointer(state, snapshot_path.name))
        write_json(event_path, event(sequence, when, task_id, self.COMMIT0, self.COMMIT1))
        return event_path, snapshot_path, pointer_path

    def test_supplied_hash_model_and_pointer_binding(self):
        value = snapshot(0, "2026-07-20T00:00:00Z", "stage19", self.COMMIT0)
        data = encoded_json(value)
        validate_snapshot(value)
        validate_pointer(pointer(value, "state_000000_20260720T000000Z.json"), data, latest_sequence=0)
        value["snapshot_sha256"] = "f" * 64
        with self.assertRaisesRegex(ContinuityError, "self-hash"):
            validate_snapshot(value)

    def test_exact_supplied_stage19_state_validates(self):
        root = Path(__file__).resolve().parents[1]
        received = root / "docs/agent/task_archive/20260720_donch_bt_bootstrap_dynamic_continuity_20260720_v1/received"
        snapshot_path = received / "INITIAL_STATE_SNAPSHOT_000000_STAGE19.json"
        pointer_path = received / "INITIAL_CURRENT_STATE_POINTER.json"
        data = snapshot_path.read_bytes()
        value = json.loads(data)
        self.assertEqual(sha256_bytes(data), "7a7bec1583f39480a290761f23ba3fcc3ee43fbaf201fd7a48afc343e8bedcd9")
        self.assertEqual(value["snapshot_sha256"], "0011b911842961b54cf8d168d9cea05c54b90fb619bc973bfc49b55692d6df1f")
        validate_snapshot(value)
        validate_pointer(json.loads(pointer_path.read_text()), data, latest_sequence=0)

    def test_pointer_filename_timestamp_must_match_snapshot(self):
        value = snapshot(0, "2026-07-20T00:00:00Z", "stage19", self.COMMIT0)
        data = encoded_json(value)
        bad = pointer(value, "state_000000_20260720T010000Z.json")
        with self.assertRaisesRegex(ContinuityError, "does not bind as_of_utc"):
            validate_pointer(bad, data)

    def test_sequence_and_atomic_pointer_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            store = self.bootstrap(base)
            event_path, snapshot_path, pointer_path = self.prepare_update(base)
            result = publish_update(store, event_path, snapshot_path, pointer_path)
            self.assertEqual(result["sequence"], 1)
            self.assertFalse(list(store.root.glob("CURRENT_STATE_POINTER.json.tmp.*")))
            self.assertEqual(json.loads(store.read_bytes("CURRENT_STATE_POINTER.json"))["sequence"], 1)
            self.assertEqual(validate_local_ledger(store.root)["events"], 1)

    def test_sequence_skip_is_rejected_before_immutable_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            store = self.bootstrap(base)
            event_path, snapshot_path, pointer_path = self.prepare_update(base, sequence=2)
            with self.assertRaisesRegex(ContinuityError, "current sequence plus one"):
                publish_update(store, event_path, snapshot_path, pointer_path)
            self.assertEqual(store.list_files("events"), [])

    def test_stale_pointer_detects_newer_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            store = self.bootstrap(base)
            _, snapshot_path, _ = self.prepare_update(base)
            store.upload_immutable(snapshot_path, f"snapshots/{snapshot_path.name}")
            with self.assertRaisesRegex(ContinuityPointerStale, "continuity_pointer_stale"):
                validate_local_ledger(store.root)

    def test_malformed_event_is_rejected(self):
        value = event(1, "2026-07-20T01:00:00Z", "task", self.COMMIT0, self.COMMIT1)
        del value["task_status"]
        with self.assertRaisesRegex(ContinuityError, "event fields differ"):
            validate_event(value)

    def test_secret_field_is_rejected_recursively(self):
        value = snapshot(0, "2026-07-20T00:00:00Z", "stage19", self.COMMIT0)
        value["active_campaign"] = {"api_key": "do-not-store"}
        value["snapshot_sha256"] = self_hash(value, "snapshot_sha256")
        with self.assertRaisesRegex(ContinuityError, "prohibited or secret field"):
            validate_snapshot(value)

    def test_round_trip_verification_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            store, readme, schema, snapshot_path, pointer_path = self.prepare_bootstrap(base, CorruptReadStore)
            with self.assertRaisesRegex(ContinuityError, "round-trip verification failed"):
                bootstrap_ledger(store, readme, schema, snapshot_path, pointer_path)
            self.assertFalse(store.exists("CURRENT_STATE_POINTER.json"))

    def test_pointer_failure_retains_immutable_event_and_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            normal = self.bootstrap(base)
            store = PointerFailureStore(normal.root)
            event_path, snapshot_path, pointer_path = self.prepare_update(base)
            with self.assertRaisesRegex(ContinuityPointerStale, "continuity_pointer_stale"):
                publish_update(store, event_path, snapshot_path, pointer_path)
            self.assertTrue(store.exists(f"events/{json.loads(event_path.read_text())['event_id']}.json"))
            self.assertTrue(store.exists(f"snapshots/{snapshot_path.name}"))
            self.assertEqual(json.loads(store.read_bytes("CURRENT_STATE_POINTER.json"))["sequence"], 0)

    def test_immutable_retry_reuses_identical_and_rejects_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            store = self.bootstrap(base)
            same = base / "same.json"
            same.write_bytes(store.read_bytes("snapshots/state_000000_20260720T000000Z.json"))
            evidence = store.upload_immutable(same, "snapshots/state_000000_20260720T000000Z.json")
            self.assertTrue(evidence["reused_identical"])
            same.write_text("different\n")
            with self.assertRaisesRegex(ContinuityError, "immutable object collision"):
                store.upload_immutable(same, "snapshots/state_000000_20260720T000000Z.json")

    def test_daily_digest_is_explicitly_non_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            event_path, _, _ = self.prepare_update(base)
            output = base / "daily" / "2026-07-20.md"
            result = generate_daily_digest([event_path], "2026-07-20", output)
            self.assertEqual(result["event_count"], 1)
            self.assertIn("not authority", output.read_text())

    def test_repository_requires_material_task_publication(self):
        root = Path(__file__).resolve().parents[1]
        instructions = (root / "AGENTS.md").read_text()
        self.assertIn("After the immutable task handoff for every material task", instructions)
        self.assertIn("Non-material chat and read-only discussion do not publish events", instructions)
        self.assertIn("continuity_pointer_stale", instructions)


if __name__ == "__main__":
    unittest.main()
