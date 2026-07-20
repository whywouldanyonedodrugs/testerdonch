#!/usr/bin/env python3
"""Validate and transactionally publish the Donch dynamic continuity ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1.0"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SNAPSHOT_PATH_RE = re.compile(r"^snapshots/state_(\d{6})_(\d{8}T\d{6}Z)\.json$")
EVENT_ID_RE = re.compile(r"^event_(\d{6})_(\d{8}T\d{6}Z)_[a-z0-9][a-z0-9_-]*$")

SNAPSHOT_FIELDS = {
    "active_campaign", "approval_packet_hash", "as_of_utc", "campaign_manifest_hash",
    "current_blockers", "data_authority_changes", "human_approval_status",
    "incidents_and_authorized_protected_actions", "last_drive_handoff", "last_task_archive",
    "last_task_id", "last_task_status", "next_authorized_action", "origin_main",
    "pending_source_refresh_items", "project_source_refresh_watermark", "repository_main",
    "repository_root", "schema_version", "sequence", "snapshot_sha256", "terminal_decisions",
    "working_tree_status",
}
SNAPSHOT_REQUIRED = {
    "schema_version", "sequence", "as_of_utc", "repository_main", "last_task_id",
    "last_task_status", "terminal_decisions", "current_blockers", "next_authorized_action",
    "project_source_refresh_watermark",
}
EVENT_FIELDS = {
    "blockers_added", "blockers_closed", "campaign_packet_changes", "data_authority_changes",
    "drive_handoff", "event_id", "event_sha256", "event_time_utc", "event_type",
    "incidents_or_authorized_protected_actions", "material_changes", "next_authorized_action",
    "prohibited_content_confirmed_absent", "project_source_refresh_reason",
    "project_source_refresh_required", "repository_main_after", "repository_main_before",
    "schema_version", "sequence", "task_archive", "task_id", "task_status",
    "terminal_decisions_added",
}
POINTER_FIELDS = {
    "as_of_utc", "last_task_id", "schema_version", "sequence", "snapshot_path",
    "snapshot_sha256",
}
PROHIBITED_CONFIRMATIONS = {
    "credentials", "private_account_data", "protected_strategy_outcomes",
    "partial_candidate_rankings", "raw_market_payload_values",
}
FORBIDDEN_KEY_PARTS = {
    "api_key", "api_secret", "access_token", "refresh_token", "bearer_token", "password",
    "credential", "private_key", "seed_phrase", "account_id", "account_number", "balance",
    "account_equity", "raw_market_payload", "raw_market_value", "protected_strategy_outcome",
    "partial_candidate_rank", "candidate_score", "candidate_return", "outcome_row", "pnl",
    "sharpe",
}
ARRAY_FIELDS = {
    "terminal_decisions", "current_blockers", "data_authority_changes",
    "incidents_and_authorized_protected_actions", "pending_source_refresh_items",
}
EVENT_ARRAY_FIELDS = {
    "blockers_added", "blockers_closed", "campaign_packet_changes", "data_authority_changes",
    "incidents_or_authorized_protected_actions", "material_changes",
    "prohibited_content_confirmed_absent", "terminal_decisions_added",
}


class ContinuityError(RuntimeError):
    """A fail-closed continuity contract error."""


class ContinuityPointerStale(ContinuityError):
    """Immutable evidence is safe but the replaceable pointer did not advance."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def encoded_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = None
    return sha256_bytes(canonical_json(payload))


def load_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ContinuityError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContinuityError(f"{label} must be a JSON object")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        return load_json_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        raise ContinuityError(f"cannot read {path}") from exc


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_utc(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContinuityError(f"{label} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContinuityError(f"{label} is not a valid UTC timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ContinuityError(f"{label} must be UTC")


def _require_sha(value: Any, label: str, pattern: re.Pattern[str] = SHA256_RE) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ContinuityError(f"{label} is not a valid hash")


def _require_string(value: Any, label: str, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ContinuityError(f"{label} must be a string{' (empty allowed)' if allow_empty else ''}")


def _require_string_array(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContinuityError(f"{label} must be an array of strings")


def reject_prohibited_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContinuityError(f"non-string key at {path}")
            normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if any(part in normalized for part in FORBIDDEN_KEY_PARTS):
                raise ContinuityError(f"prohibited or secret field at {path}.{key}")
            reject_prohibited_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_prohibited_fields(child, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ContinuityError(f"unsupported JSON value at {path}")


def validate_snapshot(value: Mapping[str, Any]) -> None:
    if set(value) - SNAPSHOT_FIELDS:
        raise ContinuityError(f"snapshot contains unknown fields: {sorted(set(value) - SNAPSHOT_FIELDS)}")
    missing = SNAPSHOT_REQUIRED - set(value)
    if missing:
        raise ContinuityError(f"snapshot missing required fields: {sorted(missing)}")
    reject_prohibited_fields(value)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ContinuityError("snapshot schema_version mismatch")
    if not _is_int(value.get("sequence")) or value["sequence"] < 0:
        raise ContinuityError("snapshot sequence must be a nonnegative integer")
    _require_utc(value.get("as_of_utc"), "snapshot.as_of_utc")
    _require_sha(value.get("repository_main"), "snapshot.repository_main", GIT_SHA_RE)
    for field in ("last_task_id", "last_task_status", "next_authorized_action"):
        _require_string(value.get(field), f"snapshot.{field}")
    for field in ARRAY_FIELDS:
        if field in value:
            _require_string_array(value[field], f"snapshot.{field}")
    if not isinstance(value.get("project_source_refresh_watermark"), dict):
        raise ContinuityError("snapshot.project_source_refresh_watermark must be an object")
    for field in ("approval_packet_hash", "campaign_manifest_hash"):
        if value.get(field) is not None:
            _require_sha(value[field], f"snapshot.{field}")
    if value.get("origin_main") is not None:
        _require_sha(value["origin_main"], "snapshot.origin_main", GIT_SHA_RE)
    if "snapshot_sha256" in value:
        _require_sha(value["snapshot_sha256"], "snapshot.snapshot_sha256")
        if value["snapshot_sha256"] != self_hash(value, "snapshot_sha256"):
            raise ContinuityError("snapshot embedded self-hash mismatch")


def validate_event(value: Mapping[str, Any]) -> None:
    if set(value) != EVENT_FIELDS:
        raise ContinuityError(
            f"event fields differ from v1 contract; missing={sorted(EVENT_FIELDS-set(value))} "
            f"unknown={sorted(set(value)-EVENT_FIELDS)}"
        )
    reject_prohibited_fields(value)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ContinuityError("event schema_version mismatch")
    if not _is_int(value.get("sequence")) or value["sequence"] < 1:
        raise ContinuityError("event sequence must be a positive integer")
    _require_utc(value.get("event_time_utc"), "event.event_time_utc")
    for field in (
        "drive_handoff", "event_id", "event_type", "next_authorized_action",
        "project_source_refresh_reason", "repository_main_after", "repository_main_before",
        "task_archive", "task_id", "task_status",
    ):
        _require_string(value.get(field), f"event.{field}", allow_empty=field == "project_source_refresh_reason")
    match = EVENT_ID_RE.fullmatch(value["event_id"])
    if not match or int(match.group(1)) != value["sequence"]:
        raise ContinuityError("event_id does not bind the event sequence")
    compact_time = value["event_time_utc"].replace("-", "").replace(":", "")
    if match.group(2) != compact_time:
        raise ContinuityError("event_id does not bind event_time_utc")
    for field in ("repository_main_before", "repository_main_after"):
        _require_sha(value[field], f"event.{field}", GIT_SHA_RE)
    for field in EVENT_ARRAY_FIELDS:
        _require_string_array(value[field], f"event.{field}")
    if set(value["prohibited_content_confirmed_absent"]) != PROHIBITED_CONFIRMATIONS:
        raise ContinuityError("event prohibited-content confirmation is incomplete")
    if value.get("project_source_refresh_required") not in (True, False):
        raise ContinuityError("event.project_source_refresh_required must be boolean")
    _require_sha(value.get("event_sha256"), "event.event_sha256")
    if value["event_sha256"] != self_hash(value, "event_sha256"):
        raise ContinuityError("event embedded self-hash mismatch")


def validate_pointer(value: Mapping[str, Any], snapshot_bytes: bytes, latest_sequence: int | None = None) -> None:
    if set(value) != POINTER_FIELDS:
        raise ContinuityError("pointer fields differ from v1 contract")
    reject_prohibited_fields(value)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ContinuityError("pointer schema_version mismatch")
    if not _is_int(value.get("sequence")) or value["sequence"] < 0:
        raise ContinuityError("pointer sequence must be a nonnegative integer")
    _require_utc(value.get("as_of_utc"), "pointer.as_of_utc")
    _require_string(value.get("last_task_id"), "pointer.last_task_id")
    _require_sha(value.get("snapshot_sha256"), "pointer.snapshot_sha256")
    match = SNAPSHOT_PATH_RE.fullmatch(str(value.get("snapshot_path", "")))
    if not match or int(match.group(1)) != value["sequence"]:
        raise ContinuityError("pointer snapshot_path does not bind sequence")
    compact_time = value["as_of_utc"].replace("-", "").replace(":", "")
    if match.group(2) != compact_time:
        raise ContinuityError("pointer snapshot_path does not bind as_of_utc")
    snapshot = load_json_bytes(snapshot_bytes, "referenced snapshot")
    validate_snapshot(snapshot)
    if sha256_bytes(snapshot_bytes) != value["snapshot_sha256"]:
        raise ContinuityError("pointer snapshot physical hash mismatch")
    if snapshot["sequence"] != value["sequence"]:
        raise ContinuityError("pointer/snapshot sequence mismatch")
    if snapshot["as_of_utc"] != value["as_of_utc"] or snapshot["last_task_id"] != value["last_task_id"]:
        raise ContinuityError("pointer/snapshot identity mismatch")
    if latest_sequence is not None and latest_sequence > value["sequence"]:
        raise ContinuityPointerStale("continuity_pointer_stale: newer immutable snapshot exists")


def _safe_relative_path(relative: str) -> str:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ContinuityError(f"unsafe continuity path: {relative}")
    return path.as_posix()


class LocalContinuityStore:
    """Filesystem store with the same immutability and pointer transaction as Drive."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, relative: str) -> Path:
        return self.root.joinpath(*PurePosixPath(_safe_relative_path(relative)).parts)

    def mkdir(self, relative: str) -> None:
        self._path(relative).mkdir(parents=True, exist_ok=True)

    def exists(self, relative: str) -> bool:
        return self._path(relative).exists()

    def read_bytes(self, relative: str) -> bytes:
        try:
            return self._path(relative).read_bytes()
        except OSError as exc:
            raise ContinuityError(f"cannot read store object {relative}") from exc

    def upload_immutable(self, local: Path, relative: str) -> dict[str, Any]:
        target = self._path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != sha256_file(local) or target.stat().st_size != local.stat().st_size:
                raise ContinuityError(f"immutable object collision: {relative}")
            evidence = verify_round_trip(self, relative, local)
            evidence["reused_identical"] = True
            return evidence
        try:
            with target.open("xb") as output, local.open("rb") as source:
                shutil.copyfileobj(source, output)
        except FileExistsError as exc:
            raise ContinuityError(f"immutable object already exists: {relative}") from exc
        return verify_round_trip(self, relative, local)

    def replace_pointer(self, local: Path, expected_current_sha256: str | None) -> dict[str, Any]:
        final = self._path("CURRENT_STATE_POINTER.json")
        if expected_current_sha256 is None:
            if final.exists():
                raise ContinuityPointerStale("continuity_pointer_stale: pointer appeared during transaction")
        elif not final.exists() or sha256_file(final) != expected_current_sha256:
            raise ContinuityPointerStale("continuity_pointer_stale: pointer changed during transaction")
        temporary = self._path(f"CURRENT_STATE_POINTER.json.tmp.{uuid.uuid4().hex}")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as output, local.open("rb") as source:
            shutil.copyfileobj(source, output)
        if sha256_file(temporary) != sha256_file(local):
            raise ContinuityPointerStale("continuity_pointer_stale: temporary pointer verification failed")
        os.replace(temporary, final)
        return verify_round_trip(self, "CURRENT_STATE_POINTER.json", local)

    def list_files(self, relative: str) -> list[str]:
        root = self._path(relative)
        if not root.exists():
            return []
        return sorted(path.relative_to(self.root).as_posix() for path in root.rglob("*") if path.is_file())


class RcloneContinuityStore:
    """Rclone-backed store rooted at the stable `_DONCH_CONTINUITY` folder."""

    def __init__(self, remote_base: str, root_folder_id: str):
        if not remote_base.startswith("qlmg_sweep_drive:") or not remote_base.rstrip("/").endswith("_DONCH_CONTINUITY"):
            raise ContinuityError("remote base must be the approved qlmg_sweep_drive _DONCH_CONTINUITY path")
        self.remote_base = remote_base.rstrip("/")
        self.root_folder_id = root_folder_id

    def _remote(self, relative: str) -> str:
        return f"{self.remote_base}/{_safe_relative_path(relative)}"

    def _run(self, arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        command = ["rclone", *arguments, "--drive-root-folder-id", self.root_folder_id]
        result = subprocess.run(command, capture_output=True)
        if check and result.returncode:
            raise ContinuityError(f"rclone operation failed ({arguments[0]})")
        return result

    def mkdir(self, relative: str) -> None:
        self._run(["mkdir", self._remote(relative)])

    def exists(self, relative: str) -> bool:
        result = self._run(["lsjson", self._remote(relative), "--stat"], check=False)
        return result.returncode == 0

    def read_bytes(self, relative: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="donch-continuity-read-") as temporary:
            target = Path(temporary) / "object"
            self._run(["copyto", self._remote(relative), str(target)])
            return target.read_bytes()

    def upload_immutable(self, local: Path, relative: str) -> dict[str, Any]:
        if self.exists(relative):
            evidence = verify_round_trip(self, relative, local)
            evidence["reused_identical"] = True
            return evidence
        self._run(["copyto", str(local), self._remote(relative), "--immutable"])
        return verify_round_trip(self, relative, local)

    def replace_pointer(self, local: Path, expected_current_sha256: str | None) -> dict[str, Any]:
        final_relative = "CURRENT_STATE_POINTER.json"
        if expected_current_sha256 is None:
            if self.exists(final_relative):
                raise ContinuityPointerStale("continuity_pointer_stale: pointer appeared during transaction")
        elif not self.exists(final_relative) or sha256_bytes(self.read_bytes(final_relative)) != expected_current_sha256:
            raise ContinuityPointerStale("continuity_pointer_stale: pointer changed during transaction")
        temporary_relative = f"CURRENT_STATE_POINTER.json.tmp.{uuid.uuid4().hex}"
        self._run(["copyto", str(local), self._remote(temporary_relative), "--immutable"])
        verify_round_trip(self, temporary_relative, local)
        try:
            self._run(["moveto", self._remote(temporary_relative), self._remote(final_relative)])
        except ContinuityError as exc:
            raise ContinuityPointerStale(
                f"continuity_pointer_stale: immutable evidence retained; temporary pointer {temporary_relative} retained"
            ) from exc
        return verify_round_trip(self, final_relative, local)

    def list_files(self, relative: str) -> list[str]:
        result = self._run(["lsf", self._remote(relative), "--recursive", "--files-only"])
        prefix = _safe_relative_path(relative).rstrip("/")
        return sorted(f"{prefix}/{line}" for line in result.stdout.decode().splitlines() if line)


def verify_round_trip(store: Any, relative: str, local: Path) -> dict[str, Any]:
    expected = local.read_bytes()
    actual = store.read_bytes(relative)
    if len(actual) != len(expected) or sha256_bytes(actual) != sha256_bytes(expected):
        raise ContinuityError(f"round-trip verification failed: {relative}")
    return {"path": relative, "bytes": len(expected), "sha256": sha256_bytes(expected)}


def bootstrap_ledger(
    store: Any, readme_path: Path, schema_path: Path, snapshot_path: Path, pointer_path: Path,
) -> dict[str, Any]:
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = load_json_bytes(snapshot_bytes, "initial snapshot")
    pointer = load_json(pointer_path)
    validate_snapshot(snapshot)
    validate_pointer(pointer, snapshot_bytes, latest_sequence=0)
    if pointer["sequence"] != 0:
        raise ContinuityError("bootstrap requires sequence zero")
    if store.exists("CURRENT_STATE_POINTER.json"):
        raise ContinuityError("continuity ledger is already bootstrapped")
    for directory in ("events", "snapshots", "daily"):
        store.mkdir(directory)
    evidence = []
    evidence.append(store.upload_immutable(readme_path, "README.md"))
    evidence.append(store.upload_immutable(schema_path, "SCHEMA.json"))
    evidence.append(store.upload_immutable(snapshot_path, pointer["snapshot_path"]))
    evidence.append(store.replace_pointer(pointer_path, expected_current_sha256=None))
    return {"status": "pass", "sequence": 0, "verified_objects": evidence}


def publish_update(store: Any, event_path: Path, snapshot_path: Path, pointer_path: Path) -> dict[str, Any]:
    event = load_json(event_path)
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = load_json_bytes(snapshot_bytes, "new snapshot")
    pointer = load_json(pointer_path)
    validate_event(event)
    validate_snapshot(snapshot)
    validate_pointer(pointer, snapshot_bytes)
    if not (event["sequence"] == snapshot["sequence"] == pointer["sequence"]):
        raise ContinuityError("event, snapshot, and pointer sequences differ")
    if event["task_id"] != snapshot["last_task_id"] or event["task_id"] != pointer["last_task_id"]:
        raise ContinuityError("event, snapshot, and pointer task identities differ")
    current_bytes = store.read_bytes("CURRENT_STATE_POINTER.json")
    current = load_json_bytes(current_bytes, "current pointer")
    current_snapshot_bytes = store.read_bytes(current["snapshot_path"])
    validate_pointer(current, current_snapshot_bytes)
    if event["sequence"] != current["sequence"] + 1:
        raise ContinuityError("new continuity sequence must equal current sequence plus one")
    expected_event_path = f"events/{event['event_id']}.json"
    if pointer["snapshot_path"] != snapshot_path.name and pointer["snapshot_path"] != f"snapshots/{snapshot_path.name}":
        raise ContinuityError("pointer snapshot_path does not name supplied snapshot")
    evidence = [
        store.upload_immutable(event_path, expected_event_path),
        store.upload_immutable(snapshot_path, pointer["snapshot_path"]),
    ]
    try:
        evidence.append(store.replace_pointer(pointer_path, sha256_bytes(current_bytes)))
    except ContinuityPointerStale:
        raise
    return {"status": "pass", "sequence": event["sequence"], "verified_objects": evidence}


def validate_local_ledger(root: Path) -> dict[str, Any]:
    for name in ("README.md", "SCHEMA.json", "CURRENT_STATE_POINTER.json"):
        if not (root / name).is_file():
            raise ContinuityError(f"missing ledger file: {name}")
    for name in ("events", "snapshots", "daily"):
        if not (root / name).is_dir():
            raise ContinuityError(f"missing ledger directory: {name}")
    snapshots: dict[int, tuple[Path, bytes]] = {}
    for path in sorted((root / "snapshots").glob("state_*.json")):
        match = SNAPSHOT_PATH_RE.fullmatch(path.relative_to(root).as_posix())
        if not match:
            raise ContinuityError(f"malformed snapshot filename: {path.name}")
        data = path.read_bytes()
        value = load_json_bytes(data, path.name)
        validate_snapshot(value)
        sequence = int(match.group(1))
        if value["sequence"] != sequence or sequence in snapshots:
            raise ContinuityError("snapshot filename/sequence mismatch or duplicate")
        snapshots[sequence] = (path, data)
    if not snapshots or sorted(snapshots) != list(range(max(snapshots) + 1)):
        raise ContinuityError("snapshot sequence is not contiguous from zero")
    pointer = load_json(root / "CURRENT_STATE_POINTER.json")
    referenced = root / str(pointer.get("snapshot_path", ""))
    if not referenced.is_file():
        raise ContinuityError("pointer references a missing snapshot")
    validate_pointer(pointer, referenced.read_bytes(), latest_sequence=max(snapshots))
    events: dict[int, Path] = {}
    for path in sorted((root / "events").glob("event_*.json")):
        value = load_json(path)
        validate_event(value)
        expected = f"{value['event_id']}.json"
        if path.name != expected or value["sequence"] in events:
            raise ContinuityError("event filename/sequence mismatch or duplicate")
        events[value["sequence"]] = path
    if sorted(events) != list(range(1, pointer["sequence"] + 1)):
        raise ContinuityError("event sequence does not cover every post-bootstrap snapshot")
    return {
        "status": "pass", "pointer_sequence": pointer["sequence"],
        "snapshots": len(snapshots), "events": len(events), "daily_is_authority": False,
    }


def generate_daily_digest(events: Iterable[Path], date: str, output: Path) -> dict[str, Any]:
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ContinuityError("digest date must be YYYY-MM-DD") from exc
    selected = []
    for path in events:
        event = load_json(path)
        validate_event(event)
        if event["event_time_utc"].startswith(date):
            selected.append(event)
    selected.sort(key=lambda item: item["sequence"])
    lines = [
        f"# Donch Continuity Digest — {date}", "",
        "This digest is a convenience summary, not authority. Verify the pointer, snapshot, events, and task handoffs.", "",
    ]
    for event in selected:
        lines.extend([
            f"## Sequence {event['sequence']}: {event['task_id']}", "",
            f"- Status: `{event['task_status']}`", f"- Event: `{event['event_id']}`",
            f"- Handoff: `{event['drive_handoff']}`", f"- Next: {event['next_authorized_action']}", "",
        ])
    if not selected:
        lines.extend(["No material events were recorded for this date.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, output)
    return {"status": "pass", "date": date, "event_count": len(selected), "sha256": sha256_file(output)}


def _store_from_args(args: argparse.Namespace) -> RcloneContinuityStore:
    return RcloneContinuityStore(args.remote_base, args.drive_root_folder_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_files = subparsers.add_parser("validate-files")
    validate_files.add_argument("--snapshot", type=Path, required=True)
    validate_files.add_argument("--pointer", type=Path, required=True)
    validate_files.add_argument("--event", type=Path)

    validate_ledger = subparsers.add_parser("validate-ledger")
    validate_ledger.add_argument("root", type=Path)

    bootstrap = subparsers.add_parser("bootstrap-drive")
    publish = subparsers.add_parser("publish-drive")
    for command in (bootstrap, publish):
        command.add_argument("--remote-base", required=True)
        command.add_argument("--drive-root-folder-id", required=True)
        command.add_argument("--snapshot", type=Path, required=True)
        command.add_argument("--pointer", type=Path, required=True)
    bootstrap.add_argument("--readme", type=Path, required=True)
    bootstrap.add_argument("--schema", type=Path, required=True)
    publish.add_argument("--event", type=Path, required=True)

    digest = subparsers.add_parser("digest")
    digest.add_argument("--events-dir", type=Path, required=True)
    digest.add_argument("--date", required=True)
    digest.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "validate-files":
            snapshot_bytes = args.snapshot.read_bytes()
            snapshot = load_json_bytes(snapshot_bytes, "snapshot")
            validate_snapshot(snapshot)
            validate_pointer(load_json(args.pointer), snapshot_bytes)
            if args.event:
                validate_event(load_json(args.event))
            result = {"status": "pass", "sequence": snapshot["sequence"]}
        elif args.command == "validate-ledger":
            result = validate_local_ledger(args.root)
        elif args.command == "bootstrap-drive":
            result = bootstrap_ledger(_store_from_args(args), args.readme, args.schema, args.snapshot, args.pointer)
        elif args.command == "publish-drive":
            result = publish_update(_store_from_args(args), args.event, args.snapshot, args.pointer)
        else:
            result = generate_daily_digest(sorted(args.events_dir.glob("event_*.json")), args.date, args.output)
    except (ContinuityError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
