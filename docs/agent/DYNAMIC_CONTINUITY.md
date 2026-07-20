# Donch Dynamic Continuity Protocol

Status: active after the 2026-07-20 bootstrap.

## Stable authority

The dynamic ledger is the `_DONCH_CONTINUITY` direct child of the approved `DONCH_BACKTESTING_HANDOFFS` Drive root. Repository addressing is:

```text
qlmg_sweep_drive:_DONCH_CONTINUITY/
--drive-root-folder-id 1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl
```

Stable folder ID: `1TEmlPRFbks5yWsrXkfkmh7a2apn-M232`  
Stable folder URL: `https://drive.google.com/drive/folders/1TEmlPRFbks5yWsrXkfkmh7a2apn-M232`

Its required structure is:

```text
README.md
SCHEMA.json
CURRENT_STATE_POINTER.json
events/
snapshots/
daily/
```

The pointer and its referenced physical snapshot hash are the current dynamic state. Machine contracts, finalized manifests, hashes, task archives, and immutable handoffs remain higher authority. Daily digests are convenience summaries only.

## Material-task rule

After a material task's ordinary immutable Drive handoff has passed round-trip verification, publish exactly one event and one complete snapshot. Material changes include repository code or instructions, data/evidence authority, terminal decisions or blockers, campaign packets, protected-purpose or incident state, and durable handoffs. Do not publish for ordinary chat, status questions, or read-only discussion that changes no durable authority.

The event sequence and snapshot sequence must equal the current pointer sequence plus one. The event records the verified task handoff. The complete snapshot carries the full current state rather than a delta. Validate locally before any write.

## Hash contracts

- `event_sha256` is SHA-256 of canonical compact JSON with `event_sha256` set to JSON `null`.
- `snapshot_sha256` inside a snapshot is SHA-256 of canonical compact JSON with `snapshot_sha256` set to JSON `null`.
- `CURRENT_STATE_POINTER.json.snapshot_sha256` is SHA-256 of the exact snapshot file bytes stored in Drive.
- JSON file output is UTF-8, sorted keys, two-space indentation, and a final newline.

Active definition files are `docs/agent/continuity/SCHEMA.json` and `docs/agent/continuity/EVENT_TEMPLATE.json`.

## Transaction and failure behavior

1. Read and hash-verify the current pointer and referenced snapshot.
2. Validate the next sequence, event, complete snapshot, and new pointer.
3. Upload the immutable event and snapshot first. An existing path may be reused only when its byte size and SHA-256 are identical.
4. Independently download and compare both immutable objects.
5. Recheck that the current pointer has not changed.
6. Upload the new pointer to a unique temporary object and round-trip verify it.
7. Move the temporary object to `CURRENT_STATE_POINTER.json`.
8. Download and independently verify the final pointer and snapshot.

If the last transaction fails, retain the immutable evidence and temporary pointer, report `continuity_pointer_stale`, and do not publish another sequence until the conflict is reconciled. Never delete or overwrite an event or snapshot.

## Content firewall

Continuity metadata may name incidents, authorizations, hashes, paths, decisions, and aggregate row counts already present in reviewed authority. It must exclude credentials, secrets, private/account data, protected strategy outcomes, partial candidate rankings, candidate returns/scores, and raw market payload values. Continuity work grants no permission to open source payloads.

## Supported commands

```bash
python3 tools/donch_continuity.py validate-files \
  --snapshot <snapshot.json> --pointer <pointer.json> --event <event.json>

python3 tools/donch_continuity.py publish-drive \
  --remote-base qlmg_sweep_drive:_DONCH_CONTINUITY \
  --drive-root-folder-id 1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl \
  --event <event.json> --snapshot <snapshot.json> --pointer <pointer.json>

python3 tools/donch_continuity.py digest \
  --events-dir <verified-local-events> --date YYYY-MM-DD --output <daily-digest.md>
```

Use the repository virtual environment when bare `python3` lacks required repository imports. The continuity tool itself uses only the Python standard library and `rclone` for authorized Drive operations.
