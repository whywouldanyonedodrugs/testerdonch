# Validation

Status: `pass_precommit`

## Authority

- Repository base, `main`, and `origin/main` were
  `405878887780f7e85fe90a601a643bee502d7d5c` at task start.
- The sequence-3 continuity event, snapshot, and pointer passed
  `tools/donch_continuity.py validate-files` with the reported physical hashes.
- The downloaded Stage 20 terminal ZIP SHA-256 is
  `3a3fd639f9e9bb0cb2b7d87828310aa27a82aef689ae085f1e6624e8d020526c`.
- The exact received instruction copy SHA-256 is
  `7fdf7cf3c4ecf24bcaaeef9218551c309cbbeaea88ec22cd53786f8b20a6cae5`.

## Deterministic packet validation

Command:

```text
/opt/testerdonch/.venv/bin/python docs/agent/task_archive/20260720_donch_bt_stage21_research_adequacy_and_core_discovery_packet_20260720_v1/build_packet.py
/opt/testerdonch/.venv/bin/python docs/agent/task_archive/20260720_donch_bt_stage21_research_adequacy_and_core_discovery_packet_20260720_v1/validate_packet.py
```

Result:

```json
{"Capitalcom_payload_opened":false,"candidate_rows":45,"new_attempts":3521,"outer_folds":8,"physical_dependencies_verified":15,"protected_rows_opened":0,"required_files":25,"status":"pass","unique_selected_cells":19}
```

The validator also checked:

- every bound Stage 20 terminal artifact SHA-256;
- 45 candidate rows and 19 unique Stage 20 selected cell IDs;
- separate symbol/day/year contributions against the stored maximum;
- development and outer base-net means against direct shard recomputation, with
  maximum absolute differences below `8e-15`;
- exact per-family and total attempt arithmetic;
- all 3,521 unique materialized configuration addresses, fixed Sobol engine,
  version, seeds, stream offsets, anchors, conditional refinements, and A2
  parent bindings;
- all 800 unique conditional survivor-control templates and the exact 19x11
  KDA02B adjudication registry;
- eight ordered forward-only quarterly outer folds and inner/outer separation;
- runtime/resource equality, four-worker lazy-scheduler bounds, and 30-minute
  heartbeat;
- every task-local contract hash in the campaign manifest;
- approval-request binding to the campaign manifest;
- 15 physically present economic dependency hashes, including market, feature,
  funding, Tier-1 lineage, TSMOM v6, first-wave, and Stage 20 authorities.

The full synthetic supervisor benchmark completed 3,521 jobs with four bounded
workers in 23.61 seconds, sampled 33.9 MB peak process-tree RSS, and wrote 1.30
MB. A complete idempotent replay reused all 3,521 markers. A separate deliberate
bound stop persisted after 103 jobs and resumed to all 3,521 jobs while reusing
the 103 valid markers. All four benchmark records are bound to registry SHA-256
`d582d8e2e5390f7f833849494cd7b4751eff667db99221ce7a8a8c27bf42f8b3` and
kept economic, protected, and Capital.com readers closed.

## Outcome firewall

No new economic screen ran. The only economic artifacts read were finalized
Stage 20 terminal response/outer tables and scored shards, which the task
explicitly authorizes for audit. No protected row, Capital.com payload, private
account data, acquisition, capture, or live action was opened.

## Static checks

All JSON files parse successfully. CSV and JSONL row counts and headers are
deterministic. A complete builder rerun produced a byte-identical hash inventory.
The build timestamp is the exact received-task archive timestamp truncated to
seconds, so rebuilding does not change contract hashes.

Independent review is recorded separately in `REVIEW.md`.
