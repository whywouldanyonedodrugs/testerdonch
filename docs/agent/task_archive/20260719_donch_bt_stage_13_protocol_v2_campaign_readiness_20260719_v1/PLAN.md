# Stage 13 execution plan

Task: `donch_bt_stage_13_protocol_v2_campaign_readiness_20260719_v1`
Start: `a457742c50e27071c7a9f4d7f4dc4c5677534ea7`
Worktree: `/opt/testerdonch-stage13-20260719`
Branch: `agent/stage13-protocol-v2-campaign-readiness-20260719`

## Authority and boundaries

- Preserve every historical decision and route-policy v1 byte-for-byte.
- Permit only Phases 0 and 1; do not read returns, forward paths, PnL, protected-period rows, or Capital.com payloads.
- Treat all 2023–2025 history already inspected by the programme as `program_exposed_historical`, not independent validation.
- Archive the supplied ZIP contents exactly and map only the current prospective method into active documentation.

## Work and acceptance

1. Verify source package, expected start, dirty recovery, authorities, and exact received hashes.
2. Apply protocol v2, family-redefinition policy v1, and campaign protocol v1 without changing historical decisions.
3. Add a standard-library-only campaign state engine with manifest, DAG, registry, beam, permission, stop, restart, budget, heartbeat, and reconciliation contracts.
4. Build deterministic outcome-free Phase 0/1 records. Unknown duration, breadth, or new-state frequency remains an explicit blocker; it is never inferred.
5. Produce a non-self-authorizing Phase 2–5 approval request for mechanically ready derivatives lanes; keep C17 separate.
6. Run unit, schema, determinism, boundary, link, secret, and diff checks; obtain an independent read-only review.
7. Reconcile the archive, commit, non-force push, and copy to the approved Drive target without overwrite.

## Rollback

The work is isolated on a dedicated branch/worktree. Rollback is deletion of the branch/worktree after preserving the task archive; the original dirty checkout is not modified. Existing authorities and old archives are never deleted.
