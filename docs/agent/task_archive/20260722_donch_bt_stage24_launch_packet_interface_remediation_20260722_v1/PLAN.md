# Stage 24 launch-packet interface remediation

Status: in_progress

Repository root: `/opt/testerdonch-stage22-20260720`

Starting commit: `039627130575e621418bd4ef6c3ed5e3b5b20771`

Branch: `agent/stage24-launch-packet-interface-20260722`

## Objective

Repair only the final packet producer so the final manifest carries the two complete population-authority mappings required by the existing production parser; prove the interface through focused fail-closed tests and one detached `shadow_no_outcome` unit, then publish a replacement approval packet without launching economics.

## Non-goals and hard boundaries

- No economic outcome, protected-period, or Capital.com access.
- No strategy, registry, control, cache, family, KDA02B, selection, materialization, accounting, benchmark, or forensic change or rerun.
- No runtime fallback from `primary_hashes`.
- Preserve continuity sequence 8, commit `0396271`, the prior Drive handoff, all prior roots/evidence, and the empty failed-launch run root.

## Verified authority and repository state

- The isolated worktree was clean at the supplied starting commit; local `main` and `origin/main` matched it.
- The original checkout remains dirty only at the unrelated tracked `code` object and is not an implementation workspace.
- The failed economic service is inactive and disabled; zero economic units executed.
- The old final manifest has scalar hashes under `primary_hashes` but omits the top-level mappings required by `CampaignOrchestrator._authority`.
- Working shadow manifests define each mapping with exactly `path`, `bytes`, `role`, and `sha256`.

## Expected code/test changes

- `tools/build_stage24_final_packet.py`: emit and validate exact top-level population-authority mappings.
- `tools/core_liquid_campaign/shadow_service.py` and `tools/run_stage22_core_liquid_campaign.py`: narrowly expose a no-outcome final-manifest interface canary through the supported service CLI.
- `unit_tests/test_core_liquid_campaign_stage24.py`: builder/schema/physical/fail-closed/round-trip tests.
- `unit_tests/test_core_liquid_campaign_shadow_service.py`: exact final-manifest one-unit canary tests.
- This task archive: authority, delta audit, commands, review, hashes, handoff and continuity record.

## Milestones and failure response

1. Implement exact mappings. Accept only if physical path/size/hash, role, and `primary_hashes` equality are enforced. Stop on schema ambiguity.
2. Add focused tests and a bounded no-outcome canary. Accept only if the real `CampaignOrchestrator._authority` consumes the regenerated manifest and one registered production-shaped unit reconciles with Telegram lifecycle calls and no orphan.
3. Regenerate a new versioned packet without changing frozen evidence. Accept only if registries/cache/evidence retain their old hashes and packet inventory round-trips.
4. Obtain one limited independent PASS with zero material blockers. Stop on a material interface, authority, isolation, reconciliation, or runtime-safety finding.
5. Commit and non-force fast-forward `main`/`origin/main`, upload a new unique five-file Drive handoff and round-trip it, then publish exactly the next continuity transaction referencing sequence 8.

## Validation

- Focused unittest modules discovered in `unit_tests/`.
- Actual packet builder against a new versioned output root.
- `git diff --check`, JSON/inventory/hash checks, secret scan.
- Actual supported CLI under a detached systemd-user one-shot service with the secure Telegram environment.

## Authorization state

`economic_run_not_authorized` for the replacement packet. The requested activity is outcome-free interface validation and publication only.

