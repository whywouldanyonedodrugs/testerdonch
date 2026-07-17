# Execution Plan

Task: `donch_bt_stage_2e_c01_level3_economic_20260717_v1`
Class: authorized bounded economic implementation and one frozen 16-definition execution.
Starting commit: `bb9aa9f9cc1b5dcf1c749349ec33f5c6c54ed131`.
Branch: `feature/stage-2e-c01-level3-economic-20260717`.

## Objective
Implement, review, commit, then execute exactly once the frozen C01 Level-3 definitions. Level-4 controls and protected outcomes remain prohibited.

## Frozen authority
Contract SHA-256 `c655e94c35412354356bb7f89c07ca17b71c2ae6537a2a1c42aa3dce928ba77d`; onset tape SHA-256 `e4587653aec82fb66ab6775284501ca768b6689a6b14cdb17a90799f32cea6b7`. The Stage 2D definition register, decision rules, Stage 2C generator/feature/cohort/reference hashes, safe Kraken market manifest, known lifecycle invalidations, and frozen shared-funding model are immutable inputs.

## Milestones
1. [complete] Verify clean synchronized start and all explicit frozen input hashes.
2. [in progress] Implement runner and synthetic mechanics/boundary/accounting tests without real outcome reads.
3. [pending] Run focused and guard suites; review diff and command; create implementation commit.
4. [pending] Execute the authorized 16-definition run once under a fresh UTC root.
5. [pending] Independently recompute artifact invariants, gates, hashes, and protected audit.
6. [pending] Complete archive/registries, one post-run documentation commit, fast-forward main, non-force push, and approved Drive handoff.

## Failure and rollback
Any authority mismatch, unsafe/mixed input, ambiguous same-bar exit, missing required execution/mark/funding/lifecycle input, failed review, or failed test stops the run. Preserve a partial fresh run root as provenance; never overwrite it. Rollback is branch deletion before publication or reverting the two new commits after publication; existing history and prior roots are never rewritten.
