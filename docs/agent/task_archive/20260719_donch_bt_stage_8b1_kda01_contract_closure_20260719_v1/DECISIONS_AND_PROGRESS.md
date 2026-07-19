# Decisions and Progress

- Verified main and origin at the required Stage 8B commit; isolated all work from three unrelated untracked main-checkout paths.
- Reconciled all 52 Stage 8B archive objects and 748 cache objects with zero missing or mismatched entries.
- Clarified that `569bb1...` is the deterministic embedded manifest-content hash; the JSON byte-level SHA-256 is separately recorded as `ee7db7...`.
- Frozen execution logic uses only authorized bar timestamps and rejects protected-boundary targets before any protected file can be opened.
- First full invocation failed before data processing due to script import bootstrap and was repaired using the established repository-root pattern.
- Second invocation failed closed on a non-bar acquisition envelope; the loader now skips such rows by schema before `read()`, matching the accepted Stage 8B authority behavior.
- The first broad relevant-suite invocation found four environment errors because the isolated worktree did not contain the repository-ignored sealed metadata fixture. The existing metadata-only fixture was linked read-only from the main checkout without opening protected payloads; the final expanded relevant suite passed 222/222.
- The authoritative build completed in 131.55 seconds with 1,240,416 KiB peak RSS, zero protected rows opened, and zero economic outputs computed.
- A full deterministic replay reproduced the cluster tape, summaries, execution counts/rejections, contract JSON, and definition register byte-for-byte.
- Final mechanics reconcile 102,136 source events into 204,272 definition/horizon execution records: 183,744 accepted and 20,528 rejected or skipped, including 20,473 actual-position overlaps and 55 missing exit bars.
- All eight primary timeout definitions pass the frozen mechanical feasibility gates. The final status remains `ready_for_human_KDA01_Level3_run_approval`; no Level-3 economic run was launched.
