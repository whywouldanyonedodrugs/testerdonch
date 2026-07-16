# Readiness Rerun Report

Status: `blocked_by_remaining_rankable_reader_paths`.

The archived first-wave readiness task was rerun under the corrected authority interpretation. The demonstrated `load_symbol_bars()` and `load_funding()` paths now pass all synthetic reader and downstream boundary tests. Overall readiness remains blocked because three active sibling raw readers bypass the helper and existing real historical manifests are not yet bound into `rankable_file_authority`.

Other unchanged blockers:

- external-review package remains `blocked_by_protocol_issue` and not release-ready;
- U2 continuous eligibility is unproven;
- central C01 effective-trial registration is incomplete;
- canonical cross-family episode identity is missing;
- C02 official Kraken spot/reference history is absent;
- C03 PIT cohort membership is blocked by lifecycle authority.

Exact rerun artifacts are under:

`docs/agent/task_archive/20260716_donch_bt_loader_boundary_repair_20260716_v1/readiness_rerun/`

Next task: `Stage_1B_remaining_rankable_reader_boundary_closure` only. No economic run is authorized.
