# Stage 2A — U2 Lifecycle Authority and Anchor Cohort

```text
task_id: donch_bt_stage_2a_u2_lifecycle_20260716_v1
repository: /opt/testerdonch
mode: direct_apply
economic_run_authorized: no
candidate_return_analysis: no
protected_outcome_access: no
capture_access: no
public_network_reads: authorized only for bounded lifecycle acquisition
```

## Objective

Create a narrow, point-in-time defensible Kraken perpetual **U2 anchor cohort** for C01 research.

Acquire and record only the public lifecycle evidence needed to establish which continuously eligible contracts can be used for the rankable interval:

```text
[2023-01-01, 2026-01-01)
```

Do not implement C01, calculate signals or returns, inspect protected outcomes, or reconstruct the complete historical Kraken universe.

## Start condition

Verify:

- repository root and active `AGENTS.md` chain;
- clean working tree;
- local `main == origin/main`;
- current machine contracts and supported commands.

Record the actual starting commit.

Do not stop merely because the commit differs from the previously reported
`c198cf0059128b2beb917eda19fbfde6695ed7a9`. If it differs, inspect the intervening commits and proceed when they contain only expected task archives, governance, or reviewed loader-boundary work. Stop only for an unexpected material code, data, or authority change.

## Authority and fixed boundaries

```text
venue: Kraken only
rankable interval: 2023-01-01 inclusive to 2026-01-01 exclusive
2026+ outcomes: protected
July 2026 capture: execution_calibration_only
paid historical vendors: prohibited
```

The current 2026 instrument roster and bar existence may support identity and coverage checks, but neither proves continuous historical eligibility. Unknown lifecycle state remains unknown.

Use this source order:

1. official Kraken machine-readable instrument or product records;
2. official Kraken listing, suspension, wind-down, delisting, settlement, and specification notices;
3. archived copies of official Kraken material when the live official page is unavailable;
4. other public sources only as a pointer to an official source, not as final authority.

No other venue may establish Kraken eligibility.

## Outcome-free cohort rule

Freeze this selection rule before inspecting lifecycle sources:

A contract may enter the U2 anchor cohort only when all are verified:

1. stable Kraken instrument identity and symbol mapping;
2. eligibility began no later than `2023-01-01`, or the contract is assigned a later verified start and the permitted research interval is explicitly shortened;
3. no unresolved suspension, post-only wind-down, delisting, settlement, or symbol-migration interval inside its claimed eligible period;
4. official trade and mark data coverage is present for the claimed interval;
5. no protected outcome data are required to establish eligibility;
6. every uncertainty is represented explicitly rather than imputed.

Exclude a contract when continuous eligibility cannot be established. Do not use price performance, candidate counts, volatility, or subsequent returns to select the cohort.

The resulting cohort may be small. Report the permitted inference boundary rather than weakening the rule to increase membership.

## Scope

### Authorized public acquisition

Public read-only requests are authorized for:

- Kraken official domains and public APIs;
- public web archives containing copies of Kraken official pages, when required;
- source headers and metadata needed to establish publication/effective dates.

Do not use paid vendors, private APIs, credentials, or capture archives.

Respect rate limits and cache the retrieved source material with URL, access time, response status, and SHA-256.

### Existing repository inputs

Inspect only:

- current instrument identity/opening-date tables;
- trade/mark coverage manifests and path metadata;
- existing lifecycle or listing records;
- relevant data-capability and authority registries;
- loader-boundary and protected-period contracts.

Do not inspect economic result tables.

### Code

Prefer the smallest approach.

- If the cohort can be built reliably through a small bounded source ledger and deterministic normalization, do that.
- Write or change code only when repeated acquisition, parsing, identity mapping, or validation requires it.
- No general lifecycle platform, data catalog, taxonomy framework, or unrelated refactor.

## Required outputs

Use existing repository conventions. Create:

```text
U2_LIFECYCLE_SOURCE_LEDGER.csv
U2_INSTRUMENT_LIFECYCLE_AUTHORITY.csv
U2_ANCHOR_COHORT.csv
U2_EXCLUSIONS_AND_UNCERTAINTY.csv
U2_COVERAGE_AND_CLAIM_BOUNDARY.md
U2_VALIDATION_REPORT.md
ARTIFACT_MANIFEST.json
```

Required lifecycle fields:

```text
instrument_id
canonical_asset_id
Kraken_symbol
contract_type
eligible_start_utc
eligible_end_utc
status_intervals
listing_source
status_or_end_source
source_publication_utc
source_access_utc
source_sha256
trade_coverage_start_utc
trade_coverage_end_utc
mark_coverage_start_utc
mark_coverage_end_utc
identity_confidence
lifecycle_confidence
unknown_fields
included_in_U2
inclusion_or_exclusion_reason
permitted_claim
```

`U2_ANCHOR_COHORT.csv` must contain only included contracts and a stable cohort/version hash.

## Tests and verification

Use synthetic fixtures for parser and interval logic. Real public source acquisition is permitted, but no real protected outcome payload may be opened.

Verify:

- source hashes and URLs;
- duplicate and conflicting lifecycle records;
- interval ordering and overlap;
- symbol/asset identity;
- no eligibility before official start;
- unknown end/status does not become active by default;
- trade and mark coverage lies inside the claimed eligible interval;
- no pre-2023 or 2026+ row enters a rankable cohort interval;
- no non-Kraken contract enters;
- deterministic cohort hash;
- zero economic metrics or candidate returns.

Run focused tests and the smallest relevant repository-supported regression suite.

## Acceptance criteria

The task succeeds when:

```text
public lifecycle source ledger: complete for every considered contract
U2 inclusion rule: frozen and outcome-free
included contract identities: verified
eligible intervals: explicit
unknown lifecycle fields: preserved
trade/mark coverage: checked
protected outcomes opened: zero
economic outputs computed: zero
cohort and source artifacts: deterministic and hashed
independent review: approve
```

A valid completion may conclude that no sufficiently defensible cohort exists. Do not weaken the authority standard to avoid that result.

## Integration authorization

After tests and independent review pass:

- create one task-scoped commit;
- fast-forward `main` only if it has no unexpected intervening change;
- push `main` to `origin/main` without force.

This authorization covers only the lifecycle authority, cohort artifacts, minimal supporting code/tests, required task archive, and factual updates to the lifecycle/data-capability authority records.

Do not modify hypothesis-family decisions, strategy code, package-protocol artifacts, C01/C02/C03 implementations, or governance instructions.

## Deferred work

Do not address:

- external-review package protocol gaps;
- effective-trial or canonical cross-family episode infrastructure;
- C01 residual/path feature implementation;
- C02 spot/reference acquisition;
- C03 breadth computation;
- funding reconstruction;
- capture;
- instruction simplification and stale-state cleanup;
- any economic run.

## Next-task decision

At completion recommend exactly one:

```text
Stage_2B_C01_identity_multiplicity_and_feature_foundation
```

if U2 is sufficient, or the smallest bounded lifecycle-repair task if it is not.

Do not implement the next task.

## Final response

```text
status:
actual_starting_commit:
public_sources_accessed:
contracts_considered:
contracts_included:
contracts_excluded:
U2_cohort_version_and_hash:
eligible_interval_summary:
main_uncertainties:
files_changed:
tests_and_results:
protected_outcomes_opened: no
economic_outputs_computed: no
local_commit:
main_commit_after_integration:
origin_main_updated:
task_archive:
artifacts_and_sha256:
authority_records_updated:
permitted_C01_claim_boundary:
remaining_blockers:
next_recommended_task:
human_approval_required:
```
