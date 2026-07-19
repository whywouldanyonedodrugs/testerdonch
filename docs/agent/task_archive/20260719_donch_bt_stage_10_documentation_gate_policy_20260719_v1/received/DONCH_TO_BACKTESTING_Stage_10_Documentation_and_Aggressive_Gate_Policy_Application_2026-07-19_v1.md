# Stage 10 — Apply Kraken Learnings and Aggressive Conditional-Alpha Gate Policy

```text
task_id: donch_bt_stage_10_documentation_gate_policy_20260719_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
new_data_acquisition: no
Capitalcom_payload_access: no
commit_authorized: yes — documentation/registry policy application only
push_authorized: yes — non-force under the standing workflow
```

## Objective

Apply a documentation-only and registry-only update that:

1. records all durable lessons from Stage 7C, Stage 8A-8C1 and Stage 9;
2. preserves every terminal machine decision and run root unchanged;
3. introduces the approved aggressive conditional-alpha gate-routing policy for future tasks;
4. updates backtesting manuals, agent guidance, continuity and registries;
5. creates complete local and Drive evidence of the application;
6. performs no economic computation.

## Current state to verify

```text
expected latest main after Stage 9:
    3ea0d320d71716a5c0890f4c924ed924224beda2

KDA01 terminal:
    KDA01_level3_repaired_no_primary_pass_stop

KDA02 terminal:
    KDA02_level3_no_primary_pass_stop

Stage 9 artifact manifest hash:
    2f19fc2e6d87d78706215d425ece94cd3629c05559e06a88d94da6e8d0b289f2

Stage 9 compact ZIP hash:
    195ab09fbcd5f5cb25ca4889be95c183f9e76ee476aaa7d8bef1045e927507e0
```

Verify repository root, `AGENTS.md`, branch/commit/remotes, working tree, current manuals and registries, task archives, supported commands and Drive target. Use an isolated worktree. Leave unrelated Capital.com files untouched.

## Files supplied by Donch

Archive and verify exact received bytes and SHA-256 for:

```text
00_READ_FIRST_Project_Source_Map_2026-07-19_rev4.md
02_STATE_Master_Continuity_Brief_2026-07-19_rev11.md
03_STATE_Current_Research_Decisions_2026-07-19_rev3.md
12_MANUAL_Test_and_Evidence_Standards_2026-07-19_rev3.md
13_GUIDE_Backtest_Claims_and_Review_2026-07-19_rev3.md
24_METHOD_Research_Efficiency_and_Search_Discipline_2026-07-19_rev2.md
27_POLICY_Aggressive_Conditional_Alpha_and_Gate_Routing_2026-07-19_rev1.md
28_STATE_Kraken_Derivatives_Analytics_and_Research_Learnings_2026-07-19_rev1.md
29_REGISTRY_Kraken_Derivatives_Learnings_and_Context_Candidates_2026-07-19_rev1.csv
30_REGISTRY_Gate_Routing_Matrix_2026-07-19_rev1.csv
BACKTESTING_AGENT_OPERATING_AND_ARCHIVE_INSTRUCTIONS_2026-07-19_rev4.md
```

Do not assume these paths already exist in the repository. Map them surgically to the repository's current documentation architecture.

## Binding policy

Past run decisions remain immutable. The new policy changes future research routing only.

Future tasks separate:

```text
integrity
economic relevance
generality/context
controls
independent validation
deployment
```

A failed unconditional year-concentration gate may route an economically positive object to:

```text
conditional_context_candidate_unvalidated
```

without converting the historical run to a pass.

Every future economic contract must declare:

```text
payoff_archetype
intended_claim_scope
hard_gates
routing_diagnostics
control_eligibility
independent_evidence_requirement
```

## KDA01 and KDA02 records

Preserve exactly:

```text
KDA01:
    KDA01_level3_repaired_no_primary_pass_stop

KDA02A:
    KDA02_level3_no_primary_pass_stop
```

Add or update separate routing records:

```text
KDA02 negative completed-purge reversal 6h:
    conditional_context_candidate_unvalidated
    post_hoc_context_hypothesis

KDA02 active purge continuation:
    mechanically_unavailable, not economically rejected

KDA02B OI vacuum:
    candidate_library_only; outcomes unopened

KDA02C isolated-versus-systemic purge breadth:
    post_hoc hypothesis presence; no evidence

KDA03:
    translation_candidate_unreviewed
```

Do not add a profitable, passing, validation or deployment label.

## Required repository outputs

At minimum create or update:

```text
docs/QLMG_PERP_PROJECT_STATE.md
current continuity and research-decision records
family/hypothesis registry
attempt/multiplicity registry
run/evidence registry
data-capability registry
defect/repair registry
agent/manual source map
research gate-routing policy
derivatives-state lessons document
machine-readable gate-routing matrix
machine-readable KDA learning/context register
```

Use existing repository names and schemas where they are authoritative. Preserve old rows and provenance.

Add a machine-readable policy object, for example:

```text
docs/agent/RESEARCH_GATE_ROUTING_POLICY.json
```

It must version the approved statuses, payoff archetypes, hard gates, routing diagnostics and no-retroactive-promotion rule.

Do not modify strategy code, loaders, simulators, data or run roots.

## Validation

Verify:

- all named Stage 7C/8/9 roots and hashes referenced by the documentation;
- no terminal decision was changed;
- old source supersession is explicit;
- no outcome or market payload reader is invoked;
- registry schemas remain valid;
- documentation links pass;
- source map resolves;
- JSON/CSV parse successfully;
- secret scan and `git diff --check` pass;
- independent review confirms the policy does not weaken authority, protected or validation gates.

## Independent review questions

1. Are integrity gates still hard?
2. Does year concentration route rather than retroactively pass KDA02?
3. Are payoff-archetype exceptions prospective only?
4. Are post-hoc context hypotheses labelled and validation-capped?
5. Are KDA01/KDA02 terminal decisions and roots unchanged?
6. Are controls still separately approval-gated?
7. Does the policy permit opportunity capture without enabling same-sample rescue?

## Archive and handoff

Archive:

```text
TASK_SPEC.md
DONCH_SOURCE_MANIFEST.json
PLAN.md
DECISIONS_AND_PROGRESS.md
CHANGED_FILES.md
COMMANDS_AND_RESULTS.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

After review:

- create task-scoped documentation/registry commit(s);
- non-force push under the standing workflow;
- upload the compact closed package with `drive_handoff: approved_default`;
- round-trip verify bytes and SHA-256;
- retain the local archive.

## Stop conditions

Stop if:

- repository state or authoritative roots diverge;
- a machine decision would be overwritten;
- strategy code or market data would change;
- protected or Capital.com payloads would be opened;
- the policy is interpreted as validation or deployment permission;
- registry migration becomes destructive;
- verification fails.

## Final response

```text
status:
actual_starting_commit:
terminal_decisions_preserved:
source_files_received_and_verified:
policy_version_and_hash:
payoff_archetypes_registered:
gate_routes_registered:
KDA_learnings_registered:
registries_updated:
documentation_updated:
strategy_code_changed: no
economic_outputs_computed: no
protected_rows_opened: no
tests_and_review:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
next_research_route:
human_approval_required:
```
