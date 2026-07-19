---
status: approved design direction; verified implementation pending
date: 2026-07-19
revision: 1.0
scope: batched unattended hypothesis research, campaign approvals, checkpoints, resource safety and quality controls
authority: explicit human direction of 2026-07-19 and operating approval rules
supersedes: one human approval and chat exchange for every routine phase transition
provenance: Stage 7C unattended acquisition; Stage 8-12 archives; task/archive governance
known limitations: economic campaign execution still requires one exact campaign-level human approval
---

# Unattended Multi-Hypothesis Research Campaigns

## Decision

Several hypotheses may be researched in one unattended campaign when their source authority, folds, search spaces, selection logic, costs, phase permissions and stop conditions are frozen in a campaign manifest.

One exact human approval may authorize multiple named economic runs. This does not waive approval; it moves approval to the bounded campaign level.

## Campaign modes

```text
measurement_only
registered_development_only
nested_development_and_outer_evaluation
controls_only
prospective_shadow_only
```

A campaign may contain different modes by hypothesis lane.

## Required campaign manifest

```text
campaign_id and version
repository/data/feature hashes
hypothesis and family IDs
programme-exposure class
fold schedule and embargo
search lanes and budgets
feature/search-space manifests
candidate-beam size and tie-break
payoff archetype options allowed before development
cost and execution contracts
multiplicity method
phase permissions
per-family and global stop conditions
resource budgets
review/checkpoint requirements
archive and Drive target
```

## Automatic progression

An agent may proceed without human intervention when:

- the next phase is explicitly authorized in the manifest;
- all phase acceptance checks pass;
- no authority, protected, Git, schema or reproducibility issue exists;
- the exact selection algorithm chooses the next candidates;
- independent review required by the manifest approves.

Routine negative results do not stop unrelated hypothesis lanes.

## Stop hierarchy

### Global stop

Stop the full campaign for:

- source/manifest mismatch;
- protected exposure;
- common timestamp or execution defect;
- unsafe Git or storage state;
- deterministic replay failure affecting shared infrastructure.

### Family stop

Stop only the affected family for:

- no mechanically valid opportunities;
- no economically relevant development candidate;
- search-budget exhaustion;
- underidentified mechanism;
- family-specific code or data defect.

Other lanes continue when independence is proven.

## Unattended state and recovery

Use transactional state with:

```text
campaign
hypothesis
phase
fold
experiment cell
candidate
review
artifact
status
attempt count
start/end time
hashes
error and recovery action
```

Requirements:

- idempotent restart;
- atomic artifacts;
- heartbeat and progress summary;
- memory/disk limits;
- per-family resource accounting;
- immutable explored-cell registry;
- no silent scope reduction;
- final reconciliation of every planned and executed cell.

## Quality controls

- independent pre-outcome review of the campaign manifest and implementation;
- automatic synthetic and bounded real-data tests;
- independent freeze review before each outer fold or one reviewed deterministic freeze engine;
- post-run recomputation of selected candidates, outer results and multiplicity statistics;
- complete archive and Drive handoff.

## Recommended batching

Hypotheses sharing data and feature infrastructure may be chained. Hypotheses with materially different authority or manual source-repair needs should use separate lanes or campaigns.

Current recommendation:

```text
Derivatives-state campaign:
    KDA02B OI vacuum
    KDA02C purge breadth
    downside completed derivatives-state rejection redevelopment

Separate catalyst campaign:
    C17 executed catalyst state
```

C17 may share one outcome-free readiness campaign but should not be forced into the same economic batch as high-frequency derivatives states.
