# Known Failure Patterns

Use this list as a pre-change and review checklist. Repair the mechanism and add a reproducing test where feasible.

## Candidate-state and boundary defects

### Maximum-hold preblocking

Failure: suppress later signals using a nominal maximum hold before the definition's actual exit is known.

Required response: generate a parent-neutral raw signal tape, simulate each definition chronologically, use actual executable `exit_ts` for non-overlap, and reconcile accepted and skipped rows.

### Artificial sample-end close

Failure: close an open position at the train boundary and count the manufactured result.

Required response: apply the frozen drop or censor rule to entry, hold, outcome, funding, and matched controls.

### Same-bar heroics

Failure: infer touch order, fill priority, or stop/target ordering that OHLCV bars cannot establish.

Required response: use conservative ambiguity rules or cap the evidence level.

## Identity and inference defects

### Outcome-before-freeze

Failure: choose candidates, parent policies, controls, or identities after seeing outcomes.

Required response: freeze and hash candidate and control economic addresses before outcome analysis; require deterministic replay.

### Named controls without unique controls

Failure: count labels as independent controls when they share the same economic address, or use placeholder/projected controls.

Required response: hash control addresses, report unique-address coverage, matched and actual unmatched subsets, and mechanism relevance.

### Rows treated as trades

Failure: treat summary rows, projected aggregate means, pooled definition rows, or exit fanout as independent event returns or a portfolio.

Required response: identify canonical economic episodes and keep summaries separate from event ledgers.

### Rankable sampling

Failure: cap or sample events for convenience and retain rankable language.

Required response: process the complete frozen population or mark the output diagnostic and non-rankable.

## Data defects

### Current-roster backfill

Failure: seed historical downloads or eligibility from the current live roster and call the universe survivorship-free.

Required response: use point-in-time lifecycle authority or declare a capped bar-existence cohort with unknown omissions.

### Pre-listing and availability leakage

Failure: use a feature or instrument before its official availability.

Required response: enforce listing and `feature_available_ts <= decision_ts` gates with boundary fixtures.

### Funding misuse

Failure: use imputed funding to activate a signal, mix exact and imputed evidence, or load protected funding before filtering.

Required response: keep signal gating independent of imputation, partition funding evidence, and filter before strategy processing.

### Price-role collapse

Failure: use one price series for fills, margin, liquidation, index anchoring, and funding.

Required response: preserve last/trade, mark, index, and signed funding roles; state missing inputs and evidence caps.

### Stale venue authority

Failure: apply old Bybit-primary guidance to active research.

Required response: use Kraken only for active output and retain old Bybit material as provenance.

## Package and operations defects

### Hash pass mistaken for release readiness

Failure: mark a package ready because present files hash correctly while required evidence is absent.

Required response: validate required content, schema, authority, protected-period exclusion, test counts, reproducibility, and disclosed blockers separately.

The 2026-07-16 package exemplifies this distinction: its recorded hash validation passed, while its status remained `blocked_by_protocol_issue` because raw event-window verification extracts were absent. Test counts were blank, one source snapshot was missing, and five families lacked reproducibility hashes.

### Unverified repository assumptions

Failure: treat `/opt/testerdonch`, a directory name, a tool path, or a remembered test command as verified current state.

Required response: resolve the Git root and discover repository-owned commands and paths. Mark unknown items `REPO_DEPENDENT_PLACEHOLDER`.

### Ambiguous bundle name

Failure: create `review.zip`, `latest.zip`, or a similar name that omits date and contents.

Required response: use `qlmg_<specific-content-slug>_<YYYYMMDD>_vNN.zip` with UTC date and collision-safe version.

### Unverified remote handoff

Failure: guess a Google Drive remote, use an unconfirmed write identity, treat size equality as content verification, or overwrite an existing bundle.

Required response: require explicit upload authorization, exact destination, authorized identity, collision policy, local hash, and remote hash or documented round-trip verification. Otherwise report `remote_handoff_blocked` and keep the local ZIP.

### Evidence deletion during repair

Failure: replace or delete old roots after correcting a defect.

Required response: write a new versioned root, retain the old root unchanged, and record supersession and reason.
