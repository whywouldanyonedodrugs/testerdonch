# First-Wave Research Readiness Rerun

## Decision

Status: `blocked_by_remaining_rankable_reader_paths`.

The demonstrated `load_symbol_bars()` and `load_funding()` defects are repaired under the corrected authority semantics. The overall first-wave firewall is still not ready because three active sibling readers bypass the new helper and no current real-data authority map is bound through `data_paths()`. No Stage 2A implementation or economic work is authorized by this rerun.

## A. Repository and archive

- Repository: `/opt/testerdonch`.
- Task branch: `fix/rankable-loader-boundary-20260716`.
- Authorized base, local `main`, and `origin/main`: `992e7928d0dd948c0bb3f3fc3c74b1095648df1b` at task start.
- Applicable chain: root `AGENTS.md`; `docs/agent/REPOSITORY_MAP.md`; `DATA_AND_PROTECTED_PERIOD_RULES.md`; `RUN_AND_ARTIFACT_CONTRACT.md`; `KNOWN_FAILURE_PATTERNS.md`; machine contracts and finalized manifests.
- Archive convention: `docs/agent/task_archive/<YYYYMMDD>_<task-id>/`.
- Original readiness archive: preserved and 22/22 manifest entries hash-verified.
- Supported environment: repository venv with unittest. The repository-map pytest command is stale for this environment because pytest is not installed; no dependency change was made.
- Drive handoff: not authorized and not attempted. No remote was tested during this task.

## B. Package and evidence status

The external-review package remains `blocked_by_protocol_issue` with `package_release_ready=false` and one unresolved schema issue. The full archive and core ZIP exist. Raw verification extracts, 14-family test/failure counts, one prior-high source snapshot, five lineage hash sets, and TSMOM MAE/MFE remain unresolved. The actual full-archive size is 20 bytes smaller than the size recorded in `decision_summary.json`; this metadata discrepancy is not repaired.

No current narrative reviewed here overstates the machine status. See `PACKAGE_PROTOCOL_STATUS.md`.

## C. Protected-data firewall

The corrected authority interpretation was applied:

- protected/mixed/calibration/prospective/external/unknown file authority is rejected before payload read;
- pre-2023 and non-Kraken rows are rejected before rankable downstream use, without imposing universal pre-open rejection;
- exact-funding file authority is required and imputed/signal-ineligible rows cannot leave the exact loader.

The two repaired loaders pass synthetic reader/downstream-spy tests. Overall status remains blocked because `load_symbol_signal_bars()`, `load_symbol_rank_close_window()`, and `a1_load_symbol_bars_window()` remain active and unguarded. Existing real files also lack a bound `rankable_file_authority` map and therefore fail closed. See `PROTECTED_BOUNDARY_VERIFICATION.md`.

## D. Degrees of freedom and canonical episodes

- Effective-trial support: `partial`. Family-local effective-trial count artifacts exist, but no current central immutable C01 attempt registry was found that records every definition, dimension, killed candidate, and follow-up.
- Canonical cross-family episode identity: `missing`. Rev8 explicitly records this gap; family-local overlap does not satisfy it.
- Exact versus nearby episode overlap: family-local implementations exist, but no one tested cross-family authority supports the first-wave requirement.
- C01 may be registered as a materially new family only after its residual/path definitions, nearest prior families, same-sample prohibitions, and complete attempted-definition ledger are frozen outcome-free.

## E. U2 universe readiness

Status: `blocked`.

No candidate symbol is admitted to a U2 anchor cohort in this report. The current instrument snapshot and opening dates cover current survivors, while all 325 lifecycle interval ends are unknown and previously delisted contracts may be absent. Reading the 2026 instrument payload was neither necessary nor authorized. BTC/ETH references in the research plan are proposed mechanism inputs, not proof of continuously eligible Kraken contracts.

Maximum permitted claim: the acquired bars describe an observed current-survivor cohort with known omissions and unknown lifecycle ends. They do not prove continuous eligibility or survivorship freedom.

Smallest remedy after firewall closure: an outcome-free official lifecycle-backed inventory with identity, verified start, conservative end/status, coverage, uncertainty, and inclusion/exclusion reason for each proposed U2 symbol.

## F. C01 readiness

| Component | Status | Current evidence and limit |
|---|---|---|
| causal BTC residual | partial | H43 provides a nearby prior, not a frozen C01 feature contract |
| causal BTC+ETH residual | missing | no accepted multivariate C01 fixture found |
| fixed rolling window | partial | generic rolling support exists; C01 boundary fixture missing |
| path efficiency | partial | A1 path-smoothness is same-sample prior only |
| jump/largest-bar contribution | missing | no accepted C01 implementation found |
| parent-market state | partial | reusable PIT regime helper exists; C01 fixture missing |
| realized-volatility state | partial | generic implementation exists; C01 contract not frozen |
| causal feature hashing | partial | generic hash infrastructure exists; C01 fields not frozen |
| candidate identity freeze | partial | generic evidence contracts exist; C01 identity absent |
| canonical episode identity | missing | no cross-family canonical layer |
| control harness | partial | real-control infrastructure exists; C01 controls not frozen |
| attempt/multiplicity registration | missing | no complete central C01 attempt ledger |
| count-only reporting | partial | generic reporting exists; C01 sign/path report absent |
| trade/mark separation | partial | central loader separates roles; sibling readers/firewall remain open |
| next-bar all-taker simulation | partial | generic support exists; C01 fixture absent |

Closest prior paths are H43 residual-lag logic, A1 path smoothness, parent-regime helpers, and generic evidence/control utilities. They are reusable components only; copying their selected thresholds or event identities would create same-sample contamination.

The smallest generator-adjacent task remains an outcome-free common-feature fixture after U2 and the remaining reader firewall close. It must freeze one primary plus at most one robustness residual/path specification, hashes, candidate/episode identity, and count-only outputs without returns.

## G. C02 readiness

Status: `blocked`.

No acquired official public Kraken 2023-2025 spot/index directory was found under the historical root, consistent with capability registry `KRAKEN_DATA_006`. Generic path and alias utilities are only partial synchronization support; there is no spot-perp identity map or synchronized panel.

The smallest later acquisition scope is outcome-free BTC, ETH, and the eventually verified U2 cohort with source/request manifests, raw/normalized hashes, symbol map, timestamp semantics, gaps/duplicates, PIT eligibility, and protected-row audit. No network request is authorized by this report.

## H. C03 readiness

Status: `partial_but_blocked`.

PIT timestamps, parent context, rolling volatility, and generic breadth/ranking utilities exist. Historical cohort membership is still a current-roster/bar-existence proxy with unknown delistings, suspensions, settlements, and lifecycle ends. Stable dated sector authority is limited; current narrative tags cannot be backfilled.

C03 cannot yet be applied as a decision-bearing continuous layer. After U2/PIT authority exists, one predeclared breadth/dispersion definition may be evaluated incrementally on a separately frozen C01/C02 tape without cutoff search.

## I. Exactly one next task

`Stage_1B_remaining_rankable_reader_boundary_closure`.

This is narrower than Stage 2A and limited to the three identified sibling readers plus the smallest existing-manifest authority handoff. It must use synthetic spies, remain outcome-free, and stop if real protected payload access or a generalized catalog is required. See `NEXT_TASK_RECOMMENDATION.md`.

## Proposed registry corrections

No substantive registry was changed. Later authorized work should record:

1. the two repaired loader paths and their 8-test spy evidence;
2. the three still-unguarded sibling readers;
3. the absent current file-authority binding;
4. U2 lifecycle uncertainty;
5. incomplete C01 multiplicity and canonical episode support;
6. the external-package 20-byte size discrepancy alongside existing blockers.

## Safety record

- Protected payloads opened: no.
- Candidate returns or economic outputs computed: no.
- Capture accessed: no.
- Data acquired: no.
- Network or remote writes: no.
- Existing result roots or prior archives modified: no.
