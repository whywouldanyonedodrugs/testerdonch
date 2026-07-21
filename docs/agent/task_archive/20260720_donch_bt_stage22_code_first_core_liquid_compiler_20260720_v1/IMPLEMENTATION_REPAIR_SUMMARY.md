# Consolidated implementation repair

The first independent review returned ten blocking classes. One bounded
pre-outcome repair pass addressed them as follows:

1. Authority is file-, hash-, campaign-, commit-, source-, universe-, funding-,
   cache-artifact- and deserialized-frame-bound.
2. The executor runs raw inputs through family engines, accounting, non-overlap
   and aggregation; it cannot accept a pre-aggregated outcome observation.
3. A1, A2, A3, A4 and KDA02B have executable formula/state paths for every
   registered value.
4. Accounting implements completed-close/next-open paths, hourly/daily exits,
   costs, funding, adverse gaps and boundary censoring.
5. Controls transform raw/config inputs and re-enter the production dispatcher.
6. Plateau, medoid, deterministic refinement, explicit-empty folds, clustered
   bootstrap, routes, beams, deduplication and materialization are executable.
7. A2 uses only slots 1–5, exact projected parents and atomic counterparts;
   multiplicity-only duplicates are absent from the execution registry.
8. Count selection is outcome-free; benchmarks invoke complete synthetic engine
   dispatches rather than timing schema normalization.
9. The supervisor has continuous resource gates, bounded submission, atomic
   markers/state, graceful stop, retry, heartbeat/health release and recovery.
10. Finalization requires an exact review binding and uses the physical schema
    sidecar hash.

No launch is performed by Stage 22.

The first full candidate build then exposed one additional mechanical defect:
the audit writer could not encode the internal negative-infinity sentinel used
for empty inner folds as strict JSON. The arithmetic remains unchanged. The
audit now serializes each empty fold as `unavailable_empty_fold` and its
negative-infinity quantile as `negative_infinity_due_to_empty_fold`, while the
selection engine continues to retain and calculate with explicit negative
infinity internally.

## V03 independent review and second bounded repair

The exact V03 candidate was independently reviewed and blocked with twelve
findings. The review bytes are archived as
`V03_INDEPENDENT_PREOUTCOME_REVIEW.json`; the blocked V03 root is immutable.
The second repair pass made the following narrow code-owned corrections:

1. Semantic-cache files are decoded after physical SHA-256 verification and
   the decoded content, fold partition, source inventory, exact funding ZIP,
   PIT universe and threshold-population provenance are re-bound at dispatch.
2. The complete stage graph and launch CLI now exist, including lazy stages,
   A2 resolution, controls, materialization, state, retry and detached-service
   construction.
3. A1/A3 are exact-one-decision frames; all context objects carry and enforce
   exact as-of/source-close/availability timestamps.
4. A4 uses completed five-minute source formulas, the registered EMA reference,
   realized-volatility estimators, q05/q95 winsorization, MAD scaling and exact
   control transformations.
5. A4 cohort contributions are divided once, summed by cohort and averaged by
   nonempty UTC-day cohorts.
6. Rank normalization, the 0.90 side-signed funding veto, linear
   breadth/dispersion overlay and every registered context lookback are exact.
7. KDA02B rearm/barrier mechanics and matched generic controls are code-owned.
8. Every control transform is derived from parent events and the exact finite
   allocator; unavailable and duplicate controls persist explicitly.
9. Common-gate routes, stratified failed materialization, complete metrics and
   explicit empty-fold arithmetic are tested.
10. Runtime uses bounded lazy process scheduling, fail-closed process-tree RSS,
    PID-churn handling, retry, graceful stop, heartbeat and health release.
11. Launch authority binds the complete launch-tree inventory and permits only
    byte-identical reviewed code in an approved commit or publication-only
    descendant.
12. Capacity inputs come from the four-worker cache/decode/dispatcher/
    supervisor path and cannot silently raise an infeasible minimum campaign.

The audit also found two semantic dependencies not enumerated in V03: fold
threshold population identities must include every formula-defining lookback
or estimator axis, and Stage-19 funding rows are USD-per-contract-unit hourly
rates rather than already-normalized basis points. Both are now explicit,
tested contracts. Funding is computed with Decimal arithmetic, fractional-hour
coverage, entry-price conversion, both registered alignments and per-hour gap
allowances.

The fresh independent review is instructed to treat production source-cache
construction, representative workload measurement, complete forensics and
terminal route/reconciliation coverage as blocking if the implementation still
leaves any post-approval invention or unattended-safety uncertainty.
