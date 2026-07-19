# KDA03 Independent Post-Run Review

Status: `approved_postrun`
Approved: `true`
Terminal status: `KDA03_level3_routes_assigned`
Run commit: `8f365a47625e338b5a518a22866d41e41244e458`

The completed KDA03 Level-3 run is approved as a reproducible policy-v1.0 routing result. Approval means the frozen run and reported routes are mechanically correct; it is not a strategy pass, validation claim, control authorization, or deployment approval.

## Decision

- Twelve primary definitions were routed.
- Eleven primary definitions are `translation_rejected` because the frozen equal-market-day base mean or median is nonpositive.
- `kda03_v1_primary_negative_completed_basis_impulse_rejection_timeout_6h` is `sample_limited_prospective_candidate`: equal-market-day base mean `+9.1569766330` bps, median `+2.9322973326` bps, 95% bootstrap lower `-8.2952626116` bps, and stress mean `-8.8430233670` bps across 1,839 trades and 691 market days.
- No primary definition is control-eligible. Controls executed: `0`.
- Robustness definitions remain diagnostics only and cannot rescue a primary.

## Independent Recomputation

### Frozen identity and execution

- The run audit matches frozen contract hash `5f1abdc7e21ab9e3a6851b21930a7af745cbb865be334f8be45c9a561d2411e7`, the approved contract/tape/register hashes, runner hash, calculation-module hash, market-manifest hash, official trade-bar authority hash, and timestamp authority hash.
- All nine approved pre-outcome hashes still match the files at commit `8f365a4`; the worktree contained no source mutation during review.
- The accepted and rejected outputs reconstruct the complete frozen schedule: `227,078` definition-event records, `199,787` accepted, and `27,291` rejected. The 18 shared schedule fields match the frozen timestamp eligibility tape for every row.
- Definition-event duplicates, accepted definition-local/symbol-local actual-exit overlaps, protected decisions, and protected exits are `0/0/0/0`.
- All `199,787` Level-3 economic addresses recompute exactly and are unique. Candidate identity survived the funding round trip.

### Official opens, direction, and costs

- Every entry and exit open was independently reread from the manifest-authorized official Kraken PF 5-minute trade bars for all 55 traded symbols.
- Entry-open mismatches: `0`; exit-open mismatches: `0`; invalid/nonpositive opens: `0`; price rejections: `0`.
- Gross return arithmetic recomputed exactly as `side * (exit_open / entry_open - 1) * 10,000` for all trades.
- Base and stress costs recomputed exactly as gross minus 14 bps and gross minus 32 bps. Maximum absolute discrepancy for gross/base/stress was `0/0/0` bps.
- Frozen event direction and priced side mismatches: `0`.

### Estimand, bootstrap, concentration, clusters, and context

- Equal-market-day weights and per-trade estimand contributions recomputed exactly for all `199,787` trades.
- The `20,158` market-day return rows recomputed exactly: trades are equal-weighted within each UTC market day, then market days receive equal weight within each definition.
- All `240,000` bootstrap draws recomputed with 10,000 resamples per definition and seed `20260719`; every draw and all 24 lower/upper summaries match.
- All 24 definition metrics, 24 policy gate rows, 24 concentration rows, 24 decisions, 48 six-hour/parent-episode sensitivity rows, and 72 basis-context rows reproduced within `1e-12` numerical tolerance.
- All `21,494` market-day, symbol, and year contributor rows—including contributor identities, trade counts, and contributions—recomputed exactly.
- The routed positive definition is not explained by one event: its largest positive event is `2.3076%` of positive event contribution, across 691 days, 1,839 parent episodes, and 49 symbols. No price, identity, schedule, arithmetic, or funding defect was found.
- `NaN` positive-year shares on economically negative definitions mean there was no positive-contribution denominator; route priority had already assigned `translation_rejected` from nonpositive mean/median.

### Funding diagnostics

- Funding remains diagnostic and excluded from economic gates and routes.
- `631,092` unique definition-event-boundary rows reconcile to all `199,787` trades with no unknown or duplicate address-boundary pair.
- Exact/imputed boundary counts, central/conservative/severe cashflow sums, and `fully_exact`/`fully_imputed`/`mixed` partitions match every trade with zero discrepancy.
- The 60-row funding partition summary reproduces exactly: `54,687` exact and `576,405` imputed boundaries. Protected funding boundaries: `0`.

### Policy routes and scope boundaries

- Policy-v1.0 flags and priority routes recompute exactly for all 24 definitions.
- Primary routes: 11 `translation_rejected`, 1 `sample_limited_prospective_candidate`, 0 other routes.
- Primary control-eligible definitions: `0`.
- Control artifacts/columns, Capital.com artifacts/columns, and protected timestamps across run parquet outputs: `0/0/0`.
- The audit records `protected_rows_opened: 0` and `controls_executed: false`.

## Run-Once and Mutation Review

Repository evidence shows one recorded authorized run: one uniquely named KDA03 Level-3 output root, one `RUN_AUDIT.json`, no sibling KDA03 Level-3 attempt root, and the runner's fresh-output-root guard. The audit records commit `8f365a47625e338b5a518a22866d41e41244e458`. Frozen source, contract, definitions, events, parents, and gates remain identical to the approved pre-outcome hashes, so no outcome-conditioned definition, threshold, horizon, context, or code mutation occurred.

## Findings

No blocking, high, medium, or low actionable finding remains.

## Remaining Claim Limits

The cohort remains current-roster/lifecycle capped and is not survivorship-free. OI history is truncated in early 2023, analytics units remain `inferred_authoritative_v1`, and funding is predominantly imputed and excluded from gates. KDA03A is a directional reference-led PF-futures proxy, not a spread or arbitrage test. The single positive route is sample-limited and unvalidated; controls and independent/prospective validation require separate authorization.
