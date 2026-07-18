# Data and Protected-Period Rules

This document does not authorize an economic run.

## Fixed boundaries

```text
approved research platforms: Kraken derivatives; manifest-authorized Capital.com instruments
rankable train start: 2023-01-01T00:00:00Z inclusive
rankable train end: 2026-01-01T00:00:00Z exclusive
protected period: 2026-01-01T00:00:00Z onward
paid historical vendor data: prohibited
July 2026 Kraken capture use: execution_calibration_only
Capital.com 2026+ default purpose: data_engineering_only
```

Apply the interval at the earliest safe input boundary and enforce it again before signals, controls, outcomes, summaries, and package export. Report counts at each gate. A zero count in a final package does not prove upstream readers were physically separated.

## Protected data

- Do not score, tune, rank, choose thresholds, build controls, select a portfolio, or validate from protected outcomes.
- Do not read protected strategy outcomes merely to confirm that code would exclude them.
- Use metadata, schemas, fixtures, or synthetic boundary rows for mechanical verification.
- Keep pre-2026 research files physically separable from 2026 engineering files where possible.
- Filter funding before protected rows enter strategy-processing memory when the storage layout permits it.
- Treat a current 2026 roster used to choose historical symbol directories as a survivorship cap, even when no 2026 return enters the model.
- Do not change the cutoff without a cited formal policy change and explicit human authorization.

## Venue and universe

- No Bybit row may enter an active Kraken output.
- No row from one platform may enter another platform's adapter or output without an explicitly approved directed source-to-target contract.
- A rankable payload reader requires a recognized platform, selected adapter, rankable purpose, provable interval, schema hash, and content hash. Mixed or externally unrankable files fail closed.
- No row before the train start may enter rankable research.
- No instrument may signal before its official listing or availability time.
- Do not call a universe survivorship-free without point-in-time listings, delistings, suspensions, expiries, settlements, and historical eligibility evidence.
- If lifecycle ends are unknown, state the cap and use a declared cohort rule. Do not fill unknown ends from the current roster.
- Capital.com acquisition scope does not authorize an all-instrument economic search. See `docs/CAPITALCOM_DATA_CONTRACT.md`.

## Price and cashflow roles

- Last/trade price governs fills.
- Mark price governs unrealized PnL, margin, and liquidation.
- Index/oracle price validates anchoring and funding where available.
- Funding is signed cashflow on notional at actual venue settlement boundaries after entry.
- Exact, mixed, imputed, and zero-boundary funding partitions remain distinct.
- Imputed funding may affect outcome cashflow only. It cannot activate a signal, gate eligibility, or increase evidence level.
- Missing index/spot data cannot be relabeled from mark-minus-trade.
- Capital.com OTC CFD bid and ask are distinct quote objects, not exchange trades. Hypothetical buys use ask and sells use bid unless a separately frozen execution contract says otherwise; midpoint is not the default.

## Point-in-time and identity

Require:

```text
feature_available_ts <= decision_ts
```

Freeze the candidate economic address before outcomes, normally including symbol, decision time, entry, stop, risk denominator, exit policy, and maximum exit. Freeze real control addresses before outcomes. Detect duplicate candidate addresses, duplicate symbol-decision candidates, and duplicate control addresses.

Use actual executable exits for chronological non-overlap. Do not use maximum-hold preblocking. Require a complete skip ledger and deterministic replay.

## Execution and boundaries

- Do not use same-bar touch fills, optimistic queue assumptions, or intrabar ordering that bars cannot establish in Tier-1 evidence.
- Do not sample or cap events in rankable output.
- Do not force-close an open position at a train boundary.
- Drop or censor boundary-crossing intervals under a predeclared rule, including funding and controls.
- Do not treat summary rows, pooled definitions, projected aggregate means, or exit fanout as independent trades.

## Current evidence limits to preserve

As of the 2026-07-16 audit, amended by the bounded 2026-07-17 Stage 7A retention probe:

- official 5-minute trade and mark bars covered the acquired 2023–2025 Kraken roster;
- exact funding was materially incomplete before late June 2025;
- no full historical index/spot, OI, basis, liquidation, CVD, spread, liquidity, or slippage series has been acquired;
- the official public Futures analytics endpoint returned reproducible bounded 2023, 2024, and 2025 sample rows for BTC/ETH open interest, liquidation volume, and futures basis, but its funding cells were empty and value units remain unverified; this is source-retention evidence only, not full-history or rankable-data authority;
- the universe was not proven survivorship-free and lifecycle ends were unknown;
- a valid bounded `interval=3600` analytics probe is complete for the frozen 24-cell Stage 7A matrix; broader retention, pagination, units, and full-universe coverage remain untested.

Re-verify these facts in the current repository before relying on them. Absence in the old audit is not proof that data remains absent.
