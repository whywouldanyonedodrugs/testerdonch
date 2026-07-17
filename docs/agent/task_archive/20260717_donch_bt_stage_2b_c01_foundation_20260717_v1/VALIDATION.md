# Validation

Status: **pass**

## Scope

This was a non-economic generator and identity validation. No candidate outcome, trade return, exit, PnL, MAE/MFE, control performance, funding, capture, or protected-period payload was opened or computed.

## Generated evidence

- Candidate rows: `1,805,966`.
- Canonical same-symbol overlap episodes: `19,776`.
- Candidate symbols represented: `269` of 278 eligible cohort symbols; symbols with no qualifying diagnostic shock remain in the count matrix with zero rows.
- Attempts registered/retained: `12/12`.
- Feature-contract hash: `c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb`.
- Data-authority hash: `b51ca71c3a9cd6425ade8255e443cc76bfc5b5882ee80c3eccfc8de11bad1a1a`.
- Diagnostic tape SHA-256: `f273b6cc851341b2e3fdd49baf6ed48b39bd6a34f475b7bf129774d0df0a1efd`.
- Generator runtime / peak RSS: `14:09.60` / `4,946,856 KiB`.

## Hard checks

`VALIDATION_EVIDENCE.json` records:

- protected, pre-2023, non-Kraken rows: `0 / 0 / 0`;
- known lifecycle interval violations: `0`;
- outcome-derived columns: `0`;
- duplicate candidate IDs/economic addresses: `0 / 0`;
- candidate identity recomputation mismatches: `0` over `9,979` deterministic sample rows;
- episode symbol/start/end/member/connectivity violations: `0 / 0 / 0 / 0 / 0`;
- shock-window and episode-input contract violations: `0`;
- z-threshold, path-state, scale-minimum, and fit-minimum violations: `0`;
- trade/mark authority-reference collisions: `0`;
- registry and count-matrix sums both equal `1,805,966`.

The known current-cohort lifecycle intervals are conservatively masked as full UTC dates:

- `PF_FETUSD`: `[2024-06-07, 2024-11-29)`;
- `PF_MNTUSD`: `[2025-02-28, 2025-07-17)`;
- `PF_RIVERUSD`: `[2025-11-21, 2026-01-01)`.

## Tests

Compile passed. The focused C01, Stage 2A/2A1 authority, rankable-loader, protected-slice, and archive-guard suites passed `58/58` with zero failures and zero errors. Exact test names and output are in `TEST_RESULTS.log`.
