# Decisions and Progress

- Verified clean synchronized `main` at `f0c12311f31c24c2683d180166450bbacf7389bf`.
- Confirmed C01/C02 Level-3 stops and C03/C16 authority-unavailable terminal records remain active.
- Confirmed the broader Stage 7A v1 prompt is superseded and will not be executed.
- Official documentation verifies the symbol-first route, epoch-second `since` and `to`, interval `3600`, and exact type names `open-interest`, `funding`, `liquidation-volume`, and `future-basis`.
- Added a fixed-matrix public-source probe with pre-open URL validation, a 48-request/50 MiB cap, local-only raw storage, and no adaptive requests.
- Synthetic and applicable repository checks passed: 46 tests, zero failures.
- Executed exactly 24 first-pass and 24 replay calls; 32,526 total response bytes.
- Verified historical hourly rows in 18/24 cells: open interest, liquidation volume, and future basis for both symbols and all three years.
- Funding returned schema-valid empty responses in all 6/6 cells.
- All populated bounds were honored; 24/24 replay structures and hashes matched; protected timestamps were zero.
- Frozen decision: `partial_historical_analytics_requires_review`.
