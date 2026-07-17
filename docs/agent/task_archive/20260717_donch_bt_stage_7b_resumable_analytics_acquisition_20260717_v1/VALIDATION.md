# Validation

- Corrected Phase A ledger: 1,080/1,080 jobs complete; attempts equal jobs; retries zero.
- Logical replay: 144/144 row counts, schema hashes, and normalized content hashes match.
- Data parts: 2,160/2,160 raw/Parquet paths rehashed and size-verified.
- Maximum rows per response: 2,000; continuation pages: 792; identical inclusive-boundary duplicates: 792; conflicting duplicates: zero.
- Corrected-attempt protected timestamps/values: 0/0.
- Gap register: 24 pass-specific rows, representing 12 replay-confirmed empty open-interest cells in the first 2023 window.
- Phase B/C launched: no/no.
- Economic outputs: zero.
- Focused and applicable repository tests: 64 passed, 0 failed.
- Compile checks passed for the acquisition and Phase-A finalization tools and tests.
