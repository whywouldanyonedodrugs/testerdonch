# Independent Review

Decision: `approve` for the bounded loader repair; first-wave readiness remains blocked.

## Scope

- Production diff is confined to the owning aggregate loader file.
- Test diff is confined to one new focused module and two synthetic fixture adaptations.
- No strategy logic, economic result, raw data, capture, acquisition, governance, package, or substantive registry changed.

## Authority semantics

- Protected/mixed/calibration/prospective/external/unknown file authority is checked before `pd.read_parquet` in both reproduced loaders.
- Pre-2023/non-Kraken rows are filtered after an authorized read but before downstream use; no stronger universal pre-open rule was introduced.
- Funding requires exact file authority, and explicit imputed/signal-ineligible rows are excluded.
- Mark input receives the same pre-read authority check.

## Test validity

- Tests use only temporary empty path placeholders and mocked in-memory payloads.
- Reader and downstream spies are independently asserted.
- Valid fixtures prove tests reach the loader and consumer rather than passing vacuously.
- The full 286-test owning module and repository guards pass.

## Residual findings

No blocking defect was found inside the authorized patch. The readiness rerun independently identified three sibling readers and absent real authority-map binding; these are explicitly deferred rather than hidden or repaired out of scope.

## Rollback

Before publication, discard the isolated branch. After the authorized local commit, revert that one commit. No data or historical roots require restoration.
