# Defect Reproduction

## Affected call paths

- Market/trade/mark: `load_symbol_bars()` in `tools/run_kraken_family_engine_aggregate_first_sweep.py` -> `pd.read_parquet()` -> timestamp filtering -> rankable engine callers.
- Funding: `load_funding()` in the same file -> `pd.read_parquet()` -> upper timestamp filtering -> signal/outcome engine callers.

## Original behavior

- `pre_holdout_files()` used only a filename-derived chunk start and did not consult purpose/rankability metadata.
- Mixed, calibration-only, prospective, external/unrankable, and unknown-purpose synthetic files reached `pd.read_parquet()`.
- `load_symbol_bars()` honored its caller-provided lower bound even when earlier than 2023 and did not remove a conflicting `venue_symbol` row.
- `load_funding()` had no rankable lower bound or row venue/exactness boundary.

## Synthetic fixture identity

`unit_tests/test_rankable_loader_boundary.py` uses temporary empty path placeholders, in-memory DataFrames returned by a mocked `pd.read_parquet`, explicit synthetic file-authority dictionaries, and mocked first downstream consumers. No real market, funding, capture, or protected payload is opened.

## Reader and downstream spy result before patch

- Invalid authority cases: expected `reader=0`, `downstream=0`, explicit error; current code raised no error and invoked the reader.
- Authorized mixed-row cases: reader invocation was allowed, but pre-2023, non-Kraken, and signal-ineligible funding rows reached the downstream spy.
- Valid authorized Kraken 2023-2025 cases: passed, establishing that the fixture reaches the real loader path.

## Failing test before patch

Command:

```text
./.venv/bin/python -m unittest -v unit_tests.test_rankable_loader_boundary
```

Result:

```text
tests run: 6
failures: 14 subtest/assertion failures
exit: 1
```

The failures cover both market and funding paths and are attributable to missing pre-reader file authority and missing pre-consumer row gates.

An attempted repository-map command using `pytest` was not a defect run because the venv does not contain pytest. No dependency was installed; the supported unittest path was used instead.

## Authority rule to enforce

- File metadata must prove `purpose=rankable_research`, `venue=kraken`, and an interval ending before `2026-01-01T00:00:00Z` before any payload read.
- Missing, malformed, protected, mixed, calibration-only, prospective, external/unrankable, or unknown authority fails closed before the reader.
- An otherwise authorized file may be read even when it contains pre-2023 or non-Kraken rows, but those rows must be removed before return to the first rankable consumer.
- This exact funding loader accepts exact funding only; imputed/signal-ineligible rows do not reach its rankable consumers. The separate shared imputation outcome-cost model is out of scope and unchanged.

## Passing test after patch

Command:

```text
./.venv/bin/python -m unittest -v unit_tests.test_rankable_loader_boundary
```

Result: 8 tests run, 8 passed, 0 failures. Invalid market, mark, and funding authorities produced zero affected payload-reader calls and zero downstream calls. Authorized mixed-row files were read, while pre-2023, non-Kraken, and signal-ineligible funding rows were absent from the downstream spy.
