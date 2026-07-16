# Protected Boundary Verification

## Result

Status: `fail_closed_not_demonstrated`

No protected market or strategy payload was opened. The failure was reproduced with synthetic in-memory frames and temporary empty files.

## Source Path

Owning loader: `tools/run_kraken_family_engine_aggregate_first_sweep.py`

- `pre_holdout_files()` rejects files only when a filename-derived chunk start is `>= 2026-01-01`.
- `load_symbol_bars()` calls `pd.read_parquet()` and only then filters rows by `start`, `end`, and `PROTECTED_TS`.
- `load_funding()` calls `pd.read_parquet()` and only then filters funding timestamps.
- `list_symbols()` restricts names to `PF_`, but `load_symbol_bars()` does not fail on a non-Kraken `venue_symbol` carried inside such a path.
- `data_paths()` does not include capture roots, which is useful separation, but it is not a universal pre-read input-manifest gate.

## Synthetic Evidence

1. A chunk named `PF_TESTUSD_20260101T000000.parquet` was not passed to the reader: pass for wholly protected filename partitions.
2. A chunk named `PF_TESTUSD_20220101T000000.parquet` was passed to the reader, then its row was filtered: fail for pre-2023 pre-read exclusion.
3. A chunk named `PF_TESTUSD_20251201T000000.parquet` containing synthetic 2025 and 2026 rows was passed to the reader, then the 2026 row was filtered: fail for mixed-file protected pre-read exclusion.
4. A synthetic funding chunk spanning 2025/2026 was passed to the reader, then filtered: fail for funding partition-before-read.
5. A synthetic row tagged `BYBIT:TESTUSDT` inside a `PF_TESTUSD` path was returned: fail for row-level venue fail-closed behavior.

Existing guard tests passed 37/37, but they verify timestamp rejection/output scanning and the older sealed-slice policy. They do not prove physical pre-read partitioning at the 2023 start, 2026 cutoff, venue boundary, and capture boundary.

## Funding Gate

The active governance contract states that imputed funding is outcome-cost only and cannot activate a signal. This task did not find a failing synthetic imputed-gate test, but it stopped before a full family-by-family gate audit because the required funding pre-read partition itself failed.

## Consequence

The received task requires the audit to stop when fail-closed behavior cannot be demonstrated without opening protected outcomes. New economic work is blocked until a pre-read input firewall is implemented and verified with reader-spy tests.
