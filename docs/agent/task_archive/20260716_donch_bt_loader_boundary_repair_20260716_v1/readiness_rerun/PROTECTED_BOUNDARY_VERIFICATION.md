# Protected Boundary Verification Rerun

## Result

Status: `repaired_paths_pass_overall_firewall_blocked`

No real protected, market, funding, or capture payload was opened.

## Repaired paths

- `load_symbol_bars()` now asserts per-file purpose, Kraken venue, proven time bounds, and rankable status before each trade or mark `pd.read_parquet()` call.
- `load_funding()` applies the same pre-read gate and additionally requires file-level `funding_type=exact`.
- Both paths filter pre-2023 and conflicting non-Kraken rows before returning to a rankable consumer.
- Signal-ineligible/imputed rows cannot leave this exact-funding loader.

Synthetic evidence in `unit_tests/test_rankable_loader_boundary.py` proves:

- protected, mixed, execution-calibration, prospective, external/unrankable, unknown, and unprovable market/funding files: rejected with zero affected reader and downstream calls;
- unrankable mark file: mark reader zero and downstream zero;
- file-level imputed funding: funding reader zero and downstream zero;
- authorized files containing pre-2023/non-Kraken rows: reader allowed, prohibited rows absent at downstream;
- valid Kraken 2023-2025 market and exact-funding rows: reader and downstream each called once.

## Remaining blocker

The readiness-wide firewall is not complete. An AST and caller scan found active sibling raw readers that still call `pd.read_parquet()` without `assert_rankable_file_authority()`:

- `load_symbol_signal_bars()` at `tools/run_kraken_family_engine_aggregate_first_sweep.py:4221`, called by a rankable stage at line 23610;
- `load_symbol_rank_close_window()` at line 4495, called by rank/Top-N paths;
- `a1_load_symbol_bars_window()` at line 25646, called by A1 cache/outcome stages.

Also, `data_paths()` does not yet populate a real `rankable_file_authority` map. Therefore existing real historical files fail closed rather than being silently admitted, which is safe but not operationally ready.

No sibling reader was patched because the current repair task permits only call paths first reproduced and traced to the archived demonstrated defect.
