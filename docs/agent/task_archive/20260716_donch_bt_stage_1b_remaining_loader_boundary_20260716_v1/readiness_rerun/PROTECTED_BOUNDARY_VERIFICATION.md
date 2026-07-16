# Protected Boundary Verification

Status: `pass_for_all_known_rankable_readers`.

Known readers:

1. `load_symbol_bars()`
2. `load_funding()`
3. `load_symbol_signal_bars()`
4. `load_symbol_rank_close_window()`
5. `a1_load_symbol_bars_window()`

For every reader, authority is asserted before `pd.read_parquet`. Synthetic protected, mixed, and unknown/unprovable authority produces an explicit `RuntimeError`, with payload-reader calls `0` and downstream calls `0`. Authorized synthetic files may be opened, but pre-2023 and non-Kraken rows are removed before rankable downstream use; valid Kraken 2023-2025 rows pass.

`data_paths()` now binds the existing download-manifest fields `dataset`, `symbol`, `parquet_path`, `status`, `chunk_start`, `chunk_end`, `rankable_pre_holdout`, and `contains_protected_period`. The metadata-only real-root check resolved 166,408 existing paths: 83,204 trade and 83,204 mark. Authority metadata includes protected-spanning files, which fail closed before payload reads.

Funding behavior from Stage 1A is unchanged. Existing historical funding manifests do not provide the same complete rankable interval fields, so real funding remains fail-closed unless explicit exact authority is supplied. Imputed or signal-ineligible funding cannot activate a signal.

- Real protected payloads opened: `0`.
- Invalid rows reaching rankable downstream: `0` in synthetic spies.
- Economic outputs computed: `0`.
