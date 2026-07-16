# Validation

## Acceptance results

- Repaired market protected/unrankable reader calls: `0`.
- Repaired funding protected/unrankable reader calls: `0`.
- Repaired mixed/calibration/prospective/external/unknown reader calls: `0`.
- Pre-2023 rows reaching focused downstream spy: `0`.
- Non-Kraken rows reaching focused downstream spy: `0`.
- Signal-ineligible/imputed funding reaching focused downstream spy: `0`.
- Valid Kraken 2023-2025 market fixture: passed.
- Valid exact Kraken 2023-2025 funding fixture: passed.
- Focused tests: `8 passed, 0 failed`.
- Owning-module regressions: `286 passed, 0 failed`.
- Repository guard tests: `9 passed, 0 failed`.
- Compile: passed.
- Real protected payloads opened: `0`.
- Economic outputs computed: `0`.

## Scope validation

The patch changes only the reproduced `load_symbol_bars()` and `load_funding()` reader paths plus a shared local helper used by their trade/mark/funding reads. Pre-2023 and row-level non-Kraken data are not rejected before an authorized file opens; they are removed before the loader returns to its first rankable consumer, matching the task's corrected authority interpretation.

## Remaining readiness cap

`load_symbol_signal_bars()`, `load_symbol_rank_close_window()`, and `a1_load_symbol_bars_window()` remain active sibling readers without the helper. They were not patched because they were not part of the pre-patch reproduced defect scope. `data_paths()` also does not yet bind real historical authority metadata. Overall first-wave readiness therefore remains blocked even though this bounded patch passes.
