# Stage 17 Independent Pre-Outcome Review

Decision: **fail closed — global stop before Telegram and economic outcomes**.

## Authority that passed

- Starting `main` and `origin/main`: `a3981b505e908b5fb617a0921f45869535e2b542`.
- External approval: `fe57d5c1efca3af3cb83c3e07b399e03c51f5dbe635b03bd48201944506c6853`.
- Stage-16 campaign manifest: `cc07499c671cf39b8ceaee91156f141dcc2c5532142af29a38a4f6830b73f23d`.
- Stage-16 packet: `c01281e50f40f95b922a04ed01c5b3d28ed325577891eed8e2ca5d32286965ca`.
- Twelve packet dependencies, 563 Stage-14 state files, the 187-symbol feature authority, and the Stage-9 KDA02 completed-purge tape were hash-verified.
- Focused Stage-16/campaign/runtime tests passed before the terminal review. The packet replay regression was repaired so tests restore the complete authority directory.
- The outcome-free boundary superset contained 2,197,950 eligible hourly symbol-boundaries with 100% materialized availability: 458,796 exact and 1,739,154 imputed; no imputed row was gate eligible.

## Binding defects

### 1. Funding cashflow semantics are incomplete

The approved funding contract requires exact signed cashflow as:

`-position_sign * relativeFundingRate * boundary_notional / entry_notional * 10000`.

The frozen shared panel contains relative rates but does not contain an approved `boundary_notional`, spot/index price, or notional ratio. This is especially unresolved for imputed boundaries. Kraken's official linear-contract specification defines relative funding payout using the spot price at funding calculation, so silently setting the ratio to one, substituting mark/trade price, or deriving spot from the future-basis field would be a new semantic choice.

The repository has no packet-bound historical spot/index payload or frozen derivation for this field. Exact raw funding rows also contain an absolute funding rate, but this does not solve imputed rows under the unchanged frozen relative-rate model.

### 2. Protected funding payload was opened by the inherited helper

`tools/run_kraken_shared_funding_imputation_model.py::load_exact_rates` reads full raw funding Parquets and applies the `<2026-01-01` filter only after deserialization. Stage 17 called this through `extend_frozen_panel_with_verified_model`. The extension fit and outputs used only pre-2026 rows, and no strategy price outcome was opened, but protected funding rows were nevertheless opened. A later diagnostic confirmed the same loader/file behavior.

Therefore the generated `protected_rows_opened: 0` assertions are invalid and are superseded by `BLOCKED_PREOUTCOME.json` and generation 2 of `CAMPAIGN_STATE.json`. The exact number of protected funding rows is deliberately not recomputed because doing so would require another unauthorized protected read.

## Stop decision

Status: `blocked_preoutcome_common_funding_and_protected_read_defect`.

- Telegram messages sent: none; gate 8 failed before gate 9.
- Economic outcome reader opened: no.
- Economic results computed: no.
- Protected strategy-outcome rows opened: no.
- Protected non-outcome funding payload opened: yes; not used in fitting or economics.
- Capital.com payload opened: no.
- Phase 2–5 executed: no.
- Phase 6 controls executed: no.

The draft runtime is not publishable because completing it would require inventing funding semantics that the approval expressly froze.

