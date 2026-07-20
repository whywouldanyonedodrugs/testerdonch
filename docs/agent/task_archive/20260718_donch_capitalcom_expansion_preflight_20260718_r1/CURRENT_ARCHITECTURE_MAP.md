# Current Architecture Map

## Summary

The repository has reusable evidence and signal-state contracts but not a general venue adapter architecture. The operative data and economic paths are Kraken-specific. Extending the shared contract boundary is smaller and safer than generalising existing Kraken runners.

## Loader and protected boundary

| Surface | Current owner | Verified behavior | Expansion consequence |
|---|---|---|---|
| Dataset paths and manifest authority | `tools/run_kraken_family_engine_aggregate_first_sweep.py:3974` and `:4014` | Hard-coded Kraken trade, mark, funding, instrument and manifest roots | Keep unchanged; add a separate Capital.com adapter and shared authority validator |
| File pre-open guard | `tools/run_kraken_family_engine_aggregate_first_sweep.py:4115` | Rejects absent/unprovable authority, non-rankable purpose, non-Kraken venue, protected/mixed files and non-exact funding before `read_parquet` | Extract the validation logic into a small source-neutral contract; retain a compatibility wrapper for Kraken |
| Row filtering | `tools/run_kraken_family_engine_aggregate_first_sweep.py:4142` | Filters pre-2023/protected and non-Kraken rows before downstream features | Capital adapter needs required platform/epic fields and equivalent row guard |
| Focused authority reader | `tools/build_kraken_c01_foundation.py:200` and `:364` | Loads only manifest-authorized 5m PF trade/mark shards | Preserve as Kraken implementation |
| Funding | `tools/run_kraken_family_engine_aggregate_first_sweep.py:4169` | Exact Kraken funding only in rankable loader; imputed rows excluded from signal eligibility | Capital financing must be a separate cashflow contract, not mapped to Kraken funding |

## Instrument and lifecycle

- Kraken symbols are primarily `PF_*USD`; source identity is often implicit in paths and symbol prefixes.
- C01 identities include `venue`, but many family economic-address helpers do not include a platform namespace.
- Lifecycle authority is built in `tools/build_kraken_u2_lifecycle_authority.py` and reference-panel authority in `tools/build_kraken_c01_reference_panel_authority.py`.
- Current authority is Kraken-specific and known to have lifecycle caps; it must not be repurposed for Capital.com epics, expiries, market status, corporate actions, or session calendars.

## Shared reusable core

- Evidence validation: `tools/qlmg_evidence_contracts.py`
- Signal-state lineage/non-overlap: `tools/qlmg_signal_state_contract.py`
- Real-control construction contract: `tools/qlmg_real_controls.py`
- Reproducibility and run manifests: mature examples in `tools/run_kraken_c01_level3_economic.py` and `tools/run_kraken_c02_level3_economic.py`
- Task archive/Drive rules: `docs/agent/RUN_AND_ARTIFACT_CONTRACT.md` and `docs/agent/DRIVE_HANDOFF_TARGET.md`

These can remain logically shared, but future rows and hashes need explicit platform/source fields.

## Event, control, and reporting identity

Candidate, control and episode IDs are implemented in multiple family modules. Retrofitting existing Kraken IDs would invalidate lineage. New multi-platform work should therefore add an outer namespace (`platform`, `instrument_uid`, translation ID) while preserving existing Kraken IDs unchanged.

## What is absent

- Capital.com reader or normalized schema
- Platform-neutral dataset manifest schema
- Capital.com bid/ask execution contract
- Capital.com instrument/lifecycle/calendar/corporate-action authority
- Directed source-to-target relationship contract
- Cross-platform event clock/first-executable-target helper
- Verified Capital.com data root or payload
