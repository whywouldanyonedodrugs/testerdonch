# Documentation and Registry Impact

No item was changed in this preflight.

| Item | Classification | Required action |
|---|---|---|
| `AGENTS.md` | `must_change_before_import` | Replace the singular active-venue boundary only after exact human approval; retain platform-specific semantics and protected-period firewall |
| `docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md` | `must_change_before_import` | Add explicit source/target platform and Capital.com data-engineering-only rules |
| `docs/agent/REPOSITORY_MAP.md` | `must_change_before_import` | Record approved Capital.com adapter/data root and commands after implementation |
| `docs/QLMG_PERP_PROJECT_STATE.md` | `must_change_before_economic_run` | Record approved multi-platform scope and current Capital.com readiness |
| `docs/QLMG_PERP_DATA_CONTRACT.md` | `can_remain_Kraken_specific` | Keep as Kraken perpetual contract; do not genericise it |
| `docs/QLMG_PERP_BACKTESTING_MANUAL.md` | `can_remain_Kraken_specific` | Keep Kraken mechanics; link future Capital.com contract |
| `docs/QLMG_PERP_VALIDATION_PROTOCOL.md` | `must_change_before_economic_run` | Add platform/translation/multiplicity and directed-contract review fields |
| `docs/agent/RUN_AND_ARTIFACT_CONTRACT.md` | `must_change_before_economic_run` | Require platform/source/target identity and source-specific costs in future run manifests |
| `docs/CAPITALCOM_DATA_CONTRACT.md` | `new_file_required` | Bid/ask, calendar, lifecycle, financing, corporate actions, metadata and execution limitations |
| `docs/CROSS_PLATFORM_RESEARCH_CONTRACT.md` | `new_file_required` | Directed source-to-target clocks, controls, mappings and claims |
| Donch hypothesis registry rev2 | `must_change_before_economic_run` | Add route fields and new translation IDs; preserve all Kraken decisions |
| Platform/data capability registry rev2 | `must_change_before_import` | Separate documented capability from acquired/verified coverage |
| Instrument identity/relationship tables | `new_file_required` | Create only after a verified Capital.com handoff; do not pre-populate from prose |
| Current Kraken candidate libraries/results | `not_applicable` | Preserve unchanged; no migration or rehash |
| Kraken venue/capture history | `can_remain_Kraken_specific` | Preserve as venue-specific authority |
