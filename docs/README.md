# Donch / QLMG Kraken Research Repository

This repository is governed as a **QLMG-inspired historical multi-platform backtesting and research** repository. Kraken derivatives and Capital.com mechanics remain platform-specific and require separate manifest authority.

## Active Docs

- `AGENTS.md` - root agent instructions and non-negotiable boundaries.
- `docs/agent/` - agent operating manuals, recovery workflow, run contract, protected-period policy, code-review guidance, and known failure patterns.
- `docs/agent/SOURCE_MAP.md` - current source order, continuity, decisions, policy, and registry paths.
- `docs/agent/RESEARCH_GATE_ROUTING_POLICY.json` - machine-readable prospective gate-routing policy.
- `docs/QLMG_PERP_PROJECT_STATE.md` - current project status and direction.
- `docs/QLMG_PERP_BACKTESTING_MANUAL.md` - active backtesting and exchange-fidelity rules.
- `docs/QLMG_PERP_DATA_CONTRACT.md` - point-in-time data, venue, price-role, and sidecar rules.
- `docs/QLMG_PERP_STRATEGY_CATALOG.md` - draft family catalog for future approved work.
- `docs/QLMG_PERP_VALIDATION_PROTOCOL.md` - validation and sealed-data policy.
- `docs/QLMG_PERP_MIGRATION_FROM_DONCH.md` - migration notes from Donch/V3/S1.

## Binding Boundaries

- Approved research platforms: Kraken derivatives and Capital.com instruments present in a verified acquisition manifest.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Protected period: `2026-01-01T00:00:00Z` onward.
- Documentation, tooling, or coding approval does not authorize an economic run.

## Legacy Status

Donchian/V3/S1 and Bybit-oriented reset material are legacy reference only unless a later authority explicitly reactivates a specific component. Pre-governance copies of revised docs are preserved under `docs/agent/superseded/20260716_pre_governance/`.

## GitHub Hygiene

GitHub should contain source, tests, and canonical docs only. Generated outputs, large recovery bundles, archives, reports, result roots, parquet/duckdb data, local bundles, and environment files are intentionally kept out of normal source history unless a task explicitly authorizes a small documentation archive.
