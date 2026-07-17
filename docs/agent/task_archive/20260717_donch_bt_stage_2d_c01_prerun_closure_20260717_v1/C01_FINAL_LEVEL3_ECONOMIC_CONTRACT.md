# C01 Final Level-3 Economic Contract

Status: frozen pre-run contract; no economic run is authorized by this file.

## Authority and immutable inputs

- Family: `C01_debetaed_residual_shock_path_bifurcation`
- Repository lineage: `45d92488a41fb97a9a30936075c19581f358357d`
- Generator contract: `3464e79a79956c881c7418840068a61e3f3a47776a5a4d3a669e98df124fd970`
- Stage 2C economic draft: `f1c8c612ea9f7ffcc2abad3f2efde36b5dfb68fde20d2769fdc5ce40ab306c13`
- Feature contract: `c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb`
- Cohort: `768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15`
- Reference panel: `2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763`
- Candidate cohort: `current_roster_bar_existence_cohort`; it is not survivorship-free and does not prove continuous tradeability.
- Train interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`; later rows are protected.

No generator, feature, cohort, reference-panel, onset, or path threshold is changed.

## Definition and multiplicity freeze

The primary model is `btc_eth_ols_daily_v1`. The BTC-only `btc_only_ols_daily_v1` model is robustness-only and cannot rescue a failed primary definition. The four branches are `positive_smooth_long`, `negative_smooth_short`, `positive_jump_completed_failure_short`, and `negative_jump_completed_failure_long`. Intermediate states are diagnostics only.

Each model has all four branches at fixed 6h and 24h timeouts: 8 primary definitions plus 8 separately registered robustness definitions, 16 total. Zero-trade definitions remain registered. Every later control and ablation is an additional multiplicity attempt.

## Entry and confirmation

Positive smooth enters long and negative smooth enters short at the next executable five-minute trade-bar open after onset. For positive jump-dominated onsets, confirmation is the first completed trade bar within 24h closing below the causally frozen dominant residual bar low, followed by short entry at the next executable open. The negative jump branch is symmetric above the dominant bar high and enters long. Dominant-bar identity and confirmation are deterministic and frozen before outcomes. No confirmation means no trade.

## Exits and non-overlap

Timeouts are fixed at 6h and 24h. Smooth stops require a completed mark close through the opposite six-hour shock-window extreme, then execute at the next trade-bar open. Jump-failure stops require a completed mark close beyond the dominant jump-bar extreme in the original shock direction, then execute at the next trade-bar open.

Each definition runs independently. Within symbol and definition, onsets are chronological; a later onset is skipped only while an actual position remains open. Eligibility resumes after the actual executed exit. No nominal maximum-hold preblocking and no combined portfolio are allowed. Accepted plus skipped rows must reconcile to eligible rows in a complete skip ledger.

## Boundary and invalid-row policy

Confirmation, entry, every stop-monitoring bar, selected timeout, funding accounting, and next-open execution must lie wholly inside the train interval. Mixed 2025/2026 payloads must not be opened. Boundary-crossing intervals are excluded; no artificial endpoint close is allowed.

Fail closed on missing next-open trade bars, missing mark bars during monitoring, non-positive structural stop distance, a stop already breached before entry, non-finite price or funding, known lifecycle-invalid intervals, duplicate economic addresses, or same-bar ambiguity. There are no touch fills, passive fills, partial fills, adds, leverage optimization, or inferred intrabar ordering.

## Outcome units, costs, and funding

Exposure is fixed notional; structural stop distance does not size trades. Primary units are fixed-notional net return and basis points. Structural R is diagnostic only, with full denominator distributions and extremes, and cannot be the sole permission metric. No post-outcome denominator floor is permitted.

- Base costs: 5 bps taker per side plus 4 bps round-trip slippage.
- Stress costs: 10 bps taker per side plus 12 bps round-trip slippage.
- Full-period gross and fee/slippage-net results are kill-screen evidence only.
- Fully exact-funded rows are the primary funding-valid subset and are reported by calendar period.
- Mixed and imputed full-period funding results are sensitivity only and never promotion evidence.
- Zero-funding-boundary rows are separate.
- No pooled funding partition can rescue a definition.

## Level-3 permission-to-test-controls rule

For a primary definition under base costs, all must hold: at least 100 executed trades; at least 20 trades in each of 2023, 2024, and 2025; positive fixed-notional mean and median net return; canonical-episode bootstrap 95% CI lower bound at least -5 bps; no symbol above 25% of aggregate net PnL; no canonical episode above 10%; no year above 70% of aggregate positive net PnL; and stress mean at least -10 bps. Bootstrap seed is `20260717` with exactly `10,000` resamples.

These gates permit Level-4 controls only. They do not establish incremental value, validation, robustness, promotion, or live readiness. If no primary definition passes, C01 stops at Level 3. BTC-only and secondary branches cannot rescue it.

## Claim boundary

A passing result may be described only as: "train-period event-ledger economics sufficient to justify predeclared controls under current-roster, funding, execution, multiplicity, and lifecycle caps."
