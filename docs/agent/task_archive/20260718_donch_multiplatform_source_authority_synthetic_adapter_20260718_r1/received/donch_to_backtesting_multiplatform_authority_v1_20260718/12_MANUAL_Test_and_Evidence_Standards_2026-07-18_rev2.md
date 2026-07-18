---
status: proposed current testing manual after application
date: 2026-07-18
revision: 2.0
scope: multi-platform historical research evidence contracts, tests, reproducibility and decision states
authority: operating contract plus verified machine contracts and run manifests
supersedes: 12_MANUAL_Test_and_Evidence_Standards_2026-07-16.md after verified application
provenance: 2026-07-16 manual; approved multi-platform policy; repository preflight
known limitations: Capital.com acquired schemas and historical economic semantics remain unavailable
---

# Test and Evidence Standards

## Fixed scope

```text
research_platforms:
  - Kraken derivatives
  - Capital.com instruments present in a verified acquisition manifest for the configured account and legal environment
rankable_interval: [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
protected_period_start: 2026-01-01T00:00:00Z
kraken_july_2026_purpose: execution_calibration_only
capitalcom_2026_plus_default_purpose: data_engineering_only
paid_historical_vendor_data: prohibited
live_trading_orders_private_account_actions: not_authorized
existing_kraken_lineage: preserve_unchanged
cross_platform_contract: directed_source_to_target_required
```

A rankable runner must fail closed before payload load when source authority, platform, purpose, interval, schema hash, or content identity cannot be proved.

## Separate status dimensions

```text
hypothesis_status
evidence_level
reproducibility_status
validation_status
deployment_status
```

No current object is validation-grade or deployment-approved.

## Evidence levels

The existing levels 0-7 remain active. Evidence depth does not transfer between platforms. Level 7 execution calibration on one platform does not validate directional economics on that platform or another.

## Required hypothesis contract

Every economic proposal freezes:

```text
hypothesis_id_and_parent_lineage
translation_id
source_platform
source_instrument_uid
target_platform_if_different
target_instrument_uid_if_different
mechanism_and_compelled_actor_or_state
direction_and_horizon
PIT_universe_and_contract_form
source_availability_time_and_minimum_lag
target_first_executable_time
activation_and_disable_rules
entry_stop_exit_timeout
required_data_and_exactness
platform_specific_price_and_cashflow_semantics
fees_funding_or_financing_spread_slippage_currency_conversion
corporate_action_and_lifecycle_treatment
main_null_control_and_ablations
reverse_direction_control_if_cross_platform
falsification_rule
multiplicity_family
protected_data_policy
acceptance_and_stop_rules
```

No economic run occurs without exact approval of this frozen contract.

## Data contract

Each input declares platform, account/legal environment where relevant, native instrument ID, path, size, SHA-256, schema hash, interval, purpose, protected classification, point-in-time meaning, exactness, availability timestamp, and known gaps.

Required guards:

- platform and purpose verified before open;
- rankable interval proven exclusive of 2026+;
- pre-2023 and wrong-platform rows removed before features;
- current-only metadata never projected backward silently;
- Kraken trade/mark/index/funding remain distinct;
- Capital.com bid/ask, financing, dividend adjustments, conversion and contract form remain distinct;
- unknown lifecycle and cost fields remain unknown;
- no missing data replaced by a differently named proxy without an explicit proxy claim.

## Identity

Existing Kraken candidate, event, control, episode, and run hashes are immutable. New platform-aware outer identities may reference them. Capital.com and directed cross-platform translations use new IDs. Direction reversal changes the contract ID.

## Execution and cashflows

### Kraken

Use trade/last for fills, mark for margin/liquidation, index for anchoring when acquired, and signed-notional funding. Preserve exact/mixed/imputed partitions.

### Capital.com

Use ask for buys and bid for sells unless a reviewed contract specifies another executable quote. Preserve bid and ask bars. Do not treat reported volume as exchange volume until verified. Model financing, premium adjustments, dividend/corporate-action cashflows, currency conversion, expiry and calendar only from effective-date authority. Missing evidence caps or blocks the claim.

### Cross-platform

Source information must be observable before the target decision. Closed targets map to the first executable target quote after reopening. Same-bar target close/high/low may not be used before availability. Include target autocorrelation, common-factor, session/reopen, reverse-direction and timestamp-null controls.

## Controls and multiplicity

Controls must be real and mechanism-relevant. Register all attempted translations and direction pairs. Cluster shared market episodes across platforms where appropriate without treating duplicate source observations as independent evidence.

## Efficient funnel

```text
verified_source_manifests
-> causal_normalized_features
-> frozen_candidate_and_control_identities
-> train_only_kill_checks
-> duplicate_and_episode_clustering
-> source_specific_exact_simulation
-> controls_and_ablations
-> purged_train_validation_and_multiplicity
-> frozen_candidate
-> separately_approved_independent_evidence
```

Early screens reject only; they do not promote.

## Reproducibility

Record commit/dirty state, code/config/data/universe/cost hashes, exact commands and exit codes, test counts, source and target manifests, purpose audits, artifact manifest, superseded roots, and independent review. Blank or unavailable fields are not passes.

## Claims

A result is always bounded to its platform(s), instrument contract, period, universe, data exactness, execution model, controls, and evidence level. Another platform’s result is a prior or comparison, not automatic validation.
