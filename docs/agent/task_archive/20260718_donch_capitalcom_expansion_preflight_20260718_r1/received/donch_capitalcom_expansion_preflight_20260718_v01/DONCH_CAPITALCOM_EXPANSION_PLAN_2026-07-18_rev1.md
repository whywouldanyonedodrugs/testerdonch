---
status: proposed; ready for review; not applied
created_utc: 2026-07-18
revision: 1
scope: expansion of Donch from Kraken perpetuals to Kraken perpetuals plus Capital.com instruments, including cross-platform research
planning_authority: current human direction to plan the expansion
controlling_current_authority: existing 2026-07-16 Donch contract remains active until an exact replacement is approved and applied
supersedes: none
provenance: current Donch source bundle; hypothesis and data-capability registries; official Capital.com API, markets, pricing, fee, and corporate-action documentation accessed 2026-07-18
known_limitations: Capital.com download contents, coverage, legal-entity/account universe, repository layout, and current backtesting Git state have not been verified in this task; no source, repository, data, or project-setting change was made
---

# Donch Capital.com Expansion Plan

## 1. Decision summary

The recommended design is a **shared research core with source-specific adapters**:

1. one Donch authority, hypothesis, evidence, protected-period, multiplicity, and review system;
2. separate Kraken and Capital.com raw/normalized data partitions and execution semantics;
3. one small instrument-identity and relationship layer for cross-platform work;
4. cross-platform hypotheses represented as explicit source-to-target contracts, not as automatic portability of evidence.

This avoids two bad extremes:

- two disconnected research programs that cannot share events, hypotheses, controls, or evidence; and
- one universal backtester that incorrectly treats a Kraken perpetual, a Capital.com share CFD, a spot commodity CFD, an index CFD, and a currency pair as mechanically interchangeable.

The expansion should preserve the current global research firewall unless the human later approves a different policy:

```text
rankable_interval: [2023-01-01, 2026-01-01)
protected_period: 2026-01-01 onward
paid_historical_vendor_data: prohibited
live_trading: not_authorized on either platform
Capital.com 2026+ default purpose: data_engineering_only
Kraken July 2026 capture purpose: execution_calibration_only
```

“All Capital.com instruments” should mean **acquisition and inventory scope**, not one undifferentiated research universe. Research eligibility remains hypothesis-, asset-class-, lifecycle-, and data-quality-specific.

## 2. Current authority state

### Verified

- Current Donch authority is Kraken-only.
- Rankable research is currently 2023-01-01 through 2025-12-31, with 2026 onward protected.
- No current strategy is validation-grade or live-ready.
- The current 214-row hypothesis registry mixes current Kraken decisions, legacy priors, forum hypotheses, and rejected claims.
- The current data capability registry is Kraken-specific.
- The observed backtesting checkout was previously heavily dirty, so direct overlay remains unsafe until a fresh repository preflight proves otherwise.

### Proposed interpretation of the present direction

The human has approved planning for a broader program covering Kraken perpetuals and Capital.com instruments. This authorizes research design and draft artifacts. It does not by itself authorize:

- applying revised Donch sources or project instructions;
- modifying the backtesting repository;
- running an economic screen;
- inspecting protected outcomes;
- placing an order or using an order endpoint;
- committing, pushing, merging, or deploying.

### Unavailable in this task

- the actual Capital.com download manifest and schemas;
- per-instrument history and gaps;
- whether 2026+ rows have already been physically separated;
- the exact Capital.com account/legal-entity instrument universe;
- historical financing, trading-hours, corporate-action, and instrument-lifecycle coverage;
- current backtesting repository commit, instructions, commands, and safe worktree state.

## 3. Minimal architecture

### 3.1 Shared core

The shared core should contain only concepts that are genuinely common:

- UTC event time and decision time;
- rankable/protected period policy;
- canonical instrument identity;
- point-in-time eligibility and lifecycle status;
- hypothesis identity and family lineage;
- event, control, and canonical episode identity;
- evidence level, reproducibility, validation, and deployment state;
- costs and execution assumptions with source-specific fields;
- multiplicity, concentration, and review records;
- artifact identity, manifests, hashes, and supersession.

### 3.2 Source-specific adapters

#### Kraken adapter

Retain current semantics: trade/last, mark, index when available, funding, perpetual lifecycle, 24/7 calendar, and Kraken-specific execution constraints.

#### Capital.com adapter

Treat Capital.com as an OTC CFD platform. Preserve at least:

```text
platform_epic
instrument_type
instrument_name
underlying_or_reference_identity
currency
expiry
market_status
opening_hours_and_timezone
bid_and_ask_OHLC
reported_volume_field_and_semantic_status
spread
margin_factor
current_overnight_fee_fields
minimum_and_maximum_deal_rules
corporate_action_or_dividend_adjustment_status
spot_vs_dated_future_vs_other_contract_form
metadata_snapshot_time_and_hash
```

Do not silently convert Capital.com bars to exchange trades or mid-price fills. Historical prices are bid/ask OHLC. Entry and exit assumptions must use the correct side. The meaning of the reported volume field must remain `unverified` until established.

### 3.3 Instrument identity layer

Create one small table, populated only as needed:

```text
instrument_uid
platform
platform_instrument_id
asset_class
contract_form
underlying_uid
quote_currency
expiry_or_undated
valid_from_utc
valid_to_utc
calendar_id
metadata_hash
```

A second mapping table is needed only for researched relationships:

```text
source_instrument_uid
target_instrument_uid
relationship_type
relationship_rationale
effective_start_utc
effective_end_utc
mapping_confidence
source_provenance
```

Do not generate all pairwise relationships among thousands of instruments.

### 3.4 Physical data layout

Raw and normalized data should remain source-separated. The exact repository paths must be discovered, but the logical boundary is:

```text
raw/kraken/
raw/capital_com/
normalized/kraken/
normalized/capital_com/
reference/instruments/
reference/calendars/
reference/platform_relationships/
research_outputs/
```

The shared layer should reference source files and hashes rather than copying all source data into a universal table.

## 4. Capital.com-specific evidence constraints

Official Capital.com documentation currently establishes that:

- the API supports the instruments available on the platform;
- all markets can be enumerated through the markets/navigation endpoints;
- historical prices are available at minute through weekly resolutions with a maximum of 1,000 rows per response;
- historical bars expose bid and ask OHLC plus a reported volume field;
- current market details include opening hours, expiry, currency, margin factor, overnight-fee fields, dealing rules, bid/offer, and market status;
- API sessions and rate limits must be managed;
- WebSocket market-data subscriptions are limited to 40 instruments at a time;
- API keys are trading-capable; there is no read-only API-key privilege, although the API can be used with a demo account;
- Capital.com CFD prices may differ from other brokers and underlying markets because the broker sets its own CFD prices using market data, liquidity providers, and pricing/risk methodology;
- spot commodity CFDs can be synthetic rolling prices derived from the nearest futures contracts;
- dividends and other predictable corporate actions can produce cash adjustments separate from price movement;
- overnight-financing timing and rules vary by instrument class and have changed over time.

Consequences for Donch:

1. The downloaded platform instrument manifest, not the public marketing count, is the acquisition authority.
2. Current market metadata does not prove historical metadata or historical costs.
3. A Capital.com price movement is evidence about the Capital.com CFD unless a separate underlying mapping is supported.
4. Historical spread can be estimated from bid/ask bars; depth, queue, and full fillability remain unavailable unless separately captured.
5. Multi-day share, index, commodity, bond, rate, forex, and ETF tests require explicit financing and adjustment treatment.
6. API credentials must be isolated from research code and order endpoints must fail closed.

## 5. Cross-platform research contract

Every cross-platform hypothesis must declare a directed tuple:

```text
source_platform_and_instrument
source_observable
source_event_time
minimum_information_lag
target_platform_and_instrument
target_first_executable_time
target_execution_semantics
holding_horizon
shared_or_distinct_underlying
common_factor_controls
reverse_direction_control
closed_market_and_reopen_rule
currency_conversion_rule
falsification_rule
multiplicity_family
```

Core rules:

- Source and target are directional; `A -> B` and `B -> A` are different hypotheses.
- A source event is usable only after it is observable.
- If the target is closed, the decision time is the first executable target quote after reopening, not the source timestamp.
- A same-timestamp bar join must not use the target bar close or high/low before the decision time.
- Same-underlying Capital.com/Kraken differences are initially a **data and pricing calibration study**, not presumed alpha.
- A result on Capital.com cannot be presented as Kraken execution evidence, and vice versa.
- Cross-platform significance must survive the target’s own autocorrelation, broad-factor, session/reopen, reverse-direction, and timestamp-null controls.

## 6. Hypothesis registry transition

Do not rewrite historical decisions. Preserve all current Kraken family rows and conclusions.

### 6.1 Minimal schema change

Add these routing fields to the next registry revision:

```text
capital_route
cross_platform_route
route_review_status
```

Suggested values:

```text
not_applicable
data_feasibility_required
translation_candidate_unreviewed
independent_platform_translation_permitted
context_only
prospective_only
blocked_by_missing_semantics
rejected_translation_preserved
```

Current Kraken status remains in the existing fields.

### 6.2 Lineage rule

A concrete Capital.com or cross-platform test receives a new translation identity linked to the parent hypothesis. It must not overwrite the Kraken row.

Example:

```text
parent_hypothesis_id: H05
new_translation_id: CAPITAL_H05_BREAKOUT_RETEST_V1
Kraken decision: mandatory retest translation rejected
Capital.com status: untested independent platform translation
```

The Capital.com result may inform the broader mechanism, but it cannot retroactively rescue or relabel the Kraken translation.

### 6.3 Initial routing logic

- Price/path, momentum, prior-high, opening-gap, strong-close, failed-break, and catalyst structures: usually `translation_candidate_unreviewed`, subject to asset-class mechanics.
- Kraken funding, OI, liquidation, mark/index, and order-book mechanisms: usually `not_applicable`, `prospective_only`, or `blocked_by_missing_semantics` unless Capital.com supplies a genuine analogue.
- Session-open hypotheses: a Capital.com translation may be materially distinct because many instruments have real exchange sessions and closures; the rejected Kraken session translation remains rejected.
- Stock/ETF catalyst research: requires a separate PIT corporate-event source; the current crypto catalyst register is not a stock-event census.
- Current rejected myths and evidence safeguards remain rejected program-wide unless the rejection was explicitly venue-mechanical.

## 7. Documentation impact

The expansion should not genericise every Kraken document. Revise the shared governance layer and add a Capital.com module.

### Must be revised before Capital.com economic research

| Current source | Proposed action |
|---|---|
| `00_READ_FIRST_Project_Source_Map.md` | Add the multi-platform read order and new Capital/cross-platform sources. |
| `01_AUTHORITY_QLMG_Operating_Contract_2026-07-16.md` | Replace singular Kraken-only scope with two research platforms; retain source-specific execution and global protected rules. |
| `02_STATE_Master_Continuity_Brief_2026-07-16_rev8.md` | Record expansion state, remote Capital download, unavailable manifest, and no economic authorization. |
| `04_REGISTRY_Hypothesis_and_Family_Status_2026-07-16.csv` | Create rev2 with Capital and cross-platform routing fields; preserve Kraken decisions. |
| `05_REGISTRY_Kraken_Data_and_Evidence_Capability_2026-07-16.csv` | Supersede with a platform-neutral capability registry containing documented versus acquired status. |
| `12_MANUAL_Test_and_Evidence_Standards_2026-07-16.md` | Replace Kraken-only boundary with target-platform contracts and a cross-platform contract. |
| `13_GUIDE_Backtest_Claims_and_Review_2026-07-16.md` | Review source platform, target platform, instrument type, cost semantics, and portability separately. |
| `15_RUNBOOK_Human_Approval_and_Orchestration_2026-07-16.md` | Add the Capital acquisition host and handoff; keep tasks self-contained. |
| `16_INDEX_Provenance_and_Supersession_Map_2026-07-16.md` | Record exact supersession and retain Kraken-specific history. |
| `DONCH_PROJECT_INSTRUCTIONS_FULL.md` and compact field | Apply only after exact human approval; update project scope and protected rules. |
| `BACKTESTING_AGENT_OPERATING_AND_ARCHIVE_INSTRUCTIONS.md` | Generalise repository scope while requiring explicit target platform per task. |
| `DONCH_TO_BACKTESTING_TASK_ARCHIVE_TEMPLATE.md` | Replace fixed `venue: Kraken only` with explicit source/target platform fields. |

### Add only three new durable sources

```text
11B_HISTORY_Capitalcom_Platform_and_Data_YYYY-MM-DD_rev1.md
17_METHOD_Cross_Platform_Research_YYYY-MM-DD_rev1.md
18_REGISTRY_Instrument_and_Platform_Relationships_YYYY-MM-DD_rev1.csv
```

### Keep venue-specific unless a later task needs them changed

- Kraken capture runbook;
- Kraken venue/data history;
- Kraken protected-capture guide;
- Kraken forum synthesis;
- Kraken microstructure notes.

The crypto catalyst register remains a crypto-event source. Do not turn it into a universal corporate-event database.

## 8. Phased execution plan

### Phase 0 — Approve the scope contract

**Outcome:** one reviewed, unambiguous policy decision before implementation.

Draft values:

```text
research_platforms: Kraken perpetuals; Capital.com acquired instruments
rankable_interval: [2023-01-01, 2026-01-01)
protected_period: 2026-01-01 onward
Capital.com 2026+ default: data_engineering_only
live_trading: not_authorized
Capital.com authenticated market-data access: acquisition only
Capital.com order/position/account-mutation endpoints: prohibited
cross-platform research: allowed only under a frozen directed contract
```

**Acceptance:** the human approves the exact replacement contract and target files/settings. Until then, status remains proposed.

### Phase 1 — Backtesting repository read-only preflight

**Owner:** backtesting agent.

**Outcome:** repository-specific minimal design, no changes and no economic run.

The agent verifies root, `AGENTS.md` chain, branch, commit, remotes, dirty state, data layout, current loaders, registries, supported commands, archive convention, and safe-worktree option. It identifies the smallest code/document surface needed for a second platform.

**Acceptance:** read-only report, proposed file tree/diff scope, tests, rollback, and verified Drive handoff package. No repository mutation.

### Phase 2 — Capital.com data handoff package

**Owner:** Capital download agent on the separate host.

**Outcome:** a closed, non-secret, hash-verified package that allows the backtesting host to understand the acquired data without re-running the download.

Required records:

```text
READ_FIRST.md
INSTRUMENT_MANIFEST.csv
DATASET_MANIFEST.csv
SCHEMA_AND_FIELD_SEMANTICS.md
COVERAGE_AND_GAPS.csv
DOWNLOAD_COMMANDS_AND_RESULTS.md
PROTECTED_PERIOD_AUDIT.json
KNOWN_LIMITATIONS.md
TRANSFER_MANIFEST.json
```

Required manifest fields include legal-entity/account environment label, epic, instrument type, resolution, minimum/maximum UTC time, row count, file size, SHA-256, missing intervals, duplicates, metadata snapshot time, and protected-purpose classification.

**Acceptance:** local and remote hashes match; no credentials; 2026+ files are physically or manifest-separated before research use.

### Phase 3 — Minimal shared-core implementation

**Owner:** backtesting agent in an isolated safe worktree after approval.

**Outcome:** platform-aware ingestion and guards, no strategy logic.

Minimum changes:

1. platform/source identity in dataset and run manifests;
2. global rankable/protected filter before load;
3. source-separated raw/normalized readers;
4. canonical instrument identity table;
5. Capital.com bid/ask bar schema and field checks;
6. calendar/first-executable-time helper;
7. tests proving Kraken behavior is unchanged.

**Acceptance:** repository-native tests pass; no economic metric or candidate ranking is produced; protected rows cannot enter rankable loaders; existing Kraken fixtures remain bitwise or semantically unchanged as defined by the repository.

### Phase 4 — Documentation and registry revision

**Owners:** Donch for canonical project sources; backtesting agent for repository-local manuals and registries.

**Outcome:** current authority, registry, capability, and continuity agree.

**Acceptance:** source map, contract, continuity, hypothesis routes, data capability, evidence manual, review guide, orchestration, agent instructions, and provenance map are internally consistent; Kraken-specific docs remain correctly scoped; no historical decision is erased.

### Phase 5 — Capital.com data-quality and mechanics audit

**Owner:** backtesting agent.

**Outcome:** platform capability report, not an economic study.

Checks by asset class:

- instrument counts and duplicate epics;
- coverage, gaps, timestamp order, DST, and market closures;
- bid/ask consistency and crossed/zero spreads;
- expiry and undated-contract identity;
- current versus historical metadata limits;
- reported volume semantics;
- corporate-action and dividend-adjustment availability;
- overnight-financing history or absence;
- currency and account-conversion implications;
- lifecycle/survivorship cap;
- source-price versus underlying-price limitations.

**Acceptance:** each field is `verified`, `inferred`, `unavailable`, or `blocked`; no missing cost is represented as zero.

### Phase 6 — Hypothesis routing, no backtests

**Owner:** Donch with repository evidence from Phase 5.

**Outcome:** every current family receives a Capital.com and cross-platform route without changing evidence level.

Prioritise only a small first queue:

1. one Capital.com-native price/path family;
2. one true-session or market-reopen family;
3. one directed macro-to-Kraken cross-platform family;
4. same-underlying Capital.com/Kraken calibration as non-economic infrastructure evidence.

**Acceptance:** no threshold, symbol, asset class, or horizon is selected from outcomes; each proposed family has required data, controls, costs, and falsification.

### Phase 7 — First economic contract

**Owner:** Donch drafts; human approves; backtesting agent executes.

Only one contract should be authorised first. It should use the smallest sufficient platform module and avoid requiring financing/corporate-action history that the audit could not verify.

**Acceptance:** frozen event and control identities, source/target platform, PIT universe, period guard, costs, multiplicity family, evidence claim boundary, tests, archive, and independent review are all explicit before execution.

## 9. Recommended first research order

The first implementation proof should not be a broad 5,500-instrument search.

1. **Non-economic:** compare Capital.com BTC/ETH CFD timestamps, bid/ask structure, sessions, and price construction with Kraken data. This validates mapping and alignment only.
2. **Capital.com-native:** test one simple price/path mechanism on one asset class with clean trading hours and no unresolved overnight financing requirement.
3. **Cross-platform:** test one directed source-to-target mechanism, preferably a broad observable macro state into Kraken, with strict time and common-factor controls.
4. Expand to shares, ETFs, commodities, rates, and corporate catalysts only as their mechanics and data contracts are verified.

## 10. Main risks and controls

| Risk | Control |
|---|---|
| Scope explosion from thousands of instruments | Inventory all; research only predeclared eligible cohorts. |
| Universal backtester overfits incompatible products | Shared governance, source-specific execution adapters. |
| Current-roster survivorship | PIT lifecycle authority or explicit current-roster/bar-existence cap. |
| Broker CFD price mistaken for underlying price | Store broker/source identity and mapping confidence; use source-specific claims. |
| Financing, dividends, rolls, and FX conversion omitted | Require explicit cost/adjustment status; unavailable is not zero. |
| Closed-market lookahead | First-executable-target-time rule and calendar tests. |
| Trading-capable Capital.com credentials | Separate acquisition service, demo where feasible, prohibited order endpoints, secret isolation. |
| 2026 contamination | Physical/manifest separation and fail-before-load guards. |
| Failed Kraken family rescued by relabelling | New platform translation ID; old Kraken decision preserved. |
| Dirty repository overlay | Read-only preflight and isolated worktree. |
| Documentation divergence | One canonical contract and registry revision, then repository-local synchronization and Drive-verified archive. |

## 11. Stop conditions

Stop before implementation or economic work if:

- the exact Capital.com instrument/data manifest is unavailable;
- protected-period separation cannot be established without inspecting outcomes;
- API credentials or tokens could enter a package or research process;
- the repository is dirty and no safe isolated target exists;
- platform price, volume, financing, adjustment, or lifecycle semantics are being guessed;
- a cross-platform join cannot prove source observability and target executability;
- a proposed Capital.com test is merely a renamed rescue of a closed Kraken translation;
- the requested change exceeds the exact approval scope.

## 12. Immediate next action

Prepare a **self-contained read-only backtesting-agent task** for Phase 1. It should request no code changes and no economic run, require `drive_handoff: approved_default`, and ask the agent to return the repository-specific minimal architecture and documentation impact package. This should precede drafting final replacement source text because repository reality may narrow the design further.

## 13. Source basis

Current Donch sources:

- `00_READ_FIRST_Project_Source_Map.md`
- `01_AUTHORITY_QLMG_Operating_Contract_2026-07-16.md`
- `02_STATE_Master_Continuity_Brief_2026-07-16_rev8.md`
- `04_REGISTRY_Hypothesis_and_Family_Status_2026-07-16.csv`
- `05_REGISTRY_Kraken_Data_and_Evidence_Capability_2026-07-16.csv`
- `12_MANUAL_Test_and_Evidence_Standards_2026-07-16.md`
- `15_RUNBOOK_Human_Approval_and_Orchestration_2026-07-16.md`
- `DONCH_PROJECT_INSTRUCTIONS_FULL.md`
- `BACKTESTING_AGENT_OPERATING_AND_ARCHIVE_INSTRUCTIONS.md`
- `DONCH_TO_BACKTESTING_TASK_ARCHIVE_TEMPLATE.md`

Official Capital.com sources accessed 2026-07-18:

- Capital.com Public API documentation;
- Capital.com markets overview;
- Capital.com market-pricing documentation;
- Capital.com overnight-funding documentation;
- Capital.com dividend-adjustment documentation;
- Capital.com broker-price comparison documentation.
