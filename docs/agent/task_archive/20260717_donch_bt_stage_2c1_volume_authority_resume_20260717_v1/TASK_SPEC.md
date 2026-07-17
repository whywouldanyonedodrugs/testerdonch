# Stage 2C1 — Candle-Volume Authority and Stage 2C Resume

```text
task_id: donch_bt_stage_2c1_volume_authority_resume_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
capture_access: no
public_network_reads: bounded data-engineering use only
```

## Objective

Resolve the historical Kraken candle-volume unit blocker with a small official-source and public-data calibration, then resume Stage 2C using a clearly labeled lagged liquidity proxy.

This task does not calculate economic outcomes.

## Accepted starting point

```text
required lineage commit:
    9b4fbe1bcf9e8a0f79fedde758534dc0d7c86611

blocked task:
    donch_bt_stage_2c_c01_event_contract_20260717_v1

blocker:
    historical candle volume unit and exact quote-volume meaning unverified
```

Record the actual clean synchronized starting commit. Do not stop solely for routine archive-only descendants.

## Clarified decision

The original Stage 2C requirement for authoritative exact traded USD notional was unnecessarily strict for a non-economic cohort-hygiene gate.

The permitted gate is now:

```text
liquidity_field:
    close_based_usd_volume_proxy

five_minute_value:
    trade_close_5m * verified_base_volume_5m

daily_value:
    sum(five_minute_value)

daily cohort:
    top 100 eligible crypto perpetuals by prior-30-calendar-day
    median daily close_based_usd_volume_proxy

minimum valid days:
    20

ranking time:
    once per UTC day using data through the prior UTC day
```

This is a causal activity/liquidity proxy. It is not exact quote volume, observed executable liquidity, capacity, spread, depth, or slippage evidence.

Do not call it `traded_usd_notional` or `quote_volume`.

Use it only if the candle `volume` field is verified as base-asset quantity for the relevant linear PF contracts under the checks below.

## Official sources to archive

Archive exact bytes or stable rendered exports, access time, URL, response metadata, and SHA-256 for:

1. Kraken Futures Market Candles documentation:
   `https://docs.kraken.com/api/docs/futures-api/charts/candles`
2. Kraken Futures public trade-history documentation:
   `https://docs.kraken.com/api/docs/futures-api/trading/get-history`
3. Kraken linear multi-collateral contract specifications:
   `https://support.kraken.com/articles/4844359082772-linear-multi-collateral-derivatives-contract-specifications`
4. EEA contract specifications where applicable:
   `https://support.kraken.com/articles/perpetual-contract-specifications-for-clients-in-the-eea`
5. Archived official Kraken versions of the linear contract-specification page covering 2023, 2024, and 2025 where retrievable.

The contract-specification evidence must show the base currency and base-unit lot/position convention. Preserve any change or ambiguity by symbol and effective interval.

## Bounded public calibration

Current 2026 public trade and candle reads are authorized only for field-unit/data-engineering calibration. They must not be joined to strategy outcomes or used to select symbols, thresholds, directions, or economic rules.

Use at least four currently tradeable linear PF contracts spanning materially different base currencies and lot scales, including:

```text
PF_XBTUSD
PF_ETHUSD
one ordinary alt with min lot 1
one scaled or fractional-lot alt
```

Freeze the exact symbols before requesting data.

For each symbol:

1. choose at least three completed five-minute intervals;
2. retrieve every public trade in each interval using the public history endpoint and pagination/backward iteration as needed;
3. retrieve the matching official trade candle;
4. compare:
   `sum(public_trade.size)` versus `candle.volume`;
5. verify timestamp boundary semantics and that no trade is omitted or double-counted;
6. record exact decimal difference.

Required minimum:

```text
symbols: >= 4
complete five-minute intervals per symbol: >= 3
total comparison intervals: >= 12
```

Do not use protected strategy outcomes. The only retained current fields may be:

```text
symbol
trade timestamp
trade size
trade price only where needed for interval identity
candle timestamp
candle volume
source identity
```

Do not retain post-interval returns or strategy joins.

## Historical semantic check

Before applying the unit to 2023–2025:

- verify that each candidate PF symbol uses the same linear base-unit convention in the relevant official or archived-official contract specification;
- identify any symbol or interval with migration, multiplier, redenomination, scaled-token, or unit ambiguity;
- exclude ambiguous symbols/intervals from the liquidity ranking;
- include the volume-authority source hash and semantic-version identifier in cohort identity.

Current semantics alone may not silently overwrite a historical contract change.

## Pass rule

Classify candle volume as verified base quantity only when all are true:

```text
official contract specification expresses order/position scale in base units
candle volume equals complete summed public trade size for every calibration interval
no unexplained rounding or multiplier difference
historical official specification is consistent for the interval used
```

Use exact decimal arithmetic. A documented display rounding tolerance may be accepted only when it is derived from official precision fields and written into the contract before comparison.

If any calibration interval materially fails, do not average away the failure. Investigate only within this narrow unit question. If unresolved, stop.

## Stage 2C resume

If the pass rule succeeds:

1. create and hash `KRAKEN_CANDLE_VOLUME_AUTHORITY.csv`;
2. create `KRAKEN_CANDLE_VOLUME_VALIDATION.md`;
3. update the data-capability authority to distinguish:
   - `base_volume: verified for listed PF symbol/intervals`;
   - `exact_quote_volume: unavailable`;
   - `close_based_usd_volume_proxy: permitted for causal cohort hygiene`;
4. resume the previously approved Stage 2C task;
5. use the clarified top-100 proxy gate exactly as specified here;
6. preserve every other Stage 2C rule unchanged.

Do not rerun Stage 2B or change C01 shock/path thresholds.

## Required tests

Synthetic tests must cover:

- complete trade aggregation equals candle volume;
- missing, duplicated, paginated, and boundary trades;
- wrong symbol or candle interval;
- fractional and scaled lot sizes;
- historical semantic-version mismatch;
- exact decimal arithmetic;
- future-day data cannot affect prior cohort ranks;
- proxy is named and labeled correctly;
- no exact-quote-volume or capacity claim;
- no outcome or protected-strategy fields.

Run the smallest relevant repository regression and guard suites.

## Outputs

In addition to the unchanged Stage 2C outputs, retain:

```text
KRAKEN_CANDLE_VOLUME_SOURCE_LEDGER.csv
KRAKEN_CANDLE_VOLUME_AUTHORITY.csv
KRAKEN_CANDLE_VOLUME_CALIBRATION.csv
KRAKEN_CANDLE_VOLUME_VALIDATION.md
LIQUIDITY_PROXY_CONTRACT.md
```

## Acceptance criteria

```text
candle base-volume unit: verified or blocked with exact evidence
exact quote volume claimed: no
liquidity proxy label: close_based_usd_volume_proxy
current calibration used for alpha: no
economic outputs computed: zero
protected strategy outcomes opened: zero
Stage 2C resumed only after unit pass: yes
independent review: approve
```

## Failure response

If base-volume authority cannot be established:

- make no C01 implementation change;
- do not substitute raw volume, `price × volume`, market cap, current max position, or another gate;
- preserve the evidence package;
- return the exact unresolved semantic conflict.

## Integration and handoff

After tests and review pass:

- create one task-scoped commit;
- integrate and non-force push under the standing reviewed-task workflow;
- upload the closed package using `drive_handoff: approved_default`;
- omit current raw trade/candle payloads from Drive;
- retain them locally under the task archive with hashes;
- verify remote size and SHA-256.

## Final response

```text
status:
actual_starting_commit:
official_sources:
calibration_symbols:
calibration_intervals:
candle_trade_size_comparisons:
historical_semantic_check:
base_volume_authority:
exact_quote_volume_status: unavailable
liquidity_proxy_contract:
Stage_2C_status:
onset_event_count:
economic_contract_draft_hash:
tests_and_results:
protected_outcomes_opened: no
economic_outputs_computed: no
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
next_status:
human_approval_required:
```
