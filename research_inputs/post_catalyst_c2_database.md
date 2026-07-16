# Post-Catalyst Continuation Base Catalyst Database

## Executive summary

I compiled a point-in-time-aware catalyst framework for systematic testing of a **post-catalyst continuation base** strategy on crypto perpetual futures, centered on assets that either had liquid perpetuals during the event window or later developed liquid perpetual markets on venues such as Binance, Bybit, Kraken, OKX, and CME/SEC-regulated channels. The working database contains **90 high/medium-confidence events** for 2020-01-01 through 2025-12-31, plus **10 excluded/noisy events** that should stay out of primary backtests. The strongest families for continuation-base testing are: **legal/regulatory overhang removal, ETF/institutional-access expansion, direct supply/float changes, and token-linked protocol utility changes**. These are the families where the mechanism plausibly persists beyond the first-event candle. Spot listings and new perpetual listings are abundant, but they are much noisier and should usually be handled as **event-day-exclusion / base-only** setups rather than breakout-chase events. citeturn14search0turn15search0turn14search2turn39search0turn21search0turn23search1turn17search1turn19search0

The source quality is uneven by family. **Court orders, SEC filings/orders, exchange listing notices, official chain/protocol blogs, and governance proposals** are the cleanest PIT sources. **Current contract-specification pages** such as Kraken’s derivatives specification page are still useful, but mainly for **effective first-trading dates** rather than first-public timestamps, so those rows should usually carry `first_public_ts_utc=unknown` and `official_confirm_ts_utc=unknown` with a note that the effective date is high-confidence but the original announcement timestamp is not recovered in this pass. citeturn28view0turn34view0

For C2 specifically, the most usable historical clusters are:

| Family | Why it is usable for C2 | Database depth |
|---|---|---:|
| Legal/regulatory repricing | Binary overhang removal or creation often changes who can hold, list, settle, or underwrite the asset | strong |
| ETF/institutional access | Creates durable new demand channels and flow-sensitive post-event digestion regimes | moderate |
| Supply shock / unlock change | Float changes are mechanically relevant to continuation vs failure | moderate |
| Protocol utility / fee / revenue linkage | Best when token capture is direct, not just “good product news” | moderate |
| Exchange access expansion | Many events, but event-day spikes are often too crowded; best as base/reclaim-only | strong |
| Leverage access expansion | Common but ambiguous; often increases noise and short-term two-sided flow | strong |
| Attention-only narrative shocks | Usually too noisy unless they become a broad sector/theme repricing | weak |

What should be deprioritized in primary backtests is not “weak price action in hindsight,” but **weak ex-ante mechanism clarity**: same-day meme listings, pure leverage additions with no new spot access, launchpool/airdrop events that were fully telegraphed, and social-attention bursts with no change in access, float, legal status, or token economics. That distinction matters because the C2 family is trying to isolate **post-event persistence**, not just volatility. citeturn23search1turn20search7turn17search1turn19search11

## Catalyst source matrix

| Source | Event types | Coverage | Timestamp quality | Free/paid | PIT confidence | Notes |
|---|---|---|---|---|---|---|
| SEC complaints, orders, statements | lawsuits, ETF filings/approvals, exchange- or token-classification shocks | BTC, ETH, XRP, BNB ecosystem, SOL/ADA/MANA/SAND and related legal shocks | usually high on filing/order date; intraday sometimes explicit | free | very high | Best primary source for U.S. legal/institutional catalysts. citeturn14search0turn14search1turn14search2turn39search0 |
| Court orders and litigation releases | summary judgment, appeals, settlements, dismissals | XRP, Grayscale/BTC ETF path | high on filing date; intraday often absent | free | very high | Best for legal-overhang removal events. citeturn15search0turn14search6turn14search12turn41search0turn41search10 |
| Binance listing and futures announcements | spot listing, deposits open, trading start, perp launch, margin expansion | very broad across mid-cap and high-beta perp universe | usually excellent; publication and effective times often explicit | free | high | Best exchange source for 2023–2025 alt catalyst history. citeturn4search0turn6search0turn8search0turn11search0turn31search0 |
| Kraken derivatives contract specs | first trading dates for perpetuals | broad alt-perp coverage, especially 2022–2025 | best for effective first-trading date; poor for first-public timestamp recovery | free | medium-high | Very useful for `effective_ts_utc`, less useful for announcement timing. citeturn28view0turn34view0 |
| OKX listing notices | pre-market futures, conversion-style derivatives access | selective but useful for 2024–2025 new-token discovery | good on published and effective times | free | high | Helpful for pre-market perp rows. citeturn24search0 |
| CME press releases | regulated futures launch | BTC/ETH institutional access pathway | high | free | very high | Important for pre-ETF institutional-access chronology. citeturn16search6turn16search2 |
| Ethereum / BNB / protocol blogs | burns, upgrades, staking, utility changes | ETH, BNB, DYDX, AAVE, LDO, PYTH | good to excellent | free | high | Best for token-linked utility or supply changes. citeturn21search0turn21search1turn23search1turn42search0turn42search8turn20search7turn20search2turn29search2turn29search1turn29search4 |
| Official project / governance docs | unlock schedule, staking proposals, inflation changes | ARB, OP, STRK, WLD, DYDX, AAVE | medium to high | free | high when the document is explicit | Best for float and tokenomics changes; still requires care on execution timing. citeturn37search0turn37search3turn38search4turn38search12turn17search1turn18search6 |
| Telegram / World official blogs | meaningful distribution or app access changes | TON, WLD | high on announcement date | free | high | Useful when integration changes actual distribution, not just branding. citeturn19search0turn19search10turn19search11turn19search13 |
| Reuters | fallback for widely documented but harder-to-retrieve primary court/market events | Grayscale court win, BlackRock ETF filing reaction | medium | free | medium | Good supporting source; use only when primary document retrieval is incomplete. citeturn41search0turn41search1turn41search10 |

The full-schema companion files prepared in the sandbox are named `post_catalyst_c2_catalyst_db_2020_2025_main.csv` and `post_catalyst_c2_catalyst_db_2020_2025_excluded.csv`. They mirror the schema requested in the prompt and keep `unknown` where timing or impact details were not recovered at high confidence.

## Main catalyst database

The full main database contains **90 high/medium-confidence rows**. To keep the report readable, the table below shows a **high-signal excerpt** across the families that matter most for C2. The companion CSV carries the full schema with fields including `known_perp_symbols`, `primary_source_url`, `supporting_source_urls`, `unknown_fields_mask`, and `notes_on_uncertainty`.

| event_id | ticker | mechanism_family | mechanism_subtype | direction | event_state | first_public_ts_utc | effective_ts_utc | durability_score_ex_ante | pre_run_risk | classification_note | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAT0001 | XRP | legal_regulatory_repricing | sec_complaint | short | confirmed | 2020-12-22 | 2020-12-22 | high | low | Clean negative legal-overhang shock. Useful as a “bad catalyst” class, not just positive-news library. | citeturn14search0 |
| CAT0002 | ETH | etf_institutional_access | regulated_futures_announce | long | announced | 2020-12-16 | 2021-02-08 | high | medium | CME’s ETH-futures announcement is a durable institutional-access signal; announcement and live date should be separate rows. | citeturn16search6 |
| CAT0003 | ETH | etf_institutional_access | regulated_futures_launch | long | executed | 2021-02-08 | 2021-02-08 | high | high | Live start of regulated ETH futures. | citeturn16search2 |
| CAT0004 | ETH | protocol_utility_fee_revenue_change | burn_path_launch | long | executed | 2021-07-15 | 2021-08-05 | high | high | EIP-1559 created direct fee-burn linkage to ETH. | citeturn21search0 |
| CAT0007 | BNB | supply_shock | auto_burn_mechanism_change | long | announced | 2021-12-22 | 2021-12-22 | medium | medium | Burn mechanism moved to formula-based Auto-Burn, increasing transparency and predictability. | citeturn23search1 |
| CAT0009 | OP | exchange_access_expansion | major_spot_listing | long | full_trading | 2022-06-01T01:00:00Z | 2022-06-01T04:00:00Z | medium | high | Strong access expansion, but launch/airdrop conditions imply high pre-run and chase risk. | citeturn32search0 |
| CAT0010 | OP | leverage_access_expansion | first_major_perpetual_listing | mixed | futures_live | unknown | 2022-06-02 | medium | high | Good example of why first-perp listings are ambiguous rather than automatically bullish. | citeturn34view0 |
| CAT0016 | APT | exchange_access_expansion | major_spot_listing | long | full_trading | 2022-10-18T01:04:00Z | 2022-10-19T01:00:00Z | medium | high | Major exchange access, but highly anticipated launch conditions. | citeturn13search2 |
| CAT0017 | APT | leverage_access_expansion | first_major_perpetual_listing | mixed | futures_live | unknown | 2022-10-19 | medium | high | Early major-perp access right into launch window. | citeturn28view0 |
| CAT0018 | OP | unlock_vesting_change | major_unlock | short | executed | 2023-05-31 | 2023-05-31 | high | high | Scheduled supply expansion belongs in the database because it can create failure-short or reclaimed-supply setups. | citeturn38search12 |
| CAT0020 | PEPE | exchange_access_expansion | major_spot_listing | long | full_trading | 2023-05-05T11:20:00Z | 2023-05-05T16:00:00Z | low | high | High-confidence event, low-durability mechanism. Good to keep, but not as a core positive family. | citeturn9search1 |
| CAT0023 | ARB | exchange_access_expansion | major_spot_listing | long | full_trading | 2023-03-23 | 2023-03-24T17:00:00Z | medium | high | Airdrop/listing transition event. Best treated as no-chase / post-base-only. | citeturn4search0 |
| CAT0025 | ARB | unlock_vesting_change | major_unlock | short | executed | 2024-03-16 | 2024-03-16 | high | high | Official docs imply circulating supply rose from about 1.537B to about 2.654B around the first large unlock. | citeturn37search0turn37search3 |
| CAT0029 | WLD | exchange_access_expansion | major_spot_listing | long | full_trading | 2023-07-24T07:02:00Z | 2023-07-24T09:00:00Z | medium | high | Massive attention and access shock, but event-day regime is unstable. | citeturn10search0 |
| CAT0031 | SEI | exchange_access_expansion | major_spot_listing | long | full_trading | 2023-08-01 | 2023-08-15T12:00:00Z | medium | high | Launchpool-to-listing transition is valid, but heavily telegraphed. | citeturn13search0 |
| CAT0035 | LDO | protocol_utility_fee_revenue_change | major_upgrade | long | executed | 2023-05-15 | 2023-05-15 | medium | medium | Lido V2 materially changed product utility via withdrawals. | citeturn29search2 |
| CAT0036 | AAVE | protocol_utility_fee_revenue_change | native_stablecoin_launch | long | executed | 2023-06-06 | 2023-07-16 | medium | medium | GHO matters because it is a new product/revenue path, not a generic roadmap item. | citeturn20search8turn20search2 |
| CAT0037 | DYDX | protocol_utility_fee_revenue_change | chain_mainnet_launch | long | executed | 2023-09-07 | 2023-10-26 | high | medium | dYdX Chain changed DYDX from governance token to chain/security/fee-utility token. | citeturn20search0turn20search7 |
| CAT0038 | DYDX | protocol_utility_fee_revenue_change | full_production_trading_enablement | long | executed | 2023-11-28 | 2023-11-28 | high | medium | For DYDX, “trading enabled” is more economically relevant than chain-genesis alone. | citeturn20search3turn20search7 |
| CAT0044 | PYTH | exchange_access_expansion | major_spot_listing | long | full_trading | 2024-02-02T07:50:00Z | 2024-02-02T12:00:00Z | medium | medium | Good access event, but less clean than legal/ETF/float changes. | citeturn8search0 |
| CAT0045 | PYTH | protocol_utility_fee_revenue_change | staking_governance_launch | long | executed | 2024-01-17 | 2024-01-17 | medium | medium | Direct token utility via official staking/governance. | citeturn29search1turn29search15 |
| CAT0049 | WIF | leverage_access_expansion | first_major_perpetual_listing | mixed | futures_live | 2024-01-18T07:25:00Z | 2024-01-18T14:15:00Z | medium | high | Useful example of derivatives access arriving before Binance spot. | citeturn11search1 |
| CAT0051 | BTC | etf_institutional_access | spot_etf_filing | long | announced | 2023-06-15 | unknown | high | medium | BlackRock’s filing is one of the cleanest filing-based catalysts in the sample. | citeturn16search0 |
| CAT0057 | XRP | legal_regulatory_repricing | court_ruling_secondary_sales | long | confirmed | 2023-07-13 | 2023-07-13 | high | low | Major legal-overhang relief for secondary-market XRP trading. | citeturn15search0 |
| CAT0058 | BTC | legal_regulatory_repricing | court_ruling_etf | long | confirmed | 2023-08-29 | 2023-08-29 | high | medium | Grayscale court win materially changed ETF path expectations. | citeturn41search0 |
| CAT0060 | BTC | etf_institutional_access | spot_etf_approval | long | confirmed | 2024-01-10 | 2024-01-10 | high | medium | Direct U.S. spot-BTC ETF approval. One of the best C2 categories in the whole sample. | citeturn14search2 |
| CAT0077 | ETH | protocol_utility_fee_revenue_change | major_upgrade_l2_fee_reduction | long | executed | 2024-02-27 | 2024-03-13T13:55:00Z | medium | high | Dencun matters, but token capture is less direct than 1559. | citeturn21search1 |
| CAT0079 | ETH | etf_institutional_access | spot_etf_approval | long | confirmed | 2024-05-23 | 2024-05-23 | high | medium | U.S. spot-ETH ETF approval. | citeturn39search0 |
| CAT0080 | ETH | etf_institutional_access | spot_etf_trading_launch | long | executed | 2024-07-23 | 2024-07-23 | high | high | Live launch of U.S. spot-ETH ETFs. | citeturn15search11turn16search15 |
| CAT0082 | WLD | unlock_vesting_change | unlock_delay_extension | long | confirmed | 2024-07-16 | 2024-07-24 | high | low | One of the cleanest positive float-overhang reductions: 80% of TFH team/investor WLD extended from 3 to 5 years. | citeturn17search1 |
| CAT0084 | TON | major_integration_distribution_access | telegram_revenue_share | long | executed | 2024-03-31 | 2024-03-31 | high | low | Telegram ad-revenue sharing is a real distribution/incentive shock, not a generic partnership headline. | citeturn19search0 |
| CAT0086 | WLD | major_integration_distribution_access | world_app_mini_apps | long | confirmed | 2024-10-17 | 2024-10-17 | medium | medium | World App 3.0 improves distribution, but token capture remains less direct than ETF/legal/float families. | citeturn19search11 |
| CAT0096 | XRP | legal_regulatory_repricing | settlement_framework | long | confirmed | 2025-05-08 | 2025-05-08 | high | medium | Major late-cycle legal-overhang reduction. | citeturn14search6 |
| CAT0097 | XRP | legal_regulatory_repricing | appeals_dismissed | long | dismissed | 2025-06-27 | 2025-06-27 | high | low | Final appeal dismissal is the cleanest “legal cloud removed” row for XRP in the sample. | citeturn14search12 |
| CAT0098 | HYPE | leverage_access_expansion | premarket_futures_launch | mixed | futures_live | 2024-12-04 | 2024-12-04T13:30:00Z | medium | high | Good example of a pre-market futures catalyst that should be tested separately from standard perp launches. | citeturn24search0 |
| CAT0099 | PYTH | protocol_utility_fee_revenue_change | oracle_integrity_staking | long | confirmed | 2024-09-02 | 2024-late-2024 | medium | medium | Meaningful utility expansion, but rollout timing is less crisp than a listing or court ruling. | citeturn29search4 |

### Normalization notes for the schema

Three schema conventions are worth making explicit. First, when the primary source disclosed only a filing/publication **date** but not an exact UTC time, I kept the field at date granularity rather than inventing an intraday timestamp. Second, for contract-specification pages such as Kraken’s, I treated the listed **“First Trading”** date as the high-confidence `effective_ts_utc` and left `first_public_ts_utc` and `official_confirm_ts_utc` as `unknown` unless a separate launch notice was retrieved. Third, `known_perp_symbols` is intentionally partial in some rows, because the goal is not to guess every venue symbol from memory but to provide enough deterministic mapping for symbol-join work without inventing unsupported venue coverage. citeturn28view0turn34view0

## Low-confidence and excluded events

These are not “false” events. They are events that should be excluded from the **primary** C2 backtest set because the mechanism is too weak, the catalyst is mostly attention, the event is duplicative, or the timing signal is too noisy.

| event_id | ticker | exclusion reason | comment | source |
|---|---|---|---|---|
| CAT0076 | BTC | attention-only / duplicate | “BlackRock buzz” is not a separate catalyst once the actual SEC filing is already captured. | citeturn41search3 |
| CAT0012 | ADA | mature-asset perp launch too weak | Kraken first-perp date is real, but novelty/demand shock is too small for primary C2 testing. | citeturn28view0 |
| CAT0013 | AVAX | mature-asset perp launch too weak | Same issue as ADA: legitimate event, weak ex-ante catalyst purity. | citeturn28view0 |
| CAT0014 | BCH | mature-asset perp launch too weak | Keep for reference, exclude from core tests. | citeturn28view0 |
| CAT0033 | SHIB | leverage-only on meme asset | High volatility, low mechanism clarity. Better used only in robustness checks. | citeturn34view0 |
| CAT0063 | PYTH | second-order margin expansion | Margin-enable events are usually too incremental relative to first spot/perp access. | citeturn8search0 |
| CAT0092 | SAGA | secondary-universe perp event | Valid effective date, but outside the cleanest liquid-universe priority list. | citeturn34view0 |
| CAT0093 | BB | secondary-universe perp event | Same reason as SAGA. | citeturn28view0 |
| CAT0094 | BEAM | secondary-universe perp event | Same reason as SAGA and BB. | citeturn28view0 |
| CAT0020 | PEPE | spot-listing mania | Keep in secondary tests if you want meme sectors, but not in a clean mechanism-first baseline. | citeturn9search1 |

The boundary here is practical. A clean C2 backtest should start with events where the market can plausibly be repricing **future access, future float, future legal constraints, or future cash-flow/usage linkage**. “Everyone noticed it at once” is not enough. citeturn14search2turn17search1turn19search0turn20search7

## Backtesting plan

### Ingestion and PIT handling

Store the catalyst file as an **event ledger**, not as a factor matrix. Each row should remain one event, with three timing columns kept distinct:

- `first_public_ts_utc`
- `official_confirm_ts_utc`
- `effective_ts_utc`

For execution logic, define the **tradable anchor timestamp** as the earliest timestamp that would have been observable to your strategy without lookahead and that still maps to a live market. In practice:

- use `first_public_ts_utc` for public filings, court rulings, and exchange announcements when time is known;
- use `official_confirm_ts_utc` when the event leaked or was rumored earlier but only became official later;
- use `effective_ts_utc` for exchange trading opens, futures-live states, hard-fork activations, unlock execution, or ETF trading launch.

If a row only has a date and no intraday time, treat it conservatively as **not tradable until the next session boundary you can implement consistently**, rather than assuming midnight UTC. This is the simplest way to avoid phantom precision and accidental lookahead. The separation is directly motivated by the disclosure patterns in SEC filings/orders, Binance notices, protocol upgrade posts, and governance/unlock docs. citeturn14search2turn39search0turn21search1turn37search0turn38search12

### Joining the catalyst file to Bybit perpetual symbols

Use a symbol-resolution table with at least these columns:

| asset_id | canonical_ticker | bybit_linear_symbol | first_live_ts | delist_ts | venue_aliases |
|---|---|---|---|---|---|

The join should happen in two steps. First, map the catalyst row by `asset_id` to a canonical asset record. Second, map that asset record to the **Bybit linear USDT perpetual that was live at the event anchor timestamp**. If the Bybit contract was not yet live, either:

- drop the row from the **primary Bybit-only sample**, or
- send it to a **cross-venue validation sample** using Binance/Kraken/OKX perps or spot, depending on what was live.

That rule is especially important for spot-listing events that happened **before** Bybit had a liquid perp. Otherwise the backtest quietly turns into “would later-listed perps have reacted to a historical spot listing,” which is not a tradable PIT experiment. citeturn28view0turn34view0turn24search0

### Event families and recommended C2 test rules

These are **inference-based trading-test rules**, not historical-return labels. They are derived from the persistence logic implied by the event mechanisms above. citeturn14search2turn39search0turn17search1turn19search0turn20search7turn23search1

| Event family | Minimum wait after event | Base length to test | Hold-above-event-low rule | Exclude event-day chase | Failure-short candidate |
|---|---:|---:|---|---|---|
| ETF / institutional-access approval or launch | 1–10 days | 1–5 days for breakouts; 3–10 days for VWAP reclaims | strict | yes | yes, especially on approval-into-pre-run |
| Legal overhang removed | 1–7 days | 2–10 days | strict | yes | yes, if market loses event VWAP after day 1 |
| Legal overhang created | 0–5 days | 1–5 days | inverse rule for shorts: fail-to-reclaim event VWAP | yes | primary short family |
| Direct supply reduction / lock-up extension | 1–10 days | 2–15 days | strict | yes | less often; mainly if extension was pre-run and reverses |
| Major unlock / float increase | 0–7 days | 1–7 days | for longs, only if price reclaims event VWAP after absorbing supply | yes | primary short family |
| Protocol utility / fee / burn / revenue linkage | 1–14 days | 2–15 days | strict | yes | only if token linkage proves weak and event low breaks |
| Major spot listing / relisting | 1–5 days | 1–10 days | moderate to strict | yes | yes, often one of the best failure-short families |
| First major perpetual listing | 1–5 days | 1–7 days | moderate | yes | yes; leverage events often overshoot then mean-revert |
| Pre-market futures / roadmap-to-trading transitions | 1–5 days | 1–7 days | moderate | yes | yes; use separate parameter set |
| Integration / distribution access | 2–15 days | 3–20 days | strict | yes | only if token-capture link is weak or largely narrative-driven |

### Strategy experiment design

The right benchmark is not “did catalysts go up.” It is:

1. **Catalyst-base entry**
   - wait for event;
   - define event high, event low, and event VWAP;
   - require post-event digestion;
   - enter only on:
     - breakout from a multi-bar base above event VWAP, or
     - reclaim of event VWAP after initial digestion.

2. **Generic breakout control**
   - same assets, same holding periods, same volatility/volume filters;
   - but without reference to catalyst dates;
   - use identical breakout and stop logic.

3. **Matched non-catalyst controls**
   - for each catalyst event, find non-event days in the same asset with similar realized volatility, ATR expansion, and trend status;
   - compare continuation probability after a base forms.

4. **Event-family stratification**
   - evaluate separately for:
     - legal/ETF,
     - spot listing,
     - leverage listing,
     - unlock/float,
     - utility/revenue,
     - integration/distribution.

That is the cleanest way to test whether the “base after a real repricing mechanism” adds signal beyond ordinary continuation. On the current evidence, the families most likely to survive this comparison are **legal/ETF**, **float change**, and the best **token-linked utility** events. Spot and perp listings will probably produce more rows, but they are exactly where you should expect the generic-breakout benchmark to catch up. citeturn14search2turn39search0turn17search1turn20search7turn23search1

### Avoiding lookahead

Avoid lookahead in four places.

First, do not use the existence of a later Bybit perp to justify including an event that was only tradable in spot at the time. Second, do not fill in missing intraday timestamps with invented times. Third, do not use hindsight to reclassify durability based on later returns. Fourth, separate **announcement**, **confirmation**, and **effective** events that occur on different dates; otherwise ETF and upgrade events get compressed into one synthetic row and your backtest can enter before the market actually had access. citeturn16search0turn14search2turn39search0turn21search1turn37search0

## Backtesting-agent prompt

Use the following compact prompt as the next-stage testing prompt:

```text
You are a backtesting agent for crypto perpetual futures.

Objective:
Test a post-catalyst continuation base strategy using a point-in-time catalyst database.

Inputs:
- catalyst CSV with fields:
  event_id, asset_id, ticker, known_perp_symbols, mechanism_family, mechanism_subtype,
  direction, event_state, first_public_ts_utc, official_confirm_ts_utc, effective_ts_utc,
  source_confidence, primary_source_type, primary_source_url, supporting_source_urls,
  headline_raw, classification_note, durability_score_ex_ante, estimated_float_impact_pct,
  estimated_access_impact, pre_run_note, pre_run_risk, event_low_px_available,
  event_vwap_available, unknown_fields_mask, notes_on_uncertainty
- Bybit linear USDT perpetual OHLCV + funding + open interest if available

Rules:
- Do not use events before the relevant Bybit perp was live
- Use the earliest tradable PIT timestamp among first_public_ts_utc, official_confirm_ts_utc,
  and effective_ts_utc that is actually observable and not missing
- If timing is date-only, do not invent intraday precision
- Exclude low-confidence/excluded-event file from primary tests
- Do not classify event quality by hindsight return

Strategy logic:
- Define event day/session from tradable PIT timestamp
- Compute event high, event low, event VWAP
- Exclude event-day chase entries
- Only allow long continuation entries when:
  1) price holds above event low
  2) a post-event base forms
  3) price either breaks the base or reclaims event VWAP
- For negative catalysts, test symmetric short logic:
  1) price fails to reclaim event VWAP or breaks lower after base
  2) use event high as invalidation where appropriate

Test grid by event family:
- ETF/legal positive: wait 1–10 days, base 2–10 days
- Spot listing/relisting: wait 1–5 days, base 1–10 days
- First perp listing: wait 1–5 days, base 1–7 days
- Unlock increase: short-first family, wait 0–7 days, base 1–7 days
- Unlock delay/reduction: long family, wait 1–10 days, base 2–15 days
- Protocol utility/revenue: wait 1–14 days, base 2–15 days
- Integration/distribution: wait 2–15 days, base 3–20 days

Outputs:
- overall performance
- performance by mechanism_family and mechanism_subtype
- performance by durability_score_ex_ante
- performance by pre_run_risk
- comparison against generic non-catalyst breakout controls
- list of event families that add signal vs generic breakouts
- list of event families that are too noisy or only work as failure-shorts
- all assumptions and all dropped rows
```

## Open questions and limitations

The strongest limitation is not event identification; it is **timestamp granularity**. Exchange announcements often provide exact times, but court rulings, governance proposals, and current contract-spec pages often do not. Those rows are still useful, but they need conservative execution rules. citeturn28view0turn34view0turn37search0turn38search12

The second limitation is family imbalance. Exchange listings and leverage listings are abundant, while truly clean float-change and token-linked utility events are much rarer. That is a feature of the market, not a defect in the database. It means the best C2 version will likely come from **fewer, higher-quality catalyst families**, not the widest possible event count. The cleanest rows are the ones where the market had to digest a new legal reality, a new institutional access path, or a real future float change. The noisiest rows are the ones where an exchange simply turned another lever on and gave the crowd a brighter screen.