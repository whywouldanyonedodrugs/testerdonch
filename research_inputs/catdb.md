# C2 Catalyst Database v2.1
## Independent adversarial audit and consolidated event register

Audit period: 2020-01-01 through 2025-12-31  
Input records: 100  
Consolidated logical records: 98  
Audit schema: fixed audited event record format  
Final suitability decision: `suitable_for_sample_limited_ingestion_only`  
Collection characterization: `only a source-verified seed database`


## 1. Executive audit summary

The three reports contain a useful primary-source backbone, but they do not form a complete or uniformly deterministic catalyst database. The audit found two exact cross-report duplicate submissions, four semantic-duplicate records, inconsistent mechanism labels for regulated derivatives, several cases where approval or publication was treated as same-day effective access, and a material record-level URL defect in the protocol report.

After exact deduplication, 98 logical records remain. Of these, 59 meet the high-confidence standard, 27 remain medium confidence, and 12 are excluded. The high-confidence register is suitable for bounded, sample-limited ingestion. The full collection is not suitable for unrestricted deterministic ingestion because first-public timing, live-effect timing, basket identity, and historical source continuity remain unresolved in part of the medium register.

| Metric | Count |
|---|---:|
| Total submitted event records | 100 |
| Logical records after exact-deduplication | 98 |
| High-confidence records after audit | 59 |
| Medium-confidence records after audit | 27 |
| Excluded records after audit | 12 |
| Exact duplicate submitted records | 2 |
| Exact duplicate pairs | 2 |
| Semantic duplicate records | 4 |
| Catalyst-cluster count, all logical records | 44 |
| Catalyst-cluster count, included records | 37 |
| Verified primary-source-backed logical records | 91 |
| Unique non-unknown primary-source URLs | 87 |
| Unresolved conflict groups | 18 |
| Source families with incomplete coverage | 8 |

The collection does not demonstrate a closed search universe. None of the reports supplies a reproducible query log, source-by-year negative-search ledger, asset-by-venue coverage grid, or stopping rule. The appropriate characterization is therefore `only a source-verified seed database`, even where individual records are well grounded.


## 2. Input-report inventory

| Input report | Submitted | Collector high | Collector medium | Collector excluded | Audit note |
|---|---:|---:|---:|---:|---|
| Legal/regulatory/ETF/institutional-access report | 39 | 24 | 9 | 6 | Strongest on SEC, U.S. courts, CFTC/CME/Cboe, and North American ETF materials; selective rather than exhaustive. |
| Protocol utility/fees/supply/float report | 27 | 17 | 5 | 5 | All record-level primary-source fields were prose labels rather than URLs; URLs were recoverable from the reference list. |
| Exchange/leverage/integration/distribution report | 34 | 24 | 5 | 5 | Strongest on Coinbase, OKX, Binance, CME, PayPal/Venmo, and Robinhood; weak on historical Bybit and Telegram-native archives. |

Collector totals before audit were 65 high-confidence, 19 medium-confidence, and 16 excluded records. Those counts were not additive across reports because two events were independently collected twice. The protocol report’s reference list preserved most underlying URLs, but every one of its 27 fixed record fields labeled “Primary source URL” contained prose rather than a URL. The audit repaired those fields from the report’s own reference section rather than adding new events.


## 3. Source-verification results

Strict record-level source support passed for 91 of 98 logical records. There are 87 unique non-unknown primary-source URLs because several asset-level records correctly share one complaint, order, or multi-asset announcement.

Source-result categories:

| Result | Logical records | Interpretation |
|---|---:|---|
| Direct official source supports event identity and recorded phase | 91 | URL and event mechanism are sufficiently aligned for high or medium classification. |
| Official source is retrospective or current-state only | 3 | SUSHI-2022-KANPAI-20, STRK-2024-UNLOCK-SCHEDULE-REVISION, and APE-2022-STAKING-LAUNCH. |
| Indirect or redirected official ecosystem source | 1 | C2-MC-004, TON/Telegram distribution. |
| Official source supports a related phase rather than the claimed launch phase | 1 | STRK-2024-STAKING-LAUNCH. |
| Source-event date mismatch | 1 | The FCA URL cited for the recorded 2024 event is a 2025 page and does not independently verify the stated 2024 timestamp. |
| No primary source | 1 | The bankruptcy basket has no event identity, URL, or asset-level timestamp. |

The most common source defect was not a dead URL but a phase mismatch: an approval page was used as an effective-access timestamp, a current documentation page was used as a contemporaneous revision announcement, or a later official retrospective was used in place of the original governance execution artifact.


## 4. Timestamp audit

| Timestamp precision | Logical records |
|---|---:|
| second | 2 |
| minute | 17 |
| date_only | 74 |
| month | 2 |
| month_and_date_only | 1 |
| quarter_and_date_only | 1 |
| unknown | 1 |

The distribution is record-level and uses the most specific timestamp precision retained anywhere in the record. A minute- or second-level label does not imply that all three timestamp fields have that precision.

Material corrections:

1. Coinbase FCM approval was separated from later customer access; effective timestamps are now unknown.
2. Coinbase Derivatives SOL and XRP self-certification dates are retained as earliest permissible dates, while first-trade timing remains unresolved.
3. The Ripple settlement framework, CAKE Tokenomics 3.0, and dYdX buyback program no longer use publication or approval as an automatically effective timestamp.
4. PayPal UK and PayPal LINK/SOL records now distinguish rollout announcement from effective availability.
5. Ethereum London and BNB BEP-95 no longer treat the retrieved activation announcement as the earliest public disclosure of the underlying proposal.
6. Chainlink v0.1 and v0.2 were upgraded to minute precision because the official launch schedules state 12:00 p.m. Eastern Time; these convert to 17:00 UTC on the relevant dates.
7. Shapella and Pectra retain second precision because the official Ethereum materials state exact UTC activation times.
8. The Arbitrum AIP-1.1 Snapshot time is retained at minute precision, while the later on-chain execution remains a separate date-level phase.

Counts by first-public year:

| First-public year | All logical records | Included records |
|---|---:|---:|
| 2020 | 5 | 5 |
| 2021 | 14 | 14 |
| 2022 | 6 | 5 |
| 2023 | 36 | 32 |
| 2024 | 13 | 10 |
| 2025 | 17 | 14 |
| unknown | 7 | 6 |


## 5. Duplicate and catalyst-cluster audit

Two submitted records were exact duplicates and were absorbed into one consolidated record each.

| Absorbed submitted record | Retained consolidated record | Basis |
|---|---|---|
| C2_XRP_2020_12_28_COINBASE_SUSPENSION | AUD_XRP_2021_01_19_COINBASE_FULL_SUSPENSION | Same asset, source, announced suspension, and 2021-01-19T18:00:00Z effective state as C2-HC-003. |
| C2_ETH_2021_02_08_CME_ETHER_FUTURES | AUD_ETH_2021_02_08_CME_FUTURES_LAUNCH | Same asset, CME launch source, first-public date, and 2021-02-08 effective date as C2-HC-004. |

Four submitted records are semantic duplicates rather than exact content duplicates.

| Excluded semantic-duplicate record | Cluster treatment |
|---|---|
| C2_EXCL_2023_06_06_SEC_COINBASE_OVERLAP_SOL_ADA_MATIC_FIL | Retained only as supporting evidence for assets first labeled in the 2023-06-05 Binance complaint. |
| C2_EXCL_2023_11_20_SEC_KRAKEN_OVERLAP | Retained only as later venue-specific support inside SEC_ASSET_SECURITY_LABELING_2023. |
| C2_EXCL_2025_ALT_ETF_EXTENSIONS_ROUTINE | Treated as procedural continuation of first-filed ETF pathways. |
| C2-EX-001 | Treated as same-venue TON leverage continuation after migration/rename. |

The consolidated register contains 44 catalyst clusters across all logical records and 37 clusters among included records. Multi-phase clusters were preserved when the information state changed: XRP complaint, limit-only state, full suspension, court clarification, relisting, and settlement framework; BTC spot-ETF denial, vacatur, approval, and launch; Ethereum fee, issuance, withdrawal, scalability, and staking-efficiency upgrades; PayPal/Venmo acquisition, checkout, wallet, transfer, and asset-expansion phases; and exchange spot versus perpetual access phases for TON, SUI, SHIB, and PEPE.

A later venue listing was not treated as a duplicate solely because another major venue listed the same asset earlier. That collector rule incorrectly excluded the Binance PEPE spot and perpetual launches; both were restored because they added distinct venue and participant access.


## 6. Mechanism conflict audit

Mechanism families were normalized to seven values: `legal_regulatory_repricing`, `regulated_institutional_access`, `exchange_spot_access`, `leverage_access`, `distribution_integration`, `protocol_utility_fee_revenue`, and `supply_float`.

Nine direction labels were changed. Eight regulated-futures or FCM records labeled `long` were changed to `mixed` because the ex-ante mechanism expands long, short, and hedging access. Ethereum Dencun was changed from `unknown` to `mixed`: it expands rollup utility while changing fee-demand and burn channels. Spot listings, brokerage additions, and wrapper launches remain `long`; access removals remain `short`; migrations and withdrawal-enablement events remain `mixed` where both utility and available-float channels change.

The broad collector family `etf_institutional_access` was split. ETF approvals, ETF launches, and regulated-wrapper events remain `regulated_institutional_access`. Futures launches, FCM distribution, margin futures, perpetuals, and margin enablement are classified as `leverage_access`.

Ticker identity was made point-in-time explicit. The 2023 Polygon records retain MATIC and list POL only as a later venue symbol. Render records use RNDR/RENDER. The Maker record remains MKR with later SKY mapping noted. TONCOIN and TON are treated as one asset identity with a ticker transition. DYDX records retain an identity warning because venue symbols can refer to different token representations.

Durability and pre-run risk were audited only against the mechanism and information state recorded in the source. They remain qualitative metadata. They should not be used as deterministic filters without an externally defined coding rubric.


## 7. High-confidence consolidated event register

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_001
Original collector event IDs: C2-HC-001
Catalyst cluster ID: PAYPAL_VENMO_CRYPTO_DISTRIBUTION_2020_2025
Asset ID: btc_eth_bch_ltc_basket
Ticker: BTC|ETH|BCH|LTC
Known major perp symbols: BTCUSDT; ETHUSDT; BCHUSDT; LTCUSDT
Mechanism family: distribution_integration
Mechanism subtype: payments_wallet_buy_hold_sell_launch
Direction: long
Event state: launched
First public timestamp UTC: 2020-10-21
Official confirmation timestamp UTC: 2020-10-21
Effective timestamp UTC: 2020-10-21
Timestamp precision: date_only
Source confidence: high
Primary source type: official corporate press release
Primary source URL: https://newsroom.paypal-corp.com/2020-10-21-PayPal-Launches-New-Service-Enabling-Users-to-Buy-Hold-and-Sell-Cryptocurrency
Supporting source URLs: https://newsroom.paypal-corp.com/news-cryptocurrency?l=50
Source publication timestamp: 2020-10-21
Raw official headline or title: PayPal Launches New Service Enabling Users to Buy, Hold and Sell Cryptocurrency
Mechanism classification: retail wallet-native buy/hold/sell access inside PayPal
Ex-ante durability: high
Estimated access impact: very high
Estimated float impact: low-to-moderate secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: exact intraday rollout order by user cohort; any state-by-state exclusions beyond the NYDFS framework noted by PayPal
Uncertainty notes: The source states the service was available in the U.S. on publication date and separately notes a wider eligibility update on 2020-11-12; this record anchors the first live access date.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_ETH_2021_02_08_CME_FUTURES_LAUNCH
Original collector event IDs: C2_ETH_2021_02_08_CME_ETHER_FUTURES; C2-HC-004
Catalyst cluster ID: ETH_US_REGULATED_DERIVATIVES_2020_2024
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP; ETHUSD_PERP
Mechanism family: leverage_access
Mechanism subtype: regulated_futures_launch
Direction: mixed
Event state: launched
First public timestamp UTC: 2020-12-16
Official confirmation timestamp UTC: 2021-02-08
Effective timestamp UTC: 2021-02-08
Timestamp precision: date_only
Source confidence: high
Primary source type: official CME press release
Primary source URL: https://www.cmegroup.com/media-room/press-releases/2021/2/08/cme_group_announceslaunchofetherfutures.html
Supporting source URLs: https://www.cmegroup.com/media-room/press-releases/2020/12/16/cme_group_to_launchetherfuturesonfebruary82021.html; https://www.cmegroup.com/notices/electronic-trading/2021/01/20210118.html
Source publication timestamp: 2021-02-08
Raw official headline or title: CME Group Announces Launch of Ether Futures
Mechanism classification: CME launched regulated Ether futures, expanding both directional and hedging access through a U.S.-regulated derivatives venue.
Ex-ante durability: high
Estimated access impact: very high regulated derivatives access
Estimated float impact: none direct; synthetic exposure and hedging capacity increased
Ex-ante pre-run risk: medium
Unknown fields: intraday first-trade timestamp not used
Uncertainty notes: The trade date is explicit, but the official launch release does not supply a first-trade UTC timestamp. This record absorbs the exact duplicate submitted by the legal report.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_XRP_2020_12_22_SEC_COMPLAINT
Original collector event IDs: C2_XRP_2020_12_22_SEC_COMPLAINT
Catalyst cluster ID: XRP_SEC_AND_US_ACCESS_2020_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRPUSD; XRP-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_enforcement_complaint_unregistered_securities_offering
Direction: short
Event state: announced
First public timestamp UTC: 2020-12-22
Official confirmation timestamp UTC: 2020-12-22
Effective timestamp UTC: 2020-12-22
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC press release
Primary source URL: https://www.sec.gov/newsroom/press-releases/2020-338
Supporting source URLs: https://www.sec.gov/newsroom/litigation/lit-releases/lr-24960 ; https://www.sec.gov/files/litigation/complaints/2020/comp-pr2020-338.pdf
Source publication timestamp: 2020-12-22
Raw official headline or title: SEC Charges Ripple and Two Executives with Conducting $1.3 Billion Unregistered Securities Offering
Mechanism classification: Formal SEC action alleging XRP-related securities-law violations, raising U.S. exchange, custody, and institutional-compliance risk around XRP.
Ex-ante durability: high
Estimated access impact: severe U.S. spot-venue and compliance overhang
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: intraday publication time
Uncertainty notes: The press release fixes the date clearly; no intraday UTC time was asserted in the source.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_002
Original collector event IDs: C2-HC-002
Catalyst cluster ID: XRP_SEC_AND_US_ACCESS_2020_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRP-PERP; XRPUSDT-SWAP
Mechanism family: exchange_spot_access
Mechanism subtype: spot_trading_limit_only_transition
Direction: short
Event state: limit_only
First public timestamp UTC: 2020-12-28
Official confirmation timestamp UTC: 2020-12-28T22:30:00Z
Effective timestamp UTC: 2020-12-28T22:30:00Z
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange blog post
Primary source URL: https://www.coinbase.com/blog/coinbase-will-suspend-trading-in-xrp-on-january-19
Supporting source URLs: https://help.coinbase.com/en/coinbase/trading-and-funding/cryptocurrency-trading-pairs/ripple-sec
Source publication timestamp: 2020-12-28
Raw official headline or title: Coinbase will suspend trading in XRP on January 19
Mechanism classification: Coinbase moved XRP trading to limit-only before the scheduled full suspension, materially restricting new order-book access.
Ex-ante durability: medium
Estimated access impact: high
Estimated float impact: low at protocol level; meaningful negative venue-accessibility effect
Ex-ante pre-run risk: low
Unknown fields: order-book microstructure during limit-only period
Uncertainty notes: Coinbase stated 2:30 p.m. PST on 2020-12-28; this converts to 22:30 UTC. It is a distinct phase from the full suspension.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_XRP_2021_01_19_COINBASE_FULL_SUSPENSION
Original collector event IDs: C2_XRP_2020_12_28_COINBASE_SUSPENSION; C2-HC-003
Catalyst cluster ID: XRP_SEC_AND_US_ACCESS_2020_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRPUSD; XRP-PERP; XRPUSDT-SWAP
Mechanism family: exchange_spot_access
Mechanism subtype: spot_trading_full_suspension_due_to_regulatory_action
Direction: short
Event state: suspended
First public timestamp UTC: 2020-12-28
Official confirmation timestamp UTC: 2020-12-28
Effective timestamp UTC: 2021-01-19T18:00:00Z
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange notice
Primary source URL: https://www.coinbase.com/blog/coinbase-will-suspend-trading-in-xrp-on-january-19
Supporting source URLs: https://help.coinbase.com/en/coinbase/trading-and-funding/cryptocurrency-trading-pairs/ripple-sec
Source publication timestamp: 2020-12-28
Raw official headline or title: Coinbase will suspend trading in XRP on January 19
Mechanism classification: Coinbase fully suspended XRP spot trading after a previously announced transition, contracting access on a major U.S. venue.
Ex-ante durability: high until legal or venue reversal
Estimated access impact: very high on the affected U.S. venue
Estimated float impact: none direct at protocol level
Ex-ante pre-run risk: low
Unknown fields: exact order-book termination sequence beyond the stated suspension time
Uncertainty notes: The source explicitly states 10:00 a.m. PST on 2021-01-19; this converts to 18:00 UTC. This record absorbs the exact duplicate submitted by the legal report.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_2021_02_18_PURPOSE_TSX_LAUNCH
Original collector event IDs: C2_BTC_2021_02_18_PURPOSE_TSX_LAUNCH
Catalyst cluster ID: BTC_NORTH_AMERICA_SPOT_ETF_ACCESS_2021
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: spot_etf_launch
Direction: long
Event state: launched
First public timestamp UTC: 2021-02-11
Official confirmation timestamp UTC: 2021-02-18
Effective timestamp UTC: 2021-02-18
Timestamp precision: date_only
Source confidence: high
Primary source type: listing venue notice
Primary source URL: https://www.tsx.com/en/resource/2543
Supporting source URLs: https://documents.purposeinvest.com/Docs/BTCC/prospectus/en/Purpose%20Bitcoin%20ETF%20Prospectus%202021-02-11.pdf ; https://investors.tmx.com/English/News--Events/news/news-details/2021/Toronto-Stock-Exchange-Lists-Worlds-First-Bitcoin-ETF-2021-2-18/default.aspx
Source publication timestamp: 2021-02-18
Raw official headline or title: T+1 Settlement for Purpose Bitcoin ETF (Symbols: BTCC.B and BTCC.U)
Mechanism classification: Directly opened spot-BTC ETF wrapper access on a major public exchange, reducing custody frictions for regulated investors.
Ex-ante durability: high
Estimated access impact: major regulated wrapper expansion
Estimated float impact: creation-unit demand possible, no fixed lock-up change
Ex-ante pre-run risk: medium
Unknown fields: OSC receipt timestamp beyond date level
Uncertainty notes: The TSX documents clearly ground listing and trading start; OSC approval detail is publicly referenced but not cleanly necessary for the launch record itself.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_2021_05_03_CME_MICRO_FUTURES
Original collector event IDs: C2_BTC_2021_05_03_CME_MICRO_FUTURES
Catalyst cluster ID: BTC_US_REGULATED_DERIVATIVES_2021_2024
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: leverage_access
Mechanism subtype: regulated_futures_launch_micro_contract
Direction: mixed
Event state: launched
First public timestamp UTC: 2021-03-30
Official confirmation timestamp UTC: 2021-05-03
Effective timestamp UTC: 2021-05-03
Timestamp precision: date_only
Source confidence: high
Primary source type: CME press release
Primary source URL: https://www.cmegroup.com/media-room/press-releases/2021/5/03/cme_group_announceslaunchofmicrobitcoinfutures.html
Supporting source URLs: https://www.cmegroup.com/media-room/press-releases/2021/3/30/cme_group_to_launchmicrobitcoinfuturesonmay3.html
Source publication timestamp: 2021-05-03
Raw official headline or title: CME Group Announces Launch of Micro Bitcoin Futures
Mechanism classification: Expanded regulated BTC derivatives access through smaller contract sizing, broadening addressable institutional and active-trader participation.
Ex-ante durability: high
Estimated access impact: meaningful contract-design expansion
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: launch-time intraday UTC
Uncertainty notes: This is an access-expansion event, not a new underlying-asset legality event.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_005
Original collector event IDs: C2-HC-005
Catalyst cluster ID: PAYPAL_VENMO_CRYPTO_DISTRIBUTION_2020_2025
Asset ID: btc_eth_bch_ltc_basket
Ticker: BTC|ETH|BCH|LTC
Known major perp symbols: BTCUSDT; ETHUSDT; BCHUSDT; LTCUSDT
Mechanism family: distribution_integration
Mechanism subtype: merchant_checkout_crypto_funding
Direction: long
Event state: launched
First public timestamp UTC: 2021-03-30
Official confirmation timestamp UTC: 2021-03-30
Effective timestamp UTC: 2021-03-30
Timestamp precision: date_only
Source confidence: high
Primary source type: official corporate press release
Primary source URL: https://newsroom.paypal-corp.com/2021-03-30-PayPal-Launches-Checkout-with-Crypto
Supporting source URLs: https://newsroom.paypal-corp.com/news-paypal?o=60
Source publication timestamp: 2021-03-30
Raw official headline or title: PayPal Launches "Checkout with Crypto"
Mechanism classification: merchant-side payment utility expansion inside PayPal wallet
Ex-ante durability: high
Estimated access impact: very high
Estimated float impact: low secondary-market circulation effect only
Ex-ante pre-run risk: medium
Unknown fields: merchant-by-merchant availability timing during ongoing expansion
Uncertainty notes: PayPal described the feature as available at millions of global online businesses and continuing to expand; the first live date is preserved, with merchant coverage breadth treated as ongoing.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_006
Original collector event IDs: C2-HC-006
Catalyst cluster ID: PAYPAL_VENMO_CRYPTO_DISTRIBUTION_2020_2025
Asset ID: btc_eth_bch_ltc_basket
Ticker: BTC|ETH|BCH|LTC
Known major perp symbols: BTCUSDT; ETHUSDT; BCHUSDT; LTCUSDT
Mechanism family: distribution_integration
Mechanism subtype: venmo_buy_hold_sell_launch
Direction: long
Event state: rollout_started
First public timestamp UTC: 2021-04-20
Official confirmation timestamp UTC: 2021-04-20
Effective timestamp UTC: 2021-04-20
Timestamp precision: date_only
Source confidence: high
Primary source type: official corporate press release
Primary source URL: https://newsroom.paypal-corp.com/2021-04-20-Introducing-Crypto-on-Venmo
Supporting source URLs: https://newsroom.paypal-corp.com/news-cryptocurrency?l=25
Source publication timestamp: 2021-04-20
Raw official headline or title: Introducing Crypto on Venmo
Mechanism classification: large-app retail crypto access added inside Venmo
Ex-ante durability: high
Estimated access impact: very high
Estimated float impact: low-to-moderate secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: exact final completion date of rollout to all eligible users
Uncertainty notes: The source says rollout began on publication date and would reach all customers within the next few weeks.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_010
Original collector event IDs: C2-HC-010
Catalyst cluster ID: SHIB_BINANCE_ACCESS_2021
Asset ID: shiba_inu
Ticker: SHIB
Known major perp symbols: 1000SHIBUSDT; SHIBUSDT-SWAP; SHIBUSDT
Mechanism family: exchange_spot_access
Mechanism subtype: first_major_binance_spot_listing
Direction: long
Event state: listed
First public timestamp UTC: 2021-05-10T07:44:00Z
Official confirmation timestamp UTC: 2021-05-10T07:44:00Z
Effective timestamp UTC: 2021-05-10T11:00:00Z
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.binance.com/en/support/announcement/detail/f1fe616e688b452f9d736753cb2d947a
Supporting source URLs: none
Source publication timestamp: 2021-05-10 07:44 UTC
Raw official headline or title: Binance Will List SHIBA INU (SHIB) in the Innovation Zone
Mechanism classification: major venue spot listing with deposits opened before trading
Ex-ante durability: medium
Estimated access impact: very high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: none material
Uncertainty notes: The announcement states users could start depositing immediately and spot trading would open at 11:00 UTC the same day.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_011
Original collector event IDs: C2-HC-011
Catalyst cluster ID: SHIB_BINANCE_ACCESS_2021
Asset ID: shiba_inu
Ticker: SHIB
Known major perp symbols: 1000SHIBUSDT; SHIBUSDT-SWAP; SHIBUSDT
Mechanism family: leverage_access
Mechanism subtype: first_major_perpetual_listing_on_binance
Direction: mixed
Event state: launched
First public timestamp UTC: 2021-05-10T10:02:00Z
Official confirmation timestamp UTC: 2021-05-10T10:02:00Z
Effective timestamp UTC: 2021-05-10T13:00:00Z
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.binance.com/en/support/announcement/detail/fa934b93fea6494fb7361d45ca7cc98a
Supporting source URLs: none
Source publication timestamp: 2021-05-10 10:02 UTC
Raw official headline or title: Binance Futures Will Launch USDT-Margined 1000SHIB Perpetual Contracts with Up to 25X Leverage
Mechanism classification: major venue perpetual launch creating material long/short access
Ex-ante durability: high
Estimated access impact: very high
Estimated float impact: none direct; synthetic float and short access increased
Ex-ante pre-run risk: medium
Unknown fields: none material
Uncertainty notes: The perp launched the same day as the Binance spot listing but is preserved separately because leverage access was independently disclosed and economically distinct.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_012
Original collector event IDs: C2-HC-012
Catalyst cluster ID: CL-COINBASE-SOL-ACCESS
Asset ID: solana
Ticker: SOL
Known major perp symbols: SOLUSDT; SOL-PERP; SOLUSD_PERP; SOL futures
Mechanism family: exchange_spot_access
Mechanism subtype: major_coinbase_spot_launch
Direction: long
Event state: available
First public timestamp UTC: 2021-05-20
Official confirmation timestamp UTC: 2021-06-17
Effective timestamp UTC: 2021-06-17
Timestamp precision: date_only
Source confidence: high
Primary source type: official exchange blog post
Primary source URL: https://www.coinbase.com/blog/solana-sol-chiliz-chz-and-keep-network-keep-are-now-available-on-coinbase
Supporting source URLs: https://www.coinbase.com/blog/solana-sol-is-launching-on-coinbase-pro
Source publication timestamp: 2021-06-17
Raw official headline or title: Solana (SOL), Chiliz (CHZ) and Keep Network (KEEP) are now available on Coinbase
Mechanism classification: major U.S. retail exchange consumer availability after delayed Pro launch process
Ex-ante durability: high
Estimated access impact: very high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: exact Pro order-book phase timestamps during delayed launch
Uncertainty notes: Coinbase first published a Pro launch notice on 2021-05-20, later updated it to a 2021-06-16 transfer window and 2021-06-17 trading date. This record anchors the actual consumer availability date.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_009
Original collector event IDs: C2-HC-009
Catalyst cluster ID: CL-COINBASE-DOGE-LAUNCH
Asset ID: dogecoin
Ticker: DOGE
Known major perp symbols: DOGEUSDT; DOGE-PERP; DOGEUSD_PERP
Mechanism family: exchange_spot_access
Mechanism subtype: major_coinbase_spot_launch
Direction: long
Event state: available
First public timestamp UTC: 2021-06-01
Official confirmation timestamp UTC: 2021-06-03
Effective timestamp UTC: 2021-06-03
Timestamp precision: date_only
Source confidence: high
Primary source type: official exchange blog post
Primary source URL: https://www.coinbase.com/blog/dogecoin-doge-is-now-available-on-coinbase
Supporting source URLs: https://www.coinbase.com/blog/dogecoin-doge-is-launching-on-coinbase-pro
Source publication timestamp: 2021-06-03
Raw official headline or title: Dogecoin (DOGE) is now available on Coinbase
Mechanism classification: major U.S. retail exchange consumer availability after Coinbase Pro deposit/ trading phase
Ex-ante durability: high
Estimated access impact: very high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: exact Pro order-book phase-change timestamps
Uncertainty notes: The supporting source shows inbound transfers began on 2021-06-01 and trading would begin on or after 9AM PT on 2021-06-03; this record anchors the consumer-app availability date.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_ETH_2021_LONDON_EIP1559
Original collector event IDs: ETH-2021-LONDON-EIP1559
Catalyst cluster ID: ETH_FEE_AND_STAKING_EVOLUTION
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: fee_burn_activation
Direction: long
Event state: activated
First public timestamp UTC: unknown
Official confirmation timestamp UTC: 2021-07-15
Effective timestamp UTC: 2021-08-05
Timestamp precision: date_only
Source confidence: high
Primary source type: official protocol blog
Primary source URL: https://blog.ethereum.org/2021/07/15/london-mainnet-announcement
Supporting source URLs: https://ethereum.org/ethereum-forks/
Source publication timestamp: 2021-07-15
Raw official headline or title: “London Mainnet Announcement.”
Mechanism classification: EIP-1559 changed Ethereum’s fee market and introduced base-fee burning, creating a direct burn linkage between network usage and ETH supply.
Ex-ante durability: permanent
Estimated access impact: medium
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: first public timestamp for EIP-1559 as a protocol proposal; exact activation time in UTC
Uncertainty notes: The July 15 post is a mainnet activation announcement, not the first public disclosure of EIP-1559. The final activation date is supported by the official fork history.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_2021_10_19_BITO_LAUNCH
Original collector event IDs: C2_BTC_2021_10_19_BITO_LAUNCH
Catalyst cluster ID: BTC_US_FUTURES_ETF_ACCESS_2021
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: futures_based_etf_launch
Direction: long
Event state: launched
First public timestamp UTC: 2021-10-18
Official confirmation timestamp UTC: 2021-10-18
Effective timestamp UTC: 2021-10-19
Timestamp precision: date_only
Source confidence: high
Primary source type: ETF issuer press release
Primary source URL: https://www.proshares.com/press-releases/proshares-to-launch-the-first-u.s.-bitcoin-linked-etf-on-october-19
Supporting source URLs: https://www.sec.gov/Archives/edgar/data/1174610/000168386321006060/f10047d1.htm
Source publication timestamp: 2021-10-18
Raw official headline or title: ProShares to Launch the First U.S. Bitcoin-Linked ETF on October 19
Mechanism classification: Introduced a U.S.-listed ETF wrapper for bitcoin-linked exposure via regulated futures rather than spot holdings.
Ex-ante durability: high
Estimated access impact: major U.S. advisor and brokerage access expansion
Estimated float impact: none direct on spot float by structure
Ex-ante pre-run risk: high
Unknown fields: exact time of first exchange execution
Uncertainty notes: The SEC filing states scheduled trading on October 19; the issuer press release states launch planned for that date.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_ETH_2021_12_06_CME_MICRO_ETHER
Original collector event IDs: C2_ETH_2021_12_06_CME_MICRO_ETHER
Catalyst cluster ID: ETH_US_REGULATED_DERIVATIVES_2020_2024
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: leverage_access
Mechanism subtype: regulated_futures_launch_micro_contract
Direction: mixed
Event state: launched
First public timestamp UTC: 2021-11-02
Official confirmation timestamp UTC: 2021-12-06
Effective timestamp UTC: 2021-12-06
Timestamp precision: date_only
Source confidence: high
Primary source type: CME special executive report
Primary source URL: https://www.cmegroup.com/content/dam/cmegroup/notices/ser/2021/11/SER-8874R.pdf
Supporting source URLs: https://www.cmegroup.com/media-room/press-releases/2021/11/02/cme_group_to_launchmicroetherfuturesondecember6.html ; https://investor.cmegroup.com/news-releases/news-release-details/cme-group-announces-launch-micro-ether-futures
Source publication timestamp: 2021-11-03
Raw official headline or title: Initial Listing of the Micro Ether Futures Contract
Mechanism classification: Extended U.S. regulated ETH derivatives access through smaller-sized contracts that lowered capital and hedging frictions.
Ex-ante durability: high
Estimated access impact: meaningful contract-design expansion
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: launch-time intraday UTC
Uncertainty notes: The effective trade date is clearly stated; the exact first print time is not.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_007
Original collector event IDs: C2-HC-007
Catalyst cluster ID: TON_OKX_AND_TELEGRAM_ACCESS_2021_2023
Asset ID: ton
Ticker: TONCOIN
Known major perp symbols: TONUSDT; TONUSDT-SWAP
Mechanism family: exchange_spot_access
Mechanism subtype: first_major_spot_listing_on_okx
Direction: long
Event state: listed
First public timestamp UTC: 2021-11-12
Official confirmation timestamp UTC: 2021-11-12
Effective timestamp UTC: 2021-11-12 11:20
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.okx.com/help/okx-will-list-the-open-network-toncoin-token-for-spot-trading
Supporting source URLs: none
Source publication timestamp: 2021-11-12
Raw official headline or title: OKX will list The Open Network’s TONCOIN token for spot trading
Mechanism classification: major venue spot listing with deposit, call auction, and spot-trading phases
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: call-auction end microstructure beyond disclosed schedule
Uncertainty notes: Deposits opened at 07:00 UTC, call auction at 11:00 UTC, spot trading at 11:20 UTC, and withdrawals on 2021-11-15 10:00 UTC; this record anchors the actual spot-trading open.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_008
Original collector event IDs: C2-HC-008
Catalyst cluster ID: TON_OKX_AND_TELEGRAM_ACCESS_2021_2023
Asset ID: ton
Ticker: TONCOIN
Known major perp symbols: TONUSDT; TONUSDT-SWAP
Mechanism family: leverage_access
Mechanism subtype: first_major_perpetual_listing_on_okx
Direction: mixed
Event state: launched
First public timestamp UTC: 2021-11-12
Official confirmation timestamp UTC: 2021-11-12
Effective timestamp UTC: 2021-11-12T12:00:00Z
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.okx.com/help/okx-to-list-usdt-margined-perpetual-for-toncoin
Supporting source URLs: none
Source publication timestamp: 2021-11-12
Raw official headline or title: OKX to List USDT-Margined Perpetual for TONCOIN
Mechanism classification: first major venue perpetual access for TONCOIN on OKX
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: none direct; synthetic float and short access increased
Ex-ante pre-run risk: medium
Unknown fields: first trade and depth ramp details
Uncertainty notes: The official OKX page’s extracted typography is irregular around “12:00.” The record uses 12:00 UTC, while treating TONCOIN as the same asset identity later standardized to TON.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_BNB_2021_BEP95
Original collector event IDs: BNB-2021-BEP95
Catalyst cluster ID: BNB_BURN_MECHANISM_EVOLUTION
Asset ID: bnb
Ticker: BNB
Known major perp symbols: BNBUSDT; BNBUSD; BNB-PERP
Mechanism family: supply_float
Mechanism subtype: real_time_fee_burn_activation
Direction: long
Event state: activated
First public timestamp UTC: unknown
Official confirmation timestamp UTC: 2021-11-18
Effective timestamp UTC: 2021-11-30
Timestamp precision: date_only
Source confidence: high
Primary source type: official chain upgrade announcement
Primary source URL: https://www.bnbchain.org/en/blog/binance-smart-chain-bruno-upgrade-v1-1-5
Supporting source URLs: https://www.binance.com/en/blog/ecosystem/421499824684903205
Source publication timestamp: 2021-11-18
Raw official headline or title: “BNB Chain Bruno Upgrade v1.1.5.”
Mechanism classification: the Bruno upgrade introduced BEP-95, a real-time burn mechanism that destroys a portion of BNB from on-chain transaction activity.
Ex-ante durability: permanent
Estimated access impact: low
Estimated float impact: medium
Ex-ante pre-run risk: medium
Unknown fields: earliest public BEP-95 proposal timestamp; exact UTC activation time
Uncertainty notes: The 2021-11-18 official upgrade notice supports the scheduled mechanism. The final 2021-11-30 activation date is corroborated by later official BNB burn documentation; earliest proposal publication was not audited.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_BNB_2021_AUTOBURN
Original collector event IDs: BNB-2021-AUTOBURN
Catalyst cluster ID: BNB_BURN_MECHANISM_EVOLUTION
Asset ID: bnb
Ticker: BNB
Known major perp symbols: BNBUSDT; BNBUSD; BNB-PERP
Mechanism family: supply_float
Mechanism subtype: auto_burn_formula_change
Direction: long
Event state: implemented
First public timestamp UTC: 2021-12-21
Official confirmation timestamp UTC: 2021-12-21
Effective timestamp UTC: 2021-12-21
Timestamp precision: date_only
Source confidence: high
Primary source type: official ecosystem blog
Primary source URL: https://www.binance.com/en/blog/ecosystem/421499824684903205
Supporting source URLs: none
Source publication timestamp: 2021-12-21
Raw official headline or title: “Introducing BNB Auto-Burn: a new Protocol for the Quarterly BNB Burn.”
Mechanism classification: BNB Auto-Burn replaced the prior exchange-volume-linked quarterly burn with an auditable formula tied to BNB Chain activity and price, making supply reduction less discretionary and more directly linked to chain conditions.
Ex-ante durability: permanent
Estimated access impact: low
Estimated float impact: medium
Ex-ante pre-run risk: medium
Unknown fields: none
Uncertainty notes: none
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_013
Original collector event IDs: C2-HC-013
Catalyst cluster ID: CL-SOL-DISTRIBUTION
Asset ID: solana
Ticker: SOL
Known major perp symbols: SOLUSDT; SOL-PERP; SOLUSD_PERP; SOL futures
Mechanism family: distribution_integration
Mechanism subtype: wallet_support_coinbase_wallet_extension
Direction: long
Event state: supported
First public timestamp UTC: 2022-03-17
Official confirmation timestamp UTC: 2022-03-17
Effective timestamp UTC: 2022-03-17
Timestamp precision: date_only
Source confidence: high
Primary source type: official product blog post
Primary source URL: https://www.coinbase.com/blog/coinbase-wallet-introduces-support-for-the-solana-ecosystem
Supporting source URLs: none
Source publication timestamp: 2022-03-17
Raw official headline or title: Coinbase Wallet introduces support for the Solana ecosystem
Mechanism classification: large-wallet integration enabling SOL and SPL custody and transfers
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low
Ex-ante pre-run risk: low
Unknown fields: mobile parity timing beyond browser-extension launch
Uncertainty notes: The initial phase covered the Coinbase Wallet browser extension for sending, receiving, and storing SOL and SPL tokens; mobile support was not part of the same effective release.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_014
Original collector event IDs: C2-HC-014
Catalyst cluster ID: ROBINHOOD_CRYPTO_DISTRIBUTION_2022_2024
Asset ID: comp_matic_sol_shib_basket
Ticker: COMP|MATIC|SOL|SHIB
Known major perp symbols: COMPUSDT; MATICUSDT; SOLUSDT; 1000SHIBUSDT
Mechanism family: distribution_integration
Mechanism subtype: brokerage_asset_addition_robinhood_us
Direction: long
Event state: listed
First public timestamp UTC: 2022-04-12
Official confirmation timestamp UTC: 2022-04-12
Effective timestamp UTC: 2022-04-12
Timestamp precision: date_only
Source confidence: high
Primary source type: official brokerage newsroom post
Primary source URL: https://robinhood.com/us/en/newsroom/robinhood-lists-four-new-crypto-assets/
Supporting source URLs: none
Source publication timestamp: 2022-04-12
Raw official headline or title: Robinhood Lists Four New Crypto Assets
Mechanism classification: major U.S. broker-app distribution expansion for four assets
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: deposit/withdrawal enablement timing for each listed asset
Uncertainty notes: Robinhood stated buy/sell access was live immediately, while deposits and withdrawals would arrive later.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_015
Original collector event IDs: C2-HC-015
Catalyst cluster ID: PAYPAL_VENMO_CRYPTO_DISTRIBUTION_2020_2025
Asset ID: btc_eth_bch_ltc_basket
Ticker: BTC|ETH|BCH|LTC
Known major perp symbols: BTCUSDT; ETHUSDT; BCHUSDT; LTCUSDT
Mechanism family: distribution_integration
Mechanism subtype: wallet_interoperability_send_receive_external_transfer
Direction: mixed
Event state: rollout_started
First public timestamp UTC: 2022-06-07
Official confirmation timestamp UTC: 2022-06-07
Effective timestamp UTC: 2022-06-07
Timestamp precision: date_only
Source confidence: high
Primary source type: official corporate press release
Primary source URL: https://newsroom.paypal-corp.com/2022-06-07-PayPal-Users-Can-Now-Transfer-Send-and-Receive-Bitcoin-Ethereum-Bitcoin-Cash-and-Litecoin
Supporting source URLs: none
Source publication timestamp: 2022-06-07
Raw official headline or title: PayPal Users Can Now Transfer, Send, and Receive Bitcoin, Ethereum, Bitcoin Cash, and Litecoin
Mechanism classification: wallet interoperability and external-address movement
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low
Ex-ante pre-run risk: low
Unknown fields: exact date of universal rollout to all eligible U.S. users
Uncertainty notes: The source says the feature was available to select U.S. users on publication date and would roll out to all eligible customers in coming weeks; the later 2022-08-12 update confirmed broader availability.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_2022_06_29_GRAYSCALE_DENIAL
Original collector event IDs: C2_BTC_2022_06_29_GRAYSCALE_DENIAL
Catalyst cluster ID: BTC_US_SPOT_ETF_ADJUDICATION_2022_2024
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: spot_etf_disapproval
Direction: short
Event state: denied
First public timestamp UTC: 2022-06-29
Official confirmation timestamp UTC: 2022-06-29
Effective timestamp UTC: 2022-06-29
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC SRO disapproval order
Primary source URL: https://www.sec.gov/files/rules/sro/nysearca/2022/34-95180.pdf
Supporting source URLs: https://www.govinfo.gov/content/pkg/USCOURTS-caDC-22-01142/pdf/USCOURTS-caDC-22-01142-0.pdf
Source publication timestamp: 2022-06-29
Raw official headline or title: Order Disapproving a Proposed Rule Change to List and Trade Shares of Grayscale Bitcoin Trust (BTC) under NYSE Arca Rule 8.201-E
Mechanism classification: Formal SEC denial of U.S. spot-BTC ETF conversion, maintaining the barrier between BTC spot ownership and the U.S. ETF wrapper.
Ex-ante durability: medium
Estimated access impact: continued restriction on U.S. spot ETF access
Estimated float impact: none direct
Ex-ante pre-run risk: high
Unknown fields: intraday order release time
Uncertainty notes: The later court vacatur is treated as a separate phase in the same catalyst cluster.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_ETH_2022_MERGE
Original collector event IDs: ETH-2022-MERGE
Catalyst cluster ID: ETH_FEE_AND_STAKING_EVOLUTION
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: issuance_and_consensus_change
Direction: long
Event state: activated
First public timestamp UTC: unknown
Official confirmation timestamp UTC: 2022-09-15
Effective timestamp UTC: 2022-09-15
Timestamp precision: date_only
Source confidence: high
Primary source type: official protocol roadmap / activation record
Primary source URL: https://ethereum.org/ethereum-forks/
Supporting source URLs: https://ethereum.org/roadmap/
Source publication timestamp: roadmap page current, describing 2022 production activation
Raw official headline or title: “Paris (The Merge).”
Mechanism classification: Ethereum moved from mining to proof-of-stake, replacing miner issuance with staking-based consensus and materially reducing issuance while making ETH staking central to network security.
Ex-ante durability: permanent
Estimated access impact: high
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: first public timestamp
Uncertainty notes: The official fork timeline supports the 2022-09-15 activation. The collector did not retrieve a defensible first-public mainnet-scheduling timestamp.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_LINK_2022_STAKING_V01
Original collector event IDs: LINK-2022-STAKING-V01
Catalyst cluster ID: LINK_STAKING_EVOLUTION
Asset ID: chainlink
Ticker: LINK
Known major perp symbols: LINKUSDT; LINKUSD; LINK-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: staking_launch
Direction: long
Event state: launched
First public timestamp UTC: 2022-11-21
Official confirmation timestamp UTC: 2022-11-21
Effective timestamp UTC: 2022-12-06T17:00:00Z
Timestamp precision: minute
Source confidence: high
Primary source type: official protocol blog
Primary source URL: https://chain.link/blog/chainlink-staking-launch-details
Supporting source URLs: https://chain.link/economics/staking
Source publication timestamp: 2022-11-21
Raw official headline or title: “The Chainlink Economics 2.0 Staking Protocol and Staking v0.1 Launch Details.”
Mechanism classification: Chainlink launched staking v0.1 on Ethereum mainnet, allowing LINK holders and node operators to stake LINK to back oracle-service performance and earn rewards, with an initial 25M LINK cap.
Ex-ante durability: long-lived
Estimated access impact: high
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: none
Uncertainty notes: The official launch-details post states 12:00 p.m. ET on 2022-12-06; December is EST, converting to 17:00 UTC. This is the early-access mainnet phase.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_ARB_2024_FIRST_TEAM_INVESTOR_UNLOCK
Original collector event IDs: ARB-2024-FIRST-TEAM-INVESTOR-UNLOCK
Catalyst cluster ID: ARB_TEAM_AND_INVESTOR_VESTING
Asset ID: arbitrum
Ticker: ARB
Known major perp symbols: ARBUSDT; ARBUSD; ARB-PERP
Mechanism family: supply_float
Mechanism subtype: major_cliff_unlock
Direction: short
Event state: scheduled_and_effective
First public timestamp UTC: 2023-03-16
Official confirmation timestamp UTC: 2023-03-16
Effective timestamp UTC: 2024-03-16
Timestamp precision: date_only
Source confidence: high
Primary source type: official token-distribution docs
Primary source URL: https://docs.arbitrum.foundation/airdrop-eligibility-distribution
Supporting source URLs: https://docs.arbitrum.foundation/assets/files/ArbitrumFoundationTransparencyReport2023-75b1b491667ad2fa1d9d574a2108f28b.pdf
Source publication timestamp: 2023-03-16
Raw official headline or title: “$ARB airdrop eligibility and distribution specifications.”
Mechanism classification: official docs stated that all investor and team tokens were subject to four-year lockups with the first unlocks one year after the token generation event on 2023-03-16 and monthly unlocks thereafter, making 2024-03-16 the first major cliff event.
Ex-ante durability: one-off cliff followed by stream
Estimated access impact: low
Estimated float impact: high
Ex-ante pre-run risk: high
Unknown fields: exact intraday unlock time
Uncertainty notes: the official docs clearly define the cliff date but do not provide an intraday timestamp in the retrieved snippet.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_RENDER_2023_SOLANA_MIGRATION
Original collector event IDs: RENDER-2023-SOLANA-MIGRATION
Catalyst cluster ID: RENDER_MIGRATION_AND_BME
Asset ID: render
Ticker: RNDR/RENDER
Known major perp symbols: RNDRUSDT; RENDERUSDT; RNDRUSD; RENDERUSD; RNDR-PERP; RENDER-PERP
Mechanism family: supply_float
Mechanism subtype: token_migration_and_chain_transition
Direction: mixed
Event state: completed
First public timestamp UTC: 2023-03-20
Official confirmation timestamp UTC: 2023-11-02
Effective timestamp UTC: 2023-11-02
Timestamp precision: date_only
Source confidence: high
Primary source type: official governance / official project announcement
Primary source URL: https://medium.com/render-token/render-network-completes-successful-upgrade-to-solana-3d7947b60aed
Supporting source URLs: https://know.rendernetwork.com/general-render-network/rndr-to-render-what-you-need-to-know
Source publication timestamp: 2023-11-02 for completion notice
Raw official headline or title: “Render Network Completes Successful Upgrade To Solana.”
Mechanism classification: Render migrated from Ethereum-based RNDR toward Solana-based RENDER, with 1:1 upgrade tooling and a new operational token framework tied to Solana deployment and later Burn-Mint Equilibrium mechanics.
Ex-ante durability: permanent
Estimated access impact: high
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: none
Uncertainty notes: RNDR is the Ethereum-era ticker and RENDER the Solana-era ticker. Venue migration dates differ; the protocol completion date does not imply simultaneous perp-symbol migration.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_ETH_2023_SHAPELLA
Original collector event IDs: ETH-2023-SHAPELLA
Catalyst cluster ID: ETH_FEE_AND_STAKING_EVOLUTION
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: staking_withdrawals_enabled
Direction: mixed
Event state: activated
First public timestamp UTC: 2023-03-28
Official confirmation timestamp UTC: 2023-03-28
Effective timestamp UTC: 2023-04-12T22:27:35Z
Timestamp precision: second
Source confidence: high
Primary source type: official protocol blog
Primary source URL: https://blog.ethereum.org/2023/03/28/shapella-mainnet-announcement
Supporting source URLs: https://ethereum.org/ethereum-forks/; https://ethereum.org/roadmap/
Source publication timestamp: 2023-03-28
Raw official headline or title: “Mainnet Shapella Announcement.”
Mechanism classification: Shapella enabled partial and full staking withdrawals from the Beacon Chain to the execution layer, increasing staking accessibility and capital flexibility while also making previously locked ETH withdrawable.
Ex-ante durability: permanent
Estimated access impact: high
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: none
Uncertainty notes: direction is mixed because the event improved staking utility but also opened a new withdrawal channel for previously locked stake.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_ARB_2023_AIP11_FOUNDATION_LOCKUP
Original collector event IDs: ARB-2023-AIP11-FOUNDATION-LOCKUP
Catalyst cluster ID: ARB_FLOAT_AND_GOVERNANCE_BASELINE
Asset ID: arbitrum
Ticker: ARB
Known major perp symbols: ARBUSDT; ARBUSD; ARB-PERP
Mechanism family: supply_float
Mechanism subtype: treasury_lockup_schedule_change
Direction: long
Event state: executed
First public timestamp UTC: 2023-04-05
Official confirmation timestamp UTC: 2023-04-17T18:00:00Z
Effective timestamp UTC: 2023-06-07
Timestamp precision: minute
Source confidence: high
Primary source type: official governance forum and onchain governance page
Primary source URL: https://www.tally.xyz/gov/arbitrum/proposal/70545629960586317780628692755032548222173912190231545322320044688071893662480?govId=eip155%3A42161%3A0x789fC99093B09aD01C34DC7251D0C89ce743e5a4
Supporting source URLs: https://forum.arbitrum.foundation/t/proposal-aip-1-1-lockup-budget-transparency/13360
Source publication timestamp: 2023-06-07 for executed updated proposal page
Raw official headline or title: “[UPDATED] AIP-1.1 - Lockup, Budget, Transparency.”
Mechanism classification: AIP-1.1 put the Foundation’s remaining 7% administrative-budget allocation into a smart-contract lockup that linearly unlocked over four years from Snapshot approval, replacing a looser immediate-access path with a structured float schedule.
Ex-ante durability: long-lived
Estimated access impact: low
Estimated float impact: high
Ex-ante pre-run risk: medium
Unknown fields: none
Uncertainty notes: The Snapshot approval time and later on-chain execution are separate phases. The lockup schedule was anchored to Snapshot approval; the on-chain execution occurred on 2023-06-07.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_018
Original collector event IDs: C2-HC-018
Catalyst cluster ID: PEPE_MAJOR_VENUE_ACCESS_2023
Asset ID: pepe
Ticker: PEPE
Known major perp symbols: PEPEUSDT; PEPEUSDT-SWAP; 1000PEPEUSDT
Mechanism family: exchange_spot_access
Mechanism subtype: first_major_okx_spot_listing
Direction: long
Event state: listed
First public timestamp UTC: 2023-05-01
Official confirmation timestamp UTC: 2023-05-01
Effective timestamp UTC: 2023-05-01 09:00
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.okx.com/help/okx-will-list-pepe-pepe-for-spot-trading
Supporting source URLs: none
Source publication timestamp: 2023-05-01
Raw official headline or title: OKX will list Pepe (PEPE) for spot trading
Mechanism classification: first major venue spot listing with same-day trading open
Ex-ante durability: medium
Estimated access impact: high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: high
Unknown fields: deposit-open timing not separately disclosed beyond the listing notice
Uncertainty notes: The source disclosed spot trading at 09:00 UTC on 2023-05-01 and withdrawals at 09:00 UTC on 2023-05-02.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_019
Original collector event IDs: C2-HC-019
Catalyst cluster ID: PEPE_MAJOR_VENUE_ACCESS_2023
Asset ID: pepe
Ticker: PEPE
Known major perp symbols: PEPEUSDT; PEPEUSDT-SWAP; 1000PEPEUSDT
Mechanism family: leverage_access
Mechanism subtype: first_major_okx_perpetual_and_margin_launch
Direction: mixed
Event state: launched
First public timestamp UTC: 2023-05-01
Official confirmation timestamp UTC: 2023-05-01
Effective timestamp UTC: 2023-05-03 04:00
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.okx.com/help/okx-to-enable-margin-trading-and-savings-and-list-perpetual-for-aidoge-pepe
Supporting source URLs: https://www.okx.com/help/okx-will-list-pepe-pepe-for-spot-trading
Source publication timestamp: 2023-05-01
Raw official headline or title: OKX to Enable Margin Trading & Savings and List Perpetual for AIDOGE, PEPE
Mechanism classification: major venue leverage launch for PEPE through perp and margin
Ex-ante durability: medium
Estimated access impact: high
Estimated float impact: none direct; synthetic float and short access increased
Ex-ante pre-run risk: high
Unknown fields: none material
Uncertainty notes: OKX disclosed both the spot listing and the later leverage access within separate notices; this record preserves the May 3 leverage phase.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_016
Original collector event IDs: C2-HC-016
Catalyst cluster ID: SUI_OKX_ACCESS_2023
Asset ID: sui
Ticker: SUI
Known major perp symbols: SUIUSDT; SUIUSDT-SWAP
Mechanism family: exchange_spot_access
Mechanism subtype: major_okx_spot_listing_with_call_auction
Direction: long
Event state: listed
First public timestamp UTC: 2023-05-02
Official confirmation timestamp UTC: 2023-05-03
Effective timestamp UTC: 2023-05-03 12:00
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.okx.com/help/update-on-the-sui-token-spot-trading-schedule
Supporting source URLs: https://www.okx.com/help/okx-will-list-sui-sui-for-spot-trading
Source publication timestamp: 2023-05-03
Raw official headline or title: Update on the SUI token spot trading schedule
Mechanism classification: major venue spot launch with updated call-auction and trading schedule
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: none material
Uncertainty notes: OKX published an original schedule on 2023-05-02 and an updated schedule on 2023-05-03; this record uses the updated effective times because they governed the live event.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_017
Original collector event IDs: C2-HC-017
Catalyst cluster ID: SUI_OKX_ACCESS_2023
Asset ID: sui
Ticker: SUI
Known major perp symbols: SUIUSDT; SUIUSDT-SWAP
Mechanism family: leverage_access
Mechanism subtype: major_okx_perpetual_and_margin_launch
Direction: mixed
Event state: launched
First public timestamp UTC: 2023-05-04
Official confirmation timestamp UTC: 2023-05-04
Effective timestamp UTC: 2023-05-05 04:00
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.okx.com/help/okx-to-enable-margin-trading-and-savings-and-list-perpetual-for-sui
Supporting source URLs: none
Source publication timestamp: 2023-05-04
Raw official headline or title: OKX to Enable Margin Trading & Savings and List Perpetual for SUI
Mechanism classification: major venue leverage access via perp and spot margin
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: none direct; synthetic float and marginable access expanded
Ex-ante pre-run risk: medium
Unknown fields: none material
Uncertainty notes: This is kept separate from the spot launch because leverage access was independently disclosed and went live the following day.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EX_004
Original collector event IDs: C2-EX-004
Catalyst cluster ID: PEPE_MAJOR_VENUE_ACCESS_2023
Asset ID: pepe
Ticker: PEPE
Known major perp symbols: PEPEUSDT; PEPEUSDT-SWAP; 1000PEPEUSDT
Mechanism family: exchange_spot_access
Mechanism subtype: additional_major_venue_spot_listing
Direction: long
Event state: listed
First public timestamp UTC: 2023-05-05 11:20
Official confirmation timestamp UTC: 2023-05-05 11:20
Effective timestamp UTC: 2023-05-05 16:00
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.binance.com/en/support/announcement/detail/f68a3bc6eb014ed9bacf1d6c71dc1134
Supporting source URLs: https://www.okx.com/help/okx-will-list-pepe-pepe-for-spot-trading
Source publication timestamp: 2023-05-05 11:20 UTC
Raw official headline or title: Binance Will List FLOKI (FLOKI) and Pepe (PEPE) in the Innovation Zone
Mechanism classification: Binance added PEPE spot trading after OKX. This is not a duplicate: a second major venue independently widened distribution and liquidity access.
Ex-ante durability: medium
Estimated access impact: high
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: none material
Uncertainty notes: The event is later than the OKX listing but remains a distinct venue-access expansion. “First venue only” is not a valid deduplication rule for access breadth.
Final inclusion status: high
Exclusion or downgrade reason: none; collector exclusion overturned because a new major venue is a distinct access expansion

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EX_005
Original collector event IDs: C2-EX-005
Catalyst cluster ID: PEPE_MAJOR_VENUE_ACCESS_2023
Asset ID: pepe
Ticker: PEPE
Known major perp symbols: PEPEUSDT; PEPEUSDT-SWAP; 1000PEPEUSDT
Mechanism family: leverage_access
Mechanism subtype: additional_major_venue_perpetual_listing
Direction: mixed
Event state: launched
First public timestamp UTC: 2023-05-05
Official confirmation timestamp UTC: 2023-05-05
Effective timestamp UTC: 2023-05-05 16:30
Timestamp precision: minute
Source confidence: high
Primary source type: official exchange announcement
Primary source URL: https://www.binance.com/en/support/announcement/detail/41993e3389654713946bcb6b9b032eaf
Supporting source URLs: https://www.okx.com/help/okx-to-enable-margin-trading-and-savings-and-list-perpetual-for-aidoge-pepe
Source publication timestamp: 2023-05-05
Raw official headline or title: Binance Futures will launch the USDⓈ-M 1000PEPE perpetual contract at 2023-05-05 16:30 (UTC), with up to 20X leverage.
Mechanism classification: Binance launched a PEPE perpetual after OKX, adding distinct leverage, shorting, and venue-distribution capacity.
Ex-ante durability: medium
Estimated access impact: high
Estimated float impact: none direct
Ex-ante pre-run risk: high
Unknown fields: none material
Uncertainty notes: The contract is later than the OKX perpetual but is not semantically duplicative because venue access and participant reach changed independently.
Final inclusion status: high
Exclusion or downgrade reason: none; collector exclusion overturned because a new major venue is a distinct leverage expansion

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_ADA_2023_06_05_SEC_BINANCE
Original collector event IDs: C2_ADA_2023_06_05_SEC_BINANCE
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: cardano
Ticker: ADA
Known major perp symbols: ADAUSDT; ADAUSD; ADA-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_complaint_asset_named_crypto_asset_security
Direction: short
Event state: announced
First public timestamp UTC: 2023-06-05
Official confirmation timestamp UTC: 2023-06-05
Effective timestamp UTC: 2023-06-05
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-101.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-101-sec-files-13-charges-against-binance-entities-founder-changpeng-zhao
Source publication timestamp: 2023-06-05
Raw official headline or title: SEC Files 13 Charges Against Binance Entities and Founder Changpeng Zhao
Mechanism classification: ADA was named in the complaint’s list of “crypto asset securities,” increasing U.S.- facing listing and institutional-risk friction.
Ex-ante durability: high
Estimated access impact: severe U.S. compliance and listing overhang
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: intraday publication time
Uncertainty notes: Same legal cluster as SOL and MATIC; kept separate for asset-level ingestion.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_ATOM_2023_06_05_SEC_BINANCE
Original collector event IDs: C2_ATOM_2023_06_05_SEC_BINANCE
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: cosmos
Ticker: ATOM
Known major perp symbols: ATOMUSDT; ATOMUSD; ATOM-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_complaint_asset_named_crypto_asset_security
Direction: short
Event state: announced
First public timestamp UTC: 2023-06-05
Official confirmation timestamp UTC: 2023-06-05
Effective timestamp UTC: 2023-06-05
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-101.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-101-sec-files-13-charges-against-binance-entities-founder-changpeng-zhao
Source publication timestamp: 2023-06-05
Raw official headline or title: SEC Files 13 Charges Against Binance Entities and Founder Changpeng Zhao
Mechanism classification: ATOM was named as a “crypto asset security,” adding a formal U.S. legal-status overhang to a major perp asset.
Ex-ante durability: high
Estimated access impact: severe U.S. compliance and listing overhang
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: intraday publication time
Uncertainty notes: Asset-specific inclusion is explicit in the complaint text.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_FIL_2023_06_05_SEC_BINANCE
Original collector event IDs: C2_FIL_2023_06_05_SEC_BINANCE
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: filecoin
Ticker: FIL
Known major perp symbols: FILUSDT; FILUSD; FIL-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_complaint_asset_named_crypto_asset_security
Direction: short
Event state: announced
First public timestamp UTC: 2023-06-05
Official confirmation timestamp UTC: 2023-06-05
Effective timestamp UTC: 2023-06-05
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-101.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-101-sec-files-13-charges-against-binance-entities-founder-changpeng-zhao
Source publication timestamp: 2023-06-05
Raw official headline or title: SEC Files 13 Charges Against Binance Entities and Founder Changpeng Zhao
Mechanism classification: FIL was specifically included in the SEC’s asset-security list, materially changing U.S. listing-risk expectations ex ante.
Ex-ante durability: high
Estimated access impact: severe U.S. compliance and listing overhang
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: intraday publication time
Uncertainty notes: This captures the first public SEC/Binance phase only.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_MATIC_2023_06_05_SEC_BINANCE
Original collector event IDs: C2_MATIC_2023_06_05_SEC_BINANCE
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: polygon
Ticker: MATIC
Known major perp symbols: MATICUSDT; MATICUSD; MATIC-PERP; POLUSDT; POL-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_complaint_asset_named_crypto_asset_security
Direction: short
Event state: announced
First public timestamp UTC: 2023-06-05
Official confirmation timestamp UTC: 2023-06-05
Effective timestamp UTC: 2023-06-05
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-101.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-101-sec-files-13-charges-against-binance-entities-founder-changpeng-zhao
Source publication timestamp: 2023-06-05
Raw official headline or title: SEC Files 13 Charges Against Binance Entities and Founder Changpeng Zhao
Mechanism classification: MATIC was expressly listed among the complaint’s “crypto asset securities,” creating asset-level legal-status and U.S. routing risk.
Ex-ante durability: high
Estimated access impact: severe U.S. compliance and listing overhang
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: venue-specific MATIC-to-POL perp migration dates
Uncertainty notes: The event concerns Polygon’s MATIC identity in 2023. POL is the later ticker migration and must not be retroactively substituted into the event timestamp.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_SOL_2023_06_05_SEC_BINANCE
Original collector event IDs: C2_SOL_2023_06_05_SEC_BINANCE
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: solana
Ticker: SOL
Known major perp symbols: SOLUSDT; SOLUSD; SOL-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_complaint_asset_named_crypto_asset_security
Direction: short
Event state: announced
First public timestamp UTC: 2023-06-05
Official confirmation timestamp UTC: 2023-06-05
Effective timestamp UTC: 2023-06-05
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-101.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-101-sec-files-13-charges-against-binance-entities-founder-changpeng-zhao
Source publication timestamp: 2023-06-05
Raw official headline or title: SEC Files 13 Charges Against Binance Entities and Founder Changpeng Zhao
Mechanism classification: SOL was named in an SEC complaint as one of the “crypto asset securities,” raising exchange-listing and institutional-participation risk in the U.S.
Ex-ante durability: high
Estimated access impact: severe U.S. compliance and listing overhang
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: intraday publication time
Uncertainty notes: This record captures the first Binance complaint phase, not later dismissal or overlapping Coinbase/Kraken phases.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_ICP_2023_06_06_SEC_COINBASE
Original collector event IDs: C2_ICP_2023_06_06_SEC_COINBASE
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: internet_computer
Ticker: ICP
Known major perp symbols: ICPUSDT; ICPUSD; ICP-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_complaint_asset_named_crypto_asset_security
Direction: short
Event state: announced
First public timestamp UTC: 2023-06-06
Official confirmation timestamp UTC: 2023-06-06
Effective timestamp UTC: 2023-06-06
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-102.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-102
Source publication timestamp: 2023-06-06
Raw official headline or title: SEC Charges Coinbase for Operating as an Unregistered Securities Exchange, Broker, and Clearing Agency
Mechanism classification: ICP appeared in the Coinbase complaint’s list of “crypto asset securities,” creating a clean asset-level U.S. listing-risk shock.
Ex-ante durability: high
Estimated access impact: severe U.S. compliance and listing overhang
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: intraday publication time
Uncertainty notes: This is included because ICP was not in the Binance list and therefore is not just a duplicate of the June 5 cluster.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_NEAR_2023_06_06_SEC_COINBASE
Original collector event IDs: C2_NEAR_2023_06_06_SEC_COINBASE
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: near_protocol
Ticker: NEAR
Known major perp symbols: NEARUSDT; NEARUSD; NEAR-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_complaint_asset_named_crypto_asset_security
Direction: short
Event state: announced
First public timestamp UTC: 2023-06-06
Official confirmation timestamp UTC: 2023-06-06
Effective timestamp UTC: 2023-06-06
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-102.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-102
Source publication timestamp: 2023-06-06
Raw official headline or title: SEC Charges Coinbase for Operating as an Unregistered Securities Exchange, Broker, and Clearing Agency
Mechanism classification: NEAR appeared in the complaint’s “crypto asset securities” list, producing an explicit U.S. legal-status and listing-risk repricing channel.
Ex-ante durability: high
Estimated access impact: severe U.S. compliance and listing overhang
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: intraday publication time
Uncertainty notes: This record, like ICP, is kept because the June 6 Coinbase complaint added assets not already covered in the June 5 Binance complaint.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_MKR_2023_SMART_BURN_ENGINE
Original collector event IDs: MKR-2023-SMART-BURN-ENGINE
Catalyst cluster ID: MKR_FEE_CAPTURE_AND_BURN_EVOLUTION
Asset ID: maker
Ticker: MKR
Known major perp symbols: MKRUSDT; MKRUSD; MKR-PERP; SKYUSDT
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: protocol_surplus_to_token_buyback_channel
Direction: long
Event state: deployed
First public timestamp UTC: 2023-06-26
Official confirmation timestamp UTC: 2023-07-14
Effective timestamp UTC: 2023-07-14
Timestamp precision: date_only
Source confidence: high
Primary source type: official governance poll and executive vote
Primary source URL: https://vote.makerdao.com/executive/template-executive-vote-blocktower-andromeda-upgrade-smart-burn-engine-deployment-keeper-job-updates-scope-defined-parameter-changes-delegate-compensation-ecosystem-actor-and-core-unit-funding-updates-spark-protocol-proxy-spell-execution-july-14-2023
Supporting source URLs: https://vote.makerdao.com/polling/QmQmxEZp
Source publication timestamp: 2023-07-14 for deployment executive; 2023-06-26 for launch-parameter poll
Raw official headline or title: “BlockTower Andromeda Upgrade, Smart Burn Engine Deployment...” / “Smart Burn Engine Launch Parameters - June 26, 2023.”
Mechanism classification: Maker replaced or supplemented older surplus handling with a Smart Burn Engine that used surplus buffer funds to purchase MKR-related exposure through the DAI/MKR market structure, directly linking protocol surplus management to MKR token economics.
Ex-ante durability: long-lived
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: later MKR-to-SKY migration mapping is outside this event phase
Uncertainty notes: This 2023 record concerns MKR. Later Sky rebranding/migration should be handled as a separate identity phase, not rewritten into this event.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_XRP_2023_07_13_RIPPLE_SUMMARY_JUDGMENT
Original collector event IDs: C2_XRP_2023_07_13_RIPPLE_SUMMARY_JUDGMENT
Catalyst cluster ID: XRP_SEC_AND_US_ACCESS_2020_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRPUSD; XRP-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: federal_court_summary_judgment_partial_legal_clarification
Direction: long
Event state: confirmed
First public timestamp UTC: 2023-07-13
Official confirmation timestamp UTC: 2023-07-13
Effective timestamp UTC: 2023-07-13
Timestamp precision: date_only
Source confidence: high
Primary source type: federal court opinion
Primary source URL: https://www.nysd.uscourts.gov/sites/default/files/2023-07/SEC%20vs%20Ripple%207-13-23.pdf
Supporting source URLs: https://www.sec.gov/newsroom/speeches-statements/crenshaw-statement-ripple-050825
Source publication timestamp: 2023-07-13
Raw official headline or title: SEC vs Ripple 7/13/23
Mechanism classification: The court split XRP transaction categories, finding institutional sales unlawful while declining to treat programmatic exchange sales the same way; that formally reduced part of XRP’s exchange-trading legal overhang.
Ex-ante durability: high
Estimated access impact: material narrowing of exchange-trading legality overhang
Estimated float impact: none direct
Ex-ante pre-run risk: high
Unknown fields: opinion release time
Uncertainty notes: The ruling did not eliminate all XRP-related litigation risk; it narrowed and differentiated it.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_LINK_2023_STAKING_V02
Original collector event IDs: LINK-2023-STAKING-V02
Catalyst cluster ID: LINK_STAKING_EVOLUTION
Asset ID: chainlink
Ticker: LINK
Known major perp symbols: LINKUSDT; LINKUSD; LINK-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: staking_upgrade_and_capacity_expansion
Direction: long
Event state: launched
First public timestamp UTC: 2023-08-25
Official confirmation timestamp UTC: 2023-08-25
Effective timestamp UTC: 2023-11-28T17:00:00Z
Timestamp precision: minute
Source confidence: high
Primary source type: official protocol blog
Primary source URL: https://chain.link/blog/chainlink-staking-v0-2-overview
Supporting source URLs: https://chain.link/economics/staking
Source publication timestamp: 2023-08-25
Raw official headline or title: “Chainlink Staking v0.2 Overview.”
Mechanism classification: v0.2 expanded the staking platform to 45M LINK, introduced unbonding, slashing for node operators, and a dynamic rewards model that could support future external reward sources, strengthening LINK’s security utility and lock-up channel.
Ex-ante durability: long-lived
Estimated access impact: high
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: none
Uncertainty notes: The official v0.2 schedule states 12:00 p.m. ET on 2023-11-28 for priority migration; December-standard offset does not apply because the date is in November after DST ended, so 17:00 UTC.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_2023_08_29_GRAYSCALE_VACATUR
Original collector event IDs: C2_BTC_2023_08_29_GRAYSCALE_VACATUR
Catalyst cluster ID: BTC_US_SPOT_ETF_ADJUDICATION_2022_2024
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: appellate_vacatur_of_spot_etf_denial
Direction: long
Event state: dismissed
First public timestamp UTC: 2023-08-29
Official confirmation timestamp UTC: 2023-08-29
Effective timestamp UTC: 2023-08-29
Timestamp precision: date_only
Source confidence: high
Primary source type: federal appellate opinion
Primary source URL: https://www.govinfo.gov/content/pkg/USCOURTS-caDC-22-01142/pdf/USCOURTS-caDC-22-01142-0.pdf
Supporting source URLs: https://www.sec.gov/files/rules/sro/nysearca/2022/34-95180.pdf
Source publication timestamp: 2023-08-29
Raw official headline or title: Grayscale Investments, LLC v. Securities and Exchange Commission
Mechanism classification: The D.C. Circuit vacated the SEC’s denial of Grayscale’s conversion request, materially improving the ex-ante path for U.S. spot-BTC ETF approval.
Ex-ante durability: high
Estimated access impact: major restoration of U.S. spot ETF approval pathway
Estimated float impact: none direct
Ex-ante pre-run risk: high
Unknown fields: opinion release time
Uncertainty notes: The court did not itself approve a spot ETF; it removed a major regulatory obstruction.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_ETH_2024_01_11_CBOE_DIGITAL_MARGIN_BTC
Original collector event IDs: C2_BTC_ETH_2024_01_11_CBOE_DIGITAL_MARGIN_BTC
Catalyst cluster ID: BTC_US_REGULATED_DERIVATIVES_2021_2024
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: leverage_access
Mechanism subtype: regulated_margin_futures_launch
Direction: mixed
Event state: launched
First public timestamp UTC: 2023-11-13
Official confirmation timestamp UTC: 2024-01-12
Effective timestamp UTC: 2024-01-11
Timestamp precision: date_only
Source confidence: high
Primary source type: regulated-market press release
Primary source URL: https://ir.cboe.com/news/news-details/2024/Cboe-Digital-Launches-Margined-Bitcoin-and-Ether-Futures-Announces-Successful-First-Trade/default.aspx
Supporting source URLs: https://ir.cboe.com/news/news-details/2023/Cboe-Digital-to-Launch-Margined-Bitcoin-and-Ether-Futures-on-January-11-2024-Backed-by-Crypto-and-Traditional-Finance-Players/default.aspx
Source publication timestamp: 2024-01-12
Raw official headline or title: Cboe Digital Launches Margined Bitcoin and Ether Futures, Announces Successful First Trade
Mechanism classification: Opened a new U.S. regulated margin-futures venue for BTC, but this is a venue- architecture expansion rather than a first-regulated-BTC-futures event.
Ex-ante durability: medium
Estimated access impact: material additional venue access
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: first-trade intraday UTC
Uncertainty notes: Cboe’s official 2024-01-12 release confirms the 2024-01-11 launch and a successful first trade; the intraday first-trade timestamp is not disclosed.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_ETH_2024_01_11_CBOE_DIGITAL_MARGIN_ETH
Original collector event IDs: C2_ETH_2024_01_11_CBOE_DIGITAL_MARGIN_ETH
Catalyst cluster ID: ETH_US_REGULATED_DERIVATIVES_2020_2024
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: leverage_access
Mechanism subtype: regulated_margin_futures_launch
Direction: mixed
Event state: launched
First public timestamp UTC: 2023-11-13
Official confirmation timestamp UTC: 2024-01-12
Effective timestamp UTC: 2024-01-11
Timestamp precision: date_only
Source confidence: high
Primary source type: regulated-market press release
Primary source URL: https://ir.cboe.com/news/news-details/2024/Cboe-Digital-Launches-Margined-Bitcoin-and-Ether-Futures-Announces-Successful-First-Trade/default.aspx
Supporting source URLs: https://ir.cboe.com/news/news-details/2023/Cboe-Digital-to-Launch-Margined-Bitcoin-and-Ether-Futures-on-January-11-2024-Backed-by-Crypto-and-Traditional-Finance-Players/default.aspx
Source publication timestamp: 2024-01-12
Raw official headline or title: Cboe Digital Launches Margined Bitcoin and Ether Futures, Announces Successful First Trade
Mechanism classification: Added another U.S. regulated ETH derivatives venue with integrated spot and clearing architecture, but not ETH’s first regulated-futures access.
Ex-ante durability: medium
Estimated access impact: material additional venue access
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: first-trade intraday UTC
Uncertainty notes: Cboe’s official 2024-01-12 release confirms the 2024-01-11 launch and a successful first trade; the intraday first-trade timestamp is not disclosed.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_CAKE_2024_MAX_SUPPLY_450M
Original collector event IDs: CAKE-2024-MAX-SUPPLY-450M
Catalyst cluster ID: CAKE_TOKENOMICS_RESETS
Asset ID: pancakeswap
Ticker: CAKE
Known major perp symbols: CAKEUSDT; CAKEUSD; CAKE-PERP
Mechanism family: supply_float
Mechanism subtype: max_supply_reduction
Direction: long
Event state: implemented
First public timestamp UTC: 2023-12-21
Official confirmation timestamp UTC: 2024-01-04
Effective timestamp UTC: 2024-01-04
Timestamp precision: date_only
Source confidence: high
Primary source type: official governance forum and official protocol blog
Primary source URL: https://blog.pancakeswap.finance/articles/pancake-swap-reduces-cake-maximum-supply-to-450-million-charting-a-course-for-deflationary-success
Supporting source URLs: https://forum.pancakeswap.finance/t/discussion-for-proposal-to-reduce-cake-token-total-supply/100; https://docs.pancakeswap.finance/protocol/cake-tokenomics
Source publication timestamp: 2024-01-04 for implementation post
Raw official headline or title: “PancakeSwap Reduces CAKE Maximum Supply to 450 Million...”
Mechanism classification: PancakeSwap reduced CAKE’s maximum supply cap from 750M to 450M, explicitly reframing CAKE around a more deflationary token model.
Ex-ante durability: permanent
Estimated access impact: low
Estimated float impact: high
Ex-ante pre-run risk: medium
Unknown fields: none
Uncertainty notes: first public timestamp uses the forum discussion start, while effective uses the official implementation announcement.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_2024_01_10_SEC_SPOT_ETF_APPROVAL
Original collector event IDs: C2_BTC_2024_01_10_SEC_SPOT_ETF_APPROVAL
Catalyst cluster ID: BTC_US_SPOT_ETF_ADJUDICATION_2022_2024
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: spot_etf_approval
Direction: long
Event state: approved
First public timestamp UTC: 2024-01-10
Official confirmation timestamp UTC: 2024-01-10
Effective timestamp UTC: 2024-01-10
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC omnibus approval order
Primary source URL: https://www.sec.gov/files/rules/sro/nysearca/2024/34-99306.pdf
Supporting source URLs: https://www.govinfo.gov/content/pkg/USCOURTS-caDC-22-01142/pdf/USCOURTS-caDC-22-01142-0.pdf
Source publication timestamp: 2024-01-10
Raw official headline or title: Order Granting Accelerated Approval of Proposed Rule Changes, as Modified by Amendments Thereto, to List and Trade Bitcoin-Based Commodity-Based Trust Shares and Trust Units
Mechanism classification: Formal SEC approval of multiple U.S. spot-BTC ETP rule changes, opening the ETF wrapper to direct bitcoin exposure.
Ex-ante durability: high
Estimated access impact: major U.S. institutional and broker-distribution expansion
Estimated float impact: creations and custody demand possible, but no fixed float change
Ex-ante pre-run risk: high
Unknown fields: exact release time UTC
Uncertainty notes: The order is the regulatory approval phase; exchange trading start is a separate phase below.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_2024_01_11_US_SPOT_ETF_LAUNCH
Original collector event IDs: C2_BTC_2024_01_11_US_SPOT_ETF_LAUNCH
Catalyst cluster ID: BTC_US_SPOT_ETF_ADJUDICATION_2022_2024
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: spot_etf_launch
Direction: long
Event state: launched
First public timestamp UTC: 2024-01-10
Official confirmation timestamp UTC: 2024-01-12
Effective timestamp UTC: 2024-01-11
Timestamp precision: date_only
Source confidence: high
Primary source type: listing venue notice
Primary source URL: https://cdn.cboe.com/resources/trader_news/2024/Trader-E-News-1-12-24.pdf
Supporting source URLs: https://www.cboe.com/us/equities/listings/listed_products/symbols/FBTC/ ; https://www.sec.gov/files/rules/sro/nysearca/2024/34-99306.pdf
Source publication timestamp: 2024-01-12
Raw official headline or title: Cboe BZX Lists Six ETPs Holding Bitcoin
Mechanism classification: Effective transition from approval to live U.S. exchange trading of spot-BTC ETPs.
Ex-ante durability: high
Estimated access impact: full live brokerage and market-maker access expansion
Estimated float impact: creation-unit demand possible, no protocol-level float change
Ex-ante pre-run risk: high
Unknown fields: exact first-trade timestamps across venues
Uncertainty notes: The Cboe notice published on 2024-01-12 confirms listings and the preceding 2024-01-11 launch date; exact first trades across venues remain unknown.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_ETH_2024_DENCUN
Original collector event IDs: ETH-2024-DENCUN
Catalyst cluster ID: ETH_FEE_AND_STAKING_EVOLUTION
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: throughput_and_rollup_fee_reduction
Direction: mixed
Event state: activated
First public timestamp UTC: unknown
Official confirmation timestamp UTC: 2024-03-13
Effective timestamp UTC: 2024-03-13
Timestamp precision: date_only
Source confidence: high
Primary source type: official roadmap
Primary source URL: https://ethereum.org/ethereum-forks/
Supporting source URLs: none
Source publication timestamp: current roadmap page
Raw official headline or title: “Dencun.”
Mechanism classification: Dencun activated blob transactions and related changes that materially expanded rollup utility and altered fee demand channels; the ex-ante direction is mixed because lower data costs and changed burn demand act through different mechanisms.
Ex-ante durability: permanent
Estimated access impact: high
Estimated float impact: low
Ex-ante pre-run risk: medium
Unknown fields: earliest first-public mainnet schedule timestamp; exact UTC activation time in the retrieved source
Uncertainty notes: Exclusion from the token report was too narrow: protocol utility was explicitly within scope. The official fork history verifies activation at date precision.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_INJ_2024_INJ30
Original collector event IDs: INJ-2024-INJ30
Catalyst cluster ID: INJ_TOKENOMICS_UPGRADES
Asset ID: injective
Ticker: INJ
Known major perp symbols: INJUSDT; INJUSD; INJ-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: dynamic_supply_and_deflation_update
Direction: long
Event state: launched
First public timestamp UTC: 2024-04-23
Official confirmation timestamp UTC: 2024-04-23
Effective timestamp UTC: 2024-04-23
Timestamp precision: date_only
Source confidence: high
Primary source type: official protocol blog
Primary source URL: https://injective.com/blog/inj-3-0-release
Supporting source URLs: https://injective.com/blog/introducing-the-inj-supply-squeeze
Source publication timestamp: 2024-04-23
Raw official headline or title: “INJ 3.0 Release.”
Mechanism classification: INJ 3.0 changed supply mechanics toward faster deflation based on staking participation and reduced supply over time through dynamic tokenomic parameters.
Ex-ante durability: phased
Estimated access impact: medium
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: none
Uncertainty notes: parameter details were partly summarized in later official material, but the core event identity and economic channel are clear.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_ETH_2024_05_23_SEC_SPOT_ETF_APPROVAL
Original collector event IDs: C2_ETH_2024_05_23_SEC_SPOT_ETF_APPROVAL
Catalyst cluster ID: ETH_US_SPOT_ETF_APPROVAL_AND_LAUNCH_2024
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: spot_etf_approval
Direction: long
Event state: approved
First public timestamp UTC: 2024-05-23
Official confirmation timestamp UTC: 2024-05-23
Effective timestamp UTC: 2024-05-23
Timestamp precision: date_only
Source confidence: high
Primary source type: SEC omnibus approval order
Primary source URL: https://www.sec.gov/files/rules/sro/nysearca/2024/34-100224.pdf
Supporting source URLs: https://www.sec.gov/files/rules/sro/nysearca/2024/34-100224.pdf
Source publication timestamp: 2024-05-23
Raw official headline or title: Order Granting Accelerated Approval of Proposed Rule Changes, as Modified by Amendments Thereto, to List and Trade Shares of Ether-Based Exchange-Traded Products
Mechanism classification: Formal SEC approval of exchange rule changes for U.S. spot-ETH ETPs.
Ex-ante durability: high
Estimated access impact: major U.S. institutional and brokerage access expansion
Estimated float impact: creations and custody demand possible, no protocol-level float change
Ex-ante pre-run risk: high
Unknown fields: release time UTC
Uncertainty notes: Registration effectiveness and exchange trading began later and are captured separately.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_021
Original collector event IDs: C2-HC-021
Catalyst cluster ID: ROBINHOOD_CRYPTO_DISTRIBUTION_2022_2024
Asset ID: sol_pepe_ada_xrp_basket
Ticker: SOL|PEPE|ADA|XRP
Known major perp symbols: SOLUSDT; PEPEUSDT; ADAUSDT; XRPUSDT
Mechanism family: distribution_integration
Mechanism subtype: brokerage_asset_addition_robinhood_us
Direction: long
Event state: listed
First public timestamp UTC: 2024-11-13
Official confirmation timestamp UTC: 2024-11-13
Effective timestamp UTC: 2024-11-13
Timestamp precision: date_only
Source confidence: high
Primary source type: official brokerage newsroom post
Primary source URL: https://www.robinhood.com/us/en/newsroom/robinhood-crypto-expands-offering-with-solana-sol-pepe-pepe-cardano-ada-amp-xrp-xrp-for-u-s-customers/
Supporting source URLs: none
Source publication timestamp: 2024-11-13
Raw official headline or title: Robinhood Crypto Expands Offering with Solana (SOL), Pepe (PEPE), Cardano (ADA) & XRP (XRP) for U.S. Customers
Mechanism classification: major U.S. broker-app distribution expansion for four assets
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: none material
Uncertainty notes: Robinhood framed this as immediate addition to its U.S. platform; the archival page does not expose separate deposit/withdrawal phases for these assets.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_022
Original collector event IDs: C2-HC-022
Catalyst cluster ID: SOL_US_REGULATED_DERIVATIVES_2025
Asset ID: solana
Ticker: SOL
Known major perp symbols: SOLUSDT; SOL-PERP; SOLUSD_PERP; SOL futures
Mechanism family: leverage_access
Mechanism subtype: regulated_futures_launch
Direction: mixed
Event state: launched
First public timestamp UTC: 2025-02-28
Official confirmation timestamp UTC: 2025-03-17
Effective timestamp UTC: 2025-03-17
Timestamp precision: date_only
Source confidence: high
Primary source type: official exchange press release
Primary source URL: https://www.cmegroup.com/media-room/press-releases/2025/2/28/cme_group_to_launchsolanasolfuturesonmarch17.html
Supporting source URLs: https://www.cmegroup.com/media-room/press-releases/2025/3/18/cme_group_announcesfirsttradesofsolanasolfutures.html
Source publication timestamp: 2025-02-28
Raw official headline or title: CME Group to Launch Solana (SOL) Futures on March
Mechanism classification: first major U.S.-regulated SOL futures access on CME
Ex-ante durability: high
Estimated access impact: very high
Estimated float impact: none direct; synthetic float and hedging access increased
Ex-ante pre-run risk: medium
Unknown fields: intraday launch timestamp beyond date-level disclosure
Uncertainty notes: The announcement set March 17 as the launch date; the follow-up release confirms first trades occurred on Sunday, March 16 for the Monday, March 17 trade date.
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_ETH_2025_PECTRA
Original collector event IDs: ETH-2025-PECTRA
Catalyst cluster ID: ETH_FEE_AND_STAKING_EVOLUTION
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: staking_compounding_and_max_effective_balance_change
Direction: long
Event state: activated
First public timestamp UTC: 2025-04-23
Official confirmation timestamp UTC: 2025-04-23
Effective timestamp UTC: 2025-05-07T10:05:11Z
Timestamp precision: second
Source confidence: high
Primary source type: official protocol blog
Primary source URL: https://blog.ethereum.org/2025/04/23/pectra-mainnet
Supporting source URLs: https://ethereum.org/roadmap/pectra/; https://ethereum.org/roadmap/pectra/maxeb/
Source publication timestamp: 2025-04-23
Raw official headline or title: “Pectra Mainnet Announcement.”
Mechanism classification: Pectra raised validator max effective balance from 32 ETH to 2,048 ETH and introduced compounding-validator improvements and execution-layer-triggered validator actions, materially changing staking capital efficiency and validator operations.
Ex-ante durability: permanent
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: medium
Unknown fields: none
Uncertainty notes: none
Final inclusion status: high
Exclusion or downgrade reason: none

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_023
Original collector event IDs: C2-HC-023
Catalyst cluster ID: XRP_US_REGULATED_DERIVATIVES_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRP-PERP; XRPUSDT-SWAP; XRP futures
Mechanism family: leverage_access
Mechanism subtype: regulated_futures_launch
Direction: mixed
Event state: launched
First public timestamp UTC: 2025-04-24
Official confirmation timestamp UTC: 2025-05-19
Effective timestamp UTC: 2025-05-19
Timestamp precision: date_only
Source confidence: high
Primary source type: official exchange press release
Primary source URL: https://www.cmegroup.com/media-room/press-releases/2025/4/24/cme_group_to_expandcryptoderivativessuitewithlaunchofxrpfutures.html
Supporting source URLs: https://www.cmegroup.com/media-room/press-releases/2025/5/20/cme_group_announcesfirsttradesofxrpfutures.html
Source publication timestamp: 2025-04-24
Raw official headline or title: CME Group to Expand Crypto Derivatives Suite with Launch of XRP Futures
Mechanism classification: first major U.S.-regulated XRP futures access on CME
Ex-ante durability: high
Estimated access impact: very high
Estimated float impact: none direct; synthetic float and hedging access increased
Ex-ante pre-run risk: medium
Unknown fields: intraday globex-open timestamp beyond date-level disclosure
Uncertainty notes: The launch notice says May 19 pending regulatory review; the follow-up confirms first trades on Sunday, May 18 for the Monday, May 19 trade date.
Final inclusion status: high
Exclusion or downgrade reason: none

## 8. Medium-confidence consolidated event register

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_MC_001
Original collector event IDs: C2-MC-001
Catalyst cluster ID: PAYPAL_VENMO_CRYPTO_DISTRIBUTION_2020_2025
Asset ID: btc_eth_bch_ltc_basket
Ticker: BTC|ETH|BCH|LTC
Known major perp symbols: BTCUSDT; ETHUSDT; BCHUSDT; LTCUSDT
Mechanism family: distribution_integration
Mechanism subtype: regional_access_expansion_paypal_uk
Direction: long
Event state: rollout_started
First public timestamp UTC: 2021-08-23
Official confirmation timestamp UTC: 2021-08-23
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: official investor-relations press release mirror
Primary source URL: https://investor.pypl.com/news-and-events/news-details/2021/PayPal-Launches-the-Ability-to-Buy-Hold-and-Sell-Cryptocurrency-in-the-UK/default.aspx
Supporting source URLs: https://newsroom.paypal-corp.com/news-cryptocurrency?l=50
Source publication timestamp: 2021-08-23
Raw official headline or title: PayPal Launches the Ability to Buy, Hold and Sell Cryptocurrency in the UK
Mechanism classification: regional payments-wallet expansion outside the U.S.
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: first eligible UK user availability and full rollout completion date
Uncertainty notes: The official release says the service started rolling out during the week of publication; same-day universal availability is not established.
Final inclusion status: medium
Exclusion or downgrade reason: medium: official UK rollout began during the publication week; exact first and full-rollout dates are unknown

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_APE_2022_STAKING_LAUNCH
Original collector event IDs: APE-2022-STAKING-LAUNCH
Catalyst cluster ID: APE_STAKING_PROGRAM
Asset ID: apecoin
Ticker: APE
Known major perp symbols: APEUSDT; APEUSD; APE-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: staking_launch
Direction: long
Event state: launched
First public timestamp UTC: unknown
Official confirmation timestamp UTC: 2022-12
Effective timestamp UTC: 2022-12
Timestamp precision: month
Source confidence: medium
Primary source type: official staking portal
Primary source URL: https://apestake.io/
Supporting source URLs: none
Source publication timestamp: current portal, describing Year 1 program parameters
Raw official headline or title: “ApeStake.io: Home.”
Mechanism classification: ApeCoin DAO’s staking program created direct APE staking utility, reward emissions, and token lock-up across four pools including APE-only and BAYC-linked pools.
Ex-ante durability: multi-year program
Estimated access impact: high
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: original AIP URLs; exact announcement, confirmation, and launch timestamps
Uncertainty notes: The current official portal verifies the program’s mechanism and allocations, but not a deterministic historical timestamp chain.
Final inclusion status: medium
Exclusion or downgrade reason: medium: current official portal verifies mechanism, not the original deterministic launch timestamp chain

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_SUSHI_2022_KANPAI_20
Original collector event IDs: SUSHI-2022-KANPAI-20
Catalyst cluster ID: SUSHI_FEE_REDIRECTION
Asset ID: sushi
Ticker: SUSHI
Known major perp symbols: SUSHIUSDT; SUSHIUSD; SUSHI-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: fee_redirection_from_stakers_to_treasury
Direction: short
Event state: executed
First public timestamp UTC: unknown
Official confirmation timestamp UTC: 2022-12
Effective timestamp UTC: 2022-12
Timestamp precision: month
Source confidence: medium
Primary source type: official retrospective blog posts
Primary source URL: https://www.sushi.com/blog/breaking-down-the-sushi-tokenomics
Supporting source URLs: https://www.sushi.com/blog/sushi-bar-faq
Source publication timestamp: 2023-11-07 and 2023-11-24
Raw official headline or title: “Breaking down the Sushi Tokenomics” / “Important Update: Kanpai 2.0 expires, xSushi revived!”
Mechanism classification: Sushi stated that in December 2022 the Kanpai proposal redirected 100% of xSUSHI protocol fees to the DAO treasury, temporarily shutting off xSUSHI fee accrual.
Ex-ante durability: temporary
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: medium
Unknown fields: original governance proposal URL; exact vote and execution timestamps
Uncertainty notes: The mechanism is supported only by official retrospective posts, not the original proposal/execution artifact.
Final inclusion status: medium
Exclusion or downgrade reason: medium: official evidence is retrospective and the original proposal/execution artifact was not retrieved

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EX_002
Original collector event IDs: C2-EX-002
Catalyst cluster ID: CL-BINANCEUS-USD-PAIR-CONTRACTION
Asset ID: ada_matic_sol_basket
Ticker: ADA|MATIC|SOL
Known major perp symbols: ADAUSDT; MATICUSDT; SOLUSDT
Mechanism family: exchange_spot_access
Mechanism subtype: fiat_pair_access_contraction_without_full_delisting
Direction: short
Event state: pairs_removed
First public timestamp UTC: 2023-07-13
Official confirmation timestamp UTC: 2023-07-13
Effective timestamp UTC: 2023-07-14 03:00
Timestamp precision: minute
Source confidence: medium
Primary source type: official help-center notice
Primary source URL: https://support.binance.us/en/articles/9843567-binance-us-will-remove-select-usd-advanced-trading-pairs
Supporting source URLs: https://support.binance.us/en/collections/10384537-announcements
Source publication timestamp: 2025-03-06
Raw official headline or title: Binance.US will remove select USD advanced trading pairs
Mechanism classification: Binance.US removed specified USD advanced-trading pairs for ADA, MATIC, and SOL while leaving other crypto-denominated access paths available.
Ex-ante durability: medium
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: low
Unknown fields: original 2023 publication shell before help-center migration
Uncertainty notes: The event is a real, partial access contraction rather than a full delisting. The official article was later republished in a migrated help center, weakening the point-in-time shell.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded to medium: partial pair-level contraction and republished source shell

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_020
Original collector event IDs: C2-HC-020
Catalyst cluster ID: XRP_SEC_AND_US_ACCESS_2020_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRP-PERP; XRPUSDT-SWAP; XRP futures
Mechanism family: exchange_spot_access
Mechanism subtype: major_relisting_after_legal_exclusion
Direction: long
Event state: re_enabled
First public timestamp UTC: 2023-07-13
Official confirmation timestamp UTC: 2023-07-13
Effective timestamp UTC: 2023-07-13
Timestamp precision: date_only
Source confidence: medium
Primary source type: official social announcement by exchange listings account
Primary source URL: https://x.com/CoinbaseAssets/status/1679575239657758721
Supporting source URLs: https://www.coinbase.com/blog/coinbase-will-suspend-trading-in-xrp-on-january-19
Source publication timestamp: 2023-07-13
Raw official headline or title: Coinbase will re-enable trading for XRP (XRP) on the XRP network.
Mechanism classification: major relisting after prolonged regulatory-access exclusion
Ex-ante durability: medium
Estimated access impact: very high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: exact transition time from announcement to fully enabled trading
Uncertainty notes: The source is an official Coinbase Assets social post rather than a stable exchange notice. The date and re-enablement are supportable, but intraday state completion is not.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: official social-post source is archive-fragile and exact re-enablement time is unknown

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_MC_005
Original collector event IDs: C2-MC-005
Catalyst cluster ID: XRP_SEC_AND_US_ACCESS_2020_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRP-PERP; XRPUSDT-SWAP; XRP futures
Mechanism family: exchange_spot_access
Mechanism subtype: regional_relisting_binance_us
Direction: long
Event state: relisted
First public timestamp UTC: 2023-07-13
Official confirmation timestamp UTC: 2023-07-14
Effective timestamp UTC: 2023-07-14
Timestamp precision: date_only
Source confidence: medium
Primary source type: official help-center listing notice
Primary source URL: https://support.binance.us/en/articles/9843566-binance-us-lists-xrp-deposit-trade-now
Supporting source URLs: https://support.binance.us/en/collections/10384537-announcements
Source publication timestamp: 2025-03-06
Raw official headline or title: Binance.US lists XRP | Deposit & trade now
Mechanism classification: regional U.S. relisting after prior exclusion
Ex-ante durability: medium
Estimated access impact: medium
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: original 2023 article permalink state before later help-center republication
Uncertainty notes: The page is official and explicitly says deposits first opened on 2023-07-13 and trading went live on 2023-07-14, but the article shell shows a 2025 publication date because of later help-center migration.
Final inclusion status: medium
Exclusion or downgrade reason: medium: event details are official but the help-center shell was republished in 2025

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_BTC_ETH_2023_08_16_COINBASE_FCM_APPROVAL_BTC
Original collector event IDs: C2_BTC_ETH_2023_08_16_COINBASE_FCM_APPROVAL_BTC
Catalyst cluster ID: BTC_US_REGULATED_DERIVATIVES_2021_2024
Asset ID: bitcoin
Ticker: BTC
Known major perp symbols: BTCUSDT; BTCUSD; XBTUSD; BTC-PERP
Mechanism family: leverage_access
Mechanism subtype: broker_fcm_approval_for_regulated_crypto_futures
Direction: mixed
Event state: regulatory_approval_received
First public timestamp UTC: 2023-08-16
Official confirmation timestamp UTC: 2023-08-16
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: exchange-affiliate announcement
Primary source URL: https://www.coinbase.com/blog/coinbase-financial-markets-inc-secures-approval-to-bring-regulated
Supporting source URLs: https://www.coinbase.com/blog/us-traders-can-now-trade-regulated-leveraged-crypto-futures-through-coinbase
Source publication timestamp: 2023-08-16
Raw official headline or title: Coinbase Financial Markets, Inc. secures approval to bring federally regulated crypto futures trading to eligible US customers
Mechanism classification: Approval for a broker/FCM access layer that materially improved U.S. regulated futures distribution, but the asset mapping is tied mainly to BTC and ETH futures already listed by Coinbase Derivatives.
Ex-ante durability: high
Estimated access impact: broker-distribution expansion
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: asset-specific customer-access start date; exact eligible-customer rollout date
Uncertainty notes: The source verifies FCM approval on 2023-08-16 but states that direct customer access would follow; approval is not equivalent to same-day live distribution.
Final inclusion status: medium
Exclusion or downgrade reason: medium: approval date is verified; asset-specific customer access began later at an unverified date

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_ETH_2023_08_16_COINBASE_FCM_APPROVAL_ETH
Original collector event IDs: C2_ETH_2023_08_16_COINBASE_FCM_APPROVAL_ETH
Catalyst cluster ID: ETH_US_REGULATED_DERIVATIVES_2020_2024
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: leverage_access
Mechanism subtype: broker_fcm_approval_for_regulated_crypto_futures
Direction: mixed
Event state: regulatory_approval_received
First public timestamp UTC: 2023-08-16
Official confirmation timestamp UTC: 2023-08-16
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: exchange-affiliate announcement
Primary source URL: https://www.coinbase.com/blog/coinbase-financial-markets-inc-secures-approval-to-bring-regulated
Supporting source URLs: https://www.coinbase.com/blog/us-traders-can-now-trade-regulated-leveraged-crypto-futures-through-coinbase
Source publication timestamp: 2023-08-16
Raw official headline or title: Coinbase Financial Markets, Inc. secures approval to bring federally regulated crypto futures trading to eligible US customers
Mechanism classification: Same approval cluster as BTC, with ETH as a covered asset through Coinbase’s already-listed regulated futures lineup.
Ex-ante durability: high
Estimated access impact: broker-distribution expansion
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: asset-specific customer-access start date; exact eligible-customer rollout date
Uncertainty notes: The source verifies FCM approval on 2023-08-16 but states that direct customer access would follow; approval is not equivalent to same-day live distribution.
Final inclusion status: medium
Exclusion or downgrade reason: medium: approval date is verified; asset-specific customer access began later at an unverified date

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_SUSHI_2024_KANPAI_EXPIRY
Original collector event IDs: SUSHI-2024-KANPAI-EXPIRY
Catalyst cluster ID: SUSHI_FEE_REDIRECTION
Asset ID: sushi
Ticker: SUSHI
Known major perp symbols: SUSHIUSDT; SUSHIUSD; SUSHI-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: fee_reversion_to_stakers
Direction: long
Event state: expired_and_reverted
First public timestamp UTC: 2023-11-24
Official confirmation timestamp UTC: 2023-11-24
Effective timestamp UTC: 2024-01-23
Timestamp precision: date_only
Source confidence: medium
Primary source type: official protocol blog and FAQ
Primary source URL: https://www.sushi.com/blog/kanpai-expires-xsushi-revived
Supporting source URLs: https://www.sushi.com/blog/sushi-bar-faq; https://www.sushi.com/blog/breaking-down-the-sushi-tokenomics
Source publication timestamp: 2023-11-24 and 2024-01-17
Raw official headline or title: “Important Update: Kanpai 2.0 expires, xSushi revived!”
Mechanism classification: on expiry, the 0.05% Sushi v2 fee stream returned to xSUSHI holders, reviving direct fee-sharing to stakers.
Ex-ante durability: long-lived
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: medium
Unknown fields: exact original governance expiration record
Uncertainty notes: the economic change is clear, but the original approval artifact for the expiry schedule was not retrieved in this pass.
Final inclusion status: medium
Exclusion or downgrade reason: medium: economic reversion is supported, but the original governance expiration artifact was not retrieved

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EX_003
Original collector event IDs: C2-EX-003
Catalyst cluster ID: CL-BONK-LISTING-CYCLE
Asset ID: bonk
Ticker: BONK
Known major perp symbols: BONKUSDT; BONK-PERP
Mechanism family: exchange_spot_access
Mechanism subtype: roadmap_then_listing_on_coinbase
Direction: long
Event state: roadmap_then_live_listing
First public timestamp UTC: 2023-12-12
Official confirmation timestamp UTC: 2023-12-14
Effective timestamp UTC: 2023-12-14
Timestamp precision: date_only
Source confidence: medium
Primary source type: official social announcements by exchange listings account
Primary source URL: https://x.com/CoinbaseAssets/status/1734739348308787524
Supporting source URLs: https://x.com/CoinbaseAssets/status/1735005527845773794 ; https://x.com/CoinbaseAssets/status/1735347827541172483
Source publication timestamp: 2023-12-12
Raw official headline or title: Asset added to the roadmap today: Bonk (BONK)
Mechanism classification: Coinbase first placed BONK on its roadmap and then announced live trading, expanding access on a major U.S. venue.
Ex-ante durability: medium
Estimated access impact: high
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: exact order-book transition time and liquidity-condition completion
Uncertainty notes: The roadmap post alone is not implemented access. The later official Coinbase Assets posts support a live listing on 2023-12-14, but social-post archival stability and conditional launch wording warrant medium confidence.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded to medium: live listing supported, but only through fragile official social posts

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_MC_004
Original collector event IDs: C2-MC-004
Catalyst cluster ID: TON_OKX_AND_TELEGRAM_ACCESS_2021_2023
Asset ID: ton
Ticker: TON
Known major perp symbols: TONUSDT; TONUSDT-SWAP
Mechanism family: distribution_integration
Mechanism subtype: messaging_platform_wallet_distribution
Direction: long
Event state: announced_and_referenced
First public timestamp UTC: 2023-Q3
Official confirmation timestamp UTC: 2023-10-06
Effective timestamp UTC: unknown
Timestamp precision: quarter_and_date_only
Source confidence: medium
Primary source type: official ecosystem report / foundation page
Primary source URL: https://ton.org/en/ton-developer-report-q3-2023
Supporting source URLs: https://ton.org/
Source publication timestamp: 2023-10-06
Raw official headline or title: TON Developer Report: Q3 2023
Mechanism classification: wallet-in-messaging distribution widening for TON inside Telegram ecosystem
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low
Ex-ante pre-run risk: medium
Unknown fields: first public announcement date; exact live rollout date; launch geography
Uncertainty notes: The official Q3 developer report references TON Space and later ecosystem integration, but the direct report URL now redirects and the original first-announcement artifact was not robustly retrievable.
Final inclusion status: medium
Exclusion or downgrade reason: medium: ecosystem report is indirect, first announcement is missing, and effective rollout timing is unresolved

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_STRK_2024_UNLOCK_SCHEDULE_REVISION
Original collector event IDs: STRK-2024-UNLOCK-SCHEDULE-REVISION
Catalyst cluster ID: STRK_FLOAT_SCHEDULE_REWRITE
Asset ID: starknet
Ticker: STRK
Known major perp symbols: STRKUSDT; STRKUSD; STRK-PERP
Mechanism family: supply_float
Mechanism subtype: unlock_postponement_and_smoothing
Direction: long
Event state: revised_schedule_reflected_in_official_docs
First public timestamp UTC: 2024-02
Official confirmation timestamp UTC: unknown
Effective timestamp UTC: 2024-04-15
Timestamp precision: month_and_date_only
Source confidence: medium
Primary source type: official token docs, supported by reputable secondary coverage
Primary source URL: https://docs.starknet.io/learn/protocol/strk
Supporting source URLs: https://blockworks.co/news/starkware-airdrop-token-lockup
Source publication timestamp: current docs page; secondary report on 2024-02-22
Raw official headline or title: “STRK - Starknet Documentation.”
Mechanism classification: official docs show a smoothed monthly unlock path beginning with 64M tokens monthly from 2024-04-15 to 2025-03-15, which aligned with the widely reported revision that reduced the initial early-contributor and investor unlock shock.
Ex-ante durability: multi-year schedule
Estimated access impact: low
Estimated float impact: high
Ex-ante pre-run risk: high
Unknown fields: contemporaneous official revision announcement; exact confirmation time
Uncertainty notes: Current official docs show the revised schedule, but the first-public date relies on secondary contemporaneous reporting. This prevents high confidence.
Final inclusion status: medium
Exclusion or downgrade reason: medium: revised schedule is visible in current official docs, but the contemporaneous official revision notice is missing

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_SOL_2024_SIMD0096
Original collector event IDs: SOL-2024-SIMD0096
Catalyst cluster ID: SOL_FEE_AND_REWARD_REDISTRIBUTION
Asset ID: solana
Ticker: SOL
Known major perp symbols: SOLUSDT; SOLUSD; SOL-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: priority_fee_distribution_change
Direction: mixed
Event state: approved_activation_unverified
First public timestamp UTC: 2024-05-08
Official confirmation timestamp UTC: 2024-05-08
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: official governance forum and official SIMD repo reference
Primary source URL: https://forum.solana.com/t/proposal-for-enabling-the-reward-full-priority-fee-to-validator-on-solana-mainnet-beta/1456
Supporting source URLs: none
Source publication timestamp: 2024-05-08
Raw official headline or title: “Proposal for Enabling the Reward Full Priority Fee to Validator on Solana Mainnet-beta.”
Mechanism classification: SIMD-0096 proposed moving priority-fee distribution from a 50/50 validator-burn split to 100% validator allocation, improving validator economics but reducing fee burn.
Ex-ante durability: permanent if active
Estimated access impact: low
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: mainnet feature-gate activation timestamp
Uncertainty notes: The proposal and validator approval are supported. The audit did not locate an official mainnet activation record, so this is an approval-state event only.
Final inclusion status: medium
Exclusion or downgrade reason: medium: proposal approval is supported; mainnet feature activation remains unverified

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_ETH_2024_07_23_US_SPOT_ETF_LAUNCH
Original collector event IDs: C2_ETH_2024_07_23_US_SPOT_ETF_LAUNCH
Catalyst cluster ID: ETH_US_SPOT_ETF_APPROVAL_AND_LAUNCH_2024
Asset ID: ethereum
Ticker: ETH
Known major perp symbols: ETHUSDT; ETHUSD; ETH-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: spot_etf_launch
Direction: long
Event state: launched
First public timestamp UTC: 2024-07-22
Official confirmation timestamp UTC: 2024-07-23
Effective timestamp UTC: 2024-07-23
Timestamp precision: date_only
Source confidence: medium
Primary source type: listing venue page
Primary source URL: https://www.cboe.com/us/equities/listings/listed_products/issuer_detail/FRNK/
Supporting source URLs: https://www.coinbase.com/blog/momentous-approvals-spot-etfs-to-usher-in-bitcoins-next-chapter-and ; https://www.sec.gov/files/rules/sro/nysearca/2024/34-100224.pdf
Source publication timestamp: 2024-07-23 (listing date on exchange page)
Raw official headline or title: Franklin Templeton Products Listed on Cboe
Mechanism classification: Transition from approved rule changes to live exchange trading for at least one of the newly approved U.S. spot-ETH products, representing the broader launch phase.
Ex-ante durability: high
Estimated access impact: full live brokerage and market-maker access expansion
Estimated float impact: creation-unit demand possible, no protocol-level float change
Ex-ante pre-run risk: high
Unknown fields: specific first-trade timestamps across all venues; exact registration-effective time
Uncertainty notes: The cited Cboe issuer page confirms a 2024-07-23 listing date but is a current listing page rather than a preserved cross-venue contemporaneous launch notice; exact registration-effective and first-trade times remain unknown.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: contemporaneous cross-venue launch evidence and exact first-trade timing are incomplete

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_STRK_2024_STAKING_LAUNCH
Original collector event IDs: STRK-2024-STAKING-LAUNCH
Catalyst cluster ID: STRK_STAKING_AND_MINT_CURVE
Asset ID: starknet
Ticker: STRK
Known major perp symbols: STRKUSDT; STRKUSD; STRK-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: staking_launch_and_reward_curve
Direction: long
Event state: launched
First public timestamp UTC: 2024-09-09
Official confirmation timestamp UTC: 2024-11-26
Effective timestamp UTC: 2024-11-26
Timestamp precision: date_only
Source confidence: medium
Primary source type: official governance/educational pages
Primary source URL: https://www.starknet.io/blog/first-community-vote-staking/
Supporting source URLs: https://www.starknet.io/blog/staking-with-liquid-staking-tokens/
Source publication timestamp: 2024-11-26 for mainnet staking launch reference in official docs
Raw official headline or title: “First Starknet community vote on mainnet” / “Starknet Staking.”
Mechanism classification: Starknet introduced a minting curve to enable staking rewards and then launched phase-one STRK staking on mainnet, creating direct staking utility and inflation-linked reward mechanics.
Ex-ante durability: phased
Estimated access impact: high
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: clean contemporaneous mainnet launch notice and exact activation time
Uncertainty notes: The official material supports the governance-enabling vote and staking framework, but the cited primary URL is a vote page rather than a preserved launch artifact. The 2024-11-26 date is retained at medium confidence.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: cited primary artifact verifies the vote/framework more clearly than the live launch timestamp

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_MC_002
Original collector event IDs: C2-MC-002
Catalyst cluster ID: ROBINHOOD_CRYPTO_DISTRIBUTION_2022_2024
Asset ID: btc_eth_sol_usdc_plus_basket
Ticker: BTC|ETH|SOL|USDC|+
Known major perp symbols: BTCUSDT; ETHUSDT; SOLUSDT
Mechanism family: distribution_integration
Mechanism subtype: brokerage_transfer_interoperability_robinhood_europe
Direction: mixed
Event state: launched
First public timestamp UTC: 2024-10-01
Official confirmation timestamp UTC: 2024-10-01
Effective timestamp UTC: 2024-10-01
Timestamp precision: date_only
Source confidence: medium
Primary source type: official brokerage newsroom post
Primary source URL: https://robinhood.com/us/en/newsroom/launching-crypto-transfers-in-europe/
Supporting source URLs: none
Source publication timestamp: 2024-10-01
Raw official headline or title: Robinhood Crypto Launches Crypto Transfers in Europe
Mechanism classification: deposits/withdrawals and self-custody interoperability for European users
Ex-ante durability: high
Estimated access impact: medium-to-high
Estimated float impact: low
Ex-ante pre-run risk: low
Unknown fields: full supported-asset list at launch from the linked help-center page
Uncertainty notes: The official page explicitly names BTC, ETH, SOL, and USDC among 20+ supported assets, but the complete basket is delegated to a help-center link.
Final inclusion status: medium
Exclusion or downgrade reason: medium: official source names only part of the 20-plus asset basket

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_SOL_2025_01_24_SPOT_SOL_ETF_FILING
Original collector event IDs: C2_SOL_2025_01_24_SPOT_SOL_ETF_FILING
Catalyst cluster ID: ALTCOIN_US_SPOT_ETF_FILINGS_2025
Asset ID: solana
Ticker: SOL
Known major perp symbols: SOLUSDT; SOLUSD; SOL-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: initial_spot_etf_rule_filing
Direction: long
Event state: filed
First public timestamp UTC: 2025-01-24
Official confirmation timestamp UTC: 2025-03-11
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: exchange rule filing
Primary source URL: https://www.nyse.com/publicdocs/nyse/markets/nyse-arca/rule-filings/filings/2025/SR-NYSEArca-2025-06.pdf
Supporting source URLs: https://www.sec.gov/files/rules/sro/nysearca/2025/34-102593.pdf ; https://www.sec.gov/files/rules/sro/nysearca/2025/34-102372.pdf
Source publication timestamp: 2025-01-24 for the exchange filing; 2025-03-11 for the SEC designation order
Raw official headline or title: Notice of Designation of a Longer Period for Commission Action on a Proposed Rule Change, as Modified by Amendment No. 1, to List and Trade Shares of the Grayscale Solana Trust under NYSE Arca Rule 8.201-E, Commodity-Based Trust Shares
Mechanism classification: Initial formal U.S. spot-SOL ETF pathway filing, but still pre-approval and embedded in a crowded multi-issuer filing wave.
Ex-ante durability: medium
Estimated access impact: meaningful expectation shift for future institutional wrapper access
Estimated float impact: none immediate
Ex-ante pre-run risk: high
Unknown fields: publication of a clean SEC notice-of-filing document tied to the first filing date in an easily parsable format
Uncertainty notes: The first-date grounding is exchange-side; later SEC documents confirm that January 24 filing date.
Final inclusion status: medium
Exclusion or downgrade reason: medium: official filing phase only; no access was yet effective and the earliest publication chain is exchange-side

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_XRP_2025_02_06_SPOT_XRP_ETF_FILING
Original collector event IDs: C2_XRP_2025_02_06_SPOT_XRP_ETF_FILING
Catalyst cluster ID: ALTCOIN_US_SPOT_ETF_FILINGS_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRPUSD; XRP-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: initial_spot_etf_rule_filing
Direction: long
Event state: filed
First public timestamp UTC: 2025-02-06
Official confirmation timestamp UTC: 2025-02-19
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: SEC notice of filing
Primary source URL: https://www.sec.gov/files/rules/sro/cboebzx/2025/34-102449.pdf
Supporting source URLs: https://cdn.cboe.com/resources/regulation/rule_filings/pending/2025/SR-CboeBZX-2025-022.pdf
Source publication timestamp: 2025-02-19
Raw official headline or title: Notice of Filing of a Proposed Rule Change to List and Trade Shares of the Canary XRP Trust Under BZX Rule 14.11(e)(4), Commodity-Based Trust Shares
Mechanism classification: Opened a formal U.S. spot-XRP ETF review pathway, but remained pre-approval and subject to routine subsequent process steps.
Ex-ante durability: medium
Estimated access impact: meaningful expectation shift for future wrapper access
Estimated float impact: none immediate
Ex-ante pre-run risk: high
Unknown fields: eventual disposition within 2025
Uncertainty notes: Included because the filing itself is official and non-routine; still medium because it did not yet grant access.
Final inclusion status: medium
Exclusion or downgrade reason: medium: official filing phase only; no approval or effective access

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_LTC_2025_02_07_SPOT_LTC_ETF_FILING
Original collector event IDs: C2_LTC_2025_02_07_SPOT_LTC_ETF_FILING
Catalyst cluster ID: ALTCOIN_US_SPOT_ETF_FILINGS_2025
Asset ID: litecoin
Ticker: LTC
Known major perp symbols: LTCUSDT; LTCUSD; LTC-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: initial_spot_etf_rule_filing
Direction: long
Event state: filed
First public timestamp UTC: 2025-02-07
Official confirmation timestamp UTC: 2025-02-19
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: SEC notice of filing
Primary source URL: https://www.sec.gov/files/rules/sro/nasdaq/2025/34-102444.pdf
Supporting source URLs: https://www.sec.gov/files/rules/sro/nasdaq/2025/34-102606.pdf
Source publication timestamp: 2025-02-19
Raw official headline or title: Notice of Filing of Proposed Rule Change to List and Trade Shares of the CoinShares Litecoin ETF under Nasdaq Rule 5711(d)
Mechanism classification: Formal start of a U.S. spot-LTC ETF approval pathway.
Ex-ante durability: medium
Estimated access impact: meaningful expectation shift for future wrapper access
Estimated float impact: none immediate
Ex-ante pre-run risk: high
Unknown fields: final SEC disposition in 2025
Uncertainty notes: Kept at medium because filing does not equal approval or launch.
Final inclusion status: medium
Exclusion or downgrade reason: medium: official filing phase only; no approval or effective access

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_SOL_2025_02_18_COINBASE_DERIVATIVES_FUTURES
Original collector event IDs: C2_SOL_2025_02_18_COINBASE_DERIVATIVES_FUTURES
Catalyst cluster ID: SOL_US_REGULATED_DERIVATIVES_2025
Asset ID: solana
Ticker: SOL
Known major perp symbols: SOLUSDT; SOLUSD; SOL-PERP
Mechanism family: leverage_access
Mechanism subtype: cftc_self_certified_futures_launch
Direction: mixed
Event state: self_certified_and_market_availability_partially_confirmed
First public timestamp UTC: 2025-02-12
Official confirmation timestamp UTC: 2025-02-12
Effective timestamp UTC: 2025-02-18
Timestamp precision: date_only
Source confidence: medium
Primary source type: CFTC self-certification
Primary source URL: https://www.cftc.gov/sites/default/files/filings/ptc/25/02/ptc02122515280.pdf
Supporting source URLs: https://www.coinbase.com/blog/coinbase-unlocking-new-opportunities-in-crypto-derivatives
Source publication timestamp: 2025-02-12
Raw official headline or title: 2025-03 Listing of Solana Futures
Mechanism classification: Coinbase Derivatives self-certified SOL futures for an on-or-after 2025-02-18 trade date, expanding regulated leverage and hedging access if and as distributed through partner platforms.
Ex-ante durability: high
Estimated access impact: major regulated derivatives access expansion
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: first-trade timestamp intraday
Uncertainty notes: The CFTC filing fixes the earliest permissible trade date. The contemporaneous Coinbase communication did not cleanly preserve a first-trade timestamp or partner-platform availability sequence.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: self-certification fixes an earliest permissible date, not a fully corroborated first-trade timestamp

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_DOGE_2025_03_03_SPOT_DOGE_ETF_FILING
Original collector event IDs: C2_DOGE_2025_03_03_SPOT_DOGE_ETF_FILING
Catalyst cluster ID: ALTCOIN_US_SPOT_ETF_FILINGS_2025
Asset ID: dogecoin
Ticker: DOGE
Known major perp symbols: DOGEUSDT; DOGEUSD; DOGE-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: initial_spot_etf_rule_filing
Direction: long
Event state: filed
First public timestamp UTC: 2025-03-03
Official confirmation timestamp UTC: 2025-04-29
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: SEC order referencing initial filing date
Primary source URL: https://www.sec.gov/files/rules/sro/nysearca/2025/34-102942.pdf
Supporting source URLs: https://www.sec.gov/files/rules/sro/nasdaq/2025/34-103032.pdf
Source publication timestamp: 2025-04-29
Raw official headline or title: Notice of Designation of a Longer Period for Commission Action on a Proposed Rule Change to List and Trade Shares of the Bitwise Dogecoin ETF under NYSE Arca Rule 8.201-E (Commodity-Based Trust Shares)
Mechanism classification: The official record confirms a March 3 filing for a spot-DOGE ETF proposal; still pre-approval and part of a broader wave of meme-asset wrapper filings.
Ex-ante durability: medium
Estimated access impact: modest-to-meaningful expectation shift
Estimated float impact: none immediate
Ex-ante pre-run risk: high
Unknown fields: which DOGE proposal should be treated as the dominant cluster anchor for later ingestion
Uncertainty notes: Multiple DOGE filing paths existed in 2025; this entry keeps the earliest clearly grounded official date.
Final inclusion status: medium
Exclusion or downgrade reason: medium: official filing phase only and multiple parallel DOGE proposals create cluster-anchor ambiguity

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_DYDX_2025_BUYBACK
Original collector event IDs: DYDX-2025-BUYBACK
Catalyst cluster ID: DYDX_REVENUE_LINKAGE
Asset ID: dydx
Ticker: DYDX
Known major perp symbols: DYDXUSDT; DYDXUSD; DYDX-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: fee_to_buyback_and_stake
Direction: long
Event state: launched
First public timestamp UTC: 2025-03-20
Official confirmation timestamp UTC: 2025-03-24
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: official protocol blog
Primary source URL: https://www.dydx.xyz/blog/dydx-buyback-program
Supporting source URLs: https://www.dydx.xyz/categories/announcements
Source publication timestamp: 2025-03-24
Raw official headline or title: “dYdX Community Launches First-Ever DYDX Buyback Program.”
Mechanism classification: the community directed 25% of net protocol fees to systematic monthly DYDX buybacks and staking, creating a direct revenue-to-token-capture linkage.
Ex-ante durability: long-lived
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: first executed purchase timestamp; whether the 2025-03-24 page update changed launch wording
Uncertainty notes: The official page exposes an original publication date and a later update date while stating the program was starting “today.” Without an archived page version or first transaction, the effective date is not deterministic. Asset mapping should distinguish exchange-era Ethereum DYDX from the dYdX Chain token representation where venue histories differ.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: official page publication/update chronology does not fix the first executed buyback

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_XRP_2025_04_21_COINBASE_DERIVATIVES_FUTURES
Original collector event IDs: C2_XRP_2025_04_21_COINBASE_DERIVATIVES_FUTURES
Catalyst cluster ID: XRP_US_REGULATED_DERIVATIVES_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRPUSD; XRP-PERP
Mechanism family: leverage_access
Mechanism subtype: cftc_self_certified_futures_launch
Direction: mixed
Event state: self_certified_with_later_product_availability_confirmation
First public timestamp UTC: 2025-04-03
Official confirmation timestamp UTC: 2025-04-03
Effective timestamp UTC: 2025-04-21
Timestamp precision: date_only
Source confidence: medium
Primary source type: CFTC self-certification
Primary source URL: https://www.cftc.gov/sites/default/files/filings/ptc/25/04/ptc04032518916.pdf
Supporting source URLs: https://www.coinbase.com/blog/coinbase-futures-spring-2025-release-more-hours-more-contracts-and-more-perpetual
Source publication timestamp: 2025-04-03
Raw official headline or title: 2025-17 Listing of the XRP Futures
Mechanism classification: Coinbase Derivatives self-certified XRP futures for an on-or-after 2025-04-21 trade date; later official product communications confirm XRP futures in the lineup.
Ex-ante durability: high
Estimated access impact: major regulated derivatives access expansion
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: first-trade timestamp intraday
Uncertainty notes: The self-certification establishes the permissible launch date, but a contemporaneous first-trade or partner-access timestamp for 2025-04-21 was not independently fixed.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: self-certification and later availability evidence do not fix the exact first live date

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_HC_024
Original collector event IDs: C2-HC-024
Catalyst cluster ID: PAYPAL_VENMO_CRYPTO_DISTRIBUTION_2020_2025
Asset ID: link_sol_basket
Ticker: LINK|SOL
Known major perp symbols: LINKUSDT; SOLUSDT; LINK-PERP; SOL-PERP
Mechanism family: distribution_integration
Mechanism subtype: payments_wallet_asset_addition_paypal_venmo
Direction: long
Event state: rollout_announced
First public timestamp UTC: 2025-04-04
Official confirmation timestamp UTC: 2025-04-04
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: official corporate press release
Primary source URL: https://newsroom.paypal-corp.com/2025-04-04-PayPal-Expands-Cryptocurrency-Offerings-with-New-Tokens-Chainlink-and-Solana-Now-Available
Supporting source URLs: none
Source publication timestamp: 2025-04-04
Raw official headline or title: PayPal Expands Cryptocurrency Offerings with New Tokens: Chainlink and Solana Now Available
Mechanism classification: wallet-native retail access expansion to additional large-cap assets
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low secondary-market redistribution only
Ex-ante pre-run risk: medium
Unknown fields: first user-cohort availability and full rollout completion dates
Uncertainty notes: The official release says LINK and SOL would appear over the following weeks. Publication is confirmation, not a deterministic live-access timestamp.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: official release announced a multi-week rollout and does not establish a single effective timestamp

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_CAKE_2025_TOKENOMICS_30
Original collector event IDs: CAKE-2025-TOKENOMICS-30
Catalyst cluster ID: CAKE_TOKENOMICS_RESETS
Asset ID: pancakeswap
Ticker: CAKE
Known major perp symbols: CAKEUSDT; CAKEUSD; CAKE-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: emissions_reduction_and_tokenomics_rewrite
Direction: long
Event state: approved_execution_not_fully_timestamped
First public timestamp UTC: 2025-04-08
Official confirmation timestamp UTC: 2025-04-15
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: official governance forum and official voting page
Primary source URL: https://forum.pancakeswap.finance/t/cake-tokenomics-proposal-3-0-true-ownership-simplified-governance-and-sustainable-growth/1237
Supporting source URLs: https://pancakeswap.finance/voting/proposal/0x79ef496c9737e48d9677a6e291ff2a549dee6729c9996398e453af8ecbf0ceb3; https://docs.pancakeswap.finance/protocol/cake-tokenomics
Source publication timestamp: 2025-04-15 for vote page
Raw official headline or title: “CAKE Tokenomics Proposal 3.0.”
Mechanism classification: Tokenomics 3.0 retired veCAKE and reduced daily CAKE emissions from about 40,000 to 22,500, explicitly targeting ongoing deflation and a simplified emission structure.
Ex-ante durability: long-lived
Estimated access impact: medium
Estimated float impact: medium
Ex-ante pre-run risk: high
Unknown fields: on-chain execution transaction or parameter-change timestamp
Uncertainty notes: The official forum and vote artifacts support proposal content and approval, but treating vote completion as immediate execution invents an implementation timestamp.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: proposal approval is supported, but on-chain execution timing was not verified

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_AVAX_2025_04_09_SPOT_AVAX_ETF_FILING
Original collector event IDs: C2_AVAX_2025_04_09_SPOT_AVAX_ETF_FILING
Catalyst cluster ID: ALTCOIN_US_SPOT_ETF_FILINGS_2025
Asset ID: avalanche
Ticker: AVAX
Known major perp symbols: AVAXUSDT; AVAXUSD; AVAX-PERP
Mechanism family: regulated_institutional_access
Mechanism subtype: initial_spot_etf_rule_filing
Direction: long
Event state: filed
First public timestamp UTC: 2025-04-09
Official confirmation timestamp UTC: 2025-04-23
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: SEC notice of filing
Primary source URL: https://www.sec.gov/files/rules/sro/nasdaq/2025/34-102917.pdf
Supporting source URLs: https://www.sec.gov/files/rules/sro/nasdaq/2025/34-103543.pdf
Source publication timestamp: 2025-04-23
Raw official headline or title: Notice of Filing of Proposed Rule Change to List and Trade Shares of the VanEck Avalanche ETF under Nasdaq Rule 5711(d)
Mechanism classification: Formal initiation of a U.S. spot-AVAX ETF pathway, but still a pre-access phase.
Ex-ante durability: medium
Estimated access impact: meaningful expectation shift for future institutional wrapper access
Estimated float impact: none immediate
Ex-ante pre-run risk: high
Unknown fields: final SEC disposition within 2025
Uncertainty notes: This is a filing-phase event only.
Final inclusion status: medium
Exclusion or downgrade reason: medium: official filing phase only; no approval or effective access

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_XRP_2025_05_08_SEC_RIPPLE_SETTLEMENT
Original collector event IDs: C2_XRP_2025_05_08_SEC_RIPPLE_SETTLEMENT
Catalyst cluster ID: XRP_SEC_AND_US_ACCESS_2020_2025
Asset ID: xrp
Ticker: XRP
Known major perp symbols: XRPUSDT; XRPUSD; XRP-PERP
Mechanism family: legal_regulatory_repricing
Mechanism subtype: sec_settlement_framework_with_pending_appeals
Direction: mixed
Event state: announced_conditional_settlement_framework
First public timestamp UTC: 2025-05-08
Official confirmation timestamp UTC: 2025-05-08
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: medium
Primary source type: SEC litigation release
Primary source URL: https://www.sec.gov/enforcement-litigation/litigation-releases/lr-26306
Supporting source URLs: https://www.sec.gov/newsroom/speeches-statements/crenshaw-statement-ripple-050825
Source publication timestamp: 2025-05-08
Raw official headline or title: SEC Announces Settlement Agreement to Resolve Civil Enforcement Action Against Ripple and Two of Its Executives
Mechanism classification: Formal settlement framework reduced litigation overhang but did not simply erase all prior legal findings, making the ex-ante mechanism two-sided rather than purely expansive.
Ex-ante durability: medium
Estimated access impact: meaningful overhang reduction, but legal settlement structure still conditional
Estimated float impact: none direct
Ex-ante pre-run risk: high
Unknown fields: final court approval and effectuation; appeal-dismissal timing; operative settlement date
Uncertainty notes: The SEC release announced a proposed framework and procedural next steps. It was not a fully effective final disposition on 2025-05-08.
Final inclusion status: medium
Exclusion or downgrade reason: downgraded: settlement framework was announced but not operative on the publication date

## 9. Excluded and duplicate event register

Exact duplicates are represented by the absorption table in Section 5 and by the combined `Original collector event IDs` field in the retained high-confidence records. They are not repeated as independent events here.

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EX_001
Original collector event IDs: C2-EX-001
Catalyst cluster ID: TON_OKX_AND_TELEGRAM_ACCESS_2021_2023
Asset ID: ton
Ticker: TON
Known major perp symbols: TONUSDT; TONUSDT-SWAP
Mechanism family: leverage_access
Mechanism subtype: post_migration_margin_and_perp_enablement_on_same_venue
Direction: mixed
Event state: announced
First public timestamp UTC: 2022-12-26
Official confirmation timestamp UTC: 2022-12-26
Effective timestamp UTC: 2022-12-28 07:00
Timestamp precision: minute
Source confidence: excluded
Primary source type: official exchange announcement
Primary source URL: https://www.okx.com/help/okx-to-enable-margin-trading-savings-and-list-perpetual-for-ton
Supporting source URLs: https://www.okx.com/help/okx-to-list-usdt-margined-perpetual-for-toncoin
Source publication timestamp: 2022-12-26
Raw official headline or title: OKX to Enable Margin Trading & Savings and List Perpetual for TON
Mechanism classification: same-venue continuation after token migration / rename
Ex-ante durability: high
Estimated access impact: medium
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: none material
Uncertainty notes: Material access already existed on OKX through the earlier TONCOIN/USDT perpetual. The 2022 notice mainly reflects post-migration continuation plus spot margin/savings.
Final inclusion status: excluded
Exclusion or downgrade reason: semantic duplicate: same-venue TON leverage access already existed; the later notice chiefly continued access after migration/rename

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EXCL_2023_02_09_KRAKEN_STAKING_SETTLEMENT
Original collector event IDs: C2_EXCL_2023_02_09_KRAKEN_STAKING_SETTLEMENT
Catalyst cluster ID: U.S._STAKING_ENFORCEMENT_2023
Asset ID: proof_of_stake_basket
Ticker: ETH/ADA/SOL/others
Known major perp symbols: ETHUSDT; ADAUSDT; SOLUSDT
Mechanism family: legal_regulatory_repricing
Mechanism subtype: platform_staking_settlement_not_clean_asset_specific
Direction: mixed
Event state: settled
First public timestamp UTC: 2023-02-09
Official confirmation timestamp UTC: 2023-02-09
Effective timestamp UTC: 2023-02-09
Timestamp precision: date_only
Source confidence: excluded
Primary source type: SEC press release
Primary source URL: https://www.sec.gov/newsroom/press-releases/2023-25
Supporting source URLs: none
Source publication timestamp: 2023-02-09
Raw official headline or title: Kraken to Discontinue Unregistered Offer and Sale of Crypto Asset Staking-As- A-Service Program and Pay $30 Million to Settle SEC Charges
Mechanism classification: Material for staking-service access, but the source does not give a sufficiently clean asset-by-asset mapping for this database’s asset-level catalyst framework.
Ex-ante durability: medium
Estimated access impact: real platform-service restriction
Estimated float impact: uncertain
Ex-ante pre-run risk: medium
Unknown fields: exact asset coverage inside the discontinued service
Uncertainty notes: A separate staking-service dataset could legitimately include this.
Final inclusion status: excluded
Exclusion or downgrade reason: excluded: material platform-level restriction, but the official source does not provide a deterministic asset-level basket

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EXCL_2023_06_06_SEC_COINBASE_OVERLAP_SOL_ADA_MATIC_FIL
Original collector event IDs: C2_EXCL_2023_06_06_SEC_COINBASE_OVERLAP_SOL_ADA_MATIC_FIL
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: multi_asset_overlap
Ticker: SOL/ADA/MATIC/FIL
Known major perp symbols: SOLUSDT; ADAUSDT; MATICUSDT; FILUSDT
Mechanism family: legal_regulatory_repricing
Mechanism subtype: duplicate_follow_on_sec_complaint
Direction: short
Event state: announced
First public timestamp UTC: 2023-06-06
Official confirmation timestamp UTC: 2023-06-06
Effective timestamp UTC: 2023-06-06
Timestamp precision: date_only
Source confidence: excluded
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-102.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-102 ; https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-101.pdf
Source publication timestamp: 2023-06-06
Raw official headline or title: SEC Charges Coinbase for Operating as an Unregistered Securities Exchange, Broker, and Clearing Agency
Mechanism classification: Real enforcement action, but duplicative for assets already captured one day earlier in the Binance complaint.
Ex-ante durability: medium
Estimated access impact: additive but not first-order for the overlapping assets
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: none material
Uncertainty notes: I retained Coinbase-only additions like ICP and NEAR, but excluded SOL/ADA/MATIC/FIL overlap as duplicate phase noise.
Final inclusion status: excluded
Exclusion or downgrade reason: semantic duplicate: the same assets had already been formally labeled in the Binance complaint one day earlier; retain only as a cluster-supporting source

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_OP_2023_2025_PROTOCOL_REVENUE_ALLOCATION
Original collector event IDs: OP-2023-2025-PROTOCOL-REVENUE-ALLOCATION
Catalyst cluster ID: OP_REVENUE_LINKAGE_DISCUSSIONS
Asset ID: optimism
Ticker: OP
Known major perp symbols: OPUSDT; OPUSD; OP-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: revenue_allocation_process_discussion
Direction: long
Event state: governance_process_discussion
First public timestamp UTC: 2023-07-19
Official confirmation timestamp UTC: 2023-07-19
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: excluded
Primary source type: official governance forum
Primary source URL: https://gov.optimism.io/t/the-future-of-optimism-governance/6471
Supporting source URLs: https://gov.optimism.io/t/allow-the-optimism-foundation-to-stake-a-portion-of-sequencer-eth-through-season-8/9710
Source publication timestamp: 2023-07-19
Raw official headline or title: “The Future of Optimism Governance.”
Mechanism classification: protocol revenue allocation discussions existed, but this pass did not retrieve an executed OP token revenue-capture change within the research window.
Ex-ante durability: unknown
Estimated access impact: unknown
Estimated float impact: unknown
Ex-ante pre-run risk: medium
Unknown fields: execution state and timestamp
Uncertainty notes: the strongest explicit OP buyback proposal retrieved was dated 2026 and therefore out of scope.
Final inclusion status: excluded
Exclusion or downgrade reason: excluded: no verified in-window OP-token revenue-capture execution

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EXCL_2023_11_20_SEC_KRAKEN_OVERLAP
Original collector event IDs: C2_EXCL_2023_11_20_SEC_KRAKEN_OVERLAP
Catalyst cluster ID: SEC_ASSET_SECURITY_LABELING_2023
Asset ID: overlap_basket
Ticker: ADA/ALGO/ATOM/FIL/FLOW/ICP/MATIC/NEAR/SOL
Known major perp symbols: ADAUSDT; ATOMUSDT; FILUSDT; ICPUSDT; MATICUSDT; NEARUSDT; SOLUSDT
Mechanism family: legal_regulatory_repricing
Mechanism subtype: duplicate_follow_on_sec_complaint
Direction: short
Event state: announced
First public timestamp UTC: 2023-11-20
Official confirmation timestamp UTC: 2023-11-20
Effective timestamp UTC: 2023-11-20
Timestamp precision: date_only
Source confidence: excluded
Primary source type: SEC complaint and press release
Primary source URL: https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-237.pdf
Supporting source URLs: https://www.sec.gov/newsroom/press-releases/2023-237
Source publication timestamp: 2023-11-20
Raw official headline or title: SEC Charges Kraken for Operating as an Unregistered Securities Exchange, Broker, Dealer, and Clearing Agency
Mechanism classification: Important for venue risk, but mostly a later duplicate of the 2023 SEC asset- security labeling cluster already captured for individual assets.
Ex-ante durability: medium
Estimated access impact: venue-specific and additive
Estimated float impact: none direct
Ex-ante pre-run risk: low
Unknown fields: none material
Uncertainty notes: The Kraken complaint does add venue-specific risk, but for database compactness I treat it as duplicate phase unless an asset first appears there.
Final inclusion status: excluded
Exclusion or downgrade reason: semantic duplicate at asset level: later venue-specific complaint repeats already recorded SEC security labeling for the listed assets

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EXCL_2024_03_11_FCA_PRO_INVESTOR_CETN_ACCESS
Original collector event IDs: C2_EXCL_2024_03_11_FCA_PRO_INVESTOR_CETN_ACCESS
Catalyst cluster ID: UK_CRYPTO_ETN_POLICY_2024_2025
Asset ID: broad_crypto_basket
Ticker: BTC/ETH/multi
Known major perp symbols: BTCUSDT; ETHUSDT
Mechanism family: exchange_spot_access
Mechanism subtype: generic_market_segment_rule_change
Direction: long
Event state: announced
First public timestamp UTC: 2024-03-11
Official confirmation timestamp UTC: 2024-03-11
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: excluded
Primary source type: FCA press release
Primary source URL: https://www.fca.org.uk/news/press-releases/fca-lift-ban-crypto-exchange-traded-notes
Supporting source URLs: https://www.fca.org.uk/news/press-releases/fca-opens-retail-access-crypto-etns
Source publication timestamp: 2025-06-06 / 2025-08-01 on the retrieved pages, both referencing the 2024 professional-investor segment change
Raw official headline or title: FCA to lift ban on crypto ETNs to support UK growth and competitiveness
Mechanism classification: Broad UK market-access rule change rather than an identifiable asset-specific catalyst.
Ex-ante durability: medium
Estimated access impact: broad venue-access change
Estimated float impact: none direct
Ex-ante pre-run risk: medium
Unknown fields: asset-specific scope
Uncertainty notes: Important policy context, but too generic for this asset-level catalyst database.
Final inclusion status: excluded
Exclusion or downgrade reason: excluded: cited primary URL is a 2025 retail-access proposal/announcement and does not independently verify the recorded 2024 professional-investor event at the stated timestamp

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_AAVE_2024_AAVENOMICS_TEMP_CHECK
Original collector event IDs: AAVE-2024-AAVENOMICS-TEMP-CHECK
Catalyst cluster ID: AAVE_REVENUE_SWITCH_PATH
Asset ID: aave
Ticker: AAVE
Known major perp symbols: AAVEUSDT; AAVEUSD; AAVE-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: fee_switch_and_buyback_roadmap
Direction: long
Event state: temp_check_and_follow_on_rfc
First public timestamp UTC: 2024-07-25
Official confirmation timestamp UTC: 2024-07-25
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: excluded
Primary source type: official governance forum
Primary source URL: https://governance.aave.com/t/temp-check-aavenomics-update/18379
Supporting source URLs: none
Source publication timestamp: 2024-07-25 / 2025-03-04
Raw official headline or title: “[TEMP CHECK] AAVEnomics update.”
Mechanism classification: would have created direct fee-switch and buyback linkages if fully implemented.
Ex-ante durability: potentially permanent
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: in-period execution timestamp
Uncertainty notes: the pass retrieved roadmap and implementation discussion, but not a verified primary in-period execution artifact.
Final inclusion status: excluded
Exclusion or downgrade reason: excluded: governance roadmap and temp-check only; no verified in-window execution

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_MC_003
Original collector event IDs: C2-MC-003
Catalyst cluster ID: PAYPAL_VENMO_CRYPTO_DISTRIBUTION_2020_2025
Asset ID: supported_tokens_unspecified_business_accounts
Ticker: unknown
Known major perp symbols: unknown
Mechanism family: distribution_integration
Mechanism subtype: business_account_crypto_enablement_paypal
Direction: mixed
Event state: launched
First public timestamp UTC: 2024-09-25
Official confirmation timestamp UTC: 2024-09-25
Effective timestamp UTC: 2024-09-25
Timestamp precision: date_only
Source confidence: excluded
Primary source type: official corporate press release
Primary source URL: https://newsroom.paypal-corp.com/2024-09-25-PayPal-Enables-Business-Accounts-to-Buy%2C-Hold-and-Sell-Cryptocurrency
Supporting source URLs: none
Source publication timestamp: 2024-09-25
Raw official headline or title: PayPal Enables Business Accounts to Buy, Hold and Sell Cryptocurrency
Mechanism classification: merchant-side crypto wallet and on-chain transfer enablement
Ex-ante durability: high
Estimated access impact: high
Estimated float impact: low
Ex-ante pre-run risk: low
Unknown fields: asset-level supported token basket at launch
Uncertainty notes: The official business-account release verifies a generic product change but does not enumerate an asset-level basket suitable for this database.
Final inclusion status: excluded
Exclusion or downgrade reason: excluded: official source does not enumerate an asset-level token basket, preventing deterministic asset mapping

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EXCL_2025_ALT_ETF_EXTENSIONS_ROUTINE
Original collector event IDs: C2_EXCL_2025_ALT_ETF_EXTENSIONS_ROUTINE
Catalyst cluster ID: ALTCOIN_US_SPOT_ETF_FILINGS_2025
Asset ID: multi_asset_alt_etf_wave
Ticker: SOL/XRP/LTC/DOGE/AVAX/etc.
Known major perp symbols: SOLUSDT; XRPUSDT; LTCUSDT; DOGEUSDT; AVAXUSDT
Mechanism family: regulated_institutional_access
Mechanism subtype: routine_extension_or_designation_order
Direction: unknown
Event state: announced
First public timestamp UTC: 2025-03-11
Official confirmation timestamp UTC: 2025-11-24
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: excluded
Primary source type: SEC extension/designation orders
Primary source URL: https://www.sec.gov/files/rules/sro/nasdaq/2025/34-102606.pdf
Supporting source URLs: https://www.sec.gov/files/rules/sro/nysearca/2025/34-103553.pdf ; https://www.sec.gov/files/rules/sro/nysearca/2025/34-103914.pdf ; https://www.sec.gov/files/rules/sro/nysearca/2025/34-104243.pdf
Source publication timestamp: 2025-03-11 through 2025-11-24
Raw official headline or title: Notice of Designation of a Longer Period for Commission Action on a Proposed Rule Change to List and Trade Shares of the CoinShares Litecoin ETF under Nasdaq Rule 5711(d)
Mechanism classification: Procedural continuation of already-public filing processes without a clean, new asset-level information shock.
Ex-ante durability: low
Estimated access impact: mostly procedural
Estimated float impact: none
Ex-ante pre-run risk: low
Unknown fields: none material
Uncertainty notes: These would clutter downstream testing without adding distinct informational phases.
Final inclusion status: excluded
Exclusion or downgrade reason: semantic duplicate/procedural continuation: routine deadline extensions add no distinct access state beyond the initial filings

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_UNI_2024_2025_FEE_SWITCH_MANDATE
Original collector event IDs: UNI-2024-2025-FEE-SWITCH-MANDATE
Catalyst cluster ID: UNI_FEE_SWITCH_ATTEMPTS
Asset ID: uniswap
Ticker: UNI
Known major perp symbols: UNIUSDT; UNIUSD; UNI-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: protocol_fee_switch_discussion_and_rfc
Direction: long
Event state: proposal_phase_only_during_period
First public timestamp UTC: 2025-05-10
Official confirmation timestamp UTC: 2025-05-10
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: excluded
Primary source type: official governance forum
Primary source URL: https://gov.uniswap.org/t/rfc-enable-0-05-protocol-fee-on-all-uniswap-v3-pools-for-one-month-experiment/25589
Supporting source URLs: https://gov.uniswap.org/t/fee-switch-design-space-next-steps/17132
Source publication timestamp: 2025-05-10 for RFC
Raw official headline or title: “[RFC] Enable 0.05% Protocol Fee on All Uniswap v3 Pools...”
Mechanism classification: direct fee capture for UNI would be economically meaningful if executed.
Ex-ante durability: potentially permanent if passed
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: execution timestamp
Uncertainty notes: within the 2020-2025 window, the retrieved evidence supports discussion and mandate- seeking, not verified execution.
Final inclusion status: excluded
Exclusion or downgrade reason: excluded: proposal/RFC only; no verified in-window execution

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_HYPE_2025_STAKING_REFERRAL_PROPOSAL
Original collector event IDs: HYPE-2025-STAKING-REFERRAL-PROPOSAL
Catalyst cluster ID: HYPE_FEE_LINKAGE_PROPOSALS
Asset ID: hyperliquid
Ticker: HYPE
Known major perp symbols: HYPEUSDT; HYPEUSD; HYPE-PERP
Mechanism family: protocol_utility_fee_revenue
Mechanism subtype: staking_referral_revenue_share_proposal
Direction: long
Event state: not_ready_for_mainnet
First public timestamp UTC: 2025-07-15
Official confirmation timestamp UTC: 2025-07-15
Effective timestamp UTC: unknown
Timestamp precision: date_only
Source confidence: excluded
Primary source type: official docs
Primary source URL: https://hyperliquid.gitbook.io/hyperliquid-docs/referrals/proposal-staking-referral-program
Supporting source URLs: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees
Source publication timestamp: 2025-07-15
Raw official headline or title: “Proposal: Staking referral program.”
Mechanism classification: the proposal would have linked HYPE staking tiers to referred-user trading fees.
Ex-ante durability: potentially long-lived
Estimated access impact: medium
Estimated float impact: low
Ex-ante pre-run risk: high
Unknown fields: execution timestamp
Uncertainty notes: the official page explicitly stated that no definitive implementation was ready for mainnet and that the design was being re-evaluated.
Final inclusion status: excluded
Exclusion or downgrade reason: excluded: official source explicitly states the design was not ready for mainnet

### AUDITED EVENT RECORD

Audited provisional event ID: AUD_C2_EXCL_BANKRUPTCY_BASKET_2022_2025
Original collector event IDs: C2_EXCL_BANKRUPTCY_BASKET_2022_2025
Catalyst cluster ID: CRYPTO_BANKRUPTCY_OUTCOMES_2022_2025
Asset ID: bankruptcy_related_basket
Ticker: FTT/SOL/BTC/ETH/creditor_claims
Known major perp symbols: SOLUSDT; BTCUSDT; ETHUSDT
Mechanism family: legal_regulatory_repricing
Mechanism subtype: bankruptcy_plan_or_settlement_without_clean_asset_level_mapping
Direction: mixed
Event state: unknown
First public timestamp UTC: unknown
Official confirmation timestamp UTC: unknown
Effective timestamp UTC: unknown
Timestamp precision: unknown
Source confidence: excluded
Primary source type: bankruptcy and estate sources not fully retrieved
Primary source URL: unknown
Supporting source URLs: none
Source publication timestamp: unknown
Raw official headline or title: unknown
Mechanism classification: Potentially relevant in some cases, but not cleanly reconstructable at asset level from the retrievable primary sources used here.
Ex-ante durability: medium
Estimated access impact: uncertain
Estimated float impact: uncertain
Ex-ante pre-run risk: medium
Unknown fields: specific estate orders; effective distributions; token-level consequences
Uncertainty notes: This is the main substantive omission area and would require targeted PACER/estate- page work.
Final inclusion status: excluded
Exclusion or downgrade reason: excluded: no event identity, primary URL, asset-level mechanism, or timestamp was verified

## 10. Source completeness matrix

| Source family | Period represented | Coverage assessment | Main limitation |
|---|---|---|---|
| SEC newsroom, litigation, complaints, and SRO orders | 2020-2025 | strong | Complaint and order support is strong; 2025 altcoin-ETF filing coverage is selective and process-heavy. |
| U.S. courts and GovInfo | 2022-2025 | strong for named cases | Ripple and Grayscale materials are grounded; PACER-only and poorly indexed dockets remain incomplete. |
| CFTC, CME, Cboe, and Coinbase Derivatives | 2021-2025 | strong mechanism; mixed live timing | Launch and self-certification documents are available; first-trade and customer-distribution timestamps are often absent. |
| Coinbase, Binance, OKX, Binance.US, and Bybit announcements | 2020-2025 | partial | Retained Coinbase/Binance/OKX records are strong; Bybit history and migrated Binance.US shells prevent a census. |
| PayPal, Venmo, and Robinhood | 2020-2025 | strong product evidence; partial rollout timing | Corporate releases support product changes, but cohort rollouts and complete launch baskets are not always fixed. |
| Protocol blogs, governance forums, and official docs | 2021-2025 | mixed | Mechanisms are generally supportable, but historical proposal/execution artifacts are inconsistent and 27 record fields required URL repair. |
| Token vesting, unlock, and float-schedule sources | 2023-2025 | weak-to-partial | Current schedules are often available without contemporaneous revision notices; vesting-heavy assets are undercovered. |
| Telegram/TON and messaging-native distribution sources | 2023 | weak | Indirect ecosystem reports and redirects do not preserve clean first-public and effective timestamps. |
| Non-U.S. regulators and ETP venues | 2021-2025 | partial | Canada examples are grounded; UK and other jurisdictions are not systematically covered. |
| Bankruptcy and estate sources | 2022-2025 | incomplete | No deterministic asset-level bankruptcy record survived the audit; docket retrieval remains a major gap. |

Eight source families remain materially incomplete for census-level use: PACER/bankruptcy dockets, non-U.S. regulators and ETP venues, Bybit historical announcements, Telegram-native distribution records, migrated Binance.US archives, protocol governance execution archives, historical vesting/unlock revisions, and fragile official social-post listing histories.


## 11. Concentration analysis

Mechanism counts:

| Normalized mechanism family | All logical records | Included records |
|---|---:|---:|
| distribution_integration | 12 | 11 |
| exchange_spot_access | 14 | 13 |
| legal_regulatory_repricing | 15 | 11 |
| leverage_access | 17 | 16 |
| protocol_utility_fee_revenue | 20 | 16 |
| regulated_institutional_access | 13 | 12 |
| supply_float | 7 | 7 |

Asset counts below cover the 86 included records. Multi-asset records are expanded into named asset exposures, so the table sums to 113 rather than 86. The Robinhood Europe record is counted only for the four assets explicitly named in its primary release; the unspecified remainder of its 20-plus basket is not inferred.

| Asset | Included event exposures | Share of 113 named exposures |
|---|---:|---:|
| ethereum | 17 | 15.0% |
| bitcoin | 15 | 13.3% |
| solana | 12 | 10.6% |
| xrp | 11 | 9.7% |
| litecoin | 6 | 5.3% |
| bitcoin_cash | 5 | 4.4% |
| pepe | 5 | 4.4% |
| cardano | 3 | 2.7% |
| polygon | 3 | 2.7% |
| chainlink | 3 | 2.7% |
| ton | 3 | 2.7% |
| shiba_inu | 3 | 2.7% |
| dogecoin | 2 | 1.8% |
| bnb | 2 | 1.8% |
| pancakeswap | 2 | 1.8% |
| starknet | 2 | 1.8% |
| arbitrum | 2 | 1.8% |
| sushi | 2 | 1.8% |
| sui | 2 | 1.8% |
| filecoin | 1 | 0.9% |
| cosmos | 1 | 0.9% |
| internet_computer | 1 | 0.9% |
| near_protocol | 1 | 0.9% |
| avalanche | 1 | 0.9% |
| maker | 1 | 0.9% |
| dydx | 1 | 0.9% |
| injective | 1 | 0.9% |
| render | 1 | 0.9% |
| apecoin | 1 | 0.9% |
| compound | 1 | 0.9% |
| usd_coin | 1 | 0.9% |
| bonk | 1 | 0.9% |

Ethereum, bitcoin, solana, and XRP account for 55 of 113 named included exposures, or 48.7%. The ten most frequently represented assets account for 70.8%. Thirteen of the 32 named assets appear only once. This concentration reflects source availability and repeated multi-phase clusters, not a balanced asset-universe design.


## 12. Unresolved conflicts

Unresolved conflict groups: 18.

1. Coinbase FCM approval versus live customer access: BTC and ETH approval is fixed at 2023-08-16; asset-specific customer availability is not.
2. Coinbase Derivatives SOL first live trade: The self-certification fixes an on-or-after date, not the first executed contract or partner-platform access.
3. Coinbase Derivatives XRP first live trade: The self-certification and later product lineup do not establish a deterministic 2025-04-21 first-trade timestamp.
4. U.S. spot-ETH cross-venue launch timing: A listing date is supported; one contemporaneous primary source does not fix registration effectiveness and first trades across all venues.
5. Ripple settlement effectuation: The 2025-05-08 release describes a conditional framework; final operative dates remain unresolved.
6. CAKE Tokenomics 3.0 execution: Approval is supportable, but the exact on-chain implementation timestamp was not verified.
7. dYdX buyback execution: The official page’s publication and update dates do not identify the first executed purchase.
8. Starknet staking launch artifact: The vote/framework is clearer than the live activation record cited by the collector.
9. Sushi Kanpai 2022 source chain: Only retrospective official material was retrieved; original proposal and execution timestamps remain missing.
10. Solana SIMD-0096 activation: Approval is supported; mainnet feature-gate activation is not.
11. Starknet unlock-schedule revision: Current official docs show the schedule, while the contemporaneous official revision notice remains missing.
12. ApeCoin staking launch chronology: The current portal documents the program but not the original AIP and exact launch timestamp.
13. TON Space and Telegram distribution chronology: The ecosystem report is indirect and the direct report URL redirects; first public and live dates remain unresolved.
14. PayPal LINK/SOL rollout: The announcement describes a multi-week rollout without a single effective timestamp.
15. Official social-post listing chronology: Coinbase XRP re-enablement and BONK listing are supportable at date level, but exact state completion and archive stability remain weak.
16. Migrated Binance.US article shells: XRP relisting and USD-pair removal contain historical dates inside pages republished under 2025 shells.
17. PayPal UK rollout: Publication identifies the rollout week, not the first or complete eligible-user availability.
18. Asset-identity migration timing: MATIC/POL, RNDR/RENDER, MKR/SKY, TONCOIN/TON, and DYDX representations require venue-specific symbol-transition tables outside these reports.


## 13. Suitability for deterministic ingestion

Final suitability decision: `suitable_for_sample_limited_ingestion_only`

Collection characterization: `only a source-verified seed database`

The 59 high-confidence records can be ingested after enforcing three parser rules: preserve `unknown` values as null rather than imputing times; use catalyst-cluster IDs to prevent phase double counting; and treat basket records as multi-asset events rather than duplicating a timestamp without an explicit basket-expansion rule.

The 27 medium-confidence records require either a source-repair flag or exclusion from strict timestamp-sensitive runs. The 12 excluded records should remain in an audit table only. The full 98-record collection should not be treated as a complete event census because coverage was not closed across asset, venue, jurisdiction, year, and source family.
