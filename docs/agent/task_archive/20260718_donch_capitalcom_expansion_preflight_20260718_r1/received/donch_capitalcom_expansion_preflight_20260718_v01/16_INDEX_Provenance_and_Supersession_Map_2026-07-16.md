---
status: current provenance and supersession index
date: 2026-07-16
revision: 1.0
scope: curated core/optional source lineage, original-source disposition, conflicts, and offline retention
authority: 00_audit inventory, duplicate map, authority map, and reconciled source synthesis
supersedes: filename freshness guesses and undocumented source replacement
provenance: SOURCE_INVENTORY.csv; DUPLICATE_AND_NEAR_DUPLICATE_MAP.csv; AUTHORITY_AND_SUPERSESSION_MAP.csv; all 44 non-review sources
known limitations: exact hashes and original paths remain in 00_audit; external-review payload lineage is indexed separately under 01_review_package
---

# Provenance and Supersession Map

## Preservation policy

No original source is deleted by this curation. The curated set uses Markdown and CSV only. Original PDFs stay local; page-bounded text extractions support synthesis but are not replacements for visual source retention. Exact identity, byte size, timestamps, and SHA-256 are in `00_audit/SOURCE_INVENTORY.csv`.

The word `supersedes` means “replaces for active planning within the stated scope.” It does not erase historical provenance.

## Canonical-output lineage

| Curated output | Main source lineage | Authority decision |
|---|---|---|
| `00_READ_FIRST_Project_Source_Map.md` | Master prompt; inventory and authority maps; full curation | Navigation only. |
| `01_AUTHORITY_QLMG_Operating_Contract_2026-07-16.md` | Master prompt; interim reassessment; rev7; backtester and capture audits | Replaces Bybit-primary and mixed-period active guidance. |
| `02_STATE_Master_Continuity_Brief_2026-07-16_rev8.md` | rev7 plus all July 16 audits and reconciled package/capture facts | Current narrative continuity below machine evidence. |
| `03_STATE_Current_Research_Decisions_2026-07-16.md` | Interim reassessment; strategic audit; rev7; 14-family lineage summary | Replaces stale cross-family prose, not machine family decisions. |
| `04_REGISTRY_Hypothesis_and_Family_Status_2026-07-16.csv` | Safe workbook sheets plus family manifests/review registry and July research | Built separately; `Current Results` excluded. |
| `05_REGISTRY_Kraken_Data_and_Evidence_Capability_2026-07-16.csv` | backtesterreport.md; Captureforwardreport.md | Acquired-data audit outranks PDF endpoint descriptions. |
| `06_CAPTURE_Forward_Capture_State_and_Runbook_2026-07-16.md` | Captureforwardreport.md; interim reassessment | Replaces implied continuity or restart readiness. |
| `07_AUDIT_Backtester_and_Evidence_Readiness_2026-07-16.md` | backtesterreport.md; package manifests and preflight | Package remains protocol-blocked. |
| `08_METHOD_Catalyst_Research_and_Audited_Register_2026-07-16.md` | catdb.md v2.1; catalyst PDF extractions | v2.1 audited counts and exclusions outrank earlier database totals. |
| `09_METHOD_Point_in_Time_Sector_and_Regime_Context_2026-07-16.md` | PIT sector and regime extracts; data audit | Seed and method only; no current taxonomy backfill. |
| `10_RESEARCH_External_and_Forum_Synthesis_2026-07-16.md` | eight-file forum package; Kraken mechanism reports | Untested priors and negative evidence only. |
| `11_HISTORY_Kraken_Venue_and_Data_2026-07-16.md` | Kraken reports; MOV provenance; backtester acquisition audit | Distinguishes described capability from acquired history. |
| `12_MANUAL_Test_and_Evidence_Standards_2026-07-16.md` | testmanual evidence sections; current audits; method extracts | Retains QA rules, replaces venue/period sections. |
| `13_GUIDE_Backtest_Claims_and_Review_2026-07-16.md` | evidence manual; review protocol findings | Replaces ambiguous promotion and pass-like wording. |
| `14_GUIDE_Protected_Capture_Calibration_2026-07-16.md` | interim protected policy; capture audit | July 2026 is calibration-only. |
| `15_RUNBOOK_Human_Approval_and_Orchestration_2026-07-16.md` | master prompt; access report; audits | Remote prompts must be self-contained. |
| Optional files | Research PDFs, forum sources, and legacy reports | Priors/context only; cannot override core or machine evidence. |

## Resolved authority conflicts

| Subject | Lower authority | Higher authority | Resolution |
|---|---|---|---|
| Active venue | `testmanual(1).txt`: Bybit primary | Master prompt and interim reassessment: Kraken only | Replace venue language; retain sound evidence policy. |
| Rankable period | Broad historical reports, including pre-2023 and mid-2026 | Current contract: 2023-2025 rankable; 2026+ protected | Broader dates are priors or calibration only. |
| Hypothesis status | July 1 workbook is stale and incomplete | Family manifests, current roots, July 16 decisions | Merge safe metadata into the refreshed registry; exclude `Current Results`. |
| Continuity revision | rev7 | Reconciled rev8 | Rev8 is current narrative; rev7 retained. |
| Review archive identity | Core embeds full manifest but omits 56 listed Parquets | Full archive physically covers 386 manifest rows | Full is local payload authority; text-only is compact reading copy. |
| Review release status | Pass-like matrices and zero recomputation mismatch | Decision summary and verification note | Preserve `blocked_by_protocol_issue` and `release_ready=false`. |
| Capture readiness | Configuration and old migration language may imply operation | Current capture audit | Stopped, capture-only, not restart-ready. |
| Catalyst totals | Original 90 main plus 10 excluded database | catdb.md v2.1 adversarial audit | 98 logical: 59 high, 27 medium, 12 excluded. |
| Kraken data availability | Research PDFs describe endpoints | Backtesting acquisition audit | “Endpoint described” does not mean “history acquired.” |
| MOV venue material | Migration and contingency guidance | Current Kraken-only contract | Offline venue-migration provenance only. |

## Original non-review sources: current narrative and registry files

| Original | Disposition |
|---|---|
| `QLMG_Project_Master_Continuity_Brief_2026-07-16_rev7.md` | Preserved; superseded for current planning by rev8. |
| `QLMG_Interim_Reassessment_Agent_Questions_and_Infrastructure_Prompt_2026-07-16(1).md` | Current correction source; split into contract, decisions, data, and protected guidance. |
| `QLMG_Research_Audit_and_Alpha_Roadmap_2026-07-16(1).md` | Current strategic audit with some suggestions superseded by the interim Kraken-only/2023-2025 correction. |
| `Captureforwardreport.md` | Current capture audit authority. |
| `backtesterreport.md` | Current backtester data/package audit authority. |
| `catdb.md` | Current catalyst audit authority. |
| `testmanual(1).txt` | Superseded as active venue manual; retained for QA/no-vendor/evidence provenance. |
| `QLMG_Hypothesis_Library_2026-07-01(1).xlsx` | Safe reference sheets only for merged registry; stale rows require repair; `Current Results` excluded. |

## Original non-review sources: forum package

All eight remain intact under `research/kraken_public_forum_alpha_scout_20260716_v1`:

| Original | Curated use |
|---|---|
| `SOURCE_LEDGER.csv` | Public-source identity and credibility provenance. |
| `RAW_CLAIM_INVENTORY.csv` | Claim, contradiction, and status provenance. |
| `SYNTHESIZED_HYPOTHESIS_REGISTER.csv` | Twelve untested cards and qualitative routes. |
| `FINAL_REPORT.md` | Main public-source synthesis. |
| `REJECTED_MYTHS_AND_GRIFTS.md` | Negative evidence and archive rules. |
| `CONTEXT_AND_INTERACTION_MAP.md` | Interaction and context priors. |
| `RESEARCH_GAPS_AND_CAPTURE_NEEDS.md` | Data and prospective capture routing. |
| `ARTIFACT_MANIFEST.json` | Package hashes; seven payload identities reported as passing. |

## Original non-review sources: PDF research reports

All 28 PDFs remain local. Their text was extracted with page boundaries and good reported extraction quality. No PDF is copied into the curated source bundle.

### QLMG and method priors

- `QLMG Cryptocurrency Momentum, Reversal, Perpetuals, Open Interest, and Liquidation Cascades-2.pdf`
- `QLMG Cryptocurrency Momentum, Reversal, Funding, Open Interest, and Liquidation Cascades-2.pdf`
- `QLMG Rulebook for Kristjan Qullamaggie Kullamägi Public Trading Setups-2.pdf`
- `QLMG Historical Crypto Perpetual Setups From 2020 Through Mid 2026-3.pdf`
- `QLMG Crypto Perpetual Futures vs Stocks for Systematic Strategy Testing-2.pdf`
- `QLMG Taxonomy of Crypto Catalysts Analogous to Stock Episodic Pivots-2.pdf`
- `QLMG Robust Backtest Protocol for Qullamaggie-Inspired Crypto Perpetual Strategies-2.pdf`
- `QLMG Simple Alpha Hypotheses for Bybit USDT Perpetuals-2.pdf`

The Bybit and broad-date content is prior/provenance only. Stable control, execution, and mechanism ideas were adapted under current Kraken rules.

### Venue migration provenance

- `MOV Contingency Venues for EU QLMG if Bybit Global Perps Become Unavailable-1.pdf`
- `MOV Kraken derivatives data availability and acquisition report for QLMG-1.pdf`
- `MOV Kraken Derivatives Versus Bybit for QLMG Perpetual Strategies-1.pdf`
- `MOV Kraken Live Trading Migration Blueprint for a Bybit USDT Perpetual Bot-1.pdf`
- `MOV EU Venue Access Report for Crypto Perpetual Futures-1.pdf`

These explain migration decisions. They are not active trading or live-system authorization.

### RES and catalyst/sector priors

- `RES Liquid Universe Continuation Strategies for Crypto Perpetual Futures-2.pdf`
- `RES Point-in-Time Regime Framework for Crypto Perpetual Futures-2.pdf`
- `RES Designing Less Depth-Sensitive Crypto Perp Tests-3.pdf`
- `RES Research Base for Liquid-Sector Episodic Pivots and Post-Catalyst Continuation-2.pdf`
- `RES Research Base for Liquid-Sector Episodic Pivots and Post-Catalyst Continuation-3.pdf`
- `Post-Catalyst Continuation Base Catalyst Database-2.pdf`
- `Point In Time Sector Seeds for Crypto Perpetual Theme Ignition-2.pdf`

The two `RES Research Base ...-2/-3.pdf` files are byte-identical by SHA-256. Both originals remain; one content lineage is cited.

### RES2 and Kraken-specific priors

- `RES2 Public-Data Crypto Perpetual Alpha Hypotheses-1.pdf`
- `RES2 Regime Map for Crypto Perpetual Alpha-1.pdf`
- `RES2 Crypto Perpetual Practitioner Alpha Map-1.pdf`
- `RES2 No-Vendor Roadmap for Testing Crypto Perpetual Alpha Hypotheses-1.pdf`
- `RES2 Crypto Perpetual Futures Alpha Hypothesis Catalogue-1.pdf`
- `Efficient Alpha Research for Python Systematic Trading-1.pdf`
- `Kraken Perpetual Futures on Kraken Through Mid-2026.pdf`
- `Kraken-Only Systematic Crypto-Perpetual Alpha Mechanisms.pdf`

These remain research priors. Kraken-specific endpoint descriptions are reconciled against the actual acquisition audit before entering the capability registry.

## External-review bundles

The full, core, reduced, and text-only extracted bundles stay local. The curated project set uses compact review outputs under `01_review_package`, not Parquet payloads or raw archives. The full package is the physical payload authority. The text-only package is the preferred reading copy. Core must not be described as byte-complete against its embedded full manifest.

## Protected workbook handling

The workbook's `Current Results` sheet is not a curation input. During generic schema discovery, a limited row view was unintentionally surfaced before the guard applied; values were discarded and not used. Safe reference sheets contain 152 reported library rows, but the final 18 are shifted/machine-unsafe and require explicit repair or exclusion in the merged registry.

## How to audit a curated claim

1. Find the canonical output in the first table.
2. Follow its source lineage to the original filename.
3. Use `00_audit/SOURCE_INVENTORY.csv` for path and SHA-256.
4. Use page-bounded text extractions for PDF prose and retain the original PDF for figures/tables.
5. Check `AUTHORITY_AND_SUPERSESSION_MAP.csv` for conflicts.
6. Check machine manifests and current run roots before accepting a narrative claim.

If the lineage cannot be reconstructed, mark the claim `unavailable` or `blocked`; do not fill the gap from memory.
