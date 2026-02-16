# Donch Autopar Online Sprint Prompt (JT-002/JT-003/JT-004)

Copy-paste this to the online team:

```
Team, please run this sprint to close the remaining parity tickets:
- JT-002 (warmup-aware schema-fail classification)
- JT-003 (full live sizing parity with backtester)
- JT-004 (decision reason taxonomy normalization)

Non-negotiable policy:
1) Keep strict schema behavior. No synthetic/default feature injection.
2) If feature data is not available at decision time, decision remains skipped with explicit reason.
3) Preserve one row per `(symbol, decision_ts)` in UTC.

==================================================
JT-002: Warmup-aware schema-fail classification
==================================================
Goal:
Separate expected young-symbol warmup misses from real integration defects.

Please implement:
1) Add objective listing-age fields:
   - `symbol_listed_at_utc` (preferred) OR `symbol_age_days`
   - source must be exchange instrument metadata (not inferred from missing candles).
2) Include listing-age context in export package:
   - either in `live_decisions.csv` per row, or in `symbol_metadata.csv` keyed by symbol.
3) Add/extend schema classification:
   - `schema_fail_class` in {`warmup_expected`, `integration_defect`, `unknown`}
4) Update `schema_diagnostics.json` to include:
   - counts by class
   - top missing fields by class
   - top affected symbols by class

Acceptance evidence to send back:
1) One sample package path with new fields present.
2) 20-row CSV sample showing `schema_fail_class` and listing-age fields.
3) `schema_diagnostics.json` snippet with per-class counts.

==================================================
JT-003: Full live sizing parity with backtester
==================================================
Goal:
Match offline backtester sizing chain exactly.

Required mechanics (must match offline semantics):
1) Regime-down equity sizing base:
   - `equity_for_sizing = equity`
   - if `(regime_up == 0) and (REGIME_BLOCK_WHEN_DOWN == false)`:
     `equity_for_sizing *= REGIME_SIZE_WHEN_DOWN`
2) Core size multiplier:
   - if signal has `risk_scale`, use it directly
   - else use dynamic multiplier:
     a) meta-prob segment from `META_SIZING_P0/P1/MIN/MAX`
     b) ETH hist downsize: if `eth_macd_hist_4h < DYN_MACD_HIST_THRESH`, multiply by `REGIME_DOWNSIZE_MULT`
     c) clamp to `[SIZE_MIN_CAP, SIZE_MAX_CAP]`
3) Risk-off probe cap:
   - compute `risk_on = 1[(regime_up==1) & (btc_trend_up==1) & (btc_vol_high==0)]`
   - if `risk_on == 0`: `size_mult = min(size_mult, RISK_OFF_PROBE_MULT)`
4) Risk mode conversion:
   - percent mode: `risk_pct_override = RISK_PCT * size_mult`
   - cash mode: `fixed_cash_override = FIXED_RISK_CASH * size_mult`
5) Qty and hard caps:
   - `risk_per_unit = abs(entry - sl)`
   - `qty = cash_risk / risk_per_unit`
   - max notional cap:
     `max_notional = equity_for_sizing * NOTIONAL_CAP_PCT_OF_EQUITY * MAX_LEVERAGE`
     and cap qty accordingly

Important:
- No hidden fallback threshold veto in order-open path when threshold is disabled.
- No alternate sizing branch bypassing this chain.

Acceptance evidence to send back:
1) Commit hash + file references for each step above.
2) 100-row parity fixture comparing live vs offline:
   - `size_mult_live`, `size_mult_bt`, abs error
   - `risk_usd_live`, `risk_usd_bt`, abs error
3) Summary stats:
   - mean abs error and p90 abs error for `size_mult` and `risk_usd`
4) 5 raw decision log lines showing all sizing fields:
   - `p_cal`, `risk_on`, `size_mult`, `risk_usd`, `qty`.

==================================================
JT-004: Decision reason taxonomy normalization
==================================================
Goal:
Make reason agreement metric stable across versions.

Please implement:
1) Canonical reason mapping in live export:
   - output both `reason_raw` and `reason_canonical`
2) Canonical set:
   - `ok`
   - `schema_fail`
   - `meta_prob`
   - `meta_scope`
   - `regime_down`
   - `regime_slope_down`
   - `atr_invalid`
   - `atr_too_small`
   - `dedup_entry`
   - `in_position`
   - `cooldown`
   - `daycap`
   - `max_open_positions`
   - `no_5m_data_after_signal`
   - `simulation_none`
   - `strategy_fail`
   - `internal_error`
   - fallback: `other`
3) Keep original raw reason unmodified for forensics.

Acceptance evidence to send back:
1) Canonical mapping table (reason_raw -> reason_canonical).
2) Sample export showing both columns.
3) One autopar run where reason-agreement is computed on canonical reason.

==================================================
Delivery format for closure
==================================================
Please reply in one message with:
1) Ticket-by-ticket status: JT-002 / JT-003 / JT-004 = done or blocked.
2) Commit hash(es).
3) Paths to produced package(s).
4) Acceptance evidence requested above.
5) Any blocked item with exact blocker + minimal workaround.
```

## Notes For Offline Team

Use this message to collect closure evidence in one round-trip.  
Do not mark a ticket done until the acceptance evidence is attached.
