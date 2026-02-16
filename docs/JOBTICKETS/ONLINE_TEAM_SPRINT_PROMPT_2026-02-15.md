# Online Team Sprint Prompt (JT-002, JT-003, JT-004)

Copy-paste message to online team:

```text
Team, please execute the following sprint so we can close three open backlog tickets:
- JT-002 (warmup-aware schema classification) [P0]
- JT-003 (full live sizing parity with backtester) [P0]
- JT-004 (decision reason taxonomy normalization) [P1]

Hard constraint:
- Do NOT inject synthetic/default feature values to bypass strict schema.
- Keep strict schema behavior fail-closed (schema_ok=false, reason=schema_fail).

================================================================
JT-002: Warmup-aware schema-fail classification from listing age
================================================================
Goal:
Use objective symbol age in export context to separate expected warmup fails from integration defects.

Required implementation:
1) Add symbol age fields:
   - run_context.json:
     - symbol_age_reference_ts_utc
     - symbol_age_source
     - warmup_days_required (default 20 unless configured)
   - live_decisions.csv (or companion diagnostics):
     - symbol_listed_at_utc (if available)
     - symbol_age_days
2) In schema_diagnostics.json, include:
   - warmup_expected_count
   - integration_defect_count
   - warmup_expected_symbols (top list)
   - integration_defect_symbols (top list)
   - classification_rule_version
3) Classification rule must explicitly use symbol age (not only missing-field names).

Acceptance (for closure):
1) Daily report clearly separates warmup_expected vs integration_defect.
2) Repeated non-warmup failures still trigger incident_recommended=true.
3) Evidence package includes one sample export showing symbol_age_days populated.

================================================================
JT-003: Full live sizing parity with backtester mechanics
================================================================
Goal:
Match live sizing chain to current offline backtester mechanics (not approximate).

Target mechanics (must match):
1) Regime-down base sizing:
   - if (regime_up==0) and (REGIME_BLOCK_WHEN_DOWN==False):
     equity_for_sizing = equity * REGIME_SIZE_WHEN_DOWN
2) risk_on state:
   - risk_on = 1[(regime_up==1) AND (btc_trend_up==1) AND (btc_vol_high==0)]
3) size_mult source:
   - if signal.risk_scale exists and is finite: size_mult = risk_scale
   - else:
     - base from p using linear P0/P1 map:
       if p<=META_SIZING_P0 => META_SIZING_MIN
       if p>=META_SIZING_P1 => META_SIZING_MAX
       else linear interpolation
     - regime multiplier:
       if eth_macd_hist_4h < DYN_MACD_HIST_THRESH => * REGIME_DOWNSIZE_MULT
     - clamp by SIZE_MIN_CAP/SIZE_MAX_CAP
4) Risk-off probe cap:
   - if risk_on==0: size_mult = min(size_mult, RISK_OFF_PROBE_MULT)
5) Risk mode overrides:
   - percent mode:
     risk_pct_override = RISK_PCT * size_mult
     fixed_cash unchanged
   - cash mode:
     fixed_cash_override = FIXED_RISK_CASH * size_mult
     risk_pct unchanged
6) risk_cash_target:
   - percent: equity_for_sizing * risk_pct_override
   - cash: fixed_cash_override
7) Final qty parity:
   - qty = risk_cash / abs(entry-stop)
   - apply NOTIONAL_CAP_PCT_OF_EQUITY * MAX_LEVERAGE cap exactly as backtester
8) Remove hidden alternate sizing path/fallback that bypasses this chain.

Acceptance (for closure):
1) Golden sizing parity test (>=100 fixture rows) passes:
   - exact match or float tolerance agreed in writing
   - size_mult, risk_cash_target, qty all matched
2) Live logs include:
   - risk_on, size_mult, risk_cash_target, risk_usd/final qty, and decision reason
3) No fallback sizing path remains active in production code.

================================================================
JT-004: Decision reason taxonomy normalization
================================================================
Goal:
Stabilize reason-level parity by normalizing live and backtest reasons to canonical codes.

Required implementation:
1) Add canonical reason map (versioned artifact), for example:
   - ok
   - below_pstar
   - schema_fail
   - scope_fail
   - regime_down
   - regime_slope_down
   - atr_invalid
   - atr_too_small
   - no_data
   - throughput_dedup
   - throughput_in_position
   - throughput_cooldown
   - throughput_daycap
   - max_open_positions
   - schedule_block
   - simulation_none
2) Export both:
   - reason_raw
   - reason_norm
   - reason_map_version
3) Keep backward compatibility for existing reason strings.

Acceptance (for closure):
1) Reason agreement metric in autopar uses normalized reasons and is stable across runs.
2) Mapping is explicit, versioned, and included in package metadata/docs.

================================================================
Evidence required back to offline team
================================================================
Please return:
1) Commit hash(es) and file list.
2) One sample autopar export path showing new fields/artifacts.
3) Parity test outputs:
   - JT-002 sample schema_diagnostics snippet with symbol_age_days.
   - JT-003 sizing parity summary (pass/fail + mismatch stats).
   - JT-004 reason map and normalized reason distribution.
4) Confirmation that no synthetic/default features were added to force schema pass.
```

