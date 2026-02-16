# Donch Autopar Live Export Spec

This spec defines what the live team should export daily for parity checks.

## Package layout

Recommended package directory name:
- `autopar_YYYY-MM-DD/`

Required files:
1. `live_decisions.csv`
2. `live_trades.csv` (closed trades in same window)
3. `symbols_active.txt` (one symbol per line)
4. `run_context.json`

Optional files:
- `live.log` (or rotated log file that contains `META_DECISION` lines)
- `settings_snapshot.json`

## 1) `live_decisions.csv` schema

Required columns:
- `symbol` (string, uppercase)
- `decision_ts` (UTC timestamp; ISO-8601 preferred)
- `decision` (`taken` or `skipped`)
- `reason` (string)

Strongly recommended columns:
- `bundle` (bundle id/hash)
- `schema_ok` (bool)
- `p_cal` (float)
- `pstar` (float or empty)
- `pstar_scope` (string)
- `scope_ok` (bool)
- `meta_ok` (bool)
- `strat_ok` (bool)
- `size_mult` (float, if available)
- `risk_usd` (float, if available)
- `risk_on` (0/1, if available)

Notes:
- One row per decision event.
- If multiple rows share `(symbol, decision_ts)`, latest row should be kept before export.
- Timestamp timezone must be explicit UTC.

## 2) `live_trades.csv` schema

Minimum columns:
- `symbol`
- `opened_at` (UTC)
- `closed_at` (UTC)
- `pnl` (net realized cash PnL)

Recommended additional columns:
- `size`, `entry_price`, `exit_reason`, `risk_usd`, `win_probability_at_entry`

## 3) `symbols_active.txt`

- Exact symbol universe used by live scanner for that day/window.
- One symbol per line, uppercase (example: `BTCUSDT`).

## 4) `run_context.json`

Required keys:
- `exported_at_utc`
- `window_start_utc`
- `window_end_utc`
- `timezone` (should be `"UTC"`)
- `bundle_id`

Recommended keys:
- `config_hash`
- `strategy_version`
- `meta_threshold`
- `meta_scope`
- `risk_mode`
- `risk_usd_base`

## Minimal live-team workflow

1. Extract `META_DECISION` lines to normalized `live_decisions.csv`.
2. Export daily closed trades to `live_trades.csv`.
3. Save symbol universe as `symbols_active.txt`.
4. Write `run_context.json`.
5. Place package in shared folder consumed by backtester machine.

## Backtester ingestion expectations

Backtester parity pipeline assumes:
- Decision timestamps can be bucketed to 5-minute bars.
- Symbol casing already normalized or normalizable.
- Missing optional columns are tolerated; missing required columns are hard errors.

