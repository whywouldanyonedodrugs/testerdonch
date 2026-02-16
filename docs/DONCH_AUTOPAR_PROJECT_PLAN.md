# Donch Autopar Project Plan

Working title: `donch_autopar`

## Objective

Run an automated daily parity loop between live-trading decisions and offline backtest decisions, then publish a concise evaluation via Telegram.

Primary KPI:
- Decision parity (enter/skip + reason-level consistency), not exact PnL equality.

Secondary KPIs:
- Relative performance drift (trade count, win rate, directional PnL behavior).
- Sizing/risk drift (`size_mult`, risk target).

## Feasibility

This project is feasible with the current codebase.

Reasons:
- Backtester already supports per-signal decision logs (`signal_decisions.csv`) when `BT_DECISION_LOG_ENABLED=True`.
- Live trader already emits rich `META_DECISION` log lines with `symbol`, `decision_ts`, `p_cal`, `meta_ok`, `strat_ok`, `reason`, and scope data.
- Telegram notifier utility already exists (`tools/telegram_notify.py`).
- Existing orchestration patterns exist in `tools/run_*` scripts.

## Architecture (Phase 1)

1. Live machine:
- Export daily decision log artifact (CSV) from `META_DECISION` log lines.
- Export daily closed-trade report (CSV) and active symbol universe file.
- Drop artifacts into shared folder (preferred) or send via scheduled transfer.

2. Backtester machine:
- Ingest live artifact package.
- Run reference scout+backtest on same date window and symbol universe.
- Compare live decisions vs backtest decisions.
- Compute drift/performance summary.
- Send Telegram digest and store JSON/HTML artifacts.

3. Scheduling:
- Daily cron/systemd timer.
- Optional on-demand run command.

## Data safety and infra constraints

- Backtester should read market data from shared parquet root (recommended: `/opt/fader2/parquet`) in read-only mode.
- Do not mutate/delete anything under fader2 working directories.
- All `donch_autopar` outputs go under `results/donch_autopar/...`.
- Use separate run directories per day; no destructive cleanup by default.

## Phase 1 deliverables

1. Live export schema and runbook:
- `docs/DONCH_AUTOPAR_LIVE_EXPORT_SPEC.md`

2. Scripts:
- `tools/donch_autopar_extract_live_meta_decisions.py`
  - Parses live logs and exports normalized daily decision CSV.
- `tools/donch_autopar_compare.py`
  - Compares live vs backtest decisions (+ optional trade-level summary).
- `tools/donch_autopar_daily.py`
  - Orchestrates ingestion, reference backtest, comparison, Telegram message.

3. Reporting artifacts per run:
- `summary.json`
- `comparison_rows.csv`
- `report.html`
- stage logs (`01_scout.log`, `02_backtest.log`, ...)

## Phase 2 (recommended next)

- Add live-side direct decision sink to DB/file (avoid log parsing dependency).
- Add reason taxonomy map (live/offline reason aliases).
- Add SLA checks and hard fail criteria:
  - low overlap coverage
  - low enter/skip agreement
  - abnormal size drift
- Add weekly trend dashboard across daily parity runs.

## Initial acceptance thresholds (suggested)

- Overlap coverage >= 85%
- Enter/skip agreement on overlap >= 90%
- Reason agreement (normalized) >= 75%
- Median size multiplier ratio live/bt in [0.8, 1.25]

These thresholds should be tuned after 1-2 weeks of baseline.

