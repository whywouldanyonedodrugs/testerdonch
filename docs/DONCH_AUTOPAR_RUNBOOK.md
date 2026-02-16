# Donch Autopar Runbook (Phase 1)

## What this gives you now

- Automated daily parity workflow on backtester machine:
  - Pull latest live export package via `rsync`
  - Ingest live package
  - (Optional) extract decisions from live logs
  - Run reference scout+backtest on same window/universe
  - Compare live vs backtest decisions
  - Compute live schema diagnostics (`schema_fail_rate`, top missing fields, top affected symbols)
  - Compute data freshness/completeness SLA diagnostics (OHLCV/OI/funding staleness + coverage)
  - Generate JSON/CSV/HTML report
  - Send Telegram summary

## Scripts

1. `tools/donch_autopar_extract_live_meta_decisions.py`
2. `tools/donch_autopar_compare.py`
3. `tools/donch_autopar_daily.py`
4. `tools/donch_autopar_pull_and_run.py`

## Live-team required output (daily)

Place one package folder (or zip) in a shared location with:
- `live_decisions.csv` (required)
- `live_trades.csv` (recommended)
- `symbols_active.txt` (recommended)
- `run_context.json` (recommended)

Spec:
- `docs/DONCH_AUTOPAR_LIVE_EXPORT_SPEC.md`

If `live_decisions.csv` is missing, backtester can attempt extraction from `live.log` using `META_DECISION` lines.

## One-off usage

### A) Extract decisions from a live log

```bash
./.venv/bin/python tools/donch_autopar_extract_live_meta_decisions.py \
  --input /path/to/live.log \
  --out /path/to/live_decisions.csv
```

### B) Compare existing live and backtest decision files

```bash
./.venv/bin/python tools/donch_autopar_compare.py \
  --live-decisions /path/to/live_decisions.csv \
  --bt-decisions /path/to/signal_decisions.csv \
  --live-trades /path/to/live_trades.csv \
  --bt-trades /path/to/trades.csv \
  --outdir results/donch_autopar/manual_compare_YYYYMMDD
```

### C) Full daily autopar run (recommended)

```bash
./.venv/bin/python tools/donch_autopar_daily.py \
  --live-input /path/to/live_package_or_zip \
  --results-root results/donch_autopar \
  --parquet-dir /opt/fader2/parquet \
  --model-dir results/offline_releases/20260209_meta_release_v2_live_safe/meta_export_pstar_042 \
  --window-days 3 \
  --scout-workers 2 \
  --tg-auto-chat
```

### D) Fully automated pull + run (no manual package path)

This is the main automation entrypoint for backtester machine.

```bash
./.venv/bin/python tools/donch_autopar_pull_and_run.py \
  --remote-src root@167.235.64.82:/srv/donch-autopar/ \
  --sync-dir /data/autopar \
  --results-root results/donch_autopar \
  --parquet-dir /opt/fader2/parquet \
  --model-dir results/offline_releases/20260209_meta_release_v2_live_safe/meta_export_pstar_042 \
  --window-days 3 \
  --max-new-packages 3 \
  --resume \
  --tg-auto-chat
```

What it does:
- pulls remote exports into local mirror dir
- discovers `autopar_YYYY-MM-DD` package(s)
- skips already-processed days using state file
- processes pending days (or latest only if requested)
- writes state to `results/donch_autopar/.autopar_state.json`
- uses lock file `results/donch_autopar/.autopar.lock` to prevent overlap

## Scheduling (backtester machine)

### Preflight (SSH to live export host)

Before enabling timer, validate pull credentials once:

```bash
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" root@167.235.64.82:/srv/donch-autopar/ /data/autopar/
```

If you get `Permission denied`, add/authorize SSH key for this machine first.  
The automation script already handles host-key acceptance for first connect.

### Option 1: systemd timer (recommended)

Files:
- `deploy/systemd/donch-autopar-backtester.service`
- `deploy/systemd/donch-autopar-backtester.timer`
- `deploy/systemd/donch-autopar-backtester.env.example`

Install:

```bash
sudo mkdir -p /etc/default
sudo cp /opt/testerdonch/deploy/systemd/donch-autopar-backtester.env.example /etc/default/donch-autopar-backtester
# edit /etc/default/donch-autopar-backtester and set DONCH_TG_BOT_TOKEN / DONCH_TG_CHAT_ID

sudo cp /opt/testerdonch/deploy/systemd/donch-autopar-backtester.service /etc/systemd/system/
sudo cp /opt/testerdonch/deploy/systemd/donch-autopar-backtester.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now donch-autopar-backtester.timer
sudo systemctl status donch-autopar-backtester.timer --no-pager
```

Manual trigger:

```bash
sudo systemctl start donch-autopar-backtester.service
sudo journalctl -u donch-autopar-backtester.service -n 200 --no-pager
```

### Option 2: cron

Cron example (daily at 00:35 UTC):

```cron
35 0 * * * cd /opt/testerdonch && ./.venv/bin/python tools/donch_autopar_pull_and_run.py --remote-src root@167.235.64.82:/srv/donch-autopar/ --sync-dir /data/autopar --results-root results/donch_autopar --parquet-dir /opt/fader2/parquet --model-dir results/offline_releases/20260209_meta_release_v2_live_safe/meta_export_pstar_042 --window-days 3 --max-new-packages 3 --resume --tg-auto-chat >> results/donch_autopar/cron.log 2>&1
```

## Safety with fader2

- Use `--parquet-dir /opt/fader2/parquet` in read-only mode.
- Do not point outputs under `/opt/fader2`.
- Donch autopar outputs stay under `results/donch_autopar`.
- This does not delete or rewrite fader2 artifacts.

## Outputs per run

For run id `<RID>`:
- `results/donch_autopar/<RID>/run_manifest.json`
- `results/donch_autopar/<RID>/live_schema_diag.json`
- `results/donch_autopar/<RID>/data_sla_diag.json`
- `results/donch_autopar/<RID>/data_sla_details.csv`
- `results/donch_autopar/<RID>/compare/summary.json`
- `results/donch_autopar/<RID>/compare/comparison_rows.csv`
- `results/donch_autopar/<RID>/compare/mismatches.csv`
- `results/donch_autopar/<RID>/compare/report.html`
- `results/donch_autopar/<RID>/logs/*.log`

## Initial practical thresholds

Pass/warn defaults:
- overlap rate (live keys covered by bt): `>= 0.50`
- enter/skip agreement on overlap: `>= 0.85`

Tune these after first 1-2 weeks of stable daily baselines.
