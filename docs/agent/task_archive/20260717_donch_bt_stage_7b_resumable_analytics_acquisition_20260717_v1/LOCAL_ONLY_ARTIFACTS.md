# Local-Only Artifacts

- Corrected run root: `results/rebaseline/phase_kraken_futures_analytics_acquisition_20260717_v1_20260717_195535`
- Corrected immutable data root: `/opt/parquet/kraken_derivatives/analytics/stage7b_v1_attempt2`
- Quarantined diagnostic run root: `results/rebaseline/phase_kraken_futures_analytics_acquisition_20260717_v1`
- Quarantined diagnostic data root: `/opt/parquet/kraken_derivatives/analytics/stage7b_v1`

The corrected SQLite job ledger, request-ledger Parquet, gap-register Parquet, raw JSON.zst responses, normalized Parquet parts, and logs remain local. The full data manifest is retained at `results/rebaseline/phase_kraken_futures_analytics_acquisition_20260717_v1_20260717_195535/KRAKEN_ANALYTICS_DATA_MANIFEST.json` and binds every source/normalized part by path, byte size, and SHA-256. Large/binary artifacts are intentionally excluded from Drive.
