# Local-Only Artifacts

Large official archives and normalized market payloads are intentionally excluded from Git and Drive.

- Raw official archives: `/opt/parquet/kraken_spot_reference/official_kraken_time_sales/raw/`
- Pilot bars: `/opt/parquet/kraken_spot_reference/official_kraken_time_sales/pilot_5m/`
- Full sparse bars: `/opt/parquet/kraken_spot_reference/official_kraken_time_sales/normalized_5m/`
- Gap masks: `/opt/parquet/kraken_spot_reference/official_kraken_time_sales/gap_masks/`
- Local immutable manifest: `/opt/parquet/kraken_spot_reference/official_kraken_time_sales/manifests/C02_SPOT_DATA_MANIFEST.json`

Exact paths, byte sizes, and SHA-256 values are in `LOCAL_ONLY_FILE_MANIFEST.csv`, `C02_SPOT_SOURCE_LEDGER.csv`, and `C02_SPOT_DATA_MANIFEST.json`. These files are data authority for a later bounded non-economic generator contract, not economic evidence.
