# Source and Capability Update

## Verified

The official public Kraken Futures analytics route returned reproducible bounded hourly rows for `open-interest`, `liquidation-volume`, and `future-basis` for PF_XBTUSD and PF_ETHUSD in each frozen 2023, 2024, and 2025 window. Each populated response contained 25 timestamps including both requested endpoints. The explicit upper bounds were honored and the exact replay matched byte-for-byte.

The `funding` route returned a valid documented schema but zero timestamps for all six tested cells. It does not change the existing exact-funding authority.

## Limits

This is not a full-history acquisition, pagination audit, universe audit, or value-unit authority. OI, liquidation, and basis units/sign remain unavailable. The result cannot support a rankable feature or economic screen without a separately authorized bounded historical authority audit.
