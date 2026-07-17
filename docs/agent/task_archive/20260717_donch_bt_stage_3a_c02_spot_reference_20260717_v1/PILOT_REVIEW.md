# Independent Pilot Review

Decision: `approve`.

The reviewer independently rebuilt all four frozen pairs from the three official Kraken archives and reproduced each normalized Parquet frame exactly. All 12 predeclared pair-window cells are non-empty. Timestamps are ordered, UTC-aware, unique at the 5-minute bar key, and strictly below `2026-01-01T00:00:00Z`; availability is the completed interval end. Pair identity is USD-only and agrees with the frozen official AssetPairs snapshot.

Exact duplicate three-column trade tuples are disclosed but not removed because the archive has no trade ID and identical executions can be distinct trades. Their volume is retained. No gap is filled. Current pair metadata is used only for identity; historical availability is established only where archive rows are observed. Full acquisition is approved under the same immutable source hashes and non-overlapping archive bounds.
