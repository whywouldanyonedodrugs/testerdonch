# Known Gaps and Limits

- Current-roster/lifecycle-capped Kraken train evidence is not a fully point-in-time historical universe.
- Analytics units are inferred-authoritative; liquidation direction is price-inferred rather than a native long/short label.
- Funding is largely imputed and is diagnostic only; it is excluded from economic gates.
- No controls, KDA02B outcomes, validation period, protected period, Capital.com payload, portfolio construction, capacity study, live execution, or production validation was run.
- The compact ZIP excludes all Parquet and bootstrap payloads; their local paths, sizes, and SHA-256 values remain in `ARTIFACT_MANIFEST.json` and the approved post-run review.
