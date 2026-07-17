# Source and Capability Update

The official public Kraken Futures analytics route reproducibly retained bounded pre-2026 rows for open interest, liquidation volume, and futures basis at 60- and 300-second intervals. Pagination is capped at 2,000 rows and uses inclusive boundaries; continuation must begin at the last returned timestamp and exact duplicate boundary rows must be reconciled.

This is bounded source-retention evidence only. The returned value shapes do not establish the necessary official units, direction, contract interpretation, or basis sign convention. No metric is rankable-authorized, basis is not funding, and no earlier hypothesis is reopened.
