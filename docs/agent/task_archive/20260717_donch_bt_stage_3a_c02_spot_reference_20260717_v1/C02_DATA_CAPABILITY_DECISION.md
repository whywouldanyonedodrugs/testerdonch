# C02 Data Capability Decision

Decision: `ready_for_C02_non_economic_generator_contract`.

Official Kraken downloadable time-and-sales provides a reproducible historical path for the bounded rankable interval. The pilot was frozen before price access and independently reproduced exactly for four pairs across all 12 required pair/window cells. Full acquisition attempted all 210 uniquely mapped Stage 2C PF assets and observed archive rows for 204.

The 204-pair panel contains 19,458,116 sparse five-minute bars derived from 234,996,053 official trades. Coverage spans `2023-01-01T00:00:00Z` through `2025-12-31T23:55:00Z` overall, but bounds and gaps vary by pair. There are 5,581,777 explicit internal gap intervals representing 24,834,785 absent five-minute slots between observed bounds. Six mapped pairs had no official archive rows. Forty-three cohort symbols had no unique current official USD identity.

This is row-observed historical authority, not proof of uninterrupted listing, lifecycle status, or survivorship-free coverage. Current AssetPairs metadata supports identity only. A later generator must consume sparse completed-bar intersections and masks exactly as frozen in `C02_SPOT_AND_PF_ALIGNMENT_CONTRACT.md`.

No C02 signal, lead/lag label, threshold, candidate return, ranking, or economic output was computed.
