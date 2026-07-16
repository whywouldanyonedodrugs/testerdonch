# Package Protocol Status

Status: `blocked_by_protocol_issue`; `package_release_ready=false`.

Root: `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/`.

Open items remain separate from the loader repair:

1. Raw event-window trade, mark, index/spot, and exact-funding verification extracts are absent.
2. Test/failure counts are blank for all 14 families.
3. The prior-high v2 runner source snapshot is missing.
4. Five family lineages lack complete reproducibility hashes: TSMOM v6, A1/compression, prior-high/reclaim v2, C2 post-catalyst, and repaired LFBS.
5. TSMOM MAE/MFE coverage is incomplete.
6. The package embeds continuity rev7 while transferred rev8 is current authority.

Archives exist:

- `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/qlmg_external_review_core_20260716_v1.zip`: 47,910,624 bytes; decision metadata records 47,910,621.
- `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/qlmg_external_review_full_20260716_v1.tar.zst`: 258,454,978 bytes; decision metadata records 258,454,998.

The recorded package-manifest hash status remains pass, but integrity does not close the protocol gaps.
