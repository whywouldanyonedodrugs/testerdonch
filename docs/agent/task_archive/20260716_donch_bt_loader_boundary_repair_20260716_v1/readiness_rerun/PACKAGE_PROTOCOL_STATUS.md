# Package Protocol Status Rerun

Status: `blocked_by_protocol_issue`; `package_release_ready=false`.

Root: `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/`

Current metadata confirms the prior open items remain:

1. Raw event-window trade, mark, index/spot, and exact-funding verification extracts remain absent.
2. Test/failure counts remain blank for all 14 families.
3. The prior-high v2 runner source snapshot remains missing.
4. Five family lineages still have incomplete reproducibility hashes: TSMOM v6, A1/compression, prior-high/reclaim v2, C2 post-catalyst, and repaired LFBS.
5. TSMOM MAE/MFE remains incomplete.
6. The package embeds continuity rev7 while transferred rev8 remains the current narrative authority.

The full archive exists at `qlmg_external_review_full_20260716_v1.tar.zst`. Its actual filesystem size is `258454978` bytes, while `decision_summary.json` records `258454998`; this 20-byte metadata discrepancy is newly recorded and not repaired here. The core ZIP exists at `47910624` bytes.

No package narrative inspected here overrides the machine blocked status. Hash integrity is not release readiness.
