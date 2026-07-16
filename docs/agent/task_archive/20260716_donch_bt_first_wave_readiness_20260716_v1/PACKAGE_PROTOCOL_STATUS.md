# Package Protocol Status

## Decision

The external-review package is not release-ready.

- Root: `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1`
- Status source: `decision_summary.json`
- Status: `blocked_by_protocol_issue`
- `package_release_ready`: `false`
- `unresolved_schema_issues`: `1`
- Full archive: `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/qlmg_external_review_full_20260716_v1.tar.zst`
- Full archive bytes: `258454998`
- Core ZIP: `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/qlmg_external_review_core_20260716_v1.zip`
- Package manifest: `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/package_manifest.csv`
- Hash index: `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/package_sha256.json`

The builder explicitly hard-codes the unresolved schema item because raw-bar verification extracts are missing: `tools/build_qlmg_external_review_package.py:541`.

## Open Items

1. Bar-level trade, mark, index/spot, and exact-funding verification extracts were not regenerated. Each family note states this explicitly at `families/<family>/VERIFICATION_DATA_NOTE.md`.
2. Test and failure counts are blank for all 14 families in `engineering/test_execution_matrix.csv`.
3. The source snapshot for `tools/run_kraken_prior_high_reclaim_v2_canonical_train_scan.py` is missing in `engineering/code_lineage.csv`.
4. Full reproducibility hashes are `not_recorded` for five lineages in `engineering/reproducibility_matrix.csv`: `tsmom_v6`, `a1_compression`, `prior_high_reclaim_v2`, `c2_post_catalyst`, and `lfbs_repaired`.
5. TSMOM has partial path coverage: `registry/family_package_statistics.csv` records `mae_available=false` and `mfe_available=false`; the package summary reports 13 full families and one partial family.
6. The package embeds continuity brief rev7. The newly transferred rev8 authority is higher-current narrative authority, but it does not mutate or repair the finalized package.

## Wording Review

`README_FIRST.md` calls the material train-only research evidence and disclaims validation, live readiness, and permission to trade. No inspected package-status narrative overturned the machine status. Hash validity proves integrity, not protocol closure.
