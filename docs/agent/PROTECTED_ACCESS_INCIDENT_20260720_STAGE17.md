# Protected Access Incident — Stage 17 — 2026-07-20

Status: bounded non-outcome protected funding access incident; open remediation dependency.

During Stage 17 preflight, `tools/run_kraken_shared_funding_imputation_model.py::load_exact_rates` deserialized complete mixed-period funding Parquet files and filtered timestamps only afterward. This opened protected-period non-outcome funding payload rows.

Preserved facts:

- protected non-outcome funding payload opened: yes;
- protected strategy-price outcomes opened: no;
- protected funding rows used for fitting or economics: no;
- exact protected-row count: unavailable and must not be recomputed by reopening payload;
- campaign economics executed: no;
- Telegram messages sent: no.

The defect is a physical read-boundary failure even though the protected rows were filtered before fitting. It does not become a zero-access claim. Stage 18 adds a metadata-first row-group guard: only a group whose trusted footer proves `max_timestamp < 2026-01-01T00:00:00Z` may be requested, followed by a payload assertion. Mixed, protected, unknown, corrupt, hash-drifted, or contradictory groups fail closed.

The current official-source Parquet layout has no safely readable row groups, so no Stage 18 funding payload was opened and no rankable exact-funding data was materialized.
