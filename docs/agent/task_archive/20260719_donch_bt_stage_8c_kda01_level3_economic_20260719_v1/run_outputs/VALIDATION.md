# Validation

Status: pass. Terminal decision: `KDA01_level3_no_primary_pass_stop`.

- Frozen contract/register/cluster hashes matched before price access.
- Schedule reconstructed `204,272` rows: `183,744` accepted and `20,528` rejected/skipped (`20,473` actual overlap; `55` missing exit bar).
- Required accepted ledger contains exactly `183,744` accepted rows; rejection ledger contains `20,528` rows.
- Price rejections: `0`; deterministic 100-row source check matched exact official PF 5m trade-bar opens at entry and exit.
- Costs reconcile exactly: gross minus 14/32 bps equals base/stress for every trade.
- Market-day reports and all 160,000 bootstrap draws recomputed deterministically.
- Funding boundary rows: `606,164`; missing/duplicate joins: `0/0`; funding remained excluded from all gates.
- Protected rows opened/output: `0/0`; controls executed: `0`.
- Primary passes: `0/8`; robustness definitions cannot rescue the result.
