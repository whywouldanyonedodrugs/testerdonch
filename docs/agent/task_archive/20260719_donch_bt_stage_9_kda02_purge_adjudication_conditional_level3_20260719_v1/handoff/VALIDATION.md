# Validation

Outcome-free generation completed for `187` symbols, `8281` parent episodes, and `5128` candidates. Duplicate episodes/events/economic addresses and protected rows are `0/0/0/0`. Feasible primary branches: `2`.

The first reviewed runner invocation stopped before output-root creation and before any economic outcome read because schedule reconciliation included deliberately omitted infeasible gate rows. That failure is preserved. The surgical identity-intersection repair passed `103` relevant tests and a cache-only v5 rebuild; fresh independent approval matched all changed hashes before the economic run.

The single authorized run produced `9,274` schedule rows, `8,999` accepted trades, `275` actual-position-overlap exclusions, and `0` price rejections. All `8,999` entry/exit prices matched official PF five-minute opens across `54` symbols. Gross, 14 bps base-net, 32 bps stress-net, `3,087` market-day rows, `80,000` bootstrap draws, concentration metrics, funding diagnostics, and gate flags independently recomputed. Primary passes: `0`; protected rows: `0`; controls and KDA02B outcomes: `0`.

Final validation: `pass` for evidence integrity and `KDA02_level3_no_primary_pass_stop` for the economic decision. Independent pre-run and post-run reviews are approved with no blockers.
