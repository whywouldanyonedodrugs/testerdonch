# Pre-Price Schedule Preflight Failure

Status: `preserved_zero_outcome_code_failure`.

The first invocation of the independently approved runner stopped in `reconstruct_schedule` before output-root creation and before any PF open, return, funding value, protected row, control outcome, KDA01 outcome, KDA02B outcome, or Capital.com payload was opened.

Root cause: schedule reconciliation treated every row in `KDA02_V2_FEASIBILITY_GATES.csv` as an economic definition, including the four deliberately omitted definitions from infeasible continuation branches. The economic register correctly contained only four feasible primary definitions plus four corresponding robustness definitions.

Exact remedy: intersect reconciliation rows with the frozen primary definition identities, require the intersection to equal the complete frozen primary-definition set, add a regression with an omitted infeasible gate row, then rebuild and independently review every changed hash before another outcome-boundary invocation.
