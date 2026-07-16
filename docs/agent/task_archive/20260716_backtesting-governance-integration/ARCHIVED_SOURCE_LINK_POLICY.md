# Archived Source Link Policy

Exact received-source files under `received/` are provenance records from the approved transfer package. They are preserved verbatim where possible.

Whole-tree markdown link validation found that `received/DONCH_PROJECT_INSTRUCTIONS_FULL.md` contains relative links to its original Donch audit context, such as `../00_audit/EXECUTION_PLAN.md`. Those files were not part of the approved ZIP and were not fetched from an unverified location.

Decision:

- Do not edit received provenance copies to make links look local.
- Exclude `received/` source copies from repository-relative link validation.
- Validate adapted repository docs, active agent manuals, skills, evals, and task-authored archive records.

Status: accepted provenance limitation, not an active-doc broken link.
