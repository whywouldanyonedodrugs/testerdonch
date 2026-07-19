# QLMG Kraken Validation Protocol

Status: active sealed-data policy amended by the 2026-07-19 Stage 10 policy application.

This protocol does not authorize an economic run.

Rankable historical research is limited to Kraken rows in `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`. The period from `2026-01-01T00:00:00Z` onward is sealed for strategy outcomes and may be used only for strategy-agnostic execution calibration unless a later formal policy change and explicit human approval say otherwise.

Validation rules:

- no protected-period strategy outcomes may be inspected for scoring, tuning, controls, selection, validation, promotion, or portfolio work;
- no final candidate is valid unless its mechanism, universe, candidate identity, control identity, costs, funding, boundaries, and validation protocol were frozen before outcomes;
- overlapping labels or horizons require purged and embargoed validation when material;
- report the search process, parameter budget, reruns, rejected alternatives, and selection-bias risk;
- distinguish mechanical hash/schema success from statistical validity and release readiness;
- preserve negative findings and blocked statuses.
- keep research-route status separate from evidence, reproducibility, validation, and deployment status;
- do not treat `conditional_context_candidate_unvalidated`, a convex-tail route, an execution-sensitive route, or a sample-limited route as validation evidence;
- require independent or prospective evidence for any post-hoc context and prohibit same-sample rescue of a closed translation.

Phase 0 and governance tasks do not run final validation. Documentation, code, archive, or review-package work must use non-economic checks and synthetic or pre-2026 fixtures as appropriate.
