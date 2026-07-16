# QLMG Kraken Validation Protocol

Status: active sealed-data policy as of 2026-07-16 UTC.

This protocol does not authorize an economic run.

Rankable historical research is limited to Kraken rows in `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`. The period from `2026-01-01T00:00:00Z` onward is sealed for strategy outcomes and may be used only for strategy-agnostic execution calibration unless a later formal policy change and explicit human approval say otherwise.

Validation rules:

- no protected-period strategy outcomes may be inspected for scoring, tuning, controls, selection, validation, promotion, or portfolio work;
- no final candidate is valid unless its mechanism, universe, candidate identity, control identity, costs, funding, boundaries, and validation protocol were frozen before outcomes;
- overlapping labels or horizons require purged and embargoed validation when material;
- report the search process, parameter budget, reruns, rejected alternatives, and selection-bias risk;
- distinguish mechanical hash/schema success from statistical validity and release readiness;
- preserve negative findings and blocked statuses.

Phase 0 and governance tasks do not run final validation. Documentation, code, archive, or review-package work must use non-economic checks and synthetic or pre-2026 fixtures as appropriate.
