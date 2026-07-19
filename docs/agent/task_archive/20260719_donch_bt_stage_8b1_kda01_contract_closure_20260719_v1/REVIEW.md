# Independent Review

Status: approve for human KDA01 Level-3 run-approval review. Economic execution remains unauthorized.

## Scope reviewed

Reviewed the actual Stage 8B1 code and artifact diff, complete Stage 8B source manifest, all 800 bound source objects, exact source tapes, cross-symbol daily and six-hour identities, timestamp-only reader projection, exact entry/exit delay boundaries, definition-local non-overlap, execution rejection accounting, amended 16-definition register, seven preserved controls, serialized contract, deterministic replay, protected boundary, and package contents.

## Findings resolved

1. The first complete invocation lacked the established script-mode repository path bootstrap and failed before data read. The bootstrap and a regression assertion were added.
2. The second invocation encountered a non-bar manifest envelope and failed closed before its payload reader. The loader now skips non-bar envelopes by schema before `read()`, matching Stage 8B authority behavior; a reader-spy regression covers it.
3. The initial Stage 8B1 manifest embedded the verified source manifest but did not carry the exact source JSON or explicit repository file hashes. The final package includes the byte-identical source manifest and hashes six Stage 8B/8B1 code/test files.
4. Four broad-suite tests initially lacked their ignored seal-metadata fixture in the isolated worktree. Linking the existing repository metadata resolved the environment issue; the final full relevant suite passed `211/211`.

## Conclusions

- Event generation is unchanged; all event IDs and source economic addresses reconcile exactly.
- Market-day identity intentionally groups symbols and both parent directions only within the same attempt and UTC onset date.
- The six-hour identity and symbol-specific parent episode cannot rescue failure under the daily cluster.
- Entry and exit timestamps use exact inclusive ten-minute availability caps; no alternative cap was tested.
- Missing exits fail closed. Actual overlap is evaluated independently by definition and symbol using eligible timeout exit timestamps.
- No price column, candidate outcome, funding outcome, control outcome, or protected row is read.
- All eight primary timeout definitions retain the original mechanical gates after timestamp availability and non-overlap.
- The Level-3 v2 contract is complete and deterministic, but this is mechanical feasibility only and provides no economic evidence.

## Remaining limits

Stage 8A/8B inferred analytics semantics, March-2023 OI retention start, current-roster/lifecycle cap, no survivorship-free claim, no economic evidence, no funding evidence, and high event/cluster counts remain active. The 55 missing-exit-bar rows are excluded rather than repaired or filled.
