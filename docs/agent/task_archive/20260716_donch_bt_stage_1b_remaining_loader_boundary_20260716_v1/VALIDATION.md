# Validation

## Defect evidence

Before production edits, focused synthetic tests failed because the three scoped readers opened protected/mixed/unknown fixtures, admitted pre-2023/non-Kraken rows, and received no authority map from `data_paths()`.

## Passing checks

- Focused loader boundary: 11/11.
- Owning aggregate module: 286/286.
- Repository guards: 9/9.
- Compile: pass.
- Diff check: pass.
- Bounded AST call-order review: 5/5 known readers.
- Real manifest metadata binding: 166,408 existing local paths; no payload opened.

## Spy results

- Protected/mixed/unknown payload-reader calls: 0 for each of the three Stage 1B readers.
- Downstream calls after fail-closed rejection: 0.
- Invalid pre-2023/non-Kraken rows reaching downstream: 0.
- Valid Kraken 2023-2025 fixtures: pass for all three readers.

No economic output, protected payload, capture, or acquisition was used.
