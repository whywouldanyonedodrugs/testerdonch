# Changed-file record

Relative to starting commit
`56059c1a2e86c91c32087bd4e4da9257b632fbd9`, the task branch adds:

- `tools/core_liquid_campaign/`: typed schema, canonical serialization,
  deterministic generators/compilers, family engines, controls, accounting,
  selection, semantic cache, executor, campaign orchestration, runtime,
  synthetic fixtures, packet builder and validators;
- `tools/build_stage22_core_liquid_campaign.py`: supported outcome-free packet
  build/replay entry point;
- `tools/run_stage22_core_liquid_campaign.py`: proposed authority-bound future
  campaign entry point, currently blocked by independent review;
- `unit_tests/test_core_liquid_campaign.py`: 25 focused deterministic tests;
- this task archive: exact received authority, plan, decisions, verification,
  command evidence, V03/V04 independent reviews, blocked completion and next
  action.

The generated V02, V03 and V04 result roots are ignored versioned evidence and
are preserved without mutation. No pre-existing tracked file from the starting
authority was modified or deleted.
