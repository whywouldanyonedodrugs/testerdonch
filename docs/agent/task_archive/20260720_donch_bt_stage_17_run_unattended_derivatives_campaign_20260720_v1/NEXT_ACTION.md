# Next Action

New exact human approval and a replacement packet are required before launch.

The replacement must freeze:

1. the authoritative linear-PF boundary-notional or boundary spot/index source and timestamp;
2. exact and imputed `relativeFundingRate` conversion to signed cashflow bps, including the notional ratio;
3. the allowed behavior when that boundary source is missing;
4. a protected-safe raw funding loader that proves row-group or predicate exclusion before payload deserialization;
5. updated synthetic and real pre-outcome tests binding those semantics.

If the required historical spot/index payload is not already present in an approved manifest, separate new-data-acquisition approval is also required. Do not reuse the current approval to make these changes or to launch.

