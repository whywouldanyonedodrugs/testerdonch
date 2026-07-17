# Stage 7C Semantic Status

All three metrics are authorized for exact public-source acquisition but blocked from economic interpretation.

- `future-basis`: field identity and positive-above-spot sign are verified; numeric unit unresolved. Every aligned returned field is retained as an exact raw string/JSON field.
- `liquidation-volume`: aggregate forced-close concept verified; side split and numeric unit/currency unresolved. Exact scalar retained as `value_raw`.
- `open-interest`: outstanding-position concept verified; four-string tuple meanings and units unresolved. Stored losslessly as `value_0_raw` through `value_3_raw`.

Every normalized row carries `semantic_status=source_authorized_economic_interpretation_blocked`, source job identity, schema hash through the ledger, original epoch timestamp, and exact raw value representation. Basis is not funding. No signal or economic use is authorized.
