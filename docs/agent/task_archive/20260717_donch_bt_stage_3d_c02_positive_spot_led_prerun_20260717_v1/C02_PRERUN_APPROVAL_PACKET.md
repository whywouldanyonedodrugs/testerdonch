# C02 Pre-Run Approval Packet

Status: `ready_for_human_C02_Level3_run_approval`.

- Final contract SHA-256: `1c4f7f6ec81fa86d1c1355ce899570bdec85e413ad4febe110f33cc4ec565496`.
- Stage 3C resolution contract: `ce65c62edfb80f5fb83e9b8b6bae1d3eb9c981f8e9a1bcad3b285fdce46cca51`.
- Stage 3C event tape: `c73344b1bd104c0816d731a1002f729b49100385a34b9b56ec4b2be66dad71ad`.
- Stage 3B source contract: `25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb`.
- Spot manifest: `3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046`.
- Stage 2C cohort: `768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15`.

Proposed later command interface (runner intentionally not implemented):

`./.venv/bin/python tools/run_kraken_c02_level3_economic.py --contract docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_FINAL_LEVEL3_ECONOMIC_CONTRACT.md --definitions docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_LEVEL3_DEFINITION_REGISTER.csv --event-set docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_POSITIVE_SPOT_LED_EVENT_SET.csv --output-root results/rebaseline/phase_kraken_c02_positive_spot_led_level3_<UTC_SUFFIX>`

Rollback: revert later task commits normally and preserve every result root. Forbidden: protected data, branch expansion, alternate thresholds/horizons, negative/perp/failure branches, controls without a primary all-pass result and human approval, validation, portfolio, or live work.
