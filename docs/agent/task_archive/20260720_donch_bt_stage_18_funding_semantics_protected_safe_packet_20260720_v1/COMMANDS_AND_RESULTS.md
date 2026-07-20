# Commands and Results

- `sha256sum` over Stage 16/17 authorities: all expected hashes matched.
- PyArrow footer-only inventory: 305 authoritative source files and row groups; 288 mixed, 17 protected, zero safe.
- `python tools/kraken_protected_safe_funding.py ...`: exit 3, expected `blocked_no_safe_rankable_absolute_funding_row_groups`; payload reads zero.
- `python -m unittest unit_tests.test_kraken_protected_safe_funding unit_tests.test_stage16_campaign_semantics unit_tests.test_qlmg_research_campaign -v`: 39 passed, zero failed.
- No economic runner, outcome reader, network acquisition, Telegram notifier, Capital.com reader, order client, or live-trading path invoked.
