# JT-014b: Exit Mechanics Extension

## Goal
Run a controlled expansion of exit mechanics on the same signals universe, without mixing entry changes.

## What is included
1. Partial TP variants:
- `PARTIAL_TP_ENABLED`
- `PARTIAL_TP1_ATR_MULT`
- `MOVE_SL_TO_BE_ON_TP1`
2. Trail variants:
- `TRAIL_AFTER_TP1`
- `TRAIL_ATR_MULT`
3. Dynamic exit scaling variants:
- `DYN_EXITS_ENABLED`
- `DYN_*` multipliers
4. Regime-flip/time-stop variants:
- `EXIT_ON_REGIME_FLIP`
- `EXIT_ON_REGIME_FLIP_GRACE_BARS`
- shorter risk-off `TIME_EXIT_HOURS`

## Runner
- `tools/run_jt014_regime_conditioned_exits.py --preset-mode mechanics`

## tmux
```bash
export DONCH_TG_BOT_TOKEN='...'
export DONCH_TG_CHAT_ID='...'
tools/run_jt014b_tmux.sh jt014b_full_20260216 --jobs 2
```

## Progress/alerts
- Telegram: `STARTED`, periodic `PROGRESS`, `DONE/WARN`.
- Local progress file: `results/jt014_regime_exits/<run_id>/_PROGRESS.json`.
