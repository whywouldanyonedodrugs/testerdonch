# JT Pipeline: 14b -> 15 -> 16

## Goal
Run the three tickets in sequence with stage notifications and stop-on-failure behavior.

## Runner
- `tools/run_jt_pipeline_14b_15_16.py`
- tmux launcher: `tools/run_jt_pipeline_14b_15_16_tmux.sh`

## tmux command
```bash
export DONCH_TG_BOT_TOKEN='...'
export DONCH_TG_CHAT_ID='...'
tools/run_jt_pipeline_14b_15_16_tmux.sh jt_seq_20260216 --jobs 2 --jt015-pstar 0.42
```

## Notifications
The pipeline sends:
1. `STARTED`
2. `STAGE_START` / `STAGE_DONE` for each stage
3. `FAILED` on first failed stage
4. `DONE` after all three stages
