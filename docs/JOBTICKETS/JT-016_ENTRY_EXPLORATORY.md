# JT-016: Entry Exploratory + JT-011 Gates

## Goal
Explore a constrained entry grid, then select candidates by walk-forward stability gates (JT-011), not raw PnL alone.

## Runner
- `tools/run_jt016_entry_exploratory.py`

## Stages
1. Build fresh signals per entry variant (scout with overrides).
2. Backtest each variant.
3. Run JT-011 walk-forward evaluator for top-K candidates.
4. Recommend by: JT-011 pass first, then total PnL cash.

## tmux
```bash
export DONCH_TG_BOT_TOKEN='...'
export DONCH_TG_CHAT_ID='...'
tools/run_jt016_tmux.sh jt016_full_20260216 --max-variants 24 --jt011-top-k 3
```

## Outputs
- `summary.csv`
- `jt011_eval.csv`
- `recommendation.json`
- `report.md`
