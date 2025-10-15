#!/usr/bin/env bash
# file: run_walk_forward.sh
# A comprehensive walk-forward analysis from 2023-01-01 to 2025-09-01.
set -euo pipefail

ARCHETYPES=(
  "avwap_lower_high_reject"
  "breakdown_retest_vwap"
  "carry_drift_fader"
)

DESIGN="grid"
SYMBOL_STRIDE="1"

# Walk-forward splits: 12 months of training, 4 months of testing.
# The window slides forward by 4 months for each split.
# Format: TRAIN_START,TRAIN_END,TEST_START,TEST_END
SPLITS=(
  "2023-01-01,2024-01-01,2024-01-01,2024-05-01"
  "2023-05-01,2024-05-01,2024-05-01,2024-09-01"
  "2023-09-01,2024-09-01,2024-09-01,2025-01-01"
  "2024-01-01,2025-01-01,2025-01-01,2025-05-01"
  "2024-05-01,2025-05-01,2025-05-01,2025-09-01"
)

for ARC in "${ARCHETYPES[@]}"; do
  echo "================================================================="
  echo "=== Starting Walk-Forward Sweep for Archetype: $ARC ==="
  echo "================================================================="

  for i in "${!SPLITS[@]}"; do
    S="${SPLITS[$i]}"
    IFS=, read -r TR_START TR_END TE_START TE_END <<<"$S"
    
    # --- RUN TRAINING PERIOD ---
    echo "--- WF Split $((i+1)): TRAINING $TR_START → $TR_END ---"
    python sweep_mr_short_params_guarded.py \
      --archetype "$ARC" \
      --design "$DESIGN" \
      --early_rungs "full" \
      --symbol_stride "$SYMBOL_STRIDE" \
      --start "$TR_START" --end "$TR_END" \
      --tag "wf${i}_train" \
      --resume

    # --- RUN TESTING (OUT-OF-SAMPLE) PERIOD ---
    echo "--- WF Split $((i+1)): TESTING  $TE_START → $TE_END ---"
    python sweep_mr_short_params_guarded.py \
      --archetype "$ARC" \
      --design "$DESIGN" \
      --early_rungs "full" \
      --symbol_stride "$SYMBOL_STRIDE" \
      --start "$TE_START" --end "$TE_END" \
      --tag "wf${i}_test" \
      --resume
  done
done

echo "--- All Walk-Forward Sweeps Completed ---"
echo "--- Aggregating All Results ---"
python rebuild_summaries.py --sweeps-dir ./results/sweeps --results-dir ./results
echo "Done. Raw results are in ./results/summary_by_variant.csv"

echo "--- Selecting Winners based on Out-of-Sample Performance ---"
python select_zoom_winners.py

echo "--- Final candidates are in results/zoom_winners.csv ---"
