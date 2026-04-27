#!/bin/bash
# Run multi-depth sensitivity analysis detached from any terminal session.
# Output goes to log file; results saved incrementally to JSON.
#
# Usage: nohup bash scripts/run_sensitivity_multi.sh &

set -e
cd /var/home/deucebucket/ai-drive/osmosis

MODEL_PATH=".hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OUTPUT="osmosis-qwen3.5-9b/sensitivity_multi.json"
LOG="osmosis-qwen3.5-9b/sensitivity_multi.log"
SAMPLES=8
MAX_LENGTH=128

echo "Starting multi-depth sensitivity analysis at $(date)" | tee "$LOG"
echo "Model: $MODEL_PATH" | tee -a "$LOG"
echo "Samples: $SAMPLES, Max length: $MAX_LENGTH" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -m osmosis.sensitivity_multi \
    --model "$MODEL_PATH" \
    --output "$OUTPUT" \
    --samples "$SAMPLES" \
    --max-length "$MAX_LENGTH" \
    -v \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Completed at $(date)" | tee -a "$LOG"
