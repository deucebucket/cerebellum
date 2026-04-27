#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

MODEL_PATH="/var/home/deucebucket/ai-drive/osmosis/.hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
OUTPUT_DIR="/var/home/deucebucket/ai-drive/osmosis/osmosis-qwen27b"
LOG="$OUTPUT_DIR/pipeline_run2.log"

echo "============================================" | tee "$LOG"
echo "  OSMOSIS RUN 2 — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "  RAM free: $(free -h | grep Mem | awk '{print $7}')" | tee -a "$LOG"
echo "  Disk free: $(df -h /var/home/deucebucket/ai-drive/ | tail -1 | awk '{print $4}')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

osmosis pipeline \
  --model "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --skip-download \
  --skip-activations \
  2>&1 | tee -a "$LOG"

echo "PIPELINE FINISHED: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
