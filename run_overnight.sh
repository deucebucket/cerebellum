#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/var/home/deucebucket/ai-drive/osmosis/.hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
OUTPUT_DIR="/var/home/deucebucket/ai-drive/osmosis/osmosis-qwen27b"
LOG="$OUTPUT_DIR/pipeline.log"

mkdir -p "$OUTPUT_DIR"

echo "============================================" | tee "$LOG"
echo "  MODEL OSMOSIS — Qwen3.6-27B" | tee -a "$LOG"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "  Model: $MODEL_PATH" | tee -a "$LOG"
echo "  Output: $OUTPUT_DIR" | tee -a "$LOG"
echo "  Disk free: $(df -h /var/home/deucebucket/ai-drive/ | tail -1 | awk '{print $4}')" | tee -a "$LOG"
echo "  RAM free: $(free -h | grep Mem | awk '{print $7}')" | tee -a "$LOG"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

export PYTHONUNBUFFERED=1

osmosis pipeline \
  --model "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --skip-download \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  FINISHED: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "  Disk free: $(df -h /var/home/deucebucket/ai-drive/ | tail -1 | awk '{print $4}')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
