#!/bin/bash
# Train repair LoRA for Osmosis 4GB Qwen 3.5 9B.
# Phase 1 (NIM data gen) = zero GPU. Phase 3-4 (training) needs GPU.
# Run detached: nohup bash scripts/run_repair_lora.sh &

set -e
cd /var/home/deucebucket/ai-drive/osmosis

MODEL_PATH=".hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OUTPUT="osmosis-qwen3.5-9b/repair-lora"
TENSOR_TYPES="osmosis-qwen3.5-9b/tensor_types_multidepth_4gb.txt"
LOG="osmosis-qwen3.5-9b/repair_lora.log"
NIM_CARD="carl"

echo "Starting repair LoRA training at $(date)" | tee "$LOG"
echo "Model: $MODEL_PATH" | tee -a "$LOG"
echo "Tensor types: $TENSOR_TYPES" | tee -a "$LOG"
echo "NIM card: $NIM_CARD" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -m osmosis.repair_nim \
    --model "$MODEL_PATH" \
    --output "$OUTPUT" \
    --tensor-types "$TENSOR_TYPES" \
    --nim-card "$NIM_CARD" \
    --domain repair \
    --lora-r 8 \
    --lora-alpha 16 \
    --lr 1e-4 \
    --epochs 3 \
    --num-samples 256 \
    --max-length 256 \
    --nim-max-tokens 256 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Completed at $(date)" | tee -a "$LOG"
