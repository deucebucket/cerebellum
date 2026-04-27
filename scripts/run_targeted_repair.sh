#!/bin/bash
# Train TARGETED repair LoRA — only patches damaged weights.
# Uses sensitivity-guided config: 86/248 modules targeted, variable rank.
# Reuses training data from full repair run (cached in repair-lora/).
# Run detached: nohup bash scripts/run_targeted_repair.sh &

set -e
cd /var/home/deucebucket/ai-drive/osmosis

MODEL_PATH=".hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OUTPUT="osmosis-qwen3.5-9b/repair-lora-targeted"
TENSOR_TYPES="osmosis-qwen3.5-9b/tensor_types_multidepth_4gb.txt"
TARGETED_CONFIG="osmosis-qwen3.5-9b/targeted_lora_config.json"
NIM_CARD="carl"
LOG="osmosis-qwen3.5-9b/repair_targeted.log"

echo "Starting TARGETED repair LoRA at $(date)" | tee "$LOG"
echo "Targets: 86/248 modules, variable rank (r=4/r=8)" | tee -a "$LOG"

# Copy training data cache from full run if available
if [ -f "osmosis-qwen3.5-9b/repair-lora/training_data.json" ]; then
    mkdir -p "$OUTPUT"
    cp "osmosis-qwen3.5-9b/repair-lora/training_data.json" "$OUTPUT/"
    echo "Reusing cached training data from full run" | tee -a "$LOG"
fi

python -m osmosis.repair_nim \
    --model "$MODEL_PATH" \
    --output "$OUTPUT" \
    --tensor-types "$TENSOR_TYPES" \
    --targeted-lora-config "$TARGETED_CONFIG" \
    --nim-card "$NIM_CARD" \
    --domain repair \
    --lr 1e-4 \
    --epochs 3 \
    --num-samples 256 \
    --max-length 256 \
    --nim-max-tokens 256 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Completed at $(date)" | tee -a "$LOG"
