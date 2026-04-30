#!/bin/bash
# Cerebellum ablation sweep for Gemma 4 E4B
# Tests each non-PLE tensor by crushing it to Q2_K while keeping everything else at baseline
# Baseline: Q3_K_M + PLE Q5_K (PPL 55.10)
set -e
cd /var/home/deucebucket/ai-drive/osmosis

export LD_LIBRARY_PATH="/var/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/cuda_runtime/lib:/var/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/cublas/lib:/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

QUANTIZE="/var/home/deucebucket/ai-drive/llama.cpp/build-cpu/bin/llama-quantize"
PPL="/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity"
SOURCE="/var/home/deucebucket/games/models/gemma-4-E4B-it-bf16.gguf"
IMATRIX="osmosis-gemma4-e4b/imatrix.dat"
PLE_BASE="osmosis-gemma4-e4b/ple_overrides_Q5K.txt"
WIKI="/var/home/deucebucket/games/osmosis-quants/wiki.test.raw"
OUTDIR="osmosis-gemma4-e4b/ablation"
TMPGGUF="/tmp/gemma4_ablation_test.gguf"

mkdir -p "$OUTDIR"

# Tensors to test — sample from different layers and components
# Focus on Q3_K tensors (most to gain from demotion or promotion)
TENSORS=(
    "blk.0.ffn_gate.weight"
    "blk.0.ffn_up.weight"
    "blk.1.ffn_gate.weight"
    "blk.1.ffn_up.weight"
    "blk.5.ffn_gate.weight"
    "blk.10.ffn_gate.weight"
    "blk.10.ffn_up.weight"
    "blk.15.ffn_gate.weight"
    "blk.20.ffn_gate.weight"
    "blk.20.ffn_up.weight"
    "blk.25.ffn_gate.weight"
    "blk.30.ffn_gate.weight"
    "blk.30.ffn_up.weight"
    "blk.35.ffn_gate.weight"
    "blk.40.ffn_gate.weight"
    "blk.40.ffn_up.weight"
    "blk.41.ffn_gate.weight"
    "blk.41.ffn_up.weight"
    "blk.0.attn_q.weight"
    "blk.0.attn_k.weight"
    "blk.10.attn_q.weight"
    "blk.20.attn_q.weight"
    "blk.30.attn_q.weight"
    "blk.40.attn_q.weight"
    "blk.41.attn_q.weight"
    "blk.41.attn_v.weight"
)

echo "=== CEREBELLUM ABLATION SWEEP — Gemma 4 E4B ==="
echo "Baseline: Q3_K_M + PLE Q5_K (PPL 55.10)"
echo "Test: crush one tensor to Q2_K, measure PPL delta"
echo "Tensors to test: ${#TENSORS[@]}"
echo "Start: $(date)"
echo ""

RESULTS_FILE="$OUTDIR/ablation_results.json"
echo '{"baseline_ppl": 55.10, "tests": {' > "$RESULTS_FILE"
FIRST=true

for tensor in "${TENSORS[@]}"; do
    label=$(echo "$tensor" | sed 's/\.weight$//' | sed 's/\./_/g')
    logfile="$OUTDIR/ablation_${label}.log"

    echo "============================================"
    echo "TESTING: $tensor"
    echo "START: $(date)"

    # Create override file: PLE at Q5_K + this tensor at Q2_K
    cat "$PLE_BASE" > /tmp/ablation_overrides.txt
    echo "${tensor}=Q2_K" >> /tmp/ablation_overrides.txt

    # Quantize
    $QUANTIZE --imatrix "$IMATRIX" \
        --tensor-type-file /tmp/ablation_overrides.txt \
        "$SOURCE" "$TMPGGUF" Q3_K_M > "$logfile" 2>&1

    # Measure PPL
    $PPL --model "$TMPGGUF" --file "$WIKI" -ngl 99 --ctx-size 2048 --chunks -1 >> "$logfile" 2>&1

    # Extract result
    ppl=$(grep "Final estimate" "$logfile" | tail -1 | grep -oP 'PPL = \K[0-9.]+')
    delta=$(python3 -c "print(f'{${ppl} - 55.10:.4f}')")

    echo "  PPL: $ppl (delta: $delta)"
    echo "DONE: $(date)"
    echo ""

    # Append to results JSON
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> "$RESULTS_FILE"
    fi
    printf '  "%s": {"ppl": %s, "gguf_tensor": "%s"}' "$label" "$ppl" "$tensor" >> "$RESULTS_FILE"

    # Clean up
    rm -f "$TMPGGUF"
done

echo '}' >> "$RESULTS_FILE"
echo '}' >> "$RESULTS_FILE"

echo ""
echo "=== ABLATION COMPLETE $(date) ==="
echo "Results: $RESULTS_FILE"
echo "ABLATION_DONE"
