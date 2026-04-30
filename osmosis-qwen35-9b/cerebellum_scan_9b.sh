#!/bin/bash
# Cerebellum per-tensor ablation scan for Qwen 3.5 9B
# Crushes each tensor individually from Q4_K_M to Q2_K, measures PPL delta
# Output: ablation_results.json in Cerebellum format
#
# Runs overnight — ~200 tensors × ~2 min each = ~7 hours
# All temp files on games drive (361GB free)

set -euo pipefail

F16_GGUF="/var/home/deucebucket/games/qwen3.5-9b-f16.gguf"
WORK_DIR="/var/home/deucebucket/games/osmosis-quants/cerebellum-9b-scan"
WIKI_DATA="/var/home/deucebucket/games/osmosis-quants/wiki.test.raw"
QUANTIZE_BIN="/var/home/deucebucket/ai-drive/llama.cpp/build-cpu/bin/llama-quantize"
PPL_BIN="/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity"
RESULTS_FILE="$WORK_DIR/ablation_results.json"
LOG_DIR="$WORK_DIR/logs"

mkdir -p "$WORK_DIR" "$LOG_DIR"

# Get list of quantizable tensors (skip f32, skip tiny ssm_alpha/beta)
get_tensors() {
    $QUANTIZE_BIN --dry-run "$F16_GGUF" /dev/null Q4_K_M 2>&1 \
        | grep -E '^\[.*\].*type = ' \
        | grep -v 'f32' \
        | sed 's/.*] //' \
        | awk '{print $1}' \
        | grep -v 'ssm_alpha\|ssm_beta'
}

# Measure PPL of a GGUF file
measure_ppl() {
    local gguf="$1"
    local logfile="$2"
    distrobox enter ai -- "$PPL_BIN" \
        -m "$gguf" \
        -f "$WIKI_DATA" \
        -ngl 99 \
        --ctx-size 2048 \
        2>&1 | tee "$logfile" \
        | grep -oP 'PPL = \K[0-9]+\.[0-9]+' | tail -1
}

# Step 1: Create Q4_K_M baseline if not exists
BASELINE_GGUF="$WORK_DIR/qwen3.5-9b-Q4_K_M-baseline.gguf"
if [ ! -f "$BASELINE_GGUF" ]; then
    echo "=== Creating Q4_K_M baseline ==="
    $QUANTIZE_BIN "$F16_GGUF" "$BASELINE_GGUF" Q4_K_M 8
    echo "Baseline created: $(ls -lh "$BASELINE_GGUF" | awk '{print $5}')"
fi

# Step 2: Measure baseline PPL if not recorded
if [ -f "$RESULTS_FILE" ]; then
    BASELINE_PPL=$(python3 -c "import json; print(json.load(open('$RESULTS_FILE'))['baseline_ppl'])" 2>/dev/null || echo "")
else
    BASELINE_PPL=""
fi

if [ -z "$BASELINE_PPL" ]; then
    echo "=== Measuring baseline PPL ==="
    BASELINE_PPL=$(measure_ppl "$BASELINE_GGUF" "$LOG_DIR/baseline_ppl.log")
    echo "Baseline PPL: $BASELINE_PPL"
    python3 -c "
import json
results = {'baseline_ppl': float('$BASELINE_PPL'), 'tests': {}}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)
"
    echo "Initialized results file"
else
    echo "=== Baseline PPL already measured: $BASELINE_PPL ==="
fi

# Step 3: Per-tensor ablation scan
echo "=== Starting per-tensor ablation scan ==="
TENSORS=$(get_tensors)
TOTAL=$(echo "$TENSORS" | wc -l)
DONE=0
SKIPPED=0

for TENSOR in $TENSORS; do
    DONE=$((DONE + 1))

    # Strip .weight suffix for the HF-style name used as key
    TENSOR_KEY=$(echo "$TENSOR" | sed 's/\.weight$//')

    # Check if already scanned
    ALREADY=$(python3 -c "
import json
r = json.load(open('$RESULTS_FILE'))
print('yes' if '$TENSOR_KEY' in r.get('tests', {}) else 'no')
" 2>/dev/null || echo "no")

    if [ "$ALREADY" = "yes" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$DONE/$TOTAL] SKIP $TENSOR_KEY (already scanned)"
        continue
    fi

    echo "[$DONE/$TOTAL] Scanning $TENSOR_KEY ..."

    # Quantize with this one tensor crushed to Q2_K
    ABLATED_GGUF="$WORK_DIR/ablated_temp.gguf"
    $QUANTIZE_BIN \
        --tensor-type "$TENSOR_KEY=q2_K" \
        "$F16_GGUF" "$ABLATED_GGUF" Q4_K_M 8 \
        2>"$LOG_DIR/quant_${TENSOR_KEY}.log"

    # Measure PPL
    PPL=$(measure_ppl "$ABLATED_GGUF" "$LOG_DIR/ppl_${TENSOR_KEY}.log")

    if [ -z "$PPL" ]; then
        echo "  ERROR: Failed to get PPL for $TENSOR_KEY"
        rm -f "$ABLATED_GGUF"
        continue
    fi

    # Record result
    python3 -c "
import json
r = json.load(open('$RESULTS_FILE'))
r['tests']['$TENSOR_KEY'] = {
    'ppl': float('$PPL'),
    'gguf_tensor': '$TENSOR_KEY'
}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(r, f, indent=2)
"

    DELTA=$(python3 -c "print(f'{float(\"$PPL\") - float(\"$BASELINE_PPL\"):+.4f}')")
    echo "  PPL=$PPL (delta=$DELTA)"

    # Clean up temp file
    rm -f "$ABLATED_GGUF"
done

echo ""
echo "=== Scan complete ==="
echo "Total: $TOTAL, Scanned: $((DONE - SKIPPED)), Skipped: $SKIPPED"
echo "Results: $RESULTS_FILE"
