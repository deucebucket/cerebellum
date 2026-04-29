#!/usr/bin/env bash
# Interaction effects test — crush MULTIPLE tensors simultaneously.
# Tests whether single-tensor ablation deltas are additive or nonlinear.
#
# Requires: completed single-tensor ablation sweep (ablation_results.json)
#
# Tests:
#   1. Crush all "demote" tensors (negative delta) at once
#   2. Escalation ladder — add safe tensors one at a time, measure cumulative PPL
#   3. Full-layer crush — all tensors in one layer to Q2_K
#
# Usage:
#   ./scripts/interaction_test.sh              # run all tests
#   ./scripts/interaction_test.sh additive     # just the additive test
#   ./scripts/interaction_test.sh ladder       # just the escalation ladder
#   ./scripts/interaction_test.sh layer        # just the full-layer test

set -euo pipefail

OSMOSIS_DIR="/var/home/deucebucket/ai-drive/osmosis"
QUANTS_DIR="/var/home/deucebucket/games/osmosis-quants"
F16_GGUF="/var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf"
IMATRIX="$OSMOSIS_DIR/osmosis-qwen36-27b/osmosis_imatrix.dat"
QUANTIZE="/var/home/deucebucket/ai-drive/llama.cpp/build-cpu/bin/llama-quantize"
WIKI_RAW="$QUANTS_DIR/wiki.test.raw"
ABLATION_RESULTS="$OSMOSIS_DIR/osmosis-qwen36-27b/ablation_results.json"
INTERACTION_RESULTS="$OSMOSIS_DIR/osmosis-qwen36-27b/interaction_results.json"
LOG_DIR="$OSMOSIS_DIR/osmosis-qwen36-27b/interaction_logs"
TEMP_GGUF="$QUANTS_DIR/qwen3.6-27b-interaction-temp.gguf"
TENSOR_TYPES_TMP="/var/tmp/osmosis-qwen36/interaction_tensor_types.txt"

mkdir -p "$LOG_DIR"

if [ ! -f "$INTERACTION_RESULTS" ]; then
    echo '{"tests": {}}' > "$INTERACTION_RESULTS"
fi

run_perplexity() {
    local gguf="$1"
    local log="$2"
    distrobox enter ai -- bash -c \
        "/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity \
        -m $gguf -ngl 99 -c 2048 -t 8 \
        -f $WIKI_RAW 2>&1" \
        2>/dev/null | grep -v 'nvidia-modprobe' > "$log"
    grep -oP 'PPL = \K[\d.]+' "$log" | tail -1
}

get_result() {
    python3 -c "
import json
with open('$INTERACTION_RESULTS') as f:
    r = json.load(f)
v = r.get('tests', {}).get('$1', {}).get('ppl')
if v: print(v)
"
}

save_result() {
    local name="$1"
    local ppl="$2"
    local tensor_count="$3"
    local expected_delta="$4"
    python3 -c "
import json
with open('$INTERACTION_RESULTS') as f:
    r = json.load(f)
baseline = 8.2556
actual_delta = $ppl - baseline
r.setdefault('tests', {})['$name'] = {
    'ppl': $ppl,
    'tensor_count': $tensor_count,
    'expected_delta': $expected_delta,
    'actual_delta': actual_delta,
    'interaction_ratio': actual_delta / $expected_delta if abs($expected_delta) > 0.001 else None
}
with open('$INTERACTION_RESULTS', 'w') as f:
    json.dump(r, f, indent=2)
"
}

# Get baseline PPL from ablation results
BASELINE_PPL=$(python3 -c "
import json
with open('$ABLATION_RESULTS') as f:
    print(json.load(f)['baseline_ppl'])
")
echo "Baseline PPL: $BASELINE_PPL"

# --- Test 1: Crush all demote tensors at once ---
run_additive_test() {
    local TEST_NAME="additive_all_demote"
    EXISTING=$(get_result "$TEST_NAME")
    if [ -n "$EXISTING" ]; then
        echo "SKIP $TEST_NAME (PPL=$EXISTING)"
        return
    fi

    echo ""
    echo "=== Test: Crush all demote tensors simultaneously ==="

    # Get demote tensors and expected sum of deltas
    python3 -c "
import json
with open('$ABLATION_RESULTS') as f:
    r = json.load(f)
baseline = r['baseline_ppl']
demote = []
for name, data in r['tests'].items():
    delta = data['ppl'] - baseline
    if delta < -0.02:
        demote.append((data['gguf_tensor'], delta))
        print(f'  {data[\"gguf_tensor\"]:45s} delta={delta:+.4f}')

# Write tensor type file
with open('$TENSOR_TYPES_TMP', 'w') as f:
    for tensor, _ in demote:
        f.write(f'{tensor}=q2_K\n')

expected = sum(d for _, d in demote)
print(f'Expected sum of deltas: {expected:+.4f}')
print(f'TENSOR_COUNT={len(demote)}')
print(f'EXPECTED_DELTA={expected:.4f}')
" | tee /dev/stderr | tail -2 > /tmp/interaction_vars.txt

    TENSOR_COUNT=$(grep TENSOR_COUNT /tmp/interaction_vars.txt | cut -d= -f2)
    EXPECTED_DELTA=$(grep EXPECTED_DELTA /tmp/interaction_vars.txt | cut -d= -f2)

    echo "Building GGUF with $TENSOR_COUNT tensors crushed to Q2_K..."
    "$QUANTIZE" --imatrix "$IMATRIX" \
        --tensor-type-file "$TENSOR_TYPES_TMP" \
        "$F16_GGUF" "$TEMP_GGUF" Q4_K_M 2>&1 | tail -3

    echo "Running perplexity..."
    PPL=$(run_perplexity "$TEMP_GGUF" "$LOG_DIR/interaction_${TEST_NAME}.log")

    if [ -z "$PPL" ]; then
        echo "ERROR: Failed to get PPL"
        return
    fi

    ACTUAL_DELTA=$(python3 -c "print(f'{$PPL - $BASELINE_PPL:+.4f}')")
    echo "PPL = $PPL (actual delta = $ACTUAL_DELTA, expected = $EXPECTED_DELTA)"

    save_result "$TEST_NAME" "$PPL" "$TENSOR_COUNT" "$EXPECTED_DELTA"
    rm -f "$TEMP_GGUF"
}

# --- Test 2: Escalation ladder ---
run_ladder_test() {
    echo ""
    echo "=== Escalation ladder: cumulative tensor crushing ==="

    # Get tensors sorted by delta (most negative first = safest)
    LADDER=$(python3 -c "
import json
with open('$ABLATION_RESULTS') as f:
    r = json.load(f)
baseline = r['baseline_ppl']
tensors = []
for name, data in r['tests'].items():
    delta = data['ppl'] - baseline
    if delta < -0.02:
        tensors.append((data['gguf_tensor'], delta))
tensors.sort(key=lambda x: x[1])
for t, d in tensors:
    print(f'{t}|{d:.4f}')
")

    CUMULATIVE_TENSORS=""
    CUMULATIVE_DELTA=0
    STEP=0

    for ENTRY in $LADDER; do
        TENSOR=$(echo "$ENTRY" | cut -d'|' -f1)
        DELTA=$(echo "$ENTRY" | cut -d'|' -f2)
        STEP=$((STEP + 1))
        CUMULATIVE_DELTA=$(python3 -c "print(f'{$CUMULATIVE_DELTA + $DELTA:.4f}')")

        TEST_NAME="ladder_step_${STEP}"
        EXISTING=$(get_result "$TEST_NAME")
        if [ -n "$EXISTING" ]; then
            echo "SKIP $TEST_NAME (PPL=$EXISTING)"
            continue
        fi

        if [ -z "$CUMULATIVE_TENSORS" ]; then
            CUMULATIVE_TENSORS="$TENSOR"
        else
            CUMULATIVE_TENSORS="$CUMULATIVE_TENSORS $TENSOR"
        fi

        echo ""
        echo "--- Ladder step $STEP: +$TENSOR (cumulative: $STEP tensors) ---"

        # Write tensor type file
        > "$TENSOR_TYPES_TMP"
        for T in $CUMULATIVE_TENSORS; do
            echo "${T}=q2_K" >> "$TENSOR_TYPES_TMP"
        done

        "$QUANTIZE" --imatrix "$IMATRIX" \
            --tensor-type-file "$TENSOR_TYPES_TMP" \
            "$F16_GGUF" "$TEMP_GGUF" Q4_K_M 2>&1 | tail -3

        PPL=$(run_perplexity "$TEMP_GGUF" "$LOG_DIR/interaction_${TEST_NAME}.log")

        if [ -z "$PPL" ]; then
            echo "ERROR: Failed to get PPL for step $STEP"
            continue
        fi

        ACTUAL_DELTA=$(python3 -c "print(f'{$PPL - $BASELINE_PPL:+.4f}')")
        echo "PPL = $PPL (actual = $ACTUAL_DELTA, expected sum = $CUMULATIVE_DELTA)"

        save_result "$TEST_NAME" "$PPL" "$STEP" "$CUMULATIVE_DELTA"
        rm -f "$TEMP_GGUF"
    done
}

# --- Test 3: Full layer crush ---
run_layer_test() {
    echo ""
    echo "=== Full layer crush: all tensors in one layer to Q2_K ==="

    for LAYER in 34 0 63; do
        TEST_NAME="full_layer_${LAYER}"
        EXISTING=$(get_result "$TEST_NAME")
        if [ -n "$EXISTING" ]; then
            echo "SKIP $TEST_NAME (PPL=$EXISTING)"
            continue
        fi

        echo ""
        echo "--- Crushing ALL tensors in layer $LAYER to Q2_K ---"

        # Generate all tensor names for this layer
        python3 -c "
tensors = [
    'attn_qkv', 'attn_gate', 'attn_norm', 'ssm_alpha', 'ssm_beta',
    'ssm_out', 'ffn_down', 'ffn_gate', 'ffn_up', 'post_attention_norm',
]
if $LAYER == 63:
    tensors = [
        'attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_q_norm',
        'attn_k_norm', 'attn_norm', 'ffn_down', 'ffn_gate', 'ffn_up',
        'post_attention_norm',
    ]
with open('$TENSOR_TYPES_TMP', 'w') as f:
    for t in tensors:
        f.write(f'blk.${LAYER}.{t}.weight=q2_K\n')
print(f'Crushing {len(tensors)} tensors in layer $LAYER')
"

        "$QUANTIZE" --imatrix "$IMATRIX" \
            --tensor-type-file "$TENSOR_TYPES_TMP" \
            "$F16_GGUF" "$TEMP_GGUF" Q4_K_M 2>&1 | tail -3

        PPL=$(run_perplexity "$TEMP_GGUF" "$LOG_DIR/interaction_${TEST_NAME}.log")

        if [ -z "$PPL" ]; then
            echo "ERROR: Failed to get PPL for layer $LAYER"
            continue
        fi

        ACTUAL_DELTA=$(python3 -c "print(f'{$PPL - $BASELINE_PPL:+.4f}')")
        echo "Layer $LAYER full crush: PPL = $PPL (delta = $ACTUAL_DELTA)"

        save_result "$TEST_NAME" "$PPL" "1" "0"
        rm -f "$TEMP_GGUF"
    done
}

# --- Run requested tests ---
MODE="${1:-all}"
case "$MODE" in
    additive) run_additive_test ;;
    ladder)   run_ladder_test ;;
    layer)    run_layer_test ;;
    all)
        run_additive_test
        run_ladder_test
        run_layer_test
        ;;
    *)
        echo "Usage: $0 [additive|ladder|layer|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Interaction Test Summary ==="
python3 -c "
import json
with open('$INTERACTION_RESULTS') as f:
    r = json.load(f)
tests = r.get('tests', {})
print(f'Completed: {len(tests)} tests')
print()
print(f'{\"Test\":30s} {\"PPL\":>8s} {\"Actual\":>8s} {\"Expected\":>8s} {\"Ratio\":>8s}')
print('-' * 80)
for name, data in sorted(tests.items()):
    ppl = data['ppl']
    actual = data['actual_delta']
    expected = data['expected_delta']
    ratio = data.get('interaction_ratio')
    ratio_str = f'{ratio:.2f}' if ratio else 'N/A'
    print(f'{name:30s} {ppl:8.4f} {actual:+8.4f} {expected:+8.4f} {ratio_str:>8s}')
print()
print('Ratio interpretation:')
print('  ~1.0 = deltas are additive (no interaction)')
print('  >1.0 = compounding (worse than expected)')
print('  <1.0 = cancellation (better than expected)')
"
