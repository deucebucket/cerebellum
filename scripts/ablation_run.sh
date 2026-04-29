#!/usr/bin/env bash
# Tensor ablation experiment — crush one tensor at a time, measure real impact.
#
# Start from Q4_K_M baseline (high quality), override ONE tensor to Q2_K,
# run perplexity, record the delta. Proves which tensors actually matter
# vs what the sensitivity proxy predicts.
#
# Resumable: checks ablation_results.json for completed tests and skips them.
#
# Usage:
#   ./scripts/ablation_run.sh              # run all remaining
#   ./scripts/ablation_run.sh 5            # run next 5 only (good for overnight batches)
#   ./scripts/ablation_run.sh --baseline   # just build and test the baseline

set -euo pipefail

OSMOSIS_DIR="/var/home/deucebucket/ai-drive/osmosis"
QUANTS_DIR="/var/home/deucebucket/games/osmosis-quants"
F16_GGUF="/var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf"
IMATRIX="$OSMOSIS_DIR/osmosis-qwen36-27b/osmosis_imatrix.dat"
QUANTIZE="/tmp/llama-cpu-build/bin/llama-quantize"
WIKI_RAW="$QUANTS_DIR/wiki.test.raw"
PLAN="$OSMOSIS_DIR/osmosis-qwen36-27b/ablation_plan.json"
RESULTS="$OSMOSIS_DIR/osmosis-qwen36-27b/ablation_results.json"
LOG_DIR="$OSMOSIS_DIR/osmosis-qwen36-27b/ablation_logs"
BASELINE_GGUF="$QUANTS_DIR/qwen3.6-27b-osmosis-imatrix-Q4_K_M.gguf"
ABLATION_GGUF="$QUANTS_DIR/qwen3.6-27b-ablation-temp.gguf"
TENSOR_TYPES_TMP="/var/tmp/osmosis-qwen36/ablation_tensor_type.txt"

# HF sensitivity name → GGUF tensor name mapping
declare -A HF_TO_GGUF=(
    ["linear_attn.in_proj_qkv"]="attn_qkv"
    ["linear_attn.in_proj_a"]="ssm_alpha"
    ["linear_attn.in_proj_b"]="ssm_beta"
    ["linear_attn.in_proj_z"]="attn_gate"
    ["linear_attn.out_proj"]="ssm_out"
    ["self_attn.q_proj"]="attn_q"
    ["self_attn.k_proj"]="attn_k"
    ["self_attn.v_proj"]="attn_v"
    ["self_attn.o_proj"]="attn_output"
    ["mlp.down_proj"]="ffn_down"
    ["mlp.gate_proj"]="ffn_gate"
    ["mlp.up_proj"]="ffn_up"
)

mkdir -p "$LOG_DIR"

# Initialize results file if missing
if [ ! -f "$RESULTS" ]; then
    echo '{"baseline_ppl": null, "tests": {}}' > "$RESULTS"
fi

hf_to_gguf_name() {
    # Convert "layer_29.linear_attn.in_proj_b" → "blk.29.ssm_beta.weight"
    local hf_name="$1"
    local layer_num=$(echo "$hf_name" | grep -oP 'layer_\K\d+')
    local component=$(echo "$hf_name" | sed 's/layer_[0-9]*\.//')
    local gguf_comp="${HF_TO_GGUF[$component]:-}"
    if [ -z "$gguf_comp" ]; then
        echo "ERROR: unknown component $component" >&2
        return 1
    fi
    echo "blk.${layer_num}.${gguf_comp}.weight"
}

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
with open('$RESULTS') as f:
    r = json.load(f)
v = r.get('tests', {}).get('$1', {}).get('ppl')
if v: print(v)
"
}

save_result() {
    local name="$1"
    local ppl="$2"
    local gguf_tensor="$3"
    python3 -c "
import json
with open('$RESULTS') as f:
    r = json.load(f)
r.setdefault('tests', {})['$name'] = {'ppl': $ppl, 'gguf_tensor': '$gguf_tensor'}
with open('$RESULTS', 'w') as f:
    json.dump(r, f, indent=2)
"
}

save_baseline() {
    python3 -c "
import json
with open('$RESULTS') as f:
    r = json.load(f)
r['baseline_ppl'] = $1
with open('$RESULTS', 'w') as f:
    json.dump(r, f, indent=2)
"
}

get_baseline() {
    python3 -c "
import json
with open('$RESULTS') as f:
    r = json.load(f)
v = r.get('baseline_ppl')
if v: print(v)
"
}

# --- Step 1: Baseline Q4_K_M ---
BASELINE_PPL=$(get_baseline)
if [ -z "$BASELINE_PPL" ]; then
    echo "=== Building Q4_K_M baseline ==="
    if [ ! -f "$BASELINE_GGUF" ]; then
        echo "Quantizing F16 → Q4_K_M with imatrix..."
        "$QUANTIZE" --imatrix "$IMATRIX" "$F16_GGUF" "$BASELINE_GGUF" Q4_K_M
    fi
    echo "Running baseline perplexity..."
    BASELINE_PPL=$(run_perplexity "$BASELINE_GGUF" "$LOG_DIR/baseline_q4km.log")
    if [ -z "$BASELINE_PPL" ]; then
        echo "ERROR: Failed to get baseline PPL"
        exit 1
    fi
    save_baseline "$BASELINE_PPL"
    echo "Baseline PPL: $BASELINE_PPL"
else
    echo "Baseline PPL: $BASELINE_PPL (cached)"
fi

if [ "${1:-}" = "--baseline" ]; then
    echo "Baseline only mode, exiting."
    exit 0
fi

# --- Step 2: Run ablations ---
MAX_TESTS="${1:-999}"
COMPLETED=0

# Read test tensors from plan
TENSOR_NAMES=$(python3 -c "
import json
with open('$PLAN') as f:
    plan = json.load(f)
for t in plan['tensors']:
    print(t['name'])
")

for HF_NAME in $TENSOR_NAMES; do
    if [ "$COMPLETED" -ge "$MAX_TESTS" ]; then
        echo "Reached batch limit ($MAX_TESTS). Resume later."
        break
    fi

    # Skip if already done
    EXISTING=$(get_result "$HF_NAME")
    if [ -n "$EXISTING" ]; then
        echo "SKIP $HF_NAME (PPL=$EXISTING)"
        continue
    fi

    GGUF_TENSOR=$(hf_to_gguf_name "$HF_NAME")
    echo ""
    echo "=== Ablation: $HF_NAME → $GGUF_TENSOR =Q2_K ==="

    # Write single-tensor override file
    echo "${GGUF_TENSOR}=q2_K" > "$TENSOR_TYPES_TMP"

    # Build GGUF with one tensor crushed
    echo "Building GGUF (Q4_K_M base, $GGUF_TENSOR → Q2_K)..."
    "$QUANTIZE" --imatrix "$IMATRIX" \
        --tensor-type-file "$TENSOR_TYPES_TMP" \
        "$F16_GGUF" "$ABLATION_GGUF" Q4_K_M 2>&1 | tail -3

    # Run perplexity
    echo "Running perplexity..."
    PPL=$(run_perplexity "$ABLATION_GGUF" "$LOG_DIR/ablation_${HF_NAME}.log")

    if [ -z "$PPL" ]; then
        echo "ERROR: Failed to get PPL for $HF_NAME"
        continue
    fi

    DELTA=$(python3 -c "print(f'{$PPL - $BASELINE_PPL:+.4f}')")
    echo "PPL = $PPL (delta = $DELTA from baseline $BASELINE_PPL)"

    save_result "$HF_NAME" "$PPL" "$GGUF_TENSOR"
    COMPLETED=$((COMPLETED + 1))

    # Clean up temp GGUF to save disk
    rm -f "$ABLATION_GGUF"
done

echo ""
echo "=== Ablation Summary ==="
python3 -c "
import json

with open('$RESULTS') as f:
    r = json.load(f)

with open('$PLAN') as f:
    plan = json.load(f)

baseline = r['baseline_ppl']
tests = r.get('tests', {})
plan_lookup = {t['name']: t for t in plan['tensors']}

print(f'Baseline Q4_K_M: {baseline}')
print(f'Completed: {len(tests)}/{len(plan[\"tensors\"])}')
print()
print(f'{\"Tensor\":45s} {\"PPL\":>8s} {\"Delta\":>8s} {\"kl2_proxy\":>10s} {\"Params\":>8s}  Reason')
print('-' * 110)

for name, data in sorted(tests.items(), key=lambda x: x[1]['ppl'], reverse=True):
    ppl = data['ppl']
    delta = ppl - baseline
    info = plan_lookup.get(name, {})
    kl2 = info.get('kl2', 0)
    pc = info.get('pc', 0)
    reason = info.get('reason', '?')
    print(f'{name:45s} {ppl:8.4f} {delta:+8.4f} {kl2:10.6f} {pc/1e6:7.1f}M  {reason}')
"
