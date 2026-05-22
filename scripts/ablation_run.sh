#!/usr/bin/env bash
# Generic Cerebellum tensor ablation helper.
#
# This is a thin, configurable wrapper around stock llama.cpp tools. It builds
# one temporary GGUF per tensor override, runs perplexity, records the measured
# PPL, and deletes the temporary GGUF.
#
# Required environment variables:
#   SOURCE_GGUF       high precision source GGUF
#   IMATRIX           llama.cpp imatrix file
#   PLAN_JSON         JSON file with {"tensors": [{"name": "...", "gguf_tensor": "..."}]}
#   RESULTS_JSON      output ablation results JSON
#   WORK_DIR          temporary output directory
#   WIKI_RAW          perplexity corpus file
#
# Optional environment variables:
#   QUANTIZE          path to llama-quantize (default: llama-quantize)
#   PERPLEXITY        path to llama-perplexity (default: llama-perplexity)
#   BASE_TYPE         baseline quant type (default: Q4_K_M)
#   ABLATE_TYPE       forced tensor quant type (default: q2_K)
#   N_GPU_LAYERS      llama.cpp -ngl value (default: 99)
#   CTX_SIZE          perplexity context size (default: 2048)
#   THREADS           perplexity thread count (default: 8)
#   MAX_TESTS         max new tensors to run in this invocation (default: all)
#
# Usage:
#   SOURCE_GGUF=model-f16.gguf \
#   IMATRIX=cerebellum_imatrix.dat \
#   PLAN_JSON=ablation_plan.json \
#   RESULTS_JSON=ablation_results.json \
#   WORK_DIR=/tmp/cerebellum-ablation \
#   WIKI_RAW=wiki.test.raw \
#   ./scripts/ablation_run.sh

set -euo pipefail

required_var() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "ERROR: required environment variable $name is not set" >&2
        exit 2
    fi
}

for name in SOURCE_GGUF IMATRIX PLAN_JSON RESULTS_JSON WORK_DIR WIKI_RAW; do
    required_var "$name"
done

QUANTIZE="${QUANTIZE:-llama-quantize}"
PERPLEXITY="${PERPLEXITY:-llama-perplexity}"
BASE_TYPE="${BASE_TYPE:-Q4_K_M}"
ABLATE_TYPE="${ABLATE_TYPE:-q2_K}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CTX_SIZE="${CTX_SIZE:-2048}"
THREADS="${THREADS:-8}"
MAX_TESTS="${MAX_TESTS:-999999}"

mkdir -p "$WORK_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

BASELINE_GGUF="$WORK_DIR/baseline-${BASE_TYPE}.gguf"
ABLATION_GGUF="$WORK_DIR/ablation-temp.gguf"
TENSOR_TYPES_TMP="$WORK_DIR/ablation_tensor_type.txt"

if [ ! -f "$RESULTS_JSON" ]; then
    printf '{"baseline_ppl": null, "tests": {}}\n' > "$RESULTS_JSON"
fi

json_get_baseline() {
    python3 - "$RESULTS_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
value = data.get("baseline_ppl")
if value is not None:
    print(value)
PY
}

json_test_done() {
    python3 - "$RESULTS_JSON" "$1" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
value = data.get("tests", {}).get(sys.argv[2], {}).get("ppl")
if value is not None:
    print(value)
PY
}

json_save_baseline() {
    python3 - "$RESULTS_JSON" "$1" <<'PY'
import json, sys
path, ppl = sys.argv[1], float(sys.argv[2])
with open(path) as f:
    data = json.load(f)
data["baseline_ppl"] = ppl
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
}

json_save_test() {
    python3 - "$RESULTS_JSON" "$1" "$2" "$3" <<'PY'
import json, sys
path, name, ppl, gguf_tensor = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
with open(path) as f:
    data = json.load(f)
data.setdefault("tests", {})[name] = {
    "ppl": ppl,
    "gguf_tensor": gguf_tensor,
}
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
}

run_perplexity() {
    local gguf="$1"
    local log="$2"
    "$PERPLEXITY" \
        -m "$gguf" \
        -ngl "$N_GPU_LAYERS" \
        -c "$CTX_SIZE" \
        -t "$THREADS" \
        -f "$WIKI_RAW" \
        > "$log" 2>&1
    grep -oE 'PPL = [0-9.]+' "$log" | awk '{print $3}' | tail -1
}

BASELINE_PPL="$(json_get_baseline)"
if [ -z "$BASELINE_PPL" ]; then
    echo "Building baseline $BASE_TYPE..."
    if [ ! -f "$BASELINE_GGUF" ]; then
        "$QUANTIZE" --imatrix "$IMATRIX" "$SOURCE_GGUF" "$BASELINE_GGUF" "$BASE_TYPE"
    fi

    echo "Running baseline perplexity..."
    BASELINE_PPL="$(run_perplexity "$BASELINE_GGUF" "$LOG_DIR/baseline.log")"
    if [ -z "$BASELINE_PPL" ]; then
        echo "ERROR: failed to parse baseline PPL from $LOG_DIR/baseline.log" >&2
        exit 1
    fi
    json_save_baseline "$BASELINE_PPL"
fi

echo "Baseline PPL: $BASELINE_PPL"

completed=0
python3 - "$PLAN_JSON" <<'PY' | while IFS=$'\t' read -r test_name gguf_tensor; do
import json, sys
with open(sys.argv[1]) as f:
    plan = json.load(f)
for item in plan.get("tensors", []):
    name = item.get("name") or item.get("hf_name") or item.get("gguf_tensor")
    gguf = item.get("gguf_tensor")
    if name and gguf:
        print(f"{name}\t{gguf}")
PY
    if [ "$completed" -ge "$MAX_TESTS" ]; then
        break
    fi

    existing="$(json_test_done "$test_name")"
    if [ -n "$existing" ]; then
        echo "SKIP $test_name (PPL=$existing)"
        continue
    fi

    echo ""
    echo "Ablating $test_name -> $gguf_tensor=$ABLATE_TYPE"
    printf '%s=%s\n' "$gguf_tensor" "$ABLATE_TYPE" > "$TENSOR_TYPES_TMP"

    "$QUANTIZE" \
        --imatrix "$IMATRIX" \
        --tensor-type-file "$TENSOR_TYPES_TMP" \
        "$SOURCE_GGUF" \
        "$ABLATION_GGUF" \
        "$BASE_TYPE"

    safe_name="$(printf '%s' "$test_name" | tr '/ .' '___')"
    ppl="$(run_perplexity "$ABLATION_GGUF" "$LOG_DIR/ablation-${safe_name}.log")"
    if [ -z "$ppl" ]; then
        echo "WARN: failed to parse PPL for $test_name" >&2
        rm -f "$ABLATION_GGUF"
        continue
    fi

    json_save_test "$test_name" "$ppl" "$gguf_tensor"
    python3 - "$ppl" "$BASELINE_PPL" <<'PY'
import sys
ppl, baseline = map(float, sys.argv[1:3])
print(f"PPL = {ppl:.6f} (delta {ppl - baseline:+.6f})")
PY

    rm -f "$ABLATION_GGUF"
    completed=$((completed + 1))
done

echo ""
echo "Results: $RESULTS_JSON"
