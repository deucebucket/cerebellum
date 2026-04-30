#!/bin/bash
# Cerebellum Full Pipeline — Gemma 4 26B-A4B-it (MoE)
# Downloads BF16, generates imatrix, finds fragile tensors, ablates, quantizes, benchmarks
set -e
cd /var/home/deucebucket/ai-drive/osmosis

export LD_LIBRARY_PATH="/var/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/cuda_runtime/lib:/var/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/cublas/lib:/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

QUANTIZE="/var/home/deucebucket/ai-drive/llama.cpp/build-cpu/bin/llama-quantize"
PPL="/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity"
GGUF_INFO="/var/home/deucebucket/ai-drive/llama.cpp/build-cpu/bin/llama-gguf-info"
WIKI="/var/home/deucebucket/games/osmosis-quants/wiki.test.raw"
WORKDIR="osmosis-gemma4-26b"
MODELS="/var/home/deucebucket/games/models"
SOURCE="$MODELS/gemma-4-26B-A4B-it-bf16.gguf"
TMPGGUF="/tmp/gemma4_26b_test.gguf"

mkdir -p "$WORKDIR/ablation"

echo "============================================================"
echo "CEREBELLUM PIPELINE — Gemma 4 26B-A4B-it (MoE)"
echo "START: $(date)"
echo "============================================================"

# ============================================================
# PHASE 1: Download BF16 GGUF
# ============================================================
echo ""
echo "=== PHASE 1: DOWNLOAD BF16 GGUF ==="
if [ -f "$SOURCE" ]; then
    echo "BF16 already exists: $(du -h "$SOURCE" | cut -f1)"
else
    echo "Downloading from ggml-org (~50 GB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    'ggml-org/gemma-4-26B-A4B-it-GGUF',
    'gemma-4-26B-A4B-it-bf16.gguf',
    local_dir='$MODELS',
    local_dir_use_symlinks=False
)
print(f'Downloaded to {path}')
"
    echo "DOWNLOAD_DONE $(date)"
    echo "Size: $(du -h "$SOURCE" | cut -f1)"
fi

# ============================================================
# PHASE 2: Identify tensor structure — find PLE/MoE tensors
# ============================================================
echo ""
echo "=== PHASE 2: TENSOR STRUCTURE ANALYSIS ==="
python3 -c "
import struct, sys

def read_gguf_tensor_names(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print('Not a GGUF file')
            return
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        # Skip KV pairs
        for _ in range(n_kv):
            # key
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8', errors='replace')
            # value type
            vtype = struct.unpack('<I', f.read(4))[0]
            # skip value based on type
            if vtype == 0:  # uint8
                f.read(1)
            elif vtype == 1:  # int8
                f.read(1)
            elif vtype == 2:  # uint16
                f.read(2)
            elif vtype == 3:  # int16
                f.read(2)
            elif vtype == 4:  # uint32
                f.read(4)
            elif vtype == 5:  # int32
                f.read(4)
            elif vtype == 6:  # float32
                f.read(4)
            elif vtype == 7:  # bool
                f.read(1)
            elif vtype == 8:  # string
                slen = struct.unpack('<Q', f.read(8))[0]
                f.read(slen)
            elif vtype == 9:  # array
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                for _ in range(alen):
                    if atype in (0, 1, 7):
                        f.read(1)
                    elif atype in (2, 3):
                        f.read(2)
                    elif atype in (4, 5, 6):
                        f.read(4)
                    elif atype == 8:
                        sl = struct.unpack('<Q', f.read(8))[0]
                        f.read(sl)
                    elif atype in (10, 12):
                        f.read(8)
                    elif atype == 11:
                        f.read(8)
            elif vtype == 10:  # uint64
                f.read(8)
            elif vtype == 11:  # int64
                f.read(8)
            elif vtype == 12:  # float64
                f.read(8)

        # Read tensor names
        names = []
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            for _ in range(n_dims):
                struct.unpack('<Q', f.read(8))[0]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            names.append(name)
        return names

names = read_gguf_tensor_names('$SOURCE')
if names:
    print(f'Total tensors: {len(names)}')
    with open('$WORKDIR/tensor_names.txt', 'w') as f:
        for n in names:
            f.write(n + '\n')

    # Categorize
    ple = [n for n in names if 'per_layer' in n or 'inp_gate' in n or '.proj.weight' in n]
    moe = [n for n in names if 'expert' in n]
    attn = [n for n in names if 'attn' in n]
    ffn = [n for n in names if 'ffn' in n and 'expert' not in n]

    print(f'PLE-like tensors: {len(ple)}')
    print(f'MoE expert tensors: {len(moe)}')
    print(f'Attention tensors: {len(attn)}')
    print(f'FFN (non-expert) tensors: {len(ffn)}')

    # Print PLE tensors
    if ple:
        print('\\n--- PLE TENSORS ---')
        for p in ple[:20]:
            print(f'  {p}')
        if len(ple) > 20:
            print(f'  ... ({len(ple)} total)')

    # Print sample expert tensors
    if moe:
        print('\\n--- SAMPLE EXPERT TENSORS ---')
        for m in moe[:10]:
            print(f'  {m}')
        print(f'  ... ({len(moe)} total)')
else:
    print('Failed to read tensors')
" 2>&1 | tee "$WORKDIR/tensor_analysis.log"
echo "TENSOR_ANALYSIS_DONE $(date)"

# ============================================================
# PHASE 3: Generate imatrix
# ============================================================
echo ""
echo "=== PHASE 3: GENERATE IMATRIX ==="
if [ -f "$WORKDIR/imatrix.dat" ]; then
    echo "Imatrix already exists"
else
    python3 -m osmosis.imatrix_stream \
        --model google/gemma-4-26B-A4B-it \
        --output "$WORKDIR/imatrix.dat" -v
    echo "IMATRIX_DONE $(date)"
fi

# ============================================================
# PHASE 4: Generate PLE overrides (if PLE tensors found)
# ============================================================
echo ""
echo "=== PHASE 4: PLE OVERRIDE GENERATION ==="
python3 -c "
with open('$WORKDIR/tensor_names.txt') as f:
    names = [l.strip() for l in f if l.strip()]

# Find PLE tensors — same patterns as E4B
ple_patterns = ['per_layer_token_embd', 'per_layer_model_proj', 'inp_gate', 'layer_output_scale', 'post_norm']
ple_tensors = []
for n in names:
    for p in ple_patterns:
        if p in n:
            ple_tensors.append(n)
            break
    # Also catch blk.X.proj.weight (PLE projection, not attn proj)
    if n.endswith('.proj.weight') and 'attn' not in n and 'ffn' not in n and 'expert' not in n:
        if n not in ple_tensors:
            ple_tensors.append(n)

if ple_tensors:
    print(f'Found {len(ple_tensors)} PLE tensors')
    for level in ['Q5_K', 'Q6_K', 'Q8_0']:
        fname = f'$WORKDIR/ple_overrides_{level}.txt'
        with open(fname, 'w') as f:
            for t in sorted(ple_tensors):
                f.write(f'{t}={level}\n')
        print(f'  Written: {fname}')
else:
    print('NO PLE TENSORS FOUND — this model uses MoE instead of PLE')
    print('Proceeding with standard Q3_K_M baseline')
    # Create empty override file
    open('$WORKDIR/ple_overrides_Q5_K.txt', 'w').close()
"
echo "PLE_OVERRIDES_DONE $(date)"

# ============================================================
# PHASE 5: Baseline quantization — Q3_K_M (no overrides)
# ============================================================
echo ""
echo "=== PHASE 5: BASELINE Q3_K_M ==="
BASELINE="$MODELS/gemma-4-26B-A4B-it-Q3_K_M.gguf"
if [ -f "$BASELINE" ]; then
    echo "Baseline already exists: $(du -h "$BASELINE" | cut -f1)"
else
    $QUANTIZE --imatrix "$WORKDIR/imatrix.dat" \
        "$SOURCE" "$BASELINE" Q3_K_M
    echo "BASELINE_QUANTIZE_DONE $(date)"
    echo "Size: $(du -h "$BASELINE" | cut -f1)"
fi

# Measure baseline PPL
echo "Measuring baseline PPL..."
$PPL --model "$BASELINE" --file "$WIKI" -ngl 99 --ctx-size 2048 --chunks -1 2>&1 | tee "$WORKDIR/ppl_baseline_q3km.log"
BASELINE_PPL=$(grep "Final estimate" "$WORKDIR/ppl_baseline_q3km.log" | tail -1 | grep -oP 'PPL = \K[0-9.]+')
echo "BASELINE_PPL: $BASELINE_PPL"
echo "BASELINE_PPL_DONE $(date)"

# ============================================================
# PHASE 6: PLE Protection Sweep (if PLE tensors exist)
# ============================================================
echo ""
echo "=== PHASE 6: PLE PROTECTION SWEEP ==="
PLE_COUNT=$(wc -l < "$WORKDIR/ple_overrides_Q5_K.txt" 2>/dev/null || echo "0")
if [ "$PLE_COUNT" -gt "0" ]; then
    echo "Testing PLE protection levels: Q5_K, Q6_K, Q8_0"
    for LEVEL in Q5_K Q6_K Q8_0; do
        OVERRIDE_FILE="$WORKDIR/ple_overrides_${LEVEL}.txt"
        OUTFILE="$MODELS/gemma-4-26B-A4B-it-Q3_K_M-ple-${LEVEL}.gguf"
        LOGFILE="$WORKDIR/ppl_q3km_ple_${LEVEL}.log"

        echo "--- Testing PLE $LEVEL ---"
        $QUANTIZE --imatrix "$WORKDIR/imatrix.dat" \
            --tensor-type-file "$OVERRIDE_FILE" \
            "$SOURCE" "$OUTFILE" Q3_K_M

        echo "Size: $(du -h "$OUTFILE" | cut -f1)"

        $PPL --model "$OUTFILE" --file "$WIKI" -ngl 99 --ctx-size 2048 --chunks -1 2>&1 | tee "$LOGFILE"
        PPL_VAL=$(grep "Final estimate" "$LOGFILE" | tail -1 | grep -oP 'PPL = \K[0-9.]+')
        echo "PLE_${LEVEL}_PPL: $PPL_VAL"
        echo "PLE_${LEVEL}_SIZE: $(du -h "$OUTFILE" | cut -f1)"
        echo "PLE_${LEVEL}_DONE $(date)"

        # Clean up intermediate quant if not the winner
        # (keep them for now, decide later)
    done

    # Find best PLE level
    echo ""
    echo "=== PLE SWEEP RESULTS ==="
    echo "Baseline Q3_K_M PPL: $BASELINE_PPL"
    for LEVEL in Q5_K Q6_K Q8_0; do
        LOGFILE="$WORKDIR/ppl_q3km_ple_${LEVEL}.log"
        PPL_VAL=$(grep "Final estimate" "$LOGFILE" | tail -1 | grep -oP 'PPL = \K[0-9.]+')
        SIZE=$(du -h "$MODELS/gemma-4-26B-A4B-it-Q3_K_M-ple-${LEVEL}.gguf" | cut -f1)
        echo "  PLE $LEVEL: PPL $PPL_VAL, Size $SIZE"
    done
    echo "PLE_SWEEP_DONE $(date)"
else
    echo "No PLE tensors — skipping PLE sweep"
    echo "Using baseline Q3_K_M as starting point for ablation"
fi

# ============================================================
# PHASE 7: Also get BF16 and Q4_K_M PPL for comparison
# ============================================================
echo ""
echo "=== PHASE 7: REFERENCE PPL (BF16, Q4_K_M) ==="

# BF16
echo "Measuring BF16 PPL..."
$PPL --model "$SOURCE" --file "$WIKI" -ngl 99 --ctx-size 2048 --chunks -1 2>&1 | tee "$WORKDIR/ppl_bf16.log"
BF16_PPL=$(grep "Final estimate" "$WORKDIR/ppl_bf16.log" | tail -1 | grep -oP 'PPL = \K[0-9.]+')
echo "BF16_PPL: $BF16_PPL"

# Q4_K_M
Q4KM="$MODELS/gemma-4-26B-A4B-it-Q4_K_M.gguf"
if [ ! -f "$Q4KM" ]; then
    $QUANTIZE --imatrix "$WORKDIR/imatrix.dat" \
        "$SOURCE" "$Q4KM" Q4_K_M
fi
echo "Q4_K_M Size: $(du -h "$Q4KM" | cut -f1)"
$PPL --model "$Q4KM" --file "$WIKI" -ngl 99 --ctx-size 2048 --chunks -1 2>&1 | tee "$WORKDIR/ppl_q4km.log"
Q4KM_PPL=$(grep "Final estimate" "$WORKDIR/ppl_q4km.log" | tail -1 | grep -oP 'PPL = \K[0-9.]+')
echo "Q4KM_PPL: $Q4KM_PPL"
echo "REFERENCE_PPL_DONE $(date)"

# ============================================================
# PHASE 8: Determine best baseline for ablation
# ============================================================
echo ""
echo "=== PHASE 8: BASELINE SELECTION ==="
echo "BF16: PPL $BF16_PPL"
echo "Q4_K_M: PPL $Q4KM_PPL"
echo "Q3_K_M: PPL $BASELINE_PPL"
if [ "$PLE_COUNT" -gt "0" ]; then
    for LEVEL in Q5_K Q6_K Q8_0; do
        PPL_VAL=$(grep "Final estimate" "$WORKDIR/ppl_q3km_ple_${LEVEL}.log" | tail -1 | grep -oP 'PPL = \K[0-9.]+')
        echo "Q3_K_M+PLE_${LEVEL}: PPL $PPL_VAL"
    done
fi
echo ""
echo "Selecting best baseline for ablation..."
# Use the best PLE level that's smaller than Q4_K_M, or plain Q3_K_M if no PLE
# This will be refined by reading the logs
echo "BASELINE_SELECTION_DONE $(date)"

echo ""
echo "============================================================"
echo "PIPELINE PHASES 1-8 COMPLETE $(date)"
echo "============================================================"
echo "Review results and run ablation sweep manually."
echo "PIPELINE_PHASE1_DONE"
