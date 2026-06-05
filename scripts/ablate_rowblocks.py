#!/usr/bin/env python3
"""Stage A row-block ablation for Cerebellum.

Per-tensor ablation tells us WHICH tensors are sensitive. This goes one
level deeper: for a single tensor T, measure how PPL changes when individual
row-blocks of T are at lower precision than the rest of the tensor.

Output is a per-row-block sensitivity map. Combined across the top-N
sensitive tensors from the per-tensor ablation, this is what enables
bidirectional reallocation: rescue critical rows in crushed tensors,
demote dead-weight rows in preserved tensors. Same file size, better
quality.

Stage A is measurement-only — the resulting GGUFs aren't shippable
without llama.cpp patches (Stage B). For Stage A, the target tensor T
is shipped at F16 in the test GGUF; only its numerical content changes
between iterations. This makes hybrid_T vs base_T an apples-to-apples
comparison — same byte layout, different fp16 values.

Setup phase (per (source, base_quant, low_quant) tuple — one-time):
  1. Quantize source at base_quant with T forced to F16 → template.gguf
  2. Quantize source at low_quant → q_low.gguf; dequantize T → low_T
  3. Quantize source at base_quant → q_base.gguf; dequantize T → base_T

Baseline:
  4. Patch base_T into template.gguf at T's data offset → measure PPL
     across all domains → ppl_baseline.

Per row-block i:
  5. hybrid_T = base_T.copy()
     hybrid_T[i*B:(i+1)*B] = low_T[i*B:(i+1)*B]
  6. Patch hybrid_T into template.gguf
  7. Measure PPL across domains → ppl_block_i, delta_block_i

Usage:
    python scripts/ablate_rowblocks.py \\
        --source-gguf /var/home/deucebucket/games/qwen3.5-9b-f16.gguf \\
        --imatrix osmosis-qwen35-9b/cerebellum_imatrix.dat \\
        --target-tensor blk.5.ffn_down.weight \\
        --base-quant Q3_K_M \\
        --low-quant Q2_K \\
        --block-size 128 \\
        --corpus-dir /var/home/deucebucket/games/cerebellum-calibration \\
        --output cerebellum-qwen35-9b/rowblock_blk5_ffndown.json
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import gguf
import gguf.quants

DEFAULT_DOMAINS = ["wiki", "code", "math", "dialogue"]
QUANTIZE_BIN = "/var/home/deucebucket/ai-drive/llama.cpp/build-cpu/bin/llama-quantize"
WORK_DIR = Path("/var/home/deucebucket/games/cerebellum-rowblock-tmp")


def resolve_ppl_bin() -> str:
    candidates = [
        os.environ.get("LLAMA_PERPLEXITY_BIN", ""),
        "/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity",
        "/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity.exe",
        "/var/home/deucebucket/ai-drive/llama.cpp/build/tools/perplexity/perplexity",
        "/var/home/deucebucket/ai-drive/llama.cpp/build-cpu/tools/perplexity/perplexity",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return candidates[1]


PPL_BIN = resolve_ppl_bin()

# CUDA libs visible to PPL_BIN under nohup
_CUDA_LIBS = (
    "/var/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/cuda_runtime/lib:"
    "/var/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/cublas/lib:"
    "/var/home/deucebucket/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
)
os.environ["LD_LIBRARY_PATH"] = f"{_CUDA_LIBS}:{os.environ.get('LD_LIBRARY_PATH', '')}"


# ---------- llama.cpp wrappers ----------

def quantize_to(source_gguf: str, imatrix: str, base_type: str,
                 override_path: str | None, out_path: str,
                 timeout: int = 1800,
                 output_tensor_type: str | None = None,
                 token_embedding_type: str | None = None) -> bool:
    """Run llama-quantize. Use output_tensor_type/token_embedding_type for
    output.weight and token_embd.weight respectively — those are special-cased
    by llama-quantize with hardcoded defaults that --tensor-type-file CANNOT
    override. The dedicated flags are the only way to override them."""
    cmd = [QUANTIZE_BIN, "--imatrix", imatrix]
    if override_path is not None:
        cmd += ["--tensor-type-file", override_path]
    if output_tensor_type is not None:
        cmd += ["--output-tensor-type", output_tensor_type]
    if token_embedding_type is not None:
        cmd += ["--token-embedding-type", token_embedding_type]
    cmd += [source_gguf, out_path, base_type]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(f"  QUANTIZE FAILED: rc={proc.returncode}", flush=True)
        print(f"  stderr tail: {proc.stderr[-400:]}", flush=True)
        return False
    return True


def write_merged_override(base_override_path: str | None, target: str,
                          target_quant: str, out_path: Path) -> str:
    """Write a tensor override file with an optional recipe plus target pin.

    The target assignment is intentionally written last because llama-quantize
    applies the last matching override for a tensor.
    """
    lines = []
    if base_override_path:
        with open(base_override_path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.split("=", 1)[0].strip() == target:
                    continue
                lines.append(line)
    lines.append(f"{target}={target_quant}")
    out_path.write_text("\n".join(lines) + "\n")
    return str(out_path)


def override_fingerprint(path: str | None) -> str:
    if not path:
        return "none"
    data = Path(path).read_bytes()
    return hashlib.sha1(data).hexdigest()[:10]


# Tensors that llama-quantize special-cases — overriding via --tensor-type-file
# is silently ignored. Must use the dedicated --output-tensor-type or
# --token-embedding-type flags instead.
SPECIAL_TENSORS = {
    "output.weight": "output_tensor_type",
    "token_embd.weight": "token_embedding_type",
}

# --output-tensor-type / --token-embedding-type only accept "non-variant" K-quants
# (Q3_K, Q4_K, Q5_K, Q6_K), not the variant-suffixed forms (Q3_K_S, Q4_K_M, etc).
# Translate user-friendly model-wide labels to the closest valid tensor-type.
SPECIAL_QUANT_NORMALIZE = {
    "Q3_K_S": "Q3_K",  # Q3_K is alias for Q3_K_M — closest valid for output/embd
    "Q3_K_M": "Q3_K",
    "Q3_K_L": "Q3_K",
    "Q4_K_S": "Q4_K",
    "Q4_K_M": "Q4_K",
    "Q5_K_S": "Q5_K",
    "Q5_K_M": "Q5_K",
    "Q2_K_S": "Q2_K",
}


def normalize_target_name(tensor_name: str) -> str:
    return tensor_name if tensor_name.endswith(".weight") else f"{tensor_name}.weight"


def unsupported_rowblock_layout_reason(tensor_name: str) -> str | None:
    """Return a fail-fast reason for tensors whose physical row layout is unresolved."""
    if tensor_name.endswith(".attn_qkv.weight"):
        return (
            "fused attn_qkv tensors have unresolved Q/K/V physical storage ordering; "
            "row-block patching can write the wrong logical rows"
        )
    return None


def special_tensor_override_plan(tensor_name: str, quant_label: str) -> dict | None:
    special_kwarg = SPECIAL_TENSORS.get(tensor_name)
    if special_kwarg is None:
        return None
    normalized = SPECIAL_QUANT_NORMALIZE.get(quant_label, quant_label)
    return {
        "tensor": tensor_name,
        "llama_quantize_flag": "--" + special_kwarg.replace("_", "-"),
        "requested_quant": quant_label,
        "effective_tensor_quant": normalized,
        "variant_normalized": normalized != quant_label,
        "reason": "--tensor-type-file is silently ignored for this tensor by llama-quantize",
    }


def rowblock_safety_report(tensor_name: str, base_quant: str, tensor_base_quant: str,
                           low_quant: str, allow_unsupported_layout: bool = False) -> dict:
    target = normalize_target_name(tensor_name)
    unsupported_reason = unsupported_rowblock_layout_reason(target)
    special_low = special_tensor_override_plan(target, low_quant)
    special_base = special_tensor_override_plan(target, tensor_base_quant)
    blocked = bool(unsupported_reason and not allow_unsupported_layout)
    return {
        "target_tensor": target,
        "rowblock_safe": not blocked,
        "blocked": blocked,
        "unsupported_layout_reason": unsupported_reason,
        "allow_unsupported_layout": allow_unsupported_layout,
        "base_quant": base_quant,
        "tensor_base_quant": tensor_base_quant,
        "low_quant": low_quant,
        "special_tensor": target in SPECIAL_TENSORS,
        "special_overrides": [row for row in [special_low, special_base] if row is not None],
        "notes": [
            "validate-only does not prove byte layout; first-block PPL sanity still guards runtime patches",
            "unsupported fused layouts should use whole-tensor ablation until layout-aware patching lands",
        ],
    }


def measure_ppl(gguf_path: str, corpus_path: str, ctx_size: int = 2048,
                timeout: int = 900) -> float | None:
    cmd = [PPL_BIN, "--model", gguf_path, "--file", corpus_path,
           "-ngl", "99", "--ctx-size", str(ctx_size), "--chunks", "-1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(f"  PPL FAILED: rc={proc.returncode}: {proc.stderr[-300:]}", flush=True)
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", proc.stderr + proc.stdout)
    return float(m.group(1)) if m else None


def measure_per_domain(gguf_path: str, corpus_dir: Path, domains: list[str]) -> dict | None:
    out = {}
    for d in domains:
        corpus = corpus_dir / f"cerebellum_calibration_{d}.txt"
        ppl = measure_ppl(gguf_path, str(corpus))
        if ppl is None:
            return None
        out[d] = ppl
        print(f"    {d}: {ppl:.4f}", flush=True)
    return out


# ---------- GGUF surgery ----------

def find_tensor_offset(gguf_path: str, tensor_name: str) -> tuple[int, int, tuple[int, ...], gguf.GGMLQuantizationType]:
    """Return (file_offset_bytes, data_size_bytes, shape, quant_type) for tensor."""
    reader = gguf.GGUFReader(gguf_path, "r")
    for t in reader.tensors:
        if t.name == tensor_name:
            # GGUFReader exposes a numpy memmap view; .data.ctypes.data is the absolute address.
            # Use the documented `t.data_offset` if present, else fall back to memmap offset math.
            base_addr = reader.data.ctypes.data
            tensor_addr = t.data.ctypes.data
            file_offset = tensor_addr - base_addr  # offset within the mmap
            # The mmap base offset within the file is reader.tensor_data_offset (or similar)
            # Actually reader maps the entire file from offset 0, so file_offset is the absolute file offset.
            return file_offset, t.n_bytes, tuple(int(x) for x in t.shape), t.tensor_type
    raise KeyError(f"tensor {tensor_name} not in {gguf_path}")


def read_tensor_dequantized(gguf_path: str, tensor_name: str) -> np.ndarray:
    """Read tensor from GGUF and return as fp32 numpy array (dequantized if quantized)."""
    reader = gguf.GGUFReader(gguf_path, "r")
    for t in reader.tensors:
        if t.name == tensor_name:
            qt = t.tensor_type
            if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                      gguf.GGMLQuantizationType.BF16):
                arr = np.array(t.data)  # already float
                # Reshape from raw bytes view to logical shape
                shape = tuple(int(x) for x in t.shape[::-1])  # GGUF stores shape reversed
                return arr.reshape(shape).astype(np.float32)
            else:
                # K-quants, I-quants, legacy quants: dequantize via gguf.quants
                shape = tuple(int(x) for x in t.shape[::-1])
                deq = gguf.quants.dequantize(t.data, qt)
                return deq.reshape(shape).astype(np.float32)
    raise KeyError(f"tensor {tensor_name} not in {gguf_path}")


def patch_f16_tensor(gguf_path: str, tensor_offset: int, fp32_data: np.ndarray):
    """Overwrite tensor bytes in-place with fp16 representation of fp32_data."""
    fp16_bytes = fp32_data.astype(np.float16).tobytes()
    with open(gguf_path, "r+b") as f:
        f.seek(tensor_offset)
        f.write(fp16_bytes)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-gguf")
    ap.add_argument("--imatrix")
    ap.add_argument("--target-tensor", required=True,
                    help="GGUF tensor name (e.g. blk.5.ffn_down.weight)")
    ap.add_argument("--base-quant", default="Q3_K_M",
                    help="Fallback quant for the rest of the model")
    ap.add_argument("--base-override-file", default=None,
                    help="Optional full recipe tensor-type file for the rest of the model")
    ap.add_argument("--tensor-base-quant", default=None,
                    help="Baseline quant for the target tensor; defaults to --base-quant")
    ap.add_argument("--low-quant", default="Q2_K",
                    help="Quant whose dequantized values get spliced into row-blocks")
    ap.add_argument("--block-size", type=int, default=128,
                    help="Rows per row-block (~1.4 MB shard target for large MLPs)")
    ap.add_argument("--corpus-dir")
    ap.add_argument("--output")
    ap.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    ap.add_argument("--allow-near-neutral-first-block", action="store_true",
                    help="Continue if block 0 is nearly neutral across domains")
    ap.add_argument("--allow-unsupported-layout", action="store_true",
                    help="Bypass fail-fast guard for known unresolved tensor storage layouts")
    ap.add_argument("--validate-only", action="store_true",
                    help="Print rowblock safety/override report and exit before quant/PPL work")
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    target = normalize_target_name(args.target_tensor)
    base_quant = args.base_quant
    tensor_base_quant = args.tensor_base_quant or base_quant
    low_quant = args.low_quant

    safety = rowblock_safety_report(target, base_quant, tensor_base_quant, low_quant,
                                    allow_unsupported_layout=args.allow_unsupported_layout)
    if args.validate_only:
        print(json.dumps(safety, indent=2, sort_keys=True))
        sys.exit(6 if safety["blocked"] else 0)

    missing = [name for name in ["source_gguf", "imatrix", "corpus_dir", "output"] if not getattr(args, name)]
    if missing:
        print(f"FATAL: missing required arguments for rowblock run: {', '.join('--' + name.replace('_', '-') for name in missing)}",
              file=sys.stderr)
        print("  For a path-free safety check, use --validate-only.", file=sys.stderr)
        sys.exit(2)

    output_path = Path(args.output)
    corpus_dir = Path(args.corpus_dir)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    unsupported_reason = unsupported_rowblock_layout_reason(target)
    if unsupported_reason and not args.allow_unsupported_layout:
        print(f"FATAL: {target} is not row-block safe yet: {unsupported_reason}.", file=sys.stderr)
        print("  Use whole-tensor ablation for this tensor until layout-aware fused-QKV patching lands.",
              file=sys.stderr)
        print("  To intentionally reproduce/debug the old behavior, pass --allow-unsupported-layout.",
              file=sys.stderr)
        sys.exit(6)

    # NOTE: llama-quantize applies hardcoded per-tensor overrides for some tensors
    # (output.weight, token_embd.weight forced to Q6_K regardless of base type).
    # If our target is one of those, plain `llama-quantize ... Q2_K` would store it
    # at Q6_K in BOTH q_low and q_base — making low_T ≡ base_T and our patches no-ops.
    # Solution: pass an explicit override that pins the target to the requested quant
    # in BOTH the low and base reference builds. Per-target work_dir prevents collision.
    target_safe = target.replace('.', '_')
    recipe_id = override_fingerprint(args.base_override_file)
    run_id = f"{base_quant}_recipe_{recipe_id}_{target_safe}_base_{tensor_base_quant}_low_{low_quant}"
    template_path = WORK_DIR / f"template_{run_id}.gguf"
    q_low_path = WORK_DIR / f"q_low_{run_id}.gguf"
    q_base_path = WORK_DIR / f"q_base_{run_id}.gguf"

    # --- Setup phase ---

    # 1. template: rest at base, target at F16 (this is what we patch)
    if not template_path.exists():
        recipe_note = f" + {args.base_override_file}" if args.base_override_file else ""
        print(f"[1/3] Building template: {target}=F16, rest at {base_quant}{recipe_note}", flush=True)
        ovr = WORK_DIR / f"template_override_{run_id}.txt"
        write_merged_override(args.base_override_file, target, "F16", ovr)
        if not quantize_to(args.source_gguf, args.imatrix, base_quant,
                            str(ovr), str(template_path)):
            print("FATAL: template build failed", file=sys.stderr)
            sys.exit(2)
    else:
        print(f"[1/3] Reusing template: {template_path}", flush=True)

    # Determine override mechanism based on target tensor.
    # For output.weight / token_embd.weight, --tensor-type-file is silently
    # ignored — must use the dedicated flags.
    special_kwarg = SPECIAL_TENSORS.get(target)

    def build_quant_ref(quant_label: str, out_path: Path, label: str):
        recipe_note = f" + {args.base_override_file}" if args.base_override_file else ""
        print(f"{label}: {target}={quant_label}, rest at {base_quant}{recipe_note}", flush=True)
        if special_kwarg is not None:
            # Translate variant suffixes to non-variant for the dedicated flags
            tensor_quant = SPECIAL_QUANT_NORMALIZE.get(quant_label, quant_label)
            if tensor_quant != quant_label:
                print(f"    note: --{special_kwarg.replace('_','-')} accepts non-variant types only; "
                      f"using {tensor_quant} for the {target} override "
                      f"(model-wide quant remains {quant_label})", flush=True)
            kwargs = {special_kwarg: tensor_quant}
            return quantize_to(args.source_gguf, args.imatrix, quant_label,
                               None, str(out_path), **kwargs)
        else:
            # Use override file for normal tensors
            ovr_file = WORK_DIR / f"override_{run_id}_{quant_label}.txt"
            write_merged_override(args.base_override_file, target, quant_label, ovr_file)
            return quantize_to(args.source_gguf, args.imatrix, base_quant,
                               str(ovr_file), str(out_path))

    # 2. q_low: target FORCED to low quant — to extract dequantized T at low precision
    if not q_low_path.exists():
        if not build_quant_ref(low_quant, q_low_path, "[2/3] Building q_low"):
            print("FATAL: q_low build failed", file=sys.stderr)
            sys.exit(2)
    else:
        print(f"[2/3] Reusing q_low: {q_low_path}", flush=True)

    # 3. q_base: target FORCED to tensor base quant — to extract dequantized T at base precision
    if not q_base_path.exists():
        if not build_quant_ref(tensor_base_quant, q_base_path, "[3/3] Building q_base"):
            print("FATAL: q_base build failed", file=sys.stderr)
            sys.exit(2)
    else:
        print(f"[3/3] Reusing q_base: {q_base_path}", flush=True)

    # --- Read dequantized views ---

    print(f"\nReading dequantized {target} from q_low and q_base...", flush=True)
    low_T = read_tensor_dequantized(str(q_low_path), target)
    base_T = read_tensor_dequantized(str(q_base_path), target)
    print(f"  low_T:  shape={low_T.shape}  dtype={low_T.dtype}", flush=True)
    print(f"  base_T: shape={base_T.shape}  dtype={base_T.dtype}", flush=True)
    if low_T.shape != base_T.shape:
        print("FATAL: shape mismatch", file=sys.stderr)
        sys.exit(3)

    # Sanity: low_T must differ meaningfully from base_T or every patch is a no-op.
    # This catches the llama-quantize hardcoded-override class of bug (output.weight,
    # token_embd) where both quants got bumped to Q6_K despite our request.
    diff = float(np.abs(low_T - base_T).mean())
    base_mag = float(np.abs(base_T).mean())
    rel_diff = diff / max(base_mag, 1e-9)
    print(f"  low vs base  mean |Δ|={diff:.6f}  (rel={rel_diff*100:.4f}% of |base|)", flush=True)
    if rel_diff < 1e-4:
        print(f"FATAL: low_T ≈ base_T (rel diff {rel_diff*100:.4f}% — every row-block patch would be a no-op).",
              file=sys.stderr)
        print(f"  Likely cause: llama-quantize applied a hardcoded override and stored {target} at the SAME",
              file=sys.stderr)
        print(f"  quant type in both q_low ({low_quant}) and q_base ({base_quant}).", file=sys.stderr)
        print(f"  Inspect with: python -c \"import gguf; r=gguf.GGUFReader('{q_low_path}','r'); ", file=sys.stderr)
        print(f"  print([(t.name,t.tensor_type.name) for t in r.tensors if t.name=='{target}'])\"",
              file=sys.stderr)
        sys.exit(4)

    # --- Find tensor offset in template (for in-place patching) ---

    offset, nbytes, shape, qtype = find_tensor_offset(str(template_path), target)
    print("\nTemplate tensor info:", flush=True)
    print(f"  offset:  {offset}", flush=True)
    print(f"  nbytes:  {nbytes}", flush=True)
    print(f"  shape:   {shape}", flush=True)
    print(f"  qtype:   {qtype.name}", flush=True)
    expected_nbytes = base_T.size * 2  # F16
    if nbytes != expected_nbytes:
        print(f"FATAL: template tensor not F16 (got {qtype.name}, {nbytes} bytes vs expected {expected_nbytes})",
              file=sys.stderr)
        sys.exit(3)

    n_rows = base_T.shape[0]
    n_blocks = (n_rows + args.block_size - 1) // args.block_size
    print("\nRow-block plan:", flush=True)
    print(f"  total rows:   {n_rows}", flush=True)
    print(f"  block size:   {args.block_size}", flush=True)
    print(f"  num blocks:   {n_blocks}", flush=True)

    # --- Baseline measurement (T = base_T at F16) ---

    print(f"\n=== baseline (T = target_base_{tensor_base_quant}) ===", flush=True)
    patch_f16_tensor(str(template_path), offset, base_T)
    t0 = time.time()
    ppl_baseline = measure_per_domain(str(template_path), corpus_dir, domains)
    if ppl_baseline is None:
        print("FATAL: baseline PPL failed", file=sys.stderr)
        sys.exit(4)
    print(f"  baseline took {time.time()-t0:.0f}s", flush=True)

    results = {
        "tensor": target,
        "source_gguf": args.source_gguf,
        "base_quant": base_quant,
        "base_override_file": args.base_override_file,
        "tensor_base_quant": tensor_base_quant,
        "low_quant": low_quant,
        "block_size_rows": args.block_size,
        "tensor_shape": list(shape),
        "n_blocks": n_blocks,
        "baseline_ppl": ppl_baseline,
        "blocks": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Per-row-block measurement ---

    t_start = time.time()
    for i in range(n_blocks):
        row_lo = i * args.block_size
        row_hi = min((i + 1) * args.block_size, n_rows)

        hybrid_T = base_T.copy()
        hybrid_T[row_lo:row_hi] = low_T[row_lo:row_hi]
        patch_f16_tensor(str(template_path), offset, hybrid_T)

        elapsed = time.time() - t_start
        print(f"\n[{i+1}/{n_blocks}] block rows {row_lo}..{row_hi-1} "
              f"(elapsed {elapsed:.0f}s)", flush=True)

        ppl_hybrid = measure_per_domain(str(template_path), corpus_dir, domains)
        if ppl_hybrid is None:
            block_result = {"start_row": row_lo, "end_row": row_hi - 1,
                            "error": "ppl_failed"}
        else:
            delta = {d: ppl_hybrid[d] - ppl_baseline[d] for d in domains}
            block_result = {"start_row": row_lo, "end_row": row_hi - 1,
                            "ppl": ppl_hybrid, "delta": delta}
        results["blocks"].append(block_result)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # First-block sanity: if PPL across ALL domains is identical to baseline
        # within float noise, the patch isn't reaching inference. Could be a
        # storage-layout / fused-tensor bug (e.g. attn_qkv). Abort fast — don't
        # waste hours producing identical numbers.
        if i == 0 and ppl_hybrid is not None and not args.allow_near_neutral_first_block:
            max_dev = max(abs(ppl_hybrid[d] - ppl_baseline[d]) for d in domains)
            if max_dev < 1e-3:
                print(f"\nFATAL: block 0 PPL is identical to baseline within {max_dev:.6f} "
                      f"across all domains.", file=sys.stderr)
                print("  This means the row-block patch is NOT reaching inference — likely a", file=sys.stderr)
                print("  storage-layout bug (fused QKV, transposed-large tensor, alignment).", file=sys.stderr)
                print("  Aborting before wasting hours of GPU on identical measurements.", file=sys.stderr)
                print("  Investigate: check tensor shape vs storage byte order; verify offset.",
                      file=sys.stderr)
                # Restore baseline before exit
                patch_f16_tensor(str(template_path), offset, base_T)
                sys.exit(5)

    # Restore base_T into template so future runs see clean state
    patch_f16_tensor(str(template_path), offset, base_T)

    print("\n=== Done ===", flush=True)
    print(f"Total time: {(time.time() - t_start):.0f}s", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
