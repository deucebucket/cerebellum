"""Streaming imatrix generator — works on models of any size.

Uses safetensors memory-mapped reads to process one tensor at a time.
Never loads the full model into RAM. A 671B model uses ~4GB peak RAM.

Produces identical output to imatrix_gen.py — same binary format, same
sensitivity formula (L2 * maxabs * variance per column, normalized 0-1).

Usage:
    python -m osmosis.imatrix_stream \
        --model deepseek-ai/DeepSeek-V3 \
        --output osmosis_imatrix.dat \
        -v
"""
import argparse
import json
import re
import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open

from .micro_quantizer import compute_channel_sensitivity


HF_TO_GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "per_layer_input_gate": "inp_gate",
    "per_layer_projection": "proj",
}

GLOBAL_TENSOR_MAP = {
    "embed_tokens_per_layer": "per_layer_token_embd",
    "per_layer_model_projection": "per_layer_model_proj",
}

LAYER_PATTERN = re.compile(
    r"(?:model\.(?:language_model\.)?)?layers\.(\d+)\.(.*?)\.weight$"
)

GLOBAL_PATTERN = re.compile(
    r"(?:model\.(?:language_model\.)?)?(embed_tokens_per_layer|per_layer_model_projection)\.weight$"
)


def write_legacy_imatrix(
    path: str,
    data: dict[str, list[float]],
    ncall: int = 1,
    dataset_name: str = "osmosis-sensitivity",
):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(data)))
        for tensor_name, values in data.items():
            name_bytes = tensor_name.encode("utf-8")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<i", ncall))
            f.write(struct.pack("<i", len(values)))
            scaled = [v * ncall for v in values]
            f.write(struct.pack(f"<{len(scaled)}f", *scaled))
        f.write(struct.pack("<i", ncall))
        ds_bytes = dataset_name.encode("utf-8")
        f.write(struct.pack("<i", len(ds_bytes)))
        f.write(ds_bytes)


def hf_name_to_gguf(hf_suffix: str) -> str | None:
    """Map HF weight suffix to GGUF tensor name suffix."""
    for hf_key, gguf_name in HF_TO_GGUF.items():
        if hf_suffix == hf_key:
            return gguf_name
    return None


def resolve_safetensors(model_path: str) -> list[Path]:
    """Find all safetensors files for a model (local or HF cached)."""
    local = Path(model_path)
    if local.is_dir():
        files = sorted(local.glob("*.safetensors"))
        if files:
            return files

    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(
        model_path,
        allow_patterns=["*.safetensors", "*.json"],
    )
    return sorted(Path(cache_dir).glob("*.safetensors"))


def build_tensor_index(shard_paths: list[Path]) -> dict[str, Path]:
    """Build {tensor_name: shard_path} index from all shards."""
    index = {}
    for path in shard_paths:
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                index[key] = path
    return index


def generate_imatrix_streaming(
    model_path: str,
    output_path: str,
    verbose: bool = False,
):
    t_start = time.time()

    print(f"Resolving safetensors for {model_path}...")
    shard_paths = resolve_safetensors(model_path)
    print(f"  Found {len(shard_paths)} shard(s)")

    print("Building tensor index...")
    tensor_index = build_tensor_index(shard_paths)
    print(f"  {len(tensor_index)} tensors indexed")

    targets = {}
    num_layers = 0
    for name in tensor_index:
        m = LAYER_PATTERN.match(name)
        if m:
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            gguf_suffix = hf_name_to_gguf(suffix)
            if gguf_suffix is None:
                continue
            gguf_name = f"blk.{layer_idx}.{gguf_suffix}.weight"
            targets[gguf_name] = name
            num_layers = max(num_layers, layer_idx + 1)
            continue
        gm = GLOBAL_PATTERN.match(name)
        if gm:
            hf_key = gm.group(1)
            gguf_name = f"{GLOBAL_TENSOR_MAP[hf_key]}.weight"
            targets[gguf_name] = name

    n_global = sum(1 for k in targets if not k.startswith("blk."))
    print(f"  {len(targets)} weight tensors across {num_layers} layers ({n_global} global)")

    imatrix_data = {}
    open_files: dict[str, safe_open] = {}
    t_compute = time.time()

    def _sort_key(k):
        parts = k.split('.')
        if parts[0] == 'blk':
            return (0, int(parts[1]), k)
        return (1, 0, k)

    for gguf_name in sorted(targets.keys(), key=_sort_key):
        hf_name = targets[gguf_name]
        shard_path = tensor_index[hf_name]
        shard_key = str(shard_path)

        if shard_key not in open_files:
            for old_key in list(open_files):
                if old_key != shard_key:
                    open_files[old_key].__exit__(None, None, None)
                    del open_files[old_key]
            ctx = safe_open(shard_key, framework="pt")
            open_files[shard_key] = ctx.__enter__()

        sf = open_files[shard_key]
        w = sf.get_tensor(hf_name)
        if w.ndim < 2:
            if verbose:
                print(f"  {gguf_name:40s} SKIP (1D, dim={w.shape[0]})")
            del w
            continue
        sens = compute_channel_sensitivity(w)
        imatrix_data[gguf_name] = sens.tolist()
        del w

        if verbose:
            vals = imatrix_data[gguf_name]
            mn, mx = min(vals), max(vals)
            print(f"  {gguf_name:40s} dim={len(vals):5d} range=[{mn:.4f}, {mx:.4f}]")

    for ctx in open_files.values():
        ctx.__exit__(None, None, None)

    t_compute_done = time.time()
    compute_elapsed = t_compute_done - t_compute

    print(f"\nComputed sensitivity for {len(imatrix_data)} tensors in {compute_elapsed:.1f}s")
    print(f"Writing imatrix to {output_path}...")
    write_legacy_imatrix(output_path, imatrix_data)

    file_size = Path(output_path).stat().st_size
    total_elapsed = time.time() - t_start
    print(f"  {file_size / 1024:.1f} KB, {len(imatrix_data)} entries")
    print(f"\nTOTAL: {total_elapsed:.1f}s  (compute: {compute_elapsed:.1f}s)")
    print(f"Peak RAM usage: streaming — model never fully loaded")
    print(f"Use with: llama-quantize --imatrix {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming imatrix generator — works on models of any size"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output imatrix file path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    generate_imatrix_streaming(args.model, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
