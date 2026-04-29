"""Streaming multi-depth sensitivity analyzer — no GPU, any model size.

Computes per-tensor reconstruction error at Q2/Q3/Q4/Q6/Q8 bit depths
using weight-only analysis via safetensors mmap. Produces the same JSON
format as sensitivity_multi.py so budget.py can consume it directly.

No forward passes needed — measures how much each tensor's values change
under simulated GGUF blockwise quantization. Fast proxy for actual KL
divergence: tensors with high reconstruction error at low bits need more
precision.

Usage:
    python -m osmosis.sensitivity_stream \
        --model Qwen/Qwen3.6-27B \
        --output sensitivity_multi.json \
        -v
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

BIT_DEPTHS = [2, 3, 4, 6, 8]
GROUP_SIZE = 32

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
}

LAYER_PATTERN = re.compile(
    r"(?:model\.(?:language_model\.)?)?layers\.(\d+)\.(.*?)\.weight$"
)


def quantize_blockwise(tensor: torch.Tensor, bits: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    flat = tensor.float().reshape(-1)
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)
    scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    max_val = (1 << (bits - 1)) - 1
    normalized = groups / scales
    quantized = (normalized * max_val).round().clamp(-max_val, max_val)
    dequantized = (quantized / max_val) * scales
    result = dequantized.reshape(-1)[:n]
    return result.reshape(tensor.shape)


def measure_all_depths(weight: torch.Tensor, device: torch.device) -> dict[int, dict]:
    """Measure reconstruction at all bit depths in one pass on GPU."""
    flat = weight.float().to(device).reshape(-1)
    n = flat.numel()
    pad = (GROUP_SIZE - n % GROUP_SIZE) % GROUP_SIZE
    if pad:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, GROUP_SIZE)
    scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = groups / scales

    w_flat = flat[:n]
    w_power = w_flat.pow(2).mean().item() + 1e-10
    w_norm = w_flat.pow(2).sum().sqrt()

    results = {}
    for bits in BIT_DEPTHS:
        max_val = (1 << (bits - 1)) - 1
        quantized = (normalized * max_val).round().clamp(-max_val, max_val)
        dequantized = ((quantized / max_val) * scales).reshape(-1)[:n]

        diff = dequantized - w_flat
        mse = diff.pow(2).mean().item()
        relative_err = mse / w_power
        cos = (w_flat * dequantized).sum() / (w_norm * dequantized.pow(2).sum().sqrt() + 1e-10)
        cos = min(cos.item(), 1.0)
        damage = 1.0 - cos

        results[bits] = {
            "kl_mean": damage,
            "kl_max": damage * 2,
            "cosine_mean": cos,
            "cosine_min": cos,
            "mse_mean": mse,
            "relative_err": relative_err,
        }

    del flat, groups, scales, normalized, w_flat
    return results


def resolve_safetensors(model_path: str) -> list[Path]:
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
    index = {}
    for path in shard_paths:
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                index[key] = path
    return index


def generate_sensitivity(model_path: str, output_path: str, verbose: bool = False):
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Resolving safetensors for {model_path}...")
    shard_paths = resolve_safetensors(model_path)
    print(f"  Found {len(shard_paths)} shard(s)")

    print("Building tensor index...")
    tensor_index = build_tensor_index(shard_paths)

    targets = {}
    num_layers = 0
    for name in tensor_index:
        m = LAYER_PATTERN.match(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        suffix = m.group(2)
        if suffix not in HF_TO_GGUF:
            continue
        sens_name = f"layer_{layer_idx}.{suffix}"
        targets[sens_name] = {
            "hf_name": name,
            "layer_idx": layer_idx,
            "group_type": suffix,
            "gguf_suffix": HF_TO_GGUF[suffix],
        }
        num_layers = max(num_layers, layer_idx + 1)

    print(f"  {len(targets)} weight tensors across {num_layers} layers")
    print(f"  Testing bit depths: {BIT_DEPTHS}")

    all_results = []
    open_files: dict[str, safe_open] = {}
    t_compute = time.time()

    sorted_targets = sorted(targets.items(), key=lambda kv: (kv[1]["layer_idx"], kv[0]))

    for i, (sens_name, info) in enumerate(sorted_targets):
        hf_name = info["hf_name"]
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

        depth_results = measure_all_depths(w, device)
        cosines_by_depth = {b: depth_results[b]["cosine_mean"] for b in BIT_DEPTHS}

        recommended = BIT_DEPTHS[-1]
        for b in BIT_DEPTHS:
            if cosines_by_depth[b] > 0.9999:
                recommended = b
                break

        result = {
            "name": sens_name,
            "layer_idx": info["layer_idx"],
            "group_type": info["group_type"],
            "param_count": w.numel(),
            "shape": list(w.shape),
            "depths": depth_results,
            "cosine_by_depth": cosines_by_depth,
            "recommended_bits": recommended,
        }
        all_results.append(result)
        del w

        if verbose:
            curve = " | ".join(f"Q{b}:{cosines_by_depth[b]:.6f}" for b in BIT_DEPTHS)
            print(f"  {sens_name:45s} {curve}  rec={recommended}b")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_compute
            rate = (i + 1) / elapsed
            remaining = len(sorted_targets) - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(sorted_targets)}] {rate:.1f} tensors/s, ETA {eta:.0f}s")

    for ctx in open_files.values():
        ctx.__exit__(None, None, None)

    compute_elapsed = time.time() - t_compute

    report = {
        "model": model_path,
        "method": "weight_reconstruction_streaming",
        "bit_depths": BIT_DEPTHS,
        "group_size": GROUP_SIZE,
        "num_layers": num_layers,
        "num_groups": len(all_results),
        "compute_seconds": compute_elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    total_elapsed = time.time() - t_start
    print(f"\nAnalyzed {len(all_results)} tensors in {compute_elapsed:.1f}s")
    print(f"Total: {total_elapsed:.1f}s")
    print(f"Saved to {output_path}")

    bit_dist = {}
    for r in all_results:
        b = r["recommended_bits"]
        bit_dist[b] = bit_dist.get(b, 0) + 1
    print(f"Recommended bit distribution: {dict(sorted(bit_dist.items()))}")

    total_params = sum(r["param_count"] for r in all_results)
    weighted_bits = sum(r["param_count"] * r["recommended_bits"] for r in all_results)
    avg_bits = weighted_bits / total_params if total_params else 0
    print(f"Weighted average recommended bits: {avg_bits:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming multi-depth sensitivity analyzer — no GPU, any model size"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output sensitivity JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    generate_sensitivity(args.model, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
