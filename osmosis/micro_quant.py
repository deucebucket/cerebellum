"""Sub-tensor micro-targeting quantization for Osmosis.

Instead of quantizing an entire tensor at one bit depth, split it into
spatial blocks and assign different quant levels to each block based on
local sensitivity. The "box on the neck" — surgical precision instead of
a hammer.

This is research code. GGUF doesn't natively support intra-tensor mixed
precision, so the end product is either:
  1. A custom GGUF writer that packs blocks at different depths
  2. A simulated-quant approach for LoRA training (quantize fp16 weights
     block-by-block at different depths, then train LoRA on top)

Usage:
    python -m osmosis.micro_quant \
        --model .hf_cache/models--Qwen--Qwen3.5-9B/snapshots/<hash> \
        --output osmosis-qwen3.5-9b/micro_sensitivity.json \
        --layer 31 --group mlp.down_proj \
        --block-size 256 --samples 8 -v
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BIT_DEPTHS = [2, 3, 4, 6, 8]
GROUP_SIZE = 32  # GGUF block quantization group size


def quantize_blockwise(tensor: torch.Tensor, bits: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Simulate GGUF-style block quantization at N bits."""
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
    return result.reshape(tensor.shape).to(tensor.dtype)


def analyze_block_sensitivity(
    weight: torch.Tensor,
    block_rows: int,
    block_cols: int,
) -> dict:
    """Measure how sensitive each spatial block of a weight matrix is.

    Splits the weight into a grid of (block_rows x block_cols) blocks.
    For each block, quantizes ONLY that block at each bit depth while
    keeping everything else at fp16. Measures reconstruction error.

    Returns a grid of sensitivity scores per block per bit depth.
    """
    rows, cols = weight.shape
    n_row_blocks = math.ceil(rows / block_rows)
    n_col_blocks = math.ceil(cols / block_cols)

    results = {
        "shape": [rows, cols],
        "block_size": [block_rows, block_cols],
        "grid": [n_row_blocks, n_col_blocks],
        "blocks": [],
    }

    for ri in range(n_row_blocks):
        for ci in range(n_col_blocks):
            r_start = ri * block_rows
            r_end = min(r_start + block_rows, rows)
            c_start = ci * block_cols
            c_end = min(c_start + block_cols, cols)

            block = weight[r_start:r_end, c_start:c_end]
            block_numel = block.numel()

            depth_scores = {}
            for bits in BIT_DEPTHS:
                quantized_block = quantize_blockwise(block, bits)
                mse = F.mse_loss(quantized_block.float(), block.float()).item()
                max_err = (quantized_block.float() - block.float()).abs().max().item()
                rel_err = mse / (block.float().pow(2).mean().item() + 1e-10)
                depth_scores[bits] = {
                    "mse": mse,
                    "max_err": max_err,
                    "relative_err": rel_err,
                }

            recommended = 8
            for b in BIT_DEPTHS:
                if depth_scores[b]["relative_err"] < 0.01:
                    recommended = b
                    break

            results["blocks"].append({
                "row_block": ri,
                "col_block": ci,
                "row_range": [r_start, r_end],
                "col_range": [c_start, c_end],
                "numel": block_numel,
                "weight_magnitude": float(block.float().abs().mean().item()),
                "weight_std": float(block.float().std().item()),
                "depths": depth_scores,
                "recommended_bits": recommended,
            })

    return results


@torch.no_grad()
def analyze_block_impact(
    model,
    module,
    layer_idx: int,
    input_ids: torch.Tensor,
    block_rows: int,
    block_cols: int,
    baseline_logprobs: torch.Tensor,
) -> dict:
    """Measure actual model output impact of quantizing each spatial block.

    Like analyze_block_sensitivity but measures KL divergence on real
    forward passes instead of just reconstruction error.
    """
    weight = module.weight.data
    rows, cols = weight.shape
    n_row_blocks = math.ceil(rows / block_rows)
    n_col_blocks = math.ceil(cols / block_cols)
    original = weight.clone()

    results = {
        "shape": [rows, cols],
        "block_size": [block_rows, block_cols],
        "grid": [n_row_blocks, n_col_blocks],
        "blocks": [],
    }

    for ri in range(n_row_blocks):
        for ci in range(n_col_blocks):
            r_start = ri * block_rows
            r_end = min(r_start + block_rows, rows)
            c_start = ci * block_cols
            c_end = min(c_start + block_cols, cols)

            block = original[r_start:r_end, c_start:c_end]

            depth_scores = {}
            for bits in BIT_DEPTHS:
                module.weight.data = original.clone()
                quantized_block = quantize_blockwise(block, bits).to(
                    device=block.device, dtype=block.dtype
                )
                module.weight.data[r_start:r_end, c_start:c_end] = quantized_block

                output = model(input_ids)
                crushed_logprobs = F.log_softmax(output.logits[0].float(), dim=-1)
                kl = F.kl_div(
                    crushed_logprobs, baseline_logprobs.exp(),
                    reduction="batchmean", log_target=False,
                ).item()

                depth_scores[bits] = {"kl": kl}

            recommended = 8
            for b in BIT_DEPTHS:
                if depth_scores[b]["kl"] < 0.001:
                    recommended = b
                    break

            results["blocks"].append({
                "row_block": ri,
                "col_block": ci,
                "row_range": [r_start, r_end],
                "col_range": [c_start, c_end],
                "numel": int(block.numel()),
                "depths": depth_scores,
                "recommended_bits": recommended,
            })

    module.weight.data = original
    return results


def visualize_block_grid(results: dict, metric: str = "recommended_bits"):
    """Print ASCII heatmap of block sensitivity grid."""
    grid_r, grid_c = results["grid"]
    blocks = results["blocks"]

    block_map = {}
    for b in blocks:
        block_map[(b["row_block"], b["col_block"])] = b

    print(f"\nBlock sensitivity map ({results['shape'][0]}x{results['shape'][1]}, "
          f"block={results['block_size'][0]}x{results['block_size'][1]}):")
    print(f"  .=Q2(crushable)  o=Q3  O=Q4  #=Q6+/Q8(sensitive)")
    print()

    for ri in range(grid_r):
        row_str = f"  r{ri:3d} "
        for ci in range(grid_c):
            b = block_map.get((ri, ci))
            if b:
                rec = b["recommended_bits"]
                chars = {2: "..", 3: "oo", 4: "OO", 6: "##", 8: "##"}
                row_str += chars.get(rec, "??")
            else:
                row_str += "  "
        print(row_str)

    rec_dist = {}
    total_params = 0
    weighted_bits = 0
    for b in blocks:
        rec = b["recommended_bits"]
        n = b["numel"]
        rec_dist[rec] = rec_dist.get(rec, 0) + 1
        total_params += n
        weighted_bits += n * rec

    avg_bits = weighted_bits / total_params if total_params else 0
    print(f"\n  Distribution: {rec_dist}")
    print(f"  Weighted average: {avg_bits:.2f} bits")
    print(f"  vs uniform Q4: 4.00 bits -> saves {(4.0 - avg_bits) / 4.0 * 100:.1f}% more")


def run_micro_analysis(
    model_path: str,
    output_path: str,
    target_layer: int,
    target_group: str,
    block_size: int = 256,
    num_samples: int = 8,
    max_length: int = 128,
    mode: str = "weight",
    verbose: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    QWEN35_WEIGHT_GROUPS = {
        "linear_attn.in_proj_qkv": ("linear_attn", "in_proj_qkv"),
        "linear_attn.in_proj_a": ("linear_attn", "in_proj_a"),
        "linear_attn.in_proj_b": ("linear_attn", "in_proj_b"),
        "linear_attn.in_proj_z": ("linear_attn", "in_proj_z"),
        "linear_attn.out_proj": ("linear_attn", "out_proj"),
        "linear_attn.conv1d": ("linear_attn", "conv1d"),
        "self_attn.q_proj": ("self_attn", "q_proj"),
        "self_attn.k_proj": ("self_attn", "k_proj"),
        "self_attn.v_proj": ("self_attn", "v_proj"),
        "self_attn.o_proj": ("self_attn", "o_proj"),
        "mlp.gate_proj": ("mlp", "gate_proj"),
        "mlp.up_proj": ("mlp", "up_proj"),
        "mlp.down_proj": ("mlp", "down_proj"),
    }

    layer = model.model.layers[target_layer]
    parent_name, weight_name = QWEN35_WEIGHT_GROUPS[target_group]
    parent = getattr(layer, parent_name)
    module = getattr(parent, weight_name)
    weight = module.weight.data

    print(f"Target: layer {target_layer} / {target_group}")
    print(f"Weight shape: {list(weight.shape)}, params: {weight.numel():,}")
    print(f"Block size: {block_size}x{block_size}")
    print(f"Mode: {mode}")

    if mode == "weight":
        results = analyze_block_sensitivity(weight, block_size, block_size)
    elif mode == "impact":
        from osmosis.sensitivity_multi import load_calibration_prompts

        prompts = load_calibration_prompts(num_samples)
        all_input_ids = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_length)
            all_input_ids.append(tokens["input_ids"].to(model.device))

        all_block_results = None
        for idx, input_ids in enumerate(all_input_ids):
            if verbose:
                print(f"  Sample {idx+1}/{len(all_input_ids)}...")
            output = model(input_ids)
            baseline = F.log_softmax(output.logits[0].float(), dim=-1)
            sample_results = analyze_block_impact(
                model, module, target_layer, input_ids,
                block_size, block_size, baseline,
            )
            if all_block_results is None:
                all_block_results = sample_results
            else:
                for i, block in enumerate(sample_results["blocks"]):
                    for bits_str, scores in block["depths"].items():
                        for k, v in scores.items():
                            all_block_results["blocks"][i]["depths"][bits_str][k] += v

        for block in all_block_results["blocks"]:
            for bits_str, scores in block["depths"].items():
                for k in scores:
                    scores[k] /= len(all_input_ids)
            best = 8
            for b in BIT_DEPTHS:
                if block["depths"][b]["kl"] < 0.001:
                    best = b
                    break
            block["recommended_bits"] = best

        results = all_block_results
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results["layer_idx"] = target_layer
    results["group_type"] = target_group
    results["mode"] = mode
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if verbose:
        visualize_block_grid(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Osmosis sub-tensor micro-targeting")
    parser.add_argument("--model", required=True, help="HF model path or local snapshot")
    parser.add_argument("--output", default="micro_sensitivity.json")
    parser.add_argument("--layer", type=int, required=True, help="Target layer index")
    parser.add_argument("--group", required=True, help="Weight group (e.g. mlp.down_proj)")
    parser.add_argument("--block-size", type=int, default=256, help="Block dimensions")
    parser.add_argument("--samples", type=int, default=8, help="Calibration samples (impact mode)")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mode", choices=["weight", "impact"], default="weight",
                        help="weight=reconstruction error only, impact=full forward pass KL")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    run_micro_analysis(
        args.model, args.output, args.layer, args.group,
        args.block_size, args.samples, args.max_length, args.mode, args.verbose,
    )


if __name__ == "__main__":
    main()
