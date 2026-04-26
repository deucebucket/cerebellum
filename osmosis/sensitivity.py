"""Sensitivity analyzer ��� the psychoacoustic model for neural networks.

For each weight group in a transformer model, measures how much the output
distribution changes when that group is quantized to 1-bit. High KL divergence
= sensitive (preserve at higher bits). Low KL = crushable.
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


WEIGHT_GROUPS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def get_layer_modules(model, layer_idx: int) -> dict:
    layer = model.model.layers[layer_idx]
    groups = {}
    for name in WEIGHT_GROUPS:
        parts = name.split("_")
        if parts[-1] == "proj":
            if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parent = layer.self_attn if hasattr(layer, "self_attn") else layer.attn
            else:
                parent = layer.mlp
            if hasattr(parent, name):
                groups[name] = getattr(parent, name)
    return groups


def quantize_1bit(tensor: torch.Tensor) -> torch.Tensor:
    scale = tensor.abs().mean()
    return torch.sign(tensor) * scale


def compute_sparsity(tensor: torch.Tensor, threshold: float = 1e-6) -> float:
    return (tensor.abs() < threshold).float().mean().item()


@torch.no_grad()
def capture_logits(model, tokenizer, prompts, max_length=128):
    all_logits = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        output = model(**tokens)
        log_probs = F.log_softmax(output.logits[0].float(), dim=-1)
        all_logits.append(log_probs.cpu())
    return all_logits


@torch.no_grad()
def measure_sensitivity(model, tokenizer, layer_idx, group_name, module,
                        baseline_logits, prompts, max_length=128):
    original = module.weight.data.clone()
    magnitude = original.abs().mean().item()
    sparsity = compute_sparsity(original)

    module.weight.data = quantize_1bit(original).to(module.weight.device)

    kl_divs = []
    for i, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        output = model(**tokens)
        crushed = F.log_softmax(output.logits[0].float(), dim=-1)
        baseline_probs = baseline_logits[i].to(crushed.device)
        kl = F.kl_div(crushed, baseline_probs.exp(), reduction="batchmean", log_target=False)
        kl_divs.append(kl.item())

    module.weight.data = original

    return {
        "name": f"layer_{layer_idx}.{group_name}",
        "layer_idx": layer_idx,
        "group_type": group_name,
        "param_count": module.weight.numel(),
        "kl_divergence": float(np.mean(kl_divs)),
        "max_kl": float(np.max(kl_divs)),
        "weight_magnitude": magnitude,
        "sparsity": sparsity,
    }


def assign_bits(groups, target_avg=2.0):
    kl_values = [g["kl_divergence"] for g in groups if not math.isinf(g["kl_divergence"])]
    if not kl_values:
        return groups
    p25 = np.percentile(kl_values, 25)
    p75 = np.percentile(kl_values, 75)
    for g in groups:
        kl = g["kl_divergence"]
        if math.isinf(kl) or kl > p75:
            g["recommended_bits"] = 4
        elif kl > p25:
            g["recommended_bits"] = 2
        else:
            g["recommended_bits"] = 1
    return groups


def analyze(model_name: str, output_path: str = "sensitivity_report.json",
            num_samples: int = 100, max_length: int = 128,
            dtype: Optional[str] = None):
    print(f"Loading model: {model_name}")
    torch_dtype = getattr(torch, dtype) if dtype else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    print("Loading calibration data (wikitext-2)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    prompts = [t for t in ds["text"] if len(t) > 50][:num_samples]
    print(f"Using {len(prompts)} calibration prompts")

    print("\nPhase 1: Capturing baseline logits...")
    baseline = capture_logits(model, tokenizer, prompts, max_length)

    num_layers = model.config.num_hidden_layers
    total = num_layers * len(WEIGHT_GROUPS)
    print(f"\nPhase 2: Measuring sensitivity ({num_layers} layers x {len(WEIGHT_GROUPS)} groups = {total})")

    all_groups = []
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        modules = get_layer_modules(model, layer_idx)
        for group_name, module in modules.items():
            result = measure_sensitivity(
                model, tokenizer, layer_idx, group_name, module,
                baseline, prompts, max_length,
            )
            all_groups.append(result)

    print("\nPhase 3: Assigning bit widths...")
    all_groups = assign_bits(all_groups)

    total_params = sum(g["param_count"] for g in all_groups)
    weighted = sum(g["param_count"] * g["recommended_bits"] for g in all_groups)
    avg_bits = weighted / total_params if total_params else 0

    report = {
        "model": model_name,
        "num_layers": num_layers,
        "num_groups": len(all_groups),
        "calibration_samples": len(prompts),
        "total_params": total_params,
        "average_bits": round(avg_bits, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": all_groups,
    }

    bits_dist = {}
    for g in all_groups:
        b = g["recommended_bits"]
        bits_dist[b] = bits_dist.get(b, 0) + 1
    print(f"Average bits: {avg_bits:.2f}")
    for bits, count in sorted(bits_dist.items()):
        print(f"  {bits}-bit: {count} groups")

    print("\nTop 10 most sensitive (preserve these):")
    for g in sorted(all_groups, key=lambda x: x["kl_divergence"], reverse=True)[:10]:
        print(f"  {g['name']:40s} KL={g['kl_divergence']:.4f} -> {g['recommended_bits']}-bit")

    print("\nTop 10 crush candidates:")
    for g in sorted(all_groups, key=lambda x: x["kl_divergence"])[:10]:
        print(f"  {g['name']:40s} KL={g['kl_divergence']:.6f} -> {g['recommended_bits']}-bit")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Model Osmosis — sensitivity analyzer")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--output", default="sensitivity_report.json", help="Output JSON path")
    parser.add_argument("--samples", type=int, default=100, help="Calibration samples")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")
    parser.add_argument("--dtype", default=None, help="Torch dtype (float16, bfloat16)")
    args = parser.parse_args()
    analyze(args.model, args.output, args.samples, args.max_length, args.dtype)


if __name__ == "__main__":
    main()
