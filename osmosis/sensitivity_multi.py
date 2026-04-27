"""Multi-depth sensitivity analyzer for Osmosis.

Measures how each weight group degrades at Q2, Q3, Q4, Q6, Q8 bit depths.
Produces a sensitivity curve per tensor — not just one number — so the budget
allocator can make informed staggering decisions.

Simulates GGUF-style block quantization: group_size=32 elements, one scale
per group, round-to-nearest N-bit integer values.

Usage:
    python -m osmosis.sensitivity_multi \
        --model .hf_cache/models--Qwen--Qwen3.5-9B/snapshots/<hash> \
        --output osmosis-qwen3.5-9b/sensitivity_multi.json \
        --samples 32 -v
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Python 3.14 breaks datasets/dill — use pyarrow directly
try:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

BIT_DEPTHS = [2, 3, 4, 6, 8]
GROUP_SIZE = 32


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


def load_calibration_prompts(num_samples=32):
    """Load calibration text. Tries pyarrow (Python 3.14 safe), falls back to built-in."""
    if HAS_PYARROW:
        try:
            path = hf_hub_download(
                repo_id="Salesforce/wikitext", repo_type="dataset",
                filename="wikitext-2-raw-v1/train/0000.parquet",
            )
            table = pq.read_table(path)
            texts = table.column("text").to_pylist()
            prompts = [t for t in texts if isinstance(t, str) and len(t) > 80][:num_samples]
            if len(prompts) >= num_samples // 2:
                return prompts
        except Exception as e:
            print(f"  pyarrow wikitext load failed: {e}")

    print("  Using synthetic calibration prompts")
    return [
        "The theory of general relativity describes gravity as a geometric property of spacetime.",
        "In computer science, a hash table is a data structure that implements an associative array.",
        "The mitochondria is the powerhouse of the cell, producing ATP through oxidative phosphorylation.",
        "Shakespeare wrote many plays including Hamlet, Macbeth, and A Midsummer Night's Dream.",
        "The French Revolution began in 1789 and led to profound political and social changes in France.",
        "Machine learning algorithms can be broadly categorized into supervised, unsupervised, and reinforcement learning.",
        "The human genome contains approximately 3 billion base pairs of DNA organized into 23 chromosome pairs.",
        "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement.",
        "The Treaty of Westphalia in 1648 established the principle of state sovereignty in Europe.",
        "Neural networks consist of interconnected nodes organized in layers that process information.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy.",
        "The Pythagorean theorem states that in a right triangle, a squared plus b squared equals c squared.",
        "Economic theory suggests that markets tend toward equilibrium where supply meets demand.",
        "The discovery of penicillin by Alexander Fleming revolutionized medicine and saved millions of lives.",
        "Artificial intelligence has made remarkable progress in natural language processing and computer vision.",
        "The laws of thermodynamics govern energy transfer and the direction of natural processes.",
        "Democracy as a form of government originated in ancient Athens around the 5th century BCE.",
        "The periodic table organizes chemical elements by atomic number and recurring chemical properties.",
        "Climate change is driven primarily by the accumulation of greenhouse gases in the atmosphere.",
        "The invention of the printing press by Gutenberg transformed the spread of knowledge in Europe.",
        "Protein folding determines the three-dimensional structure and function of biological molecules.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "Modern cryptography relies on mathematical problems that are computationally difficult to solve.",
        "The Renaissance marked a period of cultural rebirth in Europe from the 14th to 17th century.",
        "Evolutionary biology explains the diversity of life through natural selection and genetic variation.",
        "The Internet protocol suite provides end-to-end data communication specifying how data should be formatted.",
        "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
        "The Industrial Revolution transformed manufacturing processes starting in Britain in the late 1700s.",
        "Calculus was independently developed by Newton and Leibniz in the late 17th century.",
        "The human brain contains roughly 86 billion neurons connected by trillions of synapses.",
        "Plate tectonics explains the large-scale motion of Earth's lithosphere as rigid plates.",
        "The Standard Model of particle physics describes the fundamental forces and elementary particles.",
    ][:num_samples]


# Qwen 3.5 hybrid: linear attention layers + full attention every 4th
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

MIN_PARAM_COUNT = 100_000


def get_weight_groups(model, layer_idx):
    """Extract quantizable weight tensors from a model layer."""
    layer = model.model.layers[layer_idx]
    groups = {}

    for group_key, (parent_name, weight_name) in QWEN35_WEIGHT_GROUPS.items():
        parent = getattr(layer, parent_name, None)
        if parent is None:
            continue
        module = getattr(parent, weight_name, None)
        if module is None:
            continue
        if not hasattr(module, "weight"):
            continue
        w = module.weight
        if w.numel() < MIN_PARAM_COUNT:
            continue
        groups[group_key] = module

    return groups


@torch.no_grad()
def capture_baseline(model, input_ids):
    """Run model forward and capture final logits."""
    output = model(input_ids)
    return F.log_softmax(output.logits[0].float(), dim=-1)


@torch.no_grad()
def measure_group_at_depth(model, module, bits, input_ids, baseline_logprobs):
    """Quantize one weight group to N bits, measure degradation, restore."""
    original = module.weight.data.clone()
    device = module.weight.device
    dtype = module.weight.dtype

    quantized = quantize_blockwise(original, bits, GROUP_SIZE).to(device=device, dtype=dtype)
    module.weight.data = quantized

    output = model(input_ids)
    crushed_logprobs = F.log_softmax(output.logits[0].float(), dim=-1)

    baseline_probs = baseline_logprobs.exp()
    kl = F.kl_div(crushed_logprobs, baseline_probs, reduction="batchmean", log_target=False).item()

    cos = F.cosine_similarity(
        baseline_logprobs.reshape(1, -1),
        crushed_logprobs.reshape(1, -1),
    ).item()

    mse = F.mse_loss(crushed_logprobs, baseline_logprobs).item()

    module.weight.data = original
    return {"kl": kl, "cosine": cos, "mse": mse}


def analyze_multi(model_path, output_path, num_samples=32, max_length=128, verbose=False):
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

    print("Loading calibration prompts...")
    prompts = load_calibration_prompts(num_samples)
    print(f"  {len(prompts)} prompts loaded")

    all_input_ids = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_length)
        all_input_ids.append(tokens["input_ids"].to(model.device))

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    print(f"Testing bit depths: {BIT_DEPTHS}")
    print(f"Calibration samples: {len(prompts)}")

    all_results = []
    total_groups = 0
    start_time = time.time()

    for layer_idx in range(num_layers):
        groups = get_weight_groups(model, layer_idx)
        if not groups:
            continue

        layer_start = time.time()
        if verbose:
            print(f"\n--- Layer {layer_idx} ({len(groups)} groups) ---")

        for group_key, module in groups.items():
            param_count = module.weight.numel()
            shape = list(module.weight.shape)
            name = f"layer_{layer_idx}.{group_key}"

            depth_results = {}
            for bits in BIT_DEPTHS:
                kls, cosines, mses = [], [], []
                for input_ids in all_input_ids:
                    baseline = capture_baseline(model, input_ids)
                    metrics = measure_group_at_depth(model, module, bits, input_ids, baseline)
                    kls.append(metrics["kl"])
                    cosines.append(metrics["cosine"])
                    mses.append(metrics["mse"])

                depth_results[bits] = {
                    "kl_mean": float(np.mean(kls)),
                    "kl_max": float(np.max(kls)),
                    "cosine_mean": float(np.mean(cosines)),
                    "cosine_min": float(np.min(cosines)),
                    "mse_mean": float(np.mean(mses)),
                }

            cosines_by_depth = {b: depth_results[b]["cosine_mean"] for b in BIT_DEPTHS}
            jumps = {}
            for i in range(len(BIT_DEPTHS) - 1):
                lo, hi = BIT_DEPTHS[i], BIT_DEPTHS[i + 1]
                jumps[f"{lo}->{hi}"] = cosines_by_depth[hi] - cosines_by_depth[lo]

            recommended = BIT_DEPTHS[-1]
            for b in BIT_DEPTHS:
                if cosines_by_depth[b] > 0.95:
                    recommended = b
                    break

            result = {
                "name": name,
                "layer_idx": layer_idx,
                "group_type": group_key,
                "param_count": param_count,
                "shape": shape,
                "depths": depth_results,
                "cosine_by_depth": cosines_by_depth,
                "quality_jumps": jumps,
                "recommended_bits": recommended,
            }
            all_results.append(result)
            total_groups += 1

            if verbose:
                curve = " | ".join(
                    f"Q{b}:{cosines_by_depth[b]:.4f}" for b in BIT_DEPTHS
                )
                print(f"  {name:45s} {curve}  rec={recommended}b")

        layer_elapsed = time.time() - layer_start
        elapsed = time.time() - start_time
        layers_done = layer_idx + 1
        est_total = elapsed / layers_done * num_layers
        est_remaining = est_total - elapsed
        if verbose:
            print(f"  Layer {layer_idx} took {layer_elapsed:.0f}s | ETA: {est_remaining/60:.0f}m remaining")

        # Incremental save every 4 layers
        if layer_idx % 4 == 3 or layer_idx == num_layers - 1:
            partial = {
                "model": str(model_path),
                "status": "in_progress" if layer_idx < num_layers - 1 else "complete",
                "layers_done": layer_idx + 1,
                "num_layers": num_layers,
                "groups": all_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(output_path, "w") as f:
                json.dump(partial, f, indent=2)
            if verbose:
                print(f"  [checkpoint saved: {len(all_results)} groups]")

    rec_dist = {}
    for r in all_results:
        b = r["recommended_bits"]
        rec_dist[b] = rec_dist.get(b, 0) + 1

    total_params = sum(r["param_count"] for r in all_results)
    weighted_bits = sum(r["param_count"] * r["recommended_bits"] for r in all_results)
    avg_bits = weighted_bits / total_params if total_params else 0

    report = {
        "model": str(model_path),
        "status": "complete",
        "num_layers": num_layers,
        "num_groups": total_groups,
        "calibration_samples": len(prompts),
        "bit_depths_tested": BIT_DEPTHS,
        "group_size": GROUP_SIZE,
        "total_params_analyzed": total_params,
        "recommended_average_bits": round(avg_bits, 2),
        "recommendation_distribution": rec_dist,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": all_results,
    }

    print(f"\n{'='*60}")
    print(f"Multi-depth sensitivity analysis complete")
    print(f"  {total_groups} weight groups across {num_layers} layers")
    print(f"  Tested at {len(BIT_DEPTHS)} bit depths: {BIT_DEPTHS}")
    print(f"  Recommended average: {avg_bits:.2f} bits")
    print(f"  Distribution: {rec_dist}")
    print(f"  Elapsed: {(time.time() - start_time)/60:.1f} minutes")

    by_rec = sorted(all_results, key=lambda x: x["recommended_bits"])
    crushable = [r for r in by_rec if r["recommended_bits"] <= 2]
    sensitive = [r for r in by_rec if r["recommended_bits"] >= 6]

    if crushable:
        print(f"\n  CRUSHABLE ({len(crushable)} groups, can survive Q2):")
        for r in crushable[:10]:
            c2 = r["cosine_by_depth"][2]
            print(f"    {r['name']:45s} Q2 cosine={c2:.4f}")

    if sensitive:
        print(f"\n  SENSITIVE ({len(sensitive)} groups, need Q6+):")
        for r in sensitive[:10]:
            c2 = r["cosine_by_depth"][2]
            c4 = r["cosine_by_depth"][4]
            print(f"    {r['name']:45s} Q2={c2:.4f} Q4={c4:.4f}")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Osmosis multi-depth sensitivity analyzer"
    )
    parser.add_argument("--model", required=True, help="HF model path or local snapshot")
    parser.add_argument("--output", default="sensitivity_multi.json", help="Output JSON")
    parser.add_argument("--samples", type=int, default=32, help="Calibration samples")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    analyze_multi(args.model, args.output, args.samples, args.max_length, args.verbose)


if __name__ == "__main__":
    main()
