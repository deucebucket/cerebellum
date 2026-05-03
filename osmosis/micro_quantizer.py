"""AWQ-style sensitivity-guided weight pre-scaling for Qwen 3.5.

Scales important weight columns UP before quantization so they lose
less precision, then compensates by inverse-scaling the preceding
RMSNorm. Mathematical equivalence preserved — no runtime changes.

The idea: standard quantization treats all columns equally. But some
channels carry far more information than others. By amplifying high-
importance channels before quantization and de-amplifying the preceding
norm, the quantizer allocates more of its fixed bit budget to channels
that matter.

Pipeline:
  1. Load fp16 model
  2. (Optional) Run calibration prompts to measure activation magnitudes
  3. Compute per-column importance = L2 * maxabs * variance
  4. Combine weight + activation sensitivity (AWQ alpha blending)
  5. Scale weight columns, compensate RMSNorm weights
  6. Save modified fp16 model (ready for standard GGUF quantization)

Usage:
    python -m osmosis.micro_quantizer \\
        --model path/to/fp16 \\
        --output path/to/prescaled \\
        --alpha 0.5 \\
        --layers 0-47 \\
        -v
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_channel_sensitivity(weight: torch.Tensor) -> torch.Tensor:
    """Per-column importance = L2 * maxabs * variance, normalized 0-1."""
    w = weight.float()
    l2 = torch.norm(w, p=2, dim=0)
    maxabs = w.abs().amax(dim=0)
    var = w.var(dim=0, unbiased=False)
    importance = l2 * maxabs * var
    lo, hi = importance.min(), importance.max()
    if hi - lo < 1e-10:
        return torch.ones(weight.shape[1], device=weight.device)
    return (importance - lo) / (hi - lo)


def compute_scaling_factors(
    weight_sensitivity: torch.Tensor,
    activation_sensitivity: torch.Tensor | None = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    """AWQ formula: s = (sens / mean(sens))^alpha, clamped [0.5, 2.0]."""
    if activation_sensitivity is not None:
        combined = weight_sensitivity * 0.5 + activation_sensitivity * 0.5
    else:
        combined = weight_sensitivity

    mean_s = combined.mean()
    if mean_s < 1e-10:
        return torch.ones_like(combined)

    scales = (combined / mean_s).pow(alpha)
    return scales.clamp(0.5, 2.0)


def compute_activation_sensitivity(
    model,
    tokenizer,
    prompts: list[str],
    target_layers: list[int],
    max_length: int = 128,
) -> dict[int, torch.Tensor]:
    """Hook into RMSNorm inputs to capture per-channel activation magnitudes."""
    activation_stats = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            mag = x.float().abs().mean(dim=(0, 1))
            if layer_idx not in activation_stats:
                activation_stats[layer_idx] = mag.clone()
            else:
                activation_stats[layer_idx] += mag
        return hook_fn

    for idx in target_layers:
        layer = model.model.layers[idx]
        h = layer.input_layernorm.register_forward_hook(make_hook(idx))
        hooks.append(h)

    model.eval()
    count = 0
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", max_length=max_length,
                               truncation=True).to(model.device)
            model(**tokens)
            count += 1

    for h in hooks:
        h.remove()

    for idx in activation_stats:
        activation_stats[idx] /= count
        s = activation_stats[idx]
        lo, hi = s.min(), s.max()
        if hi - lo > 1e-10:
            activation_stats[idx] = (s - lo) / (hi - lo)
        else:
            activation_stats[idx] = torch.ones_like(s)

    return activation_stats


CALIBRATION_PROMPTS = [
    "Explain the concept of recursion in programming with an example.",
    "What are the main differences between TCP and UDP protocols?",
    "Write a Python function that implements binary search.",
    "Describe the architecture of a modern CPU pipeline.",
    "What is the capital of France and what is it known for?",
    "Solve: if 2x + 5 = 17, find x. Show your work.",
    "Explain how a hash table works and its time complexity.",
    "What are the SOLID principles in object-oriented design?",
]


def apply_awq_to_qwen35(
    model,
    target_layers: list[int],
    alpha: float = 0.5,
    activation_stats: dict[int, torch.Tensor] | None = None,
    verbose: bool = False,
) -> dict:
    """Scale attention + MLP input columns, compensate RMSNorm weights.

    Qwen 3.5 is a hybrid architecture: 24 linear_attn layers (GatedDeltaNet)
    + 8 self_attn layers (standard attention), every 4th layer is self_attn.
    """
    stats = {"layers_processed": 0, "avg_scale_range": []}

    for idx in target_layers:
        layer = model.model.layers[idx]
        act_sens = activation_stats.get(idx) if activation_stats else None
        is_linear = hasattr(layer, "linear_attn")

        if is_linear:
            attn = layer.linear_attn
            proj_weights = {
                "in_proj_qkv": attn.in_proj_qkv.weight.data,
                "in_proj_z": attn.in_proj_z.weight.data,
                "in_proj_a": attn.in_proj_a.weight.data,
                "in_proj_b": attn.in_proj_b.weight.data,
            }
        else:
            attn = layer.self_attn
            proj_weights = {
                "q_proj": attn.q_proj.weight.data,
                "k_proj": attn.k_proj.weight.data,
                "v_proj": attn.v_proj.weight.data,
            }

        sens_list = [compute_channel_sensitivity(w) for w in proj_weights.values()]
        attn_sens = sum(sens_list) / len(sens_list)

        attn_scales = compute_scaling_factors(attn_sens, act_sens, alpha)

        for name, w in proj_weights.items():
            orig_dtype = w.dtype
            getattr(attn, name).weight.data = (w.float() * attn_scales.unsqueeze(0)).to(orig_dtype)
        norm_dtype = layer.input_layernorm.weight.dtype
        layer.input_layernorm.weight.data = (layer.input_layernorm.weight.data.float() / attn_scales).to(norm_dtype)

        # --- MLP block: scale gate/up columns, compensate post_attention_layernorm ---
        gate_w = layer.mlp.gate_proj.weight.data
        up_w = layer.mlp.up_proj.weight.data

        gate_sens = compute_channel_sensitivity(gate_w)
        up_sens = compute_channel_sensitivity(up_w)
        mlp_sens = (gate_sens + up_sens) / 2.0

        mlp_act_sens = activation_stats.get(idx + 1000) if activation_stats else None
        mlp_scales = compute_scaling_factors(mlp_sens, mlp_act_sens, alpha)

        mlp_dtype = gate_w.dtype
        layer.mlp.gate_proj.weight.data = (gate_w.float() * mlp_scales.unsqueeze(0)).to(mlp_dtype)
        layer.mlp.up_proj.weight.data = (up_w.float() * mlp_scales.unsqueeze(0)).to(mlp_dtype)
        pnorm_dtype = layer.post_attention_layernorm.weight.dtype
        layer.post_attention_layernorm.weight.data = (layer.post_attention_layernorm.weight.data.float() / mlp_scales).to(pnorm_dtype)

        scale_range = (attn_scales.min().item(), attn_scales.max().item())
        stats["avg_scale_range"].append(scale_range)
        stats["layers_processed"] += 1

        if verbose:
            ltype = "linear" if is_linear else "selfattn"
            print(f"  Layer {idx:2d} ({ltype:8s}): attn [{scale_range[0]:.3f}, {scale_range[1]:.3f}] "
                  f"mlp [{mlp_scales.min():.3f}, {mlp_scales.max():.3f}]")

    return stats


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    result = []
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def run_prescaling(
    model_path: str,
    output_path: str,
    layers: str = "all",
    alpha: float = 0.5,
    calibrate: bool = True,
    num_samples: int = 8,
    verbose: bool = False,
):
    """Load model, calibrate, apply scaling, save modified fp16 model."""
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    num_layers = len(model.model.layers)
    target_layers = parse_layers(layers, num_layers)
    print(f"Target layers: {len(target_layers)}/{num_layers} (alpha={alpha})")

    activation_stats = None
    if calibrate:
        print(f"Running activation calibration ({num_samples} prompts)...")
        prompts = CALIBRATION_PROMPTS[:num_samples]
        activation_stats = compute_activation_sensitivity(
            model, tokenizer, prompts, target_layers
        )
        print(f"  Captured activation stats for {len(activation_stats)} layers")

    print("Moving model to CPU for weight scaling...")
    model = model.cpu()
    if activation_stats:
        activation_stats = {k: v.cpu() for k, v in activation_stats.items()}
    torch.cuda.empty_cache()

    print("Applying AWQ pre-scaling...")
    t0 = time.time()
    stats = apply_awq_to_qwen35(model, target_layers, alpha, activation_stats, verbose)
    elapsed = time.time() - t0
    print(f"  Scaled {stats['layers_processed']} layers in {elapsed:.1f}s")

    if stats["avg_scale_range"]:
        mins = [r[0] for r in stats["avg_scale_range"]]
        maxs = [r[1] for r in stats["avg_scale_range"]]
        print(f"  Scale range: [{min(mins):.3f}, {max(maxs):.3f}]")

    print(f"Saving pre-scaled model to {output_path}...")
    model.save_pretrained(output_path, max_shard_size="4GB")
    tokenizer.save_pretrained(output_path)

    meta = {
        "source_model": model_path,
        "alpha": alpha,
        "layers": layers,
        "num_layers_scaled": stats["layers_processed"],
        "calibrated": calibrate,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = Path(output_path) / "awq_prescale_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")
    print("Done. Now quantize this model with llama-quantize as usual.")


def main():
    parser = argparse.ArgumentParser(description="AWQ-style weight pre-scaling for Qwen 3.5")
    parser.add_argument("--model", required=True, help="Path to fp16 HuggingFace model")
    parser.add_argument("--output", required=True, help="Output path for pre-scaled model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Scaling exponent (0=none, 1=aggressive)")
    parser.add_argument("--layers", default="all", help="Layer spec: 'all' or '0-10,20-30'")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip activation calibration")
    parser.add_argument("--num-samples", type=int, default=8, help="Calibration prompts to use")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    run_prescaling(
        args.model, args.output,
        layers=args.layers,
        alpha=args.alpha,
        calibrate=not args.no_calibrate,
        num_samples=args.num_samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
