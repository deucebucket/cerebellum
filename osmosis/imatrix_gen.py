"""Generate a llama.cpp imatrix file from sensitivity analysis.

Runs channel sensitivity computation on each weight tensor and writes
importance scores in the legacy imatrix binary format for llama-quantize --imatrix.

Auto-detects model architecture: works with Qwen 3.5 (hybrid linear+self attn),
Qwen 3.6 (dense), and standard transformers.

Usage:
    python -m osmosis.imatrix_gen \
        --model Qwen/Qwen3.6-27B \
        --output osmosis-qwen36-27b/osmosis_imatrix.dat \
        -v
"""
import argparse
import struct
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .micro_quantizer import compute_channel_sensitivity, compute_activation_sensitivity, CALIBRATION_PROMPTS


ATTN_PROJ_NAMES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
}

MLP_PROJ_NAMES = {
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


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


def resolve_weight(layer, dotted_path: str):
    obj = layer
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    if not hasattr(obj, "weight"):
        return None
    return obj.weight.data


def discover_layer_weights(layer, idx: int) -> dict[str, torch.Tensor]:
    found = {}
    for hf_path, gguf_name in {**ATTN_PROJ_NAMES, **MLP_PROJ_NAMES}.items():
        w = resolve_weight(layer, hf_path)
        if w is not None:
            found[f"blk.{idx}.{gguf_name}.weight"] = w
    return found


def generate_imatrix(
    model_path: str,
    output_path: str,
    calibrate: bool = True,
    num_samples: int = 8,
    verbose: bool = False,
):
    device_map = "cpu" if not calibrate else "auto"
    print(f"Loading model from {model_path} (device_map={device_map})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map=device_map,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} layers")

    activation_stats = None
    if calibrate:
        print(f"Running activation calibration ({num_samples} prompts)...")
        prompts = CALIBRATION_PROMPTS[:num_samples]
        target_layers = list(range(num_layers))
        try:
            activation_stats = compute_activation_sensitivity(
                model, tokenizer, prompts, target_layers
            )
            print(f"  Captured activation stats for {len(activation_stats)} layers")
        except Exception as e:
            print(f"  Calibration failed ({e}), continuing with weight-only sensitivity")

        print("Moving model to CPU for analysis...")
        model = model.cpu()
        if activation_stats:
            activation_stats = {k: v.cpu() for k, v in activation_stats.items()}
        torch.cuda.empty_cache()

    imatrix_data = {}
    t0 = time.time()

    print(f"Computing sensitivity for {num_layers} layers...")
    for idx in range(num_layers):
        layer = model.model.layers[idx]
        weights = discover_layer_weights(layer, idx)

        for gguf_name, w in weights.items():
            sens = compute_channel_sensitivity(w)

            if activation_stats and idx in activation_stats:
                act_sens = activation_stats[idx]
                if act_sens.shape[0] == sens.shape[0]:
                    sens = sens * 0.5 + act_sens * 0.5

            importance = sens.tolist()
            imatrix_data[gguf_name] = importance

            if verbose:
                mn, mx = min(importance), max(importance)
                print(f"  {gguf_name:40s} dim={len(importance):5d} range=[{mn:.4f}, {mx:.4f}]")

    elapsed = time.time() - t0
    print(f"Computed sensitivity for {len(imatrix_data)} tensors in {elapsed:.1f}s")

    print(f"Writing imatrix to {output_path}...")
    write_legacy_imatrix(output_path, imatrix_data, ncall=num_samples)

    file_size = Path(output_path).stat().st_size
    print(f"  {file_size / 1024:.1f} KB, {len(imatrix_data)} entries")
    print("Done. Use with: llama-quantize --imatrix", output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate imatrix from sensitivity analysis")
    parser.add_argument("--model", required=True, help="Path to HuggingFace model")
    parser.add_argument("--output", required=True, help="Output imatrix file path")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip activation calibration")
    parser.add_argument("--num-samples", type=int, default=8, help="Calibration prompts")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    generate_imatrix(
        args.model, args.output,
        calibrate=not args.no_calibrate,
        num_samples=args.num_samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
