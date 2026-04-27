"""Overnight pipeline — download, analyze, crush, patch.

Usage:
    python -m osmosis.pipeline --model Qwen/Qwen3-27B --output-dir ./osmosis-qwen27b
"""
import argparse
import gc
import json
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from osmosis.sensitivity import WEIGHT_GROUPS, quantize_1bit


CALIBRATION_PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy, scientists discovered",
    "The fundamental theorem of calculus states that",
    "Once upon a time in a small village",
    "The chemical formula for water is H2O because",
    "According to recent research in machine learning",
    "The president announced today that the new policy",
    "In quantum mechanics, the uncertainty principle",
]


def phase0_download(model_name: str, cache_dir: Optional[str] = None) -> str:
    """Download model weights from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"  PHASE 0: Download {model_name}")
    print(f"{'='*60}")
    local_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        ignore_patterns=["*.bin", "*.pt", "consolidated*"],
    )
    print(f"Model cached at: {local_path}")
    return local_path


@torch.no_grad()
def phase1_activations(model_path: str, output_dir: Path, max_length: int = 128):
    """One slow forward pass with CPU offloading to capture per-layer activations."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Capture activations")
    print(f"{'='*60}")

    act_dir = output_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Loading model with CPU offload...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            activations[layer_idx] = inp.detach().cpu()
        return hook_fn

    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    num_layers = cfg.num_hidden_layers
    for i in range(num_layers):
        layer = model.model.layers[i]
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    print(f"Running {len(CALIBRATION_PROMPTS)} calibration prompts...")
    all_activations = {i: [] for i in range(num_layers)}

    for prompt in tqdm(CALIBRATION_PROMPTS, desc="Forward passes"):
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        model(**tokens)
        for i in range(num_layers):
            if i in activations:
                all_activations[i].append(activations[i])
        activations.clear()

    for h in hooks:
        h.remove()

    print("Saving activations to disk...")
    for i in tqdm(range(num_layers), desc="Saving"):
        acts = torch.cat(all_activations[i], dim=1)
        torch.save(acts, act_dir / f"layer_{i}.pt")
        all_activations[i] = None

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Activations saved to {act_dir}")
    return act_dir


@torch.no_grad()
def phase2_sensitivity(model_path: str, act_dir: Path, output_dir: Path,
                       max_length: int = 128):
    """Stream one layer at a time to GPU, crush each tensor, measure MSE."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Sensitivity analysis (streaming)")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Loading model on CPU for streaming analysis...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    print("Capturing baseline logits (CPU, slow)...")
    baseline_logits = []
    for prompt in tqdm(CALIBRATION_PROMPTS[:4], desc="Baseline"):
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        output = model(**tokens)
        log_probs = F.log_softmax(output.logits[0].float(), dim=-1)
        baseline_logits.append(log_probs)

    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    num_layers = cfg.num_hidden_layers
    all_groups = []

    checkpoint_path = output_dir / "sensitivity_checkpoint.json"
    start_layer = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as cf:
            checkpoint = json.load(cf)
        all_groups = checkpoint["groups"]
        start_layer = checkpoint["last_layer"] + 1
        print(f"  Resuming from layer {start_layer} ({len(all_groups)} groups already done)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for layer_idx in tqdm(range(start_layer, num_layers), desc="Layers", initial=start_layer, total=num_layers):
        layer = model.model.layers[layer_idx]

        modules = {}
        skip_types = ("RMSNorm", "LayerNorm", "SiLU", "GELU", "NewGELU")
        for child_name, child_module in layer.named_children():
            if type(child_module).__name__ in skip_types:
                continue
            for param_name, param_module in child_module.named_children():
                if type(param_module).__name__ in skip_types:
                    continue
                if hasattr(param_module, "weight") and param_module.weight is not None:
                    if param_module.weight.dim() >= 2:
                        modules[f"{child_name}.{param_name}"] = param_module
        if layer_idx == 0:
            print(f"\n  Layer 0 ({type(layer).__name__}): {list(modules.keys())}")

        for group_name, module in modules.items():
            original = module.weight.data.clone()
            magnitude = original.abs().mean().item()

            crushed = quantize_1bit(original)
            mse = F.mse_loss(crushed, original).item()

            if device.type == "cuda":
                orig_gpu = original.to(device)
                crush_gpu = crushed.to(device)
                cosine = F.cosine_similarity(
                    orig_gpu.flatten().unsqueeze(0),
                    crush_gpu.flatten().unsqueeze(0),
                ).item()
                del orig_gpu, crush_gpu
            else:
                cosine = F.cosine_similarity(
                    original.flatten().unsqueeze(0),
                    crushed.flatten().unsqueeze(0),
                ).item()

            module.weight.data = original

            kl_divs = []
            module.weight.data = crushed.to(module.weight.device)
            for i, prompt in enumerate(CALIBRATION_PROMPTS[:4]):
                tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                output = model(**tokens)
                crushed_logits = F.log_softmax(output.logits[0].float(), dim=-1)
                baseline_probs = torch.exp(baseline_logits[i])
                kl = F.kl_div(crushed_logits, baseline_probs, reduction="batchmean", log_target=False)
                kl_divs.append(kl.item())
            module.weight.data = original

            result = {
                "name": f"layer_{layer_idx}.{group_name}",
                "layer_idx": layer_idx,
                "group_type": group_name,
                "param_count": module.weight.numel(),
                "shape": list(module.weight.shape),
                "kl_divergence": float(np.mean(kl_divs)),
                "max_kl": float(np.max(kl_divs)),
                "mse": mse,
                "cosine_sim": cosine,
                "weight_magnitude": magnitude,
            }
            all_groups.append(result)

        with open(checkpoint_path, "w") as cf:
            json.dump({"last_layer": layer_idx, "groups": all_groups}, cf)
        print(f"  [checkpoint] layer {layer_idx}/{num_layers-1} done, {len(all_groups)} groups saved")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    kl_values = [g["kl_divergence"] for g in all_groups if np.isfinite(g["kl_divergence"])]
    p25 = np.percentile(kl_values, 25)
    p75 = np.percentile(kl_values, 75)
    for g in all_groups:
        kl = g["kl_divergence"]
        if not np.isfinite(kl) or kl > p75:
            g["recommended_bits"] = 4
        elif kl > p25:
            g["recommended_bits"] = 2
        else:
            g["recommended_bits"] = 1

    total_params = sum(g["param_count"] for g in all_groups)
    weighted = sum(g["param_count"] * g["recommended_bits"] for g in all_groups)
    avg_bits = weighted / total_params if total_params else 0

    report = {
        "model": model_path,
        "num_layers": num_layers,
        "num_groups": len(all_groups),
        "total_params": total_params,
        "average_bits": round(avg_bits, 2),
        "bit_distribution": {
            "1bit": sum(1 for g in all_groups if g["recommended_bits"] == 1),
            "2bit": sum(1 for g in all_groups if g["recommended_bits"] == 2),
            "4bit": sum(1 for g in all_groups if g["recommended_bits"] == 4),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": all_groups,
    }

    report_path = output_dir / "sensitivity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSensitivity report: {report_path}")
    print(f"Average bits: {avg_bits:.2f}")
    print(f"  1-bit: {report['bit_distribution']['1bit']} groups (crush)")
    print(f"  2-bit: {report['bit_distribution']['2bit']} groups (medium)")
    print(f"  4-bit: {report['bit_distribution']['4bit']} groups (preserve)")

    return report


def pack_1bit(tensor: torch.Tensor) -> tuple:
    """Quantize to 1-bit and pack 8 values per byte."""
    scale = tensor.abs().mean()
    signs = (tensor > 0).to(torch.uint8)
    flat = signs.flatten()
    pad_len = (8 - len(flat) % 8) % 8
    if pad_len:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8)])
    packed = torch.zeros(len(flat) // 8, dtype=torch.uint8)
    for bit in range(8):
        packed |= flat[bit::8] << bit
    return packed.numpy().tobytes(), float(scale), list(tensor.shape)


def pack_2bit(tensor: torch.Tensor) -> tuple:
    """Quantize to 2-bit (4 levels) and pack 4 values per byte."""
    scale = tensor.abs().max() / 1.5
    if scale == 0:
        scale = torch.tensor(1.0)
    normalized = (tensor / scale).clamp(-1.5, 1.5)
    levels = torch.round((normalized + 1.5) * (3.0 / 3.0)).clamp(0, 3).to(torch.uint8)
    flat = levels.flatten()
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8)])
    packed = torch.zeros(len(flat) // 4, dtype=torch.uint8)
    for pos in range(4):
        packed |= flat[pos::4] << (pos * 2)
    return packed.numpy().tobytes(), float(scale), list(tensor.shape)


def pack_4bit(tensor: torch.Tensor) -> tuple:
    """Quantize to 4-bit (16 levels) and pack 2 values per byte."""
    scale = tensor.abs().max() / 7.5
    if scale == 0:
        scale = torch.tensor(1.0)
    normalized = (tensor / scale).clamp(-7.5, 7.5)
    levels = torch.round(normalized + 7.5).clamp(0, 15).to(torch.uint8)
    flat = levels.flatten()
    pad_len = (2 - len(flat) % 2) % 2
    if pad_len:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8)])
    packed = torch.zeros(len(flat) // 2, dtype=torch.uint8)
    packed = flat[0::2] | (flat[1::2] << 4)
    return packed.numpy().tobytes(), float(scale), list(tensor.shape)


def phase3_crush(model_path: str, report_path: Path, output_dir: Path):
    """Stream through safetensors, quantize per sensitivity map, bit-pack."""
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Crush — mixed-precision quantization")
    print(f"{'='*60}")

    with open(report_path) as f:
        report = json.load(f)

    bit_map = {}
    for g in report["groups"]:
        bit_map[g["name"]] = g["recommended_bits"]

    crush_dir = output_dir / "crushed"
    crush_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_path)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        st_files = sorted(model_dir.glob("model*.safetensors"))
    print(f"Found {len(st_files)} safetensors files")

    manifest = {
        "model": report["model"],
        "format": "osmosis-v1",
        "average_bits": report["average_bits"],
        "layers": {},
    }

    total_original = 0
    total_packed = 0

    for st_file in tqdm(st_files, desc="Processing files"):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                original_bytes = tensor.numel() * tensor.element_size()
                total_original += original_bytes

                matched_group = None
                for group_key, bits in bit_map.items():
                    layer_part = group_key.split(".")[0].replace("layer_", "")
                    subpath = ".".join(group_key.split(".")[1:])
                    if f"layers.{layer_part}.{subpath}.weight" == key.replace("model.", "", 1):
                        matched_group = (group_key, bits)
                        break

                if matched_group is None:
                    out_path = crush_dir / f"{key.replace('.', '_')}.safetensors"
                    save_file({key: tensor}, str(out_path))
                    total_packed += original_bytes
                    manifest["layers"][key] = {
                        "bits": 16,
                        "file": out_path.name,
                        "shape": list(tensor.shape),
                    }
                    continue

                group_key, bits = matched_group

                if bits == 1:
                    packed_bytes, scale, shape = pack_1bit(tensor)
                elif bits == 2:
                    packed_bytes, scale, shape = pack_2bit(tensor)
                else:
                    packed_bytes, scale, shape = pack_4bit(tensor)

                out_path = crush_dir / f"{key.replace('.', '_')}.osm"
                with open(out_path, "wb") as out_f:
                    header = struct.pack("<BfII",
                                        bits,
                                        scale,
                                        shape[0],
                                        shape[1] if len(shape) > 1 else 1)
                    out_f.write(header)
                    out_f.write(packed_bytes)

                total_packed += len(packed_bytes) + 13
                manifest["layers"][key] = {
                    "bits": bits,
                    "file": out_path.name,
                    "scale": scale,
                    "shape": shape,
                    "original_bytes": original_bytes,
                    "packed_bytes": len(packed_bytes),
                }

    ratio = total_original / total_packed if total_packed else 0
    manifest["compression"] = {
        "original_bytes": total_original,
        "packed_bytes": total_packed,
        "ratio": round(ratio, 2),
    }

    manifest_path = crush_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCrush complete!")
    print(f"Original:   {total_original / 1e9:.2f} GB")
    print(f"Packed:     {total_packed / 1e9:.2f} GB")
    print(f"Ratio:      {ratio:.1f}x")
    print(f"Manifest:   {manifest_path}")

    return manifest


def run_pipeline(model_name: str, output_dir: str, cache_dir: Optional[str] = None,
                 skip_download: bool = False, skip_activations: bool = False,
                 skip_sensitivity: bool = False):
    """Run the full osmosis pipeline."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print(f"\n{'#'*60}")
    print(f"  MODEL OSMOSIS — {model_name}")
    print(f"  Output: {output}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    if skip_download:
        model_path = model_name
        print(f"Skipping download, using local path: {model_path}")
    else:
        model_path = phase0_download(model_name, cache_dir)

    act_dir = output / "activations"
    if not skip_activations:
        act_dir = phase1_activations(model_path, output)
    else:
        print("Skipping activation capture")

    report_path = output / "sensitivity_report.json"
    if not skip_sensitivity:
        report = phase2_sensitivity(model_path, act_dir, output)
    else:
        print("Skipping sensitivity analysis, loading existing report")
        with open(report_path) as f:
            report = json.load(f)

    manifest = phase3_crush(model_path, report_path, output)

    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Time: {hours}h {minutes}m")
    print(f"  Compression: {manifest['compression']['ratio']:.1f}x")
    print(f"  Average bits: {report['average_bits']}")
    print(f"  Output: {output}")
    print(f"{'#'*60}")


def main():
    parser = argparse.ArgumentParser(description="Model Osmosis — overnight pipeline")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--cache-dir", default=None, help="HF cache directory")
    parser.add_argument("--skip-download", action="store_true", help="Use --model as local path")
    parser.add_argument("--skip-activations", action="store_true", help="Skip phase 1")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip phase 2, use existing report")
    args = parser.parse_args()
    run_pipeline(args.model, args.output_dir, args.cache_dir,
                 args.skip_download, args.skip_activations, args.skip_sensitivity)


if __name__ == "__main__":
    main()
