"""Find the refusal direction in a transformer model's residual stream.

Based on "Refusal in Language Models Is Mediated by a Single Direction"
(arXiv:2406.11717). The core idea: refusal behavior is encoded as a single
direction in the residual stream. By computing the difference in mean
activations between harmful (refused) and harmless (compliant) prompts,
we can isolate this direction and ablate it.

This module works at two levels:
1. PyTorch (fp16 model) — hooks into residual stream for exact direction finding
2. llama.cpp API — uses logprob/generation differences as a proxy signal

Usage:
    python -m osmosis.refusal_direction \
        --model Qwen/Qwen3.6-27B \
        --output refusal_direction.json \
        -v
"""
import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from osmosis.refusal_prompts import REFUSAL_PAIRS, NEUTRAL_PROMPTS


class ResidualStreamRecorder:
    """Hook into residual stream (post-layernorm) at every layer."""

    def __init__(self, model):
        self.model = model
        self._hooks = []
        self.activations: dict[int, list[torch.Tensor]] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            last_token = x[:, -1, :].detach().float().cpu()
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(last_token)
        return hook_fn

    def register_hooks(self, layer_range: tuple[int, int] | None = None):
        n_layers = len(self.model.model.layers)
        start = layer_range[0] if layer_range else 0
        end = layer_range[1] if layer_range else n_layers

        for idx in range(start, min(end, n_layers)):
            layer = self.model.model.layers[idx]
            norm = getattr(layer, "input_layernorm", None)
            if norm is None:
                norm = getattr(layer, "ln_1", None)
            if norm is not None:
                h = norm.register_forward_hook(self._make_hook(idx))
                self._hooks.append(h)
        return len(self._hooks)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.activations.clear()

    def get_mean_activations(self) -> dict[int, torch.Tensor]:
        means = {}
        for layer_idx, tensors in self.activations.items():
            stacked = torch.cat(tensors, dim=0)
            means[layer_idx] = stacked.mean(dim=0)
        return means


def compute_refusal_direction(
    model,
    tokenizer,
    pairs: list[dict],
    max_length: int = 256,
    layer_range: tuple[int, int] | None = None,
    verbose: bool = False,
) -> dict:
    """Compute the refusal direction as difference-in-means.

    Returns per-layer refusal direction vectors and metadata.
    """
    recorder = ResidualStreamRecorder(model)
    n_hooks = recorder.register_hooks(layer_range)
    if verbose:
        print(f"  Registered {n_hooks} residual stream hooks")

    model.eval()

    if verbose:
        print(f"  Recording refused prompts ({len(pairs)} prompts)...")
    for i, pair in enumerate(pairs):
        tokens = tokenizer(
            pair["refused"],
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            model(**tokens)
        if verbose and (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(pairs)}]")

    refused_means = recorder.get_mean_activations()
    recorder.clear()

    if verbose:
        print(f"  Recording compliant prompts ({len(pairs)} prompts)...")
    for i, pair in enumerate(pairs):
        tokens = tokenizer(
            pair["compliant"],
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            model(**tokens)
        if verbose and (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(pairs)}]")

    compliant_means = recorder.get_mean_activations()
    recorder.remove_hooks()

    directions = {}
    layer_magnitudes = {}

    for layer_idx in sorted(refused_means.keys()):
        if layer_idx not in compliant_means:
            continue
        direction = refused_means[layer_idx] - compliant_means[layer_idx]
        magnitude = direction.norm().item()
        directions[layer_idx] = direction
        layer_magnitudes[layer_idx] = magnitude

    if layer_magnitudes:
        peak_layer = max(layer_magnitudes, key=layer_magnitudes.get)
        peak_magnitude = layer_magnitudes[peak_layer]
    else:
        peak_layer = -1
        peak_magnitude = 0.0

    if verbose:
        print(f"\n  Refusal direction magnitudes by layer:")
        for idx in sorted(layer_magnitudes.keys()):
            mag = layer_magnitudes[idx]
            bar = "#" * int(mag / peak_magnitude * 40) if peak_magnitude > 0 else ""
            marker = " <<<" if idx == peak_layer else ""
            print(f"    Layer {idx:3d}: {mag:8.4f} {bar}{marker}")
        print(f"\n  Peak refusal layer: {peak_layer} (magnitude {peak_magnitude:.4f})")

    return {
        "directions": directions,
        "magnitudes": layer_magnitudes,
        "peak_layer": peak_layer,
        "peak_magnitude": peak_magnitude,
        "n_refused": len(pairs),
        "n_compliant": len(pairs),
    }


def ablate_refusal_direction(
    model,
    directions: dict[int, torch.Tensor],
    layers_to_ablate: list[int] | None = None,
    strength: float = 1.0,
):
    """Remove the refusal direction from model weights.

    For each target layer, projects out the refusal direction from the
    output weights of attention and MLP projections. This permanently
    modifies the model weights.

    strength: 0.0 = no ablation, 1.0 = full ablation
    """
    if layers_to_ablate is None:
        layers_to_ablate = sorted(directions.keys())

    modified = 0
    for layer_idx in layers_to_ablate:
        if layer_idx not in directions:
            continue

        direction = directions[layer_idx].to(model.device)
        direction = direction / direction.norm()

        layer = model.model.layers[layer_idx]

        targets = []
        attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        if attn:
            for name in ["o_proj"]:
                proj = getattr(attn, name, None)
                if proj is not None:
                    targets.append((f"layers.{layer_idx}.attn.{name}", proj))

        mlp = getattr(layer, "mlp", None)
        if mlp:
            for name in ["down_proj"]:
                proj = getattr(mlp, name, None)
                if proj is not None:
                    targets.append((f"layers.{layer_idx}.mlp.{name}", proj))

        for proj_name, proj in targets:
            W = proj.weight.data.float()
            proj_component = torch.outer(direction, direction) @ W
            proj.weight.data -= (strength * proj_component).to(proj.weight.dtype)
            modified += 1

    return modified


def save_direction_report(
    result: dict,
    output_path: str | Path,
):
    """Save refusal direction analysis to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "peak_layer": result["peak_layer"],
        "peak_magnitude": result["peak_magnitude"],
        "n_refused": result["n_refused"],
        "n_compliant": result["n_compliant"],
        "magnitudes": {str(k): v for k, v in result["magnitudes"].items()},
        "top_layers": sorted(
            result["magnitudes"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10],
    }

    direction_path = path.with_suffix(".npz")
    direction_arrays = {
        f"layer_{k}": v.numpy() for k, v in result["directions"].items()
    }
    np.savez_compressed(direction_path, **direction_arrays)

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {path}")
    print(f"Direction vectors saved to {direction_path}")
    return path, direction_path


def main():
    parser = argparse.ArgumentParser(description="Find refusal direction in LLM")
    parser.add_argument("--model", required=True, help="HF model path")
    parser.add_argument("--output", default="refusal_direction.json")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Ablation strength (0=none, 1=full)")
    parser.add_argument("--ablate", action="store_true",
                        help="Actually modify model weights")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Save ablated model to this path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    torch_dtype = getattr(torch, args.dtype)
    if args.verbose:
        print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )

    n_layers = len(model.model.layers)
    layer_range = (
        args.layer_start,
        args.layer_end if args.layer_end else n_layers,
    )

    if args.verbose:
        print(f"  {n_layers} layers, scanning {layer_range[0]}-{layer_range[1]}")
        print(f"  {len(REFUSAL_PAIRS)} refusal/compliance pairs")

    result = compute_refusal_direction(
        model, tokenizer, REFUSAL_PAIRS,
        max_length=args.max_length,
        layer_range=layer_range,
        verbose=args.verbose,
    )

    save_direction_report(result, args.output)

    if args.ablate:
        top_layers = sorted(
            result["magnitudes"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        layers_to_ablate = [idx for idx, _ in top_layers]

        if args.verbose:
            print(f"\nAblating refusal direction from layers: {layers_to_ablate}")
            print(f"  Strength: {args.strength}")

        n_modified = ablate_refusal_direction(
            model, result["directions"],
            layers_to_ablate=layers_to_ablate,
            strength=args.strength,
        )
        print(f"  Modified {n_modified} projection weights")

        if args.save_model:
            if args.verbose:
                print(f"  Saving ablated model to {args.save_model}")
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
            print(f"  Saved.")


if __name__ == "__main__":
    main()
