#!/usr/bin/env python3
"""Apply pre-computed refusal direction ablation directly to safetensors files.

Modifies o_proj and down_proj weights in target layers by projecting out
the refusal direction. Works shard-by-shard to avoid loading the full model.

Usage:
    python scripts/apply_ablation.py \
        --model /path/to/fp16/model \
        --directions /path/to/refusal_direction_27B.npz \
        --report /path/to/refusal_direction_27B.json \
        --output /path/to/ablated-model \
        --top-n 10 \
        --strength 1.0 \
        -v
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Apply refusal ablation from saved directions")
    parser.add_argument("--model", required=True, help="HF model path (fp16 safetensors)")
    parser.add_argument("--directions", required=True, help="Path to .npz direction vectors")
    parser.add_argument("--report", required=True, help="Path to .json report (for top layers)")
    parser.add_argument("--output", required=True, help="Save ablated model here")
    parser.add_argument("--top-n", type=int, default=10, help="Ablate top N layers by magnitude")
    parser.add_argument("--strength", type=float, default=1.0, help="Ablation strength (0=none, 1=full)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    top_layers = [layer_idx for layer_idx, _ in report["top_layers"][:args.top_n]]
    if args.verbose:
        print(f"Top {args.top_n} refusal layers: {top_layers}")
        print(f"Peak: layer {report['peak_layer']} (magnitude {report['peak_magnitude']:.2f})")
        print(f"Ablation strength: {args.strength}")

    data = np.load(args.directions)
    directions = {}
    for key in data.files:
        layer_idx = int(key.replace("layer_", ""))
        directions[layer_idx] = torch.from_numpy(data[key]).squeeze(0)
    if args.verbose:
        print(f"Loaded {len(directions)} direction vectors")

    index_path = os.path.join(args.model, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    prefix = ""
    for k in weight_map:
        if ".layers.0." in k:
            prefix = k.split("layers.0.")[0]
            break

    target_tensors = {}
    for layer_idx in top_layers:
        if layer_idx not in directions:
            continue
        has_self_attn = f"{prefix}layers.{layer_idx}.self_attn.o_proj.weight" in weight_map
        has_linear_attn = f"{prefix}layers.{layer_idx}.linear_attn.out_proj.weight" in weight_map

        if has_self_attn:
            target_tensors[f"{prefix}layers.{layer_idx}.self_attn.o_proj.weight"] = layer_idx
        elif has_linear_attn:
            target_tensors[f"{prefix}layers.{layer_idx}.linear_attn.out_proj.weight"] = layer_idx

        mlp_key = f"{prefix}layers.{layer_idx}.mlp.down_proj.weight"
        if mlp_key in weight_map:
            target_tensors[mlp_key] = layer_idx

    if args.verbose:
        print(f"Target tensors to modify: {len(target_tensors)}")

    shards_to_modify = {}
    for tensor_name, layer_idx in target_tensors.items():
        if tensor_name not in weight_map:
            if args.verbose:
                print(f"  WARNING: {tensor_name} not found in weight map, skipping")
            continue
        shard = weight_map[tensor_name]
        if shard not in shards_to_modify:
            shards_to_modify[shard] = []
        shards_to_modify[shard].append((tensor_name, layer_idx))

    if args.verbose:
        print(f"Shards to modify: {len(shards_to_modify)}")
        for shard, tensors in shards_to_modify.items():
            print(f"  {shard}: {[t[0].split('.')[-3] + '.' + t[0].split('.')[-2] for t in tensors]}")

    os.makedirs(args.output, exist_ok=True)

    all_shards = set(weight_map.values())
    modified_count = 0

    for shard_name in sorted(all_shards):
        src = os.path.join(args.model, shard_name)
        dst = os.path.join(args.output, shard_name)

        if shard_name not in shards_to_modify:
            if args.verbose:
                print(f"  Copying {shard_name} (no modifications needed)")
            shutil.copy2(src, dst)
            continue

        if args.verbose:
            print(f"  Modifying {shard_name}...")

        tensors = {}
        with safe_open(src, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        for tensor_name, layer_idx in shards_to_modify[shard_name]:
            if tensor_name not in tensors:
                continue

            direction = directions[layer_idx]
            direction = direction / direction.norm()

            W = tensors[tensor_name].float()
            proj_component = torch.outer(direction, direction) @ W
            tensors[tensor_name] = (W - args.strength * proj_component).to(tensors[tensor_name].dtype)
            modified_count += 1

            if args.verbose:
                proj_norm = proj_component.norm().item()
                print(f"    {tensor_name}: removed component (norm {proj_norm:.4f})")

        save_file(tensors, dst)

    for fname in os.listdir(args.model):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(args.model, fname)
        dst = os.path.join(args.output, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    print(f"\nModified {modified_count} tensors across {len(shards_to_modify)} shards")
    print(f"Ablated model saved to {args.output}")

    meta = {
        "source_model": args.model,
        "ablation_strength": args.strength,
        "top_n_layers": args.top_n,
        "ablated_layers": top_layers,
        "n_modified_weights": modified_count,
        "peak_layer": report["peak_layer"],
        "peak_magnitude": report["peak_magnitude"],
    }
    meta_path = os.path.join(args.output, "ablation_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
