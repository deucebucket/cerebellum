#!/usr/bin/env python3
"""Apply pre-computed refusal direction ablation to a model.

Loads direction vectors from a .npz file (output of refusal_direction.py),
ablates the top-N layers, and saves the modified model.

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
import sys
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osmosis.refusal_direction import ablate_refusal_direction


def main():
    parser = argparse.ArgumentParser(description="Apply refusal ablation from saved directions")
    parser.add_argument("--model", required=True, help="HF model path (fp16)")
    parser.add_argument("--directions", required=True, help="Path to .npz direction vectors")
    parser.add_argument("--report", required=True, help="Path to .json report (for top layers)")
    parser.add_argument("--output", required=True, help="Save ablated model here")
    parser.add_argument("--top-n", type=int, default=10, help="Ablate top N layers by magnitude")
    parser.add_argument("--strength", type=float, default=1.0, help="Ablation strength (0=none, 1=full)")
    parser.add_argument("--device", default="auto")
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

    if args.verbose:
        print(f"\nLoading model: {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    if args.verbose:
        print(f"  Loaded in {time.time() - t0:.1f}s")

    if args.verbose:
        print(f"\nAblating {len(top_layers)} layers...")
    n_modified = ablate_refusal_direction(
        model, directions,
        layers_to_ablate=top_layers,
        strength=args.strength,
    )
    print(f"  Modified {n_modified} projection weights across {len(top_layers)} layers")

    if args.verbose:
        print(f"\nSaving ablated model to {args.output}...")
    t0 = time.time()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"  Saved in {time.time() - t0:.1f}s")

    meta = {
        "source_model": args.model,
        "ablation_strength": args.strength,
        "top_n_layers": args.top_n,
        "ablated_layers": top_layers,
        "n_modified_weights": n_modified,
        "peak_layer": report["peak_layer"],
        "peak_magnitude": report["peak_magnitude"],
    }
    meta_path = os.path.join(args.output, "ablation_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
