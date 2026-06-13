from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


def load_router_delta(model, delta_path: Path, *, strict: bool = True) -> list[str]:
    delta = load_file(str(delta_path), device="cpu")
    params = dict(model.named_parameters())
    loaded = []
    missing = []
    for name, tensor in delta.items():
        param = params.get(name)
        if param is None:
            missing.append(name)
            continue
        if tuple(param.shape) != tuple(tensor.shape):
            raise ValueError(f"shape mismatch for {name}: model={tuple(param.shape)} delta={tuple(tensor.shape)}")
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
        loaded.append(name)
    if strict and missing:
        raise KeyError(f"delta keys not found in model: {missing[:10]}")
    return loaded
