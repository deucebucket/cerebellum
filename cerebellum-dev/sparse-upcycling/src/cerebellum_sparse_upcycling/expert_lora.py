from __future__ import annotations

import types
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn


LORA_SUFFIXES = (
    "experts.lora_gate_up_a",
    "experts.lora_gate_up_b",
    "experts.lora_down_a",
    "experts.lora_down_b",
)


def _init_lora_pair(
    module: nn.Module,
    a_name: str,
    b_name: str,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    b_init_std: float = 0.0,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn(a_shape, generator=generator, dtype=torch.float32) * 0.01
    if b_init_std > 0:
        b = torch.randn(b_shape, generator=generator, dtype=torch.float32) * b_init_std
    else:
        b = torch.zeros(b_shape, dtype=torch.float32)
    module.register_parameter(a_name, nn.Parameter(a.to(device=device, dtype=dtype)))
    module.register_parameter(b_name, nn.Parameter(b.to(device=device, dtype=dtype)))


def _lora_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    gate_up_scale = getattr(self, "lora_gate_up_scaling", 1.0)
    down_scale = getattr(self, "lora_down_scaling", 1.0)

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        gate_up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
        if hasattr(self, "lora_gate_up_a"):
            gate_up_lora = nn.functional.linear(current_state, self.lora_gate_up_a[expert_idx])
            gate_up = gate_up + nn.functional.linear(gate_up_lora, self.lora_gate_up_b[expert_idx]) * gate_up_scale
        gate, up = gate_up.chunk(2, dim=-1)

        current_hidden_states = self.act_fn(gate) * up
        down = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
        if hasattr(self, "lora_down_a"):
            down_lora = nn.functional.linear(current_hidden_states, self.lora_down_a[expert_idx])
            down = down + nn.functional.linear(down_lora, self.lora_down_b[expert_idx]) * down_scale
        current_hidden_states = down * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def iter_expert_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if name.endswith(".mlp.experts") and hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
            yield name, module


def attach_expert_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    target_gate_up: bool = True,
    target_down: bool = True,
    seed: int = 20260529,
    gate_up_b_init_std: float = 0.0,
    down_b_init_std: float = 0.0,
) -> list[str]:
    if rank <= 0:
        raise ValueError("rank must be positive")

    attached: list[str] = []
    for idx, (name, module) in enumerate(iter_expert_modules(model)):
        dtype = module.gate_up_proj.dtype
        device = module.gate_up_proj.device
        num_experts = int(module.num_experts)
        hidden = int(module.hidden_dim)
        intermediate = int(module.intermediate_dim)

        if target_gate_up and not hasattr(module, "lora_gate_up_a"):
            _init_lora_pair(
                module,
                "lora_gate_up_a",
                "lora_gate_up_b",
                (num_experts, rank, hidden),
                (num_experts, 2 * intermediate, rank),
                dtype=dtype,
                device=device,
                seed=seed + idx * 2,
                b_init_std=gate_up_b_init_std,
            )
            module.lora_gate_up_scaling = float(alpha) / float(rank)
            attached.extend([f"{name}.lora_gate_up_a", f"{name}.lora_gate_up_b"])

        if target_down and not hasattr(module, "lora_down_a"):
            _init_lora_pair(
                module,
                "lora_down_a",
                "lora_down_b",
                (num_experts, rank, intermediate),
                (num_experts, hidden, rank),
                dtype=dtype,
                device=device,
                seed=seed + idx * 2 + 1,
                b_init_std=down_b_init_std,
            )
            module.lora_down_scaling = float(alpha) / float(rank)
            attached.extend([f"{name}.lora_down_a", f"{name}.lora_down_b"])

        module.forward = types.MethodType(_lora_forward, module)

    return attached


def trainable_lora_names(model: nn.Module) -> list[str]:
    return [name for name, _param in model.named_parameters() if name.endswith(LORA_SUFFIXES)]


def save_adapter_delta(
    model: nn.Module,
    output_path: Path,
    *,
    include_router: bool = True,
    rank: int,
    alpha: float,
) -> list[str]:
    tensors = {}
    for name, param in model.named_parameters():
        is_lora = name.endswith(LORA_SUFFIXES)
        is_router = name.endswith(".mlp.gate.weight") or name.endswith(".mlp.shared_expert_gate.weight")
        if is_lora or (include_router and is_router):
            tensors[name] = param.detach().cpu().contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output_path,
        metadata={
            "format": "pt",
            "adapter": "qwen3_5_moe_fused_expert_lora",
            "lora_rank": str(rank),
            "lora_alpha": str(alpha),
            "includes_router": str(include_router),
        },
    )
    return sorted(tensors)


def load_adapter_delta(model: nn.Module, delta_path: Path, *, strict: bool = True) -> list[str]:
    with safe_open(str(delta_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    delta = load_file(str(delta_path), device="cpu")
    lora_keys = [key for key in delta if key.endswith(LORA_SUFFIXES)]
    if lora_keys:
        first_a = next((delta[key] for key in lora_keys if key.endswith("_a")), None)
        if first_a is None:
            raise ValueError("LoRA adapter has B tensors but no A tensor to infer rank")
        rank = int(metadata.get("lora_rank") or first_a.shape[1])
        alpha = float(metadata.get("lora_alpha") or rank)
        attach_expert_lora(model, rank=rank, alpha=alpha)

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
