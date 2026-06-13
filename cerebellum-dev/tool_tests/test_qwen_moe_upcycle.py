import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "sparse-upcycling"
    / "src"
    / "cerebellum_sparse_upcycling"
    / "qwen_moe_upcycle.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_moe_upcycle", SCRIPT_PATH)
qwen_moe_upcycle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = qwen_moe_upcycle
SPEC.loader.exec_module(qwen_moe_upcycle)


def swiglu_ffn(hidden, gate, up, down):
    return torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(hidden, gate))
        * torch.nn.functional.linear(hidden, up),
        down,
    )


def tiny_spec(tmp_path, *, expert_init="sliced_dense_ffn", shared_expert_init="dense_slice", shared=2):
    return qwen_moe_upcycle.UpcycleSpec(
        name="tiny",
        source_model="local/tiny",
        layers=1,
        hidden_size=2,
        dense_intermediate_size=4,
        num_experts=2,
        top_k=1,
        expert_intermediate_size=2,
        shared_expert_intermediate_size=shared,
        output_dir=tmp_path,
        expert_init=expert_init,
        shared_expert_init=shared_expert_init,
    )


def tiny_mlp():
    gate = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    up = torch.tensor([[0.2, -0.1], [0.4, -0.3], [0.6, -0.5], [0.8, -0.7]])
    down = torch.tensor([[0.2, 0.4, 0.6, 0.8], [-0.1, -0.3, -0.5, -0.7]])
    return gate, up, down


def test_v0_sliced_experts_preserve_all_expert_sum(tmp_path):
    gate, up, down = tiny_mlp()
    tensors = qwen_moe_upcycle.convert_mlp_layer(0, gate, up, down, tiny_spec(tmp_path))

    prefix = "model.language_model.layers.0.mlp"
    gate_up = tensors[f"{prefix}.experts.gate_up_proj"]
    down_experts = tensors[f"{prefix}.experts.down_proj"]
    hidden = torch.tensor([[0.25, -0.5]])
    dense = swiglu_ffn(hidden, gate, up, down)
    expert_sum = torch.zeros_like(dense)
    for expert_idx in range(gate_up.shape[0]):
        expert_gate, expert_up = torch.nn.functional.linear(hidden, gate_up[expert_idx]).chunk(2, dim=-1)
        expert_sum += torch.nn.functional.linear(
            torch.nn.functional.silu(expert_gate) * expert_up,
            down_experts[expert_idx],
        )

    torch.testing.assert_close(expert_sum, dense)


def test_v1_residual_bridge_shared_path_matches_dense(tmp_path):
    gate, up, down = tiny_mlp()
    spec = tiny_spec(
        tmp_path,
        expert_init="zero_residual_experts",
        shared_expert_init="dense_ffn_scaled_down_projection_2x",
        shared=4,
    )
    tensors = qwen_moe_upcycle.convert_mlp_layer(0, gate, up, down, spec)

    prefix = "model.language_model.layers.0.mlp"
    torch.testing.assert_close(tensors[f"{prefix}.experts.gate_up_proj"], torch.zeros((2, 4, 2)))
    torch.testing.assert_close(tensors[f"{prefix}.experts.down_proj"], torch.zeros((2, 2, 2)))
    torch.testing.assert_close(tensors[f"{prefix}.gate.weight"], torch.zeros((2, 2)))
    torch.testing.assert_close(tensors[f"{prefix}.shared_expert.gate_proj.weight"], gate)
    torch.testing.assert_close(tensors[f"{prefix}.shared_expert.up_proj.weight"], up)
    torch.testing.assert_close(tensors[f"{prefix}.shared_expert.down_proj.weight"], down * 2)
    torch.testing.assert_close(tensors[f"{prefix}.shared_expert_gate.weight"], torch.zeros((1, 2)))

    hidden = torch.tensor([[0.25, -0.5]])
    dense = swiglu_ffn(hidden, gate, up, down)
    shared = swiglu_ffn(
        hidden,
        tensors[f"{prefix}.shared_expert.gate_proj.weight"],
        tensors[f"{prefix}.shared_expert.up_proj.weight"],
        tensors[f"{prefix}.shared_expert.down_proj.weight"],
    )
    torch.testing.assert_close(0.5 * shared, dense)
