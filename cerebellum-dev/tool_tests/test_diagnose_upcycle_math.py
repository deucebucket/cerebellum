import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "sparse-upcycling" / "scripts" / "diagnose_upcycle_math.py"
SPEC = importlib.util.spec_from_file_location("diagnose_upcycle_math", SCRIPT_PATH)
diagnose_upcycle_math = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = diagnose_upcycle_math
SPEC.loader.exec_module(diagnose_upcycle_math)


def test_relative_l2_error_is_zero_for_equal_tensors():
    tensor = torch.tensor([1.0, 2.0, 3.0])

    assert diagnose_upcycle_math.relative_l2_error(tensor, tensor.clone()) == 0.0


def test_expert_outputs_matches_dense_sliced_sum_for_tiny_swiglu():
    hidden = torch.tensor([[0.25, -0.5]])
    gate = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    up = torch.tensor([[0.2, -0.1], [0.4, -0.3], [0.6, -0.5], [0.8, -0.7]])
    down = torch.tensor([[0.2, 0.4, 0.6, 0.8], [-0.1, -0.3, -0.5, -0.7]])
    gate_up = torch.cat([gate.reshape(2, 2, 2), up.reshape(2, 2, 2)], dim=1)
    down_experts = down.reshape(2, 2, 2).permute(1, 0, 2).contiguous()

    dense = diagnose_upcycle_math.dense_ffn(hidden, gate, up, down)
    expert_sum = diagnose_upcycle_math.expert_outputs(hidden, gate_up, down_experts).sum(dim=0)

    torch.testing.assert_close(expert_sum, dense)
