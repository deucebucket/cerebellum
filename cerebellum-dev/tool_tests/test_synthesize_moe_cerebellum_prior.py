import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "sparse-upcycling"
    / "scripts"
    / "synthesize_moe_cerebellum_prior.py"
)
SPEC = importlib.util.spec_from_file_location("synthesize_moe_cerebellum_prior", SCRIPT_PATH)
synthesize_moe_cerebellum_prior = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = synthesize_moe_cerebellum_prior
SPEC.loader.exec_module(synthesize_moe_cerebellum_prior)

build_prior = synthesize_moe_cerebellum_prior.build_prior
linear_attention_layers = synthesize_moe_cerebellum_prior.linear_attention_layers
read_moe_ablation_priors = synthesize_moe_cerebellum_prior.read_moe_ablation_priors


def test_linear_attention_layers_scale_with_target_depth():
    assert linear_attention_layers(8, 4, 3) == [0, 1, 2, 4, 5, 6]
    assert linear_attention_layers(4, 0, 3) == [0, 1, 2, 3]


def test_build_prior_uses_target_layer_count_and_floors():
    overrides, reasons, applied = build_prior(
        dense_prior={},
        analysis_rows=[],
        base_quant="Q2_K",
        layers=4,
    )

    assert overrides["blk.3.ffn_down_exps.weight"] == "Q3_K"
    assert "blk.4.ffn_down_exps.weight" not in overrides
    assert overrides["blk.0.attn_qkv.weight"] == "Q4_K"
    assert "blk.3.attn_qkv.weight" not in overrides
    assert applied == []
    assert "moe expert floor from prior MoE fragility" in reasons["blk.3.ffn_down_exps.weight"]


def test_native_qwen_moe_promotion_prior_maps_split_gate_up_to_fused_target(tmp_path):
    prior_path = tmp_path / "qwen_moe_ablation.json"
    prior_path.write_text(
        json.dumps(
            {
                "promotions": [
                    {"group": "ffn_down_exps", "target_type": "Q4_K"},
                    {"group": "ffn_gate_exps", "target_type": "Q5_K"},
                    {"group": "ffn_up_exps", "target_type": "Q4_K"},
                ],
                "demotions": [
                    {"group": "ffn_down_shexp", "target_type": "Q2_K"},
                ],
            }
        )
    )
    moe_priors = read_moe_ablation_priors([prior_path])

    overrides, reasons, applied = build_prior(
        dense_prior={},
        analysis_rows=[],
        base_quant="Q2_K",
        layers=2,
        moe_priors=moe_priors,
    )

    assert overrides["blk.0.ffn_down_exps.weight"] == "Q4_K"
    assert overrides["blk.1.ffn_gate_up_exps.weight"] == "Q5_K"
    assert overrides["blk.0.ffn_down_shexp.weight"] == "Q3_K"
    assert any(item["group"] == "ffn_down_shexp" and not item["applied"] for item in applied)
    assert "native qwen moe role prior ffn_gate_exps -> Q5_K" in reasons["blk.0.ffn_gate_up_exps.weight"]

