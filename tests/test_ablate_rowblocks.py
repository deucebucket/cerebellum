import importlib.util
import json
import sys
from pathlib import Path


def load_rowblocks_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ablate_rowblocks.py"
    spec = importlib.util.spec_from_file_location("ablate_rowblocks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_attn_qkv_rowblock_layout_is_guarded():
    module = load_rowblocks_module()

    reason = module.unsupported_rowblock_layout_reason("blk.4.attn_qkv.weight")

    assert reason is not None
    assert "unresolved Q/K/V physical storage ordering" in reason


def test_regular_rowblock_layout_is_allowed():
    module = load_rowblocks_module()

    assert module.unsupported_rowblock_layout_reason("blk.10.ffn_gate.weight") is None


def test_rowblock_safety_report_blocks_fused_qkv():
    module = load_rowblocks_module()

    report = module.rowblock_safety_report(
        "blk.4.attn_qkv",
        base_quant="Q4_K_M",
        tensor_base_quant="Q4_K_M",
        low_quant="Q2_K",
    )

    assert report["target_tensor"] == "blk.4.attn_qkv.weight"
    assert report["blocked"] is True
    assert report["rowblock_safe"] is False
    assert "unresolved Q/K/V physical storage ordering" in report["unsupported_layout_reason"]


def test_rowblock_safety_report_describes_special_output_tensor():
    module = load_rowblocks_module()

    report = module.rowblock_safety_report(
        "output.weight",
        base_quant="Q3_K_S",
        tensor_base_quant="Q3_K_S",
        low_quant="Q2_K",
    )

    assert report["blocked"] is False
    assert report["special_tensor"] is True
    assert report["special_overrides"][0]["llama_quantize_flag"] == "--output-tensor-type"
    assert report["special_overrides"][1]["effective_tensor_quant"] == "Q3_K"
    assert report["special_overrides"][1]["variant_normalized"] is True


def test_validate_only_exits_before_required_paths(monkeypatch, capsys):
    module = load_rowblocks_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["ablate_rowblocks.py", "--target-tensor", "output.weight", "--validate-only"],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("validate-only should exit explicitly")

    payload = json.loads(capsys.readouterr().out)
    assert payload["target_tensor"] == "output.weight"
    assert payload["rowblock_safe"] is True


def test_missing_run_paths_point_to_validate_only(monkeypatch, capsys):
    module = load_rowblocks_module()
    monkeypatch.setattr(sys, "argv", ["ablate_rowblocks.py", "--target-tensor", "blk.10.ffn_gate.weight"])

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("normal rowblock run should require paths")

    assert "use --validate-only" in capsys.readouterr().err
