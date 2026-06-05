import importlib.util
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
