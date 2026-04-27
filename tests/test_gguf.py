"""GGUF roundtrip: pack → .osm v2 → GGUF → read → verify."""
import json
import struct
import numpy as np
import torch
import pytest
from pathlib import Path

from osmosis.pipeline import pack_1bit, pack_2bit, pack_4bit, _pack_blocks, BLOCK_SIZE
from osmosis.loader import _unpack_blocks
from osmosis.gguf_writer import (
    OsmosisGGUFWriter, GGML_TYPE_F16, GGML_TYPE_F32,
    OSMOSIS_1BIT, OSMOSIS_2BIT, OSMOSIS_4BIT, osmosis_type_for_bits,
)
from osmosis.gguf_reader import read_osmosis_gguf


def _make_crush_dir(tmp_path):
    """Create a fake crush directory with v2 block-wise tensors."""
    torch.manual_seed(42)

    t1 = torch.randn(64, 128)
    t2 = torch.randn(32, 64)
    t4 = torch.randn(16, 32)
    t16 = torch.randn(8, 16).half()

    p1_bytes, _, sh1 = pack_1bit(t1)
    p2_bytes, _, sh2 = pack_2bit(t2)
    p4_bytes, _, sh4 = pack_4bit(t4)

    crush = tmp_path / "crushed"
    crush.mkdir()

    for name, packed, shape, bits in [
        ("model.layers.0.attn.q_proj.weight", p1_bytes, sh1, 1),
        ("model.layers.0.mlp.gate_proj.weight", p2_bytes, sh2, 2),
        ("model.layers.0.mlp.down_proj.weight", p4_bytes, sh4, 4),
    ]:
        fname = name.replace(".", "_") + ".osm"
        ndims = len(shape)
        with open(crush / fname, "wb") as f:
            f.write(struct.pack("<BBB", 2, bits, ndims))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(packed)

    from safetensors.torch import save_file
    st_name = "model_embed_tokens_weight.safetensors"
    save_file({"model.embed_tokens.weight": t16}, str(crush / st_name))

    manifest = {
        "model": "test-model",
        "format": "osmosis-v2",
        "average_bits": 2.3,
        "layers": {
            "model.layers.0.attn.q_proj.weight": {
                "bits": 1, "file": "model_layers_0_attn_q_proj_weight.osm",
                "format": 2, "shape": sh1,
            },
            "model.layers.0.mlp.gate_proj.weight": {
                "bits": 2, "file": "model_layers_0_mlp_gate_proj_weight.osm",
                "format": 2, "shape": sh2,
            },
            "model.layers.0.mlp.down_proj.weight": {
                "bits": 4, "file": "model_layers_0_mlp_down_proj_weight.osm",
                "format": 2, "shape": sh4,
            },
            "model.embed_tokens.weight": {
                "bits": 16, "file": st_name,
                "shape": [8, 16],
            },
        },
        "compression": {"original_bytes": 100000, "packed_bytes": 30000, "ratio": 3.3},
    }
    with open(crush / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return crush, {"1bit": t1, "2bit": t2, "4bit": t4, "f16": t16}


def test_gguf_write_read_roundtrip(tmp_path):
    """Write GGUF with block-wise osmosis types, read back, verify."""
    torch.manual_seed(42)

    writer = OsmosisGGUFWriter()
    writer.add_metadata_string("general.architecture", "test")
    writer.add_metadata_string("osmosis.format", "osmosis-v2")

    t4 = torch.randn(32, 64)
    p4_bytes, _, sh4 = pack_4bit(t4)
    packed4 = np.frombuffer(p4_bytes, dtype=np.uint8)
    writer.add_tensor("layer.0.gate_proj", packed4, OSMOSIS_4BIT, sh4)
    writer.add_metadata_uint32_array("osmosis.shape.layer_0_gate_proj", sh4)

    f32_tensor = np.random.randn(4, 8).astype(np.float32)
    writer.add_tensor("embed", f32_tensor, GGML_TYPE_F32, [4, 8])

    gguf_path = str(tmp_path / "test.gguf")
    writer.write(gguf_path)

    result = read_osmosis_gguf(gguf_path)
    tensors = result["tensors"]
    meta = result["metadata"]

    assert meta["general.architecture"] == "test"
    assert meta["osmosis.format"] == "osmosis-v2"

    expected = _unpack_blocks(p4_bytes, sh4, 4)
    assert torch.allclose(tensors["layer.0.gate_proj"], expected, atol=1e-5)
    assert torch.allclose(tensors["embed"], torch.tensor(f32_tensor), atol=1e-6)


def test_convert_crush_to_gguf(tmp_path):
    """Full pipeline: crush dir → GGUF → read → verify."""
    from osmosis.gguf_writer import convert_osmosis_to_gguf

    crush, originals = _make_crush_dir(tmp_path)
    gguf_path = str(tmp_path / "model.gguf")
    convert_osmosis_to_gguf(str(crush), gguf_path, architecture="test")

    result = read_osmosis_gguf(gguf_path)
    tensors = result["tensors"]

    assert "model.layers.0.attn.q_proj.weight" in tensors
    assert "model.layers.0.mlp.gate_proj.weight" in tensors
    assert "model.layers.0.mlp.down_proj.weight" in tensors
    assert "model.embed_tokens.weight" in tensors

    t4_recovered = tensors["model.layers.0.mlp.down_proj.weight"]
    assert t4_recovered.shape == (16, 32)
    expected_4bit = _unpack_blocks(pack_4bit(originals["4bit"])[0], [16, 32], 4)
    assert torch.allclose(t4_recovered, expected_4bit, atol=1e-5)

    t16_recovered = tensors["model.embed_tokens.weight"].float()
    expected_f16 = originals["f16"].float()
    assert torch.allclose(t16_recovered, expected_f16, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
