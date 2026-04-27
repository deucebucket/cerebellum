"""Roundtrip tests: pack → unpack must reconstruct the quantized tensor."""
import struct

import torch
import numpy as np

from osmosis.pipeline import pack_1bit, pack_2bit, pack_4bit, _pack_blocks, BLOCK_SIZE
from osmosis.loader import _unpack_blocks, load_osm_tensor


def test_4bit_roundtrip():
    torch.manual_seed(42)
    t = torch.randn(32, 64)
    packed_bytes, _, shape = pack_4bit(t)
    recovered = _unpack_blocks(packed_bytes, shape, 4)
    assert recovered.shape == t.shape
    cos = torch.nn.functional.cosine_similarity(
        recovered.flatten().unsqueeze(0), t.flatten().unsqueeze(0)
    ).item()
    assert cos > 0.99, f"4-bit cosine too low: {cos}"


def test_2bit_roundtrip():
    torch.manual_seed(42)
    t = torch.randn(64, 128)
    packed_bytes, _, shape = pack_2bit(t)
    recovered = _unpack_blocks(packed_bytes, shape, 2)
    assert recovered.shape == t.shape
    cos = torch.nn.functional.cosine_similarity(
        recovered.flatten().unsqueeze(0), t.flatten().unsqueeze(0)
    ).item()
    assert cos > 0.85, f"2-bit cosine too low: {cos}"


def test_1bit_roundtrip():
    torch.manual_seed(42)
    t = torch.randn(128, 256)
    packed_bytes, _, shape = pack_1bit(t)
    recovered = _unpack_blocks(packed_bytes, shape, 1)
    assert recovered.shape == t.shape
    cos = torch.nn.functional.cosine_similarity(
        recovered.flatten().unsqueeze(0), t.flatten().unsqueeze(0)
    ).item()
    assert cos > 0.70, f"1-bit cosine too low: {cos}"


def test_odd_shape():
    t = torch.randn(13, 7)
    packed_bytes, _, shape = pack_1bit(t)
    recovered = _unpack_blocks(packed_bytes, shape, 1)
    assert recovered.shape == t.shape


def test_osm_v2_file_roundtrip(tmp_path):
    torch.manual_seed(99)
    t = torch.randn(64, 32)
    packed_bytes, _, shape = pack_4bit(t)

    osm_path = tmp_path / "test.osm"
    ndims = len(shape)
    with open(osm_path, "wb") as f:
        f.write(struct.pack("<BBB", 2, 4, ndims))
        for d in shape:
            f.write(struct.pack("<I", d))
        f.write(packed_bytes)

    recovered = load_osm_tensor(osm_path)
    assert recovered.shape == (64, 32)
    expected = _unpack_blocks(packed_bytes, shape, 4)
    assert torch.allclose(recovered, expected, atol=1e-5)


def test_block_size_consistency():
    """Each block should have its own scale."""
    t = torch.zeros(64)
    t[:32] = 1.0
    t[32:] = 100.0
    packed = _pack_blocks(t, 4)
    recovered = _unpack_blocks(packed, [64], 4)
    assert abs(recovered[0].item() - 1.0) < 0.2
    assert abs(recovered[32].item() - 100.0) < 15.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
