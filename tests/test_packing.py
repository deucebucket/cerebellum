"""Roundtrip tests: pack → unpack must reconstruct the quantized tensor."""
import torch
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from osmosis.pipeline import pack_1bit, pack_2bit, pack_4bit
from osmosis.loader import unpack_1bit, unpack_2bit, unpack_4bit


def test_1bit_roundtrip():
    torch.manual_seed(42)
    t = torch.randn(128, 256)
    packed_bytes, scale, shape = pack_1bit(t)
    recovered = unpack_1bit(packed_bytes, shape, scale)
    expected = torch.sign(t) * t.abs().mean()
    assert recovered.shape == t.shape
    assert torch.allclose(recovered, expected, atol=1e-5), \
        f"Max diff: {(recovered - expected).abs().max()}"


def test_2bit_roundtrip():
    torch.manual_seed(42)
    t = torch.randn(64, 128)
    packed_bytes, scale, shape = pack_2bit(t)
    recovered = unpack_2bit(packed_bytes, shape, scale)
    assert recovered.shape == t.shape
    s = t.abs().max() / 1.5
    normalized = (t / s).clamp(-1.5, 1.5)
    levels = torch.round((normalized + 1.5)).clamp(0, 3)
    expected = (levels - 1.5) * s
    assert torch.allclose(recovered, expected.float(), atol=1e-4), \
        f"Max diff: {(recovered - expected).abs().max()}"


def test_4bit_roundtrip():
    torch.manual_seed(42)
    t = torch.randn(32, 64)
    packed_bytes, scale, shape = pack_4bit(t)
    recovered = unpack_4bit(packed_bytes, shape, scale)
    assert recovered.shape == t.shape
    s = t.abs().max() / 7.5
    normalized = (t / s).clamp(-7.5, 7.5)
    levels = torch.round(normalized + 7.5).clamp(0, 15)
    expected = (levels - 7.5) * s
    assert torch.allclose(recovered, expected.float(), atol=1e-4), \
        f"Max diff: {(recovered - expected).abs().max()}"


def test_1bit_odd_shape():
    t = torch.randn(13, 7)
    packed_bytes, scale, shape = pack_1bit(t)
    recovered = unpack_1bit(packed_bytes, shape, scale)
    assert recovered.shape == t.shape


def test_osm_file_roundtrip(tmp_path):
    """Full file write → read roundtrip via .osm format."""
    import struct
    from osmosis.loader import load_osm_tensor

    torch.manual_seed(99)
    t = torch.randn(64, 32)
    packed_bytes, scale, shape = pack_2bit(t)

    osm_path = tmp_path / "test.osm"
    with open(osm_path, "wb") as f:
        f.write(struct.pack("<BfII", 2, scale, shape[0], shape[1]))
        f.write(packed_bytes)

    recovered = load_osm_tensor(osm_path)
    assert recovered.shape == (64, 32)

    repacked, scale2, shape2 = pack_2bit(t)
    expected = unpack_2bit(repacked, shape2, scale2)
    assert torch.allclose(recovered, expected, atol=1e-5)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
