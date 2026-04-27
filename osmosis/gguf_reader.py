"""GGUF reader for Osmosis block-wise quant types.

Reads GGUF v3 files written by gguf_writer.py, extracts osmosis tensors,
and unpacks them back to float using block-wise dequantization.
"""
import struct
from pathlib import Path

import numpy as np
import torch

from osmosis.gguf_writer import (
    GGUF_MAGIC, GGML_TYPE_F16, GGML_TYPE_F32,
    OSMOSIS_1BIT, OSMOSIS_2BIT, OSMOSIS_4BIT,
)
from osmosis.loader import _unpack_blocks, BLOCK_SIZE

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

OSMOSIS_BITS = {OSMOSIS_1BIT: 1, OSMOSIS_2BIT: 2, OSMOSIS_4BIT: 4}
OSMOSIS_BLOCK_BYTES = {1: 6, 2: 10, 4: 18}


def _read_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_kv_value(f):
    vtype = struct.unpack("<I", f.read(4))[0]
    if vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        if elem_type == GGUF_TYPE_UINT32:
            return [struct.unpack("<I", f.read(4))[0] for _ in range(count)]
        elif elem_type == GGUF_TYPE_FLOAT32:
            return [struct.unpack("<f", f.read(4))[0] for _ in range(count)]
        elif elem_type == GGUF_TYPE_STRING:
            return [_read_string(f) for _ in range(count)]
        else:
            raise ValueError(f"Unsupported array element type: {elem_type}")
    else:
        raise ValueError(f"Unsupported KV type: {vtype}")


def _align_offset(offset, alignment=32):
    return offset + (alignment - offset % alignment) % alignment


def read_osmosis_gguf(path: str) -> dict:
    """Read an Osmosis GGUF file and return metadata + tensors."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic: {hex(magic)})")

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        metadata = {}
        for _ in range(n_kv):
            key = _read_string(f)
            value = _read_kv_value(f)
            metadata[key] = value

        tensor_infos = []
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_infos.append({
                "name": name,
                "shape": dims,
                "type": dtype,
                "offset": offset,
            })

        data_start = _align_offset(f.tell(), 32)

        tensors = {}
        for info in tensor_infos:
            name = info["name"]
            abs_offset = _align_offset(data_start + info["offset"], 32)
            f.seek(abs_offset)

            if info["type"] == GGML_TYPE_F32:
                n_bytes = int(np.prod(info["shape"])) * 4
                raw = np.frombuffer(f.read(n_bytes), dtype=np.float32)
                tensors[name] = torch.tensor(raw.reshape(info["shape"]))

            elif info["type"] == GGML_TYPE_F16:
                n_bytes = int(np.prod(info["shape"])) * 2
                raw = np.frombuffer(f.read(n_bytes), dtype=np.float16)
                tensors[name] = torch.tensor(raw.reshape(info["shape"]))

            elif info["type"] in OSMOSIS_BITS:
                bits = OSMOSIS_BITS[info["type"]]
                safe_key = name.replace(".", "_")
                shape = metadata[f"osmosis.shape.{safe_key}"]
                num_elements = int(np.prod(shape))
                n_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
                block_bytes = OSMOSIS_BLOCK_BYTES[bits]
                n_bytes = n_blocks * block_bytes
                data = f.read(n_bytes)
                tensors[name] = _unpack_blocks(data, shape, bits)
            else:
                raise ValueError(
                    f"Unknown type {info['type']} for tensor {name}"
                )

    return {"metadata": metadata, "tensors": tensors}
