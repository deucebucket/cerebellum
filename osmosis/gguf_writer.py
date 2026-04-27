"""GGUF writer for Osmosis mixed-precision models.

Writes GGUF v3 files with custom quant types for osmosis 1/2/4-bit tensors.
Standard tensors (embeddings, norms) are stored as F16/F32.

Custom type IDs (must match ggml.h enum in osmosis llama.cpp fork):
  OSMOSIS_1BIT = 43
  OSMOSIS_2BIT = 44
  OSMOSIS_4BIT = 45

Data layout for osmosis tensors: 4-byte float32 scale prefix + packed data.
Scale also stored as GGUF metadata for the Python reader.
"""
import json
import struct
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3

# Standard GGML types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

# Osmosis custom types (must match ggml.h enum)
OSMOSIS_1BIT = 43
OSMOSIS_2BIT = 44
OSMOSIS_4BIT = 45

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


def _write_string(f, s: str):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_kv_string(f, key: str, value: str):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    _write_string(f, value)


def _write_kv_uint32(f, key: str, value: int):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", value))


def _write_kv_float32(f, key: str, value: float):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
    f.write(struct.pack("<f", value))


def _write_kv_uint32_array(f, key: str, values: list):
    _write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<Q", len(values)))
    for v in values:
        f.write(struct.pack("<I", v))


def _align(f, alignment=32):
    pos = f.tell()
    pad = (alignment - pos % alignment) % alignment
    if pad:
        f.write(b"\x00" * pad)


def osmosis_type_for_bits(bits: int) -> int:
    return {1: OSMOSIS_1BIT, 2: OSMOSIS_2BIT, 4: OSMOSIS_4BIT}[bits]


class OsmosisGGUFWriter:
    """Writes an Osmosis-compressed model to GGUF format.

    Supports both buffered mode (small models, for tests) and streaming
    mode (large models, one tensor at a time via callbacks).
    """

    def __init__(self):
        self.metadata = {}
        self.tensors = []

    def add_metadata_string(self, key: str, value: str):
        self.metadata[key] = ("string", value)

    def add_metadata_uint32(self, key: str, value: int):
        self.metadata[key] = ("uint32", value)

    def add_metadata_float32(self, key: str, value: float):
        self.metadata[key] = ("float32", value)

    def add_metadata_uint32_array(self, key: str, values: list):
        self.metadata[key] = ("uint32_array", values)

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int,
                   shape: list):
        self.tensors.append({
            "name": name,
            "data": data,
            "type": ggml_type,
            "shape": shape,
        })

    def add_tensor_info(self, name: str, nbytes: int, ggml_type: int,
                        shape: list):
        """Register tensor metadata without holding data in memory."""
        self.tensors.append({
            "name": name,
            "data": None,
            "nbytes": nbytes,
            "type": ggml_type,
            "shape": shape,
        })

    def write(self, path: str, data_callback=None):
        """Write GGUF file.

        If data_callback is provided, it's called for each tensor that has
        data=None, with (f, tensor_info) and must write the tensor data.
        """
        with open(path, "wb") as f:
            n_tensors = len(self.tensors)
            n_kv = len(self.metadata)

            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", n_tensors))
            f.write(struct.pack("<Q", n_kv))

            for key, (vtype, value) in self.metadata.items():
                if vtype == "string":
                    _write_kv_string(f, key, value)
                elif vtype == "uint32":
                    _write_kv_uint32(f, key, value)
                elif vtype == "float32":
                    _write_kv_float32(f, key, value)
                elif vtype == "uint32_array":
                    _write_kv_uint32_array(f, key, value)

            data_offset_positions = []
            for t in self.tensors:
                _write_string(f, t["name"])
                n_dims = len(t["shape"])
                f.write(struct.pack("<I", n_dims))
                for dim in t["shape"]:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", t["type"]))
                data_offset_positions.append(f.tell())
                f.write(struct.pack("<Q", 0))

            _align(f, 32)
            data_start = f.tell()

            data_offsets = []
            for t in self.tensors:
                _align(f, 32)
                data_offsets.append(f.tell() - data_start)
                if t["data"] is not None:
                    f.write(t["data"].tobytes())
                elif data_callback:
                    data_callback(f, t)
                else:
                    raise ValueError(f"No data for tensor {t['name']}")

            for i, pos in enumerate(data_offset_positions):
                f.seek(pos)
                f.write(struct.pack("<Q", data_offsets[i]))

        print(f"GGUF written: {path} ({Path(path).stat().st_size / 1e9:.2f} GB)")


def convert_osmosis_to_gguf(crush_dir: str, output_path: str,
                            architecture: str = "qwen3_5"):
    """Convert Osmosis crush directory to GGUF file (streaming, low memory)."""
    crush_path = Path(crush_dir)
    with open(crush_path / "manifest.json") as f:
        manifest = json.load(f)

    writer = OsmosisGGUFWriter()

    writer.add_metadata_string("general.architecture", architecture)
    writer.add_metadata_string("general.name",
                               manifest.get("model", "osmosis-model"))
    writer.add_metadata_string("osmosis.format", "osmosis-v1")
    writer.add_metadata_float32("osmosis.average_bits",
                                manifest.get("average_bits", 0.0))

    compression = manifest.get("compression", {})
    writer.add_metadata_float32("osmosis.compression_ratio",
                                compression.get("ratio", 0.0))

    layers = manifest["layers"]

    # Pass 1: collect tensor info (names, types, sizes) without loading data
    print("Pass 1: collecting tensor metadata...")
    tensor_sources = {}

    for key, info in layers.items():
        bits = info["bits"]
        file_name = info["file"]
        shape = info["shape"]

        if bits == 16:
            from safetensors import safe_open
            st_path = crush_path / file_name
            with safe_open(str(st_path), framework="pt", device="cpu") as sf:
                for tk in sf.keys():
                    sl = sf.get_slice(tk)
                    tensor_shape = sl.get_shape()
                    dtype_str = str(sl.get_dtype())
                    if "F16" in dtype_str or "BF16" in dtype_str:
                        ggml_type = GGML_TYPE_F16
                        nbytes = int(np.prod(tensor_shape)) * 2
                    else:
                        ggml_type = GGML_TYPE_F32
                        nbytes = int(np.prod(tensor_shape)) * 4

                    writer.add_tensor_info(tk, nbytes, ggml_type, tensor_shape)
                    tensor_sources[tk] = ("safetensors", str(st_path), tk)
        else:
            osm_path = crush_path / file_name
            fmt_version = info.get("format", 1)
            with open(osm_path, "rb") as of:
                if fmt_version == 2:
                    of.read(1)  # version byte
                    file_bits, ndims = struct.unpack("<BB", of.read(2))
                    of.read(ndims * 4)  # skip shape
                    header_size = 3 + ndims * 4
                else:
                    of.read(13)
                    file_bits = bits
                    header_size = 13
                data_size = of.seek(0, 2) - header_size

            ggml_type = osmosis_type_for_bits(file_bits)
            writer.add_tensor_info(key, data_size, ggml_type, shape)
            tensor_sources[key] = ("osm", str(osm_path), header_size)

            writer.add_metadata_uint32_array(
                f"osmosis.shape.{key.replace('.', '_')}", shape)

    print(f"  {len(tensor_sources)} tensors registered")

    # Pass 2: stream data to GGUF file, one tensor at a time
    print("Pass 2: streaming tensor data to GGUF...")
    written = [0]

    def stream_tensor(f, tensor_info):
        name = tensor_info["name"]
        source = tensor_sources[name]

        if source[0] == "safetensors":
            import torch
            _, st_path, tk = source
            with safe_open(st_path, framework="pt", device="cpu") as sf:
                pt_tensor = sf.get_tensor(tk)
                if pt_tensor.dtype == torch.bfloat16:
                    pt_tensor = pt_tensor.to(torch.float16)
                f.write(pt_tensor.numpy().tobytes())

        elif source[0] == "osm":
            _, osm_path, header_size = source
            with open(osm_path, "rb") as of:
                of.seek(header_size)
                while True:
                    chunk = of.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        written[0] += 1
        if written[0] % 100 == 0:
            print(f"  {written[0]}/{len(tensor_sources)} tensors written")

    writer.write(output_path, data_callback=stream_tensor)
    print(f"Done: {written[0]} tensors written")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert Osmosis to GGUF")
    parser.add_argument("--crush-dir", required=True,
                        help="Path to crush output directory")
    parser.add_argument("--output", required=True,
                        help="Output GGUF file path")
    parser.add_argument("--arch", default="qwen3_5",
                        help="Model architecture name")
    args = parser.parse_args()
    convert_osmosis_to_gguf(args.crush_dir, args.output, args.arch)


if __name__ == "__main__":
    main()
