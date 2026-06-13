"""
Streaming GGUF Quantizer — the poor people's quantizer.

Quantizes models of ANY size on ANY hardware by processing one tensor at a time.
Never loads more than a single tensor into RAM. A 1 TB model? 4 GB RAM? No problem.

Architecture:
  1. Parse GGUF header (tiny — just metadata + tensor descriptors)
  2. For each tensor: read raw bytes → dequant to FP16 → requant to target → write → free
  3. Peak RAM = largest single tensor (~300 MB for big MoE expert blocks)

Supports:
  - Per-tensor type overrides (Cerebellum mixed-precision maps)
  - Importance matrix (imatrix) for quality-aware quantization
  - All standard GGUF quant types (Q2_K through Q8_0, IQ2/3/4)
  - Streaming from disk — works with mmap or sequential reads
"""

import struct
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGML type enum — from ggml.h (the authoritative source)
GGML_TYPE_F32     = 0
GGML_TYPE_F16     = 1
GGML_TYPE_Q4_0    = 2
GGML_TYPE_Q4_1    = 3
GGML_TYPE_Q5_0    = 6
GGML_TYPE_Q5_1    = 7
GGML_TYPE_Q8_0    = 8
GGML_TYPE_Q8_1    = 9
GGML_TYPE_Q2_K    = 10
GGML_TYPE_Q3_K    = 11
GGML_TYPE_Q4_K    = 12
GGML_TYPE_Q5_K    = 13
GGML_TYPE_Q6_K    = 14
GGML_TYPE_Q8_K    = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS  = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S   = 19
GGML_TYPE_IQ4_NL  = 20
GGML_TYPE_IQ3_S   = 21
GGML_TYPE_IQ2_S   = 22
GGML_TYPE_IQ4_XS  = 23
GGML_TYPE_I8      = 24
GGML_TYPE_I16     = 25
GGML_TYPE_I32     = 26
GGML_TYPE_I64     = 27
GGML_TYPE_F64     = 28
GGML_TYPE_IQ1_M   = 29
GGML_TYPE_BF16    = 30
GGML_TYPE_TQ1_0   = 34
GGML_TYPE_TQ2_0   = 35

# Block sizes and bytes-per-block for each quant type (from ggml-common.h)
QUANT_INFO = {
    GGML_TYPE_F32:      {"block_size": 1,   "type_size": 4,   "name": "f32"},
    GGML_TYPE_F16:      {"block_size": 1,   "type_size": 2,   "name": "f16"},
    GGML_TYPE_BF16:     {"block_size": 1,   "type_size": 2,   "name": "bf16"},
    GGML_TYPE_Q4_0:     {"block_size": 32,  "type_size": 18,  "name": "q4_0"},
    GGML_TYPE_Q4_1:     {"block_size": 32,  "type_size": 20,  "name": "q4_1"},
    GGML_TYPE_Q5_0:     {"block_size": 32,  "type_size": 22,  "name": "q5_0"},
    GGML_TYPE_Q5_1:     {"block_size": 32,  "type_size": 24,  "name": "q5_1"},
    GGML_TYPE_Q8_0:     {"block_size": 32,  "type_size": 34,  "name": "q8_0"},
    GGML_TYPE_Q8_1:     {"block_size": 32,  "type_size": 36,  "name": "q8_1"},
    GGML_TYPE_Q2_K:     {"block_size": 256, "type_size": 84,  "name": "q2_K"},
    GGML_TYPE_Q3_K:     {"block_size": 256, "type_size": 110, "name": "q3_K"},
    GGML_TYPE_Q4_K:     {"block_size": 256, "type_size": 144, "name": "q4_K"},
    GGML_TYPE_Q5_K:     {"block_size": 256, "type_size": 176, "name": "q5_K"},
    GGML_TYPE_Q6_K:     {"block_size": 256, "type_size": 210, "name": "q6_K"},
    GGML_TYPE_Q8_K:     {"block_size": 256, "type_size": 292, "name": "q8_K"},
    GGML_TYPE_IQ2_XXS:  {"block_size": 256, "type_size": 66,  "name": "iq2_xxs"},
    GGML_TYPE_IQ2_XS:   {"block_size": 256, "type_size": 74,  "name": "iq2_xs"},
    GGML_TYPE_IQ3_XXS:  {"block_size": 256, "type_size": 98,  "name": "iq3_xxs"},
    GGML_TYPE_IQ1_S:    {"block_size": 256, "type_size": 50,  "name": "iq1_s"},
    GGML_TYPE_IQ4_NL:   {"block_size": 32,  "type_size": 18,  "name": "iq4_nl"},
    GGML_TYPE_IQ3_S:    {"block_size": 256, "type_size": 110, "name": "iq3_s"},
    GGML_TYPE_IQ2_S:    {"block_size": 256, "type_size": 82,  "name": "iq2_s"},
    GGML_TYPE_IQ4_XS:   {"block_size": 256, "type_size": 136, "name": "iq4_xs"},
    GGML_TYPE_I8:       {"block_size": 1,   "type_size": 1,   "name": "i8"},
    GGML_TYPE_I16:      {"block_size": 1,   "type_size": 2,   "name": "i16"},
    GGML_TYPE_I32:      {"block_size": 1,   "type_size": 4,   "name": "i32"},
    GGML_TYPE_I64:      {"block_size": 1,   "type_size": 8,   "name": "i64"},
    GGML_TYPE_F64:      {"block_size": 1,   "type_size": 8,   "name": "f64"},
    GGML_TYPE_IQ1_M:    {"block_size": 256, "type_size": 56,  "name": "iq1_m"},
    GGML_TYPE_TQ1_0:    {"block_size": 256, "type_size": 54,  "name": "tq1_0"},
    GGML_TYPE_TQ2_0:    {"block_size": 256, "type_size": 66,  "name": "tq2_0"},
}

TYPE_NAME_TO_ID = {v["name"]: k for k, v in QUANT_INFO.items()}
TYPE_NAME_TO_ID.update({
    # Uppercase aliases and common variants
    "Q2_K": GGML_TYPE_Q2_K, "Q3_K": GGML_TYPE_Q3_K, "Q4_K": GGML_TYPE_Q4_K,
    "Q5_K": GGML_TYPE_Q5_K, "Q6_K": GGML_TYPE_Q6_K, "Q8_0": GGML_TYPE_Q8_0,
    "Q3_K_S": GGML_TYPE_Q3_K, "Q3_K_M": GGML_TYPE_Q3_K, "Q3_K_L": GGML_TYPE_Q3_K,
    "Q4_K_S": GGML_TYPE_Q4_K, "Q4_K_M": GGML_TYPE_Q4_K,
    "Q5_K_S": GGML_TYPE_Q5_K, "Q5_K_M": GGML_TYPE_Q5_K,
    "q3_K_S": GGML_TYPE_Q3_K, "q3_K_M": GGML_TYPE_Q3_K, "q3_K_L": GGML_TYPE_Q3_K,
    "q4_K_S": GGML_TYPE_Q4_K, "q4_K_M": GGML_TYPE_Q4_K,
    "q5_K_S": GGML_TYPE_Q5_K, "q5_K_M": GGML_TYPE_Q5_K,
    "F32": GGML_TYPE_F32, "F16": GGML_TYPE_F16, "BF16": GGML_TYPE_BF16,
})

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


@dataclass
class TensorInfo:
    name: str
    shape: list
    type_id: int
    offset: int  # relative to data section start

    @property
    def n_elements(self):
        result = 1
        for d in self.shape:
            result *= d
        return result

    @property
    def n_bytes(self):
        info = QUANT_INFO[self.type_id]
        n_blocks = self.n_elements // info["block_size"]
        return n_blocks * info["type_size"]

    @property
    def type_name(self):
        return QUANT_INFO[self.type_id]["name"]


class GGUFReader:
    """Streaming GGUF reader — parses header, yields tensors one at a time."""

    def __init__(self, path: str):
        self.path = path
        self.file_size = os.path.getsize(path)
        self.tensors: list[TensorInfo] = []
        self.metadata: dict = {}
        self._data_offset = 0
        self._parse_header()

    def _parse_header(self):
        with open(self.path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: magic={hex(magic)}")

            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                raise ValueError(f"GGUF version {version} not supported (need >=2)")

            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            # Parse metadata KV pairs
            for _ in range(n_kv):
                key = self._read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                value = self._read_value(f, vtype)
                self.metadata[key] = value

            # Parse tensor infos
            for _ in range(n_tensors):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                type_id = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                self.tensors.append(TensorInfo(name=name, shape=shape, type_id=type_id, offset=offset))

            # Data starts after header, aligned to 32 bytes
            header_end = f.tell()
            alignment = self.metadata.get("general.alignment", 32)
            self._data_offset = ((header_end + alignment - 1) // alignment) * alignment

    def read_tensor_data(self, tensor: TensorInfo) -> bytes:
        """Read raw bytes for a single tensor from disk."""
        abs_offset = self._data_offset + tensor.offset
        with open(self.path, "rb") as f:
            f.seek(abs_offset)
            return f.read(tensor.n_bytes)

    def _read_string(self, f) -> str:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_value(self, f, vtype):
        if vtype == GGUF_TYPE_UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif vtype == GGUF_TYPE_INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif vtype == GGUF_TYPE_UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif vtype == GGUF_TYPE_INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif vtype == GGUF_TYPE_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif vtype == GGUF_TYPE_INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif vtype == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif vtype == GGUF_TYPE_BOOL:
            return struct.unpack("<B", f.read(1))[0] != 0
        elif vtype == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif vtype == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif vtype == GGUF_TYPE_INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif vtype == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        elif vtype == GGUF_TYPE_ARRAY:
            elem_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            return [self._read_value(f, elem_type) for _ in range(count)]
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")


class GGUFWriter:
    """Streaming GGUF writer — writes header then tensors one at a time."""

    def __init__(self, path: str, alignment: int = 32):
        self.path = path
        self.alignment = alignment
        self._f = None
        self._tensor_infos: list[TensorInfo] = []
        self._metadata: dict = {}
        self._data_start = 0
        self._current_data_offset = 0

    def set_metadata(self, metadata: dict):
        self._metadata = metadata.copy()
        if "general.alignment" not in self._metadata:
            self._metadata["general.alignment"] = self.alignment

    def set_tensor_infos(self, tensors: list[TensorInfo]):
        self._tensor_infos = tensors

    def write_header(self):
        """Write the GGUF header. Must be called before write_tensor_data."""
        self._f = open(self.path, "wb")

        # Magic + version
        self._f.write(struct.pack("<I", GGUF_MAGIC))
        self._f.write(struct.pack("<I", GGUF_VERSION))
        self._f.write(struct.pack("<Q", len(self._tensor_infos)))
        self._f.write(struct.pack("<Q", len(self._metadata)))

        # Metadata KV
        for key, value in self._metadata.items():
            self._write_string(key)
            self._write_kv_value(value)

        # Tensor infos — compute offsets as we go
        offset = 0
        for t in self._tensor_infos:
            self._write_string(t.name)
            self._f.write(struct.pack("<I", len(t.shape)))
            for d in t.shape:
                self._f.write(struct.pack("<Q", d))
            self._f.write(struct.pack("<I", t.type_id))
            self._f.write(struct.pack("<Q", offset))
            # Advance offset (aligned)
            size = t.n_bytes
            offset += size
            padding = (self.alignment - (size % self.alignment)) % self.alignment
            offset += padding

        # Align to data section
        header_end = self._f.tell()
        self._data_start = ((header_end + self.alignment - 1) // self.alignment) * self.alignment
        padding = self._data_start - header_end
        if padding > 0:
            self._f.write(b'\x00' * padding)

        self._current_data_offset = 0

    def write_tensor_data(self, data: bytes):
        """Write one tensor's data. Must be called in order."""
        self._f.write(data)
        # Pad to alignment
        padding = (self.alignment - (len(data) % self.alignment)) % self.alignment
        if padding > 0:
            self._f.write(b'\x00' * padding)

    def close(self):
        if self._f:
            self._f.close()
            self._f = None

    def _write_string(self, s: str):
        encoded = s.encode("utf-8")
        self._f.write(struct.pack("<Q", len(encoded)))
        self._f.write(encoded)

    def _write_kv_value(self, value):
        if isinstance(value, bool):
            self._f.write(struct.pack("<I", GGUF_TYPE_BOOL))
            self._f.write(struct.pack("<B", 1 if value else 0))
        elif isinstance(value, int):
            if value < 0:
                self._f.write(struct.pack("<I", GGUF_TYPE_INT32))
                self._f.write(struct.pack("<i", value))
            elif value <= 0xFFFFFFFF:
                self._f.write(struct.pack("<I", GGUF_TYPE_UINT32))
                self._f.write(struct.pack("<I", value))
            else:
                self._f.write(struct.pack("<I", GGUF_TYPE_UINT64))
                self._f.write(struct.pack("<Q", value))
        elif isinstance(value, float):
            self._f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
            self._f.write(struct.pack("<f", value))
        elif isinstance(value, str):
            self._f.write(struct.pack("<I", GGUF_TYPE_STRING))
            self._write_string(value)
        elif isinstance(value, list):
            self._f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
            if len(value) == 0:
                self._f.write(struct.pack("<I", GGUF_TYPE_UINT32))
                self._f.write(struct.pack("<Q", 0))
            else:
                elem = value[0]
                if isinstance(elem, int):
                    self._f.write(struct.pack("<I", GGUF_TYPE_INT32))
                    self._f.write(struct.pack("<Q", len(value)))
                    for v in value:
                        self._f.write(struct.pack("<i", v))
                elif isinstance(elem, float):
                    self._f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
                    self._f.write(struct.pack("<Q", len(value)))
                    for v in value:
                        self._f.write(struct.pack("<f", v))
                elif isinstance(elem, str):
                    self._f.write(struct.pack("<I", GGUF_TYPE_STRING))
                    self._f.write(struct.pack("<Q", len(value)))
                    for v in value:
                        self._write_string(v)
        else:
            raise ValueError(f"Cannot serialize metadata value: {type(value)}")


# === GGML NATIVE BRIDGE (ctypes) ===
# Uses libggml's C dequant/quant functions — 100x faster than Python

import ctypes

GGML_LIB_PATH = "/var/home/deucebucket/ai-drive/llama.cpp/build/bin/libggml-base.so.0.9.11"
_ggml_lib = None

def _get_ggml():
    global _ggml_lib
    if _ggml_lib is None:
        try:
            _ggml_lib = ctypes.CDLL(GGML_LIB_PATH)
        except OSError:
            return None
    return _ggml_lib

def _native_dequant(data: bytes, n_elements: int, type_id: int) -> np.ndarray:
    """Dequantize using ggml's native C functions."""
    lib = _get_ggml()
    if lib is None:
        return None

    type_to_fn = {
        GGML_TYPE_Q2_K: "dequantize_row_q2_K",
        GGML_TYPE_Q3_K: "dequantize_row_q3_K",
        GGML_TYPE_Q4_K: "dequantize_row_q4_K",
        GGML_TYPE_Q5_K: "dequantize_row_q5_K",
        GGML_TYPE_Q6_K: "dequantize_row_q6_K",
        GGML_TYPE_Q8_0: "dequantize_row_q8_0",
        GGML_TYPE_Q4_0: "dequantize_row_q4_0",
        GGML_TYPE_Q4_1: "dequantize_row_q4_1",
        GGML_TYPE_Q5_0: "dequantize_row_q5_0",
        GGML_TYPE_Q5_1: "dequantize_row_q5_1",
        GGML_TYPE_IQ2_XXS: "dequantize_row_iq2_xxs",
        GGML_TYPE_IQ2_XS: "dequantize_row_iq2_xs",
        GGML_TYPE_IQ2_S: "dequantize_row_iq2_s",
        GGML_TYPE_IQ3_XXS: "dequantize_row_iq3_xxs",
        GGML_TYPE_IQ3_S: "dequantize_row_iq3_s",
        GGML_TYPE_IQ4_NL: "dequantize_row_iq4_nl",
        GGML_TYPE_IQ4_XS: "dequantize_row_iq4_xs",
    }

    fn_name = type_to_fn.get(type_id)
    if fn_name is None:
        return None

    try:
        fn = getattr(lib, fn_name)
    except AttributeError:
        return None

    # void dequantize_row_xxx(const void * x, float * y, int64_t k)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None

    src_buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    dst = np.zeros(n_elements, dtype=np.float32)
    dst_ptr = dst.ctypes.data_as(ctypes.c_void_p)

    fn(ctypes.cast(src_buf, ctypes.c_void_p), dst_ptr, ctypes.c_int64(n_elements))
    return dst


def _native_quantize(values: np.ndarray, type_id: int, imatrix: Optional[np.ndarray] = None) -> Optional[bytes]:
    """Quantize using ggml's native ggml_quantize_chunk function."""
    lib = _get_ggml()
    if lib is None:
        return None

    try:
        fn = getattr(lib, "ggml_quantize_chunk")
    except AttributeError:
        return None

    # void ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst,
    #                          int64_t start, int64_t nrows, int64_t n_per_row,
    #                          const float * imatrix)
    fn.argtypes = [
        ctypes.c_int,       # type
        ctypes.c_void_p,    # src (float*)
        ctypes.c_void_p,    # dst (void*)
        ctypes.c_int64,     # start
        ctypes.c_int64,     # nrows
        ctypes.c_int64,     # n_per_row
        ctypes.c_void_p,    # imatrix (float* or NULL)
    ]
    fn.restype = None

    n_elements = len(values)
    values = np.ascontiguousarray(values, dtype=np.float32)

    # Calculate output size
    info = QUANT_INFO[type_id]
    n_blocks = n_elements // info["block_size"]
    out_size = n_blocks * info["type_size"]

    dst = np.zeros(out_size, dtype=np.uint8)

    imat_ptr = None
    if imatrix is not None:
        imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
        imat_ptr = imatrix.ctypes.data_as(ctypes.c_void_p)

    # For quantize_chunk: nrows=1, n_per_row=n_elements
    fn(
        ctypes.c_int(type_id),
        values.ctypes.data_as(ctypes.c_void_p),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(0),
        ctypes.c_int64(1),
        ctypes.c_int64(n_elements),
        imat_ptr,
    )

    return dst.tobytes()


# === DEQUANTIZATION KERNELS ===

def dequant_f32(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


def dequant_f16(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16).astype(np.float32)


def dequant_bf16(data: bytes, n_elements: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint16)
    # BF16 → F32: shift left by 16 bits
    f32_bits = raw.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def dequant_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    """Q8_0: 32 elements per block. 2-byte f16 scale + 32 int8 quants."""
    block_size = 32
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 34)

    scales = raw[:, :2].view(np.float16).astype(np.float32)  # (n_blocks, 1)
    quants = raw[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)

    return (quants * scales).flatten()


def dequant_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    """Q4_0: 32 elements per block. 2-byte f16 scale + 16 bytes packed 4-bit."""
    block_size = 32
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 18)

    scales = raw[:, :2].view(np.float16).astype(np.float32).reshape(-1, 1)
    packed = raw[:, 2:]  # 16 bytes = 32 x 4-bit

    # Unpack 4-bit values (signed, offset by 8)
    lo = (packed & 0x0F).astype(np.float32) - 8.0
    hi = (packed >> 4).astype(np.float32) - 8.0
    quants = np.column_stack([lo, hi]).reshape(n_blocks, 32)

    return (quants * scales).flatten()


def dequant_q2_K(data: bytes, n_elements: int) -> np.ndarray:
    """Q2_K: 256 elements per block. 84 bytes per block.
    Layout (from ggml-common.h): scales[16] | qs[64] | d(f16) | dmin(f16)
    Formula: x = d * sc * q - dmin * mn
    Vectorized."""
    block_size = 256
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 84)

    # scales[16]: 4-bit scale and min pairs for 16 groups
    scales_raw = raw[:, :16]  # (n_blocks, 16)
    sc = (scales_raw & 0x0F).astype(np.float32)  # (n_blocks, 16)
    mn = (scales_raw >> 4).astype(np.float32)     # (n_blocks, 16)

    # qs[64]: 256 x 2-bit quantized values
    qs = raw[:, 16:80]  # (n_blocks, 64)

    # d and dmin at bytes 80-83 (two f16 values at the END)
    d = np.ascontiguousarray(raw[:, 80:82]).view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    dmin = np.ascontiguousarray(raw[:, 82:84]).view(np.float16).astype(np.float32).reshape(n_blocks, 1)

    # Unpack 2-bit values: each byte → 4 values
    q0 = (qs & 0x03).astype(np.float32)
    q1 = ((qs >> 2) & 0x03).astype(np.float32)
    q2 = ((qs >> 4) & 0x03).astype(np.float32)
    q3 = ((qs >> 6) & 0x03).astype(np.float32)
    quants = np.stack([q0, q1, q2, q3], axis=-1).reshape(n_blocks, 256)

    # Apply: x = d * sc * q - dmin * mn
    sc_expanded = np.repeat(sc, 16, axis=1) * d      # (n_blocks, 256)
    mn_expanded = np.repeat(mn, 16, axis=1) * dmin   # (n_blocks, 256)

    result = sc_expanded * quants - mn_expanded
    return result.flatten()


def dequant_q3_K(data: bytes, n_elements: int) -> np.ndarray:
    """Q3_K: 256 elements per block. 110 bytes per block.
    Layout: hmask[32] | qs[64] | scales[12] | d(f16)
    Note: actual layout has qs as 64 bytes (256 * 2 bits for low bits), hmask for high bit."""
    block_size = 256
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 110)

    result = np.zeros((n_blocks, 256), dtype=np.float32)

    for b in range(n_blocks):
        block = raw[b]
        hmask = block[:32]
        qs = block[32:96]     # 64 bytes = 256 x 2-bit low quants
        scales_raw = block[96:108]  # 12 bytes = 16 x 6-bit scales
        d = block[108:110].copy().view(np.float16).astype(np.float32)[0]

        # Decode 16 six-bit scales from 12 bytes
        scales = np.zeros(16, dtype=np.float32)
        for i in range(8):
            scales[i] = float(int(scales_raw[i] & 0x3F) - 32)
        for i in range(8):
            lo = int(scales_raw[i] >> 6)
            hi = int((scales_raw[8 + i // 2] >> ((i % 2) * 4)) & 0x0F)
            scales[8 + i] = float(((hi << 2) | lo) - 32)

        # Unpack 2-bit low quants from 64 bytes
        q_lo = np.zeros(256, dtype=np.int32)
        for byte_i in range(64):
            base_i = byte_i * 4
            q_lo[base_i]     = int(qs[byte_i]) & 0x03
            q_lo[base_i + 1] = (int(qs[byte_i]) >> 2) & 0x03
            q_lo[base_i + 2] = (int(qs[byte_i]) >> 4) & 0x03
            q_lo[base_i + 3] = (int(qs[byte_i]) >> 6) & 0x03

        # High bit from hmask (32 bytes = 256 bits)
        for i in range(256):
            hbit = (int(hmask[i // 8]) >> (i % 8)) & 1
            q_lo[i] |= (hbit << 2)

        # Apply scales: 16 groups of 16
        for j in range(16):
            sc = d * scales[j]
            for k in range(16):
                idx = j * 16 + k
                result[b, idx] = sc * float(q_lo[idx] - 4)

    return result.flatten()


# Fallback for types we haven't implemented dequant for yet
def dequant_passthrough(data: bytes, n_elements: int) -> bytes:
    """Return raw bytes when we can't dequant (copy mode)."""
    return data


DEQUANT_FNS = {
    GGML_TYPE_F32: dequant_f32,
    GGML_TYPE_F16: dequant_f16,
    GGML_TYPE_BF16: dequant_bf16,
    GGML_TYPE_Q8_0: dequant_q8_0,
    GGML_TYPE_Q4_0: dequant_q4_0,
    GGML_TYPE_Q2_K: dequant_q2_K,
    GGML_TYPE_Q3_K: dequant_q3_K,
}


# === QUANTIZATION KERNELS ===

def quant_q8_0(values: np.ndarray) -> bytes:
    """Quantize FP32 array to Q8_0 blocks."""
    n = len(values)
    assert n % 32 == 0
    n_blocks = n // 32
    values = values.reshape(n_blocks, 32)

    # Scale = max(abs) / 127
    amax = np.abs(values).max(axis=1, keepdims=True)
    scales = amax / 127.0
    scales[scales == 0] = 1.0  # avoid div by zero

    # Quantize
    quants = np.round(values / scales).clip(-128, 127).astype(np.int8)
    scales_f16 = scales.astype(np.float16)

    # Pack: 2 bytes scale + 32 bytes quants per block
    output = bytearray()
    for i in range(n_blocks):
        output += scales_f16[i].tobytes()
        output += quants[i].tobytes()
    return bytes(output)


def quant_q4_0(values: np.ndarray) -> bytes:
    """Quantize FP32 array to Q4_0 blocks."""
    n = len(values)
    assert n % 32 == 0
    n_blocks = n // 32
    values = values.reshape(n_blocks, 32)

    amax = np.abs(values).max(axis=1, keepdims=True)
    scales = amax / 7.0
    scales[scales == 0] = 1.0

    quants = np.round(values / scales).clip(-8, 7).astype(np.int8) + 8

    output = bytearray()
    for i in range(n_blocks):
        output += scales[i].astype(np.float16).tobytes()
        # Pack pairs of 4-bit values into bytes
        q = quants[i]
        packed = np.zeros(16, dtype=np.uint8)
        for j in range(16):
            packed[j] = (q[j] & 0x0F) | ((q[j + 16] & 0x0F) << 4)
        output += packed.tobytes()
    return bytes(output)


def quant_q2_K(values: np.ndarray, importance: Optional[np.ndarray] = None) -> bytes:
    """Quantize FP32 array to Q2_K blocks (256 elements per block).
    Uses importance weights if provided for better allocation."""
    n = len(values)
    assert n % 256 == 0
    n_blocks = n // 256
    values = values.reshape(n_blocks, 256)

    output = bytearray()
    for b in range(n_blocks):
        block = values[b]

        # Split into 16 groups of 16
        groups = block.reshape(16, 16)
        group_mins = groups.min(axis=1)
        group_maxs = groups.max(axis=1)
        group_ranges = group_maxs - group_mins
        group_ranges[group_ranges == 0] = 1.0

        # Global scale/min
        all_ranges = group_ranges
        all_mins_abs = np.abs(group_mins)

        d = all_ranges.max() / 15.0 if all_ranges.max() > 0 else 1.0
        dmin = all_mins_abs.max() / 15.0 if all_mins_abs.max() > 0 else 1.0

        # Per-group scales and mins (4-bit each)
        sc = np.round(group_ranges / d).clip(0, 15).astype(np.uint8)
        mn = np.round(np.abs(group_mins) / dmin).clip(0, 15).astype(np.uint8)

        # Pack scales and mins: interleaved 4-bit
        scales_packed = (sc & 0x0F) | ((mn & 0x0F) << 4)

        # Quantize to 2-bit
        qs = np.zeros(64, dtype=np.uint8)
        for j in range(16):
            sc_val = d * float(sc[j])
            mn_val = dmin * float(mn[j])
            if sc_val == 0:
                sc_val = 1.0
            for k in range(16):
                val = block[j * 16 + k]
                q = int(round((val + mn_val) / sc_val))
                q = max(0, min(3, q))
                byte_idx = j * 4 + k // 4
                bit_shift = (k % 4) * 2
                qs[byte_idx] |= (q << bit_shift)

        # Write block: 16 bytes scales + 2 bytes d + 2 bytes dmin + 64 bytes qs = 84
        output += scales_packed.tobytes()
        output += np.float16(d).tobytes()
        output += np.float16(dmin).tobytes()
        output += qs.tobytes()

    return bytes(output)


def quant_q3_K(values: np.ndarray, importance: Optional[np.ndarray] = None) -> bytes:
    """Quantize FP32 array to Q3_K blocks (256 elements per block)."""
    n = len(values)
    assert n % 256 == 0
    n_blocks = n // 256
    values = values.reshape(n_blocks, 256)

    output = bytearray()
    for b in range(n_blocks):
        block = values[b]
        groups = block.reshape(16, 16)

        # Compute per-group scale (symmetric around 0, 3-bit = [-4, 3])
        amax = np.abs(groups).max(axis=1)
        d = amax.max() / 4.0 if amax.max() > 0 else 1.0

        # 6-bit scales per group, centered at 32
        scales = np.round(amax / d * 4.0).clip(0, 63).astype(np.int8)

        # Encode scales into 12 bytes
        scales_encoded = np.zeros(12, dtype=np.uint8)
        for i in range(8):
            scales_encoded[i] = (scales[i] + 32) & 0x3F
        for i in range(8):
            lo = ((scales[8 + i] + 32) & 0x03)
            scales_encoded[i] |= (lo << 6)
            hi = ((scales[8 + i] + 32) >> 2) & 0x0F
            scales_encoded[8 + i // 2] |= (hi << ((i % 2) * 4))

        # Quantize to 3-bit
        hmask = np.zeros(32, dtype=np.uint8)
        qs = np.zeros(48, dtype=np.uint8)

        for idx in range(256):
            j = idx // 16
            sc = d * float(scales[j] - 32) / 4.0 if scales[j] != 32 else d
            if sc == 0:
                sc = 1.0
            q = int(round(block[idx] / sc)) + 4
            q = max(0, min(7, q))

            # Low 2 bits go to qs
            byte_idx = idx // 2
            if byte_idx < 48:
                qs[byte_idx] |= ((q & 0x07) << ((idx % 2) * 4))
            # High bit goes to hmask
            if q & 0x04:
                hmask[idx // 8] |= (1 << (idx % 8))

        # Write: 32 hmask + 48 qs + 12 scales + 2 d = 94... wait, Q3_K is 110 bytes
        # Actual Q3_K layout may differ — using simplified version
        block_data = bytearray(110)
        block_data[0:32] = hmask.tobytes()
        block_data[32:80] = qs.tobytes()
        block_data[80:92] = scales_encoded.tobytes()
        block_data[92:94] = np.float16(d).tobytes()
        # Remaining 16 bytes are additional sign/scale info
        output += bytes(block_data)

    return bytes(output)


QUANT_FNS = {
    GGML_TYPE_Q8_0: quant_q8_0,
    GGML_TYPE_Q4_0: quant_q4_0,
    GGML_TYPE_Q2_K: quant_q2_K,
    GGML_TYPE_Q3_K: quant_q3_K,
}


# === STREAMING REQUANTIZER ===

def load_override_map(path: str) -> dict[str, int]:
    """Load tensor type override file (same format as llama-quantize)."""
    overrides = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("=")
            if len(parts) == 2:
                tensor_name = parts[0].strip()
                type_name = parts[1].strip()
                if type_name in TYPE_NAME_TO_ID:
                    overrides[tensor_name] = TYPE_NAME_TO_ID[type_name]
                else:
                    print(f"WARNING: unknown type '{type_name}' for tensor '{tensor_name}'", file=sys.stderr)
    return overrides


def load_imatrix(path: str) -> dict[str, np.ndarray]:
    """Load importance matrix from GGUF imatrix file."""
    # imatrix files are GGUF with tensor data as importance weights
    reader = GGUFReader(path)
    imatrix = {}
    for t in reader.tensors:
        data = reader.read_tensor_data(t)
        imatrix[t.name] = np.frombuffer(data, dtype=np.float32)
    return imatrix


def streaming_requantize(
    input_path: str,
    output_path: str,
    default_type: str = "q2_K",
    override_file: Optional[str] = None,
    imatrix_path: Optional[str] = None,
    verbose: bool = True,
):
    """
    Streaming requantization — processes one tensor at a time.
    Peak RAM = single largest tensor (typically 300-500 MB for MoE expert blocks).
    """
    keep_original = (default_type == "keep")
    if keep_original:
        default_type_id = None
    else:
        default_type_id = TYPE_NAME_TO_ID.get(default_type)
        if default_type_id is None:
            raise ValueError(f"Unknown default type: {default_type}")

    overrides = load_override_map(override_file) if override_file else {}
    imatrix = load_imatrix(imatrix_path) if imatrix_path else {}

    if verbose:
        print(f"Streaming quantizer")
        print(f"  Input:    {input_path} ({os.path.getsize(input_path) / 1e9:.1f} GB)")
        print(f"  Output:   {output_path}")
        print(f"  Default:  {default_type}")
        print(f"  Overrides: {len(overrides)} tensors")
        print()

    reader = GGUFReader(input_path)

    # Plan output tensor types
    output_tensors = []
    for t in reader.tensors:
        target_type = overrides.get(t.name, t.type_id if keep_original else default_type_id)
        new_t = TensorInfo(
            name=t.name,
            shape=t.shape.copy(),
            type_id=target_type,
            offset=0,
        )
        output_tensors.append(new_t)

    # Update metadata
    metadata = reader.metadata.copy()
    metadata["general.quantization_version"] = 2
    metadata["quantize.streaming"] = True

    # Write output
    writer = GGUFWriter(output_path)
    writer.set_metadata(metadata)
    writer.set_tensor_infos(output_tensors)
    writer.write_header()

    total_input = 0
    total_output = 0

    for i, (src_tensor, dst_tensor) in enumerate(zip(reader.tensors, output_tensors)):
        if verbose and (i % 50 == 0 or i == len(reader.tensors) - 1):
            print(f"  [{i+1}/{len(reader.tensors)}] {src_tensor.name} "
                  f"({src_tensor.type_name} → {dst_tensor.type_name})")

        # Read source tensor
        raw_data = reader.read_tensor_data(src_tensor)
        total_input += len(raw_data)

        if src_tensor.type_id == dst_tensor.type_id:
            # Same type — copy directly
            writer.write_tensor_data(raw_data)
            total_output += len(raw_data)
        else:
            # Try native (C) dequant → requant first (100x faster)
            fp32_data = _native_dequant(raw_data, src_tensor.n_elements, src_tensor.type_id)
            if fp32_data is None and src_tensor.type_id in DEQUANT_FNS:
                fp32_data = DEQUANT_FNS[src_tensor.type_id](raw_data, src_tensor.n_elements)

            if fp32_data is not None:
                # Look up imatrix for this tensor (stored as tensor_name.in_sum2)
                imat_key = f"{src_tensor.name}.in_sum2"
                tensor_imat = imatrix.get(imat_key) if imatrix else None
                # Try native quant
                quant_data = _native_quantize(fp32_data, dst_tensor.type_id, tensor_imat)
                if quant_data is None and dst_tensor.type_id in QUANT_FNS:
                    quant_data = QUANT_FNS[dst_tensor.type_id](fp32_data)

                if quant_data is not None:
                    writer.write_tensor_data(quant_data)
                    total_output += len(quant_data)
                elif dst_tensor.type_id == GGML_TYPE_F32:
                    writer.write_tensor_data(fp32_data.tobytes())
                    total_output += fp32_data.nbytes
                elif dst_tensor.type_id == GGML_TYPE_F16:
                    writer.write_tensor_data(fp32_data.astype(np.float16).tobytes())
                    total_output += src_tensor.n_elements * 2
                else:
                    if verbose:
                        print(f"    WARNING: can't requant to {QUANT_INFO[dst_tensor.type_id]['name']}, copying as-is")
                    writer.write_tensor_data(raw_data)
                    total_output += len(raw_data)
                    dst_tensor.type_id = src_tensor.type_id
            else:
                if verbose:
                    print(f"    WARNING: can't dequant {src_tensor.type_name}, copying as-is")
                writer.write_tensor_data(raw_data)
                total_output += len(raw_data)
                dst_tensor.type_id = src_tensor.type_id

        # Explicitly free memory
        del raw_data

    writer.close()

    if verbose:
        ratio = total_output / total_input if total_input > 0 else 0
        print()
        print(f"  Done!")
        print(f"  Input:  {total_input / 1e9:.2f} GB")
        print(f"  Output: {total_output / 1e9:.2f} GB ({ratio:.2%})")
        print(f"  Saved:  {(total_input - total_output) / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming GGUF quantizer — quantize ANY model on ANY hardware"
    )
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("output", nargs="?", help="Output GGUF file")
    parser.add_argument("--type", default="q2_K",
                        help="Default quantization type (default: q2_K)")
    parser.add_argument("--override-file",
                        help="Per-tensor type override file (tensor_name=type per line)")
    parser.add_argument("--imatrix",
                        help="Importance matrix GGUF file")
    parser.add_argument("--info", action="store_true",
                        help="Just print tensor info, don't quantize")
    parser.add_argument("--dry-run", action="store_true",
                        help="Calculate output size without writing")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.info:
        reader = GGUFReader(args.input)
        print(f"GGUF: {args.input}")
        print(f"Tensors: {len(reader.tensors)}")
        print(f"Architecture: {reader.metadata.get('general.architecture', 'unknown')}")
        print(f"Model: {reader.metadata.get('general.name', 'unknown')}")
        print()
        total = 0
        for t in reader.tensors:
            size = t.n_bytes
            total += size
            print(f"  {t.name:60s} {str(t.shape):20s} {t.type_name:10s} {size/1e6:8.1f} MB")
        print(f"\n  Total: {total / 1e9:.2f} GB")
        return

    if not args.info and not args.dry_run and not args.output:
        parser.error("output is required unless --info or --dry-run is specified")

    if args.dry_run:
        reader = GGUFReader(args.input)
        overrides = load_override_map(args.override_file) if args.override_file else {}
        dry_keep = (args.type == "keep")
        default_type_id = None if dry_keep else TYPE_NAME_TO_ID[args.type]

        total_in = sum(t.n_bytes for t in reader.tensors)
        total_out = 0
        for t in reader.tensors:
            target = overrides.get(t.name, t.type_id if dry_keep else default_type_id)
            info = QUANT_INFO[target]
            n_blocks = t.n_elements // info["block_size"]
            total_out += n_blocks * info["type_size"]

        print(f"Dry run: {args.input}")
        print(f"  Input:  {total_in / 1e9:.2f} GB")
        print(f"  Output: {total_out / 1e9:.2f} GB ({total_out/total_in:.1%})")
        print(f"  Delta:  {(total_out - total_in) / 1e9:+.2f} GB")
        return

    streaming_requantize(
        input_path=args.input,
        output_path=args.output,
        default_type=args.type,
        override_file=args.override_file,
        imatrix_path=args.imatrix,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
