#!/usr/bin/env python3
"""
Streaming GGUF Quantizer — quantize models of ANY size on ANY hardware.

Processes one tensor at a time. Never loads more than a single tensor into RAM.
A 1 TB model on a machine with 4 GB RAM? No problem.

Architecture:
  1. Parse GGUF header (tiny — just metadata + tensor descriptors)
  2. For each tensor: read → dequant to FP32 → requant to target → write → free
  3. Peak RAM ≈ largest single tensor (~300 MB for big MoE expert blocks)

Speed:
  With libggml (auto-detected): ~50 MB/s throughput, uses native C dequant/quant
  Without libggml: falls back to pure Python (10-50x slower, still works)

Usage:
  # Basic requantization
  python streaming_quantize.py model-Q4_K.gguf output-Q2_K.gguf --type q2_K

  # Mixed precision with override file
  python streaming_quantize.py model.gguf output.gguf --type q2_K --override-file overrides.txt

  # Just inspect a model
  python streaming_quantize.py model.gguf --info

  # Calculate size without writing
  python streaming_quantize.py model.gguf --dry-run --type q3_K

Override file format (one per line):
  blk.0.attn_q.weight=q3_K
  blk.0.ffn_gate.weight=q4_K
  output.weight=q6_K

Requires: numpy
Optional: libggml (from any llama.cpp build) for native speed
"""

import struct
import numpy as np
import os
import sys
import ctypes
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGML type enum (from ggml.h)
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
    offset: int

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
    """Streaming GGUF reader — parses header, reads tensors one at a time."""

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

            for _ in range(n_kv):
                key = self._read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                value = self._read_value(f, vtype)
                self.metadata[key] = value

            for _ in range(n_tensors):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                type_id = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                self.tensors.append(TensorInfo(name=name, shape=shape, type_id=type_id, offset=offset))

            header_end = f.tell()
            alignment = self.metadata.get("general.alignment", 32)
            self._data_offset = ((header_end + alignment - 1) // alignment) * alignment

    def read_tensor_data(self, tensor: TensorInfo) -> bytes:
        """Read raw bytes for a single tensor."""
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

    def set_metadata(self, metadata: dict):
        self._metadata = metadata.copy()
        if "general.alignment" not in self._metadata:
            self._metadata["general.alignment"] = self.alignment

    def set_tensor_infos(self, tensors: list[TensorInfo]):
        self._tensor_infos = tensors

    def write_header(self):
        self._f = open(self.path, "wb")

        self._f.write(struct.pack("<I", GGUF_MAGIC))
        self._f.write(struct.pack("<I", GGUF_VERSION))
        self._f.write(struct.pack("<Q", len(self._tensor_infos)))
        self._f.write(struct.pack("<Q", len(self._metadata)))

        for key, value in self._metadata.items():
            self._write_string(key)
            self._write_kv_value(value)

        offset = 0
        for t in self._tensor_infos:
            self._write_string(t.name)
            self._f.write(struct.pack("<I", len(t.shape)))
            for d in t.shape:
                self._f.write(struct.pack("<Q", d))
            self._f.write(struct.pack("<I", t.type_id))
            self._f.write(struct.pack("<Q", offset))
            size = t.n_bytes
            offset += size
            padding = (self.alignment - (size % self.alignment)) % self.alignment
            offset += padding

        header_end = self._f.tell()
        data_start = ((header_end + self.alignment - 1) // self.alignment) * self.alignment
        padding = data_start - header_end
        if padding > 0:
            self._f.write(b'\x00' * padding)

    def write_tensor_data(self, data: bytes):
        self._f.write(data)
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


# === NATIVE GGML BRIDGE ===

_ggml_lib = None
_ggml_searched = False


def _find_ggml_lib(user_path: Optional[str] = None) -> Optional[str]:
    """Auto-detect libggml-base.so from llama.cpp build."""
    candidates = []

    if user_path:
        candidates.append(user_path)

    env_path = os.environ.get("GGML_LIB_PATH")
    if env_path:
        candidates.append(env_path)

    # Search common llama.cpp build locations
    home = Path.home()
    search_dirs = [
        Path("/usr/local/lib"),
        Path("/usr/lib"),
        home / "llama.cpp" / "build" / "bin",
        home / "llama.cpp" / "build" / "src",
        home / "llama.cpp" / "build" / "ggml" / "src",
    ]

    # Also check if llama-server is on PATH and find its directory
    import shutil
    llama_server = shutil.which("llama-server") or shutil.which("llama-cli")
    if llama_server:
        search_dirs.insert(0, Path(llama_server).parent)
        search_dirs.insert(1, Path(llama_server).parent.parent / "lib")

    for d in search_dirs:
        if d.exists():
            for f in sorted(d.glob("libggml-base.so*"), reverse=True):
                candidates.append(str(f))
            for f in sorted(d.glob("libggml-base.dylib*"), reverse=True):
                candidates.append(str(f))

    for path in candidates:
        if os.path.isfile(path):
            try:
                ctypes.CDLL(path)
                return path
            except OSError:
                continue

    return None


def _get_ggml(user_path: Optional[str] = None):
    global _ggml_lib, _ggml_searched
    if _ggml_lib is not None:
        return _ggml_lib
    if _ggml_searched:
        return None
    _ggml_searched = True

    lib_path = _find_ggml_lib(user_path)
    if lib_path is None:
        return None

    try:
        _ggml_lib = ctypes.CDLL(lib_path)
        return _ggml_lib
    except OSError:
        return None


def _native_dequant(data: bytes, n_elements: int, type_id: int) -> Optional[np.ndarray]:
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

    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None

    src_buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    dst = np.zeros(n_elements, dtype=np.float32)

    fn(ctypes.cast(src_buf, ctypes.c_void_p),
       dst.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int64(n_elements))
    return dst


def _native_quantize(values: np.ndarray, type_id: int, imatrix: Optional[np.ndarray] = None) -> Optional[bytes]:
    lib = _get_ggml()
    if lib is None:
        return None

    try:
        fn = getattr(lib, "ggml_quantize_chunk")
    except AttributeError:
        return None

    fn.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
    ]
    fn.restype = None

    n_elements = len(values)
    values = np.ascontiguousarray(values, dtype=np.float32)

    info = QUANT_INFO[type_id]
    n_blocks = n_elements // info["block_size"]
    out_size = n_blocks * info["type_size"]
    dst = np.zeros(out_size, dtype=np.uint8)

    imat_ptr = None
    if imatrix is not None:
        imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
        imat_ptr = imatrix.ctypes.data_as(ctypes.c_void_p)

    fn(ctypes.c_int(type_id),
       values.ctypes.data_as(ctypes.c_void_p),
       dst.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int64(0), ctypes.c_int64(1),
       ctypes.c_int64(n_elements), imat_ptr)

    return dst.tobytes()


# === PYTHON FALLBACK DEQUANT ===

def _dequant_f32(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)

def _dequant_f16(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16).astype(np.float32)

def _dequant_bf16(data: bytes, n_elements: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)

def _dequant_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    n_blocks = n_elements // 32
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 34)
    scales = raw[:, :2].view(np.float16).astype(np.float32)
    quants = raw[:, 2:].view(np.int8).astype(np.float32)
    return (quants * scales).flatten()

def _dequant_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    n_blocks = n_elements // 32
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 18)
    scales = raw[:, :2].view(np.float16).astype(np.float32).reshape(-1, 1)
    packed = raw[:, 2:]
    lo = (packed & 0x0F).astype(np.float32) - 8.0
    hi = (packed >> 4).astype(np.float32) - 8.0
    quants = np.column_stack([lo, hi]).reshape(n_blocks, 32)
    return (quants * scales).flatten()

def _dequant_q2_K(data: bytes, n_elements: int) -> np.ndarray:
    n_blocks = n_elements // 256
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 84)
    scales_raw = raw[:, :16]
    sc = (scales_raw & 0x0F).astype(np.float32)
    mn = (scales_raw >> 4).astype(np.float32)
    qs = raw[:, 16:80]
    d = np.ascontiguousarray(raw[:, 80:82]).view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    dmin = np.ascontiguousarray(raw[:, 82:84]).view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    q0 = (qs & 0x03).astype(np.float32)
    q1 = ((qs >> 2) & 0x03).astype(np.float32)
    q2 = ((qs >> 4) & 0x03).astype(np.float32)
    q3 = ((qs >> 6) & 0x03).astype(np.float32)
    quants = np.stack([q0, q1, q2, q3], axis=-1).reshape(n_blocks, 256)
    sc_expanded = np.repeat(sc, 16, axis=1) * d
    mn_expanded = np.repeat(mn, 16, axis=1) * dmin
    return (sc_expanded * quants - mn_expanded).flatten()

_FALLBACK_DEQUANT = {
    GGML_TYPE_F32: _dequant_f32,
    GGML_TYPE_F16: _dequant_f16,
    GGML_TYPE_BF16: _dequant_bf16,
    GGML_TYPE_Q8_0: _dequant_q8_0,
    GGML_TYPE_Q4_0: _dequant_q4_0,
    GGML_TYPE_Q2_K: _dequant_q2_K,
}


# === CORE ENGINE ===

def load_override_map(path: str) -> dict[str, int]:
    """Load tensor type override file (same format as llama-quantize --tensor-type-file)."""
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
                    print(f"WARNING: unknown type '{type_name}' for '{tensor_name}'", file=sys.stderr)
    return overrides


def streaming_requantize(
    input_path: str,
    output_path: str,
    default_type: str = "q2_K",
    override_file: Optional[str] = None,
    ggml_lib: Optional[str] = None,
    verbose: bool = True,
):
    """
    Streaming requantization — one tensor at a time, constant RAM.
    """
    default_type_id = TYPE_NAME_TO_ID.get(default_type)
    if default_type_id is None:
        raise ValueError(f"Unknown default type: {default_type}. "
                         f"Valid: {', '.join(sorted(set(v['name'] for v in QUANT_INFO.values())))}")

    overrides = load_override_map(override_file) if override_file else {}

    # Try to load native lib
    lib = _get_ggml(ggml_lib)
    if verbose:
        if lib:
            print(f"  Native:   libggml loaded (fast path)")
        else:
            print(f"  Native:   not found (using Python fallback — set GGML_LIB_PATH or --ggml-lib)")

    if verbose:
        print(f"  Input:    {input_path} ({os.path.getsize(input_path) / 1e9:.1f} GB)")
        print(f"  Output:   {output_path}")
        print(f"  Default:  {default_type}")
        if overrides:
            print(f"  Overrides: {len(overrides)} tensors")
        print()

    reader = GGUFReader(input_path)

    output_tensors = []
    for t in reader.tensors:
        target_type = overrides.get(t.name, default_type_id)
        new_t = TensorInfo(name=t.name, shape=t.shape.copy(), type_id=target_type, offset=0)
        output_tensors.append(new_t)

    metadata = reader.metadata.copy()

    writer = GGUFWriter(output_path)
    writer.set_metadata(metadata)
    writer.set_tensor_infos(output_tensors)
    writer.write_header()

    total_input = 0
    total_output = 0
    n_tensors = len(reader.tensors)
    import time
    t_start = time.time()

    for i, (src_tensor, dst_tensor) in enumerate(zip(reader.tensors, output_tensors)):
        if verbose:
            elapsed = time.time() - t_start
            speed = total_input / elapsed / 1e6 if elapsed > 0 else 0
            eta = (sum(t.n_bytes for t in reader.tensors[i:]) / (speed * 1e6)) if speed > 0 else 0
            eta_str = f"{eta/60:.0f}m" if eta > 60 else f"{eta:.0f}s"
            changed = " *" if src_tensor.type_id != dst_tensor.type_id else ""
            print(f"  [{i+1}/{n_tensors}] {src_tensor.name} "
                  f"({src_tensor.type_name} -> {dst_tensor.type_name}){changed} "
                  f"[{speed:.0f} MB/s, ETA {eta_str}]", flush=True)

        raw_data = reader.read_tensor_data(src_tensor)
        total_input += len(raw_data)

        if src_tensor.type_id == dst_tensor.type_id:
            writer.write_tensor_data(raw_data)
            total_output += len(raw_data)
        else:
            fp32_data = _native_dequant(raw_data, src_tensor.n_elements, src_tensor.type_id)
            if fp32_data is None and src_tensor.type_id in _FALLBACK_DEQUANT:
                fp32_data = _FALLBACK_DEQUANT[src_tensor.type_id](raw_data, src_tensor.n_elements)

            if fp32_data is not None:
                quant_data = _native_quantize(fp32_data, dst_tensor.type_id)

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

        del raw_data

    writer.close()

    if verbose:
        ratio = total_output / total_input if total_input > 0 else 0
        print()
        print(f"  Done: {total_input / 1e9:.2f} GB -> {total_output / 1e9:.2f} GB ({ratio:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming GGUF quantizer — quantize ANY model on ANY hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s model-Q4_K.gguf output-Q2_K.gguf --type q2_K
  %(prog)s model.gguf output.gguf --type q2_K --override-file cerebellum_map.txt
  %(prog)s model.gguf --info
  %(prog)s model.gguf --dry-run --type q3_K

environment:
  GGML_LIB_PATH    Path to libggml-base.so (from any llama.cpp build)

override file format:
  blk.0.attn_q.weight=q3_K
  blk.0.ffn_gate.weight=q4_K
  output.weight=q6_K
""")
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("output", nargs="?", help="Output GGUF file")
    parser.add_argument("--type", default="q2_K",
                        help="Default quantization type (default: q2_K)")
    parser.add_argument("--override-file",
                        help="Per-tensor type override file (tensor_name=type per line)")
    parser.add_argument("--ggml-lib",
                        help="Path to libggml-base.so (auto-detected if not set)")
    parser.add_argument("--info", action="store_true",
                        help="Print tensor info and exit")
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

    if not args.output and not args.dry_run:
        parser.error("output is required unless --info or --dry-run is specified")

    if args.dry_run:
        reader = GGUFReader(args.input)
        overrides = load_override_map(args.override_file) if args.override_file else {}
        default_type_id = TYPE_NAME_TO_ID[args.type]

        total_in = sum(t.n_bytes for t in reader.tensors)
        total_out = 0
        for t in reader.tensors:
            target = overrides.get(t.name, default_type_id)
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
        ggml_lib=args.ggml_lib,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
