"""Cerebellum quantization engine.

This is the durable, resumable CLI engine for per-tensor quantization search.
It overlaps CPU quantization with GPU perplexity measurement for each tensor,
records every observable event, and keeps enough state on disk to recover after
process death or a system lockup.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import queue
import re
import shlex
import shutil
import sqlite3
import signal
import subprocess
import sys
import threading
import time
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_LEVELS = ["q3_K", "q2_K", "q5_K", "q6_K", "f16"]
DEFAULT_QUANTIZE = os.environ.get("LLAMA_QUANTIZE_BIN", "llama-quantize")
DEFAULT_PERPLEXITY = os.environ.get("LLAMA_PERPLEXITY_BIN", "llama-perplexity")
DEFAULT_DB = os.environ.get("CEREBELLUM_DB", str(Path.cwd() / "db" / "cerebellum.db"))
PPL_PROFILES = {
    "wiki": ["wiki.test.raw", "wikitext-2-raw-test.txt", "wikitext-test.txt"],
    "agentic": ["cerebellum_calibration_agent.txt", "cerebellum_calibration_agent_strict.txt"],
    "code": ["cerebellum_calibration_code.txt"],
    "math": ["cerebellum_calibration_math.txt"],
    "dialogue": ["cerebellum_calibration_dialogue.txt"],
    "all-around": ["cerebellum_calibration_combined.txt"],
}
BENCHMARK_SUITES = {
    "release": [
        "arc",
        "hellaswag",
        "mmlu",
        "mmlu_redux",
        "humaneval",
        "evalplus",
        "ppl",
    ],
    "release-local": [
        "arc",
        "hellaswag",
        "mmlu_redux",
        "evalplus",
        "speed",
    ],
    "frontier": [
        "mmlu_pro",
        "gpqa_diamond",
        "mmmlu",
        "hle_no_tools",
        "livecodebench_v6",
    ],
    "capability": [
        "mmlu_pro",
        "gpqa_diamond",
        "mmmlu",
        "hle_no_tools",
        "livecodebench_v6",
        "aime_2025",
        "ifeval",
        "bfcl_v3",
        "swebench_verified",
        "aider_polyglot",
    ],
    "full": [
        "arc",
        "hellaswag",
        "mmlu",
        "mmlu_redux",
        "humaneval",
        "evalplus",
        "mmlu_pro",
        "gpqa_diamond",
        "mmmlu",
        "hle_no_tools",
        "livecodebench_v6",
        "aime_2025",
        "ifeval",
        "bfcl_v3",
        "swebench_verified",
        "aider_polyglot",
        "ppl",
        "speed",
    ],
}
BENCHMARK_SUITE_PURPOSES = {
    "release": "model-card release proof: classic MCQ, code, PPL, and detailed audit artifacts",
    "release-local": "implemented local release run: ARC, HellaSwag, MMLU-Redux, EvalPlus chat, and speed",
    "frontier": "frontier public leaderboard core: MMLU-Pro, GPQA-Diamond, MMMLU, HLE no-tools, and LiveCodeBench v6",
    "capability": "expanded capability board: frontier core plus math, instruction following, tools, SWE, and coding agent checks",
    "full": "complete Cerebellum report: release, frontier/capability checks, speed, and PPL reporting",
}
HUMANEVAL_REBENCH_MODELS = [
    {"repo": "deucebucket/Qwen3.5-122B-A10B-Cerebellum-GGUF", "published": "2026-05-02"},
    {"repo": "deucebucket/Gemma-4-E2B-it-Cerebellum-v2-GGUF", "published": "2026-05-03"},
    {"repo": "deucebucket/Qwen3.6-27B-Cerebellum-GGUF", "published": "2026-04-29"},
    {"repo": "deucebucket/Qwen3-14B-Cerebellum-GGUF", "published": "2026-05-02"},
    {"repo": "deucebucket/Qwen3-32B-Cerebellum-GGUF", "published": "2026-05-02"},
    {"repo": "deucebucket/Qwen3-30B-A3B-Cerebellum-GGUF", "published": "2026-05-02"},
    {"repo": "deucebucket/Granite-4.1-30B-Cerebellum-GGUF", "published": "2026-05-02"},
    {"repo": "deucebucket/Granite-4.0-H-Small-Cerebellum-GGUF", "published": "2026-05-02"},
    {"repo": "deucebucket/Qwen3.6-35B-A3B-Cerebellum-GGUF", "published": "2026-05-02"},
    {"repo": "deucebucket/Gemma-4-26B-A4B-it-Cerebellum-v6-GGUF", "published": "2026-05-01"},
    {"repo": "deucebucket/Gemma-4-26B-A4B-it-Cerebellum-GGUF", "published": "2026-05-01"},
    {"repo": "deucebucket/Gemma-4-E4B-it-Cerebellum-v2-GGUF", "published": "2026-04-30"},
]
BENCHMARK_CATALOG = {
    "arc": {
        "name": "ARC-Challenge",
        "status": "implemented",
        "script": "scripts/benchmark_arc.py",
        "workers": 4,
        "artifacts": ["{model}_arc_results.json", "{model}_arc_detailed.jsonl"],
        "audit": "jq 'select(.correct == false)' {results_dir}/{model}_arc_detailed.jsonl | head -30",
    },
    "hellaswag": {
        "name": "HellaSwag",
        "status": "implemented",
        "script": "scripts/benchmark_hellaswag.py",
        "workers": 4,
        "artifacts": ["{model}_hellaswag_results.json", "{model}_hellaswag_detailed.jsonl"],
        "audit": "jq 'select(.correct == false)' {results_dir}/{model}_hellaswag_detailed.jsonl | head -30",
    },
    "mmlu_redux": {
        "name": "MMLU-Redux",
        "status": "implemented",
        "script": "scripts/benchmark_mmlu_redux.py",
        "workers": 4,
        "artifacts": ["{model}_mmlu_redux_results.json", "{model}_mmlu_redux_detailed.jsonl"],
        "audit": "jq 'select(.correct == false)' {results_dir}/{model}_mmlu_redux_detailed.jsonl | head -30",
    },
    "humaneval": {
        "name": "HumanEval+ / EvalPlus chat",
        "status": "implemented",
        "script": "scripts/benchmark_evalplus_chat.py",
        "workers": 1,
        "max_tokens": 4096,
        "artifacts": ["{model}_evalplus_chat_results.json", "{model}_evalplus_chat_samples.jsonl"],
        "audit": "python scripts/audit_evalplus_completions.py {results_dir}/{model}_evalplus_chat_samples.jsonl",
    },
    "evalplus": {
        "name": "EvalPlus chat",
        "status": "implemented",
        "script": "scripts/benchmark_evalplus_chat.py",
        "workers": 1,
        "max_tokens": 4096,
        "artifacts": ["{model}_evalplus_chat_results.json", "{model}_evalplus_chat_samples.jsonl"],
        "audit": "python scripts/audit_evalplus_completions.py {results_dir}/{model}_evalplus_chat_samples.jsonl",
    },
    "speed": {
        "name": "Throughput / latency",
        "status": "implemented",
        "script": "scripts/benchmark_perf.py",
        "workers": 1,
        "artifacts": ["{model}_perf_results.json"],
    },
    "mmlu": {
        "name": "MMLU",
        "status": "pending",
        "note": "Use MMLU-Redux for current release rows until a standard MMLU runner is added.",
    },
    "ppl": {
        "name": "Perplexity",
        "status": "external",
        "note": "Run llama-perplexity with the same prompt corpus/context used for ablation.",
    },
    "mmlu_pro": {
        "name": "MMLU-Pro",
        "status": "implemented",
        "script": "scripts/benchmark_lm_eval.py",
        "workers": 4,
        "env": {"LM_EVAL_TASK": "mmlu_pro", "BENCHMARK_NAME": "mmlu_pro"},
        "artifacts": ["{model}_mmlu_pro_results.json", "{model}_mmlu_pro_lm_eval/"],
        "audit": "cerebellum benchmark-audit {results_dir}/{model}_mmlu_pro_lm_eval",
    },
    "gpqa_diamond": {
        "name": "GPQA-Diamond",
        "status": "implemented",
        "script": "scripts/benchmark_lm_eval.py",
        "workers": 4,
        "env": {"LM_EVAL_TASK": "gpqa_diamond_zeroshot", "BENCHMARK_NAME": "gpqa_diamond"},
        "artifacts": ["{model}_gpqa_diamond_results.json", "{model}_gpqa_diamond_lm_eval/"],
        "audit": "cerebellum benchmark-audit {results_dir}/{model}_gpqa_diamond_lm_eval",
        "note": "GPQA is gated on Hugging Face; authenticate and accept dataset terms before running.",
    },
    "mmmlu": {
        "name": "MMMLU / Global-MMLU",
        "status": "implemented",
        "script": "scripts/benchmark_lm_eval.py",
        "workers": 4,
        "env": {"LM_EVAL_TASK": "global_mmlu", "BENCHMARK_NAME": "mmmlu"},
        "artifacts": ["{model}_mmmlu_results.json", "{model}_mmmlu_lm_eval/"],
        "audit": "cerebellum benchmark-audit {results_dir}/{model}_mmmlu_lm_eval",
    },
    "hle_no_tools": {
        "name": "HLE no-tools",
        "status": "implemented",
        "script": "scripts/benchmark_hle_no_tools.py",
        "workers": 4,
        "max_tokens": 8192,
        "artifacts": ["{model}_hle_no_tools_results.json", "{model}_hle_no_tools_predictions.json"],
        "note": "Requires the CAIS HLE eval package or HLE_EVAL_DIR and gated HF dataset access; judge model is configured separately.",
    },
    "livecodebench_v6": {
        "name": "LiveCodeBench v6",
        "status": "implemented",
        "script": "scripts/benchmark_livecodebench_v6.py",
        "workers": 1,
        "max_tokens": 4096,
        "artifacts": ["{model}_livecodebench_v6_results.json", "{model}_livecodebench_v6/"],
        "audit": "cerebellum benchmark-audit {results_dir}/{model}_livecodebench_v6",
    },
    "aime_2025": {"name": "AIME 2025", "status": "pending", "note": "Add deterministic math runner with exact-answer normalization."},
    "ifeval": {"name": "IFEval", "status": "pending", "note": "Add instruction-following runner with strict/loose scoring artifacts."},
    "bfcl_v3": {"name": "BFCL v3", "status": "pending", "note": "Add function-calling runner and JSON/tool-call validity audit."},
    "swebench_verified": {"name": "SWE-bench Verified", "status": "pending", "note": "Add patch-generation harness for larger code-specialized releases."},
    "aider_polyglot": {"name": "Aider Polyglot", "status": "pending", "note": "Add coding-agent benchmark for practical edit quality."},
}
TASK_PROFILES = {
    "general": {
        "label": "Cerebellum-General",
        "ppl_profile": "wiki",
        "ablation_metric": "ppl",
        "benchmark_suite": "release",
        "metrics": ["ppl", "arc", "hellaswag", "mmlu_redux", "evalplus"],
        "variant_suffix": "general",
        "note": "General language, reasoning, and code release profile.",
    },
    "code": {
        "label": "Cerebellum-Code",
        "ppl_profile": "code",
        "ablation_metric": "humaneval",
        "benchmark_suite": "release-local",
        "metrics": ["humaneval", "evalplus", "livecodebench_v6"],
        "variant_suffix": "code",
        "note": "Protect tensors that matter for code generation and execution accuracy.",
    },
    "reason": {
        "label": "Cerebellum-Reason",
        "ppl_profile": "wiki",
        "ablation_metric": "mmlu",
        "benchmark_suite": "release-local",
        "metrics": ["arc", "mmlu_redux", "mmlu_pro", "gpqa_diamond"],
        "variant_suffix": "reason",
        "note": "Protect tensors that matter for science, knowledge, and reasoning MCQ accuracy.",
    },
    "chat": {
        "label": "Cerebellum-Chat",
        "ppl_profile": "dialogue",
        "ablation_metric": "dialogue",
        "benchmark_suite": "release",
        "metrics": ["dialogue", "mt_bench_pending"],
        "variant_suffix": "chat",
        "note": "Protect conversational behavior; local MT-Bench-style runner is still pending.",
    },
    "tools": {
        "label": "Cerebellum-Tools",
        "ppl_profile": "agentic",
        "ablation_metric": "tool-call",
        "benchmark_suite": "release",
        "metrics": ["tool_call_pending", "json_schema_pending", "agentic"],
        "variant_suffix": "tools",
        "note": "Protect function calling and structured JSON behavior; harness adapters are pending.",
    },
    "cpu-offload": {
        "label": "Cerebellum-CPU-Offload",
        "ppl_profile": "all-around",
        "ablation_metric": "ppl",
        "benchmark_suite": "release-local",
        "metrics": ["ppl", "speed", "score_per_gib", "cpu_tok_s", "gpu_offload_layers"],
        "variant_suffix": "cpu-offload",
        "low_space_default": True,
        "resource_strategy": {
            "target": "large RAM hosts with optional GPU layer offload",
            "scratch": "streaming/low-space artifacts preferred for huge GGUFs",
            "benchmark": "record CPU/RAM tok/s, GPU-offload layer count, size GiB, and quality score",
        },
        "note": "Plan huge-model maps such as GLM-5.1 for CPU-offload speed/quality instead of VRAM-only fit.",
    },
    "legacy-gated": {
        "label": "Cerebellum-Legacy-Gated",
        "ppl_profile": "all-around",
        "ablation_metric": "ppl",
        "benchmark_suite": "release-local",
        "metrics": ["ppl", "evalplus", "arc", "hellaswag", "mmlu_redux", "speed", "size_gib"],
        "variant_suffix": "legacy-gated",
        "low_space_default": True,
        "resource_strategy": {
        "target": "old Cerebellum/Qwen-style group survivability quant selection",
        "search": "group-first Q2_K crush from a sane baseline; per-tensor refinement is optional follow-up work",
            "gate": "accept PPL wins only when benchmark smoke/full release gates stay inside thresholds",
            "protection": "attention/output/early-block golden-cow floors are explicit and visible in watch",
        },
        "note": "Use the proven group-first, benchmark-gated workflow instead of exhaustive wiki-only per-tensor hillclimb.",
    },
}
LEGACY_GATED_GROUPS = [
    {
        "name": "attn-q",
        "patterns": ["attn_q"],
        "default_floor": "q4_K",
        "why": "Classic Cerebellum tests attention groups independently at Q2_K from the baseline.",
    },
    {
        "name": "attn-k",
        "patterns": ["attn_k"],
        "default_floor": "q4_K",
        "why": "Classic Cerebellum tests attention groups independently at Q2_K from the baseline.",
    },
    {
        "name": "attn-v",
        "patterns": ["attn_v"],
        "default_floor": "q4_K",
        "why": "Value projections are often load-bearing; test as a separate group instead of assuming demotability.",
    },
    {
        "name": "attn-output",
        "patterns": ["attn_output"],
        "default_floor": "q4_K",
        "why": "Attention output projections are often sensitive; test as a separate group.",
    },
    {
        "name": "ffn-gate",
        "patterns": ["ffn_gate"],
        "default_floor": "q4_K",
        "why": "Gate weights can be regularized by lower precision, but Gemma releases need benchmark-gated proof.",
    },
    {
        "name": "ffn-up",
        "patterns": ["ffn_up"],
        "default_floor": "q4_K",
        "why": "Classic Cerebellum tests MLP projection groups independently at Q2_K from the baseline.",
    },
    {
        "name": "ffn-down",
        "patterns": ["ffn_down"],
        "default_floor": "q4_K",
        "why": "Classic Cerebellum tests MLP projection groups independently at Q2_K from the baseline.",
    },
    {
        "name": "early-blocks",
        "patterns": ["blk.0.", "blk.1.", "blk.2.", "blk.3.", "blk.4."],
        "default_floor": "q4_K",
        "why": "Gemma 4 floor is Q4; early-layer crushes must clear a benchmark gate before release.",
    },
    {
        "name": "global-output-embeddings-norms",
        "patterns": ["output.weight", "token_embd", "embd", "norm"],
        "default_floor": "q6_K",
        "why": "Global output/embedding/norm tensors are not search trash; protect unless a benchmark gate proves otherwise.",
        "scan": False,
    },
]
LEGACY_GATED_PHASES = [
    "inventory tensor groups and skip known non-quantizable/global tensors",
    "build the lower-quant baseline/candidate from the F16 source GGUF, then run and benchmark that quant output",
    "run coarse group Q2_K survivability ablations before any per-tensor hillclimb",
    "protect golden-cow groups with explicit floors",
    "run interaction/additive checks on candidate groups",
    "run benchmark smoke gates before accepting a group override map",
    "run full benchmark comparison before release or upload",
]
LEGACY_PROFILE_ROOTS = [
    Path("/var/home/deucebucket/games/osmosis-quants"),
    Path("/var/home/deucebucket/games"),
    Path("/var/home/deucebucket/games/cerebellum-calibration"),
]
PRECISION_RANK = {
    "q2_K": 2,
    "q3_K": 3,
    "q4_K": 4,
    "q5_K": 5,
    "q6_K": 6,
    "q8_0": 8,
    "f16": 16,
    "bf16": 16,
}
NON_QUANTIZABLE_SUBSTRINGS = (
    "rope",
    "embd",
    "_norm.weight",
    "ffn_gate_inp.weight",
    "layer_output_scale.weight",
    "altup",
    "laurel",
    "per_layer_model_proj",
    "ssm_conv1d",
    "shortconv.conv.weight",
    "time_mix_first.weight",
    "time_mix_w0.weight",
    "time_mix_w1.weight",
    "time_mix_w2.weight",
    "time_mix_v0.weight",
    "time_mix_v1.weight",
    "time_mix_v2.weight",
    "time_mix_a0.weight",
    "time_mix_a1.weight",
    "time_mix_a2.weight",
    "time_mix_g1.weight",
    "time_mix_g2.weight",
    "time_mix_decay_w1.weight",
    "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
    ".position_embd",
    "sam.pos_embd",
    "sam.neck.",
    "sam.net_",
    ".rel_pos",
    ".patch_embd",
    ".patch_merger",
)
EVENT_SCHEMA_VERSION = 1
EVENT_FILES = ("cerebellum_events.jsonl", "cerebellum_hill_events.jsonl")
CANDIDATE_FILES = ("cerebellum_candidates.jsonl", "cerebellum_hill_candidates.jsonl")
SUMMARY_JSON_FILES = ("cerebellum_summary.json", "cerebellum_hill_summary.json")
SUMMARY_MD_FILES = ("cerebellum_summary.md", "cerebellum_hill_summary.md")
DECISION_CSV_FILES = ("cerebellum_decisions.csv", "cerebellum_hill_decisions.csv")
INFOGRAPHIC_FILES = ("cerebellum_infographic_data.json", "cerebellum_hill_infographic_data.json")
BEST_TYPES_FILES = ("cerebellum_best_tensor_types.txt", "cerebellum_hill_best_tensor_types.txt")
CURRENT_TYPES_FILE = "cerebellum_current_tensor_types.txt"
DEFAULT_WATCH_EVENTS_LIMIT = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slug(value: str) -> str:
    value = value.strip().replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-") or "unknown"


def tensor_type_pattern(name: str) -> str:
    return f"^{re.escape(name)}$"


def tensor_type_line(name: str, qtype: str) -> str:
    return f"{tensor_type_pattern(name)}={qtype}"


def is_quantizable_tensor(name: str) -> bool:
    if not name.endswith("weight"):
        return False
    return not any(part in name for part in NON_QUANTIZABLE_SUBSTRINGS)


def quantizable_tensor_names(names: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    return [name for name in names if is_quantizable_tensor(name)]


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


class EventLog:
    def __init__(self, path: Path, run_id: str, cfg: Config | None = None):
        self.path = path
        self.run_id = run_id
        self.cfg = cfg
        self._event_id = max(
            [int(row.get("event_id") or 0) for row in read_jsonl(path)],
            default=0,
        )
        self._started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._event_id += 1
            row = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_id": self._event_id,
                "run_id": self.run_id,
                "timestamp_utc": utc_now(),
                "monotonic_s": round(time.monotonic() - self._started, 3),
                "pid": os.getpid(),
                "event": event,
                **fields,
            }
            if self.cfg is not None:
                row.setdefault("model_family", self.cfg.model_family)
                row.setdefault("model_name", self.cfg.model_name)
                row.setdefault("source_name", self.cfg.source_name)
            line = json.dumps(row, sort_keys=True) + "\n"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())


def color(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SECRET_ENV_RE = re.compile(
    r"((?:--env=)?[A-Za-z_][A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASS|KEY|AUTH)[A-Za-z0-9_]*=)([^ \t]+)"
)


def sanitize_process_cmd(cmd: str) -> str:
    return SECRET_ENV_RE.sub(r"\1<redacted>", cmd)


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def ansi_clip(text: str, width: int) -> str:
    plain = ANSI_RE.sub("", text)
    if len(plain) <= width:
        return text + " " * (width - len(plain))
    clipped = plain[: max(0, width - 1)] + ("…" if width > 0 else "")
    return clipped


def ansi_pad(text: str, width: int) -> str:
    length = visible_len(text)
    if length > width:
        return ansi_clip(text, width)
    return text + " " * (width - length)


def ansi_wrap(text: str, width: int) -> list[str]:
    if visible_len(str(text)) <= width:
        return [str(text)]
    plain = ANSI_RE.sub("", str(text))
    if width <= 1:
        return [plain[:width]]
    wrapped = textwrap.wrap(plain, width=width, break_long_words=True, break_on_hyphens=False)
    return wrapped or [""]


def kv_line(label: str, value: Any, width: int, enabled: bool, value_code: str = "37;1") -> str:
    label_part = color(f"{label:<9}", "90", enabled)
    value_text = str(value)
    value_part = color(f"{value_text:<{width - 14}}", value_code, enabled)
    return f"│ {label_part}{value_part}│"


def delta_code(delta: Any) -> str:
    if delta is None:
        return "90"
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return "90"
    if value < 0:
        return "32;1"
    if value > 0:
        return "31;1"
    return "36;1"


def delta_marker(delta: Any) -> tuple[str, str]:
    if delta is None:
        return "", "90"
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return "", "90"
    if value < 0:
        return "better", "32;1"
    if value > 0:
        return "worse", "31;1"
    return "=", "36;1"


def ablation_verdict(delta: Any, tensor: str | None = None, phase: str = "forward") -> tuple[str, str]:
    code = "90"
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return "· pending", code
    target = tensor or "tensor"
    if phase == "reverse":
        if value < 0:
            return f"✓ restore this {target}", "32;1"
        if value > 0:
            return f"✗ keep smashed {target}", "31;1"
        return f"= neutral restore for {target}", "36;1"
    if value < 0:
        return f"✓ smash this {target}", "32;1"
    if value > 0:
        return f"✗ nah, do not smash this {target}", "31;1"
    return f"= neutral for {target}", "36;1"


def candidate_measurement_verdict(row: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, str]:
    status = str(row.get("status") or "")
    if row.get("ppl") is None and status:
        labels = {
            "quantizing": ("quantizing", "33;1"),
            "queued": ("queued", "36;1"),
            "ppl": ("ppl running", "33;1"),
            "pending": ("pending", "90"),
        }
        if status in labels:
            return labels[status]
    delta = row.get("delta")
    try:
        value = float(delta)
    except (TypeError, ValueError):
        return "pending", "90"
    if value > 0:
        return "worse", "31;1"
    if value == 0:
        return "same", "36;1"
    tensor = row.get("tensor")
    try:
        ppl = float(row.get("ppl"))
    except (TypeError, ValueError):
        return "better", "32;1"
    same_tensor = []
    for candidate in rows:
        if candidate.get("tensor") != tensor:
            continue
        try:
            candidate_ppl = float(candidate.get("ppl"))
        except (TypeError, ValueError):
            continue
        same_tensor.append(candidate_ppl)
    if same_tensor and ppl <= min(same_tensor):
        return "best", "32;1"
    return "better", "32"


def in_progress_candidate_rows(events: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tensor_start_idx = next(
        (idx for idx in range(len(events) - 1, -1, -1) if events[idx].get("event") == "tensor_start" and events[idx].get("tensor")),
        None,
    )
    if tensor_start_idx is None:
        return []
    active_tensor = events[tensor_start_idx].get("tensor")
    completed = {(row.get("tensor"), row.get("level")) for row in candidates if row.get("tensor") and row.get("level")}
    by_level: dict[str, dict[str, Any]] = {}
    order: dict[str, int] = {}
    for event in events[tensor_start_idx + 1 :]:
        if event.get("tensor") != active_tensor:
            continue
        level = event.get("level")
        if not level or (active_tensor, level) in completed:
            continue
        name = str(event.get("event") or "")
        if name.startswith("quant_"):
            status = "queued" if name == "quant_finish" and event.get("returncode") == 0 else "quantizing"
        elif name.startswith("ppl_"):
            status = "done" if name == "ppl_finish" and event.get("returncode") == 0 and event.get("ppl") is not None else "ppl"
        else:
            continue
        order.setdefault(str(level), len(order))
        row = by_level.setdefault(
            str(level),
            {
                "event": "candidate_pending",
                "tensor": active_tensor,
                "level": level,
                "ppl": None,
                "delta": None,
                "size_bytes": None,
                "status": status,
            },
        )
        row["status"] = status
        if event.get("size_bytes") is not None:
            row["size_bytes"] = event.get("size_bytes")
        if event.get("ppl") is not None:
            row["ppl"] = event.get("ppl")
        if event.get("delta") is not None:
            row["delta"] = event.get("delta")
    rows = [row for level, row in by_level.items() if row.get("status") != "done" and (active_tensor, level) not in completed]
    rows.sort(key=lambda row: order.get(str(row.get("level")), 999))
    return rows


def limited_tail(rows: list[Any], limit: int) -> list[Any]:
    if limit <= 0:
        return rows
    return rows[-limit:]


def size_code(size: Any, baseline_size: Any) -> str:
    if size is None or baseline_size is None:
        return "90"
    try:
        value = int(size)
        baseline = int(baseline_size)
    except (TypeError, ValueError):
        return "90"
    if value < baseline:
        return "34;1"
    if value > baseline:
        return "33;1"
    return "36;1"


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(sec):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


def fmt_completion_time(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if value < 0:
        return "-"
    when = datetime.now().astimezone() + timedelta(seconds=value)
    return when.strftime("%a %Y-%m-%d %H:%M %Z")


def fmt_bytes(size: int | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


def fmt_bytes_dense(size: int | None) -> str:
    if size is None:
        return "-"
    if size >= 1024**3:
        return f"{size / 1024**3:.3f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    return fmt_bytes(size)


def progress_bar(done: int, total: int | None, width: int = 28) -> tuple[str, str]:
    if not total:
        return "[" + "-" * width + "]", "-"
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(ratio * width))
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    return bar, f"{done}/{total} {ratio * 100:.1f}%"


def event_age_seconds(row: dict[str, Any]) -> float | None:
    timestamp = row.get("timestamp_utc")
    if not timestamp:
        return None
    try:
        then = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds())


def event_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def process_rows_for_run(run_dir: Path) -> list[dict[str, Any]]:
    proc = subprocess.run(["ps", "-eo", "pid=,ppid=,stat=,etime=,pcpu=,pmem=,cmd="], capture_output=True, text=True)
    rows: list[dict[str, Any]] = []
    run_key = str(run_dir)
    run_name = run_dir.name
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, stat, etime, pcpu, pmem, cmd = parts
        if run_key not in cmd and run_name not in cmd:
            continue
        if "cerebellum watch" in cmd:
            continue
        kind = "process"
        if (
            "cerebellum run" in cmd
            or "cerebellum resume" in cmd
            or "cerebellum group-scan" in cmd
            or "cerebellum.hillstep run" in cmd
            or "cerebellum.hillstep resume" in cmd
            or "cerebellum.hillstep group-scan" in cmd
            or "osmosis.hillstep run" in cmd
            or "osmosis.hillstep resume" in cmd
            or "osmosis.hillstep group-scan" in cmd
        ):
            kind = "runner"
        elif cmd.startswith("/usr/bin/sh /usr/bin/distrobox") or cmd.startswith("podman exec"):
            kind = "container"
        elif "llama-quantize" in cmd:
            kind = "quantize"
        elif "llama-perplexity" in cmd:
            kind = "ppl"
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "stat": stat,
                "etime": etime,
                "pcpu": pcpu,
                "pmem": pmem,
                "kind": kind,
                "cmd": sanitize_process_cmd(cmd),
            }
        )
    rows.sort(key=lambda row: {"runner": 0, "quantize": 1, "ppl": 2, "container": 3}.get(row["kind"], 9))
    return rows


def active_work_status(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    processes: list[dict[str, Any]],
    active: dict[str, Any],
    warn_seconds: float = 300.0,
    fail_seconds: float = 900.0,
) -> dict[str, Any]:
    status = state.get("run_status")
    last_event = events[-1] if events else {}
    last_event_age = event_age_seconds(last_event)
    active_age = event_age_seconds(active)
    active_processes = [row for row in processes if row["kind"] in {"quantize", "ppl"}]
    runner_processes = [row for row in processes if row["kind"] == "runner"]
    expected_pid = str(active.get("pid") or "")
    expected_pid_alive = any(row["pid"] == expected_pid for row in processes) if expected_pid else False
    active_event = str(active.get("event") or "")
    incomplete_start = active_event in {"tensor_start", "quant_start", "ppl_start"}
    stale = bool(status == "running" and incomplete_start and not active_processes and not runner_processes)

    health = "idle"
    reason = "not running"
    code = "90"
    if status == "running":
        if active_processes:
            health = "active"
            reason = ", ".join(f"{row['kind']} pid {row['pid']} {row['etime']}" for row in active_processes[:2])
            code = "32;1"
        elif runner_processes and last_event_age is not None and last_event_age < warn_seconds:
            health = "waiting"
            reason = f"runner alive; last event {fmt_seconds(last_event_age)} ago"
            code = "33;1"
        elif runner_processes and last_event_age is not None and last_event_age < fail_seconds:
            health = "stalled?"
            reason = f"runner alive but no event for {fmt_seconds(last_event_age)}"
            code = "33;1"
        elif runner_processes:
            health = "failure suspected"
            reason = f"runner alive, no event for {fmt_seconds(last_event_age)}"
            code = "31;1"
        elif stale:
            health = "interrupted"
            reason = f"{active_event} has no live process; resume will retest current tensor"
            code = "31;1"
        else:
            health = "failure suspected"
            reason = "state says running but no runner process found"
            code = "31;1"
    return {
        "health": health,
        "reason": reason,
        "code": code,
        "last_event_age": last_event_age,
        "active_age": active_age,
        "active_processes": active_processes,
        "runner_processes": runner_processes,
        "stale": stale,
        "expected_pid": expected_pid or None,
        "expected_pid_alive": expected_pid_alive,
    }


def gpu_rows() -> list[dict[str, Any]]:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return []
    proc = subprocess.run(
        [
            nvidia,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "util": parts[2],
                "mem_used": parts[3],
                "mem_total": parts[4],
                "power": parts[5],
            }
        )
    return rows


def estimate_eta(state: dict[str, Any], active_age: float | None, total: int | None) -> tuple[str, str]:
    locked = len(state.get("locked", {}))
    if not total:
        return "-", "no total tensor count yet"
    tested = state.get("tested", [])
    if not tested:
        return "-", "waiting for first locked tensor"
    totals = state.get("totals", {})
    elapsed = (totals.get("quant_seconds") or 0.0) + (totals.get("ppl_seconds") or 0.0)
    if active_age:
        elapsed += active_age
    completed = max(1, len(tested))
    avg = elapsed / completed
    remaining = max(0, total - locked)
    eta = remaining * avg
    return fmt_seconds(eta), f"avg {fmt_seconds(avg)}/tensor from {completed} locked"


def median_value(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value and value > 0)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def runs_root_for(run_dir: Path) -> Path | None:
    parts = list(run_dir.parents)
    for parent in parts:
        if parent.name == "runs":
            return parent
    return None


def timing_candidate_rows(run_dir: Path, manifest: dict[str, Any], current_candidates: list[dict[str, Any]], limit: int = 1200) -> list[dict[str, Any]]:
    rows = [row for row in current_candidates if row.get("quant_seconds") or row.get("ppl_seconds")]
    model = manifest.get("model_name")
    family = manifest.get("model_family")
    root = runs_root_for(run_dir)
    if root and root.exists():
        files = sorted(root.rglob(CANDIDATE_FILES[0]), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
        for file in files:
            if file == first_existing(run_dir, CANDIDATE_FILES):
                continue
            for row in read_jsonl(file):
                if model and row.get("model_name") and row.get("model_name") != model:
                    continue
                if family and row.get("model_family") and row.get("model_family") != family:
                    continue
                if row.get("quant_seconds") or row.get("ppl_seconds"):
                    rows.append(row)
                    if len(rows) >= limit:
                        return rows[:limit]
    return rows[:limit]


def timing_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quant_by_level: dict[str, list[float]] = {}
    ppl_by_level: dict[str, list[float]] = {}
    quant_by_component: dict[str, list[float]] = {}
    ppl_by_component: dict[str, list[float]] = {}
    quant_all: list[float] = []
    ppl_all: list[float] = []
    for row in rows:
        level = str(row.get("level") or "")
        _layer, component = parse_tensor_name(str(row.get("tensor") or ""))
        component = component or "unknown"
        quant = row.get("quant_seconds")
        ppl = row.get("ppl_seconds")
        if isinstance(quant, (int, float)) and quant > 0:
            quant_all.append(float(quant))
            quant_by_level.setdefault(level, []).append(float(quant))
            quant_by_component.setdefault(component, []).append(float(quant))
        if isinstance(ppl, (int, float)) and ppl > 0:
            ppl_all.append(float(ppl))
            ppl_by_level.setdefault(level, []).append(float(ppl))
            ppl_by_component.setdefault(component, []).append(float(ppl))
    return {
        "rows": len(rows),
        "quant_all": median_value(quant_all),
        "ppl_all": median_value(ppl_all),
        "quant_by_level": {key: median_value(value) for key, value in quant_by_level.items()},
        "ppl_by_level": {key: median_value(value) for key, value in ppl_by_level.items()},
        "quant_by_component": {key: median_value(value) for key, value in quant_by_component.items()},
        "ppl_by_component": {key: median_value(value) for key, value in ppl_by_component.items()},
    }


def profile_seconds(profile: dict[str, Any], kind: str, level: str | None = None, component: str | None = None) -> float | None:
    by_level = profile.get(f"{kind}_by_level") or {}
    by_component = profile.get(f"{kind}_by_component") or {}
    if level and by_level.get(level):
        return float(by_level[level])
    if component and by_component.get(component):
        return float(by_component[component])
    value = profile.get(f"{kind}_all")
    return float(value) if value else None


def tensor_wall_estimate(levels: list[str], profile: dict[str, Any], component: str | None = None) -> float | None:
    if not levels:
        return None
    quant = [profile_seconds(profile, "quant", level, component) for level in levels]
    ppl = [profile_seconds(profile, "ppl", level, component) for level in levels]
    if not all(value for value in quant) or not all(value for value in ppl):
        return None
    q = [float(value) for value in quant if value is not None]
    p = [float(value) for value in ppl if value is not None]
    if len(q) != len(levels) or len(p) != len(levels):
        return None
    # HillStepper overlaps PPL(level N) with quant(level N+1) when space allows.
    wall = q[0]
    for idx in range(len(levels) - 1):
        wall += max(p[idx], q[idx + 1])
    wall += p[-1]
    return wall


def latest_event(events: list[dict[str, Any]], names: set[str], tensor: str | None = None) -> dict[str, Any]:
    return next(
        (
            row
            for row in reversed(events)
            if row.get("event") in names and (tensor is None or row.get("tensor") == tensor)
        ),
        {},
    )


def eta_summary(seconds: float | None) -> str:
    return f"{fmt_seconds(seconds)}  done {fmt_completion_time(seconds)}"


def fmt_completion_time_signed(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "-"
    when = datetime.now().astimezone() + timedelta(seconds=value)
    return when.strftime("%a %Y-%m-%d %H:%M %Z")


def eta_progress_summary(estimate: float | None, elapsed: float | None) -> str:
    if estimate is None:
        return eta_summary(None)
    remaining = float(estimate) - float(elapsed or 0.0)
    if remaining >= 0:
        return eta_summary(remaining)
    return f"overdue {fmt_seconds(abs(remaining))}  expected {fmt_completion_time_signed(remaining)}"


def eta_source_label(profile_rows: int, confidence: str) -> str:
    if profile_rows:
        return f"{confidence} from {profile_rows} prior candidate timings"
    return f"{confidence} from current run timing"


def eta_detail_values(
    run_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    active: dict[str, Any],
    active_age: float | None,
    total: int | None,
    flow: dict[str, Any],
) -> dict[str, Any]:
    profile = timing_profile(timing_candidate_rows(run_dir, manifest, candidates))
    levels = [str(level) for level in (manifest.get("levels") or state.get("levels") or DEFAULT_LEVELS)]
    tensor = str(active.get("tensor") or latest_event(events, {"tensor_start"}).get("tensor") or "")
    _layer, component = parse_tensor_name(tensor)
    event = str(active.get("event") or "")
    level = str(active.get("level") or "")
    if event.startswith("baseline_quant"):
        job_est = profile_seconds(profile, "quant", levels[0] if levels else None, component)
        label = "baseline quant"
    elif event.startswith("baseline_ppl"):
        job_est = profile_seconds(profile, "ppl", levels[0] if levels else None, component)
        label = "baseline ppl"
    elif event.startswith("quant"):
        job_est = profile_seconds(profile, "quant", level, component)
        label = f"quant {level}"
    elif event.startswith("ppl"):
        job_est = profile_seconds(profile, "ppl", level, component)
        label = f"ppl {level}"
    else:
        job_est = None
        label = event or "idle"
    job_remaining = max(0.0, (job_est or 0.0) - (active_age or 0.0)) if job_est else None
    tensor_start = latest_event(events, {"tensor_start"}, tensor or None)
    tensor_age = event_age_seconds(tensor_start)
    tensor_est = tensor_wall_estimate(levels, profile, component)
    tensor_remaining = max(0.0, (tensor_est or 0.0) - (tensor_age or 0.0)) if tensor_est else None
    if job_remaining is not None:
        tensor_remaining = max(tensor_remaining or 0.0, job_remaining)
    locked = len(state.get("locked", {}))
    remaining_tensors = max(0, (total or locked) - locked)
    phase_remaining = None
    if tensor_est:
        phase_remaining = max(0.0, remaining_tensors * tensor_est - (tensor_age or 0.0))
    flow_remaining = phase_remaining
    group = flow.get("group") or {}
    if phase_remaining is not None and group.get("total") and group.get("index"):
        groups_left_after_current = max(0, int(group["total"]) - int(group["index"]))
        flow_remaining = phase_remaining + groups_left_after_current * max(0, (total or 0) * tensor_est)
    confidence = "low"
    if profile["rows"] >= 50:
        confidence = "medium"
    if profile["rows"] >= 200 and state.get("tested"):
        confidence = "high"
    return {
        "profile_rows": profile["rows"],
        "confidence": confidence,
        "source": eta_source_label(int(profile["rows"] or 0), confidence),
        "job": {"label": label, "estimate": fmt_seconds(job_est), "remaining": eta_progress_summary(job_est, active_age), "age": fmt_seconds(active_age)},
        "tensor": {"label": tensor or "-", "estimate": fmt_seconds(tensor_est), "remaining": eta_progress_summary(tensor_est, tensor_age), "age": fmt_seconds(tensor_age)},
        "phase": {"remaining": eta_summary(phase_remaining), "tensors_remaining": remaining_tensors},
        "flow": {"remaining": eta_summary(flow_remaining), "phase": flow.get("phase") or "-"},
    }


def eta_grid_values(state: dict[str, Any], active_age: float | None, total: int | None) -> dict[str, str]:
    locked = len(state.get("locked", {}))
    tested = state.get("tested", [])
    totals = state.get("totals", {})
    elapsed = (totals.get("quant_seconds") or 0.0) + (totals.get("ppl_seconds") or 0.0)
    if active_age:
        elapsed += active_age
    avg_tensor = (elapsed / len(tested)) if tested else None
    remaining = max(0, (total or locked) - locked)
    eta = (remaining * avg_tensor) if avg_tensor else None
    by_layer: dict[str, int] = {}
    for row in tested:
        tensor = row.get("tensor", "")
        layer = tensor.split(".", 2)[1] if tensor.startswith("blk.") and "." in tensor else "other"
        by_layer[layer] = by_layer.get(layer, 0) + 1
    completed_layers = sum(1 for count in by_layer.values() if count >= 5)
    if len(tested) < 5:
        confidence = "low"
    elif completed_layers < 2:
        confidence = "medium"
    else:
        confidence = "high"
    return {
        "current": fmt_seconds(active_age),
        "avg_tensor": fmt_seconds(avg_tensor),
        "avg_layer": "-" if completed_layers == 0 or avg_tensor is None else fmt_seconds(avg_tensor * 5),
        "total": fmt_seconds(eta),
        "completion_at": fmt_completion_time(eta),
        "confidence": confidence,
    }


def locked_layer_lines(state: dict[str, Any]) -> list[str]:
    rows = state.get("tested") or []
    by_layer: dict[str, list[str]] = {}
    for row in rows:
        tensor = str(row.get("tensor") or "")
        winner = row.get("winner") or row.get("level")
        if not tensor or not winner:
            continue
        layer, component = parse_tensor_name(tensor)
        layer_label = f"blk.{layer}" if layer is not None else "global"
        component_label = component or tensor
        by_layer.setdefault(layer_label, []).append(f"{component_label}={winner}")
    if not by_layer:
        locked = state.get("locked", {})
        for tensor, winner in locked.items():
            layer, component = parse_tensor_name(str(tensor))
            layer_label = f"blk.{layer}" if layer is not None else "global"
            by_layer.setdefault(layer_label, []).append(f"{component or tensor}={winner}")
    def sort_key(label: str) -> tuple[int, str]:
        if label.startswith("blk."):
            try:
                return (int(label.split(".", 1)[1]), label)
            except ValueError:
                pass
        return (10_000, label)
    return [f"{layer:<8} " + "  ".join(entries) for layer, entries in sorted(by_layer.items(), key=lambda item: sort_key(item[0]))]


def precision_rank(level: str | None) -> int:
    return PRECISION_RANK.get(str(level or ""), 999)


def below_precision(level: str | None, floor: str) -> bool:
    return precision_rank(level) < precision_rank(floor)


def golden_cow_audit(state: dict[str, Any], manifest: dict[str, Any] | None = None, flow: dict[str, Any] | None = None) -> dict[str, Any]:
    locked = state.get("locked") or {}
    suspects: list[dict[str, Any]] = []
    counts: dict[str, int] = {
        "early_low_precision": 0,
        "attention_low_precision": 0,
        "mlp_up_gate_q2": 0,
        "global_sensitive": 0,
    }
    for tensor, level in sorted(locked.items()):
        tensor_s = str(tensor)
        level_s = str(level)
        layer, component = parse_tensor_name(tensor_s)
        reasons: list[str] = []
        floors: list[str] = []
        if layer is not None and layer <= 4 and below_precision(level_s, "q4_K"):
            reasons.append("early block <=4 below Q4")
            floors.append("q4_K")
            counts["early_low_precision"] += 1
        if component in {"attn_k", "attn_q", "attn_v", "attn_output"} and below_precision(level_s, "q4_K"):
            reasons.append("attention path below Q4")
            floors.append("q4_K")
            counts["attention_low_precision"] += 1
        if component in {"ffn_up", "ffn_gate"} and level_s == "q2_K":
            reasons.append("MLP up/gate at Q2")
            floors.append("q3_K")
            counts["mlp_up_gate_q2"] += 1
        if layer is None and any(part in tensor_s for part in ("output.weight", "embd", "norm")) and below_precision(level_s, "q6_K"):
            reasons.append("global output/embedding/norm below Q6")
            floors.append("q6_K")
            counts["global_sensitive"] += 1
        if reasons:
            floor = max(floors, key=precision_rank) if floors else "q4_K"
            suspects.append({"tensor": tensor_s, "level": level_s, "floor": floor, "reasons": reasons})
    old_method_notes = [
        "Qwen 3.6 27B used group HumanEval ablations and promotion checks, not a wiki-only per-tensor walk.",
        "Gemma E2B v1 showed blanket PPL wins can destroy benchmarks; v2 used surgical layer demotion.",
        "Use benchmark gates before accepting low-precision attention/output/early-block locks.",
    ]
    phase = (flow or {}).get("phase")
    if suspects and phase == "survivability-scan":
        recommended_action = "survivability finding; require benchmark gate before final acceptance"
    elif suspects:
        recommended_action = "benchmark-gated rollback/protection experiment"
    else:
        recommended_action = "no obvious golden-cow violations under current rules"
    return {
        "schema": "cerebellum.golden_cow_audit.v1",
        "ppl_profile": (manifest or {}).get("ppl_profile") or state.get("ppl_profile"),
        "ablation_metric": (manifest or {}).get("ablation_metric") or state.get("ablation_metric"),
        "current_ppl": state.get("current_ppl"),
        "locked": len(locked),
        "suspect_count": len(suspects),
        "counts": counts,
        "suspects": suspects,
        "old_method_notes": old_method_notes,
        "recommended_action": recommended_action,
    }


def golden_cow_watch_lines(audit: dict[str, Any], limit: int = 6) -> list[str]:
    if not audit:
        return ["No golden-cow audit available."]
    lines = [
        f"mode legacy-gated recommended  metric={audit.get('ablation_metric')}  profile={audit.get('ppl_profile')}  ppl={audit.get('current_ppl')}",
        f"locked={audit.get('locked')}  suspects={audit.get('suspect_count')}  action={audit.get('recommended_action')}",
    ]
    counts = audit.get("counts") or {}
    count_bits = [f"{key}={value}" for key, value in counts.items() if value]
    if count_bits:
        lines.append("risk counts  " + "  ".join(count_bits))
    suspects = audit.get("suspects") or []
    for row in suspects[:limit]:
        reasons = ", ".join(row.get("reasons") or [])
        lines.append(f"{row.get('level')} -> >= {row.get('floor')}  {row.get('tensor')}  {reasons}")
    if len(suspects) > limit:
        lines.append(f"... {len(suspects) - limit} more suspects; use `cerebellum legacy-plan RUN_DIR --json`")
    return lines


def same_path(a: str | Path | None, b: str | Path | None) -> bool:
    if not a or not b:
        return False
    pa = Path(str(a))
    pb = Path(str(b))
    try:
        return pa.resolve() == pb.resolve()
    except OSError:
        return str(pa) == str(pb)


def legacy_flow_context(run_dir: Path) -> dict[str, Any]:
    flow_path: Path | None = None
    for parent in [run_dir, *run_dir.parents]:
        candidate = parent / "legacy_flow.json"
        if candidate.exists():
            flow_path = candidate
            break
    if flow_path is None:
        return {}
    plan = read_json(flow_path, {})
    if not isinstance(plan, dict):
        return {}
    phase = "unknown"
    group: dict[str, Any] | None = None
    groups: list[dict[str, Any]] = []
    for row in plan.get("phases", []):
        if row.get("name") == "survivability-scan":
            groups = [item for item in row.get("groups", []) if isinstance(item, dict)]
            break
    for index, row in enumerate(groups, 1):
        if same_path(row.get("run_dir"), run_dir):
            phase = "survivability-scan"
            group = {**row, "index": index, "total": len(groups)}
            break
    if group is None and same_path(plan.get("run_dir"), run_dir):
        phase = "targeted-hillstep"
    next_phases = []
    seen = phase == "unknown"
    for row in plan.get("phases", []):
        name = row.get("name")
        if not name:
            continue
        if name == phase:
            seen = True
            continue
        if seen:
            next_phases.append(str(name))
    return {
        "schema": "cerebellum.legacy_flow_context.v1",
        "flow_path": str(flow_path),
        "mode": plan.get("mode"),
        "phase": phase,
        "group": group,
        "next_phases": next_phases[:5],
        "survivability": plan.get("survivability") or {},
        "baseline_gguf": plan.get("baseline_gguf"),
        "candidate_gguf": plan.get("candidate_gguf"),
        "watch": plan.get("watch") or {},
    }


def legacy_flow_watch_lines(
    context: dict[str, Any],
    state: dict[str, Any],
    active: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    if not context:
        return ["No legacy-flow manifest found for this run."]
    manifest = manifest or {}
    live_levels = manifest.get("levels") or state.get("levels")
    if isinstance(live_levels, list):
        live_levels_s = ",".join(str(level) for level in live_levels)
    else:
        live_levels_s = str(live_levels or context.get("survivability", {}).get("levels") or "-")
    live_commit_locks = manifest.get("commit_locks")
    if live_commit_locks is None:
        live_commit_locks = context.get("survivability", {}).get("commit_locks")
    pure_effective = manifest.get("pure_quant_effective")
    if pure_effective is None:
        pure_effective = manifest.get("pure_quant", context.get("survivability", {}).get("pure_quant"))
    scan_mode = context.get("survivability", {}).get("mode", "-")
    if live_levels_s != str(context.get("survivability", {}).get("levels") or "-"):
        scan_mode = "bidirectional no-commit survivability scan" if not live_commit_locks else "bidirectional survivability scan"
    lines = [
        f"mode {context.get('mode') or '-'}  phase={context.get('phase') or '-'}",
    ]
    group = context.get("group") or {}
    if group:
        lines.append(
            f"group {group.get('index')}/{group.get('total')} {group.get('name')}  floor={group.get('floor')}  levels={live_levels_s}"
        )
        patterns = ", ".join(str(item) for item in group.get("patterns", []))
        if patterns:
            lines.append(f"patterns {patterns}")
    survivability = context.get("survivability") or {}
    if survivability:
        lines.append(
            f"scan={scan_mode}  pure_quant_effective={pure_effective}  max_regression={survivability.get('max_regression_pct')}%"
        )
        lines.append(f"commit_locks={live_commit_locks}  target_file={Path(str(survivability.get('target_tensor_file', '-'))).name}")
    lines.append(f"baseline_ppl={state.get('current_ppl')}  active={active.get('event')} {active.get('level', '')} {active.get('tensor', '')}".strip())
    if context.get("next_phases"):
        lines.append("next " + " -> ".join(context["next_phases"]))
    return lines


def classic_ablation_root(run_dir: Path) -> Path:
    parts = list(run_dir.parts)
    for marker in ("forward", "reverse"):
        if marker in parts:
            idx = parts.index(marker)
            return Path(*parts[:idx]) if idx > 0 else Path("/")
    return run_dir


def classic_ablation_phase_lines(root: Path, phase: str, enabled: bool = False) -> list[str]:
    phase_root = root / phase
    if not phase_root.exists():
        return [f"No {phase} ablation results yet."]
    rows: list[tuple[float, str]] = []
    for path in phase_root.glob("*/" + CANDIDATE_FILES[0]):
        for row in read_jsonl(path):
            delta = row.get("delta")
            delta_s = "-" if delta is None else f"{delta:+.4f}"
            tensor = str(row.get("tensor") or path.parent.name)
            verdict, verdict_code = ablation_verdict(delta, tensor, phase=phase)
            mtime = path.stat().st_mtime if path.exists() else 0.0
            rows.append(
                (
                    mtime,
                    "".join(
                        [
                            color(f"{tensor:<12}", "37;1", enabled),
                            " ",
                            color(f"{row.get('level', '-'):<7}", "35;1", enabled),
                            " ",
                            color(f"{str(row.get('ppl', '-')):<12}", "33;1", enabled),
                            " ",
                            color(f"{delta_s:<12}", delta_code(delta), enabled),
                            " ",
                            color(f"{str(row.get('tensor_count', '-')):<7}", "36;1", enabled),
                            " ",
                            color(verdict, verdict_code, enabled),
                        ]
                    ),
                )
            )
    if not rows:
        return [f"No {phase} ablation results yet."]
    lines = [
        color(f"{'group':<12} {'quant':<7} {'ppl':<12} {'delta':<12} {'tensors':<7} verdict", "90;1", enabled),
        color("─" * 92, "90", enabled),
    ]
    lines.extend(line for _mtime, line in sorted(rows, key=lambda item: item[0]))
    return lines


def classic_ablation_candidate_rows(root: Path, phase: str) -> list[dict[str, Any]]:
    phase_root = root / phase
    rows: list[dict[str, Any]] = []
    if not phase_root.exists():
        return rows
    for path in phase_root.glob("*/" + CANDIDATE_FILES[0]):
        for row in read_jsonl(path):
            rows.append({**row, "group": row.get("tensor") or path.parent.name, "run_dir": str(path.parent)})
    return rows


def selected_forward_survivor_groups(root: Path, max_regression_pct: float = 2.0) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in classic_ablation_candidate_rows(root, "forward"):
        baseline = row.get("baseline_ppl")
        delta = row.get("delta")
        try:
            pct_delta = (float(delta) / float(baseline)) * 100.0 if baseline else 0.0
        except (TypeError, ValueError):
            continue
        if pct_delta <= max_regression_pct:
            selected.append({**row, "pct_delta": pct_delta})
    return sorted(selected, key=lambda row: str(row.get("group") or ""))


def write_forward_survivor_overrides(root: Path, output: Path, max_regression_pct: float = 2.0, target_type: str = "q2_K") -> list[dict[str, Any]]:
    selected = selected_forward_survivor_groups(root, max_regression_pct=max_regression_pct)
    overrides: dict[str, str] = {}
    for row in selected:
        type_file = row.get("type_file")
        if not type_file:
            continue
        for tensor, qtype in read_tensor_type_map(Path(str(type_file))).items():
            if normalize_quant_type_name(qtype) == normalize_quant_type_name(target_type):
                overrides[tensor] = target_type
    lines = [tensor_type_line(tensor, qtype) for tensor, qtype in sorted(overrides.items())]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return selected


def legacy_plan_payload(run_dir: Path | None = None, source_gguf: str | None = None, output_dir: str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if run_dir:
        state = read_json(run_dir / "state.json", {})
        manifest = read_json(run_dir / "manifest.json", {})
    audit = golden_cow_audit(state, manifest) if state else {}
    source = source_gguf or manifest.get("source_gguf")
    out = output_dir or (str((run_dir or Path.cwd()) / "legacy_gated") if run_dir else "legacy_gated")
    benchmark_dir = str(Path(out) / "benchmark_results")
    phases = [
        {"name": "archive-current-run", "purpose": "preserve state, candidates, decisions, checkpoints, and frozen GGUFs before rollback"},
        {
            "name": "build-lower-quant-baseline",
            "purpose": "use the F16 GGUF only as the llama-quantize source; serve and benchmark the produced lower quant output",
        },
        {"name": "baseline-full-record", "purpose": "record quant output size, type distribution, PPL, speed, HumanEval+, ARC, HellaSwag, MMLU-Redux"},
        {"name": "group-ablation", "purpose": "test groups/layer clusters before per-tensor work"},
        {"name": "golden-cow-protection", "purpose": "force floors for early blocks, attention, output, embeddings, norms, and fragile MLP groups"},
        {"name": "interaction-check", "purpose": "test additive/layer-cluster interactions before accepting a bundle"},
        {"name": "benchmark-gate", "purpose": "reject PPL wins that regress benchmark smoke/full suite beyond threshold"},
        {"name": "final-comparison", "purpose": "publish only if size, TPS, PPL, and benchmark scores justify the variant"},
    ]
    commands = {
        "current_watch": None if not run_dir else f"cerebellum watch {shlex.quote(str(run_dir))}",
        "build_quant_baseline": None
        if not source
        else (
            "llama-quantize --allow-requantize --tensor-type-file tensor_types.txt "
            f"{shlex.quote(str(source))} {shlex.quote(str(Path(out) / 'baseline-or-candidate.gguf'))} Q4_K_M"
        ),
        "serve_quant_output": f"llama-server -m {shlex.quote(str(Path(out) / 'baseline-or-candidate.gguf'))} -ngl 99 --parallel 4 -c 24576 --jinja",
        "benchmark_plan": f"cerebellum benchmark-plan --suite release-local --model MODEL --results-dir {shlex.quote(benchmark_dir)}",
        "type_compare": None if not source else "cerebellum compare-gguf-types BASE.gguf CANDIDATE.gguf --json",
        "protected_pipeline": None
        if not source
        else (
            "cerebellum pipeline-plan "
            f"--source-gguf {shlex.quote(str(source))} --output-dir {shlex.quote(str(out))} "
            "--task-profile legacy-gated --benchmark-suite release-local --low-space"
        ),
    }
    return {
        "schema": "cerebellum.legacy_gated_plan.v1",
        "run_dir": str(run_dir) if run_dir else None,
        "source_gguf": source,
        "output_dir": out,
        "profile": TASK_PROFILES["legacy-gated"],
        "protected_groups": LEGACY_GATED_GROUPS,
        "phases": phases,
        "commands": commands,
        "golden_cow_audit": audit,
    }


def legacy_plan_markdown(plan: dict[str, Any]) -> str:
    parts = [
        "# Cerebellum Legacy-Gated Plan",
        "",
        f"run: `{plan.get('run_dir') or '-'}`",
        f"source: `{plan.get('source_gguf') or '-'}`",
        f"output: `{plan.get('output_dir')}`",
        "",
        "## Why",
        "",
        str(plan["profile"]["note"]),
        "",
        "## Phases",
        "",
        markdown_table(["Phase", "Purpose"], [[row["name"], row["purpose"]] for row in plan["phases"]]),
        "",
        "## Protected Groups",
        "",
        markdown_table(
            ["Group", "Floor", "Patterns", "Why"],
            [[row["name"], row["default_floor"], ", ".join(row["patterns"]), row["why"]] for row in plan["protected_groups"]],
        ),
    ]
    audit = plan.get("golden_cow_audit") or {}
    if audit:
        parts.extend(["", "## Current Run Audit", ""])
        parts.extend(golden_cow_watch_lines(audit, limit=12))
    command_rows = [[key, str(value)] for key, value in (plan.get("commands") or {}).items() if value]
    if command_rows:
        parts.extend(["", "## Commands", "", markdown_table(["Command", "Value"], command_rows)])
    return "\n".join(parts) + "\n"


def legacy_plan_cmd(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir) if args.run_dir else None
    plan = legacy_plan_payload(run_dir, args.source_gguf, args.output_dir)
    if args.write:
        Path(args.write).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.write)
        return
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(legacy_plan_markdown(plan), end="")


def legacy_flow_payload(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_gguf)
    output_dir = Path(args.output_dir)
    run_dir = Path(args.run_dir) if args.run_dir else output_dir / "run"
    imatrix = Path(args.imatrix) if args.imatrix else output_dir / "imatrix.dat"
    model_label = slug(args.model_name or source.stem).lower()
    baseline_gguf = output_dir / f"{model_label}-{args.base_type.lower()}-baseline.gguf"
    candidate_gguf = output_dir / f"{model_label}-cerebellum.gguf"
    benchmark_dir = output_dir / "benchmark_results"
    manifest_path = Path(args.write) if args.write else output_dir / "legacy_flow.json"
    survivability_root = output_dir / "survivability"
    selected_targets = output_dir / "selected_survivability_targets.json"
    target_tensor_file = output_dir / "target_tensors.txt"
    selected_override_file = output_dir / "selected_survivability_overrides.txt"
    tensor_map = run_dir / "artifacts" / "final_types.txt"
    quant_base = [args.quantize_bin, "--allow-requantize"]
    bool_flag(quant_base, "--pure", args.pure_quant)
    tensor_map_quant_base = [args.quantize_bin, "--allow-requantize"]
    common_run = [
        "cerebellum",
        "run",
        "--source-gguf",
        str(source),
        "--imatrix",
        str(imatrix),
        "--profile",
        args.profile,
        "--metric",
        args.metric,
        "--base-type",
        args.base_type,
        "--start-type",
        args.start_type,
        "--quantize-bin",
        args.quantize_bin,
        "--perplexity-bin",
        args.perplexity_bin,
        "--gpu-layers",
        str(args.gpu_layers),
        "--ctx-size",
        str(args.ctx_size),
        "--scratch-root",
        str(args.scratch_root or output_dir / "scratch"),
    ]
    optional_flag(common_run, "--chunks", args.chunks)
    optional_flag(common_run, "--distrobox", args.distrobox)
    bool_flag(common_run, "--low-space", args.low_space)
    bool_flag(common_run, "--serial-candidates", args.serial_candidates)
    bool_flag(common_run, "--pure-quant", args.pure_quant)
    group_scan_common = [
        "cerebellum",
        "group-scan",
        "--source-gguf",
        str(source),
        "--corpus",
        str(args.corpus or "CORPUS"),
        "--imatrix",
        str(imatrix),
        "--profile",
        args.profile,
        "--family",
        str(args.family or ""),
        "--model-name",
        model_label,
        "--source-name",
        str(args.source_name or ""),
        "--base-type",
        args.base_type,
        "--start-type",
        args.start_type,
        "--target-type",
        args.survivability_target_type,
        "--quantize-bin",
        args.quantize_bin,
        "--perplexity-bin",
        args.perplexity_bin,
        "--gpu-layers",
        str(args.gpu_layers),
        "--ctx-size",
        str(args.ctx_size),
        "--hard-free-floor-gb",
        str(args.hard_free_floor_gb),
        "--min-free-gb",
        str(args.min_free_gb),
    ]
    optional_flag(group_scan_common, "--chunks", args.chunks)
    optional_flag(group_scan_common, "--distrobox", args.distrobox)
    optional_flag(group_scan_common, "--baseline-ppl", args.baseline_ppl)
    survivability_groups = []
    for group in LEGACY_GATED_GROUPS:
        if group.get("scan") is False:
            continue
        patterns = group.get("patterns") or []
        group_run_dir = survivability_root / slug(group["name"]).lower()
        tensor_regex = "|".join(re.escape(str(pattern)) for pattern in patterns)
        survivability_groups.append(
            {
                "name": group["name"],
                "run_dir": str(group_run_dir),
                "floor": group["default_floor"],
                "patterns": patterns,
                "selection_rule": (
                    "selected for the classic Cerebellum candidate when Q2_K group crush stays under "
                    f"{args.max_regression_pct:g}% PPL regression and the benchmark smoke gate does not regress"
                ),
                "command_template": shell_join(
                    [
                        *group_scan_common,
                        "--run-dir",
                        group_run_dir,
                        "--group-name",
                        group["name"],
                        "--tensor-regex",
                        tensor_regex,
                    ]
                ),
            }
        )
    phases = [
        {
            "name": "scan",
            "status": "planned",
            "purpose": "build imatrix and inventory quantizable tensors before any destructive candidate work",
            "command": shell_join(["cerebellum", "imatrix", "--model", source, "--output", imatrix]),
            "outputs": [str(imatrix)],
        },
        {
            "name": "baseline",
            "status": "planned",
            "purpose": "build and benchmark the lower-quant baseline from the F16 source",
            "command": shell_join([*quant_base, "--imatrix", imatrix, str(source), baseline_gguf, args.base_type]),
            "outputs": [str(baseline_gguf)],
        },
        {
            "name": "survivability-scan",
            "status": "planned",
            "purpose": "forward ablation: test each whole tensor group at Q2_K from the baseline",
            "groups": survivability_groups,
            "outputs": [str(survivability_root), str(selected_targets)],
        },
        {
            "name": "target-selection",
            "status": "planned",
            "purpose": "write approved group-survivor overrides plus an optional tensor-name list for targeted refinement",
            "command": shell_join(
                [
                    "cerebellum",
                    "ablation-analyze",
                    str(survivability_root),
                    "--baseline-ppl",
                    args.baseline_ppl or "BASELINE_PPL",
                    "--target-type",
                    args.survivability_target_type,
                    "--json-output",
                    selected_targets,
                    "--output",
                    selected_override_file,
                    "--tensor-output",
                    target_tensor_file,
                ]
            ),
            "outputs": [str(selected_targets), str(selected_override_file), str(target_tensor_file)],
        },
        {
            "name": "build-v1-stacked-q2",
            "status": "planned",
            "purpose": "build the fully stacked v1 candidate from all forward-survivor Q2 group overrides",
            "command": shell_join([*tensor_map_quant_base, "--imatrix", imatrix, "--tensor-type-file", selected_override_file, str(source), output_dir / f"{model_label}-cerebellum-v1-stacked.gguf", args.base_type]),
            "outputs": [str(output_dir / f"{model_label}-cerebellum-v1-stacked.gguf")],
        },
        {
            "name": "reverse-ablation",
            "status": "planned",
            "purpose": "reverse ablation: from stacked v1, restore each selected group to the Gemma Q4 floor and keep Q2 only when restoration fails to improve quality",
            "command": "planned group-scan reverse mode: use selected_survivability_overrides.txt as the base map, restore one group to q4_K, run PPL, then update the final override map",
            "outputs": [str(output_dir / "reverse_survivability_results.json"), str(output_dir / "final_group_overrides.txt")],
        },
    ]
    if args.with_targeted_hillstep:
        phases.append(
            {
                "name": "targeted-hillstep",
                "status": "optional",
                "purpose": "optional refinement only inside selected survivor targets; not the default release recipe",
                "command": shell_join([*common_run, "--run-dir", run_dir, "--levels", args.levels, "--tensor-file", target_tensor_file]),
                "outputs": [str(run_dir / "state.json"), str(tensor_map)],
            }
        )
        candidate_tensor_map = tensor_map
        candidate_purpose = "build final candidate GGUF from exact targeted-hillstep tensor map"
    else:
        candidate_tensor_map = output_dir / "final_group_overrides.txt"
        candidate_purpose = "build classic Cerebellum candidate GGUF from forward+reverse group ablation overrides"
    phases.extend(
        [
        {
            "name": "build-candidate",
            "status": "planned",
            "purpose": candidate_purpose,
            "command": shell_join(
                [
                    *tensor_map_quant_base,
                    "--imatrix",
                    imatrix,
                    "--tensor-type-file",
                    candidate_tensor_map,
                    str(source),
                    candidate_gguf,
                    args.base_type,
                ]
            ),
            "outputs": [str(candidate_gguf)],
        },
        {
            "name": "benchmark-gate",
            "status": "planned",
            "purpose": "gate against baseline on size, TPS, PPL, HumanEval+, ARC, HellaSwag, MMLU-Redux, and audit artifacts",
            "command": shell_join(
                [
                    "cerebellum",
                    "benchmark-run",
                    "--suite",
                    args.benchmark_suite,
                    "--model",
                    model_label,
                    "--results-dir",
                    benchmark_dir,
                    "--execute",
                    "--postprocess",
                    "--require-complete",
                ]
            ),
            "outputs": [str(benchmark_dir)],
        },
        ]
    )
    return {
        "schema": "cerebellum.legacy_flow.v1",
        "mode": "classic group-first Cerebellum",
        "source_gguf": str(source),
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "imatrix": str(imatrix),
        "baseline_gguf": str(baseline_gguf),
        "candidate_gguf": str(candidate_gguf),
        "manifest_path": str(manifest_path),
        "watch": {
            "private": shell_join(["cerebellum", "watch", run_dir]),
            "public": shell_join(["cerebellum", "watch", "--public", run_dir]),
            "state_model": "same run state, event log, measurements, locks, ETA, and locked-layer map as normal Cerebellum watch",
        },
        "orchestration": {
            "quant_ppl_overlap": "enabled in HillStepper: quant_worker and ppl_worker run in tandem; normal mode fills per-tensor candidates until the disk floor blocks another GGUF",
            "low_space_mode": "serializes candidate testing with queue depth 1 and prunes measured candidate GGUFs immediately",
            "cleanup": "durable state/checkpoints stay; measured non-winning candidate GGUFs are pruned unless --keep-losers or --keep-measured-candidates is set",
            "resume": shell_join(["cerebellum", "resume", run_dir, "--low-space"] if args.low_space else ["cerebellum", "resume", run_dir]),
            "queue": shell_join(["cerebellum", "queue", "add", "--kind", "pipeline", "--manifest", manifest_path]),
        },
        "protected_groups": LEGACY_GATED_GROUPS,
        "survivability": {
            "levels": args.survivability_levels,
            "target_type": args.survivability_target_type,
            "mode": "classic Q2_K no-commit group survivability scan",
            "commit_locks": False,
            "max_regression_pct": args.max_regression_pct,
            "selected_targets": str(selected_targets),
            "selected_override_file": str(selected_override_file),
            "target_tensor_file": str(target_tensor_file),
            "with_targeted_hillstep": args.with_targeted_hillstep,
            "pure_quant": args.pure_quant,
        },
        "phases": phases,
    }


def legacy_flow_markdown(plan: dict[str, Any]) -> str:
    rows = [[row["name"], row["status"], row["purpose"], row.get("command") or "-"] for row in plan["phases"]]
    group_rows = []
    for group in (next((row for row in plan["phases"] if row["name"] == "survivability-scan"), {}).get("groups") or []):
        group_rows.append([group["name"], group["floor"], ", ".join(group["patterns"]), group["command_template"]])
    parts = [
        "# Cerebellum Legacy Flow",
        "",
        f"source: `{plan['source_gguf']}`",
        f"run: `{plan['run_dir']}`",
        f"baseline: `{plan['baseline_gguf']}`",
        f"candidate: `{plan['candidate_gguf']}`",
        "",
        "## Watch",
        "",
        markdown_table(["View", "Command"], [["private", plan["watch"]["private"]], ["public", plan["watch"]["public"]]]),
        "",
        "## Orchestration",
        "",
        markdown_table(["Key", "Value"], [[key, str(value)] for key, value in plan["orchestration"].items()]),
        "",
        "## Phases",
        "",
        markdown_table(["Phase", "Status", "Purpose", "Command"], rows),
    ]
    if group_rows:
        parts.extend(["", "## Survivability Groups", "", markdown_table(["Group", "Floor", "Patterns", "Command"], group_rows)])
    return "\n".join(parts) + "\n"


def legacy_forward_group_args(args: argparse.Namespace, plan: dict[str, Any], group: dict[str, Any], run_dir: Path) -> argparse.Namespace:
    name = str(group["name"])
    patterns = [str(pattern) for pattern in group.get("patterns") or []]
    tensor_regex = "|".join(re.escape(pattern) for pattern in patterns)
    return argparse.Namespace(
        source_gguf=args.source_gguf,
        corpus=args.corpus,
        run_dir=str(run_dir),
        tensor_regex=tensor_regex,
        group_name=name,
        run_name=f"{slug(args.model_name or Path(args.source_gguf).stem)}-classic-forward-{slug(name)}-{args.survivability_target_type}",
        profile=args.profile,
        family=args.family,
        model_name=args.model_name,
        source_name=args.source_name,
        base_type=args.base_type,
        start_type=args.start_type,
        target_type=args.survivability_target_type,
        base_map=getattr(args, "base_map", None),
        baseline_ppl=args.baseline_ppl,
        imatrix=plan["imatrix"],
        quantize_bin=args.quantize_bin,
        perplexity_bin=args.perplexity_bin,
        gpu_layers=args.gpu_layers,
        ctx_size=args.ctx_size,
        chunks=args.chunks,
        distrobox=args.distrobox,
        quant_timeout=args.quant_timeout,
        ppl_timeout=args.ppl_timeout,
        min_free_gb=args.min_free_gb,
        hard_free_floor_gb=args.hard_free_floor_gb,
        token_embedding_type=args.token_embedding_type,
        preview_limit=20,
        prune_candidate=not args.keep_candidates,
        dry_run=False,
    )


def legacy_forward_pending_groups(args: argparse.Namespace, plan: dict[str, Any], forward_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for group in LEGACY_GATED_GROUPS:
        if group.get("scan") is False:
            continue
        name = str(group["name"])
        run_dir = forward_root / slug(name).lower()
        state = read_json(run_dir / "state.json", {})
        if state.get("run_status") == "complete":
            skipped.append({"group": name, "run_dir": str(run_dir), "reason": "already complete"})
            continue
        pending.append({"group": group, "name": name, "run_dir": run_dir, "args": legacy_forward_group_args(args, plan, group, run_dir)})
    return pending, skipped


def legacy_forward_run_serial(args: argparse.Namespace, plan: dict[str, Any], forward_root: Path) -> dict[str, Any]:
    pending, skipped = legacy_forward_pending_groups(args, plan, forward_root)
    executed: list[dict[str, Any]] = []
    for item in pending:
        group_scan_cmd(item["args"])
        result_state = read_json(item["run_dir"] / "state.json", {})
        executed.append({"group": item["name"], "run_dir": str(item["run_dir"]), "status": result_state.get("run_status")})
    return {"executed": executed, "skipped": skipped, "mode": "serial"}


def legacy_forward_run_pipelined(args: argparse.Namespace, plan: dict[str, Any], forward_root: Path) -> dict[str, Any]:
    pending, skipped = legacy_forward_pending_groups(args, plan, forward_root)
    executed: list[dict[str, Any]] = []
    prepared = [group_scan_prepare(item["args"]) for item in pending]
    if not prepared:
        return {"executed": executed, "skipped": skipped, "mode": "pipelined"}

    def start_quant(job: dict[str, Any]) -> dict[str, Any]:
        results: queue.Queue[tuple[dict[str, Any], int, str, float] | BaseException] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                q_rc, q_out, q_seconds = group_scan_run_quant(job)
                results.put((job, q_rc, q_out, q_seconds))
            except BaseException as exc:  # noqa: BLE001 - propagate background quant failure
                results.put(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return {"job": job, "thread": thread, "results": results}

    def finish_quant(task: dict[str, Any]) -> tuple[dict[str, Any], float]:
        task["thread"].join()
        result = task["results"].get()
        if isinstance(result, BaseException):
            raise result
        job, q_rc, q_out, q_seconds = result
        if not group_scan_finalize_quant(job, q_rc, q_out, q_seconds):
            raise SystemExit(f"group scan quantize failed for {job['group_name']}")
        return job, q_seconds

    current_task: dict[str, Any] | None = start_quant(prepared[0])
    next_index = 1
    while current_task is not None:
        current_job, current_q_seconds = finish_quant(current_task)
        next_task: dict[str, Any] | None = None
        if next_index < len(prepared):
            try:
                next_job = prepared[next_index]
                group_scan_assert_disk_floor(next_job["run_dir"], next_job.get("source", Path(next_job["args"].source_gguf)), next_job["args"], phase="pipeline_next_quant")
                current_job["events"].write("pipeline_overlap", tensor=current_job["group_name"], level=current_job["args"].target_type, next_tensor=prepared[next_index]["group_name"])
                next_task = start_quant(next_job)
                next_index += 1
            except SystemExit:
                next_task = None
        result = group_scan_run_ppl(current_job, current_q_seconds)
        executed.append({"group": current_job["group_name"], "run_dir": str(current_job["run_dir"]), "status": result.get("status"), "ppl": result.get("ppl"), "delta": result.get("delta")})
        if next_task is None and next_index < len(prepared):
            next_task = start_quant(prepared[next_index])
            next_index += 1
        current_task = next_task
    return {"executed": executed, "skipped": skipped, "mode": "pipelined"}


def legacy_flow_execute_forward(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    if not args.corpus:
        raise SystemExit("--execute-forward requires --corpus")
    if args.baseline_ppl is None:
        raise SystemExit("--execute-forward requires --baseline-ppl")
    root = Path(plan["output_dir"])
    forward_root = root / "forward"
    forward_root.mkdir(parents=True, exist_ok=True)
    result = legacy_forward_run_serial(args, plan, forward_root) if args.serial_candidates else legacy_forward_run_pipelined(args, plan, forward_root)
    return {
        "schema": "cerebellum.legacy_flow_execute_forward.v1",
        "output_dir": str(root),
        "forward_root": str(forward_root),
        **result,
    }


def legacy_flow_cmd(args: argparse.Namespace) -> None:
    plan = legacy_flow_payload(args)
    if args.execute_forward:
        manifest = Path(args.write) if args.write else Path(plan["manifest_path"])
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = legacy_flow_execute_forward(args, plan)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.write:
        output = Path(args.write)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.write)
        return
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(legacy_flow_markdown(plan), end="")


def group_scan_tensor_names(source_gguf: Path, tensor_regex: str) -> list[str]:
    try:
        from gguf import GGUFReader
    except Exception as exc:
        raise SystemExit(f"cannot import gguf reader: {exc}") from exc
    pattern = re.compile(tensor_regex)
    reader = GGUFReader(str(source_gguf))
    names = [t.name for t in reader.tensors]
    return sorted(name for name in quantizable_tensor_names(names) if pattern.search(name))


def group_scan_write_types(source: Path, path: Path, start_type: str, overrides: dict[str, str]) -> None:
    write_tensor_types_map(source, overrides, start_type, path)


def group_scan_base_overrides(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    return read_tensor_type_map(Path(path))


def group_scan_quant_cmd(args: argparse.Namespace, type_file: Path, outfile: Path) -> list[str]:
    cmd = [args.quantize_bin, "--allow-requantize"]
    if args.imatrix:
        cmd.extend(["--imatrix", str(args.imatrix)])
    if args.token_embedding_type:
        cmd.extend(["--token-embedding-type", str(args.token_embedding_type)])
    cmd.extend(["--tensor-type-file", str(type_file), str(args.source_gguf), str(outfile), args.base_type])
    return cmd


def group_scan_ppl_cmd(args: argparse.Namespace, model: Path) -> list[str]:
    cmd = [args.perplexity_bin, "--model", str(model), "--ctx-size", str(args.ctx_size), "-f", str(args.corpus), "-ngl", str(args.gpu_layers)]
    if args.chunks is not None:
        cmd.extend(["--chunks", str(args.chunks)])
    return cmd


def group_scan_prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_gguf)
    run_dir = Path(args.run_dir)
    group_name = args.group_name or slug(args.tensor_regex)
    run_id = args.run_name or f"cerebellum-group-scan-{slug(group_name)}"
    artifacts = run_dir / "artifacts"
    type_file = run_dir / "group_types.txt"
    candidate = artifacts / f"{slug(group_name)}-{args.target_type}.gguf"
    tmp_candidate = candidate.with_suffix(candidate.suffix + ".tmp")
    tensor_names = group_scan_tensor_names(source, args.tensor_regex)
    if not tensor_names:
        raise SystemExit(f"no quantizable tensors matched --tensor-regex {args.tensor_regex!r}")
    overrides = group_scan_base_overrides(getattr(args, "base_map", None))
    overrides.update({name: args.target_type for name in tensor_names})
    group_scan_write_types(source, type_file, args.start_type, overrides)
    manifest = {
        "schema": "cerebellum.group_scan.v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_gguf": str(source),
        "corpus": str(args.corpus),
        "model_family": args.family,
        "model_name": args.model_name,
        "source_name": args.source_name,
        "ppl_profile": args.profile,
        "group_name": group_name,
        "tensor_regex": args.tensor_regex,
        "tensor_count": len(tensor_names),
        "base_type": args.base_type,
        "start_type": args.start_type,
        "levels": [args.target_type],
        "target_type": args.target_type,
        "base_map": str(args.base_map) if getattr(args, "base_map", None) else None,
        "commit_locks": False,
        "measurement_mode": "classic group no-commit scan",
        "imatrix": str(args.imatrix) if args.imatrix else None,
        "quantize_bin": args.quantize_bin,
        "perplexity_bin": args.perplexity_bin,
        "gpu_layers": args.gpu_layers,
        "ctx_size": args.ctx_size,
        "chunks": args.chunks,
        "distrobox": args.distrobox,
        "hard_free_floor_gb": args.hard_free_floor_gb,
        "min_free_gb": args.min_free_gb,
        "pure_quant_effective": False,
        "type_file": str(type_file),
        "candidate_gguf": str(candidate),
    }
    state = {
        "schema": "cerebellum.group_scan_state.v1",
        "run_id": run_id,
        "run_status": "planned" if args.dry_run else "running",
        "pid": os.getpid(),
        "model_family": args.family,
        "model_name": args.model_name,
        "source_name": args.source_name,
        "current_ppl": args.baseline_ppl,
        "baseline_ppl": args.baseline_ppl,
        "current_tensor": group_name,
        "current_level": args.target_type,
        "locked": {},
        "tested": [],
        "levels": [args.target_type],
        "started_at": utc_now(),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "state.json", state)
    events = EventLog(run_dir / EVENT_FILES[0], run_id)
    candidates = EventLog(run_dir / CANDIDATE_FILES[0], run_id)
    return {
        "args": args,
        "source": source,
        "run_dir": run_dir,
        "group_name": group_name,
        "run_id": run_id,
        "artifacts": artifacts,
        "type_file": type_file,
        "candidate": candidate,
        "tmp_candidate": tmp_candidate,
        "tensor_names": tensor_names,
        "manifest": manifest,
        "state": state,
        "events": events,
        "candidates": candidates,
    }


def group_scan_required_free_gb(source: Path, args: argparse.Namespace) -> float:
    estimated_candidate_gb = max(bytes_to_gb(path_size(source)) * 0.45, 1.0)
    return max(args.min_free_gb, args.hard_free_floor_gb + estimated_candidate_gb)


def group_scan_assert_disk_floor(run_dir: Path, source: Path, args: argparse.Namespace, phase: str = "quant") -> None:
    free_gb = disk_free_gb(run_dir)
    required_free = group_scan_required_free_gb(source, args)
    if free_gb < required_free:
        raise SystemExit(f"not enough free space before {phase}: {free_gb:.1f} GiB free, need at least {required_free:.1f} GiB")


def group_scan_run_quant(job: dict[str, Any]) -> tuple[int, str, float]:
    args = job["args"]
    group_name = job["group_name"]
    artifacts = job["artifacts"]
    type_file = job["type_file"]
    candidate = job["candidate"]
    tmp_candidate = job["tmp_candidate"]
    events = job["events"]
    group_scan_assert_disk_floor(job["run_dir"], job["source"], args, phase="quant")
    artifacts.mkdir(parents=True, exist_ok=True)
    if tmp_candidate.exists():
        tmp_candidate.unlink()
    events.write("group_scan_start", tensor=group_name, level=args.target_type, tensors=len(job["tensor_names"]))
    events.write("quant_start", tensor=group_name, level=args.target_type, tmp_output=str(tmp_candidate), output=str(candidate))
    try:
        return run_external(
            group_scan_quant_cmd(args, type_file, tmp_candidate),
            args.quant_timeout,
            args.distrobox,
            heartbeat=lambda elapsed, pid: events.write("quant_heartbeat", tensor=group_name, level=args.target_type, elapsed_seconds=elapsed, child_pid=pid, tmp_output=str(tmp_candidate), size_bytes=path_size(tmp_candidate)),
        )
    except subprocess.TimeoutExpired as exc:
        return 124, str(exc), float(args.quant_timeout)


def group_scan_finalize_quant(job: dict[str, Any], q_rc: int, q_out: str, q_seconds: float) -> bool:
    args = job["args"]
    group_name = job["group_name"]
    candidate = job["candidate"]
    tmp_candidate = job["tmp_candidate"]
    events = job["events"]
    state = job["state"]
    if q_rc == 0 and tmp_candidate.exists() and path_size(tmp_candidate) > 0:
        os.replace(tmp_candidate, candidate)
    elif tmp_candidate.exists():
        tmp_candidate.unlink()
    events.write("quant_finish", tensor=group_name, level=args.target_type, returncode=q_rc, seconds=q_seconds, size_bytes=path_size(candidate), output_tail=q_out[-2000:])
    if q_rc != 0 or not candidate.exists():
        state.update({"run_status": "failed", "finished_at": utc_now(), "failure": "quantize failed"})
        atomic_write_json(job["run_dir"] / "state.json", state)
        return False
    return True


def group_scan_run_ppl(job: dict[str, Any], q_seconds: float) -> dict[str, Any]:
    args = job["args"]
    group_name = job["group_name"]
    candidate = job["candidate"]
    events = job["events"]
    candidates = job["candidates"]
    state = job["state"]
    events.write("ppl_start", tensor=group_name, level=args.target_type, model=str(candidate))
    try:
        p_rc, p_out, p_seconds = run_external(
            group_scan_ppl_cmd(args, candidate),
            args.ppl_timeout,
            args.distrobox,
            heartbeat=lambda elapsed, pid: events.write("ppl_heartbeat", tensor=group_name, level=args.target_type, elapsed_seconds=elapsed, child_pid=pid, model=str(candidate)),
        )
    except subprocess.TimeoutExpired as exc:
        p_rc, p_out, p_seconds = 124, str(exc), float(args.ppl_timeout)
    ppl, err = parse_ppl(p_out)
    delta = (ppl - args.baseline_ppl) if ppl is not None and args.baseline_ppl is not None else None
    events.write("ppl_finish", tensor=group_name, level=args.target_type, returncode=p_rc, seconds=p_seconds, ppl=ppl, ppl_error=err, delta=delta, output_tail=p_out[-2000:])
    candidates.write("candidate", tensor=group_name, level=args.target_type, tensor_count=len(job["tensor_names"]), baseline_ppl=args.baseline_ppl, ppl=ppl, ppl_error=err, delta=delta, quant_seconds=q_seconds, ppl_seconds=p_seconds, size_bytes=path_size(candidate), status="done" if p_rc == 0 and ppl is not None else "failed", type_file=str(job["type_file"]), matched_tensors=job["tensor_names"][: args.preview_limit])
    state.update({"run_status": "complete" if p_rc == 0 and ppl is not None else "failed", "finished_at": utc_now(), "last_tensor": group_name, "tested": [{"tensor": group_name, "winner": args.target_type, "ppl": ppl, "delta": delta, "tensor_count": len(job["tensor_names"]), "type_file": str(job["type_file"])}]})
    atomic_write_json(job["run_dir"] / "state.json", state)
    if args.prune_candidate and candidate.exists():
        candidate.unlink()
    return {"run_dir": str(job["run_dir"]), "group": group_name, "target_type": args.target_type, "tensor_count": len(job["tensor_names"]), "baseline_ppl": args.baseline_ppl, "ppl": ppl, "ppl_error": err, "delta": delta, "status": state["run_status"], "type_file": str(job["type_file"])}


def group_scan_cmd(args: argparse.Namespace) -> None:
    job = group_scan_prepare(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    **job["manifest"],
                    "matched_tensors": job["tensor_names"][: args.preview_limit],
                    "quant_command": shell_join(group_scan_quant_cmd(args, job["type_file"], job["candidate"])),
                    "ppl_command": shell_join(group_scan_ppl_cmd(args, job["candidate"])),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    q_rc, q_out, q_seconds = group_scan_run_quant(job)
    if not group_scan_finalize_quant(job, q_rc, q_out, q_seconds):
        raise SystemExit("group scan quantize failed")
    print(json.dumps(group_scan_run_ppl(job, q_seconds), indent=2, sort_keys=True))


def sparse_replay_default_plan(source_gguf: Path, limit: int | None = None) -> list[str]:
    names = group_scan_tensor_names(source_gguf, r".*")
    layer_re = re.compile(r"^blk\.(\d+)\.")
    layers = sorted({int(m.group(1)) for name in names if (m := layer_re.match(name))})
    selected: list[str] = []
    if layers:
        anchors = sorted({0, 1, 2, layers[len(layers) // 4], layers[len(layers) // 2], layers[-3], layers[-2], layers[-1]})
        components = ["attn_q", "attn_v", "attn_output", "ffn_gate", "ffn_down"]
        available = set(names)
        for layer in anchors:
            for component in components:
                tensor = f"blk.{layer}.{component}.weight"
                if tensor in available:
                    selected.append(tensor)
    if not selected:
        selected = names
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def sparse_replay_load_plan(args: argparse.Namespace) -> list[str]:
    if args.probe_plan:
        path = Path(args.probe_plan)
        if path.suffix.lower() == ".json":
            payload = read_json(path, {})
            rows = payload.get("tensors") if isinstance(payload, dict) else payload
            tensors = [str(row.get("name") or row.get("tensor") or row) for row in rows]
        else:
            tensors = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    else:
        tensors = sparse_replay_default_plan(Path(args.source_gguf), args.max_probes)
    if args.tensor_regex:
        pattern = re.compile(args.tensor_regex)
        tensors = [name for name in tensors if pattern.search(name)]
    if not tensors:
        raise SystemExit("sparse replay probe plan is empty")
    return tensors


def sparse_replay_write_single_type(path: Path, tensor: str, target_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{tensor_type_line(tensor, target_type)}\n", encoding="utf-8")


def sparse_replay_quant_cmd(args: argparse.Namespace, type_file: Path, outfile: Path, base_type: str | None = None) -> list[str]:
    cmd = [args.quantize_bin, "--allow-requantize"]
    if args.imatrix:
        cmd.extend(["--imatrix", str(args.imatrix)])
    if args.token_embedding_type:
        cmd.extend(["--token-embedding-type", str(args.token_embedding_type)])
    cmd.extend(["--tensor-type-file", str(type_file), str(args.source_gguf), str(outfile), base_type or args.probe_base_type])
    return cmd


def sparse_replay_ppl_cmd(args: argparse.Namespace, model: Path) -> list[str]:
    return group_scan_ppl_cmd(args, model)


def sparse_replay_ablation_payload(state: dict[str, Any], tensors: list[str]) -> dict[str, Any]:
    return {
        "schema": "cerebellum.ablation_results.v1",
        "baseline_ppl": state.get("baseline_ppl"),
        "baseline_error": state.get("baseline_ppl_error"),
        "plan": {"tensors": [{"name": name, "hf_name": name} for name in tensors]},
        "results": [
            {
                "tensor": row.get("tensor"),
                "hf_name": row.get("tensor"),
                "ppl": row.get("ppl"),
                "ppl_error": row.get("ppl_error"),
                "delta": row.get("delta"),
                "status": row.get("status", "done"),
            }
            for row in state.get("tested", [])
        ],
    }


def sparse_replay_prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_gguf)
    run_dir = Path(args.run_dir)
    tensors = sparse_replay_load_plan(args)
    run_id = args.run_name or f"cerebellum-sparse-replay-{slug(args.model_name or source.stem)}"
    manifest = {
        "schema": "cerebellum.sparse_replay.v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_gguf": str(source),
        "corpus": str(args.corpus),
        "model_family": args.family,
        "model_name": args.model_name or source.stem,
        "source_name": args.source_name or source.stem,
        "ppl_profile": args.profile,
        "base_type": args.probe_base_type,
        "start_type": args.start_type,
        "levels": [args.target_type],
        "target_type": args.target_type,
        "budget_gb": args.budget_gb,
        "final_base_type": args.final_base_type,
        "commit_locks": False,
        "measurement_mode": "qwen36-27b-v4 sparse replay",
        "imatrix": str(args.imatrix) if args.imatrix else None,
        "quantize_bin": args.quantize_bin,
        "perplexity_bin": args.perplexity_bin,
        "gpu_layers": args.gpu_layers,
        "ctx_size": args.ctx_size,
        "chunks": args.chunks,
        "distrobox": args.distrobox,
        "tensor_count": len(tensors),
    }
    state = {
        "schema": "cerebellum.sparse_replay_state.v1",
        "run_id": run_id,
        "run_status": "planned" if args.dry_run else "running",
        "pid": os.getpid(),
        "model_family": manifest["model_family"],
        "model_name": manifest["model_name"],
        "source_name": manifest["source_name"],
        "source_gguf": str(source),
        "corpus": str(args.corpus),
        "ppl_profile": args.profile,
        "base_type": args.probe_base_type,
        "start_type": args.start_type,
        "levels": [args.target_type],
        "locked": {},
        "tested": [],
        "current_ppl": args.baseline_ppl,
        "baseline_ppl": args.baseline_ppl,
        "baseline_path": str(args.baseline_gguf) if args.baseline_gguf else None,
        "last_tensor": None,
        "totals": {},
        "started_at": utc_now(),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "state.json", state)
    atomic_write_json(run_dir / "ablation_plan.json", {"schema": "cerebellum.sparse_replay_plan.v1", "tensors": [{"name": name, "hf_name": name} for name in tensors]})
    return {"args": args, "source": source, "run_dir": run_dir, "run_id": run_id, "tensors": tensors, "manifest": manifest, "state": state, "events": EventLog(run_dir / EVENT_FILES[0], run_id), "candidates": EventLog(run_dir / CANDIDATE_FILES[0], run_id), "types_dir": run_dir / "tensor_types", "artifacts": run_dir / "artifacts", "final_dir": run_dir / "final"}


def sparse_replay_run_probe(job: dict[str, Any], tensor: str, index: int, total: int) -> None:
    args = job["args"]
    events = job["events"]
    candidate_log = job["candidates"]
    state = job["state"]
    safe = slug(tensor)
    type_file = job["types_dir"] / f"{index:04d}-{safe}-{args.target_type}.txt"
    candidate = job["artifacts"] / f"{index:04d}-{safe}-{args.target_type}.gguf"
    tmp_candidate = candidate.with_suffix(candidate.suffix + ".tmp")
    sparse_replay_write_single_type(type_file, tensor, args.target_type)
    state.update({"run_status": "running", "last_tensor": tensor, "current_tensor": tensor, "current_level": args.target_type})
    atomic_write_json(job["run_dir"] / "state.json", state)
    events.write("tensor_start", tensor=tensor, level=args.target_type, index=index, total=total, baseline_ppl=state.get("baseline_ppl"))
    events.write("quant_start", tensor=tensor, level=args.target_type, tmp_output=str(tmp_candidate), output=str(candidate), type_file=str(type_file))
    job["artifacts"].mkdir(parents=True, exist_ok=True)
    if tmp_candidate.exists():
        tmp_candidate.unlink()
    try:
        q_rc, q_out, q_seconds = run_external(sparse_replay_quant_cmd(args, type_file, tmp_candidate), args.quant_timeout, args.distrobox, heartbeat=lambda elapsed, pid: events.write("quant_heartbeat", tensor=tensor, level=args.target_type, elapsed_seconds=elapsed, child_pid=pid, tmp_output=str(tmp_candidate), size_bytes=path_size(tmp_candidate)))
    except subprocess.TimeoutExpired as exc:
        q_rc, q_out, q_seconds = 124, str(exc), float(args.quant_timeout)
    if q_rc == 0 and tmp_candidate.exists() and path_size(tmp_candidate) > 0:
        os.replace(tmp_candidate, candidate)
    elif tmp_candidate.exists():
        tmp_candidate.unlink()
    events.write("quant_finish", tensor=tensor, level=args.target_type, returncode=q_rc, seconds=q_seconds, size_bytes=path_size(candidate), output_tail=q_out[-2000:])
    ppl = err = delta = None
    p_rc = 1
    p_seconds = 0.0
    if q_rc == 0 and candidate.exists():
        events.write("ppl_start", tensor=tensor, level=args.target_type, model=str(candidate))
        try:
            p_rc, p_out, p_seconds = run_external(sparse_replay_ppl_cmd(args, candidate), args.ppl_timeout, args.distrobox, heartbeat=lambda elapsed, pid: events.write("ppl_heartbeat", tensor=tensor, level=args.target_type, elapsed_seconds=elapsed, child_pid=pid, model=str(candidate)))
        except subprocess.TimeoutExpired as exc:
            p_rc, p_out, p_seconds = 124, str(exc), float(args.ppl_timeout)
        ppl, err = parse_ppl(p_out)
        delta = (ppl - state["baseline_ppl"]) if ppl is not None and state.get("baseline_ppl") is not None else None
        events.write("ppl_finish", tensor=tensor, level=args.target_type, returncode=p_rc, seconds=p_seconds, ppl=ppl, ppl_error=err, delta=delta, output_tail=p_out[-2000:])
    status = "done" if q_rc == 0 and p_rc == 0 and ppl is not None else "failed"
    row = {"tensor": tensor, "winner": args.target_type, "level": args.target_type, "baseline_ppl": state.get("baseline_ppl"), "ppl": ppl, "ppl_error": err, "delta": delta, "quant_seconds": q_seconds, "ppl_seconds": p_seconds, "size_bytes": path_size(candidate), "status": status, "type_file": str(type_file)}
    candidate_log.write("candidate", **row)
    state.setdefault("tested", []).append(row)
    state["current_ppl"] = ppl if ppl is not None else state.get("current_ppl")
    state["totals"] = totals_for_kept_candidates(state["tested"], state["tested"])
    atomic_write_json(job["run_dir"] / "state.json", state)
    atomic_write_json(job["run_dir"] / "ablation_results.json", sparse_replay_ablation_payload(state, job["tensors"]))
    if args.prune_candidates and candidate.exists():
        candidate.unlink()
    if status != "done" and not args.continue_on_failure:
        state.update({"run_status": "failed", "finished_at": utc_now(), "failure": f"probe failed: {tensor}"})
        atomic_write_json(job["run_dir"] / "state.json", state)
        raise SystemExit(f"sparse replay probe failed: {tensor}")


def sparse_replay_allocate(job: dict[str, Any]) -> Path:
    args = job["args"]
    output = job["run_dir"] / "tensor_types_sparse_replay.txt"
    cmd = [sys.executable, "-m", "cerebellum.cerebellum", "--ablation", str(job["run_dir"] / "ablation_results.json"), "--plan", str(job["run_dir"] / "ablation_plan.json"), "--source-gguf", str(args.source_gguf), "--budget-gb", str(args.budget_gb), "--base-type", args.probe_base_type, "--output", str(output)]
    if args.imatrix:
        cmd.extend(["--imatrix", str(args.imatrix)])
    if args.quantize_bin:
        cmd.extend(["--quantize-bin", str(args.quantize_bin)])
    job["events"].write("allocate_start", budget_gb=args.budget_gb, output=str(output))
    rc, out, seconds = run_external(cmd, args.allocate_timeout, args.distrobox)
    job["events"].write("allocate_finish", returncode=rc, seconds=seconds, output_tail=out[-4000:], output=str(output))
    if rc != 0 or not output.exists():
        raise SystemExit("sparse replay allocation failed")
    return output


def sparse_replay_build_final(job: dict[str, Any], tensor_types: Path) -> Path:
    args = job["args"]
    job["final_dir"].mkdir(parents=True, exist_ok=True)
    output = Path(args.final_gguf) if args.final_gguf else job["final_dir"] / f"{slug(args.model_name or Path(args.source_gguf).stem)}-cerebellum-sparse-replay.gguf"
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    job["events"].write("final_quant_start", level=args.final_base_type, type_file=str(tensor_types), tmp_output=str(tmp_output), output=str(output))
    rc, out, seconds = run_external(sparse_replay_quant_cmd(args, tensor_types, tmp_output, base_type=args.final_base_type), args.quant_timeout, args.distrobox, heartbeat=lambda elapsed, pid: job["events"].write("final_quant_heartbeat", elapsed_seconds=elapsed, child_pid=pid, tmp_output=str(tmp_output), size_bytes=path_size(tmp_output)))
    if rc == 0 and tmp_output.exists() and path_size(tmp_output) > 0:
        os.replace(tmp_output, output)
    elif tmp_output.exists():
        tmp_output.unlink()
    job["events"].write("final_quant_finish", returncode=rc, seconds=seconds, size_bytes=path_size(output), output_tail=out[-4000:], output=str(output))
    if rc != 0 or not output.exists():
        raise SystemExit("sparse replay final quant failed")
    return output


def sparse_replay_measure_final(job: dict[str, Any], model: Path) -> dict[str, Any]:
    args = job["args"]
    events = job["events"]
    events.write("final_ppl_start", model=str(model))
    rc, out, seconds = run_external(sparse_replay_ppl_cmd(args, model), args.ppl_timeout, args.distrobox, heartbeat=lambda elapsed, pid: events.write("final_ppl_heartbeat", elapsed_seconds=elapsed, child_pid=pid, model=str(model)))
    ppl, err = parse_ppl(out)
    baseline = job["state"].get("baseline_ppl")
    delta = (ppl - baseline) if ppl is not None and baseline is not None else None
    events.write("final_ppl_finish", returncode=rc, seconds=seconds, ppl=ppl, ppl_error=err, delta=delta, output_tail=out[-4000:])
    return {"returncode": rc, "seconds": seconds, "ppl": ppl, "ppl_error": err, "delta": delta}


def sparse_replay_cmd(args: argparse.Namespace) -> None:
    job = sparse_replay_prepare(args)
    if args.dry_run:
        print(json.dumps({**job["manifest"], "tensors": job["tensors"], "watch": f"cerebellum watch {job['run_dir']}"}, indent=2, sort_keys=True))
        return
    events = job["events"]
    state = job["state"]
    events.write("run_start", tensors=len(job["tensors"]), baseline_ppl=state.get("baseline_ppl"), mode="qwen36-27b-v4 sparse replay")
    for idx, tensor in enumerate(job["tensors"], start=1):
        sparse_replay_run_probe(job, tensor, idx, len(job["tensors"]))
    tensor_types = sparse_replay_allocate(job)
    final = sparse_replay_build_final(job, tensor_types)
    final_ppl = sparse_replay_measure_final(job, final)
    state.update({"run_status": "complete" if final_ppl.get("returncode") == 0 and final_ppl.get("ppl") is not None else "failed", "finished_at": utc_now(), "final": {"gguf": str(final), "tensor_types": str(tensor_types), "size_bytes": path_size(final), **final_ppl}, "current_ppl": final_ppl.get("ppl") or state.get("current_ppl")})
    atomic_write_json(job["run_dir"] / "state.json", state)
    summary = {"schema": "cerebellum.sparse_replay_summary.v1", "run_dir": str(job["run_dir"]), "final": state["final"], "tested": len(state.get("tested", []))}
    atomic_write_json(job["run_dir"] / "sparse_replay_summary.json", summary)
    events.write("run_finish", status=state["run_status"], final_gguf=str(final), final_ppl=final_ppl.get("ppl"), final_delta=final_ppl.get("delta"))
    print(json.dumps(summary, indent=2, sort_keys=True))


def public_model_card_policy_payload() -> dict[str, Any]:
    sections = [
        "Model Summary",
        "Files",
        "Quick Start",
        "Recommended Runtime Settings",
        "Benchmarks",
        "Benchmark Artifacts",
        "Hardware Requirements",
        "Quantization Summary",
        "Limitations and Caveats",
        "Provenance and Hashes",
        "License and Attribution",
        "Support / Sponsored Runs",
    ]
    safe_patterns = [
        "Lead with artifact value: base model, GGUF format, size, hardware fit, benchmarks, and runtime flags.",
        "Describe Cerebellum as sensitivity-guided mixed-precision GGUF quantization.",
        "Keep method language high-level: precision was allocated from measured sensitivity.",
        "Tie claims to benchmark artifact files and exact runtime settings.",
        "Use Cerebellum as the public brand; keep legacy package names out of public-facing prose.",
    ]
    risky_patterns = [
        "Exact tensor, group, layer, override-map, ablation-delta, allocator, or recipe details.",
        "Private automation, dashboard/control-plane details, devlogs, local ports, local paths, and upload logs.",
        "Unreviewed user benchmark files, harm-check artifacts, agent workspaces, rowblock probes, and private release notes.",
        "Overclaims such as zero loss, proof, broken alternatives, or active improvement unless audited artifacts prove it.",
    ]
    artifact_policy = [
        {"artifact": "*_arc_results.json", "public": True, "note": "public-safe after benchmark audit"},
        {"artifact": "*_hellaswag_results.json", "public": True, "note": "public-safe after benchmark audit"},
        {"artifact": "*_mmlu_redux_results.json", "public": True, "note": "public-safe after benchmark audit"},
        {"artifact": "*_evalplus*_results.json", "public": True, "note": "include EvalPlus sample/eval artifacts when audited"},
        {"artifact": "*_detailed.jsonl", "public": True, "note": "publish only when intentionally audited for release"},
        {"artifact": "upload/server/local logs", "public": False, "note": "summarize settings instead of publishing raw logs"},
    ]
    cleanup_shortlist = [
        "README.md",
        "legacy Qwen/Gemma/Granite README model cards",
        "Gemma 4 HF card templates",
        "private release notes with local paths",
        "spaces/qwen36-cerebellum/README.md",
    ]
    return {
        "schema": "cerebellum.public_model_card_policy.v1",
        "sections": sections,
        "safe_patterns": safe_patterns,
        "risky_patterns": risky_patterns,
        "artifact_policy": artifact_policy,
        "support_placement": "after benchmark tables and before credits/license",
        "cleanup_shortlist": cleanup_shortlist,
    }


def public_model_card_policy_markdown(payload: dict[str, Any]) -> str:
    parts = [
        "# Cerebellum Public Model Card Policy",
        "",
        "## Sections",
        "",
        markdown_table(["Section"], [[section] for section in payload["sections"]]),
        "",
        "## Public-Safe Language",
        "",
        markdown_table(["Pattern"], [[row] for row in payload["safe_patterns"]]),
        "",
        "## Remove From Public Cards",
        "",
        markdown_table(["Risk"], [[row] for row in payload["risky_patterns"]]),
        "",
        "## Benchmark Artifacts",
        "",
        markdown_table(
            ["Artifact", "Public", "Note"],
            [[row["artifact"], "yes" if row["public"] else "no", row["note"]] for row in payload["artifact_policy"]],
        ),
        "",
        "## Support Placement",
        "",
        payload["support_placement"],
        "",
        "## Cleanup Shortlist",
        "",
        markdown_table(["Target"], [[row] for row in payload["cleanup_shortlist"]]),
    ]
    return "\n".join(parts) + "\n"


def public_model_card_policy_cmd(args: argparse.Namespace) -> None:
    payload = public_model_card_policy_payload()
    if args.write:
        Path(args.write).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.write)
        return
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(public_model_card_policy_markdown(payload), end="")


def benchmark_score_roots() -> list[Path]:
    roots: list[Path] = []
    for root in [Path.cwd() / "benchmark_results"]:
        roots.append(root)
    roots.extend(sorted(Path.cwd().glob("cerebellum-*/benchmark_results")))
    runs_root = default_data_root()
    if runs_root.exists():
        roots.extend(path for path in runs_root.rglob("benchmarks") if path.is_dir())
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def infer_model_from_result_path(path: Path, data: dict[str, Any]) -> str:
    model = data.get("model") or data.get("model_name")
    if model:
        return str(model)
    stem = path.stem
    for suffix in [
        "_evalplus_chat_results",
        "_evalplus_results",
        "_humaneval_results",
        "_arc_results",
        "_hellaswag_results",
        "_mmlu_redux_results",
        "_mmlu_results",
        "_perf_summary",
        "_results",
    ]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def discover_recent_benchmark_scores(limit: int = 8, max_files: int = 500) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for root in benchmark_score_roots():
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            name = path.name
            if "detailed" in name or "samples" in name or name.endswith("_eval_results.json"):
                continue
            if not (name.endswith("_results.json") or name.endswith("_summary.json") or name == "benchmark_report.json"):
                continue
            candidates.append(path)
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)[:max_files]
    rows: list[dict[str, Any]] = []
    for path in candidates:
        data = read_json(path, {})
        if not isinstance(data, dict):
            continue
        metric = benchmark_metric(data)
        if not metric:
            continue
        metric_name, value = metric
        rows.append(
            {
                "model": infer_model_from_result_path(path, data),
                "benchmark": benchmark_key(infer_benchmark_name(path, data)),
                "metric": metric_name,
                "value": value,
                "path": str(path),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            }
        )
        if len(rows) >= max(1, limit):
            break
    return rows


def home_payload(limit: int = 8) -> dict[str, Any]:
    runs = []
    for run_dir in known_run_dirs()[-max(1, limit) :]:
        state = read_json(run_dir / "state.json", {})
        manifest = read_json(run_dir / "manifest.json", {})
        if not state and not manifest:
            continue
        audit = golden_cow_audit(state, manifest) if state else {}
        runs.append(
            {
                "run_dir": str(run_dir),
                "model": state.get("model_name") or manifest.get("model_name") or run_dir.name,
                "family": state.get("model_family") or manifest.get("model_family"),
                "status": state.get("run_status"),
                "profile": state.get("ppl_profile") or manifest.get("ppl_profile"),
                "metric": state.get("ablation_metric") or manifest.get("ablation_metric"),
                "locked": len(state.get("locked", {})),
                "current_ppl": state.get("current_ppl"),
                "golden_cow_suspects": audit.get("suspect_count"),
                "updated_at": state.get("updated_at") or manifest.get("created_at"),
            }
        )
    commands = [
        {"key": "1", "label": "Watch active run", "command": "cerebellum watch"},
        {"key": "2", "label": "List runs", "command": "cerebellum runs"},
        {"key": "3", "label": "Open legacy-gated plan", "command": "cerebellum legacy-plan RUN_DIR"},
        {"key": "4", "label": "Legacy automated flow", "command": "cerebellum legacy-flow --source-gguf MODEL.gguf --output-dir cerebellum-MODEL"},
        {"key": "5", "label": "Benchmark plan", "command": "cerebellum benchmark-plan --suite release-local"},
        {"key": "6", "label": "Benchmark report", "command": "cerebellum benchmark-report RESULTS_DIR --leaderboard"},
        {"key": "7", "label": "Task profiles", "command": "cerebellum task-profiles"},
        {"key": "8", "label": "Recover run", "command": "cerebellum recover RUN_DIR"},
        {"key": "9", "label": "Project inventory", "command": "cerebellum project"},
        {"key": "10", "label": "Public model-card policy", "command": "cerebellum public-card-policy"},
    ]
    return {
        "schema": "cerebellum.home.v1",
        "commands": commands,
        "recent_runs": runs,
        "recent_scores": discover_recent_benchmark_scores(limit=limit),
        "profiles": sorted(TASK_PROFILES),
        "benchmark_suites": sorted(BENCHMARK_SUITES),
    }


def home_markdown(payload: dict[str, Any]) -> str:
    parts = [
        "# Cerebellum",
        "",
        "## Menu",
        "",
        markdown_table(["#", "Action", "Command"], [[row["key"], row["label"], f"`{row['command']}`"] for row in payload["commands"]]),
    ]
    runs = payload.get("recent_runs") or []
    if runs:
        run_rows = [
            [
                row.get("model") or "-",
                row.get("status") or "-",
                str(row.get("locked") or 0),
                "-" if row.get("current_ppl") is None else str(row.get("current_ppl")),
                "-" if row.get("golden_cow_suspects") is None else str(row.get("golden_cow_suspects")),
                row.get("profile") or "-",
                row.get("run_dir") or "-",
            ]
            for row in reversed(runs)
        ]
        parts.extend(["", "## Recent Runs", "", markdown_table(["Model", "Status", "Locks", "PPL", "Risk", "Profile", "Run"], run_rows)])
    else:
        parts.extend(["", "## Recent Runs", "", "No Cerebellum runs found yet."])
    scores = payload.get("recent_scores") or []
    if scores:
        score_rows = [
            [
                row.get("model") or "-",
                row.get("benchmark") or "-",
                row.get("metric") or "-",
                f"{float(row['value']):.2f}" if row.get("value") is not None else "-",
                row.get("path") or "-",
            ]
            for row in scores
        ]
        parts.extend(["", "## Recent Scores", "", markdown_table(["Model", "Benchmark", "Metric", "Score", "Artifact"], score_rows)])
    parts.extend(
        [
            "",
            "## Profiles",
            "",
            ", ".join(f"`{name}`" for name in payload.get("profiles", [])),
            "",
            "## Benchmark Suites",
            "",
            ", ".join(f"`{name}`" for name in payload.get("benchmark_suites", [])),
            "",
            "Use `cerebellum COMMAND --help` for command-specific options.",
        ]
    )
    return "\n".join(parts) + "\n"


def home_cmd(args: argparse.Namespace) -> None:
    payload = home_payload(limit=getattr(args, "limit", 8))
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(home_markdown(payload), end="")


def parse_ppl(output: str) -> tuple[float | None, float | None]:
    for line in output.splitlines():
        if "Final estimate" not in line:
            continue
        match = re.search(r"PPL\s*=\s*([0-9.]+)(?:\s*\+/-\s*([0-9.]+))?", line)
        if match:
            ppl = float(match.group(1))
            err = float(match.group(2)) if match.group(2) else None
            return ppl, err
    return None, None


def run_external(
    cmd: list[str],
    timeout: int,
    distrobox: str | None = None,
    heartbeat: Any | None = None,
    heartbeat_interval: float = 15.0,
) -> tuple[int, str, float]:
    if distrobox:
        import shlex

        shell_cmd = shlex.join(cmd)
        cmd = ["distrobox", "enter", distrobox, "--", "bash", "-lc", shell_cmd]
    started = time.monotonic()
    if heartbeat is None:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - started
        return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), elapsed
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            proc.kill()
            output, _ = proc.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout, output=output)
        try:
            output, _ = proc.communicate(timeout=heartbeat_interval)
            elapsed = time.monotonic() - started
            return proc.returncode or 0, output or "", elapsed
        except subprocess.TimeoutExpired:
            heartbeat(round(time.monotonic() - started, 3), proc.pid)


def disk_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / (1024**3)


def bytes_to_gb(size: int | None) -> float:
    return (size or 0) / (1024**3)


def path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path_size(path)
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += path_size(file)
    return total


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def backup_run_metadata(run_dir: Path, backup_root: Path) -> dict[str, Any]:
    backup_dir = backup_root / slug(run_dir.name)
    files = [
        "state.json",
        "manifest.json",
        CURRENT_TYPES_FILE,
        EVENT_FILES[0],
        CANDIDATE_FILES[0],
        "timing.json",
    ]
    copied: list[str] = []
    for name in files:
        if copy_if_exists(run_dir / name, backup_dir / name):
            copied.append(name)
    checkpoints = run_dir / "checkpoints"
    if checkpoints.exists():
        for file in checkpoints.glob("*.json"):
            rel = Path("checkpoints") / file.name
            if copy_if_exists(file, backup_dir / rel):
                copied.append(str(rel))
    return {"backup_dir": str(backup_dir), "copied": copied}


def write_tensor_types_map(source_gguf: Path | None, locked: dict[str, str], start_type: str, path: Path) -> None:
    names: list[str] = []
    if source_gguf:
        try:
            from gguf import GGUFReader

            reader = GGUFReader(str(source_gguf))
            names = quantizable_tensor_names([t.name for t in reader.tensors])
        except Exception:
            names = []
    if not names:
        names = sorted(quantizable_tensor_names(set(locked)))
    lines = [tensor_type_line(name, locked.get(name, start_type)) for name in names]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def totals_for_kept_candidates(candidates: list[dict[str, Any]], kept: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"quant_seconds": 0.0, "ppl_seconds": 0.0, "candidates": 0, "failures": 0}
    if not kept:
        return totals
    kept_tensors = [row.get("tensor") for row in kept if row.get("tensor")]
    if not kept_tensors:
        return totals
    current = 0
    for row in candidates:
        tensor = row.get("tensor")
        if current >= len(kept_tensors):
            break
        if tensor == kept_tensors[current]:
            pass
        elif current + 1 < len(kept_tensors) and tensor == kept_tensors[current + 1]:
            current += 1
        else:
            continue
        totals["quant_seconds"] += row.get("quant_seconds") or 0.0
        totals["ppl_seconds"] += row.get("ppl_seconds") or 0.0
        totals["candidates"] += 1
        if row.get("status") == "failed":
            totals["failures"] += 1
    return totals


def gguf_field_text(gguf: Path, key: str) -> str | None:
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    try:
        field = GGUFReader(str(gguf)).fields.get(key)
    except Exception:
        return None
    if field is None:
        return None
    parts = getattr(field, "parts", [])
    if len(parts) >= 5:
        raw = parts[4]
        try:
            return bytes(raw.tolist()).decode("utf-8", errors="replace")
        except Exception:
            try:
                return bytes(raw).decode("utf-8", errors="replace")
            except Exception:
                return None
    return None


def first_existing(run_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    return run_dir / names[0]


def append_event(path: Path, event: str, **fields: Any) -> None:
    rows = read_jsonl(path)
    event_id = max([int(row.get("event_id") or 0) for row in rows], default=0) + 1
    row = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "timestamp_utc": utc_now(),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_glob(root: Path) -> list[Path]:
    manifests = list(root.glob("families/**/sources/*/runs/*/manifest.json"))
    manifests.extend(root.glob("runs/*/manifest.json"))
    manifests.extend(path for path in root.rglob("manifest.json") if (path.parent / "state.json").exists())
    return sorted(set(manifests))


def project_glob(root: Path) -> list[Path]:
    return sorted(root.glob("families/*/*/sources/*/cerebellum_project.json"))


def discover_projects(root: Path) -> list[dict[str, Any]]:
    projects: dict[str, dict[str, Any]] = {}
    for manifest_path in project_glob(root):
        source_root = manifest_path.parent
        data = read_json(manifest_path, {})
        key = str(source_root)
        projects[key] = {
            "source_root": str(source_root),
            "family": data.get("family") or source_root.parents[2].name,
            "model_name": data.get("model_name") or source_root.parents[1].name,
            "source_name": data.get("source_name") or source_root.name,
            "project_manifest": str(manifest_path),
            "imatrix": data.get("imatrix"),
            "next_command": data.get("next_command"),
            "runs": [],
        }
    for run_manifest in run_glob(root):
        source_root = run_manifest.parents[2]
        key = str(source_root)
        run = load_run(run_manifest.parent)
        state = run.get("state", {})
        manifest = run.get("manifest", {})
        item = projects.setdefault(
            key,
            {
                "source_root": str(source_root),
                "family": manifest.get("model_family") or state.get("model_family") or source_root.parents[2].name,
                "model_name": manifest.get("model_name") or state.get("model_name") or source_root.parents[1].name,
                "source_name": manifest.get("source_name") or state.get("source_name") or source_root.name,
                "project_manifest": None,
                "imatrix": manifest.get("imatrix") or state.get("imatrix"),
                "next_command": None,
                "runs": [],
            },
        )
        item["runs"].append(
            {
                "run_dir": str(run_manifest.parent),
                "run_id": manifest.get("run_id") or state.get("run_id") or run_manifest.parent.name,
                "status": state.get("run_status"),
                "locked": len(state.get("locked", {})),
                "current_ppl": state.get("current_ppl"),
                "profile": manifest.get("ppl_profile") or state.get("ppl_profile"),
                "updated_at": state.get("updated_at"),
            }
        )
    return sorted(projects.values(), key=lambda row: (str(row.get("family")), str(row.get("model_name")), str(row.get("source_name"))))


def default_data_root() -> Path:
    if os.environ.get("CEREBELLUM_DATA_ROOT"):
        return Path(os.environ["CEREBELLUM_DATA_ROOT"])
    return Path.home() / "cerebellum-runs"


def candidate_data_roots() -> list[Path]:
    roots = [
        default_data_root(),
        Path.cwd() / "cerebellum-runs",
        Path.home() / "games" / "cerebellum-runs",
        Path.home() / "ai-drive" / "cerebellum-runs",
    ]
    seen: set[str] = set()
    unique = []
    for root in roots:
        key = str(root.expanduser())
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def known_run_dirs() -> list[Path]:
    runs: list[Path] = []
    seen: set[str] = set()
    for root in candidate_data_roots():
        if not root.exists():
            continue
        for manifest in run_glob(root):
            run_dir = manifest.parent
            if "backups" in run_dir.parts:
                continue
            key = str(run_dir.resolve())
            if key not in seen:
                seen.add(key)
                runs.append(run_dir)
    return sorted(runs)


def run_matches_ref(run_dir: Path, ref: str) -> bool:
    run = load_run(run_dir)
    manifest = run.get("manifest", {})
    state = run.get("state", {})
    needles = {
        run_dir.name,
        str(manifest.get("run_id") or ""),
        str(state.get("run_id") or ""),
        str(manifest.get("model_name") or ""),
        str(state.get("model_name") or ""),
        f"{manifest.get('model_family')}/{manifest.get('model_name')}",
        f"{state.get('model_family')}/{state.get('model_name')}",
    }
    return ref in needles


def run_is_live(run_dir: Path) -> bool:
    state = read_json(run_dir / "state.json", {})
    if state.get("run_status") != "running":
        return False
    return any(row["kind"] == "runner" for row in process_rows_for_run(run_dir))


def run_sort_mtime(run_dir: Path) -> float:
    candidates = [run_dir / "state.json", run_dir / "manifest.json", first_existing(run_dir, EVENT_FILES), first_existing(run_dir, CANDIDATE_FILES)]
    return max((path.stat().st_mtime for path in candidates if path.exists()), default=0.0)


def scratch_run_root(run_dir: Path, manifest: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> Path | None:
    manifest = manifest or read_json(run_dir / "manifest.json", {})
    state = state or read_json(run_dir / "state.json", {})
    scratch = manifest.get("scratch_root") or state.get("scratch_root")
    run_id = manifest.get("run_id") or state.get("run_id") or run_dir.name
    if not scratch or not run_id:
        return None
    return Path(str(scratch)) / str(run_id)


def run_tmp_root(run_dir: Path, manifest: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> Path:
    scratch = scratch_run_root(run_dir, manifest, state)
    return (scratch / "tmp") if scratch else (run_dir / "tmp")


def run_artifacts_root(run_dir: Path, manifest: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> Path:
    scratch = scratch_run_root(run_dir, manifest, state)
    return (scratch / "artifacts") if scratch else (run_dir / "artifacts")


def resolve_run_dir(ref: str | None) -> Path:
    if ref:
        path = Path(ref)
        if path.exists():
            return path
        matches = [run_dir for run_dir in known_run_dirs() if run_matches_ref(run_dir, ref)]
        if not matches:
            raise SystemExit(f"no Cerebellum run found for {ref!r}; pass RUN_DIR or set CEREBELLUM_DATA_ROOT")
        live = [run_dir for run_dir in matches if run_is_live(run_dir)]
        chosen = live or matches
        if len(chosen) == 1:
            return chosen[0]
        options = "\n".join(f"  {path}" for path in chosen[:20])
        raise SystemExit(f"multiple Cerebellum runs match {ref!r}; pass RUN_DIR\n{options}")
    live = [run_dir for run_dir in known_run_dirs() if run_is_live(run_dir)]
    if len(live) == 1:
        return live[0]
    if not live:
        recent = sorted(known_run_dirs(), key=run_sort_mtime, reverse=True)
        if recent:
            return recent[0]
        raise SystemExit("no active Cerebellum run found; pass RUN_DIR or model name")
    options = "\n".join(f"  {path}" for path in live[:20])
    raise SystemExit(f"multiple active Cerebellum runs found; pass RUN_DIR or model name\n{options}")


def config_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "cerebellum" / "config.json"


def load_user_config() -> dict[str, Any]:
    return read_json(config_path(), {"tips": True})


def save_user_config(data: dict[str, Any]) -> None:
    atomic_write_json(config_path(), data)


def find_executable(name: str, env_var: str, common: list[Path] | None = None) -> str:
    if os.environ.get(env_var):
        return os.environ[env_var]
    found = shutil.which(name)
    if found:
        return found
    for path in common or []:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return name


def common_llama_bins(binary: str) -> list[Path]:
    roots = [
        Path.cwd() / "llama.cpp" / "build" / "bin",
        Path.cwd().parent / "llama.cpp" / "build" / "bin",
        Path.home() / "llama.cpp" / "build" / "bin",
        Path("/var/home/deucebucket/ai-drive/llama.cpp/build/bin"),
    ]
    return [root / binary for root in roots]


def resolve_ppl_corpus(profile: str, corpus: str | None) -> Path:
    if corpus:
        return Path(corpus)
    if profile == "custom":
        raise SystemExit("--corpus is required when --profile custom")
    for path in profile_candidate_paths(profile):
        if path.exists():
            return path
    raise SystemExit(f"no local corpus found for --profile {profile}; pass --corpus explicitly")


def profile_candidate_paths(profile: str) -> list[Path]:
    names = PPL_PROFILES.get(profile, [])
    roots: list[Path] = []
    if os.environ.get("CEREBELLUM_CORPUS_ROOT"):
        roots.append(Path(os.environ["CEREBELLUM_CORPUS_ROOT"]))
    roots.extend(
        [
            Path.cwd() / "corpora",
            Path.cwd(),
            Path.home() / ".cache" / "cerebellum" / "corpora",
        ]
    )
    roots.extend(LEGACY_PROFILE_ROOTS)
    paths: list[Path] = []
    for name in names:
        path = Path(name)
        if path.is_absolute():
            paths.append(path)
            continue
        paths.extend(root / name for root in roots)
    return paths


def load_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json", {})
    state = read_json(run_dir / "state.json", {})
    return {"run_dir": str(run_dir), "manifest": manifest, "state": state}


def sqlite_rows(db: Path, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def ensure_hill_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS hill_runs (
            run_id TEXT PRIMARY KEY,
            run_dir TEXT NOT NULL,
            model_family TEXT,
            model_name TEXT,
            source_name TEXT,
            source_gguf TEXT,
            base_type TEXT,
            start_type TEXT,
            levels_json TEXT,
            status TEXT,
            current_ppl REAL,
            locked_count INTEGER,
            candidate_count INTEGER,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS hill_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            tensor_name TEXT NOT NULL,
            layer_idx INTEGER,
            component TEXT,
            candidate_quant TEXT NOT NULL,
            baseline_ppl REAL,
            candidate_ppl REAL,
            delta REAL,
            ppl_error REAL,
            quant_seconds REAL,
            ppl_seconds REAL,
            size_bytes INTEGER,
            status TEXT,
            UNIQUE(run_id, tensor_name, candidate_quant),
            FOREIGN KEY(run_id) REFERENCES hill_runs(run_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_hill_candidates_tensor ON hill_candidates(tensor_name);
        CREATE INDEX IF NOT EXISTS idx_hill_candidates_component ON hill_candidates(component);
        CREATE INDEX IF NOT EXISTS idx_hill_runs_model ON hill_runs(model_family, model_name);

        CREATE TABLE IF NOT EXISTS cerebellum_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            label TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            priority INTEGER NOT NULL DEFAULT 100,
            payload_json TEXT NOT NULL,
            notes TEXT,
            started_at TEXT,
            finished_at TEXT,
            log TEXT,
            result_json TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cerebellum_jobs_status ON cerebellum_jobs(status, priority, id);
        CREATE INDEX IF NOT EXISTS idx_cerebellum_jobs_kind ON cerebellum_jobs(kind, status);

        CREATE TABLE IF NOT EXISTS cerebellum_benchmark_ingests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            suite TEXT NOT NULL,
            results_dir TEXT NOT NULL,
            ready INTEGER NOT NULL DEFAULT 0,
            blockers_json TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            audit_json TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(model, suite, results_dir)
        );

        CREATE TABLE IF NOT EXISTS cerebellum_benchmark_results (
            ingest_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            suite TEXT NOT NULL,
            benchmark_key TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            path TEXT NOT NULL,
            size_gib REAL,
            release_metadata_json TEXT NOT NULL,
            PRIMARY KEY(ingest_id, benchmark_key, metric, path),
            FOREIGN KEY(ingest_id) REFERENCES cerebellum_benchmark_ingests(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_cerebellum_benchmark_results_suite ON cerebellum_benchmark_results(suite, benchmark_key);
        CREATE INDEX IF NOT EXISTS idx_cerebellum_benchmark_results_model ON cerebellum_benchmark_results(model);
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(cerebellum_jobs)").fetchall()}
    for name, decl in {
        "started_at": "TEXT",
        "finished_at": "TEXT",
        "log": "TEXT",
        "result_json": "TEXT",
        "last_error": "TEXT",
    }.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE cerebellum_jobs ADD COLUMN {name} {decl}")


def import_run_to_db(db: Path, run_dir: Path) -> dict[str, Any]:
    report = build_report(run_dir)
    manifest = read_json(run_dir / "manifest.json", {})
    state = read_json(run_dir / "state.json", {})
    candidates = read_jsonl(first_existing(run_dir, CANDIDATE_FILES))
    conn = sqlite3.connect(db)
    try:
        ensure_hill_tables(conn)
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hill_runs
                  (run_id, run_dir, model_family, model_name, source_name, source_gguf,
                   base_type, start_type, levels_json, status, current_ppl, locked_count,
                   candidate_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report["run_id"],
                    str(run_dir),
                    report.get("model_family"),
                    report.get("model_name"),
                    report.get("source_name"),
                    manifest.get("source_gguf") or state.get("source_gguf"),
                    manifest.get("base_type") or state.get("base_type"),
                    manifest.get("start_type") or state.get("start_type"),
                    json.dumps(report.get("levels") or []),
                    report.get("status"),
                    report.get("current_ppl"),
                    report.get("locked_count"),
                    report.get("candidate_count"),
                    manifest.get("created_at") or state.get("created_at"),
                    state.get("updated_at"),
                ),
            )
            for row in candidates:
                layer, component = parse_tensor_name(row.get("tensor", ""))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO hill_candidates
                      (run_id, tensor_name, layer_idx, component, candidate_quant,
                       baseline_ppl, candidate_ppl, delta, ppl_error, quant_seconds,
                       ppl_seconds, size_bytes, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report["run_id"],
                        row.get("tensor"),
                        layer,
                        component,
                        row.get("level"),
                        row.get("baseline_ppl"),
                        row.get("ppl"),
                        row.get("delta"),
                        row.get("ppl_error"),
                        row.get("quant_seconds"),
                        row.get("ppl_seconds"),
                        row.get("size_bytes"),
                        row.get("status"),
                    ),
                )
        return {"run_id": report["run_id"], "candidates": len(candidates)}
    finally:
        conn.close()


def queue_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": args.kind}
    if args.manifest:
        manifest_path = Path(args.manifest)
        payload["manifest"] = str(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"failed to read queue manifest {manifest_path}: {exc}") from exc
        payload["pipeline"] = manifest.get("pipeline")
        payload["run_dir"] = manifest.get("run_dir")
        payload["phases"] = [row.get("name") for row in manifest.get("phases", []) if isinstance(row, dict)]
    if getattr(args, "from_phase", None):
        payload["from_phase"] = args.from_phase
    if getattr(args, "until_phase", None):
        payload["until_phase"] = args.until_phase
    if args.command:
        payload["command"] = args.command
    if args.payload_json:
        try:
            extra = json.loads(args.payload_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--payload-json must be a JSON object: {exc}") from exc
        if not isinstance(extra, dict):
            raise SystemExit("--payload-json must be a JSON object")
        payload.update(extra)
    if args.kind == "pipeline" and not payload.get("manifest"):
        raise SystemExit("queue add --kind pipeline requires --manifest")
    if args.kind in {"benchmark", "run"} and not payload.get("command"):
        raise SystemExit(f"queue add --kind {args.kind} requires --command")
    return payload


def decode_queue_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.pop("payload_json", "{}")
    try:
        row["payload"] = json.loads(payload)
    except json.JSONDecodeError:
        row["payload"] = {"_decode_error": True, "raw": payload}
    result = row.get("result_json")
    if result:
        try:
            row["result"] = json.loads(str(result))
        except json.JSONDecodeError:
            row["result"] = {"_decode_error": True, "raw": result}
    return row


def tail_text_file(path: Path, lines: int = 40, max_bytes: int = 128_000) -> str:
    if lines <= 0 or not path.exists() or not path.is_file():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(max(0, size - max_bytes))
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def queue_attach_log_tail(job: dict[str, Any], tail: int | None = None) -> dict[str, Any]:
    if not tail:
        return job
    log = job.get("log")
    if not log:
        return job
    path = Path(str(log))
    job["log_tail"] = tail_text_file(path, lines=tail)
    return job


def queue_get_job(db: Path, job_id: int, tail: int | None = None) -> dict[str, Any]:
    if not db.exists():
        raise SystemExit(f"queue job {job_id} not found")
    conn = sqlite3.connect(db)
    try:
        ensure_hill_tables(conn)
    finally:
        conn.close()
    rows = sqlite_rows(db, "SELECT * FROM cerebellum_jobs WHERE id = ?", (job_id,))
    if not rows:
        raise SystemExit(f"queue job {job_id} not found")
    return queue_attach_log_tail(decode_queue_row(rows[0]), tail=tail)


def queue_add_job(args: argparse.Namespace) -> dict[str, Any]:
    db = Path(args.db)
    payload = queue_payload_from_args(args)
    now = utc_now()
    label = args.label or payload.get("run_dir") or payload.get("manifest") or payload.get("command")
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        ensure_hill_tables(conn)
        with conn:
            cur = conn.execute(
                """
                INSERT INTO cerebellum_jobs
                  (kind, label, status, priority, payload_json, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    args.kind,
                    str(label) if label else None,
                    args.status,
                    int(args.priority),
                    json.dumps(payload, sort_keys=True),
                    args.notes,
                    now,
                    now,
                ),
            )
            job_id = int(cur.lastrowid)
    finally:
        conn.close()
    return queue_get_job(db, job_id)


def queue_list_jobs(db: Path, status: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    try:
        ensure_hill_tables(conn)
    finally:
        conn.close()
    where = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if kind:
        where.append("kind = ?")
        params.append(kind)
    sql = "SELECT * FROM cerebellum_jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY priority ASC, id ASC LIMIT ?"
    params.append(max(1, int(limit)))
    return [decode_queue_row(row) for row in sqlite_rows(db, sql, tuple(params))]


def queue_next_job(db: Path, status: str = "queued", kind: str | None = None) -> dict[str, Any] | None:
    jobs = queue_list_jobs(db, status=status, kind=kind, limit=1)
    return jobs[0] if jobs else None


def queue_update_job(
    db: Path,
    job_id: int,
    *,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    log: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    conn = sqlite3.connect(db)
    try:
        ensure_hill_tables(conn)
        with conn:
            conn.execute(
                """
                UPDATE cerebellum_jobs
                SET status = ?,
                    started_at = COALESCE(?, started_at),
                    finished_at = COALESCE(?, finished_at),
                    log = COALESCE(?, log),
                    result_json = COALESCE(?, result_json),
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    started_at,
                    finished_at,
                    log,
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    error,
                    now,
                    job_id,
                ),
            )
    finally:
        conn.close()
    return queue_get_job(db, job_id)


def queue_cancel_job(db: Path, job_id: int, reason: str | None = None) -> dict[str, Any]:
    job = queue_get_job(db, job_id)
    if job["status"] not in {"queued", "failed"}:
        raise SystemExit(f"queue job {job_id} has status {job['status']}; only queued/failed jobs can be canceled")
    return queue_update_job(db, job_id, status="canceled", finished_at=utc_now(), error=reason or "canceled")


def queue_retry_job(db: Path, job_id: int, priority: int | None = None, notes: str | None = None) -> dict[str, Any]:
    job = queue_get_job(db, job_id)
    if job["status"] not in {"failed", "canceled", "completed"}:
        raise SystemExit(f"queue job {job_id} has status {job['status']}; only failed/canceled/completed jobs can be retried")
    now = utc_now()
    conn = sqlite3.connect(db)
    try:
        ensure_hill_tables(conn)
        with conn:
            conn.execute(
                """
                UPDATE cerebellum_jobs
                SET status = 'queued',
                    priority = COALESCE(?, priority),
                    notes = COALESCE(?, notes),
                    started_at = NULL,
                    finished_at = NULL,
                    result_json = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (priority, notes, now, job_id),
            )
    finally:
        conn.close()
    return queue_get_job(db, job_id)


def queue_execute_command(command: str, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_prefix, argv = split_env_prefixed_command(command)
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(argv, cwd=Path.cwd(), env={**os.environ, **env_prefix}, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
    return {"returncode": proc.returncode, "elapsed_seconds": round(time.monotonic() - started, 3), "log": str(log_path), "command": command}


def queue_run_job(db: Path, job: dict[str, Any], execute: bool = False) -> dict[str, Any]:
    payload = job.get("payload", {})
    log_path = db.parent / "queue_logs" / f"job_{job['id']}.log"
    dry_result = {"schema": "cerebellum.queue_run.v1", "dry_run": True, "job": job, "log": str(log_path)}
    if not execute:
        return dry_result
    started_at = utc_now()
    queue_update_job(db, int(job["id"]), status="running", started_at=started_at, log=str(log_path), error=None)
    try:
        if job["kind"] == "pipeline":
            manifest = payload.get("manifest")
            if not manifest:
                raise RuntimeError("pipeline queue job missing manifest")
            result = pipeline_run_execute(
                pipeline_run_plan(
                    Path(str(manifest)),
                    from_phase=payload.get("from_phase"),
                    until_phase=payload.get("until_phase"),
                )
            )
            returncode = 1 if result.get("blocked") else 0
            result["returncode"] = returncode
        else:
            command = payload.get("command")
            if not command:
                raise RuntimeError(f"{job['kind']} queue job missing command")
            result = queue_execute_command(str(command), log_path)
            returncode = int(result["returncode"])
        status = "completed" if returncode == 0 else "failed"
        finished = queue_update_job(db, int(job["id"]), status=status, finished_at=utc_now(), log=str(result.get("log") or log_path), result=result, error=None if returncode == 0 else f"command exited {returncode}")
        return {"schema": "cerebellum.queue_run.v1", "dry_run": False, "job": finished, "result": result}
    except Exception as exc:
        failed = queue_update_job(db, int(job["id"]), status="failed", finished_at=utc_now(), log=str(log_path), result=None, error=str(exc))
        return {"schema": "cerebellum.queue_run.v1", "dry_run": False, "job": failed, "error": str(exc)}


def queue_markdown(payload: dict[str, Any]) -> str:
    rows = [
        [str(row["id"]), str(row["kind"]), str(row["status"]), str(row["priority"]), str(row.get("label") or "-")]
        for row in payload["jobs"]
    ]
    return "\n".join(
        [
            "# Cerebellum Queue",
            "",
            f"db: `{payload['db']}`",
            f"jobs: `{len(payload['jobs'])}`",
            "",
            markdown_table(["ID", "Kind", "Status", "Priority", "Label"], rows),
        ]
    ) + "\n"


def queue_run_markdown(payload: dict[str, Any]) -> str:
    job = payload.get("job")
    parts = [
        "# Cerebellum Queue Run",
        "",
        f"mode: `{'dry-run' if payload.get('dry_run') else 'execute'}`",
        f"db: `{payload.get('db', '-')}`",
    ]
    if not job:
        parts.append(f"message: `{payload.get('message', 'no job')}`")
        return "\n".join(parts) + "\n"
    parts.extend(
        [
            f"job: `{job['id']}`",
            f"kind: `{job['kind']}`",
            f"status: `{job['status']}`",
            f"label: `{job.get('label') or '-'}`",
        ]
    )
    if payload.get("result"):
        result = payload["result"]
        parts.append(f"returncode: `{result.get('returncode')}`")
        parts.append(f"log: `{result.get('log') or job.get('log') or '-'}`")
    elif payload.get("log"):
        parts.append(f"log: `{payload['log']}`")
    if payload.get("error"):
        parts.append(f"error: `{payload['error']}`")
    return "\n".join(parts) + "\n"


def queue_action_markdown(payload: dict[str, Any]) -> str:
    job = payload["jobs"][0] if payload.get("jobs") else None
    lines = ["# Cerebellum Queue", "", f"db: `{payload['db']}`"]
    if job:
        lines.extend(
            [
                f"job: `{job['id']}`",
                f"kind: `{job['kind']}`",
                f"status: `{job['status']}`",
                f"label: `{job.get('label') or '-'}`",
            ]
        )
        if job.get("last_error"):
            lines.append(f"last error: `{job['last_error']}`")
        if job.get("log_tail"):
            lines.extend(["", "```text", str(job["log_tail"]), "```"])
    return "\n".join(lines) + "\n"


def queue_cmd(args: argparse.Namespace) -> None:
    db = Path(args.db)
    if args.queue_cmd == "add":
        payload = {"schema": "cerebellum.queue.v1", "db": str(db), "jobs": [queue_add_job(args)]}
    elif args.queue_cmd == "list":
        payload = {"schema": "cerebellum.queue.v1", "db": str(db), "jobs": queue_list_jobs(db, status=args.status, kind=args.kind, limit=args.limit)}
    elif args.queue_cmd == "get":
        payload = {"schema": "cerebellum.queue.v1", "db": str(db), "jobs": [queue_get_job(db, args.id, tail=args.tail)]}
    elif args.queue_cmd == "cancel":
        payload = {"schema": "cerebellum.queue.v1", "db": str(db), "jobs": [queue_cancel_job(db, args.id, reason=args.reason)]}
    elif args.queue_cmd == "retry":
        payload = {"schema": "cerebellum.queue.v1", "db": str(db), "jobs": [queue_retry_job(db, args.id, priority=args.priority, notes=args.notes)]}
    elif args.queue_cmd == "run-next":
        job = queue_next_job(db, status=args.status, kind=args.kind)
        if job is None:
            payload = {"schema": "cerebellum.queue_run.v1", "db": str(db), "dry_run": not args.execute, "job": None, "message": "no matching queued job"}
        else:
            payload = queue_run_job(db, job, execute=args.execute)
            payload["db"] = str(db)
    else:
        raise SystemExit(f"unknown queue command: {args.queue_cmd}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.queue_cmd == "run-next":
        print(queue_run_markdown(payload), end="")
    elif args.queue_cmd in {"get", "cancel", "retry"}:
        print(queue_action_markdown(payload), end="")
    else:
        print(queue_markdown(payload), end="")


def parse_tensor_name(tensor: str) -> tuple[int | None, str | None]:
    match = re.match(r"blk\.(\d+)\.(.+)\.weight$", tensor)
    if match:
        return int(match.group(1)), match.group(2)
    if tensor.endswith(".weight"):
        return None, tensor[:-7]
    return None, None


def parse_layer_spec(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            layers.update(range(min(start, end), max(start, end) + 1))
        else:
            layers.add(int(part))
    return layers


def tensor_layer(tensor: str) -> int | None:
    layer, _component = parse_tensor_name(tensor)
    return layer


@dataclass
class Candidate:
    tensor: str
    level: str
    type_file: Path
    gguf_path: Path
    index: int
    quant_started_at: str | None = None
    quant_finished_at: str | None = None
    quant_seconds: float | None = None
    quant_returncode: int | None = None
    quant_output_tail: str = ""
    quant_ok: bool = False
    ppl_started_at: str | None = None
    ppl_finished_at: str | None = None
    ppl_seconds: float | None = None
    ppl_returncode: int | None = None
    ppl_output_tail: str = ""
    ppl: float | None = None
    ppl_error: float | None = None
    status: str = "pending"


@dataclass
class Config:
    source_gguf: Path
    corpus: Path
    ppl_profile: str
    ablation_metric: str
    run_dir: Path
    run_id: str
    model_family: str
    model_name: str
    source_name: str
    base_type: str
    start_type: str
    levels: list[str]
    quantize_bin: str
    perplexity_bin: str
    gpu_layers: int
    ctx_size: int
    chunks: int | None
    imatrix: Path | None = None
    tensor_file: Path | None = None
    base_map: Path | None = None
    scratch_root: Path | None = None
    backup_root: Path | None = None
    max_temp_gb: float = 80.0
    min_free_gb: float = 40.0
    hard_free_floor_gb: float = 10.0
    keep_winners: bool = True
    keep_losers: bool = False
    low_space: bool = False
    serial_candidates: bool = False
    prune_measured_candidates: bool = True
    distrobox: str | None = None
    quant_timeout: int = 1800
    ppl_timeout: int = 900
    color: bool = True
    plain: bool = False
    backup_every: int = 1
    token_embedding_type: str | None = "f16"
    noise_pct: float = 0.0
    layers: set[int] | None = None
    tensor_regex: str | None = None
    pure_quant: bool = False
    commit_locks: bool = True


@dataclass
class Paths:
    state: Path
    events: Path
    candidates: Path
    timing: Path
    current_types: Path
    final_types: Path
    manifest: Path
    artifacts: Path
    checkpoints: Path
    tmp: Path
    baseline: Path


class HillStepper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths = Paths(
            state=cfg.run_dir / "state.json",
            events=cfg.run_dir / EVENT_FILES[0],
            candidates=cfg.run_dir / CANDIDATE_FILES[0],
            timing=cfg.run_dir / "timing.json",
            current_types=cfg.run_dir / CURRENT_TYPES_FILE,
            final_types=cfg.run_dir / BEST_TYPES_FILES[0],
            manifest=cfg.run_dir / "manifest.json",
            artifacts=(cfg.scratch_root / cfg.run_id / "artifacts") if cfg.scratch_root else cfg.run_dir / "artifacts",
            checkpoints=cfg.run_dir / "checkpoints",
            tmp=(cfg.scratch_root / cfg.run_id / "tmp") if cfg.scratch_root else cfg.run_dir / "tmp",
            baseline=((cfg.scratch_root / cfg.run_id / "artifacts") if cfg.scratch_root else cfg.run_dir / "artifacts") / "current_baseline.gguf",
        )
        self.events = EventLog(self.paths.events, cfg.run_id, cfg)
        self.candidate_log = EventLog(self.paths.candidates, cfg.run_id, cfg)
        self.base_overrides = read_tensor_type_map(cfg.base_map) if cfg.base_map else {}
        self.stop_requested = False
        self._install_signals()

    def _install_signals(self) -> None:
        def handle(signum: int, _frame: Any) -> None:
            self.stop_requested = True
            self.events.write("signal_received", signum=signum)

        signal.signal(signal.SIGTERM, handle)
        signal.signal(signal.SIGINT, handle)

    def load_state(self) -> dict[str, Any]:
        if self.paths.state.exists():
            return json.loads(self.paths.state.read_text())
        return {
            "schema_version": 1,
            "run_id": self.cfg.run_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "model_family": self.cfg.model_family,
            "model_name": self.cfg.model_name,
            "source_name": self.cfg.source_name,
            "source_gguf": str(self.cfg.source_gguf),
            "corpus": str(self.cfg.corpus),
            "ppl_profile": self.cfg.ppl_profile,
            "ablation_metric": self.cfg.ablation_metric,
            "base_type": self.cfg.base_type,
            "start_type": self.cfg.start_type,
            "levels": self.cfg.levels,
            "base_map": str(self.cfg.base_map) if self.cfg.base_map else None,
            "base_map_count": len(self.base_overrides),
            "locked": {},
            "tested": [],
            "current_ppl": None,
            "baseline_path": str(self.paths.baseline),
            "run_status": "created",
            "last_tensor": None,
            "totals": {
                "quant_seconds": 0.0,
                "ppl_seconds": 0.0,
                "gpu_wait_seconds": 0.0,
                "cpu_wait_seconds": 0.0,
                "candidates": 0,
                "failures": 0,
            },
        }

    def save_state(self, state: dict[str, Any], checkpoint: bool = False) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.paths.state, state)
        if checkpoint:
            done = len(state.get("locked", {}))
            cp = self.paths.checkpoints / f"state-{done:05d}-{local_stamp()}.json"
            atomic_write_json(cp, state)
        if self.cfg.backup_root:
            backup = backup_run_metadata(self.cfg.run_dir, self.cfg.backup_root)
            state["last_metadata_backup"] = backup
            atomic_write_json(self.paths.state, state)
            copy_if_exists(self.paths.state, Path(backup["backup_dir"]) / "state.json")

    def write_manifest(self) -> None:
        data = {
            "schema_version": 1,
            "run_id": self.cfg.run_id,
            "tool": "cerebellum",
            "created_at": utc_now(),
            "model_family": self.cfg.model_family,
            "model_name": self.cfg.model_name,
            "source_name": self.cfg.source_name,
            "source_gguf": str(self.cfg.source_gguf),
            "corpus": str(self.cfg.corpus),
            "ppl_profile": self.cfg.ppl_profile,
            "ablation_metric": self.cfg.ablation_metric,
            "run_dir": str(self.cfg.run_dir),
            "base_type": self.cfg.base_type,
            "start_type": self.cfg.start_type,
            "levels": self.cfg.levels,
            "base_map": str(self.cfg.base_map) if self.cfg.base_map else None,
            "base_map_count": len(self.base_overrides),
            "layers": sorted(self.cfg.layers) if self.cfg.layers else None,
            "tensor_regex": self.cfg.tensor_regex,
            "low_space": self.cfg.low_space,
            "serial_candidates": self.cfg.serial_candidates,
            "prune_measured_candidates": self.cfg.prune_measured_candidates,
            "quantize_bin": self.cfg.quantize_bin,
            "perplexity_bin": self.cfg.perplexity_bin,
            "pure_quant": self.cfg.pure_quant,
            "pure_quant_effective": False,
            "pure_quant_note": "disabled for HillStepper tensor-type-file quantization because llama.cpp skips manual overrides under --pure",
            "commit_locks": self.cfg.commit_locks,
            "measurement_mode": "commit-locks" if self.cfg.commit_locks else "no-commit scan",
            "gpu_layers": self.cfg.gpu_layers,
            "ctx_size": self.cfg.ctx_size,
            "chunks": self.cfg.chunks,
            "imatrix": str(self.cfg.imatrix) if self.cfg.imatrix else None,
            "scratch_root": str(self.cfg.scratch_root) if self.cfg.scratch_root else None,
            "backup_root": str(self.cfg.backup_root) if self.cfg.backup_root else None,
            "max_temp_gb": self.cfg.max_temp_gb,
            "min_free_gb": self.cfg.min_free_gb,
            "hard_free_floor_gb": self.cfg.hard_free_floor_gb,
            "distrobox": self.cfg.distrobox,
            "quant_timeout": self.cfg.quant_timeout,
            "ppl_timeout": self.cfg.ppl_timeout,
            "token_embedding_type": self.cfg.token_embedding_type,
            "noise_pct": self.cfg.noise_pct,
            "acceptance_rule": f"lowest precision within {self.cfg.noise_pct:.4f}% of best PPL",
            "tie_break_rule": "lower precision on equal/near-equal PPL",
            "files": {
                "state": str(self.paths.state),
                "events": str(self.paths.events),
                "candidates": str(self.paths.candidates),
                "current_types": str(self.paths.current_types),
                "final_types": str(self.paths.final_types),
            },
        }
        atomic_write_json(self.paths.manifest, data)

    def discover_tensors(self) -> list[str]:
        if self.cfg.tensor_file:
            names = [line.strip() for line in self.cfg.tensor_file.read_text().splitlines() if line.strip()]
            return self.filter_tensors(names)
        try:
            from gguf import GGUFReader
        except ImportError as exc:
            raise SystemExit("gguf Python package is required unless --tensor-file is provided") from exc
        reader = GGUFReader(str(self.cfg.source_gguf))
        tensors: list[tuple[int, int, str]] = []
        priority = {
            "ffn_down": 0,
            "ffn_up": 1,
            "attn_v": 2,
            "attn_k": 3,
            "attn_q": 4,
            "attn_output": 5,
            "ffn_gate": 6,
        }
        for t in reader.tensors:
            name = t.name
            if not is_quantizable_tensor(name):
                continue
            if t.n_bytes < 1000 or "rope" in name or "embd" in name or "output_norm" in name:
                continue
            match = re.match(r"blk\.(\d+)\.(.+)\.weight", name)
            layer = int(match.group(1)) if match else -1
            ttype = match.group(2) if match else name.replace(".weight", "")
            tensors.append((layer, priority.get(ttype, 99), name))
        tensors.sort()
        return self.filter_tensors([name for _, _, name in tensors])

    def filter_tensors(self, names: list[str]) -> list[str]:
        filtered = [name for name in names if is_quantizable_tensor(name)]
        if self.cfg.layers is not None:
            filtered = [name for name in filtered if tensor_layer(name) in self.cfg.layers]
        if self.cfg.tensor_regex:
            pattern = re.compile(self.cfg.tensor_regex)
            filtered = [name for name in filtered if pattern.search(name)]
        return filtered

    def render_banner(self, tensors: int, locked: int) -> None:
        if self.cfg.plain:
            print(f"Cerebellum | {self.cfg.model_family}/{self.cfg.model_name}")
            print(f"run_dir={self.cfg.run_dir}")
            print(f"tensors={tensors} locked={locked} levels={','.join(self.cfg.levels)}")
            return
        enabled = self.cfg.color
        title = " Cerebellum "
        line = "+" + "-" * 72 + "+"
        print(color(line, "36;1", enabled))
        print(color("|" + title.center(72) + "|", "36;1", enabled))
        print(color("|" + f"{self.cfg.model_family}/{self.cfg.model_name}".center(72) + "|", "36;1", enabled))
        print(color(line, "36;1", enabled))
        print(f"Run dir : {self.cfg.run_dir}")
        print(f"Run id  : {self.cfg.run_id}")
        print(f"Levels  : {', '.join(self.cfg.levels)}")
        label = "locked" if self.cfg.commit_locks else "scanned"
        print(f"Tensors : {locked}/{tensors} {label}")
        print()

    def render_tensor_table(self, tensor: str, idx: int, total: int, baseline_ppl: float | None, rows: list[Candidate]) -> None:
        if self.cfg.plain:
            print(f"tensor={idx}/{total} {tensor} baseline_ppl={baseline_ppl}")
            return
        enabled = self.cfg.color
        print(color("-" * 92, "34", enabled))
        print(color(f"Tensor {idx}/{total}: {tensor}", "37;1", enabled))
        print(f"Baseline PPL: {baseline_ppl if baseline_ppl is not None else 'unknown'}")
        print("+--------+-----------+----------+----------+------------+------------+")
        print("| Quant  | Status    | Quant    | PPL      | PPL value  | Delta      |")
        print("+--------+-----------+----------+----------+------------+------------+")
        for c in rows:
            delta = "-"
            if c.ppl is not None and baseline_ppl is not None:
                delta = f"{c.ppl - baseline_ppl:+.2f}"
            status = c.status
            if c.status == "done":
                status = color("done", "32", enabled)
            elif c.status in {"quantizing", "ppl"}:
                status = color(c.status, "33", enabled)
            elif c.status == "failed":
                status = color("failed", "31", enabled)
            print(
                f"| {c.level:<6} | {status:<17} | {fmt_seconds(c.quant_seconds):<8} | "
                f"{fmt_seconds(c.ppl_seconds):<8} | {c.ppl if c.ppl is not None else '-':<10} | {delta:<10} |"
            )
        print("+--------+-----------+----------+----------+------------+------------+")

    def write_types(self, locked: dict[str, str], path: Path, extra: dict[str, str] | None = None) -> None:
        extra = extra or {}
        if self.cfg.tensor_file:
            names = [line.strip() for line in self.cfg.tensor_file.read_text().splitlines() if line.strip()]
            names = sorted(quantizable_tensor_names(set(names) | set(self.base_overrides) | set(locked) | set(extra)))
        else:
            try:
                from gguf import GGUFReader

                reader = GGUFReader(str(self.cfg.source_gguf))
                names = quantizable_tensor_names([t.name for t in reader.tensors])
            except Exception:
                names = sorted(quantizable_tensor_names(set(self.base_overrides) | set(locked) | set(extra)))
        if not names:
            names = sorted(quantizable_tensor_names(set(self.base_overrides) | set(locked) | set(extra)))
        merged = dict(self.base_overrides)
        merged.update(locked)
        merged.update(extra)
        lines = [tensor_type_line(name, merged.get(name, self.cfg.start_type)) for name in names]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def write_all_types_from_source(self, locked: dict[str, str], path: Path, extra: dict[str, str] | None = None) -> None:
        """Write a complete tensor map when a source GGUF is readable.

        Kept separate from write_types() so tests and explicit tensor-file runs
        can operate without parsing the GGUF.
        """
        try:
            from gguf import GGUFReader

            reader = GGUFReader(str(self.cfg.source_gguf))
            names = quantizable_tensor_names([t.name for t in reader.tensors])
        except Exception:
            names = sorted(quantizable_tensor_names(set(self.base_overrides) | set(locked) | set(extra or {})))
        merged = dict(self.base_overrides)
        merged.update(locked)
        merged.update(extra or {})
        lines = [tensor_type_line(name, merged.get(name, self.cfg.start_type)) for name in names]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def quantize_cmd(self, type_file: Path, outfile: Path) -> list[str]:
        cmd = [self.cfg.quantize_bin, "--allow-requantize"]
        # llama.cpp applies --tensor-type-file overrides only outside --pure.
        # A complete tensor map already forces the intended baseline type, so
        # do not combine --pure with per-tensor maps.
        if self.cfg.imatrix:
            cmd.extend(["--imatrix", str(self.cfg.imatrix)])
        if self.cfg.token_embedding_type:
            cmd.extend(["--token-embedding-type", self.cfg.token_embedding_type])
        cmd.extend(["--tensor-type-file", str(type_file), str(self.cfg.source_gguf), str(outfile), self.cfg.base_type])
        return cmd

    def ppl_cmd(self, model: Path) -> list[str]:
        cmd = [
            self.cfg.perplexity_bin,
            "--model",
            str(model),
            "--ctx-size",
            str(self.cfg.ctx_size),
            "-f",
            str(self.cfg.corpus),
            "-ngl",
            str(self.cfg.gpu_layers),
        ]
        if self.cfg.chunks is not None:
            cmd.extend(["--chunks", str(self.cfg.chunks)])
        return cmd

    def build_baseline_if_needed(self, state: dict[str, Any]) -> None:
        self.paths.artifacts.mkdir(parents=True, exist_ok=True)
        if self.paths.baseline.exists() and state.get("current_ppl") is not None and not state.get("baseline_invalid_after_rollback"):
            return
        self.events.write("baseline_quant_start", path=str(self.paths.baseline))
        self.write_types(state["locked"], self.paths.current_types)
        rc, output, seconds = run_external(
            self.quantize_cmd(self.paths.current_types, self.paths.baseline),
            self.cfg.quant_timeout,
            self.cfg.distrobox,
            heartbeat=lambda elapsed, pid: self.events.write(
                "baseline_quant_heartbeat",
                path=str(self.paths.baseline),
                elapsed_seconds=elapsed,
                child_pid=pid,
                size_bytes=path_size(self.paths.baseline),
            ),
        )
        self.events.write(
            "baseline_quant_finish",
            path=str(self.paths.baseline),
            returncode=rc,
            seconds=seconds,
            size_bytes=path_size(self.paths.baseline),
            output_tail=output[-2000:],
        )
        if rc != 0:
            raise SystemExit("baseline quantization failed; see events.jsonl")
        state.setdefault("start_quant_size_bytes", path_size(self.paths.baseline))
        state.setdefault("start_quant_path", str(self.paths.baseline))
        self.events.write("baseline_ppl_start", path=str(self.paths.baseline))
        rc, output, seconds = run_external(
            self.ppl_cmd(self.paths.baseline),
            self.cfg.ppl_timeout,
            self.cfg.distrobox,
            heartbeat=lambda elapsed, pid: self.events.write(
                "baseline_ppl_heartbeat",
                path=str(self.paths.baseline),
                elapsed_seconds=elapsed,
                child_pid=pid,
            ),
        )
        ppl, err = parse_ppl(output)
        self.events.write(
            "baseline_ppl_finish",
            path=str(self.paths.baseline),
            returncode=rc,
            seconds=seconds,
            ppl=ppl,
            ppl_error=err,
            output_tail=output[-2000:],
        )
        if rc != 0 or ppl is None:
            raise SystemExit("baseline PPL failed; see events.jsonl")
        state["current_ppl"] = ppl
        state.pop("baseline_invalid_after_rollback", None)
        state["totals"]["quant_seconds"] += seconds
        self.save_state(state, checkpoint=True)

    def choose_winner(self, baseline_level: str, baseline_ppl: float, candidates: list[Candidate]) -> tuple[str, float, Candidate | None, str]:
        measured = [c for c in candidates if c.ppl is not None]
        if not measured:
            return baseline_level, baseline_ppl, None, "no successful candidates"
        best_measured_ppl = min([baseline_ppl, *[c.ppl for c in measured if c.ppl is not None]])
        tolerance = abs(best_measured_ppl) * (self.cfg.noise_pct / 100.0)
        choices: list[tuple[int, float, str, Candidate | None]] = []
        if baseline_ppl <= best_measured_ppl + tolerance:
            choices.append((PRECISION_RANK.get(baseline_level, 999), baseline_ppl, baseline_level, None))
        for c in candidates:
            if c.ppl is None:
                continue
            rank = PRECISION_RANK.get(c.level, 999)
            if c.ppl <= best_measured_ppl + tolerance:
                choices.append((rank, c.ppl, c.level, c))
        choices.sort(key=lambda row: (row[0], row[1]))
        rank, ppl, level, candidate = choices[0]
        reason = f"lowest precision within {self.cfg.noise_pct:.4f}% noise window of best PPL {best_measured_ppl:.6f}"
        return level, ppl, candidate, reason

    def test_tensor(self, state: dict[str, Any], tensor: str, idx: int, total: int) -> None:
        baseline_ppl = state.get("current_ppl")
        if baseline_ppl is None:
            raise SystemExit("state has no current baseline PPL")
        safe_tensor = slug(tensor)
        tensor_tmp = self.paths.tmp / f"{idx:05d}-{safe_tensor}"
        if tensor_tmp.exists():
            shutil.rmtree(tensor_tmp)
        tensor_tmp.mkdir(parents=True, exist_ok=True)
        candidates = [
            Candidate(
                tensor=tensor,
                level=level,
                index=i,
                type_file=tensor_tmp / f"{i:02d}-{level}.types.txt",
                gguf_path=tensor_tmp / f"{i:02d}-{level}.gguf",
            )
            for i, level in enumerate(self.cfg.levels)
        ]
        self.events.write("tensor_start", tensor=tensor, index=idx, total=total, baseline_ppl=baseline_ppl)
        self.render_tensor_table(tensor, idx, total, baseline_ppl, candidates)

        # In normal mode, quantize should stay ahead of PPL until disk pressure
        # says another candidate would violate the configured free-space floor.
        queue_depth = 1 if (self.cfg.low_space or self.cfg.serial_candidates) else max(2, len(candidates))
        ready: queue.Queue[Candidate | None] = queue.Queue(maxsize=queue_depth)
        results: list[Candidate] = []
        quant_done = threading.Event()

        def quant_worker() -> None:
            for c in candidates:
                if self.stop_requested:
                    break
                estimated_candidate_gb = max(bytes_to_gb(path_size(self.paths.baseline)), 1.0)
                required_free = max(self.cfg.min_free_gb, self.cfg.hard_free_floor_gb + estimated_candidate_gb)
                while disk_free_gb(self.paths.tmp) < required_free and not self.stop_requested:
                    self.events.write(
                        "disk_wait",
                        free_gb=disk_free_gb(self.paths.tmp),
                        min_free_gb=required_free,
                        hard_floor_gb=self.cfg.hard_free_floor_gb,
                        estimated_candidate_gb=estimated_candidate_gb,
                    )
                    time.sleep(15)
                c.status = "quantizing"
                c.quant_started_at = utc_now()
                tmp_gguf = c.gguf_path.with_suffix(c.gguf_path.suffix + ".tmp")
                if tmp_gguf.exists():
                    tmp_gguf.unlink()
                self.events.write("quant_start", tensor=tensor, level=c.level, output=str(c.gguf_path), tmp_output=str(tmp_gguf))
                self.write_types(state["locked"], c.type_file, {tensor: c.level})
                try:
                    rc, output, seconds = run_external(
                        self.quantize_cmd(c.type_file, tmp_gguf),
                        self.cfg.quant_timeout,
                        self.cfg.distrobox,
                        heartbeat=lambda elapsed, pid, c=c, tmp_gguf=tmp_gguf: self.events.write(
                            "quant_heartbeat",
                            tensor=tensor,
                            level=c.level,
                            elapsed_seconds=elapsed,
                            child_pid=pid,
                            tmp_output=str(tmp_gguf),
                            size_bytes=path_size(tmp_gguf),
                        ),
                    )
                except subprocess.TimeoutExpired as exc:
                    rc, output, seconds = 124, str(exc), float(self.cfg.quant_timeout)
                c.quant_finished_at = utc_now()
                c.quant_seconds = seconds
                c.quant_returncode = rc
                c.quant_output_tail = output[-2000:]
                if rc == 0 and tmp_gguf.exists() and path_size(tmp_gguf) > 0:
                    os.replace(tmp_gguf, c.gguf_path)
                elif tmp_gguf.exists():
                    tmp_gguf.unlink()
                c.quant_ok = rc == 0 and c.gguf_path.exists() and path_size(c.gguf_path) > 0
                c.status = "queued" if c.quant_ok else "failed"
                self.events.write(
                    "quant_finish",
                    tensor=tensor,
                    level=c.level,
                    returncode=rc,
                    seconds=seconds,
                    ok=c.quant_ok,
                    size_bytes=path_size(c.gguf_path),
                    output_tail=c.quant_output_tail,
                )
                if c.quant_ok:
                    ready.put(c)
                    if self.cfg.low_space or self.cfg.serial_candidates:
                        while c not in results and not self.stop_requested:
                            time.sleep(0.25)
                else:
                    results.append(c)
            quant_done.set()
            ready.put(None)

        def ppl_worker() -> None:
            while True:
                c = ready.get()
                if c is None:
                    return
                if self.stop_requested:
                    return
                c.status = "ppl"
                c.ppl_started_at = utc_now()
                self.events.write("ppl_start", tensor=tensor, level=c.level, model=str(c.gguf_path))
                try:
                    rc, output, seconds = run_external(
                        self.ppl_cmd(c.gguf_path),
                        self.cfg.ppl_timeout,
                        self.cfg.distrobox,
                        heartbeat=lambda elapsed, pid, c=c: self.events.write(
                            "ppl_heartbeat",
                            tensor=tensor,
                            level=c.level,
                            elapsed_seconds=elapsed,
                            child_pid=pid,
                            model=str(c.gguf_path),
                        ),
                    )
                except subprocess.TimeoutExpired as exc:
                    rc, output, seconds = 124, str(exc), float(self.cfg.ppl_timeout)
                ppl, err = parse_ppl(output)
                c.ppl_finished_at = utc_now()
                c.ppl_seconds = seconds
                c.ppl_returncode = rc
                c.ppl_output_tail = output[-2000:]
                c.ppl = ppl
                c.ppl_error = err
                c.status = "done" if rc == 0 and ppl is not None else "failed"
                self.events.write(
                    "ppl_finish",
                    tensor=tensor,
                    level=c.level,
                    returncode=rc,
                    seconds=seconds,
                    ppl=ppl,
                    ppl_error=err,
                    delta=(ppl - baseline_ppl) if ppl is not None else None,
                    output_tail=c.ppl_output_tail,
                )
                self.candidate_log.write(
                    "candidate",
                    tensor=tensor,
                    level=c.level,
                    baseline_ppl=baseline_ppl,
                    ppl=ppl,
                    ppl_error=err,
                    delta=(ppl - baseline_ppl) if ppl is not None else None,
                    quant_seconds=c.quant_seconds,
                    ppl_seconds=c.ppl_seconds,
                    size_bytes=path_size(c.gguf_path),
                    status=c.status,
                )
                results.append(c)
                if (self.cfg.low_space or self.cfg.prune_measured_candidates) and not self.cfg.keep_losers:
                    baseline_level = self.base_overrides.get(tensor, self.cfg.start_type)
                    _level, _ppl, current_best, _reason = self.choose_winner(baseline_level, baseline_ppl, results)
                    survivor = tensor_tmp / "best-so-far.gguf"
                    for done in results:
                        if done is current_best and done.gguf_path.exists():
                            if done.gguf_path != survivor:
                                if survivor.exists():
                                    survivor.unlink()
                                os.replace(done.gguf_path, survivor)
                                done.gguf_path = survivor
                            continue
                        if done.gguf_path.exists() and not self.cfg.keep_losers:
                            done.gguf_path.unlink()
                self.render_tensor_table(tensor, idx, total, baseline_ppl, candidates)

        tq = threading.Thread(target=quant_worker, name="cerebellum-quant", daemon=True)
        tp = threading.Thread(target=ppl_worker, name="cerebellum-ppl", daemon=True)
        tq.start()
        tp.start()
        tq.join()
        tp.join()

        if self.stop_requested:
            self.events.write("tensor_interrupted", tensor=tensor)
            state["run_status"] = "stopped"
            state["stopped_at"] = utc_now()
            state["stop_reason"] = "signal"
            self.save_state(state, checkpoint=True)
            raise SystemExit("stop requested; state saved")

        baseline_level = self.base_overrides.get(tensor, self.cfg.start_type)
        best_level, best_ppl, best_candidate, winner_reason = self.choose_winner(baseline_level, baseline_ppl, candidates)
        if self.cfg.commit_locks:
            state["locked"][tensor] = best_level
            state["current_ppl"] = best_ppl
        state["last_tensor"] = tensor
        state["tested"].append(
            {
                "tensor": tensor,
                "winner": best_level,
                "ppl": best_ppl,
                "baseline_ppl": baseline_ppl,
                "finished_at": utc_now(),
                "reason": winner_reason,
                "committed": self.cfg.commit_locks,
            }
        )
        for c in candidates:
            state["totals"]["quant_seconds"] += c.quant_seconds or 0
            state["totals"]["ppl_seconds"] += c.ppl_seconds or 0
            state["totals"]["candidates"] += 1
            if c.status == "failed":
                state["totals"]["failures"] += 1

        self.write_types(state["locked"], self.paths.current_types)
        old_baseline = self.paths.baseline
        if self.cfg.commit_locks:
            if best_candidate is not None and best_candidate.gguf_path.exists():
                old_backup = old_baseline.with_suffix(".previous.gguf")
                if old_baseline.exists():
                    os.replace(old_baseline, old_backup)
                os.replace(best_candidate.gguf_path, old_baseline)
                if old_backup.exists():
                    old_backup.unlink()
            elif best_candidate is not None:
                tmp_baseline = old_baseline.with_suffix(old_baseline.suffix + ".tmp")
                if tmp_baseline.exists():
                    tmp_baseline.unlink()
                self.events.write("baseline_rebuild_start", path=str(old_baseline), reason="winning candidate artifact missing")
                rc, output, seconds = run_external(
                    self.quantize_cmd(self.paths.current_types, tmp_baseline),
                    self.cfg.quant_timeout,
                    self.cfg.distrobox,
                    heartbeat=lambda elapsed, pid: self.events.write(
                        "baseline_rebuild_heartbeat",
                        path=str(old_baseline),
                        elapsed_seconds=elapsed,
                        child_pid=pid,
                        size_bytes=path_size(tmp_baseline),
                    ),
                )
                self.events.write(
                    "baseline_rebuild_finish",
                    path=str(old_baseline),
                    returncode=rc,
                    seconds=seconds,
                    size_bytes=path_size(tmp_baseline),
                    output_tail=output[-2000:],
                )
                if rc != 0 or not tmp_baseline.exists() or path_size(tmp_baseline) <= 0:
                    if tmp_baseline.exists():
                        tmp_baseline.unlink()
                    raise SystemExit("baseline rebuild failed after tensor lock; see events.jsonl")
                old_backup = old_baseline.with_suffix(".previous.gguf")
                if old_baseline.exists():
                    os.replace(old_baseline, old_backup)
                os.replace(tmp_baseline, old_baseline)
                if old_backup.exists():
                    old_backup.unlink()
        self.save_state(state, checkpoint=(len(state["tested"]) % self.cfg.backup_every == 0))
        event_name = "tensor_locked" if self.cfg.commit_locks else "tensor_scanned"
        self.events.write(event_name, tensor=tensor, winner=best_level, ppl=best_ppl, baseline_ppl=baseline_ppl, reason=winner_reason)

        for c in candidates:
            if c is best_candidate and self.cfg.keep_winners and self.cfg.commit_locks:
                winner_dir = self.paths.artifacts / "winners"
                winner_dir.mkdir(parents=True, exist_ok=True)
                meta = {
                    "tensor": tensor,
                    "level": c.level,
                    "ppl": c.ppl,
                    "quant_seconds": c.quant_seconds,
                    "ppl_seconds": c.ppl_seconds,
                    "source_candidate": str(c.gguf_path),
                    "baseline_path": str(self.paths.baseline),
                }
                atomic_write_json(winner_dir / f"{idx:05d}-{safe_tensor}.json", meta)
                continue
            if not self.cfg.keep_losers and c.gguf_path.exists():
                c.gguf_path.unlink()
        if tensor_tmp.exists() and not self.cfg.keep_losers:
            shutil.rmtree(tensor_tmp, ignore_errors=True)

    def run(self) -> None:
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.artifacts.mkdir(parents=True, exist_ok=True)
        self.paths.tmp.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints.mkdir(parents=True, exist_ok=True)
        self.write_manifest()
        state = self.load_state()
        tensors = self.discover_tensors()
        completed_key = "locked" if self.cfg.commit_locks else "tested"
        completed_tensors = (
            set(state.get("locked", {}))
            if self.cfg.commit_locks
            else {row.get("tensor") for row in state.get("tested", []) if row.get("tensor")}
        )
        remaining = [t for t in tensors if t not in completed_tensors]
        state["run_status"] = "running"
        self.save_state(state)
        removed_markers = clear_terminal_markers(self.cfg.run_dir)
        self.render_banner(len(tensors), len(completed_tensors))
        self.events.write(
            "run_start",
            tensors=len(tensors),
            locked=len(state["locked"]),
            scanned=len(state.get("tested", [])) if not self.cfg.commit_locks else None,
            commit_locks=self.cfg.commit_locks,
            cleared_markers=removed_markers,
        )
        self.build_baseline_if_needed(state)
        for tensor in remaining:
            if self.stop_requested:
                break
            idx = tensors.index(tensor) + 1
            self.test_tensor(state, tensor, idx, len(tensors))
        completed_count = len(state.get(completed_key, {}))
        state["run_status"] = "complete" if completed_count == len(tensors) else "stopped"
        self.write_types(state["locked"], self.paths.final_types)
        atomic_write_json(self.paths.timing, state["totals"])
        self.save_state(state, checkpoint=True)
        marker = self.cfg.run_dir / ("COMPLETE" if state["run_status"] == "complete" else "ABORTED")
        tmp_marker = marker.with_suffix(".tmp")
        tmp_marker.write_text(utc_now() + "\n")
        os.replace(tmp_marker, marker)
        self.events.write("run_finish", status=state["run_status"], locked=len(state["locked"]), scanned=len(state.get("tested", [])), tensors=len(tensors), commit_locks=self.cfg.commit_locks)


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    root = Path(args.data_root) if args.data_root else default_data_root()
    family = slug(args.family or "unknown-family")
    model = slug(args.model_name or Path(args.source_gguf).stem)
    source = slug(args.source_name or Path(args.source_gguf).stem)
    run_name = slug(args.run_name or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_cerebellum_{args.base_type.lower()}_{source}")
    return root / "families" / family / model / "sources" / source / "runs" / run_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cerebellum quantization toolbox")
    sub = parser.add_subparsers(dest="cmd")
    home = sub.add_parser("home", help="show the Cerebellum local menu and recent run summary")
    home.add_argument("--limit", type=int, default=8)
    home.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="DEPRECATED — see `cerebellum method`; hillstep per-tensor search (only legal as a targeted pass after a group-first scan)")
    run.add_argument("--source-gguf", required=True)
    run.add_argument("--corpus", default=None, help="PPL/calibration corpus path; optional when --profile resolves locally")
    run.add_argument(
        "--profile",
        choices=["wiki", "agentic", "code", "math", "dialogue", "all-around", "custom"],
        default="custom",
        help="Named PPL target profile recorded with the run",
    )
    run.add_argument(
        "--metric",
        choices=["ppl", "humaneval", "arc", "mmlu", "tool-call", "dialogue"],
        default="ppl",
        help="Ablation scoring metric; only ppl is executable until task scorer adapters land",
    )
    run.add_argument("--family", default=None)
    run.add_argument("--model-name", default=None)
    run.add_argument("--source-name", default=None)
    run.add_argument("--data-root", default=None)
    run.add_argument("--run-name", default=None)
    run.add_argument("--run-dir", default=None)
    run.add_argument("--tensor-file", default=None)
    run.add_argument("--base-map", default=None, help="Seed every generated tensor map from an existing Cerebellum tensor-type file")
    run.add_argument("--layers", default=None, help="Target only layer numbers, e.g. 0,1,8-12")
    run.add_argument("--tensor-regex", default=None, help="Target only tensors matching this regex")
    run.add_argument("--scratch-root", default=None, help="Large GGUF artifact/temp root, separate from metadata run dir")
    run.add_argument("--backup-root", default=None, help="Mirror critical run metadata/checkpoints to this separate root")
    run.add_argument("--base-type", default="Q4_K_M")
    run.add_argument("--start-type", default="q4_K")
    run.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    run.add_argument("--imatrix", default=None)
    run.add_argument("--quantize-bin", default=DEFAULT_QUANTIZE)
    run.add_argument("--perplexity-bin", default=DEFAULT_PERPLEXITY)
    run.add_argument("--pure-quant", action="store_true", help="Pass --pure to llama-quantize so mixed presets do not promote selected tensors")
    run.add_argument("--gpu-layers", type=int, default=99)
    run.add_argument("--ctx-size", type=int, default=2048)
    run.add_argument("--chunks", type=int, default=None)
    run.add_argument("--max-temp-gb", type=float, default=80.0, help="Legacy temp budget marker recorded for compatibility")
    run.add_argument("--min-free-gb", type=float, default=40.0, help="Minimum free GB required before launching next quant")
    run.add_argument("--hard-free-floor-gb", type=float, default=10.0, help="Never launch another quant job below this free-space floor")
    run.add_argument("--distrobox", default=None, help="Run llama.cpp commands inside this distrobox")
    run.add_argument("--quant-timeout", type=int, default=1800)
    run.add_argument("--ppl-timeout", type=int, default=900)
    run.add_argument("--keep-losers", action="store_true")
    run.add_argument("--no-keep-winners", action="store_true")
    run.add_argument("--no-commit-locks", dest="commit_locks", action="store_false", help="Measure candidate tensors without carrying winners into later tensors")
    run.set_defaults(commit_locks=True)
    run.add_argument("--low-space", action="store_true", help="Serialize candidate testing and prune measured GGUFs immediately")
    run.add_argument("--serial-candidates", action="store_true", help="Do not let CPU quantization run ahead of GPU PPL")
    run.add_argument("--prune-measured-candidates", action="store_true", default=True, help="Delete measured non-winning candidate GGUFs during each tensor")
    run.add_argument("--keep-measured-candidates", dest="prune_measured_candidates", action="store_false", help="Keep measured candidate GGUFs until tensor end")
    run.add_argument("--plain", action="store_true")
    run.add_argument("--no-color", action="store_true")
    run.add_argument("--backup-every", type=int, default=1)
    run.add_argument("--token-embedding-type", default="f16")
    run.add_argument("--noise-pct", type=float, default=0.0)

    sub.add_parser("imatrix", help="generate a Cerebellum/llama.cpp imatrix for quantization")

    status = sub.add_parser("status", help="show Cerebellum run status")
    status.add_argument("run_dir", nargs="?", help="run directory or model/run name; defaults to the only active run")
    status.add_argument("--plain", action="store_true")
    status.add_argument("--no-color", action="store_true")

    events = sub.add_parser("events", help="print run events")
    events.add_argument("run_dir")
    events.add_argument("--type", default=None)
    events.add_argument("--limit", type=int, default=50)
    events.add_argument("--json", action="store_true")

    watch = sub.add_parser("watch", help="open the Cerebellum live terminal interface")
    watch.add_argument("run_dir", nargs="?", help="run directory or model/run name; defaults to the only active run")
    watch.add_argument("--interval", type=float, default=5.0)
    watch.add_argument("--once", action="store_true", help="render one frame and exit")
    watch.add_argument("--stall-warn-seconds", type=float, default=300.0)
    watch.add_argument("--stall-fail-seconds", type=float, default=900.0)
    watch.add_argument("--events-limit", type=int, default=DEFAULT_WATCH_EVENTS_LIMIT, help="event rows to show; 0 shows all")
    watch.add_argument("--measurements-limit", type=int, default=0, help="measurement rows to show; 0 shows all")
    watch.add_argument("--tui", action="store_true", help="open scrollable interactive terminal UI")
    watch.add_argument("--public", action="store_true", help="render a screenshot-safe view without tensor, candidate, path, or event details")
    watch.add_argument("--plain", action="store_true")
    watch.add_argument("--no-color", action="store_true")

    stop = sub.add_parser("stop", help="stop or repair a Cerebellum run state")
    stop.add_argument("run_dir")
    stop.add_argument("--reason", default="user")
    stop.add_argument("--no-kill", action="store_true", help="only mark state stopped; do not signal a process")

    resume = sub.add_parser("resume", help="resume an existing run from its manifest/state")
    resume.add_argument("run_dir")
    resume.add_argument("--low-space", action="store_true", help="resume with serialized/pruned low-space candidate flow")
    resume.add_argument("--no-low-space", action="store_true", help="override a saved low-space run and resume with overlapped candidate quant/PPL workers")
    resume.add_argument("--min-free-gb", type=float, default=None)
    resume.add_argument("--hard-free-floor-gb", type=float, default=None)
    resume.add_argument("--backup-root", default=None)
    resume.add_argument("--distrobox", default=None)
    resume.add_argument("--plain", action="store_true")
    resume.add_argument("--no-color", action="store_true")

    cleanup = sub.add_parser("cleanup", help="clean safe temp/artifact files without deleting durable progress")
    cleanup.add_argument("run_dir")
    cleanup.add_argument("--yes", action="store_true", help="execute deletion; default is dry-run")
    cleanup.add_argument("--old-artifacts", action="store_true", help="also delete artifacts from stopped/aborted sibling runs")
    cleanup.add_argument("--partials", action="store_true", help="delete partial temp dirs for tensors that are not locked")
    cleanup.add_argument("--force", action="store_true", help="allow partial cleanup even if the run appears active")

    rollback = sub.add_parser("rollback", help="rollback durable state to a clean tensor/layer boundary")
    rollback.add_argument("run_dir")
    rollback.add_argument("--to-locked", type=int, default=None, help="Keep only the first N locked/tested tensors")
    rollback.add_argument("--before-layer", type=int, default=None, help="Remove all locked/tested tensors at this layer and above")
    rollback.add_argument("--last-completed-layer", action="store_true", help="Remove the newest partial layer from state")
    rollback.add_argument("--yes", action="store_true", help="write the rollback; default is dry-run")
    rollback.add_argument("--force", action="store_true", help="allow rollback even if the runner appears active")

    backup = sub.add_parser("backup", help="copy critical run metadata/checkpoints to a backup root")
    backup.add_argument("run_dir")
    backup.add_argument("--to", required=True)

    recover = sub.add_parser("recover", help="print a crash-recovery plan for a run")
    recover.add_argument("run_dir", nargs="?", help="run directory or model/run name; defaults to the only active run")
    recover.add_argument("--json", action="store_true")

    runs = sub.add_parser("runs", help="list known runs under a data root")
    runs.add_argument("--data-root", "--root", dest="data_root", default=None)
    runs.add_argument("--family", default=None)
    runs.add_argument("--model", default=None)
    runs.add_argument("--status", default=None)
    runs.add_argument("--profile", default=None)
    runs.add_argument("--json", action="store_true")

    project = sub.add_parser("project", help="inspect Cerebellum model projects")
    project.add_argument("--data-root", "--root", dest="data_root", default=None)
    project.add_argument("--family", default=None)
    project.add_argument("--model", default=None)
    project.add_argument("--source", default=None)
    project.add_argument("--json", action="store_true")

    provenance = sub.add_parser("provenance", help="inspect or generate Cerebellum GGUF provenance metadata")
    provenance.add_argument("--gguf", default=None, help="GGUF to inspect for existing metadata")
    provenance.add_argument("--run-dir", default=None, help="Cerebellum run directory used to generate metadata")
    provenance.add_argument("--hash-files", action="store_true", help="compute full SHA256 hashes for large files")
    provenance.add_argument("--private", action="store_true", help="include private run IDs, raw PPL, tensor-map hashes, and event/candidate sidecar names")
    provenance.add_argument("--format", choices=["json", "env"], default="json")

    inspect_types = sub.add_parser("inspect-gguf-types", help="summarize GGUF tensor quantization types")
    inspect_types.add_argument("gguf")
    inspect_types.add_argument("--by-layer", action="store_true", help="include per-layer type counts")
    inspect_types.add_argument("--by-component", action="store_true", help="include per-component type counts")
    inspect_types.add_argument("--json", action="store_true")

    compare_types = sub.add_parser("compare-gguf-types", help="compare tensor quantization type distributions between two GGUFs")
    compare_types.add_argument("baseline")
    compare_types.add_argument("candidate")
    compare_types.add_argument("--baseline-label", default="baseline")
    compare_types.add_argument("--candidate-label", default="candidate")
    compare_types.add_argument("--reference-map", default=None, help="optional tensor-type file to compare the candidate against")
    compare_types.add_argument("--json", action="store_true")

    compare_locks = sub.add_parser("compare-locks", help="compare Cerebellum tensor locks between a run and an archive/state")
    compare_locks.add_argument("run_dir", help="current run directory or model/run name")
    compare_locks.add_argument("--against", required=True, help="state.json, checkpoint JSON, or archive directory containing state.json")
    compare_locks.add_argument("--json", action="store_true")

    finalize = sub.add_parser("finalize", help="write final reports/model card and tag GGUF provenance")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--gguf", default=None, help="Final GGUF to tag/inspect")
    finalize.add_argument("--repo-name", default=None, help="Optional HF/GitHub repo name for model-card text")
    finalize.add_argument("--output-dir", default=None, help="Defaults to RUN_DIR/finalize")
    finalize.add_argument("--hash-files", action="store_true")
    finalize.add_argument("--private", action="store_true", help="write private finalize sidecars with run IDs, raw PPL, and factory hashes")
    finalize.add_argument("--inject", action="store_true", help="Inject visible cerebellum.* metadata into --gguf when supported")
    finalize.add_argument("--metadata-tool", default=None, help="Path to gguf-set-metadata compatible tool")
    finalize.add_argument("--json", action="store_true")

    package = sub.add_parser("package", help="write portable upload/package manifest for a run")
    package.add_argument("run_dir")
    package.add_argument("--output", default=None)
    package.add_argument("--private", action="store_true", help="include raw factory sidecars such as state, events, candidates, decisions, and tensor maps")
    package.add_argument("--json", action="store_true")

    public_audit = sub.add_parser("public-audit", help="scan files for public-release safety risks")
    public_audit.add_argument("paths", nargs="*", help="files/directories to scan; defaults to tracked files")
    public_audit.add_argument("--json", action="store_true")
    public_audit.add_argument("--max-bytes", type=int, default=512_000, help="max bytes to read from each file")

    public_history = sub.add_parser("public-history-audit", help="scan Git history paths for public-release safety risks")
    public_history.add_argument("--root", default=".", help="git repository root to scan")
    public_history.add_argument("--ref", action="append", help="git ref/range to scan; defaults to --all")
    public_history.add_argument("--json", action="store_true")

    public_export = sub.add_parser("public-export", help="copy release-safe Cerebellum files into a sanitized public tree")
    public_export.add_argument("output_dir")
    public_export.add_argument("paths", nargs="*", help="files/directories to export; defaults to tracked public-safe files")
    public_export.add_argument("--clean", action="store_true", help="remove output_dir before writing")
    public_export.add_argument("--dry-run", action="store_true")
    public_export.add_argument("--json", action="store_true")
    public_export.add_argument("--max-bytes", type=int, default=512_000, help="max bytes to audit from each file")

    release_gate = sub.add_parser("release-gate", help="gate public/private release candidates before pushing")
    release_gate.add_argument("paths", nargs="*", help="files/directories to gate; defaults to tracked public-safe files")
    release_gate.add_argument("--remote", default="origin", help="target git remote name; origin is public-strict, dev is private/advisory")
    release_gate.add_argument("--benchmark-results", action="append", default=[], help="benchmark result directory/file; may be repeated")
    release_gate.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="release")
    release_gate.add_argument("--model", default=None)
    release_gate.add_argument("--require-benchmarks", action="store_true", help="fail when selected suite measured benchmark JSONs are incomplete")
    release_gate.add_argument("--max-bytes", type=int, default=512_000, help="max bytes to audit from each file")
    release_gate.add_argument("--json", action="store_true")

    inventory = sub.add_parser("artifact-inventory", help="inventory legacy Cerebellum artifacts without deleting anything")
    inventory.add_argument("root", nargs="?", default=".", help="workspace root to scan")
    inventory.add_argument("--output", default=None, help="optional JSON output path")
    inventory.add_argument("--markdown", default=None, help="optional Markdown output path")
    inventory.add_argument("--top", type=int, default=25, help="number of largest buckets/files to show")
    inventory.add_argument("--json", action="store_true")

    history = sub.add_parser("history", help="build/search a browsable Cerebellum model history index")
    history.add_argument("--root", action="append", default=[], help="root to scan; may be repeated; defaults to repo and game-drive run index")
    history.add_argument("--query", default=None, help="filter files by model/method/script/path text")
    history.add_argument("--limit", type=int, default=5000, help="max interesting files to inspect")
    history.add_argument("--include-chat-logs", action="store_true", help="also scan known local Codex/Claude/OpenCode/Gemini session roots")
    history.add_argument("--chat-root", action="append", default=[], help="extra/explicit chat-session root to scan; may be repeated")
    history.add_argument("--output", default=None, help="write searchable JSON index")
    history.add_argument("--markdown", default=None, help="write Markdown history report")
    history.add_argument("--html", default=None, help="write browser-searchable HTML view")
    history.add_argument("--json", action="store_true")

    schedule = sub.add_parser("schedule", help="run multiple Cerebellum jobs from a JSON schedule")
    schedule.add_argument("--file", default=None)
    schedule.add_argument("--template", action="store_true", help="print an example schedule JSON")
    schedule.add_argument("--dry-run", action="store_true", help="validate and print jobs without running them")

    queue_parser = sub.add_parser("queue", help="manage queued Cerebellum jobs")
    queue_parser.add_argument("--db", default=DEFAULT_DB)
    queue_parser.add_argument("--json", action="store_true")
    queue_sub = queue_parser.add_subparsers(dest="queue_cmd", required=True)
    queue_add = queue_sub.add_parser("add")
    queue_add.add_argument("--kind", choices=["pipeline", "benchmark", "run"], required=True)
    queue_add.add_argument("--label", default=None)
    queue_add.add_argument("--manifest", default=None, help="pipeline-plan JSON manifest for pipeline jobs")
    queue_add.add_argument("--command", default=None, help="explicit command for benchmark/run jobs")
    queue_add.add_argument("--from-phase", default=None, help="for pipeline jobs, start execution at this phase")
    queue_add.add_argument("--until-phase", default=None, help="for pipeline jobs, stop execution after this phase")
    queue_add.add_argument("--payload-json", default=None, help="extra JSON object merged into the queued payload")
    queue_add.add_argument("--priority", type=int, default=100)
    queue_add.add_argument("--status", default="queued")
    queue_add.add_argument("--notes", default=None)
    queue_list = queue_sub.add_parser("list")
    queue_list.add_argument("--status", default=None)
    queue_list.add_argument("--kind", choices=["pipeline", "benchmark", "run"], default=None)
    queue_list.add_argument("--limit", type=int, default=50)
    queue_get = queue_sub.add_parser("get")
    queue_get.add_argument("id", type=int)
    queue_get.add_argument("--tail", type=int, default=0, help="include the last N log lines when the job has a log")
    queue_cancel = queue_sub.add_parser("cancel")
    queue_cancel.add_argument("id", type=int)
    queue_cancel.add_argument("--reason", default=None)
    queue_retry = queue_sub.add_parser("retry")
    queue_retry.add_argument("id", type=int)
    queue_retry.add_argument("--priority", type=int, default=None)
    queue_retry.add_argument("--notes", default=None)
    queue_run_next = queue_sub.add_parser("run-next")
    queue_run_next.add_argument("--kind", choices=["pipeline", "benchmark", "run"], default=None)
    queue_run_next.add_argument("--status", default="queued")
    queue_run_next.add_argument("--execute", action="store_true", help="actually run the selected job; default is dry-run")

    pipeline_plan_parser = sub.add_parser("pipeline-plan", help="write a full Cerebellum pipeline command manifest")
    pipeline_plan_parser.add_argument("--source-gguf", required=True)
    pipeline_plan_parser.add_argument("--output-dir", required=True)
    pipeline_plan_parser.add_argument("--imatrix", default=None, help="existing or planned imatrix path; defaults to OUTPUT_DIR/imatrix.dat")
    pipeline_plan_parser.add_argument("--corpus", default=None)
    pipeline_plan_parser.add_argument("--profile", choices=["wiki", "agentic", "code", "math", "dialogue", "all-around", "custom"], default="custom")
    pipeline_plan_parser.add_argument("--family", default=None)
    pipeline_plan_parser.add_argument("--model-name", default=None)
    pipeline_plan_parser.add_argument("--source-name", default=None)
    pipeline_plan_parser.add_argument("--run-name", default=None)
    pipeline_plan_parser.add_argument("--run-dir", default=None)
    pipeline_plan_parser.add_argument("--data-root", default=None)
    pipeline_plan_parser.add_argument("--scratch-root", default=None)
    pipeline_plan_parser.add_argument("--base-type", default="Q4_K_M")
    pipeline_plan_parser.add_argument("--start-type", default="q4_K")
    pipeline_plan_parser.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    pipeline_plan_parser.add_argument("--quantize-bin", default=DEFAULT_QUANTIZE)
    pipeline_plan_parser.add_argument("--perplexity-bin", default=DEFAULT_PERPLEXITY)
    pipeline_plan_parser.add_argument("--pure-quant", action="store_true", help="Pass --pure to llama-quantize in generated commands")
    pipeline_plan_parser.add_argument("--gpu-layers", type=int, default=99)
    pipeline_plan_parser.add_argument("--ctx-size", type=int, default=2048)
    pipeline_plan_parser.add_argument("--chunks", type=int, default=None)
    pipeline_plan_parser.add_argument("--distrobox", default=None)
    pipeline_plan_parser.add_argument("--low-space", action="store_true")
    pipeline_plan_parser.add_argument("--benchmark-suite", choices=sorted(BENCHMARK_SUITES), default="release-local")
    pipeline_plan_parser.add_argument(
        "--metric",
        choices=["ppl", "humaneval", "arc", "mmlu", "tool-call", "dialogue"],
        default=None,
        help="ablation scoring metric to record and pass to Cerebellum run",
    )
    pipeline_plan_parser.add_argument("--task-profile", choices=sorted(TASK_PROFILES), default=None, help="task-specific variant profile to annotate and default profile/suite")
    pipeline_plan_parser.add_argument("--benchmark-port", type=int, default=8084)
    pipeline_plan_parser.add_argument("--repo-name", default=None)
    pipeline_plan_parser.add_argument("--write", default=None, help="write JSON manifest to this path")
    pipeline_plan_parser.add_argument("--json", action="store_true")

    pipeline_run_parser = sub.add_parser("pipeline-run", help="validate or execute a Cerebellum pipeline manifest")
    pipeline_run_parser.add_argument("--manifest", required=True, help="pipeline-plan JSON manifest")
    pipeline_run_parser.add_argument("--from-phase", default=None, help="start at this phase name")
    pipeline_run_parser.add_argument("--until-phase", default=None, help="stop after this phase name")
    pipeline_run_parser.add_argument("--execute", action="store_true", help="actually run phase commands; dry-run is default")
    pipeline_run_parser.add_argument("--json", action="store_true")

    pipeline_status_parser = sub.add_parser("pipeline-status", help="summarize pipeline-run event logs and resume point")
    pipeline_status_parser.add_argument("--manifest", required=True, help="pipeline-plan JSON manifest")
    pipeline_status_parser.add_argument("--events", default=None, help="pipeline_run_events.jsonl path; defaults next to manifest")
    pipeline_status_parser.add_argument("--json", action="store_true")

    cpu_offload_smoke = sub.add_parser("cpu-offload-smoke", help="validate huge-model CPU-offload workflow without launching quantization")
    cpu_offload_smoke.add_argument("--source-gguf", required=True)
    cpu_offload_smoke.add_argument("--output-dir", required=True)
    cpu_offload_smoke.add_argument("--model-name", default="GLM-5.1")
    cpu_offload_smoke.add_argument("--scratch-root", default=None)
    cpu_offload_smoke.add_argument("--benchmark-port", type=int, default=8084)
    cpu_offload_smoke.add_argument("--margin-gb", type=float, default=20.0)
    cpu_offload_smoke.add_argument("--skip-inspect", action="store_true", help="skip GGUF tensor parsing and validate only stat/plan/space workflow")
    cpu_offload_smoke.add_argument("--require-inspect", action="store_true", help="exit non-zero if GGUF tensor inspection fails")
    cpu_offload_smoke.add_argument("--json", action="store_true")

    cpu_offload_build = sub.add_parser("cpu-offload-build-plan", help="write a no-execute huge-model CPU-offload streaming build manifest")
    cpu_offload_build.add_argument("--source-gguf", required=True)
    cpu_offload_build.add_argument("--output-dir", required=True)
    cpu_offload_build.add_argument("--model-name", default="GLM-5.1")
    cpu_offload_build.add_argument("--scratch-root", default=None)
    cpu_offload_build.add_argument("--benchmark-port", type=int, default=8084)
    cpu_offload_build.add_argument("--margin-gb", type=float, default=20.0)
    cpu_offload_build.add_argument("--skip-inspect", action="store_true", help="skip GGUF tensor parsing and use stat/plan/space workflow only")
    cpu_offload_build.add_argument("--require-inspect", action="store_true", help="exit non-zero if GGUF tensor inspection fails")
    cpu_offload_build.add_argument("--manifest", default=None, help="pipeline manifest path; defaults under --output-dir")
    cpu_offload_build.add_argument("--write", default=None, help="write build plan JSON to this path")
    cpu_offload_build.add_argument("--json", action="store_true")

    task_profiles = sub.add_parser("task-profiles", help="list task-specific Cerebellum variant profiles")
    task_profiles.add_argument("--json", action="store_true")

    legacy_plan = sub.add_parser("legacy-plan", help="plan the old group-first benchmark-gated Cerebellum workflow")
    legacy_plan.add_argument("run_dir", nargs="?", help="optional run directory or model/run name to audit")
    legacy_plan.add_argument("--source-gguf", default=None, help="source GGUF for a new protected workflow plan")
    legacy_plan.add_argument("--output-dir", default=None, help="output directory for a new protected workflow plan")
    legacy_plan.add_argument("--write", default=None, help="write plan JSON to this path")
    legacy_plan.add_argument("--json", action="store_true")

    legacy_flow = sub.add_parser("legacy-flow", help="write the classic group-first Cerebellum flow")
    legacy_flow.add_argument("--source-gguf", required=True)
    legacy_flow.add_argument("--output-dir", required=True)
    legacy_flow.add_argument("--run-dir", default=None)
    legacy_flow.add_argument("--imatrix", default=None)
    legacy_flow.add_argument("--corpus", default=None)
    legacy_flow.add_argument("--model-name", default=None)
    legacy_flow.add_argument("--family", default=None)
    legacy_flow.add_argument("--source-name", default=None)
    legacy_flow.add_argument("--profile", choices=["wiki", "agentic", "code", "math", "dialogue", "all-around", "custom"], default="custom")
    legacy_flow.add_argument("--metric", choices=["ppl", "humaneval", "arc", "mmlu", "tool-call", "dialogue"], default="ppl")
    legacy_flow.add_argument("--base-type", default="Q4_K_M")
    legacy_flow.add_argument("--start-type", default="q4_K")
    legacy_flow.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    legacy_flow.add_argument("--survivability-levels", default="q2_K")
    legacy_flow.add_argument("--survivability-target-type", default="q2_K", help="override type written for selected group-survivor tensors")
    legacy_flow.add_argument("--with-targeted-hillstep", action="store_true", help="after group survivability, run optional per-tensor hillstep on selected targets")
    legacy_flow.add_argument("--max-regression-pct", type=float, default=2.0)
    legacy_flow.add_argument("--baseline-ppl", type=float, default=None, help="baseline PPL used to classify survivability logs")
    legacy_flow.add_argument("--quantize-bin", default=DEFAULT_QUANTIZE)
    legacy_flow.add_argument("--perplexity-bin", default=DEFAULT_PERPLEXITY)
    legacy_flow.add_argument("--pure-quant", action="store_true", help="Pass --pure through generated quantization commands")
    legacy_flow.add_argument("--gpu-layers", type=int, default=99)
    legacy_flow.add_argument("--ctx-size", type=int, default=2048)
    legacy_flow.add_argument("--chunks", type=int, default=None)
    legacy_flow.add_argument("--scratch-root", default=None)
    legacy_flow.add_argument("--distrobox", default=None)
    legacy_flow.add_argument("--quant-timeout", type=int, default=1800)
    legacy_flow.add_argument("--ppl-timeout", type=int, default=900)
    legacy_flow.add_argument("--min-free-gb", type=float, default=15.0)
    legacy_flow.add_argument("--hard-free-floor-gb", type=float, default=15.0)
    legacy_flow.add_argument("--token-embedding-type", default="f16")
    legacy_flow.add_argument("--low-space", action="store_true")
    legacy_flow.add_argument("--serial-candidates", action="store_true")
    legacy_flow.add_argument("--execute-forward", action="store_true", help="run the classic forward group ablation queue, skipping completed groups")
    legacy_flow.add_argument("--keep-candidates", action="store_true", help="keep measured forward candidate GGUFs during --execute-forward")
    legacy_flow.add_argument("--benchmark-suite", choices=sorted(BENCHMARK_SUITES), default="release-local")
    legacy_flow.add_argument("--write", default=None, help="write flow JSON to this path")
    legacy_flow.add_argument("--json", action="store_true")

    group_scan = sub.add_parser("group-scan", help="run one classic Cerebellum group ablation candidate")
    group_scan.add_argument("--source-gguf", required=True)
    group_scan.add_argument("--corpus", required=True)
    group_scan.add_argument("--run-dir", required=True)
    group_scan.add_argument("--tensor-regex", required=True)
    group_scan.add_argument("--group-name", default=None)
    group_scan.add_argument("--run-name", default=None)
    group_scan.add_argument("--profile", default="wiki")
    group_scan.add_argument("--family", default=None)
    group_scan.add_argument("--model-name", default=None)
    group_scan.add_argument("--source-name", default=None)
    group_scan.add_argument("--base-type", default="Q4_K_M")
    group_scan.add_argument("--start-type", default="q4_K")
    group_scan.add_argument("--target-type", default="q2_K")
    group_scan.add_argument("--base-map", default=None, help="existing tensor-type map to apply before this group override")
    group_scan.add_argument("--baseline-ppl", type=float, default=None)
    group_scan.add_argument("--imatrix", default=None)
    group_scan.add_argument("--quantize-bin", default=DEFAULT_QUANTIZE)
    group_scan.add_argument("--perplexity-bin", default=DEFAULT_PERPLEXITY)
    group_scan.add_argument("--gpu-layers", type=int, default=99)
    group_scan.add_argument("--ctx-size", type=int, default=2048)
    group_scan.add_argument("--chunks", type=int, default=None)
    group_scan.add_argument("--distrobox", default=None)
    group_scan.add_argument("--quant-timeout", type=int, default=1800)
    group_scan.add_argument("--ppl-timeout", type=int, default=900)
    group_scan.add_argument("--min-free-gb", type=float, default=15.0)
    group_scan.add_argument("--hard-free-floor-gb", type=float, default=15.0)
    group_scan.add_argument("--token-embedding-type", default="f16")
    group_scan.add_argument("--preview-limit", type=int, default=20)
    group_scan.add_argument("--prune-candidate", action="store_true", default=True)
    group_scan.add_argument("--keep-candidate", dest="prune_candidate", action="store_false")
    group_scan.add_argument("--dry-run", action="store_true")

    sparse_replay = sub.add_parser("sparse-replay", help="run the OG Qwen3.6-27B-v4 style sparse ablation replay pipeline")
    sparse_replay.add_argument("--source-gguf", required=True)
    sparse_replay.add_argument("--corpus", required=True)
    sparse_replay.add_argument("--run-dir", required=True)
    sparse_replay.add_argument("--baseline-ppl", type=float, required=True)
    sparse_replay.add_argument("--baseline-gguf", default=None)
    sparse_replay.add_argument("--probe-plan", default=None, help="JSON or newline tensor list; defaults to sparse layer/component anchors")
    sparse_replay.add_argument("--max-probes", type=int, default=None, help="limit default/generated probe plan")
    sparse_replay.add_argument("--tensor-regex", default=None)
    sparse_replay.add_argument("--run-name", default=None)
    sparse_replay.add_argument("--profile", default="wiki")
    sparse_replay.add_argument("--family", default=None)
    sparse_replay.add_argument("--model-name", default=None)
    sparse_replay.add_argument("--source-name", default=None)
    sparse_replay.add_argument("--probe-base-type", default="Q4_K_M", help="baseline quant used while crushing one tensor")
    sparse_replay.add_argument("--start-type", default="q4_K")
    sparse_replay.add_argument("--target-type", default="q2_K")
    sparse_replay.add_argument("--final-base-type", default="Q2_K", help="base type used for the final allocated GGUF")
    sparse_replay.add_argument("--budget-gb", type=float, required=True)
    sparse_replay.add_argument("--final-gguf", default=None)
    sparse_replay.add_argument("--imatrix", default=None)
    sparse_replay.add_argument("--quantize-bin", default=DEFAULT_QUANTIZE)
    sparse_replay.add_argument("--perplexity-bin", default=DEFAULT_PERPLEXITY)
    sparse_replay.add_argument("--gpu-layers", type=int, default=99)
    sparse_replay.add_argument("--ctx-size", type=int, default=2048)
    sparse_replay.add_argument("--chunks", type=int, default=None)
    sparse_replay.add_argument("--distrobox", default=None)
    sparse_replay.add_argument("--quant-timeout", type=int, default=1800)
    sparse_replay.add_argument("--ppl-timeout", type=int, default=900)
    sparse_replay.add_argument("--allocate-timeout", type=int, default=1800)
    sparse_replay.add_argument("--token-embedding-type", default="f16")
    sparse_replay.add_argument("--prune-candidates", action="store_true", default=True)
    sparse_replay.add_argument("--keep-candidates", dest="prune_candidates", action="store_false")
    sparse_replay.add_argument("--continue-on-failure", action="store_true")
    sparse_replay.add_argument("--dry-run", action="store_true")

    public_card_policy = sub.add_parser("public-card-policy", help="print sanitized public model-card policy")
    public_card_policy.add_argument("--write", default=None, help="write policy JSON to this path")
    public_card_policy.add_argument("--json", action="store_true")

    system = sub.add_parser("system", help="inspect local resources and tool availability")
    system.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor", help="check portable Cerebellum setup and explain fixes")
    doctor.add_argument("--source-gguf", default=None, help="optional source GGUF to inspect for model-specific conversion gotchas")
    doctor.add_argument("--json", action="store_true")

    self_test = sub.add_parser("self-test", help="run read-only Cerebellum CLI/API smoke checks")
    self_test.add_argument("--run-dir", default=None, help="Optional run directory for run-aware checks")
    self_test.add_argument("--json", action="store_true")

    plan_space = sub.add_parser("plan-space", help="DEPRECATED — see `cerebellum method`; recommend low-space scratch strategy for hillstep runs")
    plan_space.add_argument("--source-gguf", required=True)
    plan_space.add_argument("--data-root")
    plan_space.add_argument("--scratch-candidates", default="")
    plan_space.add_argument("--margin-gb", type=float, default=20.0)
    plan_space.add_argument("--json", action="store_true")

    tutorial = sub.add_parser("tutorial", help="explain Cerebellum tools and flows")
    tutorial.add_argument("topic", nargs="?", default="overview")

    tips = sub.add_parser("tips", help="turn contextual tips on/off")
    tips.add_argument("value", choices=["on", "off", "status"])

    db = sub.add_parser("db", help="browse/query Cerebellum SQLite database")
    db.add_argument("--db", default=DEFAULT_DB)
    db_sub = db.add_subparsers(dest="db_cmd", required=True)
    db_sub.add_parser("families")
    models = db_sub.add_parser("models")
    models.add_argument("--family")
    db_sub.add_parser("builds")
    db_sub.add_parser("benchmarks")
    leaderboard_db = db_sub.add_parser("leaderboard")
    leaderboard_db.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="release")
    leaderboard_db.add_argument("--weight", action="append", default=[], help="leaderboard benchmark weight as BENCHMARK=WEIGHT")
    leaderboard_db.add_argument("--limit", type=int, default=20)
    db_sub.add_parser("runs")
    import_run = db_sub.add_parser("import-run")
    import_run.add_argument("run_dir")
    query = db_sub.add_parser("query")
    query.add_argument("--sql")
    db.add_argument("--json", action="store_true")

    report = sub.add_parser("report", help="write clean Cerebellum reports")
    report.add_argument("run_dir")
    report.add_argument("--format", default="json,md,csv,infographic")
    report.add_argument("--json", action="store_true")

    benchmark_report_parser = sub.add_parser("benchmark-report", help="compare benchmark result JSON files")
    benchmark_report_parser.add_argument("paths", nargs="*", help="benchmark result JSON files or directories; use label=PATH to force a model column label")
    benchmark_report_parser.add_argument("--baseline", help="model name to use for delta calculations")
    benchmark_report_parser.add_argument("--output", help="write report to this path instead of stdout")
    benchmark_report_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of Markdown")
    benchmark_report_parser.add_argument("--no-bars", action="store_true", help="omit ASCII bar chart section")
    benchmark_report_parser.add_argument("--leaderboard", action="store_true", help="include average quality and score/GiB leaderboard")
    benchmark_report_parser.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="full", help="benchmark suite to use for leaderboard averaging")
    benchmark_report_parser.add_argument("--size", action="append", default=[], help="model size as MODEL=GiB for score/GiB leaderboard")
    benchmark_report_parser.add_argument("--size-json", help="JSON file with per-model size metadata")
    benchmark_report_parser.add_argument("--weight", action="append", default=[], help="leaderboard benchmark weight as BENCHMARK=WEIGHT; defaults to 1.0")
    benchmark_report_parser.add_argument("--list-suites", action="store_true", help="print built-in benchmark suite names and tasks")

    benchmark_plan_parser = sub.add_parser("benchmark-plan", help="print benchmark suite run plan and artifact checklist")
    benchmark_plan_parser.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="full")
    benchmark_plan_parser.add_argument("--model", default="cerebellum", help="BENCH_MODEL label for generated commands")
    benchmark_plan_parser.add_argument("--port", type=int, default=8084, help="llama-server port used by generated commands")
    benchmark_plan_parser.add_argument("--results-dir", default="benchmark_results", help="directory for benchmark artifacts")
    benchmark_plan_parser.add_argument("--require-ready", action="store_true", help="exit non-zero if any suite benchmark lacks an implemented runner")
    benchmark_plan_parser.add_argument("--json", action="store_true")

    benchmark_run_parser = sub.add_parser("benchmark-run", help="validate or execute a Cerebellum benchmark suite plan")
    benchmark_run_parser.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="release")
    benchmark_run_parser.add_argument("--model", default="cerebellum", help="BENCH_MODEL label for generated commands")
    benchmark_run_parser.add_argument("--port", type=int, default=8084, help="llama-server port used by generated commands")
    benchmark_run_parser.add_argument("--results-dir", default="benchmark_results", help="directory for benchmark artifacts and logs")
    benchmark_run_parser.add_argument("--benchmark", action="append", help="run only this benchmark key; may be repeated")
    benchmark_run_parser.add_argument("--execute", action="store_true", help="actually run benchmark commands; dry-run is default")
    benchmark_run_parser.add_argument("--postprocess", action="store_true", help="after a successful execute, write manifest/audit/report sidecars under RESULTS_DIR/postprocess")
    benchmark_run_parser.add_argument("--require-complete", action="store_true", help="with --postprocess, fail if suite result JSONs are missing")
    benchmark_run_parser.add_argument("--leaderboard", action="store_true", help="with --postprocess, include leaderboard rows in benchmark_report.json")
    benchmark_run_parser.add_argument("--size", action="append", default=[], help="with --postprocess --leaderboard, model size as MODEL=GiB")
    benchmark_run_parser.add_argument("--size-json", help="with --postprocess --leaderboard, JSON file with per-model size metadata")
    benchmark_run_parser.add_argument("--weight", action="append", default=[], help="with --postprocess --leaderboard, benchmark weight as BENCHMARK=WEIGHT")
    benchmark_run_parser.add_argument("--json", action="store_true")

    benchmark_postprocess_parser = sub.add_parser("benchmark-postprocess", help="write manifest/audit/report sidecars for existing benchmark artifacts")
    benchmark_postprocess_parser.add_argument("results_dir", help="directory containing benchmark summary/detail artifacts")
    benchmark_postprocess_parser.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="release")
    benchmark_postprocess_parser.add_argument("--model", default="cerebellum")
    benchmark_postprocess_parser.add_argument("--require-complete", action="store_true", help="fail if suite result JSONs are missing")
    benchmark_postprocess_parser.add_argument("--leaderboard", action="store_true", help="include leaderboard rows in benchmark_report.json")
    benchmark_postprocess_parser.add_argument("--size", action="append", default=[], help="model size as MODEL=GiB")
    benchmark_postprocess_parser.add_argument("--size-json", help="JSON file with per-model size metadata")
    benchmark_postprocess_parser.add_argument("--weight", action="append", default=[], help="benchmark weight as BENCHMARK=WEIGHT")
    benchmark_postprocess_parser.add_argument("--json", action="store_true")

    benchmark_ingest_parser = sub.add_parser("benchmark-ingest", help="ingest benchmark artifacts into the Cerebellum SQLite DB and report publishability")
    benchmark_ingest_parser.add_argument("results_dir", help="directory containing benchmark summary/detail artifacts")
    benchmark_ingest_parser.add_argument("--db", default=DEFAULT_DB)
    benchmark_ingest_parser.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="release")
    benchmark_ingest_parser.add_argument("--model", default="cerebellum")
    benchmark_ingest_parser.add_argument("--require-complete", action="store_true", help="exit non-zero when selected suite summary JSONs are missing")
    benchmark_ingest_parser.add_argument("--leaderboard", action="store_true", help="store leaderboard rows in the persisted report JSON")
    benchmark_ingest_parser.add_argument("--size", action="append", default=[], help="model size as MODEL=GiB for score/GiB leaderboard")
    benchmark_ingest_parser.add_argument("--size-json", help="JSON file with per-model size metadata")
    benchmark_ingest_parser.add_argument("--weight", action="append", default=[], help="leaderboard benchmark weight as BENCHMARK=WEIGHT")
    benchmark_ingest_parser.add_argument("--json", action="store_true")

    benchmark_status_parser = sub.add_parser("benchmark-status", help="summarize benchmark-run event logs and resume point")
    benchmark_status_parser.add_argument("--results-dir", default="benchmark_results", help="directory containing benchmark_run_events.jsonl")
    benchmark_status_parser.add_argument("--events", default=None, help="benchmark_run_events.jsonl path; defaults under --results-dir")
    benchmark_status_parser.add_argument("--json", action="store_true")

    rebench_plan_parser = sub.add_parser("benchmark-rebench-plan", help="plan corrected HumanEval+/release reruns for published models")
    rebench_plan_parser.add_argument("--suite", choices=["humaneval", "release"], default="humaneval")
    rebench_plan_parser.add_argument("--results-root", default="benchmark_results/rebench_20260605")
    rebench_plan_parser.add_argument("--port", type=int, default=8084)
    rebench_plan_parser.add_argument("--model", action="append", help="override affected HF repo list; may be repeated")
    rebench_plan_parser.add_argument("--correction-issue", default="#35")
    rebench_plan_parser.add_argument("--json", action="store_true")

    benchmark_manifest_parser = sub.add_parser("benchmark-manifest", help="write a hashed manifest for benchmark artifacts")
    benchmark_manifest_parser.add_argument("paths", nargs="+", help="benchmark result files/directories")
    benchmark_manifest_parser.add_argument("--suite", choices=sorted(BENCHMARK_SUITES), default="release")
    benchmark_manifest_parser.add_argument("--model", default=None, help="optional model label for the manifest")
    benchmark_manifest_parser.add_argument("--output", default=None, help="write manifest JSON/Markdown to this path")
    benchmark_manifest_parser.add_argument("--require-complete", action="store_true", help="exit non-zero if any suite benchmark lacks measured summary JSON")
    benchmark_manifest_parser.add_argument("--json", action="store_true")

    benchmark_audit_parser = sub.add_parser("benchmark-audit", help="audit benchmark detailed artifacts before publishing")
    benchmark_audit_parser.add_argument("paths", nargs="+", help="detailed JSONL/sample files or directories")
    benchmark_audit_parser.add_argument("--json", action="store_true")
    benchmark_audit_parser.add_argument("--fail-empty-pct", type=float, default=2.0)
    benchmark_audit_parser.add_argument("--fail-unknown-pct", type=float, default=5.0)
    benchmark_audit_parser.add_argument("--fail-pass-only-pct", type=float, default=5.0)

    ablation_analyze = sub.add_parser("ablation-analyze", help="analyze ablation PPL JSON/logs and write tensor overrides")
    ablation_analyze.add_argument("input", help="ablation_results.json, log file, or directory of PPL logs")
    ablation_analyze.add_argument("--baseline-ppl", type=float, help="baseline PPL; required for raw log input")
    ablation_analyze.add_argument("--tensor-group", help="tensor group for ppl_layer_N.log input, e.g. attn_q or ffn_up")
    ablation_analyze.add_argument("--target-type", default="q2_K", help="override quant type for selected classes")
    ablation_analyze.add_argument("--override-classes", default="demotable,beneficial,tolerant", help="comma-separated classes to write to --output")
    ablation_analyze.add_argument("--output", help="write llama-quantize tensor-type override file")
    ablation_analyze.add_argument("--tensor-output", help="write newline tensor names for a follow-up --tensor-file run")
    ablation_analyze.add_argument("--json-output", help="write JSON analysis sidecar")
    ablation_analyze.add_argument("--json", action="store_true", help="print JSON instead of a table")

    export = sub.add_parser("export", help="export run data for AI, infographic, or automation")
    export.add_argument("run_dir")
    export.add_argument("--kind", choices=["raw", "ai", "infographic"], default="ai")
    export.add_argument("--output")

    auth = sub.add_parser("auth", help="check HF/GitHub auth status")
    auth.add_argument("service", choices=["hf", "huggingface", "github"])

    hf_stats = sub.add_parser("hf-stats", help="summarize Hugging Face model download/like stats")
    hf_stats.add_argument("--author", default="deucebucket", help="HF user/org namespace for public rolling stats")
    hf_stats.add_argument("--period", choices=["recent", "all-time"], default="recent", help="recent uses public rolling downloads; all-time requires --publisher-org")
    hf_stats.add_argument("--publisher-org", default=None, help="HF Publisher Analytics org for all-time stats")
    hf_stats.add_argument("--limit", type=int, default=1000)
    hf_stats.add_argument("--snapshot", default=None, help="append the stats payload to a local JSONL ledger")
    hf_stats.add_argument("--json", action="store_true")

    upload = sub.add_parser("upload", help="upload Cerebellum artifacts to HF/GitHub")
    upload.add_argument("target", choices=["hf", "huggingface", "github"])
    upload.add_argument("run_dir")
    upload.add_argument("--repo")
    upload.add_argument("--repo-type", default="model")
    upload.add_argument("--branch")
    upload.add_argument("--private", action="store_true", help="upload raw factory sidecars; default uploads only public-safe sidecars")
    upload.add_argument("--dry-run", action="store_true")

    api = sub.add_parser("api", help="serve Cerebellum JSON API for automation/web UI")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8931)
    api.add_argument("--data-root", default=None)
    api.add_argument("--db", default=DEFAULT_DB)

    if (
        argv is None
        and len(sys.argv) > 1
        and sys.argv[1] not in {
            "run", "imatrix", "status", "events", "runs", "schedule", "db", "report",
            "export", "auth", "upload", "api", "system", "doctor", "self-test", "provenance", "inspect-gguf-types", "finalize", "package", "plan-space",
            "public-audit", "public-history-audit", "public-export", "release-gate", "artifact-inventory", "history",
            "benchmark-plan", "benchmark-run", "benchmark-postprocess", "benchmark-ingest", "benchmark-status", "benchmark-rebench-plan", "benchmark-manifest", "benchmark-audit",
            "benchmark-report", "cpu-offload-smoke", "cpu-offload-build-plan", "compare-gguf-types", "compare-locks",
            "tutorial", "tips", "watch", "stop", "--help", "-h",
            "cleanup", "rollback",
            "backup",
            "resume",
            "recover",
            "group-scan",
            "sparse-replay",
            "project",
        }
    ):
        argv = ["run", *sys.argv[1:]]
    return parser.parse_args(argv)


def status_cmd(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise SystemExit(f"no state.json found in {run_dir}")
    state = json.loads(state_path.read_text())
    recovery = build_recovery_plan(run_dir)
    enabled = not args.no_color and not args.plain
    locked = len(state.get("locked", {}))
    tested = state.get("tested", [])
    print(color("Cerebellum status", "36;1", enabled))
    print(f"run_dir     : {run_dir}")
    print(f"status      : {state.get('run_status')}")
    print(f"health      : {recovery.get('active_health')} - {recovery.get('active_reason')}")
    print(f"model       : {state.get('model_family')}/{state.get('model_name')}")
    print(f"locked      : {locked}")
    print(f"current_ppl : {state.get('current_ppl')}")
    print(f"last_tensor : {state.get('last_tensor')}")
    if recovery.get("interrupted"):
        print(f"resume      : {recovery.get('resume_command')}")
    if tested:
        print("\nRecent locks:")
        for row in tested[-10:]:
            print(f"  {row.get('winner'):<6} {row.get('ppl')}  {row.get('tensor')}")


def events_cmd(args: argparse.Namespace) -> None:
    path = first_existing(Path(args.run_dir), EVENT_FILES)
    if not path.exists():
        raise SystemExit(f"no event log found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.type and row.get("event") != args.type:
                continue
            rows.append(row)
    rows = rows[-args.limit :]
    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return
    for row in rows:
        print(f"{row.get('timestamp_utc')}  {row.get('event'):<20}  {row.get('tensor', '')} {row.get('level', '')}")


def watch_cmd(args: argparse.Namespace) -> None:
    if args.tui:
        if args.public:
            raise SystemExit("watch --public --tui is not supported; use `cerebellum watch --public --plain` or omit --tui")
        tui_watch_cmd(args)
        return
    grid_watch_cmd(args)
    return
    run_dir = Path(args.run_dir)
    enabled = not args.no_color and not args.plain
    try:
        while True:
            state = read_json(run_dir / "state.json", {})
            manifest = read_json(run_dir / "manifest.json", {})
            events = read_jsonl(first_existing(run_dir, EVENT_FILES))
            candidates = read_jsonl(first_existing(run_dir, CANDIDATE_FILES))
            last_events = events[-max(1, args.events_limit) :]
            last_tensor = state.get("last_tensor")
            status = state.get("run_status")
            terminal_events = {"run_stopped", "run_finish", "tensor_interrupted", "signal_received"}
            if status in {"stopped", "complete", "failed"}:
                active = next((row for row in reversed(events) if row.get("event") in terminal_events), {})
            else:
                active = next((row for row in reversed(events) if row.get("event") in {"tensor_start", "quant_start", "ppl_start"}), {})
            last_event = events[-1] if events else {}
            last_event_age = event_age_seconds(last_event)
            active_age = event_age_seconds(active)
            processes = process_rows_for_run(run_dir)
            active_processes = [row for row in processes if row["kind"] in {"quantize", "ppl"}]
            runner_processes = [row for row in processes if row["kind"] == "runner"]
            gpu_info = gpu_rows()
            health = "idle"
            health_reason = "not running"
            health_code = "90"
            if status == "running":
                if active_processes:
                    health = "active"
                    health_reason = ", ".join(f"{row['kind']} pid {row['pid']} {row['etime']}" for row in active_processes[:2])
                    health_code = "32;1"
                elif runner_processes and last_event_age is not None and last_event_age < args.stall_warn_seconds:
                    health = "waiting"
                    health_reason = f"runner alive; last event {fmt_seconds(last_event_age)} ago"
                    health_code = "33;1"
                elif runner_processes and last_event_age is not None and last_event_age < args.stall_fail_seconds:
                    health = "stalled?"
                    health_reason = f"runner alive but no event for {fmt_seconds(last_event_age)}"
                    health_code = "33;1"
                elif runner_processes:
                    health = "failure suspected"
                    health_reason = f"runner alive, no event for {fmt_seconds(last_event_age)}"
                    health_code = "31;1"
                else:
                    health = "failure suspected"
                    health_reason = "state says running but no runner process found"
                    health_code = "31;1"
            os.system("clear" if os.name != "nt" else "cls")
            width = 96
            run_id = manifest.get("run_id") or state.get("run_id") or run_dir.name
            model = f"{state.get('model_family')}/{state.get('model_name')}"
            ppl = state.get("current_ppl")
            profile = manifest.get("ppl_profile") or state.get("ppl_profile") or "custom"
            corpus = manifest.get("corpus") or state.get("corpus") or "-"
            locked = len(state.get("locked", {}))
            total_hint = next((row.get("total") for row in reversed(events) if row.get("total")), None)
            progress_visual, progress_text = progress_bar(locked, total_hint)
            baseline_path = Path(state.get("baseline_path") or run_dir / "artifacts" / "current_baseline.gguf")
            baseline_size = path_size(baseline_path) if baseline_path.exists() else None
            active_path = active.get("tmp_output") or active.get("output") or active.get("model")
            active_size = path_size(Path(active_path)) if active_path else None
            newest_candidate_size = next((row.get("size_bytes") for row in reversed(candidates) if row.get("size_bytes")), None)
            print(color("╭" + "─" * (width - 2) + "╮", "36;1", enabled))
            print(color("│" + " CEREBELLUM ".center(width - 2) + "│", "36;1", enabled))
            print(color("│" + " resource-aware mixed-precision GGUF quantization ".center(width - 2) + "│", "36", enabled))
            print(color("╰" + "─" * (width - 2) + "╯", "36;1", enabled))
            print()
            print(color("╭─ Run ─" + "─" * (width - 9) + "╮", "34;1", enabled))
            status_code = "32;1" if status == "running" else "33;1" if status == "stopped" else "36;1"
            print(kv_line("id", run_id, width, enabled, "37;1"))
            print(kv_line("model", model, width, enabled, "36;1"))
            print(kv_line("status", status, width, enabled, status_code))
            print(kv_line("profile", profile, width, enabled, "35;1"))
            print(kv_line("corpus", str(corpus)[-(width - 13):], width, enabled, "90"))
            print(kv_line("ppl", ppl, width, enabled, "33;1"))
            print(kv_line("progress", f"{progress_visual} {progress_text}", width, enabled, "32;1"))
            print(color("╰" + "─" * (width - 2) + "╯", "34;1", enabled))
            print()
            print(color("╭─ Active work ─" + "─" * (width - 16) + "╮", "37;1", enabled))
            event_code = "32;1" if str(active.get("event", "")).endswith("_start") else "36;1"
            print(kv_line("event", active.get("event"), width, enabled, event_code))
            print(kv_line("tensor", active.get("tensor"), width, enabled, "33;1"))
            print(kv_line("level", active.get("level"), width, enabled, "35;1"))
            if last_tensor:
                print(kv_line("last", last_tensor, width, enabled, "33"))
            print(color("╰" + "─" * (width - 2) + "╯", "37;1", enabled))
            print()
            print(color("╭─ GGUF sizes ─" + "─" * (width - 15) + "╮", "36;1", enabled))
            print(kv_line("current", fmt_bytes(baseline_size), width, enabled, "36;1"))
            print(kv_line("active", fmt_bytes(active_size), width, enabled, "32;1" if active_size else "90"))
            print(kv_line("recent", fmt_bytes(newest_candidate_size), width, enabled, "36"))
            if active_path:
                print(kv_line("file", str(active_path)[-(width - 13):], width, enabled, "90"))
            print(color("╰" + "─" * (width - 2) + "╯", "36;1", enabled))
            print()
            print(color("╭─ Activity / health ─" + "─" * (width - 22) + "╮", health_code, enabled))
            print(kv_line("health", health, width, enabled, health_code))
            print(kv_line("reason", health_reason[:width - 13], width, enabled, "37"))
            print(kv_line("active", fmt_seconds(active_age), width, enabled, "36;1"))
            print(kv_line("last_evt", fmt_seconds(last_event_age), width, enabled, "36;1"))
            for row in active_processes[:3]:
                line = f"{row['kind']} pid={row['pid']} etime={row['etime']} cpu={row['pcpu']}% mem={row['pmem']}%"
                proc_code = "32;1" if row["kind"] in {"quantize", "ppl"} else "37"
                print(kv_line("proc", line[:width - 13], width, enabled, proc_code))
            if status == "running" and not active_processes:
                print(kv_line("warning", "no active llama child process detected", width, enabled, "31;1"))
            print(color("╰" + "─" * (width - 2) + "╯", health_code, enabled))
            eta, eta_basis = estimate_eta(state, active_age, total_hint)
            cpu_job = next((row for row in active_processes if row["kind"] == "quantize"), None)
            gpu_job = next((row for row in active_processes if row["kind"] == "ppl"), None)
            print()
            print(color("╭─ Resources / ETA ─" + "─" * (width - 19) + "╮", "34;1", enabled))
            print(kv_line("eta", eta, width, enabled, "36;1"))
            print(kv_line("basis", eta_basis[: width - 14], width, enabled, "90"))
            if cpu_job:
                cpu_line = f"quantize pid={cpu_job['pid']} cpu={cpu_job['pcpu']}% mem={cpu_job['pmem']}% {cpu_job['etime']}"
            else:
                cpu_line = "idle or waiting"
            if gpu_job:
                gpu_line = f"ppl pid={gpu_job['pid']} cpu={gpu_job['pcpu']}% mem={gpu_job['pmem']}% {gpu_job['etime']}"
            else:
                gpu_line = "idle or waiting"
            print(kv_line("cpu_job", cpu_line[: width - 14], width, enabled, "32;1" if cpu_job else "90"))
            print(kv_line("gpu_job", gpu_line[: width - 14], width, enabled, "32;1" if gpu_job else "90"))
            for gpu in gpu_info[:2]:
                gpu_line = (
                    f"cuda:{gpu['index']} {gpu['util']}% "
                    f"vram {gpu['mem_used']}/{gpu['mem_total']} MiB power {gpu['power']} W"
                )
                print(kv_line("gpu", gpu_line[: width - 14], width, enabled, "36;1"))
            try:
                free_gb = disk_free_gb(run_dir)
                print(kv_line("disk", f"{free_gb:.1f} GiB free at run dir", width, enabled, "36"))
            except OSError:
                pass
            print(color("╰" + "─" * (width - 2) + "╯", "34;1", enabled))
            totals = state.get("totals", {})
            print()
            print(color("╭─ Timing ─" + "─" * (width - 11) + "╮", "35;1", enabled))
            timing_line = (
                f"quant {fmt_seconds(totals.get('quant_seconds'))}   "
                f"ppl {fmt_seconds(totals.get('ppl_seconds'))}   "
                f"tests {totals.get('candidates', 0)}   failures {totals.get('failures', 0)}"
            )
            print(f"│ {color(f'{timing_line:<{width - 4}}', '36;1', enabled)} │")
            print(color("╰" + "─" * (width - 2) + "╯", "35;1", enabled))
            print()
            print(color("╭─ Recent measurements ─" + "─" * (width - 24) + "╮", "32;1", enabled))
            print(f"│ {'quant':<8}{'ppl':<14}{'delta':<14}{'tensor':<{width - 42}}│")
            print(color("├" + "─" * (width - 2) + "┤", "32", enabled))
            for row in candidates[-max(1, args.measurements_limit) :]:
                delta = row.get("delta")
                delta_s = "-" if delta is None else f"{delta:+.4f}"
                level_s = color(f"{row.get('level', '-'):<8}", "35;1", enabled)
                ppl_s = color(f"{str(row.get('ppl', '-')):<14}", "33;1", enabled)
                delta_s_colored = color(f"{delta_s:<14}", delta_code(delta), enabled)
                tensor_s = color(f"{row.get('tensor', ''):<{width - 42}}", "33", enabled)
                print(f"│ {level_s}{ppl_s}{delta_s_colored}{tensor_s}│")
            print(color("╰" + "─" * (width - 2) + "╯", "32;1", enabled))
            print()
            print(color("╭─ Event stream ─" + "─" * (width - 17) + "╮", "33;1", enabled))
            for row in last_events:
                line = f"{row.get('timestamp_utc', '')[-13:]}  {row.get('event', ''):<22} {row.get('level', ''):<6} {row.get('tensor', '')}"
                print(f"│ {line:<{width - 4}} │")
            print(color("╰" + "─" * (width - 2) + "╯", "33;1", enabled))
            print()
            print(color("Ctrl+C exits the interface only. Use `cerebellum stop RUN_DIR` to stop a run.", "90", enabled))
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def child_pids(root_pid: int) -> list[int]:
    proc = subprocess.run(["ps", "-eo", "pid=,ppid=,cmd="], capture_output=True, text=True)
    children: dict[int, list[int]] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    found: list[int] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            found.append(child)
            stack.append(child)
    return found


TERMINAL_MARKERS = ("STOPPED", "ABORTED", "COMPLETE")


def clear_terminal_markers(run_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in TERMINAL_MARKERS:
        marker = run_dir / name
        if not marker.exists():
            continue
        try:
            marker.unlink()
            removed.append(name)
        except OSError:
            pass
    return removed


def stop_target_pids(run_dir: Path, events: list[dict[str, Any]]) -> list[int]:
    targets: list[int] = []

    def add(pid: Any) -> None:
        try:
            value = int(pid)
        except (TypeError, ValueError):
            return
        if value == os.getpid() or value in targets:
            return
        targets.append(value)

    recent: list[int] = []
    for row in reversed(events):
        pid = row.get("pid")
        if isinstance(pid, int) and pid not in recent:
            recent.append(pid)
        if len(recent) >= 3:
            break
    for pid in recent:
        for child in child_pids(pid):
            add(child)
        add(pid)

    for row in process_rows_for_run(run_dir):
        if row.get("kind") not in {"runner", "quantize", "ppl", "container"}:
            continue
        pid = row.get("pid")
        for child in child_pids(int(pid)):
            add(child)
        add(pid)

    return targets


def stop_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise SystemExit(f"no state.json found in {run_dir}")
    events_path = first_existing(run_dir, EVENT_FILES)
    events = read_jsonl(events_path)
    signaled: list[int] = []
    if not args.no_kill:
        for target in stop_target_pids(run_dir, events):
            if target in signaled or not process_exists(target):
                continue
            try:
                os.kill(target, signal.SIGTERM)
                signaled.append(target)
            except OSError:
                pass
    state = read_json(state_path, {})
    state["run_status"] = "stopped"
    state["stopped_at"] = utc_now()
    state["stop_reason"] = args.reason
    atomic_write_json(state_path, state)
    marker = run_dir / "STOPPED"
    marker.write_text(utc_now() + "\n", encoding="utf-8")
    append_event(events_path, "run_stopped", reason=args.reason, signaled_pids=signaled)
    print(json.dumps({"run_dir": str(run_dir), "status": "stopped", "signaled_pids": signaled}, indent=2, sort_keys=True))


def cleanup_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    state = read_json(run_dir / "state.json", {})
    manifest = read_json(run_dir / "manifest.json", {})
    locked = set(state.get("locked", {}))
    events_path = first_existing(run_dir, EVENT_FILES)
    tmp_root = run_tmp_root(run_dir, manifest, state)
    candidates: list[Path] = []
    runner_active = any(row["kind"] == "runner" for row in process_rows_for_run(run_dir))
    if tmp_root.exists():
        for child in tmp_root.iterdir():
            if not child.is_dir():
                continue
            tensor_slug = child.name.split("-", 1)[1] if "-" in child.name else child.name
            active = any(slug(tensor) == tensor_slug for tensor in locked)
            if active:
                candidates.append(child)
            elif args.partials:
                if runner_active and not args.force:
                    raise SystemExit("refusing to delete partial temp while runner is active; stop run first or pass --force")
                candidates.append(child)
    if args.old_artifacts:
        parent = run_dir.parent
        for sibling in parent.iterdir() if parent.exists() else []:
            if sibling == run_dir or not sibling.is_dir():
                continue
            sibling_state = read_json(sibling / "state.json", {})
            if sibling_state.get("run_status") in {"stopped", "aborted", "failed"}:
                artifacts = sibling / "artifacts"
                if artifacts.exists():
                    candidates.append(artifacts)
    total = sum(path_size(path) if path.is_file() else sum(path_size(file) for file in path.rglob("*") if file.is_file()) for path in candidates)
    if args.yes:
        deleted: list[str] = []
        for path in candidates:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
            deleted.append(str(path))
        append_event(events_path, "cleanup_finish", deleted=deleted, bytes_reclaimed_estimate=total)
        print(json.dumps({"mode": "deleted", "bytes_reclaimed_estimate": total, "paths": deleted}, indent=2, sort_keys=True))
        return
    print(json.dumps({"mode": "dry-run", "bytes_reclaimable_estimate": total, "paths": [str(path) for path in candidates]}, indent=2, sort_keys=True))


def rollback_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    state_path = run_dir / "state.json"
    state = read_json(state_path, {})
    if not state:
        raise SystemExit(f"no state.json found in {run_dir}")
    if run_is_live(run_dir) and not args.force:
        raise SystemExit("refusing to rollback while runner is active; stop run first or pass --force")
    tested = list(state.get("tested", []))
    keep_count = len(tested)
    if args.to_locked is not None:
        keep_count = max(0, min(len(tested), args.to_locked))
    elif args.before_layer is not None:
        keep_count = 0
        for i, row in enumerate(tested):
            layer = tensor_layer(row.get("tensor", ""))
            if layer is not None and layer >= args.before_layer:
                break
            keep_count = i + 1
    elif args.last_completed_layer:
        layers = [tensor_layer(row.get("tensor", "")) for row in tested]
        numeric = [layer for layer in layers if layer is not None]
        if numeric:
            newest = max(numeric)
            keep_count = 0
            for i, row in enumerate(tested):
                layer = tensor_layer(row.get("tensor", ""))
                if layer is not None and layer >= newest:
                    break
                keep_count = i + 1
    else:
        raise SystemExit("choose --to-locked, --before-layer, or --last-completed-layer")
    kept = tested[:keep_count]
    removed = tested[keep_count:]
    if not args.yes:
        print(json.dumps({"mode": "dry-run", "keep": keep_count, "remove": len(removed), "removed_tensors": [row.get("tensor") for row in removed]}, indent=2, sort_keys=True))
        return
    checkpoint = run_dir / "checkpoints" / f"rollback-before-{local_stamp()}.json"
    atomic_write_json(checkpoint, state)
    state["tested"] = kept
    state["locked"] = {row["tensor"]: row["winner"] for row in kept if row.get("tensor") and row.get("winner")}
    state["totals"] = totals_for_kept_candidates(read_jsonl(first_existing(run_dir, CANDIDATE_FILES)), kept)
    state["last_tensor"] = kept[-1].get("tensor") if kept else None
    state["current_ppl"] = kept[-1].get("ppl") if kept else (tested[0].get("baseline_ppl") if tested else state.get("current_ppl"))
    state["run_status"] = "stopped"
    state["rollback_at"] = utc_now()
    state["rollback_removed"] = [row.get("tensor") for row in removed]
    state["baseline_invalid_after_rollback"] = True
    state["rollback_note"] = "state rolled back; next resume will rebuild current_baseline.gguf from the rolled-back tensor type map"
    state["updated_at"] = utc_now()
    manifest = read_json(run_dir / "manifest.json", {})
    source = manifest.get("source_gguf") or state.get("source_gguf")
    start_type = manifest.get("start_type") or state.get("start_type") or "q4_K"
    write_tensor_types_map(Path(source) if source else None, state["locked"], start_type, run_dir / CURRENT_TYPES_FILE)
    atomic_write_json(state_path, state)
    atomic_write_json(run_dir / "timing.json", state["totals"])
    append_event(first_existing(run_dir, EVENT_FILES), "rollback_finish", keep=keep_count, removed=len(removed), checkpoint=str(checkpoint))
    print(json.dumps({"mode": "written", "checkpoint": str(checkpoint), "keep": keep_count, "removed": len(removed), "artifact_note": state["rollback_note"]}, indent=2, sort_keys=True))


def backup_cmd(args: argparse.Namespace) -> None:
    result = backup_run_metadata(Path(args.run_dir), Path(args.to))
    append_event(first_existing(Path(args.run_dir), EVENT_FILES), "metadata_backup", **result)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_recovery_plan(run_dir: Path) -> dict[str, Any]:
    state = read_json(run_dir / "state.json", {})
    manifest = read_json(run_dir / "manifest.json", {})
    events = read_jsonl(first_existing(run_dir, EVENT_FILES))
    run_start_idx = max((idx for idx, row in enumerate(events) if row.get("event") == "run_start"), default=-1)
    current_events = events[run_start_idx + 1 :] if run_start_idx >= 0 else events
    locked = state.get("locked", {})
    tmp_root = run_tmp_root(run_dir, manifest, state)
    artifacts_root = run_artifacts_root(run_dir, manifest, state)
    partials = [str(path) for path in tmp_root.iterdir() if path.is_dir()] if tmp_root.exists() else []
    processes = process_rows_for_run(run_dir)
    status_model = active_work_status(
        state,
        current_events,
        processes,
        next(
            (
                row
                for row in reversed(current_events)
                if row.get("event")
                in {"baseline_quant_start", "baseline_ppl_start", "tensor_start", "quant_start", "ppl_start"}
            ),
            {},
        ),
    )
    runner_active = bool(status_model["runner_processes"])
    plan = {
        "run_dir": str(run_dir),
        "status": state.get("run_status"),
        "runner_active": runner_active,
        "active_health": status_model["health"],
        "active_reason": status_model["reason"],
        "interrupted": bool(status_model["stale"]),
        "active_pid": status_model["expected_pid"],
        "active_pid_alive": status_model["expected_pid_alive"],
        "locked_count": len(locked),
        "last_tensor": state.get("last_tensor"),
        "current_ppl": state.get("current_ppl"),
        "baseline_invalid_after_rollback": bool(state.get("baseline_invalid_after_rollback")),
        "partials": partials,
        "disk_free_gb": round(disk_free_gb(run_dir), 3),
        "tmp_size_bytes": dir_size(tmp_root),
        "artifact_size_bytes": dir_size(artifacts_root),
        "resume_command": f"cerebellum resume {run_dir}",
        "safe_partial_cleanup_command": f"cerebellum cleanup {run_dir} --partials --yes",
        "backup_command": f"cerebellum backup {run_dir} --to BACKUP_ROOT",
        "notes": [],
    }
    if runner_active:
        plan["notes"].append("runner is active; do not cleanup partial temp unless intentionally stopping/forcing")
    if status_model["stale"]:
        plan["notes"].append("last start event has no live process; resume will retest the partial tensor")
    if partials and not runner_active:
        plan["notes"].append("partial temp exists and can be deleted before resume; current tensor will be retested")
    if state.get("baseline_invalid_after_rollback"):
        plan["notes"].append("next resume will rebuild baseline GGUF from rolled-back tensor types")
    if manifest.get("scratch_root"):
        plan["scratch_root"] = str(scratch_run_root(run_dir, manifest, state))
        plan["notes"].append(f"heavy temp/artifacts are under scratch_root {manifest.get('scratch_root')}")
    return plan


def resume_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = read_json(run_dir / "manifest.json", {})
    state = read_json(run_dir / "state.json", {})
    if not manifest and not state:
        raise SystemExit(f"no Cerebellum manifest/state found in {run_dir}")
    source = manifest.get("source_gguf") or state.get("source_gguf")
    corpus = manifest.get("corpus") or state.get("corpus")
    if not source:
        raise SystemExit("cannot resume: source_gguf missing from manifest/state")
    if not corpus:
        raise SystemExit("cannot resume: corpus missing from manifest/state")
    ns = argparse.Namespace(
        source_gguf=source,
        corpus=corpus,
        profile=manifest.get("ppl_profile") or state.get("ppl_profile") or "custom",
        metric=manifest.get("ablation_metric") or state.get("ablation_metric") or "ppl",
        family=manifest.get("model_family") or state.get("model_family"),
        model_name=manifest.get("model_name") or state.get("model_name"),
        source_name=manifest.get("source_name") or state.get("source_name"),
        data_root=None,
        run_name=manifest.get("run_id") or state.get("run_id") or run_dir.name,
        run_dir=str(run_dir),
        tensor_file=manifest.get("tensor_file"),
        layers=",".join(str(layer) for layer in manifest.get("layers") or []) if manifest.get("layers") else None,
        tensor_regex=manifest.get("tensor_regex"),
        scratch_root=manifest.get("scratch_root"),
        backup_root=args.backup_root or manifest.get("backup_root"),
        base_type=manifest.get("base_type") or state.get("base_type") or "Q4_K_M",
        start_type=manifest.get("start_type") or state.get("start_type") or "q4_K",
        levels=",".join(manifest.get("levels") or state.get("levels") or DEFAULT_LEVELS),
        imatrix=manifest.get("imatrix"),
        quantize_bin=manifest.get("quantize_bin") or DEFAULT_QUANTIZE,
        perplexity_bin=manifest.get("perplexity_bin") or DEFAULT_PERPLEXITY,
        pure_quant=bool(manifest.get("pure_quant", state.get("pure_quant", False))),
        gpu_layers=manifest.get("gpu_layers", 99),
        ctx_size=manifest.get("ctx_size", 2048),
        chunks=manifest.get("chunks"),
        max_temp_gb=manifest.get("max_temp_gb", 80.0),
        min_free_gb=args.min_free_gb if args.min_free_gb is not None else manifest.get("min_free_gb", 40.0),
        hard_free_floor_gb=args.hard_free_floor_gb if args.hard_free_floor_gb is not None else manifest.get("hard_free_floor_gb", 10.0),
        distrobox=args.distrobox if args.distrobox is not None else manifest.get("distrobox"),
        quant_timeout=manifest.get("quant_timeout", 1800),
        ppl_timeout=manifest.get("ppl_timeout", 900),
        keep_losers=False,
        no_keep_winners=False,
        commit_locks=bool(manifest.get("commit_locks", state.get("commit_locks", True))),
        low_space=False if args.no_low_space else (args.low_space or bool(manifest.get("low_space"))),
        serial_candidates=False if args.no_low_space else (bool(manifest.get("serial_candidates")) or args.low_space),
        prune_measured_candidates=bool(manifest.get("prune_measured_candidates", state.get("prune_measured_candidates", True))),
        plain=args.plain,
        no_color=args.no_color,
        backup_every=1,
        token_embedding_type=manifest.get("token_embedding_type", "f16"),
        noise_pct=manifest.get("noise_pct", 0.0),
    )
    run_from_namespace(ns)


def recover_cmd(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir)
    plan = build_recovery_plan(run_dir)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(color("Cerebellum recovery plan", "36;1", True))
    print(f"run_dir      : {plan['run_dir']}")
    print(f"status       : {plan['status']} runner_active={plan['runner_active']}")
    print(f"health       : {plan['active_health']} - {plan['active_reason']}")
    print(f"locked       : {plan['locked_count']} last={plan['last_tensor']}")
    print(f"ppl          : {plan['current_ppl']}")
    print(f"disk         : {plan['disk_free_gb']} GiB free")
    print(f"storage      : tmp={fmt_bytes(plan['tmp_size_bytes'])} artifacts={fmt_bytes(plan['artifact_size_bytes'])}")
    print(f"partials     : {len(plan['partials'])}")
    for note in plan["notes"]:
        print(f"note         : {note}")
    print("\nCommands:")
    print(f"  {plan['resume_command']}")
    print(f"  {plan['safe_partial_cleanup_command']}")
    print(f"  {plan['backup_command']}")


def clip(value: Any, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return (text[: width - 1] + "…") if width > 8 else text[:width]


def grid_line(left: str, right: str, width: int) -> str:
    inner = width - 4
    left_w = max(36, inner * 2 // 3)
    right_w = inner - left_w - 1
    return f"║ {ansi_pad(left, left_w)}│{ansi_pad(right, right_w)} ║"


def print_heavy_box(title: str, lines: list[str], width: int, code: str, enabled: bool) -> None:
    top = "╔═ " + title + " " + "═" * max(0, width - len(title) - 5) + "╗"
    print(color(top, code, enabled))
    for line in lines:
        if line.startswith("║"):
            print(line)
        else:
            for wrapped in ansi_wrap(line, width - 4):
                print(f"║ {ansi_pad(wrapped, width - 4)} ║")
    print(color("╚" + "═" * (width - 2) + "╝", code, enabled))


def grid_watch_cmd(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir)
    enabled = not args.no_color and not args.plain
    public_view = bool(getattr(args, "public", False))
    try:
        while True:
            model = build_watch_model(run_dir, args.stall_warn_seconds, args.stall_fail_seconds)
            state = model["state"]
            manifest = model["manifest"]
            active = model["active"]
            candidates = model["candidates"]
            events = model["events"]
            processes = model["active_processes"]
            health = model["health"]
            gpu_info = model["gpu"]
            terminal_w = shutil.get_terminal_size((118, 40)).columns
            width = max(96, min(132, terminal_w))
            os.system("clear" if os.name != "nt" else "cls")
            run_id = manifest.get("run_id") or state.get("run_id") or run_dir.name
            if public_view:
                title = f" CEREBELLUM  {state.get('run_status')}  public-safe "
            else:
                title = f" CEREBELLUM  {state.get('model_family')}/{state.get('model_name')}  {manifest.get('ppl_profile') or state.get('ppl_profile') or 'custom'}  {state.get('run_status')} "
            print(color("╔" + "═" * (width - 2) + "╗", "36;1", enabled))
            print(color("║" + title.center(width - 2) + "║", "36;1", enabled))
            print(color("╚" + "═" * (width - 2) + "╝", "36;1", enabled))
            print()

            eta = eta_grid_values(state, model["active_age"], model["total"])
            progress_left = f"{model['bar']} {model['progress']}"
            if not public_view:
                progress_left = f"{progress_left}  ppl {state.get('current_ppl')}"
            resource_bits = []
            if gpu_info:
                gpu = gpu_info[0]
                resource_bits.append(f"GPU {gpu['util']}% {gpu['mem_used']}/{gpu['mem_total']} MiB")
            if processes:
                jobs = ",".join(sorted({row["kind"] for row in processes}))
                resource_bits.append(f"jobs {jobs}")
            else:
                resource_bits.append(f"jobs {health['health']}")
            job_label = f"{active.get('event')}  age {fmt_seconds(model['active_age'])}"
            if health["stale"]:
                job_label = f"interrupted  {active.get('event')} age {fmt_seconds(model['active_age'])}"
            if public_view:
                public_job = "active" if health["health"] in {"active", "waiting"} else health["health"]
                overview = [
                    grid_line(color("progress  ", "90", enabled) + color(progress_left, "32;1", enabled), color("resources  ", "90", enabled) + color(resource_bits[0], "36;1", enabled), width),
                    grid_line(color("cpu/gpu   ", "90", enabled) + color((" | ".join(row["kind"] for row in processes[:3]) if processes else "idle"), "36;1", enabled), color("jobs       ", "90", enabled) + color(("  ".join(resource_bits[1:]) if len(resource_bits) > 1 else "idle"), "36;1", enabled), width),
                    grid_line(color("phase     ", "90", enabled) + color(public_job, health["code"], enabled), color("disk       ", "90", enabled) + color(f"{disk_free_gb(run_dir):.1f} GiB free", "36", enabled), width),
                    grid_line(color("health    ", "90", enabled) + color(health["health"], health["code"], enabled), color("confidence ", "90", enabled) + color(eta["confidence"], "32;1" if eta["confidence"] == "high" else "33;1", enabled), width),
                    grid_line(color("eta       ", "90", enabled) + color(f"current {eta['current']} total {eta['total']}", "36;1", enabled), color("done       ", "90", enabled) + color(eta["completion_at"], "36;1", enabled), width),
                ]
            else:
                eta_detail = model.get("eta_detail") or {}
                job_eta = eta_detail.get("job") or {}
                tensor_eta = eta_detail.get("tensor") or {}
                phase_eta = eta_detail.get("phase") or {}
                flow_eta = eta_detail.get("flow") or {}
                overview = [
                    grid_line(color("progress  ", "90", enabled) + color(progress_left, "32;1", enabled), color("resources  ", "90", enabled) + color(resource_bits[0], "36;1", enabled), width),
                    grid_line(color("cpu/gpu   ", "90", enabled) + color((" | ".join(row["kind"] for row in processes[:3]) if processes else "idle"), "36;1", enabled), color("jobs       ", "90", enabled) + color(("  ".join(resource_bits[1:]) if len(resource_bits) > 1 else "idle"), "36;1", enabled), width),
                    grid_line(color("tensor    ", "90", enabled) + color(f"{active.get('tensor')}  {active.get('level')}", "33;1", enabled), color("disk       ", "90", enabled) + color(f"{disk_free_gb(run_dir):.1f} GiB free", "36", enabled), width),
                    grid_line(color("current   ", "90", enabled) + color(f"job {job_label}", health["code"], enabled), color("gguf       ", "90", enabled) + color(f"base {fmt_bytes_dense(model['baseline_size'])} active {fmt_bytes_dense(model['active_size'])}", "36;1", enabled), width),
                    grid_line(color("storage   ", "90", enabled) + color(f"tmp {fmt_bytes(model['tmp_size'])} artifacts {fmt_bytes(model['artifacts_size'])}", "36;1", enabled), color("pid        ", "90", enabled) + color(str(health["expected_pid"] or "-"), health["code"], enabled), width),
                    grid_line(color("health    ", "90", enabled) + color(str(health["reason"]), health["code"], enabled), color("confidence ", "90", enabled) + color(eta_detail.get("confidence", eta["confidence"]), "32;1" if eta_detail.get("confidence", eta["confidence"]) == "high" else "33;1", enabled), width),
                    grid_line(color("job eta   ", "90", enabled) + color(f"{job_eta.get('label', '-')} est {job_eta.get('estimate', '-')} rem {job_eta.get('remaining', '-')}", "36;1", enabled), color("tensorETA ", "90", enabled) + color(f"est {tensor_eta.get('estimate', '-')} rem {tensor_eta.get('remaining', '-')}", "36;1", enabled), width),
                    grid_line(color("phase eta ", "90", enabled) + color(f"rem {phase_eta.get('remaining', '-')} ({phase_eta.get('tensors_remaining', '-')} tensors)", "36;1", enabled), color("flow eta  ", "90", enabled) + color(f"rem {flow_eta.get('remaining', '-')}", "36;1", enabled), width),
                    grid_line(color("eta src   ", "90", enabled) + color(str(eta_detail.get("source") or "-"), "36;1", enabled), color("floor      ", "90", enabled) + color(f"{manifest.get('hard_free_floor_gb', 10.0)} GiB hard floor", "31;1" if disk_free_gb(run_dir) < 20 else "36", enabled), width),
                ]
            print_heavy_box("OPERATIONS", overview, width, "34;1", enabled)
            print()

            if public_view:
                if args.once:
                    return
                time.sleep(args.interval)
                continue
            else:
                schema = str(manifest.get("schema") or state.get("schema") or "")
                is_group_scan = schema.startswith("cerebellum.group_scan")
                if not is_group_scan:
                    measure_lines = [f"{'quant':<7} {'ppl':<12} {'delta':<12} {'size':<12} tensor"]
                    measure_lines.append("─" * (width - 4))
                    visible_candidates = limited_tail(candidates, args.measurements_limit)
                    for row in visible_candidates:
                        delta = row.get("delta")
                        delta_s = "-" if delta is None else f"{delta:+.4f}"
                        verdict, verdict_code = candidate_measurement_verdict(row, candidates)
                        ppl_value = row.get("ppl")
                        ppl_s = "-" if ppl_value is None else str(ppl_value)
                        line = (
                            color(f"{row.get('level', '-'):<7}", "35;1", enabled)
                            + color(f"{ppl_s:<12}", "33;1", enabled)
                            + color(f"{delta_s:<12}", delta_code(delta), enabled)
                            + color(f"{fmt_bytes_dense(row.get('size_bytes')):<12}", size_code(row.get("size_bytes"), model["baseline_size"]), enabled)
                            + color(f"{row.get('tensor', '')}", "33", enabled)
                            + " "
                            + color(verdict, verdict_code, enabled)
                        )
                        measure_lines.append(line)
                    print_heavy_box("RECENT MEASUREMENTS", measure_lines, width, "32;1", enabled)
                    print()
                ablation_root = classic_ablation_root(run_dir)
                if (ablation_root / "forward").exists() or (ablation_root / "reverse").exists():
                    print_heavy_box("FORWARD ABLATION", classic_ablation_phase_lines(ablation_root, "forward", enabled), width, "32;1", enabled)
                    print()
                    print_heavy_box("REVERSE ABLATION", classic_ablation_phase_lines(ablation_root, "reverse", enabled), width, "35;1", enabled)
                    print()
                commit_locks = bool(manifest.get("commit_locks", True))
                if is_group_scan:
                    layer_title = "GROUP VERDICTS"
                    layer_lines = ["Forward and reverse no-commit group verdicts are shown above."]
                else:
                    layer_title = "LOCKED LAYER MAP" if commit_locks else "SCAN FINDINGS"
                    layer_lines = locked_layer_lines(state) or (["No locked tensors yet."] if commit_locks else ["No scan findings yet."])
                print_heavy_box(layer_title, layer_lines, width, "35;1", enabled)
                print()
                strategy_lines = golden_cow_watch_lines(model.get("golden_cow_audit") or {}, limit=8)
                print_heavy_box("LEGACY / GATED STRATEGY", strategy_lines, width, "31;1", enabled)
                print()
                flow_lines = legacy_flow_watch_lines(model.get("legacy_flow") or {}, state, active, manifest)
                print_heavy_box("LEGACY FLOW", flow_lines, width, "36;1", enabled)
            print()

            event_lines = [
                f"{row.get('event')} {row.get('level', '')} {row.get('tensor', '')}".strip()
                for row in limited_tail(events, args.events_limit)
            ] or ["No events yet."]
            print_heavy_box("EVENT STRIP", event_lines, width, "33;1", enabled)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


def build_watch_model(
    run_dir: Path,
    stall_warn_seconds: float = 300.0,
    stall_fail_seconds: float = 900.0,
) -> dict[str, Any]:
    state = read_json(run_dir / "state.json", {})
    manifest = read_json(run_dir / "manifest.json", {})
    events = read_jsonl(first_existing(run_dir, EVENT_FILES))
    candidates = read_jsonl(first_existing(run_dir, CANDIDATE_FILES))
    run_start_idx = max((idx for idx, row in enumerate(events) if row.get("event") == "run_start"), default=-1)
    current_events = events[run_start_idx + 1 :] if run_start_idx >= 0 else events
    run_start = events[run_start_idx] if run_start_idx >= 0 else {}
    run_start_ts = event_timestamp(run_start.get("timestamp_utc")) if run_start else None
    if run_start_ts is not None:
        candidates = [
            row
            for row in candidates
            if (candidate_ts := event_timestamp(row.get("timestamp_utc"))) is not None and candidate_ts >= run_start_ts
        ]
    current_tensors = {row.get("tensor") for row in state.get("tested", []) if row.get("tensor")}
    active_tensor = next((row.get("tensor") for row in reversed(current_events) if row.get("event") == "tensor_start" and row.get("tensor")), None)
    if active_tensor:
        current_tensors.add(active_tensor)
    visible_candidates = [row for row in candidates if row.get("tensor") in current_tensors] if current_tensors else []
    visible_candidates.extend(in_progress_candidate_rows(current_events, candidates))
    status = state.get("run_status")
    terminal_events = {"run_stopped", "run_finish", "tensor_interrupted", "signal_received", "rollback_finish"}
    if status in {"stopped", "complete", "failed"}:
        active = next((row for row in reversed(current_events) if row.get("event") in terminal_events), {})
    else:
        active = next(
            (
                row
                for row in reversed(current_events)
                if row.get("event")
                in {"baseline_quant_start", "baseline_ppl_start", "tensor_start", "quant_start", "ppl_start"}
            ),
            {},
        )
    total = run_start.get("tensors") or next((row.get("total") for row in reversed(current_events) if row.get("total")), None)
    commit_locks = bool(manifest.get("commit_locks", True))
    progress_count = len(state.get("locked", {})) if commit_locks else len(state.get("tested", []))
    bar, progress = progress_bar(progress_count, total, width=22)
    active_age = event_age_seconds(active)
    last_age = event_age_seconds(events[-1]) if events else None
    processes = process_rows_for_run(run_dir)
    active_processes = [row for row in processes if row["kind"] in {"quantize", "ppl"}]
    status_model = active_work_status(state, events, processes, active, stall_warn_seconds, stall_fail_seconds)
    eta, eta_basis = estimate_eta(state, active_age, total)
    tmp_root = run_tmp_root(run_dir, manifest, state)
    artifacts_root = run_artifacts_root(run_dir, manifest, state)
    baseline_path = Path(state.get("baseline_path") or artifacts_root / "current_baseline.gguf")
    active_path = active.get("tmp_output") or active.get("output") or active.get("model")
    flow_context = legacy_flow_context(run_dir)
    eta_detail = eta_detail_values(run_dir, state, manifest, events, candidates, active, active_age, total, flow_context)
    return {
        "state": state,
        "manifest": manifest,
        "events": events,
        "candidates": visible_candidates,
        "all_candidates": candidates,
        "active": active,
        "processes": processes,
        "active_processes": active_processes,
        "health": status_model,
        "gpu": gpu_rows(),
        "bar": bar,
        "progress": progress,
        "total": total,
        "active_age": active_age,
        "last_age": last_age,
        "eta": eta,
        "eta_basis": eta_basis,
        "baseline_path": baseline_path,
        "baseline_size": path_size(baseline_path) if baseline_path.exists() else None,
        "active_path": active_path,
        "active_size": path_size(Path(active_path)) if active_path else None,
        "tmp_size": dir_size(tmp_root),
        "artifacts_size": dir_size(artifacts_root),
        "golden_cow_audit": golden_cow_audit(state, manifest, flow_context),
        "legacy_flow": flow_context,
        "eta_detail": eta_detail,
    }


def tui_watch_cmd(args: argparse.Namespace) -> None:
    import curses

    run_dir = resolve_run_dir(args.run_dir)
    panes = ["events", "measurements", "flow", "strategy", "processes", "files"]
    offsets = {name: 0 for name in panes}
    active_pane = 0

    def draw(stdscr: Any) -> None:
        nonlocal active_pane, offsets
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(max(250, int(args.interval * 1000)))
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            for idx, fg in enumerate([curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_RED, curses.COLOR_WHITE], 1):
                curses.init_pair(idx, fg, -1)
        while True:
            model = build_watch_model(run_dir, args.stall_warn_seconds, args.stall_fail_seconds)
            h, w = stdscr.getmaxyx()
            stdscr.erase()
            state = model["state"]
            manifest = model["manifest"]
            active = model["active"]
            eta_detail = model.get("eta_detail") or {}
            phase_eta = (eta_detail.get("phase") or {}).get("remaining", "-")
            flow_eta = (eta_detail.get("flow") or {}).get("remaining", "-")
            title = " CEREBELLUM LIVE "
            stdscr.addnstr(0, max(0, (w - len(title)) // 2), title, w - 1, curses.color_pair(1) | curses.A_BOLD)
            summary = [
                f"run {manifest.get('run_id') or state.get('run_id') or run_dir.name}",
                f"model {state.get('model_family')}/{state.get('model_name')}  status {state.get('run_status')}  profile {manifest.get('ppl_profile') or state.get('ppl_profile') or 'custom'}",
                f"progress {model['bar']} {model['progress']}  ppl {state.get('current_ppl')}  phase eta {phase_eta}  flow eta {flow_eta} ({eta_detail.get('confidence', model['eta_basis'])})",
                f"active {active.get('event')} {active.get('level')} {active.get('tensor')}  age {fmt_seconds(model['active_age'])}  last event {fmt_seconds(model['last_age'])}",
                f"sizes current {fmt_bytes_dense(model['baseline_size'])}  active {fmt_bytes_dense(model['active_size'])}  tmp {fmt_bytes_dense(model['tmp_size'])}  artifacts {fmt_bytes_dense(model['artifacts_size'])}  disk {disk_free_gb(run_dir):.1f} GiB free",
            ]
            for y, line in enumerate(summary, 2):
                if y < h:
                    stdscr.addnstr(y, 0, line, w - 1, curses.color_pair(6))
            tab_y = min(8, h - 2)
            x = 0
            for idx, pane in enumerate(panes):
                label = f" {pane.upper()} "
                attr = curses.A_REVERSE | curses.color_pair(2) if idx == active_pane else curses.color_pair(1)
                stdscr.addnstr(tab_y, x, label, max(0, w - x - 1), attr)
                x += len(label) + 1
            body_top = tab_y + 2
            body_h = max(1, h - body_top - 2)
            pane = panes[active_pane]
            lines: list[str] = []
            if pane == "events":
                for row in reversed(model["events"]):
                    lines.append(f"{row.get('timestamp_utc', '')[-13:]} {row.get('event', ''):<20} {row.get('level', ''):<6} {row.get('tensor', '')}")
            elif pane == "measurements":
                for row in reversed(model["candidates"]):
                    delta = row.get("delta")
                    delta_s = "-" if delta is None else f"{delta:+.4f}"
                    marker, _marker_code = delta_marker(delta)
                    verdict, _verdict_code = ablation_verdict(delta, str(row.get("tensor") or "tensor"))
                    lines.append(f"{row.get('level', '-'):<6} ppl={row.get('ppl', '-')} delta={delta_s:<12} q={fmt_bytes_dense(row.get('size_bytes')):<12} {row.get('tensor', '')} {marker} {verdict}")
            elif pane == "strategy":
                lines.extend(golden_cow_watch_lines(model.get("golden_cow_audit") or {}, limit=body_h - 1))
            elif pane == "flow":
                lines.extend(legacy_flow_watch_lines(model.get("legacy_flow") or {}, state, active, manifest))
            elif pane == "processes":
                for row in model["processes"]:
                    lines.append(f"{row['kind']:<9} pid={row['pid']:<7} etime={row['etime']:<9} cpu={row['pcpu']:>6}% mem={row['pmem']:>5}% {row['cmd'][:90]}")
                for gpu in model["gpu"]:
                    lines.append(f"gpu       cuda:{gpu['index']} util={gpu['util']}% vram={gpu['mem_used']}/{gpu['mem_total']} MiB power={gpu['power']} W {gpu['name']}")
            else:
                paths = [
                    ("run_dir", run_dir),
                    ("baseline", model["baseline_path"]),
                ]
                if model["active_path"]:
                    paths.append(("active", Path(model["active_path"])))
                for label, path in paths:
                    lines.append(f"{label:<10} {fmt_bytes(path_size(path) if path.exists() else None):<10} {path}")
            max_offset = max(0, len(lines) - body_h)
            offsets[pane] = max(0, min(offsets[pane], max_offset))
            visible = lines[offsets[pane] : offsets[pane] + body_h]
            for idx, line in enumerate(visible):
                y = body_top + idx
                attr = curses.color_pair(6)
                if "delta=-" in line:
                    attr = curses.color_pair(2)
                elif "delta=+" in line or "failure" in line.lower():
                    attr = curses.color_pair(5)
                elif "ppl" in line or "quantize" in line:
                    attr = curses.color_pair(3)
                stdscr.addnstr(y, 0, line, w - 1, attr)
            footer = "Tab pane | arrows/PageUp/PageDown scroll | r reset | q quit | compact: watch without --tui"
            stdscr.addnstr(h - 1, 0, footer, w - 1, curses.color_pair(1))
            stdscr.refresh()
            key = stdscr.getch()
            if key in {ord("q"), ord("Q"), 27}:
                return
            if key in {9, curses.KEY_RIGHT}:
                active_pane = (active_pane + 1) % len(panes)
            elif key == curses.KEY_LEFT:
                active_pane = (active_pane - 1) % len(panes)
            elif key == curses.KEY_DOWN:
                offsets[panes[active_pane]] += 1
            elif key == curses.KEY_UP:
                offsets[panes[active_pane]] -= 1
            elif key == curses.KEY_NPAGE:
                offsets[panes[active_pane]] += body_h
            elif key == curses.KEY_PPAGE:
                offsets[panes[active_pane]] -= body_h
            elif key in {ord("r"), ord("R")}:
                offsets = {name: 0 for name in panes}

    curses.wrapper(draw)


def runs_cmd(args: argparse.Namespace) -> None:
    root = Path(args.data_root) if args.data_root else default_data_root()
    manifests = sorted(root.glob("families/*/*/sources/*/runs/*/manifest.json"))
    data = []
    for manifest in manifests:
        try:
            item = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue
        state_path = manifest.parent / "state.json"
        if state_path.exists():
            try:
                item["state"] = json.loads(state_path.read_text())
            except json.JSONDecodeError:
                item["state"] = {"run_status": "corrupt"}
        state = item.get("state", {})
        events = read_jsonl(first_existing(manifest.parent, EVENT_FILES))
        total = next((row.get("total") for row in reversed(events) if row.get("total")), None)
        locked = len(state.get("locked", {}))
        item["progress"] = {"locked": locked, "total": total, "pct": (locked / total * 100.0) if total else None}
        item["ppl_profile"] = item.get("ppl_profile") or state.get("ppl_profile") or "custom"
        item["run_dir"] = str(manifest.parent)
        if args.family and args.family not in str(item.get("model_family")):
            continue
        if args.model and args.model not in str(item.get("model_name")):
            continue
        if args.status and args.status != str(state.get("run_status")):
            continue
        if args.profile and args.profile != str(item.get("ppl_profile")):
            continue
        data.append(item)
    if args.json:
        print(json.dumps({"runs": data}, indent=2, sort_keys=True))
        return
    if not data:
        print("no runs")
        return
    headers = ["status", "profile", "progress", "ppl", "model", "run"]
    print(f"{headers[0]:<10} {headers[1]:<10} {headers[2]:<16} {headers[3]:<12} {headers[4]:<28} {headers[5]}")
    print(f"{'-'*10} {'-'*10} {'-'*16} {'-'*12} {'-'*28} {'-'*20}")
    for item in data:
        state = item.get("state", {})
        progress = item.get("progress", {})
        pct = progress.get("pct")
        progress_s = f"{progress.get('locked')}/{progress.get('total') or '?'}"
        if pct is not None:
            progress_s += f" {pct:.1f}%"
        model_s = f"{item.get('model_family')}/{item.get('model_name')}"
        print(
            f"{state.get('run_status', '?'):<10} {item.get('ppl_profile', '-'):<10} "
            f"{progress_s:<16} {str(state.get('current_ppl')):<12} {model_s:<28} {item.get('run_id')}"
        )


def project_cmd(args: argparse.Namespace) -> None:
    root = Path(args.data_root) if args.data_root else default_data_root()
    projects = discover_projects(root)
    if args.family:
        projects = [row for row in projects if args.family in str(row.get("family"))]
    if args.model:
        projects = [row for row in projects if args.model in str(row.get("model_name"))]
    if args.source:
        projects = [row for row in projects if args.source in str(row.get("source_name"))]
    if args.json:
        print(json.dumps({"data_root": str(root), "projects": projects}, indent=2, sort_keys=True))
        return
    print("Cerebellum projects")
    print(f"data_root: {root}")
    print(f"{'family':<18} {'model':<28} {'source':<18} {'runs':>4} {'imatrix':<7} source_root")
    for row in projects:
        imatrix = "yes" if row.get("imatrix") and Path(str(row.get("imatrix"))).exists() else "path" if row.get("imatrix") else "no"
        print(f"{str(row.get('family')):<18} {str(row.get('model_name')):<28} {str(row.get('source_name')):<18} {len(row.get('runs', [])):>4} {imatrix:<7} {row.get('source_root')}")
        if row.get("next_command"):
            print(f"  next: {row['next_command']}")


def db_benchmark_leaderboard(db: Path, suite: str = "release", weights: dict[str, float] | None = None, limit: int = 20) -> dict[str, Any]:
    conn = sqlite3.connect(db)
    try:
        ensure_hill_tables(conn)
    finally:
        conn.close()
    records = sqlite_rows(
        db,
        """
        SELECT r.model, r.suite, r.benchmark_key, r.benchmark, r.metric, r.value,
               r.path, r.size_gib, r.release_metadata_json, i.id AS ingest_id,
               i.results_dir, i.updated_at
        FROM cerebellum_benchmark_results r
        JOIN cerebellum_benchmark_ingests i ON i.id = r.ingest_id
        JOIN (
            SELECT model, MAX(id) AS ingest_id
            FROM cerebellum_benchmark_ingests
            WHERE suite = ? AND ready = 1
            GROUP BY model
        ) latest ON latest.ingest_id = i.id
        WHERE r.suite = ?
        ORDER BY r.model, r.benchmark_key, r.metric
        """,
        (suite, suite),
    )
    for row in records:
        try:
            metadata = json.loads(str(row.get("release_metadata_json") or "{}"))
        except json.JSONDecodeError:
            metadata = {}
        row["release_metadata"] = metadata
        if row.get("size_gib") is None and metadata.get("size_gib") is not None:
            row["size_gib"] = float(metadata["size_gib"])
        row.pop("release_metadata_json", None)
    weight_policy = leaderboard_weight_policy(suite, weights)
    leaderboard = benchmark_leaderboard(records, suite, weights=weight_policy)[: max(1, limit)]
    source_ingests = sorted(
        {
            (str(row["model"]), int(row["ingest_id"]), str(row["results_dir"]), str(row["updated_at"]))
            for row in records
        }
    )
    return {
        "schema": "cerebellum.db_benchmark_leaderboard.v1",
        "suite": suite,
        "suite_purpose": BENCHMARK_SUITE_PURPOSES.get(suite, ""),
        "weight_policy": weight_policy,
        "source_ingests": [
            {"model": model, "ingest_id": ingest_id, "results_dir": results_dir, "updated_at": updated_at}
            for model, ingest_id, results_dir, updated_at in source_ingests
        ],
        "records": len(records),
        "leaderboard": leaderboard,
    }


def db_benchmark_leaderboard_markdown(payload: dict[str, Any]) -> str:
    rows = []
    for idx, row in enumerate(payload["leaderboard"], 1):
        size = "-" if row.get("size_gib") is None else f"{float(row['size_gib']):.2f}"
        density = "-" if row.get("score_per_gib") is None else f"{float(row['score_per_gib']):.2f}"
        rows.append(
            [
                str(idx),
                row["model"],
                f"{float(row['average_score']):.2f}",
                str(row["benchmarks"]),
                size,
                density,
            ]
        )
    parts = [
        "# Cerebellum Benchmark Leaderboard",
        "",
        f"suite: `{payload['suite']}`",
        f"purpose: `{payload.get('suite_purpose') or '-'}`",
        f"records: `{payload['records']}`",
        "",
        markdown_table(["Rank", "Model", "Avg score", "Benchmarks", "Size GiB", "Score/GiB"], rows) if rows else "No ready ingested benchmark rows found.",
    ]
    if payload["source_ingests"]:
        ingest_rows = [[row["model"], str(row["ingest_id"]), row["results_dir"]] for row in payload["source_ingests"]]
        parts.extend(["", "## Source Ingests", "", markdown_table(["Model", "Ingest", "Results Dir"], ingest_rows)])
    return "\n".join(parts) + "\n"


def db_cmd(args: argparse.Namespace) -> None:
    db = Path(args.db)
    if args.db_cmd == "families":
        rows = sqlite_rows(
            db,
            """
            SELECT mf.id, mf.name, mf.vendor, COUNT(DISTINCT bm.id) AS model_count,
                   COUNT(DISTINCT b.id) AS build_count
            FROM model_families mf
            LEFT JOIN base_models bm ON bm.family_id = mf.id
            LEFT JOIN model_sources ms ON ms.base_model_id = bm.id
            LEFT JOIN builds b ON b.source_id = ms.id
            GROUP BY mf.id ORDER BY mf.name
            """,
        )
    elif args.db_cmd == "models":
        if args.family:
            rows = sqlite_rows(
                db,
                """
                SELECT mf.name AS family, bm.name, bm.hf_repo, bm.total_params_b,
                       bm.active_params_b, bm.num_layers, bm.context_length,
                       COUNT(DISTINCT b.id) AS build_count
                FROM base_models bm
                JOIN model_families mf ON mf.id = bm.family_id
                LEFT JOIN model_sources ms ON ms.base_model_id = bm.id
                LEFT JOIN builds b ON b.source_id = ms.id
                WHERE mf.name LIKE ?
                GROUP BY bm.id ORDER BY bm.name
                """,
                (args.family,),
            )
        else:
            rows = sqlite_rows(
                db,
                """
                SELECT mf.name AS family, bm.name, bm.hf_repo, bm.total_params_b,
                       bm.active_params_b, bm.num_layers, bm.context_length,
                       COUNT(DISTINCT b.id) AS build_count
                FROM base_models bm
                JOIN model_families mf ON mf.id = bm.family_id
                LEFT JOIN model_sources ms ON ms.base_model_id = bm.id
                LEFT JOIN builds b ON b.source_id = ms.id
                GROUP BY bm.id ORDER BY mf.name, bm.name
                """,
            )
    elif args.db_cmd == "builds":
        rows = sqlite_rows(
            db,
            """
            SELECT family, base_model, source_name, version, tag, size_gb,
                   bpw, override_count, scores
            FROM build_scores
            ORDER BY family, base_model, version
            """,
        )
    elif args.db_cmd == "benchmarks":
        conn = sqlite3.connect(db)
        try:
            ensure_hill_tables(conn)
        finally:
            conn.close()
        rows = sqlite_rows(
            db,
            """
            SELECT benchmark_key AS benchmark, COUNT(*) AS runs,
                   ROUND(MAX(value), 3) AS best, ROUND(AVG(value), 3) AS avg
            FROM cerebellum_benchmark_results
            GROUP BY benchmark_key ORDER BY benchmark_key
            """,
        )
    elif args.db_cmd == "leaderboard":
        payload = db_benchmark_leaderboard(db, suite=args.suite, weights=parse_weight_specs(args.weight), limit=args.limit)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(db_benchmark_leaderboard_markdown(payload), end="")
        return
    elif args.db_cmd in {"runs", "hill-runs"}:
        rows = sqlite_rows(
            db,
            """
            SELECT run_id, model_family, model_name, source_name, status,
                   current_ppl, locked_count, candidate_count, run_dir
            FROM hill_runs
            ORDER BY updated_at DESC, run_id DESC
            """,
        )
    elif args.db_cmd == "import-run":
        result = import_run_to_db(db, Path(args.run_dir))
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"imported {result['run_id']} candidates={result['candidates']}")
        return
    elif args.db_cmd == "query":
        if not args.sql:
            raise SystemExit("--sql required for db query")
        rows = sqlite_rows(db, args.sql)
    else:
        raise SystemExit(f"unknown db command: {args.db_cmd}")
    if args.json:
        print(json.dumps({"rows": rows}, indent=2, sort_keys=True))
        return
    if not rows:
        print("no rows")
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(str(row.get(h, ""))) for row in rows[:100])) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def build_report(run_dir: Path) -> dict[str, Any]:
    loaded = load_run(run_dir)
    manifest = loaded["manifest"]
    state = loaded["state"]
    candidates = read_jsonl(first_existing(run_dir, CANDIDATE_FILES))
    decisions = state.get("tested", [])
    quant_seconds = sum((row.get("quant_seconds") or 0) for row in candidates)
    ppl_seconds = sum((row.get("ppl_seconds") or 0) for row in candidates)
    by_component: dict[str, dict[str, Any]] = {}
    for row in candidates:
        layer, component = parse_tensor_name(row.get("tensor", ""))
        key = component or "unknown"
        bucket = by_component.setdefault(key, {"component": key, "tests": 0, "best_delta": None, "worst_delta": None})
        delta = row.get("delta")
        bucket["tests"] += 1
        if delta is not None:
            bucket["best_delta"] = delta if bucket["best_delta"] is None else min(bucket["best_delta"], delta)
            bucket["worst_delta"] = delta if bucket["worst_delta"] is None else max(bucket["worst_delta"], delta)
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id") or state.get("run_id") or run_dir.name,
        "model_family": manifest.get("model_family") or state.get("model_family"),
        "model_name": manifest.get("model_name") or state.get("model_name"),
        "source_name": manifest.get("source_name") or state.get("source_name"),
        "status": state.get("run_status"),
        "current_ppl": state.get("current_ppl"),
        "ppl_profile": manifest.get("ppl_profile") or state.get("ppl_profile"),
        "corpus": manifest.get("corpus") or state.get("corpus"),
        "locked_count": len(state.get("locked", {})),
        "candidate_count": len(candidates),
        "decision_count": len(decisions),
        "quant_seconds": quant_seconds,
        "ppl_seconds": ppl_seconds,
        "levels": manifest.get("levels") or state.get("levels"),
        "artifacts": manifest.get("files", {}),
        "recent_decisions": decisions[-20:],
        "component_summary": sorted(by_component.values(), key=lambda x: x["component"]),
    }


def cerebellum_metadata_block(run_dir: Path, gguf: Path | None = None, hash_files: bool = False, private: bool = False) -> dict[str, Any]:
    report = build_report(run_dir)
    manifest = read_json(run_dir / "manifest.json", {})
    files = manifest.get("files", {})
    final_types = first_existing(run_dir, BEST_TYPES_FILES)
    summary = first_existing(run_dir, SUMMARY_JSON_FILES)
    metadata = {
        "cerebellum.tool": "Cerebellum",
        "cerebellum.provenance_schema": "1",
        "cerebellum.model_family": report.get("model_family"),
        "cerebellum.model_name": report.get("model_name"),
        "cerebellum.source_name": report.get("source_name"),
        "cerebellum.ppl_profile": report.get("ppl_profile"),
        "cerebellum.base_type": manifest.get("base_type"),
        "cerebellum.start_type": manifest.get("start_type"),
        "cerebellum.levels": ",".join(manifest.get("levels") or []),
        "cerebellum.locked_count": str(report.get("locked_count")),
        "cerebellum.source_gguf_sha256": sha256_file(Path(manifest["source_gguf"])) if hash_files and manifest.get("source_gguf") else None,
        "cerebellum.final_gguf_sha256": sha256_file(gguf) if hash_files and gguf else None,
    }
    if private:
        metadata.update(
            {
                "cerebellum.run_id": report.get("run_id"),
                "cerebellum.corpus": report.get("corpus"),
                "cerebellum.candidate_count": str(report.get("candidate_count")),
                "cerebellum.current_ppl": str(report.get("current_ppl")),
                "cerebellum.run_dir_sha256": hashlib.sha256(str(run_dir).encode()).hexdigest(),
                "cerebellum.tensor_types_sha256": sha256_file(final_types) if hash_files else None,
                "cerebellum.summary_sha256": sha256_file(summary) if hash_files else None,
            }
        )
        if files:
            metadata["cerebellum.events_file"] = Path(files.get("events", "")).name
            metadata["cerebellum.candidates_file"] = Path(files.get("candidates", "")).name
    return {key: value for key, value in metadata.items() if value is not None}


def inspect_gguf_metadata(gguf: Path) -> dict[str, Any]:
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit("gguf Python package is required to inspect GGUF metadata") from exc
    reader = GGUFReader(str(gguf))
    fields: dict[str, Any] = {}
    for key, gguf_field in reader.fields.items():
        if not key.startswith("cerebellum."):
            continue
        value = getattr(gguf_field, "contents", None)
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        fields[key] = str(value)
    return fields


def gguf_quant_type_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    try:
        from gguf import GGMLQuantizationType

        return GGMLQuantizationType(int(value)).name
    except Exception:
        return str(value)


def inspect_gguf_types(gguf: Path) -> dict[str, Any]:
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit("gguf Python package is required to inspect GGUF tensor types") from exc
    reader = GGUFReader(str(gguf))
    type_counts: dict[str, int] = {}
    component_counts: dict[str, dict[str, int]] = {}
    layer_counts: dict[str, dict[str, int]] = {}
    tensor_types: dict[str, str] = {}
    quantized_tensors = 0
    for tensor in reader.tensors:
        name = str(tensor.name)
        qtype = gguf_quant_type_name(getattr(tensor, "tensor_type", "unknown"))
        tensor_types[name] = qtype
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        if is_quantizable_tensor(name):
            quantized_tensors += 1
        layer, component = parse_tensor_name(name)
        component_key = component or "other"
        component_counts.setdefault(component_key, {})
        component_counts[component_key][qtype] = component_counts[component_key].get(qtype, 0) + 1
        layer_key = f"blk.{layer}" if layer is not None else "global"
        layer_counts.setdefault(layer_key, {})
        layer_counts[layer_key][qtype] = layer_counts[layer_key].get(qtype, 0) + 1
    return {
        "gguf": str(gguf),
        "tensor_count": len(reader.tensors),
        "quantizable_tensor_count": quantized_tensors,
        "type_counts": dict(sorted(type_counts.items())),
        "component_counts": {key: dict(sorted(value.items())) for key, value in sorted(component_counts.items())},
        "layer_counts": {key: dict(sorted(value.items())) for key, value in sorted(layer_counts.items())},
        "tensor_types": dict(sorted(tensor_types.items())),
    }


def inspect_gguf_types_cmd(args: argparse.Namespace) -> None:
    summary = inspect_gguf_types(Path(args.gguf))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"GGUF tensor types: {summary['gguf']}")
    print(f"tensors: {summary['tensor_count']}  quantizable: {summary['quantizable_tensor_count']}")
    print("Type counts:")
    for qtype, count in summary["type_counts"].items():
        print(f"  {qtype:<8} {count}")
    if args.by_component:
        print("\nComponent counts:")
        for component, counts in summary["component_counts"].items():
            cells = "  ".join(f"{qtype}={count}" for qtype, count in counts.items())
            print(f"  {component:<24} {cells}")
    if args.by_layer:
        print("\nLayer counts:")
        for layer, counts in summary["layer_counts"].items():
            cells = "  ".join(f"{qtype}={count}" for qtype, count in counts.items())
            print(f"  {layer:<8} {cells}")


def inspect_gguf_types_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    gguf = query_value(qs, "gguf")
    if not gguf:
        raise ValueError("gguf query param required")
    return argparse.Namespace(
        gguf=gguf,
        by_layer=query_bool(qs, "by_layer"),
        by_component=query_bool(qs, "by_component"),
        json=True,
    )


def compare_count_maps(base: dict[str, int], candidate: dict[str, int]) -> dict[str, dict[str, int]]:
    keys = sorted(set(base) | set(candidate))
    return {
        key: {
            "baseline": int(base.get(key, 0)),
            "candidate": int(candidate.get(key, 0)),
            "delta": int(candidate.get(key, 0)) - int(base.get(key, 0)),
        }
        for key in keys
    }


def compare_nested_count_maps(
    base: dict[str, dict[str, int]],
    candidate: dict[str, dict[str, int]],
) -> dict[str, dict[str, dict[str, int]]]:
    keys = sorted(set(base) | set(candidate))
    return {key: compare_count_maps(base.get(key, {}), candidate.get(key, {})) for key in keys}


QUANT_TYPE_BITS = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.0,
    "Q6_K": 6.0,
    "Q5_K": 5.0,
    "Q4_K": 4.0,
    "Q3_K": 3.0,
    "Q2_K": 2.0,
    "IQ4_NL": 4.0,
    "IQ4_XS": 4.0,
    "IQ3_S": 3.0,
    "IQ3_XXS": 3.0,
    "IQ2_XXS": 2.0,
    "IQ2_XS": 2.0,
    "IQ2_S": 2.0,
    "IQ1_S": 1.5625,
    "IQ1_M": 1.75,
}


def quant_type_bits(qtype: str | None) -> float | None:
    if qtype is None:
        return None
    normalized = str(qtype).upper()
    if normalized in QUANT_TYPE_BITS:
        return QUANT_TYPE_BITS[normalized]
    match = re.match(r"I?Q([0-9]+)", normalized)
    if match:
        return float(match.group(1))
    return None


def normalize_quant_type_name(qtype: str | None) -> str | None:
    if qtype is None:
        return None
    return str(qtype).strip().upper()


def exact_tensor_name_from_pattern(pattern: str) -> str | None:
    pattern = pattern.strip()
    if not pattern.startswith("^") or not pattern.endswith("$"):
        return None
    body = pattern[1:-1]
    try:
        return re.sub(r"\\(.)", r"\1", body)
    except re.error:
        return None


def read_tensor_type_map(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        pattern, qtype = line.split("=", 1)
        tensor = exact_tensor_name_from_pattern(pattern)
        if tensor:
            refs[tensor] = normalize_quant_type_name(qtype) or ""
    return refs


def dynamic_quant_profile(base_tensors: dict[str, str], cand_tensors: dict[str, str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    component_bias: dict[str, dict[str, int]] = {}
    base_bits: list[float] = []
    cand_bits: list[float] = []
    for name in sorted(set(base_tensors) | set(cand_tensors)):
        if not is_quantizable_tensor(name):
            continue
        baseline = base_tensors.get(name)
        candidate = cand_tensors.get(name)
        baseline_bits = quant_type_bits(baseline)
        candidate_bits = quant_type_bits(candidate)
        if baseline_bits is not None:
            base_bits.append(baseline_bits)
        if candidate_bits is not None:
            cand_bits.append(candidate_bits)
        if baseline == candidate:
            continue
        _layer, component = parse_tensor_name(name)
        component_key = component or "other"
        bucket = component_bias.setdefault(component_key, {"promoted": 0, "demoted": 0, "missing_baseline": 0, "missing_candidate": 0, "changed_unknown": 0})
        if baseline is None:
            status = "missing_baseline"
        elif candidate is None:
            status = "missing_candidate"
        elif baseline_bits is None or candidate_bits is None:
            status = "changed_unknown"
        elif candidate_bits > baseline_bits:
            status = "promoted"
        elif candidate_bits < baseline_bits:
            status = "demoted"
        else:
            status = "changed_unknown"
        bucket[status] += 1
        rows.append({"tensor": name, "component": component, "baseline": baseline, "candidate": candidate, "status": status})
    counts = {"promoted": 0, "demoted": 0, "missing_baseline": 0, "missing_candidate": 0, "changed_unknown": 0}
    for row in rows:
        counts[row["status"]] += 1
    return {
        "changed_quantizable_tensors": len(rows),
        **counts,
        "baseline_avg_bits": None if not base_bits else sum(base_bits) / len(base_bits),
        "candidate_avg_bits": None if not cand_bits else sum(cand_bits) / len(cand_bits),
        "avg_bits_delta": None if not base_bits or not cand_bits else (sum(cand_bits) / len(cand_bits)) - (sum(base_bits) / len(base_bits)),
        "component_bias": {key: value for key, value in sorted(component_bias.items())},
    }


def compare_gguf_types(
    baseline: Path,
    candidate: Path,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    reference_map: Path | None = None,
) -> dict[str, Any]:
    base = inspect_gguf_types(baseline)
    cand = inspect_gguf_types(candidate)
    base_tensors = base["tensor_types"]
    cand_tensors = cand["tensor_types"]
    reference_tensors = read_tensor_type_map(reference_map) if reference_map else {}
    tensor_type_changes = []
    for name in sorted(set(base_tensors) | set(cand_tensors)):
        baseline_type = base_tensors.get(name)
        candidate_type = cand_tensors.get(name)
        reference_type = reference_tensors.get(name)
        baseline_bits = quant_type_bits(baseline_type)
        candidate_bits = quant_type_bits(candidate_type)
        if baseline_type is None:
            dynamic_status = "missing_baseline"
        elif candidate_type is None:
            dynamic_status = "missing_candidate"
        elif baseline_bits is None or candidate_bits is None:
            dynamic_status = "changed_unknown"
        elif candidate_bits > baseline_bits:
            dynamic_status = "promoted"
        elif candidate_bits < baseline_bits:
            dynamic_status = "demoted"
        elif baseline_type != candidate_type:
            dynamic_status = "changed_equal_bits"
        else:
            dynamic_status = "same"
        candidate_matches_reference = (
            reference_type is not None
            and normalize_quant_type_name(candidate_type) == normalize_quant_type_name(reference_type)
        )
        if baseline_type == candidate_type and (reference_type is None or candidate_matches_reference):
            continue
        layer, component = parse_tensor_name(name)
        tensor_type_changes.append(
            {
                "tensor": name,
                "layer": layer,
                "component": component,
                "baseline": baseline_type,
                "candidate": candidate_type,
                "reference": reference_type,
                "matches_reference": candidate_matches_reference if reference_type is not None else None,
                "status": "missing_baseline" if baseline_type is None else "missing_candidate" if candidate_type is None else "changed",
                "dynamic_status": dynamic_status,
                "baseline_bits": baseline_bits,
                "candidate_bits": candidate_bits,
                "bits_delta": None if baseline_bits is None or candidate_bits is None else candidate_bits - baseline_bits,
                "quantizable": is_quantizable_tensor(name),
            }
        )
    reference_mismatches = [
        row | {"status": "candidate_diverges_reference"}
        for row in tensor_type_changes
        if row.get("reference") is not None and row.get("matches_reference") is False
    ]
    return {
        "baseline": {"label": baseline_label, "gguf": str(baseline), "summary": base},
        "candidate": {"label": candidate_label, "gguf": str(candidate), "summary": cand},
        "reference_map": None if reference_map is None else {"path": str(reference_map), "tensor_count": len(reference_tensors)},
        "tensor_count_delta": int(cand["tensor_count"]) - int(base["tensor_count"]),
        "quantizable_tensor_count_delta": int(cand["quantizable_tensor_count"]) - int(base["quantizable_tensor_count"]),
        "type_counts": compare_count_maps(base["type_counts"], cand["type_counts"]),
        "component_counts": compare_nested_count_maps(base["component_counts"], cand["component_counts"]),
        "layer_counts": compare_nested_count_maps(base["layer_counts"], cand["layer_counts"]),
        "tensor_type_changes": tensor_type_changes,
        "reference_mismatches": reference_mismatches,
        "dynamic_profile": dynamic_quant_profile(base_tensors, cand_tensors),
    }


def changed_count_rows(counts: dict[str, dict[str, int]]) -> list[list[str]]:
    rows = []
    for key, values in counts.items():
        if values["delta"] == 0:
            continue
        rows.append([key, str(values["baseline"]), str(values["candidate"]), f"{values['delta']:+d}"])
    return rows


def changed_nested_rows(counts: dict[str, dict[str, dict[str, int]]], limit: int = 60) -> list[list[str]]:
    rows = []
    for bucket, values in counts.items():
        for qtype, row in values.items():
            if row["delta"] == 0:
                continue
            rows.append([bucket, qtype, str(row["baseline"]), str(row["candidate"]), f"{row['delta']:+d}"])
    return rows[:limit]


def compare_gguf_types_markdown(report: dict[str, Any]) -> str:
    parts = [
        "# GGUF Tensor Type Comparison",
        "",
        f"baseline: `{report['baseline']['label']}`",
        f"candidate: `{report['candidate']['label']}`",
        "",
        f"tensor_count_delta: {report['tensor_count_delta']:+d}",
        f"quantizable_tensor_count_delta: {report['quantizable_tensor_count_delta']:+d}",
    ]
    type_rows = changed_count_rows(report["type_counts"])
    if type_rows:
        parts.extend(["", "## Type Count Deltas", "", markdown_table(["Type", "Baseline", "Candidate", "Delta"], type_rows)])
    component_rows = changed_nested_rows(report["component_counts"])
    if component_rows:
        parts.extend(["", "## Component Deltas", "", markdown_table(["Component", "Type", "Baseline", "Candidate", "Delta"], component_rows)])
    layer_rows = changed_nested_rows(report["layer_counts"])
    if layer_rows:
        parts.extend(["", "## Layer Deltas", "", markdown_table(["Layer", "Type", "Baseline", "Candidate", "Delta"], layer_rows)])
    profile = report.get("dynamic_profile") or {}
    if profile:
        parts.extend(
            [
                "",
                "## Dynamic Quant Profile",
                "",
                markdown_table(
                    ["Metric", "Value"],
                    [
                        ["changed quantizable tensors", str(profile["changed_quantizable_tensors"])],
                        ["promoted", str(profile["promoted"])],
                        ["demoted", str(profile["demoted"])],
                        ["missing baseline", str(profile["missing_baseline"])],
                        ["missing candidate", str(profile["missing_candidate"])],
                        ["baseline avg bits", "-" if profile["baseline_avg_bits"] is None else f"{profile['baseline_avg_bits']:.2f}"],
                        ["candidate avg bits", "-" if profile["candidate_avg_bits"] is None else f"{profile['candidate_avg_bits']:.2f}"],
                        ["avg bits delta", "-" if profile["avg_bits_delta"] is None else f"{profile['avg_bits_delta']:+.2f}"],
                    ],
                ),
            ]
        )
        component_bias_rows = [
            [component, str(values["promoted"]), str(values["demoted"]), str(values["missing_baseline"]), str(values["missing_candidate"])]
            for component, values in profile.get("component_bias", {}).items()
        ]
        if component_bias_rows:
            parts.extend(["", "## Dynamic Component Bias", "", markdown_table(["Component", "Promoted", "Demoted", "Missing baseline", "Missing candidate"], component_bias_rows)])
    ref_rows = [
        [
            row["tensor"],
            str(row.get("component") or "-"),
            "-" if row.get("baseline") is None else str(row["baseline"]),
            "-" if row.get("candidate") is None else str(row["candidate"]),
            "-" if row.get("reference") is None else str(row["reference"]),
            str(row["status"]),
        ]
        for row in report.get("reference_mismatches", [])[:80]
    ]
    if ref_rows:
        ref = report.get("reference_map") or {}
        parts.extend(
            [
                "",
                "## Reference Map Mismatches",
                "",
                f"reference: `{ref.get('path', '-')}` ({ref.get('tensor_count', 0)} exact tensors)",
                "",
                markdown_table(["Tensor", "Component", "Baseline", "Candidate", "Reference", "Status"], ref_rows),
            ]
        )
    tensor_rows = [
        [
            row["tensor"],
            "-" if row.get("baseline") is None else str(row["baseline"]),
            "-" if row.get("candidate") is None else str(row["candidate"]),
            str(row.get("dynamic_status") or row["status"]),
            "-" if row.get("bits_delta") is None else f"{float(row['bits_delta']):+.2f}",
        ]
        for row in report.get("tensor_type_changes", [])[:80]
    ]
    if tensor_rows:
        parts.extend(["", "## Tensor Type Changes", "", markdown_table(["Tensor", "Baseline", "Candidate", "Dynamic status", "Bits delta"], tensor_rows)])
    if not type_rows and not component_rows and not layer_rows and not tensor_rows:
        parts.append("\nNo tensor type distribution differences detected.")
    return "\n".join(parts) + "\n"


def compare_gguf_types_cmd(args: argparse.Namespace) -> None:
    report = compare_gguf_types(
        Path(args.baseline),
        Path(args.candidate),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        reference_map=Path(args.reference_map) if args.reference_map else None,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(compare_gguf_types_markdown(report), end="")


def compare_gguf_types_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    baseline = query_value(qs, "baseline")
    candidate = query_value(qs, "candidate")
    if not baseline:
        raise ValueError("baseline query param required")
    if not candidate:
        raise ValueError("candidate query param required")
    return argparse.Namespace(
        baseline=baseline,
        candidate=candidate,
        baseline_label=query_value(qs, "baseline_label", "baseline"),
        candidate_label=query_value(qs, "candidate_label", "candidate"),
        reference_map=query_value(qs, "reference_map"),
        json=True,
    )


def state_path_for_compare(path: Path) -> Path:
    if path.is_dir():
        return path / "state.json"
    return path


def quantizable_locks_from_state(state: dict[str, Any]) -> dict[str, str]:
    locks = state.get("locked", {})
    if not isinstance(locks, dict):
        return {}
    return {
        str(tensor): str(qtype)
        for tensor, qtype in locks.items()
        if qtype is not None and is_quantizable_tensor(str(tensor))
    }


def compare_locks(current_state: dict[str, Any], against_state: dict[str, Any]) -> dict[str, Any]:
    current = quantizable_locks_from_state(current_state)
    against = quantizable_locks_from_state(against_state)
    rows: list[dict[str, Any]] = []
    same = different = missing_current = missing_against = 0
    for tensor in sorted(set(current) | set(against)):
        current_q = current.get(tensor)
        against_q = against.get(tensor)
        if current_q == against_q:
            status = "same"
            same += 1
        elif current_q is None:
            status = "missing_current"
            missing_current += 1
        elif against_q is None:
            status = "missing_against"
            missing_against += 1
        else:
            status = "different"
            different += 1
        rows.append({"tensor": tensor, "current": current_q, "against": against_q, "status": status})
    return {
        "current_locked": len(current),
        "against_locked": len(against),
        "same": same,
        "different": different,
        "missing_current": missing_current,
        "missing_against": missing_against,
        "rows": rows,
    }


def compare_locks_cmd(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir)
    against_path = state_path_for_compare(Path(args.against))
    current_state = read_json(run_dir / "state.json", {})
    against_state = read_json(against_path, {})
    summary = compare_locks(current_state, against_state)
    summary["run_dir"] = str(run_dir)
    summary["against"] = str(against_path)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"Cerebellum lock comparison: {run_dir}")
    print(f"against: {against_path}")
    print(
        f"current={summary['current_locked']} against={summary['against_locked']} "
        f"same={summary['same']} different={summary['different']} "
        f"missing_current={summary['missing_current']} missing_against={summary['missing_against']}"
    )
    for row in summary["rows"]:
        if row["status"] == "same":
            continue
        current = row["current"] or "-"
        against = row["against"] or "-"
        print(f"{row['status']:<16} {row['tensor']}  current={current}  against={against}")


def infer_benchmark_name(path: Path, data: dict[str, Any]) -> str:
    if data.get("benchmark"):
        return str(data["benchmark"])
    stem = path.stem.lower()
    for name in [
        "livecodebench_v6",
        "livecodebench",
        "hle_no_tools",
        "gpqa_diamond",
        "mmlu_pro",
        "mmmlu",
        "humaneval",
        "evalplus",
        "arc",
        "hellaswag",
        "mmlu_redux",
        "mmlu",
        "speed",
        "ppl",
    ]:
        if name in stem:
            return name
    return stem


def benchmark_key(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "arc_challenge": "arc",
        "mmlu_pro": "mmlu_pro",
        "mmlupro": "mmlu_pro",
        "gpqa": "gpqa_diamond",
        "gpqa_diamond": "gpqa_diamond",
        "hle": "hle_no_tools",
        "hle_notools": "hle_no_tools",
        "livecodebench": "livecodebench_v6",
        "lcb_v6": "livecodebench_v6",
        "humaneval_plus": "evalplus",
        "humaneval+": "evalplus",
        "evalplus_humaneval_plus_chat": "evalplus",
        "perplexity": "ppl",
    }
    return aliases.get(key, key)


def benchmark_metric(data: dict[str, Any]) -> tuple[str, float] | None:
    if data.get("pass_at_1_plus") is not None:
        return "pass@1 plus", float(data["pass_at_1_plus"])
    if data.get("pass_at_1_base") is not None:
        return "pass@1 base", float(data["pass_at_1_base"])
    if "pass_at_1_pct" in data:
        return "pass@1", float(data["pass_at_1_pct"])
    if "pass_at_1" in data:
        value = float(data["pass_at_1"])
        return "pass@1", value * 100.0 if value <= 1.0 else value
    if "accuracy" in data:
        value = float(data["accuracy"])
        return "accuracy", value * 100.0 if value <= 1.0 else value
    if "gen_tok_per_s" in data:
        return "gen tok/s", float(data["gen_tok_per_s"])
    if "tokens_per_second" in data:
        return "tok/s", float(data["tokens_per_second"])
    if "score" in data:
        value = float(data["score"])
        return "score", value * 100.0 if value <= 1.0 else value
    if "exact_match" in data:
        value = float(data["exact_match"])
        return "exact match", value * 100.0 if value <= 1.0 else value
    if "ppl" in data:
        return "ppl", float(data["ppl"])
    if "perplexity" in data:
        return "ppl", float(data["perplexity"])
    return None


def benchmark_size_gib(data: dict[str, Any]) -> float | None:
    for key in ["size_gib", "size_gb", "gguf_size_gib", "gguf_size_gb", "model_size_gib", "model_size_gb"]:
        if key in data and data[key] is not None:
            return float(data[key])
    for key in ["size_bytes", "gguf_size_bytes", "model_size_bytes"]:
        if key in data and data[key] is not None:
            return float(data[key]) / (1024**3)
    return None


def benchmark_release_metadata(data: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "bpw": ["bpw", "bits_per_weight"],
        "quant_recipe": ["quant_recipe", "recipe", "quant", "quantization", "quant_type"],
        "tensor_map": ["tensor_map", "tensor_types", "tensor_type_file"],
        "gguf_sha256": ["gguf_sha256", "model_sha256", "sha256"],
        "llama_cpp": ["llama_cpp", "llama_cpp_commit", "llama_cpp_version"],
        "runtime": ["runtime", "server", "backend"],
    }
    metadata: dict[str, Any] = {}
    size = benchmark_size_gib(data)
    if size is not None:
        metadata["size_gib"] = size
    for out_key, keys in aliases.items():
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                metadata[out_key] = value
                break
    return metadata


def parse_size_specs(specs: list[str]) -> dict[str, float]:
    sizes: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--size must be MODEL=GiB, got {spec!r}")
        model, value = spec.split("=", 1)
        sizes[model] = float(value)
    return sizes


def parse_weight_specs(specs: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--weight must be BENCHMARK=WEIGHT, got {spec!r}")
        benchmark, value = spec.split("=", 1)
        key = benchmark_key(benchmark.strip())
        if not key:
            raise SystemExit(f"--weight benchmark cannot be empty, got {spec!r}")
        weight = float(value)
        if weight < 0:
            raise SystemExit(f"--weight must be non-negative, got {spec!r}")
        weights[key] = weight
    return weights


def read_size_json(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--size-json must contain an object")
    if "models" in data and isinstance(data["models"], dict):
        data = data["models"]
    sizes: dict[str, float] = {}
    for model, value in data.items():
        if isinstance(value, dict):
            size = benchmark_size_gib(value)
            if size is not None:
                sizes[str(model)] = size
        elif value is not None:
            sizes[str(model)] = float(value)
    return sizes


def benchmark_command(entry: dict[str, Any], model: str, port: int, results_dir: str) -> str | None:
    script = entry.get("script")
    if not script:
        return None
    env = {
        "BENCH_MODEL": model,
        "BENCH_PORT": str(port),
        "BENCH_WORKERS": str(entry.get("workers", 1)),
        "RESULTS_DIR": results_dir,
    }
    if entry.get("max_tokens"):
        env["BENCH_MAX_TOKENS"] = str(entry["max_tokens"])
    for key, value in (entry.get("env") or {}).items():
        env[str(key)] = str(value).format(model=model, port=port, results_dir=results_dir)
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    args = " ".join(shlex.quote(str(arg).format(model=model, port=port, results_dir=results_dir)) for arg in entry.get("args", []))
    return f"{prefix} python {shlex.quote(str(script))}{(' ' + args) if args else ''}"


def format_artifact_template(template: str, model: str, results_dir: str) -> str:
    return template.format(model=model, results_dir=results_dir)


def benchmark_plan(suite: str, model: str, port: int, results_dir: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in BENCHMARK_SUITES[suite]:
        entry = BENCHMARK_CATALOG.get(key, {"name": key, "status": "pending"})
        artifacts = [
            str(Path(results_dir) / format_artifact_template(template, model, results_dir))
            for template in entry.get("artifacts", [])
        ]
        audit = entry.get("audit")
        if audit:
            audit = str(audit).format(model=model, results_dir=results_dir)
        rows.append(
            {
                "benchmark": key,
                "name": entry.get("name", key),
                "status": entry.get("status", "pending"),
                "workers": entry.get("workers"),
                "command": benchmark_command(entry, model, port, results_dir),
                "artifacts": artifacts,
                "audit": audit,
                "note": entry.get("note"),
            }
        )
    blockers = [
        {
            "benchmark": row["benchmark"],
            "status": row["status"],
            "reason": row.get("note") or "runner not implemented",
        }
        for row in rows
        if row["status"] != "implemented"
    ]
    return {
        "suite": suite,
        "purpose": BENCHMARK_SUITE_PURPOSES.get(suite, ""),
        "model": model,
        "port": port,
        "results_dir": results_dir,
        "rows": rows,
        "readiness": {
            "ready": not blockers,
            "implemented": sum(1 for row in rows if row["status"] == "implemented"),
            "total": len(rows),
            "blockers": blockers,
        },
    }


def benchmark_plan_markdown(plan: dict[str, Any]) -> str:
    table_rows = []
    for row in plan["rows"]:
        command = row["command"] or row.get("note") or "-"
        table_rows.append([row["benchmark"], row["status"], "-" if row["workers"] is None else str(row["workers"]), command])
    parts = [
        f"# Benchmark Plan ({plan['suite']})",
        "",
        f"purpose: `{plan.get('purpose') or '-'}`",
        f"readiness: `{'ready' if plan.get('readiness', {}).get('ready') else 'blocked'}` "
        f"({plan.get('readiness', {}).get('implemented', 0)}/{plan.get('readiness', {}).get('total', 0)} implemented)",
        "",
        markdown_table(["Benchmark", "Status", "Workers", "Command / note"], table_rows),
    ]
    blockers = plan.get("readiness", {}).get("blockers") or []
    if blockers:
        parts.extend(
            [
                "",
                "## Readiness Blockers",
                "",
                markdown_table(["Benchmark", "Status", "Reason"], [[row["benchmark"], row["status"], row["reason"]] for row in blockers]),
            ]
        )
    artifact_rows = []
    for row in plan["rows"]:
        for artifact in row["artifacts"]:
            artifact_rows.append([row["benchmark"], artifact])
    if artifact_rows:
        parts.extend(["", "## Artifacts", "", markdown_table(["Benchmark", "Artifact"], artifact_rows)])
    audit_rows = [[row["benchmark"], row["audit"]] for row in plan["rows"] if row.get("audit")]
    if audit_rows:
        parts.extend(["", "## Audit", "", markdown_table(["Benchmark", "Command"], audit_rows)])
    return "\n".join(parts) + "\n"


def benchmark_plan_cmd(args: argparse.Namespace) -> None:
    plan = benchmark_plan(args.suite, args.model, args.port, args.results_dir)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(benchmark_plan_markdown(plan), end="")
    if args.require_ready and not plan["readiness"]["ready"]:
        raise SystemExit(1)


def benchmark_run_plan(
    suite: str,
    model: str,
    port: int,
    results_dir: str,
    benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    plan = benchmark_plan(suite, model, port, results_dir)
    selected = list(plan["rows"])
    if benchmarks:
        wanted = set(benchmarks)
        known = {row["benchmark"] for row in selected}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f"unknown benchmark(s) for suite {suite}: {', '.join(unknown)}")
        selected = [row for row in selected if row["benchmark"] in wanted]
    blockers = [
        {
            "benchmark": row["benchmark"],
            "status": row["status"],
            "reason": row.get("note") or "runner not implemented",
        }
        for row in selected
        if row["status"] != "implemented" or not row.get("command")
    ]
    return {
        "schema": "cerebellum.benchmark_run.v1",
        "suite": suite,
        "purpose": plan.get("purpose", ""),
        "model": model,
        "port": port,
        "results_dir": results_dir,
        "dry_run": True,
        "benchmarks": selected,
        "blocked": bool(blockers),
        "blockers": blockers,
    }


def benchmark_run_markdown(plan: dict[str, Any]) -> str:
    rows = [
        [row["benchmark"], row["status"], "-" if row.get("workers") is None else str(row["workers"]), row.get("command") or row.get("note") or "-"]
        for row in plan["benchmarks"]
    ]
    parts = [
        f"# Benchmark Run ({plan['suite']})",
        "",
        f"mode: `{'dry-run' if plan.get('dry_run') else 'execute'}`",
        f"model: `{plan['model']}`",
        f"results_dir: `{plan['results_dir']}`",
        "",
        markdown_table(["Benchmark", "Status", "Workers", "Command / note"], rows),
    ]
    if plan.get("blockers"):
        parts.extend(["", "## Blockers", "", markdown_table(["Benchmark", "Status", "Reason"], [[row["benchmark"], row["status"], row["reason"]] for row in plan["blockers"]])])
    if plan.get("executions"):
        parts.extend(
            [
                "",
                "## Executions",
                "",
                markdown_table(["Benchmark", "Return", "Log"], [[row["benchmark"], str(row["returncode"]), row["log"]] for row in plan["executions"]]),
            ]
        )
    if plan.get("postprocess"):
        post = plan["postprocess"]
        parts.extend(
            [
                "",
                "## Postprocess",
                "",
                markdown_table(
                    ["Artifact", "Path"],
                    [
                        ["manifest", post["manifest"]],
                        ["audit", post["audit"]],
                        ["report", post["report"]],
                    ],
                ),
            ]
        )
    return "\n".join(parts) + "\n"


def append_benchmark_run_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": utc_now(), **event}, sort_keys=True) + "\n")


def split_env_prefixed_command(command: str) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    argv = shlex.split(command)
    while argv and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
        key, value = argv.pop(0).split("=", 1)
        env[key] = value
    if not argv:
        raise ValueError("command has no executable")
    return env, argv


def benchmark_run_execute(plan: dict[str, Any]) -> dict[str, Any]:
    if plan["blocked"]:
        raise SystemExit("benchmark-run has blockers; use an implemented benchmark selection before --execute")
    results_dir = Path(str(plan["results_dir"]))
    log_dir = results_dir / "benchmark_run_logs"
    event_log = results_dir / "benchmark_run_events.jsonl"
    executions: list[dict[str, Any]] = []
    for row in plan["benchmarks"]:
        benchmark = str(row["benchmark"])
        command = str(row["command"])
        log_path = log_dir / f"{slug(benchmark)}.log"
        append_benchmark_run_event(event_log, {"event": "benchmark_start", "benchmark": benchmark, "command": command})
        started = time.monotonic()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log:
            env_prefix, argv = split_env_prefixed_command(command)
            proc = subprocess.run(argv, cwd=Path.cwd(), env={**os.environ, **env_prefix}, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        elapsed = time.monotonic() - started
        execution = {
            "benchmark": benchmark,
            "command": command,
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "log": str(log_path),
        }
        executions.append(execution)
        append_benchmark_run_event(event_log, {"event": "benchmark_finish", **execution})
        if proc.returncode != 0:
            plan.update(
                {
                    "dry_run": False,
                    "blocked": True,
                    "blockers": [{"benchmark": benchmark, "status": row.get("status"), "reason": f"command exited {proc.returncode}"}],
                    "executions": executions,
                    "event_log": str(event_log),
                }
            )
            return plan
    plan.update({"dry_run": False, "executions": executions, "event_log": str(event_log)})
    return plan


def benchmark_run_postprocess(
    plan: dict[str, Any],
    *,
    require_complete: bool = False,
    leaderboard: bool = False,
    sizes: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    results_dir = Path(str(plan["results_dir"]))
    results_dir.mkdir(parents=True, exist_ok=True)
    post_dir = results_dir / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)
    manifest = benchmark_manifest([results_dir], suite=str(plan["suite"]), model=str(plan["model"]))
    audit = benchmark_audit([str(results_dir)])
    report = benchmark_report([results_dir], suite=str(plan["suite"]), leaderboard=leaderboard, sizes=sizes, weights=weights)
    manifest_path = post_dir / "benchmark_manifest.json"
    audit_path = post_dir / "benchmark_audit.json"
    report_path = post_dir / "benchmark_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blockers: list[dict[str, Any]] = []
    if require_complete and manifest["missing_measured"]:
        blockers.append(
            {
                "benchmark": ",".join(manifest["missing_measured"]),
                "status": "missing",
                "reason": "missing measured benchmark artifacts after benchmark-run postprocess",
            }
        )
    if audit["blocked"]:
        blockers.extend(
            {
                "benchmark": Path(str(item["path"])).stem,
                "status": "audit_failed",
                "reason": str(item["reason"]),
            }
            for item in audit["failures"]
        )
    return {
        "schema": "cerebellum.benchmark_postprocess.v1",
        "results_dir": str(results_dir),
        "suite": plan["suite"],
        "model": plan["model"],
        "manifest": str(manifest_path),
        "audit": str(audit_path),
        "report": str(report_path),
        "missing_measured": manifest["missing_measured"],
        "audit_failures": audit["failures"],
        "leaderboard_rows": len(report.get("leaderboard", [])),
        "blocked": bool(blockers),
        "blockers": blockers,
    }


def benchmark_postprocess_markdown(payload: dict[str, Any]) -> str:
    rows = [
        ["manifest", payload["manifest"]],
        ["audit", payload["audit"]],
        ["report", payload["report"]],
    ]
    parts = [
        "# Benchmark Postprocess",
        "",
        f"status: `{'blocked' if payload['blocked'] else 'ready'}`",
        f"results_dir: `{payload['results_dir']}`",
        f"leaderboard rows: `{payload['leaderboard_rows']}`",
        "",
        markdown_table(["Artifact", "Path"], rows),
    ]
    if payload.get("blockers"):
        parts.extend(["", "## Blockers", "", markdown_table(["Status", "Reason"], [[row["status"], row["reason"]] for row in payload["blockers"]])])
    return "\n".join(parts) + "\n"


def benchmark_postprocess_cmd(args: argparse.Namespace) -> None:
    plan = {"suite": args.suite, "model": args.model, "results_dir": args.results_dir}
    payload = benchmark_run_postprocess(
        plan,
        require_complete=args.require_complete,
        leaderboard=args.leaderboard,
        sizes={**read_size_json(args.size_json), **parse_size_specs(args.size)},
        weights=parse_weight_specs(args.weight),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(benchmark_postprocess_markdown(payload), end="")
    if payload["blocked"]:
        raise SystemExit(1)


def benchmark_ingest(
    db: Path,
    results_dir: Path,
    *,
    suite: str = "release",
    model: str = "cerebellum",
    require_complete: bool = False,
    leaderboard: bool = False,
    sizes: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    manifest = benchmark_manifest([results_dir], suite=suite, model=model)
    audit = benchmark_audit([str(results_dir)])
    report = benchmark_report([results_dir], suite=suite, leaderboard=leaderboard, sizes=sizes, weights=weights)
    blockers: list[dict[str, Any]] = []
    if require_complete and manifest["missing_measured"]:
        blockers.append(
            {
                "status": "missing",
                "benchmark": ",".join(manifest["missing_measured"]),
                "reason": "missing measured benchmark artifacts",
            }
        )
    if audit["blocked"]:
        blockers.extend(
            {
                "status": "audit_failed",
                "benchmark": Path(str(item["path"])).stem,
                "reason": str(item["reason"]),
            }
            for item in audit["failures"]
        )
    ready = not blockers
    now = utc_now()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        ensure_hill_tables(conn)
        with conn:
            cur = conn.execute(
                """
                INSERT INTO cerebellum_benchmark_ingests
                  (model, suite, results_dir, ready, blockers_json, manifest_json,
                   audit_json, report_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model, suite, results_dir) DO UPDATE SET
                    ready = excluded.ready,
                    blockers_json = excluded.blockers_json,
                    manifest_json = excluded.manifest_json,
                    audit_json = excluded.audit_json,
                    report_json = excluded.report_json,
                    updated_at = excluded.updated_at
                RETURNING id
                """,
                (
                    model,
                    suite,
                    str(results_dir),
                    1 if ready else 0,
                    json.dumps(blockers, sort_keys=True),
                    json.dumps(manifest, sort_keys=True),
                    json.dumps(audit, sort_keys=True),
                    json.dumps(report, sort_keys=True),
                    now,
                    now,
                ),
            )
            ingest_id = int(cur.fetchone()[0])
            conn.execute("DELETE FROM cerebellum_benchmark_results WHERE ingest_id = ?", (ingest_id,))
            for row in report.get("records", []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cerebellum_benchmark_results
                      (ingest_id, model, suite, benchmark_key, benchmark, metric,
                       value, path, size_gib, release_metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ingest_id,
                        str(row["model"]),
                        suite,
                        str(row["benchmark_key"]),
                        str(row["benchmark"]),
                        str(row["metric"]),
                        float(row["value"]),
                        str(row["path"]),
                        row.get("size_gib"),
                        json.dumps(row.get("release_metadata", {}), sort_keys=True),
                    ),
                )
    finally:
        conn.close()
    return {
        "schema": "cerebellum.benchmark_ingest.v1",
        "db": str(db),
        "ingest_id": ingest_id,
        "model": model,
        "suite": suite,
        "results_dir": str(results_dir),
        "ready": ready,
        "blockers": blockers,
        "missing_measured": manifest["missing_measured"],
        "audit_failures": audit["failures"],
        "records": len(report.get("records", [])),
        "leaderboard_rows": len(report.get("leaderboard", [])),
        "measured_benchmarks": manifest["measured_benchmarks"],
    }


def benchmark_ingest_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Ingest",
        "",
        f"db: `{payload['db']}`",
        f"ingest: `{payload['ingest_id']}`",
        f"model: `{payload['model']}`",
        f"suite: `{payload['suite']}`",
        f"ready: `{'yes' if payload['ready'] else 'no'}`",
        f"records: `{payload['records']}`",
        f"leaderboard rows: `{payload['leaderboard_rows']}`",
    ]
    if payload.get("missing_measured"):
        lines.append(f"missing: `{', '.join(payload['missing_measured'])}`")
    if payload.get("blockers"):
        rows = [[row["status"], row["benchmark"], row["reason"]] for row in payload["blockers"]]
        lines.extend(["", "## Blockers", "", markdown_table(["Status", "Benchmark", "Reason"], rows)])
    return "\n".join(lines) + "\n"


def benchmark_ingest_cmd(args: argparse.Namespace) -> None:
    sizes = read_size_json(args.size_json)
    sizes.update(parse_size_specs(args.size))
    weights = parse_weight_specs(args.weight)
    payload = benchmark_ingest(
        Path(args.db),
        Path(args.results_dir),
        suite=args.suite,
        model=args.model,
        require_complete=args.require_complete,
        leaderboard=args.leaderboard,
        sizes=sizes,
        weights=weights,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(benchmark_ingest_markdown(payload), end="")
    if not payload["ready"]:
        raise SystemExit(1)


def benchmark_run_cmd(args: argparse.Namespace) -> None:
    plan = benchmark_run_plan(args.suite, args.model, args.port, args.results_dir, args.benchmark)
    if args.execute:
        plan = benchmark_run_execute(plan)
        if args.postprocess and not plan["blocked"]:
            sizes = read_size_json(args.size_json)
            sizes.update(parse_size_specs(args.size))
            weights = parse_weight_specs(args.weight)
            postprocess = benchmark_run_postprocess(
                plan,
                require_complete=args.require_complete,
                leaderboard=args.leaderboard,
                sizes=sizes,
                weights=weights,
            )
            plan["postprocess"] = postprocess
            if postprocess["blocked"]:
                plan["blocked"] = True
                plan["blockers"] = postprocess["blockers"]
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(benchmark_run_markdown(plan), end="")
    if plan["blocked"]:
        raise SystemExit(1)


def benchmark_status(results_dir: Path, events_path: Path | None = None) -> dict[str, Any]:
    event_log = events_path or results_dir / "benchmark_run_events.jsonl"
    events = read_jsonl(event_log)
    rows: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for event in events:
        benchmark = str(event.get("benchmark") or "")
        if not benchmark:
            continue
        row = by_name.get(benchmark)
        if row is None:
            row = {
                "benchmark": benchmark,
                "status": "pending",
                "command": None,
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "elapsed_seconds": None,
                "log": None,
            }
            rows.append(row)
            by_name[benchmark] = row
        kind = event.get("event")
        if kind == "benchmark_start":
            row["status"] = "running"
            row["started_at"] = event.get("time")
            row["command"] = event.get("command") or row.get("command")
        elif kind == "benchmark_finish":
            returncode = int(event.get("returncode", 0))
            row["status"] = "complete" if returncode == 0 else "failed"
            row["finished_at"] = event.get("time")
            row["returncode"] = returncode
            row["elapsed_seconds"] = event.get("elapsed_seconds")
            row["log"] = event.get("log")
            row["command"] = event.get("command") or row.get("command")
    failed = next((row for row in rows if row["status"] == "failed"), None)
    running = next((row for row in rows if row["status"] == "running"), None)
    if failed:
        status = "failed"
        rerun_benchmark = failed["benchmark"]
    elif running:
        status = "running"
        rerun_benchmark = running["benchmark"]
    elif not events:
        status = "not-started"
        rerun_benchmark = None
    else:
        status = "complete"
        rerun_benchmark = None
    rerun_command = None
    if rerun_benchmark:
        source_row = failed or running
        rerun_command = source_row.get("command") if source_row else None
    return {
        "schema": "cerebellum.benchmark_status.v1",
        "results_dir": str(results_dir),
        "event_log": str(event_log),
        "event_count": len(events),
        "status": status,
        "completed_benchmarks": sum(1 for row in rows if row["status"] == "complete"),
        "failed_benchmark": failed["benchmark"] if failed else None,
        "running_benchmark": running["benchmark"] if running else None,
        "rerun_benchmark": rerun_benchmark,
        "rerun_command": rerun_command,
        "last_event": events[-1] if events else None,
        "benchmarks": rows,
    }


def benchmark_status_markdown(status: dict[str, Any]) -> str:
    rows = [
        [
            str(row["benchmark"]),
            str(row["status"]),
            "-" if row.get("returncode") is None else str(row["returncode"]),
            str(row.get("log") or "-"),
        ]
        for row in status["benchmarks"]
    ]
    parts = [
        "# Benchmark Status",
        "",
        f"results_dir: `{status['results_dir']}`",
        f"status: `{status['status']}`",
        f"events: `{status['event_count']}`",
        f"completed: `{status['completed_benchmarks']}/{len(status['benchmarks'])}`",
    ]
    if status.get("rerun_command"):
        parts.append(f"rerun: `{status['rerun_command']}`")
    parts.extend(["", markdown_table(["Benchmark", "Status", "Return", "Log"], rows)])
    return "\n".join(parts) + "\n"


def benchmark_status_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    return argparse.Namespace(
        results_dir=query_value(qs, "results_dir", "benchmark_results"),
        events=query_value(qs, "events"),
    )


def benchmark_status_cmd(args: argparse.Namespace) -> None:
    result = benchmark_status(Path(args.results_dir), Path(args.events) if args.events else None)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(benchmark_status_markdown(result), end="")


def benchmark_plan_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    suite = query_value(qs, "suite", "full")
    if suite not in BENCHMARK_SUITES:
        raise ValueError(f"unknown benchmark suite {suite}")
    try:
        port = int(query_value(qs, "port", 8084))
    except ValueError as exc:
        raise ValueError("port must be an integer") from exc
    return argparse.Namespace(
        suite=suite,
        model=query_value(qs, "model", "cerebellum"),
        port=port,
        results_dir=query_value(qs, "results_dir", "benchmark_results"),
    )


def rebench_model_label(repo: str) -> str:
    return slug(repo.split("/", 1)[-1]).lower()


def benchmark_rebench_plan(
    suite: str = "humaneval",
    results_root: str = "benchmark_results/rebench_20260605",
    port: int = 8084,
    models: list[str] | None = None,
    correction_issue: str = "#35",
) -> dict[str, Any]:
    repos = models or [row["repo"] for row in HUMANEVAL_REBENCH_MODELS]
    published = {row["repo"]: row.get("published") for row in HUMANEVAL_REBENCH_MODELS}
    benchmarks = ["humaneval"] if suite == "humaneval" else ["arc", "hellaswag", "mmlu_redux", "humaneval"]
    jobs = []
    for repo in repos:
        model = rebench_model_label(repo)
        results_dir = str(Path(results_root) / model)
        rows = []
        for key in benchmarks:
            entry = BENCHMARK_CATALOG[key]
            artifacts = [str(Path(results_dir) / format_artifact_template(template, model, results_dir)) for template in entry.get("artifacts", [])]
            audit = entry.get("audit")
            rows.append(
                {
                    "benchmark": key,
                    "command": benchmark_command(entry, model, port, results_dir),
                    "artifacts": artifacts,
                    "audit": str(audit).format(model=model, results_dir=results_dir) if audit else None,
                }
            )
        jobs.append(
            {
                "repo": repo,
                "model": model,
                "published": published.get(repo),
                "results_dir": results_dir,
                "server_note": "serve this model with llama-server before running commands; use fixed chat/EvalPlus pipeline and thinking-disabled release settings where supported",
                "benchmarks": rows,
                "post_run": {
                    "audit": shell_join(["cerebellum", "benchmark-audit", results_dir]),
                    "manifest": shell_join(["cerebellum", "benchmark-manifest", results_dir, "--suite", "release", "--model", model, "--require-complete"]),
                    "model_card_note": f"scores corrected on 2026-06-05 after HumanEval+/benchmark parser fixes; see {correction_issue}",
                },
            }
        )
    return {
        "schema": "cerebellum.benchmark_rebench_plan.v1",
        "suite": suite,
        "reason": "published model cards before the fixed HumanEval+/benchmark parser pipeline may carry false-low scores",
        "correction_issue": correction_issue,
        "model_count": len(jobs),
        "results_root": results_root,
        "port": port,
        "jobs": jobs,
        "notes": [
            "This is a plan only; it does not download models, start llama-server, or run benchmarks.",
            "Run HumanEval+/EvalPlus sequentially with one worker.",
            "Audit detailed artifacts before updating any model card.",
            "Preserve old scores in commit history and add the correction note to each card.",
        ],
    }


def benchmark_rebench_plan_markdown(plan: dict[str, Any]) -> str:
    rows = [
        [
            str(index),
            job["repo"],
            job["model"],
            job.get("published") or "-",
            job["results_dir"],
        ]
        for index, job in enumerate(plan["jobs"], 1)
    ]
    command_rows = []
    for job in plan["jobs"]:
        for row in job["benchmarks"]:
            command_rows.append([job["model"], row["benchmark"], row["command"] or "-"])
        command_rows.append([job["model"], "audit", job["post_run"]["audit"]])
        command_rows.append([job["model"], "manifest", job["post_run"]["manifest"]])
    parts = [
        "# Benchmark Rebench Plan",
        "",
        f"suite: `{plan['suite']}`",
        f"models: `{plan['model_count']}`",
        f"reason: {plan['reason']}",
        "",
        markdown_table(["#", "Repo", "Model label", "Published", "Results"], rows),
        "",
        "## Commands",
        "",
        markdown_table(["Model", "Step", "Command"], command_rows),
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in plan["notes"]],
    ]
    return "\n".join(parts) + "\n"


def benchmark_rebench_plan_cmd(args: argparse.Namespace) -> None:
    plan = benchmark_rebench_plan(args.suite, args.results_root, args.port, args.model, args.correction_issue)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(benchmark_rebench_plan_markdown(plan), end="")


def benchmark_rebench_plan_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    suite = query_value(qs, "suite", "humaneval")
    if suite not in {"humaneval", "release"}:
        raise ValueError("suite must be humaneval or release")
    try:
        port = int(query_value(qs, "port", 8084))
    except ValueError as exc:
        raise ValueError("port must be an integer") from exc
    return argparse.Namespace(
        suite=suite,
        results_root=query_value(qs, "results_root", "benchmark_results/rebench_20260605"),
        port=port,
        model=qs.get("model") or None,
        correction_issue=query_value(qs, "correction_issue", "#35"),
    )


def shell_join(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if part is not None and str(part) != "")


def optional_flag(parts: list[str], flag: str, value: Any) -> None:
    if value is not None:
        parts.extend([flag, str(value)])


def bool_flag(parts: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        parts.append(flag)


def legacy_gated_workflow_detail(
    source: Path,
    output_dir: Path,
    run_dir: Path,
    imatrix: Path,
    final_gguf: Path,
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tensor_map = run_dir / "artifacts" / "final_types.txt"
    baseline_quant = output_dir / f"{slug(args.model_name or source.stem).lower()}-baseline.gguf"
    quant_base = [args.quantize_bin, "--allow-requantize"]
    bool_flag(quant_base, "--pure", getattr(args, "pure_quant", False))
    tensor_map_quant_base = [args.quantize_bin, "--allow-requantize"]
    steps = [
        {
            "name": "scan",
            "purpose": "build imatrix and tensor inventory, then classify architecture groups before ablation",
            "command": shell_join(["cerebellum", "imatrix", "--model", source, "--output", imatrix]),
            "outputs": [str(imatrix)],
        },
        {
            "name": "lower-quant-baseline",
            "purpose": "build a lower quant from the F16 source; serve and benchmark this quant output, not the F16 source",
            "command": shell_join([*quant_base, "--imatrix", imatrix, str(source), baseline_quant, args.base_type]),
            "outputs": [str(baseline_quant)],
        },
        {
            "name": "survivability-scan",
            "purpose": "test whole blocks/groups first and record PPL/benchmark deltas before per-tensor work",
            "command": shell_join(["cerebellum", "legacy-flow", "--source-gguf", source, "--output-dir", output_dir]),
            "outputs": [str(output_dir / "legacy_flow.json")],
        },
        {
            "name": "reverse-ablation",
            "purpose": "from the first compressed candidate, restore groups one at a time to find Q2 regularization vs real damage",
            "command": "planned manual/queued group runs; do not infer from a single tensor",
            "outputs": [str(output_dir / "reverse_ablation_results.json")],
        },
        {
            "name": "targeted-hillstep",
            "purpose": "use hillstep only inside approved groups/layers after group evidence and protected floors",
            "command": shell_join(["cerebellum", "run", "--source-gguf", source, "--run-dir", run_dir, "--imatrix", imatrix, "--profile", args.profile, "--metric", args.metric or "ppl"]),
            "outputs": [str(run_dir / "state.json"), str(tensor_map)],
        },
        {
            "name": "allocate",
            "purpose": "rank promotions/demotions by measured quality gain or cost per GiB and reject bad bundles",
            "command": "planned allocator: use measured deltas, size budget, protected floors, and benchmark gates",
            "outputs": [str(tensor_map)],
        },
        {
            "name": "build",
            "purpose": "call stock llama-quantize with imatrix and exact tensor map to produce the candidate GGUF",
            "command": shell_join(
                [
                    *tensor_map_quant_base,
                    "--imatrix",
                    imatrix,
                    "--tensor-type-file",
                    tensor_map,
                    str(source),
                    final_gguf,
                    args.base_type,
                ]
            ),
            "outputs": [str(final_gguf)],
        },
        {
            "name": "gate",
            "purpose": "run size, PPL, TPS, EvalPlus, ARC, HellaSwag, MMLU-Redux, and audit wrong answers before release",
            "command": shell_join(["cerebellum", "benchmark-run", "--suite", args.benchmark_suite, "--model", final_gguf.stem, "--results-dir", benchmark_dir, "--execute", "--postprocess"]),
            "outputs": [str(benchmark_dir)],
        },
    ]
    lessons = [
        "Qwen36-27B v4 used selected tensor maps and corrected benchmark gates; it was not an exhaustive tensor walk.",
        "Qwen36-35B v1 showed blanket group demotion can break behavior; v2 reduced the override set and verified benchmarks.",
        "Gemma 4 experiments showed higher precision can be worse than the calibrated lower quant for some routed tensors.",
        "PPL is useful but not sufficient; downstream benchmarks and audit checks decide whether a candidate ships.",
    ]
    return {
        "schema": "cerebellum.legacy_gated_workflow.v1",
        "mode": "group-first benchmark-gated",
        "hillstep_role": "targeted refinement after scan/group/reverse-ablation evidence",
        "source_role": "F16 GGUF is the quantization source only; quant outputs are served and benchmarked",
        "orchestration": {
            "quant_ppl_overlap": "HillStepper overlaps candidate quantization with PPL measurement when low-space and serial-candidates are disabled",
            "low_space_mode": "serialize candidate testing and prune measured GGUFs immediately",
            "watch": shell_join(["cerebellum", "watch", run_dir]),
            "cleanup": "keep durable state/checkpoints; prune non-winning measured candidates unless explicitly kept",
        },
        "steps": steps,
        "protected_groups": LEGACY_GATED_GROUPS,
        "lessons": lessons,
    }


def pipeline_plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    run_dir = Path(args.run_dir) if args.run_dir else output_dir / "run"
    imatrix = Path(args.imatrix) if args.imatrix else output_dir / "imatrix.dat"
    source = Path(args.source_gguf)
    task_profile = TASK_PROFILES.get(args.task_profile) if args.task_profile else None
    effective_profile = str(task_profile["ppl_profile"]) if task_profile and args.profile == "custom" else args.profile
    effective_metric = args.metric or (str(task_profile["ablation_metric"]) if task_profile else "ppl")
    effective_suite = str(task_profile["benchmark_suite"]) if task_profile and args.benchmark_suite == "release" else args.benchmark_suite
    effective_low_space = bool(args.low_space or (task_profile and task_profile.get("low_space_default")))
    variant_suffix = f"-{task_profile['variant_suffix']}" if task_profile and task_profile.get("variant_suffix") != "general" else ""
    model_label = f"{slug(args.model_name or source.stem).lower()}{variant_suffix}"
    final_gguf = output_dir / f"{model_label}-cerebellum.gguf"
    benchmark_dir = output_dir / "benchmark_results"

    run_parts = [
        "cerebellum",
        "run",
        "--source-gguf",
        str(source),
        "--run-dir",
        str(run_dir),
        "--profile",
        effective_profile,
        "--metric",
        effective_metric,
        "--base-type",
        args.base_type,
        "--start-type",
        args.start_type,
        "--levels",
        args.levels,
        "--imatrix",
        str(imatrix),
        "--quantize-bin",
        args.quantize_bin,
        "--perplexity-bin",
        args.perplexity_bin,
        "--gpu-layers",
        str(args.gpu_layers),
        "--ctx-size",
        str(args.ctx_size),
    ]
    optional_flag(run_parts, "--corpus", args.corpus)
    optional_flag(run_parts, "--family", args.family)
    optional_flag(run_parts, "--model-name", args.model_name)
    optional_flag(run_parts, "--source-name", args.source_name)
    optional_flag(run_parts, "--run-name", args.run_name)
    optional_flag(run_parts, "--data-root", args.data_root)
    optional_flag(run_parts, "--scratch-root", args.scratch_root)
    optional_flag(run_parts, "--chunks", args.chunks)
    optional_flag(run_parts, "--distrobox", args.distrobox)
    bool_flag(run_parts, "--low-space", effective_low_space)
    bool_flag(run_parts, "--pure-quant", getattr(args, "pure_quant", False))

    final_quant_parts = [
        args.quantize_bin,
        "--allow-requantize",
        "--token-embedding-type",
        "f16",
        "--tensor-type-file",
        str(run_dir / "artifacts" / "final_types.txt"),
    ]
    if imatrix:
        final_quant_parts.extend(["--imatrix", str(imatrix)])
    final_quant_parts.extend([str(source), str(final_gguf), args.base_type])

    if args.distrobox:
        final_quant_command = shell_join(["distrobox", "enter", args.distrobox, "--", *final_quant_parts])
    else:
        final_quant_command = shell_join(final_quant_parts)

    finalize_parts = [
        "cerebellum",
        "finalize",
        "--run-dir",
        str(run_dir),
        "--gguf",
        str(final_gguf),
        "--output-dir",
        str(output_dir / "finalize"),
    ]
    optional_flag(finalize_parts, "--repo-name", args.repo_name)
    phases = [
        {
            "name": "imatrix",
            "status": "planned",
            "command": shell_join(["cerebellum", "imatrix", "--model", source, "--output", imatrix]),
            "outputs": [str(imatrix)],
        },
        {
            "name": "ablate",
            "status": "planned",
            "command": shell_join(run_parts),
            "outputs": [str(run_dir / "state.json"), str(run_dir / "artifacts" / "final_types.txt")],
        },
        {
            "name": "resume",
            "status": "available",
            "command": shell_join(["cerebellum", "resume", run_dir, "--low-space"] if effective_low_space else ["cerebellum", "resume", run_dir]),
            "outputs": [str(run_dir / "COMPLETE")],
        },
        {
            "name": "build-final-gguf",
            "status": "planned",
            "command": final_quant_command,
            "outputs": [str(final_gguf)],
        },
        {
            "name": "benchmark",
            "status": "planned",
            "command": shell_join(
                [
                    "cerebellum",
                    "benchmark-run",
                    "--suite",
                    effective_suite,
                    "--model",
                    model_label,
                    "--port",
                    args.benchmark_port,
                    "--results-dir",
                    benchmark_dir,
                    "--execute",
                    "--postprocess",
                    "--require-complete",
                ]
            ),
            "outputs": [str(benchmark_dir)],
        },
        {
            "name": "finalize",
            "status": "planned",
            "command": shell_join(finalize_parts),
            "outputs": [str(output_dir / "finalize")],
        },
        {
            "name": "package",
            "status": "planned",
            "command": shell_join(["cerebellum", "package", run_dir, "--output", output_dir / "package_manifest.json"]),
            "outputs": [str(output_dir / "package_manifest.json")],
        },
    ]
    cpu_offload = (
        cpu_offload_pipeline_detail(source, output_dir, run_dir, imatrix, model_label, args)
        if args.task_profile == "cpu-offload"
        else None
    )
    legacy_gated = (
        legacy_gated_workflow_detail(source, output_dir, run_dir, imatrix, final_gguf, benchmark_dir, args)
        if args.task_profile == "legacy-gated"
        else None
    )
    return {
        "pipeline": "cerebellum",
        "source_gguf": str(source),
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "imatrix": str(imatrix),
        "final_gguf": str(final_gguf),
        "benchmark_suite": effective_suite,
        "task_profile": args.task_profile,
        "task_profile_detail": task_profile,
        "resource_strategy": task_profile.get("resource_strategy") if task_profile else None,
        "cpu_offload_plan": cpu_offload,
        "legacy_gated_workflow": legacy_gated,
        "low_space": effective_low_space,
        "ppl_profile": effective_profile,
        "ablation_metric": effective_metric,
        "phases": phases,
    }


def pipeline_plan_markdown(plan: dict[str, Any]) -> str:
    rows = [[row["name"], row["status"], row["command"]] for row in plan["phases"]]
    output_rows = []
    for row in plan["phases"]:
        for output in row.get("outputs", []):
            output_rows.append([row["name"], output])
    parts = [
        "# Cerebellum Pipeline Plan",
        "",
        f"source: `{plan['source_gguf']}`",
        f"run: `{plan['run_dir']}`",
        f"final: `{plan['final_gguf']}`",
        f"low-space: `{plan.get('low_space')}`",
        "",
        markdown_table(["Phase", "Status", "Command"], rows),
    ]
    if plan.get("resource_strategy"):
        strategy = plan["resource_strategy"]
        parts.extend(
            [
                "",
                "## Resource Strategy",
                "",
                markdown_table(["Key", "Value"], [[str(key), str(value)] for key, value in strategy.items()]),
            ]
        )
    if plan.get("cpu_offload_plan"):
        offload = plan["cpu_offload_plan"]
        runtime = offload["runtime_targets"]
        streaming = offload["streaming"]
        dry_run = offload.get("streaming_quant_dry_run") or {}
        parts.extend(
            [
                "",
                "## CPU Offload Plan",
                "",
                markdown_table(
                    ["Key", "Value"],
                    [
                        ["model hint", str(offload.get("model_hint") or "-")],
                        ["source GiB", "-" if offload.get("source_size_gib") is None else f"{offload['source_size_gib']:.2f}"],
                        ["full RAM load required", str(streaming["full_model_ram_load_required"])],
                        ["scratch mode", str(streaming["scratch_mode"])],
                        ["gpu offload layers", str(runtime["gpu_offload_layers"])],
                        ["throughput probe", str(offload["throughput_probe_command"])],
                        ["dynamic compare", str(offload["dynamic_compare_command"])],
                    ],
                ),
            ]
        )
        disk_rows = [
            [
                str(row["phase"]),
                str(row["requirement"]),
                "-" if row.get("additional_gib") is None else str(row["additional_gib"]),
                str(row["note"]),
            ]
            for row in dry_run.get("disk_requirements", [])
        ]
        flow_rows = [
            [
                str(row["phase"]),
                ", ".join(str(item) for item in row.get("inputs", [])),
                ", ".join(str(item) for item in row.get("outputs", [])),
            ]
            for row in dry_run.get("artifact_flow", [])
        ]
        if disk_rows:
            parts.extend(["", "### Streaming Disk Dry Run", "", markdown_table(["Phase", "Requirement", "Add GiB", "Note"], disk_rows)])
        if flow_rows:
            parts.extend(["", "### Streaming Artifact Flow", "", markdown_table(["Phase", "Inputs", "Outputs"], flow_rows)])
    if output_rows:
        parts.extend(["", "## Outputs", "", markdown_table(["Phase", "Path"], output_rows)])
    return "\n".join(parts) + "\n"


def pipeline_plan_cmd(args: argparse.Namespace) -> None:
    plan = pipeline_plan(args)
    if args.write:
        Path(args.write).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.write)
        return
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(pipeline_plan_markdown(plan), end="")


def pipeline_run_plan(
    manifest_path: Path,
    from_phase: str | None = None,
    until_phase: str | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phases = manifest.get("phases")
    if not isinstance(phases, list) or not phases:
        raise SystemExit("pipeline manifest has no phases")
    names = [str(phase.get("name")) for phase in phases]
    if from_phase and from_phase not in names:
        raise SystemExit(f"unknown --from-phase {from_phase!r}")
    if until_phase and until_phase not in names:
        raise SystemExit(f"unknown --until-phase {until_phase!r}")
    start = names.index(from_phase) if from_phase else 0
    end = names.index(until_phase) + 1 if until_phase else len(phases)
    if start >= end:
        raise SystemExit("--from-phase must come before --until-phase")
    selected = []
    blockers = []
    for index, phase in enumerate(phases[start:end], start=start):
        command = phase.get("command")
        row = {
            "index": index,
            "name": phase.get("name"),
            "status": phase.get("status", "planned"),
            "command": command,
            "outputs": phase.get("outputs", []),
        }
        selected.append(row)
        if not command:
            blockers.append({"phase": phase.get("name"), "reason": "missing command"})
    return {
        "schema": "cerebellum.pipeline_run.v1",
        "manifest": str(manifest_path),
        "pipeline": manifest.get("pipeline"),
        "run_dir": manifest.get("run_dir"),
        "dry_run": True,
        "phases": selected,
        "blocked": bool(blockers),
        "blockers": blockers,
    }


def pipeline_run_markdown(plan: dict[str, Any]) -> str:
    rows = [
        [str(row["index"]), str(row["name"]), str(row["status"]), str(row["command"] or "-")]
        for row in plan["phases"]
    ]
    parts = [
        "# Cerebellum Pipeline Run",
        "",
        f"manifest: `{plan['manifest']}`",
        f"mode: `{'dry-run' if plan.get('dry_run') else 'execute'}`",
        "",
        markdown_table(["#", "Phase", "Status", "Command"], rows),
    ]
    if plan.get("blockers"):
        parts.extend(["", "## Blockers", "", markdown_table(["Phase", "Reason"], [[row["phase"], row["reason"]] for row in plan["blockers"]])])
    if plan.get("executions"):
        parts.extend(
            [
                "",
                "## Executions",
                "",
                markdown_table(
                    ["Phase", "Return", "Log"],
                    [[str(row["phase"]), str(row["returncode"]), str(row["log"])] for row in plan["executions"]],
                ),
            ]
        )
    return "\n".join(parts) + "\n"


def append_pipeline_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": utc_now(), **event}, sort_keys=True) + "\n")


def pipeline_run_execute(plan: dict[str, Any]) -> dict[str, Any]:
    if plan["blocked"]:
        raise SystemExit("pipeline-run has blockers; fix the manifest before --execute")
    manifest_path = Path(plan["manifest"])
    log_dir = manifest_path.parent / "pipeline_run_logs"
    event_log = manifest_path.parent / "pipeline_run_events.jsonl"
    executions: list[dict[str, Any]] = []
    for phase in plan["phases"]:
        command = str(phase["command"])
        name = slug(str(phase["name"]))
        log_path = log_dir / f"{int(phase['index']):02d}_{name}.log"
        append_pipeline_event(event_log, {"event": "phase_start", "phase": phase["name"], "command": command})
        started = time.monotonic()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(shlex.split(command), cwd=manifest_path.parent, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        elapsed = time.monotonic() - started
        row = {
            "phase": phase["name"],
            "command": command,
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "log": str(log_path),
        }
        executions.append(row)
        append_pipeline_event(event_log, {"event": "phase_finish", **row})
        if proc.returncode != 0:
            plan.update({"dry_run": False, "blocked": True, "blockers": [{"phase": phase["name"], "reason": f"command exited {proc.returncode}"}], "executions": executions, "event_log": str(event_log)})
            return plan
    plan.update({"dry_run": False, "executions": executions, "event_log": str(event_log)})
    return plan


def pipeline_run_cmd(args: argparse.Namespace) -> None:
    plan = pipeline_run_plan(Path(args.manifest), from_phase=args.from_phase, until_phase=args.until_phase)
    if args.execute:
        plan = pipeline_run_execute(plan)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(pipeline_run_markdown(plan), end="")
    if plan["blocked"]:
        raise SystemExit(1)


def pipeline_run_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    manifest = query_value(qs, "manifest")
    if not manifest:
        raise ValueError("manifest query param required")
    return argparse.Namespace(
        manifest=manifest,
        from_phase=query_value(qs, "from_phase"),
        until_phase=query_value(qs, "until_phase"),
    )


def pipeline_status(
    manifest_path: Path,
    events_path: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phases = manifest.get("phases")
    if not isinstance(phases, list) or not phases:
        raise SystemExit("pipeline manifest has no phases")
    event_log = events_path or manifest_path.parent / "pipeline_run_events.jsonl"
    events = read_jsonl(event_log)
    phase_rows: list[dict[str, Any]] = []
    phase_index: dict[str, dict[str, Any]] = {}
    for index, phase in enumerate(phases):
        name = str(phase.get("name"))
        row = {
            "index": index,
            "name": name,
            "status": "pending",
            "command": phase.get("command"),
            "outputs": phase.get("outputs", []),
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "elapsed_seconds": None,
            "log": None,
        }
        phase_rows.append(row)
        phase_index[name] = row
    for event in events:
        name = str(event.get("phase") or "")
        row = phase_index.get(name)
        if row is None:
            continue
        kind = event.get("event")
        if kind == "phase_start":
            row["status"] = "running"
            row["started_at"] = event.get("time")
            row["command"] = event.get("command") or row.get("command")
        elif kind == "phase_finish":
            returncode = int(event.get("returncode", 0))
            row["status"] = "complete" if returncode == 0 else "failed"
            row["finished_at"] = event.get("time")
            row["returncode"] = returncode
            row["elapsed_seconds"] = event.get("elapsed_seconds")
            row["log"] = event.get("log")
            row["command"] = event.get("command") or row.get("command")
    failed = next((row for row in phase_rows if row["status"] == "failed"), None)
    running = next((row for row in phase_rows if row["status"] == "running"), None)
    pending = next((row for row in phase_rows if row["status"] == "pending"), None)
    if failed:
        status = "failed"
        resume_phase = failed["name"]
    elif running:
        status = "running"
        resume_phase = running["name"]
    elif pending and any(row["status"] == "complete" for row in phase_rows):
        status = "partial"
        resume_phase = pending["name"]
    elif pending:
        status = "not-started" if not events else "partial"
        resume_phase = pending["name"]
    else:
        status = "complete"
        resume_phase = None
    resume_command = None
    if resume_phase:
        resume_command = shell_join(["cerebellum", "pipeline-run", "--manifest", manifest_path, "--from-phase", resume_phase, "--execute"])
    return {
        "schema": "cerebellum.pipeline_status.v1",
        "manifest": str(manifest_path),
        "event_log": str(event_log),
        "event_count": len(events),
        "pipeline": manifest.get("pipeline"),
        "run_dir": manifest.get("run_dir"),
        "status": status,
        "completed_phases": sum(1 for row in phase_rows if row["status"] == "complete"),
        "failed_phase": failed["name"] if failed else None,
        "running_phase": running["name"] if running else None,
        "next_phase": pending["name"] if pending else None,
        "resume_phase": resume_phase,
        "resume_command": resume_command,
        "last_event": events[-1] if events else None,
        "phases": phase_rows,
    }


def pipeline_status_markdown(status: dict[str, Any]) -> str:
    rows = [
        [
            str(row["index"]),
            str(row["name"]),
            str(row["status"]),
            "-" if row.get("returncode") is None else str(row["returncode"]),
            str(row.get("log") or "-"),
        ]
        for row in status["phases"]
    ]
    parts = [
        "# Cerebellum Pipeline Status",
        "",
        f"manifest: `{status['manifest']}`",
        f"status: `{status['status']}`",
        f"events: `{status['event_count']}`",
        f"completed: `{status['completed_phases']}/{len(status['phases'])}`",
    ]
    if status.get("resume_command"):
        parts.append(f"resume: `{status['resume_command']}`")
    parts.extend(["", markdown_table(["#", "Phase", "Status", "Return", "Log"], rows)])
    return "\n".join(parts) + "\n"


def pipeline_status_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    manifest = query_value(qs, "manifest")
    if not manifest:
        raise ValueError("manifest query param required")
    return argparse.Namespace(
        manifest=manifest,
        events=query_value(qs, "events"),
    )


def pipeline_status_cmd(args: argparse.Namespace) -> None:
    result = pipeline_status(Path(args.manifest), Path(args.events) if args.events else None)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(pipeline_status_markdown(result), end="")


def cpu_offload_pipeline_detail(
    source: Path,
    output_dir: Path,
    run_dir: Path,
    imatrix: Path,
    model_label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_size_gib = path_size(source) / (1024**3) if source.exists() else None
    final_types = run_dir / "artifacts" / "final_types.txt"
    final_gguf = output_dir / f"{model_label}-cerebellum.gguf"
    streaming_dry_run = cpu_offload_streaming_quant_dry_run(source, output_dir, run_dir, imatrix, final_types, final_gguf, source_size_gib, args)
    return {
        "profile": "cpu-offload",
        "model_hint": "glm" if "glm" in f"{source.name} {args.model_name}".lower() else None,
        "source_size_gib": source_size_gib,
        "streaming": {
            "imatrix": str(imatrix),
            "tensor_map": str(final_types),
            "scratch_mode": "low-space streaming candidates; measured losers pruned",
            "full_model_ram_load_required": False,
        },
        "streaming_quant_dry_run": streaming_dry_run,
        "runtime_targets": {
            "primary": "large RAM host with optional GPU offload",
            "record": ["cpu_tok_s", "ram_gib", "gpu_offload_layers", "size_gib", "score_per_gib"],
            "gpu_offload_layers": args.gpu_layers,
        },
        "smoke_commands": [
            shell_join(["cerebellum", "inspect-gguf-types", source, "--by-component", "--json"]),
            shell_join(
                [
                    "cerebellum",
                    "benchmark-plan",
                    "--suite",
                    "release",
                    "--model",
                    model_label,
                    "--port",
                    args.benchmark_port,
                    "--results-dir",
                    output_dir / "benchmark_results",
                ]
            ),
        ],
        "throughput_probe_command": shell_join(
            [
                "BENCH_MODEL=" + model_label,
                "BENCH_PORT=" + str(args.benchmark_port),
                "RESULTS_DIR=" + str(output_dir / "benchmark_results"),
                "python",
                "scripts/benchmark_perf.py",
            ]
        ),
        "dynamic_compare_command": shell_join(
            [
                "cerebellum",
                "compare-gguf-types",
                source,
                "UNSLOTH_DYNAMIC_GGUF",
                "--baseline-label",
                "f16",
                "--candidate-label",
                "unsloth-dynamic",
                "--reference-map",
                final_types,
            ]
        ),
        "expected_outputs": [
            str(final_gguf),
            str(final_types),
            str(output_dir / "benchmark_results"),
            str(output_dir / "package_manifest.json"),
        ],
        "auth_blockers": ["HF access may be required for gated GLM/GPQA/HLE datasets; pipeline planning itself does not require auth."],
    }


def disk_requirement_row(phase: str, requirement: str, additional_gib: float | None, note: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "requirement": requirement,
        "additional_gib": None if additional_gib is None else round(additional_gib, 3),
        "note": note,
    }


def cpu_offload_streaming_quant_dry_run(
    source: Path,
    output_dir: Path,
    run_dir: Path,
    imatrix: Path,
    final_types: Path,
    final_gguf: Path,
    source_size_gib: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_upper = source_size_gib
    final_upper = source_size_gib
    disk_requirements = [
        disk_requirement_row("inspect-source", "read GGUF metadata and tensor types", 0.0, "stat/metadata only; does not load model weights into RAM"),
        disk_requirement_row("stream-imatrix", "write calibration imatrix", 0.3, "streaming imatrix target is roughly 300 MiB"),
        disk_requirement_row("ablate", "one active candidate GGUF plus temp files", candidate_upper, "low-space mode prunes measured non-winners before the next candidate"),
        disk_requirement_row("write-final-map", "write tensor type map", 0.001, "text sidecar under run artifacts"),
        disk_requirement_row("build-final-gguf", "write final optimized GGUF", final_upper, "upper bound equals source size until the tensor map is known"),
        disk_requirement_row("benchmark", "write benchmark JSON/JSONL artifacts", 2.0, "suite-dependent; HumanEval+/frontier artifacts may be larger"),
    ]
    artifact_flow = [
        {"phase": "inspect-source", "inputs": [str(source)], "outputs": [str(output_dir / "gguf_type_inspect.json")]},
        {"phase": "stream-imatrix", "inputs": [str(source)], "outputs": [str(imatrix)]},
        {"phase": "ablate", "inputs": [str(source), str(imatrix)], "outputs": [str(run_dir / "state.json"), str(final_types)]},
        {"phase": "build-final-gguf", "inputs": [str(source), str(imatrix), str(final_types)], "outputs": [str(final_gguf)]},
        {"phase": "benchmark", "inputs": [str(final_gguf)], "outputs": [str(output_dir / "benchmark_results")]},
        {"phase": "dynamic-compare", "inputs": [str(final_types), str(final_gguf), "UNSLOTH_DYNAMIC_GGUF"], "outputs": [str(output_dir / "dynamic_gguf_compare.json")]},
    ]
    return {
        "schema": "cerebellum.cpu_offload_streaming_quant_dry_run.v1",
        "dry_run": True,
        "model_load": "metadata/stat only; no full-model RAM load required for this plan",
        "source": str(source),
        "source_size_gib": source_size_gib,
        "low_space": True,
        "scratch_root": args.scratch_root,
        "disk_requirements": disk_requirements,
        "artifact_flow": artifact_flow,
        "preflight_commands": [
            shell_join(["cerebellum", "system"]),
            shell_join(["cerebellum", "plan-space", "--source-gguf", source, "--scratch-candidates", output_dir, "--margin-gb", "20"]),
            shell_join(["cerebellum", "inspect-gguf-types", source, "--by-component", "--json"]),
        ],
        "execution_guard": "plan only; actual streaming quant build must be launched separately and monitored",
    }


def cpu_offload_hazards(source: Path, model_name: str) -> list[dict[str, str]]:
    text = f"{source.name} {model_name}".lower()
    hazards = [
        {
            "name": "huge-gguf-disk",
            "status": "active",
            "detail": "Source, one active candidate, final output, logs, and benchmark artifacts can require multiple source-size equivalents; use game-drive scratch when main drive is low.",
        },
        {
            "name": "cpu-offload-objective",
            "status": "active",
            "detail": "Allocation must record CPU tok/s, RAM GiB, GPU offload layers, size GiB, and quality score; size/PPL alone is not enough for this target.",
        },
        {
            "name": "streaming-build",
            "status": "active",
            "detail": "Planning and inspection are metadata/stat only; candidate/final GGUF phases still write large files and must be monitored.",
        },
    ]
    if "glm" in text:
        hazards.append(
            {
                "name": "glm-layout-unverified",
                "status": "needs-inspection",
                "detail": "Confirm GLM-5.1 tensor names, architecture metadata, and unsupported tensor layouts with inspect-gguf-types before a full build.",
            }
        )
    return hazards


def cpu_offload_smoke_payload(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_gguf)
    output_dir = Path(args.output_dir)
    scratch_roots = [output_dir]
    if args.scratch_root:
        scratch_roots.append(Path(args.scratch_root))
    scratch_roots.extend([default_data_root(), Path.cwd()])
    plan_args = pipeline_plan_args_from_query(
        {
            "source_gguf": [str(source)],
            "output_dir": [str(output_dir)],
            "model_name": [args.model_name],
            "task_profile": ["cpu-offload"],
            "scratch_root": [args.scratch_root] if args.scratch_root else [None],
            "benchmark_port": [str(args.benchmark_port)],
            "low_space": ["true"],
        }
    )
    plan = pipeline_plan(plan_args)
    inspect_payload: dict[str, Any] = {"skipped": bool(args.skip_inspect)}
    if not args.skip_inspect:
        try:
            summary = inspect_gguf_types(source)
            inspect_payload = {
                "ok": True,
                "tensor_count": summary["tensor_count"],
                "quantizable_tensor_count": summary["quantizable_tensor_count"],
                "type_counts": summary["type_counts"],
            }
        except (Exception, SystemExit) as exc:
            inspect_payload = {"ok": False, "error": str(exc)}
    checks = [
        {"name": "source_exists", "ok": source.exists(), "detail": str(source)},
        {"name": "source_stat_only", "ok": True, "detail": "source size computed with stat; no full-model RAM load"},
        {"name": "pipeline_plan", "ok": plan.get("task_profile") == "cpu-offload", "detail": plan.get("run_dir")},
        {"name": "space_plan", "ok": True, "detail": "computed from disk usage and source size"},
    ]
    if args.skip_inspect:
        checks.append({"name": "inspect_gguf_types", "ok": True, "detail": "skipped by --skip-inspect"})
    else:
        checks.append({"name": "inspect_gguf_types", "ok": bool(inspect_payload.get("ok")), "detail": inspect_payload.get("error") or inspect_payload.get("type_counts")})
    blockers = [row for row in checks if not row["ok"]]
    if inspect_payload.get("ok") is False and not args.require_inspect:
        blockers = [row for row in blockers if row["name"] != "inspect_gguf_types"]
    return {
        "schema": "cerebellum.cpu_offload_smoke.v1",
        "dry_run": True,
        "source": str(source),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "full_model_ram_load_required": False,
        "source_size_gib": path_size(source) / (1024**3) if source.exists() else None,
        "checks": checks,
        "blocked": bool(blockers),
        "blockers": blockers,
        "hazards": cpu_offload_hazards(source, args.model_name),
        "space": space_plan(source, scratch_roots, args.margin_gb, create_dirs=getattr(args, "create_dirs", True)),
        "pipeline": plan,
        "inspect": inspect_payload,
        "smoke_commands": [
            shell_join(["cerebellum", "plan-space", "--source-gguf", source, "--scratch-candidates", output_dir, "--margin-gb", args.margin_gb, "--json"]),
            shell_join(["cerebellum", "pipeline-plan", "--source-gguf", source, "--output-dir", output_dir, "--model-name", args.model_name, "--task-profile", "cpu-offload", "--json"]),
            shell_join(["cerebellum", "inspect-gguf-types", source, "--by-component", "--json"]),
        ],
    }


def cpu_offload_smoke_markdown(payload: dict[str, Any]) -> str:
    checks = [[row["name"], "ok" if row["ok"] else "blocked", str(row.get("detail") or "-")] for row in payload["checks"]]
    hazards = [[row["name"], row["status"], row["detail"]] for row in payload["hazards"]]
    commands = [[command] for command in payload["smoke_commands"]]
    parts = [
        "# CPU-Offload Smoke",
        "",
        f"source: `{payload['source']}`",
        f"output_dir: `{payload['output_dir']}`",
        f"full_model_ram_load_required: `{payload['full_model_ram_load_required']}`",
        f"status: `{'blocked' if payload['blocked'] else 'ready'}`",
        "",
        "## Checks",
        "",
        markdown_table(["Check", "Status", "Detail"], checks),
        "",
        "## Hazards",
        "",
        markdown_table(["Hazard", "Status", "Detail"], hazards),
        "",
        "## Commands",
        "",
        markdown_table(["Command"], commands),
    ]
    return "\n".join(parts) + "\n"


def cpu_offload_smoke_cmd(args: argparse.Namespace) -> None:
    payload = cpu_offload_smoke_payload(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(cpu_offload_smoke_markdown(payload), end="")
    if payload["blocked"]:
        raise SystemExit(1)


def cpu_offload_build_plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_gguf)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "cpu_offload_pipeline.json"
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.create_dirs = False
    smoke = cpu_offload_smoke_payload(smoke_args)
    pipeline = smoke["pipeline"]
    cpu_plan = pipeline["cpu_offload_plan"]
    dry_run = cpu_plan["streaming_quant_dry_run"]
    run_dir = Path(pipeline["run_dir"])
    commands = {
        "prepare_output_dir": shell_join(["mkdir", "-p", output_dir]),
        "write_pipeline_manifest": shell_join(
            [
                "cerebellum",
                "pipeline-plan",
                "--source-gguf",
                source,
                "--output-dir",
                output_dir,
                "--model-name",
                args.model_name,
                "--task-profile",
                "cpu-offload",
                "--write",
                manifest_path,
                "--json",
            ]
        ),
        "queue_pipeline": shell_join(["cerebellum", "queue", "add", "--kind", "pipeline", "--manifest", manifest_path]),
        "run_next": shell_join(["cerebellum", "queue", "run-next", "--execute"]),
        "status": shell_join(["cerebellum", "pipeline-status", "--manifest", manifest_path]),
        "watch_ablation": shell_join(["cerebellum", "watch", run_dir]),
        "resume_ablation": shell_join(["cerebellum", "resume", run_dir, "--low-space"]),
        "benchmark_status": shell_join(["cerebellum", "benchmark-status", "--results-dir", output_dir / "benchmark_results"]),
        "release_gate": shell_join(
            [
                "cerebellum",
                "release-gate",
                "README.md",
                "docs",
                output_dir / "benchmark_results",
                "--remote",
                "origin",
                "--benchmark-results",
                output_dir / "benchmark_results",
                "--suite",
                "release",
                "--model",
                args.model_name,
                "--require-benchmarks",
            ]
        ),
    }
    operator_steps = [
        {"step": "prepare", "command": commands["prepare_output_dir"], "notes": "create the output directory for manifests and later artifacts"},
        {"step": "preflight", "command": commands["write_pipeline_manifest"], "notes": "writes the exact cpu-offload pipeline manifest; no quantization"},
        {"step": "queue", "command": commands["queue_pipeline"], "notes": "stores the manifest in the Cerebellum queue"},
        {"step": "execute", "command": commands["run_next"], "notes": "launches the queued pipeline when ready"},
        {"step": "monitor", "command": commands["watch_ablation"], "notes": "watch the long ablation run and locked tensor map"},
        {"step": "resume", "command": commands["resume_ablation"], "notes": "resume after interruption without discarding progress"},
        {"step": "benchmarks", "command": commands["benchmark_status"], "notes": "track throughput and benchmark artifacts"},
        {"step": "public gate", "command": commands["release_gate"], "notes": "block origin pushes until public/benchmark gates pass"},
    ]
    return {
        "schema": "cerebellum.cpu_offload_build_plan.v1",
        "dry_run": True,
        "source": str(source),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "model_name": args.model_name,
        "full_model_ram_load_required": False,
        "ready": not smoke["blocked"],
        "blockers": smoke["blockers"],
        "hazards": smoke["hazards"],
        "space": smoke["space"],
        "pipeline": pipeline,
        "streaming_quant": dry_run,
        "artifact_flow": dry_run["artifact_flow"],
        "disk_requirements": dry_run["disk_requirements"],
        "expected_outputs": cpu_plan["expected_outputs"],
        "runtime_targets": cpu_plan["runtime_targets"],
        "operator_steps": operator_steps,
        "commands": commands,
        "notes": [
            "This command does not run quantization or load the full model into RAM.",
            "Use the emitted queue/run/watch commands to launch and monitor the actual GLM-scale build.",
            "CPU-offload success requires recording CPU tok/s, RAM GiB, GPU offload layers, final size, and benchmark quality.",
        ],
    }


def cpu_offload_build_plan_markdown(payload: dict[str, Any]) -> str:
    rows = [
        [row["phase"], row["requirement"], "-" if row["additional_gib"] is None else str(row["additional_gib"]), row["note"]]
        for row in payload["disk_requirements"]
    ]
    steps = [[row["step"], row["command"], row["notes"]] for row in payload["operator_steps"]]
    hazards = [[row["name"], row["status"], row["detail"]] for row in payload["hazards"]]
    parts = [
        "# CPU-Offload Build Plan",
        "",
        f"source: `{payload['source']}`",
        f"manifest: `{payload['manifest']}`",
        f"status: `{'ready' if payload['ready'] else 'blocked'}`",
        f"full_model_ram_load_required: `{payload['full_model_ram_load_required']}`",
        "",
        "## Operator Steps",
        "",
        markdown_table(["Step", "Command", "Notes"], steps),
        "",
        "## Disk Requirements",
        "",
        markdown_table(["Phase", "Requirement", "Additional GiB", "Note"], rows),
        "",
        "## Hazards",
        "",
        markdown_table(["Hazard", "Status", "Detail"], hazards),
    ]
    if payload["blockers"]:
        blockers = [[row["name"], str(row.get("detail") or "-")] for row in payload["blockers"]]
        parts.extend(["", "## Blockers", "", markdown_table(["Check", "Detail"], blockers)])
    parts.extend(["", "## Notes", "", *[f"- {note}" for note in payload["notes"]]])
    return "\n".join(parts) + "\n"


def cpu_offload_build_plan_cmd(args: argparse.Namespace) -> None:
    payload = cpu_offload_build_plan_payload(args)
    if args.write:
        output_path = Path(args.write)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(cpu_offload_build_plan_markdown(payload), end="")
    if payload["blockers"]:
        raise SystemExit(1)


def cpu_offload_smoke_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    source = query_value(qs, "source_gguf")
    output_dir = query_value(qs, "output_dir")
    if not source:
        raise ValueError("source_gguf query param required")
    if not output_dir:
        raise ValueError("output_dir query param required")
    try:
        benchmark_port = int(query_value(qs, "benchmark_port", "8084"))
    except ValueError as exc:
        raise ValueError("benchmark_port must be an integer") from exc
    try:
        margin_gb = float(query_value(qs, "margin_gb", "20.0"))
    except ValueError as exc:
        raise ValueError("margin_gb must be a number") from exc
    return argparse.Namespace(
        source_gguf=source,
        output_dir=output_dir,
        model_name=query_value(qs, "model_name", "GLM-5.1"),
        scratch_root=query_value(qs, "scratch_root"),
        benchmark_port=benchmark_port,
        margin_gb=margin_gb,
        skip_inspect=query_bool(qs, "skip_inspect", False),
        require_inspect=query_bool(qs, "require_inspect", False),
        create_dirs=False,
    )


def query_value(qs: dict[str, list[str]], key: str, default: Any = None) -> Any:
    return qs.get(key, [default])[0]


def query_bool(qs: dict[str, list[str]], key: str, default: bool = False) -> bool:
    value = str(query_value(qs, key, str(default))).lower()
    return value in {"1", "true", "yes", "on"}


def pipeline_plan_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    source = query_value(qs, "source_gguf")
    output_dir = query_value(qs, "output_dir")
    if not source:
        raise ValueError("source_gguf query param required")
    if not output_dir:
        raise ValueError("output_dir query param required")
    task_profile = query_value(qs, "task_profile")
    profile = query_value(qs, "profile", "custom")
    benchmark_suite = query_value(qs, "benchmark_suite", "release-local")
    metric = query_value(qs, "metric")
    if task_profile and task_profile not in TASK_PROFILES:
        raise ValueError(f"unknown task_profile {task_profile}")
    if profile not in {*PPL_PROFILES, "custom"}:
        raise ValueError(f"unknown profile {profile}")
    if benchmark_suite not in BENCHMARK_SUITES:
        raise ValueError(f"unknown benchmark_suite {benchmark_suite}")
    if metric and metric not in {"ppl", "humaneval", "arc", "mmlu", "tool-call", "dialogue"}:
        raise ValueError(f"unknown metric {metric}")
    return argparse.Namespace(
        source_gguf=source,
        output_dir=output_dir,
        imatrix=query_value(qs, "imatrix"),
        corpus=query_value(qs, "corpus"),
        profile=profile,
        family=query_value(qs, "family"),
        model_name=query_value(qs, "model_name"),
        source_name=query_value(qs, "source_name"),
        run_name=query_value(qs, "run_name"),
        run_dir=query_value(qs, "run_dir"),
        data_root=query_value(qs, "data_root"),
        scratch_root=query_value(qs, "scratch_root"),
        base_type=query_value(qs, "base_type", "Q4_K_M"),
        start_type=query_value(qs, "start_type", "q4_K"),
        levels=query_value(qs, "levels", ",".join(DEFAULT_LEVELS)),
        quantize_bin=query_value(qs, "quantize_bin", DEFAULT_QUANTIZE),
        perplexity_bin=query_value(qs, "perplexity_bin", DEFAULT_PERPLEXITY),
        gpu_layers=int(query_value(qs, "gpu_layers", 99)),
        ctx_size=int(query_value(qs, "ctx_size", 2048)),
        chunks=query_value(qs, "chunks"),
        distrobox=query_value(qs, "distrobox"),
        low_space=query_bool(qs, "low_space"),
        benchmark_suite=benchmark_suite,
        metric=metric,
        task_profile=task_profile,
        benchmark_port=int(query_value(qs, "benchmark_port", 8084)),
        repo_name=query_value(qs, "repo_name"),
        write=None,
        json=True,
    )


def task_profiles_markdown() -> str:
    rows = []
    for key, profile in TASK_PROFILES.items():
        rows.append(
            [
                key,
                str(profile["ppl_profile"]),
                str(profile.get("ablation_metric", "ppl")),
                str(profile["benchmark_suite"]),
                ", ".join(str(metric) for metric in profile["metrics"]),
                str(profile["note"]),
            ]
        )
    return markdown_table(["Profile", "PPL", "Ablation metric", "Bench suite", "Metrics", "Note"], rows) + "\n"


def task_profiles_cmd(args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps({"profiles": TASK_PROFILES}, indent=2, sort_keys=True))
        return
    print(task_profiles_markdown(), end="")


def benchmark_input_specs(paths: list[Any]) -> list[tuple[Path, str | None]]:
    specs: list[tuple[Path, str | None]] = []
    for item in paths:
        text = str(item)
        if "=" in text and not Path(text).exists():
            label, path_text = text.split("=", 1)
            specs.append((Path(path_text), label or None))
        else:
            specs.append((Path(text), None))
    return specs


def benchmark_files(paths: list[Any]) -> list[tuple[Path, str | None]]:
    files: list[tuple[Path, str | None]] = []
    seen: set[Path] = set()
    for path, label in benchmark_input_specs(paths):
        candidates = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate in seen:
                continue
            seen.add(candidate)
            files.append((candidate, label))
    return files


def benchmark_artifact_files(paths: list[Any]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for path, _label in benchmark_input_specs(paths):
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in {".json", ".jsonl"}:
                continue
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(candidate)
    return files


def benchmark_artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".jsonl") or "detailed" in name or "samples" in name:
        return "detail"
    return "summary"


def benchmark_records(paths: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path, label in benchmark_files(paths):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        metric = benchmark_metric(data)
        if metric is None:
            continue
        metric_name, value = metric
        benchmark = infer_benchmark_name(path, data)
        records.append(
            {
                "model": str(label or data.get("model") or path.parent.name or path.stem),
                "benchmark": benchmark,
                "benchmark_key": benchmark_key(benchmark),
                "metric": metric_name,
                "value": value,
                "correct": data.get("correct"),
                "total": data.get("total") or data.get("total_problems"),
                "size_gib": benchmark_size_gib(data),
                "release_metadata": benchmark_release_metadata(data),
                "path": str(path),
            }
        )
    records.sort(key=lambda row: (str(row["benchmark"]), str(row["model"]), str(row["path"])))
    return records


def benchmark_manifest(paths: list[Any], suite: str = "release", model: str | None = None) -> dict[str, Any]:
    records = benchmark_records(paths)
    measured = sorted({str(row["benchmark_key"]) for row in records})
    suite_keys = list(BENCHMARK_SUITES[suite])
    files = []
    for path in benchmark_artifact_files(paths):
        data: dict[str, Any] = {}
        if path.suffix.lower() == ".json":
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}
        benchmark = infer_benchmark_name(path, data)
        files.append(
            {
                "path": str(path),
                "name": path.name,
                "benchmark": benchmark,
                "benchmark_key": benchmark_key(benchmark),
                "kind": benchmark_artifact_kind(path),
                "size_bytes": path_size(path),
                "sha256": sha256_file(path),
            }
        )
    files.sort(key=lambda item: (item["benchmark_key"], item["kind"], item["path"]))
    missing_measured = [key for key in suite_keys if key not in measured]
    return {
        "schema": "cerebellum.benchmark_manifest.v1",
        "suite": suite,
        "suite_purpose": BENCHMARK_SUITE_PURPOSES.get(suite, ""),
        "model": model,
        "paths": [str(path) for path in paths],
        "artifacts": files,
        "measured_benchmarks": measured,
        "missing_measured": missing_measured,
        "records": records,
        "release_metadata": benchmark_report(paths).get("release_metadata", {}),
    }


def benchmark_manifest_markdown(manifest: dict[str, Any]) -> str:
    rows = [
        [
            item["benchmark_key"],
            item["kind"],
            item["name"],
            fmt_bytes(item["size_bytes"]),
            item["sha256"],
        ]
        for item in manifest["artifacts"]
    ]
    parts = [
        "# Benchmark Artifact Manifest",
        "",
        f"suite: `{manifest['suite']}`",
        f"purpose: `{manifest.get('suite_purpose') or '-'}`",
        f"model: `{manifest.get('model') or '-'}`",
        "",
        markdown_table(["Benchmark", "Kind", "File", "Size", "SHA256"], rows) if rows else "No benchmark artifacts found.",
    ]
    if manifest["missing_measured"]:
        parts.extend(["", "## Missing Measured Benchmarks", "", ", ".join(manifest["missing_measured"])])
    return "\n".join(parts) + "\n"


def benchmark_manifest_cmd(args: argparse.Namespace) -> None:
    manifest = benchmark_manifest(args.paths, suite=args.suite, model=args.model)
    text = json.dumps(manifest, indent=2, sort_keys=True) if args.json else benchmark_manifest_markdown(manifest)
    if args.output:
        Path(args.output).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        print(args.output)
        if args.require_complete and manifest["missing_measured"]:
            raise SystemExit(1)
        return
    print(text, end="" if text.endswith("\n") else "\n")
    if args.require_complete and manifest["missing_measured"]:
        raise SystemExit(1)


def benchmark_manifest_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    paths = qs.get("path") or qs.get("paths") or []
    if not paths:
        raise ValueError("path query param required")
    suite = query_value(qs, "suite", "release")
    if suite not in BENCHMARK_SUITES:
        raise ValueError(f"unknown benchmark suite {suite}")
    return argparse.Namespace(
        paths=paths,
        suite=suite,
        model=query_value(qs, "model"),
        require_complete=query_bool(qs, "require_complete"),
    )


def metric_is_quality_percent(metric: str) -> bool:
    metric = metric.lower()
    return metric not in {"ppl", "tok/s", "gen tok/s"} and "tok/s" not in metric


def leaderboard_weight_policy(suite: str, weights: dict[str, float] | None = None) -> dict[str, float]:
    weights = weights or {}
    return {key: float(weights.get(key, 1.0)) for key in BENCHMARK_SUITES[suite]}


def benchmark_leaderboard(
    records: list[dict[str, Any]],
    suite: str,
    sizes: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    suite_keys = set(BENCHMARK_SUITES[suite])
    weight_policy = leaderboard_weight_policy(suite, weights)
    sizes = sizes or {}
    by_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
    embedded_sizes: dict[str, float] = {}
    for row in records:
        model = str(row["model"])
        size = row.get("size_gib")
        if size is not None and model not in embedded_sizes:
            embedded_sizes[model] = float(size)
        if row["benchmark_key"] not in suite_keys:
            continue
        if not metric_is_quality_percent(str(row["metric"])):
            continue
        by_model.setdefault(model, {}).setdefault(str(row["benchmark_key"]), []).append(row)

    leaderboard: list[dict[str, Any]] = []
    for model, benchmark_rows in by_model.items():
        if not benchmark_rows:
            continue
        weighted_total = 0.0
        total_weight = 0.0
        benchmark_scores: dict[str, float] = {}
        for key, rows in benchmark_rows.items():
            weight = float(weight_policy.get(key, 1.0))
            if weight <= 0:
                continue
            score = sum(float(row["value"]) for row in rows) / len(rows)
            benchmark_scores[key] = score
            weighted_total += score * weight
            total_weight += weight
        if total_weight <= 0:
            continue
        avg = weighted_total / total_weight
        size_gib = sizes.get(model, embedded_sizes.get(model))
        score_per_gib = None if not size_gib else avg / size_gib
        leaderboard.append(
            {
                "model": model,
                "suite": suite,
                "average_score": avg,
                "benchmarks": len(benchmark_scores),
                "benchmark_scores": benchmark_scores,
                "total_weight": total_weight,
                "size_gib": size_gib,
                "score_per_gib": score_per_gib,
            }
        )
    leaderboard.sort(key=lambda row: (row["average_score"], row["score_per_gib"] or 0.0), reverse=True)
    return leaderboard


def benchmark_report(
    paths: list[Any],
    baseline: str | None = None,
    suite: str = "full",
    leaderboard: bool = False,
    sizes: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    records = benchmark_records(paths)
    models = sorted({str(row["model"]) for row in records})
    benchmarks = sorted({str(row["benchmark"]) for row in records})
    metadata_by_model: dict[str, dict[str, Any]] = {}
    for row in records:
        model = str(row["model"])
        for key, value in row.get("release_metadata", {}).items():
            if value not in (None, "") and key not in metadata_by_model.setdefault(model, {}):
                metadata_by_model[model][key] = value
    by_key = {(row["benchmark"], row["model"]): row for row in records}
    deltas: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        baseline_row = by_key.get((benchmark, baseline)) if baseline else None
        if baseline_row is None and models:
            baseline_row = next((by_key.get((benchmark, model)) for model in models if by_key.get((benchmark, model))), None)
        if baseline_row is None:
            continue
        for model in models:
            row = by_key.get((benchmark, model))
            if row is None or row is baseline_row:
                continue
            deltas.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "baseline": baseline_row["model"],
                    "metric": row["metric"],
                    "delta": row["value"] - baseline_row["value"],
                    "value": row["value"],
                    "baseline_value": baseline_row["value"],
                }
            )
    report: dict[str, Any] = {"models": models, "benchmarks": benchmarks, "records": records, "deltas": deltas, "release_metadata": metadata_by_model}
    if leaderboard:
        weight_policy = leaderboard_weight_policy(suite, weights)
        report["suite"] = {
            "name": suite,
            "purpose": BENCHMARK_SUITE_PURPOSES.get(suite, ""),
            "benchmarks": BENCHMARK_SUITES[suite],
            "weights": weight_policy,
            "average_policy": "weighted mean of measured quality-percentage benchmarks only; speed and PPL are reported but excluded",
        }
        report["leaderboard"] = benchmark_leaderboard(records, suite, sizes=sizes, weights=weight_policy)
    return report


def fmt_metric_value(row: dict[str, Any] | None) -> str:
    if row is None:
        return "-"
    value = float(row["value"])
    metric = str(row["metric"])
    suffix = "" if "tok/s" in metric or metric == "ppl" else "%"
    detail = ""
    if row.get("correct") is not None and row.get("total") is not None:
        detail = f" ({row['correct']}/{row['total']})"
    return f"{value:.2f}{suffix}{detail}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def ascii_bar(label: str, value: float, max_value: float, width: int = 24) -> str:
    filled = 0 if max_value <= 0 else int(round((value / max_value) * width))
    filled = max(0, min(width, filled))
    return f"{label:<28} [{'#' * filled}{'.' * (width - filled)}] {value:.2f}"


def benchmark_report_markdown(report: dict[str, Any], include_bars: bool = True) -> str:
    records = report["records"]
    models = report["models"]
    benchmarks = report["benchmarks"]
    by_key = {(row["benchmark"], row["model"]): row for row in records}
    rows: list[list[str]] = []
    for benchmark in benchmarks:
        metric = next((str(row["metric"]) for row in records if row["benchmark"] == benchmark), "-")
        rows.append([benchmark, metric, *[fmt_metric_value(by_key.get((benchmark, model))) for model in models]])
    parts = ["# Benchmark Comparison", "", markdown_table(["Benchmark", "Metric", *models], rows)]
    release_metadata = report.get("release_metadata") or {}
    if release_metadata:
        metadata_rows = []
        for model in models:
            metadata = release_metadata.get(model, {})
            if not metadata:
                continue
            size = "-" if metadata.get("size_gib") is None else f"{float(metadata['size_gib']):.2f}"
            metadata_rows.append(
                [
                    model,
                    size,
                    "-" if metadata.get("bpw") is None else str(metadata["bpw"]),
                    "-" if metadata.get("quant_recipe") is None else str(metadata["quant_recipe"]),
                    "-" if metadata.get("tensor_map") is None else str(metadata["tensor_map"]),
                    "-" if metadata.get("gguf_sha256") is None else str(metadata["gguf_sha256"]),
                    "-" if metadata.get("runtime") is None else str(metadata["runtime"]),
                ]
            )
        if metadata_rows:
            parts.extend(
                [
                    "",
                    "## Release Metadata",
                    "",
                    markdown_table(["Model", "Size GiB", "BPW", "Quant recipe", "Tensor map", "GGUF SHA256", "Runtime"], metadata_rows),
                ]
            )
    if report["deltas"]:
        delta_rows = [
            [row["benchmark"], row["model"], str(row["baseline"]), f"{row['delta']:+.2f}"]
            for row in report["deltas"]
        ]
        parts.extend(["", "## Deltas", "", markdown_table(["Benchmark", "Model", "Baseline", "Delta"], delta_rows)])
    if report.get("leaderboard") is not None:
        suite = report.get("suite", {})
        suite_name = suite.get("name", "full") if isinstance(suite, dict) else "full"
        suite_purpose = suite.get("purpose", "") if isinstance(suite, dict) else ""
        weights = suite.get("weights", {}) if isinstance(suite, dict) else {}
        average_policy = suite.get("average_policy", "weighted mean of measured quality-percentage benchmarks only") if isinstance(suite, dict) else "weighted mean of measured quality-percentage benchmarks only"
        leaderboard_rows = []
        for row in report["leaderboard"]:
            size = "-" if row.get("size_gib") is None else f"{float(row['size_gib']):.2f}"
            density = "-" if row.get("score_per_gib") is None else f"{float(row['score_per_gib']):.2f}"
            leaderboard_rows.append(
                [
                    str(row["model"]),
                    f"{float(row['average_score']):.2f}%",
                    str(row["benchmarks"]),
                    size,
                    density,
                ]
            )
        parts.extend(
            [
                "",
                f"## Leaderboard ({suite_name})",
                "",
                f"Purpose: {suite_purpose or '-'}",
                "",
                f"Average: {average_policy}; default weight is 1.0 per benchmark.",
                "",
                "Weights: " + ", ".join(f"{key}={float(value):g}" for key, value in weights.items()),
                "",
                markdown_table(["Model", "Avg score", "Benchmarks", "Size GiB", "Score/GiB"], leaderboard_rows),
            ]
        )
    if include_bars:
        bar_lines: list[str] = []
        for benchmark in benchmarks:
            rows_for_benchmark = [row for row in records if row["benchmark"] == benchmark]
            if not rows_for_benchmark:
                continue
            max_value = max(float(row["value"]) for row in rows_for_benchmark)
            bar_lines.append(benchmark)
            bar_lines.extend(ascii_bar(str(row["model"]), float(row["value"]), max_value) for row in rows_for_benchmark)
        if bar_lines:
            parts.extend(["", "## Bars", "", "```text", *bar_lines, "```"])
    return "\n".join(parts) + "\n"


def benchmark_report_cmd(args: argparse.Namespace) -> None:
    if args.list_suites:
        payload = {"suites": BENCHMARK_SUITES}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for name, benchmarks in BENCHMARK_SUITES.items():
                print(f"{name}: {', '.join(benchmarks)}")
        return
    if not args.paths:
        raise SystemExit("benchmark-report requires at least one path unless --list-suites is used")
    sizes = read_size_json(args.size_json)
    sizes.update(parse_size_specs(args.size))
    weights = parse_weight_specs(args.weight)
    report = benchmark_report(
        args.paths,
        baseline=args.baseline,
        suite=args.suite,
        leaderboard=args.leaderboard,
        sizes=sizes,
        weights=weights,
    )
    if args.json:
        text = json.dumps(report, indent=2, sort_keys=True)
    else:
        text = benchmark_report_markdown(report, include_bars=not args.no_bars)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(args.output)
        return
    print(text, end="" if text.endswith("\n") else "\n")


def benchmark_report_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    list_suites = query_bool(qs, "list_suites")
    paths = qs.get("path") or qs.get("paths") or []
    if not paths and not list_suites:
        raise ValueError("path query param required")
    suite = query_value(qs, "suite", "full")
    if suite not in BENCHMARK_SUITES:
        raise ValueError(f"unknown benchmark suite {suite}")
    return argparse.Namespace(
        paths=paths,
        baseline=query_value(qs, "baseline"),
        leaderboard=query_bool(qs, "leaderboard"),
        suite=suite,
        size=qs.get("size") or [],
        size_json=query_value(qs, "size_json"),
        weight=qs.get("weight") or [],
        list_suites=list_suites,
    )


def benchmark_audit_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for item in paths:
        path = Path(item)
        candidates = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(candidate)
    return files


def evalplus_completion(entry: dict[str, Any]) -> tuple[str, str]:
    prompt = str(entry.get("prompt") or entry.get("solution") or "")
    completion = str(entry.get("completion") or "")
    if not completion and "solution" in entry:
        solution = str(entry["solution"])
        if '"""' in solution:
            completion = solution[solution.rindex('"""') + 3 :]
        else:
            completion = solution
    return prompt, completion


def classify_evalplus_completion(completion: str, prompt: str = "") -> str:
    body = completion.strip()
    if not body:
        return "empty"
    if body in {"pass", "return", "return None", "None", "..."}:
        return "pass_only"
    code_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        code_lines.append(stripped)
    if not code_lines:
        return "empty"
    if all(re.fullmatch(r"(pass|return\s*(None|True|False|0|\d+|\[\]|\{\}|\"\"|'')?)", line) for line in code_lines):
        return "pass_only"
    if prompt:
        prompt_name = re.search(r"^\s*def\s+(\w+)\(", prompt, flags=re.MULTILINE)
        if prompt_name and body.count(f"def {prompt_name.group(1)}(") > 0:
            return "prompt_echo"
    return "attempt"


def audit_jsonl_file(path: Path) -> dict[str, Any]:
    counters: dict[str, int] = {}
    total = 0
    samples: list[dict[str, Any]] = []
    kind = "jsonl"
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            counters["json_error"] = counters.get("json_error", 0) + 1
            samples.append({"line": line_no, "category": "json_error", "preview": line[:160]})
            continue
        if not isinstance(entry, dict):
            continue
        total += 1
        if "correct" in entry or "predicted" in entry or "raw_response" in entry:
            kind = "mcq"
            predicted = str(entry.get("predicted") or entry.get("prediction") or "").strip()
            raw = str(entry.get("raw_response") or entry.get("response") or entry.get("content") or "")
            correct = bool(entry.get("correct"))
            if correct:
                counters["correct"] = counters.get("correct", 0) + 1
            else:
                counters["wrong"] = counters.get("wrong", 0) + 1
                if len(samples) < 30:
                    samples.append({"line": line_no, "category": "wrong", "predicted": predicted or "?", "preview": raw[:160]})
            if not raw.strip():
                counters["empty"] = counters.get("empty", 0) + 1
            if predicted in {"", "?"}:
                counters["unknown"] = counters.get("unknown", 0) + 1
        elif "completion" in entry or "solution" in entry or "task_id" in entry:
            kind = "evalplus"
            prompt, completion = evalplus_completion(entry)
            category = classify_evalplus_completion(completion, prompt)
            counters[category] = counters.get(category, 0) + 1
            if category != "attempt" and len(samples) < 30:
                samples.append({"line": line_no, "category": category, "task_id": entry.get("task_id"), "preview": completion[:160]})
        else:
            counters["unrecognized"] = counters.get("unrecognized", 0) + 1
    return {"path": str(path), "kind": kind, "total": total, "counts": counters, "samples": samples}


def pct(count: int, total: int) -> float:
    return 0.0 if total <= 0 else (count / total) * 100.0


def benchmark_audit(paths: list[str], fail_empty_pct: float = 2.0, fail_unknown_pct: float = 5.0, fail_pass_only_pct: float = 5.0) -> dict[str, Any]:
    files = benchmark_audit_files(paths)
    rows = [audit_jsonl_file(path) for path in files]
    failures: list[dict[str, Any]] = []
    for row in rows:
        total = int(row["total"])
        counts = row["counts"]
        empty_pct = pct(int(counts.get("empty", 0)), total)
        unknown_pct = pct(int(counts.get("unknown", 0)), total)
        pass_only_pct = pct(int(counts.get("pass_only", 0)), total)
        if empty_pct > fail_empty_pct:
            failures.append({"path": row["path"], "reason": "empty responses above threshold", "value_pct": empty_pct, "threshold_pct": fail_empty_pct})
        if unknown_pct > fail_unknown_pct:
            failures.append({"path": row["path"], "reason": "unknown MCQ predictions above threshold", "value_pct": unknown_pct, "threshold_pct": fail_unknown_pct})
        if pass_only_pct > fail_pass_only_pct:
            failures.append({"path": row["path"], "reason": "pass-only EvalPlus completions above threshold", "value_pct": pass_only_pct, "threshold_pct": fail_pass_only_pct})
        if counts.get("json_error"):
            failures.append({"path": row["path"], "reason": "JSONL parse errors", "count": counts["json_error"]})
    return {"files": rows, "failures": failures, "blocked": bool(failures)}


def benchmark_audit_markdown(report: dict[str, Any]) -> str:
    rows = []
    for row in report["files"]:
        counts = row["counts"]
        total = int(row["total"])
        rows.append(
            [
                row["path"],
                row["kind"],
                str(total),
                str(counts.get("correct", counts.get("attempt", 0))),
                f"{pct(int(counts.get('empty', 0)), total):.1f}%",
                f"{pct(int(counts.get('unknown', 0)), total):.1f}%",
                f"{pct(int(counts.get('pass_only', 0)), total):.1f}%",
            ]
        )
    parts = [
        "Benchmark audit " + ("blocked" if report["blocked"] else "passed"),
        "",
        markdown_table(["Path", "Kind", "Total", "Correct/attempt", "Empty", "Unknown", "Pass-only"], rows),
    ]
    if report["failures"]:
        failure_rows = [
            [
                item["path"],
                item["reason"],
                f"{item.get('value_pct', item.get('count', 0)):.1f}" if isinstance(item.get("value_pct", item.get("count", 0)), float) else str(item.get("count", "")),
                "-" if item.get("threshold_pct") is None else f"{float(item['threshold_pct']):.1f}%",
            ]
            for item in report["failures"]
        ]
        parts.extend(["", "## Failures", "", markdown_table(["Path", "Reason", "Value", "Threshold"], failure_rows)])
    return "\n".join(parts) + "\n"


def benchmark_audit_cmd(args: argparse.Namespace) -> None:
    report = benchmark_audit(
        args.paths,
        fail_empty_pct=args.fail_empty_pct,
        fail_unknown_pct=args.fail_unknown_pct,
        fail_pass_only_pct=args.fail_pass_only_pct,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(benchmark_audit_markdown(report), end="")
    if report["blocked"]:
        raise SystemExit(1)


def benchmark_audit_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    paths = qs.get("path") or qs.get("paths") or []
    if not paths:
        raise ValueError("path query param required")
    try:
        fail_empty_pct = float(query_value(qs, "fail_empty_pct", 2.0))
        fail_unknown_pct = float(query_value(qs, "fail_unknown_pct", 5.0))
        fail_pass_only_pct = float(query_value(qs, "fail_pass_only_pct", 5.0))
    except ValueError as exc:
        raise ValueError("audit thresholds must be numbers") from exc
    return argparse.Namespace(
        paths=paths,
        fail_empty_pct=fail_empty_pct,
        fail_unknown_pct=fail_unknown_pct,
        fail_pass_only_pct=fail_pass_only_pct,
    )


PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)")


def classify_ablation_pct(pct_delta: float) -> str:
    if pct_delta <= -5.0:
        return "demotable"
    if pct_delta <= -1.0:
        return "beneficial"
    if abs(pct_delta) < 1.0:
        return "tolerant"
    if pct_delta < 5.0:
        return "sensitive"
    return "critical"


def ppl_from_log(path: Path) -> float | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    matches = PPL_RE.findall(text)
    return float(matches[-1]) if matches else None


def tensor_from_layer_log(path: Path, tensor_group: str | None) -> str | None:
    layer_match = re.search(r"(?:ppl|ablation)_layer_(\d+)", path.name)
    if not layer_match:
        return None
    layer = layer_match.group(1)
    group = tensor_group
    if not group:
        group_match = re.search(r"ablation_layer_\d+\.([^/]+)\.log$", path.name)
        group = group_match.group(1) if group_match else None
    if not group:
        return None
    group = group.replace(".log", "")
    if group.startswith("blk."):
        return group if group.endswith(".weight") else f"{group}.weight"
    return f"blk.{layer}.{group}.weight" if not group.endswith(".weight") else f"blk.{layer}.{group}"


def ablation_rows_from_json(path: Path) -> tuple[float | None, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    baseline = data.get("baseline_ppl")
    tests = data.get("tests", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(tests, dict):
        return float(baseline) if baseline is not None else None, rows
    for key, value in tests.items():
        if not isinstance(value, dict) or value.get("ppl") is None:
            continue
        rows.append({"name": str(key), "tensor": str(value.get("gguf_tensor") or key), "ppl": float(value["ppl"]), "source": str(path)})
    return float(baseline) if baseline is not None else None, rows


def ablation_rows_from_logs(path: Path, tensor_group: str | None) -> list[dict[str, Any]]:
    files = sorted(path.rglob("*.log")) if path.is_dir() else [path]
    rows: list[dict[str, Any]] = []
    for file in files:
        ppl = ppl_from_log(file)
        tensor = tensor_from_layer_log(file, tensor_group)
        if ppl is None or tensor is None:
            continue
        rows.append({"name": file.stem, "tensor": tensor, "ppl": ppl, "source": str(file)})
    return rows


def analyze_ablation_input(path: Path, baseline_ppl: float | None = None, tensor_group: str | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]]
    detected_baseline: float | None = None
    if path.is_file() and path.suffix == ".json":
        detected_baseline, rows = ablation_rows_from_json(path)
    else:
        rows = ablation_rows_from_logs(path, tensor_group)
    baseline = baseline_ppl if baseline_ppl is not None else detected_baseline
    if baseline is None:
        raise SystemExit("baseline PPL is required for log input; pass --baseline-ppl")
    analyzed: list[dict[str, Any]] = []
    for row in rows:
        delta = row["ppl"] - baseline
        pct_delta = (delta / baseline) * 100.0 if baseline else 0.0
        analyzed.append({**row, "baseline_ppl": baseline, "delta": delta, "pct_delta": pct_delta, "classification": classify_ablation_pct(pct_delta)})
    analyzed.sort(key=lambda row: (row["classification"], row["pct_delta"], row["tensor"]))
    counts: dict[str, int] = {}
    for row in analyzed:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    return {"baseline_ppl": baseline, "count": len(analyzed), "class_counts": counts, "rows": analyzed}


def ablation_analyze_text(report: dict[str, Any]) -> str:
    lines = [f"Baseline PPL: {report['baseline_ppl']}", f"Ablation tests: {report['count']}", ""]
    for name in ["demotable", "beneficial", "tolerant", "sensitive", "critical"]:
        if report["class_counts"].get(name):
            lines.append(f"{name}: {report['class_counts'][name]}")
    lines.extend(["", "classification  ppl        delta      pct       tensor", "-" * 78])
    for row in report["rows"]:
        lines.append(
            f"{row['classification']:<14} {row['ppl']:<10.4f} {row['delta']:+10.4f} "
            f"{row['pct_delta']:+.2f}%   {row['tensor']}"
        )
    return "\n".join(lines) + "\n"


def ablation_analyze_cmd(args: argparse.Namespace) -> None:
    report = analyze_ablation_input(Path(args.input), baseline_ppl=args.baseline_ppl, tensor_group=args.tensor_group)
    override_classes = {item.strip() for item in args.override_classes.split(",") if item.strip()}
    overrides = [row for row in report["rows"] if row["classification"] in override_classes]
    if args.output:
        lines = [tensor_type_line(row["tensor"], args.target_type) for row in overrides]
        Path(args.output).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if getattr(args, "tensor_output", None):
        lines = [str(row["tensor"]) for row in overrides]
        Path(args.tensor_output).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(ablation_analyze_text(report), end="")
    if args.output:
        print(f"\nWrote {len(overrides)} overrides to {args.output}")
    if getattr(args, "tensor_output", None):
        print(f"\nWrote {len(overrides)} tensor names to {args.tensor_output}")


def write_report_files(run_dir: Path, report: dict[str, Any], formats: list[str]) -> list[Path]:
    written: list[Path] = []
    if "json" in formats:
        path = run_dir / SUMMARY_JSON_FILES[0]
        atomic_write_json(path, report)
        written.append(path)
    if "md" in formats:
        path = run_dir / SUMMARY_MD_FILES[0]
        lines = [
            f"# Cerebellum report: {report['run_id']}",
            "",
            f"- Model: {report.get('model_family')}/{report.get('model_name')}",
            f"- Source: {report.get('source_name')}",
            f"- Status: {report.get('status')}",
            f"- Current PPL: {report.get('current_ppl')}",
            f"- PPL profile: {report.get('ppl_profile')}",
            f"- Corpus: {report.get('corpus')}",
            f"- Locked tensors: {report.get('locked_count')}",
            f"- Candidate tests: {report.get('candidate_count')}",
            f"- Quant time: {fmt_seconds(report.get('quant_seconds'))}",
            f"- PPL time: {fmt_seconds(report.get('ppl_seconds'))}",
            "",
            "## Component summary",
            "",
            "| Component | Tests | Best delta | Worst delta |",
            "|---|---:|---:|---:|",
        ]
        for row in report["component_summary"]:
            lines.append(f"| {row['component']} | {row['tests']} | {row['best_delta']} | {row['worst_delta']} |")
        lines.extend(["", "## Recent decisions", ""])
        for row in report["recent_decisions"]:
            lines.append(f"- `{row.get('winner')}` PPL `{row.get('ppl')}`: `{row.get('tensor')}`")
        path.write_text("\n".join(lines) + "\n")
        written.append(path)
    if "csv" in formats:
        path = run_dir / DECISION_CSV_FILES[0]
        rows = report["recent_decisions"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["tensor", "winner", "ppl", "baseline_ppl", "finished_at", "reason"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        written.append(path)
    if "infographic" in formats:
        path = run_dir / INFOGRAPHIC_FILES[0]
        infographic = {
            "title": f"{report.get('model_name')} Cerebellum run",
            "subtitle": f"{report.get('locked_count')} tensors locked, PPL {report.get('current_ppl')}",
            "cards": [
                {"label": "Locked tensors", "value": report.get("locked_count")},
                {"label": "Candidates tested", "value": report.get("candidate_count")},
                {"label": "Quant time", "value": fmt_seconds(report.get("quant_seconds"))},
                {"label": "PPL time", "value": fmt_seconds(report.get("ppl_seconds"))},
            ],
            "component_summary": report["component_summary"],
            "recent_decisions": report["recent_decisions"],
        }
        atomic_write_json(path, infographic)
        written.append(path)
    return written


def report_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    report = build_report(run_dir)
    formats = [part.strip() for part in args.format.split(",") if part.strip()]
    written = write_report_files(run_dir, report, formats)
    if args.json:
        print(json.dumps({"report": report, "written": [str(p) for p in written]}, indent=2, sort_keys=True))
        return
    print(f"report for {report['run_id']}")
    for path in written:
        print(f"  wrote {path}")


def export_payload(run_dir: Path, kind: str) -> dict[str, Any]:
    report = build_report(run_dir)
    if kind == "infographic":
        return {
            "schema": "cerebellum.infographic.v1",
            "report": report,
            "prompt": (
                "Create a clean technical infographic from this Cerebellum quantization run. "
                "Show model, PPL, locked tensors, candidate count, component deltas, and recent decisions."
            ),
        }
    if kind == "ai":
        return {
            "schema": "cerebellum.ai_context.v1",
            "instruction": "Use this run data to compare quantization decisions, summarize findings, or draft model-card evidence.",
            "report": report,
        }
    return report


def export_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = export_payload(run_dir, args.kind)
    if args.output:
        path = Path(args.output)
        atomic_write_json(path, payload)
        print(path)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def provenance_cmd(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {}
    gguf = Path(args.gguf) if args.gguf else None
    if args.run_dir:
        payload["generated_metadata"] = cerebellum_metadata_block(Path(args.run_dir), gguf, args.hash_files, private=args.private)
    if gguf:
        payload["gguf"] = str(gguf)
        payload["existing_cerebellum_metadata"] = inspect_gguf_metadata(gguf)
        payload["has_cerebellum_metadata"] = bool(payload["existing_cerebellum_metadata"])
    if not payload:
        raise SystemExit("provenance requires --gguf, --run-dir, or both")
    if args.format == "env":
        for key, value in (payload.get("generated_metadata") or payload.get("existing_cerebellum_metadata") or {}).items():
            print(f"{key}={value}")
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_model_card(run_dir: Path, output_dir: Path, metadata: dict[str, Any], repo_name: str | None = None) -> Path:
    report = build_report(run_dir)
    title = repo_name or f"{report.get('model_name')} Cerebellum GGUF"
    provenance_lines = [
        f"- Model: `{metadata.get('cerebellum.model_family')}/{metadata.get('cerebellum.model_name')}`",
        f"- Source: `{metadata.get('cerebellum.source_name')}`",
        f"- PPL profile: `{metadata.get('cerebellum.ppl_profile')}`",
        f"- Locked tensors: `{metadata.get('cerebellum.locked_count')}`",
    ]
    if metadata.get("cerebellum.run_id"):
        provenance_lines.append(f"- Run ID: `{metadata.get('cerebellum.run_id')}`")
    if metadata.get("cerebellum.current_ppl"):
        provenance_lines.append(f"- Current PPL: `{metadata.get('cerebellum.current_ppl')}`")
    if metadata.get("cerebellum.candidate_count"):
        provenance_lines.append(f"- Candidate tests: `{metadata.get('cerebellum.candidate_count')}`")
    lines = [
        f"# {title}",
        "",
        "This GGUF was produced with **Cerebellum**, a resource-aware mixed-precision quantization workflow.",
        "",
        "## Cerebellum provenance",
        "",
        *provenance_lines,
        "",
        "## Metadata keys",
        "",
        "The final GGUF should include visible `cerebellum.*` metadata keys for attribution and auditability.",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Notes",
        "",
        "- This metadata is transparent provenance, not a hidden watermark.",
        "- If these keys are missing from redistributed copies, the provenance was stripped.",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "MODEL_CARD_CEREBELLUM.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def metadata_tool_path(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return shutil.which("gguf-set-metadata") or shutil.which("llama-gguf-set-metadata")


def inject_metadata(tool: str, gguf: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for key, value in metadata.items():
        cmd = [tool, str(gguf), key, str(value)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        results.append(
            {
                "key": key,
                "returncode": proc.returncode,
                "output": ((proc.stdout or "") + (proc.stderr or ""))[-1000:],
            }
        )
        if proc.returncode != 0:
            break
    return results


def finalize_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    gguf = Path(args.gguf) if args.gguf else None
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "finalize"
    report = build_report(run_dir)
    write_report_files(run_dir, report, ["json", "md", "csv", "infographic"])
    metadata = cerebellum_metadata_block(run_dir, gguf, args.hash_files, private=args.private)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "cerebellum_gguf_metadata.json"
    atomic_write_json(metadata_path, metadata)
    env_path = output_dir / "cerebellum_gguf_metadata.env"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in metadata.items()) + "\n", encoding="utf-8")
    card_path = write_model_card(run_dir, output_dir, metadata, args.repo_name)
    existing = inspect_gguf_metadata(gguf) if gguf else {}
    injection: list[dict[str, Any]] = []
    tool = metadata_tool_path(args.metadata_tool)
    if args.inject:
        if not gguf:
            raise SystemExit("--inject requires --gguf")
        if not tool:
            raise SystemExit("--inject requires gguf-set-metadata on PATH or --metadata-tool")
        injection = inject_metadata(tool, gguf, metadata)
    payload = {
        "run_dir": str(run_dir),
        "gguf": str(gguf) if gguf else None,
        "metadata": metadata,
        "existing_cerebellum_metadata": existing,
        "written": [str(metadata_path), str(env_path), str(card_path)],
        "metadata_tool": tool,
        "injection": injection,
        "mode": "private" if args.private else "public",
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("Cerebellum finalize")
    for path in payload["written"]:
        print(f"  wrote {path}")
    if gguf:
        print(f"  existing metadata keys: {len(existing)}")
    if args.inject:
        failed = [row for row in injection if row["returncode"] != 0]
        print(f"  injected keys: {len(injection) - len(failed)}/{len(metadata)}")
        if failed:
            print(f"  failed at {failed[0]['key']}: {failed[0]['output']}")
    elif gguf:
        print("  metadata not injected; rerun with --inject to tag the GGUF")


def public_release_label(report: dict[str, Any]) -> str:
    label = report.get("model_name") or report.get("model") or "cerebellum-release"
    return slug(str(label)).lower() or "cerebellum-release"


def public_package_report(report: dict[str, Any]) -> dict[str, Any]:
    family = report.get("model_family")
    name = report.get("model_name") or report.get("model") or "unknown-model"
    model = f"{family}/{name}" if family else str(name)
    return {
        "model": model,
        "status": report.get("status"),
        "release_label": public_release_label(report),
    }


def write_public_package_sidecars(run_dir: Path, report: dict[str, Any]) -> list[Path]:
    output_dir = run_dir / "public_package"
    output_dir.mkdir(parents=True, exist_ok=True)
    public_report = public_package_report(report)
    metadata_path = output_dir / "cerebellum_public_metadata.json"
    atomic_write_json(
        metadata_path,
        {
            "schema": "cerebellum.public_metadata.v1",
            "tool": "Cerebellum",
            **public_report,
            "notes": [
                "Public metadata intentionally excludes run IDs, raw PPL, tensor maps, event logs, local paths, and selection internals.",
                "Private provenance and factory artifacts belong in cerebellum-dev only.",
            ],
        },
    )
    card_path = output_dir / "MODEL_CARD_CEREBELLUM.md"
    card_path.write_text(
        "\n".join(
            [
                f"# {public_report['model']} Cerebellum GGUF",
                "",
                "This GGUF was produced with **Cerebellum**, a resource-aware mixed-precision quantization workflow.",
                "",
                "## Public Provenance",
                "",
                f"- Model: `{public_report['model']}`",
                f"- Status: `{public_report['status']}`",
                f"- Release label: `{public_report['release_label']}`",
                "",
                "Detailed benchmark artifacts, model file hashes, and runtime flags should be attached separately for published releases.",
                "",
                "Private run IDs, raw ablation logs, tensor maps, candidate PPL traces, local paths, and selection internals are intentionally omitted.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [metadata_path, card_path]


def package_files(run_dir: Path, private: bool = False) -> list[Path]:
    report = build_report(run_dir)
    public_candidates = [
        *write_public_package_sidecars(run_dir, report),
    ]
    private_candidates = [
        run_dir / "manifest.json",
        run_dir / "state.json",
        first_existing(run_dir, EVENT_FILES),
        first_existing(run_dir, CANDIDATE_FILES),
        first_existing(run_dir, SUMMARY_JSON_FILES),
        first_existing(run_dir, SUMMARY_MD_FILES),
        first_existing(run_dir, DECISION_CSV_FILES),
        first_existing(run_dir, INFOGRAPHIC_FILES),
        first_existing(run_dir, BEST_TYPES_FILES),
        *public_candidates,
    ]
    candidates = private_candidates if private else public_candidates
    return [path for path in candidates if path.exists()]


def package_manifest(run_dir: Path, private: bool = False) -> dict[str, Any]:
    report = build_report(run_dir)
    files = []
    for path in package_files(run_dir, private=private):
        size = path_size(path)
        files.append(
            {
                "name": path.name,
                "size_bytes": size,
                "sha256": sha256_file(path) if size < 128 * 1024 * 1024 else None,
                "hf_path": remote_sidecar_path(report, path, private=private),
            }
        )
        if private:
            files[-1]["path"] = str(path)
    payload = {
        "schema": "cerebellum.package.v1",
        "mode": "private" if private else "public",
        "model": f"{report.get('model_family')}/{report.get('model_name')}",
        "status": report.get("status"),
        "files": files,
        "notes": [
            (
                "Public mode writes sanitized release sidecars and excludes raw factory artifacts. "
                "Use --private only for private dev uploads."
            ),
            "If GGUF metadata is stripped, compare public sidecar provenance and model-card text.",
        ],
    }
    if private:
        payload["run_dir"] = str(run_dir)
        payload["run_id"] = report["run_id"]
        payload["ppl_profile"] = report.get("ppl_profile")
    else:
        payload.update(public_package_report(report))
    return payload


def package_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    payload = package_manifest(run_dir, private=args.private)
    files = payload["files"]
    output = Path(args.output) if args.output else run_dir / "cerebellum_package_manifest.json"
    atomic_write_json(output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"package manifest: {output}")
    for item in files:
        print(f"  {item['name']:<36} {fmt_bytes(item['size_bytes']):>10} -> {item['hf_path']}")


PUBLIC_AUDIT_PATH_PATTERNS = [
    (re.compile(r"(^|/)scripts/"), "private script path"),
    (re.compile(r"(^|/)tests/"), "test/factory path ignored for public origin"),
    (re.compile(r"(^|/)(osmosis|cerebellum)/dashboard/"), "private dashboard path"),
    (re.compile(r"(^|/)docs/devlog/"), "private devlog path"),
    (re.compile(r"devlog", re.IGNORECASE), "devlog content/path"),
    (re.compile(r"ablation", re.IGNORECASE), "raw ablation artifact/path"),
    (re.compile(r"tensor(_|-)?types|candidate|decision|events\.jsonl|state\.json", re.IGNORECASE), "raw factory state artifact/path"),
    (re.compile(r"\.env$|\.secret$", re.IGNORECASE), "secret file path"),
    (re.compile(r"\.(gguf|dat)$", re.IGNORECASE), "large model/imatrix binary path"),
]

PUBLIC_AUDIT_CONTENT_PATTERNS = [
    (re.compile(r"(?i)(HF_TOKEN|HUGGINGFACE_HUB_TOKEN|GITHUB_TOKEN|GH_TOKEN|OPENAI_API_KEY)\s*="), "credential environment assignment"),
    (re.compile(r"\b(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"), "token-like secret"),
    (re.compile(r"/var/home/deucebucket|/home/deucebucket|/var/home/[^\s'\"`]+"), "absolute local user path"),
    (re.compile(r"ai-drive|cerebellum-runs|/games/"), "machine-specific storage path"),
    (re.compile(r"tensor-selection|selection heuristics|raw ablation|streaming quant internals", re.IGNORECASE), "private method detail"),
]

PUBLIC_EXPORT_ALLOWED_PATTERNS = [
    re.compile(r"^README(\.[A-Za-z0-9_-]+)?\.md$"),
    re.compile(r"^LICENSE(\.[A-Za-z0-9_-]+)?$"),
    re.compile(r"^docs/(?!devlog/).+"),
    re.compile(r"^benchmark_results/.+"),
    re.compile(r"^[^/]+/benchmark_results/.+"),
    re.compile(r"^[^/]+/README(\.[A-Za-z0-9_-]+)?\.md$"),
    re.compile(r"^[^/]+/MODEL_CARD(\.[A-Za-z0-9_-]+)?\.md$"),
    re.compile(r"^[^/]+/model_card(\.[A-Za-z0-9_-]+)?\.md$"),
    re.compile(r"^spaces/[^/]+/(README\.md|requirements\.txt)$"),
    re.compile(r"^[^/]+\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
]

PUBLIC_EXPORT_ALLOWED_SUFFIXES = {".md", ".json", ".jsonl", ".csv", ".txt", ".png", ".jpg", ".jpeg", ".webp"}


def tracked_files() -> list[Path]:
    try:
        result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    except Exception:
        return []
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def expand_audit_paths(paths: list[str]) -> list[Path]:
    if not paths:
        return [path for path in tracked_files() if path.is_file()]
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and ".git" not in p.parts)
        elif path.is_file():
            files.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def public_audit(paths: list[str] | None = None, max_bytes: int = 512_000) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    files = expand_audit_paths(paths or [])
    for path in files:
        text_path = path.as_posix()
        for pattern, reason in PUBLIC_AUDIT_PATH_PATTERNS:
            if pattern.search(text_path):
                findings.append({"path": text_path, "kind": "path", "reason": reason, "severity": "blocker"})
        try:
            data = path.read_bytes()[:max_bytes]
        except Exception:
            continue
        if b"\x00" in data[:4096]:
            continue
        text = data.decode("utf-8", errors="ignore")
        for pattern, reason in PUBLIC_AUDIT_CONTENT_PATTERNS:
            match = pattern.search(text)
            if match:
                line_no = text[: match.start()].count("\n") + 1
                findings.append(
                    {
                        "path": text_path,
                        "kind": "content",
                        "line": line_no,
                        "reason": reason,
                        "severity": "blocker",
                    }
                )
    return {
        "files_scanned": len(files),
        "findings": findings,
        "blocked": bool(findings),
        "scope": "tracked files" if not paths else "explicit paths",
    }


def public_audit_markdown(report: dict[str, Any]) -> str:
    rows = [
        [
            item["severity"],
            item["kind"],
            item["path"],
            "-" if item.get("line") is None else str(item["line"]),
            item["reason"],
        ]
        for item in report["findings"]
    ]
    if not rows:
        return f"Public audit passed: scanned {report['files_scanned']} files.\n"
    return "\n".join(
        [
            f"Public audit blocked: scanned {report['files_scanned']} files, found {len(rows)} risks.",
            "",
            markdown_table(["Severity", "Kind", "Path", "Line", "Reason"], rows),
        ]
    ) + "\n"


def public_audit_cmd(args: argparse.Namespace) -> None:
    report = public_audit(args.paths, max_bytes=args.max_bytes)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(public_audit_markdown(report), end="")
    if report["blocked"]:
        raise SystemExit(1)


def git_history_paths(root: Path, refs: list[str] | None = None) -> list[str]:
    ref_args = refs or ["--all"]
    cmd = ["git", "-C", str(root), "log", "--name-only", "--format=", *ref_args]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"git history scan failed: {exc.stderr.strip() or exc}") from exc
    paths = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return sorted(paths)


def public_history_audit(root: Path, refs: list[str] | None = None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    paths = git_history_paths(root, refs=refs)
    risky: list[dict[str, Any]] = []
    for path_text in paths:
        reasons = []
        for pattern, reason in PUBLIC_AUDIT_PATH_PATTERNS:
            if pattern.search(path_text):
                reasons.append(reason)
        for reason in artifact_public_risks(path_text):
            if reason not in reasons:
                reasons.append(reason)
        if reasons:
            risky.append({"path": path_text, "reasons": reasons, "severity": "blocker"})
    filter_paths = sorted({row["path"] for row in risky})
    filter_repo_argv = ["git", "filter-repo", "--force", "--invert-paths"]
    for path_text in filter_paths:
        filter_repo_argv.extend(["--path", path_text])
    return {
        "schema": "cerebellum.public_history_audit.v1",
        "root": str(root),
        "refs": refs or ["--all"],
        "paths_scanned": len(paths),
        "risky_paths": risky,
        "blocked": bool(risky),
        "filter_paths": filter_paths,
        "filter_repo_argv": filter_repo_argv if filter_paths else [],
        "filter_repo_command": shell_join(filter_repo_argv) if filter_paths else "",
        "notes": [
            "Read-only history audit; this command does not rewrite history.",
            "Run filter-repo only in a reviewed public clone/branch, then force-push the public remote intentionally.",
            "Assume anything previously public may have been cached or cloned; rotate any exposed credentials.",
            "This path scan does not inspect historical blob contents for secrets.",
        ],
    }


def public_history_audit_markdown(report: dict[str, Any]) -> str:
    if not report["blocked"]:
        return f"Public history audit passed: scanned {report['paths_scanned']} historical paths.\n"
    rows = [[row["severity"], row["path"], "; ".join(row["reasons"])] for row in report["risky_paths"]]
    parts = [
        f"Public history audit blocked: scanned {report['paths_scanned']} historical paths, found {len(rows)} risky paths.",
        "",
        markdown_table(["Severity", "Path", "Reasons"], rows),
        "",
        "## Filter-Repo Plan",
        "",
        f"`{report['filter_repo_command']}`",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in report["notes"]],
    ]
    return "\n".join(parts) + "\n"


def public_history_audit_cmd(args: argparse.Namespace) -> None:
    report = public_history_audit(Path(args.root), refs=args.ref)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(public_history_audit_markdown(report), end="")
    if report["blocked"]:
        raise SystemExit(1)


ARTIFACT_TYPE_PATTERNS = [
    ("gguf", re.compile(r"\.gguf$", re.IGNORECASE)),
    ("imatrix", re.compile(r"(imatrix).*\.(dat|gguf)$", re.IGNORECASE)),
    ("benchmark", re.compile(r"benchmark_results|benchmark|humaneval|evalplus|arc|hellaswag|mmlu|gpqa|hle|livecodebench", re.IGNORECASE)),
    ("ablation", re.compile(r"ablation|candidate|measurement|ppl|perplexity|rowblock|brain_scan", re.IGNORECASE)),
    ("tensor_map", re.compile(r"tensor(_|-)?types|override|final_types|locked|types\.txt", re.IGNORECASE)),
    ("dev_note", re.compile(r"README|DEVLOG|SPEC|ISSUE|FULL_EXPERIMENT|research|findings|COMPARISON|DESIGN", re.IGNORECASE)),
    ("script_tool", re.compile(r"\.(py|sh)$|(^|/)(scripts|tools|tool_tests)/", re.IGNORECASE)),
    ("checkpoint", re.compile(r"\.(pt|pth|safetensors)$|checkpoint", re.IGNORECASE)),
    ("database", re.compile(r"\.(sqlite|sqlite3|db)$|(^|/)db/", re.IGNORECASE)),
    ("log", re.compile(r"\.(log|out|pid)$|server", re.IGNORECASE)),
    ("cache", re.compile(r"__pycache__|\.cache|_cache|unsloth_compiled_cache|\.pytest_cache|\.ruff_cache", re.IGNORECASE)),
    ("image", re.compile(r"\.(png|jpg|jpeg|webp)$", re.IGNORECASE)),
]

ARTIFACT_RISK_PATTERNS = [
    (re.compile(r"(^|/)cerebellum-dev/|(^|/)scripts/|(^|/)tests/|(^|/)db/|\.opencode/", re.IGNORECASE), "private factory/control-plane path"),
    (re.compile(r"devlog|SPEC_|ISSUE_|FULL_EXPERIMENT|research_log|DEVELOPER_LOG", re.IGNORECASE), "private dev note or strategy"),
    (re.compile(r"ablation|candidate|events\.jsonl|state\.json|tensor(_|-)?types|override|rowblock", re.IGNORECASE), "raw selection or ablation artifact"),
    (re.compile(r"agent_bench|harm_check|steering|user_bench_results|refusal", re.IGNORECASE), "private behavior/test artifact"),
    (re.compile(r"\.cache|__pycache__|\.sqlite3|\.db|\.pid|\.log$|server", re.IGNORECASE), "local state, cache, or log"),
    (re.compile(r"\.gguf$|imatrix.*\.(dat|gguf)$", re.IGNORECASE), "large binary requiring provenance before release"),
]

CLEANUP_CANDIDATE_PATTERNS = [
    (re.compile(r"__pycache__|\.pytest_cache|\.ruff_cache|unsloth_compiled_cache|\.cache", re.IGNORECASE), "cache"),
    (re.compile(r"\.pid$|watch_\d+|taildrop", re.IGNORECASE), "local transient"),
    (re.compile(r"\.log$|\.out$", re.IGNORECASE), "log; preserve only if tied to a canonical result"),
]


def artifact_file_type(path_text: str) -> str:
    for name, pattern in ARTIFACT_TYPE_PATTERNS:
        if pattern.search(path_text):
            return name
    return "other"


def artifact_storage_category(file_type: str, path_text: str) -> str:
    if file_type in {"gguf", "imatrix"}:
        return "archive/binaries"
    if file_type == "benchmark":
        return "archive/benchmarks"
    if file_type in {"ablation", "tensor_map"}:
        return "archive/ablation"
    if file_type in {"dev_note", "script_tool"} or "cerebellum-dev/" in path_text:
        return "archive/devnotes"
    if file_type in {"cache", "log"}:
        return "scratch/cache"
    if file_type == "image":
        return "public-candidates"
    return "archive/legacy-models"


def artifact_public_risks(path_text: str) -> list[str]:
    return [reason for pattern, reason in ARTIFACT_RISK_PATTERNS if pattern.search(path_text)]


def artifact_cleanup_reason(path_text: str) -> str | None:
    for pattern, reason in CLEANUP_CANDIDATE_PATTERNS:
        if pattern.search(path_text):
            return reason
    return None


def artifact_bucket_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else "."


def validate_artifact_inventory_root(root: Path, *, allow_broad: bool = False) -> Path:
    root = root.expanduser().resolve()
    if not root.exists():
        raise ValueError(f"root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"root must be a directory: {root}")
    if root == Path(root.anchor) and not allow_broad:
        raise ValueError("filesystem root inventory requires allow_broad=true")
    return root


def artifact_inventory(root: Path, top: int = 25) -> dict[str, Any]:
    root = validate_artifact_inventory_root(root, allow_broad=True)
    buckets: dict[str, dict[str, Any]] = {}
    large_files: list[dict[str, Any]] = []
    cleanup_candidates: list[dict[str, Any]] = []
    public_risk_examples: list[dict[str, Any]] = []
    file_manifest: list[dict[str, Any]] = []
    totals = {
        "files": 0,
        "bytes": 0,
        "public_risk_files": 0,
        "cleanup_candidate_files": 0,
    }
    type_counts: dict[str, int] = {}
    storage_counts: dict[str, int] = {}
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rel_text = path.relative_to(root).as_posix()
        file_type = artifact_file_type(rel_text)
        storage = artifact_storage_category(file_type, rel_text)
        risks = artifact_public_risks(rel_text)
        cleanup_reason = artifact_cleanup_reason(rel_text)
        file_manifest.append(
            {
                "path": rel_text,
                "size_bytes": stat.st_size,
                "type": file_type,
                "storage_category": storage,
                "public_risks": risks,
                "cleanup_reason": cleanup_reason,
            }
        )
        bucket_name = artifact_bucket_name(root, path)
        bucket = buckets.setdefault(
            bucket_name,
            {
                "path": bucket_name,
                "files": 0,
                "bytes": 0,
                "types": {},
                "storage_categories": {},
                "public_risk_files": 0,
                "cleanup_candidate_files": 0,
                "example_keep": [],
                "example_risks": [],
            },
        )
        bucket["files"] += 1
        bucket["bytes"] += stat.st_size
        bucket["types"][file_type] = bucket["types"].get(file_type, 0) + 1
        bucket["storage_categories"][storage] = bucket["storage_categories"].get(storage, 0) + 1
        totals["files"] += 1
        totals["bytes"] += stat.st_size
        type_counts[file_type] = type_counts.get(file_type, 0) + 1
        storage_counts[storage] = storage_counts.get(storage, 0) + 1
        if risks:
            totals["public_risk_files"] += 1
            bucket["public_risk_files"] += 1
            risk_row = {"path": rel_text, "reasons": risks[:3], "storage_category": storage}
            if len(bucket["example_risks"]) < 5:
                bucket["example_risks"].append(risk_row)
            if len(public_risk_examples) < top:
                public_risk_examples.append(risk_row)
        if cleanup_reason:
            totals["cleanup_candidate_files"] += 1
            bucket["cleanup_candidate_files"] += 1
            if len(cleanup_candidates) < top:
                cleanup_candidates.append({"path": rel_text, "reason": cleanup_reason, "size_bytes": stat.st_size})
        if file_type in {"benchmark", "tensor_map", "ablation", "dev_note", "gguf", "imatrix"} and len(bucket["example_keep"]) < 5:
            bucket["example_keep"].append({"path": rel_text, "type": file_type, "storage_category": storage})
        if stat.st_size >= 100 * 1024 * 1024:
            large_files.append({"path": rel_text, "size_bytes": stat.st_size, "type": file_type, "storage_category": storage, "public_risks": risks})
    bucket_rows = sorted(buckets.values(), key=lambda row: row["bytes"], reverse=True)
    large_files.sort(key=lambda row: row["size_bytes"], reverse=True)
    file_manifest.sort(key=lambda row: row["path"])
    return {
        "schema": "cerebellum.artifact_inventory.v1",
        "root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": "preservation-first; this report never approves deletion",
        "totals": totals,
        "type_counts": dict(sorted(type_counts.items())),
        "storage_counts": dict(sorted(storage_counts.items())),
        "files": file_manifest,
        "buckets": bucket_rows,
        "largest_buckets": bucket_rows[:top],
        "large_files": large_files[:top],
        "public_risk_examples": public_risk_examples[:top],
        "cleanup_candidates": cleanup_candidates[:top],
        "notes": [
            "Raw ablation data, tensor maps, devlogs, scripts, dashboards, local paths, caches, and logs stay private.",
            "Public candidates still require public-audit and human review before origin/HF publication.",
            "Cleanup candidates require a verified backup and separate cleanup plan before deletion.",
        ],
    }


def artifact_inventory_markdown(report: dict[str, Any]) -> str:
    bucket_rows = [
        [
            row["path"],
            str(row["files"]),
            fmt_bytes(row["bytes"]),
            str(row["public_risk_files"]),
            str(row["cleanup_candidate_files"]),
            ", ".join(f"{key}:{value}" for key, value in sorted(row["types"].items())[:5]),
        ]
        for row in report["largest_buckets"]
    ]
    large_rows = [
        [row["path"], fmt_bytes(row["size_bytes"]), row["type"], row["storage_category"]]
        for row in report["large_files"]
    ]
    risk_rows = [
        [row["path"], row["storage_category"], "; ".join(row["reasons"])]
        for row in report["public_risk_examples"]
    ]
    cleanup_rows = [
        [row["path"], fmt_bytes(row["size_bytes"]), row["reason"]]
        for row in report["cleanup_candidates"]
    ]
    parts = [
        "# Cerebellum Artifact Inventory",
        "",
        f"root: `{report['root']}`",
        f"generated: `{report['generated_at']}`",
        f"policy: `{report['policy']}`",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["files", str(report["totals"]["files"])],
                ["size", fmt_bytes(report["totals"]["bytes"])],
                ["public-risk files", str(report["totals"]["public_risk_files"])],
                ["cleanup-candidate files", str(report["totals"]["cleanup_candidate_files"])],
            ],
        ),
        "",
        "## Largest Buckets",
        "",
        markdown_table(["Path", "Files", "Size", "Risk", "Cleanup", "Top Types"], bucket_rows),
    ]
    if large_rows:
        parts.extend(["", "## Large Files", "", markdown_table(["Path", "Size", "Type", "Storage"], large_rows)])
    if risk_rows:
        parts.extend(["", "## Public-Risk Examples", "", markdown_table(["Path", "Storage", "Reasons"], risk_rows)])
    if cleanup_rows:
        parts.extend(["", "## Cleanup Candidates", "", markdown_table(["Path", "Size", "Reason"], cleanup_rows)])
    parts.extend(["", "## Notes", "", *[f"- {note}" for note in report["notes"]]])
    return "\n".join(parts) + "\n"


def artifact_inventory_cmd(args: argparse.Namespace) -> None:
    try:
        root = validate_artifact_inventory_root(Path(args.root))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    report = artifact_inventory(root, top=max(1, args.top))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(artifact_inventory_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.output:
        print(f"artifact inventory JSON: {args.output}")
    if args.markdown:
        print(f"artifact inventory Markdown: {args.markdown}")
    if not args.output and not args.markdown:
        print(artifact_inventory_markdown(report), end="")


def artifact_inventory_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    root_value = query_value(qs, "root")
    if not root_value:
        raise ValueError("root query param required")
    try:
        top = int(query_value(qs, "top", 25))
    except ValueError as exc:
        raise ValueError("top must be an integer") from exc
    if top < 1:
        raise ValueError("top must be at least 1")
    if top > 100:
        raise ValueError("top must be 100 or lower")
    root = validate_artifact_inventory_root(Path(root_value), allow_broad=query_bool(qs, "allow_broad"))
    return argparse.Namespace(root=str(root), top=top)


HISTORY_SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".bench-venv",
    ".venv",
    "venv",
    "site-packages",
    "dist-packages",
    "node_modules",
    "unsloth_compiled_cache",
}
HISTORY_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", re.IGNORECASE)
HISTORY_INLINE_PPL_RE = re.compile(r"\b(?:PPL|perplexity)\b[^0-9]{0,32}([0-9][0-9,.]*\.?[0-9]*)", re.IGNORECASE)
HISTORY_BENCH_RE = re.compile(
    r"\b(HumanEval|EvalPlus|ARC|HellaSwag|MMLU(?:-Redux|-Pro)?|GPQA(?:-Diamond)?|MMMLU|HLE|LiveCodeBench|AIME|IFEval|BFCL|SWE-bench)\b[^0-9%]{0,56}([0-9]+(?:\.[0-9]+)?)%?",
    re.IGNORECASE,
)
HISTORY_CHAT_TOOL_MARKERS = (".codex", ".claude", ".opencode", "gemini")
HISTORY_MODEL_TERM_RE = re.compile(
    r"(gemma[-_ ]?4[-_ ]?(?:e2b|e4b|12b|26b|codex)|qwen(?:3(?:\.5|\.6)?|35|36)?[-_ ]?(?:9b|14b|27b|30b|32b|122b)?|granite[-_ ]?4(?:\.1)?[-_ ]?(?:h[-_ ]?small|30b)?|glm[-_ ]?5(?:\.1)?)",
    re.IGNORECASE,
)


def history_default_roots() -> list[Path]:
    roots = [Path.cwd()]
    game_runs = Path("/var/home/deucebucket/games/cerebellum-runs")
    if game_runs.exists():
        roots.append(game_runs)
    return roots


def history_known_chat_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".codex" / "sessions",
        home / ".codex",
        home / ".claude" / "projects",
        home / ".opencode",
        home / ".local" / "share" / "opencode",
        home / ".config" / "opencode",
        home / ".gemini",
        home / ".config" / "gemini",
    ]
    return [path for path in candidates if path.exists()]


def history_path_skipped(path: Path) -> bool:
    return any(part in HISTORY_SKIP_PARTS for part in path.parts)


def history_model_id_from_path(path: Path) -> str:
    parts = path.parts
    if "models" in parts:
        idx = parts.index("models")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "families" in parts:
        idx = parts.index("families")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}/{parts[idx + 2]}"
    for part in reversed(parts):
        low = part.lower()
        if any(token in low for token in ("gemma", "qwen", "granite", "glm", "cerebellum")) and part not in {"benchmark_results", "benchmarks"}:
            return part
    return path.parent.name or path.name


def history_model_ids_from_text(text: str) -> list[str]:
    seen: list[str] = []
    for match in HISTORY_MODEL_TERM_RE.finditer(text):
        model = slug(match.group(1).replace("_", "-").replace(" ", "-")).lower()
        if model and model not in seen:
            seen.append(model)
    return seen[:12]


def history_is_chat_file(path: Path) -> bool:
    text = str(path).lower()
    return any(marker in text for marker in HISTORY_CHAT_TOOL_MARKERS) and path.suffix.lower() in {"", ".json", ".jsonl", ".md", ".txt", ".log"}


def history_model_bucket(models: dict[str, dict[str, Any]], model_id: str) -> dict[str, Any]:
    return models.setdefault(
        model_id,
        {
            "model": model_id,
            "paths": [],
            "methods": [],
            "scripts": [],
            "manifests": [],
            "ppl_results": [],
            "benchmark_results": [],
            "notes": [],
        },
    )


def history_add_unique(row: dict[str, Any], key: str, value: Any, limit: int = 100) -> None:
    if value in (None, "", []):
        return
    items = row.setdefault(key, [])
    if value not in items and len(items) < limit:
        items.append(value)


def history_float(text: str) -> float:
    return float(text.replace(",", "").rstrip(".,;:"))


def history_manifest_summary(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "run_id": data.get("run_id"),
        "profile": data.get("ppl_profile") or data.get("profile"),
        "corpus": data.get("corpus"),
        "chunks": data.get("chunks"),
        "ctx_size": data.get("ctx_size"),
        "base_type": data.get("base_type"),
        "start_type": data.get("start_type"),
        "levels": data.get("levels"),
        "commit_locks": data.get("commit_locks"),
        "current_ppl": data.get("current_ppl") or data.get("baseline_ppl"),
    }


def history_ppl_results_from_log(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    chunks = None
    ctx = None
    chunk_match = re.search(r"calculating perplexity over\s+([0-9]+)\s+chunks,\s*n_ctx=([0-9]+)", text)
    if chunk_match:
        chunks = int(chunk_match.group(1))
        ctx = int(chunk_match.group(2))
    return [
        {"path": str(path), "ppl": float(match.group(1)), "error": float(match.group(2)), "chunks": chunks, "ctx_size": ctx}
        for match in HISTORY_PPL_RE.finditer(text)
    ][-3:]


def history_extract_doc_signals(path: Path, text: str) -> dict[str, list[dict[str, Any]]]:
    methods: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(token in low for token in ("method", "workflow", "baseline", "ablation", "imatrix", "wikitext", "wiki", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "f16", "bf16")):
            if len(methods) < 30:
                methods.append({"path": str(path), "line": lineno, "text": stripped[:280]})
        for match in HISTORY_BENCH_RE.finditer(stripped):
            if len(benchmarks) < 100:
                value = history_float(match.group(2))
                if value > 100.0:
                    continue
                benchmarks.append(
                    {
                        "path": str(path),
                        "line": lineno,
                        "benchmark": benchmark_key(match.group(1)),
                        "metric": "percent",
                        "value": value,
                        "text": stripped[:280],
                    }
                )
        if "ppl" in low or "perplexity" in low:
            match = HISTORY_INLINE_PPL_RE.search(stripped)
            if match and len(notes) < 100:
                notes.append({"path": str(path), "line": lineno, "kind": "ppl-note", "value": history_float(match.group(1)), "text": stripped[:280]})
    return {"methods": methods, "benchmarks": benchmarks, "notes": notes}


def history_extract_chat_hits(path: Path, text: str) -> list[dict[str, Any]]:
    terms = ["cerebellum", "/var/home/deucebucket/ai-drive/cerebellum", "gemma", "qwen", "granite", "gguf", "imatrix", "wikitext"]
    hits: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if not any(term in low for term in terms):
            continue
        hits.append({"path": str(path), "line": lineno, "models": history_model_ids_from_text(line), "text": line.strip()[:360]})
        if len(hits) >= 20:
            break
    return hits


def history_interesting_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.name in {"manifest.json", "state.json"}
        or name.endswith((".log", ".md", ".json", ".jsonl", ".sh", ".py"))
    )


def history_scan_roots(roots: list[Path], query: str | None = None, limit: int = 5000, include_chat_logs: bool = False, chat_roots: list[Path] | None = None) -> dict[str, Any]:
    query_l = (query or "").lower()
    effective_roots = list(roots)
    if include_chat_logs:
        effective_roots.extend(chat_roots or history_known_chat_roots())
    models: dict[str, dict[str, Any]] = {}
    scanned_files = 0
    benchmark_paths: list[Path] = []
    for root in effective_roots:
        root = root.expanduser()
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if scanned_files >= max(1, limit):
                break
            is_chat = history_is_chat_file(path)
            if not path.is_file() or history_path_skipped(path) or (not history_interesting_file(path) and not is_chat):
                continue
            if query_l:
                haystack = str(path).lower()
                if query_l not in haystack:
                    try:
                        haystack += "\n" + path.read_text(encoding="utf-8", errors="replace")[:8192].lower()
                    except OSError:
                        pass
                if query_l not in haystack:
                    continue
            scanned_files += 1
            low_name = path.name.lower()
            if is_chat:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")[:512_000]
                except OSError:
                    continue
                hits = history_extract_chat_hits(path, text)
                if not hits:
                    continue
                model_ids = sorted({model for hit in hits for model in hit.get("models", [])}) or ["chat-session-evidence"]
                for chat_model_id in model_ids:
                    chat_row = history_model_bucket(models, chat_model_id)
                    history_add_unique(chat_row, "paths", str(path.parent))
                    for hit in hits:
                        history_add_unique(chat_row, "chat_logs", hit)
                continue
            model_id = history_model_id_from_path(path)
            if path.name in {"manifest.json", "state.json"}:
                row = history_model_bucket(models, model_id)
                history_add_unique(row, "paths", str(path.parent))
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                if isinstance(data, dict):
                    summary = history_manifest_summary(path, data)
                    history_add_unique(row, "manifests", summary)
                    if summary.get("current_ppl") is not None:
                        history_add_unique(row, "ppl_results", {"path": str(path), "kind": "state", "ppl": summary["current_ppl"], "profile": summary.get("profile")})
                continue
            if low_name.endswith(".log"):
                results = history_ppl_results_from_log(path)
                if not results:
                    continue
                row = history_model_bucket(models, model_id)
                history_add_unique(row, "paths", str(path.parent))
                for result in results:
                    history_add_unique(row, "ppl_results", result)
                continue
            if low_name.endswith((".json", ".jsonl")):
                if "benchmark" in str(path).lower() or any(token in low_name for token in ("arc", "mmlu", "humaneval", "evalplus", "hellaswag", "speed")):
                    benchmark_paths.append(path)
                continue
            if low_name.endswith((".sh", ".py")):
                if any(token in str(path).lower() for token in ("benchmark", "run", "quant", "ppl", "ablation", "cerebellum")):
                    row = history_model_bucket(models, model_id)
                    history_add_unique(row, "paths", str(path.parent))
                    history_add_unique(row, "scripts", str(path))
                continue
            if low_name.endswith(".md"):
                try:
                    signals = history_extract_doc_signals(path, path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
                if not signals["methods"] and not signals["benchmarks"] and not signals["notes"]:
                    continue
                row = history_model_bucket(models, model_id)
                history_add_unique(row, "paths", str(path.parent))
                for method in signals["methods"]:
                    history_add_unique(row, "methods", method)
                for bench in signals["benchmarks"]:
                    history_add_unique(row, "benchmark_results", bench)
                for note in signals["notes"]:
                    history_add_unique(row, "notes", note)
    for record in benchmark_records(benchmark_paths) if benchmark_paths else []:
        row = history_model_bucket(models, str(record["model"]))
        history_add_unique(row, "benchmark_results", record)
    model_rows = sorted(models.values(), key=lambda item: str(item["model"]).lower())
    return {
        "schema": "cerebellum.history.v1",
        "generated_at": utc_now(),
        "roots": [str(root) for root in roots],
        "chat_roots": [str(root) for root in (chat_roots or history_known_chat_roots())] if include_chat_logs else [],
        "include_chat_logs": include_chat_logs,
        "query": query,
        "scanned_files": scanned_files,
        "models": model_rows,
        "leaderboard": history_leaderboard(model_rows),
        "ranking_policy": "Best-model ranking uses measured quality-percentage benchmarks only. PPL, speed, and size are supporting context because they are not directly comparable across corpora/models.",
    }


def history_leaderboard(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        scores: dict[str, list[float]] = {}
        for bench in model.get("benchmark_results", []):
            metric = str(bench.get("metric", ""))
            if metric not in {"percent", "accuracy", "pass_at_1", "score"}:
                continue
            value = bench.get("value")
            if value is None:
                continue
            value_f = float(value)
            if metric == "pass_at_1" and value_f <= 1.0:
                value_f *= 100.0
            if value_f > 100.0:
                continue
            key = str(bench.get("benchmark_key") or bench.get("benchmark") or "unknown")
            scores.setdefault(key, []).append(value_f)
        if not scores:
            continue
        avg_scores = {key: sum(values) / len(values) for key, values in scores.items()}
        ppls = [float(row["ppl"]) for row in model.get("ppl_results", []) if row.get("ppl") is not None]
        rows.append(
            {
                "model": model["model"],
                "average_score": sum(avg_scores.values()) / len(avg_scores),
                "benchmarks": len(avg_scores),
                "benchmark_scores": avg_scores,
                "best_ppl": min(ppls) if ppls else None,
            }
        )
    rows.sort(key=lambda row: (row["average_score"], row["benchmarks"]), reverse=True)
    return rows


def history_markdown(report: dict[str, Any]) -> str:
    model_rows = []
    for model in report["models"]:
        ppls = [float(row["ppl"]) for row in model.get("ppl_results", []) if row.get("ppl") is not None]
        model_rows.append(
            [
                str(model["model"]),
                str(len(model.get("manifests", []))),
                "-" if not ppls else f"{min(ppls):.2f}",
                str(len(model.get("benchmark_results", []))),
                str(len(model.get("methods", []))),
                str(len(model.get("scripts", []))),
            ]
        )
    leaderboard_rows = [
        [
            row["model"],
            f"{float(row['average_score']):.2f}%",
            str(row["benchmarks"]),
            "-" if row.get("best_ppl") is None else f"{float(row['best_ppl']):.2f}",
        ]
        for row in report["leaderboard"][:20]
    ]
    parts = [
        "# Cerebellum History",
        "",
        f"generated: `{report['generated_at']}`",
        f"roots: `{', '.join(report['roots'])}`",
        f"chat roots: `{', '.join(report.get('chat_roots', [])) or '-'}`",
        f"query: `{report.get('query') or '-'}`",
        f"scanned files: `{report['scanned_files']}`",
        "",
        "## Models",
        "",
        markdown_table(["Model", "Manifests", "Best PPL", "Bench rows", "Recreate notes", "Scripts"], model_rows),
        "",
        "## Best-Model View",
        "",
        report["ranking_policy"],
        "",
        markdown_table(["Model", "Avg quality", "Benchmarks", "Best PPL"], leaderboard_rows) if leaderboard_rows else "No comparable benchmark rows found.",
    ]
    for model in report["models"]:
        parts.extend(["", f"## {model['model']}", ""])
        manifests = [
            [
                str(row.get("profile") or "-"),
                str(row.get("base_type") or "-"),
                str(row.get("chunks") or "-"),
                str(row.get("current_ppl") or "-"),
                str(row.get("path") or "-"),
            ]
            for row in model.get("manifests", [])[:8]
        ]
        if manifests:
            parts.extend([markdown_table(["Profile", "Base", "Chunks", "PPL", "Manifest"], manifests), ""])
        for method in model.get("methods", [])[:8]:
            parts.append(f"- method: `{method['path']}:{method['line']}` {method['text']}")
        for script in model.get("scripts", [])[:8]:
            parts.append(f"- script: `{script}`")
        for hit in model.get("chat_logs", [])[:5]:
            parts.append(f"- chat: `{hit['path']}:{hit['line']}` {hit['text']}")
    return "\n".join(parts) + "\n"


def history_html(report: dict[str, Any]) -> str:
    data = html.escape(json.dumps(report, sort_keys=True))
    model_rows = []
    for model in report["models"]:
        ppls = [float(row["ppl"]) for row in model.get("ppl_results", []) if row.get("ppl") is not None]
        search = " ".join(
            [
                str(model["model"]),
                " ".join(str(item.get("text", "")) for item in model.get("methods", [])[:12]),
                " ".join(str(item.get("text", "")) for item in model.get("chat_logs", [])[:12]),
                " ".join(str(item.get("path", "")) for item in model.get("manifests", [])[:12]),
                " ".join(str(item) for item in model.get("scripts", [])[:12]),
            ]
        ).lower()
        model_rows.append(
            "<tr data-search=\"{}\"><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(search),
                html.escape(str(model["model"])),
                "-" if not ppls else f"{min(ppls):.2f}",
                str(len(model.get("benchmark_results", []))),
                str(len(model.get("methods", []))),
            )
        )
    board_rows = "".join(
        "<tr><td>{}</td><td>{:.2f}%</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(row["model"])),
            float(row["average_score"]),
            int(row["benchmarks"]),
            "-" if row.get("best_ppl") is None else f"{float(row['best_ppl']):.2f}",
        )
        for row in report["leaderboard"][:50]
    )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>Cerebellum History</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #161616; background: #fafafa; }}
input {{ width: min(720px, 100%); padding: 10px; font-size: 16px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; background: white; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; }}
pre {{ white-space: pre-wrap; background: #fff; border: 1px solid #ddd; padding: 12px; max-height: 420px; overflow: auto; }}
</style>
<h1>Cerebellum History</h1>
<p>{html.escape(report['ranking_policy'])}</p>
<input id="q" placeholder="Search model, method, script, manifest path..." autofocus>
<h2>Best-Model View</h2>
<table><thead><tr><th>Model</th><th>Avg quality</th><th>Benchmarks</th><th>Best PPL</th></tr></thead><tbody>{board_rows}</tbody></table>
<h2>Models</h2>
<table id="models"><thead><tr><th>Model</th><th>Best PPL</th><th>Bench rows</th><th>Method notes</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table>
<h2>Raw JSON</h2>
<pre>{data}</pre>
<script>
const q = document.getElementById('q');
const rows = [...document.querySelectorAll('#models tbody tr')];
q.addEventListener('input', () => {{
  const term = q.value.trim().toLowerCase();
  rows.forEach(row => row.style.display = row.dataset.search.includes(term) ? '' : 'none');
}});
</script>
"""


def history_cmd(args: argparse.Namespace) -> None:
    roots = [Path(item) for item in args.root] if args.root else history_default_roots()
    chat_roots = [Path(item) for item in args.chat_root] if args.chat_root else None
    report = history_scan_roots(roots, query=args.query, limit=max(1, args.limit), include_chat_logs=args.include_chat_logs, chat_roots=chat_roots)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(history_markdown(report), encoding="utf-8")
    if args.html:
        Path(args.html).write_text(history_html(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.output or args.markdown or args.html:
        for label, value in [("history JSON", args.output), ("history Markdown", args.markdown), ("history HTML", args.html)]:
            if value:
                print(f"{label}: {value}")
        return
    print(history_markdown(report), end="")


def repo_relative_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return Path(path.name)


def public_export_allowed(path: Path) -> bool:
    rel = repo_relative_path(path).as_posix()
    if path.suffix.lower() not in PUBLIC_EXPORT_ALLOWED_SUFFIXES:
        return False
    return any(pattern.search(rel) for pattern in PUBLIC_EXPORT_ALLOWED_PATTERNS)


def public_export_candidates(paths: list[str] | None = None) -> list[Path]:
    if paths:
        candidates = expand_audit_paths(paths)
    else:
        candidates = [path for path in tracked_files() if path.is_file()]
    return [path for path in candidates if public_export_allowed(path)]


def public_export_plan(paths: list[str] | None = None, max_bytes: int = 512_000) -> dict[str, Any]:
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in public_export_candidates(paths or []):
        rel = repo_relative_path(path)
        audit = public_audit([str(path)], max_bytes=max_bytes)
        if audit["blocked"]:
            skipped.append({"path": rel.as_posix(), "findings": audit["findings"]})
            continue
        size = path_size(path)
        exported.append(
            {
                "path": rel.as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path) if size < 128 * 1024 * 1024 else None,
            }
        )
    exported.sort(key=lambda item: item["path"])
    skipped.sort(key=lambda item: item["path"])
    return {
        "schema": "cerebellum.public_export.v1",
        "mode": "public",
        "files": exported,
        "skipped": skipped,
        "blocked": bool(skipped),
        "notes": [
            "Exports only allowlisted public docs, model cards, benchmark artifacts, and release assets.",
            "This does not rewrite Git history; run from a clean branch/tree after history filtering or unrelated-history rebuild.",
        ],
    }


def release_gate_remote_policy(remote: str) -> dict[str, str]:
    normalized = (remote or "").strip() or "origin"
    if normalized == "origin":
        return {
            "remote": normalized,
            "visibility": "public",
            "mode": "strict",
            "reason": "origin is the public release remote",
        }
    if normalized == "dev":
        return {
            "remote": normalized,
            "visibility": "private",
            "mode": "advisory",
            "reason": "dev is the private factory remote",
        }
    return {
        "remote": normalized,
        "visibility": "unknown",
        "mode": "advisory",
        "reason": "unknown remotes are not assumed to be public origin",
    }


def release_gate_explicit_path_findings(paths: list[str] | None) -> list[dict[str, Any]]:
    if not paths:
        return []
    allowed = {path.resolve() for path in public_export_candidates(paths)}
    findings = []
    for path in expand_audit_paths(paths):
        if path.resolve() in allowed:
            continue
        findings.append(
            {
                "path": repo_relative_path(path).as_posix(),
                "kind": "path",
                "reason": "not in public export allowlist",
                "severity": "blocker",
            }
        )
    return findings


def release_gate(
    paths: list[str] | None = None,
    *,
    remote: str = "origin",
    benchmark_results: list[str] | None = None,
    suite: str = "release",
    model: str | None = None,
    require_benchmarks: bool = False,
    max_bytes: int = 512_000,
) -> dict[str, Any]:
    policy = release_gate_remote_policy(remote)
    selected_paths = paths or []
    benchmark_paths = benchmark_results or []
    audit_targets = selected_paths if selected_paths else [str(path) for path in public_export_candidates([])]
    audit = public_audit(audit_targets, max_bytes=max_bytes)
    export_plan = public_export_plan(selected_paths, max_bytes=max_bytes)
    non_allowlisted = release_gate_explicit_path_findings(selected_paths)
    benchmark = benchmark_manifest(benchmark_paths, suite=suite, model=model) if benchmark_paths else None
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if policy["visibility"] == "public":
        if audit["blocked"]:
            blockers.append({"source": "public_audit", "reason": "public audit has blocker findings", "count": len(audit["findings"])})
        if export_plan["blocked"]:
            blockers.append({"source": "public_export", "reason": "public export plan skipped audited files", "count": len(export_plan["skipped"])})
        if non_allowlisted:
            blockers.append({"source": "path_policy", "reason": "explicit paths are not public export allowlisted", "count": len(non_allowlisted)})
    else:
        if audit["blocked"]:
            warnings.append({"source": "public_audit", "reason": "private/advisory remote has public-audit findings", "count": len(audit["findings"])})
        if export_plan["blocked"]:
            warnings.append({"source": "public_export", "reason": "private/advisory remote has skipped public-export files", "count": len(export_plan["skipped"])})
        if non_allowlisted:
            warnings.append({"source": "path_policy", "reason": "private/advisory remote includes non-public-export paths", "count": len(non_allowlisted)})

    if require_benchmarks and not benchmark_paths:
        blockers.append({"source": "benchmark_manifest", "reason": "benchmark results path required", "count": 0})
    if benchmark and benchmark["missing_measured"]:
        target = blockers if require_benchmarks else warnings
        target.append(
            {
                "source": "benchmark_manifest",
                "reason": "suite has missing measured benchmarks",
                "count": len(benchmark["missing_measured"]),
                "missing": benchmark["missing_measured"],
            }
        )

    ready = not blockers
    return {
        "schema": "cerebellum.release_gate.v1",
        "remote": policy["remote"],
        "visibility": policy["visibility"],
        "mode": policy["mode"],
        "ready": ready,
        "blockers": blockers,
        "warnings": warnings,
        "paths": selected_paths,
        "benchmark_results": benchmark_paths,
        "audit": audit,
        "export_plan": export_plan,
        "non_allowlisted": non_allowlisted,
        "benchmark_manifest": benchmark,
        "notes": [
            policy["reason"],
            "origin requires public-audit clean, public-export clean, and allowlisted explicit paths.",
            "dev and unknown remotes are advisory for public-safety findings but still honor explicit benchmark gates.",
            "This gate does not rewrite Git history; scrub or rebuild public history before pushing origin.",
        ],
    }


def release_gate_markdown(report: dict[str, Any]) -> str:
    title = "Release gate passed" if report["ready"] else "Release gate blocked"
    parts = [
        f"{title}: remote `{report['remote']}` ({report['visibility']}/{report['mode']}).",
        "",
    ]
    if report["blockers"]:
        rows = [
            [item["source"], str(item.get("count", "")), item["reason"], ", ".join(item.get("missing", []))]
            for item in report["blockers"]
        ]
        parts.extend(["## Blockers", "", markdown_table(["Source", "Count", "Reason", "Missing"], rows), ""])
    if report["warnings"]:
        rows = [
            [item["source"], str(item.get("count", "")), item["reason"], ", ".join(item.get("missing", []))]
            for item in report["warnings"]
        ]
        parts.extend(["## Warnings", "", markdown_table(["Source", "Count", "Reason", "Missing"], rows), ""])
    if report["non_allowlisted"]:
        rows = [[item["path"], item["reason"]] for item in report["non_allowlisted"][:25]]
        parts.extend(["## Non-Allowlisted Explicit Paths", "", markdown_table(["Path", "Reason"], rows), ""])
    manifest = report.get("benchmark_manifest")
    if manifest:
        measured = ", ".join(manifest["measured_benchmarks"]) or "-"
        missing = ", ".join(manifest["missing_measured"]) or "-"
        parts.extend(["## Benchmarks", "", f"measured: {measured}", f"missing: {missing}", ""])
    parts.extend(["## Notes", "", *[f"- {note}" for note in report["notes"]]])
    return "\n".join(parts).rstrip() + "\n"


def release_gate_cmd(args: argparse.Namespace) -> None:
    report = release_gate(
        args.paths,
        remote=args.remote,
        benchmark_results=args.benchmark_results,
        suite=args.suite,
        model=args.model,
        require_benchmarks=args.require_benchmarks,
        max_bytes=args.max_bytes,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(release_gate_markdown(report), end="")
    if not report["ready"]:
        raise SystemExit(1)


def public_export_plan_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    try:
        max_bytes = int(query_value(qs, "max_bytes", 512_000))
    except ValueError as exc:
        raise ValueError("max_bytes must be an integer") from exc
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1")
    paths = qs.get("path") or qs.get("paths") or []
    return argparse.Namespace(paths=paths, max_bytes=max_bytes)


def public_export_cmd(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    plan = public_export_plan(args.paths, max_bytes=args.max_bytes)
    if args.dry_run:
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            print(public_export_markdown(plan), end="")
        if plan["blocked"]:
            raise SystemExit(1)
        return
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in plan["files"]:
        rel = Path(item["path"])
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rel, dest)
    manifest_path = output_dir / "cerebellum_public_export_manifest.json"
    atomic_write_json(manifest_path, plan)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(public_export_markdown(plan), end="")
        print(f"manifest: {manifest_path}")
    if plan["blocked"]:
        raise SystemExit(1)


def public_export_markdown(plan: dict[str, Any]) -> str:
    status = "blocked" if plan["blocked"] else "ready"
    lines = [f"Public export {status}: {len(plan['files'])} files selected, {len(plan['skipped'])} skipped."]
    if plan["files"]:
        rows = [[item["path"], fmt_bytes(item["size_bytes"])] for item in plan["files"]]
        lines.extend(["", markdown_table(["Path", "Size"], rows)])
    if plan["skipped"]:
        rows = []
        for item in plan["skipped"]:
            reasons = sorted({finding["reason"] for finding in item["findings"]})
            rows.append([item["path"], "; ".join(reasons)])
        lines.extend(["", "## Skipped", "", markdown_table(["Path", "Reasons"], rows)])
    return "\n".join(lines) + "\n"


def system_info() -> dict[str, Any]:
    import platform

    info: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "env": {
            "CEREBELLUM_DATA_ROOT": os.environ.get("CEREBELLUM_DATA_ROOT"),
            "CEREBELLUM_DB": os.environ.get("CEREBELLUM_DB"),
            "LLAMA_QUANTIZE_BIN": os.environ.get("LLAMA_QUANTIZE_BIN"),
            "LLAMA_PERPLEXITY_BIN": os.environ.get("LLAMA_PERPLEXITY_BIN"),
            "HF_TOKEN_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
            "GITHUB_TOKEN_present": bool(os.environ.get("GITHUB_TOKEN")),
        },
        "binaries": {},
        "memory": {},
        "gpus": [],
        "filesystems": [],
        "recommended": {},
    }
    for label, name, env_var in [
        ("llama_quantize", "llama-quantize", "LLAMA_QUANTIZE_BIN"),
        ("llama_perplexity", "llama-perplexity", "LLAMA_PERPLEXITY_BIN"),
        ("llama_server", "llama-server", "LLAMA_SERVER_BIN"),
        ("distrobox", "distrobox", "DISTROBOX_BIN"),
        ("git", "git", "GIT"),
        ("gh", "gh", "GH"),
    ]:
        common = common_llama_bins(name) if name.startswith("llama-") else None
        info["binaries"][label] = find_executable(name, env_var, common)
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            meminfo = {}
            for line in f:
                key, value = line.split(":", 1)
                meminfo[key] = int(value.strip().split()[0]) * 1024
            info["memory"] = {
                "total_bytes": meminfo.get("MemTotal"),
                "available_bytes": meminfo.get("MemAvailable"),
            }
    except OSError:
        pass
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        proc = subprocess.run(
            [
                nvidia,
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    info["gpus"].append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "vram_total_mib": int(parts[2]),
                            "vram_free_mib": int(parts[3]),
                            "vram_used_mib": int(parts[4]),
                        }
                    )
    seen: set[str] = set()
    for root in [Path.cwd(), Path.home(), Path("/tmp"), default_data_root().parent]:
        try:
            resolved = str(root.resolve())
        except OSError:
            continue
        if resolved in seen or not root.exists():
            continue
        seen.add(resolved)
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        info["filesystems"].append(
            {
                "path": str(root),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )
    writable = [fs for fs in info["filesystems"] if fs["free_bytes"] is not None]
    writable.sort(key=lambda fs: fs["free_bytes"], reverse=True)
    info["recommended"] = {
        "data_root": str(default_data_root()),
        "scratch_root": writable[0]["path"] if writable else str(default_data_root()),
        "use_distrobox": bool(shutil.which("distrobox")),
        "low_space_mode": "two-slot-pipeline",
    }
    return info


def system_cmd(args: argparse.Namespace) -> None:
    info = system_info()
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
        return
    print("Cerebellum system")
    print(f"host       : {info.get('hostname')}")
    print(f"platform   : {info.get('platform')}")
    print(f"python     : {info.get('python')}")
    print(f"cpu_count  : {info.get('cpu_count')}")
    mem = info.get("memory") or {}
    if mem.get("available_bytes"):
        print(f"ram        : {mem.get('available_bytes') / 2**30:.1f} GiB available / {mem.get('total_bytes') / 2**30:.1f} GiB total")
    print("binaries")
    for key, value in info["binaries"].items():
        print(f"  {key:16s} {value}")
    print("gpus")
    for gpu in info["gpus"]:
        print(f"  cuda:{gpu['index']} {gpu['name']} free={gpu['vram_free_mib']} MiB total={gpu['vram_total_mib']} MiB")
    print("filesystems")
    for fs in info["filesystems"]:
        print(f"  {fs['path']} free={fs['free_bytes'] / 2**30:.1f} GiB total={fs['total_bytes'] / 2**30:.1f} GiB")
    print("recommended")
    for key, value in info["recommended"].items():
        print(f"  {key:16s} {value}")


def doctor_cmd(args: argparse.Namespace) -> None:
    info = system_info()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, fix: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "fix": fix})

    quant = info["binaries"].get("llama_quantize")
    ppl = info["binaries"].get("llama_perplexity")
    add(
        "llama-quantize",
        bool(quant and Path(str(quant)).exists()),
        str(quant),
        "Install/build llama.cpp and put llama-quantize on PATH, or pass --quantize-bin.",
    )
    add(
        "llama-perplexity",
        bool(ppl and Path(str(ppl)).exists()),
        str(ppl),
        "Install/build llama.cpp and put llama-perplexity on PATH, or pass --perplexity-bin.",
    )
    distrobox = info["binaries"].get("distrobox")
    add(
        "distrobox optional",
        True,
        f"{distrobox or 'not installed'}; only needed if llama.cpp must run inside a container/toolbox",
        "Do not use --distrobox on normal host installs. Use it only when CUDA/ROCm libs live in that environment.",
    )
    add(
        "gpu",
        bool(info.get("gpus")),
        ", ".join(f"{gpu['name']} {gpu['vram_free_mib']}/{gpu['vram_total_mib']} MiB free" for gpu in info.get("gpus", [])) or "no NVIDIA GPU detected",
        "CPU runs are possible but slow. For NVIDIA, ensure nvidia-smi works and llama.cpp was built with CUDA.",
    )
    root = default_data_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        free = disk_free_gb(root)
        add("data root", os.access(root, os.W_OK), f"{root} free={free:.1f} GiB", "Set CEREBELLUM_DATA_ROOT to a writable drive.")
    except OSError as exc:
        add("data root", False, f"{root}: {exc}", "Set CEREBELLUM_DATA_ROOT to a writable drive.")
    for profile, candidates in PPL_PROFILES.items():
        paths = profile_candidate_paths(profile)
        found = next((str(path) for path in paths if path.exists()), None)
        add(
            f"profile:{profile}",
            bool(found),
            found or "not found locally",
            f"Pass --profile custom --corpus FILE, set CEREBELLUM_CORPUS_ROOT, or place a corpus under ./corpora or ~/.cache/cerebellum/corpora. Expected names: {', '.join(candidates)}",
        )
    if args.source_gguf:
        source = Path(args.source_gguf)
        arch = gguf_field_text(source, "general.architecture") if source.exists() else None
        tensor_sample = ""
        if source.exists():
            try:
                from gguf import GGUFReader

                reader = GGUFReader(str(source))
                tensor_sample = ",".join(t.name for t in reader.tensors[:8])
            except Exception:
                tensor_sample = ""
        gemma4_ok = arch == "gemma4"
        bad_gemma4_prefix = "model.language_model." in tensor_sample
        add(
            "source gguf",
            source.exists(),
            str(source),
            "Provide the converted F16/source GGUF used by Cerebellum.",
        )
        add(
            "gemma4 architecture",
            not args.source_gguf or (gemma4_ok and not bad_gemma4_prefix) or "gemma" not in source.name.lower(),
            f"general.architecture={arch or 'unknown'} sample_tensors={tensor_sample or 'unreadable'}",
            "Gemma 4 12B conversion must register Gemma4UnifiedForConditionalGeneration on llama.cpp Gemma4Model so model.language_model.* is stripped and the GGUF architecture is gemma4.",
        )
    payload = {"ok": all(row["ok"] for row in checks if not row["name"].startswith("profile:")), "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("Cerebellum doctor")
    print("Portable default: host binaries. `--distrobox NAME` is optional, not required.")
    for row in checks:
        mark = "OK" if row["ok"] else "!!"
        print(f"{mark} {row['name']}: {row['detail']}")
        if not row["ok"] and row.get("fix"):
            print(f"   fix: {row['fix']}")


def self_test_payload(run_dir: Path | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add("python", sys.version_info >= (3, 10), sys.version.split()[0])
    add("tutorials", all(topic in TUTORIALS for topic in ["overview", "recovery", "low-space", "targeting", "api"]), sorted(TUTORIALS))
    info = system_info()
    add("system_info", bool(info.get("schema_version")), {"hostname": info.get("hostname"), "python": info.get("python")})
    add("llama_quantize", Path(str(info["binaries"].get("llama_quantize", ""))).exists(), info["binaries"].get("llama_quantize"))
    add("llama_perplexity", Path(str(info["binaries"].get("llama_perplexity", ""))).exists(), info["binaries"].get("llama_perplexity"))
    add("api_catalog", True, ["/health", "/schema", "/tutorial", "/recover", "/export", "/pipeline-plan", "/inspect-gguf-types", "/compare-gguf-types", "/commands"])
    if run_dir:
        state = read_json(run_dir / "state.json", {})
        manifest = read_json(run_dir / "manifest.json", {})
        add("run_state", bool(state), str(run_dir / "state.json"))
        add("run_manifest", bool(manifest), str(run_dir / "manifest.json"))
        if state or manifest:
            recovery = build_recovery_plan(run_dir)
            add(
                "recover_payload",
                bool(recovery.get("run_dir")),
                {
                    "runner_active": recovery.get("runner_active"),
                    "locked_count": recovery.get("locked_count"),
                    "active_health": recovery.get("active_health"),
                    "interrupted": recovery.get("interrupted"),
                },
            )
            try:
                report = build_report(run_dir)
                add("report_payload", bool(report.get("run_id")), {"run_id": report.get("run_id"), "status": report.get("status")})
            except Exception as exc:
                add("report_payload", False, str(exc))
            try:
                package = package_manifest(run_dir)
                add("package_payload", bool(package.get("schema")), {"files": len(package.get("files", []))})
            except Exception as exc:
                add("package_payload", False, str(exc))
    return {
        "schema": "cerebellum.self_test.v1",
        "ok": all(row["ok"] for row in checks),
        "checks": checks,
    }


def self_test_cmd(args: argparse.Namespace) -> None:
    payload = self_test_payload(Path(args.run_dir) if args.run_dir else None)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("Cerebellum self-test")
    for row in payload["checks"]:
        mark = "OK" if row["ok"] else "!!"
        print(f"{mark} {row['name']}: {row.get('detail')}")
    if not payload["ok"]:
        raise SystemExit(1)


def space_plan(source: Path, candidates: list[Path], margin_gb: float, create_dirs: bool = True) -> dict[str, Any]:
    source_size = path_size(source)
    required_single = int(source_size * 1.7 + margin_gb * 1e9)
    required_two_slot = int(source_size * 2.4 + margin_gb * 1e9)
    rows = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.expanduser().resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if create_dirs:
                path.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(path)
            writable = os.access(path, os.W_OK)
        except OSError as exc:
            rows.append({"path": str(path), "ok": False, "error": str(exc)})
            continue
        mode = "insufficient"
        if writable and usage.free >= required_two_slot:
            mode = "two-slot-pipeline"
        elif writable and usage.free >= required_single:
            mode = "single-candidate"
        rows.append(
            {
                "path": str(path),
                "ok": True,
                "writable": writable,
                "free_bytes": usage.free,
                "source_size_bytes": source_size,
                "required_single_candidate_bytes": required_single,
                "required_two_slot_bytes": required_two_slot,
                "recommended_mode": mode,
            }
        )
    rows.sort(key=lambda row: row.get("free_bytes", -1), reverse=True)
    return {"source_gguf": str(source), "rows": rows}


def plan_space_cmd(args: argparse.Namespace) -> None:
    source = Path(args.source_gguf)
    candidates = [Path(p) for p in args.scratch_candidates.split(",") if p]
    if args.data_root:
        candidates.append(Path(args.data_root))
    candidates.extend([default_data_root(), Path.cwd(), Path("/tmp")])
    source_size = path_size(source)
    payload = space_plan(source, candidates, args.margin_gb)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"source: {source} ({source_size / 2**30:.2f} GiB)")
    for row in payload["rows"]:
        if not row.get("ok"):
            print(f"  {row['path']}: unavailable {row.get('error')}")
        else:
            print(f"  {row['path']}: free={row['free_bytes'] / 2**30:.1f} GiB mode={row['recommended_mode']}")


TUTORIALS = {
    "overview": [
        "Cerebellum is a resource-aware GGUF quantization toolbox.",
        "The normal flow is: system -> plan-space -> run -> watch/status/events -> report/export -> upload.",
        "Cerebellum builds a baseline quant, tests candidate quant levels per tensor, measures real PPL, then locks the best tensor type.",
        "Pick a PPL target with `--profile wiki`, `--profile agentic`, `--profile code`, `--profile math`, `--profile all-around`, or `--profile custom --corpus FILE`.",
        "Append-only event logs are the source of truth. SQLite is only a query index.",
    ],
    "flow": [
        "1. Run `cerebellum system` to see GPU, RAM, disks, auth, and llama.cpp binaries.",
        "2. Run `cerebellum plan-space --source-gguf model.gguf` to choose scratch strategy.",
        "3. Generate imatrix when needed with `cerebellum imatrix --model HF_OR_PATH --output cerebellum_imatrix.dat`.",
        "4. Run `cerebellum run --source-gguf ... --imatrix cerebellum_imatrix.dat --profile wiki --family ... --model-name ...`.",
        "5. Watch with `cerebellum watch RUN_DIR`, `cerebellum status RUN_DIR`, and `cerebellum events RUN_DIR`.",
        "6. Generate reports with `cerebellum report RUN_DIR`.",
        "7. Import query data with `cerebellum db import-run RUN_DIR`.",
    ],
    "imatrix": [
        "Imatrix generation is a Cerebellum feature exposed as `cerebellum imatrix`.",
        "When you pass `--family`, `--model-name`, and `--source-name`, the imatrix is stored in the same Cerebellum project tree as future runs.",
        "Default mode is streaming: it reads safetensors one tensor at a time and avoids loading the whole model into RAM.",
        "Example: `cerebellum imatrix --model Qwen/Qwen3.6-27B --family qwen --model-name qwen3.6-27b --source-name hf-safetensors -v`.",
        "Use `--mode calibrated` only when the system can load the model and you want activation calibration blended with weight sensitivity.",
        "After generation, Cerebellum writes `cerebellum_project.json` and prints the next `cerebellum run --imatrix ...` command.",
        "The legacy Python module path still exists only for compatibility during the package rename.",
    ],
    "project": [
        "A Cerebellum project is grouped as `DATA_ROOT/families/FAMILY/MODEL/sources/SOURCE/`.",
        "`imatrix/` stores imatrix files and provenance for that model source.",
        "`runs/` stores every Cerebellum run under the same model source.",
        "`cerebellum_project.json` records source identity, imatrix location, layout, and the next command.",
        "This keeps model families separate while preserving queryable long-term research data.",
    ],
    "gemma4-source": [
        "Gemma 4 12B source conversion has a llama.cpp converter gotcha before Cerebellum ever runs.",
        "The HF class is `Gemma4UnifiedForConditionalGeneration`, wrapping the text backbone at `model.language_model.*`.",
        "llama.cpp Gemma4Model knows how to strip that prefix, but the architecture name must be registered to that model handler.",
        "Patch llama.cpp conversion with `@ModelBase.register(\"Gemma4UnifiedForConditionalGeneration\")` on `Gemma4Model` before making the F16 GGUF.",
        "A good converted source GGUF should report `general.architecture=gemma4` and llama.cpp-style tensor names such as `blk.0.ffn_down.weight`.",
        "Use `cerebellum doctor --source-gguf SOURCE.gguf` to inspect this before starting a long run.",
    ],
    "low-space": [
        "Use `--scratch-root` when metadata and large GGUF artifacts should live on different drives.",
        "Default mode keeps CPU/GPU overlap for speed, but deletes measured non-winning candidate GGUFs immediately.",
        "Use `--hard-free-floor-gb 10` to prevent starting another quant job unless one estimated candidate plus 10 GiB remains.",
        "Use `--low-space` or `cerebellum resume RUN_DIR --low-space` when disk pressure matters more than CPU/GPU overlap.",
        "Use `cerebellum cleanup RUN_DIR --partials --yes` only after a stopped/crashed run; active runs are guarded.",
        "Never delete source GGUF, manifest, state, event logs, candidate logs, checkpoints, or tensor-type files.",
    ],
    "recovery": [
        "`state.json` advances only after a tensor is locked, so partial tensor data is not trusted after a crash.",
        "Run `cerebellum recover RUN_DIR` to see runner status, partial temp, disk footprint, and exact next commands.",
        "Run `cerebellum resume RUN_DIR` to restart from manifest/state without reconstructing the original run command.",
        "Run `cerebellum backup RUN_DIR --to BACKUP_ROOT` or start with `--backup-root BACKUP_ROOT` for off-drive metadata copies.",
        "Run `cerebellum rollback RUN_DIR --last-completed-layer --yes` to trim state to a clean layer boundary.",
        "After rollback, the next resume rebuilds the baseline GGUF from the rolled-back tensor map.",
    ],
    "targeting": [
        "Cerebellum is not all-or-nothing; target exact work with `--layers`, `--tensor-regex`, or `--tensor-file`.",
        "Example: `cerebellum run ... --layers 0,1,8-12` tests only those layers.",
        "Example: `cerebellum run ... --tensor-regex 'blk\\.12\\.(attn_q|attn_k)\\.weight'` tests precise tensors.",
        "Use rollback before rerunning targeted layers if you need to discard prior decisions.",
        "Use reports and candidate logs to compare targeted tests against previous full runs.",
    ],
    "api": [
        "Start automation API with `cerebellum api --host 127.0.0.1 --port 8931 --data-root DATA_ROOT`.",
        "Read run state with `/run?run_dir=RUN_DIR`, events with `/events?run_dir=RUN_DIR`, and measurements with `/measurements?run_dir=RUN_DIR`.",
        "Use `/report?run_dir=RUN_DIR` and `/export?run_dir=RUN_DIR&kind=ai` for AI-readable summaries.",
        "Use `/recover?run_dir=RUN_DIR` for AI-safe recovery planning without deleting or changing files.",
        "Use `/commands` to discover CLI command templates and `/tutorial?topic=TOPIC` to expose these tutorials to agents.",
        "Use `cerebellum self-test --run-dir RUN_DIR` or `/self-test?run_dir=RUN_DIR` for read-only smoke checks.",
        "Destructive operations like cleanup, rollback, upload, and stop remain CLI actions unless explicitly wired as authenticated POST later.",
    ],
    "provenance": [
        "Cerebellum uses visible `cerebellum.*` GGUF metadata for attribution and auditability.",
        "Run `cerebellum provenance --run-dir RUN_DIR` to generate expected metadata.",
        "Run `cerebellum provenance --gguf MODEL.gguf` to inspect whether a downloaded GGUF still carries Cerebellum tags.",
        "Run `cerebellum finalize --run-dir RUN_DIR --gguf MODEL.gguf --inject` when a compatible metadata tool is installed.",
        "This is transparent provenance, not a hidden watermark; stripped keys indicate stripped attribution.",
    ],
    "outputs": [
        "`manifest.json`: immutable run identity and config.",
        "`state.json`: current resumable state, written atomically.",
        "`cerebellum_events.jsonl`: append-only operational audit trail.",
        "`cerebellum_candidates.jsonl`: per-candidate scientific measurements.",
        "`cerebellum_best_tensor_types.txt`: final tensor-type recipe.",
        "`cerebellum_infographic_data.json`: compact data for visual summaries.",
    ],
}


def tutorial_cmd(args: argparse.Namespace) -> None:
    topic = args.topic
    if topic == "list":
        print("topics:")
        for key in sorted(TUTORIALS):
            print(f"  {key}")
        return
    lines = TUTORIALS.get(topic)
    if lines is None:
        raise SystemExit(f"unknown topic {topic}; use `tutorial list`")
    print(f"Cerebellum tutorial: {topic}")
    print()
    for line in lines:
        print(f"- {line}")


def tips_cmd(args: argparse.Namespace) -> None:
    cfg = load_user_config()
    if args.value == "status":
        print("on" if cfg.get("tips", True) else "off")
        return
    cfg["tips"] = args.value == "on"
    save_user_config(cfg)
    print(f"tips {args.value}")


def auth_cmd(args: argparse.Namespace) -> None:
    result: dict[str, Any] = {"service": args.service}
    if args.service in {"hf", "huggingface"}:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        result["env_token_present"] = bool(token)
        try:
            from huggingface_hub import HfApi
            result["whoami"] = HfApi(token=token).whoami() if token else None
        except Exception as exc:
            result["error"] = str(exc)
    elif args.service == "github":
        token = os.environ.get("GITHUB_TOKEN")
        result["env_token_present"] = bool(token)
        gh = shutil.which("gh")
        result["gh_cli"] = gh
        if gh:
            proc = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
            result["returncode"] = proc.returncode
            result["output"] = (proc.stdout + proc.stderr)[-2000:]
    else:
        raise SystemExit("service must be hf or github")
    print(json.dumps(result, indent=2, sort_keys=True))


def hf_auth_header() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_fetch_json(url: str) -> Any:
    req = Request(url, headers=hf_auth_header())
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def hf_fetch_text(url: str) -> str:
    req = Request(url, headers=hf_auth_header())
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def hf_recent_model_stats(author: str, limit: int = 1000) -> dict[str, Any]:
    url = "https://huggingface.co/api/models?" + urlencode({"author": author, "full": "true", "limit": str(limit)})
    data = hf_fetch_json(url)
    models = []
    for item in data:
        model_id = item.get("modelId") or item.get("id")
        if not model_id:
            continue
        models.append(
            {
                "modelId": model_id,
                "downloads_recent": int(item.get("downloads") or 0),
                "likes": int(item.get("likes") or 0),
                "private": bool(item.get("private")),
            }
        )
    models.sort(key=lambda row: (row["downloads_recent"], row["likes"], row["modelId"]), reverse=True)
    return {
        "schema": "cerebellum.hf_model_stats.v1",
        "author": author,
        "period": "recent",
        "metric_note": "Hugging Face public model downloads are rolling/recent counts, not lifetime totals.",
        "source": "https://huggingface.co/api/models",
        "count": len(models),
        "total_downloads_recent": sum(row["downloads_recent"] for row in models),
        "total_likes": sum(row["likes"] for row in models),
        "models": models,
    }


def hf_publisher_all_time_stats(org: str) -> dict[str, Any]:
    url = f"https://huggingface.co/organizations/{quote(org)}/settings/publisher-analytics/download-breakdown"
    text = hf_fetch_text(url)
    latest_by_repo: dict[str, dict[str, Any]] = {}
    for row in csv.DictReader(text.splitlines()):
        if row.get("repoType") != "model":
            continue
        repo = row.get("repoName") or ""
        if not repo:
            continue
        try:
            total = int(float(row.get("total") or 0))
            downloads = int(float(row.get("downloads") or 0))
        except ValueError:
            continue
        timestamp = row.get("timestamp") or ""
        previous = latest_by_repo.get(repo)
        if previous is None or timestamp >= previous["latest_timestamp"]:
            latest_by_repo[repo] = {
                "modelId": repo,
                "downloads_all_time": total,
                "latest_timestamp": timestamp,
                "latest_daily_downloads": downloads,
            }
    models = sorted(latest_by_repo.values(), key=lambda row: (row["downloads_all_time"], row["modelId"]), reverse=True)
    return {
        "schema": "cerebellum.hf_model_stats.v1",
        "author": org,
        "publisher_org": org,
        "period": "all-time",
        "metric_note": "All-time totals come from Hugging Face Publisher Analytics CSV and require eligible org/auth access.",
        "source": url,
        "count": len(models),
        "total_downloads_all_time": sum(row["downloads_all_time"] for row in models),
        "models": models,
    }


def hf_model_stats(args: argparse.Namespace) -> dict[str, Any]:
    if args.period == "all-time":
        if not args.publisher_org:
            raise SystemExit("--period all-time requires --publisher-org and HF auth with Publisher Analytics access")
        return hf_publisher_all_time_stats(args.publisher_org)
    return hf_recent_model_stats(args.author, limit=max(1, args.limit))


def hf_stats_args_from_query(qs: dict[str, list[str]]) -> argparse.Namespace:
    period = qs.get("period", ["recent"])[0]
    if period not in {"recent", "all-time"}:
        raise ValueError("period must be recent or all-time")
    try:
        limit = int(qs.get("limit", ["1000"])[0])
    except ValueError as exc:
        raise ValueError("limit must be an integer") from exc
    return argparse.Namespace(
        author=qs.get("author", ["deucebucket"])[0],
        period=period,
        publisher_org=qs.get("publisher_org", [None])[0],
        limit=limit,
        snapshot=None,
    )


def hf_stats_total_key(period: str) -> str:
    return "total_downloads_all_time" if period == "all-time" else "total_downloads_recent"


def hf_stats_download_key(period: str) -> str:
    return "downloads_all_time" if period == "all-time" else "downloads_recent"


def write_hf_stats_snapshot(report: dict[str, Any], path: Path) -> dict[str, Any]:
    period = str(report["period"])
    total_key = hf_stats_total_key(period)
    download_key = hf_stats_download_key(period)
    captured_at = utc_now()
    prior_rows = read_jsonl(path)
    previous = next(
        (
            row
            for row in reversed(prior_rows)
            if row.get("period") == period and row.get("author") == report.get("author")
        ),
        None,
    )
    models = [
        {
            "modelId": row["modelId"],
            download_key: int(row.get(download_key, 0)),
            "likes": row.get("likes"),
        }
        for row in report.get("models", [])
    ]
    total = int(report.get(total_key, 0))
    record: dict[str, Any] = {
        "schema": "cerebellum.hf_model_stats_snapshot.v1",
        "captured_at": captured_at,
        "author": report.get("author"),
        "period": period,
        "source": report.get("source"),
        "metric_note": report.get("metric_note"),
        "model_count": int(report.get("count", len(models))),
        total_key: total,
        "models": models,
    }
    if previous is not None:
        previous_total = int(previous.get(total_key, 0))
        record["previous_captured_at"] = previous.get("captured_at")
        record["delta_since_previous"] = total - previous_total
        previous_models = {
            str(row.get("modelId")): int(row.get(download_key, 0))
            for row in previous.get("models", [])
            if row.get("modelId")
        }
        record["model_deltas_since_previous"] = [
            {
                "modelId": row["modelId"],
                "delta": int(row.get(download_key, 0)) - previous_models.get(row["modelId"], 0),
            }
            for row in models
            if int(row.get(download_key, 0)) - previous_models.get(row["modelId"], 0) != 0
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "captured_at": captured_at,
        "previous_captured_at": record.get("previous_captured_at"),
        "delta_since_previous": record.get("delta_since_previous"),
    }


def hf_model_stats_markdown(report: dict[str, Any]) -> str:
    period = report["period"]
    total_key = hf_stats_total_key(period)
    download_key = hf_stats_download_key(period)
    heading = "HF model stats: all-time downloads" if period == "all-time" else "HF model stats: recent rolling downloads"
    rows = [
        [row["modelId"], str(row.get(download_key, 0)), str(row.get("likes", "-"))]
        for row in report["models"]
    ]
    lines = [
        heading,
        f"author: `{report['author']}`",
        f"models: `{report['count']}`",
        f"downloads: `{report.get(total_key, 0)}`",
        f"note: {report['metric_note']}",
    ]
    snapshot = report.get("snapshot")
    if snapshot:
        lines.append(f"snapshot: `{snapshot['path']}`")
        if snapshot.get("delta_since_previous") is not None:
            lines.append(f"delta since previous snapshot: `{snapshot['delta_since_previous']}`")
    lines.extend(["", markdown_table(["Model", "Downloads", "Likes"], rows)])
    return "\n".join(lines) + "\n"


def hf_stats_cmd(args: argparse.Namespace) -> None:
    report = hf_model_stats(args)
    if args.snapshot:
        report["snapshot"] = write_hf_stats_snapshot(report, Path(args.snapshot))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(hf_model_stats_markdown(report), end="")


def public_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return public_package_report(report)


def remote_sidecar_path(report: dict[str, Any], path: Path, private: bool = False) -> str:
    if private:
        return f"cerebellum_runs/{report['run_id']}/{path.name}"
    return f"cerebellum_releases/{public_release_label(report)}/{path.name}"


def github_upload_plan(
    report: dict[str, Any],
    files: list[Path],
    repo: str,
    branch: str | None = None,
    private: bool = False,
    include_local_paths: bool = True,
) -> dict[str, Any]:
    target_branch = branch or (f"cerebellum-run-{report['run_id']}" if private else f"cerebellum-release-{public_release_label(report)}")
    planned_files = []
    for path in files:
        planned_files.append(
            {
                "name": path.name,
                "size_bytes": path_size(path),
                "github_path": remote_sidecar_path(report, path, private=private),
            }
        )
        if include_local_paths:
            planned_files[-1]["path"] = str(path)
    return {
        "target": "github",
        "repo": repo,
        "branch": target_branch,
        "files": planned_files,
        "report": report if private else public_report_summary(report),
        "mode": "private" if private else "public",
    }


def gh_json(args: list[str], allow_fail: bool = False) -> dict[str, Any]:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        if allow_fail:
            return {"__error__": (proc.stderr or proc.stdout).strip(), "__returncode__": proc.returncode}
        raise SystemExit((proc.stderr or proc.stdout).strip() or "gh command failed")
    if not proc.stdout.strip():
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gh returned non-JSON output: {proc.stdout[-500:]}") from exc


def ensure_github_branch(repo: str, branch: str) -> None:
    ref = gh_json(["api", f"repos/{repo}/git/ref/heads/{quote(branch, safe='')}"], allow_fail=True)
    if "__error__" not in ref:
        return
    repo_info = gh_json(["api", f"repos/{repo}"])
    default_branch = repo_info.get("default_branch")
    if not default_branch:
        raise SystemExit(f"could not determine default branch for {repo}")
    base_ref = gh_json(["api", f"repos/{repo}/git/ref/heads/{quote(str(default_branch), safe='')}"])
    sha = ((base_ref.get("object") or {}).get("sha"))
    if not sha:
        raise SystemExit(f"could not determine base sha for {repo}:{default_branch}")
    gh_json(
        [
            "api",
            "--method",
            "POST",
            f"repos/{repo}/git/refs",
            "-f",
            f"ref=refs/heads/{branch}",
            "-f",
            f"sha={sha}",
        ]
    )


def github_file_sha(repo: str, branch: str, path_in_repo: str) -> str | None:
    encoded = quote(path_in_repo, safe="/")
    row = gh_json(["api", f"repos/{repo}/contents/{encoded}?ref={quote(branch, safe='')}"], allow_fail=True)
    if "__error__" in row:
        return None
    sha = row.get("sha")
    return str(sha) if sha else None


def upload_github_sidecars(repo: str, branch: str, files: list[dict[str, Any]], include_local_paths: bool = True) -> list[dict[str, Any]]:
    ensure_github_branch(repo, branch)
    uploaded = []
    for item in files:
        path = Path(item["path"])
        path_in_repo = item["github_path"]
        encoded_path = quote(path_in_repo, safe="/")
        content = base64.b64encode(path.read_bytes()).decode("ascii")
        message = f"data: upload Cerebellum sidecar {path.name}"
        args = [
            "api",
            "--method",
            "PUT",
            f"repos/{repo}/contents/{encoded_path}",
            "-f",
            f"message={message}",
            "-f",
            f"branch={branch}",
            "-f",
            f"content={content}",
        ]
        sha = github_file_sha(repo, branch, path_in_repo)
        if sha:
            args.extend(["-f", f"sha={sha}"])
        gh_json(args)
        item = {"github_path": path_in_repo, "branch": branch}
        if include_local_paths:
            item["path"] = str(path)
        uploaded.append(item)
    return uploaded


def upload_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    report = build_report(run_dir)
    files = package_files(run_dir, private=args.private)
    if args.dry_run:
        if args.target == "github" and args.repo:
            payload = github_upload_plan(report, files, args.repo, args.branch, private=args.private, include_local_paths=args.private)
        else:
            payload = {
                "target": args.target,
                "repo": args.repo,
                "files": [str(p) for p in files] if args.private else [p.name for p in files],
                "report": report if args.private else public_report_summary(report),
                "mode": "private" if args.private else "public",
            }
        print(json.dumps(payload, indent=2))
        return
    if args.target in {"hf", "huggingface"}:
        if not args.repo:
            raise SystemExit("--repo required for HF upload")
        try:
            from huggingface_hub import HfApi, upload_file
        except ImportError as exc:
            raise SystemExit("huggingface_hub is required") from exc
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        api = HfApi(token=token)
        api.create_repo(args.repo, repo_type=args.repo_type, exist_ok=True)
        for path in files:
            upload_file(
                path_or_fileobj=str(path),
                path_in_repo=remote_sidecar_path(report, path, private=args.private),
                repo_id=args.repo,
                repo_type=args.repo_type,
                token=token,
            )
            print(f"uploaded {path.name}")
    elif args.target == "github":
        gh = shutil.which("gh")
        if not gh:
            raise SystemExit("gh CLI not found")
        if not args.repo:
            raise SystemExit("--repo owner/name required for GitHub upload")
        plan = github_upload_plan(report, files, args.repo, args.branch, private=args.private, include_local_paths=True)
        uploaded = upload_github_sidecars(args.repo, plan["branch"], plan["files"], include_local_paths=args.private)
        print(json.dumps({"target": "github", "repo": args.repo, "branch": plan["branch"], "uploaded": uploaded}, indent=2))
    else:
        raise SystemExit("target must be hf or github")


class CerebellumAPI(BaseHTTPRequestHandler):
    data_root: Path = Path("/var/home/deucebucket/games/cerebellum-runs")
    db_path: Path = Path(DEFAULT_DB)

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._json({"ok": True})
        elif parsed.path == "/runs":
            rows = []
            for manifest in run_glob(self.data_root):
                item = load_run(manifest.parent)
                state = item.get("state", {})
                manifest_data = item.get("manifest", {})
                if qs.get("family") and qs["family"][0] not in str(manifest_data.get("model_family") or state.get("model_family")):
                    continue
                if qs.get("model") and qs["model"][0] not in str(manifest_data.get("model_name") or state.get("model_name")):
                    continue
                if qs.get("status") and qs["status"][0] != str(state.get("run_status")):
                    continue
                rows.append(item)
            self._json({"runs": rows})
        elif parsed.path == "/projects":
            rows = discover_projects(self.data_root)
            family = qs.get("family", [None])[0]
            model = qs.get("model", [None])[0]
            source = qs.get("source", [None])[0]
            if family:
                rows = [row for row in rows if family in str(row.get("family"))]
            if model:
                rows = [row for row in rows if model in str(row.get("model_name"))]
            if source:
                rows = [row for row in rows if source in str(row.get("source_name"))]
            self._json({"data_root": str(self.data_root), "projects": rows})
        elif parsed.path == "/run":
            run_dir = qs.get("run_dir", [None])[0]
            if not run_dir:
                self._json({"error": "run_dir query param required"}, 400)
            else:
                self._json(load_run(Path(run_dir)))
        elif parsed.path == "/events":
            run_dir = qs.get("run_dir", [None])[0]
            if not run_dir:
                self._json({"error": "run_dir query param required"}, 400)
            else:
                limit = int(qs.get("limit", ["100"])[0])
                event_type = qs.get("type", [None])[0]
                rows = read_jsonl(first_existing(Path(run_dir), EVENT_FILES))
                if event_type:
                    rows = [row for row in rows if row.get("event") == event_type]
                self._json({"events": rows[-limit:]})
        elif parsed.path == "/measurements":
            run_dir = qs.get("run_dir", [None])[0]
            if not run_dir:
                self._json({"error": "run_dir query param required"}, 400)
            else:
                limit = int(qs.get("limit", ["100"])[0])
                rows = read_jsonl(first_existing(Path(run_dir), CANDIDATE_FILES))
                self._json({"measurements": rows[-limit:]})
        elif parsed.path == "/db/families":
            rows = sqlite_rows(self.db_path, "SELECT * FROM model_families ORDER BY name")
            self._json({"rows": rows})
        elif parsed.path == "/queue":
            try:
                limit = int(qs.get("limit", ["50"])[0])
                self._json(
                    {
                        "schema": "cerebellum.queue.v1",
                        "db": str(self.db_path),
                        "jobs": queue_list_jobs(self.db_path, status=qs.get("status", [None])[0], kind=qs.get("kind", [None])[0], limit=limit),
                    }
                )
            except (ValueError, OSError, sqlite3.Error) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/queue/job":
            try:
                job_id = int(qs.get("id", ["0"])[0])
                tail = int(qs.get("tail", ["0"])[0])
                self._json({"schema": "cerebellum.queue.v1", "db": str(self.db_path), "jobs": [queue_get_job(self.db_path, job_id, tail=tail)]})
            except (ValueError, SystemExit, OSError, sqlite3.Error) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/report":
            run_dir = qs.get("run_dir", [None])[0]
            if not run_dir:
                self._json({"error": "run_dir query param required"}, 400)
            else:
                self._json(build_report(Path(run_dir)))
        elif parsed.path == "/provenance":
            run_dir = qs.get("run_dir", [None])[0]
            gguf = qs.get("gguf", [None])[0]
            if not run_dir and not gguf:
                self._json({"error": "run_dir or gguf query param required"}, 400)
            else:
                payload: dict[str, Any] = {}
                gguf_path = Path(gguf) if gguf else None
                if run_dir:
                    payload["generated_metadata"] = cerebellum_metadata_block(Path(run_dir), gguf_path, False)
                if gguf_path:
                    payload["existing_cerebellum_metadata"] = inspect_gguf_metadata(gguf_path)
                    payload["has_cerebellum_metadata"] = bool(payload["existing_cerebellum_metadata"])
                self._json(payload)
        elif parsed.path == "/recover":
            run_dir = qs.get("run_dir", [None])[0]
            if not run_dir:
                self._json({"error": "run_dir query param required"}, 400)
            else:
                self._json(build_recovery_plan(Path(run_dir)))
        elif parsed.path == "/export":
            run_dir = qs.get("run_dir", [None])[0]
            kind = qs.get("kind", ["ai"])[0]
            if not run_dir:
                self._json({"error": "run_dir query param required"}, 400)
            elif kind == "infographic":
                self._json(export_payload(Path(run_dir), "infographic"))
            elif kind == "raw":
                self._json(load_run(Path(run_dir)) | {"events": read_jsonl(first_existing(Path(run_dir), EVENT_FILES)), "measurements": read_jsonl(first_existing(Path(run_dir), CANDIDATE_FILES))})
            else:
                self._json(export_payload(Path(run_dir), "ai"))
        elif parsed.path == "/package":
            run_dir = qs.get("run_dir", [None])[0]
            if not run_dir:
                self._json({"error": "run_dir query param required"}, 400)
            else:
                self._json(package_manifest(Path(run_dir)))
        elif parsed.path == "/system":
            self._json(system_info())
        elif parsed.path == "/space":
            source = qs.get("source_gguf", [None])[0]
            if not source:
                self._json({"error": "source_gguf query param required"}, 400)
            else:
                roots = [Path(p) for p in qs.get("scratch", [])]
                if not roots:
                    roots = [self.data_root, Path.cwd(), Path.home()]
                self._json(space_plan(Path(source), roots, float(qs.get("margin_gb", ["20"])[0]), create_dirs=False))
        elif parsed.path == "/pipeline-plan":
            try:
                self._json(pipeline_plan(pipeline_plan_args_from_query(qs)))
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/pipeline-run":
            try:
                args = pipeline_run_args_from_query(qs)
                self._json(pipeline_run_plan(Path(args.manifest), from_phase=args.from_phase, until_phase=args.until_phase))
            except (ValueError, SystemExit, OSError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/pipeline-status":
            try:
                args = pipeline_status_args_from_query(qs)
                self._json(pipeline_status(Path(args.manifest), Path(args.events) if args.events else None))
            except (ValueError, SystemExit, OSError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/benchmark-plan":
            try:
                args = benchmark_plan_args_from_query(qs)
                self._json(benchmark_plan(args.suite, args.model, args.port, args.results_dir))
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/benchmark-status":
            try:
                args = benchmark_status_args_from_query(qs)
                self._json(benchmark_status(Path(args.results_dir), Path(args.events) if args.events else None))
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/cpu-offload-smoke":
            try:
                self._json(cpu_offload_smoke_payload(cpu_offload_smoke_args_from_query(qs)))
            except (ValueError, OSError, RuntimeError, SystemExit, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/benchmark-manifest":
            try:
                args = benchmark_manifest_args_from_query(qs)
                manifest = benchmark_manifest(args.paths, suite=args.suite, model=args.model)
                if args.require_complete and manifest["missing_measured"]:
                    manifest["require_complete_failed"] = True
                self._json(manifest)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/benchmark-audit":
            try:
                args = benchmark_audit_args_from_query(qs)
                self._json(
                    benchmark_audit(
                        args.paths,
                        fail_empty_pct=args.fail_empty_pct,
                        fail_unknown_pct=args.fail_unknown_pct,
                        fail_pass_only_pct=args.fail_pass_only_pct,
                    )
                )
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/benchmark-report":
            try:
                args = benchmark_report_args_from_query(qs)
                if args.list_suites:
                    self._json({"suites": BENCHMARK_SUITES})
                else:
                    sizes = read_size_json(args.size_json)
                    sizes.update(parse_size_specs(args.size))
                    weights = parse_weight_specs(args.weight)
                    self._json(
                        benchmark_report(
                            args.paths,
                            baseline=args.baseline,
                            suite=args.suite,
                            leaderboard=args.leaderboard,
                            sizes=sizes,
                            weights=weights,
                        )
                    )
            except (ValueError, SystemExit, OSError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/inspect-gguf-types":
            try:
                args = inspect_gguf_types_args_from_query(qs)
                self._json(inspect_gguf_types(Path(args.gguf)))
            except (ValueError, OSError, RuntimeError, SystemExit) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/compare-gguf-types":
            try:
                args = compare_gguf_types_args_from_query(qs)
                self._json(
                    compare_gguf_types(
                        Path(args.baseline),
                        Path(args.candidate),
                        baseline_label=args.baseline_label,
                        candidate_label=args.candidate_label,
                        reference_map=Path(args.reference_map) if args.reference_map else None,
                    )
                )
            except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/artifact-inventory":
            try:
                args = artifact_inventory_args_from_query(qs)
                self._json(artifact_inventory(Path(args.root), top=args.top))
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/history":
            try:
                limit = int(qs.get("limit", ["5000"])[0])
                roots = [Path(item) for item in (qs.get("root") or qs.get("roots") or [])] or history_default_roots()
                chat_roots = [Path(item) for item in qs.get("chat_root", [])] or None
                self._json(
                    history_scan_roots(
                        roots,
                        query=qs.get("query", [None])[0],
                        limit=limit,
                        include_chat_logs=query_bool(qs, "include_chat_logs"),
                        chat_roots=chat_roots,
                    )
                )
            except (ValueError, OSError) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/public-export-plan":
            try:
                args = public_export_plan_args_from_query(qs)
                self._json(public_export_plan(args.paths, max_bytes=args.max_bytes))
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/hf-stats":
            try:
                self._json(hf_model_stats(hf_stats_args_from_query(qs)))
            except (ValueError, SystemExit) as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/benchmark-rebench-plan":
            try:
                args = benchmark_rebench_plan_args_from_query(qs)
                self._json(benchmark_rebench_plan(args.suite, args.results_root, args.port, args.model, args.correction_issue))
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
        elif parsed.path == "/tutorial":
            topic = qs.get("topic", ["overview"])[0]
            if topic == "list":
                self._json({"topics": sorted(TUTORIALS)})
            elif topic not in TUTORIALS:
                self._json({"error": f"unknown topic {topic}", "topics": sorted(TUTORIALS)}, 404)
            else:
                self._json({"topic": topic, "lines": TUTORIALS[topic]})
        elif parsed.path == "/commands":
            self._json(
                {
                    "safe_read_only": {
                        "watch": "cerebellum watch RUN_DIR",
                        "status": "cerebellum status RUN_DIR",
                        "events": "cerebellum events RUN_DIR --limit 50",
                        "recover": "cerebellum recover RUN_DIR --json",
                        "report": "cerebellum report RUN_DIR --json",
                        "export_ai": "cerebellum export RUN_DIR --kind ai",
                        "imatrix": "cerebellum imatrix --model HF_OR_PATH --family FAMILY --model-name MODEL --source-name SOURCE",
                        "provenance": "cerebellum provenance --run-dir RUN_DIR",
                        "queue_list": "cerebellum queue list --json",
                        "queue_get": "cerebellum queue get JOB_ID --json",
                        "pipeline_plan": "cerebellum pipeline-plan --source-gguf MODEL.gguf --output-dir OUT --json",
                        "pipeline_run": "cerebellum pipeline-run --manifest pipeline.json --json",
                        "pipeline_status": "cerebellum pipeline-status --manifest pipeline.json --json",
                        "benchmark_plan": "cerebellum benchmark-plan --suite release --model MODEL --json",
                        "benchmark_status": "cerebellum benchmark-status --results-dir benchmark_results --json",
                        "cpu_offload_smoke": "cerebellum cpu-offload-smoke --source-gguf GLM.gguf --output-dir OUT --skip-inspect --json",
                        "cpu_offload_build_plan": "cerebellum cpu-offload-build-plan --source-gguf GLM.gguf --output-dir OUT --skip-inspect --json",
                        "benchmark_manifest": "cerebellum benchmark-manifest benchmark_results --suite release --model MODEL --json",
                        "benchmark_audit": "cerebellum benchmark-audit benchmark_results --json",
                        "benchmark_report": "cerebellum benchmark-report benchmark_results --leaderboard --suite frontier --json",
                        "inspect_gguf_types": "cerebellum inspect-gguf-types MODEL.gguf --by-component --by-layer --json",
                        "compare_gguf_types": "cerebellum compare-gguf-types BASE.gguf CANDIDATE.gguf --json",
                        "artifact_inventory": "cerebellum artifact-inventory ROOT --json",
                        "history": "cerebellum history --query gemma4 --include-chat-logs --json",
                        "public_export_plan": "cerebellum public-export OUT --dry-run --json",
                        "release_gate": "cerebellum release-gate README.md docs benchmark_results --remote origin --benchmark-results benchmark_results --require-benchmarks --json",
                        "benchmark_rebench_plan": "cerebellum benchmark-rebench-plan --suite humaneval --json",
                        "hf_stats": "cerebellum hf-stats --author deucebucket --json",
                    },
                    "state_changing_cli_only": {
                        "run": "cerebellum run --source-gguf MODEL.gguf --profile wiki --family FAMILY --model-name MODEL",
                        "resume": "cerebellum resume RUN_DIR --low-space",
                        "cleanup_partials": "cerebellum cleanup RUN_DIR --partials --yes",
                        "rollback_layer": "cerebellum rollback RUN_DIR --last-completed-layer --yes",
                        "backup": "cerebellum backup RUN_DIR --to BACKUP_ROOT",
                        "queue_add": "cerebellum queue add --kind pipeline --manifest pipeline.json",
                        "queue_cancel": "cerebellum queue cancel JOB_ID --reason stale",
                        "queue_retry": "cerebellum queue retry JOB_ID --priority 10",
                        "queue_run_next": "cerebellum queue run-next --execute",
                        "benchmark_run": "cerebellum benchmark-run --suite frontier --model MODEL --results-dir benchmark_results --execute",
                        "benchmark_ingest": "cerebellum benchmark-ingest benchmark_results --db db/cerebellum.db --suite release --model MODEL --require-complete",
                        "cpu_offload_smoke": "cerebellum cpu-offload-smoke --source-gguf GLM.gguf --output-dir OUT --json",
                        "finalize": "cerebellum finalize --run-dir RUN_DIR --gguf MODEL.gguf",
                    },
                    "api": {
                        "health": "/health",
                        "runs": "/runs",
                        "run": "/run?run_dir=RUN_DIR",
                        "recover": "/recover?run_dir=RUN_DIR",
                        "queue": "/queue?status=queued",
                        "queue_job": "/queue/job?id=1&tail=40",
                        "pipeline_plan": "/pipeline-plan?source_gguf=MODEL.gguf&output_dir=OUT",
                        "pipeline_run": "/pipeline-run?manifest=pipeline.json",
                        "pipeline_status": "/pipeline-status?manifest=pipeline.json",
                        "benchmark_plan": "/benchmark-plan?suite=release&model=MODEL",
                        "benchmark_status": "/benchmark-status?results_dir=benchmark_results",
                        "cpu_offload_smoke": "/cpu-offload-smoke?source_gguf=GLM.gguf&output_dir=OUT&skip_inspect=true",
                        "benchmark_manifest": "/benchmark-manifest?path=benchmark_results&suite=release&model=MODEL",
                        "benchmark_audit": "/benchmark-audit?path=benchmark_results",
                        "benchmark_report": "/benchmark-report?path=benchmark_results&leaderboard=true&suite=frontier",
                        "inspect_gguf_types": "/inspect-gguf-types?gguf=MODEL.gguf&by_component=true&by_layer=true",
                        "compare_gguf_types": "/compare-gguf-types?baseline=BASE.gguf&candidate=CANDIDATE.gguf",
                        "artifact_inventory": "/artifact-inventory?root=ROOT&top=25",
                        "history": "/history?query=gemma4&limit=5000&include_chat_logs=true",
                        "public_export_plan": "/public-export-plan?path=README.md&path=docs",
                        "benchmark_rebench_plan": "/benchmark-rebench-plan?suite=humaneval",
                        "hf_stats": "/hf-stats?author=deucebucket",
                        "tutorial": "/tutorial?topic=overview",
                    },
                }
            )
        elif parsed.path == "/schema":
            self._json(
                {
                    "schema": "cerebellum.api.v1",
                    "safety": "GET endpoints are read-only; state-changing actions are exposed as CLI command templates only.",
                    "endpoints": [
                        {"path": "/health", "params": [], "returns": "service health"},
                        {"path": "/runs", "params": ["family?", "model?", "status?"], "returns": "runs under data_root"},
                        {"path": "/projects", "params": ["family?", "model?", "source?"], "returns": "Cerebellum model projects under data_root"},
                        {"path": "/run", "params": ["run_dir"], "returns": "manifest and state"},
                        {"path": "/events", "params": ["run_dir", "limit?", "type?"], "returns": "event log rows"},
                        {"path": "/measurements", "params": ["run_dir", "limit?"], "returns": "candidate measurement rows"},
                        {"path": "/report", "params": ["run_dir"], "returns": "summary report"},
                        {"path": "/export", "params": ["run_dir", "kind=ai|infographic|raw"], "returns": "AI/infographic/raw payload"},
                        {"path": "/recover", "params": ["run_dir"], "returns": "crash recovery plan and safe commands"},
                        {"path": "/provenance", "params": ["run_dir?", "gguf?"], "returns": "generated/existing cerebellum metadata"},
                        {"path": "/package", "params": ["run_dir"], "returns": "package/upload manifest"},
                        {"path": "/queue", "params": ["status?", "kind?", "limit?"], "returns": "queued Cerebellum jobs"},
                        {"path": "/queue/job", "params": ["id", "tail?"], "returns": "single queued Cerebellum job with optional bounded log tail"},
                        {"path": "/system", "params": [], "returns": "host resources and tool availability"},
                        {"path": "/space", "params": ["source_gguf", "scratch?", "margin_gb?"], "returns": "scratch-space plan"},
                        {"path": "/pipeline-plan", "params": ["source_gguf", "output_dir", "task_profile?", "benchmark_suite?"], "returns": "pipeline phase manifest"},
                        {"path": "/pipeline-run", "params": ["manifest", "from_phase?", "until_phase?"], "returns": "pipeline dry-run phase validation"},
                        {"path": "/pipeline-status", "params": ["manifest", "events?"], "returns": "pipeline execution status and resume command from event logs"},
                        {"path": "/benchmark-plan", "params": ["suite?", "model?", "port?", "results_dir?"], "returns": "benchmark suite command/artifact readiness plan"},
                        {"path": "/benchmark-status", "params": ["results_dir?", "events?"], "returns": "benchmark execution status and rerun command from event logs"},
                        {"path": "/cpu-offload-smoke", "params": ["source_gguf", "output_dir", "model_name?", "scratch_root?", "benchmark_port?", "margin_gb?", "skip_inspect?", "require_inspect?"], "returns": "read-only huge-model CPU-offload preflight smoke payload"},
                        {"path": "/benchmark-manifest", "params": ["path", "suite?", "model?", "require_complete?"], "returns": "hashed benchmark artifact manifest"},
                        {"path": "/benchmark-audit", "params": ["path", "fail_empty_pct?", "fail_unknown_pct?", "fail_pass_only_pct?"], "returns": "benchmark detailed-artifact quality audit"},
                        {"path": "/benchmark-report", "params": ["path", "baseline?", "leaderboard?", "suite?", "size?", "size_json?", "weight?", "list_suites?"], "returns": "benchmark aggregate comparison and leaderboard report"},
                        {"path": "/inspect-gguf-types", "params": ["gguf", "by_layer?", "by_component?"], "returns": "GGUF tensor type inventory by type, component, layer, and tensor"},
                        {"path": "/compare-gguf-types", "params": ["baseline", "candidate", "baseline_label?", "candidate_label?", "reference_map?"], "returns": "GGUF tensor type comparison and Dynamic GGUF profile"},
                        {"path": "/artifact-inventory", "params": ["root", "top?", "allow_broad?"], "returns": "preservation-first legacy artifact inventory"},
                        {"path": "/history", "params": ["root?", "query?", "limit?", "include_chat_logs?", "chat_root?"], "returns": "searchable Cerebellum model/method/benchmark/chat-evidence history index"},
                        {"path": "/public-export-plan", "params": ["path?", "paths?", "max_bytes?"], "returns": "sanitized public export dry-run manifest"},
                        {"path": "/benchmark-rebench-plan", "params": ["suite=humaneval|release?", "results_root?", "port?", "model?", "correction_issue?"], "returns": "published-model corrected rebench queue"},
                        {"path": "/hf-stats", "params": ["author?", "period=recent|all-time?", "publisher_org?", "limit?"], "returns": "HF model release telemetry"},
                        {"path": "/tutorial", "params": ["topic"], "returns": "tutorial lines"},
                        {"path": "/self-test", "params": ["run_dir?"], "returns": "read-only CLI/API smoke-check payload"},
                        {"path": "/commands", "params": [], "returns": "CLI command templates"},
                        {"path": "/schema", "params": [], "returns": "this API catalog"},
                        {"path": "/db/families", "params": [], "returns": "indexed model families"},
                    ],
                    "tutorial_topics": sorted(TUTORIALS),
                }
            )
        elif parsed.path == "/self-test":
            run_dir = qs.get("run_dir", [None])[0]
            self._json(self_test_payload(Path(run_dir) if run_dir else None))
        else:
            self._json({"error": "not found"}, 404)


def api_cmd(args: argparse.Namespace) -> None:
    CerebellumAPI.data_root = Path(args.data_root) if args.data_root else default_data_root()
    CerebellumAPI.db_path = Path(args.db)
    server = ThreadingHTTPServer((args.host, args.port), CerebellumAPI)
    print(f"Cerebellum API: http://{args.host}:{args.port}")
    print("Endpoints: /health /runs /projects /run /events /measurements /report /export /recover /provenance /package /queue /queue/job /system /space /pipeline-plan /pipeline-run /pipeline-status /benchmark-plan /benchmark-status /cpu-offload-smoke /benchmark-manifest /benchmark-audit /benchmark-report /inspect-gguf-types /compare-gguf-types /artifact-inventory /history /public-export-plan /benchmark-rebench-plan /tutorial /self-test /commands /schema /db/families")
    server.serve_forever()


def schedule_cmd(args: argparse.Namespace) -> None:
    if args.template:
        template = {
            "jobs": [
                {
                    "source_gguf": "/models/model-f16.gguf",
                    "profile": "wiki",
                    "family": "example-family",
                    "model_name": "example-model",
                    "source_name": "local-f16",
                    "data_root": str(default_data_root()),
                    "scratch_root": "/large/scratch/cerebellum",
                    "base_type": "Q4_K_M",
                    "start_type": "q4_K",
                    "levels": "q3_K,q2_K,q5_K,q6_K,f16",
                    "quantize_bin": "llama-quantize",
                    "perplexity_bin": "llama-perplexity",
                    "gpu_layers": 99,
                    "ctx_size": 2048,
                    "chunks": 128,
                    "min_free_gb": 40.0,
                    "distrobox": None,
                }
            ]
        }
        print(json.dumps(template, indent=2, sort_keys=True))
        return
    if not args.file:
        raise SystemExit("schedule requires --file, or use --template")
    data = json.loads(Path(args.file).read_text())
    jobs = data.get("jobs", data if isinstance(data, list) else [])
    if not isinstance(jobs, list):
        raise SystemExit("schedule file must be a JSON list or {\"jobs\": [...]}")
    for idx, job in enumerate(jobs, 1):
        print(f"=== schedule job {idx}/{len(jobs)}: {job.get('model_name') or job.get('source_gguf')} ===")
        ns = argparse.Namespace(
            cmd="run",
            source_gguf=job["source_gguf"],
            corpus=job.get("corpus"),
            profile=job.get("profile", "custom"),
            metric=job.get("metric", "ppl"),
            family=job.get("family"),
            model_name=job.get("model_name"),
            source_name=job.get("source_name"),
            data_root=job.get("data_root"),
            run_name=job.get("run_name"),
            run_dir=job.get("run_dir"),
            tensor_file=job.get("tensor_file"),
            base_map=job.get("base_map"),
            layers=job.get("layers"),
            tensor_regex=job.get("tensor_regex"),
            scratch_root=job.get("scratch_root"),
            backup_root=job.get("backup_root"),
            base_type=job.get("base_type", "Q4_K_M"),
            start_type=job.get("start_type", "q4_K"),
            levels=job.get("levels", ",".join(DEFAULT_LEVELS)),
            imatrix=job.get("imatrix"),
            quantize_bin=job.get("quantize_bin", DEFAULT_QUANTIZE),
            perplexity_bin=job.get("perplexity_bin", DEFAULT_PERPLEXITY),
            gpu_layers=job.get("gpu_layers", 99),
            ctx_size=job.get("ctx_size", 2048),
            chunks=job.get("chunks"),
            max_temp_gb=job.get("max_temp_gb", 80.0),
            min_free_gb=job.get("min_free_gb", 40.0),
            hard_free_floor_gb=job.get("hard_free_floor_gb", 10.0),
            distrobox=job.get("distrobox"),
            quant_timeout=job.get("quant_timeout", 1800),
            ppl_timeout=job.get("ppl_timeout", 900),
            keep_losers=job.get("keep_losers", False),
            no_keep_winners=job.get("no_keep_winners", False),
            low_space=job.get("low_space", False),
            serial_candidates=job.get("serial_candidates", False),
            prune_measured_candidates=job.get("prune_measured_candidates", True),
            plain=job.get("plain", False),
            no_color=job.get("no_color", False),
            backup_every=job.get("backup_every", 1),
            token_embedding_type=job.get("token_embedding_type", "f16"),
            noise_pct=job.get("noise_pct", 0.0),
        )
        if args.dry_run:
            ns.run_dir = str(build_run_dir(ns))
            ns.resolved_corpus = str(resolve_ppl_corpus(ns.profile, ns.corpus))
            print(json.dumps(vars(ns), indent=2, sort_keys=True))
            continue
        run_from_namespace(ns)


def run_from_namespace(args: argparse.Namespace) -> None:
    metric = getattr(args, "metric", "ppl")
    if metric != "ppl":
        raise SystemExit(
            f"ablation metric {metric!r} is not executable yet; current Cerebellum run scoring supports only 'ppl'"
        )
    run_dir = build_run_dir(args)
    run_id = slug(args.run_name or run_dir.name)
    cfg = Config(
        source_gguf=Path(args.source_gguf),
        corpus=resolve_ppl_corpus(args.profile, args.corpus),
        ppl_profile=args.profile,
        ablation_metric=metric,
        run_dir=run_dir,
        run_id=run_id,
        model_family=slug(args.family or "unknown-family"),
        model_name=slug(args.model_name or Path(args.source_gguf).stem),
        source_name=slug(args.source_name or Path(args.source_gguf).stem),
        base_type=args.base_type,
        start_type=args.start_type,
        levels=[level.strip() for level in args.levels.split(",") if level.strip()],
        imatrix=Path(args.imatrix) if args.imatrix else None,
        tensor_file=Path(args.tensor_file) if args.tensor_file else None,
        base_map=Path(args.base_map) if getattr(args, "base_map", None) else None,
        layers=parse_layer_spec(args.layers),
        tensor_regex=args.tensor_regex,
        scratch_root=Path(args.scratch_root) if args.scratch_root else None,
        backup_root=Path(args.backup_root) if args.backup_root else None,
        quantize_bin=args.quantize_bin,
        perplexity_bin=args.perplexity_bin,
        pure_quant=args.pure_quant,
        commit_locks=args.commit_locks,
        gpu_layers=args.gpu_layers,
        ctx_size=args.ctx_size,
        chunks=args.chunks,
        max_temp_gb=args.max_temp_gb,
        min_free_gb=args.min_free_gb,
        hard_free_floor_gb=args.hard_free_floor_gb,
        keep_winners=not args.no_keep_winners,
        keep_losers=args.keep_losers,
        low_space=args.low_space,
        serial_candidates=args.serial_candidates,
        prune_measured_candidates=args.prune_measured_candidates,
        distrobox=args.distrobox,
        quant_timeout=args.quant_timeout,
        ppl_timeout=args.ppl_timeout,
        color=not args.no_color and sys.stdout.isatty(),
        plain=args.plain,
        backup_every=max(1, args.backup_every),
        token_embedding_type=args.token_embedding_type,
        noise_pct=args.noise_pct,
    )
    HillStepper(cfg).run()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cmd is None:
        home_cmd(argparse.Namespace(limit=8, json=False))
        return
    if args.cmd == "home":
        home_cmd(args)
        return
    if args.cmd == "imatrix":
        raise SystemExit("use the public cerebellum CLI entrypoint for imatrix generation")
    if args.cmd == "status":
        status_cmd(args)
        return
    if args.cmd == "events":
        events_cmd(args)
        return
    if args.cmd == "watch":
        watch_cmd(args)
        return
    if args.cmd == "stop":
        stop_cmd(args)
        return
    if args.cmd == "resume":
        resume_cmd(args)
        return
    if args.cmd == "cleanup":
        cleanup_cmd(args)
        return
    if args.cmd == "rollback":
        rollback_cmd(args)
        return
    if args.cmd == "backup":
        backup_cmd(args)
        return
    if args.cmd == "recover":
        recover_cmd(args)
        return
    if args.cmd == "runs":
        runs_cmd(args)
        return
    if args.cmd == "project":
        project_cmd(args)
        return
    if args.cmd == "provenance":
        provenance_cmd(args)
        return
    if args.cmd == "inspect-gguf-types":
        inspect_gguf_types_cmd(args)
        return
    if args.cmd == "compare-gguf-types":
        compare_gguf_types_cmd(args)
        return
    if args.cmd == "compare-locks":
        compare_locks_cmd(args)
        return
    if args.cmd == "finalize":
        finalize_cmd(args)
        return
    if args.cmd == "package":
        package_cmd(args)
        return
    if args.cmd == "public-audit":
        public_audit_cmd(args)
        return
    if args.cmd == "public-history-audit":
        public_history_audit_cmd(args)
        return
    if args.cmd == "public-export":
        public_export_cmd(args)
        return
    if args.cmd == "release-gate":
        release_gate_cmd(args)
        return
    if args.cmd == "artifact-inventory":
        artifact_inventory_cmd(args)
        return
    if args.cmd == "history":
        history_cmd(args)
        return
    if args.cmd == "schedule":
        schedule_cmd(args)
        return
    if args.cmd == "queue":
        queue_cmd(args)
        return
    if args.cmd == "pipeline-plan":
        pipeline_plan_cmd(args)
        return
    if args.cmd == "pipeline-run":
        pipeline_run_cmd(args)
        return
    if args.cmd == "pipeline-status":
        pipeline_status_cmd(args)
        return
    if args.cmd == "cpu-offload-smoke":
        cpu_offload_smoke_cmd(args)
        return
    if args.cmd == "cpu-offload-build-plan":
        cpu_offload_build_plan_cmd(args)
        return
    if args.cmd == "task-profiles":
        task_profiles_cmd(args)
        return
    if args.cmd == "legacy-plan":
        legacy_plan_cmd(args)
        return
    if args.cmd == "legacy-flow":
        legacy_flow_cmd(args)
        return
    if args.cmd == "group-scan":
        group_scan_cmd(args)
        return
    if args.cmd == "sparse-replay":
        sparse_replay_cmd(args)
        return
    if args.cmd == "public-card-policy":
        public_model_card_policy_cmd(args)
        return
    if args.cmd == "system":
        system_cmd(args)
        return
    if args.cmd == "doctor":
        doctor_cmd(args)
        return
    if args.cmd == "self-test":
        self_test_cmd(args)
        return
    if args.cmd == "plan-space":
        plan_space_cmd(args)
        return
    if args.cmd == "tutorial":
        tutorial_cmd(args)
        return
    if args.cmd == "tips":
        tips_cmd(args)
        return
    if args.cmd == "db":
        db_cmd(args)
        return
    if args.cmd == "report":
        report_cmd(args)
        return
    if args.cmd == "benchmark-report":
        benchmark_report_cmd(args)
        return
    if args.cmd == "benchmark-plan":
        benchmark_plan_cmd(args)
        return
    if args.cmd == "benchmark-run":
        benchmark_run_cmd(args)
        return
    if args.cmd == "benchmark-postprocess":
        benchmark_postprocess_cmd(args)
        return
    if args.cmd == "benchmark-ingest":
        benchmark_ingest_cmd(args)
        return
    if args.cmd == "benchmark-status":
        benchmark_status_cmd(args)
        return
    if args.cmd == "benchmark-rebench-plan":
        benchmark_rebench_plan_cmd(args)
        return
    if args.cmd == "benchmark-manifest":
        benchmark_manifest_cmd(args)
        return
    if args.cmd == "benchmark-audit":
        benchmark_audit_cmd(args)
        return
    if args.cmd == "ablation-analyze":
        ablation_analyze_cmd(args)
        return
    if args.cmd == "export":
        export_cmd(args)
        return
    if args.cmd == "auth":
        auth_cmd(args)
        return
    if args.cmd == "hf-stats":
        hf_stats_cmd(args)
        return
    if args.cmd == "upload":
        upload_cmd(args)
        return
    if args.cmd == "api":
        api_cmd(args)
        return
    if args.cmd != "run":
        args.cmd = "run"
    run_from_namespace(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
