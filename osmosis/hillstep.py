"""Cerebellum quantization engine.

This is the durable, resumable CLI engine for per-tensor quantization search.
It overlaps CPU quantization with GPU perplexity measurement for each tensor,
records every observable event, and keeps enough state on disk to recover after
process death or a system lockup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import re
import shutil
import sqlite3
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
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
EVENT_SCHEMA_VERSION = 1
EVENT_FILES = ("cerebellum_events.jsonl", "cerebellum_hill_events.jsonl")
CANDIDATE_FILES = ("cerebellum_candidates.jsonl", "cerebellum_hill_candidates.jsonl")
SUMMARY_JSON_FILES = ("cerebellum_summary.json", "cerebellum_hill_summary.json")
SUMMARY_MD_FILES = ("cerebellum_summary.md", "cerebellum_hill_summary.md")
DECISION_CSV_FILES = ("cerebellum_decisions.csv", "cerebellum_hill_decisions.csv")
INFOGRAPHIC_FILES = ("cerebellum_infographic_data.json", "cerebellum_hill_infographic_data.json")
BEST_TYPES_FILES = ("cerebellum_best_tensor_types.txt", "cerebellum_hill_best_tensor_types.txt")
CURRENT_TYPES_FILE = "cerebellum_current_tensor_types.txt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slug(value: str) -> str:
    value = value.strip().replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-") or "unknown"


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
        self._event_id = 0
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
    return "37;1"


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


def fmt_bytes(size: int | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


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
        if "cerebellum run" in cmd:
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
                "cmd": cmd,
            }
        )
    rows.sort(key=lambda row: {"runner": 0, "quantize": 1, "ppl": 2, "container": 3}.get(row["kind"], 9))
    return rows


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
        "confidence": confidence,
    }


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


def path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


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
    return sorted(root.glob("families/*/*/sources/*/runs/*/manifest.json"))


def default_data_root() -> Path:
    if os.environ.get("CEREBELLUM_DATA_ROOT"):
        return Path(os.environ["CEREBELLUM_DATA_ROOT"])
    return Path.home() / "cerebellum-runs"


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
        """
    )


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


def parse_tensor_name(tensor: str) -> tuple[int | None, str | None]:
    match = re.match(r"blk\.(\d+)\.(.+)\.weight$", tensor)
    if match:
        return int(match.group(1)), match.group(2)
    if tensor.endswith(".weight"):
        return None, tensor[:-7]
    return None, None


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
    scratch_root: Path | None = None
    max_temp_gb: float = 80.0
    min_free_gb: float = 40.0
    keep_winners: bool = True
    keep_losers: bool = False
    distrobox: str | None = None
    quant_timeout: int = 1800
    ppl_timeout: int = 900
    color: bool = True
    plain: bool = False
    backup_every: int = 1
    token_embedding_type: str | None = "f16"
    noise_pct: float = 0.0


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
            "base_type": self.cfg.base_type,
            "start_type": self.cfg.start_type,
            "levels": self.cfg.levels,
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
            "run_dir": str(self.cfg.run_dir),
            "base_type": self.cfg.base_type,
            "start_type": self.cfg.start_type,
            "levels": self.cfg.levels,
            "quantize_bin": self.cfg.quantize_bin,
            "perplexity_bin": self.cfg.perplexity_bin,
            "gpu_layers": self.cfg.gpu_layers,
            "ctx_size": self.cfg.ctx_size,
            "chunks": self.cfg.chunks,
            "imatrix": str(self.cfg.imatrix) if self.cfg.imatrix else None,
            "scratch_root": str(self.cfg.scratch_root) if self.cfg.scratch_root else None,
            "distrobox": self.cfg.distrobox,
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
            return [line.strip() for line in self.cfg.tensor_file.read_text().splitlines() if line.strip()]
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
            if "weight" not in name:
                continue
            if t.n_bytes < 1000 or "rope" in name or "embd" in name or "output_norm" in name:
                continue
            match = re.match(r"blk\.(\d+)\.(.+)\.weight", name)
            layer = int(match.group(1)) if match else -1
            ttype = match.group(2) if match else name.replace(".weight", "")
            tensors.append((layer, priority.get(ttype, 99), name))
        tensors.sort()
        return [name for _, _, name in tensors]

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
        print(f"Tensors : {locked}/{tensors} locked")
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
            names = sorted(set(names) | set(locked) | set(extra))
        else:
            try:
                from gguf import GGUFReader
                reader = GGUFReader(str(self.cfg.source_gguf))
                names = [t.name for t in reader.tensors]
            except ImportError:
                names = sorted(set(locked) | set(extra))
            except Exception:
                names = sorted(set(locked) | set(extra))
        if not names:
            names = sorted(set(locked) | set(extra))
        lines = []
        merged = dict(locked)
        merged.update(extra)
        for name in names:
            qtype = merged.get(name, self.cfg.start_type)
            lines.append(f"{name}={qtype}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def write_all_types_from_source(self, locked: dict[str, str], path: Path, extra: dict[str, str] | None = None) -> None:
        """Write a complete tensor map when a source GGUF is readable.

        Kept separate from write_types() so tests and explicit tensor-file runs
        can operate without parsing the GGUF.
        """
        extra = extra or {}
        try:
            from gguf import GGUFReader
            reader = GGUFReader(str(self.cfg.source_gguf))
            names = [t.name for t in reader.tensors]
        except Exception:
            names = sorted(set(locked) | set(extra))
        lines = []
        merged = dict(locked)
        merged.update(extra)
        for name in names:
            qtype = merged.get(name, self.cfg.start_type)
            lines.append(f"{name}={qtype}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def quantize_cmd(self, type_file: Path, outfile: Path) -> list[str]:
        cmd = [self.cfg.quantize_bin, "--allow-requantize"]
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
        if self.paths.baseline.exists() and state.get("current_ppl") is not None:
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
            output_tail=output[-2000:],
        )
        if rc != 0:
            raise SystemExit("baseline quantization failed; see events.jsonl")
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

        ready: queue.Queue[Candidate | None] = queue.Queue(maxsize=2)
        results: list[Candidate] = []
        quant_done = threading.Event()

        def quant_worker() -> None:
            for c in candidates:
                if self.stop_requested:
                    break
                while disk_free_gb(self.paths.tmp) < self.cfg.min_free_gb and not self.stop_requested:
                    self.events.write("disk_wait", free_gb=disk_free_gb(self.paths.tmp), min_free_gb=self.cfg.min_free_gb)
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

        best_level, best_ppl, best_candidate, winner_reason = self.choose_winner(self.cfg.start_type, baseline_ppl, candidates)
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
            }
        )
        for c in candidates:
            state["totals"]["quant_seconds"] += c.quant_seconds or 0
            state["totals"]["ppl_seconds"] += c.ppl_seconds or 0
            state["totals"]["candidates"] += 1
            if c.status == "failed":
                state["totals"]["failures"] += 1

        old_baseline = self.paths.baseline
        if best_candidate is not None and best_candidate.gguf_path.exists():
            old_backup = old_baseline.with_suffix(".previous.gguf")
            if old_baseline.exists():
                os.replace(old_baseline, old_backup)
            os.replace(best_candidate.gguf_path, old_baseline)
            if old_backup.exists():
                old_backup.unlink()
        self.write_types(state["locked"], self.paths.current_types)
        self.save_state(state, checkpoint=(len(state["locked"]) % self.cfg.backup_every == 0))
        self.events.write("tensor_locked", tensor=tensor, winner=best_level, ppl=best_ppl, baseline_ppl=baseline_ppl, reason=winner_reason)

        for c in candidates:
            if c is best_candidate and self.cfg.keep_winners:
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
        remaining = [t for t in tensors if t not in state["locked"]]
        state["run_status"] = "running"
        self.save_state(state)
        self.render_banner(len(tensors), len(state["locked"]))
        self.events.write("run_start", tensors=len(tensors), locked=len(state["locked"]))
        self.build_baseline_if_needed(state)
        for tensor in remaining:
            if self.stop_requested:
                break
            idx = tensors.index(tensor) + 1
            self.test_tensor(state, tensor, idx, len(tensors))
        state["run_status"] = "complete" if len(state["locked"]) == len(tensors) else "stopped"
        self.write_types(state["locked"], self.paths.final_types)
        atomic_write_json(self.paths.timing, state["totals"])
        self.save_state(state, checkpoint=True)
        marker = self.cfg.run_dir / ("COMPLETE" if state["run_status"] == "complete" else "ABORTED")
        tmp_marker = marker.with_suffix(".tmp")
        tmp_marker.write_text(utc_now() + "\n")
        os.replace(tmp_marker, marker)
        self.events.write("run_finish", status=state["run_status"], locked=len(state["locked"]), tensors=len(tensors))


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
    run = sub.add_parser("run", help="run or resume a Cerebellum quant search")
    run.add_argument("--source-gguf", required=True)
    run.add_argument("--corpus", default=None, help="PPL/calibration corpus path; optional when --profile resolves locally")
    run.add_argument(
        "--profile",
        choices=["wiki", "agentic", "code", "math", "dialogue", "all-around", "custom"],
        default="custom",
        help="Named PPL target profile recorded with the run",
    )
    run.add_argument("--family", default=None)
    run.add_argument("--model-name", default=None)
    run.add_argument("--source-name", default=None)
    run.add_argument("--data-root", default=None)
    run.add_argument("--run-name", default=None)
    run.add_argument("--run-dir", default=None)
    run.add_argument("--tensor-file", default=None)
    run.add_argument("--scratch-root", default=None, help="Large GGUF artifact/temp root, separate from metadata run dir")
    run.add_argument("--base-type", default="Q4_K_M")
    run.add_argument("--start-type", default="q4_K")
    run.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    run.add_argument("--imatrix", default=None)
    run.add_argument("--quantize-bin", default=DEFAULT_QUANTIZE)
    run.add_argument("--perplexity-bin", default=DEFAULT_PERPLEXITY)
    run.add_argument("--gpu-layers", type=int, default=99)
    run.add_argument("--ctx-size", type=int, default=2048)
    run.add_argument("--chunks", type=int, default=None)
    run.add_argument("--max-temp-gb", type=float, default=80.0, help="Minimum free GB required before launching next quant")
    run.add_argument("--min-free-gb", type=float, default=40.0, help="Minimum free GB required before launching next quant")
    run.add_argument("--distrobox", default=None, help="Run llama.cpp commands inside this distrobox")
    run.add_argument("--quant-timeout", type=int, default=1800)
    run.add_argument("--ppl-timeout", type=int, default=900)
    run.add_argument("--keep-losers", action="store_true")
    run.add_argument("--no-keep-winners", action="store_true")
    run.add_argument("--plain", action="store_true")
    run.add_argument("--no-color", action="store_true")
    run.add_argument("--backup-every", type=int, default=1)
    run.add_argument("--token-embedding-type", default="f16")
    run.add_argument("--noise-pct", type=float, default=0.0)

    status = sub.add_parser("status", help="show Cerebellum run status")
    status.add_argument("run_dir")
    status.add_argument("--plain", action="store_true")
    status.add_argument("--no-color", action="store_true")

    events = sub.add_parser("events", help="print run events")
    events.add_argument("run_dir")
    events.add_argument("--type", default=None)
    events.add_argument("--limit", type=int, default=50)
    events.add_argument("--json", action="store_true")

    watch = sub.add_parser("watch", help="open the Cerebellum live terminal interface")
    watch.add_argument("run_dir")
    watch.add_argument("--interval", type=float, default=5.0)
    watch.add_argument("--once", action="store_true", help="render one frame and exit")
    watch.add_argument("--stall-warn-seconds", type=float, default=300.0)
    watch.add_argument("--stall-fail-seconds", type=float, default=900.0)
    watch.add_argument("--events-limit", type=int, default=12)
    watch.add_argument("--measurements-limit", type=int, default=8)
    watch.add_argument("--tui", action="store_true", help="open scrollable interactive terminal UI")
    watch.add_argument("--plain", action="store_true")
    watch.add_argument("--no-color", action="store_true")

    stop = sub.add_parser("stop", help="stop or repair a Cerebellum run state")
    stop.add_argument("run_dir")
    stop.add_argument("--reason", default="user")
    stop.add_argument("--no-kill", action="store_true", help="only mark state stopped; do not signal a process")

    runs = sub.add_parser("runs", help="list known runs under a data root")
    runs.add_argument("--data-root", default=None)
    runs.add_argument("--family", default=None)
    runs.add_argument("--model", default=None)
    runs.add_argument("--status", default=None)
    runs.add_argument("--profile", default=None)
    runs.add_argument("--json", action="store_true")

    provenance = sub.add_parser("provenance", help="inspect or generate Cerebellum GGUF provenance metadata")
    provenance.add_argument("--gguf", default=None, help="GGUF to inspect for existing metadata")
    provenance.add_argument("--run-dir", default=None, help="Cerebellum run directory used to generate metadata")
    provenance.add_argument("--hash-files", action="store_true", help="compute full SHA256 hashes for large files")
    provenance.add_argument("--format", choices=["json", "env"], default="json")

    finalize = sub.add_parser("finalize", help="write final reports/model card and tag GGUF provenance")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--gguf", default=None, help="Final GGUF to tag/inspect")
    finalize.add_argument("--repo-name", default=None, help="Optional HF/GitHub repo name for model-card text")
    finalize.add_argument("--output-dir", default=None, help="Defaults to RUN_DIR/finalize")
    finalize.add_argument("--hash-files", action="store_true")
    finalize.add_argument("--inject", action="store_true", help="Inject visible cerebellum.* metadata into --gguf when supported")
    finalize.add_argument("--metadata-tool", default=None, help="Path to gguf-set-metadata compatible tool")
    finalize.add_argument("--json", action="store_true")

    schedule = sub.add_parser("schedule", help="run multiple Cerebellum jobs from a JSON schedule")
    schedule.add_argument("--file", default=None)
    schedule.add_argument("--template", action="store_true", help="print an example schedule JSON")
    schedule.add_argument("--dry-run", action="store_true", help="validate and print jobs without running them")

    system = sub.add_parser("system", help="inspect local resources and tool availability")
    system.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor", help="check portable Cerebellum setup and explain fixes")
    doctor.add_argument("--json", action="store_true")

    plan_space = sub.add_parser("plan-space", help="recommend low-space quant scratch strategy")
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

    export = sub.add_parser("export", help="export run data for AI, infographic, or automation")
    export.add_argument("run_dir")
    export.add_argument("--kind", choices=["raw", "ai", "infographic"], default="ai")
    export.add_argument("--output")

    auth = sub.add_parser("auth", help="check HF/GitHub auth status")
    auth.add_argument("service", choices=["hf", "huggingface", "github"])

    upload = sub.add_parser("upload", help="upload Cerebellum artifacts to HF/GitHub")
    upload.add_argument("target", choices=["hf", "huggingface", "github"])
    upload.add_argument("run_dir")
    upload.add_argument("--repo")
    upload.add_argument("--repo-type", default="model")
    upload.add_argument("--branch")
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
            "run", "status", "events", "runs", "schedule", "db", "report",
            "export", "auth", "upload", "api", "system", "doctor", "provenance", "finalize", "plan-space",
            "tutorial", "tips", "watch", "stop", "--help", "-h",
        }
    ):
        argv = ["run", *sys.argv[1:]]
    return parser.parse_args(argv)


def status_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise SystemExit(f"no state.json found in {run_dir}")
    state = json.loads(state_path.read_text())
    enabled = not args.no_color and not args.plain
    locked = len(state.get("locked", {}))
    tested = state.get("tested", [])
    print(color("Cerebellum status", "36;1", enabled))
    print(f"run_dir     : {run_dir}")
    print(f"status      : {state.get('run_status')}")
    print(f"model       : {state.get('model_family')}/{state.get('model_name')}")
    print(f"locked      : {locked}")
    print(f"current_ppl : {state.get('current_ppl')}")
    print(f"last_tensor : {state.get('last_tensor')}")
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


def stop_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise SystemExit(f"no state.json found in {run_dir}")
    events_path = first_existing(run_dir, EVENT_FILES)
    events = read_jsonl(events_path)
    pids = []
    for row in reversed(events):
        pid = row.get("pid")
        if isinstance(pid, int) and pid not in pids:
            pids.append(pid)
        if len(pids) >= 3:
            break
    signaled: list[int] = []
    if not args.no_kill:
        for pid in pids:
            targets = [*child_pids(pid), pid]
            for target in targets:
                if target == os.getpid() or target in signaled or not process_exists(target):
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
            print(f"║ {ansi_pad(line, width - 4)} ║")
    print(color("╚" + "═" * (width - 2) + "╝", code, enabled))


def grid_watch_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    enabled = not args.no_color and not args.plain
    try:
        while True:
            model = build_watch_model(run_dir)
            state = model["state"]
            manifest = model["manifest"]
            active = model["active"]
            candidates = model["candidates"]
            events = model["events"]
            processes = model["active_processes"]
            gpu_info = model["gpu"]
            terminal_w = shutil.get_terminal_size((118, 40)).columns
            width = max(96, min(132, terminal_w))
            os.system("clear" if os.name != "nt" else "cls")
            run_id = manifest.get("run_id") or state.get("run_id") or run_dir.name
            title = f" CEREBELLUM  {state.get('model_family')}/{state.get('model_name')}  {manifest.get('ppl_profile') or state.get('ppl_profile') or 'custom'}  {state.get('run_status')} "
            print(color("╔" + "═" * (width - 2) + "╗", "36;1", enabled))
            print(color("║" + title.center(width - 2) + "║", "36;1", enabled))
            print(color("╚" + "═" * (width - 2) + "╝", "36;1", enabled))
            print()

            eta = eta_grid_values(state, model["active_age"], next((row.get("total") for row in reversed(events) if row.get("total")), None))
            locked = len(state.get("locked", {}))
            total = next((row.get("total") for row in reversed(events) if row.get("total")), None)
            progress_left = f"{model['bar']} {model['progress']}  ppl {state.get('current_ppl')}"
            resource_bits = []
            if gpu_info:
                gpu = gpu_info[0]
                resource_bits.append(f"GPU {gpu['util']}% {gpu['mem_used']}/{gpu['mem_total']} MiB")
            if processes:
                jobs = ",".join(sorted({row["kind"] for row in processes}))
                resource_bits.append(f"jobs {jobs}")
            else:
                resource_bits.append("jobs idle")
            overview = [
                grid_line(color("progress  ", "90", enabled) + color(progress_left, "32;1", enabled), color("resources  ", "90", enabled) + color("  ".join(resource_bits), "36;1", enabled), width),
                grid_line(color("tensor    ", "90", enabled) + color(f"{active.get('tensor')}  {active.get('level')}", "33;1", enabled), color("disk       ", "90", enabled) + color(f"{disk_free_gb(run_dir):.1f} GiB free", "36", enabled), width),
                grid_line(color("job       ", "90", enabled) + color(f"{active.get('event')}  age {fmt_seconds(model['active_age'])}", "32;1", enabled), color("gguf       ", "90", enabled) + color(f"base {fmt_bytes(model['baseline_size'])} active {fmt_bytes(model['active_size'])}", "36;1", enabled), width),
                grid_line(color("eta       ", "90", enabled) + color(f"current {eta['current']} avg/tensor {eta['avg_tensor']} total {eta['total']}", "36;1", enabled), color("confidence ", "90", enabled) + color(eta["confidence"], "32;1" if eta["confidence"] == "high" else "33;1", enabled), width),
            ]
            print_heavy_box("OPERATIONS", overview, width, "34;1", enabled)
            print()

            measure_lines = [f"{'quant':<7} {'ppl':<12} {'delta':<12} {'size':<10} tensor"]
            measure_lines.append("─" * (width - 4))
            for row in candidates[-max(1, args.measurements_limit) :]:
                delta = row.get("delta")
                delta_s = "-" if delta is None else f"{delta:+.4f}"
                marker = "better" if isinstance(delta, (int, float)) and delta < 0 else "worse" if isinstance(delta, (int, float)) and delta > 0 else ""
                line = (
                    color(f"{row.get('level', '-'):<7}", "35;1", enabled)
                    + color(f"{str(row.get('ppl', '-')):<12}", "33;1", enabled)
                    + color(f"{delta_s:<12}", delta_code(delta), enabled)
                    + color(f"{fmt_bytes(row.get('size_bytes')):<10}", "36", enabled)
                    + color(f"{row.get('tensor', '')} {marker}", "33", enabled)
                )
                measure_lines.append(line)
            print_heavy_box("RECENT MEASUREMENTS", measure_lines, width, "32;1", enabled)
            print()

            event_parts = []
            for row in events[-min(5, max(1, args.events_limit)) :]:
                event_parts.append(f"{row.get('event')} {row.get('level', '')} {row.get('tensor', '')}".strip())
            print_heavy_box("EVENT STRIP", [" | ".join(event_parts), f"run {run_id}", "Ctrl+C exits UI only. Use `cerebellum stop RUN_DIR` to stop."], width, "33;1", enabled)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


def build_watch_model(run_dir: Path) -> dict[str, Any]:
    state = read_json(run_dir / "state.json", {})
    manifest = read_json(run_dir / "manifest.json", {})
    events = read_jsonl(first_existing(run_dir, EVENT_FILES))
    candidates = read_jsonl(first_existing(run_dir, CANDIDATE_FILES))
    status = state.get("run_status")
    terminal_events = {"run_stopped", "run_finish", "tensor_interrupted", "signal_received"}
    if status in {"stopped", "complete", "failed"}:
        active = next((row for row in reversed(events) if row.get("event") in terminal_events), {})
    else:
        active = next((row for row in reversed(events) if row.get("event") in {"tensor_start", "quant_start", "ppl_start"}), {})
    total = next((row.get("total") for row in reversed(events) if row.get("total")), None)
    locked = len(state.get("locked", {}))
    bar, progress = progress_bar(locked, total, width=22)
    active_age = event_age_seconds(active)
    last_age = event_age_seconds(events[-1]) if events else None
    processes = process_rows_for_run(run_dir)
    active_processes = [row for row in processes if row["kind"] in {"quantize", "ppl"}]
    eta, eta_basis = estimate_eta(state, active_age, total)
    baseline_path = Path(state.get("baseline_path") or run_dir / "artifacts" / "current_baseline.gguf")
    active_path = active.get("tmp_output") or active.get("output") or active.get("model")
    return {
        "state": state,
        "manifest": manifest,
        "events": events,
        "candidates": candidates,
        "active": active,
        "processes": processes,
        "active_processes": active_processes,
        "gpu": gpu_rows(),
        "bar": bar,
        "progress": progress,
        "active_age": active_age,
        "last_age": last_age,
        "eta": eta,
        "eta_basis": eta_basis,
        "baseline_path": baseline_path,
        "baseline_size": path_size(baseline_path) if baseline_path.exists() else None,
        "active_path": active_path,
        "active_size": path_size(Path(active_path)) if active_path else None,
    }


def tui_watch_cmd(args: argparse.Namespace) -> None:
    import curses

    run_dir = Path(args.run_dir)
    panes = ["events", "measurements", "processes", "files"]
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
            model = build_watch_model(run_dir)
            h, w = stdscr.getmaxyx()
            stdscr.erase()
            state = model["state"]
            manifest = model["manifest"]
            active = model["active"]
            title = " CEREBELLUM LIVE "
            stdscr.addnstr(0, max(0, (w - len(title)) // 2), title, w - 1, curses.color_pair(1) | curses.A_BOLD)
            summary = [
                f"run {manifest.get('run_id') or state.get('run_id') or run_dir.name}",
                f"model {state.get('model_family')}/{state.get('model_name')}  status {state.get('run_status')}  profile {manifest.get('ppl_profile') or state.get('ppl_profile') or 'custom'}",
                f"progress {model['bar']} {model['progress']}  ppl {state.get('current_ppl')}  eta {model['eta']} ({model['eta_basis']})",
                f"active {active.get('event')} {active.get('level')} {active.get('tensor')}  age {fmt_seconds(model['active_age'])}  last event {fmt_seconds(model['last_age'])}",
                f"sizes current {fmt_bytes(model['baseline_size'])}  active {fmt_bytes(model['active_size'])}  disk {disk_free_gb(run_dir):.1f} GiB free",
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
                    lines.append(f"{row.get('level', '-'):<6} ppl={row.get('ppl', '-')} delta={delta_s:<12} q={fmt_bytes(row.get('size_bytes')):<10} {row.get('tensor', '')}")
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
        rows = sqlite_rows(
            db,
            """
            SELECT benchmark, COUNT(*) AS runs, ROUND(MAX(score), 3) AS best,
                   ROUND(AVG(score), 3) AS avg
            FROM benchmarks GROUP BY benchmark ORDER BY benchmark
            """,
        )
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


def cerebellum_metadata_block(run_dir: Path, gguf: Path | None = None, hash_files: bool = False) -> dict[str, Any]:
    report = build_report(run_dir)
    manifest = read_json(run_dir / "manifest.json", {})
    files = manifest.get("files", {})
    final_types = first_existing(run_dir, BEST_TYPES_FILES)
    summary = first_existing(run_dir, SUMMARY_JSON_FILES)
    metadata = {
        "cerebellum.tool": "Cerebellum",
        "cerebellum.provenance_schema": "1",
        "cerebellum.run_id": report.get("run_id"),
        "cerebellum.model_family": report.get("model_family"),
        "cerebellum.model_name": report.get("model_name"),
        "cerebellum.source_name": report.get("source_name"),
        "cerebellum.ppl_profile": report.get("ppl_profile"),
        "cerebellum.corpus": report.get("corpus"),
        "cerebellum.base_type": manifest.get("base_type"),
        "cerebellum.start_type": manifest.get("start_type"),
        "cerebellum.levels": ",".join(manifest.get("levels") or []),
        "cerebellum.locked_count": str(report.get("locked_count")),
        "cerebellum.candidate_count": str(report.get("candidate_count")),
        "cerebellum.current_ppl": str(report.get("current_ppl")),
        "cerebellum.run_dir_sha256": hashlib.sha256(str(run_dir).encode()).hexdigest(),
        "cerebellum.tensor_types_sha256": sha256_file(final_types) if hash_files else None,
        "cerebellum.summary_sha256": sha256_file(summary) if hash_files else None,
        "cerebellum.source_gguf_sha256": sha256_file(Path(manifest["source_gguf"])) if hash_files and manifest.get("source_gguf") else None,
        "cerebellum.final_gguf_sha256": sha256_file(gguf) if hash_files and gguf else None,
    }
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
    for key, field in reader.fields.items():
        if not key.startswith("cerebellum."):
            continue
        value = getattr(field, "contents", None)
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        fields[key] = str(value)
    return fields


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


def export_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    report = build_report(run_dir)
    if args.kind == "infographic":
        payload = {
            "schema": "cerebellum.infographic.v1",
            "report": report,
            "prompt": (
                "Create a clean technical infographic from this Cerebellum quantization run. "
                "Show model, PPL, locked tensors, candidate count, component deltas, and recent decisions."
            ),
        }
    elif args.kind == "ai":
        payload = {
            "schema": "cerebellum.ai_context.v1",
            "instruction": "Use this run data to compare quantization decisions, summarize findings, or draft model-card evidence.",
            "report": report,
        }
    else:
        payload = report
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
        payload["generated_metadata"] = cerebellum_metadata_block(Path(args.run_dir), gguf, args.hash_files)
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
    lines = [
        f"# {title}",
        "",
        "This GGUF was produced with **Cerebellum**, a resource-aware mixed-precision quantization workflow.",
        "",
        "## Cerebellum provenance",
        "",
        f"- Run ID: `{metadata.get('cerebellum.run_id')}`",
        f"- Model: `{metadata.get('cerebellum.model_family')}/{metadata.get('cerebellum.model_name')}`",
        f"- Source: `{metadata.get('cerebellum.source_name')}`",
        f"- PPL profile: `{metadata.get('cerebellum.ppl_profile')}`",
        f"- Current PPL: `{metadata.get('cerebellum.current_ppl')}`",
        f"- Locked tensors: `{metadata.get('cerebellum.locked_count')}`",
        f"- Candidate tests: `{metadata.get('cerebellum.candidate_count')}`",
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
    metadata = cerebellum_metadata_block(run_dir, gguf, args.hash_files)
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
        info["binaries"][label] = find_executable(name, env_var)
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


def plan_space_cmd(args: argparse.Namespace) -> None:
    source = Path(args.source_gguf)
    source_size = path_size(source)
    candidates = [Path(p) for p in args.scratch_candidates.split(",") if p]
    if args.data_root:
        candidates.append(Path(args.data_root))
    candidates.extend([default_data_root(), Path.cwd(), Path("/tmp")])
    required_single = int(source_size * 1.7 + args.margin_gb * 1e9)
    required_two_slot = int(source_size * 2.4 + args.margin_gb * 1e9)
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
    payload = {"source_gguf": str(source), "rows": rows}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"source: {source} ({source_size / 2**30:.2f} GiB)")
    for row in rows:
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
        "3. Run `cerebellum run --source-gguf ... --profile wiki --family ... --model-name ...`.",
        "4. Watch with `cerebellum watch RUN_DIR`, `cerebellum status RUN_DIR`, and `cerebellum events RUN_DIR`.",
        "5. Generate reports with `cerebellum report RUN_DIR`.",
        "6. Import query data with `cerebellum db import-run RUN_DIR`.",
    ],
    "low-space": [
        "Use `--scratch-root` when metadata and large GGUF artifacts should live on different drives.",
        "Use `single-candidate` behavior when disk is tight: keep source, current baseline, and one candidate.",
        "Use `two-slot-pipeline` when space allows: CPU quantizes the next candidate while GPU measures the previous one.",
        "Never delete source GGUF, manifest, state, event logs, candidate logs, or final tensor-type files.",
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


def upload_cmd(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    report = build_report(run_dir)
    files = [
        run_dir / "manifest.json",
        run_dir / "state.json",
        first_existing(run_dir, EVENT_FILES),
        first_existing(run_dir, CANDIDATE_FILES),
        first_existing(run_dir, SUMMARY_JSON_FILES),
        first_existing(run_dir, SUMMARY_MD_FILES),
        first_existing(run_dir, BEST_TYPES_FILES),
    ]
    files = [path for path in files if path.exists()]
    if args.dry_run:
        print(json.dumps({"target": args.target, "repo": args.repo, "files": [str(p) for p in files], "report": report}, indent=2))
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
                path_in_repo=f"cerebellum_runs/{report['run_id']}/{path.name}",
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
        branch = args.branch or f"cerebellum-run-{report['run_id']}"
        print(json.dumps({"todo": "github upload requires repo worktree policy; use --dry-run for file list", "branch": branch}, indent=2))
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
                rows.append(load_run(manifest.parent))
            self._json({"runs": rows})
        elif parsed.path == "/db/families":
            rows = sqlite_rows(self.db_path, "SELECT * FROM model_families ORDER BY name")
            self._json({"rows": rows})
        elif parsed.path == "/report":
            run_dir = qs.get("run_dir", [None])[0]
            if not run_dir:
                self._json({"error": "run_dir query param required"}, 400)
            else:
                self._json(build_report(Path(run_dir)))
        else:
            self._json({"error": "not found"}, 404)


def api_cmd(args: argparse.Namespace) -> None:
    CerebellumAPI.data_root = Path(args.data_root) if args.data_root else default_data_root()
    CerebellumAPI.db_path = Path(args.db)
    server = ThreadingHTTPServer((args.host, args.port), CerebellumAPI)
    print(f"Cerebellum API: http://{args.host}:{args.port}")
    print("Endpoints: /health /runs /db/families /report?run_dir=...")
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
            family=job.get("family"),
            model_name=job.get("model_name"),
            source_name=job.get("source_name"),
            data_root=job.get("data_root"),
            run_name=job.get("run_name"),
            run_dir=job.get("run_dir"),
            tensor_file=job.get("tensor_file"),
            scratch_root=job.get("scratch_root"),
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
            distrobox=job.get("distrobox"),
            quant_timeout=job.get("quant_timeout", 1800),
            ppl_timeout=job.get("ppl_timeout", 900),
            keep_losers=job.get("keep_losers", False),
            no_keep_winners=job.get("no_keep_winners", False),
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
    run_dir = build_run_dir(args)
    run_id = slug(args.run_name or run_dir.name)
    cfg = Config(
        source_gguf=Path(args.source_gguf),
        corpus=resolve_ppl_corpus(args.profile, args.corpus),
        ppl_profile=args.profile,
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
        scratch_root=Path(args.scratch_root) if args.scratch_root else None,
        quantize_bin=args.quantize_bin,
        perplexity_bin=args.perplexity_bin,
        gpu_layers=args.gpu_layers,
        ctx_size=args.ctx_size,
        chunks=args.chunks,
        max_temp_gb=args.max_temp_gb,
        min_free_gb=args.min_free_gb,
        keep_winners=not args.no_keep_winners,
        keep_losers=args.keep_losers,
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
    if args.cmd == "runs":
        runs_cmd(args)
        return
    if args.cmd == "provenance":
        provenance_cmd(args)
        return
    if args.cmd == "finalize":
        finalize_cmd(args)
        return
    if args.cmd == "schedule":
        schedule_cmd(args)
        return
    if args.cmd == "system":
        system_cmd(args)
        return
    if args.cmd == "doctor":
        doctor_cmd(args)
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
    if args.cmd == "export":
        export_cmd(args)
        return
    if args.cmd == "auth":
        auth_cmd(args)
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
    main()
