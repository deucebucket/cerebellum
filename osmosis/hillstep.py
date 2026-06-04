"""Cerebellum quantization engine.

This is the durable, resumable CLI engine for per-tensor quantization search.
It overlaps CPU quantization with GPU perplexity measurement for each tensor,
records every observable event, and keeps enough state on disk to recover after
process death or a system lockup.
"""

from __future__ import annotations

import argparse
import csv
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
    "wiki": [
        "/var/home/deucebucket/games/osmosis-quants/wiki.test.raw",
        "/var/home/deucebucket/games/wikitext-2-raw-test.txt",
        "wikitext-test.txt",
    ],
    "agentic": [
        "/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_agent.txt",
        "/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_agent_strict.txt",
    ],
    "code": ["/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_code.txt"],
    "math": ["/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_math.txt"],
    "dialogue": ["/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_dialogue.txt"],
    "all-around": ["/var/home/deucebucket/games/cerebellum-calibration/cerebellum_calibration_combined.txt"],
}
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


def run_external(cmd: list[str], timeout: int, distrobox: str | None = None) -> tuple[int, str, float]:
    if distrobox:
        import shlex

        shell_cmd = shlex.join(cmd)
        cmd = ["distrobox", "enter", distrobox, "--", "bash", "-lc", shell_cmd]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - started
    return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), elapsed


def disk_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / (1024**3)


def path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


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
    for candidate in PPL_PROFILES.get(profile, []):
        path = Path(candidate)
        if path.exists():
            return path
    raise SystemExit(f"no local corpus found for --profile {profile}; pass --corpus explicitly")


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
        rc, output, seconds = run_external(self.ppl_cmd(self.paths.baseline), self.cfg.ppl_timeout, self.cfg.distrobox)
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
                    rc, output, seconds = run_external(self.ppl_cmd(c.gguf_path), self.cfg.ppl_timeout, self.cfg.distrobox)
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
    watch.add_argument("--plain", action="store_true")
    watch.add_argument("--no-color", action="store_true")

    stop = sub.add_parser("stop", help="stop or repair a Cerebellum run state")
    stop.add_argument("run_dir")
    stop.add_argument("--reason", default="user")
    stop.add_argument("--no-kill", action="store_true", help="only mark state stopped; do not signal a process")

    runs = sub.add_parser("runs", help="list known runs under a data root")
    runs.add_argument("--data-root", default=None)
    runs.add_argument("--json", action="store_true")

    schedule = sub.add_parser("schedule", help="run multiple Cerebellum jobs from a JSON schedule")
    schedule.add_argument("--file", required=True)

    system = sub.add_parser("system", help="inspect local resources and tool availability")
    system.add_argument("--json", action="store_true")

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
            "export", "auth", "upload", "api", "system", "plan-space",
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
    run_dir = Path(args.run_dir)
    enabled = not args.no_color and not args.plain
    try:
        while True:
            state = read_json(run_dir / "state.json", {})
            manifest = read_json(run_dir / "manifest.json", {})
            events = read_jsonl(first_existing(run_dir, EVENT_FILES))
            candidates = read_jsonl(first_existing(run_dir, CANDIDATE_FILES))
            last_events = events[-12:]
            last_tensor = state.get("last_tensor")
            status = state.get("run_status")
            terminal_events = {"run_stopped", "run_finish", "tensor_interrupted", "signal_received"}
            if status in {"stopped", "complete", "failed"}:
                active = next((row for row in reversed(events) if row.get("event") in terminal_events), {})
            else:
                active = next((row for row in reversed(events) if row.get("event") in {"tensor_start", "quant_start", "ppl_start"}), {})
            os.system("clear" if os.name != "nt" else "cls")
            width = 96
            run_id = manifest.get("run_id") or state.get("run_id") or run_dir.name
            model = f"{state.get('model_family')}/{state.get('model_name')}"
            ppl = state.get("current_ppl")
            profile = manifest.get("ppl_profile") or state.get("ppl_profile") or "custom"
            corpus = manifest.get("corpus") or state.get("corpus") or "-"
            locked = len(state.get("locked", {}))
            total_hint = next((row.get("total") for row in reversed(events) if row.get("total")), None)
            progress = f"{locked}/{total_hint}" if total_hint else str(locked)
            print(color("╭" + "─" * (width - 2) + "╮", "36;1", enabled))
            print(color("│" + " CEREBELLUM ".center(width - 2) + "│", "36;1", enabled))
            print(color("│" + " resource-aware mixed-precision GGUF quantization ".center(width - 2) + "│", "36", enabled))
            print(color("╰" + "─" * (width - 2) + "╯", "36;1", enabled))
            print()
            print(color("╭─ Run ─" + "─" * (width - 9) + "╮", "34;1", enabled))
            print(f"│ id       {run_id:<{width - 13}}│")
            print(f"│ model    {model:<{width - 13}}│")
            print(f"│ status   {str(status):<{width - 13}}│")
            print(f"│ profile  {str(profile):<{width - 13}}│")
            print(f"│ corpus   {str(corpus)[-(width - 13):]:>{width - 13}}│")
            print(f"│ ppl      {str(ppl):<{width - 13}}│")
            print(f"│ progress {progress:<{width - 13}}│")
            print(color("╰" + "─" * (width - 2) + "╯", "34;1", enabled))
            print()
            print(color("╭─ Active work ─" + "─" * (width - 16) + "╮", "37;1", enabled))
            print(f"│ event    {str(active.get('event')):<{width - 13}}│")
            print(f"│ tensor   {str(active.get('tensor')):<{width - 13}}│")
            print(f"│ level    {str(active.get('level')):<{width - 13}}│")
            if last_tensor:
                print(f"│ last     {str(last_tensor):<{width - 13}}│")
            print(color("╰" + "─" * (width - 2) + "╯", "37;1", enabled))
            totals = state.get("totals", {})
            print()
            print(color("╭─ Timing ─" + "─" * (width - 11) + "╮", "35;1", enabled))
            timing_line = (
                f"quant {fmt_seconds(totals.get('quant_seconds'))}   "
                f"ppl {fmt_seconds(totals.get('ppl_seconds'))}   "
                f"tests {totals.get('candidates', 0)}   failures {totals.get('failures', 0)}"
            )
            print(f"│ {timing_line:<{width - 4}} │")
            print(color("╰" + "─" * (width - 2) + "╯", "35;1", enabled))
            print()
            print(color("╭─ Recent measurements ─" + "─" * (width - 24) + "╮", "32;1", enabled))
            print(f"│ {'quant':<8}{'ppl':<14}{'delta':<14}{'tensor':<{width - 42}}│")
            print(color("├" + "─" * (width - 2) + "┤", "32", enabled))
            for row in candidates[-8:]:
                delta = row.get("delta")
                delta_s = "-" if delta is None else f"{delta:+.4f}"
                print(f"│ {row.get('level', '-'):<8}{str(row.get('ppl', '-')):<14}{delta_s:<14}{row.get('tensor', ''):<{width - 42}}│")
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
        data.append(item)
    if args.json:
        print(json.dumps({"runs": data}, indent=2, sort_keys=True))
        return
    for item in data:
        state = item.get("state", {})
        print(
            f"{item.get('run_id')}  {state.get('run_status', '?'):<9} "
            f"{item.get('model_family')}/{item.get('model_name')}  "
            f"locked={len(state.get('locked', {}))} ppl={state.get('current_ppl')}"
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
    if args.cmd == "schedule":
        schedule_cmd(args)
        return
    if args.cmd == "system":
        system_cmd(args)
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
