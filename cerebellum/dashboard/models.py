"""Database models for the Cerebellum dashboard."""

import json
import hashlib
import time
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = None
_engine = None
_SessionLocal = None
Base = declarative_base()

PIPELINE_PHASES = [
    "queued",
    "download",
    "convert",
    "imatrix",
    "ablate",
    "analyze",
    "allocate",
    "build",
    "benchmark",
    "publish",
    "done",
]

JOB_STATUSES = ["queued", "running", "completed", "failed", "cancelled"]


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(512), nullable=False)
    model_source = Column(String(32), default="huggingface")
    model_path = Column(String(1024), default="")
    status = Column(String(32), default="queued", index=True)
    current_phase = Column(String(32), default="queued")
    progress = Column(Float, default=0.0)
    phase_progress = Column(Float, default=0.0)
    error_message = Column(Text, default="")
    priority = Column(Integer, default=0, index=True)
    pipeline_steps = Column(Text, default="")
    config = Column(Text, default="{}")
    output_dir = Column(String(1024), default="")
    result = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    worker_id = Column(String(128), default="")

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_source": self.model_source,
            "model_path": self.model_path,
            "status": self.status,
            "current_phase": self.current_phase,
            "progress": self.progress,
            "phase_progress": self.phase_progress,
            "error_message": self.error_message,
            "priority": self.priority,
            "pipeline_steps": self._load_json(self.pipeline_steps),
            "config": self._load_json(self.config),
            "output_dir": self.output_dir,
            "result": self._load_json(self.result),
            "created_at": self._ts(self.created_at),
            "started_at": self._ts(self.started_at),
            "completed_at": self._ts(self.completed_at),
        }

    @staticmethod
    def _load_json(val):
        if isinstance(val, str) and val:
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val
        return val

    @staticmethod
    def _ts(dt):
        return dt.isoformat() if dt else None


class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(512), nullable=False)
    model_source = Column(String(32), default="huggingface")
    model_path = Column(String(1024), default="")
    cron_expr = Column(String(64), nullable=False)
    pipeline_steps = Column(Text, default="")
    config = Column(Text, default="{}")
    enabled = Column(Boolean, default=True)
    last_run = Column(DateTime, nullable=True)
    last_status = Column(String(32), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_source": self.model_source,
            "model_path": self.model_path,
            "cron_expr": self.cron_expr,
            "pipeline_steps": Job._load_json(self.pipeline_steps),
            "config": Job._load_json(self.config),
            "enabled": self.enabled,
            "last_run": Job._ts(self.last_run),
            "last_status": self.last_status,
        }


class ModelWatch(Base):
    __tablename__ = "model_watches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hf_model_id = Column(String(512), nullable=False, unique=True)
    name = Column(String(512), default="")
    architecture = Column(String(128), default="")
    params_b = Column(Float, default=0.0)
    description = Column(Text, default="")
    watched = Column(Boolean, default=True)
    auto_queue = Column(Boolean, default=False)
    min_params_b = Column(Float, default=0.0)
    max_params_b = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_checked = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "hf_model_id": self.hf_model_id,
            "name": self.name,
            "architecture": self.architecture,
            "params_b": self.params_b,
            "description": self.description,
            "watched": self.watched,
            "auto_queue": self.auto_queue,
            "min_params_b": self.min_params_b,
            "max_params_b": self.max_params_b,
            "last_checked": Job._ts(self.last_checked),
        }


class EventLog(Base):
    __tablename__ = "event_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, index=True)
    phase = Column(String(32), default="")
    level = Column(String(16), default="info")
    message = Column(Text, default="")
    progress = Column(Float, nullable=True)
    timestamp = Column(Float, default=lambda: time.time())

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "phase": self.phase,
            "level": self.level,
            "message": self.message,
            "progress": self.progress,
            "timestamp": self.timestamp,
        }


class Model(Base):
    __tablename__ = "models"

    id = Column(String(256), primary_key=True)
    display_name = Column(String(512), nullable=False)
    source_repo = Column(String(1024), default="")
    architecture = Column(String(128), default="")
    params_b = Column(Float, default=0.0)
    tokenizer = Column(String(256), default="")
    modalities = Column(Text, default="[]")
    variant_of = Column(String(256), default="")
    root_path = Column(String(2048), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "display_name": self.display_name,
            "source_repo": self.source_repo,
            "architecture": self.architecture,
            "params_b": self.params_b,
            "tokenizer": self.tokenizer,
            "modalities": Job._load_json(self.modalities) or [],
            "variant_of": self.variant_of or None,
            "root_path": self.root_path,
            "created_at": Job._ts(self.created_at),
            "updated_at": Job._ts(self.updated_at),
        }


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(256), index=True, nullable=False)
    experiment_id = Column(Integer, nullable=True, index=True)
    type = Column(String(64), default="artifact", index=True)
    path = Column(String(2048), nullable=False, unique=True)
    size_bytes = Column(Integer, default=0)
    sha256 = Column(String(64), default="")
    visibility = Column(String(32), default="private", index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "type": self.type,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256 or None,
            "visibility": self.visibility,
            "created_at": Job._ts(self.created_at),
        }


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(256), index=True, nullable=False)
    artifact_id = Column(Integer, nullable=True, index=True)
    benchmark = Column(String(128), index=True, nullable=False)
    harness = Column(String(256), default="")
    harness_revision = Column(String(128), default="")
    command = Column(Text, default="")
    server_settings = Column(Text, default="{}")
    results = Column(Text, default="{}")
    detailed_path = Column(String(2048), default="")
    status = Column(String(32), default="completed", index=True)
    audit_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "artifact_id": self.artifact_id,
            "benchmark": self.benchmark,
            "harness": self.harness,
            "harness_revision": self.harness_revision,
            "command": self.command,
            "server_settings": Job._load_json(self.server_settings) or {},
            "results": Job._load_json(self.results) or {},
            "detailed_path": self.detailed_path or None,
            "status": self.status,
            "audit_id": self.audit_id,
            "created_at": Job._ts(self.created_at),
        }


class BenchmarkAudit(Base):
    __tablename__ = "benchmark_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id = Column(Integer, index=True, nullable=False)
    auditor = Column(String(128), default="automated_script")
    passed = Column(Boolean, default=False)
    ast_syntax_fail_count = Column(Integer, default=0)
    fence_count = Column(Integer, default=0)
    prompt_echo_count = Column(Integer, default=0)
    repeated_target_def_count = Column(Integer, default=0)
    pass_only_count = Column(Integer, default=0)
    cop_out_count = Column(Integer, default=0)
    unknown_answer_count = Column(Integer, default=0)
    empty_response_fallback_count = Column(Integer, default=0)
    parse_method = Column(String(128), default="")
    first_wrong_sample_path = Column(String(2048), default="")
    inspected_sample_ids = Column(Text, default="[]")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "benchmark_run_id": self.benchmark_run_id,
            "auditor": self.auditor,
            "passed": self.passed,
            "ast_syntax_fail_count": self.ast_syntax_fail_count,
            "fence_count": self.fence_count,
            "prompt_echo_count": self.prompt_echo_count,
            "repeated_target_def_count": self.repeated_target_def_count,
            "pass_only_count": self.pass_only_count,
            "cop_out_count": self.cop_out_count,
            "unknown_answer_count": self.unknown_answer_count,
            "empty_response_fallback_count": self.empty_response_fallback_count,
            "parse_method": self.parse_method,
            "first_wrong_sample_path": self.first_wrong_sample_path or None,
            "inspected_sample_ids": Job._load_json(self.inspected_sample_ids) or [],
            "notes": self.notes,
            "created_at": Job._ts(self.created_at),
        }


class ModelCard(Base):
    __tablename__ = "model_cards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(256), index=True, nullable=False)
    generated_markdown = Column(Text, default="")
    audit_statement = Column(Text, default="")
    hf_ready = Column(Boolean, default=False)
    hf_uploaded_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    version = Column(Integer, default=1)

    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "generated_markdown": self.generated_markdown,
            "audit_statement": self.audit_statement,
            "hf_ready": self.hf_ready,
            "hf_uploaded_at": Job._ts(self.hf_uploaded_at),
            "created_at": Job._ts(self.created_at),
            "version": self.version,
        }


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task = Column(Text, default="")
    status = Column(String(32), default="queued", index=True)
    model = Column(String(128), default="")
    log_path = Column(String(2048), default="")
    diff_path = Column(String(2048), default="")
    verdict = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "task": self.task,
            "status": self.status,
            "model": self.model,
            "log_path": self.log_path or None,
            "diff_path": self.diff_path or None,
            "verdict": Job._load_json(self.verdict) or {},
            "created_at": Job._ts(self.created_at),
            "updated_at": Job._ts(self.updated_at),
        }


def stable_model_id(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    safe = "-".join(part for part in safe.split("-") if part)
    if safe:
        return safe[:180]
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def init_db(db_path: str):
    global DB_PATH, _engine, _SessionLocal
    DB_PATH = db_path
    _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)


def get_session():
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()
