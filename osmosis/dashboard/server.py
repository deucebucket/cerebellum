"""FastAPI server — REST API, WebSocket, and dashboard frontend."""

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from osmosis.dashboard.models import (
    Job,
    Schedule,
    ModelWatch,
    EventLog,
    Model,
    Artifact,
    BenchmarkRun,
    BenchmarkAudit,
    ModelCard,
    get_session,
    init_db,
    PIPELINE_PHASES,
    stable_model_id,
)
from osmosis.hillstep import DEFAULT_DB, audit_jsonl_file, queue_get_job, queue_list_jobs
from osmosis.dashboard.worker import Worker, event_bus
from osmosis.dashboard.scheduler import Scheduler


DB_PATH = os.environ.get("CEREBELLUM_DB", DEFAULT_DB)
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_FIX_CUTOFF = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc).timestamp()


class JobCreate(BaseModel):
    model_name: str
    model_source: str = "huggingface"
    model_path: str = ""
    priority: int = 0
    pipeline_steps: list[str] = []
    config: dict = {}
    output_dir: str = ""


class ScheduleCreate(BaseModel):
    model_name: str
    model_source: str = "huggingface"
    model_path: str = ""
    cron_expr: str
    pipeline_steps: list[str] = []
    config: dict = {}
    enabled: bool = True


class ModelWatchCreate(BaseModel):
    hf_model_id: str
    name: str = ""
    architecture: str = ""
    params_b: float = 0.0
    description: str = ""
    watched: bool = True
    auto_queue: bool = False
    min_params_b: float = 0.0
    max_params_b: float = 0.0


class BenchmarkResultIngest(BaseModel):
    path: str
    model_id: str = ""
    artifact_id: int | None = None
    detailed_path: str = ""
    harness: str = ""
    harness_revision: str = ""
    command: str = ""
    server_settings: dict = {}


def envelope(data=None, error: str | None = None) -> dict:
    return {"data": data, "error": error}


def control_plane_db_path() -> Path:
    return Path(DB_PATH)


worker: Optional[Worker] = None
scheduler: Optional[Scheduler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker, scheduler
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    init_db(DB_PATH)
    worker = Worker()
    scheduler = Scheduler()
    asyncio.create_task(worker.start())
    asyncio.create_task(scheduler.start())
    yield
    if worker:
        worker.stop()
    if scheduler:
        scheduler.stop()


app = FastAPI(title="Cerebellum Dashboard", version="0.1.0", lifespan=lifespan)


# ── Local Artifact Discovery ───────────────────────────────────────

def _json_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _file_size_gb(path: Path) -> float:
    try:
        return round(path.stat().st_size / 1e9, 2)
    except OSError:
        return 0.0


def _benchmark_label(path: Path) -> str:
    name = path.name
    for suffix in (
        "_evalplus_results.json",
        "_evalplus_chat_results.json",
        "_arc_results.json",
        "_hellaswag_results.json",
        "_mmlu_redux_results.json",
        "_mmlu_results.json",
        "_results.json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _benchmark_audit(path: Path, data: dict) -> dict:
    name = path.name.lower()
    benchmark = str(data.get("benchmark") or path.stem).lower()
    mtime = path.stat().st_mtime
    status = "verified"
    notes: list[str] = []

    if name.endswith("_samples_eval_results.json"):
        return {
            "status": "artifact",
            "notes": ["EvalPlus raw evaluator artifact; use the project summary JSON."],
        }

    if "evalplus_chat" in name:
        if "fresh" in name or mtime >= BENCH_FIX_CUTOFF:
            status = "verified"
            notes.append("Chat EvalPlus path with indentation normalization and ast.parse validation.")
        else:
            status = "needs_audit"
            notes.append("Chat EvalPlus result predates the indentation audit cutoff.")
    elif "evalplus" in name:
        if data.get("pass_at_1_plus") is None:
            status = "needs_audit"
            notes.append("Summary has null EvalPlus pass rates; parse/evaluator output needs repair.")
        elif mtime >= BENCH_FIX_CUTOFF:
            status = "verified"
            notes.append("EvalPlus result generated after the HumanEval fence/indent bug cutoff.")
        else:
            status = "stale"
            notes.append("Pre-fix code benchmark; old HumanEval handling can under-score indentation/fence outputs.")
    elif "humaneval" in name or "humaneval" in benchmark:
        status = "stale"
        notes.append("Legacy HumanEval path retired; rerun as EvalPlus before publishing.")
    elif "arc" in name or "arc" in benchmark:
        if mtime < BENCH_FIX_CUTOFF:
            status = "stale"
            notes.append("May be affected by ARC numeric answer-key normalization bug.")
        else:
            notes.append("Generated after ARC numeric answer-key fix.")
    elif "hellaswag" in name or "hellaswag" in benchmark:
        if mtime < BENCH_FIX_CUTOFF:
            status = "stale"
            notes.append("May be affected by old empty-response handling.")
        else:
            notes.append("Generated after benchmark correction cutoff.")
    elif "mmlu" in name or "mmlu" in benchmark:
        notes.append("No known label/indent bug, but publish only with detailed wrong-answer audit.")

    if not notes:
        notes.append("No benchmark-specific audit rule matched.")
    return {"status": status, "notes": notes}


def _summarize_benchmark(path: Path) -> dict:
    data = _json_file(path)
    benchmark = data.get("benchmark") or path.stem
    metric = None
    value = None
    for key in (
        "pass_at_1_plus",
        "pass_at_1_base",
        "accuracy",
        "acc",
        "score",
    ):
        if isinstance(data.get(key), (int, float)):
            metric = key
            value = round(float(data[key]), 2)
            break
    return {
        "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "model": data.get("model") or _benchmark_label(path),
        "benchmark": benchmark,
        "metric": metric,
        "value": value,
        "total_problems": data.get("total_problems") or data.get("total"),
        "timestamp": data.get("timestamp", ""),
        "audit": _benchmark_audit(path, data),
    }


def _summarize_live_log(path: Path) -> dict | None:
    try:
        text = path.read_text(errors="replace")[-12000:]
    except OSError:
        return None
    matches = re.findall(r"\[\s*(\d+)/(\d+)\]", text)
    if not matches:
        return None
    current, total = [int(x) for x in matches[-1]]
    return {
        "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "model": path.name.removesuffix("_evalplus_run.log"),
        "benchmark": "evalplus_humaneval_plus",
        "status": "running" if current < total else "generated",
        "current": current,
        "total": total,
        "progress": round(current / total * 100, 1) if total else 0.0,
    }


def _summarize_agent_probe(path: Path) -> dict:
    data = _json_file(path)
    models = []
    for label, item in (data.get("models") or {}).items():
        probes = item.get("probes") or []
        failed = [p for p in probes if not p.get("pass")]
        models.append({
            "label": label,
            "path": item.get("path", ""),
            "size_bytes": item.get("size_bytes", 0),
            "score": item.get("score", 0),
            "max_score": item.get("max_score", 0),
            "pass_rate": item.get("pass_rate", 0.0),
            "failed": [
                {
                    "id": p.get("id", ""),
                    "category": p.get("category", ""),
                    "score": p.get("score", 0),
                    "max_score": p.get("max_score", 0),
                }
                for p in failed
            ],
        })
    return {
        "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "timestamp": data.get("timestamp", ""),
        "schema": data.get("schema", ""),
        "explanation": data.get("explanation", {}),
        "models": models,
        "updated_at": path.stat().st_mtime,
    }


def discover_agent_probes() -> list[dict]:
    probes = []
    for path in REPO_ROOT.glob("**/*agent_probe*.json"):
        if ".bench-venv" in path.parts or "__pycache__" in path.parts:
            continue
        probes.append(_summarize_agent_probe(path))
    probes.sort(key=lambda p: p["updated_at"], reverse=True)
    return probes


def _devlog_summary(path: Path) -> dict:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        text = ""
    title = path.stem
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip() or title
            break
    return {
        "name": path.name,
        "title": title,
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "size_kb": round(path.stat().st_size / 1024, 1),
        "updated_at": path.stat().st_mtime,
        "updated": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "excerpt": text[:600],
    }


def discover_devlogs() -> list[dict]:
    candidates: list[Path] = []
    patterns = (
        "cerebellum-dev/*.md",
        "docs/*log*.md",
        "docs/*finding*.md",
        "docs/*todo*.md",
        "cerebellum-*/*log*.md",
        "cerebellum-*/*finding*.md",
        "osmosis-*/*log*.md",
        "osmosis-*/*finding*.md",
    )
    for pattern in patterns:
        candidates.extend(REPO_ROOT.glob(pattern))
    seen: set[Path] = set()
    logs = []
    for path in candidates:
        if not path.is_file() or path in seen:
            continue
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        seen.add(path)
        logs.append(_devlog_summary(path))
    logs.sort(key=lambda p: p["updated_at"], reverse=True)
    return logs


def _model_card_for_dir(path: Path) -> dict:
    ggufs = sorted(path.glob("*.gguf"))
    bench_dir = path / "benchmark_results"
    if not bench_dir.exists():
        bench_dir = path / "benchmarks"
    benchmark_files = sorted(bench_dir.glob("*_results.json")) if bench_dir.exists() else []
    # EvalPlus writes an extra *_samples_eval_results.json; keep those as artifacts,
    # but model-card summaries should prefer the project summary JSON.
    summaries = [
        _summarize_benchmark(p)
        for p in benchmark_files
        if not p.name.endswith("_samples_eval_results.json")
    ]
    live = []
    if bench_dir.exists():
        for log in sorted(bench_dir.glob("*_run.log")):
            item = _summarize_live_log(log)
            if item:
                live.append(item)
    agent_probes = []
    if bench_dir.exists():
        agent_probes = [
            _summarize_agent_probe(p)
            for p in sorted(bench_dir.glob("*agent_probe*.json"))
        ]

    docs = [
        p for p in sorted(path.glob("*.md"))
        if p.name.lower() in {
            "readme.md",
            "findings_winning_formula.md",
            "full_experiment_log.md",
            "research_log.md",
        } or "finding" in p.name.lower() or "log" in p.name.lower()
    ]

    return {
        "name": path.name,
        "path": str(path),
        "ggufs": [
            {"name": p.name, "size_gb": _file_size_gb(p), "path": str(p)}
            for p in ggufs
        ],
        "benchmarks": summaries,
        "live_benchmarks": live,
        "agent_probes": agent_probes,
        "docs": [
            {"name": p.name, "path": str(p), "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in docs
        ],
        "has_ablation": (path / "ablation_results.json").exists() or any(path.glob("ablation_results*.json")),
        "has_variants": (path / "variants").exists(),
        "updated_at": max(
            [p.stat().st_mtime for p in path.glob("*") if p.exists()],
            default=path.stat().st_mtime,
        ),
    }


def discover_model_cards() -> list[dict]:
    roots = [REPO_ROOT]
    env_roots = os.environ.get("CEREBELLUM_DISCOVERY_ROOTS", "")
    for raw in env_roots.split(":"):
        if raw.strip():
            roots.append(Path(raw).expanduser())

    seen: set[Path] = set()
    cards = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("cerebellum-*", "osmosis-*"):
            for path in root.glob(pattern):
                if not path.is_dir() or path in seen:
                    continue
                if path.name in {"cerebellum-dev"}:
                    continue
                has_artifacts = (
                    (path / "benchmark_results").exists()
                    or (path / "benchmarks").exists()
                    or any(path.glob("*.gguf"))
                    or any(path.glob("*.md"))
                    or any(path.glob("ablation_results*.json"))
                )
                if has_artifacts:
                    seen.add(path)
                    cards.append(_model_card_for_dir(path))
    cards.sort(key=lambda c: c["updated_at"], reverse=True)
    return cards


def _canonical_model_id(card: dict) -> str:
    return stable_model_id(str(card.get("name") or Path(str(card.get("path") or "model")).name))


def _artifact_type_for_path(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".gguf":
        return "quant_gguf" if any(key in name for key in ("q2", "q3", "q4", "q5", "q6", "q8", "iq")) else "f16_gguf"
    if suffix == ".dat" or "imatrix" in name:
        return "imatrix"
    if "ablation" in name and suffix == ".json":
        return "ablation"
    if "tensor" in name and suffix in {".txt", ".json"}:
        return "tensor_types"
    if "benchmark" in str(path).lower() or name.endswith("_results.json"):
        return "benchmark_result"
    if name in {"readme.md", "model_card_cerebellum.md"} or "model_card" in name:
        return "model_card"
    return "artifact"


def _artifact_visibility(path: Path) -> str:
    text = str(path).lower()
    if any(part in text for part in ("cerebellum-dev", "devlog", "ablation", "tensor_types", "events", "candidates")):
        return "private"
    return "public"


def _upsert_model_from_card(sess, card: dict) -> Model:
    model_id = _canonical_model_id(card)
    model = sess.query(Model).filter(Model.id == model_id).first()
    now = datetime.now(timezone.utc)
    if not model:
        model = Model(id=model_id, created_at=now)
        sess.add(model)
    model.display_name = str(card.get("name") or model_id)
    model.root_path = str(card.get("path") or "")
    model.updated_at = datetime.fromtimestamp(float(card.get("updated_at") or now.timestamp()), tz=timezone.utc)
    return model


def _upsert_artifact(sess, model_id: str, path_text: str, artifact_type: str | None = None, visibility: str | None = None) -> Artifact:
    path = Path(path_text)
    artifact = sess.query(Artifact).filter(Artifact.path == str(path)).first()
    if not artifact:
        artifact = Artifact(path=str(path), created_at=datetime.now(timezone.utc))
        sess.add(artifact)
    artifact.model_id = model_id
    artifact.type = artifact_type or _artifact_type_for_path(path)
    artifact.visibility = visibility or _artifact_visibility(path)
    try:
        artifact.size_bytes = path.stat().st_size
    except OSError:
        artifact.size_bytes = 0
    return artifact


def _upsert_artifacts_from_card(sess, model_id: str, card: dict) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for item in card.get("ggufs") or []:
        if item.get("path"):
            artifacts.append(_upsert_artifact(sess, model_id, str(item["path"]), "quant_gguf"))
    for item in card.get("benchmarks") or []:
        if item.get("file"):
            artifacts.append(_upsert_artifact(sess, model_id, str(item["file"]), "benchmark_result"))
    for item in card.get("docs") or []:
        if item.get("path"):
            artifacts.append(_upsert_artifact(sess, model_id, str(item["path"]), "model_card"))
    root = Path(str(card.get("path") or ""))
    if root.exists():
        for pattern in ("ablation_results*.json", "*tensor*types*.txt", "*tensor_types*.txt"):
            for path in root.glob(pattern):
                artifacts.append(_upsert_artifact(sess, model_id, str(path)))
    return artifacts


def _benchmark_key(item: dict) -> str:
    return str(item.get("benchmark") or item.get("file") or "benchmark").lower().replace(" ", "_")


def _benchmark_metric(data: dict) -> tuple[str | None, float | None]:
    for key in (
        "pass_at_1_plus",
        "pass_at_1_base",
        "pass_at_1",
        "accuracy",
        "acc",
        "score",
    ):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return key, round(float(value) * 100.0 if 0.0 <= float(value) <= 1.0 else float(value), 4)
    return None, None


def _infer_benchmark_from_result(path: Path, data: dict) -> str:
    raw = str(data.get("benchmark") or _benchmark_label(path))
    return raw.lower().replace(" ", "_").replace("-", "_")


def _upsert_benchmarks_from_card(sess, model_id: str, card: dict) -> list[BenchmarkRun]:
    runs: list[BenchmarkRun] = []
    for item in card.get("benchmarks") or []:
        benchmark = _benchmark_key(item)
        detailed_path = str(item.get("file") or "")
        run = (
            sess.query(BenchmarkRun)
            .filter(BenchmarkRun.model_id == model_id, BenchmarkRun.benchmark == benchmark, BenchmarkRun.detailed_path == detailed_path)
            .first()
        )
        if not run:
            run = BenchmarkRun(model_id=model_id, benchmark=benchmark, detailed_path=detailed_path)
            sess.add(run)
        run.status = "completed"
        run.results = json.dumps(
            {
                "metric": item.get("metric"),
                "value": item.get("value"),
                "total_problems": item.get("total_problems"),
                "audit": item.get("audit"),
            }
        )
        sess.flush()
        audit_info = item.get("audit") if isinstance(item.get("audit"), dict) else {}
        audit = sess.query(BenchmarkAudit).filter(BenchmarkAudit.benchmark_run_id == run.id).first()
        if not audit:
            audit = BenchmarkAudit(benchmark_run_id=run.id)
            sess.add(audit)
        audit.passed = str(audit_info.get("status") or "").lower() == "verified"
        audit.parse_method = "discovery_summary"
        audit.notes = "; ".join(str(note) for note in audit_info.get("notes") or [])
        sess.flush()
        run.audit_id = audit.id
        runs.append(run)
    return runs


def _audit_detail_counts(detail_path: Path) -> dict:
    row = audit_jsonl_file(detail_path)
    counts = row.get("counts") or {}
    kind = str(row.get("kind") or "")
    total = int(row.get("total") or 0)
    blockers: list[str] = []
    if int(counts.get("json_error", 0)):
        blockers.append("JSONL parse errors")
    if int(counts.get("empty", 0)):
        blockers.append("empty responses present")
    if int(counts.get("unknown", 0)):
        blockers.append("unknown MCQ predictions present")
    if int(counts.get("pass_only", 0)):
        blockers.append("pass-only EvalPlus completions present")
    if int(counts.get("prompt_echo", 0)):
        blockers.append("prompt echoes present")
    if total <= 0:
        blockers.append("detailed artifact has no samples")
    return {
        "kind": kind,
        "total": total,
        "counts": counts,
        "samples": row.get("samples") or [],
        "passed": not blockers,
        "blockers": blockers,
    }


def _build_benchmark_audit_payload(summary_path: Path, data: dict, detail_path_text: str) -> dict:
    detail_path = Path(detail_path_text) if detail_path_text else None
    benchmark = _infer_benchmark_from_result(summary_path, data)
    notes: list[str] = []
    counts: dict = {}
    samples: list = []
    blockers: list[str] = []
    parse_method = "summary_only"
    inspected: list[str] = []

    if detail_path and detail_path.exists():
        detail = _audit_detail_counts(detail_path)
        counts = detail["counts"]
        samples = detail["samples"]
        blockers.extend(detail["blockers"])
        parse_method = f"{detail['kind']}_jsonl"
        inspected = [str(item.get("task_id") or item.get("line") or "") for item in samples if item]
        notes.append(f"audited detailed JSONL with {detail['total']} samples")
    else:
        if detail_path_text:
            blockers.append("detailed artifact missing")
        if any(key in benchmark for key in ("evalplus", "humaneval", "arc", "hellaswag", "mmlu")):
            blockers.append("detailed audit artifact required before publishing")
        notes.append("summary ingested without detailed JSONL audit")

    status_audit = _benchmark_audit(summary_path, data)
    if status_audit.get("status") in {"stale", "needs_audit"}:
        blockers.append(f"discovery audit status is {status_audit['status']}")
    notes.extend(str(note) for note in status_audit.get("notes") or [])

    return {
        "passed": not blockers,
        "parse_method": parse_method,
        "counts": counts,
        "notes": "; ".join(notes + blockers),
        "inspected_sample_ids": inspected[:30],
        "first_wrong_sample_path": detail_path_text if samples else "",
    }


def _upsert_benchmark_result(sess, payload: BenchmarkResultIngest) -> tuple[BenchmarkRun, BenchmarkAudit, Artifact]:
    result_path = Path(payload.path).expanduser()
    if not result_path.exists():
        raise ValueError(f"benchmark result not found: {result_path}")
    data = _json_file(result_path)
    if not data:
        raise ValueError(f"benchmark result is not valid JSON object: {result_path}")

    model_name = str(payload.model_id or data.get("model") or result_path.parent.parent.name or result_path.stem)
    model_id = stable_model_id(model_name)
    model = sess.query(Model).filter(Model.id == model_id).first()
    now = datetime.now(timezone.utc)
    if not model:
        model = Model(id=model_id, display_name=model_name, root_path=str(result_path.parent.parent), created_at=now)
        sess.add(model)
    model.updated_at = now

    artifact = _upsert_artifact(sess, model_id, str(result_path), "benchmark_result")
    artifact.sha256 = _sha256_file(result_path)
    artifact.visibility = "public"
    if payload.artifact_id:
        artifact.experiment_id = payload.artifact_id
    sess.flush()

    benchmark = _infer_benchmark_from_result(result_path, data)
    metric, value = _benchmark_metric(data)
    detailed_path = payload.detailed_path or str(data.get("detailed_path") or "")
    run = (
        sess.query(BenchmarkRun)
        .filter(BenchmarkRun.model_id == model_id, BenchmarkRun.benchmark == benchmark, BenchmarkRun.detailed_path == detailed_path)
        .first()
    )
    if not run:
        run = BenchmarkRun(model_id=model_id, benchmark=benchmark, detailed_path=detailed_path)
        sess.add(run)
    run.artifact_id = artifact.id
    run.status = str(data.get("status") or "completed")
    run.harness = payload.harness or str(data.get("harness") or "")
    run.harness_revision = payload.harness_revision or str(data.get("harness_revision") or "")
    run.command = payload.command or str(data.get("command") or "")
    run.server_settings = json.dumps(payload.server_settings or data.get("server_settings") or {})
    run.results = json.dumps(
        {
            "metric": metric,
            "value": value,
            "total_problems": data.get("total_problems") or data.get("total"),
            "correct": data.get("correct"),
            "raw": data,
        }
    )
    sess.flush()

    audit_payload = _build_benchmark_audit_payload(result_path, data, detailed_path)
    audit = sess.query(BenchmarkAudit).filter(BenchmarkAudit.benchmark_run_id == run.id).first()
    if not audit:
        audit = BenchmarkAudit(benchmark_run_id=run.id)
        sess.add(audit)
    counts = audit_payload["counts"]
    audit.passed = bool(audit_payload["passed"])
    audit.parse_method = str(audit_payload["parse_method"])
    audit.empty_response_fallback_count = int(counts.get("empty", 0))
    audit.unknown_answer_count = int(counts.get("unknown", 0))
    audit.pass_only_count = int(counts.get("pass_only", 0))
    audit.prompt_echo_count = int(counts.get("prompt_echo", 0))
    audit.cop_out_count = int(counts.get("empty", 0))
    audit.first_wrong_sample_path = str(audit_payload["first_wrong_sample_path"])
    audit.inspected_sample_ids = json.dumps(audit_payload["inspected_sample_ids"])
    audit.notes = str(audit_payload["notes"])
    sess.flush()
    run.audit_id = audit.id
    sess.commit()
    return run, audit, artifact


def _upsert_model_card(sess, model_id: str, card: dict) -> ModelCard:
    existing = sess.query(ModelCard).filter(ModelCard.model_id == model_id, ModelCard.version == 1).first()
    if not existing:
        existing = ModelCard(model_id=model_id, version=1)
        sess.add(existing)
    existing.generated_markdown = f"# {card.get('name') or model_id}\n\nDiscovered local Cerebellum model artifacts.\n"
    existing.audit_statement = "Generated from local dashboard ingest scan."
    existing.hf_ready = False
    return existing


def ingest_discovered_model_cards(sess, cards: list[dict]) -> dict:
    model_ids: set[str] = set()
    artifact_ids: set[str] = set()
    benchmark_keys: set[tuple[str, str, str]] = set()
    for card in cards:
        model = _upsert_model_from_card(sess, card)
        sess.flush()
        model_ids.add(model.id)
        for artifact in _upsert_artifacts_from_card(sess, model.id, card):
            sess.flush()
            artifact_ids.add(str(artifact.path))
        for run in _upsert_benchmarks_from_card(sess, model.id, card):
            sess.flush()
            benchmark_keys.add((run.model_id, run.benchmark, run.detailed_path or ""))
        _upsert_model_card(sess, model.id, card)
    sess.commit()
    return {"models": len(model_ids), "artifacts": len(artifact_ids), "benchmark_runs": len(benchmark_keys)}


# ── API Routes ─────────────────────────────────────────────────────

@app.post("/api/ingest/scan")
def ingest_scan():
    sess = get_session()
    try:
        summary = ingest_discovered_model_cards(sess, discover_model_cards())
        return envelope(summary)
    except Exception as exc:
        sess.rollback()
        raise HTTPException(status_code=400, detail=envelope(None, str(exc)))
    finally:
        sess.close()


@app.post("/api/ingest/benchmark-result")
def ingest_benchmark_result(data: BenchmarkResultIngest):
    sess = get_session()
    try:
        run, audit, artifact = _upsert_benchmark_result(sess, data)
        return envelope({"benchmark_run": run.to_dict(), "audit": audit.to_dict(), "artifact": artifact.to_dict()})
    except Exception as exc:
        sess.rollback()
        raise HTTPException(status_code=400, detail=envelope(None, str(exc)))
    finally:
        sess.close()


@app.get("/api/benchmarks")
def list_benchmarks(model_id: str | None = None, limit: int = Query(200, le=1000)):
    sess = get_session()
    try:
        q = sess.query(BenchmarkRun)
        if model_id:
            q = q.filter(BenchmarkRun.model_id == model_id)
        rows = q.order_by(BenchmarkRun.created_at.desc()).limit(limit).all()
        return envelope([row.to_dict() for row in rows])
    finally:
        sess.close()


@app.get("/api/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: int):
    sess = get_session()
    try:
        row = sess.query(BenchmarkRun).filter(BenchmarkRun.id == benchmark_id).first()
        if not row:
            raise HTTPException(status_code=404, detail=envelope(None, "benchmark not found"))
        return envelope(row.to_dict())
    finally:
        sess.close()


@app.get("/api/benchmarks/{benchmark_id}/audit")
def get_benchmark_audit(benchmark_id: int):
    sess = get_session()
    try:
        run = sess.query(BenchmarkRun).filter(BenchmarkRun.id == benchmark_id).first()
        if not run:
            raise HTTPException(status_code=404, detail=envelope(None, "benchmark not found"))
        row = sess.query(BenchmarkAudit).filter(BenchmarkAudit.benchmark_run_id == benchmark_id).order_by(BenchmarkAudit.id.desc()).first()
        if not row:
            raise HTTPException(status_code=404, detail=envelope(None, "benchmark audit not found"))
        return envelope(row.to_dict())
    finally:
        sess.close()


@app.get("/api/benchmarks/{benchmark_id}/publishability")
def get_benchmark_publishability(benchmark_id: int):
    sess = get_session()
    try:
        run = sess.query(BenchmarkRun).filter(BenchmarkRun.id == benchmark_id).first()
        if not run:
            raise HTTPException(status_code=404, detail=envelope(None, "benchmark not found"))
        audit = sess.query(BenchmarkAudit).filter(BenchmarkAudit.benchmark_run_id == benchmark_id).order_by(BenchmarkAudit.id.desc()).first()
        blockers: list[str] = []
        if run.status != "completed":
            blockers.append(f"benchmark status is {run.status}")
        if not run.detailed_path:
            blockers.append("missing detailed audit artifact")
        if not audit:
            blockers.append("missing automated audit")
        elif not audit.passed:
            blockers.append(audit.notes or "automated audit failed")
        return envelope({"benchmark_run_id": run.id, "publishable": not blockers, "blockers": blockers})
    finally:
        sess.close()


@app.get("/api/models")
def list_models(limit: int = Query(100, le=500)):
    sess = get_session()
    try:
        rows = sess.query(Model).order_by(Model.updated_at.desc()).limit(limit).all()
        return envelope([row.to_dict() for row in rows])
    finally:
        sess.close()


@app.get("/api/models/watch")
def list_watched_models_route():
    sess = get_session()
    try:
        models = sess.query(ModelWatch).order_by(ModelWatch.created_at.desc()).all()
        return {"models": [m.to_dict() for m in models]}
    finally:
        sess.close()


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    sess = get_session()
    try:
        row = sess.query(Model).filter(Model.id == model_id).first()
        if not row:
            raise HTTPException(status_code=404, detail=envelope(None, "model not found"))
        return envelope(row.to_dict())
    finally:
        sess.close()


@app.get("/api/models/{model_id}/artifacts")
def get_model_artifacts(model_id: str):
    sess = get_session()
    try:
        if not sess.query(Model).filter(Model.id == model_id).first():
            raise HTTPException(status_code=404, detail=envelope(None, "model not found"))
        rows = sess.query(Artifact).filter(Artifact.model_id == model_id).order_by(Artifact.type.asc(), Artifact.path.asc()).all()
        return envelope([row.to_dict() for row in rows])
    finally:
        sess.close()


@app.get("/api/models/{model_id}/benchmarks")
def get_model_benchmarks(model_id: str):
    sess = get_session()
    try:
        if not sess.query(Model).filter(Model.id == model_id).first():
            raise HTTPException(status_code=404, detail=envelope(None, "model not found"))
        rows = sess.query(BenchmarkRun).filter(BenchmarkRun.model_id == model_id).order_by(BenchmarkRun.benchmark.asc()).all()
        audits = {
            row.benchmark_run_id: row.to_dict()
            for row in sess.query(BenchmarkAudit).filter(BenchmarkAudit.benchmark_run_id.in_([run.id for run in rows])).all()
        } if rows else {}
        return envelope([{**row.to_dict(), "audit": audits.get(row.id)} for row in rows])
    finally:
        sess.close()


@app.get("/api/models/{model_id}/benchmark-audits")
def get_model_benchmark_audits(model_id: str):
    sess = get_session()
    try:
        if not sess.query(Model).filter(Model.id == model_id).first():
            raise HTTPException(status_code=404, detail=envelope(None, "model not found"))
        run_ids = [row.id for row in sess.query(BenchmarkRun).filter(BenchmarkRun.model_id == model_id).all()]
        rows = sess.query(BenchmarkAudit).filter(BenchmarkAudit.benchmark_run_id.in_(run_ids)).order_by(BenchmarkAudit.id.asc()).all() if run_ids else []
        return envelope([row.to_dict() for row in rows])
    finally:
        sess.close()


@app.get("/api/models/{model_id}/card")
def get_model_card(model_id: str):
    sess = get_session()
    try:
        if not sess.query(Model).filter(Model.id == model_id).first():
            raise HTTPException(status_code=404, detail=envelope(None, "model not found"))
        row = sess.query(ModelCard).filter(ModelCard.model_id == model_id).order_by(ModelCard.version.desc()).first()
        if not row:
            raise HTTPException(status_code=404, detail=envelope(None, "model card not found"))
        return envelope(row.to_dict())
    finally:
        sess.close()


@app.get("/api/queue")
def list_jobs(status: Optional[str] = Query(None), limit: int = Query(50, le=200)):
    sess = get_session()
    try:
        q = sess.query(Job)
        if status:
            q = q.filter(Job.status == status)
        jobs = q.order_by(Job.created_at.desc()).limit(limit).all()
        return {"jobs": [j.to_dict() for j in jobs]}
    finally:
        sess.close()


@app.get("/api/control-plane/queue")
def list_control_plane_queue(
    status: Optional[str] = Query(None),
    kind: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    jobs = queue_list_jobs(control_plane_db_path(), status=status, kind=kind, limit=limit)
    return envelope(
        {
            "schema": "cerebellum_jobs",
            "db": str(control_plane_db_path()),
            "jobs": jobs,
        }
    )


@app.get("/api/control-plane/queue/{job_id}")
def get_control_plane_queue_job(job_id: int, tail: int = Query(40, ge=0, le=1000)):
    try:
        job = queue_get_job(control_plane_db_path(), job_id, tail=tail)
    except SystemExit as exc:
        raise HTTPException(status_code=404, detail=envelope(None, str(exc))) from exc
    return envelope(
        {
            "schema": "cerebellum_jobs",
            "db": str(control_plane_db_path()),
            "job": job,
        }
    )


@app.get("/api/model-cards")
def model_cards():
    """Discover local Cerebellum experiment artifacts outside the queue DB."""
    cards = discover_model_cards()
    return {
        "repo_root": str(REPO_ROOT),
        "count": len(cards),
        "cards": cards,
    }


@app.get("/api/agent-probes")
def agent_probes():
    """Discover quick agent behavior probe results."""
    probes = discover_agent_probes()
    return {
        "repo_root": str(REPO_ROOT),
        "count": len(probes),
        "agent_probes": probes,
        "probes": probes,
    }


@app.get("/api/devlogs")
def devlogs():
    """Discover devlogs, specs, and research notes."""
    logs = discover_devlogs()
    return {
        "repo_root": str(REPO_ROOT),
        "count": len(logs),
        "devlogs": logs,
        "logs": logs,
    }


@app.post("/api/jobs")
async def create_job(data: JobCreate):
    sess = get_session()
    try:
        steps = data.pipeline_steps
        if not steps:
            steps = [s for s in PIPELINE_PHASES if s not in ("queued", "done")]
        job = Job(
            model_name=data.model_name,
            model_source=data.model_source,
            model_path=data.model_path or data.model_name,
            status="queued",
            current_phase="queued",
            priority=data.priority,
            pipeline_steps=json.dumps(steps),
            config=json.dumps(data.config),
            output_dir=data.output_dir,
        )
        sess.add(job)
        sess.commit()
        sess.refresh(job)
        await event_bus.publish({"type": "job_update", "job": job.to_dict()})
        return job.to_dict()
    except Exception as e:
        sess.rollback()
        raise HTTPException(400, str(e))
    finally:
        sess.close()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: int):
    sess = get_session()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status == "running":
            if worker:
                worker.cancel_current()
        elif job.status == "queued":
            job.status = "cancelled"
            sess.commit()
            await event_bus.publish({"type": "job_update", "job": job.to_dict()})
        return {"status": "ok"}
    finally:
        sess.close()


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: int):
    sess = get_session()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if not job or job.status not in ("failed", "cancelled"):
            raise HTTPException(400, "Can only retry failed or cancelled jobs")
        job.status = "queued"
        job.current_phase = "queued"
        job.progress = 0.0
        job.phase_progress = 0.0
        job.error_message = ""
        job.started_at = None
        job.completed_at = None
        sess.commit()
        await event_bus.publish({"type": "job_update", "job": job.to_dict()})
        return job.to_dict()
    finally:
        sess.close()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    sess = get_session()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        return job.to_dict()
    finally:
        sess.close()


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: int, limit: int = Query(500, le=5000)):
    sess = get_session()
    try:
        logs = (
            sess.query(EventLog)
            .filter(EventLog.job_id == job_id)
            .order_by(EventLog.id.asc())
            .limit(limit)
            .all()
        )
        return {"logs": [log.to_dict() for log in logs]}
    finally:
        sess.close()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: int):
    sess = get_session()
    try:
        job = sess.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status == "running":
            raise HTTPException(400, "Cannot delete a running job")
        sess.query(EventLog).filter(EventLog.job_id == job_id).delete()
        sess.delete(job)
        sess.commit()
        return {"status": "deleted"}
    finally:
        sess.close()


@app.get("/api/schedules")
def list_schedules():
    sess = get_session()
    try:
        schedules = sess.query(Schedule).order_by(Schedule.created_at.desc()).all()
        return {"schedules": [s.to_dict() for s in schedules]}
    finally:
        sess.close()


@app.post("/api/schedules")
async def create_schedule(data: ScheduleCreate):
    sess = get_session()
    try:
        steps = data.pipeline_steps or [s for s in PIPELINE_PHASES if s not in ("queued", "done")]
        sched = Schedule(
            model_name=data.model_name,
            model_source=data.model_source,
            model_path=data.model_path or data.model_name,
            cron_expr=data.cron_expr,
            pipeline_steps=json.dumps(steps),
            config=json.dumps(data.config),
            enabled=data.enabled,
        )
        sess.add(sched)
        sess.commit()
        sess.refresh(sched)
        return sched.to_dict()
    except Exception as e:
        sess.rollback()
        raise HTTPException(400, str(e))
    finally:
        sess.close()


@app.delete("/api/schedules/{sched_id}")
def delete_schedule(sched_id: int):
    sess = get_session()
    try:
        sched = sess.query(Schedule).filter(Schedule.id == sched_id).first()
        if not sched:
            raise HTTPException(404, "Schedule not found")
        sess.delete(sched)
        sess.commit()
        return {"status": "deleted"}
    finally:
        sess.close()


@app.patch("/api/schedules/{sched_id}/toggle")
def toggle_schedule(sched_id: int):
    sess = get_session()
    try:
        sched = sess.query(Schedule).filter(Schedule.id == sched_id).first()
        if not sched:
            raise HTTPException(404, "Schedule not found")
        sched.enabled = not sched.enabled
        sess.commit()
        return sched.to_dict()
    finally:
        sess.close()


@app.get("/api/models/watch")
def list_watched_models():
    sess = get_session()
    try:
        models = sess.query(ModelWatch).order_by(ModelWatch.created_at.desc()).all()
        return {"models": [m.to_dict() for m in models]}
    finally:
        sess.close()


@app.post("/api/models/watch")
def add_watched_model(data: ModelWatchCreate):
    sess = get_session()
    try:
        existing = sess.query(ModelWatch).filter(ModelWatch.hf_model_id == data.hf_model_id).first()
        if existing:
            existing.watched = data.watched
            existing.auto_queue = data.auto_queue
            sess.commit()
            return existing.to_dict()
        m = ModelWatch(**data.model_dump())
        sess.add(m)
        sess.commit()
        sess.refresh(m)
        return m.to_dict()
    except Exception as e:
        sess.rollback()
        raise HTTPException(400, str(e))
    finally:
        sess.close()


@app.delete("/api/models/watch/{watch_id}")
def delete_watched_model(watch_id: int):
    sess = get_session()
    try:
        m = sess.query(ModelWatch).filter(ModelWatch.id == watch_id).first()
        if not m:
            raise HTTPException(404, "Watch not found")
        sess.delete(m)
        sess.commit()
        return {"status": "deleted"}
    finally:
        sess.close()


@app.get("/api/hf/search")
async def search_hf(q: str = Query(""), limit: int = Query(20, le=50)):
    """Search HuggingFace models."""
    import httpx
    url = "https://huggingface.co/api/models"
    params = {"search": q, "sort": "downloads", "direction": -1, "limit": limit}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            models = resp.json()
            result = []
            for m in models:
                card_data = m.get("cardData") or {}
                config = m.get("config") or {}
                result.append({
                    "modelId": m.get("modelId", ""),
                    "pipeline_tag": m.get("pipeline_tag", ""),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "created_at": m.get("createdAt", ""),
                    "description": card_data.get("description", ""),
                    "architectures": config.get("architectures", []),
                    "num_params": config.get("num_parameters", 0),
                })
            return {"models": result}
    except Exception as e:
        raise HTTPException(502, f"HF API error: {e}")


@app.get("/api/health")
def health():
    """Resource health and system status."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    gpu_info = {}
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    gpu_info[f"gpu_{i}"] = {
                        "name": parts[0],
                        "memory_total_mb": int(parts[1]),
                        "memory_free_mb": int(parts[2]),
                    }
    except Exception:
        pass
    sess = get_session()
    try:
        queued = sess.query(Job).filter(Job.status == "queued").count()
        running = sess.query(Job).filter(Job.status == "running").count()
        completed_today = sess.query(Job).filter(
            Job.status == "completed",
            Job.completed_at.isnot(None),
        ).count()
        return {
            "disk": {
                "total_gb": round(total / 1e9, 1),
                "free_gb": round(free / 1e9, 1),
                "free_pct": round(free / total * 100, 1),
            },
            "gpu": gpu_info,
            "queue": {"queued": queued, "running": running},
            "completed_today": completed_today,
        }
    finally:
        sess.close()


# ── WebSocket ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    q = event_bus.subscribe()
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        event_bus.unsubscribe(q)


# ── Dashboard Frontend ─────────────────────────────────────────────

DASHBOARD_HTML = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if DASHBOARD_HTML.exists():
        return HTMLResponse(DASHBOARD_HTML.read_text())
    return HTMLResponse("<h1>Cerebellum Dashboard</h1><p>Frontend not found.</p>")


def run():
    """Entry point: start the dashboard server."""
    import uvicorn
    host = os.environ.get("CEREBELLUM_HOST", "0.0.0.0")
    port = int(os.environ.get("CEREBELLUM_PORT", "8920"))
    reload = os.environ.get("CEREBELLUM_RELOAD", "0") == "1"
    uvicorn.run("osmosis.dashboard.server:app", host=host, port=port, reload=reload)
