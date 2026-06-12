import json
import sqlite3
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from cerebellum.dashboard.models import (
    Artifact,
    BenchmarkAudit,
    BenchmarkRun,
    Model,
    ModelCard,
    get_session,
    init_db,
    stable_model_id,
)
from cerebellum.dashboard.server import (
    BenchmarkResultIngest,
    get_control_plane_queue_job,
    get_benchmark_audit,
    get_benchmark_publishability,
    get_model_benchmark_audits,
    get_model,
    get_model_artifacts,
    get_model_benchmarks,
    get_model_card,
    ingest_benchmark_result,
    ingest_discovered_model_cards,
    ingest_scan,
    list_control_plane_queue,
    list_benchmarks,
    list_models,
    mobile_watch_payload,
)
from cerebellum.hillstep import DEFAULT_DB, queue_add_job


def init_tmp_db(tmp_path: Path):
    init_db(str(tmp_path / "dashboard.db"))
    return get_session()


def sample_card(tmp_path: Path) -> dict:
    root = tmp_path / "cerebellum-gemma4-test"
    bench = root / "benchmark_results"
    bench.mkdir(parents=True)
    gguf = root / "gemma-q4.gguf"
    readme = root / "README.md"
    result = bench / "gemma_arc_results.json"
    gguf.write_bytes(b"gguf")
    readme.write_text("# model\n", encoding="utf-8")
    result.write_text(json.dumps({"benchmark": "arc", "accuracy": 0.82, "total": 100}), encoding="utf-8")
    return {
        "name": "cerebellum-gemma4-test",
        "path": str(root),
        "ggufs": [{"name": gguf.name, "size_gb": 0.0, "path": str(gguf)}],
        "benchmarks": [
            {
                "file": str(result),
                "model": "cerebellum-gemma4-test",
                "benchmark": "arc",
                "metric": "accuracy",
                "value": 82.0,
                "total_problems": 100,
                "audit": {"status": "verified", "notes": ["ok"]},
            }
        ],
        "docs": [{"name": readme.name, "path": str(readme), "size_kb": 0.1}],
        "updated_at": datetime.now(timezone.utc).timestamp(),
    }


def test_mobile_watch_payload_reports_start_quant_size(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_status": "running",
                "model_family": "gemma-4",
                "model_name": "gemma-4-12b-it",
                "locked": {"blk.0.attn_q.weight": "q4_K"},
                "tested": [{"tensor": "blk.0.attn_q.weight", "winner": "q4_K", "ppl": 10.0}],
                "baseline_path": str(run_dir / "artifacts" / "current_baseline.gguf"),
                "start_quant_size_bytes": 8000000000,
                "start_quant_path": str(run_dir / "artifacts" / "current_baseline.gguf"),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "unit", "base_type": "Q4_K_M", "start_type": "q4_K", "commit_locks": True}),
        encoding="utf-8",
    )
    (run_dir / "cerebellum_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "run_start", "tensors": 2}),
                json.dumps({"event": "tensor_start", "tensor": "blk.1.ffn_down.weight", "total": 2}),
                json.dumps({"event": "quant_start", "tensor": "blk.1.ffn_down.weight", "level": "q3_K"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        json.dumps({"tensor": "blk.0.attn_q.weight", "level": "q4_K", "ppl": 10.0, "size_bytes": 8100000000}) + "\n",
        encoding="utf-8",
    )
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "current_baseline.gguf").write_bytes(b"base")
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("cerebellum.hillstep.gpu_rows", lambda: [])

    payload = mobile_watch_payload(str(run_dir))

    assert payload["schema"] == "cerebellum.mobile_watch.v1"
    assert payload["sizes"]["start_quant"]["bytes"] == 8000000000
    assert payload["sizes"]["start_quant"]["display"] == "7.451 GiB"
    assert payload["sizes"]["start_quant"]["source"] == "state.start_quant_size_bytes"
    assert payload["sizes"]["current_baseline"]["bytes"] == 4
    assert payload["progress"]["completed"] == 1
    assert payload["active"]["tensor"] == "blk.1.ffn_down.weight"


def test_mobile_watch_payload_reports_sparse_replay_final_and_benchmarks(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "sparse"
    run_dir.mkdir()
    final = run_dir / "final" / "qwen3-0.6b-cerebellum.gguf"
    final.parent.mkdir()
    final.write_bytes(b"final")
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "schema": "cerebellum.sparse_replay_state.v1",
                "run_status": "complete",
                "model_family": "qwen3",
                "model_name": "qwen3-0.6b",
                "locked": {},
                "tested": [{"tensor": "blk.0.attn_q.weight", "winner": "q2_K", "ppl": 20.0}],
                "final": {"gguf": str(final), "ppl": 19.72, "delta": -0.01, "size_bytes": 447000000},
                "benchmarks": [{"name": "arc-200", "status": "complete", "value": 41.5}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema": "cerebellum.sparse_replay.v1", "run_id": "sparse", "commit_locks": False, "measurement_mode": "qwen36-27b-v4 sparse replay"}),
        encoding="utf-8",
    )
    (run_dir / "cerebellum_events.jsonl").write_text(json.dumps({"event": "run_start", "tensors": 1}) + "\n", encoding="utf-8")
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        json.dumps({"tensor": "blk.0.attn_q.weight", "level": "q2_K", "ppl": 20.0, "size_bytes": 400000000}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("cerebellum.hillstep.gpu_rows", lambda: [])

    payload = mobile_watch_payload(str(run_dir))

    assert payload["pipeline"]["schema"] == "cerebellum.sparse_replay.v1"
    assert payload["pipeline"]["mode"] == "qwen36-27b-v4 sparse replay"
    assert payload["final"]["gguf"] == str(final)
    assert payload["final"]["size"] == "426.3 MiB"
    assert payload["benchmarks"] == [{"name": "arc-200", "status": "complete", "value": 41.5}]


def test_mobile_watch_template_has_sparse_replay_final_and_benchmark_hooks():
    template = Path("cerebellum/dashboard/templates/mobile_watch.html").read_text(encoding="utf-8")

    assert 'id="final"' in template
    assert 'id="benchmarks"' in template
    assert "data.final" in template
    assert "data.benchmarks" in template


def test_dashboard_control_plane_models_serialize_json_fields(tmp_path: Path):
    sess = init_tmp_db(tmp_path)
    try:
        model = Model(id="m", display_name="Model", modalities=json.dumps(["text", "vision"]))
        artifact = Artifact(model_id="m", type="quant_gguf", path=str(tmp_path / "m.gguf"), size_bytes=10, visibility="public")
        run = BenchmarkRun(model_id="m", benchmark="arc", results=json.dumps({"accuracy": 82.0}), server_settings=json.dumps({"ctx": 2048}))
        card = ModelCard(model_id="m", generated_markdown="# Model", hf_ready=False, version=1)
        audit = BenchmarkAudit(
            benchmark_run_id=1,
            passed=True,
            unknown_answer_count=0,
            empty_response_fallback_count=0,
            inspected_sample_ids=json.dumps(["1", "2"]),
            notes="ok",
        )
        sess.add_all([model, artifact, run, card, audit])
        sess.commit()

        assert model.to_dict()["modalities"] == ["text", "vision"]
        assert artifact.to_dict()["visibility"] == "public"
        assert run.to_dict()["results"] == {"accuracy": 82.0}
        assert run.to_dict()["server_settings"] == {"ctx": 2048}
        assert card.to_dict()["generated_markdown"] == "# Model"
        assert audit.to_dict()["passed"] is True
        assert audit.to_dict()["inspected_sample_ids"] == ["1", "2"]
    finally:
        sess.close()


def test_ingest_discovered_model_cards_is_idempotent(tmp_path: Path):
    sess = init_tmp_db(tmp_path)
    card = sample_card(tmp_path)
    model_id = stable_model_id(card["name"])
    try:
        first = ingest_discovered_model_cards(sess, [card])
        second = ingest_discovered_model_cards(sess, [card])

        assert first == {"models": 1, "artifacts": 3, "benchmark_runs": 1}
        assert second == first
        assert sess.query(Model).count() == 1
        assert sess.query(Artifact).count() == 3
        assert sess.query(BenchmarkRun).count() == 1
        assert sess.query(BenchmarkAudit).count() == 1
        assert sess.query(ModelCard).count() == 1
        assert sess.query(Model).filter(Model.id == model_id).first().display_name == card["name"]
        run = sess.query(BenchmarkRun).first()
        assert run.audit_id is not None
        assert sess.query(BenchmarkAudit).filter(BenchmarkAudit.benchmark_run_id == run.id).first().passed is True
    finally:
        sess.close()


def test_dashboard_model_routes_return_envelopes(tmp_path: Path):
    sess = init_tmp_db(tmp_path)
    card = sample_card(tmp_path)
    model_id = stable_model_id(card["name"])
    try:
        ingest_discovered_model_cards(sess, [card])
    finally:
        sess.close()

    models = list_models(limit=100)
    model = get_model(model_id)
    artifacts = get_model_artifacts(model_id)
    benchmarks = get_model_benchmarks(model_id)
    audits = get_model_benchmark_audits(model_id)
    card_payload = get_model_card(model_id)

    assert models["error"] is None
    assert models["data"][0]["id"] == model_id
    assert model["data"]["display_name"] == "cerebellum-gemma4-test"
    assert {item["type"] for item in artifacts["data"]} == {"benchmark_result", "model_card", "quant_gguf"}
    assert benchmarks["data"][0]["results"]["value"] == 82.0
    assert benchmarks["data"][0]["audit"]["passed"] is True
    assert audits["data"][0]["passed"] is True
    assert audits["data"][0]["parse_method"] == "discovery_summary"
    assert card_payload["data"]["model_id"] == model_id


def test_ingest_scan_uses_discovery_and_envelope(tmp_path: Path, monkeypatch):
    sess = init_tmp_db(tmp_path)
    sess.close()
    card = sample_card(tmp_path)
    monkeypatch.setattr("cerebellum.dashboard.server.discover_model_cards", lambda: [card])

    payload = ingest_scan()

    assert payload == {"data": {"models": 1, "artifacts": 3, "benchmark_runs": 1}, "error": None}


def test_ingest_benchmark_result_creates_publishable_audit(tmp_path: Path):
    sess = init_tmp_db(tmp_path)
    sess.close()
    result = tmp_path / "benchmark_results" / "model_arc_results.json"
    detail = tmp_path / "benchmark_results" / "model_arc_detailed.jsonl"
    result.parent.mkdir()
    result.write_text(json.dumps({"benchmark": "arc", "model": "model", "accuracy": 0.82, "total": 2}), encoding="utf-8")
    detail.write_text(
        "\n".join(
            [
                json.dumps({"correct": True, "predicted": "A", "raw_response": "A"}),
                json.dumps({"correct": True, "predicted": "B", "raw_response": "B"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = ingest_benchmark_result(BenchmarkResultIngest(path=str(result), detailed_path=str(detail)))
    run_id = payload["data"]["benchmark_run"]["id"]
    publishability = get_benchmark_publishability(run_id)
    audit = get_benchmark_audit(run_id)
    listed = list_benchmarks(model_id=stable_model_id("model"), limit=100)

    assert payload["error"] is None
    assert payload["data"]["benchmark_run"]["benchmark"] == "arc"
    assert payload["data"]["benchmark_run"]["results"]["value"] == 82.0
    assert payload["data"]["audit"]["passed"] is True
    assert payload["data"]["artifact"]["sha256"]
    assert audit["data"]["parse_method"] == "mcq_jsonl"
    assert publishability["data"] == {"benchmark_run_id": run_id, "publishable": True, "blockers": []}
    assert listed["data"][0]["id"] == run_id


def test_ingest_benchmark_result_blocks_without_detail(tmp_path: Path):
    sess = init_tmp_db(tmp_path)
    sess.close()
    result = tmp_path / "model_evalplus_results.json"
    result.write_text(json.dumps({"benchmark": "evalplus_humaneval_plus", "model": "model", "pass_at_1_plus": 0.5}), encoding="utf-8")

    payload = ingest_benchmark_result(BenchmarkResultIngest(path=str(result)))
    publishability = get_benchmark_publishability(payload["data"]["benchmark_run"]["id"])

    assert payload["data"]["audit"]["passed"] is False
    assert publishability["data"]["publishable"] is False
    assert "missing detailed audit artifact" in publishability["data"]["blockers"]


def test_dashboard_exposes_control_plane_queue_jobs(tmp_path: Path, monkeypatch):
    db = tmp_path / "cerebellum.db"
    log = tmp_path / "pipeline.log"
    log.write_text("first\nsecond\nthird\n", encoding="utf-8")
    job = queue_add_job(
        Namespace(
            db=str(db),
            kind="pipeline",
            manifest=None,
            from_phase=None,
            until_phase=None,
            command=None,
            payload_json=json.dumps({"manifest": "pipeline.json", "pipeline": "unit", "log": str(log)}),
            label="unit pipeline",
            status="queued",
            priority=5,
            notes="bridge test",
        )
    )
    conn = sqlite3.connect(db)
    try:
        with conn:
            conn.execute("UPDATE cerebellum_jobs SET log = ? WHERE id = ?", (str(log), job["id"]))
    finally:
        conn.close()
    monkeypatch.setattr("cerebellum.dashboard.server.DB_PATH", str(db))

    listed = list_control_plane_queue(status="queued", kind="pipeline", limit=10)
    fetched = get_control_plane_queue_job(job["id"], tail=2)

    assert listed["error"] is None
    assert listed["data"]["schema"] == "cerebellum_jobs"
    assert listed["data"]["jobs"][0]["id"] == job["id"]
    assert listed["data"]["jobs"][0]["kind"] == "pipeline"
    assert listed["data"]["jobs"][0]["payload"]["pipeline"] == "unit"
    assert fetched["data"]["job"]["label"] == "unit pipeline"
    assert fetched["data"]["job"]["log_tail"] == "second\nthird"


def test_dashboard_control_plane_default_db_matches_cli_queue():
    from cerebellum.dashboard import server

    assert server.DB_PATH == DEFAULT_DB
