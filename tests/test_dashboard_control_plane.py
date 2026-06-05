import json
from datetime import datetime, timezone
from pathlib import Path

from osmosis.dashboard.models import (
    Artifact,
    BenchmarkAudit,
    BenchmarkRun,
    Model,
    ModelCard,
    get_session,
    init_db,
    stable_model_id,
)
from osmosis.dashboard.server import (
    BenchmarkResultIngest,
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
    list_benchmarks,
    list_models,
)


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
    monkeypatch.setattr("osmosis.dashboard.server.discover_model_cards", lambda: [card])

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
