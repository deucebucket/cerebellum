import json
from datetime import datetime, timezone
from pathlib import Path

from osmosis.dashboard.models import (
    Artifact,
    BenchmarkRun,
    Model,
    ModelCard,
    get_session,
    init_db,
    stable_model_id,
)
from osmosis.dashboard.server import (
    get_model,
    get_model_artifacts,
    get_model_benchmarks,
    get_model_card,
    ingest_discovered_model_cards,
    ingest_scan,
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
        sess.add_all([model, artifact, run, card])
        sess.commit()

        assert model.to_dict()["modalities"] == ["text", "vision"]
        assert artifact.to_dict()["visibility"] == "public"
        assert run.to_dict()["results"] == {"accuracy": 82.0}
        assert run.to_dict()["server_settings"] == {"ctx": 2048}
        assert card.to_dict()["generated_markdown"] == "# Model"
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
        assert sess.query(ModelCard).count() == 1
        assert sess.query(Model).filter(Model.id == model_id).first().display_name == card["name"]
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
    card_payload = get_model_card(model_id)

    assert models["error"] is None
    assert models["data"][0]["id"] == model_id
    assert model["data"]["display_name"] == "cerebellum-gemma4-test"
    assert {item["type"] for item in artifacts["data"]} == {"benchmark_result", "model_card", "quant_gguf"}
    assert benchmarks["data"][0]["results"]["value"] == 82.0
    assert card_payload["data"]["model_id"] == model_id


def test_ingest_scan_uses_discovery_and_envelope(tmp_path: Path, monkeypatch):
    sess = init_tmp_db(tmp_path)
    sess.close()
    card = sample_card(tmp_path)
    monkeypatch.setattr("osmosis.dashboard.server.discover_model_cards", lambda: [card])

    payload = ingest_scan()

    assert payload == {"data": {"models": 1, "artifacts": 3, "benchmark_runs": 1}, "error": None}
