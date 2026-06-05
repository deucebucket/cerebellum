import importlib.util
import json
from pathlib import Path

from cerebellum import benchmark_manifest, benchmark_records


def load_hle_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_hle_no_tools.py"
    spec = importlib.util.spec_from_file_location("benchmark_hle_no_tools", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_extract_hle_metrics_from_score_json():
    module = load_hle_module()

    metrics = module.extract_hle_metrics("noise\n{\"score\": 0.42}\n")

    assert metrics == {"score": 42.0, "metric_source": "score"}


def test_extract_hle_metrics_from_correct_total_json():
    module = load_hle_module()

    metrics = module.extract_hle_metrics("final: {\"correct\": 7, \"total\": 20}")

    assert metrics == {"score": 35.0, "correct": 7, "total": 20, "metric_source": "correct/total"}


def test_hle_summary_with_score_counts_as_frontier_measurement(tmp_path: Path):
    summary = tmp_path / "model_hle_no_tools_results.json"
    summary.write_text(
        json.dumps({"benchmark": "hle_no_tools", "model": "model", "score": 42.0, "metric_source": "score"}),
        encoding="utf-8",
    )

    records = benchmark_records([tmp_path])
    manifest = benchmark_manifest([tmp_path], suite="frontier", model="model")

    assert records[0]["benchmark_key"] == "hle_no_tools"
    assert records[0]["metric"] == "score"
    assert records[0]["value"] == 42.0
    assert "hle_no_tools" in manifest["measured_benchmarks"]
    assert "hle_no_tools" not in manifest["missing_measured"]
