import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "compare_benchmark_results.py"
SPEC = importlib.util.spec_from_file_location("compare_benchmark_results", SCRIPT_PATH)
compare_benchmark_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = compare_benchmark_results
SPEC.loader.exec_module(compare_benchmark_results)

load_result = compare_benchmark_results.load_result
load_results = compare_benchmark_results.load_results
collapse_latest = compare_benchmark_results.collapse_latest
render_markdown = compare_benchmark_results.render_markdown
build_summary = compare_benchmark_results.build_summary
collect_paths = compare_benchmark_results.collect_paths
audit_artifacts = compare_benchmark_results.audit_artifacts


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def test_load_result_normalizes_accuracy_and_pass_at_1(tmp_path):
    arc = write_json(
        tmp_path / "model_a_arc_results.json",
        {"benchmark": "arc", "model": "model_a", "accuracy": 0.75, "correct": 3, "total": 4},
    )
    humaneval = write_json(
        tmp_path / "model_a_humaneval_results.json",
        {"benchmark": "humaneval", "model": "model_a", "pass_at_1": 0.625, "total_problems": 164},
    )

    arc_row = load_result(arc)
    humaneval_row = load_result(humaneval)

    assert arc_row.score == 75.0
    assert arc_row.score_field == "accuracy"
    assert arc_row.correct == 3
    assert arc_row.total == 4
    assert arc_row.audit_status == "needs_audit"
    assert arc_row.missing_artifacts == ("detailed_jsonl",)
    assert humaneval_row.score == 62.5
    assert humaneval_row.score_field == "pass_at_1"
    assert humaneval_row.total == 164
    assert humaneval_row.missing_artifacts == ("samples_jsonl", "eval_results")


def test_load_result_infers_model_and_benchmark_from_filename(tmp_path):
    path = write_json(tmp_path / "cerebellum_v2_mmlu_redux_results.json", {"accuracy": 57.25})

    row = load_result(path)

    assert row.model == "cerebellum_v2"
    assert row.benchmark == "mmlu_redux"
    assert row.score == 57.25


def test_collapse_latest_keeps_newest_timestamp(tmp_path):
    old = write_json(
        tmp_path / "old.json",
        {"benchmark": "arc", "model": "m", "accuracy": 70.0, "timestamp": "2026-01-01 00:00"},
    )
    new = write_json(
        tmp_path / "new.json",
        {"benchmark": "arc", "model": "m", "accuracy": 72.0, "timestamp": "2026-01-02 00:00"},
    )

    rows = collapse_latest(load_results([old, new], models=None, benchmarks=None))

    assert len(rows) == 1
    assert rows[0].score == 72.0


def test_render_markdown_includes_table_deltas_bars_and_sources(tmp_path):
    paths = [
        write_json(tmp_path / "base_arc_results.json", {"benchmark": "arc", "model": "base", "accuracy": 80.0}),
        write_json(tmp_path / "new_arc_results.json", {"benchmark": "arc", "model": "new", "accuracy": 84.0}),
        write_json(tmp_path / "base_humaneval_results.json", {"benchmark": "humaneval", "model": "base", "pass_at_1_pct": 60.0}),
        write_json(tmp_path / "new_humaneval_results.json", {"benchmark": "humaneval", "model": "new", "pass_at_1_pct": 63.0}),
    ]
    rows = collapse_latest(load_results(paths, models=None, benchmarks=None))

    markdown = render_markdown(rows, baseline_model="base")
    summary = build_summary(rows, baseline_model="base")

    assert "| `new` | 84.00 | 63.00 |" in markdown
    assert "| `new` | +4.00 (+5.0%) | +3.00 (+5.0%) |" in markdown
    assert "## ASCII Bars" in markdown
    assert "## Sources" in markdown
    assert "audit=needs_audit" in markdown
    assert summary["deltas"]["new"]["arc"]["absolute"] == 4.0


def test_collect_paths_supports_recursive_discovery(tmp_path):
    nested = tmp_path / "model" / "benchmark_results"
    nested.mkdir(parents=True)
    result = write_json(nested / "m_arc_results.json", {"benchmark": "arc", "model": "m", "accuracy": 80.0})

    assert collect_paths([tmp_path], "*_results.json", recursive=False) == []
    assert collect_paths([tmp_path], "*_results.json", recursive=True) == [result]


def test_audit_artifacts_detects_matching_files(tmp_path):
    arc = tmp_path / "m_arc_results.json"
    arc.write_text("{}")
    (tmp_path / "m_arc_detailed.jsonl").write_text("{}\n")
    humaneval = tmp_path / "m_humaneval_results.json"
    humaneval.write_text("{}")
    (tmp_path / "m_humaneval_samples.jsonl").write_text("{}\n")
    (tmp_path / "m_humaneval_samples.jsonl_results.jsonl").write_text("{}\n")

    assert audit_artifacts(arc, "arc_challenge")[0] == "ok"
    assert audit_artifacts(humaneval, "humaneval")[0] == "ok"
