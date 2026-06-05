import json
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import osmosis.hillstep as hillstep
from cerebellum import (
    EventLog,
    active_work_status,
    ablation_analyze_cmd,
    analyze_ablation_input,
    artifact_inventory,
    artifact_inventory_args_from_query,
    artifact_inventory_cmd,
    artifact_inventory_markdown,
    benchmark_audit,
    benchmark_audit_cmd,
    benchmark_audit_markdown,
    benchmark_manifest,
    benchmark_manifest_cmd,
    benchmark_manifest_markdown,
    benchmark_plan,
    benchmark_plan_cmd,
    benchmark_plan_markdown,
    benchmark_rebench_plan,
    benchmark_rebench_plan_args_from_query,
    benchmark_rebench_plan_markdown,
    benchmark_report,
    benchmark_report_markdown,
    cerebellum_metadata_block,
    clear_terminal_markers,
    cleanup_cmd,
    compare_locks,
    build_recovery_plan,
    build_watch_model,
    compare_gguf_types,
    compare_gguf_types_markdown,
    discover_projects,
    doctor_cmd,
    eta_grid_values,
    github_upload_plan,
    grid_watch_cmd,
    hf_model_stats,
    hf_model_stats_markdown,
    hf_stats_args_from_query,
    inspect_gguf_types,
    is_quantizable_tensor,
    locked_layer_lines,
    package_files,
    package_manifest,
    parse_args,
    pipeline_plan,
    pipeline_plan_args_from_query,
    pipeline_plan_markdown,
    pipeline_plan_cmd,
    pipeline_run_args_from_query,
    pipeline_run_cmd,
    pipeline_run_plan,
    public_audit,
    public_audit_cmd,
    public_audit_markdown,
    public_export_cmd,
    public_export_plan,
    public_export_plan_args_from_query,
    public_export_markdown,
    read_tensor_type_map,
    task_profiles_cmd,
    task_profiles_markdown,
    rollback_cmd,
    resolve_run_dir,
    public_report_summary,
    sanitize_process_cmd,
    stop_target_pids,
    tensor_type_line,
    upload_github_sidecars,
    watch_cmd,
    write_tensor_types_map,
)


def test_project_root_alias_parses_as_data_root():
    args = parse_args(["project", "--root", "/tmp/cerebellum-runs", "--json"])

    assert args.cmd == "project"
    assert args.data_root == "/tmp/cerebellum-runs"
    assert args.json is True


def test_inspect_gguf_types_command_parses():
    args = parse_args(["inspect-gguf-types", "/tmp/model.gguf", "--by-layer", "--json"])

    assert args.cmd == "inspect-gguf-types"
    assert args.gguf == "/tmp/model.gguf"
    assert args.by_layer is True
    assert args.json is True


def test_compare_gguf_types_command_parses():
    args = parse_args(["compare-gguf-types", "base.gguf", "cand.gguf", "--baseline-label", "q4", "--candidate-label", "dynamic", "--reference-map", "types.txt", "--json"])

    assert args.cmd == "compare-gguf-types"
    assert args.baseline == "base.gguf"
    assert args.candidate == "cand.gguf"
    assert args.baseline_label == "q4"
    assert args.candidate_label == "dynamic"
    assert args.reference_map == "types.txt"
    assert args.json is True


def test_compare_locks_command_parses():
    args = parse_args(["compare-locks", "/tmp/run", "--against", "/tmp/archive", "--json"])

    assert args.cmd == "compare-locks"
    assert args.run_dir == "/tmp/run"
    assert args.against == "/tmp/archive"
    assert args.json is True


def test_watch_public_flag_parses():
    args = parse_args(["watch", "/tmp/run", "--public", "--once", "--plain"])

    assert args.cmd == "watch"
    assert args.public is True


def test_public_tui_watch_is_rejected():
    args = parse_args(["watch", "/tmp/run", "--public", "--tui"])

    try:
        watch_cmd(args)
    except SystemExit as exc:
        assert "watch --public --tui is not supported" in str(exc)
    else:
        raise AssertionError("public TUI watch should be rejected")


def test_watch_run_dir_is_optional_for_single_active_run():
    args = parse_args(["watch", "--public", "--once", "--plain"])

    assert args.cmd == "watch"
    assert args.run_dir is None
    assert args.public is True


def test_resolve_run_dir_defaults_to_single_live_run(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"model_name":"gemma-4-12b-it"}', encoding="utf-8")
    (run_dir / "state.json").write_text('{"run_status":"running"}', encoding="utf-8")

    monkeypatch.setattr("osmosis.hillstep.known_run_dirs", lambda: [run_dir])
    monkeypatch.setattr("osmosis.hillstep.run_is_live", lambda path: path == run_dir)

    assert resolve_run_dir(None) == run_dir
    assert resolve_run_dir("gemma-4-12b-it") == run_dir


def test_active_work_status_marks_missing_started_process_interrupted():
    state = {"run_status": "running"}
    active = {
        "event": "ppl_start",
        "pid": "123456",
        "timestamp_utc": "2026-06-04T00:00:00+00:00",
    }

    status = active_work_status(state, [active], [], active)

    assert status["health"] == "interrupted"
    assert status["stale"] is True
    assert status["expected_pid"] == "123456"
    assert status["expected_pid_alive"] is False


def test_clear_terminal_markers_removes_stale_run_markers(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ["STOPPED", "ABORTED", "COMPLETE"]:
        (run_dir / name).write_text("old\n", encoding="utf-8")

    removed = clear_terminal_markers(run_dir)

    assert removed == ["STOPPED", "ABORTED", "COMPLETE"]
    assert not any((run_dir / name).exists() for name in removed)


def test_stop_target_pids_includes_detached_run_processes(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [{"pid": 100}]

    monkeypatch.setattr("osmosis.hillstep.child_pids", lambda pid: [pid + 1] if pid == 100 else [])
    monkeypatch.setattr(
        "osmosis.hillstep.process_rows_for_run",
        lambda _run_dir: [
            {"kind": "runner", "pid": "100"},
            {"kind": "ppl", "pid": "300"},
            {"kind": "quantize", "pid": "400"},
            {"kind": "process", "pid": "500"},
        ],
    )

    targets = stop_target_pids(run_dir, events)

    assert targets == [101, 100, 300, 400]


def test_benchmark_report_compares_result_jsons(tmp_path: Path):
    baseline = tmp_path / "baseline_arc_results.json"
    candidate = tmp_path / "candidate_arc_results.json"
    humaneval = tmp_path / "candidate_humaneval_results.json"
    baseline.write_text(
        json.dumps({"benchmark": "arc_challenge", "model": "q4", "accuracy": 80.0, "correct": 8, "total": 10}),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps({"benchmark": "arc_challenge", "model": "cerebellum", "accuracy": 82.5, "correct": 33, "total": 40}),
        encoding="utf-8",
    )
    humaneval.write_text(
        json.dumps({"benchmark": "humaneval", "model": "cerebellum", "pass_at_1": 0.75, "total_problems": 164}),
        encoding="utf-8",
    )

    report = benchmark_report([tmp_path], baseline="q4")
    markdown = benchmark_report_markdown(report)
    labeled = benchmark_report([f"v1={tmp_path}"])

    assert report["models"] == ["cerebellum", "q4"]
    assert labeled["models"] == ["v1"]
    assert any(row["benchmark"] == "arc_challenge" and row["delta"] == 2.5 for row in report["deltas"])
    assert "| arc_challenge | accuracy | 82.50% (33/40) | 80.00% (8/10) |" in markdown
    assert "| humaneval | pass@1 | 75.00% | - |" in markdown
    assert "## Bars" in markdown


def test_benchmark_report_leaderboard_scores_size_density(tmp_path: Path):
    mmlu_pro = tmp_path / "cerebellum_mmlu_pro_results.json"
    gpqa = tmp_path / "cerebellum_gpqa_diamond_results.json"
    lcb = tmp_path / "baseline_livecodebench_v6_results.json"
    mmlu_pro.write_text(
        json.dumps(
            {
                "benchmark": "mmlu_pro",
                "model": "cerebellum",
                "accuracy": 0.72,
                "size_gib": 8.0,
                "bpw": 4.25,
                "quant_recipe": "Cerebellum-Q4CPU",
                "tensor_map": "tensor_types.txt",
                "gguf_sha256": "abc123",
                "runtime": "llama-server",
            }
        ),
        encoding="utf-8",
    )
    gpqa.write_text(
        json.dumps({"benchmark": "gpqa_diamond", "model": "cerebellum", "score": 66.0}),
        encoding="utf-8",
    )
    lcb.write_text(
        json.dumps({"benchmark": "livecodebench_v6", "model": "baseline", "pass_at_1": 0.5}),
        encoding="utf-8",
    )

    report = benchmark_report([tmp_path], suite="frontier", leaderboard=True, sizes={"baseline": 10.0})
    markdown = benchmark_report_markdown(report, include_bars=False)

    assert report["suite"]["benchmarks"] == ["mmlu_pro", "gpqa_diamond", "mmmlu", "hle_no_tools", "livecodebench_v6"]
    assert report["suite"]["purpose"].startswith("frontier public leaderboard core")
    assert report["suite"]["average_policy"].startswith("weighted mean of measured quality-percentage")
    assert report["leaderboard"][0]["model"] == "cerebellum"
    assert report["suite"]["weights"] == {
        "mmlu_pro": 1.0,
        "gpqa_diamond": 1.0,
        "mmmlu": 1.0,
        "hle_no_tools": 1.0,
        "livecodebench_v6": 1.0,
    }
    assert report["leaderboard"][0]["average_score"] == 69.0
    assert report["leaderboard"][0]["score_per_gib"] == 8.625
    assert report["leaderboard"][0]["total_weight"] == 2.0
    assert report["release_metadata"]["cerebellum"]["bpw"] == 4.25
    assert report["release_metadata"]["cerebellum"]["quant_recipe"] == "Cerebellum-Q4CPU"
    assert "| cerebellum | 69.00% | 2 | 8.00 | 8.62 |" in markdown
    assert "| cerebellum | 8.00 | 4.25 | Cerebellum-Q4CPU | tensor_types.txt | abc123 | llama-server |" in markdown
    assert "| baseline | 50.00% | 1 | 10.00 | 5.00 |" in markdown
    assert "Purpose: frontier public leaderboard core" in markdown
    assert "Average: weighted mean of measured quality-percentage benchmarks only" in markdown
    assert "Weights: mmlu_pro=1" in markdown


def test_benchmark_report_leaderboard_uses_explicit_weights(tmp_path: Path):
    mmlu_pro = tmp_path / "cerebellum_mmlu_pro_results.json"
    gpqa = tmp_path / "cerebellum_gpqa_diamond_results.json"
    mmlu_pro.write_text(
        json.dumps({"benchmark": "mmlu_pro", "model": "cerebellum", "accuracy": 0.72, "size_gib": 8.0}),
        encoding="utf-8",
    )
    gpqa.write_text(
        json.dumps({"benchmark": "gpqa_diamond", "model": "cerebellum", "score": 66.0}),
        encoding="utf-8",
    )

    report = benchmark_report([tmp_path], suite="frontier", leaderboard=True, weights={"gpqa_diamond": 2.0})

    assert report["suite"]["weights"]["mmlu_pro"] == 1.0
    assert report["suite"]["weights"]["gpqa_diamond"] == 2.0
    assert report["leaderboard"][0]["average_score"] == 68.0
    assert report["leaderboard"][0]["score_per_gib"] == 8.5


def test_benchmark_report_command_parses():
    args = parse_args(["benchmark-report", "benchmarks/qwen36-27b", "--baseline", "q4", "--leaderboard", "--suite", "frontier", "--size", "q4=8.0", "--weight", "gpqa_diamond=2", "--json"])

    assert args.cmd == "benchmark-report"
    assert args.paths == ["benchmarks/qwen36-27b"]
    assert args.baseline == "q4"
    assert args.leaderboard is True
    assert args.suite == "frontier"
    assert args.size == ["q4=8.0"]
    assert args.weight == ["gpqa_diamond=2"]
    assert args.json is True


def test_benchmark_plan_lists_commands_and_pending_frontier():
    release = benchmark_plan("release", model="gemma4_12b", port=18080, results_dir="out")
    markdown = benchmark_plan_markdown(release)
    frontier = benchmark_plan("frontier", model="gemma4_12b", port=18080, results_dir="out")

    arc = next(row for row in release["rows"] if row["benchmark"] == "arc")
    humaneval = next(row for row in release["rows"] if row["benchmark"] == "humaneval")
    frontier_status = {row["benchmark"]: row["status"] for row in frontier["rows"]}
    mmlu_pro = next(row for row in frontier["rows"] if row["benchmark"] == "mmlu_pro")
    gpqa = next(row for row in frontier["rows"] if row["benchmark"] == "gpqa_diamond")
    hle = next(row for row in frontier["rows"] if row["benchmark"] == "hle_no_tools")
    livecodebench = next(row for row in frontier["rows"] if row["benchmark"] == "livecodebench_v6")

    assert "BENCH_MODEL=gemma4_12b" in arc["command"]
    assert frontier["purpose"].startswith("frontier public leaderboard core")
    assert "BENCH_PORT=18080" in arc["command"]
    assert "BENCH_WORKERS=4" in arc["command"]
    assert "scripts/benchmark_arc.py" in arc["command"]
    assert humaneval["workers"] == 1
    assert "BENCH_MAX_TOKENS=4096" in humaneval["command"]
    assert frontier_status == {
        "mmlu_pro": "implemented",
        "gpqa_diamond": "implemented",
        "mmmlu": "implemented",
        "hle_no_tools": "implemented",
        "livecodebench_v6": "implemented",
    }
    assert "LM_EVAL_TASK=mmlu_pro" in mmlu_pro["command"]
    assert "LM_EVAL_TASK=gpqa_diamond_zeroshot" in gpqa["command"]
    assert "BENCH_MAX_TOKENS=8192" in hle["command"]
    assert "scripts/benchmark_livecodebench_v6.py" in livecodebench["command"]
    assert release["readiness"]["ready"] is False
    assert frontier["readiness"]["ready"] is True
    assert frontier["readiness"]["implemented"] == 5
    assert frontier["readiness"]["blockers"] == []
    assert "readiness: `blocked`" in markdown
    assert "purpose: `model-card release proof" in markdown
    assert "| arc | implemented | 4 |" in markdown
    assert "out/gemma4_12b_arc_results.json" in markdown


def test_benchmark_plan_capability_suite_includes_extra_clean_benchmarks():
    plan = benchmark_plan("capability", model="gemma4_12b", port=18080, results_dir="out")
    names = [row["benchmark"] for row in plan["rows"]]

    assert names == [
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
    ]
    assert plan["purpose"].startswith("expanded capability board")
    assert plan["readiness"]["implemented"] == 5
    assert len(plan["readiness"]["blockers"]) == 5


def test_benchmark_plan_command_parses():
    args = parse_args(["benchmark-plan", "--suite", "frontier", "--model", "m", "--port", "18080", "--results-dir", "out", "--require-ready", "--json"])

    assert args.cmd == "benchmark-plan"
    assert args.suite == "frontier"
    assert args.model == "m"
    assert args.port == 18080
    assert args.results_dir == "out"
    assert args.require_ready is True
    assert args.json is True


def test_benchmark_rebench_plan_lists_affected_models_and_audits():
    plan = benchmark_rebench_plan("humaneval", "rebench", 18080, None, "#35")
    markdown = benchmark_rebench_plan_markdown(plan)

    assert plan["schema"] == "cerebellum.benchmark_rebench_plan.v1"
    assert plan["model_count"] == 12
    first = plan["jobs"][0]
    assert first["repo"] == "deucebucket/Qwen3.5-122B-A10B-Cerebellum-GGUF"
    assert first["published"] == "2026-05-02"
    assert {row["benchmark"] for row in first["benchmarks"]} == {"humaneval"}
    assert "BENCH_WORKERS=1" in first["benchmarks"][0]["command"]
    assert "BENCH_MAX_TOKENS=4096" in first["benchmarks"][0]["command"]
    assert "benchmark-audit" in first["post_run"]["audit"]
    assert "scores corrected on 2026-06-05" in first["post_run"]["model_card_note"]
    assert "# Benchmark Rebench Plan" in markdown
    assert "Qwen3.5-122B-A10B-Cerebellum-GGUF" in markdown


def test_benchmark_rebench_plan_custom_release_suite():
    plan = benchmark_rebench_plan(
        "release",
        "out",
        19000,
        ["deucebucket/Custom-Cerebellum-GGUF"],
        "#99",
    )
    job = plan["jobs"][0]
    benchmarks = {row["benchmark"] for row in job["benchmarks"]}

    assert plan["model_count"] == 1
    assert job["model"] == "custom-cerebellum-gguf"
    assert {"arc", "hellaswag", "mmlu_redux", "humaneval"} == benchmarks
    assert "BENCH_PORT=19000" in job["benchmarks"][0]["command"]
    assert "see #99" in job["post_run"]["model_card_note"]


def test_benchmark_rebench_plan_command_parses():
    args = parse_args(
        [
            "benchmark-rebench-plan",
            "--suite",
            "release",
            "--results-root",
            "out",
            "--port",
            "19000",
            "--model",
            "deucebucket/custom",
            "--correction-issue",
            "#99",
            "--json",
        ]
    )

    assert args.cmd == "benchmark-rebench-plan"
    assert args.suite == "release"
    assert args.results_root == "out"
    assert args.port == 19000
    assert args.model == ["deucebucket/custom"]
    assert args.correction_issue == "#99"
    assert args.json is True


def test_benchmark_rebench_plan_api_query_args_validate():
    args = benchmark_rebench_plan_args_from_query(
        {
            "suite": ["release"],
            "results_root": ["out"],
            "port": ["19000"],
            "model": ["deucebucket/a", "deucebucket/b"],
            "correction_issue": ["#99"],
        }
    )

    assert args.suite == "release"
    assert args.results_root == "out"
    assert args.port == 19000
    assert args.model == ["deucebucket/a", "deucebucket/b"]
    assert args.correction_issue == "#99"

    try:
        benchmark_rebench_plan_args_from_query({"suite": ["frontier"]})
    except ValueError as exc:
        assert "suite must be humaneval or release" in str(exc)
    else:
        raise AssertionError("invalid rebench suite should fail")

    try:
        benchmark_rebench_plan_args_from_query({"port": ["many"]})
    except ValueError as exc:
        assert "port must be an integer" in str(exc)
    else:
        raise AssertionError("invalid rebench port should fail")


def test_provenance_and_finalize_private_flags_parse():
    provenance = parse_args(["provenance", "--run-dir", "/tmp/run", "--private", "--hash-files"])
    finalize = parse_args(["finalize", "--run-dir", "/tmp/run", "--private", "--json"])

    assert provenance.private is True
    assert provenance.hash_files is True
    assert finalize.private is True
    assert finalize.json is True


def test_benchmark_plan_cmd_exits_when_required_suite_not_ready(capsys):
    args = parse_args(["benchmark-plan", "--suite", "capability", "--require-ready"])

    try:
        benchmark_plan_cmd(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("benchmark-plan --require-ready should fail for pending capability adapters")

    assert "Readiness Blockers" in capsys.readouterr().out


def test_benchmark_manifest_hashes_artifacts_and_tracks_missing_suite_items(tmp_path: Path):
    summary = tmp_path / "model_arc_results.json"
    detail = tmp_path / "model_arc_detailed.jsonl"
    summary.write_text(json.dumps({"benchmark": "arc", "model": "model", "accuracy": 0.8, "bpw": 4.5}), encoding="utf-8")
    detail.write_text(json.dumps({"correct": True, "predicted": "A"}) + "\n", encoding="utf-8")

    manifest = benchmark_manifest([tmp_path], suite="release", model="model")
    markdown = benchmark_manifest_markdown(manifest)

    kinds = {item["name"]: item["kind"] for item in manifest["artifacts"]}
    assert manifest["schema"] == "cerebellum.benchmark_manifest.v1"
    assert manifest["model"] == "model"
    assert manifest["suite_purpose"].startswith("model-card release proof")
    assert kinds["model_arc_results.json"] == "summary"
    assert kinds["model_arc_detailed.jsonl"] == "detail"
    assert manifest["artifacts"][0]["sha256"]
    assert "arc" in manifest["measured_benchmarks"]
    assert "hellaswag" in manifest["missing_measured"]
    assert manifest["release_metadata"]["model"]["bpw"] == 4.5
    assert "# Benchmark Artifact Manifest" in markdown
    assert "purpose: `model-card release proof" in markdown
    assert "model_arc_detailed.jsonl" in markdown


def test_benchmark_manifest_command_parses():
    args = parse_args(["benchmark-manifest", "benchmark_results", "--suite", "release", "--model", "m", "--output", "manifest.json", "--require-complete", "--json"])

    assert args.cmd == "benchmark-manifest"
    assert args.paths == ["benchmark_results"]
    assert args.suite == "release"
    assert args.model == "m"
    assert args.output == "manifest.json"
    assert args.require_complete is True
    assert args.json is True


def test_benchmark_manifest_cmd_requires_complete_suite(tmp_path: Path):
    summary = tmp_path / "model_arc_results.json"
    summary.write_text(json.dumps({"benchmark": "arc", "model": "model", "accuracy": 0.8}), encoding="utf-8")
    args = parse_args(["benchmark-manifest", str(tmp_path), "--suite", "frontier", "--require-complete", "--json"])

    try:
        benchmark_manifest_cmd(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("benchmark-manifest --require-complete should fail when suite results are missing")


def test_benchmark_audit_flags_mcq_empty_and_unknown(tmp_path: Path):
    detail = tmp_path / "arc_detailed.jsonl"
    detail.write_text(
        "\n".join(
            [
                json.dumps({"correct": True, "predicted": "A", "raw_response": "A"}),
                json.dumps({"correct": False, "predicted": "?", "raw_response": ""}),
                json.dumps({"correct": False, "predicted": "?", "raw_response": ""}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = benchmark_audit([str(detail)], fail_empty_pct=50.0, fail_unknown_pct=50.0)
    markdown = benchmark_audit_markdown(report)

    assert report["blocked"] is True
    assert report["files"][0]["kind"] == "mcq"
    assert report["files"][0]["counts"]["unknown"] == 2
    assert "empty responses above threshold" in {item["reason"] for item in report["failures"]}
    assert "Benchmark audit blocked" in markdown


def test_benchmark_audit_flags_evalplus_pass_only(tmp_path: Path):
    samples = tmp_path / "model_evalplus_samples.jsonl"
    samples.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "HumanEval/0", "prompt": "def f(x):\n", "completion": "return x + 1"}),
                json.dumps({"task_id": "HumanEval/1", "prompt": "def g(x):\n", "completion": "pass"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = benchmark_audit([str(samples)], fail_pass_only_pct=10.0)

    assert report["blocked"] is True
    assert report["files"][0]["kind"] == "evalplus"
    assert report["files"][0]["counts"]["pass_only"] == 1
    assert report["failures"][0]["reason"] == "pass-only EvalPlus completions above threshold"


def test_benchmark_audit_cmd_exits_on_blocked(tmp_path: Path, capsys):
    detail = tmp_path / "arc_detailed.jsonl"
    detail.write_text(json.dumps({"correct": False, "predicted": "?", "raw_response": ""}) + "\n", encoding="utf-8")
    args = parse_args(["benchmark-audit", str(detail), "--fail-empty-pct", "0"])

    try:
        benchmark_audit_cmd(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("benchmark-audit should exit non-zero when blocked")

    assert "Benchmark audit blocked" in capsys.readouterr().out


def test_benchmark_audit_command_parses():
    args = parse_args(["benchmark-audit", "results", "--json", "--fail-empty-pct", "1.5"])

    assert args.cmd == "benchmark-audit"
    assert args.paths == ["results"]
    assert args.json is True
    assert args.fail_empty_pct == 1.5


def test_pipeline_plan_builds_full_manifest(tmp_path: Path):
    source = tmp_path / "model-f16.gguf"
    output = tmp_path / "out"
    args = parse_args(
        [
            "pipeline-plan",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(output),
            "--profile",
            "wiki",
            "--model-name",
            "Gemma 4 12B IT",
            "--imatrix",
            str(output / "custom.imatrix"),
            "--low-space",
            "--benchmark-suite",
            "frontier",
        ]
    )

    plan = pipeline_plan(args)
    markdown = pipeline_plan_markdown(plan)
    phases = {row["name"]: row for row in plan["phases"]}

    assert list(phases) == ["imatrix", "ablate", "resume", "build-final-gguf", "benchmark", "finalize", "package"]
    assert "cerebellum imatrix" in phases["imatrix"]["command"]
    assert "cerebellum run" in phases["ablate"]["command"]
    assert "--profile wiki" in phases["ablate"]["command"]
    assert "--low-space" in phases["ablate"]["command"]
    assert "cerebellum resume" in phases["resume"]["command"]
    assert "--tensor-type-file" in phases["build-final-gguf"]["command"]
    assert "benchmark-plan --suite frontier" in phases["benchmark"]["command"]
    assert "cerebellum finalize" in phases["finalize"]["command"]
    assert "--repo-name" not in phases["finalize"]["command"]
    assert "cerebellum package" in phases["package"]["command"]
    assert str(output / "gemma-4-12b-it-cerebellum.gguf") == plan["final_gguf"]
    assert "# Cerebellum Pipeline Plan" in markdown


def test_pipeline_plan_command_writes_json(tmp_path: Path, capsys):
    manifest = tmp_path / "pipeline.json"
    args = parse_args(
        [
            "pipeline-plan",
            "--source-gguf",
            str(tmp_path / "m.gguf"),
            "--output-dir",
            str(tmp_path / "out"),
            "--write",
            str(manifest),
        ]
    )

    pipeline_plan_cmd(args)

    assert capsys.readouterr().out.strip() == str(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["pipeline"] == "cerebellum"
    assert data["phases"][0]["name"] == "imatrix"


def test_pipeline_plan_command_parses():
    args = parse_args(["pipeline-plan", "--source-gguf", "m.gguf", "--output-dir", "out", "--benchmark-suite", "release", "--json"])

    assert args.cmd == "pipeline-plan"
    assert args.source_gguf == "m.gguf"
    assert args.output_dir == "out"
    assert args.benchmark_suite == "release"
    assert args.json is True


def test_pipeline_run_command_parses():
    args = parse_args(["pipeline-run", "--manifest", "pipeline.json", "--from-phase", "ablate", "--until-phase", "benchmark", "--json"])

    assert args.cmd == "pipeline-run"
    assert args.manifest == "pipeline.json"
    assert args.from_phase == "ablate"
    assert args.until_phase == "benchmark"
    assert args.execute is False
    assert args.json is True


def test_pipeline_run_plan_slices_manifest(tmp_path: Path):
    manifest = tmp_path / "pipeline.json"
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "cerebellum",
                "run_dir": "run",
                "phases": [
                    {"name": "imatrix", "status": "planned", "command": "cerebellum imatrix", "outputs": ["imatrix.dat"]},
                    {"name": "ablate", "status": "planned", "command": "cerebellum run", "outputs": ["state.json"]},
                    {"name": "benchmark", "status": "planned", "command": "cerebellum benchmark-plan", "outputs": ["benchmark_results"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = pipeline_run_plan(manifest, from_phase="ablate", until_phase="benchmark")

    assert plan["schema"] == "cerebellum.pipeline_run.v1"
    assert plan["dry_run"] is True
    assert [row["name"] for row in plan["phases"]] == ["ablate", "benchmark"]
    assert plan["blocked"] is False


def test_pipeline_run_api_query_args_validate():
    args = pipeline_run_args_from_query(
        {
            "manifest": ["pipeline.json"],
            "from_phase": ["ablate"],
            "until_phase": ["benchmark"],
        }
    )

    assert args.manifest == "pipeline.json"
    assert args.from_phase == "ablate"
    assert args.until_phase == "benchmark"

    try:
        pipeline_run_args_from_query({})
    except ValueError as exc:
        assert "manifest query param required" in str(exc)
    else:
        raise AssertionError("pipeline-run API query should require manifest")


def test_pipeline_run_cmd_prints_dry_run_and_blocks_execute(tmp_path: Path, capsys):
    manifest = tmp_path / "pipeline.json"
    manifest.write_text(
        json.dumps({"pipeline": "cerebellum", "phases": [{"name": "imatrix", "status": "planned", "command": "cerebellum imatrix"}]}),
        encoding="utf-8",
    )
    args = parse_args(["pipeline-run", "--manifest", str(manifest)])

    pipeline_run_cmd(args)
    assert "# Cerebellum Pipeline Run" in capsys.readouterr().out

    execute_args = parse_args(["pipeline-run", "--manifest", str(manifest), "--execute"])
    try:
        pipeline_run_cmd(execute_args)
    except SystemExit as exc:
        assert "execution is not enabled yet" in str(exc)
    else:
        raise AssertionError("pipeline-run --execute should be guarded")


def test_pipeline_plan_query_args_use_task_profile_defaults():
    args = pipeline_plan_args_from_query(
        {
            "source_gguf": ["model.gguf"],
            "output_dir": ["out"],
            "task_profile": ["code"],
            "model_name": ["Model X"],
            "low_space": ["true"],
        }
    )
    plan = pipeline_plan(args)

    assert args.low_space is True
    assert plan["ppl_profile"] == "code"
    assert plan["benchmark_suite"] == "full"
    assert plan["task_profile"] == "code"
    assert plan["final_gguf"].endswith("model-x-code-cerebellum.gguf")


def test_pipeline_plan_query_args_require_source_and_output():
    try:
        pipeline_plan_args_from_query({"source_gguf": ["model.gguf"]})
    except ValueError as exc:
        assert "output_dir query param required" in str(exc)
    else:
        raise AssertionError("pipeline-plan API query should require output_dir")


def test_pipeline_plan_task_profile_sets_variant_defaults(tmp_path: Path):
    args = parse_args(
        [
            "pipeline-plan",
            "--source-gguf",
            str(tmp_path / "m.gguf"),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-name",
            "Model X",
            "--task-profile",
            "code",
        ]
    )

    plan = pipeline_plan(args)
    phases = {row["name"]: row for row in plan["phases"]}

    assert plan["task_profile"] == "code"
    assert plan["ppl_profile"] == "code"
    assert plan["benchmark_suite"] == "full"
    assert plan["task_profile_detail"]["metrics"] == ["humaneval", "evalplus", "livecodebench_v6"]
    assert plan["final_gguf"].endswith("model-x-code-cerebellum.gguf")
    assert "--profile code" in phases["ablate"]["command"]
    assert "benchmark-plan --suite full --model model-x-code" in phases["benchmark"]["command"]


def test_pipeline_plan_cpu_offload_profile_marks_low_space_and_strategy(tmp_path: Path):
    source = tmp_path / "glm-5.1-f16.gguf"
    source.write_bytes(b"gguf" * 1024)
    args = parse_args(
        [
            "pipeline-plan",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(tmp_path / "glm51-cpu"),
            "--model-name",
            "GLM 5.1",
            "--task-profile",
            "cpu-offload",
        ]
    )

    plan = pipeline_plan(args)
    phases = {row["name"]: row for row in plan["phases"]}
    markdown = pipeline_plan_markdown(plan)

    assert plan["task_profile"] == "cpu-offload"
    assert plan["ppl_profile"] == "all-around"
    assert plan["benchmark_suite"] == "full"
    assert plan["low_space"] is True
    assert plan["resource_strategy"]["target"] == "large RAM hosts with optional GPU layer offload"
    assert plan["cpu_offload_plan"]["model_hint"] == "glm"
    assert plan["cpu_offload_plan"]["source_size_gib"] > 0
    assert plan["cpu_offload_plan"]["streaming"]["full_model_ram_load_required"] is False
    assert plan["cpu_offload_plan"]["streaming"]["imatrix"] == plan["imatrix"]
    dry_run = plan["cpu_offload_plan"]["streaming_quant_dry_run"]
    assert dry_run["schema"] == "cerebellum.cpu_offload_streaming_quant_dry_run.v1"
    assert dry_run["dry_run"] is True
    assert "no full-model RAM load" in dry_run["model_load"]
    assert {row["phase"] for row in dry_run["disk_requirements"]} >= {"inspect-source", "ablate", "build-final-gguf"}
    assert {row["phase"] for row in dry_run["artifact_flow"]} >= {"stream-imatrix", "build-final-gguf", "dynamic-compare"}
    assert "inspect-gguf-types" in " ".join(dry_run["preflight_commands"])
    assert "cpu_tok_s" in plan["cpu_offload_plan"]["runtime_targets"]["record"]
    assert "scripts/benchmark_perf.py" in plan["cpu_offload_plan"]["throughput_probe_command"]
    assert "compare-gguf-types" in plan["cpu_offload_plan"]["dynamic_compare_command"]
    assert "--reference-map" in plan["cpu_offload_plan"]["dynamic_compare_command"]
    assert plan["final_gguf"].endswith("glm-5.1-cpu-offload-cerebellum.gguf")
    assert "--low-space" in phases["ablate"]["command"]
    assert "--low-space" in phases["resume"]["command"]
    assert "## Resource Strategy" in markdown
    assert "## CPU Offload Plan" in markdown
    assert "### Streaming Disk Dry Run" in markdown
    assert "### Streaming Artifact Flow" in markdown
    assert "full RAM load required" in markdown


def test_task_profiles_command_outputs_catalog(capsys):
    markdown = task_profiles_markdown()
    args = parse_args(["task-profiles", "--json"])

    task_profiles_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert "code" in data["profiles"]
    assert data["profiles"]["tools"]["ppl_profile"] == "agentic"
    assert data["profiles"]["cpu-offload"]["low_space_default"] is True
    assert "| code | code | full | humaneval, evalplus, livecodebench_v6 |" in markdown
    assert "| cpu-offload | all-around | full | ppl, speed, score_per_gib, cpu_tok_s, gpu_offload_layers |" in markdown


def test_task_profiles_command_parses():
    args = parse_args(["task-profiles", "--json"])

    assert args.cmd == "task-profiles"
    assert args.json is True


def test_ablation_analyze_json_classifies_and_writes_overrides(tmp_path: Path):
    ablation = tmp_path / "ablation_results.json"
    output = tmp_path / "types.txt"
    ablation.write_text(
        json.dumps(
            {
                "baseline_ppl": 100.0,
                "tests": {
                    "down": {"ppl": 94.0, "gguf_tensor": "blk.0.ffn_down.weight"},
                    "up": {"ppl": 100.5, "gguf_tensor": "blk.0.ffn_up.weight"},
                    "gate": {"ppl": 108.0, "gguf_tensor": "blk.0.ffn_gate.weight"},
                },
            }
        ),
        encoding="utf-8",
    )

    report = analyze_ablation_input(ablation)
    args = parse_args(["ablation-analyze", str(ablation), "--output", str(output), "--target-type", "q2_K"])
    ablation_analyze_cmd(args)

    classes = {row["tensor"]: row["classification"] for row in report["rows"]}
    assert classes["blk.0.ffn_down.weight"] == "demotable"
    assert classes["blk.0.ffn_up.weight"] == "tolerant"
    assert classes["blk.0.ffn_gate.weight"] == "critical"
    assert "^blk\\.0\\.ffn_down\\.weight$=q2_K" in output.read_text(encoding="utf-8")
    assert "^blk\\.0\\.ffn_up\\.weight$=q2_K" in output.read_text(encoding="utf-8")
    assert "ffn_gate" not in output.read_text(encoding="utf-8")


def test_ablation_analyze_log_dir_uses_tensor_group(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "ppl_layer_7.log").write_text("Final estimate: PPL = 98.0 +/- 1.0\n", encoding="utf-8")

    report = analyze_ablation_input(log_dir, baseline_ppl=100.0, tensor_group="attn_q")

    assert report["rows"][0]["tensor"] == "blk.7.attn_q.weight"
    assert report["rows"][0]["classification"] == "beneficial"


def test_ablation_analyze_command_parses():
    args = parse_args(["ablation-analyze", "logs", "--baseline-ppl", "10.0", "--tensor-group", "ffn_up"])

    assert args.cmd == "ablation-analyze"
    assert args.input == "logs"
    assert args.baseline_ppl == 10.0
    assert args.tensor_group == "ffn_up"


def test_eta_grid_values_includes_wall_clock_completion():
    state = {
        "locked": {"a": "q2_K"},
        "tested": [{"tensor": "blk.0.a"}],
        "totals": {"quant_seconds": 30.0, "ppl_seconds": 30.0},
    }

    eta = eta_grid_values(state, active_age=None, total=2)

    assert eta["total"] == "1m00s"
    assert eta["completion_at"] != "-"


def test_event_log_continues_existing_event_ids(tmp_path: Path):
    path = tmp_path / "cerebellum_events.jsonl"
    path.write_text('{"event_id": 258, "event": "cleanup_finish"}\n', encoding="utf-8")

    EventLog(path, "run").write("run_start")

    assert '"event_id": 259' in path.read_text(encoding="utf-8")


def test_sanitize_process_cmd_redacts_secret_env_values():
    cmd = "podman exec --env=GITHUB_TOKEN=abc123 HF_TOKEN=def456 NORMAL=value"

    sanitized = sanitize_process_cmd(cmd)

    assert "abc123" not in sanitized
    assert "def456" not in sanitized
    assert "--env=GITHUB_TOKEN=<redacted>" in sanitized
    assert "HF_TOKEN=<redacted>" in sanitized
    assert "NORMAL=value" in sanitized


def test_discover_projects_falls_back_to_run_manifest(tmp_path: Path):
    run_dir = (
        tmp_path
        / "families"
        / "gemma-4"
        / "gemma-4-12b-it"
        / "sources"
        / "google-f16"
        / "runs"
        / "gemma4-live"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        '{"run_id":"gemma4-live","model_family":"gemma-4","model_name":"gemma-4-12b-it","source_name":"google-f16"}',
        encoding="utf-8",
    )
    (run_dir / "state.json").write_text(
        '{"run_status":"running","current_ppl":2171.2683,"locked":{"blk.0.ffn_norm.weight":"q2_K"}}',
        encoding="utf-8",
    )

    projects = discover_projects(tmp_path)

    assert len(projects) == 1
    project = projects[0]
    assert project["family"] == "gemma-4"
    assert project["model_name"] == "gemma-4-12b-it"
    assert project["source_name"] == "google-f16"
    assert project["runs"][0]["run_id"] == "gemma4-live"
    assert project["runs"][0]["locked"] == 1


def test_tensor_type_lines_are_exact_regex_patterns():
    line = tensor_type_line("blk.0.ffn_up.weight", "q5_K")

    assert line == r"^blk\.0\.ffn_up\.weight$=q5_K"


def test_non_quantizable_norm_tensors_are_excluded():
    assert is_quantizable_tensor("blk.0.ffn_up.weight") is True
    assert is_quantizable_tensor("blk.0.ffn_norm.weight") is False
    assert is_quantizable_tensor("blk.0.attn_q_norm.weight") is False


def test_tensor_type_map_uses_exact_patterns_without_source(tmp_path: Path):
    path = tmp_path / "types.txt"

    write_tensor_types_map(None, {"blk.0.ffn_up.weight": "q5_K"}, "q4_K", path)

    assert path.read_text(encoding="utf-8") == r"^blk\.0\.ffn_up\.weight$=q5_K" + "\n"


def test_tensor_type_map_keeps_start_type_as_explicit_baseline(tmp_path: Path):
    path = tmp_path / "types.txt"

    write_tensor_types_map(None, {"blk.0.ffn_down.weight": "q4_K"}, "q4_K", path)

    assert path.read_text(encoding="utf-8") == r"^blk\.0\.ffn_down\.weight$=q4_K" + "\n"


def test_tensor_type_map_skips_noop_source_tensors(tmp_path: Path, monkeypatch):
    path = tmp_path / "types.txt"
    source = tmp_path / "source.gguf"
    source.write_bytes(b"fake")

    class Tensor:
        def __init__(self, name: str):
            self.name = name

    class Reader:
        def __init__(self, _path: str):
            self.tensors = [
                Tensor("token_embd.weight"),
                Tensor("blk.0.attn_norm.weight"),
                Tensor("blk.0.attn_q.weight"),
            ]

    monkeypatch.setitem(sys.modules, "gguf", types.SimpleNamespace(GGUFReader=Reader))

    write_tensor_types_map(source, {"blk.0.attn_q.weight": "q5_K"}, "q4_K", path)

    assert path.read_text(encoding="utf-8") == r"^blk\.0\.attn_q\.weight$=q5_K" + "\n"


def test_inspect_gguf_types_summarizes_layers_and_components(tmp_path: Path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")

    class Tensor:
        def __init__(self, name: str, tensor_type: int):
            self.name = name
            self.tensor_type = tensor_type

    class Reader:
        def __init__(self, _path: str):
            self.tensors = [
                Tensor("token_embd.weight", 1),
                Tensor("blk.0.ffn_down.weight", 12),
                Tensor("blk.0.attn_q.weight", 11),
                Tensor("blk.1.ffn_down.weight", 13),
                Tensor("blk.1.attn_norm.weight", 0),
            ]

    class Quant:
        def __init__(self, value: int):
            self.name = {0: "F32", 1: "F16", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K"}[value]

    quant = Quant
    monkeypatch.setitem(sys.modules, "gguf", types.SimpleNamespace(GGUFReader=Reader, GGMLQuantizationType=quant))

    summary = inspect_gguf_types(gguf)

    assert summary["tensor_count"] == 5
    assert summary["quantizable_tensor_count"] == 3
    assert summary["type_counts"] == {"F16": 1, "F32": 1, "Q3_K": 1, "Q4_K": 1, "Q5_K": 1}
    assert summary["component_counts"]["ffn_down"] == {"Q4_K": 1, "Q5_K": 1}
    assert summary["layer_counts"]["blk.0"] == {"Q3_K": 1, "Q4_K": 1}
    assert summary["tensor_types"]["blk.0.attn_q.weight"] == "Q3_K"


def test_read_tensor_type_map_accepts_exact_regex_lines(tmp_path: Path):
    tensor_map = tmp_path / "types.txt"
    tensor_map.write_text(
        "\n".join(
            [
                "^blk\\.0\\.ffn_down\\.weight$=q5_K",
                "^blk\\.0\\.attn_q\\.weight$=Q3_K",
                "blk\\.0\\.not_exact\\.weight=q2_K",
                "# comment",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_tensor_type_map(tensor_map) == {
        "blk.0.ffn_down.weight": "Q5_K",
        "blk.0.attn_q.weight": "Q3_K",
    }


def test_compare_gguf_types_reports_type_component_layer_deltas(tmp_path: Path, monkeypatch):
    baseline = tmp_path / "baseline.gguf"
    candidate = tmp_path / "candidate.gguf"
    baseline.write_bytes(b"fake")
    candidate.write_bytes(b"fake")

    class Tensor:
        def __init__(self, name: str, tensor_type: int):
            self.name = name
            self.tensor_type = tensor_type

    class Reader:
        def __init__(self, path: str):
            if Path(path).name == "baseline.gguf":
                self.tensors = [
                    Tensor("blk.0.ffn_down.weight", 12),
                    Tensor("blk.0.attn_q.weight", 12),
                    Tensor("blk.1.ffn_down.weight", 12),
                ]
            else:
                self.tensors = [
                    Tensor("blk.0.ffn_down.weight", 13),
                    Tensor("blk.0.attn_q.weight", 11),
                    Tensor("blk.1.ffn_down.weight", 12),
                ]

    class Quant:
        def __init__(self, value: int):
            self.name = {11: "Q3_K", 12: "Q4_K", 13: "Q5_K"}[value]

    monkeypatch.setitem(sys.modules, "gguf", types.SimpleNamespace(GGUFReader=Reader, GGMLQuantizationType=Quant))

    reference_map = tmp_path / "types.txt"
    reference_map.write_text(
        "\n".join(
            [
                "^blk\\.0\\.ffn_down\\.weight$=q5_K",
                "^blk\\.0\\.attn_q\\.weight$=q5_K",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = compare_gguf_types(baseline, candidate, baseline_label="q4", candidate_label="dynamic", reference_map=reference_map)
    markdown = compare_gguf_types_markdown(report)

    assert report["type_counts"]["Q4_K"]["delta"] == -2
    assert report["type_counts"]["Q3_K"]["delta"] == 1
    assert report["component_counts"]["ffn_down"]["Q5_K"]["delta"] == 1
    assert report["layer_counts"]["blk.0"]["Q4_K"]["delta"] == -2
    assert report["dynamic_profile"]["changed_quantizable_tensors"] == 2
    assert report["dynamic_profile"]["promoted"] == 1
    assert report["dynamic_profile"]["demoted"] == 1
    assert report["dynamic_profile"]["baseline_avg_bits"] == 4.0
    assert report["dynamic_profile"]["candidate_avg_bits"] == 4.0
    assert report["dynamic_profile"]["component_bias"]["attn_q"]["demoted"] == 1
    assert report["dynamic_profile"]["component_bias"]["ffn_down"]["promoted"] == 1
    assert report["reference_map"] == {"path": str(reference_map), "tensor_count": 2}
    assert report["reference_mismatches"] == [
        {
            "tensor": "blk.0.attn_q.weight",
            "layer": 0,
            "component": "attn_q",
            "baseline": "Q4_K",
            "candidate": "Q3_K",
            "reference": "Q5_K",
            "matches_reference": False,
            "status": "candidate_diverges_reference",
            "quantizable": True,
        }
    ]
    assert report["tensor_type_changes"] == [
        {
            "tensor": "blk.0.attn_q.weight",
            "layer": 0,
            "component": "attn_q",
            "baseline": "Q4_K",
            "candidate": "Q3_K",
            "reference": "Q5_K",
            "matches_reference": False,
            "status": "changed",
            "quantizable": True,
        },
        {
            "tensor": "blk.0.ffn_down.weight",
            "layer": 0,
            "component": "ffn_down",
            "baseline": "Q4_K",
            "candidate": "Q5_K",
            "reference": "Q5_K",
            "matches_reference": True,
            "status": "changed",
            "quantizable": True,
        },
    ]
    assert "candidate: `dynamic`" in markdown
    assert "| Q4_K | 3 | 1 | -2 |" in markdown
    assert "## Dynamic Quant Profile" in markdown
    assert "| promoted | 1 |" in markdown
    assert "| demoted | 1 |" in markdown
    assert "## Reference Map Mismatches" in markdown
    assert "| blk.0.attn_q.weight | attn_q | Q4_K | Q3_K | Q5_K | candidate_diverges_reference |" in markdown
    assert "| blk.0.attn_q.weight | Q4_K | Q3_K | changed |" in markdown


def test_compare_locks_filters_non_quantizable_entries():
    current = {
        "locked": {
            "blk.0.ffn_down.weight": "q4_K",
            "blk.0.ffn_norm.weight": "q2_K",
            "blk.0.attn_q.weight": "f16",
        }
    }
    against = {
        "locked": {
            "blk.0.ffn_down.weight": "q4_K",
            "blk.0.ffn_norm.weight": "q3_K",
            "blk.0.attn_q.weight": "q4_K",
            "blk.0.ffn_up.weight": "q5_K",
        }
    }

    summary = compare_locks(current, against)

    assert summary["current_locked"] == 2
    assert summary["against_locked"] == 3
    assert summary["same"] == 1
    assert summary["different"] == 1
    assert summary["missing_current"] == 1
    assert summary["missing_against"] == 0
    assert {row["tensor"] for row in summary["rows"]} == {
        "blk.0.ffn_down.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_up.weight",
    }


def test_watch_model_uses_latest_run_epoch_after_rollback(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        '{"run_status":"running","locked":{"blk.0.ffn_gate.weight":"q3_K"},'
        '"tested":[{"tensor":"blk.0.ffn_gate.weight"}],'
        '"totals":{"quant_seconds":30.0,"ppl_seconds":30.0}}',
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text('{"run_id":"run"}', encoding="utf-8")
    (run_dir / "cerebellum_events.jsonl").write_text(
        "\n".join(
            [
                '{"event":"tensor_start","tensor":"blk.0.ffn_norm.weight","total":616}',
                '{"event":"rollback_finish","removed":5}',
                '{"event":"run_start","tensors":328}',
                '{"event":"baseline_ppl_start"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        "\n".join(
            [
                '{"tensor":"blk.0.ffn_norm.weight","level":"q2_K"}',
                '{"tensor":"blk.0.ffn_gate.weight","level":"q3_K"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    model = build_watch_model(run_dir)

    assert model["progress"] == "1/328 0.3%"
    assert model["total"] == 328
    assert eta_grid_values(model["state"], model["active_age"], model["total"])["total"] == "5h27m"
    assert model["active"]["event"] == "baseline_ppl_start"
    assert [row["tensor"] for row in model["candidates"]] == ["blk.0.ffn_gate.weight"]


def test_watch_model_applies_stall_threshold_flags(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nowish = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(timespec="milliseconds")
    (run_dir / "state.json").write_text('{"run_status":"running"}', encoding="utf-8")
    (run_dir / "manifest.json").write_text('{"run_id":"run"}', encoding="utf-8")
    (run_dir / "cerebellum_events.jsonl").write_text(
        f'{{"event":"run_start","tensors":328,"timestamp_utc":"{nowish}"}}\n'
        f'{{"event":"baseline_ppl_start","pid":"123","timestamp_utc":"{nowish}"}}\n',
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "osmosis.hillstep.process_rows_for_run",
        lambda _run_dir: [{"kind": "runner", "pid": "123", "etime": "00:02", "cmd": "cerebellum resume run"}],
    )

    assert build_watch_model(run_dir)["health"]["health"] == "waiting"
    assert build_watch_model(run_dir, stall_warn_seconds=0.5, stall_fail_seconds=1.0)["health"]["health"] == "failure suspected"


def test_watch_model_shows_rollback_finish_before_resume(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        '{"run_status":"stopped","locked":{"blk.0.ffn_gate.weight":"q3_K"},'
        '"tested":[{"tensor":"blk.0.ffn_gate.weight"}]}',
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text('{"run_id":"run"}', encoding="utf-8")
    (run_dir / "cerebellum_events.jsonl").write_text(
        "\n".join(
            [
                '{"event":"run_start","tensors":328}',
                '{"event":"tensor_start","tensor":"blk.0.ffn_norm.weight","total":328}',
                '{"event":"run_stopped"}',
                '{"event":"rollback_finish","removed":1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text("", encoding="utf-8")

    model = build_watch_model(run_dir)

    assert model["active"]["event"] == "rollback_finish"


def test_recovery_and_watch_use_scratch_roots(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    scratch = tmp_path / "scratch"
    run_id = "run-a"
    tmp_root = scratch / run_id / "tmp" / "partial"
    artifact_root = scratch / run_id / "artifacts"
    tmp_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    (tmp_root / "part.bin").write_bytes(b"partial")
    (artifact_root / "current_baseline.gguf").write_bytes(b"artifact")
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps({"run_status": "stopped", "run_id": run_id, "locked": {}, "tested": []}),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "scratch_root": str(scratch)}),
        encoding="utf-8",
    )
    (run_dir / "cerebellum_events.jsonl").write_text('{"event":"run_start","tensors":1}\n', encoding="utf-8")
    (run_dir / "cerebellum_candidates.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr("osmosis.hillstep.process_rows_for_run", lambda _run_dir: [])

    recovery = build_recovery_plan(run_dir)
    model = build_watch_model(run_dir)

    assert recovery["partials"] == [str(tmp_root)]
    assert recovery["tmp_size_bytes"] == len(b"partial")
    assert recovery["artifact_size_bytes"] == len(b"artifact")
    assert model["tmp_size"] == len(b"partial")
    assert model["artifacts_size"] == len(b"artifact")


def test_cleanup_partials_uses_scratch_root(tmp_path: Path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    scratch = tmp_path / "scratch"
    run_id = "run-a"
    partial = scratch / run_id / "tmp" / "00001-blk-0-attn-q"
    partial.mkdir(parents=True)
    (partial / "candidate.gguf").write_bytes(b"partial")
    run_dir.mkdir()
    (run_dir / "state.json").write_text(json.dumps({"run_status": "stopped", "run_id": run_id, "locked": {}}), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "scratch_root": str(scratch)}), encoding="utf-8")
    monkeypatch.setattr("osmosis.hillstep.process_rows_for_run", lambda _run_dir: [])

    args = parse_args(["cleanup", str(run_dir), "--partials"])
    cleanup_cmd(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run"
    assert payload["paths"] == [str(partial)]


def test_rollback_refuses_active_runner_without_force(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_status": "running",
                "tested": [{"tensor": "blk.0.ffn_up.weight", "winner": "q5_K", "ppl": 1.0}],
                "locked": {"blk.0.ffn_up.weight": "q5_K"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("osmosis.hillstep.run_is_live", lambda _run_dir: True)

    try:
        rollback_cmd(types.SimpleNamespace(run_dir=str(run_dir), to_locked=0, before_layer=None, last_completed_layer=False, yes=True, force=False))
    except SystemExit as exc:
        assert "refusing to rollback while runner is active" in str(exc)
    else:
        raise AssertionError("rollback should refuse active runner")


def test_rollback_resets_timing_to_kept_epoch(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_status": "stopped",
                "source_gguf": None,
                "start_type": "q4_K",
                "current_ppl": 10.0,
                "tested": [{"tensor": "blk.0.ffn_up.weight", "winner": "q5_K", "ppl": 9.0, "baseline_ppl": 10.0}],
                "locked": {"blk.0.ffn_up.weight": "q5_K"},
                "totals": {"quant_seconds": 84.0, "ppl_seconds": 120.0, "candidates": 5, "failures": 0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text('{"start_type":"q4_K"}', encoding="utf-8")
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        '{"tensor":"blk.0.ffn_up.weight","quant_seconds":84.0,"ppl_seconds":120.0,"status":"done"}\n',
        encoding="utf-8",
    )
    (run_dir / "cerebellum_events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr("osmosis.hillstep.run_is_live", lambda _run_dir: False)

    rollback_cmd(types.SimpleNamespace(run_dir=str(run_dir), to_locked=0, before_layer=None, last_completed_layer=False, yes=True, force=False))

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    assert state["tested"] == []
    assert state["locked"] == {}
    assert state["totals"] == {"quant_seconds": 0.0, "ppl_seconds": 0.0, "candidates": 0, "failures": 0}
    assert timing == state["totals"]


def test_locked_layer_lines_groups_locked_quants_by_layer():
    state = {
        "tested": [
            {"tensor": "blk.0.ffn_down.weight", "winner": "q3_K"},
            {"tensor": "blk.0.attn_q.weight", "winner": "q4_K"},
            {"tensor": "blk.1.attn_k.weight", "winner": "q5_K"},
        ]
    }

    lines = locked_layer_lines(state)

    assert lines == [
        "blk.0    ffn_down=q3_K  attn_q=q4_K",
        "blk.1    attn_k=q5_K",
    ]


def test_public_watch_redacts_factory_details(tmp_path: Path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_status": "running",
                "model_family": "gemma-4",
                "model_name": "gemma-4-12b-it",
                "current_ppl": 2142.6025,
                "locked": {"blk.1.attn_v.weight": "q4_K"},
                "tested": [{"tensor": "blk.1.attn_v.weight"}],
                "totals": {"quant_seconds": 30.0, "ppl_seconds": 30.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        '{"run_id":"gemma4-12b-cerebellum-q4km-wiki-visible-20260604","ppl_profile":"wiki"}',
        encoding="utf-8",
    )
    (run_dir / "cerebellum_events.jsonl").write_text(
        "\n".join(
            [
                '{"event":"run_start","tensors":328}',
                '{"event":"tensor_start","tensor":"blk.1.attn_k.weight","total":328}',
                '{"event":"quant_start","level":"q3_K","tensor":"blk.1.attn_k.weight","pid":"123456"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        '{"level":"q3_K","ppl":2147.7021,"delta":5.0996,"size_bytes":8577434592,"tensor":"blk.1.attn_k.weight"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("osmosis.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("osmosis.hillstep.gpu_rows", lambda: [])
    monkeypatch.setattr("osmosis.hillstep.os.system", lambda _cmd: 0)

    grid_watch_cmd(
        types.SimpleNamespace(
            run_dir=str(run_dir),
            stall_warn_seconds=300.0,
            stall_fail_seconds=900.0,
            measurements_limit=8,
            events_limit=12,
            once=True,
            public=True,
            plain=True,
            no_color=True,
        )
    )

    output = capsys.readouterr().out
    assert "public-safe telemetry" in output
    assert "redacted" in output
    assert "gemma-4-12b-it" not in output
    assert "wiki" not in output
    assert "blk.1.attn_k.weight" not in output
    assert "q3_K" not in output
    assert "2147.7021" not in output
    assert "+5.0996" not in output
    assert "gemma4-12b-cerebellum-q4km-wiki-visible-20260604" not in output
    assert "pid" not in output.lower()


def test_private_watch_shows_locked_layer_map(tmp_path: Path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_status": "running",
                "model_family": "gemma-4",
                "model_name": "gemma-4-12b-it",
                "locked": {"blk.0.ffn_down.weight": "q3_K"},
                "tested": [{"tensor": "blk.0.ffn_down.weight", "winner": "q3_K", "ppl": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text('{"run_id":"run","ppl_profile":"wiki"}', encoding="utf-8")
    (run_dir / "cerebellum_events.jsonl").write_text(
        '{"event":"run_start","tensors":328}\n',
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr("osmosis.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("osmosis.hillstep.gpu_rows", lambda: [])
    monkeypatch.setattr("osmosis.hillstep.os.system", lambda _cmd: 0)

    grid_watch_cmd(
        types.SimpleNamespace(
            run_dir=str(run_dir),
            stall_warn_seconds=300.0,
            stall_fail_seconds=900.0,
            measurements_limit=8,
            events_limit=12,
            once=True,
            public=False,
            plain=True,
            no_color=True,
        )
    )

    output = capsys.readouterr().out
    assert "LOCKED LAYER MAP" in output
    assert "blk.0" in output
    assert "ffn_down=q3_K" in output


def test_package_files_default_to_public_safe_sidecars(tmp_path: Path):
    run_dir = tmp_path / "run"
    finalize = run_dir / "finalize"
    finalize.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"run_id":"run","model_family":"gemma","model_name":"tiny"}', encoding="utf-8")
    (run_dir / "state.json").write_text('{"run_status":"complete","locked":{},"tested":[]}', encoding="utf-8")
    raw_names = [
        "cerebellum_events.jsonl",
        "cerebellum_candidates.jsonl",
        "cerebellum_summary.json",
        "cerebellum_summary.md",
        "cerebellum_decisions.csv",
        "cerebellum_best_tensor_types.txt",
    ]
    for name in raw_names:
        (run_dir / name).write_text("{}", encoding="utf-8")
    for name in ["cerebellum_gguf_metadata.json", "cerebellum_gguf_metadata.env", "MODEL_CARD_CEREBELLUM.md"]:
        (finalize / name).write_text("safe", encoding="utf-8")

    public = {path.name for path in package_files(run_dir)}
    private = {path.name for path in package_files(run_dir, private=True)}

    assert public == {"cerebellum_public_metadata.json", "MODEL_CARD_CEREBELLUM.md"}
    assert "cerebellum_gguf_metadata.env" not in public
    assert {"manifest.json", "state.json", *raw_names}.issubset(private)


def test_package_manifest_marks_public_mode(tmp_path: Path):
    run_dir = tmp_path / "run"
    finalize = run_dir / "finalize"
    finalize.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"run_id":"run","model_family":"gemma","model_name":"tiny"}', encoding="utf-8")
    (run_dir / "state.json").write_text('{"run_status":"complete","locked":{},"tested":[]}', encoding="utf-8")
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        '{"tensor":"blk.0.attn_q.weight","delta":1.0}\n',
        encoding="utf-8",
    )
    (finalize / "MODEL_CARD_CEREBELLUM.md").write_text("safe", encoding="utf-8")

    payload = package_manifest(run_dir)

    assert payload["mode"] == "public"
    assert "run_id" not in payload
    assert "ppl_profile" not in payload
    assert "run_dir" not in payload
    assert payload["release_label"] == "tiny"
    assert [item["name"] for item in payload["files"]] == ["cerebellum_public_metadata.json", "MODEL_CARD_CEREBELLUM.md"]
    assert all("path" not in item for item in payload["files"])
    assert all("run" not in item["hf_path"] for item in payload["files"])
    assert payload["files"][0]["hf_path"] == "cerebellum_releases/tiny/cerebellum_public_metadata.json"


def test_cerebellum_metadata_block_defaults_to_public_safe_keys(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "model_family": "gemma",
                "model_name": "tiny",
                "source_name": "hf",
                "ppl_profile": "wiki",
                "corpus": "/tmp/wiki.test.raw",
                "files": {"events": "/tmp/cerebellum_events.jsonl", "candidates": "/tmp/cerebellum_candidates.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "state.json").write_text('{"run_status":"complete","current_ppl":1234.5,"locked":{},"tested":[]}', encoding="utf-8")

    public = cerebellum_metadata_block(run_dir)
    private = cerebellum_metadata_block(run_dir, private=True)

    assert "cerebellum.run_id" not in public
    assert "cerebellum.current_ppl" not in public
    assert "cerebellum.events_file" not in public
    assert private["cerebellum.run_id"] == "run"
    assert private["cerebellum.current_ppl"] == "1234.5"
    assert private["cerebellum.events_file"] == "cerebellum_events.jsonl"


def test_public_audit_blocks_private_paths_and_local_content(tmp_path: Path):
    safe = tmp_path / "README.md"
    risky_dir = tmp_path / "scripts"
    risky_dir.mkdir()
    risky = risky_dir / "run.sh"
    safe.write_text("public model card\n", encoding="utf-8")
    risky.write_text("HF_TOKEN=secret\nmodel path /var/home/deucebucket/ai-drive/model.gguf\n", encoding="utf-8")

    report = public_audit([str(tmp_path)])
    reasons = {item["reason"] for item in report["findings"]}
    markdown = public_audit_markdown(report)

    assert report["blocked"] is True
    assert "private script path" in reasons
    assert "credential environment assignment" in reasons
    assert "absolute local user path" in reasons
    assert "Public audit blocked" in markdown


def test_public_audit_passes_clean_explicit_file(tmp_path: Path):
    safe = tmp_path / "MODEL_CARD.md"
    safe.write_text("safe benchmark summary\n", encoding="utf-8")

    report = public_audit([str(safe)])

    assert report["blocked"] is False
    assert public_audit_markdown(report).startswith("Public audit passed")


def test_public_audit_command_parses():
    args = parse_args(["public-audit", "README.md", "--json", "--max-bytes", "100"])

    assert args.cmd == "public-audit"
    assert args.paths == ["README.md"]
    assert args.json is True
    assert args.max_bytes == 100


def test_public_audit_cmd_exits_on_findings(tmp_path: Path, capsys):
    risky = tmp_path / "state.json"
    risky.write_text("{}", encoding="utf-8")
    args = parse_args(["public-audit", str(risky)])

    try:
        public_audit_cmd(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("public audit should exit non-zero when blocked")

    assert "Public audit blocked" in capsys.readouterr().out


def test_public_export_command_parses():
    args = parse_args(["public-export", "out", "README.md", "--clean", "--dry-run", "--json", "--max-bytes", "100"])

    assert args.cmd == "public-export"
    assert args.output_dir == "out"
    assert args.paths == ["README.md"]
    assert args.clean is True
    assert args.dry_run is True
    assert args.json is True
    assert args.max_bytes == 100


def test_public_export_plan_selects_safe_files_and_skips_risks(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    benches = tmp_path / "benchmark_results"
    scripts = tmp_path / "scripts"
    docs.mkdir()
    benches.mkdir()
    scripts.mkdir()
    (tmp_path / "README.md").write_text("Cerebellum public card\n", encoding="utf-8")
    (docs / "benchmark_protocol.md").write_text("release benchmark protocol\n", encoding="utf-8")
    (benches / "summary.json").write_text('{"benchmark":"arc","accuracy":0.8}\n', encoding="utf-8")
    (docs / "devlog.md").write_text("devlog raw ablation\n", encoding="utf-8")
    (scripts / "factory.sh").write_text("HF_TOKEN=secret\n", encoding="utf-8")

    plan = public_export_plan(["."], max_bytes=1000)
    markdown = public_export_markdown(plan)

    exported = {item["path"] for item in plan["files"]}
    skipped = {item["path"] for item in plan["skipped"]}
    assert exported == {"README.md", "docs/benchmark_protocol.md", "benchmark_results/summary.json"}
    assert "docs/devlog.md" in skipped
    assert "scripts/factory.sh" not in exported
    assert plan["blocked"] is True
    assert "Public export blocked" in markdown


def test_public_export_plan_api_query_args_validate():
    args = public_export_plan_args_from_query(
        {
            "path": ["README.md", "docs"],
            "max_bytes": ["100"],
        }
    )

    assert args.paths == ["README.md", "docs"]
    assert args.max_bytes == 100

    try:
        public_export_plan_args_from_query({"max_bytes": ["many"]})
    except ValueError as exc:
        assert "max_bytes must be an integer" in str(exc)
    else:
        raise AssertionError("invalid public export max_bytes should fail")

    try:
        public_export_plan_args_from_query({"max_bytes": ["0"]})
    except ValueError as exc:
        assert "max_bytes must be at least 1" in str(exc)
    else:
        raise AssertionError("zero public export max_bytes should fail")


def test_public_export_cmd_copies_manifest_and_files(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("Cerebellum public card\n", encoding="utf-8")
    out = tmp_path / "public"
    args = parse_args(["public-export", str(out), "README.md", "--clean"])

    public_export_cmd(args)

    manifest = json.loads((out / "cerebellum_public_export_manifest.json").read_text(encoding="utf-8"))
    assert (out / "README.md").read_text(encoding="utf-8") == "Cerebellum public card\n"
    assert manifest["schema"] == "cerebellum.public_export.v1"
    assert manifest["files"][0]["path"] == "README.md"
    assert "manifest:" in capsys.readouterr().out


def test_artifact_inventory_categorizes_legacy_files(tmp_path: Path):
    legacy = tmp_path / "osmosis-gemma4-e2b"
    benches = legacy / "benchmark_results"
    benches.mkdir(parents=True)
    (legacy / "cerebellum_v2_overrides.txt").write_text("blk.11.ffn_gate.weight=Q2_K\n", encoding="utf-8")
    (legacy / "server.log").write_text("local server log\n", encoding="utf-8")
    (legacy / "model.gguf").write_bytes(b"gguf")
    (benches / "e2b_arc_results.json").write_text('{"accuracy":0.7}\n', encoding="utf-8")
    dev = tmp_path / "cerebellum-dev"
    dev.mkdir()
    (dev / "DEVLOG_2026-06-03_gemma4-12b.md").write_text("raw ablation note\n", encoding="utf-8")
    (tmp_path / "cerebellum_logo.png").write_bytes(b"png")

    report = artifact_inventory(tmp_path, top=10)
    markdown = artifact_inventory_markdown(report)
    buckets = {row["path"]: row for row in report["buckets"]}

    assert report["schema"] == "cerebellum.artifact_inventory.v1"
    assert report["totals"]["files"] == 6
    assert report["type_counts"]["gguf"] == 1
    assert report["type_counts"]["benchmark"] == 1
    assert report["type_counts"]["tensor_map"] == 1
    assert buckets["osmosis-gemma4-e2b"]["public_risk_files"] >= 2
    assert any(item["path"].endswith("server.log") for item in report["cleanup_candidates"])
    assert "Cerebellum Artifact Inventory" in markdown
    assert "osmosis-gemma4-e2b" in markdown


def test_artifact_inventory_command_writes_outputs(tmp_path: Path, capsys):
    (tmp_path / "README.md").write_text("public card\n", encoding="utf-8")
    output = tmp_path / "inventory.json"
    markdown = tmp_path / "inventory.md"
    args = parse_args(["artifact-inventory", str(tmp_path), "--output", str(output), "--markdown", str(markdown), "--top", "3"])

    artifact_inventory_cmd(args)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "cerebellum.artifact_inventory.v1"
    assert markdown.read_text(encoding="utf-8").startswith("# Cerebellum Artifact Inventory")
    out = capsys.readouterr().out
    assert "artifact inventory JSON" in out
    assert "artifact inventory Markdown" in out


def test_artifact_inventory_command_parses():
    args = parse_args(["artifact-inventory", ".", "--output", "inventory.json", "--markdown", "inventory.md", "--top", "5", "--json"])

    assert args.cmd == "artifact-inventory"
    assert args.root == "."
    assert args.output == "inventory.json"
    assert args.markdown == "inventory.md"
    assert args.top == 5
    assert args.json is True


def test_artifact_inventory_api_query_args_validate(tmp_path: Path):
    args = artifact_inventory_args_from_query({"root": [str(tmp_path)], "top": ["5"]})

    assert args.root == str(tmp_path.resolve())
    assert args.top == 5

    try:
        artifact_inventory_args_from_query({})
    except ValueError as exc:
        assert "root query param required" in str(exc)
    else:
        raise AssertionError("missing artifact inventory root should fail")

    try:
        artifact_inventory_args_from_query({"root": [str(tmp_path)], "top": ["many"]})
    except ValueError as exc:
        assert "top must be an integer" in str(exc)
    else:
        raise AssertionError("invalid artifact inventory top should fail")

    try:
        artifact_inventory_args_from_query({"root": ["/"]})
    except ValueError as exc:
        assert "allow_broad=true" in str(exc)
    else:
        raise AssertionError("filesystem root inventory should require explicit override")


class _FakeHTTPResponse:
    def __init__(self, text: str):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.text.encode("utf-8")


def test_hf_stats_recent_labels_public_downloads_as_recent(monkeypatch):
    payload = json.dumps(
        [
            {"modelId": "deucebucket/a", "downloads": 10, "likes": 2, "private": False},
            {"modelId": "deucebucket/b", "downloads": 5, "likes": 1, "private": False},
        ]
    )

    def fake_urlopen(request, timeout=30):
        assert "api/models" in request.full_url
        assert "author=deucebucket" in request.full_url
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr(hillstep, "urlopen", fake_urlopen)
    args = parse_args(["hf-stats", "--author", "deucebucket", "--json"])

    report = hf_model_stats(args)
    markdown = hf_model_stats_markdown(report)

    assert report["period"] == "recent"
    assert report["total_downloads_recent"] == 15
    assert "total_downloads" not in report
    assert report["models"][0]["downloads_recent"] == 10
    assert "not lifetime totals" in report["metric_note"]
    assert "recent rolling downloads" in markdown


def test_hf_stats_all_time_requires_publisher_org():
    args = parse_args(["hf-stats", "--period", "all-time"])

    try:
        hf_model_stats(args)
    except SystemExit as exc:
        assert "--publisher-org" in str(exc)
    else:
        raise AssertionError("all-time HF stats should require publisher analytics org")


def test_hf_stats_all_time_parses_publisher_analytics_csv(monkeypatch):
    csv_text = "\n".join(
        [
            "repoType,repoName,total,timestamp,downloads",
            "model,deucebucket/a,100,2026-06-01 T00:00:00.000 Z,3",
            "model,deucebucket/a,110,2026-06-02 T00:00:00.000 Z,10",
            "model,deucebucket/b,20,2026-06-02 T00:00:00.000 Z,2",
            "dataset,deucebucket/data,999,2026-06-02 T00:00:00.000 Z,9",
        ]
    )

    def fake_urlopen(request, timeout=30):
        assert "publisher-analytics/download-breakdown" in request.full_url
        return _FakeHTTPResponse(csv_text)

    monkeypatch.setattr(hillstep, "urlopen", fake_urlopen)
    args = parse_args(["hf-stats", "--period", "all-time", "--publisher-org", "deucebucket"])

    report = hf_model_stats(args)

    assert report["period"] == "all-time"
    assert report["total_downloads_all_time"] == 130
    assert [row["modelId"] for row in report["models"]] == ["deucebucket/a", "deucebucket/b"]


def test_hf_stats_command_parses():
    args = parse_args(["hf-stats", "--author", "deucebucket", "--period", "recent", "--limit", "10", "--json"])

    assert args.cmd == "hf-stats"
    assert args.author == "deucebucket"
    assert args.period == "recent"
    assert args.limit == 10
    assert args.json is True


def test_hf_stats_api_query_args_validate_period_and_limit():
    args = hf_stats_args_from_query({"author": ["deucebucket"], "period": ["recent"], "limit": ["25"]})

    assert args.author == "deucebucket"
    assert args.period == "recent"
    assert args.limit == 25

    try:
        hf_stats_args_from_query({"period": ["forever"]})
    except ValueError as exc:
        assert "period must be recent or all-time" in str(exc)
    else:
        raise AssertionError("invalid hf-stats period should fail")

    try:
        hf_stats_args_from_query({"limit": ["many"]})
    except ValueError as exc:
        assert "limit must be an integer" in str(exc)
    else:
        raise AssertionError("invalid hf-stats limit should fail")


def test_public_report_summary_strips_factory_fields():
    summary = public_report_summary(
        {
            "run_id": "run",
            "model_family": "gemma",
            "model_name": "tiny",
            "status": "running",
            "ppl_profile": "wiki",
            "locked_count": 3,
            "candidate_count": 9,
            "recent_decisions": [{"tensor": "blk.0.attn_q.weight"}],
            "artifacts": {"events": "/tmp/events.jsonl"},
        }
    )

    assert summary == {
        "model": "gemma/tiny",
        "status": "running",
        "release_label": "tiny",
    }


def test_github_upload_plan_redacts_run_id_in_public_mode(tmp_path: Path):
    sidecar = tmp_path / "cerebellum_summary.json"
    sidecar.write_text("{}", encoding="utf-8")

    plan = github_upload_plan(
        {"run_id": "gemma4-live", "model_name": "Gemma 4 12B", "status": "complete"},
        [sidecar],
        "deucebucket/cerebellum-dev",
        None,
    )

    assert plan["branch"] == "cerebellum-release-gemma-4-12b"
    assert plan["mode"] == "public"
    assert plan["files"][0]["github_path"] == "cerebellum_releases/gemma-4-12b/cerebellum_summary.json"
    assert plan["files"][0]["size_bytes"] == 2
    assert plan["report"] == {"model": "Gemma 4 12B", "status": "complete", "release_label": "gemma-4-12b"}


def test_github_upload_plan_keeps_run_paths_only_private(tmp_path: Path):
    sidecar = tmp_path / "cerebellum_summary.json"
    sidecar.write_text("{}", encoding="utf-8")

    plan = github_upload_plan(
        {"run_id": "gemma4-live", "model_name": "Gemma 4 12B", "status": "complete"},
        [sidecar],
        "deucebucket/cerebellum-dev",
        None,
        private=True,
    )

    assert plan["branch"] == "cerebellum-run-gemma4-live"
    assert plan["mode"] == "private"
    assert plan["files"][0]["github_path"] == "cerebellum_runs/gemma4-live/cerebellum_summary.json"
    assert plan["report"]["run_id"] == "gemma4-live"


def test_public_github_upload_result_omits_local_paths(tmp_path: Path, monkeypatch):
    sidecar = tmp_path / "MODEL_CARD_CEREBELLUM.md"
    sidecar.write_text("safe", encoding="utf-8")
    monkeypatch.setattr("osmosis.hillstep.ensure_github_branch", lambda _repo, _branch: None)
    monkeypatch.setattr("osmosis.hillstep.github_file_sha", lambda _repo, _branch, _path: None)
    monkeypatch.setattr("osmosis.hillstep.gh_json", lambda _args: {})

    uploaded = upload_github_sidecars(
        "deucebucket/cerebellum",
        "public",
        [{"path": str(sidecar), "github_path": "cerebellum_runs/public/MODEL_CARD_CEREBELLUM.md"}],
        include_local_paths=False,
    )

    assert uploaded == [{"github_path": "cerebellum_runs/public/MODEL_CARD_CEREBELLUM.md", "branch": "public"}]


def test_public_cli_exposes_imatrix_subcommand():
    top = subprocess.run(
        [sys.executable, "-m", "cerebellum.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "cerebellum.cli", "imatrix", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "imatrix" in top.stdout
    assert "Generate a Cerebellum/llama.cpp imatrix" in result.stdout
    assert "--model" in result.stdout


def test_doctor_warns_for_gemma4_source_architecture_mismatch(tmp_path: Path, monkeypatch, capsys):
    source = tmp_path / "gemma-4-12b-it-f16.gguf"
    source.write_bytes(b"fake")
    monkeypatch.setattr("osmosis.hillstep.gguf_field_text", lambda _path, _key: "llama")

    doctor_cmd(types.SimpleNamespace(source_gguf=str(source), json=True))

    payload = json.loads(capsys.readouterr().out)
    check = next(row for row in payload["checks"] if row["name"] == "gemma4 architecture")
    assert check["ok"] is False
    assert "Gemma4UnifiedForConditionalGeneration" in check["fix"]


def test_doctor_warns_for_gemma4_unstripped_language_model_prefix(tmp_path: Path, monkeypatch, capsys):
    source = tmp_path / "gemma-4-12b-it-f16.gguf"
    source.write_bytes(b"fake")
    monkeypatch.setattr("osmosis.hillstep.gguf_field_text", lambda _path, _key: "gemma4")

    class FakeReader:
        tensors = [types.SimpleNamespace(name="model.language_model.blk.0.ffn_down.weight")]

    monkeypatch.setattr("gguf.GGUFReader", lambda _path: FakeReader())

    doctor_cmd(types.SimpleNamespace(source_gguf=str(source), json=True))

    payload = json.loads(capsys.readouterr().out)
    check = next(row for row in payload["checks"] if row["name"] == "gemma4 architecture")
    assert check["ok"] is False
    assert "model.language_model.* is stripped" in check["fix"]
