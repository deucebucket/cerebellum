import json
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cerebellum import (
    EventLog,
    active_work_status,
    ablation_analyze_cmd,
    analyze_ablation_input,
    benchmark_plan,
    benchmark_plan_markdown,
    benchmark_report,
    benchmark_report_markdown,
    clear_terminal_markers,
    compare_locks,
    build_recovery_plan,
    build_watch_model,
    discover_projects,
    doctor_cmd,
    eta_grid_values,
    github_upload_plan,
    grid_watch_cmd,
    inspect_gguf_types,
    is_quantizable_tensor,
    locked_layer_lines,
    package_files,
    package_manifest,
    parse_args,
    pipeline_plan,
    pipeline_plan_markdown,
    pipeline_plan_cmd,
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
        json.dumps({"benchmark": "mmlu_pro", "model": "cerebellum", "accuracy": 0.72, "size_gib": 8.0}),
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
    assert report["leaderboard"][0]["model"] == "cerebellum"
    assert report["leaderboard"][0]["average_score"] == 69.0
    assert report["leaderboard"][0]["score_per_gib"] == 8.625
    assert "| cerebellum | 69.00% | 2 | 8.00 | 8.62 |" in markdown
    assert "| baseline | 50.00% | 1 | 10.00 | 5.00 |" in markdown


def test_benchmark_report_command_parses():
    args = parse_args(["benchmark-report", "benchmarks/qwen36-27b", "--baseline", "q4", "--leaderboard", "--suite", "frontier", "--size", "q4=8.0", "--json"])

    assert args.cmd == "benchmark-report"
    assert args.paths == ["benchmarks/qwen36-27b"]
    assert args.baseline == "q4"
    assert args.leaderboard is True
    assert args.suite == "frontier"
    assert args.size == ["q4=8.0"]
    assert args.json is True


def test_benchmark_plan_lists_commands_and_pending_frontier():
    release = benchmark_plan("release", model="gemma4_12b", port=18080, results_dir="out")
    markdown = benchmark_plan_markdown(release)
    frontier = benchmark_plan("frontier", model="gemma4_12b", port=18080, results_dir="out")

    arc = next(row for row in release["rows"] if row["benchmark"] == "arc")
    humaneval = next(row for row in release["rows"] if row["benchmark"] == "humaneval")
    pending = {row["benchmark"]: row["status"] for row in frontier["rows"]}

    assert "BENCH_MODEL=gemma4_12b" in arc["command"]
    assert "BENCH_PORT=18080" in arc["command"]
    assert "BENCH_WORKERS=4" in arc["command"]
    assert "scripts/benchmark_arc.py" in arc["command"]
    assert humaneval["workers"] == 1
    assert "BENCH_MAX_TOKENS=4096" in humaneval["command"]
    assert pending == {
        "mmlu_pro": "pending",
        "gpqa_diamond": "pending",
        "mmmlu": "pending",
        "hle_no_tools": "pending",
        "livecodebench_v6": "pending",
    }
    assert "| arc | implemented | 4 |" in markdown
    assert "out/gemma4_12b_arc_results.json" in markdown


def test_benchmark_plan_command_parses():
    args = parse_args(["benchmark-plan", "--suite", "frontier", "--model", "m", "--port", "18080", "--results-dir", "out", "--json"])

    assert args.cmd == "benchmark-plan"
    assert args.suite == "frontier"
    assert args.model == "m"
    assert args.port == 18080
    assert args.results_dir == "out"
    assert args.json is True


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


def test_task_profiles_command_outputs_catalog(capsys):
    markdown = task_profiles_markdown()
    args = parse_args(["task-profiles", "--json"])

    task_profiles_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert "code" in data["profiles"]
    assert data["profiles"]["tools"]["ppl_profile"] == "agentic"
    assert "| code | code | full | humaneval, evalplus, livecodebench_v6 |" in markdown


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


def test_package_files_default_to_public_safe_finalize_sidecars(tmp_path: Path):
    run_dir = tmp_path / "run"
    finalize = run_dir / "finalize"
    finalize.mkdir(parents=True)
    raw_names = [
        "manifest.json",
        "state.json",
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

    assert public == {"cerebellum_gguf_metadata.json", "cerebellum_gguf_metadata.env", "MODEL_CARD_CEREBELLUM.md"}
    assert set(raw_names).issubset(private)


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
    assert "run_dir" not in payload
    assert [item["name"] for item in payload["files"]] == ["MODEL_CARD_CEREBELLUM.md"]
    assert "path" not in payload["files"][0]


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
        "run_id": "run",
        "model": "gemma/tiny",
        "status": "running",
        "ppl_profile": "wiki",
        "locked_count": 3,
        "candidate_count": 9,
    }


def test_github_upload_plan_uses_run_sidecar_paths(tmp_path: Path):
    sidecar = tmp_path / "cerebellum_summary.json"
    sidecar.write_text("{}", encoding="utf-8")

    plan = github_upload_plan(
        {"run_id": "gemma4-live", "status": "complete"},
        [sidecar],
        "deucebucket/cerebellum-dev",
        None,
    )

    assert plan["branch"] == "cerebellum-run-gemma4-live"
    assert plan["mode"] == "public"
    assert plan["files"][0]["github_path"] == "cerebellum_runs/gemma4-live/cerebellum_summary.json"
    assert plan["files"][0]["size_bytes"] == 2


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
