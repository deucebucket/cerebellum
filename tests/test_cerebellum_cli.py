import json
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cerebellum.hillstep as hillstep
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
    benchmark_audit_args_from_query,
    benchmark_audit_cmd,
    benchmark_audit_markdown,
    benchmark_ingest,
    benchmark_ingest_cmd,
    benchmark_ingest_markdown,
    db_benchmark_leaderboard,
    db_benchmark_leaderboard_markdown,
    benchmark_manifest,
    benchmark_manifest_args_from_query,
    benchmark_manifest_cmd,
    benchmark_manifest_markdown,
    benchmark_plan,
    benchmark_plan_args_from_query,
    benchmark_plan_cmd,
    benchmark_plan_markdown,
    benchmark_postprocess_cmd,
    benchmark_records,
    benchmark_run_cmd,
    benchmark_run_execute,
    benchmark_run_markdown,
    benchmark_run_plan,
    benchmark_run_postprocess,
    benchmark_status,
    benchmark_status_args_from_query,
    benchmark_status_markdown,
    benchmark_rebench_plan,
    benchmark_rebench_plan_args_from_query,
    benchmark_rebench_plan_markdown,
    benchmark_report,
    benchmark_report_args_from_query,
    benchmark_report_markdown,
    cerebellum_metadata_block,
    clear_terminal_markers,
    cleanup_cmd,
    compare_locks,
    build_recovery_plan,
    build_watch_model,
    compare_gguf_types,
    compare_gguf_types_args_from_query,
    compare_gguf_types_markdown,
    cpu_offload_smoke_cmd,
    cpu_offload_smoke_args_from_query,
    cpu_offload_smoke_markdown,
    cpu_offload_smoke_payload,
    cpu_offload_build_plan_cmd,
    cpu_offload_build_plan_markdown,
    cpu_offload_build_plan_payload,
    discover_projects,
    doctor_cmd,
    eta_detail_values,
    eta_grid_values,
    github_upload_plan,
    grid_watch_cmd,
    hf_model_stats,
    hf_model_stats_markdown,
    hf_stats_args_from_query,
    golden_cow_audit,
    discover_recent_benchmark_scores,
    home_payload,
    home_markdown,
    history_html,
    history_markdown,
    history_scan_roots,
    write_hf_stats_snapshot,
    inspect_gguf_types,
    inspect_gguf_types_args_from_query,
    is_quantizable_tensor,
    locked_layer_lines,
    legacy_flow_markdown,
    legacy_flow_payload,
    legacy_flow_execute_forward,
    legacy_flow_watch_lines,
    legacy_plan_payload,
    package_files,
    package_manifest,
    parse_args,
    queue_add_job,
    queue_cancel_job,
    queue_cmd,
    queue_get_job,
    queue_list_jobs,
    queue_markdown,
    queue_retry_job,
    pipeline_plan,
    pipeline_plan_args_from_query,
    pipeline_plan_markdown,
    pipeline_plan_cmd,
    pipeline_run_args_from_query,
    pipeline_run_cmd,
    pipeline_run_execute,
    pipeline_run_plan,
    pipeline_status,
    pipeline_status_args_from_query,
    pipeline_status_markdown,
    public_audit,
    public_audit_cmd,
    public_audit_markdown,
    public_model_card_policy_markdown,
    public_model_card_policy_payload,
    public_history_audit,
    public_history_audit_markdown,
    public_export_cmd,
    public_export_plan,
    public_export_plan_args_from_query,
    public_export_markdown,
    release_gate,
    release_gate_cmd,
    release_gate_markdown,
    read_tensor_type_map,
    resume_cmd,
    task_profiles_cmd,
    task_profiles_markdown,
    rollback_cmd,
    run_from_namespace,
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


def test_no_args_parse_to_home_dispatch():
    args = parse_args([])

    assert args.cmd is None


def test_home_command_parses_and_renders(monkeypatch):
    monkeypatch.setattr("cerebellum.hillstep.known_run_dirs", lambda: [])

    args = parse_args(["home", "--limit", "3"])
    payload = home_payload(limit=args.limit)
    rendered = home_markdown(payload)

    assert args.cmd == "home"
    assert "Cerebellum" in rendered
    assert "cerebellum watch" in rendered
    assert "Recent Runs" in rendered


def test_home_discovers_recent_benchmark_scores(tmp_path, monkeypatch):
    results = tmp_path / "benchmark_results"
    results.mkdir()
    (results / "demo_arc_results.json").write_text(
        json.dumps({"model": "demo", "benchmark": "arc_challenge", "accuracy": 91.25}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    scores = discover_recent_benchmark_scores(limit=3)

    assert scores[0]["model"] == "demo"
    assert scores[0]["benchmark"] == "arc"
    assert scores[0]["metric"] == "accuracy"
    assert scores[0]["value"] == 91.25


def test_history_command_parses():
    args = parse_args(["history", "--root", "/tmp/cerebellum", "--query", "gemma", "--include-chat-logs", "--chat-root", "/tmp/chats", "--output", "history.json", "--markdown", "history.md", "--html", "history.html", "--json"])

    assert args.cmd == "history"
    assert args.root == ["/tmp/cerebellum"]
    assert args.query == "gemma"
    assert args.include_chat_logs is True
    assert args.chat_root == ["/tmp/chats"]
    assert args.output == "history.json"
    assert args.markdown == "history.md"
    assert args.html == "history.html"
    assert args.json is True


def test_history_scan_indexes_models_methods_ppl_and_benchmarks(tmp_path: Path):
    model_dir = tmp_path / "osmosis-gemma4-e4b"
    bench_dir = model_dir / "benchmark_results"
    bench_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(
        "## Method\n"
        "Baseline Q3_K_M with PLE Q5_K on WikiText-2.\n"
        "| HumanEval | 68.3% |\n"
        "Perplexity 55.10\n",
        encoding="utf-8",
    )
    (model_dir / "ppl_q3km_ple_Q5K.log").write_text(
        "perplexity: calculating perplexity over 142 chunks, n_ctx=2048, batch_size=2048, n_seq=1\n"
        "Final estimate: PPL = 55.1050 +/- 0.64745\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "families" / "gemma-4" / "gemma-4-12b-it" / "sources" / "google-f16" / "runs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "demo",
                "ppl_profile": "wiki",
                "corpus": "wiki.test.raw",
                "chunks": 128,
                "ctx_size": 2048,
                "base_type": "Q4_K_M",
                "start_type": "q4_K",
                "levels": ["q3_K", "q2_K"],
            }
        ),
        encoding="utf-8",
    )
    (bench_dir / "demo_arc_results.json").write_text(
        json.dumps({"model": "gemma-e4b", "benchmark": "arc_challenge", "accuracy": 85.5, "correct": 855, "total": 1000}),
        encoding="utf-8",
    )

    report = history_scan_roots([tmp_path])
    markdown = history_markdown(report)
    html = history_html(report)

    assert report["schema"] == "cerebellum.history.v1"
    assert any(row["model"] == "osmosis-gemma4-e4b" for row in report["models"])
    assert any(row["model"] == "gemma-4/gemma-4-12b-it" for row in report["models"])
    assert any(row["model"] == "gemma-e4b" for row in report["leaderboard"])
    assert "Baseline Q3_K_M" in markdown
    assert "Search model" in html
    e4b = next(row for row in report["models"] if row["model"] == "osmosis-gemma4-e4b")
    assert e4b["ppl_results"][0]["ppl"] == 55.1050


def test_history_query_filters_content(tmp_path: Path):
    model_dir = tmp_path / "osmosis-qwen36-27b"
    model_dir.mkdir()
    (model_dir / "README.md").write_text("Method: Qwen v4 WikiText ablation.\n", encoding="utf-8")

    report = history_scan_roots([tmp_path], query="qwen v4")

    assert [row["model"] for row in report["models"]] == ["osmosis-qwen36-27b"]


def test_history_scan_indexes_explicit_chat_roots(tmp_path: Path):
    repo = tmp_path / "repo"
    chats = tmp_path / ".codex" / "sessions"
    repo.mkdir()
    chats.mkdir(parents=True)
    (repo / "README.md").write_text("Cerebellum local repo.\n", encoding="utf-8")
    (chats / "session.jsonl").write_text(
        json.dumps({"text": "In /var/home/deucebucket/ai-drive/cerebellum we used Gemma 4 26B with wiki.test.raw and a router recast."}) + "\n",
        encoding="utf-8",
    )

    report = history_scan_roots([repo], include_chat_logs=True, chat_roots=[chats])
    gemma = next(row for row in report["models"] if row["model"] == "gemma-4-26b")

    assert report["include_chat_logs"] is True
    assert gemma["chat_logs"][0]["path"].endswith("session.jsonl")
    assert "wiki.test.raw" in gemma["chat_logs"][0]["text"]


def test_inspect_gguf_types_command_parses():
    args = parse_args(["inspect-gguf-types", "/tmp/model.gguf", "--by-layer", "--json"])

    assert args.cmd == "inspect-gguf-types"
    assert args.gguf == "/tmp/model.gguf"
    assert args.by_layer is True
    assert args.json is True


def test_inspect_gguf_types_api_query_args_validate():
    args = inspect_gguf_types_args_from_query(
        {
            "gguf": ["model.gguf"],
            "by_layer": ["true"],
            "by_component": ["yes"],
        }
    )

    assert args.gguf == "model.gguf"
    assert args.by_layer is True
    assert args.by_component is True
    assert args.json is True

    try:
        inspect_gguf_types_args_from_query({})
    except ValueError as exc:
        assert "gguf query param required" in str(exc)
    else:
        raise AssertionError("inspect-gguf-types API query should require gguf")


def test_compare_gguf_types_command_parses():
    args = parse_args(["compare-gguf-types", "base.gguf", "cand.gguf", "--baseline-label", "q4", "--candidate-label", "dynamic", "--reference-map", "types.txt", "--json"])

    assert args.cmd == "compare-gguf-types"
    assert args.baseline == "base.gguf"
    assert args.candidate == "cand.gguf"
    assert args.baseline_label == "q4"
    assert args.candidate_label == "dynamic"
    assert args.reference_map == "types.txt"
    assert args.json is True


def test_compare_gguf_types_api_query_args_validate():
    args = compare_gguf_types_args_from_query(
        {
            "baseline": ["base.gguf"],
            "candidate": ["cand.gguf"],
            "baseline_label": ["q4"],
            "candidate_label": ["dynamic"],
            "reference_map": ["types.txt"],
        }
    )

    assert args.baseline == "base.gguf"
    assert args.candidate == "cand.gguf"
    assert args.baseline_label == "q4"
    assert args.candidate_label == "dynamic"
    assert args.reference_map == "types.txt"
    assert args.json is True

    try:
        compare_gguf_types_args_from_query({"candidate": ["cand.gguf"]})
    except ValueError as exc:
        assert "baseline query param required" in str(exc)
    else:
        raise AssertionError("compare-gguf-types API query should require baseline")

    try:
        compare_gguf_types_args_from_query({"baseline": ["base.gguf"]})
    except ValueError as exc:
        assert "candidate query param required" in str(exc)
    else:
        raise AssertionError("compare-gguf-types API query should require candidate")


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


def test_watch_events_limit_defaults_to_five_rows():
    args = parse_args(["watch", "/tmp/run", "--once", "--plain"])

    assert args.events_limit == 5


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


def test_legacy_plan_command_parses():
    args = parse_args(["legacy-plan", "/tmp/run", "--json"])

    assert args.cmd == "legacy-plan"
    assert args.run_dir == "/tmp/run"
    assert args.json is True


def test_legacy_flow_command_defaults_to_classic_group_survivability_candidate():
    args = parse_args(
        [
            "legacy-flow",
            "--source-gguf",
            "/tmp/gemma-f16.gguf",
            "--output-dir",
            "/tmp/cerebellum-gemma",
            "--model-name",
            "gemma-4-12b-it",
            "--json",
        ]
    )

    plan = legacy_flow_payload(args)
    rendered = legacy_flow_markdown(plan)
    phase_names = [phase["name"] for phase in plan["phases"]]

    assert args.cmd == "legacy-flow"
    assert phase_names == [
        "scan",
        "baseline",
        "survivability-scan",
        "target-selection",
        "build-v1-stacked-q2",
        "reverse-ablation",
        "build-candidate",
        "benchmark-gate",
    ]
    assert plan["mode"] == "classic group-first Cerebellum"
    assert "cerebellum watch" in plan["watch"]["private"]
    assert "quant_worker" in plan["orchestration"]["quant_ppl_overlap"]
    assert "ppl_worker" in plan["orchestration"]["quant_ppl_overlap"]
    assert plan["survivability"]["mode"] == "classic Q2_K no-commit group survivability scan"
    assert plan["survivability"]["levels"] == "q2_K"
    assert plan["survivability"]["target_type"] == "q2_K"
    assert plan["survivability"]["commit_locks"] is False
    assert plan["survivability"]["with_targeted_hillstep"] is False
    assert "group-scan" in plan["phases"][2]["groups"][0]["command_template"]
    assert "--target-type q2_K" in plan["phases"][2]["groups"][0]["command_template"]
    assert "--output /tmp/cerebellum-gemma/selected_survivability_overrides.txt" in plan["phases"][3]["command"]
    assert "--tensor-type-file /tmp/cerebellum-gemma/final_group_overrides.txt" in plan["phases"][6]["command"]
    assert "--pure" not in plan["phases"][6]["command"]
    assert "Survivability Groups" in rendered
    assert "selected_survivability_overrides.txt" in rendered
    assert "reverse-ablation" in rendered


def test_legacy_flow_can_add_optional_targeted_hillstep_after_group_scan():
    args = parse_args(
        [
            "legacy-flow",
            "--source-gguf",
            "/tmp/gemma-f16.gguf",
            "--output-dir",
            "/tmp/cerebellum-gemma",
            "--with-targeted-hillstep",
        ]
    )

    plan = legacy_flow_payload(args)
    phase_names = [phase["name"] for phase in plan["phases"]]

    assert phase_names.index("survivability-scan") < phase_names.index("targeted-hillstep")
    assert phase_names.index("targeted-hillstep") < phase_names.index("build-candidate")
    assert plan["survivability"]["with_targeted_hillstep"] is True
    assert "--tensor-file /tmp/cerebellum-gemma/target_tensors.txt" in plan["phases"][6]["command"]
    assert "--tensor-type-file /tmp/cerebellum-gemma/run/artifacts/final_types.txt" in plan["phases"][7]["command"]


def test_legacy_flow_can_execute_forward_group_queue():
    args = parse_args(
        [
            "legacy-flow",
            "--source-gguf",
            "/tmp/gemma-f16.gguf",
            "--output-dir",
            "/tmp/cerebellum-gemma",
            "--corpus",
            "/tmp/wiki.test.raw",
            "--baseline-ppl",
            "5433.7967",
            "--family",
            "gemma-4",
            "--source-name",
            "google-f16",
            "--execute-forward",
            "--keep-candidates",
        ]
    )
    plan = legacy_flow_payload(args)

    assert args.execute_forward is True
    assert args.keep_candidates is True
    assert "--family gemma-4" in plan["phases"][2]["groups"][0]["command_template"]
    assert "--source-name google-f16" in plan["phases"][2]["groups"][0]["command_template"]


def test_legacy_flow_execute_forward_defaults_to_pipelined_mode(tmp_path: Path, monkeypatch):
    source = tmp_path / "model-f16.gguf"
    source.write_bytes(b"gguf")
    corpus = tmp_path / "wiki.test.raw"
    corpus.write_text("hello\n", encoding="utf-8")
    args = parse_args(
        [
            "legacy-flow",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(tmp_path / "run"),
            "--corpus",
            str(corpus),
            "--baseline-ppl",
            "10.0",
            "--model-name",
            "tiny",
            "--execute-forward",
        ]
    )
    plan = legacy_flow_payload(args)
    monkeypatch.setattr(
        "cerebellum.hillstep.LEGACY_GATED_GROUPS",
        [
            {"name": "attn-q", "patterns": ["attn_q"], "default_floor": "q4_K"},
            {"name": "attn-v", "patterns": ["attn_v"], "default_floor": "q4_K"},
        ],
    )

    def fake_prepare(ns):
        run_dir = Path(ns.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"args": ns, "run_dir": run_dir, "group_name": ns.group_name, "events": types.SimpleNamespace(write=lambda *a, **k: None)}

    monkeypatch.setattr("cerebellum.hillstep.group_scan_prepare", fake_prepare)
    monkeypatch.setattr("cerebellum.hillstep.group_scan_assert_disk_floor", lambda *a, **k: None)
    monkeypatch.setattr("cerebellum.hillstep.group_scan_run_quant", lambda job: (0, "", 1.0))
    monkeypatch.setattr("cerebellum.hillstep.group_scan_finalize_quant", lambda job, q_rc, q_out, q_seconds: True)
    monkeypatch.setattr(
        "cerebellum.hillstep.group_scan_run_ppl",
        lambda job, q_seconds: {"status": "complete", "ppl": 9.0, "delta": -1.0},
    )

    result = legacy_flow_execute_forward(args, plan)

    assert result["mode"] == "pipelined"
    assert [row["group"] for row in result["executed"]] == ["attn-q", "attn-v"]


def test_group_scan_command_parses_classic_group_ablation_unit():
    args = parse_args(
        [
            "group-scan",
            "--source-gguf",
            "/tmp/model-f16.gguf",
            "--corpus",
            "/tmp/wiki.txt",
            "--run-dir",
            "/tmp/group",
            "--tensor-regex",
            "ffn_down",
            "--group-name",
            "ffn-down",
            "--target-type",
            "q2_K",
            "--baseline-ppl",
            "123.0",
            "--dry-run",
        ]
    )

    assert args.cmd == "group-scan"
    assert args.group_name == "ffn-down"
    assert args.target_type == "q2_K"
    assert args.baseline_ppl == 123.0
    assert args.dry_run is True


def test_sparse_replay_command_parses_and_dispatches(monkeypatch):
    args = parse_args(
        [
            "sparse-replay",
            "--source-gguf",
            "/tmp/model-f16.gguf",
            "--corpus",
            "/tmp/wiki.txt",
            "--run-dir",
            "/tmp/sparse",
            "--baseline-ppl",
            "19.7",
            "--budget-gb",
            "0.462",
            "--dry-run",
        ]
    )

    assert args.cmd == "sparse-replay"
    assert args.probe_base_type == "Q4_K_M"
    assert args.target_type == "q2_K"
    assert args.budget_gb == 0.462

    called = {}
    monkeypatch.setattr(hillstep, "sparse_replay_cmd", lambda ns: called.update(vars(ns)))
    hillstep.main(
        [
            "sparse-replay",
            "--source-gguf",
            "/tmp/model-f16.gguf",
            "--corpus",
            "/tmp/wiki.txt",
            "--run-dir",
            "/tmp/sparse",
            "--baseline-ppl",
            "19.7",
            "--budget-gb",
            "0.462",
            "--dry-run",
        ]
    )
    assert called["cmd"] == "sparse-replay"


def test_legacy_flow_watch_prefers_live_manifest_for_scan_settings():
    context = {
        "mode": "automated group-first targeted-hillstep",
        "phase": "survivability-scan",
        "group": {"index": 2, "total": 4, "name": "early-blocks", "floor": "q4_K", "patterns": ["blk.0."]},
        "survivability": {
            "levels": "q3_K,q2_K",
            "mode": "lower-only no-commit survivability scan",
            "pure_quant": True,
            "commit_locks": False,
            "max_regression_pct": 2.0,
            "target_tensor_file": "/tmp/target_tensors.txt",
        },
    }
    manifest = {
        "levels": ["q3_K", "q2_K", "q5_K", "q6_K", "f16"],
        "pure_quant_effective": False,
        "commit_locks": False,
    }

    lines = legacy_flow_watch_lines(context, {"current_ppl": None}, {"event": "baseline_quant_start"}, manifest)
    rendered = "\n".join(lines)

    assert "levels=q3_K,q2_K,q5_K,q6_K,f16" in rendered
    assert "scan=bidirectional no-commit survivability scan" in rendered
    assert "pure_quant_effective=False" in rendered
    assert "commit_locks=False" in rendered


def test_ablation_analyze_can_write_tensor_names_for_targeted_followup(tmp_path: Path):
    ablation_json = tmp_path / "ablation.json"
    tensor_output = tmp_path / "targets.txt"
    ablation_json.write_text(
        json.dumps(
            {
                "baseline_ppl": 100.0,
                "tests": {
                    "good": {"gguf_tensor": "blk.0.attn_q.weight", "ppl": 99.0},
                    "bad": {"gguf_tensor": "blk.0.ffn_down.weight", "ppl": 130.0},
                },
            }
        ),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "ablation-analyze",
            str(ablation_json),
            "--tensor-output",
            str(tensor_output),
            "--json",
        ]
    )
    ablation_analyze_cmd(args)

    assert tensor_output.read_text(encoding="utf-8").splitlines() == ["blk.0.attn_q.weight"]


def test_public_card_policy_parses_and_hides_recipe_details():
    args = parse_args(["public-card-policy", "--json"])
    payload = public_model_card_policy_payload()
    rendered = public_model_card_policy_markdown(payload)

    assert args.cmd == "public-card-policy"
    assert "Cerebellum Public Model Card Policy" in rendered
    assert "Support / Sponsored Runs" in rendered
    assert "exact tensor" in rendered.lower()


def test_golden_cow_audit_flags_sensitive_low_precision_locks():
    state = {
        "ppl_profile": "wiki",
        "ablation_metric": "ppl",
        "current_ppl": 123.4,
        "locked": {
            "blk.0.attn_k.weight": "q2_K",
            "blk.6.ffn_down.weight": "q4_K",
            "blk.9.ffn_up.weight": "q2_K",
        },
    }

    audit = golden_cow_audit(state, {})

    assert audit["suspect_count"] == 2
    assert audit["counts"]["early_low_precision"] == 1
    assert audit["counts"]["attention_low_precision"] == 1
    assert audit["counts"]["mlp_up_gate_q2"] == 1


def test_golden_cow_audit_labels_survivability_findings_without_rollback():
    state = {"locked": {"blk.0.ffn_down.weight": "q2_K"}}

    audit = golden_cow_audit(state, {}, {"phase": "survivability-scan"})

    assert audit["suspect_count"] == 1
    assert audit["recommended_action"] == "survivability finding; require benchmark gate before final acceptance"


def test_legacy_plan_payload_includes_protected_groups(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_status": "stopped",
                "locked": {"blk.0.attn_output.weight": "q2_K"},
                "current_ppl": 10.0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"source_gguf": "/tmp/model.gguf", "ppl_profile": "wiki", "ablation_metric": "ppl"}),
        encoding="utf-8",
    )

    plan = legacy_plan_payload(run_dir)

    assert plan["profile"]["variant_suffix"] == "legacy-gated"
    assert plan["golden_cow_audit"]["suspect_count"] == 1
    assert any(group["name"] == "attn-output" for group in plan["protected_groups"])


def test_resolve_run_dir_defaults_to_single_live_run(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"model_name":"gemma-4-12b-it"}', encoding="utf-8")
    (run_dir / "state.json").write_text('{"run_status":"running"}', encoding="utf-8")

    monkeypatch.setattr("cerebellum.hillstep.known_run_dirs", lambda: [run_dir])
    monkeypatch.setattr("cerebellum.hillstep.run_is_live", lambda path: path == run_dir)

    assert resolve_run_dir(None) == run_dir
    assert resolve_run_dir("gemma-4-12b-it") == run_dir


def test_resolve_run_dir_falls_back_to_most_recent_completed_run(tmp_path: Path, monkeypatch):
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"
    old_run.mkdir()
    new_run.mkdir()
    (old_run / "manifest.json").write_text('{"model_name":"old"}', encoding="utf-8")
    (old_run / "state.json").write_text('{"run_status":"complete"}', encoding="utf-8")
    (new_run / "manifest.json").write_text('{"model_name":"new"}', encoding="utf-8")
    (new_run / "state.json").write_text('{"run_status":"complete"}', encoding="utf-8")
    os.utime(old_run / "state.json", (1000, 1000))
    os.utime(new_run / "state.json", (2000, 2000))

    monkeypatch.setattr("cerebellum.hillstep.known_run_dirs", lambda: [old_run, new_run])
    monkeypatch.setattr("cerebellum.hillstep.run_is_live", lambda _path: False)

    assert resolve_run_dir(None) == new_run


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

    monkeypatch.setattr("cerebellum.hillstep.child_pids", lambda pid: [pid + 1] if pid == 100 else [])
    monkeypatch.setattr(
        "cerebellum.hillstep.process_rows_for_run",
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


def test_benchmark_records_read_evalplus_chat_summary_fields(tmp_path: Path):
    summary = tmp_path / "model_evalplus_chat_results.json"
    summary.write_text(
        json.dumps(
            {
                "benchmark": "evalplus_humaneval_plus_chat",
                "model": "model",
                "pass_at_1_base": 91.46,
                "pass_at_1_plus": 89.63,
                "total_problems": 164,
            }
        ),
        encoding="utf-8",
    )

    records = benchmark_records([tmp_path])
    manifest = benchmark_manifest([tmp_path], suite="release-local", model="model")

    assert records[0]["benchmark_key"] == "evalplus"
    assert records[0]["metric"] == "pass@1 plus"
    assert records[0]["value"] == 89.63
    assert "evalplus" in manifest["measured_benchmarks"]
    assert "evalplus" not in manifest["missing_measured"]


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


def test_benchmark_report_api_query_args_validate():
    args = benchmark_report_args_from_query(
        {
            "path": ["benchmark_results", "extra_results"],
            "baseline": ["q4"],
            "leaderboard": ["true"],
            "suite": ["frontier"],
            "size": ["q4=8.0"],
            "weight": ["gpqa_diamond=2"],
        }
    )

    assert args.paths == ["benchmark_results", "extra_results"]
    assert args.baseline == "q4"
    assert args.leaderboard is True
    assert args.suite == "frontier"
    assert args.size == ["q4=8.0"]
    assert args.weight == ["gpqa_diamond=2"]

    suite_args = benchmark_report_args_from_query({"list_suites": ["true"]})
    assert suite_args.list_suites is True
    assert suite_args.paths == []

    try:
        benchmark_report_args_from_query({})
    except ValueError as exc:
        assert "path query param required" in str(exc)
    else:
        raise AssertionError("benchmark-report API query should require path unless list_suites=true")

    try:
        benchmark_report_args_from_query({"path": ["benchmark_results"], "suite": ["unknown"]})
    except ValueError as exc:
        assert "unknown benchmark suite unknown" in str(exc)
    else:
        raise AssertionError("invalid benchmark report suite should fail")


def test_benchmark_plan_lists_commands_and_pending_frontier():
    release = benchmark_plan("release", model="gemma4_12b", port=18080, results_dir="out")
    release_local = benchmark_plan("release-local", model="gemma4_12b", port=18080, results_dir="out")
    markdown = benchmark_plan_markdown(release)
    frontier = benchmark_plan("frontier", model="gemma4_12b", port=18080, results_dir="out")

    arc = next(row for row in release["rows"] if row["benchmark"] == "arc")
    speed = next(row for row in release_local["rows"] if row["benchmark"] == "speed")
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
    assert release_local["readiness"]["ready"] is True
    assert release_local["readiness"]["implemented"] == 5
    assert "scripts/benchmark_perf.py" in speed["command"]
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


def test_benchmark_run_command_parses():
    args = parse_args(
        [
            "benchmark-run",
            "--suite",
            "frontier",
            "--model",
            "gemma4",
            "--port",
            "18080",
            "--results-dir",
            "out",
            "--benchmark",
            "mmlu_pro",
            "--execute",
            "--postprocess",
            "--require-complete",
            "--leaderboard",
            "--size",
            "gemma4=7.6",
            "--weight",
            "mmlu_pro=2",
            "--json",
        ]
    )

    assert args.cmd == "benchmark-run"
    assert args.suite == "frontier"
    assert args.model == "gemma4"
    assert args.port == 18080
    assert args.results_dir == "out"
    assert args.benchmark == ["mmlu_pro"]
    assert args.execute is True
    assert args.postprocess is True
    assert args.require_complete is True
    assert args.leaderboard is True
    assert args.size == ["gemma4=7.6"]
    assert args.weight == ["mmlu_pro=2"]
    assert args.json is True


def test_benchmark_run_plan_filters_and_reports_blockers():
    plan = benchmark_run_plan("capability", model="gemma4", port=18080, results_dir="out", benchmarks=["aime_2025"])
    markdown = benchmark_run_markdown(plan)

    assert plan["schema"] == "cerebellum.benchmark_run.v1"
    assert plan["dry_run"] is True
    assert plan["blocked"] is True
    assert [row["benchmark"] for row in plan["benchmarks"]] == ["aime_2025"]
    assert plan["blockers"][0]["benchmark"] == "aime_2025"
    assert "Benchmark Run" in markdown
    assert "Blockers" in markdown

    try:
        benchmark_run_plan("frontier", model="gemma4", port=18080, results_dir="out", benchmarks=["missing"])
    except SystemExit as exc:
        assert "unknown benchmark(s) for suite frontier: missing" in str(exc)
    else:
        raise AssertionError("benchmark-run should reject unknown selected benchmark")


def test_benchmark_run_cmd_executes_selected_benchmark(tmp_path: Path, capsys, monkeypatch):
    output = tmp_path / "bench.out"
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit", ["unit_smoke"])
    monkeypatch.setitem(
        hillstep.BENCHMARK_CATALOG,
        "unit_smoke",
        {
            "name": "Unit smoke",
            "status": "implemented",
            "script": "-c",
            "workers": 1,
            "args": [f"from pathlib import Path; Path({str(output)!r}).write_text('ok')"],
            "artifacts": ["{model}_unit_smoke_results.json"],
        },
    )

    dry_args = parse_args(["benchmark-run", "--suite", "unit", "--model", "unit-model", "--results-dir", str(tmp_path), "--benchmark", "unit_smoke"])
    benchmark_run_cmd(dry_args)
    assert "# Benchmark Run" in capsys.readouterr().out

    execute_args = parse_args(["benchmark-run", "--suite", "unit", "--model", "unit-model", "--results-dir", str(tmp_path), "--benchmark", "unit_smoke", "--execute", "--json"])
    benchmark_run_cmd(execute_args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is False
    assert output.read_text(encoding="utf-8") == "ok"
    assert Path(payload["event_log"]).is_file()
    assert Path(payload["executions"][0]["log"]).is_file()


def test_benchmark_run_postprocess_writes_release_sidecars(tmp_path: Path, capsys, monkeypatch):
    summary = tmp_path / "unit-model_unit_smoke_results.json"
    detail = tmp_path / "unit-model_unit_smoke_detailed.jsonl"
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_post", ["unit_smoke"])
    monkeypatch.setitem(
        hillstep.BENCHMARK_CATALOG,
        "unit_smoke",
        {
            "name": "Unit smoke",
            "status": "implemented",
            "script": "-c",
            "workers": 1,
            "args": [
                "from pathlib import Path; "
                f"Path({str(summary)!r}).write_text('{{{{\"benchmark\":\"unit_smoke\",\"model\":\"unit-model\",\"accuracy\":0.75,\"size_gib\":4.0}}}}'); "
                f"Path({str(detail)!r}).write_text('{{{{\"correct\":true,\"predicted\":\"A\",\"raw_response\":\"A\"}}}}\\n')"
            ],
            "artifacts": ["{model}_unit_smoke_results.json", "{model}_unit_smoke_detailed.jsonl"],
        },
    )
    args = parse_args(
        [
            "benchmark-run",
            "--suite",
            "unit_post",
            "--model",
            "unit-model",
            "--results-dir",
            str(tmp_path),
            "--execute",
            "--postprocess",
            "--require-complete",
            "--leaderboard",
            "--size",
            "unit-model=4.0",
            "--json",
        ]
    )

    benchmark_run_cmd(args)
    payload = json.loads(capsys.readouterr().out)
    postprocess = payload["postprocess"]

    assert payload["blocked"] is False
    assert postprocess["schema"] == "cerebellum.benchmark_postprocess.v1"
    assert postprocess["blocked"] is False
    assert postprocess["leaderboard_rows"] == 1
    assert Path(postprocess["manifest"]).is_file()
    assert Path(postprocess["audit"]).is_file()
    assert Path(postprocess["report"]).is_file()
    manifest = json.loads(Path(postprocess["manifest"]).read_text(encoding="utf-8"))
    report = json.loads(Path(postprocess["report"]).read_text(encoding="utf-8"))
    assert manifest["missing_measured"] == []
    assert report["leaderboard"][0]["model"] == "unit-model"


def test_benchmark_run_postprocess_blocks_missing_complete_suite(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_missing", ["missing"])
    plan = {
        "suite": "unit_missing",
        "model": "unit-model",
        "results_dir": str(tmp_path),
    }

    postprocess = benchmark_run_postprocess(plan, require_complete=True)

    assert postprocess["blocked"] is True
    assert postprocess["missing_measured"] == ["missing"]
    assert postprocess["blockers"][0]["status"] == "missing"


def test_benchmark_postprocess_command_parses():
    args = parse_args(
        [
            "benchmark-postprocess",
            "benchmark_results",
            "--suite",
            "frontier",
            "--model",
            "gemma4",
            "--require-complete",
            "--leaderboard",
            "--size",
            "gemma4=7.6",
            "--weight",
            "gpqa_diamond=2",
            "--json",
        ]
    )

    assert args.cmd == "benchmark-postprocess"
    assert args.results_dir == "benchmark_results"
    assert args.suite == "frontier"
    assert args.model == "gemma4"
    assert args.require_complete is True
    assert args.leaderboard is True
    assert args.size == ["gemma4=7.6"]
    assert args.weight == ["gpqa_diamond=2"]
    assert args.json is True


def test_benchmark_postprocess_cmd_writes_sidecars_for_existing_artifacts(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_post_cli", ["unit_smoke"])
    (tmp_path / "unit-model_unit_smoke_results.json").write_text(
        json.dumps({"benchmark": "unit_smoke", "model": "unit-model", "accuracy": 0.75, "size_gib": 4.0}),
        encoding="utf-8",
    )
    (tmp_path / "unit-model_unit_smoke_detailed.jsonl").write_text(
        json.dumps({"correct": True, "predicted": "A", "raw_response": "A"}) + "\n",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "benchmark-postprocess",
            str(tmp_path),
            "--suite",
            "unit_post_cli",
            "--model",
            "unit-model",
            "--require-complete",
            "--leaderboard",
            "--size",
            "unit-model=4.0",
            "--json",
        ]
    )

    benchmark_postprocess_cmd(args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == "cerebellum.benchmark_postprocess.v1"
    assert payload["blocked"] is False
    assert payload["leaderboard_rows"] == 1
    assert Path(payload["manifest"]).is_file()
    assert Path(payload["audit"]).is_file()
    assert Path(payload["report"]).is_file()


def test_benchmark_postprocess_cmd_blocks_missing_complete_suite(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_post_missing", ["missing"])
    args = parse_args(["benchmark-postprocess", str(tmp_path), "--suite", "unit_post_missing", "--require-complete", "--json"])

    try:
        benchmark_postprocess_cmd(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("benchmark-postprocess should fail when require-complete is missing results")
    payload = json.loads(capsys.readouterr().out)

    assert payload["blocked"] is True
    assert payload["missing_measured"] == ["missing"]


def test_benchmark_ingest_persists_ready_results(tmp_path: Path, monkeypatch):
    db = tmp_path / "cerebellum.db"
    summary = tmp_path / "unit-model_unit_smoke_results.json"
    detail = tmp_path / "unit-model_unit_smoke_detailed.jsonl"
    summary.write_text(
        json.dumps({"benchmark": "unit_smoke", "model": "unit-model", "accuracy": 0.75, "size_gib": 4.0}),
        encoding="utf-8",
    )
    detail.write_text(json.dumps({"correct": True, "predicted": "A", "raw_response": "A"}) + "\n", encoding="utf-8")
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_ingest", ["unit_smoke"])

    payload = benchmark_ingest(
        db,
        tmp_path,
        suite="unit_ingest",
        model="unit-model",
        require_complete=True,
        leaderboard=True,
        sizes={"unit-model": 4.0},
    )
    markdown = benchmark_ingest_markdown(payload)
    ingests = hillstep.sqlite_rows(db, "SELECT model, suite, ready FROM cerebellum_benchmark_ingests")
    results = hillstep.sqlite_rows(db, "SELECT model, benchmark_key, metric, value FROM cerebellum_benchmark_results")

    assert payload["schema"] == "cerebellum.benchmark_ingest.v1"
    assert payload["ready"] is True
    assert payload["records"] == 1
    assert payload["leaderboard_rows"] == 1
    assert ingests == [{"model": "unit-model", "suite": "unit_ingest", "ready": 1}]
    assert results == [{"model": "unit-model", "benchmark_key": "unit_smoke", "metric": "accuracy", "value": 75.0}]
    assert "# Benchmark Ingest" in markdown


def test_benchmark_ingest_cmd_blocks_missing_complete_suite(tmp_path: Path, capsys, monkeypatch):
    db = tmp_path / "cerebellum.db"
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_ingest_missing", ["missing"])
    args = parse_args(
        [
            "benchmark-ingest",
            str(tmp_path),
            "--db",
            str(db),
            "--suite",
            "unit_ingest_missing",
            "--model",
            "unit-model",
            "--require-complete",
            "--json",
        ]
    )

    try:
        benchmark_ingest_cmd(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("benchmark-ingest should fail when require-complete is missing results")
    payload = json.loads(capsys.readouterr().out)

    assert payload["ready"] is False
    assert payload["missing_measured"] == ["missing"]
    assert payload["blockers"][0]["status"] == "missing"
    assert hillstep.sqlite_rows(db, "SELECT ready FROM cerebellum_benchmark_ingests") == [{"ready": 0}]


def test_benchmark_ingest_command_parses():
    args = parse_args(
        [
            "benchmark-ingest",
            "benchmark_results",
            "--db",
            "db/cerebellum.db",
            "--suite",
            "frontier",
            "--model",
            "gemma4",
            "--require-complete",
            "--leaderboard",
            "--size",
            "gemma4=7.6",
            "--weight",
            "gpqa_diamond=2",
            "--json",
        ]
    )

    assert args.cmd == "benchmark-ingest"
    assert args.results_dir == "benchmark_results"
    assert args.db == "db/cerebellum.db"
    assert args.suite == "frontier"
    assert args.model == "gemma4"
    assert args.require_complete is True
    assert args.leaderboard is True
    assert args.size == ["gemma4=7.6"]
    assert args.weight == ["gpqa_diamond=2"]


def test_db_benchmark_leaderboard_reads_latest_ready_ingests(tmp_path: Path, monkeypatch):
    db = tmp_path / "cerebellum.db"
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_db_board", ["arc", "mmlu"])
    for model, arc, mmlu, size in [
        ("small", 0.80, 0.60, 4.0),
        ("large", 0.90, 0.70, 10.0),
    ]:
        result_dir = tmp_path / model
        result_dir.mkdir()
        (result_dir / f"{model}_arc_results.json").write_text(
            json.dumps({"benchmark": "arc", "model": model, "accuracy": arc, "size_gib": size}),
            encoding="utf-8",
        )
        (result_dir / f"{model}_mmlu_results.json").write_text(
            json.dumps({"benchmark": "mmlu", "model": model, "accuracy": mmlu, "size_gib": size}),
            encoding="utf-8",
        )
        benchmark_ingest(db, result_dir, suite="unit_db_board", model=model, require_complete=True)

    payload = db_benchmark_leaderboard(db, suite="unit_db_board", weights={"mmlu": 2.0})
    markdown = db_benchmark_leaderboard_markdown(payload)

    assert payload["schema"] == "cerebellum.db_benchmark_leaderboard.v1"
    assert payload["records"] == 4
    assert payload["weight_policy"] == {"arc": 1.0, "mmlu": 2.0}
    assert [row["model"] for row in payload["leaderboard"]] == ["large", "small"]
    assert payload["leaderboard"][0]["average_score"] == 76.66666666666667
    assert payload["leaderboard"][0]["score_per_gib"] == 7.666666666666667
    assert "Cerebellum Benchmark Leaderboard" in markdown
    assert "Score/GiB" in markdown


def test_db_benchmark_leaderboard_cmd_outputs_json(tmp_path: Path, monkeypatch, capsys):
    db = tmp_path / "cerebellum.db"
    result_dir = tmp_path / "bench"
    result_dir.mkdir()
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_db_cli", ["arc"])
    (result_dir / "model_arc_results.json").write_text(
        json.dumps({"benchmark": "arc", "model": "model", "accuracy": 0.75, "size_gib": 5.0}),
        encoding="utf-8",
    )
    benchmark_ingest(db, result_dir, suite="unit_db_cli", model="model", require_complete=True)
    args = parse_args(["db", "--db", str(db), "--json", "leaderboard", "--suite", "unit_db_cli", "--limit", "5"])

    hillstep.db_cmd(args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == "cerebellum.db_benchmark_leaderboard.v1"
    assert payload["leaderboard"][0]["model"] == "model"
    assert payload["leaderboard"][0]["score_per_gib"] == 15.0


def test_benchmark_run_execute_stops_on_failure(tmp_path: Path, monkeypatch):
    skipped = tmp_path / "skipped.out"
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_fail", ["fail", "skip"])
    monkeypatch.setitem(
        hillstep.BENCHMARK_CATALOG,
        "fail",
        {"name": "Fail", "status": "implemented", "script": "-c", "args": ["raise SystemExit(6)"]},
    )
    monkeypatch.setitem(
        hillstep.BENCHMARK_CATALOG,
        "skip",
        {"name": "Skip", "status": "implemented", "script": "-c", "args": [f"from pathlib import Path; Path({str(skipped)!r}).write_text('bad')"]},
    )

    plan = benchmark_run_plan("unit_fail", model="unit-model", port=18080, results_dir=str(tmp_path))
    result = benchmark_run_execute(plan)

    assert result["dry_run"] is False
    assert result["blocked"] is True
    assert result["blockers"] == [{"benchmark": "fail", "status": "implemented", "reason": "command exited 6"}]
    assert [row["benchmark"] for row in result["executions"]] == ["fail"]
    assert not skipped.exists()


def test_benchmark_status_command_parses():
    args = parse_args(["benchmark-status", "--results-dir", "out", "--events", "events.jsonl", "--json"])

    assert args.cmd == "benchmark-status"
    assert args.results_dir == "out"
    assert args.events == "events.jsonl"
    assert args.json is True


def test_benchmark_status_reports_complete_run(tmp_path: Path, monkeypatch):
    output = tmp_path / "bench.out"
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_status", ["unit_smoke"])
    monkeypatch.setitem(
        hillstep.BENCHMARK_CATALOG,
        "unit_smoke",
        {
            "name": "Unit smoke",
            "status": "implemented",
            "script": "-c",
            "workers": 1,
            "args": [f"from pathlib import Path; Path({str(output)!r}).write_text('ok')"],
        },
    )
    benchmark_run_execute(benchmark_run_plan("unit_status", model="unit-model", port=18080, results_dir=str(tmp_path)))

    status = benchmark_status(tmp_path)
    markdown = benchmark_status_markdown(status)

    assert status["schema"] == "cerebellum.benchmark_status.v1"
    assert status["status"] == "complete"
    assert status["completed_benchmarks"] == 1
    assert status["rerun_command"] is None
    assert status["benchmarks"][0]["status"] == "complete"
    assert "# Benchmark Status" in markdown


def test_benchmark_status_reports_failed_rerun_command(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_status_fail", ["fail"])
    monkeypatch.setitem(
        hillstep.BENCHMARK_CATALOG,
        "fail",
        {"name": "Fail", "status": "implemented", "script": "-c", "args": ["raise SystemExit(6)"]},
    )
    benchmark_run_execute(benchmark_run_plan("unit_status_fail", model="unit-model", port=18080, results_dir=str(tmp_path)))

    status = benchmark_status(tmp_path)

    assert status["status"] == "failed"
    assert status["failed_benchmark"] == "fail"
    assert status["rerun_benchmark"] == "fail"
    assert "raise SystemExit(6)" in status["rerun_command"]
    assert status["benchmarks"][0]["returncode"] == 6


def test_benchmark_status_api_query_args_default_results_dir():
    args = benchmark_status_args_from_query({"events": ["events.jsonl"]})

    assert args.results_dir == "benchmark_results"
    assert args.events == "events.jsonl"


def test_benchmark_plan_api_query_args_validate():
    args = benchmark_plan_args_from_query(
        {
            "suite": ["frontier"],
            "model": ["gemma4_12b"],
            "port": ["18080"],
            "results_dir": ["out"],
        }
    )

    assert args.suite == "frontier"
    assert args.model == "gemma4_12b"
    assert args.port == 18080
    assert args.results_dir == "out"

    try:
        benchmark_plan_args_from_query({"suite": ["unknown"]})
    except ValueError as exc:
        assert "unknown benchmark suite unknown" in str(exc)
    else:
        raise AssertionError("invalid benchmark suite should fail")

    try:
        benchmark_plan_args_from_query({"port": ["many"]})
    except ValueError as exc:
        assert "port must be an integer" in str(exc)
    else:
        raise AssertionError("invalid benchmark port should fail")


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


def test_benchmark_manifest_api_query_args_validate():
    args = benchmark_manifest_args_from_query(
        {
            "path": ["benchmark_results", "extra_results"],
            "suite": ["frontier"],
            "model": ["m"],
            "require_complete": ["true"],
        }
    )

    assert args.paths == ["benchmark_results", "extra_results"]
    assert args.suite == "frontier"
    assert args.model == "m"
    assert args.require_complete is True

    try:
        benchmark_manifest_args_from_query({})
    except ValueError as exc:
        assert "path query param required" in str(exc)
    else:
        raise AssertionError("benchmark-manifest API query should require path")

    try:
        benchmark_manifest_args_from_query({"path": ["benchmark_results"], "suite": ["unknown"]})
    except ValueError as exc:
        assert "unknown benchmark suite unknown" in str(exc)
    else:
        raise AssertionError("invalid benchmark manifest suite should fail")


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


def test_benchmark_audit_api_query_args_validate():
    args = benchmark_audit_args_from_query(
        {
            "path": ["results", "extra.jsonl"],
            "fail_empty_pct": ["1.5"],
            "fail_unknown_pct": ["2.5"],
            "fail_pass_only_pct": ["3.5"],
        }
    )

    assert args.paths == ["results", "extra.jsonl"]
    assert args.fail_empty_pct == 1.5
    assert args.fail_unknown_pct == 2.5
    assert args.fail_pass_only_pct == 3.5

    try:
        benchmark_audit_args_from_query({})
    except ValueError as exc:
        assert "path query param required" in str(exc)
    else:
        raise AssertionError("benchmark-audit API query should require path")

    try:
        benchmark_audit_args_from_query({"path": ["results"], "fail_empty_pct": ["many"]})
    except ValueError as exc:
        assert "audit thresholds must be numbers" in str(exc)
    else:
        raise AssertionError("invalid benchmark audit threshold should fail")


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
            "--pure-quant",
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
    assert "--metric ppl" in phases["ablate"]["command"]
    assert "--low-space" in phases["ablate"]["command"]
    assert "cerebellum resume" in phases["resume"]["command"]
    assert "--pure-quant" in phases["ablate"]["command"]
    assert "--tensor-type-file" in phases["build-final-gguf"]["command"]
    assert "--pure" not in phases["build-final-gguf"]["command"]
    assert "benchmark-run --suite frontier" in phases["benchmark"]["command"]
    assert "--execute --postprocess --require-complete" in phases["benchmark"]["command"]
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


def test_pipeline_plan_default_benchmark_suite_is_executable(tmp_path: Path):
    args = parse_args(["pipeline-plan", "--source-gguf", str(tmp_path / "m.gguf"), "--output-dir", str(tmp_path / "out")])

    plan = pipeline_plan(args)

    assert plan["benchmark_suite"] == "release-local"
    assert benchmark_run_plan(plan["benchmark_suite"], model="m", port=8084, results_dir=str(tmp_path / "bench"))["blocked"] is False


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

    execute_args = parse_args(["pipeline-run", "--manifest", "pipeline.json", "--execute"])
    assert execute_args.execute is True


def test_queue_command_parses():
    args = parse_args(
        [
            "queue",
            "--db",
            "queue.db",
            "--json",
            "add",
            "--kind",
            "pipeline",
            "--manifest",
            "pipeline.json",
            "--from-phase",
            "ablate",
            "--until-phase",
            "benchmark",
            "--priority",
            "10",
        ]
    )

    assert args.cmd == "queue"
    assert args.queue_cmd == "add"
    assert args.db == "queue.db"
    assert args.kind == "pipeline"
    assert args.from_phase == "ablate"
    assert args.until_phase == "benchmark"
    assert args.manifest == "pipeline.json"
    assert args.priority == 10
    assert args.json is True

    run_next = parse_args(["queue", "--db", "queue.db", "run-next", "--kind", "benchmark", "--execute"])
    assert run_next.queue_cmd == "run-next"
    assert run_next.kind == "benchmark"
    assert run_next.status == "queued"
    assert run_next.execute is True


def test_queue_add_list_get_pipeline_manifest(tmp_path: Path):
    db = tmp_path / "queue.db"
    manifest = tmp_path / "pipeline.json"
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "cerebellum",
                "run_dir": "run",
                "phases": [
                    {"name": "imatrix", "command": "cerebellum imatrix"},
                    {"name": "ablate", "command": "cerebellum run"},
                ],
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(["queue", "--db", str(db), "add", "--kind", "pipeline", "--manifest", str(manifest), "--priority", "5", "--label", "gemma"])

    job = queue_add_job(args)
    jobs = queue_list_jobs(db)
    fetched = queue_get_job(db, job["id"])
    markdown = queue_markdown({"schema": "cerebellum.queue.v1", "db": str(db), "jobs": jobs})

    assert job["kind"] == "pipeline"
    assert job["status"] == "queued"
    assert job["priority"] == 5
    assert job["payload"]["manifest"] == str(manifest)
    assert job["payload"]["phases"] == ["imatrix", "ablate"]
    assert fetched["id"] == job["id"]
    assert [row["id"] for row in jobs] == [job["id"]]
    assert "# Cerebellum Queue" in markdown


def test_queue_run_next_executes_pipeline_phase_slice(tmp_path: Path, capsys):
    db = tmp_path / "queue.db"
    manifest = tmp_path / "pipeline.json"
    first = tmp_path / "first.out"
    second = tmp_path / "second.out"
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "cerebellum",
                "phases": [
                    {"name": "first", "status": "planned", "command": f"{sys.executable} -c \"from pathlib import Path; Path('first.out').write_text('bad')\""},
                    {"name": "second", "status": "planned", "command": f"{sys.executable} -c \"from pathlib import Path; Path('second.out').write_text('ok')\""},
                ],
            }
        ),
        encoding="utf-8",
    )
    add_args = parse_args(
        [
            "queue",
            "--db",
            str(db),
            "--json",
            "add",
            "--kind",
            "pipeline",
            "--manifest",
            str(manifest),
            "--from-phase",
            "second",
        ]
    )
    queue_cmd(add_args)
    capsys.readouterr()
    run_args = parse_args(["queue", "--db", str(db), "--json", "run-next", "--kind", "pipeline", "--execute"])

    queue_cmd(run_args)
    payload = json.loads(capsys.readouterr().out)
    job = queue_get_job(db, payload["job"]["id"])

    assert payload["dry_run"] is False
    assert payload["result"]["returncode"] == 0
    assert payload["result"]["executions"][0]["phase"] == "second"
    assert job["status"] == "completed"
    assert not first.exists()
    assert second.read_text(encoding="utf-8") == "ok"


def test_queue_cmd_lists_json(tmp_path: Path, capsys):
    db = tmp_path / "queue.db"
    add_args = parse_args(["queue", "--db", str(db), "--json", "add", "--kind", "benchmark", "--command", "cerebellum benchmark-run --suite release"])
    queue_cmd(add_args)
    added = json.loads(capsys.readouterr().out)
    list_args = parse_args(["queue", "--db", str(db), "--json", "list", "--kind", "benchmark", "--status", "queued"])

    queue_cmd(list_args)
    listed = json.loads(capsys.readouterr().out)

    assert added["schema"] == "cerebellum.queue.v1"
    assert listed["jobs"][0]["payload"]["command"] == "cerebellum benchmark-run --suite release"


def test_queue_run_next_dry_run_does_not_change_status(tmp_path: Path, capsys):
    db = tmp_path / "queue.db"
    add_args = parse_args(["queue", "--db", str(db), "--json", "add", "--kind", "benchmark", "--command", "python -c 'print(1)'"])
    queue_cmd(add_args)
    capsys.readouterr()
    run_args = parse_args(["queue", "--db", str(db), "--json", "run-next", "--kind", "benchmark"])

    queue_cmd(run_args)
    payload = json.loads(capsys.readouterr().out)
    jobs = queue_list_jobs(db)

    assert payload["schema"] == "cerebellum.queue_run.v1"
    assert payload["dry_run"] is True
    assert payload["job"]["status"] == "queued"
    assert jobs[0]["status"] == "queued"


def test_queue_run_next_executes_command_job(tmp_path: Path, capsys):
    db = tmp_path / "queue.db"
    output = tmp_path / "queue.out"
    command = f"{sys.executable} -c \"from pathlib import Path; Path({str(output)!r}).write_text('ok')\""
    add_args = parse_args(["queue", "--db", str(db), "--json", "add", "--kind", "run", "--command", command])
    queue_cmd(add_args)
    capsys.readouterr()
    run_args = parse_args(["queue", "--db", str(db), "--json", "run-next", "--kind", "run", "--execute"])

    queue_cmd(run_args)
    payload = json.loads(capsys.readouterr().out)
    job = queue_get_job(db, payload["job"]["id"])

    assert payload["dry_run"] is False
    assert payload["result"]["returncode"] == 0
    assert job["status"] == "completed"
    assert job["result_json"]
    assert Path(job["log"]).is_file()
    assert output.read_text(encoding="utf-8") == "ok"


def test_queue_cancel_retry_and_log_tail(tmp_path: Path, capsys):
    db = tmp_path / "queue.db"
    fail_cmd = f"{sys.executable} -c \"print('line1'); print('line2'); raise SystemExit(7)\""
    add_args = parse_args(["queue", "--db", str(db), "--json", "add", "--kind", "run", "--command", fail_cmd])
    queue_cmd(add_args)
    capsys.readouterr()

    run_args = parse_args(["queue", "--db", str(db), "--json", "run-next", "--kind", "run", "--execute"])
    queue_cmd(run_args)
    failed_payload = json.loads(capsys.readouterr().out)
    failed = queue_get_job(db, failed_payload["job"]["id"], tail=1)

    assert failed["status"] == "failed"
    assert failed["result"]["returncode"] == 7
    assert failed["log_tail"] == "line2"

    canceled = queue_cancel_job(db, failed["id"], reason="bad command")
    assert canceled["status"] == "canceled"
    assert canceled["last_error"] == "bad command"

    retried = queue_retry_job(db, failed["id"], priority=3, notes="retry after fix")
    assert retried["status"] == "queued"
    assert retried["priority"] == 3
    assert retried["notes"] == "retry after fix"
    assert retried["last_error"] is None
    assert retried["result_json"] is None


def test_queue_cancel_and_retry_parse():
    cancel = parse_args(["queue", "--db", "queue.db", "cancel", "12", "--reason", "stale"])
    retry = parse_args(["queue", "--db", "queue.db", "retry", "12", "--priority", "5", "--notes", "resume"])
    get = parse_args(["queue", "--db", "queue.db", "get", "12", "--tail", "20"])

    assert cancel.queue_cmd == "cancel"
    assert cancel.id == 12
    assert cancel.reason == "stale"
    assert retry.queue_cmd == "retry"
    assert retry.priority == 5
    assert retry.notes == "resume"
    assert get.tail == 20


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
                    {"name": "benchmark", "status": "planned", "command": "cerebellum benchmark-run --execute", "outputs": ["benchmark_results"]},
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


def test_pipeline_run_cmd_prints_dry_run_and_executes_phases(tmp_path: Path, capsys):
    manifest = tmp_path / "pipeline.json"
    output = tmp_path / "phase.out"
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "cerebellum",
                "phases": [
                    {"name": "imatrix", "status": "planned", "command": "cerebellum imatrix"},
                    {
                        "name": "local-smoke",
                        "status": "planned",
                        "command": f"{sys.executable} -c \"from pathlib import Path; Path('phase.out').write_text('ok')\"",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(["pipeline-run", "--manifest", str(manifest)])

    pipeline_run_cmd(args)
    assert "# Cerebellum Pipeline Run" in capsys.readouterr().out

    execute_args = parse_args(["pipeline-run", "--manifest", str(manifest), "--from-phase", "local-smoke", "--execute", "--json"])
    pipeline_run_cmd(execute_args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is False
    assert output.read_text(encoding="utf-8") == "ok"
    assert Path(payload["event_log"]).is_file()
    assert Path(payload["executions"][0]["log"]).is_file()


def test_pipeline_run_execute_stops_on_failure(tmp_path: Path):
    manifest = tmp_path / "pipeline.json"
    skipped = tmp_path / "skipped.out"
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "cerebellum",
                "phases": [
                    {"name": "fail", "status": "planned", "command": f"{sys.executable} -c \"raise SystemExit(7)\""},
                    {"name": "skip", "status": "planned", "command": f"{sys.executable} -c \"from pathlib import Path; Path('skipped.out').write_text('bad')\""},
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = pipeline_run_plan(manifest)
    result = pipeline_run_execute(plan)

    assert result["dry_run"] is False
    assert result["blocked"] is True
    assert result["blockers"] == [{"phase": "fail", "reason": "command exited 7"}]
    assert [row["phase"] for row in result["executions"]] == ["fail"]
    assert not skipped.exists()


def test_pipeline_run_execute_benchmark_phase_can_write_sidecars(tmp_path: Path):
    manifest = tmp_path / "pipeline.json"
    command = (
        f"{sys.executable} -c "
        "\"from pathlib import Path; "
        "p=Path('benchmark_results/postprocess'); "
        "p.mkdir(parents=True); "
        "Path('benchmark_results/unit_results.json').write_text('{\\\"accuracy\\\":1.0}'); "
        "Path('benchmark_results/postprocess/benchmark_manifest.json').write_text('{\\\"ready\\\":true}')\""
    )
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "cerebellum",
                "phases": [
                    {
                        "name": "benchmark",
                        "status": "planned",
                        "command": command,
                        "outputs": ["benchmark_results"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = pipeline_run_execute(pipeline_run_plan(manifest, from_phase="benchmark"))

    assert result["dry_run"] is False
    assert result["blocked"] is False
    assert (tmp_path / "benchmark_results" / "unit_results.json").is_file()
    assert (tmp_path / "benchmark_results" / "postprocess" / "benchmark_manifest.json").is_file()


def test_pipeline_status_command_parses():
    args = parse_args(["pipeline-status", "--manifest", "pipeline.json", "--events", "events.jsonl", "--json"])

    assert args.cmd == "pipeline-status"
    assert args.manifest == "pipeline.json"
    assert args.events == "events.jsonl"
    assert args.json is True


def test_pipeline_status_reports_complete_run(tmp_path: Path):
    manifest = tmp_path / "pipeline.json"
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "cerebellum",
                "run_dir": "run",
                "phases": [
                    {"name": "one", "status": "planned", "command": f"{sys.executable} -c \"print('one')\""},
                    {"name": "two", "status": "planned", "command": f"{sys.executable} -c \"print('two')\""},
                ],
            }
        ),
        encoding="utf-8",
    )
    pipeline_run_execute(pipeline_run_plan(manifest))

    status = pipeline_status(manifest)
    markdown = pipeline_status_markdown(status)

    assert status["schema"] == "cerebellum.pipeline_status.v1"
    assert status["status"] == "complete"
    assert status["completed_phases"] == 2
    assert status["resume_command"] is None
    assert [row["status"] for row in status["phases"]] == ["complete", "complete"]
    assert "# Cerebellum Pipeline Status" in markdown


def test_pipeline_status_reports_failed_resume_point(tmp_path: Path):
    manifest = tmp_path / "pipeline.json"
    manifest.write_text(
        json.dumps(
            {
                "pipeline": "cerebellum",
                "phases": [
                    {"name": "fail", "status": "planned", "command": f"{sys.executable} -c \"raise SystemExit(7)\""},
                    {"name": "skip", "status": "planned", "command": f"{sys.executable} -c \"print('skip')\""},
                ],
            }
        ),
        encoding="utf-8",
    )
    pipeline_run_execute(pipeline_run_plan(manifest))

    status = pipeline_status(manifest)

    assert status["status"] == "failed"
    assert status["failed_phase"] == "fail"
    assert status["resume_phase"] == "fail"
    assert "--from-phase fail --execute" in status["resume_command"]
    assert [row["status"] for row in status["phases"]] == ["failed", "pending"]


def test_resume_preserves_keep_measured_candidates(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_gguf": str(tmp_path / "m.gguf"),
                "corpus": str(tmp_path / "wiki.txt"),
                "ppl_profile": "wiki",
                "run_id": "unit",
                "levels": ["q2_K"],
                "prune_measured_candidates": False,
            }
        ),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(hillstep, "run_from_namespace", lambda ns: captured.update(vars(ns)))

    resume_cmd(parse_args(["resume", str(run_dir)]))

    assert captured["metric"] == "ppl"
    assert captured["prune_measured_candidates"] is False


def test_resume_can_override_saved_low_space_mode(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_gguf": str(tmp_path / "m.gguf"),
                "corpus": str(tmp_path / "wiki.txt"),
                "run_id": "unit",
                "levels": ["q3_K", "q2_K"],
                "low_space": True,
                "serial_candidates": True,
                "commit_locks": False,
            }
        ),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(hillstep, "run_from_namespace", lambda ns: captured.update(vars(ns)))

    resume_cmd(parse_args(["resume", str(run_dir), "--no-low-space"]))

    assert captured["low_space"] is False
    assert captured["serial_candidates"] is False
    assert captured["commit_locks"] is False


def test_pipeline_status_api_query_args_validate():
    args = pipeline_status_args_from_query({"manifest": ["pipeline.json"], "events": ["events.jsonl"]})

    assert args.manifest == "pipeline.json"
    assert args.events == "events.jsonl"

    try:
        pipeline_status_args_from_query({})
    except ValueError as exc:
        assert "manifest query param required" in str(exc)
    else:
        raise AssertionError("pipeline-status API query should require manifest")


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
    assert plan["benchmark_suite"] == "release-local"
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
    assert plan["ablation_metric"] == "humaneval"
    assert plan["benchmark_suite"] == "release-local"
    assert plan["task_profile_detail"]["ablation_metric"] == "humaneval"
    assert plan["task_profile_detail"]["metrics"] == ["humaneval", "evalplus", "livecodebench_v6"]
    assert plan["final_gguf"].endswith("model-x-code-cerebellum.gguf")
    assert "--profile code" in phases["ablate"]["command"]
    assert "--metric humaneval" in phases["ablate"]["command"]
    assert "benchmark-run --suite release-local --model model-x-code" in phases["benchmark"]["command"]
    assert "--execute --postprocess --require-complete" in phases["benchmark"]["command"]


def test_pipeline_plan_task_profile_allows_metric_override(tmp_path: Path):
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
            "--metric",
            "ppl",
        ]
    )

    plan = pipeline_plan(args)
    phases = {row["name"]: row for row in plan["phases"]}

    assert plan["task_profile"] == "code"
    assert plan["ablation_metric"] == "ppl"
    assert "--metric ppl" in phases["ablate"]["command"]


def test_pipeline_plan_legacy_gated_explains_og_workflow(tmp_path: Path):
    source = tmp_path / "source-f16.gguf"
    source.write_bytes(b"gguf")
    args = parse_args(
        [
            "pipeline-plan",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--task-profile",
            "legacy-gated",
        ]
    )

    plan = pipeline_plan(args)
    workflow = plan["legacy_gated_workflow"]

    assert plan["task_profile"] == "legacy-gated"
    assert workflow["hillstep_role"].startswith("targeted refinement")
    assert "quantization source only" in workflow["source_role"]
    assert [step["name"] for step in workflow["steps"]][:3] == ["scan", "lower-quant-baseline", "survivability-scan"]
    assert "quant_ppl_overlap" in workflow["orchestration"]


def test_run_rejects_non_ppl_metric_before_launch(tmp_path: Path):
    args = parse_args(
        [
            "run",
            "--source-gguf",
            str(tmp_path / "m.gguf"),
            "--metric",
            "humaneval",
        ]
    )

    try:
        run_from_namespace(args)
    except SystemExit as exc:
        assert "current Cerebellum run scoring supports only 'ppl'" in str(exc)
    else:
        raise AssertionError("run --metric humaneval should fail until task scorers exist")


def test_run_no_commit_locks_records_scan_without_advancing_baseline(tmp_path: Path, monkeypatch):
    source = tmp_path / "model.gguf"
    corpus = tmp_path / "wiki.test.raw"
    tensor_file = tmp_path / "tensors.txt"
    run_dir = tmp_path / "run"
    source.write_bytes(b"gguf")
    corpus.write_text("hello\n", encoding="utf-8")
    tensor_file.write_text("blk.0.ffn_down.weight\n", encoding="utf-8")
    quant_cmds = []

    def fake_run_external(cmd, _timeout, _distrobox, heartbeat=None):
        if heartbeat:
            heartbeat(1.0, 123)
        if "--model" in cmd:
            model = Path(cmd[cmd.index("--model") + 1])
            ppl = 100.0 if model.name == "current_baseline.gguf" else 90.0
            return 0, f"Final estimate: PPL = {ppl} +/- 1.0\n", 1.0
        quant_cmds.append(cmd)
        outfile = Path(cmd[-2])
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"candidate")
        return 0, "quant ok\n", 1.0

    monkeypatch.setattr(hillstep, "run_external", fake_run_external)
    args = parse_args(
        [
            "run",
            "--source-gguf",
            str(source),
            "--corpus",
            str(corpus),
            "--run-dir",
            str(run_dir),
            "--tensor-file",
            str(tensor_file),
            "--levels",
            "q3_K",
            "--no-commit-locks",
            "--pure-quant",
            "--min-free-gb",
            "0",
            "--hard-free-floor-gb",
            "0",
            "--no-keep-winners",
            "--plain",
            "--no-color",
        ]
    )

    run_from_namespace(args)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    events = (run_dir / "cerebellum_events.jsonl").read_text(encoding="utf-8")

    assert manifest["commit_locks"] is False
    assert manifest["pure_quant"] is True
    assert manifest["pure_quant_effective"] is False
    assert all("--pure" not in cmd for cmd in quant_cmds)
    assert state["run_status"] == "complete"
    assert state["locked"] == {}
    assert state["current_ppl"] == 100.0
    assert state["start_quant_size_bytes"] == len(b"candidate")
    assert state["tested"][0]["winner"] == "q3_K"
    assert state["tested"][0]["committed"] is False
    assert "tensor_scanned" in events
    assert "tensor_locked" not in events


def test_run_base_map_seeds_quant_maps_without_marking_tensors_complete(tmp_path: Path, monkeypatch):
    source = tmp_path / "model.gguf"
    corpus = tmp_path / "wiki.test.raw"
    tensor_file = tmp_path / "targets.txt"
    base_map = tmp_path / "final_group_overrides.txt"
    run_dir = tmp_path / "targeted"
    source.write_bytes(b"gguf")
    corpus.write_text("hello\n", encoding="utf-8")
    tensor_file.write_text("blk.0.attn_v.weight\n", encoding="utf-8")
    base_map.write_text(
        "\n".join(
            [
                r"^blk\.0\.attn_v\.weight$=q2_K",
                r"^blk\.0\.ffn_down\.weight$=q2_K",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    type_maps: list[str] = []

    def fake_run_external(cmd, _timeout, _distrobox, heartbeat=None):
        if heartbeat:
            heartbeat(1.0, 123)
        if "--model" in cmd:
            model = Path(cmd[cmd.index("--model") + 1])
            if model.name == "current_baseline.gguf":
                ppl = 100.0
            elif model.name.startswith("00-q3_K"):
                ppl = 95.0
            else:
                ppl = 110.0
            return 0, f"Final estimate: PPL = {ppl} +/- 1.0\n", 1.0
        type_file = Path(cmd[cmd.index("--tensor-type-file") + 1])
        type_maps.append(type_file.read_text(encoding="utf-8"))
        outfile = Path(cmd[-2])
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"candidate")
        return 0, "quant ok\n", 1.0

    monkeypatch.setattr(hillstep, "run_external", fake_run_external)
    args = parse_args(
        [
            "run",
            "--source-gguf",
            str(source),
            "--corpus",
            str(corpus),
            "--run-dir",
            str(run_dir),
            "--tensor-file",
            str(tensor_file),
            "--base-map",
            str(base_map),
            "--levels",
            "q3_K,q4_K",
            "--min-free-gb",
            "0",
            "--hard-free-floor-gb",
            "0",
            "--plain",
            "--no-color",
        ]
    )

    run_from_namespace(args)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    final_types = (run_dir / "cerebellum_best_tensor_types.txt").read_text(encoding="utf-8")

    assert manifest["base_map"] == str(base_map)
    assert manifest["base_map_count"] == 2
    assert state["locked"] == {"blk.0.attn_v.weight": "q3_K"}
    assert state["tested"][0]["winner"] == "q3_K"
    assert type_maps[0].count("Q2_K") == 2
    assert r"^blk\.0\.attn_v\.weight$=Q2_K" in type_maps[0]
    assert r"^blk\.0\.attn_v\.weight$=q3_K" in type_maps[1]
    assert r"^blk\.0\.ffn_down\.weight$=Q2_K" in final_types


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
    assert plan["benchmark_suite"] == "release-local"
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
    assert "plan-space --source-gguf" in " ".join(dry_run["preflight_commands"])
    assert "--scratch-candidates" in " ".join(dry_run["preflight_commands"])
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


def test_cpu_offload_smoke_command_parses():
    args = parse_args(["cpu-offload-smoke", "--source-gguf", "glm.gguf", "--output-dir", "out", "--model-name", "GLM 5.1", "--skip-inspect", "--json"])

    assert args.cmd == "cpu-offload-smoke"
    assert args.source_gguf == "glm.gguf"
    assert args.output_dir == "out"
    assert args.model_name == "GLM 5.1"
    assert args.skip_inspect is True
    assert args.json is True


def test_cpu_offload_smoke_validates_plan_without_full_model_load(tmp_path: Path, capsys):
    source = tmp_path / "glm-5.1-f16.gguf"
    source.write_bytes(b"gguf" * 1024)
    args = parse_args(
        [
            "cpu-offload-smoke",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(tmp_path / "glm51-cpu"),
            "--model-name",
            "GLM 5.1",
            "--skip-inspect",
            "--json",
        ]
    )

    payload = cpu_offload_smoke_payload(args)
    markdown = cpu_offload_smoke_markdown(payload)
    cpu_offload_smoke_cmd(args)
    cmd_payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == "cerebellum.cpu_offload_smoke.v1"
    assert payload["blocked"] is False
    assert payload["full_model_ram_load_required"] is False
    assert payload["pipeline"]["task_profile"] == "cpu-offload"
    assert payload["pipeline"]["cpu_offload_plan"]["streaming"]["full_model_ram_load_required"] is False
    assert payload["space"]["source_gguf"] == str(source)
    assert payload["inspect"]["skipped"] is True
    assert any(row["name"] == "glm-layout-unverified" for row in payload["hazards"])
    assert "CPU-Offload Smoke" in markdown
    assert cmd_payload["schema"] == "cerebellum.cpu_offload_smoke.v1"


def test_cpu_offload_smoke_api_query_args_do_not_create_output_dir(tmp_path: Path):
    source = tmp_path / "glm.gguf"
    source.write_bytes(b"gguf" * 1024)
    output_dir = tmp_path / "missing-output"
    args = cpu_offload_smoke_args_from_query(
        {
            "source_gguf": [str(source)],
            "output_dir": [str(output_dir)],
            "model_name": ["GLM 5.1"],
            "skip_inspect": ["true"],
            "margin_gb": ["1.5"],
            "benchmark_port": ["18080"],
        }
    )

    payload = cpu_offload_smoke_payload(args)

    assert args.create_dirs is False
    assert payload["schema"] == "cerebellum.cpu_offload_smoke.v1"
    assert payload["blocked"] is False
    assert not output_dir.exists()


def test_cpu_offload_smoke_api_query_args_validate_required_fields():
    try:
        cpu_offload_smoke_args_from_query({"source_gguf": ["model.gguf"]})
    except ValueError as exc:
        assert "output_dir query param required" in str(exc)
    else:
        raise AssertionError("cpu-offload-smoke API query should require output_dir")

    try:
        cpu_offload_smoke_args_from_query({"source_gguf": ["model.gguf"], "output_dir": ["out"], "benchmark_port": ["many"]})
    except ValueError as exc:
        assert "benchmark_port must be an integer" in str(exc)
    else:
        raise AssertionError("cpu-offload-smoke API query should validate benchmark_port")


def test_cpu_offload_smoke_can_block_on_inspect_failure(tmp_path: Path):
    source = tmp_path / "not-a-real.gguf"
    source.write_bytes(b"not gguf")
    args = parse_args(
        [
            "cpu-offload-smoke",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--require-inspect",
            "--json",
        ]
    )

    payload = cpu_offload_smoke_payload(args)

    assert payload["blocked"] is True
    assert any(row["name"] == "inspect_gguf_types" for row in payload["blockers"])


def test_cpu_offload_build_plan_command_parses():
    args = parse_args(
        [
            "cpu-offload-build-plan",
            "--source-gguf",
            "glm.gguf",
            "--output-dir",
            "out",
            "--model-name",
            "GLM 5.1",
            "--scratch-root",
            "/tmp/scratch",
            "--skip-inspect",
            "--write",
            "out/build.json",
            "--json",
        ]
    )

    assert args.cmd == "cpu-offload-build-plan"
    assert args.source_gguf == "glm.gguf"
    assert args.output_dir == "out"
    assert args.model_name == "GLM 5.1"
    assert args.scratch_root == "/tmp/scratch"
    assert args.skip_inspect is True
    assert args.write == "out/build.json"
    assert args.json is True


def test_cpu_offload_build_plan_emits_operator_manifest_without_full_model_load(tmp_path: Path):
    source = tmp_path / "glm-5.1-f16.gguf"
    source.write_bytes(b"gguf" * 1024)
    out = tmp_path / "glm51-cpu"
    args = parse_args(
        [
            "cpu-offload-build-plan",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(out),
            "--model-name",
            "GLM 5.1",
            "--skip-inspect",
        ]
    )

    payload = cpu_offload_build_plan_payload(args)
    markdown = cpu_offload_build_plan_markdown(payload)

    assert payload["schema"] == "cerebellum.cpu_offload_build_plan.v1"
    assert payload["ready"] is True
    assert payload["full_model_ram_load_required"] is False
    assert payload["manifest"] == str(out / "cpu_offload_pipeline.json")
    assert payload["pipeline"]["task_profile"] == "cpu-offload"
    assert {row["name"] for row in payload["pipeline"]["phases"]} >= {"imatrix", "ablate", "build-final-gguf"}
    assert {row["phase"] for row in payload["artifact_flow"]} >= {"stream-imatrix", "build-final-gguf"}
    assert payload["commands"]["prepare_output_dir"] == f"mkdir -p {out}"
    assert "queue run-next --execute" in payload["commands"]["run_next"]
    assert "release-gate" in payload["commands"]["release_gate"]
    assert "CPU-Offload Build Plan" in markdown
    assert not out.exists()


def test_cpu_offload_build_plan_cmd_writes_requested_json_only(tmp_path: Path, capsys):
    source = tmp_path / "glm.gguf"
    source.write_bytes(b"gguf" * 1024)
    out = tmp_path / "out"
    report = tmp_path / "build_plan.json"
    args = parse_args(
        [
            "cpu-offload-build-plan",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(out),
            "--skip-inspect",
            "--write",
            str(report),
            "--json",
        ]
    )

    cpu_offload_build_plan_cmd(args)
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(report.read_text(encoding="utf-8"))

    assert printed["schema"] == "cerebellum.cpu_offload_build_plan.v1"
    assert written["manifest"] == str(out / "cpu_offload_pipeline.json")
    assert not out.exists()


def test_cpu_offload_build_plan_cmd_creates_requested_write_parent(tmp_path: Path):
    source = tmp_path / "glm.gguf"
    source.write_bytes(b"gguf" * 1024)
    out = tmp_path / "out"
    report = out / "build_plan.json"
    args = parse_args(
        [
            "cpu-offload-build-plan",
            "--source-gguf",
            str(source),
            "--output-dir",
            str(out),
            "--skip-inspect",
            "--write",
            str(report),
        ]
    )

    cpu_offload_build_plan_cmd(args)

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["schema"] == "cerebellum.cpu_offload_build_plan.v1"
    assert report.exists()


def test_task_profiles_command_outputs_catalog(capsys):
    markdown = task_profiles_markdown()
    args = parse_args(["task-profiles", "--json"])

    task_profiles_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert "code" in data["profiles"]
    assert data["profiles"]["tools"]["ppl_profile"] == "agentic"
    assert data["profiles"]["cpu-offload"]["low_space_default"] is True
    assert "| code | code | humaneval | release-local | humaneval, evalplus, livecodebench_v6 |" in markdown
    assert "| cpu-offload | all-around | ppl | release-local | ppl, speed, score_per_gib, cpu_tok_s, gpu_offload_layers |" in markdown


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


def test_eta_detail_values_uses_prior_candidate_timing_for_job_tensor_phase(tmp_path: Path):
    nowish = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(timespec="milliseconds")
    state = {"locked": {"blk.0.attn_q.weight": "q4_K"}, "tested": [{"tensor": "blk.0.attn_q.weight"}]}
    manifest = {"levels": ["q3_K", "q2_K"]}
    events = [{"event": "tensor_start", "tensor": "blk.0.ffn_down.weight", "timestamp_utc": nowish}]
    candidates = [
        {"tensor": "blk.0.ffn_down.weight", "level": "q3_K", "quant_seconds": 100.0, "ppl_seconds": 200.0},
        {"tensor": "blk.0.ffn_down.weight", "level": "q2_K", "quant_seconds": 100.0, "ppl_seconds": 200.0},
    ]
    active = {"event": "ppl_start", "tensor": "blk.0.ffn_down.weight", "level": "q2_K"}

    eta = eta_detail_values(tmp_path, state, manifest, events, candidates, active, active_age=50.0, total=4, flow={})

    assert eta["source"] == "low from 2 prior candidate timings"
    assert eta["job"]["label"] == "ppl q2_K"
    assert eta["job"]["estimate"] == "3m20s"
    assert eta["tensor"]["estimate"] == "8m20s"
    assert eta["phase"]["tensors_remaining"] == 3
    assert eta["phase"]["remaining"] != "-"

    overdue = eta_detail_values(tmp_path, state, manifest, events, candidates, active, active_age=250.0, total=4, flow={})
    assert overdue["job"]["remaining"].startswith("overdue 50.0s")


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
            "dynamic_status": "demoted",
            "baseline_bits": 4.0,
            "candidate_bits": 3.0,
            "bits_delta": -1.0,
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
            "dynamic_status": "demoted",
            "baseline_bits": 4.0,
            "candidate_bits": 3.0,
            "bits_delta": -1.0,
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
            "dynamic_status": "promoted",
            "baseline_bits": 4.0,
            "candidate_bits": 5.0,
            "bits_delta": 1.0,
            "quantizable": True,
        },
    ]
    assert "candidate: `dynamic`" in markdown
    assert "| Q4_K | 3 | 1 | -2 |" in markdown
    assert "## Dynamic Quant Profile" in markdown
    assert "| promoted | 1 |" in markdown
    assert "| demoted | 1 |" in markdown
    assert "| blk.0.attn_q.weight | Q4_K | Q3_K | demoted | -1.00 |" in markdown
    assert "| blk.0.ffn_down.weight | Q4_K | Q5_K | promoted | +1.00 |" in markdown
    assert "## Reference Map Mismatches" in markdown
    assert "| blk.0.attn_q.weight | attn_q | Q4_K | Q3_K | Q5_K | candidate_diverges_reference |" in markdown


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
        "cerebellum.hillstep.process_rows_for_run",
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
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])

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
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])

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
    monkeypatch.setattr("cerebellum.hillstep.run_is_live", lambda _run_dir: True)

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
    monkeypatch.setattr("cerebellum.hillstep.run_is_live", lambda _run_dir: False)

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
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("cerebellum.hillstep.gpu_rows", lambda: [])
    monkeypatch.setattr("cerebellum.hillstep.os.system", lambda _cmd: 0)

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
    assert "progress" in output
    assert "redacted" not in output.lower()
    assert "private" not in output.lower()
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
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("cerebellum.hillstep.gpu_rows", lambda: [])
    monkeypatch.setattr("cerebellum.hillstep.os.system", lambda _cmd: 0)

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


def test_private_watch_event_strip_defaults_to_five_rows(tmp_path: Path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_status": "running",
                "model_family": "gemma-4",
                "model_name": "gemma-4-12b-it",
                "locked": {},
                "tested": [],
                "totals": {"quant_seconds": 0.0, "ppl_seconds": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text('{"run_id":"run","ppl_profile":"wiki"}', encoding="utf-8")
    (run_dir / "cerebellum_events.jsonl").write_text(
        "\n".join(json.dumps({"event": f"strip_event_{idx}"}) for idx in range(7)) + "\n",
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("cerebellum.hillstep.gpu_rows", lambda: [])
    monkeypatch.setattr("cerebellum.hillstep.os.system", lambda _cmd: 0)

    grid_watch_cmd(
        types.SimpleNamespace(
            run_dir=str(run_dir),
            stall_warn_seconds=300.0,
            stall_fail_seconds=900.0,
            measurements_limit=8,
            events_limit=5,
            once=True,
            public=False,
            plain=True,
            no_color=True,
        )
    )

    output = capsys.readouterr().out
    assert "strip_event_0" not in output
    assert "strip_event_1" not in output
    assert "strip_event_2" in output
    assert "strip_event_6" in output


def test_private_watch_shows_group_scan_verdicts_without_locks(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "classic"
    run_dir = root / "forward" / "ffn-gate"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "schema": "cerebellum.group_scan_state.v1",
                "run_status": "complete",
                "model_family": "gemma-4",
                "model_name": "gemma-4-12b-it",
                "current_ppl": 5433.7967,
                "baseline_ppl": 5433.7967,
                "current_tensor": "ffn-gate",
                "current_level": "q2_K",
                "locked": {},
                "tested": [{"tensor": "ffn-gate", "winner": "q2_K", "ppl": 53420.9426, "delta": 47987.1459}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "cerebellum.group_scan.v1",
                "run_id": "run",
                "ppl_profile": "wiki",
                "group_name": "ffn-gate",
                "commit_locks": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cerebellum_events.jsonl").write_text('{"event":"ppl_finish","level":"q2_K","tensor":"ffn-gate"}\n', encoding="utf-8")
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        json.dumps(
            {
                "level": "q2_K",
                "ppl": 53420.9426,
                "delta": 47987.1459,
                "size_bytes": 7500000000,
                "tensor": "ffn-gate",
                "tensor_count": 48,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("cerebellum.hillstep.gpu_rows", lambda: [])
    monkeypatch.setattr("cerebellum.hillstep.os.system", lambda _cmd: 0)

    grid_watch_cmd(
        types.SimpleNamespace(
            run_dir=str(run_dir),
            stall_warn_seconds=300.0,
            stall_fail_seconds=900.0,
            measurements_limit=0,
            events_limit=0,
            once=True,
            public=False,
            plain=True,
            no_color=True,
        )
    )

    output = capsys.readouterr().out
    assert "RECENT MEASUREMENTS" not in output
    assert "FORWARD ABLATION" in output
    assert "group" in output
    assert "quant" in output
    assert "verdict" in output
    assert "ffn-gate" in output
    assert "nah, do not smash this ffn-gate" in output
    assert "GROUP VERDICTS" in output
    assert "LOCKED LAYER MAP" not in output
    assert "SCAN FINDINGS" not in output


def test_private_watch_shows_sparse_replay_as_measurements_not_group_verdicts(tmp_path: Path, monkeypatch, capsys):
    run_dir = tmp_path / "sparse-replay"
    run_dir.mkdir()
    tensor = "blk.27.ffn_down.weight"
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "schema": "cerebellum.sparse_replay_state.v1",
                "run_status": "complete",
                "model_family": "qwen3",
                "model_name": "qwen3-0.6b",
                "current_ppl": 19.72,
                "baseline_ppl": 19.73,
                "locked": {},
                "tested": [{"tensor": tensor, "winner": "q2_K", "ppl": 21.6, "delta": 1.87}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "cerebellum.sparse_replay.v1",
                "run_id": "sparse",
                "ppl_profile": "wiki",
                "commit_locks": False,
                "measurement_mode": "qwen36-27b-v4 sparse replay",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cerebellum_events.jsonl").write_text(
        json.dumps({"event": "run_start", "tensors": 1}) + "\n" + json.dumps({"event": "ppl_finish", "level": "q2_K", "tensor": tensor}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        json.dumps({"event": "candidate", "level": "q2_K", "ppl": 21.6, "delta": 1.87, "size_bytes": 447000000, "tensor": tensor}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("cerebellum.hillstep.gpu_rows", lambda: [])
    monkeypatch.setattr("cerebellum.hillstep.os.system", lambda _cmd: 0)

    grid_watch_cmd(
        types.SimpleNamespace(
            run_dir=str(run_dir),
            stall_warn_seconds=300.0,
            stall_fail_seconds=900.0,
            measurements_limit=0,
            events_limit=0,
            once=True,
            public=False,
            plain=True,
            no_color=True,
        )
    )

    output = capsys.readouterr().out
    assert "RECENT MEASUREMENTS" in output
    assert "GROUP VERDICTS" not in output
    assert "blk.27.ffn_down.weight" in output
    assert "q2_K" in output


def test_candidate_measurement_verdict_uses_per_tensor_quality_labels():
    rows = [
        {"tensor": "blk.0.attn_k.weight", "level": "q3_K", "ppl": 2283.6021, "delta": -12.6008},
        {"tensor": "blk.0.attn_k.weight", "level": "q4_K", "ppl": 2285.9047, "delta": -10.2982},
        {"tensor": "blk.0.attn_k.weight", "level": "q5_K", "ppl": 2270.9098, "delta": -25.2931},
        {"tensor": "blk.0.attn_k.weight", "level": "q6_K", "ppl": 2300.5604, "delta": 4.3575},
    ]

    assert hillstep.candidate_measurement_verdict(rows[0], rows)[0] == "better"
    assert hillstep.candidate_measurement_verdict(rows[1], rows)[0] == "better"
    assert hillstep.candidate_measurement_verdict(rows[2], rows)[0] == "best"
    assert hillstep.candidate_measurement_verdict(rows[3], rows)[0] == "worse"


def test_watch_model_includes_in_progress_candidate_rows(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_status": "running",
                "locked": {"blk.0.attn_k.weight": "q5_K"},
                "tested": [{"tensor": "blk.0.attn_k.weight", "winner": "q5_K"}],
                "current_ppl": 2270.9098,
                "baseline_path": str(run_dir / "artifacts" / "current_baseline.gguf"),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"model_family": "gemma-4", "model_name": "gemma-4-12b-it", "levels": ["q3_K", "q4_K"]}),
        encoding="utf-8",
    )
    (run_dir / "cerebellum_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "run_start", "tensors": 65}),
                json.dumps({"event": "tensor_start", "tensor": "blk.0.attn_output.weight", "index": 2, "total": 65}),
                json.dumps({"event": "quant_finish", "tensor": "blk.0.attn_output.weight", "level": "q3_K", "returncode": 0, "size_bytes": 8147999968}),
                json.dumps({"event": "ppl_start", "tensor": "blk.0.attn_output.weight", "level": "q3_K"}),
                json.dumps({"event": "quant_start", "tensor": "blk.0.attn_output.weight", "level": "q4_K", "size_bytes": 3200}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "cerebellum_candidates.jsonl").write_text(
        json.dumps({"tensor": "blk.0.attn_k.weight", "level": "q5_K", "ppl": 2270.9098, "delta": -25.2931, "status": "done"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("cerebellum.hillstep.process_rows_for_run", lambda _run_dir: [])
    monkeypatch.setattr("cerebellum.hillstep.gpu_rows", lambda: [])

    model = build_watch_model(run_dir)
    pending = [row for row in model["candidates"] if row.get("tensor") == "blk.0.attn_output.weight"]

    assert [(row["level"], row["status"]) for row in pending] == [("q3_K", "ppl"), ("q4_K", "quantizing")]
    assert hillstep.candidate_measurement_verdict(pending[0], model["candidates"])[0] == "ppl running"


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


def test_public_history_audit_command_parses():
    args = parse_args(["public-history-audit", "--root", ".", "--ref", "origin/main", "--json"])

    assert args.cmd == "public-history-audit"
    assert args.root == "."
    assert args.ref == ["origin/main"]
    assert args.json is True


def test_public_history_audit_plans_filter_repo_for_risky_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        hillstep,
        "git_history_paths",
        lambda root, refs=None: ["README.md", "scripts/build.sh", "cerebellum-dev/DEVLOG.md", "tensor_types.txt"],
    )

    report = public_history_audit(tmp_path, refs=["origin/main"])
    markdown = public_history_audit_markdown(report)

    assert report["schema"] == "cerebellum.public_history_audit.v1"
    assert report["refs"] == ["origin/main"]
    assert report["blocked"] is True
    assert report["filter_paths"] == ["cerebellum-dev/DEVLOG.md", "scripts/build.sh", "tensor_types.txt"]
    assert report["filter_repo_argv"][:4] == ["git", "filter-repo", "--force", "--invert-paths"]
    assert "--path scripts/build.sh" in report["filter_repo_command"]
    assert "Filter-Repo Plan" in markdown


def test_public_history_audit_passes_clean_history(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hillstep, "git_history_paths", lambda root, refs=None: ["README.md", "docs/public_release_scope.md"])

    report = public_history_audit(tmp_path)

    assert report["blocked"] is False
    assert report["filter_repo_argv"] == []
    assert public_history_audit_markdown(report).startswith("Public history audit passed")


def test_public_history_audit_cmd_exits_on_findings(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(hillstep, "git_history_paths", lambda root, refs=None: ["scripts/factory.sh"])
    args = parse_args(["public-history-audit", "--root", str(tmp_path)])

    try:
        hillstep.public_history_audit_cmd(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("public history audit should exit non-zero when blocked")

    assert "Filter-Repo Plan" in capsys.readouterr().out


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


def test_release_gate_command_parses():
    args = parse_args(
        [
            "release-gate",
            "README.md",
            "--remote",
            "origin",
            "--benchmark-results",
            "benchmark_results",
            "--suite",
            "release",
            "--model",
            "Cerebellum",
            "--require-benchmarks",
            "--json",
            "--max-bytes",
            "100",
        ]
    )

    assert args.cmd == "release-gate"
    assert args.paths == ["README.md"]
    assert args.remote == "origin"
    assert args.benchmark_results == ["benchmark_results"]
    assert args.require_benchmarks is True
    assert args.json is True
    assert args.max_bytes == 100


def test_release_gate_origin_blocks_private_artifacts(tmp_path: Path):
    safe = tmp_path / "README.md"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    risky = scripts / "factory.sh"
    safe.write_text("public release notes\n", encoding="utf-8")
    risky.write_text("HF_TOKEN=secret\n", encoding="utf-8")

    report = release_gate([str(safe), str(risky)], remote="origin")
    markdown = release_gate_markdown(report)

    assert report["schema"] == "cerebellum.release_gate.v1"
    assert report["visibility"] == "public"
    assert report["ready"] is False
    assert {item["source"] for item in report["blockers"]} >= {"public_audit", "path_policy"}
    assert any(item["reason"] == "not in public export allowlist" for item in report["non_allowlisted"])
    assert "Release gate blocked" in markdown


def test_release_gate_dev_warns_without_blocking_private_artifacts(tmp_path: Path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    risky = scripts / "factory.sh"
    risky.write_text("HF_TOKEN=secret\n", encoding="utf-8")

    report = release_gate([str(risky)], remote="dev")

    assert report["visibility"] == "private"
    assert report["mode"] == "advisory"
    assert report["ready"] is True
    assert report["blockers"] == []
    assert {item["source"] for item in report["warnings"]} >= {"public_audit", "path_policy"}


def test_release_gate_requires_complete_benchmark_results(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_gate", ["arc", "hellaswag"])
    readme = tmp_path / "README.md"
    benches = tmp_path / "benchmark_results"
    benches.mkdir()
    readme.write_text("public release notes\n", encoding="utf-8")
    (benches / "model_arc_results.json").write_text('{"benchmark":"arc","accuracy":0.8}\n', encoding="utf-8")

    report = release_gate(
        [str(readme)],
        remote="origin",
        benchmark_results=[str(benches)],
        suite="unit_gate",
        model="model",
        require_benchmarks=True,
    )

    assert report["ready"] is False
    blocker = next(item for item in report["blockers"] if item["source"] == "benchmark_manifest")
    assert blocker["missing"] == ["hellaswag"]


def test_release_gate_accepts_clean_origin_with_complete_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(hillstep.BENCHMARK_SUITES, "unit_gate", ["arc"])
    readme = tmp_path / "README.md"
    benches = tmp_path / "benchmark_results"
    benches.mkdir()
    readme.write_text("public release notes\n", encoding="utf-8")
    (benches / "model_arc_results.json").write_text('{"benchmark":"arc","accuracy":0.8}\n', encoding="utf-8")

    report = release_gate(
        [str(readme)],
        remote="origin",
        benchmark_results=[str(benches)],
        suite="unit_gate",
        model="model",
        require_benchmarks=True,
    )

    assert report["ready"] is True
    assert report["blockers"] == []
    assert report["benchmark_manifest"]["missing_measured"] == []
    assert {item["path"] for item in report["export_plan"]["files"]} == {"README.md"}


def test_release_gate_cmd_exits_nonzero_on_blockers(tmp_path: Path, capsys):
    risky = tmp_path / "state.json"
    risky.write_text("{}", encoding="utf-8")
    args = parse_args(["release-gate", str(risky), "--remote", "origin"])

    try:
        release_gate_cmd(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("release gate should exit non-zero when blocked")

    assert "Release gate blocked" in capsys.readouterr().out


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
    assert len(report["files"]) == 6
    assert {row["path"] for row in report["files"]} == {
        "cerebellum-dev/DEVLOG_2026-06-03_gemma4-12b.md",
        "cerebellum_logo.png",
        "osmosis-gemma4-e2b/benchmark_results/e2b_arc_results.json",
        "osmosis-gemma4-e2b/cerebellum_v2_overrides.txt",
        "osmosis-gemma4-e2b/model.gguf",
        "osmosis-gemma4-e2b/server.log",
    }
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
    assert payload["files"][0]["path"] == "README.md"
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


def test_artifact_inventory_records_all_files_beyond_top(tmp_path: Path):
    for idx in range(5):
        (tmp_path / f"file-{idx}.log").write_text(f"{idx}\n", encoding="utf-8")

    report = artifact_inventory(tmp_path, top=2)

    assert len(report["cleanup_candidates"]) == 2
    assert len(report["files"]) == 5
    assert {row["path"] for row in report["files"]} == {f"file-{idx}.log" for idx in range(5)}
    assert all(row["cleanup_reason"] for row in report["files"])


def test_artifact_inventory_cli_validates_root(tmp_path: Path):
    missing = tmp_path / "missing"
    try:
        artifact_inventory_cmd(parse_args(["artifact-inventory", str(missing)]))
    except SystemExit as exc:
        assert "root does not exist" in str(exc)
    else:
        raise AssertionError("artifact-inventory should reject missing roots")


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


def test_hf_stats_command_parses_snapshot_path():
    args = parse_args(["hf-stats", "--author", "deucebucket", "--snapshot", "db/hf_downloads.jsonl"])

    assert args.cmd == "hf-stats"
    assert args.snapshot == "db/hf_downloads.jsonl"


def test_hf_stats_snapshot_appends_recent_ledger(tmp_path):
    path = tmp_path / "hf_downloads.jsonl"
    report = {
        "schema": "cerebellum.hf_model_stats.v1",
        "author": "deucebucket",
        "period": "recent",
        "metric_note": "recent rolling counts",
        "source": "https://huggingface.co/api/models",
        "count": 2,
        "total_downloads_recent": 15,
        "models": [
            {"modelId": "deucebucket/a", "downloads_recent": 10, "likes": 2},
            {"modelId": "deucebucket/b", "downloads_recent": 5, "likes": 1},
        ],
    }

    summary = write_hf_stats_snapshot(report, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert summary["path"] == str(path)
    assert summary["delta_since_previous"] is None
    assert rows[0]["schema"] == "cerebellum.hf_model_stats_snapshot.v1"
    assert rows[0]["total_downloads_recent"] == 15
    assert rows[0]["models"][0]["downloads_recent"] == 10


def test_hf_stats_snapshot_records_delta_from_previous(tmp_path):
    path = tmp_path / "hf_downloads.jsonl"
    first = {
        "author": "deucebucket",
        "period": "recent",
        "metric_note": "recent rolling counts",
        "source": "https://huggingface.co/api/models",
        "count": 1,
        "total_downloads_recent": 10,
        "models": [{"modelId": "deucebucket/a", "downloads_recent": 10, "likes": 2}],
    }
    second = {
        **first,
        "total_downloads_recent": 14,
        "models": [{"modelId": "deucebucket/a", "downloads_recent": 14, "likes": 3}],
    }

    write_hf_stats_snapshot(first, path)
    summary = write_hf_stats_snapshot(second, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert summary["delta_since_previous"] == 4
    assert len(rows) == 2
    assert rows[1]["delta_since_previous"] == 4
    assert rows[1]["model_deltas_since_previous"] == [{"modelId": "deucebucket/a", "delta": 4}]


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
    monkeypatch.setattr("cerebellum.hillstep.ensure_github_branch", lambda _repo, _branch: None)
    monkeypatch.setattr("cerebellum.hillstep.github_file_sha", lambda _repo, _branch, _path: None)
    monkeypatch.setattr("cerebellum.hillstep.gh_json", lambda _args: {})

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
    monkeypatch.setattr("cerebellum.hillstep.gguf_field_text", lambda _path, _key: "llama")

    doctor_cmd(types.SimpleNamespace(source_gguf=str(source), json=True))

    payload = json.loads(capsys.readouterr().out)
    check = next(row for row in payload["checks"] if row["name"] == "gemma4 architecture")
    assert check["ok"] is False
    assert "Gemma4UnifiedForConditionalGeneration" in check["fix"]


def test_doctor_warns_for_gemma4_unstripped_language_model_prefix(tmp_path: Path, monkeypatch, capsys):
    source = tmp_path / "gemma-4-12b-it-f16.gguf"
    source.write_bytes(b"fake")
    monkeypatch.setattr("cerebellum.hillstep.gguf_field_text", lambda _path, _key: "gemma4")

    class FakeReader:
        tensors = [types.SimpleNamespace(name="model.language_model.blk.0.ffn_down.weight")]

    monkeypatch.setattr("gguf.GGUFReader", lambda _path: FakeReader())

    doctor_cmd(types.SimpleNamespace(source_gguf=str(source), json=True))

    payload = json.loads(capsys.readouterr().out)
    check = next(row for row in payload["checks"] if row["name"] == "gemma4 architecture")
    assert check["ok"] is False
    assert "model.language_model.* is stripped" in check["fix"]


# ---------------------------------------------------------------------------
# campaign status board (cerebellum status / watch / next / method)
# ---------------------------------------------------------------------------

import cerebellum.statusboard as statusboard  # noqa: E402

STATUS_NOW = datetime(2026, 6, 12, 4, 0, 0)


def _make_campaign(root: Path, name: str, log_lines: list[str] | None = None,
                   log_rel: str = "logs/continuation.log") -> Path:
    camp = root / f"cerebellum-{name}"
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "RUN_PLAN.md").write_text("# plan\n\n## STOP LINE — review first\n")
    if log_lines is not None:
        log = camp / log_rel
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(log_lines) + "\n")
    return camp


def test_status_renders_synthetic_campaigns(tmp_path: Path):
    _make_campaign(tmp_path, "alpha", [
        "[2026-06-12 03:30:00] === STAGE 2: group ablation ===",
        "[2026-06-12 03:45:00] bench arc (WORKERS=4)",
    ])
    planned = tmp_path / "cerebellum-planned"
    planned.mkdir()
    (planned / "RUN_PLAN.md").write_text("# plan only, nothing started\n")

    out = statusboard.render_status(
        tmp_path, now=STATUS_NOW, include_processes=False, include_modal=False
    )

    assert "CEREBELLUM STATUS" in out
    assert "alpha    STATUS: ok — running" in out
    assert "STAGE 2: group ablation" in out
    assert "bench arc (WORKERS=4)" in out
    assert "15m ago" in out
    assert "human gate: RUN_PLAN.md has a stop line" in out
    assert "planned    STATUS: attention — no activity recorded" in out
    assert "no activity recorded" in out
    assert "Traceback" not in out


def test_status_flags_summary_for_human(tmp_path: Path):
    camp = _make_campaign(tmp_path, "beta", [
        "[2026-06-12 03:50:00] === Driver done. STOP HERE: review RUN_PLAN.md ===",
    ])
    (camp / "SUMMARY_FOR_HUMAN.md").write_text("read me\n")

    out = statusboard.render_status(
        tmp_path, now=STATUS_NOW, include_processes=False, include_modal=False
    )

    assert "beta    STATUS: attention — waiting on you" in out
    assert "SUMMARY_FOR_HUMAN.md is waiting for your review" in out
    assert "driver stopped at the human-review gate" in out
    assert "== WAITING FOR YOU ==" in out
    assert "- [beta]" in out


def test_status_reviewed_marker_clears_summary_flag(tmp_path: Path):
    camp = _make_campaign(tmp_path, "delta", [
        "[2026-06-12 03:50:00] verdict recorded",
    ])
    summary = camp / "SUMMARY_FOR_HUMAN.md"
    summary.write_text("read me\n")
    reviewed = camp / "SUMMARY_FOR_HUMAN.md.reviewed"
    reviewed.write_text("reviewed 2026-06-12\n")
    later = summary.stat().st_mtime + 10
    os.utime(reviewed, (later, later))

    out = statusboard.render_status(
        tmp_path, now=STATUS_NOW, include_processes=False, include_modal=False
    )
    assert "SUMMARY_FOR_HUMAN.md is waiting for your review" not in out

    # A summary regenerated AFTER the review re-flags.
    newest = reviewed.stat().st_mtime + 10
    os.utime(summary, (newest, newest))
    out = statusboard.render_status(
        tmp_path, now=STATUS_NOW, include_processes=False, include_modal=False
    )
    assert "SUMMARY_FOR_HUMAN.md is waiting for your review" in out


def test_status_stale_campaign_flags_attention(tmp_path: Path):
    _make_campaign(tmp_path, "gamma", [
        "[2026-06-11 20:00:00] still chewing on tensor 12",
    ])

    out = statusboard.render_status(
        tmp_path, now=STATUS_NOW, include_processes=False, include_modal=False
    )

    assert "gamma    STATUS: attention" in out
    assert "stalled or quietly finished" in out


def test_status_with_no_campaigns_is_graceful(tmp_path: Path):
    out = statusboard.render_status(
        tmp_path, now=STATUS_NOW, include_processes=False, include_modal=False
    )

    assert "none found" in out
    assert "nothing — go back to sleep" in out


def test_status_reads_modal_spend_from_progress_log(tmp_path: Path):
    _make_campaign(tmp_path, "flash", [
        "[2026-06-12 03:00:00] PPL ablate mla_q done (cum $1.90)",
        "[2026-06-12 03:40:00] PPL ablate token_embd done (cum $2.33)",
    ], log_rel="modal_results/results/progress.log")

    out = statusboard.render_status(
        tmp_path, now=STATUS_NOW, include_processes=False, include_modal=False
    )

    assert "modal spend so far: $2.33" in out


def test_status_modal_credits_tolerates_missing_script(tmp_path: Path):
    assert "not checked" in statusboard.modal_credits_summary(tmp_path)


def test_next_prints_now_and_next_sections(tmp_path: Path):
    backlog = tmp_path / "cerebellum-dev" / "BACKLOG.md"
    backlog.parent.mkdir(parents=True)
    backlog.write_text(
        "# Cerebellum Backlog\n\npreamble\n\n"
        "## NOW (running tonight)\n\n- finish gemma 12b gates\n\n"
        "## NEXT (committed, ordered)\n\n1. 35B tensor-level budget pass\n\n"
        "## LATER (real, not yet scheduled)\n\n- everything else\n"
    )

    out = statusboard.render_next(tmp_path)

    assert "## NOW" in out
    assert "finish gemma 12b gates" in out
    assert "35B tensor-level budget pass" in out
    assert "LATER" not in out
    assert "everything else" not in out


def test_next_missing_backlog_is_graceful(tmp_path: Path):
    out = statusboard.render_next(tmp_path)
    assert "backlog not found" in out


def test_method_prints_canon_sections(tmp_path: Path):
    method = tmp_path / "cerebellum-dev" / "knowledge" / "CURRENT_METHOD.md"
    method.parent.mkdir(parents=True)
    method.write_text(
        "# CURRENT METHOD\n\nintro\n\n"
        "## Canonical: the OG group-first, bench-gated formula\n\n"
        "1. HF/BF16 -> F16 GGUF\n2. Imatrix full coverage\n\n"
        "## Deprecated / dead (do not use)\n\n- hillstep exhaustive hill-climb\n\n"
        "## Standing operational rules\n\n- N=2 PPL workers\n"
    )

    out = statusboard.render_method(tmp_path)

    assert "THE METHOD" in out
    assert "OG group-first, bench-gated formula" in out
    assert "Imatrix full coverage" in out
    assert "hillstep exhaustive hill-climb" in out
    assert "N=2 PPL workers" not in out


def test_method_missing_file_uses_fallback(tmp_path: Path):
    out = statusboard.render_method(tmp_path)
    assert "BENCHMARK GATES" in out
    assert "DEPRECATED" in out


def test_watch_renders_once_with_iterations_limit(monkeypatch, capsys):
    monkeypatch.setattr(statusboard, "render_status", lambda: "SNAPSHOT")

    rc = statusboard.cmd_watch([], interval=0, iterations=1)

    captured = capsys.readouterr()
    assert rc == 0
    assert "SNAPSHOT" in captured.out
    assert "refreshing every" in captured.out


def test_status_rejects_arguments_with_hint(capsys):
    rc = statusboard.cmd_status(["--data-root", "/tmp"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "cerebellum hillstep status" in captured.err


def test_cli_status_subprocess_runs_zero_args(tmp_path: Path):
    _make_campaign(tmp_path, "subproc", [
        "[2026-06-12 03:30:00] === STAGE 1: baselines ===",
    ])

    result = subprocess.run(
        [sys.executable, "-m", "cerebellum.cli", "status"],
        check=True, capture_output=True, text=True, cwd=tmp_path,
    )

    assert "CEREBELLUM STATUS" in result.stdout
    assert "subproc" in result.stdout
    assert "== MODAL ==" in result.stdout


def test_cli_run_prints_deprecation_banner():
    result = subprocess.run(
        [sys.executable, "-m", "cerebellum.cli", "run", "--help"],
        check=True, capture_output=True, text=True,
    )

    assert "DEPRECATED" in result.stderr
    assert "cerebellum method" in result.stderr


def test_cli_top_help_leads_with_status_board():
    result = subprocess.run(
        [sys.executable, "-m", "cerebellum.cli", "--help"],
        check=True, capture_output=True, text=True,
    )

    assert "cerebellum status" in result.stdout
    assert "cerebellum method" in result.stdout
    assert "imatrix" in result.stdout


def test_cli_hillstep_namespace_reaches_legacy_engine():
    result = subprocess.run(
        [sys.executable, "-m", "cerebellum.cli", "hillstep", "--help"],
        check=True, capture_output=True, text=True,
    )

    assert "plan-space" in result.stdout
    assert "DEPRECATED" in result.stdout
