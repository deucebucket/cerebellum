import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "plan_cerebellum_pipeline.py"
SPEC = importlib.util.spec_from_file_location("plan_cerebellum_pipeline", SCRIPT_PATH)
plan_cerebellum_pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = plan_cerebellum_pipeline
SPEC.loader.exec_module(plan_cerebellum_pipeline)

build_steps = plan_cerebellum_pipeline.build_steps
manifest = plan_cerebellum_pipeline.manifest
write_runner = plan_cerebellum_pipeline.write_runner


def args(tmp_path):
    return Namespace(
        source_gguf=Path("/models/source.gguf"),
        imatrix=Path("/models/imatrix.dat"),
        tensors_file=Path("/runs/tensors.txt"),
        corpus_dir=Path("/runs/corpus"),
        output_dir=tmp_path / "out",
        model_name="test-model",
        base_type="Q4_K_M",
        ablate_type="Q3_K",
        domains="chat,reasoning,code",
        weights="chat:0.25,reasoning:0.5,code:0.25",
        queue_depth=1,
        ppl_workers=1,
        ctx_size=512,
        chunks=2,
        quantize_bin="/bin/llama-quantize",
        baseline_model="baseline",
    )


def test_build_steps_wires_analyzer_and_reporter(tmp_path):
    cfg = args(tmp_path)
    steps = build_steps(cfg)

    assert [step.name for step in steps] == ["ablate", "analyze-ablation", "final-quantize", "benchmark-report"]
    assert steps[0].gpu is True
    assert "scripts/ablate_multidomain.py" in steps[0].command
    assert "cerebellum-dev/tools/analyze_ablation_results.py" in steps[1].command
    assert "cerebellum-dev/tools/compare_benchmark_results.py" in steps[3].command
    assert any("tensor_types_demotable.txt" in out for out in steps[1].outputs)


def test_manifest_is_json_serializable(tmp_path):
    cfg = args(tmp_path)
    data = manifest(cfg, build_steps(cfg))

    encoded = json.dumps(data)

    assert "test-model" in encoded
    assert data["steps"][0]["shell"].startswith("python scripts/ablate_multidomain.py")


def test_write_runner_defaults_to_dry_run_guard(tmp_path):
    cfg = args(tmp_path)
    path = tmp_path / "run.sh"

    write_runner(path, build_steps(cfg), execute=False)

    text = path.read_text()
    assert "Dry-run runner" in text
    assert "exit 0" in text
    assert "scripts/ablate_multidomain.py" in text
