import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "plan_pairwise_ablation.py"
SPEC = importlib.util.spec_from_file_location("plan_pairwise_ablation", SCRIPT_PATH)
plan_pairwise_ablation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = plan_pairwise_ablation
SPEC.loader.exec_module(plan_pairwise_ablation)

load_candidates = plan_pairwise_ablation.load_candidates
build_plan = plan_pairwise_ablation.build_plan
write_override_files = plan_pairwise_ablation.write_override_files
exact_tensor_pattern = plan_pairwise_ablation.exact_tensor_pattern


def write_analysis(path):
    rows = [
        {"tensor": "blk.0.ffn_down.weight", "class": "demotable", "weighted_rel_pct": -1.0, "max_rel_pct": 0.1},
        {"tensor": "blk.1.ffn_down.weight", "class": "demotable", "weighted_rel_pct": -0.5, "max_rel_pct": 0.2},
        {"tensor": "blk.2.attn_q.weight", "class": "sacred", "weighted_rel_pct": 3.0, "max_rel_pct": 4.0},
        {"tensor": "blk.3.attn_k.weight", "class": "sacred", "weighted_rel_pct": 2.0, "max_rel_pct": 2.5},
        {"tensor": "blk.4.ffn_up.weight", "class": "tolerant", "weighted_rel_pct": 0.3, "max_rel_pct": 0.7},
    ]
    path.write_text(json.dumps({"rows": rows}))


def args(tmp_path, analysis):
    return Namespace(
        analysis=analysis,
        output_json=tmp_path / "plan.json",
        output_dir=tmp_path / "overrides",
        top_k=2,
        ablate_type="Q3_K",
        runner=None,
        source_gguf=Path("source.gguf"),
        imatrix=Path("imatrix.dat"),
        base_type="Q4_K_M",
        quantize_bin="llama-quantize",
    )


def test_build_plan_selects_pairs_and_expected_delta(tmp_path):
    analysis = tmp_path / "analysis.json"
    write_analysis(analysis)

    plan = build_plan(load_candidates(analysis), args(tmp_path, analysis))

    assert plan["candidate_counts"] == {"demotable": 2, "sacred": 2, "tolerant": 1}
    assert plan["pair_count"] == 7
    first = plan["pairs"][0]
    assert first["group"] == "sacred_x_demotable"
    assert first["expected_additive_weighted_rel_pct"] == 2.0
    assert r"^blk\.2\.attn_q\.weight$=Q3_K" in first["override_text"]


def test_write_override_files_adds_paths(tmp_path):
    analysis = tmp_path / "analysis.json"
    write_analysis(analysis)
    plan = build_plan(load_candidates(analysis), args(tmp_path, analysis))

    write_override_files(plan, tmp_path / "overrides")

    assert Path(plan["pairs"][0]["override_file"]).exists()
    assert Path(plan["pairs"][0]["override_file"]).read_text().count("Q3_K") == 2


def test_exact_tensor_pattern_anchors_and_escapes():
    assert exact_tensor_pattern("blk.0.attn_q.weight") == r"^blk\.0\.attn_q\.weight$"
