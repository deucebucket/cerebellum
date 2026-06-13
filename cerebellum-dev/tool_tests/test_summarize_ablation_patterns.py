import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "summarize_ablation_patterns.py"
SPEC = importlib.util.spec_from_file_location("summarize_ablation_patterns", SCRIPT_PATH)
summarize_ablation_patterns = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = summarize_ablation_patterns
SPEC.loader.exec_module(summarize_ablation_patterns)

load_records = summarize_ablation_patterns.load_records
aggregate = summarize_ablation_patterns.aggregate
render_markdown = summarize_ablation_patterns.render_markdown
tensor_group = summarize_ablation_patterns.tensor_group


def test_load_standard_ablation_schema(tmp_path):
    path = tmp_path / "osmosis-test" / "ablation_results.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "baseline_ppl": 10.0,
                "tests": {
                    "blk_0_attn_q": {"gguf_tensor": "blk.0.attn_q.weight", "ppl": 10.5},
                    "bad": {"gguf_tensor": "blk.1.attn_q.weight", "error": "ppl_failed"},
                },
            }
        )
    )

    rows = load_records(path)

    assert len(rows) == 1
    assert rows[0].model == "osmosis-test"
    assert rows[0].layer == 0
    assert rows[0].tensor_group == "attn_q"
    assert rows[0].delta_pct == 5.0


def test_load_analysis_and_list_schemas(tmp_path):
    analysis = tmp_path / "osmosis-a" / "analysis.json"
    analysis.parent.mkdir()
    analysis.write_text(json.dumps({"baseline_ppl": 100.0, "tensor_group": "attn_k", "results": [{"layer": 1, "ppl": 99.0}]}))
    listed = tmp_path / "osmosis-b" / "results.json"
    listed.parent.mkdir()
    listed.write_text(json.dumps([{"group": "attn_o", "layer": 2, "baseline_ppl": 100.0, "ppl": 130.0}]))

    rows = load_records(analysis) + load_records(listed)

    assert rows[0].model == "osmosis-a"
    assert rows[0].tensor_group == "attn_k"
    assert rows[0].classification == "demotable"
    assert rows[1].model == "osmosis-b"
    assert rows[1].tensor_group == "attn_o"
    assert rows[1].classification == "critical"


def test_aggregate_and_markdown_report():
    path = Path("osmosis-test/ablation_results.json")
    records = summarize_ablation_patterns.load_standard_schema(
        path,
        {
            "baseline_ppl": 100.0,
            "tests": {
                "a": {"gguf_tensor": "blk.0.ffn_down.weight", "ppl": 99.0},
                "b": {"gguf_tensor": "blk.9.ffn_down.weight", "ppl": 110.0},
            },
        },
    )

    summary = aggregate(records)
    markdown = render_markdown(records, summary)

    assert summary["coverage"]["osmosis-test"] == 2
    assert summary["by_model_group"]["osmosis-test/ffn_down"]["count"] == 2
    assert "## Most Demotable" in markdown
    assert "## Most Sensitive" in markdown


def test_tensor_group_detects_moe_experts_first():
    assert tensor_group("blk.0.ffn_gate_up_exps.weight") == "moe_ffn_gate_up_exps"
