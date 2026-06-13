import json
import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze_ablation_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_ablation_results", SCRIPT_PATH)
analyze_ablation_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analyze_ablation_results
SPEC.loader.exec_module(analyze_ablation_results)

DEFAULT_THRESHOLDS = analyze_ablation_results.DEFAULT_THRESHOLDS
analyze_rows = analyze_ablation_results.analyze_rows
build_overrides = analyze_ablation_results.build_overrides
exact_tensor_pattern = analyze_ablation_results.exact_tensor_pattern
load_ablation = analyze_ablation_results.load_ablation
parse_weights = analyze_ablation_results.parse_weights
profile_weights = analyze_ablation_results.profile_weights


def test_load_ablation_normalizes_multidomain_and_skips_errors(tmp_path):
    path = tmp_path / "ablation.json"
    path.write_text(
        json.dumps(
            {
                "baseline_ppl": {"chat": 100.0, "code": 50.0},
                "tests": {
                    "safe": {
                        "gguf_tensor": "blk.1.ffn_down",
                        "ppl": {"chat": 100.1, "code": 49.9},
                    },
                    "bad": {"gguf_tensor": "blk.2.ffn_down.weight", "error": "ppl_failed"},
                },
            }
        )
    )

    baseline, rows, skipped = load_ablation(path)

    assert baseline == {"chat": 100.0, "code": 50.0}
    assert rows == [
        {
            "label": "safe",
            "tensor": "blk.1.ffn_down.weight",
            "ppl": {"chat": 100.1, "code": 49.9},
        }
    ]
    assert skipped == [{"label": "bad", "tensor": "blk.2.ffn_down.weight", "reason": "ppl_failed"}]


def test_load_ablation_normalizes_legacy_scalar_schema(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "baseline_ppl": 10.0,
                "tests": {
                    "blk_0_ffn_gate": {"gguf_tensor": "blk.0.ffn_gate", "ppl": 10.2},
                },
            }
        )
    )

    baseline, rows, skipped = load_ablation(path)

    assert baseline == {"ppl": 10.0}
    assert rows[0]["tensor"] == "blk.0.ffn_gate.weight"
    assert rows[0]["ppl"] == {"ppl": 10.2}
    assert skipped == []


def test_analyze_rows_classifies_and_generates_anchored_overrides():
    baseline = {"chat": 100.0, "code": 50.0}
    rows = [
        {
            "label": "beneficial",
            "tensor": "blk.0.ffn_down_exps.weight",
            "ppl": {"chat": 99.0, "code": 50.0},
        },
        {
            "label": "critical",
            "tensor": "blk.1.ffn_down_exps.weight",
            "ppl": {"chat": 109.0, "code": 50.0},
        },
        {
            "label": "safe",
            "tensor": "blk.2.ffn_down_exps.weight",
            "ppl": {"chat": 100.2, "code": 50.1},
        },
    ]
    weights = parse_weights("chat:0.5,code:0.5", list(baseline))

    analyzed = analyze_rows(baseline, rows, weights, DEFAULT_THRESHOLDS)
    by_tensor = {row["tensor"]: row for row in analyzed}

    assert by_tensor["blk.0.ffn_down_exps.weight"]["class"] == "demotable"
    assert by_tensor["blk.0.ffn_down_exps.weight"]["subclass"] == "beneficial"
    assert by_tensor["blk.1.ffn_down_exps.weight"]["class"] == "critical"
    assert by_tensor["blk.2.ffn_down_exps.weight"]["class"] == "demotable"
    assert by_tensor["blk.2.ffn_down_exps.weight"]["subclass"] == "safe"

    overrides = build_overrides(analyzed, {"demotable"}, "Q3_K")
    lines = {f"{item['pattern']}={item['qtype']}" for item in overrides}
    assert r"^blk\.0\.ffn_down_exps\.weight$=Q3_K" in lines
    assert r"^blk\.2\.ffn_down_exps\.weight$=Q3_K" in lines
    assert all("blk.1" not in line for line in lines)


def test_exact_tensor_pattern_escapes_regex_metacharacters():
    assert exact_tensor_pattern("blk.0.ffn_down.weight") == r"^blk\.0\.ffn_down\.weight$"


def test_profile_weights_renormalize_to_available_domains():
    weights = profile_weights("reason", ["chat", "reasoning", "code"])

    assert set(weights) == {"chat", "reasoning", "code"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["reasoning"] > weights["chat"] > weights["code"]
