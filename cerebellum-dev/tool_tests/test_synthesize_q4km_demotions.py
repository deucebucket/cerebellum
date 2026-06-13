import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "sparse-upcycling"
    / "scripts"
    / "synthesize_q4km_demotions.py"
)
SPEC = importlib.util.spec_from_file_location("synthesize_q4km_demotions", SCRIPT_PATH)
synthesize_q4km_demotions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = synthesize_q4km_demotions
SPEC.loader.exec_module(synthesize_q4km_demotions)

read_analysis = synthesize_q4km_demotions.read_analysis
select_ablation_candidates = synthesize_q4km_demotions.select_ablation_candidates
write_ablation_candidate = synthesize_q4km_demotions.write_ablation_candidate


def write_analysis(path, rows):
    path.write_text(
        json.dumps(
            {
                "tested_count": len(rows),
                "counts": {},
                "rows": [{"tensor": tensor, "class": class_name} for tensor, class_name in rows],
            }
        )
    )


def test_select_ablation_candidates_intersects_demotables_and_blocks_sacred(tmp_path):
    first = tmp_path / "general.json"
    second = tmp_path / "reason.json"
    write_analysis(
        first,
        [
            ("blk.1.ffn_down_exps.weight", "demotable"),
            ("blk.2.ffn_down_exps.weight", "demotable"),
            ("blk.3.ffn_down_exps.weight", "sacred"),
        ],
    )
    write_analysis(
        second,
        [
            ("blk.1.ffn_down_exps.weight", "demotable"),
            ("blk.2.ffn_down_exps.weight", "sacred"),
            ("blk.4.ffn_gate_up_exps.weight", "demotable"),
        ],
    )

    selected = select_ablation_candidates(
        [read_analysis(first), read_analysis(second)],
        emit_classes={"demotable"},
        block_classes={"sacred", "critical"},
        mode="intersection",
    )

    assert selected["selected"] == ["blk.1.ffn_down_exps.weight"]
    assert selected["blocked"] == ["blk.2.ffn_down_exps.weight"]
    assert selected["missing_from_intersection"] == [
        "blk.2.ffn_down_exps.weight",
        "blk.4.ffn_gate_up_exps.weight",
    ]


def test_write_ablation_candidate_emits_manifest_and_quantize_command(tmp_path):
    analysis = tmp_path / "analysis.json"
    write_analysis(analysis, [("blk.1.ffn_down_exps.weight", "demotable")])
    output = tmp_path / "candidate.txt"
    manifest = tmp_path / "candidate_manifest.json"

    meta = write_ablation_candidate(
        analyses=[analysis],
        output=output,
        manifest=manifest,
        qtype="Q3_K",
        emit_classes={"demotable"},
        block_classes={"sacred", "critical"},
        selection="intersection",
        quantize_bin="llama-quantize",
        imatrix=tmp_path / "imatrix.dat",
        source_gguf=tmp_path / "source.gguf",
        output_gguf=tmp_path / "out.gguf",
        base_quant="Q4_K_M",
    )

    assert output.read_text() == "^blk\\.1\\.ffn_down_exps\\.weight$=Q3_K\n"
    assert meta["override_count"] == 1
    loaded = json.loads(manifest.read_text())
    assert loaded["entries"] == {"blk.1.ffn_down_exps.weight": "Q3_K"}
    assert loaded["quantize_command"] == [
        "llama-quantize",
        "--imatrix",
        str(tmp_path / "imatrix.dat"),
        "--tensor-type-file",
        str(output),
        str(tmp_path / "source.gguf"),
        str(tmp_path / "out.gguf"),
        "Q4_K_M",
    ]

