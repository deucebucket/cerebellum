import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "validate_calibration_manifest.py"
SPEC = importlib.util.spec_from_file_location("validate_calibration_manifest", SCRIPT_PATH)
validate_calibration_manifest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validate_calibration_manifest
SPEC.loader.exec_module(validate_calibration_manifest)

validate_manifest = validate_calibration_manifest.validate_manifest


def write_manifest(path, data):
    path.write_text(json.dumps(data))
    return path


def valid_manifest(builder):
    return {
        "manifest_version": "1.0",
        "policy": "reuse_public_datasets_only",
        "seed": 20260529,
        "text_corpora": [
            {
                "id": "wiki_v1",
                "hf_repo": "Salesforce/wikitext",
                "filename": "data.parquet",
                "builder": str(builder),
                "output": "/tmp/wiki.txt",
                "target_bytes": 1000,
            }
        ],
        "sparse_upcycling_jsonl": [
            {
                "id": "chat_v1",
                "hf_repo": "HuggingFaceTB/smol-smoltalk",
                "filename": "data.parquet",
                "output": "/tmp/chat.jsonl",
                "rows": 8,
            }
        ],
        "aliases": {"mix": ["wiki_v1", "chat_v1"]},
    }


def test_validate_manifest_accepts_valid_manifest(tmp_path):
    builder = tmp_path / "builder.py"
    builder.write_text("")
    path = write_manifest(tmp_path / "manifest.json", valid_manifest(builder))

    _, errors, warnings = validate_manifest(path, tmp_path)

    assert errors == []
    assert warnings == []


def test_validate_manifest_rejects_bad_alias_and_duplicate_id(tmp_path):
    data = valid_manifest("missing.py")
    data["sparse_upcycling_jsonl"][0]["id"] = "wiki_v1"
    data["aliases"]["bad"] = ["does_not_exist"]
    path = write_manifest(tmp_path / "manifest.json", data)

    _, errors, warnings = validate_manifest(path, tmp_path)

    assert any("duplicate id" in error for error in errors)
    assert any("unknown dataset id" in error for error in errors)
    assert any("builder path does not exist" in warning for warning in warnings)


def test_validate_manifest_requires_public_reuse_policy(tmp_path):
    data = valid_manifest("builder.py")
    data["policy"] = "make_new_dataset"
    path = write_manifest(tmp_path / "manifest.json", data)

    _, errors, _ = validate_manifest(path, tmp_path)

    assert "policy must be reuse_public_datasets_only" in errors
