import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "sparse-upcycling" / "scripts" / "probe_generation.py"
SPEC = importlib.util.spec_from_file_location("probe_generation", SCRIPT_PATH)
probe_generation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe_generation
SPEC.loader.exec_module(probe_generation)


def test_parse_prompts_keeps_defaults_and_adds_named_override():
    prompts = probe_generation.parse_prompts(["math=What is 2+2?", "unnamed prompt"])

    assert prompts["hello"] == "Say exactly hello."
    assert prompts["math"] == "What is 2+2?"
    assert prompts["prompt_1"] == "unnamed prompt"


def test_text_stats_flags_whitespace_only_output():
    stats = probe_generation.text_stats("\n \t\n")

    assert stats["is_empty_or_whitespace"] is True
    assert stats["non_ws_chars"] == 0
    assert stats["lines"] == 3


def test_repetition_stats_catches_repeated_token_runs():
    stats = probe_generation.repetition_stats([7, 7, 7, 3, 3, 9])

    assert stats["tokens"] == 6
    assert stats["unique_tokens"] == 3
    assert stats["max_run"] == 3
    assert stats["top_token_fraction"] == 0.5
