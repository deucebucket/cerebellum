from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ParquetSource:
    repo: str
    filename: str
    repo_type: str = "dataset"


SMOKE_SOURCES = {
    "smol-smoltalk": ParquetSource("HuggingFaceTB/smol-smoltalk", "data/train-00000-of-00004.parquet"),
    "openthoughts3": ParquetSource("open-thoughts/OpenThoughts3-1.2M", "data/train-00000-of-00120.parquet"),
    "opencodereasoning": ParquetSource("nvidia/OpenCodeReasoning", "split_0/train-00000-of-00030.parquet"),
}


def download_source(source: ParquetSource) -> Path:
    return Path(hf_hub_download(source.repo, source.filename, repo_type=source.repo_type))


def message_list_to_text(messages: list[dict[str, Any]]) -> str:
    chunks = []
    for msg in messages:
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        if isinstance(content, list):
            content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        chunks.append(f"{role}: {content}")
    return "\n".join(chunks)


def row_to_text(row: dict[str, Any]) -> str | None:
    if isinstance(row.get("messages"), list):
        return message_list_to_text(row["messages"])
    if isinstance(row.get("conversation"), list):
        return message_list_to_text(row["conversation"])
    if isinstance(row.get("conversations"), list):
        return message_list_to_text(row["conversations"])
    if row.get("input") and row.get("output"):
        return f"user: {row['input']}\nassistant: {row['output']}"
    if row.get("problem") and row.get("generated_solution"):
        return f"user: {row['problem']}\nassistant: {row['generated_solution']}"
    for key in ("text", "content", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def iter_parquet_texts(path: Path, *, max_rows: int | None = None, seed: int = 0) -> Iterator[str]:
    parquet = pq.ParquetFile(path)
    seen = 0
    groups = list(range(parquet.num_row_groups))
    random.Random(seed).shuffle(groups)
    for group in groups:
        table = parquet.read_row_group(group)
        rows = table.to_pylist()
        random.Random(seed + group).shuffle(rows)
        for row in rows:
            text = row_to_text(row)
            if text:
                yield text
                seen += 1
                if max_rows is not None and seen >= max_rows:
                    return


def materialize_smoke_jsonl(source_name: str, output: Path, *, rows: int = 128, seed: int = 0) -> Path:
    source = SMOKE_SOURCES[source_name]
    parquet_path = download_source(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for text in iter_parquet_texts(parquet_path, max_rows=rows, seed=seed):
            f.write(json.dumps({"source": source.repo, "text": text}, ensure_ascii=True) + "\n")
    return output
