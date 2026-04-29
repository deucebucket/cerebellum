#!/usr/bin/env python3
"""Deduplicate benchmark JSONL files caused by concurrent runners."""
import json
import glob
import os
import sys

RESULTS_DIR = "/var/home/deucebucket/ai-drive/osmosis/osmosis-qwen36-27b/benchmark_results"
MODEL_NAME = os.environ.get("BENCH_MODEL", "cerebellum_v4")


def dedup_jsonl(path):
    seen = set()
    unique = []
    dups = 0

    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            key_parts = []
            for k in ("question", "context", "subject"):
                if k in entry:
                    key_parts.append(str(entry[k])[:100])
            key_parts.append(entry.get("expected", ""))
            key = "|".join(key_parts)

            if key in seen:
                dups += 1
                continue
            seen.add(key)
            unique.append(line)

    if dups == 0:
        print(f"  {os.path.basename(path)}: clean ({len(unique)} entries)")
        return

    with open(path, "w") as f:
        f.writelines(unique)
    print(f"  {os.path.basename(path)}: {len(unique)} kept, {dups} duplicates removed")


def main():
    pattern = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_*_detailed.jsonl")
    files = glob.glob(pattern)
    if not files:
        print("No JSONL files found")
        return

    print(f"Deduplicating {len(files)} benchmark files...")
    for path in sorted(files):
        dedup_jsonl(path)
    print("Done.")


if __name__ == "__main__":
    main()
