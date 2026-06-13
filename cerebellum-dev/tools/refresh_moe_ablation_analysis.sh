#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$ROOT/cerebellum-dev/sparse-upcycling/runs"
ANALYZER="$ROOT/cerebellum-dev/tools/analyze_ablation_results.py"

ABLATION_JSON="${1:-$RUN_DIR/moe_ablation_routed_q4km_to_q3.json}"
PREFIX="${2:-${ABLATION_JSON%.json}_analysis}"

python "$ANALYZER" \
  --ablation "$ABLATION_JSON" \
  --weights chat:0.25,reasoning:0.5,code:0.25 \
  --output-json "${PREFIX}.json" \
  --summary-md "${PREFIX}.md" \
  --tensor-type-file "${PREFIX}_demotable_q3.txt" \
  --override-type Q3_K
