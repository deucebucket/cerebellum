#!/usr/bin/env bash
set -euo pipefail

echo 'Pairwise plan runner is dry-run only for now.'
exit 0

echo '=== blk_0_ffn_down_exps__AND__blk_2_ffn_gate_up_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_0_ffn_down_exps__AND__blk_2_ffn_gate_up_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_0_ffn_down_exps__AND__blk_2_ffn_gate_up_exps.gguf Q4_K_M

echo '=== blk_0_ffn_down_exps__AND__blk_1_ffn_down_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_0_ffn_down_exps__AND__blk_1_ffn_down_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_0_ffn_down_exps__AND__blk_1_ffn_down_exps.gguf Q4_K_M

echo '=== blk_1_ffn_gate_up_exps__AND__blk_2_ffn_gate_up_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_1_ffn_gate_up_exps__AND__blk_2_ffn_gate_up_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_1_ffn_gate_up_exps__AND__blk_2_ffn_gate_up_exps.gguf Q4_K_M

echo '=== blk_1_ffn_gate_up_exps__AND__blk_1_ffn_down_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_1_ffn_gate_up_exps__AND__blk_1_ffn_down_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_1_ffn_gate_up_exps__AND__blk_1_ffn_down_exps.gguf Q4_K_M

echo '=== blk_2_ffn_gate_up_exps__AND__blk_1_ffn_down_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_2_ffn_gate_up_exps__AND__blk_1_ffn_down_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_2_ffn_gate_up_exps__AND__blk_1_ffn_down_exps.gguf Q4_K_M

echo '=== blk_0_ffn_down_exps__AND__blk_3_ffn_gate_up_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_0_ffn_down_exps__AND__blk_3_ffn_gate_up_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_0_ffn_down_exps__AND__blk_3_ffn_gate_up_exps.gguf Q4_K_M

echo '=== blk_0_ffn_down_exps__AND__blk_3_ffn_down_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_0_ffn_down_exps__AND__blk_3_ffn_down_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_0_ffn_down_exps__AND__blk_3_ffn_down_exps.gguf Q4_K_M

echo '=== blk_0_ffn_down_exps__AND__blk_0_ffn_gate_up_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_0_ffn_down_exps__AND__blk_0_ffn_gate_up_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_0_ffn_down_exps__AND__blk_0_ffn_gate_up_exps.gguf Q4_K_M

echo '=== blk_0_ffn_down_exps__AND__blk_2_ffn_down_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_0_ffn_down_exps__AND__blk_2_ffn_down_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_0_ffn_down_exps__AND__blk_2_ffn_down_exps.gguf Q4_K_M

echo '=== blk_1_ffn_gate_up_exps__AND__blk_3_ffn_gate_up_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_1_ffn_gate_up_exps__AND__blk_3_ffn_gate_up_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_1_ffn_gate_up_exps__AND__blk_3_ffn_gate_up_exps.gguf Q4_K_M

echo '=== blk_1_ffn_gate_up_exps__AND__blk_3_ffn_down_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_1_ffn_gate_up_exps__AND__blk_3_ffn_down_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_1_ffn_gate_up_exps__AND__blk_3_ffn_down_exps.gguf Q4_K_M

echo '=== blk_1_ffn_gate_up_exps__AND__blk_0_ffn_gate_up_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_1_ffn_gate_up_exps__AND__blk_0_ffn_gate_up_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_1_ffn_gate_up_exps__AND__blk_0_ffn_gate_up_exps.gguf Q4_K_M

echo '=== blk_1_ffn_gate_up_exps__AND__blk_2_ffn_down_exps ==='
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize --imatrix /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat --tensor-type-file cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/blk_1_ffn_gate_up_exps__AND__blk_2_ffn_down_exps.txt /var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf cerebellum-dev/sparse-upcycling/runs/moe_ablation_pairwise_reason_top4_overrides/ggufs/blk_1_ffn_gate_up_exps__AND__blk_2_ffn_down_exps.gguf Q4_K_M
