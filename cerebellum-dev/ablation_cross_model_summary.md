# Cross-Model Ablation Pattern Summary

## Coverage

| Model | Tests |
| --- | ---: |
| `osmosis-gemma4-26b` | 49 |
| `osmosis-gemma4-e4b` | 26 |
| `osmosis-qwen35-9b` | 202 |
| `osmosis-qwen36-27b` | 23 |

## Group Sensitivity

| Model / Group | Count | Median Δ% | Classes |
| --- | ---: | ---: | --- |
| `osmosis-gemma4-26b/attn_k` | 30 | 0.074 | critical:4, demotable:11, sacred:3, tolerant:12 |
| `osmosis-gemma4-26b/attn_o` | 7 | 30.701 | critical:7 |
| `osmosis-gemma4-26b/ffn_down_exps` | 7 | 29.931 | critical:7 |
| `osmosis-gemma4-26b/ffn_gate_up_exps` | 5 | 27.914 | critical:5 |
| `osmosis-gemma4-e4b/attn_k` | 1 | 0.054 | tolerant:1 |
| `osmosis-gemma4-e4b/attn_q` | 6 | 0.139 | critical:1, demotable:1, tolerant:4 |
| `osmosis-gemma4-e4b/attn_v` | 1 | 0.009 | tolerant:1 |
| `osmosis-gemma4-e4b/ffn_gate` | 11 | -0.113 | demotable:5, tolerant:6 |
| `osmosis-gemma4-e4b/ffn_up` | 7 | -0.054 | demotable:2, sacred:1, tolerant:4 |
| `osmosis-qwen35-9b/attn_k` | 8 | 0.086 | demotable:1, tolerant:7 |
| `osmosis-qwen35-9b/attn_output` | 8 | 0.320 | demotable:2, sacred:1, tolerant:5 |
| `osmosis-qwen35-9b/attn_q` | 8 | 0.133 | demotable:1, tolerant:7 |
| `osmosis-qwen35-9b/attn_qkv` | 24 | 0.203 | demotable:6, sacred:6, tolerant:12 |
| `osmosis-qwen35-9b/attn_v` | 8 | 0.196 | demotable:1, sacred:2, tolerant:5 |
| `osmosis-qwen35-9b/ffn_down` | 32 | 0.190 | demotable:7, sacred:1, tolerant:24 |
| `osmosis-qwen35-9b/ffn_gate` | 32 | 0.163 | demotable:4, tolerant:28 |
| `osmosis-qwen35-9b/ffn_up` | 32 | 0.258 | demotable:3, sacred:6, tolerant:23 |
| `osmosis-qwen35-9b/other` | 24 | 0.083 | demotable:4, sacred:2, tolerant:18 |
| `osmosis-qwen35-9b/output` | 1 | 9.264 | critical:1 |
| `osmosis-qwen35-9b/ssm_out` | 24 | 0.226 | critical:1, demotable:1, sacred:1, tolerant:21 |
| `osmosis-qwen35-9b/token_embd` | 1 | 0.151 | tolerant:1 |
| `osmosis-qwen36-27b/attn_k` | 1 | -0.048 | tolerant:1 |
| `osmosis-qwen36-27b/attn_q` | 1 | 1.965 | sacred:1 |
| `osmosis-qwen36-27b/attn_qkv` | 1 | -1.617 | demotable:1 |
| `osmosis-qwen36-27b/ffn_down` | 5 | -0.235 | demotable:2, sacred:1, tolerant:2 |
| `osmosis-qwen36-27b/ffn_gate` | 5 | -0.366 | demotable:3, tolerant:2 |
| `osmosis-qwen36-27b/ffn_up` | 1 | -0.688 | demotable:1 |
| `osmosis-qwen36-27b/other` | 1 | -0.166 | tolerant:1 |
| `osmosis-qwen36-27b/ssm_alpha` | 3 | -0.028 | tolerant:3 |
| `osmosis-qwen36-27b/ssm_beta` | 4 | -0.025 | tolerant:4 |
| `osmosis-qwen36-27b/ssm_out` | 1 | 0.213 | tolerant:1 |

## Layer Buckets

| Model / Bucket | Count | Median Δ% | Classes |
| --- | ---: | ---: | --- |
| `osmosis-gemma4-26b/early` | 14 | 0.629 | critical:5, demotable:2, sacred:1, tolerant:6 |
| `osmosis-gemma4-26b/late` | 19 | 16.899 | critical:10, demotable:5, sacred:2, tolerant:2 |
| `osmosis-gemma4-26b/mid` | 16 | 2.866 | critical:8, demotable:4, tolerant:4 |
| `osmosis-gemma4-e4b/early` | 10 | 0.023 | tolerant:10 |
| `osmosis-gemma4-e4b/late` | 11 | -0.296 | critical:1, demotable:6, tolerant:4 |
| `osmosis-gemma4-e4b/mid` | 5 | 0.613 | demotable:2, sacred:1, tolerant:2 |
| `osmosis-qwen35-9b/early` | 68 | 0.143 | critical:1, demotable:24, sacred:12, tolerant:31 |
| `osmosis-qwen35-9b/late` | 69 | 0.198 | sacred:1, tolerant:68 |
| `osmosis-qwen35-9b/mid` | 63 | 0.190 | demotable:6, sacred:6, tolerant:51 |
| `osmosis-qwen35-9b/unknown` | 2 | 4.707 | critical:1, tolerant:1 |
| `osmosis-qwen36-27b/early` | 9 | -0.028 | demotable:3, tolerant:6 |
| `osmosis-qwen36-27b/late` | 9 | -0.048 | demotable:2, sacred:2, tolerant:5 |
| `osmosis-qwen36-27b/mid` | 5 | -0.235 | demotable:2, tolerant:3 |

## Most Demotable

| Model | Tensor | Group | Δ% |
| --- | --- | --- | ---: |
| `osmosis-qwen35-9b` | `blk.3.attn_v.weight` | `attn_v` | -5.476 |
| `osmosis-gemma4-26b` | `blk.23.attn_k.weight` | `attn_k` | -3.838 |
| `osmosis-qwen35-9b` | `blk.3.attn_q.weight` | `attn_q` | -3.450 |
| `osmosis-qwen35-9b` | `blk.2.attn_qkv.weight` | `attn_qkv` | -3.416 |
| `osmosis-qwen35-9b` | `blk.10.attn_gate.weight` | `other` | -2.794 |
| `osmosis-gemma4-26b` | `blk.18.attn_k.weight` | `attn_k` | -2.762 |
| `osmosis-gemma4-26b` | `blk.1.attn_k.weight` | `attn_k` | -2.571 |
| `osmosis-gemma4-26b` | `blk.28.attn_k.weight` | `attn_k` | -2.521 |
| `osmosis-qwen35-9b` | `blk.3.attn_output.weight` | `attn_output` | -2.076 |
| `osmosis-qwen36-27b` | `blk.2.ffn_gate.weight` | `ffn_gate` | -1.777 |
| `osmosis-qwen35-9b` | `blk.0.ffn_gate.weight` | `ffn_gate` | -1.733 |
| `osmosis-qwen36-27b` | `blk.32.attn_qkv.weight` | `attn_qkv` | -1.617 |
| `osmosis-qwen35-9b` | `blk.1.ffn_down.weight` | `ffn_down` | -1.447 |
| `osmosis-qwen35-9b` | `blk.6.attn_gate.weight` | `other` | -1.435 |
| `osmosis-qwen35-9b` | `blk.9.ffn_up.weight` | `ffn_up` | -1.384 |
| `osmosis-gemma4-26b` | `blk.6.attn_k.weight` | `attn_k` | -1.319 |
| `osmosis-gemma4-26b` | `blk.17.attn_k.weight` | `attn_k` | -1.230 |
| `osmosis-qwen35-9b` | `blk.4.attn_qkv.weight` | `attn_qkv` | -1.228 |
| `osmosis-gemma4-e4b` | `blk.15.ffn_gate.weight` | `ffn_gate` | -1.155 |
| `osmosis-gemma4-e4b` | `blk.40.attn_q.weight` | `attn_q` | -1.153 |
| `osmosis-qwen36-27b` | `blk.34.ffn_down.weight` | `ffn_down` | -1.146 |
| `osmosis-gemma4-e4b` | `blk.41.ffn_up.weight` | `ffn_up` | -1.144 |
| `osmosis-qwen35-9b` | `blk.6.attn_qkv.weight` | `attn_qkv` | -1.122 |
| `osmosis-qwen35-9b` | `blk.2.ssm_out.weight` | `ssm_out` | -1.104 |
| `osmosis-qwen35-9b` | `blk.12.ffn_down.weight` | `ffn_down` | -1.080 |

## Most Sensitive

| Model | Tensor | Group | Δ% |
| --- | --- | --- | ---: |
| `osmosis-qwen35-9b` | `blk.0.ssm_out.weight` | `ssm_out` | 81.630 |
| `osmosis-gemma4-26b` | `blk.29.ffn_gate_up_exps.weight` | `ffn_gate_up_exps` | 37.503 |
| `osmosis-gemma4-26b` | `blk.15.ffn_down_exps.weight` | `ffn_down_exps` | 33.559 |
| `osmosis-gemma4-26b` | `blk.20.ffn_down_exps.weight` | `ffn_down_exps` | 31.507 |
| `osmosis-gemma4-26b` | `blk.25.ffn_down_exps.weight` | `ffn_down_exps` | 31.011 |
| `osmosis-gemma4-26b` | `blk.0.attn_o.weight` | `attn_o` | 30.701 |
| `osmosis-gemma4-26b` | `blk.10.attn_o.weight` | `attn_o` | 30.701 |
| `osmosis-gemma4-26b` | `blk.20.attn_o.weight` | `attn_o` | 30.701 |
| `osmosis-gemma4-26b` | `blk.25.attn_o.weight` | `attn_o` | 30.701 |
| `osmosis-gemma4-26b` | `blk.29.attn_o.weight` | `attn_o` | 30.701 |
| `osmosis-gemma4-26b` | `blk.5.attn_o.weight` | `attn_o` | 30.538 |
| `osmosis-gemma4-26b` | `blk.29.ffn_down_exps.weight` | `ffn_down_exps` | 29.931 |
| `osmosis-gemma4-26b` | `blk.20.ffn_gate_up_exps.weight` | `ffn_gate_up_exps` | 29.542 |
| `osmosis-gemma4-26b` | `blk.10.ffn_down_exps.weight` | `ffn_down_exps` | 28.590 |
| `osmosis-gemma4-26b` | `blk.25.ffn_gate_up_exps.weight` | `ffn_gate_up_exps` | 27.914 |
| `osmosis-gemma4-26b` | `blk.10.ffn_gate_up_exps.weight` | `ffn_gate_up_exps` | 27.907 |
| `osmosis-gemma4-26b` | `blk.5.ffn_down_exps.weight` | `ffn_down_exps` | 27.821 |
| `osmosis-gemma4-26b` | `blk.15.ffn_gate_up_exps.weight` | `ffn_gate_up_exps` | 25.931 |
| `osmosis-gemma4-26b` | `blk.15.attn_o.weight` | `attn_o` | 25.186 |
| `osmosis-gemma4-26b` | `blk.0.ffn_down_exps.weight` | `ffn_down_exps` | 21.456 |
| `osmosis-gemma4-26b` | `blk.29.attn_k.weight` | `attn_k` | 16.899 |
| `osmosis-gemma4-26b` | `blk.5.attn_k.weight` | `attn_k` | 10.706 |
| `osmosis-gemma4-e4b` | `blk.41.attn_q.weight` | `attn_q` | 9.867 |
| `osmosis-qwen35-9b` | `output.weight` | `output` | 9.264 |
| `osmosis-gemma4-26b` | `blk.11.attn_k.weight` | `attn_k` | 7.599 |
