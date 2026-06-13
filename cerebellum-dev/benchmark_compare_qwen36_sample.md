# Benchmark Comparison

| Model | arc_challenge | hellaswag | humaneval | mmlu_redux |
| --- | ---: | ---: | ---: | ---: |
| `cerebellum_v2` | 84.81 | 75.02 | 63.40 | 57.29 |
| `cerebellum_v2b` | 85.67 | 75.34 | 68.30 | 58.42 |
| `cerebellum_v4_fixed` | 94.71 | 92.21 | 81.10 | 76.88 |

## Deltas vs `cerebellum_v2`

| Model | arc_challenge | hellaswag | humaneval | mmlu_redux |
| --- | ---: | ---: | ---: | ---: |
| `cerebellum_v2` | +0.00 (+0.0%) | +0.00 (+0.0%) | +0.00 (+0.0%) | +0.00 (+0.0%) |
| `cerebellum_v2b` | +0.85 (+1.0%) | +0.32 (+0.4%) | +4.90 (+7.7%) | +1.12 (+2.0%) |
| `cerebellum_v4_fixed` | +9.90 (+11.7%) | +17.19 (+22.9%) | +17.70 (+27.9%) | +19.58 (+34.2%) |

## ASCII Bars

### arc_challenge
`cerebellum_v2                 ` ####################.... 84.81
`cerebellum_v2b                ` #####################... 85.67
`cerebellum_v4_fixed           ` #######################. 94.71

### hellaswag
`cerebellum_v2                 ` ##################...... 75.02
`cerebellum_v2b                ` ##################...... 75.34
`cerebellum_v4_fixed           ` ######################.. 92.21

### humaneval
`cerebellum_v2                 ` ###############......... 63.40
`cerebellum_v2b                ` ################........ 68.30
`cerebellum_v4_fixed           ` ###################..... 81.10

### mmlu_redux
`cerebellum_v2                 ` ##############.......... 57.29
`cerebellum_v2b                ` ##############.......... 58.42
`cerebellum_v4_fixed           ` ##################...... 76.88

## Sources

- `cerebellum_v2` / `arc_challenge`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v2_arc_results.json` via `accuracy` (994/1172); audit=ok
- `cerebellum_v2` / `hellaswag`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v2_hellaswag_results.json` via `accuracy` (7534/10042); audit=ok
- `cerebellum_v2` / `humaneval`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v2_humaneval_results.json` via `pass_at_1_pct`; audit=ok
- `cerebellum_v2` / `mmlu_redux`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v2_mmlu_redux_results.json` via `accuracy` (1375/2400); audit=ok
- `cerebellum_v2b` / `arc_challenge`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v2b_arc_results.json` via `accuracy` (1004/1172); audit=ok
- `cerebellum_v2b` / `hellaswag`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v2b_hellaswag_results.json` via `accuracy` (7566/10042); audit=ok
- `cerebellum_v2b` / `humaneval`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v2b_humaneval_results.json` via `pass_at_1_pct`; audit=ok
- `cerebellum_v2b` / `mmlu_redux`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v2b_mmlu_redux_results.json` via `accuracy` (1402/2400); audit=ok
- `cerebellum_v4_fixed` / `arc_challenge`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_arc_results.json` via `accuracy` (1110/1172); audit=ok
- `cerebellum_v4_fixed` / `hellaswag`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_hellaswag_results.json` via `accuracy` (9260/10042); audit=ok
- `cerebellum_v4_fixed` / `humaneval`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_humaneval_results.json` via `pass_at_1_pct`; audit=ok
- `cerebellum_v4_fixed` / `mmlu_redux`: `osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_mmlu_redux_results.json` via `accuracy` (1845/2400); audit=ok
