# OG Cerebellum Reconstruction Guides — 2026-06-07

Private research note. Keep this in `cerebellum-dev`; do not push to public `origin`.

This reconstructs the local evidence for the successful OG Cerebellum builds and
the later Gemma 4/Qwen 3.6 variants. The important pattern is that the successful
models were not one identical workflow. Qwen 3.6 27B v4 used tensor-level
ablation plus budgeted promotion. Qwen 3.6 35B-A3B v3 used group/reverse
ablation over a Q3_K_M base and a simple 360-entry demotion recipe. Gemma 4 26B
used a hand-evolved 91-entry override map over a Q3_K_M base, then benchmark and
template audits.

> ## ⚠ CORRECTION (2026-06-13): this guide dropped the CODING ABLATION phase
>
> This reconstruction (and `docs/multi_domain_ablation.md`) flattened the real winning
> method by collapsing the per-tensor/group sieve and the budget stage together and
> describing the signal as PPL (later multi-domain PPL). **The real 27B v4 pipeline had a
> distinct CODING ABLATION phase between the PPL sieve and the budget allocation** — for
> each tensor group, demote to Q2_K and run **REAL HumanEval pass@1** (temp 0, 164
> problems), then drill the coding-critical groups by layer band. The budget stage then
> *protected* the coding-critical tensors/layers.
>
> Why it matters: on 27B, demoting `attn_qkv` to Q2_K moved PPL <1% but dropped HumanEval
> **75.0% → 28.7% (-46 pts)**. PPL — including multi-domain PPL — is blind to coding
> collapse. Multi-domain-PPL-only is **INSUFFICIENT** for any model where coding matters.
>
> Primary proof: `osmosis-qwen36-27b/coding_ablation/coding_ablation.log` and
> `osmosis-qwen36-27b/coding_ablation_layers/layer_ablation.log`. Full reconstruction:
> `knowledge/REAL_PIPELINE_RECONSTRUCTED.md`. Canonical method:
> `knowledge/CURRENT_METHOD.md` step 6. Runnable script: `scripts/coding_ablation.py`.
> Mandate: `knowledge/CRITICAL_LOST_STEP_2026-06-13.md`. Treat the "tensor-level ablation
> plus budgeted promotion" phrasing below as shorthand for: **PPL sieve → coding ablation
> (HumanEval per group, then per layer band) → budget allocation protecting the coding
> tensors.**

## Session Logs and Name Mapping

The old Claude/Cerebellum chats are recoverable even when the UI cannot continue
the sessions. They are JSONL transcript files containing user text, assistant
text, tool calls, and tool results.

Primary recovered transcript files:

- `/var/home/deucebucket/.claude/projects/-var-home-deucebucket-ai-drive-osmosis/f866b942-19bd-469c-893c-fc013b0d80c1.jsonl`
- `/var/home/deucebucket/.claude/projects/-var-home-deucebucket-ai-drive-cerebellum/1809b34c-a7fd-4b73-86de-0ba5fc158987.jsonl`
- `/var/home/deucebucket/.claude/projects/-var-home-deucebucket-ai-drive-cerebellum/5bddbf71-de3a-4f1e-98a8-e7a5ac013466.jsonl`

Useful extraction commands:

```bash
# Bash/tool chronology from old Osmosis session
jq -r 'select(.type=="assistant") | .timestamp as $ts | .message.content[]? |
  select(.type=="tool_use" and .name=="Bash") |
  [$ts, .input.description, .input.command] | @tsv' \
  /var/home/deucebucket/.claude/projects/-var-home-deucebucket-ai-drive-osmosis/f866b942-19bd-469c-893c-fc013b0d80c1.jsonl \
  > /tmp/osmosis_session_bash.tsv

# Chat text from old Osmosis session
jq -r 'select(.type=="user" or .type=="assistant") | .timestamp as $ts |
  .type as $typ |
  if $typ=="user" then
    [$ts,$typ,(.message.content // .text // "" |
      if type=="array" then
        map(if type=="object" and .type=="text" then .text else tostring end) |
        join("\n")
      else tostring end)]
  elif $typ=="assistant" then
    [$ts,$typ,(.message.content[]? | select(.type=="text") | .text)]
  else empty end | @tsv' \
  /var/home/deucebucket/.claude/projects/-var-home-deucebucket-ai-drive-osmosis/f866b942-19bd-469c-893c-fc013b0d80c1.jsonl \
  > /tmp/osmosis_session_chat.tsv

# Repeat for current Cerebellum logs
jq -r 'select(.type=="assistant") | .timestamp as $ts | .message.content[]? |
  select(.type=="tool_use" and .name=="Bash") |
  [$ts, .input.description, .input.command] | @tsv' \
  /var/home/deucebucket/.claude/projects/-var-home-deucebucket-ai-drive-cerebellum/*.jsonl \
  > /tmp/cerebellum_session_bash.tsv
```

Name mapping:

- `osmosis` was the old repo/package identity and the imatrix/engine tooling.
- `cerebellum` became the model-pipeline/release identity.
- Old paths like `ai-drive/osmosis/osmosis-qwen36-27b/osmosis_imatrix.dat`
  and later paths like `ai-drive/cerebellum/osmosis-qwen36-27b/cerebellum_imatrix.dat`
  can refer to the same experiment lineage after rename/migration.

## Imatrix Inventory Found Locally

Large/current imatrix files found on 2026-06-07:

| Size | Path | Notes |
|---:|---|---|
| 358,906,272 | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen35-122b/imatrix_unsloth.gguf` | imported Unsloth-style imatrix |
| 192,223,904 | `/var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file` | Qwen 3.6 35B v3/v2 lineage, imported |
| 108,152,016 | `/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_imatrix.dat` | Qwen 3.6 35B locally generated |
| 56,941,536 | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf` | Gemma 4 26B main v6/heretic imatrix |
| 14,759,676 | `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat` | Qwen 35/9B MoE stage artifact |
| 13,582,641 | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/cerebellum_imatrix.dat` | Qwen 3.6 27B v4 lineage; renamed from `osmosis_imatrix.dat` in later cleanup |
| 7,441,037 | `/var/home/deucebucket/games/cerebellum-pipeline-tmp/gemma4-12b/imatrix.dat` | Gemma 4 12B current pipeline tmp |
| 5,120,129 | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen35-9b/cerebellum_imatrix.dat` | Qwen 3.5 9B |
| 4,810,957 | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-e4b/imatrix.dat` | Gemma 4 E4B |
| 3,792,439 | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen35-122b/cerebellum_imatrix.dat` | Qwen 122B local |
| 2,466,952 | `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/imatrix.dat` | smaller/older Gemma 4 26B local imatrix; not the main shipped v6 one |

### Imatrix Validity Check

Checked on 2026-06-07:

- Legacy `.dat` files parse cleanly: full byte consumption, sane entry counts,
  no NaN/Inf values, no all-zero entries.
- Imported `.gguf`/`gguf_file` imatrixes are valid GGUF imatrix files
  (`general.type = imatrix`) and were loaded by llama.cpp in historical
  quantization logs.

Validated local legacy files:

| File | Entries | Chunks/ncall | Verdict |
|---|---:|---:|---|
| `osmosis-qwen36-27b/cerebellum_imatrix.dat` | 496 | 8 | Valid. Used by Qwen 3.6 27B lineage. |
| `games/qwen36-35b-v2/cerebellum_imatrix.dat` | 470 | 1 | Valid. Covers MoE/shared expert/router/SSM/full-attn groups. |
| `games/cerebellum-pipeline-tmp/gemma4-12b/imatrix.dat` | 328 | 1 | Valid format. Weight-only; benchmark behavior still must decide quality. |
| `osmosis-gemma4-e4b/imatrix.dat` | 380 | 1 | Valid and includes PLE tensors. |
| `osmosis-qwen35-9b/cerebellum_imatrix.dat` | 248 | 1 | Valid hybrid SSM coverage. |
| `osmosis-gemma4-26b/imatrix.dat` | 205 | 1 | Valid format but incomplete for MoE; do not use as the main Gemma 26B release imatrix. |

Validated imported GGUF imatrix files:

| File | GGUF tensors | Chunks | Dataset | Verdict |
|---|---:|---:|---|---|
| `games/qwen36-35b-v2/imatrix_unsloth.gguf_file` | 1020 | 76 | `unsloth_calibration_Qwen3.6-35B-A3B.txt` | Valid imported calibration imatrix. |
| `osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf` | 590 | 822 | `/training_dir/calibration_datav5.txt` | Valid imported calibration imatrix; used by shipped Gemma 26B v4-v7/heretic. |
| `osmosis-qwen35-122b/imatrix_unsloth.gguf` | 1224 | 76 | `unsloth_calibration_Qwen3.5-122B-A10B.txt` | Valid imported calibration imatrix. |

Important quality distinction:

- "Valid" means llama.cpp can parse/load the imatrix and the tensors have sane
  importance arrays.
- "Good" means the coverage and calibration objective match the architecture and
  downstream task. Weight-only imatrixes are useful, but for MoE releases the
  imported calibration imatrixes with expert coverage were stronger.
- The old Gemma 26B local `imatrix.dat` is the cautionary case: it is valid, but
  only covers 205/658-ish dense-path tensors and misses the MoE expert path.
  The successful Gemma 26B builds used `google_gemma-4-26B-A4B-it-imatrix.gguf`
  with 295 entries / 822 chunks instead.

### Final-Version Imatrix Stack

This is the practical comparison of what the successful/final models actually
used.

| Model lineage | Final/relevant build | Imatrix source | Entry/chunk signal | Assessment |
|---|---|---|---|---|
| Qwen 3.6 27B v4 | `qwen3.6-27b-cerebellum-v4.gguf` | Our legacy `cerebellum_imatrix.dat` (renamed from `osmosis_imatrix.dat`) | 496 entries, `ncall=8`, dataset `osmosis-sensitivity` | Best OG pattern. This matches the older calibrated/activation-blended generator path rather than the later pure streaming default. |
| Gemma 4 E4B v1/v2 | `Gemma-4-E4B-it-Cerebellum-v*` | Our weight-stat `imatrix.dat` | 380 entries, includes PLE (`inp_gate`, `proj`, globals), `ncall=1` | Good for this dense/PLE architecture. PLE protection mattered more than activation calibration. |
| Gemma 4 26B v4-v7 / Heretic | regular v6/v6.1, heretic v1/v1.1 | Imported `google_gemma-4-26B-A4B-it-imatrix.gguf` from bartowski/ggml lineage | GGUF imatrix, 590 tensors, 295 loaded entries, 822 chunks | Correct final path. Our local 205-entry Gemma 26B imatrix was valid but under-covered MoE experts and was worse than no imatrix in local notes. |
| Qwen 3.6 35B-A3B v2/v3 | `Qwen3.6-35B-A3B-Cerebellum-v3.gguf` | Evidence points to imported `imatrix_unsloth.gguf_file` for v3/coder lineage; local `cerebellum_imatrix.dat` also exists | imported: 1020 tensors / 76 chunks; local: 470 entries / `ncall=1` | Local file has good full-coverage weight-stat mapping; imported file is stronger calibration/coder-lineage candidate and was used in the reconstructed v3 command. |
| Gemma 4 12B current | current `targeted-attnv-earlyblocks` outputs | Our weight-stat `imatrix.dat` | 328 entries, `ncall=1` | Valid format, but not enough proof of quality. The failure was mostly accepting a PPL survivor map without benchmark gate, not file corruption. |

### What Changed From Osmosis to Cerebellum

There was no single binary-format break. The legacy `.dat` format stayed usable
and llama.cpp loads both that and newer GGUF imatrix files.

What changed:

- Naming: new files should be `cerebellum_imatrix.dat`; older files and metadata
  often say `osmosis_imatrix.dat` / `osmosis-sensitivity`.
- CLI wrapping: public docs moved from `python -m osmosis.imatrix_stream` to
  `cerebellum imatrix`, while the old modules remain compatibility internals.
- Default generation emphasis: the modern streaming path is weight-stat
  sensitivity (`L2_norm * max_abs * variance`) with `ncall=1`; the older Qwen 27B
  artifact has `ncall=8`, matching the calibrated generator path.
- Architecture caution increased: after Gemma 26B, local memory explicitly says
  incomplete MoE imatrix coverage can be worse than no imatrix, so MoE final
  builds should use either full expert coverage or an imported calibrated
  imatrix.

Core method stayed the same:

1. Use imatrix in `llama-quantize` so base quantization is importance-guided.
2. Build candidate tensor maps.
3. Run PPL ablations / reverse ablations to decide which tensor changes survive.
4. Run downstream benchmark gates before publishing.

The old loved models were not "imatrix only"; they were imatrix-guided builds
plus PPL-driven tensor-map search, then benchmark validation.

## Cross-Model Lessons

- Do not treat PPL-only wins as release gates. The OG releases had downstream
  benchmark checks and later audit corrections.
- Qwen 3.6 no-thinking benchmarks require reasoning disabled. Local notes say
  reasoning HumanEval collapsed while no-thinking HumanEval was strong.
- Gemma 4 HumanEval must use chat-template evaluation with thinking disabled.
  Raw completions produced false lows.
- The current Gemma 4 12B failure differs from OG Gemma/Qwen because it accepted
  a wiki-PPL survivor map with `attn_v` and broad early-block Q2 demotions before
  benchmark gating.
- The OG Gemma 4 26B map did not demote `attn_v`/`attn_output` globally. The
  surviving v6 map mainly touches selected `attn_q`, many `attn_k`, selected
  `ffn_up`, and `ffn_gate_up_exps`.

## Qwen 3.6 27B — Cerebellum v4

### Evidence

- Working dir: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/`
- Source HF snapshot: `/var/tmp/osmosis-qwen36/Qwen3.6-27B`
- F16 GGUF input: `/var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf`
- Final local model evidence:
  - `/var/home/deucebucket/ai-drive/models/qwen3.6-27b-cerebellum-v4.gguf`
  - logs also reference `/var/home/deucebucket/games/osmosis-quants/qwen3.6-27b-cerebellum-v4.gguf`
- Final tensor file: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/tensor_types_v4_12gb.txt`
- Ablation evidence:
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_results.json`
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_plan.json`
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/interaction_results.json`
  - `/var/home/deucebucket/ai-drive/cerebellum/start.md` records the completed 448-tensor ablation.
- Benchmark correction note:
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/benchmark_results/BENCHMARK_CORRECTIONS.md`

### What The Public Card Compresses

The model card says every tensor was individually crushed, but the local
evidence is more specific:

- `ablation_results.json` contains 23 measured PPL probes.
- `ablation_plan.json` contains 23 planned probes selected from KL/damage
  heuristics: highest/lowest KL2, highest/lowest damage, and role/layer probes.
- `tensor_types_v4_12gb.txt` contains 181 allocator overrides.
- The 181-line map exactly matches the model-card distribution:
  - `q8_0`: 7
  - `q6_K`: 41
  - `q5_K`: 70
  - `q4_K`: 22
  - `q3_K`: 19
  - `q2_K`: 22

So the exact v4 recipe was sparse PPL probe + allocator extrapolation, not a
full 851-tensor brute-force sweep. The shipped GGUF metadata still says 851
tensors because that is the model tensor count, not the count of measured
individual ablations.

### Build Recipe

The exact final shell command was not found, but logs and GGUF metadata support
this reconstruction:

```bash
python /var/home/deucebucket/ai-drive/llama.cpp/convert_hf_to_gguf.py \
  /var/tmp/osmosis-qwen36/Qwen3.6-27B \
  --outfile /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  --outtype f16

/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --imatrix /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/osmosis_imatrix.dat \
  --tensor-type-file /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/tensor_types_v4_12gb.txt \
  /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  qwen3.6-27b-cerebellum-v4.gguf \
  Q2_K
```

### Final Recipe Shape

- Base quant: `Q2_K`.
- Override count: 181 entries.
- Budget allocator command shape:

```bash
python -m osmosis.cerebellum \
  --ablation /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_results.json \
  --plan /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/ablation_plan.json \
  --source-gguf /var/tmp/osmosis-qwen36/qwen3.6-27b-f16.gguf \
  --budget-gb 12.0 \
  --base-type Q2_K \
  --imatrix /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/osmosis_imatrix.dat \
  --quantize-bin /var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --output /var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/tensor_types_v4_12gb.txt \
  -v
```

- Metadata distribution from PPL logs:
  - F32: 353 tensors
  - Q8_0: 7 tensors
  - Q2_K: 253 tensors
  - Q3_K: 77 tensors
  - Q4_K: 49 tensors
  - Q5_K: 70 tensors
  - Q6_K: 42 tensors
- This was a tensor-level promoted budget map, not a simple group demotion map.
- The model-card "sacred/demotable" examples match the local
  `ablation_results.json` baseline `8.2556` and the local probe tensors.

### Gates

Use corrected/audited result files, not early published numbers:

- Fixed HumanEval: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_humaneval_results.json`
- Fixed ARC: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_arc_results.json`
- Fixed HellaSwag: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_hellaswag_results.json`
- Fixed MMLU-Redux: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/benchmark_results/cerebellum_v4_fixed_mmlu_redux_results.json`
- Full MMLU: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-27b/benchmark_results/cerebellum_v4_mmlu_results.json`

Observed corrected anchors:

- HumanEval: 81.1
- HellaSwag: 92.21
- MMLU-Redux: 76.875
- Full MMLU: 82.52

Gotcha: benchmark folders contain stale and corrected results. Read
`BENCHMARK_CORRECTIONS.md` before publishing numbers.

## Qwen 3.6 35B-A3B — Cerebellum v3

### Evidence

- Source/model card dir: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-35b/`
- Game-drive build dir: `/var/home/deucebucket/games/qwen36-35b-v2/`
- Final GGUF: `/var/home/deucebucket/games/qwen36-35b-v2/Qwen3.6-35B-A3B-Cerebellum-v3.gguf`
- Final override file: `/var/home/deucebucket/games/qwen36-35b-v2/cerebellum_v3_overrides.txt`
- Local model card: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-35b/README.md`
- Main devlog: `/var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/DEVLOG_2026-05-01_qwen36_35b_start.md`
- Ablation logs:
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-35b/ablation/group_ablation_results.log`
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-35b/ablation/reverse_ablation_results.log`
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-35b/ablation/perlayer_ablation_results.log`
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-qwen36-35b/ablation/router_surgery_results.log`

### Build Recipe

The exact final v3 shell line was not found. The evidence-supported command
shape is:

```bash
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --imatrix /var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file \
  --tensor-type-file /var/home/deucebucket/games/qwen36-35b-v2/cerebellum_v3_overrides.txt \
  source-f16-or-bf16.gguf \
  /var/home/deucebucket/games/qwen36-35b-v2/Qwen3.6-35B-A3B-Cerebellum-v3.gguf \
  Q3_K_M
```

### Final Recipe Shape

- Base quant: `Q3_K_M`.
- Imatrix: Unsloth coder imatrix,
  `/var/home/deucebucket/games/qwen36-35b-v2/imatrix_unsloth.gguf_file`.
- Override count: 360 entries.
- Pattern: 40 layers x 9 tensor types, all `Q2_K`:
  - `ffn_gate_exps`
  - `ffn_up_exps`
  - `ffn_down_exps`
  - `ffn_gate_shexp`
  - `ffn_up_shexp`
  - `ffn_down_shexp`
  - `attn_gate`
  - `ssm_alpha`
  - `ssm_beta`
- Protected/default at Q3_K_M included `attn_qkv` and `ssm_out`.

Version evolution from local override files:

| Variant | Override file | Lines | Meaning |
|---|---|---:|---|
| v1 legacy | `osmosis-qwen36-35b/cerebellum_v1_overrides.txt` | 400 | Crushed 10 groups, including `attn_qkv`. |
| v2 intermediate | `osmosis-qwen36-35b/cerebellum_v2_overrides.txt` | 280 | Crushed 7 groups; kept `attn_qkv`, `ffn_up_exps`, and `ffn_down_exps` at baseline. |
| v3 recommended | `games/qwen36-35b-v2/cerebellum_v3_overrides.txt` | 360 | Crushed 9 groups; restored only `attn_qkv`, kept expert/shared/mixing groups at Q2_K. |

### Ablation Logic

- Forward group ablation over Q3_K_M baseline PPL `7.1758` found `ssm_out`
  was the most sensitive tested group:
  - `ssm_out Q2_K`: PPL `7.4132` (`+0.2374`)
  - `attn_qkv Q2_K`: PPL `7.2766` (`+0.1008`)
  - `ffn_down_exps Q2_K`: PPL `7.3136` (`+0.1378`)
  - `ssm_alpha Q2_K`: PPL `7.1753` (`-0.0005`)
  - `ssm_beta Q2_K`: PPL `7.1803` (`+0.0045`)
- Reverse ablation from v1 PPL `7.8484` found `attn_qkv`,
  `ffn_down_exps`, and `ffn_up_exps` benefited from un-demotion by PPL, but
  downstream benchmarks selected the v3 compromise: protect `attn_qkv`, keep
  expert/shared/mixing groups at Q2_K regularization.
- Per-layer ablation did not produce a surgical layer recipe.
- Router surgery had no useful signal.

This is the clearest OG example where PPL got worse in places but downstream
benchmarks improved. Do not overfit PPL here; the card's real claim is
benchmark-selected Q2_K regularization over gate/mixing/shared expert groups.

### Gates

- v3 benchmark dir: `/var/home/deucebucket/games/qwen36-35b-v2/benchmark_results_v3/`
- Q3_K_M baseline dir: `/var/home/deucebucket/games/qwen36-35b-v2/benchmark_results_baseline/`

Observed v3 anchors:

- ARC: 95.82
- HumanEval base/plus: 70.73 / 65.24
- HellaSwag: 92.28
- MMLU-Redux: 75.0

Baseline Q3_K_M anchors:

- ARC: 96.08
- HumanEval base/plus: 64.02 / 56.71
- HellaSwag: 91.49
- MMLU-Redux: 74.12

## Qwen 3.6 35B-A3B Heretic

### Evidence

- Devlog: `/var/home/deucebucket/games/cerebellum-heretic-qwen36-35b/DEVFLOG.md`
- Override file: `/var/home/deucebucket/games/cerebellum-heretic-qwen36-35b/v3_overrides.txt`
- Output dir: `/var/home/deucebucket/games/cerebellum-heretic-qwen36-35b/`

### Build Recipe

This was a transfer/reuse of the stock 35B v3 recipe onto a Heretic GGUF:

```bash
llama-quantize \
  --allow-requantize \
  --tensor-type-file v3_overrides.txt \
  source.gguf \
  out.gguf \
  Q2_K
```

Notes from the devlog:

- Source: `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF`
- Reused stock `cerebellum_v3_overrides.txt`.
- 360 entries, same 40 x 9 Q2_K pattern.
- Server gate: `llama-server --n-gpu-layers 99 --ctx-size 24576 --parallel 4 --reasoning off --reasoning-budget 0`.

This model regressed hard versus stock v3. Treat it as diagnostic, not the
canonical OG recipe.

## Gemma 4 26B-A4B Regular — Cerebellum v6/v6.1

### Evidence

- Working dir: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/`
- Source BF16 GGUF: `/var/home/deucebucket/games/models/gemma-4-26B-A4B-it-bf16.gguf`
- Imatrix: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf`
- Final override file: `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/cerebellum_v6_overrides.txt`
- Final packaged v6.1:
  - `/var/home/deucebucket/games/models/Gemma-4-26B-A4B-Cerebellum-v6/gemma-4-26B-A4B-it-cerebellum-v6.1-templatefix.gguf`
  - `/var/home/deucebucket/games/models/Gemma-4-26B-A4B-Cerebellum-v6/gemma-4-26b-a4b-it.mmproj.gguf`
- Main logs/cards:
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/FULL_EXPERIMENT_LOG.md`
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/research_log.md`
  - `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/hf_cards/README_regular_v6_1_templatefix.md`

### Lineage From The Logs

Gemma 26B was not the Qwen 27B sparse-budget allocator. It was a MoE-specific
hand evolution:

1. PLE sweep failed because the PLE-like auxiliary tensors were already F32.
2. Local 205-entry imatrix was valid but harmful/incomplete for MoE; switched
   to bartowski `google_gemma-4-26B-A4B-it-imatrix.gguf`.
3. Group ablation from Q3_K_M+bartowski imatrix:
   - `expert_gate_up`: PPL improved by -5.5 percent at Q2_K.
   - `attn_q`: +13.4 percent, protected.
   - `attn_k`: -12.1 percent, strongly demotable.
   - `ffn_gate`: -1.2 percent, tolerant.
   - `ffn_up`: -18.2 percent, most demotable.
4. v1: 120 overrides, `attn_q -> Q5_K`, `ffn_up/attn_k/ffn_gate_up_exps -> Q2_K`.
5. v2: 90 overrides, kept `ffn_gate_up_exps` at default; lost the useful
   regularization and was worse.
6. v3: 99 overrides, layer-level `attn_q` pruning to 9 important layers.
7. v4: 91 overrides, further task-balanced edits.
8. v5: 91 overrides, un-demoted 7 `attn_k` layers to Q3_K; PPL improved but
   HumanEval/MMLU regressed, so it was not the general release winner.
9. v6: 91 overrides, selected `attn_k` changes plus router layer 8 Q8_0 road-map
   improvement; v6.1 kept the allocation and fixed template/runtime metadata.

Override count/type distribution:

| File | Overrides | Distribution |
|---|---:|---|
| `cerebellum_v1_overrides.txt` | 120 | `Q2_K=90`, `Q5_K=30` |
| `cerebellum_v2_overrides.txt` | 90 | `Q2_K=60`, `Q5_K=30` |
| `cerebellum_v3_overrides.txt` | 99 | `Q2_K=90`, `Q5_K=9` |
| `cerebellum_v4_overrides.txt` | 91 | `Q2_K=82`, `Q5_K=9` |
| `cerebellum_v5_overrides.txt` | 91 | `Q2_K=75`, `Q3_K=7`, `Q5_K=9` |
| `cerebellum_v6_overrides.txt` | 91 | `Q2_K=78`, `Q4_K=4`, `Q5_K=9` |

### Build Recipe

```bash
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --imatrix /var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf \
  --tensor-type-file /var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/cerebellum_v6_overrides.txt \
  /var/home/deucebucket/games/models/gemma-4-26B-A4B-it-bf16.gguf \
  out.gguf \
  Q3_K_M
```

If reproducing the shipped router-surgery variant, apply the router change
afterward with `scripts/gguf_tensor_surgery.py`. Local road-map evidence says:

- Layer 8 router `F32 -> Q8_0` was the universal v6 winner:
  PPL 12,054 vs 12,356 baseline, HumanEval 72.0 vs 71.3, ARC/HellaSwag stable.
- Layer 10 router `F32 -> Q8_0` was a diagnostic "coding road": PPL improved,
  but HumanEval fell to 61.6, so it was not the shipped general choice.
- Stacking multiple "improving" routers made PPL worse; router surgery does not
  stack additively.
- K-quants on routers (`Q6_K`, `Q2_K`) broke routing. `Q8_0` was the only safe
  router-demotion format found.

Router evidence:

- `/var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/DEVLOG_2026-05-01_router_road_mapping.md`
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/ablation/surgery/results/router_surgery_ablation.log`
- `/var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/ablation/surgery/road_mapping/router_layer8_humaneval_results.json`

### Final Recipe Shape

- Base quant: `Q3_K_M`.
- Override count: 91.
- Main override families:
  - 9 selected `attn_q` tensors at `Q5_K`.
  - Many `attn_k` tensors at `Q2_K`, with selected layers at `Q4_K`.
  - 22 selected `ffn_up` tensors at `Q2_K`.
  - All `ffn_gate_up_exps` at `Q2_K`.
- This map is not a broad early-block Q2 map and not an `attn_v` demotion map.
- This is why router surgery did not appear in the tiny Qwen3-0.6B dense replay:
  it is a MoE `ffn_gate_inp.weight` path, not a dense Qwen3 tensor family.

### Gates

Historical v6.1 card anchors:

- ARC: 95.5631
- HellaSwag: 84.55
- MMLU: 71.3333

Gemma 4 HumanEval gotcha:

```bash
llama-server --jinja --reasoning auto ...
BENCH_WORKERS=1 BENCH_ENABLE_THINKING=0 BENCH_THINKING_BUDGET=0 \
  python scripts/benchmark_evalplus_chat.py
```

Do not use raw `/v1/completions` for Gemma 4 coding scores.

## Gemma 4 26B-A4B Heretic — Cerebellum v1/v1.1

### Evidence

- Source BF16 GGUF:
  `/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/build/gemma-4-26B-A4B-it-heretic-bf16.gguf`
- Quant log:
  `/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/quantize_cerebellum_v1.log`
- Final artifacts:
  - `/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/gemma-4-26B-A4B-it-heretic-cerebellum-v1.1-templatefix.gguf`
  - `/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf`
- HF release card:
  `/var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/hf_release/README.md`

### Build Recipe

Same v6 override map transferred to the Heretic BF16 source:

```bash
/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize \
  --imatrix /var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/google_gemma-4-26B-A4B-it-imatrix.gguf \
  --tensor-type-file /var/home/deucebucket/ai-drive/cerebellum/osmosis-gemma4-26b/cerebellum_v6_overrides.txt \
  /var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/build/gemma-4-26B-A4B-it-heretic-bf16.gguf \
  /var/home/deucebucket/games/models/Gemma-4-26B-A4B-it-Heretic/gemma-4-26B-A4B-it-heretic-cerebellum-v1.gguf \
  Q3_K_M
```

Local card says 658 tensors loaded and all 91 override entries matched.

### Gates

HF release anchors:

- ARC: 95.48
- HellaSwag: 83.49
- MMLU-Redux: 71.42
- Vision smoke: 6/6
- Refusal: 1/45
- HumanEval+ chat fresh: base 92.07 / plus 89.63

Raw non-chat EvalPlus was a false low. Use the Gemma 4 chat harness.

## Gemma 4 26B Codex Branch

### Evidence

- Working dir: `/var/home/deucebucket/ai-drive/cerebellum/cerebellum-gemma4-codex/`
- Game-drive transfer dir: `/var/home/deucebucket/games/cerebellum-gemma4-codex/`
- Transfer source:
  `/var/home/deucebucket/games/models/gemma4-26b-codex-dwojcik/gemma-4-26B-A4B-it.Q4_K_M.gguf`
- Transfer override file:
  `/var/home/deucebucket/games/cerebellum-gemma4-codex/cerebellum_v6_transfer_overrides.txt`
- Transfer log:
  `/var/home/deucebucket/games/cerebellum-gemma4-codex/quantize_transfer_requant.log`
- Developer log:
  `/var/home/deucebucket/ai-drive/cerebellum/cerebellum-gemma4-codex/DEVELOPER_LOG_20260522.md`

### Transfer Probe Recipe

```bash
llama-quantize \
  --allow-requantize \
  --tensor-type-file /var/home/deucebucket/games/cerebellum-gemma4-codex/cerebellum_v6_transfer_overrides.txt \
  /var/home/deucebucket/games/models/gemma4-26b-codex-dwojcik/gemma-4-26B-A4B-it.Q4_K_M.gguf \
  /var/home/deucebucket/games/cerebellum-gemma4-codex/gemma4-26b-codex-cerebellum-v6-transfer-requant.gguf \
  Q3_K_M
```

Gotcha: this is a destructive double-quantization transfer probe from Q4_K_M,
not the clean release path.

### Clean LoRA Branch

Local developer log says:

- Base BF16 plus `hotdogs/gemma4-26b-python-18k-alpaca-lora`.
- 394/394 LoRA tensors merged.
- v7 scored poorly on EvalPlus.
- v7b promoted 192 LoRA-target tensors from Q2_K to at least Q4_0 and improved.

The final clean GGUF/output dir named in docs is not present now. Only benchmark
artifacts survived under:

`/var/home/deucebucket/ai-drive/cerebellum/cerebellum-gemma4-codex/benchmark_results/`

Treat this branch as incomplete unless the missing final artifacts are restored.

## Gemma 4 12B Current Failure Contrast

Current/recent dirs:

- Legacy forward/reverse:
  `/var/home/deucebucket/games/cerebellum-runs/families/gemma-4/gemma-4-12b-it/sources/google-f16/runs/gemma4-12b-classic-forward-reverse-20260606/`
- Targeted hillstep:
  `/var/home/deucebucket/games/cerebellum-runs/families/gemma-4/gemma-4-12b-it/sources/google-f16/runs/gemma4-12b-targeted-attnv-earlyblocks-20260606/`
- Older hillclimb tmp:
  `/var/home/deucebucket/games/cerebellum-pipeline-tmp/gemma4-12b/`

What went differently:

- Forward/reverse was run, but only against wiki PPL.
- `attn_v` and `earlyblocks` survived because wiki PPL liked them.
- The final pre-hillstep map had 65 Q2 entries, including early block
  `attn_k`, `attn_output`, `attn_v`, `ffn_down`, `ffn_gate`, and `ffn_up`.
- Hillstep promoted some flagged tensors, including `blk.2.attn_v.weight` and
  `blk.4.attn_v.weight` to F16, but left many poisoned Q2 tensors.
- The benchmark gate happened after the poisoned map was already accepted.

The OG Gemma 26B evidence says not to trust this shape. It did not globally
demote `attn_v`; it used a compact 91-entry Q3_K_M-base map and then benchmarked
with the Gemma-specific chat harness.

## Immediate Reproduction Recommendation

For Gemma 4 12B, the closest OG-compatible restart is:

1. Start from Q4_K_M or Q3_K_M baseline, but do not seed from the 65-entry
   `attn_v`/earlyblocks survivor map.
2. Transfer the Gemma 26B v6 family prior cautiously:
   selected `attn_q` higher, `attn_k` lower only where validated, selected
   `ffn_up`/expert gate-up lower, no broad `attn_v` or `attn_output` demotion.
3. Run a fast benchmark gate before hillstep:
   EvalPlus chat subset, ARC subset, HellaSwag subset.
4. Only then let hillstep promote/demote individual tensors.
5. Treat wiki PPL as a diagnostic, not the release objective.
