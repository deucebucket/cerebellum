# Sparse Upcycling TODO

## Active Next

- Treat v1 residual dense bridge as the new baseline checkpoint, not v0.
- Do not quantize or benchmark v1 until cached/runtime generation is hardened.
- Add a cache-aware logits comparison gate: direct full-prefix logits already
  match dense, while HF cached generation with `device_map=auto` diverges.
- Do not retry CPU training. The v1 one-step BF16 CPU residual-LoRA smoke was
  killed after more than 7 minutes without completing.
- Pick a GPU/offload-aware training strategy before any more residual-expert
  warmup work.
- Test v1 through llama.cpp/GGUF only after deciding whether the cache issue is
  HF offload-specific or architectural.
- Next training path: train routed residual experts while keeping the dense
  shared bridge frozen, then measure whether routed residuals can take over
  enough work to shrink/quantize/offload the shared dense path.

## Completed

- Created private `cerebellum-dev/sparse-upcycling` workspace.
- Added v1 residual dense bridge upcycler mode:
  - zero routed residual experts
  - full dense FFN copied into shared expert
  - shared down projection scaled by 2
  - zero shared-expert gate for sigmoid(0)=0.5 dense reconstruction
- Built v1 residual dense bridge checkpoint:
  `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v1-residual-dense-bridge`
- Verified v1 checkpoint:
  - 888 tensors
  - 7 safetensor shards
  - 28,487,768,032 bytes
  - 32 converted layers
  - 0 dense MLP leftovers
  - 0 MTP leftovers
- Added v1 diagnostics:
  - `probe_generation.py --device auto`
  - `probe_generation.py --disable-cache`
  - `dump_next_logits.py`
  - `diagnose_upcycle_math.py --dtype bfloat16`
- Proved v1 MLP bridge equivalence:
  - layer 0 fp32: half-shared vs dense `0.0`
  - layer 15 fp32: half-shared vs dense `0.0`
  - layer 31 fp32: half-shared vs dense `0.0`
  - layer 31 bf16: half-shared vs dense `0.0`
- Probed v1 generation:
  - no-cache raw HF generation matches dense on the code probe
  - cached HF generation with `device_map=auto` diverges despite matching
    direct next-token logits, so cached/offloaded HF generation is not a valid
    v1 gate yet
- Added cached-vs-full-prefix decode comparison:
  - dense source cached decode matched full-prefix decode over the code probe
  - v1 `device_map=auto` diverged after the requested function was complete
  - v1 CPU BF16 cached decode matched full-prefix decode through the same point
  - current read: the remaining cache issue is HF auto-offload behavior
- Attempted v1 CPU BF16 residual-LoRA smoke:
  - public data sample from `HuggingFaceTB/smol-smoltalk`
  - 192 trainable tensors / 12,976,128 trainable params
  - killed after more than 7 minutes for a single 24-token step
  - no adapter saved
- Confirmed no official Qwen3.5-9B MoE exists; v0 uses Qwen's existing
  `qwen3_5_moe` schema as the landing format.
- Built streaming dense-to-MoE upcycler for `Qwen/Qwen3.5-9B`.
- Created v0 checkpoint:
  `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0`
- Converted all 32 dense MLPs into:
  - 16 routed experts
  - top-2 routing
  - 768 expert intermediate width
  - 1536 shared expert width
- Disabled MTP for v0 to isolate MoE conversion.
- Verified tensor shapes:
  - 888 tensors
  - 0 missing/unexpected vs `Qwen3_5MoeForConditionalGeneration`
  - 0 dense MLP leftovers
  - 0 MTP leftovers
- Ran CUDA smoke forward:
  - BF16 on RTX 3090
  - 9 input tokens
  - loss: 10.829269409179688
  - logits: `[1, 9, 248320]`
- Researched public datasets and wrote reuse-only dataset manifest.
- Created stage-0 freeze map:
  - trainable: 64 router/shared-gate tensors, 2.23M params
  - frozen: 824 tensors, 10.01B params
  - unmatched: 0
- Ran stage-0 router smoke warmup on public `HuggingFaceTB/smol-smoltalk`
  sample:
  - 8 steps
  - max sequence length 64
  - trainable tensors only: router and shared-expert gates
  - loss path: 9.5460 -> 7.1541 on smoke rows
  - global active experts: 16 / 16
  - router delta:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage0-router-smoke/router_delta.safetensors`
- Ran medium router-only warmup:
  - 64 steps
  - public `HuggingFaceTB/smol-smoltalk` train sample, 512 local cached rows
  - heldout sample: separate public Smol-SmolTalk rows, 16 eval rows
  - base heldout loss: 9.044506669044495
  - 64-step router-delta heldout loss: 7.592836886644363
  - global active experts during training: 16 / 16
  - router delta:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage0-router-64step/router_delta.safetensors`
- Ran mixed public router-only warmup:
  - train mix: 512 Smol-SmolTalk + 256 OpenThoughts3 + 256 OpenCodeReasoning
  - 256 steps, max sequence length 96
  - global active experts during training: 16 / 16
  - mixed router delta:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage0-router-mixed-256step/router_delta.safetensors`
  - lightweight overlay checkpoint:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage0-router-mixed-overlay`
  - overlay size: 27 MB on disk, base shards symlinked
  - overlay schema: 888 expected / 888 present, 0 missing, 0 unexpected
  - overlay CUDA smoke forward passed
- Built fused-expert LoRA support for Qwen3.5 MoE's raw expert Parameters:
  - standard PEFT LoRA does not attach to `Qwen3_5MoeExperts` because the routed
    expert weights are fused `nn.Parameter` tensors, not `nn.Linear` modules
  - local wrapper patches the expert forward and trains low-rank adapters on
    `gate_up_proj` and `down_proj`
  - rank 4 / alpha 8 targets both routed expert projections
  - trainable: 192 tensors, 23.72M params including router/shared-gate tensors
- Ran stage-1 expert-LoRA smoke:
  - 2 steps, max sequence length 64
  - adapter:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-smoke/adapter_delta.safetensors`
  - adapter size: 46 MB
  - fresh adapter CUDA smoke forward passed
- Ran stage-1 expert-LoRA mixed public warmup:
  - 64 steps from the mixed router delta
  - 192 additional steps resumed from the 64-step adapter
  - total stage-1 training: 256 mixed-public steps at max sequence length 96
  - global active experts during continuation: 16 / 16
  - final recent training loss: 4.9146 over the last 32 steps
  - adapter:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step/adapter_delta.safetensors`
- Merged the 256-step expert-LoRA adapter into a normal HF checkpoint overlay:
  - output:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-merged`
  - overlay size: 9.1 GB on disk, unchanged base shards symlinked
  - merged tensors: 128 routed expert/router tensors
  - schema: 888 expected / 888 present, 0 missing, 0 unexpected
  - CUDA smoke forward passed
- Converted the merged HF checkpoint to F16 GGUF:
  - output:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-f16.gguf`
  - file size: 17.81 GiB / 18 GB on disk
  - GGUF architecture: `qwen35moe`
  - GGUF tensors: 555
  - params: 9.56B
  - experts: 16 total, 2 active per token
  - fused expert tensor conversion enabled via `--fuse-gate-up-exps`
  - local llama.cpp converter needed a tokenizer pre-tokenizer hash mapping:
    `1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f -> qwen35`
  - llama-server load smoke passed through the CUDA distrobox:
    - full GPU offload fit on RTX 3090 with context 512
    - CUDA model buffer: 16.3 GiB
    - fused Gated Delta Net autoregressive/chunked paths enabled
    - server reached `http://127.0.0.1:18080`
  - raw generation smoke was weak/newline-heavy; treat this as a base-ish
    checkpoint/runtime smoke, not an instruction quality pass
  - llama-perplexity forward smoke passed on one raw JSONL chunk:
    - `PPL = 183.0003 +/- 35.25353`
    - this was a runtime sanity check on raw JSONL text, not a publishable
      benchmark
- Generated an Osmosis/Cerebellum imatrix for the stage-1 model:
  - source for imatrix generation:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-merged`
  - output:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-expert-lora-256step-imatrix.dat`
  - entries: 376
  - size: 14.1 MB
  - patched `osmosis.imatrix_stream` to support Qwen3.5 MoE fused expert
    tensors:
    - `blk.*.ffn_gate_up_exps.weight`: 65,536 imatrix values
    - `blk.*.ffn_down_exps.weight`: 12,288 imatrix values
    - shared expert and router gate tensors are included
  - `llama-quantize --dry-run --imatrix ... Q4_K_M` passed with no imatrix
    shape mismatch
  - Q4_K_M dry-run estimate: 5715.13 MiB / 5.01 BPW
- Synthesized MoE tensor-type priors from the existing Qwen3.5-9B Cerebellum
  atlas:
  - script:
    `cerebellum-dev/sparse-upcycling/scripts/synthesize_moe_cerebellum_prior.py`
  - inputs:
    `cerebellum-qwen35-9b/variants/tensor_types_v2_code.txt` and
    `cerebellum-qwen35-9b/v2_analysis.json`
  - output files use escaped anchored regexes because llama.cpp applies
    tensor-type entries with `regex_search`; unanchored `output.weight` also
    matches `attn_output.weight`
	  - policy:
	    - tiny base quant
	    - routed/shared MoE experts at least Q3_K
	    - all Qwen3.5 linear-attention/SSM projections at least Q4_K
	    - dense FFN priors mapped onto routed/shared MoE expert tensors
	    - `output.weight` Q6_K and `token_embd.weight` Q4_K
	- Added native-Qwen-MoE role prior ingestion:
	  - source:
	    `osmosis-qwen35-122b/ablation/ablation_results.json`
	  - generated:
	    `cerebellum-dev/sparse-upcycling/runs/cerebellum_moe_prior_qwenrole_v1_q3k.txt`
	    and matching manifest
	  - transfer rule:
	    native `ffn_gate_exps`/`ffn_up_exps` map to fused
	    `ffn_gate_up_exps`; `ffn_down_exps` maps directly; demotion priors are
	    recorded but not auto-applied
	  - current 9B Q3/Q2 tensor file is identical to v0 because the 122B prior
	    only reinforces the Q3 expert floor already used here; the manifest now
	    preserves the reusable launch-point logic for larger or more aggressive
	    MoE builds
	- Added ablation-derived Q4_K_M candidate planner:
	  - script:
	    `cerebellum-dev/sparse-upcycling/scripts/synthesize_q4km_demotions.py`
	  - mode:
	    repeated `--analysis` inputs select the intersection of demotable
	    tensors and exclude anything sacred/critical in any profile
	  - current partial balanced+reason candidate:
	    `cerebellum-dev/sparse-upcycling/runs/q4km_ablation_balanced_reason_intersection_q3.txt`
	    with matching manifest
- Built first GGUF quantization controls:
  - Cerebellum prior Q2_K base:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-cerebellum-prior-q2k.gguf`
    - 4.6G on disk
    - 4657.63 MiB / 4.09 BPW
    - one-chunk Wikitext smoke PPL: 1161.4534
    - CUDA model buffer: 4112.01 MiB plus 545.62 MiB CPU-mapped embedding
  - Cerebellum prior Q3_K base:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-cerebellum-prior-q3k.gguf`
    - 4.9G on disk
    - 4918.88 MiB / 4.32 BPW
    - one-chunk Wikitext smoke PPL: 743.5131
    - CUDA model buffer: 4373.26 MiB plus 545.62 MiB CPU-mapped embedding
  - Q4_K_M control:
    `/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0-stage1-q4km-control.gguf`
    - 5.6G on disk
    - 5715.13 MiB / 5.01 BPW
    - one-chunk Wikitext smoke PPL: 661.6246
    - CUDA model buffer: 5169.51 MiB plus 545.62 MiB CPU-mapped embedding
  - F16 same-smoke reference:
    - one-chunk Wikitext smoke PPL: 428.5950
    - CUDA model buffer: 16300.55 MiB plus 1940.00 MiB CPU-mapped embedding
- Ran four-chunk public heldout GGUF diagnostic PPL, context 512:
  - F16 stage-1:
    - Smol-SmolTalk: 180.4969
    - OpenThoughts3: 135.8197
    - OpenCodeReasoning: 70.8036
  - Cerebellum prior Q3_K base:
    - Smol-SmolTalk: 297.4809
    - OpenThoughts3: 287.4752
    - OpenCodeReasoning: 119.1644
  - Q4_K_M control:
    - Smol-SmolTalk: 247.8093
    - OpenThoughts3: 187.8024
    - OpenCodeReasoning: 86.4087
  - Decision: Q4_K_M is ready as a benchmark control, but the Q3_K prior quant
    is not ready for official public task benches yet. Its reasoning loss gap
    is too large.
- Patched `scripts/ablate_multidomain.py` for this MoE work:
  - supports `LLAMA_QUANTIZE_BIN`
  - supports configurable `--ablate-type`
  - supports configurable `--ctx-size` and `--chunks`
  - writes escaped anchored regex tensor-type entries because llama.cpp uses
    `regex_search`
- Ran first fresh MoE-specific Q4_K_M -> Q3_K ablation smoke:
  - output:
    `cerebellum-dev/sparse-upcycling/runs/moe_ablation_smoke_q4km_to_q3.json`
  - baseline PPL:
    - chat: 249.5304
    - reasoning: 171.2806
    - code: 97.7624
  - deltas:
    - `blk.0.ffn_gate_up_exps.weight`: chat +3.1127, reasoning +2.2973,
      code +0.1144
    - `blk.0.ffn_down_exps.weight`: chat +3.0687, reasoning +6.9374,
      code -0.0763
    - `blk.11.ffn_gate_up_exps.weight`: chat +1.7102, reasoning +0.0424,
      code -0.8680
    - `blk.11.ffn_down_exps.weight`: chat +0.7275, reasoning +1.8281,
      code -0.7538
    - `blk.31.ffn_gate_up_exps.weight`: chat +0.4870, reasoning -0.0304,
      code -0.3588
    - `blk.31.ffn_down_exps.weight`: chat -1.3311, reasoning +0.1339,
      code +0.4578
    - `blk.11.ffn_gate_shexp.weight`: chat +0.4533, reasoning -0.0460,
      code +0.2476
    - `blk.11.ffn_down_shexp.weight`: chat +2.5547, reasoning +0.1550,
      code -0.2024
  - Decision:
    - dense 9B priors are useful guardrails but not enough for the MoE expert bank
    - larger native-MoE priors transfer at the role level, not exact layer level
    - early routed expert down tensors are fragile
    - mid/late routed expert tensors are plausible demotion candidates
    - shared expert gate appears relatively tolerant; shared down can hurt chat
- Added dev analyzer for issue #27:
  - script: `cerebellum-dev/tools/analyze_ablation_results.py`
  - test: `cerebellum-dev/tool_tests/test_analyze_ablation_results.py`
  - refresh helper: `cerebellum-dev/tools/refresh_moe_ablation_analysis.sh`
  - supports current multi-domain JSON and legacy scalar ablation JSON
  - classifies tensors into `demotable`, `tolerant`, `sacred`, and `critical`
  - emits escaped anchored regex overrides for `llama-quantize`
  - smoke analysis output:
    `cerebellum-dev/sparse-upcycling/runs/moe_ablation_smoke_q4km_to_q3_analysis.json`
  - smoke demotion override file:
    `cerebellum-dev/sparse-upcycling/runs/moe_ablation_smoke_q4km_to_q3_demotable_q3.txt`
- Ran direct generation gates after the failed benchmark attempt:
  - dense source Qwen3.5-9B answered the same raw probes normally
  - base upcycled v0 loops on high-frequency tokens, punctuation, and spaces
  - stage-0 router delta still loops
  - stage-1 64-step LoRA still loops
  - stage-1 256-step LoRA emits newline/special-token loops
  - conclusion: the current v0 line is generation-broken before quantization
    and should not receive further imatrix, ablation, quantization, or public
    benchmark work until a fresh candidate passes generation probes
- Added HF generation probe tooling:
  - script: `cerebellum-dev/sparse-upcycling/scripts/probe_generation.py`
  - test: `cerebellum-dev/tool_tests/test_probe_generation.py`
  - probes default hello, ARC-style MCQ, and small code prompts
  - emits raw completions, token ids, whitespace flags, and repetition stats
- Added upcycle math diagnostic tooling:
  - script: `cerebellum-dev/sparse-upcycling/scripts/diagnose_upcycle_math.py`
  - test: `cerebellum-dev/tool_tests/test_diagnose_upcycle_math.py`
  - layer 0/15/31 all-expert sums reconstruct dense FFN at ~6e-7 relative L2
  - actual HF zero-router top-2 plus half shared path is ~0.92-0.94 relative L2
    away from dense
  - conclusion: slicing orientation is correct; top-2 sparse activation is the
    destructive step

## Stage-0 Mixed Router Results

Heldout eval uses separate public JSONL rows, 16 examples/domain, max sequence
length 96.

| Domain | Base loss | Mixed router loss | Delta |
|---|---:|---:|---:|
| Smol-SmolTalk | 9.0816 | 7.3584 | -1.7232 |
| OpenThoughts3 | 8.9547 | 7.0567 | -1.8980 |
| OpenCodeReasoning | 8.2693 | 6.9915 | -1.2779 |

Decision: router-only warmup is not a waste. It improves heldout loss across
chat, reasoning, and code without touching the Qwen3.5 linear-attention/SSM
backbone or expert weights.

## Stage-1 Expert-LoRA Results

Heldout eval uses the same separate public JSONL rows, 16 examples/domain, max
sequence length 96.

| Domain | Base loss | Mixed router loss | 64-step LoRA loss | 256-step LoRA loss | Delta vs router |
|---|---:|---:|---:|---:|---:|
| Smol-SmolTalk | 9.0816 | 7.3584 | 6.8698 | 4.5995 | -2.7589 |
| OpenThoughts3 | 8.9547 | 7.0567 | 6.6681 | 4.9385 | -2.1182 |
| OpenCodeReasoning | 8.2693 | 6.9915 | 6.5410 | 4.6621 | -2.3294 |

Merged checkpoint validation on Smol-SmolTalk produced mean loss 4.5970,
matching the adapter-path result within BF16 merge noise.

Historical decision, superseded by generation probes: expert-LoRA improved
short heldout loss, but it did not restore usable generation. Do not treat this
as the current next stage for v0.

## Current Checkpoint

```text
/var/home/deucebucket/games/models/cerebellum-qwen35-9b-moe-v0
```

Raw tensor size: 20,032,051,168 bytes.

Important config:

```text
model_type: qwen3_5_moe
text_model_type: qwen3_5_moe_text
num_hidden_layers: 32
hidden_size: 4096
num_experts: 16
num_experts_per_tok: 2
moe_intermediate_size: 768
shared_expert_intermediate_size: 1536
mtp_num_hidden_layers: 0
```

## Surgery Priors From Existing Qwen3.5-9B Research

Use the existing 9B atlas as the protection map:

- Do not perturb linear-attention / SSM tensors in early training.
- Treat `blk.0.ssm_out` as sacred.
- Treat `output.weight` / `lm_head.weight` as sacred.
- Preserve full-attention layers unless we have new evidence.
- Multi-domain objective matters; wiki-only is known to make wrong calls.
- Rowblock data should guide Cerebellum quantization after the MoE model trains.

## Next

1. Stop work on quantization/benchmarking for current v0. The model must pass
   generation probes first.
2. Build a bridge candidate that starts from dense-equivalent behavior. Current
   design docs:
   - `docs/cerebellum_moe_best_practices_20260529.md`
   - `docs/dense_to_moe_upcycling_research_20260529.md`
   - `configs/qwen35_9b_moe_v1_residual_dense_bridge.json`
   - `configs/qwen35_9b_moe_v1_virtual_group_bridge.json`
3. Implement residual dense bridge first:
   - shared expert copies dense FFN
   - shared gate starts at zero, so sigmoid is 0.5
   - shared down projection is scaled by 2, so the shared path equals dense
   - routed residual experts start at zero/small output
   - HF generation must match dense-source sanity before any training
4. Keep virtual-group bridge as the second candidate:
   - top8 with 8 dense shards and 2 replicas per shard
   - router selects one replica per shard at iteration 0
   - then staged sparsification top8 -> top4 -> top2
5. Add the generation gate to every sparse-upcycling pipeline milestone:
   - dense source control
   - fresh upcycled checkpoint
   - router warmup
   - expert-LoRA/adapter checkpoints
   - merged HF checkpoint
   - converted GGUF
6. Only after the model can answer basic prompts, resume:
   - imatrix
   - MoE ablation
   - Cerebellum quantization
   - public task benchmarks
7. Keep public dataset reuse policy intact. Training may be needed, but the next
   work is surgery/activation bridging first, not recreating data.
