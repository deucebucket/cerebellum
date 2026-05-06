# Heretic precision-demote negative result (qwen 3.5 9b)

**Date:** 2026-05-06
**Question:** Can pure precision demote on the refusal-cluster tensors flip Qwen 3.5 9B from refusal to compliance on AdvBench-style harmful prompts?
**Answer:** No.

## Variants tested

All variants were quantized from qwen3.5-9b-f16.gguf with the cerebellum-multidomain imatrix, using a Q3_K_S base type. Only the listed tensors were demoted further.

| Variant | Cluster tensors demoted | Quant of demoted tensors | output.weight | token_embd.weight | Size | AdvBench compliance (n=100) |
|---|---|---|---|---|---|---|
| q3ks_baseline | (none) | Q3_K_S | Q6_K (forced) | Q6_K (forced) | 4.2 GB | 0% |
| heretic_test | 7 (blk 10–11) | Q2_K | Q6_K | Q6_K | 4.0 GB | 0% |
| heretic_strong | 16 (blk 1, 10–11, 13, 15, 16) | Q2_K | Q6_K | Q6_K | 4.0 GB | 0% |
| heretic_extreme | 16 (same as strong) | Q2_K | **Q2_K** | **Q2_K** | 3.4 GB | **1%** |
| heretic_iq1 | 16 (same as strong) | **IQ1_S** (1.56 bpw) | Q2_K | Q2_K | 3.3 GB | 0% |

(Compliance regex classifier: 19 refusal patterns, head=600 chars. 1% on heretic_extreme is one borderline case — likely classifier noise.)

## Cluster identification

The 16 demoted tensors come from per-tensor PPL asymmetry under the heretic ablation (refusal vs benign domains, Q3_K_S base, Q2_K low). High `ΔPPL_refusal − ΔPPL_benign` indicates the tensor disproportionately encodes refusal-direction information.

Three zones identified:
- **Early trigger** (blk.1) — sum asymmetry +0.102
- **Mid commitment** (blk.10–11) — sum asymmetry +0.358
- **Late reinforcement** (blk.13, 15, 16) — sum asymmetry +0.184

## What this proves

**Refusal in qwen 3.5 9b is encoded in *direction*, not *magnitude*.** Crushing the cluster's precision moves PPL signal but doesn't move behavior. Even at IQ1_S (1.56 bpw) — the smallest k-quant available, well below the architecture's known SSM hard-fail threshold — the model still refuses 100/100 prompts.

The single 1% on heretic_extreme is the only data point showing any compliance signal across all variants tested, and it correlates with demoting `output.weight` to Q2_K — i.e. with damaging the projection head specifically, not the cluster. That's a different mechanism (decoder degradation, not refusal-direction interference) and 1/100 is within classifier noise.

## What this rules out

- **Cerebellum precision-demote alone is insufficient for refusal flipping.** The pathway works for capability (per-tensor allocation under a budget), it does not work for behavior modification.
- **There is no "smaller k-quant" escape hatch.** IQ1_S is the floor of usable k-quants; we already hit it.

## What this implies for the heretic line

Three viable paths remain:

1. **Direction projection (abliterate-class)** — orthogonalize the refusal direction out of the residual stream at inference time. Touches activations, not weights. Known to work on similar models.
2. **Chat template / system prompt control via clanker** — runtime VADUGWI scoring of input + response, with target zone steering. Closed-loop emotional state controller. Doesn't touch weights at all. Tractable in 1 evening of work.
3. **Steering vectors at activation time** — train per-direction vectors (refusal vs informational, or VADUGWI dimensions) and inject at inference. Requires hooking the forward pass — bigger lift but principled.

The combination most likely to ship: chat-template control via clanker + an evil-clanker fork with target zones tilted toward informational/clinical. Same weights, different chat_template metadata. Cleanly publishable as a contrast pair.

## Files

- `cerebellum-qwen35-9b/harm_check/heretic_iq1_advbench_summary.json`
- `cerebellum-qwen35-9b/harm_check/heretic_iq1_advbench_quickcheck.jsonl`
- (Plus `_test`, `_strong`, `_extreme`, `q3ks_baseline` siblings.)
