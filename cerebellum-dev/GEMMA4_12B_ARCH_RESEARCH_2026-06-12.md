# Why Gemma 4 12B resists cerebellum (arch research, 2026-06-12)

Question: 26B-A4B and the E-series loved cerebellum; 12B came back NO-SHIP
(only attn_v demotable, and even that cost 7.4 HumanEval+ pts). Why?

## The 12B is a different architecture, not just a different size

It's not `gemma4` — it's `gemma4_unified` (`Gemma4UnifiedForConditionalGeneration`),
released 2026-05-23, two months after the March `gemma4` wave. Public config facts
(hf.co/google/gemma-4-12B-it, config.json):

| | 12B (unified) | 31B dense | 26B-A4B | E4B |
|---|---|---|---|---|
| model class | gemma4_unified | gemma4 | gemma4 | gemma4 |
| vision/audio encoders | **none** (encoder-free, raw patches/waveforms → linear proj) | ~550M vision | ~550M vision | ~150M vision + ~300M audio |
| MoE | no | no | 128 experts, top-8 | no |
| PLE | no (`hidden_size_per_layer_input: 0`) | no | no | yes (256) |
| K=V sharing | `attention_k_eq_v: true`; global layers have **no separate V at all** | true | true (per config) | false |
| global attn KV | **1 KV head** (MQA), head_dim 512 | 4 | 2 | shared-KV layers |

GGUF ground truth from our own campaign dir: `attn_v` exists in only 40 of 48
layers — the 8 missing are exactly the full_attention layers (5, 11, 17, 23,
29, 35, 41, 47). The global layers literally share one K=V projection.

## Why that kills the group-level method here

Cerebellum's wins have all come from harvesting **architectural redundancy**:

- 26B-A4B: 128-expert MoE — most expert mass is cold per token; huge demotable pool.
- E-series: PLE + encoder stacks — separable structure with wildly uneven sensitivity.
- Dense Qwens: oversized attention (K/Q/output demotions sometimes *improved* PPL).

The 12B unified has none of those pools. It is already compressed by design:

1. **Encoder-free**: the same decoder weights do text + vision + audio understanding.
   More jobs per weight = less slack. There are no encoder params to sacrifice.
2. **Attention already minimized**: K=V sharing + MQA global layers. Total attention
   is 1.41 GB of the 7.38 GB Q4 file; the only tolerant group (attn_v) is 0.22 GB.
   Max possible harvest was ~0.12 GB before we started.
3. **FFN is 70% of the weight mass (5.14 GB) and uniformly screaming**: ffn_down
   +52.6% mean / +80.5% max PPL at Q2_K. Dense FFN on the hot path every token.
4. **QAT-family training**: Google ships official QAT q4_0 for the whole family —
   weights are trained to sit comfortably at ~4-bit, flattening the sensitivity
   landscape. (QAT alone doesn't explain it — 26B is also QAT'd — but it removes
   the "one fragile group, many tolerant groups" contrast cerebellum exploits.)

The ablation table reads exactly like this: 6 of 7 groups PROTECT, no contrast,
nothing to trade. "Uniform Q4_K_M/Q3_K_M are already on the frontier" is not a
campaign failure — it's the architecture doing cerebellum's job at training time.

## Heuristic for future model selection (proposed)

Before committing a campaign, score the config for **redundancy pools**:
MoE experts? encoders? PLE? oversized/unshared attention? If the answer is
"dense + encoder-free + KV-shared + official QAT", expect a 12B-style flat
ablation and deprioritize. The 12B's profile was predictable from config.json.

Corroborating the attn_v coding loss: with K=V shared in global layers and only
8 KV heads elsewhere, V carries proportionally more unique signal per weight than
in a fat-attention dense model — consistent with PPL tolerating Q2_K V while
HumanEval+ dropped 7.4 pts (drift the PPL screen can't see).

Sources:
- https://hf.co/google/gemma-4-12B-it (model card + config.json)
- https://hf.co/google/gemma-4-31B-it, hf.co/google/gemma-4-26B-A4B-it, hf.co/google/gemma-4-E4B-it (configs)
- https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf (official QAT release, 2026-06-05)
- https://unsloth.ai/docs/models/gemma-4/qat (QAT notes; q4_0 found lossy, UD-Q4_K_XL preferred)
- https://news.ycombinator.com/item?id=48385906 (unified encoder-free discussion)
- Local evidence: cerebellum-gemma4-12b/allocation_report.md, ablation_tensors.txt
