# Reasoning Drift Benchmark — KV Cache Degradation Under Quantization

## Problem
Quantized models with reasoning/thinking enabled can fall into "zombie loops" — repetitive output spirals that get worse with longer reasoning chains. This happens because:

1. Quantized KV cache accumulates rounding errors over tokens
2. Reasoning phase generates hundreds of internal tokens, each feeding back into a degrading cache
3. Errors compound until the model hits a "repetition attractor" — a garbage loop it can't escape
4. This is **non-deterministic** — depends on how long the chain runs before drift hits critical mass

## What We Don't Know
- **Which tensors cause this?** Is it attn_k? attn_v? attn_o? Some combination?
- Does the base quant level matter (Q2_K vs Q3_K_M vs Q4_K)?
- Is it the weight quantization or the KV cache quantization that's the primary driver?
- Do certain layer positions drift faster than others?
- Is there a "reasoning budget" threshold per quant level where drift becomes inevitable?

## Proposed Benchmark: Reasoning Endurance Test

### Method
1. Give the model a complex multi-step reasoning task (math proof, code architecture, philosophical argument)
2. Enable thinking/reasoning mode with progressively longer budgets: 128, 256, 512, 1024, 2048, 4096 tokens
3. Measure:
   - **Token-level entropy** over the reasoning chain (should stay stable, drops = drift)
   - **Repetition rate** (n-gram repeat frequency, spike = attractor)
   - **Output quality** (does the final answer degrade with longer chains?)
   - **Drift onset token** — at what token count does entropy first dip?
4. Compare across:
   - Different quant levels (Q2_K, Q3_K_M, Q4_K, Q5_K)
   - Different KV cache quants (q4_0, q8_0, f16)
   - Different per-tensor overrides (our Cerebellum configs)

### Ablation: Find the Guilty Tensors
For each tensor group (attn_k, attn_v, attn_o, etc.):
1. Start from a known-drifting config (e.g., full Q2_K with q4_0 KV cache)
2. Promote ONE tensor group to higher precision
3. Re-run reasoning endurance test
4. If drift onset moves later → that tensor group contributes to drift
5. Rank tensor groups by "drift contribution"

This tells us: **which tensors to protect to preserve long-chain reasoning ability.**

### Metrics
- **Drift Onset Token (DOT)**: first token where entropy drops below threshold
- **Mean Repetition Score (MRS)**: average n-gram repetition rate across the chain
- **Reasoning Quality Score (RQS)**: human or LLM-graded quality of final answer
- **Endurance Ceiling**: max reasoning budget before output becomes garbage

### Test Prompts (Designed to Force Long Chains)
1. "Prove that there are infinitely many prime numbers. Show every step of your reasoning."
2. "Design a distributed database system. Walk through every architectural decision."
3. "Write a compiler for a simple language. Think through each phase before coding."
4. "Solve this step by step: What is the 100th term of the sequence where each term is the sum of the digits of the previous term squared, starting from 7?"

## Integration with Cerebellum
- Add as a benchmark in the Command Center (Module 3)
- Add "Reasoning Endurance" column to model comparison tables
- Add drift visualization to Sensitivity Atlas: color = DOT per tensor per layer
- Task-specific quant profiles: "Reasoning" profile optimizes for maximum DOT
- **The killer feature**: Cerebellum quants that can reason longer than uniform quants at the same size

## Implementation Order
1. Build the endurance test script (measure entropy + repetition over token stream)
2. Run on current v4 Gemma 4 E4B to establish baseline DOT
3. Run on bf16 to establish ceiling
4. If v4 DOT << bf16 DOT, run per-tensor-group ablation to find guilty tensors
5. Build "reasoning-safe" override profile
6. Integrate into Command Center

## Why This Matters
Nobody else is measuring this. Bartowski, unsloth, everyone publishes PPL and benchmark scores but nobody tests "can your quant actually think for 2000 tokens without going insane?" This is a differentiator that matters to actual users — the Reddit thread proves people are hitting this in practice.

## KV Cache Quantization Interaction
The KV cache quant level (--ctk, --ctv flags in llama.cpp) interacts with weight quantization:
- f16 KV cache + Q2_K weights: cache is clean but weights are noisy → some drift
- q4_0 KV cache + Q3_K_M weights: both noisy → compound drift
- f16 KV cache + f16 weights: no drift (ceiling)
- **Question**: Can we find a weight quant config where q4_0 KV cache still works? That saves VRAM.

## Temporary Mitigation (Current)
- Set reasoning tokens to 0 in demos (what we did for the HF Space)
- Once we identify the guilty tensors, promote them and re-enable reasoning
