# Cerebellum Research Ideas — Spitball Session 2026-05-01

All ideas from late-night session. These are original concepts, not published anywhere.

---

## 1. Quantization-as-Routing-Bias (MoE Steering)

**Concept:** Use quantization noise as a steering mechanism in MoE models. Instead of treating lossy compression as pure damage, use it to bias the router toward experts that survived quantization well and away from experts that were degraded.

**How it works:**
- Identify which experts are best-preserved at Q2_K (from ablation data)
- Give the router paths to well-preserved experts MORE precision (Q5_K/Q6_K)
- Give the router paths to damaged experts LESS precision (Q2_K)
- The asymmetric quantization noise nudges routing decisions toward paths the model handles well

**Why it matters:**
- No fine-tuning needed — pure quantization-time optimization
- The model "routes around its own damage" automatically
- Enables task-specific routing: bias toward code experts for code quant, reasoning experts for reasoning quant
- Nobody has published this concept

**How to test:**
1. Map expert activation patterns per task (code vs reasoning vs knowledge)
2. Identify which experts survived Q2_K cleanly (low PPL delta when demoted)
3. Promote router weights serving those experts to Q6_K
4. Demote router weights serving damaged experts to Q2_K
5. Measure if PPL improves and if task-specific benchmarks shift

**The inverse play — crush roads to bullshit (FREE):**
- Identify which experts were DAMAGED by quantization (high PPL delta)
- Crush the router paths TO those broken experts → saves bits
- Those saved bits fund promoting roads to good experts
- Net effect: same file size, model routes around its own damage
- Self-healing through topology — don't fix the pothole, close the road
- This is CHEAPER than promoting because crushing saves bits instead of spending them

**Per-layer gate granularity:**
- The gate tensor (ffn_gate_inp) is 2816×128 — each row maps to one of 128 experts
- At tensor level: different layers' gates have different sensitivity (layer 15 = byway, layer 0 = entry ramp)
- Per-layer gate ablation: crush byway gates (free bits), preserve critical gates
- FUTURE: sub-tensor (row-level) quantization = close the road to INDIVIDUAL experts
- Would need custom quantization tooling (not supported by llama-quantize today)
- This is the surgical endgame: per-expert routing precision control

**Behavioral steering application:**
- If specific experts handle refusal/safety behavior, biasing the router away from them via quantization asymmetry reduces refusals without weight editing
- The "road" to refusal experts is made bumpy (Q2_K), the road to helpful experts is smooth (Q6_K)
- Router naturally picks smoother path → model routes around its own guardrails
- No LoRA, no abliteration, no weight editing — just a "quantization choice" on a 1.375 MB tensor
- Completely undetectable in the weights — you'd have to compare quant configs to know
- Opposite of abliteration: leaves no fingerprint in the model weights themselves
- Could also be used to REINFORCE safety (smooth road to safety experts, bumpy road away)

---

## 2. Conch Shell Architecture — Helical Layer Topology

**Concept:** Instead of a flat sequential pipe (layer 0 → 1 → ... → 29), organize model processing as a logarithmic spiral (conch shell). Each revolution through shared weights produces progressively more refined output.

**Properties:**
- Inner coils: coarse processing (token-level, can be heavily quantized)
- Outer coils: fine reasoning (high-level, needs more precision)
- The precision gradient follows the spiral naturally
- Router layers are the "seams between coils" — transition points that redirect flow

**Evidence from our data:**
- Early and late layers are consistently more sensitive than middle layers
- Layer 15 is always the least sensitive (the "seam")
- This pattern holds across tensor groups (attn_v, attn_o both show it)
- The sensitivity profile IS the shell shape — nobody designed it that way, but it emerged

**Implications for Cerebellum:**
- If the spiral structure exists in current models implicitly, Cerebellum's atlas is literally mapping the shell
- Precision allocation should follow the spiral: more bits at the structural walls, less at the seams
- Cross-model comparison could reveal if all transformers have the same shell shape

---

## 3. Multi-Helix Braided Architecture

**Concept:** Each tensor group (attention, FFN, routing) is its own helix/conch, spiraling independently but crossing at specific "ping spots" where they exchange information.

**Analogy:** Multi-strand DNA — each strand has its own structure but they interleave and cross-link.

**Properties:**
- Each helix can be quantized independently based on its sensitivity profile
- Cross-pollination points (where helices interact) are the critical precision zones
- The "braids" where strands cross would be the layers we see as universally sensitive

---

## 4. Donut/Loop Architecture for Efficient Inference

**Concept:** Universal Transformer variant — instead of 30 unique layers, use 5-8 unique layers that loop N times. Each revolution refines the previous output (MC Escher staircase that goes up while appearing to loop).

**Benefits:**
- Way fewer parameters → smaller model → easier to quantize
- Cerebellum would ablate in minutes instead of hours (only 5-8 unique layer configs)
- Natural fit for the sensitivity atlas — fewer things to measure
- Memory efficient for inference (weights stay in cache)

**Challenge:** Current hardware (NVIDIA) optimizes for the straight pipe. But if the entire donut lives on GPU (no bus transfers), looping is just a counter increment — fast.

**GPU-CPU loop variant:** Weights on GPU, control flow on CPU, "ping spots" at each revolution boundary where CPU decides: continue looping or exit and output.

---

## 5. Sensitivity-Informed Routing (The EQ Metaphor)

**Concept:** Treat the model like a stereo EQ. Each tensor group is a frequency band. Boost the bands that are starving (sensitive tensors → more bits), cut the bands that are muddy (tolerant tensors → fewer bits). The overall "volume" (file size) stays the same but the "sound" (quality) improves.

**Extensions:**
- Task-specific EQ presets: "Code" boosts different frequencies than "Reasoning"
- Cross-model EQ templates: if Gemma 4 and Qwen 3.6 have the same sensitivity shape, apply the same EQ
- Dynamic EQ: adjust precision at inference time based on what the model is being asked to do (this is speculative but theoretically possible with switchable LoRAs or quantization adapters)

---

## 6. Reasoning Drift Benchmark

**Concept:** Measure when a quantized model's reasoning chain starts degrading. Current benchmarks measure single-shot quality — nobody measures "can your quant think for 2000 tokens without going insane?"

**Metrics:**
- Drift Onset Token (DOT): when entropy first drops
- Mean Repetition Score (MRS): n-gram repeat rate
- Reasoning Quality Score (RQS): quality of final answer vs chain length
- Endurance Ceiling: max reasoning budget before zombie loop

**The play:** Find which tensors cause drift, promote them, ship quants with "reasoning-safe up to N tokens" as a feature.

Full spec: `ISSUE_reasoning_drift_benchmark.md`

---

## 7. Cross-Model Tensor Transplant (Same Architecture Only)

**Concept:** If two models have the same architecture and tensor shapes (e.g., two Qwen 3.6 27B variants), could you swap individual tensors between them? Take the best attn_v from model A, the best ffn from model B, frankenmerge at the tensor level.

**Constraints:**
- Must be same architecture, same dimensions
- Can't cross architectures (Gemma → Qwen) because shapes differ
- Could work within a model family (base → instruct → finetune of same arch)

**Testing:** Swap one tensor, measure PPL. If it doesn't explode, the tensor is architecture-compatible.

---

## Priority Order for Testing
1. Quantization-as-routing-bias (testable NOW with existing data + tools)
2. Reasoning drift benchmark (build the measurement first)
3. Sensitivity shell shape (emerges from completing the full atlas)
4. Task-specific EQ presets (need full ablation data + per-task benchmarks)
5. Donut architecture (research paper territory, not implementation)

---

## 8. Cerebellum Infographics — "Nutrition Labels" for Quants

**Concept:** Publish visual infographics with every Cerebellum release showing the real-world operating limits that no one else tests. Users share these constantly — they become the brand.

**Data Points Per Quant:**
- Context Ceiling: max context length before reasoning drift onset (DOT)
- Reasoning Budget: max thinking/reasoning tokens before zombie loop
- KV Cache Compatibility Matrix: which --ctk/--ctv settings are safe at each weight quant level
- Expert Health Map (MoE): visual of which experts survived, which are damaged
- Safe Operating Envelope: the full combo (context × reasoning × KV × temperature) that works
- Drift Risk Zone: where output starts degrading (yellow zone before red zone)

**Format:**
- Clean dark-mode graphics matching Cerebellum brand (indigo accent)
- One-page "nutrition label" style — scannable in 5 seconds
- Shareable on Reddit/Twitter/HF without needing to read a paper
- Generated automatically from benchmark data (Command Center export)

**Why it matters:**
- Nobody else publishes this data — immediate differentiator
- Users share infographics virally — free brand building
- Positions Cerebellum as "the quant people who actually test real usage"
- Creates demand for Cerebellum quants specifically ("I want the one with the longest context ceiling")
- Turns our reasoning drift benchmark into a marketing asset

**Implementation:**
1. Build the reasoning drift benchmark (Issue #6 above)
2. Add infographic generation to Command Center (template + data → SVG/PNG)
3. Auto-generate with each release
4. Post to HF model card, Reddit, Twitter

