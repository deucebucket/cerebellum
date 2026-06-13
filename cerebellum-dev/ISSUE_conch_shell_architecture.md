# Conch Shell Architecture — Adaptive Depth via Spiral Looping

## Concept

A 9B-parameter model with adaptive compute depth. Weights loop through shared layers like a spiral — each revolution refines the hidden state further. A learned "exit head" decides when to stop looping and emit output. Simple tokens exit after 1-2 revolutions. Hard reasoning loops 5-8+ times, giving 27B-72B equivalent depth from 9B of stored weights.

Key distinction from donut: the conch has an OPENING. The spiral tightens with each revolution (progressive refinement) and the model learns when its internal thought has "escaped" — when further looping won't improve the output.

## Architecture

```
Input tokens → Embedding → [Loop Block × N adaptive] → Exit Head → Output

Loop Block (shared weights, ~9B params):
  ├── Attention layers (8-12 unique)
  ├── FFN layers (8-12 unique)
  ├── Revolution counter embedding (tells the model which pass it's on)
  └── Exit confidence head (tiny MLP: hidden_dim → 1 scalar)

Exit condition:
  - confidence > threshold → break loop, emit to output head
  - confidence ≤ threshold → loop again
  - Hard cap at max_revolutions (training stability)
```

## Why It Works (Evidence from Cerebellum Data)

Our sensitivity atlas across multiple architectures shows:
- Middle layers (10-25 in a 32-layer model) have NEARLY IDENTICAL sensitivity profiles
- They're computing approximately the same function with slight refinement
- Layer 15 is consistently the "seam" — least sensitive, most redundant
- This pattern holds across Qwen, Gemma, Granite — it's architectural, not model-specific

The conch makes this EXPLICIT: instead of 20 redundant middle layers with unique weights doing ~the same thing, use 8-12 shared layers that intentionally refine.

## Training Plan (Single 3090, 24GB VRAM)

### Phase 1: Architecture Init (1 day)
- Take Qwen 3 3B or Phi-3.5-mini as base
- Identify redundant middle layers from our ablation data
- Average their weights into a shared "loop block"
- Add revolution counter embedding (learned, one per possible revolution)
- Add exit confidence head (2-layer MLP, ~1M params)

### Phase 2: Loop Training (1-2 weeks, 3090)
- Data: NIM teacher generations (Nemotron 70B, DeepSeek V4) — FREE
- Curriculum:
  - Stage 1: Fixed 2 loops, learn to use repetition (3 days)
  - Stage 2: Fixed 4 loops, learn deeper refinement (3 days)
  - Stage 3: Adaptive exit — train confidence head with teacher signal (5 days)
- Exit head training: teacher provides "correct" answer; if student matches after N loops, train exit head to fire at N
- Loss: standard cross-entropy on output + auxiliary loss on exit timing

### Phase 3: Distillation (1 week, 3090)
- Teacher: 70B+ via NIM (free inference)
- Student: our conch model
- Generate 1-5B tokens of teacher reasoning chains
- Train student to match teacher outputs AND to learn when to exit
- Hard problems (math, code) → more loops to match teacher
- Easy problems (factual, short answers) → fewer loops

### Phase 4: Cerebellum It (1 day)
- Apply sensitivity-guided quantization to the final model
- Only 8-12 unique layers to ablate — takes minutes, not hours
- Quantize to 4-5GB GGUF
- Result: 27B+ reasoning depth in a 5GB file

## Key Technical Challenges

1. **Gradient through loops**: Backprop through variable-length loops. Solution: use straight-through estimator for exit decision during training, or REINFORCE with baseline.

2. **Revolution counter**: The model needs to know "which pass am I on?" without this, it can't distinguish first-pass coarse processing from fifth-pass refinement. Solution: learned embedding added to hidden state at each loop entry.

3. **Exit head calibration**: If too eager to exit, model underthinks. If too reluctant, it wastes compute. Solution: calibrate on held-out set with known-difficulty labels.

4. **Catastrophic repetition**: Without proper training, loops produce identical output each time. Solution: curriculum that starts with fixed loops and gradually introduces adaptive exit.

## Prior Art

- Universal Transformer (Dehghani et al., 2018) — ACT halting, but never scaled past BERT-size
- PonderNet (Banino et al., 2021) — probabilistic halting, geometric prior
- ALBERT (Lan et al., 2019) — weight sharing across all layers, but no adaptive compute
- Recursive Transformers (Fan et al., 2024) — weight-shared blocks with fixed loops
- None of these combined: modern 9B scale + frontier teacher distillation + adaptive exit + Cerebellum quantization

## Success Criteria

- 9B storage / 5GB quantized file
- Benchmarks competitive with 27B models (ARC >90%, HumanEval >70%)
- Adaptive compute visible in inference: simple prompts = fast, hard prompts = slower
- Fits and runs on 3090 at interactive speeds

## Resource Requirements

- Storage: ~400GB for training data (HF datasets + NIM generations)
- Compute: Single 3090, 2-3 weeks continuous
- Cost: $0 for data (NIM is free), ~$50 in electricity
- Dependencies: transformers, peft, bitsandbytes, datasets (in distrobox for 3.10 compat)

## Connection to Cerebellum

The conch architecture is the INVERSE of what Cerebellum does:
- Cerebellum: takes a big model, finds redundancy, removes it post-hoc
- Conch: builds a small model that INTENTIONALLY uses controlled redundancy (looping) for depth

They're complementary:
1. Build the conch → gives you 9B params with 27B depth
2. Cerebellum the conch → gives you 5GB file with 27B depth
3. Ship to people with phones/laptops → mission accomplished
