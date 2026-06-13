# Vision: The Sticky Matrix (Jerry's internal notes — NON-UPSTREAM, never publish)

Captured 2026-06-11 from Jerry's articulation. This is the endgame the GGUF work hooks toward.

## The arc
1. **GGUF = the hook.** In-weights knowledge blocks on vanilla llama.cpp prove you can
   inline info, store info, and control the knowledge/context trade. Heads turn.
2. **Fork = live inlining.** Runtime injection of retrieved vectors into the residual
   stream. Streamlined RAG: no prompt stuffing, no context-window tax, no training to
   memorize — the model treats anything you point it at as its own memory.
3. **Branch → mainline.** The fork becomes a real llama.cpp branch and eventually a
   mainline merge. Upstream when it's undeniable, not before.

## The mechanism
- FFN = key-value memory (keys: gate/up match residual patterns; values: down_proj rows
  written back). Today's coder block proved trained in-weights storage works.
- Sticky matrix = same math, matrix externalized: keys/values as rows in an external
  tensor. The block trains ONCE as the read/write protocol (librarian); content is DATA.
  Swap the matrix, swap the knowledge. Memory without retraining.
- **Write path (the sticky part):** salient hidden states that pass over the matrix
  leave traces — appended/blended rows at inference time. Runtime memorization, zero
  gradients. The model outlines and stores its own memories into an external tensor.

## Hard parts (be honest)
- Write-path interference: naive accumulation degrades. Needs salience gating (what
  sticks), decay/consolidation, and indexing. Fast-weights / memorizing-transformer
  territory — but with a frozen base model and a learned protocol block, which is new.
- Usable recall is the gating evidence: in-weights instillation is proven (train-set
  PPL 7.20 -> 1.28), recall-through-access still being measured.
- Context preservation is a HARD constraint (Jerry, 2026-06-11): no candidate ships
  that regresses long-context. FFN-only blocks are context-safe by construction
  (pointwise, no attention to misfire at length).

## The pill framing (Jerry, 2026-06-11)
You can't sell a brain implant first. You sell the temporary, easily digestible pill
that proves the basis: small, reversible, zero-trust (a GGUF anyone can run on stock
llama.cpp and delete). When people start reporting the GGUFs work — recall without
inline context bloat — someone asks for the brain mod (the fork, live injection,
sticky matrix). Adoption ladder: pill -> reports -> mod. The felt benefit users will
evangelize is "it just knows, and my context window is still mine."
