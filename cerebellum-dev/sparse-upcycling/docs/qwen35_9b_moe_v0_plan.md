# Qwen3.5 9B Sparse Upcycle v0

## Why 9B

Qwen3.5-9B is popular enough that a working sparse derivative would matter, and
it is small enough to iterate on locally. The goal is not to beat every dense
9B immediately. The goal is to make a 9B-derived model with MoE structure that
Cerebellum can compress and eventually page by expert.

## First Model

Use sliced sparse upcycling rather than cloning the dense MLP into every expert.

- Keep attention, linear-attention, embeddings, norms, and output head from the
  source model.
- Replace each dense SwiGLU MLP with routed expert SwiGLU MLPs.
- Slice the original intermediate dimension across experts:
  `12288 / 16 = 768`.
- Add a small shared expert as a stabilizer for common token paths.
- Train routers and selected expert parameters before touching full-model
  fine-tuning.

## Why Not MTP First

MTP is a decode-speed feature, not the footprint unlock. For v0, the unlock is
active expert sparsity plus Cerebellum precision allocation. MTP can be revisited
after the base sparse model is stable.

## Success Criteria

Minimal useful evidence:

- HF model loads with custom sparse config.
- Forward pass and loss work on short sequences.
- Router does not collapse to one expert globally.
- Short distill/warmup improves loss over the no-training upcycle.
- Cerebellum quant has a measurable advantage over a dense Qwen3.5-9B quant at a
  comparable file-size or runtime-memory target.

## Known Risks

- Qwen3.5 hybrid layers may have fragile state/linear-attention tensors. These
  should stay out of the first architectural surgery.
- GGUF export may require either mapping onto an existing Qwen MoE schema or a
  llama.cpp patch.
- A Pi-class runtime only benefits if inactive experts actually remain cold under
  mmap/offload. That has to be measured, not assumed.
