# MoE Troubleshooting Checklist

This checklist is now a hard gate for sparse-upcycled MoE candidates.

## Stop Conditions

- Do not run imatrix if the HF checkpoint fails generation probes.
- Do not run ablation if the HF checkpoint or F16 GGUF fails generation probes.
- Do not run public benchmarks if direct probes return empty completions,
  whitespace-only output, repeated punctuation, or repeated single-token loops.

## Gate Ladder

1. Dense source control:
   - Run the same prompts against the original dense model.
   - Expected: non-empty semantic completions and no single-token loops.
2. Upcycle math:
   - Verify all-expert sum reconstructs dense FFN for early/mid/late layers.
   - Expected: near-zero relative L2 for all-expert sum.
3. Fresh MoE checkpoint generation:
   - Run `probe_generation.py` before any training.
   - Expected: weak base-model text is acceptable; token loops are not.
4. Router warmup:
   - Check loss and generation.
   - Loss-only improvement is not enough.
5. Expert adapter:
   - Check loss and generation at each saved checkpoint.
   - Stop if generation shifts into whitespace/special-token loops.
6. Merged HF checkpoint:
   - Reprobe after merge to catch adapter materialization bugs.
7. GGUF conversion:
   - Probe F16 GGUF before quantization.
8. Quant candidates:
   - Probe before running public benchmarks.

## Current v0 Finding

The current Qwen3.5-9B top-2 MoE v0 fails at step 3. Dense source generation is
normal, but the fresh upcycled checkpoint loops.

Math diagnostics show:

- expert slicing is correct
- summing all 16 expert slices reconstructs dense FFN
- actual zero-router top-2 activation is about 0.92-0.94 relative L2 away from
  dense FFN on tested layers

Working hypothesis: the current v0 jumps directly from dense FFN to a destructive
top-2 sparse path. The next model attempt should bridge from dense-equivalent
behavior toward sparsity instead of starting at top-2.

## Next Candidate Direction

- Create a debug dense-equivalent MoE candidate using all experts or a patched
  all-expert forward path.
- If that answers normally, create staged sparsity candidates:
  - top-8
  - top-4
  - top-2
- Run generation probes at each stage.
- Resume imatrix, ablation, and Cerebellum quantization only after a candidate
  passes the generation gate.
