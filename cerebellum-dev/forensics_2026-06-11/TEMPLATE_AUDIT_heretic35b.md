# Template Audit — Qwen3.6-35B-A3B-Heretic-Cerebellum-v1.gguf

**Date:** 2026-06-11
**Auditor:** Claude Code (automated forensics)
**GGUF:** `/var/home/deucebucket/games/cerebellum-staging/heretic-qwen36-35b/Qwen3.6-35B-A3B-Heretic-Cerebellum-v1.gguf`
**Verdict:** SHIP-AS-IS

---

## 1. GGUF Metadata Summary

```
GGUF version:      3
Tensors:           733
KV pairs:          45
general.name:      Qwen3.6 35B A3B Uncensored Heretic
general.architecture: qwen35moe
general.file_type: 12
tokenizer.ggml.model: gpt2
tokenizer.ggml.pre:   qwen35
tokenizer.ggml.bos_token_id: 248044  → '<|endoftext|>'
tokenizer.ggml.eos_token_id: 248046  → '<|im_end|>'
tokenizer.ggml.padding_token_id: 248044 → '<|endoftext|>'
chat_template length: 11793 chars
```

### Token ID Correctness

| Field | Our GGUF | Upstream tokenizer_config.json | Status |
|-------|----------|-------------------------------|--------|
| eos_token_id | 248046 `<\|im_end\|>` | `<\|im_end\|>` = 248046 | CORRECT |
| bos_token_id | 248044 `<\|endoftext\|>` | null (no BOS) | CORRECT (GGUF convention: use endoftext for null BOS) |
| padding_token_id | 248044 `<\|endoftext\|>` | `<\|endoftext\|>` = 248044 | CORRECT |

No token ID bugs. The Gemma-style templatefix trigger (wrong bos/eos IDs) is NOT present here.

---

## 2. Upstream Template Audit

### 2a. Qwen/Qwen3.6-35B-A3B commit history

- Initial public release: 2026-04-15 (Yingda)
- README upload: 2026-04-22 (VoyagerXHF)
- README update: 2026-04-24 (VoyagerXHF)
- **No commits after 2026-06-01** — no upstream template/tokenizer fixes to catch up with.

The chat_template in the upstream `tokenizer_config.json` (via `chat_template.jinja`) is the same template captured in our GGUF. Our template is current with upstream.

### 2b. llmfan46 discussions — template-related reports

- Discussion #4 (KoboldCPP): User misconfigured the template type (set to "alpaca"). Not a template bug — user error. Resolution: enable Jinja mode.
- Discussion #6 (Not usable): Repetitive output loop. No template cause identified.
- Discussion #7 (Qwen weirdness): Behavioral/quality complaint, not a template format issue.
- **No reports of broken chat template, wrong stop tokens, or metadata errors.**

---

## 3. froggeric/Qwen-Fixed-Chat-Templates Analysis

froggeric v20 (`qwen3.6-froggeric-v20`) is a heavily-modified template addressing 12 categories of failures in the *upstream* Qwen template. Relevant fixes and their applicability to our release:

### 3a. `.replace()` filter — NOT a bug for us

froggeric replaces all `.replace()` calls with `split(x) | join(y)` to avoid crashes in Python-incompatible Jinja engines (e.g., older minijinja).

**Our llama.cpp jinja engine** (`/home/deucebucket/ai-drive/llama-prismml/common/jinja/value.cpp`) implements `.replace()` at line 623. Confirmed supported. No crash risk.

Our template uses `.replace()` at 6 locations (lines 66, 70, 83, 87, 109, 114). All safe.

### 3b. `loop.previtem` — NOT a bug for us

Our template uses `loop.previtem` at lines 197, 200 (tool message ordering check).

Our llama.cpp jinja (`/home/deucebucket/ai-drive/llama-prismml/common/jinja/runtime.h`) implements `previtem` at line 567. Confirmed supported. No crash risk.

### 3c. `preserve_thinking` default — intentional difference

| | Our template | froggeric v20 |
|--|-------------|---------------|
| Default | `default(true)` | `default(false)` |

froggeric defaults to false to prevent KV cache invalidation in agentic/tool loops. For the **Heretic inference model** (not an agent framework template), `preserve_thinking=true` is correct — users expect thinking from prior turns to appear in context. This is an intentional behavioral choice, not a bug.

### 3d. Empty think tag format — cosmetic difference, not a shipping blocker

| | Format |
|--|--------|
| Ours | `<think>\n\n</think>\n\n` (double newline inside) |
| froggeric | `<think>\n</think>\n` (single newline) |

The double newline inside the empty block matters for **training** (creates "empty think poisoning" patterns in in-context learning). For **inference from a frozen GGUF**, both forms cause the same behavior: model sees an empty think block and begins the response. Not a shipping blocker.

### 3e. `ns_flags.enable_thinking is false` check — functionally correct

Our template uses Jinja identity check `is false` rather than `not`. In our jinja engine, `namespace.enable_thinking` is initialized as a boolean and set from the `enable_thinking` kwarg (also boolean). `is false` correctly catches `False` from any caller that passes `enable_thinking=False`. Not a bug for real usage.

### 3f. Tool instruction text — cosmetic/preference difference

froggeric enriches the tool instructions with guidance on using `<think>` blocks for planning before tool calls. Our template uses the Qwen upstream instruction text. Both produce valid tool call sequences. The froggeric instructions produce better agentic behavior at the cost of longer system prompts.

**For the Heretic model** (focused on creative/reasoning use, not agentic frameworks), the upstream instruction text is acceptable.

---

## 4. Diff Summary

### Differences from froggeric v20 that are **upstream-fix-we-are-missing**

None. Every difference between our template and froggeric v20 falls into one of:
- (a) Our jinja engine supports the "broken" feature → not broken for us
- (b) Intentional behavior choice appropriate for Heretic (preserve_thinking default)
- (c) Cosmetic (empty think newline count)
- (d) Agentic enhancement not needed for this model's use case

### Differences from upstream Qwen3.6 template

Our template IS the upstream Qwen3.6 template. There have been zero upstream template commits since initial release (2026-04-15). No fixes to catch up with.

---

## 5. Verdict

**SHIP-AS-IS**

Justification:
1. Token IDs are correct: eos=248046 (`<|im_end|>`), bos/pad per GGUF convention.
2. Chat template is current with upstream Qwen3.6 (no upstream fixes to catch up with).
3. No template bugs reported in llmfan46 discussions.
4. The froggeric v20 improvements either don't apply to our jinja engine, or are behavioral choices not appropriate for the Heretic model's use case (agentic loop robustness vs inference defaults).
5. This is not a Gemma-style token-ID bug. The Gemma templatefix releases were triggered by wrong bos/eos token IDs — a fundamentally different failure mode that is not present here.

**No templatefix release needed.**

---

## 6. Appendix: If a Future templatefix IS Needed

If the model surfaces issues in the wild that trace to template behavior, the recommended fix source is:

- **froggeric/Qwen-Fixed-Chat-Templates** `chat_template.jinja` (`qwen3.6-froggeric-v20`)
- File: `https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja`
- One-line embed version: `chat_template_oneline.txt`

The froggeric template is architecturally superior for agentic use (12 bug categories fixed, error escalation, KV cache preservation, minja compatibility). If we ship a v1.1, embed the froggeric one-liner.
