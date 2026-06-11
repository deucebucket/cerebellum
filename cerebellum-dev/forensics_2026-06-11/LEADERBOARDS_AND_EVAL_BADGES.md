# Leaderboards, Eval Badges, and Free Visibility for Cerebellum GGUFs

**Date:** 2026-06-11  
**Scope:** Free benchmark visibility strategies for deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF and upcoming Gemma4/Qwen3.6 Cerebellum quants.

---

## 1. model-index YAML — "Evaluation Results" Panel on HF

### How It Works (2026)

HF now supports **two parallel systems**:

**Legacy (still works):** `model-index:` block in YAML front matter of `README.md`. Renders as the "Evaluation results" widget on the model page. Sourced from a real TheBloke-era pattern.

**New (2026):** `.eval_results/*.yaml` files in the model repo. Results appear on the model card, link to benchmark dataset leaderboards, and show community/verified badges. These are the "Evaluation Results" panel in the new HF UI. Anyone can submit via PR to any repo — results show as "community" while the PR is open.

### Exact Legacy Schema (from TheBloke/notus-7B-v1-GGUF)

```yaml
model-index:
- name: MODEL_REPO_NAME_HERE
  results:
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: AI2 Reasoning Challenge (25-Shot)
      type: ai2_arc
      config: ARC-Challenge
      split: test
      args:
        num_few_shot: 25
    metrics:
    - name: normalized accuracy
      type: acc_norm
      value: 0.0000
    source:
      name: Cerebellum Benchmark Results
      url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HellaSwag (10-Shot)
      type: hellaswag
      split: validation
      args:
        num_few_shot: 10
    metrics:
    - name: normalized accuracy
      type: acc_norm
      value: 0.0000
    source:
      name: Cerebellum Benchmark Results
      url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: MMLU-Redux (5-Shot)
      type: cais/mmlu
      config: all
      split: test
      args:
        num_few_shot: 5
    metrics:
    - name: accuracy
      type: acc
      value: 0.0000
    source:
      name: Cerebellum Benchmark Results
      url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HumanEval+ (pass@1)
      type: openai_humaneval
      split: test
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.0000
    source:
      name: Cerebellum Benchmark Results
      url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: WikiText-2 Perplexity
      type: wikitext
      config: wikitext-2-raw-v1
      split: test
    metrics:
    - name: perplexity
      type: perplexity
      value: 0.000
    source:
      name: Cerebellum Benchmark Results
      url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
```

### New .eval_results/ Schema (2026, for HF benchmark leaderboard integration)

Place at `.eval_results/cerebellum_benchmarks.yaml` in the model repo:

```yaml
- dataset:
    id: ai2_arc
    task_id: arc_challenge
  value: 0.000
  date: "2026-06-11"
  notes: "25-shot, llama.cpp, lm-eval-harness"
  source:
    url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
    name: Cerebellum ablation-informed mixed-precision results

- dataset:
    id: hellaswag
    task_id: default
  value: 0.000
  date: "2026-06-11"
  source:
    url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
    name: Cerebellum ablation-informed mixed-precision results

- dataset:
    id: TIGER-Lab/MMLU-Pro
    task_id: default
  value: 0.000
  date: "2026-06-11"
  source:
    url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
    name: Cerebellum ablation-informed mixed-precision results

- dataset:
    id: openai/openai_humaneval
    task_id: default
  value: 0.000
  date: "2026-06-11"
  notes: "pass@1, evalplus"
  source:
    url: https://huggingface.co/deucebucket/REPO_NAME/tree/main/benchmark_results
    name: Cerebellum ablation-informed mixed-precision results
```

**Note on WikiText PPL:** There is no HF Benchmark dataset registered for WikiText PPL that would populate the new leaderboard panel. Use only the legacy `model-index:` block for PPL — it renders fine in the old widget. For the `.eval_results/` folder, stick to datasets that have an `eval.yaml` registered as a Benchmark (ai2_arc, hellaswag, MMLU-Pro, HumanEval are all registered).

### Ready-to-paste block for deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF

Replace `0.955` / `0.6829` etc. with actual values. The `source.url` should point to the JSONL/CSV in `benchmark_results/` for that run:

```yaml
model-index:
- name: Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF
  results:
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: AI2 Reasoning Challenge (25-Shot)
      type: ai2_arc
      config: ARC-Challenge
      split: test
      args:
        num_few_shot: 25
    metrics:
    - name: normalized accuracy
      type: acc_norm
      value: 0.955
    source:
      name: Cerebellum v3 Benchmark Results
      url: https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HellaSwag (10-Shot)
      type: hellaswag
      split: validation
      args:
        num_few_shot: 10
    metrics:
    - name: normalized accuracy
      type: acc_norm
      value: 0.000
    source:
      name: Cerebellum v3 Benchmark Results
      url: https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: MMLU-Redux (5-Shot)
      type: cais/mmlu
      config: all
      split: test
      args:
        num_few_shot: 5
    metrics:
    - name: accuracy
      type: acc
      value: 0.000
    source:
      name: Cerebellum v3 Benchmark Results
      url: https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HumanEval+ (pass@1)
      type: openai_humaneval
      split: test
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.6829
    source:
      name: Cerebellum v3 Benchmark Results
      url: https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: WikiText-2 Perplexity
      type: wikitext
      config: wikitext-2-raw-v1
      split: test
    metrics:
    - name: perplexity
      type: perplexity
      value: 7.157
    source:
      name: Cerebellum v3 Benchmark Results
      url: https://huggingface.co/deucebucket/Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF/tree/main/benchmark_results
```

---

## 2. UGI Leaderboard (DontPlanToEnd)

**URL:** https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard

**What it is:** Uncensored General Intelligence leaderboard. Scores models on sensitive/uncensored topic breadth (UGI score) and willingness to engage (W/10 score). Ideal for heretic/abliterated variants.

**Status as of 2026-06-11:**
- Closed for new model additions since ~October 2025
- Owner manually benchmarks all models — no automated submission pipeline
- Switched from llama.cpp/GGUF to **vLLM for inference** — only unquantized bfloat16 safetensors are being used
- GGUF is explicitly NOT the test format anymore
- No formal submission form exists yet ("planned but not implemented")
- Owner is "swamped" with no confirmed reopen date

**GGUF repos accepted?** No. vLLM runs bfloat16 safetensors.

**What you need to get considered:**
- A safetensors base model (the heretic base from llmfan46, not the GGUF)
- Contact DontPlanToEnd directly via HF discussions when the leaderboard reopens
- A discussion thread with "Evaluate these models" posts (see discussion #514) is the informal queue

**llmfan46 presence:** User llmfan46 has an active discussion (#663, ~1 month ago) on the leaderboard — model may be in the queue or already tested. Check discussion #663 for current status.

**Action:** Watch the leaderboard discussions for a reopen announcement. When it reopens, post in a "model request" thread pointing to llmfan46's heretic base model (safetensors). The GGUF repo can be mentioned as the downstream artifact, but it won't be tested directly.

---

## 3. Other Active Free Leaderboards (2026)

### Open LLM Leaderboard 2 (HuggingFace)

**URL:** https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard  
**Status:** Active (v2; v1 is archived)  
**Cost:** Free — runs on HF cluster at no cost to submitter  
**Benchmarks:** IFEval, BBH, MATH, GPQA, MUSR, MMLU-PRO  
**GGUF?** No. Requires safetensors + loads via AutoModel/AutoTokenizer. No `use_remote_code=True` support.  
**Size limits:** float16/bf16 up to 100B params; 4bit up to 560B params  
**Submission:** Go to the Space, fill in the model name (HF repo), select precision, submit. Automated eval queue.  
**Relevance for us:** Submit the heretic safetensors base (llmfan46's repo), not the GGUF. The GGUF card can then link to those results.

### EQ-Bench (eq-bench.com / sam-paech)

**URL:** https://eqbench.com/ | HF Space: https://huggingface.co/spaces/sam-paech/EQ-Bench-Leaderboard  
**Status:** Active. Multiple sub-leaderboards: EQ-Bench 3 (emotional intelligence), Creative Writing v3 (updated March 2026, uses Claude Sonnet 4.6 as judge), Longform Writing, Judgemark, BuzzBench, DiploBench.  
**Cost:** Self-funded project. Running the benchmark yourself costs ~$10-15 on OpenRouter for the full rubric+pairwise eval.  
**GGUF?** Not specified in docs. The repo runs via llama.cpp/OpenRouter-compatible APIs, so GGUF via llama.cpp server is likely feasible. Contact sam_paech on Twitter/X to confirm.  
**Submission process:**
1. Run the benchmark locally using the GitHub repo (https://github.com/EQ-bench/creative-writing-bench)
2. Use the provided `creative_bench_runs.json` file for Elo calculation (required for comparable scores)
3. Submit via email or Twitter DM to @sam_paech with: HF model link, scores, and prompt/generation config used
**Relevance:** Creative Writing v3 is a strong differentiator for heretic/uncensored models. Cerebellum's quality-preserving quantization story fits well here.

### lechmazur/writing Benchmark

**URL:** https://github.com/lechmazur/writing  
**Status:** Active. Updated June 9, 2026 (Claude Fable 5 added).  
**Cost:** Depends on inference cost for the model being evaluated (you run it yourself).  
**GGUF?** Not specified; contact @LechMazur on X.  
**Submission:** Contact @LechMazur on X. No formal form.  
**Relevance:** Top-tier creative writing leaderboard, well-indexed. Heavy compute for a 35B model.

### Open LLM Leaderboard v1

**Status:** ARCHIVED. Cannot accept new submissions. Historical reference only.

### Low-Bit Quantized Open LLM Leaderboard (Intel)

**URL:** https://www.intel.com/content/www/us/en/developer/articles/technical/low-bit-quantized-open-llm-leaderboard.html  
**Status:** Unknown current activity — Intel-run, may be stale. Worth checking.  
**Relevance:** Specifically tests quantized models. Cerebellum's mixed-precision angle is directly on-topic.

### LLM Quant Bench (gguf-bench.com)

**URL:** https://gguf-bench.com/ (403 at time of research — may require login or be invite-only)  
**Status:** Unknown. Site was accessible at some point. Worth retrying.  
**Relevance:** Directly targets GGUF quantization accuracy benchmarking.

---

## 4. HF Jobs / Community Evals — Free Compute?

**Answer:** No free tier. HF Jobs is billed by the minute from first dollar. Cheapest GPU option is Nvidia T4-small at $0.40/hr. There is no free compute allocation for running evals on HF infrastructure.

**ZeroGPU (Spaces):** The free tier gives ~25 min/day of H200 GPU time via ZeroGPU Spaces, but this is for interactive Gradio demos, not batch eval jobs. PRO plan ($9/mo) gets 8x quota. Not suitable for a full HumanEval+ or ARC run.

**Community Eval PRs (free):** The `.eval_results/*.yaml` PR mechanism is free — you run evals locally (on your own hardware) and then submit the YAML result file via a PR to any model repo. No HF compute cost. Results show as "community" badges.

---

## Recommended Action List

### Immediate (this week)

1. **Add `model-index:` block to the Qwen3.6-35B-A3B-Heretic-Cerebellum-GGUF README** using the ready-to-paste YAML above. Fill in real values for: ARC (95.5%), HumanEval+ (68.29%), Wiki PPL (7.157). Run HellaSwag and MMLU to fill those slots. This renders the "Evaluation results" widget on the model page.

2. **Add `.eval_results/cerebellum_benchmarks.yaml`** to the same repo using the new-format template above. This feeds HF's benchmark aggregation system and shows badges. For wiki PPL, use only the legacy `model-index:` block (no registered benchmark dataset for it).

3. **Make sure `benchmark_results/` directory in the repo contains the raw JSONL** so `source.url` links actually resolve to evidence. The widget badge upgrades from "community" to trusted when sources are present.

### Short-term (this month)

4. **Submit the heretic safetensors base (llmfan46's repo) to Open LLM Leaderboard 2** — free, automated, and the GGUF card can cite those results. This gives the Cerebellum card an anchor to a well-known leaderboard.

5. **Contact @sam_paech on X/Twitter for EQ-Bench Creative Writing inclusion.** Run the benchmark locally using the GitHub repo. Cost is ~$10-15 in API spend. This is the highest-signal leaderboard for uncensored/heretic models. Mention the Cerebellum angle (quality-preserving quant).

6. **Watch DontPlanToEnd/UGI-Leaderboard discussions** for reopen announcement. When it opens, post a model request thread pointing to llmfan46's heretic safetensors base.

### Repeat for each Cerebellum release

7. For every new Cerebellum quant (Gemma4, upcoming Qwen3.6 variants): add the model-index YAML block and `.eval_results/` YAML at release time, not after. This ensures the eval widget is present on day one.

8. **Compare vs same-size uniform quant in the model card** (already per memory notes). The model-index block should ideally include a companion row for the baseline so the delta is visible.

---

## Key Facts Summary

| Leaderboard | Free? | GGUF Direct? | Status | Action |
|---|---|---|---|---|
| HF model-index widget | Yes (self-hosted) | Yes | Active | Add YAML to README |
| HF .eval_results/ | Yes (self-hosted) | Yes | Active (2026) | Add YAML file via PR |
| Open LLM Leaderboard 2 | Yes | No (safetensors) | Active | Submit base model |
| UGI Leaderboard | Yes (when open) | No (vLLM bf16) | Closed ~Oct 2025 | Monitor for reopen |
| EQ-Bench Creative Writing | ~$10-15 API cost | Likely yes via llama.cpp | Active | Contact @sam_paech |
| lechmazur/writing | Compute cost only | Unknown | Active | Contact @LechMazur |
| HF Jobs eval compute | No — pay-per-minute | N/A | Active | Skip unless budget |
