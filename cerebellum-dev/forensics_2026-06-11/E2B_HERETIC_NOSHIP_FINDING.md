# E2B Heretic — NO-SHIP Finding (2026-06-11)

**Verdict: gate FAIL, abliteration-inherent damage. Not shipped.**

Source: coder3101/gemma-4-E2B-it-heretic (reported KL 0.1651 — 28x the E4B's 0.0058).
Build: v2 recipe transfer (Q3_K_M base + 3 ffn_gate Q2_K demotions, no imatrix), byte-class identical to stock v2 (3.0 GiB).

## Evidence (all same invocation: wikitext-test.txt, ctx 2048, 32 chunks, RTX 3090)
| Measurement | Value |
|---|---|
| Stock v2 quant PPL | 118.05 ± 3.14 |
| Heretic **F16** PPL | **150.16 ± 4.40** (abliteration cost at full precision: +27% over stock QUANT) |
| Heretic quant PPL | 177.27 ± 4.91 |
| Stock same-night EvalPlus (chat no-think, patched harness) | 53.66 / 48.78 |
| Heretic EvalPlus (same harness) | **22.56 / 21.95** (−31 pts) |
| ARC / HellaSwag / MMLU | 70.39 / 50.17 / 45.42 vs stock published 71.9 / 50.0 / 47.4 (MC mostly survives) |

## Interpretation — KL tolerance ceiling for heretic transfer
Same protocol, same night: E4B heretic (KL 0.0058) BEAT stock on 3/4 benches; 35B (KL 0.0015) and 27B heretics rode within ~2.5 pts. E2B (KL 0.1651) collapsed on code while MC benches survived — heavy abliteration disproportionately destroys code generation in small models. **Rule adopted: check the source's reported KL before transfer; treat anything ≳0.05 as suspect, require F16 PPL + code-bench screen before quantizing.**

## Future options
- llmfan46/gemma-4-E2B-it-ultra-uncensored-heretic as alternate source (KL unknown — screen first)
- Request/wait for a gentler E2B ablation
- Evidence archived: cerebellum-gemma4-e2b-heretic/ (dev repo) + staging dir kept
