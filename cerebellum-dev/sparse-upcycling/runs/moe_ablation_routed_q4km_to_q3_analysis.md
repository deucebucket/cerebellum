# Cerebellum Ablation Analysis

- input: `/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/sparse-upcycling/runs/moe_ablation_routed_q4km_to_q3.json`
- rows: 64 tested, 0 skipped
- overrides: 39 tensors
- counts: {'demotable': 39, 'tolerant': 16, 'sacred': 9, 'critical': 0}

| class | tensor | weighted | worst | worst domain |
| --- | --- | ---: | ---: | --- |
| sacred | `blk.14.ffn_gate_up_exps.weight` | +3.968% | +5.180% | reasoning |
| sacred | `blk.0.ffn_down_exps.weight` | +2.313% | +4.050% | reasoning |
| sacred | `blk.10.ffn_gate_up_exps.weight` | +2.369% | +3.546% | reasoning |
| sacred | `blk.5.ffn_gate_up_exps.weight` | +1.597% | +3.489% | reasoning |
| sacred | `blk.6.ffn_gate_up_exps.weight` | +2.342% | +3.146% | reasoning |
| sacred | `blk.7.ffn_gate_up_exps.weight` | +1.385% | +2.587% | reasoning |
| sacred | `blk.5.ffn_down_exps.weight` | +1.014% | +2.385% | chat |
| sacred | `blk.1.ffn_gate_up_exps.weight` | +1.769% | +2.344% | reasoning |
| sacred | `blk.6.ffn_down_exps.weight` | +1.732% | +1.830% | code |
| tolerant | `blk.3.ffn_gate_up_exps.weight` | +0.970% | +1.909% | reasoning |
| tolerant | `blk.8.ffn_down_exps.weight` | +0.883% | +1.833% | reasoning |
| tolerant | `blk.3.ffn_down_exps.weight` | +0.905% | +1.738% | reasoning |
| tolerant | `blk.12.ffn_down_exps.weight` | +0.624% | +1.624% | code |
| tolerant | `blk.19.ffn_gate_up_exps.weight` | +0.663% | +1.550% | reasoning |
| tolerant | `blk.13.ffn_gate_up_exps.weight` | +0.831% | +1.530% | chat |
| tolerant | `blk.2.ffn_gate_up_exps.weight` | +0.011% | +1.399% | chat |
| tolerant | `blk.0.ffn_gate_up_exps.weight` | +1.012% | +1.341% | reasoning |
| tolerant | `blk.2.ffn_down_exps.weight` | +0.996% | +1.282% | reasoning |
| tolerant | `blk.13.ffn_down_exps.weight` | +0.617% | +1.221% | chat |
| tolerant | `blk.11.ffn_down_exps.weight` | +0.414% | +1.067% | reasoning |
| tolerant | `blk.7.ffn_down_exps.weight` | +0.503% | +1.054% | chat |
| tolerant | `blk.14.ffn_down_exps.weight` | +0.098% | +1.053% | reasoning |
| tolerant | `blk.15.ffn_gate_up_exps.weight` | +0.140% | +1.039% | chat |
| tolerant | `blk.4.ffn_down_exps.weight` | +0.584% | +0.927% | reasoning |
| tolerant | `blk.17.ffn_gate_up_exps.weight` | +0.532% | +0.853% | reasoning |
| demotable | `blk.4.ffn_gate_up_exps.weight` | -0.011% | +0.928% | chat |
| demotable | `blk.16.ffn_down_exps.weight` | -0.018% | +0.899% | chat |
| demotable | `blk.17.ffn_down_exps.weight` | +0.466% | +0.834% | chat |
| demotable | `blk.10.ffn_down_exps.weight` | -0.035% | +0.817% | chat |
| demotable | `blk.16.ffn_gate_up_exps.weight` | -1.216% | +0.788% | chat |
| demotable | `blk.8.ffn_gate_up_exps.weight` | +0.343% | +0.785% | reasoning |
| demotable | `blk.11.ffn_gate_up_exps.weight` | -0.113% | +0.753% | chat |
| demotable | `blk.1.ffn_down_exps.weight` | -0.062% | +0.649% | chat |
| demotable | `blk.24.ffn_gate_up_exps.weight` | +0.162% | +0.570% | reasoning |
| demotable | `blk.23.ffn_gate_up_exps.weight` | +0.229% | +0.528% | chat |
| demotable | `blk.9.ffn_down_exps.weight` | +0.310% | +0.521% | reasoning |
| demotable | `blk.28.ffn_gate_up_exps.weight` | +0.441% | +0.469% | chat |
| demotable | `blk.31.ffn_down_exps.weight` | +0.023% | +0.468% | code |
| demotable | `blk.29.ffn_gate_up_exps.weight` | +0.158% | +0.468% | chat |
| demotable | `blk.18.ffn_down_exps.weight` | -0.057% | +0.462% | chat |
| demotable | `blk.24.ffn_down_exps.weight` | +0.244% | +0.457% | reasoning |
| demotable | `blk.21.ffn_down_exps.weight` | +0.007% | +0.452% | chat |
| demotable | `blk.30.ffn_gate_up_exps.weight` | +0.257% | +0.448% | reasoning |
| demotable | `blk.27.ffn_down_exps.weight` | +0.016% | +0.446% | reasoning |
| demotable | `blk.22.ffn_down_exps.weight` | +0.279% | +0.440% | reasoning |
| demotable | `blk.18.ffn_gate_up_exps.weight` | +0.031% | +0.409% | reasoning |
| demotable | `blk.15.ffn_down_exps.weight` | -0.259% | +0.395% | chat |
| demotable | `blk.20.ffn_down_exps.weight` | -0.001% | +0.376% | reasoning |
| demotable | `blk.30.ffn_down_exps.weight` | -0.169% | +0.371% | chat |
| demotable | `blk.12.ffn_gate_up_exps.weight` | -0.473% | +0.354% | chat |
| demotable | `blk.26.ffn_gate_up_exps.weight` | +0.037% | +0.336% | chat |
| demotable | `blk.23.ffn_down_exps.weight` | -0.287% | +0.289% | chat |
| demotable | `blk.26.ffn_down_exps.weight` | +0.030% | +0.288% | reasoning |
| demotable | `blk.22.ffn_gate_up_exps.weight` | +0.122% | +0.283% | chat |
| demotable | `blk.9.ffn_gate_up_exps.weight` | +0.114% | +0.268% | reasoning |
| demotable | `blk.28.ffn_down_exps.weight` | -0.366% | +0.236% | chat |
| demotable | `blk.31.ffn_gate_up_exps.weight` | -0.052% | +0.195% | chat |
| demotable | `blk.25.ffn_down_exps.weight` | -0.134% | +0.168% | chat |
| demotable | `blk.25.ffn_gate_up_exps.weight` | -0.045% | +0.122% | reasoning |
| demotable | `blk.29.ffn_down_exps.weight` | -0.101% | +0.120% | reasoning |
| demotable | `blk.20.ffn_gate_up_exps.weight` | -0.043% | +0.025% | reasoning |
| demotable | `blk.21.ffn_gate_up_exps.weight` | -0.199% | -0.114% | reasoning |
| demotable | `blk.19.ffn_down_exps.weight` | -0.479% | -0.166% | chat |
| demotable | `blk.27.ffn_gate_up_exps.weight` | -0.442% | -0.264% | reasoning |
