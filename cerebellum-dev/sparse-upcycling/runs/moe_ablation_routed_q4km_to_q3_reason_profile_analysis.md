# Cerebellum Ablation Analysis

- input: `cerebellum-dev/sparse-upcycling/runs/moe_ablation_routed_q4km_to_q3.json`
- rows: 64 tested, 0 skipped
- overrides: 39 tensors
- counts: {'demotable': 39, 'tolerant': 16, 'sacred': 9, 'critical': 0}

| class | tensor | weighted | worst | worst domain |
| --- | --- | ---: | ---: | --- |
| sacred | `blk.14.ffn_gate_up_exps.weight` | +4.488% | +5.180% | reasoning |
| sacred | `blk.0.ffn_down_exps.weight` | +3.067% | +4.050% | reasoning |
| sacred | `blk.10.ffn_gate_up_exps.weight` | +2.901% | +3.546% | reasoning |
| sacred | `blk.5.ffn_gate_up_exps.weight` | +2.353% | +3.489% | reasoning |
| sacred | `blk.6.ffn_gate_up_exps.weight` | +2.681% | +3.146% | reasoning |
| sacred | `blk.7.ffn_gate_up_exps.weight` | +1.883% | +2.587% | reasoning |
| sacred | `blk.5.ffn_down_exps.weight` | +1.091% | +2.385% | chat |
| sacred | `blk.1.ffn_gate_up_exps.weight` | +2.066% | +2.344% | reasoning |
| sacred | `blk.6.ffn_down_exps.weight` | +1.728% | +1.830% | code |
| tolerant | `blk.3.ffn_gate_up_exps.weight` | +1.421% | +1.909% | reasoning |
| tolerant | `blk.8.ffn_down_exps.weight` | +1.332% | +1.833% | reasoning |
| tolerant | `blk.3.ffn_down_exps.weight` | +1.323% | +1.738% | reasoning |
| tolerant | `blk.12.ffn_down_exps.weight` | +0.471% | +1.624% | code |
| tolerant | `blk.19.ffn_gate_up_exps.weight` | +1.074% | +1.550% | reasoning |
| tolerant | `blk.13.ffn_gate_up_exps.weight` | +0.942% | +1.530% | chat |
| tolerant | `blk.0.ffn_gate_up_exps.weight` | +1.181% | +1.341% | reasoning |
| tolerant | `blk.2.ffn_down_exps.weight` | +1.114% | +1.282% | reasoning |
| tolerant | `blk.13.ffn_down_exps.weight` | +0.772% | +1.221% | chat |
| tolerant | `blk.11.ffn_down_exps.weight` | +0.714% | +1.067% | reasoning |
| tolerant | `blk.7.ffn_down_exps.weight` | +0.379% | +1.054% | chat |
| tolerant | `blk.14.ffn_down_exps.weight` | +0.531% | +1.053% | reasoning |
| tolerant | `blk.15.ffn_gate_up_exps.weight` | +0.281% | +1.039% | chat |
| tolerant | `blk.4.ffn_down_exps.weight` | +0.745% | +0.927% | reasoning |
| tolerant | `blk.17.ffn_gate_up_exps.weight` | +0.672% | +0.853% | reasoning |
| tolerant | `blk.8.ffn_gate_up_exps.weight` | +0.560% | +0.785% | reasoning |
| demotable | `blk.2.ffn_gate_up_exps.weight` | -0.270% | +1.399% | chat |
| demotable | `blk.4.ffn_gate_up_exps.weight` | -0.047% | +0.928% | chat |
| demotable | `blk.16.ffn_down_exps.weight` | -0.029% | +0.899% | chat |
| demotable | `blk.17.ffn_down_exps.weight` | +0.478% | +0.834% | chat |
| demotable | `blk.10.ffn_down_exps.weight` | +0.257% | +0.817% | chat |
| demotable | `blk.16.ffn_gate_up_exps.weight` | -1.556% | +0.788% | chat |
| demotable | `blk.11.ffn_gate_up_exps.weight` | -0.107% | +0.753% | chat |
| demotable | `blk.1.ffn_down_exps.weight` | -0.148% | +0.649% | chat |
| demotable | `blk.24.ffn_gate_up_exps.weight` | +0.345% | +0.570% | reasoning |
| demotable | `blk.23.ffn_gate_up_exps.weight` | +0.200% | +0.528% | chat |
| demotable | `blk.9.ffn_down_exps.weight` | +0.382% | +0.521% | reasoning |
| demotable | `blk.28.ffn_gate_up_exps.weight` | +0.441% | +0.469% | chat |
| demotable | `blk.31.ffn_down_exps.weight` | +0.016% | +0.468% | code |
| demotable | `blk.29.ffn_gate_up_exps.weight` | +0.249% | +0.468% | chat |
| demotable | `blk.18.ffn_down_exps.weight` | -0.114% | +0.462% | chat |
| demotable | `blk.24.ffn_down_exps.weight` | +0.344% | +0.457% | reasoning |
| demotable | `blk.21.ffn_down_exps.weight` | -0.009% | +0.452% | chat |
| demotable | `blk.30.ffn_gate_up_exps.weight` | +0.350% | +0.448% | reasoning |
| demotable | `blk.27.ffn_down_exps.weight` | +0.208% | +0.446% | reasoning |
| demotable | `blk.22.ffn_down_exps.weight` | +0.353% | +0.440% | reasoning |
| demotable | `blk.18.ffn_gate_up_exps.weight` | +0.179% | +0.409% | reasoning |
| demotable | `blk.15.ffn_down_exps.weight` | -0.267% | +0.395% | chat |
| demotable | `blk.20.ffn_down_exps.weight` | +0.178% | +0.376% | reasoning |
| demotable | `blk.30.ffn_down_exps.weight` | -0.100% | +0.371% | chat |
| demotable | `blk.12.ffn_gate_up_exps.weight` | -0.329% | +0.354% | chat |
| demotable | `blk.26.ffn_gate_up_exps.weight` | +0.094% | +0.336% | chat |
| demotable | `blk.23.ffn_down_exps.weight` | -0.353% | +0.289% | chat |
| demotable | `blk.26.ffn_down_exps.weight` | +0.142% | +0.288% | reasoning |
| demotable | `blk.22.ffn_gate_up_exps.weight` | +0.178% | +0.283% | chat |
| demotable | `blk.9.ffn_gate_up_exps.weight` | +0.192% | +0.268% | reasoning |
| demotable | `blk.28.ffn_down_exps.weight` | -0.406% | +0.236% | chat |
| demotable | `blk.31.ffn_gate_up_exps.weight` | -0.021% | +0.195% | chat |
| demotable | `blk.25.ffn_down_exps.weight` | -0.060% | +0.168% | chat |
| demotable | `blk.25.ffn_gate_up_exps.weight` | +0.033% | +0.122% | reasoning |
| demotable | `blk.29.ffn_down_exps.weight` | +0.007% | +0.120% | reasoning |
| demotable | `blk.20.ffn_gate_up_exps.weight` | -0.009% | +0.025% | reasoning |
| demotable | `blk.21.ffn_gate_up_exps.weight` | -0.159% | -0.114% | reasoning |
| demotable | `blk.19.ffn_down_exps.weight` | -0.437% | -0.166% | chat |
| demotable | `blk.27.ffn_gate_up_exps.weight` | -0.354% | -0.264% | reasoning |
