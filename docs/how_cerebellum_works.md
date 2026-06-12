# How Cerebellum Works


deucebucket

The goal has not changed since the first build: make models smaller while
losing as little measured ability as possible, for people on 8 to 12 GB cards.

## The problem with standard quants

A standard GGUF quant assigns precision by rule. Every tensor of a given kind
gets the format the recipe says it gets, and the recipe was written once, for
all models. An importance matrix improves the rounding inside each format, but
the format assignment itself, which tensor gets how many bits, is still a
fixed rule. Nothing in it is measured from the specific model being quantized.

The rules are sensible averages, and conservative ones, so they mostly work.
Conservative rules also produce large files, and large files do not fit on
the 8 to 12 GB cards this project is for.

Cerebellum replaces the fixed rule with measurements taken from the specific
model being quantized.

## What I do

I take a part of the model, crush it to very low precision while leaving
everything else alone, and measure what breaks. Then I do it again for the
next part. The result is a damage map: some pieces measure as fragile, some
measure as indifferent, and some measure better after being crushed. That
last category has shown up in dense models, in mixture-of-experts models, and
in hybrid attention models. One interpretation is that the rounding acts like
regularization. That is a hypothesis, and I have not proven it. The
measurements are recorded either way.

With the map in hand, a fixed file size budget gets spent where the
measurements say. Parts that measured as fragile keep their precision. Parts
that measured as tolerant take the deepest cuts.

The output is a normal GGUF built with stock llama.cpp, with no fork and no
custom runtime. The measurements and builds run on one RTX 3090.

## Why each model gets its own measurements

In the models measured so far, what held in one architecture did not hold in
the next. Things the measurements showed that I would not have guessed:

- Some parameter types produced NaN output below a precision floor, with no
  gradual degradation leading up to it. The floor only showed up in the
  measurements once a build crossed it.
- In the mixture-of-experts models I measured, the fragile part was the
  opposite of what dense-model intuition says it should be.
- In one dense model, lowering the precision of attention tensors improved
  perplexity. I would not have tried that without the ablation data in front
  of me.

Every base model gets its own map, because the maps measured so far have not
transferred between architectures.

## The gates

At one point I built a search tool that optimized perplexity alone. On a test
model it improved wiki perplexity 35 percent, grew the file by a gigabyte, and
dropped HumanEval+ by 14 points. That tool is deprecated. Since then,
perplexity has been used to detect damage and never to pass a build on its
own.

So a release candidate gets benchmarked, ARC, HellaSwag, MMLU-Redux,
HumanEval+, next to a uniform-quant baseline, and the comparison goes on the
card, wins and losses both. Finished builds have been rejected at this stage.
One improved perplexity while regressing task benchmarks, so it never became
a release. Another benchmarked badly for reasons that traced back to the
source weights I had built on rather than to the quantization. I kept the
notes and never uploaded the model.

After every benchmark run I pull wrong answers and read them. That habit
exists because my early published HumanEval rows were too low: a bug in my
harness stripped indentation from generated code, which cost about 6 points
on the 27B build, and separate bugs misjudged 19 ARC answers and counted 108
empty HellaSwag responses as wrong. I corrected the scores and documented the
bugs on the model cards. So far, every bad number I have published traced
back to my harness or my process. None traced back to a model.

## The 26B, v1 to v6

The Gemma 4 26B-A4B build took six tries, which makes it a useful record of
the process. Six candidate precision maps, v1 through v6, each built into a
full GGUF and measured.

- v1 was the first map drawn from the ablation measurements.
- v2 tested a reduced version of the map. It was not released.
- v3 and v4 narrowed the map: fewer overrides, more targeted.
- v5 raised precision in a handful of additional places. Perplexity
  improved. HumanEval and MMLU-Redux regressed. It was never released.
- v6 adjusted that trade based on further measurement, and shipped.
- v6.1, three weeks later, kept the v6 tensor allocation with zero tensor
  changes. It only updated chat-template and runtime metadata.

The live card shows where v6 landed: 11 GB, winning ARC-Challenge against a
local uniform Q3_K_M baseline (95.56 vs 95.22) and losing HellaSwag (84.55 vs
86.57) and MMLU-Redux (71.33 vs 73.67) to it. The losses are printed on the
card next to the wins.

## Results

Per-question benchmark outputs for the released models are public in the
project repo, and the model cards link to their benchmark artifacts. Two
examples.

The Qwen 3.6 27B build is 12 GB and scored 81.1 percent HumanEval pass@1. A
uniform Q2_K imatrix quant of the same model, two gigabytes smaller, scored
47.0 on the same harness.

The Qwen 3.6 35B-A3B build is 11 GB; the stock Q3_K_M is 15.6 GB. On
HumanEval+ the 11 GB build scored 65.2 against the Q3_K_M's 56.7. It loses
ARC-Challenge to that same baseline, 95.8 vs 96.1, and that is on the card
too.

Everything was measured with the same harness on the same machine at
temperature 0. The pattern so far: short-answer benchmarks barely move
between methods, and the larger gaps show up on code generation.

## What I'm not writing down yet

How the parts get chosen, what the thresholds are, what the search looks
like, and what the calibration setup is. That stays private for now. The
gates are the part of the method that takes the longest to get right, and a
version of this without them produces worse models that still look fine on
perplexity. The rejected builds above are the record of that. I would rather
publish the method once, complete, than have a shortcut version of it
circulate first.

The plan is a full writeup alongside a future release. Until then, the
discussions on the model pages are open and I answer them there. The method
writeup will follow.
