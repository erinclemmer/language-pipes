---
title: Roadmap
description: What's planned for Language Pipes. Model support, quantization, robustness, and the open research question around hidden-state privacy.
---

Language Pipes is actively developed. This page collects what's planned so you
can see where the project is headed — and where a contribution would land well.
Nothing here is a dated commitment; it's a direction.

Want to help with any of these? Open a
[GitHub Discussion](https://github.com/erinclemmer/language-pipes/discussions) or
an [issue](https://github.com/erinclemmer/language-pipes/issues), and see
[CONTRIBUTING](https://github.com/erinclemmer/language-pipes/blob/main/CONTRIBUTING.md).

## Model support

- **More architectures.** New model families are added regularly. Adding one is
  self-contained and a great first contribution — see
  [adding a new model architecture](https://github.com/erinclemmer/language-pipes/blob/main/CONTRIBUTING.md#adding-a-new-model-architecture)
  and the current [supported models](./model_support.md).

## Quantization and formats

- **INT4 quantization.** An 8-bit path exists today via `LP_8_BIT_MODE`
  (LLM.int8 through `bitsandbytes`); the goal is broader, better-supported
  quantization including INT4. Default inference currently runs in `fp16`.
- **GGUF format support.** Weights currently must be `safetensors`. GGUF support
  would open up the large ecosystem of pre-quantized community models.

## Robustness

- **Failover and rerouting.** Today a layer node dropping mid-generation fails
  the request (see
  [Failure Modes and Limitations](./architecture.md#failure-modes-and-limitations)).
  Automatic rerouting to another node hosting the same layers is a natural next
  step.
- **Integrity of layer computation.** The current threat model names — but does
  not yet solve — the problem of a malicious layer node returning subtly steered
  tensors (Attack Vector #5 in [Privacy](./privacy.md)). Redundant recomputation
  or verifiable-computation approaches are open design work.

## Open research

- **Quantify recovery difficulty vs. capture depth.** The
  [privacy threat model](./privacy.md) argues that inverting hidden states back
  to a prompt becomes harder as capture happens deeper in the network, but this
  has not been measured. Capturing hidden states at layers 1, 5, 10, and 20,
  attempting recovery, and reporting success rates would turn an expectation into
  a result — and is a contribution Petals never made.

## Good first issues

Several of the items above are well-shaped for a first contribution; especially
**adding a model architecture**, which has a natural, ready supply of candidates.
Look for the [`good first issue`](https://github.com/erinclemmer/language-pipes/labels/good%20first%20issue)
label on the issue tracker.
