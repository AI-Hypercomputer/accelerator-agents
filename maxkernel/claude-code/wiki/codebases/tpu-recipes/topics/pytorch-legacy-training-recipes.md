---
title: "Topic: PyTorch/XLA legacy training recipes (archived)"
type: topic
updated: 2026-07-03
---
# PyTorch/XLA legacy training recipes (archived)

## Overview

`training/archive/` holds 12 PyTorch/XLA training recipes (Stable Diffusion 2, Llama3.0-8B/70B,
Llama3.1-405B, Mixtral-8x7B — each on Trillium, three also on v5p) that the repo explicitly marks as
**reference-only and no longer maintained**
([deprecation notice](../sources/training-archive-README.md)). They predate the MaxText-based recipe
family and represent the launch pattern this repo has since moved away from: a Hugging Face
Transformers model run directly under PyTorch/XLA, rather than a MaxText/JAX config object.

## Two launch patterns per model: GCE vs. XPK

Every Llama/Mixtral model here has two sibling recipes differing only in how the workload is
launched — `GCE/` runs directly on a single Compute-Engine-provisioned TPU VM, `XPK/` runs on a
multi-pod GKE cluster via XPK (cluster creation, virtualenv setup, workload launch, and — as an
explicit optional last step — metric processing). The Diffusion-2 and Llama2-7B (v5p) recipes have
no XPK sibling — only a single GCE-style recipe — consistent with those being smaller,
single-host-scale workloads that don't need multi-pod orchestration.

## Recipe matrix

| Model | Hardware | Launch pattern | Recipe |
|---|---|---|---|
| Stable Diffusion 2 | Trillium (v6e) | GCE-style (single recipe) | [recipe](src:training/archive/trillium/Diffusion-2-PyTorch/README.md#environment-setup) |
| Stable Diffusion 2 | v5p | GCE-style (single recipe) | [recipe](src:training/archive/v5p/Diffusion-2-PyTorch/README.md#environment-setup) |
| Llama3.0-8B | Trillium | GCE | [recipe](src:training/archive/trillium/Llama3.0-8B-PyTorch/GCE/README.md#environment-setup) |
| Llama3.0-8B | Trillium | XPK (multipod) | [recipe](src:training/archive/trillium/Llama3.0-8B-PyTorch/XPK/README.md#environment-steup) |
| Llama3.0-70B | Trillium | GCE | [recipe](src:training/archive/trillium/Llama3.0-70B-PyTorch/GCE/README.md#environment-setup) |
| Llama3.0-70B | Trillium | XPK (multipod) | [recipe](src:training/archive/trillium/Llama3.0-70B-PyTorch/XPK/README.md#environment-steup) |
| Llama3.1-405B | Trillium (1 pod) | GCE | [recipe](src:training/archive/trillium/Llama3.1-405B-PyTorch/GCE/README.md#environment-setup) |
| Llama3.1-405B | Trillium | XPK (multipod) | [recipe](src:training/archive/trillium/Llama3.1-405B-PyTorch/XPK/README.md#environment-setup) |
| Llama2-7B | v5p | GCE-style (single recipe) | [recipe](src:training/archive/v5p/Llama2-7B-PyTorch/README.md#environment-setup) |
| Mixtral-8x7B | Trillium (v6e) | GCE | [recipe](src:training/archive/trillium/Mixtral-8x7B-Pytorch/GCE/README.md#environment-setup) |
| Mixtral-8x7B | Trillium | XPK (multipod) | [recipe](src:training/archive/trillium/Mixtral-8x7B-Pytorch/XPK/README.md#environment-steup) |
| Mixtral-8x7B | v5p | GCE-style (single recipe) | [recipe](src:training/archive/v5p/Mixtral-8x7B-PyTorch/README.md#environment-setup) |

> [!inferred] Every XPK-launched recipe's own anchor for the environment-setup step is literally
> spelled `environment-steup` (a typo in the source heading) except the Llama3.1-405B/XPK recipe,
> which spells it correctly — a small but consistent signal that the 405B recipe was authored or
> revised later than its 8B/70B/Mixtral siblings.

## See also
- [MaxText training recipe pattern (Trillium + v5p)](maxtext-training-recipes.md) — the actively
  maintained recipe family that replaced this one; same models (Llama3.x, Mixtral-8x7B), different
  framework and launch config surface (`MaxTextModel` vs. raw PyTorch/XLA scripts).

## Sources
- [training/archive/README.md](../sources/training-archive-README.md)
- `training/archive/{trillium,v5p}/*/README.md` (12 recipes, cited inline above)
