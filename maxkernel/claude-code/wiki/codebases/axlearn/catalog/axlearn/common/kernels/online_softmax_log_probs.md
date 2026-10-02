---
title: 'Module: axlearn/common/kernels/online_softmax_log_probs.py'
type: catalog
provenance: extracted
module: axlearn/common/kernels/online_softmax_log_probs.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kernels.online_softmax_log_probs`/
symbols:
  online_softmax_log_probs_pallas: online_softmax_log_probs_pallas().
  _online_softmax_log_probs_kernel: _online_softmax_log_probs_kernel().
  _online_softmax_log_probs_kernel._finalize: _online_softmax_log_probs_kernel()._finalize().
  KernelConfig.top_k: KernelConfig#top_k.
  NEG_INF: NEG_INF.
  KernelOutputs.max_ref: KernelOutputs#max_ref.
  KernelOutputs.sum_exp_ref: KernelOutputs#sum_exp_ref.
  KernelOutputs.topk_vals_ref: KernelOutputs#topk_vals_ref.
  KernelOutputs.topk_idx_ref: KernelOutputs#topk_idx_ref.
  _online_softmax_log_probs_kernel._init: _online_softmax_log_probs_kernel()._init().
  KernelConfig.dot_dtype: KernelConfig#dot_dtype.
  _pad_to_tile_boundaries: _pad_to_tile_boundaries().
  KernelOutputs: KernelOutputs#
  KernelConfig: KernelConfig#
  KernelConfig.vocab_size: KernelConfig#vocab_size.
  KernelConfig.tile_v: KernelConfig#tile_v.
---
# Module: [`axlearn/common/kernels/online_softmax_log_probs.py`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py)

## Classes
### `KernelConfig`
- def: [`axlearn/common/kernels/online_softmax_log_probs.py:53`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L53)
- doc: Compile-time constants for the kernel.
- signature: `class KernelConfig:`
- members:
  - `dot_dtype` — [`L59`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L59)
  - `tile_v` — [`L57`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L57)
  - `top_k` — [`L58`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L58)
  - `vocab_size` — [`L56`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L56)
- used by: [`online_softmax_log_probs_pallas`](online_softmax_log_probs.md#online_softmax_log_probs_pallas), [`_online_softmax_log_probs_kernel`](online_softmax_log_probs.md#_online_softmax_log_probs_kernel), [`_finalize`](online_softmax_log_probs.md#_online_softmax_log_probs_kernel._finalize), [`_init`](online_softmax_log_probs.md#_online_softmax_log_probs_kernel._init)

### `KernelOutputs`
- def: [`axlearn/common/kernels/online_softmax_log_probs.py:43`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L43)
- doc: HBM output refs written on the last vocab chunk.
- signature: `class KernelOutputs:`
- members:
  - `max_ref` — [`L46`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L46)
  - `sum_exp_ref` — [`L47`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L47)
  - `topk_idx_ref` — [`L49`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L49)
  - `topk_vals_ref` — [`L48`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L48)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor)
- used by: [`online_softmax_log_probs_pallas`](online_softmax_log_probs.md#online_softmax_log_probs_pallas), [`_online_softmax_log_probs_kernel`](online_softmax_log_probs.md#_online_softmax_log_probs_kernel), [`_finalize`](online_softmax_log_probs.md#_online_softmax_log_probs_kernel._finalize)

## Functions
- `_finalize()` — [`L173`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L173)
- `_init()` — [`L83`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L83)
- `_online_softmax_log_probs_kernel(x_ref, weight_ref, outputs: KernelOutputs, running_max, running_sum_exp, topk_vals_scratch, topk_idx_scratch, *, cfg: KernelConfig)` — [`L67`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L67)
- `_pad_to_tile_boundaries(x: Tensor, weight: Tensor, *, tile_s: int, tile_v: int)` — [`L186`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L186) — Pads x and weight to tile boundaries.
- `online_softmax_log_probs_pallas(x: Tensor, weight: Tensor, *, top_k: int = 0, tile_s: int = 128, tile_v: int = 1024, interpret: bool = False)` — [`L213`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L213) — Fused streaming log-normalizer and top-k via Pallas (TPU).

## Module values
- `NEG_INF` — [`L33`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs.py#L33)

