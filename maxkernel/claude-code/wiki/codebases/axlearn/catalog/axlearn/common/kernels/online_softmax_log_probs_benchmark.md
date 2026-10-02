---
title: 'Module: axlearn/common/kernels/online_softmax_log_probs_benchmark.py'
type: catalog
provenance: extracted
module: axlearn/common/kernels/online_softmax_log_probs_benchmark.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kernels.online_softmax_log_probs_benchmark`/
symbols:
  TOP_K: TOP_K.
  N_ITERS: N_ITERS.
  BS_BASELINE: BS_BASELINE.
  BS_ONLINE_SOFTMAX: BS_ONLINE_SOFTMAX.
  TILE_S: TILE_S.
  TILE_V: TILE_V.
  PALLAS_INTERPRET: PALLAS_INTERPRET.
  baseline: baseline().
  bench: bench().
  make_inputs: make_inputs().
  main: main().
---
# Module: [`axlearn/common/kernels/online_softmax_log_probs_benchmark.py`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py)

## Functions
- `baseline(x, weight, target_ids, top_k)` — [`L61`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L61) — Baseline: full matmul + log_softmax + gather + top_k.
- `bench(fn: Callable, n: int=N_ITERS)` — [`L71`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L71) — Run fn n times, return (min_ms, all_ms).
- `main(_)` — [`L91`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L91)
- `make_inputs(batch_size)` — [`L82`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L82)

## Module values
- `BS_BASELINE` — [`L49`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L49)
- `BS_ONLINE_SOFTMAX` — [`L50`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L50)
- `N_ITERS` — [`L46`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L46)
- `PALLAS_INTERPRET` — [`L57`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L57)
- `TILE_S` — [`L53`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L53)
- `TILE_V` — [`L54`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L54)
- `TOP_K` — [`L45`](../../../../../../../raw/code/axlearn/axlearn/common/kernels/online_softmax_log_probs_benchmark.py#L45)

