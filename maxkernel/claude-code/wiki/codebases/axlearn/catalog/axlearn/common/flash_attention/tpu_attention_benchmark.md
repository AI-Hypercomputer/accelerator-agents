---
title: 'Module: axlearn/common/flash_attention/tpu_attention_benchmark.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/tpu_attention_benchmark.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.tpu_attention_benchmark`/
symbols:
  FLAGS: FLAGS.
  _BENCHMARK_CONFIGS: _BENCHMARK_CONFIGS.
  _time_call: _time_call().
  _benchmark: _benchmark().
  _print_summary: _print_summary().
  main: main().
---
# Module: [`axlearn/common/flash_attention/tpu_attention_benchmark.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_benchmark.py)

## Functions
- `_benchmark(*, batch_size: int, seq_len: int, block_size: int, num_heads: int, per_head_dim: int, num_kv_heads: Optional[int]=None, kv_cache_type: Optional[type[BaseKVCache]]=KVCache, causal: bool=True, use_bias: bool=False, sliding_window_size: Optional[int]=None, page_size: Optional[int]=None)` — [`L70`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_benchmark.py#L70) — Benchmarks TPU FlashAttention vs reference impl.
- `_print_summary(all_results: dict)` — [`L195`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_benchmark.py#L195) — Prints a formatted summary of benchmark results.
- `_time_call(fn: Callable, *, num_iters: int=10, warmup: int=3)` — [`L54`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_benchmark.py#L54) — Times average execution time for fn call over num_iters after warmup.
- `main(_)` — [`L230`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_benchmark.py#L230)

## Module values
- `FLAGS` — [`L38`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_benchmark.py#L38)
- `_BENCHMARK_CONFIGS` — [`L41`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_attention_benchmark.py#L41)

