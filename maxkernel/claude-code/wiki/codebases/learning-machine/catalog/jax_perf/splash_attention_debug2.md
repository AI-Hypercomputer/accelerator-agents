---
title: 'Module: jax_perf/splash_attention_debug2.py'
type: catalog
provenance: extracted
module: jax_perf/splash_attention_debug2.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `jax_perf.splash_attention_debug2`/
symbols:
  splash_single_device: splash_single_device().
  main: main().
  TimeMask.timestamp: TimeMask#timestamp.
  TimeMask.offset: TimeMask#offset.
  TimeMask.causal_mask_function: TimeMask#causal_mask_function().
  TimeMask.__init__: TimeMask#__init__().
  TimeMask.__eq__: TimeMask#__eq__().
  TimeMask.__hash__: TimeMask#__hash__().
  create_kernel_blocks: create_kernel_blocks().
  TimeMask: TimeMask#
  main.flash: main().flash().
---
# Module: [`jax_perf/splash_attention_debug2.py`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py)

## Classes
### `TimeMask`  ·  implements/extends _ComputableMask
- def: [`jax_perf/splash_attention_debug2.py:100`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L100)
- doc: Lazy causal mask, prevents the model from attending to future tokens.
- signature: `class TimeMask(splash_attention_mask._ComputableMask):`
- members:
  - `causal_mask_function(q_ids, kv_ids)` — [`L122`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L122)
  - `offset` — [`L110`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L110)
  - `timestamp` — [`L119`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L119)
- protocol/private: `__eq__`[`L154`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L154), `__hash__`[`L164`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L164), `__init__`[`L112`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L112)
- used by: [`splash_single_device`](splash_attention_debug2.md#splash_single_device)

## Functions
- `create_kernel_blocks(query: jax.Array, key: jax.Array, q_block_size: Optional[int] = None, kv_block_size: Optional[int] = None, q_block_repeats: Optional[int] = None, kv_block_repeats: Optional[int] = None, fuse_if_possible: bool = True, ensure_block_sizes: bool = True)` — [`L18`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L18)
- `flash(q, k, v)` — [`L231`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L231)
- `main()` — [`L197`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L197)
- `splash_single_device(query, key, value, timestamp)` — [`L172`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug2.py#L172)

