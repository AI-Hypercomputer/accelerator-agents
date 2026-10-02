---
title: 'Module: jax_perf/splash_attention_debug.py'
type: catalog
provenance: extracted
module: jax_perf/splash_attention_debug.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `jax_perf.splash_attention_debug`/
symbols:
  main: main().
  SplashAttention.__call__: SplashAttention#__call__().
  SplashAttention._qkvo_partition_spec: SplashAttention#_qkvo_partition_spec().
  SplashAttention.splash_attention_helper: SplashAttention#splash_attention_helper().
  SplashAttention._segment_ids_spec: SplashAttention#_segment_ids_spec().
  SplashAttention.create_kernel_blocks: SplashAttention#create_kernel_blocks().
  CausalMask.__init__: CausalMask#__init__().
  SplashAttention.mesh: SplashAttention#mesh.
  SplashAttention.window_size: SplashAttention#window_size.
  CausalMask.offset: CausalMask#offset.
  SplashAttention.create_mask: SplashAttention#create_mask().
  SplashAttention.dtype_block_factor: SplashAttention#dtype_block_factor().
  SplashAttention.sequence_block_factor: SplashAttention#sequence_block_factor().
  _unflatten_mask: _unflatten_mask().
  SplashAttention._batch_partitions: SplashAttention#_batch_partitions().
  CausalMask: CausalMask#
  SplashAttention.setup: SplashAttention#setup().
  CausalMask.__eq__: CausalMask#__eq__().
  CausalMask.__hash__: CausalMask#__hash__().
  SplashAttention._sequence_partitions: SplashAttention#_sequence_partitions().
  SplashAttention: SplashAttention#
  SplashAttention.static_helper: SplashAttention#static_helper().
  SplashAttention._calculate_scale_factor: SplashAttention#_calculate_scale_factor().
  SplashAttention.dtype: SplashAttention#dtype().
  CausalMask.causal_mask_function: CausalMask#causal_mask_function().
  _flatten_mask: _flatten_mask().
  main.create_timestep_mask: main().create_timestep_mask().
  main.static_apply: main().static_apply().
  main.static_apply_causal: main().static_apply_causal().
  main.flash: main().flash().
  SplashAttention.is_tpu_attention: SplashAttention#is_tpu_attention().
  CausalMask.id_to_rank: CausalMask#id_to_rank.
---
# Module: [`jax_perf/splash_attention_debug.py`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py)

## Classes
### `CausalMask`  ·  implements/extends _ComputableMask
- def: [`jax_perf/splash_attention_debug.py:353`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L353) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
- doc: Lazy causal mask, prevents the model from attending to future tokens.
- signature: `class CausalMask(splash_attention_mask._ComputableMask):`
- members:
  - `causal_mask_function(q_ids, kv_ids)` — [`L375`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L375)
  - `id_to_rank` — [`L373`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L373)
  - `offset` — [`L363`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L363)
- protocol/private: `__eq__`[`L406`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L406), `__hash__`[`L416`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L416), `__init__`[`L365`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L365)
- used by: [`main`](splash_attention_debug.md#main), [`_unflatten_mask`](splash_attention_debug.md#_unflatten_mask)

### `SplashAttention`  ·  implements/extends Module
- def: [`jax_perf/splash_attention_debug.py:36`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L36) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
- doc: Sparse flash attention implementation for JAX/Flax models.
- signature: `class SplashAttention(nn.Module):`
- members:
  - `_batch_partitions(self)` — [`L281`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L281) — Returns the partition spec for batch dimensions.
  - `_calculate_scale_factor(self, query: jax.Array)` — [`L255`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L255) — Calculates the scale factor for the attention mechanism based on the query tensor. — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `_qkvo_partition_spec(self)` — [`L307`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L307) — Returns the partition spec for query, key, value, and output tensors. — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `_segment_ids_spec(self)` — [`L325`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L325) — Returns the partition spec for segment IDs. — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `_sequence_partitions(self)` — [`L294`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L294) — Returns the partition spec for sequence dimensions.
  - `create_kernel_blocks(self, query: jax.Array, key: jax.Array, q_block_size: Optional[int] = None, kv_block_size: Optional[int] = None, q_block_repeats: Optional[int] = None, kv_block_repeats: Optional[int] = None, fuse_if_possible: bool = True, ensure_block_sizes: bool = True)` — [`L173`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L173) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `create_mask(self, q_seq: int, kv_seq: int, dynamic_mask: Optional[jax.Array], static_mask: Optional[np.ndarray] = None)` — [`L46`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L46) — Creates an attention mask for the splash attention mechanism. — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `dtype(self)` — [`L350`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L350) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `dtype_block_factor(self)` — [`L272`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L272) — Returns the block factor for the data type used in the attention mechanism. — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `is_tpu_attention(self)` — [`L268`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L268)
  - `sequence_block_factor(self)` — [`L338`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L338) — Returns the sequence block factor based on the mesh shape. — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `setup(self)` — [`L41`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L41)
  - `splash_attention_helper(self, query: jax.Array, key: jax.Array, value: jax.Array, q_segment_ids: Optional[jax.Array], kv_segment_ids: Optional[jax.Array], dynamic_mask: Optional[jax.Array], static_mask: Optional[np.ndarray], block_sizes: Optional[splash_attention_kernel.BlockSizes])` — [`L115`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L115) — Helper function to call the splash attention kernel. — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `static_helper(*args, **kwargs)` — [`L91`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L91) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `mesh` — [`L38`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L38) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
  - `window_size` — [`L39`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L39)
- protocol/private: `__call__`[`L75`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L75)
- used by: [`main`](splash_attention_debug.md#main)

## Functions
- `_flatten_mask(mask)` — [`L425`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L425)
- `_unflatten_mask(aux, data)` — [`L429`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L429)
- `create_timestep_mask(timestep: jax.Array)` — [`L470`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L470)
- `flash(q, k, v)` — [`L531`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L531) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
- `main()` — [`L437`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L437) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
- `static_apply(q, k, v)` — [`L503`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L503) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)
- `static_apply_causal(q, k, v)` — [`L517`](../../../../../raw/code/learning-machine/jax_perf/splash_attention_debug.py#L517) — documented in [jax_perf-splash_attention_debug](../../concepts/jax_perf-splash_attention_debug.md)

