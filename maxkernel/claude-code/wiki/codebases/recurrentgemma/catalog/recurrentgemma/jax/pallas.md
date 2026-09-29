---
title: 'Module: recurrentgemma/jax/pallas.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/pallas.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.pallas`/
symbols:
  compute_pallas_kernel_spec: compute_pallas_kernel_spec().
  _lru_bwd: _lru_bwd().
  multi_shard_correction: multi_shard_correction().
  lru_pallas_scan: lru_pallas_scan().
  linear_rnn_shard_corrected_pallas_call: linear_rnn_shard_corrected_pallas_call().
  pallas_lru: pallas_lru().
  reverse_block_spec: reverse_block_spec().
  pad_array_to_divisible: pad_array_to_divisible().
  make_block_specs: make_block_specs().
  make_block_shape_and_dtypes: make_block_shape_and_dtypes().
  get_acc_dtype: get_acc_dtype().
  ShardingSpec.activations_sharding_spec: ShardingSpec#activations_sharding_spec().
  _lru_fwd: _lru_fwd().
  _lru: _lru().
  PallasKernelSpec: PallasKernelSpec#
  linear_rnn_pallas_call: linear_rnn_pallas_call().
  linear_rnn_pallas_kernel: linear_rnn_pallas_kernel().
  ShardingSpec.rnn_state_sharding_spec: ShardingSpec#rnn_state_sharding_spec().
  PallasKernelSpec.grid: PallasKernelSpec#grid().
  maybe_wrap_in_complex: maybe_wrap_in_complex().
  to_blocks: to_blocks().
  from_blocks: from_blocks().
  initialize_carry: initialize_carry().
  initialize_carry.init_h0: initialize_carry().init_h0().
  ShardingSpec.activations_sharding: ShardingSpec#activations_sharding().
  ShardingSpec.rnn_state_sharding: ShardingSpec#rnn_state_sharding().
  Spec: Spec.
  LruPallasResiduals: LruPallasResiduals.
  ShardingSpec.mesh: ShardingSpec#mesh.
  linear_rnn_loop_body: linear_rnn_loop_body().
  ShardingSpec.sequence_axis_name: ShardingSpec#sequence_axis_name.
  get_num_seq_shards: get_num_seq_shards().
  PallasKernelSpec.seq_grid_size: PallasKernelSpec#seq_grid_size.
  PallasKernelSpec.singleton_tile_size: PallasKernelSpec#singleton_tile_size.
  T: T.
  ShardingSpec: ShardingSpec#
  ShardingSpec.batch_axis_name: ShardingSpec#batch_axis_name.
  ShardingSpec.activations_axis_name: ShardingSpec#activations_axis_name.
  ShardingSpec.sequence_axis_index_groups: ShardingSpec#sequence_axis_index_groups.
  PallasKernelSpec.batch_grid_size: PallasKernelSpec#batch_grid_size.
  PallasKernelSpec.dim_grid_size: PallasKernelSpec#dim_grid_size.
  PallasKernelSpec.batch_tile_size: PallasKernelSpec#batch_tile_size.
  PallasKernelSpec.seq_tile_size: PallasKernelSpec#seq_tile_size.
  PallasKernelSpec.dim_tile_size: PallasKernelSpec#dim_tile_size.
  carry_dtype: carry_dtype().
  sequence_shard_index: sequence_shard_index().
  compute_tile_size: compute_tile_size().
---
# Module: [`recurrentgemma/jax/pallas.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py)

## Classes
### `PallasKernelSpec`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/jax/pallas.py:268`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L268) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- doc: The grid and size specs for the Pallas kernel.
- signature: `class PallasKernelSpec(NamedTuple):`
- members:
  - `grid(self)` — [`L280`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L280)
  - `batch_grid_size` — [`L271`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L271) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
  - `batch_tile_size` — [`L274`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L274) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
  - `dim_grid_size` — [`L273`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L273) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
  - `dim_tile_size` — [`L276`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L276) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
  - `seq_grid_size` — [`L272`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L272) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
  - `seq_tile_size` — [`L275`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L275) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
  - `singleton_tile_size` — [`L277`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L277) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- used by: [`compute_pallas_kernel_spec`](pallas.md#compute_pallas_kernel_spec), [`_lru_bwd`](pallas.md#_lru_bwd), [`linear_rnn_shard_corrected_pallas_call`](pallas.md#linear_rnn_shard_corrected_pallas_call), [`make_block_shape_and_dtypes`](pallas.md#make_block_shape_and_dtypes), [`make_block_specs`](pallas.md#make_block_specs), [`_lru_fwd`](pallas.md#_lru_fwd), [`_lru`](pallas.md#_lru), [`linear_rnn_pallas_call`](pallas.md#linear_rnn_pallas_call)

### `ShardingSpec`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/jax/pallas.py:198`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L198)
- doc: The sharding spec for running a Pallas kernel with sharded values.
- signature: `class ShardingSpec(NamedTuple):`
- members:
  - `activations_sharding(self)` — [`L216`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L216)
  - `activations_sharding_spec(self)` — [`L208`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L208) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `rnn_state_sharding(self)` — [`L230`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L230)
  - `rnn_state_sharding_spec(self)` — [`L223`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L223)
  - `activations_axis_name` — [`L204`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L204)
  - `batch_axis_name` — [`L202`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L202)
  - `mesh` — [`L201`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L201)
  - `sequence_axis_index_groups` — [`L205`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L205)
  - `sequence_axis_name` — [`L203`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L203)
- used by: [`linear_scan`](scan.md#linear_scan), [`ShardingSpec`](scan.md#ShardingSpec)

## Functions
- `_lru(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None, reverse: bool, kernel_spec: PallasKernelSpec, seq_axis: str | None = None, seq_axis_index_groups: list[list[int]] | None = None)` — [`L746`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L746) — Runs the RNN forward pass without residuals for backprop. — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `_lru_bwd(reverse: bool, kernel_spec: PallasKernelSpec, seq_axis: str | None, seq_axis_index_groups: list[list[int]] | None, res: LruPallasResiduals, dy_and_dh_last: tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex])` — [`L792`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L792) — Runs the RNN backward pass and corrects for any sequence sharding. — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `_lru_fwd(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None, reverse: bool, kernel_spec: PallasKernelSpec, seq_axis: str | None, seq_axis_index_groups: list[list[int]] | None = None)` — [`L769`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L769) — Runs the RNN forward pass and corrects for any sequence sharding.
- `carry_dtype(dtype: jnp.dtype)` — [`L284`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L284)
- `compute_pallas_kernel_spec(x: complex_lib.RealOrComplex, max_seq_block_size: int, min_seq_block_size: int, singleton_tile_size: int = 128)` — [`L351`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L351) — Retrurns the correct Pallas grid. — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `compute_tile_size(dim_size: int, max_block_size: int, min_block_size: int)` — [`L333`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L333) — Computes the correct Pallas block size. — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `from_blocks(x: complex_lib.RealOrComplex)` — [`L326`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L326) — Reverse the effect of `to_blocks`.
- `get_acc_dtype(x: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None, acc_float_dtype: jnp.dtype = jnp.float32)` — [`L37`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L37) — Returns the accumulation dtype for the given inputs. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `get_num_seq_shards(seq_axis: str | None = None, seq_axis_index_groups: list[list[int]] | None = None)` — [`L237`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L237) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `init_h0()` — [`L464`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L464)
- `initialize_carry(h_carry_ref: complex_lib.RealOrComplex, a_prod_carry_ref: complex_lib.RealOrComplex | None, h_init_ref: complex_lib.RealOrComplex | None)` — [`L457`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L457) — Initializes the accumulator and product carries.
- `linear_rnn_loop_body(i: int, _: None, x_ref: complex_lib.RealOrComplex, a_ref: complex_lib.RealOrComplex, y_ref: complex_lib.RealOrComplex, h_carry_ref: complex_lib.RealOrComplex, a_prod_carry_ref: complex_lib.RealOrComplex | None, a_prod_ref: complex_lib.RealOrComplex | None = None, reverse: bool = False, backprop: bool = False)` — [`L477`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L477) — Evaluates a single step of a linear RNN loop.
- `linear_rnn_pallas_call(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None, reverse: bool, kernel_spec: PallasKernelSpec, compute_a_prod: Literal[True], backprop: bool)` — [`L600`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L600) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `linear_rnn_pallas_kernel(x_ref: complex_lib.RealOrComplex, a_ref: complex_lib.RealOrComplex, h_init_ref: complex_lib.RealOrComplex | None, y_ref: complex_lib.RealOrComplex, h_carry_ref: complex_lib.RealOrComplex, a_prod_ref: complex_lib.RealOrComplex | None = None, a_prod_carry_ref: complex_lib.RealOrComplex | None = None, reverse: bool = False, backprop: bool = False)` — [`L520`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L520) — A Pallas kernel for computing a linear RNN.
- `linear_rnn_shard_corrected_pallas_call(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None, reverse: bool, backprop: bool, kernel_spec: PallasKernelSpec, seq_axis: str | None = None, seq_axis_index_groups: list[list[int]] | None = None)` — [`L690`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L690) — A call to the linear RNN Pallas kernel with shard correction. — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `lru_pallas_scan(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None = None, reverse: bool = False, seq_axis: str | None = None, seq_axis_index_groups: list[list[int]] | None = None, max_seq_block_size: int = 256, min_seq_block_size: int = 16, pad_seq_to_min_block_size: bool = True, pad_last_dim_to_128: bool = False)` — [`L922`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L922) — Runs the LRU scan using a Pallas kernel. — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `make_block_shape_and_dtypes(x: complex_lib.RealOrComplex, kernel_spec: PallasKernelSpec)` — [`L429`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L429) — Returns the correct shape and dtype for each variable.
- `make_block_specs(kernel_spec: PallasKernelSpec)` — [`L395`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L395) — Returns the block specs for each variable.
- `maybe_wrap_in_complex(v: T, do_wrap: bool)` — [`L291`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L291)
- `multi_shard_correction(*, y: complex_lib.RealOrComplex, a_prod: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None, reverse: bool, h_last: complex_lib.RealOrComplex | None = None, a_prod_last: complex_lib.RealOrComplex | None = None, acc_float_dtype: jnp.dtype = jnp.float32, seq_axis: str | None = None, seq_axis_index_groups: list[list[int]] | None = None, shift_a_prod: bool = False, sync_h_last: bool = True)` — [`L73`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L73) — This codes corrects the result `y` from a single shard. — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `pad_array_to_divisible(x: complex_lib.RealOrComplex, divisor: int, axis: int, pad_on_back: bool = True, value: float = 0)` — [`L247`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L247) — Pads the variable `x` to have size along `axis` divisible by `divisor`. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `pallas_lru(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None = None, reverse: bool = False, seq_axis: str | None = None, seq_axis_index_groups: list[list[int]] | None = None, max_seq_block_size: int = 256, min_seq_block_size: int = 16, pad_seq_to_min_block_size: bool = True, pad_last_dim_to_128: bool = False)` — [`L852`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L852) — Runs the LRU scan using a Pallas kernel. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `reverse_block_spec(spec: Spec, num_seq_blocks: int)` — [`L297`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L297) — Reverses the order of accessing sequence axis tiles.
- `sequence_shard_index(seq_axis: str | None = None, seq_axis_index_groups: list[list[int]] | None = None)` — [`L55`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L55) — Returns the correct sequence shard index for this device.
- `to_blocks(x: complex_lib.RealOrComplex | None, s: int)` — [`L316`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L316) — Reshapes `x` such that it's last dim is equal to `s` by adding an extra axis.

## Module values
- `LruPallasResiduals` — [`L29`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L29) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)
- `Spec` — [`L27`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L27)
- `T` — [`L26`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L26)

