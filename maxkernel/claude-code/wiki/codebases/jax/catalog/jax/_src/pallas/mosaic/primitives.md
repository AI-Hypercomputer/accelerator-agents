---
title: 'Module: jax/_src/pallas/mosaic/primitives.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/primitives.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.primitives`/
symbols:
  dma_start_partial_discharge_rule: dma_start_partial_discharge_rule().
  dma_wait_partial_discharge_rule: dma_wait_partial_discharge_rule().
  with_memory_space_constraint: with_memory_space_constraint().
  bitcast: bitcast().
  _dma_start_pp_eqn: _dma_start_pp_eqn().
  _dma_wait_abstract_eval: _dma_wait_abstract_eval().
  _pack_elementwise_abstract_eval: _pack_elementwise_abstract_eval().
  dma_wait_p: dma_wait_p.
  _dma_wait_pp_eqn: _dma_wait_pp_eqn().
  AsyncCopyDescriptor._get_args_and_tree: AsyncCopyDescriptor#_get_args_and_tree().
  _get_ref_and_transforms: _get_ref_and_transforms().
  dma_start_partial_discharge_rule.do_discharge_dst_sem: dma_start_partial_discharge_rule().do_discharge_dst_sem().
  dma_start_partial_discharge_rule.do_discharge_src_sem: dma_start_partial_discharge_rule().do_discharge_src_sem().
  _get_dma_effects: _get_dma_effects().
  _get_ref: _get_ref().
  make_async_remote_copy: make_async_remote_copy().
  AsyncCopyDescriptor.start: AsyncCopyDescriptor#start().
  dma_start_p: dma_start_p.
  _dma_start_abstract_eval: _dma_start_abstract_eval().
  make_async_copy: make_async_copy().
  _matmul_pop_abstract_eval: _matmul_pop_abstract_eval().
  AsyncCopyDescriptor.wait: AsyncCopyDescriptor#wait().
  AsyncCopyDescriptor.wait_send: AsyncCopyDescriptor#wait_send().
  async_remote_copy: async_remote_copy().
  _dma_unflatten: _dma_unflatten().
  split_key_p: split_key_p.
  _bitcast_lowering_rule._bitcast: _bitcast_lowering_rule()._bitcast().
  AsyncCopyDescriptor.wait_recv: AsyncCopyDescriptor#wait_recv().
  touch_p: touch_p.
  touch: touch().
  _get_barrier_semaphore_abstract_eval: _get_barrier_semaphore_abstract_eval().
  prng_seed_p: prng_seed_p.
  _unpack_elementwise_abstract_eval: _unpack_elementwise_abstract_eval().
  _touch_abstract_eval: _touch_abstract_eval().
  get_barrier_semaphore_p: get_barrier_semaphore_p.
  prng_seed: prng_seed().
  trace_value_p: trace_value_p.
  matmul_push_rhs_p: matmul_push_rhs_p.
  matmul_acc_lhs_p: matmul_acc_lhs_p.
  roll: roll().
  _dma_tree_leaves: _dma_tree_leaves().
  trace_effect: trace_effect.
  matmul_pop: matmul_pop().
  join_key_p: join_key_p.
  repeat: repeat().
  _bitcast_abstract_eval: _bitcast_abstract_eval().
  _prng_random_bits_abstract_eval: _prng_random_bits_abstract_eval().
  _stochastic_round_abstract_eval: _stochastic_round_abstract_eval().
  _matmul_push_rhs_abstract_eval: _matmul_push_rhs_abstract_eval().
  _matmul_acc_lhs_abstract_eval: _matmul_acc_lhs_abstract_eval().
  async_copy: async_copy().
  store: store().
  AsyncCopyDescriptor.device_id_type: AsyncCopyDescriptor#device_id_type.
  prng_random_bits: prng_random_bits().
  _roll_lowering_rule._roll: _roll_lowering_rule()._roll().
  _dma_start_to_lojax: _dma_start_to_lojax().
  _dma_wait_to_lojax: _dma_wait_to_lojax().
  wrap_pallas_seed: wrap_pallas_seed().
  load: load().
  trace_value: trace_value().
  matmul_push_rhs: matmul_push_rhs().
  matmul_acc_lhs: matmul_acc_lhs().
  mxu_effect: mxu_effect.
  IntDeviceId: IntDeviceId.
  bitcast_p: bitcast_p.
  _prng_seed_abstract_eval: _prng_seed_abstract_eval().
  _trace_value_abstract_eval: _trace_value_abstract_eval().
  Ref: Ref.
  AsyncCopyDescriptor.device_id: AsyncCopyDescriptor#device_id.
  get_barrier_semaphore: get_barrier_semaphore().
  roll_p: roll_p.
  prng_random_bits_p: prng_random_bits_p.
  AsyncCopyDescriptor: AsyncCopyDescriptor#
  prng_effect: prng_effect.
  unwrap_pallas_seed: unwrap_pallas_seed().
  stochastic_round: stochastic_round().
  pack_elementwise: pack_elementwise().
  unpack_elementwise: unpack_elementwise().
  _touch_batch_rule: _touch_batch_rule().
  MultiDimDeviceId: MultiDimDeviceId.
  AsyncCopyDescriptor.is_remote: AsyncCopyDescriptor#is_remote().
  stochastic_round_p: stochastic_round_p.
  pack_elementwise_p: pack_elementwise_p.
  unpack_elementwise_p: unpack_elementwise_p.
  matmul_pop_p: matmul_pop_p.
  _roll_abstract_eval: _roll_abstract_eval().
  AsyncCopyDescriptor.__post_init__: AsyncCopyDescriptor#__post_init__().
  _split_key_scalar_abstract_eval: _split_key_scalar_abstract_eval().
  dma_start_partial_discharge_rule.do_discharge_dst: dma_start_partial_discharge_rule().do_discharge_dst().
  _get_elementwise_packing_factor: _get_elementwise_packing_factor().
  AsyncCopyDescriptor.src_sem: AsyncCopyDescriptor#src_sem.
  AsyncCopyDescriptor._used: AsyncCopyDescriptor#_used.
  _bitcast_lowering_rule: _bitcast_lowering_rule().
  _bitcast_batch_rule: _bitcast_batch_rule().
  _roll_lowering_rule: _roll_lowering_rule().
  _dma_flatten: _dma_flatten().
  PRNGEffect: PRNGEffect#
  TraceEffect: TraceEffect#
  MXUEffect: MXUEffect#
  Slice: Slice.
  AsyncCopyDescriptor.__del__: AsyncCopyDescriptor#__del__().
  _join_key_scalar_abstract_eval: _join_key_scalar_abstract_eval().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  AsyncCopyDescriptor.src_ref: AsyncCopyDescriptor#src_ref.
  AsyncCopyDescriptor.dst_ref: AsyncCopyDescriptor#dst_ref.
  AsyncCopyDescriptor.dst_sem: AsyncCopyDescriptor#dst_sem.
  _dma_is_high: _dma_is_high().
  MXUEffect.__str__: MXUEffect#__str__.
---
# Module: [`jax/_src/pallas/mosaic/primitives.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py)

## Classes
### `AsyncCopyDescriptor`
- def: [`jax/_src/pallas/mosaic/primitives.py:168`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L168)
- signature: `class AsyncCopyDescriptor:`
- members:
  - `is_remote(self)` — [`L193`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L193)
  - `start(self, priority: int = 0, *, add: bool = False)` — [`L212`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L212)
  - `wait(self)` — [`L223`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L223)
  - `wait_recv(self)` — [`L228`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L228)
  - `wait_send(self)` — [`L236`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L236)
  - `device_id` — [`L173`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L173)
  - `device_id_type` — [`L174`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L174)
  - `dst_ref` — [`L170`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L170)
  - `dst_sem` — [`L171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L171)
  - `src_ref` — [`L169`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L169)
  - `src_sem` — [`L172`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L172)
- protocol/private: `__del__`[`L184`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L184), `__post_init__`[`L179`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L179), `_get_args_and_tree`[`L196`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L196), `_used`[`L175`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L175)
- uses (calls/refs, reference-scoped): [`bind`](../../core.md#Primitive.bind), [`DeviceIdType`](../primitives.md#DeviceIdType), [`dma_wait_p`](primitives.md#dma_wait_p), [`dma_start_p`](primitives.md#dma_start_p), [`MESH`](../primitives.md#DeviceIdType.MESH), [`IntDeviceId`](primitives.md#IntDeviceId), [`MultiDimDeviceId`](primitives.md#MultiDimDeviceId), [`_dma_flatten`](primitives.md#_dma_flatten)
- used by: [`ag_kernel`](../../../experimental/pallas/ops/tpu/all_gather.md#ag_kernel), [`copy_out`](pipeline.md#BufferedRef.copy_out), [`wait_out`](pipeline.md#BufferedRef.wait_out), [`copy_in`](pipeline.md#BufferedRef.copy_in), [`wait_in`](pipeline.md#BufferedRef.wait_in), [`wait_and_get_loaded`](../../../experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.md#MultiPageAsyncCopyDescriptor.wait_and_get_loaded), [`make_async_remote_copy`](primitives.md#make_async_remote_copy), [`make_async_copy`](primitives.md#make_async_copy), [`async_remote_copy`](primitives.md#async_remote_copy), [`_copy_start_or_wait`](helpers.md#sync_copy._._copy_start_or_wait), [`async_copy`](primitives.md#async_copy), [`start`](../../../experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.md#MultiPageAsyncCopyDescriptor.start)

### `MXUEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/mosaic/primitives.py:1169`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1169)
- signature: `class MXUEffect(effects.Effect):`
- protocol/private: `__str__`[`L1170`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1170)
- used by: [`mxu_effect`](primitives.md#mxu_effect)

### `PRNGEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/mosaic/primitives.py:811`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L811)
- signature: `class PRNGEffect(effects.Effect):`
- used by: [`prng_effect`](primitives.md#prng_effect)

### `TraceEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/mosaic/primitives.py:1148`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1148)
- signature: `class TraceEffect(effects.Effect):`
- used by: [`trace_effect`](primitives.md#trace_effect)

## Functions
- `_bitcast(x)` — [`L87`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L87)
- `_bitcast_abstract_eval(x, *, ty)` — [`L78`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L78)
- `_bitcast_batch_rule(batched_args, batch_axes, *, ty)` — [`L108`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L108)
- `_bitcast_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, ty)` — [`L86`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L86)
- `_dma_flatten(*args)` — [`L252`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L252)
- `_dma_is_high(*avals, **params)` — [`L300`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L300)
- `_dma_start_abstract_eval(*args, tree, device_id_type, priority, add)` — [`L331`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L331)
- `_dma_start_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L365`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L365)
- `_dma_start_to_lojax(*args, tree, device_id_type, priority, add)` — [`L305`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L305)
- `_dma_tree_leaves(tree)` — [`L261`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L261)
- `_dma_unflatten(tree, flat_args)` — [`L257`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L257)
- `_dma_wait_abstract_eval(*args, tree, device_id_type, insert_dummy_device: bool)` — [`L584`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L584)
- `_dma_wait_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L610`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L610)
- `_dma_wait_to_lojax(*args, tree, device_id_type, insert_dummy_device: bool)` — [`L558`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L558)
- `_get_barrier_semaphore_abstract_eval()` — [`L776`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L776)
- `_get_dma_effects(src_ref_aval, dst_ref_aval, dst_sem_aval, src_sem_aval, device_id_aval, device_id_type)` — [`L265`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L265)
- `_get_elementwise_packing_factor(unpacked_dtype, packed_dtype)` — [`L908`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L908)
- `_get_ref(ref)` — [`L681`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L681)
- `_get_ref_and_transforms(ref)` — [`L676`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L676)
- `_join_key_scalar_abstract_eval(*seeds, impl)` — [`L869`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L869)
- `_matmul_acc_lhs_abstract_eval(lhs: jax.Array, **_)` — [`L1260`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1260)
- `_matmul_pop_abstract_eval(*, shape, dtype, **_)` — [`L1297`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1297)
- `_matmul_push_rhs_abstract_eval(ref: jax.Array, **_)` — [`L1216`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1216)
- `_pack_elementwise_abstract_eval(*xs, packed_dtype)` — [`L947`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L947)
- `_prng_random_bits_abstract_eval(*, shape)` — [`L837`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L837)
- `_prng_seed_abstract_eval(*_)` — [`L819`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L819)
- `_roll(x, shift)` — [`L152`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L152)
- `_roll_abstract_eval(x, shift, **_)` — [`L144`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L144)
- `_roll_lowering_rule(ctx: mlir.LoweringRuleContext, x, shift, *, axis, stride, stride_axis)` — [`L149`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L149)
- `_split_key_scalar_abstract_eval(seed)` — [`L853`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L853)
- `_stochastic_round_abstract_eval(x, random_bits, *, target_dtype)` — [`L894`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L894)
- `_touch_abstract_eval(ref: jax.Array)` — [`L1113`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1113)
- `_touch_batch_rule(args, dims)` — [`L1117`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1117)
- `_trace_value_abstract_eval(value, *, label)` — [`L1158`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1158)
- `_unpack_elementwise_abstract_eval(x, *, index, packed_dtype, unpacked_dtype)` — [`L1019`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1019)
- `async_copy(src_ref, dst_ref, sem, *, priority: int = 0, add: bool = False)` — [`L706`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L706) — Issues a DMA copying from src_ref to dst_ref.
- `async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id, device_id_type: primitives.DeviceIdType = primitives.DeviceIdType.MESH)` — [`L758`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L758) — Issues a remote DMA copying from src_ref to dst_ref.
- `bitcast(x: jax.Array, ty: DTypeLike)` — [`L63`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L63)
- `dma_start_partial_discharge_rule(should_discharge, in_avals, out_avals, *args, tree, device_id_type, priority, add)` — [`L389`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L389)
- `dma_wait_partial_discharge_rule(should_discharge, in_avals, out_avals, *args, tree, device_id_type, insert_dummy_device: bool)` — [`L628`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L628)
- `do_discharge_dst(dst_ref=dst_ref)` — [`L495`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L495)
- `do_discharge_dst_sem(dst_sem=dst_sem)` — [`L503`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L503)
- `do_discharge_src_sem(src_sem=src_sem)` — [`L514`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L514)
- `get_barrier_semaphore()` — [`L782`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L782) — Returns a barrier semaphore.
- `load(ref: Ref, *, mask: jax.Array | None = None)` — [`L1069`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1069) — Loads an array from the given ref.
- `make_async_copy(src_ref, dst_ref, sem)` — [`L685`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L685) — Creates a description of an asynchronous copy operation.
- `make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id: MultiDimDeviceId | IntDeviceId | None, device_id_type: primitives.DeviceIdType = primitives.DeviceIdType.MESH)` — [`L715`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L715) — Creates a description of a remote copy operation.
- `matmul_acc_lhs(acc_addr: int, lhs: jax.Array, mxu_index: int, load_staged_rhs: int | None = None)` — [`L1225`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1225) — Performs a matrix multiplication in the chosen MXU.
- `matmul_pop(acc_addr: int, shape: tuple[int, int], dtype: jax.typing.DTypeLike, mxu_index: int)` — [`L1268`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1268) — Returns the result of a matrix multiplication from the chosen MXU and zeroes the accumulator.
- `matmul_push_rhs(rhs: jax.Array, staging_register: int, mxu_index: int, *, transpose: bool = False)` — [`L1180`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1180) — Prepares the RHS for a matrix multiplication in the chosen MXU.
- `pack_elementwise(xs, *, packed_dtype)` — [`L921`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L921) — Packs multiple arrays elementwise into a single array of a narrower dtype.
- `prng_random_bits(shape)` — [`L841`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L841)
- `prng_seed(*seeds: int | jax.Array)` — [`L823`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L823) — Sets the seed for PRNG.
- `repeat(x: jax.Array, repeats: int, axis: int)` — [`L54`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L54)
- `roll(x: jax.Array, shift: jax.Array | int, axis: int, *, stride: int | None = None, stride_axis: int | None = None)` — [`L117`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L117)
- `stochastic_round(x, random_bits, *, target_dtype)` — [`L889`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L889)
- `store(ref: Ref, val: jax.Array, *, mask: jax.Array | None = None)` — [`L1085`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1085) — Stores a value to the given ref.
- `touch(ref: jax.Array | state.TransformedRef)` — [`L1103`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1103) — Adds a fake read-write dependency to the given ref.
- `trace_value(label: str, value: jax.Array)` — [`L1130`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1130) — Emit a scalar value to the current xprof trace scope.
- `unpack_elementwise(x, *, index, packed_dtype, unpacked_dtype)` — [`L976`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L976) — Unpacks an elementwise packed array.
- `unwrap_pallas_seed(seed)` — [`L860`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L860) — Splits a PRNG key into it's scalar components.
- `with_memory_space_constraint(x: jax.Array, memory_space: Any)` — [`L1034`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1034) — Constrains the memory space of an array.
- `wrap_pallas_seed(*seeds, impl)` — [`L880`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L880) — Joins scalar into a single PRNG key.

## Module values
- `IntDeviceId` — [`L49`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L49)
- `MultiDimDeviceId` — [`L50`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L50)
- `Ref` — [`L51`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L51)
- `Slice` — [`L44`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L44)
- `bitcast_p` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L60)
- `dma_start_p` — [`L297`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L297)
- `dma_wait_p` — [`L553`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L553)
- `get_barrier_semaphore_p` — [`L773`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L773)
- `join_key_p` — [`L865`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L865)
- `matmul_acc_lhs_p` — [`L1221`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1221)
- `matmul_pop_p` — [`L1265`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1265)
- `matmul_push_rhs_p` — [`L1176`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1176)
- `mxu_effect` — [`L1171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1171)
- `pack_elementwise_p` — [`L918`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L918)
- `prng_effect` — [`L813`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L813)
- `prng_random_bits_p` — [`L833`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L833)
- `prng_seed_p` — [`L807`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L807)
- `roll_p` — [`L114`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L114)
- `split_key_p` — [`L848`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L848)
- `stochastic_round_p` — [`L886`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L886)
- `touch_p` — [`L1099`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1099)
- `trace_effect` — [`L1152`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1152)
- `trace_value_p` — [`L1126`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L1126)
- `unpack_elementwise_p` — [`L973`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L973)
- `unsafe_map` — [`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L46)
- `unsafe_zip` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/primitives.py#L47)

