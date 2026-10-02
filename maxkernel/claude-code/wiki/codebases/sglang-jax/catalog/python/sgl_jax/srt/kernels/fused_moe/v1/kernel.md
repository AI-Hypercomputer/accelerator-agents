---
title: 'Module: python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.fused_moe.v1.kernel`/
symbols:
  fused_ep_moe: fused_ep_moe().
  _fused_ep_moe_kernel.run_bt: _fused_ep_moe_kernel().run_bt().
  FusedMoEBlockConfig.effective_for: FusedMoEBlockConfig#effective_for().
  validate_fused_moe_block_config: validate_fused_moe_block_config().
  FusedMoEBlockConfig.tree_flatten: FusedMoEBlockConfig#tree_flatten().
  FusedMoEBlockConfig.as_kwargs: FusedMoEBlockConfig#as_kwargs().
  _fused_ep_moe_kernel.run_shared_expert_slice: _fused_ep_moe_kernel().run_shared_expert_slice().
  _fused_ep_moe_kernel: _fused_ep_moe_kernel().
  P: P.
  _fused_ep_moe_kernel.expert_ffn.run_down_slices.run_down_bd2.body: _fused_ep_moe_kernel().expert_ffn().run_down_slices().run_down_bd2().body().
  FusedMoEBlockConfig.bf: FusedMoEBlockConfig#bf.
  FusedMoEBlockConfig.bse: FusedMoEBlockConfig#bse.
  FusedMoEBlockConfig.bts: FusedMoEBlockConfig#bts.
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices.run_gate_up_bd1.body: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices().run_gate_up_bd1().body().
  _fused_ep_moe_kernel.run_shared_expert_slice._.body_w1w3: _fused_ep_moe_kernel().run_shared_expert_slice()._().body_w1w3().
  _fused_ep_moe_kernel.run_bt.run_per_expert_pipelined: _fused_ep_moe_kernel().run_bt().run_per_expert_pipelined().
  FusedMoEBlockConfig.bt: FusedMoEBlockConfig#bt.
  FusedMoEBlockConfig.bd1: FusedMoEBlockConfig#bd1.
  _validate_fused_ep_moe_args: _validate_fused_ep_moe_args().
  cdiv: cdiv.
  FusedMoEBlockConfig.bd2: FusedMoEBlockConfig#bd2.
  _fused_ep_moe_kernel.sync_barrier: _fused_ep_moe_kernel().sync_barrier().
  FusedMoEBlockConfig: FusedMoEBlockConfig#
  FusedMoEBlockConfig.btc: FusedMoEBlockConfig#btc.
  _fused_ep_moe_kernel.expert_ffn.run_down_slices.run_down_bd2.body.run_ffn2_tile: _fused_ep_moe_kernel().expert_ffn().run_down_slices().run_down_bd2().body().run_ffn2_tile().
  _fused_ep_moe_kernel.run_shared_expert_slice._.body_w2: _fused_ep_moe_kernel().run_shared_expert_slice()._().body_w2().
  _fused_ep_moe_kernel.run_bt.compute_expert_batch: _fused_ep_moe_kernel().run_bt().compute_expert_batch().
  FusedMoEBlockConfig.bfc: FusedMoEBlockConfig#bfc.
  FusedMoEBlockConfig.bd1c: FusedMoEBlockConfig#bd1c.
  FusedMoEBlockConfig.bd2c: FusedMoEBlockConfig#bd2c.
  align_to: align_to().
  _fused_ep_moe_kernel.all_reduce_metadata: _fused_ep_moe_kernel().all_reduce_metadata().
  _fused_ep_moe_kernel.dynamic_ffn2.body: _fused_ep_moe_kernel().dynamic_ffn2().body().
  _fused_ep_moe_kernel.acc_and_store_output.run_acc_pipeline: _fused_ep_moe_kernel().acc_and_store_output().run_acc_pipeline().
  activation_fn: activation_fn().
  _fused_ep_moe_kernel.dynamic_ffn1.compute_tile: _fused_ep_moe_kernel().dynamic_ffn1().compute_tile().
  _fused_ep_moe_kernel.expert_ffn: _fused_ep_moe_kernel().expert_ffn().
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices.run_gate_up_bd1: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices().run_gate_up_bd1().
  _fused_ep_moe_kernel.acc_and_store_output: _fused_ep_moe_kernel().acc_and_store_output().
  _fused_ep_moe_kernel.get_mesh_device_id: _fused_ep_moe_kernel().get_mesh_device_id().
  _fused_ep_moe_kernel.all_reduce_metadata._all_reduce_metadata: _fused_ep_moe_kernel().all_reduce_metadata()._all_reduce_metadata().
  _fused_ep_moe_kernel.start_fetch_bw1: _fused_ep_moe_kernel().start_fetch_bw1().
  _fused_ep_moe_kernel.start_fetch_bw3: _fused_ep_moe_kernel().start_fetch_bw3().
  _fused_ep_moe_kernel.dynamic_ffn1: _fused_ep_moe_kernel().dynamic_ffn1().
  _fused_ep_moe_kernel.expert_ffn.wait_stage_a2a_s_acc_tile: _fused_ep_moe_kernel().expert_ffn().wait_stage_a2a_s_acc_tile().
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices.run_gate_up_bd1.body.run_ffn1_tile: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices().run_gate_up_bd1().body().run_ffn1_tile().
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices._run_active: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices()._run_active().
  _fused_ep_moe_kernel.expert_ffn.run_down_slices.run_down_bd2: _fused_ep_moe_kernel().expert_ffn().run_down_slices().run_down_bd2().
  _fused_ep_moe_kernel.expert_ffn._prefetch_next_expert_if_needed: _fused_ep_moe_kernel().expert_ffn()._prefetch_next_expert_if_needed().
  _fused_ep_moe_kernel.expert_ffn._run_inactive: _fused_ep_moe_kernel().expert_ffn()._run_inactive().
  _fused_ep_moe_kernel.expert_ffn._run_active: _fused_ep_moe_kernel().expert_ffn()._run_active().
  _fused_ep_moe_kernel.acc_and_store_output.wait_load_acc_bt: _fused_ep_moe_kernel().acc_and_store_output().wait_load_acc_bt().
  fused_ep_moe.kernel: fused_ep_moe().kernel().
  _fused_ep_moe_kernel.expert_ffn.run_down_slices.run_down_bd2.body._prefetch_next_expert: _fused_ep_moe_kernel().expert_ffn().run_down_slices().run_down_bd2().body()._prefetch_next_expert().
  _fused_ep_moe_kernel.run_bt.compute_expert_batch._first_load: _fused_ep_moe_kernel().run_bt().compute_expert_batch()._first_load().
  _fused_ep_moe_kernel.run_bt.run_per_expert_pipelined._first_load: _fused_ep_moe_kernel().run_bt().run_per_expert_pipelined()._first_load().
  _fused_ep_moe_kernel.start_a2a_scatter: _fused_ep_moe_kernel().start_a2a_scatter().
  _fused_ep_moe_kernel.wait_a2a_gather_recv_all: _fused_ep_moe_kernel().wait_a2a_gather_recv_all().
  _fused_ep_moe_kernel.start_fetch_se_tokens: _fused_ep_moe_kernel().start_fetch_se_tokens().
  _fused_ep_moe_kernel.acc_and_store_output.start_load_acc_bt: _fused_ep_moe_kernel().acc_and_store_output().start_load_acc_bt().
  get_dtype_packing: get_dtype_packing().
  _fused_ep_moe_kernel.wait_a2a_gather_send: _fused_ep_moe_kernel().wait_a2a_gather_send().
  _fused_ep_moe_kernel.start_a2a_scatter_batch: _fused_ep_moe_kernel().start_a2a_scatter_batch().
  _fused_ep_moe_kernel.wait_a2a_scatter_send_batch: _fused_ep_moe_kernel().wait_a2a_scatter_send_batch().
  _fused_ep_moe_kernel.dynamic_ffn1.body_init: _fused_ep_moe_kernel().dynamic_ffn1().body_init().
  _fused_ep_moe_kernel.dynamic_ffn1.body_acc: _fused_ep_moe_kernel().dynamic_ffn1().body_acc().
  _fused_ep_moe_kernel.dynamic_ffn2: _fused_ep_moe_kernel().dynamic_ffn2().
  _fused_ep_moe_kernel.dynamic_ffn2.body._ffn2_sg_body: _fused_ep_moe_kernel().dynamic_ffn2().body()._ffn2_sg_body().
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices().
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices._run_active.run_one_bd1_no_init: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices()._run_active().run_one_bd1_no_init().
  _fused_ep_moe_kernel.expert_ffn.run_down_slices: _fused_ep_moe_kernel().expert_ffn().run_down_slices().
  _fused_ep_moe_kernel.run_bt.cleanup_body_batch: _fused_ep_moe_kernel().run_bt().cleanup_body_batch().
  _fused_ep_moe_kernel.run_bt.cleanup_body: _fused_ep_moe_kernel().run_bt().cleanup_body().
  jax_allreduce_metadata_by_bt: jax_allreduce_metadata_by_bt().
  _fused_ep_moe_kernel.start_fetch_se_tokens_slice: _fused_ep_moe_kernel().start_fetch_se_tokens_slice().
  _fused_ep_moe_kernel.start_fetch_se_w1: _fused_ep_moe_kernel().start_fetch_se_w1().
  _fused_ep_moe_kernel.start_fetch_se_w3: _fused_ep_moe_kernel().start_fetch_se_w3().
  _fused_ep_moe_kernel.expert_ffn.start_stage_a2a_s_tile_from_hbm: _fused_ep_moe_kernel().expert_ffn().start_stage_a2a_s_tile_from_hbm().
  _fused_ep_moe_kernel.wait_store_output: _fused_ep_moe_kernel().wait_store_output().
  broadcast_minor: broadcast_minor().
  ref_moe: ref_moe().
  _fused_ep_moe_kernel.start_a2a_scatter._scatter_one._remote_copy: _fused_ep_moe_kernel().start_a2a_scatter()._scatter_one()._remote_copy().
  _fused_ep_moe_kernel.start_a2a_scatter_batch._scatter_one_batch._remote_copy: _fused_ep_moe_kernel().start_a2a_scatter_batch()._scatter_one_batch()._remote_copy().
  _fused_ep_moe_kernel.start_a2a_gather._remote_copy: _fused_ep_moe_kernel().start_a2a_gather()._remote_copy().
  _fused_ep_moe_kernel.wait_fetch_se_tokens: _fused_ep_moe_kernel().wait_fetch_se_tokens().
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices.run_gate_up_bd1.body._prefetch_tokens_for_bd0_bts0: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices().run_gate_up_bd1().body()._prefetch_tokens_for_bd0_bts0().
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices.run_gate_up_bd1.body.run_ffn1_tile._prefetch_tokens_for_next_bts: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices().run_gate_up_bd1().body().run_ffn1_tile()._prefetch_tokens_for_next_bts().
  _fused_ep_moe_kernel.expert_ffn.run_gate_up_slices.run_gate_up_bd1.body._prefetch_bts0_tokens_for_next_bd: _fused_ep_moe_kernel().expert_ffn().run_gate_up_slices().run_gate_up_bd1().body()._prefetch_bts0_tokens_for_next_bd().
  _fused_ep_moe_kernel.run_bt.run_per_expert_pipelined._._wait_before_buffer_reuse: _fused_ep_moe_kernel().run_bt().run_per_expert_pipelined()._()._wait_before_buffer_reuse().
  get_ep_size: get_ep_size().
  _fused_ep_moe_kernel.start_fetch_topk: _fused_ep_moe_kernel().start_fetch_topk().
  _fused_ep_moe_kernel.wait_a2a_scatter_recv: _fused_ep_moe_kernel().wait_a2a_scatter_recv().
  _fused_ep_moe_kernel.start_a2a_gather: _fused_ep_moe_kernel().start_a2a_gather().
  _fused_ep_moe_kernel.start_fetch_bw2: _fused_ep_moe_kernel().start_fetch_bw2().
  _fused_ep_moe_kernel.wait_fetch_se_tokens_slice: _fused_ep_moe_kernel().wait_fetch_se_tokens_slice().
  _fused_ep_moe_kernel.start_fetch_se_w2: _fused_ep_moe_kernel().start_fetch_se_w2().
  _fused_ep_moe_kernel.expert_ffn.start_load_stage_a2a_s_acc_tile_from_hbm: _fused_ep_moe_kernel().expert_ffn().start_load_stage_a2a_s_acc_tile_from_hbm().
  _fused_ep_moe_kernel.expert_ffn.start_store_stage_a2a_s_acc_tile_to_hbm: _fused_ep_moe_kernel().expert_ffn().start_store_stage_a2a_s_acc_tile_to_hbm().
  _fused_ep_moe_kernel.expert_ffn.with_static_bw: _fused_ep_moe_kernel().expert_ffn().with_static_bw().
  _fused_ep_moe_kernel.start_send_bo: _fused_ep_moe_kernel().start_send_bo().
  _A2A_HBM_FRACTION: _A2A_HBM_FRACTION.
  _device_hbm_bytes: _device_hbm_bytes().
  swigluoai: swigluoai().
  _fused_ep_moe_kernel.wait_fetch_topk: _fused_ep_moe_kernel().wait_fetch_topk().
  _fused_ep_moe_kernel.all_reduce_metadata._copy_precomputed: _fused_ep_moe_kernel().all_reduce_metadata()._copy_precomputed().
  _fused_ep_moe_kernel.all_reduce_metadata._local_metadata: _fused_ep_moe_kernel().all_reduce_metadata()._local_metadata().
  _fused_ep_moe_kernel.start_a2a_scatter._scatter_one: _fused_ep_moe_kernel().start_a2a_scatter()._scatter_one().
  _fused_ep_moe_kernel.start_a2a_scatter_batch._scatter_one_batch: _fused_ep_moe_kernel().start_a2a_scatter_batch()._scatter_one_batch().
  _fused_ep_moe_kernel.wait_a2a_scatter_send_batch._wait_one: _fused_ep_moe_kernel().wait_a2a_scatter_send_batch()._wait_one().
  _fused_ep_moe_kernel.wait_a2a_scatter_send: _fused_ep_moe_kernel().wait_a2a_scatter_send().
  _fused_ep_moe_kernel.wait_a2a_gather_recv_all._wait_one_expert: _fused_ep_moe_kernel().wait_a2a_gather_recv_all()._wait_one_expert().
  _fused_ep_moe_kernel.start_fetch_and_wait_se_scales: _fused_ep_moe_kernel().start_fetch_and_wait_se_scales().
  _fused_ep_moe_kernel.wait_fetch_bw1: _fused_ep_moe_kernel().wait_fetch_bw1().
  _fused_ep_moe_kernel.wait_fetch_bw2: _fused_ep_moe_kernel().wait_fetch_bw2().
  _fused_ep_moe_kernel.wait_fetch_bw3: _fused_ep_moe_kernel().wait_fetch_bw3().
  _fused_ep_moe_kernel.wait_fetch_se_w1: _fused_ep_moe_kernel().wait_fetch_se_w1().
  _fused_ep_moe_kernel.wait_fetch_se_w3: _fused_ep_moe_kernel().wait_fetch_se_w3().
  _fused_ep_moe_kernel.wait_fetch_se_w2: _fused_ep_moe_kernel().wait_fetch_se_w2().
  _fused_ep_moe_kernel.dynamic_ffn1.compute_tile._ffn1_sg_body: _fused_ep_moe_kernel().dynamic_ffn1().compute_tile()._ffn1_sg_body().
  _fused_ep_moe_kernel.expert_ffn.wait_stage_a2a_s_tile: _fused_ep_moe_kernel().expert_ffn().wait_stage_a2a_s_tile().
  _fused_ep_moe_kernel.acc_and_store_output.start_load_acc_bt._load_one: _fused_ep_moe_kernel().acc_and_store_output().start_load_acc_bt()._load_one().
  _fused_ep_moe_kernel.acc_and_store_output.wait_load_acc_bt._count_valid: _fused_ep_moe_kernel().acc_and_store_output().wait_load_acc_bt()._count_valid().
  _fused_ep_moe_kernel.acc_and_store_output.wait_load_acc_bt._._wait_one: _fused_ep_moe_kernel().acc_and_store_output().wait_load_acc_bt()._()._wait_one().
  _fused_ep_moe_kernel.acc_and_store_output.acc_gather_to_output: _fused_ep_moe_kernel().acc_and_store_output().acc_gather_to_output().
  compute_local_expert_sizes: compute_local_expert_sizes().
  FusedMoEBlockConfig.tree_unflatten: FusedMoEBlockConfig#tree_unflatten().
  _pad128: _pad128().
  _fused_ep_moe_kernel.start_a2a_scatter._scatter_one._local_copy: _fused_ep_moe_kernel().start_a2a_scatter()._scatter_one()._local_copy().
  _fused_ep_moe_kernel.start_a2a_scatter_batch._scatter_one_batch._local_copy: _fused_ep_moe_kernel().start_a2a_scatter_batch()._scatter_one_batch()._local_copy().
  _fused_ep_moe_kernel.start_a2a_gather._local_copy: _fused_ep_moe_kernel().start_a2a_gather()._local_copy().
---
# Module: [`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py)

## Classes
### `FusedMoEBlockConfig`
- def: [`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:34`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L34) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- signature: `class FusedMoEBlockConfig:`
- members:
  - `as_kwargs(self)` — [`L176`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L176)
  - `effective_for(self, *, num_tokens: int, ep_size: int, dtype: jnp.dtype, quant_block_k: int | None = None, intermediate_size: int | None = None)` — [`L46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L46) — Return the *effective* config after applying kernel override rules. — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
  - `tree_flatten(self)` — [`L147`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L147)
  - `tree_unflatten(cls, aux_data, children)` — [`L164`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L164)
  - `bd1` — [`L39`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L39) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
  - `bd1c` — [`L40`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L40)
  - `bd2` — [`L41`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L41) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
  - `bd2c` — [`L42`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L42)
  - `bf` — [`L37`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L37) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
  - `bfc` — [`L38`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L38)
  - `bse` — [`L43`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L43) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
  - `bt` — [`L35`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L35) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
  - `btc` — [`L36`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L36)
  - `bts` — [`L44`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L44) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- uses (calls/refs, reference-scoped): [`get_dtype_packing`](kernel.md#get_dtype_packing)
- used by: [`__call__`](../../../layers/fused_moe.md#FusedEPMoE.__call__), [`fused_ep_moe`](kernel.md#fused_ep_moe), [`v1_bc`](../v2/bench_compare.md#v1_bc), [`get_tuned_fused_moe_block_config`](tuned_block_configs.md#get_tuned_fused_moe_block_config), [`v1_bc_eff`](../v2/bench_compare.md#v1_bc_eff), [`validate_fused_moe_block_config`](kernel.md#validate_fused_moe_block_config), [`DEFAULT_FUSED_MOE_BLOCK_CONFIG`](tuned_block_configs.md#DEFAULT_FUSED_MOE_BLOCK_CONFIG), [`_validate_fused_ep_moe_args`](kernel.md#_validate_fused_ep_moe_args), [`V1BlockConfig`](../v2/bench_compare.md#V1BlockConfig)

## Functions
- `_all_reduce_metadata(t2e_routing_vmem, d2e_count_vmem, offsets_vmem, starts_vmem, sizes_vmem)` — [`L888`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L888)
- `_copy_precomputed(t2e_routing_vmem, d2e_count_vmem, offsets_vmem, starts_vmem, sizes_vmem)` — [`L754`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L754)
- `_count_valid(t_i, acc)` — [`L2511`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2511)
- `_device_hbm_bytes()` — [`L27`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L27) — Total HBM bytes on the local device (cached, queried once).
- `_ffn1_sg_body(sg_id, carry)` — [`L1707`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1707)
- `_ffn2_sg_body(sg_id, sg_acc)` — [`L1942`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1942)
- `_first_load()` — [`L2818`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2818)
- `_first_load()` — [`L2894`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2894)
- `_fused_ep_moe_kernel(tokens_hbm, w1_hbm, w2_hbm, w3_hbm, w1_scale_hbm, w2_scale_hbm, w3_scale_hbm, b1_hbm, b2_hbm, b3_hbm, topk_weights_hbm, topk_ids_hbm, a2a_s_x2_hbm, a2a_s_acc_x2_hbm, a2a_g_hbm, w1_shared_hbm, w3_shared_hbm, w2_shared_hbm, w1_shared_scale_hbm, w3_shared_scale_hbm, w2_shared_scale_hbm, metadata_starts_hbm, metadata_sizes_hbm, metadata_d2e_counts_hbm, output_hbm, t2e_routing_x2_smem, d2e_count_x2_smem, expert_offsets_x2_smem, expert_starts_x2_smem, expert_sizes_x2_smem, a2a_s_sends_x2_smem, a2a_g_acc_vmem, b_topk_weights_x2_vmem, b_topk_ids_x2_vmem, b_output_x2_vmem, b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem, b_w1_scale_x2_vmem, b_w3_scale_x2_vmem, b_w2_scale_x2_vmem, b_b1_x2_vmem, b_b3_x2_vmem, b_b2_x2_vmem, b_acc_vmem, b_stage_x2_vmem, a2a_s_acc_stage_x3_vmem, b_se_tokens_vmem, b_se_w1_x2_vmem, b_se_w3_x2_vmem, b_se_w2_x2_vmem, b_se_w1_scale_all, b_se_w3_scale_all, b_se_w2_scale_all, b_se_acc_vmem, token_stage_x2_sems, acc_stage_x3_sems, local_sems, send_x2_sems, recv_x2_sems, gather_send_x2_sems, a2a_gather_sem, a2a_acc_sems, barrier_sem, *, top_k: int, dp_axis_name: str, tp_axis_name: str, act_fn: str, disable_a2a: bool = False, disable_dynamic_ffn1: bool = False, disable_dynamic_ffn2: bool = False, disable_weight_load: bool = False, disable_a2a_s_tile_read: bool = False, disable_a2a_s_acc_tile_write: bool = False, disable_shared_expert: bool = False, disable_all_reduce_metadata: bool = False, disable_sync_barrier: bool = False, use_jax_allreduce_metadata: bool = True, quant_block_k: int | None = None, bt: int, bf: int, bd1: int, bd2: int, bts: int, btc: int, bfc: int, bd1c: int, bd2c: int, bse: int)` — [`L520`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L520) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `_load_one(t_i, _)` — [`L2478`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2478)
- `_local_copy(src_t_id=src_t_id, start=start, local_sz=local_sz, e_sem_id=e_sem_id)` — [`L1040`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1040)
- `_local_copy(src_t_id=src_t_id, start=start, local_sz=local_sz, e_sem_id_k=e_sem_id_k)` — [`L1102`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1102)
- `_local_copy(start=start, local_sz=local_sz, my_e_id=my_e_id, e_sem_id=e_sem_id)` — [`L1197`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1197)
- `_local_metadata(t2e_routing_vmem, d2e_count_vmem, offsets_vmem, starts_vmem, sizes_vmem)` — [`L830`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L830)
- `_pad128(x)` — [`L196`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L196) — Round up to nearest multiple of 128 (TPU DMA alignment).
- `_prefetch_bts0_tokens_for_next_bd()` — [`L2237`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2237)
- `_prefetch_next_expert()` — [`L2305`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2305)
- `_prefetch_next_expert_if_needed()` — [`L2439`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2439)
- `_prefetch_tokens_for_bd0_bts0()` — [`L2142`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2142)
- `_prefetch_tokens_for_next_bts(next_start=next_start, next_buf_id=next_buf_id, bd1_id=bd1_id)` — [`L2198`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2198)
- `_remote_copy(src_t_id=src_t_id, start=start, remote_sz=remote_sz, e_sem_id=e_sem_id, recv_id=recv_id)` — [`L1050`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1050)
- `_remote_copy(src_t_id=src_t_id, start=start, remote_sz=remote_sz, e_sem_id_k=e_sem_id_k, recv_id=recv_id)` — [`L1112`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1112)
- `_remote_copy(start=start, remote_sz=remote_sz, my_e_id=my_e_id, e_sem_id=e_sem_id, recv_id=recv_id)` — [`L1210`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1210)
- `_run_active(_)` — [`L2250`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2250)
- `_run_active(_)` — [`L2463`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2463)
- `_run_inactive(_)` — [`L2452`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2452)
- `_scatter_one(t_id, send_sz, e_sem_id=e_sem_id, local_e_id=local_e_id, bt_start=bt_start)` — [`L1020`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1020)
- `_scatter_one_batch(t_id, _, bt_start=bt_start)` — [`L1083`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1083)
- `_validate_fused_ep_moe_args(*, mesh: jax.sharding.Mesh, tokens: jax.Array, w1: jax.Array, w2: jax.Array, w3: jax.Array, topk_weights: jax.Array, topk_ids: jax.Array, top_k: int, quant_block_k: int | None, w1_scale: jax.Array | None, w2_scale: jax.Array | None, w3_scale: jax.Array | None, w1_shared: jax.Array | None, w2_shared: jax.Array | None, w3_shared: jax.Array | None, w1_shared_scale: jax.Array | None, w2_shared_scale: jax.Array | None, w3_shared_scale: jax.Array | None, b1: jax.Array | None, b2: jax.Array | None, b3: jax.Array | None, block_config: FusedMoEBlockConfig, dp_axis_name: str, tp_axis_name: str)` — [`L2993`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2993) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `_wait_before_buffer_reuse()` — [`L2918`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2918)
- `_wait_one(slot, _)` — [`L1136`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1136)
- `_wait_one(_, __)` — [`L2529`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2529)
- `_wait_one_expert(e_id, _)` — [`L1260`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1260)
- `acc_and_store_output(*, bt_sem_id, out_buf_id)` — [`L2472`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2472)
- `acc_gather_to_output(*, tile_start, out_offset, buf_id)` — [`L2546`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2546)
- `activation_fn(acc1, acc3, act_fn)` — [`L228`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L228)
- `align_to(x, a)` — [`L192`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L192)
- `all_reduce_metadata(*, bt_id, bt_sem_id, t2e_routing, starts, sizes)` — [`L743`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L743)
- `body(btc_id, __)` — [`L1917`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1917)
- `body(bw_sem_id: int)` — [`L2135`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2135)
- `body(bw_sem_id: int)` — [`L2283`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2283)
- `body_acc(i, _)` — [`L1875`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1875)
- `body_init(i, _)` — [`L1869`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1869)
- `body_w1w3(bd1_idx, carry)` — [`L2656`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2656)
- `body_w2(bd2_idx, _)` — [`L2709`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2709)
- `broadcast_minor(src, shape)` — [`L206`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L206)
- `cleanup_body(block_idx, _)` — [`L2955`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2955)
- `cleanup_body_batch(block_idx, _)` — [`L2854`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2854)
- `compute_expert_batch(local_e_id, curr_se_block)` — [`L2814`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2814)
- `compute_local_expert_sizes(topk_ids: jax.Array, num_experts: int)` — [`L3199`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L3199) — Count routed tokens per expert for one local token tile.
- `compute_tile(btc_id, is_init_mode)` — [`L1697`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1697)
- `dynamic_ffn1(t_vmem, w1_vmem, w1_scale_vmem, b1_vmem, w3_vmem, w3_scale_vmem, b3_vmem, acc1_vmem, acc3_vmem, dyn_sz, should_init, bf_id)` — [`L1650`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1650)
- `dynamic_ffn2(acc1_vmem, acc3_vmem, w2_vmem, w2_scale_vmem, b2_vmem, res_vmem, dyn_sz, should_init, bd2_id)` — [`L1880`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1880)
- `expert_ffn(bt_sem_id, e_sem_id, local_e_id)` — [`L2027`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2027)
- `fused_ep_moe(mesh: jax.sharding.Mesh, tokens: jax.Array, w1: jax.Array, w2: jax.Array, w3: jax.Array, topk_weights: jax.Array, topk_ids: jax.Array, top_k: int, *, use_grouped_topk: bool = False, num_groups: int = 1, top_k_groups: int = 1, renormalize_topk_logits: bool = False, routed_scaling_factor: float | None = None, act_fn: str = "silu", disable_a2a: bool = False, disable_dynamic_ffn1: bool = False, disable_dynamic_ffn2: bool = False, disable_weight_load: bool = False, disable_a2a_s_tile_read: bool = False, disable_a2a_s_acc_tile_write: bool = False, disable_shared_expert: bool = False, disable_all_reduce_metadata: bool = False, disable_sync_barrier: bool = False, use_jax_allreduce_metadata: bool = True, quant_block_k: int | None = None, w1_scale: jax.Array | None = None, w2_scale: jax.Array | None = None, w3_scale: jax.Array | None = None, w1_shared: jax.Array | None = None, w2_shared: jax.Array | None = None, w3_shared: jax.Array | None = None, w1_shared_scale: jax.Array | None = None, w2_shared_scale: jax.Array | None = None, w3_shared_scale: jax.Array | None = None, b1: jax.Array | None = None, b2: jax.Array | None = None, b3: jax.Array | None = None, block_config: FusedMoEBlockConfig | None = None, dp_axis_name: str = "data", tp_axis_name: str = "tensor")` — [`L3296`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L3296) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `get_dtype_packing(dtype)` — [`L201`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L201)
- `get_ep_size(mesh: jax.sharding.Mesh, dp_axis_name, tp_axis_name)` — [`L513`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L513)
- `get_mesh_device_id(ep_rank)` — [`L689`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L689)
- `jax_allreduce_metadata_by_bt(topk_ids: jax.Array, num_experts: int, bt: int, num_devices: int, dp_axis_name: str, tp_axis_name: str)` — [`L3212`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L3212) — Compute the per-bt metadata normally produced inside the Pallas kernel.
- `kernel(tokens, w1, w2, w3, w1_scale, w2_scale, w3_scale, b1, b2, b3, topk_weights, topk_ids, a2a_s_x2_hbm_scratch, a2a_s_acc_x2_hbm_scratch, a2a_g_hbm_scratch, w1_shared=None, w3_shared=None, w2_shared=None, w1_shared_scale=None, w3_shared_scale=None, w2_shared_scale=None)` — [`L3732`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L3732)
- `ref_moe(tokens: jax.Array, w1: jax.Array, w2: jax.Array, w3: jax.Array, gating_output: jax.Array, top_k: int, *, use_grouped_topk: bool = False, num_groups: int = 1, top_k_groups: int = 1, bias: jax.Array | None = None, renormalize_topk_logits: bool = False, routed_scaling_factor: float | None = None, act_fn: str = "silu", quant_block_k: int | None = None, w1_scale: (jax.Array | None) = None, w2_scale: (jax.Array | None) = None, w3_scale: (jax.Array | None) = None, b1: jax.Array | None = None, b2: jax.Array | None = None, b3: jax.Array | None = None, w1_shared: jax.Array | None = None, w2_shared: jax.Array | None = None, w3_shared: jax.Array | None = None, w1_shared_scale: jax.Array | None = None, w2_shared_scale: jax.Array | None = None, w3_shared_scale: jax.Array | None = None)` — [`L339`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L339)
- `run_acc_pipeline(i, _)` — [`L2572`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2572)
- `run_bt(bt_id, e_sem_id)` — [`L2763`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2763) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `run_down_bd2(bd2_id, bw_sem_id)` — [`L2282`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2282)
- `run_down_slices(*, bf_id: int, bw_sem_id)` — [`L2279`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2279)
- `run_ffn1_tile(token_tile_id, token_buf_id, num_token_tiles=num_token_tiles, token_tile=token_tile, dyn_sz_i32=dyn_sz_i32, bd1_id=bd1_id, w1_vmem=w1_vmem, w1_scale_vmem=w1_scale_vmem, b1_vmem=b1_vmem, w3_vmem=w3_vmem, w3_scale_vmem=w3_scale_vmem, b3_vmem=b3_vmem, should_init_ffn1=should_init_ffn1)` — [`L2176`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2176)
- `run_ffn2_tile(token_tile_id, state, *, bd2_start=bd2_start, token_tile=token_tile, dyn_sz_i32=dyn_sz_i32, num_token_tiles=num_token_tiles, w2_vmem=w2_vmem, w2_scale_vmem=w2_scale_vmem, b2_vmem=b2_vmem, should_init_ffn2=should_init_ffn2)` — [`L2352`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2352)
- `run_gate_up_bd1(*, bd1_id, bw_sem_id, token_buf_offset, should_init_ffn1: bool)` — [`L2134`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2134)
- `run_gate_up_slices(*, bf_id: int, bw_sem_id)` — [`L2133`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2133)
- `run_one_bd1_no_init(bd1_id, carry)` — [`L2258`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2258)
- `run_per_expert_pipelined(local_e_id, carry)` — [`L2890`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2890) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `run_shared_expert_slice(block_id, bt_id, bt_sem_id, out_buf_id)` — [`L2624`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2624) — Executes the computation for a single shared expert block with hardware prefetching. — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `start_a2a_gather(*, bt_sem_id, e_sem_id, local_e_id)` — [`L1184`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1184)
- `start_a2a_scatter(*, bt_sem_id, e_sem_id, local_e_id, bt_start)` — [`L1014`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1014)
- `start_a2a_scatter_batch(*, bt_sem_id, bt_start)` — [`L1077`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1077)
- `start_fetch_and_wait_se_scales()` — [`L1276`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1276)
- `start_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id)` — [`L1298`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1298)
- `start_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id)` — [`L1339`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1339)
- `start_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id)` — [`L1374`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1374)
- `start_fetch_se_tokens(bt_id)` — [`L1549`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1549)
- `start_fetch_se_tokens_slice(*, bt_start, bt_sem_id, bd1_idx, buf_id)` — [`L1487`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1487)
- `start_fetch_se_w1(grp_sem_id, block_id, bd1_idx)` — [`L1567`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1567)
- `start_fetch_se_w2(grp_sem_id, block_id, bd2_idx)` — [`L1622`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1622)
- `start_fetch_se_w3(grp_sem_id, block_id, bd1_idx)` — [`L1594`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1594)
- `start_fetch_topk(*, bt_id, priority=0)` — [`L709`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L709)
- `start_load_acc_bt(*, tile_start, buf_id)` — [`L2477`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2477)
- `start_load_stage_a2a_s_acc_tile_from_hbm(tile_start, bd2_start, buf_id)` — [`L2072`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2072)
- `start_send_bo(*, bt_id, priority=0)` — [`L2594`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2594)
- `start_stage_a2a_s_tile_from_hbm(tile_start, bd1_id, buf_id)` — [`L2044`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2044)
- `start_store_stage_a2a_s_acc_tile_to_hbm(tile_start, bd2_start, buf_id)` — [`L2106`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2106)
- `swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7)` — [`L218`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L218) — Activation used in some models such as GPT-OSS.
- `sync_barrier()` — [`L694`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L694)
- `validate_fused_moe_block_config(*, num_tokens: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, dtype: jnp.dtype, ep_size: int, quant_block_k: int | None, block_config: FusedMoEBlockConfig)` — [`L239`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L239) — Validate a (post-override) block config against kernel constraints. — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `wait_a2a_gather_recv_all(*, bt_sem_id)` — [`L1249`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1249)
- `wait_a2a_gather_send(*, bt_sem_id, e_sem_id, local_e_id)` — [`L1228`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1228)
- `wait_a2a_scatter_recv(*, bt_sem_id, e_sem_id, local_e_id)` — [`L1152`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1152)
- `wait_a2a_scatter_send(*, bt_sem_id, e_sem_id, local_e_id)` — [`L1168`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1168)
- `wait_a2a_scatter_send_batch()` — [`L1132`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1132)
- `wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id)` — [`L1415`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1415)
- `wait_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id)` — [`L1440`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1440)
- `wait_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id)` — [`L1462`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1462)
- `wait_fetch_se_tokens(bt_id)` — [`L1561`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1561)
- `wait_fetch_se_tokens_slice(*, bt_sem_id, buf_id)` — [`L1527`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1527)
- `wait_fetch_se_w1(grp_sem_id)` — [`L1582`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1582)
- `wait_fetch_se_w2(grp_sem_id)` — [`L1637`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1637)
- `wait_fetch_se_w3(grp_sem_id)` — [`L1609`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L1609)
- `wait_fetch_topk(*, bt_id)` — [`L729`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L729)
- `wait_load_acc_bt(*, buf_id, tile_start)` — [`L2506`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2506)
- `wait_stage_a2a_s_acc_tile(buf_id)` — [`L2091`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2091)
- `wait_stage_a2a_s_tile(buf_id)` — [`L2063`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2063)
- `wait_store_output(*, bt_id)` — [`L2604`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2604)
- `with_static_bw(bw_sem_id, body)` — [`L2125`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L2125)

## Module values
- `P` — [`L17`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L17)
- `_A2A_HBM_FRACTION` — [`L23`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L23)
- `cdiv` — [`L19`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py#L19)

