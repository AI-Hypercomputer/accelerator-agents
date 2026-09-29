---
title: 'Module: python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.fused_moe.v2.kernel`/
symbols:
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body().
  _fused_ep_moe_kernel.run_bt: _fused_ep_moe_kernel().run_bt().
  fused_ep_moe_v2: fused_ep_moe_v2().
  _fused_ep_moe_kernel: _fused_ep_moe_kernel().
  FusedMoEBlockConfig.bf: FusedMoEBlockConfig#bf.
  _fused_ep_moe_kernel._run_bt_post_gather: _fused_ep_moe_kernel()._run_bt_post_gather().
  P: P.
  FusedMoEBlockConfig.bt: FusedMoEBlockConfig#bt.
  FusedMoEBlockConfig.btc: FusedMoEBlockConfig#btc.
  FusedMoEBlockConfig.bse: FusedMoEBlockConfig#bse.
  FusedMoEBlockConfig.bts: FusedMoEBlockConfig#bts.
  FusedMoEBlockConfig.effective_for: FusedMoEBlockConfig#effective_for().
  validate_fused_moe_block_config: validate_fused_moe_block_config().
  FusedMoEBlockConfig: FusedMoEBlockConfig#
  _fused_ep_moe_kernel.start_prefetch_expert_bf0: _fused_ep_moe_kernel().start_prefetch_expert_bf0().
  FusedMoEBlockConfig.tree_flatten: FusedMoEBlockConfig#tree_flatten().
  _fused_ep_moe_kernel.start_a2a_gather._remote_copy: _fused_ep_moe_kernel().start_a2a_gather()._remote_copy().
  _fused_ep_moe_kernel.start_fetch_w13_w2: _fused_ep_moe_kernel().start_fetch_w13_w2().
  _fused_ep_moe_kernel.start_a2a_scatter_batch_range._scatter_one_batch._remote_copy: _fused_ep_moe_kernel().start_a2a_scatter_batch_range()._scatter_one_batch()._remote_copy().
  ref_moe: ref_moe().
  _fused_ep_moe_kernel.bt_bank_id: _fused_ep_moe_kernel().bt_bank_id().
  _fused_ep_moe_kernel.all_reduce_metadata._inkernel_allreduce: _fused_ep_moe_kernel().all_reduce_metadata()._inkernel_allreduce().
  _fused_ep_moe_kernel.wait_a2a_scatter_send_batch._wait_one: _fused_ep_moe_kernel().wait_a2a_scatter_send_batch()._wait_one().
  _fused_ep_moe_kernel.start_prefetch_expert_gr: _fused_ep_moe_kernel().start_prefetch_expert_gr().
  _fused_ep_moe_kernel.acc_and_store_output.run_acc_pipeline: _fused_ep_moe_kernel().acc_and_store_output().run_acc_pipeline().
  _fused_ep_moe_kernel.run_shared_expert_slice: _fused_ep_moe_kernel().run_shared_expert_slice().
  _fused_ep_moe_kernel.run_bt.compute_expert_batch_compact: _fused_ep_moe_kernel().run_bt().compute_expert_batch_compact().
  _fused_ep_moe_kernel.start_a2a_scatter_batch: _fused_ep_moe_kernel().start_a2a_scatter_batch().
  _fused_ep_moe_kernel.run_bt.prepare_bt_metadata: _fused_ep_moe_kernel().run_bt().prepare_bt_metadata().
  align_to: align_to().
  activation_fn: activation_fn().
  _fused_ep_moe_kernel.sync_barrier: _fused_ep_moe_kernel().sync_barrier().
  _fused_ep_moe_kernel.start_a2a_gather._local_copy: _fused_ep_moe_kernel().start_a2a_gather()._local_copy().
  get_dtype_packing: get_dtype_packing().
  _fused_ep_moe_kernel.wait_a2a_gather_send: _fused_ep_moe_kernel().wait_a2a_gather_send().
  _fused_ep_moe_kernel.acc_and_store_output: _fused_ep_moe_kernel().acc_and_store_output().
  _fused_ep_moe_kernel.wait_store_output: _fused_ep_moe_kernel().wait_store_output().
  _fused_ep_moe_kernel.start_a2a_scatter_batch_range._scatter_one_batch: _fused_ep_moe_kernel().start_a2a_scatter_batch_range()._scatter_one_batch().
  _fused_ep_moe_kernel.wait_a2a_scatter_recv: _fused_ep_moe_kernel().wait_a2a_scatter_recv().
  _fused_ep_moe_kernel.wait_a2a_gather_recv_all._wait_one_gather_recv: _fused_ep_moe_kernel().wait_a2a_gather_recv_all()._wait_one_gather_recv().
  _fused_ep_moe_kernel.expert_ffn: _fused_ep_moe_kernel().expert_ffn().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body.act_down_btc: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body().act_down_btc().
  _fused_ep_moe_kernel.acc_and_store_output.wait_load_acc_bt: _fused_ep_moe_kernel().acc_and_store_output().wait_load_acc_bt().
  _fused_ep_moe_kernel.a2a_s_ref: _fused_ep_moe_kernel().a2a_s_ref().
  _fused_ep_moe_kernel.start_a2a_scatter_batch_range._scatter_one_batch._local_copy: _fused_ep_moe_kernel().start_a2a_scatter_batch_range()._scatter_one_batch()._local_copy().
  _fused_ep_moe_kernel.start_fetch_topk: _fused_ep_moe_kernel().start_fetch_topk().
  _fused_ep_moe_kernel.wait_a2a_scatter_send_batch: _fused_ep_moe_kernel().wait_a2a_scatter_send_batch().
  _fused_ep_moe_kernel.wait_a2a_gather_recv_all: _fused_ep_moe_kernel().wait_a2a_gather_recv_all().
  _fused_ep_moe_kernel.acc_and_store_output.start_load_acc_bt: _fused_ep_moe_kernel().acc_and_store_output().start_load_acc_bt().
  _fused_ep_moe_kernel.start_send_bo: _fused_ep_moe_kernel().start_send_bo().
  _fused_ep_moe_kernel.get_mesh_device_id: _fused_ep_moe_kernel().get_mesh_device_id().
  _fused_ep_moe_kernel.a2a_s_acc_ref: _fused_ep_moe_kernel().a2a_s_acc_ref().
  _fused_ep_moe_kernel.a2a_g_ref: _fused_ep_moe_kernel().a2a_g_ref().
  _fused_ep_moe_kernel.start_fetch_w2: _fused_ep_moe_kernel().start_fetch_w2().
  _fused_ep_moe_kernel.wait_fetch_topk: _fused_ep_moe_kernel().wait_fetch_topk().
  _fused_ep_moe_kernel.all_reduce_metadata: _fused_ep_moe_kernel().all_reduce_metadata().
  _fused_ep_moe_kernel.all_reduce_metadata._inkernel_allreduce._md_push: _fused_ep_moe_kernel().all_reduce_metadata()._inkernel_allreduce()._md_push().
  _fused_ep_moe_kernel.init_a2a_scatter_batch: _fused_ep_moe_kernel().init_a2a_scatter_batch().
  _fused_ep_moe_kernel.start_a2a_scatter_batch_range: _fused_ep_moe_kernel().start_a2a_scatter_batch_range().
  _fused_ep_moe_kernel.wait_a2a_gather_recv_all._wait_one_expert: _fused_ep_moe_kernel().wait_a2a_gather_recv_all()._wait_one_expert().
  _fused_ep_moe_kernel.dequant_w1: _fused_ep_moe_kernel().dequant_w1().
  _fused_ep_moe_kernel.dequant_w3: _fused_ep_moe_kernel().dequant_w3().
  _fused_ep_moe_kernel.dequant_w2: _fused_ep_moe_kernel().dequant_w2().
  _fused_ep_moe_kernel.expert_ffn._run_inactive: _fused_ep_moe_kernel().expert_ffn()._run_inactive().
  _fused_ep_moe_kernel.expert_ffn._run_active: _fused_ep_moe_kernel().expert_ffn()._run_active().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body._gate_only_btc: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body()._gate_only_btc().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body._up_only_btc: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body()._up_only_btc().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body.gate_up_btc_direct: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body().gate_up_btc_direct().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body.act_down_btc._ffn2_sg_body: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body().act_down_btc()._ffn2_sg_body().
  _fused_ep_moe_kernel.acc_and_store_output.start_load_acc_bt._load_one: _fused_ep_moe_kernel().acc_and_store_output().start_load_acc_bt()._load_one().
  _fused_ep_moe_kernel.run_bt._se_body: _fused_ep_moe_kernel().run_bt()._se_body().
  _fused_ep_moe_kernel._run_bt_expert_only: _fused_ep_moe_kernel()._run_bt_expert_only().
  fused_ep_moe_v2.kernel: fused_ep_moe_v2().kernel().
  cdiv: cdiv.
  ref_moe._dequant: ref_moe()._dequant().
  ref_moe._deq_se: ref_moe()._deq_se().
  _fused_ep_moe_kernel.a2a_bank_for_bt: _fused_ep_moe_kernel().a2a_bank_for_bt().
  _fused_ep_moe_kernel.scatter_recv_sem: _fused_ep_moe_kernel().scatter_recv_sem().
  _fused_ep_moe_kernel.a2a_gather_sem_ref: _fused_ep_moe_kernel().a2a_gather_sem_ref().
  _fused_ep_moe_kernel.start_fetch_w1: _fused_ep_moe_kernel().start_fetch_w1().
  _fused_ep_moe_kernel.wait_fetch_w1: _fused_ep_moe_kernel().wait_fetch_w1().
  _fused_ep_moe_kernel.start_fetch_w3: _fused_ep_moe_kernel().start_fetch_w3().
  _fused_ep_moe_kernel.wait_fetch_w3: _fused_ep_moe_kernel().wait_fetch_w3().
  _env_bool: _env_bool().
  _fused_ep_moe_kernel.scatter_send_sem: _fused_ep_moe_kernel().scatter_send_sem().
  _fused_ep_moe_kernel.scatter_sends_get: _fused_ep_moe_kernel().scatter_sends_get().
  _fused_ep_moe_kernel.scatter_sends_set: _fused_ep_moe_kernel().scatter_sends_set().
  _fused_ep_moe_kernel.gather_send_sem_ref: _fused_ep_moe_kernel().gather_send_sem_ref().
  _fused_ep_moe_kernel.start_fetch_se_weights: _fused_ep_moe_kernel().start_fetch_se_weights().
  _fused_ep_moe_kernel.prequant_bt: _fused_ep_moe_kernel().prequant_bt().
  swigluoai: swigluoai().
  _align_local_tokens_for_decode: _align_local_tokens_for_decode().
  get_ep_size: get_ep_size().
  _fused_ep_moe_kernel.all_reduce_metadata._inkernel_allreduce._md_drain: _fused_ep_moe_kernel().all_reduce_metadata()._inkernel_allreduce()._md_drain().
  _fused_ep_moe_kernel.start_a2a_gather: _fused_ep_moe_kernel().start_a2a_gather().
  _fused_ep_moe_kernel.wait_fetch_w2: _fused_ep_moe_kernel().wait_fetch_w2().
  _fused_ep_moe_kernel.get_prefetch_expert_bf0_target: _fused_ep_moe_kernel().get_prefetch_expert_bf0_target().
  _fused_ep_moe_kernel.dequant_w1._dq_w1: _fused_ep_moe_kernel().dequant_w1()._dq_w1().
  _fused_ep_moe_kernel.dequant_w3._dq_w3: _fused_ep_moe_kernel().dequant_w3()._dq_w3().
  _fused_ep_moe_kernel.dequant_w2._dq_w2: _fused_ep_moe_kernel().dequant_w2()._dq_w2().
  _fused_ep_moe_kernel.expert_ffn._run_active.wait_expert_store_slot: _fused_ep_moe_kernel().expert_ffn()._run_active().wait_expert_store_slot().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body._gate_only_btc._ffn1_gate_sg: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body()._gate_only_btc()._ffn1_gate_sg().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body._up_only_btc._ffn1_up_sg: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body()._up_only_btc()._ffn1_up_sg().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body.gate_up_btc_direct._ffn1_sg_body: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body().gate_up_btc_direct()._ffn1_sg_body().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body.gate_up_btc: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body().gate_up_btc().
  _fused_ep_moe_kernel.expert_ffn._run_active.bts_body.writeback_btc: _fused_ep_moe_kernel().expert_ffn()._run_active().bts_body().writeback_btc().
  _fused_ep_moe_kernel.acc_and_store_output.wait_load_acc_bt._count_valid: _fused_ep_moe_kernel().acc_and_store_output().wait_load_acc_bt()._count_valid().
  _fused_ep_moe_kernel.acc_and_store_output.wait_load_acc_bt._._wait_one: _fused_ep_moe_kernel().acc_and_store_output().wait_load_acc_bt()._()._wait_one().
  _fused_ep_moe_kernel.acc_and_store_output.acc_gather_to_output: _fused_ep_moe_kernel().acc_and_store_output().acc_gather_to_output().
  _fused_ep_moe_kernel.setup_se_token: _fused_ep_moe_kernel().setup_se_token().
  _fused_ep_moe_kernel.wait_fetch_se_weights: _fused_ep_moe_kernel().wait_fetch_se_weights().
  _fused_ep_moe_kernel.run_bt._build_active: _fused_ep_moe_kernel().run_bt()._build_active().
  _fused_ep_moe_kernel._prequant_tokens: _fused_ep_moe_kernel()._prequant_tokens().
  FusedMoEBlockConfig.tree_unflatten: FusedMoEBlockConfig#tree_unflatten().
---
# Module: [`python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py)

## Classes
### `FusedMoEBlockConfig`
- def: [`python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py:36`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L36) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- signature: `class FusedMoEBlockConfig:`
- members:
  - `effective_for(self, *, num_tokens: int, ep_size: int)` — [`L43`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L43) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
  - `tree_flatten(self)` — [`L65`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L65)
  - `tree_unflatten(cls, aux_data, children)` — [`L70`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L70)
  - `bf` — [`L38`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L38) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
  - `bse` — [`L40`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L40) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
  - `bt` — [`L37`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L37) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
  - `btc` — [`L39`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L39) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
  - `bts` — [`L41`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L41) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- used by: [`e`](bench_v2.md#e), [`fused_ep_moe_v2`](kernel.md#fused_ep_moe_v2), [`v2_bc_eff`](bench_compare.md#v2_bc_eff), [`resolved_key`](bench_v2.md#resolved_key), [`generate_tune_candidates`](bench_v2.md#generate_tune_candidates), [`tag`](bench_v2.md#tag), [`v2_bc`](bench_compare.md#v2_bc), [`get_tuned_fused_moe_v2_block_config`](tuned_block_configs.md#get_tuned_fused_moe_v2_block_config), [`_default_cfgs`](bench_v2.md#_default_cfgs), [`bc0`](bench_v2.md#bc0), [`tag_resolved`](bench_v2.md#tag_resolved), [`bc_resolved`](bench_v2.md#bc_resolved), [`validate_fused_moe_block_config`](kernel.md#validate_fused_moe_block_config), [`DEFAULT_V2_BLOCK_CONFIG`](tuned_block_configs.md#DEFAULT_V2_BLOCK_CONFIG), [`bts`](bench_v2.md#bts)

## Functions
- `_align_local_tokens_for_decode(local_num_tokens: int)` — [`L126`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L126)
- `_build_active(le, n)` — [`L2076`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2076)
- `_count_valid(t_i, acc)` — [`L1642`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1642)
- `_deq_se(w, sc)` — [`L201`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L201)
- `_dequant(w, scale, qbk)` — [`L171`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L171)
- `_dq_w1(sg_id, _)` — [`L1062`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1062)
- `_dq_w2(sg_id, _)` — [`L1100`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1100)
- `_dq_w3(sg_id, _)` — [`L1081`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1081)
- `_env_bool(name: str, default: bool = False)` — [`L119`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L119)
- `_ffn1_gate_sg(sg_id, gate_acc, _pid=p_id)` — [`L1247`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1247)
- `_ffn1_sg_body(sg_id, carry)` — [`L1322`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1322)
- `_ffn1_up_sg(sg_id, up_acc, _pid=p_id)` — [`L1283`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1283)
- `_ffn2_sg_body(sg_id, partial_acc)` — [`L1473`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1473)
- `_fused_ep_moe_kernel(tokens_hbm, w1_hbm, w2_hbm, w3_hbm, w1_scale_hbm, w2_scale_hbm, w3_scale_hbm, topk_weights_hbm, topk_ids_hbm, a2a_s_x2_hbm, a2a_s_acc_x2_hbm, a2a_g_hbm, tokens_fp8_hbm, w1_shared_hbm, w3_shared_hbm, w2_shared_hbm, w1_shared_scale_hbm, w3_shared_scale_hbm, w2_shared_scale_hbm, output_hbm, t2e_routing_x2_smem, d2e_count_x2_smem, expert_offsets_x2_smem, expert_starts_x2_smem, expert_sizes_x2_smem, active_ids_x2_smem, n_active_x2_smem, a2a_s_sends_x2_smem, a2a_g_acc_vmem, b_topk_weights_x2_vmem, b_topk_ids_x2_vmem, b_output_x2_vmem, b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem, b_w1_scale_x2_vmem, b_w3_scale_x2_vmem, b_w2_scale_x2_vmem, b_w1_dq_vmem, b_w3_dq_vmem, b_w2_dq_vmem, b_gate_acc_vmem, b_up_acc_vmem, b_x_vmem, b_y_acc_vmem, b_y_stage_vmem, b_x_scale_vmem, x_stage_sem, y_store_sem, local_sems, send_x2_sems, recv_x2_sems, gather_send_x2_sems, a2a_gather_sem, a2a_acc_sems, md_send_sem, md_recv_sem, barrier_sem, *, top_k: int, dp_axis_name: str, tp_axis_name: str, act_fn: str, swiglu_limit: float | None = None, shared_swiglu_limit: float | None = None, enable_bt_scatter_overlap: bool = True, cross_expert_prefetch_mode: str = "full", interleave_bt: bool = True, direct_scaled_dot: bool = False, bt: int, bf: int, btc: int, bts: int, bse: int, quant_block_k: int | None = None, enable_act_quant: bool = False)` — [`L260`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L260) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- `_gate_only_btc(btc_id, ___)` — [`L1242`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1242)
- `_inkernel_allreduce(t2e_routing_vmem, d2e_count_vmem, offsets_vmem, starts_vmem, sizes_vmem)` — [`L552`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L552)
- `_load_one(t_i, _)` — [`L1612`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1612)
- `_local_copy(src_t_id=src_t_id, start=start, local_sz=local_sz, e_sem_id_k=e_sem_id_k)` — [`L702`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L702)
- `_local_copy(start=start, local_sz=local_sz, my_e_id=my_e_id, e_sem_id=e_sem_id)` — [`L797`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L797)
- `_md_drain(step, _)` — [`L614`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L614)
- `_md_push(step, _)` — [`L600`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L600)
- `_prequant_tokens(pq_bf16_buf, pq_fp8_buf, pq_load_sem, pq_store_sem)` — [`L2171`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2171)
- `_remote_copy(src_t_id=src_t_id, start=start, remote_sz=remote_sz, e_sem_id_k=e_sem_id_k, recv_id=recv_id)` — [`L715`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L715)
- `_remote_copy(start=start, remote_sz=remote_sz, my_e_id=my_e_id, e_sem_id=e_sem_id, recv_id=recv_id)` — [`L812`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L812)
- `_run_active(_)` — [`L1137`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1137)
- `_run_bt_expert_only(bt_id, e_sem_id)` — [`L2225`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2225)
- `_run_bt_post_gather(bt_id, _)` — [`L2230`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2230) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- `_run_inactive(_)` — [`L1130`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1130)
- `_scatter_one_batch(t_id, _, bt_start=bt_start)` — [`L680`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L680)
- `_se_body(block_id, _)` — [`L2096`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2096)
- `_up_only_btc(btc_id, ___)` — [`L1278`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1278)
- `_wait_one(slot, _)` — [`L751`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L751)
- `_wait_one(_, __)` — [`L1650`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1650)
- `_wait_one_expert(e_id, _)` — [`L864`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L864)
- `_wait_one_gather_recv(e_id)` — [`L852`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L852)
- `a2a_bank_for_bt(bt_id)` — [`L448`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L448)
- `a2a_g_ref(gather_bank_id, e_id, start, size)` — [`L489`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L489)
- `a2a_gather_sem_ref(gather_bank_id)` — [`L499`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L499)
- `a2a_s_acc_ref(a2a_bank_id, e_sem_id, start, size)` — [`L463`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L463)
- `a2a_s_ref(a2a_bank_id, e_sem_id, start, size)` — [`L458`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L458)
- `acc_and_store_output(*, bt_sem_id, out_buf_id, gather_bank_id)` — [`L1604`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1604)
- `acc_gather_to_output(*, tile_start, out_offset, buf_id)` — [`L1657`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1657)
- `act_down_btc(btc_id, ___)` — [`L1462`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1462)
- `activation_fn(acc1, acc3, act_fn, swiglu_limit=None)` — [`L100`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L100)
- `align_to(x, a)` — [`L82`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L82)
- `all_reduce_metadata(*, bt_id, bt_sem_id, t2e_routing)` — [`L547`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L547)
- `bt_bank_id(bt_id)` — [`L453`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L453) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- `bts_body(bts_id, next_bf0_prefetched)` — [`L1156`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1156) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- `compute_expert_batch_compact(i, carry)` — [`L2104`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2104) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- `dequant_w1(slot)` — [`L1057`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1057)
- `dequant_w2(slot)` — [`L1095`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1095)
- `dequant_w3(slot)` — [`L1076`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1076)
- `expert_ffn(bt_sem_id, e_sem_id, local_e_id, bf0_prefetched, a2a_bank_id, next_local_e_id=None, has_next=None, expert_slot=None)` — [`L1116`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1116)
- `fused_ep_moe_v2(mesh: jax.sharding.Mesh, tokens: jax.Array, w1: jax.Array, w2: jax.Array, w3: jax.Array, topk_weights: jax.Array, topk_ids: jax.Array, top_k: int, *, act_fn: str = "silu", swiglu_limit: float | None = None, shared_swiglu_limit: float | None = None, enable_bt_scatter_overlap: bool = True, w1_shared: jax.Array | None = None, w2_shared: jax.Array | None = None, w3_shared: jax.Array | None = None, w1_shared_scale: jax.Array | None = None, w2_shared_scale: jax.Array | None = None, w3_shared_scale: jax.Array | None = None, quant_block_k: int | None = None, w1_scale: jax.Array | None = None, w2_scale: jax.Array | None = None, w3_scale: jax.Array | None = None, block_config: FusedMoEBlockConfig | None = None, direct_scaled_dot: bool = False, cross_expert_prefetch_mode: str = "full", interleave_bt: bool = True, enable_act_quant: bool = False, dp_axis_name: str = "data", tp_axis_name: str = "tensor")` — [`L2296`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2296) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-bench_compare.md)
- `gate_up_btc(btc_id, ___)` — [`L1403`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1403)
- `gate_up_btc_direct(btc_id, ___)` — [`L1316`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1316)
- `gather_send_sem_ref(gather_bank_id, e_sem_id)` — [`L494`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L494)
- `get_dtype_packing(dtype)` — [`L86`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L86)
- `get_ep_size(mesh: jax.sharding.Mesh, dp_axis_name, tp_axis_name)` — [`L249`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L249)
- `get_mesh_device_id(ep_rank)` — [`L445`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L445)
- `get_prefetch_expert_bf0_target(bt_sem_id, local_e_id, *, enabled=True)` — [`L986`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L986)
- `init_a2a_scatter_batch(*, a2a_bank_id)` — [`L673`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L673)
- `kernel(tokens, w1, w2, w3, w1_scale_arg, w2_scale_arg, w3_scale_arg, topk_weights, topk_ids, a2a_s_hbm_scratch, a2a_s_acc_hbm_scratch, a2a_g_hbm_scratch, tokens_fp8_hbm_scratch=None, w1_shared=None, w3_shared=None, w2_shared=None, w1_shared_scale=None, w3_shared_scale=None, w2_shared_scale=None)` — [`L2720`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2720)
- `prepare_bt_metadata(_bt_id, _bt_sem_id)` — [`L2021`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2021)
- `prequant_bt(bt_id)` — [`L1957`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1957)
- `ref_moe(tokens, w1, w2, w3, topk_weights, topk_ids, top_k, *, act_fn="silu", swiglu_limit=None, shared_swiglu_limit=None, w1_shared=None, w2_shared=None, w3_shared=None, w1_shared_scale=None, w2_shared_scale=None, w3_shared_scale=None, quant_block_k=None, w1_scale=None, w2_scale=None, w3_scale=None)` — [`L141`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L141)
- `run_acc_pipeline(i, _)` — [`L1684`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1684)
- `run_bt(bt_id, e_sem_id, *, skip_post_gather=False)` — [`L2003`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L2003) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- `run_shared_expert_slice(block_id, bt_id, bt_sem_id, out_buf_id)` — [`L1849`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1849)
- `scatter_recv_sem(a2a_bank_id, e_sem_id)` — [`L473`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L473)
- `scatter_send_sem(a2a_bank_id, e_sem_id)` — [`L468`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L468)
- `scatter_sends_get(a2a_bank_id, e_sem_id)` — [`L478`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L478)
- `scatter_sends_set(a2a_bank_id, e_sem_id, value)` — [`L483`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L483)
- `setup_se_token(bt_id)` — [`L1737`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1737)
- `start_a2a_gather(*, bt_sem_id, e_sem_id, local_e_id, a2a_bank_id, gather_bank_id)` — [`L785`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L785)
- `start_a2a_scatter_batch(*, bt_sem_id, bt_start, a2a_bank_id)` — [`L736`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L736)
- `start_a2a_scatter_batch_range(*, bt_sem_id, bt_start, a2a_bank_id, token_start, token_end)` — [`L678`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L678)
- `start_fetch_se_weights(block_id, slot)` — [`L1775`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1775)
- `start_fetch_topk(*, bt_id, priority=0)` — [`L516`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L516)
- `start_fetch_w1(local_e_id, slot, bf_id, priority=1)` — [`L873`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L873)
- `start_fetch_w13_w2(local_e_id, slot, bf_id, *, include_w2=True)` — [`L1023`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1023)
- `start_fetch_w2(local_e_id, slot, bf_id, priority=0)` — [`L949`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L949)
- `start_fetch_w3(local_e_id, slot, bf_id, priority=1)` — [`L911`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L911)
- `start_load_acc_bt(*, tile_start, buf_id)` — [`L1610`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1610)
- `start_prefetch_expert_bf0(bt_sem_id, local_e_id, *, slot=0, priority=1, enabled=True, include_w2=False)` — [`L999`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L999)
- `start_prefetch_expert_gr(bt_sem_id, local_e_id, *, slot, enabled=True, priority=1)` — [`L1034`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1034)
- `start_send_bo(*, bt_id, priority=0)` — [`L1703`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1703)
- `swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7)` — [`L91`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L91)
- `sync_barrier()` — [`L505`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L505)
- `validate_fused_moe_block_config(*, num_tokens: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, dtype: jnp.dtype, ep_size: int, block_config: FusedMoEBlockConfig)` — [`L219`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L219) — documented in [python-sgl_jax-srt-kernels-fused_moe-v2-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v2-kernel.md)
- `wait_a2a_gather_recv_all(*, bt_sem_id, gather_bank_id)` — [`L850`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L850)
- `wait_a2a_gather_send(*, bt_sem_id, e_sem_id, local_e_id, a2a_bank_id, gather_bank_id)` — [`L830`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L830)
- `wait_a2a_scatter_recv(*, bt_sem_id, e_sem_id, local_e_id, a2a_bank_id)` — [`L768`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L768)
- `wait_a2a_scatter_send_batch(*, a2a_bank_id)` — [`L749`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L749)
- `wait_expert_store_slot(stage_slot)` — [`L1144`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1144)
- `wait_fetch_se_weights(slot)` — [`L1827`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1827)
- `wait_fetch_topk(*, bt_id)` — [`L531`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L531)
- `wait_fetch_w1(slot)` — [`L897`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L897)
- `wait_fetch_w2(slot)` — [`L973`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L973)
- `wait_fetch_w3(slot)` — [`L935`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L935)
- `wait_load_acc_bt(*, buf_id, tile_start)` — [`L1640`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1640)
- `wait_store_output(*, bt_id)` — [`L1713`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1713)
- `writeback_btc(btc_id, ___)` — [`L1567`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L1567)

## Module values
- `P` — [`L25`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L25)
- `cdiv` — [`L26`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py#L26)

