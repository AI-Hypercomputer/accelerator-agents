---
title: 'Module: tpu_inference/kernels/fused_moe/v1/kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/fused_moe/v1/kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.fused_moe.v1.kernel`/
symbols:
  _fused_ep_moe_kernel.run_per_bt: _fused_ep_moe_kernel().run_per_bt().
  _fused_ep_moe_kernel.run_per_bt.run_per_expert: _fused_ep_moe_kernel().run_per_bt().run_per_expert().
  cdiv: cdiv.
  _fused_ep_moe_kernel: _fused_ep_moe_kernel().
  _fused_ep_moe_kernel.expert_ffn: _fused_ep_moe_kernel().expert_ffn().
  fused_ep_moe: fused_ep_moe().
  _fused_ep_moe_kernel.start_fetch_next_bw: _fused_ep_moe_kernel().start_fetch_next_bw().
  P: P.
  _fused_ep_moe_kernel.sync_barrier: _fused_ep_moe_kernel().sync_barrier().
  align_to: align_to().
  _fused_ep_moe_kernel.all_reduce_metadata._all_reduce_metadata: _fused_ep_moe_kernel().all_reduce_metadata()._all_reduce_metadata().
  _fused_ep_moe_kernel.dynamic_ffn1: _fused_ep_moe_kernel().dynamic_ffn1().
  _fused_ep_moe_kernel.dynamic_ffn2: _fused_ep_moe_kernel().dynamic_ffn2().
  _fused_ep_moe_kernel.dynamic_ffn2.body: _fused_ep_moe_kernel().dynamic_ffn2().body().
  ref_moe: ref_moe().
  broadcast_minor: broadcast_minor().
  apply_act_fn: apply_act_fn().
  _fused_ep_moe_kernel.start_a2a_scatter: _fused_ep_moe_kernel().start_a2a_scatter().
  _fused_ep_moe_kernel.start_fetch_bw1: _fused_ep_moe_kernel().start_fetch_bw1().
  _fused_ep_moe_kernel.start_fetch_bw2: _fused_ep_moe_kernel().start_fetch_bw2().
  _fused_ep_moe_kernel.start_fetch_bw3: _fused_ep_moe_kernel().start_fetch_bw3().
  _fused_ep_moe_kernel.get_mesh_device_id: _fused_ep_moe_kernel().get_mesh_device_id().
  _fused_ep_moe_kernel.get_top_k: _fused_ep_moe_kernel().get_top_k().
  _fused_ep_moe_kernel.all_reduce_metadata: _fused_ep_moe_kernel().all_reduce_metadata().
  _fused_ep_moe_kernel.start_a2a_gather: _fused_ep_moe_kernel().start_a2a_gather().
  _fused_ep_moe_kernel.dynamic_ffn1.body: _fused_ep_moe_kernel().dynamic_ffn1().body().
  _fused_ep_moe_kernel.bt_acc: _fused_ep_moe_kernel().bt_acc().
  fused_ep_moe.kernel: fused_ep_moe().kernel().
  get_dtype_packing: get_dtype_packing().
  _fused_ep_moe_kernel.wait_a2a_gather_send: _fused_ep_moe_kernel().wait_a2a_gather_send().
  _fused_ep_moe_kernel.wait_send_bo: _fused_ep_moe_kernel().wait_send_bo().
  apply_scoring_fn: apply_scoring_fn().
  _fused_ep_moe_kernel.start_fetch_b_gating: _fused_ep_moe_kernel().start_fetch_b_gating().
  swigluoai: swigluoai().
  _fused_ep_moe_kernel.wait_fetch_b_gating: _fused_ep_moe_kernel().wait_fetch_b_gating().
  _fused_ep_moe_kernel.wait_a2a_scatter_recv: _fused_ep_moe_kernel().wait_a2a_scatter_recv().
  _fused_ep_moe_kernel.wait_a2a_scatter_send: _fused_ep_moe_kernel().wait_a2a_scatter_send().
  _fused_ep_moe_kernel.wait_a2a_gather_recv_all: _fused_ep_moe_kernel().wait_a2a_gather_recv_all().
  _fused_ep_moe_kernel.wait_fetch_bw1: _fused_ep_moe_kernel().wait_fetch_bw1().
  _fused_ep_moe_kernel.wait_fetch_bw2: _fused_ep_moe_kernel().wait_fetch_bw2().
  _fused_ep_moe_kernel.wait_fetch_bw3: _fused_ep_moe_kernel().wait_fetch_bw3().
  _fused_ep_moe_kernel.start_send_bo: _fused_ep_moe_kernel().start_send_bo().
---
# Module: [`tpu_inference/kernels/fused_moe/v1/kernel.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py)

## Functions
- `_all_reduce_metadata(t2e_routing_vmem, d2e_count_vmem, offsets_vmem, starts_vmem, sizes_vmem)` — [`L398`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L398)
- `_fused_ep_moe_kernel(tokens_hbm, w1_hbm, w2_hbm, w1_scale_hbm, w2_scale_hbm, b1_hbm, b2_hbm, gating_hbm, a2a_g_hbm, output_hbm, t2e_routing_x2_smem, d2e_count_x2_smem, expert_offsets_x2_smem, expert_starts_x2_smem, expert_sizes_x2_smem, a2a_s_sends_x2_smem, a2a_s_x2_vmem, a2a_s_acc_x2_vmem, a2a_g_acc_vmem, b_gating_x2_vmem, b_output_x2_vmem, b_w1_x2_vmem, b_w3_x2_vmem, b_w2_x2_vmem, b_w1_scale_x2_vmem, b_w3_scale_x2_vmem, b_w2_scale_x2_vmem, b_b1_x2_vmem, b_b3_x2_vmem, b_b2_x2_vmem, b_acc_vmem, local_sems, send_sems, recv_sems, a2a_gather_sem, a2a_acc_sem, *, top_k: int, renormalize_topk_logits: bool, ep_axis_name: str, act_fn: str, scoring_fn: str, subc_quant_w1_sz: int | None = None, subc_quant_w2_sz: int | None = None, bt: int, bf: int, bd1: int, bd2: int, btc: int, bfc: int, bd1c: int, bd2c: int)` — [`L197`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L197)
- `align_to(x, a)` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L33)
- `all_reduce_metadata(bt_sem_id, t2e_routing, starts, sizes)` — [`L393`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L393)
- `apply_act_fn(acc1, acc3, act_fn)` — [`L77`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L77)
- `apply_scoring_fn(scoring_fn: str, x)` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L53)
- `body(btc_id, _)` — [`L826`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L826)
- `body(btc_id, _)` — [`L935`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L935)
- `broadcast_minor(src, shape)` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L42)
- `bt_acc(bt_id, top_k_logits_lst)` — [`L1080`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L1080)
- `dynamic_ffn1(t_b32_vmem, w1_vmem, w1_scale_vmem, b1_vmem, w3_vmem, w3_scale_vmem, b3_vmem, acc1_vmem, acc3_vmem, dyn_sz, should_init)` — [`L784`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L784)
- `dynamic_ffn2(acc1_vmem, acc3_vmem, w2_vmem, w2_scale_vmem, b2_vmem, res_b32_vmem, dyn_sz, should_init)` — [`L905`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L905)
- `expert_ffn(bt_id, e_sem_id, local_e_id)` — [`L1005`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L1005)
- `fused_ep_moe(mesh: jax.sharding.Mesh, tokens: jax.Array, w1: jax.Array, w2: jax.Array, gating_output: jax.Array, top_k: int, *, renormalize_topk_logits: bool = False, act_fn: str = "silu", scoring_fn: str = "softmax", subc_quant_w1_sz: int | None = None, subc_quant_w2_sz: int | None = None, w1_scale: (jax.Array | None) = None, w2_scale: (jax.Array | None) = None, b1: jax.Array | None = None, b2: jax.Array | None = None, bt: int | None = None, bf: int | None = None, bd1: int | None = None, bd2: int | None = None, btc: int | None = None, bfc: int | None = None, bd1c: int | None = None, bd2c: int | None = None, ep_axis_name: str = "model")` — [`L1243`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L1243)
- `get_dtype_packing(dtype)` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L37)
- `get_mesh_device_id(ep_rank)` — [`L317`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L317)
- `get_top_k(input, top_k, renormalize_topk_logits)` — [`L351`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L351)
- `kernel(tokens, w1, w2, w1_scale, w2_scale, b1, b2, gating_output, a2a_g_hbm_scratch)` — [`L1656`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L1656)
- `ref_moe(tokens: jax.Array, w1: jax.Array, w2: jax.Array, gating_output: jax.Array, top_k: int, *, renormalize_topk_logits: bool = False, act_fn: str = "silu", scoring_fn: str = "softmax", subc_quant_w1_sz: int | None = None, subc_quant_w2_sz: int | None = None, w1_scale: (jax.Array | None) = None, w2_scale: (jax.Array | None) = None, b1: jax.Array | None = None, b2: jax.Array | None = None)` — [`L88`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L88)
- `run_per_bt(bt_id, e_sem_id)` — [`L1133`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L1133)
- `run_per_expert(local_e_id, e_sem_id)` — [`L1151`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L1151)
- `start_a2a_gather(bt_id, e_sem_id, local_e_id)` — [`L536`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L536)
- `start_a2a_scatter(bt_id, e_sem_id, local_e_id)` — [`L479`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L479)
- `start_fetch_b_gating(bt_id, priority=0)` — [`L331`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L331)
- `start_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id)` — [`L589`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L589)
- `start_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id)` — [`L627`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L627)
- `start_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id)` — [`L665`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L665)
- `start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, bd1_id, bd2_id)` — [`L765`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L765)
- `start_send_bo(bt_id, priority=0)` — [`L1109`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L1109)
- `swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7)` — [`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L65) — Activation used in some models such as GPT-OSS.
- `sync_barrier()` — [`L321`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L321)
- `wait_a2a_gather_recv_all()` — [`L579`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L579)
- `wait_a2a_gather_send(bt_id, e_sem_id, local_e_id)` — [`L562`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L562)
- `wait_a2a_scatter_recv(bt_id, e_sem_id, local_e_id)` — [`L517`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L517)
- `wait_a2a_scatter_send(bt_id, e_sem_id, local_e_id)` — [`L527`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L527)
- `wait_fetch_b_gating(bt_id)` — [`L342`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L342)
- `wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id)` — [`L705`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L705)
- `wait_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id)` — [`L725`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L725)
- `wait_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id)` — [`L745`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L745)
- `wait_send_bo(bt_id)` — [`L1118`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L1118)

## Module values
- `P` — [`L28`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L28)
- `cdiv` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py#L30)

