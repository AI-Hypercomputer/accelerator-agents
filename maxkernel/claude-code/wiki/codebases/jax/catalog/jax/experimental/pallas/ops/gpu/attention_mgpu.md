---
title: 'Module: jax/experimental/pallas/ops/gpu/attention_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/attention_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.attention_mgpu`/
symbols:
  _attention_forward.kernel._compute_wg: _attention_forward().kernel()._compute_wg().
  _attention_forward.kernel._compute_wg.kv_loop: _attention_forward().kernel()._compute_wg().kv_loop().
  _attention_bwd: _attention_bwd().
  attention_with_pipeline_emitter: attention_with_pipeline_emitter().
  _attention_bwd.kernel_dq._compute_thread: _attention_bwd().kernel_dq()._compute_thread().
  _attention_bwd.kernel_dkv._compute_thread: _attention_bwd().kernel_dkv()._compute_thread().
  _attention_forward: _attention_forward().
  _attention_bwd.kernel_dkv.q_pipeline: _attention_bwd().kernel_dkv().q_pipeline().
  attention_with_pipeline_emitter.fa3_kernel._compute_thread: attention_with_pipeline_emitter().fa3_kernel()._compute_thread().
  attention_reference: attention_reference().
  main: main().
  _attention_bwd.kernel_dq: _attention_bwd().kernel_dq().
  attention_with_pipeline_emitter.fa3_kernel.kv_pipeline: attention_with_pipeline_emitter().fa3_kernel().kv_pipeline().
  _attention_forward.entry: _attention_forward().entry().
  attention_with_pipeline_emitter.fa3_kernel: attention_with_pipeline_emitter().fa3_kernel().
  _attention_bwd.kernel_dq.kv_pipeline: _attention_bwd().kernel_dq().kv_pipeline().
  _attention_bwd.kernel_dkv: _attention_bwd().kernel_dkv().
  TuningConfig.__post_init__: TuningConfig#__post_init__().
  _attention_forward.kernel._memory_wg: _attention_forward().kernel()._memory_wg().
  _attention_forward.kernel._memory_wg._kv_loop: _attention_forward().kernel()._memory_wg()._kv_loop().
  _attention_forward.kernel: _attention_forward().kernel().
  attention: attention().
  attention_with_pipeline_emitter.fa3_kernel.perform_schedule_barrier: attention_with_pipeline_emitter().fa3_kernel().perform_schedule_barrier().
  _attention_forward.kernel._compute_wg.epilogue_kv_loop: _attention_forward().kernel()._compute_wg().epilogue_kv_loop().
  _attention_forward.kernel.perform_schedule_barrier: _attention_forward().kernel().perform_schedule_barrier().
  _attention_forward.kernel._compute_wg.kv_loop.compute_qk: _attention_forward().kernel()._compute_wg().kv_loop().compute_qk().
  _attention_forward.kernel._compute_wg.kv_loop.compute_pv: _attention_forward().kernel()._compute_wg().kv_loop().compute_pv().
  _attention_bwd.kernel_dkv.q_pipeline._compute: _attention_bwd().kernel_dkv().q_pipeline()._compute().
  attention_with_pipeline_emitter.fa3_kernel.kv_pipeline.compute_qk: attention_with_pipeline_emitter().fa3_kernel().kv_pipeline().compute_qk().
  TuningConfig.block_q_dq: TuningConfig#block_q_dq.
  TuningConfig.block_kv: TuningConfig#block_kv.
  TuningConfig.max_concurrent_steps: TuningConfig#max_concurrent_steps.
  TuningConfig.causal: TuningConfig#causal.
  _attention_forward.kernel._compute_wg.kv_loop.end_softmax_barriers: _attention_forward().kernel()._compute_wg().kv_loop().end_softmax_barriers().
  TuningConfig: TuningConfig#
  TuningConfig.block_q_dkv: TuningConfig#block_q_dkv.
  TuningConfig.block_kv_dkv: TuningConfig#block_kv_dkv.
  _attention_fwd: _attention_fwd().
  _attention_bwd.kernel_dq.kv_pipeline.compute_s: _attention_bwd().kernel_dq().kv_pipeline().compute_s().
  _attention_bwd.kernel_dq.kv_pipeline.compute_dp: _attention_bwd().kernel_dq().kv_pipeline().compute_dp().
  _attention_bwd.kernel_dkv.q_pipeline._compute_sT: _attention_bwd().kernel_dkv().q_pipeline()._compute_sT().
  TuningConfig.block_q: TuningConfig#block_q.
  TuningConfig.block_kv_dq: TuningConfig#block_kv_dq.
  _attention_forward.kernel._compute_wg.kv_loop.compute_pv._wait: _attention_forward().kernel()._compute_wg().kv_loop().compute_pv()._wait().
  TuningConfig.has_backward_blocks: TuningConfig#has_backward_blocks().
  _attention_bwd.kernel_dq.kv_pipeline.compute_dq: _attention_bwd().kernel_dq().kv_pipeline().compute_dq().
  _attention_bwd.kernel_dkv.q_pipeline.compute_dk: _attention_bwd().kernel_dkv().q_pipeline().compute_dk().
  attention_with_pipeline_emitter.fa3_kernel.kv_pipeline.compute_pv: attention_with_pipeline_emitter().fa3_kernel().kv_pipeline().compute_pv().
  PipelineCallback.__call__: PipelineCallback#__call__().
  T: T.
  PipelineCallback: PipelineCallback#
  TuningConfig.use_schedule_barrier: TuningConfig#use_schedule_barrier.
  TuningConfig.compute_wgs_bwd: TuningConfig#compute_wgs_bwd.
---
# Module: [`jax/experimental/pallas/ops/gpu/attention_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py)

## Classes
### `PipelineCallback`  ·  implements/extends Protocol
- def: [`jax/experimental/pallas/ops/gpu/attention_mgpu.py:35`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L35)
- doc: A callback that returns the same type as the input.
- signature: `class PipelineCallback(Protocol):`
- protocol/private: `__call__`[`L37`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L37)
- uses (calls/refs, reference-scoped): [`T`](attention_mgpu.md#T)
- used by: [`_compute_thread`](attention_mgpu.md#_attention_bwd.kernel_dq._compute_thread), [`_compute_thread`](attention_mgpu.md#attention_with_pipeline_emitter.fa3_kernel._compute_thread)

### `TuningConfig`
- def: [`jax/experimental/pallas/ops/gpu/attention_mgpu.py:41`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L41)
- signature: `class TuningConfig:`
- members:
  - `has_backward_blocks(self)` — [`L71`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L71)
  - `block_kv` — [`L43`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L43)
  - `block_kv_dkv` — [`L50`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L50)
  - `block_kv_dq` — [`L52`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L52)
  - `block_q` — [`L42`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L42)
  - `block_q_dkv` — [`L49`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L49)
  - `block_q_dq` — [`L51`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L51)
  - `causal` — [`L46`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L46)
  - `compute_wgs_bwd` — [`L47`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L47)
  - `max_concurrent_steps` — [`L44`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L44)
  - `use_schedule_barrier` — [`L45`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L45)
- protocol/private: `__post_init__`[`L54`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L54)
- used by: [`_compute_wg`](attention_mgpu.md#_attention_forward.kernel._compute_wg), [`_attention_bwd`](attention_mgpu.md#_attention_bwd), [`attention_with_pipeline_emitter`](attention_mgpu.md#attention_with_pipeline_emitter), [`_attention_forward`](attention_mgpu.md#_attention_forward), [`main`](attention_mgpu.md#main), [`kernel_dq`](attention_mgpu.md#_attention_bwd.kernel_dq), [`kernel_dkv`](attention_mgpu.md#_attention_bwd.kernel_dkv), [`kernel`](attention_mgpu.md#_attention_forward.kernel), [`attention`](attention_mgpu.md#attention), [`perform_schedule_barrier`](attention_mgpu.md#attention_with_pipeline_emitter.fa3_kernel.perform_schedule_barrier), [`_attention_fwd`](attention_mgpu.md#_attention_fwd)

## Functions
- `_attention_bwd(config: TuningConfig, save_residuals: bool, res, do)` — [`L350`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L350)
- `_attention_forward(q, k, v, config: TuningConfig, save_residuals: bool = False)` — [`L74`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L74)
- `_attention_fwd(q, k, v, config: TuningConfig, save_residuals: bool)` — [`L344`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L344)
- `_compute(refs)` — [`L546`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L546)
- `_compute_sT(acc_ref)` — [`L536`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L536)
- `_compute_thread(pipeline_callback: PipelineCallback,)` — [`L403`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L403)
- `_compute_thread(pipeline_callback)` — [`L496`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L496)
- `_compute_thread(pipeline_callback: PipelineCallback,)` — [`L695`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L695)
- `_compute_wg()` — [`L128`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L128)
- `_kv_loop(kv_step)` — [`L270`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L270)
- `_memory_wg()` — [`L261`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L261)
- `_wait()` — [`L214`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L214)
- `attention(q, k, v, config: TuningConfig, save_residuals: bool = False)` — [`L341`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L341)
- `attention_reference(q, k, v, causal=False, save_residuals=False)` — [`L815`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L815)
- `attention_with_pipeline_emitter(q, k, v, config: TuningConfig, save_residuals=False)` — [`L651`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L651)
- `compute_dk(acc_ref)` — [`L564`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L564)
- `compute_dp(acc_ref)` — [`L446`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L446)
- `compute_dq(acc_ref)` — [`L457`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L457)
- `compute_pv(acc_ref)` — [`L208`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L208)
- `compute_pv(acc_ref)` — [`L754`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L754)
- `compute_qk(acc_ref)` — [`L169`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L169)
- `compute_qk(acc_ref)` — [`L734`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L734)
- `compute_s(acc_ref)` — [`L437`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L437)
- `end_softmax_barriers()` — [`L193`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L193)
- `entry(q_ref, k_ref, v_ref, out_ref, lse_ref)` — [`L279`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L279)
- `epilogue_kv_loop(kv_step, _)` — [`L223`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L223)
- `fa3_kernel(q_ref, k_ref, v_ref, out_ref, lse_ref, smem_buffers, q_barriers, schedule_barrier)` — [`L682`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L682)
- `kernel(q_ref, k_ref, v_ref, out_ref, lse_ref, scoped)` — [`L109`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L109)
- `kernel_dkv(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dk_ref, dv_ref, smem_buffers, buffer_barriers, block_q: int, block_kv: int)` — [`L488`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L488)
- `kernel_dq(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dq_ref, smem_buffers, buffer_barriers, block_q, block_kv)` — [`L395`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L395)
- `kv_loop(kv_step, carry, causal: bool = False)` — [`L164`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L164)
- `kv_pipeline(_, k_smem, v_smem, k_consumed_barrier, v_consumed_barrier, carry)` — [`L433`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L433)
- `kv_pipeline(_, k_smem, v_smem, k_consumed_barrier, v_consumed_barrier, carry)` — [`L729`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L729)
- `main(unused_argv)` — [`L844`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L844)
- `perform_schedule_barrier()` — [`L117`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L117)
- `perform_schedule_barrier()` — [`L690`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L690)
- `q_pipeline(_, q_smem, do_smem, lse_smem, delta_smem, q_consumed_barrier, do_consumed_barrier, lse_consumed_barrier, delta_consumed_barrier, carry)` — [`L532`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L532)

## Module values
- `T` — [`L33`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/attention_mgpu.py#L33)

