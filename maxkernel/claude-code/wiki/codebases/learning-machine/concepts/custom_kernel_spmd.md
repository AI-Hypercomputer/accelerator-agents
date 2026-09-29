---
title: custom_kernel_spmd — torch_xla SPMD flash-attention benchmark driver
type: concept
provenance: mixed
concept: custom_kernel_spmd
updated: 2026-07-03
status: fresh
---
# custom_kernel_spmd — torch_xla SPMD flash-attention benchmark driver

<!-- connect:up:begin -->
> **Cross-repo concept:** part of [flash-attention](../../../concepts/flash-attention.md) across this wiki's repos.
<!-- connect:up:end -->
A `__main__` benchmark script that exercises [`SPMDFlashAttention`](../catalog/custom_kernel_spmd.md#SPMDFlashAttention) — a PyTorch `autograd.Function` wrapper that dispatches to a JAX-defined Pallas TPU flash-attention kernel — across a 4-axis (`data, fsdp, model, sequence`) `torch_xla` SPMD mesh, timing a full forward+backward pass on synthetic Q/K/V/mask tensors.

## Entry points
- [`SPMDFlashAttention`](../catalog/custom_kernel_spmd.md#SPMDFlashAttention) — the `FlashAttention` subclass whose `.apply(...)` call is the actual op under benchmark; [`o`](../catalog/custom_kernel_spmd.md#o) is bound to its result (`o = SPMDFlashAttention.apply(q, k, v, False, q_segment_indexes, mask, sm_scale, None, attn_spec, mesh)`).
- [`parser`](../catalog/custom_kernel_spmd.md#parser) / [`args`](../catalog/custom_kernel_spmd.md#args) — the `argparse` CLI surface (`--is-3d`, `--replicated`, `--sequence-axis`, `--model-axis`, `--ddp-axis`, `--forward-only`, `--no-mask`, `--no-rand`) that parameterizes every run of the script; every mesh/shape/mode decision downstream reads from [`args`](../catalog/custom_kernel_spmd.md#args).

## Mechanism (step-by-step)
1. The script builds a 4-D SPMD mesh from CLI axis sizes: [`num_devices`](../catalog/custom_kernel_spmd.md#num_devices) (`xr.global_runtime_device_count()`) is divided among [`model_axis`](../catalog/custom_kernel_spmd.md#model_axis), [`ddp_axis`](../catalog/custom_kernel_spmd.md#ddp_axis), [`sequence_axis`](../catalog/custom_kernel_spmd.md#sequence_axis), with the remainder assigned to [`fsdp_axis`](../catalog/custom_kernel_spmd.md#fsdp_axis) (`num_devices // sequence_axis // ddp_axis // model_axis`) — the mesh is always named `("data","fsdp","model","sequence")`, and `attn_spec` maps q/k/v's logical axes onto it (or to `(None,None,None,None,None)` for a fully-replicated baseline when `--replicated` is passed).
2. Batch sizing derives [`batch_size`](../catalog/custom_kernel_spmd.md#batch_size) from [`per_axis_batch_size`](../catalog/custom_kernel_spmd.md#per_axis_batch_size) `= 16` times [`ddp_axis`](../catalog/custom_kernel_spmd.md#ddp_axis) times [`fsdp_axis`](../catalog/custom_kernel_spmd.md#fsdp_axis), then [`minibatch`](../catalog/custom_kernel_spmd.md#minibatch) further splits it (`ddp_axis*2, batch_size//ddp_axis//2`) — the fixed problem size is [`q_seq`](../catalog/custom_kernel_spmd.md#q_seq) `= 1024*8`, [`k_seq`](../catalog/custom_kernel_spmd.md#k_seq) `= 512*8`, [`depth`](../catalog/custom_kernel_spmd.md#depth) `= 256`, [`num_heads`](../catalog/custom_kernel_spmd.md#num_heads) `= 8`.
3. Synthetic inputs [`q`](../catalog/custom_kernel_spmd.md#q), [`k`](../catalog/custom_kernel_spmd.md#k), [`v`](../catalog/custom_kernel_spmd.md#v), and [`mask`](../catalog/custom_kernel_spmd.md#mask) are created via [`new_tensor`](../catalog/custom_kernel_spmd.md#new_tensor) (`torch.ones` if `--no-rand` else `torch.rand`, selected by [`no_rand`](../catalog/custom_kernel_spmd.md#no_rand)); the `--is-3d` flag reshapes them from `(minibatch, batch, heads, seq, depth)` down to `(minibatch*batch, heads, seq, depth)` to exercise the 3D-attention code path instead of 4D.
4. Data is moved to [`device`](../catalog/custom_kernel_spmd.md#device) (`xm.xla_device()`) via `xm.send_cpu_data_to_device` using a `ShardingSpec` built from [`mask_partition_spec`](../catalog/custom_kernel_spmd.md#mask_partition_spec) / the mesh's `attn_spec` — [`sharding_spec`](../catalog/custom_kernel_spmd.md#sharding_spec) governs how q/k/v are sharded across the 4-axis mesh, and the mask/segment-id tensors get their own (shorter) partition spec since they lack a head dimension.
5. [`q_segment_indexes`](../catalog/custom_kernel_spmd.md#q_segment_indexes) — an all-ones tensor shaped `(minibatch, batch, q_seq)` — is marked-sharded the same way as the mask and passed as the flash-attention kernel's segment-id argument whenever a mask is supplied, otherwise `None` is passed instead.
6. [`sm_scale`](../catalog/custom_kernel_spmd.md#sm_scale) is fixed to `q.shape[-1] ** -0.5` (standard `1/sqrt(d)` attention scaling); if [`forward_only`](../catalog/custom_kernel_spmd.md#forward_only) is false, `q.requires_grad`/`k.requires_grad`/`v.requires_grad` are set and `retain_grad()` is called so gradients survive past the autograd graph for later inspection.
7. The timed region calls [`o`](../catalog/custom_kernel_spmd.md#o) `= SPMDFlashAttention.apply(...)`, and — unless `forward_only` — computes [`loss`](../catalog/custom_kernel_spmd.md#loss) `= o.sum()` and calls `.backward()`, bracketed by `xm.mark_step(wait=True)`/`xm.wait_device_ops()` calls so the timer captures actual device execution rather than async dispatch latency.
8. After the timed region, [`fa_q_grad`](../catalog/custom_kernel_spmd.md#fa_q_grad) and [`fa_o`](../catalog/custom_kernel_spmd.md#fa_o) copy the gradient and output back to CPU (`.cpu()`) — this loop repeats 5 times, so early iterations also absorb one-time XLA compilation cost.

## Key data structures
- [`SPMDFlashAttention`](../catalog/custom_kernel_spmd.md#SPMDFlashAttention) — subclasses `FlashAttention` from `torch_xla.experimental.custom_kernel`; its docstring (quoted in source) describes it as "a simplified wrapper on top of" JAX's `jax/experimental/pallas/ops/tpu/flash_attention.py`, i.e. the PyTorch-facing surface over a JAX-native Pallas kernel.
- [`args`](../catalog/custom_kernel_spmd.md#args) (`argparse.Namespace`) — the single source of truth for every mesh-axis size, 3D/4D mode, mask presence, and forward-only mode in the run.

## Dynamics (design intent)
> [!inferred] Reading the surrounding source (not itself part of this packet's citable subgraph): `SPMDFlashAttention.forward`/`.backward` route through a `_manual_mode` helper that calls `xs.enable_manual_sharding`/`xs.disable_manual_sharding` around the actual JAX kernel invocation — i.e. the tensors are temporarily taken *out* of `torch_xla`'s automatic SPMD partitioning, handed to a JAX-traced `_flash_attention_impl`/`_flash_attention_bwd_dq`/`_flash_attention_bwd_dkv` (imported from `jax.experimental.pallas.ops.tpu.flash_attention`) via `trace_pallas` + `torch_xla._XLAC._xla_tpu_custom_call`, and then re-wrapped into the mesh's sharding on the way out. This is the bridge mechanism that lets a JAX Pallas kernel run inside a PyTorch/XLA SPMD-sharded training step.

## Edge cases
- When `--no-mask` is passed, [`mask`](../catalog/custom_kernel_spmd.md#mask) is forced to `None` after being generated — the random mask tensor is still allocated and then discarded, rather than being skipped.
- The `--is-3d` path reshapes [`q`](../catalog/custom_kernel_spmd.md#q)/[`k`](../catalog/custom_kernel_spmd.md#k)/[`v`](../catalog/custom_kernel_spmd.md#v)/mask and switches [`mask_partition_spec`](../catalog/custom_kernel_spmd.md#mask_partition_spec) to collapse the `(data, fsdp)` axes into one tuple-valued spec component — a different sharding shape than the 4D default, not just a reshape.

## Open questions
> [!inferred] Whether this script measures a meaningful throughput number or is purely a smoke test is not resolvable from the source alone — there is no assertion on correctness (no reference-attention comparison), only a wall-clock print, consistent with this being a manual/exploratory benchmark rather than a CI-gated one.

## See also
- [basic_ops](basic_ops.md) — a sibling microbenchmark harness in the same repo, comparing `shard_map` vs GSPMD-auto sharding on plain JAX (no custom kernel involved).
