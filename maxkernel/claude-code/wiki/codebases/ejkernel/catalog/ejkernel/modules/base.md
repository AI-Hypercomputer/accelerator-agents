---
title: 'Module: ejkernel/modules/base.py'
type: catalog
provenance: extracted
module: ejkernel/modules/base.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.base`/
symbols:
  detect_platform: detect_platform().
  create_default_executor: create_default_executor().
  mesh_to_jax_mesh: mesh_to_jax_mesh().
  KernelConfig.backend: KernelConfig#backend.
  KernelConfig.__post_init__: KernelConfig#__post_init__().
  KernelConfig.platform: KernelConfig#platform.
  KernelConfig: KernelConfig#
  KernelConfig.block_q: KernelConfig#block_q.
  KernelConfig.block_k: KernelConfig#block_k.
  KernelConfig.block_d: KernelConfig#block_d.
  KernelConfig.num_warps: KernelConfig#num_warps.
  KernelConfig.num_stages: KernelConfig#num_stages.
  KernelConfig.algorithm: KernelConfig#algorithm.
  KernelConfig.priority: KernelConfig#priority.
---
# Module: [`ejkernel/modules/base.py`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py)

## Classes
### `KernelConfig`
- def: [`ejkernel/modules/base.py:177`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L177)
- doc: Configuration for kernel execution with block size tuning.
- signature: `class KernelConfig:`
- members:
  - `__post_init__(self)` — [`L205`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L205) — Normalize backend when platform is XLA.
  - `algorithm` — [`L202`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L202)
  - `backend` — [`L201`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L201) — documented in [ejkernel-modules-base](../../../concepts/ejkernel-modules-base.md)
  - `block_d` — [`L197`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L197)
  - `block_k` — [`L196`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L196)
  - `block_q` — [`L195`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L195)
  - `num_stages` — [`L199`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L199)
  - `num_warps` — [`L198`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L198)
  - `platform` — [`L200`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L200)
  - `priority` — [`L203`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L203)
- uses (calls/refs, reference-scoped): [`Backend`](../kernels/_registry.md#Backend), [`Platform`](../kernels/_registry.md#Platform), [`ANY`](../kernels/_registry.md#Backend.ANY)
- used by: [`create_default_executor`](base.md#create_default_executor)

## Functions
- `create_default_executor(persistent_cache_path: str | None = None, enable_autotuning: bool = True, warmup_iterations: int = 2, timing_iterations: int = 5)` — [`L215`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L215) — Create a default executor with standard configuration. — documented in [ejkernel-modules-base](../../../concepts/ejkernel-modules-base.md)
- `detect_platform(algorithm: str, platform: Platform | Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = "auto", prefer_pallas: bool = False, prefer_cuda: bool = False, prefer_triton: bool = False, prefer_cute: bool = False, prefer_tilelang: bool = False)` — [`L45`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L45) — Detect the best platform for a given algorithm. — documented in [ejkernel-modules-base](../../../concepts/ejkernel-modules-base.md)
- `mesh_to_jax_mesh(mesh: object)` — [`L283`](../../../../../../raw/code/ejkernel/ejkernel/modules/base.py#L283) — Return a ``jax.sharding.Mesh`` compatible object from *mesh*. — documented in [ejkernel-modules-base](../../../concepts/ejkernel-modules-base.md)

