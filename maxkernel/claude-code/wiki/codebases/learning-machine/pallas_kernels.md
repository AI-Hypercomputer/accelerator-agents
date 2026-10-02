---
title: "learning-machine — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, learning-machine]
created: 2026-07-04
updated: 2026-07-04
---

learning-machine (google's `qihqi/learning_machine` scratch/experiment repo) defines **3 small,
locally-authored Pallas kernels** across two throwaway experiment dirs — one real TPU-lowered bf16
matmul used for an MFU microbenchmark, and two `interpret=True` "hello-world" demos wired through
torchax's torch→JAX interop. All are `mosaic_tpu` when lowered; the torch_pallas pair run in Pallas
**interpret mode** (CPU emulation, no TPU lowering). None are vendored — they are original
learning/benchmark code. The repo also *imports* upstream splash/flash attention (§ Imported below),
but authors no attention kernel of its own.

## Kernels

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `matmul_kernel` / `matmul` | mosaic_tpu | bf16 tiled `x @ y` MFU microbenchmark. 3D grid `(A//512, B//2048, C//2048)`; per-M×N-tile bf16 accumulator lives in a **VMEM scratch** (`pltpu.MemorySpace.VMEM`, zeroed on `program_id(2)==0`, flushed to `z_ref` on the last K tile); `dimension_semantics=('parallel','parallel','arbitrary')` marks K sequential. `dot` runs `preferred_element_type=fp32`, accumulator stored bf16. Reports TFLOP/s + MFU against 918 TFLOP/s v6e peak. | [jax_perf/matmul.py](../../../raw/code/learning-machine/jax_perf/matmul.py#L9) · [catalog](catalog/jax_perf/matmul.md) | research (benchmark scratch) | novel |
| `add_vectors_kernel` | interpret (Pallas `interpret=True`) | Canonical Pallas "hello world": `o_ref[...] = torch.add(x, y)` written in **torch** ops, run through `torch_pallas_call` which `jax_view`s the torch kernel, builds a `pl.pallas_call`, and `torch_view`s the result. Demonstrates torchax torch↔JAX interop over Pallas. `interpret=True` → CPU emulation, not TPU-lowered. | [torch_pallas/torch_pallas.py](../../../raw/code/learning-machine/torch_pallas/torch_pallas.py#L25) · [catalog](catalog/torch_pallas/torch_pallas.md) | research (demo) | novel |
| `matmul_kernel` (torch_pallas) | interpret (Pallas `interpret=True`) | 2×2-grid blocked matmul with fused activation, again written in torch ops (`activation(torch.matmul(x, y))`) and dispatched via `torch_pallas_call`; BlockSpecs split M and N in half. Also `interpret=True` (CPU). Shows an activation-fused matmul across the torchax bridge. | [torch_pallas/torch_pallas.py](../../../raw/code/learning-machine/torch_pallas/torch_pallas.py#L42) · [catalog](catalog/torch_pallas/torch_pallas.md) | research (demo) | novel |

**Imported (not defined here):**
- `jax_perf/splash_attention_debug.py` / `splash_attention_debug2.py` — import and benchmark upstream
  `jax.experimental.pallas.ops.tpu.splash_attention` + `flash_attention`; a Flax `SplashAttention`
  module wraps `make_splash_mha` under `shard_map`. No local kernel — a timing/debug harness (its
  header records v6e-8 numbers for dynamic/static/causal/flash variants).
- `custom_kernel_spmd.py` — imports `_flash_attention_impl` / `_flash_attention_bwd_*` from the same
  upstream package for a torch_xla SPMD custom-call experiment. No local kernel.

## Notes
- **Vendoring lineage:** none — all three kernels are original. This is a personal experiment repo, so
  "stability" is *research/scratch* throughout; none of these is packaged or imported by other modules.
- **Two distinct Pallas idioms on display:** (1) `jax_perf/matmul.py` is a genuine TPU kernel exercising
  VMEM scratch accumulation + `dimension_semantics` for a real MFU measurement (the only reusable perf
  artifact here); (2) `torch_pallas/torch_pallas.py` is the interesting *interop* pattern — writing
  Pallas kernel bodies in **torch** and lowering them through torchax (`interop.jax_view` /
  `interop.torch_view`), which is the transferable idea even though both demos stay in interpret mode.
- **Gotcha for reuse:** the torch_pallas demos hard-set `interpret=True`; dropping them onto TPU would
  require removing that flag and satisfying the 8/128 alignment rules the interpret path skips. The
  `jax_perf` matmul hard-codes grid divisors (512/2048/2048), so it only runs on shapes divisible by
  those tiles.
- **Reconciliation with the cross-repo directory (2026-04-23):** learning-machine does **not** appear in
  the Pallas kernel directory or its subpages — these kernels were not previously cataloged. This page is
  the first inventory of them; they are minor (benchmark/demo grade) and unlikely to warrant a directory row.

## See also
- [learning-machine — overview](overview.md) — the repo's grounded wiki front door.
- [custom_kernel_spmd](concepts/custom_kernel_spmd.md) — the torch_xla SPMD flash-attention experiment (imports upstream, no local kernel).
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — cross-repo catalog; learning-machine is not (yet) listed there.
