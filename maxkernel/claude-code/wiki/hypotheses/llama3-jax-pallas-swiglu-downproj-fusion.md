---
title: "Pallas SwiGLU + down_proj fusion (Llama 3 8B JAX)"
type: hypothesis
tags: [llama3, jax, pallas, mosaic-tpu, swiglu, gated-linear-unit, matmul, fusion, deep-work]
created: 2026-04-27
updated: 2026-04-27
model: llama3-8b-jax
status: refuted
expected_gain: "+2-4 % step time (claimed) — invalidated by 2026-04-27 HLO inspection"
confidence: medium
effort: L
origin: jax-exp28b-profile-2026-04-26
---

Custom Pallas TPU kernel that fuses **`silu(g) * u` (SwiGLU) into the `down_proj` matmul prologue** so the MLP intermediate `silu(g)*u` (4 × 8192 × 14336 bf16 ≈ 939 MiB/layer/chip × 32 layers = 30 GiB/step of HBM traffic) never round-trips through HBM. Targets the 9.2 % loop-fusion line in [exp 28b's profile](../experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md#profile).

> [!warning] **Refuted 2026-04-27** — HLO inspection of exp 28b shows XLA already fuses this. The down-proj fusion (`%fusion.323 = ... kind=kOutput, calls=%fused_computation.40`) contains: an inner `%fusion.311 = ... calls=%fused_computation.8` that does `silu(g) * u` (negate→exp→add→divide→multiply, then multiply with `u`), the `%convolution.111` down-proj matmul reading `fusion.311`'s output directly, and a final `%add.856` residual add. **One single Mosaic `kind=kOutput` kernel** that takes `g, u, down_proj_weight, residual` and emits `hidden_state` — exactly the kernel this hypothesis proposed to write. The pallas-forge 0.65× v5e result mentioned in the original Risks was a leading indicator: XLA's TPU fuser already does this work. **Hypothesis withdrawn.** See `raw/profiles/2026-04-27-jax-hlodump-exp28b/module_0262.jit_train_step.cl_854318611.after_optimizations.hlo` (extracted to local `/tmp/hlo-exp28b-renamed/`) for the HLO evidence — search for `fused_computation.8` (the silu*u body) and `fused_computation.40` (the outer SwiGLU+down_proj+residual fusion).

## Statement

Replacing the SwiGLU body in [`model/modeling_llama3.py:_decoder_call`](../experiments/llama3_8B_autoresearch_optimization/jax/model/modeling_llama3.py) — currently three separate ops (`silu(g)`, `silu(g)*u`, `_matmul(..., down_proj)`) — with a single Pallas-Mosaic kernel that consumes `g, u, down_proj` and emits `h` directly will reduce step time by **2–4 %** at bs=4 seq=8192 with no semantic change.

## Rationale

Per layer, current SwiGLU executes:
```
g = matmul(x, gate_proj)           # writes 939 MiB to HBM
u = matmul(x,   up_proj)           # writes 939 MiB to HBM
mid = silu(g) * u                  # reads 2 × 939 MiB, writes 939 MiB
h   = matmul(mid, down_proj)       # reads 939 MiB
```

A fused kernel would:
- Read `g` and `u` blocks from HBM directly into VMEM
- Compute `silu(g_block) * u_block` in VMEM (free)
- Feed the MXU as the matmul-A operand for `down_proj`
- Write only `h` (256 MiB/layer) to HBM

Saved traffic: **939 MiB write of `mid` + 939 MiB read of `mid` = 1.88 GiB/layer × 32 layers ≈ 60 GiB/step of HBM** (matches the high-bandwidth share of the loop-fusion line in the profile, which reports 1,547 GiB/step total).

The wiki concept page [gated-linear-unit](../concepts/gated-linear-unit.md) records that **tokamax's TPU GLU path falls back to XLA**. A 2026-04-27 web + repo survey (research agent) confirmed **no off-the-shelf dense TPU Pallas SwiGLU+matmul fused kernel exists in any public repo**. Notable findings:

- **`raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py:77-85,1476-1482`** — a production TPU Pallas kernel that **already fuses `silu(gate)*up` with two grouped matmuls**. MoE-grouped, not dense, but proves the GMM-with-fused-SwiGLU pattern compiles and runs on TPU Mosaic. **Best scaffolding starting point.**
- **`raw/code/pallas-forge/pallas_forge/kernels/swiglu.py`** — minimal 2D-grid TPU Pallas SwiGLU; **measured 0.65× XLA on v5e** ([codebase page](../codebases/pallas-forge.md)). Strong yellow flag that XLA already fuses the equivalent pattern on TPU — see Risks below.
- **alphafold3 v3.0.1 `gated_linear_unit/`** — GPU-only (Pallas-on-Triton via Mosaic-GPU); algorithmic reference for the two-matmuls-share-one-activation-load pattern, NOT portable to TPU Mosaic without a rewrite.
- **upstream JAX `jax.experimental.pallas.ops.tpu/`** — no GLU file (only flash/splash/paged/megablox/matmul/random/all_gather).
- **Helion PR pytorch/helion#1637** — in-flight TPU-Pallas SwiGLU; not yet deployable.

So the kernel does not exist, and one prior TPU Pallas SwiGLU attempt (pallas-forge) lost to the XLA fused-loop baseline. **Before writing this kernel, validate via HLO dump that XLA is NOT already emitting a single fused (matmul→silu→mult→matmul) kernel for our MLP** — if it is, the win is gone.

## Proposed experiment

1. Author `pallas_swiglu_matmul.py` — Mosaic-TPU kernel:
   - Inputs: `g, u : (B*T, ffn) bf16`, `w : (ffn, hidden) bf16`
   - Block tile: matmul-style 256×128 along the (B*T, hidden) output, ffn dimension iterated with double buffering for the silu*u prologue
   - Pipeline: prefetch `g_block` and `u_block` → compute `silu(g)*u` in VMEM → emit MXU tile against `w_block` → accumulate
   - Custom_vjp: backward needs `g, u, w`; recompute `silu(g)*u` in bwd (saves nothing on the activation; backward gates+ups separately)
2. Wire into `_decoder_call` via env-flag `JAX_MLP_IMPL=pallas|xla` (so we can A/B without recompiling).
3. Run with the exp 28b stack otherwise unchanged.

## Measurement

| Metric | Method | Pass criterion |
|--------|--------|----------------|
| tok/s/chip | trainer | ≥ 7,920 (≥ +2 %) |
| MFU | trainer | ≥ 44.4 % |
| Loss step 0–8 | trainer log | within bf16 noise of exp 28b (Δ ≤ 0.005/step) |
| HBM peak | xprof memory profile | within 5 % of exp 28b |
| Loop-fusion % | xprof op_profile | ≤ 6 % (down from 9.2 %; full reduction would need RMSNorm fusion too) |
| Loop-fusion bytes | xprof | ≤ 1,200 GiB/step (down from 1,547) |

## Risks

- **XLA already does this fusion** (the biggest risk). Pallas-forge's TPU SwiGLU loses 35 % to XLA on v5e — the XLA TPU fuser may already emit the (matmul→silu→mult→matmul) pattern as a single op. **Validate-first**: dump exp 28b's optimized HLO, search for the MLP body, check whether the three matmuls + silu + mult are already inside one fusion node. If yes, abandon. If no, the kernel is worth writing. This validation is a 30-minute HLO-reading task, not a kernel-write.
- **Sigmoid precision**. Silu = `x * sigmoid(x)`; sigmoid in bf16 has known precision pitfalls near 0. If the kernel uses bf16 throughout the silu, training may diverge subtly. Test by running 50 steps and comparing loss curve to exp 28b — a single-step Δ ≤ 0.005 doesn't guarantee long-run parity.
- **MXU tile waste**. ffn=14,336 doesn't divide cleanly into 256-tile multiples (14,336 / 256 = 56 — clean) but the (B*T) dim 4 × 8192 / 8 = 4096 is fine. Should be OK; verify by checking padding overhead.
- **Backward kernel** is harder than forward. Standard pattern: implement fwd in Pallas, bwd in XLA (calling matmul + elementwise). The mid-recompute in bwd reads `g, u` from HBM again — same as XLA today; net bwd cost unchanged. Forward is where the win comes from.
- **Compile cache pollution**. New custom-call's HLO bytes change the cache key; expect 1 cold-compile per shape.

## Dependencies

- Pallas-Mosaic familiarity. **Best scaffolding**: `raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py:77-85,1476-1482` — already fuses `silu(gate)*up` with two grouped matmuls in a single TPU Pallas kernel. Strip the routing/grouping for a dense scaffold. Secondary: `jax.experimental.pallas.ops.tpu.megablox/gmm.py` (canonical grouped-matmul template; no built-in activation).
- A correctness reference: tokamax's CPU/GPU `gated_linear_unit` fp32 path can serve as a numerical oracle.
- Should be undertaken as a single project with the [RMSNorm + matmul prologue fusion hypothesis](llama3-jax-rmsnorm-matmul-prologue-fusion.md) — they share ~80 % of the kernel scaffolding (block-tile pipeline, custom_vjp boilerplate, autotune harness, tests).

## See also

- [Pallas RMSNorm + matmul prologue fusion (sibling hypothesis)](llama3-jax-rmsnorm-matmul-prologue-fusion.md) — same code shape, complementary target.
- [JAX exp 28b frontier writeup](../experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md)
- Concepts: [gated-linear-unit](../concepts/gated-linear-unit.md), [pallas-kernel](../concepts/pallas-kernel.md), [mosaic-kernel](../concepts/mosaic-kernel.md)

## Sources

- `raw/profiles/2026-04-26-jax-exp28b-sc-rsag-bs4/` — profile attesting to the 9.2 % loop-fusion / 1,547 GiB-traffic bottleneck.
- `raw/code/alphafold3/` (tag v3.0.1) — GPU Triton fused-GLU reference (algorithmic pattern only).
- `raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py` — Pallas-Mosaic fused matmul + activation pipeline reference.
