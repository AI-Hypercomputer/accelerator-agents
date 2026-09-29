# Gemma 4 E4B — native-JAX stack experiments

Experiment writeups + ledger for the **native-JAX (Flax NNX)** path through the Gemma 4 autoresearch program. Sibling of [`../../torchax/experiments/`](../../torchax/experiments/README.md) (torchax stack).

## Why two stacks?

The torchax stack was built first (it reuses HuggingFace's PyTorch model via torchax). The JAX stack is a from-scratch Flax NNX port (see [exp 34](2026-04-23-exp34-jax-baseline-accepted.md)) and reveals whether torchax's dispatch overhead, custom-call boundaries, and HF-shaped quirks are bottlenecks. Both stacks share hardware, mesh conventions, and the verdict-suffix / profile-link discipline defined in [`../../program.md`](../../program.md).

## Contents

- `2026-04-23-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages. **exp 34** is the JAX-port baseline; numbering continues from the global exp counter so commit history stays linear across both stacks.
- `OBSERVATIONS.md` — skim-and-reason aggregation log, jax-stack-scoped.
- `RESULTS.tsv` — machine-readable ledger (gitignored).

## Current state

### New regime (2026-04-24, exp 52+): fp32 master + bf16 compute AMP

The user requested a **regime change** on 2026-04-24: master weights in fp32 (for the optimizer), matmul/conv compute in bf16 (standard large-model mixed precision), and `seq_len=8192` as the new default. The flags `--weights-dtype {bf16,fp32}` and `--compute-dtype {bf16,fp32}` are now wired into both trainers (JAX full-impl; torchax stub/warn). The legacy `--dtype` flag still sets both dtypes and reproduces the pre-exp-52 behavior.

**Exp 52** (the new-regime baseline) establishes that **`seq_len=8192 b=1 fp32-master` is infeasible on v6e-4** — compile OOM at 35.18 GiB vs 31.25 GiB per-chip HBM, exceeded by 3.93 GiB. Even legacy bf16-only at seq=8192 b=1 OOMs (36.16 GiB). The `seq=2048 b=1 fp32-master / bf16-compute` configuration is the largest feasible representative of the new regime and is filed as the new baseline at **26,807 TPS**, step time 305.6 ms, loss descent 3.25 → 2.30. Non-monotonic XLA scheduling was observed (seq=4096 and seq=6144 report WORSE peak HBM than seq=8192). See [2026-04-24-exp52-jax-fp32master-seq2k-accepted.md](2026-04-24-exp52-jax-fp32master-seq2k-accepted.md) for the full OOM probe table and AMP implementation details.

### Old regime (2026-04-23, exp 34–51): bf16 everywhere

**Exp 36** remains the **old-regime** (legacy `--dtype bf16`) JAX-stack best at **34,614 TPS** (seq=1024, batch=3, fsdp=4, bf16, splash) — **+13.9 %** over exp 35 and **+3.7 % over the torchax session-best** ([exp 25, 33,372 TPS](../../torchax/experiments/2026-04-23-exp25-splash-block1024-accepted.md)). Step time 355.0 ms/step, peak HBM **27.11 GiB / 31.25 GiB = 86.75 %** (fits comfortably with 4.14 GiB of headroom). HLO-op diff vs exp 35 (b=1): splash `custom fusion` near-constant (169 → 175 ms, ×1.03) while matmul `convolution fusion` grew ×2.75 and `loop fusion` ×3.81 — per-call-overhead amortization mechanism per exp 35's predictions. New bottleneck surfaces at b=3: `loop fusion` (28.1 % of step) and `collective-permute-done` (12.2 %, didn't exist at b=1).

**Exp 37** (bf16 CE env-var gate on top of exp 36) landed flat at **34,629 TPS (+0.04 %, within noise)** — the native-JAX port was already running bf16 CE by construction since exp 34, so the torchax-exp-12-style win was a no-op-by-construction. Durable artifact: the `JAX_CE_DTYPE={bf16,fp32}` gate in `train.py`, useful for regression guards on future LM-head refactors. Peak HBM 27.45 GiB / 87.84 % (unchanged heap, +0.34 GiB stack — free headroom 3.80 GiB).

**Exp 43** (tokamax.linear_softmax_cross_entropy_loss — fused `linear + log_softmax + NLL`) is **INVALID** on architecture-precondition grounds. The public tokamax LCE API has no `logits_soft_cap` kwarg, and Gemma 4's `final_logit_softcapping=30.0` is a non-linear `sc * tanh(logits / sc)` element-wise on the `[B, S, V]` logits — cannot be folded into `hidden` or `W` (algebraically impossible for a non-linear post-matmul op), and applying it post-kernel requires materializing `[B, S, V]` which defeats the kernel's sole purpose. Zero lines of code merged. The result empirically confirms the build-target already catalogued in [program.md § "Pallas kernels to BUILD"](../../program.md) — a Gemma-aware CE kernel (inline softcap on the VMEM logits tile before streaming softmax) is the correct path. See [2026-04-23-exp43-jax-tokamax-ce-rejected.md](2026-04-23-exp43-jax-tokamax-ce-rejected.md).

**Exp 47** (marin/levanter fused linear+softcap+CE Pallas kernel) closes exp 43's softcap gap — levanter's `fused_cross_entropy_loss_and_logsumexp_penalty` applies `sc * tanh(logits/sc)` inline on each VMEM logits-tile before the streaming softmax, so `[B, S, V]` logits never materialize — **but regresses −5.61 %** (34,614 → 32,671 TPS, step time 355.0 → 376.1 ms). Parity passes in bf16 (|diff| 0.048 vs 0.05 tol) and smoke step-4 loss matches exp 36 within 0.47 %. Root cause: CE in exp 36 was <3 % of step time and XLA-fused tight with lm_head + softcap + log_softmax; swapping it for a Pallas custom-call adds ~15 ms of boundary overhead + a 1.31-GiB `w_hv` all-gather forced by the mandatory shard_map wrapper (Mosaic custom-calls cannot be auto-partitioned and the kernel needs replicated `[H, V]` weight). Same "Pallas-custom-call tax" pattern as torchax exp 33 (Pallas RMSNorm). Durable: `JAX_CE_IMPL=levanter` gate + import shim + parity harness + `return_hidden` seam on `Gemma4ForCausalLM`. See [2026-04-24-exp47-jax-levanter-ce-rejected.md](2026-04-24-exp47-jax-levanter-ce-rejected.md).

**Exp 49** (scan-over-layers, `JAX_SCAN_LAYERS=1`) replaced the 42-iter Python for-loop in `Gemma4TextModel.__call__` with two nested `jax.lax.scan`s (outer over 7 super-blocks, inner over 5 sliding layers per block; 1 full-attention layer per block inline). **Compile step-0 drops 180 s → 69.3 s (−61.5 %, 2.6× faster)** — durable compile-time win. **TPS regresses −21.2 %** (34,614 → 27,290), MFU 23.05 % → 18.17 %. Loss match clean: step 4 +0.5 %, step 19 +0.5 % (within bf16 reorder noise). Regression from wasted zero-stub matmuls on 18 KV-shared layers + per-layer `jax.checkpoint` forced inside scan body (prevents a 35-GiB activation-stack OOM) + splash+shard_map nested inside scan limiting XLA collective scheduling. **Env gate off by default**; code on main for future dev-loop use where compile-budget matters more than TPS (e.g. iterating on HLO-changing edits with cache misses). See [2026-04-24-exp49-jax-scan-layers-potential.md](2026-04-24-exp49-jax-scan-layers-potential.md).

## Queued experiments (highest-expected-gain first)

- **exp 38** — **collective-permute-done investigation**. 12.1 % of step time at b=3 (549 ms/3-step); `in_shardings` / `out_shardings` audit on the jitted step might reclaim half. Expected +5–6 %. Confidence medium. **Now highest-expected-value open hypothesis.**
- **exp 39** — **Pallas RMSNorm kernel** (210 calls/step, single-HBM-pass). Expected +3–8 % on `loop fusion`. Effort M.
- **exp 41** — b=4. Gated on exp 38 landing; 3.80 GiB free today, b=4 adds ~3.5 GiB.
- **exp 50 / exp 51** — scan-path follow-ups (cond-dispatched shared layers, relaxed remat). Only worth it if a production reason to run the scan path appears. Exp 36 (for-loop) remains default.

## See also

- [`../train.py`](../train.py) — the native-JAX trainer.
- [`../model/`](../model/) — Flax NNX port.
- [Shared program protocol](../../program.md).
- [Session ceiling analysis (torchax, exp 25 era)](../../../../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md).
