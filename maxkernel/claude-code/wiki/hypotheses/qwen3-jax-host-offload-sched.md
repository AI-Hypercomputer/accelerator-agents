---
title: "Qwen3 jax — MaxText HOST_OFFLOAD scheduler flag bundle"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "5-10% tok/s/chip (latency-hiding scheduler features; llama3-jax +10% standalone)"
confidence: high
effort: S
origin: 2026-06-12-v009-sc-offload
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — MaxText HOST_OFFLOAD scheduler flag bundle

*Hypothesis*: MaxText's `HOST_OFFLOAD_FLAGS` bundle (11 latency-hiding
scheduler flags: experimental scheduler features, memory-pressure tracking,
aggressive opt-barrier removal, async-depth prioritization, AG backward
pipelining, scheduler rerun ×2, …) lifts tok/s/chip ≥ 3% over the v007
frontier by overlapping the ~30%-of-step FSDP collectives with compute.

*Mechanism*: Pure scheduler/compiler flags (no host offload of tensors
despite the bundle name). Known-good bundle per the blueprint Phase 2 — the
llama3-jax lane's single biggest lever (+10% standalone on v6e-8, its exp 13).
Targets exactly [v007's profile signal](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md)
(collective-permute 15.7% + all-gather 14.3% poorly overlapped).

*Falsification criterion*: tok/s/chip ≤ 6,095 + 1% noise at the identical
v007 shape → refuted at this op-point.

## See also

- [v007 (frontier; profile source)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md)
- [v009 (SC offload refuted — the other collective lever)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v009-sc-offload.md)

> [!warning] Refuted by [v010](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v010-host-offload-sched.md) on 2026-06-12
> Flat (+0.7%, within noise) at seq2048/bs2; +40% compile time. The llama3
> +10% was at seq8192/bs4-5. Flag bundles are shape-gated — re-test at
> seq-8192.
