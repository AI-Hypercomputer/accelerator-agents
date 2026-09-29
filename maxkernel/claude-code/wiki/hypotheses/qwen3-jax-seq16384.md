---
title: "Qwen3 jax — seq16384 splash scaling"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "target-seq+ coverage; MFU TBD"
confidence: low
effort: S
origin: 2026-06-02-v020-tokamax-splash-knobs
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — seq16384 splash scaling

Splash attention removes the `[B,H,S,S]` score wall, so seq can scale past the 8192 target.
At longer seq the per-step MFU often *rises* (attention's linear-in-seq splash cost amortizes the
fixed matmul/collective overhead better). This probes whether the upstream-splash frontier scales
to 16384 and what MFU it reaches.

## Statement

`--seqlen=16384` on the splash+remat frontier (bs1, scheduler flags) runs without OOM and reaches
MFU ≥ the seq8192 frontier (30.4%), demonstrating headroom beyond the program-target seq.

## Rationale

- Splash makes seq16384 feasible (dense path OOMs even at seq2048 bs4).
- Useful to know the lane's behavior at 2× target seq for any future long-context requirement.
- Flag-only: `--seqlen=16384 --batch_size=1` + vmem flag (may need a larger scoped-vmem if the
  splash dkv blocks overflow — bound via `SPLASH_BKV_DKV`).

## Proposed experiment

v0NN: frontier launch with `--seqlen=16384 --batch_size=1`. Measure MFU/tok-s-chip, OOM/no-OOM, loss
sanity. Low priority (not the program target) — run as a cheap flag-only probe when a slot is free.

## Risks

- splash backward dkv VMEM OOM at 16384 (the v006/v007 failure mode at 8192) — keep scoped_vmem high,
  reduce `SPLASH_BKV_DKV` if needed.
- Low confidence it beats seq8192 MFU; primarily a coverage probe.

## Dependencies

None — flag-only.
