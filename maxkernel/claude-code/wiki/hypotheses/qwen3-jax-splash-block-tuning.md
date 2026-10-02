---
title: "Qwen3 jax — splash attention block-size sweep"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "0.5-1% MFU"
confidence: medium
effort: S
origin: 2026-06-02-v017-sc-bs4
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — splash attention block-size sweep

The splash kernel is the dominant non-matmul op on the frontier (v008 profile: splash
custom-call 11.0%). Its block sizes (`block_q`, `block_kv`, and the dkv/dq backward blocks)
are currently at the kernel defaults. Tuning them to the v6e VMEM budget may recover cycles
without any model-semantics change.

## Statement

Sweeping splash `block_q` / `block_kv` (and the backward dkv/dq blocks) around the defaults,
on the v008 frontier shape, lifts MFU ≥ +0.5 pp over 32.4% by improving MXU occupancy / reducing
VMEM spill in the attention kernel.

## Rationale

- v006/v007 showed the backward splash kernel is VMEM-sensitive (overran the 32M scoped limit at
  block=2048); block sizing directly trades VMEM footprint vs MXU tile efficiency.
- Splash is 11% of step time — even a modest kernel speedup is a real frontier nudge.
- Cheap if the trainer exposes block sizes; otherwise a small code change to thread them through.

## Proposed experiment

Expose `--splash_block_q` / `--splash_block_kv` (default = current); sweep e.g. {512, 1024, 2048}
× {1024, 2048} on the v008 shape with the vmem flag. Pick the best; verify loss parity.

## Risks

- Wrong block size re-triggers the scoped-VMEM OOM (v006/v007) — keep the `scoped_vmem` flag and
  bound the sweep.
- Gain may be < noise; low expected magnitude.

## Dependencies

May need a small trainer edit to plumb block-size flags into the splash kernel call.
