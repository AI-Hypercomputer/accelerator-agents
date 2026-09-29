---
model: qwen3
lane: jax
title: 8k sequence length baseline on v6e-8
status: open
expected_effort: S
expected_gain: 0%
---

# 8k sequence length baseline on v6e-8

## Mechanism
Establish a baseline at 8k sequence length on v6e-8 using the best valid stack we have so far (Splash attention, Tokamax CE, gate_up_fusion) at the maximum batch size that fits (likely bs=1 or bs=2). This gives us the bottleneck profile for further optimization at seq=8192.

## Predicted signal
Compilation succeeds without OOM, yielding a baseline MFU and an xprof trace.
