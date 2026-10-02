---
model: qwen3-cc-jax
lane: jax
status: drafted
gain: +1-2pp
confidence: medium
effort: S
---

# Hypothesis: Selective Rematerialization enables batch size 64

## Mechanism
Apply selective rematerialization (activation checkpointing) on top of Splash attention at `global_batch=64` (`per_chip_batch=8`). In `v049`, Splash attention at `bs=64` exceeded the 31.25GB HBM limit by only 1.83GB. We previously saw in `v003` that selective remat can save over 9GB of memory. Therefore, applying selective remat should reduce the memory footprint enough to allow `bs=64` to compile and run.

## Predicted signal
The compilation should succeed without an `RESOURCE_EXHAUSTED` error. During runtime, the step time may slightly increase due to recomputation, but the increased arithmetic intensity from doubling the batch size (from `bs=32` to `bs=64`) will amortize the communication overhead, leading to a net increase in MFU compared to the `v047` baseline (30.6% MFU).

## Falsification criterion
The compilation still OOMs despite selective remat, or the runtime MFU is worse than 30.6% due to the overhead of recomputation outweighing the batch size gains.

## HLO pre-filter
N/A — This is not a kernel replacement hypothesis; it's a memory optimization strategy.
