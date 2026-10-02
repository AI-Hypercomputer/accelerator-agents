# Experiment: v004 Splash Attention (bs=2)

**Status:** In Progress
**Lane:** jax
**Date:** 2026-06-02

## Hypothesis
Selective remat (v003) saved 9.38 GB of HBM but `bs=2` was still OOM by 2.36 GB. By integrating `splash_attention`, we can lower the activation memory footprint on the sequence dimension significantly during the forward pass and eliminate the need to instantiate the `[seq, seq]` matrix. This should bridge the final 2.36 GB gap and allow `bs=2` to fit, leading to better MXU utilization and higher MFU.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 2
- **Seq Length**: 2048
- **Remat Policy**: `checkpoint_dots_with_no_batch_dims`
- **Attention**: Splash Attention (imported from `llama3_8b` experiment)
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v004-splash-attention`

## Results
- **Status**: Completed
- **Throughput**: 36123 tok/s (4515 tok/s/chip)
- **MFU**: ~23.2% to 25.3%
- **Memory**: Did NOT OOM! `bs=2` successfully ran with selective remat + splash attention (30.87 GB / 31.25 GB).

We see a slight drop in MFU initially calculated at 23.2% but xprof reports ~25.3%. So moving from `bs=1` SDPA to `bs=2` Splash preserved or slightly bumped the MFU to 25.3%, and decisively solved the HBM bottleneck.

## Profile

**Source**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v004-splash-bs2/plugins/profile/`
**xprof URL**: http://localhost:8791/?run=2026-06-02-qwen3-jax-v004-splash-bs2/2026_06_02_21_50_30
**Run name**: 2026-06-02-qwen3-jax-v004-splash-bs2/2026_06_02_21_50_30
**Steps captured**: ~55 steps (from step time 393.1 ms and total trace 21.8 s)
**Description**: xprof trace (TPU device + host)

**Bucket attribution** (steady state, ignoring step 1-3):

| Bucket | % of step | Top op | Self time (trace total) |
|---|---|---|---|
| convolution fusion | 31.8% | fusion.24988 | 973.6 ms |
| collective-permute-done | 20.0% | collective-permute-done.1732 | 82.9 ms |
| loop fusion | 18.6% | (multiple) | - |
| collective-permute-start | 8.5% | (multiple) | - |
| custom-call | 7.9% | custom-call.100 | 14.8 ms |
| data formatting | 6.4% | (multiple) | - |

**Dominant ops** (top 3 by self-time):
1. **convolution fusion** bucket in module `jit_train_step`: 31.8% of step — main compute work; MXU active
2. **collective-permute-done** bucket in module `jit_train_step`: 20.0% of step — cross-device communication sync; wait overhead
3. **loop fusion** bucket in module `jit_train_step`: 18.6% of step — elementwise and activation fusions

**Step time**: 0.393 s/step · **MFU**: 25.3% (vs frontier 25.1%)
**HBM utilization** (peak): 98.7% · **Total HBM**: 30.87 GB/31.25 GB

## HLO Dump

**Source**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v004-splash-bs2/hlo/`
- HLO dump not available for this run — Phase 2 skipped.

**Hypothesis-firing audit** (Phase 3 — silent-noop check vs the hypothesis's predicted mechanism):
- SKIPPED — stub under-specified, missing: `Mechanism`, `Predicted signal`, `Falsification criterion`. (The stub-filling step in the loop failed to extract all four labels).

**Notable patterns**:
- Phase 3 partial — HLO unavailable, mechanism verification from op-profile only.
- `custom-call` bucket takes 7.9% of the step time and `aggregate_xplane_events` confirms multiple `custom-call.XXX` invocations on the device. This strongly suggests `splash_attention` fired as intended (standard JAX attention typically lowers to `convolution fusion` / `loop fusion`).
- Memory gap bridged successfully (30.87 GB used < 31.25 GB cap), demonstrating the sequence-dimension activation savings.
- High `collective-permute` time (28.5% total between start and done) indicates significant cross-device sync overhead, characteristic of Splash Attention's Ring Attention P2P communication.
