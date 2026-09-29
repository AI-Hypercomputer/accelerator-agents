---
variant: 8B/v6e-8
hypothesis: "Apply selective rematerialization (activation checkpointing) on top of Splash attention at `global_batch=64` (`per_chip_batch=8`)."
status: refuted
created: 2026-06-13
---

# v052: Selective Rematerialization at bs=64

## Hypothesis under test
**Hypothesis**: Wrapping the loss computation in `jax.checkpoint` with the `checkpoint_dots_with_no_batch_dims` policy will reduce activation memory sufficiently to enable training Qwen3-8B with a per-chip batch size of 8 (global 64) on TPU v6e-8 without triggering HBM Out-of-Memory.
**Mechanism**: We will wrap the training step loss function using `jax.checkpoint` with `policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims`. This instructs the compiler to persist expensive dot-product intermediates in HBM while recomputing lightweight elementwise intermediate activations during the backward pass.
**Predicted signal**: The compilation should succeed without a `RESOURCE_EXHAUSTED` error. During runtime, step time will slightly increase, but overall MFU will be higher than the 30.6% baseline due to increased batch size.
**Falsification criterion**: The compilation still OOMs despite selective remat, or the runtime MFU is worse than 30.6%.

## Setup
- **Hardware**: TPU v6e-8 (GKE cluster `charles-v6e`, project `<your-project-2>`, zone `<your-zone>`)
- **Dispatch**: GKE/XPK via the gke-cluster-runner agent.
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260613-exp052-selective-remat`
- **Command**:
  ```bash
  python -u train.py --train_steps 20 --batch_size 8 --seqlen 8192 --use_splash True --use_real_data False --profile_gcs_dir gs://<your-project>-alekseyv-asia-northeast1/autoresearch/qwen3-jax-v052-selective-remat/plugins/profile --profile_start_step 10
  ```

## Results

| Metric | This run (v052) | Notes |
|---|---|---|
| Steady step time | N/A | Crashed during compilation |
| Throughput | N/A | Crashed during compilation |
| MFU | N/A | Crashed during compilation |
| Cold compile | N/A | Aborted due to OOM |
| Exit code | 1 | |

- **Peak HBM**: 33.07 GB (exceeded physical v6e-8 capacity of 31.25 GB)
- **Wall Time**: 76 s
- **Error Headline**: `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 33.07G of 31.25G hbm. Exceeded hbm capacity by 1.83G.`

## Profile
No profile trace was generated because the training workload crashed during the compilation pass.

## HLO Dump
No HLO dump was generated because the training workload crashed during the compilation pass.

## Verdict
**Verdict**: refuted

The hypothesis was that selective rematerialization (`checkpoint_dots_with_no_batch_dims`) would reduce activation memory sufficiently to fit Qwen3-8B at a per-chip batch size of 8 (`global_batch=64`) on TPU v6e-8 with Splash attention. This is refuted because the compilation pass still encountered a fatal HBM OOM:
- **HBM Required**: 33.07 GB
- **HBM Physical Cap**: 31.25 GB
- **Exceeded by**: 1.83 GB

Compared to `v049` (Splash attention without selective remat at `bs=64`), which required 33.08 GB, selective rematerialization provided negligible memory savings (0.01 GB) in this configuration. This suggests that the memory bottleneck at this scale is dominated by the Splash attention kernel's memory allocation (which selective remat does not target) or that the `checkpoint_dots_with_no_batch_dims` policy is failing to match and rematerialize intermediate tensors effectively with Splash enabled. We must explore alternative optimization avenues such as Pallas fused RMSNorm or advanced activation checkpointing policies.
