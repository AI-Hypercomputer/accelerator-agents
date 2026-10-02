## [2026-06-02] start | /start-experiment session begin

**Op**: start
**Cluster pool**: <your-cluster>
**Parallelism**: 1
**First-pick hypothesis**: Selective SparseCore offload (reduce-scatter offload only) to minimize host transfer overhead and PCIe bottleneck.
**Notes**: session opened/re-grounded via /start-experiment.

## [2026-06-02] loop-iteration | v002 selective SparseCore offload crashed

**Op**: loop-iteration
**Experiment**: [2026-06-02-qwen3-jax-v002-selective-sc-offload.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v002-selective-sc-offload.md)
**Verdict**: invalid
**Notes**: The GKE workload crashed immediately on startup with `Unknown flag in XLA_FLAGS: --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true`. This flag is not supported by the current image's libtpu/XLA version.

## [2026-06-02] loop-iteration | v003 selective rematerialization crashed

**Op**: loop-iteration
**Experiment**: [2026-06-02-qwen3-jax-v003-selective-remat.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v003-selective-remat.md)
**Verdict**: refuted
**Notes**: The selective rematerialization policy `checkpoint_dots_with_no_batch_dims` reduced compilation-time HBM usage by 9.38 GB (from 42.99 GB to 33.61 GB), but still exceeded the physical 31.25 GB limit by 2.36 GB, resulting in a compilation HBM OOM.

## [2026-06-15] loop-iteration | v054-c fused-qknorm-rope crashed

**Op**: loop-iteration
**Experiment**: [2026-06-15-qwen3-jax-v054-fused-qknorm-rope-c.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-15-qwen3-jax-v054-fused-qknorm-rope-c.md)
**Verdict**: invalid
**Notes**: The workload crashed during JAX compilation because the `fused_qknorm_rope` Pallas kernel was not wrapped in a `shard_map`. We will wrap the kernel call in a `shard_map` using `_SPLASH_MESH` and retry.

## [2026-06-15] loop-iteration | v054-d fused-qknorm-rope crashed

**Op**: loop-iteration
**Experiment**: [2026-06-15-qwen3-jax-v054-fused-qknorm-rope-d.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-15-qwen3-jax-v054-fused-qknorm-rope-d.md)
**Verdict**: invalid
**Notes**: The workload crashed during setup with `ValueError: fused_qknorm_rope requires a registered mesh via set_splash_mesh` because `set_splash_mesh()` was only called when `use_splash=True`. I have fixed `train.py` to call it unconditionally and will retry.

## [2026-06-15] loop-iteration | v054-e fused-qknorm-rope crashed (OOM)

**Op**: loop-iteration
**Experiment**: [2026-06-15-qwen3-jax-v054-fused-qknorm-rope-e.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-15-qwen3-jax-v054-fused-qknorm-rope-e.md)
**Verdict**: invalid
**Notes**: The workload crashed with an OOM error during the first jitted_step: `RESOURCE_EXHAUSTED: Allocation (size=34359738368) would exceed memory`. The shape mentioned is `f32[4,32,8192,8192]`, which is a materialized attention matrix. The fusion must be allocating too much memory or disabling an optimization that saves memory.

## [2026-06-15] formulate-hypothesis | v055 selective-remat

**Op**: formulate-hypothesis
**Experiment**: [2026-06-15-qwen3-jax-v055-ac-selective-remat.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-15-qwen3-jax-v055-ac-selective-remat.md)
**Status**: in_progress
**Notes**: Formulated hypothesis to address HBM limits via Activation Checkpointing.
## [2026-06-15] loop-iteration | v055 selective activation checkpointing crashed

**Op**: loop-iteration
**Experiment**: [2026-06-15-qwen3-jax-v055-ac-selective-remat.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-15-qwen3-jax-v055-ac-selective-remat.md)
**Verdict**: refuted
**Notes**: The run failed during compilation. However, it was not an HBM OOM (as in previous runs), but rather a VMEM (Vector Memory) OOM inside the Pallas `splash_mha_dkv_no_residuals` backward kernel. Scoped allocation of 35.63M exceeded the 32.00M limit.

