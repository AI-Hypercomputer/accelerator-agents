## [2026-06-02] reference-result | qwen3-8b MaxText @ seq2048 = 38.0% MFU / 7,116 tok/s/chip (jax at PARITY)

Workload ale-qwen3-8b-2-1-060214-lgr Completed clean (bs4, global 65536 tok, loss 0.083->0.035). Steady-state (steps 16-19): 348.9 TFLOP/s/device -> 38.0% MFU, 7,116 tok/s/chip, 1.151s/step. **vs jax v018 (35.8% / 6,964): jax at 97.9% of MaxText tok/s/chip — near parity.** EXTERNALLY CONFIRMS the jax seq2048 frontier is at its practical ceiling (~2% residual, within noise + MaxText's heavier offload recipe). Contrast: seq8192 jax is at 76.4% of MaxText (real ~31% gap = offload-enabled bs3). Net: jax seq2048 CLOSED at parity; jax seq8192 has the actionable headroom. Profile in GCS (no HLO). Reference lane baseline now complete (seq8192 45.3% + seq2048 38.0%).

## [2026-06-02] reference-result | qwen3-8b MaxText @ seq8192 = 45.3% MFU / 6,942 tok/s/chip (CEILING established)

Workload ale-qwen3-8b-8-1-060213-9sh Completed clean (20 steps, loss 2.96->2.29 synthetic). Steady-state (steps 15-19): 415.8 TFLOP/s/device -> **45.3% MFU**, 6,942 tok/s/chip, 3.540s/step. On par with llama3-8b MaxText (44.6%) -> recipe transfers to qwen3-8b incl. QK-norm. **This is the achievable ceiling.** Cross-stack @ seq8192: MaxText 6,942 vs jax v009 5,305 = **+31% tok/s/chip** (jax ~15pp below). Gap = MaxText's offload-enabled bs3 (decoder_layer_input + 4 proj offloads) — UNTRIED on jax. REOPENS the jax seq8192 frontier (new hyp qwen3-jax-host-offload-bs3-seq8192). Profile in GCS (no HLO; recipe doesn't dump). Next: seq2048 reference (qwen3_8b_2048_ref) for the direct vs-v018 comparison.

## [2026-06-02] start | maxtext reference lane — first run dispatching

**Op**: start (reference-ceiling lane for qwen3_cc).
**Trigger**: confirmed MaxText ships `configs/models/qwen3-8b.yml` matching our architecture exactly (obs `qwen3-maxtext-config-exists.md`).
**Setup**: worktree `/mnt/disks/persist/maxtext-tpu-recipes-v0.1.4` (`tpu-recipes-v0.1.4` / `9f1820b47`), venv `/mnt/disks/persist/venv-maxtext-v0.1.4`, base image `maxtext_base_image:latest` (rebuilt 2026-06-02). Added benchmark configs `qwen3_8b_8192_ref` (seq8192 bs3) + `qwen3_8b_2048_ref` (seq2048 bs4) mirroring the llama3 `no-collective-matmul` recipe tuning.
**Cluster**: <your-cluster> (v6e-8, fsdp=8). **First dispatch**: `qwen3_8b_8192_ref` (program-target seq, directly comparable to the llama3 MaxText 44.6% reference).
**Notes**: run via `benchmark_runner xpk` (builds runner image on the base, pushes to <your-registry>, submits xpk workload). No model code in this lane — MaxText verbatim.
**Workload**: `ale-qwen3-8b-8-1-060213-9sh` (seq8192, bs3). Entrypoint `python3 -m MaxText.train ... model_name=qwen3-8b`; full MaxText recipe XLA stack incl. SparseCore-offload + HOST_OFFLOAD + jf_spmd (note: jf_spmd crashed our jax build v019 but MaxText's jax0.6.1 image accepts it).
