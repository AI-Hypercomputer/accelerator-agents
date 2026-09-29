---
title: "Qwen3 JAX v143 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 73728"
type: experiment
hypothesis: "Lowering scoped VMEM may alter the batch-4 collective-matmul-none schedule differently from the old pre-collective-matmul-none VMEM bracket and reduce contention without losing the frontier HLO family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v143 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 73728

## Hypothesis under test

**Hypothesis**: The old v068/v071 VMEM sweep showed 81920 KiB beat 73728 KiB
before collective-matmul modes were disabled. The newer batch-4
collective-matmul-none HLO family has different HBM pressure and collective
placement, so 73728 KiB may now trigger a useful schedule change or reduce
custom-call contention.

**Mechanism**: Restore the v127 simplified batch-4/global-32 frontier stack
after the shared-memory sweep, including scheduler shared-memory limit 100,
scheduler rerun 3, host send/recv 100, host-transfer overlap 24, and
collective-matmul modes `NONE`; change only:

```text
--xla_tpu_scoped_vmem_limit_kib=73728
```

from the current baseline value 81920.

**Predicted signal**: Valid train-step compile with either a same-HLO no-op or
a changed schedule. Support requires lower step time / higher best-worker
average than the v123/v127/v136-v142 band without loss drift.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion in the same 56.43k-56.46k tok/s
frontier band. A regression like old v071 refutes the retest.

## Setup

GKE workload `alekseyv-qwen3-v143-bs4-vmem73728`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v143-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem73728`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 simplified stack plus `--xla_tpu_scoped_vmem_limit_kib=73728`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v143-bs4-vmem73728` completed cleanly on both
workers with normal loss ending near 12.0460.

- Worker 0: 55,810 tok/s average, 6,976 tok/s/chip, approx 40.0% MFU.
- Worker 1: 55,791 tok/s average, 6,974 tok/s/chip, approx 40.0% MFU.
- Best observed steady steps: worker0 55,909 tok/s, worker1 55,903 tok/s.

This is a hard regression from the 56.43k-56.46k frontier band.

## Profile

XProf run
`2026-06-04-qwen3-jax-v143-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem73728/2026_06_04_15_46_45`:

- Step time average: 2021.3 ms.
- Total profile time: 113,085.9 ms; idle 289.6 ms / 0.3%.
- MXU utilization: 63.3%.
- Train-step program: `jit_train_step(5130325203490098187)`.
- Top buckets: convolution fusion 60,700.8 ms / 53.7%, custom-call
  29,362.1 ms / 26.0%, loop fusion 11,883.2 ms / 10.5%,
  all-reduce-scatter fusion 3,998.5 ms / 3.5%.
- Splash custom calls: DKV 15,484.9 ms / 13.7%; forward residuals combined
  13,877.2 ms / 12.3%.

The lower VMEM cap reduces reported bytes moved in some convolution buckets but
increases wall time and lowers MXU, matching the old VMEM-bracket failure mode.

## HLO Dump

Fresh train-step HLO dumps were emitted twice (`module_0262` and
`module_0267`) with identical content:

- Optimized HLO size: 792,825 bytes.
- SHA256:
  `b973254c94f02a50011375ed7283dd9f28824adbce31186ab89eb21c3d08c9e1`.
- Memory report: 28,870,695,356 bytes / 26.89 GiB.
- Flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=73728`,
  `--xla_tpu_scheduler_percent_shared_memory_limit=100`, scheduler rerun 3,
  and both collective-matmul modes `NONE`.

This is a real HLO change versus the 81920 KiB frontier and it increases HLO
size and compiled memory slightly.

## Verdict

Refuted. Lowering scoped VMEM to 73728 KiB in the current batch-4
collective-matmul-none family reproduces the old VMEM-bracket behavior:
changed HLO, slightly higher compiled memory, lower MXU, and worse wall time.
Do not carry 73728; keep 81920 for the frontier. Next test the upper-side VMEM
bracket point 86016 KiB once in this newer HLO family, then close VMEM if it
also regresses.
