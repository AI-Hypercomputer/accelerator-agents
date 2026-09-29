---
title: "Topic: TPU microbenchmarks (single-chip, Ironwood, Trillium)"
type: topic
updated: 2026-07-03
---
# TPU microbenchmarks (single-chip, Ironwood, Trillium)

## Overview

`microbenchmarks/` isolates individual operations (matmul, HBM copy, collectives) from full model
training/inference recipes, at three scales: single-chip (v6e-1 VM), Ironwood multi-chip (GKE
node pools, `2x2x1`/`4x4x4`), and Trillium multi-chip (XPK, v6e-256). The single-chip and Ironwood
suites both live in this same repo directly
([single-chip](../sources/microbenchmarks-README.md), [Ironwood](../sources/microbenchmarks-ironwood-README.md));
the actual benchmark scripts for the multi-chip suites live in the separate
`accelerator-microbenchmarks` repo, which these recipes clone and pin to a specific branch
(`tpu7x-auto` for the Ironwood automation harness, `trillium-collectives` for the Trillium
collectives benchmark).

## Measured results

**Ironwood collectives (all-gather, 2x2x1 vs 2x2x2 topology):**

| Topology | Elements | Bandwidth (GB/s) |
|---|---|---|
| 2x2x1 | 65,536 | 71.8 |
| 2x2x1 | 16,777,216 | 178.8 |
| 2x2x2 | 65,536 | 69.5 |
| 2x2x2 | 16,777,216 | 178.8 |

Bandwidth converges to roughly the same ceiling (~179 GB/s) at large transfer size regardless of
topology (2x2x1 vs 2x2x2), but the smaller topology is *slightly* faster at small transfer sizes —
consistent with 2x2x2 paying a small fixed per-hop latency cost that only amortizes away once the
payload is large enough
([all_gather.md](src:microbenchmarks/ironwood/collectives/all_gather.md#expected-results)).

**Ironwood HBM bandwidth:** a single measured sample shows 3,209.8 GB/s median bandwidth for an
8192 MB tensor copy at 5.35 ms median latency
([hbm.md](src:microbenchmarks/ironwood/hbm/hbm.md#run-hbm-microbenchmarks)); the doc also references
a fuller "expected bandwidth for different matrix size" table.

**Ironwood host-device transfer:** flagged explicitly as work-in-progress — "expected bandwidth
numbers are not yet finalized"
([host_device.md](src:microbenchmarks/ironwood/host_device/host_device.md#host-device-microbenchmarks-on-tpu7x-2x2x1)).

**Ironwood single-chip GEMM (bf16, gemm_multiple_run):** throughput scales from 2.6 TFLOP/s/device
at matrix size 128 to 956 TFLOP/s/device at 32768
([Ironwood hub results](../sources/microbenchmarks-ironwood-README.md)).

**Single-chip matmul (v6e-1, 8192³ bf16):** 827 TFLOPS when timed via `--trace_matcher` (TPU-only,
excludes host↔TPU communication) vs. 754 TFLOPS host-timed (includes compilation/launch/transfer
overhead) — a ~9% measurement-methodology gap purely from where the clock starts/stops
([microbenchmarks/README.md](../sources/microbenchmarks-README.md)).

## Fleet-scale automation (Ironwood only)

Running the Ironwood microbenchmark matrix (multiple topologies × multiple operation types) by hand
doesn't scale, so a dedicated automation harness batches it: Kueue-managed job launch across
topologies → automatic failure retry (up to 3x) → a final aggregator job that consolidates every
job's GCS results into summary TSVs
([automation overview](../sources/microbenchmarks-ironwood-automation-README.md)). It comes in two
flavors: a **static** variant requiring node pools to pre-exist, and a **CCC/autoscaling** variant
that provisions node pools on demand from ComputeClass templates
([autoscaling variant](../sources/microbenchmarks-ironwood-automation-autoscaling-README.md)) — the
same static-vs-CCC choice this repo's [cluster utilities](../sources/utils-ironwood-README.md) offer
for general cluster provisioning, applied here specifically to benchmark-job scheduling.

## Trillium collectives: DCN-specific tuning

The Trillium collectives benchmark (v6e-256, via XPK) is the one microbenchmark recipe that bakes
DCN (cross-slice interconnect)-specific tuning directly into the launch command rather than an XLA
flag on a model config: an enlarged `/proc/sys/net/ipv4/tcp_rmem` and
`LIBTPU_INIT_ARGS=--megascale_grpc_premap_memory_bytes=17179869184
--xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction=true`
([microbenchmarks/trillium/collectives/README.md](../sources/microbenchmarks-trillium-collectives-README.md)) —
worth comparing against the Ironwood collectives benchmark above, which measures *ICI* (intra-slice)
bandwidth rather than DCN.

## See also
- [Ironwood cluster provisioning utilities](../sources/utils-ironwood-README.md) — the standard/CCC
  node-pool scripts this topic's automation harness builds on for benchmark-job scheduling.
- [Ironwood training recipe matrix](ironwood-training-recipes.md) — the full-model-training
  counterpart these single-operation microbenchmarks isolate individual pieces of.

## Sources
- [microbenchmarks/README.md](../sources/microbenchmarks-README.md)
- [microbenchmarks/ironwood/README.md](../sources/microbenchmarks-ironwood-README.md)
- [microbenchmarks/ironwood/automation/README.md](../sources/microbenchmarks-ironwood-automation-README.md)
- [microbenchmarks/ironwood/automation/autoscaling/README.md](../sources/microbenchmarks-ironwood-automation-autoscaling-README.md)
- [microbenchmarks/requirements.txt](../sources/microbenchmarks-requirements.md)
- [microbenchmarks/trillium/collectives/README.md](../sources/microbenchmarks-trillium-collectives-README.md)
- `microbenchmarks/ironwood/collectives/all_gather.md`
- `microbenchmarks/ironwood/hbm/hbm.md`
- `microbenchmarks/ironwood/host_device/host_device.md`
