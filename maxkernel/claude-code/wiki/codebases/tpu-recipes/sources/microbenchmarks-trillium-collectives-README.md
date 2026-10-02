---
title: "Source: microbenchmarks/trillium/collectives/README.md"
type: source-page
doc: microbenchmarks/trillium/collectives/README.md
updated: 2026-07-03
---
# microbenchmarks/trillium/collectives/README.md

Trillium (v6e-256) collectives benchmarking via XPK, using the `accelerator-microbenchmarks` repo's
`trillium-collectives` branch
([instructions-for-running-collectives-benchmark-on-tpu-trillium-v6e-256](src:microbenchmarks/trillium/collectives/README.md#instructions-for-running-collectives-benchmark-on-tpu-trillium-v6e-256)).
The launch command sets two notable tuning knobs directly in the workload's shell command rather than
as XLA flags on the model config: an enlarged `/proc/sys/net/ipv4/tcp_rmem` (for the DCN cross-slice
collectives) and `LIBTPU_INIT_ARGS` with
`--megascale_grpc_premap_memory_bytes=17179869184 --xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction=true`
([run-collectives-on-v6e-256](src:microbenchmarks/trillium/collectives/README.md#run-collectives-on-v6e-256)).
Multi-slice runs are just `--num_slices=N` plus the matching `configs/Nx_v6e_256.yaml`; custom
configs can be uploaded to GCS and referenced directly in the launch command
([run-with-a-custom-yaml-config](src:microbenchmarks/trillium/collectives/README.md#run-with-a-custom-yaml-config)).

## Feeds
- [TPU microbenchmarks](../topics/tpu-microbenchmarks.md) — the Trillium (DCN-focused) collectives
  counterpart to the Ironwood (ICI-focused) collectives microbenchmarks in that topic.
