---
title: "Source: microbenchmarks/ironwood/README.md"
type: source-page
doc: microbenchmarks/ironwood/README.md
updated: 2026-07-03
---
# microbenchmarks/ironwood/README.md

The Ironwood (TPU7x) multi-chip microbenchmark hub: a `2x2x1` single-host and a `4x4x4` multi-host
GKE nodepool (68 chips total), each running the `accelerator-microbenchmarks` external repo's
compute and collectives scripts via Kubernetes Job/Pod manifests
([setup](src:microbenchmarks/ironwood/README.md#setup),
[deploying-a-single-host-job](src:microbenchmarks/ironwood/README.md#deploying-a-single-host-job),
[deploying-a-multi-host-job](src:microbenchmarks/ironwood/README.md#deploying-a-multi-host-job)).
The compute microbenchmarks
([compute](src:microbenchmarks/ironwood/README.md#compute)) cover a full FP8 GEMM pipeline family —
`gemm_multiple_run` (configurable fp8/bf16 GEMM, fp32 accumulation), `gemm_simple` (pure fp8 matmul),
`gemm` (fp8 matmul + dequantization scaling), `gemm_accum` (fp8 matmul + fp32 accumulation),
`gemm_fp8_rowwise` (dynamic row-wise/channel-wise absmax quantization before matmul), plus `add` and
`quantization`/`transpose_quantization` primitives
([gemm](src:microbenchmarks/ironwood/README.md#gemm)). The collectives microbenchmarks
([collectives](src:microbenchmarks/ironwood/README.md#collectives)) cover `psum` (all-reduce),
`psum_scatter` (reduce-scatter), `all_gather`, and `all_to_all`. Measured GEMM throughput scales from
2.6 TFLOP/s/device at matrix size 128 to 956 TFLOP/s/device at 32768
([results](src:microbenchmarks/ironwood/README.md#results)).

## Feeds
- [TPU microbenchmarks](../topics/tpu-microbenchmarks.md) — the GEMM/collectives throughput numbers
  this topic's Ironwood section is built from.
