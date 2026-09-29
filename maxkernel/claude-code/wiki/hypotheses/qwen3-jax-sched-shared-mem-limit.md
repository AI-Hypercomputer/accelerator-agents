---
title: "Qwen3 jax — scheduler shared-memory limit < 100 to bound unrolled-backward liveness"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: retired
expected_gain: "fallback: makes offload_attn + bs3 compile on the UNROLLED loop (no scan) by serializing host round-trips"
confidence: low
effort: S
origin: 2026-06-12-v024-offload-bs3
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — scheduler shared-memory limit below 100

*Retired 2026-06-12: mooted — scan + offload_attn + the completed 25-flag
stack ([v026](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v026-mt-flags-bs3.md))
fits bs3 and holds the frontier; the unrolled path no longer needs rescuing.*

*Hypothesis*: Lowering `--xla_tpu_scheduler_percent_shared_memory_limit`
from 100 to ~90 (and/or `--xla_tpu_host_transfer_overlap_limit` from 24 down
to 4–8) forces the latency-hiding scheduler to keep fewer rematted-backward
temps concurrently live, letting offload_attn + bs3 @ seq 8192 compile on
the **unrolled** loop (v024 OOM'd at 59.41/31.25 G with dozens of 384 MB
f32 backward temps live at once).

*Rationale*: [v024](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v024-offload-bs3.md)
showed the OOM is scheduling-induced liveness, not offload failure. Scan
(v025) is the structural fix; this flag route would preserve the unrolled
loop's scheduling freedom if scan's uniform-remat tax (~5.6%, v021) eats the
bs3 win.

## See also

- [qwen3-jax-host-offload-remat](qwen3-jax-host-offload-remat.md)
